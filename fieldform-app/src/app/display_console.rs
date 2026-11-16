// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Display quality and frame cost, as the panel and the console set and report them.

use crate::*;

/// The cells across the scene a control cage may be baked at. Below about 8 nothing is
/// recognisable; above 256 the bake stops being something to wait for after every edit.
const CAGE_DETAIL: std::ops::RangeInclusive<usize> = 8..=256;

impl App {
    /// Sets the display quality and says what that means. `chosen` records that a person
    /// picked it, which the startup correction then leaves alone, and keeps it for the next
    /// session.
    pub(crate) fn set_display_quality(&mut self, quality: DisplayQuality, chosen: bool) {
        self.view.quality_chosen |= chosen;
        // **A person's choice is kept; a script's is not.** A run is the application driven with
        // no hand on it, and a benchmark that asks for `Fast` to time it would otherwise leave
        // the next session on `Fast` with nothing to say why. Kept even when the rung is the one
        // already showing, because picking the guess is what makes it a choice. Said rather than
        // warned if it cannot be written: the rung has still been set, for this session.
        if chosen
            && !self.scripting.running
            && !cfg!(test)
            && let Err(why) = display_quality::save_quality(quality)
        {
            self.say(format!(
                "Display quality is {}, but could not be kept for the next session: {why}",
                quality.label()
            ));
        }
        if self.view.quality == quality {
            return;
        }
        self.view.quality = quality;
        self.say(format!(
            "Display quality: {} — {}.",
            quality.label(),
            quality.hint()
        ));
    }

    /// `display quality=<fast|balanced|full|enchilada>`, or `display` alone to hear which.
    pub(crate) fn display_from_console(&mut self, args: &Args) {
        // `display mode=field`: the View menu's Display mode, which a click could set and a line
        // could only cycle. Beside `quality` because the menu puts the two side by side.
        if let Some(wanted) = args.get("mode").and_then(|v| v.as_str().map(str::to_string)) {
            let mode = match wanted.to_ascii_lowercase().as_str() {
                "polygon" | "cage" => ViewMode::Polygon,
                "field" => ViewMode::Field,
                "overlay" | "both" => ViewMode::Overlay,
                _ => {
                    self.warn(format!(
                        "No display mode called {wanted} — polygon, field or overlay"
                    ));
                    return;
                }
            };
            self.view.mode = mode;
            // Which imported solids are drawn as triangles depends on the mode.
            self.refresh_drawn_meshes();
            self.say(format!("Display mode: {}", mode.menu_label()));
            if args.get("quality").is_none() {
                return;
            }
        }
        // `display cage=fine`, or a number of cells: View › Cage detail. A word for each preset the
        // menu offers and a number for anything between, since a line can say what a menu cannot.
        if let Some(wanted) = args.get("cage").map(|v| v.text()) {
            let cells = match wanted.to_ascii_lowercase().as_str() {
                "coarse" => Some(32),
                "normal" => Some(64),
                "fine" => Some(128),
                "finest" => Some(256),
                number => number.parse::<usize>().ok().filter(|n| CAGE_DETAIL.contains(n)),
            };
            let Some(cells) = cells else {
                self.warn(format!(
                    "No cage detail called {wanted} — coarse, normal, fine, finest, or a number of cells from {} to {}",
                    CAGE_DETAIL.start(),
                    CAGE_DETAIL.end()
                ));
                return;
            };
            // Rebuilt here because nothing else would: the number alone changes no document, and
            // a setting that waits for the next unrelated edit looks like one that does nothing.
            if cells != self.mesh_resolution {
                self.mesh_resolution = cells;
                self.rebuild_render_resources();
            }
            self.say(format!("Cage detail: {cells} cells across the scene"));
            if args.get("quality").is_none() {
                return;
            }
        }
        let Some(wanted) = args.get("quality").and_then(|v| v.as_str().map(str::to_string)) else {
            self.say(format!(
                "Display quality: {} ({}). Say `display quality=<{}>`. Cage detail: {} cells.",
                self.view.quality.label(),
                if self.view.quality_chosen { "chosen" } else { "guessed from the adapter" },
                DisplayQuality::ALL.map(|q| q.label().to_lowercase()).join("|"),
                self.mesh_resolution,
            ));
            return;
        };
        match DisplayQuality::from_name(&wanted) {
            Some(quality) => self.set_display_quality(quality, true),
            None => self.warn(format!(
                "No display quality called {wanted}. Qualities: {}",
                DisplayQuality::ALL.map(|q| q.label()).join(", ")
            )),
        }
    }

    /// Captures steps per ray for the view on screen, at each size asked for, and writes the
    /// tables, the static counts and the generated source beside the perf log.
    #[cfg(feature = "march-stats")]
    pub(crate) fn capture_march_stats(&mut self, sizes: &[u32]) {
        let stem = self
            .file
            .as_ref()
            .and_then(|(path, _)| path.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled".to_string());
        let dir = std::env::temp_dir().join("fieldform-march-stats");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.warn(format!("Could not make {}: {e}", dir.display()));
            return;
        }
        for &px in sizes {
            match self.traced.capture_march_stats(px) {
                Ok((capture, source, values)) => {
                    let converged = capture.class(1).len();
                    let capped = capture.class(3).len();
                    let total = capture.steps.len().max(1);
                    let body = format!(
                        "# {stem}, {px} px\n\n{}\n{}\n",
                        self.adapter_line,
                        capture.report()
                    );
                    let _ = std::fs::write(dir.join(format!("{stem}-{px}px.md")), body);
                    let _ = std::fs::write(dir.join(format!("{stem}.wgsl")), &source);
                    let _ = std::fs::write(
                        dir.join(format!("{stem}-static.md")),
                        format!(
                            "# {stem}, static\n\n{}\n## Copies visited per reading\n\n{}",
                            fieldform_gpu::stats::function_report(&source),
                            fieldform_gpu::stats::pattern_widths(&source, &values)
                        ),
                    );
                    self.say(format!(
                        "March stats at {px} px: {} rays, {converged} converged, {capped} out of \
                         steps ({:.2}%). Written to {}",
                        capture.steps.len(),
                        100.0 * capped as f64 / total as f64,
                        dir.display()
                    ));
                }
                Err(why) => self.warn(format!("March stats at {px} px: {why}")),
            }
        }
    }

    /// `perf` toggles per-frame timing lines; `perf on=true` and `perf on=false` set it.
    ///
    /// The loop spec's S6/S8: the application measures itself, so the machine without cargo
    /// can fill the ledger. The file opens with the adapter line, and each traced frame adds
    /// `draft 720px 43.1 ms (march 38.2, shade 0.6, cpu 2.1)` — the GPU's own time where the
    /// card writes timestamps, `(wall …)` where it does not, so an excerpt is self-identifying.
    pub(crate) fn perf_from_console(&mut self, args: &Args) {
        let on = match args.get("on") {
            Some(Value::Bool(on)) => *on,
            Some(other) => {
                other.as_f64().is_some_and(|v| v != 0.0)
                    || other.as_str().is_some_and(|s| matches!(s, "on" | "yes"))
            }
            None => self.perf.file.is_none(),
        };
        if !on {
            if self.perf.file.take().is_some() {
                self.say("Perf lines off.");
            } else {
                self.say("Perf lines were not on. `perf` starts them.");
            }
            return;
        }
        if self.perf.file.is_some() {
            self.say("Perf lines are already on.");
            return;
        }
        let path = std::env::temp_dir().join("fieldform-perf.log");
        let clock = match self.traced.timestamps_available() {
            Some(true) => "GPU timestamps",
            Some(false) => "the wall clock — this device writes no GPU timestamps",
            None => "GPU timestamps if the device writes them, the wall clock otherwise",
        };
        let head = format!("{}\nTimed by {clock}.\n", self.perf.adapter_line);
        match std::fs::write(&path, head) {
            Ok(()) => {
                self.perf.file = Some(path.clone());
                self.say(format!(
                    "Perf lines on: one per traced frame into {}, timed by {clock}. `perf` \
                     again turns them off.",
                    path.display()
                ));
            }
            Err(e) => self.warn(format!("Could not open {}: {e}", path.display())),
        }
    }

    /// Files this frame's CPU time and writes the perf line for every submission the GPU has
    /// since finished. Every frame, not only while logging: the draft tier sizes itself from
    /// the last draft's cost, and that number exists only if the costs are collected.
    pub(crate) fn note_frame_cost(&mut self, cpu_ms: f32) {
        use std::io::Write;
        // **A frame nobody could mistake for interactive says what it spent itself on.** Said in
        // the console rather than only in the perf log, because it is the answer to a question
        // the person is asking at that moment — "why did that hang?" — and because a perf log
        // has to be switched on before the hang rather than after it.
        if cpu_ms >= SLOW_FRAME_MS && !self.perf.stages.is_empty() {
            let mut worst = std::mem::take(&mut self.perf.stages);
            worst.sort_by(|a, b| b.1.total_cmp(&a.1));
            worst.truncate(3);
            let named: Vec<String> =
                worst.iter().map(|(what, ms)| format!("{what} {ms:.0} ms")).collect();
            self.say(format!(
                "That frame took {cpu_ms:.0} ms: {}",
                named.join(", ")
            ));
        }
        self.perf.stages.clear();
        self.perf.frame_cpu_ms.push_back((self.traced.frame(), cpu_ms));
        while self.perf.frame_cpu_ms.len() > 32 {
            self.perf.frame_cpu_ms.pop_front();
        }
        let reports = self.traced.take_reports();
        // **The startup guess, corrected once.** A discrete card was offered `Full` on the
        // strength of being discrete; the first draft it measures is the first fact about how
        // fast it actually is, and a draft over the loop spec's 50 ms budget is a card that
        // should start in the middle. Once only, and never over a choice somebody made: a
        // heavy document later is a heavy document, not a slow machine. A rung kept from an
        // earlier session is such a choice (`starting_quality`), and the correction is set here
        // rather than through `set_display_quality`, so it is never kept: the next session
        // measures its own first draft.
        if !self.view.quality_chosen
            && let Some(first) = reports.iter().find(|r| !r.still && !r.warm)
        {
            self.view.quality_chosen = true;
            if first.cost.ms() > DRAFT_BUDGET_MS && self.view.quality == DisplayQuality::Full {
                self.view.quality = DisplayQuality::Balanced;
                self.say(format!(
                    "The first draft frame took {:.0} ms, over the {DRAFT_BUDGET_MS:.0} ms \
                     budget, so Display quality starts on Balanced. View > Display quality \
                     changes it.",
                    first.cost.ms()
                ));
            }
        }
        let rebuilds = self.traced.take_rebuilds();
        let Some(path) = &self.perf.file else { return };
        if reports.is_empty() && rebuilds.is_empty() {
            return;
        }
        let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) else {
            return;
        };
        let elapsed = self.started.elapsed().as_secs_f64();
        for rebuild in rebuilds {
            let _ = writeln!(file, "[{elapsed:7.2}s] {}", rebuild.line());
        }
        for report in reports {
            let cpu = self
                .perf.frame_cpu_ms
                .iter()
                .find(|(frame, _)| *frame == report.frame)
                .map(|(_, ms)| *ms);
            let _ = writeln!(file, "[{elapsed:7.2}s] {}", report.line(cpu));
        }
    }
}
