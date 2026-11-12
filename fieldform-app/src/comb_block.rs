// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The curvature comb in the application: its verb, its row, its panel and its drawing.
//!
//! The block is a document thing (`fieldform_scene::CurvatureComb`); what it draws is derived from
//! its curve every time the curve or the comb changes and never saved, for the reason a mesh is not:
//! a stored reading would be a second description of the curve that could disagree with it.

use crate::outputs::row_title;
use crate::*;
use fieldform_ops::Command;
use fieldform_scene::{CombId, CombReading, CurvatureComb};
use std::time::Instant;

/// Everything about drawing and editing combs, none of it saved.
#[derive(Default)]
pub(crate) struct CombRuns {
    /// The comb being edited and when, with the combs as they were when the edit began -- the
    /// rule `set_images` follows, so a scrub on Scale is one entry on the history.
    pub(crate) gesture: Option<(CombId, Instant, Vec<CurvatureComb>)>,
    /// What each comb reads, against the curve revision and the combs it was read for. A comb
    /// is recomputed when either moves and at no other time.
    drawn: Option<(u64, Vec<CurvatureComb>, Vec<(CombId, Vec<fieldform_scene::CurveId>, CombReading)>)>,
}

impl App {
    /// The combs of the part being edited, in the order they were made.
    pub(crate) fn comb_blocks(&self) -> Vec<CombId> {
        self.doc.combs.iter().filter(|c| c.part == self.active_part).map(|c| c.id).collect()
    }

    fn comb(&self, id: CombId) -> Option<&CurvatureComb> {
        self.doc.combs.iter().find(|c| c.id == id)
    }

    /// The comb a console line names by its title, case aside.
    fn comb_named(&self, name: &str) -> Option<CombId> {
        let wanted = name.trim();
        self.doc.combs.iter().find(|c| c.title().eq_ignore_ascii_case(wanted)).map(|c| c.id)
    }

    /// Sets the whole list of combs as one undo step, or as part of the one a gesture on
    /// `gesture`'s panel already began.
    pub(crate) fn set_combs(&mut self, combs: Vec<CurvatureComb>, gesture: Option<CombId>) {
        let now = self.doc.combs.clone();
        if now == combs {
            return;
        }
        let continues = gesture.is_some()
            && self.stack.peek_undo() == Some("combs.set")
            && self
                .comb_runs
                .gesture
                .as_ref()
                .is_some_and(|(c, at, _)| Some(*c) == gesture && at.elapsed() < GESTURE_WINDOW);
        let before = match (&self.comb_runs.gesture, continues) {
            (Some((_, _, before)), true) => before.clone(),
            _ => now,
        };
        self.comb_runs.gesture = gesture.map(|c| (c, Instant::now(), before.clone()));
        self.gesture = None;
        let cmd: Box<dyn Command> = Box::new(fieldform_ops::SetCombs::new(before, combs));
        let done = if continues {
            self.stack.replace_top(&mut self.doc, cmd)
        } else {
            self.stack.execute(&mut self.doc, cmd)
        };
        match done {
            Ok(()) => self.mark_changed(),
            Err(e) => self.warn(e.to_string()),
        }
    }

    fn edit_comb(&mut self, id: CombId, gesture: bool, change: impl FnOnce(&mut CurvatureComb)) {
        let mut combs = self.doc.combs.clone();
        let Some(comb) = combs.iter_mut().find(|c| c.id == id) else { return };
        change(comb);
        self.set_combs(combs, gesture.then_some(id));
    }

    /// `comb [curve= | profile=] [scale=] [teeth=] [-flip] [-hidden] ['name']` makes a comb on the
    /// named or selected curve -- or on a whole profile, closed or open -- fitted unless `scale=`
    /// says; `comb block=<comb> …` changes one that exists, and `-fit` re-fits its scale.
    /// `comb -fit` alone re-fits the selected comb.
    pub(crate) fn comb_command(&mut self, args: &Args) {
        let verb = "comb";
        let named_block = args.get("block").map(|v| v.text());
        let selected = match self.selected_block {
            Some(outputs::BlockRef::Comb(id)) => Some(id),
            _ => None,
        };
        let names_input = args.get("curve").is_some() || args.get("profile").is_some();
        let existing = match &named_block {
            Some(name) => match self.comb_named(name) {
                Some(id) => Some(id),
                None => {
                    self.warn(format!("No curvature comb is called {}", quoted(name)));
                    return;
                }
            },
            // `comb -fit` on its own is about the comb in hand, not a new one.
            None if console_keys::flag(args, "fit") && !names_input => match selected {
                Some(id) => Some(id),
                None => {
                    self.warn("`comb -fit` re-fits the selected comb: select one, or name it with `block=`");
                    return;
                }
            },
            None => None,
        };
        // What it reads: `(curve, whole profile)`.
        let input = match (args.get("curve").map(|v| v.text()), args.get("profile").map(|v| v.text())) {
            (Some(_), Some(_)) => {
                self.warn("`comb` reads a curve or a profile — curve= or profile=, not both");
                return;
            }
            (Some(name), None) => match self.curve_named(&name) {
                Some(id) => Some((id, false)),
                None => {
                    self.warn(format!("No curve is called {}", quoted(&name)));
                    return;
                }
            },
            (None, Some(name)) => {
                let found = self.curve_or_profile_named(&name);
                self.refresh_profiles();
                match found.and_then(|id| self.profiles.key_of(id)) {
                    Some(key) => Some((key, true)),
                    None => {
                        self.warn(format!(
                            "{} is not a profile: a closed loop, or open curves held end to end by a continuity at each join",
                            quoted(&name)
                        ));
                        return;
                    }
                }
            }
            (None, None) => None,
        };
        let scale = match args.get("scale") {
            Some(v) => match v.as_f64() {
                Some(s) if s >= 0.0 && s.is_finite() => Some(s),
                _ => {
                    self.warn(format!("`{verb} scale=` is a length times a length, nought or more: {}", v.text()));
                    return;
                }
            },
            None => None,
        };
        let teeth = match args.get("teeth") {
            Some(v) => match console_keys::whole(verb, "teeth", v) {
                Ok(n) if (2..=CurvatureComb::MAX_TEETH).contains(&n) => Some(n),
                Ok(n) => {
                    self.warn(format!("`{verb} teeth=` is 2 to {}: {n}", CurvatureComb::MAX_TEETH));
                    return;
                }
                Err(why) => {
                    self.warn(why);
                    return;
                }
            },
            None => None,
        };
        let flip = args.get("flip").map(|_| console_keys::flag(args, "flip"));
        let hidden = args.get("hidden").map(|_| console_keys::flag(args, "hidden"));
        let fit = console_keys::flag(args, "fit");
        let name = args.get("name").map(|v| v.text().trim().to_string()).filter(|n| !n.is_empty());

        if let Some(id) = existing {
            let Some(mut comb) = self.comb(id).cloned() else { return };
            if let Some((curve, whole)) = input {
                comb.curve = curve;
                comb.whole_loop = whole;
            }
            if let Some(n) = teeth {
                comb.teeth = n;
            }
            if let Some(s) = scale {
                comb.scale = s;
            }
            if fit {
                match self.fit_for(&comb) {
                    Some(s) => comb.scale = s,
                    None => {
                        self.warn(format!("{} reads a curve that is gone, so there is nothing to fit to", comb.title()));
                        return;
                    }
                }
            }
            if let Some(f) = flip {
                comb.flip = f;
            }
            if let Some(h) = hidden {
                comb.hidden = h;
            }
            if name.is_some() {
                comb.name = name;
            }
            let combs = self.doc.combs.iter().map(|c| if c.id == id { comb.clone() } else { c.clone() }).collect();
            self.set_combs(combs, None);
            self.say(if fit {
                format!("{} fitted: scale {}", comb.title(), shown_number(comb.scale))
            } else {
                format!("{} changed", comb.title())
            });
            return;
        }

        // A profile row selected is a request for the whole profile, as an extrude takes it.
        let in_hand = match self.selected_block {
            Some(outputs::BlockRef::Profile(key)) => Some((key, true)),
            _ => self.curve_in_hand().map(|id| (id, false)),
        };
        let Some((curve_id, whole)) = input.or(in_hand) else {
            self.warn("A curvature comb reads a curve: select one or a profile, or name it with `comb curve=…` or `profile=…`");
            return;
        };
        let Some(curve) = self.doc.curve(curve_id).cloned() else { return };
        let mut comb = CurvatureComb::new(self.active_part, &curve);
        comb.whole_loop = whole;
        if let Some(n) = teeth {
            comb.teeth = n;
        }
        if let Some(s) = self.fit_for(&comb) {
            comb.scale = s;
        }
        if let Some(s) = scale {
            comb.scale = s;
        }
        comb.flip = flip.unwrap_or(false);
        comb.hidden = hidden.unwrap_or(false);
        comb.name = name;
        let id = comb.id;
        let title = comb.title().to_string();
        let reads = self.comb_reads(&comb);
        let mut combs = self.doc.combs.clone();
        combs.push(comb);
        self.set_combs(combs, None);
        self.select_block(outputs::BlockRef::Comb(id));
        self.say(format!("Added {title} on {}", quoted(&reads)));
    }

    /// Re-fits a comb's scale to its curve now, as one undoable edit, and says the line.
    pub(crate) fn fit_comb(&mut self, id: CombId) {
        let Some(comb) = self.comb(id) else { return };
        let title = comb.title().to_string();
        let line = format!("comb block={} -fit", quoted(&title));
        self.echo_line(line.clone());
        self.run_command_unechoed(&line);
    }

    /// What a comb reads now: its curves in order along the profile, whether they close, and
    /// whether the profile it was made on has come apart -- in which case it reads its curve
    /// alone until the profile is whole again. Nothing about the profile is stored.
    pub(crate) fn comb_path(&mut self, comb: &CurvatureComb) -> (Vec<fieldform_scene::CurveId>, bool, bool) {
        if !comb.whole_loop {
            return (vec![comb.curve], false, false);
        }
        self.refresh_profiles();
        let row = self
            .profiles
            .key_of(comb.curve)
            .and_then(|key| self.profiles.members(key).map(|m| (key, m.to_vec())));
        match row {
            Some((key, members)) => (members, !self.profiles.is_open(key), false),
            None => (vec![comb.curve], false, true),
        }
    }

    /// The comb read off its curves now, with the curves it read, or `None` with one gone.
    fn read_comb_now(&mut self, comb: &CurvatureComb) -> Option<(CombReading, Vec<fieldform_scene::CurveId>, bool)> {
        let (members, closed, opened) = self.comb_path(comb);
        let pieces: Vec<&fieldform_scene::Curve> = members.iter().map(|id| self.doc.curve(*id)).collect::<Option<_>>()?;
        let reading =
            fieldform_scene::read_comb_path(&pieces, closed, comb.scale.max(0.0), comb.teeth_drawn(), comb.flip);
        Some((reading, members, opened))
    }

    /// The scale a fit gives the comb over everything it reads, or `None` with a curve gone.
    fn fit_for(&mut self, comb: &CurvatureComb) -> Option<f64> {
        let (members, _, _) = self.comb_path(comb);
        let pieces: Vec<&fieldform_scene::Curve> = members.iter().map(|id| self.doc.curve(*id)).collect::<Option<_>>()?;
        Some(fieldform_scene::fitted_scale_path(&pieces, comb.teeth_drawn()))
    }

    /// What its row says it reads: the profile's label, or the curve's name.
    fn comb_reads(&mut self, comb: &CurvatureComb) -> String {
        if comb.whole_loop {
            self.refresh_profiles();
            if let Some(key) = self.profiles.key_of(comb.curve) {
                return self.profile_label(key);
            }
        }
        self.doc.curve(comb.curve).map(|k| k.name.clone()).unwrap_or_else(|| "missing".to_string())
    }

    /// Every comb the viewport should draw now, read from its curves: the active part's, shown,
    /// on curves that are there and not all hidden.
    pub(crate) fn comb_readings(&mut self) -> Vec<(CombId, CombReading)> {
        let fresh = self
            .comb_runs
            .drawn
            .as_ref()
            .is_some_and(|(revision, combs, _)| *revision == self.curve_revision && *combs == self.doc.combs);
        if !fresh {
            let combs = self.doc.combs.clone();
            let mut readings = Vec::new();
            for comb in &combs {
                if let Some((reading, members, _)) = self.read_comb_now(comb) {
                    readings.push((comb.id, members, reading));
                }
            }
            // The revision as it stands after the profiles were refreshed for the reading.
            self.comb_runs.drawn = Some((self.curve_revision, combs, readings));
        }
        let Some((_, _, readings)) = &self.comb_runs.drawn else { return Vec::new() };
        readings
            .iter()
            .filter(|(id, members, _)| {
                self.comb(*id).is_some_and(|c| {
                    c.part == self.active_part
                        && !c.hidden
                        && !members.iter().all(|m| self.hidden_curves.contains(m))
                })
            })
            .map(|(id, _, reading)| (*id, reading.clone()))
            .collect()
    }

    /// The comb's panel: what it reads, its scale and teeth, which side, and what it measured.
    pub(crate) fn comb_block(&mut self, ui: &mut egui::Ui, id: CombId) {
        let Some(before) = self.comb(id).cloned() else { return };
        let mut c = before.clone();
        let selected = self.selected_block == Some(outputs::BlockRef::Comb(id));
        let reads = self.comb_reads(&before);
        let now = self.read_comb_now(&before);
        let members = now.as_ref().map(|(_, m, _)| m.clone()).unwrap_or_default();
        let curve_names: Vec<String> = members
            .iter()
            .map(|m| self.doc.curve(*m).map(|k| k.name.clone()).unwrap_or_default())
            .collect();
        // The dropdown's choices: every profile, then every curve.
        self.refresh_profiles();
        let mut choices: Vec<(String, fieldform_scene::CurveId, bool)> =
            self.profiles.keys().into_iter().map(|key| (self.profile_label(key), key, true)).collect();
        let mut curves: Vec<(String, fieldform_scene::CurveId, bool)> =
            self.doc.curves.values().map(|k| (k.name.clone(), k.id, false)).collect();
        curves.sort_by(|a, b| a.0.cmp(&b.0));
        choices.extend(curves);
        let mut fit = false;
        let mut eye_clicked = false;

        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("comb", id.raw())),
            true,
        );
        self.open_for_reveal(&mut state, ui.ctx(), RowKey::Comb(id));
        state
            .show_header(ui, |ui| {
                // Hidden curves hide the comb too, and the eye says so rather than offering to
                // show what cannot be shown.
                let curve_hidden = !members.is_empty() && members.iter().all(|m| self.hidden_curves.contains(m));
                let showing = !before.hidden && !curve_hidden;
                let icon = egui::RichText::new("\u{1f441}").color(if showing {
                    ui.visuals().strong_text_color()
                } else {
                    ui.visuals().weak_text_color()
                });
                let eye = ui.add_sized(
                    egui::vec2(18.0, ui.spacing().interact_size.y),
                    egui::Button::new(icon).frame(false),
                );
                if curve_hidden {
                    eye.on_hover_text("What it reads is hidden, and a comb is drawn only on curves that are shown.");
                } else if eye.on_hover_text(if before.hidden { "Show the comb" } else { "Hide the comb" }).clicked() {
                    eye_clicked = true;
                }
                let title = format!("{}  \u{2014}  {reads}", row_title(before.name.as_deref(), CurvatureComb::DEFAULT_NAME));
                let Some(row) = self.list_row(ui, RowKey::Comb(id), &title, selected) else {
                    return;
                };
                if row.clicked() || row.secondary_clicked() {
                    self.select_block(outputs::BlockRef::Comb(id));
                }
                row.context_menu(|ui| self.block_menu(ui, outputs::BlockRef::Comb(id)));
            })
            .body(|ui| {
                ui.horizontal(|ui| {
                    row_label(ui, "Reads:");
                    egui::ComboBox::from_id_salt(("comb curve", id.raw())).selected_text(reads.clone()).show_ui(ui, |ui| {
                        for (name, curve, whole) in &choices {
                            let on = c.curve == *curve && c.whole_loop == *whole;
                            if ui.selectable_label(on, name).clicked() {
                                c.curve = *curve;
                                c.whole_loop = *whole;
                            }
                        }
                    });
                })
                .response
                .on_hover_text("A curve, or a whole profile -- a closed loop, or open curves held at their joins -- read as one. Choosing another keeps the scale, so two can be compared at one.");
                ui.horizontal(|ui| {
                    row_label(ui, "Scale:");
                    // No range on the widget: a range rewrites the value every frame and reports
                    // it as an edit. Held to nought or more after a change instead.
                    let speed = (before.scale.abs() * 0.01).max(1e-3);
                    if ui.add(egui::DragValue::new(&mut c.scale).speed(speed).suffix(" mm²")).changed() {
                        c.scale = c.scale.max(0.0);
                    }
                    fit = ui
                        .small_button("Fit")
                        .on_hover_text("Sets the scale so the longest tooth is 15 % of the size of what it reads. Done once when the comb is made, and never again on its own: a comb that re-fitted itself would hide the change it is there to show.")
                        .clicked();
                })
                .response
                .on_hover_text("A tooth's length per unit of curvature: a tooth is the scale times the curvature there.");
                ui.horizontal(|ui| {
                    row_label(ui, "Teeth:");
                    let mut n = c.teeth as f64;
                    if ui.add(egui::DragValue::new(&mut n).speed(0.5)).changed() {
                        c.teeth = (n.round() as i64).clamp(2, CurvatureComb::MAX_TEETH as i64) as u32;
                    }
                    ui.checkbox(&mut c.flip, "Flip")
                        .on_hover_text("Teeth toward the centre of curvature instead of away from it.");
                })
                .response
                .on_hover_text("Samples along each curve, both ends included.");
                match &now {
                    None => {
                        ui.label(
                            egui::RichText::new("!! A curve it reads is gone: choose another to read.")
                                .small()
                                .color(ui.visuals().warn_fg_color),
                        );
                    }
                    Some((reading, _, opened)) => {
                        if *opened {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Its profile has come apart — showing {} only until it is whole again",
                                    curve_names.first().cloned().unwrap_or_default()
                                ))
                                .small()
                                .color(ui.visuals().warn_fg_color),
                            );
                        }
                        let line = match reading.max {
                            Some(m) if m.curvature > 0.0 => {
                                let on = self.doc.curve(m.curve).map(|k| k.name.clone()).unwrap_or_default();
                                format!(
                                    "Max curvature {} /mm (radius {} mm) on {on} at t = {:.3}",
                                    shown_number(m.curvature),
                                    shown_number(1.0 / m.curvature),
                                    m.t
                                )
                            }
                            Some(_) => "Max curvature 0: straight throughout".to_string(),
                            None => "No tooth to read".to_string(),
                        };
                        ui.label(egui::RichText::new(line).small().weak());
                        if reading.skipped > 0 {
                            ui.label(
                                egui::RichText::new(format!("{} samples skipped: zero tangent", reading.skipped))
                                    .small()
                                    .weak(),
                            )
                            .on_hover_text("Coincident control points or a cusp: the curve has no direction there, so it has no curvature to draw.");
                        }
                        // Each joint as two numbers, not a verdict: the turn, which is nought at
                        // G1, and the curvature step, which is nought at G2 as well.
                        for joint in &reading.joints {
                            let from = curve_names.get(joint.after).cloned().unwrap_or_default();
                            let to = curve_names.get((joint.after + 1) % curve_names.len().max(1)).cloned().unwrap_or_default();
                            ui.label(
                                egui::RichText::new(format!(
                                    "{from} → {to}: turn {}°, curvature step {} /mm",
                                    shown_number(joint.angle_degrees),
                                    shown_number(joint.curvature_step)
                                ))
                                .small()
                                .weak(),
                            )
                            .on_hover_text("The angle between the tangents either side of the joint, nought where it is tangent-continuous (G1), and the change in curvature across it, nought where it is curvature-continuous (G2) as well. Measured from the comb's own end samples.");
                        }
                    }
                }
            });

        if eye_clicked {
            let hide = !before.hidden;
            self.echo_line(format!("comb block={} hidden={hide}", quoted(before.title())));
            self.edit_comb(id, false, |k| k.hidden = hide);
        }
        if c != before && self.comb(id).is_some() {
            let combs = self.doc.combs.iter().map(|x| if x.id == id { c.clone() } else { x.clone() }).collect();
            self.set_combs(combs, Some(id));
        }
        if fit {
            self.fit_comb(id);
        }
    }

    /// Deletes a comb, undoably. Its curve is not touched.
    pub(crate) fn delete_comb(&mut self, id: CombId) {
        let Some(title) = self.comb(id).map(|c| c.title().to_string()) else { return };
        let mut combs = self.doc.combs.clone();
        combs.retain(|c| c.id != id);
        self.set_combs(combs, None);
        self.say(format!("Deleted {title}"));
    }

    pub(crate) fn rename_comb(&mut self, id: CombId, text: &str) {
        let name = (!text.trim().is_empty()).then(|| text.trim().to_string());
        self.edit_comb(id, false, |c| c.name = name);
    }

    /// The comb's given name, for a rename box to start from.
    pub(crate) fn comb_given_name(&self, id: CombId) -> String {
        self.comb(id).and_then(|c| c.name.clone()).unwrap_or_default()
    }

    /// The curve a comb reads, for the viewport to highlight while the comb is selected.
    pub(crate) fn comb_curve(&self, id: CombId) -> Option<fieldform_scene::CurveId> {
        self.comb(id).map(|c| c.curve)
    }
}

/// A number as the panel and the console say it: four significant figures, no trailing noise.
fn shown_number(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let digits = (3 - x.abs().log10().floor() as i32).max(0) as usize;
    let text = format!("{x:.digits$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// Draws each comb: a tooth from the curve to its tip at every sample, and the envelope through
/// the tips. Screen-space segments, as the curves are drawn, so a comb is as visible as the curve
/// it sits on.
pub(crate) fn draw_combs(
    painter: &egui::Painter,
    camera: &camera::OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    readings: &[(CombId, CombReading)],
) {
    let stroke = egui::Stroke::new(1.0_f32, theme::ANALYSIS_COMB);
    let at = |p: nalgebra::Point3<f64>| camera.project_to_screen(aspect, rect, p.cast::<f32>());
    for (_, reading) in readings {
        for tooth in &reading.teeth {
            if let (Some(a), Some(b)) = (at(tooth.at), at(tooth.tip)) {
                painter.line_segment([a, b], stroke);
            }
        }
        for run in &reading.envelope {
            let mut previous: Option<egui::Pos2> = None;
            for p in run {
                let current = at(*p);
                if let (Some(a), Some(b)) = (previous, current) {
                    painter.line_segment([a, b], stroke);
                }
                previous = current;
            }
        }
    }
}

use crate::image_export::quoted;
