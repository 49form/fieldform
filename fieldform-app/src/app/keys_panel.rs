// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The keybinding panel (`interaction-spec.md` §9.2): every command with the keys it answers to,
//! rebound by pressing the keys, and every conflict listed.

use crate::*;

impl App {
    /// The keymap the person's bindings make over the registry's defaults, and what collides.
    pub(crate) fn rebuild_keymap(&mut self) {
        let (bound, conflicts) = keybindings::effective(&self.bindings);
        let mut keymap = fieldform_ops::Keymap::new();
        for (chord, command) in &bound {
            keymap.bind(chord.shortcut(), command.op, command.fixed_args());
        }
        self.keymap = keymap;
        self.binding_conflicts = conflicts;
    }

    /// Reads the person's bindings file, if there is one, saying what it could not read.
    pub(crate) fn load_bindings(&mut self) {
        let Some(path) = keybindings::path() else { return };
        let Ok(text) = std::fs::read_to_string(&path) else { return };
        let (lines, errors) = keybindings::parse(&text);
        for why in errors {
            self.warn(format!("{}: {why}", path.display()));
        }
        self.bindings = lines;
        self.rebuild_keymap();
    }

    /// Writes the person's bindings, so a rebind outlives the session. A test's `App` keeps its
    /// hands off the file, as it does every other the owner keeps.
    fn save_bindings(&mut self) {
        self.rebuild_keymap();
        if cfg!(test) {
            return;
        }
        let Some(path) = keybindings::path() else { return };
        if let Some(folder) = path.parent()
            && let Err(why) = std::fs::create_dir_all(folder)
        {
            self.warn(format!("The bindings could not be kept: {why}"));
            return;
        }
        if let Err(why) = std::fs::write(&path, keybindings::write(&self.bindings)) {
            self.warn(format!("The bindings could not be kept in {}: {why}", path.display()));
        }
    }

    /// Starts waiting for the keys to give `id`; the next chord pressed is it, Esc gives up.
    pub(crate) fn begin_capture(&mut self, id: &'static str) {
        self.capturing = Some(id);
        self.bindings_note = None;
    }

    /// The chord pressed while capturing: given to the command, or refused by name (§9.2).
    pub(crate) fn captured(&mut self, chord: commands::Chord) {
        let Some(id) = self.capturing.take() else { return };
        if chord.key == "Escape" && !chord.ctrl && !chord.alt && !chord.shift {
            self.bindings_note = None;
            return;
        }
        if let Some(why) = keybindings::refusal(&self.bindings, id, &chord) {
            self.bindings_note = Some(why);
            return;
        }
        self.bindings = keybindings::rebound(&self.bindings, id, chord.clone());
        self.save_bindings();
        let name = commands::by_id(id).map_or(id, |c| c.display_name.trim_end_matches("..."));
        self.bindings_note = Some(format!("{name} is on {}", chord.label()));
    }

    /// Takes every key off `id`, defaults included.
    pub(crate) fn clear_binding(&mut self, id: &'static str) {
        self.bindings = keybindings::cleared(&self.bindings, id);
        self.save_bindings();
    }

    /// Gives `id` its default keys back.
    pub(crate) fn reset_binding(&mut self, id: &'static str) {
        self.bindings = keybindings::reset(&self.bindings, id);
        self.save_bindings();
    }

    /// The panel itself, in the window Help › Keys opens.
    pub(crate) fn keys_panel(&mut self, ctx: &egui::Context) {
        let mut open = self.show_shortcuts;
        if !open {
            self.capturing = None;
            return;
        }
        let (bound, _) = keybindings::effective(&self.bindings);
        let mut capture: Option<&'static str> = None;
        let mut clear: Option<&'static str> = None;
        let mut reset: Option<&'static str> = None;
        let capturing = self.capturing;
        let note = self.bindings_note.clone();
        let conflicts = self.binding_conflicts.clone();
        egui::Window::new("Keys").open(&mut open).default_width(560.0).show(ctx, |ui| {
            ui.label(egui::RichText::new("The mouse is the navigation's, and no key may take it:").weak());
            for (keys, what) in [
                ("LMB drag", "tumble about the point under the cursor"),
                ("MMB drag", "pan"),
                ("RMB drag", "dolly toward the last pivot (vertical) or twist about it (horizontal)"),
                ("LMB click", "select a vertex, edge or face — or a body, in field mode"),
                ("LMB drag on the selection", "move it"),
                ("Alt+Shift", "still navigates, if your hands prefer it"),
                ("Scroll", "zoom"),
            ] {
                ui.horizontal(|ui| {
                    ui.add_sized([170.0, ui.spacing().interact_size.y], egui::Label::new(egui::RichText::new(keys).strong()));
                    ui.label(what);
                });
            }
            ui.separator();
            if let Some(note) = &note {
                ui.label(egui::RichText::new(note).color(ui.visuals().warn_fg_color));
            }
            egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                for c in commands::COMMANDS.iter().filter(|c| c.built()) {
                    let chords: Vec<String> = bound.iter().filter(|(_, b)| b.id == c.id).map(|(ch, _)| ch.label()).collect();
                    let custom = self.bindings.iter().any(|l| l.command == Some(c.id))
                        || c.chords().iter().any(|d| self.bindings.iter().any(|l| l.chord == *d));
                    ui.horizontal(|ui| {
                        let place = if c.category.is_empty() { "console".to_string() } else { c.category.join(" › ") };
                        ui.add_sized(
                            [220.0, ui.spacing().interact_size.y],
                            egui::Label::new(c.display_name.trim_end_matches("...")).truncate(),
                        )
                        .on_hover_text(format!("{place} · {}", c.console_line()));
                        let shown = if capturing == Some(c.id) {
                            egui::RichText::new("press the keys… (Esc gives up)").italics()
                        } else if chords.is_empty() {
                            egui::RichText::new("—").weak()
                        } else {
                            egui::RichText::new(chords.join(" / ")).monospace()
                        };
                        ui.add_sized([170.0, ui.spacing().interact_size.y], egui::Label::new(shown));
                        if ui.small_button("Set…").on_hover_text("Press the keys to give it").clicked() {
                            capture = Some(c.id);
                        }
                        if !chords.is_empty() && ui.small_button("Clear").clicked() {
                            clear = Some(c.id);
                        }
                        if custom && ui.small_button("Reset").on_hover_text("Its own keys back").clicked() {
                            reset = Some(c.id);
                        }
                    });
                }
            });
            ui.separator();
            ui.label(egui::RichText::new("Conflicts").strong());
            // The navigation's one claim on a command's gesture, said where the rest are (§9.2).
            ui.label(egui::RichText::new(
                "Right-drag is the navigation's dolly, which wins over the marking menu's \
                 press–drag–release; the menu opens on a right-click, and a slot is picked by \
                 clicking it or by a left-drag from its centre.",
            ).color(ui.visuals().warn_fg_color));
            for c in &conflicts {
                ui.label(egui::RichText::new(&c.why).color(ui.visuals().warn_fg_color));
            }
            if let Some(path) = keybindings::path() {
                ui.label(egui::RichText::new(format!("Kept in {}", path.display())).weak().small());
            }
        });
        self.show_shortcuts = open;
        if let Some(id) = capture {
            self.begin_capture(id);
        }
        if let Some(id) = clear {
            self.clear_binding(id);
        }
        if let Some(id) = reset {
            self.reset_binding(id);
        }
    }
}
