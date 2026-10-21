// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The shelf and the first-run hint (`interaction-spec.md` §10.2, §10.3).
//!
//! **The shelf is a person's own strip of commands, empty until they build it.** There is no
//! default toolbar: a command goes on the shelf by being dragged there from a menu row or a
//! palette row, comes off by its right-click menu, and the shelf keeps registry ids and nothing
//! else, in the settings file. A command has no icon art yet, so it shows its name.

use crate::*;

/// What a menu row or a palette row carries while it is dragged toward the shelf.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShelfItem(pub(crate) &'static str);

impl App {
    /// Reads the shelf and the palette count from the settings file. Not in a test, whose shelf
    /// starts empty whatever the owner has built.
    pub(crate) fn load_shelf(&mut self) {
        if cfg!(test) {
            return;
        }
        let text = settings::read();
        self.shelf = settings::shelf_of(text.as_deref());
        self.palette_opens = settings::palette_opens_of(text.as_deref());
    }

    fn save_shelf(&mut self) {
        if cfg!(test) {
            return;
        }
        if let Err(why) = settings::write(settings::SHELF, serde_json::json!(self.shelf)) {
            self.warn(format!("The shelf could not be kept: {why}"));
        }
    }

    /// Puts a command on the end of the shelf, if it is not there already.
    pub(crate) fn shelf_add(&mut self, id: &'static str) {
        if self.shelf.contains(&id) || commands::by_id(id).is_none() {
            return;
        }
        self.shelf.push(id);
        self.save_shelf();
    }

    /// Takes a command off the shelf.
    pub(crate) fn shelf_remove(&mut self, id: &'static str) {
        self.shelf.retain(|s| *s != id);
        self.save_shelf();
    }

    /// The shelf, under the menu bar: shown when it holds something, and as a place to drop onto
    /// while a command is being dragged.
    pub(crate) fn shelf_strip(&mut self, ui: &mut egui::Ui) {
        let dragging = egui::DragAndDrop::has_payload_of_type::<ShelfItem>(ui.ctx());
        self.shelf_drawn.clear();
        if self.shelf.is_empty() && !dragging {
            return;
        }
        let ctx = self.command_context();
        let shelf = self.shelf.clone();
        let mut run: Option<&'static commands::Command> = None;
        let mut remove: Option<&'static str> = None;
        let mut drawn = Vec::new();
        let (_, dropped) = ui.dnd_drop_zone::<ShelfItem, _>(egui::Frame::NONE.inner_margin(2.0), |ui| {
            ui.horizontal_wrapped(|ui| {
                if shelf.is_empty() {
                    ui.label(egui::RichText::new("Drop a command here to keep it on the shelf").weak());
                }
                for id in &shelf {
                    let Some(c) = commands::by_id(id) else { continue };
                    let available = (c.context)(&ctx);
                    let response = ui.add_enabled(available.is_ok(), egui::Button::new(c.display_name.trim_end_matches("...")));
                    let tip = match c.binding_label() {
                        Some(key) => format!("{} · {key}", c.console_line()),
                        None => c.console_line(),
                    };
                    let response = match available {
                        Ok(()) => response.on_hover_text(tip),
                        Err(why) => response.on_disabled_hover_text(why),
                    };
                    drawn.push((c.id, response.rect));
                    if response.clicked() {
                        run = Some(c);
                    }
                    response.context_menu(|ui| {
                        if ui.button("Remove from the shelf").clicked() {
                            remove = Some(c.id);
                            ui.close();
                        }
                    });
                }
            });
        });
        self.shelf_drawn = drawn;
        if let Some(item) = dropped {
            self.shelf_add(item.0);
        }
        if let Some(id) = remove {
            self.shelf_remove(id);
        }
        if let Some(c) = run {
            self.invoke_command(c);
        }
    }

    /// Whether the first-run hint shows: the document is empty, and the palette has been opened
    /// fewer than three times ever (§10.3).
    pub(crate) fn first_run_hint_due(&self) -> bool {
        self.palette_opens < settings::HINT_UNTIL && self.part_graph().nodes().count() == 0 && self.doc.curves.is_empty()
    }

    /// The hint itself, faint, centred in the viewport -- below the splash while the splash shows.
    pub(crate) fn draw_first_run_hint(&self, ui: &egui::Ui, rect: egui::Rect) {
        let y = if self.splash_shown { rect.bottom() - 40.0 } else { rect.center().y };
        ui.painter().text(
            egui::pos2(rect.center().x, y),
            egui::Align2::CENTER_CENTER,
            "Space — search commands · Right-click — actions",
            egui::FontId::proportional(15.0),
            ui.visuals().weak_text_color(),
        );
    }

    /// Counts a palette opening toward the hint's going, and keeps the count -- until it has gone.
    pub(crate) fn count_palette_opening(&mut self) {
        if self.palette_opens >= settings::HINT_UNTIL {
            return;
        }
        self.palette_opens += 1;
        if !cfg!(test)
            && let Err(why) = settings::write(settings::PALETTE_OPENS, serde_json::json!(self.palette_opens))
        {
            self.warn(format!("The palette count could not be kept: {why}"));
        }
    }
}
