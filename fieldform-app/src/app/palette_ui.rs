// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The command palette overlay (`interaction-spec.md` §5): Space, a few letters, Enter.

use crate::*;

/// The palette while it is open.
pub(crate) struct PaletteState {
    pub(crate) query: String,
    /// The row Enter runs.
    pub(crate) highlight: usize,
    /// Opened this frame: its field takes the keyboard.
    pub(crate) just_opened: bool,
    /// The commands it is narrowed to, when the marking menu's "More…" opened it (§7.2).
    pub(crate) only: Option<Vec<&'static str>>,
}

/// The palette's text field, so the keyboard can be handed to it.
pub(crate) const PALETTE_INPUT: &str = "command palette";

/// How many rows the palette shows at once.
const ROWS: usize = 12;

/// How many commands the palette remembers as run lately (§5.2).
pub(crate) const RECENT: usize = 20;

impl App {
    /// Opens the palette, empty, with the keyboard in its field.
    pub(crate) fn open_palette(&mut self, only: Option<Vec<&'static str>>) {
        // The key that opened it is not the first letter of the search.
        self.egui_ctx.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Text(_))));
        self.palette = Some(PaletteState { query: String::new(), highlight: 0, just_opened: true, only });
        self.count_palette_opening();
    }

    /// Notes a command as run, for the palette's recency (§5.2).
    pub(crate) fn note_recent(&mut self, id: &'static str) {
        self.recent.retain(|r| *r != id);
        self.recent.insert(0, id);
        self.recent.truncate(app::palette_ui::RECENT);
    }

    /// Draws the palette, if it is open, and runs what it is asked to.
    pub(crate) fn palette_ui(&mut self, ctx: &egui::Context) {
        let Some(mut state) = self.palette.take() else { return };
        let mut hits = palette::ranked(&state.query, &self.recent, &self.command_context());
        if let Some(only) = &state.only {
            hits.retain(|h| only.contains(&h.command.id));
        }
        state.highlight = state.highlight.min(hits.len().saturating_sub(1));
        let screen = ctx.content_rect();
        let width = 560.0_f32.min(screen.width() - 32.0);
        let at = egui::pos2(screen.center().x - width / 2.0, screen.top() + screen.height() * 0.18);
        let fid = egui::Id::new(PALETTE_INPUT);
        let mut run: Option<&'static commands::Command> = None;
        let mut passthrough: Option<String> = None;
        let mut close = false;
        egui::Area::new(egui::Id::new("command palette area"))
            .order(egui::Order::Foreground)
            .fixed_pos(at)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_width(width);
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut state.query)
                            .id(fid)
                            .desired_width(f32::INFINITY)
                            .hint_text("Search commands, or type a console line")
                            .font(egui::TextStyle::Monospace),
                    );
                    if state.just_opened {
                        state.just_opened = false;
                        edit.request_focus();
                    }
                    if edit.changed() {
                        state.highlight = 0;
                    }
                    let (up, down, enter, escape) = ui.input(|i| {
                        (
                            i.key_pressed(egui::Key::ArrowUp),
                            i.key_pressed(egui::Key::ArrowDown),
                            i.key_pressed(egui::Key::Enter),
                            i.key_pressed(egui::Key::Escape),
                        )
                    });
                    if down && !hits.is_empty() {
                        state.highlight = (state.highlight + 1).min(hits.len() - 1);
                    }
                    if up {
                        state.highlight = state.highlight.saturating_sub(1);
                    }
                    let console_line = palette::is_console_line(&state.query);
                    if console_line {
                        ui.label(egui::RichText::new("Enter runs this line in the console").weak().small());
                    }
                    ui.separator();
                    // The window of rows around the highlight, so Down past the twelfth scrolls.
                    let first = state.highlight.saturating_sub(ROWS - 1);
                    for (i, hit) in hits.iter().enumerate().skip(first).take(ROWS) {
                        let c = hit.command;
                        let highlighted = i == state.highlight && !console_line;
                        let place = if c.category.is_empty() { "console".to_string() } else { c.category.join(" › ") };
                        let right = match c.binding_label() {
                            Some(key) => format!("{}   {key}", c.console_line()),
                            None => c.console_line(),
                        };
                        let name = egui::RichText::new(c.display_name.trim_end_matches("..."));
                        let name = if hit.available.is_ok() { name } else { name.weak() };
                        let row = ui.add(
                            egui::Button::selectable(highlighted, (name, egui::RichText::new(format!("  {place}")).weak().small()))
                                .shortcut_text(egui::RichText::new(right).weak())
                                .sense(egui::Sense::click_and_drag()),
                        );
                        // Dragged onto the shelf, it stays there (§10.2).
                        row.dnd_set_drag_payload(app::shelf::ShelfItem(c.id));
                        let row = match hit.available {
                            Ok(()) if !c.hint.is_empty() => row.on_hover_text(c.hint),
                            Ok(()) => row,
                            Err(why) => row.on_hover_text(why),
                        };
                        if let Err(why) = hit.available
                            && highlighted
                        {
                            ui.label(egui::RichText::new(why).weak().small());
                        }
                        if row.clicked() && hit.available.is_ok() {
                            run = Some(c);
                        }
                    }
                    if hits.is_empty() && !console_line {
                        ui.label(egui::RichText::new("No command matches").weak());
                    }
                    // Enter in a single-line field gives its focus up, so the key is read here.
                    if enter && (edit.lost_focus() || edit.has_focus()) {
                        if console_line {
                            passthrough = Some(state.query.clone());
                        } else if let Some(hit) = hits.get(state.highlight)
                            && hit.available.is_ok()
                        {
                            run = Some(hit.command);
                        }
                    }
                    if escape {
                        close = true;
                    }
                });
            });
        if close || run.is_some() || passthrough.is_some() {
            if let Some(focused) = ctx.memory(|m| m.focused()) {
                ctx.memory_mut(|m| m.surrender_focus(focused));
            }
        } else {
            self.palette = Some(state);
        }
        if let Some(line) = passthrough {
            // The palette and the console are one input path (§5.5).
            self.run_command(&line);
        }
        if let Some(c) = run {
            self.invoke_command(c);
        }
    }
}
