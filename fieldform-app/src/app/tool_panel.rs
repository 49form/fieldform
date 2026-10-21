// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The tool panel: a pending session's numbers, typed or scrubbed in place (`interaction-spec.md`
//! §4).
//!
//! **Floating, non-modal, and with no buttons.** It opens by the pointer when a session begins,
//! remembers where it was dragged to for each command (egui keeps a window's place by its id), and
//! closes when the session is kept or taken back. Its rows are the command's params in registry
//! order -- the same keys the console reads and the echo writes -- so what is typed here is a `set`
//! of that key, and a value is a number, a parameter's name, or an expression, which becomes a
//! variable named after the row (§15.2).
//!
//! **Typing needs no click.** With a session pending and no text field holding the keyboard, a
//! digit, `-`, `.`, `(` or the first letter of a parameter's name starts typing into the row in
//! focus, seeded with that character (§4.3); Enter confirms the row, Tab confirms and moves on, Esc
//! reverts it, and Enter or Esc with nothing being typed keeps or takes back the session.

use crate::*;

/// The keys a session's panel does not offer: what the block is made from, which comes from the
/// selection, not from typing (`app::sessions`).
pub(crate) const MADE_FROM: &[&str] =
    &["target", "in", "a", "b", "toward", "input", "profile", "curve", "spine", "sections"];

/// What the panel remembers between frames.
#[derive(Default)]
pub(crate) struct Panel {
    /// The row with the keyboard's attention: highlighted, not yet typed into (§4.2).
    pub(crate) focus: usize,
    /// What is being typed into the focused row, while it is.
    pub(crate) typing: Option<String>,
    /// Typing began this frame: its field takes the keyboard, with the caret after the seed.
    pub(crate) begun: bool,
    /// Why the last value confirmed was refused, said in the panel rather than a dialog (§3.6).
    pub(crate) error: Option<String>,
    /// A middle-button scrub in progress: the row and the number it started from.
    pub(crate) scrub: Option<(usize, f64)>,
    /// Where the panel opened this session: where it was last left for this command, else by the
    /// pointer -- moved clear of the block where either would cover it (§4.1).
    pub(crate) anchor: Option<egui::Pos2>,
    /// Where each row's value was drawn last frame, for a test that presses on one.
    pub(crate) rects: Vec<egui::Rect>,
}

/// One row: a key the command reads, what the block holds under it, and its unit.
#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) key: &'static str,
    pub(crate) text: String,
    pub(crate) unit: commands::Unit,
}

impl Row {
    /// "base_radius" as "Base radius".
    pub(crate) fn label(&self) -> String {
        let spaced = self.key.replace('_', " ");
        let mut chars = spaced.chars();
        chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default()
    }

    /// The number the row holds, when it holds one number: what a scrub moves.
    pub(crate) fn number(&self) -> Option<f64> {
        self.text.trim().parse::<f64>().ok()
    }
}

/// Whether `text` is an expression rather than a value the console reads as it is: arithmetic in
/// it, and not a number (a leading minus is a number) or a list of them.
pub(crate) fn is_expression(text: &str) -> bool {
    let text = text.trim();
    text.parse::<f64>().is_err() && !text.contains(',') && text.chars().any(|c| "+-*/()^".contains(c))
}

/// The id of a row's field, so the keyboard can be handed to it and Tab taken from it.
pub(crate) fn field_id(command: &str) -> egui::Id {
    egui::Id::new(("tool panel field", command))
}

impl App {
    /// The rows the pending session's panel shows, in registry order, with what its block holds.
    pub(crate) fn panel_rows(&self) -> Vec<Row> {
        let Some(session) = &self.session else { return Vec::new() };
        let Some(node) = self.part_graph().node(session.block) else { return Vec::new() };
        let Ok(written) = echo::keys_of(node, &echo::Names { doc: &self.doc }) else { return Vec::new() };
        let held: Vec<(&'static str, String)> = written.keys.iter().chain(&written.after).cloned().collect();
        session
            .command
            .params
            .iter()
            .filter(|p| !MADE_FROM.contains(&p.key))
            .filter_map(|p| {
                held.iter().find(|(k, _)| *k == p.key).map(|(_, v)| Row { key: p.key, text: v.clone(), unit: p.unit })
            })
            .collect()
    }

    /// Confirms what was typed into `row`: a `set` of its key on the pending block (§4.3).
    ///
    /// **An expression becomes a variable named after the row** -- `Height`, then `Height_1` --
    /// holding the expression, and the field reads it, because the console takes no inline
    /// expression (item 81 A9) and the echo must say what was typed: its source text, never the
    /// number it came to (§6.3).
    pub(crate) fn panel_confirm(&mut self, row: &Row, text: &str) -> Result<(), String> {
        let Some(block) = self.session.as_ref().map(|s| s.block) else { return Ok(()) };
        let text = text.trim();
        if text.is_empty() || text == row.text {
            return Ok(());
        }
        let value = if is_expression(text) && row.unit != commands::Unit::Word {
            let base = Row { key: row.key, text: String::new(), unit: row.unit }.label().replace(' ', "_");
            let name = names::variable_name(&self.doc, &base);
            let id = self.doc.add_parameter(name.clone(), 0.0);
            if let Err(why) = self.commit_expression(id, text) {
                self.doc.parameters.remove(&id);
                return Err(why);
            }
            Value::Str(name)
        } else {
            console::parse_value(text)
        };
        self.set_block_keys(block, &Args::new().with(row.key, value))
    }

    /// Typing that starts with a key press and no click (§4.3): the characters typed this frame,
    /// when a session is pending and nothing else holds the keyboard. `true` when typing began, so
    /// the plain keys this frame are not also taken as bindings -- `h` starting `height` must not
    /// hide the selection as well.
    pub(crate) fn panel_typed(&mut self, ctx: &egui::Context) -> bool {
        if !self.floating_panel || self.session.is_none() || ctx.egui_wants_keyboard_input() {
            return false;
        }
        if self.session.as_ref().is_some_and(|s| s.panel.typing.is_some()) {
            return false;
        }
        let text: String = ctx.input(|i| {
            i.events.iter().filter_map(|e| match e {
                egui::Event::Text(t) => Some(t.as_str()),
                _ => None,
            }).collect()
        });
        let Some(first) = text.chars().next() else { return false };
        let starts_a_name = first.is_alphabetic()
            && self.doc.parameters.values().any(|p| p.name.chars().next().is_some_and(|c| c.eq_ignore_ascii_case(&first)));
        if !(first.is_ascii_digit() || "-.(".contains(first) || starts_a_name) {
            return false;
        }
        if self.panel_rows().is_empty() {
            return false;
        }
        let Some(session) = self.session.as_mut() else { return false };
        session.panel.typing = Some(text);
        session.panel.begun = true;
        session.panel.error = None;
        ctx.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Text(_))));
        true
    }

    /// Tab and Shift+Tab with a session pending: confirm what is being typed, and move the focus
    /// to the next row or the one before (§4.2).
    pub(crate) fn panel_tab(&mut self, back: bool) {
        let rows = self.panel_rows();
        if rows.is_empty() {
            return;
        }
        let Some(session) = self.session.as_mut() else { return };
        let typed = session.panel.typing.take();
        let at = session.panel.focus.min(rows.len() - 1);
        if let Some(text) = typed {
            let result = self.panel_confirm(&rows[at], &text);
            if let Some(session) = self.session.as_mut() {
                session.panel.error = result.err();
            }
        }
        if let Some(session) = self.session.as_mut() {
            session.panel.focus = if back { (at + rows.len() - 1) % rows.len() } else { (at + 1) % rows.len() };
        }
        let id = self.session.as_ref().map(|s| field_id(s.command.id));
        if let Some(id) = id {
            self.egui_ctx.memory_mut(|m| m.surrender_focus(id));
        }
    }

    /// Draws the panel of the pending session, if there is one.
    pub(crate) fn tool_panel(&mut self, ctx: &egui::Context) {
        // Off, the block's own rows in the list are where its numbers are typed.
        if !self.floating_panel {
            return;
        }
        let Some(session) = self.session.as_mut() else { return };
        let mut panel = std::mem::take(&mut session.panel);
        let command = session.command;
        let block = session.block;
        let rows = self.panel_rows();
        if rows.is_empty() && panel.error.is_none() {
            if let Some(session) = self.session.as_mut() {
                session.panel = panel;
            }
            return;
        }
        panel.focus = panel.focus.min(rows.len().saturating_sub(1));
        let window_id = egui::Id::new(("tool panel", command.id));
        // **Placed once a session, and never over what it is making** (§4.1): where the panel was
        // last left for this command -- egui keeps a window's place by its id, and the id is the
        // command's -- or beside the pointer the first time, then moved to the nearest side of
        // the block if that place would cover it. After this frame it is the person's to drag.
        let placing = panel.anchor.is_none();
        let anchor = *panel.anchor.get_or_insert_with(|| {
            let pointer = ctx.pointer_latest_pos().unwrap_or(egui::pos2(320.0, 120.0)) + egui::vec2(28.0, 28.0);
            // A panel not yet drawn has no size of its own: a row's label, value and unit across,
            // a row's height each and the title and footer down.
            let first = egui::Rect::from_min_size(pointer, egui::vec2(300.0, 70.0 + 26.0 * rows.len() as f32));
            let at = ctx.memory(|m| m.area_rect(window_id)).unwrap_or(first);
            clear_of(at, self.screen_extent(block), ctx.content_rect())
        });
        let fid = field_id(command.id);
        // What the drawing asked for, done once it has finished with the panel's borrow.
        let mut confirm: Option<(Row, String)> = None;
        let mut scrubbed: Option<(Row, f64)> = None;
        panel.rects.clear();
        let (shift, ctrl, middle_down, dx) = ctx.input(|i| {
            (i.modifiers.shift, i.modifiers.ctrl || i.modifiers.command, i.pointer.middle_down(), i.pointer.delta().x)
        });
        let window = egui::Window::new(command.display_name.trim_end_matches("..."))
            .id(window_id)
            .collapsible(false)
            .resizable(false);
        let window = if placing { window.current_pos(anchor) } else { window };
        window
            .show(ctx, |ui| {
                for (i, row) in rows.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.add_sized(egui::vec2(96.0, ui.spacing().interact_size.y), egui::Label::new(row.label()));
                        let focused = i == panel.focus;
                        match (&mut panel.typing, focused) {
                            (Some(buffer), true) => {
                                let response = ui.add(
                                    egui::TextEdit::singleline(buffer)
                                        .id(fid)
                                        .desired_width(140.0)
                                        .font(egui::TextStyle::Monospace),
                                );
                                panel.rects.push(response.rect);
                                if panel.begun {
                                    panel.begun = false;
                                    response.request_focus();
                                    let end = buffer.chars().count();
                                    let mut state = egui::TextEdit::load_state(ui.ctx(), fid).unwrap_or_default();
                                    state.cursor.set_char_range(Some(egui::text::CCursorRange::one(egui::text::CCursor::new(end))));
                                    state.store(ui.ctx(), fid);
                                } else if response.lost_focus() {
                                    let (enter, escape) = ui.input(|i| (i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Escape)));
                                    if escape {
                                        // Esc reverts the row, and a second Esc takes the session back.
                                        panel.typing = None;
                                    } else if enter || !escape {
                                        confirm = Some((row.clone(), buffer.clone()));
                                        panel.typing = None;
                                    }
                                }
                            }
                            _ => {
                                let shown = egui::RichText::new(&row.text).monospace();
                                let response = ui.add_sized(
                                    egui::vec2(140.0, ui.spacing().interact_size.y),
                                    egui::Button::selectable(focused, shown).sense(egui::Sense::click_and_drag()),
                                );
                                panel.rects.push(response.rect);
                                if response.clicked() {
                                    panel.focus = i;
                                    panel.typing = Some(row.text.clone());
                                    panel.begun = true;
                                }
                                // A middle-button drag scrubs the number (§4.4): Shift for fine
                                // steps, Ctrl for coarse, live, and one edit when the button lifts.
                                if let Some(start) = row.number() {
                                    if response.hovered() && middle_down && panel.scrub.is_none() {
                                        panel.scrub = Some((i, start));
                                        panel.focus = i;
                                    }
                                    if let Some((at, _)) = panel.scrub
                                        && at == i
                                        && middle_down
                                        && dx != 0.0
                                    {
                                        let rate = scrub_speed(start, 0.1) * if shift { 0.1 } else if ctrl { 10.0 } else { 1.0 };
                                        scrubbed = Some((row.clone(), start + f64::from(dx) * rate));
                                    }
                                }
                            }
                        }
                        ui.label(egui::RichText::new(row.unit.suffix()).weak());
                    });
                }
                if let Some(why) = &panel.error {
                    ui.label(egui::RichText::new(why).color(ui.visuals().warn_fg_color).small());
                }
                ui.separator();
                ui.label(egui::RichText::new("Enter commit · Esc cancel · Tab next").weak().small());
            });
        if !middle_down {
            panel.scrub = None;
        }
        if let Some(session) = self.session.as_mut() {
            session.panel = panel;
        }
        if let Some((row, value)) = scrubbed {
            let result = self.set_block_keys(self.session.as_ref().map_or(NodeId::new(), |s| s.block), &Args::new().with(row.key, Value::Float(value)));
            if let Some(session) = self.session.as_mut() {
                session.panel.error = result.err();
            }
        }
        if let Some((row, text)) = confirm {
            let result = self.panel_confirm(&row, &text);
            if let Some(session) = self.session.as_mut() {
                session.panel.error = result.err();
            }
        }
    }
}

/// Where a panel at `at` goes so that it covers none of `avoid`, staying inside `screen`: where it
/// is if it covers nothing already, else beside `avoid` on whichever side -- right, left, below,
/// above -- is the shortest move and fits. Where no side fits, a block that fills the view, it
/// stays put: a block that is everywhere cannot be kept clear of, and moving would only lose the
/// place the person chose.
pub(crate) fn clear_of(at: egui::Rect, avoid: Option<egui::Rect>, screen: egui::Rect) -> egui::Pos2 {
    const GAP: f32 = 16.0;
    let Some(avoid) = avoid.filter(|a| a.intersects(at)) else { return at.min };
    let size = at.size();
    let along_y = |p: egui::Pos2| egui::pos2(p.x, p.y.clamp(screen.min.y, (screen.max.y - size.y).max(screen.min.y)));
    let along_x = |p: egui::Pos2| egui::pos2(p.x.clamp(screen.min.x, (screen.max.x - size.x).max(screen.min.x)), p.y);
    [
        along_y(egui::pos2(avoid.max.x + GAP, at.min.y)),
        along_y(egui::pos2(avoid.min.x - GAP - size.x, at.min.y)),
        along_x(egui::pos2(at.min.x, avoid.max.y + GAP)),
        along_x(egui::pos2(at.min.x, avoid.min.y - GAP - size.y)),
    ]
    .into_iter()
    .filter(|p| screen.contains_rect(egui::Rect::from_min_size(*p, size)))
    .min_by(|a, b| a.distance(at.min).total_cmp(&b.distance(at.min)))
    .unwrap_or(at.min)
}

impl App {
    /// Block `id`'s box as the viewport shows it: the rectangle round its eight corners projected,
    /// cut to the viewport. `None` for a block with no finite extent -- a plane, a lattice -- or a
    /// box with a corner behind the camera, where the rectangle would say nothing true.
    pub(crate) fn screen_extent(&self, id: NodeId) -> Option<egui::Rect> {
        let (lo, hi) = self.body_extent(id).as_box()?;
        let rect = self.viewport_rect?;
        let aspect = rect.width() / rect.height().max(1.0);
        let mut out: Option<egui::Rect> = None;
        for corner in 0..8 {
            let pick = |bit: usize, a: f64, b: f64| if corner & bit == 0 { a } else { b };
            let p = Point3::new(pick(1, lo.x, hi.x), pick(2, lo.y, hi.y), pick(4, lo.z, hi.z));
            let s = self.camera.project_to_screen(aspect, rect, p.cast::<f32>())?;
            out = Some(out.map_or(egui::Rect::from_min_max(s, s), |r| r.union(egui::Rect::from_min_max(s, s))));
        }
        out.map(|r| r.intersect(rect)).filter(|r| r.is_positive())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h))
    }

    /// **A panel clear of the block stays where it is; one over it moves beside it, inside the
    /// screen; one over a block that fills the screen stays** (§4.1).
    #[test]
    fn a_panel_moves_clear_of_the_block_only_where_it_covers_it() {
        let screen = rect(0.0, 0.0, 1400.0, 900.0);
        let panel = rect(500.0, 300.0, 300.0, 150.0);
        assert_eq!(clear_of(panel, Some(rect(100.0, 100.0, 200.0, 200.0)), screen), panel.min, "clear already");
        assert_eq!(clear_of(panel, None, screen), panel.min, "nothing to keep clear of");
        let block = rect(400.0, 250.0, 300.0, 300.0);
        let moved = clear_of(panel, Some(block), screen);
        let now = egui::Rect::from_min_size(moved, panel.size());
        assert!(!now.intersects(block) && screen.contains_rect(now), "{now:?} against {block:?}");
        // Against the right edge the right side does not fit, and the shortest move that does is
        // taken -- above it, here.
        let edge = rect(1000.0, 250.0, 380.0, 300.0);
        let moved = clear_of(rect(1050.0, 300.0, 300.0, 150.0), Some(edge), screen);
        let now = egui::Rect::from_min_size(moved, egui::vec2(300.0, 150.0));
        assert!(!now.intersects(edge) && screen.contains_rect(now), "{now:?}");
        assert_eq!(moved, egui::pos2(1050.0, 84.0), "above is 216 px away, below 266 and left 366");
        let everywhere = rect(-10.0, -10.0, 1420.0, 920.0);
        assert_eq!(clear_of(panel, Some(everywhere), screen), panel.min, "nowhere to go");
    }
}
