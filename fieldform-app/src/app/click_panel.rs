// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The settings window the click tools share -- the knife, Insert Edge, Bridge and Weld -- drawn
//! the way a session's panel is (`tool_panel.rs`): rows in order, a click on a number types into
//! it, a click on a word cycles it, Tab walks the rows, and a middle-button drag scrubs a number.
//!
//! **Settings, not a session.** A click tool holds nothing pending between clicks, so there is
//! nothing here for Enter to keep or Esc to take back; the rows are the tool's own settings,
//! which carry from one click to the next and are never part of a committed line unless the line
//! has a key for them (`count=`, `divisions=`, `at=`). The tool decides what each row means; this
//! file only draws them and hands back what was changed.

use super::tool_panel::{clear_of, field_id, Panel, Row};
use crate::*;

/// Where a word row's choices are. One table for every click tool, since a key means one thing.
pub(crate) fn choices(key: &str) -> &'static [&'static str] {
    match key {
        "pick_type" => &["vertices", "edges", "both"],
        "offset_mode" => &["mm", "fraction"],
        "cut_through_faces" => &["all", "front-facing", "selected"],
        "position" => &["target", "midpoint"],
        "boundary" => &["off", "zip"],
        _ => &[],
    }
}

/// Typing with no click (§4.3): a digit, `-` or `.` starts typing into the focused row, unless it
/// is a word row. `true` when it did, so the keys are not also bindings.
pub(crate) fn typed(ctx: &egui::Context, panel: &mut Panel, rows: &[Row]) -> bool {
    if ctx.egui_wants_keyboard_input() || panel.typing.is_some() {
        return false;
    }
    let text: String = ctx.input(|i| {
        i.events.iter().filter_map(|e| match e {
            egui::Event::Text(t) => Some(t.as_str()),
            _ => None,
        }).collect()
    });
    let Some(first) = text.chars().next() else { return false };
    if !(first.is_ascii_digit() || "-.".contains(first)) {
        return false;
    }
    let at = panel.focus.min(rows.len().saturating_sub(1));
    if rows.get(at).is_none_or(|r| r.unit == commands::Unit::Word) {
        return false;
    }
    panel.typing = Some(text);
    panel.begun = true;
    panel.error = None;
    ctx.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Text(_))));
    true
}

/// Tab and Shift+Tab: what was being typed is handed back to be confirmed, and the focus moves to
/// the next row or the one before.
pub(crate) fn tab(ctx: &egui::Context, panel: &mut Panel, rows: &[Row], back: bool) -> Option<(&'static str, String)> {
    if rows.is_empty() {
        return None;
    }
    let typed = panel.typing.take();
    let at = panel.focus.min(rows.len() - 1);
    panel.focus = if back { (at + rows.len() - 1) % rows.len() } else { (at + 1) % rows.len() };
    ctx.memory_mut(|m| m.surrender_focus(field_id("click tool")));
    typed.map(|text| (rows[at].key, text))
}

/// Draws the window and returns the one change it asked for this frame, if any: a typed value
/// confirmed, a scrub, or a word cycled. `disabled` rows are greyed and cannot be changed.
pub(crate) fn window(
    ctx: &egui::Context,
    title: &str,
    rows: &[Row],
    disabled: &dyn Fn(&str) -> bool,
    panel: &mut Panel,
    footer: &str,
) -> Option<(&'static str, String)> {
    panel.focus = panel.focus.min(rows.len().saturating_sub(1));
    let window_id = egui::Id::new(("click tool panel", title));
    let placing = panel.anchor.is_none();
    let anchor = *panel.anchor.get_or_insert_with(|| {
        let pointer = ctx.pointer_latest_pos().unwrap_or(egui::pos2(320.0, 120.0)) + egui::vec2(28.0, 28.0);
        let first = egui::Rect::from_min_size(pointer, egui::vec2(300.0, 70.0 + 26.0 * rows.len() as f32));
        let at = ctx.memory(|m| m.area_rect(window_id)).unwrap_or(first);
        clear_of(at, None, ctx.content_rect())
    });
    let fid = field_id("click tool");
    let mut confirm: Option<(&'static str, String)> = None;
    let mut scrubbed: Option<(&'static str, f64)> = None;
    let mut cycled: Option<(&'static str, String)> = None;
    panel.rects.clear();
    let (shift, ctrl, middle_down, dx) = ctx.input(|i| {
        (i.modifiers.shift, i.modifiers.ctrl || i.modifiers.command, i.pointer.middle_down(), i.pointer.delta().x)
    });
    let win = egui::Window::new(title).id(window_id).collapsible(false).resizable(false);
    let win = if placing { win.current_pos(anchor) } else { win };
    win.show(ctx, |ui| {
        for (i, row) in rows.iter().enumerate() {
            let off = disabled(row.key);
            ui.add_enabled_ui(!off, |ui| {
                ui.horizontal(|ui| {
                    ui.add_sized(egui::vec2(110.0, ui.spacing().interact_size.y), egui::Label::new(row.label()));
                    let focused = i == panel.focus;
                    let words = choices(row.key);
                    match (&mut panel.typing, focused) {
                        (Some(buffer), true) if words.is_empty() && !off => {
                            let response = ui.add(
                                egui::TextEdit::singleline(buffer).id(fid).desired_width(140.0).font(egui::TextStyle::Monospace),
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
                                let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
                                if !escape && *buffer != row.shown() {
                                    confirm = Some((row.key, buffer.clone()));
                                }
                                panel.typing = None;
                            }
                        }
                        _ => {
                            let shown = egui::RichText::new(row.shown()).monospace();
                            let response = ui.add_sized(
                                egui::vec2(140.0, ui.spacing().interact_size.y),
                                egui::Button::selectable(focused, shown).sense(egui::Sense::click_and_drag()),
                            );
                            panel.rects.push(response.rect);
                            if !off && response.clicked() {
                                panel.focus = i;
                                if words.is_empty() {
                                    panel.typing = Some(row.shown());
                                    panel.begun = true;
                                } else {
                                    let at = words.iter().position(|w| *w == row.text).unwrap_or(0);
                                    cycled = Some((row.key, words[(at + 1) % words.len()].to_string()));
                                }
                            }
                            if !off
                                && let Some(start) = row.number()
                            {
                                if response.hovered() && middle_down && panel.scrub.is_none() {
                                    panel.scrub = Some((i, start));
                                    panel.focus = i;
                                }
                                if let Some((at, _)) = panel.scrub
                                    && at == i
                                    && middle_down
                                    && dx != 0.0
                                {
                                    let rate = crate::scrub_speed(start, 0.1) * if shift { 0.1 } else if ctrl { 10.0 } else { 1.0 };
                                    scrubbed = Some((row.key, start + f64::from(dx) * rate));
                                }
                            }
                        }
                    }
                    ui.label(egui::RichText::new(row.unit.suffix()).weak());
                });
            });
        }
        if let Some(why) = &panel.error {
            ui.label(egui::RichText::new(why).color(ui.visuals().warn_fg_color).small());
        }
        ui.separator();
        ui.label(egui::RichText::new(footer).weak().small());
    });
    if !middle_down {
        panel.scrub = None;
    }
    scrubbed.map(|(k, v)| (k, echo::number(v))).or(confirm).or(cycled)
}
