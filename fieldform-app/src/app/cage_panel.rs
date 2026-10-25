// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The floating panel of a pending extrude or inset: its numbers typed, scrubbed or read off the
//! drag, re-placing the cap as they change (the cage-editing notes, R8; the interaction notes §4).
//!
//! **The same panel the blocks have, over a cage session.** Rows are the lift's own console keys,
//! so what is typed here is what the kept line says: an extrude's Distance is read off the cap --
//! the mean of how far it stands from where it was lifted, along the direction -- and typing one
//! puts the cap there; Direction and Mode re-place or re-lift it; an inset's Thickness and Depth
//! re-run the inset. A cage tool has no block rows in the list, so this panel is shown for a cage
//! session whether or not the blocks' panel is on.

use super::cage_tools::CageSession;
use super::tool_panel::{clear_of, field_id, is_expression, Row};
use crate::*;
use fieldform_cage::validate::face_normal;
use fieldform_cage::FaceId;

/// Where a word row's choices are, so a click cycles them and a typed word is checked.
fn choices(op: &str, key: &str) -> &'static [&'static str] {
    match (op, key) {
        ("cage.extrude", "direction") => &["normal", "x", "y", "z"],
        (_, "mode") => &["region", "individual"],
        _ => &[],
    }
}

impl App {
    /// The direction an extrude's cap stands off along: the cap's own normal, or a world axis.
    fn extrude_direction(&self, session: &CageSession) -> Vector3<f64> {
        match session.args.get("direction").map(|v| v.text()).as_deref() {
            Some("x") => Vector3::x(),
            Some("y") => Vector3::y(),
            Some("z") => Vector3::z(),
            _ => session.normal,
        }
    }

    /// How far the cap stands from where it was lifted, along the direction: the mean over its
    /// vertices, which is the distance of a straight extrude exactly and a fair reading of a cap
    /// that was also turned.
    pub(crate) fn extrude_distance(&self, session: &CageSession) -> f64 {
        let mesh = self.cage();
        let dir = self.extrude_direction(session);
        let n = session.cap.len().max(1) as f64;
        session.cap.iter().zip(&session.cap_before).map(|(&v, b)| (mesh.vertex_position(v) - b).dot(&dir)).sum::<f64>() / n
    }

    /// Whether every cap vertex sits exactly at the straight extrude of `distance`: then the kept
    /// line can say the distance alone, and no positions.
    pub(crate) fn extrude_is_straight(&self, session: &CageSession, distance: f64) -> bool {
        let mesh = self.cage();
        let dir = self.extrude_direction(session);
        let eps = fieldform_cage::validate::epsilon_length(mesh);
        session.cap.iter().zip(&session.cap_before).all(|(&v, b)| (mesh.vertex_position(v) - (b + dir * distance)).norm() <= eps)
    }

    /// The rows: an extrude's distance read off the cap, then its direction and mode; an inset's
    /// thickness, depth and mode.
    pub(crate) fn cage_panel_rows(&self) -> Vec<Row> {
        let Some(session) = &self.cage_session else { return Vec::new() };
        let word = |key: &'static str, default: &str| Row {
            key,
            text: session.args.get(key).map(|v| v.text()).unwrap_or_else(|| default.to_string()),
            unit: commands::Unit::Word,
        };
        let number = |key: &'static str| Row {
            key,
            text: session.args.get(key).and_then(|v| v.as_f64()).map(echo::number).unwrap_or_else(|| "0".to_string()),
            unit: commands::Unit::Mm,
        };
        match session.command.op {
            "cage.extrude" => vec![
                Row { key: "distance", text: echo::number(self.extrude_distance(session)), unit: commands::Unit::Mm },
                word("direction", "normal"),
                word("mode", "region"),
            ],
            _ => vec![number("thickness"), number("depth"), word("mode", "region")],
        }
    }

    /// The lift's line as the session now stands, for the echo: the numbers it was given, and an
    /// extrude's distance where the cap sits at a straight extrude of it.
    pub(crate) fn cage_session_line(&self, session: &CageSession, distance: Option<f64>) -> String {
        let faces = session.faces.iter().map(|f| f.index().to_string()).collect::<Vec<_>>().join(",");
        let mut line = format!("{} faces={faces}", session.command.console_name);
        match session.command.op {
            "cage.extrude" => {
                line.push_str(&format!(" distance={}", echo::number(distance.unwrap_or(0.0))));
                for key in ["direction", "mode"] {
                    if let Some(v) = session.args.get(key) {
                        line.push_str(&format!(" {key}={}", v.text()));
                    }
                }
            }
            _ => {
                for key in ["thickness", "depth", "mode"] {
                    if let Some(v) = session.args.get(key) {
                        line.push_str(&format!(" {key}={}", v.text()));
                    }
                }
            }
        }
        line
    }

    /// Puts the cap at a straight extrude of `distance` along the direction, as one positions
    /// command on the session's stack, so it folds into the kept step like a drag does.
    fn place_cap(&mut self, distance: f64) {
        let Some(session) = self.cage_session.as_ref() else { return };
        let dir = self.extrude_direction(session);
        let cap = session.cap.clone();
        let target: Vec<Point3<f64>> = session.cap_before.iter().map(|b| b + dir * distance).collect();
        let mesh = self.cage();
        let current: Vec<Point3<f64>> = cap.iter().map(|&v| mesh.vertex_position(v)).collect();
        if current.iter().zip(&target).all(|(c, t)| (c - t).norm() < 1e-12) {
            return;
        }
        let cmd = self.positions_command(ControlMesh::Cage, cap, current, target);
        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
            self.warn(e.to_string());
        }
        self.after_document_change();
        if let Some(session) = self.cage_session.as_ref() {
            let faces = session.faces.clone();
            self.selection.clear();
            for f in faces {
                self.selection.toggle(Selection::Face(f));
            }
        }
    }

    /// Lifts again with the session's numbers as they now are -- the mesh put back to before the
    /// session, the lift dispatched afresh -- keeping an extrude's distance.
    fn relift(&mut self) {
        let Some(session) = self.cage_session.as_ref() else { return };
        let distance = if session.command.op == "cage.extrude" { Some(self.extrude_distance(session)) } else { None };
        // The session's entries dropped back to its mark and the mesh put back, then a fresh mark
        // for the new lift -- with the session pending again before the lift lands, so the body
        // draws it as a draft.
        let Some(mut session) = self.cage_session.take() else { return };
        let old_mark = std::mem::replace(&mut session.mark, self.stack.mark());
        self.stack.cancel_to(old_mark);
        if let Some(body) = self.doc.cage_body_mut(session.cage) {
            body.set_mesh(session.before.restore());
        }
        session.mark = self.stack.mark();
        let (op, args) = (session.command.op, session.args.clone());
        self.cage_session = Some(session);
        self.dispatch(op, &args);
        self.read_cap();
        if let Some(d) = distance {
            self.place_cap(d);
        }
    }

    /// Reads the cap off the mesh after a lift: its vertices, where they are, and its normal; and
    /// selects it so the manipulator sits on it.
    pub(crate) fn read_cap(&mut self) {
        let Some(session) = self.cage_session.as_ref() else { return };
        let faces = session.faces.clone();
        let mesh = self.cage();
        let mut cap: Vec<VertexId> = faces.iter().flat_map(|&f| mesh.face_vertices(f)).collect();
        cap.sort();
        cap.dedup();
        let cap_before: Vec<Point3<f64>> = cap.iter().map(|&v| mesh.vertex_position(v)).collect();
        let normal = faces.iter().map(|&f| face_normal(mesh, f)).sum::<Vector3<f64>>().try_normalize(1e-12).unwrap_or(Vector3::z());
        if let Some(session) = self.cage_session.as_mut() {
            session.cap = cap;
            session.cap_before = cap_before;
            session.normal = normal;
        }
        self.selection.clear();
        for f in faces {
            self.selection.toggle(Selection::Face(f));
        }
    }

    /// Confirms a row: a number placed or re-lifted, a word checked against its choices.
    pub(crate) fn cage_panel_set(&mut self, key: &'static str, text: &str) -> Result<(), String> {
        let Some(session) = self.cage_session.as_ref() else { return Ok(()) };
        let op = session.command.op;
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        let words = choices(op, key);
        if !words.is_empty() {
            let word = text.to_ascii_lowercase();
            if !words.contains(&word.as_str()) {
                return Err(format!("{key} is one of {}", words.join(", ")));
            }
            if let Some(session) = self.cage_session.as_mut() {
                session.args = session.args.clone().with(key, Value::Str(word));
            }
            if op == "cage.extrude" && key == "direction" {
                // The same distance, along the new direction.
                let d = self.cage_session.as_ref().map(|s| self.extrude_distance(s)).unwrap_or(0.0);
                self.place_cap(d);
            } else {
                self.relift();
            }
            return Ok(());
        }
        if is_expression(text) {
            return Err("type a number here; an expression needs a parameter, which a cage tool has none of".to_string());
        }
        let value: f64 = text.parse().map_err(|_| format!("{text:?} is not a number"))?;
        if !value.is_finite() {
            return Err("that is not a finite number".to_string());
        }
        if op == "cage.extrude" && key == "distance" {
            self.place_cap(value);
            return Ok(());
        }
        if op == "cage.inset" && key == "thickness" && value <= 0.0 {
            return Err("an inset's thickness is more than nought".to_string());
        }
        if let Some(session) = self.cage_session.as_mut() {
            session.args = session.args.clone().with(key, Value::Float(value));
        }
        self.relift();
        Ok(())
    }

    /// Typing with no click while a cage session is pending (§4.3): a digit, `-` or `.` starts
    /// typing into the focused row. `true` when it did, so the keys are not also bindings.
    pub(crate) fn cage_panel_typed(&mut self, ctx: &egui::Context) -> bool {
        if self.cage_session.is_none() || ctx.egui_wants_keyboard_input() {
            return false;
        }
        if self.cage_session.as_ref().is_some_and(|s| s.panel.typing.is_some()) {
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
        let rows = self.cage_panel_rows();
        let Some(session) = self.cage_session.as_mut() else { return false };
        let at = session.panel.focus.min(rows.len().saturating_sub(1));
        if rows.get(at).is_some_and(|r| r.unit == commands::Unit::Word) {
            return false;
        }
        session.panel.typing = Some(text);
        session.panel.begun = true;
        session.panel.error = None;
        ctx.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Text(_))));
        true
    }

    /// Tab and Shift+Tab with a cage session pending: confirm, then the next row or the one before.
    pub(crate) fn cage_panel_tab(&mut self, back: bool) {
        let rows = self.cage_panel_rows();
        if rows.is_empty() {
            return;
        }
        let Some(session) = self.cage_session.as_mut() else { return };
        let typed = session.panel.typing.take();
        let at = session.panel.focus.min(rows.len() - 1);
        if let Some(text) = typed {
            let result = self.cage_panel_set(rows[at].key, &text);
            if let Some(session) = self.cage_session.as_mut() {
                session.panel.error = result.err();
            }
        }
        if let Some(session) = self.cage_session.as_mut() {
            session.panel.focus = if back { (at + rows.len() - 1) % rows.len() } else { (at + 1) % rows.len() };
        }
        let id = self.cage_session.as_ref().map(|s| field_id(s.command.id));
        if let Some(id) = id {
            self.egui_ctx.memory_mut(|m| m.surrender_focus(id));
        }
    }

    /// The cap's extent on screen, for placing the panel clear of it.
    fn cap_extent(&self) -> Option<egui::Rect> {
        let session = self.cage_session.as_ref()?;
        let rect = self.viewport_rect?;
        let aspect = rect.width() / rect.height().max(1.0);
        let placement = self.control_placement();
        let mesh = self.cage();
        let mut out: Option<egui::Rect> = None;
        for &v in &session.cap {
            if let Some(s) = self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, v)) {
                out = Some(out.map_or(egui::Rect::from_min_max(s, s), |r| r.union(egui::Rect::from_min_max(s, s))));
            }
        }
        out.map(|r| r.expand(24.0))
    }

    /// Draws the pending cage session's panel.
    pub(crate) fn cage_panel(&mut self, ctx: &egui::Context) {
        let Some(session) = self.cage_session.as_mut() else { return };
        let mut panel = std::mem::take(&mut session.panel);
        let command = session.command;
        let rows = self.cage_panel_rows();
        panel.focus = panel.focus.min(rows.len().saturating_sub(1));
        let window_id = egui::Id::new(("cage tool panel", command.id));
        let placing = panel.anchor.is_none();
        let avoid = self.cap_extent();
        let anchor = *panel.anchor.get_or_insert_with(|| {
            let pointer = ctx.pointer_latest_pos().unwrap_or(egui::pos2(320.0, 120.0)) + egui::vec2(28.0, 28.0);
            let first = egui::Rect::from_min_size(pointer, egui::vec2(300.0, 70.0 + 26.0 * rows.len() as f32));
            let at = ctx.memory(|m| m.area_rect(window_id)).unwrap_or(first);
            clear_of(at, avoid, ctx.content_rect())
        });
        let fid = field_id(command.id);
        let mut confirm: Option<(&'static str, String)> = None;
        let mut scrubbed: Option<(&'static str, f64)> = None;
        let mut cycled: Option<(&'static str, String)> = None;
        panel.rects.clear();
        let (shift, ctrl, middle_down, dx) = ctx.input(|i| {
            (i.modifiers.shift, i.modifiers.ctrl || i.modifiers.command, i.pointer.middle_down(), i.pointer.delta().x)
        });
        let window = egui::Window::new(command.display_name).id(window_id).collapsible(false).resizable(false);
        let window = if placing { window.current_pos(anchor) } else { window };
        window.show(ctx, |ui| {
            for (i, row) in rows.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.add_sized(egui::vec2(96.0, ui.spacing().interact_size.y), egui::Label::new(row.label()));
                    let focused = i == panel.focus;
                    let words = choices(command.op, row.key);
                    match (&mut panel.typing, focused) {
                        (Some(buffer), true) if words.is_empty() => {
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
                                if !escape {
                                    confirm = Some((row.key, buffer.clone()));
                                }
                                panel.typing = None;
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
                                if words.is_empty() {
                                    panel.typing = Some(row.text.clone());
                                    panel.begun = true;
                                } else {
                                    // A word row cycles through its choices on a click.
                                    let at = words.iter().position(|w| *w == row.text).unwrap_or(0);
                                    cycled = Some((row.key, words[(at + 1) % words.len()].to_string()));
                                }
                            }
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
                                    let rate = crate::scrub_speed(start, 0.1) * if shift { 0.1 } else if ctrl { 10.0 } else { 1.0 };
                                    scrubbed = Some((row.key, start + f64::from(dx) * rate));
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
            ui.label(egui::RichText::new("Enter keep · Esc put back · Tab next · drag the cap").weak().small());
        });
        if !middle_down {
            panel.scrub = None;
        }
        if let Some(session) = self.cage_session.as_mut() {
            session.panel = panel;
        }
        let change = scrubbed.map(|(k, v)| (k, echo::number(v))).or(confirm).or(cycled);
        if let Some((key, text)) = change {
            let result = self.cage_panel_set(key, &text);
            if let Some(session) = self.cage_session.as_mut() {
                session.panel.error = result.err();
            }
        }
    }
}

#[allow(dead_code)]
fn unused(_: FaceId) {}
