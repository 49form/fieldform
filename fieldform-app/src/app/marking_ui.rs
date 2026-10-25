// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The marking menu in the viewport (`interaction-spec.md` §7).
//!
//! **A right-click opens it; the navigation keeps the right-drag** (§9.2, §15.3). The menu's own
//! gesture, press–drag–release on the right button, is the dolly here, and the navigation wins, so
//! the menu opens on a click and a slot is picked by clicking it -- or by a left-drag from the
//! centre, released in its direction, which is the same gesture on the other hand. The Keys panel
//! lists the conflict.

use crate::*;
use marking::Subject;

/// The marking menu while it is open.
pub(crate) struct MarkingMenu {
    /// Where it opened: its centre.
    pub(crate) at: egui::Pos2,
    pub(crate) subject: Subject,
    /// A left-button press at the centre, waiting to be released in a direction.
    pub(crate) dragging: bool,
}

/// How far from the centre the slots' labels sit.
const RADIUS: f32 = 84.0;

impl App {
    /// What the selection is, for the marking menu's table (§7.2).
    pub(crate) fn marking_subject(&self) -> Subject {
        // Elements of the cage picked are the cage tools' table, ahead of the body they belong to
        // -- and so is being in the middle of cage work at all: a cage tool armed, an extrude or
        // inset pending, or the view in Cage mode, where the cage is the thing being edited (the
        // owner, 2026-10-24: a right-click while inserting an edge opened the field's menu).
        let cage_work = self.cage_tool.is_some() || self.cage_session.is_some() || self.view_mode == ViewMode::Polygon;
        if self.control == ControlMesh::Cage && (!self.selection.is_empty() || cage_work) {
            return Subject::CageElements;
        }
        if let Some(id) = self.selected_body {
            return match self.part_graph().node(id) {
                Some(Node::Plane { .. }) => Subject::Plane,
                Some(Node::Cage(_)) => Subject::Cage,
                Some(_) => Subject::Body,
                None => Subject::Empty,
            };
        }
        if let Some(curve) = self.selected_curve {
            return if fieldform_field::profile_loop(&self.doc, curve, None).is_ok() { Subject::Profile } else { Subject::Curve };
        }
        Subject::Empty
    }

    pub(crate) fn open_marking_menu(&mut self, at: egui::Pos2) {
        let mut subject = self.marking_subject();
        // In Field + Cage, a right-click *on the cage* is the cage's menu too, whatever is
        // selected: the cage is drawn there to be worked on (the owner, 2026-10-24).
        if subject != Subject::CageElements
            && self.control == ControlMesh::Cage
            && self.view_mode.cage_is_editable()
            && let Some(rect) = self.viewport_rect
            && rect.contains(at)
        {
            let aspect = rect.width() / rect.height().max(1.0);
            let under = pick(self.control(), &self.control_placement(), &self.cage_mesh, &self.camera, aspect, rect, at);
            if !matches!(under, Selection::None) {
                subject = Subject::CageElements;
            }
        }
        self.marking = Some(MarkingMenu { at, subject, dragging: false });
    }

    /// Draws the marking menu, if it is open, and runs what is picked.
    ///
    /// **One foreground widget over the whole menu**, so egui gives the menu its own clicks and
    /// the viewport underneath sees none of them.
    pub(crate) fn marking_ui(&mut self, ctx: &egui::Context) {
        let Some(mut menu) = self.marking.take() else { return };
        let ctx_now = self.command_context();
        let table = marking::slots(menu.subject);
        // What each slot would run, and whether it may.
        let entry = |i: usize| table[i].and_then(commands::by_id).map(|c| (c, (c.context)(&ctx_now)));
        let size = RADIUS * 2.9;
        let rect = egui::Rect::from_center_size(menu.at, egui::vec2(size, size));
        let delta_of = |p: egui::Pos2| (p.x - menu.at.x, p.y - menu.at.y);
        let in_centre = |p: egui::Pos2| {
            let (x, y) = delta_of(p);
            x.hypot(y) < marking::DEAD_ZONE
        };
        let mut pick: Option<usize> = None;
        let mut close = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        let shown = egui::Area::new(egui::Id::new("marking menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(rect.min)
            .show(ctx, |ui| {
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
                let pointer = response.hover_pos().or(response.interact_pointer_pos());
                let hovered = pointer.and_then(|p| marking::slot_at(delta_of(p)));
                let visuals = ui.visuals().clone();
                let painter = ui.painter();
                painter.circle_filled(menu.at, RADIUS * 1.42, visuals.window_fill.gamma_multiply(0.94));
                painter.circle_stroke(menu.at, RADIUS * 1.42, visuals.window_stroke);
                painter.circle_stroke(menu.at, marking::DEAD_ZONE, egui::Stroke::new(1.0_f32, visuals.weak_text_color()));
                for i in 0..8 {
                    let (dx, dy) = marking::slot_offset(i, RADIUS);
                    let pos = menu.at + egui::vec2(dx, dy);
                    let (text, enabled) = if i == marking::MORE {
                        ("More…".to_string(), true)
                    } else {
                        match entry(i) {
                            Some((c, ok)) => (c.display_name.trim_end_matches("...").to_string(), ok.is_ok()),
                            None => continue,
                        }
                    };
                    let lit = hovered == Some(i) && enabled;
                    if lit {
                        painter.circle_filled(pos, 30.0, visuals.selection.bg_fill.gamma_multiply(0.6));
                    }
                    let colour = if lit {
                        visuals.strong_text_color()
                    } else if enabled {
                        visuals.text_color()
                    } else {
                        visuals.weak_text_color()
                    };
                    painter.text(pos, egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(13.0), colour);
                }
                // A left-drag from the centre, released in a direction, picks that direction; a
                // click picks the slot under it; a right-click picks too.
                //
                // **The centre is empty, at the owner's asking (2026-10-17)**, where it used to repeat
                // the last command (§7.2): a right-click made by mistake is undone by clicking again
                // where it landed, which is the centre. As a repeat, that second click ran the last
                // command, so clicking on in the same place stacked a pattern on a pattern until the
                // display timed out. Shift+R and Edit › Repeat last still repeat.
                if response.drag_started_by(egui::PointerButton::Primary) {
                    menu.dragging = ui.input(|i| i.pointer.press_origin()).is_some_and(in_centre);
                }
                if response.drag_stopped() {
                    if menu.dragging {
                        pick = hovered;
                        close = hovered.is_none();
                    }
                    menu.dragging = false;
                }
                if response.clicked() || response.secondary_clicked() {
                    match pointer {
                        Some(p) if !in_centre(p) && hovered.is_some() => pick = hovered,
                        _ => close = true,
                    }
                }
                response
            });
        // A press anywhere outside the menu closes it, as a click away from any menu does.
        if ctx.input(|i| i.pointer.any_pressed()) && !shown.inner.hovered() && !shown.inner.is_pointer_button_down_on() {
            close = true;
        }
        match pick {
            Some(marking::MORE) => {
                let on_offer: Vec<&'static str> = commands::COMMANDS
                    .iter()
                    .filter(|c| c.built() && (c.context)(&ctx_now).is_ok())
                    .map(|c| c.id)
                    .collect();
                self.open_palette(Some(on_offer));
            }
            Some(i) => match entry(i) {
                Some((c, Ok(()))) => self.invoke_command(c),
                // A dimmed slot does nothing, and the menu stays open on it.
                _ => self.marking = Some(menu),
            },
            None if close => {}
            None => self.marking = Some(menu),
        }
    }
}
