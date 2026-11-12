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
    /// The join a [`Subject::Joint`] ring was opened on: its two ends.
    pub(crate) join: Option<[fieldform_scene::CurveEnd; 2]>,
    /// G1 or G2 picked on that ring: the ring has become the list of which curve leads.
    pub(crate) lead_for: Option<fieldform_scene::JointContinuity>,
}

/// A join's continuity drawn before it is chosen: the points a lead hovered in the join's list
/// would move, moved in the document, and where each was, to put back.
///
/// **In the document, not beside it, because everything drawn reads the document**: the curves,
/// and any extrude or sweep built on them. A copy drawn instead would need a second path through
/// the overlay and the compile, and the preview would then be of something other than what a
/// click keeps. So it is held to three rules: it never reaches the undo stack (it is not an edit,
/// and the click that keeps it makes the edit afresh, as `continuity`), the echo and the autosave
/// wait while it is up, and it is put back before any command runs.
pub(crate) struct JoinPreview {
    pub(crate) relation: fieldform_scene::Relation,
    pub(crate) saved: Vec<(CurveId, usize, nalgebra::Point3<f64>)>,
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
        self.marking = Some(MarkingMenu { at, subject, dragging: false, join: None, lead_for: None });
    }

    /// The ring a right-click on a join opens: G0, G1 and G2 for those two ends.
    pub(crate) fn open_join_menu(&mut self, at: egui::Pos2, join: [fieldform_scene::CurveEnd; 2]) {
        self.marking = Some(MarkingMenu { at, subject: Subject::Joint, dragging: false, join: Some(join), lead_for: None });
    }

    /// Shows `relation` at the join, or puts back what was there with `None`. A no-op when it is
    /// already the one shown, so hovering a row costs one rebuild on entering it and none after.
    pub(crate) fn preview_join(
        &mut self,
        relation: Option<fieldform_scene::Relation>,
        replace: Option<fieldform_scene::RelationId>,
    ) {
        if self.join_preview.as_ref().map(|p| &p.relation) == relation.as_ref() {
            return;
        }
        self.put_back_join_preview();
        if let Some(relation) = relation {
            let mut after = self.doc.clone();
            if let Some(old) = replace {
                after.relations.remove(&old);
            }
            after.relations.insert(
                fieldform_scene::RelationId::new(),
                fieldform_scene::RelationRecord { relation: relation.clone(), broken: None },
            );
            let moved = fieldform_scene::relation::rederive(&mut after);
            let mut saved = Vec::new();
            for (curve, index, to) in moved {
                let Some(c) = self.doc.curve_mut(curve) else { continue };
                let Some(was) = c.control_points().get(index).copied() else { continue };
                c.set_control_point(index, to);
                saved.push((curve, index, was));
            }
            self.join_preview = Some(JoinPreview { relation, saved });
        }
        self.rebuild_curve_display();
        self.rebuild_render_resources();
    }

    /// Puts the points a preview moved back where they were, without redrawing.
    fn put_back_join_preview(&mut self) -> bool {
        let Some(preview) = self.join_preview.take() else { return false };
        // In reverse, so a point moved twice ends where it started.
        for (curve, index, was) in preview.saved.into_iter().rev() {
            if let Some(c) = self.doc.curve_mut(curve) {
                c.set_control_point(index, was);
            }
        }
        true
    }

    /// Ends a join's preview, if one is up, and the list showing it: the document as it was.
    pub(crate) fn end_join_preview(&mut self) {
        if self.put_back_join_preview() {
            self.rebuild_curve_display();
            self.rebuild_render_resources();
        }
        if self.marking.as_ref().is_some_and(|m| m.lead_for.is_some()) {
            self.marking = None;
        }
    }

    /// The list G1 or G2 opens on a join: one row per curve, the curve that leads. Hovering a
    /// row shows what it does; a click keeps it; Esc, or a click anywhere else, puts it back.
    ///
    /// (The owner, 2026-11-07: "a list / drop down of each curve and as u hover over one it
    /// automatically changes so u can see the effect of your choice in real time".)
    fn lead_list_ui(&mut self, ctx: &egui::Context, menu: MarkingMenu) {
        let (Some(join), Some(continuity)) = (menu.join, menu.lead_for) else { return };
        // An undo, or anything else, may have taken the join away under the list.
        if join_of(&self.doc, join[0]).map(|j| j.ends) != Some(join) {
            self.end_join_preview();
            return;
        }
        let [a, b] = join;
        let existing = joint_relation_between(&self.doc, a, b);
        let replace = existing.as_ref().map(|(id, _)| *id);
        let name = |id: CurveId| self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
        let g = match continuity {
            fieldform_scene::JointContinuity::G1 => 1,
            fieldform_scene::JointContinuity::G2 => 2,
        };
        let rows: Vec<(String, fieldform_scene::Relation, Option<String>, bool)> = [(a, b), (b, a)]
            .into_iter()
            .map(|(master, slave)| {
                let relation = fieldform_scene::Relation::Joint { master, slave, continuity };
                let refused = self.joint_refusal(&relation, replace);
                let current = existing.as_ref().is_some_and(|(_, r)| *r == relation);
                let label = format!("{} leads \u{2014} {} follows", name(master.curve), name(slave.curve));
                (label, relation, refused, current)
            })
            .collect();
        let mut hovered = None;
        let mut picked = None;
        let shown = egui::Area::new(egui::Id::new("join lead list"))
            .order(egui::Order::Foreground)
            .fixed_pos(menu.at)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(format!("G{g}: which curve leads?")).strong());
                    for (label, relation, refused, current) in &rows {
                        let text = if *current { format!("\u{2713} {label}") } else { label.clone() };
                        let response = ui.add_enabled(refused.is_none(), egui::Button::selectable(*current, text));
                        // **The reason under the row, not only on hover**: two greyed rows and nothing
                        // else read as a menu that does not work, which is how it was reported.
                        let response = match refused {
                            Some(why) => {
                                let response = response.on_disabled_hover_text(why.as_str());
                                ui.label(
                                    egui::RichText::new(why.as_str()).small().color(ui.visuals().warn_fg_color),
                                );
                                response
                            }
                            None => response,
                        };
                        if refused.is_none() && response.hovered() {
                            hovered = Some(relation.clone());
                        }
                        if response.clicked() {
                            picked = Some(relation.clone());
                        }
                    }
                    ui.label(egui::RichText::new("Hover to see it \u{b7} click to keep \u{b7} Esc to leave it").weak().small());
                });
            });
        let away = ctx.input(|i| i.pointer.any_pressed()) && !shown.response.contains_pointer();
        let close = ctx.input(|i| i.key_pressed(egui::Key::Escape)) || away;
        if let Some(fieldform_scene::Relation::Joint { master, .. }) = picked {
            self.put_back_join_preview();
            let warned = self.console_warnings;
            self.run_join_line(join, g, Some(master.curve));
            // Kept, it was redrawn by the edit; refused, nothing redrew the points put back.
            if self.console_warnings > warned {
                self.rebuild_curve_display();
                self.rebuild_render_resources();
            }
            return;
        }
        if close {
            self.end_join_preview();
            return;
        }
        self.preview_join(hovered, replace);
        self.marking = Some(menu);
    }

    /// A G picked on a join's ring, run as the line that says it, so the console echoes a line
    /// that replays; directly where no line could name the join.
    fn pick_join_continuity(&mut self, join: [fieldform_scene::CurveEnd; 2], command: &'static commands::Command) {
        if command.id == "curve.continuity.break" {
            match self.continuity_line(join, 0, None) {
                Some(line) => self.run_echoed(line.replace(" g=0", " -break")),
                None => self.break_join(join[0]),
            }
            return;
        }
        let g: u8 = match command.id {
            "curve.continuity.0" => 0,
            "curve.continuity.1" => 1,
            _ => 2,
        };
        self.run_join_line(join, g, None);
    }

    /// Sets a join's continuity as the `continuity` line that says it, echoed as one.
    fn run_join_line(&mut self, join: [fieldform_scene::CurveEnd; 2], g: u8, lead: Option<CurveId>) {
        match self.continuity_line(join, g, lead) {
            Some(line) => self.run_echoed(line),
            None => {
                let continuity = match g {
                    0 => None,
                    1 => Some(fieldform_scene::JointContinuity::G1),
                    _ => Some(fieldform_scene::JointContinuity::G2),
                };
                self.set_join_continuity(join[0], continuity, lead);
            }
        }
    }

    /// A line run as a menu pick is written, as the line that does it: after it ran, and only if
    /// it changed something without a refusal, since a refused line replays as a refusal.
    fn run_echoed(&mut self, line: String) {
        self.commit_session();
        self.echo_settle();
        let (moves, warned) = (self.document_moves, self.console_warnings);
        self.run_command_unechoed(&line);
        if self.document_moves != moves && self.console_warnings == warned {
            self.echo_line(line);
        }
        self.echo_rebase();
    }

    /// Which slot of a join's ring holds what is there now, to mark it: none for two ends that
    /// only touch, since nothing holds them.
    fn join_current_slot(&self, join: [fieldform_scene::CurveEnd; 2]) -> Option<&'static str> {
        match joint_relation_between(&self.doc, join[0], join[1]) {
            Some((_, fieldform_scene::Relation::Joint { continuity: fieldform_scene::JointContinuity::G1, .. })) => {
                Some("curve.continuity.1")
            }
            Some((_, fieldform_scene::Relation::Joint { continuity: fieldform_scene::JointContinuity::G2, .. })) => {
                Some("curve.continuity.2")
            }
            _ if self.join_is_held(join) => Some("curve.continuity.0"),
            _ => None,
        }
    }

    /// Whether a join's two ends sit on one point.
    fn join_is_held(&self, join: [fieldform_scene::CurveEnd; 2]) -> bool {
        let on = |e: fieldform_scene::CurveEnd| {
            let c = self.doc.curve(e.curve)?;
            let n = c.control_points().len();
            c.placed_on(if e.end == fieldform_scene::End::Start { 0 } else { n.checked_sub(1)? })
        };
        on(join[0]).is_some() && on(join[0]) == on(join[1])
    }

    /// Degree…, asked for from the palette or a key rather than from the ring: the ring of degrees
    /// opens at the middle of the viewport, where a menu with no click to anchor it goes.
    pub(crate) fn open_degree_menu(&mut self) {
        let at = self.viewport_rect.map_or(egui::pos2(300.0, 300.0), |r| r.center());
        self.marking = Some(MarkingMenu { at, subject: Subject::Degree, dragging: false, join: None, lead_for: None });
    }

    /// Draws the marking menu, if it is open, and runs what is picked.
    ///
    /// **One foreground widget over the whole menu**, so egui gives the menu its own clicks and
    /// the viewport underneath sees none of them.
    pub(crate) fn marking_ui(&mut self, ctx: &egui::Context) {
        let Some(mut menu) = self.marking.take() else { return };
        if menu.lead_for.is_some() {
            self.lead_list_ui(ctx, menu);
            return;
        }
        let ctx_now = self.command_context();
        let table = marking::slots(menu.subject);
        // What each slot would run, and whether it may. A join's ring is about the join it was
        // opened on, not the selection its entries ask for when run from anywhere else.
        let on_join = menu.join;
        let current = on_join.and_then(|j| self.join_current_slot(j));
        let entry = |i: usize| {
            table[i].and_then(commands::by_id).map(|c| {
                let ok = if on_join.is_some() { Ok(()) } else { (c.context)(&ctx_now) };
                (c, ok)
            })
        };
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
                            // What holds at the join now is marked, as a checked menu item is.
                            Some((c, ok)) if current == Some(c.id) => (format!("\u{2713} {}", c.display_name), ok.is_ok()),
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
                // Degree… opens its ring where this one was, rather than running anything.
                Some((c, Ok(()))) if c.id == "curve.degree_menu" => {
                    self.marking = Some(MarkingMenu { at: menu.at, subject: Subject::Degree, dragging: false, join: None, lead_for: None });
                }
                // G1 or G2 on a join asks which curve leads next, in a list that shows each.
                Some((c, Ok(()))) if menu.join.is_some() && matches!(c.id, "curve.continuity.1" | "curve.continuity.2") => {
                    let continuity = if c.id == "curve.continuity.1" {
                        fieldform_scene::JointContinuity::G1
                    } else {
                        fieldform_scene::JointContinuity::G2
                    };
                    self.marking = Some(MarkingMenu { lead_for: Some(continuity), ..menu });
                }
                Some((c, Ok(()))) if menu.join.is_some() => self.pick_join_continuity(menu.join.expect("checked"), c),
                Some((c, Ok(()))) => self.invoke_command(c),
                // A dimmed slot does nothing, and the menu stays open on it.
                _ => self.marking = Some(menu),
            },
            None if close => {}
            None => self.marking = Some(menu),
        }
    }
}
