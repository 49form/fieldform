// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The cage tools' shared framework: a tool armed until Esc, its preview on the cage, the one
//! command each click commits, and the keys that mean the cage when cage elements are selected.
//!
//! **No tool mode to get stuck in.** An armed tool is like the curve tool: it stays on after each
//! commit, Esc puts it down, and any other command puts it down first (R7). Each commit is one
//! operation run as its own console line through the same path a typed line takes, so it is one
//! undo step and its own record. The first tool through here is Insert Edge: a hovered edge shows
//! the ring it would insert, dashed, and a click inserts it. Extrude and inset, which drag a
//! working copy, come with their own item and read this same state.

use super::check_model::{ERROR_COLOR, ERROR_FILL};
use crate::overlay_draw::PROPAGATED_COLOR;
use crate::*;
use fieldform_cage::{edit, FaceId, VertexId};

/// An extrude or inset pending: the cap lifted and waiting to be dragged, kept by Enter as one
/// undo step, or taken back by Esc as though nothing had happened (R7).
///
/// **The lift is on the stack from the start**, as its own entry after a mark, and every drag
/// of the cap goes on after it; keeping folds them into one entry, and taking back drops them
/// all and puts the mesh back from the snapshot. That is the session machinery a block's session
/// uses (`CommandStack::mark`, `keep_since`, `cancel_to`), on a cage.
pub(crate) struct CageSession {
    pub(crate) command: &'static commands::Command,
    pub(crate) cage: CageId,
    /// The mesh before the lift, for Esc.
    pub(crate) before: fieldform_cage::MeshSnapshot,
    pub(crate) mark: fieldform_ops::StackMark,
    selection_before: Picked,
    /// The faces lifted: the cap keeps their slots.
    pub(crate) faces: Vec<fieldform_cage::FaceId>,
    /// The lift's numbers as the panel has them -- mode, direction, thickness, depth -- which are
    /// what the kept line says. An extrude's distance is read off the cap instead.
    pub(crate) args: Args,
    /// The cap's vertices after the lift, whose final positions are written after the lift's line.
    pub(crate) cap: Vec<VertexId>,
    /// Where the cap's vertices were when lifted, to tell a zero extrude from a moved one.
    pub(crate) cap_before: Vec<Point3<f64>>,
    /// The cap's average normal in the cage's space: an extrude's default direction.
    pub(crate) normal: Vector3<f64>,
    /// With the direction `vertex_normals`: where each cap vertex goes per millimetre of distance,
    /// aligned with `cap`; empty in every other direction.
    pub(crate) cap_dirs: Vec<Vector3<f64>>,
    /// The floating panel's state (`cage_panel.rs`).
    pub(crate) panel: super::tool_panel::Panel,
}

/// A cage tool, armed.
pub(crate) struct CageTool {
    pub(crate) command: &'static commands::Command,
    /// Where along the hovered edge the pointer is, as a fraction from its first vertex.
    pub(crate) t: f64,
    /// The ring the hovered edge would get, as its new edges, for the preview.
    preview: Vec<(Point3<f64>, Point3<f64>)>,
    /// Why the preview is refused, if it is.
    pub(crate) refused: Option<String>,
    /// A two-click tool's first click: the vertex to weld, or the edges of a bridge's first side.
    /// Shown highlighted until the tool runs.
    first: Vec<Selection>,
    /// A bridge's second side, gathered edge by edge until it has as many as the first.
    second: Vec<Selection>,
    /// The knife's path and preview, read only while the knife is the tool (`knife_tool.rs`).
    pub(crate) knife: super::knife_tool::Knife,
    /// The settings window's state (`click_panel.rs`), shared by every click tool that has rows.
    pub(crate) panel: super::tool_panel::Panel,
    /// Insert Edge: how many evenly spaced rings one click makes.
    pub(crate) count: u32,
    /// Bridge: how many rows of faces the bridge is made of.
    pub(crate) divisions: u32,
    /// Weld: whether the two vertices meet halfway rather than at the second.
    pub(crate) weld_midpoint: bool,
    /// Weld: whether the second click zips the whole open rim, rather than welding the one pair.
    pub(crate) weld_zip: bool,
    /// Insert Edge: how a typed offset is read, and the offset itself while one is set -- it then
    /// overrides the pointer on whichever edge is hovered, from the end nearer the pointer.
    pub(crate) offset_mode: super::knife_tool::OffsetMode,
    pub(crate) offset: Option<f64>,
    /// Insert Edge: the end of the hovered edge an offset is read from, nearer the pointer, drawn
    /// as a tick so the number is unambiguous.
    pub(crate) ref_end: Option<VertexId>,
    /// Insert Edge: the edge a press landed on, held while the pointer drags `t` along it. A drag
    /// is read against this edge wherever the pointer goes, and a release commits it.
    pub(crate) drag: Option<(VertexId, VertexId)>,
    pub(crate) dragged: bool,
}

impl CageTool {
    pub(crate) fn new(command: &'static commands::Command) -> Self {
        CageTool {
            command,
            t: 0.5,
            preview: Vec::new(),
            refused: None,
            first: Vec::new(),
            second: Vec::new(),
            knife: Default::default(),
            panel: Default::default(),
            count: 1,
            divisions: 1,
            weld_midpoint: false,
            weld_zip: false,
            offset_mode: Default::default(),
            offset: None,
            ref_end: None,
            drag: None,
            dragged: false,
        }
    }
}

impl App {
    /// Arms `command`. The selection stays: a tool reads it, and a click adds to or replaces it
    /// as always.
    pub(crate) fn arm_cage_tool(&mut self, command: &'static commands::Command) {
        if self.control != ControlMesh::Cage {
            self.warn(format!("{} works on the cage, and a surface is selected", command.display_name));
            return;
        }
        self.cancel_curve_placement_quietly();
        // Extrude and inset are sessions on the selected faces, not click tools.
        if matches!(command.op, "cage.extrude" | "cage.inset") {
            if !self.view_mode.cage_is_editable() {
                self.view_mode = ViewMode::Overlay;
            }
            self.begin_cage_session(command);
            return;
        }
        // A selection that already says everything the tool asks runs it at once: two vertices
        // for a weld, two rims for a bridge. The keyboard route, beside the clicks.
        let ready = match command.op {
            "cage.weld" => self.selection.parts().iter().filter(|s| matches!(s, Selection::Vertex(_))).count() == 2,
            "cage.bridge" => {
                let args = self.cage_selection_args(command.op, &Args::new());
                args.get("a").is_some() && args.get("b").is_some()
            }
            _ => false,
        };
        if ready {
            self.dispatch(command.op, &Args::new());
            return;
        }
        self.cage_tool = Some(CageTool::new(command));
        // Field mode draws no cage to aim at: said, rather than left to a hover that never lands.
        if !self.view_mode.cage_is_editable() {
            self.view_mode = ViewMode::Overlay;
            self.say(format!("{}: {} (switched to Field + Cage so the cage can be picked)", command.display_name, tool_hint(command)));
            return;
        }
        self.say(format!("{}: {}", command.display_name, tool_hint(command)));
    }

    /// Puts the tool down, keeping nothing: a tool has nothing pending between clicks.
    pub(crate) fn disarm_cage_tool(&mut self) -> bool {
        self.cage_tool.take().is_some()
    }

    /// Begins an extrude or inset on the selected faces: the lift at nought (or the inset at a
    /// tenth of its shortest border edge), the cap selected so the manipulator sits on it with an
    /// arrow along its normal, and the session pending until Enter or Esc.
    pub(crate) fn begin_cage_session(&mut self, command: &'static commands::Command) {
        // A session already pending is kept first: E again chains from the cap it made.
        self.keep_cage_session();
        let faces: Vec<fieldform_cage::FaceId> =
            self.selection.parts().iter().filter_map(|s| if let Selection::Face(f) = s { Some(*f) } else { None }).collect();
        if faces.is_empty() {
            self.warn(format!("{}: select the faces first", command.display_name));
            return;
        }
        let cage = self.active_cage;
        let Some(body) = self.doc.cage_body(cage) else { return };
        let mesh = body.mesh();
        let face_text = faces.iter().map(|f| f.index().to_string()).collect::<Vec<_>>().join(",");
        let args = match command.op {
            "cage.extrude" => Args::new().with("faces", Value::Str(face_text.clone())).with("distance", Value::Float(0.0)),
            _ => {
                // A tenth of the shortest edge of the faces: visible, and inside every face.
                let shortest = faces
                    .iter()
                    .flat_map(|&f| {
                        let c = mesh.face_vertices(f);
                        (0..4).map(move |i| (c[i], c[(i + 1) % 4]))
                    })
                    .map(|(a, b)| (mesh.vertex_position(a) - mesh.vertex_position(b)).norm())
                    .fold(f64::INFINITY, f64::min);
                let thickness = shortest * 0.1;
                Args::new().with("faces", Value::Str(face_text.clone())).with("thickness", Value::Float(thickness))
            }
        };
        let before = fieldform_cage::MeshSnapshot::of(mesh);
        let selection_before = self.selection.clone();
        let mark = self.stack.mark();
        let moves = self.document_moves;
        // The session is pending *before* the lift lands, because the lift's walls have no
        // height yet and the body reads the pending session to know to draw them as a draft
        // (`cages_changed`) rather than refuse them as collapsed. The cap is filled in after.
        self.cage_session = Some(CageSession {
            command,
            cage,
            before,
            mark,
            selection_before,
            faces,
            args: args.clone(),
            cap: Vec::new(),
            cap_before: Vec::new(),
            normal: Vector3::z(),
            cap_dirs: Vec::new(),
            panel: Default::default(),
        });
        // The lift, dispatched rather than typed: its line is written when the session is kept,
        // not now, so nothing is recorded that Esc would have to unsay.
        self.dispatch(command.op, &args);
        if self.document_moves == moves
            && let Some(session) = self.cage_session.take()
        {
            self.stack.cancel_to(session.mark);
            return;
        }
        // The cap keeps the faces' slots, so the selection it had is the cap now.
        self.read_cap();
        self.say(format!(
            "{}: drag the cap with the arrow (or turn and scale it), or type its numbers in the panel; Enter keeps, Esc puts it back{}",
            command.display_name,
            if command.op == "cage.extrude" { ", E extrudes again from the new cap" } else { "" }
        ));
    }

    /// Keeps the pending extrude or inset as one undo step, and writes its lines: the lift, then
    /// where the cap ended up. A cap that never moved is an extrude of nothing, and is taken back
    /// instead -- no undo entry, no hidden zero-height walls.
    pub(crate) fn keep_cage_session(&mut self) {
        let Some(session) = self.cage_session.take() else { return };
        let mesh = self.cage();
        let after: Vec<Point3<f64>> = session.cap.iter().map(|&v| mesh.vertex_position(v)).collect();
        let eps = fieldform_cage::validate::epsilon_length(mesh);
        let moved = session.cap_before.iter().zip(&after).any(|(b, a)| (a - b).norm() > eps);
        if session.command.op == "cage.extrude" && !moved {
            self.cage_session = Some(session);
            self.cancel_cage_session();
            self.say("extrude cancelled \u{2014} no distance");
            return;
        }
        // The lift's line says its numbers: an extrude pulled straight says its distance and
        // nothing more; a cap also turned or scaled, or an inset's cap moved, says where it
        // ended up after the lift's line.
        let (line, positions) = if session.command.op == "cage.extrude" {
            let d = self.extrude_distance(&session);
            if self.extrude_is_straight(&session, d) {
                (self.cage_session_line(&session, Some(d)), false)
            } else {
                (self.cage_session_line(&session, Some(0.0)), true)
            }
        } else {
            (self.cage_session_line(&session, None), moved)
        };
        self.stack.keep_since(session.mark, session.command.display_name);
        self.gesture = None;
        self.last_command = Some(session.command);
        // Kept, the cage is judged again: a cap pulled to a height is a solid, and one an inset
        // folded or a drag put back at nought is red, said where the faults are drawn.
        let was_red = self.doc.cage_is_red(session.cage);
        if let Some(body) = self.doc.cage_body_mut(session.cage) {
            body.end_draft();
        }
        if self.doc.cage_is_red(session.cage) != was_red {
            self.rebuild_render_resources();
        }
        self.echo_line(line);
        if positions {
            let vertices: Vec<String> = session.cap.iter().map(|v| v.index().to_string()).collect();
            let to: Vec<String> = after.iter().map(|p| format!("{},{},{}", echo::number(p.x), echo::number(p.y), echo::number(p.z))).collect();
            self.echo_line(format!("setpositions vertices={} to={}", vertices.join(","), to.join(";")));
        }
        self.echo_rebase();
        self.say(format!("{} kept", session.command.display_name));
    }

    /// Takes the pending extrude or inset back: the history and the mesh as they were.
    pub(crate) fn cancel_cage_session(&mut self) {
        let Some(session) = self.cage_session.take() else { return };
        self.stack.cancel_to(session.mark);
        if let Some(body) = self.doc.cage_body_mut(session.cage) {
            body.set_mesh(session.before.restore());
        }
        self.gesture = None;
        self.gizmo_drag = None;
        self.move_state = None;
        self.selection = session.selection_before;
        self.after_document_change();
        self.echo_base.moves = self.document_moves;
        self.say(format!("{} taken back", session.command.display_name));
    }


    fn cancel_curve_placement_quietly(&mut self) {
        if self.pending_curve.is_some() {
            self.cancel_curve_placement();
        }
    }

    /// Reads the pointer against the hovered edge and refreshes the preview. Once a frame while a
    /// tool is armed.
    pub(crate) fn cage_tool_hover(&mut self, aspect: f32, rect: egui::Rect, pointer: Option<egui::Pos2>, snap_half: bool, ctrl: bool, alt: bool) {
        let Some(op) = self.cage_tool.as_ref().map(|t| t.command.op) else { return };
        if let Some(tool) = self.cage_tool.as_mut() {
            tool.preview.clear();
            tool.refused = None;
        }
        // The knife reads faces and vertices as well as edges, and finds its own point.
        if op == "cage.cut" {
            self.knife_hover(aspect, rect, pointer, snap_half, ctrl, alt);
            return;
        }
        // A press on an edge holds it while the pointer drags `t` along it, wherever the pointer goes.
        let dragged_edge = self.cage_tool.as_ref().and_then(|t| t.drag);
        let (a, b) = match (dragged_edge, self.hover) {
            (Some(edge), _) => edge,
            (None, Selection::Edge(a, b)) => (a, b),
            _ => {
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.ref_end = None;
                }
                return;
            }
        };
        let Some(pos) = pointer else { return };
        let (count, mode, offset) = self.cage_tool.as_ref().map_or((1, Default::default(), None), |t| (t.count, t.offset_mode, t.offset));
        // Everything read from the mesh first, so the tool can then be written to.
        let (t, preview, refused, ref_end) = {
            let mesh = self.cage();
            let placement = self.control_placement();
            let pa = self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, a));
            let pb = self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, b));
            let t = match (pa, pb) {
                (Some(pa), Some(pb)) if (pb - pa).length_sq() > 1.0 => {
                    let d = pb - pa;
                    (((pos - pa).dot(d)) / d.length_sq()).clamp(0.02, 0.98) as f64
                }
                _ => 0.5,
            };
            // The end an offset is measured from: the one nearer the pointer.
            let ref_end = match (pa, pb) {
                (Some(pa), Some(pb)) if (pb - pos).length() < (pa - pos).length() => b,
                _ => a,
            };
            let t = if snap_half { 0.5 } else { t };
            // A typed offset overrides the pointer; several rings are evenly spaced and ignore it.
            let t = match offset {
                Some(value) if count == 1 && !snap_half => super::knife_tool::t_from_offset(mesh, (a, b), ref_end, mode, value),
                _ => t,
            };
            let mut preview = Vec::new();
            let mut refused = None;
            if op == "cage.insert_ring" {
                match edit::insert_rings(mesh, (a, b), t, count as usize) {
                    Ok(result) => preview = result.ring_edges,
                    Err(e) => refused = Some(e.reason),
                }
            }
            (t, preview, refused, ref_end)
        };
        if let Some(tool) = self.cage_tool.as_mut() {
            tool.t = t;
            tool.preview = preview;
            tool.refused = refused;
            tool.ref_end = Some(ref_end);
        }
    }

    /// The hovered edge's `t` as the line says it: a fraction from its first vertex.
    pub(crate) fn ring_line(&self) -> Option<String> {
        let tool = self.cage_tool.as_ref()?;
        let (a, b) = match (tool.drag, self.hover) {
            (Some(edge), _) => edge,
            (None, Selection::Edge(a, b)) => (a, b),
            _ => return None,
        };
        let count = if tool.count > 1 { format!(" count={}", tool.count) } else { String::new() };
        Some(format!("insertring edge={}-{} t={}{count}", a.index(), b.index(), echo::number(tool.t)))
    }

    /// A click on the cage while a tool is armed: the tool's commit, as its own console line.
    /// `true` when the click was the tool's and not a selection.
    pub(crate) fn cage_tool_click(&mut self, shift: bool, pos: Option<egui::Pos2>, aspect: f32, rect: egui::Rect) -> bool {
        let Some(tool) = self.cage_tool.as_ref() else { return false };
        if tool.command.op == "cage.cut" {
            return self.knife_click(shift, pos, aspect, rect);
        }
        if tool.refused.is_some() {
            let why = tool.refused.clone().unwrap_or_default();
            self.warn(why);
            return true;
        }
        let op = tool.command.op;
        let edge_text = |a: VertexId, b: VertexId| format!("{}-{}", a.index(), b.index());
        let line = match (op, self.hover) {
            ("cage.insert_ring", Selection::Edge(..)) => match self.ring_line() {
                Some(line) => line,
                None => return false,
            },
            // **Weld: the vertex to move, then the vertex it moves onto.** Shift on the second
            // meets halfway.
            ("cage.weld", Selection::Vertex(v)) => match tool.first.first() {
                None => {
                    self.cage_tool.as_mut().expect("armed").first = vec![Selection::Vertex(v)];
                    self.say("Weld: now click the vertex it moves onto (Shift to meet halfway)");
                    return true;
                }
                Some(&Selection::Vertex(from)) if from == v => return true,
                Some(&Selection::Vertex(from)) => {
                    let at = if shift || tool.weld_midpoint { " at=midpoint" } else { "" };
                    let how = if tool.weld_zip { "zip" } else { "pairs" };
                    format!("weld {how}={}{at}", edge_text(from, v))
                }
                _ => return true,
            },
            // **Bridge: one edge, then the other** (the owner, 2026-10-24: edge to edge, not rim
            // to rim). A click starts a side with that edge; Shift adds the clicked edge to the
            // side being gathered; a plain click once the first side has its edges starts the
            // second; and the bridge runs the moment the second side has as many edges as the
            // first. So edge → edge is two clicks, and a hole's two facing edges fill it.
            ("cage.bridge", Selection::Edge(a, b)) => {
                let same = |s: &Selection| matches!(*s, Selection::Edge(p, q) if (p == a && q == b) || (p == b && q == a));
                let edge = Selection::Edge(a, b);
                let tool = self.cage_tool.as_mut().expect("armed");
                let (first_n, second_n) = (tool.first.len(), tool.second.len());
                if tool.first.iter().any(same) || tool.second.iter().any(same) {
                    self.warn("That edge is on a side already");
                    return true;
                }
                let gathering_second = second_n > 0 || (!shift && first_n > 0);
                if gathering_second {
                    tool.second.push(edge);
                } else {
                    tool.first.push(edge);
                }
                let (first_n, second_n) = (tool.first.len(), tool.second.len());
                if second_n == 0 {
                    self.say(format!(
                        "Bridge: {first_n} edge{} on the first side; Shift-click to add another, or click an edge of the other side",
                        if first_n == 1 { "" } else { "s" }
                    ));
                    return true;
                }
                if second_n < first_n {
                    self.say(format!("Bridge: {second_n} of {first_n} edges on the second side; Shift-click the next"));
                    return true;
                }
                let edges_of = |side: &[Selection]| -> Vec<(VertexId, VertexId)> {
                    side.iter().filter_map(|s| if let Selection::Edge(p, q) = s { Some((*p, *q)) } else { None }).collect()
                };
                let text = |c: &[(VertexId, VertexId)]| c.iter().map(|&(p, q)| edge_text(p, q)).collect::<Vec<_>>().join(",");
                let divisions = if tool.divisions > 1 { format!(" divisions={}", tool.divisions) } else { String::new() };
                format!("bridge a={} b={}{divisions}", text(&edges_of(&tool.first)), text(&edges_of(&tool.second)))
            }
            _ => return false,
        };
        self.run_command(&line);
        if let Some(tool) = self.cage_tool.as_mut() {
            tool.first.clear();
            tool.second.clear();
        }
        // What the tool made is not what was picked before it: the selection starts over.
        self.selection.clear();
        true
    }

    /// The tool's preview: the ring it would insert, dashed cyan, as the propagated part of a cut
    /// is drawn -- the edges the person did not point at.
    pub(crate) fn draw_cage_tool_preview(&self, painter: &egui::Painter, aspect: f32, rect: egui::Rect) {
        let Some(tool) = self.cage_tool.as_ref() else { return };
        if tool.command.op == "cage.cut" {
            self.draw_knife(painter, aspect, rect);
            return;
        }
        let placement = self.control_placement();
        let stroke = egui::Stroke::new(2.0_f32, PROPAGATED_COLOR);
        // Insert Edge: a tick across the end a typed offset is measured from, so the number is
        // unambiguous (`cage-knife-spec.md` §3, the same mark the knife draws).
        if tool.command.op == "cage.insert_ring"
            && let Some(end) = tool.ref_end
            && let Some((a, b)) = tool.drag.or(if let Selection::Edge(a, b) = self.hover { Some((a, b)) } else { None })
        {
            let mesh = self.cage();
            let other = if end == a { b } else { a };
            if let (Some(s), Some(o)) = (
                self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, end)),
                self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, other)),
            ) {
                let dir = (o - s).normalized();
                let perp = egui::vec2(-dir.y, dir.x) * 6.0;
                painter.line_segment([s - perp, s + perp], egui::Stroke::new(2.0_f32, crate::overlay_draw::PREVIEW_COLOR));
            }
        }
        for (p, q) in &tool.preview {
            let to_screen = |p: &Point3<f64>| {
                let w = placement.transform_point(p);
                Point3::new(w.x as f32, w.y as f32, w.z as f32)
            };
            draw_dashed_polyline(painter, &self.camera, aspect, rect, &[to_screen(p), to_screen(q)], stroke);
        }
        // A two-click tool's sides, held bright until it runs.
        for &part in tool.first.iter().chain(&tool.second) {
            draw_highlight(painter, self.cage(), &placement, &self.camera, aspect, rect, part, SELECTION_COLOR, SELECTION_FILL, 3.0);
        }
        if let (Some(why), Selection::Edge(a, b)) = (&tool.refused, self.hover) {
            let mesh = self.cage();
            draw_highlight(painter, mesh, &placement, &self.camera, aspect, rect, Selection::Edge(a, b), ERROR_COLOR, ERROR_FILL, 3.0);
            if let Some(at) = self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, a)) {
                painter.text(at + egui::vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, why, egui::FontId::proportional(12.0), ERROR_COLOR);
            }
        }
    }

    /// Whether the armed tool has a settings window: the knife, Insert Edge, Bridge and Weld.
    pub(crate) fn cage_tool_has_panel(&self) -> bool {
        self.cage_tool.as_ref().is_some_and(|t| matches!(t.command.op, "cage.cut" | "cage.insert_ring" | "cage.bridge" | "cage.weld"))
    }

    /// The armed tool's rows, in order.
    pub(crate) fn cage_tool_panel_rows(&self) -> Vec<super::tool_panel::Row> {
        use super::tool_panel::Row;
        let Some(tool) = self.cage_tool.as_ref() else { return Vec::new() };
        match tool.command.op {
            "cage.cut" => self.knife_panel_rows(),
            "cage.insert_ring" => vec![
                Row { key: "count", text: tool.count.to_string(), unit: commands::Unit::Count },
                Row { key: "offset_mode", text: tool.offset_mode.word().to_string(), unit: commands::Unit::Word },
                Row {
                    key: "offset",
                    text: tool.offset.map_or_else(|| "\u{2014}".to_string(), echo::number),
                    unit: if tool.offset_mode == super::knife_tool::OffsetMode::Mm { commands::Unit::Mm } else { commands::Unit::Plain },
                },
            ],
            "cage.bridge" => vec![Row { key: "divisions", text: tool.divisions.to_string(), unit: commands::Unit::Count }],
            "cage.weld" => vec![
                Row { key: "position", text: if tool.weld_midpoint { "midpoint" } else { "target" }.to_string(), unit: commands::Unit::Word },
                Row { key: "boundary", text: if tool.weld_zip { "zip" } else { "off" }.to_string(), unit: commands::Unit::Word },
            ],
            _ => Vec::new(),
        }
    }

    /// Rows that mean nothing just now, greyed in the window: the knife's, while a Cut Through
    /// line is placed; Insert Edge's offset, when several evenly spaced rings are asked for.
    pub(crate) fn cage_tool_row_disabled(&self, key: &str) -> bool {
        let Some(tool) = self.cage_tool.as_ref() else { return false };
        match tool.command.op {
            "cage.cut" => tool.knife.through.is_some() && key != "cut_through_faces",
            "cage.insert_ring" => tool.count > 1 && matches!(key, "offset_mode" | "offset"),
            _ => false,
        }
    }

    /// Confirms a row of the armed tool's window.
    pub(crate) fn cage_tool_panel_set(&mut self, key: &'static str, text: &str) -> Result<(), String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        let whole = |max: f64| -> Result<u32, String> {
            let v: f64 = text.parse().map_err(|_| format!("{text:?} is not a number"))?;
            if !v.is_finite() || v < 1.0 || v > max {
                return Err(format!("a whole number from 1 to {max}"));
            }
            Ok(v.round() as u32)
        };
        let op = self.cage_tool.as_ref().map(|t| t.command.op).unwrap_or("");
        if op == "cage.cut" {
            return self.knife_panel_set(key, text);
        }
        match (op, key) {
            ("cage.insert_ring", "count") => {
                let n = whole(64.0)?;
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.count = n;
                }
            }
            ("cage.insert_ring", "offset_mode") => {
                let Some(mode) = super::knife_tool::OffsetMode::from_word(&text.to_ascii_lowercase()) else {
                    return Err("offset mode is one of mm, fraction".to_string());
                };
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.offset_mode = mode;
                }
            }
            ("cage.insert_ring", "offset") => {
                let value = if matches!(text.to_ascii_lowercase().as_str(), "off" | "none" | "auto") {
                    None
                } else {
                    let v: f64 = text.parse().map_err(|_| format!("{text:?} is not a number (or `off`)"))?;
                    if !v.is_finite() || v <= 0.0 {
                        return Err("an offset is more than nought".to_string());
                    }
                    Some(v)
                };
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.offset = value;
                }
            }
            ("cage.bridge", "divisions") => {
                let n = whole(256.0)?;
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.divisions = n;
                }
            }
            ("cage.weld", "position") => {
                let midpoint = match text.to_ascii_lowercase().as_str() {
                    "target" => false,
                    "midpoint" => true,
                    _ => return Err("position is one of target, midpoint".to_string()),
                };
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.weld_midpoint = midpoint;
                }
            }
            ("cage.weld", "boundary") => {
                let zip = match text.to_ascii_lowercase().as_str() {
                    "off" => false,
                    "zip" => true,
                    _ => return Err("boundary is one of off, zip".to_string()),
                };
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.weld_zip = zip;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Typing with no click while a tool with a window is armed.
    pub(crate) fn cage_tool_panel_typed(&mut self, ctx: &egui::Context) -> bool {
        if !self.cage_tool_has_panel() {
            return false;
        }
        let rows = self.cage_tool_panel_rows();
        let Some(tool) = self.cage_tool.as_mut() else { return false };
        super::click_panel::typed(ctx, &mut tool.panel, &rows)
    }

    /// Tab and Shift+Tab with a tool window up.
    pub(crate) fn cage_tool_panel_tab(&mut self, back: bool) {
        let rows = self.cage_tool_panel_rows();
        let ctx = self.egui_ctx.clone();
        let Some(tool) = self.cage_tool.as_mut() else { return };
        let confirmed = super::click_panel::tab(&ctx, &mut tool.panel, &rows, back);
        if let Some((key, text)) = confirmed {
            let result = self.cage_tool_panel_set(key, &text);
            if let Some(tool) = self.cage_tool.as_mut() {
                tool.panel.error = result.err();
            }
        }
    }

    /// Draws the armed tool's window.
    pub(crate) fn cage_tool_panel(&mut self, ctx: &egui::Context) {
        if !self.cage_tool_has_panel() {
            return;
        }
        let rows = self.cage_tool_panel_rows();
        let off: Vec<&'static str> = rows.iter().map(|r| r.key).filter(|k| self.cage_tool_row_disabled(k)).collect();
        let Some(tool) = self.cage_tool.as_mut() else { return };
        let title = tool.command.display_name;
        let footer = match tool.command.op {
            "cage.cut" => "Enter cuts \u{b7} Esc discards \u{b7} Tab next",
            "cage.insert_ring" => "click inserts \u{b7} drag slides \u{b7} type an offset \u{b7} Esc stops",
            _ => "Tab next \u{b7} Esc stops",
        };
        let mut panel = std::mem::take(&mut tool.panel);
        let change = super::click_panel::window(ctx, title, &rows, &|k| off.contains(&k), &mut panel, footer);
        if let Some(tool) = self.cage_tool.as_mut() {
            tool.panel = panel;
        }
        if let Some((key, text)) = change {
            let result = self.cage_tool_panel_set(key, &text);
            if let Some(tool) = self.cage_tool.as_mut() {
                tool.panel.error = result.err();
            }
        }
    }

    /// A press while a tool is armed: `true` when it landed on something the tool drags -- the
    /// knife's active point, or the edge Insert Edge would slide a ring along -- so the camera
    /// does not take the drag.
    pub(crate) fn cage_tool_press(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) -> bool {
        let Some(op) = self.cage_tool.as_ref().map(|t| t.command.op) else { return false };
        match op {
            "cage.cut" => self.knife_active_point_near(aspect, rect, pos),
            "cage.insert_ring" => {
                let Selection::Edge(a, b) = self.hover else { return false };
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.drag = Some((a, b));
                    tool.dragged = false;
                }
                true
            }
            _ => false,
        }
    }

    /// The pointer moved with that press still down.
    pub(crate) fn cage_tool_drag(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        match self.cage_tool.as_ref().map(|t| t.command.op) {
            Some("cage.cut") => self.knife_drag_active(aspect, rect, pos),
            Some("cage.insert_ring") => {
                if let Some(tool) = self.cage_tool.as_mut() {
                    tool.dragged = true;
                }
            }
            _ => {}
        }
    }

    /// The button came up: a ring dragged into place is inserted where it was left; a press that
    /// never became a drag was a click, which has already inserted it.
    pub(crate) fn cage_tool_release(&mut self) {
        let Some(tool) = self.cage_tool.as_ref() else { return };
        if tool.command.op != "cage.insert_ring" || tool.drag.is_none() {
            return;
        }
        let dragged = tool.dragged;
        let line = if dragged && tool.refused.is_none() { self.ring_line() } else { None };
        if let Some(tool) = self.cage_tool.as_mut() {
            tool.drag = None;
            tool.dragged = false;
        }
        if let Some(line) = line {
            self.run_command(&line);
            self.selection.clear();
        }
    }

    /// `selectloop`, `selectring`, `selectgrow`, `selectshrink`: the selection grown along the
    /// cage's structure (`cage_select`). Loop and ring start from the first picked edge.
    pub(crate) fn select_along(&mut self, op: &str) {
        if self.control != ControlMesh::Cage {
            self.warn("Selecting along the cage works on the cage; a surface is being edited");
            return;
        }
        let picked: Vec<Selection> = self.selection.parts().to_vec();
        let mesh = self.cage();
        let grown = match op {
            "select.grow" => cage_select::grow(mesh, &picked),
            "select.shrink" => cage_select::shrink(mesh, &picked),
            _ => {
                let Some(Selection::Edge(a, b)) = picked.iter().find(|s| matches!(s, Selection::Edge(..))).copied() else {
                    self.warn("Select an edge first");
                    return;
                };
                let along = if op == "select.loop" { cage_select::edge_loop(mesh, (a, b)) } else { cage_select::edge_ring(mesh, (a, b)) };
                let mut all = picked.clone();
                for s in along {
                    if !all.contains(&s) {
                        all.push(s);
                    }
                }
                all
            }
        };
        let n = grown.len();
        self.selection.clear();
        for s in grown {
            self.selection.toggle(s);
        }
        self.say(format!("{n} selected"));
    }

    /// Fills the element arguments a cage operation takes from the selection, where a key or a
    /// menu ran it without any: the faces picked for `deletefaces`, the two rims for `bridge`,
    /// the pair for `weld`, the rim edge for `fillhole`, the edge for `insertring`.
    pub(crate) fn cage_selection_args(&self, op: &str, args: &Args) -> Args {
        let args = args_from_selection(op, args, self.selection.parts());
        // Bridge faces reads two regions, which the mesh tells apart: the selected faces split by
        // shared edges, and exactly two groups are the two sides.
        if op == "cage.bridge_faces" && args.get("a").is_none() && args.get("b").is_none() {
            let faces: Vec<FaceId> = self.selection.parts().iter().filter_map(|s| if let Selection::Face(f) = s { Some(*f) } else { None }).collect();
            let groups = face_groups(self.cage(), &faces);
            if groups.len() == 2 {
                let text = |g: &[FaceId]| g.iter().map(|f| f.index().to_string()).collect::<Vec<_>>().join(",");
                return args.with("a", Value::Str(text(&groups[0]))).with("b", Value::Str(text(&groups[1])));
            }
        }
        args
    }
}

/// The console line a cage verb ran as: its word and every key it was given, in a fixed order,
/// naming the cage only where the document holds more than one -- the echo's convention, so the
/// same edit writes the same line in every run. `None` for an op no command has a word for.
pub(crate) fn cage_console_line(op: &str, args: &Args, several_cages: bool) -> Option<String> {
    let word = crate::commands::by_id(op)?.console_name;
    let mut keys: Vec<(String, String)> = args
        .iter()
        .filter(|(k, _)| *k != "cage" || several_cages)
        .map(|(k, v)| (k.to_string(), v.text()))
        .collect();
    keys.sort();
    let mut line = word.to_string();
    for (k, v) in keys {
        line.push_str(&format!(" {k}={v}"));
    }
    Some(line)
}

/// `faces` grouped into edge-connected regions, each in the order given.
pub(crate) fn face_groups(mesh: &QuadMesh, faces: &[FaceId]) -> Vec<Vec<FaceId>> {
    let mut groups: Vec<Vec<FaceId>> = Vec::new();
    let shares_edge = |f: FaceId, g: FaceId| {
        let (cf, cg) = (mesh.face_vertices(f), mesh.face_vertices(g));
        (0..4).any(|i| {
            let (a, b) = (cf[i], cf[(i + 1) % 4]);
            (0..4).any(|j| (cg[j] == a && cg[(j + 1) % 4] == b) || (cg[j] == b && cg[(j + 1) % 4] == a))
        })
    };
    for &f in faces {
        let touching: Vec<usize> = groups.iter().enumerate().filter(|(_, g)| g.iter().any(|&x| shares_edge(f, x))).map(|(i, _)| i).collect();
        match touching.as_slice() {
            [] => groups.push(vec![f]),
            [i] => groups[*i].push(f),
            [i, rest @ ..] => {
                groups[*i].push(f);
                for &j in rest.iter().rev() {
                    let taken = groups.remove(j);
                    groups[*i].extend(taken);
                }
            }
        }
    }
    groups
}

/// The element arguments a cage operation reads from what is picked, free of the `App` so a test
/// holds every key's spelling: `faces=` for delete, `edges=` for dissolve, `edge=` for fill hole
/// and insert ring, `pairs=` for weld, `a=`/`b=` for bridge. A key already given is kept.
pub(crate) fn args_from_selection(op: &str, args: &Args, picked: &[Selection]) -> Args {
    {
        let faces: Vec<String> = picked.iter().filter_map(|s| if let Selection::Face(f) = s { Some(f.index().to_string()) } else { None }).collect();
        let edges: Vec<(VertexId, VertexId)> = picked.iter().filter_map(|s| if let Selection::Edge(a, b) = s { Some((*a, *b)) } else { None }).collect();
        let vertices: Vec<VertexId> = picked.iter().filter_map(|s| if let Selection::Vertex(v) = s { Some(*v) } else { None }).collect();
        let edge_text = |e: &(VertexId, VertexId)| format!("{}-{}", e.0.index(), e.1.index());
        let mut args = args.clone();
        let missing = |k: &str| args.get(k).is_none();
        match op {
            "cage.delete_faces" if missing("faces") && !faces.is_empty() => {
                args = args.with("faces", Value::Str(faces.join(",")));
            }
            "cage.dissolve_edges" | "cage.crease" | "cage.uncrease" if missing("edges") && !edges.is_empty() => {
                args = args.with("edges", Value::Str(edges.iter().map(edge_text).collect::<Vec<_>>().join(",")));
            }
            "cage.fill_hole" if missing("edge") => {
                if let Some(e) = edges.first() {
                    args = args.with("edge", Value::Str(edge_text(e)));
                }
            }
            "cage.insert_ring" if missing("edge") => {
                if let Some(e) = edges.first() {
                    args = args.with("edge", Value::Str(edge_text(e)));
                }
            }
            "cage.weld" if missing("pairs") && vertices.len() == 2 => {
                args = args.with("pairs", Value::Str(format!("{}-{}", vertices[0].index(), vertices[1].index())));
            }
            "cage.bridge" if missing("a") && missing("b") && edges.len() >= 2 => {
                // Two rims: the selected edges split into connected chains, the first two taken.
                let chains = chains_of(&edges);
                if chains.len() == 2 {
                    let text = |c: &[(VertexId, VertexId)]| c.iter().map(edge_text).collect::<Vec<_>>().join(",");
                    args = args.with("a", Value::Str(text(&chains[0]))).with("b", Value::Str(text(&chains[1])));
                }
            }
            _ => {}
        }
        args
    }
}

/// Edges grouped into chains by shared vertices.
fn chains_of(edges: &[(VertexId, VertexId)]) -> Vec<Vec<(VertexId, VertexId)>> {
    let mut chains: Vec<Vec<(VertexId, VertexId)>> = Vec::new();
    for &e in edges {
        let touching: Vec<usize> = chains
            .iter()
            .enumerate()
            .filter(|(_, c)| c.iter().any(|&(a, b)| a == e.0 || a == e.1 || b == e.0 || b == e.1))
            .map(|(i, _)| i)
            .collect();
        match touching.as_slice() {
            [] => chains.push(vec![e]),
            [i] => chains[*i].push(e),
            [i, rest @ ..] => {
                chains[*i].push(e);
                for &j in rest.iter().rev() {
                    let taken = chains.remove(j);
                    chains[*i].extend(taken);
                }
            }
        }
    }
    chains
}

/// What a tool says when it is armed.
fn tool_hint(command: &commands::Command) -> &'static str {
    match command.op {
        "cage.insert_ring" => "hover an edge and click to insert a ring there; Shift for the midpoint; Esc to stop",
        "cage.weld" => "click the vertex to move, then the vertex it moves onto (Shift to meet halfway); Esc to stop",
        "cage.bridge" => "click an edge, then the edge to bridge it to; Shift-click adds edges to a side; Esc to stop",
        "cage.cut" => "click points on the cage's edges or vertices; Enter cuts, Backspace takes a point back, Esc discards the path",
        _ => "click on the cage; Esc to stop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_cage::FaceId;

    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }

    fn text(args: &Args, key: &str) -> Option<String> {
        args.get(key).map(|v| v.text())
    }

    /// **Every cage verb a key or a menu can run reads its elements from the selection**, under
    /// the key its console form spells. Delete on a selected loop was refused for want of this:
    /// the dissolve verb had no arm here and asked for `edges=` that nothing filled.
    #[test]
    fn every_cage_verb_reads_its_elements_from_the_selection() {
        let loop_ = [Selection::Edge(v(0), v(1)), Selection::Edge(v(1), v(2))];
        let args = args_from_selection("cage.dissolve_edges", &Args::new(), &loop_);
        assert_eq!(text(&args, "edges").as_deref(), Some("0-1,1-2"));
        for op in ["cage.crease", "cage.uncrease"] {
            assert_eq!(text(&args_from_selection(op, &Args::new(), &loop_), "edges").as_deref(), Some("0-1,1-2"), "{op}");
        }

        let faces = [Selection::Face(FaceId::from_index(3)), Selection::Face(FaceId::from_index(5))];
        assert_eq!(text(&args_from_selection("cage.delete_faces", &Args::new(), &faces), "faces").as_deref(), Some("3,5"));

        assert_eq!(text(&args_from_selection("cage.fill_hole", &Args::new(), &loop_), "edge").as_deref(), Some("0-1"));
        assert_eq!(text(&args_from_selection("cage.insert_ring", &Args::new(), &loop_), "edge").as_deref(), Some("0-1"));

        let pair = [Selection::Vertex(v(3)), Selection::Vertex(v(7))];
        assert_eq!(text(&args_from_selection("cage.weld", &Args::new(), &pair), "pairs").as_deref(), Some("3-7"));

        // Two rims: two chains that share no vertex.
        let rims = [
            Selection::Edge(v(0), v(1)),
            Selection::Edge(v(1), v(2)),
            Selection::Edge(v(10), v(11)),
            Selection::Edge(v(11), v(12)),
        ];
        let args = args_from_selection("cage.bridge", &Args::new(), &rims);
        assert_eq!(text(&args, "a").as_deref(), Some("0-1,1-2"));
        assert_eq!(text(&args, "b").as_deref(), Some("10-11,11-12"));

        // A key already given is kept, not overwritten by the selection.
        let given = Args::new().with("edges", Value::Str("5-6".into()));
        assert_eq!(text(&args_from_selection("cage.dissolve_edges", &given, &loop_), "edges").as_deref(), Some("5-6"));
    }

    /// **Selected faces split into their edge-connected regions**, which is how B with faces
    /// selected tells a bridge's two sides apart: a cube's top and bottom are two groups, two
    /// faces sharing an edge are one, and a face touching another only at a corner is its own.
    #[test]
    fn selected_faces_group_by_shared_edges() {
        let cube = fieldform_cage::Cage::cube();
        let f = |i: usize| fieldform_cage::FaceId::from_index(i);
        // -Z and +Z: apart. -Z and -Y: an edge shared.
        assert_eq!(face_groups(cube.mesh(), &[f(0), f(1)]).len(), 2);
        assert_eq!(face_groups(cube.mesh(), &[f(0), f(2)]).len(), 1);
        // Four sides round the cube: one ring; with the top as well, still one group.
        assert_eq!(face_groups(cube.mesh(), &[f(2), f(3), f(4), f(5)]).len(), 1);
        assert_eq!(face_groups(cube.mesh(), &[f(2), f(5), f(0), f(1)]).len(), 1, "joined through the sides");
        assert!(face_groups(cube.mesh(), &[]).is_empty());
    }
}
