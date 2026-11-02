// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Sketches: curves that lie on a plane block and follow it.
//!
//! **A 2D sketch follows its plane; a 3D sketch is free.** A curve drawn while sketching on a plane
//! block (`draw.sketch`, or `on=` from the console) is laid in that plane's own coordinates, and the
//! refresh re-lays it from the plane at the start of every rebuild, so dragging the plane, turning
//! it, sliding it over a surface or changing the body under it moves the sketch and everything built
//! on it. A control point taken off the plane on purpose -- snapped to an end in space, typed there
//! -- makes it a 3D sketch: the record is made Fixed and the console says so, and it never goes back
//! on its own (the owner, 2026-10-31). The arrangement is a point on a surface's: the curve's stored
//! points stay the truth for every reader, the record is the rule that re-derives them, and the one
//! reading of a plane's frame is the kernel's `resolve_frame`.
//!
//! **A changed plane is told from a changed curve by the frame the sketch was last laid in.** The
//! plane stood still and a point is not where it was laid: the person moved it, and its new place
//! across the plane is read off (or it left, and the sketch is 3D). The plane moved: the sketch goes
//! with it. Both at once is the plane's doing.

use super::*;
use fieldform_scene::{Curve, Frame, Laid, Sketch, ON_PLANE_MM};

impl App {
    /// The frame of a plane block in world, as a sketch lays itself in it, or why not.
    pub(crate) fn plane_laid(&self, plane: NodeId) -> Result<Laid, String> {
        if !matches!(self.part_graph().node(plane), Some(Node::Plane { .. })) {
            return Err("the plane it lies on is gone".to_string());
        }
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        let (origin, rotation) =
            fieldform_field::resolve_frame(&self.doc, self.part_graph(), &Frame::From { node: plane }, &numbers)
                .map_err(|_| "the plane it lies on cannot be read".to_string())?;
        Ok(Laid::new(Point3::from(origin), rotation * Vector3::x(), rotation * Vector3::y()))
    }

    /// Whether `curve`'s plane is built on the curve: through what the plane stands on, and through
    /// any other sketch's curve met on the way. Such a sketch has nowhere to be laid.
    fn sketch_reads_itself(&self, curve: CurveId, plane: NodeId) -> bool {
        let graph = self.part_graph();
        let mut seen = std::collections::HashSet::new();
        let mut planes = vec![plane];
        while let Some(p) = planes.pop() {
            if !seen.insert(p) {
                continue;
            }
            for block in graph.subtree_of(p) {
                let Some(node) = graph.node(block) else { continue };
                for read in node.curves() {
                    if read == curve {
                        return true;
                    }
                    if let Some(other) = self.doc.sketches.get(&read).filter(|s| s.live) {
                        planes.push(other.plane);
                    }
                }
            }
        }
        false
    }

    /// The live sketches in the order they must be laid: one whose plane stands on another sketch's
    /// curve comes after that sketch.
    fn sketch_order(&self) -> Vec<CurveId> {
        let graph = self.part_graph();
        let mut out: Vec<CurveId> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let ids: Vec<CurveId> = self.doc.sketches.keys().copied().collect();
        for id in &ids {
            let mut stack = vec![(*id, false)];
            while let Some((cur, expanded)) = stack.pop() {
                if expanded {
                    out.push(cur);
                    continue;
                }
                if !seen.insert(cur) {
                    continue;
                }
                stack.push((cur, true));
                let Some(sketch) = self.doc.sketches.get(&cur) else { continue };
                let mut below: Vec<CurveId> = graph
                    .subtree_of(sketch.plane)
                    .into_iter()
                    .filter_map(|b| graph.node(b))
                    .flat_map(|n| n.curves())
                    .filter(|c| *c != cur && self.doc.sketches.contains_key(c))
                    .collect();
                below.sort_by_key(|c| c.raw());
                below.dedup();
                for b in below {
                    stack.push((b, false));
                }
            }
        }
        out
    }

    /// Lays every live sketch on its plane, writing the curves that moved. Answers whether any did.
    ///
    /// **At the start of a rebuild, after the points on surfaces**, since a plane may stand on one:
    /// a curve's points are what an extrude compiles from, so they have to be laid before the
    /// compile reads them. Nothing here calls the rebuild.
    pub(crate) fn refresh_sketches(&mut self) -> bool {
        if self.doc.sketches.is_empty() {
            return false;
        }
        let mut moved = false;
        for id in self.sketch_order() {
            let Some(sketch) = self.doc.sketches.get(&id).cloned() else { continue };
            if !sketch.live {
                continue;
            }
            // The curve is gone and the record waits for an undo to bring it back.
            let Some(points) = self.doc.curve(id).map(|c| c.control_points().to_vec()) else { continue };
            let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
            let plane = sketch.plane;
            if self.part_graph().node(plane).is_none() {
                self.fix_sketch(id, format!("{name} no longer follows a plane — the plane it lay on went"));
                continue;
            }
            if self.sketch_reads_itself(id, plane) {
                self.fix_sketch(id, format!("{name} is now fixed — the plane it lies on is built on it"));
                continue;
            }
            let frame = match self.plane_laid(plane) {
                Ok(frame) => frame,
                Err(why) => {
                    self.fix_sketch(id, format!("{name} is now fixed — {why}"));
                    continue;
                }
            };
            let placed: Vec<bool> =
                (0..points.len()).map(|i| self.doc.curve(id).is_some_and(|c| c.placed_on(i).is_some())).collect();
            let mut after = sketch.clone();
            let plane_name = self.body_label(plane);
            // **Edits first, read against the frame they were made in.** A point not where the
            // last laying put it was moved by hand since, whether or not the plane has moved too
            // (a typed `set` does not rebuild, so both can have happened by now): across the plane
            // it is the sketch's new shape, off it the sketch has become a 3D one.
            let mut strayed: Option<String> = None;
            match sketch.laid {
                Some(laid) => {
                    for (i, p) in points.iter().enumerate() {
                        let Some(local) = after.local.get_mut(i) else { break };
                        let (read, off) = laid.local_of(*p);
                        if placed[i] || (laid.lay(*local) - p).norm() > 1e-9 {
                            if off.abs() > ON_PLANE_MM {
                                strayed = Some(if placed[i] {
                                    format!("{name} left {plane_name} — a point of it sits on a block off the plane, so it is a 3D sketch now")
                                } else {
                                    format!("{name} left {plane_name} — a 3D sketch now, it no longer follows the plane")
                                });
                                break;
                            }
                            *local = read;
                        }
                    }
                }
                // Laid for the first time: its shape is read off the plane as it stands.
                None => after.local = points.iter().map(|p| frame.local_of(*p).0).collect(),
            }
            if let Some(why) = strayed {
                self.fix_sketch(id, why);
                continue;
            }
            // **Then laid in the plane's frame as it is now**, which is what makes it follow. A
            // point that sits on a point block is that block's and is left to it; off the plane
            // where the plane now is, it takes the sketch with it.
            for (i, local) in after.local.iter().enumerate() {
                let Some(p) = points.get(i) else { break };
                if placed[i] {
                    if frame.local_of(*p).1.abs() > ON_PLANE_MM {
                        strayed = Some(format!(
                            "{name} left {plane_name} — a point of it sits on a block off the plane, so it is a 3D sketch now"
                        ));
                        break;
                    }
                    continue;
                }
                let laid = frame.lay(*local);
                if (laid - p).norm_squared() > 1e-24
                    && let Some(c) = self.doc.curve_mut(id)
                {
                    c.set_control_point(i, laid);
                    moved = true;
                }
            }
            if let Some(why) = strayed {
                self.fix_sketch(id, why);
                continue;
            }
            after.laid = Some(frame);
            if after != sketch {
                self.doc.sketches.insert(id, after);
            }
        }
        if moved {
            fieldform_scene::relation::rederive(&mut self.doc);
            self.rebuild_curve_display();
        }
        moved
    }

    /// A sketch made Fixed outside the history, with the reason said: the edit that caused it
    /// stands, and Live can be asked for again once the curve lies flat on its plane.
    fn fix_sketch(&mut self, id: CurveId, why: String) {
        if let Some(sketch) = self.doc.sketches.get_mut(&id) {
            sketch.live = false;
        }
        self.say(why);
    }

    /// The plane a new curve is being drawn on, if a sketch is under way: what links the curve.
    pub(crate) fn sketching_on(&self) -> Option<NodeId> {
        match self.drawing_plane {
            DrawingPlane::Node(plane) => Some(plane),
            _ => None,
        }
    }

    /// A sketch record for `curve` on `plane`, its points read off the plane as they are, and the
    /// curve with those points laid exactly. `None` when the plane cannot be read.
    pub(crate) fn sketch_of(&self, curve: &Curve, plane: NodeId) -> Option<(Sketch, Curve)> {
        let frame = self.plane_laid(plane).ok()?;
        let sketch = Sketch::of(plane, &frame, curve.control_points());
        let mut laid = curve.clone();
        for (i, local) in sketch.local.iter().enumerate() {
            laid.set_control_point(i, frame.lay(*local));
        }
        Some((sketch, laid))
    }

    /// The command that puts a new curve in: on its own, or with the sketch record that lays it on
    /// `plane` — one entry, so the curve and its link go and come back together.
    pub(crate) fn insert_curve_command(&self, curve: Curve, plane: Option<NodeId>) -> Box<dyn fieldform_ops::Command> {
        match plane.and_then(|p| self.sketch_of(&curve, p)) {
            Some((sketch, laid)) => Box::new(fieldform_ops::Batch::new(
                "curve.sketch",
                vec![
                    Box::new(SetCurve::insert(laid)) as Box<dyn fieldform_ops::Command>,
                    Box::new(fieldform_ops::SetSketch::insert(curve.id, sketch)),
                ],
            )),
            None => Box::new(SetCurve::insert(curve)),
        }
    }

    /// Lays `curve` flat on `plane` and makes it a sketch there, as one undo step: the panel's
    /// "Lie on", which is also how a 3D sketch is made 2D again — on purpose, never on its own.
    pub(crate) fn lie_on_plane(&mut self, curve: CurveId, plane: NodeId) {
        let Some(before) = self.doc.curve(curve).cloned() else { return };
        let Some((sketch, laid)) = self.sketch_of(&before, plane) else {
            self.warn(format!("{} cannot be read as a plane to lie on", self.body_label(plane)));
            return;
        };
        let was = self.doc.sketches.get(&curve).cloned();
        let commands: Vec<Box<dyn fieldform_ops::Command>> = vec![
            Box::new(SetCurve::new(curve, Some(before.clone()), Some(laid))),
            Box::new(fieldform_ops::SetSketch::new(curve, was, Some(sketch))),
        ];
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(fieldform_ops::Batch::new("sketch.lie", commands))) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!("{} lies on {} and follows it", before.name, self.body_label(plane)));
    }

    /// Lets a sketch go of its plane, as one undo step: a free curve where it stands.
    pub(crate) fn free_sketch(&mut self, curve: CurveId) {
        let Some(was) = self.doc.sketches.get(&curve).cloned() else { return };
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(fieldform_ops::SetSketch::remove(curve, was))) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
        let name = self.doc.curve(curve).map(|c| c.name.clone()).unwrap_or_default();
        self.say(format!("{name} is free of its plane"));
    }

    /// Makes a Fixed sketch Live again, if it lies flat on its plane; says why not otherwise.
    pub(crate) fn sketch_live_again(&mut self, curve: CurveId) {
        let Some(was) = self.doc.sketches.get(&curve).cloned() else { return };
        let name = self.doc.curve(curve).map(|c| c.name.clone()).unwrap_or_default();
        let plane_name = self.body_label(was.plane);
        let flat = match (self.plane_laid(was.plane), self.doc.curve(curve)) {
            (Ok(frame), Some(c)) => c.control_points().iter().all(|p| frame.local_of(*p).1.abs() <= ON_PLANE_MM),
            _ => false,
        };
        if !flat {
            self.warn(format!(
                "{name} is not flat on {plane_name} — drag its points onto the plane, or choose Lie on {plane_name} to flatten it"
            ));
            return;
        }
        if self.sketch_reads_itself(curve, was.plane) {
            self.warn(format!("{name} stays fixed — {plane_name} is built on it"));
            return;
        }
        let live = Sketch { live: true, laid: None, ..was.clone() };
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(fieldform_ops::SetSketch::new(curve, Some(was), Some(live)))) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!("{name} follows {plane_name} again"));
    }

    /// Where a dragged point of `curve` goes: onto the sketch's plane while the sketch is live,
    /// and where it was asked to go otherwise. **Drags stay on the plane**, so ordinary editing
    /// cannot take a 2D sketch off it by accident; a snap is the one move that is meant to, and
    /// the caller leaves a snapped end out of this.
    pub(crate) fn onto_sketch_plane(&self, curve: CurveId, p: Point3<f64>) -> Point3<f64> {
        match self.doc.sketches.get(&curve) {
            Some(sketch) if sketch.live => match sketch.laid {
                Some(laid) => {
                    let n = laid.normal();
                    let (_, off) = laid.local_of(p);
                    p - n * off
                }
                None => p,
            },
            _ => p,
        }
    }

    /// The sketches among `curves` whose points a gesture has just taken off their plane: each is
    /// made Fixed in the same entry as the move, with the reason said, so one undo brings back both
    /// the flat curve and its link.
    pub(crate) fn sketch_breaks(&mut self, curves: &[CurveId]) -> Vec<Box<dyn fieldform_ops::Command>> {
        let mut out: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
        for id in curves {
            let Some(sketch) = self.doc.sketches.get(id).cloned().filter(|s| s.live) else { continue };
            let Some(laid) = sketch.laid else { continue };
            let Some(curve) = self.doc.curve(*id) else { continue };
            if curve.control_points().iter().all(|p| laid.local_of(*p).1.abs() <= ON_PLANE_MM) {
                continue;
            }
            let name = curve.name.clone();
            let plane_name = self.body_label(sketch.plane);
            self.say(format!("{name} left {plane_name} — a 3D sketch now, it no longer follows the plane"));
            let fixed = Sketch { live: false, ..sketch.clone() };
            out.push(Box::new(fieldform_ops::SetSketch::new(*id, Some(sketch), Some(fixed))));
        }
        out
    }

    /// The rows of a sketch's block: which plane it lies on and whether it follows it, and for a
    /// free curve the offer to lie on one.
    pub(crate) fn sketch_rows(&mut self, ui: &mut egui::Ui, id: CurveId) {
        let graph = self.part_graph().clone();
        let planes: Vec<(NodeId, String)> = graph
            .nodes()
            .filter(|(_, n)| matches!(n, Node::Plane { .. }))
            .map(|(n, _)| (n, self.body_label(n)))
            .collect();
        let mut lie_on: Option<NodeId> = None;
        let mut free = false;
        let mut live_again = false;
        let mut fix = false;
        match self.doc.sketches.get(&id).cloned() {
            Some(sketch) => {
                ui.horizontal(|ui| {
                    row_label(ui, "On:");
                    ui.label(egui::RichText::new(self.body_label(sketch.plane)).italics())
                        .on_hover_text("The plane this sketch lies on, in that plane's own coordinates");
                    if ui
                        .add(mode_button(sketch.live, "Follows"))
                        .on_hover_text("A 2D sketch: laid on the plane again whenever the plane moves")
                        .clicked()
                        && !sketch.live
                    {
                        live_again = true;
                    }
                    if ui
                        .add(mode_button(!sketch.live, "Fixed"))
                        .on_hover_text("Left where it is, whatever the plane does — a 3D sketch")
                        .clicked()
                        && sketch.live
                    {
                        fix = true;
                    }
                    if ui.button("Free").on_hover_text("Forgets the plane altogether").clicked() {
                        free = true;
                    }
                });
                if !sketch.live && ui.button(format!("Lie on {}", self.body_label(sketch.plane))).on_hover_text("Flattens it onto the plane and makes it follow again").clicked() {
                    lie_on = Some(sketch.plane);
                }
            }
            None => {
                ui.horizontal(|ui| {
                    row_label(ui, "On:");
                    egui::ComboBox::from_id_salt((id.raw(), "sketch_plane"))
                        .selected_text("no plane")
                        .show_ui(ui, |ui| {
                            if planes.is_empty() {
                                ui.add_enabled(false, egui::Button::new("no plane yet").frame(false))
                                    .on_disabled_hover_text("Add a plane and it appears here; the curve would lie flat on it and follow it.");
                            }
                            for (plane, name) in &planes {
                                if ui.selectable_label(false, name).on_hover_text("Flattens the curve onto this plane and makes it a 2D sketch that follows it").clicked() {
                                    lie_on = Some(*plane);
                                }
                            }
                        });
                });
            }
        }
        if let Some(plane) = lie_on {
            self.lie_on_plane(id, plane);
        }
        if free {
            self.free_sketch(id);
        }
        if live_again {
            self.sketch_live_again(id);
        }
        if fix && let Some(was) = self.doc.sketches.get(&id).cloned() {
            let fixed = Sketch { live: false, ..was.clone() };
            self.gesture = None;
            if let Err(e) = self.stack.execute(&mut self.doc, Box::new(fieldform_ops::SetSketch::new(id, Some(was), Some(fixed)))) {
                self.warn(e.to_string());
            }
            self.mark_changed();
        }
    }
}
