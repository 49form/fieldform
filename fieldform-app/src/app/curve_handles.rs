// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A curve's control points in the viewport: finding, selecting and snapping them.

use crate::*;

impl App {
    /// The curves whose control points are drawn and can be clicked: the selected one always,
    /// and every visible curve while View > Show control points is on -- asked for as "points
    /// should stay turned on", so a point can be picked without first finding its curve.
    pub(crate) fn handle_curves(&self) -> Vec<CurveId> {
        let mut out: Vec<CurveId> = Vec::new();
        if let Some(id) = self.selected_curve
            && !self.curve_ui.hidden.contains(&id)
            && self.doc.curve(id).is_some()
        {
            out.push(id);
        }
        if self.curve_ui.show_control_points {
            let mut rest: Vec<CurveId> = self
                .doc
                .curves
                .keys()
                .copied()
                .filter(|id| Some(*id) != self.selected_curve && !self.curve_ui.hidden.contains(id))
                .collect();
            rest.sort_by_key(|id| id.raw());
            out.extend(rest);
        }
        out
    }

    /// Every drawn handle's place on screen, curve by curve, the selected curve's first.
    pub(crate) fn handles_on_screen(&self, aspect: f32, rect: egui::Rect) -> Vec<(CurveId, usize, egui::Pos2)> {
        let mut out = Vec::new();
        for id in self.handle_curves() {
            let Some(c) = self.doc.curve(id) else { continue };
            for (i, p) in c.control_points().iter().enumerate() {
                if let Some(at) = self.camera.project_to_screen(aspect, rect, p.cast::<f32>()) {
                    out.push((id, i, at));
                }
            }
        }
        out
    }

    /// Which control point is under `pos`, if one is within reach: the nearest on any curve
    /// whose handles are drawn. The selected curve's come first, so a tie is theirs.
    pub(crate) fn control_point_at(
        &self,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Option<(CurveId, usize)> {
        const THRESHOLD: f32 = 10.0;
        let mut best: Option<((CurveId, usize), f32)> = None;
        for (id, i, at) in self.handles_on_screen(aspect, rect) {
            let distance = at.distance(pos);
            if distance <= THRESHOLD && best.is_none_or(|(_, best)| distance < best) {
                best = Some(((id, i), distance));
            }
        }
        best.map(|(hit, _)| hit)
    }

    /// Every picked control point, the selected curve's first.
    pub(crate) fn picked_points(&self) -> Vec<(CurveId, usize)> {
        let mut out: Vec<(CurveId, usize)> = match self.selected_curve {
            Some(id) => self.selected_points.iter().map(|i| (id, *i)).collect(),
            None => Vec::new(),
        };
        out.extend(self.curve_ui.other_points.iter().copied());
        out
    }

    pub(crate) fn is_picked(&self, curve: CurveId, index: usize) -> bool {
        (self.selected_curve == Some(curve) && self.selected_points.contains(&index))
            || self.curve_ui.other_points.contains(&(curve, index))
    }

    /// This point alone, its curve taking the selection.
    pub(crate) fn pick_only(&mut self, curve: CurveId, index: usize) {
        if self.selected_curve != Some(curve) {
            self.select_curve(curve);
            // A press on another curve's point is a click on that curve to edit it.
            self.notebook.curve_picked = Some(curve);
        }
        self.selected_points = vec![index];
        self.curve_ui.other_points.clear();
    }

    /// Adds this point to what is picked, or takes it away -- on the selected curve or on
    /// another, which is what lets one selection span curves.
    pub(crate) fn toggle_point(&mut self, curve: CurveId, index: usize) {
        match self.selected_curve {
            None => self.pick_only(curve, index),
            Some(id) if id == curve => toggle_in(&mut self.selected_points, index),
            Some(_) => toggle_in(&mut self.curve_ui.other_points, (curve, index)),
        }
    }

    /// Makes `picked` the point selection, keeping the selected curve as the one that holds
    /// the handles when any point is on it.
    pub(crate) fn set_point_selection(&mut self, picked: &[(CurveId, usize)]) {
        match split_selection(picked, self.selected_curve) {
            Some((primary, indices, others)) => {
                if self.selected_curve != Some(primary) {
                    self.select_curve(primary);
                }
                self.selected_points = indices;
                self.curve_ui.other_points = others;
            }
            None => self.clear_point_selection(),
        }
    }

    pub(crate) fn clear_point_selection(&mut self) {
        self.selected_points.clear();
        self.curve_ui.other_points.clear();
    }

    /// Picks every drawn handle inside the box from `from` to `to` -- replacing what was
    /// picked, or adding to it -- and says how many. A box too small to have been meant
    /// changes nothing, since that press was a click.
    pub(crate) fn finish_marquee(
        &mut self,
        aspect: f32,
        rect: egui::Rect,
        from: egui::Pos2,
        to: egui::Pos2,
        adding: bool,
    ) {
        let bx = egui::Rect::from_two_pos(from, to);
        if bx.width() < 3.0 && bx.height() < 3.0 {
            return;
        }
        let inside = points_in_box(&self.handles_on_screen(aspect, rect), bx);
        let mut picked = if adding { self.picked_points() } else { Vec::new() };
        for p in inside {
            if !picked.contains(&p) {
                picked.push(p);
            }
        }
        self.set_point_selection(&picked);
        self.selection.clear();
        let n = self.picked_points().len();
        self.say(format!("{n} control point{} picked", if n == 1 { "" } else { "s" }));
    }

    /// The curve under the cursor, if one is close enough to have been aimed at.
    ///
    /// **A curve could not be picked in the viewport at all until this.** Control points are
    /// clickable, but only once their curve is selected, and a curve could only be selected from
    /// the block list — so the way to reach a curve you were looking at was to find its row in a
    /// panel, which is exactly backwards for the one kind of geometry you draw by hand.
    ///
    /// Measured against the *drawn* polyline rather than the control polygon, because the drawn
    /// line is what the click is aimed at. A degree-9 curve's control points can sit a long way
    /// from the curve itself, and picking against them would mean clicking empty space to select
    /// something and clicking the visible line to miss it.
    ///
    /// Hidden curves are not offered: a click cannot mean something you cannot see.
    pub(crate) fn curve_at(&self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) -> Option<CurveId> {
        // Generous, because a curve is one pixel wide and nobody can hit that. The same figure
        // a control point uses, which keeps the two feeling like one target.
        const THRESHOLD: f32 = 10.0;
        let mut best: Option<(CurveId, f32)> = None;

        for curve in self.doc.curves.values() {
            if self.curve_ui.hidden.contains(&curve.id) {
                continue;
            }
            // At a tolerance proportional to the curve, so a big one is not tessellated into
            // thousands of segments for a hit test. Picking wants shape, not precision.
            let drawn = curve.tessellate((curve.length(0.5) * 1e-2).max(1e-3));
            let mut previous: Option<egui::Pos2> = None;
            for p in &drawn {
                let screen = self.camera.project_to_screen(aspect, rect, p.cast::<f32>());
                // A point behind the camera projects to nothing, and the segment leading to it
                // cannot be measured — so the chain restarts rather than joining across it.
                let Some(screen) = screen else {
                    previous = None;
                    continue;
                };
                if let Some(from) = previous {
                    let distance = distance_to_segment(pos, from, screen);
                    if distance <= THRESHOLD && best.is_none_or(|(_, b)| distance < b) {
                        best = Some((curve.id, distance));
                    }
                }
                previous = Some(screen);
            }
        }
        best.map(|(id, _)| id)
    }

    /// The corners of the square drawn for a plane, in world: the one geometry the drawing and
    /// the picking share, so a click on the square is a click on what is seen.
    pub(crate) fn plane_square(&self, origin: Point3<f64>, facing: Rotation3<f64>) -> [Point3<f64>; 4] {
        let (lo, hi) = self.pick_bounds();
        let scene = (hi - lo).norm();
        // Sized from the scene, and from the view when the scene has no size to give — a part
        // that is nothing but a plane, or an empty document. Either way the square is a drawing
        // aid, so it is deliberately bigger than what it cuts.
        let span = if scene > 1e-6 {
            scene * 0.6
        } else {
            self.camera.view_half_height(origin.cast::<f32>()) as f64
        };
        let u = facing * Vector3::x() * span;
        let v = facing * Vector3::y() * span;
        [origin + u + v, origin - u + v, origin - u - v, origin + u - v]
    }

    /// The plane whose square is under `pos`: by its outline, within a control point's reach,
    /// and with `inside` by its area too.
    ///
    /// **The outline on a press, the area only where nothing else is.** A plane standing on a
    /// face lies over that face, so a click anywhere on it would otherwise take every click meant
    /// for the body under it; the outline is the plane's own and nothing else's. A click that hits
    /// no body is free to mean the square it landed in (the owner, 2026-10-31: "if there's a plane
    /// you can click on it, you should be able to select it").
    pub(crate) fn plane_at(&self, aspect: f32, rect: egui::Rect, pos: egui::Pos2, inside: bool) -> Option<NodeId> {
        const THRESHOLD: f32 = 8.0;
        let mut best: Option<(NodeId, f32)> = None;
        for (id, origin, facing) in self.plane_frames() {
            let screen: Vec<egui::Pos2> = self
                .plane_square(origin, facing)
                .iter()
                .filter_map(|p| self.camera.project_to_screen(aspect, rect, p.cast::<f32>()))
                .collect();
            if screen.len() != 4 {
                continue;
            }
            let on_outline = (0..4)
                .map(|i| distance_to_segment(pos, screen[i], screen[(i + 1) % 4]))
                .fold(f32::INFINITY, f32::min);
            let distance = if on_outline <= THRESHOLD {
                on_outline
            } else if inside && view_cube::point_in_convex_polygon(pos, &screen) {
                // Behind every outline hit, and the smaller square first among areas.
                THRESHOLD + 1.0 + (screen[0].distance(screen[2]))
            } else {
                continue;
            };
            if best.is_none_or(|(_, b)| distance < b) {
                best = Some((id, distance));
            }
        }
        best.map(|(id, _)| id)
    }

    /// The end of another curve that `p` should snap to, if one is near enough.
    ///
    /// **Ends only, and never the curve's own.** A profile is closed by bringing two ends
    /// together, so those are the points worth landing on exactly; an interior control point is
    /// not somewhere anything meets. Excluding `moving`'s own curve is what stops a two-point
    /// line collapsing onto itself the moment it is dragged with the modifier down.
    ///
    /// The reach is a fraction of what is on screen rather than a fixed distance in millimetres,
    /// so it means the same thing on a bracket and on an airframe — the same reasoning `Home`
    /// already uses for framing.
    pub(crate) fn endpoint_near(
        &self,
        aspect: f32,
        rect: egui::Rect,
        cursor: egui::Pos2,
        moving: Option<CurveId>,
    ) -> Option<(CurveId, Point3<f64>)> {
        let mut best: Option<(CurveId, Point3<f64>, f32)> = None;
        for curve in self.doc.curves.values() {
            // `moving` is `None` for a curve still being placed: it has no end of its own yet,
            // so every end in reach is a target.
            if Some(curve.id) == moving || self.curve_ui.hidden.contains(&curve.id) {
                continue;
            }
            let points = curve.control_points();
            let (Some(&first), Some(&last)) = (points.first(), points.last()) else {
                continue;
            };
            for end in [first, last] {
                let Some(screen) = self.camera.project_to_screen(aspect, rect, end.cast::<f32>())
                else {
                    continue;
                };
                let distance = screen.distance(cursor);
                if distance <= SNAP_PIXELS && best.is_none_or(|(_, _, b)| distance < b) {
                    best = Some((curve.id, end, distance));
                }
            }
        }
        best.map(|(id, at, _)| (id, at))
    }

    /// The end a snap would take right now, if the modifier is held and something is in reach.
    ///
    /// Shared by the drawing and by the drag, so the box cannot promise a snap the drag would
    /// not make: one question, asked once, answered in one place.
    pub(crate) fn snap_target(
        &self,
        aspect: f32,
        rect: egui::Rect,
        ui: &egui::Ui,
    ) -> Option<Point3<f64>> {
        if !ui.input(|i| i.modifiers.ctrl || i.modifiers.command) {
            return None;
        }
        let cursor = ui.input(|i| i.pointer.hover_pos())?;
        if !rect.contains(cursor) {
            return None;
        }
        // A curve being placed has no end of its own yet, so every end in reach is a target —
        // and the box is the same box, so what a click would take is what the drag would.
        if self.curve_ui.pending.is_some() {
            return self.endpoint_near(aspect, rect, cursor, None).map(|(_, at)| at);
        }
        // Mid-drag the search runs from the moving point rather than the cursor, so the box has
        // to as well — a target drawn where the drag will not look is worse than none.
        let from = match &self.curve_ui.point_drag {
            Some(state) => self
                .doc
                .curve(state.curve)
                .and_then(|c| c.control_points().get(state.indices[0]).copied())
                .and_then(|p| self.camera.project_to_screen(aspect, rect, p.cast::<f32>()))
                .unwrap_or(cursor),
            None => cursor,
        };
        // Which curve is moving, so its own ends are not offered as targets. Mid-drag that is
        // the curve being dragged; otherwise it is whatever is selected, since that is what a
        // drag would move if one began now.
        let moving = match &self.curve_ui.point_drag {
            Some(state) => state.curve,
            None => self.selected_curve?,
        };
        // Only an end can be snapped, so only offer a target when an end is what would move.
        let count = self.doc.curve(moving).map(|c| c.control_points().len())?;
        let lone_end = self.curve_ui.other_points.is_empty()
            && match self.selected_points.as_slice() {
                [i] if *i == 0 || *i + 1 == count => true,
                _ => false,
            };
        if !lone_end {
            return None;
        }
        self.endpoint_near(aspect, rect, from, Some(moving)).map(|(_, at)| at)
    }

    /// The XYZ a block carries as its own position, when it has one and it is its own.
    ///
    /// **This is what tells a block that knows where it is from one that does not.** A sphere,
    /// a cuboid and a torus carry a centre; a cylinder, a cone and a capsule carry the pair of
    /// ends they are drawn between; a plane carries its origin and a point *is* one. A boolean,
    /// a lattice, a mesh and a cage carry none — their position is wherever their operands put
    /// them, so there is nothing here to write and a placement is the only way to move them.
    ///
    /// `None` too when any of those places is taken *from* another block. Such a place follows
    /// what it sits on, by design; overwriting it would quietly detach the block from the point
    /// it was put on, which is the opposite of what dragging it should mean.
    ///
    /// A `Node::Transform` is excluded although it carries a place: `about` is the pivot it
    /// turns around, not where it is. Writing a drag there would move the pivot and leave the
    /// body still.
    pub(crate) fn own_anchors(&self, id: NodeId) -> Option<Vec<Point3<f64>>> {
        let node = self.part_graph().node(id)?;
        if let Node::Point { at } = node {
            return Some(vec![self.resolve_point(at)]);
        }
        if matches!(node, Node::Transform { .. }) {
            return None;
        }
        let places = node.places();
        if places.is_empty() || places.iter().any(|p| p.source().is_some()) {
            return None;
        }
        Some(places.iter().map(|p| self.resolve_place(p)).collect())
    }

    /// The turn a block carries as its own, resolved.
    ///
    /// `None` for a block with no facing of its own, which is what sends a rotate back to the
    /// placement path.
    pub(crate) fn own_turn(&self, id: NodeId) -> Option<Rotation3<f64>> {
        let turn = self.part_graph().node(id)?.turn()?;
        // **Only a turn this block owns.** One aimed at a plane or a line is not the block's to
        // set: dragging the rotate handle would have to either break the aim or move the thing
        // it is aimed at, and neither is what the handle promises. `None` sends the drag to the
        // placement path, which wraps the block in one — the aim is kept, and the extra turn is
        // a block of its own that can be deleted.
        turn.angles()?;
        Some(self.aim_of(turn))
    }

    /// Which way a turn points, whether it is typed or taken from something.
    ///
    /// Through the kernel's `turn_rotation`, which is what both compilers read, so the square
    /// drawn for a plane and the field built from it cannot disagree about where it faces.
    pub(crate) fn aim_of(&self, turn: &Turn) -> Rotation3<f64> {
        guard_stack("aim_of");
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::turn_rotation(&self.doc, self.part_graph(), turn, &numbers)
    }

    /// Writes one back, as the degrees the block stores.
    pub(crate) fn set_own_turn(graph: &mut Graph, id: NodeId, turn: Rotation3<f64>) {
        let (x, y, z) = turn.euler_angles();
        if let Some(node) = graph.node_mut(id)
            && let Some(slot) = node.turn_mut()
            // An aimed turn is not written over: `own_turn` already refused to offer it, and
            // the two have to agree about which turns a drag may change.
            && slot.angles().is_some()
        {
            *slot = Turn::By([
                ParamRef::Literal(x.to_degrees()),
                ParamRef::Literal(y.to_degrees()),
                ParamRef::Literal(z.to_degrees()),
            ]);
        }
    }

    /// Which block a drag on `id` should actually move.
    ///
    /// **Itself, unless its position is somebody else's.** A block whose place is taken *from*
    /// another block has no coordinates of its own — the numbers in its panel are that block's,
    /// read through. So dragging it moves the block it sits on, and the numbers on screen are the
    /// numbers that change.
    ///
    /// The alternative was writing literals into the place, which updates the panel by cutting
    /// the link the panel is showing: the point stays in the list, nothing reads it, and the V
    /// that said "this is a variable" has quietly stopped being true. Reported as a sphere whose
    /// coordinates would not move once they had been made a variable.
    ///
    /// One place with one source, because that is the only case with one answer. A cylinder with
    /// its two ends on two different points has no single block to move, so it keeps the
    /// placement path — and a block sitting on a *plane* keeps it too, since a plane is a facing
    /// as much as a position and dragging one is not the same gesture.
    pub(crate) fn drag_moves(&self, id: NodeId) -> NodeId {
        let Some(node) = self.part_graph().node(id) else { return id };
        if matches!(node, Node::Transform { .. }) {
            return id;
        }
        let places = node.places();
        let [place] = places.as_slice() else { return id };
        match place.source() {
            Some(source)
                if matches!(self.part_graph().node(source), Some(Node::Point { .. })) =>
            {
                source
            }
            _ => id,
        }
    }

    /// Writes those positions back, as the literals the block stores.
    pub(crate) fn set_own_anchors(graph: &mut Graph, id: NodeId, points: &[Point3<f64>]) {
        let at_of = |p: &Point3<f64>| {
            [
                ParamRef::Literal(p.x),
                ParamRef::Literal(p.y),
                ParamRef::Literal(p.z),
            ]
        };
        let Some(node) = graph.node_mut(id) else { return };
        if let Node::Point { at } = node {
            if let Some(p) = points.first() {
                *at = at_of(p);
            }
            return;
        }
        for (slot, p) in node.places_mut().into_iter().zip(points) {
            *slot = fieldform_scene::Place::At(at_of(p));
        }
    }
}
