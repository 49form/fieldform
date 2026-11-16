// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What is drawn from the document: the meshes, the curve display and the bounds of the scene.

use crate::*;

impl App {
    /// What the camera is looking at, for the clip planes and the marcher's reach.
    ///
    /// **Its own step, because a drag needs it without the rest of the rebuild.** The far plane
    /// is the scene's centre plus twice its radius, and both came only from the full rebuild --
    /// so a body dragged past where the part *was* when that last ran met the far plane and was
    /// cut off along a straight line, until the hand came off and the rebuild moved the plane.
    /// One extent walk over the bodies is cheap enough for every frame of a drag; the cage bake
    /// and the picking compile beside it in the rebuild are not, and are what the drag skips.
    pub(crate) fn refresh_scene_bounds(&mut self) -> (Point3<f64>, Point3<f64>) {
        let (lo, hi) = self.pick_bounds();
        let centre = Point3::from(((lo.coords + hi.coords) * 0.5).cast::<f32>());
        let radius = ((hi - lo).norm() * 0.5).max(1e-6) as f32;
        self.camera.set_scene(centre, radius);
        (lo, hi)
    }

    /// Brings everything derived from the part up to date after it changes: curve ends settle
    /// onto their points, the cage is re-triangulated, the scene bounds are refreshed, the traced
    /// view is marked stale, the field that picking traces is compiled, axes on a surface follow
    /// their body, and the drawn meshes are uploaded again.
    ///
    /// Called whenever the part, or what is visible of it, changes. A cage drag does not come here
    /// each frame — `update_after_vertex_move` rewrites only the faces it moved — and `end_move`
    /// calls it once the gesture is committed.
    pub(crate) fn rebuild_render_resources(&mut self) {
        let whole = std::time::Instant::now();
        // Whatever asked for this changed the part or what is visible of it, which is what the
        // overlay reads; opening a file and starting a new one reach the readings this way.
        self.document_moved();
        // Before everything else here, because a point's position is in the graph and all of what
        // follows reads it. See `refresh_surface_points`.
        self.refresh_surface_points();
        // And the sketches after them, since a plane may stand on a point: a curve's points are
        // what an extrude compiles from. See `refresh_sketches`.
        self.refresh_sketches();
        // **Curve ends catch up here, because this is what runs when the part changes.**
        // It was called on binding an end, on editing a variable and on opening a document —
        // and not when the *point itself* moved, which is the one case anybody would try first.
        // Dragging a point's X went through the block panel, which rebuilds but never asked a
        // curve to follow, so the end stayed where it was and the feature looked inert.
        //
        // Before the cage mesh and the bounds below, since both read geometry this may move.
        if self.settle_curves_on_points() {
            self.rebuild_curve_display();
        }
        self.cage_mesh = CageMesh::from_cage(self.control());

        // What the camera is looking at, for the clip planes and the marcher's reach. Here
        // because this runs whenever the part, or what is visible of it, changes -- and on its
        // own for the drags that move a body without running the rest of this.
        let (lo, hi) = self.refresh_scene_bounds();

        // The traced view holds a compiled shader and a baked cage, both of which this rebuild
        // has just invalidated. Marking rather than rebuilding here keeps the cost off cage
        // edits that are never looked at in that mode.
        self.traced.invalidate(traced::Quality::Final);

        // The field picking traces. Built here rather than per click, because compiling it
        // subdivides the cage and a click should not pay for that.
        //
        // And it is the *only* compile: the panel used to call `compile_error`, which compiled
        // the whole graph again to ask whether it could be compiled — from inside the inspector,
        // so once a frame. On a 109 000-triangle STEP import that is four BVH builds and four
        // crease passes sixty times a second, for an answer that only changes when the document
        // does. One compile, both consumers.
        //
        // Through the leaf cache, so hiding a body does not rebuild every other body's BVH.
        // What a show/hide changes is which roots are in the graph; the meshes are the same
        // meshes, and rebuilding them was 90 ms of the pause on a 109 000-triangle import.
        // From the box measured above, which is the one the overlay's readings keep.
        let tolerance = (hi - lo).norm() * 1e-3;
        let graph = self.display_graph();
        let compiling = std::time::Instant::now();
        let compiled = GraphField::compile_cached(
            &self.doc,
            &graph,
            tolerance.max(1e-4),
            &mut self.leaves,
        );
        self.note_stage("compiling the part for picking", compiling);
        match compiled {
            Ok(field) => {
                self.pick_field = Some(field);
                self.compile_error = None;
                self.draft_refused = None;
            }
            Err(e) => {
                self.pick_field = None;
                // A raw node id means nothing in a panel that lists blocks by name, so the
                // errors that carry one say which block instead. This is the difference
                // between a document that looks broken and one that says what to change:
                // a blend against a lattice draws nothing and exports nothing, and until it
                // named the block there was no way to tell which of several was at fault.
                self.compile_error = Some(self.explain_compile(&e));
                self.draft_refused = match e {
                    fieldform_field::CompileError::DraftCrossesTheSection { node, .. } => Some(node),
                    _ => None,
                };
            }
        }

        // The control cage is the only mesh the viewport draws. Everything smooth — the
        // limit surface included — is the traced view's job, evaluated per pixel.
        // **An axis on a surface follows its body**, and this is where it catches up: after the
        // field is compiled, because the rule needs the body's field, and before what is drawn is
        // rebuilt. Only the curve display is refreshed — never this function — so it cannot loop.
        if self.refresh_surface_axes() {
            self.rebuild_curve_display();
        }
        self.refresh_drawn_meshes();
        self.note_stage("rebuilding what is drawn", whole);
    }

    /// Rebuilds the triangle buffers the viewport draws and uploads them.
    ///
    /// Separate from the rest of the rebuild because the *mode* changes what is in them and
    /// nothing else about the part: cycling to Polygon needs the imported solids as triangles,
    /// where in Field mode the tracer is already drawing them. Pressing Tab should not pay for
    /// a compile, and an edit should not skip this.
    pub(crate) fn refresh_drawn_meshes(&mut self) {
        let started = std::time::Instant::now();
        self.rebuild_imported_mesh();
        let resources = CageRenderResources::new(
            &self.wgpu_render_state.device,
            self.wgpu_render_state.target_format,
            &self.cage_mesh,
            &self.imported_mesh,
        );
        self.wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);
        self.note_stage("the drawing resources", started);
    }

    /// Re-tessellates every curve for drawing.
    ///
    /// 0.05 mm is far finer than a screen pixel at any sane framing, so the drawn curve is
    /// smooth rather than faceted, and this runs on import rather than per frame.
    pub(crate) fn rebuild_curve_display(&mut self) {
        let started = std::time::Instant::now();
        // Every path that changes a curve comes through here, which is what makes this the
        // counter the profile rows are cached against.
        self.curve_ui.revision = self.curve_ui.revision.wrapping_add(1);
        // And the overlay's readings, for the same reason: a sweep's extent is its curve's.
        self.document_moved();
        self.curve_ui.display = self
            .doc
            .curves
            .values()
            .map(|c| {
                let points = c
                    .tessellate(0.05)
                    .into_iter()
                    .map(|p| Point3::new(p.x as f32, p.y as f32, p.z as f32))
                    .collect();
                (c.id, points)
            })
            .collect();
        // HashMap order is arbitrary; sort so the curve list in the UI does not reshuffle
        // between frames.
        self.curve_ui.display.sort_by_key(|(id, _)| id.raw());
        self.note_stage("the curve overlay", started);
    }

    /// Bounds over everything in the scene — the cage and every curve — as centre and radius.
    ///
    /// `None` only when there is nothing at all to frame.
    pub(crate) fn scene_bounds(&self) -> Option<(Point3<f32>, f32)> {
        let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        fn grow(
            min: &mut Point3<f32>,
            max: &mut Point3<f32>,
            lo: Point3<f32>,
            hi: Point3<f32>,
        ) {
            *min = Point3::new(min.x.min(lo.x), min.y.min(lo.y), min.z.min(lo.z));
            *max = Point3::new(max.x.max(hi.x), max.y.max(hi.y), max.z.max(hi.z));
        }

        // Every visible body, from the graph — which is the thing the viewport draws, and so
        // the only description of the scene that cannot disagree with it. This used to be the
        // cage and the curves alone, so a part made of primitives contributed *nothing*: a
        // 300 mm sphere framed to the 100 mm cage sitting invisibly behind it, and an import
        // framed to the same. The cage is one body among several, not the scene.
        let part = self.part_graph();
        let consumed = part.consumed();
        for body in part.roots() {
            if !is_visible_among(&self.doc, part, &consumed, body) {
                continue;
            }
            // An unbounded body — a half-space, a lattice — contributes no box and does not
            // veto the others', the same rule `pick_bounds` follows. A plane has no extent to
            // frame, and letting it erase everything else would leave the camera nowhere.
            if let Some((lo, hi)) = self.body_extent(body).as_box() {
                grow(&mut min, &mut max, lo.cast::<f32>(), hi.cast::<f32>());
            }
        }

        // Curves are drawn but are not bodies, so they are their own contribution.
        for (id, points) in &self.curve_ui.display {
            if self.curve_ui.hidden.contains(id) {
                continue;
            }
            for p in points {
                grow(&mut min, &mut max, *p, *p);
            }
        }

        // And surfaces, for the same reason. Framing has to reach an imported surface or a
        // document holding nothing else opens looking at empty space.
        for (id, edges) in &self.surface_display {
            if self.hidden_surfaces.contains(id) {
                continue;
            }
            for [a, b] in edges {
                grow(&mut min, &mut max, *a, *a);
                grow(&mut min, &mut max, *b, *b);
            }
        }

        // The cage last and only if nothing else spoke. It is in the graph when a body draws
        // it, so it is already counted above in the ordinary case; this is the empty document,
        // where framing the cage that Create > Cage would bring back beats framing nothing.
        if !min.x.is_finite() {
            let (centre, radius) = cage_bounds(self.cage());
            grow(
                &mut min,
                &mut max,
                centre - Vector3::repeat(radius),
                centre + Vector3::repeat(radius),
            );
        }
        if !min.x.is_finite() {
            return None;
        }
        let center = Point3::from((min.coords + max.coords) * 0.5);
        Some((center, ((max - min).norm() * 0.5).max(0.1)))
    }

    /// What framing should aim at: the selected body if there is one, else the whole scene.
    ///
    /// "Frame selection" has to mean the selection when there is one — framing the document
    /// while one body is selected is the command declining to do what it is called.
    pub(crate) fn framing_bounds(&self) -> Option<(Point3<f32>, f32)> {
        // A block something consumes frames whatever its own eye says, as the manipulator keeps
        // its handles (`gizmo_frame`): that eye only draws it beside its consumer as well, and
        // framing a selection never asked whether the selection was drawn through something.
        if let Some(body) = self.selected_body
            && !(self.eye_flag(body) == EyeFlag::Hidden && !self.is_visible(body))
            && let Some((lo, hi)) = self.body_extent(body).as_box()
        {
            let (lo, hi) = (lo.cast::<f32>(), hi.cast::<f32>());
            let centre = Point3::from((lo.coords + hi.coords) * 0.5);
            return Some((centre, ((hi - lo).norm() * 0.5).max(0.1)));
        }
        self.scene_bounds()
    }

    /// The selected curve's bounding-box diagonal in millimetres, for scaling defaults to it.
    pub(crate) fn selected_curve_size(&self) -> Option<f64> {
        let curve = self.doc.curves.get(&self.selected_curve?)?;
        let points = curve.tessellate(0.5);
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &points {
            min = Point3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Point3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        }
        min.x.is_finite().then(|| (max - min).norm())
    }

    /// The box the model occupies, for sizing picking tolerances: [`fieldform_field::pick_bounds`]
    /// over what the viewport draws and the points it paints, as the overlay's readings keep it.
    pub(crate) fn pick_bounds(&self) -> (Point3<f64>, Point3<f64>) {
        self.overlay_readings().pick_bounds
    }

    /// Where one body's surface can be.
    ///
    /// Delegates to `fieldform_field::graph_extent` rather than walking the graph here. This
    /// used to be its own copy of that walk, which meant the viewport and anything else asking
    /// the same question of the same graph could answer it differently — the failure this
    /// codebase keeps meeting. One walk, three states: see `Extent` for why an `Option` could
    /// not say what a half-space needs it to.
    pub(crate) fn body_extent(&self, id: NodeId) -> Extent {
        fieldform_field::graph_extent(&self.doc, self.part_graph(), id)
    }
}
