// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The point the camera turns about, found under the pointer.

use crate::*;

impl App {
    /// How far a pick has to march before giving up: to the model, and then across it.
    ///
    /// It used to be a multiple of the model's own size, which is wrong whenever the camera is
    /// further away than the model is wide. Hide a metre cage, add a hundred-millimetre sphere
    /// in the middle of it, and the camera is still framed for the cage — so the budget was a
    /// few hundred millimetres and the sphere was two metres away. Every click missed, the
    /// pivot stayed where it was, and it looked as though the pivot were stuck to the geometry
    /// that had just been hidden.
    pub(crate) fn trace_reach(&self, eye: Point3<f64>, bounds: (Point3<f64>, Point3<f64>)) -> f64 {
        let centre = Point3::from((bounds.0.coords + bounds.1.coords) * 0.5);
        let across = (bounds.1 - bounds.0).norm();
        (eye - centre).norm() + across.max(1.0) * 2.0
    }

    /// Finds the point to tumble about, under the cursor.
    ///
    /// Traces the field rather than raycasting a mesh of it. A mesh is a mode-dependent
    /// approximation of what is on screen — in traced mode a noticeably coarser one — so
    /// picking against it made the pivot land differently depending on which view happened to
    /// be showing, and miss entirely near a silhouette. The field is the same in every mode.
    ///
    /// Curves are picked too, in screen space, because a curve has no surface to hit but is
    /// still something worth pivoting around.
    pub(crate) fn pick_pivot(
        &self,
        origin: Point3<f32>,
        dir: Vector3<f32>,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Option<Point3<f32>> {
        let bounds = self.pick_bounds();
        let radius = ((bounds.1 - bounds.0).norm() * 0.5).max(1e-6);

        // Only what the mode actually draws can be aimed at, and a cage may not stand in for
        // its field or the reverse. A cage strictly *encloses* its limit surface — a
        // Catmull-Clark cube's limit is roughly a sphere well inside the box — so the two are
        // wrong for each other in opposite directions. Picking the field in polygon mode
        // misses every click except near the middle of the cage; picking the cage in field
        // mode lands the pivot in empty space in front of the surface being looked at.
        //
        // Falling back from one to the other is therefore not a kindness, it is the bug. Only
        // overlay draws both, and there the field is what reads as solid.
        let (may_pick_cage, may_pick_field) = match self.view.mode {
            ViewMode::Polygon => (true, false),
            ViewMode::Field => (false, true),
            ViewMode::Overlay => (true, true),
        };

        if may_pick_field {
            if let Some(field) = &self.pick_field {
                let eye = Point3::new(origin.x as f64, origin.y as f64, origin.z as f64);
                let d = Vector3::new(dir.x as f64, dir.y as f64, dir.z as f64).normalize();
                // Generous budget: a click happens once, and giving up early looks exactly
                // like hitting nothing.
                let reach = self.trace_reach(eye, bounds);
                if let Some(hit) = trace_field(field, eye, d, reach, radius * 1e-4, 512) {
                    return Some(Point3::new(hit.x as f32, hit.y as f32, hit.z as f32));
                }
            }
        }

        // Only if the cage is actually on screen. Hiding a body has to take it out of every
        // answer the viewport gives, not only the drawn one — a pivot landing on geometry that
        // is not there reads as the pivot being broken, which is how this was noticed.
        if may_pick_cage && self.control_visible() {
            if let Some((hit, _)) = self.cage_mesh.raycast(origin, dir) {
                return Some(hit);
            }
        }

        // No surface under the cursor: try the curves, which are drawn but have no solid to
        // hit. Screen-space, because that is how the user aimed at them.
        const CURVE_PICK_PIXELS: f32 = 10.0;
        let mut best: Option<(f32, Point3<f32>)> = None;
        for curve in self
            .doc
            .curves
            .values()
            .filter(|c| !self.curve_ui.hidden.contains(&c.id))
        {
            for p in curve.tessellate((radius * 2e-3).max(1e-3)) {
                let world = Point3::new(p.x as f32, p.y as f32, p.z as f32);
                let Some(screen) = self.camera.project_to_screen(aspect, rect, world) else {
                    continue;
                };
                let d = screen.distance(pos);
                if d <= CURVE_PICK_PIXELS && best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, world));
                }
            }
        }
        best.map(|(_, p)| p)
    }
}
