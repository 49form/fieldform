// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The drawing grid: where it lies, how it is drawn, and its console verb.

use crate::*;

impl App {
    /// The grid on the drawing plane for this view, or `None` when there is no plane, the grid
    /// is off, or the plane cannot be seen.
    ///
    /// The pitch is measured where the grid is looked at: a millimetre along the plane's X at
    /// the window's centre, projected. In perspective that varies across the plane and the
    /// centre is the honest single answer; in the square view the plane was set from, it is
    /// exact everywhere. Asked by the drawing and by every snap, so what a snap takes is
    /// exactly what is on screen.
    pub(crate) fn drawing_grid(&self, aspect: f32, rect: egui::Rect) -> Option<PlaneGrid> {
        // Orthographic views only. In perspective the far lines crowd together and the grid
        // read as noise over the model (owner, 2026-09-06); a grid is for the view you measure
        // off, and that is the square one. Not drawn means not snapped to, as everywhere.
        if !self.view.show_grid || self.camera.projection != Projection::Orthographic {
            return None;
        }
        let frame = self.drawing_frame()?;
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        let (translation, rotation) =
            fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers).ok()?;
        // **And only square to the plane.** A grid seen from an angle reads as a grid on
        // something else -- a sketch's plane left behind when the view was turned to the world's
        // front (the owner, 2026-10-31) -- and a grid is for the view you measure off, which is
        // the square one. Not drawn means not snapped to, as everywhere.
        let normal = (rotation * Vector3::z()).cast::<f32>();
        if self.camera.forward().dot(&normal).abs() < GRID_SQUARE_COS {
            return None;
        }
        // The window follows the pivot: its foot on the plane is the centre.
        let poi = self.camera.last_poi.cast::<f64>();
        let local = rotation.inverse() * (poi.coords - translation);
        let on_plane = |x: f64, y: f64| {
            Point3::from(rotation * Vector3::new(x, y, 0.0) + translation).cast::<f32>()
        };
        let at = self.camera.project_to_screen(aspect, rect, on_plane(local.x, local.y))?;
        let one_mm =
            self.camera.project_to_screen(aspect, rect, on_plane(local.x + 1.0, local.y))?;
        let px_per_mm = at.distance(one_mm) as f64;
        let pitch = grid_pitch(px_per_mm);
        let centre = ((local.x / pitch).round() as i64, (local.y / pitch).round() as i64);
        // Enough lines to cover the viewport in the square view, and a cap so an oblique
        // perspective — where the far lines crowd together anyway — does not draw hundreds.
        let diagonal = (rect.width().powi(2) + rect.height().powi(2)).sqrt() as f64;
        let cells = (diagonal / (pitch * px_per_mm)).ceil() as i64;
        let reach = (cells / 2 + 2).clamp(4, 60);
        Some(PlaneGrid { translation, rotation, pitch, px_per_mm, centre, reach })
    }

    /// Draws the grid, line by line, projected from the plane.
    ///
    /// A line on a plane projects to a line, so each is exact between its two ends however the
    /// plane is seen. An end behind the camera drops its line rather than streaking it across
    /// the viewport — the same rule the curves follow.
    pub(crate) fn draw_grid(&self, painter: &egui::Painter, aspect: f32, rect: egui::Rect, grid: &PlaneGrid) {
        let on_plane = |x: f64, y: f64| {
            Point3::from(grid.rotation * Vector3::new(x, y, 0.0) + grid.translation).cast::<f32>()
        };
        let project = |x: f64, y: f64| self.camera.project_to_screen(aspect, rect, on_plane(x, y));
        let colour = |index: i64| {
            if index == 0 {
                GRID_AXIS
            } else if index % 5 == 0 {
                GRID_MAJOR
            } else {
                GRID_MINOR
            }
        };
        let (cx, cy) = grid.centre;
        let span = |c: i64| ((c - grid.reach) as f64 * grid.pitch, (c + grid.reach) as f64 * grid.pitch);
        let (x0, x1) = span(cx);
        let (y0, y1) = span(cy);
        for k in (cx - grid.reach)..=(cx + grid.reach) {
            let x = k as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x, y0), project(x, y1)) {
                painter.line_segment([a, b], egui::Stroke::new(1.0_f32, colour(k)));
            }
        }
        for l in (cy - grid.reach)..=(cy + grid.reach) {
            let y = l as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x0, y), project(x1, y)) {
                painter.line_segment([a, b], egui::Stroke::new(1.0_f32, colour(l)));
            }
        }

        // **The numbers, which are the point of a grid.** Lines alone say that something is
        // regular; they do not say how big it is, and a part on screen has no size until
        // something on screen is labelled. Asked for directly: "I want actual numbers displayed
        // on it -- either every 10 mm or 100 or 1000 -- so users know the size."
        //
        // Every line when there is room to write between them, every fifth when there is not.
        // The pitch is a decade, so a fifth is not a number anybody thinks in -- but it is a
        // *label* rather than a step, and a crowded margin is unreadable at any spacing.
        let cell_px = grid.pitch * grid.px_per_mm;
        let step = if cell_px >= 64.0 { 1 } else { 5 };
        let font = egui::FontId::proportional(11.0);
        // Whole millimetres always, since a decade pitch times an integer is one.
        let say = |v: f64| format!("{}", v.round() as i64);
        for k in (cx - grid.reach)..=(cx + grid.reach) {
            if k % step != 0 {
                continue;
            }
            let x = k as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x, y0), project(x, y1))
                && let Some(at) = grid_label_at(rect, a, b)
            {
                painter.text(at, egui::Align2::CENTER_CENTER, say(x), font.clone(), GRID_LABEL);
            }
        }
        for l in (cy - grid.reach)..=(cy + grid.reach) {
            if l % step != 0 {
                continue;
            }
            let y = l as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x0, y), project(x1, y))
                && let Some(at) = grid_label_at(rect, a, b)
            {
                painter.text(at, egui::Align2::CENTER_CENTER, say(y), font.clone(), GRID_LABEL);
            }
        }

        // And what one cell is, once, in the corner -- so the numbers have a unit without every
        // one of them carrying it.
        painter.text(
            egui::pos2(rect.right() - 8.0, rect.bottom() - 8.0),
            egui::Align2::RIGHT_BOTTOM,
            format!("grid {} mm", say(grid.pitch)),
            egui::FontId::proportional(11.0),
            GRID_LABEL,
        );
    }

    /// `point` moved to the nearest intersection of the grid that is drawn, if one is.
    pub(crate) fn grid_snap(&self, aspect: f32, rect: egui::Rect, point: Point3<f64>) -> Option<Point3<f64>> {
        let grid = self.drawing_grid(aspect, rect)?;
        Some(snap_to_grid(point, grid.translation, grid.rotation, grid.pitch))
    }

    /// `grid` toggles the drawing plane's grid; `grid on=true` and `grid on=false` set it.
    pub(crate) fn grid_from_console(&mut self, args: &Args) {
        self.view.show_grid = match args.get("on") {
            Some(Value::Bool(on)) => *on,
            Some(other) => {
                other.as_f64().is_some_and(|v| v != 0.0)
                    || other.as_str().is_some_and(|s| matches!(s, "on" | "yes"))
            }
            None => !self.view.show_grid,
        };
        self.say_grid();
    }

    /// Says where the grid stands, and what is still needed to see it.
    ///
    /// One sentence for three switches — the console, the View menu and the button under the
    /// ViewCube — so they cannot describe the same state three ways. "On" is not the same as
    /// "shown": the grid also needs a drawing plane and an orthographic view, and a switch that
    /// says "on" while nothing appears reads as broken, so the missing piece is named.
    pub(crate) fn say_grid(&mut self) {
        if !self.view.show_grid {
            self.say("Grid off — nothing to snap to");
        } else if self.drawing_plane == DrawingPlane::None {
            self.say(
                "Grid on — it appears once a drawing plane is set and the view is \
                 orthographic. Hold Shift to snap to it",
            );
        } else if self.camera.projection != Projection::Orthographic {
            self.say(
                "Grid on — it shows in an orthographic view; the toggle is under the \
                 ViewCube, left of this one",
            );
        } else {
            self.say(
                "Grid on — hold Shift while dragging a point or placing a line to snap to it",
            );
        }
    }
}
