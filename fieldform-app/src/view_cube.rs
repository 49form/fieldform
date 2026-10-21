// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The view cube: the orientation gizmo in the viewport's corner, and the snap a click on it starts.

use super::*;

// ---------------------------------------------------------------------------
// ViewCube: orientation gizmo, upper-right of the viewport.
// ---------------------------------------------------------------------------

const VIEWCUBE_SIZE: f32 = 90.0;
const VIEWCUBE_MARGIN: f32 = 12.0;
/// The gap between the two buttons under the cube, and the grid toggle's width.
const VIEWCUBE_BUTTON_GAP: f32 = 6.0;
const VIEWCUBE_GRID_BUTTON: f32 = 18.0;

/// Below this the viewport has no room to spare and the cube hides rather than sitting on top
/// of geometry there is already too little of.
pub(crate) const VIEWCUBE_MIN_VIEWPORT: f32 = 240.0;

/// Where a face's outer cells give way to its middle one, as a fraction of the half-extent.
///
/// This is the one number that decides how big the edge and corner targets read against the
/// faces, so it is named rather than buried: at 0.5 the middle cell is half the face across
/// and each outer band a quarter, which is about what a conventional view cube looks like. Raising it
/// makes faces easier to hit and corners harder.
const VIEWCUBE_CELL_EDGE: f32 = 0.5;

/// How long a click-to-snap takes.
pub(crate) const SNAP_SECONDS: f32 = 0.25;

/// One of the twenty-six regions of the cube: six faces, twelve edges, eight corners.
///
/// A sign vector with each component in `{-1, 0, +1}` and not all zero — which is exactly
/// twenty-six values, and splits into the three kinds by how many components are non-zero.
/// The same vector is both *where you clicked* and *where the camera goes*, and that is the
/// point of representing it this way: hit-testing and camera targeting cannot drift apart
/// because there is only one vocabulary between them.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct CubeRegion {
    pub(crate) sign: [i8; 3],
}

impl CubeRegion {
    /// The direction from the target toward the eye — the view direction reversed, since a
    /// camera looking at the front face stands in front of it.
    pub(crate) fn direction(self) -> Vector3<f32> {
        Vector3::new(self.sign[0] as f32, self.sign[1] as f32, self.sign[2] as f32).normalize()
    }

    /// Which way is up once the camera is there, before the camera has its say.
    ///
    /// World up, except where the view direction *is* world up: straight down and straight up
    /// leave the usual choice degenerate, its cross product with the view direction being
    /// zero. Those two take a fixed axis here, and `OrbitCamera::up_keeping_roll` then
    /// replaces it with the axis nearest the roll the camera already has -- snapped to an
    /// axis, which is what a *raw* derivation lacked when repeated Top/Bottom clicks slowly
    /// rotated the view. The fixed axis is what remains when the camera has no roll to keep.
    pub(crate) fn up(self) -> Vector3<f32> {
        if self.sign[0] == 0 && self.sign[1] == 0 {
            Vector3::y()
        } else {
            Vector3::z()
        }
    }

    /// The face label, for the six regions that are faces.
    pub(crate) fn label(self) -> Option<&'static str> {
        match self.sign {
            [-1, 0, 0] => Some("FRONT"),
            [1, 0, 0] => Some("BACK"),
            [0, 1, 0] => Some("LEFT"),
            [0, -1, 0] => Some("RIGHT"),
            [0, 0, 1] => Some("TOP"),
            [0, 0, -1] => Some("BOTTOM"),
            _ => None,
        }
    }
}

/// The six face normals, as regions.
pub(crate) fn view_cube_faces() -> [CubeRegion; 6] {
    [
        CubeRegion { sign: [-1, 0, 0] },
        CubeRegion { sign: [1, 0, 0] },
        CubeRegion { sign: [0, 1, 0] },
        CubeRegion { sign: [0, -1, 0] },
        CubeRegion { sign: [0, 0, 1] },
        CubeRegion { sign: [0, 0, -1] },
    ]
}

/// Face colouring per the CAD RGB=XYZ convention: X faces red, Y green, Z blue.
fn view_cube_colour(face: CubeRegion) -> egui::Color32 {
    let axis = face.sign.iter().position(|c| *c != 0).unwrap_or(0);
    match axis {
        0 => egui::Color32::from_rgba_unmultiplied(196, 64, 64, 128),
        1 => egui::Color32::from_rgba_unmultiplied(70, 160, 80, 128),
        _ => egui::Color32::from_rgba_unmultiplied(64, 110, 196, 128),
    }
}

/// The span an outer/middle/outer band occupies along one in-plane axis.
fn view_cube_band(step: i8) -> (f32, f32) {
    let t = VIEWCUBE_CELL_EDGE;
    match step {
        -1 => (-1.0, -t),
        0 => (-t, t),
        _ => (t, 1.0),
    }
}

/// A face, cut into the nine cells that name its regions.
///
/// The cube is drawn out of exactly the quads that are hit-tested, rather than being drawn as
/// six faces and picked by classifying a raycast against them. Both routes give the same
/// twenty-six answers, but only this one cannot disagree with itself: the cell the cursor is
/// over is the cell that gets tinted, because they are the same quad. The rendering and
/// picking of this widget stay in one place, which is what §8 of the spec asks for.
pub(crate) fn view_cube_cells(face: CubeRegion) -> Vec<(CubeRegion, [Vector3<f32>; 4])> {
    let axis = face.sign.iter().position(|c| *c != 0).expect("a face has one non-zero axis");
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let outward = face.sign[axis] as f32;

    let mut cells = Vec::with_capacity(9);
    for a in [-1i8, 0, 1] {
        for b in [-1i8, 0, 1] {
            let (u0, u1) = view_cube_band(a);
            let (v0, v1) = view_cube_band(b);
            let corner = |cu: f32, cv: f32| {
                let mut p = Vector3::zeros();
                p[axis] = outward;
                p[u] = cu;
                p[v] = cv;
                p
            };
            let mut sign = [0i8; 3];
            sign[axis] = face.sign[axis];
            sign[u] = a;
            sign[v] = b;
            cells.push((
                CubeRegion { sign },
                [
                    corner(u0, v0),
                    corner(u1, v0),
                    corner(u1, v1),
                    corner(u0, v1),
                ],
            ));
        }
    }
    cells
}

pub(crate) fn view_cube_rect(viewport: egui::Rect) -> egui::Rect {
    let max = egui::pos2(
        viewport.max.x - VIEWCUBE_MARGIN,
        viewport.min.y + VIEWCUBE_MARGIN + VIEWCUBE_SIZE,
    );
    let min = egui::pos2(max.x - VIEWCUBE_SIZE, viewport.min.y + VIEWCUBE_MARGIN);
    egui::Rect::from_min_max(min, max)
}

/// Where the Home button sits: just left of the cube, on its top row.
pub(crate) fn view_cube_home_rect(cube: egui::Rect) -> egui::Rect {
    let size = 18.0;
    egui::Rect::from_min_size(
        egui::pos2(cube.min.x - size - 4.0, cube.min.y),
        egui::vec2(size, size),
    )
}

/// Where the projection toggle sits: under the cube, the left of a pair centred with the grid
/// toggle.
pub(crate) fn view_cube_projection_rect(cube: egui::Rect) -> egui::Rect {
    let size = egui::vec2(26.0, 18.0);
    let pair = size.x + VIEWCUBE_BUTTON_GAP + VIEWCUBE_GRID_BUTTON;
    egui::Rect::from_min_size(
        egui::pos2(cube.center().x - pair * 0.5, cube.max.y + 6.0),
        size,
    )
}

/// Where the grid toggle sits: under the cube, to the right of the projection toggle.
///
/// Beside the projection toggle because the two are answered together: the grid shows only in
/// an orthographic view, so the switch that makes it appear is the one next to it.
pub(crate) fn view_cube_grid_rect(cube: egui::Rect) -> egui::Rect {
    let projection = view_cube_projection_rect(cube);
    egui::Rect::from_min_size(
        egui::pos2(projection.max.x + VIEWCUBE_BUTTON_GAP, projection.min.y),
        egui::vec2(VIEWCUBE_GRID_BUTTON, projection.height()),
    )
}

/// The cube and everything that hangs off it, as one rectangle.
///
/// What the viewport asks before treating a press as its own. The buttons under and beside
/// the cube lie outside the cube's square, so a press on one of them used to be a press on the
/// viewport as well — the projection toggle also picked whatever body sat behind it, and the
/// grid toggle would also have placed a point while a line was being placed.
pub(crate) fn view_cube_controls_rect(viewport: egui::Rect) -> egui::Rect {
    let cube = view_cube_rect(viewport);
    cube.union(view_cube_home_rect(cube))
        .union(view_cube_projection_rect(cube))
        .union(view_cube_grid_rect(cube))
}

/// A "camera" with the main camera's exact rotation (same eye-to-target direction and up
/// vector) but fixed at a constant distance from the origin — so the cube it renders always
/// stays world-axis-aligned while visibly tracking every tumble/twist of the main view.
fn view_cube_view_proj(camera: &OrbitCamera) -> Matrix4<f32> {
    const DIST: f32 = 3.0;
    // Orthographic, not perspective: a ViewCube is an orientation indicator, not a scene
    // render, and perspective foreshortening on such a small/close widget just reads as
    // distortion. Half-extent 1.6 comfortably fits the unit cube (half-extent 1) at any
    // rotation, including corner-on views, without clipping.
    const ORTHO_HALF: f32 = 1.6;
    let dir = (camera.eye - camera.target).normalize();
    let eye = Point3::origin() + dir * DIST;
    let view = Matrix4::look_at_rh(&eye, &Point3::origin(), &camera.up);
    let proj = Matrix4::new_orthographic(
        -ORTHO_HALF,
        ORTHO_HALF,
        -ORTHO_HALF,
        ORTHO_HALF,
        0.1,
        10.0,
    );
    proj * view
}

/// Projects a ViewCube-local point to a screen position within `rect`, plus a depth value
/// (larger = farther) for back-to-front painting.
fn view_cube_project(vp: &Matrix4<f32>, rect: egui::Rect, p: Vector3<f32>) -> Option<(egui::Pos2, f32)> {
    let clip = vp * Vector4::new(p.x, p.y, p.z, 1.0);
    if clip.w <= 1e-4 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let depth = clip.z / clip.w;
    let x = rect.min.x + (ndc_x * 0.5 + 0.5) * rect.width();
    let y = rect.min.y + (1.0 - (ndc_y * 0.5 + 0.5)) * rect.height();
    Some((egui::pos2(x, y), depth))
}

fn point_in_convex_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    let mut sign = 0.0_f32;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let edge = b - a;
        let to_p = p - a;
        let cross = edge.x * to_p.y - edge.y * to_p.x;
        if cross.abs() > 1e-6 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// One cell of the cube, ready to paint and to test against the cursor.
struct CubeCell {
    region: CubeRegion,
    face: CubeRegion,
    screen: [egui::Pos2; 4],
    depth: f32,
}

/// Every cell of every face turned toward the viewer, farthest first.
///
/// Back-to-front so that nearer cells paint over farther ones and, walked in reverse, win the
/// cursor: what is on top is what gets clicked.
fn view_cube_cells_on_screen(camera: &OrbitCamera, rect: egui::Rect) -> Vec<CubeCell> {
    let vp = view_cube_view_proj(camera);
    let toward_viewer = (camera.eye - camera.target).normalize();

    // A small positive threshold, not just `<= 0.0`: right at edge-on, a face's corners
    // project to a near-zero-width sliver, and the outline stroke's miter-join math blows up
    // on that degenerate shape, producing a spike far outside the polygon. Not drawing a face
    // slightly before it goes perfectly edge-on avoids that entirely, and is imperceptible.
    const EDGE_ON_EPSILON: f32 = 0.05;

    let mut out = Vec::with_capacity(54);
    for face in view_cube_faces() {
        let normal = face.direction();
        if normal.dot(&toward_viewer) <= EDGE_ON_EPSILON {
            continue;
        }
        for (region, corners) in view_cube_cells(face) {
            let mut screen = [egui::Pos2::ZERO; 4];
            let mut depth_sum = 0.0;
            let mut ok = true;
            for (i, c) in corners.iter().enumerate() {
                match view_cube_project(&vp, rect, *c) {
                    Some((p, d)) => {
                        screen[i] = p;
                        depth_sum += d;
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                out.push(CubeCell {
                    region,
                    face,
                    screen,
                    depth: depth_sum / 4.0,
                });
            }
        }
    }
    out.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    out
}

/// Draws the ViewCube and handles interaction with it. Passively tracks the main camera's
/// orientation every frame; clicking any of its twenty-six regions swings to that view, and
/// dragging the cube itself orbits the camera around its current target (a simple spin, not
/// POI-anchored like the main viewport's LMB tumble — the cube has no scene geometry of its
/// own to pick a pivot from).
///
/// Returns the region clicked this frame, if one was.
///
/// The widget only has the camera, and clicking a *face* now means more than orienting: it sets
/// the plane new curves are drawn on. Rather than hand it the whole application, it says what
/// was clicked and the caller decides what that means — which also keeps "what a face click
/// does" in one place instead of two.
pub(crate) fn show_view_cube(
    ui: &egui::Ui,
    camera: &mut OrbitCamera,
    viewport: egui::Rect,
    grid: &mut bool,
) -> Option<CubeRegion> {
    if viewport.width() < VIEWCUBE_MIN_VIEWPORT || viewport.height() < VIEWCUBE_MIN_VIEWPORT {
        return None;
    }
    let rect = view_cube_rect(viewport);
    let home = view_cube_home_rect(rect);
    let projection_rect = view_cube_projection_rect(rect);
    let grid_rect = view_cube_grid_rect(rect);
    let response = ui.interact(
        rect,
        ui.id().with("view_cube"),
        egui::Sense::click_and_drag(),
    );
    let home_response = ui.interact(home, ui.id().with("view_cube_home"), egui::Sense::click());
    let projection_response = ui
        .interact(
            projection_rect,
            ui.id().with("view_cube_projection"),
            egui::Sense::click(),
        )
        .on_hover_text(format!(
            "{} — click for {}",
            camera.projection.label(),
            camera.projection.toggled().label().to_lowercase(),
        ));
    let grid_response = ui
        .interact(grid_rect, ui.id().with("view_cube_grid"), egui::Sense::click())
        .on_hover_text(match (*grid, camera.projection) {
            (false, _) => "Grid hidden — click to show it on the drawing plane",
            (true, Projection::Orthographic) => {
                "Grid on the drawing plane — click to hide. Shift snaps to it"
            }
            (true, Projection::Perspective) => {
                "Grid on, shown only in an orthographic view — the toggle to the left. \
                 Click to hide"
            }
        });

    // Only react to start/continue/stop of *this widget's own* drag — never a blanket
    // "not dragging => end_tumble()", since that would also fire (and wrongly cancel) a
    // tumble the main viewport is mid-drag on, given this runs every frame regardless.
    if response.drag_started() {
        let target = camera.target;
        camera.begin_tumble(target);
    }
    if response.dragged() {
        camera.tumble_drag(response.drag_delta());
    }
    if response.drag_stopped() {
        camera.end_tumble();
    }

    // Clipped generously enough to take in the Home button, which sits outside the cube's own
    // square. No backdrop panel — just the cube and its icon over the viewport.
    let painter = ui
        .painter()
        .with_clip_rect(rect.union(home).union(projection_rect).union(grid_rect).expand(4.0));

    let cells = view_cube_cells_on_screen(camera, rect);

    // Nearest first when asking what the cursor is over, since painting went the other way:
    // the cell drawn last is the one visibly on top, and so the one a click means.
    let hovered = response.hover_pos().and_then(|pos| {
        cells
            .iter()
            .rev()
            .find(|c| point_in_convex_polygon(pos, &c.screen))
            .map(|c| c.region)
    });

    for cell in &cells {
        let base = view_cube_colour(cell.face);
        let fill = if Some(cell.region) == hovered {
            egui::Color32::from_rgba_unmultiplied(255, 214, 64, 190)
        } else {
            base
        };
        painter.add(egui::Shape::convex_polygon(
            cell.screen.to_vec(),
            fill,
            egui::Stroke::new(0.6_f32, egui::Color32::from_black_alpha(90)),
        ));
    }

    // The label goes on after every cell of that face, so the grid lines do not run through
    // the text.
    for cell in &cells {
        let Some(label) = cell.region.label() else { continue };
        let centre = egui::pos2(
            cell.screen.iter().map(|p| p.x).sum::<f32>() / 4.0,
            cell.screen.iter().map(|p| p.y).sum::<f32>() / 4.0,
        );
        painter.text(
            centre,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(9.0),
            egui::Color32::from_white_alpha(220),
        );
    }

    // The face outlines, over the cells, so the cube still reads as a cube rather than as a
    // grid of tiles.
    for face in view_cube_faces() {
        let normal = face.direction();
        if normal.dot(&(camera.eye - camera.target).normalize()) <= 0.05 {
            continue;
        }
        let vp = view_cube_view_proj(camera);
        let axis = face.sign.iter().position(|c| *c != 0).unwrap_or(0);
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let outward = face.sign[axis] as f32;
        let corner = |cu: f32, cv: f32| {
            let mut p = Vector3::zeros();
            p[axis] = outward;
            p[u] = cu;
            p[v] = cv;
            p
        };
        let outline: Option<Vec<egui::Pos2>> =
            [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)]
                .iter()
                .map(|c| view_cube_project(&vp, rect, *c).map(|(p, _)| p))
                .collect();
        if let Some(points) = outline {
            for i in 0..4 {
                painter.line_segment(
                    [points[i], points[(i + 1) % 4]],
                    egui::Stroke::new(1.2_f32, egui::Color32::from_black_alpha(190)),
                );
            }
        }
    }

    // Home: a house, drawn rather than typed, so it does not depend on the font having the
    // glyph.
    let house = if home_response.hovered() {
        egui::Color32::from_rgb(255, 214, 64)
    } else {
        egui::Color32::from_white_alpha(200)
    };
    let c = home.center();
    let s = home.width() * 0.32;
    painter.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(c.x, c.y - s * 1.2),
            egui::pos2(c.x + s * 1.2, c.y),
            egui::pos2(c.x - s * 1.2, c.y),
        ],
        house,
        egui::Stroke::NONE,
    ));
    painter.rect_filled(
        egui::Rect::from_min_max(
            egui::pos2(c.x - s * 0.7, c.y),
            egui::pos2(c.x + s * 0.7, c.y + s),
        ),
        0.0,
        house,
    );

    // The projection toggle, under the cube. Its icon is the thing it switches to, drawn
    // rather than lettered: converging lines for perspective, parallel ones for orthographic.
    let is_ortho = camera.projection == Projection::Orthographic;
    let ink = if projection_response.hovered() {
        egui::Color32::from_rgb(255, 214, 64)
    } else {
        egui::Color32::from_white_alpha(200)
    };
    let p = projection_rect;
    let stroke = egui::Stroke::new(1.3_f32, ink);
    let (left, right) = (p.min.x + 3.0, p.max.x - 3.0);
    let (top, bottom) = (p.min.y + 4.0, p.max.y - 4.0);
    // Showing the *other* mode would be a button that lies about what it is: a toolbar icon
    // says what you are in, and the tooltip says what clicking does.
    let inset = if is_ortho { 0.0 } else { (p.width() - 6.0) * 0.22 };
    painter.line_segment([egui::pos2(left, top), egui::pos2(right, top)], stroke);
    painter.line_segment(
        [egui::pos2(left + inset, bottom), egui::pos2(right - inset, bottom)],
        stroke,
    );
    painter.line_segment([egui::pos2(left, top), egui::pos2(left + inset, bottom)], stroke);
    painter.line_segment([egui::pos2(right, top), egui::pos2(right - inset, bottom)], stroke);

    if projection_response.clicked() {
        camera.projection = camera.projection.toggled();
    }

    // The grid toggle, beside the projection toggle: a lattice of two lines each way. A
    // switch shows its state, so it is bright while the grid is on and dim while it is off —
    // and half-dim while the grid is on but the view is perspective, where it is not drawn,
    // which is the case that would otherwise read as the switch being broken.
    let ink = if grid_response.hovered() {
        egui::Color32::from_rgb(255, 214, 64)
    } else if *grid && is_ortho {
        egui::Color32::from_white_alpha(200)
    } else if *grid {
        egui::Color32::from_white_alpha(110)
    } else {
        egui::Color32::from_white_alpha(60)
    };
    let stroke = egui::Stroke::new(1.2_f32, ink);
    let lattice = grid_rect.shrink(3.0);
    painter.rect_stroke(lattice, 0.0, stroke, egui::StrokeKind::Middle);
    for t in [1.0 / 3.0, 2.0 / 3.0] {
        let x = lattice.min.x + lattice.width() * t;
        let y = lattice.min.y + lattice.height() * t;
        painter.line_segment(
            [egui::pos2(x, lattice.min.y), egui::pos2(x, lattice.max.y)],
            stroke,
        );
        painter.line_segment(
            [egui::pos2(lattice.min.x, y), egui::pos2(lattice.max.x, y)],
            stroke,
        );
    }
    if grid_response.clicked() {
        *grid = !*grid;
    }

    if home_response.clicked() {
        // Home is the standard isometric, framed on whatever is on screen. Framing reuses the
        // camera's own scene extent rather than a fixed fallback distance, so Home means the
        // same thing on a bracket and on an airframe.
        camera.begin_snap(CubeRegion { sign: [1, 1, 1] });
        camera.frame_scene();
    }
    if response.clicked()
        && let Some(region) = hovered
    {
        // Clicking a face does *not* switch the viewport to orthographic, which is a convention
        // some CAD tools follow — decided against deliberately rather than by omission, since
        // there is a projection toggle right below the cube and changing it from up here would
        // be a surprise.
        camera.begin_snap(region);
        return Some(region);
    }
    None
}
