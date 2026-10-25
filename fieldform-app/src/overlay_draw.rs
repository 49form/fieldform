// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What the viewport paints over the traced image, and the colours it paints in.

use super::*;

/// Curves are reference geometry, not editable cage geometry, so they are drawn in a colour
/// that reads as "input" rather than competing with the selection highlights.
pub(crate) const CURVE_COLOR: egui::Color32 = egui::Color32::from_rgb(210, 160, 90);
pub(crate) const CURVE_SELECTED_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 205, 120);
/// The B-rep edges of an import, when the original CAD is shown over the field it became.
///
/// Cool against the warm curve colours, because these are not editable and a curve is: the two
/// are both lines on the same screen and confusing them would mean reaching for a handle that
/// is not there.
const BREP_EDGE_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 190, 235);
/// The dashed polygon joining a curve's control points — the curve's own colour, faded, so it
/// reads as belonging to the curve while staying clearly behind it.
pub(crate) const CONTROL_POLYGON_COLOR: egui::Color32 =
    egui::Color32::from_rgba_premultiplied(105, 84, 49, 130);

/// Draws an import's B-rep edges over whatever is already on screen.
///
/// What makes an import read as CAD rather than as a mesh. The lines come from the same OCCT
/// pass the triangles do, so this is decoration on the tessellation rather than a second opinion
/// about where the surface is — the distinction `import.md` warns about, and the reason this is
/// a painter call rather than a second field.
///
/// Drawn segment by segment for the same reason `draw_curves` is: a point behind the camera
/// projects to a wild coordinate, and a whole polyline drawn at once would streak across the
/// viewport instead of simply breaking.
///
/// No depth test, which is deliberate rather than overlooked. There is no depth buffer in this
/// pass, so the edges show through the body — and on a machined part that reads as a wireframe
/// of the original, which is what is wanted from a toggle called "the original CAD". Hiding the
/// far ones would mean depth-testing against the traced field, which is a per-pixel question
/// this pass does not have the answer to.
pub(crate) fn draw_brep_edges(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    edges: &[Vec<Point3<f32>>],
) {
    let stroke = egui::Stroke::new(1.0_f32, BREP_EDGE_COLOR);
    for line in edges {
        let mut previous: Option<egui::Pos2> = None;
        for p in line {
            let current = camera.project_to_screen(aspect, rect, *p);
            if let (Some(a), Some(b)) = (previous, current) {
                painter.line_segment([a, b], stroke);
            }
            previous = current;
        }
    }
}

/// Draws every curve in the document as a projected polyline.
///
/// Screen-space line segments rather than GPU geometry: a curve has no thickness to shade,
/// it must stay visible when it passes behind the surface it is about to cut, and this way it
/// costs nothing to keep on screen while the field is being re-meshed.
pub(crate) fn draw_curves(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    curves: &[(CurveId, Vec<Point3<f32>>)],
    selected: &[CurveId],
    stale: &[CurveId],
) {
    for (id, points) in curves {
        // **A line that could not be re-derived is drawn as one.** An axis on a surface keeps its
        // last place when its body moves out from under it or its U and V stop reaching the
        // surface, and its panel says so — but the viewport drew it exactly like a line that is
        // current, so what was on screen was a shape the document no longer derives. Reported as
        // an axis that looked misaligned, which it was not: it was not being placed at all.
        //
        // Kept rather than hidden, which is the same judgement the panel makes: the line is still
        // where it was, and taking geometry off the screen because a number has gone out of range
        // is a bigger surprise than showing it in the colour of a warning.
        // **Dashed, not merely tinted.** The first attempt at this gave a stale axis its own
        // colour -- and every curve colour here is the same orange, so `rgb(230,160,60)` against
        // the ordinary `rgb(210,160,90)` said nothing at all. A dash is a difference in *shape*,
        // which reads at any size, against any background, and to anyone who does not separate
        // those hues.
        let dashed = stale.contains(id);
        let stroke = if dashed {
            egui::Stroke::new(1.5_f32, CURVE_COLOR)
        } else if selected.contains(id) {
            egui::Stroke::new(2.5_f32, CURVE_SELECTED_COLOR)
        } else {
            egui::Stroke::new(1.5_f32, CURVE_COLOR)
        };
        // Segments are drawn independently so a point behind the camera breaks the line
        // instead of projecting to a wild coordinate and streaking across the viewport.
        let mut previous: Option<egui::Pos2> = None;
        for p in points {
            let current = camera.project_to_screen(aspect, rect, *p);
            if let (Some(a), Some(b)) = (previous, current) {
                if dashed {
                    dashed_line(painter, a, b, stroke);
                } else {
                    painter.line_segment([a, b], stroke);
                }
            }
            previous = current;
        }
    }
}

/// Draws a polyline dashed, segment by segment, as `draw_curves` draws a stale axis -- for a
/// line the document does not hold and the viewport should not draw as though it did.
pub(crate) fn draw_dashed_polyline(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    points: &[Point3<f32>],
    stroke: egui::Stroke,
) {
    let mut previous: Option<egui::Pos2> = None;
    for p in points {
        let current = camera.project_to_screen(aspect, rect, *p);
        if let (Some(a), Some(b)) = (previous, current) {
            dashed_line(painter, a, b, stroke);
        }
        previous = current;
    }
}

/// Draws `a` to `b` as a dashed line.
///
/// **Counted, not walked to the end.** A projected length can be enormous -- a seed a long way
/// outside a small body puts its leader tens of thousands of pixels across -- and a dash every few
/// pixels along that is thousands of segments a frame, which is a stopped viewport. The cap is
/// well past the diagonal of any viewport, so a line that fits is dashed whole and one that does
/// not heads off the edge, which is what it looks like anyway.
fn dashed_line(painter: &egui::Painter, a: egui::Pos2, b: egui::Pos2, stroke: egui::Stroke) {
    const MOST: usize = 400;
    const STEP: f32 = 6.0;
    let span = b - a;
    let length = span.length();
    if length <= 1.0 {
        return;
    }
    let unit = span / length;
    let dashes = ((length / STEP).ceil() as usize).min(MOST);
    for i in 0..dashes {
        let at = i as f32 * STEP;
        let to = (at + STEP * 0.6).min(length);
        painter.line_segment([a + unit * at, a + unit * to], stroke);
    }
}

/// Draws a surface axis's seed: a ring where the seed is, and a leader to where the axis stands.
///
/// **Three numbers between 0 and 1 are not a place anybody can picture**, which is how it was
/// reported: "the seed X, Y and Z is a bit confusing, I can see that it does something ... is there
/// a way to display where this seed is". Drawn, it needs no explaining: the ring is the seed, the
/// leader runs to the surface point nearest it, and that *is* the rule the axis is placed by.
///
/// A ring rather than a disc, because the seed is usually inside the body and a filled dot would
/// read as something sitting on the surface. Dashed leader, so it is plainly a relationship and
/// not a piece of geometry.
pub(crate) fn draw_axis_seed(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    seed: Point3<f64>,
    stands: Point3<f64>,
) {
    // Cast where the camera wants them: every projected point in this file is `f32`, and the
    // document's are `f64` -- the one boundary the display crosses.
    let shrink = |p: Point3<f64>| Point3::new(p.x as f32, p.y as f32, p.z as f32);
    let (Some(a), Some(b)) = (
        camera.project_to_screen(aspect, rect, shrink(seed)),
        camera.project_to_screen(aspect, rect, shrink(stands)),
    ) else {
        return;
    };
    let colour = egui::Color32::from_rgba_unmultiplied(255, 190, 90, 220);
    // Dashed by hand: egui has no dashed stroke, and a leader drawn solid reads as a line of the
    // model rather than as a note about one.
    dashed_line(painter, a, b, egui::Stroke::new(1.0_f32, colour));
    painter.circle_stroke(a, 4.0_f32, egui::Stroke::new(1.5_f32, colour));
    // A small cross through the ring, so it is findable against a busy surface and reads as a
    // marker rather than as a hole.
    for (dx, dy) in [(3.0_f32, 0.0_f32), (0.0, 3.0)] {
        painter.line_segment(
            [a - egui::vec2(dx, dy), a + egui::vec2(dx, dy)],
            egui::Stroke::new(1.0_f32, colour),
        );
    }
}

/// Draws one surface's control mesh as a wireframe.
///
/// Segments independently, for the reason `draw_curves` gives: a point behind the camera
/// projects to nothing, and a line drawn to it streaks across the viewport.
pub(crate) fn draw_surface_wire(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    edges: &[[Point3<f32>; 2]],
) {
    let stroke = egui::Stroke::new(1.0_f32, SURFACE_COLOR);
    for [a, b] in edges {
        if let (Some(pa), Some(pb)) = (
            camera.project_to_screen(aspect, rect, *a),
            camera.project_to_screen(aspect, rect, *b),
        ) {
            painter.line_segment([pa, pb], stroke);
        }
    }
}

/// Draws the cage's edges, one pixel wide, so a face's extent is visible against its neighbours'
/// -- two coplanar faces shade exactly alike, and without their shared edge drawn a cage reads as
/// far sparser than it is. Asked for by the owner, 2026-10-24.
///
/// **Only the edges of faces turned toward the camera**, on a closed cage: the cage draws with no
/// depth buffer, so every edge drawn would be a wireframe with the far side showing through, and
/// facing is the same test picking uses to keep the far side off offer. An open mesh has two sides
/// and gets every edge, as its picking does. A fold that hides a facing face behind another still
/// shows that face's edges, which is the approximation accepted here.
pub(crate) fn draw_cage_edges(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    mesh: &QuadMesh,
    placement: &Similarity3<f64>,
) {
    let eye = camera.eye;
    let open = mesh.is_open();
    let mut drawn: std::collections::HashSet<(VertexId, VertexId)> = std::collections::HashSet::new();
    let stroke = egui::Stroke::new(1.0_f32, CAGE_EDGE_COLOR);
    for face in mesh.face_ids() {
        let v = mesh.face_vertices(face);
        let p: [Point3<f32>; 4] = [
            vertex_pos(mesh, placement, v[0]),
            vertex_pos(mesh, placement, v[1]),
            vertex_pos(mesh, placement, v[2]),
            vertex_pos(mesh, placement, v[3]),
        ];
        if !open {
            let normal = (p[2] - p[0]).cross(&(p[3] - p[1]));
            if normal.norm_squared() >= 1e-20 && normal.dot(&(eye - p[0])) <= 0.0 {
                continue;
            }
        }
        for i in 0..4 {
            let (a, b) = (v[i], v[(i + 1) % 4]);
            if !drawn.insert((a.min(b), a.max(b))) {
                continue;
            }
            if let (Some(pa), Some(pb)) = (
                camera.project_to_screen(aspect, rect, p[i]),
                camera.project_to_screen(aspect, rect, p[(i + 1) % 4]),
            ) {
                painter.line_segment([pa, pb], stroke);
            }
        }
    }
}

/// The cage's edges: darker than its faces and faint enough to read as structure, not as marks.
const CAGE_EDGE_COLOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(40, 48, 64, 160);

/// **Dimmer than a curve.** A surface is a grid of dozens of quads where a curve is one line,
/// so the same weight would make it the loudest thing on screen while being the thing least
/// often worked on — it is scaffolding for the lattice laid over it.
const SURFACE_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 140, 170);

/// Draws a gapped crosshair at the given screen position, marking the active tumble/dolly
/// pivot — same visual language as `_draw_poi` in the reference camera.
pub(crate) fn draw_poi_marker(painter: &egui::Painter, center: egui::Pos2) {
    const SIZE: f32 = 8.0;
    const GAP: f32 = 3.0;
    let stroke = egui::Stroke::new(2.0_f32, egui::Color32::from_rgba_unmultiplied(255, 153, 0, 230));
    let ticks = [
        (egui::vec2(-SIZE, 0.0), egui::vec2(-GAP, 0.0)),
        (egui::vec2(GAP, 0.0), egui::vec2(SIZE, 0.0)),
        (egui::vec2(0.0, -SIZE), egui::vec2(0.0, -GAP)),
        (egui::vec2(0.0, GAP), egui::vec2(0.0, SIZE)),
    ];
    for (from, to) in ticks {
        painter.line_segment([center + from, center + to], stroke);
    }
}

pub(crate) const SELECTION_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 210, 60);
/// A point or a plane that is in the part and is not the thing being edited.
///
/// **The same yellow, duller.** These two blocks draw nothing of their own -- a point encloses
/// no volume and a plane has no edges -- so without a mark there is a row in the list and empty
/// space beside it, which is what was reported. They are the same kind of thing as the selected
/// one and are shown the same way; only the brightness says which is which, so a point does not
/// change into a different-looking object at the moment it is clicked.
pub(crate) const GUIDE_COLOR: egui::Color32 = egui::Color32::from_rgb(196, 160, 48);
/// A plane's square when it is not selected. Fainter than the selected one for the same reason
/// that one is faint at all: it stands for something infinite.
pub(crate) const GUIDE_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(46, 38, 8, 26);
/// A selected plane's preview square. Faint, because it stands for something infinite and
/// should not read as a surface the model is behind.
pub(crate) const PLANE_PREVIEW_FILL: egui::Color32 =
    egui::Color32::from_rgba_premultiplied(70, 58, 12, 40);
pub(crate) const SELECTION_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(90, 75, 15, 70);
/// A cage tool's preview of what a click would make: the drawn part green, and the part the
/// cage adds for itself -- a ring propagated from a cut, the ring an edge would get -- cyan and
/// dashed, so what the person did not draw reads as such.
#[allow(dead_code)]
pub(crate) const PREVIEW_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 220, 110);
pub(crate) const PROPAGATED_COLOR: egui::Color32 = egui::Color32::from_rgb(80, 210, 230);
pub(crate) const HOVER_COLOR: egui::Color32 = egui::Color32::from_rgb(220, 225, 230);
pub(crate) const HOVER_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(60, 62, 65, 45);

/// Draws a face outline as independent line segments rather than relying on
/// `convex_polygon`'s built-in stroke. A polygon's stroke needs a miter-join computation at
/// each corner, which blows up (drawing a stray line off to infinity) when the polygon is
/// nearly degenerate — exactly what happens when a face is viewed edge-on. Independent
/// segments have no shared-joint math to blow up, so the outline stays correct (just very
/// thin) at any viewing angle instead of needing to hide the face near edge-on.
fn draw_face_highlight(
    painter: &egui::Painter,
    points: &[egui::Pos2],
    fill: egui::Color32,
    outline: egui::Color32,
    outline_width: f32,
) {
    painter.add(egui::Shape::convex_polygon(
        points.to_vec(),
        fill,
        egui::Stroke::NONE,
    ));
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        painter.line_segment([a, b], egui::Stroke::new(outline_width, outline));
    }
}

/// Highlights a vertex/edge/face: a dot on a vertex, a thick line on an edge, or an
/// outlined/filled quad on a face. Used for both the current selection (bright) and
/// whatever's under the cursor (dim), so the two need visually distinct colors/weights.
pub(crate) fn draw_highlight(
    painter: &egui::Painter,
    cage: &QuadMesh,
    placement: &Similarity3<f64>,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    selection: Selection,
    color: egui::Color32,
    fill: egui::Color32,
    weight: f32,
) {
    match selection {
        Selection::None => {}
        Selection::Vertex(v) => {
            if let Some(p) = camera.project_to_screen(aspect, rect, vertex_pos(cage, placement, v))
            {
                painter.circle_filled(p, weight * 2.5, color);
            }
        }
        Selection::Edge(a, b) => {
            let pa = camera.project_to_screen(aspect, rect, vertex_pos(cage, placement, a));
            let pb = camera.project_to_screen(aspect, rect, vertex_pos(cage, placement, b));
            if let (Some(pa), Some(pb)) = (pa, pb) {
                painter.line_segment([pa, pb], egui::Stroke::new(weight, color));
            }
        }
        Selection::Face(f) => {
            let corners = cage.face_vertices(f);
            let screen: Option<Vec<egui::Pos2>> = corners
                .iter()
                .map(|&v| camera.project_to_screen(aspect, rect, vertex_pos(cage, placement, v)))
                .collect();
            if let Some(points) = screen {
                draw_face_highlight(painter, &points, fill, color, weight);
            }
        }
    }
}
