// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Small readings of a cage: where a vertex is, its bounds, and why none is on screen.

use crate::*;

/// Document geometry is `f64` (D2); the camera, picking and highlight code work in `f32`,
/// where screen precision is a few thousand pixels and `f64` would buy nothing. This is the
/// single read-side cast, so the narrowing is explicit at every use rather than implicit.
pub(crate) fn vertex_pos(cage: &QuadMesh, placement: &Similarity3<f64>, v: VertexId) -> Point3<f32> {
    (placement * cage.vertex_position(v)).cast::<f32>()
}

/// Bounding-box center and radius (half the diagonal) of a cage's vertices, for `/`'s
/// re-center-on-object behavior.
pub(crate) fn cage_bounds(cage: &QuadMesh) -> (Point3<f32>, f32) {
    let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in cage.vertices() {
        let p = v.position.cast::<f32>();
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }
    let center = Point3::new(
        (min.x + max.x) * 0.5,
        (min.y + max.y) * 0.5,
        (min.z + max.z) * 0.5,
    );
    let radius = ((max - min).norm() * 0.5).max(0.1);
    (center, radius)
}

/// The smallest and largest face of a surface, as a span in millimetres.
///
/// **What a conformal lattice's cell size is actually set by**, which is the one thing about
/// that map a user cannot work out from the controls: there is no cell size to type, because
/// `s` and `t` run nought to one across every face however big it is. So the panel says the
/// range the mesh produces and lets the number be checked rather than believed.
///
/// A face's span is its mean edge length — not its diagonal and not its area, because the cells
/// are laid out along the edges and that is the direction they are counted in.
pub(crate) fn face_span_mm(mesh: &fieldform_cage::QuadMesh) -> Option<(f64, f64)> {
    let mut small = f64::INFINITY;
    let mut large: f64 = 0.0;
    for face in mesh.face_ids() {
        let v = mesh.face_vertices(face);
        let mut total = 0.0;
        for k in 0..4 {
            let a = mesh.vertex_position(v[k]);
            let b = mesh.vertex_position(v[(k + 1) % 4]);
            total += (b - a).norm();
        }
        let span = total / 4.0;
        small = small.min(span);
        large = large.max(span);
    }
    (small.is_finite() && large > 0.0).then_some((small, large))
}

/// Why no cage is on screen in a mode that would otherwise draw one.
///
/// Four different situations look identical — an empty viewport — and each has a different
/// answer, so the one thing not to do is say "no cage".
pub(crate) fn cage_absence_reason(app: &App) -> String {
    let graph = app.part_graph();
    let Some(node) = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Cage(c) if *c == app.active_cage))
        .map(|(id, _)| id)
    else {
        return "This part has no cage block — Create > Cage adds one".to_string();
    };
    if graph.roots().contains(&node) && !app.is_visible(node) {
        return "The cage block is hidden — switch its eye back on".to_string();
    }
    match app.body_containing(node) {
        Some(body) if !app.is_visible(body) => {
            "The block containing the cage is hidden — switch its eye back on".to_string()
        }
        _ => "Nothing draws the cage at the moment".to_string(),
    }
}

/// The cage's own extent, with a margin, for sizing its bake.
///
/// The margin matters rather than being tidiness: outside its bake the shader bounds the
/// distance using the distance to the box, and that bound is only useful while the surface
/// keeps clear of the faces.
pub(crate) fn cage_box(cage: &QuadMesh) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    if !lo.x.is_finite() {
        return (Point3::origin(), Point3::origin());
    }
    let margin = (hi - lo).norm() * 0.08;
    let m = Vector3::new(margin, margin, margin);
    (lo - m, hi + m)
}
