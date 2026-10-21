// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What a click on the control mesh lands on: the vertex, edge or face under the pointer.

use super::*;

/// The vertices on faces the camera can see, for a cage that folds over itself.
///
/// Picking a vertex or an edge is a screen-space test — project a handful of points and take the
/// nearest to the cursor — which is cheap and has no idea what is in front of what. On a cage
/// simple enough to be convex that is the same answer; on one that folds over itself it selects
/// through the model, and the vertex you meant is behind the one you got.
///
/// **A face's own facing, not true occlusion.** A quad whose normal points away from the eye is
/// on the far side of the surface it belongs to, so nothing on it should be reachable. That
/// keeps silhouette vertices — they touch a front face *and* a back one, and one is enough —
/// and rejects the whole of the far side, which is the case that goes wrong. It does not reject
/// a near-side vertex hidden behind a *different* part of the cage; that is true occlusion, it
/// needs a raycast per candidate, and the project notes' picking rule says what it would have to be
/// tested against. Backlog item 22 carries the rest.
///
/// The quad is normalled from its diagonals, which is right for a non-planar quad in a way that
/// picking three of its corners is not: a control cage's faces are routinely warped, and three
/// corners would answer with whichever three were chosen.
fn camera_facing_vertices(
    cage: &QuadMesh,
    placement: &Similarity3<f64>,
    eye: Point3<f32>,
) -> std::collections::HashSet<VertexId> {
    // **An open surface has two sides, and either may be the one looked at.** A cage is closed,
    // so a face turned away is the far side of the solid and its points are behind the near
    // side; a surface a lattice follows is a sheet, and seen from underneath every face turns
    // away while every point is in plain view. Facing says nothing there, so every point is on
    // offer and the occlusion test — which is two-sided — decides where the sheet folds.
    if cage.is_open() {
        return cage.vertex_ids().collect();
    }
    let mut out = std::collections::HashSet::new();
    for face in cage.face_ids() {
        let v = cage.face_vertices(face);
        let p: Vec<Point3<f32>> = v.iter().map(|id| vertex_pos(cage, placement, *id)).collect();
        let normal = (p[2] - p[0]).cross(&(p[3] - p[1]));
        // A degenerate quad has no side to be on, so it is left pickable rather than silently
        // dropped — the alternative is a hole in the picking wherever the cage is pinched.
        let towards = eye - p[0];
        if normal.norm_squared() < 1e-20 || normal.dot(&towards) > 0.0 {
            out.extend(v.iter().copied());
        }
    }
    out
}

/// Whether the cage itself stands between the eye and a point on it.
///
/// **Facing is not occlusion, and this is the difference.** A vertex on a face that points at the
/// camera is on the near side of *that* face, and can still be behind a different part of a cage
/// that folds over itself. Rejecting the far side was the cheap half of backlog item 22; this is
/// the other.
///
/// Tested against the **control mesh**, not the limit surface. A cage strictly encloses the
/// surface it describes, so the two disagree about what is hidden — and the project notes' picking
/// rule settles which to use: picking follows whatever the current mode draws, and cage mode
/// draws the control mesh.
///
/// The tolerance is what keeps a vertex from occluding itself. It lies *on* the faces that meet
/// there, so the ray toward it hits those faces at its own distance; anything nearer than that
/// by more than a whisker is something else in the way.
pub(crate) fn occluded(
    cage_mesh: &CageMesh,
    local_eye: Point3<f32>,
    local_point: Point3<f32>,
    tolerance: f32,
) -> bool {
    let toward = local_point - local_eye;
    let reach = toward.norm();
    if reach <= tolerance {
        return false;
    }
    match cage_mesh.raycast(local_eye, toward) {
        Some((hit, _)) => (hit - local_eye).norm() < reach - tolerance,
        None => false,
    }
}

/// Finds the closest cage vertex to `pos` (in the viewport's local pixel coordinates),
/// within `threshold` pixels.
fn pick_vertex(
    cage: &QuadMesh,
    placement: &Similarity3<f64>,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    pos: egui::Pos2,
    threshold: f32,
    visible: &std::collections::HashSet<VertexId>,
    occlusion: Option<(&CageMesh, Point3<f32>, f32)>,
) -> Option<VertexId> {
    // Everything near the cursor, nearest first, rather than only the nearest.
    //
    // **The order matters because occlusion is expensive.** A raycast is a pass over the
    // triangles, so asking it of every vertex is a pass per vertex — far too much for a click on
    // a cage of any size. Asking it only of the handful the cursor is actually over costs a
    // handful of passes, and the answer is the same: the nearest candidate that is not hidden.
    let mut near: Vec<(VertexId, f32)> = Vec::new();
    for id in cage.vertex_ids() {
        // Nothing on the far side of the surface, however near the cursor it projects.
        if !visible.contains(&id) {
            continue;
        }
        let Some(screen) = camera.project_to_screen(aspect, rect, vertex_pos(cage, placement, id))
        else {
            continue;
        };
        let dist = screen.distance(pos);
        if dist <= threshold {
            near.push((id, dist));
        }
    }
    near.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (id, _) in near {
        let Some((mesh, eye, tolerance)) = occlusion else {
            return Some(id);
        };
        if !occluded(mesh, eye, local_vertex(cage, id), tolerance) {
            return Some(id);
        }
    }
    None
}

/// A cage vertex in the cage's own space, which is where its mesh and its raycast live.
fn local_vertex(cage: &QuadMesh, id: VertexId) -> Point3<f32> {
    cage.vertex_position(id).cast::<f32>()
}

/// Distance from `p` to the 2D segment `a`-`b`.
fn point_segment_distance(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq < 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Finds the closest cage edge to `pos`, within `threshold` pixels.
fn pick_edge(
    cage: &QuadMesh,
    placement: &Similarity3<f64>,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    pos: egui::Pos2,
    threshold: f32,
    visible: &std::collections::HashSet<VertexId>,
    occlusion: Option<(&CageMesh, Point3<f32>, f32)>,
) -> Option<(VertexId, VertexId)> {
    let mut near: Vec<((VertexId, VertexId), f32)> = Vec::new();
    for (a, b) in cage.edges() {
        // An edge is reachable when either end is on a face the camera can see, which keeps
        // the silhouette: an edge along it has one front face and one back one.
        if !visible.contains(&a) && !visible.contains(&b) {
            continue;
        }
        let (Some(sa), Some(sb)) = (
            camera.project_to_screen(aspect, rect, vertex_pos(cage, placement, a)),
            camera.project_to_screen(aspect, rect, vertex_pos(cage, placement, b)),
        ) else {
            continue;
        };
        let dist = point_segment_distance(pos, sa, sb);
        if dist <= threshold {
            near.push(((a, b), dist));
        }
    }
    near.sort_by(|x, y| x.1.total_cmp(&y.1));
    for ((a, b), _) in near {
        let Some((mesh, eye, tolerance)) = occlusion else {
            return Some((a, b));
        };
        // Judged at its own midpoint. An end can be on the silhouette, where one of its faces
        // faces the camera and the other does not, and testing an end would then reject an edge
        // whose middle is plainly in view.
        let mid = nalgebra::center(&local_vertex(cage, a), &local_vertex(cage, b));
        if !occluded(mesh, eye, mid, tolerance) {
            return Some((a, b));
        }
    }
    None
}

/// Picks whatever's under `pos`: closest vertex within its threshold, else closest edge
/// within its (smaller) threshold, else whatever face a ray through `pos` hits. Vertices and
/// edges win over faces at the same screen position since they're the more precise target.
pub(crate) fn pick(
    cage: &QuadMesh,
    placement: &Similarity3<f64>,
    cage_mesh: &CageMesh,
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    pos: egui::Pos2,
) -> Selection {
    const VERTEX_THRESHOLD: f32 = 12.0;
    const EDGE_THRESHOLD: f32 = 8.0;

    // Computed once for both tests rather than per candidate: it is a pass over the faces, and
    // the two tests below would otherwise each make it.
    let visible = camera_facing_vertices(cage, placement, camera.eye);

    // The eye in the cage's own space, which is where the mesh and its raycast live — the same
    // move the face test below makes, and for the same reason: carrying the ray back costs one
    // inverse, and carrying the mesh forward would cost every triangle.
    let inverse_placement = placement.inverse();
    let eye_in_cage = (inverse_placement * camera.eye.cast::<f64>()).cast::<f32>();
    // A thousandth of the cage, which is far below anything a person can aim at and far above
    // the rounding in a ray-triangle test. It is what stops a vertex occluding itself: it lies
    // on the faces meeting there, so the ray toward it hits them at its own distance.
    let reach = cage
        .vertex_ids()
        .map(|id| cage.vertex_position(id).coords.norm() as f32)
        .fold(0.0f32, f32::max);
    let tolerance = (reach * 1e-3).max(1e-6);
    let occlusion = Some((cage_mesh, eye_in_cage, tolerance));

    if let Some(v) = pick_vertex(
        cage, placement, camera, aspect, rect, pos, VERTEX_THRESHOLD, &visible, occlusion,
    ) {
        return Selection::Vertex(v);
    }
    if let Some((a, b)) = pick_edge(
        cage, placement, camera, aspect, rect, pos, EDGE_THRESHOLD, &visible, occlusion,
    ) {
        return Selection::Edge(a, b);
    }
    // The face test goes the other way round from the two above. Those project a handful of
    // points forward into screen space, which is cheap; this one would have to transform every
    // triangle of the mesh. Carrying the *ray* back into the cage's own space instead costs one
    // inverse and leaves the mesh — and its raycast — untouched.
    let (origin, dir) = camera.pick_ray(aspect, pos, rect);
    let inverse = placement.inverse();
    let local_origin = (inverse * origin.cast::<f64>()).cast::<f32>();
    let local_dir = inverse.transform_vector(&dir.cast::<f64>()).cast::<f32>();
    if let Some((_point, face)) = cage_mesh.raycast(local_origin, local_dir) {
        return Selection::Face(face);
    }
    Selection::None
}
