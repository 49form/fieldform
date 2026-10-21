// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Removing triangles a mesh does not need, judged against the field it came from.
//!
//! # Why this can be exact here, and is a guess everywhere else
//!
//! A general mesh simplifier has only the mesh: it decides whether a collapse is acceptable by
//! how far the new vertex sits from the *planes of the triangles it replaced* — Garland and
//! Heckbert's quadric error, which is the right answer available to someone who has thrown the
//! surface away. This crate has not thrown it away. The field is still here, so a collapse can
//! be judged by asking the surface directly where the new vertex landed, and a vertex that
//! stays on the surface within the export tolerance is one nothing was lost by.
//!
//! That difference matters most exactly where a mesh is worst. A quadric measures against the
//! triangles, so a fringe of slivers standing off a knife edge measures as *agreeing with
//! itself* and survives; the field says it is not on the surface and it goes.
//!
//! # What it will not do
//!
//! It will not tear or fold the mesh. Two conditions are checked before every collapse and both
//! are refusals rather than repairs:
//!
//! - **The link condition.** An edge may collapse only when the vertices adjacent to both of its
//!   ends are exactly the vertices of the triangles being removed. Collapsing across a shortcut
//!   between two parts of the surface pinches it into something that is no longer a surface, and
//!   no amount of care afterwards can undo that.
//! - **No triangle may flip.** Every triangle that survives keeps facing the way it did. A
//!   collapse that turns one inside out has moved the surface through itself.
//!
//! Neither is an optimisation. A simplifier that skips them produces a mesh that looks finer
//! and is broken, which is worse than one that is merely heavy.

use std::collections::{HashMap, HashSet};

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::{FieldMesh, ScalarField};

/// What a simplification pass did, so a caller can say whether it was worth running.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Simplified {
    pub triangles_before: usize,
    pub triangles_after: usize,
    /// Collapses refused because the surface would have moved too far.
    pub refused_for_error: usize,
    /// Collapses refused because the mesh would have stopped being one.
    pub refused_for_topology: usize,
}

impl Simplified {
    /// The fraction of triangles removed, between 0 and 1.
    pub fn saved(&self) -> f64 {
        if self.triangles_before == 0 {
            0.0
        } else {
            1.0 - self.triangles_after as f64 / self.triangles_before as f64
        }
    }
}

/// Collapses every edge that can go without moving the surface more than `tolerance_mm`.
///
/// Sweeps until nothing more will collapse. Each sweep considers the shortest edges first, so
/// the slivers a dual contour leaves along a crease are the first to go and the large triangles
/// on a flat face are left alone — which is the shape of saving that was wanted, rather than a
/// uniform thinning that would soften everything equally.
pub fn simplify(
    mesh: &mut FieldMesh,
    field: &(dyn ScalarField + Sync),
    tolerance_mm: f64,
) -> Simplified {
    simplify_cancellable(mesh, field, tolerance_mm, &crate::Cancel::default())
        .expect("a token nobody holds is never set")
}

/// [`simplify`], or `None` if `cancel` was set first. The mesh is left part-reduced when that
/// happens, which is fine: a cancelled tessellation throws its mesh away.
pub(crate) fn simplify_cancellable(
    mesh: &mut FieldMesh,
    field: &(dyn ScalarField + Sync),
    tolerance_mm: f64,
    cancel: &crate::Cancel,
) -> Option<Simplified> {
    let before = mesh.triangles.len();
    let mut report = Simplified {
        triangles_before: before,
        triangles_after: before,
        refused_for_error: 0,
        refused_for_topology: 0,
    };
    if mesh.triangles.is_empty() {
        return Some(report);
    }

    // Where each vertex has ended up. A collapsed vertex points at the one it became, and
    // `resolve` follows the chain — so a vertex collapsed twice is not two vertices.
    let mut moved_to: Vec<u32> = (0..mesh.positions.len() as u32).collect();
    let mut live: Vec<[u32; 3]> = mesh.triangles.clone();
    let mut gone = vec![false; live.len()];

    for _ in 0..24 {
        let collapsed = sweep(
            mesh,
            field,
            tolerance_mm,
            &mut moved_to,
            &mut live,
            &mut gone,
            &mut report,
            cancel,
        );
        if cancel.is_cancelled() {
            return None;
        }
        if collapsed == 0 {
            break;
        }
    }

    // Rebuild, dropping the triangles that went and the vertices nothing points at any more.
    let mut kept: Vec<[u32; 3]> = Vec::with_capacity(live.len());
    for (i, t) in live.iter().enumerate() {
        if gone[i] {
            continue;
        }
        let t = [resolve(&moved_to, t[0]), resolve(&moved_to, t[1]), resolve(&moved_to, t[2])];
        if t[0] != t[1] && t[1] != t[2] && t[2] != t[0] {
            kept.push(t);
        }
    }

    let mut renumber: HashMap<u32, u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for t in &mut kept {
        for slot in t.iter_mut() {
            let fresh = *renumber.entry(*slot).or_insert_with(|| {
                positions.push(mesh.positions[*slot as usize]);
                normals.push(mesh.normals[*slot as usize]);
                positions.len() as u32 - 1
            });
            *slot = fresh;
        }
    }

    mesh.positions = positions;
    mesh.normals = normals;
    mesh.triangles = kept;
    report.triangles_after = mesh.triangles.len();
    Some(report)
}

/// Where a vertex has ended up after any collapses so far.
fn resolve(moved_to: &[u32], mut v: u32) -> u32 {
    while moved_to[v as usize] != v {
        v = moved_to[v as usize];
    }
    v
}

/// One pass over the edges, shortest first. Returns how many collapsed.
fn sweep(
    mesh: &mut FieldMesh,
    field: &(dyn ScalarField + Sync),
    tolerance_mm: f64,
    moved_to: &mut [u32],
    live: &[[u32; 3]],
    gone: &mut [bool],
    report: &mut Simplified,
    cancel: &crate::Cancel,
) -> usize {
    // Which triangles touch each vertex, over what is still there.
    let mut around: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, t) in live.iter().enumerate() {
        if gone[i] {
            continue;
        }
        for &v in t {
            around.entry(resolve(moved_to, v)).or_default().push(i);
        }
    }

    // Candidate edges, shortest first: the slivers a crease leaves are the ones worth removing,
    // and a long edge on a flat face is already doing its job.
    let mut edges: Vec<(f64, u32, u32)> = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for (i, t) in live.iter().enumerate() {
        if gone[i] {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (resolve(moved_to, t[k]), resolve(moved_to, t[(k + 1) % 3]));
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            if seen.insert(key) {
                let d = (mesh.positions[a as usize] - mesh.positions[b as usize]).norm();
                edges.push((d, key.0, key.1));
            }
        }
    }
    edges.sort_by(|x, y| x.0.total_cmp(&y.0));

    let mut collapsed = 0;
    let mut touched: HashSet<u32> = HashSet::new();
    for (n, (_, a, b)) in edges.into_iter().enumerate() {
        // Every thousand edges rather than every one: a collapse is a handful of field queries,
        // so a thousand of them is still a small fraction of a second.
        if n % 1024 == 0 && cancel.is_cancelled() {
            return collapsed;
        }
        // **Nothing in an accepted collapse's neighbourhood may move again this sweep.**
        //
        // Not just the two ends: every vertex of every triangle the collapse touched. Those
        // triangles were *checked* against the field in their new shape, and moving one of
        // their corners afterwards invalidates a check that has already been passed. Guarding
        // only the two ends left that hole, and it is what let a mesh claiming 0.1 mm stray
        // 4.6 mm — each collapse legal on its own, and the neighbours' approvals quietly
        // expired.
        //
        // A sweep therefore accepts fewer collapses, and the sweeps repeat until nothing more
        // will go, which recovers the throughput without recovering the error.
        if touched.contains(&a) || touched.contains(&b) {
            continue;
        }
        let (a, b) = (resolve(moved_to, a), resolve(moved_to, b));
        if a == b {
            continue;
        }
        let (Some(fa), Some(fb)) = (around.get(&a), around.get(&b)) else { continue };

        // Where the merged vertex goes: onto the surface, from the midpoint.
        //
        // One Newton step, which is exact for a plane and close for anything flatter than the
        // edge is long. The check below is what decides whether close was close enough, so this
        // only has to be a good guess rather than a solution.
        let mid = Point3::from(
            (mesh.positions[a as usize].coords + mesh.positions[b as usize].coords) * 0.5,
        );
        let value = field.distance(mid);
        let normal = crate::mesh::surface_normal(field, mid, tolerance_mm.max(1e-6) * 0.5);
        let target = mid - normal * value;

        // **The field decides, and it is asked between the vertices as well as at them.**
        //
        // A quadric would ask the triangles being removed, which agree with themselves even
        // when they are a fringe standing off the true surface. The field says where the
        // surface actually is.
        //
        // But asking only where the new vertex lands is not enough, and a test caught it: every
        // vertex of a tetrahedron inscribed in a sphere sits exactly on that sphere, so a check
        // at the vertices alone permits simplifying a sphere to four triangles. What moves is
        // the surface *between* them. So each triangle that would survive is sampled at its
        // centroid and its edge midpoints, which is where a flat triangle strays furthest from
        // a curved surface.
        let ring: Vec<usize> = {
            let mut r: Vec<usize> = fa.iter().chain(fb.iter()).copied().collect();
            r.sort_unstable();
            r.dedup();
            r.into_iter().filter(|i| !gone[*i]).collect()
        };
        let dying: Vec<usize> = ring
            .iter()
            .copied()
            .filter(|i| {
                let v: Vec<u32> =
                    live[*i].iter().map(|x| resolve(moved_to, *x)).collect();
                v.contains(&a) && v.contains(&b)
            })
            .collect();

        let at = |v: u32| {
            let v = resolve(moved_to, v);
            if v == a || v == b { target } else { mesh.positions[v as usize] }
        };
        let mut strays = field.distance(target).abs() > tolerance_mm;
        for &i in &ring {
            // `continue`, not `break`. A dying triangle is simply not checked — it is going
            // away — but the ones after it in the ring still must be, and breaking here skipped
            // every triangle past the first casualty. That single word is what let a mesh
            // claiming 0.1 mm stray 4.1 mm: most of each collapse's neighbourhood was never
            // looked at.
            if dying.contains(&i) {
                continue;
            }
            if strays {
                break;
            }
            let t = live[i];
            let (p0, p1, p2) = (at(t[0]), at(t[1]), at(t[2]));
            strays = !within_tolerance(field, p0, p1, p2, tolerance_mm);
        }
        if strays {
            report.refused_for_error += 1;
            continue;
        }

        if !link_condition(mesh, live, moved_to, fa, fb, a, b, &dying)
            || flips(mesh, live, moved_to, &ring, &dying, a, b, target)
        {
            report.refused_for_topology += 1;
            continue;
        }

        for i in dying {
            gone[i] = true;
        }
        // **The surviving vertex takes the position the check approved.**
        //
        // Without this the collapse keeps `a` where it was, which is somewhere the check never
        // looked — up to half an edge from the point it validated. A test caught it: at a
        // tolerance of zero a vertex still moved 0.025 mm, which is the distance between the
        // point that was approved and the point that was kept.
        mesh.positions[a as usize] = target;
        mesh.normals[a as usize] =
            crate::mesh::surface_normal(field, target, tolerance_mm.max(1e-6) * 0.5);
        moved_to[b as usize] = a;
        touched.insert(a);
        touched.insert(b);
        for &i in &ring {
            for &v in &live[i] {
                touched.insert(resolve(moved_to, v));
            }
        }
        collapsed += 1;
    }

    collapsed
}

/// Swaps the diagonal of adjacent triangle pairs where doing so makes them less like slivers.
///
/// # Why shape is worth a pass of its own
///
/// A dual contour puts one vertex per cell and joins them across edges, so its triangles inherit
/// the grid's shape rather than the surface's. At a level change or a crease that produces long
/// thin ones, and a wireframe of slivers reads as an unconsidered mesh at the same triangle count
/// as a wireframe of near-equilateral ones. It also matters beyond looks: a sliver carries a
/// poorly conditioned normal, and anything integrating over the mesh weights it badly.
///
/// A flip changes only *which* diagonal joins four points. It moves no vertex, so it cannot move
/// the surface — but the two new triangles are not the two old ones, and a flat pair replacing a
/// folded pair can cut a corner the old pair followed. So each flip is still checked against the
/// field, by the same [`within_tolerance`] a collapse uses.
///
/// Refuses, in every case, rather than repairing afterwards:
///
/// - an edge not shared by exactly two triangles, since a flip is only defined for a pair;
/// - a flip whose new edge already exists, which would give two triangles the same three corners;
/// - a flip that turns either triangle over, or that leaves the pair no better than it found it.
///
/// Returns how many flips were made.
pub fn improve_shape(
    mesh: &mut FieldMesh,
    field: &(dyn ScalarField + Sync),
    tolerance_mm: f64,
) -> usize {
    improve_shape_cancellable(mesh, field, tolerance_mm, &crate::Cancel::default())
        .expect("a token nobody holds is never set")
}

/// [`improve_shape`], or `None` if `cancel` was set first. Checked between flip passes, of
/// which there are a handful and the first does most of the work.
pub(crate) fn improve_shape_cancellable(
    mesh: &mut FieldMesh,
    field: &(dyn ScalarField + Sync),
    tolerance_mm: f64,
    cancel: &crate::Cancel,
) -> Option<usize> {
    let mut total = 0;
    // Converges on its own — on the measured document the passes made 11774, 2449, 450, 71, 0.
    // The cap is here so a pathological mesh cannot spin, not because it is normally reached.
    for _ in 0..8 {
        if cancel.is_cancelled() {
            return None;
        }
        let made = flip_pass(mesh, field, tolerance_mm);
        if made == 0 {
            break;
        }
        total += made;
    }
    Some(total)
}

/// The smallest angle in a triangle, in radians. Sixty degrees is equilateral; near zero is a
/// sliver.
fn smallest_angle(a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> f64 {
    let mut worst = std::f64::consts::PI;
    for (p, q, r) in [(a, b, c), (b, c, a), (c, a, b)] {
        let (Some(u), Some(v)) = ((q - p).try_normalize(1e-12), (r - p).try_normalize(1e-12))
        else {
            return 0.0;
        };
        worst = worst.min(u.dot(&v).clamp(-1.0, 1.0).acos());
    }
    worst
}

/// Whether a triangle faces the way the field says out is.
///
/// **The old pair's normals are not the authority on this, and using them cost twice as many
/// inward-facing triangles as the mesh started with.** Along a knife edge a sliver's normal is
/// noise — the three corners are nearly collinear, so the cross product points wherever the
/// rounding fell. Two of those agreeing with each other says nothing, and a flip that inherits
/// their answer inherits the noise. The field's gradient at the new triangle's own centre is
/// defined regardless of what shape the triangles around it are in, and it is what the export
/// check measures winding against, so it is what a flip is held to.
fn faces_outward(field: &(dyn ScalarField + Sync), a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> bool {
    let normal = (b - a).cross(&(c - a));
    let centre = Point3::from((a.coords + b.coords + c.coords) / 3.0);
    // Stepped by the triangle's own size: a fixed step reads as noise on a triangle far smaller
    // than it, and cuts the corner on one far larger.
    let h = (normal.norm().sqrt() * 0.25).clamp(1e-4, 1.0);
    let d = |o: Vector3<f64>| field.distance(centre + o);
    let grad = Vector3::new(
        d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
        d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
        d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
    );
    normal.dot(&grad) > 0.0
}

/// Above this smallest angle a pair is left alone. Thirty degrees is half of equilateral.
const WORTH_FLIPPING: f64 = 30.0 * std::f64::consts::PI / 180.0;

fn flip_pass(mesh: &mut FieldMesh, field: &(dyn ScalarField + Sync), tolerance_mm: f64) -> usize {
    // Which triangles each edge belongs to, and which vertices are already joined.
    let mut of_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut joined: HashSet<(u32, u32)> = HashSet::new();
    for (i, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            of_edge.entry(key).or_default().push(i);
            joined.insert(key);
        }
    }

    let mut made = 0;
    // One flip per triangle per pass. A triangle changed by one flip has stale neighbours for
    // the next, and flipping against stale adjacency is how a mesh acquires two triangles with
    // the same corners.
    let mut settled = vec![false; mesh.triangles.len()];

    for ((a, c), faces) in &of_edge {
        if faces.len() != 2 {
            continue;
        }
        let (i, j) = (faces[0], faces[1]);
        if settled[i] || settled[j] {
            continue;
        }
        // **Which triangle traverses the shared edge which way decides the new winding**, and
        // ignoring it was worth six times more inward-facing triangles than the mesh started
        // with. In a consistently wound surface one of the pair runs `a → c` and the other
        // `c → a`; naming them by that rather than by which came first is what makes the two
        // new triangles come out facing the way the old pair did.
        let traverses = |t: [u32; 3], from: u32, to: u32| {
            (0..3).any(|k| t[k] == from && t[(k + 1) % 3] == to)
        };
        let (first, second) = if traverses(mesh.triangles[i], *a, *c) {
            (i, j)
        } else if traverses(mesh.triangles[j], *a, *c) {
            (j, i)
        } else {
            // Neither runs a → c: the pair is not consistently wound here, and flipping across
            // an inconsistency would bake it in rather than mend it.
            continue;
        };
        let Some(b) = mesh.triangles[first].iter().copied().find(|v| v != a && v != c) else {
            continue;
        };
        let Some(d) = mesh.triangles[second].iter().copied().find(|v| v != a && v != c) else {
            continue;
        };
        if b == d || joined.contains(&(b.min(d), b.max(d))) {
            continue;
        }

        let p = |v: u32| mesh.positions[v as usize];
        let before = smallest_angle(p(*a), p(b), p(*c)).min(smallest_angle(p(*a), p(*c), p(d)));
        // **Only a pair that is actually poorly shaped is worth the field queries.** A flip can
        // only repair a quad whose diagonal was chosen badly; where both triangles are already
        // decent there is nothing to win, and checking anyway is what made this pass cost as
        // much as the mesher that produced the input.
        if before > WORTH_FLIPPING {
            continue;
        }
        let after = smallest_angle(p(b), p(*a), p(d)).min(smallest_angle(p(b), p(d), p(*c)));
        if after <= before {
            continue;
        }

        // The quad's boundary runs b → a → d → c, so a fan from `b` keeps the winding the pair
        // already had. Nothing is negotiated here: the order follows from the traversal above.
        let one = [b, *a, d];
        let two = [b, d, *c];

        // A flip moves no vertex, but the two new triangles are not the two old ones — a flat
        // pair replacing a folded one cuts a corner the old pair followed. Same check a collapse
        // makes, for the same reason.
        if !within_tolerance(field, p(one[0]), p(one[1]), p(one[2]), tolerance_mm)
            || !within_tolerance(field, p(two[0]), p(two[1]), p(two[2]), tolerance_mm)
        {
            continue;
        }
        if !faces_outward(field, p(one[0]), p(one[1]), p(one[2]))
            || !faces_outward(field, p(two[0]), p(two[1]), p(two[2]))
        {
            continue;
        }

        mesh.triangles[first] = one;
        mesh.triangles[second] = two;
        // **Within one pass, two pairs can create the same edge**, which leaves it shared by
        // four triangles — non-manifold, and it took the export from 58 such edges to 272 before
        // this line existed. `settled` stops a *triangle* being touched twice; the new edge has
        // to be recorded for the same reason.
        joined.insert((b.min(d), b.max(d)));
        settled[first] = true;
        settled[second] = true;
        made += 1;
    }
    made
}

/// Whether a triangle stays within `tolerance_mm` of the field across its whole area.
///
/// **The sampling has to scale with the triangle, and a fixed set of probes was the bug.**
/// Checking the centroid and the three edge midpoints bounds nothing: as a collapse makes
/// triangles larger those four points sit further apart, the bulge between them grows unseen,
/// and each accepted collapse compounds the last. Measured on a real document it produced a
/// mesh straying 6.3 mm from the field while claiming a tolerance of 0.1 mm — a 95% saving that
/// was not a saving but a different, wrong shape.
///
/// So the grid is chosen from the triangle's own size: enough rows that no two samples are more
/// than about a tolerance apart along an edge. A triangle the size of the tolerance costs one
/// sample; a triangle a hundred times that costs a hundredth of its area in samples, which is
/// the price of the claim being true.
///
/// Capped, because a collapse that would need more than this many samples to justify is one
/// worth refusing outright: past that size the triangle is no longer approximating the surface,
/// it is replacing it.
fn within_tolerance(
    field: &(dyn ScalarField + Sync),
    p0: Point3<f64>,
    p1: Point3<f64>,
    p2: Point3<f64>,
    tolerance_mm: f64,
) -> bool {
    const MOST_ROWS: usize = 12;
    let longest = (p1 - p0).norm().max((p2 - p1).norm()).max((p0 - p2).norm());
    // One row per tolerance of edge, so neighbouring samples are about a tolerance apart.
    let rows = ((longest / tolerance_mm.max(1e-9)).ceil() as usize).clamp(1, MOST_ROWS + 1);
    if rows > MOST_ROWS {
        return false;
    }
    for i in 0..=rows {
        for j in 0..=(rows - i) {
            let (u, v) = (i as f64 / rows as f64, j as f64 / rows as f64);
            let p = p0 + (p1 - p0) * u + (p2 - p0) * v;
            if field.distance(p).abs() > tolerance_mm {
                return false;
            }
        }
    }
    true
}

/// Whether collapsing `a`-`b` keeps the mesh a surface.
///
/// The vertices shared by the two ends must be exactly the vertices opposite the triangles being
/// removed. Anything more is a shortcut across the surface, and collapsing along it pinches two
/// separate sheets together.
#[allow(clippy::too_many_arguments)]
fn link_condition(
    _mesh: &FieldMesh,
    live: &[[u32; 3]],
    moved_to: &[u32],
    fa: &[usize],
    fb: &[usize],
    a: u32,
    b: u32,
    dying: &[usize],
) -> bool {
    let ring_of = |faces: &[usize], v: u32| -> HashSet<u32> {
        let mut out = HashSet::new();
        for &i in faces {
            for &x in &live[i] {
                let x = resolve(moved_to, x);
                if x != v {
                    out.insert(x);
                }
            }
        }
        out
    };
    let mut shared = ring_of(fa, a);
    shared.retain(|x| ring_of(fb, b).contains(x));
    shared.remove(&a);
    shared.remove(&b);

    let mut opposite: HashSet<u32> = HashSet::new();
    for &i in dying {
        for &x in &live[i] {
            let x = resolve(moved_to, x);
            if x != a && x != b {
                opposite.insert(x);
            }
        }
    }
    shared == opposite
}

/// Whether any surviving triangle would turn over.
#[allow(clippy::too_many_arguments)]
fn flips(
    mesh: &FieldMesh,
    live: &[[u32; 3]],
    moved_to: &[u32],
    ring: &[usize],
    dying: &[usize],
    a: u32,
    b: u32,
    target: Point3<f64>,
) -> bool {
    for &i in ring {
        if dying.contains(&i) {
            continue;
        }
        let t = live[i];
        let at = |v: u32| {
            let v = resolve(moved_to, v);
            if v == a || v == b { target } else { mesh.positions[v as usize] }
        };
        let (p0, p1, p2) = (
            mesh.positions[resolve(moved_to, t[0]) as usize],
            mesh.positions[resolve(moved_to, t[1]) as usize],
            mesh.positions[resolve(moved_to, t[2]) as usize],
        );
        let was = (p1 - p0).cross(&(p2 - p0));
        let (q0, q1, q2) = (at(t[0]), at(t[1]), at(t[2]));
        let now = (q1 - q0).cross(&(q2 - q0));
        // Turned by more than a right angle, or collapsed to nothing. A small turn is the
        // ordinary consequence of moving a vertex and is what the error check has already
        // approved; a large one means the triangle has been pulled through its own plane.
        if now.norm() <= 0.0 || was.dot(&now) <= 0.0 {
            return true;
        }
    }
    false
}

/// Slides vertices along the surface until the triangles between them are better shaped.
///
/// # Why this and not more flipping
///
/// [`improve_shape`] swaps the diagonal of a quad whose four corners were fine and whose diagonal
/// was picked badly. Most dual-contouring slivers are not that: the quad itself is thin, because
/// a level change put its corners in a row, and no choice of diagonal helps. Measured, flipping
/// moved the mean smallest angle by a tenth of a degree. The corners have to move.
///
/// # Why it is allowed to move them
///
/// A vertex is not data — the mesher chose it, and any point on the surface would have served.
/// So a vertex may go anywhere on the surface, and the whole method is: propose the average of
/// its neighbours, keep only the part of that step which runs *along* the surface, and put it
/// back on the surface with a Newton step down the gradient. Repeat.
///
/// # What it must not do, and what stops it
///
/// **Round off the creases.** This is the failure that makes naive Laplacian smoothing useless
/// for CAD: sliding a vertex along a sharp edge is free by every measure this function has —
/// it stays on the surface, its triangles stay within tolerance — and the edge comes out ragged
/// or gone. A crease is where the incident face normals disagree, so a vertex whose normals
/// spread beyond [`CREASE_SPREAD`] is not moved at all. That is a deliberate over-refusal: it
/// also pins vertices merely *near* a crease, which is why the fringe of triangles along one
/// stays as the mesher left it.
///
/// **Turn a triangle over, or take one off the surface.** Every triangle touching a moved vertex
/// is checked afterwards and the move is undone if any fails. Unlike a collapse, the check is
/// the triangle's centre rather than a grid across it: a collapse makes triangles bigger and its
/// error grows in the middle of the new ones, while a relaxation leaves them the size they were
/// and moves them a fraction of that. The grid would cost a hundredfold for an answer that
/// cannot differ much.
///
/// Returns how many vertices ended up somewhere new.
pub fn relax(mesh: &mut FieldMesh, field: &(dyn ScalarField + Sync), tolerance_mm: f64) -> usize {
    relax_cancellable(mesh, field, tolerance_mm, &crate::Cancel::default())
        .expect("a token nobody holds is never set")
}

/// [`relax`], or `None` if `cancel` was set first — checked by every proposal, which is where
/// a pass spends its time, and between the acceptances.
pub(crate) fn relax_cancellable(
    mesh: &mut FieldMesh,
    field: &(dyn ScalarField + Sync),
    tolerance_mm: f64,
    cancel: &crate::Cancel,
) -> Option<usize> {
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); mesh.positions.len()];
    let mut touching: Vec<Vec<u32>> = vec![Vec::new(); mesh.positions.len()];
    for (t, tri) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let v = tri[k] as usize;
            touching[v].push(t as u32);
            for o in [tri[(k + 1) % 3], tri[(k + 2) % 3]] {
                if !neighbours[v].contains(&o) {
                    neighbours[v].push(o);
                }
            }
        }
    }

    // Pinned once, from the mesh as it arrived. Recomputing it per pass would let a vertex that
    // relaxation had *made* smooth start moving on the second pass, which is how a crease erodes
    // one pass at a time rather than all at once.
    let pinned: Vec<bool> = (0..mesh.positions.len())
        .map(|v| on_a_crease(mesh, &touching[v]))
        .collect();

    let mut moved = 0;
    for _ in 0..RELAX_PASSES {
        // **Proposed for every vertex at once, accepted one at a time.** Working out where a
        // vertex would like to go is twenty-odd field queries and reads nothing but positions, so
        // it parallelizes exactly; accepting has to be serial, because two neighbours moving
        // against the same stale positions can both be individually fine and jointly fold the
        // triangle between them.
        let want: Vec<Option<Point3<f64>>> = (0..mesh.positions.len())
            .into_par_iter()
            .map(|v| {
                if pinned[v] || neighbours[v].is_empty() || cancel.is_cancelled() {
                    return None;
                }
                proposed(mesh, field, v, &neighbours[v])
            })
            .collect();

        if cancel.is_cancelled() {
            return None;
        }
        let mut this_pass = 0;
        for v in 0..mesh.positions.len() {
            if v % 4096 == 0 && cancel.is_cancelled() {
                return None;
            }
            let Some(to) = want[v] else { continue };
            let was = mesh.positions[v];
            // Recorded before the move, because the question is whether this move turned them
            // over -- not whether they agreed with the field to begin with. A triangle the mesher
            // wound inward is the flip pass's business, and relaxation silently righting some of
            // them would make each pass's effect depend on what the last one happened to fix.
            let facing: Vec<_> = touching[v].iter().map(|&t| normal_of(mesh, t)).collect();
            mesh.positions[v] = to;
            if touching[v].iter().zip(&facing).all(|(&t, &was_facing)| {
                still_good(mesh, field, t, was_facing, tolerance_mm)
            }) {
                this_pass += 1;
            } else {
                mesh.positions[v] = was;
            }
        }
        if this_pass == 0 {
            break;
        }
        moved += this_pass;
    }
    Some(moved)
}

fn normal_of(mesh: &FieldMesh, t: u32) -> Vector3<f64> {
    let tri = mesh.triangles[t as usize];
    let p = |i: usize| mesh.positions[tri[i] as usize];
    (p(1) - p(0)).cross(&(p(2) - p(0)))
}

/// How far incident face normals may spread before a vertex counts as sitting on a crease and is
/// left alone. Forty degrees: wide enough not to pin the ordinary curvature of a sphere sampled
/// at the export's own tolerance, narrow enough to catch a boolean's seam.
const CREASE_SPREAD: f64 = 40.0 * std::f64::consts::PI / 180.0;

/// Passes. Converges quickly — the step is damped, so the later passes move almost nothing, and
/// the loop stops on its own when a pass moves nothing at all.
const RELAX_PASSES: usize = 5;

/// How much of the way toward the neighbours' average a vertex goes in one pass. Damped, because
/// the full step overshoots on a curved surface and the Newton correction then has to drag it
/// back, which costs more field queries than taking a smaller step twice.
const RELAX_STEP: f64 = 0.5;

fn on_a_crease(mesh: &FieldMesh, touching: &[u32]) -> bool {
    let normals: Vec<_> = touching
        .iter()
        .filter_map(|&t| {
            let tri = mesh.triangles[t as usize];
            let p = |i: usize| mesh.positions[tri[i] as usize];
            (p(1) - p(0)).cross(&(p(2) - p(0))).try_normalize(1e-12)
        })
        .collect();
    // A vertex the mesher left with no usable normal at all is pinned too: nothing here can tell
    // whether moving it would be safe.
    if normals.len() != touching.len() {
        return true;
    }
    normals
        .iter()
        .any(|a| normals.iter().any(|b| a.dot(b).clamp(-1.0, 1.0).acos() > CREASE_SPREAD))
}

/// Where a vertex would like to go: toward its neighbours, along the surface, and back onto it.
fn proposed(
    mesh: &FieldMesh,
    field: &(dyn ScalarField + Sync),
    v: usize,
    neighbours: &[u32],
) -> Option<Point3<f64>> {
    let here = mesh.positions[v];
    let mut sum = Vector3::zeros();
    for &n in neighbours {
        sum += mesh.positions[n as usize].coords;
    }
    let toward = sum / neighbours.len() as f64 - here.coords;

    // The step's length sets the scale everything else is measured at — a fixed one reads as
    // noise on a fine mesh and cuts corners on a coarse one.
    let h = (toward.norm() * 0.5).clamp(1e-5, 1.0);
    let up = gradient(field, here, h).try_normalize(1e-12)?;

    // Only the part running along the surface. The part along the normal is the mesher's answer
    // to where the surface *is*, and this function has no better one.
    let along = toward - up * toward.dot(&up);
    let mut p = here + along * RELAX_STEP;

    // Back onto the surface. Two steps: the first does nearly all of it, the second catches the
    // curvature the first ignored.
    for _ in 0..2 {
        let d = field.distance(p);
        let g = gradient(field, p, h);
        let n2 = g.norm_squared();
        if n2 < 1e-18 {
            return None;
        }
        p -= g * (d / n2);
    }
    Some(p)
}

fn gradient(field: &(dyn ScalarField + Sync), p: Point3<f64>, h: f64) -> Vector3<f64> {
    let d = |o: Vector3<f64>| field.distance(p + o);
    Vector3::new(
        d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
        d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
        d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
    ) / (2.0 * h)
}

/// Whether a triangle is still on the surface and still facing the way it was, after one of its
/// corners moved.
///
/// One field query. Asking [`faces_outward`] instead would be six, for a stricter question than
/// this pass is entitled to ask: a triangle the mesher wound inward was inward before the vertex
/// moved, and refusing the move over it would make relaxation's reach depend on defects it is not
/// the one fixing.
fn still_good(
    mesh: &FieldMesh,
    field: &(dyn ScalarField + Sync),
    t: u32,
    was_facing: Vector3<f64>,
    tolerance_mm: f64,
) -> bool {
    let tri = mesh.triangles[t as usize];
    let p = |i: usize| mesh.positions[tri[i] as usize];
    // A triangle that has collapsed to a line has no facing to check and no business surviving.
    let normal = (p(1) - p(0)).cross(&(p(2) - p(0)));
    if normal.norm() < 1e-18 || normal.dot(&was_facing) <= 0.0 {
        return false;
    }
    let centre = Point3::from((p(0).coords + p(1).coords + p(2).coords) / 3.0);
    field.distance(centre).abs() <= tolerance_mm
}

/// Merges pairs of corners the mesher left in the same place, and drops the triangles that
/// become nothing.
///
/// # Why this is safe when dropping a zero-area triangle generally is not
///
/// It was tried once as "drop every triangle with no area" and it tore roughly two open edges per
/// triangle dropped. The reasoning offered for it — that a collapsed triangle's remaining edges
/// are the same edge traversed both ways, so they cancel and the surface stays closed — is true,
/// but only when two of its corners are in the *same place*. A **collinear** triangle has three
/// distinct corners in a line: its three edges are three different edges, and removing it leaves
/// a hole. On the measured document the two kinds ran 44 coincident against 104 collinear, so the
/// blanket rule was wrong about seventy per cent of what it was deleting.
///
/// This is the subset the argument actually covers. It only ever merges two corners *of one
/// triangle* that are already coincident, which is why it cannot join sheets that merely pass
/// near each other.
///
/// # What it is for
///
/// Backlog 27's fringe, the half neither collapsing nor relaxing can reach: a collapse cannot
/// shorten an edge that has no length, and relaxation cannot separate two vertices it is pulling
/// together. A quad straddling a wall that has thinned to nothing emits a triangle whose two
/// corners landed on top of each other; welding them is what makes it one triangle instead of a
/// degenerate pair.
///
/// Returns how many triangles were removed.
pub fn weld(mesh: &mut FieldMesh, tolerance_mm: f64) -> usize {
    // Union-find over vertices, but only ever joining two corners of one triangle.
    let mut parent: Vec<u32> = (0..mesh.positions.len() as u32).collect();
    fn root(parent: &mut Vec<u32>, mut v: u32) -> u32 {
        while parent[v as usize] != v {
            parent[v as usize] = parent[parent[v as usize] as usize];
            v = parent[v as usize];
        }
        v
    }

    for tri in &mesh.triangles {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            if a == b {
                continue;
            }
            let apart = (mesh.positions[a as usize] - mesh.positions[b as usize]).norm();
            if apart > tolerance_mm {
                continue;
            }
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                // Toward the lower index, so the result does not depend on triangle order.
                let (keep, gone) = (ra.min(rb), ra.max(rb));
                parent[gone as usize] = keep;
            }
        }
    }

    let before = mesh.triangles.len();
    let mut welded = Vec::with_capacity(before);
    for tri in &mesh.triangles {
        let t = [
            root(&mut parent, tri[0]),
            root(&mut parent, tri[1]),
            root(&mut parent, tri[2]),
        ];
        // A repeated corner is the degenerate pair collapsing into nothing. Its two remaining
        // edges are one edge traversed both ways, so the surface closes over its absence.
        if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
            continue;
        }
        welded.push(t);
    }
    mesh.triangles = welded;

    // `positions` and `normals` are left as they are. A vertex nothing indexes costs eight bytes
    // in memory and nothing in the file — the STL writer reads through the triangles — and
    // compacting them would mean renumbering, which is a second chance to get an index wrong for
    // no gain.
    before - mesh.triangles.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_field;

    struct Sphere {
        radius: f64,
    }
    impl ScalarField for Sphere {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.coords.norm() - self.radius
        }
        fn normal(&self, p: Point3<f64>) -> Option<nalgebra::Vector3<f64>> {
            p.coords.try_normalize(1e-12)
        }
    }

    fn edge_uses(mesh: &FieldMesh) -> HashMap<(u32, u32), usize> {
        let mut uses = HashMap::new();
        for t in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        uses
    }

    /// Simplifying removes triangles and keeps the surface where the tolerance says.
    ///
    /// **Checked over the triangles, not at their corners**, because that is the only check that
    /// means anything: three corners can sit exactly on a sphere while the triangle between them
    /// bulges away from it, which is how an earlier version of this claimed a 95% saving that
    /// was really a different shape.
    #[test]
    fn simplifying_a_sphere_removes_triangles_and_keeps_the_surface() {
        let field = Sphere { radius: 10.0 };
        let r = 14.0;
        let tolerance = 0.2;
        let mut mesh = mesh_field(
            &field,
            (Point3::new(-r, -r, -r), Point3::new(r, r, r)),
            48,
        );
        let before = mesh.triangles.len();
        let report = simplify(&mut mesh, &field, tolerance);

        assert_eq!(report.triangles_before, before);
        assert!(report.saved() > 0.1, "removed only {:.1}%", report.saved() * 100.0);

        // Sampled across each surviving triangle, at the points a flat triangle strays furthest.
        let mut worst = 0.0f64;
        for t in &mesh.triangles {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            for (u, v) in [
                (1.0 / 3.0, 1.0 / 3.0),
                (0.5, 0.5),
                (0.5, 0.0),
                (0.0, 0.5),
                (0.25, 0.25),
            ] {
                let p = a + (b - a) * u + (c - a) * v;
                worst = worst.max(field.distance(p).abs());
            }
        }
        assert!(
            worst <= tolerance + 1e-9,
            "a point on the mesh sits {worst:.4} mm off a {tolerance} mm tolerance"
        );
    }

    /// And it does not tear the mesh: no edge is left belonging to one triangle.
    #[test]
    fn simplifying_never_tears_the_surface() {
        let field = Sphere { radius: 10.0 };
        let r = 14.0;
        let mut mesh = mesh_field(
            &field,
            (Point3::new(-r, -r, -r), Point3::new(r, r, r)),
            40,
        );
        let torn_before = edge_uses(&mesh).values().filter(|n| **n == 1).count();
        assert_eq!(torn_before, 0, "the mesher should not have torn it either");

        simplify(&mut mesh, &field, 0.05);
        let torn = edge_uses(&mesh).values().filter(|n| **n == 1).count();
        assert_eq!(torn, 0, "{torn} edges belong to one triangle after simplifying");
    }

    /// A tolerance of nothing removes what costs nothing, and nothing else.
    ///
    /// **The test that found the error criterion was wrong.** Judging a collapse only by where
    /// the new *vertex* lands permitted removing 98.3% of a sphere at zero tolerance, because
    /// every vertex of a tetrahedron inscribed in a sphere sits exactly on that sphere. What
    /// moves is the surface between the vertices, so the check samples the triangles that would
    /// survive as well — after which the same case removes 3.8%.
    ///
    /// And that 3.8% is the point rather than a residue: at zero tolerance the only collapses
    /// left are the ones that move the surface nowhere at all, which is precisely the fringe of
    /// degenerate slivers a dual contour leaves along a crease. Removing those is what this
    /// pass is for.
    #[test]
    fn a_tolerance_of_nothing_removes_slivers_and_leaves_the_shape() {
        let field = Sphere { radius: 10.0 };
        let r = 14.0;
        let mut mesh = mesh_field(
            &field,
            (Point3::new(-r, -r, -r), Point3::new(r, r, r)),
            32,
        );
        let slivers = |m: &FieldMesh| {
            m.triangles
                .iter()
                .filter(|t| {
                    let (a, b, c) = (
                        m.positions[t[0] as usize],
                        m.positions[t[1] as usize],
                        m.positions[t[2] as usize],
                    );
                    (b - a).cross(&(c - a)).norm() * 0.5 < 1e-6
                })
                .count()
        };
        let off_sphere = |m: &FieldMesh| {
            m.positions
                .iter()
                .map(|p| (p.coords.norm() - 10.0).abs())
                .fold(0.0f64, f64::max)
        };
        let before = slivers(&mesh);
        // Dual contouring does not put its vertices exactly on the surface — the solved point
        // is near it, not on it — so the baseline is measured rather than assumed to be zero.
        // That is what this test is about: simplifying must not make it worse.
        let was_off = off_sphere(&mesh);
        let report = simplify(&mut mesh, &field, 0.0);

        assert!(
            report.saved() < 0.10,
            "a zero tolerance removed {:.1}%, which is more than the slivers",
            report.saved() * 100.0
        );
        assert!(
            slivers(&mesh) < before.max(1),
            "it should have removed slivers: {before} before, {} after",
            slivers(&mesh)
        );
        // And the shape is no further from the sphere than the mesher left it, which is what
        // a zero tolerance promises: nothing that moves the surface.
        let now_off = off_sphere(&mesh);
        assert!(
            now_off <= was_off + 1e-9,
            "worst deviation went from {was_off:.6} mm to {now_off:.6} mm at zero tolerance"
        );
    }
    #[test]
    fn flipping_a_diagonal_keeps_every_triangle_facing_out() {
        // A sphere meshed coarsely has plenty of badly shaped pairs, so the pass has work to do.
        // **Adaptive, because that is where slivers come from.** A uniform grid meets a sphere
        // at much the same angle everywhere and produces triangles this pass rightly declines to
        // touch; it is the level changes that leave long thin ones.
        let field = Creased;
        let mut mesh = sliverish(&field);
        let outward_before = facing_outward(&mesh, &field);

        let made = improve_shape(&mut mesh, &field, FLIP_TOLERANCE);
        assert!(made > 0, "a level change leaves diagonals worth swapping");

        // The count that regressed six-fold when the flip inherited its winding from the old
        // pair's normals instead of asking the field.
        assert!(
            facing_outward(&mesh, &field) >= outward_before,
            "a flip may not leave fewer triangles facing out than it found"
        );
    }

    #[test]
    fn flipping_never_gives_an_edge_more_than_two_triangles() {
        let field = Creased;
        let mut mesh = sliverish(&field);
        let before = shared_by_more_than_two(&mesh);
        improve_shape(&mut mesh, &field, FLIP_TOLERANCE);
        // Two pairs in one pass creating the same new edge is what this asserts against; it took
        // the user's document from 58 such edges to 272 before the pass recorded what it made.
        assert_eq!(
            shared_by_more_than_two(&mesh),
            before,
            "a flip may not make an edge non-manifold"
        );
    }

    #[test]
    fn flipping_leaves_the_mesh_on_the_surface() {
        let field = Creased;
        let mut mesh = sliverish(&field);
        improve_shape(&mut mesh, &field, FLIP_TOLERANCE);
        // A flip moves no vertex, so the corners are exactly where the mesher put them — but the
        // triangle between them is new, and its middle is what the tolerance is about.
        for t in &mesh.triangles {
            let c = (mesh.positions[t[0] as usize].coords
                + mesh.positions[t[1] as usize].coords
                + mesh.positions[t[2] as usize].coords)
                / 3.0;
            assert!(field.distance(Point3::from(c)).abs() < 0.6);
        }
    }

    /// Proportionate to [`sliverish`]'s triangles, which run to 6 mm. At 0.1 mm — the figure the
    /// measured export uses on triangles a tenth the size — every flip here is correctly refused
    /// for cutting the crease corner, and the pass does nothing.
    const FLIP_TOLERANCE: f64 = 1.0;

    /// Two overlapping spheres. **The crease is the point**: a smooth sphere's slivers come from
    /// genuinely thin quads, where swapping the diagonal cannot help and the pass correctly makes
    /// no flips at all. It is where the surface turns a corner that the mesher lays down quads
    /// whose diagonal was picked badly.
    struct Creased;

    impl ScalarField for Creased {
        fn distance(&self, p: Point3<f64>) -> f64 {
            let a = (p.coords - Vector3::new(-4.0, 0.0, 0.0)).norm() - 8.0;
            let b = (p.coords - Vector3::new(4.0, 0.0, 0.0)).norm() - 8.0;
            a.min(b)
        }
    }

    /// A mesh with the level changes that produce slivers, which is what a flip is for.
    fn sliverish(field: &(dyn ScalarField + Sync)) -> FieldMesh {

        crate::mesh_field_adaptive(
            field,
            (Point3::new(-14.0, -14.0, -14.0), Point3::new(14.0, 14.0, 14.0)),
            &crate::MeshTolerance {
                chord_mm: 0.15,
                angle_deg: 20.0,
                max_edge_mm: 6.0,
                min_cell_mm: 0.5,
            },
        )
    }

    /// How many triangles wind the way the field says out is.
    fn facing_outward(mesh: &FieldMesh, field: &(dyn ScalarField + Sync)) -> usize {
        mesh.triangles
            .iter()
            .filter(|t| {
                let p = |i: usize| mesh.positions[t[i] as usize];
                faces_outward(field, p(0), p(1), p(2))
            })
            .count()
    }

    fn shared_by_more_than_two(mesh: &FieldMesh) -> usize {
        let mut count: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        count.values().filter(|&&n| n > 2).count()
    }

    #[test]
    fn relaxing_makes_the_triangles_better_shaped() {
        let field = Creased;
        let mut mesh = sliverish(&field);
        let before = mean_smallest_angle(&mesh);

        let moved = relax(&mut mesh, &field, FLIP_TOLERANCE);
        assert!(moved > 0, "a dual contour has vertices worth sliding");

        // The measurement this pass exists for. On the user's document it went 38.7 to 47.9
        // degrees, where flipping alone had moved it by a tenth of one.
        let after = mean_smallest_angle(&mesh);
        assert!(
            after > before + 2.0,
            "mean smallest angle went {before:.1} to {after:.1} deg"
        );
    }

    #[test]
    fn relaxing_leaves_every_vertex_on_the_surface() {
        let field = Creased;
        let mut mesh = sliverish(&field);
        relax(&mut mesh, &field, FLIP_TOLERANCE);
        // Unlike a flip, this *moves* vertices, so the corners are no longer where the mesher put
        // them and being back on the surface is a claim rather than a given. The Newton step is
        // what makes it true; without it a tangential slide across a curved surface lands off it
        // by the sagitta of the step.
        for &p in &mesh.positions {
            assert!(
                field.distance(p).abs() < FLIP_TOLERANCE,
                "a vertex ended up {:.3} mm off the surface",
                field.distance(p)
            );
        }
    }

    #[test]
    fn relaxing_does_not_slide_a_vertex_along_a_crease() {
        let field = Creased;
        let mut mesh = sliverish(&field);
        let before = mesh.positions.clone();

        // Which vertices sit on the crease, decided the same way the pass decides it.
        let mut touching: Vec<Vec<u32>> = vec![Vec::new(); mesh.positions.len()];
        for (t, tri) in mesh.triangles.iter().enumerate() {
            for k in 0..3 {
                touching[tri[k] as usize].push(t as u32);
            }
        }
        let creased: Vec<usize> = (0..mesh.positions.len())
            .filter(|&v| on_a_crease(&mesh, &touching[v]))
            .collect();
        assert!(
            !creased.is_empty(),
            "two overlapping spheres meet at a crease; the test needs one to guard"
        );

        relax(&mut mesh, &field, FLIP_TOLERANCE);

        // **The failure that makes plain Laplacian smoothing useless for CAD.** Sliding a vertex
        // along a sharp edge is free by every other measure here — it stays on the surface, and
        // its triangles stay within tolerance — and the edge comes out ragged. Nothing but this
        // pin prevents it, so the pin is what gets asserted.
        for v in creased {
            assert_eq!(
                mesh.positions[v], before[v],
                "a vertex on the crease moved"
            );
        }
    }

    fn mean_smallest_angle(mesh: &FieldMesh) -> f64 {
        let total: f64 = mesh
            .triangles
            .iter()
            .map(|t| {
                let p = |i: usize| mesh.positions[t[i] as usize];
                smallest_angle(p(0), p(1), p(2)).to_degrees()
            })
            .sum();
        total / mesh.triangles.len() as f64
    }

    #[test]
    fn welding_removes_a_pair_of_coincident_corners_and_leaves_a_collinear_one() {
        // **The distinction the blanket "drop zero-area triangles" rule got wrong.** Both of
        // these have no area. Only the first one's remaining edges cancel.
        let mut mesh = FieldMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0), // on top of the one before it
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0), // three distinct corners, in a line
            ],
            normals: vec![Vector3::z(); 6],
            triangles: vec![[0, 1, 2], [3, 4, 5]],
        };

        let dropped = weld(&mut mesh, 1e-9);

        assert_eq!(dropped, 1);
        assert_eq!(
            mesh.triangles,
            vec![[3, 4, 5]],
            "the collinear triangle has three different edges; removing it would leave a hole"
        );
    }

    #[test]
    fn welding_leaves_the_surface_as_closed_as_it_found_it() {
        let field = Creased;
        let mut mesh = sliverish(&field);
        let before = open_edges(&mesh);

        weld(&mut mesh, 1e-9);

        // The measurement that reverted the first attempt at this: dropping degenerate triangles
        // wholesale cost about two open edges each.
        assert_eq!(open_edges(&mesh), before, "welding opened the surface");
    }

    /// Edges used by exactly one triangle — a hole's boundary.
    fn open_edges(mesh: &FieldMesh) -> usize {
        let mut count: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        count.values().filter(|&&n| n == 1).count()
    }

}

