// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Bodies that meet face to face by construction, built as one body so the face they share never
//! exists (backlog item 113).
//!
//! **The defect this removes.** A joined mirror of a body whose face lies on the mirror plane was
//! the union of the body and its reflection, and a union of two bodies that touch keeps the face
//! they share *inside* the solid. Measured on a 20 x 20 rectangle extruded 15 mm and mirrored about
//! its own edge: the field at the plane, 7.5 mm deep in solid material, read **0.000** where the true
//! depth is 7.5. The sign is right everywhere, so the body alone renders clean. The owner met it
//! as a dotted dark line along the mirror plane of a wing mirrored about its own profile and
//! filleted to a sphere (`centerline-blend1.fform`): drawn by one build, the mirror shows the line
//! and the same solid built as one extrude with a second side does not, and with this rewrite the
//! owner's file draws pixel for pixel as that one extrude.
//!
//! **Why it is fixed here and not in the union.** No operation on two finished fields can remove a
//! face buried between them: two values at a point do not determine how deep that point is in their
//! union. So the fix is one level up — where the bodies are still sections, merge the sections and
//! build one body, and there is no shared face to bury.
//!
//! **A rewrite both compilers read**, as [`crate::with_turns_flattened`] is: the CPU field, the
//! generated shader and the viewport's own reading of a section all call
//! [`with_shared_faces_merged`] first, so they cannot disagree about which body is drawn.

use fieldform_scene::{
    Curve, CurveId, Document, Graph, Node, NodeId, Outline, ParamRef, Sides, Sweep,
};
use nalgebra::{Point3, Vector3};

/// How close to the mirror plane a section's vertex has to be to count as lying on it, in mm.
///
/// A distance rather than zero because a section is sampled from curves and placed through
/// transforms, and a vertex typed at nought arrives a few ulps away from it. A micron is far below
/// anything a person models and far above that noise.
const ON_THE_PLANE_MM: f64 = 1e-3;

/// A closed section that touches a plane along one edge, joined to its reflection across that plane
/// with the shared edge left out: one loop, symmetric about the plane, whose inside is the union of
/// the two.
///
/// `None` unless the section lies wholly on one side of the plane and meets it along exactly one
/// contiguous run of vertices — an edge, not a single point. A section that crosses the plane, keeps
/// clear of it, lies in it, or touches it in two separate places is left as it is: the first is not
/// a reflection anyone joined, the second has no shared face, and the last two would make a loop
/// with a hole or no area.
///
/// **The winding is kept.** A reflection reverses a loop's turning, and walking the reflected arc
/// backwards restores it, so the merged loop turns the way the original did and an extrude of it
/// faces the same way.
pub fn merge_loop_across_plane(
    points: &[Point3<f64>],
    at: Point3<f64>,
    normal: Vector3<f64>,
) -> Option<Vec<Point3<f64>>> {
    let normal = normal.try_normalize(1e-12)?;
    let mut loop_points: Vec<Point3<f64>> = points.to_vec();
    // A closed loop listed with its first point repeated at the end is the same loop.
    if loop_points.len() > 1
        && (loop_points[0] - loop_points[loop_points.len() - 1]).norm() <= ON_THE_PLANE_MM
    {
        loop_points.pop();
    }
    let n = loop_points.len();
    if n < 3 {
        return None;
    }
    let side: Vec<f64> = loop_points.iter().map(|p| (p - at).dot(&normal)).collect();
    // Wholly on the positive side or wholly on the negative: either is one side of the plane, and
    // the reflection is the same either way.
    let one_side = side.iter().all(|s| *s >= -ON_THE_PLANE_MM)
        || side.iter().all(|s| *s <= ON_THE_PLANE_MM);
    if !one_side {
        return None;
    }
    let on: Vec<bool> = side.iter().map(|s| s.abs() <= ON_THE_PLANE_MM).collect();
    let touching = on.iter().filter(|o| **o).count();
    if touching < 2 || touching == n {
        return None;
    }
    // The start of each run of on-plane vertices, read cyclically; exactly one run is allowed.
    let starts: Vec<usize> = (0..n).filter(|&i| on[i] && !on[(i + n - 1) % n]).collect();
    let [first_on] = starts.as_slice() else { return None };
    let mut last_on = *first_on;
    while on[(last_on + 1) % n] {
        last_on = (last_on + 1) % n;
    }
    // On the plane exactly, so the merged loop is exactly symmetric and its two halves meet
    // without a sliver.
    let onto_plane = |p: Point3<f64>| p - normal * (p - at).dot(&normal);
    let reflect = |p: Point3<f64>| p - normal * (2.0 * (p - at).dot(&normal));
    // The arc off the plane, from the run's last vertex round to its first: both ends on the plane,
    // everything between them clear of it.
    let mut arc: Vec<Point3<f64>> = Vec::new();
    let mut i = last_on;
    loop {
        arc.push(loop_points[i]);
        if i == *first_on {
            break;
        }
        i = (i + 1) % n;
    }
    let (head, tail) = (onto_plane(arc[0]), onto_plane(arc[arc.len() - 1]));
    let last = arc.len() - 1;
    arc[0] = head;
    arc[last] = tail;
    // The arc, then its reflection walked back, which ends where the arc began.
    let mut merged = arc.clone();
    merged.extend(arc[1..last].iter().rev().map(|p| reflect(*p)));
    Some(merged)
}

/// The id the merged section of `mirror` is given in the rewritten document: a function of the
/// mirror alone, so the three readers of the rewrite produce the same curve and the shader's
/// binding names the curve the viewport bakes.
pub fn merged_section_curve(mirror: NodeId) -> CurveId {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ("fieldform merged mirror section", mirror.raw()).hash(&mut h);
    CurveId::from_raw(h.finish())
}

/// One joined mirror that is built as one extrude, and what it is built from.
struct Merge {
    mirror: NodeId,
    extrude: NodeId,
    node: Node,
    /// The merged section, when the merge made one; a mirror across the profile's own plane keeps
    /// the profile it had and only gains a second side.
    curve: Option<Curve>,
}

/// The joined mirrors of `graph` that can be built as one extrude.
///
/// **Two shapes of mirror, both of an extrude swept straight out of its plane:**
///
/// **Across the profile's own plane, the solid starting on it** — the start face is the shared
/// face, the whole profile of it. The two halves are exactly an extrude with a second side
/// ([`Sides::Two`]) of the same depth and the same draft, which is one closed form with the
/// profile's plane inside the material; that is why `Sides::Two` exists rather than two extrudes
/// and a union. Any draft, typed or driven: leaning in or out, both halves start from the same
/// face, and the second side's draft is measured from the start plane as side one's is, so it is
/// the reflection. The start must be a typed nought and there must be one side to begin with: a
/// solid set off the plane does not touch it, and one already on both sides crosses it.
///
/// **Through the profile's plane, along the extrude** — each condition for its own reason:
///
/// - the profile is **closed**, since a section edge on the plane becomes the side wall shared;
/// - the plane **contains the extrude's direction**, so the shared face is a side wall and not a
///   cap or a slant;
/// - the **draft is a typed number, nought or more**. Draft that leans out overlaps the two halves,
///   and growing a merged section is exactly growing each half and joining them, so the merge is
///   the same body. Draft that leans in parts them above the base: there is no shared face, the
///   groove between them is real geometry, and the union already draws it truly. A draft driven
///   by a parameter is left as a union, because a value edit does not regenerate the shader and
///   could otherwise carry the draft across nought without the decision following it;
/// - the section meets the plane along one edge ([`merge_loop_across_plane`]).
fn merges(document: &Document, graph: &Graph) -> Vec<Merge> {
    let numbers = |r: &ParamRef| document.resolve(r);
    let mut out = Vec::new();
    for (mirror, node) in graph.nodes() {
        let Node::Mirror { input, frame, join: true } = node else { continue };
        // A slot left empty is a mirror of nothing, which the compile refuses by name.
        let Some(input) = input.source() else { continue };
        let Some(Node::Extrude { outline, sweep, start, height, sides, draft, corners, rims }) =
            graph.node(input)
        else {
            continue;
        };
        let Sweep::Normal = sweep else { continue };
        let Ok((origin, rotation)) = crate::resolve_frame(document, graph, frame, &numbers) else {
            continue;
        };
        let plane_normal = rotation * Vector3::z();
        let Ok(section) = crate::extrude_section(
            document,
            outline,
            sweep,
            crate::profile::PROFILE_TURN_DEGREES,
        ) else {
            continue;
        };
        let facing = section.normal.dot(&plane_normal);
        if (facing.abs() - 1.0).abs() <= 1e-9 {
            let on_the_plane =
                (section.origin - Point3::from(origin)).dot(&plane_normal).abs() <= ON_THE_PLANE_MM;
            let starts_there = matches!(start, ParamRef::Literal(at) if *at == 0.0);
            if on_the_plane && starts_there && matches!(sides, Sides::One) {
                out.push(Merge {
                    mirror,
                    extrude: input,
                    node: Node::Extrude {
                        outline: outline.clone(),
                        sweep: Sweep::Normal,
                        start: start.clone(),
                        height: height.clone(),
                        sides: Sides::Two { height: height.clone(), draft: draft.clone() },
                        draft: draft.clone(),
                        corners: corners.clone(),
                        // **Carried, so the merge is the mirrored solid's rounding** at both far
                        // ends. The face on the plane loses its rim, which is what joining the two
                        // halves means: a mirror asked to join them would otherwise leave a groove
                        // round the seam where each half's rim fell away. So a rim on the far end
                        // alone is a rim on both of the merged solid's ends, and one on the start
                        // alone -- the face on the plane -- is no rim at all.
                        rims: match rims.at {
                            fieldform_scene::RimEnds::Start => fieldform_scene::Rims::hard(),
                            _ => fieldform_scene::Rims { at: fieldform_scene::RimEnds::Both, ..*rims },
                        },
                    },
                    curve: None,
                });
            }
            continue;
        }
        let Outline::Closed { .. } = outline else { continue };
        let ParamRef::Literal(lean) = draft else { continue };
        if *lean < 0.0 {
            continue;
        }
        if facing.abs() > 1e-9 {
            continue;
        }
        let Some(points) =
            merge_loop_across_plane(&section.points, Point3::from(origin), plane_normal)
        else {
            continue;
        };
        let id = merged_section_curve(mirror);
        let Ok(curve) = Curve::polyline(id, "merged section", points, true) else { continue };
        out.push(Merge {
            mirror,
            extrude: input,
            node: Node::Extrude {
                outline: Outline::closed(id),
                sweep: Sweep::Normal,
                start: start.clone(),
                height: height.clone(),
                sides: sides.clone(),
                draft: draft.clone(),
                corners: corners.clone(),
                rims: *rims,
            },
            curve: Some(curve),
        });
    }
    out
}

/// `document` and `graph` with every joined mirror that can be one body built as one body, or `None`
/// when there is none — which is every document without such a mirror, and costs nothing for them.
///
/// The mirror keeps its id, so its name, its visibility and everything keyed by it still find it;
/// it simply is an extrude of the merged section now. The extrude it reflected is removed when
/// nothing else read it, because a body left standing beside the merged one would be unioned with
/// it — and a union of the half with the whole brings the buried face straight back.
pub fn with_shared_faces_merged(document: &Document, graph: &Graph) -> Option<(Document, Graph)> {
    // The switch `FIELDFORM_NO_MIRROR_FOLD` is, and for its reason: the merged body and the union
    // it replaces are supposed to be the same surface, so the way to show what the merge changes
    // on screen is to draw both with one build and compare.
    if std::env::var_os("FIELDFORM_NO_SHARED_FACE_MERGE").is_some() {
        return None;
    }
    let found = merges(document, graph);
    if found.is_empty() {
        return None;
    }
    let mut doc = document.clone();
    let mut out = graph.clone();
    for merge in &found {
        if let Some(curve) = &merge.curve {
            doc.curves.insert(curve.id, curve.clone());
        }
        if let Some(node) = out.node_mut(merge.mirror) {
            *node = merge.node.clone();
        }
    }
    for merge in &found {
        let still_read = out.nodes().any(|(_, n)| n.inputs().contains(&merge.extrude));
        if !still_read {
            out.remove(merge.extrude);
        }
    }
    Some((doc, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point3<f64> {
        Point3::new(x, y, 0.0)
    }

    /// The plane y = 0, which the sections below touch along their bottom edge.
    fn y_plane() -> (Point3<f64>, Vector3<f64>) {
        (Point3::origin(), Vector3::y())
    }

    /// Twice the signed area of a loop in the XY plane: positive when it turns anticlockwise.
    fn turning(points: &[Point3<f64>]) -> f64 {
        (0..points.len())
            .map(|i| {
                let (a, b) = (points[i], points[(i + 1) % points.len()]);
                a.x * b.y - b.x * a.y
            })
            .sum()
    }

    /// **A section touching the plane along an edge becomes its whole, symmetric self.**
    #[test]
    fn a_section_touching_the_plane_merges_with_its_reflection() {
        let (at, n) = y_plane();
        let half = [p(-10.0, 0.0), p(10.0, 0.0), p(10.0, 20.0), p(-10.0, 20.0)];
        let merged = merge_loop_across_plane(&half, at, n).expect("it touches along an edge");
        let (lo, hi) = merged.iter().fold((f64::MAX, f64::MIN), |(lo, hi), q| (lo.min(q.y), hi.max(q.y)));
        assert_eq!((lo, hi), (-20.0, 20.0), "the reflection doubles it across the plane");
        assert!(merged.iter().all(|q| q.x.abs() <= 10.0 + 1e-12));
        // Symmetric: every vertex's reflection is a vertex.
        for q in &merged {
            assert!(merged.iter().any(|r| (r - Point3::new(q.x, -q.y, q.z)).norm() < 1e-12), "{q:?}");
        }
        // And it still turns the way the original did, so an extrude of it faces the same way.
        assert!(turning(&half) > 0.0 && turning(&merged) > 0.0);
        assert!((turning(&merged) - 2.0 * turning(&half)).abs() < 1e-9, "twice the area");
    }

    /// The run on the plane may wrap round the end of the list; the merge is the same.
    #[test]
    fn a_run_on_the_plane_that_wraps_round_the_list_merges_the_same() {
        let (at, n) = y_plane();
        let half = [p(10.0, 0.0), p(10.0, 20.0), p(-10.0, 20.0), p(-10.0, 0.0)];
        let merged = merge_loop_across_plane(&half, at, n).expect("touches along the wrapped edge");
        assert!((turning(&merged) - 2.0 * turning(&half)).abs() < 1e-9);
    }

    /// **Left alone unless there is a shared face**: clear of the plane, crossing it, or touching
    /// it at a single point.
    #[test]
    fn a_section_with_no_edge_on_the_plane_is_left_as_it_is() {
        let (at, n) = y_plane();
        let clear = [p(-10.0, 1.0), p(10.0, 1.0), p(10.0, 21.0), p(-10.0, 21.0)];
        assert!(merge_loop_across_plane(&clear, at, n).is_none(), "clear of the plane");
        let crossing = [p(-10.0, -1.0), p(10.0, -1.0), p(10.0, 19.0), p(-10.0, 19.0)];
        assert!(merge_loop_across_plane(&crossing, at, n).is_none(), "crossing it");
        let point = [p(0.0, 0.0), p(10.0, 10.0), p(-10.0, 10.0)];
        assert!(merge_loop_across_plane(&point, at, n).is_none(), "a point is not a face");
        let two_feet = [
            p(-10.0, 0.0), p(-5.0, 0.0), p(-5.0, 10.0), p(5.0, 10.0),
            p(5.0, 0.0), p(10.0, 0.0), p(10.0, 20.0), p(-10.0, 20.0),
        ];
        assert!(merge_loop_across_plane(&two_feet, at, n).is_none(), "two runs: a hole, not a merge");
    }

    /// The same mirror always names the same merged curve, and two mirrors never share one.
    #[test]
    fn a_merged_section_is_named_by_its_mirror_alone() {
        let (a, b) = (NodeId::new(), NodeId::new());
        assert_eq!(merged_section_curve(a), merged_section_curve(a));
        assert_ne!(merged_section_curve(a), merged_section_curve(b));
    }
}
