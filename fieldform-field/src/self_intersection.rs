// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a cage's limit surface passes through itself, and where on the cage it does.
//!
//! **Why this is a red fault and not a warning.** The cage field takes its sign from the
//! angle-weighted pseudonormal test over the tessellated limit surface, and that test is correct
//! for a closed surface that does not cross itself. Where the surface does, points are classified
//! inside or outside inconsistently -- the field is wrong, not missing -- and nothing else
//! reports it. So it is found here, on **the same tessellation the field is built from**: a
//! coarser proxy can miss a thin fold the field's own triangles would meet.
//!
//! The work is a bounding-volume hierarchy traversed against itself for the pairs of triangles
//! whose boxes overlap, then an exact triangle-triangle test on each pair, leaving out the pairs
//! that always touch: neighbours sharing a vertex or an edge, and pairs at the same extraordinary
//! vertex, where the evaluator snaps samples to the vertex's limit position and can make a
//! zero-area touch that is an artefact, not a fold.

use crate::bvh::Bvh;
use fieldform_cage::{Cage, FaceId, VertexId};
use fieldform_subd::{LimitSurface, Tessellation};
use nalgebra::{Point3, Vector3};
use std::collections::HashSet;

/// What the check found: the cage faces whose surface crosses another's, and how many triangle
/// pairs crossed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SelfIntersections {
    pub faces: Vec<FaceId>,
    pub crossings: usize,
}

impl SelfIntersections {
    pub fn is_clean(&self) -> bool {
        self.faces.is_empty()
    }
}

/// The check, on `cage` tessellated at `levels` -- the level the field would be built at.
///
/// `epsilon` is the absolute tolerance: two patches touching within it are not reported, so a
/// model that is deliberately near-touching does not turn red. The caller passes the cage's
/// `epsilon_length`.
pub fn find(cage: &Cage, levels: usize, epsilon: f64) -> SelfIntersections {
    let tessellation = LimitSurface::new(cage).tessellate(levels);
    find_in(cage, &tessellation, epsilon)
}

/// The check on a tessellation already taken, for a caller that has one.
pub fn find_in(cage: &Cage, t: &Tessellation, epsilon: f64) -> SelfIntersections {
    let bvh = Bvh::build(&t.positions, &t.triangles);
    // The extraordinary cage vertex each triangle touches, if it touches one at a corner of its
    // face: a triangle with a corner at (0,0), (1,0), (0,1) or (1,1) of its face's parameter
    // square sits on that face corner, and the cage vertex there is read off the face.
    let extraordinary: Vec<Option<VertexId>> = (0..t.triangles.len())
        .map(|i| {
            let face = t.triangle_faces[i];
            let corners = cage.face_vertices(face);
            t.triangle_uvs[i].iter().find_map(|&[u, v]| {
                let corner = match (u < 1e-9, u > 1.0 - 1e-9, v < 1e-9, v > 1.0 - 1e-9) {
                    (true, _, true, _) => Some(0),
                    (_, true, true, _) => Some(1),
                    (_, true, _, true) => Some(2),
                    (true, _, _, true) => Some(3),
                    _ => None,
                }?;
                let vertex = corners[corner];
                (cage.valence(vertex) != 4).then_some(vertex)
            })
        })
        .collect();

    // **The evaluator's snap near an extraordinary vertex collapses the last ring of grid
    // triangles onto the vertex's limit position** -- three corners at one point, or two -- and a
    // triangle with no area has no plane, so the separating-axis test cannot tell it from
    // anything it touches. Those are the artefact the check must not read as a fold: a triangle
    // with no area is left out, and so is a pair that meets at a point, since two patches that
    // share a position are neighbours whatever their indices say.
    let eps2 = epsilon * epsilon;
    let degenerate: Vec<bool> = t
        .triangles
        .iter()
        .map(|tri| {
            let [a, b, c] = tri.map(|i| t.positions[i as usize]);
            (b - a).cross(&(c - a)).norm() * 0.5 < eps2
        })
        .collect();

    let mut faces: HashSet<FaceId> = HashSet::new();
    let mut crossings = 0usize;
    bvh.overlapping_pairs(&t.positions, &t.triangles, epsilon, |a, b| {
        let (a, b) = (a as usize, b as usize);
        if degenerate[a] || degenerate[b] {
            return;
        }
        let (ta, tb) = (t.triangles[a], t.triangles[b]);
        // Neighbours always touch: by a shared vertex, or by a shared position.
        if ta.iter().any(|x| tb.contains(x)) {
            return;
        }
        let p = ta.map(|i| t.positions[i as usize]);
        let q = tb.map(|i| t.positions[i as usize]);
        if p.iter().any(|x| q.iter().any(|y| (x - y).norm() < epsilon)) {
            return;
        }
        // Both at one extraordinary vertex: the evaluator's snap there, not a fold.
        if let (Some(x), Some(y)) = (extraordinary[a], extraordinary[b]) {
            if x == y {
                return;
            }
        }
        if triangles_intersect(&p, &q, epsilon) {
            crossings += 1;
            faces.insert(t.triangle_faces[a]);
            faces.insert(t.triangle_faces[b]);
        }
    });
    let mut faces: Vec<FaceId> = faces.into_iter().collect();
    faces.sort();
    SelfIntersections { faces, crossings }
}

/// Whether two triangles cross, by the separating-axis test with an absolute tolerance: the
/// triangles are apart if some axis among the two normals and the nine edge-pair cross products
/// separates their projections by more than `epsilon`. Exact for triangles that genuinely cross,
/// and quiet for two that touch within the tolerance, which is the case the check must not
/// report.
pub fn triangles_intersect(p: &[Point3<f64>; 3], q: &[Point3<f64>; 3], epsilon: f64) -> bool {
    let mut axes: Vec<Vector3<f64>> = Vec::with_capacity(17);
    let ep = [p[1] - p[0], p[2] - p[1], p[0] - p[2]];
    let eq = [q[1] - q[0], q[2] - q[1], q[0] - q[2]];
    let np = ep[0].cross(&ep[1]);
    let nq = eq[0].cross(&eq[1]);
    axes.push(np);
    axes.push(nq);
    for a in &ep {
        for b in &eq {
            axes.push(a.cross(b));
        }
    }
    // **Two triangles in one plane** -- two of a flat face's grid, say -- are separated by none of
    // the axes above: both normals project them to one value and every edge pair's cross product
    // lies along the normal. The axes that tell them apart are in the plane, across each edge.
    for a in &ep {
        axes.push(np.cross(a));
    }
    for b in &eq {
        axes.push(nq.cross(b));
    }
    for axis in axes {
        let n = axis.norm();
        if n < 1e-18 {
            continue;
        }
        let axis = axis / n;
        let project = |t: &[Point3<f64>; 3]| {
            let d: Vec<f64> = t.iter().map(|v| axis.dot(&v.coords)).collect();
            (d.iter().cloned().fold(f64::INFINITY, f64::min), d.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        };
        let (plo, phi) = project(p);
        let (qlo, qhi) = project(q);
        if plo > qhi - epsilon || qlo > phi - epsilon {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_cage::validate::epsilon_length;
    use fieldform_cage::Cage;

    fn cube_with(move_vertex: Option<(usize, Point3<f64>)>) -> Cage {
        let cube = Cage::cube_of(10.0);
        let mut positions: Vec<Point3<f64>> = cube.vertex_ids().map(|v| cube.vertex_position(v)).collect();
        if let Some((i, p)) = move_vertex {
            positions[i] = p;
        }
        let faces: Vec<Vec<u32>> =
            cube.face_ids().map(|f| cube.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
        Cage::from_faces(positions, &faces).unwrap()
    }

    #[test]
    fn two_crossing_triangles_are_found_and_two_apart_are_not() {
        let a = [Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0), Point3::new(0.0, 2.0, 0.0)];
        let b = [Point3::new(0.5, 0.5, -1.0), Point3::new(0.5, 0.5, 1.0), Point3::new(1.5, 0.5, 0.0)];
        assert!(triangles_intersect(&a, &b, 1e-9));
        let apart = [Point3::new(0.5, 0.5, 0.1), Point3::new(0.5, 0.5, 1.0), Point3::new(1.5, 0.5, 0.5)];
        assert!(!triangles_intersect(&a, &apart, 1e-9));
        // Touching within the tolerance is not a crossing.
        let touching = [Point3::new(0.5, 0.5, 1e-10), Point3::new(0.5, 0.5, 1.0), Point3::new(1.5, 0.5, 0.5)];
        assert!(!triangles_intersect(&a, &touching, 1e-9));
    }

    /// **A clean cube is clean, at its own corners too**: every corner is valence 3, and the
    /// evaluator's snap there is not reported as a fold (the notes' test 4).
    /// Two shells of one cage passing through each other: the plainest self-intersection, and the
    /// faces reported are exactly the six that interpenetrate.
    #[test]
    fn two_overlapping_cubes_in_one_cage_are_found_on_the_faces_that_meet() {
        let a = Cage::cube_of(10.0);
        let mut positions: Vec<Point3<f64>> = a.vertex_ids().map(|v| a.vertex_position(v)).collect();
        let mut faces: Vec<Vec<u32>> =
            a.face_ids().map(|f| a.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
        for v in a.vertex_ids() {
            positions.push(a.vertex_position(v) + Vector3::new(4.0, 3.0, 2.0));
        }
        for f in a.face_ids() {
            faces.push(a.face_vertices(f).iter().map(|v| v.index() as u32 + 8).collect());
        }
        let two = Cage::from_faces(positions, &faces).unwrap();
        let found = find(&two, 4, epsilon_length(&two));
        // The first cube's +X, +Y, +Z and the second's -X, -Y, -Z, in face order.
        assert_eq!(found.faces, [1, 3, 5, 6, 8, 10].map(FaceId::from_index));
    }

    /// **A whole face pushed through the opposite one is an inverted shell, not a crossing** --
    /// the surface turns inside out without passing through itself, which is why that case is
    /// fixed on promotion and never shown red.
    #[test]
    fn a_face_pushed_through_the_opposite_face_turns_the_shell_inside_out_without_crossing() {
        let a = Cage::cube_of(10.0);
        let mut positions: Vec<Point3<f64>> = a.vertex_ids().map(|v| a.vertex_position(v)).collect();
        let faces: Vec<Vec<u32>> =
            a.face_ids().map(|f| a.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
        for p in &mut positions[4..8] {
            p.z = -9.0;
        }
        let pushed = Cage::from_faces(positions, &faces).unwrap();
        assert!(find(&pushed, 4, epsilon_length(&pushed)).is_clean());
    }

    #[test]
    fn a_cube_and_a_refined_cube_are_clean_at_their_extraordinary_corners() {
        for cage in [cube_with(None), fieldform_subd::subdivide(&cube_with(None)).unwrap()] {
            let found = find(&cage, 4, epsilon_length(&cage));
            assert!(found.is_clean(), "{found:?}");
        }
    }

    /// A vertex dragged down through the opposite face: the surface passes through itself, and the
    /// faces reported are the ones around that vertex and the one it went through (test 2).
    ///
    /// **Measured, not assumed, 2026-10-24.** The notes' first fixture dragged the corner
    /// diagonally through the opposite *corner*, and the limit surface stays embedded under that:
    /// a smooth surface deforms where its cage passes through itself, and at depth 4 and 5 no
    /// crossing appears for a corner at (-3,-3,-3) through (-9,-9,-9), nor for one dragged out
    /// sideways to x = -15. Dragged straight down through the bottom face it crosses from about
    /// twice the cube's height below -- (0, 0, -12) gives 66 crossings on faces 0, 3 and 5 --
    /// and that is the fixture.
    #[test]
    fn a_vertex_dragged_through_the_opposite_face_is_a_self_intersection_on_the_right_faces() {
        let cage = cube_with(Some((6, Point3::new(0.0, 0.0, -12.0))));
        let found = find(&cage, 4, epsilon_length(&cage));
        assert!(!found.is_clean());
        assert!(found.crossings > 0);
        // Vertex 6 is on faces +Z (1), +Y (3) and +X (5); it went through -Z (0).
        let moved: HashSet<FaceId> = cage.adjacent_faces(VertexId::from_index(6)).into_iter().collect();
        assert!(found.faces.iter().any(|f| moved.contains(f)), "a face around the moved vertex: {found:?}");
        assert!(found.faces.contains(&FaceId::from_index(0)), "and the face it went through: {found:?}");
        // The diagonal drag the notes first described does not cross, at either depth.
        let diagonal = cube_with(Some((6, Point3::new(-9.0, -9.0, -9.0))));
        assert!(find(&diagonal, 4, epsilon_length(&diagonal)).is_clean());
        assert!(find(&diagonal, 5, epsilon_length(&diagonal)).is_clean());
    }

    /// **Two bodies 0.01 mm apart in one cage are not red** (test 3): the tolerance is a millionth
    /// of the model, and a hundredth of a millimetre on a 20 mm model is a gap.
    #[test]
    fn two_cubes_a_hundredth_of_a_millimetre_apart_are_not_a_self_intersection() {
        let a = Cage::cube_of(10.0);
        let mut positions: Vec<Point3<f64>> = a.vertex_ids().map(|v| a.vertex_position(v)).collect();
        let mut faces: Vec<Vec<u32>> =
            a.face_ids().map(|f| a.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
        for v in a.vertex_ids() {
            positions.push(a.vertex_position(v) + Vector3::new(10.01, 0.0, 0.0));
        }
        for f in a.face_ids() {
            faces.push(a.face_vertices(f).iter().map(|v| v.index() as u32 + 8).collect());
        }
        let two = Cage::from_faces(positions, &faces).unwrap();
        let found = find(&two, 4, epsilon_length(&two));
        assert!(found.is_clean(), "{found:?}");
    }

    #[test]
    fn the_overlap_traversal_finds_every_pair_a_brute_force_pass_finds() {
        let cage = cube_with(Some((6, Point3::new(-9.0, -9.0, -9.0))));
        let t = LimitSurface::new(&cage).tessellate(2);
        let bvh = Bvh::build(&t.positions, &t.triangles);
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        bvh.overlapping_pairs(&t.positions, &t.triangles, 0.0, |a, b| pairs.push((a, b)));
        pairs.sort();
        let before = pairs.len();
        pairs.dedup();
        assert_eq!(before, pairs.len(), "each pair once");
        let boxes: Vec<(Point3<f64>, Point3<f64>)> = t
            .triangles
            .iter()
            .map(|tri| {
                let ps = tri.map(|i| t.positions[i as usize]);
                (ps[0].inf(&ps[1]).inf(&ps[2]), ps[0].sup(&ps[1]).sup(&ps[2]))
            })
            .collect();
        let mut brute = Vec::new();
        for a in 0..boxes.len() {
            for b in a + 1..boxes.len() {
                let (al, ah) = boxes[a];
                let (bl, bh) = boxes[b];
                if (0..3).all(|i| al[i] <= bh[i] && bl[i] <= ah[i]) {
                    brute.push((a as u32, b as u32));
                }
            }
        }
        assert_eq!(pairs, brute);
    }
}
