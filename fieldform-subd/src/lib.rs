// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Catmull-Clark subdivision of a [`Cage`], and exact evaluation of the limit surface it
//! converges to.
//!
//! Originally intended to wrap `opensubdiv-petite` (Pixar's OpenSubdiv), but that crate's
//! bundled C++ fails to build on this project's MSVC toolchain (confirmed by spike; the
//! failure is inside OpenSubdiv's own sources, not ours). So the math lives here instead.
//!
//! Two things are exact, not approximations:
//! - [`subdivide`] implements the standard Catmull-Clark rules. Applied to a closed all-quad
//!   cage it yields another closed all-quad cage, so it composes with itself and preserves
//!   [`Cage`]'s invariants.
//! - [`limit_positions`] gives the *exact* positions the cage's vertices converge to on the
//!   limit surface, in closed form — no iteration, no approximation.
//!
//! Note that a cage vertex's limit position generally differs from its cage position: that's
//! the point of the whole thing (a cube cage's corners pull inward toward a sphere-like
//! form). Faces of the limit surface interpolate between these.
//!
//! [`LimitSurface`] extends that from *vertices* to *arbitrary points*, evaluating the limit
//! surface at any `(u, v)` on a face — exactly where the patch is regular. That is what lets
//! accuracy be a stated bound rather than a hardcoded subdivision level.

mod evaluate;
mod patch;

pub use evaluate::{faces_influenced_by, LimitSurface, Tessellation};

use fieldform_cage::{Cage, CageError, VertexId};
use nalgebra::{Point3, Vector3};

/// Applies one step of Catmull-Clark subdivision, producing a cage with 4x the faces.
///
/// The standard rules, for a closed all-quad mesh:
/// - **Face point**: the centroid of each face's vertices.
/// - **Edge point**: the average of the edge's two endpoints and its two adjacent face
///   points.
/// - **Vertex point**: `(F + 2R + (n-3)P) / n`, where `P` is the original position, `F` the
///   average of adjacent face points, `R` the average of adjacent edge *midpoints*, and `n`
///   the valence.
///
/// Each original quad becomes four new quads, each joining the original vertex, the two
/// edge points flanking it, and the face point.
pub fn subdivide(cage: &Cage) -> Result<Cage, CageError> {
    let vertex_count = cage.vertex_count();
    let face_count = cage.face_count();

    // New vertex layout: [updated original vertices | face points | edge points].
    let mut positions: Vec<Point3<f64>> = Vec::with_capacity(vertex_count + face_count * 3);

    // --- updated original vertex points ---
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        let n = cage.valence(v);

        let face_avg = average(
            cage.adjacent_faces(v)
                .into_iter()
                .map(|f| face_centroid(cage, f)),
        );
        let edge_mid_avg = average(
            cage.adjacent_vertices(v)
                .into_iter()
                .map(|a| midpoint(p, cage.vertex_position(a))),
        );

        let n_f = n as f64;
        let updated = (face_avg.coords + edge_mid_avg.coords * 2.0 + p.coords * (n_f - 3.0)) / n_f;
        positions.push(Point3::from(updated));
    }

    // --- face points ---
    let face_point_base = positions.len();
    for f in cage.face_ids() {
        positions.push(face_centroid(cage, f));
    }

    // --- edge points ---
    // Indexed by the same order `cage.edges()` yields, so faces can look them up below.
    let edge_point_base = positions.len();
    let edge_list: Vec<(VertexId, VertexId)> = cage.edges().collect();
    for &(a, b) in &edge_list {
        let (f0, f1) = cage
            .edge_faces(a, b)
            .expect("a closed cage's every edge has two faces");
        let pa = cage.vertex_position(a);
        let pb = cage.vertex_position(b);
        let sum = pa.coords + pb.coords + face_centroid(cage, f0).coords + face_centroid(cage, f1).coords;
        positions.push(Point3::from(sum / 4.0));
    }

    // Map each undirected edge to its new edge point's index. A linear scan of `edge_list`
    // per face corner would make subdivision O(faces * edges) — quadratic, and painful at
    // the levels needed for a smooth surface — so build the lookup once.
    let edge_lookup: std::collections::HashMap<(usize, usize), u32> = edge_list
        .iter()
        .enumerate()
        .map(|(i, &(a, b))| (undirected(a, b), (edge_point_base + i) as u32))
        .collect();
    let edge_index = |a: VertexId, b: VertexId| -> u32 {
        *edge_lookup
            .get(&undirected(a, b))
            .expect("every face edge is in the cage's edge list")
    };

    // --- new faces: each quad splits into 4 ---
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(face_count * 4);
    for f in cage.face_ids() {
        let corners = cage.face_vertices(f);
        let fp = (face_point_base + f.index()) as u32;
        for i in 0..4 {
            let prev = corners[(i + 3) % 4];
            let cur = corners[i];
            let next = corners[(i + 1) % 4];
            faces.push(vec![
                cur.index() as u32,
                edge_index(cur, next),
                fp,
                edge_index(prev, cur),
            ]);
        }
    }

    Cage::from_faces(positions, &faces)
}

/// Applies [`subdivide`] `levels` times.
pub fn subdivide_n(cage: &Cage, levels: usize) -> Result<Cage, CageError> {
    let mut current = subdivide(cage)?;
    for _ in 1..levels {
        current = subdivide(&current)?;
    }
    Ok(current)
}

/// Exact limit positions of the cage's vertices — where each converges after infinite
/// subdivision, in closed form.
///
/// For a vertex of valence `n` with adjacent vertices `A_i` and, per adjacent quad, the
/// vertex diagonally opposite (`D_i`):
///
/// ```text
/// P_limit = (n^2 * P + 4 * sum(A_i) + sum(D_i)) / (n * (n + 5))
/// ```
///
/// At `n = 4` this reduces to the standard bicubic B-spline limit mask
/// `(16P + 4*sum(A) + sum(D)) / 36`, which is the correctness check to reason from.
pub(crate) fn limit_positions(cage: &Cage) -> Vec<Point3<f64>> {
    cage.vertex_ids()
        .map(|v| limit_position(cage, v))
        .collect()
}

/// Exact limit position of a single vertex — see [`limit_positions`].
pub fn limit_position(cage: &Cage, v: VertexId) -> Point3<f64> {
    let p = cage.vertex_position(v);
    let n = cage.valence(v);
    let n_f = n as f64;

    let adjacent_sum = cage
        .adjacent_vertices(v)
        .into_iter()
        .fold(Vector3::zeros(), |acc, a| acc + cage.vertex_position(a).coords);

    // The diagonal vertex of each incident quad: the one that shares no edge with `v`.
    let diagonal_sum = cage
        .adjacent_faces(v)
        .into_iter()
        .fold(Vector3::zeros(), |acc, f| {
            let corners = cage.face_vertices(f);
            let i = corners
                .iter()
                .position(|&c| c == v)
                .expect("an incident face contains this vertex");
            acc + cage.vertex_position(corners[(i + 2) % 4]).coords
        });

    Point3::from(
        (p.coords * n_f * n_f + adjacent_sum * 4.0 + diagonal_sum) / (n_f * (n_f + 5.0)),
    )
}

/// A cage whose vertices sit at their limit positions. Subdividing a few times and then
/// pushing to the limit gives a dense, smooth mesh that lies *on* the limit surface (rather
/// than the control cage, which merely bounds it).
pub fn limit_cage(cage: &Cage) -> Result<Cage, CageError> {
    let positions = limit_positions(cage);
    let faces: Vec<Vec<u32>> = cage
        .face_ids()
        .map(|f| {
            cage.face_vertices(f)
                .iter()
                .map(|v| v.index() as u32)
                .collect()
        })
        .collect();
    Cage::from_faces(positions, &faces)
}

/// Canonical (order-independent) key for an undirected edge.
fn undirected(a: VertexId, b: VertexId) -> (usize, usize) {
    let (i, j) = (a.index(), b.index());
    if i < j { (i, j) } else { (j, i) }
}

fn face_centroid(cage: &Cage, f: fieldform_cage::FaceId) -> Point3<f64> {
    let sum = cage
        .face_vertices(f)
        .iter()
        .fold(Vector3::zeros(), |acc, v| acc + cage.vertex_position(*v).coords);
    Point3::from(sum / 4.0)
}

fn midpoint(a: Point3<f64>, b: Point3<f64>) -> Point3<f64> {
    Point3::from((a.coords + b.coords) / 2.0)
}

fn average(points: impl Iterator<Item = Point3<f64>>) -> Point3<f64> {
    let mut sum = Vector3::zeros();
    let mut count = 0.0;
    for p in points {
        sum += p.coords;
        count += 1.0;
    }
    Point3::from(sum / count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn subdivision_quadruples_faces_and_stays_a_valid_cage() {
        let cube = Cage::cube();
        let s1 = subdivide(&cube).expect("subdividing a valid cage yields a valid cage");
        // 6 quads -> 24. Vertices: 8 original + 6 face points + 12 edge points = 26.
        // Edges: each of 24 quads has 4 edges, each shared by 2 faces = 48.
        assert_eq!(s1.face_count(), 24);
        assert_eq!(s1.vertex_count(), 26);
        assert_eq!(s1.edge_count(), 48);
        // Still a closed genus-0 surface.
        assert_eq!(s1.euler_characteristic(), 2);

        let s2 = subdivide(&s1).unwrap();
        assert_eq!(s2.face_count(), 96);
        assert_eq!(s2.euler_characteristic(), 2);
    }

    #[test]
    fn subdivision_preserves_the_cube_centroid() {
        // Catmull-Clark is an affine-invariant averaging scheme, so a symmetric shape's
        // centroid must not drift.
        let cube = Cage::cube();
        let s = subdivide_n(&cube, 3).unwrap();
        let centroid = s
            .vertex_ids()
            .fold(Vector3::zeros(), |acc, v| acc + s.vertex_position(v).coords)
            / s.vertex_count() as f64;
        assert_relative_eq!(centroid.x, 0.0, epsilon = 1e-5);
        assert_relative_eq!(centroid.y, 0.0, epsilon = 1e-5);
        assert_relative_eq!(centroid.z, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn cube_corner_limit_position_matches_hand_computation() {
        // Hand-computed for the unit cube from Cage::cube(), corner (-0.5, -0.5, -0.5):
        //   valence n = 3, so n(n+5) = 24 and n^2 = 9
        //   sum(adjacent)  = (0.5,-0.5,-0.5) + (-0.5,0.5,-0.5) + (-0.5,-0.5,0.5)
        //                  = (-0.5, -0.5, -0.5)
        //   sum(diagonals) = (0.5,0.5,-0.5) + (0.5,-0.5,0.5) + (-0.5,0.5,0.5)
        //                  = (0.5, 0.5, 0.5)
        //   numerator = 9*(-0.5,-0.5,-0.5) + 4*(-0.5,-0.5,-0.5) + (0.5,0.5,0.5)
        //             = (-6, -6, -6)
        //   limit = (-6,-6,-6)/24 = (-0.25, -0.25, -0.25)
        let cube = Cage::cube();
        let v0 = cube
            .vertex_ids()
            .find(|&v| {
                let p = cube.vertex_position(v);
                p.x < 0.0 && p.y < 0.0 && p.z < 0.0
            })
            .unwrap();
        let limit = limit_position(&cube, v0);
        assert_relative_eq!(limit.x, -0.25, epsilon = 1e-6);
        assert_relative_eq!(limit.y, -0.25, epsilon = 1e-6);
        assert_relative_eq!(limit.z, -0.25, epsilon = 1e-6);
    }

    #[test]
    fn limit_surface_pulls_the_cube_inward_toward_a_sphere() {
        // The whole premise of field mode: a cube cage's limit surface is a rounded,
        // sphere-like form strictly inside the cage.
        let cube = Cage::cube();
        let cage_corner_radius = 0.5f64 * 3.0f64.sqrt(); // ~0.866
        for p in limit_positions(&cube) {
            assert!(
                p.coords.norm() < cage_corner_radius,
                "limit point {p:?} should lie strictly inside the cage's corner radius"
            );
        }

        // Subdividing repeatedly and taking the limit should converge toward a consistent
        // smooth surface: every point roughly equidistant from the centre.
        let dense = limit_cage(&subdivide_n(&cube, 3).unwrap()).unwrap();
        let radii: Vec<f64> = dense
            .vertex_ids()
            .map(|v| dense.vertex_position(v).coords.norm())
            .collect();
        let min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = radii.iter().cloned().fold(0.0f64, f64::max);
        // Not a true sphere (it's a subdivided cube), but it should be far rounder than the
        // cage, whose radii range over a factor of sqrt(3) ~ 1.73.
        assert!(
            max / min < 1.2,
            "subdivided limit surface should be near-spherical, got min={min} max={max}"
        );
    }

    #[test]
    fn regular_valence_4_limit_matches_the_bicubic_bspline_mask() {
        // After one subdivision, the original cube face points become valence-4 vertices.
        // For those, the general formula must reduce exactly to the standard bicubic
        // B-spline limit mask (16P + 4*sum(A) + sum(D)) / 36.
        let s = subdivide(&Cage::cube()).unwrap();
        let regular: Vec<_> = s.vertex_ids().filter(|&v| s.valence(v) == 4).collect();
        assert!(!regular.is_empty(), "expected some valence-4 vertices");

        for v in regular {
            let p = s.vertex_position(v);
            let adj: Vector3<f64> = s
                .adjacent_vertices(v)
                .into_iter()
                .fold(Vector3::zeros(), |acc, a| acc + s.vertex_position(a).coords);
            let diag: Vector3<f64> = s.adjacent_faces(v).into_iter().fold(
                Vector3::zeros(),
                |acc, f| {
                    let corners = s.face_vertices(f);
                    let i = corners.iter().position(|&c| c == v).unwrap();
                    acc + s.vertex_position(corners[(i + 2) % 4]).coords
                },
            );
            let expected = Point3::from((p.coords * 16.0 + adj * 4.0 + diag) / 36.0);
            let actual = limit_position(&s, v);
            assert_relative_eq!(actual.x, expected.x, epsilon = 1e-6);
            assert_relative_eq!(actual.y, expected.y, epsilon = 1e-6);
            assert_relative_eq!(actual.z, expected.z, epsilon = 1e-6);
        }
    }
}
