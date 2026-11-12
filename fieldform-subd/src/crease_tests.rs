// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Sharp creases: the rules in both subdivisions, the fast path's reflection across a crease, and
//! the descent that handles every other creased vertex -- each held against the one it must agree
//! with.

use crate::evaluate::regular_grid;
use crate::{limit_position, subdivide, subdivide_n, LimitSurface};
use fieldform_cage::{Cage, FaceId, VertexId};
use nalgebra::{Point3, Vector3};

/// Moves every vertex by a fixed pseudo-random amount, so nothing below passes because the cage
/// happened to be symmetric or planar.
fn jittered(cage: &Cage, amount: f64) -> Cage {
    let mut out = cage.clone();
    for v in cage.vertex_ids() {
        let h = |k: usize| (((v.index() * 7919 + k * 104_729) % 1000) as f64 / 1000.0 - 0.5) * amount;
        let p = cage.vertex_position(v);
        out.set_vertex_position(v, p + Vector3::new(h(1), h(2), h(3)));
    }
    out
}

/// Where `(u, v)` on a face lands one subdivision down: the child face and its coordinates, in
/// the order `subdivide` emits a face's four children.
fn child_of(face: FaceId, u: f64, v: f64) -> (FaceId, f64, f64) {
    let (q, cu, cv) = if u < 0.5 && v < 0.5 {
        (0, 2.0 * u, 2.0 * v)
    } else if v < 0.5 {
        (1, 2.0 * v, 2.0 * (1.0 - u))
    } else if u >= 0.5 {
        (2, 2.0 * (1.0 - u), 2.0 * (1.0 - v))
    } else {
        (3, 2.0 * (1.0 - v), 2.0 * u)
    };
    (FaceId::from_index(face.index() * 4 + q), cu, cv)
}

/// A curved cage with every kind of creased vertex on it: a straight run, darts at its ends, a
/// loop round one face that turns at each corner (one of them a valence-three pole), and three
/// creases meeting at a corner.
pub(crate) fn every_kind_of_crease() -> Cage {
    let mut cage = jittered(&subdivide_n(&Cage::cube_of(10.0), 2).unwrap(), 0.8);
    // A loop round face 0: its corner 0 is an original cube corner (valence three, so a pole on
    // a crease that turns there); its other corners are valence four and the loop turns at each.
    let ring = cage.face_vertices(FaceId::from_index(0));
    for k in 0..4 {
        cage.set_creased(ring[k], ring[(k + 1) % 4], true);
    }
    // Three creases meeting: every edge at an original corner far from face 0.
    let corner = cage
        .vertex_ids()
        .filter(|&v| cage.valence(v) == 3 && !ring.contains(&v))
        .max_by(|&a, &b| {
            let d = |v: VertexId| (cage.vertex_position(v) - cage.vertex_position(ring[0])).norm();
            d(a).partial_cmp(&d(b)).unwrap()
        })
        .unwrap();
    for n in cage.adjacent_vertices(corner) {
        cage.set_creased(corner, n, true);
    }
    // A straight run of three edges through two valence-four vertices, touching nothing creased
    // so far: its ends are darts, and the faces beside its middle edge are fast-path faces.
    let touched = |cage: &Cage, v: VertexId| cage.creases_at(v) > 0 || cage.adjacent_vertices(v).iter().any(|&n| cage.creases_at(n) > 0);
    let quiet = |cage: &Cage, v: VertexId| {
        cage.valence(v) == 4 && !touched(cage, v) && cage.adjacent_vertices(v).iter().all(|&n| !touched(cage, n))
    };
    // Round a vertex of four, the neighbour opposite `from`.
    let opposite = |cage: &Cage, at: VertexId, from: VertexId| {
        let n = cage.adjacent_vertices(at);
        n[(n.iter().position(|&x| x == from).unwrap() + 2) % 4]
    };
    let (m0, m1, m2, m3) = cage
        .vertex_ids()
        .filter(|&v| quiet(&cage, v))
        .find_map(|m1| {
            let n = cage.adjacent_vertices(m1);
            let (m0, m2) = (n[0], n[2]);
            if !quiet(&cage, m2) {
                return None;
            }
            let m3 = opposite(&cage, m2, m1);
            (cage.valence(m0) == 4 && cage.valence(m3) == 4).then_some((m0, m1, m2, m3))
        })
        .expect("a quiet straight run of three edges");
    cage.set_creased(m0, m1, true);
    cage.set_creased(m1, m2, true);
    cage.set_creased(m2, m3, true);
    cage
}

/// The fixture has what it claims, or the tests below prove nothing about it.
#[test]
fn the_fixture_has_a_straight_run_darts_turns_a_pole_and_a_junction() {
    let cage = every_kind_of_crease();
    let class = |v: VertexId| (cage.creases_at(v), cage.crease_runs_straight(v), cage.valence(v));
    let all: Vec<_> = cage.vertex_ids().map(class).collect();
    assert!(all.iter().any(|&(k, s, _)| k == 2 && s), "a straight crease vertex");
    assert!(all.iter().any(|&(k, _, _)| k == 1), "a dart");
    assert!(all.iter().any(|&(k, s, n)| k == 2 && !s && n == 4), "a turn at valence four");
    assert!(all.iter().any(|&(k, _, n)| k == 2 && n == 3), "a pole on a crease");
    assert!(all.iter().any(|&(k, _, _)| k >= 3), "a junction");
}

/// **Refine keeps a fully creased surface exactly.** Subdividing with the sharp rules and both
/// halves creased gives the same limit surface, so the original evaluated at `(u, v)` and its
/// child evaluated where `(u, v)` lands must agree. Sampled away from every corner's last eighth,
/// the one place the two evaluations legitimately differ (the descent stops there and takes the
/// corner's limit). This holds the fast path, the descent and both sets of sharp rules together.
#[test]
fn a_refined_creased_cage_has_the_limit_surface_of_the_one_it_came_from() {
    let coarse = every_kind_of_crease();
    let fine = subdivide(&coarse).unwrap();
    assert_eq!(fine.crease_count(), coarse.crease_count() * 2, "each creased edge is two halves");
    let mut a = LimitSurface::new(&coarse);
    let mut b = LimitSurface::new(&fine);
    let mut worst = 0.0f64;
    for f in coarse.face_ids() {
        for i in 1..7 {
            for j in 1..7 {
                let (u, v) = (i as f64 / 7.0, j as f64 / 7.0);
                let p = a.evaluate(f, u, v).unwrap();
                let (child, cu, cv) = child_of(f, u, v);
                let q = b.evaluate(child, cu, cv).unwrap();
                worst = worst.max((p - q).norm());
            }
        }
    }
    assert!(worst < 1e-9, "refined surface moved by {worst} mm");

    // The control: without the sharp rules on the refined side, the same comparison must fail,
    // or it could not see a wrong rule.
    let mut smooth = fine.clone();
    smooth.clear_creases();
    let mut c = LimitSurface::new(&smooth);
    let mut moved = 0.0f64;
    for f in coarse.face_ids() {
        let p = a.evaluate(f, 0.5, 0.3).unwrap();
        let (child, cu, cv) = child_of(f, 0.5, 0.3);
        moved = moved.max((p - c.evaluate(child, cu, cv).unwrap()).norm());
    }
    assert!(moved > 1e-3, "the creases changed nothing ({moved} mm), so the test is blind");
}

/// **The fast path's reflection across a crease is the subdivision's crease.** Every face beside
/// a straight crease is evaluated in closed form by reflecting across it; one subdivision with the
/// sharp rules, evaluated in its children, is the answer it has to give. Restricted to the faces
/// the fast path takes, so a wrong reflection cannot hide behind the descent.
#[test]
fn a_face_beside_a_straight_crease_is_the_patch_the_sharp_rules_make() {
    let coarse = every_kind_of_crease();
    let fine = subdivide(&coarse).unwrap();
    let mut a = LimitSurface::new(&coarse);
    let mut b = LimitSurface::new(&fine);
    let beside: Vec<FaceId> = coarse
        .face_ids()
        .filter(|&f| regular_grid(&coarse, f).is_some())
        .filter(|&f| {
            let c = coarse.face_vertices(f);
            (0..4).any(|k| coarse.is_creased(c[k], c[(k + 1) % 4]))
        })
        .collect();
    assert!(!beside.is_empty(), "no fast-path face beside a crease");
    for f in beside {
        for i in 0..=8 {
            for j in 0..=8 {
                let (u, v) = (i as f64 / 8.0, j as f64 / 8.0);
                let p = a.evaluate(f, u, v).unwrap();
                let (child, cu, cv) = child_of(f, u, v);
                let q = b.evaluate(child, cu, cv).unwrap();
                assert!((p - q).norm() < 1e-9, "face {f:?} at ({u}, {v}): {} mm", (p - q).norm());
            }
        }
    }
}

/// **The two sides of a crease do not see each other.** Moving a vertex above a creased loop
/// leaves every face below it exactly where it was -- bit for bit, since nothing below reads it --
/// where the same move on the uncreased cage moves faces below.
#[test]
fn moving_one_side_of_a_crease_leaves_the_other_side_bit_for_bit() {
    let base = subdivide_n(&Cage::cube_of(10.0), 2).unwrap();
    let equator: Vec<(VertexId, VertexId)> = base
        .edges()
        .filter(|&(a, b)| base.vertex_position(a).z.abs() < 1e-9 && base.vertex_position(b).z.abs() < 1e-9)
        .collect();
    assert!(equator.len() >= 8);
    let mut creased = jittered(&base, 0.5);
    let plain = creased.clone();
    for &(a, b) in &equator {
        creased.set_creased(a, b, true);
    }
    // On or below the equator: the row touching it from below is the one a move above would
    // reach, if anything could.
    let below: Vec<FaceId> =
        base.face_ids().filter(|&f| base.face_vertices(f).iter().all(|&v| base.vertex_position(v).z < 1e-9)).collect();
    // A vertex one row above the equator.
    let lifted = base
        .vertex_ids()
        .filter(|&v| base.vertex_position(v).z > 1e-9)
        .min_by(|&a, &b| base.vertex_position(a).z.partial_cmp(&base.vertex_position(b).z).unwrap())
        .unwrap();

    let sample = |cage: &Cage| -> Vec<Point3<f64>> {
        let mut s = LimitSurface::new(cage);
        below.iter().flat_map(|&f| [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.2, 0.9)].map(|(u, v)| s.evaluate(f, u, v).unwrap())).collect()
    };
    let moved = |cage: &Cage| {
        let mut c = cage.clone();
        c.set_vertex_position(lifted, cage.vertex_position(lifted) + Vector3::new(0.0, 0.0, 3.0));
        c
    };
    let (before, after) = (sample(&creased), sample(&moved(&creased)));
    let differing: Vec<usize> = (0..before.len()).filter(|&i| before[i] != after[i]).collect();
    assert!(differing.is_empty(), "faces below the crease moved: {:?}", differing.iter().map(|&i| (below[i / 4], i % 4, (before[i] - after[i]).norm())).collect::<Vec<_>>());
    assert_ne!(sample(&plain), sample(&moved(&plain)), "the control: uncreased, the move reaches below");
}

/// **A cube with every edge creased is a cube**: corners where three creases meet do not move,
/// each edge is a straight crease between them, and each face is its own flat square.
#[test]
fn a_cube_with_every_edge_creased_is_a_cube() {
    let mut cube = Cage::cube_of(10.0);
    for (a, b) in cube.edges().collect::<Vec<_>>() {
        cube.set_creased(a, b, true);
    }
    let t = LimitSurface::new(&cube).tessellate(4);
    for p in &t.positions {
        let biggest = p.x.abs().max(p.y.abs()).max(p.z.abs());
        assert!((biggest - 5.0).abs() < 1e-9, "{p:?} is not on the cube's surface");
    }
    let centre = LimitSurface::new(&cube).evaluate(FaceId::from_index(1), 0.5, 0.5).unwrap();
    assert!((centre - Point3::new(0.0, 0.0, 5.0)).norm() < 1e-9, "the +Z face's centre is {centre:?}");
}

/// A vertex on a straight crease lands on the crease's B-spline, `(1, 4, 1)/6` of its crease
/// neighbours, and the closed form, the evaluator and a deep subdivision all say so.
#[test]
fn a_crease_vertex_lands_on_its_crease_curve() {
    let cage = every_kind_of_crease();
    let v = cage.vertex_ids().find(|&v| cage.crease_runs_straight(v)).unwrap();
    let ends = cage.creased_neighbours(v);
    let expected = Point3::from((cage.vertex_position(ends[0]).coords + cage.vertex_position(v).coords * 4.0 + cage.vertex_position(ends[1]).coords) / 6.0);
    assert!((limit_position(&cage, v) - expected).norm() < 1e-12);
    let face = cage.adjacent_faces(v)[0];
    let k = cage.face_vertices(face).iter().position(|&c| c == v).unwrap();
    let (u, w) = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)][k];
    let evaluated = LimitSurface::new(&cage).evaluate(face, u, w).unwrap();
    assert!((evaluated - expected).norm() < 1e-9, "{evaluated:?} vs {expected:?}");
}

/// **The two sides of a crease shade with their own normals.** On the creased cube a triangle
/// touching an edge carries its own face's normal at its corners on the edge, and the welded
/// vertex there carries the bisector -- never one side's, which would depend on which face was
/// tessellated first.
#[test]
fn a_crease_keeps_each_sides_normal_and_its_vertices_take_the_bisector() {
    let mut cube = Cage::cube_of(10.0);
    for (a, b) in cube.edges().collect::<Vec<_>>() {
        cube.set_creased(a, b, true);
    }
    let t = LimitSurface::new(&cube).tessellate(3);
    assert!(!t.crease_corner_normals.is_empty());
    for &(tri, corners) in &t.crease_corner_normals {
        let [a, b, c] = t.triangles[tri as usize].map(|i| t.positions[i as usize]);
        let facet = (b - a).cross(&(c - a)).normalize();
        for n in corners {
            assert!((n - facet).norm() < 1e-9, "a corner normal {n:?} on a flat face whose normal is {facet:?}");
        }
    }
    // A vertex in the middle of an edge (+X, +Z) has the bisector.
    let on_edge = t.positions.iter().position(|p| (p.x - 5.0).abs() < 1e-9 && (p.z - 5.0).abs() < 1e-9 && p.y.abs() < 1.0).unwrap();
    let expected = Vector3::new(1.0, 0.0, 1.0).normalize();
    assert!((t.normals[on_edge] - expected).norm() < 1e-9, "{:?}", t.normals[on_edge]);
}

/// **An open rim is the B-spline of its vertices, whichever way it runs in a face.** Each rim
/// vertex's limit is `(1, 4, 1)/6` of itself and its two rim neighbours, and a mesh corner stays
/// where it is. The corner of a face's control grid used to be reflected in `v` even when the rim
/// ran along the face's `u` side, which keeps the surface continuous -- the two faces at a rim
/// vertex make the same mistake -- and puts it in the wrong place: on this panel a rim vertex
/// landed off its limit on every face whose `u` side was the rim, and on none whose `v` side was.
/// The same reflection is what a crease uses on every face beside it.
#[test]
fn an_open_rim_passes_through_its_vertices_limits_on_every_face() {
    let n = 4usize;
    let positions: Vec<Point3<f64>> = (0..=n)
        .flat_map(|j| (0..=n).map(move |i| (i, j)))
        .map(|(i, j)| {
            let (x, y) = (i as f64 * 10.0, j as f64 * 10.0);
            let wobble = (((i * 31 + j * 17) % 7) as f64 - 3.0) * 0.6;
            Point3::new(x, y, 0.02 * x * y - 0.01 * x * x + wobble)
        })
        .collect();
    let faces: Vec<Vec<u32>> = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .map(|(i, j)| {
            let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
            vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]
        })
        .collect();
    let mesh = fieldform_cage::QuadMesh::from_faces(positions, &faces).unwrap();
    let mut s = LimitSurface::over(&mesh);
    let corner_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let mut worst = 0.0f64;
    for v in mesh.vertex_ids().filter(|&v| mesh.vertex_on_boundary(v)) {
        let p = mesh.vertex_position(v);
        let rim: Vec<VertexId> = mesh.adjacent_vertices(v).into_iter().filter(|&w| mesh.vertex_on_boundary(w) && mesh.edge_faces(v, w).is_none()).collect();
        let expected = match rim.as_slice() {
            [a, b] if mesh.valence(v) == 2 => Point3::from((mesh.vertex_position(*a).coords + p.coords * 4.0 + mesh.vertex_position(*b).coords) / 6.0),
            _ => p,
        };
        for f in mesh.adjacent_faces(v) {
            let k = mesh.face_vertices(f).iter().position(|&c| c == v).unwrap();
            let (u, w) = corner_uv[k];
            worst = worst.max((s.evaluate(f, u, w).unwrap() - expected).norm());
        }
    }
    assert!(worst < 1e-9, "a rim vertex is {worst} mm off its limit");
}

/// An uncreased cage is evaluated exactly as before creases existed: nothing above is read.
#[test]
fn an_uncreased_cage_has_no_crease_corners() {
    let t = LimitSurface::new(&jittered(&subdivide_n(&Cage::cube_of(10.0), 1).unwrap(), 0.5)).tessellate(3);
    assert!(t.crease_corner_normals.is_empty());
}
