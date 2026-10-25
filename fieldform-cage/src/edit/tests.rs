// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The primitives held to the cage-editing notes §4: quad and manifold by construction, the
//! Euler characteristic moving as the topology says, untouched indices unmoved, refusals refusing.

use super::*;
use crate::Cage;

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn mesh(positions: Vec<Point3<f64>>, faces: &[Vec<u32>]) -> QuadMesh {
    QuadMesh::from_faces(positions, faces).expect("a valid corpus mesh")
}

pub(crate) fn cube() -> QuadMesh {
    Cage::cube().mesh().clone()
}

/// One linear split: every quad into four, through its edge midpoints and centre. Closed stays
/// closed and every face of a cube becomes a 2×2 grid, which is the refined cube the spec's corpus
/// names without needing `fieldform-subd` here.
pub(crate) fn split(m: &QuadMesh) -> QuadMesh {
    let mut positions = m.position_list();
    let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut faces = Vec::new();
    for quad in m.quad_lists() {
        let mut midpoint = |a: u32, b: u32, positions: &mut Vec<Point3<f64>>| {
            *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                positions.push(lerp(positions[a as usize], positions[b as usize], 0.5));
                (positions.len() - 1) as u32
            })
        };
        let c = &quad;
        let m01 = midpoint(c[0], c[1], &mut positions);
        let m12 = midpoint(c[1], c[2], &mut positions);
        let m23 = midpoint(c[2], c[3], &mut positions);
        let m30 = midpoint(c[3], c[0], &mut positions);
        let centre = Point3::from(
            c.iter().map(|&i| positions[i as usize].coords).sum::<nalgebra::Vector3<f64>>() / 4.0,
        );
        positions.push(centre);
        let f = (positions.len() - 1) as u32;
        faces.push(vec![c[0], m01, f, m30]);
        faces.push(vec![c[1], m12, f, m01]);
        faces.push(vec![c[2], m23, f, m12]);
        faces.push(vec![c[3], m30, f, m23]);
    }
    mesh(positions, &faces)
}

/// A tube of `around × along` quads; with `wrap` its two ends are joined into a torus. Ring `j`
/// sits at angle `j` of `along`, so the last ring of an open tube lies exactly on the first.
pub(crate) fn torus_grid(around: usize, along: usize, wrap: bool) -> QuadMesh {
    let rings = if wrap { along } else { along + 1 };
    let mut positions = Vec::new();
    for j in 0..rings {
        let phi = std::f64::consts::TAU * j as f64 / along as f64;
        for i in 0..around {
            let theta = std::f64::consts::TAU * i as f64 / around as f64;
            let r = 3.0 + theta.cos();
            positions.push(Point3::new(r * phi.cos(), r * phi.sin(), theta.sin()));
        }
    }
    let at = |i: usize, j: usize| ((j % rings) * around + i % around) as u32;
    let mut faces = Vec::new();
    for j in 0..along {
        for i in 0..around {
            faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    mesh(positions, &faces)
}

/// A flat `w × h` grid of unit quads; `index(i, j)` is the vertex at column `i`, row `j`.
pub(crate) fn grid(w: usize, h: usize, skip: &[(usize, usize)]) -> QuadMesh {
    let mut positions = Vec::new();
    for j in 0..=h {
        for i in 0..=w {
            positions.push(Point3::new(i as f64, j as f64, 0.0));
        }
    }
    let at = |i: usize, j: usize| (j * (w + 1) + i) as u32;
    let mut faces = Vec::new();
    for j in 0..h {
        for i in 0..w {
            if !skip.contains(&(i, j)) {
                faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
    }
    let (positions, faces, _) = without_unused(positions, faces);
    mesh(positions, &faces)
}

/// Five quads round a pentagonal hole: both of its boundary loops have five edges.
fn pentagon_ring() -> QuadMesh {
    let mut positions = Vec::new();
    for r in [1.0, 2.0] {
        for k in 0..5 {
            let a = std::f64::consts::TAU * k as f64 / 5.0;
            positions.push(Point3::new(r * a.cos(), r * a.sin(), 0.0));
        }
    }
    let faces: Vec<Vec<u32>> = (0..5u32)
        .map(|k| vec![k, 5 + k, 5 + (k + 1) % 5, (k + 1) % 5])
        .collect();
    mesh(positions, &faces)
}

/// §4.1: whatever a primitive returns rebuilds through the constructor unchanged — asserted, so an
/// in-place shortcut that skipped the rebuild would fail here.
fn assert_rebuilds(m: &QuadMesh) {
    let again = QuadMesh::from_faces(m.position_list(), &m.quad_lists())
        .expect("a primitive's result is a valid mesh");
    assert_eq!(again.geometry_fingerprint(), m.geometry_fingerprint());
}

/// §4.3: every vertex the primitive did not make is where it was, bit for bit, at the same index.
fn assert_old_vertices_unmoved(before: &QuadMesh, after: &QuadMesh) {
    for i in 0..before.vertex_count() {
        assert_eq!(
            after.vertex_position(v(i)).coords.map(f64::to_bits),
            before.vertex_position(v(i)).coords.map(f64::to_bits),
            "vertex {i} moved"
        );
    }
}

/// The edge `(a, b)` of a mesh, found by position — the corpus is built by hand, so this is how a
/// test names "the right-hand edge" without counting indices.
fn edge_at(m: &QuadMesh, a: [f64; 3], b: [f64; 3]) -> (VertexId, VertexId) {
    let find = |p: [f64; 3]| {
        m.vertex_ids()
            .find(|&x| (m.vertex_position(x) - Point3::from(p)).norm() < 1e-9)
            .unwrap_or_else(|| panic!("no vertex at {p:?}"))
    };
    (find(a), find(b))
}

#[test]
fn a_ring_round_a_cube_splits_four_faces_and_leaves_it_closed() {
    let before = cube();
    let result = insert_ring(&before, (v(0), v(1)), 0.25).unwrap();
    assert_rebuilds(&result.mesh);
    assert_eq!(result.mesh.face_count(), 10, "four faces of the ring, each now two");
    assert_eq!(result.new_vertices.len(), 4);
    assert_eq!(result.replaced.len(), 4);
    assert_eq!(result.mesh.euler_characteristic(), 2, "χ unchanged");
    assert!(Cage::try_from_mesh(result.mesh.clone()).is_ok(), "still closed");
    assert_old_vertices_unmoved(&before, &result.mesh);
}

/// **The loop is straight.** Every ring edge on a cube is parallel to the one named, so a loop
/// that carried `t` from the same side all the way round puts every new vertex at x = −0.25; one
/// that measured from alternate ends would put half of them at +0.25.
#[test]
fn a_ring_carries_t_from_the_same_side_all_the_way_round() {
    let result = insert_ring(&cube(), (v(0), v(1)), 0.25).unwrap();
    for &n in &result.new_vertices {
        let x = result.mesh.vertex_position(n).x;
        assert!((x + 0.25).abs() < 1e-12, "a new vertex at x = {x}");
    }
    // Named from the other end, the same loop — walked from another half-edge, so in another index
    // order, which is why positions are compared rather than fingerprints.
    let other = insert_ring(&cube(), (v(1), v(0)), 0.75).unwrap();
    let mut a: Vec<[u64; 3]> = result.new_vertices.iter()
        .map(|&n| result.mesh.vertex_position(n).coords.map(f64::to_bits).into()).collect();
    let mut b: Vec<[u64; 3]> = other.new_vertices.iter()
        .map(|&n| other.mesh.vertex_position(n).coords.map(f64::to_bits).into()).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

/// A split face keeps its slot for its first half, so untouched faces keep their indices.
#[test]
fn a_ring_keeps_every_untouched_face_at_its_index() {
    let before = split(&cube());
    let result = insert_ring(&before, (v(0), v(8)), 0.5).unwrap();
    for f in before.face_ids() {
        if !result.replaced.contains_key(&f) {
            assert_eq!(result.mesh.face_vertices(f), before.face_vertices(f), "face {f:?}");
        }
    }
    assert_eq!(result.mesh.euler_characteristic(), 2);
}

#[test]
fn a_ring_on_an_open_strip_stops_at_its_rim() {
    // A loop crossing an end of the strip runs its whole length: every face, split once.
    let strip = grid(3, 1, &[]);
    let (a, b) = edge_at(&strip, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let result = insert_ring(&strip, (a, b), 0.5).unwrap();
    assert_rebuilds(&result.mesh);
    assert_eq!(result.mesh.face_count(), 6, "all three faces, each split once");
    assert_eq!(result.mesh.euler_characteristic(), 1, "still a disc");
    // A loop crossing a side of the strip meets the other side at once: one face.
    let (a, b) = edge_at(&strip, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let result = insert_ring(&strip, (a, b), 0.5).unwrap();
    assert_eq!(result.mesh.face_count(), 4, "the one face between the strip's two sides");
}

#[test]
fn a_ring_at_an_end_of_its_edge_is_refused() {
    let err = insert_ring(&cube(), (v(0), v(1)), 1.0).unwrap_err();
    assert_eq!(err.highlight.edges, vec![(v(0), v(1))]);
    assert!(err.cause.is_none(), "a refusal, not a bug");
}

/// §4.2: delete one face of a closed cage and χ drops by one and it is open; fill the hole and χ
/// is back and it promotes.
#[test]
fn deleting_a_face_opens_the_cage_and_filling_the_hole_closes_it() {
    let before = cube();
    let deleted = delete_faces(&before, &[FaceId::from_index(1)]).unwrap();
    assert_rebuilds(&deleted.mesh);
    assert!(deleted.mesh.is_open());
    assert_eq!(deleted.mesh.euler_characteristic(), 1);
    assert_eq!(deleted.mesh.vertex_count(), 8, "every corner still has a face");
    let face_map = deleted.face_map.as_ref().unwrap();
    assert_eq!(face_map[1], None);
    assert_eq!(face_map[2], Some(FaceId::from_index(1)));

    let rim = deleted.mesh.edges().find(|&(a, b)| {
        deleted.mesh.directed(a, b).is_some_and(|h| deleted.mesh.is_boundary(h))
            || deleted.mesh.directed(b, a).is_some_and(|h| deleted.mesh.is_boundary(h))
    });
    let filled = fill_hole(&deleted.mesh, rim.unwrap()).unwrap();
    assert_rebuilds(&filled.mesh);
    assert_eq!(filled.mesh.euler_characteristic(), 2, "χ restored");
    assert!(Cage::try_from_mesh(filled.mesh.clone()).is_ok(), "and it promotes");
    assert_eq!(filled.new_faces.len(), 1, "a four-sided hole is one quad");
    assert_old_vertices_unmoved(&deleted.mesh, &filled.mesh);
}

#[test]
fn deleting_faces_drops_the_vertices_left_with_none() {
    let strip = grid(2, 1, &[]);
    let (a, _) = edge_at(&strip, [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]);
    let right = strip.adjacent_faces(a)[0];
    let result = delete_faces(&strip, &[right]).unwrap();
    assert_eq!(result.mesh.vertex_count(), 4, "the right-hand pair went with their face");
    let map = result.vertex_map.unwrap();
    assert_eq!(map.iter().filter(|m| m.is_none()).count(), 2);
}

#[test]
fn deleting_faces_that_would_leave_a_bowtie_is_refused() {
    let g = grid(2, 2, &[]);
    let centre = edge_at(&g, [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]).0;
    // The two faces on one diagonal of the centre go, leaving the other two touching at it.
    let faces = g.adjacent_faces(centre);
    let err = delete_faces(&g, &[faces[0], faces[2]]).unwrap_err();
    assert_eq!(err.highlight.vertices, vec![centre]);
    assert!(err.cause.is_none());
}

#[test]
fn deleting_every_face_is_refused() {
    let m = grid(1, 1, &[]);
    assert!(delete_faces(&m, &[FaceId::from_index(0)]).is_err());
}

/// A six-sided hole takes two quads by one chord; an eight-sided one takes the inset ring and three.
#[test]
fn a_six_and_an_eight_sided_hole_fill_with_quads_and_close() {
    let fine = split(&split(&cube()));
    // Two neighbouring faces of one 4×4 side of the cube: a hole of six edges.
    let f0 = FaceId::from_index(0);
    let shared = fine.face_vertices(f0);
    let neighbour = fine
        .face_ids()
        .find(|&f| f != f0 && fine.face_vertices(f).iter().filter(|x| shared.contains(x)).count() == 2)
        .unwrap();
    let opened = delete_faces(&fine, &[f0, neighbour]).unwrap();
    let rim = boundary_rim_edge(&opened.mesh);
    let filled = fill_hole(&opened.mesh, rim).unwrap();
    assert_eq!(filled.new_faces.len(), 2, "n = 6: one chord, two quads");
    assert!(Cage::try_from_mesh(filled.mesh.clone()).is_ok());
    assert_eq!(filled.mesh.euler_characteristic(), 2);

    // Three in a row: a hole of eight edges.
    let third = fine
        .face_ids()
        .find(|&f| {
            f != f0
                && f != neighbour
                && fine.face_vertices(f).iter().filter(|x| fine.face_vertices(neighbour).contains(x)).count() == 2
                && fine.face_vertices(f).iter().filter(|x| shared.contains(x)).count() == 0
        })
        .unwrap();
    let opened = delete_faces(&fine, &[f0, neighbour, third]).unwrap();
    let rim = boundary_rim_edge(&opened.mesh);
    assert_eq!(boundary_loop(&opened.mesh, rim).unwrap().len(), 8);
    let filled = fill_hole(&opened.mesh, rim).unwrap();
    assert_eq!(filled.new_faces.len(), 8 + 3, "an inset ring of eight, then three quads inside it");
    assert_eq!(filled.new_vertices.len(), 8);
    assert!(Cage::try_from_mesh(filled.mesh.clone()).is_ok());
    assert_eq!(filled.mesh.euler_characteristic(), 2);
}

fn boundary_rim_edge(m: &QuadMesh) -> (VertexId, VertexId) {
    let he = (0..m.half_edges.len())
        .map(HalfEdgeId::new)
        .find(|&h| m.is_boundary(h))
        .expect("an open mesh");
    (m.origin(he), m.destination(he))
}

/// §4.5: a hole of five edges is refused, naming the rim, and nothing is returned to commit.
#[test]
fn a_five_sided_hole_is_refused() {
    let ring = pentagon_ring();
    let err = fill_hole(&ring, (v(0), v(1))).unwrap_err();
    assert!(err.reason.contains("odd number of edges"));
    assert_eq!(err.highlight.edges.len(), 5);
}

#[test]
fn filling_from_an_interior_edge_is_refused() {
    let g = grid(2, 1, &[]);
    let (a, b) = edge_at(&g, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
    assert!(fill_hole(&g, (a, b)).is_err());
}

/// §4.2: weld the two rims of an open tube into a torus and χ is 0.
#[test]
fn welding_the_two_ends_of_a_tube_makes_a_torus() {
    let (around, along) = (6, 8);
    let tube = torus_grid(around, along, false);
    assert_eq!(tube.euler_characteristic(), 0, "a tube: two boundaries");
    let pairs: Vec<(VertexId, VertexId)> =
        (0..around).map(|i| (v(along * around + i), v(i))).collect();
    let result = weld(&tube, &pairs).unwrap();
    assert_rebuilds(&result.mesh);
    assert!(!result.mesh.is_open());
    assert_eq!(result.mesh.euler_characteristic(), 0, "a torus");
    assert_eq!(result.mesh.vertex_count(), around * along);
    let map = result.vertex_map.unwrap();
    assert_eq!(map[along * around], Some(v(0)), "a welded vertex maps to what it became");
    assert_eq!(
        result.mesh.geometry_fingerprint(),
        torus_grid(around, along, true).geometry_fingerprint(),
        "the same mesh the torus generator makes"
    );
}

#[test]
fn a_weld_into_a_bowtie_is_refused() {
    // Two quads apart; welding one corner of the second onto a corner of the first touches them at
    // one vertex only.
    let positions = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(2.0, 2.0, 0.0),
        Point3::new(3.0, 2.0, 0.0),
        Point3::new(3.0, 3.0, 0.0),
        Point3::new(2.0, 3.0, 0.0),
    ];
    let m = mesh(positions, &[vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    let err = weld(&m, &[(v(4), v(2))]).unwrap_err();
    assert_eq!(err.highlight.vertices, vec![v(2)]);
    assert!(err.cause.is_none());
}

#[test]
fn a_weld_that_collapses_a_face_is_refused() {
    let m = grid(1, 1, &[]);
    let err = weld(&m, &[(v(0), v(1))]).unwrap_err();
    assert_eq!(err.highlight.faces, vec![FaceId::from_index(0)]);
}

/// Two quads with a gap between them; bridging the facing edges closes it with one quad.
fn two_quads_with_a_gap(flip_second: bool) -> QuadMesh {
    let positions = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
        Point3::new(3.0, 0.0, 0.0),
        Point3::new(3.0, 1.0, 0.0),
        Point3::new(2.0, 1.0, 0.0),
    ];
    let second = if flip_second { vec![7, 6, 5, 4] } else { vec![4, 5, 6, 7] };
    mesh(positions, &[vec![0, 1, 2, 3], second])
}

#[test]
fn a_bridge_between_two_facing_edges_makes_one_strip() {
    let m = two_quads_with_a_gap(false);
    let result = bridge(&m, &[(v(1), v(2))], &[(v(4), v(7))], 1).unwrap();
    assert_rebuilds(&result.mesh);
    assert_eq!(result.mesh.face_count(), 3);
    assert_eq!(result.mesh.euler_characteristic(), 1, "one disc now");
    assert_old_vertices_unmoved(&m, &result.mesh);

    let deeper = bridge(&m, &[(v(1), v(2))], &[(v(4), v(7))], 3).unwrap();
    assert_eq!(deeper.mesh.face_count(), 2 + 3);
    assert_eq!(deeper.new_vertices.len(), 4, "two rings of two between the rims");
}

/// **Between two quads facing opposite ways the bridge is built, twisted** — not refused.
///
/// A rim's direction is its own, so pairing the second rim backwards keeps both sides' winding
/// for *any* two disjoint rims: the spec's "refuse only if neither direction is consistent" cannot
/// arise from the rims alone. What opposite facing does make is a quad whose two halves point
/// opposite ways, which is a geometric fault and the validator's to show amber (twisted face,
/// the cage-validation notes §3), not a topological one for the primitive to refuse.
#[test]
fn a_bridge_between_opposite_facing_quads_is_built_twisted() {
    let m = two_quads_with_a_gap(true);
    let result = bridge(&m, &[(v(1), v(2))], &[(v(4), v(7))], 1).unwrap();
    assert_rebuilds(&result.mesh);
    let q = result.mesh.face_vertices(result.new_faces[0]).map(|x| result.mesh.vertex_position(x));
    let n1 = (q[1] - q[0]).cross(&(q[2] - q[0]));
    let n2 = (q[2] - q[0]).cross(&(q[3] - q[0]));
    assert!(n1.dot(&n2) < 0.0, "its two halves face opposite ways");
}

/// The two rims of a tube bridged are a torus, whichever ring position the loops started from.
#[test]
fn a_bridge_between_two_loops_closes_a_tube() {
    let (around, along) = (5, 3);
    // An open tube that does not come back round: the last ring stops short of the first.
    let mut tube_positions = Vec::new();
    for j in 0..=along {
        for i in 0..around {
            let a = std::f64::consts::TAU * i as f64 / around as f64;
            tube_positions.push(Point3::new(a.cos(), a.sin(), j as f64));
        }
    }
    let at = |i: usize, j: usize| (j * around + i % around) as u32;
    let faces: Vec<Vec<u32>> = (0..along)
        .flat_map(|j| (0..around).map(move |i| vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]))
        .collect();
    let tube = mesh(tube_positions, &faces);
    let ring = |j: usize| -> Vec<(VertexId, VertexId)> {
        (0..around).map(|i| (v(at(i, j) as usize), v(at(i + 1, j) as usize))).collect()
    };
    let result = bridge(&tube, &ring(0), &ring(along), 2).unwrap();
    assert_rebuilds(&result.mesh);
    assert!(!result.mesh.is_open());
    assert_eq!(result.mesh.euler_characteristic(), 0);
    assert_eq!(result.new_faces.len(), around * 2);
}

#[test]
fn a_bridge_between_rims_of_different_lengths_is_refused() {
    // A single quad (a rim of four) beside a strip of two (a rim of six), not touching.
    let mut positions: Vec<Point3<f64>> = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    for (x, y) in [(3.0, 0.0), (4.0, 0.0), (5.0, 0.0), (5.0, 1.0), (4.0, 1.0), (3.0, 1.0)] {
        positions.push(Point3::new(x, y, 0.0));
    }
    let m = mesh(positions, &[vec![0, 1, 2, 3], vec![4, 5, 8, 9], vec![5, 6, 7, 8]]);
    let quad_rim: Vec<_> = (0..4).map(|i| (v(i), v((i + 1) % 4))).collect();
    let strip_rim = vec![(v(4), v(5)), (v(5), v(6)), (v(6), v(7)), (v(7), v(8)), (v(8), v(9)), (v(9), v(4))];
    let err = bridge(&m, &quad_rim, &strip_rim, 1).unwrap_err();
    assert!(err.reason.contains("4 and 6"), "{}", err.reason);
}

/// **Dissolving the ring Insert Edge made puts the cube back**, vertex for vertex and face for
/// face: the inverse the owner asked Delete on a loop to be.
#[test]
fn dissolving_an_inserted_ring_restores_the_mesh() {
    let before = cube();
    let ring = insert_ring(&before, (v(0), v(1)), 0.25).unwrap();
    let new: Vec<VertexId> = ring.new_vertices.clone();
    // The loop's edges are the ones between new vertices.
    let loop_edges: Vec<(VertexId, VertexId)> =
        ring.mesh.edges().filter(|(a, b)| new.contains(a) && new.contains(b)).collect();
    assert_eq!(loop_edges.len(), 4);
    let back = dissolve_edges(&ring.mesh, &loop_edges).unwrap();
    assert_rebuilds(&back.mesh);
    // The cube again: the same eight points at the same indices, and the same six faces as sets
    // of corners -- a merged face may start its winding at another corner, which is no change.
    assert_old_vertices_unmoved(&before, &back.mesh);
    assert_eq!(back.mesh.vertex_count(), 8);
    assert_eq!(back.mesh.face_count(), 6);
    let corners = |m: &QuadMesh| -> Vec<Vec<usize>> {
        let mut all: Vec<Vec<usize>> = m
            .face_ids()
            .map(|f| {
                let mut c: Vec<usize> = m.face_vertices(f).iter().map(|v| v.index()).collect();
                c.sort();
                c
            })
            .collect();
        all.sort();
        all
    };
    assert_eq!(corners(&back.mesh), corners(&before));
    assert!(Cage::try_from_mesh(back.mesh.clone()).is_ok());
    let map = back.vertex_map.unwrap();
    assert!(new.iter().all(|n| map[n.index()].is_none()), "the ring's vertices went with it");
}

/// A single interior edge of a cube leaves a hexagon, which is refused rather than built.
#[test]
fn dissolving_one_edge_of_a_cube_is_refused_as_a_non_quad() {
    let err = dissolve_edges(&cube(), &[(v(0), v(1))]).unwrap_err();
    assert!(err.reason.contains("not a quad"), "{}", err.reason);
    assert!(err.cause.is_none());
    let strip = grid(2, 1, &[]);
    let (a, b) = edge_at(&strip, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let err = dissolve_edges(&strip, &[(a, b)]).unwrap_err();
    assert!(err.reason.contains("open edge"), "{}", err.reason);
}

/// The corpus the spec names, each primitive applied where it applies, every result rebuilt.
#[test]
fn every_primitive_result_on_the_corpus_rebuilds() {
    let corpus = [cube(), split(&cube()), torus_grid(6, 8, true), grid(3, 3, &[(1, 1)])];
    for m in &corpus {
        let (a, b) = m.edges().next().unwrap();
        if let Ok(r) = insert_ring(m, (a, b), 0.5) {
            assert_rebuilds(&r.mesh);
            assert_eq!(r.mesh.euler_characteristic(), m.euler_characteristic());
        }
        // Deleting the annulus's corner face leaves its neighbours touching at the hole's corner,
        // a bowtie, so a refusal there is the right answer.
        match delete_faces(m, &[FaceId::from_index(0)]) {
            Ok(r) => assert_rebuilds(&r.mesh),
            Err(e) => assert!(e.cause.is_none() && !e.highlight.vertices.is_empty(), "{e:?}"),
        }
    }
}

/// **Bridge faces takes the two regions off and bridges their rims as one step**: the inner 2×2
/// of a twice-split cube's top and bottom become a tunnel through it — closed, genus one — with
/// the bridge's quads and no vertex added. A whole face of the plain cube bridged to the one
/// opposite is refused, because the bridge's side edges would be the cube's own vertical edges;
/// sides that touch, or a face on both, are refused by name.
#[test]
fn bridging_two_face_regions_makes_a_tunnel_through_the_cube() {
    let m = split(&split(&cube()));
    // The inner 2×2 of each side: the small faces whose centroid is within a quarter of the axis.
    let inner = |z: f64| -> Vec<FaceId> {
        m.face_ids()
            .filter(|&f| {
                let c = m.face_vertices(f).map(|v| m.vertex_position(v));
                let centroid = (c[0].coords + c[1].coords + c[2].coords + c[3].coords) / 4.0;
                c.iter().all(|p| (p.z - z).abs() < 1e-9) && centroid.x.abs() < 0.25 && centroid.y.abs() < 0.25
            })
            .collect()
    };
    let (top, bottom) = (inner(0.5), inner(-0.5));
    assert_eq!((top.len(), bottom.len()), (4, 4));
    let result = bridge_faces(&m, &top, &bottom, 1).unwrap();
    assert_rebuilds(&result.mesh);
    assert!(Cage::try_from_mesh(result.mesh.clone()).is_ok(), "closed");
    assert_eq!(result.mesh.euler_characteristic(), 0, "a torus: χ = 0");
    assert_eq!(result.mesh.face_count(), 96 - 8 + 8, "eight faces off, eight bridge quads on");
    assert_eq!(result.new_faces.len(), 8);
    assert_eq!(result.new_vertices.len(), 0);
    assert_eq!(result.mesh.vertex_count(), m.vertex_count() - 2, "each region's centre vertex went with it");
    let deeper = bridge_faces(&m, &top, &bottom, 3).unwrap();
    assert_eq!(deeper.new_faces.len(), 24);
    assert_eq!(deeper.new_vertices.len(), 16);

    let plain = cube();
    let (t, b) = (FaceId::from_index(1), FaceId::from_index(0));
    let along_the_walls = bridge_faces(&plain, &[t], &[b], 1).unwrap_err();
    assert!(along_the_walls.cause.is_none(), "a refusal, not a bug: {}", along_the_walls.reason);
    let touching = bridge_faces(&plain, &[t], &[FaceId::from_index(2)], 1).unwrap_err();
    assert!(touching.reason.contains("touch"), "{}", touching.reason);
    let twice = bridge_faces(&plain, &[t], &[t], 1).unwrap_err();
    assert!(twice.reason.contains("both sides"), "{}", twice.reason);
}

/// **Two opposite edges of one hole bridge into the quad that fills it** — what the Bridge tool
/// does when it is clicked edge to edge on a hole, which the owner expects to close it.
#[test]
fn bridging_two_opposite_edges_of_a_hole_fills_it() {
    let deleted = delete_faces(&cube(), &[FaceId::from_index(1)]).unwrap();
    let rim: Vec<(VertexId, VertexId)> = deleted
        .mesh
        .edges()
        .filter(|&(a, b)| {
            deleted.mesh.directed(a, b).is_some_and(|h| deleted.mesh.is_boundary(h))
                || deleted.mesh.directed(b, a).is_some_and(|h| deleted.mesh.is_boundary(h))
        })
        .collect();
    assert_eq!(rim.len(), 4);
    let first = rim[0];
    let opposite = rim.iter().copied().find(|&(a, b)| a != first.0 && a != first.1 && b != first.0 && b != first.1).unwrap();
    let bridged = bridge(&deleted.mesh, &[first], &[opposite], 1).unwrap();
    assert_rebuilds(&bridged.mesh);
    assert_eq!(bridged.mesh.face_count(), 6);
    assert!(Cage::try_from_mesh(bridged.mesh.clone()).is_ok(), "the cube is closed again");
    assert_eq!(bridged.new_vertices.len(), 0);
}
