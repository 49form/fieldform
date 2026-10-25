// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Growing a selection on the cage along its structure: an edge loop, an edge ring, the
//! neighbours of what is picked, and the step back from them.
//!
//! These are the selection additions the cage tools take a pre-selection from (the cage-editing
//! notes, instructions §0.4). Pure over the mesh, so they are tested without a window; the
//! commands that run them and the double-click that reaches them are the application's.

use crate::selection::Selection;
use fieldform_cage::{FaceId, QuadMesh, VertexId};

/// The half-edge `a → b`, found from either end.
fn directed(mesh: &QuadMesh, a: VertexId, b: VertexId) -> Option<fieldform_cage::HalfEdgeId> {
    let f = mesh.adjacent_faces(a).into_iter().find(|&f| {
        let c = mesh.face_vertices(f);
        (0..4).any(|i| c[i] == a && c[(i + 1) % 4] == b)
    })?;
    mesh.face_half_edges(f).into_iter().find(|&he| mesh.half_edge(he).origin == a && mesh.destination_of(he) == b)
}

/// The edge opposite `he` in its face.
fn opposite(mesh: &QuadMesh, he: fieldform_cage::HalfEdgeId) -> fieldform_cage::HalfEdgeId {
    let n = mesh.half_edge(he).next;
    mesh.half_edge(n).next
}

/// **An edge loop**: from the edge, on through each vertex of valence four to the edge across
/// from it, both ways, until it closes or reaches a vertex that is not valence four or a rim.
/// A loop stops at a star point because there is no "straight on" there, which is what makes a
/// pole the end of a loop in every quad modeller.
pub(crate) fn edge_loop(mesh: &QuadMesh, edge: (VertexId, VertexId)) -> Vec<Selection> {
    let mut out: Vec<(VertexId, VertexId)> = vec![edge];
    for (mut from, mut to) in [(edge.0, edge.1), (edge.1, edge.0)] {
        loop {
            // At `to`, valence four: the edge across is the one not sharing a face with `from → to`.
            if mesh.valence(to) != 4 || mesh.vertex_on_boundary(to) {
                break;
            }
            let around = mesh.adjacent_vertices(to);
            let Some(i) = around.iter().position(|&v| v == from) else { break };
            let across = around[(i + 2) % around.len()];
            let next = (to, across);
            if next == edge || next == (edge.1, edge.0) || out.contains(&next) || out.contains(&(across, to)) {
                break;
            }
            out.push(next);
            from = to;
            to = across;
        }
    }
    out.into_iter().map(|(a, b)| Selection::Edge(a, b)).collect()
}

/// **An edge ring**: from the edge, across each face to the edge opposite, both ways, until it
/// closes or reaches a rim. Every quad has an opposite edge, so a ring passes a pole where a loop
/// stops at one -- which is what `insert_ring` splits along.
pub(crate) fn edge_ring(mesh: &QuadMesh, edge: (VertexId, VertexId)) -> Vec<Selection> {
    let Some(start) = directed(mesh, edge.0, edge.1).or_else(|| directed(mesh, edge.1, edge.0)) else {
        return vec![Selection::Edge(edge.0, edge.1)];
    };
    let mut out: Vec<(VertexId, VertexId)> = vec![(mesh.half_edge(start).origin, mesh.destination_of(start))];
    let held = |out: &[(VertexId, VertexId)], a: VertexId, b: VertexId| out.contains(&(a, b)) || out.contains(&(b, a));
    // Both ways from the edge: through its own face, and through the face on its other side.
    let mut starts = vec![start];
    if !mesh.is_boundary(start) {
        starts.push(mesh.half_edge(start).twin);
    }
    for mut he in starts {
        loop {
            let across = opposite(mesh, he);
            let (a, b) = (mesh.half_edge(across).origin, mesh.destination_of(across));
            if held(&out, a, b) {
                break;
            }
            out.push((a, b));
            if mesh.is_boundary(across) {
                break;
            }
            he = mesh.half_edge(across).twin;
        }
    }
    out.into_iter().map(|(a, b)| Selection::Edge(a, b)).collect()
}

/// **Grow**: everything a step out from what is picked. A vertex takes its edges, an edge its
/// two faces, a face its edge-neighbours -- the step that reaches the next of the same kind for
/// faces, and the next kind up for the others, so growing a face selection stays faces.
pub(crate) fn grow(mesh: &QuadMesh, picked: &[Selection]) -> Vec<Selection> {
    let mut out: Vec<Selection> = picked.to_vec();
    let add = |s: Selection, out: &mut Vec<Selection>| {
        if !out.contains(&s) {
            out.push(s);
        }
    };
    for &part in picked {
        match part {
            Selection::Vertex(v) => {
                for n in mesh.adjacent_vertices(v) {
                    add(Selection::Edge(v, n), &mut out);
                }
            }
            Selection::Edge(a, b) => {
                if let Some((f, g)) = mesh.edge_faces(a, b) {
                    add(Selection::Face(f), &mut out);
                    add(Selection::Face(g), &mut out);
                } else if let Some(f) = mesh.adjacent_faces(a).into_iter().find(|&f| mesh.face_vertices(f).contains(&b)) {
                    add(Selection::Face(f), &mut out);
                }
            }
            Selection::Face(f) => {
                let corners = mesh.face_vertices(f);
                for i in 0..4 {
                    if let Some((p, q)) = mesh.edge_faces(corners[i], corners[(i + 1) % 4]) {
                        add(Selection::Face(if p == f { q } else { p }), &mut out);
                    }
                }
            }
            Selection::None => {}
        }
    }
    out
}

/// **Shrink**: the faces whose every edge-neighbour is also picked; edges and vertices with a
/// picked face on both sides. The step back from `grow`, so a region's border goes first.
pub(crate) fn shrink(mesh: &QuadMesh, picked: &[Selection]) -> Vec<Selection> {
    let faces: Vec<FaceId> = picked.iter().filter_map(|s| if let Selection::Face(f) = s { Some(*f) } else { None }).collect();
    let has = |f: FaceId| faces.contains(&f);
    picked
        .iter()
        .copied()
        .filter(|part| match *part {
            Selection::Face(f) => {
                let c = mesh.face_vertices(f);
                (0..4).all(|i| match mesh.edge_faces(c[i], c[(i + 1) % 4]) {
                    Some((p, q)) => has(if p == f { q } else { p }),
                    None => false,
                })
            }
            Selection::Edge(a, b) => mesh.edge_faces(a, b).is_some_and(|(p, q)| has(p) && has(q)),
            Selection::Vertex(v) => mesh.adjacent_faces(v).iter().all(|&f| has(f)),
            Selection::None => false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_cage::Cage;

    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }

    fn split_cube() -> QuadMesh {
        fieldform_subd::subdivide(&Cage::cube_of(10.0)).unwrap().mesh().clone()
    }

    fn edges(sel: &[Selection]) -> usize {
        sel.iter().filter(|s| matches!(s, Selection::Edge(..))).count()
    }

    /// On a cube every corner is valence 3, so a loop is the one edge; on the cube refined once,
    /// the edges between face-centres run round the cube in a loop of eight, through the eight
    /// edge-midpoints, which are valence four.
    #[test]
    fn a_loop_runs_through_valence_four_vertices_and_stops_at_a_pole() {
        let cube = Cage::cube_of(10.0);
        assert_eq!(edges(&edge_loop(&cube, (v(0), v(1)))), 1, "a corner is a pole");
        let m = split_cube();
        // Subdividing smooths, so vertices are found by what they are rather than where: an edge
        // midpoint is valence four with two valence-three neighbours (the old corners); its other
        // two neighbours are face centres, valence four. The edge between them is on a girth loop.
        let mid = m
            .vertex_ids()
            .find(|&x| m.valence(x) == 4 && m.adjacent_vertices(x).iter().filter(|&&n| m.valence(n) == 3).count() == 2)
            .expect("an edge midpoint");
        let centre = m.adjacent_vertices(mid).into_iter().find(|&c| m.valence(c) == 4).expect("a face centre");
        let loop_ = edge_loop(&m, (mid, centre));
        assert_eq!(edges(&loop_), 8, "round the cube: {loop_:?}");
    }

    /// A ring crosses faces to their opposite edges: on a cube the ring through one edge is the
    /// four parallel edges, closed.
    #[test]
    fn a_ring_crosses_faces_to_their_opposite_edges_and_closes() {
        let cube = Cage::cube_of(10.0);
        let ring = edge_ring(&cube, (v(0), v(1)));
        assert_eq!(edges(&ring), 4, "{ring:?}");
        // Every edge in the ring is parallel to x.
        for s in &ring {
            if let Selection::Edge(a, b) = s {
                let d = cube.vertex_position(*a) - cube.vertex_position(*b);
                assert!(d.y.abs() < 1e-9 && d.z.abs() < 1e-9, "{d:?}");
            }
        }
    }

    /// On an open strip a ring stops at the rim and includes the rim's edge.
    #[test]
    fn a_ring_on_an_open_strip_ends_at_the_rim() {
        let strip = QuadMesh::from_faces(
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(1.0, 0.0, 0.0),
                nalgebra::Point3::new(2.0, 0.0, 0.0),
                nalgebra::Point3::new(2.0, 1.0, 0.0),
                nalgebra::Point3::new(1.0, 1.0, 0.0),
                nalgebra::Point3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 4, 5], vec![1, 2, 3, 4]],
        )
        .unwrap();
        let ring = edge_ring(&strip, (v(1), v(4)));
        assert_eq!(edges(&ring), 3, "the shared edge and both ends: {ring:?}");
    }

    /// Grow from a face takes its four neighbours; shrink takes the border back off.
    #[test]
    fn grow_then_shrink_returns_to_a_face_region() {
        let m = split_cube();
        let start = vec![Selection::Face(FaceId::from_index(0))];
        let grown = grow(&m, &start);
        assert_eq!(grown.len(), 5, "{grown:?}");
        let back = shrink(&m, &grown);
        assert_eq!(back, start);
        // On a flat 3×3 grid, every face shrinks to the middle one -- the only face whose four
        // neighbours are all picked -- and the middle one grows to the plus of five.
        let mut positions = Vec::new();
        for j in 0..4 {
            for i in 0..4 {
                positions.push(nalgebra::Point3::new(i as f64, j as f64, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * 4 + i) as u32;
        let faces: Vec<Vec<u32>> = (0..3).flat_map(|j| (0..3).map(move |i| vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])).collect();
        let grid = QuadMesh::from_faces(positions, &faces).unwrap();
        let all: Vec<Selection> = grid.face_ids().map(Selection::Face).collect();
        assert_eq!(shrink(&grid, &all), vec![Selection::Face(FaceId::from_index(4))]);
        assert_eq!(grow(&grid, &[Selection::Face(FaceId::from_index(4))]).len(), 5);
        // Grow from a vertex gives its edges; from an edge, its two faces.
        assert_eq!(grow(&m, &[Selection::Vertex(v(0))]).len(), 4, "the vertex and its three edges");
        assert_eq!(grow(&m, &[Selection::Edge(v(0), m.adjacent_vertices(v(0))[0])]).len(), 3);
    }
}
