// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a conformal lattice's extent, with a shell and without one.
    use super::*;
    use fieldform_scene::{CellMap, Graph, LatticeCell, LatticePhase, Node, ParamRef};

    fn panel() -> fieldform_cage::QuadMesh {
        let n = 3usize;
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                positions.push(Point3::new(i as f64 * 10.0, j as f64 * 10.0, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let mut faces = Vec::new();
        for j in 0..n {
            for i in 0..n {
                faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a flat panel")
    }

    fn lattice(thickness: Option<f64>) -> (Document, Graph, NodeId) {
        let mut doc = Document::new();
        let surface = doc.add_quad_mesh(panel(), "panel");
        let mut graph = Graph::empty();
        let id = graph.add(Node::Lattice {
            frame: fieldform_scene::Frame::default(),
            cell: LatticeCell::Gyroid,
            map: CellMap::Conformal {
                surface,
                cells_per_face: 2,
                depth: ParamRef::Literal(10.0),
                thickness: thickness.map(ParamRef::Literal),
            },
            phase: LatticePhase::sheet(ParamRef::Literal(1.0)),
        });
        (doc, graph, id)
    }

    /// **A shelled conformal lattice has a box, and it is the surface's grown by half the
    /// shell.** The application decides whether a block is drawn as a solid by asking this, so
    /// a shelled lattice reporting `Unbounded` is a part that never appears — which is what the
    /// first build of shells did, reported as "i see nothing".
    #[test]
    fn a_shelled_conformal_lattice_is_bounded_by_its_surface_and_half_its_shell() {
        let (doc, graph, id) = lattice(Some(8.0));
        let (lo, hi) = graph_extent(&doc, &graph, id)
            .as_box()
            .expect("a shelled lattice is bounded");
        assert!((lo - Point3::new(-4.0, -4.0, -4.0)).norm() < 1e-9, "low corner {lo:?}");
        assert!((hi - Point3::new(34.0, 34.0, 4.0)).norm() < 1e-9, "high corner {hi:?}");
    }

    /// And one with no shell tiles outward forever, so it bounds nothing — the reason a lattice
    /// on its own is otherwise drawn as reference geometry rather than traced as a solid.
    #[test]
    fn an_unconfined_conformal_lattice_bounds_nothing() {
        let (doc, graph, id) = lattice(None);
        assert!(graph_extent(&doc, &graph, id).is_unbounded());
    }
