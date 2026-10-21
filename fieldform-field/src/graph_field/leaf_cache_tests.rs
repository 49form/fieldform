// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests that the leaf cache never answers for geometry that has moved since it was built.
    use super::*;
    use fieldform_scene::{CellMap, Graph, LatticeCell, LatticePhase, Node, ParamRef};

    /// **A cage edited inside its own box is not answered from the cache.** The cache keyed a
    /// cage by its id and subdivision depth, and the depth only moves with the cage's size — so
    /// pulling a corner of a cube *inwards*, which leaves three corners on every face of the box,
    /// kept the old subdivision and the field went on answering for the cube.
    #[test]
    fn a_cage_moved_inside_its_own_box_is_not_answered_from_the_cache() {
        let mut doc = Document::new();
        let cage = doc.add_cage(fieldform_cage::Cage::cube_of(20.0));
        let mut graph = Graph::empty();
        graph.add(Node::Cage(cage));
        let mut cache = LeafCache::default();
        let probe = Point3::new(-6.0, -6.0, -6.0);
        let before = GraphField::compile_cached(&doc, &graph, 0.05, &mut cache)
            .expect("compiles")
            .distance(probe);

        let corner = {
            let c = doc.cage(cage).expect("there");
            c.vertex_ids()
                .find(|&v| (c.vertex_position(v) - Point3::new(-10.0, -10.0, -10.0)).norm() < 1e-9)
                .expect("a cube has that corner")
        };
        doc.cage_mut(cage)
            .expect("there")
            .set_vertex_position(corner, Point3::new(-4.0, -4.0, -4.0));

        let cached = GraphField::compile_cached(&doc, &graph, 0.05, &mut cache)
            .expect("compiles")
            .distance(probe);
        let fresh = GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("compiles")
            .distance(probe);
        assert!((fresh - before).abs() > 0.1, "the edit should move the surface: {before} -> {fresh}");
        assert!((cached - fresh).abs() < 1e-9, "cached {cached} against fresh {fresh}");
    }

    /// And the same for a surface a conformal lattice follows, which is cached the same way and
    /// can now be edited the same way.
    #[test]
    fn a_moved_surface_is_not_answered_from_the_cache() {
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
        let mut doc = Document::new();
        let surface = doc.add_quad_mesh(
            fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a flat panel"),
            "panel",
        );
        let mut graph = Graph::empty();
        graph.add(Node::Lattice {
            frame: fieldform_scene::Frame::default(),
            cell: LatticeCell::Gyroid,
            map: CellMap::Conformal {
                surface,
                cells_per_face: 2,
                depth: ParamRef::Literal(10.0),
                thickness: Some(ParamRef::Literal(8.0)),
            },
            phase: LatticePhase::sheet(ParamRef::Literal(1.0)),
        });
        let mut cache = LeafCache::default();
        let probe = Point3::new(13.0, 17.0, 1.5);
        let before = GraphField::compile_cached(&doc, &graph, 0.05, &mut cache)
            .expect("compiles")
            .distance(probe);

        let inner = fieldform_cage::VertexId::from_index(at(1, 1) as usize);
        doc.quad_meshes
            .get_mut(&surface)
            .expect("there")
            .set_vertex_position(inner, Point3::new(10.0, 10.0, 6.0));

        let cached = GraphField::compile_cached(&doc, &graph, 0.05, &mut cache)
            .expect("compiles")
            .distance(probe);
        let fresh = GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("compiles")
            .distance(probe);
        assert!((fresh - before).abs() > 1e-3, "the edit should move the lattice: {before} -> {fresh}");
        assert!((cached - fresh).abs() < 1e-9, "cached {cached} against fresh {fresh}");
    }
