// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A cage the document holds open is treated like an unfinished block (R12,
//! the cage-editing notes §2.2, §4.7): it and everything reading it are left out, the part's
//! other bodies compile, and a part with nothing else is refused by the cage's own name.
    use super::*;
    use fieldform_cage::{Cage, CageBody, QuadMesh};
    use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef, Place};
    use nalgebra::Point3;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// A 10 mm cube with its top face gone: an open body.
    fn open_cube() -> CageBody {
        let cube = Cage::cube_of(10.0);
        let positions = cube.vertex_ids().map(|v| cube.vertex_position(v)).collect();
        let faces: Vec<Vec<u32>> = cube
            .face_ids()
            .skip(1)
            .map(|f| cube.face_vertices(f).iter().map(|v| v.index() as u32).collect())
            .collect();
        CageBody::new(QuadMesh::from_faces(positions, &faces).unwrap())
    }

    fn sphere_at(x: f64) -> Node {
        Node::Sphere { centre: Place::At([lit(x), lit(0.0), lit(0.0)]), radius: lit(4.0) }
    }

    /// **The sphere beside a red cage is still drawn, and still reads as itself.** The whole part
    /// refuses by the cage's name, as it does for a holed body; the graph the viewport draws has
    /// the red body pruned, exactly as a holed one is, and that compiles. Before this the cage took
    /// the whole part down with `MissingCage` and nothing drew.
    #[test]
    fn a_red_cage_leaves_the_rest_of_the_part_drawn() {
        let mut doc = Document::new();
        let cage = doc.add_cage_body(open_cube());
        let mut graph = Graph::from_cage(cage);
        let ball = graph.add(sphere_at(30.0));

        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3).map(|_| ()),
            Err(CompileError::OpenCage(id, _)) if id == cage
        ));
        let (drawn, _) = crate::display_graph_with_stand_ins(&doc, &graph, &|_| false);
        assert_eq!(drawn.roots(), vec![ball], "the red cage is pruned, the sphere stays");
        let field = GraphField::compile_graph(&doc, &drawn, 1e-3).expect("the sphere compiles alone");
        assert!(field.distance(Point3::new(30.0, 0.0, 0.0)) < -3.9, "inside the sphere");
        assert!(field.distance(Point3::new(0.0, 0.0, 0.0)) > 20.0, "where the cage was there is nothing");

        let incomplete = doc.incomplete_in(&graph);
        let cage_node = graph.nodes().find(|(_, n)| matches!(n, Node::Cage(_))).unwrap().0;
        assert!(incomplete.contains(&cage_node), "the red cage is left out like a holed block");
        assert_eq!(incomplete.len(), 1, "and the sphere is not");
    }

    /// **A block reading a red cage goes with it**, closed downstream as a holed block's readers
    /// are, and the whole-part refusal names the cage rather than a hole it does not have.
    #[test]
    fn a_body_reading_a_red_cage_is_refused_by_the_cages_name() {
        let mut doc = Document::new();
        let cage = doc.add_cage_body(open_cube());
        let mut graph = Graph::from_cage(cage);
        let cage_node = graph.roots()[0];
        let ball = graph.add(sphere_at(3.0));
        let union = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![cage_node.into(), ball.into()],
        });

        match GraphField::compile_graph(&doc, &graph, 1e-3).map(|_| ()) {
            Err(CompileError::OpenCage(id, why)) => {
                assert_eq!(id, cage);
                assert_eq!(why, "open: 1 hole, 4 edges");
            }
            other => panic!("expected the open-cage refusal, got {other:?}"),
        }
        let incomplete = doc.incomplete_in(&graph);
        assert!(incomplete.contains(&cage_node) && incomplete.contains(&union));
        assert!(!incomplete.contains(&ball), "the sphere is whole; only what reads the cage goes with it");
    }

    /// Closing the hole brings the cage back without anything else changing.
    #[test]
    fn closing_the_cage_puts_its_field_back() {
        let mut doc = Document::new();
        let cage = doc.add_cage_body(open_cube());
        let graph = Graph::from_cage(cage);
        assert!(matches!(GraphField::compile_graph(&doc, &graph, 1e-3), Err(CompileError::OpenCage(..))));

        doc.cage_body_mut(cage).unwrap().set_mesh(Cage::cube_of(10.0).mesh().clone());
        assert!(!doc.cage_is_red(cage));
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("closed again");
        assert!(field.distance(Point3::origin()) < -4.0, "inside the cube's limit surface");
    }

    /// A cage the document does not hold at all is still a missing cage, not an open one.
    #[test]
    fn a_cage_not_in_the_document_is_still_reported_missing() {
        let doc = Document::new();
        let graph = Graph::from_cage(fieldform_scene::CageId::new());
        assert!(matches!(GraphField::compile_graph(&doc, &graph, 1e-3), Err(CompileError::MissingCage(_))));
    }
