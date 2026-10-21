// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a boolean over a list of operands, as the generator folds it and refuses it.

    use super::*;
    use fieldform_scene::Document;

    fn sphere(graph: &mut Graph, x: f64) -> NodeId {
        graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([ParamRef::Literal(x), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
            radius: ParamRef::Literal(10.0),
        })
    }

    /// A boolean of three is a statement per operand, not a nest — the shape the part's own
    /// body fold already had to take.
    #[test]
    fn a_boolean_of_three_folds_as_statements_rather_than_a_nest() {
        let mut graph = Graph::empty();
        let a = sphere(&mut graph, -30.0);
        let b = sphere(&mut graph, 0.0);
        let c = sphere(&mut graph, 30.0);
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![a.into(), b.into(), c.into()] });
        let program = generate(&Document::new(), &graph).expect("generates");
        let source = program.field_function();
        assert!(source.contains("var acc"), "{source}");
        assert!(!source.contains("max(max("), "{source}");
        // The normal pass carries the sign a subtracted operand's gradient needs.
        assert!(program.source.contains("var bg"), "{}", program.source);
    }

    #[test]
    fn a_boolean_with_no_operands_is_refused_before_it_reaches_the_gpu() {
        let mut graph = Graph::empty();
        let empty = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: Vec::new() });
        assert_eq!(
            generate(&Document::new(), &graph).err(),
            Some(CodegenError::NothingToCombine(empty))
        );
    }
