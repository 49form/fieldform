// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of critical marks and of a boolean's operand list, through a save, a load and an edit.
    use super::*;

    fn marked_graph() -> (Graph, NodeId, NodeId) {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: crate::Place::origin(),
            radius: ParamRef::Literal(10.0),
        });
        let bore = graph.add(Node::Sphere {
            centre: crate::Place::origin(),
            radius: ParamRef::Literal(4.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Difference,
            a: ball.into(),
            b: bore.into(),
            fillet: Fillet { radius: ParamRef::Literal(2.0), continuity: Continuity::G1, second: None },
        });
        assert!(graph.mark_critical(Critical { tolerance: 0.05, ..Critical::new(bore) }));
        (graph, ball, bore)
    }

    #[test]
    fn critical_marks_survive_a_save_and_a_load() {
        let (graph, _, bore) = marked_graph();
        let text = serde_json::to_string(&graph).unwrap();
        let loaded: Graph = serde_json::from_str(&text).unwrap();
        assert_eq!(loaded.critical(), graph.critical());
        assert_eq!(loaded.critical()[0].node, bore);
        assert_eq!(loaded.critical()[0].tolerance, 0.05);
    }

    /// The half of "additive" that keeps a 3.0 build opening what it can: a part nobody has
    /// marked is written exactly as it was before marks existed.
    #[test]
    fn an_unmarked_graph_writes_no_critical_list_at_all() {
        let (mut graph, _, bore) = marked_graph();
        assert!(graph.unmark_critical(bore));
        let text = serde_json::to_string(&graph).unwrap();
        assert!(!text.contains("critical"), "an empty list was written: {text}");
    }

    #[test]
    fn a_mark_on_a_node_the_file_does_not_hold_refuses_to_load() {
        let (graph, _, _) = marked_graph();
        let mut json: serde_json::Value = serde_json::to_value(&graph).unwrap();
        let ghost = NodeId::new();
        json["critical"][0]["node"] = serde_json::to_value(ghost).unwrap();
        let refused = serde_json::from_value::<Graph>(json).unwrap_err().to_string();
        assert!(
            refused.contains(&ghost.to_string()),
            "the refusal has to name the node: {refused}"
        );
    }

    #[test]
    fn removing_a_node_takes_its_mark_and_marking_twice_is_one_mark() {
        let (mut graph, _, bore) = marked_graph();
        assert!(graph.mark_critical(Critical { tolerance: 0.2, ..Critical::new(bore) }));
        assert_eq!(graph.critical().len(), 1);
        assert_eq!(graph.critical()[0].tolerance, 0.2);
        assert!(!graph.mark_critical(Critical::new(NodeId::new())), "a node that is not here");
        graph.remove(bore);
        assert!(graph.critical().is_empty());
    }

    #[test]
    fn a_hardened_graph_keeps_its_shape_and_only_the_named_fillet() {
        let (graph, _, bore) = marked_graph();
        let blend = graph.roots()[0];
        let hard = graph.hardened(None);
        assert!(hard.node(blend).unwrap().fillet().unwrap().is_plainly_hard());
        assert_eq!(hard.node(blend).unwrap().inputs(), graph.node(blend).unwrap().inputs());
        let kept = graph.hardened(Some(blend));
        assert!(!kept.node(blend).unwrap().fillet().unwrap().is_plainly_hard());
        let alone = graph.subtree(bore).unwrap();
        assert_eq!(alone.roots(), vec![bore]);
        assert_eq!(alone.len(), 1);
    }

    fn sphere() -> Node {
        Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(1.0),
        }
    }

    #[test]
    fn a_boolean_lists_every_operand_as_an_input() {
        let mut g = Graph::empty();
        let a = g.add(sphere());
        let b = g.add(sphere());
        let c = g.add(sphere());
        let joined = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into(), c.into()] });
        assert_eq!(g.node(joined).unwrap().inputs(), vec![a, b, c]);
        // Consumed, so none of the three is a body any more and the boolean is the one root.
        assert_eq!(g.roots(), vec![joined]);
    }

    #[test]
    fn a_boolean_of_three_round_trips() {
        let mut g = Graph::empty();
        let a = g.add(sphere());
        let b = g.add(sphere());
        let c = g.add(sphere());
        g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![a.into(), b.into(), c.into()] });
        let json = serde_json::to_string(&g).unwrap();
        let back: Graph = serde_json::from_str(&json).unwrap();
        for (id, node) in g.nodes() {
            assert_eq!(back.node(id), Some(node));
        }
    }

    /// The pair spelling is not read. Deliberately: a build that read `a` and `b` into a
    /// two-element list would be a migration, and nobody depends on saved files yet. A file
    /// with one is refused at open with the field named, which is D3's rule for a known kind
    /// carrying a field this build does not have.
    #[test]
    fn a_boolean_written_as_a_pair_is_refused_rather_than_misread() {
        let json = format!(
            r#"{{"kind":"boolean","op":"union","a":"{}","b":"{}"}}"#,
            NodeId::new(),
            NodeId::new()
        );
        // Preserved and refused with the fields named, which is what this format does with a
        // known kind carrying fields a build cannot account for -- not a parse error. The
        // distinction matters: the node survives a round trip and the message can say which
        // fields it could not read, instead of the document failing with a byte offset.
        let node = serde_json::from_str::<Node>(&json).expect("preserved rather than rejected");
        let Node::Unknown(unknown) = node else {
            panic!("a boolean written as a pair was read as something this build understands");
        };
        assert_eq!(
            *unknown.reason(),
            Unrecognised::Fields(vec!["a".to_string(), "b".to_string()])
        );
    }
