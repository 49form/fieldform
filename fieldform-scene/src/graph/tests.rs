// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The composition graph's tests: its walks and edits, and every node's round trip through a file.

    /// A point is construction, never a body: making one leaves the bodies as they were, in
    /// the order they were in.
    #[test]
    fn a_point_is_never_a_body_and_making_one_leaves_the_bodies_as_they_were() {
        let lit = crate::ParamRef::Literal;
        let mut g = super::Graph::empty();
        let a = g.add(super::Node::Sphere { centre: crate::Place::At([lit(0.0); 3]), radius: lit(5.0) });
        let b = g.add(super::Node::Sphere { centre: crate::Place::At([lit(20.0), lit(0.0), lit(0.0)]), radius: lit(5.0) });
        let before = g.roots();
        let mut sorted = vec![a, b];
        sorted.sort_by_key(|id| id.raw());
        assert_eq!(before, sorted);
        for i in 0..4 {
            let point = g.add(super::Node::Point { at: [lit(i as f64), lit(0.0), lit(0.0)] });
            assert!(!g.roots().contains(&point), "a point is not a body");
        }
        assert_eq!(g.roots(), before, "four joints later, the bodies are the same two, in order");
    }

    /// **A block drawn beside the bodies keeps its place in the list between frames.**
    ///
    /// Reported as the selection outline jittering, and only while a block was being drawn
    /// beside the bodies. The viewport draws a consumed block by adding a second, unconsumed
    /// reference to it, and [`Graph::roots`] sorts by id — so a *minted* id put that copy in a
    /// different position on every frame, and everything keyed on the position moved with it:
    /// the index the shader outlines, and the still's own key, which restarted the accumulation
    /// forever rather than letting it settle.
    ///
    /// Built twice and compared, because once cannot fail. With a fresh id the two orders
    /// differ whenever the id lands on the other side of a real root, which is about half the
    /// time per root — so a handful of rounds is a near-certain catch and twenty is a
    /// formality.
    #[test]
    fn a_stand_in_lands_in_the_same_place_in_the_drawn_order_every_time() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: crate::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: lit(4.0),
            kind: PatternKind::Linear { step: [lit(30.0), lit(0.0), lit(0.0)] },
            join: false,
            fillet: Fillet::hard(),
        });
        graph.add(Node::Sphere {
            centre: crate::Place::At([lit(0.0), lit(90.0), lit(0.0)]),
            radius: lit(10.0),
        });

        let order = |round: usize| {
            let mut copy = graph.clone();
            let stand_in = copy.add_stand_in(ball).expect("the sphere is in the graph");
            let roots = copy.roots();
            assert!(roots.contains(&stand_in), "round {round}: the stand-in is not a root");
            roots
        };

        let first = order(0);
        for round in 1..20 {
            assert_eq!(first, order(round), "the drawn order moved between rounds");
        }
    }

    /// And asking twice is one copy, not two: a second stand-in would be a second body in the
    /// list, which is the same failure from the other side.
    #[test]
    fn a_block_has_only_one_stand_in() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: crate::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        assert!(graph.add_stand_in(ball).is_some());
        assert!(graph.add_stand_in(ball).is_none(), "a block grew a second stand-in");
        assert_eq!(graph.len(), 2, "one block and its stand-in");
    }

    /// **Downstream of a block is everything built on it, however far up and through any edge,
    /// and nothing else.** It is what the pickers refuse, because handing a block something
    /// that reads it is a cycle: a consumer missed here is a loop offered, and a block counted
    /// that is not built on it is a choice taken away for nothing.
    ///
    /// The sphere stands on a point, an offset is built on the sphere and a union on that, with
    /// a second sphere beside. A place is a graph edge as much as an operand is, so the sphere
    /// is downstream of the point; what the sphere is built from, and the sphere beside it, are
    /// not downstream of it.
    #[test]
    fn everything_built_on_a_block_is_downstream_of_it_and_nothing_else_is() {
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let point = g.add(Node::Point { at: [lit(0.0); 3] });
        let ball = g.add(Node::Sphere {
            centre: crate::Place::From { node: point },
            radius: lit(5.0),
        });
        let offset = g.add(Node::Offset { input: ball.into(), distance: lit(1.0) });
        let beside = g.add(Node::Sphere {
            centre: crate::Place::At([lit(20.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        let union = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![offset.into(), beside.into()] });

        let set = |ids: &[NodeId]| ids.iter().copied().collect::<std::collections::HashSet<_>>();
        assert_eq!(g.downstream_of(ball), set(&[offset, union]), "not its point, nor the other");
        assert_eq!(g.downstream_of(point), set(&[ball, offset, union]), "a place is an edge");
        assert!(g.downstream_of(union).is_empty(), "nothing reads the body at the top");
    }

    /// **A block's consumers are the blocks one step up, through any edge, each counted once.**
    /// Whether a block is consumed decides whether a delete may take a joint point with it, and
    /// how many consume it is what a panel quotes as "used n times": an edge missed reads a block
    /// in use as spare, and a reader counted twice reads one owner as two.
    ///
    /// A sphere stands on a point and a ramp reads the same point; a union is handed the sphere
    /// for both operands, and an offset of the union is driven by the ramp. So a place, a ramp's
    /// input and a driver are each an edge, the union is one reader however many times it names
    /// the sphere, and the offset two steps above the sphere is not one of its consumers.
    #[test]
    fn a_block_s_consumers_are_the_blocks_one_step_up_each_counted_once() {
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let point = g.add(Node::Point { at: [lit(0.0); 3] });
        let ball = g.add(Node::Sphere {
            centre: crate::Place::From { node: point },
            radius: lit(5.0),
        });
        let ramp = g.add(Node::Ramp {
            input: point.into(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: RampShape::Linear,
        });
        let union = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![ball.into(), ball.into()] });
        let offset = g.add(Node::Offset { input: union.into(), distance: ParamRef::Field(ramp) });

        let set = |ids: &[NodeId]| ids.iter().copied().collect::<std::collections::HashSet<_>>();
        let of = |id: NodeId| g.consumers_of(id).collect::<Vec<_>>();
        assert_eq!(of(ball), vec![union], "named twice, read once, and not by the offset above");
        assert_eq!(set(&of(point)), set(&[ball, ramp]), "a place and a ramp's input are edges");
        assert_eq!(of(point).len(), 2, "each reader once");
        assert_eq!(of(ramp), vec![offset], "a driver is an edge");
        assert_eq!(of(union), vec![offset]);
        assert!(of(offset).is_empty(), "nothing reads the body at the top");

        for (id, _) in g.nodes() {
            assert_eq!(g.is_consumed(id), !of(id).is_empty(), "consumed is having a consumer");
        }
    }

    /// **A block that names itself is its own consumer.** That is a cycle, and nothing compiles
    /// one, but a file or a script can hold one; the answer is the plain one, and a caller that
    /// wants the *other* readers of a block leaves it out itself. Held because the two kinds of
    /// caller differ on exactly this, and a reader that dropped the block silently would make
    /// one of them wrong without saying so.
    #[test]
    fn a_block_that_names_itself_is_its_own_consumer() {
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let ball = g.add(Node::Sphere {
            centre: crate::Place::At([lit(0.0); 3]),
            radius: lit(5.0),
        });
        let offset = g.add(Node::Offset { input: ball.into(), distance: lit(1.0) });
        if let Some(Node::Offset { input, .. }) = g.node_mut(offset) {
            *input = offset.into();
        }

        assert_eq!(g.consumers_of(offset).collect::<Vec<_>>(), vec![offset]);
        assert!(g.is_consumed(offset));
        assert!(g.consumed().contains(&offset));
        assert!(!g.is_consumed(ball), "the sphere it used to read is read by nothing now");
    }

    /// **The consumed set is every block something reads: [`Graph::is_consumed`] asked of every
    /// block at once.** It is what the bodies and the evaluation order are both seeded from, so a
    /// block it leaves out is a head that should not be, and one it keeps is a body gone missing.
    ///
    /// A ramp nothing reads is outside it and is still not a body, which is why the heads and the
    /// bodies are two different lists. And an edge to a block taken away without its readers being
    /// rewired still counts, because the set is of what is read rather than of what is here.
    #[test]
    fn the_consumed_set_is_every_block_something_reads_and_nothing_else() {
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let point = g.add(Node::Point { at: [lit(0.0); 3] });
        let ball = g.add(Node::Sphere {
            centre: crate::Place::From { node: point },
            radius: lit(5.0),
        });
        let offset = g.add(Node::Offset { input: ball.into(), distance: lit(1.0) });
        let idle = g.add(Node::Ramp {
            input: point.into(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: RampShape::Linear,
        });

        let consumed = g.consumed();
        let read: std::collections::HashSet<NodeId> = [point, ball].into_iter().collect();
        assert_eq!(consumed, read, "the point and the sphere, and neither head");
        for (id, _) in g.nodes() {
            assert_eq!(consumed.contains(&id), g.is_consumed(id), "one pass agrees with each ask");
        }
        assert!(!consumed.contains(&idle) && !g.roots().contains(&idle), "a head, not a body");
        assert_eq!(g.roots(), vec![offset]);

        g.remove(ball);
        assert!(g.consumed().contains(&ball), "the offset still names it");
    }

    /// **A block's subtree is itself and everything it is built from, through any edge, and
    /// nothing else.** It is what a body weighs, which rows travel with it when it is dragged, and
    /// what stays when a block is measured on its own: an edge missed leaves out a block the body
    /// cannot be drawn without, and a block counted that it does not read moves a row that should
    /// have stayed where it was.
    ///
    /// A sphere stands on a point, and a ramp reading the same point drives an offset of the
    /// sphere; a union joins the offset, the sphere itself and a second sphere beside. So a place,
    /// an operand and a driver are each an edge, and what is above a block or beside it is not in
    /// its subtree. Read backwards it is [`Graph::downstream_of`], and its nodes are exactly the
    /// ones [`Graph::subtree`] copies.
    #[test]
    fn a_block_s_subtree_is_itself_and_everything_it_is_built_from_and_nothing_else() {
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let point = g.add(Node::Point { at: [lit(0.0); 3] });
        let ball = g.add(Node::Sphere {
            centre: crate::Place::From { node: point },
            radius: lit(5.0),
        });
        let ramp = g.add(Node::Ramp {
            input: point.into(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: RampShape::Linear,
        });
        let offset = g.add(Node::Offset { input: ball.into(), distance: ParamRef::Field(ramp) });
        let beside = g.add(Node::Sphere {
            centre: crate::Place::At([lit(20.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        let union =
            g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![offset.into(), ball.into(), beside.into()] });

        let set = |ids: &[NodeId]| ids.iter().copied().collect::<std::collections::HashSet<_>>();
        assert_eq!(g.subtree_of(point), set(&[point]), "a leaf is its own subtree");
        assert_eq!(g.subtree_of(ball), set(&[ball, point]), "a place is an edge");
        assert_eq!(g.subtree_of(ramp), set(&[ramp, point]), "so is a ramp's input");
        assert_eq!(g.subtree_of(offset), set(&[offset, ball, point, ramp]), "and a driver");
        assert_eq!(g.subtree_of(union), set(&[union, offset, ball, point, ramp, beside]));

        for (a, _) in g.nodes() {
            for (b, _) in g.nodes() {
                if a != b {
                    assert_eq!(
                        g.subtree_of(a).contains(&b),
                        g.downstream_of(b).contains(&a),
                        "built from is downstream read backwards"
                    );
                }
            }
            let copied: std::collections::HashSet<NodeId> =
                g.subtree(a).expect("a block that is here").nodes().map(|(id, _)| id).collect();
            assert_eq!(copied, g.subtree_of(a), "the graph form holds exactly these");
        }
    }

    /// **A subtree names a block taken away without its readers being rewired, and a cycle ends
    /// the walk.** The set is of what is named, as the consumed set is, so a caller that wants the
    /// nodes asks for each one -- and [`Graph::subtree`], which does, leaves the missing block
    /// out. A cycle is what nothing compiles and a file or a script can still hold, and asking
    /// about one has to answer rather than walk forever.
    #[test]
    fn a_subtree_names_a_block_taken_away_and_ends_where_a_cycle_closes() {
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let point = g.add(Node::Point { at: [lit(0.0); 3] });
        let ball = g.add(Node::Sphere {
            centre: crate::Place::From { node: point },
            radius: lit(5.0),
        });
        let offset = g.add(Node::Offset { input: ball.into(), distance: lit(1.0) });

        g.remove(ball);
        let set = |ids: &[NodeId]| ids.iter().copied().collect::<std::collections::HashSet<_>>();
        assert_eq!(g.subtree_of(offset), set(&[offset, ball]), "named, and nothing past it");
        assert_eq!(g.subtree(offset).expect("the offset is here").len(), 1, "only what is here");
        assert_eq!(g.subtree_of(ball), set(&[ball]), "the block asked about is always in it");
        assert!(g.subtree(ball).is_none(), "but there is no graph of a block that is not here");

        if let Some(Node::Offset { input, .. }) = g.node_mut(offset) {
            *input = offset.into();
        }
        assert_eq!(g.subtree_of(offset), set(&[offset]));
    }

    /// A block's own turn becomes that block inside a placement, and keeps its identity.
    ///
    /// **The lowering is the whole definition of what a turn means**, and both the field and the
    /// shader read it, so the property worth holding is that it produces the composition they
    /// each already know how to evaluate — a placement turning the block about its own place.
    ///
    /// The turned block *keeps its id*: whatever consumed it still consumes it, the roots are
    /// the same roots, and anything keyed by node still points at what it named. Getting that
    /// wrong would not fail to compile — it would quietly drop the block out of the part.
    #[test]
    fn a_blocks_own_turn_lowers_to_a_placement_that_keeps_its_id() {
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let box_id = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([lit(1.0), lit(0.0), lit(0.0)]),
            size: [lit(10.0); 3],
            turn: Turn::By([lit(0.0), lit(0.0), lit(45.0)]),
        });
        let straight = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::origin(),
            size: [lit(10.0); 3],
            turn: no_turn(),
        });
        let cut = g.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![box_id.into(), straight.into()],
        });

        let out = g.with_turns_expanded();
        match out.node(box_id) {
            Some(Node::Transform { input, rotation, about, .. }) => {
                assert_eq!(rotation[2], lit(45.0), "the turn moved to the placement");
                assert!(
                    matches!(input.source().and_then(|id| out.node(id)), Some(Node::Cuboid { turn, .. }) if turn_is_none(turn)),
                    "the shape inside kept its size and lost its turn"
                );
                assert_eq!(
                    about.typed().map(|xyz| xyz[0]),
                    Some(lit(1.0)),
                    "it turns about its own place, not the part's origin"
                );
            }
            other => panic!("a turned block should lower to a placement, got {other:?}"),
        }
        // The consumer still names the same node, so nothing dropped out of the part.
        assert_eq!(out.node(cut).map(|n| n.inputs()), Some(vec![box_id, straight]));
        // And a block that was never turned is left exactly as it was.
        assert!(matches!(out.node(straight), Some(Node::Cuboid { .. })));
    }


    /// A document written before frames existed reads as the frame it meant.
    ///
    /// **The part of D9's migration that could go wrong quietly.** A twist that once said
    /// `axis: "x"` and an origin, read as the default frame, would come back winding about Z
    /// through nothing — a valid twist of the wrong body, with nothing to report. So the old
    /// spellings are read here from the JSON a previous build actually wrote.
    #[test]
    fn a_document_written_before_frames_reads_as_the_frame_it_meant() {
        // A twist, in the shape the writer used before this: an axis name and an origin.
        let old_twist = serde_json::json!({
            "kind": "twist",
            "input": "0000000000000001",
            "degrees_per_mm": { "literal": 3.0 },
            "axis": "x",
            "origin": [{ "literal": 5.0 }, { "literal": 0.0 }, { "literal": 0.0 }],
        });
        let node: Node = serde_json::from_value(old_twist).expect("reads");
        let Node::Twist { frame, .. } = node else { panic!("not a twist: {node:?}") };
        assert_eq!(
            frame,
            Frame::at_along(Axis::X, [ParamRef::Literal(5.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
            "a twist about X through (5,0,0) should read as that frame",
        );

        // A mirror across a plane node, which only a mirror could once say.
        let old_mirror = serde_json::json!({
            "kind": "mirror",
            "input": "0000000000000001",
            "plane": "node",
            "plane_node": "0000000000000002",
            "join": true,
        });
        let node: Node = serde_json::from_value(old_mirror).expect("reads");
        let Node::Mirror { frame, .. } = node else { panic!("not a mirror: {node:?}") };
        assert!(
            matches!(frame, Frame::From { .. }),
            "a mirror across a plane node should read as a frame taken from it, got {frame:?}",
        );
        // And it is an edge, so that plane stops being a body — which is what it meant before.
        assert!(frame.source().is_some());

        // A radial pattern, and one written with neither, which meant the world frame.
        let old_pattern = serde_json::json!({
            "kind": "pattern",
            "input": "0000000000000001",
            "count": 6,
            "kind_": "radial",
            "kind": "pattern",
        });
        let _ = old_pattern;
        let bare_twist = serde_json::json!({
            "kind": "twist",
            "input": "0000000000000001",
            "degrees_per_mm": { "literal": 1.0 },
        });
        let node: Node = serde_json::from_value(bare_twist).expect("reads");
        let Node::Twist { frame, .. } = node else { panic!("not a twist") };
        assert!(frame.is_default(), "a twist with neither field meant the world frame");
    }

    /// And a frame written by this build reads back as itself, both ways round.
    #[test]
    fn a_frame_survives_being_written_and_read_again() {
        for frame in [
            Frame::default(),
            Frame::at_along(Axis::Y, [ParamRef::Literal(1.0), ParamRef::Literal(2.0), ParamRef::Literal(3.0)]),
            Frame::From { node: crate::NodeId::new() },
        ] {
            let node = Node::Twist {
                input: crate::NodeId::new().into(),
                degrees_per_mm: ParamRef::Literal(2.0),
                frame,
            };
            let json = serde_json::to_value(&node).expect("writes");
            let back: Node = serde_json::from_value(json).expect("reads");
            let Node::Twist { frame: read_back, .. } = back else { panic!("not a twist") };
            assert_eq!(read_back, frame);
        }
    }
    use super::*;
    use crate::id::{CurveId, ParamId};
    use crate::{
        Frame, LoftSmooth, Operand, Outline, Place, Section, Sweep, WallEnds, WallSide,
    };

    /// A graph through JSON and back, which is what saving and reopening a part is.
    fn round_trip(g: &Graph) -> Graph {
        serde_json::from_str(&serde_json::to_string(g).unwrap()).unwrap()
    }

    /// Every node this build writes is a node it can read back.
    ///
    /// **The guard above refuses a field it has not heard of, and its list is kept by hand**, so
    /// a field added to the writer and to the reader and not to the list makes a document this
    /// build wrote unopenable by this build -- with a message naming the field, which is the
    /// only mercy in it. That is not hypothetical: `symmetric` was written and read for a while
    /// and never admitted, and every saved symmetric extrude was refused on open.
    ///
    /// So: the nodes whose fields are *conditional* -- written only when they are not the
    /// default, which is exactly when a round-trip test built from defaults will not see them.
    /// A node that comes back `Unknown` has fallen in the same hole.
    #[test]
    fn every_node_this_build_writes_is_a_node_it_can_read_back() {
        let lit = ParamRef::Literal;
        let curve = CurveId::new();
        let conditional = vec![
            // Written only when true; defaulted to false on read.
            Node::Extrude { rims: Default::default(),
                outline: Outline::closed(curve),
                sweep: Sweep::Normal,
                height: lit(40.0),
                start: lit(0.0),
                sides: crate::Sides::Symmetric,
                draft: lit(3.0),
                corners: Default::default(),
            },
            Node::Extrude { rims: Default::default(),
                outline: Outline::closed(curve),
                sweep: Sweep::Normal,
                height: lit(40.0),
                start: lit(0.0),
                sides: crate::Sides::One,
                draft: lit(0.0),
                corners: Default::default(),
            },
            // A wall with everything at its default, which writes neither a side nor its ends,
            // and one with neither at its default, which writes both.
            Node::Extrude { rims: Default::default(),
                outline: Outline::Wall {
                    curve,
                    thickness: lit(2.0),
                    side: WallSide::Centre,
                    ends: WallEnds::Square,
                },
                sweep: Sweep::Normal,
                height: lit(40.0),
                start: lit(0.0),
                sides: crate::Sides::One,
                draft: lit(0.0),
                corners: Default::default(),
            },
            Node::Extrude { rims: Default::default(),
                outline: Outline::Wall {
                    curve,
                    thickness: lit(2.0),
                    side: WallSide::Right,
                    ends: WallEnds::Round,
                },
                sweep: Sweep::Normal,
                height: lit(40.0),
                start: lit(0.0),
                sides: crate::Sides::Symmetric,
                draft: lit(0.0),
                corners: Default::default(),
            },
            Node::Offset { input: NodeId::new().into(), distance: lit(-2.0) },
            Node::Pipe { curve, radius: lit(5.0) },
            Node::Revolve { rims: Default::default(), outline: Outline::closed(curve), frame: Frame::default(), angle: crate::ParamRef::Literal(360.0), sides: crate::AngleSides::One },
            Node::Sweep { rims: Default::default(),
                outline: Outline::closed(curve),
                spine: curve,
                section: crate::Section::Skewed,
                law: Default::default(),
            },
            // And each of them on a wall with neither its side nor its ends at the default.
            Node::Revolve { rims: Default::default(),
                outline: Outline::Wall {
                    curve,
                    thickness: lit(2.0),
                    side: WallSide::Left,
                    ends: WallEnds::Round,
                },
                frame: Frame::default(),
                angle: crate::ParamRef::Literal(360.0),
                sides: crate::AngleSides::One,
            },
            Node::Sweep { rims: Default::default(),
                outline: Outline::Wall {
                    curve,
                    thickness: lit(2.0),
                    side: WallSide::Right,
                    ends: WallEnds::Round,
                },
                spine: curve,
                section: crate::Section::Skewed,
                law: Default::default(),
            },
        ];
        for node in conditional {
            let mut g = Graph::empty();
            // A node holding an input needs one that exists, or the reader is right to refuse.
            let node = match node {
                Node::Offset { distance, .. } => {
                    let under = g.add(Node::Sphere {
                        centre: crate::Place::At([lit(0.0); 3]),
                        radius: lit(10.0),
                    });
                    Node::Offset { input: under.into(), distance }
                }
                other => other,
            };
            let id = g.add(node.clone());
            let back = round_trip(&g);
            assert!(
                !matches!(back.node(id), Some(Node::Unknown(_))),
                "{node:?} wrote a field the reader does not admit: {:?}",
                back.node(id),
            );
            assert_eq!(back.node(id), Some(&node), "{node:?} did not survive a save");
        }
    }

    #[test]
    fn a_graph_round_trips_through_json() {
        let curve = CurveId::new();
        let mut g = Graph::from_cage(CageId::new());
        let cage = g.roots()[0];
        let pipe = g.add(Node::Pipe { curve, radius: ParamRef::Named(ParamId::new()) });
        let blend = g.add(Node::Blend {
            op: BooleanOp::Union,
            a: cage.into(),
            b: pipe.into(),
            fillet: Fillet { radius: ParamRef::Literal(3.0), continuity: Continuity::G1, second: None },
        });

        let back = round_trip(&g);
        assert_eq!(back.roots(), vec![blend], "joining two bodies leaves one");
        assert_eq!(back.len(), 3);
        for (id, node) in g.nodes() {
            assert_eq!(back.node(id), Some(node), "node {id} survived");
        }
    }

    /// **A lattice's cell sizes are among its parameters, on every map, after the ones it already
    /// listed.** The map was bound with `..`, so a variable used only as a cell size was one nothing
    /// walking a node's parameters could see: called unused, and removable while still read. The
    /// new numbers go last so the phase keeps index 0, which `wire_driver` reaches a ramp by, and
    /// both walks list them in the same order.
    #[test]
    fn a_lattices_cell_size_is_one_of_its_parameters_on_every_map() {
        let size = ParamRef::Named(ParamId::new());
        let maps = [
            CellMap::Rectangular { cell: size },
            CellMap::Cylindrical { cell_radius: size, cell_height: ParamRef::Literal(5.0), arc_count: 12 },
            CellMap::Spherical { cell_radius: size, azimuth_count: 12, polar_count: 6 },
            CellMap::Conformal {
                surface: crate::QuadMeshId::new(),
                cells_per_face: 2,
                depth: size,
                thickness: Some(ParamRef::Literal(1.0)),
            },
        ];
        for map in maps {
            let mut node = Node::Lattice {
                frame: Frame::default(),
                cell: LatticeCell::Gyroid,
                map,
                phase: LatticePhase::sheet(ParamRef::Literal(2.0)),
            };
            let read: Vec<ParamRef> = node.param_refs().into_iter().copied().collect();
            let written: Vec<ParamRef> = node.param_refs_mut().into_iter().map(|r| *r).collect();
            assert_eq!(read, written, "the two walks disagree on {}", map.tag());
            assert!(read.contains(&size), "a {} map's size is not a parameter", map.tag());
            assert_eq!(read[0], ParamRef::Literal(2.0), "the phase keeps index 0 on {}", map.tag());
            let first_map = read.len() - map.params().count();
            assert_eq!(read[first_map], size, "the map's numbers come last on {}", map.tag());
        }
    }

    /// A walled sheet keeps its offset through a save, and a sheet nobody has offset writes the
    /// object it always wrote -- no key an older reader would have to refuse.
    #[test]
    fn a_walled_sheet_round_trips_and_a_plain_one_writes_what_it_always_wrote() {
        let lattice = |phase: LatticePhase| Node::Lattice {
            frame: Frame::default(),
            cell: LatticeCell::Gyroid,
            map: CellMap::Rectangular { cell: ParamRef::Literal(20.0) },
            phase,
        };
        let mut g = Graph::empty();
        let plain = g.add(lattice(LatticePhase::sheet(ParamRef::Literal(2.0))));
        let walled = g.add(lattice(LatticePhase::Sheet {
            thickness: ParamRef::Literal(2.0),
            offset: ParamRef::Literal(1.5),
        }));

        let text = serde_json::to_string(&g).unwrap();
        assert_eq!(text.matches("\"offset\"").count(), 1, "only the walled one says so: {text}");
        let back = round_trip(&g);
        assert_eq!(back.node(plain), g.node(plain));
        assert_eq!(back.node(walled), g.node(walled));
        // A parameter like the thickness beside it, so it can be named and resolved.
        let refs = g.node(walled).unwrap().param_refs();
        assert_eq!(refs[0], &ParamRef::Literal(2.0), "the phase's own number stays first");
        assert_eq!(refs[1], &ParamRef::Literal(1.5));
    }

    /// Two curves by id and nothing else, which is what makes the frame a computed thing: a
    /// saved sweep carries no orientation for a reload to disagree with.
    #[test]
    fn a_sweep_round_trips_with_both_its_curves() {
        let (profile, spine) = (CurveId::new(), CurveId::new());
        let mut g = Graph::empty();
        let swept = || Node::Sweep { rims: Default::default(),
            outline: Outline::closed(profile),
            spine,
            section: Default::default(),
            law: Default::default(),
        };
        let sweep = g.add(swept());

        let back = round_trip(&g);
        assert_eq!(back.roots(), vec![sweep], "a sweep is a body");
        assert_eq!(back.node(sweep), Some(&swept()));
        assert_eq!(
            back.node(sweep).map(|n| n.curves()),
            Some(vec![profile, spine]),
            "both curves are wiring, so renaming or deleting either reaches the sweep"
        );
    }

    /// **A loft round-trips with every section, its centerline and its smoothness**, and is not
    /// met as something unknown -- the failure the known-fields list exists to prevent, and the
    /// one a text round trip cannot see, since an unknown node is preserved byte for byte.
    ///
    /// Both halves of the optional fields are held: a derived centerline and G0 write nothing,
    /// a named one and G1 come back named. And every curve is wiring, so deleting a section's
    /// curve or the centerline is refused while the loft stands on it.
    #[test]
    fn a_loft_round_trips_with_every_section_and_says_what_it_stands_on() {
        let (a, b, c, spine) = (CurveId::new(), CurveId::new(), CurveId::new(), CurveId::new());
        let sections = vec![Outline::closed(a), Outline::closed(b), Outline::closed(c)];

        // Derived and straight: the two optional fields are written as nothing.
        let mut g = Graph::empty();
        let plain = Node::Loft { rims: Default::default(),
            sections: sections.clone(),
            spine: None,
            smooth: LoftSmooth::G0,
        };
        let id = g.add(plain.clone());
        let text = serde_json::to_string(&g).expect("writes");
        assert!(!text.contains("\"spine\""), "a derived centerline names nothing: {text}");
        assert!(!text.contains("\"smooth\""), "and straight is what nothing means: {text}");
        let back = round_trip(&g);
        assert_eq!(back.node(id), Some(&plain), "a plain loft comes back as itself");
        assert!(
            !matches!(back.node(id), Some(Node::Unknown(_))),
            "and is read as a loft, not preserved as something unknown"
        );
        assert_eq!(back.roots(), vec![id], "a loft is a body");

        // Named and smooth: both come back.
        let mut g = Graph::empty();
        let named = Node::Loft { rims: Default::default(), sections, spine: Some(spine), smooth: LoftSmooth::G1 };
        let id = g.add(named.clone());
        let back = round_trip(&g);
        assert_eq!(back.node(id), Some(&named));
        assert_eq!(
            back.node(id).map(|n| n.curves()),
            Some(vec![a, b, c, spine]),
            "every section and the centerline are wiring"
        );
        assert_eq!(back.node(id).and_then(|n| n.aimed_by()), Some(spine));
    }

    /// A symmetric extrude round-trips, and is not met as something unknown.
    ///
    /// The writer writes `symmetric` only when it is true, the reader read it either way, and
    /// the list of fields the reader *admits* for an extrude had never been told -- so every
    /// document holding a symmetric extrude was refused on open with the field named, and the
    /// node kept unevaluated. Both halves of the extrude are held here: the plain one writes
    /// what it always wrote, the symmetric one comes back symmetric.
    #[test]
    fn a_symmetric_extrude_round_trips_and_is_not_met_as_unknown() {
        let profile = CurveId::new();
        let extrude = |symmetric: bool| Node::Extrude { rims: Default::default(),
            outline: Outline::closed(profile),
            sweep: crate::Sweep::Normal,
            height: ParamRef::Literal(40.0),
            draft: ParamRef::Literal(0.0),
            start: ParamRef::Literal(0.0),
            sides: crate::Sides::symmetric_if(symmetric),
            corners: Default::default(),
        };
        for symmetric in [false, true] {
            let mut g = Graph::empty();
            let id = g.add(extrude(symmetric));
            let back = round_trip(&g);
            assert_eq!(back.node(id), Some(&extrude(symmetric)), "symmetric = {symmetric}");
        }
    }

    /// Where an extrude's material begins and which way it lies are written only when they are
    /// not what an older document means, so an extrude that uses neither writes the bytes it
    /// always wrote; a start and a second side round-trip; and a file saying both `symmetric`
    /// and `height_2` is two answers to one question, kept and refused rather than guessed at
    /// (`extrude-revolve-options-spec.md` §2.1).
    #[test]
    fn an_extrudes_start_and_second_side_are_written_only_when_they_are_something() {
        let profile = CurveId::new();
        let plain = |sides: crate::Sides, start: f64| Node::Extrude { rims: Default::default(),
            outline: Outline::closed(profile),
            sweep: crate::Sweep::Normal,
            height: ParamRef::Literal(40.0),
            draft: ParamRef::Literal(0.0),
            start: ParamRef::Literal(start),
            sides,
            corners: Default::default(),
        };
        let text = |node: &Node| serde_json::to_value(node).expect("serialises");

        // What an older document holds, and nothing more.
        let was = text(&plain(crate::Sides::One, 0.0));
        let keys: Vec<&str> = was.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["draft", "height", "kind", "profile", "sweep"], "and in a map's order");
        assert_eq!(
            text(&plain(crate::Sides::Symmetric, 0.0))["symmetric"],
            serde_json::json!(true),
            "half each way is still the one word it was"
        );

        let two = plain(
            crate::Sides::Two { height: ParamRef::Literal(5.0), draft: ParamRef::Literal(3.0) },
            10.0,
        );
        let written = text(&two);
        assert!(written.get("symmetric").is_none(), "two sides are not the symmetric flag");
        assert_eq!(written["start"], text(&plain(crate::Sides::One, 10.0))["start"]);
        assert!(written.get("height_2").is_some() && written.get("draft_2").is_some());
        let mut g = Graph::empty();
        let id = g.add(two.clone());
        assert_eq!(round_trip(&g).node(id), Some(&two));

        // A second side with no draft written is upright, as side one's is.
        let mut without = written.clone();
        without.as_object_mut().unwrap().remove("draft_2");
        let read: Node = serde_json::from_value(without).expect("reads");
        assert!(matches!(
            read,
            Node::Extrude { sides: crate::Sides::Two { draft: ParamRef::Literal(d), .. }, .. }
                if d == 0.0
        ));

        // Both spellings at once is refused, not resolved.
        let mut both = written;
        both.as_object_mut().unwrap().insert("symmetric".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<Node>(both).is_err());
    }

    #[test]
    fn a_primitive_round_trips_with_its_centre_and_size() {
        // Raw geometry: no cage, no curve, nothing to reference — which also makes these the
        // only nodes whose whole definition is numbers.
        let param = ParamId::new();
        let mut g = Graph::empty();
        let sphere = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(1.0), ParamRef::Literal(2.0), ParamRef::Named(param)]),
            radius: ParamRef::Literal(7.5),
        });
        let cube = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [ParamRef::Named(param); 3],
            turn: no_turn(),
        });

        let back = round_trip(&g);
        assert_eq!(back.node(sphere), g.node(sphere));
        assert_eq!(back.node(cube), g.node(cube));
        assert_eq!(back.roots().len(), 2, "each is a body of its own");

        let value = serde_json::to_value(g.node(sphere).unwrap()).unwrap();
        assert_eq!(value["kind"], "sphere");
        assert_eq!(value["centre"][2], serde_json::json!({ "named": param.to_string() }));
    }

    /// A `"cube"` written by a build that predates the general box still opens, as a box with
    /// three equal sides.
    ///
    /// The kind was retired rather than kept, so this is the one thing standing between an
    /// older document and a node the reader would file under `Unknown` — and a document whose
    /// every solid became unreadable is exactly the failure D3 exists to prevent. The literal
    /// JSON is written out rather than produced by this build, because a test that generates
    /// its own input can only ever agree with itself about what the old format was.
    #[test]
    fn a_cube_written_by_an_older_build_reads_back_as_a_box_with_equal_sides() {
        let old = serde_json::json!({
            "kind": "cube",
            "centre": [{ "literal": 1.0 }, { "literal": 2.0 }, { "literal": 3.0 }],
            "size": { "literal": 10.0 },
        });
        let node: Node = serde_json::from_value(old).expect("an older cube still reads");
        assert_eq!(
            node,
            Node::Cuboid { edge: Default::default(),
                centre: crate::Place::At([
                    ParamRef::Literal(1.0),
                    ParamRef::Literal(2.0),
                    ParamRef::Literal(3.0),
                ]),
                size: [ParamRef::Literal(10.0); 3],
                turn: no_turn(),
            },
        );

        // And it is written back under the new kind, not the old one: the format carries one
        // way to say "box", so a document opened and saved stops being ambiguous.
        let value = serde_json::to_value(&node).unwrap();
        assert_eq!(value["kind"], "cuboid");
        assert_eq!(value["size"][1], serde_json::json!({ "literal": 10.0 }));
    }

    /// A box's three sides survive the trip separately.
    ///
    /// The failure this catches is one side written and read back into all three, which a cube
    /// cannot distinguish from correct behaviour — so the test that can is the one whose sides
    /// all differ.
    #[test]
    fn a_boxs_three_sides_survive_json_separately() {
        let mut g = Graph::empty();
        let boxed = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [
                ParamRef::Literal(40.0),
                ParamRef::Literal(10.0),
                ParamRef::Literal(3.0),
            ],
            turn: no_turn(),
        });
        assert_eq!(round_trip(&g).node(boxed), g.node(boxed));
    }

    /// A rectangular lattice writes exactly the object it wrote before cylindrical maps
    /// existed, so a document saved by this build still opens in one that predates them.
    ///
    /// The cell size moved *in the Rust type* from the node to the map, which is the split this
    /// whole change is for. It deliberately did not move in the file: `map` stays a string tag
    /// beside a flat `cell_size`, because that is the shape this format's forward-compatibility
    /// machinery understands. Nesting the parameters would change the field's JSON *type*, and
    /// a type change is the one thing a reader cannot refuse gracefully.
    #[test]
    fn a_rectangular_lattice_serializes_exactly_as_it_did_before_cell_maps_had_parameters() {
        let node = Node::Lattice {
            frame: Frame::default(),
            cell: LatticeCell::Gyroid,
            map: CellMap::Rectangular { cell: ParamRef::Literal(20.0) },
            phase: LatticePhase::sheet(ParamRef::Literal(2.0)),
        };
        let value = serde_json::to_value(&node).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "kind": "lattice",
                "cell": "gyroid",
                "map": "rectangular",
                "cell_size": { "literal": 20.0 },
                "thickness": { "literal": 2.0 },
            }),
        );
    }

    /// A solid lattice writes the tag and the density, and reads back as what it was.
    ///
    /// The tag is written only by the phase that is new, so the test above still holds: a sheet
    /// writes what it always wrote and a document nobody has switched is unchanged to the byte.
    /// This is the other half of that — the new spelling has to survive the round trip, or the
    /// phase is a thing you can set and not a thing you can save.
    #[test]
    fn a_solid_lattice_round_trips_with_its_density() {
        let node = Node::Lattice {
            frame: Frame::default(),
            cell: LatticeCell::SchwarzD,
            map: CellMap::Rectangular { cell: ParamRef::Literal(20.0) },
            phase: LatticePhase::Solid { density: ParamRef::Literal(0.4) },
        };
        let value = serde_json::to_value(&node).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "kind": "lattice",
                "cell": "schwarz_d",
                "map": "rectangular",
                "cell_size": { "literal": 20.0 },
                "phase": "solid",
                "density": { "literal": 0.4 },
            }),
        );
        assert_eq!(serde_json::from_value::<Node>(value).unwrap(), node);
    }

    /// A beam lattice writes its radius and reads back as what it was.
    ///
    /// The third phase, and the one whose number is a plain length rather than a fraction or a
    /// thickness read through a table — so the round trip is the only thing standing between a
    /// typed radius and the shader.
    #[test]
    fn a_beam_lattice_round_trips_with_its_radius() {
        let node = Node::Lattice {
            frame: Frame::default(),
            cell: LatticeCell::Bcc,
            map: CellMap::Rectangular { cell: ParamRef::Literal(20.0) },
            phase: LatticePhase::Strut { radius: ParamRef::Literal(1.2) },
        };
        let value = serde_json::to_value(&node).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "kind": "lattice",
                "cell": "bcc",
                "map": "rectangular",
                "cell_size": { "literal": 20.0 },
                "phase": "strut",
                "radius": { "literal": 1.2 },
            }),
        );
        assert_eq!(serde_json::from_value::<Node>(value).unwrap(), node);
    }

    /// A phase this build has never heard of is preserved and refused, not read as a sheet.
    ///
    /// **The specific trap this guards.** Reading the phase falls back to a sheet when the field
    /// is absent, which is what makes every document written before the phase existed open
    /// correctly. Without the vocabulary declaration, that same fallback would swallow a phase
    /// from a *newer* build — a beam lattice's, say — and quietly open it as a sheet with
    /// whatever number happened to be lying beside it. That is the failure D3 exists to prevent,
    /// and here it would produce a readable document showing the wrong solid.
    #[test]
    fn a_lattice_phased_a_way_this_build_does_not_know_is_kept_rather_than_read_as_a_sheet() {
        let newer = serde_json::json!({
            "kind": "lattice",
            "cell": "gyroid",
            "map": "rectangular",
            "cell_size": { "literal": 20.0 },
            "thickness": { "literal": 2.0 },
            "phase": "midsurface",
        });
        let node: Node = serde_json::from_value(newer.clone()).unwrap();
        let Node::Unknown(unknown) = &node else {
            panic!("read as {node:?} rather than refused");
        };
        assert_eq!(
            unknown.reason(),
            &Unrecognised::Value {
                field: "phase".to_string(),
                value: "midsurface".to_string(),
            },
        );
        // And it is kept byte-for-byte, so saving the document back does not destroy the part
        // that this build could not read.
        assert_eq!(serde_json::to_value(&node).unwrap(), newer);
    }

    /// An extrude written when it still carried a frame opens, and sweeps straight out.
    ///
    /// The literal JSON is written out rather than produced by this build, for the reason the
    /// lattice test below gives: a test that generates its own input can only agree with itself
    /// about what the old format was.
    ///
    /// **Not merely that it opens.** The frame is dropped rather than converted into a
    /// direction, and that is the claim worth pinning: the frame's aim was supposed to be the
    /// profile's own plane, which is now derived, so keeping the old number would preserve
    /// exactly the disagreement the split removed.
    #[test]
    fn an_extrude_written_with_a_frame_still_reads() {
        let profile = CurveId::new();
        let old = serde_json::json!({
            "kind": "extrude",
            "profile": profile.to_string(),
            "frame": { "at": { "origin": [{ "literal": 0.0 }, { "literal": 0.0 },
                                           { "literal": 0.0 }],
                               "axis": [{ "literal": 1.0 }, { "literal": 0.0 },
                                        { "literal": 0.0 }] } },
            "height": { "literal": 6.0 },
        });
        let node: Node = serde_json::from_value(old).expect("an older extrude still reads");
        assert_eq!(
            node,
            Node::Extrude { rims: Default::default(),
                outline: Outline::closed(profile),
                sweep: Sweep::Normal,
                height: ParamRef::Literal(6.0),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: crate::Sides::One,
                corners: Default::default(),
            }
        );
    }

    /// The promise an outline was added under: an extrude of a closed profile writes what it
    /// wrote before there was any other kind, field for field.
    ///
    /// The JSON is written out rather than produced by this build, so that what it is held to
    /// is the format as it was and not whatever the writer happens to do now.
    #[test]
    fn an_extrude_of_a_closed_profile_writes_what_it_always_wrote() {
        let profile = CurveId::new();
        let node = Node::Extrude { rims: Default::default(),
            outline: Outline::closed(profile),
            sweep: Sweep::Normal,
            height: ParamRef::Literal(40.0),
            draft: ParamRef::Literal(3.0),
            start: ParamRef::Literal(0.0),
            sides: crate::Sides::One,
            corners: Default::default(),
        };
        let as_it_was = serde_json::json!({
            "kind": "extrude",
            "profile": profile.to_string(),
            "sweep": "normal",
            "height": { "literal": 40.0 },
            "draft": { "literal": 3.0 },
        });
        assert_eq!(serde_json::to_value(&node).unwrap(), as_it_was);
        assert_eq!(
            serde_json::to_string(&node).unwrap(),
            serde_json::to_string(&as_it_was).unwrap(),
            "and byte for byte, not only value for value"
        );
        assert_eq!(serde_json::from_value::<Node>(as_it_was).unwrap(), node);
    }

    /// A wall is written flat, in the profile's place, with what is default left out — and
    /// what is left out comes back as the default.
    #[test]
    fn a_wall_is_written_in_the_profiles_place_and_its_defaults_are_left_out() {
        let curve = CurveId::new();
        let wall = |side, ends| Node::Extrude { rims: Default::default(),
            outline: Outline::Wall { curve, thickness: ParamRef::Literal(2.5), side, ends },
            sweep: Sweep::Normal,
            height: ParamRef::Literal(40.0),
            draft: ParamRef::Literal(0.0),
            start: ParamRef::Literal(0.0),
            sides: crate::Sides::One,
            corners: Default::default(),
        };

        let plain = wall(WallSide::Centre, WallEnds::Square);
        let written = serde_json::to_value(&plain).unwrap();
        assert_eq!(
            written,
            serde_json::json!({
                "kind": "extrude",
                "wall": curve.to_string(),
                "thickness": { "literal": 2.5 },
                "sweep": "normal",
                "height": { "literal": 40.0 },
                "draft": { "literal": 0.0 },
            })
        );
        assert_eq!(serde_json::from_value::<Node>(written).unwrap(), plain);

        let sided = wall(WallSide::Left, WallEnds::Round);
        let written = serde_json::to_value(&sided).unwrap();
        assert_eq!(written["side"], "left");
        assert_eq!(written["ends"], "round");
        assert!(written.get("profile").is_none(), "a wall is not also a profile");
        assert_eq!(serde_json::from_value::<Node>(written).unwrap(), sided);
    }

    /// An extrude is built from a profile or from a wall. A file that names both, or hangs a
    /// thickness on a profile, is saying something this build cannot mean — and a side it has
    /// not heard of is one it cannot draw. Each is kept as it came and refused, not guessed at
    /// and not allowed to fail the document.
    #[test]
    fn an_extrude_this_build_cannot_mean_is_kept_and_refused() {
        let curve = CurveId::new().to_string();
        let base = |extra: serde_json::Value| {
            let mut node = serde_json::json!({
                "kind": "extrude",
                "sweep": "normal",
                "height": { "literal": 40.0 },
                "draft": { "literal": 0.0 },
            });
            for (key, value) in extra.as_object().unwrap() {
                node[key] = value.clone();
            }
            node
        };
        for (what, newer) in [
            ("both", base(serde_json::json!({ "profile": curve, "wall": curve,
                                               "thickness": { "literal": 2.0 } }))),
            ("a thickness on a profile",
             base(serde_json::json!({ "profile": curve, "thickness": { "literal": 2.0 } }))),
            ("a side from a later build",
             base(serde_json::json!({ "wall": curve, "thickness": { "literal": 2.0 },
                                      "side": "offset" }))),
            ("ends from a later build",
             base(serde_json::json!({ "wall": curve, "thickness": { "literal": 2.0 },
                                      "ends": "mitred" }))),
        ] {
            let node: Node = serde_json::from_value(newer.clone())
                .unwrap_or_else(|e| panic!("{what} failed the read outright: {e}"));
            assert!(matches!(node, Node::Unknown(_)), "{what} was read as {node:?}");
            assert_eq!(serde_json::to_value(&node).unwrap(), newer, "{what} was not kept whole");
        }
        // A wall with no thickness is damaged rather than newer, and that is an error.
        let damaged = base(serde_json::json!({ "wall": curve }));
        assert!(serde_json::from_value::<Node>(damaged).is_err());
    }

    /// The outline carries its own wiring, so a wall's curve cannot be deleted from under it
    /// and its thickness can be named and driven — and the three numbers an extrude always
    /// had are where they always were.
    #[test]
    fn a_wall_extrudes_curve_and_thickness_are_the_graphs_to_see() {
        let (curve, other) = (CurveId::new(), CurveId::new());
        let mut node = Node::Extrude { rims: Default::default(),
            outline: Outline::Wall {
                curve,
                thickness: ParamRef::Literal(2.5),
                side: WallSide::Left,
                ends: WallEnds::Square,
            },
            sweep: Sweep::Normal,
            height: ParamRef::Literal(40.0),
            draft: ParamRef::Literal(3.0),
            start: ParamRef::Literal(0.0),
            sides: crate::Sides::One,
            corners: Default::default(),
        };
        assert_eq!(node.curves(), vec![curve]);
        for id in node.curves_mut() {
            *id = other;
        }
        assert_eq!(node.curves(), vec![other], "a paste must be able to redirect a wall's curve");
        // The thickness after the depth and the draft, where it has been since a wall arrived,
        // and the start after it, where it arrived later, and the rims' reach last of all, where
        // it arrived latest -- so no number moved for another.
        assert_eq!(
            node.param_refs(),
            vec![
                &ParamRef::Literal(40.0),
                &ParamRef::Literal(3.0),
                &ParamRef::Literal(2.5),
                &ParamRef::Literal(0.0),
                &ParamRef::Literal(0.0)
            ]
        );
    }

    /// A revolve and a sweep, each on the same outline, for the tests below: what an extrude
    /// was held to when it took an outline, the other two are held to as well.
    fn revolved(outline: Outline) -> Node {
        Node::Revolve { rims: Default::default(), outline, frame: Frame::default(), angle: crate::ParamRef::Literal(360.0), sides: crate::AngleSides::One }
    }

    fn swept(outline: Outline, spine: CurveId) -> Node {
        Node::Sweep { rims: Default::default(), outline, spine, section: Section::Skewed, law: Default::default() }
    }

    /// The same promise for the other two blocks an outline went onto: a revolve and a sweep
    /// of a closed profile write what they wrote before either could take anything else.
    #[test]
    fn a_revolve_and_a_sweep_of_a_closed_profile_write_what_they_always_wrote() {
        let (profile, spine) = (CurveId::new(), CurveId::new());
        for (node, as_it_was) in [
            (
                revolved(Outline::closed(profile)),
                serde_json::json!({
                    "kind": "revolve",
                    "profile": profile.to_string(),
                    // The frame's own form is not what is being held here.
                    "frame": serde_json::to_value(Frame::default()).unwrap(),
                }),
            ),
            (
                swept(Outline::closed(profile), spine),
                serde_json::json!({
                    "kind": "sweep",
                    "profile": profile.to_string(),
                    "spine": spine.to_string(),
                    "section": "skewed",
                }),
            ),
        ] {
            assert_eq!(serde_json::to_value(&node).unwrap(), as_it_was);
            assert_eq!(
                serde_json::to_string(&node).unwrap(),
                serde_json::to_string(&as_it_was).unwrap(),
                "and byte for byte, not only value for value"
            );
            assert_eq!(serde_json::from_value::<Node>(as_it_was).unwrap(), node);
        }
    }

    /// A wall is written the one way on all three blocks: flat, in the profile's place, with
    /// what is default left out and coming back as the default.
    #[test]
    fn a_wall_is_written_in_the_profiles_place_on_a_revolve_and_a_sweep() {
        let (curve, spine) = (CurveId::new(), CurveId::new());
        let wall = |side, ends| Outline::Wall { curve, thickness: ParamRef::Literal(2.5), side, ends };
        let both = |side, ends| [revolved(wall(side, ends)), swept(wall(side, ends), spine)];

        for plain in both(WallSide::Centre, WallEnds::Square) {
            let written = serde_json::to_value(&plain).unwrap();
            assert_eq!(written["wall"], curve.to_string());
            assert_eq!(written["thickness"], serde_json::json!({ "literal": 2.5 }));
            for absent in ["profile", "side", "ends"] {
                assert!(written.get(absent).is_none(), "{absent} was written for {plain:?}");
            }
            assert_eq!(serde_json::from_value::<Node>(written).unwrap(), plain);
        }
        for sided in both(WallSide::Right, WallEnds::Round) {
            let written = serde_json::to_value(&sided).unwrap();
            assert_eq!(written["side"], "right");
            assert_eq!(written["ends"], "round");
            assert_eq!(serde_json::from_value::<Node>(written).unwrap(), sided);
        }
    }

    /// And what neither may be: built from both a profile and a wall, a profile with a
    /// thickness hung on it, or a wall with a side or an end this build has no word for. Each
    /// is kept as it came and refused, as an extrude saying the same is.
    #[test]
    fn a_revolve_or_a_sweep_this_build_cannot_mean_is_kept_and_refused() {
        let curve = CurveId::new().to_string();
        let bases = [
            serde_json::json!({
                "kind": "revolve",
                "frame": serde_json::to_value(Frame::default()).unwrap(),
            }),
            serde_json::json!({ "kind": "sweep", "spine": curve, "section": "square" }),
        ];
        for base in bases {
            let with = |extra: serde_json::Value| {
                let mut node = base.clone();
                for (key, value) in extra.as_object().unwrap() {
                    node[key] = value.clone();
                }
                node
            };
            let thick = serde_json::json!({ "literal": 2.0 });
            for (what, newer) in [
                ("both", with(serde_json::json!({ "profile": curve, "wall": curve, "thickness": thick }))),
                ("a thickness on a profile", with(serde_json::json!({ "profile": curve, "thickness": thick }))),
                (
                    "a side from a later build",
                    with(serde_json::json!({ "wall": curve, "thickness": thick, "side": "offset" })),
                ),
                (
                    "ends from a later build",
                    with(serde_json::json!({ "wall": curve, "thickness": thick, "ends": "mitred" })),
                ),
            ] {
                let node: Node = serde_json::from_value(newer.clone())
                    .unwrap_or_else(|e| panic!("{what} failed the read outright: {e}"));
                assert!(matches!(node, Node::Unknown(_)), "{what} was read as {node:?}");
                assert_eq!(serde_json::to_value(&node).unwrap(), newer, "{what} was not kept whole");
            }
            // A wall with no thickness is damaged rather than newer, and that is an error.
            assert!(serde_json::from_value::<Node>(with(serde_json::json!({ "wall": curve }))).is_err());
        }
    }

    /// The outline carries its own wiring onto these two as it did onto an extrude: the
    /// wall's curve is a curve the graph depends on, and its thickness a number that can be
    /// named — last, so what each block already listed stays where it was.
    #[test]
    fn a_walls_curve_and_thickness_are_the_graphs_to_see_on_a_revolve_and_a_sweep() {
        let (curve, spine, other) = (CurveId::new(), CurveId::new(), CurveId::new());
        let thickness = ParamRef::Literal(2.5);
        let wall =
            Outline::Wall { curve, thickness, side: WallSide::Left, ends: WallEnds::Square };

        let mut revolve = revolved(wall);
        assert_eq!(revolve.curves(), vec![curve]);
        let frame_alone = revolved(Outline::closed(curve)).param_refs().len();
        assert_eq!(revolve.param_refs().len(), frame_alone + 1);
        // The thickness after the frame's, where it has been since a wall arrived, the angle
        // after it, where it arrived later, and the rims' reach last of all (3.21) -- so no
        // number moved for another.
        assert_eq!(
            revolve.param_refs().iter().rev().nth(2),
            Some(&&thickness),
            "the thickness goes after the frame's"
        );
        assert_eq!(revolve.param_refs().iter().rev().nth(1), Some(&&ParamRef::Literal(360.0)));
        assert_eq!(revolve.param_refs().last(), Some(&&ParamRef::Literal(0.0)), "a hard rim's reach, last");
        for id in revolve.curves_mut() {
            *id = other;
        }
        assert_eq!(revolve.curves(), vec![other], "a paste must be able to redirect it");

        let mut sweep = swept(wall, spine);
        assert_eq!(sweep.curves(), vec![curve, spine]);
        // The thickness first, then the plain law's three numbers: a twist of nought and an
        // extent of the whole path, which every sweep now carries -- and the rims' reach last.
        assert_eq!(sweep.param_refs().first(), Some(&&thickness));
        assert_eq!(sweep.param_refs().len(), 5);
        assert_eq!(swept(Outline::closed(curve), spine).param_refs().len(), 4);
        for id in sweep.curves_mut() {
            *id = other;
        }
        assert_eq!(sweep.curves(), vec![other, other]);
    }

    /// Each of the three blocks built on a section hands it back, to be read or exchanged, and
    /// says which curve its section's walk is to leave out — and nothing else has either.
    #[test]
    fn the_blocks_built_on_a_section_hand_back_its_outline_and_what_they_are_aimed_by() {
        let (curve, spine, line) = (CurveId::new(), CurveId::new(), CurveId::new());
        let wall = Outline::Wall {
            curve,
            thickness: ParamRef::Literal(2.0),
            side: WallSide::Right,
            ends: WallEnds::Round,
        };
        let extrude = Node::Extrude { rims: Default::default(),
            outline: wall,
            sweep: crate::Sweep::Along { curve: line },
            height: ParamRef::Literal(10.0),
            draft: ParamRef::Literal(0.0),
            start: ParamRef::Literal(0.0),
            sides: crate::Sides::One,
            corners: Default::default(),
        };
        let mut blocks = [(extrude, Some(line)), (revolved(wall), None), (swept(wall, spine), Some(spine))];
        for (block, aimed_by) in &mut blocks {
            assert_eq!(block.outline(), Some(&wall));
            assert_eq!(block.aimed_by(), *aimed_by);
            // Exchanged for the other kind in place, which is what the Wall checkbox does.
            *block.outline_mut().expect("it has one") = Outline::closed(curve);
            assert_eq!(block.outline(), Some(&Outline::closed(curve)));
        }
        let ball = Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(1.0),
        };
        assert_eq!(ball.outline(), None);
        assert_eq!(ball.aimed_by(), None);
    }

    #[test]
    fn a_pattern_saved_before_the_count_was_a_variable_still_opens() {
        // The shape every document written before 2026-09-30 carries: a bare number.
        let text = serde_json::json!({
            "kind": "pattern",
            "input": NodeId::new(),
            "count": 6,
            "arrangement": "linear",
            "step": [ParamRef::Literal(12.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
            "join": true,
            "radius": ParamRef::Literal(0.0),
            "continuity": Continuity::G1,
        });
        let node: Node = serde_json::from_value(text).expect("the old spelling still reads");
        let Node::Pattern { count, .. } = &node else {
            panic!("read back as {node:?}, not a pattern")
        };
        assert_eq!(
            *count,
            ParamRef::Literal(6.0),
            "a bare number has exactly one reading, and this is it",
        );

        // And it is written back as a bare number, which is what keeps the bump minor: a document
        // nobody has parameterized is byte for byte what the previous format wrote.
        let out = serde_json::to_value(&node).expect("serializes");
        assert_eq!(out["count"], serde_json::json!(6), "still the old spelling, so 3.3 opens it");

        // A count that genuinely names a variable cannot be written that way, and is not.
        let named = Node::Pattern {
            options: Default::default(),
            input: NodeId::new().into(),
            count: ParamRef::Named(crate::ParamId::new()),
            kind: PatternKind::Linear {
                step: [ParamRef::Literal(12.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
            },
            join: true,
            fillet: Fillet::hard(),
        };
        let out = serde_json::to_value(&named).expect("serializes");
        assert!(
            !out["count"].is_number(),
            "a variable count has to be written in the richer spelling: {out}",
        );
        assert_eq!(
            serde_json::from_value::<Node>(out).expect("reads back"),
            named,
            "and round-trips as itself",
        );
    }

    /// **A count is always a whole number, and says when it had to become one.**
    ///
    /// The owner reversed the first ruling on the consequence: refusing a fraction meant a variable
    /// driven through one — `copies / 2`, a ratio, an interpolation — broke the part for a value
    /// nobody typed. So a fraction rounds, and [`CountFit`] is what keeps that from being silent,
    /// which is the clause `arc_count`'s doc actually insists on.
    #[test]
    fn a_pattern_count_is_always_whole_and_says_when_it_rounded() {
        let mut doc = crate::Document::new();
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(6.0)),
            Ok((6, CountFit::Exact)),
            "six is six, and nothing is said about it",
        );
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(3.5)),
            Ok((4, CountFit::Rounded(3.5))),
            "a fraction builds a whole number of copies rather than breaking the part",
        );
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(3.4)),
            Ok((3, CountFit::Rounded(3.4))),
            "to the nearest, following arc_count_for",
        );
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(0.6)),
            Ok((1, CountFit::Rounded(0.6))),
            "refused on the rounded value, so this is one copy rather than nothing",
        );
        assert_eq!(
            CountFit::Rounded(3.5).say(4).as_deref(),
            Some("4 copies, rounded from 3.5"),
            "and the figure built is said beside the expression that did not say it",
        );
        assert_eq!(CountFit::Exact.say(6), None, "an exact count has nothing to announce");

        // What cannot be repaired into a usable number is still refused.
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(0.0)),
            Err(CountError::NotPositive(0.0)),
            "a pattern of nothing is a broken expression, not the body",
        );
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(-4.0)),
            Err(CountError::NotPositive(-4.0)),
            "and neither is a negative one",
        );
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(f64::NAN)),
            Err(CountError::NotFinite),
            "a NaN is refused rather than saturating to one copy",
        );
        assert_eq!(
            pattern_count(&doc, &ParamRef::Field(NodeId::new())),
            Err(CountError::Driven),
            "a ramp's value varies with position and a number of copies cannot",
        );

        // The ceiling caps, and reports the cap in preference to the rounding, because a cap moves
        // the count by more than half and is the bigger surprise.
        assert_eq!(
            pattern_count(&doc, &ParamRef::Literal(10_000.6)),
            Ok((PATTERN_COUNT_MAX, CountFit::Capped(10_000.6))),
            "capped, and that is what the row says",
        );

        // Asked of the document, never read off the parameter: an expression resolves through
        // everything it reads, and a name nothing holds is the same `None` a failure gives.
        let id = doc.add_parameter("copies", 4.0);
        assert_eq!(pattern_count(&doc, &ParamRef::Named(id)), Ok((4, CountFit::Exact)));
        assert_eq!(
            pattern_count(&crate::Document::new(), &ParamRef::Named(id)),
            Err(CountError::Unresolved),
            "a parameter the document does not hold is refused, not read as nought",
        );

        // **The stored value is untouched.** A parameter is document-wide, so rounding what is
        // stored would move anything else reading the same name — a radius, a spacing — to suit a
        // count. Quantise on read is the whole of the rule.
        let half = doc.add_parameter("half", 3.5);
        assert_eq!(pattern_count(&doc, &ParamRef::Named(half)), Ok((4, CountFit::Rounded(3.5))));
        assert_eq!(
            doc.parameter_value(half),
            Some(3.5),
            "the parameter still says 3.5 — only the count rounded",
        );
    }

    /// **The two parameter lists agree, position for position.**
    ///
    /// A slot's index in `param_refs` is used as a handle to write through `param_refs_mut` — the
    /// app's `wire_driver` reaches a socket by index and `release_drivers` writes back by
    /// enumeration — and until the count was added nothing checked that the two lists were built in
    /// the same order. Adding a field to one and not the other, or appending to one and prepending
    /// to the other, writes a ramp into the wrong number and nothing says a word.
    #[test]
    fn a_nodes_two_parameter_lists_are_in_the_same_order() {
        let input = NodeId::new();
        let nodes = [
            Node::Pattern {
                options: Default::default(),
                input: input.into(),
                count: ParamRef::Literal(6.0),
                kind: PatternKind::Linear {
                    step: [ParamRef::Literal(1.0), ParamRef::Literal(2.0), ParamRef::Literal(3.0)],
                },
                join: false,
                fillet: Fillet::of(ParamRef::Literal(4.0)),
            },
            Node::Pattern {
                options: Default::default(),
                input: input.into(),
                count: ParamRef::Literal(7.0),
                kind: PatternKind::Radial { frame: Frame::default() },
                join: true,
                fillet: Fillet::hard(),
            },
        ];
        for node in nodes {
            let mut copy = node.clone();
            let read: Vec<ParamRef> = node.param_refs().into_iter().copied().collect();
            let written: Vec<ParamRef> =
                copy.param_refs_mut().into_iter().map(|r| *r).collect();
            assert_eq!(read, written, "the two lists disagree for {node:?}");
            let Node::Pattern { count, kind, options, .. } = &node else { unreachable!() };
            // **The count is the last of what a pattern held before it had options**, which is
            // what leaves every other index where it was. The options' own numbers follow it: a
            // ring's angle, and a row's second direction when it has one.
            let after = match kind {
                PatternKind::Linear { .. } => options.second.map_or(0, |_| 4),
                PatternKind::Radial { .. } => 1,
            };
            assert_eq!(
                read.iter().rev().nth(after),
                Some(count),
                "the count stayed where it was, with only the options' numbers after it",
            );
        }
    }

    fn leaning_box(centre: crate::Place) -> Node {
        let lit = ParamRef::Literal;
        Node::SkewedBox {
            centre,
            edges: [
                [lit(10.0), lit(0.0), lit(0.0)],
                [lit(4.0), lit(8.0), lit(0.0)],
                [lit(3.0), lit(2.0), lit(6.0)],
            ],
        }
    }

    /// A skewed box is written as `skewed_box` with its centre and its three edges, and comes back
    /// the same node — the property every kind has, and the one that catches a kind whose reader and
    /// writer disagree on a field name.
    #[test]
    fn a_skewed_box_comes_back_as_a_skewed_box() {
        let node = leaning_box(crate::Place::at(1.0, 2.0, 3.0));
        let text = serde_json::to_value(&node).expect("serializes");
        assert_eq!(text["kind"], serde_json::json!("skewed_box"), "{text}");
        assert!(text["edges"].as_array().is_some_and(|e| e.len() == 3), "{text}");
        let back: Node = serde_json::from_value(text).expect("deserializes");
        assert_eq!(back, node, "a skewed box stopped being one");
    }

    /// All twelve edge numbers, and the centre's three, are parameters the document can drive — and
    /// the same list is what `param_refs_mut` hands back to be edited, so a variable bound to one
    /// edge component edits that component and no other.
    #[test]
    fn every_number_a_skewed_box_holds_is_a_parameter_in_order() {
        let mut node = leaning_box(crate::Place::at(0.0, 0.0, 0.0));
        assert_eq!(node.param_refs().len(), 3 + 9, "a centre's three and nine edge components");
        for (i, slot) in node.param_refs_mut().into_iter().enumerate() {
            *slot = ParamRef::Literal(100.0 + i as f64);
        }
        let Node::SkewedBox { centre, edges } = &node else { panic!("still a skewed box") };
        assert_eq!(*centre, crate::Place::at(100.0, 101.0, 102.0));
        assert_eq!(edges[0][0], ParamRef::Literal(103.0), "the first edge follows the centre");
        assert_eq!(edges[2][2], ParamRef::Literal(111.0), "and the last component is last");
    }

    /// **A skewed box standing on a point block is wired to it.** `places` is the list that makes a
    /// place a graph edge, a remap and a cycle check all at once (D9), and it is a wildcard match —
    /// so leaving the new node out compiles, and silently makes its centre unable to follow a point.
    #[test]
    fn a_skewed_box_whose_centre_is_a_point_depends_on_it() {
        let mut g = Graph::empty();
        let point = g.add(Node::Point { at: [ParamRef::Literal(0.0); 3] });
        let body = g.add(leaning_box(crate::Place::From { node: point }));
        assert_eq!(g.node(body).unwrap().inputs(), vec![point], "the point is an input of the box");
        assert!(
            !g.roots().contains(&point),
            "and a point read as a place stops being a body of its own"
        );
    }

    /// A pattern comes back from a round trip as a pattern.
    ///
    /// **Not that the text survives — that the node does.** The pattern's own kind was written
    /// under `"kind"`, which is the same key the *node's* kind uses, so the second overwrote the
    /// first: a saved pattern came back tagged `"radial"`, which is not a node kind, and was
    /// preserved and refused. The document still opened and the pattern was simply gone.
    ///
    /// Nothing caught it, and that is the part worth keeping. A round trip through this module
    /// *passed*, because an unrecognised node is preserved byte for byte on purpose (D3) — so
    /// the JSON went out and came back identical while the thing it described stopped existing.
    /// Comparing the text and comparing the node are different questions, and only one of them
    /// was being asked.
    #[test]
    fn a_pattern_comes_back_as_a_pattern() {
        let input = NodeId::new();
        for kind in [
            PatternKind::Linear { step: [ParamRef::Literal(12.0), ParamRef::Literal(0.0),
                                         ParamRef::Literal(0.0)] },
            PatternKind::Radial { frame: Frame::default() },
        ] {
            for (join, fillet) in
                [(true, Fillet::hard()), (false, Fillet::of(ParamRef::Literal(3.0)))]
            {
                let node =
                    Node::Pattern {
                        options: Default::default(),
                        input: input.into(),
                        count: ParamRef::Literal(7.0),
                        kind: kind.clone(),
                        join,
                        fillet,
                    };
                let text = serde_json::to_value(&node).expect("serializes");
                // The node's own kind survives, which is what was being overwritten.
                assert_eq!(text["kind"], serde_json::json!("pattern"), "{text}");
                let back: Node = serde_json::from_value(text).expect("deserializes");
                assert_eq!(back, node, "a pattern stopped being a pattern");
            }
        }
    }

    fn a_row() -> PatternKind {
        let lit = ParamRef::Literal;
        PatternKind::Linear { step: [lit(12.0), lit(0.0), lit(0.0)] }
    }

    fn pattern_with(kind: PatternKind, options: PatternOptions) -> Node {
        Node::Pattern {
            input: NodeId::new().into(),
            count: ParamRef::Literal(7.0),
            kind,
            join: false,
            fillet: Fillet::hard(),
            options,
        }
    }

    /// **A pattern that uses none of the options writes none of their keys**, which is what lets a
    /// document nobody has touched them in open in a build that has never heard of them. The
    /// bytes are what is asked, not the node: a round trip through this build would pass whether
    /// or not an extra `"symmetric": false` had crept into every pattern in every saved file.
    #[test]
    fn a_pattern_using_no_options_writes_none_of_their_keys() {
        for kind in [a_row(), PatternKind::Radial { frame: Frame::default() }] {
            let text = serde_json::to_value(pattern_with(kind, PatternOptions::default()))
                .expect("serializes");
            for key in ["symmetric", "skip", "spacing", "step2", "count2", "angle"] {
                assert!(text.get(key).is_none(), "an unused option {key:?} was written: {text}");
            }
        }
    }

    /// Every option, on the kind it means something to, comes back as it went.
    #[test]
    fn every_pattern_option_comes_back_as_it_went() {
        let lit = ParamRef::Literal;
        let row = PatternOptions {
            spacing: Spacing::Extent,
            symmetric: true,
            second: Some(PatternSecond {
                step: [lit(0.0), lit(9.0), lit(0.0)],
                count: lit(4.0),
            }),
            skip: vec![1, 4, 9],
            ..PatternOptions::default()
        };
        let ring = PatternOptions {
            symmetric: true,
            angle: lit(180.0),
            skip: vec![2],
            ..PatternOptions::default()
        };
        for (kind, options) in
            [(a_row(), row), (PatternKind::Radial { frame: Frame::default() }, ring)]
        {
            let node = pattern_with(kind, options);
            let text = serde_json::to_value(&node).expect("serializes");
            let back: Node = serde_json::from_value(text.clone()).expect("deserializes");
            assert_eq!(back, node, "an option was lost on the way through: {text}");
        }
    }

    /// A ring's file does not carry a row's keys, nor a row's a ring's: an option that means
    /// nothing to a kind is not written for it, so it cannot be mistaken for one that does.
    #[test]
    fn an_option_that_means_nothing_to_a_kind_is_not_written_for_it() {
        let lit = ParamRef::Literal;
        let everything = PatternOptions {
            spacing: Spacing::Extent,
            symmetric: true,
            angle: lit(90.0),
            second: Some(PatternSecond { step: [lit(1.0); 3], count: lit(2.0) }),
            skip: vec![1],
        };
        let row = serde_json::to_value(pattern_with(a_row(), everything.clone())).unwrap();
        assert!(row.get("angle").is_none(), "a row was given a ring's angle: {row}");
        let ring = serde_json::to_value(pattern_with(
            PatternKind::Radial { frame: Frame::default() },
            everything,
        ))
        .unwrap();
        for key in ["spacing", "step2", "count2"] {
            assert!(ring.get(key).is_none(), "a ring was given a row's {key:?}: {ring}");
        }
    }

    /// A second direction's count is a variable like the first, written as one only when it is.
    #[test]
    fn a_second_count_that_is_a_variable_survives_and_a_plain_one_stays_a_bare_number() {
        let lit = ParamRef::Literal;
        let plain = pattern_with(
            a_row(),
            PatternOptions {
                second: Some(PatternSecond { step: [lit(0.0), lit(9.0), lit(0.0)], count: lit(3.0) }),
                ..PatternOptions::default()
            },
        );
        let text = serde_json::to_value(&plain).unwrap();
        assert_eq!(text["count2"], serde_json::json!(3), "{text}");
        let named = pattern_with(
            a_row(),
            PatternOptions {
                second: Some(PatternSecond {
                    step: [lit(0.0), lit(9.0), lit(0.0)],
                    count: ParamRef::Named(ParamId::new()),
                }),
                ..PatternOptions::default()
            },
        );
        let text = serde_json::to_value(&named).unwrap();
        let back: Node = serde_json::from_value(text).unwrap();
        assert_eq!(back, named);
    }

    /// Half a second direction is not one, and the missing half is not guessed: the node is kept
    /// and refused, with its bytes, which is what every unrecognised thing gets (D3).
    #[test]
    fn a_step_without_its_count_is_kept_and_refused_rather_than_guessed() {
        let node = serde_json::json!({
            "kind": "pattern", "input": NodeId::new(), "count": 4, "arrangement": "linear",
            "step": [{ "literal": 5.0 }, { "literal": 0.0 }, { "literal": 0.0 }],
            "step2": [{ "literal": 0.0 }, { "literal": 5.0 }, { "literal": 0.0 }],
        });
        let read: Node = serde_json::from_value(node.clone()).unwrap();
        assert!(matches!(read, Node::Unknown(_)), "a half-written grid was read as {read:?}");
        assert_eq!(serde_json::to_value(&read).unwrap(), node, "and its bytes were kept");
    }

    /// A spacing this build does not know is kept, not read as the nearest one it does.
    #[test]
    fn a_spacing_this_build_does_not_know_is_kept_and_refused() {
        let node = serde_json::json!({
            "kind": "pattern", "input": NodeId::new(), "count": 4, "arrangement": "linear",
            "step": [{ "literal": 5.0 }, { "literal": 0.0 }, { "literal": 0.0 }],
            "spacing": "golden",
        });
        let read: Node = serde_json::from_value(node).unwrap();
        assert!(matches!(read, Node::Unknown(_)), "{read:?}");
    }

    /// **The option numbers are parameters, in a stable order after the old ones.** A slot's index
    /// is a handle (`wire_driver` reaches a socket by it), so what was at an index before has to
    /// still be there: the count stays the last of what a plain pattern holds, the options follow
    /// it, and the two walks over the list agree on every length.
    #[test]
    fn the_option_numbers_follow_the_old_ones_and_both_walks_agree() {
        let lit = ParamRef::Literal;
        let plain = pattern_with(a_row(), PatternOptions::default());
        let old = plain.param_refs().len();
        let mut grid = pattern_with(
            a_row(),
            PatternOptions {
                second: Some(PatternSecond { step: [lit(1.0), lit(2.0), lit(3.0)], count: lit(5.0) }),
                ..PatternOptions::default()
            },
        );
        assert_eq!(grid.param_refs().len(), old + 4, "a second direction's step and count");
        assert_eq!(grid.param_refs().len(), grid.param_refs_mut().len());
        for (i, slot) in grid.param_refs_mut().into_iter().enumerate() {
            *slot = ParamRef::Literal(100.0 + i as f64);
        }
        let Node::Pattern { options, count, .. } = &grid else { unreachable!() };
        assert_eq!(*count, ParamRef::Literal(100.0 + (old - 1) as f64), "the count kept its place");
        let second = options.second.expect("still there");
        assert_eq!(second.count, ParamRef::Literal(100.0 + (old + 3) as f64));

        let mut ring = pattern_with(
            PatternKind::Radial { frame: Frame::default() },
            PatternOptions { angle: lit(200.0), ..PatternOptions::default() },
        );
        assert_eq!(ring.param_refs().len(), ring.param_refs_mut().len());
        let last = ring.param_refs_mut().pop().expect("the angle");
        *last = ParamRef::Literal(77.0);
        let Node::Pattern { options, .. } = &ring else { unreachable!() };
        assert_eq!(options.angle, ParamRef::Literal(77.0), "the angle is the ring's last number");
    }

    /// A pattern written before it could leave the body out still holds it.
    ///
    /// The literal JSON rather than one this build produced, for the reason the lattice test
    /// below gives: a file generated here can only agree with this build about what the old
    /// format was. Absent `join` has to mean `true`, because that is what every pattern ever
    /// saved did -- reading it as `false` would quietly delete a copy from every one of them.
    #[test]
    fn a_pattern_written_before_it_could_hold_only_copies_still_holds_the_body() {
        let old = serde_json::json!({
            "kind": "pattern",
            "input": NodeId::new(),
            "count": 6,
            "arrangement": "linear",
            "step": [{ "literal": 12.0 }, { "literal": 0.0 }, { "literal": 0.0 }],
        });
        let node: Node = serde_json::from_value(old).expect("an older pattern still reads");
        assert!(
            matches!(node, Node::Pattern { join: true, .. }),
            "an older pattern came back holding only its copies: {node:?}"
        );
    }

    /// And a lattice written before the split reads back meaning what it meant.
    ///
    /// The literal JSON is written out rather than produced by this build, because a test that
    /// generates its own input can only ever agree with itself about what the old format was.
    #[test]
    fn a_lattice_written_before_the_split_still_reads() {
        let old = serde_json::json!({
            "kind": "lattice",
            "cell": "schwarz_p",
            "map": "rectangular",
            "cell_size": { "literal": 12.5 },
            "thickness": { "literal": 1.25 },
        });
        let node: Node = serde_json::from_value(old).expect("an older lattice still reads");
        assert_eq!(
            node,
            Node::Lattice {
                frame: Frame::default(),
                cell: LatticeCell::SchwarzP,
                map: CellMap::Rectangular { cell: ParamRef::Literal(12.5) },
                phase: LatticePhase::sheet(ParamRef::Literal(1.25)),
            },
        );
    }

    /// A cylindrical lattice round-trips with its count intact and still an integer.
    #[test]
    fn a_cylindrical_lattice_round_trips_with_a_whole_arc_count() {
        let mut g = Graph::empty();
        let node = g.add(Node::Lattice {
            frame: Frame::default(),
            cell: LatticeCell::Gyroid,
            map: CellMap::Cylindrical {
                cell_radius: ParamRef::Literal(5.0),
                cell_height: ParamRef::Literal(7.0),
                arc_count: 38,
            },
            phase: LatticePhase::sheet(ParamRef::Literal(1.0)),
        });
        assert_eq!(round_trip(&g).node(node), g.node(node));

        let value = serde_json::to_value(g.node(node).unwrap()).unwrap();
        assert_eq!(value["map"], "cylindrical");
        // A number, not a string, and whole: the seam depends on it.
        assert_eq!(value["arc_count"], serde_json::json!(38));
    }

    /// A build that predates cylindrical maps preserves one and refuses to evaluate it, rather
    /// than reading it as some other lattice.
    ///
    /// Simulated the only way it can be from inside this build: a map value this build has
    /// never heard of. The mechanism is the same one — an unrecognised value of a tagged field
    /// — so what this asserts is that the gate is actually wired to `map`, which is what makes
    /// a future third tiling safe.
    #[test]
    fn a_lattice_tiled_a_way_this_build_does_not_know_is_kept_rather_than_guessed() {
        let newer = serde_json::json!({
            "kind": "lattice",
            "cell": "gyroid",
            // A tiling no build has. "spherical" was this test's stand-in until it became real,
            // and then "conformal" was, until *that* became real -- at which point this test
            // went on passing by asserting that a conformal lattice is unreadable, which it
            // then was, in the application, for every document that held one. So the
            // placeholder is no longer a word anybody would name a tiling.
            "map": "a tiling from a build newer than this one",
            "cell_radius": { "literal": 5.0 },
            "thickness": { "literal": 1.0 },
        });
        let node: Node = serde_json::from_value(newer.clone()).expect("preserved, not an error");
        assert!(matches!(node, Node::Unknown(_)), "got {node:?}");
        // And it is written back byte for byte, so opening and saving loses nothing.
        assert_eq!(serde_json::to_value(&node).unwrap(), newer);
    }

    /// A sweep this build writes is a sweep this build reads, however its section is held.
    ///
    /// **The same guard the maps have, for the same reason.** A value the writer can produce and
    /// the reader's vocabulary has not been told about comes back as `Unknown`, which is the
    /// format doing exactly what it promises — and cost a conformal lattice its own save once.
    /// One of each, written and read back.
    #[test]
    fn every_section_this_build_writes_is_one_it_reads() {
        for section in [Section::Square, Section::Skewed, Section::Parallel] {
            let node = Node::Sweep { rims: Default::default(),
                outline: Outline::closed(CurveId::new()),
                spine: CurveId::new(),
                section,
                law: Default::default(),
            };
            let written = serde_json::to_value(&node).expect("a sweep writes");
            let read: Node = serde_json::from_value(written.clone()).expect("and reads");
            assert!(!matches!(read, Node::Unknown(_)), "came back unreadable: {written}");
            assert_eq!(read, node);
        }
    }

    /// And a sweep written before the choice existed reads as the one it was: square.
    #[test]
    fn a_sweep_written_without_a_section_reads_as_square() {
        let older = serde_json::json!({
            "kind": "sweep",
            "profile": CurveId::new(),
            "spine": CurveId::new(),
        });
        let read: Node = serde_json::from_value(older).expect("an older sweep still reads");
        assert!(
            matches!(read, Node::Sweep { section: Section::Square, .. }),
            "got {read:?}"
        );
    }

    #[test]
    fn every_cell_this_build_writes_is_one_it_reads() {
        // The rule the conformal map's lost save taught: a value written to the format has to
        // be in the reader's vocabulary, or the document comes back "Unreadable" from its own
        // save. A match rather than a list, so a cell added to the enum stops this compiling
        // until it is walked here too.
        let every = |c: LatticeCell| match c {
            LatticeCell::Gyroid
            | LatticeCell::SchwarzP
            | LatticeCell::SchwarzD
            | LatticeCell::SchoenIwp
            | LatticeCell::Neovius
            | LatticeCell::FischerKochS
            | LatticeCell::Cubic
            | LatticeCell::Bcc
            | LatticeCell::Fcc
            | LatticeCell::Octet
            | LatticeCell::Diamond
            | LatticeCell::Honeycomb
            | LatticeCell::VoronoiFoam { .. }
            | LatticeCell::VoronoiBeam { .. } => c,
        };
        for cell in [
            LatticeCell::Gyroid,
            LatticeCell::SchwarzP,
            LatticeCell::SchwarzD,
            LatticeCell::SchoenIwp,
            LatticeCell::Neovius,
            LatticeCell::FischerKochS,
            LatticeCell::Cubic,
            LatticeCell::Bcc,
            LatticeCell::Fcc,
            LatticeCell::Octet,
            LatticeCell::Diamond,
            LatticeCell::Honeycomb,
            LatticeCell::VoronoiFoam { jitter: ParamRef::Literal(0.4), seed: 7 },
            LatticeCell::VoronoiBeam { jitter: ParamRef::Literal(1.0), seed: u64::MAX },
        ] {
            let node = Node::Lattice {
                cell: every(cell),
                frame: Frame::default(),
                map: CellMap::Rectangular { cell: ParamRef::Literal(10.0) },
                phase: if cell.is_beam() || matches!(cell, LatticeCell::VoronoiBeam { .. }) {
                    LatticePhase::Strut { radius: ParamRef::Literal(1.0) }
                } else {
                    LatticePhase::sheet(ParamRef::Literal(1.0))
                },
            };
            let written = serde_json::to_value(&node).expect("a lattice writes");
            let read: Node = serde_json::from_value(written.clone()).expect("and reads");
            assert!(!matches!(read, Node::Unknown(_)), "came back unreadable: {written}");
            assert_eq!(read, node);
        }
    }

    /// A lattice this build writes is a lattice this build reads, whatever its map.
    ///
    /// The list of map names a reader accepts and the arms that read them are two descriptions
    /// of one vocabulary, and they disagreed: `conformal` could be written and could be read
    /// and was refused in between, so a conformal lattice did not survive its own save. One of
    /// each map, written and read back, and none of them may come back unknown.
    #[test]
    fn every_map_this_build_writes_is_one_it_reads() {
        let lit = ParamRef::Literal;
        let maps = [
            CellMap::Rectangular { cell: lit(10.0) },
            CellMap::Cylindrical { cell_radius: lit(5.0), cell_height: lit(8.0), arc_count: 12 },
            CellMap::Spherical { cell_radius: lit(5.0), azimuth_count: 12, polar_count: 6 },
            CellMap::Conformal {
                surface: crate::QuadMeshId::new(),
                cells_per_face: 1,
                depth: lit(36.0),
                thickness: None,
            },
            // As the owner's document has it: confined to a shell of the surface.
            CellMap::Conformal {
                surface: crate::QuadMeshId::new(),
                cells_per_face: 2,
                depth: lit(36.0),
                thickness: Some(lit(20.0)),
            },
        ];
        for map in maps {
            let node = Node::Lattice {
                cell: LatticeCell::SchwarzP,
                frame: Frame::default(),
                map,
                phase: LatticePhase::sheet(lit(2.0)),
            };
            let written = serde_json::to_value(&node).expect("a lattice writes");
            let read: Node = serde_json::from_value(written.clone()).expect("and reads");
            assert!(!matches!(read, Node::Unknown(_)), "came back unreadable: {written}");
            assert_eq!(read, node);
        }
    }

    /// Every round solid, through JSON and back with each of its own measurements intact.
    ///
    /// Written out one kind at a time rather than looped, because the thing that would go
    /// wrong is a field written under one name and read under another — a cone's two radii
    /// swapped, or its two endpoints — and a loop over a shared shape could not catch that.
    #[test]
    fn every_round_solid_round_trips_with_its_own_measurements() {
        let param = ParamId::new();
        let base = [ParamRef::Literal(1.0), ParamRef::Literal(2.0), ParamRef::Named(param)];
        let top = [ParamRef::Literal(4.0), ParamRef::Literal(5.0), ParamRef::Literal(6.0)];
        let mut g = Graph::empty();
        let ends = crate::Ends::between(crate::Place::At(base), crate::Place::At(top));
        let cylinder = g.add(Node::Cylinder { edge: Default::default(), ends, radius: ParamRef::Literal(3.0) });
        let cone = g.add(Node::Cone { edge: Default::default(),
            ends,
            base_radius: ParamRef::Literal(5.0),
            top_radius: ParamRef::Literal(2.0),
        });
        let torus = g.add(Node::Torus {
            centre: crate::Place::At(base),
            major_radius: ParamRef::Literal(6.0),
            minor_radius: ParamRef::Literal(1.5),
            turn: no_turn(),
        });
        let capsule = g.add(Node::Capsule {
            ends,
            start_radius: ParamRef::Literal(4.0),
            end_radius: ParamRef::Literal(8.0),
        });

        let back = round_trip(&g);
        for id in [cylinder, cone, torus, capsule] {
            assert_eq!(back.node(id), g.node(id));
        }
        assert_eq!(back.roots().len(), 4, "each is a body of its own");

        // The kind tags, and the pairs that could be swapped without changing the kind: a
        // cone's two radii, and a capsule's two ends.
        let value = serde_json::to_value(g.node(cone).unwrap()).unwrap();
        assert_eq!(value["kind"], "cone");
        assert_eq!(value["base_radius"], serde_json::json!({ "literal": 5.0 }));
        assert_eq!(value["top_radius"], serde_json::json!({ "literal": 2.0 }));
        let value = serde_json::to_value(g.node(capsule).unwrap()).unwrap();
        assert_eq!(value["kind"], "capsule");
        // Through the `Ends` now, and still the same question: the pair is ordered, and a
        // writer that swapped them would be invisible to a round-trip that only counted fields.
        assert_eq!(value["ends"]["at"]["from"][0], serde_json::json!({ "literal": 1.0 }));
        assert_eq!(value["ends"]["at"]["to"][0], serde_json::json!({ "literal": 4.0 }));
        assert_eq!(value["start_radius"], serde_json::json!({ "literal": 4.0 }));
        assert_eq!(value["end_radius"], serde_json::json!({ "literal": 8.0 }));
        let value = serde_json::to_value(g.node(torus).unwrap()).unwrap();
        assert_eq!(value["kind"], "torus");
        assert_eq!(value["major_radius"], serde_json::json!({ "literal": 6.0 }));
        assert_eq!(serde_json::to_value(g.node(cylinder).unwrap()).unwrap()["kind"], "cylinder");
    }

    #[test]
    fn a_placement_round_trips_and_keeps_the_body_it_wraps_off_the_root_list() {
        let param = ParamId::new();
        let mut g = Graph::empty();
        let sphere = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(5.0),
        });
        let placed = g.add(Node::Transform {
            input: sphere.into(),
            translation: [
                ParamRef::Literal(7.0),
                ParamRef::Literal(-3.0),
                ParamRef::Named(param),
            ],
            rotation: [ParamRef::Literal(0.0), ParamRef::Literal(90.0), ParamRef::Literal(0.0)],
            scale: ParamRef::Literal(2.5),
            about: crate::Place::origin(),
        });

        let back = round_trip(&g);
        assert_eq!(back.node(placed), g.node(placed));
        assert_eq!(back.node(sphere), g.node(sphere));

        // The wrapped sphere is consumed, so the placement is the body and the sphere is not.
        // A placement that left both as roots would draw the body twice, once in each place.
        assert_eq!(back.roots(), vec![placed]);

        let value = serde_json::to_value(g.node(placed).unwrap()).unwrap();
        assert_eq!(value["kind"], "transform");
        assert_eq!(value["input"], sphere.to_string());
        assert_eq!(value["scale"], serde_json::json!({ "literal": 2.5 }));
        assert_eq!(value["translation"][2], serde_json::json!({ "named": param.to_string() }));
    }

    /// An attractor names a body *and* a curve, which no other node does — so it is the one
    /// kind where a round trip has two different sorts of reference to lose.
    #[test]
    fn an_attractor_round_trips_with_the_body_and_the_curve_it_names() {
        let param = ParamId::new();
        let curve = CurveId::new();
        let mut g = Graph::empty();
        let ball = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(5.0),
        });
        let pulled = g.add(Node::Attract {
            input: ball.into(),
            toward: Attractor::Curve(curve),
            strength: ParamRef::Literal(0.75),
            falloff: ParamRef::Named(param),
        });

        let back = round_trip(&g);
        assert_eq!(back.node(pulled), g.node(pulled));
        assert_eq!(back.node(ball), g.node(ball));
        // The wrapped body is consumed, so the attractor is the body and the sphere is not —
        // otherwise the part would draw both the pulled shape and the one it was pulled from.
        assert_eq!(back.roots(), vec![pulled]);

        let value = serde_json::to_value(g.node(pulled).unwrap()).unwrap();
        assert_eq!(value["kind"], "attract");
        assert_eq!(value["input"], ball.to_string());
        assert_eq!(value["curve"], curve.to_string());
        assert_eq!(value["strength"], serde_json::json!({ "literal": 0.75 }));
        assert_eq!(value["falloff"], serde_json::json!({ "named": param.to_string() }));
    }

    /// The three walks that have to agree about an attractor's edges, checked together because
    /// a node kind added to one and forgotten in another is the failure they are arranged to
    /// prevent — and it looks like nothing at all until a copy shares a subtree or a driven
    /// parameter evaluates in the wrong order.
    #[test]
    fn an_attractor_reports_the_body_and_the_curve_as_edges() {
        let curve = CurveId::new();
        let mut g = Graph::empty();
        let ball = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(5.0),
        });
        let pulled = g.add(Node::Attract {
            input: ball.into(),
            toward: Attractor::Curve(curve),
            strength: ParamRef::Literal(0.5),
            falloff: ParamRef::Literal(10.0),
        });
        let node = g.node(pulled).unwrap();
        assert_eq!(node.inputs(), vec![ball]);
        assert_eq!(node.curves(), vec![curve]);
        assert_eq!(node.param_refs().len(), 2);
        assert!(node.is_body(), "an attracted body is a body");

        // And both references are redirected on a copy. A node that kept its original input
        // would be a second body quietly sharing one subtree.
        let fresh_body = NodeId::new();
        let mut copy = node.clone();
        copy.remap_inputs(&std::collections::HashMap::from([(ball, fresh_body)]));
        assert_eq!(copy.inputs(), vec![fresh_body]);
        let fresh_curve = CurveId::new();
        for c in copy.curves_mut() {
            *c = fresh_curve;
        }
        assert_eq!(copy.curves(), vec![fresh_curve]);
    }

    /// A build without placements is the one that has to survive opening a file with them, and
    /// the format's answer is to keep the node rather than drop the body it holds.
    #[test]
    fn a_placement_an_older_build_cannot_read_is_kept_verbatim() {
        let json = serde_json::json!({
            "kind": "transform",
            "input": "0000000000000001",
            "translation": [{"literal": 1.0}, {"literal": 0.0}, {"literal": 0.0}],
            "rotation": [{"literal": 0.0}, {"literal": 0.0}, {"literal": 0.0}],
            "scale": {"literal": 1.0},
            "shear": {"literal": 0.2},
        });
        let node: Node = serde_json::from_value(json.clone()).unwrap();
        let Node::Unknown(unknown) = &node else {
            panic!("a transform carrying an unaccounted field should not have been read");
        };
        assert_eq!(unknown.reason(), &Unrecognised::Fields(vec!["shear".to_string()]));
        assert_eq!(serde_json::to_value(&node).unwrap(), json);
    }

    /// Placing a body that is already an operand has to *replace* it in whatever consumed it.
    ///
    /// This is the shape of a real bug: dragging a sphere that was one side of a blend added a
    /// placement beside it rather than in front of it, so the sphere ended up with two
    /// consumers. The part then had a second body drawing an unblended copy, and the shader
    /// generator refused the graph outright — the viewport went blank.
    #[test]
    fn placing_an_operand_puts_the_placement_between_it_and_its_consumer() {
        let mut g = Graph::empty();
        let cage = g.add(Node::Cage(CageId::new()));
        let sphere = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(5.0),
        });
        let blend = g.add(Node::Blend {
            op: BooleanOp::Union,
            a: cage.into(),
            b: sphere.into(),
            fillet: Fillet { radius: ParamRef::Literal(1.0), continuity: Continuity::G1, second: None },
        });
        assert_eq!(g.roots(), vec![blend], "the blend is the only body to begin with");

        let placed = g.add(Node::Transform {
            input: sphere.into(),
            translation: [ParamRef::Literal(3.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
            rotation: [ParamRef::Literal(0.0); 3],
            scale: ParamRef::Literal(1.0),
            about: crate::Place::origin(),
        });
        g.rewire_consumers(sphere, placed);

        // Still one body: the blend, now blending against the placed sphere.
        assert_eq!(g.roots(), vec![blend]);
        let Some(Node::Blend { b, .. }) = g.node(blend) else { panic!("the blend went missing") };
        assert_eq!(*b, placed.into(), "the blend should consume the placement, not the bare sphere");
        // And the placement still holds the sphere, rather than having been redirected to
        // itself.
        let Some(Node::Transform { input, .. }) = g.node(placed) else { panic!("no placement") };
        assert_eq!(*input, sphere.into());
    }

    /// Placing a body that nothing consumes changes nothing about who consumes what — the
    /// wrapper simply becomes the body in its place.
    #[test]
    fn placing_a_body_that_is_already_a_root_leaves_it_the_only_body() {
        let mut g = Graph::empty();
        let sphere = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(5.0),
        });
        let placed = g.add(Node::Transform {
            input: sphere.into(),
            translation: [ParamRef::Literal(0.0); 3],
            rotation: [ParamRef::Literal(0.0); 3],
            scale: ParamRef::Literal(1.0),
            about: crate::Place::origin(),
        });
        g.rewire_consumers(sphere, placed);
        assert_eq!(g.roots(), vec![placed]);
    }

    #[test]
    fn a_node_is_written_as_an_object_tagged_by_kind() {
        // The file's shape is part of what "open format" claims, so it is asserted rather
        // than left to whatever serde's default enum representation happens to be.
        let cage = CageId::new();
        let value = serde_json::to_value(Node::Cage(cage)).unwrap();
        assert_eq!(value["kind"], "cage");
        assert_eq!(value["cage"], cage.to_string());

        let value = serde_json::to_value(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![NodeId::new().into(), NodeId::new().into()],
        })
        .unwrap();
        assert_eq!(value["kind"], "boolean");
        assert_eq!(value["op"], "difference");
    }

    #[test]
    fn a_saved_graph_has_the_same_bytes_every_time() {
        // Teams diff these. Hash order is not stable between runs, so a graph that
        // serialized in map order would show every node as changed at random.
        let mut g = Graph::from_cage(CageId::new());
        for _ in 0..16 {
            g.add(Node::Cage(CageId::new()));
        }
        let once = serde_json::to_string(&g).unwrap();
        for _ in 0..8 {
            assert_eq!(serde_json::to_string(&round_trip(&g)).unwrap(), once);
        }
    }
    /// Removing a body leaves its operands behind, which is why hiding cannot use it alone.
    ///
    /// Pinned here because two callers depend on the opposite halves of it: deleting a block
    /// wants the operands kept — you are undoing an operation and the shapes survive it — and
    /// hiding one does not, since drawing what a hidden body was made of is the one answer
    /// that is certainly wrong. A shelled cube hidden used to draw the solid cube.
    #[test]
    fn removing_a_body_promotes_what_it_consumed_to_bodies() {
        let mut g = Graph::empty();
        let cube = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [ParamRef::Literal(40.0); 3],
            turn: no_turn(),
        });
        let shelled = g.add(Node::Shell { input: cube.into(), thickness: ParamRef::Literal(3.4) });
        assert_eq!(g.roots(), vec![shelled]);

        assert!(g.remove_body(shelled));
        assert_eq!(g.roots(), vec![cube], "the cube it was made from is a body again");
    }

    /// Deleting a ramp leaves everything it drove standing, on a plain number.
    ///
    /// **The acceptance case for `ramp-decision.md` §7.2, and it needed the code to change.** A
    /// ramp is a placeable block now, so it can be selected and deleted while a morph and an
    /// offset are both reading it. `remove_cascading` moves a driver's edge as it moves any
    /// other, so without `release_drivers` it would hand each socket one of its block's
    /// *operands* — a body's field driving an amount, which is refused at compile as
    /// `DrivenParameterNeedsARamp`. The part would go dark on a delete, with the error naming a
    /// node rather than the click.
    ///
    /// Two consumers on purpose: one ramp driving two sockets is the arrangement a ramp living
    /// inside the row it drove could not express, so it is the case worth proving survives.
    #[test]
    fn deleting_a_ramp_leaves_both_of_its_consumers_on_a_plain_number() {
        use crate::{Graph, Node, ParamRef, Place, RampShape};

        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(30.0),
        });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(46.0); 3],
            turn: crate::no_turn(),
        });
        let point = graph.add(Node::Point { at: [lit(0.0); 3] });
        let ramp = graph.add(Node::Ramp {
            input: point.into(),
            in_min: lit(0.0),
            in_max: lit(40.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: RampShape::Smooth,
        });
        // One ramp, two sockets.
        let morph = graph.add(Node::Morph { a: ball.into(), b: block.into(), t: ParamRef::Field(ramp) });
        let shrunk = graph.add(Node::Offset { input: morph.into(), distance: ParamRef::Field(ramp) });

        // A ramp is never a body, however many things read it or none.
        assert!(!graph.roots().contains(&ramp), "a ramp is not a body");

        graph.release_drivers(ramp, 0.5);
        graph.remove_cascading(ramp);

        assert!(graph.node(morph).is_some(), "the morph survived its driver");
        assert!(graph.node(shrunk).is_some(), "the offset survived its driver");
        assert_eq!(
            graph.node(morph).map(|n| n.param_refs().first().map(|r| **r)),
            Some(Some(ParamRef::Literal(0.5))),
            "the morph's amount fell back to a number rather than to another node"
        );
        assert_eq!(
            graph.node(shrunk).map(|n| n.param_refs().first().map(|r| **r)),
            Some(Some(ParamRef::Literal(0.5))),
            "the offset's distance fell back to a number rather than to another node"
        );
        assert!(graph.node(ramp).is_none(), "the ramp itself is gone");
    }

    /// Deleting an operand leaves the blocks that used it standing.
    ///
    /// **The complaint was that it did not.** A consumer used to be rewired to the surviving
    /// operand and then deleted itself, with the recursion carrying on upward — so removing one
    /// sphere took the boolean, and whatever the boolean fed, and so on to the root. For anyone
    /// repurposing a definition rather than building one from scratch, that is the whole file.
    ///
    /// The chain here is three deep on purpose. Collapsing one level was already survivable by
    /// accident; what was not is that the deletion kept climbing, so the test has to look above
    /// the block that lost its input as well as at it.
    #[test]
    fn deleting_an_operand_keeps_the_blocks_that_used_it() {
        use crate::{BooleanOp, Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = |x: f64| Node::Sphere {
            centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        };
        let mut graph = Graph::empty();
        let a = graph.add(ball(0.0));
        let b = graph.add(ball(12.0));
        let c = graph.add(ball(24.0));
        let cut = graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![a.into(), b.into()] });
        let joined = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![cut.into(), c.into()] });
        let shell = graph.add(Node::Shell { input: joined.into(), thickness: lit(2.0) });

        graph.remove_cascading(b);

        // The block that lost an operand is still there, still a difference, with the edge
        // moved onto what it has left.
        let Some(Node::Boolean { op, inputs }) = graph.node(cut) else {
            panic!("the boolean was deleted along with its operand");
        };
        assert_eq!(*op, BooleanOp::Difference, "it changed what it was");
        assert_eq!(
            inputs.iter().map(Operand::source).collect::<Vec<_>>(),
            vec![Some(a), Some(a)],
            "the missing edge should move to what remains",
        );

        // And everything above it, which is the part that used to go.
        assert!(graph.node(joined).is_some(), "the union above it was deleted");
        assert!(graph.node(shell).is_some(), "the shell above that was deleted");
        assert!(graph.node(a).is_some() && graph.node(c).is_some(), "an untouched body went");
        assert!(graph.node(b).is_none(), "the deleted block is still there");
    }

    /// A boolean of three simply drops the operand, rather than collapsing to a survivor.
    ///
    /// "These five, joined" less one is "these four, joined" — the block still means what it
    /// meant. Only a pair has nothing left to be, and that case is the test above.
    #[test]
    fn deleting_one_of_three_operands_leaves_a_boolean_of_two() {
        use crate::{BooleanOp, Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = |x: f64| Node::Sphere {
            centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        };
        let mut graph = Graph::empty();
        let a = graph.add(ball(0.0));
        let b = graph.add(ball(12.0));
        let c = graph.add(ball(24.0));
        let joined = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into(), c.into()] });

        graph.remove_cascading(b);

        let Some(Node::Boolean { inputs, .. }) = graph.node(joined) else {
            panic!("the boolean went with its operand");
        };
        assert_eq!(
            inputs.iter().map(Operand::source).collect::<Vec<_>>(),
            vec![Some(a), Some(c)],
            "the row should have gone, and only the row",
        );
    }

    /// A block with nothing left to act on still goes, and takes only what depends on it.
    ///
    /// The other half of the same rule: an offset of nothing is nothing, so it cannot stay —
    /// but a body standing beside it, sharing no edge, has nothing to do with the deletion.
    #[test]
    fn deleting_the_only_input_removes_what_cannot_stand_without_it() {
        use crate::{Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let grown = graph.add(Node::Offset { input: ball.into(), distance: lit(2.0) });
        let shelled = graph.add(Node::Shell { input: grown.into(), thickness: lit(1.0) });
        let bystander = graph.add(Node::Sphere {
            centre: Place::At([lit(40.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });

        graph.remove_cascading(ball);

        assert!(graph.node(grown).is_none(), "an offset of nothing should not remain");
        assert!(graph.node(shelled).is_none(), "nor a shell of that");
        assert!(graph.node(bystander).is_some(), "an unrelated body was taken too");
    }

    /// A ball above the origin, and a plane through it for a block to be aimed by.
    fn a_ball_and_a_plane() -> (Graph, NodeId, NodeId) {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: crate::Place::At([lit(0.0), lit(0.0), lit(30.0)]),
            radius: lit(10.0),
        });
        let plane = graph.add(Node::Plane {
            origin: crate::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: no_turn(),
        });
        (graph, ball, plane)
    }

    /// Deletes `body` and asserts what the cases below share: `consumer`, made of nothing else,
    /// goes with it; `read`, which it was only aimed at or driven by, stays; and what is left
    /// evaluates, so no edge is left pointing at the deleted block.
    fn assert_goes_with_its_body(mut graph: Graph, body: NodeId, consumer: NodeId, read: NodeId) {
        graph.remove_cascading(body);
        assert!(graph.node(body).is_none(), "the deleted body is still there");
        assert!(
            graph.node(consumer).is_none(),
            "the block outlived its only operand, acting now on what it only read: {:?}",
            graph.node(consumer),
        );
        assert!(graph.node(read).is_some(), "what the block only read was deleted with it");
        assert!(graph.evaluation_order().is_some(), "what is left does not evaluate");
    }

    /// **Deleting the body a mirror reflects takes the mirror, not the plane it is aimed by.**
    ///
    /// The stand-in for a deleted operand used to be the consumer's first other *input*, and the
    /// block a frame is taken from is an input -- so the mirror was handed its own plane and went
    /// on reflecting a half-space (refactor plan §3.3, root cause 1). The plane said where the
    /// mirror stands; it was never something to reflect. A stand-in comes from the operands, a
    /// mirror has one, and with it gone there is nothing left to mirror.
    #[test]
    fn deleting_the_body_a_mirror_reflects_takes_the_mirror_and_leaves_its_plane() {
        let (mut graph, ball, plane) = a_ball_and_a_plane();
        let mirror =
            graph.add(Node::Mirror { input: ball.into(), frame: Frame::From { node: plane }, join: true });
        assert_goes_with_its_body(graph, ball, mirror, plane);
    }

    /// **Deleting the body a twist winds takes the twist, not the plane it winds about.**
    ///
    /// A twist of a plane is a half-space wound about its own normal, which nothing can build --
    /// the delete that left one was the case the check before a delete was added for.
    #[test]
    fn deleting_the_body_a_twist_winds_takes_the_twist_and_leaves_its_plane() {
        let (mut graph, ball, plane) = a_ball_and_a_plane();
        let twist = graph.add(Node::Twist {
            input: ball.into(),
            degrees_per_mm: ParamRef::Literal(2.0),
            frame: Frame::From { node: plane },
        });
        assert_goes_with_its_body(graph, ball, twist, plane);
    }

    /// **Deleting the body a ring repeats takes the ring, not the plane it turns about.**
    ///
    /// A radial pattern's frame is the same `Frame` a mirror's is (D9), so it lost its body the
    /// same way and was handed the same thing: six copies of a half-space.
    #[test]
    fn deleting_the_body_a_ring_repeats_takes_the_ring_and_leaves_its_plane() {
        let (mut graph, ball, plane) = a_ball_and_a_plane();
        let ring = graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(6.0),
            kind: PatternKind::Radial { frame: Frame::From { node: plane } },
            join: false,
            fillet: Fillet::hard(),
        });
        assert_goes_with_its_body(graph, ball, ring, plane);
    }

    /// **Deleting the body a placement moves takes the placement, not the point it pivots on.**
    ///
    /// A pivot taken from a point is a place, and a place is an input, so the placement was
    /// handed the point and went on moving it: a body made of a block that is never one, drawing
    /// nothing where the ball had been.
    #[test]
    fn deleting_the_body_a_placement_moves_takes_the_placement_and_leaves_its_pivot() {
        let (mut graph, ball, _) = a_ball_and_a_plane();
        let lit = ParamRef::Literal;
        let pivot = graph.add(Node::Point { at: [lit(5.0), lit(0.0), lit(0.0)] });
        let placed = graph.add(Node::Transform {
            input: ball.into(),
            translation: [lit(0.0); 3],
            rotation: [lit(0.0), lit(0.0), lit(45.0)],
            scale: lit(1.0),
            about: crate::Place::From { node: pivot },
        });
        assert_goes_with_its_body(graph, ball, placed, pivot);
    }

    /// **Deleting the body an attract warps takes the attract, not the point it pulls toward.**
    ///
    /// The one case here in which what the block reads is listed among its `operands`, because
    /// the point is sampled inside whatever placement wraps the attract and the placement walk
    /// has to reach it. It is still where the body is pulled and never a body to pull, so it
    /// cannot stand in for the one deleted.
    #[test]
    fn deleting_the_body_an_attract_warps_takes_the_attract_and_leaves_its_target() {
        let (mut graph, ball, _) = a_ball_and_a_plane();
        let lit = ParamRef::Literal;
        let target = graph.add(Node::Point { at: [lit(20.0), lit(0.0), lit(30.0)] });
        let pulled = graph.add(Node::Attract {
            input: ball.into(),
            toward: Attractor::Point(target),
            strength: lit(0.5),
            falloff: lit(10.0),
        });
        assert_goes_with_its_body(graph, ball, pulled, target);
    }

    /// **Deleting the body a driven offset grows takes the offset, not the ramp driving it.**
    ///
    /// A driver is an input as well, so the offset was handed its ramp: an offset of a number,
    /// whose zero set is wherever the ramp happens to cross zero -- not a shape anybody asked for,
    /// which is why a ramp is never a body (`Node::is_body`).
    #[test]
    fn deleting_the_body_a_driven_offset_grows_takes_the_offset_and_leaves_its_ramp() {
        let (mut graph, ball, _) = a_ball_and_a_plane();
        let lit = ParamRef::Literal;
        let from = graph.add(Node::Point { at: [lit(0.0); 3] });
        let ramp = graph.add(Node::Ramp {
            input: from.into(),
            in_min: lit(0.0),
            in_max: lit(40.0),
            out_min: lit(0.0),
            out_max: lit(2.0),
            shape: RampShape::Smooth,
        });
        let grown = graph.add(Node::Offset { input: ball.into(), distance: ParamRef::Field(ramp) });
        assert_goes_with_its_body(graph, ball, grown, ramp);
    }

    /// A ramp nothing reads is still evaluable — it is not a cycle.
    ///
    /// The bug this is here for: `evaluation_order` seeded its walk from `roots`, and checked
    /// that it had reached every node, on the reasoning that anything not a body is consumed
    /// and so reachable from whatever consumes it. Excluding ramps from the bodies broke that
    /// reasoning — an unconsumed ramp is neither — and the whole part then reported a cycle,
    /// with an empty block list because a graph reporting a cycle has no roots to list.
    ///
    /// It surfaced as a part that would not evaluate at all, holding the last picture that
    /// had, so the viewport showed geometry the document no longer described.
    #[test]
    fn a_node_that_is_not_a_body_and_that_nothing_reads_is_not_a_cycle() {
        let mut g = Graph::empty();
        let body = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [ParamRef::Literal(40.0); 3],
            turn: no_turn(),
        });
        let plane = g.add(Node::Plane { origin: crate::Place::At([ParamRef::Literal(0.0); 3]), flipped: false,
            turn: no_turn(),
        });
        let ramp = g.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(1.0),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(1.0),
            shape: RampShape::Linear,
        });

        assert_eq!(g.roots(), vec![body], "only the cuboid is a body");
        let order = g.evaluation_order().expect("an unread ramp is not a cycle");
        for id in [body, plane, ramp] {
            assert!(order.contains(&id), "every node should be in the order");
        }
        // And the plane under it comes first, since the ramp reads it.
        let place = |id: NodeId| order.iter().position(|o| *o == id).unwrap();
        assert!(place(plane) < place(ramp));
    }

    /// A ramp and the reference reading it both survive a round trip.
    #[test]
    fn a_ramp_and_the_parameter_it_drives_round_trip() {
        let mut g = Graph::empty();
        let plane = g.add(Node::Plane {
            origin: crate::Place::At([
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(-20.0),
            ]),
            flipped: false,
            turn: no_turn(),
        });
        let ramp = g.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(40.0),
            out_min: ParamRef::Literal(-2.0),
            out_max: ParamRef::Literal(8.0),
            shape: RampShape::Smoother,
        });
        let body = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [ParamRef::Literal(40.0); 3],
            turn: no_turn(),
        });
        let offset = g.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });

        let back = round_trip(&g);
        assert_eq!(back.node(ramp), g.node(ramp));
        assert_eq!(back.node(offset), g.node(offset));
        // And the edge survives with it, which is what keeps the ramp from becoming a body on
        // reopening a file.
        assert_eq!(back.roots(), vec![offset]);
    }

    /// A ramp shape this build has never heard of is preserved and refused, not guessed at.
    ///
    /// The shape decides how fast the ramp can move, and therefore the bound everything above
    /// it divides by. Reading `smoother` as `linear` would produce a body of the right rough
    /// shape whose field claims more room than it has — the exact failure D3's unknown-value
    /// handling exists to prevent.
    #[test]
    fn an_unknown_ramp_shape_is_kept_and_refused() {
        let raw = serde_json::json!({
            "kind": "ramp",
            "input": NodeId::new(),
            "in_min": { "literal": 0.0 },
            "in_max": { "literal": 40.0 },
            "out_min": { "literal": 0.0 },
            "out_max": { "literal": 10.0 },
            "shape": "sigmoid",
        });
        let node: Node = serde_json::from_value(raw.clone()).unwrap();
        let Node::Unknown(unknown) = &node else { panic!("expected an unreadable node: {node:?}") };
        assert_eq!(
            unknown.reason,
            Unrecognised::Value { field: "shape".into(), value: "sigmoid".into() },
        );
        // Written back exactly as it arrived.
        assert_eq!(serde_json::to_value(&node).unwrap(), raw);
    }

    /// A driven reference reads and writes as its own tag, beside a literal and a name.
    #[test]
    fn a_driven_parameter_reference_writes_as_a_field() {
        let id = NodeId::new();
        let r = ParamRef::Field(id);
        let json = serde_json::to_value(r).unwrap();
        assert_eq!(json, serde_json::json!({ "field": id }));
        assert_eq!(serde_json::from_value::<ParamRef>(json).unwrap(), r);
        assert_eq!(r.driver(), Some(id));
        assert_eq!(ParamRef::Literal(3.0).driver(), None);
    }

    /// A driven parameter is an edge, so it decides evaluation order and catches a cycle.
    ///
    /// The whole reason `param_refs` exists: an edge recorded in one place and not the other
    /// gives a graph where a ramp evaluates after the thing that reads it, which is not a bug
    /// with a visible symptom — it is a panic or a stale value depending on the walk.
    #[test]
    fn a_driven_parameter_is_an_edge_for_ordering_and_for_cycles() {
        let mut g = Graph::empty();
        let plane = g.add(Node::Plane { origin: crate::Place::At([ParamRef::Literal(0.0); 3]), flipped: false,
            turn: no_turn(),
        });
        let ramp = g.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(1.0),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(1.0),
            shape: RampShape::Linear,
        });
        let body = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [ParamRef::Literal(40.0); 3],
            turn: no_turn(),
        });
        let offset = g.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });

        assert!(g.node(offset).unwrap().inputs().contains(&ramp));
        let order = g.evaluation_order().expect("no cycle yet");
        let place = |id: NodeId| order.iter().position(|o| *o == id).unwrap();
        assert!(place(ramp) < place(offset), "the ramp has to be evaluated first");

        // And a ramp reading the body it ends up driving is a cycle, caught by the same edge.
        if let Some(Node::Ramp { input, .. }) = g.node_mut(ramp) {
            *input = offset.into();
        }
        assert!(g.evaluation_order().is_none(), "a ramp reading its own consumer is a cycle");
    }

    /// Duplicating a subtree carries the driven edge with it, rather than leaving the copy
    /// pointing at the original's ramp.
    #[test]
    fn duplicating_a_driven_offset_duplicates_what_drives_it() {
        let mut g = Graph::empty();
        let plane = g.add(Node::Plane { origin: crate::Place::At([ParamRef::Literal(0.0); 3]), flipped: false,
            turn: no_turn(),
        });
        let ramp = g.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(1.0),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(1.0),
            shape: RampShape::Linear,
        });
        let body = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [ParamRef::Literal(40.0); 3],
            turn: no_turn(),
        });
        let offset = g.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });

        let copy = g.duplicate_subtree(offset).expect("a body can be duplicated");
        let Some(Node::Offset { distance, .. }) = g.node(copy) else { panic!("a copied offset") };
        let copied_ramp = distance.driver().expect("the copy is still driven");
        assert_ne!(copied_ramp, ramp, "the copy should have its own ramp");
        // And its own plane under that, or moving one would move both.
        let held = g.node(copied_ramp).unwrap().inputs();
        assert_eq!(held.len(), 1);
        assert_ne!(held[0], plane);
    }


    #[test]
    fn a_node_kind_this_build_does_not_know_is_kept_verbatim() {
        // The property the marketplace rests on: someone without the plugin opens the file,
        // saves it, and the plugin's node comes back unchanged.
        let json = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "acme.lattice",
                    "cell": 5.0,
                    "nested": {"anything": [1, 2, 3]}
                }
            },
            "output": "0000000000000001"
        }"#;
        let g: Graph = serde_json::from_str(json).unwrap();
        let unknown = g.node(NodeId::from_raw(2)).unwrap();
        let Node::Unknown(u) = unknown else { panic!("expected the lattice to be unknown") };
        assert_eq!(u.kind(), Some("acme.lattice"));
        assert_eq!(*u.reason(), Unrecognised::Kind);

        let written = serde_json::to_value(&g).unwrap();
        let original: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(
            written["nodes"]["0000000000000002"],
            original["nodes"]["0000000000000002"],
            "the node this build could not read came back exactly as it went in"
        );
    }

    #[test]
    fn a_known_kind_carrying_an_unknown_field_is_kept_and_refused_too() {
        // A newer build adding `falloff` to a blend. Evaluating without it would give a
        // shape that is wrong in a way nothing downstream can detect, so this is preserved
        // as unreadable rather than read as a plain blend.
        let json = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "blend",
                    "a": "0000000000000001",
                    "b": "0000000000000001",
                    "radius": {"literal": 3.0},
                    "falloff": "exponential"
                }
            },
            "output": "0000000000000002"
        }"#;
        let g: Graph = serde_json::from_str(json).unwrap();
        let Node::Unknown(u) = g.node(NodeId::from_raw(2)).unwrap() else {
            panic!("a blend with a field we cannot account for is not a blend we can evaluate")
        };
        assert_eq!(u.kind(), Some("blend"));
        assert_eq!(*u.reason(), Unrecognised::Fields(vec!["falloff".to_string()]));
        assert!(u.to_string().contains("falloff"), "the message names what it could not read");

        let written = serde_json::to_value(&g).unwrap();
        let original: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(written["nodes"], original["nodes"], "including the field it cannot use");
    }

    #[test]
    fn a_continuity_this_build_does_not_know_is_kept_rather_than_failing_the_document() {
        // The case the continuity vocabulary exists to survive: a blend of a continuity a later
        // build has (G3 here — G2 stopped being the example on 2026-10-02, when it was built),
        // opened by this one. Serde's own answer to an unknown enum
        // variant is to fail the whole parse, which would make one newer blend cost the user
        // the entire document.
        let json = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "blend", "op": "union",
                    "a": "0000000000000001", "b": "0000000000000001",
                    "radius": {"literal": 3.0}, "continuity": "g3"
                }
            },
            "output": "0000000000000002"
        }"#;
        let g: Graph = serde_json::from_str(json).expect("a newer continuity still opens");
        let Node::Unknown(u) = g.node(NodeId::from_raw(2)).unwrap() else {
            panic!("a blend whose continuity is unknown cannot be evaluated as a G1 one")
        };
        assert_eq!(
            *u.reason(),
            Unrecognised::Value { field: "continuity".into(), value: "g3".into() }
        );
        assert!(u.to_string().contains("g3"), "the message names the value: {u}");

        // And it comes back out byte-identical, like any other node this build cannot read.
        let written = serde_json::to_value(&g).unwrap();
        let original: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(written["nodes"], original["nodes"]);
    }

    /// G2 is a value this build knows now, and the two places a fillet is written both read it —
    /// a blend, and the fold of a pattern, which had no vocabulary entry for it at all and would
    /// have failed the whole parse on a continuity it did not know.
    #[test]
    fn a_g2_fillet_round_trips_on_a_blend_and_on_a_pattern_and_an_unknown_one_on_a_pattern_is_kept() {
        let blend = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "blend", "op": "union",
                    "a": "0000000000000001", "b": "0000000000000001",
                    "radius": {"literal": 3.0}, "continuity": "g2"
                }
            },
            "output": "0000000000000002"
        }"#;
        let g: Graph = serde_json::from_str(blend).expect("a G2 blend opens");
        match g.node(NodeId::from_raw(2)) {
            Some(Node::Blend { fillet, .. }) => assert_eq!(fillet.continuity, Continuity::G2),
            other => panic!("a G2 blend is a blend this build can evaluate, got {other:?}"),
        }
        let written = serde_json::to_value(&g).unwrap();
        let original: serde_json::Value = serde_json::from_str(blend).unwrap();
        assert_eq!(written["nodes"], original["nodes"]);

        let pattern = |continuity: &str| {
            format!(
                r#"{{
                "nodes": {{
                    "0000000000000001": {{"kind": "cage", "cage": "00000000000000ff"}},
                    "0000000000000002": {{
                        "kind": "pattern", "input": "0000000000000001", "count": 4,
                        "arrangement": "linear",
                        "step": [{{"literal": 10.0}}, {{"literal": 0.0}}, {{"literal": 0.0}}],
                        "join": false, "radius": {{"literal": 2.0}}, "continuity": "{continuity}"
                    }}
                }},
                "output": "0000000000000002"
            }}"#
            )
        };
        let g: Graph = serde_json::from_str(&pattern("g2")).expect("a G2 pattern opens");
        match g.node(NodeId::from_raw(2)) {
            Some(Node::Pattern { fillet, .. }) => assert_eq!(fillet.continuity, Continuity::G2),
            other => panic!("a pattern with a G2 fold is readable, got {other:?}"),
        }
        let written = serde_json::to_value(&g).unwrap();
        let original: serde_json::Value = serde_json::from_str(&pattern("g2")).unwrap();
        assert_eq!(written["nodes"], original["nodes"]);

        let g: Graph = serde_json::from_str(&pattern("g3"))
            .expect("a pattern whose fold has a newer continuity still opens");
        assert!(matches!(g.node(NodeId::from_raw(2)), Some(Node::Unknown(_))));
    }

    /// **A box's or a cylinder's edge round-trips, and a sharp one writes nothing** (3.19) -- the
    /// rims' arrangement: `edge` the reach and, off G1, `edge_continuity`; a newer shape keeps the
    /// node, and a shape with no reach is refused as damaged.
    #[test]
    fn a_box_or_cylinders_edge_round_trips_and_a_sharp_one_writes_nothing() {
        let lit = ParamRef::Literal;
        let origin = crate::Place::At([lit(0.0); 3]);
        let shapes = [
            ("cuboid", Node::Cuboid { centre: origin, size: [lit(4.0), lit(5.0), lit(6.0)], turn: crate::no_turn(), edge: Fillet::hard() }),
            ("cylinder", Node::Cylinder { ends: crate::Ends::between(origin, crate::Place::At([lit(0.0), lit(0.0), lit(9.0)])), radius: lit(3.0), edge: Fillet::hard() }),
        ];
        for (kind, sharp) in shapes {
            let mut graph = Graph::empty();
            let id = graph.add(sharp);
            let key = serde_json::to_value(id).unwrap().as_str().unwrap().to_string();
            let written = serde_json::to_value(&graph).unwrap();
            assert!(written["nodes"][&key].get("edge").is_none(), "{kind}: a sharp edge writes nothing");
            assert!(written["nodes"][&key].get("edge_continuity").is_none(), "{kind}");
            for continuity in [Continuity::G1, Continuity::G2, Continuity::Chamfer] {
                let mut cut = graph.clone();
                match cut.node_mut(id) {
                    Some(Node::Cuboid { edge, .. } | Node::Cylinder { edge, .. }) => {
                        *edge = Fillet { radius: lit(1.5), continuity, second: None }
                    }
                    other => panic!("{kind}: {other:?}"),
                }
                let written = serde_json::to_value(&cut).unwrap();
                let fields = &written["nodes"][&key];
                assert_eq!(fields["edge"], serde_json::json!({"literal": 1.5}), "{kind} {continuity:?}");
                assert_eq!(fields.get("edge_continuity").is_some(), continuity != Continuity::G1, "{kind}: G1 is the default");
                let back: Graph = serde_json::from_value(written.clone()).expect("reads back");
                assert_eq!(serde_json::to_value(&back).unwrap(), written, "{kind} {continuity:?}: byte for byte");
                // A newer shape keeps the node; a shape with no reach is a damaged file.
                let mut newer = written.clone();
                newer["nodes"][&key]["edge_continuity"] = serde_json::json!("g3");
                let newer: Graph = serde_json::from_value(newer).expect("a newer shape still opens");
                assert!(matches!(newer.node(id), Some(Node::Unknown(_))), "{kind}: kept, not guessed");
                let mut damaged = written.clone();
                damaged["nodes"][&key].as_object_mut().unwrap().remove("edge");
                damaged["nodes"][&key]["edge_continuity"] = serde_json::json!("g2");
                assert!(serde_json::from_value::<Graph>(damaged).is_err(), "{kind}: a shape with no reach");
            }
        }
    }

    /// **A profile solid's rims round-trip, and a hard rim writes nothing** -- every extrude,
    /// sweep and loft written before 3.17 is written as it was, and one with rims writes `rims`
    /// and, off G1, `rims_continuity`. A shape this build has not heard of keeps the node rather
    /// than drawing it with a rim it was not; a shape with no reach is refused as damaged.
    #[test]
    fn a_profile_solids_rims_and_their_ends_round_trip_and_a_hard_rim_writes_nothing() {
        let extrude = |rims: &str| {
            format!(
                r#"{{
                "nodes": {{
                    "0000000000000002": {{
                        "kind": "extrude", "profile": "00000000000000aa", "sweep": "normal",
                        "height": {{"literal": 30.0}}, "draft": {{"literal": 0.0}}{rims}
                    }}
                }},
                "output": "0000000000000002"
            }}"#
            )
        };
        let plain: Graph = serde_json::from_str(&extrude("")).expect("an extrude opens");
        let Some(Node::Extrude { rims, .. }) = plain.node(NodeId::from_raw(2)) else { panic!("an extrude") };
        assert_eq!(*rims, crate::Rims::hard(), "absent is hard");
        let written = serde_json::to_value(&plain).unwrap();
        assert!(written["nodes"]["0000000000000002"].get("rims").is_none(), "a hard rim writes nothing");

        use crate::RimEnds;
        for (text, continuity, at) in [
            (r#", "rims": {"literal": 2.0}"#, Continuity::G1, RimEnds::Both),
            (r#", "rims": {"literal": 2.0}, "rims_continuity": "g2""#, Continuity::G2, RimEnds::Both),
            (r#", "rims": {"literal": 2.0}, "rims_continuity": "chamfer""#, Continuity::Chamfer, RimEnds::Both),
            // Which ends, written only when not both (3.18).
            (r#", "rims": {"literal": 2.0}, "rims_at": "start""#, Continuity::G1, RimEnds::Start),
            (r#", "rims": {"literal": 2.0}, "rims_continuity": "g2", "rims_at": "end""#, Continuity::G2, RimEnds::End),
        ] {
            let json = extrude(text);
            let g: Graph = serde_json::from_str(&json).expect("a rimmed extrude opens");
            match g.node(NodeId::from_raw(2)) {
                Some(Node::Extrude { rims, .. }) => {
                    assert_eq!(*rims, crate::Rims { radius: ParamRef::Literal(2.0), continuity, at, second: None }, "{text}")
                }
                other => panic!("{text}: {other:?}"),
            }
            let written = serde_json::to_value(&g).unwrap();
            let original: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(written["nodes"], original["nodes"], "{text}: byte for byte");
        }
        let newer: Graph =
            serde_json::from_str(&extrude(r#", "rims": {"literal": 2.0}, "rims_continuity": "g3""#))
                .expect("a newer shape still opens");
        assert!(matches!(newer.node(NodeId::from_raw(2)), Some(Node::Unknown(_))), "and is kept, not guessed");
        let newer: Graph =
            serde_json::from_str(&extrude(r#", "rims": {"literal": 2.0}, "rims_at": "middle""#))
                .expect("a newer end still opens");
        assert!(matches!(newer.node(NodeId::from_raw(2)), Some(Node::Unknown(_))), "an unknown end is kept too");
        assert!(
            serde_json::from_str::<Graph>(&extrude(r#", "rims_continuity": "g2""#)).is_err(),
            "a shape with no reach is a damaged file"
        );
    }

    /// **A chamfer round-trips on a blend, written as the word a build without it will keep and
    /// refuse** -- `"chamfer"` under the `"continuity"` key (`round-chamfer-spec.md` §2) -- and
    /// is the family at exponent one, with no apex radius to quote.
    #[test]
    fn a_chamfer_round_trips_on_a_blend_and_is_the_family_at_exponent_one() {
        let blend = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "blend", "op": "difference",
                    "a": "0000000000000001", "b": "0000000000000001",
                    "radius": {"literal": 2.0}, "continuity": "chamfer"
                }
            },
            "output": "0000000000000002"
        }"#;
        let g: Graph = serde_json::from_str(blend).expect("a chamfered blend opens");
        match g.node(NodeId::from_raw(2)) {
            Some(Node::Blend { fillet, .. }) => assert_eq!(fillet.continuity, Continuity::Chamfer),
            other => panic!("a chamfered blend is a blend this build can evaluate, got {other:?}"),
        }
        let written = serde_json::to_value(&g).unwrap();
        let original: serde_json::Value = serde_json::from_str(blend).unwrap();
        assert_eq!(written["nodes"], original["nodes"]);
        assert_eq!(Continuity::Chamfer.exponent(), 1.0);
        assert_eq!(Continuity::Chamfer.label(), "Chamfer");
        assert_eq!(Continuity::Chamfer.apex_radius_ratio(), 0.0, "a crease, not a bend");
        assert_eq!(Continuity::ALL, [Continuity::G1, Continuity::G2, Continuity::Chamfer]);
    }

    /// The exponent and the label are what everything downstream reads, so pin them: G2 is the
    /// cubic fold, and the tightest-bend ratio the panel quotes is the table in `the project backlog`.
    #[test]
    fn g2_is_the_cubic_fold_and_its_apex_ratio_is_the_measured_one() {
        assert_eq!(Continuity::G1.exponent(), 2.0);
        assert_eq!(Continuity::G2.exponent(), 3.0);
        assert_eq!((Continuity::G1.label(), Continuity::G2.label()), ("G1", "G2"));
        // `fieldform-field/examples/blend_profile.rs` and `the project backlog`: 1.000 k and 0.561 k.
        assert!((Continuity::G1.apex_radius_ratio() - 1.0).abs() < 1e-12);
        assert!((Continuity::G2.apex_radius_ratio() - 0.5612).abs() < 5e-5);
    }

    #[test]
    fn an_unknown_boolean_operation_is_kept_the_same_way() {
        // The other vocabulary. Nothing plans to grow it, but the rule should not depend on
        // which field happened to be thought about.
        let json = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "boolean", "op": "symmetric_difference",
                    "a": "0000000000000001", "b": "0000000000000001"
                }
            },
            "output": "0000000000000002"
        }"#;
        let g: Graph = serde_json::from_str(json).expect("a newer operation still opens");
        assert!(matches!(g.node(NodeId::from_raw(2)), Some(Node::Unknown(_))));
    }

    /// **A missing field is damaged rather than newer**, so it is an error and not an
    /// [`Node::Unknown`]: there is no future build in which a blend has no radius, so accepting it
    /// would preserve nothing and hide a broken file.
    ///
    /// **Asked of `radius`, where it used to be asked of `b`.** A slot may be absent since 112 —
    /// that is what an empty slot *is* on disk (`notebook-spec.md` §7.2) — so `b` stopped being an
    /// example of this rule and became an example of its one exception. The rule itself is
    /// unchanged and is what the second half pins: every field that is not a slot still has to be
    /// there.
    #[test]
    fn a_node_missing_a_field_its_kind_requires_is_an_error_not_an_unknown() {
        let json = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "blend", "op": "union", "a": "0000000000000001",
                    "b": "0000000000000001", "continuity": "g1"
                }
            },
            "output": "0000000000000002"
        }"#;
        let error = serde_json::from_str::<Graph>(json).unwrap_err().to_string();
        assert!(error.contains("\"radius\""), "names the missing field: {error}");

        // And the exception, beside it so the two cannot drift: a blend with no `b` opens, with
        // that slot empty. It is a block waiting to be told what to read, not a damaged file.
        let emptied = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "blend", "op": "union", "a": "0000000000000001",
                    "radius": {"literal": 2.0}, "continuity": "g1"
                }
            },
            "output": "0000000000000002"
        }"#;
        let g: Graph = serde_json::from_str(emptied).expect("an empty slot is a state, not damage");
        let Some(Node::Blend { a, b, .. }) = g.node(NodeId::from_raw(2)) else {
            panic!("still a blend");
        };
        assert_eq!(a.source(), Some(NodeId::from_raw(1)), "the slot that was named is filled");
        assert!(b.is_empty(), "and the one that was absent is empty");
    }

    /// **A block above an empty slot cannot be evaluated either**, so it is incomplete too.
    ///
    /// The whole reason [`Graph::incomplete`] is a closure and not a list of holed blocks. A walk
    /// that skipped only the block with the hole would hand its consumer an operand that compiled
    /// to nothing, which is the panic
    /// `a_body_with_an_empty_slot_is_refused_by_name_rather_than_panicking` records: a `HashMap`
    /// index on a key the walk had deliberately left out.
    #[test]
    fn a_block_above_an_empty_slot_is_incomplete_too() {
        let mut g = Graph::empty();
        let ball = g.add(Node::Sphere {
            centre: Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(10.0),
        });
        let holed = g.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![ball.into(), Operand::empty()],
        });
        let above = g.add(Node::Shell { input: holed.into(), thickness: ParamRef::Literal(1.0) });
        let higher = g.add(Node::Offset { input: above.into(), distance: ParamRef::Literal(1.0) });

        let incomplete = g.incomplete();
        assert!(incomplete.contains(&holed), "the block with the hole");
        assert!(incomplete.contains(&above), "and the one reading it");
        assert!(incomplete.contains(&higher), "and the one reading that, however far up");
        assert!(!incomplete.contains(&ball), "but not the one below it, which is fine");
    }

    /// The property every later step leans on: skip the set and every block you still reach has
    /// every operand present, so no walk needs a guard of its own.
    #[test]
    fn the_incomplete_set_is_closed_downstream() {
        let mut g = Graph::empty();
        let ball = g.add(Node::Sphere {
            centre: Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(10.0),
        });
        let holed = g.add(Node::Shell { input: Operand::empty(), thickness: ParamRef::Literal(1.0) });
        let both = g.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![ball.into(), holed.into()],
        });
        let incomplete = g.incomplete();
        for (id, node) in g.nodes() {
            if incomplete.contains(&id) {
                continue;
            }
            let mut every = true;
            node.for_each_operand_slot_public(&mut |slot| every &= !slot.is_empty());
            assert!(every, "a block left in has no empty slot of its own");
            for input in node.inputs() {
                assert!(
                    !incomplete.contains(&input),
                    "and reads nothing that was skipped, which is what closed downstream means",
                );
            }
        }
        assert!(incomplete.contains(&both), "the boolean reads the holed shell, so it is out");
        assert!(!incomplete.contains(&ball), "and the sphere is still in");
    }

    /// Ordered, so the same graph always refuses with the same block: a message that names a
    /// different block each time it is asked is a message nobody can act on.
    #[test]
    fn the_first_empty_slot_is_the_same_one_every_time() {
        let mut g = Graph::empty();
        let a = g.add(Node::Shell { input: Operand::empty(), thickness: ParamRef::Literal(1.0) });
        let b = g.add(Node::Offset { input: Operand::empty(), distance: ParamRef::Literal(1.0) });
        let (first, slot) = g.first_empty_operand().expect("there are two");
        assert_eq!(slot, 0, "the slot's index within the block");
        assert_eq!(first, a.min(b), "the lower id, whichever that is, and not a map's order");
        for _ in 0..8 {
            assert_eq!(g.first_empty_operand().map(|(id, _)| id), Some(a.min(b)));
        }
    }

    #[test]
    fn a_graph_whose_output_or_operands_are_missing_is_refused() {
        let missing_output = r#"{
            "nodes": {"0000000000000001": {"kind": "cage", "cage": "00000000000000ff"}},
            "output": "0000000000000009"
        }"#;
        assert!(serde_json::from_str::<Graph>(missing_output).is_err());

        let dangling = r#"{
            "nodes": {
                "0000000000000001": {
                    "kind": "boolean", "op": "union",
                    "inputs": ["0000000000000009", "0000000000000009"]
                }
            },
            "output": "0000000000000001"
        }"#;
        assert!(serde_json::from_str::<Graph>(dangling).is_err());
    }

    #[test]
    fn trivial_graph_is_one_cage_node() {
        let cage = CageId::new();
        let g = Graph::from_cage(cage);
        assert_eq!(g.len(), 1);
        assert_eq!(g.referenced_cages(), vec![cage]);
        assert_eq!(g.evaluation_order().unwrap().len(), 1);
    }

    #[test]
    fn blend_evaluates_inputs_before_the_blend() {
        let (ca, cb) = (CageId::new(), CageId::new());
        let mut g = Graph::from_cage(ca);
        let a = g.roots()[0];
        let b = g.add(Node::Cage(cb));
        let blend = g.add(Node::Blend {
            op: BooleanOp::Union,
            a: a.into(),
            b: b.into(),
            fillet: Fillet { radius: ParamRef::Literal(5.0), continuity: Continuity::G1, second: None },
        });

        let order = g.evaluation_order().unwrap();
        assert_eq!(order.len(), 3);
        assert_eq!(*order.last().unwrap(), blend, "the body evaluates last");
        let pos = |id: NodeId| order.iter().position(|&n| n == id).unwrap();
        assert!(pos(a) < pos(blend) && pos(b) < pos(blend));
    }

    #[test]
    fn a_cycle_is_refused_rather_than_looping_forever() {
        // Unreachable through the UI, but graphs also arrive from scripts and files.
        let mut g = Graph::from_cage(CageId::new());
        let a = g.roots()[0];
        let b = g.add(Node::Cage(CageId::new()));
        let blend = g.add(Node::Blend {
            op: BooleanOp::Union,
            a: a.into(),
            b: b.into(),
            fillet: Fillet { radius: ParamRef::Literal(1.0), continuity: Continuity::G1, second: None },
        });
        // Point a node at its own consumer.
        g.nodes.insert(
            a,
            Node::Blend {
                op: BooleanOp::Union,
                a: blend.into(),
                b: b.into(),
                fillet: Fillet { radius: ParamRef::Literal(1.0), continuity: Continuity::G1, second: None },
            },
        );
        // Every node is now consumed by another, so the graph has no bodies at all — which is
        // the shape a cycle takes here, and it must be refused rather than drawn as nothing.
        assert!(g.roots().is_empty());
        assert!(g.evaluation_order().is_none(), "cycle must be detected");
    }

    #[test]
    fn an_unjoined_pipe_is_a_second_body_rather_than_a_replacement() {
        // The whole point of bodies: building a pipe does not make the cage disappear, and
        // nothing has to be joined before it can be seen.
        let mut g = Graph::from_cage(CageId::new());
        let cage = g.roots()[0];
        let pipe = g.add(Node::Pipe {
            curve: CurveId::new(),
            radius: ParamRef::Literal(2.0),
        });
        assert_eq!(g.roots(), {
            let mut both = vec![cage, pipe];
            both.sort_by_key(|id| id.raw());
            both
        });

        // Joining them is what leaves one body.
        let joined = g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![cage.into(), pipe.into()] });
        assert_eq!(g.roots(), vec![joined]);
    }

    #[test]
    fn removing_an_operation_leaves_the_bodies_it_joined() {
        // Deleting a boolean undoes the operation; it does not throw away what it was applied
        // to. The shapes were bodies before something consumed them and are bodies again.
        let mut g = Graph::from_cage(CageId::new());
        let cage = g.roots()[0];
        let pipe = g.add(Node::Pipe {
            curve: CurveId::new(),
            radius: ParamRef::Literal(2.0),
        });
        let joined = g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![cage.into(), pipe.into()] });
        assert_eq!(g.roots(), vec![joined]);

        assert!(g.remove_body(joined));
        assert_eq!(g.len(), 2, "only the operation went");
        assert_eq!(g.roots(), {
            let mut both = vec![cage, pipe];
            both.sort_by_key(|id| id.raw());
            both
        });
    }

    #[test]
    fn removing_a_leaf_removes_exactly_that_leaf() {
        let mut g = Graph::from_cage(CageId::new());
        let cage = g.roots()[0];
        let pipe = g.add(Node::Pipe {
            curve: CurveId::new(),
            radius: ParamRef::Literal(2.0),
        });
        assert!(g.remove_body(pipe));
        assert_eq!(g.roots(), vec![cage]);
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn only_a_body_can_be_removed() {
        // An interior node has a consumer, and removing it would leave that consumer pointing
        // at nothing — a graph that no longer describes anything.
        let mut g = Graph::from_cage(CageId::new());
        let cage = g.roots()[0];
        let pipe = g.add(Node::Pipe {
            curve: CurveId::new(),
            radius: ParamRef::Literal(2.0),
        });
        let joined = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![cage.into(), pipe.into()] });

        assert!(!g.remove_body(pipe), "the pipe is consumed by the union");
        assert_eq!(g.len(), 3, "and nothing was removed");
        assert!(g.remove_body(joined), "the union itself is a body");
    }

    #[test]
    fn a_format_1_graph_opens_with_its_inert_nodes_as_bodies() {
        // Format 1 named one output and treated everything else as inert. Nothing is dropped
        // on the way in — so the pipe below, which that format would not have drawn, is a body
        // now. That is the correction rather than a regression: the only inert node the app
        // ever wrote was what "Pipe only" left behind, and its disappearing is the bug this
        // model exists to fix.
        let json = r#"{
            "nodes": {
                "0000000000000001": {"kind": "cage", "cage": "00000000000000ff"},
                "0000000000000002": {
                    "kind": "pipe", "curve": "00000000000000ee", "radius": {"literal": 2.0}
                }
            },
            "output": "0000000000000001"
        }"#;
        let g: Graph = serde_json::from_str(json).unwrap();
        assert_eq!(g.len(), 2, "nothing was dropped");
        assert_eq!(g.roots(), vec![NodeId::from_raw(1), NodeId::from_raw(2)]);

        // And what this build writes back has no output at all: bodies are derived.
        let written: serde_json::Value = serde_json::to_value(&g).unwrap();
        assert!(written.get("output").is_none());
    }

    #[test]
    fn a_format_1_output_naming_a_node_that_is_not_there_is_refused() {
        let json = r#"{
            "nodes": {"0000000000000001": {"kind": "cage", "cage": "00000000000000ff"}},
            "output": "0000000000000009"
        }"#;
        assert!(serde_json::from_str::<Graph>(json).is_err());
    }

    #[test]
    fn a_named_parameter_can_drive_a_blend() {
        let param = ParamId::new();
        let r = ParamRef::Named(param);
        assert_eq!(r.resolve(&|id| (id == param).then_some(3.5)), Some(3.5));
        assert_eq!(r.resolve(&|_| None), None, "an unresolved name is not silently zero");
        assert_eq!(ParamRef::Literal(2.0).resolve(&|_| None), Some(2.0));
    }

    #[test]
    fn referenced_cages_deduplicates() {
        // Two nodes reading the same cage is normal — a part blended with a moved copy of
        // itself — and the caller wants each cage once.
        let cage = CageId::new();
        let mut g = Graph::from_cage(cage);
        g.add(Node::Cage(cage));
        assert_eq!(g.referenced_cages(), vec![cage]);
    }

    /// The check that stops a body being placed twice.
    ///
    /// Dragging a body wraps it in a placement. The block list shows the node *inside* that
    /// placement as its own row, so selecting it and dragging again used to add a second
    /// placement around the first — the movement split across a nest of them, and every drag
    /// adding another. This is what a caller asks instead.
    #[test]
    fn a_placed_body_reports_the_placement_that_already_holds_it() {
        let mut g = Graph::empty();
        let sphere = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(1.0),
        });
        assert_eq!(g.enclosing_transform(sphere), None, "nothing holds it yet");

        let placed = g.add(Node::Transform {
            input: sphere.into(),
            translation: [ParamRef::Literal(1.0); 3],
            rotation: [ParamRef::Literal(0.0); 3],
            scale: ParamRef::Literal(1.0),
            about: crate::Place::origin(),
        });
        assert_eq!(g.enclosing_transform(sphere), Some(placed), "the placement is found");
        // And the placement itself is not inside one, so a drag on it moves it rather than
        // looking for another.
        assert_eq!(g.enclosing_transform(placed), None);
    }

    /// A body consumed by something that is not a placement is still unplaced, so dragging it
    /// gets a placement of its own rather than moving the boolean above it.
    #[test]
    fn a_body_inside_a_boolean_is_not_reported_as_placed() {
        let mut g = Graph::empty();
        let a = g.add(Node::Sphere {
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(1.0),
        });
        let b = g.add(Node::Cuboid { edge: Default::default(),
            centre: crate::Place::At([ParamRef::Literal(0.0); 3]),
            size: [ParamRef::Literal(1.0); 3],
            turn: no_turn(),
        });
        g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        assert_eq!(g.enclosing_transform(a), None);
        assert_eq!(g.enclosing_transform(b), None);
    }

    /// **Taking a block out of the slot that reads it empties that slot and leaves the block**
    /// (`notebook-spec.md` §7.2): the holder is left waiting to be told what to read, the block
    /// itself is untouched, and a block the holder never read changes nothing.
    ///
    /// Every slot naming it, which is what the row the gesture is made on stands for: a holder
    /// reading one block twice has one row, so one drag off it is the holder letting go.
    #[test]
    fn a_holder_told_to_stop_reading_a_block_empties_every_slot_that_named_it() {
        use crate::{BooleanOp, Graph, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut g = Graph::empty();
        let ball = g.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(5.0) });
        let box_ = g.add(Node::Sphere { centre: Place::At([lit(20.0), lit(0.0), lit(0.0)]), radius: lit(5.0) });
        let union = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![ball.into(), box_.into()] });

        let mut node = g.node(union).expect("the union").clone();
        assert!(node.stop_reading(ball), "it read the ball, so a slot was emptied");
        assert_eq!(node.operands(), vec![box_], "the other slot is untouched");
        match &node {
            Node::Boolean { inputs, .. } => {
                assert!(inputs[0].is_empty(), "the ball's slot is a hole, not a closed-up row");
                assert_eq!(inputs.len(), 2, "and the row is still there to be filled");
            }
            other => panic!("still a boolean, not {other:?}"),
        }
        assert!(g.node(ball).is_some(), "the block it let go of is still in the graph");

        // A block it never read: nothing to empty, and nothing emptied.
        let lone = g.add(Node::Sphere { centre: Place::At([lit(99.0); 3]), radius: lit(1.0) });
        let before = node.clone();
        assert!(!node.stop_reading(lone), "it never read that one");
        assert_eq!(node, before, "so the node is as it was");

        // Both slots of a holder that reads one block twice, because the list draws it one row.
        let mut twice = Node::Boolean { op: BooleanOp::Union, inputs: vec![ball.into(), ball.into()] };
        assert!(twice.stop_reading(ball));
        assert_eq!(twice.operands(), Vec::new(), "one drag off the one row lets go of both");
    }
