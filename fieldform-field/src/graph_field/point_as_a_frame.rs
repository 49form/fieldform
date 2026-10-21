// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a point as a frame's source: the place it supplies and the facing it leaves upright.
    use super::*;
    use fieldform_scene::{Document, Graph, Node, ParamRef};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// **A point can be a frame's source**, giving the place and leaving the facing upright.
    ///
    /// Asked for directly: draw a point and put another block's origin on it, so moving the
    /// point moves everything aimed at it. Only a plane could answer before, so a point was a
    /// block you could make and not use for the one thing it is for.
    ///
    /// The facing is `+Z` because a point genuinely has none — the same answer a zero axis
    /// already gets, rather than a second convention for the same absence.
    #[test]
    fn a_point_supplies_a_frames_place_and_leaves_its_facing_upright() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let point = graph.add(Node::Point { at: [lit(3.0), lit(-4.0), lit(5.0)] });
        let numbers = |r: &ParamRef| r.resolve(&|_| None);

        let (origin, rotation) = resolve_frame(
            &doc,
            &graph,
            &fieldform_scene::Frame::From { node: point },
            &numbers,
        )
        .expect("a point can supply a frame");
        assert!((origin - nalgebra::Vector3::new(3.0, -4.0, 5.0)).norm() < 1e-9);
        assert!(
            (rotation * nalgebra::Vector3::z() - nalgebra::Vector3::z()).norm() < 1e-9,
            "a point has no direction, so the frame stays upright"
        );

        // And something that is neither a place nor a facing is still refused by name.
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(1.0) });
        assert!(matches!(
            resolve_frame(&doc, &graph, &fieldform_scene::Frame::From { node: ball }, &numbers),
            Err(CompileError::NotAFrameSource(_))
        ));
    }
    /// **A plane can sit on a point**, which is what a place taken from a block is for.
    ///
    /// Asked for directly: draw a point and put a plane's origin on it, so moving the point
    /// moves the plane. Before this a centre was three loose numbers and the only way to put two
    /// blocks in one spot was to type the spot twice — which is two descriptions of one fact,
    /// and they part company the first time either is edited.
    #[test]
    fn a_plane_sits_where_the_point_it_is_placed_on_is() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let point = graph.add(Node::Point { at: [lit(2.0), lit(0.0), lit(7.0)] });
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::From { node: point },
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let numbers = |r: &ParamRef| r.resolve(&|_| None);

        let at = resolve_place(&doc, &graph, &fieldform_scene::Place::From { node: plane }, &numbers)
            .expect("a plane is a place too");
        assert!((at - nalgebra::Point3::new(2.0, 0.0, 7.0)).norm() < 1e-9);

        // And the field agrees: the half-space starts at the point's height, not at zero.
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        assert!(field.distance(nalgebra::Point3::new(0.0, 0.0, 6.0)) < 0.0, "below is solid");
        assert!(field.distance(nalgebra::Point3::new(0.0, 0.0, 8.0)) > 0.0, "above is air");

        // The edge is a graph edge, so the point is an operand and no longer a body of its own.
        assert!(graph.node(plane).expect("there").inputs().contains(&point));
        assert!(!graph.roots().contains(&point), "a point used as a place is not a body");
    }

    /// **A body placed on a point is not made of it**, which is what deleting the point must
    /// respect.
    ///
    /// Reported as a plane disappearing when the point under it was deleted — and it was not
    /// the plane, it was every body that had been placed on that point, removed with it. A place
    /// taken from a block is an *input*, so it evaluates first and is not a body of its own, and
    /// the delete cascade treated it like a boolean's operand: something the consumer could not
    /// exist without.
    ///
    /// The two are different. A boolean without an operand has nothing left to be. A sphere
    /// sitting on a point is a sphere — the point said where, and removing it should leave the
    /// sphere where it was told. This asserts the graph shape that makes the distinction
    /// possible: the edge exists, so the cascade *would* reach it, which is why detaching has to
    /// happen before the cascade rather than being left to it.
    #[test]
    fn a_place_is_an_input_but_not_the_kind_a_cascade_may_follow() {
        let mut graph = Graph::empty();
        let point = graph.add(Node::Point { at: [lit(1.0), lit(2.0), lit(3.0)] });
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::From { node: point },
            radius: lit(5.0),
        });

        // The edge is real: that is what keeps the point out of the body list and evaluates it
        // first, and it is also what would carry a cascade into the sphere.
        assert!(graph.node(ball).expect("there").inputs().contains(&point));
        assert!(!graph.roots().contains(&point));

        // And it is reachable *only* through the place, so detaching the place removes the
        // edge entirely -- there is no second reason the sphere depends on the point.
        let mut detached = graph.clone();
        if let Some(node) = detached.node_mut(ball)
            && let Some(slot) = node.places_mut().first_mut()
        {
            **slot = fieldform_scene::Place::at(1.0, 2.0, 3.0);
        }
        assert!(
            !detached.node(ball).expect("there").inputs().contains(&point),
            "with the place typed, nothing links the sphere to the point"
        );
        assert!(
            !detached.is_consumed(point),
            "and nothing reads the point, so deleting it takes nothing with it"
        );
        assert!(!detached.roots().contains(&point), "a point is never a body, read or not");
    }

    /// **A block placed on a point lands on the point, whatever is wrapped around it.**
    ///
    /// Reported as "putting a plane on a point does not work — the origin is off in space", and
    /// it was: a block's own coordinates are read inside whatever placements wrap it, so handing
    /// one block's numbers to another mixed two frames. Measured before the fix — a plane on a
    /// point at `z = 7` with a placement of `+20` above it put its surface at `z = 27`. Every
    /// plane created while drawing on a plane carries such a placement, so this was most of them.
    #[test]
    fn a_placement_above_a_block_does_not_move_it_off_the_point_it_sits_on() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let point = graph.add(Node::Point { at: [lit(0.0), lit(0.0), lit(7.0)] });
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::From { node: point },
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        // The wrapper a plane gets when it is made while drawing on one.
        graph.add(Node::Transform {
            input: plane.into(),
            translation: [lit(0.0), lit(0.0), lit(20.0)],
            rotation: [lit(0.0); 3],
            scale: lit(1.0),
            about: fieldform_scene::Place::origin(),
        });

        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        // The half-space starts at the point, so just below is solid and just above is air.
        assert!(field.distance(Point3::new(0.0, 0.0, 6.0)) < 0.0, "below the point is solid");
        assert!(field.distance(Point3::new(0.0, 0.0, 8.0)) > 0.0, "above it is air");
        // And emphatically not twenty higher, which is where the placement used to put it.
        assert!(
            field.distance(Point3::new(0.0, 0.0, 26.0)) > 0.0,
            "the placement must not be added to the point's own position"
        );
    }

    /// Something that is neither a point nor a plane cannot supply a place, and says so.
    #[test]
    fn only_a_point_or_a_plane_can_supply_a_place() {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(1.0) });
        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        assert!(matches!(
            resolve_place(&Document::new(), &graph, &fieldform_scene::Place::From { node: ball }, &numbers),
            Err(CompileError::NotAPlaceSource(_))
        ));
    }

