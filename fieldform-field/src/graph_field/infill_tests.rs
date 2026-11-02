// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of the infill block: that it is the shape the five blocks it replaces make, and that each
//! way of getting it wrong that would still look right is held shut.
    use super::*;
    use crate::{Normalization, ScalarField};
    use fieldform_scene::{
        BooleanOp, CellMap, Fillet, Frame, Graph, LatticeCell, LatticePhase, Node, Operand,
        ParamRef, Place,
    };
    use nalgebra::Point3;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn cube(graph: &mut Graph, side: f64) -> NodeId {
        graph.add(Node::Cuboid {
            edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(side); 3],
            turn: fieldform_scene::no_turn(),
        })
    }

    fn ball(graph: &mut Graph, radius: f64) -> NodeId {
        graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(radius) })
    }

    fn cage(doc: &mut Document, graph: &mut Graph, side: f64) -> NodeId {
        let id = fieldform_scene::CageId::new();
        doc.cages.insert(id, fieldform_cage::Cage::cube_of(side).into());
        graph.add(Node::Cage(id))
    }

    /// Two cubes run together with a fillet, so the body's own field is only a bound.
    fn blended(graph: &mut Graph) -> NodeId {
        let a = cube(graph, 30.0);
        let b = ball(graph, 22.0);
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            inputs: vec![a.into(), b.into()],
            fillet: Fillet::of(lit(4.0)),
        })
    }

    fn lattice(graph: &mut Graph, cell: LatticeCell, phase: LatticePhase) -> NodeId {
        graph.add(Node::Lattice {
            frame: Frame::default(),
            cell,
            map: CellMap::Rectangular { cell: lit(10.0) },
            phase,
        })
    }

    fn gyroid(graph: &mut Graph) -> NodeId {
        lattice(graph, LatticeCell::Gyroid, LatticePhase::sheet(lit(1.0)))
    }

    fn infill(graph: &mut Graph, body: NodeId, wall: f64, fill: NodeId, join: Fillet) -> NodeId {
        graph.add(Node::Infill {
            input: body.into(),
            wall: lit(wall),
            fill: fill.into(),
            join,
        })
    }

    /// The construction this block replaces, spelt out block by block.
    fn five_blocks(graph: &mut Graph, body: NodeId, wall: f64, fill: NodeId) -> NodeId {
        let skin = graph.add(Node::Shell { input: body.into(), thickness: lit(wall) });
        let inner = graph.add(Node::Offset { input: body.into(), distance: lit(-wall) });
        let core = graph.add(Node::Boolean {
            op: BooleanOp::Intersection,
            inputs: vec![inner.into(), fill.into()],
        });
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![skin.into(), core.into()] })
    }

    /// One block of `graph` as a field of its own.
    fn field_of(doc: &Document, graph: &Graph, node: NodeId) -> GraphField {
        let alone = graph.subtree(node).expect("the block is in the graph");
        GraphField::compile_graph(doc, &alone, 0.25).expect("the block compiles")
    }

    /// Points spread evenly through a cube of `half` each way, the same ones every run.
    fn points(half: f64, count: u32) -> impl Iterator<Item = Point3<f64>> {
        let radical = |mut i: u32, base: u32| {
            let (mut value, mut unit) = (0.0, 1.0 / f64::from(base));
            while i > 0 {
                value += f64::from(i % base) * unit;
                i /= base;
                unit /= f64::from(base);
            }
            value
        };
        (1..=count).map(move |i| {
            let at = |base| (radical(i, base) * 2.0 - 1.0) * half;
            Point3::new(at(2), at(3), at(5))
        })
    }

    /// **T1.** Across a cube, a sphere, a cage and a blended body, a sheet, a solid and a strut.
    /// Sign, not value: the five blocks and the one are the same solid and not the same number.
    #[test]
    fn an_infill_is_the_five_blocks_it_replaces() {
        type Body = (&'static str, fn(&mut Document, &mut Graph) -> NodeId);
        let bodies: [Body; 4] = [
            ("cube", |_, g| cube(g, 40.0)),
            ("sphere", |_, g| ball(g, 22.0)),
            ("cage", |d, g| cage(d, g, 40.0)),
            ("blend", |_, g| blended(g)),
        ];
        type Fill = (&'static str, fn(&mut Graph) -> NodeId);
        let fills: [Fill; 3] = [
            ("sheet", gyroid),
            ("solid", |g| {
                lattice(g, LatticeCell::SchwarzP, LatticePhase::Solid { density: lit(0.3) })
            }),
            ("strut", |g| {
                lattice(g, LatticeCell::Bcc, LatticePhase::Strut { radius: lit(0.8) })
            }),
        ];
        for (body_name, make_body) in bodies {
            for (fill_name, make_fill) in fills {
                let mut doc = Document::new();
                let mut graph = Graph::empty();
                let body = make_body(&mut doc, &mut graph);
                let fill = make_fill(&mut graph);
                let one = infill(&mut graph, body, 2.0, fill, Fillet::hard());
                let five = five_blocks(&mut graph, body, 2.0, fill);
                let (one, five) = (field_of(&doc, &graph, one), field_of(&doc, &graph, five));

                let mut material = 0;
                for p in points(26.0, 100_000) {
                    let (a, b) = (one.distance(p), five.distance(p));
                    // On the surface either side may round to either sign.
                    if a.abs() < 1e-6 || b.abs() < 1e-6 {
                        continue;
                    }
                    assert_eq!(
                        a < 0.0,
                        b < 0.0,
                        "{fill_name} in a {body_name} at {p:?}: {a} against {b}",
                    );
                    material += usize::from(a < 0.0);
                }
                assert!(material > 1000, "{fill_name} in a {body_name}: only {material} inside");
            }
        }
    }

    /// **T2.** The reason the outer `max` comes last, out to a join three times the wall.
    #[test]
    fn no_join_can_grow_the_part() {
        for radius in [0.5, 2.0, 6.0] {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let body = cube(&mut graph, 40.0);
            let fill = gyroid(&mut graph);
            let part = infill(&mut graph, body, 2.0, fill, Fillet::of(lit(radius)));
            let (outside, field) = (field_of(&doc, &graph, body), field_of(&doc, &graph, part));
            for p in points(26.0, 100_000) {
                if outside.distance(p) > 1e-6 {
                    assert!(
                        field.distance(p) > 0.0,
                        "a join of {radius} put material outside the body at {p:?}",
                    );
                }
            }
            // Every face reads the body's own surface, skin present, whatever is behind it.
            for q in points(19.0, 2_000) {
                let on_a_face = Point3::new(20.0, q.y, q.z);
                assert!(
                    field.distance(on_a_face).abs() < 1e-9,
                    "the face moved by {} at {on_a_face:?}",
                    field.distance(on_a_face),
                );
            }
            assert_eq!(
                graph_extent(&doc, &graph, part).as_box(),
                graph_extent(&doc, &graph, body).as_box(),
                "and the box is the body's",
            );
        }
    }

    /// **The instrument above can see the fault it is there for.** The same join, applied last
    /// as a union of the two parts, puts material outside the body. If this stops failing the
    /// test above proves nothing.
    #[test]
    fn a_join_applied_after_the_clip_does_grow_it() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 40.0);
        let fill = gyroid(&mut graph);
        let skin = graph.add(Node::Shell { input: body.into(), thickness: lit(2.0) });
        let inner = graph.add(Node::Offset { input: body.into(), distance: lit(-2.0) });
        let core = graph.add(Node::Boolean {
            op: BooleanOp::Intersection,
            inputs: vec![inner.into(), fill.into()],
        });
        let wrong = graph.add(Node::Blend {
            op: BooleanOp::Union,
            inputs: vec![skin.into(), core.into()],
            fillet: Fillet::of(lit(6.0)),
        });
        let (outside, field) = (field_of(&doc, &graph, body), field_of(&doc, &graph, wrong));
        assert!(
            points(26.0, 100_000).any(|p| outside.distance(p) > 1e-6 && field.distance(p) < 0.0),
            "the wrong order stopped growing the part, so this no longer tests the right one",
        );
    }

    /// The join only ever adds material, and only where a lattice wall meets the skin: reading
    /// `smin`, it differs from `min` only where both terms are within the radius.
    #[test]
    fn a_join_only_acts_where_the_fill_meets_the_skin() {
        let k = 1.5;
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 40.0);
        let fill = gyroid(&mut graph);
        let hard = infill(&mut graph, body, 2.0, fill, Fillet::hard());
        let joined = infill(&mut graph, body, 2.0, fill, Fillet::of(lit(k)));
        let f = field_of(&doc, &graph, body);
        let l = field_of(&doc, &graph, fill);
        let (hard, joined) = (field_of(&doc, &graph, hard), field_of(&doc, &graph, joined));

        let mut grew = 0;
        for p in points(21.0, 100_000) {
            let (was, now) = (hard.distance(p) < 0.0, joined.distance(p) < 0.0);
            assert!(now || !was, "a join took material away at {p:?}");
            if was != now {
                grew += 1;
                let (body_at, fill_at) = (f.distance(p), l.distance(p));
                assert!(
                    -body_at - 2.0 < k && fill_at < k * 2.0,
                    "a join reached a point {} from the skin and {fill_at} from the fill: {p:?}",
                    -body_at - 2.0,
                );
            }
        }
        assert!(grew > 0, "the join did nothing at all");
    }

    /// **T3.** Exactly the intersection, to the number and not only the sign, because the sign is
    /// the part the formula gets right.
    #[test]
    fn a_wall_of_nought_is_plain_infill_and_ignores_its_join() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = ball(&mut graph, 22.0);
        let fill = gyroid(&mut graph);
        let plain = graph.add(Node::Boolean {
            op: BooleanOp::Intersection,
            inputs: vec![body.into(), fill.into()],
        });
        let bare = infill(&mut graph, body, 0.0, fill, Fillet::hard());
        let joined = infill(&mut graph, body, 0.0, fill, Fillet::of(lit(3.0)));
        let plain = field_of(&doc, &graph, plain);
        let bare = field_of(&doc, &graph, bare);
        let joined = field_of(&doc, &graph, joined);
        for p in points(26.0, 100_000) {
            assert_eq!(plain.distance(p), bare.distance(p), "at {p:?}");
            assert_eq!(plain.distance(p), joined.distance(p), "with a join, at {p:?}");
        }
    }

    /// **The reason a wall of nought does not run through the formula.** Read through it the
    /// field is `|F|` wherever the lattice is not: it touches nought on the outer surface and
    /// comes back up, a skin with no thickness. The mesher is not troubled and a ray is, which
    /// drew every lattice face behind that surface in contour lines.
    #[test]
    fn a_wall_of_nought_leaves_no_skin_for_a_ray_to_stop_on() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 40.0);
        let fill = gyroid(&mut graph);
        let part = infill(&mut graph, body, 0.0, fill, Fillet::hard());
        let (l, field) = (field_of(&doc, &graph, fill), field_of(&doc, &graph, part));
        let mut open = 0;
        for q in points(19.0, 2_000) {
            let on_a_face = Point3::new(20.0, q.y, q.z);
            let reads = l.distance(on_a_face);
            if reads > 0.1 {
                open += 1;
                assert!(
                    field.distance(on_a_face) >= reads - 1e-9,
                    "the outer surface reads {} over open lattice at {on_a_face:?}",
                    field.distance(on_a_face),
                );
            }
        }
        assert!(open > 100, "the face was almost all lattice, so this tested nothing");
    }

    /// **T4.** On a body whose field only bounds its distance, the wall is never thinner than
    /// typed. Walked from the real surface, found by bisection, because reading the field would
    /// measure the field against itself.
    #[test]
    fn a_wall_on_a_bound_is_never_thinner_than_typed() {
        let wall = 2.0;
        let doc = Document::new();
        let mut graph = Graph::empty();
        let a = ball(&mut graph, 18.0);
        let b = cube(&mut graph, 30.0);
        let body = graph.add(Node::Morph { a: a.into(), b: b.into(), t: lit(0.5) });
        assert!(!carries_true_distance(&doc, &graph, body), "this has to be a bound to test one");
        let fill = lattice(&mut graph, LatticeCell::Bcc, LatticePhase::Strut { radius: lit(0.3) });
        let part = infill(&mut graph, body, wall, fill, Fillet::hard());
        let (f, field) = (field_of(&doc, &graph, body), field_of(&doc, &graph, part));

        for direction in points(1.0, 2_000) {
            let Some(out) = direction.coords.try_normalize(1e-6) else { continue };
            let (mut inside, mut outside) = (0.0, 60.0);
            for _ in 0..60 {
                let middle: f64 = (inside + outside) / 2.0;
                if f.distance(Point3::from(out * middle)) < 0.0 {
                    inside = middle;
                } else {
                    outside = middle;
                }
            }
            for depth in [0.1, 0.5, 1.0, 1.5, 1.9] {
                let p = Point3::from(out * (inside - depth));
                assert!(
                    field.distance(p) < 0.0,
                    "a point {depth} in from the surface is not wall: {p:?}",
                );
            }
        }
    }

    /// **T5.** The wall meets in the middle and the answer is the plain solid, with nothing wrong:
    /// a solid part is a valid part.
    #[test]
    fn a_wall_past_the_deepest_point_leaves_the_plain_solid() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 20.0);
        let fill = gyroid(&mut graph);
        let full = infill(&mut graph, body, 10.5, fill, Fillet::hard());
        let hollow = infill(&mut graph, body, 9.0, fill, Fillet::hard());
        let f = field_of(&doc, &graph, body);
        let (full, hollow) = (field_of(&doc, &graph, full), field_of(&doc, &graph, hollow));
        for p in points(12.0, 50_000) {
            let inside = f.distance(p) < -1e-6;
            assert_eq!(full.distance(p) < 0.0, inside, "at {p:?}");
        }
        // And just short of the middle it is not solid: the same fill finds a void there.
        assert!(
            points(1.0, 10_000).any(|p| hollow.distance(p) > 0.0),
            "a nine millimetre wall on a twenty millimetre cube left no cavity",
        );
    }

    #[test]
    fn a_wall_below_nought_is_refused_by_name() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 20.0);
        let fill = gyroid(&mut graph);
        infill(&mut graph, body, -1.0, fill, Fillet::hard());
        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3),
            Err(CompileError::InfillWallBelowNought(t)) if t == -1.0,
        ));
    }

    /// A shell refuses this and an infill must not differ: two tools hollowing one body two ways
    /// is a bug report waiting to happen.
    #[test]
    fn a_body_with_no_bounds_is_refused_as_a_shell_refuses_one() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let fill = gyroid(&mut graph);
        infill(&mut graph, plane, 2.0, fill, Fillet::hard());
        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3),
            Err(CompileError::InfillNeedsBounds),
        ));
    }

    #[test]
    fn a_wall_that_follows_a_ramp_is_refused_by_name() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 20.0);
        let fill = gyroid(&mut graph);
        let reads = ball(&mut graph, 1.0);
        let ramp = graph.add(Node::Ramp {
            input: reads.into(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(1.0),
            out_max: lit(2.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        graph.add(Node::Infill {
            input: body.into(),
            wall: ParamRef::Field(ramp),
            fill: fill.into(),
            join: Fillet::hard(),
        });
        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3),
            Err(CompileError::ParameterCannotBeDriven("an infill's wall")),
        ));
    }

    /// An empty slot is a legal state, and the block is named rather than the compile falling
    /// over or a lowered piece of it being blamed.
    #[test]
    fn an_empty_slot_names_the_block_and_nothing_inside_it() {
        let doc = Document::new();
        for empty_fill in [false, true] {
            let mut graph = Graph::empty();
            let body = cube(&mut graph, 20.0);
            let fill = gyroid(&mut graph);
            let part = graph.add(Node::Infill {
                input: if empty_fill { body.into() } else { Operand::empty() },
                wall: lit(2.0),
                fill: if empty_fill { Operand::empty() } else { fill.into() },
                join: Fillet::hard(),
            });
            assert_eq!(with_infills_lowered(&doc, &graph).unwrap(), None, "nothing to lower");
            match GraphField::compile_graph(&doc, &graph, 1e-3) {
                Err(CompileError::EmptySlot { node, .. }) => assert_eq!(node, part),
                other => panic!("expected the block named, got {:?}", other.map(|_| ())),
            }
        }
    }

    /// A named wall is asked of the document, and a new value is a new wall.
    #[test]
    fn a_wall_named_by_a_parameter_follows_it() {
        let mut doc = Document::new();
        let wall = doc.add_parameter("wall", 2.0);
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 40.0);
        let fill = ball(&mut graph, 0.5);
        graph.add(Node::Infill {
            input: body.into(),
            wall: ParamRef::Named(wall),
            fill: fill.into(),
            join: Fillet::hard(),
        });
        let in_the_wall_or_not = |doc: &Document| {
            let field = GraphField::compile_graph(doc, &graph, 1e-3).unwrap();
            field.distance(Point3::new(17.0, 0.0, 0.0)) < 0.0
        };
        assert!(!in_the_wall_or_not(&doc), "three in from the face, behind a two millimetre skin");
        assert!(doc.set_parameter_literal(wall, 4.0));
        assert!(in_the_wall_or_not(&doc), "and inside a four millimetre one");
    }

    /// Any body fills, since the arithmetic does not care, and the result of one can be filled
    /// again.
    #[test]
    fn any_body_is_a_fill_and_an_infill_is_a_body() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 40.0);
        let rib = ball(&mut graph, 5.0);
        let first = infill(&mut graph, body, 2.0, rib, Fillet::hard());
        let core = gyroid(&mut graph);
        let second = infill(&mut graph, first, 1.0, core, Fillet::hard());
        assert_eq!(graph.roots(), vec![second], "both inputs consumed, the lattice with them");
        let field = field_of(&doc, &graph, second);
        assert!(field.distance(Point3::new(0.0, 0.0, 0.0)) < 0.0, "the rib is material");
        assert!(field.distance(Point3::new(19.5, 0.0, 0.0)) < 0.0, "and the skin is");
    }

    /// **The panel asks without compiling and the two must not part**, as
    /// `the_graph_walk_agrees_with_the_compiler_about_true_distance` holds the walk to the
    /// compiler for everything else: same claim, same rate, and approximate exactly when a join
    /// rounds over a bound.
    #[test]
    fn an_infill_claims_what_its_lowering_claims() {
        type Fill = fn(&mut Graph) -> NodeId;
        let exact: Fill = |g| lattice(g, LatticeCell::Bcc, LatticePhase::Strut { radius: lit(0.8) });
        let bound: Fill = gyroid;
        for (fill, fill_is_exact) in [(exact, true), (bound, false)] {
            for (wall, radius) in [(2.0, 0.0), (2.0, 1.0), (0.0, 0.0), (0.0, 1.0)] {
                let doc = Document::new();
                let mut graph = Graph::empty();
                let body = cube(&mut graph, 40.0);
                let filler = fill(&mut graph);
                let join = if radius > 0.0 { Fillet::of(lit(radius)) } else { Fillet::hard() };
                let part = infill(&mut graph, body, wall, filler, join);

                let field = field_of(&doc, &graph, part);
                assert_eq!(
                    carries_true_distance(&doc, &graph, part),
                    field.normalization() == Normalization::Distance,
                    "the walk and the compiler part at wall {wall}, join {radius}, exact fill {fill_is_exact}",
                );
                assert!(
                    (graph_lipschitz(&doc, &graph, part) - field.lipschitz()).abs() < 1e-9,
                    "the rate at wall {wall}, join {radius}: {} against {}",
                    graph_lipschitz(&doc, &graph, part),
                    field.lipschitz(),
                );
                assert_eq!(
                    approximate_blends(&doc, &graph).contains(&part),
                    wall > 0.0 && radius > 0.0 && !fill_is_exact,
                    "the announcement at wall {wall}, join {radius}, exact fill {fill_is_exact}",
                );
            }
        }
    }

    /// The lowering leaves the block's id alone, because everything keyed on a body finds it by
    /// that, and leaves no block behind that has become a body by being rewritten around.
    #[test]
    fn lowering_keeps_the_bodies_they_were() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 40.0);
        let fill = gyroid(&mut graph);
        let part = infill(&mut graph, body, 2.0, fill, Fillet::of(lit(1.0)));
        let lowered = with_infills_lowered(&doc, &graph).unwrap().expect("there is one to lower");
        assert_eq!(lowered.roots(), vec![part]);
        assert!(matches!(lowered.node(part), Some(Node::Boolean { .. })));
    }

    /// **The figure the panel offers as a number to type**, so it has to be the wall at which
    /// the cavity closes, and not a sample's spacing short of it.
    #[test]
    fn the_depth_reported_is_where_the_cavity_closes() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 20.0);
        let fill = gyroid(&mut graph);
        let part = infill(&mut graph, body, 2.0, fill, Fillet::hard());
        let cavity = infill_cavity(&doc, &graph, part, 20_000).expect("a body to measure");
        assert!((cavity.deepest - 10.0).abs() < 0.01, "read {}", cavity.deepest);
        assert!(cavity.has_room(9.99), "just under it leaves a cavity");
        assert!(!cavity.has_room(10.0), "and that wall itself does not, since it closes there");
    }

    #[test]
    fn a_fill_that_misses_the_cavity_is_reported() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 20.0);
        let far = graph.add(Node::Sphere {
            centre: Place::At([lit(100.0), lit(0.0), lit(0.0)]),
            radius: lit(1.0),
        });
        let empty = infill(&mut graph, body, 2.0, far, Fillet::hard());
        let cavity = infill_cavity(&doc, &graph, empty, 20_000).unwrap();
        assert!(!cavity.fill_lands, "the fill is eighty millimetres away");
        // 20^3 - 16^3 of 20^3.
        assert!((cavity.solid_share - 0.488).abs() < 0.01, "read {}", cavity.solid_share);
    }

    /// A thin sheet can fall between every sample. The report must not call that missing, since
    /// it would be saying so about a lattice that is on screen.
    #[test]
    fn a_thin_sheet_between_the_samples_is_not_reported_missing() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = cube(&mut graph, 40.0);
        let sheet = lattice(&mut graph, LatticeCell::Gyroid, LatticePhase::sheet(lit(0.01)));
        let part = infill(&mut graph, body, 2.0, sheet, Fillet::hard());
        let cavity = infill_cavity(&doc, &graph, part, 2_000).unwrap();
        assert!(cavity.fill_lands);
    }

    #[test]
    fn a_graph_with_no_infill_clones_nothing() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        cube(&mut graph, 40.0);
        assert_eq!(with_infills_lowered(&doc, &graph).unwrap(), None);
    }
