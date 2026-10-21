// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a profile solid's rims -- rounded or chamfered where the side meets each end --
//! through the whole graph. `round-chamfer-spec.md` §4 and §6.
    use super::*;
    use fieldform_scene::{
        Continuity, Curve, CurveId, Document, Rims, Graph, Node, Outline, ParamRef, Sides, Sweep,
    };
    use nalgebra::{Point3, Vector3};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// A square forty across, drawn on the ground about the origin.
    fn square(doc: &mut Document) -> CurveId {
        let id = CurveId::new();
        let corners = [(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)];
        let points = corners.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect();
        doc.curves.insert(id, Curve::polyline(id, "square", points, true).expect("closed"));
        id
    }

    fn rims(reach: f64, continuity: Continuity) -> Rims {
        Rims { radius: lit(reach), continuity, at: fieldform_scene::RimEnds::Both }
    }

    /// The square extruded thirty up, as told, with its rims as told.
    fn extruded(
        sweep: Sweep,
        draft: f64,
        sides: Sides,
        rims: Rims,
    ) -> Result<GraphField, CompileError> {
        let mut doc = Document::new();
        let id = square(&mut doc);
        let mut graph = Graph::empty();
        graph.add(Node::Extrude {
            outline: Outline::closed(id),
            sweep,
            start: lit(0.0),
            height: lit(30.0),
            draft: lit(draft),
            sides,
            corners: Default::default(),
            rims,
        });
        GraphField::compile_graph(&doc, &graph, 1e-4)
    }

    fn upright(rims: Rims) -> GraphField {
        extruded(Sweep::Normal, 0.0, Sides::One, rims).expect("compiles")
    }

    /// **A rounded rim on a square extrude is a quarter circle of its reach**, through the middle
    /// of an edge: the side face `x = 20` meets the top `z = 30` along a quarter circle of radius
    /// four about `(16, 26)`, at G1 -- the spec's first part-B acceptance. At G2 the same rounding
    /// leaves each face at the same place and bends tighter between; a chamfer leaves them there
    /// too and runs straight.
    #[test]
    fn a_rounded_rim_on_a_square_extrude_is_a_quarter_circle_of_its_reach() {
        let k = 4.0;
        let g1 = upright(rims(k, Continuity::G1));
        for step in 0..=18 {
            let a = std::f64::consts::FRAC_PI_2 * step as f64 / 18.0;
            let (x, z) = (20.0 - k + k * a.cos(), 30.0 - k + k * a.sin());
            let on = g1.distance(Point3::new(x, 0.0, z));
            assert!(on.abs() < 1e-9, "at {} degrees the arc reads {on}", step * 5);
            let (inward, outward) = (Vector3::new(a.cos(), 0.0, a.sin()) * 0.05, Vector3::new(a.cos(), 0.0, a.sin()) * 0.05);
            assert!(g1.distance(Point3::new(x, 0.0, z) - inward) < 0.0, "inside the arc");
            assert!(g1.distance(Point3::new(x, 0.0, z) + outward) > 0.0, "outside the arc");
        }
        // Every shape leaves each face where the reach says, and is the hard solid away from it.
        for shape in Continuity::ALL {
            let field = upright(rims(k, shape));
            let label = shape.label();
            assert!(field.distance(Point3::new(19.99, 0.0, 30.0 - k - 0.05)) < 0.0, "{label}: on the side below the reach");
            assert!(field.distance(Point3::new(20.0 - k - 0.05, 0.0, 29.99)) < 0.0, "{label}: on the top inside the reach");
            assert!(field.distance(Point3::new(19.9, 0.0, 29.9)) > 0.0, "{label}: the corner is gone");
            // A reach clear of the rim the value is the hard solid's, to the bit, at every
            // shape: an upright extrude's readings are square, so its rims read the exact corner
            // (`Rim::cut`), which is the hard corner away from the arc.
            let hard = upright(Rims::hard());
            let divisor = 1.0;
            for p in [Point3::new(0.0, 0.0, 15.0), Point3::new(30.0, 0.0, 15.0), Point3::new(0.0, 0.0, 40.0)] {
                assert!(
                    (field.distance(p) - hard.distance(p) / divisor).abs() < 1e-9,
                    "{label}: a rim changes nothing a reach clear of it, at {p:?}: {} against {}",
                    field.distance(p),
                    hard.distance(p)
                );
            }
        }
        // The chamfer runs straight from leg to leg.
        let chamfer = upright(rims(k, Continuity::Chamfer));
        for step in 0..=10 {
            let s = step as f64 / 10.0;
            let p = Point3::new(20.0 - k * s, 0.0, 30.0 - k + k * s);
            assert!(chamfer.distance(p).abs() < 1e-9, "the bevel at {p:?} reads {}", chamfer.distance(p));
        }
    }

    /// **A rim of nought is the extrude it always was, to the bit**, whatever shape it names --
    /// which is what lets a document that has not asked compile to exactly what it did.
    #[test]
    fn a_rim_of_nought_is_the_extrude_it_always_was() {
        let hard = upright(Rims::hard());
        for shape in Continuity::ALL {
            let field = upright(rims(0.0, shape));
            for i in 0..200 {
                let p = Point3::new(-30.0 + 0.31 * i as f64, 7.0 - 0.07 * i as f64, -5.0 + 0.23 * i as f64);
                assert_eq!(field.distance(p), hard.distance(p), "{shape:?} at {p:?}");
            }
            assert_eq!(field.normalization(), hard.normalization());
        }
    }

    /// **Never overstates**, on the solids where the spec's own argument fails: a draft, two
    /// sides drafted apart, and a lean -- each with its section's gradient out of square with
    /// the slab's, which is why the hard solid takes the larger of the two there rather than
    /// the corner. The rim goes through the blend's bound, which holds at any angle, so a walk
    /// from every probe as far as it claims in six directions crosses no surface.
    #[test]
    fn a_rim_never_overstates_on_a_draft_two_sides_or_a_lean() {
        let toward = |x: f64, z: f64| Sweep::Toward { xyz: [lit(x), lit(0.0), lit(z)] };
        let cases = [
            ("drafted, G1", extruded(Sweep::Normal, 12.0, Sides::One, rims(3.0, Continuity::G1))),
            (
                "two sides drafted apart, G2",
                extruded(Sweep::Normal, 8.0, Sides::Two { height: lit(20.0), draft: lit(-6.0) }, rims(2.5, Continuity::G2)),
            ),
            ("leaning, chamfered", extruded(toward(1.0, 2.0), 0.0, Sides::One, rims(3.0, Continuity::Chamfer))),
            (
                "leaning and drafted, G1",
                extruded(toward(0.6, 2.0), 10.0, Sides::Symmetric, rims(2.0, Continuity::G1)),
            ),
        ];
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        for (what, field) in cases {
            let field = field.expect(what);
            assert_eq!(field.normalization(), crate::Normalization::LowerBound, "{what}");
            for i in 0..12 {
                for j in 0..6 {
                    for k in 0..12 {
                        let from = Point3::new(-33.0 + 6.1 * i as f64, -27.0 + 9.3 * j as f64, -38.0 + 7.7 * k as f64);
                        let claimed = field.distance(from);
                        if claimed <= 0.0 {
                            continue;
                        }
                        for d in dirs {
                            for step in 1..=10 {
                                let at = from + d * (claimed * step as f64 / 10.0);
                                assert!(
                                    field.distance(at) > -1e-9,
                                    "{what}: from {from:?} along {d:?} the surface is crossed inside \
                                     a claimed {claimed}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// **A rim larger than the solid has room for is refused, naming what ran out**: half its
    /// length, past which the two ends' roundings meet, and the section's depth, past which the
    /// rounding meets itself across the section.
    #[test]
    fn a_rim_too_large_for_the_solid_is_refused_naming_what_ran_out() {
        match extruded(Sweep::Normal, 0.0, Sides::One, rims(16.0, Continuity::G1)) {
            Err(CompileError::RimTooLarge { what, room, .. }) => {
                assert_eq!(what, "solid's length");
                assert!((room - 15.0).abs() < 1e-9, "half of thirty: {room}");
            }
            other => panic!("a 16 mm rim on a 30 mm extrude: {:?}", other.err()),
        }
        // The square is twenty deep from its edge to its middle; a long extrude so the length is
        // not what refuses.
        let mut doc = Document::new();
        let id = square(&mut doc);
        let mut graph = Graph::empty();
        graph.add(Node::Extrude {
            outline: Outline::closed(id),
            sweep: Sweep::Normal,
            start: lit(0.0),
            height: lit(200.0),
            draft: lit(0.0),
            sides: Sides::One,
            corners: Default::default(),
            rims: rims(21.0, Continuity::Chamfer),
        });
        match GraphField::compile_graph(&doc, &graph, 1e-4) {
            Err(CompileError::RimTooLarge { what, .. }) => assert_eq!(what, "section's depth"),
            other => panic!("a 21 mm rim on a square twenty deep: {:?}", other.err()),
        }
        assert!(extruded(Sweep::Normal, 0.0, Sides::One, rims(14.9, Continuity::G1)).is_ok());
    }

    /// The square swept along `spine`'s points under `law`, with its rims as told.
    fn swept(spine: &[Point3<f64>], closed: bool, law: fieldform_scene::SweepLaw, rims: Rims) -> Result<GraphField, CompileError> {
        let mut doc = Document::new();
        let profile = square(&mut doc);
        let path = CurveId::new();
        doc.curves.insert(path, Curve::polyline(path, "path", spine.to_vec(), closed).expect("a path"));
        let mut graph = Graph::empty();
        graph.add(Node::Sweep { outline: Outline::closed(profile), spine: path, section: Default::default(), law, rims });
        GraphField::compile_graph(&doc, &graph, 1e-4)
    }

    /// **A sweep's rims round where its side meets each end, as an extrude's do**: along a
    /// straight path up from the square, the rim at the far end is the same quarter circle of
    /// its reach the extrude's is, and the near end is rounded the same way.
    #[test]
    fn a_straight_sweeps_rims_are_the_extrudes() {
        let k = 4.0;
        let line = [Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 30.0)];
        let field = swept(&line, false, Default::default(), rims(k, Continuity::G1)).expect("a rimmed sweep");
        for step in 0..=18 {
            let a = std::f64::consts::FRAC_PI_2 * step as f64 / 18.0;
            let (x, z) = (20.0 - k + k * a.cos(), 30.0 - k + k * a.sin());
            let out = Vector3::new(a.cos(), 0.0, a.sin()) * 0.05;
            assert!(field.distance(Point3::new(x, 0.0, z) - out) < 0.0, "inside the far arc at {} degrees", step * 5);
            assert!(field.distance(Point3::new(x, 0.0, z) + out) > 0.0, "outside it");
            let (x, z) = (20.0 - k + k * a.cos(), k - k * a.sin());
            let out = Vector3::new(a.cos(), 0.0, -a.sin()) * 0.05;
            assert!(field.distance(Point3::new(x, 0.0, z) - out) < 0.0, "inside the near arc");
            assert!(field.distance(Point3::new(x, 0.0, z) + out) > 0.0, "outside it");
        }
    }

    /// **Never overstates on a bent, tapered sweep**: the section changing along the path and
    /// the frame turning with it are the case the spec's argument needed a divisor for, and the
    /// blend's bound needs none. A walk from every probe as far as it claims crosses no surface.
    #[test]
    fn a_rimmed_sweep_never_overstates_on_a_bent_tapered_path() {
        let bend: Vec<Point3<f64>> = (0..=40)
            .map(|i| {
                let a = 0.8 * std::f64::consts::FRAC_PI_2 * i as f64 / 40.0;
                Point3::new(60.0 * (1.0 - a.cos()), 0.0, 60.0 * a.sin())
            })
            .collect();
        let law = fieldform_scene::SweepLaw {
            taper: fieldform_scene::Taper::Scale(lit(0.6)),
            ..Default::default()
        };
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        for shape in Continuity::ALL {
            let field = swept(&bend, false, law.clone(), rims(3.0, shape)).expect("a rimmed tapered sweep");
            for i in 0..10 {
                for j in 0..5 {
                    for k in 0..10 {
                        let from = Point3::new(-30.0 + 9.0 * i as f64, -30.0 + 15.0 * j as f64, -20.0 + 9.0 * k as f64);
                        let claimed = field.distance(from);
                        if claimed <= 0.0 {
                            continue;
                        }
                        for d in dirs {
                            for step in 1..=8 {
                                let at = from + d * (claimed * step as f64 / 8.0);
                                assert!(
                                    field.distance(at) > -1e-9,
                                    "{shape:?}: from {from:?} along {d:?} the surface is crossed inside a claimed {claimed}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// **A closed path has no ends, so a rim on one is refused by name**, and one too large for
    /// an open path's run is refused as the extrude's is.
    #[test]
    fn a_sweeps_rims_are_refused_on_a_closed_path_and_when_too_large() {
        let ring: Vec<Point3<f64>> = (0..24)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 24.0;
                Point3::new(80.0 * (1.0 - a.cos()), 0.0, 80.0 * a.sin())
            })
            .collect();
        let closed = swept(&ring, true, Default::default(), rims(2.0, Continuity::G1));
        assert!(matches!(closed, Err(CompileError::RimsOnAClosedPath)), "{:?}", closed.err());
        let line = [Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 30.0)];
        match swept(&line, false, Default::default(), rims(15.5, Continuity::G1)) {
            Err(CompileError::RimTooLarge { what, .. }) => assert_eq!(what, "sweep's run"),
            other => panic!("a 15.5 mm rim on a 30 mm sweep: {:?}", other.err()),
        }
    }

    fn rims_at(reach: f64, continuity: Continuity, at: fieldform_scene::RimEnds) -> Rims {
        Rims { radius: lit(reach), continuity, at }
    }

    /// Whether the corner where the side `x = 20` meets the face at `z` has been rounded off: a
    /// point a hair inside both faces is inside a sharp corner and outside a rounded one.
    fn corner_is_rounded(field: &GraphField, z: f64, inward: f64) -> bool {
        field.distance(Point3::new(19.95, 0.0, z + inward * 0.05)) > 0.0
    }

    /// **A rim on one end rounds that end and leaves the other sharp** -- the owner's ask,
    /// 2026-10-17. The square extruded thirty up starts at its profile face and ends at its far
    /// one; with two sides it starts at side two's face. The rounded end is the same quarter
    /// circle a rim on both ends makes.
    #[test]
    fn a_rim_on_one_end_rounds_that_end_and_leaves_the_other_sharp() {
        use fieldform_scene::RimEnds::{Both, End, Start};
        let k = 4.0;
        for (at, bottom, top) in [(Both, true, true), (Start, true, false), (End, false, true)] {
            let field = upright(rims_at(k, Continuity::G1, at));
            assert_eq!(corner_is_rounded(&field, 0.0, 1.0), bottom, "{at:?}: the profile face");
            assert_eq!(corner_is_rounded(&field, 30.0, -1.0), top, "{at:?}: the far face");
        }
        // The rounded end is the arc a rim on both makes, to the last digit.
        let both = upright(rims_at(k, Continuity::G1, Both));
        let end = upright(rims_at(k, Continuity::G1, End));
        for step in 0..=18 {
            let a = std::f64::consts::FRAC_PI_2 * step as f64 / 18.0;
            let on = Point3::new(20.0 - k + k * a.cos(), 0.0, 30.0 - k + k * a.sin());
            assert!(end.distance(on).abs() < 1e-9, "the end's arc at {} degrees", step * 5);
            assert!((end.distance(on) - both.distance(on)).abs() < 1e-9);
        }
        // Two sides, ten down and thirty up: the start is side two's face, at z = -10.
        let two = Sides::Two { height: lit(10.0), draft: lit(0.0) };
        let start = extruded(Sweep::Normal, 0.0, two.clone(), rims_at(k, Continuity::G1, Start)).expect("two sides");
        assert!(corner_is_rounded(&start, -10.0, 1.0), "side two's face is the start");
        assert!(!corner_is_rounded(&start, 30.0, -1.0), "side one's is left sharp");
        let end = extruded(Sweep::Normal, 0.0, two, rims_at(k, Continuity::G1, End)).expect("two sides");
        assert!(!corner_is_rounded(&end, -10.0, 1.0) && corner_is_rounded(&end, 30.0, -1.0), "and the end, side one's");
    }

    /// **One end may take the whole length**, where both share it: the roundings of two ends meet
    /// at half the length, and one has nothing to meet.
    #[test]
    fn a_rim_on_one_end_may_take_the_length_both_ends_share() {
        use fieldform_scene::RimEnds::{Both, End};
        // The square is forty across, so its depth leaves room for twenty; thirty up, both ends
        // have fifteen each and one has the thirty.
        assert!(matches!(
            extruded(Sweep::Normal, 0.0, Sides::One, rims_at(16.0, Continuity::G1, Both)),
            Err(CompileError::RimTooLarge { what: "solid's length", .. })
        ));
        assert!(extruded(Sweep::Normal, 0.0, Sides::One, rims_at(16.0, Continuity::G1, End)).is_ok());
    }

    /// **And never overstates**, on the solids where the two ends are least alike: a drafted
    /// extrude, two sides drafted apart, a lean, and a bent tapered sweep, at each shape and each
    /// end alone.
    #[test]
    fn a_rim_on_one_end_never_overstates() {
        use fieldform_scene::RimEnds::{End, Start};
        let toward = |x: f64, z: f64| Sweep::Toward { xyz: [lit(x), lit(0.0), lit(z)] };
        let bend: Vec<Point3<f64>> = (0..=40)
            .map(|i| {
                let a = 0.8 * std::f64::consts::FRAC_PI_2 * i as f64 / 40.0;
                Point3::new(60.0 * (1.0 - a.cos()), 0.0, 60.0 * a.sin())
            })
            .collect();
        let law = fieldform_scene::SweepLaw { taper: fieldform_scene::Taper::Scale(lit(0.6)), ..Default::default() };
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        for at in [Start, End] {
            for shape in Continuity::ALL {
                let cases = [
                    ("drafted", extruded(Sweep::Normal, 12.0, Sides::One, rims_at(3.0, shape, at))),
                    (
                        "two sides drafted apart",
                        extruded(Sweep::Normal, 8.0, Sides::Two { height: lit(20.0), draft: lit(-6.0) }, rims_at(2.5, shape, at)),
                    ),
                    ("leaning", extruded(toward(1.0, 2.0), 0.0, Sides::Symmetric, rims_at(3.0, shape, at))),
                    ("a bent tapered sweep", swept(&bend, false, law.clone(), rims_at(3.0, shape, at))),
                ];
                for (what, field) in cases {
                    let field = field.expect(what);
                    for i in 0..10 {
                        for j in 0..5 {
                            for k in 0..10 {
                                let from = Point3::new(-33.0 + 9.0 * i as f64, -27.0 + 13.0 * j as f64, -38.0 + 11.0 * k as f64);
                                let claimed = field.distance(from);
                                if claimed <= 0.0 {
                                    continue;
                                }
                                for d in dirs {
                                    for step in 1..=8 {
                                        let p = from + d * (claimed * step as f64 / 8.0);
                                        assert!(
                                            field.distance(p) > -1e-9,
                                            "{what}, {shape:?} at {at:?}: from {from:?} along {d:?} \
                                             the surface is crossed inside a claimed {claimed}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// **A joined mirror of an extrude standing on its plane carries its rims to the far ends**:
    /// the merge builds the two halves as one two-sided extrude, whose start is the reflection's
    /// far face. So a rim on the far end alone is on both of the merged solid's ends, and one on
    /// the profile face alone -- the face on the plane, which the merge takes away -- is none.
    #[test]
    fn a_joined_mirror_carries_a_one_ended_rim_to_both_far_ends() {
        use fieldform_scene::RimEnds::{Both, End, Start};
        for (at, merged) in [(Both, Some(Both)), (End, Some(Both)), (Start, None)] {
            let mut doc = Document::new();
            let profile = square(&mut doc);
            let mut graph = Graph::empty();
            let extrude = graph.add(Node::Extrude {
                outline: Outline::closed(profile),
                sweep: Sweep::Normal,
                start: lit(0.0),
                height: lit(30.0),
                draft: lit(0.0),
                sides: Sides::One,
                corners: Default::default(),
                rims: rims_at(4.0, Continuity::G1, at),
            });
            let plane = fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0), lit(0.0), lit(0.0)]);
            graph.add(Node::Mirror { input: extrude.into(), frame: plane, join: true });
            let (_, rewritten) =
                crate::with_shared_faces_merged(&doc, &graph).expect("a section on its plane merges");
            let rims = rewritten
                .nodes()
                .find_map(|(_, node)| match node {
                    Node::Extrude { rims, sides: Sides::Two { .. }, .. } => Some(*rims),
                    _ => None,
                })
                .expect("the merged two-sided extrude");
            match merged {
                Some(ends) => assert_eq!((rims.at, rims.radius), (ends, lit(4.0)), "{at:?}"),
                None => assert!(rims.is_plainly_hard(), "{at:?}: the rim was on the face the merge removes"),
            }
        }
    }

    fn cuboid_with(edge: fieldform_scene::Fillet) -> Result<GraphField, CompileError> {
        let mut graph = Graph::empty();
        graph.add(Node::Cuboid {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0), lit(24.0), lit(16.0)],
            turn: fieldform_scene::no_turn(),
            edge,
        });
        GraphField::compile_graph(&Document::new(), &graph, 1e-4)
    }

    fn cylinder_with(edge: fieldform_scene::Fillet) -> Result<GraphField, CompileError> {
        let mut graph = Graph::empty();
        graph.add(Node::Cylinder {
            ends: fieldform_scene::Ends::between(
                fieldform_scene::Place::At([lit(0.0); 3]),
                fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(10.0)]),
            ),
            radius: lit(8.0),
            edge,
        });
        GraphField::compile_graph(&Document::new(), &graph, 1e-4)
    }

    fn cone_with(rt: f64, edge: fieldform_scene::Fillet) -> Result<GraphField, CompileError> {
        let mut graph = Graph::empty();
        graph.add(Node::Cone {
            ends: fieldform_scene::Ends::between(
                fieldform_scene::Place::At([lit(0.0); 3]),
                fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(20.0)]),
            ),
            base_radius: lit(8.0),
            top_radius: lit(rt),
            edge,
        });
        GraphField::compile_graph(&Document::new(), &graph, 1e-4)
    }

    /// **A cone's edge is held to its section's inradius for a round and its shortest edge for a
    /// chamfer**, by name: a pointed cone of base 8 and height 20 has 5.42 mm of room for a round
    /// and 8 for a chamfer. A round within the room compiles, and a hard edge is the cone it was.
    #[test]
    fn a_cone_edge_too_large_is_refused_naming_what_ran_out() {
        let edge = |r: f64, c: Continuity| fieldform_scene::Fillet { radius: lit(r), continuity: c };
        match cone_with(0.0, edge(5.5, Continuity::G1)) {
            Err(CompileError::EdgeTooLarge { what, room, .. }) => {
                assert_eq!(what, "cone's section");
                // 8·20/(8 + √464), the triangle's inradius.
                assert!((room - 5.4163).abs() < 1e-3, "{room}");
            }
            other => panic!("5.5 on a pointed cone: {:?}", other.err()),
        }
        assert!(cone_with(0.0, edge(5.0, Continuity::G1)).is_ok());
        match cone_with(0.0, edge(8.5, Continuity::Chamfer)) {
            Err(CompileError::EdgeTooLarge { what, .. }) => assert_eq!(what, "cone's shortest edge"),
            other => panic!("8.5 chamfered: {:?}", other.err()),
        }
        let hard = cone_with(3.0, fieldform_scene::Fillet::hard()).expect("a cone");
        let nought = cone_with(3.0, edge(0.0, Continuity::G2)).expect("a cone");
        for p in [Point3::new(9.0, 0.0, 1.0), Point3::new(2.0, 1.0, 19.5), Point3::new(0.0, 0.0, 10.0)] {
            assert_eq!(hard.distance(p), nought.distance(p), "a reach of nought is hard at {p:?}");
        }
    }

    /// **An edge larger than the solid has room for is refused, naming what ran out**: the
    /// box's smallest half-side, and the cylinder's radius or half its length -- the spec's
    /// `r > min(b)`, so the half-side itself is allowed. A hard edge is the solid it always was.
    #[test]
    fn a_box_or_cylinder_edge_too_large_is_refused_naming_what_ran_out() {
        let edge = |r: f64| fieldform_scene::Fillet { radius: lit(r), continuity: Continuity::G1 };
        match cuboid_with(edge(8.5)) {
            Err(CompileError::EdgeTooLarge { what, room, .. }) => {
                assert_eq!((what, room), ("box's smallest half-side", 8.0))
            }
            other => panic!("8.5 on a box 16 deep: {:?}", other.err()),
        }
        assert!(cuboid_with(edge(8.0)).is_ok(), "the half-side itself, the face a line");
        match cylinder_with(edge(5.5)) {
            Err(CompileError::EdgeTooLarge { what, .. }) => assert_eq!(what, "cylinder's half-length"),
            other => panic!("5.5 on a cylinder 10 long: {:?}", other.err()),
        }
        let hard = cuboid_with(fieldform_scene::Fillet::hard()).expect("a box");
        let nought = cuboid_with(fieldform_scene::Fillet { radius: lit(0.0), continuity: Continuity::Chamfer }).expect("a box");
        for p in [Point3::new(25.0, 3.0, 9.0), Point3::new(19.9, 11.9, 7.9), Point3::new(0.0, 0.0, 0.0)] {
            assert_eq!(hard.distance(p), nought.distance(p), "a reach of nought is hard at {p:?}");
        }
    }

    /// A rectangle from radius 30 to 40, ten high, in the XZ plane -- clear of the Z axis it is
    /// turned about, so its ring has an inner wall at 30.
    fn ring_section(doc: &mut Document, inner: f64) -> CurveId {
        let id = CurveId::new();
        let pts = vec![
            Point3::new(inner, 0.0, -5.0),
            Point3::new(40.0, 0.0, -5.0),
            Point3::new(40.0, 0.0, 5.0),
            Point3::new(inner, 0.0, 5.0),
        ];
        doc.curves.insert(id, Curve::polyline(id, "ring", pts, true).expect("closed"));
        id
    }

    fn revolved(inner: f64, angle: f64, sides: fieldform_scene::AngleSides, rims: Rims) -> Result<GraphField, CompileError> {
        let mut doc = Document::new();
        let profile = ring_section(&mut doc, inner);
        let mut graph = Graph::empty();
        graph.add(Node::Revolve {
            outline: Outline::closed(profile),
            frame: fieldform_scene::Frame::default(),
            angle: lit(angle),
            sides,
            rims,
        });
        GraphField::compile_graph(&doc, &graph, 1e-4)
    }

    /// A point at radius `r`, `rel` degrees on from the wedge's start, at height `z`.
    fn at(r: f64, rel_degrees: f64, z: f64) -> Point3<f64> {
        let a = rel_degrees.to_radians();
        Point3::new(r * a.cos(), r * a.sin(), z)
    }

    /// **A partial revolve's rim is the blend's arc between the turned side and the cap**: on a
    /// ring turned ninety degrees, the outer wall `r = 40` meets the start cap along a quarter
    /// circle of the reach, read in the section's distance against the distance to the cap's
    /// plane -- which near the cap is `r·sin(rel)`.
    #[test]
    fn a_partial_revolves_rim_is_a_quarter_circle_of_its_reach() {
        let k = 3.0;
        let field = revolved(30.0, 90.0, fieldform_scene::AngleSides::One, rims(k, Continuity::G1)).expect("a rimmed wedge");
        for step in 0..=18 {
            let a = std::f64::consts::FRAC_PI_2 * step as f64 / 18.0;
            // The arc's point: `x = -k + k cos a` in from the outer wall, `y = -k + k sin a`
            // in from the cap's plane, so the radius and the angle that put it there.
            let r = 40.0 - k + k * a.cos();
            let rel = ((k - k * a.sin()) / r).asin().to_degrees();
            let on = at(r, rel, 0.0);
            assert!(field.distance(on).abs() < 1e-6, "at {} degrees the arc reads {}", step * 5, field.distance(on));
            // Outward along the arc's normal: more radius, less angle.
            let out = Point3::new(
                (r + 0.05 * a.cos()) * (rel.to_radians() - 0.05 * a.sin() / r).cos(),
                (r + 0.05 * a.cos()) * (rel.to_radians() - 0.05 * a.sin() / r).sin(),
                0.0,
            );
            let inward = Point3::new(
                (r - 0.05 * a.cos()) * (rel.to_radians() + 0.05 * a.sin() / r).cos(),
                (r - 0.05 * a.cos()) * (rel.to_radians() + 0.05 * a.sin() / r).sin(),
                0.0,
            );
            assert!(field.distance(out) > 0.0, "outside the arc");
            assert!(field.distance(inward) < 0.0, "inside the arc");
        }
        // Away from the caps the wedge is the hard ring: a point mid-way reads its wall distance
        // outside, and inside a depth between the hard one and the hard one through the scale the
        // lean of its readings costs, `√(1 + k/r_min)` (`Rim::cut`).
        assert!((field.distance(at(42.0, 45.0, 0.0)) - 2.0).abs() < 1e-9);
        let through = (1.0 + k / 30.0).sqrt();
        let depth = field.distance(at(35.0, 45.0, 0.0));
        assert!(depth <= -5.0 / through + 1e-9 && depth >= -5.0 - 1e-9, "the depth, mid-wedge: {depth}");
    }

    /// **A rim on one cap leaves the other sharp**: the start cap is the one the angle is
    /// measured from. A point a hair inside the outer wall and the cap is inside a sharp corner
    /// and outside a rounded one.
    #[test]
    fn a_rim_on_one_cap_of_a_revolve_leaves_the_other_sharp() {
        use fieldform_scene::RimEnds::{Both, End, Start};
        let k = 3.0;
        let corner_rounded = |field: &GraphField, rel: f64| field.distance(at(39.95, rel, 0.0)) > 0.0;
        for (ends, start, end) in [(Both, true, true), (Start, true, false), (End, false, true)] {
            let field = revolved(30.0, 90.0, fieldform_scene::AngleSides::One, rims_at(k, Continuity::G1, ends)).expect("compiles");
            // A twentieth of a degree in from each cap is 0.035 mm, well inside the reach.
            assert_eq!(corner_rounded(&field, 0.05), start, "{ends:?}: the start cap");
            assert_eq!(corner_rounded(&field, 89.95), end, "{ends:?}: the end cap");
        }
    }

    /// **A rimmed revolve never overstates**, inside the wedge and out, past a half turn and
    /// across a thin gap, at every shape and each end -- the case the reading's argument was
    /// made for.
    #[test]
    fn a_rimmed_revolve_never_overstates() {
        use fieldform_scene::AngleSides::{One, Symmetric, Two};
        use fieldform_scene::RimEnds::{Both, End, Start};
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        let cases: Vec<(&str, f64, fieldform_scene::AngleSides, f64)> = vec![
            ("a quarter", 90.0, One, 3.0),
            ("past a half turn", 270.0, One, 3.0),
            ("a thin gap", 350.0, Symmetric, 2.0),
            ("two ways", 60.0, Two { angle: lit(80.0) }, 2.5),
        ];
        for (what, angle, sides, k) in cases {
            for ends in [Both, Start, End] {
                for shape in Continuity::ALL {
                    let field = revolved(30.0, angle, sides.clone(), rims_at(k, shape, ends)).expect(what);
                    assert_eq!(field.normalization(), crate::Normalization::LowerBound);
                    for i in 0..12 {
                        for j in 0..12 {
                            for l in 0..5 {
                                let p = Point3::new(-50.0 + 9.1 * i as f64, -50.0 + 9.1 * j as f64, -12.0 + 6.0 * l as f64);
                                let claimed = field.distance(p);
                                if claimed <= 0.0 {
                                    continue;
                                }
                                for d in dirs {
                                    for step in 1..=8 {
                                        let q = p + d * (claimed * step as f64 / 8.0);
                                        assert!(
                                            field.distance(q) > -1e-9,
                                            "{what}, {shape:?} at {ends:?}: from {p:?} along {d:?} the surface is \
                                             crossed inside a claimed {claimed}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// **Refused by name**: rims on a full revolve, which has no ends; a rim longer than the
    /// section's distance from the axis lets through at the wedge's angle, or at half the gap it
    /// leaves; and a section on the axis, which has no room at all.
    #[test]
    fn a_revolves_rims_are_refused_on_a_full_turn_and_when_too_large() {
        use fieldform_scene::AngleSides::One;
        let full = revolved(30.0, 360.0, One, rims(2.0, Continuity::G1));
        assert!(matches!(full, Err(CompileError::RimsOnAFullRevolve)), "{:?}", full.err());
        // Ninety degrees, both ends: each cap has forty-five to itself, 30·sin 45° = 21.2 mm.
        match revolved(30.0, 90.0, One, rims(22.0, Continuity::G1)) {
            Err(CompileError::RimTooLarge { what, room, .. }) => {
                assert_eq!(what, "revolve's angle");
                assert!((room - 30.0 * 45f64.to_radians().sin()).abs() < 1e-9, "{room}");
            }
            other => panic!("22 on a quarter: {:?}", other.err()),
        }
        // Ten degrees of gap: half of it at radius 30 is 2.61 mm.
        match revolved(30.0, 350.0, One, rims(3.0, Continuity::G1)) {
            Err(CompileError::RimTooLarge { what, .. }) => assert_eq!(what, "gap the revolve leaves"),
            other => panic!("3 across a 10 degree gap: {:?}", other.err()),
        }
        assert!(revolved(30.0, 350.0, One, rims(2.5, Continuity::G1)).is_ok());
        match revolved(0.0, 90.0, One, rims(1.0, Continuity::G1)) {
            Err(CompileError::RimTooLarge { what, room, .. }) => {
                assert_eq!((what, room), ("section's distance from the axis", 0.0));
            }
            other => panic!("a section on the axis: {:?}", other.err()),
        }
        // Ten high: a rim of six would meet itself across the section.
        match revolved(30.0, 90.0, One, rims(6.0, Continuity::G1)) {
            Err(CompileError::RimTooLarge { what, .. }) => assert_eq!(what, "section's depth"),
            other => panic!("deeper than the section: {:?}", other.err()),
        }
    }

    /// The surface point along a corner's diagonal, found by bisection from the sharp corner
    /// inward, and the normal there -- the diagonal itself, which every shape's curve is
    /// symmetric about.
    fn on_the_diagonal(field: &GraphField, corner: Point3<f64>, inward: Vector3<f64>) -> (Point3<f64>, Vector3<f64>) {
        let (mut lo, mut hi) = (0.0, 20.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if field.distance(corner + inward * mid) > 0.0 { lo = mid } else { hi = mid }
        }
        (corner + inward * (0.5 * (lo + hi)), -inward)
    }

    /// **A rim reads the distance along its own normal where its readings are square** --
    /// the owner's "the occlusion looks wrong", 2026-10-18. The max-norm bound every rim used to
    /// read gave `0.707` of the step along the arc's normal at forty-five degrees, on every rim
    /// and every shape (`examples/rim_ao_reading.rs`), which the still's occlusion painted as a
    /// shadow nothing cast. An upright extrude's plane and slab are square, so its G1 arc and its
    /// chamfer's bevel read the step exactly; a partial revolve's lean by at most the reach over
    /// the section's distance from the axis, so it reads the step through `√(1 + k/r_min)`. G2's
    /// cube norm reads `0.89`, a bound by its nature. The drafted extrude keeps the bound.
    #[test]
    fn a_rim_reads_the_distance_along_its_normal_where_its_readings_are_square() {
        let k = 3.0;
        let steps = [0.2, 0.5, 1.0, 2.0];
        let ratio_at = |field: &GraphField, corner: Point3<f64>, inward: Vector3<f64>, h: f64| {
            let (p, n) = on_the_diagonal(field, corner, inward);
            field.distance(p + n * h) / h
        };
        for (shape, floor, exact) in [(Continuity::G1, 0.999, true), (Continuity::Chamfer, 0.999, true), (Continuity::G2, 0.85, false)] {
            let upright = upright(rims(k, shape));
            let corner = Point3::new(20.0, 0.0, 30.0);
            let inward = Vector3::new(-1.0, 0.0, -1.0).normalize();
            for s in steps {
                let r = ratio_at(&upright, corner, inward, s * k);
                assert!(r >= floor && r <= 1.0 + 1e-9, "upright extrude, {shape:?} at {s}k reads {r}");
                if exact {
                    assert!((r - 1.0).abs() < 1e-6, "upright extrude, {shape:?} at {s}k reads {r}, not the distance");
                }
            }
            // The ring from 30 to 40 turned a quarter: the outer wall meets the start cap at
            // (40, 0, 0), the cap plane y = 0 and the wall r = 40.
            let wedge = revolved(30.0, 90.0, fieldform_scene::AngleSides::One, rims(k, shape)).expect("a rimmed wedge");
            // Through `√(1 + k/r_min)` = 1.049 here, less a little for the arc's own normal not
            // being quite the diagonal's where the readings lean: 0.945 measured at G1 against
            // 0.953, against the bound's 0.707 before.
            let through = (1.0 + k / 30.0).sqrt();
            for s in steps {
                let r = ratio_at(&wedge, Point3::new(40.0, 0.0, 0.0), Vector3::new(-1.0, 1.0, 0.0).normalize(), s * k);
                // Beyond the reach the leaning reading is clamped and the rest taken back
                // (`Rim::join_square`), which reads 0.87 at two reaches: still a bound, still
                // well over the max-norm's 0.707, and where occlusion's last sample falls.
                let least = if s > 1.0 { if exact { 0.85 } else { 0.70 } } else { floor / through - 0.02 };
                assert!(r >= least && r <= 1.0 + 1e-9, "partial revolve, {shape:?} at {s}k reads {r} (through {through:.3})");
            }
        }
        // And the drafted extrude keeps the bound, whose reading along the arc is the max-norm's.
        let drafted = extruded(Sweep::Normal, 12.0, Sides::One, rims(k, Continuity::G1)).expect("drafted");
        let (p, n) = on_the_diagonal(&drafted, Point3::new(20.0 - 30.0 * 12f64.to_radians().tan(), 0.0, 30.0), Vector3::new(-1.0, 0.0, -1.0).normalize());
        let r = drafted.distance(p + n * k) / k;
        assert!(r < 0.9, "the drafted extrude's rim is still the bound: {r}");
    }
