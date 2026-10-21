// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a thin extrude through the whole graph, against the solids it should make.
    use super::*;
    use crate::Normalization;
    use fieldform_scene::{
        Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Sweep, WallEnds, WallSide,
    };
    use nalgebra::{Point3, Vector3};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// A degree-one curve through `points`, put in the document.
    fn drawn(doc: &mut Document, points: Vec<Point3<f64>>, closed: bool) -> CurveId {
        let id = CurveId::new();
        doc.curves.insert(id, Curve::polyline(id, "line", points, closed).expect("a polyline"));
        id
    }

    fn thin(curve: CurveId, thickness: f64, side: WallSide, ends: WallEnds, sweep: Sweep) -> Node {
        Node::Extrude { rims: Default::default(),
            outline: Outline::Wall { curve, thickness: lit(thickness), side, ends },
            sweep,
            height: lit(20.0),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        }
    }

    /// The field of a graph holding one block.
    fn of(doc: &Document, node: Node) -> GraphField {
        let mut graph = Graph::empty();
        graph.add(node);
        GraphField::compile_graph(doc, &graph, 1e-4).expect("a thin extrude compiles")
    }

    /// Points all round a 60 x 8 x 20 slab and through it, off its faces and its axes.
    fn probes() -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for i in 0..17 {
            for j in 0..9 {
                for k in 0..9 {
                    out.push(Point3::new(
                        -13.0 + i as f64 * 5.3 + 0.11 * j as f64,
                        -17.0 + j as f64 * 4.1 + 0.07 * k as f64,
                        -9.0 + k as f64 * 4.7 + 0.05 * i as f64,
                    ));
                }
            }
        }
        out
    }

    /// The exact distance to a box, inside and out.
    fn box_distance(p: Point3<f64>, lo: [f64; 3], hi: [f64; 3]) -> f64 {
        let d: Vec<f64> =
            (0..3).map(|i| (p[i] - 0.5 * (lo[i] + hi[i])).abs() - 0.5 * (hi[i] - lo[i])).collect();
        let outside = (d[0].max(0.0).powi(2) + d[1].max(0.0).powi(2) + d[2].max(0.0).powi(2)).sqrt();
        if outside > 0.0 { outside } else { d[0].max(d[1]).max(d[2]) }
    }

    /// The spec's second acceptance case, and the plainest thing a wall can be: a straight line
    /// drawn on the ground, given a thickness and pulled up, is a cuboid — to the kernel's own
    /// `Cuboid`, at every probe, and for a wall to either side as much as for one about the
    /// line. It also pins which way is left: walking the line and looking down on the plane.
    #[test]
    fn a_square_ended_wall_on_a_straight_line_extrudes_to_a_cuboid() {
        let mut doc = Document::new();
        let line = drawn(&mut doc, vec![Point3::origin(), Point3::new(60.0, 0.0, 0.0)], false);
        for (side, y_lo, y_hi) in [
            (WallSide::Centre, -4.0, 4.0),
            (WallSide::Left, 0.0, 8.0),
            (WallSide::Right, -8.0, 0.0),
        ] {
            let wall = of(&doc, thin(line, 8.0, side, WallEnds::Square, Sweep::Normal));
            let cuboid = of(
                &doc,
                Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([
                        lit(30.0),
                        lit(0.5 * (y_lo + y_hi)),
                        lit(10.0),
                    ]),
                    size: [lit(60.0), lit(8.0), lit(20.0)],
                    turn: fieldform_scene::no_turn(),
                },
            );
            for p in probes() {
                let (a, b) = (wall.distance(p), cuboid.distance(p));
                assert!((a - b).abs() < 1e-9, "{side:?} at {p:?}: the wall {a}, the cuboid {b}");
                let exact = box_distance(p, [0.0, y_lo, 0.0], [60.0, y_hi, 20.0]);
                assert!((a - exact).abs() < 1e-9, "{side:?} at {p:?}: {a} against {exact}");
            }
            assert_eq!(wall.normalization(), Normalization::Distance, "{side:?}");
        }
    }

    /// The spec's first: with round ends the same wall is a stadium — a rectangle with a half
    /// round on each end — pulled up into a prism.
    #[test]
    fn a_round_ended_wall_on_a_straight_line_extrudes_to_a_stadium_prism() {
        let mut doc = Document::new();
        let line = drawn(&mut doc, vec![Point3::origin(), Point3::new(60.0, 0.0, 0.0)], false);
        let wall = of(&doc, thin(line, 8.0, WallSide::Centre, WallEnds::Round, Sweep::Normal));
        for p in probes() {
            // The stadium in the plane: how far from the line, less half the thickness.
            let along = p.x.clamp(0.0, 60.0);
            let stadium = (p.x - along).hypot(p.y) - 4.0;
            // And the prism over it, by the same corner an extrusion takes.
            let slab = (p.z - 10.0).abs() - 10.0;
            let exact =
                if stadium > 0.0 && slab > 0.0 { stadium.hypot(slab) } else { stadium.max(slab) };
            let got = wall.distance(p);
            assert!((got - exact).abs() < 1e-9, "at {p:?}: {got} against {exact}");
        }
        assert_eq!(wall.normalization(), Normalization::Distance);
    }

    /// The spec's third: a wall about a circle is a tube, which is one cylinder less another —
    /// and it moves at unit rate in the bore as well as outside, so the distance it claims is
    /// one. On the inside of the circle and on the outside of it the tube moves over by half
    /// its wall, and which of those "left" is comes from the way the circle was drawn.
    #[test]
    fn a_wall_on_a_circle_extrudes_to_a_tube() {
        let mut doc = Document::new();
        let (radius, thickness) = (30.0, 6.0);
        // Counter-clockwise seen from above, so left of travel is toward the middle.
        let ring: Vec<Point3<f64>> = (0..360)
            .map(|i| {
                let a = (i as f64).to_radians();
                Point3::new(radius * a.cos(), radius * a.sin(), 0.0)
            })
            .collect();
        let circle = drawn(&mut doc, ring, true);
        let sagitta = radius * (1.0 - (0.5f64).to_radians().cos()) + 1e-9;

        for (side, middle) in [
            (WallSide::Centre, radius),
            (WallSide::Left, radius - 0.5 * thickness),
            (WallSide::Right, radius + 0.5 * thickness),
        ] {
            let tube = of(&doc, thin(circle, thickness, side, WallEnds::Square, Sweep::Normal));
            let mut outside = Vec::new();
            for i in 0..120 {
                let (a, rho) = ((i as f64 * 37.3).to_radians(), 6.0 + i as f64 * 0.41);
                let z = -6.0 + (i % 11) as f64 * 3.1;
                let p = Point3::new(rho * a.cos(), rho * a.sin(), z);
                let ring = (rho - middle).abs() - 0.5 * thickness;
                let slab = (z - 10.0).abs() - 10.0;
                let exact = if ring > 0.0 && slab > 0.0 { ring.hypot(slab) } else { ring.max(slab) };
                let got = tube.distance(p);
                assert!((got - exact).abs() < sagitta, "{side:?} at {p:?}: {got} against {exact}");
                if exact > 0.5 {
                    outside.push(p);
                }
            }
            assert!(outside.len() > 50, "the probes must reach the bore and the outside");
            let off = crate::gradient_deviation(&tube, &outside, 1e-5);
            assert!(off < 1e-4, "{side:?} is {off} off unit rate");
            assert_eq!(tube.normalization(), Normalization::Distance);
        }
    }

    /// **A straight line has no plane of its own, so it is given one a person would predict.**
    /// Drawn on the ground it stands upright. Told which way it is swept, it squares itself to
    /// that — so a rib pulled along a typed direction is exact, where a plane picked without
    /// regard to the sweep would have made the same rib a leaning extrude and an approximate
    /// one. And a line that is itself upright, with nothing to go by, faces along X.
    #[test]
    fn a_straight_wall_stands_upright_unless_told_where_it_is_swept() {
        let mut doc = Document::new();
        let on_the_ground =
            drawn(&mut doc, vec![Point3::new(5.0, 5.0, 0.0), Point3::new(45.0, 35.0, 0.0)], false);
        let in_a_back_view =
            drawn(&mut doc, vec![Point3::origin(), Point3::new(40.0, 0.0, 30.0)], false);
        // Away from the others: two lines that share an end are one chain, and a bent one.
        let upright = drawn(
            &mut doc,
            vec![Point3::new(90.0, 90.0, 0.0), Point3::new(90.0, 90.0, 50.0)],
            false,
        );
        let plane = |curve: CurveId, sweep: &Sweep| {
            let outline = Outline::Wall {
                curve,
                thickness: lit(2.0),
                side: WallSide::Centre,
                ends: WallEnds::Square,
            };
            extrude_section(&doc, &outline, sweep, 1.0).expect("a line is a chain")
        };

        let ground = plane(on_the_ground, &Sweep::Normal);
        assert!((ground.normal - Vector3::z()).norm() < 1e-12, "{:?}", ground.normal);
        // And measured from midway between the line's own two ends, which no tessellation moves.
        assert_eq!(ground.origin, Point3::new(25.0, 20.0, 0.0));

        // With nothing said, as near upright as a sloping line allows: square to the line, in
        // the vertical plane the line lies in.
        let sloping = plane(in_a_back_view, &Sweep::Normal);
        assert!((sloping.normal - Vector3::new(-0.6, 0.0, 0.8)).norm() < 1e-12, "{:?}", sloping.normal);
        // Told it is swept along Y, it faces Y, and the sweep is straight out of its plane.
        let along_y = Sweep::Toward { xyz: [lit(0.0), lit(3.0), lit(0.0)] };
        let squared = plane(in_a_back_view, &along_y);
        assert!((squared.normal - Vector3::y()).norm() < 1e-12, "{:?}", squared.normal);
        // So the rib is the shape asked for, unsheared, and reads true: a 2 mm wall along that
        // line pulled 20 mm along Y. On the line and half way up it is 1 mm deep; a millimetre
        // short of the plane it is 1 mm clear; and 3 mm off the line, across it, it is 2 mm
        // clear. It is still *reported* as a bound, as any typed sweep is, and that is the
        // existing rule's to say: only `Sweep::Normal` is exact by construction.
        let rib = of(&doc, thin(in_a_back_view, 2.0, WallSide::Centre, WallEnds::Square, along_y));
        let reads = |p: Point3<f64>, expected: f64| {
            let got = rib.distance(p);
            assert!((got - expected).abs() < 1e-9, "at {p:?}: {got}, not {expected}");
        };
        reads(Point3::new(20.0, 10.0, 15.0), -1.0);
        reads(Point3::new(20.0, -1.0, 15.0), 1.0);
        reads(Point3::new(20.0 - 1.8, 10.0, 15.0 + 2.4), 2.0);
        assert_eq!(rib.normalization(), Normalization::LowerBound);

        let standing = plane(upright, &Sweep::Normal);
        assert!((standing.normal - Vector3::x()).norm() < 1e-12, "{:?}", standing.normal);
    }

    /// The walk a wall's block makes is the walk the shader's polyline is built from, at a
    /// different fineness — so where the section is measured *from* must not move with the
    /// fineness, or the two would put one wall in two places.
    #[test]
    fn a_walls_plane_does_not_move_with_how_finely_its_curve_is_sampled() {
        let mut doc = Document::new();
        // A quarter-round cubic, which samples to a different point count at each fineness.
        let id = CurveId::new();
        let control = vec![
            Point3::new(40.0, 0.0, 5.0),
            Point3::new(40.0, 22.0, 5.0),
            Point3::new(22.0, 40.0, 5.0),
            Point3::new(0.0, 40.0, 5.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let arc = Curve::new(id, "arc", 3, control, vec![1.0; 4], knots, false).expect("a cubic");
        doc.curves.insert(id, arc);
        let outline = Outline::Wall {
            curve: id,
            thickness: lit(3.0),
            side: WallSide::Left,
            ends: WallEnds::Square,
        };
        let coarse = extrude_section(&doc, &outline, &Sweep::Normal, 10.0).unwrap();
        let fine = extrude_section(&doc, &outline, &Sweep::Normal, 0.5).unwrap();
        assert!(fine.points.len() > 2 * coarse.points.len(), "the two must differ in fineness");
        assert_eq!(coarse.origin, fine.origin);
        assert!((coarse.normal - fine.normal).norm() < 1e-12);
        assert!((fine.normal - Vector3::z()).norm() < 1e-12);
        // Midway between the arc's two ends, and so off the arc itself.
        assert!((fine.origin - Point3::new(20.0, 20.0, 5.0)).norm() < 1e-12, "{:?}", fine.origin);
    }

    /// **A wall along a loop is laid where the profile of that loop is**, so a tube is carried
    /// along a path from the point the rod it was drawn from is carried from, and turning Wall
    /// on moves nothing. A profile's origin is the centroid of its sampled points, which does
    /// move with the sampling — so the control here is the profile itself, read at two
    /// finenesses and found in two places, where the wall is in one.
    #[test]
    fn a_wall_along_a_loop_is_laid_where_the_profile_of_that_loop_is() {
        let mut doc = Document::new();
        // A quarter round closed by two straight lines. Sampling by angle puts every point but
        // one on the arc, so the centroid of the points goes toward the arc as it gets finer.
        let id = CurveId::new();
        let control = vec![
            Point3::new(40.0, 0.0, 5.0),
            Point3::new(40.0, 22.0, 5.0),
            Point3::new(22.0, 40.0, 5.0),
            Point3::new(0.0, 40.0, 5.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let arc = Curve::new(id, "arc", 3, control, vec![1.0; 4], knots, false).expect("a cubic");
        doc.curves.insert(id, arc);
        drawn(&mut doc, vec![Point3::new(0.0, 40.0, 5.0), Point3::new(0.0, 0.0, 5.0)], false);
        drawn(&mut doc, vec![Point3::new(0.0, 0.0, 5.0), Point3::new(40.0, 0.0, 5.0)], false);
        let wall = Outline::Wall {
            curve: id,
            thickness: lit(2.0),
            side: WallSide::Centre,
            ends: WallEnds::Square,
        };
        let profile = Outline::closed(id);
        let laid = |outline: &Outline, degrees: f64| {
            extrude_section(&doc, outline, &Sweep::Normal, degrees).expect("a loop")
        };

        let compiled_at = crate::profile::PROFILE_TURN_DEGREES;
        let as_a_profile = laid(&profile, compiled_at);
        for degrees in [10.0, compiled_at, 0.5] {
            let as_a_wall = laid(&wall, degrees);
            assert!(as_a_wall.closed);
            assert_eq!(as_a_wall.origin, as_a_profile.origin, "walked at {degrees} degrees");
            assert!((as_a_wall.normal - as_a_profile.normal).norm() < 1e-12);
        }
        // The control: the same reading of the profile does move, and by more than a rounding.
        let moved = (laid(&profile, 10.0).origin - as_a_profile.origin).norm();
        assert!(moved > 1e-3, "the profile's own origin moved {moved} mm between finenesses");
    }

    /// Each by the number or the thing to change, and none of them a solid that was not asked
    /// for.
    #[test]
    fn a_wall_that_cannot_be_built_is_refused_by_name() {
        let mut doc = Document::new();
        let line = drawn(&mut doc, vec![Point3::origin(), Point3::new(60.0, 0.0, 0.0)], false);
        let refused = |doc: &Document, node: Node| {
            let mut graph = Graph::empty();
            graph.add(node);
            GraphField::compile_graph(doc, &graph, 1e-4).err().expect("this must not compile")
        };

        let none = refused(&doc, thin(line, 0.0, WallSide::Centre, WallEnds::Square, Sweep::Normal));
        assert!(
            matches!(none, CompileError::BadWall(_, crate::WallError::NoThickness(t)) if t == 0.0),
            "{none}"
        );
        assert!(none.to_string().contains("thickness"), "{none}");

        // Three legs that leave the plane the first two lie in.
        let twisted = drawn(
            &mut doc,
            vec![
                Point3::origin(),
                Point3::new(30.0, 0.0, 0.0),
                Point3::new(30.0, 30.0, 0.0),
                Point3::new(30.0, 30.0, 12.0),
            ],
            false,
        );
        let bent = refused(&doc, thin(twisted, 2.0, WallSide::Centre, WallEnds::Square, Sweep::Normal));
        assert!(matches!(bent, CompileError::BadWall(_, crate::WallError::NotPlanar(_))), "{bent}");

        // A thickness that names a parameter the document does not hold has no value.
        let unnamed = Node::Extrude { rims: Default::default(),
            outline: Outline::Wall {
                curve: line,
                thickness: ParamRef::Named(fieldform_scene::ParamId::new()),
                side: WallSide::Centre,
                ends: WallEnds::Square,
            },
            sweep: Sweep::Normal,
            height: lit(20.0),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        };
        assert!(matches!(refused(&doc, unnamed), CompileError::UnresolvedRadius));

        // And a curve that is not there is a missing curve, as it is for a profile.
        let gone = thin(CurveId::new(), 2.0, WallSide::Centre, WallEnds::Square, Sweep::Normal);
        assert!(matches!(
            refused(&doc, gone),
            CompileError::BadLoop(_, crate::LoopError::MissingCurve(_))
        ));
    }

    /// What the panel asks without compiling has to be what the compiler would say: a thin
    /// extrude straight out of its plane is a distance, and one that leans or is drafted is a
    /// bound, exactly as a thick one is.
    #[test]
    fn a_thin_extrude_claims_a_distance_exactly_when_a_thick_one_would() {
        let mut doc = Document::new();
        let line = drawn(
            &mut doc,
            vec![Point3::origin(), Point3::new(40.0, 0.0, 0.0), Point3::new(40.0, 30.0, 0.0)],
            false,
        );
        let leaning = Sweep::Toward { xyz: [lit(1.0), lit(0.0), lit(2.0)] };
        for (sweep, draft, exact) in [
            (Sweep::Normal, 0.0, true),
            (Sweep::Normal, 5.0, false),
            (leaning, 0.0, false),
        ] {
            let mut graph = Graph::empty();
            let node = graph.add(Node::Extrude { rims: Default::default(),
                outline: Outline::Wall {
                    curve: line,
                    thickness: lit(3.0),
                    side: WallSide::Left,
                    ends: WallEnds::Round,
                },
                sweep,
                height: lit(20.0),
                draft: lit(draft),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
            assert_eq!(field.normalization() == Normalization::Distance, exact, "draft {draft}");
            assert_eq!(carries_true_distance(&doc, &graph, node), exact, "draft {draft}");
        }
    }
