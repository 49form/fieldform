// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of the thin revolve and the open-section sweep, against solids known in closed form.
    use super::*;
    use crate::Normalization;
    use fieldform_scene::{
        Curve, CurveId, Document, Frame, Graph, Node, Outline, ParamRef, Section, WallEnds,
        WallSide,
    };
    use nalgebra::Point3;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn drawn(doc: &mut Document, points: &[(f64, f64, f64)], closed: bool) -> CurveId {
        let id = CurveId::new();
        let points = points.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect();
        doc.curves.insert(id, Curve::polyline(id, "curve", points, closed).expect("a polyline"));
        id
    }

    fn wall(curve: CurveId, thickness: f64, side: WallSide, ends: WallEnds) -> Outline {
        Outline::Wall { curve, thickness: lit(thickness), side, ends }
    }

    /// The field of a graph holding one block, walked: at a tolerance too fine to bake at.
    fn of(doc: &Document, node: Node) -> GraphField {
        let mut graph = Graph::empty();
        graph.add(node);
        GraphField::compile_graph(doc, &graph, 1e-4).expect("compiles")
    }

    /// The signed distance to a rectangle, in whatever two coordinates it is asked in.
    fn rectangle(q: (f64, f64), lo: (f64, f64), hi: (f64, f64)) -> f64 {
        let (dx, dy) = (
            (q.0 - 0.5 * (lo.0 + hi.0)).abs() - 0.5 * (hi.0 - lo.0),
            (q.1 - 0.5 * (lo.1 + hi.1)).abs() - 0.5 * (hi.1 - lo.1),
        );
        dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0)
    }

    /// Points about the axis at every radius and height the cases below reach.
    fn about_the_axis() -> Vec<Point3<f64>> {
        (0..400)
            .map(|i| {
                let (a, rho) = ((i as f64 * 37.3).to_radians(), 0.3 + (i % 53) as f64 * 0.83);
                Point3::new(rho * a.cos(), rho * a.sin(), -9.0 + (i % 29) as f64 * 2.37)
            })
            .collect()
    }

    /// **A straight line turned about an axis is a tube or a washer, and its side is the side
    /// it is seen to be on in the half-plane**: the axis running up, the radius out to the
    /// right. So a line drawn upward has the axis on its left, and one drawn outward has up on
    /// its left. Exact, to the last digit, since a straight wall is a rectangle in `(r, z)`
    /// and a revolve of an exact section is exact.
    #[test]
    fn a_line_turned_about_an_axis_is_a_tube_or_a_washer_to_the_side_it_is_seen_on() {
        let mut doc = Document::new();
        let up = drawn(&mut doc, &[(30.0, 0.0, 0.0), (30.0, 0.0, 50.0)], false);
        let out = drawn(&mut doc, &[(10.0, 0.0, 5.0), (40.0, 0.0, 5.0)], false);
        for (what, curve, side, lo, hi) in [
            ("a tube about its curve", up, WallSide::Centre, (28.0, 0.0), (32.0, 50.0)),
            ("a tube inside its curve", up, WallSide::Left, (26.0, 0.0), (30.0, 50.0)),
            ("a tube outside its curve", up, WallSide::Right, (30.0, 0.0), (34.0, 50.0)),
            ("a washer about its curve", out, WallSide::Centre, (10.0, 3.0), (40.0, 7.0)),
            ("a washer above its curve", out, WallSide::Left, (10.0, 5.0), (40.0, 9.0)),
            ("a washer below its curve", out, WallSide::Right, (10.0, 1.0), (40.0, 5.0)),
        ] {
            let solid = of(
                &doc,
                Node::Revolve { rims: Default::default(),
                    outline: wall(curve, 4.0, side, WallEnds::Square),
                    frame: Frame::default(),
                    angle: fieldform_scene::ParamRef::Literal(360.0),
                    sides: fieldform_scene::AngleSides::One,
                },
            );
            assert_eq!(solid.normalization(), Normalization::Distance, "{what}");
            for p in about_the_axis() {
                let exact = rectangle((p.x.hypot(p.y), p.z), lo, hi);
                let got = solid.distance(p);
                assert!((got - exact).abs() < 1e-9, "{what} at {p:?}: {got} against {exact}");
            }
        }
    }

    /// **A band that reaches across the axis is material on this side of it too.** A curve a
    /// millimetre from the axis, with three millimetres of wall toward it, has two of them
    /// beyond the axis — and turned about it they come back as the radii between nought and
    /// two. So the solid is a rod of radius two, not a tube of radius one: read from one side
    /// alone, the point at a radius of one and a half is outside the wall, and it is half a
    /// millimetre inside the rod.
    #[test]
    fn a_wall_that_reaches_across_the_axis_is_solid_through_it() {
        let mut doc = Document::new();
        let near = drawn(&mut doc, &[(1.0, 0.0, 0.0), (1.0, 0.0, 30.0)], false);
        let rod = of(
            &doc,
            Node::Revolve { rims: Default::default(),
                outline: wall(near, 3.0, WallSide::Left, WallEnds::Square),
                frame: Frame::default(),
                angle: fieldform_scene::ParamRef::Literal(360.0),
                sides: fieldform_scene::AngleSides::One,
            },
        );
        let (mut outside, mut inside) = (0, 0);
        // Close about the axis, where the rod is: the wider set would put nine probes in it.
        let close = (0..400).map(|i| {
            let (a, rho) = ((i as f64 * 37.3).to_radians(), 0.05 + (i % 37) as f64 * 0.11);
            Point3::new(rho * a.cos(), rho * a.sin(), -4.0 + (i % 41) as f64 * 0.95)
        });
        for p in close {
            let exact = rectangle((p.x.hypot(p.y), p.z), (-2.0, 0.0), (2.0, 30.0));
            let got = rod.distance(p);
            if exact > 0.0 {
                outside += 1;
                assert!((got - exact).abs() < 1e-9, "at {p:?}: {got} against {exact}");
            } else {
                // Where the wall and its mirror overlap the depth is a union's: never more
                // than the rod's, and the same side of the surface.
                inside += 1;
                assert!(got < 0.0 && got >= exact - 1e-9, "at {p:?}: {got} against {exact}");
            }
        }
        assert!(outside > 100 && inside > 100, "{outside} probes outside the rod, {inside} in it");
        assert!((rod.distance(Point3::new(1.5, 0.0, 15.0)) + 0.5).abs() < 1e-9);

        // The control: the same wall to the other side keeps clear of the axis, and is a tube.
        let tube = of(
            &doc,
            Node::Revolve { rims: Default::default(),
                outline: wall(near, 3.0, WallSide::Right, WallEnds::Square),
                frame: Frame::default(),
                angle: fieldform_scene::ParamRef::Literal(360.0),
                sides: fieldform_scene::AngleSides::One,
            },
        );
        assert!(tube.distance(Point3::new(0.5, 0.0, 15.0)) > 0.0, "the bore of a tube is empty");
    }

    /// **An open line carried along a path is a slab**, which is the solid a sweep could not
    /// make at all: its section is a rectangle the line's length by the wall's thickness, and
    /// a straight line has no plane of its own, so it stands in the one squared to the way
    /// the path sets off. Left is left as the section is seen from ahead on the path, looking
    /// back at it.
    #[test]
    fn an_open_line_carried_along_a_straight_path_is_a_slab() {
        let mut doc = Document::new();
        let line = drawn(&mut doc, &[(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)], false);
        let path = drawn(&mut doc, &[(0.0, 0.0, 0.0), (0.0, 50.0, 0.0)], false);
        for (what, side, z_lo, z_hi) in [
            ("about its curve", WallSide::Centre, -2.0, 2.0),
            ("to its left", WallSide::Left, -4.0, 0.0),
            ("to its right", WallSide::Right, 0.0, 4.0),
        ] {
            let slab = of(
                &doc,
                Node::Sweep { rims: Default::default(),
                    outline: wall(line, 4.0, side, WallEnds::Square),
                    spine: path,
                    section: Section::Square,
                    law: Default::default(),
                },
            );
            for i in 0..300 {
                let p = Point3::new(
                    -17.0 + (i % 13) as f64 * 2.9,
                    -8.0 + (i % 31) as f64 * 2.2,
                    -7.0 + (i % 7) as f64 * 2.3,
                );
                // A box: the section's rectangle in (x, z), and the path's length in y.
                let section = rectangle((p.x, p.z), (-10.0, z_lo), (10.0, z_hi));
                let along = (p.y - 25.0).abs() - 25.0;
                let exact = if section > 0.0 && along > 0.0 {
                    section.hypot(along)
                } else {
                    section.max(along)
                };
                let got = slab.distance(p);
                assert!((got - exact).abs() < 1e-9, "{what} at {p:?}: {got} against {exact}");
            }
        }
    }

    /// **An open section is carried by the point its path sets off from, and turns about it.**
    /// A line carried round a quarter circle is a flat ring; a path drawn from one end of the
    /// line puts that end on the path, so the ring runs from the path's radius inward by the
    /// line's length. The same line drawn the other way round is the same ring, because the
    /// path still sets off from the same place -- which is what the middle-of-the-ends rule
    /// this replaced (earlier on 2026-10-02) was for, and what the end-it-was-drawn-from rule
    /// before that got wrong.
    #[test]
    fn an_open_section_turns_about_the_point_its_path_sets_off_from() {
        use std::f64::consts::FRAC_1_SQRT_2;
        for ends in [[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], [(10.0, 0.0, 0.0), (0.0, 0.0, 0.0)]] {
            let mut doc = Document::new();
            let line = drawn(&mut doc, &ends, false);
            // A quarter circle of radius 50 about (50, 0, 0), setting off along +Y.
            let path = CurveId::new();
            let quarter = Curve::new(
                path,
                "path",
                2,
                vec![Point3::origin(), Point3::new(0.0, 50.0, 0.0), Point3::new(50.0, 50.0, 0.0)],
                vec![1.0, FRAC_1_SQRT_2, 1.0],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                false,
            )
            .expect("a rational quadratic");
            doc.curves.insert(path, quarter);
            let ring = of(
                &doc,
                Node::Sweep { rims: Default::default(),
                    outline: wall(line, 2.0, WallSide::Centre, WallEnds::Square),
                    spine: path,
                    section: Section::Square,
                    law: Default::default(),
                },
            );
            // The path sets off from the line's end at the origin and bends about (50, 0, 0),
            // which the line points at: so the line lies between 40 and 50 mm from there,
            // whichever way it was drawn. Half way round:
            let at = |radius: f64, z: f64| {
                Point3::new(50.0 - radius * FRAC_1_SQRT_2, radius * FRAC_1_SQRT_2, z)
            };
            for (radius, z, inside) in [
                (45.0, 0.0, true),
                (40.3, 0.0, true),
                (49.7, 0.0, true),
                (39.7, 0.0, false),
                (50.3, 0.0, false),
                (45.0, 0.7, true),
                (45.0, 1.3, false),
            ] {
                let got = ring.distance(at(radius, z));
                assert_eq!(
                    got < 0.0,
                    inside,
                    "{radius} mm out and {z} up, drawn from {:?}: {got}",
                    ends[0]
                );
            }
            // And its two edges are where that puts them, to the path's own flattening.
            for radius in [40.0, 50.0] {
                let got = ring.distance(at(radius, 0.0));
                assert!(got.abs() < 1e-3, "the edge {radius} mm out reads {got}");
            }
        }
    }

    /// The contract, where neither solid has a closed form: a one-sided wall on a chain with a
    /// sharp corner in it, turned about an axis and carried round a bend. Walked and from a
    /// grid, each says what it is, and neither claims more room than there is.
    #[test]
    fn a_thin_revolve_and_an_open_sweep_keep_the_contract_walked_and_baked() {
        let mut doc = Document::new();
        // In the half-plane y = 0, clear of the axis, with a corner past a right angle.
        let hook = drawn(
            &mut doc,
            &[(20.0, 0.0, -10.0), (34.0, 0.0, -4.0), (30.0, 0.0, 12.0), (22.0, 0.0, 4.0)],
            false,
        );
        // A section in the plane x = 0, and a path that leaves it and bends in two planes.
        let section = drawn(
            &mut doc,
            &[(0.0, -6.0, -4.0), (0.0, 4.0, -5.0), (0.0, 7.0, 3.0), (0.0, 1.0, 1.0)],
            false,
        );
        let path = drawn(
            &mut doc,
            &[(0.0, 0.0, 0.0), (30.0, 4.0, 0.0), (58.0, 16.0, 9.0), (80.0, 36.0, 24.0)],
            false,
        );
        let probes: Vec<Point3<f64>> = (0..500)
            .map(|i| {
                Point3::new(
                    -45.0 + (i % 23) as f64 * 6.1,
                    -44.0 + (i % 17) as f64 * 5.7,
                    -22.0 + (i % 11) as f64 * 5.3,
                )
            })
            .collect();
        for (what, node, walked) in [
            (
                "a thin revolve",
                Node::Revolve { rims: Default::default(),
                    outline: wall(hook, 3.0, WallSide::Left, WallEnds::Round),
                    frame: Frame::default(),
                    angle: fieldform_scene::ParamRef::Literal(360.0),
                    sides: fieldform_scene::AngleSides::One,
                },
                Normalization::Distance,
            ),
            (
                "an open-section sweep",
                Node::Sweep { rims: Default::default(),
                    outline: wall(section, 2.0, WallSide::Right, WallEnds::Square),
                    spine: path,
                    section: Section::Square,
                    law: Default::default(),
                },
                // A sweep is a bound whatever it carries, divided by its own stretch.
                Normalization::LowerBound,
            ),
        ] {
            let mut graph = Graph::empty();
            graph.add(node);
            for (tolerance, said) in [(1e-4, walked), (1.0, Normalization::LowerBound)] {
                let field = GraphField::compile_graph(&doc, &graph, tolerance).expect("compiles");
                assert_eq!(field.normalization(), said, "{what} at a tolerance of {tolerance}");
                let over =
                    crate::overstatement(&field, &probes, &crate::probe_directions(), 0.05, 1.2);
                assert!(over < 1e-6, "{what}, tolerance {tolerance}: claimed {over} mm too much");
                assert!(
                    probes.iter().any(|p| field.distance(*p) < 0.0),
                    "{what}: no probe fell inside it, so nothing was tested"
                );
            }
        }
    }
