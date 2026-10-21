// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a revolve through part of a turn, through the whole graph.
    use super::*;
    use crate::Normalization;
    use fieldform_scene::{
        AngleSides, Curve, CurveId, Document, Frame, Graph, Node, Outline, ParamRef, WallEnds,
        WallSide,
    };
    use nalgebra::{Point3, Vector3};
    use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

    const R: f64 = 30.0;
    const A: f64 = 8.0;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// An exact circle of radius `A` about `(R, 0, 0)` in the half-plane `y = 0`, as a closed
    /// rational quadratic, so that the profile lies in one half-plane about the world's Z.
    fn circle(doc: &mut Document) -> CurveId {
        let w = FRAC_1_SQRT_2;
        let corners = [
            (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0),
            (0.0, -1.0), (1.0, -1.0), (1.0, 0.0),
        ];
        let control = corners.iter().map(|&(u, v)| Point3::new(R + A * u, 0.0, A * v)).collect();
        let weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];
        let knots = vec![0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0];
        let id = CurveId::new();
        doc.curves.insert(id, Curve::new(id, "circle", 2, control, weights, knots, true).unwrap());
        id
    }

    fn revolved(doc: &Document, outline: Outline, angle: f64, sides: AngleSides) -> GraphField {
        let mut graph = Graph::empty();
        graph.add(Node::Revolve { rims: Default::default(), outline, frame: Frame::default(), angle: lit(angle), sides });
        GraphField::compile_graph(doc, &graph, 1e-4).expect("compiles")
    }

    /// Points all round the torus the circle makes, in and out of it, at every angle.
    fn probes() -> Vec<Point3<f64>> {
        (0..900)
            .map(|i| {
                let a = (i as f64 * 0.7).to_radians() * 7.3;
                let rho = 12.0 + (i % 23) as f64 * 1.7;
                Point3::new(rho * a.cos(), rho * a.sin(), -14.0 + (i % 19) as f64 * 1.55)
            })
            .collect()
    }

    /// The exact signed distance to the half torus `y >= 0`: the torus's surface where its
    /// nearest point lies on that side, else the nearer rim circle; the two half-disc caps; and
    /// inside, the depth to the nearest of those. Written from the solid's own pieces and not
    /// from the leaf's cases.
    fn half_torus(p: Point3<f64>) -> f64 {
        let rho = p.x.hypot(p.y);
        let torus = (rho - R).hypot(p.z) - A;
        // The nearest point of the full torus, by azimuth.
        let nearest_on_that_side = p.y >= 0.0;
        // Distance to a disc of radius `A` about `(R·c, R·s, 0)` in the plane at that angle.
        let disc = |angle: f64| {
            let (c, s) = (angle.cos(), angle.sin());
            // In the disc's plane: `u` along the radius, `z` up, and `h` off the plane.
            let u = p.x * c + p.y * s;
            let h = -p.x * s + p.y * c;
            let in_plane = (u - R).hypot(p.z);
            let d = (in_plane - A).max(0.0);
            d.hypot(h)
        };
        // Distance to the rim circle of that disc, which is where the torus surface ends.
        let rim = |angle: f64| {
            let (c, s) = (angle.cos(), angle.sin());
            let u = p.x * c + p.y * s;
            let h = -p.x * s + p.y * c;
            let in_plane = (u - R).hypot(p.z);
            (in_plane - A).hypot(h)
        };
        let surface = if nearest_on_that_side { torus.abs() } else { rim(0.0).min(rim(PI)) };
        let caps = disc(0.0).min(disc(PI));
        let nearest = surface.min(caps);
        let inside = torus < 0.0 && p.y > 0.0;
        if inside { -nearest } else { nearest }
    }

    /// **A circle turned half way is the half torus**: exactly so everywhere outside it, and
    /// inside never deeper than it is -- the one place the leaf is a bound.
    #[test]
    fn a_half_turn_of_a_circle_is_the_half_torus_outside_and_never_deeper_inside() {
        let mut doc = Document::new();
        let id = circle(&mut doc);
        let wedge = revolved(&doc, Outline::closed(id), 180.0, AngleSides::One);
        assert_eq!(wedge.normalization(), Normalization::LowerBound);
        let (mut outside, mut inside) = (0, 0);
        for p in probes() {
            let (got, want) = (wedge.distance(p), half_torus(p));
            if want > 0.0 {
                outside += 1;
                // To the profile's own flattening: a rational circle sampled at a degree.
                assert!((got - want).abs() < 2e-3, "outside at {p:?}: {got} against {want}");
            } else {
                inside += 1;
                assert!(got <= 0.0, "inside at {p:?}: {got} but the solid is there");
                assert!(got >= want - 2e-3, "inside at {p:?}: {got} is deeper than {want}");
            }
        }
        assert!(outside > 300 && inside > 50, "{outside} outside, {inside} inside");
        // The control: the other half is empty, and the whole turn is the torus.
        assert!(wedge.distance(Point3::new(0.0, -R, 0.0)) > A - 1e-3);
        let whole = revolved(&doc, Outline::closed(id), 360.0, AngleSides::One);
        assert_eq!(whole.normalization(), Normalization::Distance);
        let torus = |p: Point3<f64>| (p.x.hypot(p.y) - R).hypot(p.z) - A;
        for p in probes() {
            assert!((whole.distance(p) - torus(p)).abs() < 2e-3);
        }
    }

    /// **Half each way through a quarter is one way through a quarter, turned back by an
    /// eighth**, pointwise -- which is what makes `Symmetric` a spelling and not a solid.
    #[test]
    fn a_symmetric_quarter_turn_is_a_one_way_quarter_turn_turned_back_by_half() {
        let mut doc = Document::new();
        let id = circle(&mut doc);
        let symmetric = revolved(&doc, Outline::closed(id), 90.0, AngleSides::Symmetric);
        let one_way = revolved(&doc, Outline::closed(id), 90.0, AngleSides::One);
        let two_ways = revolved(&doc, Outline::closed(id), 60.0, AngleSides::Two { angle: lit(30.0) });
        let turned = |p: Point3<f64>, by: f64| {
            let (c, s) = (by.cos(), by.sin());
            Point3::new(p.x * c - p.y * s, p.x * s + p.y * c, p.z)
        };
        for p in probes() {
            let want = one_way.distance(turned(p, 45f64.to_radians()));
            let got = symmetric.distance(p);
            assert!((got - want).abs() < 1e-9, "symmetric at {p:?}: {got} against {want}");
            // Two ways, 60 forward and 30 back, is the same wedge as one way turned back 30.
            let want = one_way.distance(turned(p, 30f64.to_radians()));
            let got = two_ways.distance(p);
            assert!((got - want).abs() < 1e-9, "two ways at {p:?}: {got} against {want}");
        }
    }

    /// **A wedge past a half turn picks the nearer cap for every angle outside it**, and the
    /// field is continuous where the two caps are equally near. Probed either side of that
    /// line, which for a 270 degree wedge is 45 degrees from each cap.
    #[test]
    fn a_wedge_past_a_half_turn_picks_the_nearer_cap() {
        let mut doc = Document::new();
        let id = circle(&mut doc);
        let wedge = revolved(&doc, Outline::closed(id), 270.0, AngleSides::One);
        // The caps are at 0 and 270 degrees; outside is 270 to 360, and 315 is equidistant.
        let at = |degrees: f64, rho: f64, z: f64| {
            let a = degrees.to_radians();
            Point3::new(rho * a.cos(), rho * a.sin(), z)
        };
        let just_before = wedge.distance(at(314.9, R, 0.0));
        let just_after = wedge.distance(at(315.1, R, 0.0));
        assert!((just_before - just_after).abs() < 0.2, "{just_before} against {just_after}");
        // Each is the distance to the nearer cap's disc, computed on its own: from the point
        // at 300 degrees the end cap at 270 is 30 degrees off, the start cap 60.
        let p = at(300.0, R, 0.0);
        let to_end = R * 30f64.to_radians().sin();
        assert!((wedge.distance(p) - to_end).abs() < 2e-3, "{}", wedge.distance(p));
        // And from 330 the start cap, at 0, is the nearer: 30 degrees off as well.
        let p = at(330.0, R, 0.0);
        assert!((wedge.distance(p) - to_end).abs() < 2e-3, "{}", wedge.distance(p));
        // Never more room than there is, over the whole set, which crosses both caps.
        let over = crate::overstatement(&wedge, &probes(), &crate::probe_directions(), 0.05, 1.2);
        assert!(over < 1e-6, "claimed {over} mm more room than there was");
    }

    /// **A thin revolve of a line beside the axis through part of a turn is a tube section**:
    /// inside the wedge it reads what the full tube reads, and outside it the nearer cap, each
    /// cap being the wall's own stadium in its half-plane.
    #[test]
    fn a_partial_thin_revolve_is_a_section_of_the_tube() {
        let mut doc = Document::new();
        let line = CurveId::new();
        let (h, t) = (20.0, 4.0);
        doc.curves.insert(
            line,
            Curve::polyline(line, "line", vec![Point3::new(R, 0.0, -h), Point3::new(R, 0.0, h)], false)
                .unwrap(),
        );
        let wall =
            Outline::Wall { curve: line, thickness: lit(t), side: WallSide::Centre, ends: WallEnds::Round };
        let full = revolved(&doc, wall, 360.0, AngleSides::One);
        let part = revolved(&doc, wall, 120.0, AngleSides::One);
        // The wall in its half-plane: a stadium, the segment from (R, -h) to (R, h) grown t/2.
        let stadium = |u: f64, z: f64| ((u - R).hypot(z.abs().max(h) - h)) - 0.5 * t;
        let cap = |angle_to_cap: f64, rho: f64, z: f64| {
            let (u, off) = (rho * angle_to_cap.cos(), rho * angle_to_cap.sin());
            crate::profile::cap_distance(stadium(u, z), off)
        };
        for p in probes() {
            let angle = p.y.atan2(p.x).rem_euclid(TAU).to_degrees();
            let got = part.distance(p);
            if angle <= 120.0 {
                let want = full.distance(p);
                if want >= 0.0 {
                    assert!((got - want).abs() < 1e-9, "in the wedge at {p:?}: {got} against {want}");
                } else {
                    assert!(got <= 0.0 && got >= want - 1e-9, "inside at {p:?}: {got} against {want}");
                }
            } else {
                let rho = p.x.hypot(p.y);
                let to_start = (360.0 - angle).min(angle).to_radians();
                let to_end = (angle - 120.0).min(360.0 - (angle - 120.0)).to_radians();
                let want = cap(to_start, rho, p.z).min(cap(to_end, rho, p.z));
                assert!((got - want).abs() < 1e-9, "outside at {p:?}: {got} against {want}");
            }
        }
    }

    /// **Nothing is refused by the number, and a full turn is the revolve it always was**: 360
    /// and more compile to the same exact solid, and the graph walk agrees with the leaf about
    /// which revolves carry a true distance.
    #[test]
    fn an_angle_of_nothing_is_refused_and_a_full_turn_is_the_revolve_it_was() {
        let mut doc = Document::new();
        let id = circle(&mut doc);
        let build = |angle: f64, sides: AngleSides| {
            let mut graph = Graph::empty();
            let node =
                graph.add(Node::Revolve { rims: Default::default(), outline: Outline::closed(id), frame: Frame::default(), angle: lit(angle), sides });
            (graph, node)
        };
        for (angle, sides) in [(0.0, AngleSides::One), (-90.0, AngleSides::One), (30.0, AngleSides::Two { angle: lit(-30.0) })] {
            let (graph, _) = build(angle, sides);
            let err = GraphField::compile_graph(&doc, &graph, 1e-4).err().expect("refused");
            assert!(matches!(err, CompileError::RevolveTurnsNothing(_)), "{err}");
        }
        let (whole, node) = build(360.0, AngleSides::One);
        let (more, _) = build(400.0, AngleSides::Symmetric);
        let (whole, more) = (
            GraphField::compile_graph(&doc, &whole, 1e-4).unwrap(),
            GraphField::compile_graph(&doc, &more, 1e-4).unwrap(),
        );
        for p in probes() {
            assert!((whole.distance(p) - more.distance(p)).abs() < 1e-12);
        }
        assert_eq!(whole.normalization(), Normalization::Distance);
        let (graph, full_node) = build(360.0, AngleSides::One);
        assert!(carries_true_distance(&doc, &graph, full_node));
        let (graph, part_node) = build(180.0, AngleSides::One);
        assert!(!carries_true_distance(&doc, &graph, part_node));
        let _ = (node, Vector3::<f64>::z());
    }
