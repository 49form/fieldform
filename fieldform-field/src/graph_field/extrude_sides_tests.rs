// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of an extrude's start and its sides, through the whole graph.
    use super::*;
    use crate::Normalization;
    use fieldform_scene::{Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Sides, Sweep};
    use nalgebra::{Point3, Vector3};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// A square twenty across, drawn on the ground about the origin.
    fn square(doc: &mut Document) -> CurveId {
        let id = CurveId::new();
        let corners = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)];
        let points = corners.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect();
        doc.curves.insert(id, Curve::polyline(id, "square", points, true).expect("closed"));
        id
    }

    /// The field of the square extruded as told, along its normal or a typed direction.
    fn extruded(sweep: Sweep, start: f64, height: f64, draft: f64, sides: Sides) -> GraphField {
        let mut doc = Document::new();
        let id = square(&mut doc);
        let mut graph = Graph::empty();
        graph.add(Node::Extrude { rims: Default::default(),
            outline: Outline::closed(id),
            sweep,
            start: lit(start),
            height: lit(height),
            draft: lit(draft),
            sides,
            corners: Default::default(),
        });
        GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles")
    }

    fn two(height: f64, draft: f64) -> Sides {
        Sides::Two { height: lit(height), draft: lit(draft) }
    }

    /// Points all round and through a solid forty across and sixty tall about the origin.
    fn probes() -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for i in 0..15 {
            for j in 0..15 {
                for k in 0..21 {
                    out.push(Point3::new(
                        -21.0 + i as f64 * 3.1 + 0.13 * k as f64,
                        -22.0 + j as f64 * 3.2 + 0.07 * i as f64,
                        -31.0 + k as f64 * 3.05 + 0.11 * j as f64,
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

    fn agree(what: &str, a: &GraphField, b: &dyn Fn(Point3<f64>) -> f64) {
        for p in probes() {
            let (got, want) = (a.distance(p), b(p));
            assert!((got - want).abs() < 1e-9, "{what} at {p:?}: {got} against {want}");
        }
    }

    /// **A start is a translation along the sweep and nothing else**: the same solid ten
    /// millimetres further on, with its draft opening from where it now begins. Straight up,
    /// and along a typed lean, where the start is carried along the lean with it.
    #[test]
    fn a_start_moves_the_solid_along_the_sweep_and_changes_nothing_else() {
        for draft in [0.0, 20.0] {
            let at_the_plane = extruded(Sweep::Normal, 0.0, 20.0, draft, Sides::One);
            let begun = extruded(Sweep::Normal, 10.0, 20.0, draft, Sides::One);
            agree(&format!("draft {draft}, upright"), &begun, &|p| {
                at_the_plane.distance(p - Vector3::new(0.0, 0.0, 10.0))
            });
            // Along (1, 0, 1): the start is ten millimetres along that line.
            let lean = Sweep::Toward { xyz: [lit(1.0), lit(0.0), lit(1.0)] };
            let leaning = extruded(lean.clone(), 0.0, 20.0, draft, Sides::One);
            let begun = extruded(lean, 10.0, 20.0, draft, Sides::One);
            let along = Vector3::new(1.0, 0.0, 1.0).normalize() * 10.0;
            agree(&format!("draft {draft}, leaning"), &begun, &|p| leaning.distance(p - along));
        }
    }

    /// **Two sides alike are the symmetric extrude of twice the depth**, pointwise, drafted
    /// or not -- which is what makes `Symmetric` a spelling of `Two` rather than a third
    /// solid.
    #[test]
    fn two_equal_sides_are_the_symmetric_extrude_of_twice_the_depth() {
        for draft in [0.0, 15.0, -10.0] {
            let both = extruded(Sweep::Normal, 0.0, 40.0, draft, Sides::Symmetric);
            let two_sided = extruded(Sweep::Normal, 0.0, 20.0, draft, two(20.0, draft));
            agree(&format!("draft {draft}"), &two_sided, &|p| both.distance(p));
        }
    }

    /// **Two sides of different depths and no draft are a box**, to the kernel's own exact
    /// box at every probe, and a distance -- the one join two extrudes and a union could not
    /// make, since they would meet at a face `min` reads nought through.
    #[test]
    fn two_sides_of_different_depths_are_one_box_and_a_distance() {
        let boss = extruded(Sweep::Normal, 0.0, 20.0, 0.0, two(5.0, 0.0));
        assert_eq!(boss.normalization(), Normalization::Distance);
        agree("a box 20 up and 5 down", &boss, &|p| {
            box_distance(p, [-10.0, -10.0, -5.0], [10.0, 10.0, 20.0])
        });
        // Off the surfaces and the creases, where a distance's gradient is one.
        let inside: Vec<Point3<f64>> = probes()
            .into_iter()
            .filter(|p| {
                boss.distance(*p).abs() > 0.5
                    && (p.z - 20.0).abs() > 0.5
                    && (p.z + 5.0).abs() > 0.5
                    && (p.x.abs() - 10.0).abs() > 0.5
                    && (p.y.abs() - 10.0).abs() > 0.5
            })
            .collect();
        let deviation = crate::gradient_deviation(&boss, &inside, 1e-4);
        assert!(deviation < 1e-6, "unit gradient off by {deviation}");

        // And a negative depth runs side one the other way, side two up.
        let flipped = extruded(Sweep::Normal, 0.0, -20.0, 0.0, two(5.0, 0.0));
        agree("the same box the other way up", &flipped, &|p| {
            box_distance(p, [-10.0, -10.0, -20.0], [10.0, 10.0, 5.0])
        });
    }

    /// **Two sides drafted differently never claim more room than there is**, upright and
    /// leaning, with a start: the bound is the steeper side's, so the shallower side reads
    /// short and neither reads long. The control is the same solid drafted alike, which the
    /// symmetric case already holds.
    #[test]
    fn two_sides_drafted_differently_never_overstate() {
        for (what, sweep) in [
            ("upright", Sweep::Normal),
            ("leaning", Sweep::Toward { xyz: [lit(0.4), lit(0.2), lit(1.0)] }),
        ] {
            let solid = extruded(sweep, 4.0, 20.0, 25.0, two(12.0, -8.0));
            assert_eq!(solid.normalization(), Normalization::LowerBound);
            let over =
                crate::overstatement(&solid, &probes(), &crate::probe_directions(), 0.05, 1.2);
            assert!(over < 1e-6, "{what}: claimed {over} mm more room than there was");
            assert!(probes().iter().any(|p| solid.distance(*p) < 0.0), "{what}: nothing inside");
        }
        // Each side's draft shows on its own side, measured from the start plane at z = 4:
        // eleven down, closing in at eight degrees, the wall is 11 tan 8 = 1.55 mm inside the
        // drawn square; nineteen up, opening at twenty-five, it is 8.86 mm outside it.
        let upright = extruded(Sweep::Normal, 4.0, 20.0, 25.0, two(12.0, -8.0));
        let closing = upright.distance(Point3::new(9.0, 0.0, 4.0 - 11.0));
        assert!(closing > 0.0, "1 mm inside the drawn wall, on the closed-in side: {closing}");
        let opening = upright.distance(Point3::new(13.0, 0.0, 4.0 + 19.0));
        assert!(opening < 0.0, "3 mm outside the drawn wall, on the opened-out side: {opening}");
    }
