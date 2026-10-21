// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a blend across a face its operands share, and what it may claim about itself there.
    use super::*;
    use super::blend::{blend_bound, smooth_min};
    use crate::Normalization;
    use fieldform_scene::{BooleanOp, Continuity, Document, Fillet, Graph, Node, ParamRef};
    use nalgebra::{Point3, Vector3};

    const K: f64 = 5.0;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn slab(graph: &mut Graph, x_lo: f64, x_hi: f64) -> fieldform_scene::NodeId {
        graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit((x_lo + x_hi) / 2.0), lit(0.0), lit(-10.0)]),
            size: [lit(x_hi - x_lo), lit(40.0), lit(20.0)],
            turn: fieldform_scene::no_turn(),
        })
    }

    /// Two boxes overlapping along x, x∈[-40,10] and x∈[-10,40], tops coplanar at z = 0.
    fn two_slabs_sharing_a_top() -> GraphField {
        two_slabs_sharing_a_top_at(Continuity::G1)
    }

    /// The same, at any continuity — the shape G2 is held to as G1 is.
    fn two_slabs_sharing_a_top_at(continuity: Continuity) -> GraphField {
        let mut graph = Graph::empty();
        let a = slab(&mut graph, -40.0, 10.0);
        let b = slab(&mut graph, -10.0, 40.0);
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: a.into(),
            b: b.into(),
            fillet: Fillet { radius: lit(K), continuity },
        });
        GraphField::compile_graph(&Document::new(), &graph, 1e-3).expect("compiles")
    }

    /// Seven copies of one slab at a fifth of its length, so five of them overlap at once and
    /// every top is coplanar — the rounded fold's version of the shared face.
    fn seven_overlapping_slabs() -> GraphField {
        let mut graph = Graph::empty();
        let body = slab(&mut graph, -25.0, 25.0);
        graph.add(Node::Pattern {
            options: Default::default(),
            input: body.into(),
            count: fieldform_scene::ParamRef::Literal(7.0),
            kind: fieldform_scene::PatternKind::Linear { step: [lit(10.0), lit(0.0), lit(0.0)] },
            join: true,
            fillet: Fillet { radius: lit(K), continuity: Continuity::G1 },
        });
        GraphField::compile_graph(&Document::new(), &graph, 1e-3).expect("compiles")
    }

    /// Above the shared top, off every lattice the march's step could land on.
    fn probes_above(x_lo: f64, x_hi: f64) -> Vec<Point3<f64>> {
        let mut points = vec![Point3::new(0.0, 0.0, 3.0)];
        for i in 0..9 {
            for j in 0..6 {
                let (t, u) = (i as f64 / 8.0, j as f64);
                points.push(Point3::new(
                    x_lo + (x_hi - x_lo) * t + 0.137,
                    -11.0 + 2.9 * u,
                    1.713 + 1.37 * u,
                ));
            }
        }
        points
    }

    /// **Documents behaviour; not a bug to remove** (spec §2.1). On the shared top a = b = z, so
    /// the zero is where `k − 2^(1/n)(k − z) = 0`. Kept across the §3 fix because that fix may
    /// only change the magnitude: if this number moves, the zero set moved.
    #[test]
    fn a_shared_face_under_a_blend_rises_by_the_documented_amount() {
        let field = two_slabs_sharing_a_top();
        let (mut lo, mut hi) = (0.0f64, K);
        assert!(field.distance(Point3::new(0.0, 0.0, lo)) < 0.0);
        assert!(field.distance(Point3::new(0.0, 0.0, hi)) > 0.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if field.distance(Point3::new(0.0, 0.0, mid)) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let expected = K * (1.0 - 1.0 / 2f64.sqrt());
        assert!(
            (lo - expected).abs() < 1e-6,
            "the shared top sits at z = {lo}, and the form puts it at {expected}"
        );
    }

    #[test]
    fn a_blend_never_overstates() {
        let field = two_slabs_sharing_a_top();
        let at = Point3::new(0.0, 0.0, 3.0);
        let truth = 3.0 - K * (1.0 - 1.0 / 2f64.sqrt());
        let over = crate::overstatement(
            &field,
            &probes_above(-9.0, 9.0),
            &crate::probe_directions(),
            0.01,
            1.2,
        );
        assert!(
            over < 1e-6,
            "the blend claimed {over} mm more room than it had; at (0,0,3) it reads {} against \
             a true {truth}",
            field.distance(at),
        );
        // The claim that goes with it: a magnitude that had to be divided is not a distance.
        assert_eq!(field.normalization(), Normalization::LowerBound);
    }

    /// G2 on the shared top: `a = b = z`, so the zero is where `k − 2^(1/3)(k − z) = 0` — the
    /// same derivation as G1's with n = 3, and a *different* number, which is the exponent doing
    /// what it should and the field reading it from the fillet rather than assuming two.
    #[test]
    fn a_g2_shared_face_rises_by_its_own_amount() {
        let field = two_slabs_sharing_a_top_at(Continuity::G2);
        let (mut lo, mut hi) = (0.0f64, K);
        assert!(field.distance(Point3::new(0.0, 0.0, lo)) < 0.0);
        assert!(field.distance(Point3::new(0.0, 0.0, hi)) > 0.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if field.distance(Point3::new(0.0, 0.0, mid)) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let expected = K * (1.0 - 2f64.powf(-1.0 / 3.0));
        assert!(
            (lo - expected).abs() < 1e-6,
            "the G2 shared top sits at z = {lo}, and the form puts it at {expected}"
        );
        let g1 = K * (1.0 - 1.0 / 2f64.sqrt());
        assert!((expected - g1).abs() > 0.1, "G2 and G1 must not agree here: {expected} vs {g1}");
    }

    #[test]
    fn a_g2_blend_never_overstates_and_says_it_is_a_lower_bound() {
        let field = two_slabs_sharing_a_top_at(Continuity::G2);
        let over = crate::overstatement(
            &field,
            &probes_above(-9.0, 9.0),
            &crate::probe_directions(),
            0.01,
            1.2,
        );
        assert!(over < 1e-6, "the G2 blend claimed {over} mm more room than it had");
        assert_eq!(field.normalization(), Normalization::LowerBound);
    }

    /// What [`blend_bound`] is claimed to be, held against the raw fillet over a grid of operand
    /// values: the same zero set and sign; never more than a millimetre a millimetre of the
    /// *larger* operand change, which is what makes it a bound whatever the operands' gradients
    /// are doing; between option A's value and the raw one; and the nearer operand itself once
    /// the other is a radius further off.
    #[test]
    fn the_blends_bound_is_the_max_norm_distance_to_its_own_zero_set() {
        let n = 2.0;
        let steps: Vec<f64> = (-150..=200).map(|i| f64::from(i) * 0.1 + 0.0137).collect();
        let mut steepest = 0.0f64;
        for &a in &steps {
            for &b in &steps {
                let (h, f) = (blend_bound(a, b, K, n), smooth_min(a, b, K, n));
                if f.abs() > 1e-9 {
                    assert_eq!(h > 0.0, f > 0.0, "sign at ({a}, {b}): {h} against {f}");
                    assert!(h.abs() <= f.abs() + 1e-12, "longer than the raw form at ({a}, {b})");
                    assert!(
                        h.abs() >= f.abs() / 2f64.sqrt() - 1e-12,
                        "shorter than option A at ({a}, {b}): {h} against {f}"
                    );
                }
                if b >= a.abs() + K {
                    // To the last bit once clear of the line `b = |a| + k` itself, where the
                    // arc ties with the operand and may round a bit low -- never high.
                    assert!(
                        h <= a && a - h < 1e-12,
                        "a radius clear of the join it is the nearer operand: {h} against {a}"
                    );
                }
                for (da, db) in [(0.07, 0.0), (0.0, 0.07), (0.07, 0.07), (0.07, -0.07), (-0.05, 0.07)] {
                    let moved = blend_bound(a + da, b + db, K, n);
                    let step = f64::max(f64::abs(da), f64::abs(db));
                    steepest = steepest.max((moved - h).abs() / step);
                }
            }
        }
        assert!(steepest <= 1.0 + 1e-9, "it rose {steepest} for one of the larger operand's change");
    }

    /// **Option B as the spec words it would have been wrong here**, which is why it was not
    /// built as worded. Above the slabs' shared top and *outside* the zone Z -- both operands
    /// read more than `k` -- the raw fillet returns `min(a, b)`, and the raised face is
    /// 0.293 k nearer than that. "Exact outside Z" overstates by that much, everywhere above
    /// the overlap. The bound reads the distance to the raised face, exactly.
    #[test]
    fn above_a_shared_face_the_blend_is_exact_where_the_raw_form_overstates() {
        let field = two_slabs_sharing_a_top();
        let rise = K * (1.0 - 1.0 / 2f64.sqrt());
        for z in [K + 0.5, 2.0 * K, 3.0 * K] {
            let value = field.distance(Point3::new(0.0, 0.0, z));
            assert!((value - (z - rise)).abs() < 1e-9, "at z = {z} it reads {value}, truly {}", z - rise);
            // The premise: the raw form says `z` there, which is more than there is.
            assert_eq!(smooth_min(z, z, K, 2.0), z);
        }
        let high: Vec<Point3<f64>> =
            probes_above(-9.0, 9.0).into_iter().map(|p| Point3::new(p.x, p.y, p.z + 2.0 * K)).collect();
        let over = crate::overstatement(&field, &high, &crate::probe_directions(), 0.01, 1.2);
        assert!(over < 1e-6, "above the zone it claimed {over} mm more room than it had");
    }

    /// Seven fillets in a chain cost the tracer what none costs, and the chain still never
    /// overstates. Under option A each blend divided by √2 and handed the next a value that
    /// much short, so the chain read 2^(7/2) = 11 times short and was marched accordingly:
    /// 27 million readings of a scene that took 4 million unioned.
    #[test]
    fn a_chain_of_blends_is_not_short_by_a_divisor_a_link() {
        let mut graph = Graph::empty();
        let ball = |graph: &mut Graph, x: f64| {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(0.0)]),
                radius: lit(9.0),
            })
        };
        let mut joined = ball(&mut graph, 0.0);
        for i in 1..8 {
            let next = ball(&mut graph, 15.0 * f64::from(i));
            joined = graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: joined.into(),
                b: next.into(),
                fillet: Fillet { radius: lit(3.0), continuity: Continuity::G1 },
            });
        }
        let doc = Document::new();
        assert_eq!(step_multiplier(&doc, &graph), 1.0);
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        // Beside the first ball and far from every join, the chain reads that ball's own
        // distance -- through seven blends, to the last bit.
        let beside = Point3::new(-20.0, 3.0, 1.0);
        assert_eq!(field.distance(beside), (beside - Point3::origin()).norm() - 9.0);
        // And nowhere does it claim more room than there is.
        let probes: Vec<Point3<f64>> = (0..60)
            .map(|i| {
                let t = f64::from(i);
                Point3::new(-12.0 + 2.13 * t, 10.5 + 3.0 * (t * 0.7).sin(), 4.0 * (t * 0.31).cos())
            })
            .collect();
        let over = crate::overstatement(&field, &probes, &crate::probe_directions(), 0.01, 1.2);
        assert!(over < 1e-6, "the chain claimed {over} mm more room than it had");
    }

    /// A body of two blended slabs with a boss blended on and a bore blended out — the bore
    /// through the slabs' overlap, off-centre so nothing is aligned with anything.
    fn drilled(critical: Option<f64>) -> (Graph, fieldform_scene::NodeId) {
        let mut graph = Graph::empty();
        let a = slab(&mut graph, -40.0, 10.0);
        let b = slab(&mut graph, -10.0, 40.0);
        let fillet = |k: f64| Fillet { radius: lit(k), continuity: Continuity::G1 };
        let body = graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet: fillet(3.0) });
        let boss = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(9.0), lit(3.0), lit(1.0)]),
            radius: lit(7.0),
        });
        let bossed =
            graph.add(Node::Blend { op: BooleanOp::Union, a: body.into(), b: boss.into(), fillet: fillet(4.0) });
        let bore = graph.add(Node::Cylinder { edge: Default::default(),
            ends: fieldform_scene::Ends::between(
                fieldform_scene::Place::At([lit(1.5), lit(-2.0), lit(-30.0)]),
                fieldform_scene::Place::At([lit(1.5), lit(-2.0), lit(20.0)]),
            ),
            radius: lit(4.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Difference,
            a: bossed.into(),
            b: bore.into(),
            fillet: fillet(2.5),
        });
        if let Some(clearance) = critical {
            graph.mark_critical(fieldform_scene::Critical {
                clearance: lit(clearance),
                ..fieldform_scene::Critical::new(bore)
            });
        }
        (graph, bore)
    }

    /// Points scattered through the drilled body's neighbourhood, off every lattice.
    fn scattered(n: usize) -> Vec<Point3<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64 + 0.37;
                Point3::new(
                    1.5 + 16.0 * (t * 0.618).sin() * (t * 0.113).cos(),
                    -2.0 + 16.0 * (t * 0.271).cos(),
                    -6.0 + 14.0 * (t * 0.173).sin(),
                )
            })
            .collect()
    }

    /// **What "exact" means for a protected face** (spec §6.1, and the proof beside
    /// `taper_ratio`). Within the clearance band every fillet's radius is nought, so:
    /// the part is solid exactly where the hard part is, at every point of the band — the
    /// surface is the hard boolean's, air side included — and wherever the point is not inside
    /// *both* operands of a blend the value is the hard one's over a fixed positive constant.
    ///
    /// Not `==` on the value, which is what the spec's first draft asked for: a tapering blend
    /// divides by a constant so that it never overstates, and `x / D · D` is not `x` in floating
    /// point. The zero set is what a bore's diameter is made of, and that is compared exactly.
    #[test]
    fn a_protected_face_is_where_the_hard_boolean_puts_it() {
        let doc = Document::new();
        for clearance in [0.0, 0.75] {
            let (graph, bore) = drilled(Some(clearance));
            let part = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
            let hard = GraphField::compile_graph(&doc, &graph.hardened(None), 1e-3).unwrap();
            let own =
                GraphField::compile_graph(&doc, &graph.subtree(bore).unwrap(), 1e-3).unwrap();
            let mut in_band = 0;
            for p in scattered(60_000) {
                if own.distance(p).abs() > clearance {
                    continue;
                }
                in_band += 1;
                let (f, h) = (part.distance(p), hard.distance(p));
                assert_eq!(
                    f < 0.0,
                    h < 0.0,
                    "clearance {clearance}: at {p:?} the part reads {f} and the hard part {h}"
                );
                assert_eq!(f == 0.0, h == 0.0);
            }
            // The band of a zero clearance is a surface, which scattered points do not land on;
            // that case is carried by the walk along the bore wall below.
            assert!(clearance == 0.0 || in_band > 500, "only {in_band} points fell in the band");

            // Down the bore's own wall, on its zero set to the last bit the cylinder allows.
            for i in 0..400 {
                let t = i as f64 * 0.0157;
                let on_wall =
                    Point3::new(1.5 + 4.0 * t.cos(), -2.0 + 4.0 * t.sin(), -19.0 + 0.045 * i as f64);
                let (f, h) = (part.distance(on_wall), hard.distance(on_wall));
                assert_eq!(f < 0.0, h < 0.0, "on the bore wall at {on_wall:?}: {f} against {h}");
                assert!(f.abs() <= h.abs() + 1e-15, "the wall moved: {f} against {h}");
            }
        }
    }

    /// The same, said of the band rather than the face: with a clearance, the part is the hard
    /// part across the whole land, and an unprotected one is not.
    #[test]
    fn clearance_keeps_the_whole_band_as_the_hard_part() {
        let doc = Document::new();
        let (graph, bore) = drilled(Some(1.0));
        let own = GraphField::compile_graph(&doc, &graph.subtree(bore).unwrap(), 1e-3).unwrap();
        let hard = GraphField::compile_graph(&doc, &graph.hardened(None), 1e-3).unwrap();
        let protected = GraphField::compile_graph(&doc, &graph, 1e-3).unwrap();
        let bare = GraphField::compile_graph(&doc, &graph.without_critical(), 1e-3).unwrap();
        let (mut differs_bare, mut differs_protected) = (0, 0);
        for p in scattered(60_000) {
            if own.distance(p).abs() > 1.0 {
                continue;
            }
            let solid = hard.distance(p) < 0.0;
            differs_bare += usize::from((bare.distance(p) < 0.0) != solid);
            differs_protected += usize::from((protected.distance(p) < 0.0) != solid);
        }
        assert_eq!(differs_protected, 0);
        assert!(differs_bare > 20, "the unprotected fillet barely entered the band: {differs_bare}");
    }

    /// §6.2: tapering adds a term to the gradient, and the divisor has to pay for it. Asked as
    /// the contract — `overstatement` — and as the rate itself, which is the sharper of the two.
    #[test]
    fn the_tapered_blend_never_overstates() {
        let doc = Document::new();
        let (graph, _) = drilled(Some(0.5));
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        assert_eq!(field.normalization(), Normalization::LowerBound);
        let probes: Vec<Point3<f64>> =
            scattered(900).into_iter().filter(|p| field.distance(*p) > 0.0).collect();
        assert!(probes.len() > 100);
        let over = crate::overstatement(&field, &probes, &crate::probe_directions(), 0.01, 1.2);
        assert!(over < 1e-6, "the tapered part claimed {over} mm more room than it had");

        // And the rate, inside and out, since a shell reads the inside.
        let h = 1e-5;
        let mut steepest: f64 = 0.0;
        for p in scattered(20_000) {
            let d = |o: Vector3<f64>| field.distance(p + o) - field.distance(p - o);
            let g = Vector3::new(d(Vector3::x() * h), d(Vector3::y() * h), d(Vector3::z() * h))
                / (2.0 * h);
            steepest = steepest.max(g.norm());
        }
        assert!(steepest <= 1.0 + 1e-6, "the tapered part's value moves at {steepest} per mm");

        // The same shape with the old `min` at nought would step here: continuity across the
        // band's edge, deep in the overlap where both operands are negative.
        let (graph, bore) = drilled(Some(0.5));
        let own = GraphField::compile_graph(&doc, &graph.subtree(bore).unwrap(), 1e-3).unwrap();
        let mut worst_step: f64 = 0.0;
        for i in 0..2000 {
            let t = i as f64 * 0.00314;
            // Straddling |F_bore| = clearance from inside the bore's wall material.
            let at = |r: f64| Point3::new(1.5 + r * t.cos(), -2.0 + r * t.sin(), -8.0);
            let (inside, outside) = (at(4.5 - 1e-7), at(4.5 + 1e-7));
            assert!((own.distance(inside) - 0.5).abs() < 1e-6);
            worst_step = worst_step.max((field.distance(inside) - field.distance(outside)).abs());
        }
        assert!(worst_step < 1e-6, "the value steps {worst_step} across the band's edge");
    }

    #[test]
    fn a_rounded_pattern_never_overstates() {
        let field = seven_overlapping_slabs();
        let over = crate::overstatement(
            &field,
            &probes_above(-20.0, 80.0),
            &crate::probe_directions(),
            0.01,
            1.2,
        );
        assert!(
            over < 1e-6,
            "the rounded pattern claimed {over} mm more room than it had; at (30,0,3) it reads {}",
            field.distance(Point3::new(30.0, 0.0, 3.0)),
        );
        assert_eq!(field.normalization(), Normalization::LowerBound);
    }
