// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The compiled graph's tests: each node's field, normal, interval, extent and divisor.
    use super::*;
    use super::blend::{blend_bound, smooth_min};
    use crate::{CageField, Extent, Normalization, PatternLayout};
    use nalgebra::{Rotation3, Vector3};

    /// One body ringed about an axis, and the same body written out as placements and unioned.
    ///
    /// The union is ground truth by construction: it is what a pattern *means*. `at_x` puts the
    /// axis somewhere other than through the body, which is the case the fold used to get wrong.
    fn ring_pair(at_x: f64, count: u32) -> (GraphField, GraphField) {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let ball = |graph: &mut fieldform_scene::Graph| {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(20.0),
            })
        };

        let mut folded = fieldform_scene::Graph::empty();
        let body = ball(&mut folded);
        folded.add(Node::Pattern {
            options: Default::default(),
            input: body.into(),
            count: fieldform_scene::ParamRef::Literal(f64::from(count)),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::at_along(
                    fieldform_scene::Axis::Z,
                    [lit(at_x), lit(0.0), lit(0.0)],
                ),
            },
            join: true,
            fillet: Fillet::hard(),
        });

        let mut spelled = fieldform_scene::Graph::empty();
        let body = ball(&mut spelled);
        let mut copies = vec![body];
        for i in 1..count {
            copies.push(spelled.add(Node::Transform {
                input: body.into(),
                translation: [lit(0.0); 3],
                rotation: [lit(0.0), lit(0.0), lit(360.0 * f64::from(i) / f64::from(count))],
                scale: lit(1.0),
                about: fieldform_scene::Place::At([lit(at_x), lit(0.0), lit(0.0)]),
            }));
        }
        spelled.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: copies.into_iter().map(Into::into).collect(),
            });

        (
            GraphField::compile_graph(&doc, &folded, 1e-3).expect("the fold compiles"),
            GraphField::compile_graph(&doc, &spelled, 1e-3).expect("the placements compile"),
        )
    }

    /// Whatever a join divides its value by, the tracer is told.
    ///
    /// The shading march scales what it reads by `step_multiplier`, and the step budget is bought
    /// with it. A rounded pattern of six divides by the square root of however many copies its
    /// window sums; left out of the multiplier, a fillet of a thousandth of a millimetre turned
    /// the whole body dark, which is how this was reported. A hard join divides by nothing and
    /// must cost nothing.
    #[test]
    fn a_rounded_join_is_paid_for_in_steps() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let ring = |k: f64| {
            let mut graph = fieldform_scene::Graph::empty();
            let body = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(20.0),
            });
            let kind = fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::at_along(
                    fieldform_scene::Axis::Z,
                    [lit(35.0), lit(0.0), lit(0.0)],
                ),
            };
            let reach = pattern_reach(&doc, &graph, body, &kind, 6, k);
            graph.add(Node::Pattern {
                options: Default::default(),
                input: body.into(),
                count: fieldform_scene::ParamRef::Literal(6.0),
                kind,
                join: true,
                fillet: Fillet { radius: lit(k), continuity: fieldform_scene::Continuity::G1 },
            });
            (graph, reach)
        };
        let (hard, _) = ring(0.0);
        assert_eq!(step_multiplier(&doc, &hard), 1.0);
        for k in [0.001, 5.0] {
            let (rounded, reach) = ring(k);
            let divisor = pattern_join_divisor(reach, 6, 0, 2.0, 0.0);
            assert!(divisor > 1.4, "the premise: this ring's window sums at least two copies");
            assert_eq!(step_multiplier(&doc, &rounded), divisor, "at a radius of {k}");
        }

        let pair = |k: f64| {
            let mut graph = fieldform_scene::Graph::empty();
            let ball = |graph: &mut fieldform_scene::Graph, x: f64| {
                graph.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(0.0)]),
                    radius: lit(20.0),
                })
            };
            let (a, b) = (ball(&mut graph, 0.0), ball(&mut graph, 30.0));
            graph.add(Node::Blend {
                op: fieldform_scene::BooleanOp::Union,
                a: a.into(),
                b: b.into(),
                fillet: Fillet { radius: lit(k), continuity: fieldform_scene::Continuity::G1 },
            });
            graph
        };
        assert_eq!(step_multiplier(&doc, &pair(0.0)), 1.0);
        // A circular blend is bounded without dividing (`blend_bound`), so it costs the tracer
        // nothing -- which is the point of it, since what it cost used to compound down a
        // chain. A rounded *pattern* still divides, above.
        assert_eq!(step_multiplier(&doc, &pair(5.0)), 1.0);
    }

    /// **Rounding a fold costs nothing in the field the fold already was.**
    ///
    /// D8 asks every source whether it still moves at a millimetre per millimetre, and the
    /// honest form of that question here is comparative. A union of copies does not have a
    /// unit gradient everywhere and never did: at the medial surface between two copies the
    /// nearest one changes and the gradient turns over, which is a fact about the shape rather
    /// than a fault in the field — a ring of six spheres reads 0.112 off unity with a hard
    /// fold, measured. So the property worth holding is that the rounding does not make it any
    /// worse, and the two numbers coming out *equal* is what says the smooth minimum is
    /// composing distances rather than bending them.
    #[test]
    fn rounding_a_fold_leaves_its_gradient_where_the_union_had_it() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        // The compiled ring, and what its value was divided by. A rounded fold divides by
        // W^(1/n) so as never to overstate (blend-integrity-spec §3 A), which puts its gradient
        // at 1/W^(1/n) by construction. The property here is about the form underneath — that
        // the smooth minimum composes distances rather than bending them — so it is measured
        // with the division taken back out.
        struct Undivided(GraphField, f64);
        impl ScalarField for Undivided {
            fn distance(&self, p: Point3<f64>) -> f64 {
                self.0.distance(p) * self.1
            }
        }
        let ring = |fillet: Fillet| {
            let mut graph = fieldform_scene::Graph::empty();
            let body = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(20.0),
            });
            let kind = fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::at_along(
                    fieldform_scene::Axis::Z,
                    [lit(35.0), lit(0.0), lit(0.0)],
                ),
            };
            let k = fillet.radius.resolve(&|_| None).expect("a literal");
            let divisor = if k > 0.0 {
                pattern_join_divisor(pattern_reach(&doc, &graph, body, &kind, 6, k), 6, 0, 2.0, 0.0)
            } else {
                1.0
            };
            graph.add(Node::Pattern { options: Default::default(), input: body.into(), count: fieldform_scene::ParamRef::Literal(6.0), kind, join: true, fillet });
            Undivided(GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles"), divisor)
        };

        let points: Vec<Point3<f64>> = (0..4000)
            .map(|i| {
                let t = f64::from(i) / 4000.0;
                let a = t * std::f64::consts::TAU;
                let r = 70.0 + 40.0 * (t * 13.0).sin();
                Point3::new(35.0 + r * a.cos(), r * a.sin(), 40.0 * (t * 7.0).cos())
            })
            .collect();

        let hard = crate::gradient_deviation(&ring(Fillet::hard()), &points, 1e-4);
        let rounded = crate::gradient_deviation(&ring(Fillet::of(lit(8.0))), &points, 1e-4);
        assert!(
            rounded <= hard + 1e-6,
            "rounding moved the gradient from {hard} off unity to {rounded}"
        );
    }

    /// **A rounded fold is the n-ary smooth minimum of the copies it folds.**
    ///
    /// Ground truth is `smooth_min` applied to every instance at once, from the same function
    /// the blend block uses — not the fold's own arithmetic restated, which would only check
    /// that the loop agrees with itself.
    ///
    /// A row rather than a ring, so the instances that matter are few and the ones outside the
    /// window contribute nothing to either side. Spheres pitched closer than their diameter, so
    /// they actually overlap and there is a seam to round.
    #[test]
    fn a_rounded_fold_blends_its_copies_the_way_a_blend_block_would() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let count = 5u32;
        let pitch = 30.0;
        let ball = 20.0;
        let k = 6.0;

        let mut graph = fieldform_scene::Graph::empty();
        let body = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(ball),
        });
        let kind = fieldform_scene::PatternKind::Linear {
            step: [lit(pitch), lit(0.0), lit(0.0)],
        };
        graph.add(Node::Pattern {
            options: Default::default(),
            input: body.into(),
            count: fieldform_scene::ParamRef::Literal(f64::from(count)),
            kind: kind.clone(),
            join: true,
            fillet: Fillet::of(lit(k)),
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        // What the fold divides its value by so as never to overstate (blend-integrity-spec
        // §3 A). The form below is the *undivided* one, which is what decides the shape, so
        // the comparison is made with the division taken back out.
        let divisor = pattern_join_divisor(
            pattern_reach(&doc, &graph, body, &kind, count, k),
            count as usize,
            0,
            2.0,
            0.0,
        );

        // Every instance, by hand, combined by the n-ary form the fold is meant to be.
        let honestly = |p: Point3<f64>| {
            let mut least = f64::INFINITY;
            let mut inside = 0.0f64;
            for i in 0..count {
                let centre = Point3::new(pitch * f64::from(i), 0.0, 0.0);
                let d = (p - centre).norm() - ball;
                least = least.min(d);
                inside += (k - d).max(0.0).powi(2);
            }
            k.max(least) - inside.sqrt()
        };

        let mut worst: f64 = 0.0;
        for i in 0..4000 {
            let t = f64::from(i) / 4000.0;
            let p = Point3::new(
                -30.0 + t * 200.0,
                26.0 * (t * 23.0).sin(),
                22.0 * (t * 11.0).cos(),
            );
            worst = worst.max((field.distance(p) * divisor - honestly(p)).abs());
        }
        assert!(worst < 1e-9, "the rounded fold differs from its own copies by {worst} mm");

        // And it is a rounding rather than a shift: at the seam between two copies the rounded
        // fold sits *below* the hard one, by more than nothing and by no more than the radius.
        let mut hard = fieldform_scene::Graph::empty();
        let body = hard.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(ball),
        });
        hard.add(Node::Pattern {
            options: Default::default(),
            input: body.into(),
            count: fieldform_scene::ParamRef::Literal(f64::from(count)),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(pitch), lit(0.0), lit(0.0)],
            },
            join: true,
            fillet: Fillet::hard(),
        });
        let plain = GraphField::compile_graph(&doc, &hard, 1e-3).expect("compiles");
        let seam = Point3::new(pitch * 0.5, ball * 0.9, 0.0);
        let (rounded, sharp) = (field.distance(seam) * divisor, plain.distance(seam));
        assert!(
            rounded < sharp - 1e-6 && rounded > sharp - k - 1e-9,
            "at the seam the rounded fold read {rounded} against {sharp} hard, radius {k}"
        );
    }

    /// **A rounded fold over a centred row with copies left out joins exactly the copies that are
    /// there.** The sum a rounded fold accumulates counts every copy it walks, so a copy that is
    /// skipped must be missing from the sum as well as from the minimum — a skipped copy still
    /// summed would pull the seam toward a ball that is not drawn. Ground truth is the n-ary form
    /// over the copies listed by hand, from the definition, with the divisor the layout asks for.
    #[test]
    fn a_rounded_fold_joins_only_the_copies_that_are_held() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let (count, pitch, ball, k) = (7u32, 30.0, 20.0, 6.0);
        let mut graph = fieldform_scene::Graph::empty();
        let body = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(ball),
        });
        let node = graph.add(Node::Pattern {
            input: body.into(),
            count: lit(f64::from(count)),
            kind: fieldform_scene::PatternKind::Linear { step: [lit(pitch), lit(0.0), lit(0.0)] },
            join: false,
            fillet: Fillet::of(lit(k)),
            options: fieldform_scene::PatternOptions {
                symmetric: true,
                skip: vec![1, 5],
                ..Default::default()
            },
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        let layout = PatternLayout::of(&doc, &graph, node).expect("a layout");
        // Seven centred copies at −3 … +3 gaps; the middle one stands on the body and the block
        // does not hold the body, and copies 1 and 5 are skipped: four copies remain.
        assert_eq!(layout.held(), 4, "{:?}", layout.excluded);
        let divisor = pattern_join_divisor(
            pattern_reach_in(&doc, &graph, body, &layout, k),
            layout.total as usize,
            layout.left_out(),
            2.0,
            0.0,
        );
        let held: Vec<f64> = (0..count)
            .filter(|i| ![1, 3, 5].contains(i))
            .map(|i| pitch * (f64::from(i) - 3.0))
            .collect();
        let honestly = |p: Point3<f64>| {
            let (mut least, mut inside) = (f64::INFINITY, 0.0f64);
            for x in &held {
                let d = (p - Point3::new(*x, 0.0, 0.0)).norm() - ball;
                least = least.min(d);
                inside += (k - d).max(0.0).powi(2);
            }
            k.max(least) - inside.sqrt()
        };
        let mut worst: f64 = 0.0;
        for i in 0..4000 {
            let t = f64::from(i) / 4000.0;
            let p = Point3::new(
                -110.0 + t * 220.0,
                26.0 * (t * 23.0).sin(),
                22.0 * (t * 11.0).cos(),
            );
            worst = worst.max((field.distance(p) * divisor - honestly(p)).abs());
        }
        assert!(worst < 1e-9, "the rounded fold differs from the copies it holds by {worst} mm");
    }

    /// **A block that holds only its copies is exactly those copies.**
    ///
    /// Instance zero is the body where it already is, and a block that holds it consumes the
    /// body -- which takes the body out of the list, since a root is a node nothing consumes.
    /// The Mirror rule, applied to the block beside it (D9). So the ground truth here is the
    /// placements *one* to `count - 1`, unioned, and the body must be absent from the fold
    /// rather than merely present twice.
    #[test]
    fn a_pattern_holding_only_its_copies_leaves_the_body_out() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let count = 8u32;
        let at_x = 60.0;

        let mut folded = fieldform_scene::Graph::empty();
        let body = folded.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(12.0),
        });
        folded.add(Node::Pattern {
            options: Default::default(),
            input: body.into(),
            count: fieldform_scene::ParamRef::Literal(f64::from(count)),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::at_along(
                    fieldform_scene::Axis::Z,
                    [lit(at_x), lit(0.0), lit(0.0)],
                ),
            },
            join: false,
            fillet: Fillet::hard(),
        });

        let mut spelled = fieldform_scene::Graph::empty();
        let body = spelled.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(12.0),
        });
        let mut copies: Vec<fieldform_scene::NodeId> = Vec::new();
        for i in 1..count {
            copies.push(spelled.add(Node::Transform {
                input: body.into(),
                translation: [lit(0.0); 3],
                rotation: [lit(0.0), lit(0.0), lit(360.0 * f64::from(i) / f64::from(count))],
                scale: lit(1.0),
                about: fieldform_scene::Place::At([lit(at_x), lit(0.0), lit(0.0)]),
            }).into());
        }
        spelled.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: copies.into_iter().map(Into::into).collect(),
            });

        let a = GraphField::compile_graph(&doc, &folded, 1e-3).expect("the fold compiles");
        let b = GraphField::compile_graph(&doc, &spelled, 1e-3).expect("the copies compile");
        let mut worst: f64 = 0.0;
        for i in 0..3000 {
            let t = f64::from(i) / 3000.0;
            let angle = t * std::f64::consts::TAU;
            let r = at_x + 20.0 * (t * 37.0).sin();
            let p = Point3::new(
                at_x + r * angle.cos(),
                r * angle.sin(),
                11.0 * (t * 17.0).cos(),
            );
            worst = worst.max((a.distance(p) - b.distance(p)).abs());
        }
        assert!(worst < 1e-9, "copies-only differs from its own copies by {worst} mm");

        // And the body really is gone from it: at the body's own centre the fold reads the
        // distance to the *nearest copy*, not zero-minus-the-radius.
        let middle = Point3::origin();
        assert!(
            a.distance(middle) > 0.0,
            "the body is still in a block that says it holds only the copies"
        );
    }

    /// **The fold is measured from the body, not from the world origin.**
    ///
    /// Reported as a sphere patterned 24 times about x = 125 that stopped looking like spheres.
    /// It was not a shader fault and not the window's width: the window was centred on the
    /// frame's zero angle while the body sat twelve cells away from it, so with a window four
    /// cells wide the instance that owned each point was never tried. Moving the axis out is
    /// what exposes it, which is why the axis moves here and the body does not.
    #[test]
    fn a_ring_folds_to_the_same_field_wherever_its_axis_is() {
        for at_x in [0.0, 30.0, 125.0, 400.0] {
            let (folded, spelled) = ring_pair(at_x, 24);
            let mut worst: f64 = 0.0;
            for i in 0..2000 {
                let t = f64::from(i) / 2000.0;
                let angle = t * std::f64::consts::TAU;
                let r = at_x + 30.0 * (t * 37.0).sin();
                let p = Point3::new(
                    at_x + r * angle.cos(),
                    r * angle.sin(),
                    18.0 * (t * 17.0).cos(),
                );
                worst = worst.max((folded.distance(p) - spelled.distance(p)).abs());
            }
            assert!(
                worst < 1e-9,
                "a ring about x = {at_x} differs from its own placements by {worst} mm"
            );
        }
    }

    /// The same for a row, where the body's offset is a distance along the step rather than an
    /// angle about an axis. A lug 300 mm along a 100 mm step is three cells from the origin.
    #[test]
    fn a_row_folds_to_the_same_field_wherever_its_body_starts() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        for start in [0.0, 100.0, 300.0, 900.0] {
            let sphere = |graph: &mut fieldform_scene::Graph| {
                graph.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([lit(start), lit(0.0), lit(0.0)]),
                    radius: lit(20.0),
                })
            };

            let mut folded = fieldform_scene::Graph::empty();
            let body = sphere(&mut folded);
            folded.add(Node::Pattern {
                options: Default::default(),
                input: body.into(),
                count: fieldform_scene::ParamRef::Literal(5.0),
                kind: fieldform_scene::PatternKind::Linear {
                    step: [lit(100.0), lit(0.0), lit(0.0)],
                },
            
                join: true,
                fillet: Fillet::hard(),
            });

            let mut spelled = fieldform_scene::Graph::empty();
            let body = sphere(&mut spelled);
            let mut copies = vec![body];
            for i in 1..5 {
                copies.push(spelled.add(Node::Transform {
                    input: body.into(),
                    translation: [lit(100.0 * f64::from(i)), lit(0.0), lit(0.0)],
                    rotation: [lit(0.0); 3],
                    scale: lit(1.0),
                    about: fieldform_scene::Place::origin(),
                }));
            }
            spelled.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: copies.into_iter().map(Into::into).collect(),
            });

            let a = GraphField::compile_graph(&doc, &folded, 1e-3).expect("the fold compiles");
            let b =
                GraphField::compile_graph(&doc, &spelled, 1e-3).expect("the placements compile");
            let mut worst: f64 = 0.0;
            for i in 0..2000 {
                let t = f64::from(i) / 2000.0;
                let p = Point3::new(
                    start - 60.0 + t * 620.0,
                    30.0 * (t * 23.0).sin(),
                    25.0 * (t * 11.0).cos(),
                );
                worst = worst.max((a.distance(p) - b.distance(p)).abs());
            }
            assert!(
                worst < 1e-9,
                "a row starting at {start} differs from its own placements by {worst} mm"
            );
        }
    }

    /// Two spheres in the same place, one twice the other, morphed by `t`.
    fn morphed(t: f64) -> Result<GraphField, CompileError> {
        let lit = ParamRef::Literal;
        let mut graph = fieldform_scene::Graph::empty();
        let small = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let large = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(20.0),
        });
        graph.add(Node::Morph { a: small.into(), b: large.into(), t: lit(t) });
        GraphField::compile_graph(&Document::new(), &graph, 1e-4)
    }

    /// The ends of a morph are the two bodies exactly, to the last bit.
    ///
    /// **Checked rather than eyeballed**, because "looks like the sphere" is what a morph that
    /// is subtly wrong everywhere also looks like at the ends. A convex combination with `t = 0`
    /// must not merely approximate `a` — `1.0 * a + 0.0 * b` is `a`, and if it is not, the
    /// arithmetic has grown a term it should not have.
    #[test]
    fn a_morph_at_nought_and_one_is_each_operand_exactly() {
        let lit = ParamRef::Literal;
        let mut plain = fieldform_scene::Graph::empty();
        plain.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let small = GraphField::compile_graph(&Document::new(), &plain, 1e-4).expect("compiles");

        let mut plain = fieldform_scene::Graph::empty();
        plain.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(20.0),
        });
        let large = GraphField::compile_graph(&Document::new(), &plain, 1e-4).expect("compiles");

        let (at_zero, at_one) = (morphed(0.0).expect("compiles"), morphed(1.0).expect("compiles"));
        for i in 0..500 {
            let a = i as f64 * 0.7;
            let p = Point3::new(a.sin() * 30.0, a.cos() * 30.0, (a * 0.3).sin() * 30.0);
            assert_eq!(at_zero.distance(p), small.distance(p), "t = 0 is not `a` at {p:?}");
            assert_eq!(at_one.distance(p), large.distance(p), "t = 1 is not `b` at {p:?}");
        }
    }

    /// Half way between a 10 mm and a 20 mm sphere is the 15 mm one.
    ///
    /// The case a morph is *for*, and the one where the arithmetic is checkable by hand: two
    /// concentric spheres have fields `|p| − 10` and `|p| − 20`, whose average is `|p| − 15`.
    /// So the result is exactly a sphere, not merely something round.
    #[test]
    fn two_concentric_spheres_morph_through_the_sphere_between_them() {
        let half = morphed(0.5).expect("compiles");
        for i in 0..400 {
            let a = i as f64 * 0.9;
            let dir =
                nalgebra::Vector3::new(a.sin(), a.cos(), (a * 0.7).sin()).normalize();
            for r in [1.0f64, 9.0, 15.0, 21.0, 40.0] {
                let p = Point3::from(dir * r);
                assert!(
                    (half.distance(p) - (r - 15.0)).abs() < 1e-9,
                    "at radius {r} the morph reads {} rather than {}",
                    half.distance(p),
                    r - 15.0
                );
            }
        }
    }

    /// A morph's interval contains every value the field takes over the box.
    ///
    /// Checklist item 5. An interval that is too narrow prunes away geometry that is there,
    /// and the failure shows up as a hole somewhere else entirely.
    #[test]
    fn a_morphs_interval_contains_the_field_over_the_box() {
        for t in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
            let field = morphed(t).expect("compiles");
            let (lo, hi) = (Point3::new(-30.0, -30.0, -30.0), Point3::new(30.0, 30.0, 30.0));
            let (i_lo, i_hi) = field.interval(lo, hi);
            for i in 0..25 {
                for j in 0..25 {
                    for k in 0..25 {
                        let p = Point3::new(
                            lo.x + (hi.x - lo.x) * i as f64 / 24.0,
                            lo.y + (hi.y - lo.y) * j as f64 / 24.0,
                            lo.z + (hi.z - lo.z) * k as f64 / 24.0,
                        );
                        let v = field.distance(p);
                        assert!(
                            v >= i_lo - 1e-9 && v <= i_hi + 1e-9,
                            "t = {t}: {v} at {p:?} is outside [{i_lo}, {i_hi}]"
                        );
                    }
                }
            }
        }
    }

    /// A morph's analytic normal agrees with differencing its own distance.
    ///
    /// Checklist item 6, and the reason it is not optional: a hand-derived gradient can be unit
    /// length and point the wrong way, which looks like lighting until somebody exports.
    #[test]
    fn a_morphs_normal_agrees_with_a_central_difference() {
        let field = morphed(0.5).expect("compiles");
        let h = 1e-5;
        let mut worst: f64 = 0.0;
        for i in 0..200 {
            let a = i as f64 * 1.1;
            let dir = nalgebra::Vector3::new(a.sin(), a.cos(), (a * 0.7).sin()).normalize();
            // On the surface of the 15 mm sphere the morph describes.
            let p = Point3::from(dir * 15.0);
            let Some(analytic) = field.normal(p) else { continue };
            let numeric = nalgebra::Vector3::new(
                field.distance(p + nalgebra::Vector3::x() * h)
                    - field.distance(p - nalgebra::Vector3::x() * h),
                field.distance(p + nalgebra::Vector3::y() * h)
                    - field.distance(p - nalgebra::Vector3::y() * h),
                field.distance(p + nalgebra::Vector3::z() * h)
                    - field.distance(p - nalgebra::Vector3::z() * h),
            )
            .normalize();
            worst = worst.max(analytic.angle(&numeric).to_degrees());
        }
        assert!(worst < 0.1, "the analytic normal is {worst} degrees off the differenced one");
    }

    /// The degenerate case is handled rather than assumed away.
    ///
    /// Two spheres facing each other morph to a field whose gradients oppose along the axis
    /// between them, where the combination is near zero and normalizing it returns rounding
    /// noise — the failure that cost the attractor two wrong versions. What is asserted is that
    /// a normal still comes back and is a unit vector, not what direction it points: at a point
    /// where the gradient genuinely vanishes there is no right answer, only a defined one.
    #[test]
    fn a_morph_still_has_a_normal_where_its_gradients_oppose() {
        let lit = ParamRef::Literal;
        let mut graph = fieldform_scene::Graph::empty();
        let left = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(-6.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        let right = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(6.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        graph.add(Node::Morph { a: left.into(), b: right.into(), t: lit(0.5) });
        let field = GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles");

        for z in [-4.0f64, -2.0, 0.0, 2.0, 4.0] {
            let p = Point3::new(0.0, 0.0, z);
            let n = field.normal(p).expect("a normal, even where the gradients cancel");
            assert!(
                (n.norm() - 1.0).abs() < 1e-9,
                "the normal at {p:?} is {} long, so it is noise",
                n.norm()
            );
        }
    }

    /// A driven morph never rises faster than one per millimetre.
    ///
    /// **The whole of what the divisor is for.** `∇M` picks up `(B − A)·∇t` when `t` varies, so
    /// the value climbs faster than a distance can and a ray march trusting it steps through the
    /// surface — the speckle that a wrong bound produces, and which looks like a rendering bug
    /// rather than an arithmetic one.
    ///
    /// Measured along random segments rather than argued: the largest rise per millimetre the
    /// field actually takes, against the 1 it promises.
    #[test]
    fn a_driven_morph_never_rises_faster_than_a_distance_may() {
        let lit = ParamRef::Literal;
        let mut graph = fieldform_scene::Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(30.0),
        });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(46.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        // A tight band, which is the case that makes the divisor large and is exactly what a
        // person reaching for "sharp transition" types.
        let point = graph.add(Node::Point { at: [lit(0.0), lit(0.0), lit(0.0)] });
        let ramp = graph.add(Node::Ramp {
            input: point.into(),
            in_min: lit(8.0),
            in_max: lit(14.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: fieldform_scene::RampShape::Smoother,
        });
        graph.add(Node::Morph { a: ball.into(), b: block.into(), t: ParamRef::Field(ramp) });
        let field = GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles");

        let step = 0.02;
        let mut worst: f64 = 0.0;
        for i in 0..3000 {
            let a = i as f64 * 0.37;
            let dir = nalgebra::Vector3::new(a.sin(), (a * 1.7).cos(), (a * 0.9).sin())
                .try_normalize(1e-12)
                .unwrap_or_else(nalgebra::Vector3::x);
            let from = Point3::from(dir * (5.0 + (a * 0.3).sin().abs() * 25.0));
            let here = field.distance(from);
            let there = field.distance(from + dir * step);
            worst = worst.max((there - here).abs() / step);
        }
        assert!(
            worst <= 1.0 + 1e-3,
            "a driven morph rose at {worst} per millimetre, so the divisor is too small"
        );
    }

    /// A ray along a driven morph's silhouette converges within the budget the multiplier buys.
    ///
    /// **The reported bug, as a property.** A driven morph divides its value to stay a lower
    /// bound; a sphere-tracer steps by what it reads; so the same number that makes the field
    /// honest makes every step shorter, and a fixed budget then covers that much less of the
    /// ray. A ray that runs out is indistinguishable from a ray that hit nothing, so the surface
    /// went missing — along the silhouette first, because that is where a ray runs closest to
    /// the surface for longest.
    ///
    /// The scene is the reported one (`morph-variable2.fieldform`) and the ray is the one that
    /// failed: 0.1 mm inside the limb, straight down the view axis. At 512 steps it never
    /// arrives; at `512 × step_multiplier` it does. Both halves are asserted, because a test
    /// that only checks the second would pass just as well if the budget were irrelevant.
    #[test]
    fn a_ray_along_a_driven_morphs_silhouette_arrives_within_the_budget_the_multiplier_buys() {
        let lit = ParamRef::Literal;
        let mut graph = fieldform_scene::Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(50.0),
        });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(100.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let point = graph.add(Node::Point { at: [lit(0.0), lit(-44.333_342_2), lit(0.0)] });
        let ramp = graph.add(Node::Ramp {
            input: point.into(),
            in_min: lit(17.0),
            in_max: lit(86.602_540_4),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: fieldform_scene::RampShape::Smooth,
        });
        graph.add(Node::Morph { a: ball.into(), b: block.into(), t: ParamRef::Field(ramp) });
        let document = Document::new();
        let field = GraphField::compile_graph(&document, &graph, 0.25).expect("compiles");

        // `march.wgsl`'s loop, in the two constants that decide how far a step goes.
        let eps = 86.602_540_4 * 1e-5;
        let arrives = |budget: u32| {
            let origin = Point3::new(49.9, 0.0, 400.0);
            let dir = nalgebra::Vector3::new(0.0, 0.0, -1.0);
            let mut t = 0.0;
            for _ in 0..budget {
                if t > 900.0 {
                    return false;
                }
                let d = field.distance(origin + dir * t);
                if d < eps {
                    let ahead = field.distance(origin + dir * (t + 4.0 * eps));
                    if ahead < 0.0 || ahead < d {
                        return true;
                    }
                    t += 4.0 * eps;
                    continue;
                }
                t += (d * 0.9).max(eps);
            }
            false
        };

        let multiplier = step_multiplier(&document, &graph);
        assert!(
            multiplier > 4.0,
            "a morph over a 70 mm band across a 173 mm body should cost several times the \
             steps, and this says {multiplier}"
        );
        assert!(
            !arrives(512),
            "this ray used to reach the surface at 512 steps, so the scene no longer \
             reproduces what was reported and the test below proves nothing"
        );
        assert!(
            arrives((512.0 * multiplier).ceil() as u32),
            "the ray still runs out at the budget the multiplier buys, so the silhouette is \
             still missing"
        );
    }

    /// A twist divides, so the step budget is bought against the divided value.
    ///
    /// **This read x1.0 for as long as the twist existed.** `Op::Twist` returns
    /// `inner / stretch`, and `step_multiplier`'s walk had an arm for the attractor's identical
    /// division and none for this one — so a wound body was marched as though its value were a
    /// true distance, which is the arithmetic that lost the morph's silhouette and darkened the
    /// pattern's shading. Asserted against the bound rather than a constant, since the stretch is
    /// read off the body's own extent.
    ///
    /// The control is the same twist at no rate at all: a body wound by nothing is not divided,
    /// and a walk that charged for the node rather than for what it does would fail here.
    #[test]
    fn a_twist_s_division_reaches_the_step_budget() {
        let lit = ParamRef::Literal;
        let document = Document::new();
        let wound = |degrees: f64| {
            let mut graph = fieldform_scene::Graph::empty();
            let bar = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(60.0), lit(60.0), lit(20.0)],
                turn: fieldform_scene::no_turn(),
            });
            let id = graph.add(Node::Twist {
                input: bar.into(),
                degrees_per_mm: lit(degrees),
                frame: fieldform_scene::Frame::default(),
            });
            (graph, id)
        };

        let (still, id) = wound(0.0);
        assert_eq!(
            step_multiplier(&document, &still),
            1.0,
            "a body wound by nothing is divided by nothing"
        );

        let (turning, id_turning) = wound(10.0);
        let stretch = twist_stretch(&document, &turning, id_turning);
        assert!(
            stretch > 1.5,
            "a 60 mm bar wound at 10 degrees a millimetre stretches by x{stretch:.2}, which is \
             not enough of a divisor for this case to be measuring anything"
        );
        assert_eq!(
            step_multiplier(&document, &turning),
            stretch,
            "the step budget is bought against the undivided value"
        );
        let _ = id;
    }

    /// Every divisor along a chain is paid for in steps, not just the largest.
    ///
    /// Each divisor makes its own output 1-Lipschitz again, so they do not compound in the
    /// *value* — but each is a further constant factor between what is read and how far it is
    /// safe to travel, so they do compound in the *step*. A morph of a morph is that much
    /// shorter-stepping again, and a multiplier taking the maximum rather than the product
    /// would under-buy exactly there.
    #[test]
    fn the_step_multiplier_is_the_product_along_a_chain_not_the_largest_link() {
        let lit = ParamRef::Literal;
        let ball = |r: f64| Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(r),
        };
        let driver = |graph: &mut fieldform_scene::Graph| {
            let point = graph.add(Node::Point { at: [lit(0.0); 3] });
            graph.add(Node::Ramp {
                input: point.into(),
                in_min: lit(0.0),
                in_max: lit(60.0),
                out_min: lit(0.0),
                out_max: lit(1.0),
                shape: fieldform_scene::RampShape::Smooth,
            })
        };
        let document = Document::new();

        let mut one = fieldform_scene::Graph::empty();
        let a = one.add(ball(30.0));
        let b = one.add(ball(40.0));
        let r = driver(&mut one);
        let inner = one.add(Node::Morph { a: a.into(), b: b.into(), t: ParamRef::Field(r) });
        let single = step_multiplier(&document, &one);

        // The same morph again, with the first as an operand.
        let mut two = one.clone();
        let c = two.add(ball(50.0));
        let r2 = driver(&mut two);
        two.add(Node::Morph { a: inner.into(), b: c.into(), t: ParamRef::Field(r2) });
        let stacked = step_multiplier(&document, &two);

        assert!(
            stacked > single * 1.5,
            "one driven morph costs x{single:.1} and two stacked cost x{stacked:.1}, so the \
             second is not being paid for"
        );
    }

    /// One ramp drives two sockets at once, and both read it.
    ///
    /// **The case a ramp living inside the row it drove could not express**, which is why
    /// `ramp-decision.md` §7.2 makes it a placeable block and why this is worth asserting rather
    /// than assuming. It falls out of `ParamRef::Field` naming a node — nothing here is new — so
    /// what the test guards is that nothing downstream quietly assumes one driver per ramp: the
    /// slot map, the divisor each consumer computes, and the op list all have to cope with a
    /// node that two others point at.
    #[test]
    fn one_ramp_can_drive_two_sockets_at_once() {
        let lit = ParamRef::Literal;
        let mut graph = fieldform_scene::Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(30.0),
        });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(46.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let point = graph.add(Node::Point { at: [lit(0.0), lit(0.0), lit(-40.0)] });
        let ramp = graph.add(Node::Ramp {
            input: point.into(),
            in_min: lit(10.0),
            in_max: lit(80.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: fieldform_scene::RampShape::Smooth,
        });
        let morph = graph.add(Node::Morph { a: ball.into(), b: block.into(), t: ParamRef::Field(ramp) });
        // The second consumer. An offset takes millimetres where the morph took a fraction, and
        // one ramp answering both is exactly the sharing being tested.
        graph.add(Node::Offset { input: morph.into(), distance: ParamRef::Field(ramp) });

        let document = Document::new();
        let field = GraphField::compile_graph(&document, &graph, 0.25)
            .expect("one ramp feeding two sockets compiles");

        // And it evaluates: a field that compiled but read NaN would pass a smoke test and fail
        // every picture.
        for z in [-40.0, -10.0, 0.0, 10.0, 40.0] {
            let d = field.distance(Point3::new(0.0, 0.0, z));
            assert!(d.is_finite(), "the field read {d} at z = {z}");
        }

        // Both divisors are real. The morph's is the driven one; asking for it also proves the
        // shared node did not confuse which consumer is being priced.
        let divisor = driven_morph_divisor(&document, &graph, morph);
        assert!(
            divisor > 1.0,
            "a driven morph should cost more than one step per step, and this says {divisor}"
        );
    }

    /// A ramp driving a morph past nought-to-one is refused, as a typed amount already was.
    ///
    /// **The rule existed and one path skipped it.** A literal `t` outside `[0, 1]` has always
    /// been refused as a file saying something it cannot mean; the same number arriving from a
    /// ramp went unchecked. `t = 7.6` is not "more B", it is `(1 - 7.6)*A + 7.6*B`, an
    /// extrapolation past both shapes.
    ///
    /// The cost is not only meaning, which is why this was found from a performance report
    /// rather than a wrong picture: a driven morph is divided by `1 + sup|B - A|*L_t`, and `L_t`
    /// is the ramp's output span over its input span. The reported scene mapped a 2 mm band onto
    /// 0.6..7.6 and priced at x165.7 -- every ray step a hundred and sixty-fifth of what it
    /// could be. The same band mapped onto its legal 0..1 prices at x24.5.
    #[test]
    fn a_ramp_driving_a_morph_outside_nought_to_one_is_refused_by_name() {
        let lit = ParamRef::Literal;
        let scene = |out_min: f64, out_max: f64| {
            let mut graph = fieldform_scene::Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(30.0),
            });
            let block = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(46.0); 3],
                turn: fieldform_scene::no_turn(),
            });
            let point = graph.add(Node::Point { at: [lit(0.0); 3] });
            let ramp = graph.add(Node::Ramp {
                input: point.into(),
                in_min: lit(6.0),
                in_max: lit(8.0),
                out_min: lit(out_min),
                out_max: lit(out_max),
                shape: fieldform_scene::RampShape::Linear,
            });
            graph.add(Node::Morph { a: ball.into(), b: block.into(), t: ParamRef::Field(ramp) });
            GraphField::compile_graph(&Document::new(), &graph, 0.25)
        };

        match scene(0.6, 7.6) {
            Err(CompileError::MorphDriverOutOfRange(lo, hi)) => {
                assert!((lo - 0.6).abs() < 1e-9 && (hi - 7.6).abs() < 1e-9, "{lo} to {hi}");
            }
            Err(e) => panic!("refused, but as {e:?} rather than by name"),
            Ok(_) => panic!("a ramp driving a morph to 7.6 compiled, and it should not have"),
        }
        // Backwards is the same range and is fine; the ends are read either way round.
        assert!(scene(1.0, 0.0).is_ok(), "a ramp running one to nought is a morph in reverse");
        assert!(scene(0.0, 1.0).is_ok(), "and the ordinary direction still compiles");
        // A negative end is as meaningless as one past one.
        assert!(matches!(
            scene(-0.2, 1.0),
            Err(CompileError::MorphDriverOutOfRange(..))
        ));
    }

    /// An attractor drawn toward a point pulls, and toward one thing only.
    ///
    /// **A point reaches the warp as a spine of one vertex**, which the nearest-place search
    /// already answers with that vertex — `Spine::nearest` has the case written out and the
    /// shader's `closest_on_curve` falls back to the first point when there is no segment to
    /// walk. So what is worth asserting is not the arithmetic but that the target reaches it:
    /// a body near the point should be drawn toward it and a body beyond the reach should not
    /// move at all.
    #[test]
    fn an_attractor_drawn_toward_a_point_pulls_material_toward_it() {
        let lit = ParamRef::Literal;
        let document = Document::new();
        let ball = |graph: &mut fieldform_scene::Graph| {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(20.0),
            })
        };

        let mut plain = fieldform_scene::Graph::empty();
        ball(&mut plain);
        let plain = GraphField::compile_graph(&document, &plain, 0.25).expect("compiles");

        let mut graph = fieldform_scene::Graph::empty();
        let body = ball(&mut graph);
        let point = graph.add(Node::Point { at: [lit(40.0), lit(0.0), lit(0.0)] });
        graph.add(Node::Attract {
            input: body.into(),
            toward: fieldform_scene::Attractor::Point(point),
            strength: lit(0.8),
            falloff: lit(60.0),
        });
        let pulled = GraphField::compile_graph(&document, &graph, 0.25).expect("compiles");

        // **Compared as surfaces, not as values.** An attractor divides its whole field by its
        // own stretch bound to stay a lower bound, so every reading changes even where the warp
        // moves nothing. Where the surface *is* is the thing a person sees, and the thing the
        // falloff is a claim about.
        let surface_along = |field: &GraphField, dir: Vector3<f64>| {
            let (mut lo, mut hi) = (0.0f64, 300.0f64);
            for _ in 0..80 {
                let mid = (lo + hi) * 0.5;
                match field.distance(Point3::origin() + dir * mid) < 0.0 {
                    true => lo = mid,
                    false => hi = mid,
                }
            }
            (lo + hi) * 0.5
        };

        // Toward the point, the surface has been drawn out past the 20 mm it started at.
        let toward = surface_along(&pulled, Vector3::x());
        assert!(
            toward > surface_along(&plain, Vector3::x()) + 1e-3,
            "toward the point the surface should have moved out, and it is at {toward}"
        );

        // Away from it, past the reach, the surface has not moved at all -- which is what makes
        // the falloff a reach rather than a weight.
        let away = surface_along(&pulled, -Vector3::x());
        assert!(
            (away - surface_along(&plain, -Vector3::x())).abs() < 1e-6,
            "past the reach the surface should be where it was, and it is at {away}"
        );

        // And the point is not a body: it is consumed as the target, so it draws nothing and
        // the part is still one solid.
        assert_eq!(graph.roots().len(), 1, "the point is consumed, not drawn");
    }

    /// A driven morph whose operand has no bounds is refused by name.
    #[test]
    fn a_driven_morph_against_an_unbounded_body_is_refused_by_name() {
        let lit = ParamRef::Literal;
        let mut graph = fieldform_scene::Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(30.0),
        });
        // A half-space, which is the one leaf with no box of its own.
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let point = graph.add(Node::Point { at: [lit(0.0); 3] });
        let ramp = graph.add(Node::Ramp {
            input: point.into(),
            in_min: lit(0.0),
            in_max: lit(40.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: fieldform_scene::RampShape::Smooth,
        });
        graph.add(Node::Morph { a: ball.into(), b: plane.into(), t: ParamRef::Field(ramp) });
        match GraphField::compile_graph(&Document::new(), &graph, 1e-4) {
            Err(CompileError::MorphNeedsBounds) => {}
            Err(e) => panic!("refused, but as {e:?} rather than by name"),
            Ok(_) => panic!("an unbounded operand should have been refused"),
        }
    }

    /// A morph's amount outside nought to one is refused rather than clamped.
    #[test]
    fn a_morph_outside_its_range_is_refused_by_name() {
        for t in [-0.5f64, 1.5] {
            match morphed(t) {
                Err(CompileError::MorphOutOfRange(v)) => assert_eq!(v, t),
                Err(e) => panic!("t = {t} was refused, but as {e:?} rather than by name"),
                Ok(_) => panic!("t = {t} compiled, and it should not have"),
            }
        }
    }

    use crate::gradient_deviation;
    use fieldform_cage::Cage;
    use fieldform_scene::{
        BooleanOp, Continuity, Curve, CurveId, Document, Fillet, LatticePhase, Node, ParamRef,
        Part, PartId,
    };

    /// A blend to a point is a shape above the gap and a phantom below it.
    ///
    /// **Both halves matter and the panel depends on the number between them.** Blending a solid
    /// to a point is what a hard boolean cannot do — `min` and `max` read only sign, and a
    /// point's field never goes negative — but it only works above a threshold. Below it the
    /// field reads exactly nought at the point and nothing else changes: no material, and the
    /// solid still ends where it did. That nought is a tangent touch with no inside, and reading
    /// only the value at the point is how it passes for success.
    ///
    /// The threshold is the distance from the point to the surface, measured at a ratio of
    /// 1.0000 over gaps of 10 to 80 mm by `examples/boolean_with_nothing.rs`. The panel defaults
    /// the radius to half again that, so this asserts both sides of it.
    #[test]
    fn a_blend_reaches_a_point_only_once_its_radius_clears_the_gap() {
        let lit = ParamRef::Literal;
        let (radius, out) = (20.0, 50.0);
        let gap = out - radius;

        let built = |blend: f64| {
            let mut graph = fieldform_scene::Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(radius),
            });
            let point = graph.add(Node::Point { at: [lit(out), lit(0.0), lit(0.0)] });
            graph.add(Node::Blend {
                op: fieldform_scene::BooleanOp::Union,
                a: ball.into(),
                b: point.into(),
                fillet: Fillet { radius: lit(blend), continuity: Continuity::G1 },
            });
            GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
        };
        let at_point = |f: &GraphField| f.distance(Point3::new(out, 0.0, 0.0));

        // Under the gap: the field touches nought at the point and encloses nothing there.
        let touching = built(gap * 0.5);
        assert!(
            at_point(&touching).abs() < 1e-9,
            "the premise: a small radius reads nought at the point, {}",
            at_point(&touching)
        );
        // Just outside the ball on the way to the point, still nothing.
        assert!(
            touching.distance(Point3::new(radius + 1.0, 0.0, 0.0)) > 0.0,
            "a radius under the gap should leave the space between empty"
        );

        // The default the panel picks, which is half again the gap.
        let reaching = built(gap * 1.5);
        assert!(
            at_point(&reaching) < -1e-6,
            "the panel's default should put material at the point, not touch it: {}",
            at_point(&reaching)
        );
        assert!(
            reaching.distance(Point3::new(radius + 1.0, 0.0, 0.0)) < 0.0,
            "and should join the point to the solid rather than leaving a gap"
        );
    }

    /// A square, symmetric about its plane, and the same square built as two extrudes unioned.
    fn opposed_and_symmetric(half: f64, depth: f64) -> (GraphField, GraphField) {
        let mut doc = Document::new();
        let id = CurveId::new();
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "Square".to_string(),
                vec![
                    Point3::new(-half, -half, 0.0),
                    Point3::new(half, -half, 0.0),
                    Point3::new(half, half, 0.0),
                    Point3::new(-half, half, 0.0),
                ],
                true,
            )
            .expect("a closed square"),
        );
        let lit = ParamRef::Literal;
        let mut joined = fieldform_scene::Graph::empty();
        let up = joined.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(depth * 0.5),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        let down = joined.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(-depth * 0.5),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        joined.add(Node::Boolean {
            op: fieldform_scene::BooleanOp::Union,
            inputs: vec![up.into(), down.into()],
        });
        let mut single = fieldform_scene::Graph::empty();
        single.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(depth),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::Symmetric,
            corners: Default::default(),
        });
        (
            GraphField::compile_graph(&doc, &joined, 1e-4).expect("compiles"),
            GraphField::compile_graph(&doc, &single, 1e-4).expect("compiles"),
        )
    }

    /// An extrude taken both ways is the same solid as two extrudes unioned, without the join.
    ///
    /// **Both halves of that matter and neither implies the other.** The same shape is what
    /// makes "both directions" a substitute for building it by hand, and it is checked by sign
    /// rather than by value because `min` is only a bound inside its operands: the two fields
    /// legitimately disagree throughout the material and agree exactly outside it.
    ///
    /// Without the join is the reason to have it. Two solids sharing a face are what `min`
    /// cannot describe — it reads nought right through the middle, which a subtract turns into
    /// a sheet of surface in the open. `examples/symmetric_extrude.rs` prints both tables.
    #[test]
    fn an_extrude_taken_both_ways_is_the_union_without_the_join() {
        let (half, depth) = (20.0, 30.0);
        let (joined, single) = opposed_and_symmetric(half, depth);

        // The join, where the two descriptions part company.
        assert!(
            joined.distance(Point3::origin()).abs() < 1e-9,
            "the premise: the union reads nought at the shared face"
        );
        assert!(
            (single.distance(Point3::origin()) + depth * 0.5).abs() < 1e-6,
            "the symmetric one reads the distance to the nearer end, not nought: {}",
            single.distance(Point3::origin())
        );

        // And the same solid: the sign never differs, and outside it neither do the values.
        let mut outside = 0.0f64;
        for i in 0..25 {
            for j in 0..25 {
                for k in 0..25 {
                    let p = Point3::new(
                        -30.0 + 60.0 * i as f64 / 24.0,
                        -30.0 + 60.0 * j as f64 / 24.0,
                        -25.0 + 50.0 * k as f64 / 24.0,
                    );
                    let (a, b) = (joined.distance(p), single.distance(p));
                    assert_eq!(
                        a > 0.0,
                        b > 0.0,
                        "disagreed about inside at {p:?}: {a} against {b}"
                    );
                    if b > 0.0 {
                        outside = outside.max((a - b).abs());
                    }
                }
            }
        }
        assert!(outside < 1e-9, "outside the solid the two differ by {outside} mm");
    }

    /// An extrude's bounds cover the solid, rather than a box of the right size in the wrong place.
    ///
    /// **This was wrong for every extrude that is not symmetric**, and wrong invisibly: the box
    /// ran `[-h/2, +h/2]` about the profile's plane, which is the correct *size* and half a
    /// depth from the correct place — a leftover from when an extrude straddled its plane. So
    /// the far half of the solid sat outside its own bounds and empty space sat inside them,
    /// and nothing looked obviously broken because the box was never the wrong shape.
    ///
    /// Asserted against the field, not against a number: whatever the box is, every point the
    /// field calls solid has to be inside it.
    #[test]
    fn an_extrudes_bounds_hold_the_solid_it_actually_makes() {
        let (half, depth) = (20.0, 30.0);
        for symmetric in [false, true] {
            let mut doc = Document::new();
            let id = CurveId::new();
            doc.curves.insert(
                id,
                Curve::polyline(
                    id,
                    "Square".to_string(),
                    vec![
                        Point3::new(-half, -half, 0.0),
                        Point3::new(half, -half, 0.0),
                        Point3::new(half, half, 0.0),
                        Point3::new(-half, half, 0.0),
                    ],
                    true,
                )
                .expect("a closed square"),
            );
            let mut graph = fieldform_scene::Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(depth),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::symmetric_if(symmetric),
                corners: Default::default(),
            });
            let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
            let Extent::Bounded { lo, hi } = part_extent(&doc, &graph) else {
                panic!("symmetric = {symmetric}: an extrude is a bounded solid");
            };

            let mut solid = 0usize;
            for i in 0..30 {
                for j in 0..30 {
                    for k in 0..40 {
                        let p = Point3::new(
                            -30.0 + 60.0 * i as f64 / 29.0,
                            -30.0 + 60.0 * j as f64 / 29.0,
                            -40.0 + 80.0 * k as f64 / 39.0,
                        );
                        if field.distance(p) >= 0.0 {
                            continue;
                        }
                        solid += 1;
                        assert!(
                            p.x >= lo.x - 1e-6
                                && p.y >= lo.y - 1e-6
                                && p.z >= lo.z - 1e-6
                                && p.x <= hi.x + 1e-6
                                && p.y <= hi.y + 1e-6
                                && p.z <= hi.z + 1e-6,
                            "symmetric = {symmetric}: material at {p:?} is outside \
                             {lo:?}..{hi:?}"
                        );
                    }
                }
            }
            assert!(solid > 100, "symmetric = {symmetric}: found almost no material to check");
        }
    }

    /// A document holding one 2 mm cube cage and one 100 mm line through the origin along X.
    fn document() -> (Document, fieldform_scene::CageId, CurveId) {
        let mut doc = Document::new();

        let cage = Cage::cube();
        let cage_id = fieldform_scene::CageId::new();
        doc.cages.insert(cage_id, cage);

        let curve_id = CurveId::new();
        let curve = Curve::polyline(
            curve_id,
            "spine",
            vec![Point3::new(-50.0, 0.0, 0.0), Point3::new(50.0, 0.0, 0.0)],
            false,
        )
        .unwrap();
        doc.curves.insert(curve_id, curve);

        (doc, cage_id, curve_id)
    }

    fn part_with(graph: Graph) -> Part {
        Part {
            id: PartId::new(),
            name: "test".into(),
            material: None,
            graph,
            inputs: Vec::new(),
        }
    }

    /// Densely samples a box and returns the extreme values the composed field takes in it.
    fn observed_range(field: &GraphField, lo: Point3<f64>, hi: Point3<f64>) -> (f64, f64) {
        let n = 10;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    let t = |a: f64, b: f64, s: usize| a + (b - a) * s as f64 / n as f64;
                    let p = Point3::new(t(lo.x, hi.x, i), t(lo.y, hi.y, j), t(lo.z, hi.z, k));
                    let d = field.distance(p);
                    min = min.min(d);
                    max = max.max(d);
                }
            }
        }
        (min, max)
    }

    /// Every way of composing two fields has to preserve the containment property, and each
    /// one gets it differently: `min` and `max` carry through monotonically, difference has to
    /// swap the negated operand's endpoints, and a blend has to account for dipping below a
    /// hard union. A sign error in any of them produces a bound that looks plausible and
    /// quietly excludes real geometry.
    #[test]
    fn a_composed_interval_contains_every_value_the_graph_takes() {
        let (doc, cage_id, curve_id) = document();

        let boxes = [
            (Point3::new(-2.0, -2.0, -2.0), Point3::new(2.0, 2.0, 2.0)),
            (Point3::new(4.0, -3.0, -3.0), Point3::new(12.0, 3.0, 3.0)),
            (
                Point3::new(-60.0, -20.0, -20.0),
                Point3::new(60.0, 20.0, 20.0),
            ),
            (Point3::new(30.0, 30.0, 30.0), Point3::new(40.0, 40.0, 40.0)),
        ];

        for op in [
            BooleanOp::Union,
            BooleanOp::Difference,
            BooleanOp::Intersection,
        ] {
            let mut graph = Graph::from_cage(cage_id);
            let cage_node = graph.roots()[0];
            let pipe = graph.add(Node::Pipe {
                curve: curve_id,
                radius: ParamRef::Literal(5.0),
            });
            graph.add(Node::Boolean {
                op,
                inputs: vec![cage_node.into(), pipe.into()],
            });

            let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();
            for (lo, hi) in boxes {
                let (bound_lo, bound_hi) = field.interval(lo, hi);
                let (seen_lo, seen_hi) = observed_range(&field, lo, hi);
                assert!(
                    bound_lo <= seen_lo && bound_hi >= seen_hi,
                    "{op:?}: interval ({bound_lo:.3}, {bound_hi:.3}) fails to contain \
                     observed ({seen_lo:.3}, {seen_hi:.3}) in {lo:?}..{hi:?}",
                );
            }
        }

        // A blend, which is the one that is not simply monotonic.
        let mut graph = Graph::from_cage(cage_id);
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(5.0),
        });
        let _blended = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: cage_node.into(),
            b: pipe.into(),
            fillet: Fillet { radius: ParamRef::Literal(4.0), continuity: Continuity::G1 },
        });

        let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();
        for (lo, hi) in boxes {
            let (bound_lo, bound_hi) = field.interval(lo, hi);
            let (seen_lo, seen_hi) = observed_range(&field, lo, hi);
            assert!(
                bound_lo <= seen_lo && bound_hi >= seen_hi,
                "blend: interval ({bound_lo:.3}, {bound_hi:.3}) fails to contain observed \
                 ({seen_lo:.3}, {seen_hi:.3}) in {lo:?}..{hi:?}",
            );
        }
    }

    /// The property the whole blend rests on: the number the user typed is the radius of the
    /// fillet they get.
    ///
    /// Two perpendicular walls, blended. A rolling ball of radius k tangent to both sits
    /// centred at (k, k), so every point of the fillet is exactly k from there — which is a
    /// stronger statement than a curvature measurement at one point, and it is checked along
    /// the whole profile rather than at the apex.
    ///
    /// The polynomial smooth minimum this replaced fails this test at every point: it measured
    /// 0.707 k by curvature at the apex and 0.854 k by where the surface sat, so there was no
    /// radius to check against.
    #[test]
    fn the_fillet_is_a_circle_of_the_radius_it_was_asked_for() {
        let k = 7.0;
        let n = fieldform_scene::Continuity::G1.exponent();
        for i in 0..=100 {
            // Walk one wall from the tangency point in to the apex.
            let a = k * i as f64 / 100.0;
            // Where the blended surface crosses this line. The search stops at the tangency
            // point rather than running past it: beyond `k` the surface *is* the flat wall,
            // so `f` is zero all the way along it and a wider bracket would converge onto the
            // wall instead of onto the arc.
            let (mut lo, mut hi) = (-4.0 * k, k);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if smooth_min(a, mid, k, n) > 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let b = 0.5 * (lo + hi);
            let from_centre = ((k - a).powi(2) + (k - b).powi(2)).sqrt();
            assert!(
                (from_centre - k).abs() < 1e-9,
                "at a = {a}, the surface sits {from_centre} from the ball centre, not {k}"
            );
        }
    }

    /// Curvature of the zero set of `smooth_min(a, b, k, n) = 0` at a point on it, in the (a, b)
    /// plane — which is the plane of two perpendicular walls, `a = x` and `b = y`.
    fn blend_curvature(k: f64, n: f64, a: f64, b: f64) -> f64 {
        let f = |a: f64, b: f64| smooth_min(a, b, k, n);
        let h = k * 1e-4;
        let fx = (f(a + h, b) - f(a - h, b)) / (2.0 * h);
        let fy = (f(a, b + h) - f(a, b - h)) / (2.0 * h);
        let fxx = (f(a + h, b) - 2.0 * f(a, b) + f(a - h, b)) / (h * h);
        let fyy = (f(a, b + h) - 2.0 * f(a, b) + f(a, b - h)) / (h * h);
        let fxy = (f(a + h, b + h) - f(a + h, b - h) - f(a - h, b + h) + f(a - h, b - h))
            / (4.0 * h * h);
        (fx * fx * fyy - 2.0 * fx * fy * fxy + fy * fy * fxx).abs() / (fx * fx + fy * fy).powf(1.5)
    }

    /// Where the blended surface crosses the line `a = const`, on the arc rather than the wall.
    fn blend_b_at(k: f64, n: f64, a: f64) -> f64 {
        let (mut lo, mut hi) = (-4.0 * k, k);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if smooth_min(a, mid, k, n) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// **What the number typed means, for G1 and G2 alike: reach** (ruled 2026-10-02).
    ///
    /// The zero set leaves face A exactly where `b = k` — `k` from the *other* face — whatever
    /// the exponent, so switching shape never moves where the rounding starts. What *does* change
    /// with the exponent is the bend inside, and it is the closed form `Continuity` quotes:
    /// `2^(1/2 − 1/n)/(n − 1)` times `k` at a right angle, read here off the field itself.
    /// Curvature at the join is what makes G2 G2: a step from nought to 1/k at G1, nearly nothing
    /// at n = 3.
    #[test]
    fn a_g2_fillet_starts_at_the_same_reach_as_g1_and_bends_to_the_closed_form_radius() {
        use fieldform_scene::Continuity::{G1, G2};
        let k = 7.0;
        for continuity in [G1, G2] {
            let n = continuity.exponent();
            // The reach: on the wall (a = 0) the surface is the wall from b = k outward and
            // leaves it at b = k, for any exponent.
            assert!(smooth_min(0.0, k, k, n).abs() < 1e-12, "n = {n}: not on the wall at b = k");
            assert!(smooth_min(0.0, 3.0 * k, k, n).abs() < 1e-12, "n = {n}: off the wall past k");
            assert!(
                smooth_min(0.0, 0.999 * k, k, n) < 0.0,
                "n = {n}: still on the wall inside the reach, so it left late"
            );

            // The tightest bend, on the diagonal, against the closed form.
            let apex = blend_b_at(k, n, k * (1.0 - 2f64.powf(-1.0 / n)));
            assert!(
                (apex - k * (1.0 - 2f64.powf(-1.0 / n))).abs() < 1e-9,
                "n = {n}: the diagonal point is not on the zero set"
            );
            let radius = 1.0 / blend_curvature(k, n, apex, apex);
            let expected = continuity.apex_radius_ratio() * k;
            assert!(
                (radius - expected).abs() < 1e-3 * k,
                "n = {n}: the tightest bend is {radius} mm, and the closed form says {expected}"
            );
        }
        // A circle is the same radius all the way round; a G2 fillet is not, and that is the
        // point of it. Curvature in units of 1/k, taken a hair off the wall: G1 is already at
        // 1/k there, G2 comes to it from nought — near the join the arc is `a ≈ v³/3k²`, so its
        // curvature is about `2v/k²` and goes to nought *linearly* in how far along the wall it
        // has got, which is why the probe has to be very close (0.0144·k along at a = 1e-6·k).
        let curvature_at = |n: f64, a: f64| blend_curvature(k, n, a, blend_b_at(k, n, a)) * k;
        let g1 = curvature_at(2.0, 1e-3 * k);
        assert!((g1 - 1.0).abs() < 1e-2, "G1 is a circle of curvature 1/k: {g1}");
        let (far, near) = (curvature_at(3.0, 1e-3 * k), curvature_at(3.0, 1e-6 * k));
        assert!(near < far && far < 0.5, "G2 curvature should fall toward the join: {far}, {near}");
        assert!(near < 0.05, "G2 should be nearly flat at the join, not 1/k: {near}");
    }

    /// The bound a G2 blend reports, held to what it is claimed to be: one millimetre a
    /// millimetre of the *larger* operand change, in the max-norm, so that a march trusting it
    /// cannot step past the surface whatever the operands' gradients are doing — and the same
    /// sign as the raw fillet, so it moves no surface. At n ≠ 2 it is the raw form over 2^(1/n),
    /// which is sound; this is what says so.
    #[test]
    fn the_g2_blends_bound_moves_no_faster_than_the_larger_operand_and_keeps_the_surface() {
        let n = fieldform_scene::Continuity::G2.exponent();
        let k = 4.0;
        let step = 0.05;
        let values: Vec<f64> = (-120..=160).map(|i| f64::from(i) * 0.1 + 0.0137).collect();
        let mut steepest = 0.0f64;
        for &a in &values {
            for &b in &values {
                let h = blend_bound(a, b, k, n);
                assert_eq!(
                    h < 0.0,
                    smooth_min(a, b, k, n) < 0.0,
                    "({a}, {b}): the bound changed which side of the surface this is"
                );
                for (da, db) in [(step, 0.0), (0.0, step), (step, step), (step, -step)] {
                    let moved = blend_bound(a + da, b + db, k, n);
                    let allowed = f64::max(da.abs(), db.abs());
                    steepest = steepest.max((moved - h).abs() / allowed);
                }
            }
        }
        assert!(steepest <= 1.0 + 1e-9, "the bound rises {steepest} mm per millimetre");
    }

    /// **A chamfer between two square faces cuts each back exactly its leg, with a plane
    /// between**, and its bound neither overstates nor moves the surface. The quadrant
    /// `x ≤ 0, y ≤ 0` intersected with a chamfer of leg `k`: the zero set leaves the face `y = 0`
    /// at `x = −k` and the face `x = 0` at `y = −k`, and between them is the straight bevel
    /// `x + y = −k` — the corner cut at 45°, which is what a drawing means by a chamfer of `k`.
    /// Against the true distance to that cut quadrant, measured by sampling its boundary, the
    /// value never claims more room than there is (`round-chamfer-spec.md` §6).
    #[test]
    fn a_chamfer_cuts_each_square_face_back_its_leg_and_never_overstates() {
        let k = 5.0;
        let n = fieldform_scene::Continuity::Chamfer.exponent();
        let at = |x: f64, y: f64| blend(BooleanOp::Intersection, x, y, k, n, k, 0.0);
        // The legs, a hair inside each face: still the face's own material past the leg, cut
        // away short of it. (On the face itself the value is nought past the leg, as a face's is.)
        let skin = 1e-3;
        assert!(at(-k - 0.01, -skin) < 0.0 && at(-k + 0.01, -skin) > 0.0, "the leg along y = 0");
        assert!(at(-skin, -k - 0.01) < 0.0 && at(-skin, -k + 0.01) > 0.0, "the leg along x = 0");
        // Straight between them: every point of the bevel is on the surface.
        for i in 0..=20 {
            let x = -k * i as f64 / 20.0;
            let y = -k - x;
            assert!(at(x, y).abs() < 1e-9, "({x}, {y}) is on the bevel and reads {}", at(x, y));
        }
        // The true distance to the cut quadrant: the boundary sampled densely, the sign by
        // membership.
        let boundary: Vec<(f64, f64)> = (0..=4000)
            .map(|i| {
                let s = i as f64 / 4000.0 * (2.0 * 40.0 + k * std::f64::consts::SQRT_2);
                if s <= 40.0 {
                    (-k - 40.0 + s, 0.0)
                } else if s <= 40.0 + k * std::f64::consts::SQRT_2 {
                    let t = (s - 40.0) / (k * std::f64::consts::SQRT_2);
                    (-k + k * t, -k * t)
                } else {
                    (0.0, -k - (s - 40.0 - k * std::f64::consts::SQRT_2))
                }
            })
            .collect();
        let mut worst = f64::NEG_INFINITY;
        for i in -40..=40 {
            for j in -40..=40 {
                let (x, y) = (i as f64 * 0.37 - 4.0, j as f64 * 0.37 - 4.0);
                let inside = x <= 0.0 && y <= 0.0 && x + y <= -k;
                let truth = boundary
                    .iter()
                    .map(|(bx, by)| (x - bx).hypot(y - by))
                    .fold(f64::INFINITY, f64::min);
                let value = at(x, y);
                assert_eq!(value < 0.0, inside, "({x}, {y}) is on the wrong side");
                worst = worst.max(value.abs() - truth);
            }
        }
        // The sampled boundary is 0.02 mm apart at worst, so the truth is known to half that.
        assert!(worst <= 0.011, "the chamfer claims {worst} mm more room than there is");
    }

    /// The chamfer's closed-form bound, held as the G2 one is: no faster than the larger
    /// operand change, the raw form's sign everywhere, and between the raw form and its half
    /// -- never worse than the spec's division, and the nearer operand outright a leg clear of
    /// the join.
    #[test]
    fn the_chamfers_bound_moves_no_faster_than_the_larger_operand_and_keeps_the_surface() {
        let n = fieldform_scene::Continuity::Chamfer.exponent();
        let k = 4.0;
        let step = 0.05;
        let values: Vec<f64> = (-120..=160).map(|i| f64::from(i) * 0.1 + 0.0137).collect();
        let mut steepest = 0.0f64;
        for &a in &values {
            for &b in &values {
                let (h, raw) = (blend_bound(a, b, k, n), smooth_min(a, b, k, n));
                assert_eq!(h < 0.0, raw < 0.0, "({a}, {b}): the bound changed the side");
                assert!(
                    h.abs() <= raw.abs() + 1e-9 && h.abs() >= raw.abs() * 0.5 - 1e-9,
                    "({a}, {b}): the bound {h} is outside the raw {raw} and its half"
                );
                if b >= a.abs() + k {
                    assert!((h - a.min(b)).abs() < 1e-12, "({a}, {b}) is a leg clear: {h}");
                }
                for (da, db) in [(step, 0.0), (0.0, step), (step, step), (step, -step)] {
                    let moved = blend_bound(a + da, b + db, k, n);
                    steepest = steepest.max((moved - h).abs() / f64::max(da.abs(), db.abs()));
                }
            }
        }
        assert!(steepest <= 1.0 + 1e-9, "the bound rises {steepest} mm per millimetre");
    }

    /// **A pattern's fold refuses a chamfer by name, and still folds G1 and G2** -- the
    /// refusal is about the chamfer having no order-free form across many copies, not about
    /// the fold.
    #[test]
    fn a_patterns_fold_refuses_a_chamfer_and_still_rounds() {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([ParamRef::Literal(30.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
            radius: ParamRef::Literal(12.0),
        });
        let pattern = graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(6.0),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::at_along(
                    fieldform_scene::Axis::Z,
                    [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
                ),
            },
            join: true,
            fillet: fieldform_scene::Fillet {
                radius: ParamRef::Literal(3.0),
                continuity: fieldform_scene::Continuity::Chamfer,
            },
        });
        let doc = Document::new();
        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3),
            Err(CompileError::ChamferOnAPattern)
        ));
        for shape in [fieldform_scene::Continuity::G1, fieldform_scene::Continuity::G2] {
            let mut rounded = graph.clone();
            if let Some(Node::Pattern { fillet, .. }) = rounded.node_mut(pattern) {
                fillet.continuity = shape;
            }
            assert!(GraphField::compile_graph(&doc, &rounded, 1e-3).is_ok(), "{shape:?} folds");
        }
    }

    /// The bound the blend's interval is built on, asserted directly rather than trusted.
    ///
    /// A fillet only adds material, so the blend never rises above a hard minimum. How far it
    /// can fall below one is what the interval needs: at most `(2^(1/n) - 1)` times the gap
    /// between `k` and the hard minimum. Unlike the polynomial blend's flat `k/4`, this one
    /// grows with depth — the cost of being exact at the surface, which is where it matters.
    #[test]
    fn a_blend_never_rises_above_a_hard_boolean_and_falls_below_it_by_a_bounded_amount() {
        let k = 4.0;
        let n = fieldform_scene::Continuity::G1.exponent();
        let slack = 2f64.powf(1.0 / n) - 1.0;
        let mut worst_ratio: f64 = 0.0;
        for i in -200..=200 {
            for j in -200..=200 {
                let (a, b) = (i as f64 * 0.05, j as f64 * 0.05);
                let hard = a.min(b);
                let dip = hard - smooth_min(a, b, k, n);
                assert!(
                    dip >= -1e-12,
                    "a blend must never exceed a hard minimum, got {dip}"
                );
                let allowed = slack * (k - hard).max(0.0);
                assert!(
                    dip <= allowed + 1e-12,
                    "at ({a}, {b}) the blend dipped {dip}, past the {allowed} the interval allows"
                );
                if allowed > 1e-9 {
                    worst_ratio = worst_ratio.max(dip / allowed);
                }
            }
        }
        // And the bound is reached, so it is not quietly loose: the worst case is the diagonal,
        // where both gaps are equal.
        assert!(
            worst_ratio > 0.999,
            "the bound should be tight; worst was {worst_ratio}"
        );
    }

    /// Outside the fillet, the operands are left exactly as they are — no wandering seam far
    /// from the join, which the polynomial blend had over a band twice as wide as its radius.
    #[test]
    fn a_blend_changes_nothing_further_than_its_radius_from_the_join() {
        let k = 3.0;
        let n = fieldform_scene::Continuity::G1.exponent();
        for i in 0..=200 {
            let a = k + k * i as f64 / 100.0; // from the tangency point outward
            for j in -100..=100 {
                let b = j as f64 * 0.1;
                if b < k {
                    continue;
                }
                assert_eq!(
                    smooth_min(a, b, k, n),
                    a.min(b),
                    "the blend moved a surface {a} and {b} away from the join"
                );
            }
        }
    }

    /// Subtraction and intersection are the same fillet with signs flipped, so they get the
    /// same radius. Checked through `blend` rather than by re-deriving it, since that is the
    /// function the graph actually evaluates.
    #[test]
    fn every_blended_boolean_fillets_by_the_same_radius() {
        let k = 5.0;
        let n = fieldform_scene::Continuity::G1.exponent();
        for i in -100..=100 {
            for j in -100..=100 {
                let (a, b) = (i as f64 * 0.1, j as f64 * 0.1);
                // Each is the one bound with its operands negated and its answer negated back,
                // and that bound has the raw fillet's zero set and sign -- so the surface each
                // makes is the fillet's, of the radius asked for.
                let difference = blend(BooleanOp::Difference, a, b, k, n, k, 0.0);
                let intersection = blend(BooleanOp::Intersection, a, b, k, n, k, 0.0);
                assert_eq!(difference, -blend_bound(-a, b, k, n));
                assert_eq!(intersection, -blend_bound(-a, -b, k, n));
                for (bounded, raw) in [
                    (difference, -smooth_min(-a, b, k, n)),
                    (intersection, -smooth_min(-a, -b, k, n)),
                ] {
                    assert!(
                        raw.abs() < 1e-9 || (bounded > 0.0) == (raw > 0.0),
                        "at ({a}, {b}) the bound reads {bounded} where the fillet reads {raw}"
                    );
                }
                // And each collapses to its hard form once the radius is gone.
                assert_eq!(blend(BooleanOp::Union, a, b, 0.0, n, 0.0, 0.0), a.min(b));
                assert_eq!(blend(BooleanOp::Difference, a, b, 0.0, n, 0.0, 0.0), a.max(-b));
                assert_eq!(blend(BooleanOp::Intersection, a, b, 0.0, n, 0.0, 0.0), a.max(b));
            }
        }
    }

    /// The blend's interval, checked on a box small enough that the leaves' own slack cannot
    /// hide the dip.
    ///
    /// The wider boxes above pass with or without the `k/4` term, because centre-and-radius
    /// leaf bounds carry enough slack to absorb it. Shrinking the box drives that slack toward
    /// zero and leaves the dip as the only thing standing between the bound and the values the
    /// field takes — so this fails if the term is dropped.
    ///
    /// The blend radius is deliberately wider than the gap between the two fields. The 1 mm
    /// cage sits wholly inside the 5 mm pipe, so their distances stay about 4.5 mm apart and
    /// never cross; a narrow blend would sit pinned at one end of its range, where it
    /// degenerates to a hard `min` and the dip is zero. A radius of 12 puts the blend in its
    /// interior, which is the only regime where the term this test exists for does anything.
    #[test]
    fn a_blends_interval_holds_where_the_blend_is_actually_blending() {
        let (doc, cage_id, curve_id) = document();

        let pipe_only = {
            let mut g = Graph::from_cage(cage_id);
            let _p = g.add(Node::Pipe {
                curve: curve_id,
                radius: ParamRef::Literal(5.0),
            });
            GraphField::compile(&doc, &part_with(g), 0.01).unwrap()
        };
        let cage_only =
            GraphField::compile(&doc, &part_with(Graph::from_cage(cage_id)), 0.01).unwrap();

        let mut graph = Graph::from_cage(cage_id);
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(5.0),
        });
        let _blended = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: cage_node.into(),
            b: pipe.into(),
            fillet: Fillet { radius: ParamRef::Literal(12.0), continuity: Continuity::G1 },
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();

        let centre = Point3::new(0.0, 2.0, 0.0);
        let half = Vector3::repeat(0.002);
        let (lo, hi) = (centre - half, centre + half);

        // Establish the test is in the regime it claims to be: the blended value has to sit
        // meaningfully below a hard union here, or the k/4 term is not being exercised and
        // passing would prove nothing.
        let hard_min = cage_only.distance(centre).min(pipe_only.distance(centre));
        let dip = hard_min - field.distance(centre);
        assert!(
            dip > 1.0,
            "the blend should be dipping well below a hard min here, got {dip:.3}"
        );

        let (bound_lo, bound_hi) = field.interval(lo, hi);
        let (seen_lo, seen_hi) = observed_range(&field, lo, hi);
        assert!(
            bound_lo <= seen_lo && bound_hi >= seen_hi,
            "interval ({bound_lo:.6}, {bound_hi:.6}) fails to contain observed \
             ({seen_lo:.6}, {seen_hi:.6}) on a box where the blend dips {dip:.3}",
        );
    }

    /// Both leaves here are true distance fields, so the graph must report the default
    /// constant. This is the baseline the lattice case will move off: when a leaf reports
    /// something larger, `lipschitz_at` has to carry it to the root rather than leaving the
    /// graph claiming a unit gradient it no longer has.
    #[test]
    fn a_graph_of_true_distance_fields_stays_one_lipschitz() {
        let (doc, cage_id, curve_id) = document();
        let mut graph = Graph::from_cage(cage_id);
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(5.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: cage_node.into(),
            b: pipe.into(),
            fillet: Fillet { radius: ParamRef::Literal(4.0), continuity: Continuity::G1 },
        });

        let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();
        assert_eq!(field.lipschitz(), 1.0);
    }

    /// The bound has to be tight enough to be worth computing: a box well clear of both the
    /// cage and the pipe should be provably empty, which is what lets a caller skip it.
    #[test]
    fn a_box_clear_of_the_whole_graph_is_proved_empty() {
        let (doc, cage_id, curve_id) = document();
        let mut graph = Graph::from_cage(cage_id);
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(5.0),
        });
        let _united = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![cage_node.into(), pipe.into()],
        });

        let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();
        let (bound_lo, _) = field.interval(
            Point3::new(100.0, 100.0, 100.0),
            Point3::new(110.0, 110.0, 110.0),
        );
        assert!(
            bound_lo > 0.0,
            "a box 100 mm clear of everything should be provably outside, got {bound_lo}"
        );
    }

    /// The user-facing shape of the feature: a curve plus a radius becomes solid geometry.
    #[test]
    fn a_pipe_node_becomes_a_tube_of_the_given_radius() {
        let (doc, _, curve_id) = document();
        let mut graph = Graph::empty();
        let _pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(5.0),
        });

        let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();

        // On the spine: one radius inside.
        assert!((field.distance(Point3::origin()) + 5.0).abs() < 1e-9);
        // Exactly on the surface.
        assert!(field.distance(Point3::new(0.0, 5.0, 0.0)).abs() < 1e-9);
        // Well outside.
        assert!((field.distance(Point3::new(0.0, 25.0, 0.0)) - 20.0).abs() < 1e-9);
    }

    /// Radius comes from a parameter, so changing the parameter changes the geometry with no
    /// replay and no re-edit. That is the parametric layer's whole claim, tested at its
    /// smallest.
    #[test]
    fn the_radius_can_come_from_a_named_parameter() {
        let (mut doc, _, curve_id) = document();
        let param = fieldform_scene::ParamId::new();
        doc.parameters.insert(
            param,
            fieldform_scene::Parameter {
                id: param,
                name: "pipe_radius".into(),
                value: fieldform_scene::ParamValue::Literal(8.0),
            },
        );

        let mut graph = Graph::empty();
        let _pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Named(param),
        });

        let field = GraphField::compile(&doc, &part_with(graph.clone()), 0.01).unwrap();
        assert!(field.distance(Point3::new(0.0, 8.0, 0.0)).abs() < 1e-9);

        // Change only the parameter, recompile, and the tube is a different size.
        if let Some(p) = doc.parameters.get_mut(&param) {
            p.value = fieldform_scene::ParamValue::Literal(3.0);
        }
        let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();
        assert!(field.distance(Point3::new(0.0, 3.0, 0.0)).abs() < 1e-9);
        assert!(
            field.distance(Point3::new(0.0, 8.0, 0.0)) > 0.0,
            "should now be outside"
        );
    }

    /// Union and difference against the cage — the two the user asked for. The cube cage is
    /// about 1 mm across at its limit surface, so a 5 mm pipe through the origin swallows it.
    #[test]
    fn a_pipe_unions_and_subtracts_against_a_cage() {
        let (doc, cage_id, curve_id) = document();

        let build = |op: BooleanOp| {
            let mut graph = Graph::from_cage(cage_id);
            let cage_node = graph.roots()[0];
            let pipe = graph.add(Node::Pipe {
                curve: curve_id,
                radius: ParamRef::Literal(5.0),
            });
            graph.add(Node::Boolean {
                op,
                inputs: vec![cage_node.into(), pipe.into()],
            });
            GraphField::compile(&doc, &part_with(graph), 0.01).unwrap()
        };

        // A point 3 mm out along Y: inside the pipe, outside the little cube.
        let probe = Point3::new(0.0, 3.0, 0.0);

        let union = build(BooleanOp::Union);
        assert!(
            union.distance(probe) < 0.0,
            "union should contain the pipe: {}",
            union.distance(probe)
        );

        let difference = build(BooleanOp::Difference);
        assert!(
            difference.distance(probe) > 0.0,
            "subtracting the pipe should leave this point empty: {}",
            difference.distance(probe)
        );

        // And subtraction must actually remove material from the cage: the cage centre is
        // inside the cube but also inside the pipe, so the difference must be outside.
        assert!(
            difference.distance(Point3::origin()) > 0.0,
            "the pipe should have cut through the cage centre"
        );
    }

    /// Subtraction is intersection with a complement, so it must be sensitive to which operand
    /// is which — a symmetric implementation would pass a union test and still be wrong.
    #[test]
    fn difference_is_not_symmetric() {
        let (doc, cage_id, curve_id) = document();
        let mut graph = Graph::from_cage(cage_id);
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(5.0),
        });
        let _pipe_minus_cage = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![pipe.into(), cage_node.into()],
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.01).unwrap();

        // Pipe minus cage: 3 mm out along Y is still pipe, and the cage does not reach there.
        assert!(field.distance(Point3::new(0.0, 3.0, 0.0)) < 0.0);
        // But the origin is inside the cage, so it has been removed from the pipe.
        assert!(field.distance(Point3::origin()) > 0.0);
    }

    /// The level has to track the tolerance, and the numbers here come from
    /// `cage_field_accuracy`: a 1000 mm cube (1732 mm diagonal) is 14.4 mm out at level 2,
    /// 3.5 mm at level 3, 0.9 mm at level 4.
    #[test]
    fn the_cage_level_follows_the_requested_tolerance() {
        let diagonal = 1732.0;

        // The app default: 64 cells over ~2100 mm of bounds is a ~33 mm cell, so a ~16 mm
        // tolerance. Position-wise level 3 would do; the shading floor lifts it to 4.
        assert_eq!(levels_for_tolerance(diagonal, 16.4), MIN_CAGE_LEVELS);
        // A tolerance tight enough to need more than the floor gets more than the floor.
        assert_eq!(levels_for_tolerance(diagonal, 1.0), 5);
        assert_eq!(levels_for_tolerance(diagonal, 0.25), 6);
        // A coarse ask must not drop below the floor, and a fine one must not run away.
        assert_eq!(levels_for_tolerance(diagonal, 10_000.0), MIN_CAGE_LEVELS);
        assert_eq!(levels_for_tolerance(diagonal, 1e-9), MAX_CAGE_LEVELS);
        // Nonsense inputs land on the floor rather than panicking or looping.
        assert_eq!(levels_for_tolerance(0.0, 1.0), MIN_CAGE_LEVELS);
        assert_eq!(levels_for_tolerance(diagonal, 0.0), MIN_CAGE_LEVELS);
    }

    /// A small model must not be forced to pay for accuracy nothing can display: the same
    /// tolerance on a part a thousand times smaller needs far less subdivision.
    #[test]
    fn a_smaller_cage_needs_fewer_levels_for_the_same_tolerance() {
        assert!(levels_for_tolerance(1.7, 1.0) < levels_for_tolerance(1732.0, 1.0));
    }

    /// Exact points and normals on a cage's limit surface, forty-one to a face edge: dense
    /// enough that the worst reading has stopped growing, since sixty-one and eighty-one read
    /// within a few hundredths of a degree of it on every box below. Eleven was not: it put a
    /// cube's level 3 at 0.25 deg against 0.41. Some of these land on a tessellation's
    /// vertices, where every level reads exactly, so they add nothing to a worst case; the
    /// density is what makes up for them.
    fn limit_points(cage: &fieldform_cage::Cage) -> Vec<(Point3<f64>, Vector3<f64>)> {
        let mut surface = fieldform_subd::LimitSurface::new(cage);
        let mut truth = Vec::new();
        for f in cage.face_ids() {
            let mut samples = Vec::new();
            surface
                .tessellate_face(f, 41, &mut samples)
                .expect("a box's faces evaluate");
            truth.extend(samples);
        }
        truth
    }

    /// A box cage with these sides, centred on the origin: the starting cube, stretched.
    fn box_cage(x: f64, y: f64, z: f64) -> fieldform_cage::Cage {
        let mut cage = fieldform_cage::Cage::cube_of(1.0);
        let corners: Vec<_> = cage.vertex_ids().collect();
        for v in corners {
            let p = cage.vertex_position(v);
            cage.set_vertex_position(v, Point3::new(p.x * x, p.y * y, p.z * z));
        }
        cage
    }

    /// **The floor answers to shading, and a cube is the one shape that flatters it.** Position
    /// is `levels_for_tolerance`'s business, so what a floor guarantees is a surface that does
    /// not read as faceted when nobody asked for accuracy, and one degree is where a shaded
    /// normal error stops being visible (the spine's threshold, `main.rs`). A cube's faces turn
    /// through the gentlest angle a six-face cage can, so level 3 shades it 0.41 deg out and
    /// the floor once looked free to fall to 3. Stretch it and the turn tightens over the same
    /// triangles: a box twice as long as it is wide is 1.21 deg out at level 3 and 0.29 at 4, a
    /// slab a third as high as it is wide 2.58 against 0.67. So the floor holds a box of
    /// ordinary proportions inside a degree and the level below it does not; past three to one
    /// no constant floor does, which `MIN_CAGE_LEVELS` records rather than this test.
    #[test]
    fn the_cage_floor_is_the_lowest_level_that_shades_a_slab_a_third_as_high_as_it_is_wide_within_a_degree()
    {
        let worst_normal = |cage: &fieldform_cage::Cage, levels: usize| {
            let field = CageField::from_cage(cage, levels).expect("a box subdivides");
            limit_points(cage)
                .iter()
                .map(|(p, n)| field.closest(*p).normal.angle(n).to_degrees())
                .fold(0.0, f64::max)
        };
        let slab = box_cage(1000.0, 1000.0, 1000.0 / 3.0);
        for (what, cage) in [
            ("the starting cube", fieldform_cage::Cage::cube_of(1000.0)),
            ("a box twice as long as it is wide", box_cage(1000.0, 500.0, 500.0)),
            ("a slab a third as high as it is wide", slab.clone()),
        ] {
            let at_floor = worst_normal(&cage, MIN_CAGE_LEVELS);
            assert!(
                at_floor < 1.0,
                "level {MIN_CAGE_LEVELS} shades {what} {at_floor:.2} deg out, past a degree"
            );
        }
        let below = worst_normal(&slab, MIN_CAGE_LEVELS - 1);
        assert!(
            below > 1.0,
            "level {} shades the slab {below:.2} deg out, inside a degree, so the floor could drop",
            MIN_CAGE_LEVELS - 1
        );
    }

    /// The floor answers to shading, so wherever a tolerance asks for more than the floor gives,
    /// the tolerance is all that holds a cage's position, and it is measured here rather than
    /// trusted: at the tolerances the viewport and picking ask of the starting cube, and at a
    /// millimetre, the level chosen puts the field's surface inside the tolerance.
    /// `CAGE_ERROR_COEFFICIENT` was fitted to this cube's side and is applied to its diagonal,
    /// so it errs fine by about √3, and that is the margin these readings show.
    #[test]
    fn the_level_a_tolerance_gets_holds_the_cage_field_within_that_tolerance() {
        let cage = fieldform_cage::Cage::cube_of(1000.0);
        let truth = limit_points(&cage);
        let diagonal = 1000.0 * 3f64.sqrt();
        // The viewport over padded bounds and over the cube's own box, picking's thousandth of
        // the diagonal, and a millimetre.
        for tolerance in [16.4, 1000.0 / 128.0, diagonal * 1e-3, 1.0] {
            let levels = levels_for_tolerance(diagonal, tolerance);
            let field = CageField::from_cage(&cage, levels).expect("a cube subdivides");
            let worst = truth
                .iter()
                .map(|(p, _)| field.distance(*p).abs())
                .fold(0.0, f64::max);
            assert!(
                worst <= tolerance,
                "{tolerance:.2} mm asked, and level {levels} is {worst:.2} mm out"
            );
        }
    }

    /// A cube cage, offset so two of these can be made to overlap partially.
    fn cube_at(size: f64, center: Point3<f64>) -> fieldform_cage::Cage {
        let cube = fieldform_cage::Cage::cube_of(size);
        let positions: Vec<Point3<f64>> = cube
            .vertex_ids()
            .map(|v| cube.vertex_position(v) + center.coords)
            .collect();
        let faces: Vec<Vec<u32>> = cube
            .face_ids()
            .map(|f| {
                cube.face_vertices(f)
                    .iter()
                    .map(|v| v.index() as u32)
                    .collect()
            })
            .collect();
        fieldform_cage::Cage::from_faces(positions, &faces).unwrap()
    }

    /// A document with two 4 mm cube cages, `a` centred at the origin and `b` offset 3 mm
    /// along X so they overlap over `x in [1, 2]` — enough for each op's surface to include
    /// stretches defined by `a` alone, by `b` alone, and (for difference/intersection) by `b`
    /// cutting into `a`.
    fn two_cages() -> (Document, fieldform_scene::CageId, fieldform_scene::CageId) {
        let mut doc = Document::new();
        let a_id = fieldform_scene::CageId::new();
        doc.cages.insert(a_id, fieldform_cage::Cage::cube_of(4.0));
        let b_id = fieldform_scene::CageId::new();
        doc.cages
            .insert(b_id, cube_at(4.0, Point3::new(3.0, 0.0, 0.0)));
        (doc, a_id, b_id)
    }

    /// The whole point of [`GraphField::normal_at`]: it must pick the same operand `eval`
    /// used to produce the value, not blend or guess. "Near one shape's own surface" is not
    /// by itself enough to say which operand wins — for union it is (whichever is closer sets
    /// the min), but intersection and difference also depend on where that point sits relative
    /// to the *other* operand, which is why each op below gets its own, worked-through probes
    /// rather than one shared pair.
    #[test]
    fn union_normal_delegates_to_whichever_operand_is_actually_closer() {
        let (doc, a_id, b_id) = two_cages();
        let tolerance = 0.5;
        let standalone_a =
            CageField::from_cage(&doc.cages[&a_id], levels_for_tolerance(4.0, tolerance)).unwrap();
        let standalone_b =
            CageField::from_cage(&doc.cages[&b_id], levels_for_tolerance(4.0, tolerance)).unwrap();

        let mut graph = Graph::from_cage(a_id);
        let a_node = graph.roots()[0];
        let b_node = graph.add(Node::Cage(b_id));
        let _union = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![a_node.into(), b_node.into()],
        });
        let field = GraphField::compile(&doc, &part_with(graph), tolerance).unwrap();

        // Far left face of `a`: outside `b` entirely (`b` starts at x = 1), so `a`'s own
        // distance (0) beats `b`'s (positive) and the union's min picks `a`.
        let on_a_alone = Point3::new(-2.0, 0.0, 0.0);
        let got = field.normal(on_a_alone).expect("a is analytic");
        let want = standalone_a.closest(on_a_alone).normal;
        assert!(got.angle(&want) < 1e-6, "got {got:?}, want {want:?}");

        // Far right face of `b`: outside `a` entirely (`a` ends at x = 2), so `b` wins by the
        // same reasoning, mirrored.
        let on_b_alone = Point3::new(5.0, 0.0, 0.0);
        let got = field.normal(on_b_alone).expect("b is analytic");
        let want = standalone_b.closest(on_b_alone).normal;
        assert!(got.angle(&want) < 1e-6, "got {got:?}, want {want:?}");
    }

    /// Intersection needs both: standing on `a`'s own surface only wins if that point is also
    /// *inside* `b` (otherwise `b`'s large positive distance dominates the max, however close
    /// `a` is) — the opposite of union's intuition.
    #[test]
    fn intersection_normal_delegates_to_whichever_operand_the_boundary_sits_inside() {
        let (doc, a_id, b_id) = two_cages();
        let tolerance = 0.5;
        let standalone_a =
            CageField::from_cage(&doc.cages[&a_id], levels_for_tolerance(4.0, tolerance)).unwrap();
        let standalone_b =
            CageField::from_cage(&doc.cages[&b_id], levels_for_tolerance(4.0, tolerance)).unwrap();

        let mut graph = Graph::from_cage(a_id);
        let a_node = graph.roots()[0];
        let b_node = graph.add(Node::Cage(b_id));
        let _intersection = graph.add(Node::Boolean {
            op: BooleanOp::Intersection,
            inputs: vec![a_node.into(), b_node.into()],
        });
        let field = GraphField::compile(&doc, &part_with(graph), tolerance).unwrap();

        // a's right face, x = 2, sits inside b (which spans x in [1, 5]): a's distance (0)
        // beats b's (negative, since inside), so the max picks a.
        let a_face_inside_b = Point3::new(2.0, 0.0, 0.0);
        let got = field.normal(a_face_inside_b).expect("a is analytic");
        let want = standalone_a.closest(a_face_inside_b).normal;
        assert!(got.angle(&want) < 1e-6, "got {got:?}, want {want:?}");

        // b's left face, x = 1, sits inside a: b's distance (0) beats a's (negative), so the
        // max picks b.
        let b_face_inside_a = Point3::new(1.0, 0.0, 0.0);
        let got = field.normal(b_face_inside_a).expect("b is analytic");
        let want = standalone_b.closest(b_face_inside_a).normal;
        assert!(got.angle(&want) < 1e-6, "got {got:?}, want {want:?}");
    }

    /// The straightforward half of difference: outside `b` entirely, `a`'s own value (0)
    /// beats `-b`'s (negative), so `a` wins unnegated. The
    /// `difference_negates_the_cutting_operand_where_it_bites_into_the_other` test below
    /// covers the other half, where `b` wins and must be negated.
    #[test]
    fn difference_normal_is_a_unnegated_where_b_never_reaches() {
        let (doc, a_id, b_id) = two_cages();
        let tolerance = 0.5;
        let standalone_a =
            CageField::from_cage(&doc.cages[&a_id], levels_for_tolerance(4.0, tolerance)).unwrap();

        let mut graph = Graph::from_cage(a_id);
        let a_node = graph.roots()[0];
        let b_node = graph.add(Node::Cage(b_id));
        let _difference = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![a_node.into(), b_node.into()],
        });
        let field = GraphField::compile(&doc, &part_with(graph), tolerance).unwrap();

        let on_a_alone = Point3::new(-2.0, 0.0, 0.0);
        let got = field.normal(on_a_alone).expect("a is analytic");
        let want = standalone_a.closest(on_a_alone).normal;
        assert!(got.angle(&want) < 1e-6, "got {got:?}, want {want:?}");
    }

    /// The sign flip is the one non-obvious part of the derivation: where `b` cuts into `a`,
    /// the difference's boundary is `b`'s surface, but the *removed* material is inside `b`,
    /// so the composed outward direction is `b`'s own normal negated.
    #[test]
    fn difference_negates_the_cutting_operand_where_it_bites_into_the_other() {
        let (doc, a_id, b_id) = two_cages();
        let tolerance = 0.5;
        let standalone_b =
            CageField::from_cage(&doc.cages[&b_id], levels_for_tolerance(4.0, tolerance)).unwrap();

        let mut graph = Graph::from_cage(a_id);
        let a_node = graph.roots()[0];
        let b_node = graph.add(Node::Cage(b_id));
        let _difference = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![a_node.into(), b_node.into()],
        });
        let field = GraphField::compile(&doc, &part_with(graph), tolerance).unwrap();

        // b's left face at x = 1 sits inside a (which spans to x = 2): the bite `b` cuts out
        // of `a`.
        let bite = Point3::new(1.0, 0.0, 0.0);
        let got = field.normal(bite).expect("b is analytic");
        let want = -standalone_b.closest(bite).normal;
        assert!(
            got.angle(&want) < 1e-6,
            "at the bite: got {got:?}, want {want:?} (negated b)"
        );
    }

    /// A blend mixes both operands continuously — there is no single active operand to
    /// delegate to, so it must fall back rather than guess.
    #[test]
    fn blend_normal_is_none_so_the_caller_falls_back_to_finite_differences() {
        let (doc, a_id, b_id) = two_cages();
        let mut graph = Graph::from_cage(a_id);
        let a_node = graph.roots()[0];
        let b_node = graph.add(Node::Cage(b_id));
        let _blend = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: a_node.into(),
            b: b_node.into(),
            fillet: Fillet { radius: ParamRef::Literal(1.0), continuity: Continuity::G1 },
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.5).unwrap();

        assert!(field.normal(Point3::new(1.5, 0.0, 0.0)).is_none());
    }

    /// A node written by a build this one is older than, as it arrives from a file.
    fn unknown_node() -> Node {
        serde_json::from_str(r#"{"kind": "acme.lattice", "cell": 5.0}"#)
            .expect("an unreadable node deserializes to Unknown rather than failing")
    }

    #[test]
    fn a_node_this_build_cannot_read_refuses_to_compile_and_says_which() {
        let (doc, a_id, _) = two_cages();
        let mut graph = Graph::from_cage(a_id);
        let cage_node = graph.roots()[0];
        let lattice = graph.add(unknown_node());
        let _union = graph.add(Node::Boolean {
            op: fieldform_scene::BooleanOp::Union,
            inputs: vec![cage_node.into(), lattice.into()],
        });

        let Err(error) = GraphField::compile(&doc, &part_with(graph), 0.5) else {
            panic!("a graph holding a node this build cannot read must not compile");
        };
        assert!(matches!(error, CompileError::UnreadableNode(_)));
        let message = error.to_string();
        assert!(
            message.contains("acme.lattice"),
            "names the node kind: {message}"
        );
    }

    #[test]
    fn an_unreadable_node_the_output_does_not_reach_still_refuses() {
        // "Unreachable" is itself a reading of a node this build could not read: what an
        // unknown node consumes is exactly the part that failed to parse, so a graph holding
        // one cannot be shown to be independent of it.
        let (doc, a_id, _) = two_cages();
        let mut graph = Graph::from_cage(a_id);
        graph.add(unknown_node());

        assert!(matches!(
            GraphField::compile(&doc, &part_with(graph), 0.5),
            Err(CompileError::UnreadableNode(_))
        ));
    }

    #[test]
    fn a_point_on_a_surface_names_the_body_it_belongs_to() {
        // What clicking a body in the viewport rests on: the composed field says where the
        // surface is, and this says whose it is.
        let doc = Document::new();
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(10.0),
        });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([
                ParamRef::Literal(50.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ]),
            size: [ParamRef::Literal(10.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let field = GraphField::compile_graph(&doc, &graph, 0.1).unwrap();

        assert_eq!(
            field.body_at(Point3::new(10.0, 0.0, 0.0), 0.01),
            Some(sphere)
        );
        assert_eq!(field.body_at(Point3::new(45.0, 0.0, 0.0), 0.01), Some(cube));
        assert_eq!(
            field.body_at(Point3::new(25.0, 25.0, 25.0), 0.01),
            None,
            "a click on empty space belongs to nothing"
        );
    }

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// A sphere of `radius` at the origin, placed by the given transform, compiled.
    fn placed_sphere(
        radius: f64,
        translation: [f64; 3],
        rotation: [f64; 3],
        scale: f64,
    ) -> Result<GraphField, CompileError> {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(radius),
        });
        graph.add(Node::Transform {
            input: sphere.into(),
            translation: translation.map(lit),
            rotation: rotation.map(lit),
            scale: lit(scale),
            about: fieldform_scene::Place::origin(),
        });
        GraphField::compile(&doc, &part_with(graph), 0.05)
    }

    /// A sphere of `radius` at the origin, offset by `distance`.
    fn offset_sphere(radius: f64, distance: f64) -> Result<GraphField, CompileError> {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(radius),
        });
        graph.add(Node::Offset {
            input: sphere.into(),
            distance: lit(distance),
        });
        GraphField::compile(&doc, &part_with(graph), 0.05)
    }

    /// The claim the node rests on: offsetting a sphere gives a sphere of the offset radius,
    /// *and* the value away from it is still the true distance to the new surface.
    ///
    /// Both halves, because checking only the surface would pass an implementation that moved
    /// the boundary by some other means and left every other reading describing the old shape
    /// — which is exactly the failure that has no symptom until something blends against it.
    #[test]
    fn an_offset_sphere_is_a_sphere_of_the_offset_radius() {
        for (radius, distance) in [(10.0, 2.5), (10.0, -4.0), (10.0, 0.0), (3.0, 17.0)] {
            let field = offset_sphere(radius, distance).unwrap();
            let grown = radius + distance;
            let centre = Point3::new(0.0, 0.0, 0.0);

            assert!(
                (field.distance(centre) - -grown).abs() < 1e-9,
                "offset {distance} put the centre at the wrong depth",
            );
            assert!(
                field.distance(centre + Vector3::new(grown, 0.0, 0.0)).abs() < 1e-9,
                "offset {distance} put the surface in the wrong place",
            );
            // 5 mm out from the *new* surface reads 5, which is the half a surface-only test
            // would miss.
            assert!(
                (field.distance(centre + Vector3::new(grown + 5.0, 0.0, 0.0)) - 5.0).abs() < 1e-9,
                "offset {distance} stopped reporting a true distance",
            );
        }
    }

    /// D8's property on the new node.
    ///
    /// Cheap to assert and the whole reason offset is worth having as a node: it preserves
    /// `Distance` exactly, so an offset body can still be blended. An implementation that
    /// scaled instead of shifted would put the surface in a plausible place and fail here.
    #[test]
    fn an_offset_field_still_moves_at_unit_rate() {
        for distance in [-4.0, 0.0, 2.5, 40.0] {
            let field = offset_sphere(10.0, distance).unwrap();
            let points: Vec<Point3<f64>> = (0..40)
                .map(|i| {
                    let t = i as f64;
                    Point3::new(60.0 + t, -14.0 + 0.7 * t, 9.0 - 0.3 * t)
                })
                .collect();
            assert!(
                gradient_deviation(&field, &points, 1e-4) < 1e-6,
                "offset {distance} broke the unit-gradient property",
            );
        }
    }

    /// The consequence of preserving `Distance`: an offset body is still blendable.
    ///
    /// The refusal a blend makes is the only thing in the graph that reads normalization, so
    /// this is where "offset preserves it" becomes something a user can feel. A `LowerBound`
    /// here would mean every offset body is permanently un-filletable.
    #[test]
    fn an_offset_body_can_still_be_blended() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let a = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let grown = graph.add(Node::Offset {
            input: a.into(),
            distance: lit(2.0),
        });
        let b = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(14.0), lit(0.0), lit(0.0)]),
            radius: lit(6.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: grown.into(),
            b: b.into(),
            fillet: Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::G1 },
        });
        assert!(GraphField::compile(&doc, &part_with(graph), 0.05).is_ok());
    }

    /// Shrinking a body by more than its own thickness leaves nothing, and that is an answer
    /// rather than an error: the field stays well-formed and reads positive everywhere.
    ///
    /// Asserted because the alternative — refusing it — is a tempting-looking guard that would
    /// make a shell of a thin wall fail at compile time instead of simply being empty, at a
    /// node the user never touched.
    #[test]
    fn shrinking_a_body_away_leaves_a_field_that_is_simply_empty() {
        let field = offset_sphere(10.0, -25.0).unwrap();
        for p in [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(0.0, 9.0, 0.0),
            Point3::new(30.0, 30.0, 30.0),
        ] {
            assert!(
                field.distance(p) > 0.0,
                "{p:?} was still inside a body shrunk to nothing"
            );
        }
    }

    /// A hollow body is a body minus a shrunken copy, and that composition has to produce a
    /// wall of the thickness asked for — measured inward, so the outside does not move.
    ///
    /// This is the whole of "shell" in this architecture: there is no shell node to test, so
    /// the thing worth asserting is that the two nodes it is built from really do compose into
    /// one. A 2 mm wall on a 10 mm sphere means solid from 8 to 10 and empty inside 8.
    #[test]
    fn hollowing_a_sphere_leaves_a_wall_of_the_thickness_asked_for() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let solid = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let inner = graph.add(Node::Offset {
            input: solid.into(),
            distance: lit(-2.0),
        });
        graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![solid.into(), inner.into()],
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        let at = |r: f64| field.distance(Point3::new(r, 0.0, 0.0));
        // Hollow at the middle, solid through the wall, outside beyond the surface.
        assert!(at(0.0) > 0.0, "the middle should be hollow");
        assert!(
            at(7.0) > 0.0,
            "7 mm out is inside the cavity, not in the wall"
        );
        assert!(at(9.0) < 0.0, "9 mm out is in the wall");
        assert!(at(11.0) > 0.0, "11 mm out is outside the body");
        // And the wall's two faces are where they were asked to be.
        assert!(at(8.0).abs() < 1e-9, "the inner face should sit at 8 mm");
        assert!(at(10.0).abs() < 1e-9, "the outer face should sit at 10 mm");
    }

    /// A part of ten thousand bodies evaluates instead of overflowing the stack.
    ///
    /// Not a hypothetical since the STEP importer began splitting assemblies into one body per
    /// solid: the count is the file's, not the user's, and a big assembly reaches five figures
    /// without anyone doing anything unusual. Folding the roots as nested pairs made this
    /// recursion as deep as the part is wide, and `many_bodies.rs` measured the overflow at
    /// exactly this size.
    ///
    /// Ten thousand rather than a smaller number that would run faster: the point is the
    /// depth, and any count that fits on the stack proves nothing about one that does not.
    #[test]
    fn ten_thousand_bodies_evaluate_rather_than_overflowing_the_stack() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        for i in 0..10_000 {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(i as f64 * 3.0), lit(0.0), lit(0.0)]),
                radius: lit(1.0),
            });
        }
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        // The union really is every body, not merely the first: each sphere's own centre is a
        // radius deep, wherever along the row it sits.
        for i in [0, 1, 4_999, 9_999] {
            let centre = Point3::new(i as f64 * 3.0, 0.0, 0.0);
            assert!(
                (field.distance(centre) - -1.0).abs() < 1e-9,
                "body {i} was not part of the union",
            );
        }
        // And a point between two of them is outside both.
        assert!(field.distance(Point3::new(1.5, 0.0, 0.0)) > 0.0);
    }

    /// A twisted box, built for the tests below.
    fn twisted(degrees_per_mm: f64) -> Result<GraphField, CompileError> {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let boxed = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0), lit(12.0), lit(60.0)],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Twist {
            input: boxed.into(),
            degrees_per_mm: lit(degrees_per_mm),
            frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0); 3]),
        });
        GraphField::compile(&doc, &part_with(graph), 0.05)
    }

    /// The whole contract a warp owes: it may understate the distance and must never overstate
    /// it.
    ///
    /// A twist is not an isometry — it carries material sideways by more the further it is from
    /// the axis — so the value has to be divided by how much it can stretch. Get that factor
    /// wrong and the field claims more room than there is, which is a ray stepping through a
    /// wall and geometry with holes in it.
    ///
    /// Marched *past* what the field claims, because a march that stops at the claim can only
    /// ever confirm it.
    #[test]
    fn a_twisted_body_never_overstates_the_room_before_its_surface() {
        for rate in [0.5, 2.0, 5.0] {
            let field = twisted(rate).unwrap();
            for i in 0..24 {
                for j in 0..6 {
                    let p = Point3::new(
                        -34.0 + i as f64 * 3.0,
                        -18.0 + j as f64 * 6.0,
                        -28.0 + (i % 7) as f64 * 8.0,
                    );
                    let claimed = field.distance(p);
                    if claimed <= 0.0 {
                        continue;
                    }
                    for d in [
                        Vector3::new(1.0, 0.0, 0.0),
                        Vector3::new(0.0, 1.0, 0.0),
                        Vector3::new(0.0, 0.0, 1.0),
                        Vector3::new(0.577, 0.577, 0.577),
                        Vector3::new(-0.577, 0.577, -0.577),
                    ] {
                        let mut t = 0.0;
                        while t < claimed * 1.2 {
                            if field.distance(p + d * t) <= 0.0 {
                                assert!(
                                    t >= claimed - 1e-6,
                                    "rate {rate}: claimed {claimed} but a surface is {t} away",
                                );
                                break;
                            }
                            t += 0.05;
                        }
                    }
                }
            }
        }
    }

    /// Winding about X, Y or Z gives three different bodies, and each leaves its own axis
    /// alone.
    ///
    /// A twist about an axis cannot move material *along* that axis — the rotation is in the
    /// plane across it — so a point on the axis stays put whatever the rate. That is the
    /// property that catches a permutation written the wrong way round, which is otherwise a
    /// body that looks twisted and is twisted about the wrong thing.
    #[test]
    fn a_twist_leaves_its_own_axis_alone_and_the_three_axes_differ() {
        let doc = Document::new();
        let build = |axis: fieldform_scene::Axis| {
            let mut graph = Graph::empty();
            // Different in all three directions, so no axis can pass by symmetry.
            let boxed = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(50.0), lit(20.0), lit(9.0)],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Twist {
                input: boxed.into(),
                degrees_per_mm: lit(3.0),
                frame: fieldform_scene::Frame::along(axis),
            });
            GraphField::compile(&doc, &part_with(graph), 0.05).unwrap()
        };
        let (x, y, z) = (
            build(fieldform_scene::Axis::X),
            build(fieldform_scene::Axis::Y),
            build(fieldform_scene::Axis::Z),
        );

        // On each axis the body is unmoved, so the value there matches the untwisted one.
        let mut graph = Graph::empty();
        graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(50.0), lit(20.0), lit(9.0)],
            turn: fieldform_scene::no_turn(),
        });
        let plain = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
        for (field, on_axis) in [
            (&x, Point3::new(18.0, 0.0, 0.0)),
            (&y, Point3::new(0.0, 8.0, 0.0)),
            (&z, Point3::new(0.0, 0.0, 3.5)),
        ] {
            // Both inside, and the twist only ever *understates*, so the sign is what agrees.
            assert!(
                field.distance(on_axis) <= 0.0 && plain.distance(on_axis) <= 0.0,
                "a point on the axis should be unmoved at {on_axis:?}",
            );
        }

        // And the three are genuinely different bodies.
        let probe = Point3::new(16.0, 7.0, 3.0);
        let (dx, dy, dz) = (x.distance(probe), y.distance(probe), z.distance(probe));
        assert!(
            (dx - dy).abs() > 1e-6 || (dy - dz).abs() > 1e-6,
            "the axis should change the result: {dx}, {dy}, {dz}",
        );
    }

    /// The axis passes through the origin the node names, not through the world's.
    ///
    /// Moving that point moves what stays still, which is the observable difference between a
    /// twist about a line and a twist about the world axis it is parallel to.
    #[test]
    fn a_twists_origin_decides_which_line_stays_still() {
        let doc = Document::new();
        let build = |ox: f64| {
            let mut graph = Graph::empty();
            let boxed = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(40.0), lit(40.0), lit(30.0)],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Twist {
                input: boxed.into(),
                degrees_per_mm: lit(4.0),
                frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(ox), lit(0.0), lit(0.0)]),
            });
            GraphField::compile(&doc, &part_with(graph), 0.05).unwrap()
        };
        let at_zero = build(0.0);
        let at_ten = build(10.0);
        // High up, where the winding has turned things a long way, the two disagree — which
        // they could not if the origin were ignored.
        let probe = Point3::new(6.0, 6.0, 14.0);
        assert!(
            (at_zero.distance(probe) - at_ten.distance(probe)).abs() > 1e-6,
            "moving the axis should change the body",
        );
    }

    /// A twist of nothing is the body it wraps.
    ///
    /// The cheapest check that the winding is applied in the right direction and that the
    /// stretch factor reduces to one — both of which would otherwise only show as a subtly
    /// wrong shape.
    #[test]
    fn a_twist_of_zero_degrees_is_the_body_itself() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0), lit(12.0), lit(60.0)],
            turn: fieldform_scene::no_turn(),
        });
        let plain = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
        let twisted = twisted(0.0).unwrap();
        for i in 0..40 {
            let t = i as f64;
            let p = Point3::new(-30.0 + t * 1.5, -12.0 + t * 0.6, -25.0 + t * 1.2);
            assert!(
                (plain.distance(p) - twisted.distance(p)).abs() < 1e-9,
                "at {p:?}"
            );
        }
    }

    /// The winding actually happens, and in the direction the sign says.
    ///
    /// A point that sits outside the untwisted box near the top can be brought *inside* by the
    /// twist, which is the observable difference between a warp and a no-op.
    #[test]
    fn a_twist_carries_material_around_with_height() {
        let field = twisted(3.0).unwrap();
        // Near the top of a 60 mm box, 3 degrees per mm has turned it by about 90 degrees — so
        // what was the long axis now points along Y.
        let z = 28.0;
        let along_x = field.distance(Point3::new(18.0, 0.0, z));
        let along_y = field.distance(Point3::new(0.0, 18.0, z));
        assert!(
            along_y < along_x,
            "at the top the body should have turned toward Y: {along_y} against {along_x}",
        );
    }

    /// A frame taken from a plane is a graph edge, so the plane stops being a body of its own.
    ///
    /// The property that makes frames *interchangeable* rather than merely identical (D9):
    /// several blocks can share one plane, and moving it moves all of them. That only works if
    /// the reference is a real edge — evaluated first, not a body, and a cycle through it still
    /// a cycle — and the edge is wired generically, so this is really a test that `frames()`
    /// reaches `inputs()` without the lattice arm knowing anything about it.
    #[test]
    fn a_frame_taken_from_a_plane_is_an_edge_and_the_plane_stops_being_a_body() {
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane { origin: fieldform_scene::Place::At([lit(0.0); 3]), flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let lattice = graph.add(Node::Lattice {
            cell: fieldform_scene::LatticeCell::Gyroid,
            map: fieldform_scene::CellMap::Rectangular { cell: lit(10.0) },
            phase: LatticePhase::sheet(lit(1.0)),
            frame: fieldform_scene::Frame::From { node: plane },
        });

        let node = graph.node(lattice).expect("the lattice is there");
        assert!(
            node.inputs().contains(&plane),
            "the plane should be an input of the lattice, got {:?}",
            node.inputs()
        );
        // And so it is consumed: a plane being used as a frame is not a body in its own right,
        // the same rule a boolean's operands follow.
        assert!(
            !graph.roots().contains(&plane),
            "a plane supplying a frame should stop being a body"
        );
    }

    /// A square profile extruded is the box that square and that depth describe.
    ///
    /// Through the graph rather than against `Extrusion` directly, so the curve lookup, the
    /// tessellation, the frame and the placement are all in the path — the parts a unit test of
    /// the field itself cannot reach.
    fn closed_square(half: f64) -> fieldform_scene::Curve {
        // Degree one, so the tessellation is the control polygon and the profile is exactly the
        // square asked for rather than a rounded approximation to it.
        fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-half, -half, 0.0),
                Point3::new(half, -half, 0.0),
                Point3::new(half, half, 0.0),
                Point3::new(-half, half, 0.0),
            ],
            true,
        )
        .expect("a closed polyline is a curve")
    }

    #[test]
    fn an_extruded_square_is_the_box_it_describes() {
        let mut doc = Document::new();
        let curve = closed_square(10.0);
        let id = curve.id;
        doc.curves.insert(id, curve);

        let mut graph = Graph::empty();
        graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(6.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");

        // An extrude runs *from* its plane, so a six-deep one spans z in [0, 6] and its
        // middle is at three. Three from there to the nearest face, which is the depth rather
        // than the profile.
        assert!((field.distance(Point3::new(0.0, 0.0, 3.0)) + 3.0).abs() < 1e-6);
        // Straight out from a side, level with the middle.
        assert!((field.distance(Point3::new(14.0, 0.0, 3.0)) - 4.0).abs() < 1e-6);
        // And above the top, which is now at six rather than at three.
        assert!((field.distance(Point3::new(0.0, 0.0, 8.0)) - 2.0).abs() < 1e-6);
        // The plane itself is the base: on it, and one below it is outside.
        assert!(field.distance(Point3::new(0.0, 0.0, 0.0)).abs() < 1e-6);
        assert!((field.distance(Point3::new(0.0, 0.0, -1.0)) - 1.0).abs() < 1e-6);
    }

    /// Aiming the frame turns the solid, which is what says the frame is being read.
    #[test]
    fn an_extrusion_aimed_along_x_is_thick_along_x() {
        let mut doc = Document::new();
        // Drawn in the plane the frame reads, which for an X-aimed extrude is YZ. A square in
        // XY would be refused as not planar — correctly, and that refusal is its own test.
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(0.0, -10.0, -10.0),
                Point3::new(0.0, 10.0, -10.0),
                Point3::new(0.0, 10.0, 10.0),
                Point3::new(0.0, -10.0, 10.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let mut graph = Graph::empty();
        // **Nothing aims it.** The profile lies in the YZ plane and the extrude finds that for
        // itself, which is the whole of what replaced the frame this test used to set by hand.
        graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(6.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");

        // Three from the middle to the nearest face along X now, and the profile's own reach
        // has moved into Y and Z. The solid runs from the plane rather than about it, so along
        // X it spans [0, 6] and its middle is at three.
        assert!((field.distance(Point3::new(3.0, 0.0, 0.0)) + 3.0).abs() < 1e-6);
        assert!((field.distance(Point3::new(8.0, 0.0, 0.0)) - 2.0).abs() < 1e-6);
        assert!((field.distance(Point3::new(3.0, 14.0, 0.0)) - 4.0).abs() < 1e-6);
    }

    /// A folded pattern answers what trying every instance would, at any count.
    ///
    /// **The property the fold is worth having for.** A pattern used to evaluate its input once
    /// per instance, so a hundred copies cost a hundred bodies and a thousand cost a thousand —
    /// which is a mesh instancing cost wearing a field's clothes, and the thing this whole
    /// architecture exists to avoid. A lattice has never paid it: it folds the query point into
    /// one cell and asks once, which is why a gyroid finer than any mesher could hold traces in
    /// a fraction of a second.
    ///
    /// So the fold has to give the *same answer*, not merely a fast one — an approximation here
    /// would be a field that reports a distance no surface is at. Checked against the honest
    /// minimum over every instance, at a count where the difference in work is a hundredfold.
    #[test]
    fn a_folded_pattern_answers_what_trying_every_instance_would() {
        let radius = 4.0;
        let pitch = 20.0;
        let count = 200u32;

        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(radius),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(f64::from(count)),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(pitch), lit(0.0), lit(0.0)],
            },
            join: true,
            fillet: Fillet::hard(),
        });
        let doc = Document::new();
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");

        // Every instance, by hand, which is what the fold is a shortcut for.
        let honestly = |p: Point3<f64>| {
            (0..count)
                .map(|i| (p - Point3::new(pitch * i as f64, 0.0, 0.0)).norm() - radius)
                .fold(f64::INFINITY, f64::min)
        };

        let mut worst: f64 = 0.0;
        for i in 0..60 {
            for j in 0..20 {
                // Across the whole row and out past both ends, where the window has to stop
                // rather than wrap — a row is not a turn.
                let p = Point3::new(-60.0 + i as f64 * 75.0, -30.0 + j as f64 * 3.0, 2.0);
                worst = worst.max((field.distance(p) - honestly(p)).abs());
            }
        }
        assert!(worst < 1e-9, "the fold disagreed by {worst:.6} mm");
    }

    /// And the same for a turn, where the window wraps instead of stopping.
    #[test]
    fn a_folded_radial_pattern_wraps_rather_than_stopping() {
        let radius = 5.0;
        let out = 60.0;
        let count = 120u32;

        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(out), lit(0.0), lit(0.0)]),
            radius: lit(radius),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(f64::from(count)),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::default(),
            },
            join: true,
            fillet: Fillet::hard(),
        });
        let doc = Document::new();
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");

        let honestly = |p: Point3<f64>| {
            (0..count)
                .map(|i| {
                    let turn = std::f64::consts::TAU * i as f64 / count as f64;
                    let centre = Point3::new(out * turn.cos(), out * turn.sin(), 0.0);
                    (p - centre).norm() - radius
                })
                .fold(f64::INFINITY, f64::min)
        };

        let mut worst: f64 = 0.0;
        for i in 0..40 {
            for j in 0..40 {
                // Including right across the seam at angle zero, which is where a window that
                // clamped instead of wrapping would be wrong.
                let p = Point3::new(-80.0 + i as f64 * 4.0, -80.0 + j as f64 * 4.0, 1.0);
                worst = worst.max((field.distance(p) - honestly(p)).abs());
            }
        }
        assert!(worst < 1e-9, "the fold disagreed by {worst:.6} mm");
    }

    /// A baked profile makes its solid a lower bound, and says so.
    ///
    /// **The contract, not a courtesy.** A grid's reading has a margin subtracted so that
    /// interpolation can never overstate — which is the definition of a lower bound — and
    /// `ScalarField` requires a field returning one to report it, because a blend mixes by
    /// magnitude and an operand reading short pulls the seam toward itself. Claiming an exact
    /// distance while answering from a grid is the silent wrong shape D8 exists to prevent.
    ///
    /// And the other half: a tolerance fine enough not to bake still gets a true distance, which
    /// is what export asks for. So this checks both sides of the same switch.
    #[test]
    fn a_baked_profile_reports_a_lower_bound_and_an_exact_one_does_not() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let mut graph = Graph::empty();
        graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(20.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
        });

        // Coarse enough to bake: a grid inside a millimetre over a twenty-millimetre square.
        let coarse = GraphField::compile_graph(&doc, &graph, 1.0).expect("compiles");
        assert_eq!(
            coarse.normalization(),
            Normalization::LowerBound,
            "a baked profile has to announce that it reads short",
        );

        // Fine enough not to: the boundary is walked and the distance is true.
        let fine = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
        assert_eq!(
            fine.normalization(),
            Normalization::Distance,
            "an unbaked profile is still exact, which is what export gets",
        );
    }

    /// A revolve keeps working when its axis is tilted off the profile's plane.
    ///
    /// **The case this was reported on**: draw a profile, aim the block at a line, nudge the
    /// line, and the block stopped compiling -- the old rule demanded the profile lie in one
    /// half-plane about the axis, and coplanarity is not something anybody can hold by hand.
    ///
    /// A small tilt is the case that has to be *right*, not merely allowed: the solid must stay
    /// where it was. Larger ones are the test below, which records what they cost.
    #[test]
    fn a_revolve_survives_its_axis_being_nudged_off_the_profile_s_plane() {
        let mut doc = Document::new();
        // A square out at y in [10, 30], in the YZ plane, so world Z lies in its plane.
        let profile = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Section".to_string(),
            vec![
                Point3::new(0.0, 10.0, -10.0),
                Point3::new(0.0, 30.0, -10.0),
                Point3::new(0.0, 30.0, 10.0),
                Point3::new(0.0, 10.0, 10.0),
            ],
            true,
        )
        .expect("closed");
        let id = profile.id;
        doc.curves.insert(id, profile);

        let revolve = |axis: [f64; 3]| {
            let mut graph = Graph::empty();
            graph.add(Node::Revolve { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                frame: fieldform_scene::Frame::At {
                    origin: [lit(0.0); 3],
                    axis: [lit(axis[0]), lit(axis[1]), lit(axis[2])],
                },
                angle: fieldform_scene::ParamRef::Literal(360.0),
                sides: fieldform_scene::AngleSides::One,
            });
            GraphField::compile_graph(&doc, &graph, 1e-3)
        };

        let flat = revolve([0.0, 0.0, 1.0]).expect("in the plane, as before");
        // Two degrees out of the profile's plane, which is a nudge of the line's far end.
        let tilted = revolve([0.035, 0.0, 1.0]).expect("and still compiles once tilted");

        // A tube from r = 10 to r = 30, still in the same place. The tilt does move it -- that
        // is what tilting an axis is for -- by about the radius times the tangent of the angle,
        // so 30 mm out and two degrees over is a millimetre. What must not happen is the shape
        // changing into something else, which is what the fold would do.
        let mut worst: f64 = 0.0;
        for i in 0..30 {
            for j in 0..30 {
                let p = Point3::new(-40.0 + i as f64 * 2.8, 0.0, -20.0 + j as f64 * 1.4);
                worst = worst.max((flat.distance(p) - tilted.distance(p)).abs());
            }
        }
        // Bounded by the lean the tilt causes at the far edge, not by nothing.
        assert!(worst < 1.5, "a two degree tilt moved the field by {worst:.3} mm");

        // And it is still the same solid: a tube with a hollow middle and a bounded outside.
        assert!(tilted.distance(Point3::new(20.0, 0.0, 0.0)) < 0.0, "solid through the wall");
        assert!(tilted.distance(Point3::new(0.0, -20.0, 0.0)) < 0.0, "all the way round");
        assert!(tilted.distance(Point3::origin()) > 0.0, "hollow down the middle");
        assert!(tilted.distance(Point3::new(45.0, 0.0, 0.0)) > 0.0, "and bounded outside");
    }

    /// Far off the plane, a revolve turns the profile's shadow rather than the profile.
    ///
    /// **Recorded because it is the price of the test above, and it is not obvious.** The solid
    /// is built from the profile's *boundary* mapped into `(r, z)`, and the image of a boundary
    /// is not the boundary of an image: a square held out to the side has both of its sides map
    /// to the same radius, and they cancel under the even-odd fill rather than enclosing the
    /// ring somebody might have pictured. What is left is a true rendering of the footprint and
    /// not the shape that was drawn.
    ///
    /// So this is the limit of the relaxation, held here as a fact rather than as a warning:
    /// near the profile's plane a revolve turns what you drew, and far from it, its shadow.
    #[test]
    fn far_off_the_plane_a_revolve_turns_the_profile_s_shadow() {
        let mut doc = Document::new();
        // A square in the plane y = 20, spread about 53 degrees around the world Z axis.
        let profile = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, 20.0, -10.0),
                Point3::new(10.0, 20.0, -10.0),
                Point3::new(10.0, 20.0, 10.0),
                Point3::new(-10.0, 20.0, 10.0),
            ],
            true,
        )
        .expect("closed");
        let id = profile.id;
        doc.curves.insert(id, profile);

        let mut graph = Graph::empty();
        graph.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(id), frame: fieldform_scene::Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3)
            .expect("it compiles rather than refusing, which is the change");

        // The axis is hollow, and it is bounded outside: a solid, just not the drawn one.
        assert!(field.distance(Point3::origin()) > 0.0, "hollow down the axis");
        assert!(field.distance(Point3::new(40.0, 0.0, 0.0)) > 0.0, "and bounded outside");
        // The spread the panel reports, which is how somebody would know they are here.
        let world = crate::profile_loop(&doc, id, None).expect("a loop").points;
        let span = crate::Profile::azimuth_span(&world, |p| p);
        assert!((span - 53.13).abs() < 0.5, "spans about 53 degrees, measured {span:.2}");
    }

    /// A profile that is not in the ground plane extrudes, straight out and typed alike.
    ///
    /// **Every other test here draws on the ground**, where the profile's frame is the identity
    /// and a world direction and a local one are the same vector. That is exactly the case that
    /// cannot catch a missing rotation, and the application draws on whatever plane the person
    /// picked -- YZ in the report this came from.
    #[test]
    fn a_profile_off_the_ground_plane_extrudes_both_ways() {
        // A square in the YZ plane, so its normal is +X and no rotation is the identity.
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(0.0, -10.0, -10.0),
                Point3::new(0.0, 10.0, -10.0),
                Point3::new(0.0, 10.0, 10.0),
                Point3::new(0.0, -10.0, 10.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        // The normal the panel would seed a typed sweep with.
        let found = crate::profile_loop(&doc, id, None).expect("a loop");
        let (_, normal) = crate::plane_of(&found.points).expect("planar");

        for sweep in [
            fieldform_scene::Sweep::Normal,
            fieldform_scene::Sweep::Toward {
                xyz: [lit(normal.x), lit(normal.y), lit(normal.z)],
            },
        ] {
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(), outline: fieldform_scene::Outline::closed(id), sweep: sweep.clone(), height: lit(20.0), draft: lit(0.0), start: lit(0.0), sides: fieldform_scene::Sides::One, corners: Default::default() });
            let field = GraphField::compile_graph(&doc, &graph, 1e-3)
                .unwrap_or_else(|e| panic!("{sweep:?} should compile, got {e}"));
            // Twenty deep *from* the plane, so the middle of the solid is ten along X and the
            // plane itself is its base rather than its waist.
            assert!(field.distance(Point3::new(10.0, 0.0, 0.0)) < 0.0, "{sweep:?}: inside");
            assert!(field.distance(Point3::new(30.0, 0.0, 0.0)) > 0.0, "{sweep:?}: outside");
            assert!(field.distance(Point3::new(-1.0, 0.0, 0.0)) > 0.0, "{sweep:?}: not behind");
            // And it is swept along X, not along Z: the profile spans only 10 in Z.
            assert!(field.distance(Point3::new(0.0, 0.0, 30.0)) > 0.0, "{sweep:?}: not up");
        }
    }

    /// Typing the profile's own normal as a direction is exactly as good as sweeping straight out.
    ///
    /// **The surface-agreement check cannot see this**, which is why it needs a test of its own:
    /// scaling a field about its own zero set moves no surface, so the GPU/CPU comparison —
    /// which measures where the surface *is* — reported a clean 0.0132 mm while the two sides
    /// disagreed about every distance off it. What was wrong: `SkewExtrusion` never took the
    /// corner outside two faces, on the grounds that a shear destroys the perpendicularity
    /// Pythagoras needs. True, but a sweep along the normal is not sheared at all, so choosing
    /// "Typed" on a working extrude and touching nothing quietly cost up to a third of the
    /// distance at the corners.
    #[test]
    fn a_sweep_typed_along_the_normal_is_as_exact_as_a_straight_one() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let build = |sweep: fieldform_scene::Sweep| {
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(), outline: fieldform_scene::Outline::closed(id), sweep, height: lit(20.0), draft: lit(0.0), start: lit(0.0), sides: fieldform_scene::Sides::One, corners: Default::default() });
            GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles")
        };
        let straight = build(fieldform_scene::Sweep::Normal);
        let typed = build(fieldform_scene::Sweep::Toward {
            xyz: [lit(0.0), lit(0.0), lit(1.0)],
        });

        // Everywhere, not only at the corner the fix was about: the two are the same field.
        let mut worst: f64 = 0.0;
        for i in 0..20 {
            for j in 0..20 {
                for k in 0..20 {
                    let p = Point3::new(
                        -25.0 + i as f64 * 2.5,
                        -25.0 + j as f64 * 2.5,
                        -25.0 + k as f64 * 2.5,
                    );
                    worst = worst.max((straight.distance(p) - typed.distance(p)).abs());
                }
            }
        }
        assert!(worst < 1e-9, "typing the normal cost {worst:.4} mm of distance");
    }

    /// A drafted extrude's extent holds the walls the draft moved out, on both sides of a
    /// symmetric one.
    ///
    /// **Found by the step-11 design review, and live before step 11 existed.** `extent_of`'s
    /// extrude arm bound the draft into its `..` and boxed only the drawn profile, so a frustum
    /// flaring 20 mm on every side had a box of the base alone — and the ray's start at the scene
    /// sphere, the view's framing and the bake's bounds all read that box.
    #[test]
    fn a_drafted_extrudes_extent_holds_the_walls_the_draft_moved_out() {
        // The frustum `a_draft_makes_a_frustum_by_moving_the_walls_not_the_shape` builds below:
        // ten either side, twenty tall, forty-five degrees, so eight up the wall is at eighteen.
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);
        for symmetric in [false, true] {
            let mut graph = Graph::empty();
            let node = graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(20.0),
                draft: lit(45.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::symmetric_if(symmetric),
                corners: Default::default(),
            });
            let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
            let (lo, hi) = match graph_extent(&doc, &graph, node) {
                Extent::Bounded { lo, hi } => (lo, hi),
                other => panic!("a drafted extrude is bounded, not {other:?}"),
            };
            // **Every point the field calls solid lies inside the box.** An extent was a hint —
            // this function's own doc said a wrong box "costs resolution" — until the ray's start
            // at the scene sphere, the view's framing and the per-tile body cull all read it as a
            // guarantee. A frustum flares by |height| · tan(draft) on every side, so sampling out
            // to twice that is what would find a box that stopped at the drawn profile.
            let mut outside = Vec::new();
            for xi in -30..=30 {
                for zi in -25..=25 {
                    let p = Point3::new(f64::from(xi), 0.0, f64::from(zi));
                    let inside_box = (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]);
                    if field.distance(p) < 0.0 && !inside_box {
                        outside.push(p);
                    }
                }
            }
            assert!(
                outside.is_empty(),
                "symmetric {symmetric}: {} solid points outside the extent {lo:?}..{hi:?}, first {:?}",
                outside.len(),
                outside.first()
            );
        }
    }

    /// A leaning extrude's box ends where the solid does, drafted or not.
    ///
    /// **The depth is measured along the sweep**, so a square pulled twenty millimetres along
    /// a direction forty-five degrees off its normal rises `20 / √2` and no further. The box
    /// carried it to a height of twenty: too long for a plain lean, which nothing reports, and
    /// with a draft too short behind — the flare was padded where the far end was not, so the
    /// real far end's trailing wall stood outside (`tile-mask-design-01941c5.md` §1.8 R4).
    ///
    /// `extent_holds_profile_solids` says no solid point is outside the box. This says the
    /// other half, in closed form: the box is the one the two ends make, and no bigger.
    #[test]
    fn a_leaning_extrudes_extent_ends_where_the_solid_does() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);
        // How far the far end is carried, across and up: twenty along (1, 0, 1).
        let s = 20.0 / 2.0f64.sqrt();
        for (draft, symmetric, lo, hi) in [
            (0.0, false, [-10.0, -10.0, 0.0], [10.0 + s, 10.0, s]),
            (0.0, true, [-10.0 - s / 2.0, -10.0, -s / 2.0], [10.0 + s / 2.0, 10.0, s / 2.0]),
            // Forty-five degrees of draft moves the far walls out by the whole depth, which is
            // more than the lean carries them across: the far end's trailing wall is behind
            // the base's, at `s - 30` where the base stops at -10.
            (45.0, false, [s - 30.0, -30.0, 0.0], [30.0 + s, 30.0, s]),
            (45.0, true, [-20.0 - s / 2.0, -20.0, -s / 2.0], [20.0 + s / 2.0, 20.0, s / 2.0]),
        ] {
            let mut graph = Graph::empty();
            let node = graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Toward { xyz: [lit(1.0), lit(0.0), lit(1.0)] },
                height: lit(20.0),
                draft: lit(draft),
                start: lit(0.0),
                sides: fieldform_scene::Sides::symmetric_if(symmetric),
                corners: Default::default(),
            });
            let (got_lo, got_hi) = match graph_extent(&doc, &graph, node) {
                Extent::Bounded { lo, hi } => (lo, hi),
                other => panic!("a leaning extrude is bounded, not {other:?}"),
            };
            for k in 0..3 {
                assert!(
                    (got_lo[k] - lo[k]).abs() < 1e-9 && (got_hi[k] - hi[k]).abs() < 1e-9,
                    "draft {draft}, symmetric {symmetric}: {got_lo:?}..{got_hi:?}, not {lo:?}..{hi:?}"
                );
            }
            // And the field agrees about the wall that was left out: just inside the box's
            // trailing face, under the far end, is solid, and just outside it is not.
            if draft > 0.0 && !symmetric {
                let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
                let under_the_far_end = |x: f64| Point3::new(x, 0.0, s - 0.05);
                assert!(field.distance(under_the_far_end(lo[0] + 0.05)) < 0.0);
                assert!(field.distance(under_the_far_end(lo[0] - 0.05)) > 0.0);
            }
        }
    }

    /// A draft makes a frustum: wider at the top, narrower at the bottom, waist as drawn.
    ///
    /// **An offset, not a scale**, and this is what says so. The square here is centred, so a
    /// scale and an offset would agree about it — the *second* profile is deliberately off to
    /// one side, where they part company: scaling would slide it away from the origin as it
    /// tapered, while an offset leaves its middle where it was drawn and only moves its walls.
    #[test]
    fn a_draft_makes_a_frustum_by_moving_the_walls_not_the_shape() {
        let mut doc = Document::new();
        // Ten either side of the origin, twenty tall, drafted forty-five degrees so the
        // arithmetic is checkable by hand: the wall moves out by exactly the height it rises.
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let mut graph = Graph::empty();
        graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(20.0),
            draft: lit(45.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");

        // **The base is the shape as drawn**, since an extrude starts at its plane — which is
        // what a moulding drawing means and what makes the draft dimensionable. Just inside ten
        // is solid at the base, just outside is not.
        assert!(field.distance(Point3::new(9.0, 0.0, 1.0)) < 0.0, "the base is as drawn");
        assert!(field.distance(Point3::new(11.0, 0.0, 1.0)) > 0.0, "and stops where drawn");

        // Eight up, the wall has moved out by eight: seventeen is inside, nineteen is not.
        assert!(field.distance(Point3::new(17.0, 0.0, 8.0)) < 0.0, "wider above");
        assert!(field.distance(Point3::new(19.0, 0.0, 8.0)) > 0.0, "but not unboundedly");

        // And it opens the one way: below the plane there is no solid at all now.
        assert!(field.distance(Point3::new(0.0, 0.0, -1.0)) > 0.0, "nothing below the plane");
    }

    /// A negative depth grows the other way out of the plane, draft and all.
    ///
    /// **Which side only became a question when the extrude stopped straddling.** Centred, the
    /// solid went both ways and the sign of the depth meant nothing; starting at the plane, it
    /// is the whole answer to "which side" — so it has to be a number the block already carries
    /// rather than a flag beside it, or the two could disagree.
    ///
    /// The draft has to follow it. A flipped extrude still opens *away* from its base: measured
    /// from the plane along the extrusion rather than along the axis, which is the same thing
    /// only while the depth is positive.
    #[test]
    fn a_negative_depth_extrudes_the_other_way_and_takes_its_draft_with_it() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let build = |depth: f64| {
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(depth),
                draft: lit(45.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles")
        };

        let up = build(20.0);
        let down = build(-20.0);

        // Each grows its own way and not the other.
        assert!(up.distance(Point3::new(0.0, 0.0, 10.0)) < 0.0, "up is solid above");
        assert!(up.distance(Point3::new(0.0, 0.0, -10.0)) > 0.0, "and hollow below");
        assert!(down.distance(Point3::new(0.0, 0.0, -10.0)) < 0.0, "down is solid below");
        assert!(down.distance(Point3::new(0.0, 0.0, 10.0)) > 0.0, "and hollow above");

        // And the draft opens away from the base in both, rather than closing into it: eight
        // from the plane, the wall has moved out by eight either way.
        assert!(up.distance(Point3::new(17.0, 0.0, 8.0)) < 0.0, "up widens away");
        assert!(down.distance(Point3::new(17.0, 0.0, -8.0)) < 0.0, "down widens away too");

        // A mirror image, which is the whole claim: the same solid reflected in its own plane.
        let mut worst: f64 = 0.0;
        for i in 0..30 {
            for j in 0..30 {
                let x = -30.0 + i as f64 * 2.0;
                let z = -25.0 + j as f64 * 1.7;
                worst = worst.max(
                    (up.distance(Point3::new(x, 0.0, z))
                        - down.distance(Point3::new(x, 0.0, -z)))
                    .abs(),
                );
            }
        }
        assert!(worst < 1e-9, "the two are not reflections: {worst:.6} mm apart");
    }

    /// A drafted extrude never says a point is further away than it is.
    ///
    /// **A sloped wall reads long**, which is the direction a `ScalarField` may never be wrong
    /// in: a ray told too large a number steps through the surface. The correction is the
    /// secant of the draft angle and it is a single multiply, which makes it exactly the kind
    /// of thing that is easy to leave out, apply twice, or apply the wrong way round — and
    /// nothing about the picture would look wrong if it were, because the zero set is
    /// unaffected. Only a measurement catches it.
    #[test]
    fn a_drafted_extrude_never_overstates_how_far_away_it_is() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let mut graph = Graph::empty();
        graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(20.0),
            draft: lit(30.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
        assert_eq!(field.normalization(), Normalization::LowerBound, "and it says so");

        // Marched from outside: the value at a point can never exceed the distance to the
        // nearest point that is actually inside.
        let mut worst: f64 = 0.0;
        for i in 0..36 {
            for j in 0..36 {
                let p = Point3::new(-35.0 + i as f64 * 2.0, 0.0, -25.0 + j as f64 * 1.4);
                let claimed = field.distance(p);
                if claimed <= 0.0 {
                    continue;
                }
                let mut nearest = f64::INFINITY;
                for a in 0..70 {
                    for b in 0..70 {
                        let q = Point3::new(
                            -35.0 + a as f64 * 1.0,
                            0.0,
                            -25.0 + b as f64 * 0.72,
                        );
                        if field.distance(q) < 0.0 {
                            nearest = nearest.min((q - p).norm());
                        }
                    }
                }
                if nearest.is_finite() {
                    worst = worst.max(claimed - nearest);
                }
            }
        }
        assert!(worst <= 1e-9, "overstated the distance by {worst:.4} mm");
    }

    /// **A lean does not shorten a solid**, whatever the panel's warning says next to it.
    ///
    /// The warning quotes a percentage, and a percentage beside a depth field reads as a
    /// percentage *of the depth* -- which is what it was read as. It is not: it is how short
    /// the field reports the space *around* the body, which is a `LowerBound` and costs a
    /// fillet its radius, not the solid its size. The depth is measured along the sweep and the
    /// solid is exactly that deep at every lean.
    ///
    /// Asserted here rather than left to the panel's wording, because wording is not a
    /// guarantee and this is.
    #[test]
    fn a_leaning_extrude_is_as_deep_as_it_was_asked_to_be() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let depth = 20.0;
        for degrees in [0.0f64, 10.0, 32.0, 45.0, 60.0, 75.0] {
            let lean = degrees.to_radians();
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Toward {
                    xyz: [lit(lean.sin()), lit(0.0), lit(lean.cos())],
                },
                height: lit(depth),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");

            // Along the sweep from the middle of the profile, until the material stops.
            let dir = nalgebra::Vector3::new(lean.sin(), 0.0, lean.cos());
            let step = 1e-4;
            let mut t = 0.0;
            while t < depth * 3.0
                && field.distance(Point3::origin() + dir * (t + step)) <= 0.0
            {
                t += step;
            }
            let measured = t + step;
            assert!(
                (measured - depth).abs() < 1e-3,
                "a {degrees} degree lean came out {measured:.4} mm deep, not {depth}"
            );
        }
    }

    /// **A fillet reads its operands out to its radius, so the band has to go that far too.**
    ///
    /// The same defect a draft hit, with a different consumer. A baked profile answers finely
    /// only inside its band; past it the coarse level answers, with a margin eight times wider
    /// and blocks eight cells across. A blend is exactly the operation that turns a step in an
    /// operand into a visible crease, and a 10 mm fillet against a band four tolerances wide
    /// reads almost nothing but coarse -- which reached the owner as terraces on the surface.
    ///
    /// The giveaway was that the picture improved *while being dragged*: a draft frame is
    /// compiled coarser, and the band was `4 x tolerance`, so a worse setting bought a wider
    /// band. Quality up, picture worse.
    ///
    /// Asserted as the field being smooth where the fillet reads, because that is what a blend
    /// needs from an operand and what a terrace is the absence of.
    #[test]
    fn a_fillet_reads_a_baked_profile_over_ground_the_band_covers() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-20.0, -20.0, 0.0),
                Point3::new(20.0, -20.0, 0.0),
                Point3::new(20.0, 20.0, 0.0),
                Point3::new(-20.0, 20.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        for radius in [1.0f64, 5.0, 10.0] {
            let mut graph = Graph::empty();
            let solid = graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(16.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(26.0), lit(0.0), lit(8.0)]),
                radius: lit(12.0),
            });
            graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: solid.into(),
                b: ball.into(),
                fillet: Fillet { radius: lit(radius), continuity: fieldform_scene::Continuity::G1 },
            });

            // The band has to reach the fillet's radius either side of the boundary, whatever
            // the tolerance -- that is the whole change, since it used to reach four tolerances.
            let (lo, hi) = profile_reads_out_to(&doc, &graph, solid);
            assert!(
                hi >= radius && lo <= -radius,
                "a {radius} mm fillet needs the band to cover +/-{radius}, got {lo} to {hi}"
            );

            // And baking changes that field smoothly rather than in steps.
            //
            // **Against the unbaked scene, not against unit gradient.** A blend's own field is
            // not a distance: where its two operands' gradients align, `smooth_min` reaches
            // `sqrt(2)` by construction. So the property is that the *bake* adds nothing sharp
            // -- a difference that varies smoothly moves the surface, one that steps creases
            // it, and a crease is what a terrace is.
            let baked = GraphField::compile_graph(&doc, &graph, 0.25).expect("compiles");
            let exact = GraphField::compile_graph(&doc, &graph, 1e-5).expect("compiles");
            let step = 0.0025;
            let mut roughest: f64 = 0.0;
            let mut worst: f64 = 0.0;
            let mut z = 1.0;
            while z < 15.0 {
                let mut x = 20.0 - radius;
                let mut previous = f64::NAN;
                while x < 20.0 + radius {
                    let at = Point3::new(x, 0.0, z);
                    let gap = baked.distance(at) - exact.distance(at);
                    worst = worst.max(gap.abs());
                    if previous.is_finite() {
                        roughest = roughest.max(((gap - previous) / step).abs());
                    }
                    previous = gap;
                    x += step;
                }
                z += 0.5;
            }
            assert!(
                roughest < 1.0,
                "a {radius} mm fillet's operand was changed by baking at {roughest:.1} mm per \
                 mm -- a step, which a blend turns into a terrace"
            );
            assert!(
                worst < 0.5,
                "a {radius} mm fillet's operand moved {worst:.3} mm when baked"
            );
        }
    }

    /// **The band has to cover the contours a draft reads, not just the profile's boundary.**
    ///
    /// A baked profile answers finely near its own zero set and coarsely everywhere else, and
    /// the two levels carry different margins -- so the edge of the band is a *step* in the
    /// value. That never mattered while the only thing anyone read was the boundary itself,
    /// which is what the band is built around.
    ///
    /// A draft moves the surface off it. At height `z` a drafted wall is the `d = z*tan(draft)`
    /// contour, so a 12 mm extrude at 15 degrees reads out to 3.2 mm while the band was four
    /// tolerances -- and the wall landed on the step. Measured at 174 mm of value per millimetre
    /// of travel, where a distance may change by at most one: a ray told it has that much room
    /// steps straight through the surface, which reached the owner as walls that pinched to a
    /// knife edge and speckled.
    ///
    /// The draft tests and the bake tests both missed it because each held the other half
    /// fixed -- the draft tests compile too fine to bake, and the bake tests have no draft.
    /// `gpu_matches_cpu` missed it for the same reason and now carries the pair as a scene.
    #[test]
    fn a_drafted_extrude_stays_a_distance_across_the_contours_it_reads() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let depth = 12.0;
        let draft = 15.0;
        for angle in [draft, -draft] {
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(depth),
                draft: lit(angle),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            // Coarse enough to bake, which is the whole point: at a display tolerance the
            // profile answers from a grid, and that is where the step lives.
            let tolerance = 0.25;
            let field = GraphField::compile_graph(&doc, &graph, tolerance).expect("compiles");

            let reads = depth * draft.to_radians().tan();
            // Finer than a cell, because the step is one cell wide and a probe wider than it
            // walks straight past -- which is how a first attempt at this measurement reported
            // that everything was fine.
            let step = tolerance * 0.01;
            let mut steepest: f64 = 0.0;
            let mut at = (0.0, 0.0);
            // The contours the wall passes through, and half the band's own slack either side
            // -- which is what a normal's central difference reaches for. Not the whole
            // interior: the band covers the side a draft reads and four tolerances of margin,
            // deliberately, because banding both ways doubles the blocks to store contours
            // nothing asks for. `BakedProfile::new` says so where the sign is taken.
            let (near, far) = if angle > 0.0 {
                (10.0 - 0.5, 10.0 + reads + 0.5)
            } else {
                (10.0 - reads - 0.5, 10.0 + 0.5)
            };
            for level in 1..=11 {
                let z = level as f64 * depth / 12.0;
                let mut x = near;
                let mut previous = field.distance(Point3::new(x, 0.0, z));
                while x < far {
                    x += step;
                    let here = field.distance(Point3::new(x, 0.0, z));
                    let rate = ((here - previous) / step).abs();
                    if rate > steepest {
                        steepest = rate;
                        at = (x, z);
                    }
                    previous = here;
                }
            }
            assert!(
                steepest < 1.05,
                "a {angle} degree draft changed by {steepest:.1} mm per mm at \
                 ({:.2}, {:.2}) -- a distance cannot, so a ray steps through the surface there",
                at.0,
                at.1
            );
        }
    }

    /// A leaning sweep shears the solid, and never overstates the distance while doing it.
    ///
    /// **The exactness is the part worth asserting.** A shear does not preserve distance: the
    /// profile is measured in sheared coordinates, so its reading is *longer* than the truth,
    /// which is the one thing a `ScalarField` may never be. The correction is a single factor
    /// and it is easy to leave out, drop, or apply the wrong way round — and nothing about the
    /// shape on screen would look wrong if it were, because the zero set is unaffected. Only a
    /// measurement catches it.
    #[test]
    fn a_leaning_extrude_never_overstates_how_far_away_it_is() {
        let mut doc = Document::new();
        // A square on the ground, so the normal is +Z and a lean is easy to reason about.
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        // Forty-five degrees over in X: the top face sits directly above where the bottom's
        // trailing edge was.
        let mut graph = Graph::empty();
        graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Toward {
                xyz: [lit(1.0), lit(0.0), lit(1.0)],
            },
            height: lit(20.0),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("a lean compiles");

        // It leans the way it was told. Halfway up the sweep the solid has shifted in +X by as
        // much as it has risen, so the centre line runs diagonally: a point on that line is
        // inside, and the point the same height above the *original* centre is not, because the
        // solid has moved out from under it.
        // A quarter of the way up rather than at the very top, which is the face itself and
        // measures as zero to within rounding -- an assertion that would pass or fail on the
        // last bit of a float is not an assertion.
        let up = 5.0 * (0.5_f64).sqrt();
        assert!(field.distance(Point3::new(up, 0.0, up)) < 0.0, "leaned");
        assert!(
            field.distance(Point3::new(-up - 17.0, 0.0, up)) > 0.0,
            "not the other way"
        );

        // And it reports as a lower bound, which is what says the reading is short rather than
        // merely different.
        assert_eq!(field.normalization(), Normalization::LowerBound);

        // Never further than the truth. Marched from outside toward the solid: the value at a
        // point can never exceed the distance to the nearest point that is actually inside.
        let mut worst: f64 = 0.0;
        for i in 0..14 {
            for j in 0..14 {
                let p = Point3::new(
                    -30.0 + i as f64 * 4.0,
                    0.0,
                    -20.0 + j as f64 * 4.0,
                );
                let claimed = field.distance(p);
                if claimed <= 0.0 {
                    continue;
                }
                // The nearest inside point found by sampling; the claim must not exceed it.
                let mut nearest = f64::INFINITY;
                for a in 0..70 {
                    for b in 0..70 {
                        let q = Point3::new(
                            -30.0 + a as f64 * 1.0,
                            0.0,
                            -20.0 + b as f64 * 0.8,
                        );
                        if field.distance(q) < 0.0 {
                            nearest = nearest.min((q - p).norm());
                        }
                    }
                }
                if nearest.is_finite() {
                    worst = worst.max(claimed - nearest);
                }
            }
        }
        assert!(worst <= 1e-9, "overstated the distance by {worst:.4} mm");
    }

    /// A rectangle clear of the axis revolves into a tube of those radii.
    #[test]
    fn a_revolved_rectangle_is_a_tube() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Wall".to_string(),
            vec![
                Point3::new(10.0, 0.0, -4.0),
                Point3::new(20.0, 0.0, -4.0),
                Point3::new(20.0, 0.0, 4.0),
                Point3::new(10.0, 0.0, 4.0),
            ],
            true,
        )
        .expect("closed");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let mut graph = Graph::empty();
        graph.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(id), frame: fieldform_scene::Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");

        // Mid-wall, and the same all the way round — which is the whole claim of a revolve.
        for (x, y) in [(15.0, 0.0), (0.0, 15.0), (-15.0, 0.0), (10.6, 10.6)] {
            let d = field.distance(Point3::new(x, y, 0.0));
            let r = (x as f64).hypot(y);
            let expect = -(4.0f64).min((r - 10.0).min(20.0 - r));
            assert!((d - expect).abs() < 1e-3, "at ({x}, {y}) got {d}, wanted {expect}");
        }
        // The bore is empty.
        assert!(field.distance(Point3::new(0.0, 0.0, 0.0)) > 0.0);
    }

    /// An open curve with nothing to join to is refused, rather than producing a solid with no
    /// inside.
    ///
    /// **The refusal moved rather than went.** A profile used to have to be one closed curve, and
    /// this asserted that by name. It is now a closed *loop*, which several open curves can make
    /// between them — so an open curve is only refused once the walk has looked for a neighbour
    /// and found none. What it says has improved with the move: not "this curve is open", which
    /// is a rule, but how wide the gap is and where, which is somewhere to go and look.
    #[test]
    fn an_open_profile_is_refused_rather_than_guessed_at() {
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Open".to_string(),
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
            ],
            false,
        )
        .expect("an open polyline is a curve");
        let id = curve.id;
        doc.curves.insert(id, curve);

        let mut graph = Graph::empty();
        graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(5.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
        });
        let refused = GraphField::compile_graph(&doc, &graph, 1e-3);
        assert!(matches!(
            refused,
            Err(CompileError::BadLoop(_, crate::LoopError::NotClosed { .. })),
        ));
        // And it names the gap, which is the half worth having.
        let Err(e) = refused else { panic!("an open curve alone is not a profile") };
        let said = e.to_string();
        assert!(said.contains("gap"), "{said}");
    }

    /// Two curves meeting at both ends are a profile.
    ///
    /// **The case the owner reported as still not working**, and the smallest loop that can
    /// exist: two straight lines meeting at both ends enclose nothing, so a two-curve loop has
    /// to bow. Worth its own test beside the three-line one, because two is where the walk both
    /// starts and finishes on its second piece and the closure check has the least room.
    #[test]
    fn two_curves_meeting_at_both_ends_are_a_profile() {
        let mut doc = Document::new();
        // Two arcs, as degree-2 Beziers bowing opposite ways between the same two points.
        let ends = [Point3::new(0.0, 0.0, 0.0), Point3::new(40.0, 0.0, 0.0)];
        let mut ids = Vec::new();
        for bow in [12.0_f64, -12.0] {
            let curve = fieldform_scene::Curve::new(
                fieldform_scene::CurveId::new(),
                format!("Arc {bow}"),
                2,
                vec![ends[0], Point3::new(20.0, bow, 0.0), ends[1]],
                vec![1.0; 3],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                false,
            )
            .expect("three points make a degree-2 curve");
            ids.push(curve.id);
            doc.curves.insert(curve.id, curve);
        }

        for seat in &ids {
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(*seat),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(5.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            let field = GraphField::compile_graph(&doc, &graph, 1e-3)
                .expect("two arcs meeting at both ends are a profile");
            // Inside the lens they enclose, and up inside the depth: the plane is the base of
            // an extrusion now, so a point on it is on the surface rather than within.
            assert!(field.distance(Point3::new(20.0, 0.0, 2.5)) < 0.0);
            // Outside it, past the wider of the two bows.
            assert!(field.distance(Point3::new(20.0, 20.0, 2.5)) > 0.0);
        }

        // And the walk says it used both, which is what the block reports.
        let found = crate::profile_loop(&doc, ids[0], None).expect("a loop");
        assert_eq!(found.curves.len(), 2);
    }

    /// Three open curves meeting end to end are a profile, and any of them will do as the seat.
    ///
    /// **The case the user hit**: two curves drawn with their ends touching, and an extrude that
    /// would not take them. A profile was one curve, and it is a loop now — walked outward from
    /// whichever curve the node names, so a triangle drawn as three lines extrudes without
    /// anything being joined first.
    #[test]
    fn open_curves_whose_ends_meet_make_a_profile() {
        let corners = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(20.0, 0.0, 0.0),
            Point3::new(0.0, 20.0, 0.0),
        ];
        let mut doc = Document::new();
        let mut ids = Vec::new();
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let curve = fieldform_scene::Curve::polyline(
                fieldform_scene::CurveId::new(),
                format!("Side {a}"),
                vec![corners[a], corners[b]],
                false,
            )
            .expect("two points are a curve");
            ids.push(curve.id);
            doc.curves.insert(curve.id, curve);
        }

        // Seeded from each side in turn, because the walk must not depend on which one is named.
        for seat in &ids {
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(*seat),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(6.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            let field = GraphField::compile_graph(&doc, &graph, 1e-3)
                .expect("three lines meeting end to end are a profile");
            // Inside the triangle and within the extrusion, which runs up from the plane.
            assert!(field.distance(Point3::new(4.0, 4.0, 3.0)) < 0.0);
            // Outside it, in the corner the hypotenuse cuts off.
            assert!(field.distance(Point3::new(18.0, 18.0, 3.0)) > 0.0);
            // And past the height.
            assert!(field.distance(Point3::new(4.0, 4.0, 20.0)) > 0.0);
        }
    }

    /// A profile straddling the axis folds onto one side rather than being refused.
    ///
    /// **This asserted a refusal until the owner reversed it.** The rule was that a profile had
    /// to lie in one half-plane about its axis, and it was doing more harm than good: it also
    /// refused every profile whose axis had merely been nudged, which is most of them. What
    /// happens instead is the fold the old comment described -- the far side lands on the near
    /// one -- and the field is an exact distance to whatever that leaves. It is still not the
    /// shape somebody drew, so the panel reports the spread.
    #[test]
    fn a_profile_crossing_the_revolve_axis_folds_onto_one_side() {
        let mut doc = Document::new();
        let curve = closed_square(10.0);
        let id = curve.id;
        doc.curves.insert(id, curve);

        // The square is centred on the origin, so revolving it about Z sweeps material through
        // the axis and back out the other side.
        let mut graph = Graph::empty();
        graph.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(id), frame: fieldform_scene::Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3)
            .expect("it builds rather than refusing");
        // Bounded by the reach of the fold, which is the square's own diagonal.
        assert!(field.distance(Point3::new(30.0, 0.0, 0.0)) > 0.0);

        // And the spread is a full half turn, which is what the panel shows somebody.
        let world = crate::profile_loop(&doc, id, None).expect("a loop").points;
        assert!((crate::Profile::azimuth_span(&world, |p| p) - 180.0).abs() < 1.0);
    }

    /// A frame aimed at something that cannot supply one is refused by name.
    #[test]
    fn a_frame_taken_from_something_that_is_not_a_plane_is_refused() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(5.0) });
        graph.add(Node::Lattice {
            cell: fieldform_scene::LatticeCell::Gyroid,
            map: fieldform_scene::CellMap::Rectangular { cell: lit(10.0) },
            phase: LatticePhase::sheet(lit(1.0)),
            frame: fieldform_scene::Frame::From { node: ball },
        });
        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3),
            Err(CompileError::NotAFrameSource(_))
        ));
    }

    /// Aiming a cylindrical lattice down X makes it wrap X.
    ///
    /// The property the whole frame exists for. Judged by where the field is solid rather than
    /// by inspecting the ops: a cylindrical map's cells close up near its own axis (see
    /// `LatticeMap::degenerate_radius`), so the axis is the line the material is continuous
    /// along — and that line moves when the lattice is aimed.
    #[test]
    fn aiming_a_cylindrical_lattice_moves_the_axis_it_wraps() {
        let doc = Document::new();
        let build = |axis: [f64; 3]| {
            let mut graph = Graph::empty();
            graph.add(Node::Lattice {
                cell: fieldform_scene::LatticeCell::Gyroid,
                map: fieldform_scene::CellMap::Cylindrical {
                    cell_radius: lit(10.0),
                    cell_height: lit(10.0),
                    arc_count: 8,
                },
                phase: LatticePhase::sheet(lit(1.0)),
                frame: fieldform_scene::Frame::At {
                    origin: [lit(0.0); 3],
                    axis: axis.map(ParamRef::Literal),
                },
            });
            GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles")
        };

        // Along its own axis a cylindrical lattice is solid at the centre, because the cells
        // have collapsed together there. Perpendicular to it, it is not.
        let up = build([0.0, 0.0, 1.0]);
        let along_x = build([1.0, 0.0, 0.0]);

        let on_z = Point3::new(0.0, 0.0, 20.0);
        let on_x = Point3::new(20.0, 0.0, 0.0);

        assert!(
            up.distance(on_z) < 0.0,
            "a Z-up lattice should be solid along Z, got {}",
            up.distance(on_z)
        );
        assert!(
            along_x.distance(on_x) < 0.0,
            "an X-aimed lattice should be solid along X, got {}",
            along_x.distance(on_x)
        );
        // And the two disagree away from the shared origin, which is what says the aiming did
        // something rather than being quietly ignored.
        assert!(
            (up.distance(on_x) - along_x.distance(on_x)).abs() > 1e-6,
            "aiming changed nothing at {on_x:?}"
        );
    }

    /// The shortest turn from +Z, including the two cases a cross product cannot give.
    #[test]
    fn a_rotation_from_z_carries_z_onto_the_axis_asked_for() {
        for axis in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 2.0, -3.0),
            // Exactly opposite: a whole circle of shortest turns, and the cross product is zero.
            Vector3::new(0.0, 0.0, -1.0),
        ] {
            let turned = rotation_from_z(axis) * Vector3::z();
            let want = axis.normalize();
            assert!(
                (turned - want).norm() < 1e-9,
                "{axis:?} became {turned:?} rather than {want:?}"
            );
        }
        // A zero axis cannot point anywhere, so it means the default rather than a NaN.
        assert_eq!(rotation_from_z(Vector3::zeros()), Rotation3::identity());
    }

    #[test]
    fn a_blend_against_a_rectangular_beam_lattice_is_not_marked_approximate() {
        let lit = |v: f64| ParamRef::Literal(v);
        let beam_blend = |map: fieldform_scene::CellMap| {
            let mut g = Graph::empty();
            let sphere = g.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(10.0),
            });
            let lattice = g.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell: fieldform_scene::LatticeCell::Bcc,
                map,
                phase: LatticePhase::Strut { radius: lit(0.4) },
            });
            g.add(Node::Blend {
                op: fieldform_scene::BooleanOp::Union,
                a: sphere.into(),
                b: lattice.into(),
                fillet: Fillet { radius: lit(1.0), continuity: fieldform_scene::Continuity::G1 },
            });
            g
        };
        let doc = Document::new();

        // **The first lattice a blend can be trusted with.** Every surface cell reads short, so
        // the seam sits toward it and the panel marks the radius approximate; a beam cell on an
        // even rectangular map is a union of capsules and reads true, so there is nothing to
        // mark. Reported rather than assumed: this is the number the user typed coming out.
        let flat = beam_blend(fieldform_scene::CellMap::Rectangular { cell: lit(2.0) });
        assert!(
            approximate_blends(&doc, &flat).is_empty(),
            "a rectangular beam blend was marked approximate",
        );

        // And the same cell under a curved map is not, because the conversion out of cell space
        // divides by the largest of three frequencies that no longer agree.
        let curved = beam_blend(fieldform_scene::CellMap::Cylindrical {
            cell_radius: lit(2.0),
            cell_height: lit(2.0),
            arc_count: 12,
        });
        assert_eq!(
            approximate_blends(&doc, &curved).len(),
            1,
            "a cylindrical beam blend was not marked approximate",
        );
    }

    /// A cell paired with a banding it cannot give is refused, not drawn.
    ///
    /// The panel offers a cell only the bandings it supports, so this pairing cannot be clicked
    /// into being — but a document can carry it, by hand or from a build whose vocabulary
    /// differed, and every arm downstream would then read a number that means something else. A
    /// wall thickness on a beam is not a smaller beam; it is a number with no meaning on that
    /// path, and drawing *something* would be the silent failure.
    #[test]
    fn a_beam_cell_asked_for_a_wall_is_refused_rather_than_drawn() {
        let lit = |v: f64| ParamRef::Literal(v);
        let mut g = Graph::empty();
        g.add(Node::Lattice {
            frame: fieldform_scene::Frame::default(),
            cell: fieldform_scene::LatticeCell::Cubic,
            map: fieldform_scene::CellMap::Rectangular { cell: lit(2.0) },
            phase: LatticePhase::sheet(lit(0.2)),
        });
        let doc = Document::new();
        let Err(err) = GraphField::compile_graph(&doc, &g, 1e-4) else {
            panic!("a cubic cell has no wall to give, and this compiled");
        };
        assert!(
            matches!(err, CompileError::CellCannotTakeThatPhase(..)),
            "refused as {err:?} rather than for the reason it was refused for",
        );

        // The mirror of it: a surface cell asked for a strut radius.
        let mut g = Graph::empty();
        g.add(Node::Lattice {
            frame: fieldform_scene::Frame::default(),
            cell: fieldform_scene::LatticeCell::Gyroid,
            map: fieldform_scene::CellMap::Rectangular { cell: lit(2.0) },
            phase: LatticePhase::Strut { radius: lit(0.2) },
        });
        let Err(err) = GraphField::compile_graph(&doc, &g, 1e-4) else {
            panic!("a gyroid has no struts to give a radius to, and this compiled");
        };
        assert!(matches!(err, CompileError::CellCannotTakeThatPhase(..)), "{err:?}");
    }

    /// The graph walk and the compiler must agree about what is a true distance.
    ///
    /// Two implementations of D8's precondition: `carries_true_distance` reads the scene graph
    /// so a panel can mark a fillet approximate without compiling, and `normalization_of` reads
    /// compiled ops so a compiled field can report what it carries. A value computed in two
    /// places is the failure this codebase keeps meeting -- the gradient bound drifted between
    /// Rust and WGSL exactly this way -- so the two are checked against each other over every
    /// shape of graph rather than trusted to stay in step.
    ///
    /// The compiler is the authority: the walk is held to the compiled field's `normalization`,
    /// and then to the consequence a user meets, whether a blend over the operand is reported
    /// approximate. The blend compiles either way.
    #[test]
    fn the_graph_walk_agrees_with_the_compiler_about_true_distance() {
        let doc = Document::new();
        let lit = |v: f64| ParamRef::Literal(v);

        // Every operand shape that reaches a different arm of the walk.
        let builders: Vec<(&str, fn(&mut Graph) -> NodeId)> = vec![
            ("sphere", |g| {
                g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(5.0),
                })
            }),
            ("lattice", |g| {
                g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::Gyroid,
                    map: fieldform_scene::CellMap::Rectangular {
                        cell: ParamRef::Literal(2.0),
                    },
                    phase: LatticePhase::sheet(ParamRef::Literal(0.2)),
                })
            }),
            ("rectangular beam lattice", |g| {
                g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::Bcc,
                    map: fieldform_scene::CellMap::Rectangular {
                        cell: ParamRef::Literal(2.0),
                    },
                    phase: LatticePhase::Strut { radius: ParamRef::Literal(0.2) },
                })
            }),
            ("rectangular honeycomb", |g| {
                g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::Honeycomb,
                    map: fieldform_scene::CellMap::Rectangular {
                        cell: ParamRef::Literal(2.0),
                    },
                    phase: LatticePhase::sheet(ParamRef::Literal(0.2)),
                })
            }),
            ("cylindrical beam lattice", |g| {
                g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::Cubic,
                    map: fieldform_scene::CellMap::Cylindrical {
                        cell_radius: ParamRef::Literal(2.0),
                        cell_height: ParamRef::Literal(2.0),
                        arc_count: 12,
                    },
                    phase: LatticePhase::Strut { radius: ParamRef::Literal(0.2) },
                })
            }),
            // A box's or a cylinder's edge: exact rounded at G1, a bound at G2 or chamfered.
            ("box rounded at G1", |g| {
                g.add(Node::Cuboid {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    size: [ParamRef::Literal(10.0); 3],
                    turn: fieldform_scene::no_turn(),
                    edge: fieldform_scene::Fillet { radius: ParamRef::Literal(1.0), continuity: fieldform_scene::Continuity::G1 },
                })
            }),
            ("box at G2", |g| {
                g.add(Node::Cuboid {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    size: [ParamRef::Literal(10.0); 3],
                    turn: fieldform_scene::no_turn(),
                    edge: fieldform_scene::Fillet { radius: ParamRef::Literal(1.0), continuity: fieldform_scene::Continuity::G2 },
                })
            }),
            ("cylinder chamfered", |g| {
                g.add(Node::Cylinder {
                    ends: fieldform_scene::Ends::between(
                        fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                        fieldform_scene::Place::At([ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(10.0)]),
                    ),
                    radius: ParamRef::Literal(4.0),
                    edge: fieldform_scene::Fillet { radius: ParamRef::Literal(1.0), continuity: fieldform_scene::Continuity::Chamfer },
                })
            }),
            ("cylinder chamfered back nought", |g| {
                g.add(Node::Cylinder {
                    ends: fieldform_scene::Ends::between(
                        fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                        fieldform_scene::Place::At([ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(10.0)]),
                    ),
                    radius: ParamRef::Literal(4.0),
                    edge: fieldform_scene::Fillet { radius: ParamRef::Literal(0.0), continuity: fieldform_scene::Continuity::Chamfer },
                })
            }),
            ("boolean of two spheres", |g| {
                let a = g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(5.0),
                });
                let b = g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(1.0); 3]),
                    radius: ParamRef::Literal(5.0),
                });
                g.add(Node::Boolean {
                    op: BooleanOp::Union,
                    inputs: vec![a.into(), b.into()],
                })
            }),
            ("boolean with a lattice", |g| {
                let a = g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(5.0),
                });
                let b = g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::Gyroid,
                    map: fieldform_scene::CellMap::Rectangular {
                        cell: ParamRef::Literal(2.0),
                    },
                    phase: LatticePhase::sheet(ParamRef::Literal(0.2)),
                });
                g.add(Node::Boolean {
                    op: BooleanOp::Intersection,
                    inputs: vec![a.into(), b.into()],
                })
            }),
            ("shelled sphere", |g| {
                let a = g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(5.0),
                });
                g.add(Node::Shell {
                    input: a.into(),
                    thickness: ParamRef::Literal(0.5),
                })
            }),
            ("shelled lattice boolean", |g| {
                let a = g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(5.0),
                });
                let b = g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::Gyroid,
                    map: fieldform_scene::CellMap::Rectangular {
                        cell: ParamRef::Literal(2.0),
                    },
                    phase: LatticePhase::sheet(ParamRef::Literal(0.2)),
                });
                let i = g.add(Node::Boolean {
                    op: BooleanOp::Intersection,
                    inputs: vec![a.into(), b.into()],
                });
                g.add(Node::Shell {
                    input: i.into(),
                    thickness: ParamRef::Literal(0.5),
                })
            }),
            ("offset sphere", |g| {
                let a = g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(5.0),
                });
                g.add(Node::Offset {
                    input: a.into(),
                    distance: ParamRef::Literal(1.0),
                })
            }),
            // **The extrudes, which this test had none of** -- and so never noticed that the
            // walk claimed exactness for a leaning or drafted one while the compiler knew
            // better. A curve cannot be built by these closures, since they take only a graph,
            // so the extrude cases are checked separately below against a document that has
            // one; these three keep the shape of the list honest about what it does cover.
            ("transformed sphere", |g| {
                let a = g.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(5.0),
                });
                g.add(Node::Transform {
                    input: a.into(),
                    rotation: [ParamRef::Literal(0.0); 3],
                    translation: [ParamRef::Literal(1.0); 3],
                    scale: ParamRef::Literal(1.0),
                    about: fieldform_scene::Place::origin(),
                })
            }),
        ];

        for (name, make) in builders {
            let mut graph = Graph::empty();
            let operand = make(&mut graph);
            let walk_says = carries_true_distance(&doc, &graph, operand);

            // What the compiled ops say about the same node, asked through the compiled
            // field's own `normalization` — which is `normalization_of` over the ops, the
            // other implementation and the one this must not drift from.
            let compiled = GraphField::compile_graph(&doc, &graph, 1e-3)
                .expect("the operand alone always compiles");
            assert_eq!(
                walk_says,
                compiled.normalization() == Normalization::Distance,
                "{name}: the graph walk and the compiled ops disagree"
            );

            // A blend of the operand with a plain sphere, which always carries a true
            // distance -- so the pair is approximate exactly when the operand is not.
            let other = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(9.0); 3]),
                radius: lit(1.0),
            });
            graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: operand.into(),
                b: other.into(),
                fillet: Fillet { radius: lit(0.5), continuity: fieldform_scene::Continuity::G1 },
            });
            // And the blend over it is reported as approximate exactly when the operand is
            // not a true distance. That is the consequence the answer now has, so it is what
            // is worth asserting: the blend itself always compiles.
            assert!(
                GraphField::compile_graph(&doc, &graph, 1e-3).is_ok(),
                "{name}: a blend is no longer refused"
            );
            let flagged = !approximate_blends(&doc, &graph).is_empty();
            assert_eq!(
                walk_says, !flagged,
                "{name}: true-distance={walk_says} but the blend was flagged={flagged}"
            );
        }

        // **And the extrudes**, which need a curve in the document and so cannot be built by
        // the closures above. Their absence is why the walk claimed a leaning extrude carried
        // true distance while the compiler returned `LowerBound` for it -- so a fillet against
        // one came out about twice the radius typed with nothing saying so.
        let mut doc = Document::new();
        let curve = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square".to_string(),
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let profile = curve.id;
        doc.curves.insert(profile, curve);

        let lean = 32f64.to_radians();
        let rounded = Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::G1 };
        for (name, sweep, draft, rims) in [
            ("extruded straight", fieldform_scene::Sweep::Normal, 0.0, Fillet::hard()),
            ("extruded with a draft", fieldform_scene::Sweep::Normal, 15.0, Fillet::hard()),
            (
                "extruded at a lean",
                fieldform_scene::Sweep::Toward {
                    xyz: [lit(lean.sin()), lit(0.0), lit(lean.cos())],
                },
                0.0,
                Fillet::hard(),
            ),
            (
                "extruded at a lean with a draft",
                fieldform_scene::Sweep::Toward {
                    xyz: [lit(lean.sin()), lit(0.0), lit(lean.cos())],
                },
                15.0,
                Fillet::hard(),
            ),
            // Straight and upright, which is exact until its rims are rounded.
            ("extruded straight with rounded rims", fieldform_scene::Sweep::Normal, 0.0, rounded),
            (
                "extruded straight with a rim of nought that names a shape",
                fieldform_scene::Sweep::Normal,
                0.0,
                Fillet { radius: lit(0.0), continuity: fieldform_scene::Continuity::Chamfer },
            ),
        ] {
            let mut graph = Graph::empty();
            let operand = graph.add(Node::Extrude { rims: rims.into(),
                outline: fieldform_scene::Outline::closed(profile),
                sweep,
                height: lit(20.0),
                draft: lit(draft),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            let walk_says = carries_true_distance(&doc, &graph, operand);

            // The compiler's own answer, which is the thing the walk is standing in for.
            let compiled = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
            assert_eq!(
                walk_says,
                compiled.normalization() == Normalization::Distance,
                "{name}: the walk and the compiler disagree about true distance"
            );

            let other = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(9.0); 3]),
                radius: lit(1.0),
            });
            graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: operand.into(),
                b: other.into(),
                fillet: Fillet { radius: lit(0.5), continuity: fieldform_scene::Continuity::G1 },
            });
            let flagged = !approximate_blends(&doc, &graph).is_empty();
            assert_eq!(
                walk_says, !flagged,
                "{name}: true-distance={walk_says} but the blend was flagged={flagged}"
            );
        }

        // **And a sweep**, which answers `false` on every spine: a lower bound always, since
        // the section planes converge on the inside of a bend and the value is divided by that.
        // Held here so the walk cannot drift toward "exact on a straight spine", which the
        // compiler would contradict -- a straight spine is what an extrude is for.
        let path = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Bend",
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 40.0),
                Point3::new(0.0, 40.0, 80.0),
            ],
            false,
        )
        .expect("open");
        let spine = path.id;
        doc.curves.insert(spine, path);
        let mut graph = Graph::empty();
        let operand = graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
        let walk_says = carries_true_distance(&doc, &graph, operand);
        let compiled = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        assert_eq!(
            walk_says,
            compiled.normalization() == Normalization::Distance,
            "sweep: the walk and the compiler disagree about true distance"
        );
        assert!(!walk_says, "a sweep never claims to be a true distance");

        // **And a holed scene, which is the one shape the compiler has no answer to agree with.**
        // A block with an empty slot is refused rather than compiled, so there is no
        // `normalization` to hold the walk to -- which makes the property here the other half of
        // the agreement: the walk must take the *weaker* answer where it cannot be checked. The
        // `ScalarField` contract leaves it no choice, since there is no value a missing input
        // could supply that does not overestimate a distance.
        //
        // Added because this test's known gap is recorded above: eight scenes and no extrude, and
        // the walk was wrong about extrudes for exactly as long as the list did not name one.
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(5.0),
        });
        // A shell with nothing in it, and a boolean reading that shell: the hole is one block
        // below the boolean, which is the case `Graph::incomplete` closes downstream over.
        let holed = graph.add(Node::Shell {
            input: fieldform_scene::Operand::empty(),
            thickness: lit(0.5),
        });
        let above = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![sphere.into(), holed.into()],
        });
        for (name, id) in [("the holed shell", holed), ("the boolean above it", above)] {
            assert!(
                !carries_true_distance(&doc, &graph, id),
                "{name}: an empty slot takes the weaker answer, whatever is wired beside it"
            );
        }
        assert!(
            carries_true_distance(&doc, &graph, sphere),
            "and a wired sibling is untouched by its neighbour's hole"
        );
    }

    /// A blend with an empty operand is never called exact, which is this walk's whole part in
    /// the `ScalarField` contract.
    ///
    /// `distance` must never *over*estimate, and there is no value a missing input can supply that
    /// satisfies that -- so an empty slot produces a refusal or an exclusion, never a number.
    ///
    /// **The trap is one character wide.** `true_distance_at`'s default for a *missing node*
    /// answers `true`, and it must: it stands for a leaf the walk cannot see, and nothing is drawn
    /// from it. An empty slot answering the same would leave a holed blend out of
    /// `approximate_blends` **as though its radius were honest**, and the panel would quote a
    /// fillet radius the field is in no position to produce -- which is the silent wrong shape the
    /// surviving half of D8 exists to prevent.
    ///
    /// The two answers are about different things, which is why both are asserted here: `false`
    /// keeps anything downstream from inheriting a claim, while the list is what puts
    /// `= approximate` beside a radius -- and a radius the block cannot apply at all is not a
    /// radius to annotate. What the panel says about this blend is that the slot is empty
    /// (`notebook-spec.md` 7.2).
    #[test]
    fn a_blend_with_an_empty_operand_is_never_called_exact() {
        let doc = Document::new();
        let lit = |v: f64| ParamRef::Literal(v);
        let ball = |g: &mut Graph, x: f64| {
            g.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(0.0)]),
                radius: lit(5.0),
            })
        };

        // Each slot in turn and both at once, because the answer cannot depend on which half is
        // wired: a true distance in one slot must not carry the blend.
        for (name, wire_a, wire_b) in [
            ("nothing in a", false, true),
            ("nothing in b", true, false),
            ("nothing in either", false, false),
        ] {
            let mut graph = Graph::empty();
            let slot = |g: &mut Graph, wired: bool, x: f64| match wired {
                true => fieldform_scene::Operand::filled(ball(g, x)),
                false => fieldform_scene::Operand::empty(),
            };
            let a = slot(&mut graph, wire_a, 0.0);
            let b = slot(&mut graph, wire_b, 4.0);
            let blend = graph.add(Node::Blend {
                op: BooleanOp::Union,
                a,
                b,
                fillet: Fillet { radius: lit(1.0), continuity: fieldform_scene::Continuity::G1 },
            });
            assert!(
                !carries_true_distance(&doc, &graph, blend),
                "{name}: a blend that cannot be evaluated does not carry a true distance"
            );
            assert!(
                !approximate_blends(&doc, &graph).contains(&blend),
                "{name}: the panel's message for a holed blend is the empty slot, not a radius"
            );
        }

        // **And a hard blend is no exception.** A radius of nought is a plain boolean, which
        // `min`/`max` make exact over either normalization -- so that arm reads its operands
        // rather than answering `false` outright, and an empty one has to carry through it.
        let mut graph = Graph::empty();
        let wired = ball(&mut graph, 0.0);
        let hard = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: wired.into(),
            b: fieldform_scene::Operand::empty(),
            fillet: Fillet { radius: lit(0.0), continuity: fieldform_scene::Continuity::G1 },
        });
        assert!(carries_true_distance(&doc, &graph, wired), "the wired operand is a distance");
        assert!(
            !carries_true_distance(&doc, &graph, hard),
            "and a hard boolean of a sphere and a hole is not a distance to anything"
        );
    }

    /// A block with an empty slot bounds nothing and moves at no rate -- the other two walks that
    /// run every frame before any compile, each answering for the hole the way its own
    /// "node is not there" default already did.
    ///
    /// `Extent::Empty` rather than a box, because a box around nothing would hand the view's
    /// framing, a bake's bounds and the per-tile cull material that is not there; nought rather
    /// than one, because a rate is a statement about a value and a block with a hole has none.
    /// Both are closed upward by the same reasoning `Graph::incomplete` is closed by: a modifier
    /// reading a holed block has no value either.
    #[test]
    fn a_block_with_an_empty_slot_bounds_nothing_and_moves_at_no_rate() {
        let doc = Document::new();
        let lit = |v: f64| ParamRef::Literal(v);
        let mut graph = Graph::empty();
        let holed = graph.add(Node::Shell {
            input: fieldform_scene::Operand::empty(),
            thickness: lit(0.5),
        });
        let above = graph.add(Node::Twist {
            input: holed.into(),
            degrees_per_mm: lit(10.0),
            frame: fieldform_scene::Frame::default(),
        });
        for (name, id) in [("the holed shell", holed), ("the twist above it", above)] {
            assert_eq!(
                crate::graph_extent(&doc, &graph, id),
                Extent::Empty,
                "{name}: nothing to read is nothing to bound"
            );
            assert_eq!(
                crate::graph_lipschitz(&doc, &graph, id),
                0.0,
                "{name}: nothing to read is a rate of nothing"
            );
        }
    }

    /// A honeycomb is measured in millimetres across a flat period, so only a rectangular map
    /// can carry it; the rest are refused by name rather than drawn stretched. And the map it
    /// does get takes the cell's own shape, so the hexagons are regular: both sublattices of
    /// centres are the same distance from their walls.
    #[test]
    fn a_honeycomb_takes_its_period_from_the_cell_and_refuses_a_map_with_none_to_give() {
        let doc = Document::new();
        let lattice = |map: fieldform_scene::CellMap| {
            let mut graph = Graph::empty();
            graph.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell: fieldform_scene::LatticeCell::Honeycomb,
                map,
                phase: LatticePhase::sheet(ParamRef::Literal(1.0)),
            });
            graph
        };
        let across = 10.0;
        let field = GraphField::compile_graph(
            &doc,
            &lattice(fieldform_scene::CellMap::Rectangular { cell: ParamRef::Literal(across) }),
            1e-3,
        )
        .expect("a rectangular honeycomb compiles");
        assert_eq!(field.normalization(), Normalization::Distance);
        for centre in [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3f64.sqrt() * across * 0.5, across * 0.5, 0.0),
        ] {
            assert!((field.distance(centre) - (across - 1.0) * 0.5).abs() < 1e-9);
        }

        let around = fieldform_scene::CellMap::Cylindrical {
            cell_radius: ParamRef::Literal(10.0),
            cell_height: ParamRef::Literal(10.0),
            arc_count: 12,
        };
        match GraphField::compile_graph(&doc, &lattice(around), 1e-3).map(|_| ()) {
            Err(CompileError::CellCannotTakeThatMap(cell, map)) => {
                assert_eq!((cell, map), ("Honeycomb", "cylindrical"))
            }
            other => panic!("expected the map to be refused by name, got {other:?}"),
        }
    }


    /// **The cell's own numbers have to come from the document**, and nothing else in the
    /// compile would notice if they did not: `CellKind::from` hands over a jitter of nought,
    /// which is a regular grid -- a perfectly plausible foam, of entirely the wrong shape. So
    /// this asks the compiled field whether it moved when the document said to.
    ///
    /// Two axes, because a seed and a jitter fail independently: a jitter that never arrives
    /// gives boxes, and a seed that never arrives gives the same arrangement for every re-roll.
    #[test]
    fn a_voronoi_cells_jitter_and_seed_reach_the_compiled_field() {
        let doc = Document::new();
        let built = |jitter: f64, seed: u64| {
            let mut graph = Graph::empty();
            graph.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell: fieldform_scene::LatticeCell::VoronoiFoam {
                    jitter: ParamRef::Literal(jitter),
                    seed,
                },
                map: fieldform_scene::CellMap::Rectangular { cell: ParamRef::Literal(10.0) },
                phase: LatticePhase::sheet(ParamRef::Literal(1.0)),
            });
            let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
            let probes: Vec<Point3<f64>> = (0..60)
                .map(|i| {
                    Point3::new(
                        -13.0 + 0.917 * i as f64,
                        4.3 - 0.731 * i as f64,
                        1.7 + 0.523 * i as f64,
                    )
                })
                .collect();
            probes.iter().map(|p| field.distance(*p)).collect::<Vec<_>>()
        };
        let plain = built(0.0, 7);
        let jittered = built(1.0, 7);
        let rerolled = built(1.0, 8);
        let moved = |a: &[f64], b: &[f64]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
        };
        assert!(
            moved(&plain, &jittered) > 1.0,
            "the document's jitter never reached the field: the grid moved {} mm",
            moved(&plain, &jittered),
        );
        assert!(
            moved(&jittered, &rerolled) > 1.0,
            "the document's seed never reached the field: a re-roll moved {} mm",
            moved(&jittered, &rerolled),
        );
    }


    /// **A sweep starts at its profile, wherever the path was drawn.** Reported by the owner:
    /// *"the profile is swept from the actual profile/plane — it should not matter where the
    /// sweeping curve is. right now the profile/sweep geometry seems to jump to the center of
    /// the sweeping curve."* The section is flattened about its origin and then ridden along
    /// the spine, so a flattening about a point the path does not pass through re-centres the
    /// profile on the curve: a square drawn at x = 100 came out as a bar at x = 0.
    ///
    /// Checked as the shape a person would look for — solid where the profile is, empty where
    /// the curve is — and then checked again with the path drawn somewhere else entirely, which
    /// for a *straight* path must give the same field: the section is carried by the point
    /// beneath the path's start and never turns, so where that point is makes no difference.
    /// A bent path is another matter, and
    /// `a_sweep_is_carried_by_the_point_its_path_sets_off_from` says what it does.
    #[test]
    fn a_sweep_is_carried_from_its_profile_and_not_from_its_path() {
        // A 20 mm square about (100, 0, 0), and a path that does not touch it: straight up.
        let built = |path_at: Point3<f64>| {
            let mut doc = Document::new();
            let square = fieldform_scene::Curve::polyline(
                fieldform_scene::CurveId::new(),
                "Square",
                vec![
                    Point3::new(90.0, -10.0, 0.0),
                    Point3::new(110.0, -10.0, 0.0),
                    Point3::new(110.0, 10.0, 0.0),
                    Point3::new(90.0, 10.0, 0.0),
                ],
                true,
            )
            .expect("closed");
            let path = fieldform_scene::Curve::polyline(
                fieldform_scene::CurveId::new(),
                "Up",
                vec![path_at, path_at + Vector3::new(0.0, 0.0, 50.0)],
                false,
            )
            .expect("open");
            let (profile, spine) = (square.id, path.id);
            doc.curves.insert(profile, square);
            doc.curves.insert(spine, path);
            let mut graph = Graph::empty();
            let node = graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
            let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
            let extent = graph_extent(&doc, &graph, node).as_box().expect("a sweep is bounded");
            (field, extent)
        };

        let (field, (lo, hi)) = built(Point3::origin());
        // Solid through the profile, all the way along what the path describes.
        for z in [1.0, 25.0, 49.0] {
            let inside = field.distance(Point3::new(100.0, 0.0, z));
            assert!(inside < 0.0, "the sweep should be solid at its profile, read {inside} at z = {z}");
        }
        // And nothing where the path itself is, which is where it used to appear.
        let at_the_path = field.distance(Point3::new(0.0, 0.0, 25.0));
        assert!(
            at_the_path > 0.0,
            "the sweep jumped to the path: {at_the_path} at the path's own middle"
        );
        // The box holds it. **Not tight**: the extent is the path grown by the section's reach
        // from the point that carries it, which is here a hundred millimetres from the square
        // -- the arm the section is carried on, and the radius it could swing through if the
        // path bent. A box fitted to the frames the section actually takes would be tighter,
        // and is a follow-up; this one is a safe superset.
        assert!(lo.x <= 90.0 && hi.x >= 110.0, "the box must hold the profile: {lo:?}..{hi:?}");

        // **Where the curve was drawn does not matter.** The same profile and the same path
        // shape, drawn a long way off, is the same solid.
        let (moved, _) = built(Point3::new(-250.0, 175.0, -30.0));
        for p in [
            Point3::new(100.0, 0.0, 25.0),
            Point3::new(100.0, 9.0, 5.0),
            Point3::new(0.0, 0.0, 25.0),
            Point3::new(100.0, 0.0, 75.0),
        ] {
            let (a, b) = (field.distance(p), moved.distance(p));
            assert!(
                (a - b).abs() < 1e-9,
                "moving the path changed the solid at {p:?}: {a} against {b}"
            );
        }
    }

    /// **The point the path sets off from is the point that rides it** -- ruled by the owner
    /// 2026-10-02, on a 500 mm outline whose path was drawn from one corner round a sixty-degree
    /// bend: carried by its centroid, the corner left the spine within millimetres and the
    /// outline ended 400 mm clear of the curve's end (`fieldform-io/examples/sweep_follows_spine.rs`).
    ///
    /// A square given a wall, with a quarter-circle path drawn from one of its corners: the
    /// corner is in the wall's band, so the field read *on the spine* is inside from end to end.
    /// Carried by the centroid it would be inside only until the frame had turned far enough to
    /// swing the corner off, which for a bend of this radius is about a third of the way.
    #[test]
    fn a_sweep_is_carried_by_the_point_its_path_sets_off_from() {
        let mut doc = Document::new();
        // A 40 mm square in the YZ plane with a corner at the origin, so its normal is +X.
        let square = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square",
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 40.0, 0.0),
                Point3::new(0.0, 40.0, 40.0),
                Point3::new(0.0, 0.0, 40.0),
            ],
            true,
        )
        .expect("closed");
        // A quarter circle of radius 80 from that corner, setting off along +X and turning
        // toward -Z: the square is on the outside of the bend and reaches 57 mm from the corner.
        let arc: Vec<Point3<f64>> = (0..=40)
            .map(|i| {
                let a = std::f64::consts::FRAC_PI_2 * i as f64 / 40.0;
                Point3::new(80.0 * a.sin(), 0.0, 80.0 * (a.cos() - 1.0))
            })
            .collect();
        let path = fieldform_scene::Curve::polyline(fieldform_scene::CurveId::new(), "Arc", arc.clone(), false)
            .expect("open");
        let (outline, spine) = (square.id, path.id);
        doc.curves.insert(outline, square);
        doc.curves.insert(spine, path);
        let wall = fieldform_scene::Outline::Wall {
            curve: outline,
            thickness: ParamRef::Literal(4.0),
            side: fieldform_scene::WallSide::Centre,
            ends: fieldform_scene::WallEnds::Square,
        };
        let mut graph = Graph::empty();
        graph.add(Node::Sweep { rims: Default::default(), outline: wall.clone(), spine, section: Default::default(), law: Default::default() });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        // The path starts in the section's plane, so nothing carries the solid off it.
        let offset = sweep_offset(&doc, &wall, spine).expect("a section");
        assert!(offset.norm() < 1e-9, "a path drawn in the plane is not carried off it: {offset:?}");
        // Inside all along the spine -- the band is 2 mm either side of the corner. The two
        // end samples lie on the caps themselves and read nought, so they are left out.
        for (i, p) in arc.iter().enumerate().skip(1).take(arc.len() - 2) {
            let d = field.distance(*p);
            assert!(
                d < 0.0,
                "the corner left the spine at sample {i} of 40 ({p:?}): read {d}"
            );
        }
        // And the far corner of the square swings with the frame: half way round, the frame
        // has turned 45 degrees about +Y, so the section's +Z has been carried to
        // (sin 45, 0, cos 45) and its +Y is still +Y.
        let half = std::f64::consts::FRAC_PI_4;
        let station = Point3::new(80.0 * half.sin(), 0.0, 80.0 * (half.cos() - 1.0));
        let far = station + Vector3::new(0.0, 40.0, 0.0) + Vector3::new(half.sin(), 0.0, half.cos()) * 40.0;
        let there = field.distance(far);
        assert!(
            there < 0.0,
            "the far corner should have turned with the frame to {far:?}: read {there}"
        );
    }

    /// **The step budget and the shader's divisor know what the law costs.** `sweep_stretch`
    /// answers over the graph, before anything has compiled, for the tracer's budget
    /// (`step_multiplier`) and for the slot codegen writes; the leaf computes its own for the
    /// field. Under a law the graph's number was v1's, with nothing of the taper or the twist
    /// in it -- a shader dividing by one number while the kernel divided by another, which is
    /// a surface in two places.
    ///
    /// Checked against the field itself rather than against a second copy of the arithmetic:
    /// the leaf's reading times the graph's divisor is the undivided section-and-slab value,
    /// which on a straight path with no law is the extrude's own distance. So the two agree
    /// exactly when multiplying by the graph's number undoes the division the leaf made --
    /// asserted here by reading a point whose undivided value is known.
    #[test]
    fn the_graphs_sweep_divisor_is_the_one_the_leaf_divided_by() {
        use fieldform_scene::{RampShape, SweepLaw, Taper};
        let lit = ParamRef::Literal;
        // A straight path, so the bend contributes nothing and what is left is the law's own
        // cost: the undivided reading at a point beside a 20 mm square is known exactly.
        let mut doc = Document::new();
        let square = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square",
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let path = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Up",
            vec![Point3::origin(), Point3::new(0.0, 0.0, 100.0)],
            false,
        )
        .expect("open");
        let (profile, spine) = (square.id, path.id);
        doc.curves.insert(profile, square);
        doc.curves.insert(spine, path);
        let outline = fieldform_scene::Outline::closed(profile);

        let laws = [
            ("plain", SweepLaw::default(), 1.0),
            // A twist of 90 degrees over 100 mm, G0: the section's reach is 10√2, so
            // K = (π/2 / 100) · 10√2 and the divisor is √(1 + K²).
            (
                "twisted",
                SweepLaw { twist: lit(90.0), ..Default::default() },
                (1.0 + (std::f64::consts::FRAC_PI_2 / 100.0 * 10.0 * std::f64::consts::SQRT_2)
                    .powi(2))
                .sqrt(),
            ),
            // The same turn eased in and out is 1.5 times as steep in the middle (G1).
            (
                "twisted, G1",
                SweepLaw { twist: lit(90.0), shape: RampShape::Smooth, ..Default::default() },
                (1.0 + (1.5 * std::f64::consts::FRAC_PI_2 / 100.0 * 10.0
                    * std::f64::consts::SQRT_2)
                    .powi(2))
                .sqrt(),
            ),
            // A 5 degree draft leans every wall by its tangent, and nothing else moves.
            (
                "drafted",
                SweepLaw { taper: Taper::Draft(lit(5.0)), ..Default::default() },
                (1.0 + 5f64.to_radians().tan().powi(2)).sqrt(),
            ),
        ];
        for (what, law, expected) in laws {
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(),
                outline: outline.clone(),
                spine,
                section: Default::default(),
                law: law.clone(),
            });
            let over_the_graph = sweep_stretch(&doc, &outline, spine, &law);
            assert!(
                (over_the_graph - expected).abs() < 1e-9,
                "{what}: the graph says {over_the_graph}, the law's own arithmetic {expected}"
            );
            // The tracer buys its steps from the same number.
            assert!(
                (step_multiplier(&doc, &graph) - over_the_graph).abs() < 1e-9,
                "{what}: the step budget does not charge the law"
            );
            // And it is the number the leaf divided by: at the middle of the start face, a
            // point 30 mm out along +x reads 20 mm from the section's wall before the
            // division, whatever the law does to the section elsewhere -- the start is where
            // every law begins and none of them has moved anything yet.
            let field = GraphField::compile_graph(&doc, &graph, 1e-3)
                .unwrap_or_else(|e| panic!("{what}: does not compile: {e}"));
            let read = field.distance(Point3::new(30.0, 0.0, 0.0));
            assert!(
                (read * over_the_graph - 20.0).abs() < 1e-6,
                "{what}: {read} times {over_the_graph} is not the 20 mm the section is away"
            );
        }
    }

    /// **A square section under the plain law reads its walls truly round a bend, and every
    /// reader of the graph knows it.** The leaf reads the section as it is inside the spine's
    /// tube (`sweep-spec.md` §4); `sweep_tube` tells the shader the same tube; the shading is
    /// told the reading needs no scaling; and the step budget still buys the stretch, which the
    /// caps read from inside are divided by. A section held at an angle keeps dividing all of
    /// it, on every side.
    #[test]
    fn a_sweep_read_in_its_tube_is_read_alike_by_the_leaf_the_shader_and_the_shading() {
        let mut doc = Document::new();
        let square = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square",
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        // A quarter turn of radius 100, setting off up +Z from the square's middle and bending
        // toward +Y, so the square's x axis is the bend's binormal all the way.
        let turn = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Turn",
            (0..=90)
                .map(|i| {
                    let a = (i as f64).to_radians();
                    Point3::new(0.0, 100.0 * (1.0 - a.cos()), 100.0 * a.sin())
                })
                .collect(),
            false,
        )
        .expect("open");
        let (profile, spine) = (square.id, turn.id);
        doc.curves.insert(profile, square);
        doc.curves.insert(spine, turn);
        let outline = fieldform_scene::Outline::closed(profile);
        let law = fieldform_scene::SweepLaw::default();
        let sweep = |section| {
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(), outline: outline.clone(), spine, section, law: law.clone() });
            graph
        };
        let stretch = sweep_stretch(&doc, &outline, spine, &law);
        assert!(stretch > 1.1, "a 20 mm square round a bend of 100 costs {stretch}");

        let square_held = sweep(fieldform_scene::Section::Square);
        let (reach, tube) = sweep_tube(&doc, &outline, spine, &fieldform_scene::Section::Square, &law)
            .expect("a section reaching 14 mm has a tube round a bend of 100");
        assert!((reach - 10.0 * std::f64::consts::SQRT_2).abs() < 1e-9, "reach {reach}");
        assert!(tube > 99.0 && tube <= 100.0, "a quarter turn of 100 has a tube of {tube}");
        assert!(
            sweep_reads_true(&doc, &outline, spine, &fieldform_scene::Section::Square, &law),
            "a turn in one plane reads its walls as they are"
        );
        assert!((step_multiplier(&doc, &square_held) - stretch).abs() < 1e-9, "the budget buys the caps");
        assert_eq!(value_scale_with(&doc, &square_held, &[]), 1.0, "the shading scales nothing");
        // Halfway round, 5 mm out past the wall the bend leaves alone: the leaf reads 5, less
        // the thousandth of a millimetre a frame leaning half a degree off each segment can
        // account for at 15 from the spine, where dividing by the stretch read 5 / 1.16.
        let field = GraphField::compile_graph(&doc, &square_held, 1e-3).expect("compiles");
        let half = std::f64::consts::FRAC_PI_4;
        let beside = Point3::new(15.0, 100.0 * (1.0 - half.cos()), 100.0 * half.sin());
        let read = field.distance(beside);
        assert!(read <= 5.0 && read > 5.0 - 2e-3, "5 mm out from the wall reads {read}");

        // Held at an angle, everything is divided, as it was.
        let skewed = sweep(fieldform_scene::Section::Skewed);
        assert!(sweep_tube(&doc, &outline, spine, &fieldform_scene::Section::Skewed, &law).is_none());
        assert!(!sweep_reads_true(&doc, &outline, spine, &fieldform_scene::Section::Skewed, &law));
        // The shading's scale is the stretch over the lean, which here is next to none: the
        // profile lies square to where the path sets off, within the half a degree the flattened
        // bend's first leg turns.
        let lean = sweep_lean(&doc, &outline, spine, &fieldform_scene::Section::Skewed);
        assert!((lean - 1.0).abs() < 1e-3, "a profile square to the path's start leans {lean}");
        assert!((value_scale_with(&doc, &skewed, &[]) - stretch / lean).abs() < 1e-9);
        let field = GraphField::compile_graph(&doc, &skewed, 1e-3).expect("compiles");
        let read = field.distance(beside);
        assert!((read * stretch - 5.0).abs() < 1e-3, "held at an angle, 5 mm reads {read}");
    }

    /// **A section held at an angle reads its square faces short by the lean, and the shading's
    /// scale puts the lean back.** The owner's wing: a slab held flush at sixty degrees to a
    /// straight spine. The sweep shortens every reading by the cosine, 0.5, which is exact on the
    /// walls that lean and half the truth above the top face, where the occlusion took it for
    /// material close by. `sweep_lean` says 0.5, the field above the top face reads half, and
    /// the shading's scale times that reading is the distance itself.
    #[test]
    fn a_leaning_sections_shading_scale_puts_its_lean_back() {
        let mut doc = Document::new();
        let profile = CurveId::new();
        let slab = fieldform_scene::Curve::polyline(
            profile,
            "slab",
            vec![
                Point3::new(0.0, -20.0, -2.0),
                Point3::new(0.0, 20.0, -2.0),
                Point3::new(0.0, 20.0, 2.0),
                Point3::new(0.0, -20.0, 2.0),
            ],
            true,
        )
        .expect("closed");
        let spine = CurveId::new();
        let line = fieldform_scene::Curve::polyline(
            spine,
            "spine",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(-100.0, -173.205, 0.0)],
            false,
        )
        .expect("open");
        doc.curves.insert(profile, slab);
        doc.curves.insert(spine, line);
        let outline = fieldform_scene::Outline::closed(profile);
        let law = fieldform_scene::SweepLaw::default();
        let section = fieldform_scene::Section::Skewed;
        let lean = sweep_lean(&doc, &outline, spine, &section);
        assert!((lean - 0.5).abs() < 1e-3, "sixty degrees leans {lean}");
        assert!((sweep_lean(&doc, &outline, spine, &fieldform_scene::Section::Square) - 1.0).abs() < 1e-12);
        let mut graph = Graph::empty();
        graph.add(Node::Sweep { outline: outline.clone(), spine, section, law, rims: Default::default() });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        // Five above the top face, halfway along.
        let above = Point3::new(-50.0, -86.6, 7.0);
        let read = field.distance(above);
        let scale = value_scale_with(&doc, &graph, &[]);
        assert!((read * scale - 5.0).abs() < 0.05, "5 mm above the top face reads {read}, scaled by {scale}");
        assert!(read < 3.0, "the reading itself is the shortened one, {read}");
    }

    /// **A loft compiles from the graph, with its centerline named or derived, and every
    /// reader agrees what it is.** The leaf is tested on its own in `lofted.rs`; this is the
    /// wiring: the compile, the extent and the step budget all reach a loft through one helper
    /// (`loft_parts`), and a wall offered as a section is refused by name.
    #[test]
    fn a_loft_compiles_from_the_graph_and_its_readers_agree() {
        let mut doc = Document::new();
        let ring = |r: f64, z: f64| -> Vec<Point3<f64>> {
            (0..96)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / 96.0;
                    Point3::new(r * a.cos(), r * a.sin(), z)
                })
                .collect()
        };
        let mut add = |points: Vec<Point3<f64>>| {
            let curve =
                fieldform_scene::Curve::polyline(fieldform_scene::CurveId::new(), "ring", points, true)
                    .expect("closed");
            let id = curve.id;
            doc.curves.insert(id, curve);
            id
        };
        let (a, b) = (add(ring(20.0, 0.0)), add(ring(8.0, 60.0)));
        let sections =
            vec![fieldform_scene::Outline::closed(a), fieldform_scene::Outline::closed(b)];

        // No centerline: one is derived, straight up between two concentric rings, so the solid
        // is the cone of radii 20 and 8 over 60 mm.
        let mut graph = Graph::empty();
        let node = graph.add(Node::Loft { rims: Default::default(),
            sections: sections.clone(),
            spine: None,
            smooth: fieldform_scene::LoftSmooth::G0,
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("a loft compiles");
        assert!(field.distance(Point3::new(0.0, 0.0, 30.0)) < 0.0, "solid on its axis");
        assert!(field.distance(Point3::new(14.5, 0.0, 30.0)) > 0.0, "outside the cone's waist");
        assert!(field.distance(Point3::new(13.5, 0.0, 30.0)) < 0.0, "inside it");
        assert_eq!(field.normalization(), Normalization::LowerBound);
        assert!(!carries_true_distance(&doc, &graph, node), "an interpolation is not a distance");

        // The box holds it, and the step budget charges what the leaf divides by.
        let (lo, hi) = graph_extent(&doc, &graph, node).as_box().expect("a loft is bounded");
        assert!(lo.x <= -20.0 && hi.x >= 20.0 && lo.z <= 0.0 && hi.z >= 60.0, "{lo:?}..{hi:?}");
        let stretch = loft_stretch(&doc, &sections, None, fieldform_scene::LoftSmooth::G0);
        assert!(
            (step_multiplier(&doc, &graph) - stretch).abs() < 1e-9,
            "the step budget charges the loft's own divisor"
        );

        // A wall as a section is refused, naming which section.
        let mut graph = Graph::empty();
        graph.add(Node::Loft { rims: Default::default(),
            sections: vec![
                fieldform_scene::Outline::closed(a),
                fieldform_scene::Outline::Wall {
                    curve: b,
                    thickness: ParamRef::Literal(2.0),
                    side: fieldform_scene::WallSide::Centre,
                    ends: fieldform_scene::WallEnds::Square,
                },
            ],
            spine: None,
            smooth: fieldform_scene::LoftSmooth::G0,
        });
        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3),
            Err(CompileError::WallAsLoftSection(1))
        ));
    }

    /// **The extent holds a smooth loft's bulge.** A Hermite through a small section, two large
    /// ones and a small one -- the ordinary bottle -- overshoots the large pair by about a
    /// quarter of the radius step per span, and the listed sections' reach alone left that ring
    /// outside the box the per-tile cull reads, so tiles seeing only the belly drew background
    /// where the field has material. A G0 loft cannot bulge and keeps the sections' own box.
    /// Found in review; the belly is the review's probe (the rail reaches 34.55 mm at z = 60).
    #[test]
    fn the_extent_holds_a_smooth_lofts_bulge() {
        let mut doc = Document::new();
        let ring = |r: f64, z: f64| -> Vec<Point3<f64>> {
            (0..64)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / 64.0;
                    Point3::new(r * a.cos(), r * a.sin(), z)
                })
                .collect()
        };
        let mut add = |points: Vec<Point3<f64>>| {
            let curve =
                fieldform_scene::Curve::polyline(fieldform_scene::CurveId::new(), "ring", points, true)
                    .expect("closed");
            let id = curve.id;
            doc.curves.insert(id, curve);
            id
        };
        let sections: Vec<fieldform_scene::Outline> = [(10.0, 0.0), (30.0, 10.0), (30.0, 110.0), (10.0, 120.0)]
            .into_iter()
            .map(|(r, z)| fieldform_scene::Outline::closed(add(ring(r, z))))
            .collect();
        // 34.3, a little under the rail's 34.55: the resampled ring's edges sit inside its
        // vertices by their chord sag, and a probe on the number itself measured that.
        let belly = Point3::new(34.3, 0.0, 60.0);
        for (smooth, bulges) in
            [(fieldform_scene::LoftSmooth::G1, true), (fieldform_scene::LoftSmooth::G0, false)]
        {
            let mut graph = Graph::empty();
            let node = graph.add(Node::Loft { rims: Default::default(), sections: sections.clone(), spine: None, smooth });
            let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("a loft compiles");
            assert_eq!(field.distance(belly) < 0.0, bulges, "{smooth:?} at the belly");
            let (lo, hi) = graph_extent(&doc, &graph, node).as_box().expect("bounded");
            assert!(
                (hi.x >= belly.x && lo.x <= -belly.x && hi.y >= belly.y && lo.y <= -belly.y) == bulges
                    || !bulges && hi.x >= 30.0,
                "{smooth:?}: {lo:?}..{hi:?}"
            );
            if !bulges {
                assert!(hi.x < 30.5, "G0 keeps the sections' own box: {hi:?}");
            }
        }
    }

    /// A 20 mm square and a spine that turns once, by ninety degrees, `corner` millimetres
    /// along -- the two things every sweep test below is built from.
    fn sweep_document(corner: f64) -> (Document, fieldform_scene::CurveId, fieldform_scene::CurveId) {
        let mut doc = Document::new();
        let square = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Square",
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let path = fieldform_scene::Curve::polyline(
            fieldform_scene::CurveId::new(),
            "Elbow",
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, corner),
                Point3::new(0.0, corner, corner),
            ],
            false,
        )
        .expect("open");
        let (profile, spine) = (square.id, path.id);
        doc.curves.insert(profile, square);
        doc.curves.insert(spine, path);
        (doc, profile, spine)
    }

    /// **The refusal has a name, and it fires at the fold and not before.** The square reaches
    /// 14.1 mm from its centre. Turning ninety degrees over 20 mm legs is a bend of radius
    /// 12.7 mm, so the section is wider than the bend and the sweep would pass through itself;
    /// over 40 mm legs the radius is 25.5 mm and the same square goes round. Both are asserted,
    /// so the test is the threshold rather than that something, somewhere, was refused.
    #[test]
    fn a_sweep_wider_than_its_tightest_bend_is_refused_by_name() {
        let (doc, profile, spine) = sweep_document(20.0);
        let mut graph = Graph::empty();
        graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
        match GraphField::compile_graph(&doc, &graph, 1e-3).map(|_| ()) {
            Err(CompileError::BadSweep(named, crate::SweepError::FoldsThroughItself { .. })) => {
                assert_eq!(named, spine, "the refusal names the spine that bends too hard")
            }
            other => panic!("expected the fold to be refused by name, got {other:?}"),
        }

        let (doc, profile, spine) = sweep_document(40.0);
        let mut graph = Graph::empty();
        graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
        assert!(
            GraphField::compile_graph(&doc, &graph, 1e-3).is_ok(),
            "the same square goes round a bend it fits"
        );
    }

    /// The box is the spine grown by the profile's reach, which is where the solid can be and
    /// no further -- and the field agrees with it: a point inside the first straight run reads
    /// inside, and one off to the side by more than the reach reads outside.
    #[test]
    fn a_swept_body_is_bounded_by_its_spine_grown_by_its_profile() {
        let (doc, profile, spine) = sweep_document(40.0);
        let mut graph = Graph::empty();
        graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
        let Extent::Bounded { lo, hi } = part_extent(&doc, &graph) else {
            panic!("a sweep along a finite spine is bounded");
        };
        let reach = 200f64.sqrt();
        assert!(lo.x <= -reach + 1e-6 && hi.x >= reach - 1e-6, "across the section: {lo:?} {hi:?}");
        assert!(hi.z >= 40.0 + reach - 1e-6 && hi.y >= 40.0 + reach - 1e-6, "to the far end: {hi:?}");

        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        assert_eq!(field.normalization(), Normalization::LowerBound);
        assert!(field.distance(Point3::new(0.0, 0.0, 10.0)) < 0.0, "inside the first run");
        assert!(field.distance(Point3::new(30.0, 0.0, 10.0)) > 0.0, "beside it, past the reach");
        // The cost the tracer is told about is the leaf's own stretch, not a guess.
        let told = step_multiplier(&doc, &graph);
        assert!(told > 1.0 && told < 3.0, "a gentle bend costs a little: {told}");
    }

    /// A straight curve along Z, and a document holding it.
    fn attractor_document() -> (Document, fieldform_scene::CurveId) {
        let mut doc = Document::new();
        let id = fieldform_scene::CurveId::new();
        let curve = fieldform_scene::Curve::polyline(
            id,
            "attractor",
            vec![Point3::new(0.0, 0.0, -60.0), Point3::new(0.0, 0.0, 60.0)],
            false,
        )
        .expect("a two-point polyline is a curve");
        doc.curves.insert(id, curve);
        (doc, id)
    }

    /// **The one test section 4 of the spec asks for before anything else.** The sample point is
    /// pushed away from the curve, so the material has to come toward it -- and if the sign were
    /// inverted the result would still be a deformation, still look deliberate, and be the wrong
    /// shape. Nothing else here would catch that.
    #[test]
    fn a_sphere_leans_toward_the_curve_not_away() {
        let (doc, curve) = attractor_document();
        // Where the sphere's near and far faces sit along +X. Bisected between a point known
        // to be outside and one known to be inside, rather than marched: the field is a lower
        // bound, so a march stops short of the surface by whatever the bound understates.
        let faces = |strength: f64| {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
            graph.add(Node::Attract {
                input: ball.into(),
                toward: fieldform_scene::Attractor::Curve(curve),
                strength: lit(strength),
                falloff: lit(40.0),
            });
            let field = GraphField::compile(&doc, &part_with(graph), 0.05).expect("compiles");
            let at = |x: f64| field.distance(Point3::new(x, 0.0, 0.0));
            let inside = 30.0;
            assert!(at(inside) < 0.0, "the sphere's centre should be solid");
            let cross = |outside: f64| {
                let (mut lo, mut hi) = (outside, inside);
                for _ in 0..100 {
                    let mid = 0.5 * (lo + hi);
                    if at(mid) > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            };
            (cross(0.0), cross(100.0))
        };

        let (near_off, far_off) = faces(0.0);
        let (near_on, far_on) = faces(1.0);
        let centre_off = 0.5 * (near_off + far_off);
        let centre_on = 0.5 * (near_on + far_on);
        assert!(
            centre_on < centre_off - 0.5,
            "the sphere should move toward the curve at x = 0: centre went {centre_off} -> \
             {centre_on}"
        );
        // And it *stretches* rather than sliding: the near face, deeper in the falloff, moves
        // further than the far one. A translation would move both by the same amount, which is
        // what a warp applied with a constant weight would degenerate to.
        let near_moved = near_off - near_on;
        let far_moved = far_off - far_on;
        assert!(
            near_moved > far_moved + 0.1,
            "expected a stretch, not a slide: near moved {near_moved}, far moved {far_moved}"
        );
    }

    /// Not "close enough to the unwarped body" -- the same numbers. A body with the attractor
    /// switched off is what somebody compares the warped one against, and a comparison that has
    /// to allow a tolerance cannot tell a no-op from a very small warp.
    #[test]
    fn a_pull_of_zero_is_bit_for_bit_the_body_it_wraps() {
        let (doc, curve) = attractor_document();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        let plain = GraphField::compile_graph(&doc, &graph, 0.05).expect("compiles");
        graph.add(Node::Attract {
            input: ball.into(),
            toward: fieldform_scene::Attractor::Curve(curve),
            strength: lit(0.0),
            falloff: lit(40.0),
        });
        let warped = GraphField::compile_graph(&doc, &graph, 0.05).expect("compiles");
        for i in 0..40 {
            for j in 0..40 {
                let p = Point3::new(i as f64 * 2.0 - 20.0, j as f64 * 2.0 - 40.0, 3.5);
                assert_eq!(
                    plain.distance(p),
                    warped.distance(p),
                    "a pull of zero changed the value at {p:?}"
                );
            }
        }
    }

    /// Section 14's acceptance criterion, and the reason the box is grown by `0.26 * s * R`
    /// rather than by the whole falloff radius: a tighter box is only worth having if it holds.
    ///
    /// Marched rather than reasoned about. A body reaching outside the box it reports is one
    /// the mesher clips and the ray marcher starts inside of, and neither failure looks like a
    /// bounds problem when it appears.
    #[test]
    fn an_attracted_body_stays_inside_the_box_it_reports() {
        let (doc, curve) = attractor_document();
        for strength in [0.5, 1.0, -0.8] {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
            let node = graph.add(Node::Attract {
                input: ball.into(),
                toward: fieldform_scene::Attractor::Curve(curve),
                strength: lit(strength),
                falloff: lit(40.0),
            });
            let (lo, hi) = graph_extent(&doc, &graph, node)
                .as_box()
                .expect("an attracted sphere is bounded");
            let field = GraphField::compile_graph(&doc, &graph, 0.05).expect("compiles");
            // Swept well outside the reported box, so that material beyond it is found rather
            // than assumed absent.
            let n = 48;
            for i in 0..=n {
                for j in 0..=n {
                    for k in 0..=n {
                        let t = |m: usize| -70.0 + 140.0 * m as f64 / n as f64;
                        let p = Point3::new(t(i), t(j), t(k));
                        if field.distance(p) > 0.0 {
                            continue;
                        }
                        let inside = (0..3).all(|a| p[a] >= lo[a] - 1e-9 && p[a] <= hi[a] + 1e-9);
                        assert!(
                            inside,
                            "strength {strength}: solid at {p:?} lies outside the reported box \
                             {lo:?}..{hi:?}"
                        );
                    }
                }
            }
        }
    }

    /// The contract itself: [`ScalarField::distance`] must never overestimate. This is what the
    /// stretch division buys, and the reason it is not simply left out for being a small effect.
    ///
    /// Asked with `overstatement` rather than `gradient_deviation`, because the answer to "is
    /// this a distance" is deliberately no. What must hold is the weaker claim, and it must hold
    /// for a push as well as a pull -- which is where the asymmetric divisor earns its place. A
    /// push divided by 1, on the reasoning that shrinking cannot stretch, fails this.
    #[test]
    fn an_attracted_body_never_claims_more_room_than_it_has() {
        let (doc, curve) = attractor_document();
        let mut points = Vec::new();
        for i in 0..11 {
            for j in 0..11 {
                for k in 0..5 {
                    // Off the round numbers, so no probe lands on the sphere's own centre where
                    // the gradient has no direction -- see `probe_points` in
                    // `attractor_profile.rs` for the measurement that mistake spoiled.
                    points.push(Point3::new(
                        -5.0 + 60.0 * (i as f64 + 0.31) / 11.0,
                        -22.0 + 44.0 * (j as f64 + 0.31) / 11.0,
                        -10.0 + 20.0 * (k as f64 + 0.31) / 5.0,
                    ));
                }
            }
        }
        let dirs = crate::probe_directions();
        for strength in [0.5, 1.0, 1.4, -0.5, -0.95] {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
            graph.add(Node::Attract {
                input: ball.into(),
                toward: fieldform_scene::Attractor::Curve(curve),
                strength: lit(strength),
                falloff: lit(40.0),
            });
            let field = GraphField::compile_graph(&doc, &graph, 0.05).expect("compiles");
            let over = crate::overstatement(&field, &points, &dirs, 0.004, 1.2);
            // One step of slack, since the march can only locate a crossing to its own step.
            assert!(
                over <= 0.005,
                "strength {strength}: overstates the distance by {over} mm"
            );
        }
    }

    /// The warp is not an isometry, so the value is a lower bound -- and the graph walk and the
    /// compiler have to say so in the same breath, which is what
    /// `the_graph_walk_agrees_with_the_compiler_about_true_distance` stands on for every kind.
    ///
    /// A boolean over it stays exact, because min and max depend only on sign.
    #[test]
    fn an_attracted_body_blends_only_approximately_and_booleans_exactly() {
        let (doc, curve) = attractor_document();
        let build = |blend: bool| {
            let mut graph = Graph::empty();
            let boxed = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(40.0), lit(12.0), lit(60.0)],
                turn: fieldform_scene::no_turn(),
            });
            let pulled = graph.add(Node::Attract {
                input: boxed.into(),
                toward: fieldform_scene::Attractor::Curve(curve),
                strength: lit(0.6),
                falloff: lit(25.0),
            });
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(20.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
            if blend {
                graph.add(Node::Blend {
                    op: BooleanOp::Union,
                    a: pulled.into(),
                    b: ball.into(),
                    fillet: Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::G1 },
                });
            } else {
                graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![pulled.into(), ball.into()] });
            }
            let compiled = GraphField::compile(&doc, &part_with(graph.clone()), 0.05);
            (graph, compiled)
        };
        let (_, boolean) = build(false);
        assert!(boolean.is_ok(), "a boolean over an attractor is fine");
        let (graph, blended) = build(true);
        assert!(blended.is_ok(), "a blend over an attractor is allowed and reported");
        // Reported is the whole of what makes it allowed -- see the note on
        // `approximate_blends`. A blend that looked exact is the failure the old compile-time
        // refusal guarded against.
        assert_eq!(
            approximate_blends(&doc, &graph).len(),
            1,
            "the blend over an attracted body has to be named as approximate"
        );
    }

    /// An attractor wraps a field with no bounds without refusing it, which a twist cannot.
    ///
    /// Worth a test because the difference is easy to lose: a twist has to divide by a stretch
    /// taken from the body's own extent, so an unbounded operand leaves it nothing honest to
    /// divide by and it refuses. An attractor's divisor is `1 + strength` wherever the body is,
    /// so there is nothing to refuse -- and a half-space is the operand that tells them apart.
    #[test]
    fn an_attractor_accepts_a_body_a_twist_would_refuse() {
        let (doc, curve) = attractor_document();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane { origin: fieldform_scene::Place::At([lit(0.0); 3]), flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Attract {
            input: plane.into(),
            toward: fieldform_scene::Attractor::Curve(curve),
            strength: lit(0.5),
            falloff: lit(20.0),
        });
        assert!(
            GraphField::compile_graph(&doc, &graph, 0.05).is_ok(),
            "an attractor over a half-space should compile"
        );

        let mut twisted = Graph::empty();
        let plane = twisted.add(Node::Plane { origin: fieldform_scene::Place::At([lit(0.0); 3]), flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        twisted.add(Node::Twist {
            input: plane.into(),
            degrees_per_mm: lit(2.0),
            frame: fieldform_scene::Frame::default(),
        });
        assert!(
            GraphField::compile_graph(&doc, &twisted, 0.05).is_err(),
            "a twist over a half-space still has no extent to take a stretch from"
        );
    }

    /// **The test that was missing, and the defect it would have caught.**
    ///
    /// A normal is a claim about which way the surface faces, and the only thing that can check
    /// it is the field itself: the gradient of the composed field, differenced. The shader
    /// carried the *unwarped* body's normal through unchanged for a while, which is right only
    /// where the warp scales everything equally — nowhere, in general — and wrong worst where
    /// the radial factor collapses. It rendered as a dark band across the bulge and was
    /// reported as a shadow that should not have been there.
    ///
    /// `gpu_matches_cpu` could not have caught it: it measures where the surface *is*, and this
    /// is about which way it faces.
    #[test]
    fn an_attracted_surface_faces_where_its_own_field_says_it_does() {
        let (doc, curve) = attractor_document();
        for strength in [0.4, 1.0, 1.35, -0.6] {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
            graph.add(Node::Attract {
                input: ball.into(),
                toward: fieldform_scene::Attractor::Curve(curve),
                strength: lit(strength),
                falloff: lit(40.0),
            });
            let field = GraphField::compile_graph(&doc, &graph, 0.05).expect("compiles");

            // Points spread over the warped sphere's surface, found by bisecting outward along
            // rays from its centre — the analytic normal is only a claim about the surface, so
            // checking it in empty space would be checking nothing anybody sees.
            let mut checked = 0;
            let mut worst: f64 = 0.0;
            for i in 0..12 {
                for j in 0..12 {
                    let theta = std::f64::consts::PI * (i as f64 + 0.5) / 12.0;
                    let phi = 2.0 * std::f64::consts::PI * (j as f64 + 0.5) / 12.0;
                    let dir = Vector3::new(
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    );
                    let centre = Point3::new(30.0, 0.0, 0.0);
                    if field.distance(centre) > 0.0 {
                        continue;
                    }
                    let (mut lo, mut hi) = (0.0, 60.0);
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        if field.distance(centre + dir * mid) <= 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let on = centre + dir * (0.5 * (lo + hi));

                    let Some(analytic) = field.normal(on) else { continue };
                    // The field's own gradient, differenced. The step is coarse enough to see
                    // past floating-point noise and fine enough that the warp is locally linear.
                    let h = 1e-4;
                    let d = |o: Vector3<f64>| field.distance(on + o);
                    let grad = Vector3::new(
                        d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
                        d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
                        d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
                    );
                    let Some(differenced) = grad.try_normalize(1e-12) else { continue };
                    // The angle between them, which is what shading actually reads.
                    let angle = analytic.dot(&differenced).clamp(-1.0, 1.0).acos().to_degrees();
                    worst = worst.max(angle);
                    checked += 1;
                }
            }
            assert!(checked > 100, "strength {strength}: only {checked} surface points found");
            assert!(
                // Measured below 0.005 degrees -- the formula is exact, and what is left is
                // the differencing step. Set two orders of magnitude under the 5.13 degrees
                // the missing axial term produced, which is the defect this exists to catch,
                // rather than at the noise floor where it would be flaky.
                worst < 0.05,
                "strength {strength}: the analytic normal is {worst:.4} degrees off the \
                 field's own gradient, over {checked} surface points"
            );
        }
    }

    /// A twisted body reports a lower bound, so a blend over it is approximate rather than
    /// refused, and a boolean over it is exact.
    ///
    /// This is the consequence users meet, and the reason the declaration matters: a blend
    /// mixes by magnitude, so a field that understates itself pulls the seam toward itself and
    /// produces a fillet that is not the radius asked for.
    #[test]
    fn a_twisted_body_blends_only_approximately_and_booleans_exactly() {
        let doc = Document::new();
        let build = |blend: bool| {
            let mut graph = Graph::empty();
            let boxed = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(40.0), lit(12.0), lit(60.0)],
                turn: fieldform_scene::no_turn(),
            });
            let twist = graph.add(Node::Twist {
                input: boxed.into(),
                degrees_per_mm: lit(2.0),
                frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0); 3]),
            });
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(20.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
            if blend {
                graph.add(Node::Blend {
                    op: BooleanOp::Union,
                    a: twist.into(),
                    b: ball.into(),
                    fillet: Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::G1 },
                });
            } else {
                graph.add(Node::Boolean {
                    op: BooleanOp::Union,
                    inputs: vec![twist.into(), ball.into()],
                });
            }
            GraphField::compile(&doc, &part_with(graph), 0.05)
        };
        assert!(build(false).is_ok(), "a boolean over a twist is fine");
        // No longer refused -- `approximate_blends` carries why the answer changed -- but the
        // reason it once was still holds, so it must still be reported. A twist does not
        // preserve distance, so the fillet is not the radius asked for.
        assert!(build(true).is_ok(), "a blend over a twist is allowed now");
    }

    /// A twist needs to know how far the body reaches from the axis, and refuses when it
    /// cannot.
    ///
    /// The stretch factor is `r · rate`, so an unbounded body has no factor — and inventing one
    /// would produce a field that overstates its distance, which is the failure the whole
    /// division exists to prevent. Refusing names the problem instead.
    #[test]
    fn twisting_something_unbounded_is_refused_rather_than_guessed() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Twist {
            input: plane.into(),
            degrees_per_mm: lit(2.0),
            frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0); 3]),
        });
        assert!(matches!(
            GraphField::compile(&doc, &part_with(graph), 0.05),
            Err(CompileError::TwistNeedsBounds),
        ));
    }

    /// **A plane cannot be shelled**, because a shelled half-space is an infinite slab.
    ///
    /// Reported from the application: the operation was offered and went through. It is the
    /// same shape of mistake a twist would make against an unbounded body and refuses to --
    /// shelling is the body minus the body shrunk, which on something with no bounds is a wall
    /// with no edges. It renders perfectly well and is not a thing anyone can mesh, measure or
    /// make, which is the worst way for an operation to be wrong.
    ///
    /// Refused in the compile rather than greyed out of the menu: a control that vanishes
    /// teaches nothing about why, which this codebase learned once already with the revolve
    /// axis.
    #[test]
    fn a_body_with_no_bounds_cannot_be_shelled() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Shell { input: plane.into(), thickness: lit(2.0) });
        assert!(matches!(
            GraphField::compile_graph(&doc, &graph, 1e-3),
            Err(CompileError::ShellNeedsBounds),
        ));

        // And a bounded one still shells, so the refusal is about the bounds and not about
        // shelling having been broken.
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(10.0) });
        graph.add(Node::Shell { input: ball.into(), thickness: lit(2.0) });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");
        // Hollow: the middle is air, the wall is solid, outside is air.
        assert!(field.distance(Point3::new(0.0, 0.0, 0.0)) > 0.0, "hollow in the middle");
        assert!(field.distance(Point3::new(9.0, 0.0, 0.0)) < 0.0, "solid in the wall");
        assert!(field.distance(Point3::new(11.0, 0.0, 0.0)) > 0.0, "air outside");
    }

    /// A sphere mirrored across a plane is a sphere on the other side of it, and the distance
    /// away from it is still a distance.
    #[test]
    fn a_mirrored_body_lands_where_the_reflection_puts_it() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        graph.add(Node::Mirror {
            input: sphere.into(),
            frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::X, [lit(5.0), lit(0.0), lit(0.0)]),
            join: false,
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        // Reflected across x = 5: a centre at 30 lands at -20.
        let centre = Point3::new(-20.0, 0.0, 0.0);
        assert!(
            (field.distance(centre) - -10.0).abs() < 1e-9,
            "the centre is a radius deep"
        );
        assert!(field.distance(centre + Vector3::new(10.0, 0.0, 0.0)).abs() < 1e-9);
        // And the original is gone: a mirror reflects, it does not add.
        assert!(field.distance(Point3::new(30.0, 0.0, 0.0)) > 0.0);
    }

    /// A joined mirror is the body *and* its reflection, from one node.
    ///
    /// The behaviour the second block used to provide. It matters that this is one node rather
    /// than a node beside a union: the input is evaluated at two points, which is the case
    /// shader generation refused until point contexts landed, and the reason the operation was
    /// two blocks in the list for something a user did once.
    /// A 20 x 20 rectangle standing on y = 0, extruded 15 mm with `draft`, and either mirrored about
    /// y = 0 with its original (`joined`) or set beside its own reflection by a union (the reference
    /// the merge has to equal) — the owner's scene of backlog item 113 in its smallest form.
    fn half_on_the_plane(draft: f64, joined: bool) -> (Document, Graph) {
        let mut doc = Document::new();
        let id = fieldform_scene::CurveId::new();
        doc.curves.insert(
            id,
            fieldform_scene::Curve::polyline(
                id,
                "Profile".to_string(),
                vec![
                    Point3::new(-10.0, 0.0, 0.0),
                    Point3::new(10.0, 0.0, 0.0),
                    Point3::new(10.0, 20.0, 0.0),
                    Point3::new(-10.0, 20.0, 0.0),
                ],
                true,
            )
            .expect("a closed rectangle standing on y = 0"),
        );
        let mut graph = Graph::empty();
        let extrude = graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(15.0),
            draft: lit(draft),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        let plane = || {
            fieldform_scene::Frame::at_along(fieldform_scene::Axis::Y, [lit(0.0), lit(0.0), lit(0.0)])
        };
        if joined {
            graph.add(Node::Mirror { input: extrude.into(), frame: plane(), join: true });
        } else {
            let reflection = graph.add(Node::Mirror { input: extrude.into(), frame: plane(), join: false });
            graph.add(Node::Boolean {
                op: fieldform_scene::BooleanOp::Union,
                inputs: vec![extrude.into(), reflection.into()],
            });
        }
        (doc, graph)
    }

    /// **A mirrored extrude whose section touches its plane reads its true depth at the plane**
    /// (backlog item 113).
    ///
    /// The defect, measured: the joined mirror was the union of the half and its reflection, which
    /// kept the face they share buried in the solid, and the field 7.5 mm deep at the plane read
    /// 0.000. Built as one extrude of the merged section, there is no such face, and the depth is
    /// the distance to the nearest real face — here the top and bottom caps, both 7.5 mm away.
    #[test]
    fn a_mirrored_extrude_touching_its_plane_reads_its_true_depth_at_the_plane() {
        let (doc, graph) = half_on_the_plane(0.0, true);
        let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
        for y in [0.0, 0.05, -0.05, 0.3, -0.3] {
            let d = field.distance(Point3::new(0.0, y, 7.5));
            assert!((d + 7.5).abs() < 1e-6, "at y = {y}, 7.5 mm deep, the field read {d}");
        }
        // Leaning out, the halves used to overlap and still bury their walls; merged, nothing is
        // buried, and the depth is a bound on the real one rather than a fraction of a millimetre.
        let (doc, graph) = half_on_the_plane(5.0, true);
        let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
        let d = field.distance(Point3::new(0.0, 0.0, 7.5));
        assert!(d < -5.0, "drafted out, the field at the plane read {d}, not a buried wall");
    }

    /// **The merge is the same body as the union it replaces**, so building it as one changes what
    /// the field says about depth and nothing about where the surface is.
    #[test]
    fn a_merged_mirror_is_the_same_body_as_the_union_it_replaces() {
        for draft in [0.0, 5.0] {
            let (doc, merged) = half_on_the_plane(draft, true);
            let (reference, union) = half_on_the_plane(draft, false);
            assert!(crate::with_shared_faces_merged(&doc, &merged).is_some(), "it merges");
            assert!(crate::with_shared_faces_merged(&reference, &union).is_none(), "the reference does not");
            let one = GraphField::compile_graph(&doc, &merged, 1e-4).expect("compiles");
            let two = GraphField::compile_graph(&reference, &union, 1e-4).expect("compiles");
            let mut checked = 0;
            for xi in -8..=8 {
                for yi in -8..=8 {
                    for zi in -2..=10 {
                        let q = Point3::new(xi as f64 * 1.75, yi as f64 * 3.0, zi as f64 * 2.0);
                        let (a, b) = (one.distance(q), two.distance(q));
                        // Outside the union is exact, and so is the merge: they agree there. Inside,
                        // only the sign has to agree, because the union's depth is the defect.
                        if b > 1e-6 {
                            assert!((a - b).abs() < 1e-6, "draft {draft} at {q:?}: merged {a}, union {b}");
                        } else if b < -1e-6 {
                            assert!(a < 0.0, "draft {draft} at {q:?}: solid in the union, not merged");
                        }
                        checked += 1;
                    }
                }
            }
            assert!(checked > 3000);
        }
    }

    /// **Draft that leans in is left a union**, because the two halves really do part above the
    /// base: there is no shared face to remove, and the groove between them is the geometry asked for.
    #[test]
    fn an_inward_draft_keeps_the_groove_between_its_halves() {
        let (doc, graph) = half_on_the_plane(-5.0, true);
        assert!(crate::with_shared_faces_merged(&doc, &graph).is_none(), "no shared face, no rewrite");
        let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
        let d = field.distance(Point3::new(0.0, 0.0, 7.5));
        assert!(d > 0.5, "halfway up the plane is in the groove, outside the body: read {d}");
    }

    /// A 20 x 20 square in z = 0, extruded 15 mm up with `draft` and set going at `start`, with
    /// `sides`, and mirrored about z = `plane` with its original (`joined`) or set beside its own
    /// reflection by a union — the owner's second scene of backlog item 113, a wing mirrored about
    /// the plane its profile was drawn in, in its smallest form.
    fn mirrored_across_its_profile(
        draft: f64,
        start: f64,
        sides: fieldform_scene::Sides,
        plane: f64,
        joined: bool,
    ) -> (Document, Graph) {
        let mut doc = Document::new();
        let id = fieldform_scene::CurveId::new();
        doc.curves.insert(
            id,
            fieldform_scene::Curve::polyline(
                id,
                "Profile".to_string(),
                vec![
                    Point3::new(-10.0, -10.0, 0.0),
                    Point3::new(10.0, -10.0, 0.0),
                    Point3::new(10.0, 10.0, 0.0),
                    Point3::new(-10.0, 10.0, 0.0),
                ],
                true,
            )
            .expect("a closed square in z = 0"),
        );
        let mut graph = Graph::empty();
        let extrude = graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(15.0),
            draft: lit(draft),
            start: lit(start),
            sides,
            corners: Default::default(),
        });
        let frame = || {
            fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0), lit(0.0), lit(plane)])
        };
        if joined {
            graph.add(Node::Mirror { input: extrude.into(), frame: frame(), join: true });
        } else {
            let reflection = graph.add(Node::Mirror { input: extrude.into(), frame: frame(), join: false });
            graph.add(Node::Boolean {
                op: fieldform_scene::BooleanOp::Union,
                inputs: vec![extrude.into(), reflection.into()],
            });
        }
        (doc, graph)
    }

    /// **A mirror across an extrude's own profile reads its true depth at the plane** (backlog
    /// item 113, the owner's `centerline-blend1`).
    ///
    /// The start face is the face the two halves share, the whole profile of it, and the union
    /// kept it buried: 10 mm from the nearest wall, the field at the plane read nought. Built as
    /// one extrude with a second side, the plane is inside the material and the depth is the wall.
    #[test]
    fn a_mirror_across_an_extrudes_own_profile_reads_its_true_depth_at_the_plane() {
        use fieldform_scene::Sides;
        let (doc, union) = mirrored_across_its_profile(0.0, 0.0, Sides::One, 0.0, false);
        let buried = GraphField::compile_graph(&doc, &union, 1e-4).expect("compiles");
        let before = buried.distance(Point3::new(0.0, 0.0, 0.0));
        assert!(before.abs() < 1e-6, "the defect, as measured: the union read {before} at the plane");
        for draft in [0.0, -1.5, 3.0] {
            let (doc, graph) = mirrored_across_its_profile(draft, 0.0, Sides::One, 0.0, true);
            assert!(crate::with_shared_faces_merged(&doc, &graph).is_some(), "draft {draft} merges");
            let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
            for (x, depth) in [(0.0, 10.0), (5.0, 5.0)] {
                let d = field.distance(Point3::new(x, 0.0, 0.0));
                // At the plane the section is the profile whatever the draft, so the wall is where
                // it was drawn; a drafted extrude may read short of it but never long.
                assert!(d <= -0.9 * depth && d >= -depth - 1e-6, "draft {draft} at x = {x}: read {d}");
            }
        }
    }

    /// **The same body as the union it replaces**, at every draft: the merge changes what the field
    /// says about depth and nothing about where the surface is.
    #[test]
    fn a_mirror_across_its_profile_is_the_same_body_as_the_union_it_replaces() {
        use fieldform_scene::Sides;
        for draft in [0.0, -1.5, 3.0] {
            let (doc, merged) = mirrored_across_its_profile(draft, 0.0, Sides::One, 0.0, true);
            let (union_doc, union) = mirrored_across_its_profile(draft, 0.0, Sides::One, 0.0, false);
            assert!(crate::with_shared_faces_merged(&union_doc, &union).is_none(), "the reference does not");
            let one = GraphField::compile_graph(&doc, &merged, 1e-4).expect("compiles");
            let two = GraphField::compile_graph(&union_doc, &union, 1e-4).expect("compiles");
            let mut checked = 0;
            for xi in -8..=8 {
                for yi in -8..=8 {
                    for zi in -10..=10 {
                        let q = Point3::new(xi as f64 * 1.75, yi as f64 * 1.75 + 0.3, zi as f64 * 2.0);
                        let (a, b) = (one.distance(q), two.distance(q));
                        if b > 1e-6 {
                            assert!(a > 0.0, "draft {draft} at {q:?}: empty in the union, solid merged ({a})");
                            // Upright, both are exact outside and must agree to the last digit; a
                            // drafted wall is a bound in either, so only the side is held there.
                            if draft == 0.0 {
                                assert!((a - b).abs() < 1e-6, "at {q:?}: merged {a}, union {b}");
                            }
                        } else if b < -1e-6 {
                            assert!(a < 0.0, "draft {draft} at {q:?}: solid in the union, not merged ({a})");
                        }
                        checked += 1;
                    }
                }
            }
            assert!(checked > 6000);
        }
    }

    /// **Whatever divides its reading now is something `may_divide` names** (backlog item 114). The
    /// march settles a hit onto its surface only in a program `may_divide` says may need it, so a
    /// body it missed would be drawn proud of itself again, by its divisor in thresholds.
    #[test]
    fn whatever_divides_its_reading_now_is_a_kind_may_divide_names() {
        use fieldform_scene::Sides;
        let (doc, drafted) = mirrored_across_its_profile(3.0, 0.0, Sides::One, 0.0, false);
        assert!(crate::value_scale_with(&doc, &drafted, &[]) > 1.0, "a drafted extrude divides");
        assert!(crate::may_divide(&drafted), "and is named");
        // Upright, the same extrude divides by nothing now and is named all the same: its draft is
        // a number a drag can move, and a drag must not regenerate the shader.
        let (doc, upright) = mirrored_across_its_profile(0.0, 0.0, Sides::One, 0.0, false);
        assert_eq!(crate::value_scale_with(&doc, &upright, &[]), 1.0);
        assert!(crate::may_divide(&upright), "an extrude may divide whatever its draft is now");
        let mut sphere = Graph::empty();
        sphere.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(20.0) });
        assert!(!crate::may_divide(&sphere), "a sphere never divides, so its program never settles");
    }

    /// **Left a union where the halves do not share the start face**: a solid set going off its
    /// plane does not touch the mirror, one already on both sides crosses it, and a mirror plane
    /// beside the profile is a different reflection.
    #[test]
    fn a_mirror_that_does_not_share_the_start_face_is_left_a_union() {
        use fieldform_scene::Sides;
        let cases = [
            ("set going 2 mm off the plane", mirrored_across_its_profile(0.0, 2.0, Sides::One, 0.0, true)),
            ("on both sides already", mirrored_across_its_profile(0.0, 0.0, Sides::Symmetric, 0.0, true)),
            ("mirrored a millimetre away", mirrored_across_its_profile(0.0, 0.0, Sides::One, 1.0, true)),
        ];
        for (what, (doc, graph)) in cases {
            assert!(crate::with_shared_faces_merged(&doc, &graph).is_none(), "{what}: rewritten");
        }
    }

    /// **A joined mirror whose body reaches its plane is not called an exact distance** (backlog
    /// item 113, option 1 — the minimum its "Done when" asks for).
    ///
    /// A joined mirror is `min(f(p), f(mirror(p)))`, and a union of operands that touch keeps the
    /// face they share buried inside the solid: the sign is right everywhere, which is why the
    /// mirror alone renders clean, but the magnitude reads nought where the solid is deep. A blend
    /// reads magnitudes, so it takes that face for a surface and grooves along it. The field is
    /// still a valid lower bound, so nothing is illegal — what was wrong is *claiming* an exact
    /// distance, which a mirror did because its arm sat with the rigid motions. The reflection is
    /// one; the join is not. the project notes' rule is the one at stake: an approximate blend must
    /// announce itself where its radius is typed, and at nought degrees of draft the owner's
    /// fillet would have been wrong with nothing said at all.
    ///
    /// **A cuboid, because the merge is for a mirror of an extrude.** Cases 1 and 2 remove the
    /// buried face where they reach; this is the case no rewrite covers, which is what the
    /// announcement is for. Both answers are asserted every time:
    /// `carries_true_distance` and the compiled ops' `normalization` are two implementations of
    /// one decision and the pair is what must not drift.
    #[test]
    fn a_joined_mirror_whose_body_reaches_its_plane_is_not_called_an_exact_distance() {
        let doc = Document::new();
        // A 20 mm cube, placed so its bound touches, straddles or clears the plane at x = 0.
        let build = |centre_x: f64, join: bool| {
            let mut graph = Graph::empty();
            let cube = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(centre_x), lit(0.0), lit(0.0)]),
                size: [lit(20.0), lit(20.0), lit(20.0)],
                turn: fieldform_scene::no_turn(),
            });
            let mirror = graph.add(Node::Mirror {
                input: cube.into(),
                frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::X, [lit(0.0); 3]),
                join,
            });
            (graph, mirror)
        };

        let cases = [
            ("a face on the plane", 10.0, true, false),
            ("straddling the plane", 0.0, true, false),
            ("a 5 mm gap to the plane", 15.0, true, true),
            // Not joined: the body is only moved, which is a rigid motion and exact however near
            // the plane it lands. Nothing is unioned, so nothing is buried.
            ("a face on the plane, not joined", 10.0, false, true),
        ];
        for (what, centre_x, join, exact) in cases {
            let (graph, mirror) = build(centre_x, join);
            // No rewrite reaches a mirrored cuboid, so this is the mirror's own arm answering and
            // not cases 1 and 2 quietly removing the face.
            assert!(
                crate::with_shared_faces_merged(&doc, &graph).is_none(),
                "{what}: the merge took it, so this case says nothing about the announcement",
            );
            assert_eq!(
                crate::carries_true_distance(&doc, &graph, mirror),
                exact,
                "{what}: the walk",
            );
            let compiled = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
            assert_eq!(
                compiled.normalization() == Normalization::Distance,
                exact,
                "{what}: the compiled ops disagree with the walk",
            );
        }

        // And the consequence, which is the whole point of the answer: a fillet against such a
        // mirror is named, so the panel marks it `≈ approximate` beside the radius.
        let (mut graph, mirror) = build(10.0, true);
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(40.0), lit(0.0)]),
            radius: lit(5.0),
        });
        let blend = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: mirror.into(),
            b: ball.into(),
            fillet: Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::G1 },
        });
        assert!(
            crate::approximate_blends(&doc, &graph).contains(&blend),
            "a fillet against a mirror that buries a face has to announce itself",
        );
    }

    /// **A mirror the merge builds as one body is still an exact distance.**
    ///
    /// Where option 1 meets cases 1 and 2, and the reason `carries_true_distance` reads
    /// `with_shared_faces_merged` before it walks anything. The compiler rewrites first, so it
    /// compiles one extrude and reports an exact distance; a walk that did not would see a joined
    /// mirror reaching its plane and call it approximate. Then the panel would mark `≈ approximate`
    /// on a fillet that is exactly the radius asked for, and
    /// `the_graph_walk_agrees_with_the_compiler_about_true_distance` would be the test saying so.
    #[test]
    fn a_mirror_built_as_one_body_is_still_an_exact_distance() {
        let (doc, graph) = half_on_the_plane(0.0, true);
        assert!(
            crate::with_shared_faces_merged(&doc, &graph).is_some(),
            "this is the merged case, or the test is about nothing",
        );
        let root = graph.roots()[0];
        assert!(
            crate::carries_true_distance(&doc, &graph, root),
            "the merge leaves one extrude, which is exact, and the walk has to read the rewrite",
        );
        let compiled = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
        assert_eq!(compiled.normalization(), Normalization::Distance, "and so does the compiler");

        // The union the merge does not reach is left claiming exactness, which is deliberate: the
        // announcement for a *union* of touching bodies is option 3, the owner's call, and is not
        // what this change made. Pinned so that choosing it later is a visible decision.
        let (reference, union) = half_on_the_plane(0.0, false);
        assert!(crate::with_shared_faces_merged(&reference, &union).is_none(), "no mirror to merge");
        let root = union.roots()[0];
        assert!(
            crate::carries_true_distance(&reference, &union, root),
            "option 3 is not built; a hand-written union of touching bodies still claims exact",
        );
    }

    #[test]
    fn a_joined_mirror_keeps_the_body_as_well_as_its_reflection() {
        let doc = Document::new();
        let build = |join: bool| {
            let mut graph = Graph::empty();
            let sphere = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
            graph.add(Node::Mirror {
                input: sphere.into(),
                frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::X, [lit(5.0), lit(0.0), lit(0.0)]),
                join,
            });
            GraphField::compile(&doc, &part_with(graph), 0.05).unwrap()
        };

        let here = Point3::new(30.0, 0.0, 0.0);
        let there = Point3::new(-20.0, 0.0, 0.0);

        // Joined: both are solid, each a radius deep.
        let joined = build(true);
        assert!(
            (joined.distance(here) - -10.0).abs() < 1e-9,
            "the original should remain"
        );
        assert!(
            (joined.distance(there) - -10.0).abs() < 1e-9,
            "the reflection should be there"
        );

        // Not joined: only the reflection, which is a move rather than a mirror.
        let bare = build(false);
        assert!(bare.distance(here) > 0.0, "unjoined, the original is gone");
        assert!((bare.distance(there) - -10.0).abs() < 1e-9);
    }

    /// A `Plane` node can be the mirror, and the mirror follows it.
    ///
    /// The half an axis cannot express. It reads the plane rather than a copy of its numbers,
    /// so moving the plane moves the mirror — which is the whole reason to point at one instead
    /// of typing an origin.
    #[test]
    fn a_plane_node_can_be_the_mirror_and_moving_it_moves_the_reflection() {
        let doc = Document::new();
        let build = |plane_x: f64| {
            let mut graph = Graph::empty();
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At([lit(plane_x), lit(0.0), lit(0.0)]),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let sphere = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(30.0)]),
                radius: lit(10.0),
            });
            graph.add(Node::Mirror {
                input: sphere.into(),
                frame: fieldform_scene::Frame::From { node: plane },
                join: false,
            });
            GraphField::compile(&doc, &part_with(graph), 0.05).unwrap()
        };

        // A `Plane` faces +Z in its own frame, so a plane at z = 0 reflects z.
        let at_zero = build(0.0);
        assert!(
            (at_zero.distance(Point3::new(0.0, 0.0, -30.0)) - -10.0).abs() < 1e-9,
            "reflected through z = 0",
        );

        // The plane is consumed, so it stops being a body of its own — the same rule a
        // boolean's operands follow, and what keeps a half-space from filling the viewport.
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(5.0),
        });
        graph.add(Node::Mirror {
            input: sphere.into(),
            frame: fieldform_scene::Frame::From { node: plane },
            join: true,
        });
        assert!(
            !graph.roots().contains(&plane),
            "a mirroring plane is not a body"
        );
    }

    /// A mirror whose plane names something that is not a plane is refused, not guessed at.
    #[test]
    fn a_mirror_pointed_at_a_node_that_is_not_a_plane_is_refused() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(5.0),
        });
        let other = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(20.0); 3]),
            radius: lit(5.0),
        });
        graph.add(Node::Mirror {
            input: ball.into(),
            frame: fieldform_scene::Frame::From { node: other },
            join: false,
        });
        assert!(matches!(
            GraphField::compile(&doc, &part_with(graph), 0.05),
            Err(CompileError::NotAFrameSource(_)),
        ));
    }

    /// D8's property. A reflection is an isometry, so unlike a scale there is nothing to undo
    /// and the gradient must come through at exactly unit length.
    #[test]
    fn a_mirrored_field_still_moves_at_unit_rate() {
        for axis in [
            fieldform_scene::Axis::X,
            fieldform_scene::Axis::Y,
            fieldform_scene::Axis::Z,
        ] {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let sphere = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(7.0), lit(-3.0), lit(2.0)]),
                radius: lit(10.0),
            });
            graph.add(Node::Mirror {
                input: sphere.into(),
                frame: fieldform_scene::Frame::at_along(axis, [lit(1.5), lit(2.5), lit(-4.0)]),
                join: false,
            });
            let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
            let points: Vec<Point3<f64>> = (0..40)
                .map(|i| {
                    let t = i as f64;
                    Point3::new(40.0 + t, -14.0 + 0.7 * t, 9.0 - 0.3 * t)
                })
                .collect();
            assert!(
                gradient_deviation(&field, &points, 1e-4) < 1e-6,
                "{axis:?} broke the unit-gradient property",
            );
        }
    }

    /// Mirroring twice about the same plane is the original body.
    ///
    /// The cheapest check that the reflection is its own inverse, which catches a sign or a
    /// factor of two in the plane offset — the two mistakes this arithmetic invites.
    #[test]
    fn mirroring_twice_about_one_plane_gives_the_body_back() {
        let doc = Document::new();
        let build = |times: usize| {
            let mut graph = Graph::empty();
            let mut node = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(12.0), lit(4.0), lit(-6.0)]),
                size: [lit(20.0), lit(8.0), lit(14.0)],
                turn: fieldform_scene::no_turn(),
            });
            for _ in 0..times {
                node = graph.add(Node::Mirror {
                    input: node.into(),
                    frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::X, [lit(3.0), lit(0.0), lit(0.0)]),
                    join: false,
                });
            }
            GraphField::compile(&doc, &part_with(graph), 0.05).unwrap()
        };
        let (plain, twice) = (build(0), build(2));
        for i in 0..40 {
            let t = i as f64;
            let p = Point3::new(-30.0 + t * 2.0, -10.0 + t * 0.4, 8.0 - t * 0.3);
            assert!(
                (plain.distance(p) - twice.distance(p)).abs() < 1e-9,
                "mirroring twice changed the body at {p:?}",
            );
        }
    }

    /// A mirrored body is still blendable, which is the consequence of a reflection preserving
    /// distance exactly — and the thing a caller would notice if it did not.
    #[test]
    fn a_mirrored_body_can_still_be_blended() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let a = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let mirrored = graph.add(Node::Mirror {
            input: a.into(),
            frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::X, [lit(8.0), lit(0.0), lit(0.0)]),
            join: false,
        });
        let b = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(14.0), lit(0.0), lit(0.0)]),
            radius: lit(6.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: mirrored.into(),
            b: b.into(),
            fillet: Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::G1 },
        });
        assert!(GraphField::compile(&doc, &part_with(graph), 0.05).is_ok());
    }

    /// Every `LowerBound` source in the kernel, checked by the one property they all owe.
    ///
    /// D8 asks for the unit-gradient test on every source that claims `Distance`;
    /// `gradient_deviation` is that, and it is applied as a matter of course. A `LowerBound`
    /// source is exempt from *that* question by construction — its magnitude is deliberately
    /// not a distance — and owes a different one, which is the contract itself:
    /// **`distance` must never overestimate**. Nothing was asking it in one place.
    ///
    /// Five hand-rolled versions of this walk had grown across two files, each with its own
    /// probe points, its own step and its own directions — which is how one of them comes to
    /// test a weaker thing than the others without anyone noticing. `overstatement` is the
    /// shared form, and this is the list of what must pass it.
    ///
    /// A source added to the kernel that reports `LowerBound` belongs in this test. That is the
    /// authoring rule the backlog's checklist states, and this is where it is enforced.
    #[test]
    fn every_lower_bound_source_keeps_the_contract_it_owes() {
        let mut doc = Document::new();
        // The curve the thin extrude below is built on: three runs through the middle of the
        // probes, the last doubling back past a right angle so the wall is mitred there.
        let wall_curve = fieldform_scene::CurveId::new();
        doc.curves.insert(
            wall_curve,
            fieldform_scene::Curve::polyline(
                wall_curve,
                "chain",
                vec![
                    Point3::new(-25.0, -15.0, -5.0),
                    Point3::new(5.0, -15.0, -5.0),
                    Point3::new(15.0, 10.0, -5.0),
                    Point3::new(-10.0, 0.0, -5.0),
                ],
                false,
            )
            .expect("a polyline"),
        );
        let probes: Vec<Point3<f64>> = (0..14)
            .flat_map(|i| {
                (0..6).map(move |j| {
                    let t = i as f64;
                    let u = j as f64;
                    // Off every axis and every diagonal, so a bound that happens to hold where
                    // the geometry is aligned is not what is being measured.
                    Point3::new(
                        -33.0 + t * 4.7 + u * 0.9,
                        -21.0 + t * 3.1 - u * 1.7,
                        -17.0 + t * 2.3 + u * 2.1,
                    )
                })
            })
            .collect();
        let directions = crate::probe_directions();

        // Each is built as a part, so what is measured is the field a viewport would trace
        // rather than a leaf in isolation — the composition is where a bound gets lost.
        let cases: Vec<(&str, Graph)> = vec![
            ("a gyroid lattice", {
                let mut g = Graph::empty();
                let cube = g.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([lit(0.0); 3]),
                    size: [lit(80.0); 3],
                    turn: fieldform_scene::no_turn(),
                });
                let cell = g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::Gyroid,
                    map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0) },
                    phase: LatticePhase::sheet(lit(2.0)),
                });
                g.add(Node::Boolean {
                    op: BooleanOp::Intersection,
                    inputs: vec![cube.into(), cell.into()],
                });
                g
            }),
            ("a cylindrical lattice", {
                let mut g = Graph::empty();
                let tube = g.add(Node::Cylinder { edge: Default::default(),
                    ends: fieldform_scene::Ends::between(
                        fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(-40.0)]),
                        fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(40.0)])),
                    radius: lit(35.0),
                });
                let cell = g.add(Node::Lattice {
                    frame: fieldform_scene::Frame::default(),
                    cell: fieldform_scene::LatticeCell::SchwarzP,
                    map: fieldform_scene::CellMap::Cylindrical {
                        cell_radius: lit(12.0),
                        cell_height: lit(12.0),
                        arc_count: 8,
                    },
                    phase: LatticePhase::sheet(lit(2.0)),
                });
                g.add(Node::Boolean {
                    op: BooleanOp::Intersection,
                    inputs: vec![tube.into(), cell.into()],
                });
                g
            }),
            ("a twisted bar", {
                let mut g = Graph::empty();
                let bar = g.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([lit(0.0); 3]),
                    size: [lit(60.0), lit(18.0), lit(70.0)],
                    turn: fieldform_scene::no_turn(),
                });
                g.add(Node::Twist {
                    input: bar.into(),
                    degrees_per_mm: lit(2.0),
                    frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0); 3]),
                });
                g
            }),
            ("a driven offset", {
                let mut g = Graph::empty();
                let plane = g.add(Node::Plane {
                    origin: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(-20.0)]),
                    flipped: false,
                    turn: fieldform_scene::no_turn(),
                });
                let ramp = g.add(Node::Ramp {
                    input: plane.into(),
                    in_min: lit(0.0),
                    in_max: lit(40.0),
                    out_min: lit(-6.0),
                    out_max: lit(9.0),
                    shape: fieldform_scene::RampShape::Smoother,
                });
                let body = g.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([lit(0.0); 3]),
                    size: [lit(40.0), lit(30.0), lit(40.0)],
                    turn: fieldform_scene::no_turn(),
                });
                g.add(Node::Offset {
                    input: body.into(),
                    distance: ParamRef::Field(ramp),
                });
                g
            }),
            // **A wall is a distance in its own plane**, so there is no wall here that is a
            // bound by itself — which is what `outline-wall-spec.md` first expected this list
            // to hold. What makes a thin extrude a bound is what makes a thick one a bound: a
            // lean and a draft. Both at once, so the wall's reading goes through the shear and
            // the slope together, on a chain with a mitre and a round end in it.
            ("a leaning, drafted thin extrude", {
                let mut g = Graph::empty();
                g.add(Node::Extrude { rims: Default::default(),
                    outline: fieldform_scene::Outline::Wall {
                        curve: wall_curve,
                        thickness: lit(5.0),
                        side: fieldform_scene::WallSide::Left,
                        ends: fieldform_scene::WallEnds::Round,
                    },
                    sweep: fieldform_scene::Sweep::Toward { xyz: [lit(0.3), lit(0.1), lit(1.0)] },
                    height: lit(22.0),
                    draft: lit(8.0),
                    start: lit(0.0),
                    sides: fieldform_scene::Sides::One,
                    corners: Default::default(),
                });
                g
            }),
        ];

        for (what, graph) in cases {
            let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
            assert_eq!(
                field.normalization(),
                Normalization::LowerBound,
                "{what} should be declaring a lower bound",
            );
            let over = crate::overstatement(&field, &probes, &directions, 0.05, 1.2);
            assert!(
                over < 1e-6,
                "{what} claimed {over} mm more room than it had",
            );
        }
    }

    /// A shelled cube is a wall of the thickness asked for, measured inward.
    ///
    /// The arithmetic in full: a 40 mm cube walled to 5 mm is solid from x = 15 to x = 20 on
    /// the mid-plane and hollow inside that. Measured on the surface positions rather than on
    /// the values, since the field between two faces is the distance to the nearer one.
    #[test]
    fn a_shelled_body_is_a_wall_of_the_thickness_asked_for() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Shell {
            input: cube.into(),
            thickness: lit(5.0),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        // Inside the wall.
        assert!(
            field.distance(Point3::new(17.5, 0.0, 0.0)) < 0.0,
            "the wall should be solid"
        );
        // Hollow within it, and outside beyond it.
        assert!(
            field.distance(Point3::new(0.0, 0.0, 0.0)) > 0.0,
            "the middle should be empty"
        );
        assert!(field.distance(Point3::new(25.0, 0.0, 0.0)) > 0.0);
        // The two surfaces are exactly the thickness apart.
        assert!(
            field.distance(Point3::new(20.0, 0.0, 0.0)).abs() < 1e-9,
            "the outer face"
        );
        assert!(
            field.distance(Point3::new(15.0, 0.0, 0.0)).abs() < 1e-9,
            "the inner face"
        );
    }

    /// The outside does not move, which is what "measured inward" has to mean.
    #[test]
    fn shelling_a_body_leaves_its_outside_dimensions_alone() {
        let doc = Document::new();
        let build = |shelled: bool| {
            let mut graph = Graph::empty();
            let cube = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(40.0), lit(30.0), lit(50.0)],
                turn: fieldform_scene::no_turn(),
            });
            if shelled {
                graph.add(Node::Shell {
                    input: cube.into(),
                    thickness: lit(4.0),
                });
            }
            GraphField::compile(&doc, &part_with(graph), 0.05).unwrap()
        };
        let (solid, walled) = (build(false), build(true));
        for p in [
            Point3::new(20.0, 0.0, 0.0),
            Point3::new(0.0, 15.0, 0.0),
            Point3::new(0.0, 0.0, 25.0),
        ] {
            assert!(solid.distance(p).abs() < 1e-9);
            assert!(
                walled.distance(p).abs() < 1e-9,
                "the outer surface moved at {p:?}"
            );
        }
    }

    /// A wall thicker than the body leaves nothing, which is a true answer rather than an error.
    #[test]
    fn a_wall_thicker_than_the_body_leaves_the_body_solid() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Shell {
            input: cube.into(),
            thickness: lit(60.0),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
        // Nothing was hollowed out, so the middle is still solid.
        assert!(field.distance(Point3::new(0.0, 0.0, 0.0)) < 0.0);
    }

    /// A shell is a boolean of two distance fields, so it is one — and stays blendable.
    #[test]
    fn a_shelled_body_can_still_be_blended() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let walled = graph.add(Node::Shell {
            input: cube.into(),
            thickness: lit(4.0),
        });
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(40.0); 3]),
            radius: lit(10.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: walled.into(),
            b: ball.into(),
            fillet: Fillet { radius: lit(3.0), continuity: fieldform_scene::Continuity::G1 },
        });
        assert!(GraphField::compile(&doc, &part_with(graph), 0.05).is_ok());
    }

    /// The shelled body is consumed, so a shell is one block rather than a wall beside the
    /// solid it came from.
    #[test]
    fn a_shelled_body_stops_being_a_body_of_its_own() {
        let mut graph = Graph::empty();
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let walled = graph.add(Node::Shell {
            input: cube.into(),
            thickness: lit(4.0),
        });
        assert_eq!(graph.roots(), vec![walled], "one block, not two");
    }

    /// A body whose offset varies with height, and the plane driving it.
    fn driven_offset(out_min: f64, out_max: f64, shape: fieldform_scene::RampShape) -> Graph {
        let mut graph = Graph::empty();
        // A plane at z = -20 facing +Z, so its value is `z + 20` — a field whose reading is
        // known exactly, which is what lets the assertions below be equalities.
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(-20.0)]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(0.0),
            in_max: lit(40.0),
            out_min: lit(out_min),
            out_max: lit(out_max),
            shape,
        });
        let body = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0), lit(40.0), lit(40.0)],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Offset {
            input: body.into(),
            distance: ParamRef::Field(ramp),
        });
        graph
    }

    /// The offset is what the ramp says it is, at each height.
    ///
    /// The arithmetic in full, so a sign or a swapped end has nowhere to hide: the box's face
    /// is at x = 20, the plane reads `z + 20`, and a linear ramp over `[0, 40]` onto `[0, 10]`
    /// makes the offset `(z + 20) / 4`. So the surface at height `z` is at
    /// `20 + (z + 20) / 4`, which is 22.5 near the bottom of the box and 27.5 near the top.
    ///
    /// Probed inside the box's height rather than on its faces: exactly at `z = ±20` the
    /// cuboid reads zero along the whole ray, so there is no sign change to bisect for and the
    /// measurement would say nothing about the ramp.
    #[test]
    fn a_driven_offset_moves_the_surface_by_what_the_ramp_reads() {
        let doc = Document::new();
        let graph = driven_offset(0.0, 10.0, fieldform_scene::RampShape::Linear);
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        for (z, expected) in [(-10.0, 22.5), (0.0, 25.0), (10.0, 27.5)] {
            // Bisect for the surface rather than trusting the value, since a driven offset is
            // a lower bound and its magnitude is deliberately not the distance.
            let mut lo = 0.0;
            let mut hi = 80.0;
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if field.distance(Point3::new(mid, 0.0, z)) < 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            assert!(
                (lo - expected).abs() < 1e-6,
                "at z = {z} the surface should be at {expected}, found {lo}",
            );
        }
    }

    /// The value never overstates the room before the surface — the whole contract a driven
    /// parameter has to keep.
    ///
    /// This is what the divisor buys, and marching *past* what the field claims is the only
    /// way to test it: a march that stops at the claim can only ever confirm it.
    #[test]
    fn a_driven_offset_never_overstates_the_room_before_its_surface() {
        let doc = Document::new();
        for (a, b) in [(0.0, 10.0), (10.0, 0.0), (-8.0, 8.0)] {
            for shape in fieldform_scene::RampShape::ALL {
                let field = GraphField::compile(&doc, &part_with(driven_offset(a, b, shape)), 0.05)
                    .unwrap();
                for i in 0..24 {
                    for j in 0..8 {
                        let from = Point3::new(
                            -40.0 + i as f64 * 3.5,
                            -22.0 + j as f64 * 6.0,
                            -30.0 + (i % 7) as f64 * 9.0,
                        );
                        let claimed = field.distance(from);
                        if claimed <= 0.0 {
                            continue;
                        }
                        for d in [
                            Vector3::new(1.0, 0.0, 0.0),
                            Vector3::new(0.0, 1.0, 0.0),
                            Vector3::new(0.0, 0.0, 1.0),
                            Vector3::new(0.577, 0.577, 0.577),
                            Vector3::new(-0.577, 0.577, -0.577),
                        ] {
                            let mut t = 0.0;
                            while t < claimed * 1.2 {
                                if field.distance(from + d * t) <= 0.0 {
                                    assert!(
                                        t >= claimed - 1e-6,
                                        "{shape:?} over ({a}, {b}): claimed {claimed} at                                          {from:?} but a surface is {t} away",
                                    );
                                    break;
                                }
                                t += 0.05;
                            }
                        }
                    }
                }
            }
        }
    }

    /// A driven offset is a lower bound, so a blend over it is refused at compile time.
    ///
    /// The consequence the decision accepted rather than worked around: a variable-thickness
    /// wall cannot be filleted, and the refusal is the honest form of that. Making it work is
    /// a separate question about whether blends may take bounded operands.
    #[test]
    fn a_driven_offset_is_a_lower_bound_and_a_blend_refuses_it() {
        let doc = Document::new();
        let mut graph = driven_offset(0.0, 10.0, fieldform_scene::RampShape::Smooth);
        let varying = *graph.roots().first().expect("the offset is the only body");
        let other = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(60.0); 3]),
            radius: lit(10.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: varying.into(),
            b: other.into(),
            fillet: Fillet { radius: lit(3.0), continuity: fieldform_scene::Continuity::G1 },
        });
        // Allowed, and reported. A driven offset's gradient is not unit, so the fillet is
        // approximate -- which the panel now says rather than the compiler forbidding it.
        assert!(GraphField::compile(&doc, &part_with(graph.clone()), 0.05).is_ok());
        assert!(
            !approximate_blends(&doc, &graph).is_empty(),
            "a blend over a driven offset has to be reported as approximate",
        );

        // And a *fixed* offset of the same body is not refused, so the refusal is about the
        // varying rather than about offsets.
        let mut graph = Graph::empty();
        let body = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let grown = graph.add(Node::Offset {
            input: body.into(),
            distance: lit(5.0),
        });
        let other = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(60.0); 3]),
            radius: lit(10.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: grown.into(),
            b: other.into(),
            fillet: Fillet { radius: lit(3.0), continuity: fieldform_scene::Continuity::G1 },
        });
        assert!(GraphField::compile(&doc, &part_with(graph), 0.05).is_ok());
    }

    /// Only a ramp may drive a parameter. Anything else is refused, and says which rule.
    #[test]
    fn a_parameter_driven_by_something_that_is_not_a_ramp_is_refused() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let not_a_ramp = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(5.0),
        });
        graph.add(Node::Offset {
            input: body.into(),
            distance: ParamRef::Field(not_a_ramp),
        });
        assert!(matches!(
            GraphField::compile(&doc, &part_with(graph), 0.05),
            Err(CompileError::DrivenParameterNeedsARamp),
        ));
    }

    /// Every parameter but an offset's distance refuses to be driven, and names itself.
    ///
    /// The mechanism is general and the permission is not — deliberately. Each further
    /// consumer is its own question about what varying costs it, and shipping them together
    /// would mean several unverified bounds instead of one measured.
    #[test]
    fn a_driven_parameter_anywhere_but_an_offset_is_refused_by_name() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(0.0),
            in_max: lit(40.0),
            out_min: lit(1.0),
            out_max: lit(4.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: ParamRef::Field(ramp),
        });

        match GraphField::compile(&doc, &part_with(graph), 0.05) {
            Err(CompileError::ParameterCannotBeDriven(what)) => {
                assert!(
                    what.contains("sphere"),
                    "the message should name the block: {what}"
                );
            }
            Err(other) => panic!("expected a refusal naming the block, got {other:?}"),
            Ok(_) => panic!("a driven sphere radius should be refused"),
        }
    }

    /// A ramp is not a body, so it never draws and never joins the part's union.
    #[test]
    fn a_ramp_is_not_a_body_even_when_nothing_reads_it() {
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(0.0),
            in_max: lit(1.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        assert!(
            !graph.roots().contains(&ramp),
            "a ramp is a value, not a body"
        );
        // And what it reads is consumed, so a driving plane does not fill the viewport.
        assert!(!graph.roots().contains(&plane));
    }

    /// A ramp with no input range is refused rather than clamped into a step.
    ///
    /// A zero span makes the slope unbounded, so there is no number for a consumer to divide
    /// by — and a bound that cannot be stated is the one thing this design does not allow.
    #[test]
    fn a_ramp_with_an_empty_input_range_is_refused() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(7.0),
            in_max: lit(7.0),
            out_min: lit(0.0),
            out_max: lit(5.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        let body = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Offset {
            input: body.into(),
            distance: ParamRef::Field(ramp),
        });
        assert!(matches!(
            GraphField::compile(&doc, &part_with(graph), 0.05),
            Err(CompileError::EmptyRampRange),
        ));
    }

    /// The extent covers the body at its most grown, not at the offset it happens to start at.
    #[test]
    fn a_driven_offsets_extent_covers_the_largest_the_ramp_reaches() {
        let doc = Document::new();
        let graph = driven_offset(0.0, 10.0, fieldform_scene::RampShape::Linear);
        let (lo, hi) = crate::part_extent(&doc, &graph)
            .as_box()
            .expect("a bounded body");
        // The box is 40 across, so ±20, grown by the ramp's largest output.
        assert!(
            hi.x >= 30.0 - 1e-9,
            "the box should reach 30, found {}",
            hi.x
        );
        assert!(lo.x <= -30.0 + 1e-9);
    }

    /// Two boxes meeting at a right angle along x = y = 0, their seam filleted by a radius that
    /// follows a ramp up Z: `k_lo` at z = -20, `k_hi` at z = +20 (`round-chamfer-spec.md` §9.1).
    ///
    /// Chosen so the answer is a closed form: just off the seam both boxes read their own face's
    /// distance, `x` and `y`, so along the diagonal `x = y = t` every form's zero set is at
    /// `t = k · (1 − 2^(−1/n))` -- half the leg for a chamfer, `k(1 − 1/√2)` for the circle.
    fn driven_blend(
        k_lo: f64,
        k_hi: f64,
        continuity: fieldform_scene::Continuity,
        shape: fieldform_scene::RampShape,
    ) -> Graph {
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(-20.0)]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(0.0),
            in_max: lit(40.0),
            out_min: lit(k_lo),
            out_max: lit(k_hi),
            shape,
        });
        let a = graph.add(Node::Cuboid {
            edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(-50.0), lit(0.0), lit(0.0)]),
            size: [lit(100.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let b = graph.add(Node::Cuboid {
            edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0), lit(-50.0), lit(0.0)]),
            size: [lit(100.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: a.into(),
            b: b.into(),
            fillet: Fillet { radius: ParamRef::Field(ramp), continuity },
        });
        graph
    }

    /// The fillet is the radius the ramp reads where the seam is, at each height and at every
    /// shape -- a radius that varies is still a radius.
    #[test]
    fn a_driven_blends_radius_is_the_ramps_reading_where_the_seam_is() {
        let doc = Document::new();
        for continuity in fieldform_scene::Continuity::ALL {
            let graph = driven_blend(2.0, 6.0, continuity, fieldform_scene::RampShape::Linear);
            let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
            let n = Fillet { radius: lit(1.0), continuity }.exponent();
            for (z, k) in [(-20.0, 2.0), (0.0, 4.0), (20.0, 6.0)] {
                let expected = k * (1.0 - 2f64.powf(-1.0 / n));
                // Bisect along the diagonal for the surface rather than trusting the value: a
                // driven blend is a bound and its magnitude is deliberately not the distance.
                let (mut lo, mut hi) = (0.0, 10.0);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if field.distance(Point3::new(mid, mid, z)) < 0.0 { lo = mid } else { hi = mid }
                }
                assert!(
                    (lo - expected).abs() < 1e-6,
                    "{continuity:?} at z = {z}: the seam should be {expected} out, found {lo}",
                );
            }
        }
    }

    /// The value never overstates the room before the surface -- the whole contract a driven
    /// radius has to keep, and what `driven_blend_divisor` buys. Marched *past* the claim, as
    /// the offset's test is, because a march that stops at the claim can only confirm it.
    #[test]
    fn a_driven_blend_never_overstates_the_room_before_its_surface() {
        let doc = Document::new();
        for continuity in fieldform_scene::Continuity::ALL {
            for shape in fieldform_scene::RampShape::ALL {
                let field =
                    GraphField::compile(&doc, &part_with(driven_blend(1.0, 9.0, continuity, shape)), 0.05)
                        .unwrap();
                for i in 0..16 {
                    for j in 0..8 {
                        let from = Point3::new(
                            -4.0 + i as f64 * 1.75,
                            -3.0 + j as f64 * 3.0,
                            -28.0 + (i % 7) as f64 * 9.0,
                        );
                        let claimed = field.distance(from);
                        if claimed <= 0.0 {
                            continue;
                        }
                        for d in [
                            Vector3::new(1.0, 0.0, 0.0),
                            Vector3::new(0.0, 1.0, 0.0),
                            Vector3::new(0.0, 0.0, 1.0),
                            Vector3::new(-0.707, -0.707, 0.0),
                            Vector3::new(-0.577, -0.577, 0.577),
                            Vector3::new(-0.577, -0.577, -0.577),
                        ] {
                            let mut t = 0.0;
                            while t < claimed * 1.2 {
                                if field.distance(from + d * t) <= 0.0 {
                                    assert!(
                                        t >= claimed - 1e-6,
                                        "{continuity:?} {shape:?}: claimed {claimed} at {from:?} \
                                         but a surface is {t} away along {d:?}",
                                    );
                                    break;
                                }
                                t += 0.02;
                            }
                        }
                    }
                }
            }
        }
    }

    /// What a driven radius divides by is the ramp's rate times what the form's reading moves
    /// per millimetre of radius -- one for the bounded circle and chamfer, `1 + 2^(−1/n)` for
    /// the divided forms -- and the compiled field reports it as a bound, the walk agrees, and
    /// the blend is *not* marked approximate: its operands are exact, and the radius it has
    /// where it has one is the radius asked for.
    #[test]
    fn a_driven_blend_divides_by_the_rate_its_radius_costs_and_is_a_bound() {
        let doc = Document::new();
        for continuity in fieldform_scene::Continuity::ALL {
            let graph = driven_blend(2.0, 6.0, continuity, fieldform_scene::RampShape::Linear);
            let blend = *graph.roots().first().expect("the blend is the only body");
            let n = Fillet { radius: lit(1.0), continuity }.exponent();
            // The ramp runs four millimetres of radius over forty of height.
            let rate = 0.1;
            let moves = if n == 1.0 || n == 2.0 { 1.0 } else { 1.0 + 2f64.powf(-1.0 / n) };
            let divisor = crate::driven_blend_divisor(&doc, &graph, blend);
            assert!(
                (divisor - (1.0 + moves * rate)).abs() < 1e-9,
                "{continuity:?}: divisor {divisor}, expected {}",
                1.0 + moves * rate
            );
            let field = GraphField::compile(&doc, &part_with(graph.clone()), 0.05).unwrap();
            assert_eq!(field.normalization(), Normalization::LowerBound);
            assert!(!carries_true_distance(&doc, &graph, blend));
            assert!(approximate_blends(&doc, &graph).is_empty(), "{continuity:?} was marked approximate");
            assert!(
                (field.lipschitz() - crate::graph_lipschitz(&doc, &graph, blend)).abs() < 1e-9,
                "{continuity:?}: the walk says {} and the ops {}",
                crate::graph_lipschitz(&doc, &graph, blend),
                field.lipschitz()
            );
            // And the reach the graph reads for it is the ramp's top, not nought.
            assert!((crate::fillet_reach(&doc, &graph, graph.node(blend).unwrap().fillet().unwrap()) - 6.0).abs() < 1e-12);
        }
    }

    /// The graph-level Lipschitz walk agrees with the compiled field's own answer.
    ///
    /// They have to, and nothing else would notice if they stopped: the shader divides by the
    /// graph-level number and the CPU by whichever it used, and a mismatch moves no surface —
    /// it shows only as a tracer stepping further than the field justifies.
    #[test]
    fn the_graph_level_lipschitz_agrees_with_the_compiled_fields() {
        let doc = Document::new();
        for (a, b) in [(0.0, 10.0), (0.0, 60.0), (-8.0, 8.0)] {
            for shape in fieldform_scene::RampShape::ALL {
                let graph = driven_offset(a, b, shape);
                let root = *graph.roots().first().unwrap();
                let field = GraphField::compile(&doc, &part_with(graph.clone()), 0.05).unwrap();
                assert!(
                    (field.lipschitz() - crate::graph_lipschitz(&doc, &graph, root)).abs() < 1e-9,
                    "{shape:?} over ({a}, {b})",
                );
            }
        }
    }

    /// The shape constants are the real peak slopes, differenced rather than asserted.
    ///
    /// A factor that is too small is a bound that does not bound, and everything above it then
    /// divides by too little and claims room it does not have. These were derived by hand —
    /// `s′ = 6t(1 − t)` and `s′ = 30t²(1 − t)²`, both peaking at `t = ½` — so they are checked
    /// against the shapes themselves.
    #[test]
    fn each_ramp_shapes_slope_factor_is_the_steepest_that_shape_gets() {
        for shape in fieldform_scene::RampShape::ALL {
            let h = 1e-6;
            let mut worst: f64 = 0.0;
            for i in 0..=10_000 {
                let t = i as f64 / 10_000.0;
                let slope = (shape.apply(t + h) - shape.apply(t - h)) / (2.0 * h);
                worst = worst.max(slope.abs());
            }
            let claimed = shape.slope_factor();
            assert!(
                worst <= claimed + 1e-4,
                "{shape:?} reaches {worst}, above its claimed {claimed} — the bound does not bound",
            );
            // And not wastefully loose either, or every driven parameter pays for slack that
            // is not there.
            assert!(
                worst >= claimed - 1e-3,
                "{shape:?} only reaches {worst} against a claimed {claimed}",
            );
        }
    }

    /// A row of copies, from one node, each where it was asked for.
    ///
    /// The count includes the original — six means six bodies, not seven — which is the
    /// off-by-one every pattern tool has to settle out loud.
    #[test]
    fn a_linear_pattern_puts_a_body_at_every_step_and_counts_the_original() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(4.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(6.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(20.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        for i in 0..6 {
            let centre = Point3::new(20.0 * i as f64, 0.0, 0.0);
            assert!(
                (field.distance(centre) - -4.0).abs() < 1e-9,
                "copy {i} should be solid at {centre:?}",
            );
        }
        // And no seventh, which is what counting the original prevents.
        assert!(
            field.distance(Point3::new(120.0, 0.0, 0.0)) > 0.0,
            "there is no seventh copy"
        );
        // Nor anything between two of them.
        assert!(field.distance(Point3::new(10.0, 0.0, 0.0)) > 0.0);
    }

    /// A ring of copies, evenly spaced about the axis and the point it passes through.
    #[test]
    fn a_radial_pattern_spaces_its_copies_evenly_about_the_axis() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        // Off the axis, so turning it actually moves it.
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(4.0),
            kind: fieldform_scene::PatternKind::Radial { frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0); 3]) },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        for i in 0..4 {
            let turn = std::f64::consts::TAU * i as f64 / 4.0;
            let centre = Point3::new(30.0 * turn.cos(), 30.0 * turn.sin(), 0.0);
            assert!(
                (field.distance(centre) - -5.0).abs() < 1e-9,
                "copy {i} should be solid at {centre:?}",
            );
        }
        // Between two of them there is nothing.
        let between = std::f64::consts::FRAC_PI_4;
        assert!(field.distance(Point3::new(30.0 * between.cos(), 30.0 * between.sin(), 0.0)) > 0.0);
    }

    /// The turn is about the point the node names, not about the world origin.
    #[test]
    fn a_radial_patterns_origin_decides_what_it_turns_about() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(4.0),
            kind: fieldform_scene::PatternKind::Radial { frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(10.0), lit(0.0), lit(0.0)]) },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();

        // Turned about x = 10, the quarter-turn copy sits at (10, 20) rather than (0, 30).
        assert!((field.distance(Point3::new(10.0, 20.0, 0.0)) - -5.0).abs() < 1e-9);
        assert!(field.distance(Point3::new(0.0, 30.0, 0.0)) > 0.0);
    }

    /// A pattern of a true distance field is one, so it can still be blended.
    ///
    /// Every instance is a rigid motion, which maps distances to distances — a property of the
    /// motions rather than of the union, which is why it holds however many copies there are.
    #[test]
    fn a_pattern_stays_blendable_however_many_copies() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(4.0),
        });
        let row = graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(12.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(6.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let bar = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(33.0), lit(0.0), lit(0.0)]),
            size: [lit(80.0), lit(3.0), lit(3.0)],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: row.into(),
            b: bar.into(),
            fillet: Fillet { radius: lit(1.0), continuity: fieldform_scene::Continuity::G1 },
        });
        assert!(GraphField::compile(&doc, &part_with(graph), 0.05).is_ok());
    }

    /// D8's property, on the union of placements. A rigid motion has nothing to undo, so the
    /// gradient must come through at exactly unit length however many instances there are.
    #[test]
    fn a_patterned_field_still_moves_at_unit_rate() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(25.0), lit(0.0), lit(0.0)]),
            radius: lit(6.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(5.0),
            kind: fieldform_scene::PatternKind::Radial { frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0); 3]) },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
        // A path across the ring, off the seams where two instances tie and the union's
        // gradient genuinely has a corner.
        let points: Vec<Point3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                Point3::new(3.0 + t * 0.9, 1.7 + t * 0.31, 4.0 + t * 0.13)
            })
            .collect();
        assert!(gradient_deviation(&field, &points, 1e-4) < 1e-6);
    }

    /// A count of one is the body, not an empty field.
    #[test]
    fn a_pattern_of_one_is_the_body_itself() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(2.0); 3]),
            radius: lit(4.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(1.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(20.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.05).unwrap();
        assert!((field.distance(Point3::new(2.0, 2.0, 2.0)) - -4.0).abs() < 1e-9);
        assert!(
            field.distance(Point3::new(22.0, 2.0, 2.0)) > 0.0,
            "and no second copy"
        );
    }

    /// The patterned body is consumed, so a pattern draws as one body rather than as the copies
    /// beside the original sitting inside the first of them.
    #[test]
    fn a_patterned_body_stops_being_a_body_of_its_own() {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(4.0),
        });
        let row = graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(3.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(20.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        assert_eq!(graph.roots(), vec![row]);
    }

    /// The extent covers every copy, not only the one the body was drawn at.
    ///
    /// The bound is what empty-space skipping trusts, so a pattern whose extent covered the
    /// first instance alone would have the tracer march straight through the rest.
    #[test]
    fn a_patterns_extent_covers_every_copy() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(4.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: fieldform_scene::ParamRef::Literal(4.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(20.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let (lo, hi) = crate::part_extent(&doc, &graph)
            .as_box()
            .expect("a pattern of spheres is bounded");
        assert!(lo.x <= -4.0 + 1e-9, "the first copy is inside");
        assert!(
            hi.x >= 64.0 - 1e-9,
            "and so is the last, at 60 plus its radius"
        );
    }

    /// D8's property, on the node that is most at risk of failing it.
    ///
    /// A transform is the one composition that touches the *argument* rather than the value,
    /// so it is the one that can silently stop returning a distance — divide the point by the
    /// scale and forget to multiply the result back, and every reading is off by that factor
    /// while the surface stays exactly where it belongs. Nothing downstream notices except
    /// blends, and they notice as a wrong fillet rather than as an error.
    #[test]
    fn a_placed_field_still_moves_at_unit_rate() {
        // A rotation that is not a multiple of 90 degrees about any axis, so an axis-swap
        // mistake cannot pass by symmetry.
        for scale in [0.25, 1.0, 3.5] {
            let field = placed_sphere(10.0, [7.0, -3.0, 2.0], [23.0, -47.0, 11.0], scale).unwrap();
            // Away from the surface and from the centre, where the gradient is genuinely
            // undefined rather than wrong.
            let points: Vec<Point3<f64>> = (0..40)
                .map(|i| {
                    let t = i as f64;
                    Point3::new(20.0 + t, -14.0 + 0.7 * t, 9.0 - 0.3 * t)
                })
                .collect();
            assert!(
                gradient_deviation(&field, &points, 1e-4) < 1e-6,
                "scale {scale} broke the unit-gradient property",
            );
        }
    }

    /// The surface has to land where the placement says, and the *distance* has to scale with
    /// it. Checking only the surface would pass an implementation that divides the argument
    /// and never multiplies the result back.
    #[test]
    fn a_placement_moves_the_surface_and_rescales_the_distance() {
        let field = placed_sphere(10.0, [7.0, -3.0, 2.0], [0.0, 0.0, 0.0], 3.0).unwrap();
        let centre = Point3::new(7.0, -3.0, 2.0);

        // Radius 10 scaled by 3 is a sphere of radius 30 about the translated centre.
        assert!((field.distance(centre) - -30.0).abs() < 1e-9);
        assert!(field.distance(centre + Vector3::new(30.0, 0.0, 0.0)).abs() < 1e-9);
        // And a point 5 mm outside it reads 5, not 5/3 or 15.
        assert!((field.distance(centre + Vector3::new(35.0, 0.0, 0.0)) - 5.0).abs() < 1e-9);
    }

    /// A rotation about the body's own origin must not drag the body sideways: a sphere is
    /// rotationally symmetric, so any movement here is the transform composing in the wrong
    /// order — rotating about the world origin instead of the placed one.
    #[test]
    fn rotating_a_symmetric_body_about_its_own_centre_does_not_move_it() {
        let plain = placed_sphere(10.0, [7.0, -3.0, 2.0], [0.0, 0.0, 0.0], 1.0).unwrap();
        let turned = placed_sphere(10.0, [7.0, -3.0, 2.0], [30.0, 40.0, 50.0], 1.0).unwrap();
        for p in [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(20.0, 5.0, -8.0),
            Point3::new(7.0, -3.0, 2.0),
        ] {
            assert!((plain.distance(p) - turned.distance(p)).abs() < 1e-9);
        }
    }

    /// The reason the scale is uniform: a similarity maps distances to distances, so a placed
    /// body is still a true distance field and a blend will still accept it. Per-axis scale
    /// would forfeit exactly this, and the forfeit would surface at some later blend rather
    /// than at the drag that caused it.
    #[test]
    fn a_placed_body_is_still_a_distance_field_a_blend_will_accept() {
        let (doc, cage_id, _) = document();
        let mut graph = Graph::from_cage(cage_id);
        let cage_node = graph.roots()[0];
        let placed = graph.add(Node::Transform {
            input: cage_node.into(),
            translation: [lit(1.0), lit(0.0), lit(0.0)],
            rotation: [lit(15.0), lit(0.0), lit(0.0)],
            scale: lit(2.0),
            about: fieldform_scene::Place::origin(),
        });
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(3.0), lit(0.0), lit(0.0)]),
            radius: lit(1.0),
        });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: placed.into(),
            b: sphere.into(),
            fillet: Fillet { radius: lit(0.3), continuity: Continuity::G1 },
        });
        assert!(GraphField::compile(&doc, &part_with(graph), 0.05).is_ok());
    }

    /// The interval over a *rotated* box is the case that can quietly go wrong: bounding the
    /// inner field over the original box rather than the mapped one produces a bound that
    /// excludes real geometry, and empty-space skipping would then march straight through a
    /// surface that is there.
    #[test]
    fn a_placed_interval_contains_every_value_it_takes() {
        let field = placed_sphere(10.0, [7.0, -3.0, 2.0], [23.0, -47.0, 11.0], 1.5).unwrap();
        let boxes = [
            (Point3::new(-5.0, -5.0, -5.0), Point3::new(5.0, 5.0, 5.0)),
            (Point3::new(0.0, -10.0, -6.0), Point3::new(20.0, 6.0, 10.0)),
            (Point3::new(40.0, 40.0, 40.0), Point3::new(50.0, 50.0, 50.0)),
        ];
        for (lo, hi) in boxes {
            let (bound_lo, bound_hi) = field.interval(lo, hi);
            let (seen_lo, seen_hi) = observed_range(&field, lo, hi);
            assert!(
                bound_lo <= seen_lo + 1e-9 && bound_hi >= seen_hi - 1e-9,
                "bound [{bound_lo}, {bound_hi}] excludes observed [{seen_lo}, {seen_hi}]",
            );
        }
    }

    /// Zero or negative scale has no shape, so it is named rather than quietly clamped into
    /// one.
    #[test]
    fn a_scale_that_is_not_positive_is_refused_and_says_so() {
        for bad in [0.0, -2.0] {
            let result = placed_sphere(10.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], bad);
            assert!(
                matches!(result, Err(CompileError::NonPositiveScale(_))),
                "a scale of {bad} was not refused",
            );
        }
    }

    /// **A block can be aimed at a line.** A cuboid, a torus and a plane are defined about their
    /// own `+Z` and could be turned only by typing angles, where a cylinder has been pointable at
    /// a line since its ends could name one. Asked for as "all SDF primitives should be able to
    /// be aligned along an axis".
    ///
    /// Measured as what the shape encloses rather than as the angles it came to: a long box is
    /// long in Z until it is aimed, and long in X afterwards, which is the fact somebody aiming
    /// one is after. Comparing the angles would only compare one reading with itself.
    #[test]
    fn a_block_aimed_along_a_line_points_the_way_the_line_runs() {
        // The document's own spine runs from (-50, 0, 0) to (50, 0, 0), so along +X.
        let (doc, _, curve) = document();
        let lit = ParamRef::Literal;
        let long_box = |turn| {
            let mut graph = Graph::empty();
            graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(10.0), lit(10.0), lit(60.0)],
                turn,
            });
            GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles")
        };
        let upright = long_box(fieldform_scene::no_turn());
        let aimed = long_box(fieldform_scene::Turn::Along { curve });

        let up = Point3::new(0.0, 0.0, 25.0);
        let across = Point3::new(25.0, 0.0, 0.0);
        assert!(upright.distance(up) < 0.0, "an unturned box is long in Z");
        assert!(upright.distance(across) > 0.0, "an unturned box is not long in X");
        assert!(aimed.distance(across) < 0.0, "a box aimed along the line is long in X");
        assert!(aimed.distance(up) > 0.0, "and no longer long in Z");
    }

    /// **A block standing where a curve starts moves when that end does.**
    ///
    /// Reported of the plane a pipe's radius ramp reads: its *facing* followed the curve and its
    /// position did not, because a place could name a block and not a curve. "The plane should
    /// be tied to the point of the curve, so when we move the point of the curve the plane also
    /// moves with it."
    #[test]
    fn a_place_at_a_curve_s_start_follows_that_end_when_it_moves() {
        let mut doc = Document::new();
        let curve = CurveId::new();
        doc.curves.insert(
            curve,
            Curve::polyline(
                curve,
                "spine",
                vec![Point3::new(1.0, 2.0, 3.0), Point3::new(40.0, 0.0, 0.0)],
                false,
            )
            .expect("a two-point line"),
        );
        let graph = Graph::empty();
        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        let at = fieldform_scene::Place::StartOf { curve };

        let before = resolve_place(&doc, &graph, &at, &numbers).expect("a curve has a start");
        assert!((before - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-9, "got {before:?}");

        // Drag that end somewhere else, the way the viewport does.
        doc.curve_mut(curve)
            .expect("the curve")
            .set_control_point(0, Point3::new(-7.0, 5.0, 11.0));
        let after = resolve_place(&doc, &graph, &at, &numbers).expect("still a start");
        assert!(
            (after - Point3::new(-7.0, 5.0, 11.0)).norm() < 1e-9,
            "the place did not follow the end it names, got {after:?}",
        );

        // And it is a curve edge, so the curve cannot be deleted out from under it and a paste
        // renumbers it with everything else.
        let plane = {
            let mut g = Graph::empty();
            let id = g.add(Node::Plane {
                origin: at,
                flipped: false,
                turn: fieldform_scene::Turn::Tangent { curve },
            });
            g.node(id).expect("the plane").curves()
        };
        assert_eq!(plane, vec![curve, curve], "both the place and the facing name it");
    }

    /// **A field shortened against a region answers the same as the field it came from.**
    ///
    /// Backlog item 17. Pruning is only worth anything if it is invisible: the whole point is to
    /// skip work that could not have changed the answer, so any difference inside the region is a
    /// branch deleted that should have been kept. Measured against the full field at points
    /// spread through the box rather than argued from the interval arithmetic, because the
    /// arithmetic is what is being tested.
    #[test]
    fn a_field_pruned_to_a_region_answers_exactly_as_the_whole_one_does() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        // Forty balls scattered through a metre, unioned -- the wide `min` tree a pattern, a
        // scatter or a part of many bodies all produce.
        let mut bodies = Vec::new();
        for i in 0..40 {
            let f = f64::from(i);
            let at = |seed: f64| ((f * seed).fract() - 0.5) * 1000.0;
            bodies.push(graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([
                    lit(at(0.754_877_666_2)),
                    lit(at(0.569_840_290_9)),
                    lit(at(0.430_159_709_1)),
                ]),
                radius: lit(20.0),
            }).into());
        }
        graph.add(Node::Boolean { op: fieldform_scene::BooleanOp::Union, inputs: bodies });
        let field = GraphField::compile_graph(&Document::new(), &graph, 1e-3).expect("compiles");

        let mut shortened = 0usize;
        let mut empty = 0usize;
        let mut regions = 0usize;
        let grid = 6;
        let step = 1000.0 / f64::from(grid);
        for i in 0..grid {
            for j in 0..grid {
                for k in 0..grid {
                    regions += 1;
                    let lo = Point3::new(
                        -500.0 + f64::from(i) * step,
                        -500.0 + f64::from(j) * step,
                        -500.0 + f64::from(k) * step,
                    );
                    let hi = Point3::new(lo.x + step, lo.y + step, lo.z + step);
                    let region = field.for_region(lo, hi);
                    let Some(pruned) = region.field() else {
                        empty += 1;
                        continue;
                    };
                    if pruned.op_count() < field.op_count() {
                        shortened += 1;
                    }
                    // The same answer, everywhere inside the box it was pruned against.
                    for n in 0..40 {
                        let t = f64::from(n) / 39.0;
                        let p = Point3::new(
                            lo.x + t * step,
                            lo.y + ((t * 7.0).fract()) * step,
                            lo.z + ((t * 13.0).fract()) * step,
                        );
                        let (whole, part) = (field.distance(p), pruned.distance(p));
                        assert!(
                            (whole - part).abs() < 1e-9,
                            "pruned {part} against {whole} at {p:?}, which is a branch deleted                              that could still have been the answer",
                        );
                    }
                }
            }
        }
        assert!(empty > regions / 2, "most of a metre is empty; only {empty} of {regions} were");
        assert!(shortened > 0, "nothing was shortened, so nothing was being tested");
    }

    /// **Across a curve is not along it**, and for anything but a straight line the two differ.
    ///
    /// Reported of a pipe on a degree-five spine: its radius ramp read a plane at the curve's
    /// start, and that plane came out tilted by an angle nobody had typed. It was facing along
    /// the *chord* -- `Frame::Along`, which is the right reading for "where is this axis" and
    /// the wrong one for "which way is this curve going here". "It should just take the normal
    /// of whatever the curve is doing."
    #[test]
    fn across_a_curve_reads_its_tangent_and_along_it_reads_its_chord() {
        let mut doc = Document::new();
        let curve = CurveId::new();
        // Sets off along +X and turns hard, so the chord and the starting tangent disagree by a
        // wide angle -- the case that was reported, and the one a straight line hides.
        doc.curves.insert(
            curve,
            Curve::polyline(
                curve,
                "bend",
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(50.0, 0.0, 0.0),
                    Point3::new(50.0, 50.0, 0.0),
                ],
                false,
            )
            .expect("a three-point line"),
        );
        let graph = Graph::empty();
        let numbers = |r: &ParamRef| r.resolve(&|_| None);

        let across =
            resolve_frame(&doc, &graph, &fieldform_scene::Frame::Tangent { curve }, &numbers)
                .expect("a curve has a tangent at its start");
        let along =
            resolve_frame(&doc, &graph, &fieldform_scene::Frame::Along { curve }, &numbers)
                .expect("and a chord");

        // Both start where the curve starts.
        assert!(across.0.norm() < 1e-9 && along.0.norm() < 1e-9);

        // Across follows the way it sets off, which is +X.
        let tangent = across.1 * Vector3::z();
        assert!(
            (tangent - Vector3::x()).norm() < 1e-6,
            "across the curve should face +X, got {tangent:?}",
        );
        // Along follows the chord, which here is a long way from that.
        let chord = along.1 * Vector3::z();
        let apart = tangent.dot(&chord).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            apart > 30.0,
            "the two readings should differ on a bent curve, and differ by {apart:.1} degrees",
        );

        // A turn reads the same thing through the same arm.
        let spun =
            turn_rotation(&doc, &graph, &fieldform_scene::Turn::Tangent { curve }, &numbers)
                * Vector3::z();
        assert!((spun - tangent).norm() < 1e-9, "a turn and a frame disagree");
    }

    /// **A pipe's radius can vary along it, and the field still never overestimates.**
    ///
    /// Backlog item 2. The shape is the one the item asked for -- thin at one end, fat at the
    /// other -- and the property is the one the contract asks for: a varying radius makes
    /// `d(p) - R(p)` rise faster than a distance may, so it is divided by `1 + L` exactly as a
    /// driven offset is. Measured rather than argued, since a bound that is wrong shows up as a
    /// tracer stepping through a surface rather than as a number anybody reads.
    #[test]
    fn a_pipe_of_a_varying_radius_is_that_radius_and_never_claims_more_room_than_it_has() {
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let curve = CurveId::new();
        doc.curves.insert(
            curve,
            Curve::polyline(
                curve,
                "spine",
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(100.0, 0.0, 0.0)],
                false,
            )
            .expect("a two-point line"),
        );

        let mut graph = Graph::empty();
        // A plane at the origin facing +X, so distance to it is distance along the pipe.
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::By([lit(0.0), lit(90.0), lit(0.0)]),
        });
        // Two at one end, ten at the other.
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(0.0),
            in_max: lit(100.0),
            out_min: lit(2.0),
            out_max: lit(10.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        let pipe = graph.add(Node::Pipe { curve, radius: ParamRef::Field(ramp) });
        assert_eq!(graph.roots(), vec![pipe], "the ramp and its plane are not bodies");

        let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");

        // The radius at each end, found by walking out from the axis until the sign changes.
        let radius_at = |x: f64| {
            let mut r = 0.0;
            while r < 40.0 {
                if field.distance(Point3::new(x, 0.0, r)) > 0.0 {
                    return r;
                }
                r += 0.01;
            }
            f64::NAN
        };
        let (near, far) = (radius_at(5.0), radius_at(95.0));
        assert!((near - 2.4).abs() < 0.1, "near the start the radius is {near:.2}, wanted ~2.4");
        assert!((far - 9.6).abs() < 0.1, "near the end the radius is {far:.2}, wanted ~9.6");

        // And the contract, which is what the divisor is for: walked past the claim, in every
        // direction, from points all round the pipe.
        let mut points = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                points.push(Point3::new(
                    -10.0 + 10.0 * f64::from(i),
                    -20.0 + 4.0 * f64::from(j),
                    -20.0 + 3.5 * f64::from(j),
                ));
            }
        }
        let directions = [
            Vector3::x(),
            -Vector3::x(),
            Vector3::y(),
            -Vector3::y(),
            Vector3::z(),
            -Vector3::z(),
        ];
        let over = crate::overstatement(&field, &points, &directions, 0.05, 1.2);
        assert!(over < 1e-9, "the field claimed {over:.4} mm of room it has not got");
    }

    /// **A direction between two points aims a frame and a turn alike** -- backlog items 25 and
    /// 18, which both asked for exactly this and both parked on settling the reference once
    /// rather than growing a narrower copy per block.
    ///
    /// So it is settled once, as `Frame::Between`, and a `Turn` reads it through that same arm.
    /// Measured as what the blocks come out pointing along, since that is the fact somebody
    /// moving a point is after.
    #[test]
    fn a_direction_between_two_points_aims_a_frame_and_a_turn_alike() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        // Two points, and the direction between them is +X.
        let a = graph.add(Node::Point { at: [lit(0.0), lit(0.0), lit(0.0)] });
        let b = graph.add(Node::Point { at: [lit(10.0), lit(0.0), lit(0.0)] });

        let doc = Document::new();
        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        let between = fieldform_scene::Frame::Between { from: a, to: b };
        let (origin, rotation) =
            resolve_frame(&doc, &graph, &between, &numbers).expect("two points make a frame");
        assert!(origin.norm() < 1e-9, "it sits at the first of the two");
        let aim = rotation * Vector3::z();
        assert!((aim - Vector3::x()).norm() < 1e-9, "and points at the second, got {aim:?}");

        // A turn reading the same two comes to the same rotation, because it reads the same arm.
        let turn = fieldform_scene::Turn::Between { from: a, to: b };
        let spun = turn_rotation(&doc, &graph, &turn, &numbers) * Vector3::z();
        assert!((spun - aim).norm() < 1e-9, "a turn and a frame disagree: {spun:?} vs {aim:?}");

        // Both points are read as inputs, so neither is a body of its own and a paste or a
        // delete can find them.
        let ring = graph.add(Node::Torus {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            major_radius: lit(30.0),
            minor_radius: lit(5.0),
            turn,
        });
        let mut named = graph.node(ring).expect("the torus").inputs();
        named.sort_by_key(|id| id.raw());
        let mut want = vec![a, b];
        want.sort_by_key(|id| id.raw());
        assert_eq!(named, want, "a facing between two points carries both edges");
        assert_eq!(graph.roots(), vec![ring], "and neither point is a body");

        // Ends in the same place have no direction between them, and say `+Z` rather than
        // dividing by nothing -- the state you pass through while dragging one of them.
        let stuck = fieldform_scene::Frame::Between { from: a, to: a };
        let (_, flat) = resolve_frame(&doc, &graph, &stuck, &numbers).expect("still answers");
        assert!((flat * Vector3::z() - Vector3::z()).norm() < 1e-9);
    }

    /// **A block that is aimed at another is not inside it.**
    ///
    /// A reference makes its source an input, so a walk that composes placements down through
    /// every input puts the plane a cuboid is aimed at *inside* the cuboid's own placement --
    /// and the cuboid's placement is, in this case, the plane's own rotation read back through
    /// the aim. So the plane came out turned twice. Reported as "I rotate the plane by 45
    /// degrees, it looks like it was rotated by 90, but the cube is rotated 45": the cube was
    /// right, and the square drawn for the plane was not.
    #[test]
    fn a_block_aimed_at_a_plane_does_not_place_that_plane_inside_itself() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::By([lit(45.0), lit(0.0), lit(0.0)]),
        });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(10.0); 3],
            turn: fieldform_scene::Turn::From { node: plane },
        });

        // The cuboid's own turn, expanded into the placement it means.
        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        let flat = with_turns_flattened(&Document::new(), &graph, &numbers);

        // Where the plane sits, asked of the flattened graph the way the panel asks it. The
        // answer includes the plane's *own* turn -- that is what a placement walk is for -- and
        // must not include the cuboid's, which is the same 45 degrees read back through the aim.
        let above = placement_above(&Document::new(), &flat, plane, None, &numbers);
        let (x, _, _) = above.isometry.rotation.to_rotation_matrix().euler_angles();
        assert!(
            (x.to_degrees() - 45.0).abs() < 1e-6,
            "the plane came out turned by {:.1} degrees, and its own turn is 45",
            x.to_degrees(),
        );

        // And the cuboid is turned by the 45 it read, which is the half that was always right.
        let (bx, _, _) = placement_above(&Document::new(), &flat, block, None, &numbers)
            .isometry
            .rotation
            .to_rotation_matrix()
            .euler_angles();
        assert!((bx.to_degrees() - 45.0).abs() < 1e-6, "the block reads {:.1}", bx.to_degrees());
    }

    /// **A pivot taken through a placement does not walk back into itself.**
    ///
    /// `placement_similarity` resolves its pivot, a pivot resolves a place, and a place taken
    /// from a placement resolves *that* placement -- three functions in a ring. Each carries a
    /// depth now, and the one thing that matters is that the depth is carried *through* the
    /// ring: a hand-off that starts a fresh count makes the bound unreachable, which is how a
    /// guarded walk still overflowed the stack.
    #[test]
    fn a_pivot_taken_through_a_placement_does_not_walk_back_into_itself() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::origin(),
            radius: lit(5.0),
        });
        let a = graph.add(Node::Transform {
            input: ball.into(),
            translation: [lit(1.0), lit(0.0), lit(0.0)],
            rotation: [lit(0.0); 3],
            scale: lit(1.0),
            about: fieldform_scene::Place::origin(),
        });
        let b = graph.add(Node::Transform {
            input: a.into(),
            translation: [lit(0.0), lit(2.0), lit(0.0)],
            rotation: [lit(0.0); 3],
            scale: lit(1.0),
            // Its pivot is taken from the placement below it, and that one's from this.
            about: fieldform_scene::Place::From { node: a },
        });
        if let Some(Node::Transform { about, .. }) = graph.node_mut(a) {
            *about = fieldform_scene::Place::From { node: b };
        }

        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        // The assertion is that these return. `resolve_place` enters the ring at the place end
        // and a compile enters it at the placement end, and the two used to be guarded to
        // different degrees -- the place chain counted its own hops while the hand-off back into
        // `placement_similarity` began again at nought.
        let at = resolve_place(&Document::new(), &graph, &fieldform_scene::Place::From { node: b }, &numbers);
        assert!(at.is_ok() || at.is_err(), "either answer will do; coming back is the point");
        let built = GraphField::compile_graph(&Document::new(), &graph, 1e-3);
        assert!(built.is_ok() || built.is_err(), "and the same from the other end");
    }

    /// **A facing that leads back to itself is answered, not followed for ever.**
    ///
    /// Built directly, because the panel refuses to offer a source that leads back -- and this is
    /// about what happens when the graph says it anyway. That is the state a document passes
    /// through while it is being edited, and the panel asks these questions every frame on
    /// whatever the graph currently says.
    ///
    /// The failure this guards against cannot be caught by `should_panic`: running out of stack
    /// aborts the process without unwinding, so no hook runs and no log is written. The test is
    /// simply that this returns.
    #[test]
    fn a_facing_that_leads_back_to_itself_is_answered_rather_than_followed_for_ever() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let a = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::By([lit(0.0); 3]),
        });
        let b = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::From { node: a },
        });
        if let Some(Node::Plane { turn, .. }) = graph.node_mut(a) {
            *turn = fieldform_scene::Turn::From { node: b };
        }

        let doc = Document::new();
        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        // **The assertion is that this returns.** What it returns once the bound is reached is
        // the identity, which is what every reader of a facing already falls back to when one
        // cannot be read -- a block left pointing where it was pointing, rather than an error
        // raised in the middle of a frame. The only wrong answer here is no answer.
        let (origin, rotation) =
            resolve_frame(&doc, &graph, &fieldform_scene::Frame::From { node: a }, &numbers)
                .expect("a place is still readable, whatever the facing says");
        assert!(origin.iter().all(|v| v.is_finite()), "and it is a place, not a NaN");
        assert!(
            (rotation.matrix().determinant() - 1.0).abs() < 1e-9,
            "and a rotation, not whatever fell out of a walk that gave up",
        );
        // And the other half of the pair: a turn reading a frame reading that turn.
        let turn = fieldform_scene::Turn::From { node: a };
        let spin = turn_rotation(&doc, &graph, &turn, &numbers);
        assert!((spin.matrix().determinant() - 1.0).abs() < 1e-9);
    }

    /// **A block defined between two ends can stand on a plane**, which is the third way of
    /// saying where its axis is and the one a cylinder could not use: its ends could name a
    /// curve, and nothing else. Asked for as "if you create a cylinder -- or any object that
    /// requires an axis -- you should be able to provide as input under Axis a plane, and that
    /// object should re-orient itself along it".
    ///
    /// Measured as where the solid is rather than as the numbers it stores: the cylinder must
    /// enclose a point along the plane's normal from its origin, and not the same distance along
    /// the world's up, which is where it sat before it was stood on anything.
    #[test]
    fn a_cylinder_standing_on_a_plane_runs_along_that_plane_s_normal() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        // Facing +X: a quarter turn about Y takes +Z there.
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::at(5.0, 0.0, 0.0),
            flipped: false,
            turn: fieldform_scene::Turn::By([lit(0.0), lit(90.0), lit(0.0)]),
        });
        let rod = graph.add(Node::Cylinder { edge: Default::default(),
            ends: fieldform_scene::Ends::On { node: plane, length: lit(40.0) },
            radius: lit(3.0),
        });
        assert_eq!(graph.roots(), vec![rod], "the plane is an operand, not a body");

        let field = GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles");
        assert!(
            field.distance(Point3::new(25.0, 0.0, 0.0)) < 0.0,
            "the rod should run along +X from the plane's origin",
        );
        assert!(
            field.distance(Point3::new(5.0, 0.0, 20.0)) > 0.0,
            "and no longer along the world's up, which is where it was before",
        );
    }

    /// A facing taken from another block is read **once**, before and after flattening.
    ///
    /// **Reported as a mirror landing somewhere nobody asked for.** Flattening a turn writes its
    /// rotation into a placement, and used to leave the whole turn on the block inside as well —
    /// because a turn is a graph edge and dropping it makes what it names a body again. But this
    /// function reads a turn, so a plane aimed at another was turned once for the picture and
    /// twice for anything aimed at it: a square drawn at 40 degrees and a mirror reflecting
    /// across 80. Nothing could see it, because both compilers read the flattened graph and so
    /// both were wrong together, which is exactly what a CPU-against-GPU residual cannot catch.
    ///
    /// [`fieldform_scene::Turn::Absorbed`] is the edge without the rotation.
    /// `aimed_turn_probe` prints the three numbers.
    #[test]
    fn a_facing_taken_from_another_block_is_read_once_after_flattening() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let mut graph = Graph::empty();
        let reference = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::By([lit(0.0), lit(40.0), lit(0.0)]),
        });
        let aimed = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::From { node: reference },
        });
        // Consumed, so the aimed plane is an operand rather than a body of its own.
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(50.0), lit(0.0)]),
            radius: lit(10.0),
        });
        let mirror = graph.add(Node::Mirror {
            input: ball.into(),
            frame: fieldform_scene::Frame::From { node: aimed },
            join: true,
        });

        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        let frame = fieldform_scene::Frame::From { node: aimed };
        let pitch = |g: &Graph| {
            let (_, rotation) = resolve_frame(&doc, g, &frame, &numbers).expect("resolves");
            ((rotation * Vector3::z()).z.clamp(-1.0, 1.0)).acos().to_degrees()
        };

        let written = pitch(&graph);
        assert!(
            (written - 40.0).abs() < 1e-9,
            "aimed at a plane pitched 40 degrees, the graph as written reads {written:.2}"
        );
        let flat = with_turns_flattened(&doc, &graph, &numbers);
        let lowered = pitch(&flat);
        assert!(
            (lowered - written).abs() < 1e-9,
            "flattening turned it again: {written:.2} degrees as written, {lowered:.2} after"
        );

        // And the half that made the whole turn worth keeping: what it names is still an
        // operand, so it is neither drawn as a body nor dropped out of the graph that draws it.
        assert!(
            flat.node(reference).is_some(),
            "the plane it is aimed at has to survive the lowering -- the viewport draws its \
             square from this graph"
        );
        assert!(
            !flat.roots().contains(&reference),
            "and must not have become a body, which is a half-space across the viewport"
        );
        assert_eq!(flat.roots(), vec![mirror], "one body before, the same one after");
    }

    /// A plane is aimed at where it is drawn.
    ///
    /// Its own field reads its origin with `resolve_place_for`, which brings a place that names
    /// another block back into the frame this one is written in; a frame aimed at it used to
    /// read the same origin with `resolve_place_at`, which does only half of that, and then the
    /// placement applied the other half a second time. So a plane standing on a point, with a
    /// move anywhere above either of them, was drawn in one place and mirrored about another —
    /// by exactly the move, 10 mm each way in `plane_origin_probe`.
    ///
    /// Measured by asking the plane's *own field* what the distance is at the point a frame
    /// names. A plane passes through its own origin, so the answer is zero or it is the gap.
    #[test]
    fn a_plane_is_aimed_at_where_it_is_drawn() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        let moved = |g: &mut Graph, input, up: f64| {
            g.add(Node::Transform {
                input,
                translation: [lit(0.0), lit(0.0), lit(up)],
                rotation: [lit(0.0); 3],
                scale: lit(1.0),
                about: fieldform_scene::Place::origin(),
            })
        };

        // Each arrangement names the plane a frame should find. The two that carry a move over a
        // referenced place are the ones that used to disagree.
        let arrangements: Vec<(&str, Box<dyn Fn(&mut Graph) -> NodeId>)> = vec![
            (
                "a plane standing on a point",
                Box::new(move |g: &mut Graph| {
                    let pt = g.add(Node::Point { at: [lit(0.0), lit(0.0), lit(7.0)] });
                    g.add(Node::Plane {
                        origin: fieldform_scene::Place::From { node: pt },
                        flipped: false,
                        turn: fieldform_scene::no_turn(),
                    })
                }),
            ),
            (
                "a plane on a point, the plane then moved up 10",
                Box::new(move |g: &mut Graph| {
                    let pt = g.add(Node::Point { at: [lit(0.0), lit(0.0), lit(7.0)] });
                    let pl = g.add(Node::Plane {
                        origin: fieldform_scene::Place::From { node: pt },
                        flipped: false,
                        turn: fieldform_scene::no_turn(),
                    });
                    moved(g, pl.into(), 10.0)
                }),
            ),
            (
                "a plane on a point that has itself been moved up 10",
                Box::new(move |g: &mut Graph| {
                    let pt = g.add(Node::Point { at: [lit(0.0), lit(0.0), lit(7.0)] });
                    let raised = moved(g, pt.into(), 10.0);
                    g.add(Node::Plane {
                        origin: fieldform_scene::Place::From { node: raised },
                        flipped: false,
                        turn: fieldform_scene::no_turn(),
                    })
                }),
            ),
            (
                "a plane on typed numbers, then moved up 10",
                Box::new(move |g: &mut Graph| {
                    let pl = g.add(Node::Plane {
                        origin: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(7.0)]),
                        flipped: false,
                        turn: fieldform_scene::no_turn(),
                    });
                    moved(g, pl.into(), 10.0)
                }),
            ),
        ];

        for (label, build) in arrangements {
            let mut graph = Graph::empty();
            let plane = build(&mut graph);
            let field = GraphField::compile_graph(&doc, &graph, 0.5).expect("the plane compiles");
            let flat = with_turns_flattened(&doc, &graph, &numbers);
            let frame = fieldform_scene::Frame::From { node: plane };
            let (at, _) = resolve_frame(&doc, &flat, &frame, &numbers).expect("resolves");
            let gap = field.distance(Point3::from(at));
            assert!(
                gap.abs() < 1e-9,
                "{label}: a frame names z = {:.3}, and the plane's own field reads {gap:.3} mm \
                 there -- so the plane drawn and the plane aimed at are {gap:.3} mm apart",
                at.z,
            );
        }
    }

    /// A cached compile answers exactly what an uncached one would, whatever it is asked before.
    ///
    /// **The one property a cache has to have, tested as the property rather than as a list of
    /// the ways it could be broken.** A baked profile is 98% of a sweep's compile
    /// (`examples/sweep_compile_cost.rs`), so it is worth keeping; and a bake kept for the wrong
    /// profile is a shape on screen the document does not describe, with nothing to report it.
    /// That is not hypothetical here — three caches keyed by id alone served stale geometry after
    /// an in-place edit, which is what `Profile::fingerprint` and this test exist to stop.
    ///
    /// So: one cache, a sequence of documents pushed through it in order, and every one of them
    /// also compiled fresh. The two have to agree at every sample. The sequence carries the cases
    /// five readers set to break the key came up with:
    ///
    /// - the same graph twice, which must hit;
    /// - the *spine* moved, which must hit, because a path is not a profile;
    /// - the *profile* moved in place under its own id, which must miss;
    /// - two revolves off one sketch about different axes, which never flatten a loop into a
    ///   plane at all and so have one loop and two profiles;
    /// - an extrude and a revolve off one sketch, which have equal loops, equal tolerances and
    ///   equal `ignore` arguments, and differ only in what the profile becomes.
    #[test]
    fn a_cached_compile_answers_what_an_uncached_one_would() {
        let lit = ParamRef::Literal;
        // A square sketch in the XZ plane, off to one side so a revolve about Z has a bore.
        let sketch = |shift: f64| {
            let mut doc = Document::new();
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(
                    id,
                    "section",
                    vec![
                        Point3::new(40.0 + shift, 0.0, -10.0),
                        Point3::new(60.0 + shift, 0.0, -10.0),
                        Point3::new(60.0 + shift, 0.0, 10.0),
                        Point3::new(40.0 + shift, 0.0, 10.0),
                    ],
                    true,
                )
                .expect("a closed square"),
            );
            (doc, id)
        };
        let spine_along = |doc: &mut Document, tip: Point3<f64>| {
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(
                    id,
                    "spine",
                    vec![Point3::new(50.0, 0.0, 0.0), tip],
                    false,
                )
                .expect("a line"),
            );
            id
        };
        let revolve_about = |x: f64| {
            fieldform_scene::Frame::At {
                origin: [lit(x), lit(0.0), lit(0.0)],
                axis: [lit(0.0), lit(0.0), lit(1.0)],
            }
        };

        let mut documents: Vec<(&str, Document, Graph)> = Vec::new();
        for (label, tip) in [
            ("a sweep", Point3::new(50.0, 90.0, 0.0)),
            ("the same sweep again", Point3::new(50.0, 90.0, 0.0)),
            ("the same sweep, path dragged", Point3::new(50.0, 140.0, 30.0)),
        ] {
            let (mut doc, profile) = sketch(0.0);
            let spine = spine_along(&mut doc, tip);
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
            documents.push((label, doc, graph));
        }
        {
            // The profile itself moved, under its own id: the edit a cache keyed by id misses.
            let (mut doc, profile) = sketch(25.0);
            let spine = spine_along(&mut doc, Point3::new(50.0, 90.0, 0.0));
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
            documents.push(("the same sweep, profile moved", doc, graph));
        }
        {
            // Two revolves off one sketch about different axes -- one loop, two profiles.
            let (doc, profile) = sketch(0.0);
            let mut graph = Graph::empty();
            graph.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), frame: revolve_about(0.0), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
            graph.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), frame: revolve_about(30.0), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
            documents.push(("two revolves, two axes", doc, graph));
        }
        {
            // An extrude and a revolve off one sketch: equal loops, equal everything but what
            // the profile becomes.
            let (doc, profile) = sketch(0.0);
            let mut graph = Graph::empty();
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(profile),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(20.0),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            graph.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), frame: revolve_about(0.0), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
            documents.push(("an extrude and a revolve", doc, graph));
        }

        let mut cache = LeafCache::default();
        for (label, doc, graph) in &documents {
            let cached = GraphField::compile_cached(doc, graph, 0.5, &mut cache)
                .expect("compiles with a cache");
            let fresh = GraphField::compile_graph(doc, graph, 0.5).expect("compiles without one");
            // Across the whole region the bodies live in, inside and out.
            let mut worst = 0.0f64;
            let mut at = Point3::origin();
            for i in 0..13 {
                for j in 0..13 {
                    for k in 0..13 {
                        let f = |n: usize| -100.0 + 200.0 * (n as f64 / 12.0);
                        let p = Point3::new(f(i), f(j), f(k));
                        let d = (cached.distance(p) - fresh.distance(p)).abs();
                        if d > worst {
                            worst = d;
                            at = p;
                        }
                    }
                }
            }
            assert!(
                worst < 1e-9,
                "{label}: the cache answered {worst} mm away from a fresh compile at {at:?} -- \
                 a bake was kept for a profile it does not belong to"
            );
        }
    }

    /// Keeping a bake is worth doing, and the drag is the reason.
    ///
    /// Not a timing test -- those belong in an example -- but a count: the same profile asked for
    /// twice is one grid, and a profile that has moved is two. Without this the property above
    /// would still pass with a cache that never hit anything.
    #[test]
    fn a_profile_is_baked_once_while_only_the_path_moves() {
        let lit = ParamRef::Literal;
        let profile = fieldform_scene::CurveId::new();
        let build = |tip: Point3<f64>, shift: f64| {
            let mut doc = Document::new();
            doc.curves.insert(
                profile,
                fieldform_scene::Curve::polyline(
                    profile,
                    "section",
                    vec![
                        Point3::new(40.0 + shift, 0.0, -10.0),
                        Point3::new(60.0 + shift, 0.0, -10.0),
                        Point3::new(60.0 + shift, 0.0, 10.0),
                        Point3::new(40.0 + shift, 0.0, 10.0),
                    ],
                    true,
                )
                .expect("a closed square"),
            );
            let spine = fieldform_scene::CurveId::new();
            doc.curves.insert(
                spine,
                fieldform_scene::Curve::polyline(
                    spine,
                    "spine",
                    vec![Point3::new(50.0, 0.0, 0.0), tip],
                    false,
                )
                .expect("a line"),
            );
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
            (doc, graph)
        };
        let _ = lit;

        let mut cache = LeafCache::default();
        let (doc, graph) = build(Point3::new(50.0, 90.0, 0.0), 0.0);
        GraphField::compile_cached(&doc, &graph, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 1, "one profile, one grid");

        // The path moved: a new spine, the same profile, so no new grid.
        let (doc, graph) = build(Point3::new(50.0, 140.0, 30.0), 0.0);
        GraphField::compile_cached(&doc, &graph, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 1, "dragging the path must not rebake the profile");

        // The profile moved, under its own id, while the path stayed: a different grid, since
        // the section is flattened about the point the path sets off from and has moved
        // relative to it. (While the section was flattened about its own centroid this was the
        // same grid, and the line below read one for that reason rather than this one.) The
        // old grid is let go of on the way in to the compile that follows, which does not ask
        // for it.
        let (doc, graph) = build(Point3::new(50.0, 140.0, 30.0), 25.0);
        GraphField::compile_cached(&doc, &graph, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 2, "a moved profile is a new grid beside the old");
        GraphField::compile_cached(&doc, &graph, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 1, "and the old one goes on the next compile");
    }

    /// **What a block is aimed at is consumed by it, before and after the turn is flattened.**
    ///
    /// A turn that names a plane is a graph edge, so that plane stops being a body of its own —
    /// the rule a boolean's operands, a frame's source and a place's all follow. The part that
    /// had to be arranged for is the second half: rewriting the turn into the placement it means
    /// must not drop the edge, or the plane becomes a root again and a root plane is a half-space
    /// drawn across the whole viewport. `with_turns_expanded_using` leaves the reference on the
    /// block inside for exactly this reason.
    #[test]
    fn a_plane_a_block_is_aimed_at_is_not_a_body_before_or_after_flattening() {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::By([lit(0.0), lit(90.0), lit(0.0)]),
        });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(10.0), lit(10.0), lit(60.0)],
            turn: fieldform_scene::Turn::From { node: plane },
        });
        assert_eq!(graph.roots(), vec![block], "the plane is an operand, not a body");

        let doc = Document::new();
        let numbers = |r: &ParamRef| r.resolve(&|_| None);
        let flat = with_turns_flattened(&doc, &graph, &numbers);
        assert!(
            !flat.roots().contains(&plane),
            "flattening the turn brought the plane back as a body, which draws a half-space",
        );
        assert_eq!(flat.roots(), vec![block], "one body before, the same one body after");
    }

    #[test]
    fn a_missing_curve_is_an_error_not_a_panic() {
        let (doc, _, _) = document();
        let mut graph = Graph::empty();
        let _pipe = graph.add(Node::Pipe {
            curve: CurveId::new(),
            radius: ParamRef::Literal(1.0),
        });
        assert!(matches!(
            GraphField::compile(&doc, &part_with(graph), 0.01),
            Err(CompileError::MissingCurve(_))
        ));
    }

    /// **A body that needs an empty slot filled is refused by name, not panicked on.**
    ///
    /// It used to panic, at `slot[i]` in `compile` -- a `HashMap` index on a key the ramp skip had
    /// deliberately left out -- while the comment beside that skip claimed leaving it out was
    /// "what makes a consumer that points at it fail by name rather than by panicking on a missing
    /// index". No `CompileError` named the case, so there was nothing for it to fail by. The guard
    /// that first stopped the panic asked every node for an input absent from `slot`;
    /// `Graph::incomplete` supersedes it by being closed downstream, so the index sites carry no
    /// guard at all and this test is what holds the panic gone.
    ///
    /// **A boolean rather than a shell, which is the whole point of the case.** A shell is refused
    /// first by `ShellNeedsBounds`, so it looks safe and is only safe by accident; a boolean
    /// indexes `slot` directly with nothing between but an `is_empty` guard. Reachable from a
    /// loaded file, and from `boolean in=<name>` on the console, which resolves a name without
    /// asking whether it is a body.
    ///
    /// **And the refusal names the ramp, not the boolean that met it.** The hole is a block below
    /// the body, which is the one a person can do something about -- see `empty_slot`.
    #[test]
    fn a_body_with_an_empty_slot_is_refused_by_name_rather_than_panicking() {
        let (doc, _, _) = document();
        let mut graph = Graph::empty();
        let ramp = graph.add(Node::Ramp {
            input: fieldform_scene::Operand::empty(),
            shape: fieldform_scene::RampShape::Linear,
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(1.0),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(1.0),
        });
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ]),
            radius: ParamRef::Literal(10.0),
        });
        // Boolean indexes `slot` directly, with only an is_empty guard before it.
        let _union = graph.add(Node::Boolean {
            op: fieldform_scene::BooleanOp::Union,
            inputs: vec![sphere.into(), ramp.into()],
        });
        assert!(
            matches!(
                GraphField::compile(&doc, &part_with(graph), 0.01),
                Err(CompileError::EmptySlot { node, slot }) if node == ramp && slot == 0
            ),
            "a body reading an unwired ramp is refused, naming the ramp and its slot",
        );
    }

    /// **A ramp with nothing to read is not an error on its own.** It is an error only when
    /// something reads it, which the test above holds.
    ///
    /// The state a ramp has been in since `ramp-decision.md` 7.2 and every slot can be in since
    /// `notebook-spec.md` 7.2: placed, and not yet told what to read. A ramp is never a body
    /// (`Node::is_body`), so nothing draws it and nothing needs it to evaluate -- and the skip
    /// that leaves it out of the op list must not take the part's real body with it.
    ///
    /// Asserted on the *value*, not merely on `Ok`: a compile that quietly folded the ramp in
    /// would still return a field, and the sphere's own distance is what says it did not.
    #[test]
    fn an_unwired_ramp_still_compiles_to_nothing() {
        let (doc, _, _) = document();
        let mut graph = Graph::empty();
        let _ramp = graph.add(Node::Ramp {
            input: fieldform_scene::Operand::empty(),
            shape: fieldform_scene::RampShape::Linear,
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(1.0),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(1.0),
        });
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ]),
            radius: ParamRef::Literal(10.0),
        });
        let field = GraphField::compile(&doc, &part_with(graph), 0.01)
            .expect("a ramp nothing reads is a legal part, and the sphere is its body");
        assert!(
            (field.distance(Point3::origin()) + 10.0).abs() < 1e-9,
            "the part is its sphere: -r at the centre, with the ramp contributing nothing"
        );
        assert_eq!(
            field.body_at(Point3::new(10.0, 0.0, 0.0), 0.01),
            Some(sphere),
            "and the surface belongs to the sphere, so the ramp was left out rather than folded in"
        );
    }

    /// The extent walk, on the case the three-state answer exists for.
    mod extents {
        use super::*;
        use crate::Extent;
        use fieldform_scene::{Document, Graph, Node, ParamRef};

        fn at(x: f64, y: f64, z: f64) -> [ParamRef; 3] {
            [
                ParamRef::Literal(x),
                ParamRef::Literal(y),
                ParamRef::Literal(z),
            ]
        }

        fn unit_cube(graph: &mut Graph) -> fieldform_scene::NodeId {
            graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ]),
                size: [ParamRef::Literal(2.0); 3],
                turn: fieldform_scene::no_turn(),
            })
        }

        #[test]
        fn a_plane_is_unbounded_and_a_point_bounds_to_itself() {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At(at(0.0, 0.0, 5.0)),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let point = graph.add(Node::Point {
                at: at(1.0, 2.0, 3.0),
            });
            assert_eq!(crate::graph_extent(&doc, &graph, plane), Extent::Unbounded);
            assert_eq!(
                crate::graph_extent(&doc, &graph, point),
                Extent::at(Point3::new(1.0, 2.0, 3.0)),
            );
        }

        /// The reason a half-space is worth having at all: cut a finite part with it and the
        /// result is finite. Under the old `Option` this could not be said, because the
        /// plane's "no box" was indistinguishable from "contributes nothing".
        #[test]
        fn a_part_cut_by_an_infinite_plane_is_still_bounded() {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let cube = unit_cube(&mut graph);
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At(at(0.0, 0.0, 0.0)),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });

            let cut = graph.add(Node::Boolean {
                op: BooleanOp::Intersection,
                inputs: vec![cube.into(), plane.into()],
            });
            let (lo, hi) = crate::graph_extent(&doc, &graph, cut)
                .as_box()
                .expect("an intersection with a bounded operand is bounded");
            assert_eq!(lo, Point3::new(-1.0, -1.0, -1.0));
            assert_eq!(hi, Point3::new(1.0, 1.0, 1.0));

            // And subtracting one: a cut cannot reach outside what is being cut.
            let mut graph = Graph::empty();
            let cube = unit_cube(&mut graph);
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At(at(0.0, 0.0, 0.0)),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let carved = graph.add(Node::Boolean {
                op: BooleanOp::Difference,
                inputs: vec![cube.into(), plane.into()],
            });
            assert!(crate::graph_extent(&doc, &graph, carved).as_box().is_some());
        }

        /// The other half, and the one that was silently wrong before: a union really is
        /// unbounded, and saying otherwise would clip whatever gets baked from it.
        #[test]
        fn a_part_unioned_with_an_infinite_plane_is_unbounded() {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let cube = unit_cube(&mut graph);
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At(at(0.0, 0.0, 0.0)),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let both = graph.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: vec![cube.into(), plane.into()],
            });
            assert_eq!(crate::graph_extent(&doc, &graph, both), Extent::Unbounded);

            // And it survives being placed: no rigid motion makes a half-space finite.
            let placed = graph.add(Node::Transform {
                input: both.into(),
                translation: at(5.0, 0.0, 0.0),
                rotation: at(0.0, 30.0, 0.0),
                scale: ParamRef::Literal(2.0),
                about: fieldform_scene::Place::origin(),
            });
            assert_eq!(crate::graph_extent(&doc, &graph, placed), Extent::Unbounded);
        }

        /// A fillet bulges past both operands, so the blend's box has to allow for it — and the
        /// room it needs is `0.293·k`, not `k`.
        ///
        /// **The number was `k` here, and `k` is the radius rather than the reach.** At G1 the
        /// value is `max(k, m) − √((k−a)₊² + (k−b)₊²)`, so where the two operands read alike at
        /// `m < k` it is `k − √2(k−m)`, which crosses nought at `m = k(1 − 1/√2) = 0.293·k`.
        /// Straight out from one body alone, where the other is further than `k`, only one term
        /// survives and the value is `m` itself — so a fillet adds **nothing** in that direction,
        /// and the old box reserved `k` of room there for material that is never made.
        ///
        /// Which was harmless while it was only loose, and became load-bearing when the growth
        /// moved from the join to its operands (`extent_below`): the same `k` over an operand
        /// whose value reads short of its distance was short by that operand's divisor, which is
        /// §1.8 R2. So this asserts the derived reach, and then checks it by sampling — because a
        /// derivation asserted against itself is not a test.
        #[test]
        fn a_blend_reserves_room_for_its_fillet_where_a_boolean_does_not() {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let k = 0.5f64;
            let a = unit_cube(&mut graph);
            let b = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At(at(2.0, 0.0, 0.0)),
                radius: ParamRef::Literal(1.0),
            });
            let hard = graph.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: vec![a.into(), b.into()],
            });
            let soft = graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: a.into(),
                b: b.into(),
                fillet: Fillet { radius: ParamRef::Literal(k), continuity: fieldform_scene::Continuity::G1 },
            });
            let reach = k * (1.0 - 2f64.powf(-0.5));
            let (hlo, hhi) = crate::graph_extent(&doc, &graph, hard).as_box().unwrap();
            let (slo, shi) = crate::graph_extent(&doc, &graph, soft).as_box().unwrap();
            assert!(
                (slo.x - (hlo.x - reach)).abs() < 1e-12,
                "the blend reserves {:.4} where the fillet reaches {reach:.4}",
                hlo.x - slo.x
            );
            assert!((shi.x - (hhi.x + reach)).abs() < 1e-12);
            // Larger than the hard boolean's, which is the half of this that was always the
            // point: a box that did not grow at all would pass every assertion above at k = 0.
            assert!(slo.x < hlo.x && shi.x > hhi.x);

            // **And the box holds.** The blend's own field, sampled over the box grown by half
            // its size: no point outside the box may be solid. This is the property the 153
            // cases in `tests/extent_holds_*.rs` assert at scale; here it is what stops the
            // arithmetic above from being checked against itself.
            let field = crate::GraphField::compile_graph(&doc, &graph, 1e-4)
                .expect("a blend compiles");
            let (lo, hi) = (slo, shi);
            let pad = (hi - lo).norm() * 0.5;
            let steps = 24;
            for i in 0..=steps {
                for j in 0..=steps {
                    for l in 0..=steps {
                        let t = |n: usize, axis: usize| {
                            lo[axis] - pad
                                + (hi[axis] - lo[axis] + 2.0 * pad) * n as f64 / steps as f64
                        };
                        let p = Point3::new(t(i, 0), t(j, 1), t(l, 2));
                        let outside = (0..3).any(|d| p[d] < lo[d] - 1e-9 || p[d] > hi[d] + 1e-9);
                        if outside {
                            assert!(
                                field.distance(p) > -1e-9,
                                "solid at {p:?}, which is outside the box {lo:?}..{hi:?}"
                            );
                        }
                    }
                }
            }
        }

        /// A part's extent is the union of its roots, because a part's field is their union —
        /// so one unbounded body makes the whole part unbounded.
        #[test]
        fn a_parts_extent_is_the_union_of_its_bodies() {
            let doc = Document::new();
            let mut graph = Graph::empty();
            unit_cube(&mut graph);
            let (lo, hi) = crate::part_extent(&doc, &graph)
                .as_box()
                .expect("one bounded body");
            assert_eq!(
                (lo, hi),
                (Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0))
            );

            graph.add(Node::Plane {
                origin: fieldform_scene::Place::At(at(0.0, 0.0, 0.0)),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            assert_eq!(crate::part_extent(&doc, &graph), Extent::Unbounded);
        }

        /// A graph with nothing in it bounds nothing, and says so as `Empty` rather than as a
        /// box around the origin — which would frame a view on geometry that is not there.
        #[test]
        fn an_empty_graph_has_an_empty_extent() {
            let doc = Document::new();
            assert_eq!(crate::part_extent(&doc, &Graph::empty()), Extent::Empty);
        }
    }
