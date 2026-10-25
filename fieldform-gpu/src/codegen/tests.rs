// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The generator's tests: what each block writes, the slots both passes count, and what recompiles.

    use fieldform_scene::Fillet;

    /// A block aimed along a line turns about that line, and follows it when it moves.
    ///
    /// **The axis a radial pattern turns about, drawn rather than typed.** A pattern about a
    /// line through x = 30 puts its copies in a ring around x = 30; the same pattern with the
    /// line moved puts them somewhere else, without the node changing. That second half is what
    /// separates a reference from six numbers copied at build time.
    #[test]
    fn a_frame_taken_along_a_line_aims_where_the_line_points() {
        use fieldform_field::ScalarField;
        use nalgebra::Point3;
        let lit = ParamRef::Literal;
        let mut doc = fieldform_scene::Document::new();

        // An axis up +Z, standing at x = 30 rather than through the origin, so a frame that
        // ignored the origin and kept only the direction would still fail this.
        let axis = fieldform_scene::id::CurveId::new();
        let put = |doc: &mut fieldform_scene::Document, x: f64| {
            doc.curves.insert(
                axis,
                fieldform_scene::Curve::polyline(
                    axis,
                    "Axis",
                    vec![Point3::new(x, 0.0, 0.0), Point3::new(x, 0.0, 40.0)],
                    false,
                )
                .expect("two points make a line"),
            );
        };
        put(&mut doc, 30.0);

        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(38.0), lit(0.0), lit(0.0)]),
            radius: lit(3.0),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(4.0),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::Along { curve: axis },
            },
        
            join: true,
            fillet: Fillet::hard(),
        });

        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("a frame along a line compiles");
        // Four about x = 30 at radius 8: the quarter turn lands one at (30, 8, 0).
        assert!(field.distance(Point3::new(30.0, 8.0, 0.0)) < 0.0, "turned about the line");
        // And nothing is orbiting the world origin, which is where a frame that lost its origin
        // would have put them.
        assert!(field.distance(Point3::new(0.0, 8.0, 0.0)) > 0.0);

        // Move the line; the ring moves with it.
        put(&mut doc, -30.0);
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("still compiles");
        assert!(field.distance(Point3::new(30.0, 8.0, 0.0)) > 0.0, "left where it was");

        generate(&doc, &graph).expect("a frame along a line lowers");
    }

    /// A cylinder built between a line's ends sits on that line, and follows it when it moves.
    ///
    /// **The whole of what the user asked for**, checked as a property rather than by reading
    /// the code: declare a line, feed it to a solid, and the solid is where the line is. The
    /// second half is the part that would be easy to get subtly wrong — a value copied at build
    /// time would pass the first assertion and fail this one.
    #[test]
    fn a_solid_built_between_a_curve_s_ends_follows_the_curve() {
        use fieldform_field::ScalarField;
        use nalgebra::Point3;
        let lit = ParamRef::Literal;
        let mut doc = fieldform_scene::Document::new();

        let line = fieldform_scene::id::CurveId::new();
        doc.curves.insert(
            line,
            fieldform_scene::Curve::polyline(
                line,
                "Strut",
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 50.0)],
                false,
            )
            .expect("two points make a line"),
        );

        let mut graph = Graph::empty();
        graph.add(Node::Cylinder { edge: Default::default(),
            ends: fieldform_scene::Ends::From { curve: line },
            radius: lit(5.0),
        });

        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("a cylinder on a curve compiles");
        // Along the line, well inside the radius.
        assert!(field.distance(Point3::new(0.0, 0.0, 25.0)) < 0.0);
        // Past the far end, outside it.
        assert!(field.distance(Point3::new(0.0, 0.0, 60.0)) > 0.0);
        // Beside the line, outside the radius.
        assert!(field.distance(Point3::new(20.0, 0.0, 25.0)) > 0.0);

        // **Move the line; the cylinder moves.** Nothing about the node changed — this is the
        // difference between naming a curve and copying two triples out of it.
        doc.curves.insert(
            line,
            fieldform_scene::Curve::polyline(
                line,
                "Strut",
                vec![Point3::new(40.0, 0.0, 0.0), Point3::new(40.0, 0.0, 50.0)],
                false,
            )
            .expect("two points make a line"),
        );
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("still compiles");
        assert!(field.distance(Point3::new(40.0, 0.0, 25.0)) < 0.0, "moved with the line");
        assert!(field.distance(Point3::new(0.0, 0.0, 25.0)) > 0.0, "left where it was");

        // And it lowers to a shader, which is the half that has its own arithmetic: the ends are
        // resolved to numbers at generation time because the shader has no curves in it.
        generate(&doc, &graph).expect("a cylinder on a curve lowers");
    }

    /// The three two-point solids take their ends the same way, so one can be swapped for
    /// another without rebuilding what it was told.
    ///
    /// D9's actual claim, as an assertion: not that they each hold two points, but that the
    /// *same value* drives all three. A cone that took its ends in its own spelling would fail
    /// to compile here, which is the point of writing it.
    #[test]
    fn the_two_point_solids_are_told_where_they_go_the_same_way() {
        let lit = ParamRef::Literal;
        let doc = fieldform_scene::Document::new();
        let ends = fieldform_scene::Ends::between(
            fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(30.0)]));

        let mut graph = Graph::empty();
        graph.add(Node::Cylinder { edge: Default::default(), ends, radius: lit(4.0) });
        graph.add(Node::Cone { edge: Default::default(), ends, base_radius: lit(4.0), top_radius: lit(0.0) });
        graph.add(Node::Capsule { ends, start_radius: lit(4.0), end_radius: lit(4.0) });

        fieldform_field::GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("all three take the same ends");
        generate(&doc, &graph).expect("and all three lower");
    }

    /// One body used by two operations is a graph this codebase can already express.
    ///
    /// **The question a user asked, answered by building it.** "Add a sphere, use it in one
    /// boolean, then use the *same* sphere in a second one" is a node with two consumers — a DAG
    /// rather than a tree. `Graph::enclosing_transform`'s doc comment claimed such a state was
    /// unreachable because shader generation refused to sample a node at two points; that was
    /// true until point contexts were built (backlog 21) and has been wrong since.
    ///
    /// Both halves are checked, because the two failure modes are different: the CPU could
    /// evaluate a shared node happily while the shader generator refused it, which is exactly
    /// the shape the old refusal had.
    #[test]
    fn one_body_can_be_an_operand_of_two_others() {
        use fieldform_field::ScalarField;
        use nalgebra::Point3;
        let doc = fieldform_scene::Document::new();
        let lit = ParamRef::Literal;

        let mut graph = Graph::empty();
        let shared = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(20.0) });
        let left = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(-15.0), lit(0.0), lit(0.0)]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let right = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(15.0), lit(0.0), lit(0.0)]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        // The same sphere, named twice.
        let a = graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![left.into(), shared.into()] });
        let b = graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![right.into(), shared.into()] });

        // Two bodies, and the sphere is not one of them: it is consumed, so it is not a root.
        // That is the whole model — a body is a node nothing else consumes.
        let roots = graph.roots();
        assert_eq!(roots.len(), 2, "the two booleans are the bodies");
        assert!(roots.contains(&a) && roots.contains(&b));
        assert!(!roots.contains(&shared), "a consumed node is not a body");

        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 0.05)
            .expect("a shared operand compiles");
        // Inside the left cube and inside the sphere, so inside the part.
        assert!(field.distance(Point3::new(-15.0, 0.0, 0.0)) < 0.0);
        // Inside the right cube and inside the sphere: the sphere being spent on the left did
        // not consume it.
        assert!(field.distance(Point3::new(15.0, 0.0, 0.0)) < 0.0);
        // Inside the sphere but in neither cube.
        assert!(field.distance(Point3::new(0.0, 0.0, 18.0)) > 0.0);

        generate(&doc, &graph).expect("a shared operand lowers to a shader");
    }

    /// **A chamfer's second leg is one slot after the first in both passes, and scales the
    /// operands the way the field does** (`round-chamfer-spec.md` §9.2): on a blend the join
    /// reads `min` of the two legs over operands scaled by it, and on a rim the leg rides in the
    /// vector's exponent slot, negated, which `rim_shape` in the prelude reads back.
    #[test]
    fn a_chamfers_second_leg_is_a_slot_after_the_first_in_both_passes() {
        let lit = ParamRef::Literal;
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(10.0), lit(0.0), lit(0.0)]),
            radius: lit(12.0),
        });
        let block = graph.add(Node::Cuboid {
            edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(-8.0), lit(0.0), lit(0.0)]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Blend {
            op: fieldform_scene::BooleanOp::Union,
            a: ball.into(),
            b: block.into(),
            fillet: Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::Chamfer, second: Some(lit(5.0)) },
        });
        let program = generate(&doc, &graph).expect("generates");
        let r = program.params.iter().position(|p| *p == lit(2.0)).expect("the first leg's slot");
        assert_eq!(program.params.get(r + 1), Some(&lit(5.0)), "the second leg's slot follows it");
        let body = program.field_function();
        let m = format!("min(params[{r}], params[{}])", r + 1);
        let scaled = format!("blend_bound(((n1) * {m} / params[{}])", r + 1);
        assert!(body.contains(&scaled), "the value pass scales the operands by the lesser leg: {body}");
        assert!(
            body.contains(&format!("blend_bound_normal(((v1) * {m} / params[{}])", r + 1)),
            "and the normal pass the same: {body}"
        );
        let (plain, _) = {
            let mut g2 = Graph::empty();
            let a = g2.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(10.0), lit(0.0), lit(0.0)]), radius: lit(12.0) });
            let b = g2.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(-8.0), lit(0.0), lit(0.0)]), size: [lit(20.0); 3], turn: fieldform_scene::no_turn() });
            g2.add(Node::Blend { op: fieldform_scene::BooleanOp::Union, a: a.into(), b: b.into(), fillet: Fillet { radius: lit(2.0), continuity: fieldform_scene::Continuity::Chamfer, second: None } });
            (generate(&doc, &g2).expect("generates"), ())
        };
        assert_eq!(program.params.len(), plain.params.len() + 1, "one slot more than one leg");

        // A rim's second leg: negated in the exponent's place, the slot after the reach's.
        let mut doc = fieldform_scene::Document::new();
        let curve = fieldform_scene::CurveId::new();
        doc.curves.insert(
            curve,
            fieldform_scene::Curve::polyline(
                curve,
                "square",
                vec![
                    nalgebra::Point3::new(0.0, 0.0, 0.0),
                    nalgebra::Point3::new(10.0, 0.0, 0.0),
                    nalgebra::Point3::new(10.0, 10.0, 0.0),
                    nalgebra::Point3::new(0.0, 10.0, 0.0),
                ],
                true,
            )
            .expect("closed"),
        );
        let mut graph = Graph::empty();
        graph.add(Node::Extrude {
            outline: fieldform_scene::Outline::closed(curve),
            sweep: fieldform_scene::Sweep::Normal,
            start: lit(0.0),
            height: lit(5.0),
            draft: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
            rims: fieldform_scene::Rims {
                radius: lit(1.25),
                continuity: fieldform_scene::Continuity::Chamfer,
                at: fieldform_scene::RimEnds::Both,
                second: Some(lit(0.75)),
            },
        });
        let program = generate(&doc, &graph).expect("generates");
        let r = program.params.iter().position(|p| *p == lit(1.25)).expect("the reach's slot");
        assert_eq!(program.params.get(r + 1), Some(&lit(0.75)), "the second leg's slot follows the reach's");
        let rim = format!("vec4<f32>(params[{r}], -params[{}], 0.0, 1.0)", r + 1);
        // Once in the value pass, and twice in the normal pass, whose body hands its value up
        // beside its normal.
        assert_eq!(program.field_function().matches(&rim).count(), 3, "both passes write the rim with its leg negated in the exponent's place: {}", program.field_function());
    }

    /// **The normal pass reads every profile solid through its `_bake` copy, and the value pass
    /// through the original.** The copies are the prelude's profile-solid region instantiated
    /// again by name (`instantiate_bake_copies`); what this holds is that the region covers the
    /// solids the normal pass reads, that a copy reads its section through
    /// `profile_distance_bake`, and that the generated bodies call the right one in each pass.
    #[test]
    fn the_normal_pass_reads_every_profile_solid_through_its_bake_copy() {
        for name in [
            "extrude_distance",
            "extrude_rim_distance",
            "extrude_rim_normal",
            "revolve_distance",
            "revolve_normal",
            "revolve_wedge_rim_distance",
            "revolve_wedge_rim_normal",
            "wall_distance",
            "wall_extrude_rim_distance",
            "wall_revolve_wedge_rim_distance",
            "sweep_distance",
            "sweep_value_normal",
            "sweep_law_rim_value_normal",
            "wall_sweep_rim_value_normal",
            "loft_distance",
            "loft_value_normal",
        ] {
            assert!(bake_copy_names().iter().any(|n| n == name), "{name} is in the region");
            assert!(prelude().contains(&format!("fn {name}_bake(")), "{name} has its copy");
        }
        let copy_start = prelude().find("fn extrude_rim_distance_bake(").expect("the copy");
        let copy = &prelude()[copy_start..copy_start + 700];
        assert!(copy.contains("profile_distance_bake("), "a copy reads the section's bake: {copy}");
        assert!(!copy.contains(" profile_distance("), "and never the walk: {copy}");
        assert_eq!(
            PRELUDE.matches("// unbaked-section-walk").count(),
            1,
            "the one walk the renderer cuts when every section is baked"
        );

        let lit = ParamRef::Literal;
        let mut doc = fieldform_scene::Document::new();
        let curve = fieldform_scene::CurveId::new();
        doc.curves.insert(
            curve,
            fieldform_scene::Curve::polyline(
                curve,
                "square",
                vec![
                    nalgebra::Point3::new(0.0, 0.0, 0.0),
                    nalgebra::Point3::new(10.0, 0.0, 0.0),
                    nalgebra::Point3::new(10.0, 10.0, 0.0),
                    nalgebra::Point3::new(0.0, 10.0, 0.0),
                ],
                true,
            )
            .expect("closed"),
        );
        let mut graph = Graph::empty();
        graph.add(Node::Extrude {
            outline: fieldform_scene::Outline::closed(curve),
            sweep: fieldform_scene::Sweep::Normal,
            start: lit(0.0),
            height: lit(5.0),
            draft: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
            rims: Default::default(),
        });
        let body = generate(&doc, &graph).expect("generates").field_function().to_string();
        assert_eq!(body.matches("extrude_distance(").count(), 1, "the value pass reads the original: {body}");
        assert_eq!(body.matches("extrude_distance_bake(").count(), 1, "the normal pass's value, its copy");
        assert_eq!(body.matches("extrude_normal_bake(").count(), 1, "and its normal, its copy");
        assert_eq!(body.matches("extrude_normal(").count(), 0, "never the walking normal");
    }

    /// **The normal fold keeps the hit body's seam scale and no other's.** Each body's normal
    /// starts from one, and only the body the fold takes hands its scale on, so a fillet in a
    /// body the ray passed over cannot brighten a pixel on a plain face of the body it hit
    /// (`seam_scale` in the prelude, `round-chamfer-spec.md` §4.4).
    #[test]
    fn the_normal_fold_keeps_the_hit_bodys_seam_scale() {
        let lit = ParamRef::Literal;
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(10.0), lit(0.0), lit(0.0)]),
            radius: lit(12.0),
        });
        let block = graph.add(Node::Cuboid {
            edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(-8.0), lit(0.0), lit(0.0)]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Blend {
            op: fieldform_scene::BooleanOp::Union,
            a: ball.into(),
            b: block.into(),
            fillet: Fillet { radius: lit(3.0), continuity: fieldform_scene::Continuity::G1, second: None },
        });
        graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(60.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        let program = generate(&doc, &graph).expect("the shader compiles");
        let normal = program
            .source
            .split("fn field_normal(")
            .nth(1)
            .expect("a normal fold");
        assert_eq!(normal.matches("seam_scale = 1.0; let r").count(), 2, "each body starts from one");
        assert_eq!(normal.matches("hit_scale = seam_scale;").count(), 2, "and the taken body hands its on");
        assert!(
            normal.contains("seam_scale = hit_scale;\n    return normalize(nearest.xyz);"),
            "the fold leaves the hit body's scale for the march"
        );
    }

    /// **A varying fillet radius divides by the number the field divides by**, and reaches the
    /// buffer as a divisor and not as a slot (`round-chamfer-spec.md` §9.1). The pipe's test
    /// below, for the pipe's reason: the two paths lower the radius separately and the divisor
    /// is the one thing that can silently differ.
    #[test]
    fn a_varying_fillet_radius_divides_by_the_bound_the_field_uses() {
        let lit = ParamRef::Literal;
        let doc = fieldform_scene::Document::new();
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
            out_min: lit(1.0),
            out_max: lit(6.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(10.0), lit(0.0), lit(0.0)]),
            radius: lit(12.0),
        });
        let block = graph.add(Node::Cuboid {
            edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(-8.0), lit(0.0), lit(0.0)]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let blend = graph.add(Node::Blend {
            op: fieldform_scene::BooleanOp::Union,
            a: ball.into(),
            b: block.into(),
            fillet: Fillet { radius: ParamRef::Field(ramp), continuity: fieldform_scene::Continuity::G1, second: None },
        });

        let program = generate(&doc, &graph).expect("the shader compiles");
        let divisor = fieldform_field::driven_blend_divisor(&doc, &graph, blend);
        assert!(divisor > 1.0, "a varying radius costs a bound, got {divisor}");
        let d = program
            .params
            .iter()
            .position(|r| *r == ParamRef::Literal(divisor))
            .expect("the shader divides by what the field divides by, from a slot");
        assert!(!program.source.contains(&format!("{divisor:?}")), "and not as a literal");
        assert!(
            program.source.contains(&format!("/ params[{d}]")),
            "the blend's value is divided by its slot"
        );
        assert!(
            !program.params.iter().any(|r| matches!(r, ParamRef::Field(_))),
            "a driven radius must not reach the parameter buffer",
        );
        // The fixed form reads a slot where the driven one reads the ramp's own value.
        assert!(program.source.contains("blend_bound("), "the circle's bound is the form");
    }

    /// **A varying pipe radius divides by the number the field divides by.**
    ///
    /// Backlog item 2. The two paths lower it separately -- the field into `Op::Offset` over a
    /// spine of radius nought, the shader into the same expression -- so the one thing that can
    /// silently differ is the divisor. A mismatch moves no surface and so never shows as a
    /// residual; it shows only as a tracer stepping further than the field justifies, which is
    /// the failure `driven_offset_divisor`'s own note warns about.
    #[test]
    fn a_varying_pipe_radius_divides_by_the_bound_the_field_uses() {
        let lit = ParamRef::Literal;
        let mut doc = fieldform_scene::Document::new();
        let curve = fieldform_scene::CurveId::new();
        doc.curves.insert(
            curve,
            fieldform_scene::Curve::polyline(
                curve,
                "spine",
                vec![
                    nalgebra::Point3::new(0.0, 0.0, 0.0),
                    nalgebra::Point3::new(60.0, 0.0, 0.0),
                ],
                false,
            )
            .expect("a two-point line"),
        );

        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::Turn::Along { curve },
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(0.0),
            in_max: lit(60.0),
            out_min: lit(1.0),
            out_max: lit(6.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        let pipe = graph.add(Node::Pipe { curve, radius: ParamRef::Field(ramp) });

        let program = generate(&doc, &graph).expect("the shader compiles");
        let divisor = fieldform_field::driven_pipe_divisor(&doc, &graph, pipe);
        assert!(divisor > 1.0, "a varying radius costs a bound, got {divisor}");
        // In a slot, not in the source: the bound is the curve's shape, and a number written
        // into the shader made every step of dragging a spine's point a new shader to compile.
        assert!(
            program.params.contains(&ParamRef::Literal(divisor)),
            "the shader does not divide by {divisor:?}, which is what the field divides by",
        );
        assert!(!program.source.contains(&format!("{divisor:?}")), "and not as a literal");
        // And no parameter slot for the radius, since it is an expression now -- the two passes
        // count slots and would shift every number after it if only one of them skipped it.
        assert!(
            !program.params.iter().any(|r| matches!(r, ParamRef::Field(_))),
            "a driven radius must not reach the parameter buffer",
        );
    }

    /// **Lowering does not turn an ingredient into a body.**
    ///
    /// A lattice aimed at a plane is rewritten into a placement over a lattice facing its own
    /// way, and the frame that is dropped on the way was the only thing consuming that plane.
    /// Unconsumed means a root, a root means a body, and a plane as a body is a shaded half-space
    /// across the whole viewport -- which is what was reported of the gyroid tool's own "make a
    /// plane". The CPU path never showed it, because it reads the frame where it stands rather
    /// than lowering it away.
    #[test]
    fn lowering_a_frame_does_not_leave_the_plane_it_read_as_a_body() {
        let lit = ParamRef::Literal;
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::origin(),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let lattice = graph.add(Node::Lattice {
            cell: fieldform_scene::LatticeCell::Gyroid,
            map: fieldform_scene::CellMap::Rectangular { cell: lit(12.0) },
            phase: LatticePhase::sheet(lit(1.5)),
            frame: fieldform_scene::Frame::From { node: plane },
        });
        assert_eq!(graph.roots(), vec![lattice], "the plane is an operand, not a body");

        let lowered = only_the_same_bodies(&graph, expand_shells(&expand_frames(&doc, &graph)));
        assert!(
            !lowered.roots().contains(&plane),
            "the lowering left the plane as a body, which draws it as a solid",
        );
        assert_eq!(lowered.roots(), graph.roots(), "and the bodies are the bodies they were");
    }

    /// A lowering must describe the same solid it lowered.
    ///
    /// **The guard for a bug that shipped.** `expand_frames` rewrites an aimed block as an
    /// unaimed one under a placement, and for a twist that needs a *sandwich* — into the frame,
    /// wind, back out — because a twist wraps a body and a single placement would hand that body
    /// a point in the frame's local space. It was written as a single placement, and the only
    /// thing that noticed was a CPU/GPU residual of 248 mm on a scene that needs a GPU to run.
    ///
    /// This asks the cheaper question: compile the graph and its lowering on the *CPU* and
    /// compare them. Nothing about the shader is involved, so it runs anywhere, and any lowering
    /// that stops describing what it lowered fails here first.
    #[test]
    fn lowering_a_frame_describes_the_same_solid() {
        use fieldform_field::ScalarField;
        let doc = fieldform_scene::Document::new();
        let lit = ParamRef::Literal;

        for frame in [
            fieldform_scene::Frame::default(),
            fieldform_scene::Frame::along(fieldform_scene::Axis::X),
            fieldform_scene::Frame::at_along(
                fieldform_scene::Axis::Y,
                [lit(7.0), lit(-3.0), lit(2.0)],
            ),
            fieldform_scene::Frame::At {
                origin: [lit(4.0), lit(1.0), lit(-2.0)],
                axis: [lit(1.0), lit(2.0), lit(-1.0)],
            },
        ] {
            for twisted in [false, true] {
                let mut graph = Graph::empty();
                let body = graph.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([lit(0.0); 3]),
                    size: [lit(40.0), lit(30.0), lit(50.0)],
                    turn: fieldform_scene::no_turn(),
                });
                if twisted {
                    graph.add(Node::Twist {
                        input: body.into(),
                        degrees_per_mm: lit(3.0),
                        frame,
                    });
                } else {
                    graph.add(Node::Lattice {
                        cell: fieldform_scene::LatticeCell::Gyroid,
                        map: fieldform_scene::CellMap::Rectangular { cell: lit(12.0) },
                        phase: LatticePhase::sheet(lit(1.5)),
                        frame,
                    });
                }

                let lowered = expand_frames(&doc, &graph);
                let a = fieldform_field::GraphField::compile_graph(&doc, &graph, 1e-3)
                    .expect("the graph compiles");
                let b = fieldform_field::GraphField::compile_graph(&doc, &lowered, 1e-3)
                    .expect("the lowering compiles");

                for i in 0..60 {
                    let t = i as f64;
                    let p = nalgebra::Point3::new(
                        -30.0 + 1.7 * t,
                        -20.0 + 0.9 * t,
                        -25.0 + 1.3 * t,
                    );
                    let (x, y) = (a.distance(p), b.distance(p));
                    // **The same solid, which is the sign — not the same number.**
                    //
                    // A twist divides its value by a bound on how far the warp can stretch
                    // material, and that bound is measured from the body's box. Lowering rotates
                    // the body under a placement, and an axis-aligned box of a rotated body is
                    // larger than the box it was rotated from — so the lowering divides by a
                    // little more. Both are honest bounds, neither overstates, and dividing a
                    // field by a constant does not move its zero set.
                    //
                    // So the property is the shape: every point is inside for both or outside
                    // for both. That is what the 248 mm bug broke — a body moved — and what a
                    // lowering has to preserve.
                    assert_eq!(
                        x < 0.0,
                        y < 0.0,
                        "{} with {frame:?} at {p:?}: {x} against {y}",
                        if twisted { "a twist" } else { "a lattice" },
                    );
                }
            }
        }
    }
    use super::*;
    use fieldform_scene::{CageId, Continuity, CurveId, ParamRef};

    /// Parses generated WGSL and returns the validation error, if any.
    ///
    /// The whole reason this module is separate from the renderer: `naga` is the same front end
    /// wgpu uses, so a shader that parses and validates here is one the driver will accept.
    /// Getting that check without a GPU is what makes codegen testable in CI and on a laptop
    /// with no adapter, instead of only being testable by looking at a picture.
    fn validate(source: &str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|e| format!("parse failed: {}", e.emit_to_string(source)))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| format!("validation failed: {e:?}"))?;
        Ok(())
    }

    fn cage_graph() -> Graph {
        Graph::from_cage(CageId::new())
    }

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// The value of a reference the generator has already resolved to a literal.
    fn literal(r: &ParamRef) -> f64 {
        match r {
            ParamRef::Literal(v) => *v,
            other => panic!("expected a literal, got {other:?}"),
        }
    }

    /// The plane a mirror is emitted about, read back out of the generated source.
    ///
    /// Only two shapes ever reach here — `vec3<f32>(params[a], params[b], params[c])` and the
    /// same wrapped in `aimed(..)` — so this is a substitution rather than a parser.
    fn mirror_plane(
        program: &ShaderProgram,
    ) -> (nalgebra::Vector3<f64>, nalgebra::Vector3<f64>) {
        // The generated half only: the prelude declares `fn mirror_point(..)` and that
        // declaration is not the call site being read.
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("the generated half is marked");
        // Either spelling: a mirror whose body is all on one side folds the point onto that
        // side instead of reflecting it, and the plane it names is the same plane.
        let generated = &program.source[cut..];
        let call = generated
            .split("mirror_fold(")
            .nth(1)
            .or_else(|| generated.split("mirror_point(").nth(1))
            .expect("the shader mirrors something");
        let args = &call[..call.find(");").expect("a complete call")];
        let read = |arg: &str| -> nalgebra::Vector3<f64> {
            let inside = &arg[arg.find("vec3<f32>(").expect("a vector") + "vec3<f32>(".len()..];
            let mut out = nalgebra::Vector3::zeros();
            for (i, piece) in inside.split(',').take(3).enumerate() {
                let piece = piece.trim().trim_end_matches(')');
                out[i] = match piece.strip_prefix("params[") {
                    Some(rest) => {
                        let slot: usize =
                            rest[..rest.find(']').expect("a slot")].parse().expect("a number");
                        literal(&program.params[slot])
                    }
                    None => piece.parse().expect("a literal"),
                };
            }
            out
        };
        // `p0, at, normal` — the point first, which is not one of the two being read.
        let mut parts = args.splitn(2, ',');
        let _ = parts.next();
        let rest = parts.next().expect("an origin and a normal");
        let cut = rest.find("),").expect("two vectors") + 1;
        let (at, normal) = (read(&rest[..cut]), read(&rest[cut..]));
        (at, normal.normalize())
    }

    /// A mirror is aimed where the kernel says it is aimed, whatever it is aimed at.
    ///
    /// **The one block that resolved its own frame, and it had drifted three ways.** Every other
    /// aimed block goes through `frame_placement`, which calls the kernel's `resolve_frame`; the
    /// mirror had a private copy that still believed a plane faces `+Z`, that "the kernel refuses
    /// a frame that is not a plane", and that a named parameter's value can be read off the
    /// parameter. So a mirror aimed at a turned plane, or at a point, reflected about the world
    /// XY plane through the origin — a body in entirely the wrong place, with the panel, the
    /// picking and the CPU field all saying otherwise.
    ///
    /// The kernel is the authority, read on the same lowered graph the generator reads: a block
    /// carrying a turn *is* that block inside a `Node::Transform` by then, which is the very
    /// thing the old `_` arm fell through on.
    #[test]
    fn a_mirror_is_aimed_where_the_kernel_says_it_is() {
        use fieldform_scene::{Document, Frame, Place, Turn};
        let cases: Vec<(&str, Box<dyn Fn(&mut Graph) -> Frame>)> = vec![
            (
                "a typed frame",
                Box::new(|_: &mut Graph| Frame::At {
                    origin: [lit(10.0), lit(0.0), lit(0.0)],
                    axis: [lit(0.0), lit(3.0), lit(4.0)],
                }),
            ),
            (
                "a plane lying flat",
                Box::new(|g: &mut Graph| {
                    let p = g.add(Node::Plane {
                        origin: Place::At([lit(0.0), lit(25.0), lit(0.0)]),
                        flipped: false,
                        turn: Turn::default(),
                    });
                    Frame::From { node: p }
                }),
            ),
            (
                "a plane that has been turned",
                Box::new(|g: &mut Graph| {
                    let p = g.add(Node::Plane {
                        origin: Place::At([lit(0.0), lit(25.0), lit(0.0)]),
                        flipped: false,
                        turn: Turn::By([lit(35.0), lit(0.0), lit(0.0)]),
                    });
                    Frame::From { node: p }
                }),
            ),
            (
                "a plane that has been turned and flipped",
                Box::new(|g: &mut Graph| {
                    let p = g.add(Node::Plane {
                        origin: Place::At([lit(4.0), lit(25.0), lit(-6.0)]),
                        flipped: true,
                        turn: Turn::By([lit(0.0), lit(50.0), lit(20.0)]),
                    });
                    Frame::From { node: p }
                }),
            ),
            (
                "a point, which gives the place and leaves the facing upright",
                Box::new(|g: &mut Graph| {
                    let p = g.add(Node::Point { at: [lit(0.0), lit(18.0), lit(7.0)] });
                    Frame::From { node: p }
                }),
            ),
        ];

        for (label, build) in cases {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let frame = build(&mut graph);
            let body = graph.add(Node::Sphere {
                centre: Place::At([lit(0.0), lit(60.0), lit(0.0)]),
                radius: lit(10.0),
            });
            graph.add(Node::Mirror { input: body.into(), frame, join: true });

            let program = generate(&doc, &graph).expect("a mirror generates");
            let (at, normal) = mirror_plane(&program);

            // The kernel's own answer, on the graph the generator actually compiles: turns are
            // placements by then, which is the case the old code could not see.
            let resolve = |r: &ParamRef| -> Option<f64> { doc.resolve(r) };
            let lowered = fieldform_field::with_turns_flattened(&doc, &graph, &resolve);
            let Some(Node::Mirror { frame, .. }) =
                lowered.nodes().find(|(_, n)| matches!(n, Node::Mirror { .. })).map(|(_, n)| n)
            else {
                unreachable!("the mirror is still there")
            };
            let (want_at, rotation) =
                fieldform_field::resolve_frame(&doc, &lowered, frame, &resolve)
                    .expect("the kernel reads every frame here");
            let want_normal = rotation * nalgebra::Vector3::z();

            assert!(
                (at - want_at).norm() < 1e-6,
                "{label}: the shader mirrors about a plane at {at:?}, the kernel about one at \
                 {want_at:?}"
            );
            // Either sign is the same plane, so only the line matters.
            assert!(
                normal.cross(&want_normal).norm() < 1e-6,
                "{label}: the shader faces {normal:?}, the kernel {want_normal:?}"
            );
        }
    }

    /// Both compilers divide an aimed twist by the same number.
    ///
    /// **Written to settle a claim that turned out to be false**, and kept because the reasoning
    /// behind it is sound and only the conclusion was wrong. A twist stretches material sideways,
    /// so its value is divided by how far the body reaches across the axis; dividing by too
    /// little overstates distance, and an overstated distance is a ray stepping through the
    /// surface. Nothing would report it — the divisor is a constant over the whole field, so it
    /// leaves the zero set exactly where it was and `gpu_matches_cpu` sees nothing.
    ///
    /// And the two sides genuinely fold over different graphs: the kernel over
    /// `with_turns_flattened` alone (`let graph = &turned`), this crate over
    /// `only_the_same_bodies(.., expand_shells(expand_frames(.., &turned)))`. `expand_frames`
    /// rewrites an *aimed* twist as an unaimed one wrapped in a placement, with its body pushed
    /// through the inverse — so the reach is measured across a different axis, of a body in a
    /// different place, and there is no reason in the shape of the code for the answers to match.
    ///
    /// They match anyway, because the rewrite is a rigid motion and a reach is invariant under
    /// one. Measured, the shader's divisor over the kernel's:
    ///
    /// | twist frame | kernel | shader | ratio |
    /// |---|---|---|---|
    /// | the default, about +Z at the origin | 1.1467 | 1.1467 | 1.000 |
    /// | aimed along +Z, moved off the origin | 1.3076 | 1.3076 | 1.000 |
    /// | aimed along +Y | 1.4304 | 1.4304 | 1.000 |
    /// | aimed along +Y, moved off the origin | 1.6784 | 1.6784 | 1.000 |
    #[test]
    fn both_compilers_divide_an_aimed_twist_by_the_same_number() {
        use fieldform_scene::{Document, Frame, Place, Turn};
        let lit = |v: f64| ParamRef::Literal(v);
        let frames = [
            ("the default", Frame::default()),
            (
                "aimed along +Z, moved off the origin",
                Frame::At {
                    origin: [lit(30.0), lit(0.0), lit(-20.0)],
                    axis: [lit(0.0), lit(0.0), lit(1.0)],
                },
            ),
            (
                "aimed along +Y",
                Frame::At {
                    origin: [lit(0.0); 3],
                    axis: [lit(0.0), lit(1.0), lit(0.0)],
                },
            ),
            (
                "aimed along +Y, moved off the origin",
                Frame::At {
                    origin: [lit(30.0), lit(0.0), lit(-20.0)],
                    axis: [lit(0.0), lit(1.0), lit(0.0)],
                },
            ),
        ];
        let rate = 90.0 / 162.0;
        for (label, frame) in frames {
            let doc = Document::new();
            let mut graph = Graph::empty();
            let bar = graph.add(Node::Cuboid { edge: Default::default(),
                centre: Place::At([lit(0.0); 3]),
                size: [lit(110.0), lit(36.0), lit(180.0)],
                turn: Turn::default(),
            });
            graph.add(Node::Twist {
                input: bar.into(),
                degrees_per_mm: lit(rate),
                frame: frame.clone(),
            });

            // The kernel's arithmetic, on the graph the kernel folds over.
            let resolve = |r: &ParamRef| -> Option<f64> { doc.resolve(r) };
            let turned = fieldform_field::with_turns_flattened(&doc, &graph, &resolve);
            let (_, twist) = turned
                .nodes()
                .find(|(_, n)| matches!(n, Node::Twist { .. }))
                .expect("the twist survives flattening");
            let Node::Twist { input, .. } = twist else { unreachable!() };
            let (lo, hi) = fieldform_field::operand_extent(&doc, &turned, input)
                .as_box()
                .expect("a bounded bar");
            let (translation, rotation) =
                fieldform_field::resolve_frame(&doc, &turned, &frame, &resolve).expect("resolves");
            let reach = fieldform_field::reach_across_axis((lo, hi), translation, rotation);
            let kernel = (1.0 + (reach * rate.to_radians()).powi(2)).sqrt().max(1.0);

            // The shader's, read where it is now uploaded rather than written: the twist arm
            // emits `(value / params[n])`, since a divisor in the source made the one number a
            // twist has into the one number that could not be dragged.
            let program = generate(&doc, &graph).expect("a twist generates");
            let cut = program
                .source
                .find("// ---- generated from the composition graph ----")
                .expect("the generated half is marked");
            let shader = program.source[cut..]
                .lines()
                .filter_map(|line| {
                    let at = line.rfind(" / params[")?;
                    let rest = &line[at + " / params[".len()..];
                    let end = rest.find(']')?;
                    let slot: usize = rest[..end].parse().ok()?;
                    match program.params.get(slot)? {
                        ParamRef::Literal(v) if *v > 1.0 => Some(*v),
                        _ => None,
                    }
                })
                .next()
                .expect("the twist uploads its divisor");

            assert!(
                (shader - kernel).abs() < 1e-9,
                "{label}: the shader divides by {shader}, the kernel by {kernel}"
            );
        }
    }

    /// The shader gives up on an axis at the same length the field does.
    ///
    /// **Two spellings of one number, so they are checked against each other rather than
    /// remembered.** A `Frame::At` axis is the one frame value that stays live in the buffer, so
    /// it is the one the shader has to decide about itself; everything else is resolved on this
    /// side and uploaded already unit. Written at 1e-12 first against the kernel's 1e-9, which
    /// left a band three decades wide where `rotation_from_z` had already answered "no axis, face
    /// +Z" and the shader was still normalising and lighting a surface by it.
    ///
    /// `<=` rather than `<`, because that is what nalgebra's `try_normalize` does: it refuses at
    /// `n <= min_norm`.
    #[test]
    fn the_shader_gives_up_on_an_axis_at_the_same_length_the_field_does() {
        let floor = fieldform_field::AXIS_FLOOR;
        assert!(
            PRELUDE.contains(&format!("if (l <= {floor:e})")),
            "`aimed` has to give up at {floor:e}, the length `rotation_from_z` gives up at"
        );
        // And the kernel really does give up there, rather than just below it -- which is the
        // half of the rule that is easy to write down wrong.
        let at_the_floor = nalgebra::Vector3::new(floor, 0.0, 0.0);
        assert_eq!(
            fieldform_field::rotation_from_z(at_the_floor) * nalgebra::Vector3::z(),
            nalgebra::Vector3::z(),
            "an axis exactly at the floor is no axis"
        );
        let just_above = nalgebra::Vector3::new(floor * 2.0, 0.0, 0.0);
        assert!(
            (fieldform_field::rotation_from_z(just_above) * nalgebra::Vector3::z())
                .dot(&nalgebra::Vector3::x())
                > 0.99,
            "and one just above it is an axis"
        );
    }

    /// An axis of nothing at all is `+Z`, not a division by zero.
    ///
    /// The kernel's `rotation_from_z` says so deliberately: a zero vector is a state you pass
    /// through while typing one component at a time, and refusing it turns typing into an error.
    /// WGSL's `normalize` of a zero vector is undefined and comes back NaN, which reaches the
    /// march as a blank viewport a long way from the cause — so the shader says it too, in
    /// `aimed`.
    #[test]
    fn a_mirror_with_no_axis_faces_up_rather_than_reading_nothing() {
        use fieldform_scene::{Document, Frame, Place};
        let doc = Document::new();
        let mut graph = Graph::empty();
        let body = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0), lit(60.0), lit(0.0)]),
            radius: lit(10.0),
        });
        graph.add(Node::Mirror {
            input: body.into(),
            frame: Frame::At { origin: [lit(0.0); 3], axis: [lit(0.0); 3] },
            join: true,
        });
        let program = generate(&doc, &graph).expect("a mirror generates");
        assert!(
            !program.source.contains("normalize(vec3<f32>(params["),
            "a frame's axis goes through `aimed`, which has an answer for nothing"
        );
        assert!(program.source.contains("aimed(vec3<f32>(params["));
        validate(&program.source).expect("naga accepts it");
    }

    /// The generated shader has to declare its point chain before anything samples it, and a
    /// transform's own line is emitted after the subtree that needs the point — so an
    /// implementation that declared them inline would produce source referring to a variable
    /// that does not exist yet. naga catches that here, without a GPU.
    #[test]
    fn a_placed_body_generates_a_shader_that_declares_its_point_before_using_it() {
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        graph.add(Node::Transform {
            input: sphere.into(),
            translation: [lit(7.0), lit(-3.0), lit(2.0)],
            rotation: [lit(23.0), lit(-47.0), lit(11.0)],
            scale: lit(1.5),
            about: fieldform_scene::Place::origin(),
        });
        let program = generate(&fieldform_scene::Document::new(), &graph).unwrap();

        assert!(program.source.contains("let p1 = to_local_about(p0"), "{}", program.source);
        // Ten numbers per placement: three translation, three rotation, one scale, three pivot.
        assert_eq!(program.params.len(), 4 + 10);
        validate(&program.source).unwrap();
    }

    /// Nested placements compose, and the inner one has to be declared from the outer one's
    /// point rather than from the world point — otherwise the inner body ignores the outer
    /// placement entirely, which reads as "the transform did nothing" only for the child.
    #[test]
    fn a_placement_inside_a_placement_chains_from_its_parents_point() {
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        let inner = graph.add(Node::Transform {
            input: sphere.into(),
            translation: [lit(1.0), lit(0.0), lit(0.0)],
            rotation: [lit(0.0), lit(0.0), lit(0.0)],
            scale: lit(2.0),
            about: fieldform_scene::Place::origin(),
        });
        graph.add(Node::Transform {
            input: inner.into(),
            translation: [lit(0.0), lit(4.0), lit(0.0)],
            rotation: [lit(0.0), lit(0.0), lit(90.0)],
            scale: lit(1.0),
            about: fieldform_scene::Place::origin(),
        });
        let program = generate(&fieldform_scene::Document::new(), &graph).unwrap();

        // The outer transform is nearer the root, so it opens the first point, and the inner
        // one chains off it.
        assert!(program.source.contains("let p1 = to_local_about(p0"), "{}", program.source);
        assert!(program.source.contains("let p2 = to_local_about(p1"), "{}", program.source);
        validate(&program.source).unwrap();
    }

    /// A placement's pivot reaches the shader as three numbers, not as a recompile.
    ///
    /// **Because moving a pivot is moving a number.** It is where a body turns and grows from,
    /// so it is exactly the sort of thing somebody drags — and a pivot generated into the
    /// source would rebuild the program on every frame of that drag, which is the one cost
    /// this whole parameter buffer exists to avoid.
    ///
    /// Checked by its value rather than only by its slot count: the three could be present and
    /// hold anything, and a pivot the shader reads as the origin is a pivot that does nothing.
    #[test]
    fn a_placements_pivot_is_three_parameters_rather_than_a_new_shader() {
        let build = |pivot: [f64; 3]| {
            let mut graph = Graph::empty();
            let sphere = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
                radius: lit(5.0),
            });
            graph.add(Node::Transform {
                input: sphere.into(),
                translation: [lit(0.0), lit(0.0), lit(0.0)],
                rotation: [lit(0.0), lit(0.0), lit(0.0)],
                scale: lit(2.0),
                about: fieldform_scene::Place::At([
                    lit(pivot[0]),
                    lit(pivot[1]),
                    lit(pivot[2]),
                ]),
            });
            generate(&fieldform_scene::Document::new(), &graph).unwrap()
        };
        let origin = build([0.0, 0.0, 0.0]);
        let moved = build([10.0, -4.0, 2.5]);

        assert_eq!(origin.source, moved.source, "moving a pivot rebuilt the shader");
        assert_eq!(origin.params.len(), moved.params.len());
        // Spelled as an index rather than searched for, because *where* it lands is the thing
        // that has to hold: the placement is the first context emitted, so its ten numbers come
        // first in the order the two passes both count them -- move, turn, scale, pivot. A pass
        // that pushed them in a different order would still contain these three somewhere.
        let pivot: Vec<f64> = moved.params[7..10].iter().map(literal).collect();
        assert_eq!(pivot, vec![10.0, -4.0, 2.5]);
        validate(&moved.source).unwrap();
    }

    /// Everything the distance pass generated: the per-body functions and the fold over them.
    ///
    /// Counting a call across the whole source counts the prelude's *definition* of it and the
    /// normal pass's copy as well, which is how the first version of these two tests came out
    /// at five instead of two.
    ///
    /// **Not just `fn field`.** Each body became a function of its own so that a driver's
    /// optimiser sees several small problems rather than one large one (backlog item 34), so
    /// the composition these tests count occurrences in is spread across `body_0`, `body_1`
    /// and the fold. Bounded at `fn field_normal`, because the normal pass emits the same
    /// calls again for its own reasons and counting those would double every answer.
    fn field_body(source: &str) -> &str {
        let generated = source.split("// ---- generated from the composition graph ----")
            .nth(1)
            .expect("a generated shader has a generated section");
        // Bounded where the *normal pass* begins, which is its first body function and no
        // longer `fn field_normal` -- that pass emits the same calls again for its own reasons,
        // and counting those would double every answer here.
        let end = generated
            .find("fn body_normal_")
            .or_else(|| generated.find("fn field_normal("))
            .unwrap_or(generated.len());
        &generated[..end]
    }

    #[test]
    fn a_body_placed_twice_is_emitted_twice_rather_than_refused() {
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        for offset in [1.0, 9.0] {
            graph.add(Node::Transform {
                input: sphere.into(),
                translation: [lit(offset), lit(0.0), lit(0.0)],
                rotation: [lit(0.0), lit(0.0), lit(0.0)],
                scale: lit(1.0),
                about: fieldform_scene::Place::origin(),
            });
        }
        let program = generate(&fieldform_scene::Document::new(), &graph)
            .expect("one body under two placements is expressible");

        // Two placements, so two points and two copies of the sphere's expression. Counting
        // them is what distinguishes this from a shader that quietly evaluated the body once
        // and placed it twice, which would put both copies in the same place.
        assert_eq!(field_body(&program.source).matches("sphere_distance(").count(), 2);
        assert!(program.source.contains("let p1 ="));
        assert!(program.source.contains("let p2 ="));
        let _ = sphere;
    }

    /// The mixed case, and the one a user actually reached: a body consumed once from *inside*
    /// a placement and once from outside it. The two-placements case is the obvious one; this
    /// is the one that arrives by accident, because an operand picker that offers every node in
    /// the part will happily offer the inside of a placed block.
    ///
    /// It has to be refused for the same reason: the sphere would be sampled at the placed
    /// point for the blend and at the world point for the union, and each node is emitted once.
    #[test]
    fn a_body_used_both_inside_and_outside_a_placement_is_emitted_at_both_points() {
        let mut graph = Graph::empty();
        let cage = graph.add(Node::Cage(CageId::new()));
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        let placed = graph.add(Node::Transform {
            input: sphere.into(),
            translation: [lit(3.0), lit(0.0), lit(0.0)],
            rotation: [lit(0.0), lit(0.0), lit(0.0)],
            scale: lit(1.0),
            about: fieldform_scene::Place::origin(),
        });
        let blended = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: cage.into(),
            b: placed.into(),
            fillet: Fillet { radius: lit(1.0), continuity: Continuity::G1, second: None },
        });
        // And now the same sphere again, from outside the placement.
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![blended.into(), sphere.into()],
        });

        let program = generate(&fieldform_scene::Document::new(), &graph)
            .expect("a body inside and outside a placement is expressible");

        // Once at the caller's own point and once at the placement's, which is exactly what
        // the CPU field does by recursing into the same node twice.
        assert_eq!(field_body(&program.source).matches("sphere_distance(").count(), 2);
        let _ = sphere;
    }

    /// A graded lattice is the ordinary lattice call with the ramp's expression where the
    /// sizing slot was read, over the field's own divisor. The slot stays, holding nothing, so
    /// the normal pass counts the same eight; and an aimed one is refused here as it is there.
    #[test]
    fn a_graded_lattice_writes_its_ramp_where_the_slot_was_read_and_divides() {
        let graded = |frame: fieldform_scene::Frame| {
            let mut graph = Graph::empty();
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At([lit(0.0); 3]),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let ramp = graph.add(Node::Ramp {
                input: plane.into(),
                in_min: lit(-40.0),
                in_max: lit(40.0),
                out_min: lit(1.0),
                out_max: lit(3.0),
                shape: fieldform_scene::RampShape::Linear,
            });
            let lattice = graph.add(Node::Lattice {
                frame,
                cell: fieldform_scene::LatticeCell::Gyroid,
                map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0) },
                phase: fieldform_scene::LatticePhase::sheet(ParamRef::Field(ramp)),
            });
            (graph, lattice)
        };
        let doc = fieldform_scene::Document::new();

        let (graph, lattice) = graded(fieldform_scene::Frame::default());
        let program = generate(&doc, &graph).expect("a graded lattice generates a shader");
        let divisor = fieldform_field::driven_lattice_divisor(&doc, &graph, lattice);
        assert!(divisor > 1.0, "a grade divides: {divisor}");
        let body = field_body(&program.source);
        assert!(body.contains(&format!("/ {divisor:?}")), "the field's own divisor is written");
        assert!(
            !program.params.iter().any(|p| matches!(p, ParamRef::Field(_))),
            "the sizing slot holds a placeholder, never the reference: {:?}",
            program.params
        );
        validate(&program.source).expect("the graded lattice's shader parses and validates");

        let aimed = fieldform_scene::Frame::At {
            origin: [lit(0.0); 3],
            axis: [lit(1.0), lit(0.0), lit(0.0)],
        };
        let (graph, _) = graded(aimed);
        assert_eq!(
            generate(&doc, &graph).err(),
            Some(CodegenError::GradedLatticeCannotBeAimed),
            "the shader refuses what the field refuses"
        );
    }

    /// Three bindings in the order the shader reads them -- profile, spine, frame -- one slot
    /// for whether the spine closes, and the stretch written as a literal from the graph: the
    /// same figure the CPU leaf divides by, so the two paths cannot disagree about it.
    #[test]
    fn a_sweep_binds_its_profile_spine_and_frame_and_its_shader_validates() {
        let mut doc = fieldform_scene::Document::new();
        let square = fieldform_scene::Curve::polyline(
            CurveId::new(),
            "Square",
            vec![
                nalgebra::Point3::new(-10.0, -10.0, 0.0),
                nalgebra::Point3::new(10.0, -10.0, 0.0),
                nalgebra::Point3::new(10.0, 10.0, 0.0),
                nalgebra::Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        )
        .expect("closed");
        let path = fieldform_scene::Curve::polyline(
            CurveId::new(),
            "Elbow",
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(0.0, 0.0, 40.0),
                nalgebra::Point3::new(0.0, 40.0, 80.0),
            ],
            false,
        )
        .expect("open");
        let (profile, spine) = (square.id, path.id);
        doc.curves.insert(profile, square);
        doc.curves.insert(spine, path);
        let mut graph = Graph::empty();
        let sweep = graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });

        let program = generate(&doc, &graph).expect("a sweep generates a shader");
        assert_eq!(
            program.curves,
            vec![CurveUse::Profile(sweep), CurveUse::Spine(spine), CurveUse::Frame(sweep)],
            "profile, spine, frame, in that order"
        );
        let stretch = fieldform_field::sweep_stretch(&doc, &fieldform_scene::Outline::closed(profile), spine, &Default::default());
        assert!(stretch > 1.0, "a bent spine costs something: {stretch}");
        let (reach, tube) = fieldform_field::sweep_tube(&doc, &fieldform_scene::Outline::closed(profile), spine, &Default::default(), &Default::default())
            .expect("a 45 degree elbow is nowhere near closing the tube of a 20 mm square");
        // Fourteen slots, all in the buffer and none in the source, so that reshaping the spine,
        // moving the profile or changing how the section is held is a buffer write and not a
        // shader: the spine is open; the bend is the graph's own number, with the section's
        // reach and the spine's tube beside it, since a square section under the plain law
        // reads as it is within that tube (`sweep-spec.md` section 4); three say how far the
        // solid is moved to start at its profile, nought here because this profile is drawn
        // about the spine's start; and seven say how the section is held -- two axes and the
        // code saying how it is held. The axes are the profile's own in the world -- the XY
        // plane here, so `(1,0,0)` and `(0,1,0)` -- and `0` is square to the path, which ignores
        // them.
        let moved = fieldform_field::sweep_offset(&doc, &fieldform_scene::Outline::closed(profile), spine).expect("both curves");
        assert!(moved.norm() < 1e-9, "this profile sits on its spine's start: {moved:?}");
        let (ua, vb) = fieldform_field::sweep_axes(&doc, &fieldform_scene::Outline::closed(profile), spine).expect("a planar profile");
        // This profile is drawn in the XY plane, so its own axes are the world's.
        assert!((ua - nalgebra::Vector3::x()).norm() < 1e-9);
        assert!((vb - nalgebra::Vector3::y()).norm() < 1e-9);
        assert_eq!(
            program.params,
            vec![
                ParamRef::Literal(0.0),
                ParamRef::Literal(stretch),
                ParamRef::Literal(reach),
                ParamRef::Literal(tube),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(1.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(1.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ],
            "open, the stretch, the reach and the tube, the move to the profile, then the profile's axes and how the              section is held (0 = square)"
        );
        assert!(!program.source.contains(&format!("{stretch:?}")), "and not as a literal");
        validate(&program.source).expect("the sweep's shader parses and validates");
    }

    /// **A loft's stations are slots, and so is its smoothness, so neither a dragged section
    /// nor the G0/G1 toggle rebuilds the shader** -- the owner's ruling over `loft-spec.md`
    /// section 8, which had them as literals. Two slots per section, the station and the
    /// binding its projected section is read from, then the smoothness and the bound; the
    /// normal pass counts the same eight and names the same table, and the two smoothnesses
    /// generate one source.
    #[test]
    fn a_loft_writes_its_stations_as_slots_and_its_smoothness_rebuilds_no_shader() {
        use fieldform_scene::LoftSmooth;
        let mut doc = fieldform_scene::Document::new();
        let ring = |r: f64, z: f64| -> Vec<nalgebra::Point3<f64>> {
            (0..32)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / 32.0;
                    nalgebra::Point3::new(r * a.cos(), r * a.sin(), z)
                })
                .collect()
        };
        let mut add = |r: f64, z: f64| {
            let curve =
                fieldform_scene::Curve::polyline(CurveId::new(), "ring", ring(r, z), true)
                    .expect("closed");
            let id = curve.id;
            doc.curves.insert(id, curve);
            id
        };
        let sections: Vec<fieldform_scene::Outline> = [(20.0, 0.0), (12.0, 30.0), (8.0, 60.0)]
            .into_iter()
            .map(|(r, z)| fieldform_scene::Outline::closed(add(r, z)))
            .collect();
        let built = |smooth: LoftSmooth| {
            let mut graph = Graph::empty();
            let node = graph.add(Node::Loft { rims: Default::default(), sections: sections.clone(), spine: None, smooth });
            (generate(&doc, &graph).expect("a loft generates a shader"), node)
        };

        let (g0, node) = built(LoftSmooth::G0);
        let (g1, smooth_node) = built(LoftSmooth::G1);
        assert_eq!(g0.source, g1.source, "the smoothness is a slot, so the source is one source");

        // The bindings: the centerline and its frame keyed by the node, then one per section --
        // and at G1 the rails' slopes beside them.
        assert_eq!(
            g0.curves,
            vec![
                CurveUse::LoftSpine(node),
                CurveUse::Frame(node),
                CurveUse::LoftSection(node, 0),
                CurveUse::LoftSection(node, 1),
                CurveUse::LoftSection(node, 2),
            ]
        );
        // At G1 each section's slopes are interned right after it, so the two sit side by side.
        assert_eq!(
            g1.curves[2..],
            [
                CurveUse::LoftSection(smooth_node, 0),
                CurveUse::LoftSlope(smooth_node, 0),
                CurveUse::LoftSection(smooth_node, 1),
                CurveUse::LoftSlope(smooth_node, 1),
                CurveUse::LoftSection(smooth_node, 2),
                CurveUse::LoftSlope(smooth_node, 2),
            ]
        );
        // The slots: `[station, binding, slopes]` per section, the smoothness, the bound.
        assert_eq!(g0.params.len(), 11, "three per section, then the smoothness and the bound");
        let value = |program: &ShaderProgram, k: usize| match program.params[k] {
            ParamRef::Literal(v) => v,
            ref other => panic!("slot {k} is {other:?}, not a number"),
        };
        // Three coaxial rings derive a straight centerline, so the stations are the rings'
        // heights, and each names the binding of its own section -- and its slopes' at G1,
        // its own again at G0.
        for (k, height) in [0.0, 30.0, 60.0].into_iter().enumerate() {
            assert!((value(&g0, 3 * k) - height).abs() < 1e-6, "station {k}: {}", value(&g0, 3 * k));
            assert_eq!(value(&g0, 3 * k + 1) as usize, 2 + k, "section {k}'s binding");
            assert_eq!(value(&g0, 3 * k + 2) as usize, 2 + k, "G0 names the section again");
            assert_eq!(value(&g1, 3 * k + 1) as usize, 2 + 2 * k, "G1's section, beside its slopes");
            assert_eq!(value(&g1, 3 * k + 2) as usize, 3 + 2 * k, "G1 names the slopes");
        }
        assert_eq!((value(&g0, 9), value(&g1, 9)), (0.0, 1.0), "G0 then G1");
        let stretch = fieldform_field::loft_stretch(&doc, &sections, None, LoftSmooth::G0);
        assert_eq!(value(&g0, 10), stretch, "the bound the leaf divides by");
        assert!(!g0.source.contains(&format!("{stretch:?}")), "and not as a literal");

        // Both passes read the table from the same slot with the same count, and the value
        // pass's call is the plain one.
        let tail = ", 0u, 3u, params[9], params[10])";
        assert!(
            field_body(&g0.source).contains("loft_distance(0u, 1u, p")
                && field_body(&g0.source).contains(tail),
            "{}",
            field_body(&g0.source)
        );
        assert_eq!(g0.source.matches(tail).count(), 2, "once per pass");
        assert!(g0.source.contains("loft_value_normal_bake(0u, 1u, "), "the normal pass walks once");
        validate(&g0.source).expect("the loft's shader parses and validates");

        // **And with the table somewhere other than slot nought.** Alone, a miscount between
        // the two passes cannot show; after a sphere's four slots it would read the sphere's
        // radius as a station. From the table on, the two calls must name the same slots.
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(15.0),
        });
        let loft = graph.add(Node::Loft { rims: Default::default(), sections: sections.clone(), spine: None, smooth: LoftSmooth::G1 });
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![ball.into(), loft.into()] });
        let joined = generate(&doc, &graph).expect("a loft beside a sphere generates");
        // In the generated part, past the prelude that defines the functions.
        let generated = joined
            .source
            .split("// ---- generated from the composition graph ----")
            .nth(1)
            .expect("a generated section");
        let from_the_table = |name: &str| -> String {
            let start = generated.find(name).expect(name) + name.len();
            let call = &generated[start..];
            let call = &call[..call.find(')').expect("the call closes")];
            // spine, frame, point, then the table, the count and the two slots.
            call.splitn(4, ", ").nth(3).expect("seven arguments").to_string()
        };
        // The union's operands are emitted back to front, so the loft comes first and its
        // table still starts at nought here -- what this holds is that the two passes agree
        // wherever the table lands, which is the claim; the sphere's slots follow the loft's.
        assert_eq!(from_the_table("loft_distance("), from_the_table("loft_value_normal_bake("));
        assert_eq!(joined.params.len(), 15, "the loft's eleven and the sphere's four");
        validate(&joined.source).expect("the joined shader parses and validates");
    }

    /// **A loft that names a pipe's curve as its centerline leaves the pipes a run.** The loft
    /// binds its centerline as `LoftSpine(node)` and not as that curve's `Spine`, so the pipes'
    /// spine count stays one each and `looped_run` still loops them -- and the curve is
    /// uploaded twice, once as the pipes walk it and once as the loft flattened it, which is
    /// what keeps the stations in `params` arcs along the polyline the loft walks. Held by a
    /// test because a comment is all that held it: interning the named spine as `Spine(curve)`
    /// to save an upload would break the run and the stations at once and stay green.
    #[test]
    fn a_loft_on_a_pipes_curve_leaves_the_pipes_a_run_and_binds_its_own_spine() {
        let (mut doc, mut graph) = spokes(RUN_FLOOR);
        // The spoke along +x, which the rings below sit beside; the map's order is nobody's,
        // and the spoke along -x has a `y` of a few units in the sixteenth place too.
        let spoke = *doc
            .curves
            .iter()
            .find(|(_, c)| c.control_points()[0].y.abs() < 1e-9 && c.control_points()[0].x > 0.0)
            .map(|(id, _)| id)
            .expect("a spoke along x");
        let ring = |r: f64, z: f64| -> Vec<nalgebra::Point3<f64>> {
            (0..24)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / 24.0;
                    nalgebra::Point3::new(60.0 + r * a.cos(), r * a.sin(), z)
                })
                .collect()
        };
        let mut sections = Vec::new();
        for (r, z) in [(8.0, 0.0), (5.0, 20.0)] {
            let curve = fieldform_scene::Curve::polyline(CurveId::new(), "ring", ring(r, z), true)
                .expect("closed");
            let id = curve.id;
            doc.curves.insert(id, curve);
            sections.push(fieldform_scene::Outline::closed(id));
        }
        let loft = graph.add(Node::Loft { rims: Default::default(),
            sections,
            spine: Some(spoke),
            smooth: fieldform_scene::LoftSmooth::G0,
        });
        let program = generate(&doc, &graph).expect("generates");
        assert_eq!(
            field_body(&program.source).matches("pipe_run_distance(").count(),
            1,
            "the pipes are still one loop"
        );
        assert_eq!(
            program.curves.iter().filter(|c| **c == CurveUse::Spine(spoke)).count(),
            1,
            "the spoke is bound once for the pipes"
        );
        assert!(program.curves.contains(&CurveUse::LoftSpine(loft)), "and once more as the loft's own");
    }

    /// **A sweep that has not asked for a law compiles to the call it always was, byte for
    /// byte** -- `sweep-v2-spec.md` section 8's first acceptance, and the same promise a full
    /// revolve makes. One that has asked is a different call with six more slots, so setting a
    /// law rebuilds the shader and dragging any of its numbers refills a buffer. Seven of them:
    /// the taper and its amount, the twist, the shape, the window's two ends, and whether a
    /// draft mitres its corners.
    #[test]
    fn a_plain_law_generates_what_it_always_did_and_a_set_one_seven_more_slots() {
        use fieldform_scene::{RampShape, SweepLaw, Taper};
        let lit = ParamRef::Literal;
        let built = |law: SweepLaw| {
            let mut doc = fieldform_scene::Document::new();
            let square = fieldform_scene::Curve::polyline(
                CurveId::new(),
                "Section",
                vec![
                    nalgebra::Point3::new(-5.0, -5.0, 0.0),
                    nalgebra::Point3::new(5.0, -5.0, 0.0),
                    nalgebra::Point3::new(5.0, 5.0, 0.0),
                    nalgebra::Point3::new(-5.0, 5.0, 0.0),
                ],
                true,
            )
            .expect("closed");
            let path = fieldform_scene::Curve::polyline(
                CurveId::new(),
                "Elbow",
                vec![
                    nalgebra::Point3::new(0.0, 0.0, 0.0),
                    nalgebra::Point3::new(0.0, 0.0, 40.0),
                    nalgebra::Point3::new(0.0, 40.0, 80.0),
                ],
                false,
            )
            .expect("open");
            let (profile, spine) = (square.id, path.id);
            doc.curves.insert(profile, square);
            doc.curves.insert(spine, path);
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(profile),
                spine,
                section: Default::default(),
                law,
            });
            generate(&doc, &graph).expect("a sweep generates a shader")
        };

        let plain = built(SweepLaw::default());
        assert!(
            field_body(&plain.source).contains("sweep_distance(")
                && !field_body(&plain.source).contains("sweep_law_"),
            "the plain law must compile to the call it always was"
        );
        assert_eq!(plain.params.len(), 14, "and to its fourteen slots");

        for (what, law, kind, amount, shape) in [
            ("a scale", SweepLaw { taper: Taper::Scale(lit(0.5)), ..Default::default() }, 1.0, 0.5, 0.0),
            (
                "a draft",
                SweepLaw { taper: Taper::Draft(lit(5.0)), ..Default::default() },
                2.0,
                5f64.to_radians().tan(),
                0.0,
            ),
            (
                "a twist, eased",
                SweepLaw { twist: lit(90.0), shape: RampShape::Smooth, ..Default::default() },
                0.0,
                0.0,
                1.0,
            ),
        ] {
            let under = built(law);
            assert!(
                field_body(&under.source).contains("sweep_law_distance(")
                    && under.source.contains("= sweep_law_value_normal_bake("),
                "{what}: a law'd sweep calls the law'd functions"
            );
            assert_eq!(under.params.len(), 21, "{what}: seven more slots than the plain fourteen");
            // A law that moves the section has no tube, so both of its slots say so, and the
            // whole reading is divided as it was.
            assert_eq!(
                under.params[2..4],
                [ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
                "{what}: no tube under a law that moves the section"
            );
            // The six, in the order the shader's own `SweepLaw` reads them.
            let six = &under.params[14..];
            assert_eq!(six[0], ParamRef::Literal(kind), "{what}: which taper");
            assert!(
                matches!(six[1], ParamRef::Literal(v) if (v - amount).abs() < 1e-12),
                "{what}: the taper's amount, as the kernel reads it: {:?}",
                six[1]
            );
            assert_eq!(six[3], ParamRef::Literal(shape), "{what}: the law's shape");
            // The window is the whole path, in millimetres of arc rather than percent: this elbow
            // is 40 mm up and then 40 each way, which flattens to 96.57 mm.
            assert_eq!(six[4], ParamRef::Literal(0.0));
            assert!(
                matches!(six[5], ParamRef::Literal(v) if (v - 96.568_542_494_923_8).abs() < 1e-9),
                "{what}: the window's end is the path's length: {:?}",
                six[5]
            );
            assert!(
                !field_body(&under.source).contains("90.0")
                    && !field_body(&under.source).contains("0.5,"),
                "{what}: and none of them is a literal in the source"
            );
            validate(&under.source).expect("a law'd sweep's shader parses and validates");
        }
    }

    #[test]
    fn a_node_this_build_cannot_read_generates_nothing_and_says_which() {
        // The GPU path has to refuse what the CPU path refuses. A shader that quietly left
        // the node out would still render — the wrong shape, with nothing to notice.
        let mut graph = cage_graph();
        graph.add(
            serde_json::from_str(r#"{"kind": "acme.lattice", "cell": 5.0}"#)
                .expect("an unreadable node deserializes to Unknown rather than failing"),
        );

        let Err(error) = generate(&fieldform_scene::Document::new(), &graph) else {
            panic!("a graph holding a node this build cannot read must not generate a shader");
        };
        assert!(matches!(error, CodegenError::UnreadableNode(_)));
        assert!(error.to_string().contains("acme.lattice"), "names the kind: {error}");
    }

    /// A part with an empty slot in it generates no shader, and the refusal names the block
    /// whose slot is empty.
    ///
    /// **The hazard the refusal above answers, from the other direction.** There is no number a
    /// missing input could supply that keeps `distance` from overestimating, so a shader written
    /// with the block left out would draw the wrong shape with nothing to notice -- and the CPU
    /// compile refuses the same part, by the same name, for the same stated reason.
    ///
    /// Four graphs, because the refusal is of the *part* rather than of whatever met the hole:
    /// a holed body refuses by its own name; a hole under a tower of blocks still names the slot
    /// to fill rather than the body at the top; a ramp with nothing to read is named when
    /// something drives a number from it, which is the case the ramp's own refusal used to cover
    /// on its own; and a holed block nothing reads is nobody's problem, which is what a block
    /// placed and not yet wired has been since `ramp-decision.md` §7.2.
    #[test]
    fn the_shader_refuses_an_empty_slot_by_name() {
        let doc = fieldform_scene::Document::new();
        let centre = || fieldform_scene::Place::At([lit(0.0); 3]);
        let ball = |graph: &mut Graph| {
            graph.add(Node::Sphere { centre: centre(), radius: lit(10.0) })
        };
        let refusal = |graph: &Graph, what: &str| -> CodegenError {
            match generate(&doc, graph) {
                Err(error) => error,
                Ok(program) => {
                    panic!("a shader was generated for {what}: {}", program.source)
                }
            }
        };

        // A union of a ball and nothing: the block that is holed is the body itself.
        let mut graph = Graph::empty();
        let only = ball(&mut graph);
        let holed = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![only.into(), Operand::empty()],
        });
        let error = refusal(&graph, "a body with an empty slot");
        assert_eq!(
            error,
            CodegenError::EmptySlot(holed),
            "the block with the hole, named: {error}"
        );
        assert!(
            error.to_string().contains(&format!("{holed:?}")),
            "and named in what the refusal says, which is what reaches the panel: {error}"
        );

        // The same hole with two blocks standing on it. `Graph::incomplete` is closed
        // downstream, so all three are unwritable -- and the one to fill is still the boolean.
        let mut graph = Graph::empty();
        let only = ball(&mut graph);
        let holed = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![only.into(), Operand::empty()],
        });
        let shrunk = graph.add(Node::Offset { input: holed.into(), distance: lit(1.0) });
        graph.add(Node::Offset { input: shrunk.into(), distance: lit(1.0) });
        assert_eq!(
            refusal(&graph, "a body built on an empty slot"),
            CodegenError::EmptySlot(holed),
            "the slot to fill, not the body that could not be written"
        );

        // A ramp with nothing to read, driving an offset's distance: the case the generator's
        // one ramp-shaped refusal used to answer, now answered by every slot's.
        let mut graph = Graph::empty();
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: centre(),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: Operand::empty(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(0.0),
            out_max: lit(5.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        graph.add(Node::Offset { input: cube.into(), distance: ParamRef::Field(ramp) });
        assert_eq!(
            refusal(&graph, "an offset driven by a ramp with nothing to read"),
            CodegenError::EmptySlot(ramp),
            "the ramp is the block to fill"
        );

        // And the line that is drawn: a holed block nothing reads is not written, and the body
        // beside it is. A ramp is never a body, so nothing is reading this one.
        let mut graph = Graph::empty();
        graph.add(Node::Cuboid { edge: Default::default(),
            centre: centre(),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Ramp {
            input: Operand::empty(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(0.0),
            out_max: lit(5.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        let program = generate(&doc, &graph)
            .expect("a block nothing reads is not the part's problem");
        validate(&program.source).expect("and what is written is a valid shader");
    }

    /// The runtime has to be valid on its own, or every failure below is really this one.
    #[test]
    fn the_prelude_is_valid_wgsl() {
        // A trivial entry point, since a module of nothing but functions is otherwise unused.
        let source = format!(
            "{PRELUDE}\n@compute @workgroup_size(1) fn main() {{ \
             _ = smooth_min(1.0, 2.0, 0.5, 2.0); }}\n"
        );
        validate(&source).unwrap();
    }

    /// Every node kind, composed, parsed and validated as a real shader.
    #[test]
    fn a_composed_graph_generates_valid_wgsl() {
        let cage = CageId::new();
        let curve = CurveId::new();

        let mut graph = Graph::from_cage(cage);
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe { curve, radius: ParamRef::Literal(5.0) });
        let cut = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![cage_node.into(), pipe.into()],
        });
        let second = graph.add(Node::Pipe { curve, radius: ParamRef::Literal(2.0) });
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: cut.into(),
            b: second.into(),
            fillet: Fillet { radius: ParamRef::Literal(3.0), continuity: Continuity::G1, second: None },
        });

        let program = generate(&fieldform_scene::Document::new(), &graph).unwrap();
        validate(&program.source).unwrap();

        assert_eq!(program.baked, vec![BakedLeaf::Cage(cage)]);
        // One curve used twice binds once.
        assert_eq!(program.curves, vec![CurveUse::Spine(curve)]);
        assert_eq!(program.params.len(), 3);
    }

    /// The property the whole design exists for.
    ///
    /// If a literal were compiled into the shader, dragging a slider on an unnamed radius would
    /// mean regenerating and recompiling — which is the interaction the parametric layer is
    /// supposed to make free. Two graphs differing only in their numbers must produce identical
    /// source.
    #[test]
    fn changing_a_number_does_not_change_the_shader() {
        let (cage, curve) = (CageId::new(), CurveId::new());

        let build = |radius: f64| {
            let mut graph = Graph::from_cage(cage);
            let cage_node = graph.roots()[0];
            let pipe = graph.add(Node::Pipe { curve, radius: ParamRef::Literal(radius) });
            graph.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: vec![cage_node.into(), pipe.into()],
            });
            generate(&fieldform_scene::Document::new(), &graph).unwrap()
        };

        let a = build(5.0);
        let b = build(9000.0);
        assert_eq!(a.source, b.source, "the shader must not depend on parameter values");
        assert_eq!(a.params, vec![ParamRef::Literal(5.0)]);
        assert_eq!(b.params, vec![ParamRef::Literal(9000.0)]);

        // And the literal must not appear in the *generated* part, which is the direct form
        // of the same assertion. The prelude is excluded because it is a fixed body of code
        // that legitimately contains numbers of its own — a smootherstep's coefficients, for
        // one — and searching the whole source for a bare "5.0)" matched "- 15.0)" there.
        let generated = a.source.strip_prefix(prelude()).expect("the prelude comes first");
        assert!(!generated.contains("5.0)"), "the radius was inlined into the shader");
    }

    /// What keeps a join from overstating is decided from the radius **at run time**, so the
    /// radius stays a buffer write (blend-integrity-spec §3).
    ///
    /// Counted rather than compared pairwise: every radius dragged through, over a single blend,
    /// a blend of blends and a rounded pattern, must come to exactly one shader each — nought
    /// included for the blends, since anything about the radius written as a literal would
    /// have had to change the moment a fillet went hard. And each shader has to actually carry
    /// what bounds it, or the count would be one for the wrong reason: a blend is bounded by
    /// `blend_bound`, which divides nothing and so compounds nothing down a chain, and a rounded
    /// pattern still by its divisor.
    #[test]
    fn dragging_a_joins_radius_compiles_one_shader_and_the_join_is_bounded() {
        let slab = |graph: &mut Graph, x: f64| {
            graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(-10.0)]),
                size: [lit(50.0), lit(40.0), lit(20.0)],
                turn: fieldform_scene::no_turn(),
            })
        };
        let fillet = |k: f64| Fillet { radius: lit(k), continuity: Continuity::G1, second: None };
        let single = |k: f64| {
            let mut graph = Graph::empty();
            let (a, b) = (slab(&mut graph, -15.0), slab(&mut graph, 15.0));
            graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet: fillet(k) });
            graph
        };
        let chained = |k: f64| {
            let mut graph = single(k);
            let inner = graph.roots()[0];
            let boss = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(4.0)]),
                radius: lit(9.0),
            });
            graph.add(Node::Blend { op: BooleanOp::Union, a: inner.into(), b: boss.into(), fillet: fillet(k) });
            graph
        };
        let patterned = |k: f64| {
            let mut graph = Graph::empty();
            let body = slab(&mut graph, 0.0);
            graph.add(Node::Pattern {
                options: Default::default(),
                input: body.into(),
                count: ParamRef::Literal(7.0),
                kind: fieldform_scene::PatternKind::Linear {
                    step: [lit(10.0), lit(0.0), lit(0.0)],
                },
                join: true,
                fillet: fillet(k),
            });
            graph
        };

        let doc = fieldform_scene::Document::new();
        let shaders = |build: &dyn Fn(f64) -> Graph, radii: &[f64]| {
            let sources: std::collections::HashSet<String> = radii
                .iter()
                .map(|k| generate(&doc, &build(*k)).expect("generates").source)
                .collect();
            sources
        };
        for (what, build, radii, call) in [
            ("a blend", &single as &dyn Fn(f64) -> Graph, &[0.0, 0.5, 5.0, 12.0][..], "blend_bound("),
            ("a blend of blends", &chained, &[0.0, 0.5, 5.0, 12.0][..], "blend_bound("),
            // Not nought: a literal zero is a *hard* fold, which is a different graph shape by
            // `Fillet::is_plainly_hard` and always was.
            ("a rounded pattern", &patterned, &[0.5, 2.0, 5.0, 12.0][..], "pattern_join_divisor("),
        ] {
            let sources = shaders(build, radii);
            assert_eq!(sources.len(), 1, "dragging the radius of {what} recompiled");
            let source = sources.into_iter().next().expect("one");
            let generated = source.strip_prefix(PRELUDE).expect("the prelude comes first");
            assert!(generated.contains(call), "{what} is not bounded in its shader");
            // And a blend outside a part with critical marks divides by nothing at all.
            assert!(
                !generated.contains("blend_divisor("),
                "{what} still divides, which is what compounded down a chain"
            );
            validate(&source).unwrap_or_else(|e| panic!("{what}: {e}"));
        }
    }

    /// A part with a critical bore, its fillets nested every way a fillet can be: at the world
    /// point, under a placement, under a joined mirror, and as a rounded pattern.
    fn protected_part(k: f64, clearance: f64, marked: bool) -> Graph {
        let fillet = |k: f64| Fillet { radius: lit(k), continuity: Continuity::G1, second: None };
        let mut graph = Graph::empty();
        let block = |graph: &mut Graph, x: f64| {
            graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(-10.0)]),
                size: [lit(50.0), lit(40.0), lit(20.0)],
                turn: fieldform_scene::no_turn(),
            })
        };
        let (a, b) = (block(&mut graph, -15.0), block(&mut graph, 15.0));
        let body = graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet: fillet(k) });
        let placed = graph.add(Node::Transform {
            input: body.into(),
            translation: [lit(1.0), lit(2.0), lit(3.0)],
            rotation: [lit(10.0), lit(20.0), lit(30.0)],
            scale: lit(1.0),
            about: fieldform_scene::Place::origin(),
        });
        let both = graph.add(Node::Mirror {
            input: placed.into(),
            frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Y, [lit(0.0); 3]),
            join: true,
        });
        let lug = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(-30.0), lit(0.0), lit(6.0)]),
            radius: lit(8.0),
        });
        let lugs = graph.add(Node::Pattern {
            options: Default::default(),
            input: lug.into(),
            // Three, so the window covers the pattern at every radius dragged through: whether it
            // does is a generated branch, and was before any of this.
            count: ParamRef::Literal(3.0),
            kind: fieldform_scene::PatternKind::Linear { step: [lit(10.0), lit(0.0), lit(0.0)] },
            join: true,
            fillet: fillet(k),
        });
        let lugged =
            graph.add(Node::Blend { op: BooleanOp::Union, a: both.into(), b: lugs.into(), fillet: fillet(k) });
        let bore = graph.add(Node::Cylinder { edge: Default::default(),
            ends: fieldform_scene::Ends::between(
                fieldform_scene::Place::At([lit(2.0), lit(1.0), lit(-40.0)]),
                fieldform_scene::Place::At([lit(2.0), lit(1.0), lit(40.0)]),
            ),
            radius: lit(5.0),
        });
        graph.add(Node::Blend { op: BooleanOp::Difference, a: lugged.into(), b: bore.into(), fillet: fillet(k) });
        if marked {
            graph.mark_critical(fieldform_scene::Critical {
                clearance: lit(clearance),
                ..fieldform_scene::Critical::new(bore)
            });
        }
        graph
    }

    /// §6.4's two promises about the shader, counted: **dragging `k` — or the clearance — on a
    /// protected blend is a buffer write, and adding or removing a mark is a rebuild.** And the
    /// protected shader has to validate with its fillets under every kind of frame, since the
    /// gap reaches them through a module variable rather than through any of their signatures.
    #[test]
    fn dragging_k_on_a_protected_blend_does_not_recompile() {
        let doc = fieldform_scene::Document::new();
        let source = |k: f64, c: f64, marked: bool| {
            generate(&doc, &protected_part(k, c, marked)).expect("generates").source
        };
        let dragged: std::collections::HashSet<String> =
            [(0.5, 0.0), (3.0, 0.0), (9.0, 0.0), (3.0, 0.75), (3.0, 4.0)]
                .into_iter()
                .map(|(k, c)| source(k, c, true))
                .collect();
        assert_eq!(dragged.len(), 1, "dragging a protected radius or a clearance recompiled");

        let protected = dragged.into_iter().next().expect("one");
        validate(&protected).expect("the protected shader validates");
        let generated = protected.strip_prefix(PRELUDE).expect("the prelude comes first");
        for needed in [
            "fn critical_gap(",
            "crit_gap = critical_gap(p);",
            "smooth_min_tapering(",
            "blend_taper_divisor(",
            "pattern_taper_divisor(",
            "taper_hit",
        ] {
            assert!(generated.contains(needed), "the protected shader has no `{needed}`");
        }
        // No fillet left on the untapered path: one would be a fillet that ignores the mark.
        assert!(!generated.contains("smooth_min("), "a blend was left untapered");

        let bare = source(3.0, 0.0, false);
        assert_ne!(bare, protected, "marking a surface critical has to rebuild the shader");
        let generated = bare.strip_prefix(PRELUDE).expect("the prelude comes first");
        assert!(
            !generated.contains("crit_gap") && !generated.contains("tapering"),
            "an unmarked part pays for protection it does not have"
        );
    }

    /// The divisor is the one number the two paths cannot be seen to disagree about — it moves
    /// no surface — so it is asked of one function, and this holds the shader to the answer.
    #[test]
    fn the_shader_divides_a_tapering_blend_by_the_bound_the_cpu_does() {
        let doc = fieldform_scene::Document::new();
        let graph = protected_part(3.0, 0.5, true);
        let source = generate(&doc, &graph).expect("generates").source;
        let mut blends = 0;
        for (id, node) in graph.nodes() {
            if node.fillet().is_none() {
                continue;
            }
            let rho = fieldform_field::taper_ratio(&doc, &graph, id);
            assert_eq!(rho, 1.0, "every leaf here moves at one per millimetre");
            blends += 1;
        }
        assert_eq!(blends, 4);
        assert_eq!(source.matches(", 2.0, 1.0))").count() >= 3, true, "the ratio was not written");
        // And at one, the blend's divisor is the spec's 2^(1+1/n) − 1.
        let d = 2f64.powf(0.5) + (2f64.powf(0.5) - 1.0) * 1.0;
        assert!((d - (2f64.powf(1.5) - 1.0)).abs() < 1e-12);
    }

    /// Changing the graph's *shape* is the one thing that does need a new program.
    #[test]
    fn changing_the_graph_changes_the_shader() {
        let (cage, curve) = (CageId::new(), CurveId::new());

        let build = |op: BooleanOp| {
            let mut graph = Graph::from_cage(cage);
            let cage_node = graph.roots()[0];
            let pipe = graph.add(Node::Pipe { curve, radius: ParamRef::Literal(5.0) });
            graph.add(Node::Boolean { op, inputs: vec![cage_node.into(), pipe.into()] });
            generate(&fieldform_scene::Document::new(), &graph).unwrap().source
        };

        assert_ne!(build(BooleanOp::Union), build(BooleanOp::Difference));
    }

    /// Named and literal parameters are both buffer slots, so a named one costs nothing extra
    /// and an unnamed one is no less live.
    #[test]
    fn named_and_literal_parameters_are_both_buffer_slots() {
        let param = fieldform_scene::ParamId::new();
        let mut graph = Graph::from_cage(CageId::new());
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe {
            curve: CurveId::new(),
            radius: ParamRef::Named(param),
        });
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![cage_node.into(), pipe.into()],
        });

        let program = generate(&fieldform_scene::Document::new(), &graph).unwrap();
        assert_eq!(program.params, vec![ParamRef::Named(param)]);
        assert!(program.field_function().contains("params[0]"));
    }

    /// **The surface walk is compiled only into programs that follow a surface.** It was in
    /// the prelude, reached through a runtime branch no compiler can remove, and it cost every
    /// lattice file about 1.3 seconds a cold pipeline whether anything in it was conformal or
    /// not — reported as File → Open hanging. Held both ways, because the failure in either
    /// direction is silent: left in, it is only slow; left out of a program that needs it, the
    /// shader names a function nothing defines.
    #[test]
    fn only_a_program_that_follows_a_surface_carries_the_surface_walk() {
        let lit = ParamRef::Literal;
        let lattice = |map: fieldform_scene::CellMap| {
            let mut graph = Graph::empty();
            graph.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell: fieldform_scene::LatticeCell::Gyroid,
                map,
                phase: fieldform_scene::LatticePhase::sheet(lit(1.0)),
            });
            graph
        };

        let plain = generate(
            &fieldform_scene::Document::new(),
            &lattice(fieldform_scene::CellMap::Rectangular { cell: lit(10.0) }),
        )
        .expect("a rectangular lattice generates");
        assert!(plain.surfaces.is_empty());
        assert!(
            !plain.source.contains("fn surface_at"),
            "a rectangular lattice carries the surface walk it never calls"
        );

        let mut doc = fieldform_scene::Document::new();
        let panel = {
            let mut positions = Vec::new();
            for j in 0..=2 {
                for i in 0..=2 {
                    positions.push(nalgebra::Point3::new(f64::from(i), f64::from(j), 0.0));
                }
            }
            let at = |i: u32, j: u32| j * 3 + i;
            let faces: Vec<Vec<u32>> = (0..2)
                .flat_map(|j| (0..2).map(move |i| vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]))
                .collect();
            fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a panel")
        };
        let surface = doc.add_quad_mesh(panel, "panel");
        // `generate` validates what it emits before returning, so an `Ok` here is also the proof
        // that the appended file defines everything the call names.
        let conformal = generate(
            &doc,
            &lattice(fieldform_scene::CellMap::Conformal {
                surface,
                cells_per_face: 2,
                depth: lit(5.0),
                thickness: Some(lit(2.0)),
            }),
        )
        .expect("a conformal lattice generates, and validates");
        assert_eq!(conformal.surfaces, vec![(surface, 2)]);
        assert!(conformal.source.contains("fn surface_at"));
        assert!(conformal.source.contains("conformal_lattice_distance("));
    }

    /// `n` bent pipes on their own spokes in one union — the shape `union_loop_cost` measures.
    fn spokes(n: usize) -> (fieldform_scene::Document, Graph) {
        let mut doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let mut pipes = Vec::new();
        for i in 0..n {
            let turn = std::f64::consts::TAU * i as f64 / n.max(1) as f64;
            let (c, s) = (turn.cos(), turn.sin());
            let id = fieldform_scene::CurveId::new();
            let at = |r: f64, z: f64| nalgebra::Point3::new(r * c, r * s, z);
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(
                    id,
                    "spine",
                    vec![at(20.0, 0.0), at(80.0, 25.0), at(140.0, -10.0)],
                    false,
                )
                .expect("a bent line"),
            );
            pipes.push(graph.add(Node::Pipe { curve: id, radius: lit(9.0) }));
        }
        if pipes.len() > 1 {
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: pipes.into_iter().map(Into::into).collect() });
        }
        (doc, graph)
    }

    /// `n` rectangular blades on their own spokes in one union — the second kind of run, and the
    /// shape `union_loop_cost` and `extrude_spokes_cost` both measure.
    fn blades(n: usize) -> (fieldform_scene::Document, Graph) {
        let mut doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let mut ids = Vec::new();
        for i in 0..n {
            let turn = std::f64::consts::TAU * i as f64 / n.max(1) as f64;
            let (c, s) = (turn.cos(), turn.sin());
            // Along the spoke by `r`, across it by `w`.
            let at = |r: f64, w: f64| nalgebra::Point3::new(r * c - w * s, r * s + w * c, 0.0);
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(
                    id,
                    "blade",
                    vec![at(20.0, -6.0), at(200.0, -6.0), at(200.0, 6.0), at(20.0, 6.0)],
                    true,
                )
                .expect("a rectangle"),
            );
            ids.push(graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(40.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                draft: lit(0.0),
                corners: Default::default(),
            }));
        }
        if ids.len() > 1 {
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.into_iter().map(Into::into).collect() });
        }
        (doc, graph)
    }

    /// The whole point of the loop: the source stops growing with the operand count.
    ///
    /// Not "is byte-identical", which it cannot be — the count's own slot sits after the run's
    /// radii, so its index moves with the run's length. One line either way is what a loop costs;
    /// what it must not do is grow by a walk an operand, which is what cost 123 seconds.
    #[test]
    fn a_union_of_pipes_writes_one_walk_however_many_pipes_it_takes() {
        let mut sizes = Vec::new();
        for n in [RUN_FLOOR, 8, 16, 32] {
            let (doc, graph) = spokes(n);
            let program = generate(&doc, &graph).expect("generates");
            let body = field_body(&program.source);
            assert_eq!(
                body.matches("pipe_run_distance(").count(),
                1,
                "one loop at {n} pipes, not one call each"
            );
            assert_eq!(
                body.matches("pipe_distance(").count(),
                0,
                "no pipe is written out beside the loop at {n} pipes"
            );
            sizes.push(program.field_function().lines().count());
        }
        // And below the floor it is written out as before, which is what the floor is for: the
        // frame regressed 12-15% at two and three pipes. See `RUN_FLOOR`'s own table.
        for n in 2..RUN_FLOOR {
            let (doc, graph) = spokes(n);
            let program = generate(&doc, &graph).expect("generates");
            let body = field_body(&program.source);
            assert_eq!(
                body.matches("pipe_run_distance(").count(),
                0,
                "{n} pipes is below the floor and is written out"
            );
            assert_eq!(body.matches("pipe_distance(").count(), n, "one call each at {n}");
        }
        let (least, most) = (sizes[0], *sizes.last().expect("rows"));
        assert!(
            most <= least + 2,
            "the generated source grew from {least} to {most} lines over 2 to 32 pipes, so \
             something is still being written out per operand"
        );
    }

    /// A pattern of a lug, then a sphere as a body of its own, so the sphere's slots come after
    /// everything the pattern pushes — which is where a miscount shows.
    fn pattern_then_sphere(
        kind: fieldform_scene::PatternKind,
        count: u32,
        join: bool,
        fillet: Fillet,
        options: fieldform_scene::PatternOptions,
    ) -> (fieldform_scene::Document, Graph) {
        let mut graph = Graph::empty();
        let lug = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(40.0), lit(0.0), lit(0.0)]),
            size: [lit(10.0), lit(8.0), lit(6.0)],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Pattern {
            input: lug.into(),
            count: lit(f64::from(count)),
            kind,
            join,
            fillet,
            options,
        });
        graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0), lit(70.0), lit(0.0)]),
            radius: lit(9.0),
        });
        (fieldform_scene::Document::new(), graph)
    }

    /// The first parameter slot a call reads, to compare what the two passes agree a body owns.
    fn first_slot_after(text: &str, call: &str) -> usize {
        let at = text.find(call).unwrap_or_else(|| panic!("{call} is not in the source"));
        text[at..]
            .split("params[")
            .nth(1)
            .and_then(|rest| rest.split(']').next())
            .and_then(|s| s.trim().parse().ok())
            .expect("a slot index")
    }

    /// **A pattern's options add slots, and the two passes must add the same number.** The value
    /// pass pushes them and the normal pass only counts, so a count that differs by one shifts every
    /// body after the pattern — here a sphere — and the picture is lit by numbers that mean
    /// something else, with nothing in the source to say so. Held for each shape of option, since
    /// each changes how many slots there are: nothing, only the first copy left out, a skip list
    /// of one word, and one of three.
    #[test]
    fn a_pattern_with_options_leaves_the_next_body_its_own_slots_in_both_passes() {
        use fieldform_scene::{Frame, PatternKind, PatternOptions, Spacing};
        let row = || PatternKind::Linear { step: [lit(30.0), lit(0.0), lit(0.0)] };
        let ring = || PatternKind::Radial { frame: Frame::default() };
        let cases: Vec<(&str, PatternKind, u32, bool, PatternOptions)> = vec![
            ("a plain row", row(), 6, true, PatternOptions::default()),
            ("a row leaving out its first copy", row(), 6, false, PatternOptions::default()),
            (
                "a row with one skip word",
                row(),
                9,
                true,
                PatternOptions { skip: vec![3], ..Default::default() },
            ),
            (
                "a centred whole run with three skip words",
                row(),
                60,
                true,
                PatternOptions {
                    spacing: Spacing::Extent,
                    symmetric: true,
                    skip: vec![3, 30, 55],
                    ..Default::default()
                },
            ),
            (
                "a grid walked by window both ways",
                row(),
                12,
                true,
                PatternOptions {
                    symmetric: true,
                    second: Some(fieldform_scene::PatternSecond {
                        step: [lit(0.0), lit(30.0), lit(5.0)],
                        count: lit(9.0),
                    }),
                    skip: vec![5, 30],
                    ..Default::default()
                },
            ),
            (
                "a grid short enough to be walked whole along its columns",
                row(),
                3,
                false,
                PatternOptions {
                    second: Some(fieldform_scene::PatternSecond {
                        step: [lit(0.0), lit(30.0), lit(5.0)],
                        count: lit(40.0),
                    }),
                    ..Default::default()
                },
            ),
            ("a plain ring", ring(), 8, true, PatternOptions::default()),
            (
                "an arc with a skip",
                ring(),
                12,
                true,
                PatternOptions { angle: lit(200.0), skip: vec![4], ..Default::default() },
            ),
            (
                "a centred ring leaving out the copy on the body",
                ring(),
                9,
                false,
                PatternOptions { symmetric: true, ..Default::default() },
            ),
        ];
        for (what, kind, count, join, options) in cases {
            let (doc, graph) = pattern_then_sphere(kind, count, join, Fillet::hard(), options);
            let program = generate(&doc, &graph).expect("generates");
            let generated = program
                .source
                .split("// ---- generated from the composition graph ----")
                .nth(1)
                .expect("a generated shader has a generated section");
            let split = generated
                .find("fn body_normal_")
                .or_else(|| generated.find("fn field_normal("))
                .expect("a normal pass");
            let (values, normals) = generated.split_at(split);
            let (a, b) =
                (first_slot_after(values, "sphere_distance("), first_slot_after(normals, "sphere_normal("));
            assert_eq!(a, b, "{what}: the value pass puts the sphere at slot {a}, the normal pass at {b}");
            assert!(a >= 8, "{what}: the sphere is at slot {a}, so the pattern's own slots were not counted first");
        }
    }

    /// **A skip list is paid for only by a pattern that has one.** Leaving out the first copy is
    /// what leaving out the body always meant and is a slot, not a loop; a pattern that skips
    /// nothing generates no test for it, so every document that predates options generates the loop
    /// it always did.
    #[test]
    fn only_a_pattern_that_skips_more_than_its_first_copy_pays_for_a_skip_test() {
        use fieldform_scene::{PatternKind, PatternOptions};
        let row = || PatternKind::Linear { step: [lit(30.0), lit(0.0), lit(0.0)] };
        let body_of = |join: bool, options: PatternOptions| {
            let (doc, graph) = pattern_then_sphere(row(), 9, join, Fillet::hard(), options);
            let program = generate(&doc, &graph).expect("generates");
            field_body(&program.source).matches("pattern_skipped(").count()
        };
        assert_eq!(body_of(true, PatternOptions::default()), 0, "a plain pattern");
        assert_eq!(body_of(false, PatternOptions::default()), 0, "only the first copy left out");
        assert!(
            body_of(true, PatternOptions { skip: vec![3], ..Default::default() }) >= 1,
            "a skipped copy is tested for"
        );
    }

    /// The skip list's encoding: copy `i` is bit `i % 24` of word `i / 24`, in words an `f32` holds
    /// exactly. Copies 3, 30 and 55 of sixty are bit 3 of word 0, bit 6 of word 1 and bit 7 of
    /// word 2 — 8, 64 and 128 — and the words sit together in the buffer, in that order.
    #[test]
    fn a_skip_list_is_written_as_exact_words_of_twenty_four_copies() {
        use fieldform_scene::{PatternKind, PatternOptions};
        let (doc, graph) = pattern_then_sphere(
            PatternKind::Linear { step: [lit(30.0), lit(0.0), lit(0.0)] },
            60,
            true,
            Fillet::hard(),
            PatternOptions { skip: vec![3, 30, 55], ..Default::default() },
        );
        let program = generate(&doc, &graph).expect("generates");
        let literals: Vec<Option<f64>> = program
            .params
            .iter()
            .map(|r| match r {
                ParamRef::Literal(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert!(
            literals.windows(3).any(|w| w == [Some(8.0), Some(64.0), Some(128.0)]),
            "no run of 8, 64, 128 in {literals:?}"
        );
    }

    /// **The failure this whole arrangement is exposed to.** The two passes agree about the
    /// parameter buffer by counting the same things in the same order, so a run taken in the value
    /// pass and not the normal pass — or one that pushes a different number of slots in each —
    /// shifts every parameter after it, and the picture is lit by numbers that mean something
    /// else. Nothing in the source says so; it simply looks wrong.
    ///
    /// So: the value pass's own count of slots, against the normal pass's, on the same graph.
    #[test]
    fn a_looped_run_pushes_the_same_slots_in_both_passes() {
        // **Both kinds**, and both sides of the floor, since the two passes have to agree either
        // way and the run is only one of the two things they count. An extrude's twelve slots a
        // member is the figure most exposed to this: it is written once in WGSL and once in
        // `RunKind::slots_each`, and nothing but this holds the two to the same number.
        let builders: [(&str, fn(usize) -> (fieldform_scene::Document, Graph)); 2] =
            [("pipes", spokes), ("blades", blades)];
        for (kind, build) in builders {
            for n in [2usize, 3, RUN_FLOOR, 5, 8, 16] {
                let (doc, graph) = build(n);
                let program = generate(&doc, &graph).expect("generates");
                // Every slot the normal pass reads must exist in the list the value pass built,
                // and the highest it reads must be the last — the two are one buffer.
                let highest = program
                    .source
                    .split("params[")
                    .skip(1)
                    .filter_map(|rest| rest.split(']').next())
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .max()
                    .expect("the shader reads parameters");
                assert_eq!(
                    highest + 1,
                    program.params.len(),
                    "at {n} {kind} the shader reads {} slots and the program declares {}",
                    highest + 1,
                    program.params.len()
                );
            }
        }
    }

    /// The same property on the other kind of run: a fan of blades writes one walk.
    ///
    /// **Its own test rather than a loop over both kinds**, because the two leave different things
    /// behind. A pipe's call disappears outright; a blade's does too, but its *profile binding*
    /// stays — one an operand, because a profile is a different polyline per blade and the loop
    /// reads `first + k` across them. So the source stops growing while the bindings go on
    /// growing, and both halves of that are the arrangement working.
    #[test]
    fn a_union_of_extrudes_writes_one_walk_however_many_blades_it_takes() {
        let mut sizes = Vec::new();
        for n in [RUN_FLOOR, 8, 16, 32] {
            let (doc, graph) = blades(n);
            let program = generate(&doc, &graph).expect("generates");
            let body = field_body(&program.source);
            assert_eq!(
                body.matches("extrude_run_distance(").count(),
                1,
                "one loop at {n} blades, not one call each"
            );
            assert_eq!(
                body.matches("extrude_distance(").count(),
                0,
                "no blade is written out beside the loop at {n}"
            );
            assert_eq!(
                program.curves.len(),
                n,
                "a profile binding a blade at {n}, which is what `first + k` walks"
            );
            sizes.push(program.field_function().lines().count());
        }
        // And below the floor it is written out as before. Whether four is the right floor for a
        // blade is `union_loop_cost`'s blades rows to answer; that it is *the* floor either kind
        // is checked here.
        for n in 2..RUN_FLOOR {
            let (doc, graph) = blades(n);
            let program = generate(&doc, &graph).expect("generates");
            let body = field_body(&program.source);
            assert_eq!(
                body.matches("extrude_run_distance(").count(),
                0,
                "{n} blades is below the floor and is written out"
            );
            assert_eq!(body.matches("extrude_distance(").count(), n, "one call each at {n}");
        }
        let (least, most) = (sizes[0], *sizes.last().expect("rows"));
        assert!(
            most <= least + 2,
            "the generated source grew from {least} to {most} lines over {RUN_FLOOR} to 32 \
             blades, so something is still being written out per operand"
        );
    }

    /// A run is one kind of block, and a second consumer takes a member out of one.
    ///
    /// **Measured at a floor of two, so what is being checked is the condition and not the
    /// floor.** That distinction is worth the `generate_with_run_floor`: the pipes' own refusal
    /// test reads below `RUN_FLOOR`, where a refused run and a short one look identical, so it
    /// cannot tell which of the two it caught.
    #[test]
    fn a_run_stops_at_a_change_of_block_and_at_a_second_consumer() {
        // Three blades then three pipes in one union. Each is a run of three at this floor, and
        // the loop has one body, so they must not join: the longest wins and the tie goes to the
        // earlier, which is the decision `looped_run` takes for both passes at once.
        let (mut doc, blade_graph) = blades(3);
        let mut graph = Graph::empty();
        let mut inputs = Vec::new();
        for (_, node) in blade_graph.nodes() {
            if let Node::Extrude { rims: _, outline, sweep, start, height, sides, draft, corners: _ } = node {
                inputs.push(graph.add(Node::Extrude { rims: Default::default(),
                    outline: *outline,
                    sweep: sweep.clone(),
                    height: *height,
                    start: *start,
                    sides: sides.clone(),
                    draft: *draft,
                    corners: Default::default(),
                }));
            }
        }
        for i in 0..3 {
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(
                    id,
                    "spine",
                    vec![
                        nalgebra::Point3::new(0.0, i as f64 * 40.0, 60.0),
                        nalgebra::Point3::new(60.0, i as f64 * 40.0, 60.0),
                    ],
                    false,
                )
                .expect("a line"),
            );
            inputs.push(graph.add(Node::Pipe { curve: id, radius: lit(5.0) }));
        }
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
        let mixed = generate_with_run_floor(&doc, &graph, 2).expect("generates");
        let body = field_body(&mixed.source);
        assert_eq!(
            body.matches("extrude_run_distance(").count(),
            1,
            "the blades are a run of their own"
        );
        assert_eq!(
            body.matches("pipe_run_distance(").count(),
            0,
            "the pipes are the later of two equal runs and lose the tie"
        );
        assert_eq!(
            body.matches("pipe_distance(").count(),
            3,
            "and so are written out, one call each"
        );

        // A blade the union reads and a blend in the same union reads too. Looping it would not
        // remove the copy the blend's operand emits, so it is refused -- and the run around it
        // breaks in two, which at this floor leaves the first half of two and not a run of four.
        //
        // The blend goes *into the same union* rather than standing as its own root, so this is
        // one body and the count below is one body's worth. As a second root it would be two
        // bodies, the shared blade's line written once in each, and the figure would be about how
        // many bodies the graph has rather than about the run.
        let (doc, four) = blades(4);
        let mut graph = Graph::empty();
        let mut ids = Vec::new();
        for (_, node) in four.nodes() {
            if let Node::Extrude { rims: _, outline, sweep, start, height, sides, draft, corners: _ } = node {
                ids.push(graph.add(Node::Extrude { rims: Default::default(),
                    outline: *outline,
                    sweep: sweep.clone(),
                    height: *height,
                    start: *start,
                    sides: sides.clone(),
                    draft: *draft,
                    corners: Default::default(),
                }));
            }
        }
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(20.0),
        });
        let rounded = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: ids[2].into(),
            b: ball.into(),
            fillet: fieldform_scene::Fillet { radius: lit(3.0), continuity: Continuity::G1, second: None },
        });
        let mut inputs = ids.clone();
        inputs.push(rounded);
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
        let shared = generate_with_run_floor(&doc, &graph, 2).expect("generates");
        let body = field_body(&shared.source);
        assert_eq!(
            body.matches("extrude_run_distance(").count(),
            1,
            "the two blades before the shared one are still a run"
        );
        assert_eq!(
            body.matches("extrude_distance(").count(),
            2,
            "the shared blade and the one after it are written out, one line each"
        );
    }

    /// A run is refused where it would be wrong, and the pipes are written out as before.
    ///
    /// Two of the three conditions, because they are the ones a real document meets: a pipe read
    /// by something else as well as the union, and two pipes on one spine. Both would break the
    /// consecutive interning `pipe_run_distance` reads.
    #[test]
    fn a_shared_pipe_or_a_shared_spine_is_not_walked_as_a_run() {
        // Two pipes on one spine: the run's spines would not be distinct bindings at all.
        let mut doc = fieldform_scene::Document::new();
        let curve = fieldform_scene::CurveId::new();
        doc.curves.insert(
            curve,
            fieldform_scene::Curve::polyline(
                curve,
                "spine",
                vec![nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(60.0, 0.0, 0.0)],
                false,
            )
            .expect("a line"),
        );
        let mut graph = Graph::empty();
        let a = graph.add(Node::Pipe { curve, radius: lit(5.0) });
        let b = graph.add(Node::Pipe { curve, radius: lit(9.0) });
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        let shared_spine = generate(&doc, &graph).expect("generates");
        assert_eq!(
            field_body(&shared_spine.source).matches("pipe_run_distance(").count(),
            0,
            "two pipes on one spine are not a run"
        );

        // A pipe the union reads and a blend reads too: looping it here would not remove the copy
        // the blend emits, and its spine is interned for that call as well.
        let (mut doc, _) = spokes(0);
        let mut graph = Graph::empty();
        let mut ids = Vec::new();
        for i in 0..3 {
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(
                    id,
                    "spine",
                    vec![
                        nalgebra::Point3::new(0.0, i as f64 * 40.0, 0.0),
                        nalgebra::Point3::new(60.0, i as f64 * 40.0, 0.0),
                    ],
                    false,
                )
                .expect("a line"),
            );
            ids.push(graph.add(Node::Pipe { curve: id, radius: lit(5.0) }));
        }
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(20.0),
        });
        // The middle pipe is read twice, so the run around it is broken in two and neither half
        // reaches the floor.
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: ids[1].into(),
            b: ball.into(),
            fillet: fieldform_scene::Fillet { radius: lit(3.0), continuity: Continuity::G1, second: None },
        });
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.clone().into_iter().map(Into::into).collect() });
        let shared_pipe = generate(&doc, &graph).expect("generates");
        assert_eq!(
            field_body(&shared_pipe.source).matches("pipe_run_distance(").count(),
            0,
            "a run whose members are read elsewhere is not walked"
        );
    }

    /// A union's spines are bound in the order the union lists its operands.
    ///
    /// **Not true before the loop existed, and nothing said so.** The bindings were whatever the
    /// emission order interned, and `graph.evaluation_order()` reaches a union's operands back to
    /// front — two pipes came out as bindings 1 and 0. Anything that pairs a polyline with a
    /// binding by position was therefore pairing them wrongly, which is what `gpu_matches_cpu`
    /// was doing: its four-spoke scene read 45 mm off until the run's pre-pass fixed the order.
    ///
    /// A caller should still bind by id rather than by position. This is here because the run's
    /// `first + k` depends on the order outright, so it is a property now and not an accident.
    #[test]
    fn a_union_binds_its_spines_in_the_order_it_lists_them() {
        for n in [RUN_FLOOR, 8, 16] {
            let (doc, graph) = spokes(n);
            let listed: Vec<fieldform_scene::CurveId> = graph
                .nodes()
                .find_map(|(_, node)| match node {
                    Node::Boolean { inputs, .. } => Some(
                        inputs
                            .iter()
                            .map(|slot| match slot.source().and_then(|id| graph.node(id)) {
                                Some(Node::Pipe { curve, .. }) => *curve,
                                _ => panic!("every operand is a pipe"),
                            })
                            .collect(),
                    ),
                    _ => None,
                })
                .expect("a union");
            let program = generate(&doc, &graph).expect("generates");
            let bound: Vec<fieldform_scene::CurveId> = program
                .curves
                .iter()
                .map(|use_of| match use_of {
                    CurveUse::Spine(id) => *id,
                    other => panic!("only spines here, not {other:?}"),
                })
                .collect();
            assert_eq!(bound, listed, "at {n} pipes");
        }
    }

    /// The size report says what it claims, and its point chain is not a constant.
    ///
    /// **The last part is why this test exists.** `bytes`, `bodies` and `nodes` would be obviously
    /// wrong if they were wrong. A depth that is always 1 would look exactly like a graph that
    /// changes no point, and `shader_size_cost`'s two arms both read 1 legitimately — so nothing
    /// there would catch the column being dead. A mirror over a twist is the case that must read
    /// deeper, and it is also the case that costs: each of them makes the subtree below it be
    /// written again at a new point.
    #[test]
    fn a_programs_size_report_counts_bodies_nodes_and_the_point_chain() {
        // Three bodies, nothing that moves a point.
        let mut flat = Graph::empty();
        for i in 0..3 {
            flat.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(i as f64 * 40.0), lit(0.0), lit(0.0)]),
                radius: lit(10.0),
            });
        }
        let doc = fieldform_scene::Document::new();
        let program = generate(&doc, &flat).expect("generates");
        assert_eq!(program.size.bodies, 3);
        assert_eq!(program.size.nodes, 3);
        assert_eq!(program.size.bytes, program.source.len());
        assert_eq!(program.size.deepest_point_chain, 1, "nothing here moves a point");

        // A mirror over a twist over a cuboid: one body, and two nodes that each declare a point.
        let mut deep = Graph::empty();
        let block = deep.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(40.0), lit(0.0), lit(0.0)]),
            size: [lit(30.0), lit(30.0), lit(60.0)],
            turn: fieldform_scene::Turn::default(),
        });
        let twisted = deep.add(Node::Twist {
            input: block.into(),
            degrees_per_mm: lit(0.3),
            frame: fieldform_scene::Frame::default(),
        });
        deep.add(Node::Mirror {
            input: twisted.into(),
            frame: fieldform_scene::Frame::At { origin: [lit(0.0); 3], axis: [lit(1.0), lit(0.0), lit(0.0)] },
            join: true,
        });
        let deeper = generate(&doc, &deep).expect("generates");
        assert_eq!(deeper.size.bodies, 1, "a joined mirror swallows what it reflected");
        assert_eq!(deeper.size.nodes, 3);
        assert!(
            deeper.size.deepest_point_chain > program.size.deepest_point_chain,
            "a mirror over a twist declares points the flat graph does not: {} against {}",
            deeper.size.deepest_point_chain,
            program.size.deepest_point_chain,
        );
    }

    /// Shader caching and reproducible builds (D7) both need the same graph to compile to the
    /// same bytes, so ordering must not depend on hash iteration order.
    #[test]
    fn the_same_graph_always_generates_the_same_source() {
        let graph = cage_graph();
        let first = generate(&fieldform_scene::Document::new(), &graph).unwrap();
        for _ in 0..8 {
            assert_eq!(generate(&fieldform_scene::Document::new(), &graph).unwrap(), first);
        }
    }

    /// A difference negates its second operand. Getting this backwards produces a shape that
    /// renders confidently and is inside out, so it is worth pinning directly.
    #[test]
    fn a_difference_negates_the_operand_it_subtracts() {
        let mut graph = Graph::from_cage(CageId::new());
        let a = graph.roots()[0];
        let b = graph.add(Node::Pipe {
            curve: CurveId::new(),
            radius: ParamRef::Literal(1.0),
        });
        graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![a.into(), b.into()],
        });

        // Asserted on the negation rather than on slot numbers, which follow evaluation
        // order and are not the thing under test.
        let f = generate(&fieldform_scene::Document::new(), &graph).unwrap().field_function().to_string();
        assert!(
            // By its own name, or by the name its mask gives it: a dear operand is read through
            // `mo..` so that a cell which rules it out need not evaluate it.
            f.contains("max(") && (f.contains(", -(n") || f.contains(", -(mo")),
            "the subtracted operand should be negated, got:
{f}"
        );
    }
    /// The shader numbers the bodies the way `Document::folded_bodies` does, and says which one it hit.
    ///
    /// The highlight rests on this and nothing enforces it across the two crates: the app
    /// counts a selected body's position among the bodies it is drawing, and the shader reports
    /// an index from its own fold. If the two lists ever disagree in order or in membership,
    /// selecting one body tints another — which looks like a picking bug and is not.
    ///
    /// **Both sides ask the document, not the graph.** This used to assert that the order was
    /// `Graph::roots`' raw-id sort, which is stable between runs and is decided by nothing: a new
    /// body took a rank drawn uniformly among the existing ones and renamed the generated function
    /// of every body it landed in front of. `body-order-decision.md` carries the ruling, and the
    /// test below it measures what it recovers.
    #[test]
    fn the_shader_numbers_its_bodies_the_way_the_document_does() {
        let mut graph = Graph::empty();
        graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(5.0) });
        graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(20.0); 3]), size: [lit(8.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(40.0); 3]), radius: lit(5.0) });

        // The property both sides rely on: each asks `Document::folded_bodies` and neither invents
        // an order of its own. With nothing stamped, every body ties at zero and the id breaks
        // it — the one case where this is still the old answer, and it must still be *an*
        // answer rather than hash order.
        let document = fieldform_scene::Document::new();
        let bodies = document.folded_bodies(&graph);
        assert_eq!(bodies.len(), 3);
        let mut sorted = bodies.clone();
        sorted.sort_by_key(|id| id.raw());
        assert_eq!(bodies, sorted, "unstamped bodies tie, and the id is the tiebreak");

        let program = generate(&document, &graph).unwrap();
        validate(&program.source).unwrap();

        // One assignment per body, in order, and the first is the base case rather than a
        // comparison — so a three-body part numbers them 0, 1, 2.
        for i in 1..3 {
            assert!(
                program.source.contains(&format!("owner = {i}u;")),
                "body {i} should set its own index: {}",
                program.source,
            );
        }
        assert!(program.source.contains("owner = 0u;"));
        assert!(!program.source.contains("owner = 3u;"), "there is no fourth body");
    }

    /// **A body already in the part keeps its generated function when another body is added.**
    ///
    /// This is the whole of what body order is for. `split_into_bodies` names each body function
    /// by its position in the fold, so a body that changes rank is a body the driver compiles
    /// again — 1.1 s to 3.8 s cold on the scenes `examples/wing_compile_cost.rs` times. Ordered by
    /// `NodeId`, a new body went last only 1/(n+1) of the time and renamed everything it landed in
    /// front of; `examples/body_order_probe.rs` measures that at 67% of creations on a two-body
    /// part and 83% on a five-body one.
    ///
    /// Asserted on the generated text rather than on the order, because the text is what the
    /// driver is handed. Twenty rounds, because the failure is a random id landing in front of a
    /// body rather than behind it — one round would pass by luck a quarter of the time.
    ///
    /// **With a control that must show no difference**: regenerating the same document twice. If
    /// that ever differs, this test is measuring generation instability and not body order, and
    /// the first assertion means nothing.
    #[test]
    fn a_body_already_in_the_part_keeps_its_function_when_another_is_added() {
        use fieldform_scene::Place;
        // A stamp per body, as `Document::created` holds one: distinct and increasing.
        let stamp = |document: &mut fieldform_scene::Document, id: NodeId, at: u64| {
            document.created.insert(id, 1_700_000_000_000_000 + at);
        };
        let ball = |graph: &mut Graph, x: f64| {
            graph.add(Node::Sphere { centre: Place::At([lit(x), lit(0.0), lit(0.0)]), radius: lit(9.0) })
        };

        for round in 0..20 {
            let mut document = fieldform_scene::Document::new();
            let mut graph = Graph::empty();
            let existing: Vec<NodeId> = (0..3).map(|i| ball(&mut graph, i as f64 * 30.0)).collect();
            for (i, id) in existing.iter().enumerate() {
                stamp(&mut document, *id, i as u64);
            }

            let before = generate(&document, &graph).unwrap();
            // The control: nothing has changed, so nothing may differ.
            assert_eq!(
                before.source,
                generate(&document, &graph).unwrap().source,
                "round {round}: the same document generated twice must be byte-identical",
            );

            // A fourth body, made after the three. Its id is fresh and therefore random, which
            // is exactly what used to decide its rank.
            let fresh = ball(&mut graph, 999.0);
            stamp(&mut document, fresh, 3);
            let after = generate(&document, &graph).unwrap();

            // Each of the three keeps the function it had. `body_3` is the new one, so the whole
            // of the old program's body functions must appear unchanged in the new one.
            for i in 0..3 {
                let was = body_function(before.field_function(), i);
                let now = body_function(after.field_function(), i);
                assert_eq!(
                    was, now,
                    "round {round}: body_{i} was rewritten by a creation that did not touch it",
                );
            }
            assert!(
                after.field_function().contains("fn body_3("),
                "round {round}: the new body should be the one added function",
            );
        }
    }

    /// One generated body function, by index, for comparing two programs body by body.
    #[cfg(test)]
    fn body_function(source: &str, index: usize) -> String {
        let opening = format!("fn body_{index}(");
        let from = source.find(&opening).unwrap_or_else(|| {
            panic!("no body_{index} in the generated source:\n{source}");
        });
        let rest = &source[from..];
        // Each generated function is followed by a blank line, so the next one is where this
        // one ends -- read out of the text rather than re-derived, the same choice `closure_of`
        // makes and for the same reason. Trailing blank lines trimmed, because the *last*
        // function in a program has a different number of them from a middle one and that is
        // formatting rather than content.
        let to = rest.find("\n\nfn ").unwrap_or(rest.len());
        rest[..to].trim_end().to_string()
    }

    /// **Arranging the block list does not change one byte of the generated source.**
    ///
    /// The reason the list's order and the compilers' order are two different things. A drag is
    /// presentation — it changes nothing about the model — so if it reached here it would rename
    /// every body function it moved past and rebuild the whole part, 1.1 s to 3.8 s cold. That is
    /// the cost `body-order-decision.md` removed, and a row moved two places is a poor thing to
    /// spend it on. `block-order-spec.md` §4 carries the argument.
    ///
    /// Every permutation of four bodies, because the failure would be one arrangement leaking
    /// through rather than all of them.
    #[test]
    fn arranging_the_list_does_not_change_the_generated_source() {
        use fieldform_scene::Place;
        let mut document = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let bodies: Vec<NodeId> = (0..4)
            .map(|i| {
                let id = graph.add(Node::Sphere {
                    centre: Place::At([lit(i as f64 * 30.0), lit(0.0), lit(0.0)]),
                    radius: lit(9.0),
                });
                document.created.insert(id, 1_700_000_000_000_000 + i as u64 + 1);
                id
            })
            .collect();
        let untouched = generate(&document, &graph).unwrap().source;

        // All 24 orders, written as ranks the way `App::arrange_block` writes them.
        let mut order = [0usize, 1, 2, 3];
        let mut seen = 0;
        permutations(&mut order, 4, &mut |rows: &[usize]| {
            seen += 1;
            for (rank, which) in rows.iter().enumerate() {
                document.arrangement.insert(bodies[*which], rank as u32);
            }
            assert_eq!(
                generate(&document, &graph).unwrap().source,
                untouched,
                "arrangement {rows:?} reached the shader",
            );
            // And the list really is in that order, so the test is not passing because the
            // arrangement was ignored everywhere.
            let listed: Vec<NodeId> = rows.iter().map(|i| bodies[*i]).collect();
            assert_eq!(document.listed_bodies(&graph), listed, "the list follows the ranks");
        });
        assert_eq!(seen, 24, "every order of four");
    }

    /// **Where a block sits in the list does not change one byte of the source.**
    ///
    /// Spec test 18 (`block-ownership-spec.md` §2.2, §12). Making a block a variable lifts its row
    /// out of the block that reads it and putting it back puts the row back, and both write the
    /// block's stored ownership and nothing else. So neither may reach here: the ownership is a
    /// document annotation, the graph is exactly what it was, and a row that changed depth is a
    /// row redrawn rather than a part recompiled. This asserted the same of a name while a name was
    /// what lifted a row out; `nesting_reaches_neither_the_fold_nor_the_list_order` is the same
    /// property over the two orders.
    #[test]
    fn an_ownership_change_does_not_change_the_generated_source() {
        use fieldform_scene::{BooleanOp, Ownership, Part, PartId, Place};
        let mut document = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            size: [lit(20.0), lit(20.0), lit(20.0)],
            turn: fieldform_scene::no_turn(),
        });
        let union =
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        for (i, id) in [sphere, cube, union].iter().enumerate() {
            document.created.insert(*id, 1_700_000_000_000_000 + i as u64 + 1);
        }
        document.add_part(Part {
            id: PartId::new(),
            name: "Part".into(),
            material: None,
            graph: graph.clone(),
            inputs: Vec::new(),
        });
        // Settled, as every edit leaves a document: the union adopts both.
        document.settle_ownership();
        let nested = generate(&document, &graph).unwrap().source;
        let nesting = document.nesting_in(&graph);
        assert_eq!(nesting.owners_of(sphere), [union], "the sphere starts inside the union");
        assert_eq!(nesting.owners_of(cube), [union]);

        // Made variables the way `SetOwnership` makes one: the ownership and a name, then a settle.
        for (id, name) in [(sphere, "hub"), (cube, "block")] {
            document.ownership_nodes.insert(id, Ownership::Variable);
            document.node_meta.entry(id).or_default().name = Some(name.to_string());
            document.settle_ownership();
            assert_eq!(
                generate(&document, &graph).unwrap().source,
                nested,
                "making {name} a variable reached the shader",
            );
        }
        // And they really did leave, so this is not passing because nothing happened.
        let nesting = document.nesting_in(&graph);
        assert!(nesting.owners_of(sphere).is_empty(), "a variable has a row of its own");
        assert!(nesting.owners_of(cube).is_empty());

        // Put back inside, keeping the names, which is what tells the stored ownership from a rule
        // read off names: by names they would stay out.
        for id in [sphere, cube] {
            document.ownership_nodes.insert(id, Ownership::Nested);
            document.settle_ownership();
            assert_eq!(
                generate(&document, &graph).unwrap().source,
                nested,
                "putting {id:?} back inside reached the shader",
            );
        }
        let nesting = document.nesting_in(&graph);
        assert_eq!(nesting.owners_of(sphere), [union], "back inside, name and all");
        assert_eq!(nesting.owners_of(cube), [union]);
    }

    /// Heap's algorithm, so the test above can say "every order" and mean it.
    #[cfg(test)]
    fn permutations(order: &mut [usize], k: usize, each: &mut impl FnMut(&[usize])) {
        if k == 1 {
            each(order);
            return;
        }
        for i in 0..k {
            permutations(order, k - 1, each);
            if k % 2 == 0 {
                order.swap(i, k - 1);
            } else {
                order.swap(0, k - 1);
            }
        }
    }

    /// A dear body with a finite box and a true-distance value is asked its box before itself,
    /// in both passes; a body whose value is a lower bound is not, because the guard would
    /// change the number it folds to; a body cheaper than the test is not, because the test
    /// would cost more than it saves; and the box takes six slots after every node's own.
    #[test]
    fn a_true_distance_body_is_guarded_by_its_box_and_a_lower_bound_body_is_not() {
        use fieldform_scene::{BooleanOp, CellMap, Frame, LatticeCell, LatticePhase, Place};
        let doc = fieldform_scene::Document::new();
        // **Guards only, not every box test.** A ruled-out body now contributes its *cull
        // extent* through the same `box_guard`, so counting the function's name counts both and
        // this test would read a body as guarded because the tile skipped it. A guard is the one
        // bound to a name up front; the extent's is inline in the fold.
        let guard = |source: &str| source.matches(" = box_guard(p, ").count();
        // A body of four overlapping spheres: dear enough to guard, and inside the one ball.
        let dear = |graph: &mut Graph, x: f64, r: f64| {
            let parts: Vec<NodeId> = (0..4)
                .map(|k| {
                    graph.add(Node::Sphere {
                        centre: Place::At([lit(x + k as f64 * 0.001), lit(0.0), lit(0.0)]),
                        radius: lit(r),
                    })
                })
                .collect();
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: parts.into_iter().map(Into::into).collect() })
        };

        // **The cull is asked for by name, here and in every test below that mentions it**, since
        // it is off unless the environment says otherwise -- see `cull_wanted` for why. A test
        // that called `generate` would be asserting about whichever way the machine running it
        // happened to be set.
        let culled = |graph: &Graph| generate_for_measurement(&doc, graph, true, true).unwrap();

        // Two dear bodies apart: both guarded, in both the value fold and the normal fold.
        let mut graph = Graph::empty();
        dear(&mut graph, 0.0, 5.0);
        dear(&mut graph, 40.0, 5.0);
        let program = culled(&graph);
        validate(&program.source).unwrap();
        assert_eq!(guard(&program.source), 4, "two bodies, two passes:\n{}", program.source);
        // **The guard's own arm still culls**, so the twelve slots between them are the guard's
        // and nothing else -- which is what makes this a control for the guard rather than for
        // the pair. The cull has its own switch for the same reason.
        let unguarded = generate_for_measurement(&doc, &graph, false, true).unwrap();
        assert_eq!(guard(&unguarded.source), 0);
        assert_eq!(program.params.len(), unguarded.params.len() + 12);
        assert!(unguarded.cull_base.is_some(), "the guard off does not turn the cull off");
        // **Read through `body_boxes` rather than by position**: the cull's extents are pushed
        // after the guards, so a slice taken from the end of the unguarded program's params lands
        // on those instead -- which is how this test would have passed while measuring the wrong
        // six numbers.
        let corners: Vec<f64> = program
            .body_boxes
            .iter()
            .flatten()
            .flat_map(|first| &program.params[*first..*first + 6])
            .map(|p| match p {
                ParamRef::Literal(v) => *v,
                _ => unreachable!("a box corner is a number"),
            })
            .collect();
        assert_eq!(corners.len(), 12, "two bodies, six slots each");
        let near = |wanted: f64, below: bool| {
            corners.iter().any(|&v| if below { v < wanted && v > wanted - 0.1 } else { v > wanted && v < wanted + 0.1 })
        };
        assert!(near(-5.0, true), "the first body's low corner, padded: {corners:?}");
        assert!(near(5.0, false), "the first body's high corner, padded: {corners:?}");
        assert!(near(35.0, true), "the second body's low x corner, padded: {corners:?}");
        assert!(near(45.0, false), "the second body's high x corner, padded: {corners:?}");

        // A lattice-filled dear body reads a lower bound, so it is evaluated at every reading.
        let mut graph = Graph::empty();
        dear(&mut graph, 0.0, 5.0);
        let solid = dear(&mut graph, 40.0, 5.0);
        let lattice = graph.add(Node::Lattice {
            frame: Frame::default(),
            cell: LatticeCell::Gyroid,
            map: CellMap::Rectangular { cell: lit(2.0) },
            phase: LatticePhase::sheet(lit(0.3)),
        });
        graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![solid.into(), lattice.into()] });
        let program = culled(&graph);
        validate(&program.source).unwrap();
        assert_eq!(guard(&program.source), 2, "the solid body alone, in two passes");

        // Two single primitives: each costs about what the test does, so neither is guarded.
        let mut graph = Graph::empty();
        graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(5.0) });
        graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(40.0), lit(0.0), lit(0.0)]),
            size: [lit(10.0), lit(4.0), lit(6.0)],
            turn: fieldform_scene::no_turn(),
        });
        let program = culled(&graph);
        assert_eq!(guard(&program.source), 0, "cheap bodies are not guarded");
        assert_eq!(body_weight(&graph, graph.roots()[0]), 1);

        // A lone body is the fold's whole answer, so there is nothing to guard it against.
        let mut graph = Graph::empty();
        dear(&mut graph, 0.0, 5.0);
        let program = culled(&graph);
        assert_eq!(guard(&program.source), 0);
    }

    /// Every body gets a cull extent, where only some get a guard box -- and the cull's extent
    /// strictly contains the guard's, because it has to bound the surface that is *drawn*.
    ///
    /// **The two sets are not the same set, and that is the point of having both.** A guard box
    /// stands in for a body during the march, so it is only given to a body whose value is a true
    /// distance and which is dear enough to be worth a test (`body_box_slots`). A cull extent
    /// answers a different question -- can this body be anywhere down this ray at all -- which is
    /// asked once per tile, not per sample, and is sound for any body whose surface is inside a
    /// box. So the lattice-filled body here has no guard and does have a cull extent.
    #[test]
    fn every_body_gets_a_cull_extent_where_only_some_get_a_guard_box() {
        use fieldform_scene::{BooleanOp, CellMap, Frame, LatticeCell, LatticePhase, Place};
        let doc = fieldform_scene::Document::new();
        let dear = |graph: &mut Graph, x: f64| {
            let parts: Vec<NodeId> = (0..4)
                .map(|k| {
                    graph.add(Node::Sphere {
                        centre: Place::At([lit(x + k as f64 * 0.001), lit(0.0), lit(0.0)]),
                        radius: lit(5.0),
                    })
                })
                .collect();
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: parts.into_iter().map(Into::into).collect() })
        };

        let mut graph = Graph::empty();
        dear(&mut graph, 0.0);
        let solid = dear(&mut graph, 40.0);
        let lattice = graph.add(Node::Lattice {
            frame: Frame::default(),
            cell: LatticeCell::Gyroid,
            map: CellMap::Rectangular { cell: lit(2.0) },
            phase: LatticePhase::sheet(lit(0.3)),
        });
        graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![solid.into(), lattice.into()] });
        let program = generate_for_measurement(&doc, &graph, true, true).unwrap();
        validate(&program.source).unwrap();

        // One guard box between them, and a cull extent for each.
        assert_eq!(program.body_boxes.iter().filter(|b| b.is_some()).count(), 1);
        let base = program.cull_base.expect("two bodies cull");
        assert_eq!(program.params.len(), base + 12, "six slots for each of the two bodies");

        // Both bodies' extents are there and each is wider than the body it bounds. The plain
        // body spans -5 to 5; the latticed one 35 to 45, cut by a gyroid that reaches no further.
        // Which comes first is `folded_bodies`' order and is not this test's business, so each is
        // looked for rather than indexed -- the same reason the guard test says either may lead.
        let at = |i: usize| match &program.params[base + i] {
            ParamRef::Literal(v) => *v,
            other => unreachable!("a cull corner is a number, got {other:?}"),
        };
        for (lo_x, hi_x) in [(-5.0_f64, 5.0_f64), (35.0, 45.0)] {
            let found = (0..2).any(|body| {
                let o = body * 6;
                at(o) < lo_x && at(o) > lo_x - 1.0 && at(o + 3) > hi_x && at(o + 3) < hi_x + 1.0
            });
            let spans: Vec<f64> = (0..12).map(at).collect();
            assert!(found, "no body spans {lo_x} to {hi_x} with a pad: {spans:?}");
        }

        // **The cull's box contains the guard's**, which is the ordering the whole arrangement
        // needs: the guard may be tight because it is compared against a distance, and the cull
        // must be slack because it is compared against the surface a march will actually find.
        let guarded = program.body_boxes.iter().position(|b| b.is_some()).unwrap();
        let guard_at = program.body_boxes[guarded].unwrap();
        for k in 0..3 {
            let (g_lo, g_hi) = (&program.params[guard_at + k], &program.params[guard_at + 3 + k]);
            let (ParamRef::Literal(g_lo), ParamRef::Literal(g_hi)) = (g_lo, g_hi) else {
                unreachable!("a guard corner is a number")
            };
            assert!(at(guarded * 6 + k) < *g_lo, "cull low {k} is below the guard's");
            assert!(at(guarded * 6 + 3 + k) > *g_hi, "cull high {k} is above the guard's");
        }

        // Both passes ask the tile, once per body each.
        for body in 0..2u32 {
            assert_eq!(
                program.source.matches(&format!("!tile_ruled_out({body}u)")).count(),
                2,
                "both passes ask, once each"
            );
        }
        // **And a long step is vouched before the grid may hop from its landing**, which is what
        // makes dropping a body safe at all: the kept fold is an exact distance to everything the
        // ray can hit, so Keinert's vouch holds -- provided it is actually performed, and for a
        // day a clear landing cell let the hop skip it. Asserted here, next to the drop it
        // licenses, rather than left to `march.wgsl` -- the two are one decision.
        assert!(
            MARCH.contains("may_hop = (known + before) * STEP_SAFETY >= reach;"),
            "the landing cell's clearance vouches the step, or the reading does"
        );
        assert!(
            !MARCH.contains("select(skip_f(7u), 1.0, tile_live)"),
            "and a masked ray is not made to march plainly"
        );
        let _ = base;
    }

    /// A body no box can hold is never ruled out, and one with no surface at all always is.
    ///
    /// **Both are written as boxes rather than as a special case**, because the cull reads six
    /// numbers and has no third thing to read. An unbounded body gets ±1e30, which every tile's
    /// frustum meets; an empty one gets that box *inverted*, `lo` above `hi`, which the cull
    /// rejects before it tests a plane. The inversion is the load-bearing half: a frustum has four
    /// side planes and no far plane, so an empty body written as a far-away box would be accepted
    /// by any tile looking down the axis it sat on.
    #[test]
    fn an_unbounded_body_is_never_ruled_out_and_an_empty_one_always_is() {
        use fieldform_scene::{BooleanOp, Place};
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        // A half-space: infinite, so no box describes it.
        graph.add(Node::Plane {
            origin: Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        // Two spheres that miss each other, intersected: no surface anywhere.
        let a = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(5.0) });
        let b =
            graph.add(Node::Sphere { centre: Place::At([lit(100.0), lit(0.0), lit(0.0)]), radius: lit(5.0) });
        graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![a.into(), b.into()] });

        let program = generate_for_measurement(&doc, &graph, true, true).unwrap();
        validate(&program.source).unwrap();
        let base = program.cull_base.expect("two bodies cull");
        let at = |i: usize| match &program.params[base + i] {
            ParamRef::Literal(v) => *v,
            other => unreachable!("a cull corner is a number, got {other:?}"),
        };
        // One of each, looked for rather than indexed: which body `folded_bodies` leads with is
        // not what this test is about.
        let everywhere = |o: usize| (0..3).all(|k| at(o + k) == -1e30 && at(o + 3 + k) == 1e30);
        let nowhere = |o: usize| (0..3).all(|k| at(o + k) == 1e30 && at(o + 3 + k) == -1e30);
        let spans: Vec<f64> = (0..12).map(at).collect();
        assert!(
            (everywhere(0) && nowhere(6)) || (everywhere(6) && nowhere(0)),
            "one body everywhere, one nowhere: {spans:?}"
        );
    }

    /// Culling off leaves the fold the text it was before any of this existed.
    ///
    /// **Three ways it is off, and the fold has to be identical under all of them**: a lone body,
    /// which has nothing to be culled against; the measuring switch, which exists to compare
    /// against today; and -- untested here because it is 257 spheres -- more bodies than
    /// `CULL_CAP`. The assertion is against the *unculled text*, not against a flag, because the
    /// claim worth holding is that nothing in the march changed, and a flag read back is only a
    /// claim that we remembered to set it.
    #[test]
    fn culling_off_generates_what_it_always_did() {
        use fieldform_scene::{BooleanOp, Place};
        let doc = fieldform_scene::Document::new();
        let dear = |graph: &mut Graph, x: f64| {
            let parts: Vec<NodeId> = (0..4)
                .map(|k| {
                    graph.add(Node::Sphere {
                        centre: Place::At([lit(x + k as f64 * 0.001), lit(0.0), lit(0.0)]),
                        radius: lit(5.0),
                    })
                })
                .collect();
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: parts.into_iter().map(Into::into).collect() })
        };

        // A lone body.
        let mut graph = Graph::empty();
        dear(&mut graph, 0.0);
        let lone = generate_for_measurement(&doc, &graph, true, true).unwrap();
        assert_eq!(lone.cull_base, None);
        // The *call* is what a frame pays for, and it is absent; the declaration is there for
        // `march.wgsl`'s one fixed prologue to compile against, saying nought bodies.
        assert!(!lone.source.contains("!tile_ruled_out("), "the fold does not ask");
        assert!(lone.source.contains("const CULL_BODIES: u32 = 0u;"), "and says why not");

        // And two bodies with the switch off.
        let mut graph = Graph::empty();
        dear(&mut graph, 0.0);
        dear(&mut graph, 40.0);
        // And two bodies with the cull's own switch off, the guard left on -- which is the arm
        // that isolates what the cull changed.
        let off = generate_for_measurement(&doc, &graph, true, false).unwrap();
        assert_eq!(off.cull_base, None);
        assert!(!off.source.contains("!tile_ruled_out("));
        assert!(off.source.contains("const CULL_BODIES: u32 = 0u;"));

        // **And the fold itself is the text it was before step 11 existed**, which is the claim
        // worth holding: with the cull off, every line the march runs per sample is the line it
        // always ran. Asserted as the shape rather than by stripping the on-arm's text, because
        // the on-arm is now a branch around the body rather than a conjunct inside its condition
        // -- there is nothing to strip, and a test that tried would be rewriting one fold into the
        // other and then checking its own arithmetic.
        let fold = |source: &str| {
            let from = source.find("fn field(p: vec3<f32>) -> f32 {").expect("the value fold");
            let to = source[from..].find("\n}\n").expect("its end") + from;
            source[from..to].to_string()
        };
        let plain = fold(&off.source);
        assert!(
            plain.contains("    if (g0 < best) { let b0 = body_0(p); if (b0 < best) { best = b0; owner = 0u; } }"),
            "the guard alone, as it always was:\n{plain}"
        );
        assert!(!plain.contains("tile_ruled_out"), "and nothing of the tile in it:\n{plain}");
        assert!(!plain.contains("min(best, box_guard"), "nor any extent as a bound:\n{plain}");

        let on = generate_for_measurement(&doc, &graph, true, true).unwrap();
        assert_ne!(fold(&on.source), plain, "the on arm is a different fold");
    }

    /// More bodies than the mask can carry switches the cull off rather than covering some.
    ///
    /// **A partial mask would break the one rule everything here rests on** -- a set bit rules a
    /// body out -- because a body past the end has no bit, and a bit that is absent reads as
    /// "keep", which is sound, while a *read* past the end of the array does not read as zero in
    /// WGSL. So the cap is a switch, not a truncation, and this is what says so.
    #[test]
    fn more_bodies_than_the_cap_switches_the_cull_off() {
        use fieldform_scene::Place;
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        for k in 0..=CULL_CAP {
            graph.add(Node::Sphere {
                centre: Place::At([lit(k as f64 * 20.0), lit(0.0), lit(0.0)]),
                radius: lit(5.0),
            });
        }
        let program = generate_for_measurement(&doc, &graph, true, true).unwrap();
        assert_eq!(graph.roots().len(), CULL_CAP + 1);
        assert_eq!(program.cull_base, None, "one past the cap is off, not truncated");
        assert!(!program.source.contains("!tile_ruled_out("));

        // And exactly at the cap it is on, so the boundary is the cap and not one below it.
        let mut graph = Graph::empty();
        for k in 0..CULL_CAP {
            graph.add(Node::Sphere {
                centre: Place::At([lit(k as f64 * 20.0), lit(0.0), lit(0.0)]),
                radius: lit(5.0),
            });
        }
        let program = generate_for_measurement(&doc, &graph, true, true).unwrap();
        assert!(program.cull_base.is_some(), "the cap itself is carried");
        assert!(program.source.contains("const MASK_WORDS: u32 = 8u;"), "256 bodies, eight words");
    }

    /// The tile mask is dead before any ray that is not the primary one.
    ///
    /// **A secondary ray leaves the tile, so the tile's bits do not describe it.** A shadow ray
    /// goes toward the light, which can be across the view: a body outside this tile's frustum
    /// can perfectly well stand between this surface and the sun, and a mask left live would
    /// delete the shadow it casts. Occlusion is the same at a shorter reach, and the tapering
    /// blend's stencil reads four points beside the ray rather than on it.
    ///
    /// Held on the *source*, because nothing else can hold it: naga will not say that a variable
    /// was still set at a return, and a GPU test would have to compare two lit frames to notice
    /// a missing shadow. One rule covers all three -- the mask dies with the primary ray -- so
    /// what this checks is every way out of `trace_primary`, which is where it is loaded.
    #[test]
    fn the_tile_mask_is_dead_before_any_secondary_ray() {
        let body = |source: &str, name: &str| {
            let at = source.find(&format!("fn {name}(")).expect("the function is there");
            let open = source[at..].find('{').expect("a body") + at;
            let mut depth = 0usize;
            for (i, c) in source[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return source[open..open + i].to_string();
                        }
                    }
                    _ => {}
                }
            }
            unreachable!("unbalanced braces in {name}");
        };

        // **Comments are stripped first**, and the first draft of this did not: the prose
        // explaining the arrangement contains the word "return", and the test duly reported the
        // comment as an unguarded one. A textual check has to read code or it is reading itself.
        let code = |source: &str| {
            source
                .lines()
                .map(|l| match l.find("//") {
                    Some(at) => &l[..at],
                    None => l,
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let primary = code(&body(MARCH, "trace_primary"));
        let loaded = primary.find("load_tile_mask(").expect("the mask is loaded here");
        let returns = primary.matches("return ").count();
        assert!(returns >= 4, "expected several ways out, found {returns}:\n{primary}");
        let mut unguarded = Vec::new();
        for (i, _) in primary.match_indices("return ") {
            // A return above the load is safe by `tile_live`'s own initialiser -- the mask is not
            // live yet, so there is nothing to clear.
            if i < loaded {
                continue;
            }
            match primary[..i].rfind("tile_live = false;") {
                Some(at) if at > loaded => {}
                _ => unguarded.push(primary[i..].lines().next().unwrap_or("").trim().to_string()),
            }
        }
        assert!(unguarded.is_empty(), "a return with the mask still live: {unguarded:?}");

        // And the stencil puts it back, rather than leaving the rest of the normal unculled.
        use fieldform_scene::{BooleanOp, Continuity, Fillet, Place};
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let a = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let b =
            graph.add(Node::Sphere { centre: Place::At([lit(14.0), lit(0.0), lit(0.0)]), radius: lit(10.0) });
        let join = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: a.into(),
            b: b.into(),
            fillet: Fillet { radius: lit(3.0), continuity: Continuity::G1, second: None },
        });
        // A second body, so the part culls at all -- one body is below the gate.
        graph.add(Node::Sphere { centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]), radius: lit(10.0) });
        graph.mark_critical(fieldform_scene::Critical {
            clearance: lit(1.0),
            ..fieldform_scene::Critical::new(join)
        });
        let program = generate_for_measurement(&doc, &graph, true, true).expect("generates");
        validate(&program.source).unwrap();
        let normal = code(&body(&program.source, "field_normal"));
        assert!(normal.contains("let was_live = tile_live;"), "the stencil saves it:\n{normal}");
        assert!(normal.contains("tile_live = was_live;"), "and puts it back:\n{normal}");
    }

    /// A part with one body still reports it, rather than leaving the index unset.
    #[test]
    fn a_single_body_part_still_names_its_body() {
        let mut graph = Graph::empty();
        graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(5.0) });
        let program = generate(&fieldform_scene::Document::new(), &graph).unwrap();
        validate(&program.source).unwrap();
        assert!(program.source.contains("owner = 0u;"));
    }

    /// A ramp dragged to an empty input range is an error, not a shader saying `inf`.
    ///
    /// The crash this is here for. An empty range has unbounded slope, `graph_lipschitz` says
    /// so honestly with infinity, and that arrived as a driven offset's divisor — where `{:?}`
    /// wrote it as `inf`. WGSL has no such literal, so it parsed as an undefined identifier;
    /// wgpu treats a shader that will not compile as fatal, and the application panicked
    /// pointing at line 900 of generated source rather than at the number that caused it.
    ///
    /// Reachable by dragging a ramp's From and To together in the panel, which is an ordinary
    /// thing to do while scrubbing two numbers.
    #[test]
    fn a_ramp_with_an_empty_input_range_is_refused_rather_than_written_as_inf() {
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane { origin: fieldform_scene::Place::At([lit(0.0); 3]), flipped: false,
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
        let body = graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });

        match generate(&doc, &graph) {
            Err(CodegenError::NotANumber(what)) => assert!(what.contains("offset")),
            Err(other) => panic!("expected the number to be named, got {other}"),
            Ok(program) => panic!(
                "a shader was generated for an unbounded ramp: {}",
                program.source,
            ),
        }
    }

    /// Every generated shader is parsed and validated before it is returned.
    ///
    /// Not an assertion about one scene but about the pipeline: `generate` runs the same naga
    /// front end wgpu does, so the only way to reach the driver with something unparseable is
    /// for this check to be removed. Checked by feeding it something that must not parse.
    #[test]
    fn generated_source_is_validated_before_it_can_reach_the_driver() {
        assert!(validated("fn field(p: vec3<f32>) -> f32 { return inf; }").is_err());
        // And a real one passes, so the check is not vacuously failing everything.
        let mut graph = Graph::empty();
        graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(5.0) });
        let program = generate(&fieldform_scene::Document::new(), &graph).unwrap();
        assert!(validated(&program.source).is_ok());
    }

    /// A driven offset generates a shader that reads the ramp and divides by the shared bound.
    ///
    /// The divisor is the assertion worth making here rather than in a picture: it scales the
    /// field without moving the surface, so a shader dividing by a different number than the
    /// CPU produces an identical image and a tracer that steps further than the field
    /// justifies. Only reading the generated source catches that.
    #[test]
    fn a_driven_offset_generates_a_ramp_and_the_same_divisor_the_cpu_uses() {
        let doc = fieldform_scene::Document::new();
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
            out_min: lit(0.0),
            out_max: lit(10.0),
            shape: fieldform_scene::RampShape::Linear,
        });
        let body = graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let offset =
            graph.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });

        let program = generate(&doc, &graph).unwrap();
        validate(&program.source).unwrap();
        assert!(program.source.contains("ramp("), "{}", program.source);

        // A body of Lipschitz 1 and a ramp of 10/40 = 0.25 make 1.25, and that is what the
        // shader has to divide by.
        let divisor = fieldform_field::driven_offset_divisor(&doc, &graph, offset);
        assert!((divisor - 1.25).abs() < 1e-9, "{divisor}");
        assert!(
            program.params.contains(&ParamRef::Literal(divisor)),
            "the shader should divide by {divisor}, from a slot: {:?}",
            program.params,
        );
        assert!(!program.source.contains(&format!("/ {divisor:?}")), "and not as a literal");
    }

    /// A ramp's numbers are generated constants, not parameter-buffer slots.
    ///
    /// Deliberate, and load-bearing: the divisor above is computed from those numbers at
    /// generation time, so a range that could change without a rebuild would leave the divisor
    /// stale — and a divisor that is too small is a tracer stepping through a wall.
    #[test]
    fn a_ramps_numbers_are_baked_rather_than_left_changeable_behind_the_divisor() {
        let doc = fieldform_scene::Document::new();
        let build = |out_max: f64| {
            let mut graph = Graph::empty();
            let plane = graph.add(Node::Plane { origin: fieldform_scene::Place::At([lit(0.0); 3]), flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let ramp = graph.add(Node::Ramp {
                input: plane.into(),
                in_min: lit(0.0),
                in_max: lit(40.0),
                out_min: lit(0.0),
                out_max: lit(out_max),
                shape: fieldform_scene::RampShape::Linear,
            });
            let body = graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(40.0); 3],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });
            generate(&doc, &graph).unwrap()
        };
        let a = build(10.0);
        let b = build(20.0);
        // Same parameter count, different source: the numbers moved into the code rather than
        // into the buffer, which is what forces a rebuild when they change.
        assert_eq!(a.params.len(), b.params.len());
        assert_ne!(a.source, b.source, "changing a ramp must change the shader");
    }

    /// Each easing generates its own code, and the three differ.
    ///
    /// The shape rides into WGSL as a float because the shader has no enums, and a comparison
    /// written the wrong way round would silently pick one easing for all three — which looks
    /// plausible and would put every driven surface in the wrong place by a bounded amount.
    #[test]
    fn each_ramp_shape_generates_a_different_shader() {
        let doc = fieldform_scene::Document::new();
        let sources: Vec<String> = fieldform_scene::RampShape::ALL
            .iter()
            .map(|shape| {
                let mut graph = Graph::empty();
                let plane = graph.add(Node::Plane { origin: fieldform_scene::Place::At([lit(0.0); 3]), flipped: false,
                    turn: fieldform_scene::no_turn(),
                });
                let ramp = graph.add(Node::Ramp {
                    input: plane.into(),
                    in_min: lit(0.0),
                    in_max: lit(40.0),
                    out_min: lit(0.0),
                    out_max: lit(10.0),
                    shape: *shape,
                });
                let body =
                    graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(40.0); 3],
                        turn: fieldform_scene::no_turn(),
                    });
                graph.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });
                let program = generate(&doc, &graph).unwrap();
                validate(&program.source).unwrap();
                program.source
            })
            .collect();
        assert_ne!(sources[0], sources[1]);
        assert_ne!(sources[1], sources[2]);
        assert_ne!(sources[0], sources[2]);
    }

    /// A fixed offset still keeps its distance in the parameter buffer, so scrubbing it does
    /// not rebuild the shader.
    ///
    /// The other half of the trade above: only a *driven* offset pays for a rebuild, because
    /// only a driven one has a divisor tied to the number.
    #[test]
    fn a_fixed_offsets_distance_stays_a_parameter() {
        let doc = fieldform_scene::Document::new();
        let build = |d: f64| {
            let mut graph = Graph::empty();
            let body = graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(40.0); 3],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Offset { input: body.into(), distance: lit(d) });
            generate(&doc, &graph).unwrap()
        };
        let (a, b) = (build(5.0), build(9.0));
        assert_eq!(a.source, b.source, "a fixed offset should not rebuild the shader");
        assert_ne!(a.params, b.params);
    }


    /// The shader's gradient bounds are the CPU's, for every lattice kind.
    ///
    /// The bound is the number a lattice's value is divided by, and it lives in two languages:
    /// `TpmsKind::gradient_bound` on the CPU and `tpms_gradient_bound` in `prelude.wgsl`. They
    /// were written to agree and nothing checked that they still did — so tightening the
    /// gyroid's on one side left the two dividing by different numbers.
    ///
    /// **`gpu_matches_cpu` cannot catch this**, which is why it needs a test of its own: the
    /// bound scales the value without moving its zero set, so both paths put the surface in
    /// exactly the same place and the residual stays at the tracer's threshold. What differs is
    /// how far a ray is told it may step — invisible in a still image, and the difference
    /// between a wall being stepped through or not if the numbers ever cross the wrong way.
    #[test]
    fn the_shader_divides_a_lattice_by_the_same_bound_the_cpu_does() {
        // Parsed from the prelude rather than duplicated here, since a third copy of these
        // numbers would be the very thing this is guarding against.
        let source = PRELUDE;
        let start = source
            .find("fn lattice_gradient_bound")
            .expect("the prelude declares the bound");
        // To its own closing brace, not a fixed window: the next function also returns
        // constants, and a window that reached them would be reading the wrong numbers.
        let end = start + source[start..].find("
}").expect("the function closes");
        let body = &source[start..end];

        // The shader's own branch structure, read out of it rather than restated here — a third
        // copy of these constants would be the very thing this is guarding against. Each entry
        // is a `cell < N` guard and the value returned under it; the last is the fallthrough.
        let mut guards: Vec<(f64, f64)> = Vec::new();
        let mut fallthrough: Option<f64> = None;
        let mut pending: Option<f64> = None;
        for line in body.lines().map(str::trim) {
            if let Some(rest) = line.strip_prefix("if cell < ").and_then(|r| r.strip_suffix(" {")) {
                pending = rest.parse::<f64>().ok();
            } else if let Some(value) = line
                .strip_prefix("return ")
                .and_then(|r| r.strip_suffix(';'))
                .and_then(|n| n.parse::<f64>().ok())
            {
                match pending.take() {
                    Some(threshold) => guards.push((threshold, value)),
                    None => fallthrough = Some(value),
                }
            }
        }
        let fallthrough = fallthrough.expect("the function returns something unguarded");

        // **Checking the mapping, not the set.** This test used to assert only that each CPU
        // bound was *somewhere* among the shader's returns, which was all the shape allowed
        // while every kind shared `sqrt(3)` — and it said out loud that a kind with its own
        // number would have to restore the branch. Three of them arrived, so it does: the
        // shader's guards are walked with each kind's own index, and a bound that is right for
        // some other cell no longer passes.
        // Every cell, beams included: a beam's bound is 1 where a surface's is between
        // sqrt(3) and 7·sqrt(3), so this is exactly where the two families must not be confused.
        for kind in EVERY_LATTICE_CELL {
            let index = lattice_cell_index(kind);
            let shader = guards
                .iter()
                .find(|(threshold, _)| index < *threshold)
                .map(|(_, value)| *value)
                .unwrap_or(fallthrough);
            let cpu = fieldform_field::CellKind::from(kind).gradient_bound();
            assert!(
                (cpu - shader).abs() < 1e-5,
                "{kind:?} (cell {index}): the CPU divides by {cpu}, the shader by {shader}",
            );
        }
    }

    /// The shader's density table is the CPU's, entry for entry, for every lattice cell.
    ///
    /// A density is not a length, so unlike a wall thickness it cannot be converted during
    /// codegen and handed over as a number — folding it would freeze a density driven by a named
    /// parameter into the shader. So the table is transcribed into WGSL, and a transcribed table
    /// is exactly the thing that drifts: `the_shader_divides_a_lattice_by_the_same_bound_the_cpu_
    /// does` exists because a constant in two languages already drifted once.
    ///
    /// **`gpu_matches_cpu` cannot catch this either**, and for a sharper reason than it cannot
    /// catch the bound. A wrong entry moves the surface, so the two paths would disagree — but
    /// only for a *solid* lattice at a density between the stored tenths, and every scene it
    /// checks is a sheet. A table that is wrong in one entry looks like a lattice from every
    /// angle.
    #[test]
    fn the_shader_reads_a_density_through_the_same_table_the_cpu_does() {
        // Parsed out of the prelude rather than restated here, since a third copy of these
        // numbers would be the very thing this is guarding against.
        let source = PRELUDE;
        let start = source
            .find("fn tpms_density_entry")
            .expect("the prelude declares the density table");
        let end = start + source[start..].find("\n}").expect("the function closes");
        let body = &source[start..end];

        let table = |name: &str| -> Vec<f64> {
            let at = body
                .find(&format!("var {name} = array<f32, 11>("))
                .unwrap_or_else(|| panic!("the prelude declares {name}: {body}"));
            let open = at + body[at..].find('(').expect("the array opens");
            let close = open + body[open..].find(')').expect("it closes");
            body[open + 1..close]
                .split(',')
                .map(|n| n.trim())
                .filter(|n| !n.is_empty())
                .map(|n| n.parse::<f64>().expect("a number"))
                .collect()
        };

        for (kind, name) in [
            (fieldform_field::TpmsKind::Gyroid, "gyroid"),
            (fieldform_field::TpmsKind::SchwarzP, "schwarz_p"),
            (fieldform_field::TpmsKind::SchwarzD, "schwarz_d"),
        ] {
            let shader = table(name);
            let cpu = kind.density_isovalues();
            assert_eq!(
                shader.len(),
                cpu.len(),
                "{kind:?}: the shader carries {} entries and the CPU {}",
                shader.len(),
                cpu.len(),
            );
            for (i, (s, c)) in shader.iter().zip(cpu.iter()).enumerate() {
                assert!(
                    (s - c).abs() < 1e-6,
                    "{kind:?} at {}% density: the shader reads {s}, the CPU {c}",
                    i * 10,
                );
            }
        }
    }

    /// A pattern's shader is the same size at five hundred copies as at six.
    ///
    /// **The property the fold exists for, and the one the old lowering could not have.** Every
    /// instance used to be written out — a hundred copies emitted the body a hundred times —
    /// which is a mesh instancing cost wearing a field's clothes, and this codebase has already
    /// met the end of that road once, at two thousand bodies and a twenty-two-million-character
    /// expression. A field should not care: a lattice folds a point into one cell and asks
    /// once, which is why a gyroid finer than any mesher could hold traces in a fraction of a
    /// second, and a pattern's instances are just as regular.
    ///
    /// Compared by *length*, because that is what grew. The two sources differ only in the
    /// digits of the count itself, so a handful of characters is the whole allowance.
    #[test]
    fn a_patterns_shader_is_the_same_size_however_many_copies_there_are() {
        let source = |count: u32| {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(4.0) });
            graph.add(Node::Pattern {
                options: Default::default(),
                input: ball.into(),
                count: ParamRef::Literal(f64::from(count)),
                kind: fieldform_scene::PatternKind::Linear {
                    step: [lit(20.0), lit(0.0), lit(0.0)],
                },
                join: true,
                fillet: Fillet::hard(),
            });
            generate(&fieldform_scene::Document::new(), &graph)
                .expect("generates")
                .source
                .len()
        };

        let few = source(6);
        let many = source(500);
        assert!(
            many <= few + 8,
            "six copies generate {few} characters and five hundred generate {many}",
        );
    }

    /// The body is emitted once per step of the window, not once per copy.
    ///
    /// The window is what the fold looks through — the cell a point falls in, and the few either
    /// side that could still hold the nearest material — so a sphere that fits comfortably
    /// inside its own pitch is written three times whatever the count is. Counting the sphere
    /// calls says which of the two shapes the shader has.
    #[test]
    fn a_patterns_body_is_emitted_once_per_window_step() {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(4.0) });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(200.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(40.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let program = generate(&fieldform_scene::Document::new(), &graph).expect("generates");
        let bodies = program.source.matches("sphere_distance(").count();

        // The window is a handful of steps and the body is written once per step in each of the
        // two passes -- the value and the normal -- so the honest bound is a small constant
        // rather than one. What it must not be is two hundred.
        assert!(bodies <= 16, "expected a windowful of instances, got {bodies}");

        // And the same number at six copies as at two hundred, which is the claim: the shader's
        // shape follows the body's size, not the count.
        let mut small = Graph::empty();
        let ball = small.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(4.0) });
        small.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(6.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(40.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let fewer = generate(&fieldform_scene::Document::new(), &small)
            .expect("generates")
            .source
            .matches("sphere_distance(")
            .count();
        assert_eq!(bodies, fewer, "the count changed how much shader there is");
    }

    /// A linear pattern's copies land a step apart, and the last one is where it should be.
    ///
    /// The arithmetic moved from the lowering into the shader when patterns stopped being
    /// expanded, so what used to be checkable by counting `Transform` nodes is now checkable
    /// only by asking the field. Which is the better question anyway: where the copies *are* is
    /// the claim, and how they were arranged is an implementation of it.
    #[test]
    fn a_linear_patterns_copies_land_a_step_apart() {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(4.0) });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(6.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(20.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let doc = fieldform_scene::Document::new();
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 1e-3)
            .expect("compiles");
        use fieldform_field::ScalarField;
        for i in 0..6 {
            let at = nalgebra::Point3::new(20.0 * i as f64, 0.0, 0.0);
            assert!(field.distance(at) < 0.0, "copy {i} should be solid at its centre");
        }
        // And it stops: there is no seventh.
        assert!(field.distance(nalgebra::Point3::new(120.0, 0.0, 0.0)) > 0.0);
    }

    /// A count of one is the body, with nothing arranged around it.
    #[test]
    fn a_pattern_of_one_is_the_body_alone() {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(4.0) });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(1.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(20.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let doc = fieldform_scene::Document::new();
        use fieldform_field::ScalarField;
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 1e-3)
            .expect("compiles");
        assert!(field.distance(nalgebra::Point3::origin()) < 0.0, "the body is there");
        assert!(
            field.distance(nalgebra::Point3::new(20.0, 0.0, 0.0)) > 0.0,
            "and there is no second copy",
        );
        validate(&generate(&doc, &graph).unwrap().source).unwrap();
    }

    /// A ring of `input`, six round, hard.
    fn ring_of(graph: &mut Graph, input: fieldform_scene::NodeId) {
        graph.add(Node::Pattern {
            options: Default::default(),
            input: input.into(),
            count: ParamRef::Literal(6.0),
            kind: fieldform_scene::PatternKind::Radial { frame: fieldform_scene::Frame::default() },
            join: true,
            fillet: Fillet::hard(),
        });
    }

    /// A pattern that is pruned says so in the source, and one that is not says why.
    ///
    /// The floor `(|q - c| - rho) / D` is only sound beneath a sub-field that never reads below
    /// it, which an intersection does not promise: its value is the larger operand's, and that
    /// may be a lower bound of anything. So the claim is twofold -- the prune is there when it
    /// can be, and where it cannot be the loop is the old loop with the reason written beside
    /// it, which is the only place anyone would look.
    #[test]
    fn a_pattern_loop_says_whether_it_is_pruned_and_why_not() {
        let doc = fieldform_scene::Document::new();
        let ball = |graph: &mut Graph, x: f64| {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(0.0)]),
                radius: lit(4.0),
            })
        };

        let mut pruned = Graph::empty();
        let b = ball(&mut pruned, 30.0);
        ring_of(&mut pruned, b);
        let with = generate_with_options(&doc, &pruned, true, true).unwrap();
        assert!(with.source.contains("pruned by the sub-field's sphere"));
        assert!(with.source.contains("if (pattern_floor("));
        validate(&with.source).unwrap();

        // Switched off, the same graph is the old loop and five parameters shorter.
        let without = generate_with_options(&doc, &pruned, true, false).unwrap();
        assert!(!without.source.contains("if (pattern_floor("));
        assert_eq!(with.params.len(), without.params.len() + PRUNE_SLOTS);
        validate(&without.source).unwrap();

        let mut refused = Graph::empty();
        let (l, r) = (ball(&mut refused, 30.0), ball(&mut refused, 33.0));
        let lens = refused.add(Node::Boolean {
            op: fieldform_scene::BooleanOp::Intersection,
            inputs: vec![l.into(), r.into()],
        });
        ring_of(&mut refused, lens);
        let source = generate_with_options(&doc, &refused, true, true).unwrap().source;
        assert!(source.contains("loop, unpruned: "), "the reason is in the source");
        assert!(!source.contains("if (pattern_floor("));
        validate(&source).unwrap();
    }

    /// Reshaping a spine is a buffer write, not a new shader.
    ///
    /// Reported as "dragging the point on a sweeping curve is painfully slow". A sweep's stretch
    /// is the convergence of its section planes on its tightest bend, so it moves with every
    /// control point -- and it was written into the source as a number, which made each step of
    /// the drag a shader the driver had never seen: 1.1 to 1.2 s of pipeline a step, measured in
    /// the application, against 6 ms for a source it had.
    #[test]
    fn bending_a_sweeps_spine_does_not_change_the_shader() {
        use fieldform_scene::{Curve, Document};
        use nalgebra::Point3;
        let source_for = |bulge: f64| {
            let mut doc = Document::new();
            let profile = fieldform_scene::CurveId::new();
            doc.curves.insert(
                profile,
                Curve::polyline(
                    profile,
                    "section",
                    vec![
                        Point3::new(-3.0, -3.0, 0.0),
                        Point3::new(3.0, -3.0, 0.0),
                        Point3::new(3.0, 3.0, 0.0),
                        Point3::new(-3.0, 3.0, 0.0),
                    ],
                    true,
                )
                .expect("a square"),
            );
            let spine = fieldform_scene::CurveId::new();
            doc.curves.insert(
                spine,
                Curve::polyline(
                    spine,
                    "spine",
                    vec![
                        Point3::new(0.0, 0.0, 0.0),
                        Point3::new(40.0, bulge, 0.0),
                        Point3::new(80.0, 0.0, 0.0),
                    ],
                    false,
                )
                .expect("a bent line"),
            );
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
            let program = generate(&doc, &graph).expect("a sweep generates");
            (program.source, fieldform_field::sweep_stretch(&doc, &fieldform_scene::Outline::closed(profile), spine, &Default::default()))
        };
        let (gentle, a) = source_for(5.0);
        let (sharp, b) = source_for(25.0);
        assert!(a != b, "the premise: the two spines bend differently, {a} and {b}");
        // Curve ids are fresh each time and are not in the source either, or this would fail
        // for the wrong reason.
        assert_eq!(gentle, sharp);
    }

    /// A sweep walks its spine once per point it is asked about, not eight times.
    ///
    /// **This is the whole of the owner's five-to-thirty-second freeze.** The driver unrolls the
    /// spine's index into every call site and its compile time is worse than cubic in how much
    /// it has to unroll, so what matters is the number of call sites, not what they cost to run.
    /// The value pass asks once. The normal pass used to ask eight times for the same point —
    /// once for the value and once for each arm of a central difference — and `wing_compile_cost`
    /// measured that at 8419 ms against 3756 for this.
    ///
    /// Counted in the generated half only: the prelude declares the walk once and that copy is
    /// not unrolled anywhere.
    #[test]
    fn a_sweep_walks_its_spine_once_for_a_value_and_once_for_a_normal() {
        use fieldform_scene::{Curve, Document};
        use nalgebra::Point3;
        let mut doc = Document::new();
        let profile = fieldform_scene::CurveId::new();
        doc.curves.insert(
            profile,
            Curve::polyline(
                profile,
                "section",
                vec![
                    Point3::new(-3.0, -3.0, 0.0),
                    Point3::new(3.0, -3.0, 0.0),
                    Point3::new(3.0, 3.0, 0.0),
                    Point3::new(-3.0, 3.0, 0.0),
                ],
                true,
            )
            .expect("a square"),
        );
        let spine = fieldform_scene::CurveId::new();
        doc.curves.insert(
            spine,
            Curve::polyline(
                spine,
                "spine",
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 80.0)],
                false,
            )
            .expect("a line"),
        );
        let mut graph = Graph::empty();
        graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
        let source = generate(&doc, &graph).expect("a sweep generates").source;
        let cut = source
            .find("// ---- generated from the composition graph ----")
            .expect("the generated half is marked");
        let generated = &source[cut..];
        assert_eq!(
            generated.matches("sweep_distance(").count(),
            1,
            "the value pass asks once and nothing else asks at all"
        );
        assert_eq!(
            generated.matches("sweep_value_normal_bake(").count(),
            1,
            "the normal pass asks once, for the value and the normal together"
        );
    }

    /// Remembering which sources passed never turns a failure into a pass.
    ///
    /// `validated` skips the parse for a source it has already accepted, which is most of what
    /// an edit step cost. What it must not do is accept one it has not: wgpu's answer to a
    /// shader that does not compile is to take the process down.
    #[test]
    fn a_source_is_remembered_only_once_it_has_passed() {
        let good = generate(&fieldform_scene::Document::new(), &{
            let mut graph = Graph::empty();
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(7.25),
            });
            graph
        })
        .expect("a sphere generates")
        .source;
        assert!(validated(&good).is_ok());
        assert!(validated(&good).is_ok(), "and again, from memory");
        let bad = format!("{good}\nfn broken() -> f32 {{ return inf; }}\n");
        assert!(validated(&bad).is_err());
        assert!(validated(&bad).is_err(), "a failure is not remembered as anything");
        // Nor does having seen its prefix pass make it pass.
        assert!(validated(&good).is_ok());
        assert!(validated(&bad).is_err());
    }

    /// A pattern's copies are masked only where the argument for masking them holds.
    ///
    /// The mask rules a copy out of a cell of the *world* from one reading at the cell's centre,
    /// which is sound for a hard `min` over copies that see the world's own point and for
    /// nothing else as it stands: a rounded fold sums its copies, and a pattern under a
    /// placement sees some other region than the cell. So the first kind gets the skip and the
    /// record, the other two get neither, and all three still validate -- which they would not
    /// if the two passes had come to count the mask's slot differently.
    #[test]
    fn only_a_hard_pattern_at_the_worlds_own_point_has_its_copies_masked() {
        let doc = fieldform_scene::Document::new();
        let ring = |fillet: Fillet, placed: bool| {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
                radius: lit(4.0),
            });
            let pattern = graph.add(Node::Pattern {
                options: Default::default(),
                input: ball.into(),
                count: ParamRef::Literal(6.0),
                kind: fieldform_scene::PatternKind::Radial { frame: fieldform_scene::Frame::default() },
                join: true,
                fillet,
            });
            if placed {
                graph.add(Node::Transform {
                    input: pattern.into(),
                    translation: [lit(5.0), lit(0.0), lit(0.0)],
                    rotation: [lit(0.0); 3],
                    scale: lit(1.0),
                    about: fieldform_scene::Place::At([lit(0.0); 3]),
                });
            }
            let program = generate(&doc, &graph).expect("generates");
            validate(&program.source).expect("and validates");
            program
        };
        let generated = |program: &ShaderProgram| {
            program.source.strip_prefix(PRELUDE).expect("the prelude comes first").to_string()
        };

        let hard = ring(Fillet::hard(), false);
        let source = generated(&hard);
        assert!(source.contains("if (copy_masked("), "the skip is in the loop");
        assert!(source.contains("if (mask_mode == 1u)"), "and the record after it");

        let rounded = ring(Fillet { radius: lit(2.0), continuity: Continuity::G1, second: None }, false);
        assert!(!generated(&rounded).contains("copy_masked("));
        assert!(!generated(&rounded).contains("if (mask_mode == 1u)"));

        let placed = ring(Fillet::hard(), true);
        assert!(!generated(&placed).contains("copy_masked("));
        assert!(!generated(&placed).contains("if (mask_mode == 1u)"));

    }

    /// Every mask bit says which node it stands for, in the order the shader numbers them.
    ///
    /// The bits are the only handle on whether masking a *particular* operand ever fires, and
    /// counting `masked_operand_N` in the source cannot say which block put one there. Read
    /// against the grid by `fieldform-gpu/examples/mask_fire_rate.rs`.
    #[test]
    fn a_program_names_the_node_behind_every_mask_bit() {
        let doc = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let pipe = |graph: &mut Graph| graph.add(Node::Pipe { curve: CurveId::new(), radius: lit(5.0) });
        let (a, b) = (pipe(&mut graph), pipe(&mut graph));
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        let program = generate_with_masks(&doc, &graph, true).expect("generates");
        assert_eq!(program.masked, vec![a, b]);
        for (bit, id) in program.masked.iter().enumerate() {
            assert!(program.source.contains(&format!("fn masked_operand_{bit}(")), "{id:?}");
        }

        // And nothing to name where nothing is masked.
        let mut cheap = Graph::empty();
        let ball = |graph: &mut Graph, x: f64| {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(0.0)]),
                radius: lit(4.0),
            })
        };
        let (a, b) = (ball(&mut cheap, 0.0), ball(&mut cheap, 20.0));
        cheap.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        assert!(generate_with_masks(&doc, &cheap, true).expect("generates").masked.is_empty());

        // And nothing at all with the feature switched off.
        assert!(generate_with_masks(&doc, &graph, false).expect("generates").masked.is_empty());
    }

    /// A mask counts only the work its own operand does, not work the join reads anyway.
    ///
    /// Two pipes unioned are each read by the union alone, so each is worth a mask. Put an
    /// offset of the *first* pipe in beside them and the first pipe is read twice: masking the
    /// offset can skip the offset's own subtraction and nothing else, because `closure_of`
    /// copies the pipe into the offset's function and the union reads the pipe regardless.
    /// This is what `mask_fire_rate` found on WING-2 -- bits set in 71.5% of the near cells,
    /// buying nothing.
    #[test]
    fn a_mask_counts_only_the_work_its_own_operand_does() {
        let doc = fieldform_scene::Document::new();
        let pipe = |graph: &mut Graph| {
            graph.add(Node::Pipe { curve: CurveId::new(), radius: lit(5.0) })
        };

        let mut apart = Graph::empty();
        let (a, b) = (pipe(&mut apart), pipe(&mut apart));
        apart.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        assert_eq!(generate(&doc, &apart).expect("generates").masked, vec![a, b]);

        let mut shared = Graph::empty();
        let (a, b) = (pipe(&mut shared), pipe(&mut shared));
        let skin = shared.add(Node::Offset { input: a.into(), distance: lit(-2.0) });
        shared.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into(), skin.into()] });
        let masked = generate(&doc, &shared).expect("generates").masked;
        assert!(!masked.contains(&skin), "the offset saves one subtraction, not a pipe walk");
        assert!(!masked.contains(&a), "and the pipe two of them read is nobody's to skip");
        assert_eq!(masked, vec![b]);
    }

    /// A join masks only the operand it drops for reading *far*.
    ///
    /// An intersection drops an operand for reading very negative -- deep inside its own solid,
    /// which is where the march hardly goes; so does the kept side of a difference. Measured on
    /// WING-2, whose top difference keeps a blend that the grid ruled out in 0.0% of its cells
    /// on both grids. The cut operands of a difference and both sides of a union are dropped
    /// for reading far, and keep their masks.
    #[test]
    fn a_join_masks_only_the_operand_it_drops_for_reading_far() {
        let doc = fieldform_scene::Document::new();
        let pipes = |graph: &mut Graph| {
            let pipe =
                || Node::Pipe { curve: CurveId::new(), radius: lit(5.0) };
            (graph.add(pipe()), graph.add(pipe()))
        };
        let masked = |op: BooleanOp| {
            let mut graph = Graph::empty();
            let (a, b) = pipes(&mut graph);
            graph.add(Node::Boolean { op, inputs: vec![a.into(), b.into()] });
            (generate(&doc, &graph).expect("generates").masked, a, b)
        };

        let (union, a, b) = masked(BooleanOp::Union);
        assert_eq!(union, vec![a, b], "both sides of a union are dropped for reading far");

        let (intersection, _, _) = masked(BooleanOp::Intersection);
        assert!(intersection.is_empty(), "an intersection drops for reading deep inside");

        let (difference, _, b) = masked(BooleanOp::Difference);
        assert_eq!(difference, vec![b], "a difference keeps the first and masks what it cuts");
    }

    /// A dear operand of a join is read through a mask, and a cheap one is not.
    ///
    /// Two pipes in a union: each is a walk of its spine, and over most of space only one of
    /// them can be the nearer, so each becomes a function the join calls unless the cell has
    /// ruled it out -- and leaves a record for the pass that fills the cells. Two spheres in
    /// the same union are cheaper to evaluate than to ask about, and the source says nothing
    /// of masks at all. Blended instead of unioned, the pipes are masked the same way, with the
    /// radius in the record, since a blend reaches that much further than a `min` does.
    #[test]
    fn a_joins_dear_operands_are_masked_and_its_cheap_ones_are_not() {
        let doc = fieldform_scene::Document::new();
        // Asked for rather than defaulted: the masks are off on measurement (see
        // `OperandMasks`), and what they emit when asked for is still theirs to hold.
        let generated = |graph: &Graph| {
            let program = generate_with_masks(&doc, graph, true).expect("generates");
            validate(&program.source).expect("and validates");
            program.source.strip_prefix(PRELUDE).expect("the prelude comes first").to_string()
        };
        let pipes = |graph: &mut Graph| {
            let pipe = |graph: &mut Graph| {
                graph.add(Node::Pipe { curve: CurveId::new(), radius: lit(5.0) })
            };
            (pipe(graph), pipe(graph))
        };

        let mut unioned = Graph::empty();
        let (a, b) = pipes(&mut unioned);
        unioned.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        let source = generated(&unioned);
        for bit in 0..2 {
            assert!(source.contains(&format!("fn masked_operand_{bit}(")), "{source}");
            assert!(source.contains(&format!("if (!operand_masked({bit}u))")));
            assert!(source.contains(&format!("ops_seen = ops_seen | (1u << {bit}u)")));
        }
        // What a `min` ignores.
        assert!(source.contains("var mo0 = 1e30;"));

        let mut subtracted = Graph::empty();
        let (a, b) = pipes(&mut subtracted);
        subtracted.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![a.into(), b.into()] });
        let source = generated(&subtracted);
        // A difference is a `max` of the first and the others negated, so what it ignores is a
        // very small first operand and a very large subtracted one. Only the subtracted ones are
        // masked -- the first is dropped for reading deep inside itself, where the march hardly
        // goes -- so the one mask is the cut tool, and it stands in as very large.
        assert!(source.contains("var mo0 = 1e30;"), "{source}");
        assert!(!source.contains("var mo1 ="), "the kept body is not masked");

        let mut blended = Graph::empty();
        let (a, b) = pipes(&mut blended);
        blended.add(Node::Blend {
            op: BooleanOp::Union,
            a: a.into(),
            b: b.into(),
            fillet: Fillet { radius: lit(3.0), continuity: Continuity::G1, second: None },
        });
        let source = generated(&blended);
        assert!(source.contains("fn masked_operand_1("));
        assert!(source.contains("if (mask_mode == 1u)"), "the blend leaves its record");

        let mut cheap = Graph::empty();
        let ball = |graph: &mut Graph, x: f64| {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(0.0)]),
                radius: lit(4.0),
            })
        };
        let (a, b) = (ball(&mut cheap, 0.0), ball(&mut cheap, 20.0));
        cheap.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        let source = generated(&cheap);
        assert!(!source.contains("masked_operand_") && !source.contains("operand_masked("));
    }

    /// A named step drives a pattern, resolved where the spacing is worked out.
    ///
    /// The multiplying moved into the shader when patterns stopped being expanded, so this asks
    /// the field where the copies landed rather than reading placements out of a lowered graph.
    /// The claim is the same one either way: a parameter that says fifteen spaces them fifteen
    /// apart.
    #[test]
    fn a_patterns_named_step_is_resolved_where_the_multiplying_happens() {
        let spacing = fieldform_scene::ParamId::new();
        let mut doc = fieldform_scene::Document::new();
        doc.parameters.insert(
            spacing,
            fieldform_scene::Parameter {
                id: spacing,
                name: "spacing".into(),
                value: fieldform_scene::ParamValue::Literal(15.0),
            },
        );
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(4.0) });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(3.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [ParamRef::Named(spacing), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        use fieldform_field::ScalarField;
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 1e-3)
            .expect("compiles");
        for i in 0..3 {
            let at = nalgebra::Point3::new(15.0 * i as f64, 0.0, 0.0);
            assert!(field.distance(at) < 0.0, "copy {i} should be solid at its centre");
        }
        assert!(field.distance(nalgebra::Point3::new(45.0, 0.0, 0.0)) > 0.0, "and no fourth");
        validate(&generate(&doc, &graph).unwrap().source).unwrap();
    }

    /// A pattern of 512 is the ceiling the panel names, and it has to generate.
    ///
    /// The count is capped because every instance is written out. This is what that costs, and
    /// it is the number the panel's tooltip quotes: **489,512 characters** — 478 KB — for 512
    /// copies of a sphere, which is the cheapest body there is. The bound below is loose on
    /// purpose; it is there to catch a lowering that starts growing with the square of the
    /// count, not to pin the exact size.
    #[test]
    fn the_largest_pattern_the_panel_allows_still_generates() {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(4.0) });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: ball.into(),
            count: ParamRef::Literal(512.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(9.0), lit(0.0), lit(0.0)],
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
        let program = generate(&fieldform_scene::Document::new(), &graph).unwrap();
        validate(&program.source).unwrap();
        assert!(
            program.source.len() < 1_000_000,
            "{} characters for 512 copies",
            program.source.len(),
        );
    }


    /// A shader for a thousand bodies stays a straight line rather than a nested expression.
    ///
    /// The value fold nested one `min` per body and the normal fold was worse than deep — it
    /// was *quadratic in length*, because the running minimum appeared inside each step's
    /// condition as well as inside the next one's. Simulated at 2 000 bodies that expression
    /// alone reached 22.3 million characters, and generating 10 000 did not finish in ten
    /// minutes. Flattened it is about 435 characters a body.
    ///
    /// The assertion is on length rather than on nesting because length is what actually broke
    /// — and a bound this loose still fails by orders of magnitude if the quadratic form
    /// returns.
    #[test]
    fn a_thousand_bodies_generate_a_shader_that_grows_with_them_not_with_their_square() {
        let mut graph = Graph::empty();
        for i in 0..1_000 {
            graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(i as f64 * 3.0),
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(0.0),
                ]),
                radius: ParamRef::Literal(1.0),
            });
        }
        let program = generate(&fieldform_scene::Document::new(), &graph).expect("a thousand bodies generate");
        // The generated part alone: the prelude in front of it is fixed text, and grew past the
        // margin this bound left when the profile solids were instantiated a second time.
        assert!(
            program.field_function().len() < 1_000_000,
            "{} characters for 1000 bodies — the fold is nesting again",
            program.field_function().len(),
        );
    }


    /// **A node two bodies share is written into both, and the copies are the same line.**
    ///
    /// `notebook-spec.md` §15's shared-instance bullet, and the detector it carries since that
    /// bullet was corrected on 2026-10-17. The bullet used to say the generated source evaluates a
    /// shared node once, which `e4b3b5f` made false: a node feeding two bodies is written into
    /// both, measured and kept. So what stops a shared instance diverging is not one evaluation --
    /// there are two -- but one *node*. The slot map is keyed by node across the whole program, so
    /// the line reading it carries the same `n` and the same `params` wherever it is written, and
    /// two copies that could drift apart would be two blocks wearing one name.
    ///
    /// Nothing else asserts this. `a_shared_pipe_or_a_shared_spine_is_not_walked_as_a_run` counts
    /// the copies, which is the interning question rather than this one, and
    /// `the_same_graph_always_generates_the_same_source` compares whole programs across runs
    /// rather than two copies within one.
    ///
    /// The shells are deliberately unequal, so the comparison cannot pass by the two bodies being
    /// the same function: one subtracts `params[4]` and the other `params[5]`, while the sphere's
    /// own line is `params[0..3]` in both.
    #[test]
    fn a_node_two_bodies_share_is_written_into_both_as_the_same_line() {
        use fieldform_scene::{Ownership, Part, PartId, Place};

        // One sphere, two shells of different thickness reading it: the shells are the bodies and
        // the sphere is the shared instance, which is the §7.3 shape rather than a contrived one.
        let mut document = fieldform_scene::Document::new();
        let mut graph = Graph::empty();
        let sphere = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(7.0),
        });
        let thin = graph.add(Node::Shell { input: sphere.into(), thickness: lit(1.0) });
        let thick = graph.add(Node::Shell { input: sphere.into(), thickness: lit(2.0) });
        for (i, id) in [sphere, thin, thick].iter().enumerate() {
            document.created.insert(*id, 1_700_000_000_000_000 + i as u64 + 1);
        }
        document.add_part(Part {
            id: PartId::new(),
            name: "Part".into(),
            material: None,
            graph: graph.clone(),
            inputs: Vec::new(),
        });
        document.settle_ownership();

        // It really is one block two bodies read, or the rest of the test is about nothing. Which
        // declares it (R1 inverted 2026-10-24, `notebook-spec.md` §17) — and that changes nothing
        // here: what §15 is about is the *generated source*, where one node feeding two bodies is
        // written into both either way. The ownership is asserted only to pin the setup.
        assert_eq!(
            document.ownership_nodes.get(&sphere),
            Some(&Ownership::Variable),
            "two readers declare it",
        );
        let nesting = document.nesting_in(&graph);
        assert_eq!(
            nesting.owners_of(sphere).to_vec(),
            Vec::new(),
            "so no slot holds it: both shells reference it",
        );
        let mut readers = vec![thin, thick];
        readers.sort();
        assert_eq!(nesting.readers_of(sphere), 2, "and both of them read it");
        let _ = readers;
        assert_eq!(document.folded_bodies(&graph), vec![thin, thick], "the shells are the bodies");

        let source = generate(&document, &graph).unwrap().source;
        let first = body_function(&source, 0);
        let second = body_function(&source, 1);

        // The shared node's own line, found the way `a_shared_pipe_or_a_shared_spine_is_not_walked
        // _as_a_run` finds its calls: by the leaf's emitted name, not by a slot number, which
        // follows evaluation order and is not the thing under test.
        let shared_lines = |body: &str| -> Vec<String> {
            body.lines()
                .filter(|l| l.contains("sphere_distance("))
                .map(|l| l.trim().to_string())
                .collect()
        };
        let in_first = shared_lines(&first);
        let in_second = shared_lines(&second);
        assert_eq!(in_first.len(), 1, "written into the first body once:\n{first}");
        assert_eq!(in_second.len(), 1, "written into the second body once:\n{second}");
        assert_eq!(
            in_first, in_second,
            "the copies of one shared node differ, so the two readers no longer read one block:\n\
             body_0: {in_first:?}\nbody_1: {in_second:?}",
        );

        // And the equality above is a fact about that one line rather than about the bodies, which
        // are never the same text here: each shell owns its own slot and its own `params` index, so
        // they differ even given equal thicknesses. What rules out the equality passing on two
        // *empty* lists is the pair of counts above, not this.
        assert_ne!(first, second, "each shell should carry its own slots");

        // The other half of the bullet: an edit to the shared block reaches every reader. Edited
        // in the graph, because a radius is a `params` slot the driver writes rather than text --
        // so the edit that *can* show here is a structural one.
        let mut edited = graph.clone();
        *edited.node_mut(sphere).expect("the shared block is in the graph") = Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            size: [lit(14.0), lit(14.0), lit(14.0)],
            turn: fieldform_scene::no_turn(),
        };
        let after = generate(&document, &edited).unwrap().source;
        let (one, two) = (body_function(&after, 0), body_function(&after, 1));
        let box_lines = |body: &str| -> Vec<String> {
            body.lines()
                .filter(|l| l.contains("cuboid_distance("))
                .map(|l| l.trim().to_string())
                .collect()
        };
        assert_eq!(box_lines(&one).len(), 1, "the edit reached the first reader:\n{one}");
        assert_eq!(box_lines(&two), box_lines(&one), "and reached the second as the same line");
        assert!(
            shared_lines(&one).is_empty() && shared_lines(&two).is_empty(),
            "neither reader should still be reading the block that was replaced",
        );
    }
