// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of what the generator writes for a profile solid's rims: the text a hard rim leaves
//! alone, the call and the slot a rounded one adds, and the two passes counting the same.

    use super::*;
    use fieldform_scene::{
        Continuity, Curve, CurveId, Document, Fillet, Rims, Outline, ParamRef, Sweep, WallEnds, WallSide,
    };
    use nalgebra::Point3;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn drawn(doc: &mut Document, points: &[(f64, f64)], closed: bool) -> CurveId {
        let id = CurveId::new();
        let points = points.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect();
        doc.curves.insert(id, Curve::polyline(id, "curve", points, closed).expect("a polyline"));
        id
    }

    const SQUARE: [(f64, f64); 4] = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)];
    const HOOK: [(f64, f64); 4] = [(0.0, 0.0), (40.0, 0.0), (50.0, 25.0), (20.0, 20.0)];

    fn extrude(outline: Outline, draft: f64, rims: Rims) -> Node {
        Node::Extrude {
            outline,
            sweep: Sweep::Normal,
            height: lit(20.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            draft: lit(draft),
            corners: if draft != 0.0 { fieldform_scene::Corners::Sharp } else { Default::default() },
            rims,
        }
    }

    /// The highest parameter slot the source reads, which has to be the last one declared: the
    /// value pass pushes the slots and the normal pass only counts them.
    fn slots_read(program: &ShaderProgram) -> usize {
        program
            .source
            .split("params[")
            .skip(1)
            .filter_map(|rest| rest.split(']').next())
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .max()
            .expect("the shader reads parameters")
            + 1
    }

    /// **A hard rim is the text it always was, byte for byte; a rounded one is the `_rim_` call
    /// with its reach as one more slot, last, and its shape as a literal** -- for a thick extrude,
    /// a mitred one, a wall and a mitred wall round a loop, the four extrude calls there are. The
    /// spec's own acceptance (`round-chamfer-spec.md` §5) asks for the first; the two passes
    /// counting the same slots is what keeps the reach in the slot it was pushed to.
    #[test]
    fn a_hard_rim_is_the_text_it_always_was_and_a_rounded_one_is_one_slot_more() {
        let mut doc = Document::new();
        let square = drawn(&mut doc, &SQUARE, true);
        let hook = drawn(&mut doc, &HOOK, false);
        let cases: [(&str, Outline, f64, &str); 4] = [
            ("a thick extrude", Outline::closed(square), 0.0, "extrude"),
            ("a mitred extrude", Outline::closed(square), 5.0, "extrude_sharp"),
            (
                "a wall",
                Outline::Wall { curve: hook, thickness: lit(3.0), side: WallSide::Left, ends: WallEnds::Round },
                0.0,
                "wall_extrude",
            ),
            (
                "a mitred wall round a loop",
                Outline::Wall { curve: square, thickness: lit(3.0), side: WallSide::Centre, ends: WallEnds::Round },
                5.0,
                "wall_extrude_sharp",
            ),
        ];
        for (what, outline, draft, name) in cases {
            let build = |rims: Rims| {
                let mut graph = Graph::empty();
                graph.add(extrude(outline.clone(), draft, rims));
                generate(&doc, &graph)
            };
            let hard = build(Rims::hard()).expect(what);
            // A rim of nought names a shape and is still hard.
            let nought = build(Rims { radius: lit(0.0), continuity: Continuity::Chamfer, at: fieldform_scene::RimEnds::Both, second: None }).expect(what);
            assert_eq!(hard.source, nought.source, "{what}: a rim of nought is hard");
            assert!(!hard.field_function().contains("_rim_"), "{what}: a hard rim calls no rim function");
            assert!(hard.field_function().contains(&format!("{name}_distance(")), "{what}");

            for shape in Continuity::ALL {
                let rounded = build(Rims { radius: lit(1.5), continuity: shape, at: fieldform_scene::RimEnds::Both, second: None }).expect(what);
                let body = rounded.field_function();
                assert!(body.contains(&format!("{name}_rim_distance(")), "{what}: {body}");
                assert!(body.contains(&format!("{name}_rim_normal_bake(")), "{what}: the normal too, through its bake copy");
                assert!(!body.contains(&format!("{name}_distance(")), "{what}: not also the plain call");
                assert_eq!(rounded.params.len(), hard.params.len() + 1, "{what}: one slot more");
                assert_eq!(rounded.params.last(), Some(&lit(1.5)), "{what}: the reach, last");
                assert_eq!(slots_read(&rounded), rounded.params.len(), "{what}: both passes count it");
                // The exponent and the ends as literals, then the square-corner scale: one for
                // the upright calls, nought for the drafted.
                let n = format!("{:?}, 0.0, ", shape.exponent() as f32);
                assert!(body.contains(&format!("params[{}], {n}", hard.params.len())), "{what}: {shape:?}'s exponent as a literal");
            }
        }
    }

    /// **A loft's rims are its one call renamed with a slot after its bound**, the hard loft
    /// keeping its text; and the normal pass's one walk reads the same slot.
    #[test]
    fn a_rimmed_loft_is_its_rim_call_with_one_slot_after_its_bound() {
        let mut doc = Document::new();
        let ring = |doc: &mut Document, r: f64, z: f64| {
            let id = CurveId::new();
            let points = (0..24)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / 24.0;
                    Point3::new(r * a.cos(), r * a.sin(), z)
                })
                .collect();
            doc.curves.insert(id, Curve::polyline(id, "ring", points, true).expect("closed"));
            id
        };
        let sections = vec![Outline::closed(ring(&mut doc, 20.0, 0.0)), Outline::closed(ring(&mut doc, 12.0, 40.0))];
        let build = |rims: Rims| {
            let mut graph = Graph::empty();
            graph.add(Node::Loft { sections: sections.clone(), spine: None, smooth: fieldform_scene::LoftSmooth::G1, rims });
            generate(&doc, &graph).expect("a loft generates and validates")
        };
        let hard = build(Rims::hard());
        assert!(!hard.field_function().contains("_rim_"), "a hard loft calls no rim function");
        let rounded = build(Rims { radius: lit(2.0), continuity: Continuity::G2, at: fieldform_scene::RimEnds::Both, second: None });
        let body = rounded.field_function();
        assert!(body.contains("loft_rim_distance("), "{body}");
        assert!(body.contains("loft_rim_value_normal_bake("), "{body}");
        assert_eq!(rounded.params.len(), hard.params.len() + 1);
        assert_eq!(rounded.params.last(), Some(&lit(2.0)));
        assert_eq!(slots_read(&rounded), rounded.params.len(), "both passes count it");
    }

    /// **A sweep's rims turn whichever of its five calls it is into the `_rim_` twin**, with one
    /// slot after everything the sweep pushes, in both passes; a hard sweep keeps its text. The
    /// plain, the law'd, and a wall under a law -- three of the five, the other two differing
    /// from these only in which bindings they read.
    #[test]
    fn a_rimmed_sweep_is_its_call_renamed_with_one_slot_last() {
        let mut doc = Document::new();
        let square = drawn(&mut doc, &SQUARE, true);
        let hook = drawn(&mut doc, &HOOK, false);
        let path = CurveId::new();
        doc.curves.insert(
            path,
            Curve::polyline(path, "path", vec![Point3::new(20.0, 20.0, -5.0), Point3::new(20.0, 30.0, 60.0)], false)
                .expect("a path"),
        );
        let taper = fieldform_scene::SweepLaw {
            taper: fieldform_scene::Taper::Scale(lit(0.7)),
            ..Default::default()
        };
        let cases = [
            ("the plain sweep", Outline::closed(square), fieldform_scene::SweepLaw::default(), "sweep"),
            ("a law'd sweep", Outline::closed(square), taper.clone(), "sweep_law"),
            (
                "a wall under a law",
                Outline::Wall { curve: hook, thickness: lit(3.0), side: WallSide::Left, ends: WallEnds::Round },
                taper,
                "wall_sweep_law",
            ),
        ];
        for (what, outline, law, name) in cases {
            let build = |rims: Rims| {
                let mut graph = Graph::empty();
                graph.add(Node::Sweep { outline: outline.clone(), spine: path, section: Default::default(), law: law.clone(), rims });
                generate(&doc, &graph).expect(what)
            };
            let hard = build(Rims::hard());
            assert!(!hard.field_function().contains("_rim_"), "{what}: a hard sweep calls no rim function");
            let rounded = build(Rims { radius: lit(1.0), continuity: Continuity::Chamfer, at: fieldform_scene::RimEnds::Both, second: None });
            let body = rounded.field_function();
            assert!(body.contains(&format!("{name}_rim_distance(")), "{what}: {body}");
            assert!(body.contains(&format!("{name}_rim_value_normal_bake(")), "{what}: {body}");
            assert_eq!(rounded.params.len(), hard.params.len() + 1, "{what}");
            assert_eq!(rounded.params.last(), Some(&lit(1.0)), "{what}");
            assert_eq!(slots_read(&rounded), rounded.params.len(), "{what}: both passes count it");
        }
    }

    /// **A rimmed extrude ends a union's run** -- the loop body has no slot for a rim -- and the
    /// extrudes either side of it still loop, so a rim on one blade of many costs that blade and
    /// no other.
    #[test]
    fn a_rimmed_extrude_is_written_out_beside_a_run_of_plain_ones() {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let mut inputs = Vec::new();
        for i in 0..(RUN_FLOOR + 2) {
            let at = i as f64 * 60.0;
            let id = drawn(&mut doc, &[(at, 0.0), (at + 40.0, 0.0), (at + 40.0, 40.0), (at, 40.0)], true);
            let rims = if i == RUN_FLOOR + 1 { Rims { radius: lit(2.0), continuity: Continuity::G1, at: fieldform_scene::RimEnds::Both, second: None } } else { Rims::hard() };
            inputs.push(graph.add(extrude(Outline::closed(id), 0.0, rims)).into());
        }
        graph.add(Node::Boolean { op: fieldform_scene::BooleanOp::Union, inputs });
        let program = generate(&doc, &graph).expect("generates");
        let body = program.field_function();
        assert_eq!(body.matches("extrude_run_distance(").count(), 1, "the plain ones still loop: {body}");
        // Once in each pass: the normal pass writes a body's value beside its normal, through
        // the reading's `_bake` copy.
        assert_eq!(body.matches("extrude_rim_distance(").count(), 1, "the rimmed one is its own call");
        assert_eq!(body.matches("extrude_rim_distance_bake(").count(), 1, "and the normal pass reads its bake copy");
    }

    /// **Which ends is a third literal in the rim's vector**, after the exponent -- 0 both, 1 the
    /// start, 2 the end, as `rim_join_ends` reads it -- and adds no slot: the reach is still the
    /// one slot a rim pushes.
    #[test]
    fn a_rims_ends_are_a_literal_in_its_vector_and_cost_no_slot() {
        use fieldform_scene::RimEnds;
        let mut doc = Document::new();
        let square = drawn(&mut doc, &SQUARE, true);
        let mut slots = None;
        for (at, literal) in [(RimEnds::Both, "0.0"), (RimEnds::Start, "1.0"), (RimEnds::End, "2.0")] {
            let mut graph = fieldform_scene::Graph::empty();
            graph.add(extrude(
                Outline::closed(square),
                0.0,
                Rims { radius: lit(2.0), continuity: Continuity::G2, at, second: None },
            ));
            let program = generate(&doc, &graph).expect("a rimmed extrude");
            let body = program.field_function();
            let r = program.params.len() - 1;
            // An upright, undrafted extrude's rims read the exact corner: the fourth number is one.
            assert!(body.contains(&format!("vec4<f32>(params[{r}], 3.0, {literal}, 1.0)")), "{at:?}: {body}");
            assert_eq!(*slots.get_or_insert(program.params.len()), program.params.len(), "{at:?}");
            assert_eq!(slots_read(&program), program.params.len(), "{at:?}: both passes count it");
        }
    }

    /// **A box's or a cylinder's edge is its `_edge_` call with one slot more, in both passes**,
    /// and a hard one -- nought, whatever shape it names -- is the call it always was.
    #[test]
    fn an_edged_box_cylinder_or_cone_is_its_edge_call_with_one_slot_more() {
        let doc = Document::new();
        let cases: [(&str, fn(Fillet) -> Node); 3] = [
            ("cone", |edge| Node::Cone {
                ends: fieldform_scene::Ends::between(
                    fieldform_scene::Place::At([lit(0.0); 3]),
                    fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(10.0)]),
                ),
                base_radius: lit(8.0),
                top_radius: lit(3.0),
                edge,
            }),
            ("cuboid", |edge| Node::Cuboid {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(40.0), lit(24.0), lit(16.0)],
                turn: fieldform_scene::no_turn(),
                edge,
            }),
            ("cylinder", |edge| Node::Cylinder {
                ends: fieldform_scene::Ends::between(
                    fieldform_scene::Place::At([lit(0.0); 3]),
                    fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(10.0)]),
                ),
                radius: lit(8.0),
                edge,
            }),
        ];
        for (name, node) in cases {
            let build = |edge: Fillet| {
                let mut graph = fieldform_scene::Graph::empty();
                graph.add(node(edge));
                generate(&doc, &graph).expect(name)
            };
            let hard = build(Fillet::hard());
            let nought = build(Fillet { radius: lit(0.0), continuity: Continuity::Chamfer, second: None });
            assert_eq!(hard.field_function(), nought.field_function(), "{name}: nought is hard");
            assert!(!hard.field_function().contains("_edge_"), "{name}: a hard edge calls no edge function");
            for (shape, n) in [(Continuity::G1, "2.0"), (Continuity::G2, "3.0"), (Continuity::Chamfer, "1.0")] {
                let cut = build(Fillet { radius: lit(2.0), continuity: shape, second: None });
                let body = cut.field_function();
                let r = hard.params.len();
                assert!(body.contains(&format!("{name}_edge_distance(")), "{name} {shape:?}: {body}");
                assert!(body.contains(&format!("{name}_edge_normal(")), "{name} {shape:?}: the normal too");
                assert!(body.contains(&format!("vec2<f32>(params[{r}], {n})")), "{name} {shape:?}: the reach last, the exponent a literal");
                assert_eq!(cut.params.len(), r + 1, "{name}: one slot more");
                assert_eq!(slots_read(&cut), cut.params.len(), "{name}: both passes count it");
            }
        }
    }

    /// **A partial revolve's rims are its wedge call renamed, with one slot after the wedge's
    /// two, in both passes**, over a profile and over a wall; a full revolve's call is untouched,
    /// since the compile refuses rims on one.
    #[test]
    fn a_rimmed_revolve_is_its_wedge_call_renamed_with_one_slot_last() {
        let mut doc = Document::new();
        let square = drawn(&mut doc, &[(30.0, -5.0), (40.0, -5.0), (40.0, 5.0), (30.0, 5.0)], true);
        let hook = drawn(&mut doc, &[(30.0, -5.0), (40.0, 0.0), (30.0, 5.0)], false);
        let cases = [
            ("a profile", Outline::closed(square), "revolve_wedge"),
            (
                "a wall",
                Outline::Wall { curve: hook, thickness: lit(2.0), side: WallSide::Left, ends: WallEnds::Round },
                "wall_revolve_wedge",
            ),
        ];
        for (what, outline, name) in cases {
            let build = |angle: f64, rims: Rims| {
                let mut graph = fieldform_scene::Graph::empty();
                graph.add(Node::Revolve {
                    outline: outline.clone(),
                    frame: fieldform_scene::Frame::default(),
                    angle: lit(angle),
                    sides: fieldform_scene::AngleSides::One,
                    rims,
                });
                generate(&doc, &graph).expect(what)
            };
            let hard = build(120.0, Rims::hard());
            assert!(!hard.field_function().contains("_rim_"), "{what}: a hard wedge calls no rim function");
            let rounded = build(120.0, Rims { radius: lit(1.0), continuity: Continuity::G2, at: fieldform_scene::RimEnds::Start, second: None });
            let body = rounded.field_function();
            assert!(body.contains(&format!("{name}_rim_distance(")), "{what}: {body}");
            assert!(body.contains(&format!("{name}_rim_normal_bake(")), "{what}: the normal too, through its bake copy");
            // The reach's slot and the square-corner scale's, which the generator's own tests
            // leave at nought, the bound, having no compile to hand one over.
            assert_eq!(rounded.params.len(), hard.params.len() + 2, "{what}: two slots more");
            assert_eq!(rounded.params.last(), Some(&lit(0.0)), "{what}: the scale, last, nought here");
            let r = hard.params.len();
            assert!(body.contains(&format!("params[{r}], 3.0, 1.0, params[{}])", r + 1)), "{what}: exponent and end as literals, the scale a slot");
            assert_eq!(slots_read(&rounded), rounded.params.len(), "{what}: both passes count it");
            let full = build(360.0, Rims { radius: lit(1.0), continuity: Continuity::G1, at: fieldform_scene::RimEnds::Both, second: None });
            assert!(!full.field_function().contains("_rim_"), "{what}: a full revolve keeps its call");
        }
    }
