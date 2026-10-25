// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of what the generator writes for a thin extrude: its bindings, its slots and its text.

    use super::*;
    use fieldform_field::WallPart;
    use fieldform_scene::{
        BooleanOp, Curve, CurveId, Document, Outline, ParamRef, Sweep, WallEnds, WallSide,
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

    fn extrude(outline: Outline) -> Node {
        Node::Extrude { rims: Default::default(),
            outline,
            sweep: Sweep::Normal,
            height: lit(20.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            draft: lit(0.0),
            corners: Default::default(),
        }
    }

    fn wall(curve: CurveId, thickness: f64, side: WallSide, ends: WallEnds) -> Outline {
        Outline::Wall { curve, thickness: lit(thickness), side, ends }
    }

    const HOOK: [(f64, f64); 4] = [(0.0, 0.0), (40.0, 0.0), (50.0, 25.0), (20.0, 20.0)];
    const SQUARE: [(f64, f64); 4] = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)];

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

    fn value(r: &ParamRef) -> f64 {
        match r {
            ParamRef::Literal(v) => *v,
            other => panic!("{other:?} is not a literal"),
        }
    }

    #[test]
    fn a_thin_extrude_binds_its_walls_three_polylines_and_reads_nineteen_slots() {
        let mut doc = Document::new();
        let curve = drawn(&mut doc, &HOOK, false);
        let mut graph = Graph::empty();
        let id = graph.add(extrude(wall(curve, 3.0, WallSide::Left, WallEnds::Round)));
        let program = generate(&doc, &graph).expect("generates, and validates");

        // In the order the call passes them, which is the order the shader's signature has.
        assert_eq!(
            program.curves,
            [
                CurveUse::Wall(id, WallPart::Chain),
                CurveUse::Wall(id, WallPart::Faces),
                CurveUse::Wall(id, WallPart::Sides),
            ]
        );
        let body = program.field_function();
        assert!(body.contains("wall_extrude_distance(0u, 1u, 2u, "), "{body}");
        assert!(body.contains("wall_extrude_normal_bake(0u, 1u, 2u, "), "{body}");
        assert!(!body.contains("extrude_distance(0u, to_local"), "it is not also a thick one");

        // A thick extrude's fifteen, then the wall's four: the thickness as the document holds
        // it, which way the band lies, whether the ends are round, whether the chain closes.
        assert_eq!(program.params.len(), 19);
        assert_eq!(slots_read(&program), 19, "the two passes must count the same slots");
        let told: Vec<f64> = program.params[15..].iter().map(value).collect();
        assert_eq!(told, [3.0, 1.0, 1.0, 0.0]);
    }

    /// **The contract `generate` rests on, for a wall.** A program is keyed on structure so
    /// that changing a number is a buffer write. Everything about a wall but the fact of it is
    /// a number: how thick, which side, how it ends — and whether its chain closes, which is
    /// not even an edit to the block. Dragging two ends together must not rebuild a shader.
    #[test]
    fn nothing_about_a_wall_but_that_it_is_one_reaches_the_shaders_text() {
        let source_and_numbers = |points: &[(f64, f64)], closed, thickness, side, ends| {
            let mut doc = Document::new();
            let curve = drawn(&mut doc, points, closed);
            let mut graph = Graph::empty();
            graph.add(extrude(wall(curve, thickness, side, ends)));
            let program = generate(&doc, &graph).expect("generates");
            let numbers: Vec<f64> = program.params[15..].iter().map(value).collect();
            // The node's own id is random and is not in the text; the bindings are by index.
            (program.field_function().to_string(), numbers)
        };
        let (base, numbers) =
            source_and_numbers(&HOOK, false, 3.0, WallSide::Left, WallEnds::Square);
        assert_eq!(numbers, [3.0, 1.0, 0.0, 0.0]);
        for (what, points, closed, thickness, side, ends, expected) in [
            ("thicker", &HOOK, false, 7.5, WallSide::Left, WallEnds::Square, [7.5, 1.0, 0.0, 0.0]),
            ("to the right", &HOOK, false, 3.0, WallSide::Right, WallEnds::Square, [3.0, -1.0, 0.0, 0.0]),
            ("about the curve", &HOOK, false, 3.0, WallSide::Centre, WallEnds::Square, [3.0, 0.0, 0.0, 0.0]),
            ("round-ended", &HOOK, false, 3.0, WallSide::Left, WallEnds::Round, [3.0, 1.0, 1.0, 0.0]),
            // Counter-clockwise, so left is inside, which is the negative side of a loop.
            ("closed", &SQUARE, true, 3.0, WallSide::Left, WallEnds::Square, [3.0, -1.0, 0.0, 1.0]),
        ] {
            let (source, numbers) = source_and_numbers(points, closed, thickness, side, ends);
            assert_eq!(source, base, "a wall {what} changed the shader's text");
            assert_eq!(numbers, expected, "a wall {what}");
        }
    }

    /// Which way a loop's band lies is not on the node: "left" is inside a loop drawn one way
    /// round and outside the same loop drawn the other. The shader is told the answer, worked
    /// out where the wall is built, as the side of the loop's own distance the band is on.
    #[test]
    fn a_wall_about_a_loop_is_told_its_side_from_the_way_the_loop_winds() {
        let backwards: Vec<(f64, f64)> = SQUARE.iter().rev().copied().collect();
        for (points, side, facing) in [
            (SQUARE.to_vec(), WallSide::Left, -1.0),
            (SQUARE.to_vec(), WallSide::Right, 1.0),
            (backwards.clone(), WallSide::Left, 1.0),
            (backwards, WallSide::Right, -1.0),
            (SQUARE.to_vec(), WallSide::Centre, 0.0),
        ] {
            let mut doc = Document::new();
            let curve = drawn(&mut doc, &points, true);
            let mut graph = Graph::empty();
            graph.add(extrude(wall(curve, 2.0, side, WallEnds::Square)));
            let program = generate(&doc, &graph).expect("generates");
            assert_eq!(value(&program.params[16]), facing, "{side:?}");
            assert_eq!(value(&program.params[18]), 1.0, "a loop is closed");
        }
    }

    /// A union of extrudes in the order given, thin where the flag says so, each on a square
    /// of its own: the program, and the extrudes' ids in list order.
    fn union_of(kinds: &[bool]) -> (ShaderProgram, Vec<NodeId>) {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let ids: Vec<NodeId> = kinds
            .iter()
            .enumerate()
            .map(|(i, thin)| {
                let shift = 60.0 * i as f64;
                let moved: Vec<(f64, f64)> = SQUARE.iter().map(|&(x, y)| (x + shift, y)).collect();
                let curve = drawn(&mut doc, &moved, true);
                graph.add(extrude(if *thin {
                    wall(curve, 2.0, WallSide::Centre, WallEnds::Square)
                } else {
                    Outline::closed(curve)
                }))
            })
            .collect();
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.clone().into_iter().map(Into::into).collect() });
        let program = generate(&doc, &graph).expect("generates, and validates");
        assert_eq!(slots_read(&program), program.params.len(), "{kinds:?}");
        (program, ids)
    }

    /// The distance pass alone. A union's normal is the nearest operand's, so the normal pass
    /// writes every operand's value a second time to find out which that is, and a count over
    /// the whole text comes out doubled — the first version of these tests read two walls
    /// where the graph has one.
    fn distance_pass(program: &ShaderProgram) -> &str {
        let generated = program.field_function();
        let end = generated
            .find("fn body_normal_")
            .or_else(|| generated.find("fn field_normal("))
            .unwrap_or(generated.len());
        &generated[..end]
    }

    /// How a pass reads its extrudes: thick ones written out, thick ones walked, thin ones
    /// written out, thin ones walked. Counted apart with care, because each thin call's name
    /// ends in the thick one's.
    fn calls(pass: &str) -> [usize; 4] {
        let count = |name: &str| pass.matches(name).count();
        let (thin, thin_run) = (count("wall_extrude_distance("), count("wall_extrude_run_distance("));
        [
            count("extrude_distance(") - thin,
            count("extrude_run_distance(") - thin_run,
            thin,
            thin_run,
        ]
    }

    /// **A fan of thin extrudes is one walk, as a fan of thick ones is** — and from two, not
    /// four, because two walls written out are seven seconds of the driver's time
    /// (`WALL_RUN_FLOOR`). The bindings are three a member and consecutive, the slots nineteen a
    /// member, and the normal pass has to count exactly what the value pass pushed.
    #[test]
    fn a_fan_of_thin_extrudes_is_one_walk() {
        for n in [2usize, 3, 8, 16] {
            let (program, ids) = union_of(&vec![true; n]);
            assert_eq!(calls(distance_pass(&program)), [0, 0, 0, 1], "{n} thin extrudes");
            // The normal pass walks the run once as well, and asks the winner alone for its
            // normal: a wall's six differenced copies, written once however long the fan is.
            let whole = program.field_function();
            assert_eq!(whole.matches("wall_extrude_run_nearest(").count(), 1, "{n}");
            assert_eq!(whole.matches("wall_extrude_normal(").count(), 0, "{n}");

            // Member `k` at `first + 3k` and the two after it, in list order.
            let expected: Vec<CurveUse> = ids
                .iter()
                .flat_map(|id| WALL_PARTS.map(|part| CurveUse::Wall(*id, part)))
                .collect();
            assert_eq!(program.curves, expected, "{n}");
            // Each member's own nineteen, which stay though its line is dropped; the run's
            // nineteen a member; and the count.
            assert_eq!(program.params.len(), 38 * n + 1, "{n}");
            assert_eq!(value(&program.params[38 * n]), n as f64);
        }
        // One wall is not a run, whatever the floor.
        let (alone, _) = union_of(&[true, false]);
        assert_eq!(calls(distance_pass(&alone)), [1, 0, 1, 0]);
    }

    /// The loop reads member `k`'s wall at `slot + 19k + 15`, so what is there has to be that
    /// member's and not its neighbour's — which a fan of identical blades would never show.
    #[test]
    fn each_member_of_a_thin_run_carries_its_own_wall() {
        let walls = [
            (2.0, WallSide::Centre, WallEnds::Square, [2.0, 0.0, 0.0, 0.0]),
            (3.5, WallSide::Left, WallEnds::Round, [3.5, 1.0, 1.0, 0.0]),
            (5.0, WallSide::Right, WallEnds::Square, [5.0, -1.0, 0.0, 0.0]),
        ];
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let inputs: Vec<fieldform_scene::NodeId> = walls
            .iter()
            .enumerate()
            .map(|(i, &(thickness, side, ends, _))| {
                let moved: Vec<(f64, f64)> =
                    HOOK.iter().map(|&(x, y)| (x + 80.0 * i as f64, y)).collect();
                let curve = drawn(&mut doc, &moved, false);
                graph.add(extrude(wall(curve, thickness, side, ends)))
            })
            .collect();
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
        let program = generate(&doc, &graph).expect("generates");

        // After the three members' own slots: the run's, nineteen a member, then the count.
        let base = 19 * walls.len();
        for (k, (_, side, _, expected)) in walls.iter().enumerate() {
            let at = base + 19 * k + 15;
            let told: Vec<f64> = program.params[at..at + 4].iter().map(value).collect();
            assert_eq!(told, *expected, "member {k}, to the {side:?}");
        }
        assert_eq!(value(&program.params[base + 19 * walls.len()]), 3.0);
    }

    /// **A run is of one kind.** A loop has one body, and a thick extrude and a thin one are
    /// two calls — one binding and fifteen slots against three and nineteen — so each ends the
    /// other's stretch, the way a pipe beside an extrude does. A union takes its longest run
    /// and the earlier of two the same length, and what is left is written out.
    #[test]
    fn a_run_of_extrudes_is_all_thick_or_all_thin() {
        let (thick, thin) = (false, true);
        for (kinds, expected) in [
            // Four thick in a row are a run, and the thin one after them is not in it.
            (vec![thick, thick, thick, thick, thin], [0, 1, 1, 0]),
            // Split two and two by a thin one, no stretch is long enough to be a run.
            (vec![thick, thick, thin, thick, thick], [4, 0, 1, 0]),
            // Two thin are a run where two thick are not.
            (vec![thin, thin, thick, thick], [2, 0, 0, 1]),
            // The longer run wins, whichever kind it is.
            (vec![thin, thin, thin, thick, thick, thick, thick], [0, 1, 3, 0]),
            (vec![thin, thin, thin, thin, thin, thick, thick, thick, thick], [4, 0, 0, 1]),
            // And the earlier wins a tie.
            (vec![thick, thick, thick, thick, thin, thin, thin, thin], [0, 1, 4, 0]),
        ] {
            let (program, _) = union_of(&kinds);
            assert_eq!(calls(distance_pass(&program)), expected, "{kinds:?}");
        }
    }

    /// The switch a measurement uses to ask for calls has to give it calls, for walls as for
    /// the rest — or the table that set `WALL_RUN_FLOOR` could not have been made.
    #[test]
    fn a_floor_above_the_usual_one_writes_thin_extrudes_out() {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let inputs: Vec<fieldform_scene::NodeId> = (0..4)
            .map(|i| {
                let moved: Vec<(f64, f64)> =
                    HOOK.iter().map(|&(x, y)| (x + 80.0 * i as f64, y)).collect();
                let curve = drawn(&mut doc, &moved, false);
                graph.add(extrude(wall(curve, 2.0, WallSide::Centre, WallEnds::Square)))
            })
            .collect();
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
        let as_calls = generate_with_run_floor(&doc, &graph, usize::MAX).expect("generates");
        assert_eq!(calls(distance_pass(&as_calls)), [0, 0, 4, 0]);
        assert_eq!(as_calls.params.len(), 19 * 4);
        let as_a_loop = generate(&doc, &graph).expect("generates");
        assert_eq!(calls(distance_pass(&as_a_loop)), [0, 0, 0, 1]);
    }

    /// The other half of the byte-compare: an extrude of a closed profile asks for nothing a
    /// wall asks for. Not its functions, not its bindings, not its slots.
    #[test]
    fn a_closed_outline_generates_nothing_of_a_wall() {
        let mut doc = Document::new();
        let curve = drawn(&mut doc, &SQUARE, true);
        let mut graph = Graph::empty();
        let id = graph.add(extrude(Outline::closed(curve)));
        let program = generate(&doc, &graph).expect("generates");
        assert!(!program.field_function().contains("wall_"), "{}", program.field_function());
        assert_eq!(program.curves, [CurveUse::Profile(id)]);
        assert_eq!(program.params.len(), 15);
    }
