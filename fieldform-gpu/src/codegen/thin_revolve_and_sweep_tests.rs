// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of what the generator writes for a thin revolve and an open-section sweep.

    use super::*;
    use fieldform_field::WallPart;
    use fieldform_scene::{
        Curve, CurveId, Document, Frame, Outline, ParamRef, Section, WallEnds, WallSide,
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

    /// A chain in the half-plane y = 0, clear of the axis, with a sharp corner in it.
    const SILHOUETTE: [(f64, f64, f64); 4] =
        [(20.0, 0.0, -10.0), (34.0, 0.0, -4.0), (30.0, 0.0, 12.0), (22.0, 0.0, 4.0)];
    /// A section in the plane x = 0, and a path that leaves it and bends.
    const SECTION: [(f64, f64, f64); 4] =
        [(0.0, -6.0, -4.0), (0.0, 4.0, -5.0), (0.0, 7.0, 3.0), (0.0, 1.0, 1.0)];
    const PATH: [(f64, f64, f64); 4] =
        [(0.0, 0.0, 0.0), (30.0, 4.0, 0.0), (58.0, 16.0, 9.0), (80.0, 36.0, 24.0)];

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

    fn numbers(program: &ShaderProgram, from: usize) -> Vec<f64> {
        program.params[from..]
            .iter()
            .map(|r| match r {
                ParamRef::Literal(v) => *v,
                other => panic!("{other:?} is not a literal"),
            })
            .collect()
    }

    fn revolve(silhouette: &[(f64, f64, f64)], outline: impl Fn(CurveId) -> Outline) -> (ShaderProgram, NodeId) {
        let mut doc = Document::new();
        let curve = drawn(&mut doc, silhouette, false);
        let mut graph = Graph::empty();
        let id = graph.add(Node::Revolve { rims: Default::default(), outline: outline(curve), frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
        (generate(&doc, &graph).expect("generates, and validates"), id)
    }

    fn sweep(outline: impl Fn(CurveId) -> Outline) -> (ShaderProgram, NodeId, CurveId) {
        let mut doc = Document::new();
        let section = drawn(&mut doc, &SECTION, false);
        let path = drawn(&mut doc, &PATH, false);
        let mut graph = Graph::empty();
        let id = graph.add(Node::Sweep { rims: Default::default(),
            outline: outline(section),
            spine: path,
            section: Section::Square,
            law: Default::default(),
        });
        (generate(&doc, &graph).expect("generates, and validates"), id, path)
    }

    #[test]
    fn a_thin_revolve_binds_its_walls_three_polylines_and_reads_eleven_slots() {
        let (program, id) =
            revolve(&SILHOUETTE, |c| wall(c, 3.0, WallSide::Left, WallEnds::Round));
        assert_eq!(
            program.curves,
            [
                CurveUse::Wall(id, WallPart::Chain),
                CurveUse::Wall(id, WallPart::Faces),
                CurveUse::Wall(id, WallPart::Sides),
            ]
        );
        let body = program.field_function();
        assert!(body.contains("wall_revolve_distance(0u, 1u, 2u, "), "{body}");
        assert!(body.contains("wall_revolve_normal(0u, 1u, 2u, "), "{body}");
        // A frame's six, the wall's four, and whether its band crosses the axis: it does not.
        assert_eq!(program.params.len(), 11);
        assert_eq!(slots_read(&program), 11, "the two passes must count the same slots");
        assert_eq!(numbers(&program, 6), [3.0, 1.0, 1.0, 0.0, 0.0]);
    }

    /// **Whether a band crosses the axis is a number, because a thickness is.** A wall a
    /// millimetre from the axis crosses it at three millimetres toward it and not at half of
    /// one, and dragging between the two has to refill a slot rather than rebuild a shader —
    /// so the second reading is a loop the slot bounds, and the text is one text.
    #[test]
    fn a_band_that_crosses_the_axis_is_told_in_a_slot_and_not_in_the_text() {
        let near = [(1.0, 0.0, 0.0), (1.0, 0.0, 30.0)];
        let (keeps_clear, _) = revolve(&near, |c| wall(c, 0.5, WallSide::Left, WallEnds::Square));
        let (crosses, _) = revolve(&near, |c| wall(c, 3.0, WallSide::Left, WallEnds::Square));
        assert_eq!(numbers(&keeps_clear, 10), [0.0]);
        assert_eq!(numbers(&crosses, 10), [1.0]);
        assert_eq!(keeps_clear.field_function(), crosses.field_function());
    }

    #[test]
    fn an_open_section_sweep_binds_its_wall_then_its_spine_and_frame() {
        let (program, id, path) = sweep(|c| wall(c, 2.0, WallSide::Right, WallEnds::Square));
        assert_eq!(
            program.curves,
            [
                CurveUse::Wall(id, WallPart::Chain),
                CurveUse::Wall(id, WallPart::Faces),
                CurveUse::Wall(id, WallPart::Sides),
                CurveUse::Spine(path),
                CurveUse::Frame(id),
            ],
            "the section where a profile's was, then the spine and its frame"
        );
        let body = program.field_function();
        // Spine, frame, then the wall's three, which is the order the shader's signature has.
        assert!(body.contains("wall_sweep_distance(3u, 4u, 0u, 1u, 2u, "), "{body}");
        assert!(body.contains("wall_sweep_value_normal(3u, 4u, 0u, 1u, 2u, "), "{body}");
        // A sweep's fourteen, then the wall's four.
        assert_eq!(program.params.len(), 18);
        assert_eq!(slots_read(&program), 18, "the two passes must count the same slots");
        assert_eq!(numbers(&program, 14), [2.0, -1.0, 0.0, 0.0]);
    }

    /// The contract `generate` rests on, for these two as for a thin extrude: everything about
    /// a wall but the fact of it is a number, and a closed profile asks for nothing of one.
    #[test]
    fn a_walls_numbers_stay_out_of_a_revolves_and_a_sweeps_text() {
        let kinds = [
            (3.0, WallSide::Left, WallEnds::Square),
            (7.5, WallSide::Right, WallEnds::Round),
            (3.0, WallSide::Centre, WallEnds::Round),
        ];
        let texts = |make: &dyn Fn(f64, WallSide, WallEnds) -> ShaderProgram| -> Vec<String> {
            kinds.iter().map(|&(t, side, ends)| make(t, side, ends).field_function().to_string()).collect()
        };
        let revolved = texts(&|t, side, ends| revolve(&SILHOUETTE, |c| wall(c, t, side, ends)).0);
        let swept = texts(&|t, side, ends| sweep(|c| wall(c, t, side, ends)).0);
        for (what, of) in [("revolve", &revolved), ("sweep", &swept)] {
            assert!(of.iter().all(|text| *text == of[0]), "a thin {what}'s text changed with a number");
        }

        let (closed_revolve, id) = {
            let mut doc = Document::new();
            let ring = [(20.0, 0.0, -5.0), (30.0, 0.0, -5.0), (30.0, 0.0, 5.0), (20.0, 0.0, 5.0)];
            let curve = drawn(&mut doc, &ring, true);
            let mut graph = Graph::empty();
            let id = graph.add(Node::Revolve { rims: Default::default(), outline: Outline::closed(curve), frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
            (generate(&doc, &graph).expect("generates"), id)
        };
        assert!(!closed_revolve.field_function().contains("wall_"));
        assert_eq!(closed_revolve.curves, [CurveUse::Profile(id)]);
        assert_eq!(closed_revolve.params.len(), 6);
    }
