// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a drafted extrude held to the room its section has, through the whole graph.
    use super::*;
    use fieldform_scene::{Corners, Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Rims, Sides, Sweep};
    use nalgebra::Point3;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// An L drawn on the ground: 30 by 30, its arms 6 mm thick.
    fn l_shape(doc: &mut Document) -> CurveId {
        let id = CurveId::new();
        let corners = [(0.0, 0.0), (30.0, 0.0), (30.0, 6.0), (6.0, 6.0), (6.0, 30.0), (0.0, 30.0)];
        let points = corners.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect();
        doc.curves.insert(id, Curve::polyline(id, "L", points, true).expect("closed"));
        id
    }

    /// The L extruded 20 mm up with this draft, these sides and these corners.
    fn drafted(draft: f64, sides: Sides, corners: Corners, sweep: Sweep) -> Result<GraphField, CompileError> {
        let mut doc = Document::new();
        let id = l_shape(&mut doc);
        let mut graph = Graph::empty();
        graph.add(Node::Extrude {
            outline: Outline::closed(id),
            sweep,
            start: lit(0.0),
            height: lit(20.0),
            draft: lit(draft),
            sides,
            corners,
            rims: Rims::hard(),
        });
        GraphField::compile_graph(&doc, &graph, 1e-4)
    }

    /// How far the L's walls may move in before its section is gone: for a mitre, the 3 mm at
    /// which both arms close and the outline crosses itself; for a rounding, the elbow's
    /// inscribed circle, which touches both outer walls and the reflex corner and so has the
    /// radius `6·(2 − √2)` = 3.51 mm -- an offset keeps that much after the arms have gone.
    fn room(corners: Corners) -> f64 {
        match corners {
            Corners::Sharp => 3.0,
            Corners::Round => 6.0 * (2.0 - std::f64::consts::SQRT_2),
        }
    }

    /// **Refused just past the room and built just before it**, round or sharp, upright or
    /// leaning. Over 20 mm a sharp draft may lean 8.53° in and a round one 9.98°; a hundredth of
    /// a degree either side is the whole difference.
    #[test]
    fn a_drafted_extrude_is_refused_just_past_where_its_section_closes_and_built_just_before() {
        let leaning = Sweep::Toward { xyz: [lit(0.3), lit(0.0), lit(1.0)] };
        for corners in Corners::ALL {
            let closing = (room(corners) / 20.0).atan().to_degrees();
            for sweep in [Sweep::Normal, leaning.clone()] {
                let before = drafted(-(closing - 0.01), Sides::One, corners, sweep.clone());
                assert!(before.is_ok(), "{corners:?} {sweep:?} refused a draft that fits: {:?}", before.err());
                match drafted(-(closing + 0.01), Sides::One, corners, sweep.clone()) {
                    Err(CompileError::DraftCrossesTheSection { room: said, .. }) => assert!(
                        (said - room(corners)).abs() < 1e-6,
                        "{corners:?}: the L has {} mm, said {said}",
                        room(corners)
                    ),
                    other => panic!("{corners:?} {sweep:?} built a draft past its room: {:?}", other.err()),
                }
            }
        }
    }

    /// **A sharp draft whose outline crosses itself is refused by name.** Drafted in 12° over
    /// 20 mm, the L's walls move 4.25 mm, past the 3 mm at which its arms close and the mitred
    /// outline crosses itself -- 14.11 mm up. This used to build, and the walk read whatever the
    /// crossed outline's parity said; the refusal says how far the walls may go, where, and the
    /// draft that fits.
    #[test]
    fn a_sharp_draft_whose_outline_crosses_itself_is_refused_naming_the_draft_that_fits() {
        let err = drafted(-12.0, Sides::One, Corners::Sharp, Sweep::Normal)
            .err()
            .expect("the mitred outline crosses itself before the top");
        let text = err.to_string();
        assert!(text.contains("cross itself"), "{text}");
        for number in ["12.0\u{b0} inward", "3.00 mm", "14.11 mm", "20.00 mm", "8.5\u{b0}"] {
            assert!(text.contains(number), "{number:?} is missing from: {text}");
        }
    }

    /// **Each side of a two-sided extrude is held to its own draft**, and a symmetric one to
    /// half its depth -- the draft opens from the start plane, so each half moves the walls by
    /// its own length.
    #[test]
    fn each_side_and_each_half_is_held_to_its_own_length() {
        // Side one fits easily; side two's 5 mm at 35° moves the walls 3.5 mm, past 3.
        let two = Sides::Two { height: lit(5.0), draft: lit(-35.0) };
        assert!(matches!(
            drafted(-1.0, two, Corners::Sharp, Sweep::Normal),
            Err(CompileError::DraftCrossesTheSection { length, .. }) if (length - 5.0).abs() < 1e-12
        ));
        // Half of 20 at 15° is 2.68 mm, which fits; all of it would not.
        assert!(drafted(-15.0, Sides::Symmetric, Corners::Sharp, Sweep::Normal).is_ok());
        assert!(drafted(-15.0, Sides::One, Corners::Sharp, Sweep::Normal).is_err());
    }

    /// **A widening draft on the L is never refused**: its one reflex corner moves into its own
    /// notch, which no edge faces, so the mitre never meets itself; and rounding only grows.
    #[test]
    fn a_widening_draft_on_an_l_is_built_either_way() {
        for corners in Corners::ALL {
            assert!(drafted(40.0, Sides::One, corners, Sweep::Normal).is_ok(), "{corners:?}");
        }
    }
