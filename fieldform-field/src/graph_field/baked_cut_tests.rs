// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a baked section cut out of a block, and the room it claims in the slot.
    use super::*;
    use crate::Normalization;
    use fieldform_scene::{
        BooleanOp, Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Sweep, WallEnds,
        WallSide,
    };
    use nalgebra::Point3;

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// A block with a section cut clean through it: a square profile sixty across, or a wall
    /// twenty thick about the same square. Compiled at `tolerance_mm`, which is what decides
    /// whether the section answers from a grid.
    fn cut(thin: bool, tolerance_mm: f64) -> (GraphField, bool) {
        let mut doc = Document::new();
        let id = CurveId::new();
        let corners = [(-30.0, -30.0), (30.0, -30.0), (30.0, 30.0), (-30.0, 30.0)];
        let points = corners.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect();
        doc.curves.insert(id, Curve::polyline(id, "square", points, true).expect("closed"));
        let mut graph = Graph::empty();
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(110.0), lit(110.0), lit(40.0)],
            turn: fieldform_scene::no_turn(),
        });
        let section = graph.add(Node::Extrude { rims: Default::default(),
            outline: if thin {
                Outline::Wall {
                    curve: id,
                    thickness: lit(20.0),
                    side: WallSide::Centre,
                    ends: WallEnds::Square,
                }
            } else {
                Outline::closed(id)
            },
            sweep: Sweep::Normal,
            height: lit(100.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::Symmetric,
            draft: lit(0.0),
            corners: Default::default(),
        });
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), section.into()] });
        let field = GraphField::compile_graph(&doc, &graph, tolerance_mm).expect("compiles");
        let baked = field.normalization() == Normalization::LowerBound;
        (field, baked)
    }

    /// Points through the block's middle, a good many of them in the slot.
    fn probes() -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for i in 0..27 {
            for j in 0..27 {
                for z in [-12.0, 0.0, 9.0] {
                    out.push(Point3::new(-50.3 + 3.87 * i as f64, -50.9 + 3.91 * j as f64, z));
                }
            }
        }
        out
    }

    #[test]
    fn a_baked_section_cut_out_of_a_block_claims_no_more_room_in_the_slot_than_there_is() {
        for thin in [false, true] {
            let what = if thin { "a wall" } else { "a profile" };
            // The control: walked, the section is exact and so is its complement.
            let (walked, on_a_grid) = cut(thin, 1e-4);
            assert!(!on_a_grid, "{what} baked at a tolerance meant to be too fine for one");
            let over = crate::overstatement(&walked, &probes(), &crate::probe_directions(), 0.05, 1.2);
            assert!(over < 0.06, "{what}, walked, claimed {over} mm more room than there was");

            // Coarse enough to bake, and coarse enough that the margins are millimetres.
            let (baked, on_a_grid) = cut(thin, 2.0);
            assert!(on_a_grid, "{what} was not baked at a tolerance meant to be coarse enough");
            let over = crate::overstatement(&baked, &probes(), &crate::probe_directions(), 0.05, 1.2);
            assert!(over < 0.06, "{what}, baked and cut out, claimed {over} mm more room than there was");
        }
    }
