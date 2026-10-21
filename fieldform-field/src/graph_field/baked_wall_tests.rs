// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a thin extrude baked to a grid, and the shelf the grid is kept on.
    use super::*;
    use crate::Normalization;
    use fieldform_scene::{
        Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Sweep, WallEnds, WallSide,
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

    fn thin(curve: CurveId, thickness: f64, side: WallSide, height: f64) -> Node {
        Node::Extrude { rims: Default::default(),
            outline: Outline::Wall { curve, thickness: lit(thickness), side, ends: WallEnds::Square },
            sweep: Sweep::Normal,
            height: lit(height),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            draft: lit(0.0),
            corners: Default::default(),
        }
    }

    /// The wall's half of `a_baked_profile_reports_a_lower_bound_and_an_exact_one_does_not`,
    /// and both sides of the same switch: answering from a grid it is a bound and says so, and
    /// at a tolerance too fine for a grid it is the distance export is owed. With the contract
    /// itself checked where it is a bound, on a chain with a mitre in it.
    #[test]
    fn a_baked_wall_reports_a_lower_bound_and_an_exact_one_does_not() {
        let mut doc = Document::new();
        let chain = drawn(&mut doc, &[(-25.0, -15.0), (5.0, -15.0), (15.0, 10.0), (-10.0, 0.0)], false);
        let mut graph = Graph::empty();
        graph.add(thin(chain, 5.0, WallSide::Left, 20.0));

        let coarse = GraphField::compile_graph(&doc, &graph, 1.0).expect("compiles");
        assert_eq!(
            coarse.normalization(),
            Normalization::LowerBound,
            "a baked wall has to announce that it reads short",
        );
        let probes: Vec<Point3<f64>> = (0..12)
            .flat_map(|i| (0..12).flat_map(move |j| (0..4).map(move |k| (i, j, k))))
            .map(|(i, j, k)| {
                Point3::new(-38.0 + 6.1 * i as f64, -29.0 + 5.3 * j as f64, -9.0 + 11.7 * k as f64)
            })
            .collect();
        let over = crate::overstatement(&coarse, &probes, &crate::probe_directions(), 0.05, 1.2);
        assert!(over < 1e-6, "the baked wall claimed {over} mm more room than it had");

        let fine = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
        assert_eq!(
            fine.normalization(),
            Normalization::Distance,
            "an unbaked wall is still exact, which is what export gets",
        );
    }

    /// **A wall's grid goes on the shelf a profile's does, under a key of its own.** Kept, so
    /// that a height being dragged does not walk the chain again for every sample; and never a
    /// profile's, though the two are keyed by the same four numbers and here are over the very
    /// same points — the grids hold different fields, and a wall handed its loop's would be the
    /// solid disc and not the ring.
    #[test]
    fn a_walls_grid_is_kept_between_compiles_and_is_not_its_loops() {
        let mut doc = Document::new();
        let square = drawn(&mut doc, &[(-30.0, -30.0), (30.0, -30.0), (30.0, 30.0), (-30.0, 30.0)], true);
        let of = |node: Node| {
            let mut graph = Graph::empty();
            graph.add(node);
            graph
        };
        let thick = of(Node::Extrude { rims: Default::default(),
            outline: Outline::closed(square),
            sweep: Sweep::Normal,
            height: lit(20.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            draft: lit(0.0),
            corners: Default::default(),
        });
        let middle = Point3::new(0.0, 0.0, 10.0);

        let mut cache = LeafCache::default();
        let solid = GraphField::compile_cached(&doc, &thick, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 1, "one profile, one grid");
        assert!(solid.distance(middle) < 0.0, "the middle of a thick extrude is material");

        // The wall about the same loop, with the profile's grid still on the shelf.
        let ring = of(thin(square, 8.0, WallSide::Centre, 20.0));
        let hollow = GraphField::compile_cached(&doc, &ring, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 2, "the wall's grid beside the profile's, not in its place");
        assert!(hollow.distance(middle) > 0.0, "the wall was handed its loop's grid");
        let fresh = GraphField::compile_graph(&doc, &ring, 0.5).expect("compiles");
        for at in [middle, Point3::new(29.0, 3.0, 4.0), Point3::new(41.0, -37.0, 22.0)] {
            assert!(
                (hollow.distance(at) - fresh.distance(at)).abs() < 1e-9,
                "the cache answered differently from a fresh compile at {at:?}",
            );
        }

        // Taller: the same wall, so the same grid, and the profile's is let go of unasked for.
        let taller = of(thin(square, 8.0, WallSide::Centre, 35.0));
        GraphField::compile_cached(&doc, &taller, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 1, "a height being dragged must not rebake the wall");

        // Thicker: a different wall, so a different grid.
        let thicker = of(thin(square, 9.0, WallSide::Centre, 35.0));
        GraphField::compile_cached(&doc, &thicker, 0.5, &mut cache).expect("compiles");
        assert_eq!(cache.profiles.len(), 2, "a thickness is part of what a wall's grid is of");
    }
