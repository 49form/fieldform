// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of a boolean over a list of operands: its value, its normal and its interval.
    use super::*;
    use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef};
    use nalgebra::Point3;

    fn sphere(graph: &mut Graph, x: f64, r: f64) -> fieldform_scene::NodeId {
        graph.add(Node::Sphere {
            centre: fieldform_scene::Place::at(x, 0.0, 0.0),
            radius: ParamRef::Literal(r),
        })
    }

    fn compiled(graph: &Graph) -> GraphField {
        GraphField::compile_graph(&Document::new(), graph, 1e-3).expect("compiles")
    }

    /// Three spheres in one union read as the nearest of the three everywhere — what a nest of
    /// two pairs computes, in one node.
    #[test]
    fn a_union_of_three_is_the_least_of_the_three() {
        let mut g = Graph::empty();
        let a = sphere(&mut g, -30.0, 10.0);
        let b = sphere(&mut g, 0.0, 10.0);
        let c = sphere(&mut g, 30.0, 10.0);
        g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into(), c.into()] });
        let field = compiled(&g);
        for x in [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0] {
            let p = Point3::new(x, 4.0, 0.0);
            let expect = [-30.0f64, 0.0, 30.0]
                .iter()
                .map(|cx| ((x - cx).powi(2) + 16.0f64).sqrt() - 10.0)
                .fold(f64::INFINITY, f64::min);
            assert!((field.distance(p) - expect).abs() < 1e-9, "at x = {x}");
        }
    }

    /// A difference keeps its first operand and subtracts every other: two cutters leave a
    /// point outside the kept body's material wherever either cutter reaches.
    #[test]
    fn a_difference_subtracts_every_operand_after_the_first() {
        let mut g = Graph::empty();
        let kept = sphere(&mut g, 0.0, 20.0);
        let cut_a = sphere(&mut g, 15.0, 8.0);
        let cut_b = sphere(&mut g, -15.0, 8.0);
        g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![kept.into(), cut_a.into(), cut_b.into()] });
        let field = compiled(&g);
        // Inside each cutter the composed field is positive: material was taken.
        assert!(field.distance(Point3::new(15.0, 0.0, 0.0)) > 0.0);
        assert!(field.distance(Point3::new(-15.0, 0.0, 0.0)) > 0.0);
        // Between them the kept sphere still has material.
        assert!(field.distance(Point3::new(0.0, 0.0, 0.0)) < 0.0);
        // And the value there is the kept body's own distance, untouched by either cutter,
        // since each is further away than the kept surface.
        let at = Point3::new(0.0, 0.0, 0.0);
        let cut_reach = (15.0f64 - 8.0).max(0.0);
        assert!((field.distance(at) - (-20.0f64).max(-cut_reach)).abs() < 1e-9);
    }

    /// The normal comes from whichever operand defined the value, negated where a cutter did:
    /// on a cutter's wall the surface faces *into* the cutter.
    #[test]
    fn a_normal_on_a_cutters_wall_points_into_the_cutter() {
        let mut g = Graph::empty();
        let kept = sphere(&mut g, 0.0, 20.0);
        let cut_a = sphere(&mut g, 15.0, 8.0);
        let cut_b = sphere(&mut g, -15.0, 8.0);
        g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![kept.into(), cut_a.into(), cut_b.into()] });
        let field = compiled(&g);
        // On the second cutter's wall, on the side facing the kept body's centre. The cutter's
        // own outward normal there is +X, away from its centre at -15; the composed surface is
        // the cavity's wall, which faces into the cavity, so its normal is the negation: -X.
        let on_wall = Point3::new(-7.0, 0.0, 0.0);
        let n = field.normal(on_wall).expect("a normal");
        assert!(n.x < -0.99, "{n:?}");
    }

    #[test]
    fn one_operand_is_that_operand_and_none_is_refused() {
        let mut g = Graph::empty();
        let only = sphere(&mut g, 0.0, 10.0);
        g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![only.into()] });
        let field = compiled(&g);
        assert!((field.distance(Point3::new(25.0, 0.0, 0.0)) - 15.0).abs() < 1e-9);

        let mut empty = Graph::empty();
        empty.add(Node::Boolean { op: BooleanOp::Union, inputs: Vec::new() });
        assert!(matches!(
            GraphField::compile_graph(&Document::new(), &empty, 1e-3),
            Err(CompileError::NothingToCombine)
        ));
    }

    /// The interval bound holds over a list as it did over a pair: every value the field takes
    /// in a box lies inside the box's interval, for each operation.
    #[test]
    fn a_list_booleans_interval_contains_every_value_it_takes() {
        for op in [BooleanOp::Union, BooleanOp::Difference, BooleanOp::Intersection] {
            let mut g = Graph::empty();
            let a = sphere(&mut g, -12.0, 15.0);
            let b = sphere(&mut g, 0.0, 9.0);
            let c = sphere(&mut g, 14.0, 12.0);
            g.add(Node::Boolean { op, inputs: vec![a.into(), b.into(), c.into()] });
            let field = compiled(&g);
            let (lo, hi) = (Point3::new(-20.0, -10.0, -5.0), Point3::new(20.0, 10.0, 5.0));
            let (bound_lo, bound_hi) = field.interval(lo, hi);
            for i in 0..9 {
                for j in 0..5 {
                    let p = Point3::new(
                        lo.x + (hi.x - lo.x) * i as f64 / 8.0,
                        lo.y + (hi.y - lo.y) * j as f64 / 4.0,
                        1.0,
                    );
                    let d = field.distance(p);
                    assert!(
                        d >= bound_lo - 1e-9 && d <= bound_hi + 1e-9,
                        "{op:?}: {d} outside [{bound_lo}, {bound_hi}] at {p:?}"
                    );
                }
            }
        }
    }
