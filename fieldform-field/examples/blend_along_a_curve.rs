// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether the metaball a blend makes against a point also works against a *curve*.
//!
//! `blend_with_a_point.rs` shows that blending with a point swells the surface toward it,
//! because `smooth_min` mixes by magnitude inside the radius and a point's field is a true
//! distance. A curve reaches the field the same way, through `Node::Pipe`, whose value is the
//! distance to the curve less the radius — so a pipe of no radius *is* the curve's distance
//! field, and the same argument should hold along its whole length rather than at one place.
//!
//! This measures whether it does: how much material the blend adds, whether the swelling
//! follows the curve, and whether the result is still a distance the tracer may trust.
//!
//! Run with: `cargo run --release -p fieldform-field --example blend_along_a_curve`

use fieldform_field::{approximate_blends, carries_true_distance, GraphField, ScalarField};
use fieldform_scene::{Fillet, BooleanOp, Continuity, Curve, CurveId, Document, Graph, Node, ParamRef, Place};
use nalgebra::Point3;

const RADIUS: f64 = 20.0;
/// How far out along +X the curve runs, so it sits 6 mm clear of the ball.
const OUT: f64 = 26.0;
/// Half the curve's length along Y.
const SPAN: f64 = 15.0;
const HALF: f64 = 60.0;
const STEPS: usize = 120;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn solid_mm3(field: &GraphField) -> f64 {
    let step = 2.0 * HALF / STEPS as f64;
    let cell = step * step * step;
    let mut solid = 0.0;
    for i in 0..STEPS {
        for j in 0..STEPS {
            for k in 0..STEPS {
                let p = Point3::new(
                    -HALF + (i as f64 + 0.5) * step,
                    -HALF + (j as f64 + 0.5) * step,
                    -HALF + (k as f64 + 0.5) * step,
                );
                if field.distance(p) < 0.0 {
                    solid += cell;
                }
            }
        }
    }
    solid
}

/// How far the surface reaches along +X at height `y`, by bisection.
fn reach_at(field: &GraphField, y: f64) -> f64 {
    let (mut inside, mut outside) = (0.0, HALF);
    if field.distance(Point3::new(0.0, y, 0.0)) >= 0.0 {
        return f64::NAN;
    }
    for _ in 0..60 {
        let mid = 0.5 * (inside + outside);
        if field.distance(Point3::new(mid, y, 0.0)) < 0.0 {
            inside = mid;
        } else {
            outside = mid;
        }
    }
    0.5 * (inside + outside)
}

/// A straight curve running along Y, `OUT` millimetres out on X.
fn with_a_curve(doc: &mut Document) -> CurveId {
    let id = CurveId::new();
    let curve = Curve::new(
        id,
        "edge",
        1,
        vec![Point3::new(OUT, -SPAN, 0.0), Point3::new(OUT, SPAN, 0.0)],
        vec![1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        false,
    )
    .expect("a two-point degree-1 curve");
    doc.curves.insert(id, curve);
    id
}

/// The ball, blended with a pipe of `pipe_radius` on that curve at `blend_radius`.
fn ball_blended_with_a_curve(pipe_radius: f64, blend_radius: f64) -> (Document, Graph) {
    let mut doc = Document::new();
    let curve = with_a_curve(&mut doc);
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(RADIUS),
    });
    let pipe = graph.add(Node::Pipe { curve, radius: lit(pipe_radius) });
    graph.add(Node::Blend {
        op: BooleanOp::Union,
        a: ball.into(),
        b: pipe.into(),
        fillet: Fillet { radius: lit(blend_radius), continuity: Continuity::G1 },
    });
    (doc, graph)
}

fn main() {
    println!(
        "A {RADIUS} mm ball at the origin, G1-blended with a curve running along Y at x = {OUT},\n\
         which is {} mm clear of the surface. The curve reaches the field as a pipe, whose value\n\
         is the distance to it less the radius -- so a pipe of no radius is the curve itself.\n",
        OUT - RADIUS
    );

    let doc = Document::new();
    let mut plain = Graph::empty();
    plain.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(RADIUS) });
    let alone = GraphField::compile_graph(&doc, &plain, 1e-3).expect("compiles");
    let base = solid_mm3(&alone);
    println!(
        "  {:>6}  {:>6}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}",
        "pipe r", "blend", "solid mm3", "added", "reach y=0", "y=10", "y=25"
    );
    println!(
        "  {:>6}  {:>6}  {:>10.0}  {:>9}  {:>9.2}  {:>9.2}  {:>9.2}",
        "-",
        "-",
        base,
        "-",
        reach_at(&alone, 0.0),
        reach_at(&alone, 10.0),
        reach_at(&alone, 25.0)
    );

    for pipe_radius in [0.0, 2.0] {
        for blend_radius in [8.0, 16.0] {
            let (doc, graph) = ball_blended_with_a_curve(pipe_radius, blend_radius);
            let field = match GraphField::compile_graph(&doc, &graph, 1e-3) {
                Ok(f) => f,
                Err(e) => {
                    println!("  {pipe_radius:>6.1}  {blend_radius:>6.0}  refused: {e}");
                    continue;
                }
            };
            let solid = solid_mm3(&field);
            println!(
                "  {pipe_radius:>6.1}  {blend_radius:>6.0}  {solid:>10.0}  {:>+9.0}  {:>9.2}  {:>9.2}  {:>9.2}",
                solid - base,
                reach_at(&field, 0.0),
                reach_at(&field, 10.0),
                reach_at(&field, 25.0)
            );
        }
    }

    // The soundness question, the same one the point had to answer: a fillet whose radius will
    // not be the radius asked for has to say so (D8).
    let (doc, graph) = ball_blended_with_a_curve(0.0, 16.0);
    let node = graph.roots()[0];
    let pipe = graph.node(node).expect("there").inputs()[1];
    println!(
        "\n  the pipe carries a true distance: {}\n  \
         the blended result carries a true distance: {}\n  \
         blends this build would mark approximate: {:?}",
        carries_true_distance(&doc, &graph, pipe),
        carries_true_distance(&doc, &graph, node),
        approximate_blends(&doc, &graph),
    );

    println!(
        "\nThe swelling follows the curve, which is what a point cannot do. Read the two reach\n\
         columns against the plain ball: on its own the surface falls from 20.00 mm at y = 0 to\n\
         17.32 at y = 10, because a sphere narrows. Blended, it holds near 28 mm at both. That\n\
         flatness along Y is the ridge. The y = 25 column reads nothing for every row alike --\n\
         it is past the ball and past the curve's own {SPAN} mm, so there is no material on that\n\
         axis to measure either way."
    );
}
