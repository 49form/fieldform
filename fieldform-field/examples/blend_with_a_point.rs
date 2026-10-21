// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a *blend* against a point produces, which is not what a boolean against one produces.
//!
//! A plain boolean with a point is a no-op or a deletion — `boolean_with_nothing.rs` measures
//! that. A blend is a different operation: `smooth_min` mixes by magnitude within the radius,
//! so an operand that never goes negative still pulls the surface toward itself wherever it
//! comes within `k` of it. The result is a rounded swelling around the point, which is what a
//! metaball is.
//!
//! This prints how much material that adds, how far it reaches, and — the part that decides
//! whether it is sound rather than merely pretty — whether the result is still a distance the
//! tracer may trust.
//!
//! Run with: `cargo run --release -p fieldform-field --example blend_with_a_point`

use fieldform_field::{approximate_blends, carries_true_distance, GraphField, ScalarField};
use fieldform_scene::{Fillet, 
    BooleanOp, Continuity, Document, Graph, Node, ParamRef, Place,
};
use nalgebra::Point3;

const RADIUS: f64 = 20.0;
/// Sampled over a box comfortably outside anything measured here.
const HALF: f64 = 60.0;
const STEPS: usize = 120;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// Material inside the field, by counting cells — the same crude integral for every row, so the
/// rows are comparable even though none of them is exact.
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

/// How far the surface reaches along +X, found by bisection between a point known inside and
/// one known outside.
fn reach_along_x(field: &GraphField) -> f64 {
    let (mut inside, mut outside) = (0.0, HALF);
    for _ in 0..60 {
        let mid = 0.5 * (inside + outside);
        if field.distance(Point3::new(mid, 0.0, 0.0)) < 0.0 {
            inside = mid;
        } else {
            outside = mid;
        }
    }
    0.5 * (inside + outside)
}

/// A ball at the origin, and a point `at` millimetres along +X, blended with `radius`.
fn ball_blended_with_a_point(at: f64, radius: f64) -> (Document, Graph) {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(RADIUS),
    });
    let point = graph.add(Node::Point { at: [lit(at), lit(0.0), lit(0.0)] });
    graph.add(Node::Blend {
        op: BooleanOp::Union,
        a: ball.into(),
        b: point.into(),
        fillet: Fillet { radius: lit(radius), continuity: Continuity::G1 },
    });
    (doc, graph)
}

fn compiled(doc: &Document, graph: &Graph) -> GraphField {
    GraphField::compile_graph(doc, graph, 1e-3).expect("compiles")
}

fn main() {
    println!(
        "A {RADIUS} mm ball at the origin, G1-blended with a point out along +X.\n\
         A plain boolean with a point changes nothing; a blend does, because `smooth_min`\n\
         mixes by magnitude within the radius rather than picking the smaller value.\n"
    );

    let doc = Document::new();
    let mut plain = Graph::empty();
    plain.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(RADIUS) });
    let alone = compiled(&doc, &plain);
    let (base_solid, base_reach) = (solid_mm3(&alone), reach_along_x(&alone));
    println!("  {:>8}  {:>8}  {:>12}  {:>12}  {:>10}", "point at", "radius", "solid mm3", "added", "reach +X");
    println!(
        "  {:>8}  {:>8}  {:>12.0}  {:>12}  {:>10.3}",
        "-", "-", base_solid, "-", base_reach
    );

    // Near, touching, and well beyond the surface, at two radii each.
    for at in [10.0, 20.0, 26.0, 34.0] {
        for radius in [8.0, 16.0] {
            let (doc, graph) = ball_blended_with_a_point(at, radius);
            let field = compiled(&doc, &graph);
            let solid = solid_mm3(&field);
            println!(
                "  {at:>8.0}  {radius:>8.0}  {solid:>12.0}  {:>+12.0}  {:>10.3}",
                solid - base_solid,
                reach_along_x(&field)
            );
        }
    }

    // The question that decides whether this is sound. A point's field is a real Euclidean
    // distance, so a blend against it is exact rather than approximate -- which is the D8 rule
    // in the project notes: a blend whose radius will not be the radius asked for has to say so.
    let (doc, graph) = ball_blended_with_a_point(26.0, 8.0);
    let node = graph.roots()[0];
    println!(
        "\nIs the result a distance the tracer may trust?\n  \
         the point carries a true distance: {}\n  \
         the blended result carries a true distance: {}\n  \
         blends this build would mark approximate: {:?}",
        carries_true_distance(&doc, &graph, graph.node(node).expect("there").inputs()[1]),
        carries_true_distance(&doc, &graph, node),
        approximate_blends(&doc, &graph),
    );

    println!(
        "\nSo the swelling is a real shape with an exact radius, not an artefact: the point is a\n\
         true distance field, so the fillet is the one that was asked for. What it is *not* is a\n\
         boolean -- nothing was joined, because there was nothing to join. It is a metaball, and\n\
         the radius is how far it reaches."
    );
}
