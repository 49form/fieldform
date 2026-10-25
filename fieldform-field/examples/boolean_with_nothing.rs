// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a point can be combined with, since the panel has to decide what to offer.
//!
//! A point's field is the distance to it: never negative, zero at exactly one place. That makes
//! it useless to a *hard* boolean, which depends only on sign — `min` and `max` against
//! something that never goes negative cannot add or remove material, which this prints rather
//! than argues.
//!
//! A **blend** is a different question with a different answer, and the reason this example is
//! not simply a case for refusing points. `smooth_min` mixes by magnitude rather than by sign,
//! so it reaches toward a zero-volume operand and pulls a spike out to it, with the surface
//! passing exactly through the point.
//!
//! A plane is the opposite case and is here for contrast: it encloses a half-space, so
//! intersecting with one is the standard way to trim a part and must keep working.
//!
//! Run with: `cargo run --release -p fieldform-field --example boolean_with_nothing`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{BooleanOp, Continuity, Document, Graph, Node, ParamRef, Place};
use nalgebra::Point3;

const RADIUS: f64 = 20.0;

fn main() {
    println!(
        "A {RADIUS} mm ball at the origin, combined with a point sitting 8 mm off centre.\n\
         Sampled on a grid, how much material is left and where the surface is:\n"
    );
    println!("  {:>28}  {:>10}  {:>14}", "", "solid mm3", "surface");
    let ball = only_the_ball();
    report("the ball alone", &ball);
    for op in [BooleanOp::Union, BooleanOp::Difference, BooleanOp::Intersection] {
        report(&format!("{op:?} with a point"), &with_a_point(op));
    }
    println!(
        "\nUnion and Difference leave the ball exactly as it was: `min` and `max` against a field\n\
         that is never negative cannot add or remove material, so the only thing either adds to\n\
         the zero set is the single point itself, which is smaller than a pixel. Intersection is\n\
         worse than useless \u{2014} the whole body goes, because there is no volume in common\n\
         between a solid and something with none."
    );

    // What the field says near a point that is *outside* the solid, which is the part that is
    // not merely a no-op. A point buried in the ball degrades nothing: everywhere outside, the
    // ball's own surface is nearer than a point inside it, so `min` never picks the point.
    // Put it outside and the union starts answering with the distance to it.
    let outside = Point3::new(40.0, 0.0, 0.0);
    let near = Point3::new(40.0, 0.0, 6.0);
    println!(
        "\nWhether it costs anything depends on where the point is. Inside the ball it never wins\n\
         a `min`, so the field is untouched -- which is why the reading above is the same either\n\
         way. Outside, it does. With the point at 40 mm and a sample 6 mm from it:\n"
    );
    let plain = only_the_ball();
    let joined = point_outside();
    println!(
        "  the ball alone reads {:.3} mm there, and the union reads {:.3}",
        plain.distance(near),
        joined.distance(near)
    );
    println!(
        "  at the point itself the union reads {:.6}, which is a surface as far as any threshold\n\
         \x20 test can tell. The march confirms its hits now, so it steps through instead of\n\
         \x20 stopping on it -- but it is still a zero set that is not a shape.",
        joined.distance(outside)
    );

    // ---------------------------------------------------------------- and what a blend does
    //
    // The interesting half, and the reason a point is worth offering after all. `smooth_min` is
    // not `min`: below the radius it mixes both operands by how far inside the blend each is, so
    // an operand with no volume still moves the surface.
    println!(
        "\nA blend is a different question. `smooth_min` mixes by magnitude rather than by sign,\n\
         so it reaches for a point instead of ignoring it. The same ball, blended to a point\n\
         40 mm out along X:\n"
    );
    println!(
        "  {:>8}  {:>14}  {:>12}  {:>12}  {:>16}",
        "radius", "at the point", "volume mm3", "added", "reaches to x ="
    );
    let base = solid_volume(&only_the_ball());
    for radius in [2.0f64, 10.0, 20.0, 30.0, 34.0, 36.0, 40.0, 60.0] {
        let field = blended(radius);
        let volume = solid_volume(&field);
        println!(
            "  {radius:>8.1}  {:>14.6}  {volume:>12.0}  {:>12.0}  {:>16.2}",
            field.distance(Point3::new(40.0, 0.0, 0.0)),
            volume - base,
            far_x(&field)
        );
    }
    println!(
        "\n**Two regimes, and the first one is a trap.** Below a threshold the reading at the\n\
         point is exactly nought and nothing else changes: no volume, and the solid still stops\n\
         where the ball did. That nought is a single tangent touch -- `smooth_min(a, 0, k)` is\n\
         `k - ((k-a)+^n + k^n)^(1/n)`, and with the ball further than `k` away the first term\n\
         vanishes and what is left is exactly `k - k`. It is a zero set with no inside, which is\n\
         the same phantom a union with a point leaves and is not a shape.\n\n\
         Above the threshold the reading goes negative, the volume grows and the solid reaches\n\
         past the point. That is the case worth having: a blend to a point is a *blob that\n\
         swallows the point*, not a taper that ends at it."
    );

    // Where the change happens, as a fraction of the gap, because the panel has to choose a
    // radius when somebody blends to a point and a default below this does nothing visible.
    println!(
        "\nWhere it changes, for a few gaps between the ball's surface and the point:\n"
    );
    println!("  {:>8}  {:>16}  {:>14}", "gap mm", "connects above", "as a ratio");
    for out in [30.0f64, 40.0, 60.0, 100.0] {
        let gap = out - RADIUS;
        // The smallest radius that puts material at the point rather than a touch.
        let mut lo = 0.0;
        let mut hi = gap * 4.0;
        for _ in 0..40 {
            let mid = (lo + hi) * 0.5;
            let at = blended_to(mid, out).distance(Point3::new(out, 0.0, 0.0));
            if at < -1e-9 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        println!("  {gap:>8.1}  {hi:>16.3}  {:>14.4}", hi / gap);
    }
    println!(
        "\nThe ratio is exactly one at every gap, which is as clean a rule as this could give: the\n\
         blend radius has to be **greater than the distance from the point to the surface**\n\
         it is being blended to. Anything less, however finely tuned, is the phantom.\n\n\
         And at a radius of nought there is nothing at all, because `smooth_min` with no radius\n\
         *is* `min` -- the case in the first table that does nothing. So the rule for a point is\n\
         not which operand it is, but that the operation has to be a blend and that its radius\n\
         has to clear the gap."
    );

    println!(
        "\nA plane, for contrast, encloses a half-space and is the standard way to trim a part:"
    );
    report("Intersection with a plane", &with_a_plane());
    println!(
        "\nSo the rule is not \u{201c}refuse anything that is not a solid\u{201d}, and it is not\n\
         \u{201c}a point cannot be an operand\u{201d} either. A *hard* boolean needs both operands\n\
         to enclose material; a blend does not, and needs a radius above nought instead. A plane\n\
         encloses a half-space and works either way."
    );
}

/// How much material a field encloses, by sampling — a wider box than `report` uses, because a
/// blend to a point 40 mm out reaches well outside the ball's own neighbourhood.
fn solid_volume(field: &GraphField) -> f64 {
    let (mut inside, n, span) = (0usize, 60, 120.0);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = Point3::new(
                    -span / 2.0 + span * i as f64 / (n - 1) as f64,
                    -span / 2.0 + span * j as f64 / (n - 1) as f64,
                    -span / 2.0 + span * k as f64 / (n - 1) as f64,
                );
                if field.distance(p) < 0.0 {
                    inside += 1;
                }
            }
        }
    }
    inside as f64 * (span / (n - 1) as f64).powi(3)
}

/// How far the solid reaches along +X, which is the direction the spike goes.
fn far_x(field: &GraphField) -> f64 {
    let mut x = 60.0;
    while x > -30.0 && field.distance(Point3::new(x, 0.0, 0.0)) > 0.0 {
        x -= 0.001;
    }
    x
}

/// The ball blended to a point 40 mm out, which is well clear of its surface.
fn blended(radius: f64) -> GraphField {
    blended_to(radius, 40.0)
}

/// The same, with the point at a chosen distance along X.
fn blended_to(radius: f64, out: f64) -> GraphField {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(RADIUS),
    });
    let point = graph.add(Node::Point { at: [lit(out), lit(0.0), lit(0.0)] });
    graph.add(Node::Blend {
        op: BooleanOp::Union,
        a: ball.into(),
        b: point.into(),
        // The only continuity there is, so "set it to G1" is the same instruction as "make it a
        // blend" — which is what the panel now does when a point is picked.
        fillet: fieldform_scene::Fillet { radius: lit(radius), continuity: Continuity::G1, second: None },
    });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
}

/// Solid volume by sampling, and whether any surface remains.
fn report(name: &str, field: &GraphField) {
    let (mut inside, mut total) = (0usize, 0usize);
    let n = 60;
    let span = 60.0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = Point3::new(
                    -span / 2.0 + span * i as f64 / (n - 1) as f64,
                    -span / 2.0 + span * j as f64 / (n - 1) as f64,
                    -span / 2.0 + span * k as f64 / (n - 1) as f64,
                );
                total += 1;
                if field.distance(p) < 0.0 {
                    inside += 1;
                }
            }
        }
    }
    let cell = (span / (n - 1) as f64).powi(3);
    let volume = inside as f64 * cell;
    let surface = if inside == 0 {
        "none — nothing left"
    } else if inside == total {
        "none — solid everywhere sampled"
    } else {
        "present"
    };
    println!("  {name:>28}  {volume:>10.0}  {surface:>14}");
}

fn only_the_ball() -> GraphField {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(RADIUS),
    });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
}

fn with_a_point(op: BooleanOp) -> GraphField {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(RADIUS),
    });
    let point = graph.add(Node::Point { at: [lit(8.0), lit(0.0), lit(0.0)] });
    graph.add(Node::Boolean { op, inputs: vec![ball.into(), point.into()] });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
}

/// The same union, with the point out where it can win a `min`.
fn point_outside() -> GraphField {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(RADIUS),
    });
    let point = graph.add(Node::Point { at: [lit(40.0), lit(0.0), lit(0.0)] });
    graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![ball.into(), point.into()] });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
}

fn with_a_plane() -> GraphField {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(RADIUS),
    });
    let plane = graph.add(Node::Plane {
        origin: Place::At([lit(0.0), lit(0.0), lit(5.0)]),
        flipped: false,
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![ball.into(), plane.into()] });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
}
