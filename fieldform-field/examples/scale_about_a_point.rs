// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a placement's pivot does, and what it costs not to have one.
//!
//! Asked for as "a scale tool, to scale an object up from a point by a factor". A placement
//! already scales; what it could not do is say where from, so a body not centred on the local
//! origin swung away as it grew and the way to stop it was arithmetic that changes every time
//! the factor does.
//!
//! Run with: `cargo run --release -p fieldform-field --example scale_about_a_point`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Document, Graph, Node, ParamRef, Place};
use nalgebra::Point3;

/// Where the ball is, and how big.
const AT: [f64; 3] = [40.0, 0.0, 0.0];
const RADIUS: f64 = 10.0;

fn main() {
    println!(
        "A {RADIUS} mm ball at x = {}, scaled about three different points.\n\
         Where its near and far faces end up, and what stayed still:\n",
        AT[0]
    );
    println!(
        "  {:>22}  {:>8}  {:>8}  {:>8}  {}",
        "pivot", "factor", "near", "far", "fixed point"
    );
    for (name, pivot) in [
        ("the origin", [0.0, 0.0, 0.0]),
        ("the ball's own centre", AT),
        ("its near face", [AT[0] - RADIUS, 0.0, 0.0]),
    ] {
        for factor in [1.0f64, 2.0] {
            let field = built(factor, pivot);
            let (near, far) = faces(&field);
            let held = held_still(&field, pivot);
            println!(
                "  {name:>22}  {factor:>8.1}  {near:>8.2}  {far:>8.2}  {held}",
            );
        }
    }

    println!(
        "\nScaled about the origin, a ball 40 mm out goes to 80 mm out: doubling the distance is\n\
         what scaling about the origin means, and it is the answer nobody wants from a tool\n\
         called `scale`. About its own centre it grows in place. About its near face, that face\n\
         stays put and the ball grows away from it \u{2014} which is what somebody scaling a part\n\
         off a datum is asking for."
    );

    // A pivot is a fixed point, so the check is that the point does not move. Measured rather
    // than reasoned: the cancelling translation is easy to write with a sign the wrong way
    // round, and the shape still looks plausible when it is.
    println!("\nHow far the pivot itself moves, over a range of factors:\n");
    println!("  {:>8}  {:>26}", "factor", "the pivot moved by");
    for factor in [0.25f64, 0.5, 1.0, 1.5, 2.0, 4.0] {
        let pivot = [AT[0] - RADIUS, 0.0, 0.0];
        let field = built(factor, pivot);
        // The near face is on the pivot, so the field there says how far the surface has left
        // it. Zero means the point is genuinely fixed.
        let at = Point3::new(pivot[0], pivot[1], pivot[2]);
        println!("  {factor:>8.2}  {:>26.9}", field.distance(at).abs());
    }
}

/// Where the ball's near and far faces sit along X, by walking in from outside.
fn faces(field: &GraphField) -> (f64, f64) {
    let inside = |x: f64| field.distance(Point3::new(x, 0.0, 0.0)) < 0.0;
    let mut near = f64::NAN;
    let mut far = f64::NAN;
    let mut was = false;
    let mut x = -50.0;
    while x < 400.0 {
        let now = inside(x);
        if now && !was {
            near = x;
        }
        if was && !now {
            far = x;
        }
        was = now;
        x += 0.001;
    }
    (near, far)
}

/// Which of the two faces, if either, the scale left where it was.
fn held_still(field: &GraphField, pivot: [f64; 3]) -> &'static str {
    let at = Point3::new(pivot[0], pivot[1], pivot[2]);
    match field.distance(at).abs() < 1e-6 {
        true => "the pivot, on the surface",
        false => "",
    }
}

fn built(factor: f64, pivot: [f64; 3]) -> GraphField {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(AT[0]), lit(AT[1]), lit(AT[2])]),
        radius: lit(RADIUS),
    });
    graph.add(Node::Transform {
        input: ball.into(),
        translation: [lit(0.0); 3],
        rotation: [lit(0.0); 3],
        scale: lit(factor),
        about: Place::At([lit(pivot[0]), lit(pivot[1]), lit(pivot[2])]),
    });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
}
