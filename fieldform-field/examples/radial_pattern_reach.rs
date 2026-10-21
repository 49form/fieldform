// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Does a radial pattern look at enough of its neighbours?
//!
//! Reported: "made a sphere and patterned it 24 times, circular, around x=125 — no longer looks
//! like a sphere is being patterned, rather its intersection. Increasing the X coordinate breaks
//! it the same way."
//!
//! A pattern is not evaluated once per instance. The point is folded into one cell and a window
//! of neighbouring cells is checked, and `pattern_reach` decides how wide that window is — one
//! answer, asked by the field and the shader alike, so that neither can be wrong on its own. If
//! that number is too small the copies are clipped to their own wedges, which is exactly what
//! "looks like its intersection" describes.
//!
//! So this compares the pattern against the thing it is supposed to be equal to: the same body
//! written out as `count` placements, unioned. That union is ground truth by construction.
//!
//! Run with: `cargo run --release -p fieldform-field --example radial_pattern_reach`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Fillet, 
    Axis, BooleanOp, Document, Frame, Graph, Node, ParamRef, PatternKind, Place,
};
use nalgebra::Point3;

const COUNT: u32 = 24;
const RADIUS_MM: f64 = 20.0;

/// The pattern as the block builds it: one body, folded.
fn folded(x: f64) -> Graph {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
        radius: lit(RADIUS_MM),
    });
    graph.add(Node::Pattern {
        options: Default::default(),
        input: ball.into(),
        count: ParamRef::Literal(f64::from(COUNT)),
        kind: PatternKind::Radial {
            frame: Frame::at_along(Axis::Z, [lit(0.0); 3]),
        },
    
        join: true,
        fillet: Fillet::hard(),
    });
    graph
}

/// The same thing written out: what the fold is supposed to equal.
fn spelled_out(x: f64) -> Graph {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
        radius: lit(RADIUS_MM),
    });
    let mut copies = vec![ball];
    for i in 1..COUNT {
        copies.push(graph.add(Node::Transform {
            input: ball.into(),
            translation: [lit(0.0); 3],
            rotation: [
                lit(0.0),
                lit(0.0),
                lit(360.0 * f64::from(i) / f64::from(COUNT)),
            ],
            scale: lit(1.0),
            about: Place::origin(),
        }));
    }
    graph.add(Node::Boolean { op: BooleanOp::Union, inputs: copies.into_iter().map(Into::into).collect() });
    graph
}

fn main() {
    let doc = Document::new();
    println!(
        "A {RADIUS_MM} mm sphere, patterned {COUNT} times about Z, against the same {COUNT}\n\
         placements unioned. `reach` is how many cells either side the fold looks at; `cell` is\n\
         how wide one is at that radius, and `ball` how wide the sphere is seen from the axis.\n"
    );
    println!(
        "  {:>7}  {:>6}  {:>9}  {:>9}  {:>9}  {:>12}  {:>9}",
        "x mm", "reach", "cell deg", "ball deg", "needs", "worst mm", "agrees"
    );

    for x in [0.0f64, 30.0, 60.0, 125.0, 200.0, 400.0] {
        let folded_graph = folded(x);
        let ball = folded_graph
            .nodes()
            .find(|(_, n)| matches!(n, Node::Sphere { .. }))
            .map(|(id, _)| id)
            .expect("the sphere");
        let kind = PatternKind::Radial {
            frame: Frame::at_along(Axis::Z, [ParamRef::Literal(0.0); 3]),
        };
        let reach = fieldform_field::pattern_reach(&doc, &folded_graph, ball, &kind, COUNT, 0.0);

        let a = GraphField::compile_graph(&doc, &folded_graph, 1e-3).expect("compiles");
        let b = GraphField::compile_graph(&doc, &spelled_out(x), 1e-3).expect("compiles");

        // How wide the ball actually is from the axis, and how many cells that spans -- the
        // number the window has to cover.
        let cell = 360.0 / f64::from(COUNT);
        let ball_deg = if x <= RADIUS_MM {
            360.0
        } else {
            2.0 * (RADIUS_MM / x).asin().to_degrees()
        };
        let needs = (ball_deg / cell / 2.0).ceil() as usize;

        // Sampled through the ring the copies live in, where a wedge that has been clipped shows.
        let mut worst = 0.0f64;
        for i in 0..4000 {
            let t = f64::from(i) / 4000.0;
            let angle = t * std::f64::consts::TAU;
            let r = x + RADIUS_MM * (1.4 * (t * 37.0).sin());
            let p = Point3::new(
                r * angle.cos(),
                r * angle.sin(),
                RADIUS_MM * 0.9 * (t * 17.0).cos(),
            );
            worst = worst.max((a.distance(p) - b.distance(p)).abs());
        }

        println!(
            "  {x:>7.0}  {reach:>6}  {cell:>9.1}  {ball_deg:>9.1}  {needs:>9}  {worst:>12.3}  {:>9}",
            if worst < 1e-6 { "yes" } else { "NO" },
        );
    }

    println!(
        "\n`agrees` is the whole test: a fold that looks at enough neighbours is *equal* to the\n\
         placements it stands for, everywhere. Where it says no, the window is too narrow and\n\
         the copies are being clipped to their own wedges -- which is what was reported as the\n\
         pattern looking like an intersection."
    );
}
