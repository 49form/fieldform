// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A torus whose facing is taken from a line — the click that overflowed the stack.
//!
//! Reported with the answer in it: "crashed again — tried providing the angle input a line on the
//! torus block", and this time stderr was captured and says `thread 'main' has overflowed its
//! stack`. So it is recursion, not a panic and not a hang, and the shape of the scene is known.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example torus_along_a_line`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Curve, Document, Graph, Node, ParamRef, Place, Turn};
use nalgebra::Point3;

fn main() {
    let lit = ParamRef::Literal;
    let mut doc = Document::new();
    let curve = fieldform_scene::CurveId::new();
    doc.curves.insert(
        curve,
        Curve::polyline(
            curve,
            "Line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(40.0, 0.0, 10.0)],
            false,
        )
        .expect("a two-point line"),
    );

    let mut graph = Graph::empty();
    let ring = graph.add(Node::Torus {
        centre: Place::At([lit(0.0); 3]),
        major_radius: lit(30.0),
        minor_radius: lit(8.0),
        turn: Turn::Along { curve },
    });
    println!("roots: {:?}", graph.roots());
    println!("the torus names the curve: {:?}", graph.node(ring).map(|n| n.curves()));

    let numbers = |r: &ParamRef| -> Option<f64> { r.resolve(&|_| None) };
    println!("\nflattening the turn...");
    let flat = fieldform_field::with_turns_flattened(&doc, &graph, &numbers);
    println!("  {} nodes, {} roots", flat.len(), flat.roots().len());
    println!(
        "  evaluation order: {}",
        match flat.evaluation_order() {
            Some(order) => format!("{} nodes", order.len()),
            None => "REFUSED as cyclic".to_string(),
        }
    );

    println!("\ncompiling...");
    match GraphField::compile_graph(&doc, &graph, 1e-4) {
        Ok(field) => println!(
            "  the field compiles; at the centre it reads {:.3}",
            field.distance(Point3::origin())
        ),
        Err(e) => println!("  the field REFUSED: {e}"),
    }
    match fieldform_gpu::generate(&doc, &graph) {
        Ok(_) => println!("  the shader compiles"),
        Err(e) => println!("  the shader REFUSED: {e}"),
    }
}
