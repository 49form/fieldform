// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Prints the shader a radial pattern generates, beside the numbers the field folds with.
//!
//! Reported: a sphere patterned 24 times about an axis at x = 125 stops looking like spheres.
//! `radial_pattern_reach` already showed the CPU field is exactly right there, so what is left
//! is the shader -- and the fastest way to read a shader is to read it.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example ring_shader`

use fieldform_scene::{Fillet, Axis, Document, Frame, Graph, Node, ParamRef, PatternKind, Place};

fn main() {
    let lit = ParamRef::Literal;
    let doc = Document::new();
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0); 3]),
        radius: lit(20.0),
    });
    let kind = PatternKind::Radial {
        frame: Frame::at_along(Axis::Z, [lit(125.0), lit(0.0), lit(0.0)]),
    };
    let reach = fieldform_field::pattern_reach(&doc, &graph, ball, &kind, 24, 0.0);
    graph.add(Node::Pattern { options: Default::default(), input: ball.into(), count: ParamRef::Literal(24.0), kind, join: true, fillet: Fillet::hard() });

    println!("reach {reach} cells either side of {} in the window\n", 2 * reach + 1);
    let program = fieldform_gpu::generate(&doc, &graph).expect("generates");
    for (i, p) in program.params.iter().enumerate() {
        println!("  params[{i}] = {p:?}");
    }
    println!("\n{}", program.field_function());
}
