// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a pattern costs as its count rises.
//!
//! Every instance is a placement of the *same* body, which is what makes a pattern parametric —
//! but a shader has no loops over graph nodes, so each placement emits the body's expression
//! again. The generated source therefore grows with the count, and the question this answers is
//! where that stops being reasonable.
//!
//! Worth knowing because Modify > Pattern offers a count, and a knob that can hang the
//! application is worse than a knob with a smaller range.
//!
//! `cargo run --release -p fieldform-gpu --example pattern_cost`

use fieldform_gpu::codegen;
use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef};

/// A radial pattern of `count` lugs, built the way `App::apply_pattern` builds one.
fn pattern(count: u32) -> Graph {
    let mut graph = Graph::empty();
    let lug = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([ParamRef::Literal(30.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
        size: [
            ParamRef::Literal(30.0),
            ParamRef::Literal(12.0),
            ParamRef::Literal(40.0),
        ],
        turn: fieldform_scene::no_turn(),
    });
    let mut combined = lug;
    for i in 1..count {
        let placed = graph.add(Node::Transform {
            input: lug.into(),
            translation: [ParamRef::Literal(0.0); 3],
            rotation: [
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(360.0 * i as f64 / count as f64),
            ],
            scale: ParamRef::Literal(1.0),
            about: fieldform_scene::Place::origin(),
        });
        combined = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![combined.into(), placed.into()],
        });
    }
    graph
}

fn main() {
    let doc = Document::new();
    println!("A radial pattern of one body, at rising counts.\n");
    println!("{:>7}  {:>10}  {:>12}  {:>10}", "copies", "params", "WGSL chars", "generate");

    for count in [2u32, 6, 12, 24, 48, 96, 192, 384, 512] {
        let graph = pattern(count);
        let started = std::time::Instant::now();
        let program = match codegen::generate(&doc, &graph) {
            Ok(p) => p,
            Err(e) => {
                println!("{count:>7}  refused — {e}");
                continue;
            }
        };
        let ms = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "{count:>7}  {:>10}  {:>12}  {ms:>8.1} ms",
            program.params.len(),
            program.source.len(),
        );
    }

    println!(
        "\nThe body's expression is emitted once per placement, because each is evaluated at a\n\
         different point — so both columns grow with the count and neither is a surprise. What\n\
         matters is where a shader stops being something a driver will compile quickly."
    );
}
