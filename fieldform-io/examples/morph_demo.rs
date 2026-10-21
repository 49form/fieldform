// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Writes a document demonstrating `Node::Morph`, to open in the application.
//!
//! Two bodies **in the same place** — the arrangement a constant morph is for. Two bodies apart
//! from each other average to nothing in between, so a demo built that way would show the shape
//! vanishing and read as a bug rather than as the point.
//!
//! Run with: `cargo run --release -p fieldform-io --example morph_demo -- <path.fieldform>`

use fieldform_scene::{Document, Graph, Node, ParamRef, Part, PartId, Place};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "morph-demo.fieldform".to_string());

    let lit = ParamRef::Literal;
    let mm = |v: f64| lit(v);
    let centre = Place::At([mm(0.0), mm(0.0), mm(0.0)]);

    let mut graph = Graph::empty();
    // A ball and a block on the same centre, sized so neither swallows the other: the cube's
    // corners reach past the sphere and its faces fall inside it, so half way between them is a
    // shape that is neither — which is the thing worth looking at.
    let ball = graph.add(Node::Sphere { centre, radius: mm(50.0) });
    let block = graph.add(Node::Cuboid { edge: Default::default(),
        centre,
        size: [mm(78.0), mm(78.0), mm(78.0)],
        turn: fieldform_scene::no_turn(),
    });
    let morph = graph.add(Node::Morph { a: ball.into(), b: block.into(), t: lit(0.5) });

    let mut doc = Document::new();
    let part = PartId::new();
    doc.parts.insert(
        part,
        Part {
            id: part,
            name: "Morph demo".to_string(),
            material: None,
            graph,
            inputs: Vec::new(),
        },
    );
    // Named, so the block list says what to drag rather than leaving it to be found.
    doc.node_meta.entry(morph).or_default().name = Some("Ball to block".to_string());

    match fieldform_io::document::save_new(&doc, std::path::Path::new(&path)) {
        Ok(()) => println!(
            "wrote {path}\n\
             \x20 open it, select \u{201c}Ball to block\u{201d}, and drag Amount from 0 to 100%.\n\
             \x20 0% is the sphere, 100% is the cube, and everything between is neither."
        ),
        Err(e) => {
            eprintln!("could not write {path}: {e}");
            std::process::exit(1);
        }
    }
}
