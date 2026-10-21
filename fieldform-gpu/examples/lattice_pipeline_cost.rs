// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a lattice shader costs to compile, cold.
//!
//! **Reported: "every time I go to file open, it's now taking a while ... it briefly hangs the
//! software"** — after the conformal lattice landed, and on opening a lattice file. The suspect
//! is the surface walk: `lattice_distance` reaches it through `if map > 2.5`, a branch on a
//! buffer value that no compiler can resolve statically, so every lattice shader carries a
//! 64-deep traversal loop whether or not anything in the document follows a surface.
//!
//! Timed cold. The driver caches compiled shaders by their source, and the source is structure
//! only, so an unchanged scene is served from disk in a fraction of what it cost the first time
//! — which is exactly the number a user does *not* pay after an edit. Each run therefore unions
//! the lattice's part with a clock-derived number of small spheres, which makes the structure
//! novel without changing what is being measured by more than a few leaves.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example lattice_pipeline_cost`

use fieldform_gpu::{generate, Renderer};
use fieldform_scene::{
    BooleanOp, CellMap, Document, Graph, LatticeCell, LatticePhase, Node, ParamRef,
};

fn scene(salt: usize) -> (Document, Graph) {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let lit = ParamRef::Literal;
    let cube = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        size: [lit(100.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::sheet(lit(2.0)),
    });
    let infill = graph.add(Node::Boolean {
        op: BooleanOp::Intersection,
        inputs: vec![cube.into(), lattice.into()],
    });
    let mut parts = vec![infill];
    for i in 0..salt {
        parts.push(graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(200.0 + i as f64 * 10.0), lit(0.0), lit(0.0)]),
            radius: lit(1.0),
        }));
    }
    if parts.len() > 1 {
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: parts.into_iter().map(Into::into).collect() });
    }
    (doc, graph)
}

fn main() {
    let clock = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as usize)
        .unwrap_or(0);
    println!("  {:>6}  {:>10}  {:>11}  {:>8}", "salt", "generate", "pipeline", "chars");
    for run in 0..2 {
        // `FIELDFORM_LATTICE_SALT` pins it, for a before-and-after across a change to the
        // generated code: the source differs between the two builds anyway, so both are cold,
        // and the same salt keeps the sphere count from being what was measured.
        let salt = std::env::var("FIELDFORM_LATTICE_SALT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|s| s + run * 90)
            .unwrap_or(1 + (clock + run * 7) % 97);
        let (doc, graph) = scene(salt);
        let started = std::time::Instant::now();
        let program = generate(&doc, &graph).expect("generates");
        let generated = started.elapsed().as_secs_f64() * 1e3;
        let started = std::time::Instant::now();
        match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("no GPU available ({e})");
                return;
            }
        }
        let pipeline = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "  {salt:>6}  {generated:>7.1} ms  {pipeline:>8.1} ms  {:>8}",
            program.source.len()
        );
    }
}
