// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a curve drag costs before anything reaches the GPU, and whether it has to regenerate
//! the shader at all.
//!
//! `cargo run -p fieldform-gpu --example curve_drag_cost -- <path.fform>`
//!
//! Written for `plane-test.fform` (2026-09-26), whose console log put a curve drag at
//! "draft: generating the shader 1208 ms, preparing the bands 221 ms" per rebuild, and the
//! settled pass after release at another 1314 ms of generation. The generated source is
//! structure alone -- every number is a `params[..]` read -- so the question is whether moving
//! a control point changes the source at all, and if it does not, what the 1.2 s is spent on.
//!
//! Measured on that document, dev profile, machine idle:
//!
//! | | ms |
//! |---|---|
//! | `generate` | 321 |
//! | `generate` after moving one control point by 1 mm | 329, source **identical** |
//! | of which `GraphField::compile` at 0.1 mm | 110 |
//! | of which `profile_loop` for the three profile blocks | 59 |
//! | `generate` again, nothing moved | 326 |
//!
//! So a curve drag regenerates a shader it already has, and the renderer would have kept its
//! pipeline anyway (`Renderer::rebuilt_from` keeps it when the source is this program's).
//! The 1.2 s the log showed was this 320 ms under two `cargo test --workspace` runs sharing
//! the machine. Skipping the generation when nothing structural changed is the fix, and which
//! numbers are structural is what master's `125c428` probe is establishing.

use std::time::Instant;

use fieldform_scene::Node;
use nalgebra::Vector3;

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let path = std::env::args().nth(1).expect("a .fform path");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let part = doc.parts.values().next().expect("a part");
    let graph = &part.graph;
    let read: Vec<fieldform_scene::CurveId> =
        graph.nodes().flat_map(|(_, n)| n.curves()).collect();
    println!(
        "{} curves, {} read by blocks, {} nodes",
        doc.curves.len(),
        read.len(),
        graph.nodes().count()
    );

    // What the traced view calls "generating the shader".
    let started = Instant::now();
    let program = fieldform_gpu::generate_with(&doc, graph, &[]).expect("generates");
    let generate_ms = ms(started);
    println!("generate: {generate_ms:.0} ms, {} chars", program.source.len());

    // The CPU compile, which the picking path runs and which generation contains the shape of.
    let started = Instant::now();
    let _field = fieldform_field::GraphField::compile(&doc, part, 0.1).expect("compiles");
    println!("GraphField::compile at 0.1 mm: {:.0} ms", ms(started));

    // The profile walks with their polylines, one per block that reads a profile.
    let started = Instant::now();
    let mut walks = 0;
    for (_, node) in graph.nodes() {
        let (profile, ignore) = match node {
            Node::Extrude { outline, sweep, .. } => (outline.curve(), sweep.source()),
            Node::Revolve { outline, frame, .. } => (outline.curve(), frame.curve_source()),
            Node::Sweep { outline, spine, .. } => (outline.curve(), Some(*spine)),
            _ => continue,
        };
        let _ = fieldform_field::profile_loop(&doc, profile, ignore);
        walks += 1;
    }
    println!("profile_loop for {walks} profile blocks: {:.0} ms", ms(started));

    // Move one control point of a curve a block reads, by a millimetre, and generate again.
    let curve = read[0];
    let mut moved = doc.clone();
    let c = moved.curve_mut(curve).expect("the curve");
    let p = c.control_points()[0];
    c.set_control_point(0, p + Vector3::new(1.0, 0.0, 0.0));
    let started = Instant::now();
    let again = fieldform_gpu::generate_with(&moved, graph, &[]).expect("generates");
    let again_ms = ms(started);
    let identical = program.source == again.source;
    println!(
        "after moving one control point: generate {again_ms:.0} ms, source identical: {identical}"
    );
    if !identical {
        let a: Vec<&str> = program.source.lines().collect();
        let b: Vec<&str> = again.source.lines().collect();
        let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        println!(
            "  {} lines before, {} after, {differing} lines differ",
            a.len(),
            b.len()
        );
        if let Some((i, (x, y))) = a.iter().zip(&b).enumerate().find(|(_, (x, y))| x != y) {
            println!("  first difference at line {}:\n    {x}\n    {y}", i + 1);
        }
    }

    // A second generation of the unchanged document, so a warm cache is not mistaken for
    // the geometry not mattering.
    let started = Instant::now();
    let _ = fieldform_gpu::generate_with(&doc, graph, &[]).expect("generates");
    println!("generate again, nothing moved: {:.0} ms", ms(started));
}
