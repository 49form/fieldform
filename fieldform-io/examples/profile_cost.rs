// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the profile rows cost on a real document: the partition, and one loop walk per curve.
//!
//! `cargo run --release -p fieldform-io --example profile_cost -- <path.fform>`
//!
//! Written for `plane-test.fform` (2026-09-26), reported "very slow" after the profile rows
//! landed. The rows are cached behind a curve-change counter, so the question is what one
//! recompute costs -- it runs on every frame of a drag -- and whether it is the walk or the
//! tessellation underneath it.

use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("a .fform path");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let open: Vec<_> = doc.curves.values().filter(|c| !c.is_closed()).collect();
    println!(
        "{} curves, {} open, {} nodes",
        doc.curves.len(),
        open.len(),
        doc.parts.values().map(|p| p.graph.nodes().count()).sum::<usize>()
    );

    // Tessellating every open curve once, at the walk's own angle: the floor for any walk.
    let started = Instant::now();
    let points: usize = open
        .iter()
        .map(|c| c.tessellate_to_angle(1.0, (c.length(0.5) * 1e-5).max(1e-4)).len())
        .sum();
    println!(
        "tessellate every open curve once: {:.1} ms, {points} points",
        started.elapsed().as_secs_f64() * 1e3
    );

    let started = Instant::now();
    let rows = fieldform_field::profile_partition(&doc);
    let partition_ms = started.elapsed().as_secs_f64() * 1e3;
    println!(
        "profile_partition: {partition_ms:.1} ms, {} rows: {:?}",
        rows.len(),
        rows.iter().map(|r| r.len()).collect::<Vec<_>>()
    );

    // One polyline walk per open curve, which is what the partition did underneath.
    let started = Instant::now();
    let mut ok = 0;
    for c in &open {
        if fieldform_field::profile_loop(&doc, c.id, None).is_ok() {
            ok += 1;
        }
    }
    println!(
        "profile_loop from every open curve: {:.1} ms, {ok} close",
        started.elapsed().as_secs_f64() * 1e3
    );

    // The ends-only walk from every open curve, which is what it does now.
    let started = Instant::now();
    let mut ok = 0;
    for c in &open {
        if fieldform_field::loop_from(&doc, c.id, None).is_ok() {
            ok += 1;
        }
    }
    println!(
        "loop_from every open curve: {:.3} ms, {ok} close",
        started.elapsed().as_secs_f64() * 1e3
    );

    let started = Instant::now();
    let _ = doc
        .curves
        .values()
        .map(|c| c.tessellate(0.05).len())
        .sum::<usize>();
    println!("curve display rebuild (0.05 mm): {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
}
