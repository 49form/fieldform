// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where the time goes when the viewport rebuilds.
//!
//! The architecture notes claim a performance seam: changing a parameter should be
//! "re-evaluate only — cheap, interactive", while a cage edit rebuilds one leaf and a graph
//! topology change rebuilds the tape. The app does not have that seam — `build_composed_mesh`
//! calls `GraphField::compile_graph` and then `mesh_field`, and `compile_graph` rebuilds every
//! cage field from scratch, so dragging a radius slider pays for everything.
//!
//! Before fixing that it is worth knowing which part actually costs anything. If meshing
//! dominates, caching leaves buys nothing and the interactivity problem is somewhere else
//! entirely. This measures the three stages separately at the resolutions and subdivision
//! levels the app really uses.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example mesh_rebuild_cost`

use fieldform_cage::Cage;
use fieldform_field::{CageField, GraphField, levels_for_tolerance, mesh_field};
use fieldform_scene::{BooleanOp, CageId, Curve, CurveId, Document, Graph, Node, ParamRef};
use nalgebra::Point3;
use std::time::Instant;

const CAGE_SIZE_MM: f64 = 1000.0;

fn main() {
    let (doc, graph) = scene();
    let cage = &doc.cages[doc.cages.keys().next().unwrap()];

    println!("scene: {CAGE_SIZE_MM} mm cube cage, minus a pipe along a curve\n");

    // --- stage 1: building one cage field, which is what compile does per cage node ---
    println!("--- CageField::from_cage (subdivision + BVH) ---");
    println!("  level   triangles      build");
    for level in 2..=6 {
        let start = Instant::now();
        let field = CageField::from_cage(cage, level).expect("valid cage");
        let build = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {level:>5}   {:>9}   {build:>7.1} ms",
            field.triangle_count()
        );
    }

    // --- stage 2: compiling the whole graph, which is stage 1 plus the op list ---
    println!("\n--- GraphField::compile_graph ---");
    println!("  tolerance    level      compile");
    for resolution in [32usize, 64, 128] {
        let tolerance = (CAGE_SIZE_MM / resolution as f64 * 0.5).max(1e-4);
        let level = levels_for_tolerance(CAGE_SIZE_MM, tolerance);
        let start = Instant::now();
        let field = GraphField::compile_graph(&doc, &graph, tolerance).expect("valid graph");
        let compile = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&field);
        println!("  {tolerance:>7.1} mm    {level:>5}    {compile:>8.1} ms");
    }

    // --- stage 3: meshing, which is what the viewport uploads ---
    println!("\n--- compile then mesh, as build_composed_mesh does ---");
    println!("  cells    compile      mesh     total   compile share");
    for resolution in [32usize, 64, 128] {
        let tolerance = (CAGE_SIZE_MM / resolution as f64 * 0.5).max(1e-4);

        let start = Instant::now();
        let field = GraphField::compile_graph(&doc, &graph, tolerance).expect("valid graph");
        let compile = start.elapsed().as_secs_f64() * 1000.0;

        let bounds = (
            Point3::new(-600.0, -600.0, -600.0),
            Point3::new(600.0, 600.0, 600.0),
        );
        let start = Instant::now();
        let mesh = mesh_field(&field, bounds, resolution);
        let mesh_ms = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(mesh.triangle_count());

        let total = compile + mesh_ms;
        println!(
            "  {resolution:>5}   {compile:>7.1}ms  {mesh_ms:>7.1}ms  {total:>7.1}ms   {:>6.1}%",
            compile / total * 100.0
        );
    }

    println!(
        "\nThe compile share is what caching cage fields could remove. Whatever is left is\n\
         meshing, and no amount of caching touches it."
    );
}

/// The app's default composed scene: a cage with a pipe subtracted from it.
fn scene() -> (Document, Graph) {
    let mut doc = Document::new();
    let cage_id = CageId::new();
    doc.cages.insert(cage_id, Cage::cube_of(CAGE_SIZE_MM));

    let curve_id = CurveId::new();
    let s = CAGE_SIZE_MM;
    let curve = Curve::polyline(
        curve_id,
        "probe",
        vec![
            Point3::new(-s, -0.18 * s, -0.10 * s),
            Point3::new(-0.3 * s, -0.05 * s, 0.02 * s),
            Point3::new(0.3 * s, 0.10 * s, 0.05 * s),
            Point3::new(s, 0.22 * s, 0.12 * s),
        ],
        false,
    )
    .expect("valid polyline");
    doc.curves.insert(curve_id, curve);

    let mut graph = Graph::from_cage(cage_id);
    let cage_node = graph.roots()[0];
    let pipe = graph.add(Node::Pipe {
        curve: curve_id,
        radius: ParamRef::Literal(111.0),
    });
    graph.add(Node::Boolean {
        op: BooleanOp::Difference,
        inputs: vec![cage_node.into(), pipe.into()],
    });

    (doc, graph)
}
