// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What hiding one body in an assembly actually costs, step by step.
//!
//! Reported 2026-09-02: an assembly of twenty or thirty small solids hangs for close to a
//! second every time a body is hidden. Two rounds of fixes had already removed the per-frame
//! graph compile, the per-action BVH rebuild and the redundant triangle upload — so what is
//! left is whatever scales with the *number of bodies*, and the point of this is to find out
//! which part that is rather than to guess a fourth time.
//!
//! Hiding removes a root from the display graph, which means a different graph, which means:
//! a new shader is generated, parsed and validated; the driver compiles it; every leaf is
//! baked again; and the atlas is written. Each of those is a candidate and they are decades
//! apart in cost.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example hide_cost`

use fieldform_field::{GraphField, LeafCache, MeshField, ScalarField};
use fieldform_gpu::{codegen, Renderer};
use fieldform_scene::{Document, Graph, MeshData, Node, ParamRef};
use nalgebra::{Point3, Vector3};

const BODIES: usize = 30;
/// Triangles per body, which for a small part in an assembly is a few thousand.
const SIDES: usize = 24;

fn main() {
    let (doc, graph) = assembly(BODIES);
    let leaves: Vec<MeshField> = doc
        .meshes
        .values()
        .map(|m| {
            let positions = m.positions.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect();
            MeshField::from_mesh(positions, m.triangles.clone()).expect("closed")
        })
        .collect();
    let total: usize = doc.meshes.values().map(|m| m.triangles.len()).sum();
    println!("{BODIES} bodies, {total} triangles in all.\n");

    // 1. The CPU compile, which the leaf cache already covers.
    let mut cache = LeafCache::default();
    GraphField::compile_cached(&doc, &graph, 0.25, &mut cache).expect("compiles");
    let t = std::time::Instant::now();
    GraphField::compile_cached(&doc, &graph, 0.25, &mut cache).expect("compiles");
    println!("compile (cached leaves)   {:>8.1} ms", ms(t));

    // 2. Generating the shader, and the naga parse that now validates it.
    let t = std::time::Instant::now();
    let program = codegen::generate(&doc, &graph).expect("generates");
    println!(
        "codegen + validate        {:>8.1} ms   ({} chars of WGSL)",
        ms(t),
        program.source.len(),
    );

    // 3. Everything the renderer does: the driver's shader compile, every leaf baked, and the
    //    atlas written. This is the step a hide pays in full.
    let fields: Vec<(&(dyn ScalarField + Sync), (Point3<f64>, Point3<f64>))> = leaves
        .iter()
        .zip(doc.meshes.values())
        .map(|(f, m)| {
            let (lo, hi) = m.bounds().expect("a closed mesh is bounded");
            (
                f as &(dyn ScalarField + Sync),
                (Point3::new(lo[0], lo[1], lo[2]), Point3::new(hi[0], hi[1], hi[2])),
            )
        })
        .collect();

    // The device is created once when the application starts, so it is set up here and left
    // out of the timing — measuring it would put four seconds of driver initialisation into a
    // number that is supposed to describe one click.
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("an adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("a device")
    });

    // Baking, which is what a rebuild used to redo in full and what the cache now keeps.
    let resolution = fieldform_gpu::bake_resolution_limit(&device, fields.len(), 64);
    let t = std::time::Instant::now();
    let bakes: Vec<std::sync::Arc<fieldform_gpu::BakedField>> = fields
        .iter()
        .map(|(f, b)| std::sync::Arc::new(fieldform_gpu::BakedField::new(*f, *b, resolution, 6.0)))
        .collect();
    println!("baking all {BODIES} bodies      {:>8.1} ms", ms(t));

    // And what is left once they are reused: the shader the driver compiles, and the atlas
    // write. That is what a show/hide costs now.
    for pass in 1..=2 {
        let t = std::time::Instant::now();
        match Renderer::with_device(device.clone(), queue.clone(), &program, &bakes, &[], &[]) {
            Ok(_) => println!("shader + atlas, bakes kept {:>7.1} ms   (pass {pass})", ms(t)),
            Err(e) => println!("renderer: {e}"),
        }
    }

    println!(
        "\nThe bake was the wait, and it is the one thing a show/hide does not change: a body that
         stayed visible has the same geometry, the same box and the same resolution. What is
         left is the shader, which genuinely is a different shader."
    );
}

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// An assembly of small closed solids, spread out so none of them overlaps.
fn assembly(count: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    for i in 0..count {
        let at = Vector3::new((i % 6) as f64 * 60.0, (i / 6) as f64 * 60.0, 0.0);
        let (positions, triangles) = drum(20.0, 24.0, SIDES, at);
        let id = doc.add_mesh(MeshData {
            source: None,
            edges: Vec::new(),
            name: format!("part {i}"),
            positions: positions.iter().map(|p| [p.x, p.y, p.z]).collect(),
            triangles,
            tolerance_mm: Some(0.05),
            open_edges: 0,
        });
        graph.add(Node::Mesh(id));
    }
    let _ = ParamRef::Literal(0.0);
    (doc, graph)
}

/// A closed cylinder, which is the shape most small machined parts approximate.
fn drum(
    radius: f64,
    height: f64,
    sides: usize,
    at: Vector3<f64>,
) -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
    let h = height * 0.5;
    let mut positions =
        vec![Point3::from(at + Vector3::new(0.0, 0.0, -h)), Point3::from(at + Vector3::new(0.0, 0.0, h))];
    for i in 0..sides {
        let a = std::f64::consts::TAU * i as f64 / sides as f64;
        let (x, y) = (radius * a.cos(), radius * a.sin());
        positions.push(Point3::from(at + Vector3::new(x, y, -h)));
        positions.push(Point3::from(at + Vector3::new(x, y, h)));
    }
    let lo = |i: usize| (2 + 2 * (i % sides)) as u32;
    let hi = |i: usize| (3 + 2 * (i % sides)) as u32;
    let mut triangles = Vec::new();
    for i in 0..sides {
        triangles.push([lo(i), lo(i + 1), hi(i + 1)]);
        triangles.push([lo(i), hi(i + 1), hi(i)]);
        triangles.push([0, lo(i + 1), lo(i)]);
        triangles.push([1, hi(i), hi(i + 1)]);
    }
    (positions, triangles)
}
