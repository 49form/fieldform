// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the driver charges to turn a part's shader into a pipeline, and how that grows.
//!
//! **Written from a freeze.** The owner mirrored a swept wing, included the mirror in a blend
//! with a fuselage, and the application stopped for seconds at a time; changing a boolean to a
//! union, and then to G1, did it again. Its own slow-frame line named the stage every time —
//! *"settled: the driver's pipeline and uploads 30453 ms"* — which is not the bake, not the
//! compile, and not an upload: it is `create_render_pipeline`, on the interface thread.
//!
//! This measures it on a real document: how long the driver takes, and how much source it was
//! handed.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example pipeline_cost -- <file.fform>`

fn main() {
    let path = std::env::args().nth(1).expect("a .fform file");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let part = doc.parts_ordered().first().copied().expect("a part");
    let graph = &part.graph;
    println!("{}: {} nodes, {} bodies", path, graph.nodes().count(), graph.roots().len());

    let began = std::time::Instant::now();
    let program = fieldform_gpu::generate(&doc, graph).expect("the part generates a shader");
    let generated = began.elapsed().as_secs_f64() * 1e3;
    let lines = program.source.lines().count();
    println!(
        "generated {:.1} KB of WGSL in {generated:.0} ms: {lines} lines, {} curves, {} baked \
         leaves, {} parameters",
        program.source.len() as f64 / 1024.0,
        program.curves.len(),
        program.baked.len(),
        program.params.len(),
    );

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
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: fieldform_gpu::required_limits(&adapter),
                ..Default::default()
            })
            .await
            .expect("a device")
    });
    println!("{}\n", device.adapter_info().name);

    // The bakes the shader wants, so the renderer is built with what it really has. Their own
    // cost is the bake, measured elsewhere; here they are only the atlas the pipeline needs.
    let bakes: Vec<std::sync::Arc<fieldform_gpu::BakedField>> = Vec::new();
    if !program.baked.is_empty() {
        println!(
            "note: this part has {} baked leaf(s); they are left out, so the number below is the \
             pipeline alone",
            program.baked.len()
        );
    }

    for round in 1..=3 {
        let t = std::time::Instant::now();
        let built = fieldform_gpu::Renderer::with_device(
            device.clone(),
            queue.clone(),
            &program,
            &bakes,
            &[],
            &[],
        );
        let ms = t.elapsed().as_secs_f64() * 1e3;
        match built {
            Ok(_) => println!("round {round}: the driver took {ms:>8.0} ms"),
            Err(e) => println!("round {round}: refused after {ms:.0} ms: {e}"),
        }
    }
}
