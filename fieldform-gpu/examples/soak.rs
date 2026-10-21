// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Does the lighting path's GPU memory plateau, or climb?
//!
//! Written after the application hung with egui unable to allocate a 24 KB staging buffer: the
//! device was out of memory or lost, and nothing in the logs said which side had eaten it. This
//! does what a user does for minutes — drags at the interactive size, stills at the full size
//! with occlusion and eight jittered frames, material and environment switches, a selection
//! outline — thousands of times, and reads wgpu's own allocator as it goes. A leak is a number
//! that keeps climbing cycle after cycle; a healthy path plateaus within the first hundred.
//!
//! Two numbers are read, because they answer different questions. wgpu's allocator report is
//! what *this crate's* buffers and textures occupy, live, on the device. `nvidia-smi` is what the
//! driver holds for the whole process — pipelines, descriptor pools, staging, everything — so a
//! climb there without one in the allocator points below wgpu, and a climb in both points here.
//!
//! Run with:
//! `cargo run -p fieldform-gpu --example soak [cycles]`

use fieldform_gpu::{
    generate, halton_jitter, Camera, Draft, EnvironmentChoice, Look, Occlusion, Renderer,
    ShadingResources,
};
use fieldform_scene::{BooleanOp, Document, Graph, LatticePhase, Material, Node, ParamRef};
use nalgebra::{Point3, Vector3};

const SIZE_MM: f64 = 100.0;
/// The viewport a still is traced at, and the drag frame's cap, as the application has them.
const STILL: (u32, u32) = (1920, 1080);
const DRAG: (u32, u32) = (720, 405);

fn gpu_memory_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().lines().next()?.trim().parse().ok()
}

fn main() {
    let cycles: u32 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(400);

    let doc = Document::new();
    let mut graph = Graph::empty();
    let cube = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
        size: [ParamRef::Literal(SIZE_MM * 0.7); 3],
        turn: fieldform_scene::no_turn(),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: fieldform_scene::LatticeCell::Gyroid,
        map: fieldform_scene::CellMap::Rectangular { cell: ParamRef::Literal(SIZE_MM * 0.12) },
        phase: LatticePhase::sheet(ParamRef::Literal(SIZE_MM * 0.012)),
    });
    graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![cube.into(), lattice.into()] });
    graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([ParamRef::Literal(-SIZE_MM * 0.5), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
        radius: ParamRef::Literal(SIZE_MM * 0.15),
    });
    let program = generate(&doc, &graph).expect("the graph generates");

    let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
            return;
        }
    };
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| match p {
            ParamRef::Literal(v) => *v as f32,
            _ => unreachable!("literals only"),
        })
        .collect();
    renderer.set_params(&values);

    let environments: Vec<_> = EnvironmentChoice::ALL.iter().map(|c| (*c, c.bake())).collect();
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &environments[0].1)
        .expect("the lighting pipeline builds");
    let presets = Material::presets();

    let half = SIZE_MM * 0.5;
    let bounds = (Point3::new(-half, -half, -half), Point3::new(half, half, half));
    let report = |device: &wgpu::Device| -> (f64, f64, usize, usize) {
        match device.generate_allocator_report() {
            Some(r) => (
                r.total_allocated_bytes as f64 / 1e6,
                r.total_reserved_bytes as f64 / 1e6,
                r.allocations.len(),
                r.blocks.len(),
            ),
            None => (0.0, 0.0, 0, 0),
        }
    };

    println!("| cycle | frames | wgpu allocated | wgpu reserved | allocations | blocks | driver (nvidia-smi) |");
    println!("|---|---|---|---|---|---|---|");
    let mut frames = 0u64;
    let mut peak_reserved = 0.0f64;
    let mut environment = 0usize;
    for cycle in 0..cycles {
        // What a user does: a few frames of drag, then a still that converges.
        let material = Material::from(presets[(cycle as usize / 7) % presets.len()].material);
        if cycle % 25 == 0 && cycle > 0 {
            environment = (environment + 1) % environments.len();
            shading.set_environment(renderer.device(), renderer.queue(), &environments[environment].1);
        }
        let choice = environments[environment].0;

        let mut camera = Camera::framing(bounds, DRAG.0 as f64 / DRAG.1 as f64);
        let mut look = Look::for_environment(choice, material.clone(), camera.forward, camera.right, camera.up);
        look.selected = if cycle % 3 == 0 { Some(0) } else { None };
        look.outline_radius = 1;
        look.draft = if cycle % 5 == 0 { Some(Draft::new(Vector3::z(), 3.0, 1.0)) } else { None };
        for _ in 0..3 {
            renderer
                .present(&mut shading, &camera, DRAG.0, DRAG.1, 512, &look, 0)
                .expect("the drag frame presents");
            frames += 1;
        }

        let mut camera_still = Camera::framing(bounds, STILL.0 as f64 / STILL.1 as f64);
        look.outline_radius = 2;
        camera_still.occlusion = Some(Occlusion::for_scene(look.lights[0].direction, half * 3f64.sqrt()));
        for frame in 0..8u32 {
            camera_still.jitter = halton_jitter(frame);
            renderer
                .present(&mut shading, &camera_still, STILL.0, STILL.1, 512, &look, frame)
                .expect("the still frame presents");
            frames += 1;
        }
        // The swapchain paces the application; here nothing does, so wait for the GPU each
        // cycle or the queue runs ahead by thousands of frames and the number measured is the
        // backlog rather than the steady state.
        renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("the GPU finishes");
        camera.jitter = [0.0, 0.0];

        let (allocated, reserved, allocations, blocks) = report(renderer.device());
        peak_reserved = peak_reserved.max(reserved);
        if cycle % 25 == 0 || cycle + 1 == cycles {
            let driver = gpu_memory_used_mib()
                .map(|m| format!("{m} MiB"))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "| {cycle:>5} | {frames:>6} | {allocated:>9.1} MB | {reserved:>9.1} MB | {allocations:>6} | {blocks:>3} | {driver} |"
            );
        }
    }
    let (allocated, reserved, _, _) = report(renderer.device());
    println!(
        "{frames} frames over {cycles} cycles; wgpu reserved {reserved:.1} MB at the end against a \
         peak of {peak_reserved:.1} MB, {allocated:.1} MB of it in live allocations."
    );
}
