// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a lit frame costs, per tier, with the readback out of the way.
//!
//! The viewport used to cap every frame at 720 pixels because the image was read back to the
//! CPU, and `traced_cost` measured that copy as the whole cost. This measures what is left once
//! nothing is read back: the march and the lighting pass, timed by waiting for the GPU to finish
//! — on the cheapest field there is and on the dearest, at the interactive size and at the two
//! monitor sizes a still is traced at, with and without the occlusion and shadow marches a still
//! turns on.
//!
//! Which adapter served it matters, because on this machine there are two and the numbers differ
//! by the gap between them (shading-spec.md §9.1). `Renderer::new` asks for the high-performance
//! one, which is the discrete card — the same one the application logs at startup.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example shading_cost`

use fieldform_gpu::{
    generate, BakeSettings, Camera, EnvironmentMaps, Look, Occlusion, Renderer, ShadingResources,
    Studio,
};
use fieldform_scene::{BooleanOp, Document, Graph, LatticePhase, Material, Node, ParamRef};
use nalgebra::Point3;

const SIZE_MM: f64 = 100.0;

/// Rows per band in the banded column. Coarse enough that a 1080-row frame is nine
/// submissions, which is about what a slow card would be given at its budget.
const BAND_ROWS: u32 = 128;

fn sphere() -> Graph {
    let mut graph = Graph::empty();
    graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
        radius: ParamRef::Literal(SIZE_MM * 0.4),
    });
    graph
}

fn gyroid_cube() -> Graph {
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
    graph
}

fn main() {
    let doc = Document::new();
    let scenes = [("sphere", sphere()), ("cube infilled with a gyroid", gyroid_cube())];
    let sizes = [(720u32, 720u32), (1280, 720), (1920, 1080)];

    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    // One device for everything, as the application has: the lighting resources are built on it
    // once and every scene's renderer shares it.
    let mut shared: Option<(Renderer, ShadingResources)> = None;

    // Two clocks per cell: the wall time from submit to the GPU finishing, which is how this
    // table was first measured, and beside it the GPU's own march + shade timestamps where the
    // device writes them -- the number the application's `perf` verb reports, so the two can
    // be read against each other.
    // And the still once more in bands of rows, each its own submission, which is how the
    // application traces one (loop spec S2). The work is the same; only the packets are
    // smaller, so the column should read within noise of the one before it -- and if it does
    // not, the per-submission overhead is the number to look at.
    println!(
        "| scene | frame | drag tier (no occlusion) | still tier (occlusion + shadow) | \
         still in {BAND_ROWS}-row bands |"
    );
    println!("|---|---|---|---|---|");
    println!("| | | wall / gpu (march + shade) | wall / gpu (march + shade) | wall, bands |");
    for (label, graph) in &scenes {
        let program = generate(&doc, graph).expect("the graph generates");
        let renderer = match &shared {
            None => match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
                    return;
                }
            },
            Some((first, _)) => Renderer::with_device(
                first.device().clone(),
                first.queue().clone(),
                &program,
                &[],
                &[],
                &[],
            )
            .expect("the renderer builds on the shared device"),
        };
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| match p {
                ParamRef::Literal(v) => *v as f32,
                _ => unreachable!("these scenes use literals"),
            })
            .collect();
        renderer.set_params(&values);
        if shared.is_none() {
            let shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
                .expect("the lighting pipeline builds");
            shared = Some((
                Renderer::with_device(
                    renderer.device().clone(),
                    renderer.queue().clone(),
                    &program,
                    &[],
                    &[],
                    &[],
                )
                .expect("the renderer builds"),
                shading,
            ));
        }
        let shading = &mut shared.as_mut().expect("set above").1;

        let half = SIZE_MM * 0.5;
        let bounds = (Point3::new(-half, -half, -half), Point3::new(half, half, half));
        for (width, height) in sizes {
            let mut camera = Camera::framing(bounds, width as f64 / height as f64);
            let look = Look::studio(Material::default(), camera.forward, camera.right, camera.up);
            let mut row = format!("| {label} | {width}×{height} |");
            for still in [false, true] {
                camera.occlusion = still.then(|| {
                    Occlusion::for_scene(look.lights[0].direction, SIZE_MM * 0.5 * 3f64.sqrt())
                });
                // One warm-up so pipeline and allocation costs are not in the number, then the
                // median of several, since a laptop GPU's clock wanders.
                let mut samples = Vec::new();
                let mut gpu = Vec::new();
                for i in 0..6 {
                    let started = std::time::Instant::now();
                    let ticket = renderer
                        .present(shading, &camera, width, height, 512, &look, 0)
                        .expect("the frame presents");
                    renderer
                        .device()
                        .poll(wgpu::PollType::wait_indefinitely())
                        .expect("the GPU finishes");
                    if i > 0 {
                        samples.push(started.elapsed().as_secs_f64() * 1e3);
                    }
                    for (done, cost) in shading.costs() {
                        if done == ticket && i > 0 {
                            gpu.push(cost);
                        }
                    }
                }
                samples.sort_by(f64::total_cmp);
                gpu.sort_by(|a, b| a.ms().total_cmp(&b.ms()));
                let gpu = gpu.get(gpu.len() / 2).and_then(|c| Some((c.march_ms?, c.shade_ms?)));
                row.push_str(&format!(
                    " {:.1} ms / {} |",
                    samples[samples.len() / 2],
                    match gpu {
                        Some((march, shade)) =>
                            format!("{:.1} ms ({march:.1} + {shade:.1})", march + shade),
                        None => "no timestamps".to_string(),
                    }
                ));
            }
            // The still again, banded: the whole frame's wall time across every band.
            let mut samples = Vec::new();
            for i in 0..6 {
                let started = std::time::Instant::now();
                let mut first = 0;
                while first < height {
                    let last = (first + BAND_ROWS).min(height);
                    renderer
                        .present_rows(shading, &camera, width, height, 512, &look, 0, first..last)
                        .expect("the band presents");
                    first = last;
                }
                renderer
                    .device()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("the GPU finishes");
                if i > 0 {
                    samples.push(started.elapsed().as_secs_f64() * 1e3);
                }
                // Drained so the timing ring never fills: a band that finds no free slot is
                // simply not measured, but the point here is the wall time anyway.
                let _ = shading.costs();
            }
            samples.sort_by(f64::total_cmp);
            row.push_str(&format!(
                " {:.1} ms, {} |",
                samples[samples.len() / 2],
                height.div_ceil(BAND_ROWS)
            ));
            println!("{row}");
        }
    }
}
