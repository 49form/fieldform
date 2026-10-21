// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What one step of an edit costs the interface's thread, part by part.
//!
//! Plan §5 V2. Every change to a document goes through the viewport's rebuild: the graph is
//! generated into WGSL, the curves are re-sampled, and a renderer is built around the result.
//! During a drag that happens once a frame, on the thread the interface runs on, so whatever it
//! costs comes straight out of the frame rate — before the card has marched a single ray.
//!
//! This times the parts on a real document, the way the application runs them, so that what is
//! cut is what costs. The pipeline figure is the *warm* one: the same source the driver has
//! already compiled, which is what a drag step is once no number that moves is written into the
//! shader (`bending_a_sweeps_spine_does_not_change_the_shader`).
//!
//! `cargo run --release -p fieldform-io --example edit_step_cost -- <file.fieldform>…`

use std::time::Instant;

use fieldform_gpu::{generate, CurveUse, Renderer};
use fieldform_scene::{Document, Graph};
use nalgebra::Point3;

const ROUNDS: usize = 30;

fn median(mut ms: Vec<f64>) -> f64 {
    ms.sort_by(f64::total_cmp);
    ms[ms.len() / 2]
}

fn timed<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut samples = Vec::with_capacity(ROUNDS);
    let mut last = None;
    for _ in 0..ROUNDS {
        let began = Instant::now();
        let out = f();
        samples.push(began.elapsed().as_secs_f64() * 1e3);
        last = Some(out);
    }
    (median(samples), last.expect("at least one round"))
}

type Curves = Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)>;

/// The curves as this check can supply them: spines only. `None` for a graph wanting more.
fn spines(doc: &Document, program: &fieldform_gpu::ShaderProgram, floor: f64) -> Option<Curves> {
    program
        .curves
        .iter()
        .map(|use_of| match use_of {
            CurveUse::Spine(id) => Some((doc.curves[id].tessellate_to_angle(1.0, floor), None)),
            _ => None,
        })
        .collect()
}

fn measure(label: &str, doc: &Document, graph: &Graph, shared: &mut Option<Renderer>) {
    let (generate_ms, program) = timed(|| generate(doc, graph).expect("the graph generates"));
    let floor = fieldform_field::part_extent(doc, graph)
        .as_box()
        .map_or(1e-4, |(lo, hi)| ((hi - lo).norm() * 1e-5).max(1e-4));
    let (curves_ms, curves) = timed(|| spines(doc, &program, floor));
    let Some(curves) = curves else {
        println!("| {label} | {generate_ms:.1} | — | — | — | wants more than spines, which is all this supplies |");
        return;
    };
    if shared.is_none() {
        *shared = Some(
            pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0))
                .expect("a GPU, which is where a renderer is built"),
        );
    }
    let (device, queue) = {
        let first = shared.as_ref().expect("set above");
        (first.device().clone(), first.queue().clone())
    };
    // A source the driver has never seen, and cannot have: the same program with the time
    // written into it. Without that this column reads the driver's cache on
    // every run but the first, which is how a compile of seconds came to read as 7 ms.
    let mut never_seen = generate(doc, graph).expect("the graph generates");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    // In code the march reaches, because nothing less gets past the driver's cache: a comment
    // is gone before the driver sees the module, and a function nothing calls is gone before
    // the cache looks. The march's step margin, moved in its seventh place, is a different
    // program to a compiler and the same one to anybody else -- and this renderer draws nothing.
    let margin = format!("const STEP_SAFETY: f32 = 0.9{:06};", nonce % 1_000_000);
    assert!(never_seen.source.contains("const STEP_SAFETY: f32 = 0.9;"));
    never_seen.source = never_seen.source.replace("const STEP_SAFETY: f32 = 0.9;", &margin);
    let cold = Instant::now();
    let _ = Renderer::with_device(device.clone(), queue.clone(), &never_seen, &[], &curves, &[])
        .expect("the renderer builds");
    let cold_ms = cold.elapsed().as_secs_f64() * 1e3;
    let kept = Renderer::with_device(device.clone(), queue.clone(), &program, &[], &curves, &[])
        .expect("the renderer builds");
    let (fresh_ms, _) = timed(|| {
        Renderer::with_device(device.clone(), queue.clone(), &program, &[], &curves, &[])
            .expect("the renderer builds")
    });
    let (reused_ms, reused) = timed(|| {
        Renderer::rebuilt_from(&kept, &program, &[], &curves, &[]).expect("the renderer builds")
    });
    // A kept pipeline draws what a compiled one draws, to the bit: it is the same pipeline.
    if let Some((lo, hi)) = fieldform_field::part_extent(doc, graph).as_box() {
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
            .collect();
        let camera = fieldform_gpu::Camera::framing((lo, hi), 1.5);
        kept.set_params(&values);
        reused.set_params(&values);
        let a = kept.render(&camera, 240, 160, 1024).expect("renders");
        let b = reused.render(&camera, 240, 160, 1024).expect("renders");
        assert!(a == b, "the kept pipeline drew a different frame");
    }
    println!(
        "| {label} | {generate_ms:.1} | {curves_ms:.1} | {cold_ms:.0} | {fresh_ms:.1} | {reused_ms:.1} |"
    );
}

fn main() {
    println!("Medians of {ROUNDS}, milliseconds of the interface's thread.\n");
    println!("| document | generate | re-sample the spines | renderer, source never seen | renderer, source seen | renderer, pipeline kept |");
    println!("|---|---|---|---|---|---|");
    let mut shared = None;
    for path in std::env::args().skip(1) {
        match fieldform_io::open(std::path::Path::new(&path)) {
            Ok((doc, _)) => {
                let name = std::path::Path::new(&path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or(path.clone());
                let graphs: Vec<Graph> = doc
                    .parts
                    .values()
                    .filter(|p| !p.graph.roots().is_empty())
                    .map(|p| p.graph.clone())
                    .collect();
                for graph in graphs {
                    measure(&format!("`{name}`"), &doc, &graph, &mut shared);
                }
            }
            Err(e) => println!("could not open {path}: {e}"),
        }
    }
}
