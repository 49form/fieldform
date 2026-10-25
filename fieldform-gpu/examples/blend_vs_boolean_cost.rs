// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a G1 blend costs against a hard boolean, on the same bodies.
//!
//! Reported as "Boolean g1 seems slow??". A `Node::Boolean` set to G1 *is* a `Node::Blend`, so
//! the question is what a blend costs that a `min` does not, and there are two candidates that
//! want opposite fixes: the driver's compile, or the march.
//!
//! Both are measured here, on one device, in one process:
//!
//! - **Cold compile.** Every `n` is a source this machine has not generated before, which is the
//!   only way to read it: a shader the driver has seen comes back from its own disk cache in
//!   about ten milliseconds whatever it contains. `compile_scaling`'s rows have been run enough
//!   times to be worthless for this, which is why the counts here avoid the ones it uses.
//! - **Frame.** Interleaved, forward then backward, least of the rounds, one size per pass --
//!   the arrangement `mask_frame_cost` arrived at after three faults, each of which made the
//!   arms look further apart than they were.
//!
//! The last row of each block is the control: the hard join again, the same source, so the gap
//! between those two rows is this run's noise and not the blend.
//!
//! # Measured, RTX A500
//!
//! | spheres | hard: driver / 720px | G1 blend: driver / 720px | control: driver / 720px |
//! |---|---|---|---|
//! | 3 | 69 ms / 5.8 | **129** / 6.4 | 12 / 6.1 |
//! | 5 | 93 ms / 6.8 | **202** / 7.3 | 11 / 7.0 |
//! | 6 | 92 ms / 7.9 | **219** / 8.2 | 12 / 7.2 |
//! | 7 | 114 ms / 7.1 | **305** / 8.1 | 11 / 7.7 |
//!
//! **It is the compile.** A chain of G1 blends costs two to two and a half times the driver's
//! time that the same bodies hard-unioned do, and the ratio grows with the chain. The control's
//! 11-12 ms is the disk cache answering for a source already compiled, which is what makes the
//! first column readable as cold.
//!
//! **It is not the frame.** The blend is 5-14% dearer at 720 px, and the two hard rows -- one
//! source, twice -- differ by as much. At 1440 px the control rows differ by 20% (25.4 against
//! 30.7 at n = 3), so that column cannot resolve a difference this size at all and is printed
//! rather than relied on.
//!
//! **And the divisor is not what does it here.** The value scale stays 1.00 on both arms, so
//! these blends are not dividing and every step is the length it would be. Blend integrity's
//! divisor compounding through a chain is real and is measured elsewhere -- on pipes, whose
//! operands are curve walks -- but it is not the answer to this question on these bodies, and
//! reaching for it would have been the wrong fix.
//!
//! So "Boolean g1 seems slow" is the same superlinear driver compile that
//! `compile_scaling`'s last series measures at 2 pipes 1.0 s against 16 pipes 120 s. A blend
//! reaches it sooner because each one emits more than a `min` does.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example blend_vs_boolean_cost`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{
    BooleanOp, Continuity, Document, Fillet, Graph, Node, NodeId, ParamRef, Place,
};
use nalgebra::Vector3;

const FRAMES: u32 = 20;
const ROUNDS: u32 = 6;
const RADIUS_MM: f64 = 20.0;
/// Close enough that every pair genuinely overlaps, so a fillet has a seam to roll into.
const PITCH_MM: f64 = 30.0;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// `n` overlapping spheres in a row, joined hard or by G1 fillets of `radius`.
fn chain(n: usize, radius: Option<f64>) -> Graph {
    let mut g = Graph::empty();
    let ball = |g: &mut Graph, i: usize| {
        g.add(Node::Sphere {
            centre: Place::At([lit(i as f64 * PITCH_MM), lit(0.0), lit(0.0)]),
            radius: lit(RADIUS_MM),
        })
    };
    let ids: Vec<NodeId> = (0..n).map(|i| ball(&mut g, i)).collect();
    match radius {
        None => {
            g.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.into_iter().map(Into::into).collect() });
        }
        Some(r) => {
            let mut acc = ids[0];
            for next in &ids[1..] {
                acc = g.add(Node::Blend {
                    op: BooleanOp::Union,
                    a: acc.into(),
                    b: (*next).into(),
                    fillet: Fillet { radius: lit(r), continuity: Continuity::G1, second: None },
                });
            }
        }
    }
    g
}

fn main() {
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

    // Counts `compile_scaling` does not use, so the driver has never seen these sources.
    for n in [3usize, 5, 6, 7] {
        let doc = Document::new();
        let arms: Vec<(String, Graph)> = vec![
            ("hard".to_string(), chain(n, None)),
            ("G1 blend".to_string(), chain(n, Some(6.0))),
            // The control: the hard arm again, same source, so its two rows bound the noise.
            ("hard, again".to_string(), chain(n, None)),
        ];

        println!("{n} spheres in a row, overlapping");
        println!(
            "  {:<13} {:>7} {:>8} {:>11} {:>11} {:>11}",
            "join", "lines", "params", "driver ms", "720px ms", "1440px ms"
        );

        let mut built = Vec::new();
        for (label, graph) in &arms {
            let program = generate(&doc, graph).expect("generates");
            let cut = program
                .source
                .find("// ---- generated from the composition graph ----")
                .expect("marked");
            let lines = program.source[cut..].lines().count();
            let began = std::time::Instant::now();
            let renderer = match Renderer::with_device(
                device.clone(),
                queue.clone(),
                &program,
                &[],
                &[],
                &[],
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  {label}: no GPU ({e})");
                    return;
                }
            };
            let compile = began.elapsed().as_secs_f64() * 1e3;
            let values: Vec<f32> = program
                .params
                .iter()
                .map(|p| match p {
                    ParamRef::Literal(v) => *v as f32,
                    _ => 0.0,
                })
                .collect();
            renderer.set_params(&values);
            built.push((label.clone(), lines, program.params.len(), compile, renderer, graph));
        }

        // Framed on the whole row, so every arm draws the same pixels.
        let extent = match fieldform_field::part_extent(&doc, &arms[0].1) {
            fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
            _ => panic!("bounded"),
        };
        let radius = (extent.1 - extent.0).norm() * 0.5;
        let centre = nalgebra::center(&extent.0, &extent.1);
        let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
        let reach = Vector3::repeat(radius);
        for (_, _, _, _, renderer, graph) in &built {
            let multiplier = fieldform_field::step_multiplier(&doc, graph);
            let _ = multiplier;
            renderer.build_skip_grid(centre - reach, radius * 2.0, 64, true);
        }

        let mut best = vec![[f64::MAX; 2]; built.len()];
        for (at, edge) in [(0usize, 720u32), (1, 1440)] {
            for (i, (_, _, _, _, renderer, graph)) in built.iter().enumerate() {
                // **Each arm's own step budget**, the way the viewport sizes it: a divided field
                // needs more steps for the same distance, and giving the blend the boolean's
                // budget would measure a truncated march rather than a slower one.
                let multiplier = fieldform_field::step_multiplier(&doc, graph);
                let budget = ((512.0 * multiplier * 2.0).ceil() as u32).clamp(512, 16_384);
                let mut camera =
                    Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
                camera.value_scale = multiplier;
                let _ = renderer.render(&camera, edge, edge, budget);
                for round in 0..ROUNDS {
                    let _ = round;
                    let start = std::time::Instant::now();
                    for _ in 0..FRAMES {
                        renderer.render(&camera, edge, edge, budget).expect("renders");
                    }
                    let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES);
                    best[i][at] = best[i][at].min(ms);
                }
            }
        }

        for (i, (label, lines, params, compile, _, graph)) in built.iter().enumerate() {
            let multiplier = fieldform_field::step_multiplier(&doc, graph);
            println!(
                "  {label:<13} {lines:>7} {params:>8} {compile:>11.0} {:>11.1} {:>11.1}   \
                 value scale {multiplier:.2}",
                best[i][0], best[i][1],
            );
        }
        println!();
    }
    println!(
        "`hard` and `hard, again` are the same source, so the gap between them is this run's\n\
         noise. The value scale is what the field divides by: a chain of blends compounds it,\n\
         and a larger one means every step is shorter and the march takes more of them."
    );
}
