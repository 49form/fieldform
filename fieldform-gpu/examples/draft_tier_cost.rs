// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the draft tier would save by not compiling the two marches it never runs.
//!
//! **The prize, measured before the mechanism is built.** `union-compile-8fa06b7.md` found that
//! `march` reads the field at five places and `field_normal` once, so every operand of a
//! composition is written into one entry point six times over, and that the driver's compile is a
//! power of about 2 to 2.5 in the result. Two of those six are the occlusion's five samples and the
//! shadow's walk, and `march.wgsl` guards both with a **uniform**:
//!
//! ```wgsl
//! if camera.key_light.w > 0.5 {
//!     open = occlusion(p, normal, camera.occlusion.x);
//!     lit = shadow(p, normal, camera.key_light.xyz, camera.occlusion.y, camera.occlusion.z);
//! }
//! ```
//!
//! A uniform, not a codegen switch — so the **draft** tier, which by the display rules has no
//! occlusion and no shadow, still waits for the driver to compile both. Four copies of six at a
//! power of 2.3 predicts about 0.40 of the compile. That is an estimate from a fitted exponent, and
//! this is the reading.
//!
//! **How, without building anything.** The two calls are replaced by `1.0` in the generated source.
//! That removes the only references to `occlusion` and `shadow` from `march`, so the driver stops
//! inlining them — which is exactly what a draft-only entry point would achieve, without the
//! plumbing to have two of them. If the saving is real the plumbing is worth building; if it is not,
//! this file is the reason not to.
//!
//! **The control is the frame.** The draft never took that branch, so a draft frame must read the
//! same either way. A draft that got *faster* would mean the branch was costing something at run
//! time too, and a draft that got slower would mean this measurement is not measuring what it says.
//!
//! **Measured on an RTX A500 Laptop GPU, Vulkan, 2026-09-26.** Cold pipeline, and a 720 px draft as
//! the least of six rounds of twenty frames:
//!
//! | shape | compile: with both → draft only | 720 px |
//! |---|---|---|
//! | one pipe | 334 → **162** (0.49×) | 6.5 → 6.5 |
//! | three pipes, unioned (below `RUN_FLOOR`, so written out) | 2 576 → **973** (0.38×) | 10.1 → 10.3 |
//! | eight pipes, looped | 599 → **324** (0.54×) | 19.3 → 18.6 |
//! | one sphere | 107 → **89** (0.83×) | 6.6 → 6.6 |
//!
//! **So about half the compile, on anything that walks** — 0.38 to 0.54, against the 0.40 the power
//! law predicted. A sphere saves little (0.83), which is the right shape of answer: the two extra
//! copies cost in proportion to what a copy contains, and a sphere's is 48 inlined lines against a
//! pipe's 1 835. The dearest row saves the most, which is the row a person waits through.
//!
//! **The control passed on the second reading, and the first one is worth recording.** The
//! eight-pipe row first read 25.1 ms against 18.6 — 22%, where the two arms must be identical
//! because the draft never takes the branch that was removed. Re-run, the same arm read 19.3, so
//! the 25.1 was a slow first touch and not an effect: this machine has shown that scatter before
//! (`union-compile-8fa06b7.md` records a 2× swing on an unchanged row). Had it reproduced, the
//! reading to make would have been register pressure — two fewer copies live in `march` is better
//! occupancy — and that would have been a second, separate finding rather than a fault. It did not
//! reproduce, so the honest statement is the plain one: **the draft frame is unchanged, and only
//! the compile moves.**
//!
//! ## As built, which is a smaller saving than the simulation promised
//!
//! Two entry points sharing `trace_primary`, the still's pipeline made on the first frame whose
//! camera carries an `Occlusion`. Cold, on counts nothing else in the repo had compiled since the
//! split:
//!
//! | shape | module | draft pipeline | still pipeline | first picture now |
//! |---|---|---|---|---|
//! | two pipes, unioned | 8 | **366** | 476 | **0.77×** |
//! | five pipes, looped | 6 | **202** | 251 | **0.81×** |
//! | twelve pipes, looped | 6 | **200** | 276 | **0.73×** |
//!
//! **0.73 to 0.81, against the 0.38 to 0.54 the simulation measured.** Both are cold and both are
//! like-for-like within themselves — the ratio here is module + draft against module + still, and
//! `march_still`, the still's entry point as it then was, was exactly the old shape: one entry point
//! carrying everything. So the gap is real and worth stating rather than quoting the flattering
//! number.
//!
//! **What it looks like, unproven.** The simulation's baseline was the *pre-refactor* monolith, and
//! that monolith appears to have been dearer than `march_still` as it stood then — one pipe read
//! 334 ms before and about 230 ms after, for the same work. If that holds, giving the driver a function boundary
//! at `trace_primary` made the combined pipeline cheaper too, which lowers both sides of this ratio
//! while leaving the absolute first-picture figure roughly where the simulation put it. That is a
//! plausible reading of two single samples, not a measurement: the two numbers it rests on were
//! taken on different builds, and nothing here isolates the boundary's own effect. Treat **0.73 to
//! 0.81 as the claim** and the rest as the thing to measure if it matters.
//!
//! ## Since loop spec P5, the still's pipeline is its two marches alone
//!
//! The still's occlusion and shadow are an entry point of their own (`occlude`), dispatched after
//! the draft's march -- built so that they could be marched for one pixel in four, which is off,
//! and kept because the split pays at every pixel too. The still traces with the draft's
//! pipeline, and what it compiles on its first frame is the composition twice rather than eight
//! times. Cold, on new counts, 2026-10-13; the still's column includes one 720-pixel frame:
//!
//! | shape | module | draft pipeline | still pipeline | still / draft |
//! |---|---|---|---|---|
//! | three pipes, looped | 11 | 564 | 194 | **0.34×** |
//! | six pipes, looped | 11 | 194 | 90 | **0.46×** |
//! | nine pipes, looped | 10 | 205 | 93 | **0.45×** |
//!
//! Against 1.24 to 1.38 times the draft's in the table above. The first row's draft is the
//! process's first compile and reads high for it. That frame was at one pixel in four, which was
//! then how a still was drawn.
//!
//! Since one in four was turned off, the still's frame is at every pixel, and the pipeline carries
//! the fill's second dispatch besides (loop spec P5). Cold, on new counts, the same day:
//!
//! | shape | module | draft pipeline | still pipeline | still / draft |
//! |---|---|---|---|---|
//! | four pipes, looped | 11 | 208 | 98 | **0.47×** |
//! | seven pipes, looped | 13 | 201 | 119 | **0.59×** |
//! | ten pipes, looped | 12 | 211 | 118 | **0.56×** |
//!
//! Single cold readings, as all of these are, so the step from the table above is the frame's
//! occlusion marched at four times the pixels and the extra code together, and neither is
//! separated here.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example draft_tier_cost`

use fieldform_scene::{BooleanOp, Curve, CurveId, Document, Graph, Node, ParamRef};
use nalgebra::{Point3, Vector3};

const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
const SPINE_DEGREES: f64 = 1.0;

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
    let info = device.adapter_info();
    println!("{} on {:?}\n", info.name, info.backend);

    // **Simulated first, built after.** The arms that measured the prize replaced the two calls
    // with `1.0` in the generated source; that is no longer possible, because the split exists and
    // the still's entry point read `found.p` where the old body read `p`. Their assertion caught it
    // rather than letting a silently unmatched pattern make both arms identical and the saving
    // read as zero. The table above is that measurement, kept because it is what justified
    // building this; below is the mechanism measuring itself.
    println!("what the first picture costs, and what the still tier defers");
    println!(
        "  {:<26} {:>8} {:>15} {:>15} {:>14}",
        "shape", "module", "draft pipeline", "still pipeline", "still / draft"
    );
    // **Counts nothing else in this repo has compiled**, because a compile figure is only cold
    // once and every other harness here has been run. Two, five and twelve were cold for the
    // first table above and have been compiled since; three, six and nine for the second, and
    // four, seven and ten for the third. Change them if a row ever reads a handful of
    // milliseconds; that is the cache, not a saving.
    for (label, n) in [
        ("four pipes, looped", 4usize),
        ("seven pipes, looped", 7),
        ("ten pipes, looped", 10),
    ] {
        let (doc, graph) = spokes(n);
        let program = fieldform_gpu::generate(&doc, &graph).expect("generates");
        let (module_ms, draft_ms, still_ms) =
            first_picture(&device, &queue, &doc, &graph, &program);
        println!(
            "  {label:<26} {module_ms:>8.0} {draft_ms:>15.0} {still_ms:>15.0} {:>13.2}x",
            still_ms / draft_ms.max(1e-9)
        );
    }
    println!(
        "\n`module` is naga's parse and validation, paid once either way. `draft pipeline` is what \n\
         the first picture waits for; `still pipeline` is the still's occlusion and shadow, made \n\
         on the first frame whose camera carries an Occlusion. Before loop spec P5 the still's \n\
         pipeline carried the primary march as well, and cost 1.2 to 1.4 times the draft's."
    );
}

/// naga's front end, the draft pipeline, and the still's: what the first picture waits for, which
/// is the first two, and what the first still waits for beyond it -- the `occlude` pipeline's
/// compile with one 720-pixel still frame, timed round the first still render. That last is not a
/// compile alone, so the ratio of the two pipelines reads a little high, by about a frame.
fn first_picture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    doc: &Document,
    graph: &Graph,
    program: &fieldform_gpu::ShaderProgram,
) -> (f64, f64, f64) {
    let extent = match fieldform_field::part_extent(doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("unbounded"),
    };
    let size = (extent.1 - extent.0).norm();
    let centre = nalgebra::center(&extent.0, &extent.1);
    let radius = size * 0.5;
    let floor = (size * 1e-5).max(1e-4);
    let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
        .curves
        .iter()
        .map(|use_of| match *use_of {
            fieldform_gpu::CurveUse::Spine(id) => (
                doc.curves[&id].tessellate_to_angle(SPINE_DEGREES, floor),
                None,
            ),
            other => panic!("only spines here, not {other:?}"),
        })
        .collect();

    let renderer = fieldform_gpu::Renderer::with_device(
        device.clone(),
        queue.clone(),
        program,
        &[],
        &curves,
        &[],
    )
    .expect("builds");
    // From the renderer's own parts rather than a stopwatch round the call, which also packs the
    // buffers and the atlas -- see `BuildCost`.
    let cost = renderer.build_cost();
    let (module_ms, draft_ms) = (f64::from(cost.module_ms), f64::from(cost.pipelines_ms));

    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| match p {
            ParamRef::Literal(v) => *v as f32,
            _ => 0.0,
        })
        .collect();
    renderer.set_params(&values);
    let multiplier = fieldform_field::step_multiplier(doc, graph);
    let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
        .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);
    let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
    let mut camera =
        fieldform_gpu::Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    camera.value_scale = multiplier;
    // A draft first, so the still's figure is the still pipeline alone rather than it plus
    // whatever a first dispatch costs.
    let _ = renderer.render(&camera, 720, 720, budget);
    camera.occlusion = Some(fieldform_gpu::Occlusion::for_scene(
        Vector3::new(0.3, -0.5, 0.8).normalize(),
        radius * 3f64.sqrt(),
    ));
    let began = std::time::Instant::now();
    let _ = renderer.render(&camera, 720, 720, budget);
    let still_ms = began.elapsed().as_secs_f64() * 1e3;
    (module_ms, draft_ms, still_ms)
}

fn spokes(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let mut pipes = Vec::new();
    for i in 0..n {
        let turn = std::f64::consts::TAU * i as f64 / n as f64;
        let (c, s) = (turn.cos(), turn.sin());
        let id = CurveId::new();
        let at = |r: f64, z: f64| Point3::new(r * c, r * s, z);
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "spine",
                vec![
                    at(20.0, 0.0),
                    at(80.0, 25.0),
                    at(140.0, -10.0),
                    at(200.0, 15.0),
                ],
                false,
            )
            .expect("a bent line"),
        );
        pipes.push(graph.add(Node::Pipe {
            curve: id,
            radius: ParamRef::Literal(9.0),
        }));
    }
    if pipes.len() > 1 {
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: pipes.into_iter().map(Into::into).collect(),
        });
    }
    (doc, graph)
}
