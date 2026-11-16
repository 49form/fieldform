// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a run of pipes costs and buys when the pipes are joined by a blend, or tapered by one
//! ramp, or both -- the two shapes `union_loop_cost`'s loop never reached.
//!
//! **Because a frame of bones is modelled exactly that way, and it took minutes to compile.**
//! A chair of twenty curved struts a side, each a pipe whose radius one ramp tapers, all joined
//! by one G2 blend and mirrored: the application rebuilt its view in 325 s, then 596 s, and a
//! still could not be drawn inside the snapshot's limit at all. `document_cost` on the saved
//! file laid it at the driver's door, and at the two conditions a run had: the same twenty
//! pipes hard-unioned at a plain radius compiled in 2.7 s, and joined by the blend in 505 s --
//! after which the first 720 px frame ran past the driver's watchdog and lost the device.
//!
//! A union of pipes has been one loop since `pipe_run_distance`. A blend of them was still a
//! call a pipe, because a blend folds as a chain rather than a `min`; and a pipe whose radius
//! a ramp drives ended any run, because it is the offset expression rather than a number. Both
//! are loops now (`blend_run` and `RunKind::DrivenPipes` in codegen), and this is the table
//! that says what each is worth and whether the floor a union's run has is the floor theirs
//! should have.
//!
//! Three arms, one process, one device, every source cold:
//!
//! - **as calls** -- `RUN_FLOOR` at `usize::MAX`, which never loops: a call a pipe and a join a
//!   link, with the operand masks a written-out join has.
//! - **as a loop** -- one walk, the count in the buffer, and a member read only as far as it
//!   could still change the answer (`spine_distance_within` in the prelude), which rules a far
//!   spine out at its box and a near one down to the stretch that is near.
//! - **loop, unpruned** -- the same walk with that switched off, by making every member's
//!   reach unlimited. What the loop alone is worth, and so what ruling members out adds.
//!
//! And the control: **spheres**, which no run of pipes can touch, so every arm compiles the
//! same source and their spread is this machine's noise rather than an assertion about it.
//! Only the control's frame columns mean anything; its later arms' compiles are the driver's
//! cache.
//!
//! **What it found** -- an RTX A500 Laptop GPU on Vulkan, 2026-11-14, `-- 2 3 4 5 8 16`,
//! milliseconds; compile as calls → as a loop, then the 720 px frame as calls → as a loop and,
//! in brackets, the loop with nothing ruled out:
//!
//! | pipes | blended | | tapered | | blended, tapered | |
//! |---|---|---|---|---|---|---|
//! | 2 | 668 → 494 | 6.9 → 7.0 (7.7) | 1 021 → 729 | 7.1 → 7.8 (7.6) | 696 → 527 | 6.8 → 6.8 (7.3) |
//! | 3 | 1 410 → 763 | 7.9 → 7.3 (8.4) | 1 398 → 738 | 7.0 → 7.7 (8.4) | 945 → 559 | 7.5 → 7.2 (8.1) |
//! | 4 | 1 994 → 810 | 9.2 → 7.9 (9.7) | 1 794 → 722 | 7.6 → 7.6 (9.3) | 1 437 → 535 | 8.9 → 7.6 (10.4) |
//! | 5 | 3 926 → 812 | 11.5 → 9.0 (10.6) | 3 124 → 715 | 9.3 → 8.5 (10.2) | 2 365 → 576 | 11.0 → 8.7 (10.2) |
//! | 8 | 14 095 → 787 | 31.1 → 11.3 (15.7) | 13 335 → 719 | 22.7 → 10.9 (15.2) | 11 087 → 535 | 28.6 → 10.3 (14.5) |
//! | 16 | 41 319 → 783 | 83.8 → 19.8 (31.7) | 40 260 → 482 | 54.1 → 17.4 (29.6) | 33 553 → 546 | 84.0 → 18.3 (32.3) |
//!
//! At 1440 px the sixteen-pipe rows read 241.3 → 59.0, 154.7 → 52.0 and 215.4 → 53.3. The
//! controls, whose three arms are the same bytes, read 5.3, 5.7 and 6.0 ms at 720 px for
//! eight spheres and 6.5, 6.9 and 6.8 for nine -- so up to an eighth between arms is this
//! machine and this running order, and nothing below four pipes clears it. The loop's own
//! compile is flat, half a second to eight tenths, whatever the count; written out it is a
//! power of the count, as a union's was. And the chair this began with, through
//! `document_cost` on its saved file: 414 s of pipeline and a lost device at the first frame,
//! against 2.7 s and 247 ms a frame at 720 px.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example blend_run_cost`
//! Counts from the command line, as `union_loop_cost` takes them: `-- 2 3 4 5` is the reading
//! for the floor. The `as calls` arm at sixteen blended pipes is minutes of the driver's time,
//! which is the compile this is here to remove.

use fieldform_gpu::{Camera, Renderer, generate_with_run_floor};
use fieldform_scene::{
    BooleanOp, Continuity, Curve, CurveId, Document, Fillet, Graph, Node, NodeId, ParamRef, Place,
    RampShape,
};
use nalgebra::{Point3, Vector3};

/// The viewport's own numbers, from `traced/mod.rs`.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
const SPINE_DEGREES: f64 = 1.0;
const FRAMES: u32 = 20;
/// How many times round the arms. The least of the rounds is the reading.
const ROUNDS: u32 = 6;
/// The settled grid the viewport marches, from `traced/mod.rs`. The masks read it, so an arm
/// measured without it pays for the apparatus with the benefit switched off.
const SKIP_CELLS_SETTLED: u32 = 64;
/// What the blend reaches by, and what the ramp tapers a pipe between.
const BLEND_MM: f64 = 6.0;
const THICK_MM: f64 = 12.0;
const THIN_MM: f64 = 5.0;

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

    let asked: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let counts = if asked.is_empty() { vec![2, 4, 8, 16] } else { asked };
    let shapes: Vec<(&str, fn(usize) -> (Document, Graph), Vec<usize>)> = vec![
        ("blended", blended, counts.clone()),
        ("tapered", tapered, counts.clone()),
        ("blended, tapered", blended_tapered, counts),
        // The controls: no run can form over spheres, so the arms share their bytes.
        ("spheres (control)", spheres, vec![8]),
        ("spheres (control)", spheres, vec![9]),
    ];

    println!(
        "  {:<18} {:>3} {:<15} {:>7} {:>7} {:>10} {:>9} {:>10}",
        "shape", "n", "arm", "lines", "params", "compile ms", "720px ms", "1440px ms"
    );
    for (label, build, counts) in shapes {
        for n in counts {
            let (doc, graph) = build(n);
            for (arm, lines, params, compile, frames) in measure(&device, &queue, &doc, &graph) {
                println!(
                    "  {label:<18} {n:>3} {arm:<15} {lines:>7} {params:>7} {compile:>10.0} \
                     {:>9.1} {:>10.1}",
                    frames[0], frames[1]
                );
            }
        }
    }
    println!(
        "\nLeast of {ROUNDS} rounds of {FRAMES} frames, arms interleaved, one device. A compile is \n\
         cold once: a source the driver has seen comes back from its cache in tens of milliseconds."
    );
}

/// The arms of one graph: built first, then timed round by round, interleaved -- timed one
/// after another they measure the running order, as `union_loop_cost` records.
#[allow(clippy::type_complexity)]
fn measure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    doc: &Document,
    graph: &Graph,
) -> Vec<(&'static str, usize, usize, f64, [f64; 2])> {
    let extent = match fieldform_field::part_extent(doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("this document is unbounded"),
    };
    let size = (extent.1 - extent.0).norm();
    let centre = nalgebra::center(&extent.0, &extent.1);
    let radius = size * 0.5;
    let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
    let multiplier = fieldform_field::step_multiplier(doc, graph);
    let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
        .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);

    let mut built = Vec::new();
    for (arm, floor, pruned) in [
        ("as calls", usize::MAX, true),
        ("as a loop", 2usize, true),
        ("loop, unpruned", 2usize, false),
    ] {
        let mut program = generate_with_run_floor(doc, graph, floor).expect("generates");
        if !pruned {
            // Nothing is ever out of reach: every member is searched with no cap, so none is
            // ruled out. The same walk otherwise, to the line.
            let head = "fn spine_distance_within(index: u32, p: vec3<f32>, reach: f32) -> f32 {";
            assert_eq!(program.source.matches(head).count(), 1, "the prelude's one reader");
            let unlimited = "fn spine_distance_within(index: u32, p: vec3<f32>, asked: f32) -> f32 {\n    \
                             let reach = 1e30 + 0.0 * asked;";
            program.source = program.source.replace(head, unlimited);
        }
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("marked");
        let lines = program.source[cut..].lines().count();
        let curves = curves_for(doc, graph, &program);

        let began = std::time::Instant::now();
        let renderer =
            Renderer::with_device(device.clone(), queue.clone(), &program, &[], &curves, &[])
                .expect("the driver builds it");
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
        let reach = Vector3::repeat(radius);
        renderer.build_skip_grid(centre - reach, radius * 2.0, SKIP_CELLS_SETTLED, true);
        built.push((arm, lines, program.params.len(), compile, renderer));
    }

    let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    camera.value_scale = multiplier;
    let mut best = vec![[f64::MAX; 2]; built.len()];
    // Size outermost: a renderer reallocates when asked for a size it did not draw last.
    for (edge_at, edge) in [(0usize, 720u32), (1, 1440)] {
        for (_, _, _, _, renderer) in &built {
            let _ = renderer.render(&camera, edge, edge, budget);
        }
        for round in 0..ROUNDS {
            let order: Vec<usize> =
                if round % 2 == 0 { (0..built.len()).collect() } else { (0..built.len()).rev().collect() };
            for i in order {
                let renderer = &built[i].4;
                let start = std::time::Instant::now();
                for _ in 0..FRAMES {
                    renderer.render(&camera, edge, edge, budget).expect("renders");
                }
                let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES);
                best[i][edge_at] = best[i][edge_at].min(ms);
            }
        }
    }
    built
        .into_iter()
        .enumerate()
        .map(|(i, (arm, lines, params, compile, _))| (arm, lines, params, compile, best[i]))
        .collect()
}

/// `n` bent pipes on their own spokes, meeting near the hub, not yet joined -- `union_loop_cost`'s
/// spokes, each radius a number or the one ramp's reading.
fn bones(n: usize, ramp: Option<NodeId>, mut graph: Graph) -> (Document, Graph, Vec<NodeId>) {
    let mut doc = Document::new();
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
                "bone",
                vec![at(20.0, 0.0), at(80.0, 25.0), at(140.0, -10.0), at(200.0, 15.0)],
                false,
            )
            .expect("a bent line"),
        );
        let radius = match ramp {
            Some(ramp) => ParamRef::Field(ramp),
            None => ParamRef::Literal(9.0),
        };
        pipes.push(graph.add(Node::Pipe { curve: id.into(), radius }));
    }
    (doc, graph, pipes)
}

/// A ramp reading how far a point stands from the hub: thick there, thin at the tips.
fn hub_ramp(graph: &mut Graph) -> NodeId {
    let nought = ParamRef::Literal(0.0);
    let hub = graph
        .add(Node::Sphere { centre: Place::At([nought, nought, nought]), radius: ParamRef::Literal(1.0) });
    graph.add(Node::Ramp {
        input: hub.into(),
        in_min: ParamRef::Literal(0.0),
        in_max: ParamRef::Literal(200.0),
        out_min: ParamRef::Literal(THICK_MM),
        out_max: ParamRef::Literal(THIN_MM),
        shape: RampShape::Linear,
    })
}

fn joined(mut graph: Graph, pipes: Vec<NodeId>, blended: bool) -> Graph {
    let inputs = pipes.into_iter().map(Into::into).collect();
    if blended {
        graph.add(Node::Blend {
            op: BooleanOp::Union,
            inputs,
            fillet: Fillet {
                radius: ParamRef::Literal(BLEND_MM),
                continuity: Continuity::G2,
                second: None,
            },
        });
    } else {
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs });
    }
    graph
}

/// Plain pipes, joined by one blend.
fn blended(n: usize) -> (Document, Graph) {
    let (doc, graph, pipes) = bones(n, None, Graph::empty());
    (doc, joined(graph, pipes, true))
}

/// Pipes one ramp tapers, in one union.
fn tapered(n: usize) -> (Document, Graph) {
    let mut graph = Graph::empty();
    let ramp = hub_ramp(&mut graph);
    let (doc, graph, pipes) = bones(n, Some(ramp), graph);
    (doc, joined(graph, pipes, false))
}

/// Both: a frame of bones as somebody models one.
fn blended_tapered(n: usize) -> (Document, Graph) {
    let mut graph = Graph::empty();
    let ramp = hub_ramp(&mut graph);
    let (doc, graph, pipes) = bones(n, Some(ramp), graph);
    (doc, joined(graph, pipes, true))
}

/// The control: `n` spheres unioned, which no run of pipes can touch.
fn spheres(n: usize) -> (Document, Graph) {
    let mut graph = Graph::empty();
    let inputs: Vec<NodeId> = (0..n)
        .map(|i| {
            graph.add(Node::Sphere {
                centre: Place::At([
                    ParamRef::Literal(i as f64 * 45.0),
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(0.0),
                ]),
                radius: ParamRef::Literal(20.0),
            })
        })
        .collect();
    graph.add(Node::Boolean {
        op: BooleanOp::Union,
        inputs: inputs.into_iter().map(Into::into).collect(),
    });
    (Document::new(), graph)
}

/// Every spine the program asked for, tessellated as the application tessellates it.
fn curves_for(
    doc: &Document,
    graph: &Graph,
    program: &fieldform_gpu::ShaderProgram,
) -> Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> {
    let extent = match fieldform_field::part_extent(doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("bounded"),
    };
    let floor = ((extent.1 - extent.0).norm() * 1e-5).max(1e-4);
    program
        .curves
        .iter()
        .map(|use_of| match *use_of {
            fieldform_gpu::CurveUse::Spine(id) => {
                (doc.curves[&id].tessellate_to_angle(SPINE_DEGREES, floor), None)
            }
            other => panic!("this harness builds spines, not {other:?}"),
        })
        .collect()
}
