// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a union walked as a loop costs and buys, against the same union written out.
//!
//! **The reading that decides `RUN_FLOOR`.** `union-compile-8fa06b7.md` measured that a union of
//! sixteen pipes costs the driver 122 931 ms cold, because `march` reads the field at six places
//! and each operand is written into all six — 1 835 inlined lines a pipe, against a sphere's 48 —
//! and the driver's compile is a power of about 2–2.5 in that size. `pipe_run_distance` collapses
//! a run of plain pipes into one walk behind a loop whose count is a buffer value, which is backlog
//! item 34's fix one level up from the pattern's window it was built for.
//!
//! **It is not free, and what it costs is the operand masks.** A mask is a function and a bit per
//! operand, and a run has collapsed exactly that distinction: its members are one call now, so
//! there is nothing per-operand left to skip. `8fa06b7` measured the masks at 17.0 ms against 20.8
//! on eight pipes on spokes at 720 px — a fifth of the frame, on precisely the shape a run is. So
//! the loop trades some frame for a great deal of compile, and neither number may be assumed.
//!
//! Two arms, one process, one device, every source cold — the prelude changed when
//! `pipe_run_distance` was added, so every arm's source is one this machine has never compiled:
//!
//! - **as calls** — `RUN_FLOOR` at `usize::MAX`, which never loops. What the crate emitted before,
//!   masks and all.
//! - **as a loop** — one walk, the count in the buffer, no masks on the run.
//!
//! And the control: **spheres**, which no run can touch, so both arms compile the same source and
//! their spread is this machine's noise floor rather than an assertion about it. Measured at 8 and
//! 9 spheres it is up to 9% at 720 px and 1% at 1440 — which is what makes four pipes' +1% nothing
//! and three pipes' +12% something. Only the control's *frame* columns mean anything: its two arms
//! are byte-identical by construction, so the second one's compile is the driver's cache and reads
//! about 10 ms whatever it contains.
//!
//! **And blades, which is the second kind of run.** `extrude_spokes_cost.rs` put sixteen extrudes
//! on the same spokes and they cost the same 124 s, so the blow-up was never the spine walk: an
//! extrude is 2 328 inlined lines an operand, most of it `extrude_normal` central-differencing
//! `extrude_distance` six times over. `extrude_run_distance` collapses a fan the same way, which
//! is a second `RUN_FLOOR` question and not the same one — whether four is still where the frame
//! stops regressing is for this table to say, not for the pipes' table to be assumed over.
//!
//! What it found: four again, and `RUN_FLOOR`'s own table carries the rows. At sixteen blades the
//! compile goes 49 148 → 394 ms and the frame 28.3 → 12.4 at 720 px.
//!
//! **One thing to watch in a reading of this table**: a compile figure is only cold once, and a
//! shader whose *emitted* module has not changed is served from the driver's disk cache in tens of
//! milliseconds. Adding the two extrude-run functions to the prelude changed no existing shader's
//! module at all — naga strips what nothing reaches — so the pipes' compile column came back from
//! the cache on the run that measured the blades. The frame columns are never cached.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example union_loop_cost`
//! Several minutes on its first run: the `as calls` arm at sixteen pipes is the two-minute compile
//! this is here to remove.

use fieldform_gpu::{Camera, Renderer, generate_with_run_floor};
use fieldform_scene::{
    BooleanOp, Curve, CurveId, Document, Graph, Node, NodeId, ParamRef, Place,
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
/// The export tolerance a profile is baked at, from `traced/rebuild.rs`.
const TOLERANCE_MM: f64 = 0.5;

/// Where a blade starts and ends along its spoke, and how wide and tall it is —
/// `extrude_spokes_cost`'s numbers, so the two tables are rows of one series.
const INNER_MM: f64 = 20.0;
const OUTER_MM: f64 = 200.0;
const HALF_WIDTH_MM: f64 = 6.0;
const HEIGHT_MM: f64 = 40.0;

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

    // **Counts from the command line**, because a compile reading is only cold once: the driver
    // serves a source it has seen from its own cache. `-- 3 5` is what pinned `RUN_FLOOR`, the
    // question being where the frame stops regressing rather than where the compile starts paying.
    let asked: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let counts = if asked.is_empty() {
        vec![2, 4, 8, 16]
    } else {
        asked
    };
    let shapes: Vec<(&str, fn(usize) -> (Document, Graph), Vec<usize>)> = vec![
        ("pipes", spokes, counts.clone()),
        ("blades", blades, counts),
        // The control. A sphere is not a pipe, so no run can form and both arms generate the
        // same bytes; whatever they differ by is the floor under every figure above.
        ("spheres (control)", spheres, vec![8]),
        // The second control: nine spheres, whose source both arms share too, so the pair above
        // and this one bracket the noise rather than resting on one reading of it.
        ("spheres (control)", spheres, vec![9]),
    ];

    println!(
        "  {:<18} {:>3} {:<10} {:>7} {:>7} {:>10} {:>9} {:>10}",
        "shape", "n", "arm", "lines", "params", "compile ms", "720px ms", "1440px ms"
    );
    for (label, build, counts) in shapes {
        for n in counts {
            let (doc, graph) = build(n);
            let rows = measure(&device, &queue, &doc, &graph, n);
            for (arm, lines, params, compile, frames) in rows {
                println!(
                    "  {label:<18} {n:>3} {arm:<10} {lines:>7} {params:>7} {compile:>10.0} \
                     {:>9.1} {:>10.1}",
                    frames[0], frames[1]
                );
            }
        }
    }
    println!(
        "\nThe compile is paid once per shape — and again for every combination of hidden bodies, \n\
         since hiding one is a different graph and so a different shader. The frame is paid sixty \n\
         times a second. Least of {ROUNDS} rounds of {FRAMES} frames, arms interleaved, one device."
    );
}

/// Both arms of one graph: built first, then timed round by round.
///
/// **Built first and interleaved afterwards, because timing them one after another measures the
/// running order.** `mask_frame_cost` found 8% at 720 px and 15% at 1440 handed to whichever arm
/// went last, on byte-identical sources — the card's clocks fall as a run warms. One device for
/// both, for the same reason it records: three renderers holding their own devices put arms 15%
/// apart on identical source.
#[allow(clippy::type_complexity)]
fn measure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    doc: &Document,
    graph: &Graph,
    n: usize,
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
    for (arm, floor) in [("as calls", usize::MAX), ("as a loop", 2usize)] {
        let program = generate_with_run_floor(doc, graph, floor).expect("generates");
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
    let _ = n;

    let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    camera.value_scale = multiplier;
    let mut best = vec![[f64::MAX; 2]; built.len()];
    // Size outermost: a renderer reallocates when asked for a size it did not draw last.
    for (edge_at, edge) in [(0usize, 720u32), (1, 1440)] {
        for (_, _, _, _, renderer) in &built {
            let _ = renderer.render(&camera, edge, edge, budget);
        }
        for round in 0..ROUNDS {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..built.len()).collect()
            } else {
                (0..built.len()).rev().collect()
            };
            for i in order {
                let renderer = &built[i].4;
                let start = std::time::Instant::now();
                for _ in 0..FRAMES {
                    renderer
                        .render(&camera, edge, edge, budget)
                        .expect("renders");
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

/// `n` bent pipes on their own spokes, meeting only near the hub, in one union — the shape
/// `union-compile-8fa06b7.md` measured and the one `OperandMasks`' own table is built on.
fn spokes(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let mut pipes = Vec::new();
    for i in 0..n {
        let turn = std::f64::consts::TAU * i as f64 / n as f64;
        let (c, s) = (turn.cos(), turn.sin());
        let id = fieldform_scene::CurveId::new();
        let at = |r: f64, z: f64| Point3::new(r * c, r * s, z);
        doc.curves.insert(
            id,
            fieldform_scene::Curve::polyline(
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
    graph.add(Node::Boolean {
        op: BooleanOp::Union,
        inputs: pipes.into_iter().map(Into::into).collect(),
    });
    (doc, graph)
}

/// `n` rectangular blades on the same spokes, extruded in Z and unioned — the other kind of run.
///
/// `extrude_spokes_cost`'s fan exactly. The rectangle is built in the XY plane and turned by hand
/// rather than placed under a `Transform`, because a placement is a point change and would make
/// every blade's subtree be written out again — which is the other thing that grows a shader, and
/// is not what this is measuring.
fn blades(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let mut ids: Vec<NodeId> = Vec::new();
    for i in 0..n {
        let turn = std::f64::consts::TAU * i as f64 / n as f64;
        let (c, s) = (turn.cos(), turn.sin());
        // Along the spoke by `r`, across it by `w`.
        let at = |r: f64, w: f64| Point3::new(r * c - w * s, r * s + w * c, 0.0);
        let id = CurveId::new();
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "blade",
                vec![
                    at(INNER_MM, -HALF_WIDTH_MM),
                    at(OUTER_MM, -HALF_WIDTH_MM),
                    at(OUTER_MM, HALF_WIDTH_MM),
                    at(INNER_MM, HALF_WIDTH_MM),
                ],
                true,
            )
            .expect("a rectangle"),
        );
        ids.push(graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: ParamRef::Literal(HEIGHT_MM),
            start: ParamRef::Literal(0.0),
            sides: fieldform_scene::Sides::One,
            draft: ParamRef::Literal(0.0),
            corners: Default::default(),
        }));
    }
    if ids.len() > 1 {
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: ids.into_iter().map(Into::into).collect(),
        });
    }
    (doc, graph)
}

/// The control: `n` spheres unioned, which no run can touch, so both arms are the same bytes.
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

/// Every curve the program asked for, tessellated as the application tessellates it.
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
            fieldform_gpu::CurveUse::Spine(id) => (
                doc.curves[&id].tessellate_to_angle(SPINE_DEGREES, floor),
                None,
            ),
            // A profile is uploaded flattened into its block's own frame and with its narrow band
            // baked, which is `extrude_spokes_cost`'s arrangement and the application's: an arm
            // that uploaded the polyline alone would be measuring the walk the bake replaces.
            fieldform_gpu::CurveUse::Profile(node) => {
                let Some(Node::Extrude { outline, sweep, .. }) = graph.node(node) else {
                    unreachable!("every profile here belongs to an extrude");
                };
                let world = fieldform_field::profile_loop_turning(
                    doc,
                    outline.curve(),
                    SPINE_DEGREES,
                    sweep.source(),
                )
                .expect("a closed loop")
                .points;
                let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                let rotation = fieldform_field::rotation_from_z(normal);
                let into = move |p: Point3<f64>| {
                    Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                };
                let built = fieldform_field::Profile::in_plane(&world, into, TOLERANCE_MM)
                    .expect("planar");
                let points: Vec<Point3<f64>> = built
                    .points()
                    .iter()
                    .map(|p| Point3::new(p.x, p.y, 0.0))
                    .collect();
                let reads = fieldform_field::profile_reads_out_to(doc, graph, node);
                let (lo, hi) = built.bounds();
                let bake = fieldform_field::bake_cells((hi - lo).max(), TOLERANCE_MM, reads)
                    .map(|cells| built.baked(cells, reads))
                    .and_then(|p| p.bake().cloned());
                (points, bake)
            }
            other => panic!("this harness builds spines and profiles, not {other:?}"),
        })
        .collect()
}
