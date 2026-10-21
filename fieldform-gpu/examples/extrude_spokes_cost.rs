// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a fan of radial extrudes costs the driver and the frame, as the blades multiply.
//!
//! **The companion to `compile_scaling`'s pipe spokes, on the one block that is not a curve
//! walk.** Sixteen pipes on spokes cold-compile in about 120 seconds against 1.0 for two, and the
//! question that leaves is whether the blow-up belongs to the *spine walk* specifically. An
//! extrude reaches the shader a different way -- its profile is an uploaded polyline and a baked
//! narrow band, not a buffer walked per reading -- so a fan of extrudes at the same counts
//! answers it. They cost the same, so it is not the walk.
//!
//! **What it is, established elsewhere and not by this file:** `union-compile-8fa06b7.md` on the
//! compile-and-cull branch instruments the stages and finds the driver inlines the whole
//! composition into `march` six times over -- `field` at five call sites plus `field_normal` --
//! so sixteen pipes become 30 319 inlined lines and 564 loop headers in one function, and the
//! cost is a power of about 2.3 in *that* size. naga parses in 7-14 ms and validates in 1.
//!
//! So the variable is **inlined size**, not the operand count, and the block kind is what sets
//! it: 48 inlined lines an operand for a sphere, 1 835 for a pipe, 2 328 for an extrude. Thirty-two
//! spheres and *one* pipe are the same inlined size and the same cost. Sixteen spheres would not
//! blow up at all; sixteen extrudes do, by a different route from the pipes and at the same price.
//! That is what this table adds -- and why a fix aimed at curve walks would have missed it.
//!
//! The shape: `n` rectangular blades in the XY plane, each running outward along its own spoke
//! from radius 20 to 200, extruded 40 in Z, unioned. At sixteen they overlap near the hub, so the
//! union is one connected body and the operands are genuinely apart at the rim -- which is the
//! case the skip grid and the operand masks are for, rather than a row of boxes that share
//! nothing.
//!
//! Every count is a source this machine has never generated, which is the only way to read a
//! driver figure: a shader it has compiled before comes back from its own disk cache in about ten
//! milliseconds whatever it contains. The last row is the control -- sixteen blades again, the
//! same source -- so its driver column shows what the cache answers and its frame columns bound
//! this run's noise.
//!
//! The frame timing follows the arrangement `mask_frame_cost` arrived at after three faults, each
//! of which made the arms look further apart than they were: one device for every scene, the
//! image size as the outermost loop so no renderer reallocates between calls, the scenes walked
//! forward then backward so the running order does not settle on one, and the least of the rounds
//! as the reading.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example extrude_spokes_cost`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{BooleanOp, Curve, CurveId, Document, Graph, Node, NodeId, ParamRef};
use nalgebra::{Point3, Vector3};

const FRAMES: u32 = 20;
const ROUNDS: u32 = 6;
/// The viewport's own numbers, from `traced/mod.rs`.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
const SKIP_CELLS_SETTLED: u32 = 64;
const SPINE_DEGREES: f64 = 1.0;
const TOLERANCE_MM: f64 = 0.5;

/// Where a blade starts and ends along its spoke, and how wide and tall it is.
const INNER_MM: f64 = 20.0;
const OUTER_MM: f64 = 200.0;
const HALF_WIDTH_MM: f64 = 6.0;
const HEIGHT_MM: f64 = 40.0;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// `n` rectangular blades on their own spokes, extruded in Z and unioned.
///
/// The rectangle is built in the XY plane and turned by hand rather than placed under a
/// `Transform`, because a placement is a point change and would make every blade's subtree be
/// written out again -- which is the other thing that grows a shader, and is not what this is
/// measuring. One blade, one profile curve, one `Extrude`.
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
            height: lit(HEIGHT_MM),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            draft: lit(0.0),
            corners: Default::default(),
        }));
    }
    if ids.len() > 1 {
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.into_iter().map(Into::into).collect() });
    }
    (doc, graph)
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

    // The counts `compile_scaling`'s pipe series uses, so the two tables line up row for row --
    // and sixteen twice, the second being the control.
    let counts: Vec<(usize, bool)> =
        vec![(1, false), (2, false), (4, false), (8, false), (16, false), (16, true)];

    struct Built {
        label: String,
        lines: usize,
        params: usize,
        compile: f64,
        renderer: Renderer,
        multiplier: f64,
    }
    let mut built: Vec<Built> = Vec::new();
    // Every scene framed on the sixteen-blade fan, so the pixels a frame costs are the same
    // pixels at every count and the series measures the field rather than the framing.
    let widest = blades(16);
    let extent = match fieldform_field::part_extent(&widest.0, &widest.1) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("a fan of blades is bounded"),
    };
    let radius = (extent.1 - extent.0).norm() * 0.5;
    let centre = nalgebra::center(&extent.0, &extent.1);
    let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
    let reach = Vector3::repeat(radius);

    for (n, control) in counts {
        let (doc, graph) = blades(n);
        let program = generate(&doc, &graph).expect("generates");
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("every program carries the marker");
        let lines = program.source[cut..].lines().count();
        let curves = curves_for(&doc, &graph, &program);

        let began = std::time::Instant::now();
        let renderer = match Renderer::with_device(
            device.clone(),
            queue.clone(),
            &program,
            &[],
            &curves,
            &[],
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{n} blades: no GPU ({e})");
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
        renderer.build_skip_grid(centre - reach, radius * 2.0, SKIP_CELLS_SETTLED, true);

        built.push(Built {
            label: if control { format!("{n} (again)") } else { format!("{n}") },
            lines,
            params: program.params.len(),
            compile,
            renderer,
            multiplier: fieldform_field::step_multiplier(&doc, &graph),
        });
    }

    let mut best = vec![[f64::MAX; 2]; built.len()];
    for (at, edge) in [(0usize, 720u32), (1, 1440)] {
        for b in &built {
            let budget = budget_for(b.multiplier);
            let camera = framed(eye, centre, radius, b.multiplier);
            let _ = b.renderer.render(&camera, edge, edge, budget);
        }
        for round in 0..ROUNDS {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..built.len()).collect()
            } else {
                (0..built.len()).rev().collect()
            };
            for i in order {
                let b = &built[i];
                let budget = budget_for(b.multiplier);
                let camera = framed(eye, centre, radius, b.multiplier);
                let start = std::time::Instant::now();
                for _ in 0..FRAMES {
                    b.renderer.render(&camera, edge, edge, budget).expect("renders");
                }
                let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES);
                best[i][at] = best[i][at].min(ms);
            }
        }
    }

    println!(
        "{:<12} {:>8} {:>8} {:>12} {:>11} {:>12}",
        "blades", "lines", "params", "driver ms", "720px ms", "1440px ms"
    );
    for (i, b) in built.iter().enumerate() {
        println!(
            "{:<12} {:>8} {:>8} {:>12.0} {:>11.1} {:>12.1}",
            b.label, b.lines, b.params, b.compile, best[i][0], best[i][1],
        );
    }
    println!(
        "\nEvery row but the last is a source this machine had never compiled, so its driver\n\
         figure is cold. The last is the sixteen-blade source a second time: its driver column\n\
         is the disk cache answering, and the gap between its two frame columns and the row\n\
         above is this run's noise rather than anything about the geometry."
    );
}

fn budget_for(multiplier: f64) -> u32 {
    ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
        .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET)
}

fn framed(eye: Point3<f64>, centre: Point3<f64>, radius: f64, multiplier: f64) -> Camera {
    let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    camera.value_scale = multiplier;
    camera
}

/// The profiles the shader asks for, prepared the way the application prepares them.
fn curves_for(
    doc: &Document,
    graph: &Graph,
    program: &fieldform_gpu::ShaderProgram,
) -> Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> {
    program
        .curves
        .iter()
        .map(|use_of| match *use_of {
            fieldform_gpu::CurveUse::Profile(node) => {
                let Some(Node::Extrude { outline, sweep, .. }) = graph.node(node) else {
                    unreachable!("every profile here belongs to an extrude");
                };
                let world =
                    fieldform_field::profile_loop_turning(doc, outline.curve(), SPINE_DEGREES, sweep.source())
                        .expect("a closed loop")
                        .points;
                let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                let rotation = fieldform_field::rotation_from_z(normal);
                let into = move |p: Point3<f64>| {
                    Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                };
                let built =
                    fieldform_field::Profile::in_plane(&world, into, TOLERANCE_MM).expect("planar");
                let points: Vec<Point3<f64>> =
                    built.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                let reads = fieldform_field::profile_reads_out_to(doc, graph, node);
                let (lo, hi) = built.bounds();
                let bake = fieldform_field::bake_cells((hi - lo).max(), TOLERANCE_MM, reads)
                    .map(|cells| built.baked(cells, reads))
                    .and_then(|p| p.bake().cloned());
                (points, bake)
            }
            other => unreachable!("a fan of blades asks for nothing else: {other:?}"),
        })
        .collect()
}
