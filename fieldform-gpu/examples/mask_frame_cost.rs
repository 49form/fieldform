// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the operand masks buy per frame, against what they cost to compile.
//!
//! **The reading that decides whether anything changes.** `OperandMasks` skips an operand when a
//! cell cannot contain it, which makes the march cheaper; it also emits a function and a bit per
//! masked operand, and on the owner's wing that measured 3215 ms of driver compile against 1052
//! with the masks off (`compile_scaling` carries how that was found). A three-fold compile for a
//! faster frame is a good trade in a renderer and a bad one in a modeller, where every change of
//! shape — including hiding a body, which is a different graph and so a different shader — pays
//! the compile again.
//!
//! So: the same document, both ways, in one process on one device. The compile is timed once and
//! the frame many times, at the two sizes the viewport actually uses — the 720-pixel interactive
//! tier while anything moves, and the window's own size once it is quiet.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example mask_frame_cost -- <file.fform>`

use fieldform_gpu::{generate_with_masks, Camera, Renderer};
use fieldform_scene::{Document, Graph, Node, ParamRef};
use nalgebra::{Point3, Vector3};

/// The viewport's own numbers, from `traced/mod.rs`.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
const SPINE_DEGREES: f64 = 1.0;
const TOLERANCE_MM: f64 = 0.5;
const FRAMES: u32 = 20;
/// How many times round the arms. The least of the rounds is the reading; see the loop.
const ROUNDS: u32 = 6;
/// The settled grid the viewport marches, from `traced/mod.rs`.
const SKIP_CELLS_SETTLED: u32 = 64;

fn main() {
    // With no file, the case the feature was built for: "far from the pipe a blend with the
    // pipe *is* the other body, outside the cube a lattice intersected with the cube *is* the
    // cube". A document that does not contain one cannot say whether masking pays.
    let (doc, owned, path) = match std::env::args().nth(1).as_deref() {
        // The two shapes `framed_capture` measured V6 on, where masking was worth 1.8x --
        // re-measurable here because the verdict has to survive the case the feature won.
        Some("spokes") => {
            let (doc, graph) = spokes(false);
            (doc, graph, "eight pipes in one union".to_string())
        }
        Some("spokes-blended") => {
            let (doc, graph) = spokes(true);
            (doc, graph, "the same eight, chained by 3 mm blends".to_string())
        }
        Some("lattice") | None => {
            let (doc, graph) = lattice_in_a_cube();
            (doc, graph, "a lattice intersected with a cube (synthetic)".to_string())
        }
        Some(path) => {
            let (doc, _) = fieldform_io::open(std::path::Path::new(path)).expect("opens");
            let graph = doc.parts_ordered().first().copied().expect("a part").graph.clone();
            (doc, graph, path.to_string())
        }
    };
    let graph = &owned;

    let extent = match fieldform_field::part_extent(&doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("this document is unbounded"),
    };
    let size = (extent.1 - extent.0).norm();
    let centre = nalgebra::center(&extent.0, &extent.1);
    let radius = size * 0.5;
    let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
    let multiplier = fieldform_field::step_multiplier(&doc, graph);
    let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
        .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);

    println!("{path}");
    println!("step budget {budget}, value scale {multiplier:.2}\n");

    // **Three arms, because "masked is slower" has two causes that want opposite fixes.**
    // `off` is the composition without the feature. `on` is the feature working. `on, unread`
    // compiles every masked operand into its own function and its branch, and then fills the
    // grid with `copy_masks` false, which is the one flag `field_here` reads before it takes
    // either mask out of the grid -- so the shader carries the whole apparatus and skips
    // nothing. The gap between `off` and `on, unread` is what the masks cost where they never
    // fire; the gap between `on, unread` and `on` is what the skipping buys back.
    //
    // A document with a `Pattern` in it pays a second thing in that middle arm, because the
    // same flag also switches off the copy masks, which are a different feature (plan Sec.5 V6).
    let arms = [("on", true, true), ("on, unread", true, false), ("off", false, true)];

    // Built first, all of them, and timed afterwards round by round.
    //
    // **Because timing them one after another measured the running order, not the arms.** With
    // a document that masks nothing -- so all three sources are byte-identical -- whichever arm
    // went last read 8% slow at 720 px and 15% slow at 1440, reproducibly over three runs. The
    // card's clocks fall as the run warms, and a table that walks the arms in order hands the
    // whole of that drift to the last one. Round robin spreads it evenly, and the least of the
    // rounds is the reading with the least of it in.
    // **One device for all three, because three of them contend.** `Renderer::new` asks for
    // its own adapter and device; holding three alive at once put the arms 15% apart at 1440
    // with byte-identical sources, and neither round robin nor reversing the order touched it.
    // `with_device` is what the application uses for the same reason.
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

    let mut built = Vec::new();
    for (label, masked, read) in arms {
        // Asked for by argument rather than through the environment, so all three compositions
        // are built in one process against one device — the only way to compare them
        // without the driver's disk cache deciding the answer.
        let program = generate_with_masks(&doc, graph, masked).expect("generates");
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("marked");
        let lines = program.source[cut..].lines().count();
        let curves = curves_for(&doc, graph, &program);

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
                eprintln!("no GPU available ({e})");
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

        // **The grid the masks read, which this harness did not build at first.** Without it
        // every mask bit is nought, nothing is ever skipped, and the masked arm pays for the
        // machinery with the benefit switched off -- which is not a comparison, it is a
        // measurement of the overhead alone. The application fills it from the scene's own cube
        // (`traced/draw.rs`, `build_skip_grid`), coarse while anything moves and fine once it
        // rests; this takes the settled one, since a frame measured over twenty repeats is a rested
        // view by definition. Caught by another session reading the harness rather than the
        // result.
        let reach = Vector3::repeat(radius);
        renderer.build_skip_grid(centre - reach, radius * 2.0, SKIP_CELLS_SETTLED, read);

        built.push((label, lines, program.params.len(), compile, renderer));
    }

    let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    camera.value_scale = multiplier;
    let mut best = vec![[f64::MAX; 2]; built.len()];
    // One size at a time, then round by round, the arms forward then backward.
    //
    // **Each nesting level here is a bias that was measured rather than guessed at.** A renderer
    // reallocates its buffers when it is asked for a size it did not draw last, so asking one
    // arm for 720 and 1440 alternately paid that on every call: size is the outermost loop, and
    // each renderer sees one size for a whole pass. Reversing the arms on the odd rounds keeps
    // the running order from settling on one of them. And the least of the rounds is the
    // reading, because what is left is the card's clocks falling as the run warms, which only
    // ever makes a frame look slower.
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
                    renderer.render(&camera, edge, edge, budget).expect("renders");
                }
                let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES);
                best[i][edge_at] = best[i][edge_at].min(ms);
            }
        }
    }

    println!(
        "{:<12} {:>8} {:>8} {:>12} {:>14} {:>14}",
        "masks", "lines", "params", "compile ms", "720px ms", "1440px ms"
    );
    for (i, (label, lines, params, compile, _)) in built.iter().enumerate() {
        println!(
            "{label:<12} {lines:>8} {params:>8} {compile:>12.0} {:>14.1} {:>14.1}",
            best[i][0], best[i][1],
        );
    }
    println!(
        "\nThe compile is paid once per shape — and again for every combination of hidden\n\
         bodies, since hiding one is a different graph. The frame is paid sixty times a second."
    );
}

/// **The shape the masks were shown to win on** -- eight bent pipes on their own spokes, meeting
/// only near the hub, joined in one union or chained by blends. Copied from
/// `fieldform-io/examples/framed_capture.rs`, whose own comment says of it: "What the operand
/// masks are for, which none of the owner's documents happens to be."
fn spokes(blended: bool) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let mut pipes = Vec::new();
    for i in 0..8 {
        let turn = std::f64::consts::TAU * f64::from(i) / 8.0;
        let (c, s) = (turn.cos(), turn.sin());
        let id = fieldform_scene::CurveId::new();
        let at = |r: f64, z: f64| Point3::new(r * c, r * s, z);
        doc.curves.insert(
            id,
            fieldform_scene::Curve::polyline(
                id,
                "spine",
                vec![at(20.0, 0.0), at(80.0, 25.0), at(140.0, -10.0), at(200.0, 15.0)],
                false,
            )
            .expect("a bent line"),
        );
        pipes.push(graph.add(Node::Pipe { curve: id, radius: ParamRef::Literal(9.0) }));
    }
    if blended {
        let mut joined = pipes[0];
        for next in &pipes[1..] {
            joined = graph.add(Node::Blend {
                op: fieldform_scene::BooleanOp::Union,
                a: joined.into(),
                b: (*next).into(),
                fillet: fieldform_scene::Fillet {
                    radius: ParamRef::Literal(3.0),
                    continuity: fieldform_scene::Continuity::G1,
                },
            });
        }
    } else {
        graph.add(Node::Boolean { op: fieldform_scene::BooleanOp::Union, inputs: pipes.into_iter().map(Into::into).collect() });
    }
    (doc, graph)
}

/// The shape `OperandMasks` was written for: a gyroid intersected with a cube, where outside
/// the cube the answer *is* the cube and the lattice need not be read at all.
///
/// Small cells, so the lattice is genuinely the dear operand — which is the premise the whole
/// argument rests on.
fn lattice_in_a_cube() -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let mut g = Graph::empty();
    let cube = g.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        size: [lit(200.0), lit(200.0), lit(200.0)],
        turn: fieldform_scene::Turn::default(),
    });
    let cell = g.add(Node::Lattice {
        cell: fieldform_scene::LatticeCell::Gyroid,
        map: fieldform_scene::CellMap::Rectangular { cell: lit(12.0) },
        phase: fieldform_scene::LatticePhase::Sheet {
            thickness: lit(1.5),
            offset: lit(0.0),
        },
        frame: fieldform_scene::Frame::default(),
    });
    g.add(Node::Boolean {
        op: fieldform_scene::BooleanOp::Intersection,
        inputs: vec![cube.into(), cell.into()],
    });
    (Document::new(), g)
}

/// The spines, frames and profiles the shader asks for, read the way the application reads them.
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
            // No scene here mitres its draft, so there are no corners to displace.
            fieldform_gpu::CurveUse::Miters(_) => {
                unreachable!("this scene's draft rounds its corners")
            }
            fieldform_gpu::CurveUse::LoftSpine(_)
            | fieldform_gpu::CurveUse::LoftSection(..)
            | fieldform_gpu::CurveUse::LoftSlope(..) => {
                unreachable!("this scene lofts nothing")
            }
            fieldform_gpu::CurveUse::Frame(node) => {
                let Some(Node::Sweep { outline, spine, .. }) = graph.node(node) else {
                    unreachable!("a frame belongs to a sweep");
                };
                let points = doc.curves[spine].tessellate_to_angle(SPINE_DEGREES, floor);
                let frame = fieldform_field::sweep_frame(doc, outline, *spine, &points)
                    .expect("a spine with a direction");
                (frame, None)
            }
            fieldform_gpu::CurveUse::Profile(node) => {
                let (curve, ignore) = match graph.node(node) {
                    Some(Node::Extrude { outline, sweep, .. }) => (outline.curve(), sweep.source()),
                    Some(Node::Sweep { outline, spine, .. }) => (outline.curve(), Some(*spine)),
                    Some(Node::Revolve { outline, frame, .. }) => {
                        (outline.curve(), frame.curve_source())
                    }
                    _ => unreachable!("a profile belongs to an extrude, a revolve or a sweep"),
                };
                let world = fieldform_field::profile_loop_turning(doc, curve, SPINE_DEGREES, ignore)
                    .expect("a closed loop")
                    .points;
                let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                let rotation = fieldform_field::rotation_from_z(normal);
                let into = move |p: Point3<f64>| {
                    Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                };
                let profile =
                    fieldform_field::Profile::in_plane(&world, into, TOLERANCE_MM).expect("planar");
                let points: Vec<Point3<f64>> =
                    profile.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                let reads = fieldform_field::profile_reads_out_to(doc, graph, node);
                let (lo, hi) = profile.bounds();
                let bake = fieldform_field::bake_cells((hi - lo).max(), TOLERANCE_MM, reads)
                    .map(|cells| profile.baked(cells, reads))
                    .and_then(|p| p.bake().cloned());
                (points, bake)
            }
            fieldform_gpu::CurveUse::Point(_) => unreachable!("nothing here attracts"),
            fieldform_gpu::CurveUse::Wall(..) => unreachable!("nothing here is a thin extrude"),
        })
        .collect()
}
