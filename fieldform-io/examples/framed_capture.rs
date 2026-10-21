// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The march's ray classes with the model framed, which V0's captures were not.
//!
//! Plan §5: V0 captured each document's view *as opened*, where the model fills under a third
//! of the frame and 63–71 % of rays miss the scene's sphere outright. "Framed, the converging
//! share grows and the balance shifts toward per-reading cost and away from empty-space
//! skipping. Recapture one framed view before fixing V5 against V6" — and "if framed views put
//! the converging share above ~50 %, V6 goes first."
//!
//! The share that decides it is of *readings*, not of rays: a ray that converges and a ray that
//! passes the far plane are not the same amount of work. Both are printed.
//!
//! The view is `Camera::framing`: the three-quarter view that fits the part's box to the frame,
//! which is what the viewport's own framing does and what every comparison example uses.
//!
//! `cargo run --release -p fieldform-io --features march-stats --example framed_capture -- <file>…`

use fieldform_gpu::{generate, Camera, CurveUse, Renderer};
use fieldform_scene::{Document, Graph, Node, ParamRef, Place};
use nalgebra::Point3;

const WIDTH: u32 = 1080;
const HEIGHT: u32 = 732;
/// Cells a side of the skip grid, nought being none, and whether the march reads the copy masks
/// the same fill records (V6).
///
/// And how far past a reading the march steps (V8): one is the plain march.
const GRIDS: [(u32, bool, f32); 9] = [
    (0, false, 1.0),
    (32, false, 1.0),
    (32, true, 1.0),
    (64, false, 1.0),
    (64, true, 1.0),
    (64, true, 1.2),
    (64, true, 1.4),
    (64, true, 1.6),
    (0, false, 1.4),
];

fn capture(label: &str, doc: &Document, graph: &Graph) {
    let Some((lo, hi)) = fieldform_field::part_extent(doc, graph).as_box() else {
        println!("| {label} | — | — | — | — | — | no finite bounds to frame |");
        return;
    };
    let program = generate(doc, graph).expect("the graph generates");
    let floor = ((hi - lo).norm() * 1e-5).max(1e-4);
    let curves: Option<Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)>> = program
        .curves
        .iter()
        .map(|use_of| match use_of {
            CurveUse::Spine(id) => Some((doc.curves[id].tessellate_to_angle(1.0, floor), None)),
            _ => None,
        })
        .collect();
    let Some(curves) = curves else {
        println!("| {label} | — | — | — | — | — | wants more than spines, which is all this supplies |");
        return;
    };
    let renderer = pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0))
        .expect("a GPU, which is where the march runs");
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    let mut camera = Camera::framing((lo, hi), f64::from(WIDTH) / f64::from(HEIGHT));
    // As the viewport marches a bounded part: from where the ray enters the scene's sphere.
    // Left off, every ray that misses the sphere is counted as one that passed the far plane.
    camera.start_at_scene = true;
    let budget = (1024.0 * fieldform_field::step_multiplier(doc, graph)).ceil() as u32;
    // The cube the skip grid covers: about the sphere the rays start on.
    let centre = Point3::from((lo.coords + hi.coords) * 0.5);
    let radius = (hi - lo).norm() * 0.5;
    let corner = centre - nalgebra::Vector3::repeat(radius);

    let maps = fieldform_gpu::EnvironmentMaps::bake(
        &fieldform_gpu::Studio::default(),
        fieldform_gpu::BakeSettings::default(),
    );
    let shading = &mut fieldform_gpu::ShadingResources::new(renderer.device(), renderer.queue(), &maps)
        .expect("the lighting pipeline builds");
    let wait = || renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("the GPU finishes");
    let mut plain: Option<Vec<fieldform_gpu::Hit>> = None;
    // The frame the grid alone drew, at each size, for the masked one to be held to: a mask
    // leaves out only copies that cannot be the minimum, so the frame is the same to the bit.
    let mut unmasked: std::collections::HashMap<u32, Vec<fieldform_gpu::Hit>> = Default::default();
    for (cells, masks, relax) in GRIDS {
        renderer.set_relaxation(relax);
        let build_ms = if cells == 0 {
            renderer.clear_skip_grid();
            0.0
        } else {
            // Twice, timing the second: the first carries whatever the card was doing before.
            renderer.build_skip_grid(corner, radius * 2.0, cells, masks);
            wait();
            let began = std::time::Instant::now();
            renderer.build_skip_grid(corner, radius * 2.0, cells, masks);
            wait();
            began.elapsed().as_secs_f64() * 1e3
        };
        let shot = renderer.march_stats(&camera, WIDTH, HEIGHT, budget).expect("the march runs");
        let mut ray_count = [0u64; 4];
        let mut readings = [0u64; 4];
        for (steps, exit) in shot.steps.iter().zip(&shot.exits) {
            ray_count[*exit as usize] += 1;
            readings[*exit as usize] += u64::from(*steps);
        }
        let all: u64 = readings.iter().sum();
        // How wide one reading is: calls of a pattern's sub-field a reading, where there is one.
        let called: u64 = shot.calls.iter().map(|c| u64::from(*c)).sum();
        let calls = if called == 0 {
            "—".to_string()
        } else {
            format!("{:.2}", called as f64 / (all as f64).max(1.0))
        };
        let share = |n: u64| 100.0 * n as f64 / (all as f64).max(1.0);
        // What the card pays for, as against what the rays ask for. The march runs in groups of
        // 8 × 8 pixels in lockstep, so a group is busy for as long as its *longest* ray is, and
        // every other ray in it waits: the cost is the sum over groups of the longest, times
        // sixty-four, not the sum over rays.
        let (w, h) = (WIDTH as usize, HEIGHT as usize);
        let mut lockstep = 0u64;
        for gy in (0..h).step_by(8) {
            for gx in (0..w).step_by(8) {
                let mut longest = 0u32;
                for y in gy..(gy + 8).min(h) {
                    for x in gx..(gx + 8).min(w) {
                        longest = longest.max(shot.steps[y * w + x]);
                    }
                }
                lockstep += u64::from(longest) * 64;
            }
        }

        // A draft's march, by the card's own timestamps: presented as the viewport presents
        // it, nothing read back. The first few are thrown away as the clocks coming up.
        let look = fieldform_gpu::Look::studio(
            fieldform_scene::Material::default(),
            camera.forward,
            camera.right,
            camera.up,
        );
        let mut frames: Vec<f64> = Vec::new();
        for round in 0..14 {
            renderer
                .present(shading, &camera, 720, 488, budget, &look, 0)
                .expect("the draft presents");
            wait();
            let mut cost = None;
            for _ in 0..50 {
                if let Some((_, c)) = shading.costs().pop() {
                    cost = c.march_ms;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
                let _ = renderer.device().poll(wgpu::PollType::Poll);
            }
            if round >= 5 {
                frames.push(f64::from(cost.expect("this measurement needs GPU timestamps")));
            }
        }
        frames.sort_by(f64::total_cmp);
        let frame_ms = frames[frames.len() / 2];

        // And the picture: the same pixels hit, at the same depth.
        let hits = renderer.render(&camera, WIDTH, HEIGHT, budget).expect("renders");
        let agreement = match &plain {
            None => "—".to_string(),
            Some(base) => {
                let differ = base.iter().zip(&hits).filter(|(a, b)| a.is_hit() != b.is_hit()).count();
                let deepest = base
                    .iter()
                    .zip(&hits)
                    .filter(|(a, b)| a.is_hit() && b.is_hit())
                    .map(|(a, b)| (a.depth - b.depth).abs())
                    .fold(0.0, f64::max);
                // Which way the disagreements go is what says whose they are: a grid that
                // skipped through a surface lands *deeper* than the plain march did.
                let both = || base.iter().zip(&hits).filter(|(a, b)| a.is_hit() && b.is_hit());
                let deeper = both().filter(|(a, b)| b.depth - a.depth > 0.05).count();
                let shallower = both().filter(|(a, b)| a.depth - b.depth > 0.05).count();
                format!(
                    "{differ} pixels differ in hit; depth within {deepest:.4} mm; past 0.05 mm: {deeper} deeper, {shallower} shallower, of {}",
                    both().count()
                )
            }
        };
        let agreement = if masks && relax <= 1.0 {
            let same = unmasked.get(&cells).is_some_and(|grid_only| *grid_only == hits);
            assert!(same, "the copy masks changed the frame, which they must not by a bit");
            format!("{agreement}; **identical to the grid alone, to the bit**")
        } else {
            agreement
        };
        if cells > 0 && !masks && relax <= 1.0 {
            unmasked.insert(cells, hits.clone());
        }
        if plain.is_none() {
            plain = Some(hits);
        }
        println!(
            "| {label} | {} | {:.2} M | {:.2} M | {:.0} % / {:.0} % | {calls} | {frame_ms:.1} ms | {build_ms:.1} ms | {agreement} |",
            {
                let grid = match (cells, masks) {
                    (0, _) => "none".to_string(),
                    (n, false) => format!("{n}³"),
                    (n, true) => format!("{n}³ and masks"),
                };
                if relax > 1.0 { format!("{grid}, stepping {relax} × the reading") } else { grid }
            },
            all as f64 / 1e6,
            lockstep as f64 / 1e6,
            share(readings[1]),
            share(readings[2]),
        );
    }
}

fn main() {
    println!("{WIDTH} × {HEIGHT} for the readings, 720 × 488 for the frame; the model framed.\n");
    println!("| scene | skip grid | field readings | readings in lockstep groups of 8 × 8 | by converging / passing rays | sub-field calls a reading | a 720 px draft march, GPU clock | filling the grid | against no grid |");
    println!("|---|---|---|---|---|---|---|---|---|");
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
                    capture(&format!("`{name}`"), &doc, &graph);
                }
            }
            Err(e) => println!("could not open {path}: {e}"),
        }
    }
    // What the operand masks are for, which none of the owner's documents happens to be: one
    // body joined out of several dear operands that are nowhere near each other. Separate
    // *bodies* would have the box guard; operands of one boolean have had nothing.
    for (label, blended) in [
        ("synthetic: eight pipes in one union", false),
        ("synthetic: the same eight, each blended to the next at 3 mm", true),
    ] {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let mut pipes = Vec::new();
        for i in 0..8 {
            let turn = std::f64::consts::TAU * f64::from(i) / 8.0;
            let (c, s) = (turn.cos(), turn.sin());
            let id = fieldform_scene::CurveId::new();
            // A bent spine out along its own spoke, so the pipes meet only near the hub.
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
            graph.add(Node::Boolean {
                op: fieldform_scene::BooleanOp::Union,
                inputs: pipes.into_iter().map(Into::into).collect(),
            });
        }
        capture(label, &doc, &graph);
    }

    // V0's floor: one sphere and nothing else.
    let mut graph = Graph::empty();
    graph.add(Node::Sphere {
        centre: Place::At([ParamRef::Literal(0.0); 3]),
        radius: ParamRef::Literal(50.0),
    });
    capture("a single sphere", &Document::new(), &graph);
}
