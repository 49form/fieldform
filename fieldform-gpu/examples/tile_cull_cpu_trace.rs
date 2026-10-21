// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! `trace_primary` re-run on the CPU for one pixel of a document, with the tile's own mask, to
//! find *which line* of the march loses a hit at relaxation 1.4 -- the GPU sweep can say that a
//! hit is lost and not where.
//!
//! `tile_cull_sweep.rs` found two frames of the owner's five-shell document where the masked
//! march, over-relaxed, loses a hit the unmasked march finds, and both only with the skip grid
//! on. Keinert's vouch on the kept field -- the new reading and the old one together must cover
//! the long step -- is a sound argument that a relaxed step never crosses a kept surface, so the
//! loss has to be a place where the loop does not run that argument. This file walks the loop
//! step by step with the same arithmetic in `f32`, and says what each step did.
//!
//! Arms, per pixel of the column named:
//!
//! - `U`  unmasked, relaxed, grid on -- what the viewport draws without the cull;
//! - `M`  masked, relaxed, grid on -- the arm the sweep found runs in;
//! - `P`  masked, plain, grid on -- what master ships for a live tile;
//! - `G`  masked, relaxed, grid off -- the sweep's control that showed scatter only;
//! - `V`  masked, relaxed, grid on, **a relaxed step is vouched before any hop**: the grid's own
//!   clearance at the landing point vouches it where it can, and a reading does otherwise;
//! - `UV` unmasked with the same change, since it would apply to every ray.
//!
//! Run with the sweep's own frame:
//! `cargo run --release -p fieldform-gpu --example tile_cull_cpu_trace -- <doc.fform> 47 3.6 230 196`
//! and `--frame` to march the whole 720 x 405 frame through `U`, `M`, `V` and `UV` and count.

use fieldform_field::{GraphField, ScalarField};
use fieldform_gpu::{generate_for_measurement, Camera};
use fieldform_scene::{Graph, NodeId, ParamRef};
use nalgebra::{Point3, Vector3};


const WIDTH: u32 = 720;
const HEIGHT: u32 = 405;

// The loop itself lives in the crate, shared with `cpu_march_gate.rs`, so there is one copy to
// hold to the shader and not two; `the_replica_runs_the_shader_s_numbers` pins it there.
use fieldform_gpu::cpu_march::{march, pixel_ray, ruled_out_on_the_cpu, scene_entry, Arm, Grid, Hop, V3};

fn only_body(graph: &Graph, keep: NodeId) -> Graph {
    let mut out = graph.clone();
    for root in graph.roots() {
        if root == keep {
            continue;
        }
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(node) = out.node(id) {
                stack.extend(node.inputs());
                out.remove(id);
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("a document path");
    let turn: u32 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(47);
    let away: f64 = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(3.6);
    let px: u32 = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(230);
    let py: u32 = args.get(5).and_then(|a| a.parse().ok()).unwrap_or(196);
    let lift: f64 = args.get(6).and_then(|a| a.parse().ok()).unwrap_or(0.0);
    let frame_wanted = args.iter().any(|a| a == "--frame");

    let (doc, _) = fieldform_io::open(std::path::Path::new(path)).expect("opens");
    let graph = doc.parts_ordered().first().copied().expect("a part").graph.clone();
    let bounds = match fieldform_field::part_extent(&doc, &graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("unbounded"),
    };
    // **In codegen's order, not `Graph::roots`' .** The fold -- and so the GPU's owner index and
    // the cull slots -- is built over `Document::folded_bodies`, which ranks by creation stamp;
    // `roots()` ranks by id, and the first draft of this read every slot against the wrong body.
    let roots = doc.folded_bodies(&graph);
    println!("{} bodies; bounds {:?} .. {:?}", roots.len(), bounds.0, bounds.1);
    let bodies: Vec<GraphField> = roots
        .iter()
        .map(|&r| GraphField::compile_graph(&doc, &only_body(&graph, r), 0.01).expect("compiles"))
        .collect();
    for (i, b) in bodies.iter().enumerate() {
        // Where each body is, from where its field reads nought along x from the centre.
        let probe = |p: Point3<f64>| b.distance(p);
        let c = Point3::new(150.0, 150.0, 0.0);
        println!("  body {i}: id {:?}, reads {:.1} at the scene's middle", roots[i], probe(c));
    }

    let program = generate_for_measurement(&doc, &graph, true, true).expect("generates");
    let base = program.cull_base.expect("two or more bodies");
    let slot = |body: usize, k: usize| match &program.params[base + body * 6 + k] {
        ParamRef::Literal(v) => *v,
        other => unreachable!("{other:?}"),
    };

    let centre = Point3::new(
        (bounds.0.x + bounds.1.x) * 0.5,
        (bounds.0.y + bounds.1.y) * 0.5,
        (bounds.0.z + bounds.1.z) * 0.5,
    );
    let scene_radius = (bounds.1 - bounds.0).norm() * 0.5;
    let camera = sweep_camera(bounds, turn, lift, away);
    let eye = camera.eye;
    let epsilon = camera.epsilon as f32;
    let far = camera.far as f32;
    println!(
        "turn {turn} lift {lift} away {away} {}; eye {:.1?}; epsilon {epsilon:.5}; scene radius {scene_radius:.2}",
        if camera.orthographic { "ortho" } else { "persp" },
        eye
    );

    let all = |p: V3| -> f32 {
        let q = Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
        bodies.iter().map(|b| b.distance(q) as f32).fold(3.4e38f32, f32::min)
    };
    let started = std::time::Instant::now();
    let reach = Vector3::repeat(scene_radius);
    let grid = Grid::build(centre - reach, scene_radius * 2.0, 64, &all);
    println!("grid: 64 cells of {:.3} mm, built in {:.1?}", grid.cell, started.elapsed());

    let tile = (px / 8, py / 8);
    let mut ruled = [false; 64];
    for b in 0..roots.len() {
        let (out, gaps) = ruled_out_on_the_cpu(
            &camera,
            (WIDTH, HEIGHT),
            tile,
            Point3::new(slot(b, 0), slot(b, 1), slot(b, 2)),
            Point3::new(slot(b, 3), slot(b, 4), slot(b, 5)),
        );
        ruled[b] = out;
        println!(
            "  tile {tile:?}: body {b} {} -- gaps {:.1} {:.1} {:.1} {:.1} {:.1}",
            if out { "RULED OUT" } else { "kept" },
            gaps[0], gaps[1], gaps[2], gaps[3], gaps[4]
        );
    }
    let kept = |p: V3| -> f32 {
        let q = Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
        bodies
            .iter()
            .enumerate()
            .filter(|(i, _)| !ruled[*i])
            .map(|(_, b)| b.distance(q) as f32)
            .fold(3.4e38f32, f32::min)
    };

    let arms: [(&str, Arm); 6] = [
        ("U ", Arm { field: &all, grid: Some(&grid), relax: 1.4, hop: Hop::Unvouched, settles: program.settles, exact: None }),
        ("M ", Arm { field: &kept, grid: Some(&grid), relax: 1.4, hop: Hop::Unvouched, settles: program.settles, exact: None }),
        ("P ", Arm { field: &kept, grid: Some(&grid), relax: 1.0, hop: Hop::Unvouched, settles: program.settles, exact: None }),
        ("G ", Arm { field: &kept, grid: None, relax: 1.4, hop: Hop::Unvouched, settles: program.settles, exact: None }),
        ("V ", Arm { field: &kept, grid: Some(&grid), relax: 1.4, hop: Hop::Vouched, settles: program.settles, exact: None }),
        ("UV", Arm { field: &all, grid: Some(&grid), relax: 1.4, hop: Hop::Vouched, settles: program.settles, exact: None }),
    ];

    println!();
    println!("column x={px}, rows {}..{}: depth per arm (readings), '-' is a miss", py.saturating_sub(10), py + 10);
    for y in py.saturating_sub(10)..=py + 10 {
        let (origin, dir) = pixel_ray(&camera, (WIDTH, HEIGHT), px, y);
        let Some(start) = scene_entry(&camera, origin, dir) else {
            println!("  y {y:3}: misses the scene sphere");
            continue;
        };
        let mut line = format!("  y {y:3}:");
        for (name, arm) in &arms {
            let o = march(arm, origin, dir, start, far, epsilon, 512, false);
            line += &match o.hit {
                Some(t) => format!("  {name} {t:8.3} ({:3})", o.readings),
                None => format!("  {name}        - ({:3})", o.readings),
            };
            if !o.unvouched.is_empty() {
                line += &format!("[{} unvouched hop]", o.unvouched.len());
            }
        }
        println!("{line}");
    }

    println!();
    println!("pixel ({px},{py}), the masked relaxed arm M step by step:");
    let (origin, dir) = pixel_ray(&camera, (WIDTH, HEIGHT), px, py);
    if let Some(start) = scene_entry(&camera, origin, dir) {
        let o = march(&arms[1].1, origin, dir, start, far, epsilon, 512, true);
        println!("  -> {:?} after {} readings", o.hit, o.readings);
        for (t0, reach, before, to) in &o.unvouched {
            let along = |t: f32| origin + dir * t;
            println!(
                "  unvouched hop: relaxed from t={t0:.3} (read {before:.3}, true {:.3}) by {reach:.3} to {:.3}, hopped on to {to:.3}; \
                 true distance at the landing {:.3}, kept {:.3}",
                all(along(*t0)),
                t0 + reach,
                all(along(t0 + reach)),
                kept(along(t0 + reach)),
            );
        }
        println!();
        println!("pixel ({px},{py}), the vouched arm V step by step:");
        let o = march(&arms[4].1, origin, dir, start, far, epsilon, 512, true);
        println!("  -> {:?} after {} readings", o.hit, o.readings);
        println!();
        println!("pixel ({px},{py}), the unmasked arm U step by step:");
        let o = march(&arms[0].1, origin, dir, start, far, epsilon, 512, true);
        println!("  -> {:?} after {} readings", o.hit, o.readings);
        for (t0, reach, before, to) in &o.unvouched {
            let along = |t: f32| origin + dir * t;
            println!(
                "  unvouched hop: relaxed from t={t0:.3} (read {before:.3}) by {reach:.3} to {:.3}, hopped on to {to:.3}; \
                 true distance at the landing {:.3}",
                t0 + reach,
                all(along(t0 + reach)),
            );
        }
        println!();
        println!("pixel ({px},{py}), the unmasked vouched arm UV step by step:");
        let o = march(&arms[5].1, origin, dir, start, far, epsilon, 512, true);
        println!("  -> {:?} after {} readings", o.hit, o.readings);
    }

    const NAMES: [&str; 5] = ["U", "M", "P", "V", "UV"];
    const PAIRS: [(usize, usize); 5] = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 3)];
    if frame_wanted {
        println!();
        println!("whole frame, every tile's own mask, arms U / M / P / V / UV:");
        let started = std::time::Instant::now();
        let pixels = whole_frame(&bodies, &grid, &camera, &slot, epsilon, far, program.settles);
        let mut readings = [0u64; 5];
        let mut hits = [0u64; 5];
        for p in &pixels {
            for k in 0..5 {
                readings[k] += u64::from(p[k].1);
                hits[k] += u64::from(p[k].0.is_some());
            }
        }
        for k in 0..5 {
            println!("  {:2}: {} hits, {} readings", NAMES[k], hits[k], readings[k]);
        }
        for (a, b) in PAIRS {
            let d = compare(&pixels, a, b, epsilon);
            println!(
                "  {} vs {}: {} lost, {} gained, {} moved over 1 mm, {} jitter; longest column run {} at {:?}",
                NAMES[a], NAMES[b], d.lost, d.gained, d.moved, d.jitter, d.run, d.at
            );
        }
        println!("  ({:.1?})", started.elapsed());
    }

    if args.iter().any(|a| a == "--sweep") {
        println!();
        println!("the sweep's 768 cameras on the CPU; a line per frame with a run of 4 or more in any pair:");
        let started = std::time::Instant::now();
        let mut readings = [0u64; 5];
        let mut frames_differing = [0usize; 5];
        let mut lost = [0usize; 5];
        let mut gained = [0usize; 5];
        let mut moved = [0usize; 5];
        let mut longest: [(usize, String); 5] = Default::default();
        let mut frames = 0usize;
        for turn in 0..48u32 {
            for &lift in &[-0.35_f64, 0.0, 0.3, 0.75] {
                for &away in &[0.95_f64, 1.25, 2.2, 3.6] {
                    let camera = sweep_camera(bounds, turn, lift, away);
                    let pixels = whole_frame(&bodies, &grid, &camera, &slot, epsilon, far, program.settles);
                    frames += 1;
                    for p in &pixels {
                        for k in 0..5 {
                            readings[k] += u64::from(p[k].1);
                        }
                    }
                    let mut line = String::new();
                    let mut worth_a_line = false;
                    for (i, (a, b)) in PAIRS.iter().enumerate() {
                        let d = compare(&pixels, *a, *b, epsilon);
                        if d.lost + d.gained + d.moved > 0 {
                            frames_differing[i] += 1;
                        }
                        lost[i] += d.lost;
                        gained[i] += d.gained;
                        moved[i] += d.moved;
                        if d.run > longest[i].0 {
                            longest[i] = (d.run, format!("turn {turn} lift {lift:+.2} away {away:.2} at {:?}", d.at));
                        }
                        if d.run >= 4 {
                            worth_a_line = true;
                        }
                        line += &format!(
                            "  {} vs {}: {}L {}G {}M {}j run {}",
                            NAMES[*a], NAMES[*b], d.lost, d.gained, d.moved, d.jitter, d.run
                        );
                    }
                    if worth_a_line {
                        println!(
                            "  turn {turn:2} lift {lift:+.2} away {away:.2} {}:{line}",
                            if camera.orthographic { "ortho" } else { "persp" }
                        );
                    }
                }
            }
        }
        println!("{frames} frames in {:.1?}", started.elapsed());
        for k in 0..5 {
            println!("  {:2}: {} readings in all ({:.3} of U)", NAMES[k], readings[k], readings[k] as f64 / readings[0] as f64);
        }
        for (i, (a, b)) in PAIRS.iter().enumerate() {
            println!(
                "  {} vs {}: {} frames differ in material: {} lost, {} gained, {} moved over 1 mm; longest run {} ({})",
                NAMES[*a], NAMES[*b], frames_differing[i], lost[i], gained[i], moved[i], longest[i].0, longest[i].1
            );
        }
    }
}

/// Every pixel of a frame through the five arms, with each tile's own mask: `U`, `M`, `P`, `V`
/// and `UV`, as (depth, readings).
fn whole_frame(
    bodies: &[GraphField],
    grid: &Grid,
    camera: &Camera,
    slot: &(dyn Fn(usize, usize) -> f64 + Sync),
    epsilon: f32,
    far: f32,
    // The program's `SETTLES`, so every arm runs the loop the shader was generated with.
    settles: bool,
) -> Vec<[(Option<f32>, u32); 5]> {
    let rows: Vec<u32> = (0..HEIGHT).collect();
    let results: Vec<Vec<[(Option<f32>, u32); 5]>> = std::thread::scope(|s| {
        let handles: Vec<_> = rows
            .chunks(16)
            .map(|chunk| {
                s.spawn(move || {
                    let all = |p: V3| -> f32 {
                        let q = Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
                        bodies.iter().map(|b| b.distance(q) as f32).fold(3.4e38f32, f32::min)
                    };
                    let mut out = Vec::new();
                    for &y in chunk {
                        let mut masks: Vec<[bool; 64]> = Vec::new();
                        for tx in 0..WIDTH.div_ceil(8) {
                            let mut ruled = [false; 64];
                            for b in 0..bodies.len() {
                                ruled[b] = ruled_out_on_the_cpu(
                                    camera,
                                    (WIDTH, HEIGHT),
                                    (tx, y / 8),
                                    Point3::new(slot(b, 0), slot(b, 1), slot(b, 2)),
                                    Point3::new(slot(b, 3), slot(b, 4), slot(b, 5)),
                                )
                                .0;
                            }
                            masks.push(ruled);
                        }
                        for x in 0..WIDTH {
                            let ruled = masks[(x / 8) as usize];
                            let kept = |p: V3| -> f32 {
                                let q = Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
                                bodies
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| !ruled[*i])
                                    .map(|(_, b)| b.distance(q) as f32)
                                    .fold(3.4e38f32, f32::min)
                            };
                            let (origin, dir) = pixel_ray(camera, (WIDTH, HEIGHT), x, y);
                            let mut r = [(None, 0u32); 5];
                            if let Some(start) = scene_entry(camera, origin, dir) {
                                let arms = [
                                    Arm { field: &all, grid: Some(grid), relax: 1.4, hop: Hop::Unvouched, settles, exact: None },
                                    Arm { field: &kept, grid: Some(grid), relax: 1.4, hop: Hop::Unvouched, settles, exact: None },
                                    Arm { field: &kept, grid: Some(grid), relax: 1.0, hop: Hop::Unvouched, settles, exact: None },
                                    Arm { field: &kept, grid: Some(grid), relax: 1.4, hop: Hop::Vouched, settles, exact: None },
                                    Arm { field: &all, grid: Some(grid), relax: 1.4, hop: Hop::Vouched, settles, exact: None },
                                ];
                                for (k, arm) in arms.iter().enumerate() {
                                    let o = march(arm, origin, dir, start, far, epsilon, 512, false);
                                    r[k] = (o.hit, o.readings);
                                }
                            }
                            out.push(r);
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a row chunk")).collect()
    });
    results.into_iter().flatten().collect()
}

/// How two arms differ over a frame: pixels, and the longest run down a column, as the sweep
/// measures it.
struct Diff {
    /// `a` hit and `b` missed.
    lost: usize,
    /// `a` missed and `b` hit.
    gained: usize,
    /// Both hit, more than a millimetre apart.
    moved: usize,
    /// Both hit, more than two thresholds apart but within a millimetre.
    jitter: usize,
    /// The longest column run of lost, gained or moved.
    run: usize,
    at: (u32, u32),
}

fn compare(pixels: &[[(Option<f32>, u32); 5]], a: usize, b: usize, epsilon: f32) -> Diff {
    let differs = |x: Option<f32>, y: Option<f32>| match (x, y) {
        (Some(x), Some(y)) => (x - y).abs() > 1.0,
        (None, None) => false,
        _ => true,
    };
    let mut d = Diff { lost: 0, gained: 0, moved: 0, jitter: 0, run: 0, at: (0, 0) };
    for p in pixels {
        match (p[a].0, p[b].0) {
            (Some(_), None) => d.lost += 1,
            (None, Some(_)) => d.gained += 1,
            (Some(x), Some(y)) if (x - y).abs() > 1.0 => d.moved += 1,
            (Some(x), Some(y)) if (x - y).abs() > 2.0 * epsilon => d.jitter += 1,
            _ => {}
        }
    }
    let mut longest = 0usize;
    let mut at = (0u32, 0u32);
    for x in 0..WIDTH {
        let mut run = 0usize;
        for y in 0..HEIGHT {
            let p = &pixels[(y * WIDTH + x) as usize];
            if differs(p[a].0, p[b].0) {
                run += 1;
                if run > longest {
                    longest = run;
                    at = (x, y + 1 - run as u32);
                }
            } else {
                run = 0;
            }
        }
    }
    d.run = longest;
    d.at = at;
    d
}

/// The sweep's own camera for one of its 768 frames.
fn sweep_camera(bounds: (Point3<f64>, Point3<f64>), turn: u32, lift: f64, away: f64) -> Camera {
    let centre = Point3::new(
        (bounds.0.x + bounds.1.x) * 0.5,
        (bounds.0.y + bounds.1.y) * 0.5,
        (bounds.0.z + bounds.1.z) * 0.5,
    );
    let scene_radius = (bounds.1 - bounds.0).norm() * 0.5;
    let a = f64::from(turn) * std::f64::consts::TAU / 48.0;
    let radius = 64.0 * away;
    let eye = Point3::new(
        centre.x + radius * a.cos() * lift.cos(),
        centre.y + radius * a.sin() * lift.cos(),
        centre.z + radius * lift.sin(),
    );
    let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, scene_radius), 35.0, f64::from(WIDTH) / f64::from(HEIGHT));
    camera.start_at_scene = true;
    camera.orthographic = turn % 2 == 1;
    if camera.orthographic {
        camera.half_height = scene_radius;
    }
    camera
}
