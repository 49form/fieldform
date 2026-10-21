// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a per-tile active-body mask could buy, bounded before it is built — plan step 11.
//!
//! **The decision this is for.** `tile-pruning-spec.md` §2 is two days of work: a cull pass, a
//! workgroup mask, two folds out of codegen, and a mask load above every early return. It makes
//! the *frame* faster on scenes of many top-level bodies. But `shader_size_cost.rs` measured that
//! those same scenes are **compile**-bound first — 100 bodies is 1 826 ms to compile against a
//! frame of 17 to 127 ms, and 200 bodies is 8 148 ms. So the prize wants a number before the two
//! days, not after.
//!
//! **The oracle, and why it is a close one.** Step 11 skips a body a tile cannot contain. The same
//! skip already exists per *point*: step 4 gives each body worth guarding a box, and
//! `box_guard` refuses a body whose box is no nearer than the running minimum. So a body can be
//! made to vanish from the march **without touching the shader** — write a box far away into its
//! six slots and the existing guard skips it everywhere.
//!
//! That is what this does. The shader, the pipeline and the scene are identical in every arm; only
//! six numbers per body change. What it measures is therefore the *body evaluation* a tile cull
//! would save, with none of the confounds of a smaller shader.
//!
//! **It is a lower bound, in two named ways.** Step 11's skip is one workgroup-uniform bit where
//! this pays a box test per body per step, so a real mask saves more than this shows. And a body
//! this blanks is gone for every ray, not just the tiles that cannot see it — so the active bodies
//! are chosen spread across the grid, keeping rays terminating where they did rather than running
//! to the far plane and costing more than they should.
//!
//! **What it cannot tell you** is whether the picture is right; a blanked body is missing material.
//! That is what `masked_matches_unmasked` is for, if step 11 is built.
//!
//! **Measured on an RTX A500 Laptop GPU, Vulkan, 2026-09-26.** 720 px, least of six rounds of
//! twenty frames:
//!
//! | bodies | mean / worst a tile | all | per tile | none active | bound |
//! |---|---|---|---|---|---|
//! | 10 × 4 spheres | 1.7 / 4 | 8.7 | 7.0 | 6.8 | 1.24× |
//! | 50 × 4 spheres | 3.3 / 10 | 39.3 | 13.9 | 11.0 | **2.83×** |
//! | 100 × 4 spheres | 4.4 / 13 | 133.3 | 44.1 | 20.3 | **3.02×** |
//! | 10 × 8 spheres | 2.1 / 5 | 10.7 | 7.4 | 6.9 | 1.45× |
//! | 50 × 8 spheres | 4.3 / 15 | 44.7 | 17.7 | 11.3 | **2.53×** |
//! | 100 × 8 spheres | 5.9 / 18 | 144.0 | 49.8 | 31.9 | **2.89×** |
//!
//! **So step 11 is worth about 2.5 to 3× on the frame at 50 to 100 bodies**, and little at ten.
//! The control holds: `none active` is 20.3 ms against 133.3, so blanking does reach the guard.
//!
//! **And why it is 3× rather than 20×, which the counts alone would suggest.** A tile can contain
//! 4.4 of 100 bodies, so a cull drops about 95% of them — but `none active` is 20.3 ms, and that is
//! a floor step 11 cannot touch: the rays still cross the scene, the grid is still read, the
//! primary loop still runs. Of 133.3 ms, about 113 is body evaluation, and culling to four leaves
//! about 24 of it. That is 4.7× on the part it can reach, diluted to 3.0× by the floor. **A frame
//! prize is bounded by its floor, and the body counts do not show it.**
//!
//! Run with: `cargo run --release -p fieldform-gpu --example tile_cull_oracle`

use fieldform_gpu::{Camera, Renderer};
use fieldform_scene::{Document, Graph, Node, ParamRef, Place};
use nalgebra::{Point3, Vector3};

const RADIUS_MM: f64 = 10.0;
/// Three radii between centres, as `spread_bodies_cost` spaces them: apart, not touching.
const PITCH_MM: f64 = RADIUS_MM * 3.0;
/// One tile is one workgroup, which `march.wgsl` declares as 8 x 8.
const TILE: u32 = 8;
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const FRAMES: u32 = 20;
const ROUNDS: u32 = 6;
const EDGE: u32 = 720;

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

    // Four spheres a body and eight, which are the weights `spread_bodies_cost` measured the
    // guard winning at. One sphere a body is below `GUARD_WEIGHT`, so those bodies have no box
    // at all and this oracle cannot reach them -- said out loud below rather than left as a gap.
    for per_body in [4usize, 8] {
        for count in [10usize, 50, 100] {
            run(&device, &queue, count, per_body);
        }
    }
    println!(
        "\n`all` is every body's true box. `per tile` is the mean bodies a tile can contain, with\n\
         every other body's box written far away so the existing guard skips it. `bound` is the\n\
         ratio, and it is a **lower** bound on what step 11 would buy: a mask is one workgroup\n\
         bit where this pays a box test a body a step.\n\
         \n\
         The control is `none active`: every box blanked, so nothing can be hit. If that is not\n\
         far cheaper than `all`, the blanking is not reaching the guard and no other row here\n\
         means anything."
    );
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, count: usize, per_body: usize) {
    let (graph, (lo, hi)) = spheres(count, per_body);
    let doc = Document::new();
    let program = fieldform_gpu::generate(&doc, &graph).expect("generates");
    let boxed = program.body_boxes.iter().filter(|b| b.is_some()).count();

    let centre = nalgebra::center(&lo, &hi);
    let radius = (hi - lo).norm() * 0.5;
    let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
    let camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    let budget = (STEP_BUDGET * STEP_HEADROOM) as u32;

    // **How many bodies a tile can contain**, by the same rays the shader makes -- see
    // `primary_ray` in `camera.wgsl`, whose projection this inverts. A body counts for a tile
    // when its box's screen rectangle meets that tile, which is what a frustum test decides; a
    // body behind the eye counts for every tile, conservatively.
    let tiles = EDGE.div_ceil(TILE);
    let mut per_tile = vec![0u32; (tiles * tiles) as usize];
    let extents: Vec<(Point3<f64>, Point3<f64>)> = body_boxes(count, per_body);
    for (blo, bhi) in &extents {
        match screen_rect(&camera, *blo, *bhi, EDGE) {
            Some((x0, y0, x1, y1)) => {
                for ty in (y0 / TILE)..=(y1 / TILE).min(tiles - 1) {
                    for tx in (x0 / TILE)..=(x1 / TILE).min(tiles - 1) {
                        per_tile[(ty * tiles + tx) as usize] += 1;
                    }
                }
            }
            None => per_tile.iter_mut().for_each(|c| *c += 1),
        }
    }
    // Over the tiles that see anything: a tile of empty sky has nothing to skip and averaging it
    // in would flatter the bound.
    let busy: Vec<u32> = per_tile.iter().copied().filter(|c| *c > 0).collect();
    let mean = if busy.is_empty() {
        0.0
    } else {
        busy.iter().map(|c| f64::from(*c)).sum::<f64>() / busy.len() as f64
    };
    let worst = busy.iter().copied().max().unwrap_or(0);

    let curves = Vec::new();
    let renderer =
        Renderer::with_device(device.clone(), queue.clone(), &program, &[], &curves, &[])
            .expect("builds");

    let base: Vec<f32> = program
        .params
        .iter()
        .map(|p| match p {
            ParamRef::Literal(v) => *v as f32,
            _ => 0.0,
        })
        .collect();

    // Active bodies spread across the grid rather than taken from one end, so rays terminate
    // where they did. A block of neighbours left active would leave most of the screen empty and
    // every ray there running to the far plane, which costs more and is not what a tile cull does.
    let keep = (mean.round() as usize).clamp(1, count);
    let all = time(&renderer, &camera, &base, budget);
    let culled = time(
        &renderer,
        &camera,
        &blanked(&base, &program, |i| i * keep % count < keep),
        budget,
    );
    let none = time(
        &renderer,
        &camera,
        &blanked(&base, &program, |_| false),
        budget,
    );

    println!(
        "{count:>4} bodies x {per_body} spheres  ({boxed} boxed)  tiles: mean {mean:>5.1}, worst {worst:>3}"
    );
    println!(
        "     all {all:>7.1} ms    per tile ({keep} active) {culled:>7.1} ms    none active {none:>7.1} ms \
         -> bound {:>5.2}x",
        all / culled.max(1e-9)
    );
}

/// `base` with every body's box written far away except those `keep` admits.
fn blanked(
    base: &[f32],
    program: &fieldform_gpu::ShaderProgram,
    keep: impl Fn(usize) -> bool,
) -> Vec<f32> {
    let mut values = base.to_vec();
    for (i, slot) in program.body_boxes.iter().enumerate() {
        let Some(slot) = slot else { continue };
        if keep(i) {
            continue;
        }
        // A degenerate box a long way off. `box_guard` returns the distance to it, so every point
        // reads it as further than the running minimum and the body is never evaluated.
        for k in 0..3 {
            values[slot + k] = 1.0e6;
            values[slot + 3 + k] = 1.0e6;
        }
    }
    values
}

fn time(renderer: &Renderer, camera: &Camera, values: &[f32], budget: u32) -> f64 {
    renderer.set_params(values);
    let _ = renderer.render(camera, EDGE, EDGE, budget);
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let start = std::time::Instant::now();
        for _ in 0..FRAMES {
            renderer
                .render(camera, EDGE, EDGE, budget)
                .expect("renders");
        }
        best = best.min(start.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES));
    }
    best
}

/// The screen rectangle a world box covers, in pixels, or `None` when it reaches behind the eye.
///
/// Inverts `primary_ray`: a point's `u` is `dot(d, right) / (z * half_h * aspect)` and its `v` is
/// `dot(d, up) / (z * half_h)`, with `z = dot(d, forward)`.
fn screen_rect(
    camera: &Camera,
    lo: Point3<f64>,
    hi: Point3<f64>,
    edge: u32,
) -> Option<(u32, u32, u32, u32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for i in 0..8 {
        let corner = Point3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let d = corner - camera.eye;
        let z = d.dot(&camera.forward);
        if z <= 1e-6 {
            return None;
        }
        let u = d.dot(&camera.right) / (z * camera.half_height * camera.aspect);
        let v = d.dot(&camera.up) / (z * camera.half_height);
        let px = (u + 1.0) * 0.5 * f64::from(edge);
        let py = (1.0 - v) * 0.5 * f64::from(edge);
        x0 = x0.min(px);
        y0 = y0.min(py);
        x1 = x1.max(px);
        y1 = y1.max(py);
    }
    let last = f64::from(edge - 1);
    if x1 < 0.0 || y1 < 0.0 || x0 > last || y0 > last {
        return Some((0, 0, 0, 0));
    }
    Some((
        x0.clamp(0.0, last) as u32,
        y0.clamp(0.0, last) as u32,
        x1.clamp(0.0, last) as u32,
        y1.clamp(0.0, last) as u32,
    ))
}

/// Each body's own extent, in the order `Graph::roots` returns them — which for this graph is the
/// order they were added, since a root's id rises with insertion.
fn body_boxes(count: usize, per_body: usize) -> Vec<(Point3<f64>, Point3<f64>)> {
    let side = (count as f64).cbrt().ceil() as usize;
    (0..count)
        .map(|i| {
            let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
            let c = Point3::new(
                x as f64 * PITCH_MM,
                y as f64 * PITCH_MM,
                z as f64 * PITCH_MM,
            );
            // The body is `per_body` spheres stepped a third of a radius apart, as `spheres`
            // builds them, so its extent is the first sphere's grown by that walk.
            let walk = RADIUS_MM / 3.0 * (per_body - 1) as f64;
            (
                Point3::new(c.x - RADIUS_MM, c.y - RADIUS_MM, c.z - RADIUS_MM),
                Point3::new(c.x + RADIUS_MM + walk, c.y + RADIUS_MM, c.z + RADIUS_MM),
            )
        })
        .collect()
}

/// `count` bodies on a cubic grid, each a union of `per_body` overlapping spheres, and the box
/// they all lie in. Copied in shape from `spread_bodies_cost.rs` so the two are comparable.
fn spheres(count: usize, per_body: usize) -> (Graph, (Point3<f64>, Point3<f64>)) {
    let side = (count as f64).cbrt().ceil() as usize;
    let mut graph = Graph::empty();
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for i in 0..count {
        let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
        let c = Point3::new(
            x as f64 * PITCH_MM,
            y as f64 * PITCH_MM,
            z as f64 * PITCH_MM,
        );
        let ball = |graph: &mut Graph, at: Point3<f64>| {
            graph.add(Node::Sphere {
                centre: Place::At([
                    ParamRef::Literal(at.x),
                    ParamRef::Literal(at.y),
                    ParamRef::Literal(at.z),
                ]),
                radius: ParamRef::Literal(RADIUS_MM),
            })
        };
        if per_body > 1 {
            let parts: Vec<_> = (0..per_body)
                .map(|k| {
                    ball(
                        &mut graph,
                        Point3::new(c.x + k as f64 * RADIUS_MM / 3.0, c.y, c.z),
                    )
                })
                .collect();
            graph.add(Node::Boolean {
                op: fieldform_scene::BooleanOp::Union,
                inputs: parts.into_iter().map(Into::into).collect(),
            });
        } else {
            ball(&mut graph, c);
        }
        lo = Point3::new(lo.x.min(c.x), lo.y.min(c.y), lo.z.min(c.z));
        hi = Point3::new(hi.x.max(c.x), hi.y.max(c.y), hi.z.max(c.z));
    }
    let pad = Vector3::repeat(RADIUS_MM * 2.0);
    (graph, (lo - pad, hi + pad))
}
