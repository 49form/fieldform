// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How far `tile_cull`'s five planes sit from where they should, in the shader's `f32`, and how
//! that compares with the pad that is supposed to cover it.
//!
//! **What this measures and why it exists.** `cpu_march_gate` holds the GPU's tile mask to the
//! CPU's `f64` planes and found one tile on the 8-shell scene keeping a body the CPU ruled out by
//! 0.12 mm -- too far to be rounding on a dot product, and it was not: the plane's second vector
//! was written `b.origin + b.dir - a.origin`, which under perspective adds a unit vector to the
//! eye's coordinates and subtracts them again, so the direction came back rounded at the eye's
//! scale (an ulp of 3e-5 at 600 mm) and the cross product of two near-parallel rays divided
//! that by the hundredth of a radian between them. The error grows with the eye's distance from
//! the origin, and the pad does not: `CULL_PAD_FLOOR_MM` is 0.5 mm, and the fraction is of the
//! body, not of the scene. So the question is at what scene size the planes' error reaches the
//! pad -- that is where the mask starts deleting material at tile edges.
//!
//! Both spellings are run here in `f32` against the `f64` replica, over every tile and every body
//! of the 8-shell scene at three scales, from the gate's perspective camera and its orthographic
//! one. The numbers go in `tile-mask-design-01941c5.md` §1.7e.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example cull_plane_precision`

use fieldform_gpu::cpu_march::ruled_out_on_the_cpu;
use fieldform_gpu::Camera;
use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef, Place};
use nalgebra::{Point3, Vector3};

const WIDTH: u32 = 720;
const HEIGHT: u32 = 405;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// The gate's shells scene, scaled.
fn shells(count: usize, pitch: f64, radius: f64, wall: f64) -> Graph {
    let mut graph = Graph::empty();
    let side = (count as f64).cbrt().ceil() as usize;
    for k in 0..count {
        let (x, y, z) = (k % side, (k / side) % side, k / (side * side));
        let at = [lit(x as f64 * pitch), lit(y as f64 * pitch), lit(z as f64 * pitch)];
        let outer = graph.add(Node::Sphere { centre: Place::At(at.clone()), radius: lit(radius) });
        let inner = graph.add(Node::Sphere { centre: Place::At(at), radius: lit(radius - wall) });
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![outer.into(), inner.into()] });
    }
    graph
}

/// The gate's camera for a turn.
fn camera_for(bounds: (Point3<f64>, Point3<f64>), turn: u32, turns: u32, lift: f64, away: f64) -> Camera {
    let centre = nalgebra::center(&bounds.0, &bounds.1);
    let scene_radius = (bounds.1 - bounds.0).norm() * 0.5;
    let a = turn as f64 * std::f64::consts::TAU / turns as f64;
    let radius = scene_radius * away;
    let eye = Point3::new(
        centre.x + radius * a.cos() * lift.cos(),
        centre.y + radius * a.sin() * lift.cos(),
        centre.z + radius * lift.sin(),
    );
    let mut camera =
        Camera::looking_at(eye, centre, Vector3::z(), (centre, scene_radius), 35.0, WIDTH as f64 / HEIGHT as f64);
    camera.start_at_scene = true;
    camera.orthographic = turn % 2 == 1;
    if camera.orthographic {
        camera.half_height = scene_radius;
    }
    camera
}

type F3 = Vector3<f32>;

fn f3(v: Vector3<f64>) -> F3 {
    F3::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Which way the plane's second vector is written.
#[derive(Clone, Copy)]
enum Spelling {
    /// `b.origin + b.dir - a.origin`: the direction rounded at the eye's scale.
    DirectionFirst,
    /// `(b.origin - a.origin) + b.dir`: as `tile_cull` has it now.
    OriginsFirst,
}

/// `tile_cull` in `f32`, step for step, with the second vector spelt either way: the five gaps
/// in millimetres, signed as the shader signs them.
fn gaps_in_f32(camera: &Camera, tile: (u32, u32), lo: F3, hi: F3, spelling: Spelling) -> [f32; 5] {
    let eye = F3::new(camera.eye.x as f32, camera.eye.y as f32, camera.eye.z as f32);
    let (forward, right, up) = (f3(camera.forward), f3(camera.right), f3(camera.up));
    let half_h = camera.half_height as f32;
    let aspect = camera.aspect as f32;
    let ray = |u: f32, v: f32| -> (F3, F3) {
        let across = right * (u * half_h * aspect) + up * (v * half_h);
        if camera.orthographic {
            (eye + across, forward.normalize())
        } else {
            (eye, (forward + across).normalize())
        }
    };
    let second = |from: (F3, F3), to: (F3, F3)| -> F3 {
        match spelling {
            Spelling::DirectionFirst => to.0 + to.1 - from.0,
            Spelling::OriginsFirst => (to.0 - from.0) + to.1,
        }
    };
    let (width, height) = (WIDTH as f32, HEIGHT as f32);
    let (tx, ty, t) = (tile.0 as f32, tile.1 as f32, 8.0f32);
    let pad = 1.0f32;
    let u0 = (tx * t - pad) / width * 2.0 - 1.0;
    let u1 = ((tx + 1.0) * t + pad) / width * 2.0 - 1.0;
    let v0 = 1.0 - (ty * t - pad) / height * 2.0;
    let v1 = 1.0 - ((ty + 1.0) * t + pad) / height * 2.0;
    let corner = [ray(u0, v0), ray(u1, v0), ray(u1, v1), ray(u0, v1)];
    let middle = ray((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    let mut gaps = [0.0f32; 5];
    for k in 0..5 {
        let (n, through) = if k < 4 {
            let a = corner[k];
            let b = corner[(k + 1) % 4];
            let mut n = a.1.cross(&second(a, b));
            let away = corner[(k + 2) % 4];
            if n.dot(&second(a, away)) < 0.0 {
                n = -n;
            }
            (n, a.0)
        } else {
            (middle.1, middle.0)
        };
        let far = F3::new(
            if n.x > 0.0 { hi.x } else { lo.x },
            if n.y > 0.0 { hi.y } else { lo.y },
            if n.z > 0.0 { hi.z } else { lo.z },
        );
        gaps[k] = n.dot(&(far - through)) / n.norm();
    }
    gaps
}

fn main() {
    println!("| scene | eye from origin | camera | pad | direction first: worst plane error | origins first: worst plane error |");
    println!("|---|---|---|---|---|---|");
    for scale in [1.0, 10.0, 100.0] {
        let (pitch, radius, wall) = (300.0 * scale, 50.0 * scale, 2.0 * scale);
        let doc = Document::new();
        let graph = shells(8, pitch, radius, wall);
        let bounds = match fieldform_field::part_extent(&doc, &graph) {
            fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
            _ => panic!("bounded"),
        };
        // The cull's extents, padded as `cull_extent_slots` pads them.
        let pad = (radius * 2.0 * 3f64.sqrt() * 1e-3).max(0.5);
        let side = 2usize;
        let extents: Vec<(Point3<f64>, Point3<f64>)> = (0..8)
            .map(|k| {
                let (x, y, z) = (k % side, (k / side) % side, k / (side * side));
                let c = Vector3::new(x as f64, y as f64, z as f64) * pitch;
                (Point3::from(c - Vector3::repeat(radius + pad)), Point3::from(c + Vector3::repeat(radius + pad)))
            })
            .collect();
        for turn in [4u32, 1] {
            let camera = camera_for(bounds, turn, 6, 0.0, if turn == 4 { 2.4 } else { 1.3 });
            let mut worst = [0.0f64; 2];
            for ty in 0..HEIGHT.div_ceil(8) {
                for tx in 0..WIDTH.div_ceil(8) {
                    for (lo, hi) in &extents {
                        let (_, exact) = ruled_out_on_the_cpu(&camera, (WIDTH, HEIGHT), (tx, ty), *lo, *hi);
                        for (i, spelling) in [Spelling::DirectionFirst, Spelling::OriginsFirst].into_iter().enumerate() {
                            let got = gaps_in_f32(&camera, (tx, ty), f3(lo.coords), f3(hi.coords), spelling);
                            for k in 0..5 {
                                worst[i] = worst[i].max((f64::from(got[k]) - exact[k]).abs());
                            }
                        }
                    }
                }
            }
            println!(
                "| 8 shells ×{scale} | {:.0} mm | {} | {pad:.2} mm | {:.4} mm{} | {:.4} mm{} |",
                camera.eye.coords.norm(),
                if camera.orthographic { "ortho" } else { "persp" },
                worst[0],
                if worst[0] > pad { " **past the pad**" } else { "" },
                worst[1],
                if worst[1] > pad { " **past the pad**" } else { "" },
            );
        }
    }
}
