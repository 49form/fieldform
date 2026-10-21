// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A cutter that shares a face with the body it cuts, looked at along that face.
//!
//! Reported by the owner: "when a boolean is performed and two surfaces are basically identical
//! values, at certain angles we can still see a surface there — almost like z-buffer fighting".
//!
//! Subtract a box from a block so that they share a face, and the field along the shared plane
//! is `max(x, -x)`: it touches nought all across it and is negative nowhere. Nothing is there,
//! and a march that stops wherever the value is small stops on it all the same. `CONFIRM` in
//! `march.wgsl` exists for this and believes a candidate only if the value goes on falling
//! behind it — which separates a sheet from a surface when the ray crosses it, and cannot when
//! the ray runs along it.
//!
//! The block is 100 mm and the cutter takes its +X half, flush with the +X, ±Y and ±Z faces, so
//! what is left lies wholly in `x ≤ 0`. Any hit with `x` well above nought is a surface that is
//! not there. Counted over a sweep of views from square-on to the cut to nearly along it.
//!
//! `cargo run --release -p fieldform-gpu --example coincident_faces`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef, Place};
use nalgebra::{Point3, Vector3};

const WIDTH: u32 = 720;
const HEIGHT: u32 = 480;

fn main() {
    let lit = ParamRef::Literal;
    let doc = Document::new();
    let mut graph = Graph::empty();
    let block = graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        size: [lit(100.0), lit(100.0), lit(100.0)],
        turn: fieldform_scene::no_turn(),
    });
    let cutter = graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(25.0), lit(0.0), lit(0.0)]),
        size: [lit(50.0), lit(100.0), lit(100.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), cutter.into()] });

    let program = generate(&doc, &graph).expect("generates");
    let renderer = pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0))
        .expect("a GPU, which is where the march runs");
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);

    println!("{WIDTH} × {HEIGHT}; what is left of the block lies in x ≤ 0.\n");
    println!("| the view's angle off the shared +Y face | pixels that hit | hits where nothing is (x > 1 mm) |");
    println!("|---|---|---|");
    let mut phantom_total = 0usize;
    let mut hit_total = 0usize;
    let centre = Point3::new(0.0, 0.0, 0.0);
    let radius = 50.0 * 3f64.sqrt();
    for degrees in [60.0f64, 30.0, 15.0, 8.0, 4.0, 2.0, 1.0] {
        // From +X and a little +Y, looking back at the cut: the vanished half's +Y face lies
        // between the eye and the body, and the angle is how steeply the view meets it.
        let a = degrees.to_radians();
        let dir = Vector3::new(a.cos(), a.sin(), 0.35).normalize();
        let eye = centre + dir * 320.0;
        let camera = Camera::looking_at(
            eye,
            centre,
            Vector3::z(),
            (centre, radius),
            40.0,
            f64::from(WIDTH) / f64::from(HEIGHT),
        );
        let hits = renderer.render(&camera, WIDTH, HEIGHT, 1024).expect("renders");
        // Where each hit is, from the ray the camera made for it.
        let mut phantom = 0usize;
        let mut total = 0usize;
        for (i, hit) in hits.iter().enumerate() {
            if !hit.is_hit() {
                continue;
            }
            total += 1;
            let (px, py) = ((i as u32 % WIDTH) as f64 + 0.5, (i as u32 / WIDTH) as f64 + 0.5);
            let u = (px / f64::from(WIDTH) * 2.0 - 1.0) * camera.half_height * camera.aspect;
            let v = (1.0 - py / f64::from(HEIGHT) * 2.0) * camera.half_height;
            let ray = (camera.forward + camera.right * u + camera.up * v).normalize();
            let at = camera.eye + ray * hit.depth;
            if at.x > 1.0 {
                phantom += 1;
            }
        }
        phantom_total += phantom;
        hit_total += total;
        println!("| {degrees}° | {total} | **{phantom}** |");
    }
    // Not nought, and the few that are left are all in one place: on the lines where two of the
    // vanished faces met, where the look-ahead across one sheet lands in the thin zone of the
    // other and reads "still falling". A pixel or two a frame, against a third of the frame
    // before. A second look-ahead would clear them and would cost a lattice -- whose surface is
    // "still falling" at nearly every hit -- a reading a pixel, so they are left and counted.
    let allowed = hit_total / 10_000;
    println!("
{phantom_total} phantom hits of {hit_total}; no more than {allowed} allowed.");
    assert!(
        phantom_total <= allowed,
        "a surface was drawn where the cutter had taken everything: {phantom_total} of {hit_total}"
    );
}
