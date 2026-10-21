// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Step 1 of getting off the mesher: render a lattice on the GPU, with nothing stored.
//!
//! `gyroid_spike` established that a metre of gyroid traces in 0.65 s at 1920² on the CPU while
//! being 6.25e9 triangles if meshed. This asks the next question — whether the vendored
//! `fidget-wgpu` can do the same work on the GPU, and how fast.
//!
//! Nothing here goes through [`ScalarField`](fieldform_field::ScalarField). The whole point is
//! that a lattice is closed-form: it compiles to a Fidget tape and evaluates on the GPU with no
//! CPU callback, no per-point variable, and no geometry stored anywhere. A cage field cannot do
//! that — `FieldTerm::sample` answers per point on the CPU, and the GPU generates its own
//! sample points — which is why baking cages is step 2 rather than optional.
//!
//! # This renderer does not sphere-trace
//!
//! Worth knowing, because it changes what the field has to guarantee. Fidget's voxel path
//! subdivides tiles using interval arithmetic and prunes the tape as it descends, rather than
//! marching rays. So the Lipschitz bound that `gyroid_spike` had to be careful about does not
//! apply here at all: nothing steps along a ray, so nothing can step through a wall. What the
//! renderer needs is a correct *sign* and *interval bounds* that are tight enough to prune —
//! and Fidget's interval `sin`/`cos` are quadrant-aware, so a trig lattice bounds well.
//!
//! That means the raw `|F| - c` form is used directly, without dividing by `sqrt(6) k`. The
//! division scales the value but does not move the zero set, so the surface is identical.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example gpu_lattice [-- <resolution>]`

use fidget_core::context::Tree;
use fidget_core::vm::VmShape;
use fidget_wgpu::voxel::{RenderConfig, RenderSize};
use nalgebra::{Matrix4, Rotation3};
use std::time::Instant;

/// Cells across the block. Rendering happens in a [-1, 1] cube, so this is what sets the
/// lattice period rather than any millimetre figure — 20 cells across a 1 m block is the
/// 50 mm case from `gyroid_spike`.
const CELLS: f64 = 20.0;

/// Wall thickness as a fraction of the block, matching 1 mm in 1000 mm.
const WALL_FRACTION: f64 = 0.001;

/// Typical |grad F| at the surface as a multiple of `k`, as measured in `gyroid_spike`.
const SURFACE_GRADIENT_FACTOR: f64 = 1.5;

fn main() {
    let sizes: Vec<u32> = match std::env::args().nth(1).and_then(|a| a.parse().ok()) {
        Some(n) => vec![n],
        None => vec![256, 512, 1024],
    };

    let gpu = match pollster::block_on(fidget_wgpu::Gpu::init()) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU available: {e}");
            eprintln!("this example needs a working wgpu adapter; nothing else here does.");
            return;
        }
    };

    let ctx = fidget_wgpu::voxel::Context::new(&gpu);
    let mut buffers = ctx.buffers();
    let mut out = ctx.image_buffer();

    // A three-quarter view, so faces are seen at a glance where any sampling artefact shows
    // worst. Built directly rather than through fidget-gui's View3, to avoid the dependency.
    let world_to_model: Matrix4<f32> =
        Rotation3::from_euler_angles(0.55f32, 0.9, 0.0).to_homogeneous();

    // A sphere alongside the lattice, as the control. Fidget's voxel path prunes with interval
    // arithmetic, and a lattice is the worst case for pruning: its surface is everywhere, so
    // almost no tile can be discarded. A sphere is the ordinary case, where most of the volume
    // is provably empty. Rendering both separates "the GPU path is slow" from "this model is
    // hostile to the thing that makes it fast", and those have completely different answers.
    let (sx, sy, sz) = Tree::axes();
    let sphere = (sx.square() + sy.square() + sz.square()).sqrt() - Tree::constant(0.85);

    for (label, tree) in [("sphere", sphere), ("lattice", lattice_block())] {
        let shape = VmShape::from(tree);
        let render_shape = match gpu.shape(&shape) {
            Ok(s) => s,
            Err(e) => {
                // Reported precisely rather than as a generic failure: a tape too long for the
                // GPU buffer is a real ceiling on model complexity, and knowing where it sits
                // matters more than this one example working.
                eprintln!("{label}: could not be prepared for the GPU: {e}");
                continue;
            }
        };

        println!("\n{label}: {} tape operations", shape.size());
        println!("  pixels       first       warm         fps");
        for &size in &sizes {
            let config = RenderConfig {
                image_size: RenderSize::from(size),
                world_to_model,
            };

            let start = Instant::now();
            let image = ctx
                .run(&render_shape, &mut buffers, &mut out, config.clone())
                .expect("the GPU render succeeds");
            let first = start.elapsed().as_secs_f64();

            // The first call pays for pipeline and buffer setup. A second is what a viewport
            // would actually see frame to frame.
            let start = Instant::now();
            let image = ctx
                .run(&render_shape, &mut buffers, &mut out, config)
                .unwrap_or(image);
            let warm = start.elapsed().as_secs_f64();

            println!(
                "  {size:>5}^2  {:>7.1} ms  {:>7.1} ms   {:>7.1}",
                first * 1000.0,
                warm * 1000.0,
                1.0 / warm,
            );
            write_png(&format!("gpu-{label}-{size}.png"), &image, size);
        }
    }

    println!(
        "\nNothing was stored. The lattice is 6.25e9 triangles and ~225 GB if meshed\n\
         (measured in gyroid_spike); the tape above is what was uploaded instead.\n\
         Note this path renders a voxel *column* per pixel, so an N^2 image is an N^3\n\
         evaluation before pruning — which is why the sphere and the lattice differ."
    );
}

/// A gyroid wall, intersected with a block, as closed-form Fidget math.
///
/// This is the whole model: no cage, no per-point variable, no geometry. Everything below is
/// operations on `x`, `y` and `z`.
fn lattice_block() -> Tree {
    let (x, y, z) = Tree::axes();

    // One full period per cell across a domain running -1 to 1.
    let k = std::f64::consts::TAU / (2.0 / CELLS);
    let kt = Tree::constant(k);
    let (sx, sy, sz) = (
        (x.clone() * kt.clone()).sin(),
        (y.clone() * kt.clone()).sin(),
        (z.clone() * kt.clone()).sin(),
    );
    let (cx, cy, cz) = (
        (x.clone() * kt.clone()).cos(),
        (y.clone() * kt.clone()).cos(),
        (z.clone() * kt).cos(),
    );

    let f = sx.clone() * cy.clone() + sy * cz.clone() + sz * cx;

    // Half-thickness in |F| units: near the surface F changes at about
    // SURFACE_GRADIENT_FACTOR * k per unit of distance, so a wall of `t` needs |F| < 1.5 k t/2.
    // The domain is 2 units across, so a wall of WALL_FRACTION of the block is 2 * that.
    let wall = 2.0 * WALL_FRACTION;
    let c = SURFACE_GRADIENT_FACTOR * k * wall * 0.5;
    let sheet = f.abs() - Tree::constant(c);

    // Intersect with a block inset from the render cube, so the lattice is a finite solid with
    // visible faces rather than something running off every edge of the frame.
    let half = 0.85;
    let block = x.abs().max(y.abs()).max(z.abs()) - Tree::constant(half);

    sheet.max(block)
}

/// Shades the geometry buffer and writes a PNG.
///
/// `depth == 0` is an empty pixel. Everything else is lit with the same Lambert term the CPU
/// examples use, so the two sets of images can be set side by side.
fn write_png(name: &str, image: &fidget_raster::voxel::Image, size: u32) {
    const BACKGROUND: [u8; 3] = [24, 26, 30];
    let light = nalgebra::Vector3::new(0.35f32, 0.6, 1.0).normalize();

    let mut buf = Vec::with_capacity((size * size) as usize * 3);
    for pixel in image.iter() {
        if pixel.depth == 0 {
            buf.extend_from_slice(&BACKGROUND);
            continue;
        }
        let n = nalgebra::Vector3::from(pixel.normal);
        let n = n.try_normalize(1e-9).unwrap_or_else(nalgebra::Vector3::z);
        let shade = 0.15 + 0.85 * n.dot(&light).max(0.0);
        let v = (shade.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        buf.extend_from_slice(&[v, v, v]);
    }

    image::save_buffer(name, &buf, size, size, image::ColorType::Rgb8).expect("writable path");
    println!("  wrote {name}");
}
