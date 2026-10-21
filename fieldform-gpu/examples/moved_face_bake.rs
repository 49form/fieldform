// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What moving one cage face does to the cost of baking it.
//!
//! **Written from a report of the viewport crawling after a face is moved** (owner, 2026-09-19).
//! The application's slow-frame line named the stage: `baking Cage(…) at 32 — 745 ms`, where a
//! draft bake of the starting cage is 14.5 ms (`bake_cost.rs`). Nothing else in that frame
//! reached a millisecond, including building the field that is being sampled. So the question is
//! what a moved face costs a sample, and this asks it directly.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example moved_face_bake`

use fieldform_cage::Cage;
use fieldform_field::{CageField, levels_for_tolerance};
use nalgebra::{Point3, Vector3};

fn main() {
    const SIZE: f64 = 1000.0;
    const RESOLUTION: usize = 32;

    println!(
        "  {:<28} {:>7} {:>10} {:>12} {:>14}",
        "cage", "levels", "build", "bake at 32", "per sample"
    );

    for lift in [0.0f64, 1.0, 10.0, 100.0, 500.0, 1000.0] {
        let mut cage = Cage::cube_of(SIZE);
        // The top face, as a face drag moves it: its four corners, together.
        let top: Vec<_> = cage
            .vertex_ids()
            .filter(|v| (cage.vertex_position(*v).z - SIZE * 0.5).abs() < 1e-9)
            .collect();
        for v in &top {
            let p = cage.vertex_position(*v);
            cage.set_vertex_position(*v, p + Vector3::new(0.0, 0.0, lift));
        }

        let (lo, hi) = box_of(&cage);
        let extent = (hi - lo).norm();
        // The application's own numbers: the viewport's tolerance, and the band it bakes.
        let tolerance = (extent / 64.0 * 0.5).max(1e-4);
        let levels = levels_for_tolerance(extent, tolerance);
        let band = extent * 0.15;

        let t = std::time::Instant::now();
        let field = CageField::from_cage(&cage, levels).expect("subdivides");
        let build_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let baked = fieldform_gpu::BakedField::new(&field, (lo, hi), RESOLUTION, band);
        let bake_ms = t.elapsed().as_secs_f64() * 1e3;
        let samples = baked.dims[0] * baked.dims[1] * baked.dims[2];

        println!(
            "  top face lifted {lift:>6.0} mm  {levels:>7} {build_ms:>8.1} ms {bake_ms:>9.0} ms \
             {:>11.1} \u{00b5}s",
            bake_ms * 1e3 / samples as f64
        );
    }

    // And one sample at a time, to see where in the box the cost is.
    let mut cage = Cage::cube_of(SIZE);
    let top: Vec<_> = cage
        .vertex_ids()
        .filter(|v| (cage.vertex_position(*v).z - SIZE * 0.5).abs() < 1e-9)
        .collect();
    for v in &top {
        let p = cage.vertex_position(*v);
        cage.set_vertex_position(*v, p + Vector3::new(0.0, 0.0, 500.0));
    }
    let (lo, hi) = box_of(&cage);
    let extent = (hi - lo).norm();
    let levels = levels_for_tolerance(extent, (extent / 64.0 * 0.5).max(1e-4));
    let field = CageField::from_cage(&cage, levels).expect("subdivides");
    println!("\n  where the time goes, one reading at a time:");
    for (name, p) in [
        ("well outside", Point3::new(hi.x * 2.0, hi.y * 2.0, hi.z * 2.0)),
        ("just outside a face", Point3::new(0.0, 0.0, hi.z + extent * 0.01)),
        ("on the surface", Point3::new(0.0, 0.0, hi.z)),
        ("inside", Point3::origin()),
    ] {
        let t = std::time::Instant::now();
        let mut v = 0.0;
        for _ in 0..200 {
            v += field.distance(p);
        }
        println!(
            "    {name:<22} {:>8.1} \u{00b5}s per reading (value {:.2} mm)",
            t.elapsed().as_secs_f64() * 1e6 / 200.0,
            v / 200.0
        );
    }
}

fn box_of(cage: &Cage) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = lo.inf(&p);
        hi = hi.sup(&p);
    }
    (lo, hi)
}
