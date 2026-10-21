// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the traced viewport's cage bake costs, in whatever profile this was built with.
//!
//! The bake is the one operation that blocks the UI, so it decides whether a build is usable.
//! Unoptimized it is ~70x slower, which is the difference between a viewport that settles in a
//! second and one that looks like it has hung — and that difference is why the workspace's
//! `[profile.dev.package]` settings compile the numeric crates optimized even in a dev build.
//!
//! Run in both to see it:
//! `cargo run -p fieldform-gpu --example bake_cost`
//! `cargo run --release -p fieldform-gpu --example bake_cost`

use fieldform_cage::Cage;
use fieldform_field::{levels_for_tolerance, CageField};
use fieldform_gpu::BakedField;
use nalgebra::Point3;

fn main() {
    // The app's starting cage and its Mesh setting, so these are the numbers a user pays.
    let cage = Cage::cube_of(1000.0);
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    let extent = hi - lo;
    let cell = extent.x.max(extent.y).max(extent.z) / 64.0;
    let levels = levels_for_tolerance(extent.norm(), (cell * 0.5).max(1e-4));

    let start = std::time::Instant::now();
    let field = CageField::from_cage(&cage, levels).expect("a cube subdivides");
    println!("cage field at {levels} levels: {:?}", start.elapsed());

    let band = extent.norm() * 0.15;
    // 32 is what a drag pays per frame; the rest are what the View > Display resolution
    // setting can settle at, so the numbers a user is choosing between are the numbers here.
    for resolution in [32usize, 64, 128, 192, 256] {
        let start = std::time::Instant::now();
        let baked = BakedField::new(&field, (lo, hi), resolution, band);
        println!(
            "bake at {resolution}: {:?}  ({} MB)",
            start.elapsed(),
            baked.size_bytes() / 1_000_000
        );
    }
}
