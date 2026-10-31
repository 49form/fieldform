// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the settled bake costs, and how well it reads, as the cage grows -- the numbers a choice
//! of bake resolution *from the cage* would rest on, rather than from the display rung alone.
//!
//! `bake_cost` times the bake on the starting cube. That is one cage: it says what a cell count
//! costs and nothing about whether a cage of 6 000 or 24 000 faces changes the answer. This
//! refines the cube to those sizes and, for each resolution a rung offers, prints the field's
//! build (tessellation and BVH, at the level the app would pick for that resolution), the bake
//! itself, and how far the baked value strays from the field's own distance near the surface --
//! the figure that says what the extra cells buy.
//!
//!     cargo run -p fieldform-gpu --example bake_vs_cage
//!
//! Run it in the dev profile: that is the build a person runs, with the numeric crates optimized
//! by the workspace's `[profile.dev.package]` settings.

use fieldform_cage::Cage;
use fieldform_field::{levels_for_tolerance, CageField};
use fieldform_gpu::BakedField;
use nalgebra::Point3;
use std::time::Instant;

fn main() {
    println!("cage faces | cells | levels | field build | bake | memory | strays (99%) | strays (max) | cell size");
    let mut cage = Cage::cube_of(1000.0);
    for refinements in 0..=6 {
        if refinements > 0 {
            cage = fieldform_subd::subdivide(&cage).expect("a refined cube is a cage");
        }
        if !matches!(refinements, 2 | 4 | 5 | 6) {
            continue;
        }
        let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
        for v in cage.vertex_ids() {
            let p = cage.vertex_position(v);
            lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        let extent = hi - lo;
        let band = extent.norm() * 0.15;
        // What each size is asked for: the finest settings on the biggest cages take minutes (the
        // 6 144-face cage at 192 cells took 128 s in the first run), so the table stops short of them.
        let rungs: &[usize] = match cage.face_count() {
            0..=1_600 => &[64, 128, 192],
            _ => &[64, 96, 128],
        };
        for &cells in rungs {
            let cell = extent.x.max(extent.y).max(extent.z) / cells as f64;
            let levels = levels_for_tolerance(extent.norm(), (cell * 0.5).max(1e-4));
            let started = Instant::now();
            let field = CageField::from_cage(&cage, levels).expect("a cage subdivides");
            let built = started.elapsed();
            let started = Instant::now();
            let baked = BakedField::new(&field, (lo, hi), cells, band);
            let bake = started.elapsed();

            // How far the bake strays from the field it was taken from, at points within half the
            // band of the surface: a fixed stride through the box, so every row is read at the
            // same places.
            let mut errors = Vec::new();
            let steps = 40usize;
            for i in 0..steps {
                for j in 0..steps {
                    for k in 0..steps {
                        let f = |n: usize| (n as f64 + 0.5) / steps as f64;
                        let p = Point3::new(
                            lo.x + extent.x * f(i),
                            lo.y + extent.y * f(j),
                            lo.z + extent.z * f(k),
                        );
                        let exact = field.distance(p);
                        if exact.abs() < band * 0.5 {
                            errors.push((baked.sample(p) - exact).abs());
                        }
                    }
                }
            }
            errors.sort_by(|a, b| a.total_cmp(b));
            let pick = |q: f64| errors.get(((errors.len() as f64 * q) as usize).min(errors.len().saturating_sub(1))).copied().unwrap_or(0.0);
            println!(
                "{:>10} | {:>5} | {:>6} | {:>8.0} ms | {:>6.0} ms | {:>4} MB | {:>9.3} mm | {:>9.3} mm | {:>6.2} mm",
                cage.face_count(),
                cells,
                levels,
                built.as_secs_f64() * 1e3,
                bake.as_secs_f64() * 1e3,
                baked.size_bytes() / 1_000_000,
                pick(0.99),
                pick(1.0),
                cell,
            );
        }
    }
}
