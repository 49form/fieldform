// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether two tessellations at once take as long as one, or as long as two.
//!
//! **Written to check a claim in backlog item 53 before it became a promise.** The item said two
//! tessellations "split the cores and finish at about the time they would have one after the
//! other", on the belief that the mesher parallelises. It mostly does not: `mesh_field_adaptive`
//! is one recursion on one thread, and only `relax` — part of "Improve triangle shape", on by
//! default — uses `rayon`. So the honest expectation was the opposite, and this measures which.
//!
//! A sphere filled with a gyroid, meshed alone and then twice at once on two threads, with and
//! without the improve step. If two at once take about as long as one, they ran side by side.
//!
//! Measured 2026-09-18, 16 logical cores, 2 162 592 triangles each:
//!
//! ```text
//!                                 one alone  two at once     ratio
//!   mesh only                        15.39s       21.13s     1.37x
//!   mesh + improve shape             50.79s       54.12s     1.07x
//! ```
//!
//! Side by side. The single mesh uses one core, which is also where the next speed-up is.
//!
//! Run with: `cargo run --release -p fieldform-field --example concurrent_tessellation`

use fieldform_field::{Cancel, MeshTolerance, Progress, ScalarField, tessellate};
use nalgebra::Point3;

/// A gyroid sheet cut to a ball — cheap to write down, and dense enough in surface that the
/// mesher has real work to do.
struct Infill;

impl ScalarField for Infill {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let k = std::f64::consts::TAU / 8.0;
        let (x, y, z) = (p.x * k, p.y * k, p.z * k);
        let g = (x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()).abs() - 0.3;
        let wall = g / (k * 1.8);
        let ball = p.coords.norm() - 20.0;
        wall.max(ball)
    }

    fn normalization(&self) -> fieldform_field::Normalization {
        fieldform_field::Normalization::LowerBound
    }
}

fn once(improve: bool) -> (f64, usize) {
    let tolerance =
        MeshTolerance { chord_mm: 0.02, angle_deg: 15.0, max_edge_mm: 0.0, min_cell_mm: 0.08 };
    let bounds = (Point3::new(-21.0, -21.0, -21.0), Point3::new(21.0, 21.0, 21.0));
    let started = std::time::Instant::now();
    let out = tessellate(
        &Infill,
        bounds,
        &tolerance,
        improve,
        false,
        &Cancel::new(),
        &Progress::new(),
    )
    .expect("not cancelled");
    (started.elapsed().as_secs_f64(), out.mesh.triangles.len())
}

fn main() {
    println!(
        "  {:<28} {:>10} {:>12} {:>9}",
        "", "one alone", "two at once", "ratio"
    );
    for improve in [false, true] {
        let (alone, triangles) = once(improve);
        let started = std::time::Instant::now();
        let a = std::thread::spawn(move || once(improve));
        let b = std::thread::spawn(move || once(improve));
        let _ = (a.join().unwrap(), b.join().unwrap());
        let both = started.elapsed().as_secs_f64();
        println!(
            "  {:<28} {:>9.2}s {:>11.2}s {:>8.2}x   ({triangles} triangles each)",
            if improve { "mesh + improve shape" } else { "mesh only" },
            alone,
            both,
            both / alone,
        );
    }
    println!(
        "\n  A ratio near 1 means the two ran side by side; near 2 means they queued for the\n  \
         same cores. Machine: {} logical cores.",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
}
