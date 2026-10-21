// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How far a `CageField` sits from the true limit surface, per subdivision level.
//!
//! `CageField` answers distance queries against a *tessellation* of the limit surface, so its
//! zero level set is only as round as that tessellation. The app's field view evaluates the
//! limit surface exactly, but a composed field goes through `CageField` — so the two can
//! disagree, and a cube can look smoothly round in one and faceted in the other.
//!
//! This measures the disagreement directly: sample the exact limit surface, ask the field how
//! far those points are from its own surface, and report the error. Zero would mean the field
//! agrees with the truth.
//!
//! Run with: `cargo run --release -p fieldform-field --example cage_field_accuracy`

use fieldform_cage::Cage;
use fieldform_field::{CageField, mesh_field};
use fieldform_subd::LimitSurface;
use nalgebra::Point3;

fn main() {
    let size = 1000.0; // the app's starting cube, in millimetres
    let cage = Cage::cube_of(size);

    // Ground truth: points that lie exactly on the limit surface.
    let mut surface = LimitSurface::new(&cage);
    let mut truth = Vec::new();
    // 11 samples = 10 intervals per edge. Deliberately not a power of two: a 9-sample grid
    // lands exactly on the vertices of a level-3 subdivision, so truth and approximation
    // coincide by construction and every level from 3 up reports a perfect zero. That is a
    // property of the sampling, not of the field.
    let n = 11;
    for f in cage.face_ids() {
        let mut samples = Vec::new();
        surface.tessellate_face(f, n, &mut samples).expect("valid");
        truth.extend(samples.into_iter().map(|(p, _)| p));
    }
    println!(
        "{} exact points on the limit surface of a {size} mm cube\n",
        truth.len()
    );

    println!(
        "{:>7}  {:>10}  {:>12}  {:>12}  {:>10}  {:>12}",
        "levels", "triangles", "worst error", "mean error", "build", "mesh @64"
    );
    println!("{}", "-".repeat(88));

    for levels in 1..=6 {
        let started = std::time::Instant::now();
        let field = CageField::from_cage(&cage, levels).expect("valid");
        let build = started.elapsed();

        let mut worst: f64 = 0.0;
        let mut total = 0.0;
        for p in &truth {
            // A point on the true surface should read zero. Whatever it reads instead is how
            // far this field's surface is from the real one, at that point.
            let e = field.distance(*p).abs();
            worst = worst.max(e);
            total += e;
        }

        // What it actually costs in the app: meshing the field at the default resolution.
        let half = size * 0.75;
        let t = std::time::Instant::now();
        let mesh = mesh_field(
            &field,
            (
                Point3::new(-half, -half, -half),
                Point3::new(half, half, half),
            ),
            64,
        );
        let mesh_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{:>7}  {:>10}  {:>9.2} mm  {:>9.2} mm  {:>8.0} ms  {:>9.0} ms ({} tris)",
            levels,
            field.triangle_count(),
            worst,
            total / truth.len() as f64,
            build.as_secs_f64() * 1000.0,
            mesh_ms,
            mesh.triangle_count()
        );
    }

    println!();
    println!(
        "Error is a fraction of the model: on a {size} mm cube, 1% is {} mm.",
        size * 0.01
    );
    println!("The app's composed fields use DEFAULT_CAGE_LEVELS.");
}
