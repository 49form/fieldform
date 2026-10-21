// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Companion to `normal_accuracy`: does querying `CageField::closest(p).normal` directly —
//! the exact, barycentric-interpolated analytic normal from `LimitSurface::tessellate` — fix
//! what finite-differencing `CageField::distance` could not?
//!
//! `normal_accuracy` measured the old path: `CageField` only ever answered "how far", so
//! shading had to recover a normal by finite-differencing that scalar field, and the field's
//! gradient is piecewise constant across each triangle in the underlying triangulation — the
//! facet normal, however small the finite-difference step. That produced 8.99 degrees of
//! worst-case normal error at cage level 2, dropping only to 1.48 degrees at level 4, and
//! never fully vanishing at any level: `h` is fixed by mesh resolution, so it eventually
//! stops tracking triangle size altogether.
//!
//! `CageField` now stores an exact analytic normal per vertex, sourced the same way
//! `LimitSurface::tessellate_face` already did for the (unwelded) cage-editing view — so this
//! should be small at every level, not just at the levels expensive enough to shrink triangles
//! below the finite-difference step.
//!
//! Run with: `cargo run --release -p fieldform-field --example shading_normal_accuracy`

use fieldform_cage::Cage;
use fieldform_field::CageField;
use fieldform_subd::LimitSurface;
use nalgebra::{Point3, Vector3};

fn main() {
    let size = 1000.0;
    let cage = Cage::cube_of(size);

    // Exact points and exact normals on the limit surface.
    let mut surface = LimitSurface::new(&cage);
    let mut truth: Vec<(Point3<f64>, Vector3<f64>)> = Vec::new();
    for f in cage.face_ids() {
        let mut samples = Vec::new();
        surface.tessellate_face(f, 11, &mut samples).expect("valid");
        truth.extend(samples);
    }

    println!("{} exact surface points on a {size} mm cube\n", truth.len());
    println!(
        "{:>7}  {:>14}  {:>14}",
        "levels", "worst normal", "mean normal"
    );
    println!("{}", "-".repeat(41));

    for levels in 2..=5 {
        let field = CageField::from_cage(&cage, levels).expect("valid");

        let mut worst_angle: f64 = 0.0;
        let mut total_angle = 0.0;

        for (p, exact_normal) in &truth {
            let n = field.closest(*p).normal;
            let angle = n.dot(exact_normal).clamp(-1.0, 1.0).acos().to_degrees();
            worst_angle = worst_angle.max(angle);
            total_angle += angle;
        }

        println!(
            "{:>7}  {:>11.2} deg  {:>11.2} deg",
            levels,
            worst_angle,
            total_angle / truth.len() as f64
        );
    }

    println!();
    println!("Compare against normal_accuracy's finite-difference numbers: 8.99 / 4.90 / 1.48");
    println!("/ 0.18 degrees worst-case at the same four levels. closest(p).normal should stay");
    println!("small even at level 2, since it no longer depends on triangles shrinking below h.");
}
