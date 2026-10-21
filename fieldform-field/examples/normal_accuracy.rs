// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where the faceting in a composed view actually comes from.
//!
//! Two candidates, and they need different fixes:
//!
//! 1. **Vertex placement.** Surface Nets puts one vertex per cell, so the surface is
//!    quantised at cell scale. Fixed by more cells.
//! 2. **Normals.** A composed field's shading normal is the gradient of `CageField`, which
//!    measures distance to a *triangulation*. That gradient is piecewise constant across each
//!    triangle and jumps at the seams, so the shading shows the triangulation however fine the
//!    mesh is. Not fixed by more cells.
//!
//! The plain field view has the same coarse geometry and looks smooth, because its normals
//! come from the limit surface analytically. That is the hint this example checks.
//!
//! Run with: `cargo run --release -p fieldform-field --example normal_accuracy`

use fieldform_cage::Cage;
use fieldform_field::{CageField, ScalarField, mesh_field};
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
        "{:>7}  {:>12}  {:>14}  {:>14}",
        "levels", "worst pos", "worst normal", "mean normal"
    );
    println!("{}", "-".repeat(56));

    for levels in 2..=5 {
        let field = CageField::from_cage(&cage, levels).expect("valid");

        // The mesher's step at resolution 64 over ~1.5 m of bounds, which is the finite
        // difference it actually uses for shading normals.
        let h = (size * 1.5 / 64.0) * 0.5;

        let mut worst_pos: f64 = 0.0;
        let mut worst_angle: f64 = 0.0;
        let mut total_angle = 0.0;

        for (p, exact_normal) in &truth {
            worst_pos = worst_pos.max(field.distance(*p).abs());

            let d = |o: Vector3<f64>| field.distance(*p + o);
            let gradient = Vector3::new(
                d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
                d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
                d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
            );
            let Some(n) = gradient.try_normalize(1e-12) else {
                continue;
            };
            let angle = n.dot(exact_normal).clamp(-1.0, 1.0).acos().to_degrees();
            worst_angle = worst_angle.max(angle);
            total_angle += angle;
        }

        println!(
            "{:>7}  {:>9.2} mm  {:>11.2} deg  {:>11.2} deg",
            levels,
            worst_pos,
            worst_angle,
            total_angle / truth.len() as f64
        );
    }

    // --- control: an exact field, so any error left is the mesher's alone ---
    struct Sphere {
        r: f64,
    }
    impl ScalarField for Sphere {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.coords.norm() - self.r
        }
    }

    let r = size * 0.5;
    let half = size * 0.75;
    let mesh = mesh_field(
        &Sphere { r },
        (
            Point3::new(-half, -half, -half),
            Point3::new(half, half, half),
        ),
        64,
    );
    let mut worst_pos: f64 = 0.0;
    let mut worst_angle: f64 = 0.0;
    let mut total_angle = 0.0;
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        worst_pos = worst_pos.max((p.coords.norm() - r).abs());
        let exact = p.coords.normalize();
        let a = n.dot(&exact).clamp(-1.0, 1.0).acos().to_degrees();
        worst_angle = worst_angle.max(a);
        total_angle += a;
    }
    println!(
        "
Control - Surface Nets on an EXACT sphere field, same resolution 64:"
    );
    println!(
        "  worst position {:.2} mm, worst normal {:.2} deg, mean normal {:.2} deg",
        worst_pos,
        worst_angle,
        total_angle / mesh.positions.len() as f64
    );
    println!("  (whatever error remains here is the mesher's; the rest is the field's)");

    println!();
    println!("Shading shows normal error, not position error. A degree or two is invisible;");
    println!("ten degrees reads as a facet edge however fine the mesh underneath is.");
}
