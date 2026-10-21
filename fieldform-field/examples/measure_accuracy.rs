// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What measuring a field costs, and what it buys, at each resolution.
//!
//! Mass properties come from meshing the field and integrating the mesh, so the accuracy is the
//! mesher's and the cost is the mesher's. Both are worth knowing before a number from this is
//! quoted at anybody: a volume is the kind of figure that gets copied into a quote, and "within
//! a percent" is a different claim from "exact".
//!
//! Measured against a sphere, whose volume and area are known in closed form.
//!
//! `cargo run --release -p fieldform-field --example measure_accuracy`

use fieldform_field::{MassProperties, Sphere};
use nalgebra::{Point3, Vector3};

fn main() {
    const R: f64 = 20.0;
    let sphere = Sphere {
        centre: Point3::origin(),
        radius: R,
    };
    let bounds = (
        Point3::from(Vector3::repeat(-30.0)),
        Point3::from(Vector3::repeat(30.0)),
    );

    let exact_volume = 4.0 / 3.0 * std::f64::consts::PI * R.powi(3);
    let exact_area = 4.0 * std::f64::consts::PI * R * R;

    println!("A 40 mm sphere: {exact_volume:.1} mm3, {exact_area:.1} mm2.\n");
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}  {:>9}",
        "cells", "volume err", "area err", "centre err", "time"
    );

    for n in [16usize, 32, 48, 64, 96, 128, 192] {
        let started = std::time::Instant::now();
        let m = MassProperties::of(&sphere, bounds, n).expect("a sphere has a surface");
        let secs = started.elapsed().as_secs_f64();
        println!(
            "{n:>6}  {:>9.3}%  {:>9.3}%  {:>8.4}mm  {secs:>8.3}s",
            (m.volume_mm3 - exact_volume).abs() / exact_volume * 100.0,
            (m.surface_area_mm2 - exact_area).abs() / exact_area * 100.0,
            (m.centre_of_mass - Point3::origin()).norm(),
        );
    }

    println!(
        "\nThe volume error should fall by about four each time the cell count doubles: the\n\
         mesher places each vertex by interpolating along an edge, so the surface it produces is\n\
         wrong to second order rather than to first. Counting samples inside instead would fall\n\
         off by two, which is the reason this measures a mesh and not a grid."
    );
}
