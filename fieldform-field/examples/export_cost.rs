// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What an STL export costs, on the scene that exposed the question.
//!
//! A gyroid inside a sphere, which is the case the user exported first and found faceted. It is
//! the hard case on purpose: a lattice is almost entirely surface, so nearly every cell of the
//! sampling grid contains geometry and none of the usual "most of the box is empty" savings
//! apply.
//!
//! The point of the table is that **the resolution is the lever, and the display setting is not
//! the right number for it**. Export used to share `mesh_resolution`, which is capped at 256
//! because that is where sphere-tracing stops being interactive, and defaults to 64. Six cells
//! per gyroid wall is what 64 buys on this part, and it looks exactly as faceted as that
//! implies.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example export_cost`

use fieldform_field::{CellKind, Lattice, LatticeMap, Phase, ScalarField, TpmsKind};
use nalgebra::{Point3, Vector3};

/// The part: a 100 mm sphere with a 12.5 mm gyroid in it, 1.5 mm walls. Eight cells across the
/// sphere, which is the ratio the screenshot showed.
const RADIUS_MM: f64 = 50.0;
const CELL_MM: f64 = 12.5;
const WALL_MM: f64 = 1.5;

/// A sphere intersected with a lattice, as the graph would compose it.
struct SphereGyroid {
    lattice: Lattice,
}

impl ScalarField for SphereGyroid {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let sphere = p.coords.norm() - RADIUS_MM;
        sphere.max(self.lattice.distance(p))
    }

    fn lipschitz(&self) -> f64 {
        // A max of two fields varies no faster than the faster of them.
        self.lattice.lipschitz().max(1.0)
    }

    fn normalization(&self) -> fieldform_field::Normalization {
        fieldform_field::Normalization::LowerBound
    }
}

fn main() {
    let part = SphereGyroid {
        lattice: Lattice {
            cell: CellKind::Tpms(TpmsKind::Gyroid),
            map: LatticeMap::Rectangular {
                cell: Vector3::new(CELL_MM, CELL_MM, CELL_MM),
            },
            phase: Phase::sheet(WALL_MM),
        },
    };

    let half = RADIUS_MM * 1.05;
    let bounds = (
        Point3::new(-half, -half, -half),
        Point3::new(half, half, half),
    );

    println!(
        "A {:.0} mm sphere holding a {CELL_MM} mm gyroid with {WALL_MM} mm walls.",
        RADIUS_MM * 2.0
    );
    println!(
        "{:.1} cells of lattice across the diameter.\n",
        RADIUS_MM * 2.0 / CELL_MM
    );
    println!(
        "{:>6}  {:>10}  {:>12}  {:>9}  {:>9}  {:>8}",
        "cells", "cell (mm)", "triangles", "STL", "time", "per wall"
    );

    for resolution in [64usize, 128, 256, 512] {
        let started = std::time::Instant::now();
        let mesh = fieldform_field::mesh_field(&part, bounds, resolution);
        let took = started.elapsed().as_secs_f64();
        let cell_mm = (half * 2.0) / resolution as f64;
        let bytes = 84 + mesh.triangles.len() * 50;
        println!(
            "{resolution:>6}  {cell_mm:>10.3}  {:>12}  {:>7.1} MB  {took:>8.1}s  {:>8.1}",
            mesh.triangles.len(),
            bytes as f64 / (1024.0 * 1024.0),
            WALL_MM / cell_mm,
        );
    }

    println!(
        "\n'per wall' is how many sampling cells span one {WALL_MM} mm wall. Below about two,\n\
         the grid cannot represent the wall at all and the mesher is reporting the grid rather\n\
         than the part."
    );
}
