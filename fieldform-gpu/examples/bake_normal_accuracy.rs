// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Why a baked cage shades as though it were tessellated, and what fixes it.
//!
//! The traced viewport shows facets on a surface that has no triangles anywhere in it. This
//! measures where they come from.
//!
//! # The suspicion
//!
//! Trilinear interpolation is C0 but not C1: the value is continuous across a cell boundary,
//! the *gradient* is not. So a baked field has a gradient that jumps from cell to cell, and
//! shading it by central differences reproduces those cells as flat facets — exactly as
//! differencing a distance-to-triangles field reproduced triangles, which is the bug
//! `ScalarField::normal` was introduced to fix. Same shape of failure, one layer down.
//!
//! If that is right, the error should depend on the difference step `h` relative to the bake
//! cell: steps smaller than a cell sample inside one cell and see its flat gradient, steps
//! spanning several average across them.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example bake_normal_accuracy`

use fieldform_cage::Cage;
use fieldform_field::{CageField, ScalarField};
use fieldform_gpu::{BakedField, BakedFieldSource};
use nalgebra::{Point3, Vector3};

const CAGE_MM: f64 = 100.0;

fn main() {
    let cage = Cage::cube_of(CAGE_MM);
    let field = CageField::from_cage(&cage, 5).expect("the cage builds");

    let half = CAGE_MM * 0.9;
    let bounds = (
        Point3::new(-half, -half, -half),
        Point3::new(half, half, half),
    );

    // Points on the limit surface, found by walking in from outside along a spread of
    // directions. Shading error only means anything where the surface actually is.
    let probes = surface_points(&field, 400);
    println!("{} surface points on a {CAGE_MM} mm cage\n", probes.len());

    println!("The traced viewport bakes at 200 and differences with h ~= 0.5 mm.\n");
    println!("  bake   cell      h     h/cell    worst      mean   (degrees from the true normal)");

    println!("bake cost, which is what a cage edit pays before the traced view updates:");
    for resolution in [24usize, 32, 48, 64, 96, 128] {
        let start = std::time::Instant::now();
        let b = BakedField::new(&field, bounds, resolution, CAGE_MM * 0.35);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {resolution:>4}  {:>5}^3 samples  {ms:>7.1} ms  {:>6.1} MB",
            b.dims[0],
            b.size_bytes() as f64 / 1e6
        );
    }
    println!();

    for resolution in [100usize, 200, 400] {
        let baked = BakedField::new(&field, bounds, resolution, CAGE_MM * 0.35);
        let cell = baked.cell.x;
        let source = BakedFieldSource(baked);

        for h_cells in [0.5, 1.0, 2.0, 4.0] {
            let h = cell * h_cells;
            let (worst, mean) = normal_error(&field, &source, &probes, h);
            println!(
                "  {resolution:>4}  {cell:>5.2}  {h:>5.2}   {h_cells:>5.1}   {worst:>7.2}   {mean:>7.2}"
            );
        }
        // The fix: the normal the cage already knew, baked alongside the distance and
        // interpolated, instead of rediscovered by differencing.
        let (worst, mean) = stored_normal_error(&field, &source, &probes);
        println!("  {resolution:>4}  {cell:>5.2}  stored      -    {worst:>7.2}   {mean:>7.2}");
        println!();
    }

    // The same measurement against the *unbaked* cage, to separate the two causes. Baking is
    // not the only thing differencing gets wrong -- a distance-to-triangles field has its own
    // piecewise-constant gradient, which is the facet problem `ScalarField::normal` was added
    // to fix. Anything above this line that is worse than this row is the bake's contribution.
    println!("Differencing the cage directly, with no bake in the way:");
    for h in [0.25, 0.5, 1.0, 2.0] {
        let (worst, mean) = normal_error_against(&field, &field, &probes, h);
        println!("         -  {h:>5.2}       -    {worst:>7.2}   {mean:>7.2}");
    }
    println!(
        "\nThe cage's own analytic normal is exact by construction: it is evaluated on the\n\
         limit surface rather than recovered from samples of it. That is what the meshed\n\
         viewport shades with, and what baking currently discards."
    );
}

/// Points on the limit surface, by sphere-tracing inward from a spread of directions.
fn surface_points(field: &CageField, count: usize) -> Vec<Point3<f64>> {
    let mut points = Vec::new();
    // A deterministic spiral over the sphere, so the sample set is the same every run.
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    for i in 0..count {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as f64;
        let dir = Vector3::new(r * theta.cos(), y, r * theta.sin());

        let mut p = Point3::origin() + dir * (CAGE_MM * 1.5);
        let inward = -dir;
        for _ in 0..200 {
            let d = field.distance(p);
            if d.abs() < 1e-6 {
                break;
            }
            p += inward * d;
        }
        if field.distance(p).abs() < 1e-4 {
            points.push(p);
        }
    }
    points
}

/// Worst and mean angle, in degrees, between the baked field's differenced normal and the
/// cage's own analytic one.
fn normal_error(
    field: &CageField,
    baked: &BakedFieldSource,
    probes: &[Point3<f64>],
    h: f64,
) -> (f64, f64) {
    normal_error_against(field, baked, probes, h)
}

/// Angle between the cage's analytic normal and the one stored in the bake.
fn stored_normal_error(
    field: &CageField,
    baked: &BakedFieldSource,
    probes: &[Point3<f64>],
) -> (f64, f64) {
    let mut worst: f64 = 0.0;
    let mut total = 0.0;
    for &p in probes {
        let truth = field.closest(p).normal;
        let g = baked.normal(p).expect("a bake always has a stored normal");
        let angle = truth.dot(&g).clamp(-1.0, 1.0).acos().to_degrees();
        worst = worst.max(angle);
        total += angle;
    }
    (worst, total / probes.len() as f64)
}

/// Angle between the cage's analytic normal and whatever differencing `sampled` produces.
fn normal_error_against(
    field: &CageField,
    sampled: &dyn ScalarField,
    probes: &[Point3<f64>],
    h: f64,
) -> (f64, f64) {
    let mut worst: f64 = 0.0;
    let mut total = 0.0;
    for &p in probes {
        let truth = field.closest(p).normal;
        let g = central_difference(sampled, p, h);
        let angle = truth.dot(&g).clamp(-1.0, 1.0).acos().to_degrees();
        worst = worst.max(angle);
        total += angle;
    }
    (worst, total / probes.len() as f64)
}


fn central_difference(field: &dyn ScalarField, p: Point3<f64>, h: f64) -> Vector3<f64> {
    let d = |o: Vector3<f64>| field.distance(p + o);
    Vector3::new(
        d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
        d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
        d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
    )
    .try_normalize(1e-18)
    .unwrap_or_else(Vector3::z)
}
