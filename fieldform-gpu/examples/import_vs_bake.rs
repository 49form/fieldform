// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which of the two approximations between a STEP file and the screen actually limits it.
//!
//! An imported solid passes through two lossy steps before it is drawn, and they are decades
//! apart in size, so only one of them is worth tuning:
//!
//! 1. **Tessellation.** OCCT facets the real surfaces at a chord height — 0.05 mm by default.
//! 2. **The bake.** Those triangles become a narrow-band grid, because a mesh field is the one
//!    leaf that cannot be evaluated in closed form. The viewport samples the grid, never the
//!    triangles.
//!
//! The question is which one a user sees as faceting. It is measured here on a **sphere**,
//! whose exact distance is known analytically — so the error reported is the pipeline's and
//! not a second approximation standing in for the truth.
//!
//! Run with the display resolutions the app actually offers:
//! `cargo run --release -p fieldform-gpu --example import_vs_bake`

use fieldform_field::{MeshField, ScalarField};
use nalgebra::{Point3, Vector3};

/// A sphere as triangles, faceted to roughly `chord_mm`.
///
/// The subdivision count comes from the chord-height relation for a circle: a chord subtending
/// angle t on radius r sits r(1 - cos(t/2)) below the arc, so t = 2*acos(1 - chord/r).
fn tessellated_sphere(radius: f64, chord_mm: f64) -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
    let angle = 2.0 * (1.0 - chord_mm / radius).clamp(-1.0, 1.0).acos();
    let rings = ((std::f64::consts::PI / angle).ceil() as usize).max(4);
    let segments = rings * 2;

    // Single vertices at the poles, not a whole collapsed ring. A ring of coincident points
    // leaves degenerate triangles whose edges are used once, and `MeshField` correctly refuses
    // the result as not closed — an open mesh has no inside for a sign to refer to.
    let mut positions = vec![Point3::new(0.0, 0.0, radius)];
    for i in 1..rings {
        let theta = std::f64::consts::PI * i as f64 / rings as f64;
        for j in 0..segments {
            let phi = std::f64::consts::TAU * j as f64 / segments as f64;
            positions.push(Point3::new(
                radius * theta.sin() * phi.cos(),
                radius * theta.sin() * phi.sin(),
                radius * theta.cos(),
            ));
        }
    }
    positions.push(Point3::new(0.0, 0.0, -radius));

    let north = 0u32;
    let south = (positions.len() - 1) as u32;
    // Ring `i` (1-based) starts one past the north pole.
    let at = |i: usize, j: usize| (1 + (i - 1) * segments + j % segments) as u32;

    let mut triangles = Vec::new();
    for j in 0..segments {
        triangles.push([north, at(1, j), at(1, j + 1)]);
        triangles.push([south, at(rings - 1, j + 1), at(rings - 1, j)]);
    }
    for i in 1..rings - 1 {
        for j in 0..segments {
            triangles.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
            triangles.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    (positions, triangles)
}

/// A narrow-band grid over `field`, sampled trilinearly — what the viewport reads.
struct Baked {
    values: Vec<f32>,
    n: usize,
    lo: Point3<f64>,
    step: f64,
}

impl Baked {
    fn of(field: &dyn ScalarField, half: f64, n: usize) -> Self {
        let lo = Point3::new(-half, -half, -half);
        let step = (half * 2.0) / (n - 1) as f64;
        let mut values = vec![0f32; n * n * n];
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let p = lo + Vector3::new(a as f64, b as f64, c as f64) * step;
                    values[(a * n + b) * n + c] = field.distance(p) as f32;
                }
            }
        }
        Self { values, n, lo, step }
    }

    fn sample(&self, p: Point3<f64>) -> f64 {
        let g = (p - self.lo) / self.step;
        let clamp = |v: f64| v.clamp(0.0, (self.n - 2) as f64);
        let (x, y, z) = (clamp(g.x), clamp(g.y), clamp(g.z));
        let (i, j, k) = (x as usize, y as usize, z as usize);
        let (fx, fy, fz) = (x - i as f64, y - j as f64, z - k as f64);
        let at = |i: usize, j: usize, k: usize| self.values[(i * self.n + j) * self.n + k] as f64;
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let c00 = lerp(at(i, j, k), at(i + 1, j, k), fx);
        let c01 = lerp(at(i, j, k + 1), at(i + 1, j, k + 1), fx);
        let c10 = lerp(at(i, j + 1, k), at(i + 1, j + 1, k), fx);
        let c11 = lerp(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), fx);
        lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz)
    }
}

/// Directions spread over the sphere, deterministic so runs can be compared.
fn directions(n: usize) -> Vec<Vector3<f64>> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            // A Fibonacci-ish spiral: even coverage without needing randomness.
            let z = 1.0 - 2.0 * (t + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = t * 2.399_963_229_728_653;
            Vector3::new(r * phi.cos(), r * phi.sin(), z)
        })
        .collect()
}

/// Worst and mean angle between the read field's gradient and the true sphere normal, in
/// degrees.
///
/// The quantity that decides whether a curved face *looks* smooth. Position error and normal
/// error are different questions, and a pipeline can be tight on one and loose on the other —
/// which is the whole reason both are measured here.
fn normal_error(
    read: &dyn Fn(Point3<f64>) -> f64,
    radius: f64,
    dirs: &[Vector3<f64>],
    h: f64,
) -> (f64, f64) {
    let mut worst: f64 = 0.0;
    let mut total = 0.0;
    for d in dirs {
        // On the true surface, where the shading actually happens.
        let p = Point3::from(d * radius);
        let axis = |e: Vector3<f64>| (read(p + e * h) - read(p - e * h)) / (2.0 * h);
        let g = Vector3::new(axis(Vector3::x()), axis(Vector3::y()), axis(Vector3::z()));
        let len = g.norm();
        if len < 1e-12 {
            continue;
        }
        let cos = (g / len).dot(d).clamp(-1.0, 1.0);
        let e = cos.acos().to_degrees();
        worst = worst.max(e);
        total += e;
    }
    (worst, total / dirs.len() as f64)
}

/// Worst and mean distance from the true sphere, over the sampled directions.
fn error_against_sphere(
    read: &dyn Fn(Point3<f64>) -> f64,
    radius: f64,
    dirs: &[Vector3<f64>],
) -> (f64, f64) {
    let mut worst: f64 = 0.0;
    let mut total = 0.0;
    for d in dirs {
        // Bisect along the ray for where the read field changes sign: that is where this
        // pipeline puts the surface, and the true answer is `radius`.
        let (mut lo, mut hi) = (radius * 0.5, radius * 1.5);
        for _ in 0..60 {
            let mid = (lo + hi) * 0.5;
            if read(Point3::from(d * mid)) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let e = ((lo + hi) * 0.5 - radius).abs();
        worst = worst.max(e);
        total += e;
    }
    (worst, total / dirs.len() as f64)
}

fn main() {
    const RADIUS: f64 = 50.0;
    // The band covers the model with a little room, the way the app's own bake does.
    const HALF: f64 = 60.0;
    let dirs = directions(4000);

    println!(
        "A 100 mm sphere, through the two steps an imported solid takes.\n\
         Error is distance from the true surface, over 4000 directions.\n"
    );

    println!("Tessellation alone (no bake) — what the import chord height buys:");
    let mut meshes = Vec::new();
    for chord in [0.1, 0.05, 0.01] {
        let (positions, triangles) = tessellated_sphere(RADIUS, chord);
        let n = triangles.len();
        let field = MeshField::from_mesh(positions, triangles).expect("a closed sphere");
        let (worst, _) = error_against_sphere(&|p| field.distance(p), RADIUS, &dirs);
        // Differenced over a whole cell of the finest bake, which is the scale the shader
        // actually reads the gradient at.
        let (nw, nm) = normal_error(&|p| field.distance(p), RADIUS, &dirs, 0.3);
        println!(
            "  chord {chord:>5.2} mm ({n:>7} tris): pos worst {worst:>7.4} mm  |               normal worst {nw:>6.2}deg, mean {nm:>6.2}deg"
        );
        meshes.push((chord, field));
    }

    println!("\nAfter the bake, at the three resolutions View > Display resolution offers:");
    for (label, n) in [("Coarse", 64usize), ("Medium", 128), ("High", 192)] {
        let cell = (HALF * 2.0) / (n - 1) as f64;
        for (chord, field) in &meshes {
            let baked = Baked::of(field, HALF, n);
            let (worst, _) = error_against_sphere(&|p| baked.sample(p), RADIUS, &dirs);
            // Differenced over one cell, which is the scale the shader reads it at.
            let (nw, nm) = normal_error(&|p| baked.sample(p), RADIUS, &dirs, cell);
            println!(
                "  {label:<7} {n:>3} cells ({cell:>4.2} mm), chord {chord:>5.2}: pos {worst:>7.4} mm | normal worst {nw:>6.2}deg, mean {nm:>6.2}deg"
            );
        }
    }

    println!(
        "\nIf the numbers in a row barely move with the chord height, the bake is what the\n\
         surface is limited by and a finer import buys nothing anyone can see."
    );
}
