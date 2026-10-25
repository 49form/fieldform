// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How round a Catmull-Clark prism is, by the number of sides its cage has.
//!
//! A cube extruded along one axis is a four-sided prism, and the owner asked whether its limit
//! surface -- which is the field the viewport traces -- is a true circle in cross-section. It
//! cannot be: the limit surface is piecewise cubic polynomial and a circle is not. What matters
//! is by how much, and how fast that falls as the control polygon gains sides, so this measures
//! it: a long open prism of `n` sides, its limit surface sampled round the middle ring, and the
//! cross-section's radius read as its least and greatest. The error of a mesh or a bake is not
//! in these figures -- they are of the surface itself.
//!
//! Run with: `cargo run --release -p fieldform-subd --example roundness`

use fieldform_cage::{FaceId, QuadMesh};
use fieldform_subd::LimitSurface;
use nalgebra::Point3;

/// Rings along the prism, so the middle ring is far from the open ends.
const ALONG: usize = 12;
/// Samples across each face of the middle ring.
const SAMPLES: usize = 64;

fn main() {
    println!("A prism of n sides, circumradius 50 mm, limit surface sampled round its middle.");
    println!("The radius it settles to is smaller than the cage's; roundness is the spread of that radius.\n");
    println!("{:>6} {:>14} {:>14} {:>14} {:>12} {:>12}", "sides", "r min (mm)", "r max (mm)", "r mean (mm)", "spread", "max − min");
    for n in [3usize, 4, 5, 6, 8, 12, 16, 24] {
        let mesh = prism(n, 50.0, ALONG, 20.0);
        let mut surface = LimitSurface::over(&mesh);
        let middle = ALONG / 2;
        let mut radii = Vec::with_capacity(n * SAMPLES);
        for i in 0..n {
            let face = FaceId::from_index(middle * n + i);
            for s in 0..SAMPLES {
                let u = s as f64 / SAMPLES as f64;
                let p = surface.evaluate(face, u, 0.5).expect("a prism evaluates");
                radii.push((p.x * p.x + p.y * p.y).sqrt());
            }
        }
        let (lo, hi) = radii.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &r| (lo.min(r), hi.max(r)));
        let mean = radii.iter().sum::<f64>() / radii.len() as f64;
        println!(
            "{:>6} {:>14.4} {:>14.4} {:>14.4} {:>11.3}% {:>9.4} mm",
            n,
            lo,
            hi,
            mean,
            (hi - lo) / mean * 100.0,
            hi - lo
        );
    }
}

/// An open prism: `n` vertices a ring, `along` quads long, each ring `pitch` apart.
fn prism(n: usize, radius: f64, along: usize, pitch: f64) -> QuadMesh {
    let mut positions = Vec::new();
    for j in 0..=along {
        let z = (j as f64 - along as f64 / 2.0) * pitch;
        for i in 0..n {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            positions.push(Point3::new(radius * a.cos(), radius * a.sin(), z));
        }
    }
    let at = |i: usize, j: usize| (j * n + i % n) as u32;
    let faces: Vec<Vec<u32>> = (0..along)
        .flat_map(|j| (0..n).map(move |i| vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]))
        .collect();
    QuadMesh::from_faces(positions, &faces).expect("a prism is a valid open mesh")
}
