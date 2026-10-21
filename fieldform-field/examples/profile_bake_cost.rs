// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether baking a profile to a grid is better than sampling its curve more finely.
//!
//! The backlog asks for this before the work rather than after: a bake replaces one error with
//! another, and it is only worth building if the one it introduces is smaller than the one it
//! removes.
//!
//! **What is being compared.** Today a profile is a polyline, and how finely the curve was
//! sampled is both its accuracy and its per-query cost — the boundary is walked segment by
//! segment at every sample. A bake breaks that tie: the grid can be made from a polyline far
//! finer than anything could afford to walk, and the lookup costs the same whatever the boundary
//! has in it. What it gives up is the margin that keeps interpolation from overstating.
//!
//! So the question is one comparison: **is the bake's margin smaller than the chord error of the
//! polyline it replaces**, at a resolution anybody would use?
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example profile_bake_cost`

use fieldform_field::{BakedProfile, Profile};
use fieldform_scene::{Curve, CurveId};
use nalgebra::{Point2, Point3};
use std::time::Instant;

fn main() {
    let curve = ring();

    // Ground truth: far finer than anything would ship, so both candidates are measured against
    // the curve rather than against each other.
    let truth = profile_at(&curve, 0.1);
    println!("reference: {} points at 0.1 deg\n", truth.points().len());

    println!("What a polyline costs, which is what we have today:");
    println!("  max turn   points   worst error vs the curve");
    for degrees in [4.0, 2.0, 1.0] {
        let p = profile_at(&curve, degrees);
        println!(
            "  {degrees:>6} deg   {:>6}   {:>8.4} mm",
            p.points().len(),
            worst_gap(&p, &truth),
        );
    }

    println!("\nWhat a bake costs, made from the reference polyline:");
    println!("  cells   margin       memory   blocks kept   at the surface");
    for cells in [128usize, 256, 512, 1024, 2048] {
        let baked = BakedProfile::new(&truth, cells, (0.0, 0.0));
        let (kept, total) = baked.stored_blocks();
        let (near, _) = worst_baked_gap(&baked, &truth);
        // What the same fineness would have cost stored as a full square, for comparison.
        let dense = total * (8 + 1) * (8 + 1);
        println!(
            "  {cells:>5}   {:>7.4} mm   {:>4} KB   {kept:>4} / {total:<5}   {near:>8.4} mm   \
             (dense would be {:>5} KB)",
            baked.margin(),
            baked.samples() * 4 / 1024,
            dense * 4 / 1024,
        );
    }

    println!("\nWhat a lookup costs against a walk, per million queries:");
    println!("  boundary points   walk        baked lookup");
    for degrees in [4.0, 1.0, 0.25] {
        let p = profile_at(&curve, degrees);
        let baked = BakedProfile::new(&p, 256, (0.0, 0.0));
        let (walk, look) = timings(&p, &baked);
        println!(
            "  {:>15}   {walk:>6.1} ms   {look:>6.1} ms",
            p.points().len(),
        );
    }

    // What an edit costs, which is the price of all of the above: a bake searches the boundary
    // once per grid point, and it happens again every time the curve moves.
    println!("
What making the grid costs, which is paid once per edit:");
    println!("  cells    boundary points   bake");
    for (cells, degrees) in [(128usize, 1.0), (256, 1.0), (512, 1.0), (256, 4.0)] {
        let p = profile_at(&curve, degrees);
        let start = Instant::now();
        let baked = BakedProfile::new(&p, cells, (0.0, 0.0));
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&baked);
        println!("  {cells:>5}   {:>15}   {ms:>6.1} ms", p.points().len());
    }

    println!(
        "\nThe bake's error does not move with the boundary's point count and the walk's does,\n\
         which is the whole of the argument: one of these two columns is a dial and the other\n\
         is a constant."
    );
}

/// A closed cubic ring, standing in for a profile somebody drew.
fn ring() -> Curve {
    let mut points = Vec::new();
    for i in 0..12 {
        let t = i as f64 / 12.0 * std::f64::consts::TAU;
        points.push(Point3::new(100.0 * t.cos(), 70.0 * t.sin(), 0.0));
    }
    let n = points.len();
    Curve::new(
        CurveId::new(),
        "Ring".to_string(),
        3,
        points,
        vec![1.0; n],
        (0..n + 4).map(|i| i as f64).collect(),
        true,
    )
    .expect("a closed cubic ring")
}

fn profile_at(curve: &Curve, degrees: f64) -> Profile {
    let floor = (curve.length(0.5) * 1e-5).max(1e-4);
    let world = curve.tessellate_to_angle(degrees, floor);
    Profile::in_plane(&world, |p| p, 1e-3).expect("planar")
}

/// The worst the coarse profile disagrees with the fine one, over a grid across both.
fn worst_gap(coarse: &Profile, truth: &Profile) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..120 {
        for j in 0..120 {
            let q = Point2::new(-140.0 + i as f64 * 2.4, -110.0 + j as f64 * 1.9);
            worst = worst.max((coarse.distance(q) - truth.distance(q)).abs());
        }
    }
    worst
}

/// The worst disagreement near the surface, and the worst far from it.
///
/// **The split that matters once the bake has two levels.** Near the boundary the fine blocks
/// answer and the gap is the accuracy of the thing -- it is where the surface *is*. Far from it
/// the coarse level answers, deliberately looser, and what that costs is a ray taking an extra
/// step rather than a surface sitting in the wrong place. Reporting one number over both would
/// have said the bake got eight times worse when it had got four times cheaper.
fn worst_baked_gap(baked: &BakedProfile, truth: &Profile) -> (f64, f64) {
    // **The window has to follow the band, not a round number in millimetres.** A fixed two
    // millimetres is inside the band at a coarse bake and several blocks outside it at a fine
    // one -- so the fine bakes appeared to get *worse*, when what they were reporting was the
    // coarse level answering somewhere the fine level had correctly decided not to store.
    // Eight margins is about one block either side, which is the band.
    let band = baked.margin() * 8.0;
    let (mut near, mut far): (f64, f64) = (0.0, 0.0);
    for i in 0..400 {
        for j in 0..400 {
            let q = Point2::new(-140.0 + i as f64 * 0.72, -110.0 + j as f64 * 0.57);
            let gap = (baked.distance(q) - truth.distance(q)).abs();
            if truth.distance(q).abs() < band {
                near = near.max(gap);
            } else {
                far = far.max(gap);
            }
        }
    }
    (near, far)
}

/// Milliseconds per million queries, walking and looking up.
fn timings(profile: &Profile, baked: &BakedProfile) -> (f64, f64) {
    let queries: Vec<Point2<f64>> = (0..1000)
        .map(|i| {
            let t = i as f64 * 0.01;
            Point2::new(t.cos() * 130.0, t.sin() * 95.0)
        })
        .collect();

    let start = Instant::now();
    let mut sink = 0.0f64;
    for _ in 0..1000 {
        for q in &queries {
            sink += profile.distance(*q);
        }
    }
    let walk = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    for _ in 0..1000 {
        for q in &queries {
            sink += baked.distance(*q);
        }
    }
    let look = start.elapsed().as_secs_f64() * 1000.0;

    std::hint::black_box(sink);
    (walk, look)
}
