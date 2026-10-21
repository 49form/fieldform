// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How many points a profile polyline is handed to the shader as.
//!
//! The shader walks every point of a profile at every sample, so this number multiplies the cost
//! of an extrude or a revolve directly — and a pattern multiplies it again by its count.
//!
//! Run with: `cargo run --release -p fieldform-field --example profile_points`

use fieldform_scene::{Curve, CurveId};
use nalgebra::Point3;

fn main() {
    let id = CurveId::new();
    let curve = Curve::new(
        id,
        "Arc".to_string(),
        2,
        vec![
            Point3::new(-100.0, 0.0, 0.0),
            Point3::new(0.0, 120.0, 0.0),
            Point3::new(100.0, 0.0, 0.0),
        ],
        vec![1.0; 3],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        false,
    )
    .expect("a degree two curve");

    println!("A 200 mm degree-two curve, tessellated at various tolerances:\n");
    println!("  tolerance      points");
    for tolerance in [1e-4, 1e-3, 1e-2, 0.05, 0.25, 1.0] {
        println!("  {tolerance:>9} mm  {:>6}", curve.tessellate(tolerance).len());
    }
    println!(
        "\nThe profile walk uses the join tolerance for this, which is 1e-4 mm — chosen so that\n\
         two ends either meet or do not, independently of display quality. It is the *ends* that\n\
         needed that, not the sampling in between."
    );
    println!(
        "\nBy turned angle instead, which is what a silhouette actually shows: {} points at one \
         degree.",
        curve.tessellate_to_angle(1.0, (curve.length(0.5) * 1e-5).max(1e-4)).len()
    );

    // And through the walk an extrude or a revolve actually uses, which is what the shader is
    // handed and what it scans linearly at every sample.
    // Closed, because a loop is what a profile is -- an open curve has no inside and the walk
    // refuses it, which is what an earlier version of this measured as zero points.
    let ring_id = CurveId::new();
    let mut ring_points = Vec::new();
    for i in 0..12 {
        let t = i as f64 / 12.0 * std::f64::consts::TAU;
        ring_points.push(Point3::new(100.0 * t.cos(), 80.0 * t.sin(), 0.0));
    }
    let n = ring_points.len();
    let ring = Curve::new(
        ring_id,
        "Ring".to_string(),
        3,
        ring_points,
        vec![1.0; n],
        (0..n + 4).map(|i| i as f64).collect(),
        true,
    )
    .expect("a closed cubic ring");
    let mut doc = fieldform_scene::Document::new();
    doc.curves.insert(ring_id, ring);
    let id = ring_id;
    // Through the walk an extrude or a revolve actually uses, at the angles the three Display
    // resolution settings ask for. What must NOT happen here is the same number twice: that is
    // the shape of a floor binding, and it is exactly how a coarse profile shipped while the
    // setting appeared to be doing something.
    println!("\n  max turn      points through `profile_loop`");
    for degrees in [4.0, 2.0, 1.0] {
        let n = fieldform_field::profile_loop_turning(&doc, id, degrees, None)
            .map(|f| f.points.len())
            .unwrap_or(0);
        println!("  {degrees:>6} deg   {n:>6}");
    }

    // Points against the angle the polyline is allowed to turn at a joint, which is what shows
    // as banding: shading is the normal, and a facet holds one normal across its whole width.
    println!("\n  max turn      points");
    for degrees in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
        let n = doc.curves[&id].tessellate_to_angle(degrees, 1e-4).len();
        println!("  {degrees:>6}\u{b0}   {n:>6}");
    }
}
