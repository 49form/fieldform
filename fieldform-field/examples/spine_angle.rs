// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a curve's tessellation tolerance actually costs a pipe's shading.
//!
//! A tube's surface is every point at `radius` from the spine, so its normal points away from
//! whichever segment is nearest — and turns a corner at every joint. The visible artefact is
//! therefore an *angle*, while the tolerance that produces it is a *distance*, and the two are
//! related by the curvature rather than by a constant:
//!
//! ```text
//! turn per joint  ≈  2 · sqrt(2 · tolerance / radius_of_curvature)
//! ```
//!
//! The square root is the whole problem. Halving the tolerance only cuts the turn by 30%, so a
//! tolerance chosen to put the spine within a fraction of a millimetre of the true curve can
//! still leave several degrees at each joint — which reads as bands running across the tube and,
//! through a blend, across whatever it was blended into.
//!
//! `cargo run -p fieldform-field --example spine_angle`

use fieldform_scene::{Curve, CurveId};
use nalgebra::Point3;

/// The largest turn between consecutive segments, in degrees.
fn worst_turn(points: &[Point3<f64>]) -> f64 {
    let mut worst: f64 = 0.0;
    for w in points.windows(3) {
        let a = w[1] - w[0];
        let b = w[2] - w[1];
        if let (Some(a), Some(b)) = (a.try_normalize(1e-12), b.try_normalize(1e-12)) {
            worst = worst.max(a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    worst
}

fn main() {
    // A cubic Bézier about the size of the one in the app: a couple of hundred millimetres
    // across, turning through most of a right angle.
    let id = CurveId::new();
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 120.0, 0.0),
        Point3::new(120.0, 200.0, 0.0),
        Point3::new(200.0, 200.0, 0.0),
    ];
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let curve = Curve::new(id, "arc", 3, points, vec![1.0; 4], knots, false).unwrap();

    println!("curve {:.1} mm long\n", curve.length(0.01));
    println!(
        "{:>12}  {:>9}  {:>13}",
        "tolerance", "segments", "worst turn"
    );

    // The middle of this range is what the viewport currently asks for: the app uses
    // `extent.norm() * 1e-3`, which on a model this size is about a third of a millimetre.
    for tolerance in [2.0, 1.0, 0.35, 0.1, 0.02, 0.005, 0.001] {
        let spine = curve.tessellate(tolerance);
        println!(
            "{tolerance:>10.3} mm  {:>9}  {:>11.2}°",
            spine.len(),
            worst_turn(&spine)
        );
    }

    for target in [2.0, 1.0, 0.5] {
        let spine = curve.tessellate_to_angle(target, 1e-4);
        println!(
            "asking for {target} deg gives {} segments, worst turn {:.2} deg",
            spine.len(),
            worst_turn(&spine),
        );
    }

    println!(
        "\nA joint turning by more than about a degree is visible as a band on a tube, because\n\
         shading is the normal and the normal turns by exactly that much across it."
    );
}
