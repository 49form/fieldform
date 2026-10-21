// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What an attractor actually does to a body, and what it costs the field's guarantees.
//!
//! Five questions, none of which is answered by reading the formula:
//!
//! - **Which way does it pull?** Getting the sign backwards is the standard failure of this
//!   technique, and it fails silently — the result is still a deformation, just the wrong one.
//!   Measured as where the surface of a sphere ends up relative to where it started.
//! - **How far does anything move?** The bounding box the node reports is grown by
//!   `MAX_PULL_FRACTION · strength · falloff`, so a body that moved further than that would be
//!   a body the mesher clipped and the ray marcher missed.
//! - **Is it still a distance field?** D8's `gradient_deviation` answers this, and §6 of the
//!   spec asks for the answer to be recorded rather than assumed either way.
//! - **Is it at least an honest lower bound?** `overstatement` answers that, which is the
//!   different question a `LowerBound` field has to pass. This is the one that matters: a
//!   field claiming more room than it has is a ray stepping through a wall.
//! - **What does the cut locus cost?** Where a curve bends, points on the inside of the bend
//!   have two nearest points, and the warp direction jumps between them. That makes the warped
//!   field genuinely discontinuous there. Nothing in the construction avoids it, so the useful
//!   thing is to know how big it is and when it appears.
//!
//! Run with: `cargo run --release -p fieldform-field --example attractor_profile`

use fieldform_field::{
    Normalization, ScalarField, Spine, gradient_deviation, max_pull, overstatement,
    probe_directions, warp_point,
};
use nalgebra::{Point3, Vector3};

/// A body drawn toward a spine, standing in for what `Op::Attract` evaluates.
///
/// Written against `ScalarField` rather than through the graph so the example can vary the
/// pieces independently — but it is the same three steps in the same order, and it divides by
/// the same `attract::stretch`. If this and the compiled op ever disagree, the property tests
/// in `graph_field.rs` fail first.
struct Attracted<'a> {
    inner: &'a dyn ScalarField,
    spine: &'a Spine,
    strength: f64,
    falloff: f64,
}

impl ScalarField for Attracted<'_> {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let warped = match self.spine.nearest(p) {
            Some(n) => warp_point(p, n.point, self.strength, self.falloff),
            None => p,
        };
        self.inner.distance(warped) / fieldform_field::stretch(self.strength)
    }

    fn normalization(&self) -> Normalization {
        Normalization::LowerBound
    }
}

struct Sphere {
    centre: Point3<f64>,
    radius: f64,
}

impl ScalarField for Sphere {
    fn distance(&self, p: Point3<f64>) -> f64 {
        (p - self.centre).norm() - self.radius
    }
}

/// How far along the ray from `from` in direction `dir` the surface first is.
///
/// **Scanned for a bracket, then bisected**, rather than bisected over the whole span. A ray
/// through a body crosses the surface twice, so the two ends of the span have the same sign and
/// a plain bisection sees no crossing at all — which is not a near miss, it is the answer
/// "there is no surface here" for a ray that goes straight through the middle of one.
///
/// Bisection rather than sphere-tracing for the crossing itself: the field is a lower bound, so
/// a march stops short of the surface by whatever the bound understates, and this has to
/// measure a position to better than that.
fn surface_along(
    f: &dyn ScalarField,
    from: Point3<f64>,
    dir: Vector3<f64>,
    reach: f64,
) -> Option<f64> {
    let dir = dir.normalize();
    let outside = f.distance(from) > 0.0;
    let steps = 20_000;
    let at = |t: f64| reach * t;
    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        if (f.distance(from + dir * at(t)) > 0.0) != outside {
            let (mut lo, mut hi) = ((i - 1) as f64 / steps as f64, t);
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if (f.distance(from + dir * at(mid)) > 0.0) == outside {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some(at(0.5 * (lo + hi)));
        }
    }
    None
}

/// A grid of points around a body, which is where a field's claims are worth checking.
///
/// **Offset off the lattice on purpose**, and it is not tidiness. A sphere's field has one
/// point where the gradient is not 1 — its own centre, where the distance has no direction —
/// and a grid that lands exactly on it reports a gradient deviation of 1.0 for a field that is
/// a perfect distance field everywhere it is defined. That artifact showed up here as a
/// deviation of 1.0 at *zero* strength, where the warp is provably an exact no-op, which is
/// what gave it away: the number that cannot be a property of the warp is the one that says
/// the harness is wrong.
fn probe_points(lo: Point3<f64>, hi: Point3<f64>, per_axis: usize) -> Vec<Point3<f64>> {
    let mut out = Vec::new();
    for i in 0..per_axis {
        for j in 0..per_axis {
            for k in 0..per_axis {
                let t = |n: usize, a: f64, b: f64| {
                    // An off-centre fraction, not a half-step. A half-step over an odd count
                    // lands one sample exactly in the middle of the span, which is where a
                    // centred body keeps its singularity -- so the obvious offset moved every
                    // sample except the one that mattered.
                    a + (b - a) * (n as f64 + 0.31) / per_axis as f64
                };
                out.push(Point3::new(
                    t(i, lo.x, hi.x),
                    t(j, lo.y, hi.y),
                    t(k, lo.z, hi.z),
                ));
            }
        }
    }
    out
}

fn main() {
    // A straight curve along Z through the origin, and a sphere sitting beside it. The one
    // arrangement where every answer is available by inspection, which is what makes it the
    // right place to check the sign.
    let line = Spine::new(vec![
        Point3::new(0.0, 0.0, -60.0),
        Point3::new(0.0, 0.0, 60.0),
    ]);
    let sphere = Sphere { centre: Point3::new(30.0, 0.0, 0.0), radius: 10.0 };
    let falloff = 40.0;

    println!("A sphere of radius 10 centred 30 mm from a straight curve, reach 40 mm.");
    println!();
    println!("           near face   far face    centre     width    what moved");
    println!("  pull      (mm)        (mm)        (mm)       (mm)");

    let unwarped = {
        let near = surface_along(&sphere, Point3::origin(), Vector3::x(), 100.0).unwrap();
        let far = surface_along(
            &sphere,
            Point3::new(100.0, 0.0, 0.0),
            -Vector3::x(),
            100.0,
        )
        .unwrap();
        (near, 100.0 - far)
    };

    for strength in [0.0, 0.25, 0.5, 1.0, 1.45] {
        let field = Attracted { inner: &sphere, spine: &line, strength, falloff };
        let near = surface_along(&field, Point3::origin(), Vector3::x(), 100.0).unwrap();
        let far =
            100.0 - surface_along(&field, Point3::new(100.0, 0.0, 0.0), -Vector3::x(), 100.0)
                .unwrap();
        let centre = 0.5 * (near + far);
        let moved = if strength == 0.0 {
            "nothing (exact no-op)".to_string()
        } else if centre < 0.5 * (unwarped.0 + unwarped.1) - 1e-9 {
            format!("toward the curve by {:.3}", 0.5 * (unwarped.0 + unwarped.1) - centre)
        } else {
            "AWAY — THE SIGN IS BACKWARDS".to_string()
        };
        println!(
            "  {strength:<8.2}  {near:<10.3}  {far:<10.3}  {centre:<9.3}  {:<8.3}  {moved}",
            far - near,
        );
    }

    println!();
    println!("The near face moves less than the far one: the sphere stretches toward the");
    println!("curve rather than sliding toward it, which is the whole difference between a");
    println!("domain warp and a translation.");

    // ---------------------------------------------------------------------------------
    println!();
    println!("Displacement against the bound the reported box is grown by.");
    println!();
    println!("  pull    reach    furthest moved    0.26·s·R    headroom");
    for (strength, reach) in [(0.25, 40.0), (0.5, 40.0), (1.0, 40.0), (1.0, 10.0), (2.0, 40.0)] {
        let mut worst: f64 = 0.0;
        let n = 20_000;
        for i in 0..=n {
            let d = reach * 1.2 * i as f64 / n as f64;
            let p = Point3::new(d, 0.0, 0.0);
            let g = warp_point(p, Point3::new(0.0, 0.0, 0.0), strength, reach);
            worst = worst.max((g - p).norm());
        }
        let bound = max_pull(strength, reach);
        println!(
            "  {strength:<6.2}  {reach:<7.1}  {worst:<16.4}  {bound:<10.4}  {:+.6}",
            bound - worst,
        );
    }

    // ---------------------------------------------------------------------------------
    println!();
    println!("Is it still a distance field? (D8's gradient_deviation: 0 means yes.)");
    println!("Is it an honest lower bound? (overstatement: 0 means every claim held.)");
    println!();
    println!("  pull    |∇F|−1 worst    overstatement (mm)   verdict");

    let shell = probe_points(
        Point3::new(-5.0, -25.0, -25.0),
        Point3::new(65.0, 25.0, 25.0),
        11,
    );
    let dirs = probe_directions();
    for strength in [0.0, 0.25, 0.5, 1.0, 1.45] {
        let field = Attracted { inner: &sphere, spine: &line, strength, falloff };
        let deviation = gradient_deviation(&field, &shell, 1e-4);
        let over = overstatement(&field, &shell, &dirs, 0.002, 1.2);
        let verdict = if deviation < 1e-6 {
            "true distance"
        } else if over <= 0.004 {
            "lower bound holds"
        } else {
            "OVERSTATES — CONTRACT BROKEN"
        };
        println!("  {strength:<6.2}  {deviation:<15.4}  {over:<20.4} {verdict}");
    }
    println!();
    println!("The overstatement column is accurate to its 0.002 mm step, so anything at or");
    println!("below one step is a pass rather than a small failure.");

    // ---------------------------------------------------------------------------------
    println!();
    println!("The cut locus: inside a bend, a point has two nearest points on the curve and");
    println!("the warp direction jumps between them, so the field is genuinely discontinuous");
    println!("there. What governs the size of the jump is the *turn* at the bend.");
    println!();
    println!("  turn      reach   worst jump (mm)   bound 2·sin(turn/2)·0.26·s·R   where");
    for turn_deg in [5.0f64, 15.0, 45.0, 90.0, 150.0] {
        let reach = 20.0;
        let strength = 1.0;
        // Two segments meeting at the origin: one along -X, the other turned by `turn` from
        // straight ahead, so a turn of 0 is a straight line with no cut locus at all.
        let half = (180.0 - turn_deg).to_radians() / 2.0;
        let a_dir = Vector3::new(-half.sin(), half.cos(), 0.0);
        let b_dir = Vector3::new(half.sin(), half.cos(), 0.0);
        let bend = Spine::new(vec![
            Point3::origin() + a_dir * 60.0,
            Point3::origin(),
            Point3::origin() + b_dir * 60.0,
        ]);
        // The bisector runs up +Y between them. A body straddling it, off-centre so the two
        // sides are not equal by symmetry.
        let body = Sphere { centre: Point3::new(3.0, 18.0, 2.5), radius: 22.0 };
        let field = Attracted { inner: &body, spine: &bend, strength, falloff: reach };
        let mut worst: f64 = 0.0;
        let mut at = 0.0;
        let n = 8000;
        for i in 1..n {
            let t = 60.0 * i as f64 / n as f64;
            let h = 1e-5;
            let p = Point3::new(0.0, t, 0.0);
            let jump = (field.distance(p + Vector3::new(-h, 0.0, 0.0))
                - field.distance(p + Vector3::new(h, 0.0, 0.0)))
            .abs();
            if jump > worst {
                worst = jump;
                at = t;
            }
        }
        // The two warp directions differ by the turn angle, so the sample point jumps by
        // `2·sin(turn/2)` times the displacement -- and the displacement is capped by
        // `MAX_PULL_FRACTION · s · R`, exactly as the bounding box is.
        let bound = 2.0 * (turn_deg.to_radians() / 2.0).sin() * max_pull(strength, reach);
        println!(
            "  {turn_deg:<8.0}  {reach:<6.1}  {worst:<16.4}  {bound:<29.4} {at:.1} mm from it",
        );
    }
    println!();
    println!("The jump tracks the bound at a steady ~0.30 of it up to a right angle, and falls");
    println!("off past that only because the body stops covering where the peak would be. So");
    println!("it is proportional to the turn, which is the useful part: a tessellated");
    println!("smooth curve turns by a fraction of a degree per joint, so its cut locus is");
    println!("there and is negligible. A deliberate hard corner is where this is visible, and");
    println!("the bound says how visible. It is never zero at a polyline corner, however");
    println!("small the reach -- shrinking the reach shrinks the crack rather than removing it.");
    println!();
    println!("(Kept for the record: the reach sweep below, at a fixed right-angle corner.)");
    println!();
    println!("  reach   bend radius   worst jump in the field (mm)   where");

    // Two segments meeting at the origin, running along -X and +Y. The inside of that bend is
    // the quadrant `x < 0, y > 0`, and the two nearest points swap along the ray `(-t, t)` --
    // which is where this walks. The *outside* of a corner has no cut locus at all: the vertex
    // itself is the unique nearest point over the whole wedge, which is why measuring on the
    // wrong side of a bend finds nothing and reads as a pass.
    for (reach, label) in [(5.0, "well inside"), (20.0, "reaches the corner"), (40.0, "past it")] {
        let bend = Spine::new(vec![
            Point3::new(-60.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 60.0, 0.0),
        ]);
        // **Off the bisector, deliberately.** Centred *on* it, the whole arrangement — curve,
        // body and the two probe points — is symmetric about that plane, so the two sides
        // return equal values by construction and the jump reads as exactly zero however large
        // it really is. That is a blind measurement, not a passing one, and it is the shape of
        // mistake this repo has been bitten by before: a number that cannot be anything but
        // zero is not evidence.
        let big = Sphere { centre: Point3::new(-14.0, 26.0, 4.0), radius: 25.0 };
        let field = Attracted { inner: &big, spine: &bend, strength: 1.0, falloff: reach };
        let mut worst: f64 = 0.0;
        let mut at = 0.0;
        let n = 4000;
        for i in 0..n {
            let t = 60.0 * i as f64 / n as f64;
            // Two points straddling the bisector by a hair, on either side of it.
            let h = 1e-5;
            let a = Point3::new(-t - h, t - h, 0.0);
            let b = Point3::new(-t + h, t + h, 0.0);
            let jump = (field.distance(a) - field.distance(b)).abs();
            if jump > worst {
                worst = jump;
                at = t * 2.0f64.sqrt();
            }
        }
        println!("  {reach:<7.1} {:<19} {worst:<29.4} {at:.1} mm from the corner", label);
    }
    // ---------------------------------------------------------------------------------
    // The question the jump raises, and the one that actually matters: a discontinuous field
    // is not Lipschitz, so dividing by the warp's stretch does not make it a lower bound
    // there. If the contract fails at a corner then a ray following this field walks through
    // the surface, which is a hole in the picture rather than a seam on it.
    println!();
    println!("Does the lower-bound contract still hold at a corner? (0 means yes.)");
    println!();
    println!("  turn    overstatement (mm)   verdict");
    let dirs = probe_directions();
    for turn_deg in [0.0f64, 15.0, 45.0, 90.0, 150.0] {
        let half = (180.0 - turn_deg).to_radians() / 2.0;
        let bend = Spine::new(vec![
            Point3::new(-60.0 * half.sin(), 60.0 * half.cos(), 0.0),
            Point3::origin(),
            Point3::new(60.0 * half.sin(), 60.0 * half.cos(), 0.0),
        ]);
        let body = Sphere { centre: Point3::new(3.0, 18.0, 2.5), radius: 22.0 };
        let field = Attracted { inner: &body, spine: &bend, strength: 1.0, falloff: 20.0 };
        let probes = probe_points(
            Point3::new(-45.0, -12.0, -30.0),
            Point3::new(45.0, 55.0, 30.0),
            17,
        );
        let over = overstatement(&field, &probes, &dirs, 0.002, 1.2);
        let verdict = if over <= 0.004 {
            "holds"
        } else {
            "OVERSTATES — a ray walks through the surface here"
        };
        println!("  {turn_deg:<6.0}  {over:<20.4} {verdict}");
    }

    println!();
    println!("A jump means the surface can crack open along the bisector inside a bend. It is");
    println!("bounded rather than unbounded — the displacement has the same magnitude on both");
    println!("sides and only its direction differs — but it does not go away, so the honest");
    println!("statement is: an attractor aimed at a curve with a sharp corner in it will show");
    println!("a seam inside that corner, of about the size in the table above.");
}
