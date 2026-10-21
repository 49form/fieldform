// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Sharp corners on a drafted wall, measured before anything is built on the idea.
//!
//! Asked by the owner, 2026-10-03: *"as for sharp vs round on drafted corners - can we have
//! both? and have a toggle to turn on sharp?"* A draft is an **offset** of the section, and an
//! offset of a convex corner is an arc by definition -- the corner rounds, by the offset's own
//! amount, and grows rounder as the draft grows. A mitred corner is what a moulding drawing
//! usually means instead: the two walls stay planes and meet at an edge.
//!
//! # The method this measures
//!
//! The rounded offset is `d(q) − t`, where `d` is the section's own distance. The mitred one is
//! **the same walk over displaced vertices**: at each corner the two offset lines meet at
//! `v + t·m`, with `m = (n₁ + n₂) / (1 + n₁·n₂)` the standard miter vector of the two edges'
//! outward normals. The miter vectors are a property of the section and are computed once; the
//! per-query cost is then the walk it always was, with each vertex moved. If that holds, the
//! toggle is cheap on both sides -- the shader does the same displacement from the same
//! numbers.
//!
//! Four things have to be true, and this prints each:
//!
//! 1. **It is the mitre.** At a corner the sharp contour stands `t/sin(θ/2)` from the vertex
//!    along the bisector, where the round one stands `t`.
//! 2. **It is still a distance**, not a bound: the reading never overstates and never
//!    understates by more than the polygon walk's own exactness.
//! 3. **The walls still lean at the draft angle**, which is what makes it a draft at all --
//!    the mitre edge is steeper in plan, and the faces either side of it are not.
//! 4. **Where it stops working**: as a corner sharpens, `1 + n₁·n₂ → 0` and the miter runs
//!    away. The limit is printed as the angle at which the mitre is four offsets long, which
//!    is where every other CAD tool bevels instead.
//!
//! Run with: `cargo run --release -p fieldform-field --example mitred_draft`

use nalgebra::{Point2, Vector2};

/// The outward normal of the edge `a → b`, for a polygon wound counter-clockwise.
fn outward(a: Point2<f64>, b: Point2<f64>) -> Vector2<f64> {
    let e = (b - a).normalize();
    Vector2::new(e.y, -e.x)
}

/// Twice the signed area: positive when the polygon is wound counter-clockwise.
fn winding(points: &[Point2<f64>]) -> f64 {
    let n = points.len();
    (0..n)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum()
}

/// The miter vector at each vertex: where that corner goes, per millimetre of offset.
///
/// `(n₁ + n₂) / (1 + n₁·n₂)` is the point where the two offset lines meet, which is the whole
/// of the mitre. Its length is `1/cos(half the turn)`, so it runs away as a corner sharpens --
/// `limit` is what that is held to, and a corner past it keeps the round answer.
fn miters(points: &[Point2<f64>], limit: f64) -> Vec<Vector2<f64>> {
    let n = points.len();
    let flip = if winding(points) < 0.0 { -1.0 } else { 1.0 };
    (0..n)
        .map(|i| {
            let before = points[(i + n - 1) % n];
            let here = points[i];
            let after = points[(i + 1) % n];
            let n1 = outward(before, here) * flip;
            let n2 = outward(here, after) * flip;
            let share = 1.0 + n1.dot(&n2);
            if share < 1e-9 {
                return Vector2::zeros();
            }
            let m = (n1 + n2) / share;
            if m.norm() > limit { m.normalize() * limit } else { m }
        })
        .collect()
}

/// The plain polygon distance: negative inside. `Profile::walk`'s own arithmetic, over
/// whatever points it is handed -- which is the whole trick.
fn walk(points: &[Point2<f64>], q: Point2<f64>) -> f64 {
    let n = points.len();
    let mut nearest = f64::INFINITY;
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        let ab = b - a;
        let t = if ab.norm_squared() > 0.0 {
            ((q - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        nearest = nearest.min((q - (a + ab * t)).norm());
        if (a.y > q.y) != (b.y > q.y) {
            let x = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if q.x < x {
                inside = !inside;
            }
        }
    }
    if inside { -nearest } else { nearest }
}

/// The section offset outward by `t` with its corners mitred: the same walk, moved vertices.
fn sharp(points: &[Point2<f64>], miters: &[Vector2<f64>], t: f64, q: Point2<f64>) -> f64 {
    let moved: Vec<Point2<f64>> =
        points.iter().zip(miters).map(|(p, m)| p + m * t).collect();
    walk(&moved, q)
}

fn main() {
    // An L: four right angles out and one in, so a reflex corner is in the measurement too.
    let l: Vec<Point2<f64>> = [
        (0.0, 0.0),
        (40.0, 0.0),
        (40.0, 15.0),
        (18.0, 15.0),
        (18.0, 40.0),
        (0.0, 40.0),
    ]
    .iter()
    .map(|&(x, y)| Point2::new(x, y))
    .collect();
    let limit = 4.0;
    let m = miters(&l, limit);
    println!("an L of {} corners; miter limit {limit}\u{d7}", l.len());
    for (i, (p, v)) in l.iter().zip(&m).enumerate() {
        println!(
            "  corner {i} at ({:.0}, {:.0}): miter ({:+.3}, {:+.3}), {:.3}\u{d7} the offset",
            p.x,
            p.y,
            v.x,
            v.y,
            v.norm()
        );
    }

    // 1. It is the mitre: at the square corner (40, 0) the sharp contour stands t·√2 out along
    //    the bisector, where the round one stands t.
    let t = 6.0;
    let corner = Point2::new(40.0, 0.0);
    let bisector = Vector2::new(1.0, -1.0).normalize();
    println!("\n1. the corner at (40, 0), offset {t} mm along its bisector:");
    for (what, reach) in [("round", t), ("sharp", t * std::f64::consts::SQRT_2)] {
        let at = corner + bisector * reach;
        let round = walk(&l, at) - t;
        let mitred = sharp(&l, &m, t, at);
        println!(
            "   {reach:.3} mm out ({what}'s own reach): round reads {round:+.4}, sharp reads \
             {mitred:+.4}"
        );
    }

    // 2. The two agree where there is no corner to disagree about, and the sharp solid holds
    //    everything the round one does. **Not** against the displaced polygon, which is what
    //    `sharp` is -- comparing a function with itself measures nothing, and the first draft
    //    of this did exactly that.
    let moved: Vec<Point2<f64>> = l.iter().zip(&m).map(|(p, v)| p + v * t).collect();
    let mut worst_face: f64 = 0.0;
    let mut smallest: f64 = f64::INFINITY;
    let mut smallest_at = Point2::origin();
    for i in -30..=90 {
        for j in -30..=90 {
            let q = Point2::new(i as f64 * 0.7, j as f64 * 0.7);
            let (round, mitred) = (walk(&l, q) - t, sharp(&l, &m, t, q));
            // Everywhere: mitring a corner adds material and never takes it away, so the
            // sharp reading is never the larger. The margin is what the corner is worth.
            if round - mitred < smallest {
                smallest = round - mitred;
                smallest_at = q;
            }
        }
    }
    // Along the outward normal from the middle of each edge, where the nearest thing is that
    // face whichever way the corners are cut: the two must be the same number. (Asking this of
    // a grid instead is what the first draft did, and it measured the corners it meant to
    // exclude -- a point far out past a corner's bisector is still that corner's.)
    let flip = if winding(&l) < 0.0 { -1.0 } else { 1.0 };
    for i in 0..l.len() {
        let (a, b) = (l[i], l[(i + 1) % l.len()]);
        let mid = nalgebra::center(&a, &b);
        let out = outward(a, b) * flip;
        for away in [0.5, 3.0, 9.0] {
            let q = mid + out * away;
            let gap = ((walk(&l, q) - t) - sharp(&l, &m, t, q)).abs();
            if gap > 1e-9 {
                println!(
                    "   face {i} mid ({:.1}, {:.1}) out {away}: round {:+.3} sharp {:+.3}",
                    mid.x, mid.y, walk(&l, q) - t, sharp(&l, &m, t, q)
                );
            }
            worst_face = worst_face.max(gap);
        }
    }
    println!(
        "\n2. off the middle of every face the two agree to {worst_face:.2e} mm, and the sharp \
         solid never holds less than the round one (worst {smallest:+.2e} mm at ({:.1}, {:.1}))",
        smallest_at.x, smallest_at.y
    );
    // The displaced boundary has to stay a simple polygon, or the walk's inside test is
    // reading a shape that crosses itself. Checked by brute force on the non-adjacent pairs.
    let crossing = |a: Point2<f64>, b: Point2<f64>, c: Point2<f64>, d: Point2<f64>| {
        let side = |p: Point2<f64>, q: Point2<f64>, r: Point2<f64>| {
            (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)
        };
        side(a, b, c) * side(a, b, d) < 0.0 && side(c, d, a) * side(c, d, b) < 0.0
    };
    let n = moved.len();
    let mut crossings = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if j == i || (j + 1) % n == i || (i + 1) % n == j {
                continue;
            }
            if crossing(moved[i], moved[(i + 1) % n], moved[j], moved[(j + 1) % n]) {
                crossings += 1;
            }
        }
    }
    println!("   the displaced boundary crosses itself {crossings} times at this offset");

    // 3. The walls still lean at the draft angle. At height z the contour is `z·tan(draft)`, so
    //    a wall's face moves that far per millimetre of height -- measured square to the face,
    //    away from a mitre edge.
    println!("\n3. the wall's lean, measured off the middle of the long face (y = 0):");
    for draft in [3.0f64, 10.0, 25.0] {
        let tan = draft.to_radians().tan();
        let face = Point2::new(20.0, 0.0);
        let out = Vector2::new(0.0, -1.0);
        let rise = 10.0;
        let (lo, hi) = (rise * 0.0, rise);
        let edge_at = |z: f64| {
            // Where the contour crosses this ray, found by bisection on the sharp reading.
            let (mut a, mut b) = (0.0, 40.0);
            for _ in 0..80 {
                let mid = 0.5 * (a + b);
                if sharp(&l, &m, z * tan, face + out * mid) < 0.0 {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            0.5 * (a + b)
        };
        let measured = ((edge_at(hi) - edge_at(lo)) / rise).atan().to_degrees();
        println!("   draft {draft:>4.0}\u{b0}: the face leans {measured:.4}\u{b0}");
    }

    // 4. Where the mitre runs away. The miter's length is 1/cos(half the turn), so a corner
    //    that turns through `a` has a mitre `1/sin(a/2)` long -- 1.41 at a right angle, 3.86 at
    //    15 degrees of included angle.
    println!("\n4. how long the mitre is, by the corner's included angle:");
    for included in [150.0f64, 90.0, 60.0, 30.0, 15.0, 5.0] {
        let reach = 1.0 / (included.to_radians() * 0.5).sin();
        let held = reach.min(limit);
        println!(
            "   {included:>5.0}\u{b0}: {reach:>7.2}\u{d7} the offset{}",
            if reach > limit { format!(", held to {held:.2}\u{d7}") } else { String::new() }
        );
    }
}
