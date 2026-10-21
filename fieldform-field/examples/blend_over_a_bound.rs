// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a fillet becomes when one of the two bodies it joins reads short.
//!
//! A leaning sweep is a shear, and a sheared distance is divided by a factor to stay a lower
//! bound — so the body reports less room beside it than there is. `lean_error.rs` measures that
//! factor. This measures what it does to a blend, which is the only place a user meets it.
//!
//! Worth measuring rather than reasoning about, because "the fillet comes out bigger" is the
//! wrong shape of answer. `Node::Blend` *is* the other wall beyond `k` — for `a >= k` the
//! formula collapses to exactly `b` — so the join is tangent where `a = k`, and an operand
//! reading `s·a` is tangent where `a = k/s`. In the arc itself the surface satisfies
//! `(k − s·x)² + (k − y)² = k²`, which is an ellipse with semi-axes `k/s` and `k`, not a circle.
//!
//! So the fillet is not scaled. It is stretched along one axis, which spreads its footprint
//! across the *true* body and leaves its footprint on the leaning one alone, and leaves it with
//! no single radius at all.
//!
//! Run with: `cargo run --release -p fieldform-field --example blend_over_a_bound`

/// The blend this codebase evaluates — a copy of `smooth_min` in `graph_field.rs`, which is
/// private. `blend_profile.rs` carries the same copy and the same warning: if these drift from
/// the original, `the_fillet_is_a_circle_of_the_radius_it_was_asked_for` fails first.
fn smooth_min(a: f64, b: f64, k: f64, n: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let x = (k - a).max(0.0);
    let y = (k - b).max(0.0);
    k.max(a.min(b)) - (x.powf(n) + y.powf(n)).powf(1.0 / n)
}

const K: f64 = 5.0;

fn main() {
    println!(
        "A {K} mm fillet where two bodies meet, one of them a leaning extrude whose distance\n\
         reads short by the factor s that `lean_error.rs` measures. G1 (n = 2), which is the\n\
         exact circular case at s = 1.\n"
    );
    println!(
        "  {:>5}  {:>7}   {:>14}  {:>14}   {:>9}  {:>9}",
        "lean", "s", "spreads across", "spreads across", "tightest", "flattest"
    );
    println!(
        "  {:>5}  {:>7}   {:>14}  {:>14}   {:>9}  {:>9}",
        "", "", "the true body", "the leaner", "radius", "radius"
    );
    for lean in [0.0f64, 10.0, 20.0, 32.0, 45.0] {
        let s = fieldform_field::lean_shortfall(lean);
        let f = move |x: f64, y: f64| smooth_min(s * x, y, K, 2.0);
        // `y = 0` is the true body's own surface, and `x = 0` is the leaner's. Walking each one
        // out from the corner finds where the blend rejoins it, which is a real edge rather
        // than an asymptote: beyond the tangent point the blend *equals* the wall.
        let across_true = rejoins(&f, |t| (t, 0.0));
        let across_leaner = rejoins(&f, |t| (0.0, t));
        let (tight, flat) = radius_range(&f, s);
        println!(
            "  {lean:>4.0}°  {s:>7.3}   {across_true:>14.2}  {across_leaner:>14.2}   \
             {tight:>9.2}  {flat:>9.2}"
        );
    }
    println!(
        "\nThe s = 1 row is all {K}s, which is what a {K} mm fillet means: tangent to both bodies\n\
         {K} from the corner, one radius throughout. That row is the check on the rest — a\n\
         measurement that cannot reproduce the case with a known answer is measuring itself.\n\n\
         Below it the fillet is not a larger fillet. Its footprint grows across the *true* body\n\
         while staying put on the leaning one, and its radius stops being a radius: tighter than\n\
         asked at one end and flatter at the other. An ordinary CAD fillet's footprint also\n\
         varies — with the angle between the faces — but its radius is whatever was typed. This\n\
         one's is not, which is the part that would not be predicted."
    );
}

/// How far along a body's own surface the blend runs before rejoining it, from the corner.
fn rejoins(f: &dyn Fn(f64, f64) -> f64, walk: impl Fn(f64) -> (f64, f64)) -> f64 {
    let departs = |t: f64| {
        let (x, y) = walk(t);
        f(x, y).abs() > 1e-9
    };
    let (mut lo, mut hi) = (0.0, 40.0 * K);
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if departs(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// The tightest and flattest radius the blended arc actually turns through.
///
/// Sampled **inside the arc only** — where both readings are under `k` — because outside it the
/// surface is a flat wall and its curvature is not the fillet's. A first attempt sampled a band
/// across the corner instead and reported 10.00 for the flattest radius at s = 1, where the
/// answer is 5.00 by construction: it had walked off the arc onto the wall.
fn radius_range(f: &dyn Fn(f64, f64) -> f64, s: f64) -> (f64, f64) {
    let (mut tight, mut flat) = (f64::INFINITY, 0.0f64);
    let h = 1e-5;
    for step in 1..200 {
        // Across the arc's own span in y, which runs from the corner to the tangent point.
        let y = K * step as f64 / 200.0;
        // And the x on the surface at that height, in closed form from the arc's equation, so
        // the sample is on the surface rather than near it.
        let inner = K * K - (K - y) * (K - y);
        if inner <= 0.0 {
            continue;
        }
        let x = (K - inner.sqrt()) / s;
        if !(x.is_finite() && f(x, y).abs() < 1e-6) {
            continue;
        }
        let fx = (f(x + h, y) - f(x - h, y)) / (2.0 * h);
        let fy = (f(x, y + h) - f(x, y - h)) / (2.0 * h);
        let fxx = (f(x + h, y) - 2.0 * f(x, y) + f(x - h, y)) / (h * h);
        let fyy = (f(x, y + h) - 2.0 * f(x, y) + f(x, y - h)) / (h * h);
        let fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h))
            / (4.0 * h * h);
        let grad2 = fx * fx + fy * fy;
        let num = fx * fx * fyy - 2.0 * fx * fy * fxy + fy * fy * fxx;
        if grad2 < 1e-12 || num.abs() < 1e-9 {
            continue;
        }
        let radius = (grad2.powf(1.5) / num).abs();
        if !radius.is_finite() || radius > 1e5 {
            continue;
        }
        tight = tight.min(radius);
        flat = flat.max(radius);
    }
    (tight, flat)
}
