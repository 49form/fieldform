// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a blend of radius k actually measures, and what the exponent does to the join.
//!
//! Two perpendicular walls (`a = x`, `b = y`, both exact distance fields near the corner) are
//! blended, and the resulting profile is measured two independent ways:
//!
//! - **by position** — a rolling ball of radius r tangent to both walls touches the diagonal
//!   at `r(1 − 1/√2)` from the corner, so the crossing point implies a radius;
//! - **by curvature** — the implicit curve's curvature at that point implies another.
//!
//! A circular fillet makes the two agree. A blend that is not an arc does not, which is how
//! the polynomial smooth minimum that used to be here was found to have no radius at all: the
//! same k measured 0.707 k one way and 0.854 k the other.
//!
//! Run with: `cargo run --release -p fieldform-field --example blend_profile`

/// The blend this codebase evaluates, for an arbitrary exponent — a copy of `smooth_min` in
/// `graph_field.rs`, which is private. Kept in step by the tables below: if the two ever
/// disagree, `the_fillet_is_a_circle_of_the_radius_it_was_asked_for` fails first.
fn smooth_min(a: f64, b: f64, k: f64, n: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let x = (k - a).max(0.0);
    let y = (k - b).max(0.0);
    k.max(a.min(b)) - (x.powf(n) + y.powf(n)).powf(1.0 / n)
}

/// The polynomial smooth minimum (the classic form) this replaced, for comparison.
fn polynomial_smin(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    (b + (a - b) * h) - k * h * (1.0 - h)
}

/// Where the blended surface crosses the line `x = y`.
fn apex(f: &dyn Fn(f64, f64) -> f64, k: f64) -> f64 {
    let (mut lo, mut hi) = (-4.0 * k, k);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid, mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Curvature of the implicit curve `f = 0` at `(at, at)`.
fn curvature(f: &dyn Fn(f64, f64) -> f64, k: f64, at: f64) -> f64 {
    let h = k * 1e-4;
    let fx = (f(at + h, at) - f(at - h, at)) / (2.0 * h);
    let fy = (f(at, at + h) - f(at, at - h)) / (2.0 * h);
    let fxx = (f(at + h, at) - 2.0 * f(at, at) + f(at - h, at)) / (h * h);
    let fyy = (f(at, at + h) - 2.0 * f(at, at) + f(at, at - h)) / (h * h);
    let fxy = (f(at + h, at + h) - f(at + h, at - h) - f(at - h, at + h) + f(at - h, at - h))
        / (4.0 * h * h);
    let num = fx * fx * fyy - 2.0 * fx * fy * fxy + fy * fy * fxx;
    let den = (fx * fx + fy * fy).powf(1.5);
    (num / den).abs()
}

/// Curvature at a point on the profile above `x`, for reading the join.
fn curvature_at(f: &dyn Fn(f64, f64) -> f64, k: f64, x: f64) -> f64 {
    let (mut lo, mut hi) = (-4.0 * k, k);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(x, mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let y = 0.5 * (lo + hi);
    let h = k * 1e-4;
    let fx = (f(x + h, y) - f(x - h, y)) / (2.0 * h);
    let fy = (f(x, y + h) - f(x, y - h)) / (2.0 * h);
    let fxx = (f(x + h, y) - 2.0 * f(x, y) + f(x - h, y)) / (h * h);
    let fyy = (f(x, y + h) - 2.0 * f(x, y) + f(x, y - h)) / (h * h);
    let fxy =
        (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4.0 * h * h);
    let num = fx * fx * fyy - 2.0 * fx * fy * fxy + fy * fy * fxx;
    let den = (fx * fx + fy * fy).powf(1.5);
    (num / den).abs()
}

fn main() {
    println!("Does the radius mean the radius? Two measurements of the same fillet.\n");
    println!(
        "{:>6}  {:>22}  {:>14}  {:>16}",
        "k", "blend", "by curvature", "by position"
    );
    println!("{}", "-".repeat(64));
    for k in [1.0, 5.0, 20.0] {
        let circular = |a: f64, b: f64| smooth_min(a, b, k, 2.0);
        let polynomial = |a: f64, b: f64| polynomial_smin(a, b, k);
        for (label, f) in [
            ("G1 (this build)", &circular as &dyn Fn(f64, f64) -> f64),
            ("polynomial (was)", &polynomial as &dyn Fn(f64, f64) -> f64),
        ] {
            let a = apex(f, k);
            let by_curvature = 1.0 / curvature(f, k, a);
            // A ball of radius r tangent to both walls touches at r(1 - 1/sqrt 2).
            let by_position = a / (1.0 - 1.0 / 2f64.sqrt());
            println!(
                "{k:>6.1}  {label:>22}  {:>11.3} k  {:>13.3} k",
                by_curvature / k,
                by_position / k
            );
        }
    }

    println!("\nCurvature along the profile, from the tangency point in toward the apex.");
    println!("A circle carries its curvature right up to the wall — that step from zero is what");
    println!("makes it G1. A larger exponent lets it reach the wall at zero, which is G2.\n");
    let k = 10.0;
    print!("{:>8}", "x");
    for n in [2.0, 2.5, 3.0, 4.0] {
        print!("{:>14}", format!("n={n}"));
    }
    println!();
    println!("{}", "-".repeat(64));
    for x in [9.99, 9.9, 9.5, 9.0, 8.0, 6.0, 4.0] {
        print!("{x:>8.2}");
        for n in [2.0, 2.5, 3.0, 4.0] {
            let f = |a: f64, b: f64| smooth_min(a, b, k, n);
            print!("{:>14.5}", curvature_at(&f, k, x));
        }
        println!();
    }

    println!("\nApex radius per exponent, as a multiple of k:");
    for n in [2.0, 2.5, 3.0, 4.0] {
        let f = |a: f64, b: f64| smooth_min(a, b, k, n);
        let a = apex(&f, k);
        println!("  n={n:<4} {:.3} k", 1.0 / curvature(&f, k, a) / k);
    }
    println!();
    println!("Only n = 2 has one radius by both measures, which is why G1 is the exponent the");
    println!("radius control means. A G2 blend would report its reach along the wall instead.");
}
