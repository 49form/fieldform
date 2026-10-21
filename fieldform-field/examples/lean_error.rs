// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How far short a leaning extrude's distance reads, measured against the truth.
//!
//! A sweep that is not along the profile's normal is a shear, and a sheared distance is not a
//! distance: the field slides back along the sweep to reach the plane, which stretches the
//! measurement, so the reading is multiplied by `|d·n|` to become a lower bound again. That
//! makes it *safe* — it never overstates — but it leaves the question anybody actually asks,
//! which is how far off it is.
//!
//! Answered here by brute force: sample points around a leaning solid, ask the field, and ask
//! separately for the nearest point that is genuinely inside. The ratio of the two is the
//! shortfall, and it is reported per lean angle so the number the panel shows can be a measured
//! one rather than a plausible one.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example lean_error`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Curve, CurveId, Document, Graph, Node, ParamRef, Sweep};
use nalgebra::Point3;

const HALF: f64 = 10.0;
const DEPTH: f64 = 20.0;

fn main() {
    println!("A {}x{} square swept {DEPTH} deep, leaning in X.\n", HALF * 2.0, HALF * 2.0);

    // **First, that the solid is the size it was asked for**, because the warning this table
    // feeds says a percentage and a percentage in a panel beside a depth reads as a percentage
    // *of the depth*. It is not. The shortfall is in what the field reports about the space
    // around the solid; the solid itself is exactly as long as it was told to be, at every
    // lean, and this measures that rather than asserting it.
    println!("  lean    depth asked   depth measured along the sweep");
    for degrees in [0.0f64, 10.0, 32.0, 45.0, 60.0, 75.0] {
        println!("  {degrees:>4.0}°   {DEPTH:>11.3}   {:>29.3}", measured_depth(degrees));
    }

    println!("\n  lean    cos(lean)   worst reading / truth   where");
    for degrees in [0.0f64, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0] {
        let (ratio, at) = worst_ratio(degrees);
        println!(
            "  {degrees:>4.0}°   {:>8.3}   {ratio:>21.3}   ({:.0}, {:.0}, {:.0})",
            degrees.to_radians().cos(),
            at.x,
            at.y,
            at.z,
        );
    }
    println!(
        "\nThe ratio is what the field reports divided by the true distance, so 1.000 is exact\n\
         and 0.800 means it reads twenty per cent short. Compare it against cos(lean)."
    );
}

/// How long the solid actually is along its sweep, found by marching until it ends.
///
/// The depth is measured *along the sweep direction*, so this walks that direction from the
/// plane and reports where the material stops -- which is the number a user typed, or is not.
fn measured_depth(degrees: f64) -> f64 {
    let lean = degrees.to_radians();
    let dir = nalgebra::Vector3::new(lean.sin(), 0.0, lean.cos());
    let field = leaning(degrees);
    // Along the sweep from the middle of the profile, in steps far finer than the answer needs.
    let step = 1e-4;
    let mut t = 0.0;
    while t < DEPTH * 3.0 {
        if field.distance(Point3::origin() + dir * (t + step)) > 0.0 {
            break;
        }
        t += step;
    }
    t + step
}

/// The worst reading-to-truth ratio over a grid around the solid, and where it happened.
fn worst_ratio(degrees: f64) -> (f64, Point3<f64>) {
    let lean = degrees.to_radians();
    let field = leaning(degrees);
    // **The solid's own surface, generated rather than searched for.** The first version of
    // this compared against a grid of points found to be inside, and reported 0.818 at zero
    // lean -- where the field is exact. That was the grid's spacing, not the field: at the
    // point it flagged, the nearest *sampled* inside point is 3.85 away and the nearest real
    // one is 3.16, which is the ratio it printed. A reference has to be at least as accurate as
    // the thing it judges, and a lattice of samples is not.
    //
    // This solid is a parallelepiped and its boundary is known in closed form: the profile
    // square slid along the sweep. So the surface is enumerated exactly -- two caps and four
    // sides -- and the only error left is the spacing along it.
    //
    // **From the plane, not about it.** This said `[-DEPTH/2, +DEPTH/2]` while an extrude
    // straddled its plane, and kept saying it after one started at the plane instead -- so the
    // reference was a solid twenty millimetres from the one being measured, and the ratios it
    // printed were the gap between them rather than anything about a lean. It showed as 0.371
    // at *zero* lean, where the field is exact by construction and the only possible answer is
    // 1.000. A reference that disagrees with a case whose answer is known is reporting on
    // itself; this row is the check that says so, which is why the table below starts at zero.
    let dir = nalgebra::Vector3::new(lean.sin(), 0.0, lean.cos());
    let mut surface = Vec::new();
    let step = 0.25;
    let n = (HALF * 2.0 / step) as i32;
    let m = (DEPTH / step) as i32;
    for i in 0..=n {
        let a = -HALF + i as f64 * step;
        // The two caps, over the whole square.
        for j in 0..=n {
            let b = -HALF + j as f64 * step;
            for t in [0.0, DEPTH] {
                surface.push(Point3::new(a, b, 0.0) + dir * t);
            }
        }
        // The four sides, over the depth.
        for k in 0..=m {
            let t = k as f64 * step;
            for q in [
                Point3::new(a, -HALF, 0.0),
                Point3::new(a, HALF, 0.0),
                Point3::new(-HALF, a, 0.0),
                Point3::new(HALF, a, 0.0),
            ] {
                surface.push(q + dir * t);
            }
        }
    }

    let mut worst = (f64::INFINITY, Point3::origin());
    for i in 0..15 {
        for j in 0..15 {
            for k in 0..15 {
                // Centred on the solid rather than on the origin, since the solid is no
                // longer centred on the origin either.
                let p = Point3::new(
                    -35.0 + i as f64 * 5.0,
                    -35.0 + j as f64 * 5.0,
                    -20.0 + k as f64 * 4.3,
                );
                let claimed = field.distance(p);
                // Outside only, and far enough out that the surface spacing is negligible
                // beside the distance being judged.
                if claimed < 3.0 {
                    continue;
                }
                let truth = surface
                    .iter()
                    .map(|q| (q - p).norm())
                    .fold(f64::INFINITY, f64::min);
                if truth > 1e-9 && claimed / truth < worst.0 {
                    worst = (claimed / truth, p);
                }
            }
        }
    }
    worst
}

/// The square, swept `degrees` off its own normal. One builder, so the two measurements above
/// cannot end up describing different solids.
fn leaning(degrees: f64) -> GraphField {
    let mut doc = Document::new();
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            "Square".to_string(),
            vec![
                Point3::new(-HALF, -HALF, 0.0),
                Point3::new(HALF, -HALF, 0.0),
                Point3::new(HALF, HALF, 0.0),
                Point3::new(-HALF, HALF, 0.0),
            ],
            true,
        )
        .expect("closed"),
    );
    let lean = degrees.to_radians();
    let mut graph = Graph::empty();
    graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(id),
        sweep: Sweep::Toward {
            xyz: [
                ParamRef::Literal(lean.sin()),
                ParamRef::Literal(0.0),
                ParamRef::Literal(lean.cos()),
            ],
        },
        height: ParamRef::Literal(DEPTH),
        draft: ParamRef::Literal(0.0),
        start: ParamRef::Literal(0.0),
        sides: fieldform_scene::Sides::One,
        corners: Default::default(),
    });
    GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles")
}
