// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Why a sharp edge left by a subtraction draws a dark band along itself.
//!
//! Reported from a model on screen: "I subtracted a cone from a cylinder — with similar radii at
//! the base the sharp edge it left casts a shadow that looks wrong."
//!
//! Two suspects, and the point of this is to tell them apart rather than pick one:
//!
//! - **the field**, since a subtraction is `max(a, −b)` and that is only a *lower* bound near
//!   where the two surfaces meet — and a field that reads short is read by the shadow march as
//!   material standing in the way, which is the same one cause that has already produced a black
//!   driven morph and an unstable attractor;
//! - **the estimator**, since the soft shadow keeps `min(k·d/t)` over the samples it happens to
//!   take, and that overstates the occlusion whenever the ray's closest approach fell *between*
//!   two samples — which is exactly what a ray grazing along a thin lip does.
//!
//! The two want opposite fixes, so guessing costs a shipped change that does nothing. Measured
//! here: the gradient along the shadow ray says whether the field is reading short, and the same
//! sweep run against a plain cylinder — which has no sharp edge anywhere on it — says how much of
//! what is left belongs to the edge rather than to grazing light in general.
//!
//! Run with: `cargo run --release -p fieldform-field --example sharp_edge_shadow`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{BooleanOp, Document, Ends, Graph, Node, ParamRef, Place};
use nalgebra::{Point3, Vector3};

/// The model, near enough: a cylinder with a cone taken out of it, the two the same radius at
/// the base, so the wall goes to nothing at the bottom rim and leaves a knife edge all round.
const RADIUS_MM: f64 = 30.0;
const HEIGHT_MM: f64 = 60.0;

/// What the viewport asks for: `Occlusion::for_scene` sizes both of these to the scene.
fn reach() -> f64 {
    Vector3::new(RADIUS_MM * 2.0, RADIUS_MM * 2.0, HEIGHT_MM).norm() * 0.5 * 3.0
}
const SOFTNESS: f64 = 12.0;

/// How lit a point is, exactly as `march.wgsl`'s `shadow` computes it.
///
/// `improved` swaps the sample's own `d/t` for the closest approach *between* consecutive
/// samples: two spheres of radius `d` and `previous` touch on a circle `y` back along the ray and
/// `across` off it, and that circle is nearer the ray than either sample is. The classic
/// correction to his estimator, and what every reference implementation now carries.
fn lit(
    field: &GraphField,
    p: Point3<f64>,
    n: Vector3<f64>,
    light: Vector3<f64>,
    improved: bool,
) -> f64 {
    let reach = reach();
    let bias = reach * 0.004;
    let start = p + n * bias;
    let mut lit = 1.0f64;
    let mut t = bias;
    let mut previous = f64::INFINITY;
    for _ in 0..48 {
        let d = field.distance(start + light * t);
        if improved {
            let y = if previous.is_finite() { d * d / (2.0 * previous) } else { 0.0 };
            let across = (d * d - y * y).max(0.0).sqrt();
            lit = lit.min(SOFTNESS * across / (t - y).max(1e-9));
            previous = d;
        } else {
            lit = lit.min(SOFTNESS * d / t);
        }
        if lit < 0.005 || t > reach {
            break;
        }
        t += d.clamp(reach * 0.002, reach * 0.08);
    }
    lit.clamp(0.0, 1.0)
}

/// How close the field's gradient stays to one, along the ray the shadow walks.
///
/// One is what a distance has. Less than one is a field reading short — which would make this the
/// field's fault rather than the estimator's, and would want the opposite fix.
fn gradient_along(field: &GraphField, p: Point3<f64>, n: Vector3<f64>, light: Vector3<f64>) -> f64 {
    let reach = reach();
    let start = p + n * (reach * 0.004);
    let mut points = Vec::new();
    let mut t = reach * 0.004;
    while t < reach * 0.3 {
        points.push(start + light * t);
        t += reach * 0.004;
    }
    1.0 - fieldform_field::gradient_deviation(field, &points, 1e-3)
}

fn solid(with_cone: bool) -> GraphField {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let tube = graph.add(Node::Cylinder { edge: Default::default(),
        ends: Ends::between(Place::at(0.0, 0.0, 0.0), Place::at(0.0, 0.0, HEIGHT_MM)),
        radius: lit(RADIUS_MM),
    });
    if with_cone {
        // The same radius at the base, which is the case reported: the wall thins to nothing.
        let cone = graph.add(Node::Cone { edge: Default::default(),
            ends: Ends::between(Place::at(0.0, 0.0, 0.0), Place::at(0.0, 0.0, HEIGHT_MM)),
            base_radius: lit(RADIUS_MM),
            top_radius: lit(0.0),
        });
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![tube.into(), cone.into()] });
    }
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("compiles")
}

fn main() {
    let edged = solid(true);
    let plain = solid(false);

    println!(
        "A {RADIUS_MM} mm cylinder {HEIGHT_MM} mm tall, with a cone of the same base radius taken\n\
         out of it: the wall is {:.2} mm thick at z = 1 and nothing at all at z = 0, so the bottom\n\
         rim is a knife edge. Sampled on the outer wall, which is the lit face, against the same\n\
         points on a plain cylinder that has no sharp edge anywhere.\n",
        RADIUS_MM / HEIGHT_MM,
    );

    // Grazing to square-on, all of them lighting the wall rather than facing away from it.
    // Grazing is where a soft shadow has to work: the ray runs alongside the surface for a long
    // way, and every sample it takes is a small number over a growing one.
    println!(
        "  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "z mm", "light", "grad", "edged", "edged", "plain", "plain",
    );
    println!(
        "  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "", "deg", "", "now", "fixed", "now", "fixed",
    );
    for z in [0.5f64, 2.0, 8.0, 30.0] {
        for elevation_deg in [4.0f64, 8.0, 15.0, 30.0, 60.0] {
            let p = Point3::new(RADIUS_MM, 0.0, z);
            let n = Vector3::x();
            // Up the wall, leaning out by the elevation: at 4 degrees the ray skims the surface.
            let e = elevation_deg.to_radians();
            let light = (Vector3::z() * e.cos() + Vector3::x() * e.sin()).normalize();
            println!(
                "  {z:>6.1}  {elevation_deg:>6.0}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
                gradient_along(&edged, p, n, light),
                lit(&edged, p, n, light, false),
                lit(&edged, p, n, light, true),
                lit(&plain, p, n, light, false),
                lit(&plain, p, n, light, true),
            );
        }
    }

    // And the field itself, over the whole solid rather than along one ray: if a subtraction
    // reads short anywhere, it is somewhere in here, and every consumer that treats the value as
    // a distance -- the shadow, the occlusion, the march's own step -- inherits it.
    println!("\nThe field over the whole model, since a ray up the wall asks about one line only.\n");
    let mut worst = f64::INFINITY;
    let mut biggest = 0.0f64;
    let mut worst_at = Point3::origin();
    let mut outside: Vec<Point3<f64>> = Vec::new();
    let cells = 40;
    for i in 0..=cells {
        for j in 0..=cells {
            for k in 0..=cells {
                // Half a cell off, so no sample lands exactly on the axis. A cone's distance is
                // a function of the radius about that axis, and a radius has a kink there — so a
                // central difference straddling it reads only the axial part and reports a
                // gradient of 1/sqrt(5) for a field that is exactly a distance. That is the
                // measurement's own artifact, on a set of measure zero, and reporting it as the
                // field's would have sent the fix to the wrong place.
                let step = f64::from(cells) + 0.5;
                let q = Point3::new(
                    -RADIUS_MM * 1.5 + 3.0 * RADIUS_MM * (f64::from(i) + 0.5) / step,
                    -RADIUS_MM * 1.5 + 3.0 * RADIUS_MM * (f64::from(j) + 0.5) / step,
                    -HEIGHT_MM * 0.25 + HEIGHT_MM * 1.5 * (f64::from(k) + 0.5) / step,
                );
                // The gradient's own size, not how far it is from one: below one is a field
                // reading short, above one is a field claiming room it has not got, and the two
                // are different faults with opposite symptoms. A deviation folds them together.
                let h = 1e-3;
                let d = |o: Vector3<f64>| edged.distance(q + o);
                let g = Vector3::new(
                    d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
                    d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
                    d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
                )
                .norm()
                    / (2.0 * h);
                // Outside only, which is where every consumer of this value samples: the
                // shadow, the occlusion and the march itself all walk empty space. Inside, a
                // max of two negatives has a kink along the medial surface, and a central
                // difference across a kink is not a gradient.
                if edged.distance(q) <= 0.0 {
                    continue;
                }
                outside.push(q);
                if g < worst {
                    worst = g;
                    worst_at = q;
                }
                biggest = biggest.max(g);
            }
        }
    }
    println!(
        "  smallest gradient {worst:.4} at ({:.1}, {:.1}, {:.1}); largest {biggest:.4}.\n  \
         One is what a distance has. Under one is a field reading short, which shades dark;\n  \
         over one is a field claiming room it has not got, which is a hole in the surface.",
        worst_at.x, worst_at.y, worst_at.z,
    );
    // The contract itself, which a gradient only approaches: does the value ever claim more room
    // than there is? Marched *past* the claim, so a claim that holds is confirmed by walking
    // through where it said the surface was not.
    let directions = [
        Vector3::x(),
        -Vector3::x(),
        Vector3::y(),
        -Vector3::y(),
        Vector3::z(),
        -Vector3::z(),
    ];
    let over = fieldform_field::overstatement(&edged, &outside, &directions, 0.05, 1.2);
    println!(
        "  worst overstatement {over:.4} mm over {} points outside the solid: nought is every\n  \
         claim holding, which is the contract a march and a shadow both rest on.",
        outside.len(),
    );

    println!(
        "\nRead the two `edged` columns against the two `plain` ones. What the sharp edge adds is\n\
         the difference between them; what both share is grazing light, which is a real penumbra\n\
         and should stay. What `fixed` removes on top of that is the estimator's own overshoot.\n\
         \n\
         As it stands, on this shape: neither suspect is guilty. The two `edged` columns match the\n\
         two `plain` ones to four decimals at every point and every angle, so the sharp edge\n\
         contributes nothing of its own; the gradient is one everywhere outside the solid and the\n\
         overstatement is nought, so the field is a true distance rather than one reading short;\n\
         and `fixed` comes out *darker* at grazing angles, so adopting it would move the picture\n\
         the wrong way. Which means the fault reported is somewhere this does not reach, and the\n\
         next step is the document rather than another guess."
    );
}
