// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Why a driven morph goes missing where the surface turns away from the camera.
//!
//! Reported from a model on screen: a sphere morphed to a cube with the amount driven by a ramp,
//! `tracing cost ×8.5`, drawn with holes eaten out of it along the silhouette — the top edge and
//! the right side, exactly where the surface is most nearly edge-on. Everything facing the camera
//! is fine.
//!
//! Two suspects, and the point of this is to tell them apart rather than to pick one:
//!
//! - the **step budget**, since a divided field reports 8.5× short and so takes 8.5× as many
//!   steps to cross the same millimetre, while `STEP_BUDGET` stayed at 512;
//! - the **hit confirmation**, since a grazing ray's value dips and comes back up without ever
//!   going negative, which is the shape `CONFIRM` was written to reject.
//!
//! **A scanline rather than a chosen ray.** Aiming at a point known to be on the surface makes
//! every ray converge and measures nothing: the pixels in question are the ones whose rays pass
//! *near* the surface, which is a sweep across the silhouette. So this walks a row of rays by
//! impact parameter, exactly as a row of pixels does, and reports where the hits stop.
//!
//! The march below is `march.wgsl` transcribed: same `STEP_SAFETY`, same `CONFIRM`, same
//! `epsilon`, same order of tests. A difference between it and the shader would make the whole
//! measurement meaningless, so it is copied rather than approximated.
//!
//! Run with: `cargo run --release -p fieldform-field --example grazing_rays`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Document, Graph, Node, ParamRef, Place, RampShape};
use nalgebra::{Point3, Vector3};

/// `march.wgsl`'s constants, which this has to match to be evidence of anything.
const STEP_SAFETY: f64 = 0.9;
const CONFIRM: f64 = 4.0;
/// `traced/mod.rs`'s `STEP_BUDGET`.
const BUDGET: u32 = 512;

/// What a march did, rather than only whether it hit.
struct March {
    hit: bool,
    steps: u32,
    /// Steps spent on the `CONFIRM` branch: a candidate looked at and not believed.
    rejected: u32,
    /// How close the ray ever came to the surface, in millimetres of reported value.
    closest: f64,
    /// True where the loop ended because `budget` ran out rather than because it passed `far`.
    exhausted: bool,
}

/// `march.wgsl`'s loop, line for line.
fn march(
    field: &dyn ScalarField,
    origin: Point3<f64>,
    dir: Vector3<f64>,
    far: f64,
    eps: f64,
    budget: u32,
) -> March {
    let mut t = 0.0;
    let mut steps = 0;
    let mut rejected = 0;
    let mut closest = f64::INFINITY;
    let mut past_far = false;
    for _ in 0..budget {
        steps += 1;
        if t > far {
            past_far = true;
            break;
        }
        let d = field.distance(origin + dir * t);
        closest = closest.min(d);
        if d < eps {
            let ahead = field.distance(origin + dir * (t + CONFIRM * eps));
            if ahead < 0.0 || ahead < d {
                return March { hit: true, steps, rejected, closest, exhausted: false };
            }
            rejected += 1;
            t += CONFIRM * eps;
            continue;
        }
        t += (d * STEP_SAFETY).max(eps);
    }
    March { hit: false, steps, rejected, closest, exhausted: !past_far && steps >= budget }
}

/// The scene from `morph-variable2.fieldform`, rebuilt so this runs without the file.
fn scene(driven: bool) -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let centre = Place::At([lit(0.0), lit(0.0), lit(0.0)]);
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere { centre: centre.clone(), radius: lit(50.0) });
    let block = graph.add(Node::Cuboid { edge: Default::default(),
        centre,
        size: [lit(100.0), lit(100.0), lit(100.0)],
        turn: fieldform_scene::no_turn(),
    });
    let t = match driven {
        false => lit(0.5),
        true => {
            let point = graph.add(Node::Point { at: [lit(0.0), lit(0.0), lit(0.0)] });
            let moved = graph.add(Node::Transform {
                input: point.into(),
                translation: [lit(0.0), lit(-44.333_342_220_994_07), lit(0.0)],
                rotation: [lit(0.0), lit(0.0), lit(0.0)],
                scale: lit(1.0),
                about: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            });
            let ramp = graph.add(Node::Ramp {
                input: moved.into(),
                in_min: lit(17.0),
                in_max: lit(86.602_540_378_443_86),
                out_min: lit(0.0),
                out_max: lit(1.0),
                shape: RampShape::Smooth,
            });
            ParamRef::Field(ramp)
        }
    };
    graph.add(Node::Morph { a: ball.into(), b: block.into(), t });
    (Document::new(), graph)
}

/// One row of the picture: rays down -Z, offset across X, as a column of pixels through the edge.
fn scanline(field: &dyn ScalarField, eps: f64, budget: u32, offsets: &[f64]) -> Vec<(f64, March)> {
    offsets
        .iter()
        .map(|&x| {
            let origin = Point3::new(x, 0.0, 400.0);
            let dir = Vector3::new(0.0, 0.0, -1.0);
            (x, march(field, origin, dir, 900.0, eps, budget))
        })
        .collect()
}

fn main() {
    // The model is 100 mm across, so this is the viewport's own epsilon at that size.
    let radius = 86.602_540_378_443_86;
    let eps = radius * 1e-5;

    println!(
        "A 50 mm sphere morphed to a 100 mm cube, the amount a smooth ramp over 17..86.6 mm from\n\
         a point 44.3 mm below centre -- the reported scene, which the panel prices at x8.5.\n\
         Epsilon {eps:.5} mm, budget {BUDGET} steps, both as the viewport uses them.\n\n\
         A row of rays straight down -Z, offset across X. The body reaches to x = 50, so the\n\
         silhouette is there and the rows below walk up to it and past it.\n"
    );

    // Dense right at the limb, since that is where the picture breaks.
    let offsets: Vec<f64> = (0..=10)
        .map(|i| i as f64 * 4.0)
        .chain([42.0, 44.0, 46.0, 47.0, 48.0, 49.0, 49.5, 49.9, 50.1, 51.0, 54.0])
        .collect();

    for driven in [false, true] {
        let (document, graph) = scene(driven);
        let field = GraphField::compile_graph(&document, &graph, 0.25).expect("compiles");
        let morph = graph.nodes().find(|(_, n)| matches!(n, Node::Morph { .. })).unwrap().0;
        let divisor = fieldform_field::driven_morph_divisor(&document, &graph, morph);
        println!(
            "{}  (divisor x{divisor:.1})",
            match driven {
                false => "--- constant amount ---",
                true => "--- amount varies through space ---",
            }
        );
        println!(
            "  {:>8}  {:>6}  {:>6}  {:>8}  {:>9}  {:>8}",
            "x mm", "hit", "steps", "rejected", "closest", "ran out"
        );
        for (x, out) in scanline(&field, eps, BUDGET, &offsets) {
            println!(
                "  {x:>8.1}  {:>6}  {:>6}  {:>8}  {:>9.5}  {:>8}",
                match out.hit {
                    true => "yes",
                    false => "NO",
                },
                out.steps,
                out.rejected,
                out.closest,
                match out.exhausted {
                    true => "YES",
                    false => "-",
                }
            );
        }
        println!();
    }

    // ---------------------------------------------------------------- is it the budget?
    let (document, graph) = scene(true);
    let field = GraphField::compile_graph(&document, &graph, 0.25).expect("compiles");
    println!(
        "The driven case with the budget raised. If the holes are exhaustion, they close here;\n\
         if they are the confirmation refusing a grazing candidate, no budget helps.\n"
    );
    println!("  {:>8}  {:>10}  {:>12}  {:>12}", "budget", "rays hit", "worst steps", "ran out");
    for budget in [512u32, 1024, 2048, 4096, 8192] {
        let row = scanline(&field, eps, budget, &offsets);
        let hits = row.iter().filter(|(_, m)| m.hit).count();
        let worst = row.iter().map(|(_, m)| m.steps).max().unwrap_or(0);
        let out = row.iter().filter(|(_, m)| m.exhausted).count();
        println!("  {budget:>8}  {hits:>4} of {:<3}  {worst:>12}  {out:>12}", row.len());
    }
}
