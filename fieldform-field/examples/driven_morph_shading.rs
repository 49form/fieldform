// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Why a body goes dark the moment its morph's amount starts varying through space.
//!
//! Reported from a model on screen: adding a morph and a ramp turned the shading from light to
//! dark. Nothing about the lighting changed, so the suspicion is the field — and it is the same
//! root cause as the missing silhouette (`grazing_rays.rs`), reaching a second consumer.
//!
//! **Ambient occlusion and the soft shadow both read the field's value as a distance.** That is
//! what they are: `occlusion` walks a few steps along the normal and accumulates how much less
//! clearance it found than it travelled, and `shadow` keeps the smallest ratio of clearance to
//! distance travelled. Both assume the number they read is roughly how far away the surface is.
//!
//! A driven morph divides its value to keep it a lower bound. The value stays *honest* — it never
//! overestimates, so the tracer is safe — but it is no longer a good estimate of how far away
//! anything is. Every sample reads short, both functions conclude that material is close on all
//! sides, and the body shades as though it were at the bottom of a hole.
//!
//! The two functions below are `march.wgsl`'s, transcribed. A difference between them and the
//! shader would make the whole measurement meaningless, so they are copied rather than described.
//!
//! Run with: `cargo run --release -p fieldform-field --example driven_morph_shading`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Document, Graph, Node, ParamRef, Place, RampShape};
use nalgebra::{Point3, Vector3};

/// `march.wgsl`'s `occlusion`, line for line. 1 is nothing nearby, 0 is fully enclosed.
fn occlusion(field: &dyn ScalarField, p: Point3<f64>, n: Vector3<f64>, radius: f64, scale: f64) -> f64 {
    let mut occluded = 0.0;
    let mut falloff = 1.0;
    for i in 1..=5 {
        let h = radius * f64::from(i) / 5.0;
        let d = field.distance(p + n * h) * scale;
        occluded += (h - d) * falloff;
        falloff *= 0.7;
    }
    (1.0 - 1.5 * occluded / radius).clamp(0.0, 1.0)
}

/// `march.wgsl`'s `shadow`, line for line. 1 is fully lit, 0 is fully shadowed.
fn shadow(
    field: &dyn ScalarField,
    p: Point3<f64>,
    n: Vector3<f64>,
    light: Vector3<f64>,
    reach: f64,
    softness: f64,
    scale: f64,
) -> f64 {
    let start = p + n * (reach * 0.004);
    let mut lit: f64 = 1.0;
    let mut t = reach * 0.004;
    for _ in 0..48 {
        let d = field.distance(start + light * t) * scale;
        lit = lit.min(softness * d / t);
        if lit < 0.005 || t > reach {
            break;
        }
        t += d.clamp(reach * 0.002, reach * 0.08);
    }
    lit.clamp(0.0, 1.0)
}

/// The reported scene: a sphere morphed to a cube, the amount optionally driven.
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
            let point = graph.add(Node::Point { at: [lit(0.0), lit(-44.33), lit(0.0)] });
            let ramp = graph.add(Node::Ramp {
                input: point.into(),
                in_min: lit(17.0),
                in_max: lit(86.6),
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

/// Walks in along the +Z axis to find the surface, so the samples sit where a pixel would.
fn surface_on_z(field: &dyn ScalarField) -> Point3<f64> {
    let (mut lo, mut hi) = (0.0f64, 200.0f64);
    for _ in 0..80 {
        let mid = (lo + hi) * 0.5;
        match field.distance(Point3::new(0.0, 0.0, mid)) < 0.0 {
            true => lo = mid,
            false => hi = mid,
        }
    }
    Point3::new(0.0, 0.0, (lo + hi) * 0.5)
}

fn main() {
    // The viewport's own numbers for a 100 mm part: occlusion reaches a tenth of the model,
    // the shadow across it.
    let radius = 86.6;
    let ao_reach = radius * 0.1;
    let shadow_reach = radius * 2.0;
    let softness = 8.0;
    let light = Vector3::new(0.4, 0.5, 0.77).normalize();

    println!(
        "A 50 mm sphere morphed to a 100 mm cube, sampled on the +Z pole where the surface is\n\
         flat to the camera and should be close to fully open. Occlusion 1 is nothing nearby,\n\
         0 is buried; shadow 1 is fully lit.\n"
    );
    println!(
        "  {:>34}  {:>10}  {:>9}  {:>7}",
        "scene", "multiplier", "occlusion", "shadow"
    );

    for (label, driven) in [("constant amount", false), ("amount varies through space", true)] {
        let (document, graph) = scene(driven);
        let field = GraphField::compile_graph(&document, &graph, 0.25).expect("compiles");
        let multiplier = fieldform_field::step_multiplier(&document, &graph);
        let p = surface_on_z(&field);
        let n = Vector3::z();

        // As the shader reads it today: the value straight from the field.
        let open = occlusion(&field, p, n, ao_reach, 1.0);
        let lit = shadow(&field, p, n, light, shadow_reach, softness, 1.0);
        println!("  {label:>34}  {multiplier:>10.1}  {open:>9.3}  {lit:>7.3}");
    }

    // ---------------------------------------------------------------- the correction
    println!(
        "\nThe same driven scene with the samples multiplied back by the divisor, which is what\n\
         turns a value that was divided to stay honest into an estimate of how far away the\n\
         surface is again:\n"
    );
    println!("  {:>10}  {:>9}  {:>7}", "scale", "occlusion", "shadow");
    let (document, graph) = scene(true);
    let field = GraphField::compile_graph(&document, &graph, 0.25).expect("compiles");
    let multiplier = fieldform_field::step_multiplier(&document, &graph);
    let p = surface_on_z(&field);
    let n = Vector3::z();
    for scale in [1.0, multiplier * 0.5, multiplier] {
        let open = occlusion(&field, p, n, ao_reach, scale);
        let lit = shadow(&field, p, n, light, shadow_reach, softness, scale);
        println!("  {scale:>10.2}  {open:>9.3}  {lit:>7.3}");
    }

    // And the constant scene for comparison, which is what it should look like.
    let (document, graph) = scene(false);
    let field = GraphField::compile_graph(&document, &graph, 0.25).expect("compiles");
    let p = surface_on_z(&field);
    let open = occlusion(&field, p, Vector3::z(), ao_reach, 1.0);
    let lit = shadow(&field, p, Vector3::z(), light, shadow_reach, softness, 1.0);
    println!("\n  the constant scene, for comparison: occlusion {open:.3}, shadow {lit:.3}");
}
