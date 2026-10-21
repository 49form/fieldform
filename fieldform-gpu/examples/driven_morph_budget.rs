// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the step budget is worth on a driven morph, in pixels recovered and milliseconds spent.
//!
//! `fieldform-field/examples/grazing_rays.rs` establishes the cause on the CPU: a driven morph
//! divides its value to stay a lower bound, so the tracer steps that much shorter, and a fixed
//! budget covers that much less of the ray. A ray that runs out looks exactly like a ray that hit
//! nothing, so the surface goes missing — first along the silhouette, where a ray runs closest to
//! the surface for longest.
//!
//! This is the other half: the same scene through the real shader, counting the pixels the budget
//! decides and timing what they cost. The budget the viewport now picks is
//! `512 × step_multiplier`, which is 4352 here, so that row is the one to read.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example driven_morph_budget`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{Document, Graph, Node, ParamRef, Place, RampShape};
use nalgebra::Point3;

/// A control: primitives and a boolean, which report true distance and divide by nothing. Here
/// to say whether a larger floor costs anything on ordinary geometry, since the answer decides
/// whether the floor may be raised for everyone or only for the fields that need it.
fn plain() -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let centre = Place::At([lit(0.0), lit(0.0), lit(0.0)]);
    let mut graph = Graph::empty();
    let block = graph.add(Node::Cuboid { edge: Default::default(),
        centre: centre.clone(),
        size: [lit(100.0), lit(100.0), lit(100.0)],
        turn: fieldform_scene::no_turn(),
    });
    let ball = graph.add(Node::Sphere { centre, radius: lit(62.0) });
    graph.add(Node::Boolean {
        op: fieldform_scene::BooleanOp::Difference,
        inputs: vec![block.into(), ball.into()],
    });
    (Document::new(), graph)
}

/// The scene from `morph-variable2.fieldform`, as reported.
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

fn main() {
    for (label, build) in [
        ("a cube with a sphere cut out of it, true distance", plain as fn() -> (Document, Graph)),
        ("constant amount", || scene(false)),
        ("amount varies through space", || scene(true)),
    ] {
        let (doc, graph) = build();
        let multiplier = fieldform_field::step_multiplier(&doc, &graph);
        // `traced/rebuild.rs`: STEP_BUDGET * multiplier * STEP_HEADROOM, marked in the table below.
        let chosen = (512.0 * multiplier * 2.0).ceil() as u32;
        println!(
            "\n=== {label} — step multiplier ×{multiplier:.1}, so the viewport picks {chosen} ==="
        );

        let program = generate(&doc, &graph).expect("generates");
        let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 64, 1.0)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no GPU available ({e})");
                return;
            }
        };
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| match p {
                ParamRef::Literal(v) => *v as f32,
                _ => 0.0,
            })
            .collect();
        renderer.set_params(&values);

        // Framed as the viewport frames a 100 mm part, and at the tier a still frame uses.
        let centre = Point3::origin();
        let eye = Point3::new(160.0, 120.0, 90.0);
        let edge = 720u32;
        let camera =
            Camera::looking_at(eye, centre, nalgebra::Vector3::z(), (centre, 86.6), 45.0, 1.0);

        let mut rows = Vec::new();
        let mut budgets = vec![512u32, 1024, 2048, 4352, 8192, 16384, chosen];
        budgets.sort_unstable();
        budgets.dedup();
        for budget in budgets {
            let _ = renderer.render(&camera, edge, edge, budget);
            let start = std::time::Instant::now();
            let mut hits = 0;
            for _ in 0..5 {
                let frame = renderer.render(&camera, edge, edge, budget).expect("renders");
                hits = frame.iter().filter(|h| h.is_hit()).count();
            }
            rows.push((budget, hits, start.elapsed().as_secs_f64() * 1000.0 / 5.0));
        }
        // The largest budget is the truth: every ray that can converge has by then.
        let settled = rows.last().map(|r| r.1).unwrap_or(0);
        println!("  {:>8}  {:>10}  {:>10}  {:>9}", "budget", "pixels hit", "missing", "frame");
        for (budget, hits, ms) in rows {
            println!(
                "  {budget:>8}  {hits:>10}  {:>10}  {ms:>6.1} ms{}",
                settled.saturating_sub(hits),
                match budget == chosen {
                    true => "   <- the viewport's own",
                    false => "",
                }
            );
        }
    }
}
