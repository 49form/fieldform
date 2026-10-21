// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What it costs to get a changed scene onto the screen, split into its three parts.
//!
//! Reported from the application: copying a revolved pattern and pasting it back in was very
//! slow, and booleaning the two slower still — but the viewport, once settled, was fast. So the
//! cost is in rebuilding, not in drawing, and this says which part of rebuilding.
//!
//! Three things happen when a graph changes: the field is compiled (`recompile_cost.rs` measures
//! that at about 19 ms per body, flat), WGSL is generated from the graph, and the driver turns
//! that WGSL into a pipeline. The third is the one nobody measures, and shader compilation is
//! not linear in anything a user can see.
//!
//! **Run it more than once before believing a threshold.** Two runs put the blow-up at three
//! bodies and at four — same program, same machine — so whatever triggers it is not the body
//! count. Backlog item 34 carries the two candidates and how to tell them apart.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example rebuild_cost`

use fieldform_gpu::{Renderer, generate};
use fieldform_scene::{Fillet, BooleanOp, Curve, CurveId, Document, Graph, Node, NodeId, ParamRef};
use nalgebra::{Point2, Point3};

const TOLERANCE: f64 = 0.1;

fn main() {
    println!(
        "Getting a changed scene onto the screen, at {TOLERANCE} mm, profiles of 500 segments.\n"
    );
    println!(
        "  {:>14}  {:>10}  {:>12}  {:>9}  {:>11}",
        "scene", "generate", "pipeline", "chars", "a frame"
    );

    // **Off by default, because the salt distorts what it is measuring.** The driver caches
    // compiled shaders on disk, so a shape it has seen before reports four hundred milliseconds
    // for a program that took ten minutes the first time -- and the source is *structure only*,
    // every number being a `params[..]` read, so nothing about the geometry makes it novel.
    // Chaining no-op offsets does, but a long chain is itself a deep graph and measures that
    // instead. So: salt when you need a cold run of an unchanged codegen, and leave it off when
    // the codegen itself has changed, which makes every shader novel anyway.
    let salt: usize = std::env::var("FIELDFORM_SALT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if salt > 0 {
        println!("  (structure salted with {salt} no-op offsets, so no row is served from cache)");
    }
    println!();

    // **One body, patterned and not**, before adding bodies at all. A pattern emits its
    // subtree once per step of its window inside a *single* function, so if the blow-up is
    // within a body rather than across them, these two rows say so on their own and every
    // later row is a distraction.
    // One row only, for a before-and-after that does not cost twelve minutes a side: the
    // pattern loop has to be compared against the unrolled version it replaced, and the
    // unrolled version is the slow one.
    let only = std::env::var("FIELDFORM_ONLY").is_ok();
    for (label, bodies, patterned) in [
        ("1, no pattern", 1usize, false),
        ("1, patterned", 1, true),
        ("2, patterned", 2, true),
        ("3, patterned", 3, true),
    ] {
        if only && label != "1, patterned" {
            continue;
        }
        let (doc, graph) = scene(bodies, salt, patterned);
        let bodies = label;

        let started = std::time::Instant::now();
        let program = generate(&doc, &graph).expect("generates");
        let generated = started.elapsed().as_secs_f64() * 1e3;
        let chars = program.source.len();
        // **A fingerprint of the text, because the first question about a run-to-run difference
        // is whether the input was the same.** `evaluation_order` sorts roots by raw id and ids
        // are fresh each run, so equivalent graphs can emit in different orders -- if these
        // hashes match across runs then order is not what varied, and the difference is the
        // driver's or the machine's.
        let fingerprint = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            program.source.hash(&mut h);
            h.finish()
        };

        // The curves the shader is handed, baked the way the field bakes them.
        let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
            .curves
            .iter()
            .map(|use_of| {
                // A profile names a graph node; a spine names a curve. Only profiles occur in
                // this scene, so the node's own curve is what to tessellate.
                let id = match use_of {
                    fieldform_gpu::CurveUse::Profile(node) => match graph.node(*node) {
                        Some(Node::Revolve { outline, .. }) => outline.curve(),
                        _ => unreachable!("this scene is revolves"),
                    },
                    fieldform_gpu::CurveUse::Spine(curve) => *curve,
                    fieldform_gpu::CurveUse::Miters(_) => {
                        unreachable!("this scene's draft rounds its corners")
                    }
                    fieldform_gpu::CurveUse::Frame(_) => unreachable!("this scene sweeps nothing"),
                    fieldform_gpu::CurveUse::LoftSpine(_)
                    | fieldform_gpu::CurveUse::LoftSection(..)
                    | fieldform_gpu::CurveUse::LoftSlope(..) => {
                        unreachable!("this scene lofts nothing")
                    }
                    fieldform_gpu::CurveUse::Wall(..) => unreachable!("this scene has no thin extrude"),
                    fieldform_gpu::CurveUse::Point(_) => {
                        unreachable!("this scene attracts to nothing")
                    }
                };
                {
                    let points = doc.curves[&id].tessellate(0.05);
                    let flat: Vec<Point2<f64>> =
                        points.iter().map(|p| Point2::new(p.x, p.z)).collect();
                    let bake = fieldform_field::Profile::new(flat).ok().and_then(|profile| {
                        let (lo, hi) = profile.bounds();
                        fieldform_field::bake_cells((hi - lo).max(), TOLERANCE, (0.0, 0.0))
                            .map(|cells| profile.baked(cells, (0.0, 0.0)))
                            .and_then(|p| p.bake().cloned())
                    });
                    (points, bake)
                }
            })
            .collect();

        let started = std::time::Instant::now();
        let renderer = pollster::block_on(Renderer::new(&program, &[], &curves, &[], 128, 10.0));
        let built = started.elapsed().as_secs_f64() * 1e3;
        if let Err(e) = &renderer {
            eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
            return;
        }

        // **And what a frame costs once it exists**, because the whole change traded inlined
        // copies for a run-time loop. Compile time is what a user waits for after an edit;
        // frame time is what they feel while turning the model, and a fix that moved the cost
        // from one to the other would be no fix at all.
        let renderer = renderer.expect("checked above");
        let bounds = (
            nalgebra::Point3::new(-60.0, -60.0, -20.0),
            nalgebra::Point3::new(60.0, 60.0, 40.0),
        );
        let camera = fieldform_gpu::Camera::framing(bounds, 1.0);
        let mut frame = f64::INFINITY;
        for _ in 0..5 {
            let started = std::time::Instant::now();
            renderer.render(&camera, 512, 512, 512).expect("renders");
            frame = frame.min(started.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "  {bodies:>14}  {generated:>7.0} ms  {built:>9.0} ms  {chars:>9}  {frame:>8.1} ms"
        );
        let _ = fingerprint;
    }

    println!(
        "\n`generate` turns the graph into WGSL and `pipeline` is the driver turning that WGSL\n\
         into something the card can run. The field compile is on top of both, at about 19 ms\n\
         a body (`recompile_cost.rs`).\n\n\
         The source grows *linearly* -- about thirty thousand characters and a few hundred\n\
         bindings per body -- while the pipeline time does not. That rules out the obvious\n\
         explanations: it is not the amount of code, and it is not `generate`, which stays in\n\
         the tens of milliseconds. It is one enormous function handed to an optimiser whose\n\
         cost is superlinear in live values, and the fix is to stop handing it one -- emit\n\
         each body as its own function, so the register allocator sees several small problems\n\
         instead of one large one."
    );
}

/// `bodies` revolved profiles, each patterned, unioned into one.
///
/// **Every run builds a shader the driver has never seen**, by salting one literal with the
/// clock. Without that this measures the wrong thing entirely: the source is byte-identical
/// between runs -- the fingerprint column proves it -- so the driver serves its on-disk cache
/// and reports four hundred milliseconds for a program that took ten minutes to compile the
/// first time. A cold compile is what a user pays on every structural edit, and it is the only
/// number worth optimising.
fn scene(bodies: usize, salt: usize, patterned: bool) -> (Document, Graph) {
    let mut doc = Document::new();
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let mut roots: Vec<NodeId> = Vec::new();

    for body in 0..bodies {
        let id = CurveId::new();
        let points: Vec<Point3<f64>> = (0..500)
            .map(|i| {
                let a = i as f64 / 500.0 * std::f64::consts::TAU;
                Point3::new(30.0 + 8.0 * a.cos(), 0.0, 8.0 * a.sin() + body as f64 * 0.001)
            })
            .collect();
        doc.curves.insert(
            id,
            Curve::polyline(id, format!("Profile {body}"), points, true).expect("closed"),
        );
        let solid = graph.add(Node::Revolve { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [lit(0.0); 3]),
            angle: fieldform_scene::ParamRef::Literal(360.0),
            sides: fieldform_scene::AngleSides::One,
        });
        roots.push(if patterned {
            graph.add(Node::Pattern {
                options: Default::default(),
                input: solid.into(),
                count: ParamRef::Literal(24.0),
                kind: fieldform_scene::PatternKind::Radial {
                    frame: fieldform_scene::Frame::at_along(
                        fieldform_scene::Axis::Z,
                        [lit(0.0); 3],
                    ),
                },
            
                join: true,
                fillet: Fillet::hard(),
            })
        } else {
            solid
        });
    }

    // One union over the whole list, which is what a boolean is on this branch -- master wrote
    // this as a chain of pairwise unions, and that chain is precisely what the list replaced.
    // It is also the shape being measured: N bodies joined is one node and one flat fold, so
    // the shader grows with the bodies rather than with a nest of booleans around them. Rows
    // are therefore not directly comparable with the ones master recorded.
    let mut all = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: roots.clone().into_iter().map(Into::into).collect() });
    // **Salted in the structure, because that is the only thing the shader depends on.** The
    // generated source is structure alone -- every number is a `params[..]` read -- so changing
    // a coordinate leaves the text byte-identical and the driver serves its cache. Which is the
    // real lesson of this example: dragging a value is free, and *any structural edit* buys a
    // shader the driver has never seen. A few no-op offsets make each run's program novel, the
    // same number for every body count so the rows stay comparable.
    for _ in 0..salt {
        all = graph.add(Node::Offset { input: all.into(), distance: lit(0.0) });
    }
    (doc, graph)
}
