// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What it costs to rebuild a scene's field, which is what happens on every edit.
//!
//! Reported from the application: copying a revolved pattern and pasting it back in worked but
//! was very slow, and booleaning the two was slower still — while the viewport, once it had
//! settled, was fast. That pair of facts points away from the ray march and at the compile: the
//! shader draws the result quickly, so what is slow is producing it.
//!
//! The suspect is the profile bake. Every extrude and revolve walks its profile's boundary onto
//! a grid, which is the most expensive thing a compile does, and nothing caches it between
//! compiles — so every drag of every number pays for it again, once per copy of the body.
//!
//! Run with: `cargo run --release -p fieldform-field --example recompile_cost`

use fieldform_field::GraphField;
use fieldform_scene::{Fillet, BooleanOp, Curve, CurveId, Document, Graph, Node, NodeId, ParamRef};
use nalgebra::Point3;

const TOLERANCE: f64 = 0.1;

fn main() {
    println!(
        "Rebuilding a scene's field, at {TOLERANCE} mm, with a profile of 500 segments —\n\
         which is what a tessellated curve comes to.\n"
    );
    println!("  {:>34}  {:>12}  {:>10}", "scene", "to compile", "per body");

    for (label, bodies, patterned) in [
        ("one revolve", 1, false),
        ("one revolve, patterned 24", 1, true),
        ("two revolves, patterned", 2, true),
        ("two patterned revolves, booleaned", 2, true),
        ("four patterned revolves, booleaned", 4, true),
    ] {
        let joined = label.contains("boolean");
        let (doc, graph) = scene(bodies, patterned, joined);
        // Three runs, best of, because a compile allocates and the first pays for warming.
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let built = GraphField::compile_graph(&doc, &graph, TOLERANCE);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            assert!(built.is_ok(), "{label} compiles");
            best = best.min(ms);
        }
        println!("  {label:>34}  {best:>9.0} ms  {:>7.0} ms", best / bodies as f64);
    }

    println!(
        "\nIf the cost is flat per body, the compile is repeating whole-body work once per\n\
         copy — and the only expensive thing a compile does per body is bake its profile onto\n\
         a grid by walking the boundary. A pattern is folded, so patterning is nearly free; a\n\
         second copy of the same profile is not, because nothing remembers the first."
    );
}

/// `bodies` revolved profiles, optionally patterned, optionally booleaned into one.
fn scene(bodies: usize, patterned: bool, joined: bool) -> (Document, Graph) {
    let mut doc = Document::new();
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let mut roots: Vec<NodeId> = Vec::new();

    for body in 0..bodies {
        let id = CurveId::new();
        // A rounded rectangle clear of the axis, at the segment count a tessellated curve
        // reaches. Each body gets its own curve, which is what copy-and-paste produces.
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
        let root = if patterned {
            graph.add(Node::Pattern {
                options: Default::default(),
                input: solid.into(),
                count: fieldform_scene::ParamRef::Literal(24.0),
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
        };
        roots.push(root);
    }

    // One union over the whole list rather than a chain of pairs, which is what a boolean is
    // on this branch -- the chain is exactly what the list replaced.
    if joined && roots.len() > 1 {
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: roots.iter().map(|&n| n.into()).collect() });
    }
    (doc, graph)
}
