// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a pattern of a ramp-driven pipe costs, and which part of it is expensive.
//!
//! Reported: "when I pattern a pipe generated on a 3-degree curve using a ramp — pattern being
//! the default setting — camera tumbling in the viewport crawls hard."
//!
//! Three things changed when a pipe's radius learned to vary, and each could be the one that
//! costs: the **shader** grew a ramp and a plane inside the pipe's expression, and a pattern
//! emits that expression once per instance; the **step budget** rose, because a divided field
//! reads short and `step_multiplier` buys back the steps; and the **spine** is walked per
//! evaluation, which a finely tessellated degree-three curve makes long.
//!
//! They want different fixes, so this separates them before anything is changed.
//!
//! `cargo run --release -p fieldform-gpu --example patterned_pipe_cost`

use fieldform_field::GraphField;
use fieldform_scene::{Fillet, 
    BooleanOp, Curve, Document, Graph, Node, ParamRef, Place, RampShape, Turn,
};
use nalgebra::Point3;

/// How the pattern is built: the block the menu makes, or the placements it used to make.
#[derive(Clone, Copy, PartialEq)]
enum How {
    /// `Node::Pattern`, which `Modify > Pattern` adds. The shader emits the body **once** as a
    /// function and calls it in a loop bounded by the pattern's own window.
    Block,
    /// One `Transform` and one `Boolean` per copy, which is what a pattern was before the node
    /// kind existed -- and what the shader has to inline once per copy.
    Placements,
}

/// A degree-three curve of the kind somebody draws, and a pipe on it.
fn scene(driven: bool, count: u32, how: How) -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let mut doc = Document::new();
    let curve = fieldform_scene::CurveId::new();
    doc.curves.insert(
        curve,
        {
            // A single-span cubic, built the way the application builds one: a clamped knot
            // vector and unit weights.
            let points = vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(40.0, 30.0, 10.0),
                Point3::new(80.0, -20.0, 20.0),
                Point3::new(120.0, 10.0, 0.0),
            ];
            let knots: Vec<f64> = std::iter::repeat_n(0.0, 4)
                .chain(std::iter::repeat_n(1.0, 4))
                .collect();
            Curve::new(curve, "spine", 3, points, vec![1.0; 4], knots, false)
                .expect("a four-point cubic")
        },
    );

    let mut graph = Graph::empty();
    let radius = if driven {
        // What `App::drive_pipe` builds: a plane square across the curve at its start, and a
        // ramp from a quarter of the radius to the whole of it along the curve.
        let plane = graph.add(Node::Plane {
            origin: Place::StartOf { curve },
            flipped: false,
            turn: Turn::Tangent { curve },
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: lit(0.0),
            in_max: lit(120.0),
            out_min: lit(1.25),
            out_max: lit(5.0),
            shape: RampShape::Smooth,
        });
        ParamRef::Field(ramp)
    } else {
        lit(5.0)
    };
    let pipe = graph.add(Node::Pipe { curve, radius });

    if how == How::Block && count > 1 {
        // What the menu actually builds.
        graph.add(Node::Pattern {
            options: Default::default(),
            input: pipe.into(),
            count: ParamRef::Literal(f64::from(count)),
            kind: fieldform_scene::PatternKind::Linear {
                step: [lit(20.0), lit(0.0), lit(0.0)],
            },
            join: true,
            fillet: Fillet::hard(),
        });
        return (doc, graph);
    }
    let mut combined = pipe;
    for i in 1..count {
        let placed = graph.add(Node::Transform {
            input: pipe.into(),
            translation: [lit(0.0), lit(0.0), lit(20.0 * f64::from(i))],
            rotation: [lit(0.0); 3],
            scale: lit(1.0),
            about: Place::origin(),
        });
        combined = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![combined.into(), placed.into()],
        });
    }
    (doc, graph)
}

fn main() {
    println!(
        "A cubic spine, a pipe on it, and a pattern of that pipe. `fixed` is a plain radius and\n\
         `driven` is one varying along the curve, which is the difference being measured.\n"
    );
    // How long the spine is, in segments, at the tolerance the viewport asks for. Every
    // evaluation of a pipe walks all of them, and every copy in a pattern walks them again.
    {
        let (doc, _) = scene(false, 1, How::Block);
        let curve = doc.curves.values().next().expect("the spine");
        for tolerance in [1.0f64, 0.25, 0.05] {
            println!(
                "  spine at {tolerance:>5} mm tolerance: {:>5} segments",
                curve.tessellate(tolerance).len().saturating_sub(1),
            );
        }
        println!(
            "  spine at 1 degree of turn:    {:>5} segments
",
            curve.tessellate_to_angle(1.0, 0.05).len().saturating_sub(1),
        );
    }

    println!(
        "  {:>10}  {:>7}  {:>8}  {:>10}  {:>9}  {:>9}",
        "built as", "copies", "radius", "shader ch", "steps x", "ns/eval"
    );
    for how in [How::Block, How::Placements] {
        for count in [1u32, 4, 12, 40] {
        for driven in [false, true] {
            let (doc, graph) = scene(driven, count, how);

            let started = std::time::Instant::now();
            let program = fieldform_gpu::generate(&doc, &graph).expect("the shader compiles");
            let generate = started.elapsed();

            let started = std::time::Instant::now();
            let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("the field compiles");
            let compile = started.elapsed();
            let _ = &field;

            let steps = fieldform_field::step_multiplier(&doc, &graph);

            // What one field reading costs. The shader does the same work per step per pixel,
            // so this is the number a tumble is paying, multiplied by steps and by pixels.
            use fieldform_field::ScalarField;
            let probe = |i: u32| {
                let t = f64::from(i) * 0.37;
                Point3::new(60.0 + 40.0 * t.sin(), 20.0 * t.cos(), 10.0 * (t * 0.7).sin())
            };
            let rounds = 20_000u32;
            let started = std::time::Instant::now();
            let mut sink = 0.0f64;
            for i in 0..rounds {
                sink += field.distance(probe(i));
            }
            let per = started.elapsed().as_nanos() as f64 / f64::from(rounds);
            assert!(sink.is_finite());

            let _ = (generate, compile);
            println!(
                "  {:>10}  {count:>7}  {:>8}  {:>10}  {steps:>9.2}  {per:>9.0}",
                if how == How::Block { "pattern" } else { "placements" },
                if driven { "driven" } else { "fixed" },
                program.source.len(),
            );
        }
        }
    }

    println!(
        "
What this found, on this machine.

         The varying radius is not the cost: it adds two or three percent per evaluation and five
         percent to the step budget, and neither is a crawl. Nor is the spine, which is 37
         segments at the finest tolerance the viewport asks for.

         The cost is a pattern of a *pipe*, and it is inherent rather than wasteful.
         `Node::Pattern` is built well: the shader stays the same size whether there are four
         copies or forty, because the body is emitted once as a function and called in a loop
         bounded by the pattern's own window. But that window has to visit every instance which
         could reach the point being asked about, and a 120 mm pipe stepped 20 mm apart overlaps
         about nine of its neighbours everywhere. So one field reading costs about nine pipe
         readings. The plateau is the window doing its job: forty copies cost what twelve did.

         The old way is the one to avoid, and the menu already avoids it: placements inline the
         body once per copy, so forty of them is a 170 000-character shader and forty-six
         microseconds a reading, against a constant 95 000 and nine.

         Where a win is left: the inner loop is `pipe_distance`, which walks every segment of the
         spine on every evaluation -- 37 of them, times nine instances, per step, per pixel. That
         is what to attack if this has to be faster, and it is the same shape as backlog item 17's
         per-region pruning rather than anything about patterns."
    );
}
