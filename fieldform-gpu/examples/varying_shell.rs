// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a varying-thickness shell built from a capsule actually compiles, on both paths.
//!
//! Reported: "I made a capsule then offset it with a varying ramp, declared it a variable, and
//! after that I am no longer able to see the original capsule — I was trying to subtract the
//! original from the offset one to make a shell with a varying wall. I did try this with a cube
//! and a ramp with a plane and it worked; not sure if the capsule and the line/pipe caused it."
//!
//! Two things could be wrong and they have nothing to do with each other: the *composition* may
//! refuse to build, or it may build and be invisible because a body grown outward encloses the
//! body it was grown from. This answers the first, on both compilers, so the second is not
//! investigated while an error is sitting underneath it.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example varying_shell`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    BooleanOp, Curve, Document, Ends, Graph, Node, ParamRef, Place, RampShape,
};
use nalgebra::Point3;

fn main() {
    let lit = ParamRef::Literal;
    let mut doc = Document::new();

    // A line for the pipe to follow, since "the line/pipe" is named in the report as a suspect.
    let curve = fieldform_scene::CurveId::new();
    doc.curves.insert(
        curve,
        Curve::polyline(
            curve,
            "spine",
            // Across the capsule rather than along it, so distance to the line -- and with it
            // the wall the ramp drives -- changes from one end of the capsule to the other.
            // A line along the axis would be the same distance everywhere on it and would
            // measure nothing.
            vec![Point3::new(-40.0, 0.0, -40.0), Point3::new(-40.0, 0.0, 40.0)],
            false,
        )
        .expect("a two-point line"),
    );

    let mut graph = Graph::empty();
    let capsule = graph.add(Node::Capsule {
        ends: Ends::between(Place::at(-20.0, 0.0, 0.0), Place::at(20.0, 0.0, 0.0)),
        start_radius: lit(10.0),
        end_radius: lit(10.0),
    });
    // What the ramp reads: distance to the line, which is what a pipe's field is.
    let pipe = graph.add(Node::Pipe { curve, radius: lit(1.0) });
    let ramp = graph.add(Node::Ramp {
        input: pipe.into(),
        in_min: lit(0.0),
        in_max: lit(30.0),
        out_min: lit(1.0),
        out_max: lit(6.0),
        shape: RampShape::Linear,
    });
    let grown = graph.add(Node::Offset { input: capsule.into(), distance: ParamRef::Field(ramp) });
    let shell = graph.add(Node::Boolean {
        op: BooleanOp::Difference,
        inputs: vec![grown.into(), capsule.into()],
    });

    println!("Bodies: {:?}", graph.roots());
    println!("The shell is {}", if graph.roots() == vec![shell] { "the one body" } else { "NOT the only body" });

    match GraphField::compile_graph(&doc, &graph, 1e-4) {
        Ok(field) => {
            // Along the line, where the wall should thicken from one end to the other.
            println!("\n  {:>8}  {:>10}  {:>10}", "x mm", "value", "wall mm");
            for x in [-40.0f64, -20.0, 0.0, 20.0, 40.0] {
                // Out along +Z from the axis: find where the shell's outer and inner walls are.
                let outer = crossing(&field, Point3::new(x, 0.0, 40.0), -1.0);
                let inner = crossing(&field, Point3::new(x, 0.0, 0.0), 1.0);
                let wall = match (outer, inner) {
                    (Some(o), Some(i)) => format!("{:.3}", o - i),
                    _ => "-".to_string(),
                };
                println!(
                    "  {x:>8.1}  {:>10.3}  {wall:>10}",
                    field.distance(Point3::new(x, 0.0, 0.0)),
                );
            }
        }
        Err(e) => println!("\nThe field REFUSED to compile: {e}"),
    }

    match fieldform_gpu::generate(&doc, &graph) {
        Ok(_) => println!("\nThe shader compiles."),
        Err(e) => println!("\nThe shader REFUSED to compile: {e}"),
    }

    println!(
        "\nAs it stands: the composition is sound on both paths and the wall does vary, so the\n\
         capsule and the pipe are not the difficulty. What was reported is the second thing —\n\
         a body grown outward encloses the body it was grown from, so the original is inside it\n\
         and invisible until something takes a bite out of the outer one. The fix is therefore\n\
         in the panel rather than the kernel: every row carries an eye now, and a block that\n\
         something else consumes can be drawn beside the bodies whether or not it is named."
    );
}

/// Where the field changes sign along +Z from `from`, walking in `step`'s direction.
fn crossing(field: &GraphField, from: Point3<f64>, step: f64) -> Option<f64> {
    let mut z = from.z;
    let mut last = field.distance(Point3::new(from.x, from.y, z));
    for _ in 0..8000 {
        z += step * 0.01;
        let here = field.distance(Point3::new(from.x, from.y, z));
        if here.signum() != last.signum() {
            return Some(z);
        }
        last = here;
    }
    None
}
