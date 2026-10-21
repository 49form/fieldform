// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Does flattening an *aimed* turn apply it twice?
//!
//! `Graph::with_turns_expanded_using` rewrites a block that carries a turn into that block inside
//! a `Node::Transform`, keeping the original id on the placement. It clears a **typed** turn from
//! the block it wraps, and deliberately does not clear an **aimed** one — the comment says why:
//! an aimed turn is also a graph *edge*, and dropping it would make the plane it points at a body
//! again. It then says "Nothing downstream reads a turn: the placement is what everything sees."
//!
//! `resolve_frame`'s `Node::Plane` arm reads one. So this asks the same frame twice — once of the
//! graph as written, once of the graph both compilers actually read — and prints the facing each
//! gives. If they differ, the rotation is in the answer twice.
//!
//! Run with: `cargo run -p fieldform-field --example aimed_turn_probe`

use fieldform_scene::{Document, Frame, Graph, Node, ParamRef, Place, Turn};
use fieldform_field::ScalarField;
use nalgebra::Vector3;

fn main() {
    let doc = Document::new();
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();

    // A reference facing: a plane pitched 40 degrees about Y, typed.
    let reference = graph.add(Node::Plane {
        origin: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        flipped: false,
        turn: Turn::By([lit(0.0), lit(40.0), lit(0.0)]),
    });
    // A second plane *aimed* at it, which is what "face the way that one faces" means.
    let aimed = graph.add(Node::Plane {
        origin: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        flipped: false,
        turn: Turn::From { node: reference },
    });
    // Something that reads it as a frame, so the plane is consumed rather than a body.
    let body = graph.add(Node::Sphere {
        centre: Place::At([lit(0.0), lit(50.0), lit(0.0)]),
        radius: lit(10.0),
    });
    graph.add(Node::Mirror {
        input: body.into(),
        frame: Frame::From { node: aimed },
        join: true,
    });

    let resolve = |r: &ParamRef| -> Option<f64> { r.resolve(&|pid| doc.parameter_value(pid)) };
    let frame = Frame::From { node: aimed };

    let say = |label: &str, g: &Graph| {
        match fieldform_field::resolve_frame(&doc, g, &frame, &resolve) {
            Ok((at, rotation)) => {
                let n: Vector3<f64> = rotation * Vector3::z();
                // The pitch away from +Z, which is the whole of the question here.
                let pitch = n.z.clamp(-1.0, 1.0).acos().to_degrees();
                println!(
                    "{label:<34} origin {:>7.3?}  facing ({:+.4}, {:+.4}, {:+.4})  {pitch:.2} deg off +Z",
                    at.as_slice(),
                    n.x,
                    n.y,
                    n.z
                );
            }
            Err(e) => println!("{label:<34} refused: {e}"),
        }
    };

    println!("the reference plane is pitched 40 deg about Y; the second is aimed at it,");
    println!("so \"aimed at\" is right at 40 deg and applied twice at 80.\n");

    say("the graph as written", &graph);
    let lowered = fieldform_field::with_turns_flattened(&doc, &graph, &resolve);
    say("the graph both compilers read", &lowered);

    // And what the plane is *drawn* as, which is the other half of the question: both compilers
    // build a plane leaf from `Node::Plane { origin, flipped, .. }`, discarding the turn and
    // leaving the placement to carry it. So if the frame reads 80 and the surface is at 40, the
    // plane a mirror uses and the plane on screen are not the same plane.
    {
        let mut drawn = Graph::empty();
        let reference = drawn.add(Node::Plane {
            origin: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            flipped: false,
            turn: Turn::By([lit(0.0), lit(40.0), lit(0.0)]),
        });
        // A body, so it has a surface to measure. Nothing consumes it.
        drawn.add(Node::Plane {
            origin: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            flipped: false,
            turn: Turn::From { node: reference },
        });
        let field = fieldform_field::GraphField::compile_graph(&doc, &drawn, 0.5)
            .expect("compiles");
        // Differenced, because what is wanted is the surface the marcher would find.
        let h = 1e-4;
        let at = nalgebra::Point3::new(3.0, 0.0, 7.0);
        let g = Vector3::new(
            field.distance(at + Vector3::x() * h) - field.distance(at - Vector3::x() * h),
            field.distance(at + Vector3::y() * h) - field.distance(at - Vector3::y() * h),
            field.distance(at + Vector3::z() * h) - field.distance(at - Vector3::z() * h),
        )
        .normalize();
        println!(
            "
the same plane as a body, measured off its own field:  {:.2} deg off +Z",
            g.z.abs().clamp(-1.0, 1.0).acos().to_degrees()
        );
    }

    // And the same for a typed turn, which the lowering *does* clear — the control, so a
    // difference above cannot be blamed on the lowering in general.
    let mut typed = Graph::empty();
    let plane = typed.add(Node::Plane {
        origin: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        flipped: false,
        turn: Turn::By([lit(0.0), lit(40.0), lit(0.0)]),
    });
    let ball = typed.add(Node::Sphere {
        centre: Place::At([lit(0.0), lit(50.0), lit(0.0)]),
        radius: lit(10.0),
    });
    typed.add(Node::Mirror {
        input: ball.into(),
        frame: Frame::From { node: plane },
        join: true,
    });
    let frame = Frame::From { node: plane };
    let say_typed = |label: &str, g: &Graph| {
        if let Ok((_, rotation)) = fieldform_field::resolve_frame(&doc, g, &frame, &resolve) {
            let n: Vector3<f64> = rotation * Vector3::z();
            println!(
                "{label:<34} {:.2} deg off +Z",
                n.z.clamp(-1.0, 1.0).acos().to_degrees()
            );
        }
    };
    println!("\nthe control, a typed turn of the same 40 deg, which the lowering does clear:");
    say_typed("the graph as written", &typed);
    let lowered = fieldform_field::with_turns_flattened(&doc, &typed, &resolve);
    say_typed("the graph both compilers read", &lowered);
}
