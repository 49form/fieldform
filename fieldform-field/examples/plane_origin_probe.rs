// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Does a plane sit where a frame aimed at it says it sits?
//!
//! `resolve_frame`'s `Node::Plane` arm reads the plane's origin with `resolve_place_at`. The
//! plane's own field, and everything that draws it, reach it through `resolve_place_for`, which
//! applies the placements above the consumer. If the two part company then a mirror aimed at a
//! plane reflects about a plane that is not the one on screen.
//!
//! The test is direct rather than by inspection: ask the frame where the plane is, then ask the
//! **plane's own field** what the distance is at that point. A plane passes through its own
//! origin, so the answer is zero when the two agree and is the gap in millimetres when they do
//! not.
//!
//! Run with: `cargo run -p fieldform-field --example plane_origin_probe`

use fieldform_field::ScalarField;
use fieldform_scene::{Document, Frame, Graph, Node, ParamRef, Place, Turn};
use nalgebra::Point3;

fn main() {
    println!("{:<52} {:>12} {:>14}", "arrangement", "frame origin", "field there");
    for (label, build) in cases() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let plane = build(&mut graph);
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 0.5)
            .expect("the plane compiles");
        let resolve = |r: &ParamRef| -> Option<f64> { r.resolve(&|pid| doc.parameter_value(pid)) };
        // The graph both compilers read, which is where a frame is resolved during a compile.
        let lowered = fieldform_field::with_turns_flattened(&doc, &graph, &resolve);
        let frame = Frame::From { node: plane };
        match fieldform_field::resolve_frame(&doc, &lowered, &frame, &resolve) {
            Ok((at, _)) => {
                let here = field.distance(Point3::from(at));
                println!("{label:<52} {:>12.3} {here:>13.3} mm", at.z);
            }
            Err(e) => println!("{label:<52} refused: {e}"),
        }
    }
    println!(
        "\nzero in the last column means the plane a frame names is the plane on screen;\n\
         anything else is how far apart they are."
    );
}

type Build = Box<dyn Fn(&mut Graph) -> fieldform_scene::NodeId>;

fn cases() -> Vec<(&'static str, Build)> {
    let lit = ParamRef::Literal;
    vec![
        (
            "a plane on typed numbers",
            Box::new(move |g: &mut Graph| {
                g.add(Node::Plane {
                    origin: Place::At([lit(0.0), lit(0.0), lit(7.0)]),
                    flipped: false,
                    turn: Turn::default(),
                })
            }),
        ),
        (
            "a plane standing on a point",
            Box::new(move |g: &mut Graph| {
                let pt = g.add(Node::Point { at: [lit(0.0), lit(0.0), lit(7.0)] });
                g.add(Node::Plane {
                    origin: Place::From { node: pt },
                    flipped: false,
                    turn: Turn::default(),
                })
            }),
        ),
        (
            "a plane on a point, the plane then moved up 10",
            Box::new(move |g: &mut Graph| {
                let pt = g.add(Node::Point { at: [lit(0.0), lit(0.0), lit(7.0)] });
                let pl = g.add(Node::Plane {
                    origin: Place::From { node: pt },
                    flipped: false,
                    turn: Turn::default(),
                });
                moved(g, pl, 10.0)
            }),
        ),
        (
            "a plane on a point that has itself been moved up 10",
            Box::new(move |g: &mut Graph| {
                let pt = g.add(Node::Point { at: [lit(0.0), lit(0.0), lit(7.0)] });
                let raised = moved(g, pt, 10.0);
                g.add(Node::Plane {
                    origin: Place::From { node: raised },
                    flipped: false,
                    turn: Turn::default(),
                })
            }),
        ),
        (
            "a plane on typed numbers, then moved up 10",
            Box::new(move |g: &mut Graph| {
                let pl = g.add(Node::Plane {
                    origin: Place::At([lit(0.0), lit(0.0), lit(7.0)]),
                    flipped: false,
                    turn: Turn::default(),
                });
                moved(g, pl, 10.0)
            }),
        ),
    ]
}

fn moved(g: &mut Graph, input: fieldform_scene::NodeId, up: f64) -> fieldform_scene::NodeId {
    let lit = ParamRef::Literal;
    g.add(Node::Transform {
        input: input.into(),
        translation: [lit(0.0), lit(0.0), lit(up)],
        rotation: [lit(0.0); 3],
        scale: lit(1.0),
        about: Place::origin(),
    })
}
