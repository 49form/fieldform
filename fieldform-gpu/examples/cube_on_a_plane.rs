// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The scene that killed the window: a plane, a point, a cube aimed at the plane, hidden.
//!
//! Read off the console log, in the order it happened: a plane was added, a point was made, a
//! cube was typed, "Make a plane facing this way" made a second plane and aimed the cube at it,
//! and then that plane was hidden. The window went with the next frame and left no panic log,
//! which rules out a panic and leaves a stack overflow or a hang — neither of which the hook
//! catches, since both abort without unwinding.
//!
//! **What this rules out.** The scene itself is sound: the roots are what they should be, the
//! flattened graph has an evaluation order, and both compilers accept it. So the fault is not in
//! the composition but in the walks the *panel* runs over it every frame — and those had two
//! unbounded mutual recursions in them, both introduced when a facing gained the ability to name
//! a block: `resolve_frame` reads a turn and a turn reads a frame, and `placement_above`
//! composes a placement whose pivot walks the placements above its own source. Each had a
//! counter for steps within one walk, which is a different quantity and does not stop the stack
//! growing. `REFERENCE_DEPTH` is the bound; the test beside it is that a facing which leads back
//! to itself returns at all.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example cube_on_a_plane`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Document, Graph, Node, ParamRef, Place, Turn};
use nalgebra::Point3;

fn main() {
    let lit = ParamRef::Literal;
    let doc = Document::new();
    let mut graph = Graph::empty();

    // "Added a plane", then "Point is a point at 0,0,0" -- the plane's own origin V.
    let point = graph.add(Node::Point { at: [lit(0.0); 3] });
    let first = graph.add(Node::Plane {
        origin: Place::From { node: point },
        flipped: false,
        turn: Turn::By([lit(0.0); 3]),
    });
    // "> cube"
    let cube = graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(0.0); 3]),
        size: [lit(50.0); 3],
        turn: Turn::By([lit(0.0); 3]),
    });
    // "Plane is a plane -- turn it and everything aimed at it follows": a second plane, made
    // from the cube's own facing, with the cube then aimed at it.
    let second = graph.add(Node::Plane {
        origin: Place::At([lit(0.0); 3]),
        flipped: false,
        turn: Turn::By([lit(0.0), lit(30.0), lit(0.0)]),
    });
    if let Some(Node::Cuboid { turn, .. }) = graph.node_mut(cube) {
        *turn = Turn::From { node: second };
    }

    println!("roots: {} of them", graph.roots().len());
    println!("  the cube is a body: {}", graph.roots().contains(&cube));
    println!("  the first plane is a body: {}", graph.roots().contains(&first));
    println!("  the aimed-at plane is a body: {}", graph.roots().contains(&second));

    let numbers = |r: &ParamRef| -> Option<f64> { r.resolve(&|_| None) };
    let flat = fieldform_field::with_turns_flattened(&doc, &graph, &numbers);
    println!("\nflattened: {} nodes, {} roots", flat.len(), flat.roots().len());
    println!(
        "  evaluation order: {}",
        match flat.evaluation_order() {
            Some(order) => format!("{} nodes", order.len()),
            None => "REFUSED as cyclic".to_string(),
        }
    );

    match GraphField::compile_graph(&doc, &graph, 1e-4) {
        Ok(field) => println!(
            "\nthe field compiles; at the origin it reads {:.3}",
            field.distance(Point3::origin())
        ),
        Err(e) => println!("\nthe field REFUSED: {e}"),
    }
    match fieldform_gpu::generate(&doc, &graph) {
        Ok(_) => println!("the shader compiles"),
        Err(e) => println!("the shader REFUSED: {e}"),
    }

}
