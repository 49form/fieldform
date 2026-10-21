// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether the shader generator accepts the blends the kernel accepts.
//!
//! `cargo run -p fieldform-gpu --example blend_shader`

use fieldform_scene::{BooleanOp, Continuity, Document, Graph, LatticePhase, Node, ParamRef};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn check(name: &str, graph: &Graph) {
    let doc = Document::new();
    match fieldform_gpu::generate(&doc, graph) {
        Ok(_) => println!("  {name:<44} generates"),
        Err(e) => println!("  {name:<44} REFUSED: {e}"),
    }
}

fn main() {
    println!("what the shader generator does to each graph:\n");

    let mut g = Graph::empty();
    let s = g.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(50.0) });
    let c = g.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(100.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    g.add(Node::Blend { op: BooleanOp::Union, a: s.into(), b: c.into(), fillet: fieldform_scene::Fillet { radius: lit(6.0), continuity: Continuity::G1 } });
    check("blend(sphere, cube)", &g);

    let mut g = Graph::empty();
    let s = g.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(50.0) });
    let c = g.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(100.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    let _y = g.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: fieldform_scene::LatticeCell::Gyroid,
        map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::sheet(lit(2.0)),
    });
    g.add(Node::Blend { op: BooleanOp::Union, a: s.into(), b: c.into(), fillet: fieldform_scene::Fillet { radius: lit(6.0), continuity: Continuity::G1 } });
    check("blend(sphere, cube) + a loose gyroid body", &g);

    let mut g = Graph::empty();
    let s = g.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(50.0) });
    let y = g.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: fieldform_scene::LatticeCell::Gyroid,
        map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::sheet(lit(2.0)),
    });
    g.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![s.into(), y.into()] });
    check("boolean(sphere, gyroid)", &g);
}
