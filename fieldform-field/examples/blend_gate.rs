// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which graphs the blend gate refuses, and which it must not.
//!
//! `cargo run -p fieldform-field --example blend_gate`

use fieldform_scene::{Fillet, BooleanOp, Continuity, Document, Graph, LatticePhase, Node, ParamRef};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn sphere(g: &mut Graph) -> fieldform_scene::NodeId {
    g.add(Node::Sphere {
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        radius: lit(50.0),
    })
}
fn cuboid(g: &mut Graph) -> fieldform_scene::NodeId {
    g.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        size: [lit(100.0); 3],
        turn: fieldform_scene::no_turn(),
    })
}
fn gyroid(g: &mut Graph) -> fieldform_scene::NodeId {
    g.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: fieldform_scene::LatticeCell::Gyroid,
        map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::sheet(lit(2.0)),
    })
}

fn check(name: &str, graph: &Graph) {
    let doc = Document::new();
    match fieldform_field::GraphField::compile_graph(&doc, graph, 1e-3) {
        Ok(_) => println!("  {name:<44} compiles"),
        Err(e) => println!("  {name:<44} REFUSED: {e}"),
    }
}

fn main() {
    println!("what the gate does to each graph:\n");

    // Sphere blended with a cube. Both are true distance fields; this must be allowed.
    let mut g = Graph::empty();
    let (s, c) = (sphere(&mut g), cuboid(&mut g));
    g.add(Node::Blend {
        op: BooleanOp::Union,
        a: s.into(),
        b: c.into(),
        fillet: Fillet { radius: lit(6.0), continuity: Continuity::G1 },
    });
    check("blend(sphere, cube)", &g);

    // The same, with a lattice sitting in the document as a separate body.
    let mut g = Graph::empty();
    let (s, c) = (sphere(&mut g), cuboid(&mut g));
    let _ = gyroid(&mut g);
    g.add(Node::Blend {
        op: BooleanOp::Union,
        a: s.into(),
        b: c.into(),
        fillet: Fillet { radius: lit(6.0), continuity: Continuity::G1 },
    });
    check("blend(sphere, cube) + a loose gyroid body", &g);

    // A lattice under one operand. This one must be refused.
    let mut g = Graph::empty();
    let (s, c, y) = (sphere(&mut g), cuboid(&mut g), gyroid(&mut g));
    let isect = g.add(Node::Boolean {
        op: BooleanOp::Intersection,
        inputs: vec![s.into(), y.into()],
    });
    g.add(Node::Blend {
        op: BooleanOp::Union,
        a: isect.into(),
        b: c.into(),
        fillet: Fillet { radius: lit(6.0), continuity: Continuity::G1 },
    });
    check("blend(intersect(sphere, gyroid), cube)", &g);

    // A plain boolean against a lattice, which is always allowed.
    let mut g = Graph::empty();
    let (s, y) = (sphere(&mut g), gyroid(&mut g));
    g.add(Node::Boolean {
        op: BooleanOp::Intersection,
        inputs: vec![s.into(), y.into()],
    });
    check("boolean(sphere, gyroid)", &g);
}
