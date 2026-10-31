// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Tests of the frame a point on a body is measured in: it follows the body and ignores its size.
use super::*;
use fieldform_scene::{BooleanOp, Document, Graph, Node, NodeId, ParamRef, Place, Turn};
use nalgebra::{Point3, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn numbers(r: &ParamRef) -> Option<f64> {
    r.resolve(&|_| None)
}

fn cuboid(g: &mut Graph, at: [f64; 3], size: [f64; 3], turn: [f64; 3]) -> NodeId {
    g.add(Node::Cuboid {
        centre: Place::At(at.map(lit)),
        size: size.map(lit),
        turn: Turn::By(turn.map(lit)),
        edge: Default::default(),
    })
}

/// **Growing the body elsewhere leaves the frame alone.** A frame that was a fraction of the
/// bounding box moved every point on a bracket when a boss was unioned onto its far end.
#[test]
fn a_frame_does_not_change_when_the_body_grows_elsewhere() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let slab = cuboid(&mut g, [0.0; 3], [40.0, 20.0, 10.0], [0.0; 3]);
    let before = body_frame(&doc, &g, slab, &numbers).expect("a cuboid has a frame");
    let boss = cuboid(&mut g, [60.0, 0.0, 0.0], [10.0; 3], [0.0; 3]);
    g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![slab.into(), boss.into()] });
    let after = body_frame(&doc, &g, slab, &numbers).expect("still has one");
    let p = Point3::new(5.0, 5.0, 5.0);
    assert!((before * p - after * p).norm() < 1e-12);
    assert!((after * p - p).norm() < 1e-12, "a cuboid at the origin is measured in world");
}

/// **A cuboid's own turn turns the frame**, so a point stays at the same place on the face.
#[test]
fn a_turned_cuboid_carries_its_frame_with_it() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let slab = cuboid(&mut g, [10.0, 0.0, 0.0], [40.0, 20.0, 10.0], [0.0, 0.0, 30.0]);
    let frame = body_frame(&doc, &g, slab, &numbers).expect("has a frame");
    // Turned about its own centre: the frame's origin is the centre and its X is turned 30 deg.
    let origin = frame * Point3::origin();
    assert!((origin - Point3::new(10.0, 0.0, 0.0)).norm() < 1e-9, "{origin:?}");
    let x = frame * Vector3::x();
    let c = 30f64.to_radians();
    assert!((x - Vector3::new(c.cos(), c.sin(), 0.0)).norm() < 1e-9, "{x:?}");
}

/// **A placement above or below carries the frame**, scale included, so a body scaled by two
/// carries its point to the same place on it.
#[test]
fn a_placement_carries_the_frame_and_its_scale() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let ball = g.add(Node::Sphere { centre: Place::At([lit(1.0), lit(2.0), lit(3.0)]), radius: lit(5.0) });
    let moved = g.add(Node::Transform {
        input: ball.into(),
        translation: [lit(100.0), lit(0.0), lit(0.0)],
        rotation: [lit(0.0); 3],
        scale: lit(2.0),
        about: Place::At([lit(0.0); 3]),
    });
    let frame = body_frame(&doc, &g, moved, &numbers).expect("has a frame");
    let p = frame * Point3::origin();
    assert!((p - Point3::new(102.0, 4.0, 6.0)).norm() < 1e-9, "{p:?}");
    assert!((frame.scaling() - 2.0).abs() < 1e-12);
    // Asking from the sphere inside gives the same frame as asking of the placement itself.
    let inner = body_frame(&doc, &g, ball, &numbers).expect("has a frame");
    assert!((inner * Point3::origin() - p).norm() < 1e-9);
}

/// **A union has no frame of its own**: it adds nothing to the placements above it.
#[test]
fn a_union_is_measured_in_the_frame_above_it() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [5.0, 0.0, 0.0], [4.0; 3], [0.0; 3]);
    let b = cuboid(&mut g, [-5.0, 0.0, 0.0], [4.0; 3], [0.0; 3]);
    let both = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
    let frame = body_frame(&doc, &g, both, &numbers).expect("has a frame");
    assert!((frame * Point3::new(1.0, 2.0, 3.0) - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
}
