// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A hole through the whole graph: the shapes the design note says each type makes, held to
//! references written without the leaf's own arithmetic, and the refusals.

use super::*;
use crate::{Normalization, ScalarField};
use fieldform_scene::{
    BooleanOp, Document, Ends, Graph, HoleDepth, HoleKind, HoleTip, Node, NodeId, ParamRef, Place,
};
use nalgebra::{Point3, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// From `entry` along `dir`, one millimetre: the two points the tool stores.
fn ends(entry: [f64; 3], dir: [f64; 3]) -> Ends {
    Ends::between(
        Place::at(entry[0], entry[1], entry[2]),
        Place::at(entry[0] + dir[0], entry[1] + dir[1], entry[2] + dir[2]),
    )
}

fn compiled(node: Node) -> Result<(GraphField, NodeId, Graph), CompileError> {
    let mut graph = Graph::empty();
    let id = graph.add(node);
    GraphField::compile_graph(&Document::new(), &graph, 0.05).map(|f| (f, id, graph))
}

fn spread(lo: [f64; 3], hi: [f64; 3], n: usize) -> Vec<Point3<f64>> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let t = |a: usize| (a as f64 + 0.37) / n as f64;
                out.push(Point3::new(
                    lo[0] + (hi[0] - lo[0]) * t(i),
                    lo[1] + (hi[1] - lo[1]) * t(j),
                    lo[2] + (hi[2] - lo[2]) * t(k),
                ));
            }
        }
    }
    out
}

/// **A simple blind hole with a flat bottom is a cylinder**, pointwise, and claims a distance --
/// the cylinder from the top of its lead-in, its radius above the entry, to its bottom.
#[test]
fn a_simple_flat_blind_hole_is_the_cylinder_it_cuts() {
    let dir = Vector3::new(0.2, -0.4, -1.0).normalize();
    let entry = Point3::new(3.0, 1.0, 10.0);
    let (hole, _, _) = compiled(Node::Hole {
        ends: ends([entry.x, entry.y, entry.z], [dir.x, dir.y, dir.z]),
        kind: HoleKind::Simple { diameter: lit(8.0) },
        depth: HoleDepth::Blind(lit(15.0)),
        tip: HoleTip::Flat,
    })
    .expect("compiles");
    let end = entry + dir * 15.0;
    let top = entry - dir * 4.0;
    let (cylinder, _, _) = compiled(Node::Cylinder {
        ends: Ends::between(Place::at(top.x, top.y, top.z), Place::at(end.x, end.y, end.z)),
        radius: lit(4.0),
        edge: Default::default(),
    })
    .expect("compiles");
    assert_eq!(hole.normalization(), Normalization::Distance);
    for p in spread([-10.0, -10.0, -10.0], [15.0, 15.0, 15.0], 12) {
        assert!((hole.distance(p) - cylinder.distance(p)).abs() < 1e-9, "at {p:?}");
    }
}

/// Counterbore and countersink step where their numbers say: walking out from the axis at each
/// depth, the cut ends at the radius the section has there.
#[test]
fn a_counterbore_and_a_countersink_step_at_their_sizes() {
    let down = [0.0, 0.0, -1.0];
    let radius_at = |field: &GraphField, depth: f64| -> f64 {
        let mut r = 0.0;
        while field.distance(Point3::new(r, 0.0, -depth)) < 0.0 {
            r += 0.0005;
        }
        r
    };
    let (bored, _, _) = compiled(Node::Hole {
        ends: ends([0.0; 3], down),
        kind: HoleKind::Counterbore { diameter: lit(6.6), bore_diameter: lit(11.0), bore_depth: lit(6.5) },
        depth: HoleDepth::Blind(lit(20.0)),
        tip: HoleTip::Flat,
    })
    .expect("compiles");
    assert!((radius_at(&bored, 3.0) - 5.5).abs() < 1e-3);
    assert!((radius_at(&bored, 10.0) - 3.3).abs() < 1e-3);
    // The shoulder: just above and below the bore's depth, at a radius between the two.
    assert!(bored.distance(Point3::new(4.5, 0.0, -6.4)) < 0.0);
    assert!(bored.distance(Point3::new(4.5, 0.0, -6.6)) > 0.0);

    let (sunk, _, _) = compiled(Node::Hole {
        ends: ends([0.0; 3], down),
        kind: HoleKind::Countersink { diameter: lit(6.6), sink_diameter: lit(13.0), sink_angle: lit(90.0) },
        depth: HoleDepth::Blind(lit(20.0)),
        tip: HoleTip::Flat,
    })
    .expect("compiles");
    // A ninety-degree sink narrows a millimetre of radius per millimetre of depth.
    for depth in [0.5, 1.5, 2.5] {
        assert!((radius_at(&sunk, depth) - (6.5 - depth)).abs() < 1e-3, "at {depth}");
    }
    assert!((radius_at(&sunk, 8.0) - 3.3).abs() < 1e-3);
}

/// **A through hole stays through when its body grows**: lengthen the block under it and the
/// hole still opens out of the far side, with nothing about the hole changed.
#[test]
fn a_through_hole_still_exits_when_the_body_grows() {
    for height in [20.0, 200.0] {
        let mut graph = Graph::empty();
        let block = graph.add(Node::Cuboid {
            centre: Place::at(0.0, 0.0, -height * 0.5),
            size: [lit(40.0), lit(40.0), lit(height)],
            turn: fieldform_scene::no_turn(),
            edge: Default::default(),
        });
        let hole = graph.add(Node::Hole {
            ends: ends([0.0; 3], [0.0, 0.0, -1.0]),
            kind: HoleKind::Simple { diameter: lit(6.0) },
            depth: HoleDepth::Through,
            tip: HoleTip::Drill(lit(118.0)),
        });
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), hole.into()] });
        let field = GraphField::compile_graph(&Document::new(), &graph, 0.05).expect("compiles");
        // Along the axis, from above the top face to below the bottom one: never material, and
        // never nought -- the entry's own face is not drawn across the mouth.
        for i in 0..=100 {
            let z = 1.0 - (height + 2.0) * i as f64 / 100.0;
            assert!(field.distance(Point3::new(0.0, 0.0, z)) > 0.5, "height {height}: {z} reads {}", field.distance(Point3::new(0.0, 0.0, z)));
        }
        // And the block is still there beside it.
        assert!(field.distance(Point3::new(10.0, 0.0, -height * 0.5)) < 0.0);
    }
}

/// **The way into the material is the gradient turned round and normalised**: on a sphere it
/// points at the centre, whether the sphere is a distance or the same sphere through a twist of
/// nothing, which is a bound and would give a short vector if the length were trusted.
#[test]
fn the_way_in_points_at_a_sphere_s_centre_through_a_bound_too() {
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere { centre: Place::at(2.0, -1.0, 3.0), radius: lit(10.0) });
    let exact = GraphField::compile_graph(&Document::new(), &graph, 0.05).expect("compiles");
    graph.add(Node::Twist { input: ball.into(), degrees_per_mm: lit(0.0), frame: Default::default() });
    let bound = GraphField::compile_graph(&Document::new(), &graph, 0.05).expect("compiles");
    assert_eq!(bound.normalization(), Normalization::LowerBound);
    let centre = Point3::new(2.0, -1.0, 3.0);
    for dir in [Vector3::new(1.0, 2.0, 2.0) / 3.0, Vector3::z(), Vector3::new(-0.6, 0.0, 0.8)] {
        let at = centre + dir * 10.0;
        for field in [&exact, &bound] {
            let into = crate::into_material(field, at).expect("a gradient");
            assert!((into.norm() - 1.0).abs() < 1e-9);
            assert!((into + dir).norm() < 1e-4, "{into:?} against {:?}", -dir);
        }
    }
}

/// **A tapped M6** is drilled to the thread's minor diameter and threaded to its depth: across
/// the threaded length the cut reaches out to the major radius and in to the minor, and below it
/// the drill alone, at the minor. The whole is a bound.
#[test]
fn a_tapped_hole_is_its_drill_and_its_thread() {
    let (field, _, _) = compiled(Node::Hole {
        ends: ends([0.0; 3], [0.0, 0.0, -1.0]),
        kind: HoleKind::Tapped { designation: "M6".into(), thread_depth: lit(10.0), clearance: lit(0.0) },
        depth: HoleDepth::Blind(lit(14.0)),
        tip: HoleTip::Drill(lit(118.0)),
    })
    .expect("compiles");
    assert_eq!(field.normalization(), Normalization::LowerBound);
    let h = 0.5 * 3f64.sqrt() * 1.0;
    let (major, minor) = (3.0, 3.0 - 0.625 * h);
    let reach = |phi: f64, depth: f64| -> f64 {
        let mut r = 0.0;
        while field.distance(Point3::new(r * phi.cos(), r * phi.sin(), -depth)) < 0.0 {
            r += 0.0005;
        }
        r
    };
    let (mut out, mut inn) = (0.0f64, f64::INFINITY);
    for a in 0..24 {
        for zi in 0..40 {
            let r = reach(std::f64::consts::TAU * a as f64 / 24.0, 3.0 + zi as f64 * 0.025);
            out = out.max(r);
            inn = inn.min(r);
        }
    }
    assert!((out - major).abs() < 2e-3, "outermost {out}, major {major}");
    assert!((inn - minor).abs() < 2e-3, "innermost {inn}, minor {minor}");
    // Below the thread, the drill.
    for a in 0..8 {
        let r = reach(std::f64::consts::TAU * a as f64 / 8.0, 12.0);
        assert!((r - minor).abs() < 1e-3, "{r}");
    }
}

/// Every refusal by name.
#[test]
fn a_hole_refuses_what_it_cannot_be_by_name() {
    let down = ends([0.0; 3], [0.0, 0.0, -1.0]);
    let refused = |kind: HoleKind, depth: HoleDepth, tip: HoleTip| {
        compiled(Node::Hole { ends: down, kind, depth, tip }).err().expect("refused")
    };
    let blind = |d: f64| HoleDepth::Blind(lit(d));
    assert!(matches!(
        compiled(Node::Hole { ends: ends([0.0; 3], [0.0; 3]), kind: HoleKind::default(), depth: blind(5.0), tip: HoleTip::Flat }),
        Err(CompileError::HoleNeedsADirection)
    ));
    assert!(matches!(refused(HoleKind::Simple { diameter: lit(0.0) }, blind(5.0), HoleTip::Flat), CompileError::HoleSizeNotPositive("diameter", _)));
    assert!(matches!(refused(HoleKind::default(), blind(-1.0), HoleTip::Flat), CompileError::HoleSizeNotPositive("depth", _)));
    assert!(matches!(
        refused(HoleKind::Counterbore { diameter: lit(6.0), bore_diameter: lit(5.0), bore_depth: lit(2.0) }, blind(10.0), HoleTip::Flat),
        CompileError::HoleStepNotWider("counterbore", _, _)
    ));
    assert!(matches!(
        refused(HoleKind::Counterbore { diameter: lit(6.0), bore_diameter: lit(10.0), bore_depth: lit(12.0) }, blind(10.0), HoleTip::Flat),
        CompileError::HoleStepTooDeep("counterbore", _, _)
    ));
    assert!(matches!(
        refused(HoleKind::Countersink { diameter: lit(6.0), sink_diameter: lit(12.0), sink_angle: lit(180.0) }, blind(10.0), HoleTip::Flat),
        CompileError::HoleAngle("countersink", _)
    ));
    assert!(matches!(refused(HoleKind::default(), blind(10.0), HoleTip::Drill(lit(0.0))), CompileError::HoleAngle("drill point", _)));
    assert!(matches!(
        refused(HoleKind::Tapped { designation: "M7".into(), thread_depth: lit(5.0), clearance: lit(0.0) }, blind(10.0), HoleTip::Flat),
        CompileError::UnknownThreadDesignation(_)
    ));
    assert!(matches!(
        refused(HoleKind::Tapped { designation: "M6".into(), thread_depth: lit(12.0), clearance: lit(0.0) }, blind(10.0), HoleTip::Flat),
        CompileError::ThreadDeeperThanHole(_, _)
    ));
    // A through hole has no bottom for a step or a thread to run past.
    assert!(compiled(Node::Hole {
        ends: down,
        kind: HoleKind::Counterbore { diameter: lit(6.0), bore_diameter: lit(10.0), bore_depth: lit(50.0) },
        depth: HoleDepth::Through,
        tip: HoleTip::Flat,
    })
    .is_ok());
}

/// The extent of a blind hole holds every point it cuts; a through hole has none to hold.
#[test]
fn a_blind_hole_s_extent_holds_what_it_cuts() {
    let (field, id, graph) = compiled(Node::Hole {
        ends: ends([4.0, -3.0, 2.0], [0.3, 0.5, -0.8]),
        kind: HoleKind::Countersink { diameter: lit(6.0), sink_diameter: lit(12.0), sink_angle: lit(90.0) },
        depth: HoleDepth::Blind(lit(12.0)),
        tip: HoleTip::Drill(lit(118.0)),
    })
    .expect("compiles");
    let (lo, hi) = crate::graph_extent(&Document::new(), &graph, id).as_box().expect("bounded");
    let mut cut = 0;
    for p in spread([-15.0, -15.0, -20.0], [20.0, 20.0, 10.0], 40) {
        if field.distance(p) < 0.0 {
            cut += 1;
            for a in 0..3 {
                assert!(p[a] >= lo[a] - 1e-9 && p[a] <= hi[a] + 1e-9, "{p:?} outside {lo:?}..{hi:?}");
            }
        }
    }
    assert!(cut > 20, "{cut}");
    let (_, id, graph) = compiled(Node::Hole {
        ends: ends([0.0; 3], [0.0, 0.0, -1.0]),
        kind: HoleKind::default(),
        depth: HoleDepth::Through,
        tip: HoleTip::Flat,
    })
    .expect("compiles");
    assert!(crate::graph_extent(&Document::new(), &graph, id).as_box().is_none());
}
