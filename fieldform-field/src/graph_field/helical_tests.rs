// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A coil and a thread through the whole graph: the blocks compile to the leaves they are,
//! aimed where their frames say, refused by name where they cannot be built, bounded where they
//! are, and mirrored by the other hand.

use super::*;
use crate::{Normalization, ScalarField};
use fieldform_scene::{
    CoilSection, CoilSize, Document, Frame, Graph, Hand, Node, NodeId, ParamRef,
};
use nalgebra::{Point3, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn coil_node(frame: Frame, hand: Hand) -> Node {
    Node::Coil {
        frame,
        diameter: lit(20.0),
        size: CoilSize::TurnsPitch { turns: lit(2.5), pitch: lit(6.0) },
        section: CoilSection::Square { wire: lit(2.0) },
        hand,
    }
}

fn thread_node(designation: Option<&str>, clearance: f64, hand: Hand) -> Node {
    Node::Thread {
        frame: Frame::default(),
        major: lit(8.0),
        pitch: lit(1.25),
        length: lit(20.0),
        hand,
        clearance: lit(clearance),
        designation: designation.map(str::to_string),
    }
}

fn compiled(node: Node) -> Result<(GraphField, NodeId, Graph), CompileError> {
    let mut graph = Graph::empty();
    let id = graph.add(node);
    GraphField::compile_graph(&Document::new(), &graph, 0.05).map(|f| (f, id, graph))
}

/// Points spread through a box, for every comparison below.
fn spread(lo: Point3<f64>, hi: Point3<f64>, n: usize) -> Vec<Point3<f64>> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let t = |a: usize| (a as f64 + 0.37) / n as f64;
                out.push(Point3::new(
                    lo.x + (hi.x - lo.x) * t(i),
                    lo.y + (hi.y - lo.y) * t(j),
                    lo.z + (hi.z - lo.z) * t(k),
                ));
            }
        }
    }
    out
}

/// A coil aimed by a turned and moved frame is the unaimed coil read at the point carried into
/// that frame, and both claim a bound.
#[test]
fn an_aimed_coil_is_the_coil_read_in_its_frame() {
    let axis = Vector3::new(0.3, -0.5, 1.0);
    let origin = Vector3::new(4.0, -2.0, 7.0);
    let frame = Frame::At { origin: [lit(origin.x), lit(origin.y), lit(origin.z)], axis: [lit(axis.x), lit(axis.y), lit(axis.z)] };
    let (aimed, _, _) = compiled(coil_node(frame, Hand::Right)).expect("compiles");
    let (plain, _, _) = compiled(coil_node(Frame::default(), Hand::Right)).expect("compiles");
    assert_eq!(aimed.normalization(), Normalization::LowerBound);
    assert_eq!(plain.normalization(), Normalization::LowerBound);
    let rotation = crate::rotation_from_z(axis);
    for p in spread(Point3::new(-15.0, -15.0, -10.0), Point3::new(20.0, 15.0, 30.0), 8) {
        let local = Point3::from(rotation.inverse_transform_vector(&(p.coords - origin)));
        assert!((aimed.distance(p) - plain.distance(local)).abs() < 1e-9, "at {p:?}");
    }
}

/// A named thread is the table's: M8 reads exactly as a typed 8 by 1.25, whatever the typed
/// fields beside the name hold.
#[test]
fn a_named_thread_reads_its_major_and_pitch_from_the_table() {
    let (named, _, _) = compiled(Node::Thread {
        frame: Frame::default(),
        major: lit(99.0),
        pitch: lit(9.0),
        length: lit(20.0),
        hand: Hand::Right,
        clearance: lit(0.0),
        designation: Some("M8".into()),
    })
    .expect("compiles");
    let (typed, _, _) = compiled(thread_node(None, 0.0, Hand::Right)).expect("compiles");
    for p in spread(Point3::new(-6.0, -6.0, -2.0), Point3::new(6.0, 6.0, 22.0), 9) {
        assert_eq!(named.distance(p), typed.distance(p), "at {p:?}");
    }
}

/// The radial extremes of the thread are its major and minor radii, and a clearance moves both
/// by itself.
#[test]
fn a_thread_reaches_its_major_and_minor_radii_and_a_clearance_moves_both() {
    for clearance in [0.0, 0.1, -0.1] {
        let (field, _, _) = compiled(thread_node(None, clearance, Hand::Right)).expect("compiles");
        let h = 0.5 * 3f64.sqrt() * 1.25;
        let (major, minor) = (4.0 + clearance, 4.0 - 0.625 * h + clearance);
        // Over a turn of angles and a pitch of heights, the outermost material and the innermost
        // gap, found along rays out from the axis.
        let (mut outer, mut inner) = (0.0f64, f64::INFINITY);
        for a in 0..24 {
            let phi = std::f64::consts::TAU * a as f64 / 24.0;
            for zi in 0..50 {
                let z = 8.0 + 1.25 * zi as f64 / 50.0;
                let mut r = 2.0;
                while field.distance(Point3::new(r * phi.cos(), r * phi.sin(), z)) < 0.0 {
                    r += 0.0005;
                }
                outer = outer.max(r);
                inner = inner.min(r);
            }
        }
        assert!((outer - major).abs() < 2e-3, "clearance {clearance}: outermost {outer}, major {major}");
        assert!((inner - minor).abs() < 2e-3, "clearance {clearance}: innermost {inner}, minor {minor}");
    }
}

/// A left-handed coil and thread are the right-handed ones mirrored through a plane holding the
/// axis, pointwise -- the mirror block's reflection, not a second rule.
#[test]
fn the_left_hand_is_the_right_hand_mirrored() {
    for (right, left) in [
        (coil_node(Frame::default(), Hand::Right), coil_node(Frame::default(), Hand::Left)),
        (thread_node(None, 0.0, Hand::Right), thread_node(None, 0.0, Hand::Left)),
    ] {
        let mut graph = Graph::empty();
        let body = graph.add(right);
        graph.add(Node::Mirror {
            input: body.into(),
            frame: Frame::At { origin: [lit(0.0); 3], axis: [lit(0.0), lit(1.0), lit(0.0)] },
            join: false,
        });
        let mirrored = GraphField::compile_graph(&Document::new(), &graph, 0.05).expect("compiles");
        let (left, _, _) = compiled(left).expect("compiles");
        for p in spread(Point3::new(-12.0, -12.0, -3.0), Point3::new(12.0, 12.0, 21.0), 9) {
            assert!((mirrored.distance(p) - left.distance(p)).abs() < 1e-9, "at {p:?}");
        }
    }
}

/// Every refusal the design note asks for, by name: a pitch at nought, a section across the
/// axis, turns that would fuse, a designation the table does not have, a length of nought, and
/// a number a ramp drives.
#[test]
fn a_coil_and_a_thread_refuse_what_they_cannot_be_by_name() {
    let refused = |node: Node| compiled(node).err().expect("refused");
    let coil = |diameter: f64, pitch: f64, wire: f64, turns: f64| Node::Coil {
        frame: Frame::default(),
        diameter: lit(diameter),
        size: CoilSize::TurnsPitch { turns: lit(turns), pitch: lit(pitch) },
        section: CoilSection::Circle { wire: lit(wire) },
        hand: Hand::Right,
    };
    assert!(matches!(refused(coil(20.0, 0.0, 2.0, 1.0)), CompileError::HelixNeedsAPositivePitch("coil", _)));
    assert!(matches!(refused(coil(2.0, 6.0, 2.0, 1.0)), CompileError::HelixSectionCrossesTheAxis("coil", _)));
    assert!(matches!(refused(coil(20.0, 1.5, 2.0, 1.0)), CompileError::CoilTurnsWouldFuse(_, _)));
    assert!(matches!(refused(coil(20.0, 6.0, 2.0, 0.0)), CompileError::CoilNeedsTurns(_)));
    assert!(matches!(refused(coil(20.0, 6.0, 0.0, 1.0)), CompileError::CoilNeedsAWire(_)));
    // The size said as height and pitch derives the turns, and a height of nought has none.
    assert!(matches!(
        refused(Node::Coil {
            frame: Frame::default(),
            diameter: lit(20.0),
            size: CoilSize::HeightPitch { height: lit(0.0), pitch: lit(6.0) },
            section: CoilSection::default(),
            hand: Hand::Right,
        }),
        CompileError::CoilNeedsTurns(_)
    ));
    match refused(thread_node(Some("M7"), 0.0, Hand::Right)) {
        CompileError::UnknownThreadDesignation(name) => assert_eq!(name, "M7"),
        other => panic!("{other:?}"),
    }
    let thread = |major: f64, pitch: f64, length: f64| Node::Thread {
        frame: Frame::default(),
        major: lit(major),
        pitch: lit(pitch),
        length: lit(length),
        hand: Hand::Right,
        clearance: lit(0.0),
        designation: None,
    };
    assert!(matches!(refused(thread(8.0, 0.0, 20.0)), CompileError::HelixNeedsAPositivePitch("thread", _)));
    assert!(matches!(refused(thread(8.0, 1.25, 0.0)), CompileError::ThreadNeedsALength(_)));
    // A pitch so coarse for the diameter that the root would cross the axis.
    assert!(matches!(refused(thread(2.0, 4.0, 20.0)), CompileError::HelixSectionCrossesTheAxis("thread", _)));

    // And a ramp driving a number, as on every block but the few that bound it.
    let mut graph = Graph::empty();
    let plane = graph.add(Node::Plane {
        origin: fieldform_scene::Place::At([lit(0.0); 3]),
        flipped: false,
        turn: fieldform_scene::no_turn(),
    });
    let ramp = graph.add(Node::Ramp {
        input: plane.into(),
        in_min: lit(0.0),
        in_max: lit(40.0),
        out_min: lit(18.0),
        out_max: lit(24.0),
        shape: fieldform_scene::RampShape::Linear,
    });
    graph.add(Node::Coil {
        frame: Frame::default(),
        diameter: ParamRef::Field(ramp),
        size: CoilSize::default(),
        section: CoilSection::default(),
        hand: Hand::Right,
    });
    match GraphField::compile_graph(&Document::new(), &graph, 0.05) {
        Err(CompileError::ParameterCannotBeDriven(what)) => assert!(what.contains("coil"), "{what}"),
        other => panic!("a driven diameter compiled: {:?}", other.err()),
    }
}

/// Every solid point of an aimed coil and of a thread lies inside the extent the graph reports.
#[test]
fn the_extent_holds_every_solid_point_of_a_coil_and_a_thread() {
    let frame = Frame::At { origin: [lit(5.0), lit(0.0), lit(-3.0)], axis: [lit(1.0), lit(0.2), lit(0.4)] };
    for node in [coil_node(frame, Hand::Left), thread_node(Some("M8"), 0.1, Hand::Right)] {
        let (field, id, graph) = compiled(node).expect("compiles");
        let extent = crate::graph_extent(&Document::new(), &graph, id);
        let (lo, hi) = extent.as_box().expect("bounded");
        let mut solid = 0;
        for p in spread(Point3::new(-30.0, -30.0, -30.0), Point3::new(30.0, 30.0, 30.0), 40) {
            if field.distance(p) < 0.0 {
                solid += 1;
                for axis in 0..3 {
                    assert!(p[axis] >= lo[axis] - 1e-9 && p[axis] <= hi[axis] + 1e-9, "{p:?} outside {lo:?}..{hi:?}");
                }
            }
        }
        assert!(solid > 20, "the grid has to meet the solid for this to test anything: {solid}");
    }
}

/// A coil a thousandth of a turn long, at a pitch far larger than its wire, is a short straight
/// piece of wire: solid where the section is, and read about as far off it as the section is.
#[test]
fn a_coil_of_a_thousandth_of_a_turn_is_a_short_piece_of_its_section() {
    let (field, _, _) = compiled(Node::Coil {
        frame: Frame::default(),
        diameter: lit(20.0),
        size: CoilSize::TurnsPitch { turns: lit(0.001), pitch: lit(100.0) },
        section: CoilSection::Square { wire: lit(2.0) },
        hand: Hand::Right,
    })
    .expect("compiles");
    // In the middle of the piece, about 0.03 mm in from either end face.
    let mid = 0.0005 * std::f64::consts::TAU;
    let at = |r: f64, z: f64| field.distance(Point3::new(r * mid.cos(), r * mid.sin(), z + 0.05));
    assert!(at(10.0, 0.0) < 0.0, "the middle of the piece is solid");
    // Two millimetres out from the square's outer face, and above its top: never more than two,
    // and no less than two over the map's stretch -- which at a pitch of a hundred on an inner
    // radius of nine is 3.8, the cost the design note expected of a steep, tight helix.
    let stretch = crate::helix_stretch(100.0, 9.0);
    assert!(stretch > 3.0, "{stretch}");
    for (r, z) in [(13.0, 0.0), (10.0, 3.0)] {
        let d = at(r, z);
        assert!(d >= 2.0 / stretch - 1e-6 && d <= 2.0 + 1e-9, "at r {r} z {z}: {d}");
    }
}
