// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a document evaluates to must survive being written to a file and read back.
//!
//! The round-trip tests in `fieldform-io` compare the *document* before and after; this
//! compares the **field**, which is the thing the user is actually looking at. A serialization
//! bug that lands a curve's knots in the wrong order, or drops a boolean's operand, produces a
//! document that still compares equal in every way anyone thought to check and a shape that is
//! quietly different.
//!
//! The scene is deliberately the awkward one rather than the convenient one — a cage, a curve
//! swept into a pipe, a subtraction and a blend driven by a named parameter — because a
//! single-node graph exercises almost none of the format.

use fieldform_cage::Cage;
use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Fillet, BooleanOp, Continuity, Curve, CurveId, Document, Node, ParamRef, Part};
use nalgebra::Point3;

fn composed_document() -> Document {
    let mut doc = Document::with_cage(Cage::cube_of(60.0));
    let cage_id = *doc.cages.keys().next().expect("with_cage inserts one");
    let blend_cage = doc.add_cage(Cage::cube_of(25.0));
    let blend_radius = doc.add_parameter("fillet", 6.0);

    let curve = Curve::polyline(
        CurveId::new(),
        "spine",
        vec![
            Point3::new(-50.0, 0.0, 5.0),
            Point3::new(0.0, 10.0, -5.0),
            Point3::new(50.0, -5.0, 5.0),
        ],
        false,
    )
    .expect("three points make a polyline");
    let curve_id = curve.id;
    doc.curves.insert(curve_id, curve);

    let mut part = Part::from_cage("bracket", cage_id);
    let cage_node = part.graph.roots()[0];
    let pipe = part.graph.add(Node::Pipe {
        curve: curve_id,
        radius: ParamRef::Literal(8.0),
    });
    let drilled = part.graph.add(Node::Boolean {
        op: BooleanOp::Difference,
        inputs: vec![cage_node.into(), pipe.into()],
    });
    // Shrunk a little, so the document carries an offset too. Negative deliberately: a sign
    // dropped in the file reads back as a body grown where it should have been reduced, and
    // the surface is somewhere plausible either way.
    let thinned = part.graph.add(Node::Offset {
        input: drilled.into(),
        distance: ParamRef::Literal(-1.5),
    });
    let boss = part.graph.add(Node::Cage(blend_cage));
    part.graph.add(Node::Blend {
        op: BooleanOp::Union,
        a: thinned.into(),
        b: boss.into(),
        fillet: Fillet { radius: ParamRef::Named(blend_radius), continuity: Continuity::G1, second: None },
    });
    part.material = fieldform_scene::Material::preset("Brushed aluminium");

    doc.parts.clear();
    doc.add_part(part);
    doc
}

/// The field a document's only part evaluates to.
fn field_of(doc: &Document) -> GraphField {
    let part = doc.parts.values().next().expect("one part");
    GraphField::compile(doc, part, 0.5).expect("the composed graph compiles")
}

/// Points spread across the model, including inside the cage, inside the pipe it subtracts,
/// out in the blend, and well outside everything.
fn probes() -> Vec<Point3<f64>> {
    let mut points = Vec::new();
    for i in -4..=4 {
        for j in -4..=4 {
            for k in -4..=4 {
                points.push(Point3::new(
                    i as f64 * 11.0,
                    j as f64 * 11.0,
                    k as f64 * 11.0,
                ));
            }
        }
    }
    points
}

#[test]
fn a_document_evaluates_identically_after_a_round_trip_through_a_file() {
    let doc = composed_document();
    let archive = fieldform_io::to_archive(&doc).expect("the document serializes");
    let mut bytes = Vec::new();
    archive
        .write_to(std::io::Cursor::new(&mut bytes))
        .expect("the archive writes");
    let reopened = fieldform_io::from_archive(
        &fieldform_io::Archive::read_from(std::io::Cursor::new(&bytes)).expect("it reads back"),
    )
    .expect("and is a document");

    let before = field_of(&doc);
    let after = field_of(&reopened);
    for p in probes() {
        let (a, b) = (before.distance(p), after.distance(p));
        assert_eq!(
            a, b,
            "the field differs at {p:?} after a round trip: {a} then {b}"
        );
    }
}

#[test]
fn the_named_parameter_still_drives_the_blend_after_a_round_trip() {
    // A parameter that came back as a literal, or a reference that lost its id, would leave
    // the geometry looking right and the document no longer parametric.
    let mut doc = composed_document();
    let archive = fieldform_io::to_archive(&doc).expect("the document serializes");
    let mut reopened = fieldform_io::from_archive(&archive).expect("and reads back");

    let param = *doc.parameters.keys().next().expect("one parameter");
    let widen = |doc: &mut Document| {
        doc.parameters
            .get_mut(&param)
            .expect("the parameter kept its id")
            .value = fieldform_scene::ParamValue::Literal(14.0);
    };
    widen(&mut doc);
    widen(&mut reopened);

    let (before, after) = (field_of(&doc), field_of(&reopened));
    for p in probes() {
        assert_eq!(before.distance(p), after.distance(p), "at {p:?}");
    }
}

/// **A pattern's options survive a file, and the angle that is a variable stays one.** The options
/// are written only where they are not what an older document meant, so the case is the one that
/// writes all of them: a ring that stops short of a turn by a named angle, centred, with copies left
/// out, and a row spaced by its whole run. A key that was dropped on the way out would still draw a
/// ring — a full one — so the field is compared, and the angle is then edited on both.
#[test]
fn a_patterns_options_and_its_named_angle_survive_a_round_trip() {
    let lit = ParamRef::Literal;
    let mut doc = Document::new();
    let angle = doc.add_parameter("sweep", 200.0);
    let mut part = Part {
        id: fieldform_scene::PartId::new(),
        name: "fan".to_string(),
        material: None,
        graph: fieldform_scene::Graph::empty(),
        inputs: Vec::new(),
    };
    let ball = part.graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
        radius: lit(3.0),
    });
    part.graph.add(Node::Pattern {
        input: ball.into(),
        count: lit(25.0),
        kind: fieldform_scene::PatternKind::Radial { frame: fieldform_scene::Frame::default() },
        join: true,
        fillet: Fillet::hard(),
        options: fieldform_scene::PatternOptions {
            symmetric: true,
            angle: ParamRef::Named(angle),
            skip: vec![3, 20],
            ..Default::default()
        },
    });
    let rod = part.graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([lit(-60.0), lit(-40.0), lit(0.0)]),
        radius: lit(3.0),
    });
    part.graph.add(Node::Pattern {
        input: rod.into(),
        count: lit(15.0),
        kind: fieldform_scene::PatternKind::Linear { step: [lit(70.0), lit(0.0), lit(0.0)] },
        join: true,
        fillet: Fillet::hard(),
        options: fieldform_scene::PatternOptions {
            spacing: fieldform_scene::Spacing::Extent,
            skip: vec![6],
            ..Default::default()
        },
    });
    doc.parts.clear();
    doc.add_part(part);

    let archive = fieldform_io::to_archive(&doc).expect("the document serializes");
    let mut reopened = fieldform_io::from_archive(&archive).expect("and reads back");
    let probes = || {
        (-40..=40).flat_map(|i| {
            (-30..=30).map(move |j| Point3::new(f64::from(i) * 2.0, f64::from(j) * 2.0, 0.5))
        })
    };
    let (before, after) = (field_of(&doc), field_of(&reopened));
    for p in probes() {
        assert_eq!(before.distance(p), after.distance(p), "at {p:?}");
    }

    let original = field_of(&doc);
    for d in [&mut doc, &mut reopened] {
        d.parameters.get_mut(&angle).expect("the parameter kept its id").value =
            fieldform_scene::ParamValue::Literal(120.0);
    }
    let (before, after) = (field_of(&doc), field_of(&reopened));
    let mut moved = false;
    for p in probes() {
        assert_eq!(before.distance(p), after.distance(p), "after the edit, at {p:?}");
        moved |= (before.distance(p) - original.distance(p)).abs() > 1e-6;
    }
    assert!(moved, "changing the angle changed nothing, so it was not driving the arc");
}

/// **A skewed box's edges are a nest of three triples, and one of the nine is a variable.** The
/// nesting is the part of its spelling no other node has, so the case is a box whose second edge's
/// middle component names a parameter: a reference that came back as a literal, or an edge that
/// came back in another slot, would still draw *a* leaning box.
#[test]
fn a_skewed_box_keeps_its_edges_and_its_variable_through_a_round_trip() {
    let mut doc = Document::new();
    let lean = doc.add_parameter("lean", 18.0);
    let lit = ParamRef::Literal;
    let mut part = Part {
        id: fieldform_scene::PartId::new(),
        name: "leaning".to_string(),
        material: None,
        graph: fieldform_scene::Graph::empty(),
        inputs: Vec::new(),
    };
    part.graph.add(Node::SkewedBox {
        centre: fieldform_scene::Place::At([lit(4.0), lit(-6.0), lit(9.0)]),
        edges: [
            [lit(40.0), lit(0.0), lit(0.0)],
            [lit(12.0), ParamRef::Named(lean), lit(0.0)],
            [lit(7.0), lit(5.0), lit(30.0)],
        ],
    });
    doc.parts.clear();
    doc.add_part(part);

    let archive = fieldform_io::to_archive(&doc).expect("the document serializes");
    let mut reopened = fieldform_io::from_archive(&archive).expect("and reads back");
    let (before, after) = (field_of(&doc), field_of(&reopened));
    for p in probes() {
        assert_eq!(before.distance(p), after.distance(p), "at {p:?}");
    }

    // Driven, not frozen: change the parameter on both and they must still agree — and differ
    // from what they were, or the case drove nothing.
    let original = field_of(&doc);
    for d in [&mut doc, &mut reopened] {
        d.parameters.get_mut(&lean).expect("the parameter kept its id").value =
            fieldform_scene::ParamValue::Literal(31.0);
    }
    let (before, after) = (field_of(&doc), field_of(&reopened));
    let mut moved = false;
    for p in probes() {
        assert_eq!(before.distance(p), after.distance(p), "after the edit, at {p:?}");
        moved |= (before.distance(p) - original.distance(p)).abs() > 1e-6;
    }
    assert!(moved, "changing the parameter changed nothing, so the case drove nothing");
}
