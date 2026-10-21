// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! An expression parameter reaches the field the way a typed one does, and fails the way an
//! unresolvable name does — never as a NaN.
//!
//! `fieldform-scene` proves the arithmetic; this proves the number lands in a radius. The
//! sphere is the shape to measure it with: its distance at the origin is exactly `-r`, so a
//! radius that came through wrong would be read straight off the field.

use fieldform_field::{graph_extent, CompileError, GraphField, ScalarField};
use fieldform_scene::{
    CellMap, Document, Graph, LatticeCell, LatticePhase, Node, ParamId, ParamRef, Part, PartId,
    Place,
};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// A part that is one sphere at the origin, with the radius given.
fn ball(name: &str, radius: ParamRef) -> Part {
    let mut graph = Graph::empty();
    graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius });
    Part { id: PartId::new(), name: name.into(), material: None, graph, inputs: Vec::new() }
}

/// The radius a part's field says it has, read at the origin.
fn radius_of(doc: &Document, part: PartId) -> Result<f64, CompileError> {
    let field = GraphField::compile(doc, &doc.parts[&part], 0.5)?;
    Ok(-field.distance(Point3::origin()))
}

/// The worked case: `radius = 5`, `boss_r = radius * 5 - 2`, a sphere of `boss_r` measures
/// 23 mm, and 28 mm after `radius` becomes 6 with nothing else touched.
#[test]
fn a_sphere_of_boss_r_measures_23_then_28_when_radius_becomes_6() {
    let mut doc = Document::new();
    let radius = doc.add_parameter("radius", 5.0);
    let boss_r = doc.add_parameter("boss_r", 0.0);
    doc.set_parameter_expression(boss_r, "radius * 5 - 2").unwrap();
    let part = doc.add_part(ball("boss", ParamRef::Named(boss_r)));

    assert_eq!(radius_of(&doc, part).unwrap(), 23.0);
    doc.set_parameter_literal(radius, 6.0);
    assert_eq!(radius_of(&doc, part).unwrap(), 28.0);
}

/// The test that keeps NaN out of the field: a division by zero in a parameter refuses to
/// compile with the same error an unknown name gives, rather than producing a shape.
#[test]
fn a_failed_expression_refuses_to_compile_exactly_as_an_unknown_name_does() {
    let mut doc = Document::new();
    let broken = doc.add_parameter("broken", 0.0);
    doc.set_parameter_expression(broken, "1 / 0").unwrap();
    let failed = doc.add_part(ball("failed", ParamRef::Named(broken)));
    let unknown = doc.add_part(ball("unknown", ParamRef::Named(ParamId::new())));

    let from_failure = radius_of(&doc, failed).unwrap_err();
    let from_nothing = radius_of(&doc, unknown).unwrap_err();
    assert!(matches!(from_failure, CompileError::UnresolvedRadius), "{from_failure}");
    assert_eq!(from_failure.to_string(), from_nothing.to_string());
}

/// The document-scoping ruling made testable: one `wall`, read by two parts, moves both.
#[test]
fn two_parts_reading_wall_both_change_when_it_does() {
    let mut doc = Document::new();
    let wall = doc.add_parameter("wall", 2.0);
    let doubled = doc.add_parameter("doubled", 0.0);
    doc.set_parameter_expression(doubled, "wall * 2").unwrap();
    let housing = doc.add_part(ball("housing", ParamRef::Named(wall)));
    let lid = doc.add_part(ball("lid", ParamRef::Named(doubled)));

    let both = |doc: &Document| (radius_of(doc, housing).unwrap(), radius_of(doc, lid).unwrap());
    assert_eq!(both(&doc), (2.0, 4.0));
    doc.set_parameter_literal(wall, 3.0);
    assert_eq!(both(&doc), (3.0, 6.0));
}

/// The extent sizes the view and the bake, and it has to see a computed radius the way it sees
/// a typed one. This is the bug the first person to try the feature found: the panel said 53,
/// the extent walk read only a typed literal and fell back to 0, and the view was sized to a
/// point while a 53 mm sphere filled it.
#[test]
fn a_computed_radius_sizes_the_extent_like_a_typed_one() {
    let mut doc = Document::new();
    let r = doc.add_parameter("r", 0.0);
    doc.set_parameter_expression(r, "51 + 2").unwrap();
    let computed = doc.add_part(ball("computed", ParamRef::Named(r)));
    let typed = doc.add_part(ball("typed", ParamRef::Literal(53.0)));
    let point = doc.add_part(ball("point", ParamRef::Literal(0.0)));

    let extent = |part: PartId| {
        let graph = &doc.parts[&part].graph;
        graph_extent(&doc, graph, graph.roots()[0])
    };
    assert_eq!(extent(computed), extent(typed));
    assert_ne!(extent(computed), extent(point), "a radius of nothing is a different box");
}


/// A computed number sizes a lattice, which is the two halves of this branch meeting.
///
/// **Neither side could have this on its own.** Expression parameters arrived while the lattice
/// was growing a phase and eleven cells, and the interesting thing is the product: a strut radius
/// or a wall thickness is a `ParamRef` like any other, so it can be *derived* — `strut = wall / 3`
/// — and a printer's minimum feature becomes one number the whole document follows.
///
/// Measured on the field rather than on the parameter, because the parameter resolving is
/// `fieldform-scene`'s business and this is about the number landing in a strut. A beam lattice is
/// the right instrument: its radius is exact, with nothing averaged or tabulated between the
/// number typed and the geometry built, so the distance at a strut's own centre is exactly minus
/// the radius.
#[test]
fn a_computed_number_sizes_a_beam_lattice_and_moving_it_moves_the_struts() {
    let mut doc = Document::new();
    let wall = doc.add_parameter("wall", 3.0);
    let strut = doc.add_parameter("strut", 0.0);
    doc.set_parameter_expression(strut, "wall / 3").unwrap();

    let mut graph = Graph::empty();
    // Cubic, so a strut runs along X straight through the origin and the field at the origin is
    // exactly minus the radius.
    graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Cubic,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::Strut { radius: ParamRef::Named(strut) },
    });
    let part = doc.add_part(Part {
        id: PartId::new(),
        name: "beams".into(),
        material: None,
        graph,
        inputs: Vec::new(),
    });

    let radius_at_a_strut = |doc: &Document| {
        let field = GraphField::compile(doc, &doc.parts[&part], 0.5).expect("compiles");
        -field.distance(Point3::origin())
    };

    assert!(
        (radius_at_a_strut(&doc) - 1.0).abs() < 1e-9,
        "wall/3 of 3 mm should be a 1 mm strut, got {}",
        radius_at_a_strut(&doc),
    );
    // Move the one number the expression reads, and the struts follow.
    doc.set_parameter_literal(wall, 7.5);
    assert!(
        (radius_at_a_strut(&doc) - 2.5).abs() < 1e-9,
        "wall/3 of 7.5 mm should be a 2.5 mm strut, got {}",
        radius_at_a_strut(&doc),
    );
}

/// The same for a solid lattice's density, which is the one that goes through a table.
///
/// Worth its own case because the path differs: a strut radius is subtracted from a distance,
/// where a density is read through a measured per-cell table before it becomes an isovalue. Both
/// have to accept a computed number, and the table is the one that could quietly clamp it.
#[test]
fn a_computed_number_sizes_a_solid_lattice() {
    let mut doc = Document::new();
    let fraction = doc.add_parameter("fraction", 0.0);
    doc.set_parameter_expression(fraction, "1 / 4").unwrap();

    let mut graph = Graph::empty();
    graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::Solid { density: ParamRef::Named(fraction) },
    });
    let part = doc.add_part(Part {
        id: PartId::new(),
        name: "infill".into(),
        material: None,
        graph,
        inputs: Vec::new(),
    });

    // A quarter of every cell is material, counted the way the density table was built.
    let field = GraphField::compile(&doc, &doc.parts[&part], 0.5).expect("compiles");
    let n = 30;
    let mut material = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let at = Point3::new(
                    (i as f64 + 0.5) * 20.0 / n as f64,
                    (j as f64 + 0.5) * 20.0 / n as f64,
                    (k as f64 + 0.5) * 20.0 / n as f64,
                );
                if field.distance(at) < 0.0 {
                    material += 1;
                }
            }
        }
    }
    let achieved = material as f64 / (n * n * n) as f64;
    assert!(
        (achieved - 0.25).abs() < 0.03,
        "1/4 should be a quarter of the cell, got {achieved}",
    );
}
