// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a lattice's material can already be graded through space, and at what cost.
//!
//! The question this answers before anything is built: an offset is `F − d`, and for a lattice
//! that is a *thicker wall* or a *fatter strut* — the field is already in millimetres, so
//! subtracting a driven amount grows the material by that amount. If a ramp over an offset over a
//! lattice already grades it, the feature is a panel away rather than a mechanism away.

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    CellMap, Document, Graph, LatticeCell, LatticePhase, Node, ParamRef, Part, PartId,
    Place, RampShape,
};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// The same, over a stated input range, so a grade's *steepness* can be varied.
fn graded_over(
    cell: LatticeCell,
    phase: LatticePhase,
    extra: f64,
    over: f64,
) -> (Document, PartId) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase,
    });
    let plane = graph.add(Node::Plane {
        origin: Place::At([lit(0.0); 3]),
        flipped: false,
        turn: fieldform_scene::no_turn(),
    });
    let ramp = graph.add(Node::Ramp {
        input: plane.into(),
        in_min: lit(-over / 2.0),
        in_max: lit(over / 2.0),
        out_min: lit(0.0),
        out_max: lit(extra),
        shape: RampShape::Linear,
    });
    graph.add(Node::Offset { input: lattice.into(), distance: ParamRef::Field(ramp) });
    let part = Part {
        id: PartId::new(),
        name: "graded".into(),
        material: None,
        graph,
        inputs: Vec::new(),
    };
    let id = part.id;
    doc.add_part(part);
    (doc, id)
}

/// A lattice graded along Z by a ramp read off a plane at the origin.
///
/// `out_min`/`out_max` are how much extra material the offset adds, so 0 at z = −40 and `extra`
/// at z = +40.
fn graded(cell: LatticeCell, phase: LatticePhase, extra: f64) -> (Document, PartId) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase,
    });
    // A plane through the origin: its field is the signed distance along its normal, which is
    // the simplest thing to read a gradient off.
    let plane = graph.add(Node::Plane {
        origin: Place::At([lit(0.0); 3]),
        flipped: false,
        turn: fieldform_scene::no_turn(),
    });
    let ramp = graph.add(Node::Ramp {
        input: plane.into(),
        in_min: lit(-40.0),
        in_max: lit(40.0),
        out_min: lit(0.0),
        out_max: lit(extra),
        shape: RampShape::Linear,
    });
    let graded = graph.add(Node::Offset { input: lattice.into(), distance: ParamRef::Field(ramp) });
    let _ = graded;
    let part = Part {
        id: PartId::new(),
        name: "graded".into(),
        material: None,
        graph,
        inputs: Vec::new(),
    };
    let id = part.id;
    doc.add_part(part);
    (doc, id)
}

/// How much material there is in a slab around the given Z, as a fraction of it.
///
/// Z because that is the axis the ramp grades along: a plane's field is the signed distance
/// along its own normal, and an unturned plane's normal is +Z.
fn density_near(doc: &Document, part: PartId, z: f64) -> f64 {
    let field = GraphField::compile(doc, &doc.parts[&part], 0.5).expect("compiles");
    let n = 24;
    let mut material = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let at = Point3::new(
                    (i as f64 + 0.5) * 20.0 / n as f64,
                    (j as f64 + 0.5) * 20.0 / n as f64,
                    z + (k as f64 + 0.5) * 20.0 / n as f64,
                );
                if field.distance(at) < 0.0 {
                    material += 1;
                }
            }
        }
    }
    material as f64 / (n * n * n) as f64
}

/// **The question.** A ramp over an offset over a lattice: is there more material where the ramp
/// says there should be?
#[test]
fn an_offset_driven_by_a_ramp_already_grades_a_sheet_lattice() {
    let (doc, part) = graded(
        LatticeCell::Gyroid,
        LatticePhase::sheet(lit(1.0)),
        3.0,
    );
    let thin = density_near(&doc, part, -30.0);
    let thick = density_near(&doc, part, 20.0);
    assert!(
        thick > thin * 1.5,
        "the ramp asked for more material at +z and got {thick:.3} against {thin:.3}",
    );
}

/// The same for a beam cell, where the arithmetic is exactly a strut radius.
#[test]
fn an_offset_driven_by_a_ramp_already_grades_a_beam_lattice() {
    let (doc, part) = graded(
        LatticeCell::Bcc,
        LatticePhase::Strut { radius: lit(0.8) },
        2.5,
    );
    let thin = density_near(&doc, part, -30.0);
    let thick = density_near(&doc, part, 20.0);
    assert!(
        thick > thin * 1.5,
        "the ramp asked for fatter struts at +z and got {thick:.3} against {thin:.3}",
    );
}

/// And a solid, where growing the labyrinth is what "denser" means.
#[test]
fn an_offset_driven_by_a_ramp_already_grades_a_solid_lattice() {
    let (doc, part) = graded(
        LatticeCell::Gyroid,
        LatticePhase::Solid { density: lit(0.3) },
        2.0,
    );
    let thin = density_near(&doc, part, -30.0);
    let thick = density_near(&doc, part, 20.0);
    assert!(
        thick > thin * 1.2,
        "the ramp asked for a denser solid at +z and got {thick:.3} against {thin:.3}",
    );
}

/// The safety property, which the divisor exists for: a graded lattice must still never claim
/// more room than there is.
#[test]
fn a_graded_lattice_never_overstates_the_room_before_its_material() {
    let (doc, part) = graded(
        LatticeCell::Gyroid,
        LatticePhase::sheet(lit(1.0)),
        3.0,
    );
    let field = GraphField::compile(&doc, &doc.parts[&part], 0.5).expect("compiles");
    let n = 11;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let from = Point3::new(
                    20.0 * i as f64 / n as f64,
                    20.0 * j as f64 / n as f64,
                    -35.0 + 70.0 * k as f64 / n as f64,
                );
                let claimed = field.distance(from);
                if claimed <= 0.0 {
                    continue;
                }
                for step in 1..=24 {
                    let t = claimed * step as f64 / 24.0;
                    let at = Point3::new(from.x, from.y, from.z + t);
                    assert!(
                        field.distance(at) > -1e-9,
                        "crossed material {t} into a claimed {claimed} at {from:?}",
                    );
                }
            }
        }
    }
}

/// What grading costs a ray, which is the thing that decides whether this is usable.
///
/// **A divisor is paid in steps, and a ray that runs out of them is indistinguishable from one
/// that hit nothing** — so the surface goes missing rather than getting slower.
/// `step_multiplier`'s own doc records ×8.5 leaving a limb ray needing 2791 steps with 512.
/// A lattice starts from a value already divided by a gradient bound, so grading multiplies two
/// costs, and the worst pairing in the catalogue is a graded Neovius.
///
/// Printed rather than asserted tight: the number is the finding, and a threshold here would
/// only encode today's ramp range.
#[test]
fn what_grading_costs_a_ray_is_measured_rather_than_assumed() {
    for (label, cell, phase, extra) in [
        ("gyroid sheet", LatticeCell::Gyroid, LatticePhase::sheet(lit(1.0)), 3.0),
        ("neovius sheet", LatticeCell::Neovius, LatticePhase::sheet(lit(1.0)), 3.0),
        ("bcc struts", LatticeCell::Bcc, LatticePhase::Strut { radius: lit(0.8) }, 2.5),
    ] {
        let (doc, part) = graded(cell, phase, extra);
        let graph = &doc.parts[&part].graph;
        let multiplier = fieldform_field::step_multiplier(&doc, graph);
        println!("{label}: step multiplier {multiplier:.2}x");
        assert!(
            multiplier.is_finite() && multiplier >= 1.0,
            "{label}: {multiplier} is not a step cost",
        );
    }
}

/// Where the step cost actually bites: it is the grade's *steepness*, not the grading.
///
/// `L = k·|Δout|/|Δin|` times the input's own constant, so the same 3 mm of extra material costs
/// nothing spread over 80 mm and a great deal squeezed into 3. The gentle end is the one a part
/// is usually designed at, which is why the figures above are reassuring and this table is the
/// one to read before ramping hard.
#[test]
fn the_step_cost_of_a_grade_is_its_steepness() {
    println!("3 mm of extra material, spread over:");
    for over in [80.0, 40.0, 10.0, 3.0, 1.0, 0.3] {
        let (doc, part) = graded_over(
            LatticeCell::Gyroid,
            LatticePhase::sheet(lit(1.0)),
            3.0,
            over,
        );
        let graph = &doc.parts[&part].graph;
        println!(
            "   {over:>5.1} mm  ->  {:.2}x steps",
            fieldform_field::step_multiplier(&doc, graph),
        );
    }
}

/// The wall's actual thickness along the grade, against what the ramp asked for.
///
/// **The test that should have existed from the start.** The three above check that there is
/// *more* material where the ramp says and that the field never overstates; neither says how
/// *much* more, and a user's eye found that out on a Fischer-Koch S sheet that went solid halfway
/// along its ramp.
///
/// A beam's field is a true distance, so an offset of `d` grows it by `d`. A surface cell's is a
/// lower bound that reads short by its bound-to-gradient ratio, and shorter still in flat regions
/// — so an offset grows the wall until the *value* has fallen by `d`, which is amplified and
/// accelerates as the wall is pushed toward the cell's extrema. This measures the chord across
/// the wall at three heights and pins both facts: the gyroid overshoots the naive figure at
/// mid-ramp, and Fischer-Koch S has no far side to find at all.
/// The chords across a sheet's wall in the 20 mm slab above `z0`, sorted.
///
/// From every probe close to the surface, march along the cell's own gradient both ways to the
/// wall's two faces and add the halves. None at all means no far side was found within two
/// cells: the walls have met and the slab is solid.
fn wall_chords(field: &GraphField, kind: fieldform_field::TpmsKind, z0: f64) -> Vec<f64> {
    let tau = std::f64::consts::TAU;
    let cell = 20.0;
    let n = 14;
    let mut chords = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let at = Point3::new(
                    (i as f64 + 0.5) * cell / n as f64,
                    (j as f64 + 0.5) * cell / n as f64,
                    z0 + (l as f64 + 0.5) * cell / n as f64,
                );
                let args = Point3::from(at.coords * tau / cell);
                if kind.f(args).abs() > 0.25 {
                    continue;
                }
                let Some(dir) = kind.gradient(args).try_normalize(1e-9) else { continue };
                let crossing = |sign: f64| -> Option<f64> {
                    let step = 0.05;
                    let mut t = 0.0;
                    while field.distance(at + dir * (sign * t)) < 0.0 {
                        t += step;
                        if t > 2.0 * cell {
                            return None;
                        }
                    }
                    let (mut inside, mut outside) = (t - step, t);
                    for _ in 0..30 {
                        let mid = 0.5 * (inside + outside);
                        if field.distance(at + dir * (sign * mid)) < 0.0 {
                            inside = mid;
                        } else {
                            outside = mid;
                        }
                    }
                    Some(0.5 * (inside + outside))
                };
                if let (Some(a), Some(b)) = (crossing(1.0), crossing(-1.0)) {
                    chords.push(a + b);
                }
            }
        }
    }
    chords.sort_by(f64::total_cmp);
    chords
}

/// A lattice whose phase's own number follows a ramp along Z: `low` at z = -40, `high` at +40.
///
/// The thing `graded` stood in for with an offset. `phase` is handed the reference to put where
/// its number goes, so the same helper grades a wall, a density and a strut.
fn driven(
    cell: LatticeCell,
    map: CellMap,
    frame: fieldform_scene::Frame,
    phase: fn(ParamRef) -> LatticePhase,
    low: f64,
    high: f64,
) -> (Document, PartId) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let plane = graph.add(Node::Plane {
        origin: Place::At([lit(0.0); 3]),
        flipped: false,
        turn: fieldform_scene::no_turn(),
    });
    let ramp = graph.add(Node::Ramp {
        input: plane.into(),
        in_min: lit(-40.0),
        in_max: lit(40.0),
        out_min: lit(low),
        out_max: lit(high),
        shape: RampShape::Linear,
    });
    graph.add(Node::Lattice { frame, cell, map, phase: phase(ParamRef::Field(ramp)) });
    let part = Part {
        id: PartId::new(),
        name: "driven".into(),
        material: None,
        graph,
        inputs: Vec::new(),
    };
    let id = part.id;
    doc.add_part(part);
    (doc, id)
}

fn rectangular() -> CellMap {
    CellMap::Rectangular { cell: lit(20.0) }
}

/// A plain lattice with a typed wall, to hold a graded one against.
fn fixed_wall(cell: LatticeCell, thickness: f64) -> (Document, PartId) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell,
        map: rectangular(),
        phase: LatticePhase::sheet(lit(thickness)),
    });
    let part = Part {
        id: PartId::new(),
        name: "fixed".into(),
        material: None,
        graph,
        inputs: Vec::new(),
    };
    let id = part.id;
    doc.add_part(part);
    (doc, id)
}

/// **A graded wall at a place is the typed wall of that thickness.** The same 2 mm to 5 mm
/// grade `how_much_an_offset_actually_grows_a_sheet_along_the_grade` measures through an offset,
/// where a gyroid asked for 5 mm reads 7.4. Driven, the gyroid's median is within a few per cent
/// of a typed wall's at every station.
///
/// Held against the typed wall rather than against the number, because the typed wall is what
/// is promised. A Fischer-Koch S wall past about a sixth of its cell has already merged with its
/// neighbours when it is *typed* -- the chord finds the far side of the next wall -- and the
/// graded one does exactly the same, which is the property and not a failure of it.
#[test]
fn a_driven_wall_is_the_typed_wall_of_that_thickness() {
    use fieldform_field::TpmsKind;
    let median = |chords: &[f64]| if chords.is_empty() { f64::INFINITY } else { chords[chords.len() / 2] };
    for (name, lattice_cell, kind) in [
        ("gyroid", LatticeCell::Gyroid, TpmsKind::Gyroid),
        ("fischer-koch s", LatticeCell::FischerKochS, TpmsKind::FischerKochS),
    ] {
        let (doc, part) = driven(
            lattice_cell,
            rectangular(),
            fieldform_scene::Frame::default(),
            |thickness| LatticePhase::sheet(thickness),
            2.0,
            5.0,
        );
        let graded = GraphField::compile(&doc, &doc.parts[&part], 0.5).expect("compiles");
        for z0 in [-40.0, -10.0, 20.0] {
            // The ramp's value at the middle of the slab the chords are taken in.
            let asked = 2.0 + 3.0 * (z0 + 10.0 + 40.0) / 80.0;
            let (typed_doc, typed_part) = fixed_wall(lattice_cell, asked);
            let typed = GraphField::compile(&typed_doc, &typed_doc.parts[&typed_part], 0.5)
                .expect("compiles");
            let (got, want) = (
                median(&wall_chords(&graded, kind, z0)),
                median(&wall_chords(&typed, kind, z0)),
            );
            println!(
                "{name:>15}  z {z0:>4.0}  asked {asked:.2} mm  graded {got:6.2} mm  typed {want:6.2} mm"
            );
            if want < 1.5 * asked {
                // A wall that is still a wall. The slab spans a quarter of the ramp, so the
                // graded median is of walls a little either side of the typed one.
                assert!(
                    (got - want).abs() < 0.15 * want,
                    "{name} at z {z0}: graded {got:.2} mm against a typed {want:.2} mm"
                );
            } else {
                // Merged when typed, so merged when graded.
                assert!(
                    got > 1.5 * asked,
                    "{name} at z {z0}: a typed wall has merged ({want:.2} mm) and the graded one                      reads {got:.2} mm"
                );
            }
        }
    }
}

/// The contract, for every phase a ramp can drive and on a map that is not a uniform scale.
/// Marched along the grade, which is the direction the driven term adds its rate in, and across
/// it.
#[test]
fn a_driven_phase_never_overstates_the_room_before_its_material() {
    let cylindrical = || CellMap::Cylindrical {
        cell_radius: lit(20.0),
        cell_height: lit(20.0),
        arc_count: 12,
    };
    let cases: Vec<(&str, LatticeCell, CellMap, fn(ParamRef) -> LatticePhase, f64, f64)> = vec![
        ("a wall", LatticeCell::Gyroid, rectangular(), |t| LatticePhase::sheet(t), 0.5, 5.0),
        ("a steep wall", LatticeCell::Neovius, rectangular(), |t| LatticePhase::sheet(t), 0.5, 4.0),
        ("a density", LatticeCell::SchwarzP, rectangular(), |d| LatticePhase::Solid { density: d }, 0.1, 0.8),
        ("a strut", LatticeCell::Bcc, rectangular(), |r| LatticePhase::Strut { radius: r }, 0.5, 4.0),
        ("a wall about an axis", LatticeCell::Gyroid, cylindrical(), |t| LatticePhase::sheet(t), 0.5, 4.0),
    ];
    for (name, cell, map, phase, low, high) in cases {
        let (doc, part) = driven(cell, map, fieldform_scene::Frame::default(), phase, low, high);
        let field = GraphField::compile(&doc, &doc.parts[&part], 0.5).expect("compiles");
        let n = 9;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    // Clear of the axis, where a cylindrical map has closed up and says so.
                    let from = Point3::new(
                        45.0 + 20.0 * i as f64 / n as f64,
                        5.0 + 20.0 * j as f64 / n as f64,
                        -35.0 + 70.0 * k as f64 / n as f64,
                    );
                    let claimed = field.distance(from);
                    if claimed <= 0.0 {
                        continue;
                    }
                    for dir in [nalgebra::Vector3::z(), nalgebra::Vector3::x(), -nalgebra::Vector3::z()] {
                        for step in 1..=24 {
                            let t = claimed * step as f64 / 24.0;
                            assert!(
                                field.distance(from + dir * t) > -1e-9,
                                "{name}: crossed material {t} into a claimed {claimed} at {from:?}",
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Only the phase's own number, and only on a lattice read where the ramp is read.
#[test]
fn a_ramp_may_drive_a_lattices_phase_and_nothing_else_on_it() {
    use fieldform_field::CompileError;
    // A cell size that varies is a coordinate warp, not a number that varies.
    let (mut doc, part) = driven(
        LatticeCell::Gyroid,
        rectangular(),
        fieldform_scene::Frame::default(),
        |t| LatticePhase::sheet(t),
        1.0,
        2.0,
    );
    let ramp = doc.parts[&part]
        .graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Ramp { .. }))
        .map(|(id, _)| id)
        .expect("the ramp");
    let lattice = doc.parts[&part]
        .graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Lattice { .. }))
        .map(|(id, _)| id)
        .expect("the lattice");
    if let Some(Node::Lattice { map, phase, .. }) =
        doc.parts.get_mut(&part).unwrap().graph.node_mut(lattice)
    {
        *map = CellMap::Rectangular { cell: ParamRef::Field(ramp) };
        *phase = LatticePhase::sheet(lit(1.0));
    }
    match GraphField::compile(&doc, &doc.parts[&part], 0.5).map(|_| ()) {
        Err(CompileError::ParameterCannotBeDriven(what)) => assert_eq!(what, "a lattice's cell size"),
        other => panic!("a driven cell size should be refused by name, got {other:?}"),
    }

    // An aimed lattice is asked in its own frame, and the ramp would be read there too.
    let aimed = fieldform_scene::Frame::At {
        origin: [lit(0.0); 3],
        axis: [lit(1.0), lit(0.0), lit(0.0)],
    };
    let (doc, part) = driven(
        LatticeCell::Gyroid,
        rectangular(),
        aimed,
        |t| LatticePhase::sheet(t),
        1.0,
        2.0,
    );
    match GraphField::compile(&doc, &doc.parts[&part], 0.5).map(|_| ()) {
        Err(CompileError::GradedLatticeCannotBeAimed) => {}
        other => panic!("a graded lattice with its own frame should be refused, got {other:?}"),
    }
}

/// A beam is the one lattice that reports a true distance, and grading it ends that: the value
/// is divided by the ramp's rate. The graph walk and the compiler have to say so together, or a
/// fillet against a graded beam would not be marked approximate.
#[test]
fn a_graded_beam_is_a_bound_and_both_walks_say_so() {
    let (doc, part) = driven(
        LatticeCell::Bcc,
        rectangular(),
        fieldform_scene::Frame::default(),
        |r| LatticePhase::Strut { radius: r },
        1.0,
        3.0,
    );
    let graph = &doc.parts[&part].graph;
    let lattice = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Lattice { .. }))
        .map(|(id, _)| id)
        .expect("the lattice");
    assert!(!fieldform_field::carries_true_distance(&doc, graph, lattice));
    let field = GraphField::compile(&doc, &doc.parts[&part], 0.5).expect("compiles");
    assert_eq!(field.normalization(), fieldform_field::Normalization::LowerBound);
}

/// What the grade costs a ray, which is the ramp's rate times how hard the phase's number moves
/// the value: a half of `g / B` for a wall, so less than the `1 + L` of the offset that used to
/// stand in for it.
#[test]
fn a_graded_wall_costs_less_than_the_offset_that_stood_in_for_it() {
    let (doc, part) = driven(
        LatticeCell::Gyroid,
        rectangular(),
        fieldform_scene::Frame::default(),
        |t| LatticePhase::sheet(t),
        2.0,
        5.0,
    );
    let rate = 3.0 / 80.0;
    let driven_cost = fieldform_field::step_multiplier(&doc, &doc.parts[&part].graph);
    println!("a wall graded at {rate:.4} per mm costs x{driven_cost:.4}; an offset costs x{:.4}", 1.0 + rate);
    assert!(driven_cost > 1.0, "a grade is not free");
    assert!(driven_cost < 1.0 + 0.5 * rate + 1e-12, "never more than half the ramp's rate");
}

#[test]
fn how_much_an_offset_actually_grows_a_sheet_along_the_grade() {
    use fieldform_field::TpmsKind;
    // (name, z0, median chord, chords found)
    let mut rows: Vec<(&str, f64, f64, usize)> = Vec::new();
    for (name, lattice_cell, kind) in [
        ("gyroid", LatticeCell::Gyroid, TpmsKind::Gyroid),
        ("fischer-koch s", LatticeCell::FischerKochS, TpmsKind::FischerKochS),
    ] {
        let (doc, part) = graded(lattice_cell, LatticePhase::sheet(lit(2.0)), 3.0);
        let field = GraphField::compile(&doc, &doc.parts[&part], 0.5).expect("compiles");
        let k = kind.gradient_bound() / kind.surface_gradient();
        println!("{name}: 2 mm sheet, ramp 0..3 mm over z = -40..40, cost ratio k = {k:.2}");
        for z0 in [-40.0, 0.0, 40.0] {
            let d = 3.0 * (z0 + 40.0) / 80.0;
            let chords = wall_chords(&field, kind, z0);
            let median = if chords.is_empty() { f64::NAN } else { chords[chords.len() / 2] };
            println!(
                "   z {z0:>4.0}  ramp d = {d:.2} mm   wall median {median:6.2} mm   \
                 naive 2+2d = {:5.2}   ({} chords; none = the walls have met)",
                2.0 + 2.0 * d,
                chords.len(),
            );
            rows.push((name, z0, median, chords.len()));
        }
    }

    let at = |name: &str, z: f64| rows.iter().find(|r| r.0 == name && r.1 == z).copied().unwrap();
    // The gyroid, the tightest cell, already overshoots the naive 5 mm at mid-ramp...
    let (_, _, gyroid_mid, gyroid_n) = at("gyroid", 0.0);
    assert!(gyroid_n > 0 && gyroid_mid > 5.0, "gyroid at mid-ramp: {gyroid_mid} over {gyroid_n}");
    // ...and Fischer-Koch S, at nearly four times the ratio, has gone solid there: no chord
    // finds a far side. If this ever finds one, grading has changed — read the number, do not
    // just loosen this.
    let (_, _, _, fks_n) = at("fischer-koch s", 0.0);
    assert_eq!(fks_n, 0, "a Fischer-Koch S sheet should be solid by mid-ramp");
}
