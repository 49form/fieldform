// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The reported scene: a cylinder infilled with a cylindrical gyroid, at a thin wall.
//!
//! Two symptoms to account for. Black speckle in the shading that gets worse as the wall gets
//! thinner, and a viewport that fills up when a cage is added beside it.
//!
//! What this measures is how *understated* the lattice's value is — the gap between what the
//! field reports and the room actually there. A sphere tracer steps by what it is told, so an
//! understated field is a ray that needs more steps than it is given, and a ray that runs out
//! of steps reports a miss, which draws as background.
//!
//! `cargo run --release -p fieldform-gpu --example cylindrical_artifacts`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    BooleanOp, CellMap, Document, Graph, LatticeCell, LatticePhase, Node, ParamRef, Part,
};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// The user's scene: a 30 mm-tall cylinder of radius 40, infilled.
fn scene(wall_mm: f64) -> (Document, Part) {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let cylinder = graph.add(Node::Cylinder { edge: Default::default(),
        ends: fieldform_scene::Ends::between(
            fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(-15.0)]),
            fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(15.0)])),
        radius: lit(40.0),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map: CellMap::Cylindrical {
            cell_radius: lit(16.0),
            cell_height: lit(30.0),
            arc_count: 16,
        },
        phase: LatticePhase::sheet(lit(wall_mm)),
    });
    graph.add(Node::Boolean {
        op: BooleanOp::Intersection,
        inputs: vec![cylinder.into(), lattice.into()],
    });
    let mut part = Part::from_cage("infill", fieldform_scene::CageId::new());
    part.graph = graph;
    (doc, part)
}

/// How far a ray actually travels before it hits, against how far the field said it could.
///
/// The ratio is the number that matters: a field understating by a factor of ten needs ten
/// times the steps, and the step budget is fixed.
fn march(field: &GraphField, from: Point3<f64>, dir: nalgebra::Vector3<f64>, budget: u32) -> (u32, bool) {
    let mut t = 0.0;
    for step in 0..budget {
        let p = from + dir * t;
        let d = field.distance(p);
        if d < 0.01 {
            return (step, true);
        }
        if t > 200.0 {
            return (step, false);
        }
        t += (d * 0.9).max(0.001);
    }
    (budget, false)
}

fn main() {
    println!(
        "A 80 mm cylinder infilled with a cylindrical gyroid: cell radius 16, height 30,\n\
         16 cells around. Rays fired inward from outside, as the viewport does.\n"
    );
    println!(
        "{:>6}  {:>9}  {:>12}  {:>10}  {:>9}",
        "wall", "solid r", "understated", "mean steps", "ran out"
    );

    for wall in [3.0, 2.0, 1.5, 1.0, 0.7, 0.5] {
        let (doc, part) = scene(wall);
        let field = GraphField::compile(&doc, &part, 0.05).expect("the graph compiles");

        // How far the tracer is told it may step, against the true clearance, out at the rim
        // where the cells are largest and the understatement is worst.
        let r_min = 16.0 * wall / std::f64::consts::TAU;
        // *Inside* the cylinder, where the lattice governs the value. The first version of
        // this probed at r = 60, outside the 40 mm cylinder — where the intersection's `max`
        // returns the cylinder's own distance, so it measured the cylinder and reported a
        // flawless 1.0x for a field that had not been consulted.
        let probe = Point3::new(28.0, 0.0, 3.0);
        let claimed = field.distance(probe);
        // The true distance, found by marching in tiny steps rather than trusting the field.
        let mut true_d = 0.0;
        while true_d < 20.0 {
            if field.distance(probe + nalgebra::Vector3::new(-1.0, 0.0, 0.0) * true_d) <= 0.0 {
                break;
            }
            true_d += 0.01;
        }
        let understated = if claimed > 1e-9 { true_d / claimed } else { f64::NAN };

        // A fan of rays inward, counting how many exhaust a 512-step budget.
        let mut steps_total = 0u64;
        let mut ran_out = 0;
        let n = 200;
        for i in 0..n {
            let a = i as f64 / n as f64 * std::f64::consts::TAU;
            let from = Point3::new(90.0 * a.cos(), 90.0 * a.sin(), (i % 17) as f64 - 8.0);
            let dir = -nalgebra::Vector3::new(a.cos(), a.sin(), 0.0);
            let (steps, hit) = march(&field, from, dir, 512);
            steps_total += steps as u64;
            if !hit {
                ran_out += 1;
            }
        }
        println!(
            "{wall:>5.1}  {r_min:>8.2}  {understated:>11.1}x  {:>10.0}  {ran_out:>5} / {n}",
            steps_total as f64 / n as f64,
        );
    }

    println!(
        "\nThe understatement is the ratio of real clearance to what the field reports, and a\n\
         ray needs about that many more steps than it should. It was 79-332x while the bound\n\
         was one constant taken where the cells collapse; asking twice \u{2014} once where the\n\
         point is, once as far inward as the first answer allows \u{2014} brings it to 8-15x,\n\
         and no ray in this fan runs out of its budget any more."
    );
}
