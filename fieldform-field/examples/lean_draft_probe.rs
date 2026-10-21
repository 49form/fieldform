// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which leaning, drafted extrudes claim more room than there is, by side arrangement.
//!
//! Written when the two-sided case of `extrude_sides_tests` overstated by 1.37 mm along a
//! lean and the one-sided case had never been asked the same question in that test. Prints
//! the worst overstatement for each arrangement, upright and leaning, drafted and not.
//!
//! Run with: `cargo run --release -p fieldform-field --example lean_draft_probe`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Sides, Sweep};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn main() {
    let probes: Vec<Point3<f64>> = (0..15)
        .flat_map(|i| (0..15).flat_map(move |j| (0..21).map(move |k| (i, j, k))))
        .map(|(i, j, k)| {
            Point3::new(
                -21.0 + i as f64 * 3.1 + 0.13 * k as f64,
                -22.0 + j as f64 * 3.2 + 0.07 * i as f64,
                -31.0 + k as f64 * 3.05 + 0.11 * j as f64,
            )
        })
        .collect();
    let sweeps = [
        ("upright", Sweep::Normal),
        ("leaning (0.4, 0.2, 1)", Sweep::Toward { xyz: [lit(0.4), lit(0.2), lit(1.0)] }),
        ("leaning (1, 0, 1)", Sweep::Toward { xyz: [lit(1.0), lit(0.0), lit(1.0)] }),
    ];
    let arrangements: [(&str, Sides); 4] = [
        ("one side", Sides::One),
        ("symmetric", Sides::Symmetric),
        ("two sides, alike", Sides::Two { height: lit(12.0), draft: lit(25.0) }),
        ("two sides, 25 and -8", Sides::Two { height: lit(12.0), draft: lit(-8.0) }),
    ];
    println!("{:<24} {:<24} {:>6} {:>6} {:>12}", "sweep", "sides", "start", "draft", "overstated");
    for (sweep_name, sweep) in &sweeps {
        for (name, sides) in &arrangements {
            for (start, draft) in [(0.0, 0.0), (0.0, 25.0), (4.0, 25.0)] {
                let mut doc = Document::new();
                let id = CurveId::new();
                let corners = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)];
                let points = corners.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect();
                doc.curves.insert(id, Curve::polyline(id, "square", points, true).unwrap());
                let mut graph = Graph::empty();
                graph.add(Node::Extrude { rims: Default::default(),
                    outline: Outline::closed(id),
                    sweep: sweep.clone(),
                    start: lit(start),
                    height: lit(20.0),
                    draft: lit(draft),
                    sides: sides.clone(),
                    corners: Default::default(),
                });
                let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
                let over = fieldform_field::overstatement(
                    &field,
                    &probes,
                    &fieldform_field::probe_directions(),
                    0.05,
                    1.2,
                );
                let inside = probes.iter().filter(|p| field.distance(**p) < 0.0).count();
                println!(
                    "{sweep_name:<24} {name:<24} {start:>6} {draft:>6} {over:>9.4} mm  ({inside} probes inside)"
                );
            }
        }
    }
}
