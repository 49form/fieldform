// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Writes a `.fieldform` document of `n` pipes on spokes in one union, for testing the union loop.
//!
//! **The shape the loop is for, as a document rather than as a harness.** `union_loop_cost`
//! measures it on synthetic graphs; this is the same thing to open in the app, because what the
//! change is worth is a first render you do not wait for, and no harness can show that.
//!
//! Sixteen pipes by default. Before the loop that document's shader cost the driver **95 060 ms**
//! cold — a minute and a half of frozen application on the first sight of it — and its frame was
//! 36.4 ms at 720 px. After: **339 ms** and 18.7 ms. See `union-compile-8fa06b7.md`.
//!
//! `FIELDFORM_NO_UNION_LOOP=1` on the *application* puts the old composition back, so the same
//! file opened twice is the before and the after.
//!
//! Run with: `cargo run -p fieldform-io --example spoke_union_demo -- <path> [n]`

use fieldform_scene::{
    BooleanOp, CageId, Curve, CurveId, Document, Graph, Node, ParamId, ParamRef, ParamValue,
    Parameter, Part, PartId,
};
use nalgebra::Point3;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "spoke-union.fieldform".to_string());
    let path = std::path::PathBuf::from(path);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(16);

    let mut doc = Document::new();

    // **A cage nothing references, because the application will not open a document without
    // one.** `App::open_document` looks for a cage in the part's graph and then anywhere in the
    // document, and refuses with "That document has no cage to edit" if it finds neither. Left
    // unreferenced: it is here to satisfy that check, and draws nothing in field mode.
    doc.cages.insert(CageId::new(), fieldform_cage::Cage::cube_of(50.0).into());

    // One named radius read by every pipe, so dragging it re-evaluates rather than recompiling —
    // which is also the thing to check: the loop's count is a buffer value, so changing a radius
    // must not put a driver compile inside the drag.
    let radius = ParamId::new();
    doc.parameters.insert(
        radius,
        Parameter {
            id: radius,
            name: "radius".to_string(),
            value: ParamValue::Literal(9.0),
        },
    );

    let mut graph = Graph::empty();
    let mut pipes = Vec::new();
    for i in 0..n {
        let turn = std::f64::consts::TAU * i as f64 / n as f64;
        let (c, s) = (turn.cos(), turn.sin());
        let id = CurveId::new();
        let at = |r: f64, z: f64| Point3::new(r * c, r * s, z);
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                &format!("spoke {}", i + 1),
                // Bent, and each bent differently, so a run that read the wrong spine would show
                // as a pipe in the wrong place rather than as the same picture.
                vec![
                    at(20.0, 0.0),
                    at(80.0, 25.0 - 2.0 * i as f64),
                    at(140.0, -10.0 + i as f64),
                    at(200.0, 15.0),
                ],
                false,
            )
            .expect("a bent line"),
        );
        pipes.push(graph.add(Node::Pipe { curve: id, radius: ParamRef::Named(radius) }));
    }
    graph.add(Node::Boolean { op: BooleanOp::Union, inputs: pipes.into_iter().map(Into::into).collect() });

    let part = Part {
        id: PartId::new(),
        name: format!("{n} pipes, one union"),
        material: None,
        graph,
        inputs: Vec::new(),
    };
    doc.parts.insert(part.id, part);

    match fieldform_io::save_new(&doc, &path) {
        Ok(()) => {
            println!("wrote {} — {n} pipes on spokes, one union", path.display());
            println!();
            println!("Open it with File > Open. What to watch:");
            println!("  the first render. The union is walked as one loop, so the driver compiles");
            println!("  one spine walk instead of {n} — measured 339 ms against 95 060.");
            println!();
            println!("Then drag `radius` in the parameters panel: every pipe follows, and nothing");
            println!("recompiles, because the count and the radii are buffer values.");
            println!();
            println!("For the before: launch the app with FIELDFORM_NO_UNION_LOOP=1 and open the");
            println!("same file. Same picture, and you will wait for it.");
        }
        Err(e) => {
            eprintln!("could not write {}: {e}", path.display());
            std::process::exit(1);
        }
    }
}
