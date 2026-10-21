// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Writes a `.fieldform` document that demonstrates `Node::Attract`, for opening in the app.
//!
//! Two identical bars beside one curve, drawn toward it and pushed away from it, so the two
//! signs are visible against each other rather than one at a time. Both read the *same* reach
//! parameter, so dragging that one number moves both — which is the point of the parametric
//! layer and the thing a screenshot cannot show.
//!
//! Three named parameters, rather than literals, because that is what makes the demo
//! interactive: an attractor's divisor is `1 + strength` and nothing else, so changing either
//! number re-evaluates rather than rebuilding the shader.
//!
//! Run with: `cargo run -p fieldform-io --example attractor_demo -- <path>`
//! Then File > Open it in the app. Defaults to `attractor-demo.fieldform` beside the workspace.

use fieldform_scene::{
    CageId, Curve, CurveId, Document, Graph, Node, ParamId, ParamRef, ParamValue, Parameter, Part,
    PartId, Place,
};
use nalgebra::Point3;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "attractor-demo.fieldform".to_string());
    let path = std::path::PathBuf::from(path);

    let mut doc = Document::new();

    // **A cage nothing references, because the application will not open a document without
    // one.** `App::open_document` looks for a cage in the part's graph and then anywhere in the
    // document, and refuses with "That document has no cage to edit" if it finds neither — so a
    // document made entirely of primitives and curves cannot be reopened, even though it is
    // perfectly legal and is exactly what `cube` then `line` then `attract` produces. The app
    // only avoids this itself because the document it starts from always carries a cage.
    //
    // Left unreferenced rather than added to the graph: it is here to satisfy that check, not
    // to be part of the demo, and it draws nothing in field mode.
    doc.cages.insert(CageId::new(), fieldform_cage::Cage::cube_of(50.0));

    // The three numbers the demo is driven by. Named rather than literal so the panel offers
    // them by name and a change costs a buffer write.
    let mut named = |name: &str, value: f64| {
        let id = ParamId::new();
        doc.parameters.insert(
            id,
            Parameter { id, name: name.to_string(), value: ParamValue::Literal(value) },
        );
        ParamRef::Named(id)
    };
    let pull = named("pull", 1.0);
    let push = named("push", -0.9);
    let reach = named("reach", 130.0);

    // A degree-3 Bezier running the length of both bars, above them **in Z**, which is this
    // application's up: a rail offset in Y is a pull straight into the screen from the default
    // view, and an effect you have to orbit to see is not a demonstration. Smooth on purpose: a
    // polyline with a hard corner in it has a cut locus, and the seam that produces is a real
    // property of the technique rather than something a demo should open on. See §16 of
    // the attractor-curve-spec design note.
    let curve_id = CurveId::new();
    let curve = Curve::new(
        curve_id,
        "Rail",
        3,
        vec![
            Point3::new(-260.0, 0.0, 100.0),
            Point3::new(-90.0, 0.0, 60.0),
            Point3::new(90.0, 0.0, 60.0),
            Point3::new(260.0, 0.0, 100.0),
        ],
        vec![1.0; 4],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        false,
    )
    .expect("a degree-3 Bezier with four control points");
    doc.curves.insert(curve_id, curve);

    let mut graph = Graph::empty();
    // **Side by side along the rail, not stacked across it.** Stacked, both bars deform
    // downward -- one toward the rail and one away, but both the same way on screen -- so the
    // two signs read as one effect at two strengths. Side by side they sit the same distance
    // from the rail and move in opposite directions, which is the whole point of showing both.
    let bar = |graph: &mut Graph, x: f64| {
        graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([
                ParamRef::Literal(x),
                ParamRef::Literal(0.0),
                ParamRef::Literal(-45.0),
            ]),
            size: [
                ParamRef::Literal(210.0),
                ParamRef::Literal(90.0),
                ParamRef::Literal(90.0),
            ],
            turn: fieldform_scene::no_turn(),
        })
    };

    // Drawn toward the rail: the top face bows up into it and the corners stay behind, which is
    // what makes this a stretch rather than a translation.
    let drawn = bar(&mut graph, -130.0);
    graph.add(Node::Attract { input: drawn.into(), toward: fieldform_scene::Attractor::Curve(curve_id), strength: pull, falloff: reach });

    // And pushed down by the same rail, at the same reach. Not a mirrored pull: the divisor is
    // `1 + (11/16)|s|` on this side rather than `1 + s`, which is the correction that keeps the
    // field from claiming room it has not got.
    let pressed = bar(&mut graph, 130.0);
    graph.add(Node::Attract { input: pressed.into(), toward: fieldform_scene::Attractor::Curve(curve_id), strength: push, falloff: reach });

    let part = Part {
        id: PartId::new(),
        name: "Attractor demo".to_string(),
        material: None,
        graph,
        inputs: Vec::new(),
    };
    doc.parts.insert(part.id, part);

    match fieldform_io::save_new(&doc, &path) {
        Ok(()) => {
            println!("wrote {}", path.display());
            println!();
            println!("Open it with File > Open. Two bars under one rail:");
            println!("  the left one is drawn up toward it, the right one pushed down away.");
            println!("Both read the same `reach`, so one number moves both.");
        }
        Err(e) => {
            eprintln!("could not write {}: {e}", path.display());
            std::process::exit(1);
        }
    }
}
