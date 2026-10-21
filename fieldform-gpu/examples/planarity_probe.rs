// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where does a profile bent out of its own plane get caught? In one of three places, and they
//! disagree.
//!
//! Asked because `draggable_numbers` reports "a buffer write" for a profile lifted out of plane,
//! and the profile-group session expected "refused" on the strength of `Profile::in_plane` in
//! `GraphField::compile`. Both are right about their own half, which is the finding. At a
//! compile tolerance of 0.1 mm:
//!
//! | lift mm | `generate` | `GraphField::compile` | `plane_of` |
//! |---|---|---|---|
//! | 0 | ok | ok | planar |
//! | 0.1 | ok | ok | planar |
//! | 0.5 | ok | **refused**, strays 0.125 mm | planar |
//! | 2 | ok | **refused**, strays 0.499 mm | planar |
//! | 40 | ok | **refused**, strays 7.276 mm | planar |
//!
//! - **The shader always generates**, at any lift, because a profile reaches it as an uploaded
//!   buffer. How the points sit is the caller's problem, not the generator's — which is why the
//!   `draggable_numbers` row is right and is not saying what it looks like it is saying.
//! - **The CPU field refuses** once the stray passes the tolerance it was compiled at. That is
//!   the path picking uses, so this is where a user meets it.
//! - **`plane_of` says planar at every lift, including 7 mm.** It fits a plane and only fails on
//!   degenerate input; the residual is `Profile::in_plane`'s question, not its. Worth knowing
//!   because a caller that checks planarity by asking `plane_of` — which reads like the obvious
//!   thing to ask — has not checked anything.
//!
//! The three columns disagreeing is the part to keep. A document can generate a shader and
//! refuse to compile a field, and those are the viewport and the picking.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example planarity_probe`

use fieldform_scene::{Curve, CurveId, Document, Graph, Node, ParamRef};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn document(lift: f64) -> (Document, Graph, CurveId) {
    let mut doc = Document::new();
    let mut g = Graph::empty();
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            "section",
            vec![
                Point3::new(0.0, -15.0, 0.0),
                Point3::new(30.0, -15.0, 0.0),
                Point3::new(30.0, 15.0, lift),
                Point3::new(0.0, 15.0, 0.0),
            ],
            true,
        )
        .expect("a quad"),
    );
    g.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(id),
        sweep: fieldform_scene::Sweep::Normal,
        height: lit(40.0),
        start: lit(0.0),
        sides: fieldform_scene::Sides::One,
        draft: lit(0.0),
        corners: Default::default(),
    });
    (doc, g, id)
}

fn main() {
    println!("{:<10} {:>12} {:>16} {:>14}", "lift mm", "generate", "GraphField", "plane_of");
    for lift in [0.0, 0.1, 0.5, 2.0, 40.0] {
        let (doc, graph, curve) = document(lift);

        let generated = fieldform_gpu::generate(&doc, &graph).map(|_| "ok").unwrap_or("refused");

        // What the CPU path says, which is what the application compiles for picking.
        let compiled = match fieldform_field::GraphField::compile_graph(&doc, &graph, 0.1) {
            Ok(_) => "ok".to_string(),
            Err(e) => format!("refused: {e}"),
        };

        // And what the caller has to do before it can hand the shader its profile buffer --
        // `curves_for` in the harnesses here, `traced/rebuild.rs` in the application.
        let plane = fieldform_field::profile_loop_turning(&doc, curve, 1.0, None)
            .map(|loop_| fieldform_field::plane_of(&loop_.points).is_some())
            .map(|planar| if planar { "planar" } else { "NOT planar" })
            .unwrap_or("no loop");

        println!("{lift:<10} {generated:>12} {compiled:>16} {plane:>14}");
    }
    println!(
        "\nThe shader's source never moves: a profile reaches it as an uploaded buffer, so how\n\
         the points sit is the caller's problem and not the generator's."
    );
}
