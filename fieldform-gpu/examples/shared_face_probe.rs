// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a document's joined mirrors are built as one body, and what the field reads at each
//! mirror's plane, inside the solid — the measurement behind backlog item 113.
//!
//! A joined mirror of a section touching its plane used to keep the face the two halves share
//! buried in the solid, reading nought where the true depth was the distance to the nearest real
//! face. `with_shared_faces_merged` builds such a mirror as one extrude of the merged section; this
//! says, for a real document, whether it did, and what the field says along a line crossing the
//! plane through the middle of each mirrored body.
//!
//! `cargo run --release -p fieldform-gpu --example shared_face_probe -- <doc.fform>`

use fieldform_field::ScalarField;
use fieldform_scene::Node;
use nalgebra::{Point3, Vector3};

fn main() {
    let path = std::env::args().nth(1).expect("a .fform path");
    let (doc, _archive) = fieldform_io::open(std::path::Path::new(&path)).expect("the document opens");
    for (part_id, part) in &doc.parts {
        let graph = &part.graph;
        let merged = fieldform_field::with_shared_faces_merged(&doc, graph);
        println!("part {part_id:?}: rewrite {}", if merged.is_some() { "APPLIES" } else { "does not apply" });
        let field = fieldform_field::GraphField::compile_graph(&doc, graph, 1e-4).expect("compiles");
        for (id, node) in graph.nodes() {
            let Node::Mirror { input, join, .. } = node else { continue };
            let Some(input) = input.source() else { continue };
            let extent = fieldform_field::graph_extent(&doc, graph, input).as_box();
            println!("  mirror {id:?} join {join}, input box {extent:?}");
            if let Some((lo, hi)) = extent {
                let mid = nalgebra::center(&lo, &hi);
                // Across y = 0, which is where this probe's documents put their mirror plane.
                let row: Vec<String> = [-2.0, -0.5, 0.0, 0.5, 2.0]
                    .iter()
                    .map(|dy| {
                        let q = Point3::new(mid.x, *dy, mid.z) + Vector3::zeros();
                        format!("{:+.3}", field.distance(q))
                    })
                    .collect();
                println!("    field across the plane at x {:.1}, z {:.1}: {}", mid.x, mid.z, row.join(" "));
            }
        }
    }
}
