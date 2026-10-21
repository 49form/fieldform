// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Opens a `.fieldform` document and says whether its graph compiles, body by body.
//!
//! Written to answer a bug report that looked like two bugs -- a document opening to a blank
//! viewport, and File > Export showing no dialog -- and is one: a graph that will not compile
//! draws nothing and exports nothing, and both paths report it somewhere easy to miss.
//!
//! `cargo run -p fieldform-io --example open_and_compile -- <path>`

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: open_and_compile <file.fieldform>");
        std::process::exit(2);
    };

    let (doc, _archive) = match fieldform_io::open(std::path::Path::new(&path)) {
        Ok(v) => v,
        Err(e) => {
            println!("could not open: {e}");
            return;
        }
    };

    println!("parts: {}  cages: {}  meshes: {}", doc.parts.len(), doc.cages.len(), doc.meshes.len());
    for (id, part) in &doc.parts {
        let roots = part.graph.roots();
        println!("\npart {id:?} \"{}\" -- {} nodes, {} roots", part.name, part.graph.len(), roots.len());
        for r in &roots {
            let kind = part.graph.node(*r).map(|n| format!("{n:?}").split_whitespace().next().unwrap_or("?").to_string()).unwrap_or_default();
            print!("  root {r:?} ({kind}): ");
            match fieldform_field::GraphField::compile_graph(&doc, &part.graph, 1e-3) {
                Ok(_) => print!("compiles"),
                Err(e) => print!("REFUSED -- {e}"),
            }
            // And what the app derives from it: the extent is what sizes the camera, the
            // tracer's reach and the export box, so an unbounded answer starves all three.
            // The shader generator, which is what the viewport actually draws through. A
            // graph the kernel accepts and this refuses is a document you can export and
            // cannot see -- the mirror image of the bug that started this.
            match fieldform_gpu::generate(&doc, &part.graph) {
                Ok(p) => print!("; shader {} bytes", p.source.len()),
                Err(e) => print!("; SHADER REFUSED -- {e}"),
            }
            let extent = fieldform_field::graph_extent(&doc, &part.graph, *r);
            match extent.as_box() {
                Some((lo, hi)) => println!(
                    "; extent [{:.1} {:.1} {:.1}] .. [{:.1} {:.1} {:.1}]",
                    lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
                ),
                None => println!("; extent UNBOUNDED"),
            }
        }
    }
}
