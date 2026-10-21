// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What an OBJ import says about a mesh it cannot make a cage of.
//!
//! **The refusal moved in item 45** — from inside the half-edge constructor to the `Cage`
//! newtype — and it is load-bearing: an open surface has no inside for a sign to refer to, so
//! `CageField` depends on it. A refusal that became generic, or stopped naming the edge, would
//! be a regression nothing else would catch.
//!
//! Run with: `cargo run -p fieldform-io --example obj_refusal -- <file.obj> [...]`

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: obj_refusal <file.obj> [...]");
        return;
    }
    for file in files {
        let path = std::path::Path::new(&file);
        match fieldform_io::import_cages(path) {
            Ok(cages) => {
                println!("{file}");
                for (i, cage) in cages.iter().enumerate() {
                    println!(
                        "  cage {i}: {} vertices, {} edges, {} faces, euler {}",
                        cage.vertex_count(),
                        cage.edge_count(),
                        cage.face_count(),
                        cage.euler_characteristic(),
                    );
                }
            }
            Err(e) => println!("{file}\n  refused: {e}"),
        }
    }
}
