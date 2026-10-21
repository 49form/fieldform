// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Runs the STEP importer over a real file at every healing strength, and reports what happened.
//!
//! A status line has room for a sentence; a file that will not come in needs a page. This is
//! what turned "the import errors" into separate findings per file, and it is the first thing
//! to reach for when a part arrives open that ought to be a solid.
//!
//! Run with:
//! `cargo run --release -p fieldform-io --example step_diagnose -- <file.stp> [chord_mm]`

use std::collections::HashMap;

use fieldform_scene::MeshData;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: step_diagnose <file.stp> [chord_mm]");
        std::process::exit(2);
    };
    let chord: f64 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or_else(|| fieldform_io::StepOptions::default().linear_deflection_mm);

    let meta = std::fs::metadata(&path).expect("the file exists");
    println!("{path}\n  {:.1} MB", meta.len() as f64 / 1e6);

    // What the file says it is. The application protocols differ in what they carry, and the
    // header is the cheapest thing to check when a reader takes one file and refuses another.
    if let Ok(text) = std::fs::read_to_string(&path) {
        for line in text.lines().take(60) {
            let line = line.trim();
            if line.starts_with("FILE_SCHEMA") {
                println!("  {line}");
            }
        }
    }

    // Every preset, because which one a file needs is exactly the question, and the answer
    // differs per exporter rather than following from anything about the part.
    for (name, preset) in fieldform_io::StepOptions::presets() {
        // The linear deflection from the command line, so the presets are compared at one
        // detail level rather than at four.
        let preset = fieldform_io::StepOptions { linear_deflection_mm: chord, ..preset };
        println!("\n--- {name} | {} ---", preset.summary());
        let started = std::time::Instant::now();
        match fieldform_io::import_step_with(path.as_ref(), "part", preset) {
            Ok((meshes, curves, report)) => {
                let closed = meshes.iter().filter(|m| m.is_closed()).count();
                let triangles: usize = meshes.iter().map(MeshData::triangle_count).sum();
                let solids = if report.solids_found == 0 {
                    "no solids in the file (surface model)".to_owned()
                } else {
                    format!("{} of {} solids meshed", report.solids_converted, report.solids_found)
                };
                // The curves that bound no face, which a diagnosis of "no solids in the file"
                // is incomplete without: a sketch exported as STEP is all curves and no solids,
                // and reporting only the absence reads as a failure rather than as the shape.
                let loose = match curves.len() {
                    0 => String::new(),
                    n => format!(" | {n} of {} curves", report.curves_found),
                };
                println!(
                    "  {:.1}s | {solids}{loose} | {closed} of {} closed | {triangles} triangles",
                    started.elapsed().as_secs_f64(),
                    meshes.len(),
                );
                if !report.unconverted.is_empty() {
                    println!("  unconverted: {:?}", report.unconverted);
                }
                // The largest few, since an assembly's interesting failures are usually its
                // big parts and printing four hundred rows helps nobody.
                let mut by_size: Vec<&MeshData> = meshes.iter().collect();
                by_size.sort_by_key(|m| std::cmp::Reverse(m.triangle_count()));
                for mesh in by_size.iter().take(4) {
                    describe(mesh);
                }
            }
            Err(e) => println!("  REFUSED: {e}"),
        }
    }
}

/// One mesh's shape and, more importantly, why it is or is not closed.
fn describe(mesh: &MeshData) {
    // Holes and seams mean different things and have opposite remedies: a hole is a face that
    // is missing or does not meet its neighbour, and healing closes it; a seam is geometry
    // that was never one solid, and only splitting separates it.
    let mut uses: HashMap<(u32, u32), usize> = HashMap::new();
    for t in &mesh.triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let once = uses.values().filter(|c| **c == 1).count();
    let many = uses.values().filter(|c| **c > 2).count();

    let verdict = if once == 0 && many == 0 {
        "CLOSED".to_owned()
    } else if many > once {
        format!("open, mostly seams ({many} shared by 3+)")
    } else {
        format!("open ({once} edges)")
    };
    println!("    {}: {} triangles, {verdict}", mesh.name, mesh.triangle_count());
}
