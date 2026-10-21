// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Times a real OBJ import end to end, at cage sizes up to ~100k faces.
//!
//! This exists because the roadmap's import figure ("~9 s at 100k faces") was an
//! extrapolation from a single 24,576-face datum scaled by 4 — it assumed the cost is
//! linear in face count, which is exactly the assumption worth checking before optimising
//! anything. So this generates cages at three sizes, writes each to a real OBJ file, and
//! times the stages `App::import_obj` actually runs.
//!
//! It measures the CPU stages only. GPU buffer creation needs a device and a surface, so
//! it is not covered here; that is a separate frame-time measurement.
//!
//! It used to time the limit-surface tessellation too, which was most of the cost. That mesh
//! is gone — the viewport traces the field instead of meshing it — so what import runs
//! synchronously is now the parse and the control-cage triangulation, and this measures those.
//! The traced view's cage bake is the cost that replaced it, and it is *not* measured here:
//! it happens on entering field mode rather than at import, and `bake_normal_accuracy` in
//! fieldform-gpu is where its numbers live.
//!
//! The budget from the commercial-readiness design note is **under 2 s** for a 100k
//! face import.
//!
//! Run with: `cargo run --release -p fieldform-app --example import_check`

// The real display/picking mesh, compiled from the app's own source rather than copied, so
// this measures the code that ships. `dead_code` because the example uses a few of its
// entry points, not all.
#[allow(dead_code)]
#[path = "../src/cage_mesh.rs"]
mod cage_mesh;

use cage_mesh::CageMesh;
use fieldform_cage::Cage;
use fieldform_subd::subdivide_n;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("Import cost, budget 2000 ms\n");
    println!(
        "{:>8}  {:>9}  {:>9}  {:>11}  {:>9}  {:>8}",
        "faces", "obj MB", "parse", "cage mesh", "total", "verdict"
    );
    println!("{}", "-".repeat(64));

    let mut rows: Vec<(usize, f64)> = Vec::new();

    for k in 5..=7 {
        let cage = subdivide_n(&Cage::cube(), k).expect("cube subdivides");
        let faces = cage.face_count();

        let path = std::env::temp_dir().join(format!("fieldform_import_check_{faces}.obj"));
        let bytes = write_obj(&cage, &path);
        drop(cage); // import must not benefit from the in-memory original

        // --- stage 1: parse the file and build the half-edge cage (fieldform-io) ---
        let t = Instant::now();
        let mut cages = fieldform_io::import_cages(&path).expect("generated OBJ is a valid cage");
        let parse = t.elapsed();
        let cage = cages.remove(0);
        assert_eq!(cage.face_count(), faces, "round-tripped cage lost faces");

        // --- stages 2-4: what `rebuild_render_resources` does, minus the GPU upload ---
        let t = Instant::now();
        let _cage_mesh = CageMesh::from_cage(&cage);
        let cage_mesh_time = t.elapsed();

        let total = parse + cage_mesh_time;
        let total_ms = total.as_secs_f64() * 1000.0;
        rows.push((faces, total_ms));

        println!(
            "{:>8}  {:>9.1}  {:>7.0}ms  {:>9.0}ms  {:>7.0}ms  {:>8}",
            faces,
            bytes as f64 / (1024.0 * 1024.0),
            parse.as_secs_f64() * 1000.0,
            cage_mesh_time.as_secs_f64() * 1000.0,
            total_ms,
            if total_ms <= 2000.0 { "under" } else { "OVER" },
        );

        let _ = std::fs::remove_file(&path);
    }

    // The question this example exists to answer: does cost actually scale linearly?
    println!();
    for pair in rows.windows(2) {
        let (f0, t0) = pair[0];
        let (f1, t1) = pair[1];
        println!(
            "{f0} -> {f1} faces: {:.1}x the faces, {:.1}x the time",
            f1 as f64 / f0 as f64,
            t1 / t0
        );
    }
    println!();
    println!("GPU buffer creation is not included — it needs a device, and is a separate");
    println!("frame-time measurement. Neither is the traced view's cage bake, which is paid on");
    println!("entering field mode rather than at import: see fieldform-gpu's bake examples.");
}

/// Writes a cage as a quad OBJ, so the timed import parses a real file rather than a
/// synthetic in-memory buffer. Not timed: authoring the file is not part of import.
fn write_obj(cage: &Cage, path: &Path) -> u64 {
    let file = std::fs::File::create(path).expect("temp dir is writable");
    let mut w = BufWriter::new(file);
    for v in cage.vertices() {
        writeln!(w, "v {} {} {}", v.position.x, v.position.y, v.position.z).expect("write");
    }
    for f in cage.face_ids() {
        let c = cage.face_vertices(f);
        writeln!(
            w,
            "f {} {} {} {}",
            c[0].index() + 1,
            c[1].index() + 1,
            c[2].index() + 1,
            c[3].index() + 1
        )
        .expect("write");
    }
    w.flush().expect("flush");
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}
