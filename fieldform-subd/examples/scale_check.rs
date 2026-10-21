// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Measures the cost of building and updating the limit-surface tessellation as cage size
//! grows, against the performance budget in the commercial-readiness design note.
//!
//! The number that matters is the **edit** column: it must stay flat as cage size grows,
//! because Catmull-Clark's support is local. Anything that turns it into a curve is a
//! regression, however fast the absolute numbers look on the machine of the day.
//!
//! Run with: `cargo run --release -p fieldform-subd --example scale_check`

use fieldform_cage::{Cage, VertexId};
use fieldform_subd::{faces_influenced_by, limit_cage, subdivide_n, LimitSurface};
use std::time::Instant;

/// Samples per patch edge. 4 gives a 4x4 grid per face, i.e. 18 triangles.
const N: usize = 4;

fn main() {
    println!(
        "{:>8}  {:>12}  {:>12}  {:>10}  {:>8}",
        "faces", "old rebuild", "new rebuild", "edit", "verdict"
    );
    println!("{}", "-".repeat(60));

    for k in 1..=6 {
        let cage = subdivide_n(&Cage::cube(), k).expect("valid");
        let faces = cage.face_count();

        // --- old path: subdivide the whole cage 4x, project to limit, remesh ---
        let old = if faces <= 2000 {
            let t = Instant::now();
            let dense = subdivide_n(&cage, 4).expect("valid");
            let _ = limit_cage(&dense).expect("valid");
            Some(t.elapsed())
        } else {
            None // too slow to be worth waiting for; the trend is already clear
        };

        // --- new path: evaluate an NxN patch grid per face ---
        let mut surface = LimitSurface::new(&cage);
        let mut buf = Vec::with_capacity(N * N);
        let t = Instant::now();
        for f in cage.face_ids() {
            buf.clear();
            surface.tessellate_face(f, N, &mut buf).expect("valid");
        }
        let new_full = t.elapsed();

        // --- the number that decides the use case: cost of moving one vertex ---
        let v = VertexId::from_index(cage.vertex_count() / 2);
        let influenced = faces_influenced_by(&cage, v);
        let t = Instant::now();
        for &f in &influenced {
            buf.clear();
            surface.tessellate_face(f, N, &mut buf).expect("valid");
        }
        let edit = t.elapsed();

        let edit_ms = edit.as_secs_f64() * 1000.0;
        let verdict = if edit_ms <= 16.0 { "60fps" } else { "MISS" };

        println!(
            "{:>8}  {:>12}  {:>10.1}ms  {:>8.3}ms  {:>8}",
            faces,
            match old {
                Some(d) => format!("{:.1}ms", d.as_secs_f64() * 1000.0),
                None => "(skipped)".to_string(),
            },
            new_full.as_secs_f64() * 1000.0,
            edit_ms,
            verdict
        );
    }

    println!();
    println!("edit = re-tessellating only the faces a moved vertex influences.");
    println!("It must stay flat as cage size grows; that is the whole claim.");
}
