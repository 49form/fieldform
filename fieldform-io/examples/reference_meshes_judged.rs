// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the exporter makes of real documents, judged against their fields.
//!
//! The application's own path, step for step: the part's graph with its reference bodies left
//! out, the box its bodies reach, the detail preset's tolerances for that box, the field compiled
//! at half the floor, and the shape pass on. Each mesh is then judged by `judge_mesh`, and its
//! volume compared with the field's own anchored measurement.
//!
//! `cargo run --release -p fieldform-io --example reference_meshes_judged -- <file.fform> ...`
//!
//! `DETAILS=Coarse,Normal` keeps only the presets named; `WHERE=1` prints where each worst
//! reading was; `CLUSTERS=1` groups the vertices more than three chords off by 10 mm box and
//! lists the mesh's connected pieces with their volumes; `COMPILE=<k>` compiles the field at `k`
//! chords instead of the application's half floor, which is how the outward bias of a baked
//! profile was told apart from the mesher's own error.

use std::time::Instant;

use fieldform_field::{judge_mesh, tessellate, Anchored, Cancel, GraphField, MassProperties, Progress};
use fieldform_scene::MeshDetail;
use nalgebra::Point3;

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: reference_meshes_judged <file.fform> ...");
        std::process::exit(2);
    }
    println!(
        "{:<14} {:<7} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:>9} {:>8} {:>6}  {}",
        "file", "detail", "tris", "chord", "off", "off99", "unc", "unc99", "vol diff", "slack", "time", "faults"
    );
    for path in files {
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        let (doc, _archive) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
        for part in doc.parts.values() {
            let graph = &part.graph;
            let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut hi = -lo;
            for r in graph.roots() {
                if let Some((a, b)) = fieldform_field::graph_extent(&doc, graph, r).as_box() {
                    lo = lo.inf(&a);
                    hi = hi.sup(&b);
                }
            }
            let e = hi - lo;
            let longest = e.x.max(e.y).max(e.z);
            if !longest.is_finite() {
                continue;
            }
            let measured = {
                let field = GraphField::compile_graph(&doc, graph, 1e-4).expect("compiles");
                let mut h = (longest / 512.0).max(1.0 / 64.0);
                loop {
                    match MassProperties::anchored(&field, (lo, hi), h) {
                        Anchored::Measured(m) => break m.props.volume_mm3,
                        Anchored::TooFine { .. } => h *= 2.0,
                        Anchored::Empty => break f64::NAN,
                    }
                }
            };
            let only = std::env::var("DETAILS").unwrap_or_default();
            for detail in [MeshDetail::Coarse, MeshDetail::Normal, MeshDetail::Full] {
                if !only.is_empty() && !only.contains(detail.label()) {
                    continue;
                }
                let tolerance = detail.tolerance(longest);
                let compile = match std::env::var("COMPILE").as_deref() {
                    Ok(f) if f.parse::<f64>().is_ok() => tolerance.chord_mm * f.parse::<f64>().unwrap(),
                    _ => (tolerance.min_cell_mm * 0.5).max(1e-4),
                };
                let field = GraphField::compile_graph(&doc, graph, compile).expect("compiles");
                let started = Instant::now();
                let Some(done) =
                    tessellate(&field, (lo, hi), &tolerance, true, false, &Cancel::new(), &Progress::default())
                else {
                    continue;
                };
                let took = started.elapsed().as_secs_f64();
                let mesh = done.mesh;
                let grown = nalgebra::Vector3::repeat(tolerance.min_cell_mm * 4.0);
                let j = judge_mesh(&field, (lo - grown, hi + grown), &mesh.positions, &mesh.triangles, tolerance.chord_mm);
                let chord = tolerance.chord_mm;
                let mut faults = Vec::new();
                for (n, what) in [
                    (j.open_edges, "open"),
                    (j.overloaded_edges, "overloaded"),
                    (j.pinched_vertices, "pinched"),
                    (j.degenerate, "degenerate"),
                    (j.misoriented_edges, "misoriented"),
                    (j.inward_shells, "inward shells"),
                    (j.facing_in, "facing in"),
                    (j.off_surface.unsettled + j.uncovered.unsettled, "unsettled"),
                ] {
                    if n > 0 {
                        faults.push(format!("{n} {what}"));
                    }
                }
                println!(
                    "{:<14} {:<7} {:>8} {:>8.4} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>+8.3}% {:>7.3}% {:>5.1}s  {}",
                    name,
                    detail.label(),
                    j.triangles,
                    chord,
                    j.off_surface.worst / chord,
                    j.off_surface.p99 / chord,
                    j.uncovered.worst / chord,
                    j.uncovered.p99 / chord,
                    (j.volume_mm3 - measured) / measured * 100.0,
                    j.volume_slack() / measured * 100.0,
                    took,
                    faults.join(", "),
                );
                if std::env::var_os("WHERE").is_some() {
                    println!("    off at {:?}, uncovered at {:?}, facing in at {:?}", j.off_surface.at, j.uncovered.at, j.facing_in_at);
                }
                if std::env::var_os("CLUSTERS").is_some() {
                    // Where the vertices more than three chords off the field gather, by 10 mm
                    // box, with which side of the surface they sit on.
                    let mut boxes: std::collections::HashMap<[i64; 3], (usize, usize, f64)> = Default::default();
                    for p in &mesh.positions {
                        let p = *p;
                        let d = fieldform_field::ScalarField::distance(&field, p);
                        if d.abs() > 3.0 * chord {
                            let k = [(p.x / 10.0).floor() as i64, (p.y / 10.0).floor() as i64, (p.z / 10.0).floor() as i64];
                            let e = boxes.entry(k).or_default();
                            if d < 0.0 { e.0 += 1 } else { e.1 += 1 }
                            e.2 = e.2.max(d.abs());
                        }
                    }
                    let mut boxes: Vec<_> = boxes.into_iter().collect();
                    boxes.sort_by_key(|(_, (a, b, _))| std::cmp::Reverse(a + b));
                    for (k, (inside, outside, worst)) in boxes.iter().take(12) {
                        println!("    box {:?}0 mm: {inside} in, {outside} out, worst {worst:.3}", k);
                    }
                    // Each connected piece's signed volume and box.
                    let mut parent: Vec<usize> = (0..mesh.positions.len()).collect();
                    fn find(parent: &mut [usize], mut i: usize) -> usize {
                        while parent[i] != i {
                            parent[i] = parent[parent[i]];
                            i = parent[i];
                        }
                        i
                    }
                    for t in &mesh.triangles {
                        for k in 1..3 {
                            let (a, b) = (find(&mut parent, t[0] as usize), find(&mut parent, t[k] as usize));
                            parent[a] = b;
                        }
                    }
                    let mut pieces: std::collections::HashMap<usize, (f64, usize, Point3<f64>, Point3<f64>)> = Default::default();
                    for t in &mesh.triangles {
                        let r = find(&mut parent, t[0] as usize);
                        let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
                        let e = pieces.entry(r).or_insert((0.0, 0, a, a));
                        e.0 += a.coords.dot(&b.coords.cross(&c.coords)) / 6.0;
                        e.1 += 1;
                        for p in [a, b, c] {
                            e.2 = e.2.inf(&p);
                            e.3 = e.3.sup(&p);
                        }
                    }
                    let mut pieces: Vec<_> = pieces.into_values().collect();
                    pieces.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
                    for (v, n, lo, hi) in pieces.iter().take(8) {
                        println!("    piece {v:.1} mm3, {n} tris, {:?} to {:?}", lo.coords.as_slice(), hi.coords.as_slice());
                    }
                    println!("    field volume {measured:.1} mm3");
                }
            }
        }
    }
}
