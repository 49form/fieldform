// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What bounding a facet's *angle* costs, against what leaving it to OCCT's default gives.
//!
//! # Why this is not the chord height
//!
//! Chord height bounds how far a facet sits from the surface it stands in for. Faceting is a
//! *shading* artefact, and shading is the normal — so the quantity that decides whether a
//! curved face looks smooth is how far the normal turns from one facet to the next.
//!
//! The two are related by the radius: a facet subtending angle t on radius r sits
//! `r(1 - cos(t/2))` below the arc, so the same chord height allows a much larger angle on a
//! small radius. A 0.05 mm chord on a 50 mm radius turns 5.1 degrees; on a 5 mm fillet it turns
//! 16. This is the same mistake the pipe spine made, recorded in `spine_angle.rs`: tessellating
//! to a distance while the visible artefact is an angle.
//!
//! `BRepMesh_IncrementalMesh(shape, deflection)` — the form this crate used until now — leaves
//! the angular deflection at OCCT's default of **0.5 rad, about 28 degrees**.
//!
//! Run against a real file:
//! `cargo run --release -p fieldform-io --features step --example angular_sweep -- part.stp`

use opencascade::{mesh::Mesher, primitives::Shape};

/// The distribution of angles between the normals of triangles sharing an edge, in degrees:
/// the median, the 95th and the 99th percentile.
///
/// **Not the maximum.** A solid has genuinely sharp edges — a chamfer, a boss against a face —
/// and those are 90 degrees whatever the tessellation does, so the largest crease is measuring
/// the part rather than the mesh. It also moves between settings, because different
/// tessellations put their worst sliver in different places: the first version of this reported
/// a *worse* maximum for a finer setting, which is how the metric gave itself away.
///
/// Percentiles separate the two. Genuine edges are a small tail; tessellation creases are the
/// bulk, so the median and the 95th are what a curved face actually looks like.
///
/// This is what a viewer sees as a crease. Measured over the mesh rather than derived from the
/// settings, because the settings are what was *asked for* and this is what arrived.
fn dihedrals(vertices: &[[f64; 3]], indices: &[usize]) -> (f64, f64, f64) {
    use std::collections::HashMap;
    let normal = |t: &[usize]| {
        let (a, b, c) = (vertices[t[0]], vertices[t[1]], vertices[t[2]]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        (len > 1e-12).then(|| [n[0] / len, n[1] / len, n[2] / len])
    };

    let mut by_edge: HashMap<(usize, usize), Vec<[f64; 3]>> = HashMap::new();
    for t in indices.chunks_exact(3) {
        let Some(n) = normal(t) else { continue };
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            by_edge.entry((a.min(b), a.max(b))).or_default().push(n);
        }
    }
    let mut angles: Vec<f64> = Vec::new();
    for normals in by_edge.values() {
        if normals.len() != 2 {
            continue;
        }
        let (a, b) = (normals[0], normals[1]);
        let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
        angles.push(dot.acos().to_degrees());
    }
    if angles.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    angles.sort_by(f64::total_cmp);
    let at = |q: f64| angles[((angles.len() - 1) as f64 * q) as usize];
    (at(0.5), at(0.95), at(0.99))
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: angular_sweep <file.stp>");
        std::process::exit(2);
    };

    println!("{path}\n");
    println!(
        "{:<10} {:<10} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "chord mm", "angle", "triangles", "median", "95th", "99th", "time"
    );

    for chord in [0.1, 0.05, 0.01] {
        // `None` is the two-argument constructor: OCCT's own default angular deflection, which
        // is what every import before this measurement used.
        for angle in [None, Some(0.2), Some(0.1)] {
            // Re-read for every row. `BRepMesh_IncrementalMesh` stores its triangulation
            // *on the shape* and skips any face already meshed finely enough, so meshing one
            // shape repeatedly measures the first call and then nothing. The first version of
            // this sweep did exactly that and reported identical triangle counts for every
            // setting — the giveaway was the time, 0.47 s for the first row and 0.02 s for
            // every row after it.
            let Ok(shape) = Shape::read_step(&path) else {
                eprintln!("could not read {path}");
                std::process::exit(1);
            };
            let started = std::time::Instant::now();
            let mesher = match angle {
                None => Mesher::try_new(&shape, chord),
                // Threaded: a shape meshes face by face and the faces are independent.
                Some(a) => Mesher::try_new_angular(&shape, chord, a, true),
            };
            let Ok(mesher) = mesher else {
                println!("{chord:<10} {angle:?} — would not mesh");
                continue;
            };
            let Ok(mesh) = mesher.mesh() else {
                println!("{chord:<10} {angle:?} — would not mesh");
                continue;
            };
            let secs = started.elapsed().as_secs_f64();

            let vertices: Vec<[f64; 3]> =
                mesh.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
            let (median, p95, p99) = dihedrals(&vertices, &mesh.indices);
            let label = match angle {
                None => "default".to_string(),
                Some(a) => format!("{:.1}deg", a.to_degrees()),
            };
            println!(
                "{chord:<10} {label:<10} {:>10} {median:>6.1}deg {p95:>5.1}deg {p99:>5.1}deg                  {secs:>6.2}s",
                mesh.indices.len() / 3,
            );
        }
        println!();
    }
}
