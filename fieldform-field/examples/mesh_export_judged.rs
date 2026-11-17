// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether what the exporter writes is the part: closed, outward, and within its chord of the
//! field, for shapes whose volume is known on paper.
//!
//! Each shape is tessellated at each detail preset, three ways -- as meshed, with the shape pass
//! the export runs by default, and reduced as well -- and judged with `judge_mesh`. The columns are
//! the two one-sided distances as a multiple of the chord asked for (worst, and the 99th
//! percentile), and the mesh's volume against the closed form beside the most its distance allows.
//!
//! `cargo run --release -p fieldform-field --example mesh_export_judged [shape]`

use std::f64::consts::PI;
use std::time::Instant;

use fieldform_field::{judge_mesh, tessellate, Cancel, Progress, ScalarField};
use fieldform_scene::MeshDetail;
use nalgebra::{Point3, Vector3};

struct Sdf<F: Fn(Point3<f64>) -> f64 + Send + Sync>(F);

impl<F: Fn(Point3<f64>) -> f64 + Send + Sync> ScalarField for Sdf<F> {
    fn distance(&self, p: Point3<f64>) -> f64 {
        (self.0)(p)
    }
}

struct Case {
    name: &'static str,
    field: Box<dyn ScalarField + Sync>,
    half: Vector3<f64>,
    volume: f64,
}

fn cases() -> Vec<Case> {
    let sphere = |c: Point3<f64>, r: f64| move |p: Point3<f64>| (p - c).norm() - r;
    let (ra, rb) = (sphere(Point3::new(-5.0, 0.0, 0.0), 10.0), sphere(Point3::new(5.0, 0.0, 0.0), 10.0));
    let half_box = Vector3::new(10.0, 15.0, 20.0);
    let (a, b, c, d) = (20.0, 30.0, 40.0, 3.0);
    let cuboid = move |p: Point3<f64>| {
        let q = p.coords.abs() - half_box;
        q.sup(&Vector3::zeros()).norm() + q.max().min(0.0)
    };
    vec![
        Case {
            name: "ball r 10",
            field: Box::new(Sdf(sphere(Point3::origin(), 10.0))),
            half: Vector3::repeat(10.0),
            volume: 4.0 / 3.0 * PI * 1000.0,
        },
        Case {
            name: "box 20x30x40",
            field: Box::new(Sdf(cuboid)),
            half: half_box,
            volume: a * b * c,
        },
        Case {
            name: "box grown 3",
            field: Box::new(Sdf(move |p| cuboid(p) - d)),
            half: half_box.add_scalar(d),
            volume: a * b * c + 2.0 * d * (a * b + b * c + c * a) + PI * d * d * (a + b + c) + 4.0 / 3.0 * PI * d.powi(3),
        },
        Case {
            name: "lens",
            field: Box::new(Sdf(move |p| ra(p).max(rb(p)))),
            half: Vector3::new(5.0, 8.7, 8.7),
            volume: PI * 50.0 * 100.0 / 12.0,
        },
        Case {
            name: "bicylinder r 10",
            field: Box::new(Sdf(|p: Point3<f64>| p.y.hypot(p.z).max(p.x.hypot(p.z)) - 10.0)),
            half: Vector3::repeat(10.0),
            volume: 16.0 * 1000.0 / 3.0,
        },
        Case {
            name: "torus 30 / 5",
            field: Box::new(Sdf(|p: Point3<f64>| (p.x.hypot(p.y) - 30.0).hypot(p.z) - 5.0)),
            half: Vector3::new(35.0, 35.0, 5.0),
            volume: 2.0 * PI * PI * 30.0 * 25.0,
        },
    ]
}

fn main() {
    let only = std::env::args().nth(1);
    println!(
        "{:<16} {:<9} {:<8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:>9} {:>9} {:>6}  {}",
        "shape", "detail", "passes", "tris", "chord", "off", "off99", "unc", "unc99", "vol err", "slack", "time", "faults"
    );
    for case in cases() {
        if only.as_deref().is_some_and(|o| !case.name.contains(o)) {
            continue;
        }
        let bounds = (Point3::from(-case.half), Point3::from(case.half));
        let longest = 2.0 * case.half.max();
        for detail in [MeshDetail::Coarse, MeshDetail::Normal, MeshDetail::Full] {
            let tolerance = detail.tolerance(longest);
            for (passes, improve, reduce) in [("as meshed", false, false), ("shaped", true, false), ("reduced", true, true)] {
                let started = Instant::now();
                let Some(done) = tessellate(
                    case.field.as_ref(),
                    bounds,
                    &tolerance,
                    improve,
                    reduce,
                    &Cancel::new(),
                    &Progress::default(),
                ) else {
                    continue;
                };
                let meshing = started.elapsed().as_secs_f64();
                let mesh = done.mesh;
                let j = judge_mesh(case.field.as_ref(), bounds, &mesh.positions, &mesh.triangles, tolerance.chord_mm);
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
                if j.shells != 1 {
                    faults.push(format!("{} shells", j.shells));
                }
                println!(
                    "{:<16} {:<9} {:<8} {:>8} {:>8.4} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>+8.4}% {:>8.4}% {:>5.1}s  {}",
                    case.name,
                    detail.label(),
                    passes,
                    j.triangles,
                    chord,
                    j.off_surface.worst / chord,
                    j.off_surface.p99 / chord,
                    j.uncovered.worst / chord,
                    j.uncovered.p99 / chord,
                    (j.volume_mm3 - case.volume) / case.volume * 100.0,
                    j.volume_slack() / case.volume * 100.0,
                    meshing,
                    faults.join(", "),
                );
                if std::env::var_os("WHERE").is_some() {
                    println!("    off worst at {:?}, uncovered worst at {:?}, facing in at {:?}", j.off_surface.at, j.uncovered.at, j.facing_in_at);
                }
            }
        }
        println!();
    }
}
