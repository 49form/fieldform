// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A document measured against the CAD file it was built from, as two signed distance fields.
//!
//! The reference goes through the real importer and arrives as a `MeshField`, a true signed
//! distance, so the comparison is field to field rather than picture to picture. It answers three
//! things, because each hides what the others show:
//!
//! - **How far the reference's surface lies from ours**, read at points scattered over it. Our
//!   field's magnitude is only a lower bound in general, so the figure is called a reading.
//! - **How far our surface lies from the reference's**, read at points on ours pulled onto the
//!   zero set. Here the distance is the reference's own, so it is a true one. A feature we added
//!   shows only in this direction and a feature we lost only in the other.
//! - **The solid that differs**, by sign on a grid: a thin error over a large face barely moves a
//!   maximum and shows here.
//!
//! The worst places are listed apart from each other, which is what says *which feature* is off.
//! The importer has to be built first, and it is found beside this example's own build:
//!
//! `cargo build -p fieldform-import`
//! `cargo run -p fieldform-io --example compare_to_step -- <part.fform> <reference.stp> [map=x,-z,y] [cell=1.0] [points=200000] [linear=0.02] [compile=0.001]`
//!
//! `map` says where each of the reference's axes went, for a part modelled in other axes.
//! `compile` is the tolerance our part is compiled at; the default is fine enough to read every
//! section exactly rather than from its display grid.
//!
//! **It ends by saying pass or fail against two limits, because a number alone is not an answer.**
//! `tolerance=` is the widest band the part's own drawing allows, so missing it is a wrong part.
//! The targets are ours and far tighter: `worst=0.1`, `p99=0.05` in millimetres and `volume=0.1`
//! in percent, which say whether the kernel is faithful to the reference. They are proposed
//! figures rather than a published standard. Nothing below the importer's `linear` deflection
//! can be measured, since that is how far the reference's triangles sit from its true surface.

use fieldform_field::{GraphField, MeshField, ScalarField};
use nalgebra::{Point3, Vector3};
use std::path::{Path, PathBuf};

/// The order-keeping map over `count` items, on every core.
fn parallel<T: Send>(count: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk = count.div_ceil(threads).max(1);
    let work = &work;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    ((t * chunk).min(count)..((t + 1) * chunk).min(count)).map(work).collect::<Vec<T>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("a worker finished")).collect()
    })
}

/// A small generator, so a run can be repeated exactly and two runs compared.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Reads `map=x,-z,y` as which reference axis, and which sign, lands on each of ours.
fn parse_map(text: &str) -> [(usize, f64); 3] {
    let mut out = [(0, 1.0), (1, 1.0), (2, 1.0)];
    for (slot, word) in text.split(',').enumerate().take(3) {
        let (sign, axis) = word.strip_prefix('-').map_or((1.0, word), |rest| (-1.0, rest));
        let index = match axis {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            other => panic!("`map` takes x, y, z with an optional minus, not {other:?}"),
        };
        out[slot] = (index, sign);
    }
    out
}

/// The importer beside this example's build: `target/debug/examples/this` to `target/debug`.
fn importer() -> PathBuf {
    let here = std::env::current_exe().expect("this program has a path");
    let name = if cfg!(windows) { "fieldform-import.exe" } else { "fieldform-import" };
    let found = here.parent().and_then(Path::parent).map(|dir| dir.join(name));
    match found {
        Some(path) if path.exists() => path,
        _ => panic!("no importer beside this build: run `cargo build -p fieldform-import` first"),
    }
}

/// Worst first, each at least `apart` from the ones before it.
fn worst_apart(points: &[Point3<f64>], reads: &[f64], apart: f64, keep: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..reads.len()).collect();
    order.sort_by(|a, b| reads[*b].abs().total_cmp(&reads[*a].abs()));
    let mut out: Vec<usize> = Vec::new();
    for i in order {
        if out.len() == keep {
            break;
        }
        if out.iter().all(|j| (points[i] - points[*j]).norm() >= apart) {
            out.push(i);
        }
    }
    out
}

/// Prints the readings and answers `(p99, worst)` for the verdict.
fn report(title: &str, points: &[Point3<f64>], reads: &[f64]) -> (f64, f64) {
    let mut sorted: Vec<f64> = reads.iter().map(|r| r.abs()).collect();
    sorted.sort_by(f64::total_cmp);
    let at = |share: f64| sorted[((sorted.len() - 1) as f64 * share) as usize];
    println!("\n{title}, {} points", reads.len());
    println!(
        "  mean {:.4}  median {:.4}  p99 {:.4}  p99.9 {:.4}  worst {:.4} mm",
        sorted.iter().sum::<f64>() / sorted.len().max(1) as f64,
        at(0.5),
        at(0.99),
        at(0.999),
        at(1.0)
    );
    println!("  worst places, at least 5 mm apart:");
    for i in worst_apart(points, reads, 5.0, 8) {
        let p = points[i];
        println!("    {:+.3} mm at ({:.2}, {:.2}, {:.2})", reads[i], p.x, p.y, p.z);
    }
    (at(0.99), at(1.0))
}

fn main() {
    let mut paths = Vec::new();
    let (mut map, mut cell, mut count, mut linear) = (parse_map("x,y,z"), 1.0f64, 200_000usize, 0.02f64);
    let (mut worst_limit, mut p99_limit, mut volume_limit) = (0.1f64, 0.05f64, 0.1f64);
    let mut tolerance: Option<f64> = None;
    // **Fine enough that a section does not answer from a grid.** At 0.05 mm every profile under
    // about a hundred millimetres across is baked, and a bake reads a quarter of the tolerance
    // outside its outline on every wall, so a small extrude measured 0.0125 mm proud while a large
    // one beside it was exact. The comparison is of the part, not of a display convenience.
    let mut compile = 1e-3f64;
    for arg in std::env::args().skip(1) {
        match arg.split_once('=') {
            Some(("compile", v)) => compile = v.parse().expect("`compile` is a number"),
            Some(("map", v)) => map = parse_map(v),
            Some(("cell", v)) => cell = v.parse().expect("`cell` is a number"),
            Some(("points", v)) => count = v.parse().expect("`points` is a number"),
            Some(("linear", v)) => linear = v.parse().expect("`linear` is a number"),
            Some(("worst", v)) => worst_limit = v.parse().expect("`worst` is a number"),
            Some(("p99", v)) => p99_limit = v.parse().expect("`p99` is a number"),
            Some(("volume", v)) => volume_limit = v.parse().expect("`volume` is a number"),
            Some(("tolerance", v)) => tolerance = Some(v.parse().expect("`tolerance` is a number")),
            _ => paths.push(arg),
        }
    }
    let [part_path, step_path] = paths.as_slice() else {
        eprintln!("usage: compare_to_step <part.fform> <reference.stp> [map=x,-z,y] [cell=1.0] [points=200000] [linear=0.02] [compile=0.001]");
        std::process::exit(2);
    };

    // The reference, through the same process the application imports with.
    let out = std::env::temp_dir().join(format!("fieldform-compare-{}.mesh", std::process::id()));
    let started = std::time::Instant::now();
    let status = std::process::Command::new(importer())
        .arg(format!("in={step_path}"))
        .arg(format!("out={}", out.display()))
        .arg(format!("linear={linear}"))
        .status()
        .expect("the importer starts");
    assert!(status.success(), "the importer refused {step_path}: {status}");
    let mut file = std::fs::File::open(&out).expect("the importer left a mesh");
    let (meshes, _) = fieldform_io::mesh_wire::read_payload(&mut file).expect("a mesh this build can read");
    drop(file);
    let _ = std::fs::remove_file(&out);
    println!("imported {} solid(s) from {step_path} in {:.1} s", meshes.len(), started.elapsed().as_secs_f64());

    let mut positions: Vec<Point3<f64>> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for mesh in &meshes {
        let base = positions.len() as u32;
        positions.extend(mesh.positions.iter().map(|p| {
            Point3::new(p[map[0].0] * map[0].1, p[map[1].0] * map[1].1, p[map[2].0] * map[2].1)
        }));
        triangles.extend(mesh.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
    }
    // A reflection turns the winding over, and the sign of the field with it.
    let image = |axis: usize| {
        let mut v = Vector3::zeros();
        for (ours, (theirs, sign)) in map.iter().enumerate() {
            if *theirs == axis {
                v[ours] = *sign;
            }
        }
        v
    };
    if image(0).cross(&image(1)).dot(&image(2)) < 0.0 {
        for t in &mut triangles {
            t.swap(1, 2);
        }
    }
    let (mut lo, mut hi) = (Point3::from([f64::INFINITY; 3]), Point3::from([f64::NEG_INFINITY; 3]));
    for p in &positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let exact_volume: f64 = triangles
        .iter()
        .map(|t| {
            let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
            a.coords.dot(&b.coords.cross(&c.coords)) / 6.0
        })
        .sum();
    println!(
        "reference: {} triangles, volume {:.0} mm3, box {:.3} x {:.3} x {:.3} mm",
        triangles.len(),
        exact_volume,
        hi.x - lo.x,
        hi.y - lo.y,
        hi.z - lo.z
    );

    // Area-weighted, so a large face is not outvoted by the small ones its fillets make.
    let areas: Vec<f64> = triangles
        .iter()
        .scan(0.0, |sum, t| {
            let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
            *sum += (b - a).cross(&(c - a)).norm() * 0.5;
            Some(*sum)
        })
        .collect();
    let whole = *areas.last().expect("the reference has triangles");
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let mut on_reference: Vec<Point3<f64>> = positions.clone();
    on_reference.extend((0..count).map(|_| {
        let wanted = rng.unit() * whole;
        let i = areas.partition_point(|a| *a < wanted).min(triangles.len() - 1);
        let t = triangles[i];
        let (mut u, mut v) = (rng.unit(), rng.unit());
        if u + v > 1.0 {
            (u, v) = (1.0 - u, 1.0 - v);
        }
        let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
        a + (b - a) * u + (c - a) * v
    }));
    let reference = MeshField::from_mesh(positions, triangles).expect("the reference is a closed solid");

    let (doc, _archive) = fieldform_io::open(Path::new(part_path)).expect("the document opens");
    let (_, part) = doc.parts.iter().next().expect("the document has a part");
    let started = std::time::Instant::now();
    let ours = GraphField::compile(&doc, part, compile).expect("the part compiles");
    println!("compiled at {compile} mm in {:.2} s", started.elapsed().as_secs_f64());

    let from_reference = parallel(on_reference.len(), |i| ours.distance(on_reference[i]));
    let (p99_in, worst_in) = report("the reference's surface, read in our field", &on_reference, &from_reference);

    // Our surface, found by walking from a point near it, since a mesh of it would round every
    // edge by a cell and that error would be charged to the part.
    let margin = 3.0;
    let (from, to) = (lo - Vector3::repeat(margin), hi + Vector3::repeat(margin));
    let longest = (to - from).max();
    let resolution = (longest / 0.75).ceil() as usize;
    let surface = fieldform_field::mesh_field(&ours, (from, to), resolution);
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let step = (surface.positions.len() / count.max(1)).max(1);
    let seeds: Vec<Point3<f64>> = surface
        .positions
        .iter()
        .step_by(step)
        .map(|p| p + Vector3::new(rng.unit(), rng.unit(), rng.unit()) * 1e-3)
        .collect();
    let on_ours: Vec<Point3<f64>> = parallel(seeds.len(), |i| {
        let mut p = seeds[i];
        let h = 1e-4;
        for _ in 0..4 {
            let d = ours.distance(p);
            let g = Vector3::new(
                ours.distance(p + Vector3::x() * h) - ours.distance(p - Vector3::x() * h),
                ours.distance(p + Vector3::y() * h) - ours.distance(p - Vector3::y() * h),
                ours.distance(p + Vector3::z() * h) - ours.distance(p - Vector3::z() * h),
            ) / (2.0 * h);
            let n2 = g.norm_squared();
            if n2 < 1e-9 {
                break;
            }
            p -= g * (d / n2);
        }
        p
    });
    let from_ours = parallel(on_ours.len(), |i| reference.distance(on_ours[i]));
    let (p99_out, worst_out) = report("our surface, read in the reference's field (true distance)", &on_ours, &from_ours);

    // The solid that differs, by sign at the middle of each cell.
    let cells = [0, 1, 2].map(|k| ((to[k] - from[k]) / cell).ceil() as usize);
    let layers = parallel(cells[2], |z| {
        let (mut both, mut only_ours, mut only_theirs) = (0usize, 0usize, 0usize);
        for y in 0..cells[1] {
            for x in 0..cells[0] {
                let p = from + Vector3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5) * cell;
                match (ours.distance(p) < 0.0, reference.distance(p) < 0.0) {
                    (true, true) => both += 1,
                    (true, false) => only_ours += 1,
                    (false, true) => only_theirs += 1,
                    (false, false) => {}
                }
            }
        }
        (both, only_ours, only_theirs)
    });
    let v = cell * cell * cell;
    let (both, only_ours, only_theirs) =
        layers.iter().fold((0, 0, 0), |sum, l| (sum.0 + l.0, sum.1 + l.1, sum.2 + l.2));
    let (b, a, t) = (both as f64 * v, only_ours as f64 * v, only_theirs as f64 * v);
    println!("\nthe solids on a {cell} mm grid");
    println!("  ours {:.0}, the reference {:.0} mm3", b + a, b + t);
    let differs = 100.0 * (a + t) / (b + t).max(1.0);
    println!("  only ours {a:.0}, only the reference {t:.0} mm3: {differs:.3}% of the reference differs");

    // Both directions count: a lost feature shows in one and an added one only in the other.
    let (worst, p99) = (worst_in.max(worst_out), p99_in.max(p99_out));
    let verdict = |ok: bool| if ok { "pass" } else { "FAIL" };
    println!("
against the limits");
    if let Some(limit) = tolerance {
        println!("  {} drawing tolerance: worst {worst:.3} of {limit} mm", verdict(worst <= limit));
    }
    println!("  {} worst point: {worst:.3} of {worst_limit} mm", verdict(worst <= worst_limit));
    println!("  {} 99% of points: {p99:.3} of {p99_limit} mm", verdict(p99 <= p99_limit));
    println!("  {} volume: {differs:.3}% of {volume_limit}%", verdict(differs <= volume_limit));
}
