// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Every STEP file in a folder, through the real importer and into a signed distance field.
//!
//! `compare_to_step` asks whether one part matches its reference. This asks the question that
//! comes first and needs no reference at all: does a real file arrive, closed, the right way out,
//! and does the field built from it agree about what is inside? Each file reports:
//!
//! - **Whether it imported, and how long it took**, under a limit, in the importer's own process,
//!   so a file that hangs or crashes it is one line of the table and not the end of the run.
//! - **How many solids were closed**, since an open mesh cannot be a field at all.
//! - **The volume by the divergence theorem**, which is what lets several exports of one part be
//!   held against each other when nothing else says what the answer is.
//! - **Sign errors**: random points in each solid's box, the field's sign against an odd or even
//!   count of crossings along a ray. The two share no code, so a wrong winding, a leak or a bad
//!   pseudonormal cannot pass both. Nought is the only acceptable count.
//!
//! `cargo build -p fieldform-import`
//! `cargo run -p fieldform-io --example step_corpus -- <folder or file>... out=report.csv [limit=300] [points=300] [linear=mm]`

use fieldform_field::MeshField;
use nalgebra::{Point3, Vector3};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The most solids tested per file, largest first: an assembly can hold thousands, and the
/// hundredth washer says little the first twenty did not.
const MOST_SOLIDS: usize = 20;

/// The importer beside this example's build.
fn importer() -> PathBuf {
    let here = std::env::current_exe().expect("this program has a path");
    let name = if cfg!(windows) { "fieldform-import.exe" } else { "fieldform-import" };
    let found = here.parent().and_then(Path::parent).map(|dir| dir.join(name));
    match found {
        Some(path) if path.exists() => path,
        _ => panic!("no importer beside this build: run `cargo build -p fieldform-import` first"),
    }
}

/// A small generator, so a failure can be run again exactly.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How many triangles the ray from `p` along `dir` crosses.
fn crossings(p: Point3<f64>, dir: Vector3<f64>, positions: &[Point3<f64>], triangles: &[[u32; 3]]) -> usize {
    triangles
        .iter()
        .filter(|t| {
            let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
            let (e1, e2) = (b - a, c - a);
            let h = dir.cross(&e2);
            let det = e1.dot(&h);
            if det.abs() < 1e-14 {
                return false;
            }
            let s = p - a;
            let u = s.dot(&h) / det;
            if !(0.0..=1.0).contains(&u) {
                return false;
            }
            let q = s.cross(&e1);
            let v = dir.dot(&q) / det;
            v >= 0.0 && u + v <= 1.0 && e2.dot(&q) / det > 0.0
        })
        .count()
}

/// Points in the box where the field and the ray count disagree, across every core.
fn sign_errors(field: &MeshField, positions: &[Point3<f64>], triangles: &[[u32; 3]], points: usize, seed: u64) -> usize {
    let (mut lo, mut hi) = (Point3::from([f64::INFINITY; 3]), Point3::from([f64::NEG_INFINITY; 3]));
    for p in positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut rng = Rng(seed);
    // An off-axis direction, since an axis-aligned ray runs along the edges of an axis-aligned part.
    let dir = Vector3::new(0.5773, 0.6123, 0.5401).normalize();
    let samples: Vec<Point3<f64>> = (0..points)
        .map(|_| {
            Point3::new(
                lo.x + rng.unit() * (hi.x - lo.x),
                lo.y + rng.unit() * (hi.y - lo.y),
                lo.z + rng.unit() * (hi.z - lo.z),
            )
        })
        .collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk = samples.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = samples
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .filter(|p| {
                            // Points on the surface have no sign worth comparing.
                            let d = field.distance(**p);
                            d.abs() > 1e-6 && (d < 0.0) != (crossings(**p, dir, positions, triangles) % 2 == 1)
                        })
                        .count()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a worker finished")).sum()
    })
}

/// One file, as one line of the report.
fn test_file(path: &Path, limit: Duration, points: usize, linear: Option<f64>) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().replace(',', ";")).unwrap_or_default();
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let out = std::env::temp_dir().join(format!("fieldform-corpus-{}.mesh", std::process::id()));
    let mut command = std::process::Command::new(importer());
    command.arg(format!("in={}", path.display())).arg(format!("out={}", out.display()));
    if let Some(linear) = linear {
        command.arg(format!("linear={linear}"));
    }
    let started = Instant::now();
    let failed = |status: &str, seconds: f64| format!("{name},{bytes},{status},{seconds:.1},,,,,,,,,,");
    let mut child = match command.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn() {
        Ok(child) => child,
        Err(_) => return failed("no importer", 0.0),
    };
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&out);
                return failed("timed out", started.elapsed().as_secs_f64());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => return failed("lost the importer", started.elapsed().as_secs_f64()),
        }
    };
    let seconds = started.elapsed().as_secs_f64();
    if !status.success() {
        let _ = std::fs::remove_file(&out);
        return failed(&format!("refused {}", status.code().unwrap_or(-1)), seconds);
    }
    let read = std::fs::File::open(&out)
        .ok()
        .and_then(|mut f| fieldform_io::mesh_wire::read_payload(&mut f).ok());
    let _ = std::fs::remove_file(&out);
    let Some((mut meshes, _)) = read else { return failed("unreadable", seconds) };

    let solids = meshes.len();
    let open: usize = meshes.iter().filter(|m| m.open_edges > 0).count();
    let triangles: usize = meshes.iter().map(|m| m.triangles.len()).sum();
    let (mut lo, mut hi) = (Point3::from([f64::INFINITY; 3]), Point3::from([f64::NEG_INFINITY; 3]));
    let mut volume = 0.0;
    for mesh in &meshes {
        for p in &mesh.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        volume += mesh
            .triangles
            .iter()
            .map(|t| {
                let at = |i: u32| Vector3::from(mesh.positions[i as usize]);
                at(t[0]).dot(&at(t[1]).cross(&at(t[2]))) / 6.0
            })
            .sum::<f64>();
    }
    meshes.sort_by_key(|m| std::cmp::Reverse(m.triangles.len()));
    let (mut tested, mut refused, mut errors) = (0usize, 0usize, 0usize);
    for (i, mesh) in meshes.iter().filter(|m| m.open_edges == 0).take(MOST_SOLIDS).enumerate() {
        let positions: Vec<Point3<f64>> = mesh.positions.iter().map(|p| Point3::from(*p)).collect();
        match MeshField::from_mesh(positions.clone(), mesh.triangles.clone()) {
            Ok(field) => {
                tested += 1;
                errors += sign_errors(&field, &positions, &mesh.triangles, points, 0x2545F4914F6CDD1D ^ i as u64);
            }
            Err(_) => refused += 1,
        }
    }
    format!(
        "{name},{bytes},ok,{seconds:.1},{solids},{open},{triangles},{volume:.3},{:.3},{:.3},{:.3},{tested},{refused},{errors}",
        hi.x - lo.x,
        hi.y - lo.y,
        hi.z - lo.z
    )
}

fn main() {
    let mut files: Vec<PathBuf> = Vec::new();
    let (mut report, mut limit, mut points, mut linear) = (None, 300u64, 300usize, None);
    for arg in std::env::args().skip(1) {
        match arg.split_once('=') {
            Some(("out", v)) => report = Some(PathBuf::from(v)),
            Some(("limit", v)) => limit = v.parse().expect("`limit` is seconds"),
            Some(("points", v)) => points = v.parse().expect("`points` is a number"),
            Some(("linear", v)) => linear = Some(v.parse::<f64>().expect("`linear` is millimetres")),
            _ => {
                let path = PathBuf::from(&arg);
                if path.is_dir() {
                    let mut found: Vec<PathBuf> = std::fs::read_dir(&path)
                        .expect("the folder reads")
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| {
                            p.extension().is_some_and(|e| {
                                let e = e.to_string_lossy().to_ascii_lowercase();
                                e == "stp" || e == "step"
                            })
                        })
                        .collect();
                    found.sort();
                    files.extend(found);
                } else {
                    files.push(path);
                }
            }
        }
    }
    let Some(report) = report else {
        eprintln!("usage: step_corpus <folder or file>... out=report.csv [limit=300] [points=300] [linear=mm]");
        std::process::exit(2);
    };
    let mut csv = std::fs::File::create(&report).expect("the report can be written");
    writeln!(csv, "file,bytes,status,seconds,solids,open,triangles,volume_mm3,dx,dy,dz,tested,refused,sign_errors")
        .expect("the report can be written");
    for (i, path) in files.iter().enumerate() {
        let line = test_file(path, Duration::from_secs(limit), points, linear);
        println!("[{}/{}] {line}", i + 1, files.len());
        writeln!(csv, "{line}").expect("the report can be written");
    }
}
