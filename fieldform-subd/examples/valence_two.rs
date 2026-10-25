// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Measures what the evaluator does at a **valence-2 vertex** — the one kind of vertex the
//! knife's diagonal cuts make (the cage-editing notes, patterns C and D), and the one kind the
//! published smoothness results for Catmull-Clark say nothing about, since they start at three.
//!
//! The question is whether a diagonal cut leaves a dent or a crease where none was drawn. It is
//! answered against a control: the same face of the same cage cut the other way, by a ring
//! through its opposite edges, which is the cut the knife makes everywhere else. Both are
//! measured against the uncut surface:
//!
//! - **position**: how far the cut surface strays from the uncut one, within one face of the
//!   new vertex, as the farthest sample's distance to the uncut tessellation;
//! - **normal continuity**: the largest turn between the normals of neighbouring samples
//!   within that reach — a crease is a turn that does not shrink as the samples get closer;
//! - **curvature**: that turn divided by the samples' spacing, against the uncut face's own.
//!
//! Run with: `cargo run --release -p fieldform-subd --example valence_two`

use fieldform_cage::edit::insert_ring;
use fieldform_cage::{Cage, FaceId, QuadMesh, VertexId};
use fieldform_subd::{subdivide_n, LimitSurface, Tessellation};
use nalgebra::{Point3, Vector3};

/// Grid levels per face: 2^L + 1 samples a side. Three of them, because the one question that
/// matters -- crease or curvature -- is answered by what the turn between neighbouring samples
/// does as they get closer: a crease keeps its angle, a curved surface halves it with the step.
const LEVELS: [usize; 3] = [4, 5, 6];

fn main() {
    // A rounded cube: twice subdivided, so a face in the middle of a side has four regular
    // corners and real curvature. The cut is the kind of face a user would cut.
    let base = subdivide_n(&Cage::cube_of(100.0), 2).expect("a cube subdivides");
    let face = middle_of_a_side(&base);
    let corners = base.face_vertices(face);

    let centre = {
        let sum = corners.iter().fold(Vector3::zeros(), |a, &v| a + base.vertex_position(v).coords);
        Point3::from(sum / 4.0)
    };
    // One cage edge of the cut face, in millimetres: the reach every figure is taken within.
    let reach = (base.vertex_position(corners[0]) - base.vertex_position(corners[1])).norm();

    // Pattern C: the face's diagonal, through a new vertex at its centroid.
    let diagonal = {
        let mut positions: Vec<Point3<f64>> = base.vertex_ids().map(|v| base.vertex_position(v)).collect();
        let p = positions.len() as u32;
        positions.push(centre);
        let faces: Vec<Vec<u32>> = base
            .face_ids()
            .flat_map(|f| {
                let c = base.face_vertices(f).map(|v| v.index() as u32);
                if f == face {
                    vec![vec![c[0], c[1], c[2], p], vec![c[2], c[3], c[0], p]]
                } else {
                    vec![c.to_vec()]
                }
            })
            .collect();
        QuadMesh::from_faces(positions, &faces).expect("a diagonal cut is a valid mesh")
    };
    assert_eq!(diagonal.valence(VertexId::from_index(diagonal.vertex_count() - 1)), 2);

    // Pattern A, the control: a ring through the face's first edge at its middle.
    let ring = insert_ring(base.mesh(), (corners[0], corners[1]), 0.5).expect("a ring").mesh;

    // Where the diagonal's new vertex ends up on the surface: the limit mask moves it off the
    // centroid, and the turn *at* the vertex is read there.
    let p_limit = {
        let cage = Cage::try_from_mesh(diagonal.clone()).map_err(|(_, e)| e).expect("closed");
        fieldform_subd::limit_position(&cage, VertexId::from_index(cage.vertex_count() - 1))
    };

    println!("Rounded cube of 100 mm, twice subdivided; the cut face is {reach:.2} mm a side.");
    println!("Figures within {reach:.2} mm of its centre; the step is the grid spacing at that level.\n");
    println!(
        "{:<7} {:<14} {:>10} {:>12} {:>14} {:>16} {:>14}",
        "grid", "cut", "step (mm)", "strays (mm)", "max turn (°)", "turn/step (°/mm)", "turn at P (°)"
    );
    for &levels in &LEVELS {
        let uncut = LimitSurface::new(&base).tessellate(levels);
        let step = reach / (1u32 << levels) as f64;
        let rows = [
            ("uncut", LimitSurface::new(&base).tessellate(levels), None),
            ("ring (A)", LimitSurface::over(&ring).tessellate(levels), None),
            ("diagonal (C)", LimitSurface::over(&diagonal).tessellate(levels), Some(p_limit)),
        ];
        for (name, tess, at) in &rows {
            let m = measure(tess, &uncut, centre, reach);
            let at_p = at.map(|p| turn_at(tess, p));
            println!(
                "{:<7} {:<14} {:>10.3} {:>12.4} {:>14.3} {:>16.3} {:>14}",
                format!("{}²", (1u32 << levels) + 1),
                name,
                step,
                m.stray,
                m.max_turn_degrees,
                m.max_turn_per_mm,
                at_p.map_or("-".to_string(), |(t, n)| format!("{t:.3} ({n})")),
            );
        }
    }
}

/// A face of a rounded cube whose corners are all regular and whose middle is on a side.
fn middle_of_a_side(cage: &Cage) -> FaceId {
    cage.face_ids()
        .filter(|&f| cage.face_vertices(f).iter().all(|&v| cage.valence(v) == 4))
        .max_by(|&a, &b| {
            let far = |f: FaceId| {
                let c = cage.face_vertices(f);
                let sum = c.iter().fold(Vector3::zeros(), |s, &v| s + cage.vertex_position(v).coords) / 4.0;
                // Nearest the middle of a side: one coordinate large, the other two small.
                let mut a = [sum.x.abs(), sum.y.abs(), sum.z.abs()];
                a.sort_by(|x, y| x.partial_cmp(y).unwrap());
                a[2] - a[1] - a[0]
            };
            far(a).partial_cmp(&far(b)).unwrap()
        })
        .expect("a twice-subdivided cube has regular faces")
}

struct Measure {
    stray: f64,
    max_turn_degrees: f64,
    max_turn_per_mm: f64,
}

/// The three figures for `tess` against `reference`, over the samples within `reach` of `at`.
fn measure(tess: &Tessellation, reference: &Tessellation, at: Point3<f64>, reach: f64) -> Measure {
    let near: Vec<usize> = (0..tess.positions.len()).filter(|&i| (tess.positions[i] - at).norm() <= reach).collect();
    let stray = near.iter().map(|&i| distance_to(reference, tess.positions[i])).fold(0.0, f64::max);

    let mut max_turn: f64 = 0.0;
    let mut max_per_mm: f64 = 0.0;
    for tri in &tess.triangles {
        for k in 0..3 {
            let (a, b) = (tri[k] as usize, tri[(k + 1) % 3] as usize);
            if (tess.positions[a] - at).norm() > reach || (tess.positions[b] - at).norm() > reach {
                continue;
            }
            let step = (tess.positions[a] - tess.positions[b]).norm();
            if step < 1e-9 {
                continue;
            }
            let cos = tess.normals[a].dot(&tess.normals[b]).clamp(-1.0, 1.0);
            let turn = cos.acos().to_degrees();
            max_turn = max_turn.max(turn);
            max_per_mm = max_per_mm.max(turn / step);
        }
    }
    Measure { stray, max_turn_degrees: max_turn, max_turn_per_mm: max_per_mm }
}

/// The widest angle between the normals of the samples that sit on `at` — the welded grid
/// corner, which every face touching it evaluates once, and the samples a step away.
fn turn_at(tess: &Tessellation, at: Point3<f64>) -> (f64, usize) {
    let on: Vec<usize> = (0..tess.positions.len()).filter(|&i| (tess.positions[i] - at).norm() < 1e-6).collect();
    let mut worst: f64 = 0.0;
    for &i in &on {
        for tri in &tess.triangles {
            if tri.contains(&(i as u32)) {
                for &j in tri {
                    let cos = tess.normals[i].dot(&tess.normals[j as usize]).clamp(-1.0, 1.0);
                    worst = worst.max(cos.acos().to_degrees());
                }
            }
        }
    }
    (worst, on.len())
}

/// Distance from `p` to the nearest triangle of `tess`. Brute force: the tessellations here
/// are a few hundred thousand triangles and the samples a few hundred.
fn distance_to(tess: &Tessellation, p: Point3<f64>) -> f64 {
    tess.triangles
        .iter()
        .map(|t| {
            point_triangle_distance(p, tess.positions[t[0] as usize], tess.positions[t[1] as usize], tess.positions[t[2] as usize])
        })
        .fold(f64::INFINITY, f64::min)
}

/// The classic closest-point-on-triangle by Voronoi region.
fn point_triangle_distance(p: Point3<f64>, a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.norm();
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.norm();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).norm();
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.norm();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).norm();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).norm();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).norm()
}
