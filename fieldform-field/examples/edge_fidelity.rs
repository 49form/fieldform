// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How sharp a sharp edge comes out, measured on geometry whose edges are known exactly.
//!
//! A 40 mm cube, whose twelve edges are exact lines and eight corners exact points. Meshed two
//! ways that describe the *same* surface:
//!
//! - **exact** — the closed-form box distance, so every normal the mesher asks for is analytic.
//! - **max of three slabs** — the same shape as a boolean intersection builds it, which is what
//!   a sphere intersected with a lattice actually is.
//!
//! Splitting them separates two possible causes of a blunt edge. If the intersection is worse,
//! the loss is in composition. If they agree, the limit is the mesher's one-vertex-per-cell
//! rule and nothing above it will help.
//!
//! Reported separately for vertices **near a crease** and vertices **on a flat face**, because
//! a mean over the whole surface is dominated by the faces and would hide exactly the thing
//! being asked about.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example edge_fidelity`

use fieldform_field::{ScalarField, mesh_field};
use nalgebra::{Point3, Vector3};

const HALF_MM: f64 = 20.0;

/// A rotation that puts none of the cube's faces on a grid plane.
///
/// The axis-aligned cube is the *easy* case and flatters the mesher: every crease lies exactly
/// on a row of cell boundaries, so the vertex that resolves it has nowhere wrong to go. A real
/// part is not aligned to the export grid, and a lattice cut by a sphere is not aligned to
/// anything, so the tilted cube is the one that answers the question actually being asked.
fn tilt() -> nalgebra::Rotation3<f64> {
    nalgebra::Rotation3::from_euler_angles(0.3, 0.4, 0.5)
}

/// The tilted cube, in the frame the cube is axis-aligned in.
fn untilt(p: Point3<f64>) -> Point3<f64> {
    tilt().inverse() * p
}

/// The closed-form box distance: exact everywhere, inside and out.
struct ExactBox;
impl ScalarField for ExactBox {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let q = Vector3::new(
            p.x.abs() - HALF_MM,
            p.y.abs() - HALF_MM,
            p.z.abs() - HALF_MM,
        );
        let outside = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
        let inside = q.x.max(q.y).max(q.z).min(0.0);
        outside + inside
    }
}

/// The same cube as three slabs intersected, which is how a boolean builds it. Inside, this is
/// the Chebyshev distance rather than the Euclidean one, so it is a lower bound -- the zero set
/// and every crease are identical.
struct SlabBox;
impl ScalarField for SlabBox {
    fn distance(&self, p: Point3<f64>) -> f64 {
        (p.x.abs() - HALF_MM)
            .max(p.y.abs() - HALF_MM)
            .max(p.z.abs() - HALF_MM)
    }
    fn normalization(&self) -> fieldform_field::Normalization {
        fieldform_field::Normalization::LowerBound
    }
}

/// How close `p` is to one of the cube's twelve edges: the second-largest of the three
/// per-axis distances to a face plane. Zero on an edge, and large in the middle of a face.
fn crease_proximity(p: Point3<f64>) -> f64 {
    let mut d = [
        (p.x.abs() - HALF_MM).abs(),
        (p.y.abs() - HALF_MM).abs(),
        (p.z.abs() - HALF_MM).abs(),
    ];
    d.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    d[1]
}

/// The same two fields, rotated. `distance` is invariant under a rigid motion, so evaluating
/// the unrotated field at the unrotated point is exact rather than an approximation.
struct Tilted<F>(F);
impl<F: ScalarField + Sync> ScalarField for Tilted<F> {
    fn distance(&self, p: Point3<f64>) -> f64 {
        self.0.distance(untilt(p))
    }
    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        self.0.normal(untilt(p)).map(|n| tilt() * n)
    }
    fn normalization(&self) -> fieldform_field::Normalization {
        self.0.normalization()
    }
    fn lipschitz(&self) -> f64 {
        self.0.lipschitz()
    }
}

fn report(name: &str, field: &(dyn ScalarField + Sync), tilted: bool, resolution: usize) {
    let r = HALF_MM * 1.6;
    let mesh = mesh_field(
        field,
        (Point3::new(-r, -r, -r), Point3::new(r, r, r)),
        resolution,
    );
    let cell = (r * 2.0) / resolution as f64;

    let (mut crease_worst, mut crease_sum, mut crease_n) = (0.0f64, 0.0f64, 0usize);
    let (mut face_worst, mut face_sum, mut face_n) = (0.0f64, 0.0f64, 0usize);
    for &v in &mesh.positions {
        // The *exact* distance is the yardstick for both fields: SlabBox understates inside,
        // and grading it against its own understatement would flatter it. Measured in the
        // cube's own frame, where "exact" has a closed form.
        let local = if tilted { untilt(v) } else { v };
        let err = ExactBox.distance(local).abs();
        if crease_proximity(local) < cell * 1.5 {
            crease_worst = crease_worst.max(err);
            crease_sum += err;
            crease_n += 1;
        } else {
            face_worst = face_worst.max(err);
            face_sum += err;
            face_n += 1;
        }
    }

    // Slivers: triangles whose area is a small fraction of a cell face. These are the specks
    // that appear scattered over an otherwise smooth surface.
    let mut slivers = 0usize;
    for t in &mesh.triangles {
        let (a, b, c) = (
            mesh.positions[t[0] as usize],
            mesh.positions[t[1] as usize],
            mesh.positions[t[2] as usize],
        );
        let area = (b - a).cross(&(c - a)).norm() * 0.5;
        if area < cell * cell * 1e-3 {
            slivers += 1;
        }
    }

    println!(
        "{name:>12}  {resolution:>5}  {cell:>7.3}  {:>8}  {:>9.4}  {:>8.4}  {:>9.4}  {:>8.4}  \
         {:>7}",
        mesh.triangles.len(),
        crease_worst,
        if crease_n > 0 {
            crease_sum / crease_n as f64
        } else {
            0.0
        },
        face_worst,
        if face_n > 0 {
            face_sum / face_n as f64
        } else {
            0.0
        },
        slivers,
    );
}

fn main() {
    println!(
        "A {} mm cube. Error is |exact distance| at each meshed vertex.\n",
        HALF_MM * 2.0
    );
    println!(
        "{:>12}  {:>5}  {:>7}  {:>8}  {:>20}  {:>20}  {:>7}",
        "field", "cells", "cell", "tris", "near a crease (mm)", "on a face (mm)", "slivers"
    );
    println!(
        "{:>12}  {:>5}  {:>7}  {:>8}  {:>9}  {:>8}  {:>9}  {:>8}  {:>7}",
        "", "", "", "", "worst", "mean", "worst", "mean", ""
    );
    for resolution in [64usize, 128, 256] {
        report("exact", &ExactBox, false, resolution);
        report("slabs", &SlabBox, false, resolution);
        report("exact tilt", &Tilted(ExactBox), true, resolution);
        report("slabs tilt", &Tilted(SlabBox), true, resolution);
    }
}
