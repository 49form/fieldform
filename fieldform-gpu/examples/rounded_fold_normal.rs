// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Is a rounded fold's carried normal the surface's actual normal?
//!
//! **The one thing `gpu_matches_cpu` cannot see.** That example measures where the surface *is*,
//! by asking the CPU field how far the GPU's hit points are from it — which says nothing about
//! the direction the shader then lights them with. A normal that is wrong by a few degrees puts
//! the surface in exactly the right place and shades it as though it were a different shape.
//!
//! The rounded fold is the first composition to carry a normal out of *more than two* operands.
//! `smooth_min_normal` weights a pair by how far inside the fillet each is; the fold accumulates
//! the same weighting over its whole window. This checks that accumulation against the gradient
//! of the field the fold actually produces, which is the only independent answer available.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example rounded_fold_normal`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    Axis, Document, Fillet, Frame, Graph, Node, ParamRef, PatternKind, Place,
};
use nalgebra::{Point3, Vector3};

const BALL_MM: f64 = 20.0;
const RING_MM: f64 = 60.0;
const COUNT: u32 = 12;

/// The shader's own accumulation, in `f64`: one running minimum with its gradient, and one
/// running sum of each instance's gradient weighted by how far inside the fillet it is.
///
/// Written out here rather than called, because the point is to check *this arithmetic* and the
/// shader's copy of it is WGSL. The two must stay the same statement; if this is edited, the
/// generated loop in `codegen.rs` is the other half.
fn carried_normal(p: Point3<f64>, k: f64, n: f64) -> Vector3<f64> {
    let step = std::f64::consts::TAU / f64::from(COUNT);
    let axis = Point3::new(RING_MM, 0.0, 0.0);
    let mut least = f64::INFINITY;
    let mut nearest = Vector3::z();
    let mut weighted = Vector3::zeros();
    for i in 0..COUNT {
        let turn = -step * f64::from(i);
        let (c, s) = (turn.cos(), turn.sin());
        // Back to the frame, turned, and out again -- `pattern_point_radial`'s three steps.
        let local = p - axis;
        let spun = Vector3::new(local.x * c - local.y * s, local.x * s + local.y * c, local.z);
        let q = axis + spun;
        // The instance's own body is the sphere at the origin; its value and gradient there.
        let r = q.coords.norm();
        let d = r - BALL_MM;
        let g = if r > 1e-9 { q.coords / r } else { Vector3::z() };
        // Brought back out by the same angle the point went in by.
        let back = -turn;
        let (bc, bs) = (back.cos(), back.sin());
        let out = Vector3::new(g.x * bc - g.y * bs, g.x * bs + g.y * bc, g.z);
        if d < least {
            least = d;
            nearest = out;
        }
        let x = (k - d).max(0.0);
        weighted += x.powf(n - 1.0) * out;
    }
    if weighted.norm_squared() > 0.0 {
        weighted.normalize()
    } else {
        nearest
    }
}

fn main() {
    let lit = ParamRef::Literal;
    let doc = Document::new();

    println!(
        "A {BALL_MM} mm sphere ringed {COUNT} times at {RING_MM} mm, rounded. `worst` is the\n\
         angle between the carried normal and the field's own gradient, over points on the\n\
         surface; `seam` is the same over the points where more than one copy is inside the\n\
         fillet, which is where the two could differ and the only part worth measuring.\n"
    );
    println!(
        "  {:>9}  {:>8}  {:>12}  {:>12}",
        "radius mm", "on seam", "worst deg", "seam deg"
    );

    for k in [0.0f64, 4.0, 10.0, 18.0, 26.0] {
        let mut graph = Graph::empty();
        let body = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(BALL_MM),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: body.into(),
            count: ParamRef::Literal(f64::from(COUNT)),
            kind: PatternKind::Radial {
                frame: Frame::at_along(Axis::Z, [lit(RING_MM), lit(0.0), lit(0.0)]),
            },
            join: true,
            fillet: if k > 0.0 { Fillet::of(lit(k)) } else { Fillet::hard() },
        });
        let field = GraphField::compile_graph(&doc, &graph, 1e-3).expect("compiles");

        let (mut worst, mut seam_worst, mut on_seam) = (0.0f64, 0.0f64, 0usize);
        let mut looked = 0usize;
        for i in 0..20000 {
            let t = f64::from(i) / 20000.0;
            let a = t * std::f64::consts::TAU;
            let b = (t * 11.0).fract() * std::f64::consts::PI;
            // A direction from the ring's axis, walked in until the surface is crossed.
            let dir = Vector3::new(b.sin() * a.cos(), b.sin() * a.sin(), b.cos());
            let mut at = Point3::new(RING_MM, 0.0, 0.0) + dir * (RING_MM + BALL_MM + 30.0);
            let mut hit = false;
            for _ in 0..200 {
                let d = field.distance(at);
                if d.abs() < 1e-6 {
                    hit = true;
                    break;
                }
                at -= dir * d.max(1e-7);
                if (at - Point3::new(RING_MM, 0.0, 0.0)).norm() < 1e-3 {
                    break;
                }
            }
            if !hit {
                continue;
            }
            looked += 1;

            let h = 1e-5;
            let grad = Vector3::new(
                field.distance(at + Vector3::new(h, 0.0, 0.0))
                    - field.distance(at - Vector3::new(h, 0.0, 0.0)),
                field.distance(at + Vector3::new(0.0, h, 0.0))
                    - field.distance(at - Vector3::new(0.0, h, 0.0)),
                field.distance(at + Vector3::new(0.0, 0.0, h))
                    - field.distance(at - Vector3::new(0.0, 0.0, h)),
            );
            if grad.norm() < 1e-9 {
                continue;
            }
            let grad = grad.normalize();
            let carried = carried_normal(at, k, 2.0);
            let angle = carried.dot(&grad).clamp(-1.0, 1.0).acos().to_degrees();
            worst = worst.max(angle);

            // How many copies are inside the fillet here: more than one is a seam.
            let step = std::f64::consts::TAU / f64::from(COUNT);
            let inside = (0..COUNT)
                .filter(|i| {
                    let turn = -step * f64::from(*i);
                    let (c, s) = (turn.cos(), turn.sin());
                    let l = at - Point3::new(RING_MM, 0.0, 0.0);
                    let spun =
                        Vector3::new(l.x * c - l.y * s, l.x * s + l.y * c, l.z);
                    (Point3::new(RING_MM, 0.0, 0.0) + spun).coords.norm() - BALL_MM < k
                })
                .count();
            if inside > 1 {
                on_seam += 1;
                seam_worst = seam_worst.max(angle);
            }
        }
        println!(
            "  {k:>9.1}  {:>7.0}%  {worst:>12.3}  {seam_worst:>12.3}",
            on_seam as f64 / looked.max(1) as f64 * 100.0,
        );
    }

    println!(
        "\nA hard fold is the control: there the carried normal is one instance's own, so any\n\
         angle in that row is a fault in this example rather than in the shader. The rounded\n\
         rows are the measurement, and `seam deg` is the number that matters -- away from a\n\
         seam every weight but one is nought and the answer cannot differ."
    );
}
