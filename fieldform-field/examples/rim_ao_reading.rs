// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the field reads along a rim's own normal, against the distance it should read.
//!
//! **Why this exists.** The still's occlusion steps out from a hit along the normal and reads
//! the field at each step; wherever it reads less than the step it darkens, taking the shortfall
//! for nearby material. A rim's value is `Rim::join`, a bound: the max-norm distance to the
//! rounded corner's zero set, which is what lets it hold at any angle between its two readings.
//! A bound reads short, and along the arc's normal at forty-five degrees the max-norm reads
//! `1/√2` of the Euclidean distance -- which occlusion would paint as a 29 % shadow on every
//! rounded rim. This prints the ratio, at the steps occlusion takes, for each rim and each shape,
//! so the mechanism is a number before it is a fix. The fix, `Rim::cut`, reads the exact corner
//! where the readings are square; spec §4.4 keeps both tables.
//!
//! **And the same for a blend between two faces**, at three seam angles: the reading along the
//! bisector is `cos(θ/2)` of the step -- 0.707 square, 0.5 at 120°, 0.866 at 60° -- which is what
//! occlusion sees of every fillet in a document. Spec §4.4 lists what could be done about it.
//!
//! Run with: `cargo run -p fieldform-field --example rim_ao_reading`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    AngleSides, Continuity, Curve, CurveId, Document, Frame, Graph, Node, Outline, ParamRef,
    RimEnds, Rims, Sides, Sweep,
};
use nalgebra::{Point3, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn rims(k: f64, c: Continuity) -> Rims {
    Rims { radius: lit(k), continuity: c, at: RimEnds::Both, second: None }
}

/// A square forty across on the ground, extruded thirty up.
fn extruded(k: f64, c: Continuity) -> GraphField {
    let mut doc = Document::new();
    let id = CurveId::new();
    let pts = [(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect();
    doc.curves.insert(id, Curve::polyline(id, "square", pts, true).unwrap());
    let mut graph = Graph::empty();
    graph.add(Node::Extrude {
        outline: Outline::closed(id),
        sweep: Sweep::Normal,
        start: lit(0.0),
        height: lit(30.0),
        draft: lit(0.0),
        sides: Sides::One,
        corners: Default::default(),
        rims: rims(k, c),
    });
    GraphField::compile_graph(&doc, &graph, 1e-4).unwrap()
}

/// A ring from radius 30 to 40, ten high, turned ninety degrees about Z.
fn revolved(k: f64, c: Continuity) -> GraphField {
    let mut doc = Document::new();
    let id = CurveId::new();
    let pts = vec![
        Point3::new(30.0, 0.0, -5.0),
        Point3::new(40.0, 0.0, -5.0),
        Point3::new(40.0, 0.0, 5.0),
        Point3::new(30.0, 0.0, 5.0),
    ];
    doc.curves.insert(id, Curve::polyline(id, "ring", pts, true).unwrap());
    let mut graph = Graph::empty();
    graph.add(Node::Revolve {
        outline: Outline::closed(id),
        frame: Frame::default(),
        angle: lit(90.0),
        sides: AngleSides::One,
        rims: rims(k, c),
    });
    GraphField::compile_graph(&doc, &graph, 1e-4).unwrap()
}

/// The surface point along the corner's diagonal from the sharp corner `corner`, going in by
/// `inward` (a unit vector), found by bisection, and the normal there -- the diagonal itself,
/// which every shape's curve is symmetric about.
fn on_the_diagonal(field: &GraphField, corner: Point3<f64>, inward: Vector3<f64>) -> (Point3<f64>, Vector3<f64>) {
    let (mut lo, mut hi) = (0.0, 20.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if field.distance(corner + inward * mid) > 0.0 { lo = mid } else { hi = mid }
    }
    (corner + inward * (0.5 * (lo + hi)), -inward)
}

/// Two faces meeting along the z axis at a dihedral angle set by `turn`: a half-space `x ≤ 0`
/// and a second one turned `turn` degrees about z from `y ≤ 0`, so their outward normals are
/// `(1, 0, 0)` and `(−sin, cos, 0)` and the inside angle is 90° at nought, 120° at 30°, 60° at
/// −30°. Unioned with a fillet of reach `k`, which is the blend every seam in a document is.
fn seamed(k: f64, c: Continuity, turn: f64) -> (GraphField, Vector3<f64>) {
    let mut graph = Graph::empty();
    let a = graph.add(Node::Cuboid {
        edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(-50.0), lit(0.0), lit(0.0)]),
        size: [lit(100.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    let b = graph.add(Node::Cuboid {
        edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0), lit(-50.0), lit(0.0)]),
        size: [lit(100.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    let b = graph.add(Node::Transform {
        input: b.into(),
        translation: [lit(0.0); 3],
        rotation: [lit(0.0), lit(0.0), lit(turn)],
        scale: lit(1.0),
        about: fieldform_scene::Place::origin(),
    });
    graph.add(Node::Blend {
        op: fieldform_scene::BooleanOp::Union,
        a: a.into(),
        b: b.into(),
        fillet: fieldform_scene::Fillet { radius: lit(k), continuity: c, second: None },
    });
    let (s, co) = turn.to_radians().sin_cos();
    let bisector = (Vector3::new(1.0, 0.0, 0.0) + Vector3::new(-s, co, 0.0)).normalize();
    (GraphField::compile_graph(&Document::new(), &graph, 1e-4).unwrap(), bisector)
}

/// The fillet's surface point out along the seam's bisector from the seam line, and the normal
/// there, which the symmetry makes the bisector itself.
fn on_the_seam(field: &GraphField, out: Vector3<f64>) -> (Point3<f64>, Vector3<f64>) {
    let (mut lo, mut hi) = (0.0, 20.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if field.distance(Point3::origin() + out * mid) < 0.0 { lo = mid } else { hi = mid }
    }
    (Point3::origin() + out * (0.5 * (lo + hi)), out)
}

fn main() {
    let k = 3.0;
    println!("reading / step along a blend's normal at the seam's bisector, k = {k} mm");
    println!("{:<44} {:>8} {:>8} {:>8} {:>8}", "blend between two faces", "0.2k", "0.5k", "1k", "2k");
    for shape in Continuity::ALL {
        for (angle, turn) in [(90.0, 0.0), (120.0, 30.0), (60.0, -30.0)] {
            let (field, out) = seamed(k, shape, turn);
            let (p, n) = on_the_seam(&field, out);
            let ratio = |h: f64| field.distance(p + n * h) / h;
            println!(
                "{:<44} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                format!("{angle:.0} degree seam, {shape:?}"),
                ratio(0.2 * k),
                ratio(0.5 * k),
                ratio(k),
                ratio(2.0 * k)
            );
        }
    }
    println!();
    println!("reading / step along the rim's normal at the corner's diagonal, k = {k} mm");
    println!("{:<44} {:>8} {:>8} {:>8} {:>8}", "rim", "0.2k", "0.5k", "1k", "2k");
    for shape in Continuity::ALL {
        let cases: Vec<(String, GraphField, Point3<f64>, Vector3<f64>)> = vec![
            (
                format!("upright extrude, {shape:?}"),
                extruded(k, shape),
                Point3::new(20.0, 0.0, 30.0),
                Vector3::new(-1.0, 0.0, -1.0).normalize(),
            ),
            (
                format!("partial revolve, outer wall at the cap, {shape:?}"),
                revolved(k, shape),
                // The corner where the outer wall meets the start cap, at mid-height: the cap
                // plane is y = 0 and the wall r = 40, so the diagonal runs in -x and +y.
                Point3::new(40.0, 0.0, 0.0),
                Vector3::new(-1.0, 1.0, 0.0).normalize(),
            ),
        ];
        for (what, field, corner, inward) in cases {
            let (p, n) = on_the_diagonal(&field, corner, inward);
            let ratio = |h: f64| field.distance(p + n * h) / h;
            println!(
                "{:<44} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                what,
                ratio(0.2 * k),
                ratio(0.5 * k),
                ratio(k),
                ratio(2.0 * k)
            );
        }
    }
    println!();
    println!("1.000 is the distance; occlusion darkens by about 1.5 x (1 - ratio) at each step.");
}
