// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A wall on a curve that inflects, probed for what "breaks".
//!
//! Reported 2026-10-02 from the application: *"when a curve of degree 2 or more inflects the
//! wall-based extrude tends to break."* The thin-extrude scenes in `gpu_matches_cpu` carry an S
//! and pass, so whatever breaks is in a case they do not cover. This builds the cases a person
//! draws -- an S of degree 2 and of degree 3, with lobes equal and unequal, on the ground and
//! on a vertical drawing plane, to each side -- and asks three things of each: does it compile,
//! where does its plane come out, and does the field agree with the band itself about what is
//! inside.
//!
//! The band is checked by brute force: a point is in a centre wall where the nearest point of
//! the finely sampled curve is within half the thickness, and for a one-sided wall where it is
//! on that side -- left and right read in the section's own frame, the one the field lays the
//! wall in, since a hand-picked pair of axes on a vertical plane is as likely to be left-handed
//! as not (the first run of this said 220 probes were on the wrong side for exactly that
//! reason, and nothing was). Square ends are left out of the comparison by probing only points
//! whose nearest curve point is well inside the curve's ends.
//!
//! Run with: `cargo run --release -p fieldform-field --example inflecting_wall_probe`

use fieldform_field::{ExtrudeSection, GraphField, ScalarField};
use fieldform_scene::{
    Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Sweep, WallEnds, WallSide,
};
use nalgebra::{Point3, Vector3};

const THICKNESS_MM: f64 = 4.0;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// An S through these control points: `lift` is how the ground-plane points are carried onto
/// a vertical plane, or left on the ground.
fn s_curve(
    doc: &mut Document,
    degree: usize,
    control: Vec<(f64, f64)>,
    lift: &dyn Fn(f64, f64) -> Point3<f64>,
) -> CurveId {
    let n = control.len();
    let mut knots = vec![0.0; degree + 1];
    for i in 1..(n - degree) {
        knots.push(i as f64);
    }
    knots.extend(std::iter::repeat((n - degree) as f64).take(degree + 1));
    let points: Vec<Point3<f64>> = control.iter().map(|&(x, y)| lift(x, y)).collect();
    let curve = Curve::new(CurveId::new(), "S", degree, points, vec![1.0; n], knots, false)
        .expect("a spline");
    let id = curve.id;
    doc.curves.insert(id, curve);
    id
}

fn main() {
    // Equal lobes cancel in signed area; unequal ones do not.
    let equal_3 = vec![(0.0, 0.0), (30.0, 25.0), (60.0, -25.0), (90.0, 0.0)];
    let unequal_3 = vec![(0.0, 0.0), (30.0, 40.0), (60.0, -15.0), (90.0, 0.0)];
    let equal_2 = vec![(0.0, 0.0), (22.0, 30.0), (45.0, 0.0), (68.0, -30.0), (90.0, 0.0)];
    let unequal_2 = vec![(0.0, 0.0), (22.0, 45.0), (45.0, 0.0), (68.0, -15.0), (90.0, 0.0)];
    let ground = |x: f64, y: f64| Point3::new(x, y, 0.0);
    let back_view = |x: f64, y: f64| Point3::new(x, 10.0, y);
    let planes: [(&str, &dyn Fn(f64, f64) -> Point3<f64>); 2] =
        [("on the ground", &ground), ("in a back view", &back_view)];

    for (plane_name, lift) in planes {
        for (shape, degree, control) in [
            ("equal lobes", 3, &equal_3),
            ("unequal lobes", 3, &unequal_3),
            ("equal lobes", 2, &equal_2),
            ("unequal lobes", 2, &unequal_2),
        ] {
            for side in [WallSide::Centre, WallSide::Left, WallSide::Right] {
                let mut doc = Document::new();
                let id = s_curve(&mut doc, degree, control.clone(), lift);
                let outline = Outline::Wall {
                    curve: id,
                    thickness: lit(THICKNESS_MM),
                    side,
                    ends: WallEnds::Square,
                };
                let label = format!("degree {degree}, {shape}, {plane_name}, {side:?}");
                let section =
                    fieldform_field::extrude_section(&doc, &outline, &Sweep::Normal, 1.0);
                let normal = match &section {
                    Ok(s) => format!(
                        "normal ({:.3}, {:.3}, {:.3})",
                        s.normal.x, s.normal.y, s.normal.z
                    ),
                    Err(e) => format!("no section: {e}"),
                };
                let Ok(section) = section else {
                    println!("{label}: {normal}");
                    continue;
                };
                let mut graph = Graph::empty();
                graph.add(Node::Extrude { rims: Default::default(),
                    outline,
                    sweep: Sweep::Normal,
                    height: lit(20.0),
                    draft: lit(0.0),
                    start: lit(0.0),
                    sides: fieldform_scene::Sides::One,
                    corners: Default::default(),
                });
                let field = match GraphField::compile_graph(&doc, &graph, 1e-4) {
                    Ok(f) => f,
                    Err(e) => {
                        println!("{label}: {normal}; REFUSED: {e}");
                        continue;
                    }
                };
                let (wrong, worst_over, checked) = disagreement(&doc, id, side, &section, &field);
                println!(
                    "{label}: {normal}; {checked} probes, {wrong} on the wrong side, worst \
                     overstatement {worst_over:.4} mm"
                );
            }
        }
    }
}

/// How many probe points the field puts on the other side of the surface from the band
/// itself, and the field's worst overstatement of distance, over a grid through the solid.
fn disagreement(
    doc: &Document,
    id: CurveId,
    side: WallSide,
    section: &ExtrudeSection,
    field: &GraphField,
) -> (usize, f64, usize) {
    let curve = &doc.curves[&id];
    // The curve, very finely, read in the section's own frame, as the wall is.
    let fine: Vec<Point3<f64>> = (0..=4000)
        .map(|i| {
            let (a, b) = curve.domain();
            curve.evaluate(a + (b - a) * i as f64 / 4000.0)
        })
        .collect();
    let rotation = fieldform_field::rotation_from_z(section.normal);
    let plane_xy = |p: Point3<f64>| {
        let local = rotation.inverse_transform_vector(&(p - section.origin));
        (local.x, local.y)
    };
    // Half way up the extrude, so the probes are in the solid's own slab and not at a face.
    let lift = |x: f64, y: f64| {
        section.origin + rotation * Vector3::new(x, y, 10.0)
    };
    let half = 0.5 * THICKNESS_MM;
    let mut wrong = 0;
    let mut checked = 0;
    let mut worst_over: f64 = 0.0;
    let probes: Vec<Point3<f64>> = (0..60)
        .flat_map(|i| {
            (0..60).map(move |j| (-10.0 + 110.0 * i as f64 / 59.0, -50.0 + 100.0 * j as f64 / 59.0))
        })
        .map(|(x, y)| lift(x, y))
        .collect();
    for p in probes {
        let (px, py) = plane_xy(p);
        // Nearest fine sample, and the lateral side of it, from the tangent there.
        let (mut best, mut at) = (f64::INFINITY, 0usize);
        for (k, q) in fine.iter().enumerate() {
            let (qx, qy) = plane_xy(*q);
            let d = ((px - qx).powi(2) + (py - qy).powi(2)).sqrt();
            if d < best {
                best = d;
                at = k;
            }
        }
        // Keep clear of the ends, where square ends and the capsule-like brute force differ.
        if at < 200 || at > 3800 {
            continue;
        }
        let (ax, ay) = plane_xy(fine[at.saturating_sub(1)]);
        let (bx, by) = plane_xy(fine[(at + 1).min(4000)]);
        let (tx, ty) = (bx - ax, by - ay);
        let (qx, qy) = plane_xy(fine[at]);
        // Left of travel: the cross of the tangent with the offset.
        let cross = tx * (py - qy) - ty * (px - qx);
        let inside_band = match side {
            WallSide::Centre => best <= half,
            WallSide::Left => cross >= 0.0 && best <= THICKNESS_MM,
            WallSide::Right => cross <= 0.0 && best <= THICKNESS_MM,
        };
        // Within a tessellation's sag of the surface either answer is fine.
        let margin = 0.05;
        let band_distance = match side {
            WallSide::Centre => best - half,
            _ if inside_band => -(THICKNESS_MM - best).min(best),
            _ => best.max(0.0),
        };
        if band_distance.abs() < margin {
            continue;
        }
        checked += 1;
        let got = field.distance(p);
        if (got < 0.0) != inside_band {
            wrong += 1;
        }
        // Outside, the field may not claim more room than the band leaves.
        if !inside_band && got > band_distance + 1e-3 {
            worst_over = worst_over.max(got - band_distance);
        }
    }
    (wrong, worst_over, checked)
}
