// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Every point a profile solid's field calls solid lies inside the box `graph_extent` gives it.
//!
//! **The extent is a guarantee now, not a hint.** The GPU ray starts at the scene sphere built
//! from it, and the per-tile body cull skips a body wherever its box misses a tile — so a box that
//! is too small is material that is not drawn, with nothing on screen to say why. A drafted
//! extrude's box once stopped at its base (0a94ac9); this file asks the same question of every
//! other way a profile becomes a solid: extrudes straight and leaning, one-sided and symmetric,
//! drafted out, in and not at all; revolves; and sweeps along bent spines with each section.
//!
//! The survey is a regular grid over the box grown by half its size (at least 10 mm) on every
//! side, plus a fine sheet of samples just outside each face, where a box that is short by a
//! tessellation's sag would be caught. A violation is a sample reading below -1e-6 that lies more
//! than 1e-6 mm outside the box. Violating cases are left failing, on purpose: they are reports.

use fieldform_field::{graph_extent, Extent, GraphField, ScalarField};
use fieldform_scene::{Curve, CurveId, Document, Frame, Graph, Node, NodeId, ParamRef, Section, Sweep};
use nalgebra::{Point3, Rotation3, Unit, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// The tolerance the task compiles at: fine enough that no profile is baked, so the field is
/// the polyline's own exact distance and anything found is the extent's fault.
const FINE: f64 = 1e-4;

// ---------------------------------------------------------------------------------------------
// Building curves
// ---------------------------------------------------------------------------------------------

fn polyline(doc: &mut Document, points: Vec<Point3<f64>>, closed: bool) -> CurveId {
    let curve = Curve::polyline(CurveId::new(), "c", points, closed).expect("a polyline");
    let id = curve.id;
    doc.curves.insert(id, curve);
    id
}

/// A clamped, uniform B-spline through the given control polygon.
fn spline(doc: &mut Document, degree: usize, control: Vec<Point3<f64>>) -> CurveId {
    let n = control.len();
    let mut knots = vec![0.0; degree + 1];
    for i in 1..(n - degree) {
        knots.push(i as f64);
    }
    knots.extend(std::iter::repeat((n - degree) as f64).take(degree + 1));
    let weights = vec![1.0; n];
    let curve =
        Curve::new(CurveId::new(), "s", degree, control, weights, knots, false).expect("a spline");
    let id = curve.id;
    doc.curves.insert(id, curve);
    id
}

/// An exact circle as a closed rational quadratic, of radius `r` about `centre` in the plane
/// spanned by `u` and `v`, **started `phase` degrees round** so its extremes fall mid-span rather
/// than on a knot, where a tessellation always samples them.
fn circle(
    doc: &mut Document,
    centre: Point3<f64>,
    u: Vector3<f64>,
    v: Vector3<f64>,
    r: f64,
    phase: f64,
) -> CurveId {
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let (s, c) = phase.to_radians().sin_cos();
    let (a, b) = (u * c + v * s, v * c - u * s);
    let corners = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
    ];
    let control: Vec<Point3<f64>> =
        corners.iter().map(|&(x, y)| centre + (a * x + b * y) * r).collect();
    let weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];
    let knots = vec![0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0];
    let curve = Curve::new(CurveId::new(), "circle", 2, control, weights, knots, true)
        .expect("a rational circle");
    let id = curve.id;
    doc.curves.insert(id, curve);
    id
}

/// A non-convex L, off the origin, in the ground plane: its box is not centred on anything and
/// its re-entrant corner is where a hull-shaped bound would be wrong.
fn l_shape() -> Vec<Point3<f64>> {
    [(20.0, 10.0), (60.0, 10.0), (60.0, 25.0), (35.0, 25.0), (35.0, 50.0), (20.0, 50.0)]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect()
}

/// The same L carried off the ground plane — tilted about an oblique axis and moved — so the
/// profile's own frame is not the world's and the box has to be taken of a rotated solid.
fn tilted_l_shape() -> Vec<Point3<f64>> {
    let turn = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(1.0, 2.0, 0.5)), 0.9);
    let shift = Vector3::new(30.0, -40.0, 20.0);
    l_shape().into_iter().map(|p| turn * p + shift).collect()
}

fn extrude(doc: &Document, profile: CurveId, sweep: Sweep, height: f64, draft: f64, symmetric: bool) -> (Graph, NodeId) {
    let _ = doc;
    let mut graph = Graph::empty();
    let node = graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(profile),
        sweep,
        height: lit(height),
        draft: lit(draft),
        start: lit(0.0),
        sides: fieldform_scene::Sides::symmetric_if(symmetric),
        corners: Default::default(),
    });
    (graph, node)
}

fn toward(x: f64, y: f64, z: f64) -> Sweep {
    Sweep::Toward { xyz: [lit(x), lit(y), lit(z)] }
}

// ---------------------------------------------------------------------------------------------
// The survey
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct Outcome {
    extent: Extent,
    samples: usize,
    solid: usize,
    /// How far outside the box the worst solid sample lies, and where, and what it read.
    worst: f64,
    at: Option<Point3<f64>>,
    value: f64,
}

/// How far `p` lies outside the box, as a Euclidean distance: nought inside.
fn outside(p: Point3<f64>, lo: Point3<f64>, hi: Point3<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        let d = (lo[i] - p[i]).max(p[i] - hi[i]).max(0.0);
        s += d * d;
    }
    s.sqrt()
}

/// Samples the grown box on a grid and on sheets just outside each face.
///
/// `fallback` is the region an `Empty` or `Unbounded` answer is surveyed over instead, so that
/// an Empty box can still be caught hiding material.
fn survey(field: &GraphField, extent: Extent, fallback: (Point3<f64>, Point3<f64>)) -> Outcome {
    let (lo, hi, bounded) = match extent {
        Extent::Bounded { lo, hi } => (lo, hi, true),
        _ => (fallback.0, fallback.1, false),
    };
    let size = hi - lo;
    let grow = Vector3::new(
        (size.x * 0.5).max(10.0),
        (size.y * 0.5).max(10.0),
        (size.z * 0.5).max(10.0),
    );
    let (glo, ghi) = if bounded { (lo - grow, hi + grow) } else { (lo, hi) };

    let mut out = Outcome { extent, samples: 0, solid: 0, worst: 0.0, at: None, value: 0.0 };
    let visit = |p: Point3<f64>, out: &mut Outcome| {
        out.samples += 1;
        let d = field.distance(p);
        if d < -1e-6 {
            out.solid += 1;
            let far = match extent {
                Extent::Bounded { lo, hi } => outside(p, lo, hi),
                // No box encloses nothing: every solid sample is a violation of an Empty.
                Extent::Empty => f64::INFINITY,
                Extent::Unbounded => 0.0,
            };
            if far > out.worst {
                out.worst = far;
                out.at = Some(p);
                out.value = d;
            }
        }
    };

    // The main grid.
    let n = 50usize;
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                let t = Vector3::new(i as f64, j as f64, k as f64) / n as f64;
                let p = glo + (ghi - glo).component_mul(&t);
                visit(p, &mut out);
            }
        }
    }

    // Sheets just past each face, where a box short by a sag or a margin would be caught: the
    // main grid's spacing is millimetres, and what those miss is hundredths.
    if bounded {
        let m = 110usize;
        let pad = size * 0.02 + Vector3::repeat(0.5);
        for axis in 0..3 {
            let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
            for side in [-1.0f64, 1.0] {
                for off in [1e-4, 0.02, 0.1] {
                    for i in 0..=m {
                        for j in 0..=m {
                            let mut p = Point3::origin();
                            p[axis] = if side < 0.0 { lo[axis] - off } else { hi[axis] + off };
                            p[a] = (lo[a] - pad[a]) + (size[a] + 2.0 * pad[a]) * i as f64 / m as f64;
                            p[b] = (lo[b] - pad[b]) + (size[b] + 2.0 * pad[b]) * j as f64 / m as f64;
                            visit(p, &mut out);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Compiles, surveys, prints what was found, and asserts the property.
fn holds(label: &str, doc: &Document, graph: &Graph, node: NodeId, tolerance: f64) -> Outcome {
    let field = GraphField::compile_graph(doc, graph, tolerance)
        .unwrap_or_else(|e| panic!("{label}: does not compile: {e}"));
    let extent = graph_extent(doc, graph, node);
    let fallback = (Point3::new(-300.0, -300.0, -300.0), Point3::new(300.0, 300.0, 300.0));
    let out = survey(&field, extent, fallback);
    println!(
        "CASE {label}: extent {:?}; {} samples, {} solid; worst outside {:.6} mm at {:?} (reads {:.6})",
        out.extent, out.samples, out.solid, out.worst, out.at, out.value
    );
    assert!(out.solid > 0, "{label}: the survey saw no material at all, so it proves nothing");
    match out.extent {
        Extent::Bounded { .. } | Extent::Empty => assert!(
            out.worst <= 1e-6,
            "{label}: a solid point lies {:.6} mm outside the extent {:?}, at {:?} (field reads {:.6})",
            out.worst,
            out.extent,
            out.at,
            out.value
        ),
        Extent::Unbounded => {}
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Extrude, straight out of the plane
// ---------------------------------------------------------------------------------------------

#[test]
fn a_straight_one_sided_extrude_of_an_off_origin_l_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, 0.0, false);
    holds("normal / one-sided / no draft / L", &doc, &g, n, FINE);
}

#[test]
fn a_straight_symmetric_extrude_of_an_off_origin_l_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, 0.0, true);
    holds("normal / symmetric / no draft / L", &doc, &g, n, FINE);
}

#[test]
fn a_straight_one_sided_extrude_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, 20.0, false);
    holds("normal / one-sided / draft +20 / L", &doc, &g, n, FINE);
}

#[test]
fn a_straight_symmetric_extrude_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, 20.0, true);
    holds("normal / symmetric / draft +20 / L", &doc, &g, n, FINE);
}

#[test]
fn a_straight_one_sided_extrude_drafted_in_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, -10.0, false);
    holds("normal / one-sided / draft -10 / L", &doc, &g, n, FINE);
}

#[test]
fn a_straight_symmetric_extrude_drafted_in_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, -10.0, true);
    holds("normal / symmetric / draft -10 / L", &doc, &g, n, FINE);
}

#[test]
fn a_flipped_extrude_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, -30.0, 20.0, false);
    holds("normal / one-sided / height -30 / draft +20 / L", &doc, &g, n, FINE);
}

#[test]
fn a_straight_drafted_extrude_off_the_ground_plane_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, tilted_l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, 20.0, false);
    holds("normal / one-sided / draft +20 / tilted L", &doc, &g, n, FINE);
}

#[test]
fn a_straight_extrude_of_a_spline_profile_holds_every_solid_point() {
    // Two open curves joined end to end: a cubic bulging out past its control polygon's box in
    // neither direction but past its own end points in y, closed by a straight chord.
    let mut doc = Document::new();
    let arc = spline(
        &mut doc,
        3,
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 30.0, 0.0),
            Point3::new(40.0, 30.0, 0.0),
            Point3::new(50.0, 0.0, 0.0),
        ],
    );
    let _chord = polyline(&mut doc, vec![Point3::new(50.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)], false);
    let (g, n) = extrude(&doc, arc, Sweep::Normal, 25.0, 0.0, false);
    holds("normal / one-sided / no draft / spline + chord loop", &doc, &g, n, FINE);
}

/// The one case compiled at a display tolerance rather than the fine one: here the profile is
/// **baked**, and a baked profile reads short by a margin by design — so a point just past the
/// drawn wall can read solid. The extent is the drawn profile's box exactly, with nothing to
/// spare.
#[test]
#[ignore = "drawn-surface slack of a few tenths of a millimetre or less (bake margin, tessellation, rounding) -- tile-mask-design-01941c5.md 1.8 R7"]
fn a_straight_extrude_baked_at_display_tolerance_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, Sweep::Normal, 30.0, 0.0, false);
    holds("normal / one-sided / no draft / L / tolerance 0.05 (baked)", &doc, &g, n, 0.05);
}

// ---------------------------------------------------------------------------------------------
// Extrude, leaning (Sweep::Toward and Sweep::Along)
// ---------------------------------------------------------------------------------------------

#[test]
fn a_leaning_one_sided_extrude_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(1.0, 0.0, 1.0), 30.0, 0.0, false);
    holds("toward (1,0,1) / one-sided / no draft / L", &doc, &g, n, FINE);
}

#[test]
fn a_leaning_symmetric_extrude_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(1.0, 0.0, 1.0), 30.0, 0.0, true);
    holds("toward (1,0,1) / symmetric / no draft / L", &doc, &g, n, FINE);
}

/// Forty degrees of draft on a forty-five degree lean: the far end's walls have moved out by
/// 30·tan 40° = 25.2 mm, and the lean has carried that end only 30·sin 45° = 21.2 mm across —
/// so the back wall of the far end is 4 mm behind the base.
#[test]
fn a_leaning_one_sided_extrude_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(1.0, 0.0, 1.0), 30.0, 40.0, false);
    holds("toward (1,0,1) / one-sided / draft +40 / L", &doc, &g, n, FINE);
}

/// A gentle lean and an ordinary draft, which is the combination somebody would actually build.
#[test]
fn a_slightly_leaning_one_sided_extrude_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(0.2, 0.0, 1.0), 30.0, 20.0, false);
    holds("toward (0.2,0,1) / one-sided / draft +20 / L", &doc, &g, n, FINE);
}

#[test]
fn a_leaning_symmetric_extrude_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(1.0, 0.0, 1.0), 30.0, 40.0, true);
    holds("toward (1,0,1) / symmetric / draft +40 / L", &doc, &g, n, FINE);
}

#[test]
fn a_leaning_one_sided_extrude_drafted_in_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(1.0, 0.0, 1.0), 30.0, -10.0, false);
    holds("toward (1,0,1) / one-sided / draft -10 / L", &doc, &g, n, FINE);
}

#[test]
fn a_leaning_symmetric_extrude_drafted_in_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(1.0, 0.0, 1.0), 30.0, -10.0, true);
    holds("toward (1,0,1) / symmetric / draft -10 / L", &doc, &g, n, FINE);
}

/// Leaning below the plane: the direction's normal component is negative, so the solid grows
/// down out of the plane although the depth is positive.
#[test]
fn an_extrude_leaning_below_its_plane_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude(&doc, p, toward(-1.0, 1.0, -1.5), 30.0, 30.0, false);
    holds("toward (-1,1,-1.5) / one-sided / draft +30 / L", &doc, &g, n, FINE);
}

#[test]
fn a_leaning_drafted_extrude_off_the_ground_plane_holds_every_solid_point() {
    let mut doc = Document::new();
    let pts = tilted_l_shape();
    let p = polyline(&mut doc, pts.clone(), true);
    // Lean the sweep 45 degrees off the tilted plane's own normal, whatever that is.
    let (_, normal) = fieldform_field::plane_of(&pts).expect("planar");
    let across = normal.cross(&Vector3::x()).normalize();
    let d = normal + across;
    let (g, n) = extrude(&doc, p, toward(d.x, d.y, d.z), 30.0, 40.0, false);
    holds("toward 45deg off normal / one-sided / draft +40 / tilted L", &doc, &g, n, FINE);
}

#[test]
fn an_extrude_along_a_curve_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    // Drawn nowhere near the profile: only its direction counts.
    let axis = polyline(&mut doc, vec![Point3::new(-50.0, -50.0, 0.0), Point3::new(-30.0, -40.0, 30.0)], false);
    let (g, n) = extrude(&doc, p, Sweep::Along { curve: axis }, 30.0, 0.0, false);
    holds("along a line / one-sided / no draft / L", &doc, &g, n, FINE);
}

#[test]
fn an_extrude_along_a_curve_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let axis = polyline(&mut doc, vec![Point3::new(-50.0, -50.0, 0.0), Point3::new(-30.0, -40.0, 30.0)], false);
    let (g, n) = extrude(&doc, p, Sweep::Along { curve: axis }, 30.0, 40.0, false);
    holds("along a line / one-sided / draft +40 / L", &doc, &g, n, FINE);
}

#[test]
fn a_symmetric_extrude_along_a_curve_drafted_out_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let axis = polyline(&mut doc, vec![Point3::new(-50.0, -50.0, 0.0), Point3::new(-30.0, -40.0, 30.0)], false);
    let (g, n) = extrude(&doc, p, Sweep::Along { curve: axis }, 30.0, 40.0, true);
    holds("along a line / symmetric / draft +40 / L", &doc, &g, n, FINE);
}

#[test]
fn an_extrude_along_a_curve_drafted_in_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let axis = polyline(&mut doc, vec![Point3::new(-50.0, -50.0, 0.0), Point3::new(-30.0, -40.0, 30.0)], false);
    let (g, n) = extrude(&doc, p, Sweep::Along { curve: axis }, 30.0, -10.0, false);
    holds("along a line / one-sided / draft -10 / L", &doc, &g, n, FINE);
}

// ---------------------------------------------------------------------------------------------
// Revolve (full turns only: the node has no angle, so there is no partial revolve to build)
// ---------------------------------------------------------------------------------------------

#[test]
fn a_revolved_rectangle_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(
        &mut doc,
        vec![
            Point3::new(20.0, 0.0, -10.0),
            Point3::new(35.0, 0.0, -10.0),
            Point3::new(35.0, 0.0, 15.0),
            Point3::new(20.0, 0.0, 15.0),
        ],
        true,
    );
    let mut g = Graph::empty();
    let n = g.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(p), frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
    holds("revolve / rectangle in XZ / default frame", &doc, &g, n, FINE);
}

#[test]
fn a_revolved_l_about_an_oblique_off_origin_axis_holds_every_solid_point() {
    let mut doc = Document::new();
    let origin = Point3::new(30.0, -20.0, 10.0);
    let axis = Vector3::new(1.0, 1.0, 2.0).normalize();
    let across = axis.cross(&Vector3::x()).normalize();
    // The L in the half-plane (r = across, z = axis) about the frame's origin.
    let pts: Vec<Point3<f64>> = [(10.0, -10.0), (30.0, -10.0), (30.0, 0.0), (18.0, 0.0), (18.0, 20.0), (10.0, 20.0)]
        .iter()
        .map(|&(r, z)| origin + across * r + axis * z)
        .collect();
    let p = polyline(&mut doc, pts, true);
    let mut g = Graph::empty();
    let n = g.add(Node::Revolve { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(p),
        frame: Frame::At {
            origin: [lit(origin.x), lit(origin.y), lit(origin.z)],
            axis: [lit(1.0), lit(1.0), lit(2.0)],
        },
        angle: fieldform_scene::ParamRef::Literal(360.0),
        sides: fieldform_scene::AngleSides::One,
    });
    holds("revolve / L / oblique off-origin axis", &doc, &g, n, FINE);
}

#[test]
fn a_revolved_profile_crossing_its_axis_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(
        &mut doc,
        vec![
            Point3::new(-10.0, 0.0, 0.0),
            Point3::new(25.0, 0.0, 0.0),
            Point3::new(25.0, 0.0, 20.0),
            Point3::new(-10.0, 0.0, 20.0),
        ],
        true,
    );
    let mut g = Graph::empty();
    let n = g.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(p), frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
    holds("revolve / rectangle straddling the axis", &doc, &g, n, FINE);
}

/// A profile made of two open curves joined end to end. The compile walks the loop
/// (`profile_loop`), and so does the extent.
///
/// It read only the curve the node names until a revolve took an outline, and this case was
/// 11.7 mm outside its box (`tile-mask-design-01941c5.md` 1.8, R6). A wall's chain may be several
/// curves too, so the arm had to walk what the compile walks for either kind, and the loop
/// came with it.
#[test]
fn a_revolved_loop_of_two_curves_holds_every_solid_point() {
    let mut doc = Document::new();
    let first = polyline(
        &mut doc,
        vec![Point3::new(10.0, 0.0, 0.0), Point3::new(30.0, 0.0, 0.0), Point3::new(30.0, 0.0, 20.0)],
        false,
    );
    let _second = polyline(
        &mut doc,
        vec![Point3::new(30.0, 0.0, 20.0), Point3::new(45.0, 0.0, 40.0), Point3::new(10.0, 0.0, 40.0), Point3::new(10.0, 0.0, 0.0)],
        false,
    );
    let mut g = Graph::empty();
    let n = g.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(first), frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
    holds("revolve / loop of two open polylines", &doc, &g, n, FINE);
}

/// A true circle, so the ring's outermost radius is reached between tessellation samples.
///
/// One of R7's cases, and closed with R6 by the same change: the extent tessellated the curve
/// for itself, at its own tolerance, and so boxed a different polyline from the one the solid
/// is built on. It reads that one now, and a box of the very points the solid has cannot be
/// short of them.
#[test]
fn a_revolved_circle_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = circle(&mut doc, Point3::new(40.0, 0.0, 0.0), Vector3::x(), Vector3::z(), 10.0, 30.0);
    let mut g = Graph::empty();
    let n = g.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(p), frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
    holds("revolve / rational circle, phase 30deg (torus)", &doc, &g, n, FINE);
}

// ---------------------------------------------------------------------------------------------
// Sweep along a bent spine, each section
// ---------------------------------------------------------------------------------------------

/// A 12 mm square about the origin in the ground plane.
fn small_square(doc: &mut Document) -> CurveId {
    polyline(
        doc,
        vec![
            Point3::new(-6.0, -6.0, 0.0),
            Point3::new(6.0, -6.0, 0.0),
            Point3::new(6.0, 6.0, 0.0),
            Point3::new(-6.0, 6.0, 0.0),
        ],
        true,
    )
}

/// A cubic spine that leaves the profile twenty degrees off its normal and bends over by about
/// thirty more — enough that the three sections differ, not so much that a parallel one lies
/// along the path.
fn bent_spline_spine(doc: &mut Document) -> CurveId {
    spline(
        doc,
        3,
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(15.0, 0.0, 40.0),
            Point3::new(20.0, 10.0, 80.0),
            Point3::new(40.0, 20.0, 110.0),
        ],
    )
}

/// A polyline spine with one corner.
fn bent_polyline_spine(doc: &mut Document) -> CurveId {
    polyline(
        doc,
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 40.0), Point3::new(30.0, 5.0, 80.0)],
        false,
    )
}

fn sweep(profile: CurveId, spine: CurveId, section: Section) -> (Graph, NodeId) {
    let mut g = Graph::empty();
    let n = g.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section, law: Default::default() });
    (g, n)
}

#[test]
fn a_square_section_swept_along_a_bent_spline_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = small_square(&mut doc);
    let s = bent_spline_spine(&mut doc);
    let (g, n) = sweep(p, s, Section::Square);
    holds("sweep / square section / bent cubic spine", &doc, &g, n, FINE);
}

#[test]
fn a_skewed_section_swept_along_a_bent_spline_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = small_square(&mut doc);
    let s = bent_spline_spine(&mut doc);
    let (g, n) = sweep(p, s, Section::Skewed);
    holds("sweep / skewed (flush start) section / bent cubic spine", &doc, &g, n, FINE);
}

#[test]
fn a_parallel_section_swept_along_a_bent_spline_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = small_square(&mut doc);
    let s = bent_spline_spine(&mut doc);
    let (g, n) = sweep(p, s, Section::Parallel);
    holds("sweep / parallel section / bent cubic spine", &doc, &g, n, FINE);
}

#[test]
fn a_square_section_swept_along_a_cornered_polyline_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = small_square(&mut doc);
    let s = bent_polyline_spine(&mut doc);
    let (g, n) = sweep(p, s, Section::Square);
    holds("sweep / square section / cornered polyline spine", &doc, &g, n, FINE);
}

#[test]
fn a_skewed_section_swept_along_a_cornered_polyline_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = small_square(&mut doc);
    let s = bent_polyline_spine(&mut doc);
    let (g, n) = sweep(p, s, Section::Skewed);
    holds("sweep / skewed section / cornered polyline spine", &doc, &g, n, FINE);
}

#[test]
fn a_parallel_section_swept_along_a_cornered_polyline_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = small_square(&mut doc);
    let s = bent_polyline_spine(&mut doc);
    let (g, n) = sweep(p, s, Section::Parallel);
    holds("sweep / parallel section / cornered polyline spine", &doc, &g, n, FINE);
}

/// An off-origin, non-convex profile, with the spine setting off 30 mm below its plane: the
/// sweep is carried up to start at the profile (`sweep_offset`), and so must the box be.
///
/// The spine used to set off 130 mm to one side as well. Since 2026-10-02 a section is carried
/// by the point its path sets off from, so that drew the L on a 130 mm arm round an 80 mm bend
/// -- a fold, refused by name -- where before the L's centroid was quietly put on the path.
#[test]
fn an_off_origin_l_swept_along_a_spine_drawn_elsewhere_holds_every_solid_point() {
    let mut doc = Document::new();
    let pts: Vec<Point3<f64>> = [(90.0, -5.0), (105.0, -5.0), (105.0, 0.0), (95.0, 0.0), (95.0, 8.0), (90.0, 8.0)]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect();
    let p = polyline(&mut doc, pts, true);
    let s = spline(
        &mut doc,
        3,
        vec![
            Point3::new(97.0, 2.0, -30.0),
            Point3::new(97.0, 2.0, 10.0),
            Point3::new(117.0, 12.0, 50.0),
            Point3::new(147.0, 12.0, 70.0),
        ],
    );
    for section in [Section::Square, Section::Skewed, Section::Parallel] {
        let (g, n) = sweep(p, s, section);
        holds(&format!("sweep / {section:?} / off-origin L, spine drawn elsewhere"), &doc, &g, n, FINE);
    }
}

/// **A section that grows on the way must stay in its box.** The extent is the path grown by
/// the section's reach from the point carrying it, and under a taper that reach is not the
/// drawn one: doubled, the section is twice as far from the path at the end, and a draft
/// pushes every wall out by `length · tan(angle)`. A box drawn to the drawn reach clips the
/// solid, which is the one failure an extent must not have.
#[test]
fn a_tapered_or_drafted_sweep_holds_every_solid_point() {
    use fieldform_scene::{SweepLaw, Taper};
    let laws = [
        ("scaled to 2", SweepLaw { taper: Taper::Scale(lit(2.0)), ..Default::default() }),
        ("scaled to 0.4", SweepLaw { taper: Taper::Scale(lit(0.4)), ..Default::default() }),
        ("drafted 8 degrees out", SweepLaw { taper: Taper::Draft(lit(8.0)), ..Default::default() }),
        ("twisted 150 degrees", SweepLaw { twist: lit(150.0), ..Default::default() }),
        (
            "scaled and turned over a window",
            SweepLaw {
                taper: Taper::Scale(lit(1.6)),
                corners: Default::default(),
                twist: lit(90.0),
                shape: fieldform_scene::RampShape::Smooth,
                extent: [lit(15.0), lit(80.0)],
            },
        ),
    ];
    for (what, law) in laws {
        let mut doc = Document::new();
        let p = small_square(&mut doc);
        let s = bent_spline_spine(&mut doc);
        let mut g = Graph::empty();
        let n = g.add(Node::Sweep { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(p),
            spine: s,
            section: Section::Square,
            law,
        });
        holds(&format!("sweep / {what} / bent cubic spine"), &doc, &g, n, FINE);
    }
}

/// A spine that turns back on itself in x, so its widest reach falls between samples of a
/// coarse tessellation rather than at an end.
#[test]
fn a_square_section_swept_round_an_arc_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = small_square(&mut doc);
    // A half circle of radius 50 in XZ, starting straight up out of the profile.
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let control = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 50.0),
        Point3::new(50.0, 0.0, 50.0),
        Point3::new(100.0, 0.0, 50.0),
        Point3::new(100.0, 0.0, 0.0),
    ];
    let weights = vec![1.0, w, 1.0, w, 1.0];
    let knots = vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0];
    let arc = Curve::new(CurveId::new(), "arc", 2, control, weights, knots, false).expect("an arc");
    let s = arc.id;
    doc.curves.insert(s, arc);
    let (g, n) = sweep(p, s, Section::Square);
    holds("sweep / square section / half-circle spine r50", &doc, &g, n, FINE);
}

// ---------------------------------------------------------------------------------------------
// Thin extrudes: a curve given a thickness
// ---------------------------------------------------------------------------------------------

/// An open chain off the origin, with a corner each way and one sharper than a right angle, so
/// the wall has a round, a mitre and two ends for a box to hold.
fn hooked_chain() -> Vec<Point3<f64>> {
    [(20.0, 10.0), (60.0, 10.0), (70.0, 35.0), (40.0, 30.0), (30.0, 55.0)]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn thin(
    curve: CurveId,
    thickness: f64,
    side: fieldform_scene::WallSide,
    ends: fieldform_scene::WallEnds,
    sweep: Sweep,
    height: f64,
    draft: f64,
    symmetric: bool,
) -> (Graph, NodeId) {
    let mut graph = Graph::empty();
    let node = graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::Wall { curve, thickness: lit(thickness), side, ends },
        sweep,
        height: lit(height),
        draft: lit(draft),
        start: lit(0.0),
        sides: fieldform_scene::Sides::symmetric_if(symmetric),
        corners: Default::default(),
    });
    (graph, node)
}

/// A wall's curve is not its reach: the band goes a thickness past it, to one side or both, and
/// round a corner it goes that far in every direction.
#[test]
fn a_thin_extrude_to_either_side_of_a_hooked_chain_holds_every_solid_point() {
    use fieldform_scene::{WallEnds, WallSide};
    let mut doc = Document::new();
    let c = polyline(&mut doc, hooked_chain(), false);
    for (side, ends) in [
        (WallSide::Left, WallEnds::Square),
        (WallSide::Right, WallEnds::Round),
        (WallSide::Centre, WallEnds::Round),
        (WallSide::Centre, WallEnds::Square),
    ] {
        let (g, n) = thin(c, 6.0, side, ends, Sweep::Normal, 30.0, 0.0, false);
        holds(&format!("wall / {side:?} / {ends:?} / hooked chain"), &doc, &g, n, FINE);
    }
}

#[test]
fn a_thin_extrude_drafted_out_and_symmetric_holds_every_solid_point() {
    use fieldform_scene::{WallEnds, WallSide};
    let mut doc = Document::new();
    let c = polyline(&mut doc, hooked_chain(), false);
    let (g, n) = thin(c, 6.0, WallSide::Left, WallEnds::Square, Sweep::Normal, 30.0, 15.0, true);
    holds("wall / left / symmetric / draft +15 / hooked chain", &doc, &g, n, FINE);
    let (g, n) = thin(c, 6.0, WallSide::Centre, WallEnds::Round, Sweep::Normal, -30.0, 15.0, false);
    holds("wall / centre / height -30 / draft +15 / hooked chain", &doc, &g, n, FINE);
}

#[test]
fn a_thin_extrude_off_the_ground_plane_and_leaning_holds_every_solid_point() {
    use fieldform_scene::{WallEnds, WallSide};
    let turn = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(1.0, 2.0, 0.5)), 0.9);
    let shift = Vector3::new(30.0, -40.0, 20.0);
    let mut doc = Document::new();
    let tilted = hooked_chain().into_iter().map(|p| turn * p + shift).collect();
    let c = polyline(&mut doc, tilted, false);
    let (g, n) = thin(c, 6.0, WallSide::Right, WallEnds::Square, Sweep::Normal, 30.0, 0.0, false);
    holds("wall / right / tilted chain", &doc, &g, n, FINE);
    let (g, n) = thin(c, 6.0, WallSide::Left, WallEnds::Round, toward(1.0, 0.3, 2.0), 30.0, 0.0, false);
    holds("wall / left / tilted chain / leaning", &doc, &g, n, FINE);
}

/// A wall is boxed by the arm a profile is, so it had a profile's fault: where a draft outruns
/// the lean, the far end's trailing wall is behind the base's and the box has to reach back
/// for it.
///
/// **About its curve and round-ended, on purpose.** A wall's own box is its chain's grown by
/// the band on every side, which is exact only where the band really does go that far — a
/// round end, the outside of a sharp corner, the flank of a run. A one-sided, square-ended wall
/// has millimetres of slack there, and this fault hid inside them: the same two leans on such
/// a wall passed with the fault still in.
#[test]
fn a_thin_extrude_leaning_and_drafted_out_holds_every_solid_point() {
    use fieldform_scene::{WallEnds, WallSide};
    let mut doc = Document::new();
    let c = polyline(&mut doc, hooked_chain(), false);
    let (g, n) =
        thin(c, 6.0, WallSide::Centre, WallEnds::Round, toward(1.0, 0.0, 1.0), 30.0, 40.0, false);
    holds("wall / centre / round / toward (1,0,1) / draft +40 / hooked chain", &doc, &g, n, FINE);
}

/// The same below the plane, where the walls left behind are the other two.
#[test]
fn a_thin_extrude_leaning_below_its_plane_and_drafted_out_holds_every_solid_point() {
    use fieldform_scene::{WallEnds, WallSide};
    let mut doc = Document::new();
    let c = polyline(&mut doc, hooked_chain(), false);
    let (g, n) =
        thin(c, 6.0, WallSide::Centre, WallEnds::Round, toward(-1.0, 1.0, -1.5), 30.0, 30.0, false);
    holds("wall / centre / round / toward (-1,1,-1.5) / draft +30 / hooked chain", &doc, &g, n, FINE);
}

/// A straight line has no plane to fit, so the box has to be laid in the one the wall was.
#[test]
fn a_thin_extrude_of_a_straight_line_holds_every_solid_point() {
    use fieldform_scene::{WallEnds, WallSide};
    let mut doc = Document::new();
    let line = polyline(
        &mut doc,
        vec![Point3::new(15.0, -20.0, 5.0), Point3::new(65.0, 25.0, 30.0)],
        false,
    );
    let (g, n) = thin(line, 5.0, WallSide::Left, WallEnds::Square, Sweep::Normal, 25.0, 0.0, false);
    holds("wall / left / straight line / upright", &doc, &g, n, FINE);
    let (g, n) = thin(line, 5.0, WallSide::Centre, WallEnds::Round, toward(0.0, 1.0, 0.2), 25.0, 0.0, true);
    holds("wall / centre / straight line / swept sideways", &doc, &g, n, FINE);
}

/// Outside a loop the wall is past the loop's own points by its whole thickness; inside it is
/// not past them at all.
#[test]
fn a_thin_extrude_about_a_closed_loop_holds_every_solid_point() {
    use fieldform_scene::{WallEnds, WallSide};
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    for side in [WallSide::Left, WallSide::Right, WallSide::Centre] {
        let (g, n) = thin(p, 4.0, side, WallEnds::Square, Sweep::Normal, 30.0, 0.0, false);
        holds(&format!("wall / {side:?} / closed L"), &doc, &g, n, FINE);
    }
}

// ---------------------------------------------------------------------------------------------
// Thin revolves and open-section sweeps: the same curve given a thickness, turned and carried
// ---------------------------------------------------------------------------------------------

/// A chain in the half-plane y = 0, clear of the axis, with a corner past a right angle in it.
fn hooked_about_the_axis() -> Vec<Point3<f64>> {
    [(20.0, -10.0), (34.0, -4.0), (30.0, 12.0), (22.0, 4.0)]
        .iter()
        .map(|&(r, z)| Point3::new(r, 0.0, z))
        .collect()
}

/// The band reaches a thickness past the curve, so the ring it turns out is that much wider
/// and taller than the curve's own — to whichever side the wall is.
#[test]
fn a_thin_revolve_holds_every_solid_point() {
    use fieldform_scene::{Outline, WallEnds, WallSide};
    let mut doc = Document::new();
    let c = polyline(&mut doc, hooked_about_the_axis(), false);
    for (side, ends) in [
        (WallSide::Left, WallEnds::Square),
        (WallSide::Right, WallEnds::Round),
        (WallSide::Centre, WallEnds::Round),
    ] {
        let mut g = Graph::empty();
        let n = g.add(Node::Revolve { rims: Default::default(),
            outline: Outline::Wall { curve: c, thickness: lit(5.0), side, ends },
            frame: Frame::default(),
            angle: fieldform_scene::ParamRef::Literal(360.0),
            sides: fieldform_scene::AngleSides::One,
        });
        holds(&format!("thin revolve / {side:?} / {ends:?} / hooked chain"), &doc, &g, n, FINE);
    }
}

/// A wall whose band crosses the axis comes back as radii on this side of it, which is further
/// out than the curve ever goes.
#[test]
fn a_thin_revolve_whose_band_crosses_the_axis_holds_every_solid_point() {
    use fieldform_scene::{Outline, WallEnds, WallSide};
    let mut doc = Document::new();
    let near = polyline(
        &mut doc,
        vec![Point3::new(2.0, 0.0, 0.0), Point3::new(3.0, 0.0, 30.0)],
        false,
    );
    let mut g = Graph::empty();
    let n = g.add(Node::Revolve { rims: Default::default(),
        outline: Outline::Wall {
            curve: near,
            thickness: lit(9.0),
            side: WallSide::Left,
            ends: WallEnds::Square,
        },
        frame: Frame::default(),
        angle: fieldform_scene::ParamRef::Literal(360.0),
        sides: fieldform_scene::AngleSides::One,
    });
    holds("thin revolve / left / band across the axis", &doc, &g, n, FINE);
}

/// An open section carried round a bend: the path grown by how far the band can be from the
/// section's own origin, which is a curve point and a thickness beyond it.
#[test]
fn an_open_section_sweep_holds_every_solid_point() {
    use fieldform_scene::{Outline, WallEnds, WallSide};
    let mut doc = Document::new();
    let section = polyline(
        &mut doc,
        vec![
            Point3::new(0.0, -6.0, -4.0),
            Point3::new(0.0, 4.0, -5.0),
            Point3::new(0.0, 7.0, 3.0),
            Point3::new(0.0, 1.0, 1.0),
        ],
        false,
    );
    let path = polyline(
        &mut doc,
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(30.0, 4.0, 0.0),
            Point3::new(58.0, 16.0, 9.0),
            Point3::new(80.0, 36.0, 24.0),
        ],
        false,
    );
    for (side, ends) in [(WallSide::Right, WallEnds::Square), (WallSide::Centre, WallEnds::Round)] {
        let mut g = Graph::empty();
        let n = g.add(Node::Sweep { rims: Default::default(),
            outline: Outline::Wall { curve: section, thickness: lit(2.0), side, ends },
            spine: path,
            section: Section::Square,
            law: Default::default(),
        });
        holds(&format!("open-section sweep / {side:?} / {ends:?}"), &doc, &g, n, FINE);
    }
}

// ---------------------------------------------------------------------------------------------
// Extrudes with a start, and with two sides (`extrude-revolve-options-spec.md` §2)
// ---------------------------------------------------------------------------------------------

fn extrude_sides(
    profile: CurveId,
    sweep: Sweep,
    start: f64,
    height: f64,
    draft: f64,
    sides: fieldform_scene::Sides,
) -> (Graph, NodeId) {
    let mut graph = Graph::empty();
    let node = graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(profile),
        sweep,
        start: lit(start),
        height: lit(height),
        draft: lit(draft),
        sides,
        corners: Default::default(),
    });
    (graph, node)
}

fn two(height: f64, draft: f64) -> fieldform_scene::Sides {
    fieldform_scene::Sides::Two { height: lit(height), draft: lit(draft) }
}

/// A start carries the whole solid, flare and all, along the sweep -- so a box that forgot the
/// start would sit on the plane with the solid ten millimetres above it.
#[test]
fn an_extrude_begun_off_its_plane_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude_sides(p, Sweep::Normal, 10.0, 30.0, 20.0, fieldform_scene::Sides::One);
    holds("normal / start 10 / one-sided / draft +20 / L", &doc, &g, n, FINE);
    let (g, n) = extrude_sides(p, toward(1.0, 0.0, 1.0), 10.0, 30.0, 40.0, fieldform_scene::Sides::One);
    holds("toward (1,0,1) / start 10 / one-sided / draft +40 / L", &doc, &g, n, FINE);
    // And a start behind the plane, with the depth running back past it.
    let (g, n) = extrude_sides(p, Sweep::Normal, -25.0, 30.0, 20.0, fieldform_scene::Sides::Symmetric);
    holds("normal / start -25 / symmetric / draft +20 / L", &doc, &g, n, FINE);
}

/// Two sides each flare by their own depth and draft, so the box has to be padded by each
/// side's own amount at its own end -- and along a lean each end is where the lean carries it.
#[test]
fn a_two_sided_extrude_drafted_differently_holds_every_solid_point() {
    let mut doc = Document::new();
    let p = polyline(&mut doc, l_shape(), true);
    let (g, n) = extrude_sides(p, Sweep::Normal, 0.0, 20.0, 30.0, two(12.0, 10.0));
    holds("normal / two sides 20 and 12 / drafts +30 and +10 / L", &doc, &g, n, FINE);
    let (g, n) = extrude_sides(p, Sweep::Normal, 5.0, -20.0, 10.0, two(12.0, 35.0));
    holds("normal / start 5 / height -20 / two sides / drafts +10 and +35 / L", &doc, &g, n, FINE);
    let (g, n) = extrude_sides(p, toward(1.0, 0.0, 1.0), 4.0, 20.0, 25.0, two(12.0, 40.0));
    holds("toward (1,0,1) / start 4 / two sides / drafts +25 and +40 / L", &doc, &g, n, FINE);
    let (g, n) = extrude_sides(p, toward(-1.0, 1.0, -1.5), 0.0, 30.0, 30.0, two(15.0, -10.0));
    holds("toward (-1,1,-1.5) / two sides / drafts +30 and -10 / L", &doc, &g, n, FINE);
}
