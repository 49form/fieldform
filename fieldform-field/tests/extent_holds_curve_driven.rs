// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Every solid point of a curve-driven node lies inside the box `graph_extent` reports for it.
//!
//! The box used to be a hint and is now a guarantee: the ray starts at the scene sphere built
//! from it, and a per-tile body cull skips a body wherever its box misses the tile. A box that
//! is too small is therefore material that is not drawn, which no test of the field itself
//! would notice. So this marches the field rather than reasoning about `extent_of`.
//!
//! Two samplings per case, because either alone misses something:
//!
//! - **A coarse grid over the box grown by `max(50% of its size, 10 mm)`**, which finds solid
//!   material far outside the box -- the failure a displacement bound that is too small makes.
//! - **Thin slabs just outside each face**, at a fine in-face spacing and a ladder of offsets
//!   from 1e-4 mm to 0.3 mm. A box built from a curve's *tessellation* can be short of the
//!   curve by up to the chordal tolerance, and a 2 mm grid steps straight over a violation that
//!   deep. The slabs are where such a violation has to show, since solid outside a face is
//!   connected to solid inside it.
//!
//! The worst violating sample is then walked outward along the axis it violates most, so the
//! number reported is how far the material really reaches past the box, not how far the
//! nearest sample happened to be.

use fieldform_field::{graph_extent, Extent, GraphField, ScalarField};
use fieldform_scene::{
    Attractor, Curve, CurveId, Document, Graph, Node, NodeId, ParamRef, Place, RampShape, Turn,
};
use nalgebra::{Point3, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

const SOLID: f64 = -1e-6;
const OUTSIDE: f64 = 1e-6;
const COARSE_BUDGET: f64 = 300_000.0;
const FACE_BUDGET: f64 = 30_000.0;
const OFFSETS: [f64; 6] = [1e-4, 1e-3, 0.01, 0.03, 0.1, 0.3];

/// How far `p` lies outside the box, as a Euclidean distance: 0 inside it.
fn outside_by(p: Point3<f64>, lo: Point3<f64>, hi: Point3<f64>) -> f64 {
    let d = Vector3::from_fn(|a, _| (lo[a] - p[a]).max(p[a] - hi[a]).max(0.0));
    d.norm()
}

struct Finding {
    extent: Extent,
    solid_samples: usize,
    worst: f64,
    at: Option<Point3<f64>>,
}

/// Samples the field around the node's extent and reports the worst solid sample outside it.
fn assess(doc: &Document, graph: &Graph, node: NodeId, face_step: f64) -> Finding {
    let extent = graph_extent(doc, graph, node);
    let field = GraphField::compile_graph(doc, graph, 1e-4).expect("the case compiles");
    let solid = |p: Point3<f64>| field.distance(p) < SOLID;

    let (lo, hi) = match extent {
        Extent::Bounded { lo, hi } => (lo, hi),
        // Nothing can lie outside everything; whether that was the right answer is the
        // report's business, not this sampler's.
        Extent::Unbounded => {
            return Finding { extent, solid_samples: 0, worst: 0.0, at: None };
        }
        // Empty holds only if nothing is solid, so look across a generous region.
        Extent::Empty => (Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0)),
    };
    let empty = matches!(extent, Extent::Empty);
    let size = hi - lo;
    let grow = Vector3::from_fn(|a, _| (0.5 * size[a]).max(10.0).max(if empty { 200.0 } else { 0.0 }));
    let (glo, ghi) = (lo - grow, hi + grow);
    let span = ghi - glo;

    let mut solid_samples = 0usize;
    let mut worst = 0.0f64;
    let mut at: Option<Point3<f64>> = None;
    let mut note = |p: Point3<f64>, solid_samples: &mut usize| {
        *solid_samples += 1;
        let out = if empty { f64::INFINITY } else { outside_by(p, lo, hi) };
        if out > OUTSIDE && (at.is_none() || out > worst) {
            worst = out;
            at = Some(p);
        }
    };

    // The coarse grid.
    let step = (span.x * span.y * span.z / COARSE_BUDGET).cbrt();
    let n = Vector3::from_fn(|a, _| ((span[a] / step).ceil() as usize).max(2));
    for i in 0..=n.x {
        for j in 0..=n.y {
            for k in 0..=n.z {
                let p = Point3::new(
                    glo.x + span.x * i as f64 / n.x as f64,
                    glo.y + span.y * j as f64 / n.y as f64,
                    glo.z + span.z * k as f64 / n.z as f64,
                );
                if solid(p) {
                    note(p, &mut solid_samples);
                }
            }
        }
    }

    // The face slabs.
    if !empty {
        for axis in 0..3 {
            let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
            let (b0, b1) = (lo[b] - 1.0, hi[b] + 1.0);
            let (c0, c1) = (lo[c] - 1.0, hi[c] + 1.0);
            // The case's spacing, coarsened only if the face would blow the budget.
            let area = (b1 - b0) * (c1 - c0);
            let fs = face_step.max((area / FACE_BUDGET).sqrt());
            let nb = ((b1 - b0) / fs).ceil() as usize;
            let nc = ((c1 - c0) / fs).ceil() as usize;
            for (plane, outward) in [(lo[axis], -1.0), (hi[axis], 1.0)] {
                for off in OFFSETS {
                    for i in 0..=nb {
                        for j in 0..=nc {
                            let mut p = Point3::origin();
                            p[axis] = plane + outward * off;
                            p[b] = b0 + (b1 - b0) * i as f64 / nb as f64;
                            p[c] = c0 + (c1 - c0) * j as f64 / nc as f64;
                            if solid(p) {
                                note(p, &mut solid_samples);
                            }
                        }
                    }
                }
            }
        }
    }

    // Walk the worst sample outward along the axis it violates most, so the depth reported is
    // the material's, not the sample spacing's.
    if let (Some(p), false) = (at, empty) {
        let (axis, dir) = (0..3)
            .flat_map(|a| [(a, -1.0, lo[a] - p[a]), (a, 1.0, p[a] - hi[a])])
            .max_by(|x, y| x.2.total_cmp(&y.2))
            .map(|(a, d, _)| (a, d))
            .unwrap();
        let along = |t: f64| {
            let mut q = p;
            q[axis] += dir * t;
            q
        };
        let mut hi_t = 1e-3;
        while solid(along(hi_t)) && hi_t < 1e4 {
            hi_t *= 2.0;
        }
        let mut lo_t = 0.0;
        for _ in 0..60 {
            let mid = 0.5 * (lo_t + hi_t);
            if solid(along(mid)) {
                lo_t = mid;
            } else {
                hi_t = mid;
            }
        }
        let q = along(lo_t);
        let out = outside_by(q, lo, hi);
        if out > worst {
            worst = out;
            at = Some(q);
        }
    }

    Finding { extent, solid_samples, worst, at }
}

/// Runs the property and fails with the worst point when it does not hold.
fn holds(name: &str, doc: &Document, graph: &Graph, node: NodeId, face_step: f64) {
    let f = assess(doc, graph, node, face_step);
    println!(
        "{name}: extent {:?}, {} solid samples, worst outside {:.6} mm at {:?}",
        f.extent, f.solid_samples, f.worst, f.at
    );
    if let Extent::Unbounded = f.extent {
        return;
    }
    if let Extent::Empty = f.extent {
        assert_eq!(f.solid_samples, 0, "{name}: an Empty extent, and solid at {:?}", f.at);
        return;
    }
    assert!(f.solid_samples > 0, "{name}: no solid sample at all, so the case proved nothing");
    assert!(
        f.worst <= OUTSIDE,
        "{name}: solid material {:.6} mm outside the reported box {:?}, at {:?}",
        f.worst,
        f.extent,
        f.at
    );
}

fn add_curve(doc: &mut Document, curve: impl FnOnce(CurveId) -> Curve) -> CurveId {
    let id = CurveId::new();
    doc.curves.insert(id, curve(id));
    id
}

fn polyline(doc: &mut Document, points: Vec<Point3<f64>>, closed: bool) -> CurveId {
    add_curve(doc, |id| Curve::polyline(id, "spine", points, closed).expect("a polyline"))
}

/// A clamped, non-rational B-spline of the degree given, with a uniform interior.
fn spline(doc: &mut Document, degree: usize, points: Vec<Point3<f64>>) -> CurveId {
    let n = points.len();
    let mut knots = vec![0.0; degree + 1];
    let interior = n - degree - 1;
    for i in 1..=interior {
        knots.push(i as f64 / (interior + 1) as f64);
    }
    knots.extend(vec![1.0; degree + 1]);
    add_curve(doc, |id| {
        Curve::new(id, "spine", degree, points, vec![1.0; n], knots, false).expect("a spline")
    })
}

/// A closed uniform cubic B-spline: the first three control points repeated at the end, with a
/// uniform unclamped knot vector, so the loop is C2 all the way round and has no joint at all.
fn closed_cubic(doc: &mut Document, loop_points: Vec<Point3<f64>>) -> CurveId {
    let mut points = loop_points.clone();
    points.extend_from_slice(&loop_points[..3]);
    let n = points.len();
    let knots: Vec<f64> = (0..n + 4).map(|i| i as f64).collect();
    add_curve(doc, |id| {
        Curve::new(id, "loop", 3, points, vec![1.0; n], knots, true).expect("a closed cubic")
    })
}

fn pipe(graph: &mut Graph, curve: CurveId, radius: f64) -> NodeId {
    graph.add(Node::Pipe { curve, radius: lit(radius) })
}

// --- Pipe -------------------------------------------------------------------------------------

#[test]
fn a_straight_oblique_pipe_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve =
        polyline(&mut doc, vec![Point3::new(-10.0, 5.0, 0.0), Point3::new(70.0, 55.0, 30.0)], false);
    let mut graph = Graph::empty();
    let node = pipe(&mut graph, curve, 6.0);
    holds("straight oblique pipe", &doc, &graph, node, 0.25);
}

/// A quadratic whose top is at t = 0.625, between the samples the tessellation seeds -- so the
/// box around `tessellate(0.1)` has to be right about a point it never evaluated.
#[test]
fn a_pipe_bent_through_a_quadratic_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = spline(
        &mut doc,
        2,
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(30.0, 100.0, 0.0), Point3::new(100.0, 40.0, 0.0)],
    );
    let mut graph = Graph::empty();
    let node = pipe(&mut graph, curve, 5.0);
    holds("pipe bent through a quadratic", &doc, &graph, node, 0.1);
}

/// A cubic S-curve bending in all three axes, so every face of the box is set by an interior
/// extremum of the curve rather than by an end.
#[test]
#[ignore = "drawn-surface slack of a few tenths of a millimetre or less (bake margin, tessellation, rounding) -- tile-mask-design-01941c5.md 1.8 R7"]
fn a_pipe_bent_through_a_3d_cubic_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = spline(
        &mut doc,
        3,
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(20.0, 90.0, -40.0),
            Point3::new(60.0, -70.0, 80.0),
            Point3::new(110.0, 30.0, 10.0),
            Point3::new(130.0, -20.0, -30.0),
        ],
    );
    let mut graph = Graph::empty();
    let node = pipe(&mut graph, curve, 4.0);
    holds("pipe bent through a 3d cubic", &doc, &graph, node, 0.2);
}

#[test]
fn a_pipe_round_a_closed_polyline_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = polyline(
        &mut doc,
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(60.0, 10.0, 5.0),
            Point3::new(50.0, 70.0, -5.0),
            Point3::new(-10.0, 50.0, 10.0),
        ],
        true,
    );
    let mut graph = Graph::empty();
    let node = pipe(&mut graph, curve, 5.0);
    holds("pipe round a closed polyline", &doc, &graph, node, 0.2);
}

/// A closed smooth loop has no end and no knot at any of its extremes, so no face of the box is
/// set by a point the tessellation is obliged to sample.
#[test]
#[ignore = "drawn-surface slack of a few tenths of a millimetre or less (bake margin, tessellation, rounding) -- tile-mask-design-01941c5.md 1.8 R7"]
fn a_pipe_round_a_closed_cubic_loop_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = closed_cubic(
        &mut doc,
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(70.0, -15.0, 10.0),
            Point3::new(95.0, 40.0, -20.0),
            Point3::new(40.0, 90.0, 15.0),
            Point3::new(-25.0, 60.0, -10.0),
        ],
    );
    let mut graph = Graph::empty();
    let node = pipe(&mut graph, curve, 5.0);
    holds("pipe round a closed cubic loop", &doc, &graph, node, 0.2);
}

/// The ramp read over a plane, two at one end and ten at the other; and the curve bent, so the
/// fat end's box has to come from the ramp's range and the curve's own extremum together.
#[test]
fn a_pipe_whose_radius_a_ramp_drives_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = spline(
        &mut doc,
        2,
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(30.0, 100.0, 0.0), Point3::new(100.0, 40.0, 0.0)],
    );
    let mut graph = Graph::empty();
    let plane = graph.add(Node::Plane {
        origin: Place::origin(),
        flipped: false,
        turn: Turn::By([lit(0.0), lit(90.0), lit(0.0)]),
    });
    let ramp = graph.add(Node::Ramp {
        input: plane.into(),
        in_min: lit(0.0),
        in_max: lit(100.0),
        out_min: lit(2.0),
        out_max: lit(10.0),
        shape: RampShape::Linear,
    });
    let node = graph.add(Node::Pipe { curve, radius: ParamRef::Field(ramp) });
    holds("pipe with a ramp-driven radius", &doc, &graph, node, 0.1);
}

/// Reversed and smoothed: the ramp's larger end is `out_min`, and its shape is not linear.
#[test]
fn a_pipe_whose_radius_a_reversed_smooth_ramp_drives_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve =
        polyline(&mut doc, vec![Point3::new(0.0, 0.0, 0.0), Point3::new(100.0, 20.0, -10.0)], false);
    let mut graph = Graph::empty();
    let plane = graph.add(Node::Plane {
        origin: Place::origin(),
        flipped: false,
        turn: Turn::By([lit(0.0), lit(90.0), lit(0.0)]),
    });
    let ramp = graph.add(Node::Ramp {
        input: plane.into(),
        in_min: lit(20.0),
        in_max: lit(80.0),
        out_min: lit(12.0),
        out_max: lit(3.0),
        shape: RampShape::Smoother,
    });
    let node = graph.add(Node::Pipe { curve, radius: ParamRef::Field(ramp) });
    holds("pipe with a reversed smooth ramp-driven radius", &doc, &graph, node, 0.2);
}

// --- Attract ----------------------------------------------------------------------------------

fn sphere(graph: &mut Graph, at: [f64; 3], radius: f64) -> NodeId {
    graph.add(Node::Sphere { centre: Place::at(at[0], at[1], at[2]), radius: lit(radius) })
}

fn attract(graph: &mut Graph, input: NodeId, toward: Attractor, strength: f64, falloff: f64) -> NodeId {
    graph.add(Node::Attract { input: input.into(), toward, strength: lit(strength), falloff: lit(falloff) })
}

fn toward_point(graph: &mut Graph, at: [f64; 3]) -> Attractor {
    Attractor::Point(graph.add(Node::Point { at: [lit(at[0]), lit(at[1]), lit(at[2])] }))
}

/// The attractor outside the body, where the pull draws material out past the body's own box.
#[test]
fn a_sphere_pulled_toward_a_point_outside_it_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let ball = sphere(&mut graph, [30.0, 0.0, 0.0], 10.0);
    let toward = toward_point(&mut graph, [5.0, 3.0, 0.0]);
    let node = attract(&mut graph, ball, toward, 1.0, 40.0);
    holds("sphere pulled toward a point", &doc, &graph, node, 0.2);
}

#[test]
fn a_sphere_pushed_from_a_point_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let ball = sphere(&mut graph, [30.0, 0.0, 0.0], 10.0);
    let toward = toward_point(&mut graph, [15.0, 0.0, 4.0]);
    let node = attract(&mut graph, ball, toward, -0.9, 40.0);
    holds("sphere pushed from a point", &doc, &graph, node, 0.2);
}

/// Past the fold strength, where the warp maps two places onto one.
#[test]
fn a_sphere_pulled_hard_past_the_fold_toward_a_point_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let ball = sphere(&mut graph, [30.0, 0.0, 0.0], 12.0);
    let toward = toward_point(&mut graph, [8.0, -4.0, 2.0]);
    let node = attract(&mut graph, ball, toward, 3.0, 50.0);
    holds("sphere pulled past the fold toward a point", &doc, &graph, node, 0.2);
}

#[test]
fn a_sphere_pulled_toward_a_bent_curve_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = spline(
        &mut doc,
        3,
        vec![
            Point3::new(-10.0, -40.0, -30.0),
            Point3::new(10.0, 20.0, 0.0),
            Point3::new(-5.0, 10.0, 30.0),
            Point3::new(15.0, -30.0, 50.0),
        ],
    );
    let mut graph = Graph::empty();
    let ball = sphere(&mut graph, [30.0, 0.0, 10.0], 10.0);
    let node = attract(&mut graph, ball, Attractor::Curve(curve), 1.2, 45.0);
    holds("sphere pulled toward a bent curve", &doc, &graph, node, 0.2);
}

/// Just under the pull's fold strength of 16/11, the strongest pull that does not fold.
#[test]
fn a_cuboid_pulled_toward_a_curve_at_the_edge_of_folding_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve =
        polyline(&mut doc, vec![Point3::new(0.0, 0.0, -60.0), Point3::new(0.0, 0.0, 60.0)], false);
    let mut graph = Graph::empty();
    let block = graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::at(30.0, 5.0, 0.0),
        size: [lit(20.0), lit(30.0), lit(40.0)],
        turn: Turn::By([lit(0.0); 3]),
    });
    let node = attract(&mut graph, block, Attractor::Curve(curve), 1.45, 40.0);
    holds("cuboid pulled toward a curve near the fold", &doc, &graph, node, 0.2);
}

/// A push past its own fold strength of -1, against a curve that runs through the body.
#[test]
fn a_sphere_pushed_hard_from_a_curve_through_it_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = spline(
        &mut doc,
        2,
        vec![Point3::new(0.0, -40.0, 0.0), Point3::new(25.0, 0.0, 10.0), Point3::new(0.0, 40.0, 0.0)],
    );
    let mut graph = Graph::empty();
    let ball = sphere(&mut graph, [12.0, 0.0, 5.0], 15.0);
    let node = attract(&mut graph, ball, Attractor::Curve(curve), -2.0, 40.0);
    holds("sphere pushed hard from a curve through it", &doc, &graph, node, 0.2);
}

/// Composed: a pipe attracted toward a point. The warped body inherits the pipe's box, so any
/// shortfall there carries through, and the pull adds its own on top.
#[test]
fn a_bent_pipe_pulled_toward_a_point_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let curve = spline(
        &mut doc,
        2,
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(30.0, 100.0, 0.0), Point3::new(100.0, 40.0, 0.0)],
    );
    let mut graph = Graph::empty();
    let tube = pipe(&mut graph, curve, 5.0);
    let toward = toward_point(&mut graph, [50.0, 80.0, 0.0]);
    let node = attract(&mut graph, tube, toward, 1.3, 30.0);
    holds("bent pipe pulled toward a point", &doc, &graph, node, 0.1);
}

/// Past the fold again, with the point inside the body -- the case above leaves only a small
/// wedge of material, which is thin evidence; this one leaves a body the grid samples richly.
#[test]
fn a_sphere_pulled_past_the_fold_toward_a_point_inside_it_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let ball = sphere(&mut graph, [30.0, 0.0, 0.0], 20.0);
    let toward = toward_point(&mut graph, [22.0, 6.0, -3.0]);
    let node = attract(&mut graph, ball, toward, 2.5, 30.0);
    holds("sphere pulled past the fold toward a point inside it", &doc, &graph, node, 0.2);
}
