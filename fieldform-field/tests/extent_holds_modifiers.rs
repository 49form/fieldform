// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Every modifier's extent holds every point its field calls solid.
//!
//! **A box that is too small is missing material**, not a loose framing: the GPU ray starts at
//! the scene sphere built from `graph_extent`, and a per-tile body cull treats the box as a
//! guarantee. So each case here builds one modifier over a known body, samples a regular grid
//! over its extent grown by half its size (at least 10 mm) on every side, and fails if any
//! clearly solid sample (`distance < -1e-6`) lies more than 1e-6 mm outside the box.
//!
//! The modifiers are the nodes whose box is derived from their input's: offset, shell,
//! placement, mirror and twist. Each case names the property it holds, and a case that fails is
//! left failing — it is a report, not something to paper over here.

use fieldform_field::{graph_extent, Extent, GraphField, ScalarField};
use fieldform_scene::{
    no_turn, Axis, Document, Frame, Graph, Node, NodeId, ParamRef, Place, RampShape,
};
use nalgebra::{Point3, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn at(x: f64, y: f64, z: f64) -> Place {
    Place::At([lit(x), lit(y), lit(z)])
}

fn cuboid(graph: &mut Graph, centre: [f64; 3], size: [f64; 3]) -> NodeId {
    graph.add(Node::Cuboid { edge: Default::default(),
        centre: at(centre[0], centre[1], centre[2]),
        size: size.map(lit),
        turn: no_turn(),
    })
}

fn sphere(graph: &mut Graph, centre: [f64; 3], radius: f64) -> NodeId {
    graph.add(Node::Sphere { centre: at(centre[0], centre[1], centre[2]), radius: lit(radius) })
}

/// What a grid over the grown extent found.
struct Verdict {
    extent: Extent,
    /// How far outside the box the worst solid sample lies, in millimetres.
    worst: f64,
    worst_at: Option<Point3<f64>>,
    worst_value: f64,
    solid: usize,
    samples: usize,
}

/// The Euclidean distance from `p` to the box, nought inside it.
fn outside(p: Point3<f64>, lo: Point3<f64>, hi: Point3<f64>) -> f64 {
    let d = Vector3::new(
        (lo.x - p.x).max(p.x - hi.x).max(0.0),
        (lo.y - p.y).max(p.y - hi.y).max(0.0),
        (lo.z - p.z).max(p.z - hi.z).max(0.0),
    );
    d.norm()
}

/// Samples the node's field over its extent grown by `max(50 %, 10 mm)` per side, `n` samples
/// per axis. An extent with no box (empty or unbounded) is sampled over `fallback` instead, so
/// an `Empty` that hides material still shows up as solid samples.
fn measure(doc: &Document, graph: &Graph, fallback: (Point3<f64>, Point3<f64>), n: usize) -> Verdict {
    let roots = graph.roots();
    assert_eq!(roots.len(), 1, "each case is one body");
    let node = roots[0];
    let extent = graph_extent(doc, graph, node);
    let field = GraphField::compile_graph(doc, graph, 1e-4).expect("the case compiles");
    let (lo, hi) = extent.as_box().unwrap_or(fallback);
    let size = hi - lo;
    let pad = size.map(|s| (0.5 * s).max(10.0));
    let (glo, ghi) = (lo - pad, hi + pad);
    let mut v = Verdict {
        extent,
        worst: 0.0,
        worst_at: None,
        worst_value: 0.0,
        solid: 0,
        samples: 0,
    };
    let step = (ghi - glo) / (n - 1) as f64;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = Point3::new(
                    glo.x + step.x * i as f64,
                    glo.y + step.y * j as f64,
                    glo.z + step.z * k as f64,
                );
                v.samples += 1;
                let d = field.distance(p);
                if d < -1e-6 {
                    v.solid += 1;
                    if let Extent::Bounded { lo, hi } = extent {
                        let out = outside(p, lo, hi);
                        if out > v.worst {
                            v.worst = out;
                            v.worst_at = Some(p);
                            v.worst_value = d;
                        }
                    }
                }
            }
        }
    }
    v
}

/// Prints the verdict and asserts the property. `Unbounded` holds trivially; `Empty` holds only
/// if nothing was solid; `Bounded` holds if no solid sample lies outside it.
fn holds(case: &str, v: Verdict) {
    println!(
        "CASE {case}: extent {:?}; {} of {} samples solid; worst outside {:.6} mm at {:?} (field {:.6})",
        v.extent, v.solid, v.samples, v.worst, v.worst_at.map(|p| [p.x, p.y, p.z]), v.worst_value,
    );
    match v.extent {
        Extent::Unbounded => {}
        Extent::Empty => assert_eq!(v.solid, 0, "{case}: an Empty extent over solid material"),
        Extent::Bounded { .. } => assert!(
            v.worst <= 1e-6,
            "{case}: solid {:.6} mm outside the extent at {:?} (field {:.6}); extent {:?}",
            v.worst,
            v.worst_at,
            v.worst_value,
            v.extent,
        ),
    }
}

fn around_origin(r: f64) -> (Point3<f64>, Point3<f64>) {
    (Point3::new(-r, -r, -r), Point3::new(r, r, r))
}

const N: usize = 64;

// ---------------------------------------------------------------------------------------------
// Offset

#[test]
fn a_grown_cuboid_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, [5.0, -3.0, 2.0], [40.0, 20.0, 10.0]);
    graph.add(Node::Offset { input: body.into(), distance: lit(6.0) });
    holds("offset +6 of a cuboid", measure(&Document::new(), &graph, around_origin(50.0), N));
}

#[test]
fn a_shrunk_cuboid_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, [5.0, -3.0, 2.0], [40.0, 20.0, 10.0]);
    graph.add(Node::Offset { input: body.into(), distance: lit(-3.0) });
    holds("offset -3 of a cuboid", measure(&Document::new(), &graph, around_origin(50.0), N));
}

/// Eroded past its own half-thickness, the body is gone. The extent is the input's box rather
/// than `Empty`, which is loose but holds; this case says which it is.
#[test]
fn an_eroded_away_sphere_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = sphere(&mut graph, [4.0, 0.0, -2.0], 10.0);
    graph.add(Node::Offset { input: body.into(), distance: lit(-15.0) });
    let v = measure(&Document::new(), &graph, around_origin(50.0), N);
    assert_eq!(v.solid, 0, "an offset past the radius leaves nothing");
    holds("offset -15 of a 10 mm sphere (eroded away)", v);
}

/// The ramp reads a plane at z = -20 facing +Z, over `[0, 40]`, so the offset grows with height.
fn driven_offset(out_min: f64, out_max: f64, shape: RampShape) -> Graph {
    let mut graph = Graph::empty();
    let plane = graph.add(Node::Plane { origin: at(0.0, 0.0, -20.0), flipped: false, turn: no_turn() });
    let ramp = graph.add(Node::Ramp {
        input: plane.into(),
        in_min: lit(0.0),
        in_max: lit(40.0),
        out_min: lit(out_min),
        out_max: lit(out_max),
        shape,
    });
    let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 40.0, 40.0]);
    graph.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });
    graph
}

#[test]
fn a_ramp_driven_offset_s_extent_holds_every_solid_point() {
    for shape in RampShape::ALL {
        let graph = driven_offset(0.0, 10.0, shape);
        holds(
            &format!("offset driven by a {shape:?} ramp 0..10"),
            measure(&Document::new(), &graph, around_origin(60.0), N),
        );
    }
}

/// The ramp run backwards, and reaching into erosion at one end: the grow is the larger end.
#[test]
fn a_reversed_ramp_driven_offset_s_extent_holds_every_solid_point() {
    let graph = driven_offset(12.0, -5.0, RampShape::Linear);
    holds(
        "offset driven by a reversed ramp 12..-5",
        measure(&Document::new(), &graph, around_origin(60.0), N),
    );
}

/// **An offset of a field that understates its distance reaches further than the offset.** A
/// twist divides by its stretch, so `F/stretch − d < 0` wherever the true distance is under
/// `d · stretch` — and the extent grows the twisted box by `d` alone.
#[test]
fn a_grown_twisted_body_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 12.0, 60.0]);
    let twisted = graph.add(Node::Twist {
        input: body.into(),
        degrees_per_mm: lit(10.0),
        frame: Frame::along(Axis::Z),
    });
    graph.add(Node::Offset { input: twisted.into(), distance: lit(8.0) });
    holds(
        "offset +8 of a cuboid twisted 10 deg/mm (LowerBound operand)",
        measure(&Document::new(), &graph, around_origin(80.0), N),
    );
}

/// The same fact through the other divisor an offset can meet: a *driven* offset is divided by
/// `1 + |ramp slope|`, so a fixed offset on top of it reaches `d · divisor` past the driven body.
#[test]
fn a_grown_driven_offset_s_extent_holds_every_solid_point() {
    let mut graph = driven_offset(0.0, 10.0, RampShape::Linear);
    let inner = graph.roots()[0];
    graph.add(Node::Offset { input: inner.into(), distance: lit(20.0) });
    holds(
        "offset +20 of a ramp-driven offset (LowerBound operand)",
        measure(&Document::new(), &graph, around_origin(80.0), N),
    );
}

/// A half-space offset is still a half-space, so `Unbounded` is the right answer here and not a
/// lazy one.
#[test]
fn a_grown_plane_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let plane = graph.add(Node::Plane { origin: at(0.0, 0.0, 0.0), flipped: false, turn: no_turn() });
    graph.add(Node::Offset { input: plane.into(), distance: lit(5.0) });
    let v = measure(&Document::new(), &graph, around_origin(30.0), 16);
    assert!(v.extent.is_unbounded());
    holds("offset +5 of a plane", v);
}

// ---------------------------------------------------------------------------------------------
// Shell

#[test]
fn a_shelled_cuboid_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, [3.0, 1.0, -2.0], [30.0, 24.0, 18.0]);
    graph.add(Node::Shell { input: body.into(), thickness: lit(2.0) });
    holds("shell 2 of a cuboid", measure(&Document::new(), &graph, around_origin(50.0), 96));
}

#[test]
fn a_shelled_sphere_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = sphere(&mut graph, [-6.0, 2.0, 0.0], 15.0);
    graph.add(Node::Shell { input: body.into(), thickness: lit(3.0) });
    holds("shell 3 of a sphere", measure(&Document::new(), &graph, around_origin(50.0), 96));
}

/// A negative wall: the cavity is the body *grown*, so nothing is left.
#[test]
fn a_negatively_shelled_cuboid_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [30.0, 24.0, 18.0]);
    graph.add(Node::Shell { input: body.into(), thickness: lit(-4.0) });
    holds("shell -4 of a cuboid", measure(&Document::new(), &graph, around_origin(50.0), N));
}

/// A shell of an understating body: the wall is the body minus its shrunk self, so it stays
/// inside the body whatever the divisor.
#[test]
fn a_shelled_twisted_body_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 12.0, 60.0]);
    let twisted = graph.add(Node::Twist {
        input: body.into(),
        degrees_per_mm: lit(4.0),
        frame: Frame::along(Axis::Z),
    });
    graph.add(Node::Shell { input: twisted.into(), thickness: lit(2.0) });
    holds("shell 2 of a twisted cuboid", measure(&Document::new(), &graph, around_origin(80.0), 96));
}

// ---------------------------------------------------------------------------------------------
// Transform

fn placed(translation: [f64; 3], rotation: [f64; 3], scale: f64, about: Place) -> (Graph, NodeId) {
    let mut graph = Graph::empty();
    // Unequal on every axis and off the origin, so no symmetry can hide a wrong placement.
    let body = cuboid(&mut graph, [8.0, -5.0, 3.0], [30.0, 12.0, 6.0]);
    let id = graph.add(Node::Transform {
        input: body.into(),
        translation: translation.map(lit),
        rotation: rotation.map(lit),
        scale: lit(scale),
        about,
    });
    (graph, id)
}

#[test]
fn a_translated_cuboid_s_extent_holds_every_solid_point() {
    let (graph, _) = placed([25.0, -40.0, 12.0], [0.0; 3], 1.0, Place::origin());
    holds("transform translate", measure(&Document::new(), &graph, around_origin(80.0), N));
}

#[test]
fn a_turned_cuboid_s_extent_holds_every_solid_point() {
    let (graph, _) = placed([0.0; 3], [30.0, 45.0, 60.0], 1.0, Place::origin());
    holds("transform rotate (30,45,60)", measure(&Document::new(), &graph, around_origin(80.0), N));
}

#[test]
fn a_scaled_cuboid_s_extent_holds_every_solid_point() {
    for s in [2.5, 0.4] {
        let (graph, _) = placed([3.0, 0.0, -2.0], [0.0; 3], s, Place::origin());
        holds(&format!("transform scale {s}"), measure(&Document::new(), &graph, around_origin(80.0), N));
    }
}

#[test]
fn a_cuboid_turned_and_scaled_about_a_typed_pivot_s_extent_holds_every_solid_point() {
    let (graph, _) = placed([4.0, 6.0, -3.0], [20.0, -35.0, 70.0], 1.8, at(-30.0, 25.0, 10.0));
    holds(
        "transform rotate+scale 1.8 about (-30,25,10)",
        measure(&Document::new(), &graph, around_origin(80.0), N),
    );
}

/// A pivot taken from a point block, which the placement consumes.
#[test]
fn a_cuboid_turned_about_a_point_block_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, [8.0, -5.0, 3.0], [30.0, 12.0, 6.0]);
    let pivot = graph.add(Node::Point { at: [lit(40.0), lit(-20.0), lit(15.0)] });
    graph.add(Node::Transform {
        input: body.into(),
        translation: [lit(0.0); 3],
        rotation: [lit(0.0), lit(50.0), lit(-110.0)],
        scale: lit(1.5),
        about: Place::From { node: pivot },
    });
    holds(
        "transform rotate+scale 1.5 about a Point block at (40,-20,15)",
        measure(&Document::new(), &graph, around_origin(100.0), N),
    );
}

/// Two placements, the outer one pivoting about a point, so the composition order is tested too.
#[test]
fn a_twice_placed_cuboid_s_extent_holds_every_solid_point() {
    let (mut graph, inner) = placed([10.0, 0.0, 0.0], [0.0, 0.0, 40.0], 2.0, at(5.0, 5.0, 0.0));
    graph.add(Node::Transform {
        input: inner.into(),
        translation: [lit(-7.0), lit(3.0), lit(9.0)],
        rotation: [lit(65.0), lit(0.0), lit(15.0)],
        scale: lit(0.7),
        about: at(-12.0, 4.0, 20.0),
    });
    holds("transform of a transform", measure(&Document::new(), &graph, around_origin(100.0), N));
}

// ---------------------------------------------------------------------------------------------
// Mirror

fn mirrored(centre: [f64; 3], frame: Frame, join: bool) -> Graph {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, centre, [20.0, 10.0, 8.0]);
    graph.add(Node::Mirror { input: body.into(), frame, join });
    graph
}

#[test]
fn a_mirrored_cuboid_clear_of_the_plane_s_extent_holds_every_solid_point() {
    for join in [false, true] {
        let graph = mirrored([30.0, 4.0, -2.0], Frame::at_along(Axis::X, [lit(5.0), lit(0.0), lit(0.0)]), join);
        holds(
            &format!("mirror x=5, body clear of the plane, join {join}"),
            measure(&Document::new(), &graph, around_origin(60.0), N),
        );
    }
}

#[test]
fn a_mirrored_cuboid_across_the_plane_s_extent_holds_every_solid_point() {
    for join in [false, true] {
        let graph = mirrored([8.0, 4.0, -2.0], Frame::at_along(Axis::X, [lit(5.0), lit(0.0), lit(0.0)]), join);
        holds(
            &format!("mirror x=5, body crossing the plane, join {join}"),
            measure(&Document::new(), &graph, around_origin(60.0), N),
        );
    }
}

/// A plane tilted off every world axis, where the reflected box is no longer axis-aligned.
#[test]
fn a_cuboid_mirrored_in_a_tilted_plane_s_extent_holds_every_solid_point() {
    for (centre, what) in [([25.0, 10.0, 5.0], "clear of"), ([2.0, 1.0, 0.0], "crossing")] {
        for join in [false, true] {
            let frame = Frame::At {
                origin: [lit(-3.0), lit(2.0), lit(1.0)],
                axis: [lit(1.0), lit(0.7), lit(-0.4)],
            };
            let graph = mirrored(centre, frame, join);
            holds(
                &format!("mirror tilted plane, body {what} the plane, join {join}"),
                measure(&Document::new(), &graph, around_origin(60.0), N),
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Twist

fn twisted(body_centre: [f64; 3], rate: f64, frame: Frame) -> Graph {
    let mut graph = Graph::empty();
    let body = cuboid(&mut graph, body_centre, [40.0, 12.0, 60.0]);
    graph.add(Node::Twist { input: body.into(), degrees_per_mm: lit(rate), frame });
    graph
}

#[test]
fn a_cuboid_twisted_about_each_axis_s_extent_holds_every_solid_point() {
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        let graph = twisted([0.0, 0.0, 0.0], 6.0, Frame::along(axis));
        holds(
            &format!("twist 6 deg/mm about {axis:?}"),
            measure(&Document::new(), &graph, around_origin(80.0), 80),
        );
    }
}

#[test]
fn an_off_axis_cuboid_twisted_about_each_axis_s_extent_holds_every_solid_point() {
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        let graph = twisted([25.0, -10.0, 15.0], 6.0, Frame::along(axis));
        holds(
            &format!("twist 6 deg/mm about {axis:?}, body off the axis"),
            measure(&Document::new(), &graph, around_origin(80.0), 80),
        );
    }
}

#[test]
fn a_cuboid_twisted_in_a_tilted_offset_frame_s_extent_holds_every_solid_point() {
    let frame = Frame::At {
        origin: [lit(7.0), lit(-4.0), lit(3.0)],
        axis: [lit(1.0), lit(0.0), lit(1.0)],
    };
    let graph = twisted([0.0, 0.0, 0.0], 6.0, frame);
    holds(
        "twist 6 deg/mm, frame at (7,-4,3) along (1,0,1)",
        measure(&Document::new(), &graph, around_origin(80.0), 80),
    );
}

#[test]
fn a_cuboid_twisted_hard_s_extent_holds_every_solid_point() {
    for rate in [20.0, 45.0] {
        let frame = Frame::At {
            origin: [lit(-5.0), lit(8.0), lit(0.0)],
            axis: [lit(0.3), lit(-1.0), lit(0.5)],
        };
        let graph = twisted([6.0, 0.0, -4.0], rate, frame);
        holds(
            &format!("twist {rate} deg/mm, tilted frame"),
            measure(&Document::new(), &graph, around_origin(80.0), 80),
        );
    }
}
