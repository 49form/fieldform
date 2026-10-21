// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A primitive's extent encloses every point its field calls solid.
//!
//! `graph_extent` began as a hint for framing the view, and it is now a guarantee: the ray
//! starts at the scene sphere built from it, and a per-tile body cull skips a body wherever its
//! box misses the tile. A box that is too small there is *missing material*, not a loose frame.
//! So each case here builds one primitive, asks the extent, and samples the compiled field on a
//! grid over that box grown by half its size (at least 10 mm) on every side. Any clearly solid
//! sample outside the box is a violation, and the worst one is reported with how far out it
//! lies and where.
//!
//! Measured on the field rather than reasoned from the formula, because the field is what the
//! tracer draws: a turn is lowered into a placement before compiling, so the only way to know
//! the box agrees with what is drawn is to ask what is drawn.

use fieldform_field::{graph_extent, Extent, GraphField, ScalarField};
use fieldform_scene::{Document, Ends, Graph, Node, NodeId, ParamRef, Place, Turn};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn at(x: f64, y: f64, z: f64) -> Place {
    Place::at(x, y, z)
}

fn turned(x: f64, y: f64, z: f64) -> Turn {
    Turn::By([lit(x), lit(y), lit(z)])
}

/// What the sampling found.
#[derive(Debug)]
struct Reading {
    extent: Extent,
    /// Clearly solid samples anywhere in the sampled region.
    solid: usize,
    /// Of those, the ones outside the box by more than 1e-6 mm.
    outside: usize,
    /// The farthest outside, and where.
    worst: Option<(f64, Point3<f64>, f64)>,
}

/// Euclidean distance from `p` to the box, nought inside it.
fn outside_by(p: Point3<f64>, lo: Point3<f64>, hi: Point3<f64>) -> f64 {
    let d = |v: f64, a: f64, b: f64| (a - v).max(v - b).max(0.0);
    let (dx, dy, dz) = (d(p.x, lo.x, hi.x), d(p.y, lo.y, hi.y), d(p.z, lo.z, hi.z));
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Samples the compiled field over a region and measures it against the node's extent.
///
/// A bounded extent is sampled over itself grown by `max(50 % of its size, 10 mm)` per axis. An
/// empty or unbounded one has no box to grow, so `fallback` gives the region instead — and an
/// unbounded one is then read for whether it *should* have been: solid samples on the region's
/// faces are the evidence that material runs off to infinity.
fn read(doc: &Document, graph: &Graph, node: NodeId, fallback: (Point3<f64>, Point3<f64>)) -> Reading {
    let extent = graph_extent(doc, graph, node);
    let field = GraphField::compile_graph(doc, graph, 1e-4).expect("the case compiles");
    let (lo, hi) = match extent {
        Extent::Bounded { lo, hi } => {
            let grow = |a: f64, b: f64| ((b - a).abs() * 0.5).max(10.0);
            (
                Point3::new(lo.x - grow(lo.x, hi.x), lo.y - grow(lo.y, hi.y), lo.z - grow(lo.z, hi.z)),
                Point3::new(hi.x + grow(lo.x, hi.x), hi.y + grow(lo.y, hi.y), hi.z + grow(lo.z, hi.z)),
            )
        }
        _ => fallback,
    };
    // 71 per axis is ~358k samples; odd, so the centre plane of a symmetric region is sampled.
    let n = 71usize;
    let mut reading = Reading { extent, solid: 0, outside: 0, worst: None };
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let t = |c: usize| c as f64 / (n - 1) as f64;
                let p = Point3::new(
                    lo.x + (hi.x - lo.x) * t(i),
                    lo.y + (hi.y - lo.y) * t(j),
                    lo.z + (hi.z - lo.z) * t(k),
                );
                let v = field.distance(p);
                if !(v < -1e-6) {
                    continue;
                }
                reading.solid += 1;
                let out = match extent {
                    Extent::Bounded { lo, hi } => outside_by(p, lo, hi),
                    Extent::Unbounded => 0.0,
                    // Any solid point at all is outside an empty extent.
                    Extent::Empty => f64::INFINITY,
                };
                if out > 1e-6 {
                    reading.outside += 1;
                    if reading.worst.map_or(true, |(w, _, _)| out > w) {
                        reading.worst = Some((out, p, v));
                    }
                }
            }
        }
    }
    println!("{reading:?}");
    reading
}

fn assert_holds(case: &str, r: &Reading) {
    assert!(r.solid > 0, "{case}: no solid sample at all, so the case measured nothing: {r:?}");
    assert!(
        r.outside == 0,
        "{case}: {} of {} solid samples lie outside the extent; worst {:?} (outside by mm, \
         point, field value); extent {:?}",
        r.outside,
        r.solid,
        r.worst,
        r.extent,
    );
}

fn one(node: Node) -> (Document, Graph, NodeId) {
    let mut graph = Graph::empty();
    let id = graph.add(node);
    (Document::new(), graph, id)
}

fn around_origin(half: f64) -> (Point3<f64>, Point3<f64>) {
    (Point3::new(-half, -half, -half), Point3::new(half, half, half))
}

// --- Sphere ---------------------------------------------------------------------------------

#[test]
fn an_offset_sphere_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Sphere { centre: at(12.0, -7.0, 30.0), radius: lit(9.0) });
    assert_holds("offset sphere", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_sphere_placed_on_a_point_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let seed = graph.add(Node::Point { at: [lit(-15.0), lit(4.0), lit(22.0)] });
    let id = graph.add(Node::Sphere { centre: Place::From { node: seed }, radius: lit(6.0) });
    // The point is consumed as the sphere's place, so the sphere is the only body.
    assert_eq!(graph.roots(), vec![id]);
    let doc = Document::new();
    assert_holds("sphere on a point", &read(&doc, &graph, id, around_origin(60.0)));
}

// --- Cuboid ---------------------------------------------------------------------------------

#[test]
fn an_unturned_cuboid_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Cuboid { edge: Default::default(),
        centre: at(5.0, -3.0, 8.0),
        size: [lit(40.0), lit(10.0), lit(6.0)],
        turn: Turn::default(),
    });
    assert_holds("unturned cuboid", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_turned_cuboid_s_extent_holds_every_solid_point() {
    // A 40 x 10 x 6 bar turned 30 degrees about Z: its corners reach y = +-14.3, far past the
    // +-5 an axis-aligned box of its sides would claim.
    let (doc, graph, id) = one(Node::Cuboid { edge: Default::default(),
        centre: at(0.0, 0.0, 0.0),
        size: [lit(40.0), lit(10.0), lit(6.0)],
        turn: turned(0.0, 0.0, 30.0),
    });
    assert_holds("cuboid turned 30 deg about z", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_cuboid_turned_about_all_three_axes_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Cuboid { edge: Default::default(),
        centre: at(10.0, 20.0, -5.0),
        size: [lit(30.0), lit(8.0), lit(4.0)],
        turn: turned(20.0, 35.0, 50.0),
    });
    assert_holds("cuboid turned 20/35/50 deg", &read(&doc, &graph, id, around_origin(60.0)));
}

// --- Cylinder, cone, capsule ----------------------------------------------------------------

#[test]
fn an_oblique_cylinder_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Cylinder { edge: Default::default(),
        ends: Ends::At { from: at(-10.0, 5.0, 0.0), to: at(20.0, -15.0, 25.0) },
        radius: lit(6.0),
    });
    assert_holds("oblique cylinder", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_cylinder_standing_on_a_turned_plane_s_extent_holds_every_solid_point() {
    let mut graph = Graph::empty();
    let plane = graph.add(Node::Plane {
        origin: at(5.0, 0.0, 3.0),
        flipped: false,
        turn: turned(50.0, 0.0, 20.0),
    });
    let id = graph.add(Node::Cylinder { edge: Default::default(),
        ends: Ends::On { node: plane, length: lit(30.0) },
        radius: lit(5.0),
    });
    // The plane is consumed as the cylinder's axis, so it is not a half-space of its own here.
    assert_eq!(graph.roots(), vec![id]);
    let doc = Document::new();
    assert_holds("cylinder on a turned plane", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn an_oblique_cone_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Cone { edge: Default::default(),
        ends: Ends::At { from: at(0.0, 0.0, -10.0), to: at(15.0, 10.0, 20.0) },
        base_radius: lit(12.0),
        top_radius: lit(0.0),
    });
    assert_holds("oblique cone to a point", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_cone_widening_toward_its_top_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Cone { edge: Default::default(),
        ends: Ends::At { from: at(0.0, 0.0, 0.0), to: at(-8.0, 12.0, 18.0) },
        base_radius: lit(3.0),
        top_radius: lit(14.0),
    });
    assert_holds("cone widening upward", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_cone_with_a_negative_top_radius_s_extent_holds_every_solid_point() {
    // Not a shape anybody means, but a radius is a parameter and can be driven below nought.
    // The extent grows a ball of `-5` around the top, which is an inverted box.
    let (doc, graph, id) = one(Node::Cone { edge: Default::default(),
        ends: Ends::At { from: at(0.0, 0.0, 0.0), to: at(0.0, 0.0, 20.0) },
        base_radius: lit(10.0),
        top_radius: lit(-5.0),
    });
    assert_holds("cone with top radius -5", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn an_oblique_capsule_with_unequal_radii_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Capsule {
        ends: Ends::At { from: at(-12.0, -4.0, 3.0), to: at(14.0, 9.0, -11.0) },
        start_radius: lit(8.0),
        end_radius: lit(3.0),
    });
    assert_holds("oblique capsule 8/3", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_capsule_one_end_swallowing_the_other_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Capsule {
        ends: Ends::At { from: at(0.0, 0.0, 0.0), to: at(3.0, 2.0, 1.0) },
        start_radius: lit(12.0),
        end_radius: lit(2.0),
    });
    assert_holds("capsule 12/2, one swallows other", &read(&doc, &graph, id, around_origin(60.0)));
}

// --- Torus ----------------------------------------------------------------------------------

#[test]
fn an_unturned_torus_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Torus {
        centre: at(4.0, -6.0, 10.0),
        major_radius: lit(15.0),
        minor_radius: lit(3.0),
        turn: Turn::default(),
    });
    assert_holds("unturned torus", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_turned_torus_s_extent_holds_every_solid_point() {
    // Tilted 40 degrees about X, its ring rises to z = +-11.9 where an unturned box says +-3.
    let (doc, graph, id) = one(Node::Torus {
        centre: at(0.0, 0.0, 0.0),
        major_radius: lit(15.0),
        minor_radius: lit(3.0),
        turn: turned(40.0, 0.0, 0.0),
    });
    assert_holds("torus turned 40 deg about x", &read(&doc, &graph, id, around_origin(60.0)));
}

#[test]
fn a_torus_whose_tube_is_wider_than_its_ring_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(Node::Torus {
        centre: at(0.0, 0.0, 0.0),
        major_radius: lit(4.0),
        minor_radius: lit(9.0),
        turn: Turn::default(),
    });
    assert_holds("self-intersecting torus 4/9", &read(&doc, &graph, id, around_origin(60.0)));
}

// --- Point ----------------------------------------------------------------------------------

/// A point's extent is the one place it sits.
///
/// **Not read through `read`, unlike every case above it, because a point cannot be compiled on
/// its own.** `Node::is_body` stopped admitting one on 2026-09-26 (`profile-group-spec.md`
/// §4.5, and the project notes' "a body is a root of its graph"), so a part holding only a point has no
/// roots and `compile_graph` answers `Empty`. That is the rule working, not a gap: sampling a
/// field for solid points is the wrong question to ask of something whose field is never
/// negative, which is what `encloses_material` says and what this test used to check the long
/// way round.
#[test]
fn a_point_s_extent_is_the_one_place_it_sits() {
    let (doc, graph, id) = one(Node::Point { at: [lit(3.0), lit(-2.0), lit(7.0)] });
    let extent = graph_extent(&doc, &graph, id);
    let Extent::Bounded { lo, hi } = extent else {
        panic!("a point is somewhere: {extent:?}");
    };
    assert_eq!(lo, hi, "and it has no size: {extent:?}");
    assert_eq!(lo, Point3::new(3.0, -2.0, 7.0), "and it is where it was put: {extent:?}");
}

// --- Plane ----------------------------------------------------------------------------------

/// Unbounded is only the *right* answer if the solid really runs off: checked by finding solid
/// samples on the faces of a large region on the solid side.
fn plane_is_rightly_unbounded(case: &str, flipped: bool, turn: Turn) {
    let (doc, graph, id) = one(Node::Plane { origin: at(2.0, -3.0, 5.0), flipped, turn });
    let r = read(&doc, &graph, id, around_origin(500.0));
    assert!(r.extent.is_unbounded(), "{case}: a half-space has no box: {r:?}");
    assert!(r.solid > 0, "{case}: a half-space is solid somewhere: {r:?}");
    // Solid at the corners of the region far from the plane on its solid side: the material is
    // not confined, so no box would have been honest.
    let field = GraphField::compile_graph(&doc, &graph, 1e-4).unwrap();
    let corners = (0..8).map(|c| {
        Point3::new(
            if c & 1 == 0 { -1e6 } else { 1e6 },
            if c & 2 == 0 { -1e6 } else { 1e6 },
            if c & 4 == 0 { -1e6 } else { 1e6 },
        )
    });
    let solid_far = corners.filter(|p| field.distance(*p) < -1e-6).count();
    assert!(solid_far >= 1, "{case}: no solid point a kilometre out, so Unbounded was lazy");
}

#[test]
fn a_plane_is_rightly_unbounded() {
    plane_is_rightly_unbounded("plane", false, Turn::default());
}

#[test]
fn a_flipped_turned_plane_is_rightly_unbounded() {
    plane_is_rightly_unbounded("flipped turned plane", true, turned(30.0, 15.0, 0.0));
}
