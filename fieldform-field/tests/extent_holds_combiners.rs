// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Every point a combiner's field calls solid lies inside the box `graph_extent` gives it.
//!
//! An extent used to be a hint; the ray's start at the scene sphere and the per-tile body cull
//! now read it as a guarantee, so a box that is too small is missing material rather than lost
//! resolution. These cases cover the nodes that *combine* others — booleans, blends, morphs —
//! and each one samples a grid over its box grown by half its size (ten millimetres at least)
//! on every side, looking for a clearly solid sample outside the box.
//!
//! A case that fails here is a finding, and is left failing on purpose: the fix belongs in
//! `extent_of`, not in this file.

use fieldform_field::{graph_extent, Extent, GraphField, ScalarField};
use fieldform_scene::{
    no_turn, Axis, BooleanOp, CellMap, Continuity, Document, Fillet, Frame, Graph, LatticeCell,
    LatticePhase, Node, NodeId, ParamRef, Place, RampShape, Turn,
};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn at(x: f64, y: f64, z: f64) -> Place {
    Place::At([lit(x), lit(y), lit(z)])
}

fn sphere(g: &mut Graph, c: [f64; 3], r: f64) -> NodeId {
    g.add(Node::Sphere { centre: at(c[0], c[1], c[2]), radius: lit(r) })
}

fn cuboid(g: &mut Graph, c: [f64; 3], size: [f64; 3]) -> NodeId {
    g.add(Node::Cuboid { edge: Default::default(),
        centre: at(c[0], c[1], c[2]),
        size: [lit(size[0]), lit(size[1]), lit(size[2])],
        turn: no_turn(),
    })
}

fn turned_cuboid(g: &mut Graph, c: [f64; 3], size: [f64; 3], degrees_z: f64) -> NodeId {
    g.add(Node::Cuboid { edge: Default::default(),
        centre: at(c[0], c[1], c[2]),
        size: [lit(size[0]), lit(size[1]), lit(size[2])],
        turn: Turn::By([lit(0.0), lit(0.0), lit(degrees_z)]),
    })
}

fn boolean(g: &mut Graph, op: BooleanOp, inputs: Vec<NodeId>) -> NodeId {
    g.add(Node::Boolean { op, inputs: inputs.into_iter().map(Into::into).collect() })
}

fn blend(g: &mut Graph, op: BooleanOp, a: NodeId, b: NodeId, radius: f64) -> NodeId {
    blend_at(g, op, a, b, radius, Continuity::G1)
}

fn blend_at(
    g: &mut Graph,
    op: BooleanOp,
    a: NodeId,
    b: NodeId,
    radius: f64,
    continuity: Continuity,
) -> NodeId {
    g.add(Node::Blend {
        op,
        a: a.into(),
        b: b.into(),
        fillet: Fillet { radius: lit(radius), continuity, second: None },
    })
}

/// A cuboid wound about the world Z axis through the origin.
fn twisted_cuboid(g: &mut Graph, size: [f64; 3], degrees_per_mm: f64) -> NodeId {
    let body = cuboid(g, [0.0; 3], size);
    g.add(Node::Twist {
        input: body.into(),
        degrees_per_mm: lit(degrees_per_mm),
        frame: Frame::at_along(Axis::Z, [lit(0.0); 3]),
    })
}

/// A gyroid sheet trimmed to a cuboid, which is how a lattice becomes a finite body.
fn trimmed_gyroid(g: &mut Graph, c: [f64; 3], size: [f64; 3]) -> NodeId {
    let lattice = g.add(Node::Lattice {
        cell: LatticeCell::Gyroid,
        map: CellMap::Rectangular { cell: lit(10.0) },
        phase: LatticePhase::sheet(lit(1.5)),
        frame: Frame::default(),
    });
    let trim = cuboid(g, c, size);
    boolean(g, BooleanOp::Intersection, vec![lattice, trim])
}

/// What one case found.
#[derive(Debug)]
struct Finding {
    extent: Extent,
    samples: usize,
    solid_samples: usize,
    /// Euclidean distance from the box to the worst solid sample outside it, or for an `Empty`
    /// extent the depth of the deepest solid sample (there is no box to be outside of).
    worst_outside: f64,
    worst_point: Option<Point3<f64>>,
    worst_value: f64,
}

/// Samples the grown box and returns the worst violation.
///
/// `empty_probe` is the region searched when the extent is `Empty`, since there is no box to
/// grow; `spacing` is the finest step the node's features want, and is coarsened only as far
/// as the sample budget forces.
fn probe(
    doc: &Document,
    graph: &Graph,
    node: NodeId,
    spacing: f64,
    empty_probe: (Point3<f64>, Point3<f64>),
) -> Finding {
    let field = GraphField::compile_graph(doc, graph, 1e-4).expect("the case compiles");
    let extent = graph_extent(doc, graph, node);
    let (lo, hi, boxed) = match extent {
        Extent::Bounded { lo, hi } => (lo, hi, true),
        Extent::Empty => (empty_probe.0, empty_probe.1, false),
        // Holds trivially; the caller says whether it was the right answer.
        Extent::Unbounded => {
            return Finding {
                extent,
                samples: 0,
                solid_samples: 0,
                worst_outside: 0.0,
                worst_point: None,
                worst_value: 0.0,
            };
        }
    };
    let mut from = [0.0; 3];
    let mut to = [0.0; 3];
    for i in 0..3 {
        let grow = if boxed { (0.5 * (hi[i] - lo[i])).max(10.0) } else { 0.0 };
        from[i] = lo[i] - grow;
        to[i] = hi[i] + grow;
    }
    let volume: f64 = (0..3).map(|i| (to[i] - from[i]).max(1e-9)).product();
    let budget = 350_000.0;
    let step = spacing.max((volume / budget).cbrt());
    let counts: Vec<usize> =
        (0..3).map(|i| ((to[i] - from[i]) / step).ceil().max(1.0) as usize + 1).collect();

    let mut finding = Finding {
        extent,
        samples: 0,
        solid_samples: 0,
        worst_outside: 0.0,
        worst_point: None,
        worst_value: 0.0,
    };
    for ix in 0..counts[0] {
        for iy in 0..counts[1] {
            for iz in 0..counts[2] {
                let t = |i: usize, n: usize| {
                    if counts[i] == 1 {
                        0.5 * (from[i] + to[i])
                    } else {
                        from[i] + (to[i] - from[i]) * n as f64 / (counts[i] - 1) as f64
                    }
                };
                let p = Point3::new(t(0, ix), t(1, iy), t(2, iz));
                finding.samples += 1;
                let d = field.distance(p);
                if d >= -1e-6 {
                    continue;
                }
                finding.solid_samples += 1;
                let outside = if boxed {
                    let gap = |i: usize| (lo[i] - p[i]).max(p[i] - hi[i]).max(0.0);
                    (gap(0).powi(2) + gap(1).powi(2) + gap(2).powi(2)).sqrt()
                } else {
                    -d
                };
                if (boxed && outside > 1e-6 && outside > finding.worst_outside)
                    || (!boxed && outside > finding.worst_outside)
                {
                    finding.worst_outside = outside;
                    finding.worst_point = Some(p);
                    finding.worst_value = d;
                }
            }
        }
    }
    eprintln!("{finding:?}");
    finding
}

/// The property: nothing solid outside the box, and for an `Empty` extent nothing solid at all.
fn assert_holds(case: &str, f: &Finding) {
    match f.extent {
        Extent::Empty => assert!(
            f.solid_samples == 0,
            "{case}: extent is Empty but {} of {} samples are solid; deepest {:.4} mm at {:?}",
            f.solid_samples,
            f.samples,
            f.worst_outside,
            f.worst_point
        ),
        _ => assert!(
            f.worst_point.is_none(),
            "{case}: a solid sample (value {:.4}) lies {:.4} mm outside the extent {:?}, at {:?}",
            f.worst_value,
            f.worst_outside,
            f.extent,
            f.worst_point
        ),
    }
}

fn nowhere() -> (Point3<f64>, Point3<f64>) {
    (Point3::new(-40.0, -40.0, -40.0), Point3::new(40.0, 40.0, 40.0))
}

fn bounded(f: &Finding) -> bool {
    matches!(f.extent, Extent::Bounded { .. })
}

// ---------------------------------------------------------------------------------------------
// Boolean
// ---------------------------------------------------------------------------------------------

#[test]
fn a_union_of_two_spheres_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-8.0, 0.0, 0.0], 10.0);
    let b = sphere(&mut g, [9.0, 3.0, -2.0], 7.0);
    let u = boolean(&mut g, BooleanOp::Union, vec![a, b]);
    let f = probe(&doc, &g, u, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("union of two", &f);
}

#[test]
fn a_union_of_three_mixed_bodies_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-15.0, 0.0, 0.0], 8.0);
    let b = cuboid(&mut g, [0.0, 10.0, 5.0], [12.0, 6.0, 20.0]);
    let c = turned_cuboid(&mut g, [15.0, -5.0, 0.0], [20.0, 6.0, 8.0], 30.0);
    let u = boolean(&mut g, BooleanOp::Union, vec![a, b, c]);
    let f = probe(&doc, &g, u, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("union of three", &f);
}

#[test]
fn a_difference_of_two_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [0.0; 3], [30.0, 20.0, 10.0]);
    let b = sphere(&mut g, [15.0, 0.0, 0.0], 8.0);
    let d = boolean(&mut g, BooleanOp::Difference, vec![a, b]);
    let f = probe(&doc, &g, d, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("difference of two", &f);
}

#[test]
fn a_difference_of_three_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 15.0);
    let b = cuboid(&mut g, [0.0, 0.0, 15.0], [40.0, 40.0, 10.0]);
    let c = sphere(&mut g, [-15.0, 0.0, 0.0], 6.0);
    let d = boolean(&mut g, BooleanOp::Difference, vec![a, b, c]);
    let f = probe(&doc, &g, d, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("difference of three", &f);
}

#[test]
fn an_intersection_of_two_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 12.0);
    let b = cuboid(&mut g, [8.0, 4.0, 0.0], [20.0, 20.0, 30.0]);
    let i = boolean(&mut g, BooleanOp::Intersection, vec![a, b]);
    let f = probe(&doc, &g, i, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("intersection of two", &f);
}

/// Three operands, one of them the unbounded half-space: the bounded ones must bound it.
#[test]
fn an_intersection_of_three_with_a_half_space_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 14.0);
    let b = turned_cuboid(&mut g, [3.0, 0.0, 0.0], [20.0, 20.0, 40.0], 30.0);
    let c = g.add(Node::Plane { origin: at(0.0, 0.0, 4.0), flipped: false, turn: no_turn() });
    let i = boolean(&mut g, BooleanOp::Intersection, vec![a, b, c]);
    let f = probe(&doc, &g, i, 0.5, nowhere());
    assert!(bounded(&f), "a finite body cut by a half-space is finite: {:?}", f.extent);
    assert_holds("intersection of three with a plane", &f);
}

/// Two spheres whose boxes miss: `Empty` is right only if nothing is solid.
#[test]
fn an_intersection_of_disjoint_bodies_is_empty_and_nothing_is_solid() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-10.5, 0.0, 0.0], 10.0);
    let b = sphere(&mut g, [10.5, 0.0, 0.0], 10.0);
    let i = boolean(&mut g, BooleanOp::Intersection, vec![a, b]);
    let f = probe(&doc, &g, i, 0.25, (Point3::new(-25.0, -15.0, -15.0), Point3::new(25.0, 15.0, 15.0)));
    assert!(matches!(f.extent, Extent::Empty), "{:?}", f.extent);
    assert_holds("disjoint intersection", &f);
}

/// A union with a half-space really is unbounded; the check is that it says so.
#[test]
fn a_union_with_a_half_space_is_unbounded_which_is_the_right_answer() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0, 0.0, 20.0], 10.0);
    let b = g.add(Node::Plane { origin: at(0.0, 0.0, 0.0), flipped: false, turn: no_turn() });
    let u = boolean(&mut g, BooleanOp::Union, vec![a, b]);
    let f = probe(&doc, &g, u, 0.5, nowhere());
    assert!(matches!(f.extent, Extent::Unbounded), "{:?}", f.extent);
    assert_holds("union with a plane", &f);
}

// ---------------------------------------------------------------------------------------------
// Blend, exact operands
// ---------------------------------------------------------------------------------------------

#[test]
fn a_g1_blend_union_of_two_spheres_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-10.0, 0.0, 0.0], 9.0);
    let b = sphere(&mut g, [10.0, 0.0, 0.0], 9.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 5.0);
    let f = probe(&doc, &g, u, 0.4, nowhere());
    assert!(bounded(&f));
    assert_holds("G1 blend union", &f);
}

/// A fillet larger than either body, bridging a gap between them.
#[test]
fn a_blend_union_with_a_large_radius_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [-20.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = sphere(&mut g, [20.0, 0.0, 0.0], 6.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 40.0);
    let f = probe(&doc, &g, u, 0.75, nowhere());
    assert!(bounded(&f));
    assert_holds("large-radius blend union", &f);
}

/// An L of two bars with a big fillet in the inside corner.
#[test]
fn a_large_radius_fillet_in_an_inside_corner_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [0.0, 0.0, 0.0], [60.0, 8.0, 8.0]);
    let b = cuboid(&mut g, [-26.0, 26.0, 0.0], [8.0, 60.0, 8.0]);
    let u = blend(&mut g, BooleanOp::Union, a, b, 25.0);
    let f = probe(&doc, &g, u, 0.75, nowhere());
    assert!(bounded(&f));
    assert_holds("corner fillet", &f);
}

#[test]
fn a_g1_blend_difference_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [0.0; 3], [30.0, 30.0, 12.0]);
    let b = sphere(&mut g, [0.0, 0.0, 8.0], 10.0);
    let d = blend(&mut g, BooleanOp::Difference, a, b, 4.0);
    let f = probe(&doc, &g, d, 0.4, nowhere());
    assert!(bounded(&f));
    assert_holds("G1 blend difference", &f);
}

#[test]
fn a_blend_difference_with_a_large_radius_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [0.0; 3], [30.0, 30.0, 12.0]);
    let b = cuboid(&mut g, [15.0, 0.0, 0.0], [10.0, 50.0, 50.0]);
    let d = blend(&mut g, BooleanOp::Difference, a, b, 30.0);
    let f = probe(&doc, &g, d, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("large-radius blend difference", &f);
}

#[test]
fn a_g1_blend_intersection_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 14.0);
    let b = cuboid(&mut g, [6.0, 0.0, 0.0], [20.0, 20.0, 20.0]);
    let i = blend(&mut g, BooleanOp::Intersection, a, b, 4.0);
    let f = probe(&doc, &g, i, 0.4, nowhere());
    assert!(bounded(&f));
    assert_holds("G1 blend intersection", &f);
}

#[test]
fn a_blend_intersection_with_a_large_radius_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 14.0);
    let b = cuboid(&mut g, [6.0, 0.0, 0.0], [20.0, 20.0, 20.0]);
    let i = blend(&mut g, BooleanOp::Intersection, a, b, 40.0);
    let f = probe(&doc, &g, i, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("large-radius blend intersection", &f);
}

/// Boxes that miss each other: the smooth maximum is at least the hard one, so nothing here.
#[test]
fn a_blend_intersection_of_disjoint_bodies_is_empty_and_nothing_is_solid() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-12.0, 0.0, 0.0], 10.0);
    let b = sphere(&mut g, [12.0, 0.0, 0.0], 10.0);
    let i = blend(&mut g, BooleanOp::Intersection, a, b, 20.0);
    let f = probe(&doc, &g, i, 0.4, (Point3::new(-30.0, -20.0, -20.0), Point3::new(30.0, 20.0, 20.0)));
    assert!(matches!(f.extent, Extent::Empty), "{:?}", f.extent);
    assert_holds("disjoint blend intersection", &f);
}

// ---------------------------------------------------------------------------------------------
// Blend over lower-bound operands
// ---------------------------------------------------------------------------------------------

/// One twisted operand, one exact: the exact one's box still confines the fillet, because the
/// fillet needs *both* operands under the radius.
#[test]
fn a_blend_union_of_a_twisted_body_and_a_sphere_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], 10.0);
    let b = sphere(&mut g, [0.0, 0.0, 12.0], 8.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 10.0);
    let f = probe(&doc, &g, u, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("twisted with sphere", &f);
}

/// Two twisted operands. Each reads `d / stretch` beyond its body, with `stretch` about five
/// here, so both sit under the radius much further out than the radius the box was grown by.
#[test]
fn a_blend_union_of_two_twisted_bodies_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], 10.0);
    let b = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], -10.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 10.0);
    let f = probe(&doc, &g, u, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("two twisted", &f);
}

/// The same pair with a large fillet, which is where the understatement is multiplied most.
#[test]
fn a_large_radius_blend_union_of_two_twisted_bodies_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], 10.0);
    let b = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], -10.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 30.0);
    let f = probe(&doc, &g, u, 0.75, nowhere());
    assert!(bounded(&f));
    assert_holds("two twisted, large radius", &f);
}

/// A gentle twist on a tall bar, the kind a person actually makes: stretch about 1.4.
#[test]
fn a_blend_union_of_two_gently_twisted_bars_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = twisted_cuboid(&mut g, [30.0, 6.0, 60.0], 3.0);
    let b = twisted_cuboid(&mut g, [6.0, 30.0, 60.0], 3.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 8.0);
    let f = probe(&doc, &g, u, 0.6, nowhere());
    assert!(bounded(&f));
    assert_holds("two gently twisted", &f);
}

#[test]
fn a_blend_difference_of_twisted_bodies_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], 10.0);
    let b = twisted_cuboid(&mut g, [40.0, 10.0, 30.0], -10.0);
    let d = blend(&mut g, BooleanOp::Difference, a, b, 10.0);
    let f = probe(&doc, &g, d, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("twisted difference", &f);
}

#[test]
fn a_blend_intersection_of_twisted_bodies_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], 10.0);
    let b = twisted_cuboid(&mut g, [40.0, 40.0, 10.0], -10.0);
    let i = blend(&mut g, BooleanOp::Intersection, a, b, 10.0);
    let f = probe(&doc, &g, i, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("twisted intersection", &f);
}

/// A trimmed gyroid is a lower bound inside, but the trim is exact outside, so its box holds.
#[test]
fn a_blend_union_of_a_trimmed_lattice_and_a_sphere_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = trimmed_gyroid(&mut g, [0.0; 3], [30.0, 30.0, 30.0]);
    let b = sphere(&mut g, [18.0, 0.0, 0.0], 8.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 6.0);
    let f = probe(&doc, &g, u, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("trimmed lattice with sphere", &f);
}

#[test]
fn a_large_radius_blend_of_two_trimmed_lattices_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = trimmed_gyroid(&mut g, [-12.0, 0.0, 0.0], [20.0, 20.0, 20.0]);
    let b = trimmed_gyroid(&mut g, [12.0, 0.0, 0.0], [20.0, 20.0, 20.0]);
    let u = blend(&mut g, BooleanOp::Union, a, b, 25.0);
    let f = probe(&doc, &g, u, 0.6, nowhere());
    assert!(bounded(&f));
    assert_holds("two trimmed lattices", &f);
}

/// A bare lattice fills space, so a blend with one is unbounded, and that is the right answer.
#[test]
fn a_blend_union_with_a_bare_lattice_is_unbounded_which_is_the_right_answer() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = g.add(Node::Lattice {
        cell: LatticeCell::Gyroid,
        map: CellMap::Rectangular { cell: lit(10.0) },
        phase: LatticePhase::sheet(lit(1.5)),
        frame: Frame::default(),
    });
    let b = sphere(&mut g, [0.0; 3], 10.0);
    let u = blend(&mut g, BooleanOp::Union, a, b, 4.0);
    let f = probe(&doc, &g, u, 0.5, nowhere());
    assert!(matches!(f.extent, Extent::Unbounded), "{:?}", f.extent);
    assert_holds("bare lattice", &f);
}

/// Two intersections that are each empty, with a hair's gap between their spheres — and the
/// blend over them makes material that no box of their solids can hold.
///
/// An empty intersection's field is still small in the gap: it is `max` of two nearly-zero
/// distances. A blend fills wherever *both* operands read under its radius, solid or not, so
/// material appears where neither operand has any — 888 solid samples here.
///
/// **This is the case that settles how an extent has to be computed, and it asserted the opposite
/// for a while.** It was written expecting `Empty`, since `Empty` is what the walk said and both
/// operands' solids really are empty; it was then marked as a known violation
/// (`tile-mask-design-01941c5.md` §1.8 R3) because `assert_holds` cannot be satisfied by an
/// `Empty` box over material that exists. No growth applied *after* the walk can repair it —
/// `Extent::Empty.grown(anything)` is still `Empty` — which is why `extent_below` carries the
/// threshold *down* to the operands instead, where an empty intersection of two spheres a hair
/// apart has a perfectly good box at the radius the fillet asks about.
///
/// So what is asserted is the property and not the old answer: the box is finite, and every solid
/// point is inside it.
#[test]
fn a_blend_union_of_two_empty_intersections_holds_the_material_it_makes() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a1 = sphere(&mut g, [-10.5, 0.0, 0.0], 10.0);
    let a2 = sphere(&mut g, [10.5, 0.0, 0.0], 10.0);
    let b1 = sphere(&mut g, [0.0, -10.5, 0.0], 10.0);
    let b2 = sphere(&mut g, [0.0, 10.5, 0.0], 10.0);
    let a = boolean(&mut g, BooleanOp::Intersection, vec![a1, a2]);
    let b = boolean(&mut g, BooleanOp::Intersection, vec![b1, b2]);
    let u = blend(&mut g, BooleanOp::Union, a, b, 10.0);
    let f = probe(&doc, &g, u, 0.25, (Point3::new(-20.0, -20.0, -15.0), Point3::new(20.0, 20.0, 15.0)));
    assert!(bounded(&f), "{:?}", f.extent);
    // The half that makes the rest worth asserting: a box over a shape with no material in it
    // holds trivially, and this shape has material.
    assert!(
        f.solid_samples > 0,
        "nothing is solid here, so this case no longer reproduces what it was written for"
    );
    assert_holds("two empty intersections", &f);
}

// ---------------------------------------------------------------------------------------------
// Morph
// ---------------------------------------------------------------------------------------------

/// Two spheres far apart at t = 1/2 — the case that shrinks to nothing between them.
#[test]
fn a_morph_of_two_distant_spheres_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-12.0, 0.0, 0.0], 15.0);
    let b = sphere(&mut g, [12.0, 0.0, 0.0], 15.0);
    let m = g.add(Node::Morph { a: a.into(), b: b.into(), t: lit(0.5) });
    let f = probe(&doc, &g, m, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("morph at a half", &f);
}

#[test]
fn a_morph_of_a_sphere_into_a_turned_cuboid_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 15.0);
    let b = turned_cuboid(&mut g, [5.0, 0.0, 0.0], [40.0, 10.0, 10.0], 30.0);
    let m = g.add(Node::Morph { a: a.into(), b: b.into(), t: lit(0.3) });
    let f = probe(&doc, &g, m, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("morph sphere to cuboid", &f);
}

/// `t` from a ramp over the distance to a point, running the whole legal 0..1.
#[test]
fn a_ramp_driven_morph_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 20.0);
    let b = cuboid(&mut g, [10.0, 0.0, 0.0], [30.0, 30.0, 30.0]);
    let point = g.add(Node::Point { at: [lit(20.0), lit(0.0), lit(0.0)] });
    let ramp = g.add(Node::Ramp {
        input: point.into(),
        in_min: lit(5.0),
        in_max: lit(25.0),
        out_min: lit(0.0),
        out_max: lit(1.0),
        shape: RampShape::Smooth,
    });
    let m = g.add(Node::Morph { a: a.into(), b: b.into(), t: ParamRef::Field(ramp) });
    let f = probe(&doc, &g, m, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("ramp-driven morph", &f);
}

/// A ramp running backwards (1 near, 0 far), over a pair that do not overlap.
#[test]
fn a_reversed_ramp_driven_morph_of_separated_bodies_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-15.0, 0.0, 0.0], 10.0);
    let b = cuboid(&mut g, [15.0, 0.0, 0.0], [16.0, 16.0, 16.0]);
    let point = g.add(Node::Point { at: [lit(0.0), lit(0.0), lit(0.0)] });
    let ramp = g.add(Node::Ramp {
        input: point.into(),
        in_min: lit(30.0),
        in_max: lit(0.0),
        out_min: lit(0.0),
        out_max: lit(1.0),
        shape: RampShape::Linear,
    });
    let m = g.add(Node::Morph { a: a.into(), b: b.into(), t: ParamRef::Field(ramp) });
    let f = probe(&doc, &g, m, 0.5, nowhere());
    assert!(bounded(&f));
    assert_holds("reversed ramp-driven morph", &f);
}

// ---------------------------------------------------------------------------------------------
// G2: the same cases at the cubic fold
//
// The extent walk reads the exponent off the fillet (`join_reach` is `k(1 − W^(−1/n))`), so a G2
// blend reaches a different distance from a G1 one at the same radius — 0.206·k against 0.293·k
// for two operands. These say that the box the walk draws is still a box that holds the solid,
// which is the guarantee the tile mask is built on; a constant left at its G1 value would show
// here as solid outside the box.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_g2_blend_union_of_two_spheres_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [-10.0, 0.0, 0.0], 9.0);
    let b = sphere(&mut g, [10.0, 0.0, 0.0], 9.0);
    let u = blend_at(&mut g, BooleanOp::Union, a, b, 5.0, Continuity::G2);
    let f = probe(&doc, &g, u, 0.4, nowhere());
    assert!(bounded(&f));
    assert_holds("G2 blend union", &f);
}

#[test]
fn a_g2_blend_union_with_a_large_radius_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [-20.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = sphere(&mut g, [20.0, 0.0, 0.0], 6.0);
    let u = blend_at(&mut g, BooleanOp::Union, a, b, 40.0, Continuity::G2);
    let f = probe(&doc, &g, u, 0.75, nowhere());
    assert!(bounded(&f));
    assert_holds("large-radius G2 blend union", &f);
}

#[test]
fn a_large_radius_g2_fillet_in_an_inside_corner_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [0.0, 0.0, 0.0], [60.0, 8.0, 8.0]);
    let b = cuboid(&mut g, [-26.0, 26.0, 0.0], [8.0, 60.0, 8.0]);
    let u = blend_at(&mut g, BooleanOp::Union, a, b, 25.0, Continuity::G2);
    let f = probe(&doc, &g, u, 0.75, nowhere());
    assert!(bounded(&f));
    assert_holds("G2 corner fillet", &f);
}

#[test]
fn a_g2_blend_difference_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = cuboid(&mut g, [0.0; 3], [30.0, 30.0, 12.0]);
    let b = sphere(&mut g, [0.0, 0.0, 8.0], 10.0);
    let d = blend_at(&mut g, BooleanOp::Difference, a, b, 4.0, Continuity::G2);
    let f = probe(&doc, &g, d, 0.4, nowhere());
    assert!(bounded(&f));
    assert_holds("G2 blend difference", &f);
}

#[test]
fn a_g2_blend_intersection_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let a = sphere(&mut g, [0.0; 3], 14.0);
    let b = cuboid(&mut g, [6.0, 0.0, 0.0], [20.0, 20.0, 20.0]);
    let i = blend_at(&mut g, BooleanOp::Intersection, a, b, 4.0, Continuity::G2);
    let f = probe(&doc, &g, i, 0.4, nowhere());
    assert!(bounded(&f));
    assert_holds("G2 blend intersection", &f);
}
