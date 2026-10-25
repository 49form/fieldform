// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! `graph_extent` is about to be a guarantee rather than a hint: the per-tile body cull skips a
//! body wherever its box misses a tile, so a box that stops short of the material is material
//! the viewport never draws. This file holds the repeating and the sampled kinds to that — the
//! pattern (a row and a ring, holding instance zero and not, hard and rounded), the lattice
//! confined to a solid under every cell map, the cage and the imported mesh — and confirms the
//! ramp stays out of the bodies.
//!
//! **The property, measured rather than argued.** A regular grid over the box grown by half its
//! size (at least 10 mm) on every side; any sample the field calls clearly solid
//! (`distance < -1e-6`) that lies more than 1e-6 mm outside the box is a violation, and the worst
//! one is printed with where it is. Samples *inside* the box are skipped rather than evaluated —
//! they cannot violate anything — which spends the whole budget where a violation can be.
//!
//! A failing case here is a finding, left failing on purpose until `extent_of` is fixed.

use fieldform_cage::{Cage, QuadMesh};
use fieldform_field::{graph_extent, Extent, GraphField, ScalarField};
use fieldform_scene::{
    Axis, BooleanOp, CellMap, Document, Fillet, Frame, Graph, LatticeCell, LatticePhase,
    MeshData, Node, NodeId, Operand, ParamRef, PatternKind, Place, RampShape, Turn,
};
use nalgebra::Point3;
use rayon::prelude::*;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// The most grid points a case may lay down. "A few hundred thousand", as the task set it.
const BUDGET: f64 = 400_000.0;

/// What sampling found.
#[derive(Debug)]
struct Finding {
    extent: Extent,
    /// Grid points evaluated (those outside the box).
    evaluated: usize,
    /// Solid samples anywhere in the evaluated set.
    solid: usize,
    /// The worst violation: how far outside the box, and where.
    worst: Option<(f64, Point3<f64>, f64)>,
}

/// Euclidean distance from `p` to the box, nought inside it.
fn outside(p: Point3<f64>, lo: Point3<f64>, hi: Point3<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        let e = (lo[i] - p[i]).max(p[i] - hi[i]).max(0.0);
        s += e * e;
    }
    s.sqrt()
}

/// Samples a grid over `[lo, hi]`, skipping the points inside `skip` when one is given.
fn sample(
    field: &GraphField,
    lo: Point3<f64>,
    hi: Point3<f64>,
    skip: Option<(Point3<f64>, Point3<f64>)>,
) -> (usize, usize, Option<(f64, Point3<f64>, f64)>) {
    let size = hi - lo;
    let volume = size.x.max(1e-3) * size.y.max(1e-3) * size.z.max(1e-3);
    let h = (volume / BUDGET).cbrt();
    let n = [0, 1, 2].map(|i| ((size[i] / h).ceil() as usize).max(1) + 1);
    let at = |i: usize, j: usize, k: usize| {
        Point3::new(
            lo.x + size.x * i as f64 / (n[0] - 1) as f64,
            lo.y + size.y * j as f64 / (n[1] - 1) as f64,
            lo.z + size.z * k as f64 / (n[2] - 1) as f64,
        )
    };
    let results: Vec<(usize, usize, Option<(f64, Point3<f64>, f64)>)> = (0..n[0])
        .into_par_iter()
        .map(|i| {
            let (mut evaluated, mut solid, mut worst) = (0usize, 0usize, None);
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let p = at(i, j, k);
                    let out = match skip {
                        Some((blo, bhi)) => {
                            let o = outside(p, blo, bhi);
                            if o <= 1e-6 {
                                continue;
                            }
                            o
                        }
                        None => f64::INFINITY,
                    };
                    evaluated += 1;
                    let d = field.distance(p);
                    if d < -1e-6 {
                        solid += 1;
                        let better = match worst {
                            None => true,
                            Some((w, _, _)) => out > w,
                        };
                        if better {
                            worst = Some((out, p, d));
                        }
                    }
                }
            }
            (evaluated, solid, worst)
        })
        .collect();
    results.into_iter().fold((0, 0, None), |(e, s, w), (e2, s2, w2)| {
        let w = match (w, w2) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(if b.0 > a.0 { b } else { a }),
        };
        (e + e2, s + s2, w)
    })
}

/// Holds one node of a graph to the property.
///
/// An `Empty` extent is sampled over `around`, which the case supplies since there is no box to
/// grow; an `Unbounded` one is not sampled at all, since it holds trivially.
fn check(doc: &Document, graph: &Graph, node: NodeId, around: Option<(Point3<f64>, Point3<f64>)>) -> Finding {
    let extent = graph_extent(doc, graph, node);
    let field = GraphField::compile_graph(doc, graph, 1e-4).expect("compiles");
    match extent {
        Extent::Unbounded => Finding { extent, evaluated: 0, solid: 0, worst: None },
        Extent::Empty => {
            let (lo, hi) = around.expect("an empty extent is sampled over a region the case gives");
            let (evaluated, solid, worst) = sample(&field, lo, hi, None);
            Finding { extent, evaluated, solid, worst }
        }
        Extent::Bounded { lo, hi } => {
            let size = hi - lo;
            let grow = size.map(|s| (0.5 * s).max(10.0));
            let (evaluated, solid, worst) = sample(&field, lo - grow, hi + grow, Some((lo, hi)));
            Finding { extent, evaluated, solid, worst }
        }
    }
}

/// Prints the finding as one line the report can quote, then asserts it holds.
fn holds(name: &str, finding: Finding) {
    println!(
        "CASE {name}: extent {:?}; evaluated {} outside the box, {} solid; worst {:?}",
        finding.extent, finding.evaluated, finding.solid, finding.worst
    );
    match finding.extent {
        Extent::Empty => assert_eq!(
            finding.solid, 0,
            "{name}: an empty extent, but the field is solid at {:?}",
            finding.worst
        ),
        _ => {
            if let Some((out, p, d)) = finding.worst {
                assert!(
                    out <= 1e-6,
                    "{name}: solid at {p:?} (distance {d}), {out} mm outside the extent {:?}",
                    finding.extent
                );
            }
        }
    }
}

/// The only root of a graph, which every case here builds.
fn root(graph: &Graph) -> NodeId {
    let roots = graph.roots();
    assert_eq!(roots.len(), 1, "one body per case");
    roots[0]
}

fn sphere(graph: &mut Graph, c: [f64; 3], r: f64) -> NodeId {
    graph.add(Node::Sphere { centre: Place::At(c.map(lit)), radius: lit(r) })
}

fn cuboid(graph: &mut Graph, c: [f64; 3], size: [f64; 3]) -> NodeId {
    graph.add(Node::Cuboid { edge: Default::default(), centre: Place::At(c.map(lit)), size: size.map(lit), turn: Turn::default() })
}

// A count a case states as a whole number, which is what every case here wants, handed to the node
// as the `ParamRef` a count has been since it could be driven by a variable (item 70). The
// conversion is here rather than at the thirty call sites because what these cases are about is
// the extent, and a count that reads 6.0 says nothing about one.
fn linear(graph: &mut Graph, input: NodeId, count: u32, step: [f64; 3], join: bool, fillet: Fillet) -> NodeId {
    let count = lit(f64::from(count));
    graph.add(Node::Pattern { options: Default::default(), input: input.into(), count, kind: PatternKind::Linear { step: step.map(lit) }, join, fillet })
}

fn radial(graph: &mut Graph, input: NodeId, count: u32, frame: Frame, join: bool, fillet: Fillet) -> NodeId {
    let count = lit(f64::from(count));
    graph.add(Node::Pattern { options: Default::default(), input: input.into(), count, kind: PatternKind::Radial { frame }, join, fillet })
}

// ---------------------------------------------------------------------------------------------
// Pattern, hard joins
// ---------------------------------------------------------------------------------------------

#[test]
fn a_hard_row_of_spheres_holding_its_body_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = sphere(&mut g, [5.0, -3.0, 2.0], 8.0);
    linear(&mut g, b, 5, [12.0, 4.0, -3.0], true, Fillet::hard());
    let r = root(&g);
    holds("linear, join, hard", check(&doc, &g, r, None));
}

#[test]
fn a_hard_row_of_spheres_not_holding_its_body_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = sphere(&mut g, [5.0, -3.0, 2.0], 8.0);
    let p = linear(&mut g, b, 5, [12.0, 4.0, -3.0], false, Fillet::hard());
    holds("linear, no join, hard", check(&doc, &g, p, None));
}

/// About a tilted axis through a point off the body, so a wrong centre or a wrong axis in the
/// extent's rotation would carry the box somewhere the copies are not.
#[test]
fn a_hard_ring_about_a_tilted_offset_axis_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = cuboid(&mut g, [30.0, 5.0, 0.0], [16.0, 6.0, 10.0]);
    let frame = Frame::At { origin: [lit(4.0), lit(-2.0), lit(3.0)], axis: [lit(1.0), lit(2.0), lit(3.0)] };
    radial(&mut g, b, 7, frame, true, Fillet::hard());
    let r = root(&g);
    holds("radial tilted, join, hard", check(&doc, &g, r, None));
}

#[test]
fn a_hard_ring_not_holding_its_body_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = cuboid(&mut g, [30.0, 5.0, 0.0], [16.0, 6.0, 10.0]);
    let frame = Frame::At { origin: [lit(4.0), lit(-2.0), lit(3.0)], axis: [lit(1.0), lit(2.0), lit(3.0)] };
    let p = radial(&mut g, b, 7, frame, false, Fillet::hard());
    holds("radial tilted, no join, hard", check(&doc, &g, p, None));
}

/// A pattern of one, not holding its body: the evaluator keeps instance zero rather than an
/// empty body, and the extent has to make the same decision.
#[test]
fn a_pattern_of_one_not_holding_its_body_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = sphere(&mut g, [3.0, 0.0, 0.0], 6.0);
    let p = linear(&mut g, b, 1, [20.0, 0.0, 0.0], false, Fillet::hard());
    holds("linear count 1, no join", check(&doc, &g, p, None));
}

// ---------------------------------------------------------------------------------------------
// Pattern, rounded joins
// ---------------------------------------------------------------------------------------------

/// Overlapping spheres, where the fillet fills the necks. The necks lie inside the row's box,
/// so this one is expected to hold — it is the control for the cuboid cases below.
#[test]
fn a_rounded_row_of_overlapping_spheres_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = sphere(&mut g, [0.0, 0.0, 0.0], 20.0);
    let p = linear(&mut g, b, 5, [30.0, 0.0, 0.0], true, Fillet::of(lit(6.0)));
    holds("linear spheres, join, fillet 6", check(&doc, &g, p, None));
}

/// Cuboids touching face to face, with coplanar tops. On the seam above the top face each copy
/// reads the same small distance `d`, and the n-ary smooth minimum is negative there for
/// `d < k (1 - 1/sqrt 2)` — a ridge standing proud of the faces, outside every copy's box.
#[test]
fn a_rounded_row_of_touching_cuboids_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = cuboid(&mut g, [0.0, 0.0, 0.0], [20.0, 20.0, 20.0]);
    let p = linear(&mut g, b, 3, [20.0, 0.0, 0.0], true, Fillet::of(lit(8.0)));
    holds("linear touching cuboids, join, fillet 8", check(&doc, &g, p, None));
}

/// Heavily overlapping cuboids, so many copies read the same distance above one point of the
/// top face and the sum over them is large: `d < k (1 - 1/sqrt W)`.
#[test]
fn a_rounded_row_of_heavily_overlapping_cuboids_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = cuboid(&mut g, [0.0, 0.0, 0.0], [20.0, 20.0, 20.0]);
    let p = linear(&mut g, b, 12, [2.0, 0.0, 0.0], true, Fillet::of(lit(8.0)));
    holds("linear overlapping cuboids, join, fillet 8", check(&doc, &g, p, None));
}

/// Two bars crossing at the axis (a ring of four about Z), tops coplanar over the crossing.
#[test]
fn a_rounded_ring_of_crossing_bars_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = cuboid(&mut g, [0.0, 0.0, 0.0], [60.0, 10.0, 10.0]);
    let p = radial(&mut g, b, 4, Frame::along(Axis::Z), true, Fillet::of(lit(6.0)));
    holds("radial crossing bars, join, fillet 6", check(&doc, &g, p, None));
}

/// The same ring holding only the copies, so the rounding runs between copies one to three.
#[test]
fn a_rounded_ring_not_holding_its_body_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = cuboid(&mut g, [0.0, 0.0, 0.0], [60.0, 10.0, 10.0]);
    let p = radial(&mut g, b, 4, Frame::along(Axis::Z), false, Fillet::of(lit(6.0)));
    holds("radial crossing bars, no join, fillet 6", check(&doc, &g, p, None));
}

/// Spokes of a hub that meet only near the axis, which is what the fillet is for on a hub.
#[test]
fn a_rounded_ring_of_spokes_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let b = cuboid(&mut g, [20.0, 0.0, 0.0], [40.0, 8.0, 8.0]);
    let frame = Frame::at_along(Axis::Z, [lit(0.0), lit(0.0), lit(0.0)]);
    let p = radial(&mut g, b, 6, frame, true, Fillet::of(lit(5.0)));
    holds("radial spokes, join, fillet 5", check(&doc, &g, p, None));
}

// ---------------------------------------------------------------------------------------------
// Lattice
// ---------------------------------------------------------------------------------------------

/// A lattice on its own tiles all of space, and `Unbounded` is the right answer rather than a
/// lazy one: it is solid arbitrarily far from the origin.
#[test]
fn a_bare_lattice_is_unbounded_because_it_is_solid_everywhere() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let l = g.add(Node::Lattice {
        cell: LatticeCell::Gyroid,
        frame: Frame::default(),
        map: CellMap::Rectangular { cell: lit(10.0) },
        phase: LatticePhase::sheet(lit(1.0)),
    });
    let extent = graph_extent(&doc, &g, l);
    let field = GraphField::compile_graph(&doc, &g, 1e-4).expect("compiles");
    // Solid somewhere in a cell far away, which is what makes Unbounded the honest answer.
    let far = (0..1000)
        .map(|i| Point3::new(5000.0 + i as f64 * 0.01, 5000.0, 5000.0))
        .any(|p| field.distance(p) < 0.0);
    println!("CASE bare gyroid lattice: extent {extent:?}; solid 5 m away: {far}");
    assert_eq!(extent, Extent::Unbounded);
    assert!(far, "a gyroid has material in every cell");
}

/// A lattice confined to a solid by a hard intersection.
fn confined(cell: LatticeCell, map: CellMap, phase: LatticePhase, solid: impl FnOnce(&mut Graph) -> NodeId) -> (Document, Graph, NodeId) {
    let doc = Document::new();
    let mut g = Graph::empty();
    let l = g.add(Node::Lattice { cell, frame: Frame::default(), map, phase });
    let s = solid(&mut g);
    let i = g.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![l.into(), s.into()] });
    (doc, g, i)
}

#[test]
fn a_rectangular_gyroid_sheet_in_a_cuboid_s_extent_holds_every_solid_point() {
    let (doc, g, i) = confined(
        LatticeCell::Gyroid,
        CellMap::Rectangular { cell: lit(10.0) },
        LatticePhase::sheet(lit(1.0)),
        |g| cuboid(g, [3.0, -2.0, 1.0], [40.0, 30.0, 20.0]),
    );
    holds("gyroid sheet rectangular in cuboid", check(&doc, &g, i, None));
}

#[test]
fn a_rectangular_bcc_strut_in_a_cuboid_s_extent_holds_every_solid_point() {
    let (doc, g, i) = confined(
        LatticeCell::Bcc,
        CellMap::Rectangular { cell: lit(10.0) },
        LatticePhase::Strut { radius: lit(1.0) },
        |g| cuboid(g, [0.0, 0.0, 0.0], [40.0, 30.0, 20.0]),
    );
    holds("bcc strut rectangular in cuboid", check(&doc, &g, i, None));
}

#[test]
fn a_rectangular_honeycomb_in_a_cuboid_s_extent_holds_every_solid_point() {
    let (doc, g, i) = confined(
        LatticeCell::Honeycomb,
        CellMap::Rectangular { cell: lit(8.0) },
        LatticePhase::sheet(lit(1.0)),
        |g| cuboid(g, [0.0, 0.0, 0.0], [40.0, 30.0, 20.0]),
    );
    holds("honeycomb rectangular in cuboid", check(&doc, &g, i, None));
}

#[test]
fn a_rectangular_voronoi_foam_in_a_cuboid_s_extent_holds_every_solid_point() {
    let (doc, g, i) = confined(
        LatticeCell::VoronoiFoam { jitter: lit(0.8), seed: 7 },
        CellMap::Rectangular { cell: lit(10.0) },
        LatticePhase::sheet(lit(1.0)),
        |g| cuboid(g, [0.0, 0.0, 0.0], [40.0, 30.0, 20.0]),
    );
    holds("voronoi foam rectangular in cuboid", check(&doc, &g, i, None));
}

#[test]
fn a_cylindrical_gyroid_solid_in_a_cylinder_s_extent_holds_every_solid_point() {
    let (doc, g, i) = confined(
        LatticeCell::Gyroid,
        CellMap::Cylindrical { cell_radius: lit(8.0), cell_height: lit(8.0), arc_count: 12 },
        LatticePhase::Solid { density: lit(0.3) },
        |g| {
            g.add(Node::Cylinder { edge: Default::default(),
                ends: fieldform_scene::Ends::At {
                    from: Place::At([lit(0.0), lit(0.0), lit(-15.0)]),
                    to: Place::At([lit(0.0), lit(0.0), lit(15.0)]),
                },
                radius: lit(25.0),
            })
        },
    );
    holds("gyroid solid cylindrical in cylinder", check(&doc, &g, i, None));
}

#[test]
fn a_cylindrical_voronoi_beam_in_a_cuboid_s_extent_holds_every_solid_point() {
    let (doc, g, i) = confined(
        LatticeCell::VoronoiBeam { jitter: lit(0.6), seed: 3 },
        CellMap::Cylindrical { cell_radius: lit(8.0), cell_height: lit(8.0), arc_count: 10 },
        LatticePhase::Strut { radius: lit(1.0) },
        |g| cuboid(g, [0.0, 0.0, 0.0], [50.0, 50.0, 20.0]),
    );
    holds("voronoi beam cylindrical in cuboid", check(&doc, &g, i, None));
}

#[test]
fn a_spherical_schwarz_p_sheet_in_a_sphere_s_extent_holds_every_solid_point() {
    let (doc, g, i) = confined(
        LatticeCell::SchwarzP,
        CellMap::Spherical { cell_radius: lit(8.0), azimuth_count: 12, polar_count: 6 },
        LatticePhase::sheet(lit(1.0)),
        |g| sphere(g, [0.0, 0.0, 0.0], 25.0),
    );
    holds("schwarz P sheet spherical in sphere", check(&doc, &g, i, None));
}

/// A rounded intersection, whose box is the hard one grown by the radius.
#[test]
fn a_gyroid_blended_into_a_cuboid_s_extent_holds_every_solid_point() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let l = g.add(Node::Lattice {
        cell: LatticeCell::Gyroid,
        frame: Frame::default(),
        map: CellMap::Rectangular { cell: lit(10.0) },
        phase: LatticePhase::sheet(lit(1.5)),
    });
    let s = cuboid(&mut g, [0.0, 0.0, 0.0], [40.0, 30.0, 20.0]);
    let b = g.add(Node::Blend { op: BooleanOp::Intersection, a: l.into(), b: s.into(), fillet: Fillet::of(lit(3.0)) });
    holds("gyroid sheet blend-intersect cuboid, fillet 3", check(&doc, &g, b, None));
}

/// An open quad panel, `n` by `n` faces of `pitch`, with its middle vertices lifted by `bump`.
fn panel(n: usize, pitch: f64, bump: f64) -> QuadMesh {
    let mut positions = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let inner = i > 0 && i < n && j > 0 && j < n;
            positions.push(Point3::new(i as f64 * pitch, j as f64 * pitch, if inner { bump } else { 0.0 }));
        }
    }
    let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
    let mut faces = Vec::new();
    for j in 0..n {
        for i in 0..n {
            faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    QuadMesh::from_faces(positions, &faces).expect("a panel")
}

fn conformal(surface: QuadMesh, thickness: Option<f64>, cell: LatticeCell, phase: LatticePhase) -> (Document, Graph, NodeId) {
    let mut doc = Document::new();
    let surface = doc.add_quad_mesh(surface, "panel");
    let mut g = Graph::empty();
    let l = g.add(Node::Lattice {
        cell,
        frame: Frame::default(),
        map: CellMap::Conformal { surface, cells_per_face: 2, depth: lit(10.0), thickness: thickness.map(lit) },
        phase,
    });
    (doc, g, l)
}

/// The one lattice with a box of its own. Off the rim of the open panel the shell is measured to
/// the boundary, so it rounds over the edge rather than running on in the panel's plane.
#[test]
fn a_shelled_conformal_gyroid_on_a_flat_panel_s_extent_holds_every_solid_point() {
    let (doc, g, l) = conformal(panel(3, 10.0, 0.0), Some(8.0), LatticeCell::Gyroid, LatticePhase::sheet(lit(1.0)));
    holds("conformal gyroid, flat panel, shell 8", check(&doc, &g, l, None));
}

#[test]
#[ignore = "drawn-surface slack of a few tenths of a millimetre or less (bake margin, tessellation, rounding) -- tile-mask-design-01941c5.md 1.8 R7"]
fn a_shelled_conformal_bcc_on_a_domed_panel_s_extent_holds_every_solid_point() {
    let (doc, g, l) = conformal(panel(4, 10.0, 8.0), Some(6.0), LatticeCell::Bcc, LatticePhase::Strut { radius: lit(1.0) });
    holds("conformal bcc, domed panel, shell 6", check(&doc, &g, l, None));
}

#[test]
#[ignore = "drawn-surface slack of a few tenths of a millimetre or less (bake margin, tessellation, rounding) -- tile-mask-design-01941c5.md 1.8 R7"]
fn a_shelled_conformal_solid_on_a_domed_panel_s_extent_holds_every_solid_point() {
    let (doc, g, l) = conformal(panel(4, 10.0, 8.0), Some(6.0), LatticeCell::SchwarzD, LatticePhase::Solid { density: lit(0.5) });
    holds("conformal schwarz D solid, domed panel, shell 6", check(&doc, &g, l, None));
}

/// Without a shell a conformal lattice tiles outward forever, so it is unbounded on its own and
/// bounded by what it is intersected with.
#[test]
fn an_unshelled_conformal_gyroid_in_a_cuboid_s_extent_holds_every_solid_point() {
    let (doc, mut g, l) = conformal(panel(3, 10.0, 0.0), None, LatticeCell::Gyroid, LatticePhase::sheet(lit(1.0)));
    assert_eq!(graph_extent(&doc, &g, l), Extent::Unbounded, "a conformal tiling with no shell bounds nothing");
    let s = cuboid(&mut g, [15.0, 15.0, 0.0], [30.0, 30.0, 12.0]);
    let i = g.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![l.into(), s.into()] });
    holds("conformal gyroid, no shell, in cuboid", check(&doc, &g, i, None));
}

// ---------------------------------------------------------------------------------------------
// Cage and mesh
// ---------------------------------------------------------------------------------------------

#[test]
fn a_cube_cage_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let cage = doc.add_cage(Cage::cube_of(20.0));
    let mut g = Graph::empty();
    let c = g.add(Node::Cage(cage));
    holds("cage cube_of(20)", check(&doc, &g, c, None));
}

/// A corner dragged out, so the hull is no longer a cube and the limit surface is pulled toward
/// the corner without reaching it.
#[test]
fn a_cage_with_a_corner_dragged_out_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let id = doc.add_cage(Cage::cube_of(20.0));
    {
        let c = doc.cage_mesh_mut(id).expect("there");
        let corner = c
            .vertex_ids()
            .find(|&v| (c.vertex_position(v) - Point3::new(10.0, 10.0, 10.0)).norm() < 1e-9)
            .expect("a cube has that corner");
        c.set_vertex_position(corner, Point3::new(24.0, 18.0, 30.0));
    }
    let mut g = Graph::empty();
    let c = g.add(Node::Cage(id));
    holds("cage cube_of(20), corner dragged out", check(&doc, &g, c, None));
}

/// A cage as the body of a ring, which is where the two halves of this group meet.
#[test]
fn a_ring_of_cages_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let cage = doc.add_cage(Cage::cube_of(12.0));
    let mut g = Graph::empty();
    let c = g.add(Node::Cage(cage));
    let frame = Frame::at_along(Axis::Y, [lit(20.0), lit(0.0), lit(0.0)]);
    let p = radial(&mut g, c, 5, frame, true, Fillet::hard());
    holds("radial of cages about Y", check(&doc, &g, p, None));
}

/// A closed tetrahedron, wound outward.
#[test]
fn a_closed_mesh_s_extent_holds_every_solid_point() {
    let mut doc = Document::new();
    let triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let mesh = MeshData {
        name: "tetra".into(),
        positions: vec![[5.0, -4.0, 2.0], [25.0, -4.0, 2.0], [5.0, 16.0, 2.0], [5.0, -4.0, 22.0]],
        open_edges: MeshData::count_open_edges(&triangles),
        triangles,
        tolerance_mm: None,
        source: None,
        edges: Vec::new(),
    };
    assert_eq!(mesh.open_edges, 0);
    let id = doc.add_mesh(mesh);
    let mut g = Graph::empty();
    let m = g.add(Node::Mesh(id));
    holds("mesh tetrahedron", check(&doc, &g, m, None));
}

// ---------------------------------------------------------------------------------------------
// Ramp
// ---------------------------------------------------------------------------------------------

/// A ramp is a number, not a shape: never a root, its own extent is `Empty`, and it cannot widen
/// a part's box. Checked with a ramp reading a sphere, since a wired one is the one that could
/// have been mistaken for a body.
#[test]
fn a_ramp_is_not_a_body_and_bounds_nothing() {
    let doc = Document::new();
    let mut g = Graph::empty();
    let ball = sphere(&mut g, [0.0, 0.0, 0.0], 10.0);
    let ramp = g.add(Node::Ramp {
        input: ball.into(),
        in_min: lit(-10.0),
        in_max: lit(10.0),
        out_min: lit(-5.0),
        out_max: lit(5.0),
        shape: RampShape::Linear,
    });
    let unwired = g.add(Node::Ramp {
        input: Operand::empty(),
        in_min: lit(0.0),
        in_max: lit(1.0),
        out_min: lit(-3.0),
        out_max: lit(3.0),
        shape: RampShape::Linear,
    });
    let roots = g.roots();
    println!(
        "CASE ramp: roots {roots:?}, ramp extent {:?}, unwired ramp extent {:?}, part extent {:?}",
        graph_extent(&doc, &g, ramp),
        graph_extent(&doc, &g, unwired),
        fieldform_field::part_extent(&doc, &g),
    );
    assert!(!roots.contains(&ramp) && !roots.contains(&unwired), "a ramp is never a body");
    assert!(!g.node(ramp).unwrap().is_body());
    assert_eq!(graph_extent(&doc, &g, ramp), Extent::Empty);
    assert_eq!(graph_extent(&doc, &g, unwired), Extent::Empty);
    // The ramp consumes the sphere, so the part has no body at all and its box is empty.
    assert_eq!(fieldform_field::part_extent(&doc, &g), Extent::Empty);
}
