// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Walking a node's inputs allocates nothing, and hands over what `Node::inputs` lists, in its
//! order.
//!
//! **Why the walk has to be free.** `Graph::roots`, the evaluation order, `Graph::consumed`, who
//! reads a block and `Document::settle_ownership` each ask every node of a part for its inputs, and
//! they run on every edit — each frame of a drag included, since every drag frame is a graph edit
//! that settles. The list `inputs` builds was a fresh allocation every time it was asked, and so
//! were the lists it is built from: a parameter list, a frame list, a place list, a turn's or a
//! frame's blocks. A sphere standing on a block cost four, and a two-sided extrude of a wall eight.
//!
//! **Its own binary, because counting allocations means owning the allocator.** The count is kept
//! per thread, so the harness running tests side by side cannot charge one with another's.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use fieldform_scene::{
    AngleSides, Attractor, BooleanOp, CageId, CellMap, Corners, CurveId, Ends, Fillet, Frame,
    Graph, LatticeCell, LatticePhase, LoftSmooth, MeshId, Node, NodeId, Outline, ParamRef,
    Operand, PatternKind, PatternOptions, PatternSecond, Place, QuadMeshId, RampShape, Section,
    Sides,
    Sweep, SweepLaw, Taper, Turn, WallEnds, WallSide,
};

/// The system allocator, with every allocation counted on the thread that made it.
struct Counted;

thread_local! {
    static MADE: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // `try_with`, because a thread being torn down can still allocate on its way out.
        let _ = MADE.try_with(|made| made.set(made.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTED: Counted = Counted;

/// How many allocations `work` made on this thread.
fn allocations_during(work: impl FnOnce()) -> usize {
    let before = MADE.with(Cell::get);
    work();
    MADE.with(Cell::get) - before
}

/// One node, and the inputs it reads in the order `Node::inputs` gives them: its operands, then
/// whatever drives a parameter in the order the parameters are listed, then what it is pointed at
/// — its frames' blocks, its places', its turn's and its ends'.
///
/// Every edge names a block of its own, so a list in the wrong order cannot match by accident.
struct Case {
    what: &'static str,
    node: Node,
    inputs: Vec<NodeId>,
}

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn driven(by: NodeId) -> ParamRef {
    ParamRef::Field(by)
}

/// Every kind of node, each with every kind of edge it can hold set to a block of its own, and
/// the kinds with more than one way to hold an edge more than once.
fn every_kind_with_every_edge() -> Vec<Case> {
    let n = NodeId::new;
    let curve = CurveId::new;
    let mut cases = Vec::new();
    let mut case = |what: &'static str, node: Node, inputs: Vec<NodeId>| {
        cases.push(Case { what, node, inputs });
    };
    let wall = |thickness: ParamRef| Outline::Wall {
        curve: CurveId::new(),
        thickness,
        side: WallSide::Centre,
        ends: WallEnds::Square,
    };

    case("a cage", Node::Cage(CageId::new()), vec![]);
    case("a mesh", Node::Mesh(MeshId::new()), vec![]);

    let (r, p) = (n(), n());
    case(
        "a sphere standing on a block",
        Node::Sphere { centre: Place::From { node: p }, radius: driven(r) },
        vec![r, p],
    );
    let (x, z, r) = (n(), n(), n());
    case(
        "a sphere at a driven place",
        Node::Sphere { centre: Place::At([driven(x), lit(1.0), driven(z)]), radius: driven(r) },
        vec![x, z, r],
    );

    let (s0, s2, t0, t1) = (n(), n(), n(), n());
    case(
        "a cuboid faced between two blocks",
        Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [driven(s0), lit(2.0), driven(s2)],
            turn: Turn::Between { from: t0, to: t1 },
        },
        vec![s0, s2, t0, t1],
    );
    let (c, a) = (n(), n());
    case(
        "a cuboid on a block, turned by a driven angle",
        Node::Cuboid { edge: Default::default(),
            centre: Place::From { node: c },
            size: [lit(1.0); 3],
            turn: Turn::By([lit(0.0), driven(a), lit(0.0)]),
        },
        vec![a, c],
    );
    case(
        "a cuboid on the start of a curve, aimed along another",
        Node::Cuboid { edge: Default::default(),
            centre: Place::StartOf { curve: curve() },
            size: [lit(1.0); 3],
            turn: Turn::Along { curve: curve() },
        },
        vec![],
    );

    let (e, c) = (n(), n());
    case(
        "a skewed box on a block",
        Node::SkewedBox {
            centre: Place::From { node: c },
            edges: [[lit(1.0); 3], [lit(0.0), lit(1.0), driven(e)], [lit(0.0); 3]],
        },
        vec![e, c],
    );

    let (f, z, r) = (n(), n(), n());
    case(
        "a cylinder from a block to a driven place",
        Node::Cylinder { edge: Default::default(),
            ends: Ends::At {
                from: Place::From { node: f },
                to: Place::At([lit(0.0), lit(0.0), driven(z)]),
            },
            radius: driven(r),
        },
        vec![z, r, f],
    );
    let (l, b, on) = (n(), n(), n());
    case(
        "a cone standing on a block",
        Node::Cone { edge: Default::default(),
            ends: Ends::On { node: on, length: driven(l) },
            base_radius: driven(b),
            top_radius: lit(0.0),
        },
        vec![l, b, on],
    );
    let (s, e) = (n(), n());
    case(
        "a capsule along a curve",
        Node::Capsule {
            ends: Ends::From { curve: curve() },
            start_radius: driven(s),
            end_radius: driven(e),
        },
        vec![s, e],
    );

    let (m, c, t) = (n(), n(), n());
    case(
        "a torus on a block, faced by another",
        Node::Torus {
            centre: Place::From { node: c },
            major_radius: driven(m),
            minor_radius: lit(1.0),
            turn: Turn::From { node: t },
        },
        vec![m, c, t],
    );

    let (o, t0, t1) = (n(), n(), n());
    case(
        "a plane on a block whose facing went to a placement",
        Node::Plane {
            origin: Place::From { node: o },
            flipped: false,
            turn: Turn::Absorbed { nodes: [Some(t0), Some(t1)], curve: None },
        },
        vec![o, t0, t1],
    );
    let (x, y) = (n(), n());
    case(
        "a plane at a driven place, turned by a driven angle",
        Node::Plane {
            origin: Place::At([driven(x), lit(0.0), lit(0.0)]),
            flipped: true,
            turn: Turn::By([lit(0.0), driven(y), lit(0.0)]),
        },
        vec![x, y],
    );
    case(
        "a plane aimed at the start of a curve",
        Node::Plane {
            origin: Place::At([lit(0.0); 3]),
            flipped: false,
            turn: Turn::Tangent { curve: curve() },
        },
        vec![],
    );

    let p = n();
    case("a point at a driven place", Node::Point { at: [lit(0.0), driven(p), lit(0.0)] }, vec![p]);

    let r = n();
    case("a pipe", Node::Pipe { curve: curve(), radius: driven(r) }, vec![r]);

    let (t, sc, tw, ex) = (n(), n(), n(), n());
    case(
        "a sweep of a wall, scaled and twisted",
        Node::Sweep { rims: Default::default(),
            outline: wall(driven(t)),
            spine: curve(),
            section: Section::Square,
            law: SweepLaw {
                taper: Taper::Scale(driven(sc)),
                corners: Corners::Round,
                twist: driven(tw),
                shape: RampShape::Smooth,
                extent: [lit(0.0), driven(ex)],
            },
        },
        vec![t, sc, tw, ex],
    );

    let (a, b) = (n(), n());
    case(
        "a loft through two walls and a profile",
        Node::Loft { rims: Default::default(),
            sections: vec![wall(driven(a)), Outline::closed(curve()), wall(driven(b))],
            spine: Some(curve()),
            smooth: LoftSmooth::G1,
        },
        vec![a, b],
    );

    let (pt, po, j, d, th, f0, f1) = (n(), n(), n(), n(), n(), n(), n());
    case(
        "a Voronoi sheet on a surface, framed between two blocks",
        Node::Lattice {
            cell: LatticeCell::VoronoiFoam { jitter: driven(j), seed: 7 },
            frame: Frame::Between { from: f0, to: f1 },
            map: CellMap::Conformal {
                surface: QuadMeshId::new(),
                cells_per_face: 2,
                depth: driven(d),
                thickness: Some(driven(th)),
            },
            phase: LatticePhase::Sheet { thickness: driven(pt), offset: driven(po) },
        },
        vec![pt, po, j, d, th, f0, f1],
    );
    let (density, o, ax, cr, ch) = (n(), n(), n(), n(), n());
    case(
        "a solid gyroid on a cylinder, framed by driven numbers",
        Node::Lattice {
            cell: LatticeCell::Gyroid,
            frame: Frame::At {
                origin: [driven(o), lit(0.0), lit(0.0)],
                axis: [lit(0.0), lit(0.0), driven(ax)],
            },
            map: CellMap::Cylindrical {
                cell_radius: driven(cr),
                cell_height: driven(ch),
                arc_count: 8,
            },
            phase: LatticePhase::Solid { density: driven(density) },
        },
        vec![density, o, ax, cr, ch],
    );
    let (radius, cr, f) = (n(), n(), n());
    case(
        "a strut lattice on a sphere, framed by a block",
        Node::Lattice {
            cell: LatticeCell::Bcc,
            frame: Frame::From { node: f },
            map: CellMap::Spherical { cell_radius: driven(cr), azimuth_count: 8, polar_count: 4 },
            phase: LatticePhase::Strut { radius: driven(radius) },
        },
        vec![radius, cr, f],
    );
    let cell = n();
    case(
        "a rectangular lattice framed along a curve",
        Node::Lattice {
            cell: LatticeCell::VoronoiBeam { jitter: lit(0.5), seed: 3 },
            frame: Frame::Along { curve: curve() },
            map: CellMap::Rectangular { cell: driven(cell) },
            phase: LatticePhase::sheet(lit(1.0)),
        },
        vec![cell],
    );

    let (h, dr, sx, w, st, h2, d2) = (n(), n(), n(), n(), n(), n(), n());
    case(
        "an extrude of a wall, two-sided, toward a driven direction",
        Node::Extrude { rims: Default::default(),
            outline: wall(driven(w)),
            sweep: Sweep::Toward { xyz: [driven(sx), lit(0.0), lit(1.0)] },
            start: driven(st),
            height: driven(h),
            sides: Sides::Two { height: driven(h2), draft: driven(d2) },
            draft: driven(dr),
            corners: Corners::Sharp,
        },
        vec![h, dr, sx, w, st, h2, d2],
    );
    case(
        "an extrude of a profile along a curve",
        Node::Extrude { rims: Default::default(),
            outline: Outline::closed(curve()),
            sweep: Sweep::Along { curve: curve() },
            start: lit(0.0),
            height: lit(10.0),
            sides: Sides::Symmetric,
            draft: lit(0.0),
            corners: Corners::Round,
        },
        vec![],
    );

    let (o, w, a, a2) = (n(), n(), n(), n());
    case(
        "a revolve of a wall about a driven axis, two ways",
        Node::Revolve { rims: Default::default(),
            outline: wall(driven(w)),
            frame: Frame::At { origin: [lit(0.0), driven(o), lit(0.0)], axis: [lit(1.0); 3] },
            angle: driven(a),
            sides: AngleSides::Two { angle: driven(a2) },
        },
        vec![o, w, a, a2],
    );
    let f = n();
    case(
        "a revolve about a block's axis",
        Node::Revolve { rims: Default::default(),
            outline: Outline::closed(curve()),
            frame: Frame::From { node: f },
            angle: lit(360.0),
            sides: AngleSides::One,
        },
        vec![f],
    );
    case(
        "a revolve about a curve's tangent",
        Node::Revolve { rims: Default::default(),
            outline: Outline::closed(curve()),
            frame: Frame::Tangent { curve: curve() },
            angle: lit(90.0),
            sides: AngleSides::Symmetric,
        },
        vec![],
    );

    let (a, b, k) = (n(), n(), n());
    case(
        "a blend with a driven radius",
        Node::Blend {
            op: BooleanOp::Union,
            a: a.into(),
            b: b.into(),
            fillet: Fillet::of(driven(k)),
        },
        vec![a, b, k],
    );

    let (x, y) = (n(), n());
    case(
        "a boolean handed one body twice",
        Node::Boolean { op: BooleanOp::Difference, inputs: vec![x.into(), y.into(), x.into()] },
        vec![x, y, x],
    );

    let (a, b, t) = (n(), n(), n());
    case("a morph", Node::Morph { a: a.into(), b: b.into(), t: driven(t) }, vec![a, b, t]);

    let (input, d, f) = (n(), n(), n());
    case(
        "a twist framed by a block",
        Node::Twist { input: input.into(), degrees_per_mm: driven(d), frame: Frame::From { node: f } },
        vec![input, d, f],
    );

    let (input, p, s) = (n(), n(), n());
    case(
        "an attractor drawn to a point",
        Node::Attract {
            input: input.into(),
            toward: Attractor::Point(p),
            strength: driven(s),
            falloff: lit(10.0),
        },
        vec![input, p, s],
    );
    let (input, f) = (n(), n());
    case(
        "an attractor drawn to a curve",
        Node::Attract {
            input: input.into(),
            toward: Attractor::Curve(curve()),
            strength: lit(0.5),
            falloff: driven(f),
        },
        vec![input, f],
    );

    let (input, k, count, angle, f0, f1) = (n(), n(), n(), n(), n(), n());
    case(
        "a ring framed between two blocks, rounded, counted and opened by driven numbers",
        Node::Pattern {
            input: input.into(),
            count: driven(count),
            kind: PatternKind::Radial { frame: Frame::Between { from: f0, to: f1 } },
            join: false,
            fillet: Fillet::of(driven(k)),
            options: PatternOptions { angle: driven(angle), ..Default::default() },
        },
        vec![input, k, count, angle, f0, f1],
    );
    let (input, s0, s2, second) = (n(), n(), n(), n());
    case(
        "a grid stepped and counted by driven numbers",
        Node::Pattern {
            input: input.into(),
            count: lit(3.0),
            kind: PatternKind::Linear { step: [driven(s0), lit(0.0), lit(0.0)] },
            join: true,
            fillet: Fillet::hard(),
            options: PatternOptions {
                second: Some(PatternSecond {
                    step: [lit(0.0), lit(0.0), driven(s2)],
                    count: driven(second),
                }),
                ..Default::default()
            },
        },
        vec![input, s0, s2, second],
    );

    let (input, o) = (n(), n());
    case(
        "a mirror at a driven origin",
        Node::Mirror {
            input: input.into(),
            frame: Frame::At { origin: [driven(o), lit(0.0), lit(0.0)], axis: [lit(1.0); 3] },
            join: true,
        },
        vec![input, o],
    );
    let (input, f) = (n(), n());
    case(
        "a mirror in a block",
        Node::Mirror { input: input.into(), frame: Frame::From { node: f }, join: false },
        vec![input, f],
    );

    let (input, t) = (n(), n());
    case("a shell", Node::Shell { input: input.into(), thickness: driven(t) }, vec![input, t]);

    let (input, lo, hi) = (n(), n(), n());
    case(
        "a ramp reading a block",
        Node::Ramp {
            input: input.into(),
            in_min: driven(lo),
            in_max: lit(1.0),
            out_min: lit(0.0),
            out_max: driven(hi),
            shape: RampShape::Smoother,
        },
        vec![input, lo, hi],
    );
    case(
        "a ramp reading nothing yet",
        Node::Ramp {
            input: Operand::empty(),
            in_min: lit(0.0),
            in_max: lit(1.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: RampShape::Linear,
        },
        vec![],
    );

    let (input, d) = (n(), n());
    case("an offset", Node::Offset { input: input.into(), distance: driven(d) }, vec![input, d]);

    let (input, tx, ry, sc, about) = (n(), n(), n(), n(), n());
    case(
        "a placement turning about a block",
        Node::Transform {
            input: input.into(),
            translation: [driven(tx), lit(0.0), lit(0.0)],
            rotation: [lit(0.0), driven(ry), lit(0.0)],
            scale: driven(sc),
            about: Place::From { node: about },
        },
        vec![input, tx, ry, sc, about],
    );

    let unknown: Node = serde_json::from_value(serde_json::json!({ "kind": "from_a_newer_build" }))
        .expect("a kind this build has not heard of is kept rather than refused");
    case("a kind this build cannot read", unknown, vec![]);

    cases
}

/// The kind's name, by a match with no wildcard: a kind added to `Node` does not compile here until
/// it is named, and then `the_sample_holds_every_kind` fails until it has a case.
fn kind(node: &Node) -> &'static str {
    match node {
        Node::Cage(_) => "cage",
        Node::Mesh(_) => "mesh",
        Node::Sphere { .. } => "sphere",
        Node::Cuboid { .. } => "cuboid",
        Node::SkewedBox { .. } => "skewed box",
        Node::Cylinder { .. } => "cylinder",
        Node::Cone { .. } => "cone",
        Node::Lattice { .. } => "lattice",
        Node::Extrude { .. } => "extrude",
        Node::Revolve { .. } => "revolve",
        Node::Plane { .. } => "plane",
        Node::Point { .. } => "point",
        Node::Torus { .. } => "torus",
        Node::Capsule { .. } => "capsule",
        Node::Pipe { .. } => "pipe",
        Node::Sweep { .. } => "sweep",
        Node::Loft { .. } => "loft",
        Node::Blend { .. } => "blend",
        Node::Boolean { .. } => "boolean",
        Node::Morph { .. } => "morph",
        Node::Twist { .. } => "twist",
        Node::Attract { .. } => "attract",
        Node::Pattern { .. } => "pattern",
        Node::Mirror { .. } => "mirror",
        Node::Shell { .. } => "shell",
        Node::Ramp { .. } => "ramp",
        Node::Offset { .. } => "offset",
        Node::Transform { .. } => "transform",
        Node::Unknown(_) => "unknown",
    }
}

/// The sample the other tests read is every kind there is, so none of them passes by not asking.
#[test]
fn the_sample_holds_every_kind() {
    let held: std::collections::BTreeSet<&str> =
        every_kind_with_every_edge().iter().map(|case| kind(&case.node)).collect();
    assert_eq!(held.len(), 29, "a kind of node has no case: {held:?}");
}

/// **A node's inputs are its operands, then what drives its parameters, then what it is pointed
/// at** — each in its own order, and a block named twice listed twice.
///
/// Held against lists written out here rather than taken from the code, because the order is not
/// decoration: the evaluation order pushes inputs in it, which names every function in the
/// generated shader, and a delete picks the block standing in for a deleted operand from the front
/// of it. The walk and the list must agree with these and with each other.
#[test]
fn a_nodes_inputs_are_its_operands_then_its_drivers_then_what_it_is_aimed_at() {
    for case in every_kind_with_every_edge() {
        assert_eq!(case.node.inputs(), case.inputs, "listing {}", case.what);
        let mut walked = Vec::new();
        case.node.for_each_input(|id| walked.push(id));
        assert_eq!(walked, case.inputs, "walking {}", case.what);
    }
}

/// **Walking a node's inputs allocates nothing**, whatever kind it is and whatever it is wired to.
#[test]
fn walking_a_nodes_inputs_allocates_nothing() {
    let cases = every_kind_with_every_edge();
    // The arm that must count something: the list allocates, so a nought below is the walk's and
    // not a counter that counts nothing.
    let listed = allocations_during(|| {
        for case in &cases {
            black_box(case.node.inputs());
        }
    });
    assert!(listed >= cases.iter().filter(|c| !c.inputs.is_empty()).count(), "{listed}");
    for case in &cases {
        let mut seen = 0;
        let made = allocations_during(|| {
            case.node.for_each_input(|id| {
                black_box(id);
                seen += 1;
            })
        });
        assert_eq!(made, 0, "walking {} allocated", case.what);
        assert_eq!(seen, case.inputs.len(), "walking {} missed an edge", case.what);
    }
}

/// **Asking who reads a block allocates nothing either**, and finds the one block that does.
///
/// `Graph::consumers_of` and `Graph::is_consumed` ask every node of the part whether it reads the
/// block, so a list per node was a list per node per question.
#[test]
fn asking_who_reads_a_block_allocates_nothing() {
    let mut graph = Graph::empty();
    let mut read: Vec<(NodeId, NodeId)> = Vec::new();
    for case in every_kind_with_every_edge() {
        let inputs = case.inputs.clone();
        let reader = graph.add(case.node);
        read.extend(inputs.into_iter().map(|input| (input, reader)));
    }
    let blocks: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
    for (input, reader) in &read {
        let mut found = Vec::with_capacity(4);
        let mut consumed = false;
        let made = allocations_during(|| {
            found.extend(graph.consumers_of(*input));
            consumed = graph.is_consumed(*input);
        });
        assert_eq!(made, 0, "asking who reads {input:?} allocated");
        assert_eq!(found, vec![*reader], "who reads {input:?}");
        assert!(consumed, "whether {input:?} is read");
    }
    for block in blocks {
        let made = allocations_during(|| {
            assert!(!graph.is_consumed(block), "nothing reads {block:?}");
        });
        assert_eq!(made, 0, "asking whether {block:?} is read allocated");
    }
}

/// **A node's two parameter lists are in the same order**, for every kind.
///
/// `param_refs` and `param_refs_mut` are two spellings of one list, and an index into it is a
/// handle — the app reaches a socket by `param_refs_mut().get_mut(slot)`, and the drivers `inputs`
/// reports come off `param_refs`. Every slot is numbered through the one list and read back
/// through the other.
#[test]
fn every_kinds_two_parameter_lists_are_in_the_same_order() {
    for case in every_kind_with_every_edge() {
        let mut numbered = case.node.clone();
        let slots = numbered.param_refs_mut().len();
        for (i, slot) in numbered.param_refs_mut().into_iter().enumerate() {
            *slot = lit(i as f64);
        }
        let read: Vec<ParamRef> = numbered.param_refs().into_iter().copied().collect();
        let expected: Vec<ParamRef> = (0..slots).map(|i| lit(i as f64)).collect();
        assert_eq!(read, expected, "the two lists disagree for {}", case.what);
    }
}
