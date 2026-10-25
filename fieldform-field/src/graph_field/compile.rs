// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Compiling a part's graph into a list of ops, and evaluating that list as one field.

use super::{
    LeafCache, bake_to_tolerance, blend, blend_divisor, build_loft, driven_blend_divisor,
    driven_lattice_divisor, driven_morph_divisor, driven_offset_divisor, driven_pipe_divisor,
    driven_refusal,
    euler_degrees_to_rotation, extrude_section, levels_for_tolerance, lipschitz_of, loft_parts,
    mesh_size, mirror_one_side, normalization_of, operand_extent, pattern_grid_anchor_in,
    pattern_join_divisor, pattern_origin_in, pattern_reach_axes_in, pattern_reach_in,
    profile_reads_out_to, ramp_lipschitz, resolve_ends, resolve_frame, resolve_pivot,
    resolve_place_for, resolve_placement, resolve_point, revolve_wedge, rotation_from_z,
    smooth_min_interval, surface_levels, sweep_direction, sweep_law, sweep_offset,
    sweep_outline_section, taper_ratio, tapered, to_local, twist_stretch, wall_about_axis,
    wall_bakes_under, wall_in_plane, with_turns_flattened,
};
use crate::{CageField, CurveField, Normalization, PatternLayout, PatternShape, ScalarField};
use fieldform_scene::{BooleanOp, Document, Graph, Node, NodeId, Part};
use nalgebra::{Isometry3, Point3, Rotation3, Vector3};

#[derive(thiserror::Error, Debug)]
pub enum CompileError {
    #[error("a critical surface's clearance names a parameter that has no value")]
    UnresolvedClearance,
    #[error("graph node {0:?} references a cage that is not in the document")]
    MissingCage(fieldform_scene::CageId),
    /// The cage is in the document and is not a solid: a tool has it open, or it loaded that way.
    /// The string is the state as a person reads it ("open: 1 hole, 4 edges").
    #[error("cage {0} is not a solid ({1})")]
    OpenCage(fieldform_scene::CageId, String),
    #[error("graph node {0:?} references a mesh that is not in the document")]
    MissingMesh(fieldform_scene::MeshId),
    #[error("a conformal lattice references a surface {0:?} that is not in the document")]
    MissingQuadMesh(fieldform_scene::QuadMeshId),
    #[error("the surface {0:?} cannot be followed: {1}")]
    BadQuadMesh(fieldform_scene::QuadMeshId, fieldform_cage::CageError),
    #[error("the mesh {0:?} cannot become a field: {1}")]
    BadMesh(fieldform_scene::MeshId, crate::MeshError),
    #[error("graph node {0:?} references a curve that is not in the document")]
    MissingCurve(fieldform_scene::CurveId),
    #[error(
        "curve {0:?} is open, and a profile has to be closed: a field's sign says whether a \n         point is inside, and an open curve has no inside. Close it, or sweep along it instead."
    )]
    ProfileNotClosed(fieldform_scene::CurveId),
    #[error("curve {0:?} cannot be a profile: {1}")]
    BadProfile(fieldform_scene::CurveId, crate::ProfileError),
    #[error("that sweep has no direction to follow — its curve has no length, or its vector is zero")]
    SweepHasNoDirection,
    #[error(
        "that sweep runs along the profile's own plane, so it would never leave it and there is \
         no solid. Lean it toward the profile's face."
    )]
    SweepLiesInThePlane,
    #[error(
        "a draft of {0:.1} degrees is past vertical, so the walls would fold back through \
         themselves. Between -89 and 89 is a wall; ninety is a plane, and beyond it is a shape \
         nobody drew."
    )]
    DraftTooSteep(f64),
    /// A draft asked to mitre the corners of a **wall**, which V1 does not do — a band has two
    /// faces and mitring both wants its own spec. `SweepError::SharpCornersOnAWall` is the
    /// sweep's half of the same refusal; both exist so that a block saying Sharp never draws
    /// Round without saying so.
    #[error(
        "a draft can only mitre the corners of a closed profile, not of a wall — a band has \
         two faces and mitring both is not built yet; round its corners, or draft the profile \
         the wall runs along instead"
    )]
    SharpCornersOnAWall,
    #[error("{1}")]
    BadLoop(fieldform_scene::CurveId, crate::LoopError),
    #[error(
        "graph node {0:?} is a skewed box whose three edges lie in one plane, so it has no volume \
         — change one edge so the three no longer lie flat"
    )]
    FlatSkewedBox(fieldform_scene::NodeId),
    #[error("curve {0:?} cannot carry that wall: {1}")]
    BadWall(fieldform_scene::CurveId, crate::WallError),
    #[error("curve {0:?} cannot carry that profile: {1}")]
    BadSweep(fieldform_scene::CurveId, crate::SweepError),
    #[error("{0}")]
    BadLoft(crate::LoftError),
    /// A wall offered as a loft section. V1 lofts closed outlines only (`loft-spec.md` §1);
    /// interpolating a band's distance between two bands is a different question, deferred.
    #[error(
        "section {0} of the loft is a wall, and a loft passes through closed sections only — \
         untick Wall on it, or loft the curve it runs along"
    )]
    WallAsLoftSection(usize),
    /// A chamfer asked of a pattern's fold. Across many copies a bevel has no order-free form:
    /// where three copies meet, the bevel between two of them is cut again by the third, and
    /// which pair is cut first changes the shape (`round-chamfer-spec.md` §2). A rounded fold's
    /// `pow`-sum has no such order, which is why G1 and G2 fold and this does not.
    #[error(
        "a pattern's copies cannot be chamfered where they meet, because three copies meeting \
         give a different bevel for each pair cut first — round the fold (G1 or G2), or chamfer \
         two bodies with a blend"
    )]
    ChamferOnAPattern,
    /// A rim rounded or chamfered by more than the solid has to give: past half its length the
    /// two ends' roundings meet, and past the section's depth the rounding meets itself across
    /// the section (`round-chamfer-spec.md` §4). Says which, and the most there is room for.
    #[error(
        "the rims are cut back {reach:.2} mm, and the {what} leaves room for {room:.2} mm at \
         most — make the rims smaller, or the solid {bigger}"
    )]
    RimTooLarge { reach: f64, room: f64, what: &'static str, bigger: &'static str },
    /// A box's or a cylinder's edges cut back further than the solid has room for: past the
    /// smallest half-side, the roundings of opposite edges meet.
    #[error(
        "the edges are cut back {reach:.2} mm, and the {what} leaves room for {room:.2} mm at \
         most — make the edges smaller, or the {block} larger"
    )]
    EdgeTooLarge { reach: f64, room: f64, what: &'static str, block: &'static str },
    /// Rims asked of a sweep along a closed path, which has no ends to round.
    #[error("a sweep along a closed path has no ends, so it has no rims to round — set its rims to hard")]
    RimsOnAClosedPath,
    /// Rims asked of a revolve all the way round, which has no ends to round.
    #[error("a revolve all the way round has no ends, so it has no rims to round — set its rims to hard, or its angle under 360")]
    RimsOnAFullRevolve,
    /// A chamfer's second leg on a blend whose radius follows a ramp.
    #[error("a chamfer with two legs cannot also follow a ramp — give the blend one radius, or one leg")]
    SecondLegNeedsAFixedRadius,
    /// A chamfer's second leg on a box's, a cylinder's or a cone's edges, whose faces have no
    /// first and second.
    #[error("a box's, a cylinder's or a cone's edges are cut back alike, so they take one leg, not two")]
    SecondLegOnAnEdge,
    #[error(
        "curve {0:?} crosses the axis it is being revolved about, so the solid would pass \n         through itself. Move the profile clear of the axis, or move the axis."
    )]
    ProfileCrossesTheAxis(fieldform_scene::CurveId),
    #[error("a radius or size parameter could not be resolved to a number")]
    UnresolvedRadius,
    #[error(
        "a revolve through {0} degrees turns nothing: give it an angle above nought, up to a \
         full turn of 360"
    )]
    RevolveTurnsNothing(f64),
    #[error("a primitive's centre could not be resolved to three numbers")]
    UnresolvedCentre,
    #[error("a blend's radius parameter could not be resolved to a number")]
    UnresolvedBlend,
    #[error(
        "node {0:?} cannot supply a frame \u{2014} a plane gives a place and a facing, and a \
         point gives a place. Aim it at one of those, at a line, or type the frame."
    )]
    NotAFrameSource(fieldform_scene::NodeId),
    #[error(
        "node {0:?} cannot supply a place \u{2014} a point is one, and so is a plane's origin. \
         Aim it at one of those, or type the position."
    )]
    NotAPlaceSource(fieldform_scene::NodeId),
    #[error(
        "blend {0:?} has an input that only bounds distance from below \u{2014} a lattice \
         does, and so does anything built from one \u{2014} so its radius would not be the \
         radius asked for. Join them with a Boolean instead, which depends only on sign and \
         accepts either kind."
    )]
    BlendNeedsDistanceFields(fieldform_scene::NodeId),
    #[error(
        "the graph contains {0}, so this build cannot evaluate it — the rest of the document \
         still opens, edits and saves, and the node is written back unchanged"
    )]
    UnreadableNode(String),
    #[error("the graph has a cycle, so it has no evaluation order")]
    Cyclic,
    #[error("the part has no bodies, so there is nothing to evaluate")]
    Empty,
    #[error("a boolean has no operands, so there is nothing to combine")]
    NothingToCombine,
    /// A block whose slot is empty, named with the slot to fill.
    ///
    /// **The one refusal a hole gets**, wherever the compile meets one: a body that cannot be
    /// evaluated, or a driven parameter whose ramp cannot. It replaces `ReadsSomethingUnwired`,
    /// which asked per node over every edge it held -- [`Graph::incomplete`] is closed downstream
    /// and skipped whole, so this hazard has one mechanism now rather than two.
    #[error(
        "a block has not been told what to read \u{2014} node {node:?}'s slot {slot} is empty, \
         so there is nothing to evaluate. Put a block in it, or take out what reads it."
    )]
    EmptySlot { node: fieldform_scene::NodeId, slot: usize },
    #[error(
        "a transform's scale resolved to {0}, but a scale must be positive: zero collapses          the body to nothing and a negative one turns it inside out"
    )]
    NonPositiveScale(f64),
    /// A driven morph whose operands have no box to bound `sup|B − A|` over.
    #[error(
        "a morph whose amount varies through space needs both bodies to be bounded, and one of \
         these is not \u{2014} intersect it with something first"
    )]
    MorphNeedsBounds,
    /// A morph's `t` outside `[0, 1]`, which is a file saying something it cannot mean.
    #[error("a morph's amount must be between 0 and 1, and this one is {0}")]
    MorphOutOfRange(f64),
    /// The same, said by a ramp rather than by a number.
    #[error(
        "a morph's amount must be between 0 and 1, and the ramp driving this one gives {0} to \
         {1} — change the ramp's output range"
    )]
    MorphDriverOutOfRange(f64, f64),
    #[error("an offset's distance could not be resolved to a number")]
    UnresolvedOffset,
    #[error(
        "a cylindrical lattice needs at least one cell around its circumference, and this one \
         asks for none"
    )]
    EmptyArcCount,
    /// A pattern's count did not give a whole number of copies.
    ///
    /// **What is left after rounding, which is what cannot be repaired at all.** The count is a
    /// `ParamRef`, so an expression can land on 3.5, on nothing, on zero or on a ramp —
    /// [`fieldform_scene::pattern_count`] rounds the first of those and refuses the rest. A fraction
    /// was refused too until the owner reversed it the same day, on the consequence: a variable
    /// driven through a fraction broke the part for a value nobody typed. What stays refused is a
    /// count that would have to be invented rather than rounded. The message carries the reason, so
    /// it can be said beside the field the expression was typed into.
    #[error("{0}")]
    PatternCount(String),
    /// **A pattern's options asked for something that has no copies**: a whole run of one copy, an
    /// arc of one, an angle that is not an angle, a skip of the body itself, nothing left over. The
    /// message names which, so it can be said beside the field that was typed into.
    #[error("{0}")]
    Pattern(String),
    #[error(
        "a conformal lattice needs at least one cell across each face, and this one asks for \
         none"
    )]
    EmptyCellCount,
    #[error(
        "a {0} cell cannot be built as a {1} — a beam has no two sides to wall and a surface has          no struts to give a radius to"
    )]
    CellCannotTakeThatPhase(&'static str, &'static str),
    #[error(
        "a {0} cell cannot be laid over a {1} map yet \u{2014} its walls are measured in \
         millimetres across a flat period, and only a rectangular map is one. Use a rectangular \
         map, and turn or place the lattice to aim it."
    )]
    CellCannotTakeThatMap(&'static str, &'static str),
    #[error(
        "a {0} cell cannot follow a surface yet \u{2014} its seeds are found by searching the \
         cells around a point, and a conformal map gives each face its own coordinates, so the \
         search would stop at every face edge. Use a rectangular, cylindrical or spherical map."
    )]
    CellCannotTakeAConformalMap(&'static str),
    #[error(
        "a twist has to know how far the body reaches from the axis in order to say how far a \
         ray may step, and this one wraps something with no bounds"
    )]
    TwistNeedsBounds,
    #[error(
        "a shell turns a body into a wall of a given thickness, and this one wraps something \
         with no bounds \u{2014} a shelled half-space is an infinite slab, which is not a body"
    )]
    ShellNeedsBounds,
    #[error("a mirror's plane names a node that is not a plane")]
    MirrorNeedsAPlane,
    #[error(
        "a parameter is driven by a node that is not a ramp — only a ramp may drive a \
         parameter, because only a ramp states how fast it can change, and that number is \
         what everything above it divides by"
    )]
    DrivenParameterNeedsARamp,
    #[error(
        "a lattice whose wall, density or strut follows a ramp cannot also be aimed by its own \
         frame yet \u{2014} the ramp would be read in the lattice's frame rather than the \
         world's. Leave the lattice's frame as it is and move the finished body with a \
         placement, which carries both."
    )]
    GradedLatticeCannotBeAimed,
    #[error(
        "{0} cannot be driven by a field yet — only an offset's distance can. The rest resolve \
         to a single number when the graph is compiled, and giving them a value that varies \
         with position would silently cost each one its distance normalization"
    )]
    ParameterCannotBeDriven(&'static str),
    #[error("a ramp's own numbers could not be resolved")]
    UnresolvedRamp,
    #[error("an attractor's strength or falloff radius could not be resolved")]
    UnresolvedAttractor,
    /// An attractor pointed at a block that is not a point.
    #[error("an attractor can be drawn toward a curve or a point, and this names neither")]
    AttractorIsNotAPoint,
    #[error(
        "a ramp maps an input range onto an output range, and this one's input range is empty \
         — every point would map to the same end and the mapping would be a constant"
    )]
    EmptyRampRange,
    #[error("a transform's translation or rotation could not be resolved to three numbers")]
    UnresolvedPlacement,
    #[error("failed to build a cage field: {0}")]
    Cage(#[from] fieldform_cage::CageError),
}

/// Where each node's op landed, indexed by the slot that reads it.
///
/// **A newtype over the map so the arms below stay `slot[input]`: an index, not a question.**
/// Every operand of a node the compile loop reaches is in here, because [`Graph::incomplete`] is
/// closed downstream and is skipped whole -- so a `None` arm at each of the thirteen index sites
/// would be thirteen arms that cannot be taken, and a guard whoever adds the next kind of block
/// has to remember. That arrangement is exactly what let a boolean reading an unwired ramp bring
/// the compile down at the index, which
/// `a_body_with_an_empty_slot_is_refused_by_name_rather_than_panicking` records; the skip set is
/// the one mechanism that replaced it.
#[derive(Default)]
struct Slots(std::collections::HashMap<NodeId, usize>);

impl Slots {
    fn insert(&mut self, id: NodeId, at: usize) {
        self.0.insert(id, at);
    }

    /// Where a node's op landed, for a reader holding an id rather than a slot -- a parameter's
    /// driver, or a body being folded in. `None` for a node this compile skipped.
    fn get(&self, id: &NodeId) -> Option<&usize> {
        self.0.get(id)
    }
}

impl std::ops::Index<&NodeId> for Slots {
    type Output = usize;
    fn index(&self, id: &NodeId) -> &usize {
        &self.0[id]
    }
}

/// So an operand reads as what it is: the op the block in it compiled to.
impl std::ops::Index<&fieldform_scene::Operand> for Slots {
    type Output = usize;
    fn index(&self, operand: &fieldform_scene::Operand) -> &usize {
        match operand.source() {
            Some(id) => &self.0[&id],
            // Unreachable, and said here rather than left to a missing key further in: the blocks
            // with empty slots, and everything above them, are skipped before any op is built, so
            // a slot read here belongs to a node the loop has already got past.
            None => unreachable!("a block with an empty slot never reaches the op list"),
        }
    }
}

/// The refusal a hole gets, named from the slot that is empty.
///
/// **Not from the block that met it**, which is rarely the one to fill in: [`Graph::incomplete`]
/// is closed downstream, so the body refused at the fold can be any number of blocks above the
/// slot that is actually empty. In id order, through [`Graph::first_empty_operand`], so one graph
/// refuses with one block however its nodes happen to be stored -- a message naming a different
/// block each time it is asked is a message nobody can act on. `met` stands in only if the hole
/// has closed between the walk and the refusal, which nothing does.
fn empty_slot(graph: &Graph, met: NodeId) -> CompileError {
    let (node, slot) = graph.first_empty_operand().unwrap_or((met, 0));
    CompileError::EmptySlot { node, slot }
}

/// The open cage a body is, or reads, if any — so the refusal names the cage rather than a hole
/// the body does not have (R12).
pub fn red_cage_behind(
    document: &Document,
    graph: &Graph,
    body: NodeId,
) -> Option<fieldform_scene::CageId> {
    graph.nodes().find_map(|(id, n)| match n {
        Node::Cage(cage)
            if document.cage_is_red(*cage) && (id == body || graph.downstream_of(id).contains(&body)) =>
        {
            Some(*cage)
        }
        _ => None,
    })
}

/// The refusal for a part whose body is an open cage, with the state as a person reads it.
pub(crate) fn open_cage(document: &Document, cage: fieldform_scene::CageId) -> CompileError {
    let why = document
        .cage_body(cage)
        .and_then(|b| b.describe_error())
        .unwrap_or_else(|| "open".to_string());
    CompileError::OpenCage(cage, why)
}

/// The refusal a driven parameter gets when its driver has no op.
///
/// Absent from `slot` for one of two reasons, and they want different words: the ramp has a hole
/// below it and was skipped with everything [`Graph::incomplete`] names, or there is no such block
/// at all. The `Op::Ramp` check each caller makes next stays for a driver that *is* here and is
/// something else.
fn no_driver(
    graph: &Graph,
    incomplete: &std::collections::HashSet<NodeId>,
    driver: NodeId,
) -> CompileError {
    if incomplete.contains(&driver) {
        empty_slot(graph, driver)
    } else {
        CompileError::DrivenParameterNeedsARamp
    }
}

/// A number a node needs, which may vary from point to point.
///
/// The whole of the field-driven-parameter mechanism at the evaluation end. A `Fixed` amount
/// changes nothing: the node that reads it is exactly what it was. A `Driven` one costs the
/// distance contract, and the cost is not optional bookkeeping — `F(p) - d(p)` has gradient
/// `∇F - ∇d`, so its magnitude lies in `[1 - L, 1 + L]` and it is no longer a distance. The
/// consumer divides by `1 + L` and reports [`Normalization::LowerBound`], which is the honest
/// pair: a value that never overstates the room to the surface, declared as such.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Amount {
    Fixed(f64),
    /// The op index of a [`Op::Ramp`].
    Driven(usize),
}

impl Amount {
    /// The op this reads, to be renumbered when a field is copied. A driven amount is an edge
    /// into the op list like any other, and a copy that forgot it would point into the list it
    /// came from.
    fn driver_mut(&mut self) -> Option<&mut usize> {
        match self {
            Amount::Driven(index) => Some(index),
            Amount::Fixed(_) => None,
        }
    }
}

/// A blend's two operand readings and radius as a two-legged chamfer reads them: scaled so the
/// bevel of legs `k` along `a` and `second` along `b` is the equal-legged bevel's zero set at
/// `min(k, second)`. Unchanged where there is no second leg.
pub(crate) fn two_legs(a: f64, b: f64, k: f64, second: Option<f64>) -> (f64, f64, f64) {
    match second {
        Some(s_b) if s_b > 0.0 && k > 0.0 => {
            let m = k.min(s_b);
            (a * m / s_b, b * m / k, m)
        }
        _ => (a, b, k),
    }
}

/// Whether a blend's radius rounds anything: a positive number, or a ramp, which is positive
/// somewhere or there would be no reason to wire it. Read wherever a fixed radius was compared
/// with nought -- a hard join is exact and costs no taper, a rounded one neither.
fn may_round(radius: &Amount) -> bool {
    match radius {
        Amount::Fixed(k) => *k > 0.0,
        Amount::Driven(_) => true,
    }
}

/// Which instance of a pattern a point falls nearest to, without trying them all.
///
/// Only enough to answer that — the placements themselves are stored beside it, so this says how
/// to *index* them rather than repeating what they are.
#[derive(Clone, Debug)]
pub(crate) enum Fold {
    /// Copies marching along a step. The cell is how many whole steps the point is along it,
    /// counted from where the body itself sits — see `origin`.
    Linear { unit: Vector3<f64>, pitch: f64, origin: f64 },
    /// Copies spread evenly around a turn. The cell is the point's angle about the axis,
    /// divided by the angle between neighbours, and measured from the body's own angle.
    Radial { at: Point3<f64>, axis: Rotation3<f64>, count: usize, origin: f64 },
    /// Copies spread over part of a turn, `step` radians apart. Unlike a full turn it does not wrap:
    /// the copies have two ends, and a point past either belongs to the nearer one.
    Arc { at: Point3<f64>, axis: Rotation3<f64>, count: usize, step: f64, origin: f64 },
    /// Copies in rows and columns. `anchor` is where copy 0 stands and `steps` are the two
    /// directions; `dual` reads a point's two coordinates in steps; `reach` is the window along
    /// each. The cell is a pair, so a grid's window is a [`Fold::grid_window`] and not a cell.
    Grid {
        anchor: Point3<f64>,
        steps: [Vector3<f64>; 2],
        dual: [Vector3<f64>; 2],
        counts: [usize; 2],
        reach: [usize; 2],
    },
}

impl Fold {
    /// The instance a point sits nearest the middle of, as an index into the placements.
    ///
    /// **Counted from the body, not from the world origin.** Instance `i` is the body moved by
    /// `i` steps, so the instance covering a point is the number of steps between the *body's*
    /// position and the point's — and subtracting the body's own position is what says so.
    /// Without it the fold assumes the body already sits at step zero, which is true only when
    /// it happens to straddle the axis' zero angle or the step's zero offset. A sphere ringed
    /// about an axis 125 mm away sits twelve cells from that assumption, so a window four cells
    /// wide never contained the instance that owned the point and the field came back empty.
    fn cell(&self, p: Point3<f64>) -> isize {
        match self {
            Fold::Linear { unit, pitch, origin } => {
                if *pitch <= 1e-12 {
                    return 0;
                }
                ((p.coords.dot(unit) - origin) / pitch).round() as isize
            }
            Fold::Radial { at, axis, count, origin } => {
                let local = axis.inverse_transform_vector(&(p - at));
                let angle = local.y.atan2(local.x);
                let step = std::f64::consts::TAU / *count as f64;
                ((angle - origin) / step).round() as isize
            }
            Fold::Arc { at, axis, count, step, origin } => {
                let local = axis.inverse_transform_vector(&(p - at));
                let angle = local.y.atan2(local.x);
                // **The point's offset from the first copy, taken the short way round about the
                // arc's own middle.** An angle has two readings a turn apart, and which one the
                // fold takes decides which end a point behind the arc belongs to. Centred on the
                // middle, the two ends swap exactly at the arc's far side, which is where the
                // nearer one changes — so a clamp of the cell lands on the end that is nearest.
                let middle = (*count as f64 - 1.0) * 0.5 * step;
                let offset = angle - origin;
                let offset = offset - std::f64::consts::TAU * ((offset - middle) / std::f64::consts::TAU).round();
                (offset / step).round() as isize
            }
            // A grid's cell is two numbers; see `grid_window`.
            Fold::Grid { .. } => 0,
        }
    }

    /// The copies of a grid a point has to look at, as indices (column `i mod columns` of row
    /// `i div columns`).
    ///
    /// **Centred on the copy nearest the point among those that exist**, which is not the cell the
    /// point's own coordinates round to once the point is outside the grid. A row is clamped at its
    /// ends and that is exactly right, because distance along a line is monotone. A grid clamped
    /// coordinate by coordinate is right only when its steps are square: at an oblique grid the
    /// nearest copy to a point past one edge is found by sliding along that edge to where the point
    /// projects, not by keeping the other coordinate. So the centre is the minimiser of the distance
    /// over the box of copies — which for a convex quadratic is the unconstrained solution if it is
    /// inside, else the best of the four edges and the four corners — and each direction is then
    /// windowed by its own reach, or walked whole when the window covers it.
    fn grid_window(&self, p: Point3<f64>) -> Vec<isize> {
        let Fold::Grid { anchor, steps, dual, counts, reach } = self else {
            unreachable!("only a grid has a grid window");
        };
        let u = p - anchor;
        let (columns, rows) = (counts[0] as f64 - 1.0, counts[1] as f64 - 1.0);
        let (e1, e2) = (steps[0], steps[1]);
        let mut best = (f64::INFINITY, 0.0f64, 0.0f64);
        let mut offer = |a: f64, b: f64| {
            let (a, b) = (a.clamp(0.0, columns), b.clamp(0.0, rows));
            let cost = (u - e1 * a - e2 * b).norm_squared();
            if cost < best.0 {
                best = (cost, a, b);
            }
        };
        offer(u.dot(&dual[0]), u.dot(&dual[1]));
        for a in [0.0, columns] {
            offer(a, (u - e1 * a).dot(&e2) / e2.norm_squared());
        }
        for b in [0.0, rows] {
            offer((u - e2 * b).dot(&e1) / e1.norm_squared(), b);
        }
        for a in [0.0, columns] {
            for b in [0.0, rows] {
                offer(a, b);
            }
        }
        let along = |centre: f64, n: usize, r: usize| -> Vec<isize> {
            if 2 * r + 1 >= n {
                return (0..n as isize).collect();
            }
            let centre = centre.round() as isize;
            (-(r as isize)..=r as isize)
                .map(|o| centre + o)
                .filter(|i| *i >= 0 && *i < n as isize)
                .collect()
        };
        let columns_in = along(best.1, counts[0], reach[0]);
        let rows_in = along(best.2, counts[1], reach[1]);
        rows_in
            .iter()
            .flat_map(|b| columns_in.iter().map(move |a| b * counts[0] as isize + a))
            .collect()
    }

    /// Whether the instances wrap round, which decides what happens past either end.
    fn wraps(&self) -> bool {
        matches!(self, Fold::Radial { .. })
    }
}

/// One evaluated step of the graph.
#[derive(Clone)]
pub(crate) enum Op {
    /// A field with no inputs: a cage's limit surface, or a pipe around a curve.
    Leaf(std::sync::Arc<dyn ScalarField + Send + Sync>),
    /// A lattice whose phase's number -- wall, density or strut radius -- is a ramp's value at
    /// the point: a graded lattice. `graded-lattice-spec.md` section 4.
    ///
    /// **Not an offset over a lattice**, which is what grading was first built as and what it
    /// still is for a beam. A surface cell's field reads short, so an offset by `d` grows its
    /// wall by `d` times the shortfall, accelerating as the wall leaves the surface: measured, a
    /// gyroid's 2 mm wall under a 0 to 3 mm ramp is 7.4 mm at mid-ramp and a Fischer-Koch S is
    /// solid by then. Driving the phase's own number puts the ramp where the thickness is
    /// converted into an isovalue, so the wall follows it with the fixed lattice's accuracy.
    Lattice {
        lattice: std::sync::Arc<crate::Lattice>,
        /// The op index of the [`Op::Ramp`] that says what the number is here.
        sizing: usize,
        /// See [`driven_lattice_divisor`].
        divisor: f64,
    },
    /// A hard boolean over a list, evaluated by a loop.
    ///
    /// Also the union of a part's bodies: the root fold *is* a union, and it is the reason the
    /// list exists. Nesting one op per body made the recursion below as deep as the part had
    /// bodies, and a STEP assembly is one body per solid — 10 000 of them overflowed the stack
    /// outright, and nothing about that was the user's graph. A loop's depth is whatever the
    /// deepest single operand is, and the count stops entering into it.
    Boolean {
        op: BooleanOp,
        inputs: Vec<usize>,
    },
    Blend {
        op: BooleanOp,
        a: usize,
        b: usize,
        /// The reach, which may follow a ramp: a fillet larger at one end of a seam than the
        /// other is this number varying through space (`round-chamfer-spec.md` §9.1).
        radius: Amount,
        exponent: f64,
        /// How fast the gap to a critical surface can move against how fast the operands can,
        /// or nought in a part with no critical marks. See [`taper_ratio`].
        taper: f64,
        /// What the value is divided by when the radius varies, one otherwise. The zero set of
        /// the blend moves by at most the radius's own change, so the value rises at up to
        /// `1 + s` per millimetre with `s` the ramp's rate, and dividing by that keeps it a
        /// bound -- the same arithmetic as a driven offset's. See [`driven_blend_divisor`].
        divisor: f64,
        /// A chamfer's second leg, where its two legs differ: how far along `b` the bevel
        /// reaches from `a`, `radius` being how far along `a` it reaches from `b`
        /// (`round-chamfer-spec.md` §9.2). Read by scaling the operands so the bevel is the
        /// equal-legged one's, as [`Rim::scaled`] does for a rim. `None` for every other shape.
        second: Option<f64>,
    },
    /// Two fields averaged: `(1 − t)·A + t·B`, with `t` clamped to `[0, 1]`.
    ///
    /// **1-Lipschitz for free at constant `t`**, which is why `divisor` is 1 there and why
    /// dragging `t` costs nothing. Every field composed here already divides by its own bound,
    /// so a convex combination of two of them cannot rise faster than either. A driven `t` adds
    /// `(B − A)·∇t` and needs a real divisor — phase 2, `morph-spec.md` §4.3.
    Morph {
        a: usize,
        b: usize,
        t: Amount,
        divisor: f64,
    },
    /// Another op's boundary moved by a distance. See [`Node::Offset`].
    ///
    /// The distance may itself vary with position, in which case the value is divided by the
    /// bound in `divisor` — see [`Amount`] for why that division is not optional.
    Offset {
        input: usize,
        distance: Amount,
        divisor: f64,
    },
    /// A field read as a number and mapped onto a range. See [`fieldform_scene::Node::Ramp`].
    ///
    /// An op like any other so that its input compiles once and is shared, but it is never a
    /// body: nothing reaches it except an [`Amount::Driven`], because a ramp is left out of the
    /// graph's roots.
    Ramp {
        input: usize,
        in_min: f64,
        /// `in_max - in_min`, never zero — a zero span is refused at compile time, since it
        /// makes the mapping a step and the slope unbounded.
        in_span: f64,
        out_min: f64,
        out_span: f64,
        shape: fieldform_scene::RampShape,
    },
    /// Another op, wound about Z. `radians_per_mm` is the rate; `stretch` is the factor its
    /// value has to be divided by to stay a lower bound. See [`Node::Twist`].
    Twist {
        input: usize,
        radians_per_mm: f64,
        stretch: f64,
        /// 0, 1 or 2 — the axis wound about, and `at` a point it passes through.
        axis: usize,
        at: Point3<f64>,
    },
    /// Another op, drawn toward a curve. See [`Node::Attract`] and the `attract` module.
    ///
    /// The spine is shared rather than copied because a pipe sweeping the same curve compiles
    /// one too, and the two measuring against different flattenings of it would put the pull
    /// and the tube in slightly different places — see [`crate::Spine::of_curve`].
    Attract {
        input: usize,
        spine: std::sync::Arc<crate::Spine>,
        strength: f64,
        falloff: f64,
        /// `1 + strength`, resolved once. Kept beside the strength rather than recomputed per
        /// sample for the same reason a twist keeps its own: the divisor and the warp have to
        /// be the same pair of numbers, and a sample that warped by one and divided by another
        /// is a field that overstates its distance.
        stretch: f64,
    },
    /// Another op, repeated: the union of the same body under each of several rigid motions.
    ///
    /// The motions are stored already inverted, because evaluating a placed field means asking
    /// the unplaced one where the point came from — so inverting once at compile time saves
    /// doing it per instance per sample.
    ///
    /// Rigid, so every instance preserves distance and the union of them is a true distance
    /// field: a pattern stays blendable. That is a property of the *motions* rather than of the
    /// union, which is why it survives however many there are.
    Pattern {
        input: usize,
        /// Every instance's inverse placement, in order. What an instance *is*; the window
        /// below indexes into these rather than replacing them.
        inverse: Vec<Isometry3<f64>>,
        /// How to work out which instance a point belongs to, without trying them all.
        fold: Fold,
        /// How many instances either side of that one can still hold the nearest material.
        ///
        /// **This is what makes a thousand copies cost what three do.** A pattern used to
        /// evaluate its input once per instance, so a hundred copies cost a hundred bodies --
        /// which is a mesh instancing cost wearing a field's clothes, and the thing a field is
        /// supposed to make unnecessary. A lattice has never done it: it folds the query point
        /// into one cell and asks once, which is why a gyroid with more cells than a mesher
        /// could hold traces in a fraction of a second (`gyroid_spike.rs`).
        ///
        /// A pattern's instances are a regular lattice of one body, so the same fold applies.
        /// The cell a point falls in holds the nearest instance *unless* the body reaches out
        /// of its own cell, so the window is that reach measured in cells, plus one for the
        /// half cell the point itself can sit off centre. A body that fits its pitch gives a
        /// window of three, whatever the count is; one that does not is bounded by its own
        /// size rather than by the count, which is the property worth having.
        reach: usize,
        /// For each instance, whether the block leaves it out.
        ///
        /// A copy is left out when the skip list names it, and the copy standing on the body is
        /// left out too when the block does not hold the body (`join` on the node) — see
        /// [`PatternLayout`]. Everything that walks the instances has to agree about this: the
        /// value, the normal and the interval each skip the same ones, or the surface is in one
        /// place and the shading or the bounds are about another.
        excluded: Vec<bool>,
        /// How many of them there are, counted once — the divisor reads it at every sample.
        left_out: usize,
        /// How far back from the join the copies are rounded into each other. Zero is a hard
        /// union, which is what the fold has always been.
        radius: f64,
        /// The exponent the rounding raises its terms to — [`fieldform_scene::Fillet`]'s
        /// continuity, resolved.
        exponent: f64,
        /// As [`Op::Blend`]'s: nought unless the part has critical marks.
        taper: f64,
    },
    /// Another op, reflected in the plane through `at` with unit normal `normal`, and joined
    /// to itself when `join`.
    ///
    /// The cheapest composition here: a reflection is an isometry, so the value comes back
    /// unchanged and only the argument moves. Nothing has to be undone afterwards.
    ///
    /// A general plane rather than an axis index, because both ways of naming one — an axis, or
    /// a `Node::Plane` — resolve to a point and a direction at compile time. Keeping the two
    /// apart any further would be carrying the *question* into the evaluator instead of the
    /// answer.
    Mirror {
        input: usize,
        at: Point3<f64>,
        normal: Vector3<f64>,
        join: bool,
        /// Which side of the plane the body lies wholly on, when it lies wholly on one.
        ///
        /// **A joined mirror asks its body twice, and this is how it asks once.** See
        /// [`mirror_one_side`] for when it is known and why folding the query onto the body's
        /// side gives the same number the `min` would. `None` for a body that straddles its own
        /// mirror plane, or one that cannot be bounded — then both halves are asked, as before.
        one_side: Option<f64>,
        /// Whether the body reaches the plane, so the join keeps a surface buried in the solid.
        ///
        /// **Carried rather than derived, because `normalization_of` cannot derive it.** An
        /// extent is a walk over the *graph* — `extent.rs` takes a document and a node — and the
        /// ops hold no box, so the arm answering for a mirror has nothing to compute it from.
        /// Asked once, of the graph, by the same `mirror_reaches_its_plane` the panel's walk
        /// asks, which is what keeps the compiled answer and the walk's from drifting: the
        /// arrangement `one_side` above already has.
        reaches_its_plane: bool,
    },
    /// Another op, placed. See [`Node::Transform`] for why the scale is one factor and not
    /// three.
    ///
    /// The rotation is kept as a matrix rather than as the node's Euler angles because it is
    /// applied at every sample point, and rebuilding it from three angles per query would
    /// cost more than the leaf underneath it.
    Transform {
        input: usize,
        rotation: Rotation3<f64>,
        translation: Vector3<f64>,
        scale: f64,
    },
}

impl Op {
    /// Every op index this one names, to be renumbered when a field is copied.
    ///
    /// **Beside the evaluator deliberately.** An op kind added here and forgotten there produces
    /// a copy still pointing into the list it came from -- the same mistake `Node::remap_inputs`
    /// is written next to `Node::inputs` to prevent, one level down.
    fn inputs_mut(&mut self) -> Vec<&mut usize> {
        match self {
            Op::Leaf(_) => Vec::new(),
            Op::Lattice { sizing, .. } => vec![sizing],
            Op::Boolean { inputs, .. } => inputs.iter_mut().collect(),
            Op::Blend { a, b, radius, .. } => {
                let mut out = vec![a, b];
                out.extend(radius.driver_mut());
                out
            }
            Op::Morph { a, b, t, .. } => {
                let mut out = vec![a, b];
                out.extend(t.driver_mut());
                out
            }
            Op::Offset { input, distance, .. } => {
                let mut out = vec![input];
                out.extend(distance.driver_mut());
                out
            }
            // `axis` is one of three world axes and `reach` is a count of instances; neither
            // names an op, which is why they are not here.
            Op::Ramp { input, .. }
            | Op::Twist { input, .. }
            | Op::Attract { input, .. }
            | Op::Pattern { input, .. }
            | Op::Mirror { input, .. }
            | Op::Transform { input, .. } => vec![input],
        }
    }
}

/// What a region of space holds, once a field has been asked about it.
///
/// **The rejection is the point, not a special case.** Most of a part's bounds is empty, and one
/// interval evaluation says so — `brick_timing` measures 82% to 95% of bricks answered this way,
/// each for the cost of a single bound. A caller that meshes, bakes or marches can then skip the
/// region entirely rather than sampling it to discover the same thing a thousand times.
pub enum Region {
    /// No surface here: the field is positive throughout, so the region is outside everything.
    Empty,
    /// No surface here either, and for the opposite reason: the region is wholly inside.
    Solid,
    /// A surface crosses this region, and here is the field shortened to what can matter in it.
    Field(GraphField),
}

impl Region {
    /// The shortened field, when there is a surface to evaluate.
    pub fn field(&self) -> Option<&GraphField> {
        match self {
            Region::Field(field) => Some(field),
            _ => None,
        }
    }
}

/// A compiled graph, evaluable as a single field.
pub struct GraphField {
    ops: Vec<Op>,
    output: usize,
    /// Each critical surface's own field, and its clearance. See [`Self::gap_at`].
    ///
    /// Compiled from the marked node's subtree *alone*, which carries no marks — so a critical
    /// surface that is itself a blended body is measured at its full radii, and the gap does
    /// not depend on itself.
    protect: Vec<(std::sync::Arc<GraphField>, f64)>,
    /// Where each body landed, so a point on the surface can be traced back to the body it
    /// belongs to. Kept in the compiled field rather than recomputed because the mapping from
    /// node to op only exists here.
    bodies: Vec<(fieldform_scene::NodeId, usize)>,
    /// What each driven morph was actually divided by, keyed by the node.
    ///
    /// **Published so the shader can divide by the same number.** The bound is read over the two
    /// operands' intervals, which exist only once the op list is built — so the CPU can compute
    /// it and `generate` cannot. A number the two paths work out differently is one surface in
    /// two places, so the shader is handed this rather than left to derive its own.
    morph_divisors: Vec<(fieldform_scene::NodeId, f64)>,
}

impl GraphField {
    /// Compiles a part's graph against the document its nodes reference.
    ///
    /// `tolerance_mm` is how far any surface in the result may sit from the true one. It sets
    /// the curve tessellation for pipes and the subdivision level for cages, so a single
    /// number governs the accuracy of the whole composition.
    pub fn compile(
        document: &Document,
        part: &Part,
        tolerance_mm: f64,
    ) -> Result<Self, CompileError> {
        Self::compile_graph(document, &part.graph, tolerance_mm)
    }

    /// The same, reusing leaves built by an earlier compile. See [`LeafCache`].
    pub fn compile_cached(
        document: &Document,
        graph: &Graph,
        tolerance_mm: f64,
        cache: &mut LeafCache,
    ) -> Result<Self, CompileError> {
        cache.retain_live(document);
        Self::build(document, graph, tolerance_mm, Some(cache))
    }

    /// Compiles a graph directly, for callers holding one outside a part.
    pub fn compile_graph(
        document: &Document,
        graph: &Graph,
        tolerance_mm: f64,
    ) -> Result<Self, CompileError> {
        Self::build(document, graph, tolerance_mm, None)
    }

    fn build(
        document: &Document,
        graph: &Graph,
        tolerance_mm: f64,
        mut cache: Option<&mut LeafCache>,
    ) -> Result<Self, CompileError> {
        // **Bodies that meet face to face by construction are built as one** (backlog item 113),
        // read first and from the same function the shader reads, so the two draw one body. A
        // joined mirror of a section touching its plane used to be a union that kept the shared
        // face buried in the solid — reading nought 7.5 mm deep — and every blend grooved along
        // it. Nothing to do, and nothing cloned, for a document without such a mirror.
        if let Some((merged, merged_graph)) = crate::with_shared_faces_merged(document, graph) {
            return Self::build(&merged, &merged_graph, tolerance_mm, cache);
        }
        // Per-node numbers the shader has to agree with this field on and cannot work out for
        // itself: a rimmed revolve's square-corner scale, read from its section's distance from
        // the axis (`Rim::cut`). Handed over beside the driven morphs' divisors.
        let mut rim_scales: Vec<(fieldform_scene::NodeId, f64)> = Vec::new();
        // Checked across the whole graph rather than only what the output reaches, because
        // "unreachable" is itself a reading of a node this build could not read: an unknown
        // node's operands are the part that could not be parsed. Refusing the document is
        // the honest answer, and the file is preserved either way.
        if let Some((_, node)) = graph.nodes().find(|(_, n)| matches!(n, Node::Unknown(_))) {
            let Node::Unknown(unknown) = node else {
                unreachable!()
            };
            return Err(CompileError::UnreadableNode(unknown.to_string()));
        }

        // Which parameters may vary through space, checked in one place rather than as a
        // `None` falling out of each node's own resolver — those would report "unresolved
        // radius", which points at a typo rather than at the real answer, that this input
        // takes a number.
        //
        // The list is short on purpose. The decision this implements says to build the
        // mechanism against **one** consumer and let the rest refuse, because each further one
        // is its own question about what varying costs it: a driven blend radius, a driven
        // twist rate and a driven lattice thickness each need their own bound, and shipping
        // them all at once would mean four unverified ones instead of one measured.
        if let Some(refusal) = graph.nodes().find_map(|(_, n)| driven_refusal(n)) {
            return Err(refusal);
        }

        // A block that carries its own turn becomes that block inside a placement, which is
        // what the turn means. Done here rather than arm by arm so the field and the shader
        // read one description of it -- see `Graph::with_turns_expanded`.
        //
        // Through `with_turns_flattened` rather than the graph's own method, because a turn can
        // be aimed at a plane or a line and working out where those point needs the document.
        // The numbers are read the same way `resolve` below reads them; it cannot be used here
        // because it is built against the expanded graph this produces.
        let numbers = |r: &fieldform_scene::ParamRef| -> Option<f64> { document.resolve(r) };
        let turned = with_turns_flattened(document, graph, &numbers);
        let graph = &turned;
        // In the document's order, like the shader's: the two compile the same part and an op
        // list numbered differently from the generated source is two descriptions of it.
        let order = document.evaluation_order_of(graph).ok_or(CompileError::Cyclic)?;

        // Graph node ids map to positions in `ops` by where they fall in evaluation order, so
        // every operand is already built by the time something references it.
        let mut slot = Slots::default();
        // Where each driven morph landed, so the second pass finds them without matching ops
        // back to nodes a second time.
        let mut driven_morphs: Vec<(NodeId, usize)> = Vec::new();
        let mut ops = Vec::with_capacity(order.len());

        // An expression that failed -- a division by zero, a reference to another failure --
        // is `None` here exactly as an unknown name is, and surfaces as UnresolvedRadius
        // rather than reaching the field as NaN.
        let resolve = |r: &fieldform_scene::ParamRef| -> Option<f64> { document.resolve(r) };

        // Asked once, over the expanded graph the ops are built from. It is a walk of every
        // node's edges, so inside the loop it would be that walk once per node.
        //
        // Of the document rather than the graph: a cage block whose cage is open is in it too
        // (R12), with everything reading it, since a red cage has no field exactly as an empty
        // slot has no value.
        let incomplete = document.incomplete_in(graph);

        for id in &order {
            let node = graph
                .node(*id)
                .expect("evaluation_order only yields real nodes");
            // **A block with an empty slot compiles to nothing, and so does everything above
            // it.** An empty slot is a legal state -- a block placed from the Create menu before
            // it has been told what to read -- and there is no value a missing input could supply
            // that keeps `distance` from overestimating, so nothing is invented for one here.
            //
            // `incomplete` is closed downstream, which is what makes one `continue` enough: a
            // node this loop does reach has every operand in `slot`, so the index sites below
            // need no guard of their own and whoever adds the next kind of block has nothing to
            // remember. A *body* that needs one of these is refused by name at the fold below,
            // and a ramp nothing reads just compiles to nothing, as it always has.
            if incomplete.contains(id) {
                continue;
            }
            let op = match node {
                Node::Cage(cage_id) => {
                    // A red cage never reaches here -- `incomplete` skipped it above -- so
                    // `None` is a cage the document does not hold at all.
                    let cage = document
                        .cage(*cage_id)
                        .ok_or(CompileError::MissingCage(*cage_id))?;
                    let levels = levels_for_tolerance(mesh_size(cage), tolerance_mm);
                    let key = (*cage_id, levels, cage.geometry_fingerprint());
                    let existing = cache
                        .as_ref()
                        .and_then(|c| c.cages.get(&key))
                        .map(std::sync::Arc::clone);
                    let leaf = match existing {
                        Some(leaf) => leaf,
                        None => {
                            let built: std::sync::Arc<dyn ScalarField + Send + Sync> =
                                std::sync::Arc::new(CageField::from_cage(cage, levels)?);
                            if let Some(c) = cache.as_mut() {
                                c.keep_cage(key, std::sync::Arc::clone(&built));
                            }
                            built
                        }
                    };
                    Op::Leaf(leaf)
                }
                Node::Mesh(id) => {
                    let mesh = document
                        .meshes
                        .get(id)
                        .ok_or(CompileError::MissingMesh(*id))?;
                    // No tolerance argument: an imported mesh is already triangles, and there
                    // is nothing left to subdivide more finely. Whatever accuracy the source
                    // was tessellated at is what the document has, which is exactly why
                    // `MeshData` records it rather than letting it be guessed at later.
                    let existing = cache
                        .as_ref()
                        .and_then(|c| c.meshes.get(id))
                        .map(std::sync::Arc::clone);
                    let leaf = match existing {
                        Some(leaf) => leaf,
                        None => {
                            let positions = mesh
                                .positions
                                .iter()
                                .map(|p| Point3::new(p[0], p[1], p[2]))
                                .collect();
                            let built: std::sync::Arc<dyn ScalarField + Send + Sync> =
                                std::sync::Arc::new(
                                    crate::MeshField::from_mesh(positions, mesh.triangles.clone())
                                        .map_err(|e| CompileError::BadMesh(*id, e))?,
                                );
                            if let Some(c) = cache.as_mut() {
                                c.meshes.insert(*id, std::sync::Arc::clone(&built));
                            }
                            built
                        }
                    };
                    Op::Leaf(leaf)
                }
                // Closed-form leaves: no cage to subdivide, no curve to tessellate, and
                // exact at any tolerance — the argument the rest of this function threads
                // around simply does not apply to them.
                Node::Sphere { centre, radius } => Op::Leaf(std::sync::Arc::new(crate::Sphere {
                    centre: resolve_place_for(document, graph, *id, centre, &resolve)?,
                    radius: resolve(radius).ok_or(CompileError::UnresolvedRadius)?,
                })),
                Node::Cuboid { centre, size, edge, .. } => {
                    let cuboid = crate::Cuboid {
                        centre: resolve_place_for(document, graph, *id, centre, &resolve)?,
                        size: {
                            let mut sides = [0.0; 3];
                            for (slot, side) in sides.iter_mut().zip(size) {
                                *slot = resolve(side).ok_or(CompileError::UnresolvedRadius)?;
                            }
                            Vector3::from(sides)
                        },
                    };
                    // **A hard edge is the box it always was**, leaf and all; a rounded or
                    // chamfered one is held to the smallest half-side, past which the rounding
                    // of opposite edges would meet.
                    match resolved_edge(edge, &resolve)? {
                        None => Op::Leaf(std::sync::Arc::new(cuboid)),
                        Some(cut) => {
                            let half = cuboid.size.abs().min() * 0.5;
                            check_edge(cut.reach, half, "box's smallest half-side", "box")?;
                            Op::Leaf(std::sync::Arc::new(crate::EdgedCuboid { cuboid, edge: cut }))
                        }
                    }
                }
                Node::SkewedBox { centre, edges } => {
                    let mut vectors = [Vector3::zeros(); 3];
                    for (vector, edge) in vectors.iter_mut().zip(edges) {
                        for (slot, component) in vector.iter_mut().zip(edge) {
                            *slot = resolve(component).ok_or(CompileError::UnresolvedRadius)?;
                        }
                    }
                    let centre = resolve_place_for(document, graph, *id, centre, &resolve)?;
                    Op::Leaf(std::sync::Arc::new(
                        crate::SkewedBox::new(centre, vectors).ok_or(CompileError::FlatSkewedBox(*id))?,
                    ))
                }
                Node::Cylinder { ends, radius, edge } => {
                    let (base, top) = resolve_ends(ends, document, graph, &resolve)?;
                    let cylinder = crate::Cylinder {
                        base,
                        top,
                        radius: resolve(radius).ok_or(CompileError::UnresolvedRadius)?,
                    };
                    // Held to the radius and to half the length, the cylinder's two half-sides.
                    match resolved_edge(edge, &resolve)? {
                        None => Op::Leaf(std::sync::Arc::new(cylinder)),
                        Some(cut) => {
                            check_edge(cut.reach, cylinder.radius.abs(), "cylinder's radius", "cylinder")?;
                            let half = (top - base).norm() * 0.5;
                            check_edge(cut.reach, half, "cylinder's half-length", "cylinder")?;
                            Op::Leaf(std::sync::Arc::new(crate::EdgedCylinder { cylinder, edge: cut }))
                        }
                    }
                }
                Node::Cone { ends, base_radius, top_radius, edge } => {
                    let (base, top) = resolve_ends(ends, document, graph, &resolve)?;
                    let cone = crate::Cone {
                        base,
                        top,
                        base_radius: resolve(base_radius).ok_or(CompileError::UnresolvedRadius)?,
                        top_radius: resolve(top_radius).ok_or(CompileError::UnresolvedRadius)?,
                    };
                    // Held to what its section has room for: the inradius for a round, the
                    // shortest edge for a chamfer (`EdgedCone::room`). A hard edge is the cone it
                    // always was, and so is a cone with no length, whose edges are nowhere.
                    match resolved_edge(edge, &resolve)? {
                        None => Op::Leaf(std::sync::Arc::new(cone)),
                        Some(cut) => {
                            let (room, what) = crate::EdgedCone::room(&cone, cut.exponent);
                            check_edge(cut.reach, room, what, "cone")?;
                            match crate::EdgedCone::new(cone, cut) {
                                Some(edged) => Op::Leaf(std::sync::Arc::new(edged)),
                                None => Op::Leaf(std::sync::Arc::new(cone)),
                            }
                        }
                    }
                }
                Node::Torus {
                    centre,
                    major_radius,
                    minor_radius,
                    ..
                } => Op::Leaf(std::sync::Arc::new(crate::Torus {
                    centre: resolve_place_for(document, graph, *id, centre, &resolve)?,
                    major_radius: resolve(major_radius).ok_or(CompileError::UnresolvedRadius)?,
                    minor_radius: resolve(minor_radius).ok_or(CompileError::UnresolvedRadius)?,
                })),
                Node::Lattice {
                    frame,
                    cell,
                    map,
                    phase,
                } => {
                    let mut map = match map {
                        fieldform_scene::CellMap::Rectangular { cell } => {
                            let size = resolve(cell).ok_or(CompileError::UnresolvedRadius)?;
                            crate::LatticeMap::Rectangular {
                                cell: Vector3::new(size, size, size),
                            }
                        }
                        fieldform_scene::CellMap::Spherical {
                            cell_radius,
                            azimuth_count,
                            polar_count,
                        } => {
                            if *azimuth_count == 0 || *polar_count == 0 {
                                return Err(CompileError::EmptyArcCount);
                            }
                            crate::LatticeMap::Spherical {
                                cell_radius: resolve(cell_radius)
                                    .ok_or(CompileError::UnresolvedRadius)?,
                                azimuth_count: *azimuth_count,
                                polar_count: *polar_count,
                            }
                        }
                        // **The surface is built here and cached here**, beside the cage
                        // it costs the same as. A tessellation and a BVH over it are a pure
                        // function of a mesh that does not move, and every show/hide and menu
                        // action recompiles the graph -- which is the measurement that put a
                        // cache in front of cages in the first place.
                        fieldform_scene::CellMap::Conformal {
                            surface,
                            cells_per_face,
                            depth,
                            thickness,
                        } => {
                            // Refused rather than clamped, exactly as an arc count is: no cells
                            // across a face describes no tiling, and choosing one for the user
                            // would draw a lattice nobody asked for.
                            if *cells_per_face == 0 {
                                return Err(CompileError::EmptyCellCount);
                            }
                            let mesh = document
                                .quad_meshes
                                .get(surface)
                                .ok_or(CompileError::MissingQuadMesh(*surface))?;
                            let levels = surface_levels(*cells_per_face);
                            let key = (*surface, levels, mesh.geometry_fingerprint());
                            let existing = cache
                                .as_ref()
                                .and_then(|c| c.surfaces.get(&key))
                                .map(std::sync::Arc::clone);
                            let built = match existing {
                                Some(map) => map,
                                None => {
                                    let map = std::sync::Arc::new(
                                        crate::SurfaceMap::new(mesh, levels).map_err(|e| {
                                            CompileError::BadQuadMesh(*surface, e)
                                        })?,
                                    );
                                    if let Some(c) = cache.as_mut() {
                                        c.keep_surface(key, std::sync::Arc::clone(&map));
                                    }
                                    map
                                }
                            };
                            crate::LatticeMap::Conformal {
                                surface: built,
                                cells_per_face: *cells_per_face,
                                depth: resolve(depth).ok_or(CompileError::UnresolvedRadius)?,
                                // Halved once, here, rather than in the innermost loop of the
                                // field and again in the shader.
                                reach: match thickness {
                                    Some(t) => Some(
                                        resolve(t).ok_or(CompileError::UnresolvedRadius)? * 0.5,
                                    ),
                                    None => None,
                                },
                            }
                        }
                        fieldform_scene::CellMap::Cylindrical {
                            cell_radius,
                            cell_height,
                            arc_count,
                        } => {
                            // Refused rather than clamped. A count of zero describes no tiling
                            // at all, and picking one on the user's behalf would draw a lattice
                            // nobody asked for at a spacing nobody chose.
                            if *arc_count == 0 {
                                return Err(CompileError::EmptyArcCount);
                            }
                            crate::LatticeMap::Cylindrical {
                                cell_radius: resolve(cell_radius)
                                    .ok_or(CompileError::UnresolvedRadius)?,
                                cell_height: resolve(cell_height)
                                    .ok_or(CompileError::UnresolvedRadius)?,
                                arc_count: *arc_count,
                            }
                        }
                    };
                    // The two arms carry different quantities and neither converts into the
                    // other, which is why the phase travels as a pair rather than as a flag
                    // beside one number. A density stays a density all the way down: the
                    // isovalue it means is per cell and is looked up where the field is
                    // evaluated, so the CPU and the shader read it from the same table.
                    // **A driven phase.** Only the phase's own number may follow a ramp --
                    // `driven_refusal` has already said no to anything else on the block -- and
                    // only a ramp may drive it, because only a ramp states how fast it changes.
                    let driven = match phase.param().driver() {
                        None => None,
                        Some(driver) => {
                            // The ramp is read at the point the lattice is asked at, and an
                            // aimed lattice is asked in its own frame. Refused by name rather
                            // than graded along the wrong axis.
                            if !frame.is_default() {
                                return Err(CompileError::GradedLatticeCannotBeAimed);
                            }
                            let at = *slot
                                .get(&driver)
                                .ok_or_else(|| no_driver(graph, &incomplete, driver))?;
                            let Op::Ramp { out_min, out_span, .. } = ops[at] else {
                                return Err(CompileError::DrivenParameterNeedsARamp);
                            };
                            Some((at, out_min, out_min + out_span))
                        }
                    };
                    // The number the block's own lattice carries. Typed, or -- when a ramp
                    // drives it -- the larger end of the ramp's range, floored at a tenth of
                    // it: that lattice is only asked for an extent and a feature size, and the
                    // thin end of a grade that starts at nothing has no feature to resolve.
                    let sizing = |typed: &fieldform_scene::ParamRef| match driven {
                        Some((_, a, b)) => Some(a.min(b).max(0.1 * a.max(b))),
                        None => resolve(typed),
                    };
                    // Through the one conversion the property readout also reads, so what a
                    // block is said to be and what it is built as cannot drift apart.
                    let phase = crate::phase_of(phase, sizing, resolve)
                        .map_err(|_| CompileError::UnresolvedRadius)?;
                    let kind: crate::CellKind = (*cell).into();
                    // **A prism's period is not a cube, and the map takes its shape from the
                    // cell** -- backlog item 49's contract change. The size typed is the
                    // hexagon's flat-to-flat width; the period along the other in-plane axis is
                    // whatever that cell says it is. Any other map has no flat period to give,
                    // and says so rather than drawing stretched hexagons.
                    if let crate::CellKind::Prism(prism) = kind {
                        match &mut map {
                            crate::LatticeMap::Rectangular { cell } => {
                                *cell = cell.component_mul(&prism.natural_aspect());
                            }
                            other => {
                                return Err(CompileError::CellCannotTakeThatMap(
                                    kind.label(),
                                    match other {
                                        crate::LatticeMap::Cylindrical { .. } => "cylindrical",
                                        crate::LatticeMap::Spherical { .. } => "spherical",
                                        _ => "conformal",
                                    },
                                ));
                            }
                        }
                    }
                    // **Refused rather than drawn.** The panel only ever offers a cell the
                    // bandings it supports, so this pairing cannot be made by clicking — but a
                    // document can carry it, by hand or from a build whose vocabulary differed,
                    // and every arm downstream would then be reading a number that means
                    // something else. Saying so is the only honest answer; guessing a phase would
                    // draw a plausible lattice of the wrong kind.
                    if !kind.supported_bandings().contains(&phase.banding()) {
                        return Err(CompileError::CellCannotTakeThatPhase(
                            kind.label(),
                            match phase.banding() {
                                fieldform_scene::Banding::Sheet => "sheet",
                                fieldform_scene::Banding::Solid => "solid",
                                fieldform_scene::Banding::Strut => "strut",
                            },
                        ));
                    }
                    let pattern = crate::Lattice { cell: kind, map, phase };
                    // **The cell's own numbers, asked of the document.** A jitter is a
                    // `ParamRef` like any other, so it is resolved here rather than read off the
                    // parameter -- and `with_scatter` clamps it, because a seed leaving its own
                    // cell breaks every bound the measure rests on.
                    let pattern = match (cell.jitter(), cell.seed()) {
                        (Some(jitter), Some(seed)) => {
                            // A conformal map hands each face its own `s` and `t`, so the
                            // integer cell a seed is hashed from means something different on
                            // either side of a face edge and the neighbour search stops there.
                            // Refused by name; the other three maps are continuous in their
                            // cell coordinates and wrap through `LatticeMap::cell_period`.
                            if matches!(pattern.map, crate::LatticeMap::Conformal { .. }) {
                                return Err(CompileError::CellCannotTakeAConformalMap(
                                    kind.label(),
                                ));
                            }
                            pattern.with_scatter(
                                resolve(jitter).ok_or(CompileError::UnresolvedRadius)?,
                                crate::voronoi_seed(seed),
                            )
                        }
                        _ => pattern,
                    };
                    let pattern = std::sync::Arc::new(pattern);
                    // Aimed by wrapping, not by teaching the lattice about frames.
                    //
                    // `to_cell_coords` takes a world point and reads a pattern laid out about
                    // the world's own axes, so putting the point into the lattice's frame first
                    // is the whole of it — and a rigid motion is exactly what `Op::Transform`
                    // already applies, with its normal and its interval bound already worked
                    // out. A frame inside `Lattice` would be a second implementation of a
                    // rotation this crate already has, and the two could disagree.
                    if let Some((at, _, _)) = driven {
                        Op::Lattice {
                            lattice: pattern,
                            sizing: at,
                            divisor: driven_lattice_divisor(document, graph, *id),
                        }
                    } else if frame.is_default() {
                        Op::Leaf(pattern)
                    } else {
                        let leaf = ops.len();
                        ops.push(Op::Leaf(pattern));
                        let (translation, rotation) = resolve_frame(document, graph, frame, &resolve)?;
                        Op::Transform {
                            input: leaf,
                            rotation,
                            translation,
                            scale: 1.0,
                        }
                    }
                }
                Node::Plane { origin, flipped, .. } => Op::Leaf(std::sync::Arc::new(crate::Plane {
                    origin: resolve_place_for(document, graph, *id, origin, &resolve)?,
                    flipped: *flipped,
                })),
                Node::Point { at } => Op::Leaf(std::sync::Arc::new(crate::Point {
                    at: resolve_point(at, &resolve)?,
                })),
                Node::Capsule {
                    ends,
                    start_radius,
                    end_radius,
                } => {
                    let (start, end) = resolve_ends(ends, document, graph, &resolve)?;
                    Op::Leaf(std::sync::Arc::new(crate::Capsule {
                        start,
                        end,
                        start_radius: resolve(start_radius)
                            .ok_or(CompileError::UnresolvedRadius)?,
                        end_radius: resolve(end_radius).ok_or(CompileError::UnresolvedRadius)?,
                    }))
                }
                Node::Pipe { curve, radius } => {
                    let c = document
                        .curves
                        .get(curve)
                        .ok_or(CompileError::MissingCurve(*curve))?;
                    match radius.driver() {
                        None => {
                            let r = resolve(radius).ok_or(CompileError::UnresolvedRadius)?;
                            Op::Leaf(std::sync::Arc::new(CurveField::new(c, r, tolerance_mm)))
                        }
                        // **Built as the offset it is.** A pipe of a varying radius is the
                        // distance to its own spine, offset by that radius — so the spine goes
                        // in at radius nought and `Op::Offset` does the rest, reusing the
                        // arithmetic, the divisor and the tests an offset already has rather
                        // than growing a second op that would have to agree with them.
                        Some(driver) => {
                            let at = *slot
                                .get(&driver)
                                .ok_or_else(|| no_driver(graph, &incomplete, driver))?;
                            if !matches!(ops[at], Op::Ramp { .. }) {
                                return Err(CompileError::DrivenParameterNeedsARamp);
                            }
                            let spine = ops.len();
                            ops.push(Op::Leaf(std::sync::Arc::new(CurveField::new(
                                c,
                                0.0,
                                tolerance_mm,
                            ))));
                            Op::Offset {
                                input: spine,
                                distance: Amount::Driven(at),
                                divisor: driven_pipe_divisor(document, graph, *id),
                            }
                        }
                    }
                }
                // Read and built by the one helper the extent and the divisor ask too.
                // **Built at the default tolerance, whatever the compile's own** -- as every
                // other reader of a loft builds it. The shader's stations are computed by
                // codegen, which has no tolerance to hand, and its spine, frame and sections
                // come from the same build; a loft flattened two ways is a loft in two places,
                // and the station a section sits at moves with the spine's flattening. The
                // sections themselves are walked by angle, not by tolerance, so nothing here is
                // coarser than the compile asked for.
                Node::Loft { sections, spine, smooth, rims } => {
                    let parts = loft_parts(
                        document,
                        sections,
                        *spine,
                        crate::profile::PROFILE_TURN_DEGREES,
                        crate::DEFAULT_TOLERANCE_MM,
                    )?;
                    let _ = tolerance_mm;
                    let built = build_loft(&parts, *smooth)?;
                    let rims = resolved_rim(rims, &resolve)?;
                    if rims.is_hard() {
                        Op::Leaf(built)
                    } else {
                        // Held as the extrude's are: half the run between the first section and
                        // the last, or all of it for one end, and the shallower end section's
                        // depth -- of both, which is the stricter answer for one end.
                        let stations = built.stations();
                        let run = stations.last().copied().unwrap_or(0.0) - stations[0];
                        check_rim(rims.reach, rims.room_along(run), "loft's length", "longer")?;
                        check_rim(rim_second_leg(&rims), built.end_depth(), "end sections' depth", "fuller at its ends")?;
                        Op::Leaf(std::sync::Arc::new(built.with_rims(rims)))
                    }
                }
                Node::Sweep { outline, spine, section, law, rims } => {
                    // The section is read exactly as an extrude reads its own -- the same walk,
                    // the same plane, the same bake -- with the spine left out of the walk,
                    // since a path that happens to touch the section is not part of it. From
                    // `sweep_outline_section`, which the extent and the shader ask too.
                    let laid = sweep_outline_section(
                        document,
                        outline,
                        *spine,
                        crate::profile::PROFILE_TURN_DEGREES,
                    )?;
                    let normal = laid.normal;
                    let rotation = rotation_from_z(normal);
                    let translation = laid.origin.coords;
                    let into = move |p: Point3<f64>| {
                        Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
                    };
                    let built: crate::Footprint = match outline {
                        fieldform_scene::Outline::Closed { profile } => {
                            let built =
                                crate::Profile::in_plane(&laid.points, into, tolerance_mm.max(1e-6))
                                    .map_err(|e| CompileError::BadProfile(*profile, e))?;
                            bake_to_tolerance(
                                built,
                                tolerance_mm,
                                profile_reads_out_to(document, graph, *id),
                                cache.as_deref_mut(),
                            )
                            .into()
                        }
                        // An open section, which is what a sweep could not take: a curve given
                        // a thickness, carried the way a profile is, and baked by its rule --
                        // unless a law moves it, for the reason `wall_bakes_under` gives.
                        fieldform_scene::Outline::Wall { curve, thickness, side, ends } => {
                            let thickness =
                                resolve(thickness).ok_or(CompileError::UnresolvedRadius)?;
                            let wall = wall_in_plane(
                                &laid,
                                *curve,
                                thickness,
                                *side,
                                *ends,
                                tolerance_mm.max(1e-6),
                            )?;
                            if wall_bakes_under(law) {
                                bake_to_tolerance(
                                    wall,
                                    tolerance_mm,
                                    profile_reads_out_to(document, graph, *id),
                                    cache.as_deref_mut(),
                                )
                                .into()
                            } else {
                                wall.into()
                            }
                        }
                    };
                    let path = document
                        .curves
                        .get(spine)
                        .ok_or(CompileError::MissingCurve(*spine))?;
                    // The one flattening a curve gets (`Spine::of_curve`), so this and a pipe
                    // along the same curve put its surface in the same place.
                    let path = crate::Spine::of_curve(path, tolerance_mm);
                    // The profile's own +u axis seeds the frame, so the section starts as drawn
                    // and is carried from there. The frame is computed from the spine and never
                    // stored -- there is no second description of it to disagree with the curve.
                    // The same seed `sweep_section` reads for the shader's copy of the frame.
                    let seed = rotation * Vector3::x();
                    // How the section is held, in the kernel's own vocabulary: the profile's
                    // basis is what both held sections are measured from.
                    let held = match section {
                        fieldform_scene::Section::Square => crate::Held::Square,
                        fieldform_scene::Section::Skewed => crate::Held::Flush {
                            normal,
                            u: rotation * Vector3::x(),
                            v: rotation * Vector3::y(),
                        },
                        fieldform_scene::Section::Parallel => crate::Held::Parallel {
                            normal,
                            u: rotation * Vector3::x(),
                            v: rotation * Vector3::y(),
                        },
                    };
                    // How the section changes on the way, with its numbers asked of the
                    // document; the kernel refuses what cannot be built, by name.
                    let law = sweep_law(document, law)?;
                    let section_depth = built.depth();
                    let swept = crate::Swept::under_law(built, path, seed, held, law)
                        .map_err(|e| CompileError::BadSweep(*spine, e))?;
                    // The rims, held as an extrude's are -- and refused on a closed path, which
                    // has no ends for a rim to round.
                    let rims = resolved_rim(rims, &resolve)?;
                    let swept = if rims.is_hard() {
                        swept
                    } else {
                        if document.curves.get(spine).is_some_and(|c| c.is_closed()) {
                            return Err(CompileError::RimsOnAClosedPath);
                        }
                        check_rim(rims.reach, rims.room_along(swept.run()), "sweep's run", "longer")?;
                        check_rim(rim_second_leg(&rims), section_depth, "section's depth", "thicker")?;
                        swept.with_rims(rims)
                    };
                    // Carried to the profile. See [`sweep_offset`]: the spine gives the shape of
                    // the path and the profile gives where it starts, so a curve drawn off to
                    // one side sweeps the profile where the profile is.
                    let offset =
                        sweep_offset(document, outline, *spine).unwrap_or_else(Vector3::zeros);
                    if offset.norm() <= 1e-12 {
                        Op::Leaf(std::sync::Arc::new(swept))
                    } else {
                        let inner = ops.len();
                        ops.push(Op::Leaf(std::sync::Arc::new(swept)));
                        Op::Transform {
                            input: inner,
                            rotation: Rotation3::identity(),
                            translation: offset,
                            scale: 1.0,
                        }
                    }
                }
                Node::Extrude { outline, sweep, start, height, sides, draft, corners, rims } => {
                    // Which curves, and in which plane. For a profile that is the loop its curve
                    // is part of -- the curve itself when it already closes and several joined
                    // end to end when it does not -- and for a wall it is the chain, which need
                    // not close. Walked rather than stored; see `profile_loop`. A curve giving
                    // the sweep its direction is not part of what is being swept.
                    //
                    // **The plane is the section's own, computed and never stored.** It used to
                    // be a `Frame` on the node, which meant the plane a profile was read in and
                    // the plane it actually lay in were two descriptions that could disagree --
                    // and every way of editing the frame was a way of making them.
                    //
                    // Both from `extrude_section`, which the extent and the shader ask too, so
                    // that the three cannot lay the same section in different places.
                    let section = extrude_section(
                        document,
                        outline,
                        sweep,
                        crate::profile::PROFILE_TURN_DEGREES,
                    )?;
                    let h = resolve(height).ok_or(CompileError::UnresolvedRadius)?;
                    let rotation = rotation_from_z(section.normal);
                    let translation = section.origin.coords;
                    let into = move |p: Point3<f64>| {
                        Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
                    };
                    let built: crate::Footprint = match outline {
                        fieldform_scene::Outline::Closed { profile } => {
                            // Flat to within the tolerance the whole compile is working at,
                            // rather than a number of its own: a profile is as planar as
                            // anything else here is accurate, and a second tolerance would be a
                            // second thing to tune.
                            let built = crate::Profile::in_plane(
                                &section.points,
                                into,
                                tolerance_mm.max(1e-6),
                            )
                            .map_err(|e| CompileError::BadProfile(*profile, e))?;
                            // Answered from a grid when the tolerance allows one, which is the
                            // whole of what makes a profile cheap to ask -- see
                            // `Profile::baked`. The shader bakes from the same rule so the two
                            // still describe one surface.
                            bake_to_tolerance(
                                built,
                                tolerance_mm,
                                profile_reads_out_to(document, graph, *id),
                                cache.as_deref_mut(),
                            )
                            .into()
                        }
                        // **A wall is answered from a grid by the rule a profile is**, and the
                        // grid is of the wall's own distance: the curve's would have to be folded
                        // about the band's middle after its margin was taken off, which turns
                        // short into long -- see `Wall::baked`. Ruled 2026-10-02, on what
                        // walking a long chain at every reading was measured to cost.
                        fieldform_scene::Outline::Wall { curve, thickness, side, ends } => {
                            let thickness =
                                resolve(thickness).ok_or(CompileError::UnresolvedRadius)?;
                            let wall = wall_in_plane(
                                &section,
                                *curve,
                                thickness,
                                *side,
                                *ends,
                                tolerance_mm.max(1e-6),
                            )?;
                            bake_to_tolerance(
                                wall,
                                tolerance_mm,
                                profile_reads_out_to(document, graph, *id),
                                cache.as_deref_mut(),
                            )
                            .into()
                        }
                    };

                    // Which way it is swept, in the profile's own frame. `Normal` is `+Z` there
                    // by construction, which is what makes the exact case exact rather than
                    // nearly so.
                    let leaning = match sweep {
                        fieldform_scene::Sweep::Normal => None,
                        other => {
                            let world_dir = sweep_direction(document, other, &resolve)
                                .ok_or(CompileError::SweepHasNoDirection)?;
                            let local = rotation.inverse_transform_vector(&world_dir);
                            let local = local
                                .try_normalize(1e-12)
                                .ok_or(CompileError::SweepHasNoDirection)?;
                            // Edge-on to the profile there is no solid at all, and the field
                            // would be divided by nothing. Refused by name rather than produced
                            // as an empty shape somebody has to work out the reason for.
                            if local.z.abs() < 1e-6 {
                                return Err(CompileError::SweepLiesInThePlane);
                            }
                            Some(local)
                        }
                    };

                    // Degrees on the wire and a tangent in the field, converted once here: the
                    // number a person types is an angle, and the arithmetic wants the slope.
                    // Doing it at every sample would be the same conversion a million times.
                    let slope = |draft: &fieldform_scene::ParamRef| {
                        let degrees = resolve(draft).ok_or(CompileError::UnresolvedRadius)?;
                        if !(-89.0..=89.0).contains(&degrees) {
                            return Err(CompileError::DraftTooSteep(degrees));
                        }
                        Ok(degrees.to_radians().tan())
                    };
                    let draft_tangent = slope(draft)?;
                    // A wall's corners cannot be mitred yet; see the variant. Only when there
                    // is a draft, since with upright walls the two answers are one solid.
                    let any_draft = draft_tangent != 0.0
                        || matches!(sides, fieldform_scene::Sides::Two { draft, .. }
                            if slope(draft).map(|t| t != 0.0).unwrap_or(false));
                    if outline.is_wall() && any_draft && !corners.is_default() {
                        return Err(CompileError::SharpCornersOnAWall);
                    }
                    // Where along the sweep the material begins, and which way it lies about
                    // the plane -- side two with a slope of its own, held to the same limit.
                    let start = resolve(start).ok_or(CompileError::UnresolvedRadius)?;
                    let depth = match sides {
                        fieldform_scene::Sides::One => crate::Depth::One,
                        fieldform_scene::Sides::Symmetric => crate::Depth::Symmetric,
                        fieldform_scene::Sides::Two { height, draft } => crate::Depth::Two {
                            height: resolve(height).ok_or(CompileError::UnresolvedRadius)?,
                            draft_tangent: slope(draft)?,
                        },
                    };

                    let mut rims = resolved_rim(rims, &resolve)?;
                    if !rims.is_hard() {
                        // Half the length, past which the two ends' roundings meet -- all of
                        // it when only one end is rounded; and the section's depth at the
                        // narrowest end that is rounded, past which the rounding meets itself
                        // across the section. A narrowing draft takes from a far end: side
                        // one's is the end, and side two's -- the profile's own, undrafted, for
                        // one side -- the start.
                        let (one, two) = match depth {
                            crate::Depth::Two { height: other, draft_tangent: other_tangent } => {
                                ((h.abs(), draft_tangent), (other.abs(), other_tangent))
                            }
                            crate::Depth::Symmetric => ((h.abs() * 0.5, draft_tangent), (h.abs() * 0.5, draft_tangent)),
                            crate::Depth::One => ((h.abs(), draft_tangent), (0.0, 0.0)),
                        };
                        let length = one.0 + two.0;
                        check_rim(rims.reach, rims.room_along(length), "solid's length", "longer")?;
                        let narrowed = match rims.ends {
                            fieldform_scene::RimEnds::Both => (one.0 * one.1).min(two.0 * two.1),
                            fieldform_scene::RimEnds::End => one.0 * one.1,
                            fieldform_scene::RimEnds::Start => two.0 * two.1,
                        }
                        .min(0.0);
                        check_rim(
                            rim_second_leg(&rims),
                            built.depth() + narrowed,
                            "section's depth",
                            "thicker",
                        )?;
                    }
                    // **Square readings read the exact corner** (`Rim::cut`): the profile's plane
                    // and the slab are square to each other when the sweep is the profile's own
                    // normal and no draft tilts a wall -- the extrude most are -- so its rims
                    // read the distance along their normal rather than the bound's `0.707`.
                    // The generator decides from the same facts (`extrude_rims_are_square`).
                    let steepest = match depth {
                        crate::Depth::Two { draft_tangent: other, .. } => draft_tangent.abs().max(other.abs()),
                        _ => draft_tangent.abs(),
                    };
                    if leaning.is_none() && steepest == 0.0 {
                        rims.square = Some(1.0);
                    }
                    let solid = ops.len();
                    ops.push(Op::Leaf(match leaning {
                        None => std::sync::Arc::new(crate::Extrusion {
                            footprint: built,
                            start,
                            depth,
                            // The sign is kept: it is which way out of the plane the solid
                            // grows, which is a question that only exists now that it starts
                            // there rather than straddling.
                            height: h,
                            draft_tangent,
                            corners: *corners,
                            rims,
                        }) as std::sync::Arc<dyn ScalarField + Send + Sync>,
                        Some(direction) => std::sync::Arc::new(crate::SkewExtrusion {
                            footprint: built,
                            start,
                            height: h,
                            depth,
                            direction,
                            draft_tangent,
                            corners: *corners,
                            rims,
                        }),
                    }));
                    Op::Transform {
                        input: solid,
                        rotation,
                        translation,
                        scale: 1.0,
                    }
                }
                Node::Revolve { outline, frame, angle, sides, rims } => {
                    let (translation, rotation) = resolve_frame(document, graph, frame, &resolve)?;
                    let into = move |p: Point3<f64>| {
                        Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
                    };
                    // The half-plane the section lies in, which a partial revolve starts from.
                    let starts_at;
                    let built: crate::Footprint = match outline {
                        fieldform_scene::Outline::Closed { profile } => {
                            // The loop this curve is part of, exactly as an extrude reads it.
                            let world =
                                crate::profile_loop(document, *profile, frame.curve_source())
                                    .map_err(|e| CompileError::BadLoop(*profile, e))?
                                    .points;
                            starts_at = crate::Profile::azimuth(&world, into);
                            // The profile's radial footprint, whether or not it lies in the
                            // axis's plane -- `about_axis` says why that is no longer refused.
                            let built = crate::Profile::about_axis(&world, into)
                                .map_err(|e| CompileError::BadProfile(*profile, e))?;
                            // A revolve reads its profile's own boundary, plus whatever
                            // consumes it.
                            bake_to_tolerance(
                                built,
                                tolerance_mm,
                                profile_reads_out_to(document, graph, *id),
                                cache.as_deref_mut(),
                            )
                            .into()
                        }
                        // A thin revolve: the chain laid in the half-plane by the map a
                        // profile is laid by, and given its thickness there.
                        fieldform_scene::Outline::Wall { curve, thickness, side, ends } => {
                            let chain =
                                crate::outline_chain(document, *curve, frame.curve_source())
                                    .map_err(|e| CompileError::BadLoop(*curve, e))?;
                            starts_at = crate::Profile::azimuth(&chain.points, into);
                            let thickness =
                                resolve(thickness).ok_or(CompileError::UnresolvedRadius)?;
                            let wall =
                                wall_about_axis(&chain, *curve, thickness, *side, *ends, into)?;
                            bake_to_tolerance(
                                wall,
                                tolerance_mm,
                                profile_reads_out_to(document, graph, *id),
                                cache.as_deref_mut(),
                            )
                            .into()
                        }
                    };
                    // How far round, and which way about the section's own half-plane. All
                    // the way round is the solid every revolve was, read exactly as before;
                    // anything less is a wedge, and nothing at all is refused by the number.
                    let solid = ops.len();
                    let (from, total) = revolve_wedge(document, angle, sides)?;
                    let mut rims = resolved_rim(rims, &resolve)?;
                    ops.push(if total >= 360.0 - 1e-9 {
                        if !rims.is_hard() {
                            return Err(CompileError::RimsOnAFullRevolve);
                        }
                        Op::Leaf(std::sync::Arc::new(crate::Revolution { footprint: built }))
                    } else {
                        if !rims.is_hard() {
                            // **Held so that no two rim readings meet** (`RevolvedWedge`): the
                            // section's distance from the axis -- a section on it has no room at
                            // all -- times the sine of the angle each rounded cap has to itself
                            // (half the sweep with both rounded), and of half the gap the revolve
                            // leaves, across which the two sides face each other; and the
                            // section's depth, as an extrude's rim is.
                            let r_min = built.bounds().0.x;
                            check_rim(rims.reach, r_min.max(0.0), "section's distance from the axis", "further from the axis")?;
                            let to_itself = match rims.ends {
                                fieldform_scene::RimEnds::Both => total * 0.5,
                                _ => total,
                            };
                            let room = |degrees: f64| r_min * degrees.min(90.0).to_radians().sin();
                            check_rim(rims.reach, room(to_itself), "revolve's angle", "turned further")?;
                            check_rim(rims.reach, room((360.0 - total) * 0.5), "gap the revolve leaves", "turned less")?;
                            check_rim(rim_second_leg(&rims), built.depth(), "section's depth", "thicker")?;
                            // **The exact corner, through the lean its own radius makes**
                            // (`Rim::cut`): the section is read in the query's meridian and the
                            // cap's distance across its plane, and a point `y` out from that
                            // plane at radius `r` has the two leaning by `y/r` -- within the
                            // reach, by at most `k/r_min`. The scale rides with the shader's
                            // slots through `morph_divisors`.
                            let scale = (1.0 + rims.reach / r_min).sqrt();
                            rims.square = Some(scale);
                            rim_scales.push((*id, scale));
                        }
                        Op::Leaf(std::sync::Arc::new(
                            crate::RevolvedWedge {
                                footprint: built,
                                from: starts_at + from.to_radians(),
                                sweep: total.to_radians(),
                                rims: Default::default(),
                            }
                            .with_rims(rims),
                        ))
                    });
                    Op::Transform {
                        input: solid,
                        rotation,
                        translation,
                        scale: 1.0,
                    }
                }
                Node::Boolean { op, inputs } => {
                    // Refused rather than evaluated as "infinitely far": a body that is
                    // nowhere would look exactly like a body that is hidden.
                    if inputs.is_empty() {
                        return Err(CompileError::NothingToCombine);
                    }
                    Op::Boolean {
                        op: *op,
                        inputs: inputs.iter().map(|i| slot[i]).collect(),
                    }
                }
                Node::Blend { op, a, b, fillet } => {
                    // Allowed, and *reported* rather than refused — see
                    // [`approximate_blends`]. The short version: a blend against a
                    // lower-bound operand is inaccurate by a bounded, one-directional amount,
                    // and refusing it outright removed a shape people want to make.
                    // The radius may follow a ramp (spec §9.1), read at the point the blend is
                    // asked at -- a ramp like any other, so "larger toward the top" is a ramp
                    // off a plane and "larger along the seam" a ramp aimed along it. Only a
                    // ramp, because only a ramp states how fast it changes, which is what the
                    // divisor is made of.
                    let (radius, divisor) = match fillet.radius.driver() {
                        None => (
                            Amount::Fixed(
                                resolve(&fillet.radius).ok_or(CompileError::UnresolvedBlend)?,
                            ),
                            1.0,
                        ),
                        Some(driver) => {
                            let at = *slot
                                .get(&driver)
                                .ok_or_else(|| no_driver(graph, &incomplete, driver))?;
                            if !matches!(ops[at], Op::Ramp { .. }) {
                                return Err(CompileError::DrivenParameterNeedsARamp);
                            }
                            // Through the graph-level walk the shader generator reads too, so
                            // both paths divide by one number.
                            (Amount::Driven(at), driven_blend_divisor(document, graph, *id))
                        }
                    };
                    // A chamfer's second leg, resolved; a round carries none
                    // (`Fillet::second_leg`). Not with a radius that follows a ramp: the scaling
                    // would follow it too, and that is a second bound nothing has asked for.
                    let second = match fillet.second_leg() {
                        None => None,
                        Some(s) => {
                            if fillet.radius.driver().is_some() {
                                return Err(CompileError::SecondLegNeedsAFixedRadius);
                            }
                            Some(resolve(s).ok_or(CompileError::UnresolvedBlend)?.max(0.0))
                        }
                    };
                    Op::Blend {
                        op: *op,
                        a: slot[a],
                        b: slot[b],
                        radius,
                        exponent: fillet.exponent(),
                        taper: taper_ratio(document, graph, *id),
                        divisor,
                        second,
                    }
                }
                Node::Twist {
                    input,
                    degrees_per_mm,
                    frame,
                } => {
                    let rate = resolve(degrees_per_mm)
                        .ok_or(CompileError::UnresolvedRadius)?
                        .to_radians();
                    // How far the warp can stretch material, which is what the value must be
                    // divided by to stay a lower bound.
                    //
                    // A twist turns a point by `rate · z`, so two points at the same height and
                    // radius `r` are carried apart by `r · rate` per millimetre of height. That
                    // grows without limit with radius — so the bound is taken from the *body's
                    // own* extent, which the graph already knows how to compute.
                    //
                    // A body with no bounds has no such radius, and there is nothing honest to
                    // divide by. Refused rather than guessed: a made-up bound is a field that
                    // overstates its distance, which is a ray stepping through a wall.
                    let extent = operand_extent(document, graph, input);
                    let Some((lo, hi)) = extent.as_box() else {
                        return Err(CompileError::TwistNeedsBounds);
                    };
                    // The reach measured *across the chosen axis*, since that is the direction
                    // the winding carries material sideways in. Measured in the axis's own
                    // frame, so the same arithmetic serves all three.
                    // Wound about the frame's own +Z, and the frame is what carries that
                    // wherever it points — the same move the lattice makes, and for the same
                    // reason: `Op::Twist` already winds about an axis correctly, so putting the
                    // points into the frame first is the whole of aiming it. A twist that knew
                    // about arbitrary axes itself would be a second implementation of a
                    // rotation this crate has three of already.
                    let (translation, rotation) = resolve_frame(document, graph, frame, &resolve)?;
                    let at = Point3::origin();
                    let axis = 2usize;
                    let _ = (lo, hi);
                    // **Through the one reader**, because three places need this number and two
                    // of them were computing nothing at all. See [`twist_stretch`].
                    let stretch = twist_stretch(document, graph, *id);
                    if frame.is_default() {
                        Op::Twist {
                            input: slot[input],
                            radians_per_mm: rate,
                            stretch,
                            axis,
                            at,
                        }
                    } else {
                        // **A sandwich, not a wrapper, and the difference is the whole bug.**
                        //
                        // A lattice is a leaf: one transform in front of it puts the query
                        // point in its frame and that is the end of it. A twist *wraps a body*,
                        // so a single transform would also hand the body a point in the frame's
                        // local space — and the body lives in world space. The result was a
                        // twist about the right axis applied to a body that had been moved.
                        //
                        // What is wanted is `R . twist . R⁻¹` on the point: go into the frame,
                        // wind about its own Z, come back out, then look the body up. So the
                        // body is wrapped in the *inverse* placement and the whole is wrapped
                        // in the placement.
                        //
                        // `Op::Transform` evaluates its input at `R⁻¹(p − t)`, so the inverse
                        // of that map is the transform with rotation `R⁻¹` and translation
                        // `−R⁻¹t`.
                        let back = ops.len();
                        ops.push(Op::Transform {
                            input: slot[input],
                            rotation: rotation.inverse(),
                            translation: -(rotation.inverse() * translation),
                            scale: 1.0,
                        });
                        let wound = ops.len();
                        ops.push(Op::Twist {
                            input: back,
                            radians_per_mm: rate,
                            stretch,
                            axis,
                            at,
                        });
                        Op::Transform {
                            input: wound,
                            rotation,
                            translation,
                            scale: 1.0,
                        }
                    }
                }
                Node::Attract { input, toward, strength, falloff } => {
                    // **One spine either way.** A curve becomes its tessellation; a point
                    // becomes a spine of one vertex, which the nearest-point search already
                    // answers with that vertex -- see `Spine::nearest`, which has the case
                    // written out, and `closest_on_curve` in the shader, which falls back to
                    // the first point when there is no segment to walk. So a point is not a
                    // special case in the arithmetic, only in what names it.
                    let spine = match toward {
                        fieldform_scene::Attractor::Curve(curve) => {
                            let c = document
                                .curves
                                .get(curve)
                                .ok_or(CompileError::MissingCurve(*curve))?;
                            crate::Spine::of_curve(c, tolerance_mm)
                        }
                        fieldform_scene::Attractor::Point(node) => {
                            let Some(Node::Point { at }) = graph.node(*node) else {
                                return Err(CompileError::AttractorIsNotAPoint);
                            };
                            let p = Point3::new(
                                resolve(&at[0]).ok_or(CompileError::UnresolvedAttractor)?,
                                resolve(&at[1]).ok_or(CompileError::UnresolvedAttractor)?,
                                resolve(&at[2]).ok_or(CompileError::UnresolvedAttractor)?,
                            );
                            crate::Spine::new(vec![p])
                        }
                    };
                    let strength = resolve(strength).ok_or(CompileError::UnresolvedAttractor)?;
                    let falloff = resolve(falloff).ok_or(CompileError::UnresolvedAttractor)?;
                    // Unlike a twist, nothing here has to be taken from the body's extent: the
                    // warp's largest stretch is `1 + strength` wherever the body happens to be,
                    // so an attractor wraps a field with no bounds without having to refuse it.
                    Op::Attract {
                        input: slot[input],
                        spine: std::sync::Arc::new(spine),
                        strength,
                        falloff,
                        stretch: crate::attract::stretch(strength),
                    }
                }
                Node::Pattern { input, fillet, .. } => {
                    if fillet.continuity == fieldform_scene::Continuity::Chamfer
                        && !fillet.is_plainly_hard()
                    {
                        return Err(CompileError::ChamferOnAPattern);
                    }
                    // **One layout, asked for and not worked out again here.** The count, the
                    // step or the angle, the options and the copies left out are all answered by
                    // [`PatternLayout`], which refuses by name what has no copies instead of
                    // repairing it — so a broken expression cannot arrive as one plausible copy.
                    let layout = PatternLayout::of(document, graph, *id).map_err(|refusal| {
                        match refusal {
                            crate::PatternRefusal::Count(e) => CompileError::PatternCount(e.say()),
                            other => CompileError::Pattern(other.say()),
                        }
                    })?;
                    // **The block being repeated, which the window is measured from.** `slot`
                    // gives the op this arm folds; the reach and the origin are readings of the
                    // *graph*, so they need the block itself. Opened here rather than at each of
                    // the four, and said as a refusal rather than an unwrap because a slot is the
                    // one thing in this arm that can be empty and the answer to an empty one is
                    // never a number. `incomplete` has already skipped a pattern with a hole, so
                    // nothing reaches this.
                    let body = input.source().ok_or_else(|| empty_slot(graph, *id))?;
                    let inverse: Vec<Isometry3<f64>> =
                        (0..layout.total).map(|i| layout.placement(i).inverse()).collect();
                    // How far back the copies are rounded into each other, which the window
                    // has to know about before it is sized: a fillet reads its operands out to
                    // its radius, so a copy that far away contributes.
                    let radius = resolve(&fillet.radius)
                        .ok_or(CompileError::UnresolvedBlend)?
                        .max(0.0);
                    // **One answer about the window, asked rather than worked out again.**
                    // This arm used to compute its own `cells_reached` beside `pattern_reach`'s
                    // -- two derivations of one number, which is two chances to differ, and the
                    // shader asks the other one. That is the shape of bug this file has met
                    // under several names; there is one function now and everybody calls it.
                    let reach = pattern_reach_in(document, graph, body, &layout, radius);
                    // Which way the copies march, and where the body itself sits along it.
                    let origin = pattern_origin_in(document, graph, body, &layout);
                    let fold = match &layout.shape {
                        PatternShape::Row { step, .. } => {
                            let pitch = step.norm();
                            let unit = step.unscale(pitch.max(1e-12));
                            Fold::Linear { unit, pitch, origin }
                        }
                        PatternShape::Grid { step, count, step2, count2, .. } => {
                            let dual = crate::pattern_layout::grid_dual(step, step2)
                                .expect("a layout refuses two steps along one line");
                            Fold::Grid {
                                anchor: pattern_grid_anchor_in(document, graph, body, &layout),
                                steps: [*step, *step2],
                                dual,
                                counts: [*count as usize, *count2 as usize],
                                reach: pattern_reach_axes_in(
                                    document, graph, body, &layout, radius,
                                ),
                            }
                        }
                        PatternShape::Ring { at, axis, count, arc: false, .. } => Fold::Radial {
                            at: *at,
                            axis: *axis,
                            count: *count as usize,
                            origin,
                        },
                        PatternShape::Ring { at, axis, count, step, arc: true, .. } => Fold::Arc {
                            at: *at,
                            axis: *axis,
                            count: *count as usize,
                            step: *step,
                            origin,
                        },
                    };
                    Op::Pattern {
                        input: slot[input],
                        inverse,
                        fold,
                        reach,
                        left_out: layout.left_out(),
                        excluded: layout.excluded.clone(),
                        // Allowed over a lower-bound operand and *reported* rather than
                        // refused, exactly as a blend block is -- see [`approximate_blends`],
                        // which now lists a rounded fold for the same reason it lists a blend.
                        radius,
                        exponent: fillet.exponent(),
                        taper: taper_ratio(document, graph, *id),
                    }
                }
                Node::Mirror { input, frame, join } => {
                    // One reader for the plane, shared with every other block that takes a
                    // frame (D9). The two cases this used to spell out — an axis through a
                    // point, and a plane node fed in — are `Frame::At` and `Frame::From`, and
                    // neither is a mirror's business any more.
                    //
                    // Which side of the plane is solid does not matter to a reflection: a plane
                    // and its flip are the same mirror. `resolve_frame` turns the flip into the
                    // opposite normal and the mirror is indifferent to it, which is right
                    // rather than merely harmless.
                    let (at, rotation) = resolve_frame(document, graph, frame, &resolve)?;
                    let at = Point3::from(at);
                    let normal = rotation * Vector3::z();
                    Op::Mirror {
                        input: slot[input],
                        at,
                        normal,
                        join: *join,
                        // Asked of the graph, not of the resolved plane above, so this side and
                        // the shader decide it the same way from the same two facts.
                        one_side: mirror_one_side(document, graph, *id),
                        reaches_its_plane: crate::mirror_reaches_its_plane(document, graph, *id),
                    }
                }
                Node::Ramp {
                    input,
                    in_min,
                    in_max,
                    out_min,
                    out_max,
                    shape,
                } => {
                    let lo = resolve(in_min).ok_or(CompileError::UnresolvedRamp)?;
                    let hi = resolve(in_max).ok_or(CompileError::UnresolvedRamp)?;
                    let out_lo = resolve(out_min).ok_or(CompileError::UnresolvedRamp)?;
                    let out_hi = resolve(out_max).ok_or(CompileError::UnresolvedRamp)?;
                    // Refused rather than clamped to something small. A zero span is a step
                    // function: its slope is unbounded, so there is no number for a consumer
                    // to divide by, and every bound above it would be a claim with nothing
                    // behind it. `hi` below `lo` is fine — it runs the mapping backwards.
                    if (hi - lo).abs() < f64::EPSILON {
                        return Err(CompileError::EmptyRampRange);
                    }
                    Op::Ramp {
                        input: slot[input],
                        in_min: lo,
                        in_span: hi - lo,
                        out_min: out_lo,
                        out_span: out_hi - out_lo,
                        shape: *shape,
                    }
                }
                // One node, two ops: the body, minus an inward offset of itself. The
                // arithmetic is the composition it has always been — what changed is that a
                // person sees one block and answers one question. Building the offset here
                // rather than as its own node is what makes that true without giving shells a
                // second evaluation path to keep in step with offsets.
                Node::Shell { input, thickness } => {
                    // **A body with no bounds has no wall.** Shelling is the body minus the
                    // body shrunk, and on a half-space that difference is an infinite slab: it
                    // renders, and it is not a thing anyone can mesh, measure or make. A plane
                    // is documented as "infinite until you cut with it", and this is the other
                    // half of that sentence.
                    //
                    // Refused rather than quietly produced, the same way a twist refuses an
                    // unbounded input, and refused in the compile rather than hidden in the
                    // menu -- a control that vanishes teaches nothing, and this codebase has
                    // already learned that once with the revolve axis.
                    if operand_extent(document, graph, input).as_box().is_none() {
                        return Err(CompileError::ShellNeedsBounds);
                    }
                    let body = slot[input];
                    let t = resolve(thickness).ok_or(CompileError::UnresolvedOffset)?;
                    // Negative, because the inner solid is the body *shrunk*: `Op::Offset`
                    // subtracts, so shrinking by `t` is subtracting `-t`.
                    let cavity = ops.len();
                    ops.push(Op::Offset {
                        input: body,
                        distance: Amount::Fixed(-t),
                        divisor: 1.0,
                    });
                    Op::Boolean {
                        op: BooleanOp::Difference,
                        inputs: vec![body, cavity],
                    }
                }
                Node::Morph { a, b, t } => {
                    if t.driver().is_some() {
                        driven_morphs.push((*id, ops.len()));
                    }
                    let amount = match t.driver() {
                        // The divisor this needs cannot be worked out here: it is bounded from
                        // the two operands' *intervals*, and those are read over the finished
                        // op list. Left at one and patched below, which is why `divisor` is a
                        // field rather than something folded into the expression.
                        Some(driver) => {
                            let at = *slot
                                .get(&driver)
                                .ok_or_else(|| no_driver(graph, &incomplete, driver))?;
                            let Op::Ramp { out_min, out_span, .. } = ops[at] else {
                                return Err(CompileError::DrivenParameterNeedsARamp);
                            };
                            // **The rule a literal `t` already obeys, applied where a ramp says
                            // it instead.** A morph at `t = 7.6` is not "more B", it is
                            // `(1 - 7.6)*A + 7.6*B` -- an extrapolation past both shapes, which
                            // is what the literal check calls a file saying something it cannot
                            // mean. This path skipped the check entirely, so the rule held for a
                            // number typed in and not for the same number arriving from a ramp.
                            //
                            // The cost is not only meaning. A driven morph is divided by
                            // `1 + sup|B - A|*L_t`, and `L_t` is the ramp's output span over its
                            // input span -- so an output range seven times wider than it may be
                            // makes every step seven times shorter. Reported as a viewport that
                            // stopped being usable: a 2 mm band mapped onto 0.6..7.6 prices at
                            // x165.7, where that same band mapped onto its legal 0..1 is x24.
                            let ends = (out_min, out_min + out_span);
                            let (lo, hi) = (ends.0.min(ends.1), ends.0.max(ends.1));
                            if lo < -1e-9 || hi > 1.0 + 1e-9 {
                                return Err(CompileError::MorphDriverOutOfRange(lo, hi));
                            }
                            // **Both operands must be bounded**, because `sup|B − A|` is read
                            // over the box that encloses them and an unbounded body has no box.
                            // A morph of a bare lattice against a plane has nothing to bound and
                            // nothing to guess -- intersect it with something first, which is
                            // the answer item 8 already gives.
                            for side in [a, b] {
                                if operand_extent(document, graph, side).as_box().is_none() {
                                    return Err(CompileError::MorphNeedsBounds);
                                }
                            }
                            Amount::Driven(at)
                        }
                        None => {
                            let v = resolve(t).ok_or(CompileError::UnresolvedRadius)?;
                            // Refused rather than clamped at compile: a literal outside the
                            // range is a file saying something it cannot mean, and silently
                            // reading it as 0 or 1 hides which.
                            if !(0.0..=1.0).contains(&v) {
                                return Err(CompileError::MorphOutOfRange(v));
                            }
                            Amount::Fixed(v)
                        }
                    };
                    Op::Morph {
                        a: slot[a],
                        b: slot[b],
                        t: amount,
                        // One while `t` is constant, and not a placeholder: see `Op::Morph`.
                        // The driven case goes through the same function the shader calls, so
                        // the two cannot divide by different numbers.
                        divisor: match amount {
                            Amount::Fixed(_) => 1.0,
                            Amount::Driven(_) => driven_morph_divisor(document, graph, *id),
                        },
                    }
                }
                Node::Offset { input, distance } => {
                    let inner = slot[input];
                    match distance.driver() {
                        None => Op::Offset {
                            input: inner,
                            // Any finite distance is accepted, including one that erodes the
                            // body away entirely: `F - d` is still a well-formed field, and an
                            // empty result is a true answer rather than an error.
                            distance: Amount::Fixed(
                                resolve(distance).ok_or(CompileError::UnresolvedOffset)?,
                            ),
                            divisor: 1.0,
                        },
                        Some(driver) => {
                            let at = *slot
                                .get(&driver)
                                .ok_or_else(|| no_driver(graph, &incomplete, driver))?;
                            if !matches!(ops[at], Op::Ramp { .. }) {
                                return Err(CompileError::DrivenParameterNeedsARamp);
                            }
                            // Through the shared graph-level walk rather than over the ops
                            // being built, so that the shader — which has no ops — divides by
                            // the same number. A mismatch would move no surface and so would
                            // never appear as a residual; it would show only as a tracer
                            // stepping further than the field justifies.
                            Op::Offset {
                                input: inner,
                                distance: Amount::Driven(at),
                                divisor: driven_offset_divisor(document, graph, *id),
                            }
                        }
                    }
                }
                Node::Transform {
                    input,
                    translation,
                    rotation,
                    scale,
                    about,
                } => {
                    let t = resolve_placement(translation, &resolve)?;
                    let r = resolve_placement(rotation, &resolve)?;
                    let s = resolve(scale).ok_or(CompileError::UnresolvedRadius)?;
                    // Refused rather than clamped. Clamping would place the body at some size
                    // nobody asked for; the manipulator already stops a drag before it gets
                    // here, so a non-positive scale in a file means the file describes
                    // something with no shape, and saying so is more use than guessing.
                    if s <= 0.0 {
                        return Err(CompileError::NonPositiveScale(s));
                    }
                    let turn = euler_degrees_to_rotation(r);
                    // The pivot folded into the translation, which is all a pivot is: `c` is a
                    // fixed point of `x -> c + s·R·(x − c) + t`, and that is this placement with
                    // `c − s·R·c` added to its move. So nothing below here knows about pivots.
                    let c = resolve_pivot(document, graph, *id, about, &resolve)?;
                    Op::Transform {
                        input: slot[input],
                        rotation: turn,
                        translation: t + c.coords - turn * (c.coords * s),
                        scale: s,
                    }
                }
                Node::Unknown(_) => unreachable!("refused above, before any op was built"),
            };
            slot.insert(*id, ops.len());
            ops.push(op);
        }

        // A part's field is the union of its bodies. One body needs no wrapper; several fold
        // into hard unions, which is what "two solids sitting side by side" means as a field —
        // and it reuses the boolean the rest of this file already handles, so intervals and
        // normals need nothing new.
        //
        // **In the document's body order**, which is the order the shader folds in — so the
        // two descriptions of one part are folded the same way rather than each by its own
        // rule. `min` is commutative, so the sign at a point does not depend on this; what
        // does is `body_at`'s answer where two bodies are equally near, and being the same
        // answer the shader would give is the point of asking one function.
        let mut bodies: Vec<(fieldform_scene::NodeId, usize)> = Vec::new();
        for id in document.folded_bodies(graph) {
            // **A body with a hole is refused, not drawn.** It was skipped above with everything
            // `incomplete` names, so there is no op to fold in -- and there is nothing to fold in
            // its place, because no value a missing input could supply keeps `distance` from
            // overestimating. The viewport prunes such a body and draws its siblings
            // (`notebook-spec.md` 7.2); a compile asked for the whole part says why it cannot.
            if incomplete.contains(&id) {
                // A body that is, or reads, an open cage is refused for that, not for a hole
                // it does not have.
                if let Some(cage) = red_cage_behind(document, graph, id) {
                    return Err(open_cage(document, cage));
                }
                return Err(empty_slot(graph, id));
            }
            bodies.push((id, slot[&id]));
        }
        let folded: Vec<usize> = bodies.iter().map(|(_, op)| *op).collect();
        if folded.is_empty() {
            return Err(CompileError::Empty);
        }
        // One body is itself: wrapping it would put an op in the way of every sample for no
        // reason, and `bodies` already records where it landed.
        let output = if let [only] = folded[..] {
            only
        } else {
            ops.push(Op::Boolean { op: BooleanOp::Union, inputs: folded });
            ops.len() - 1
        };
        // The critical surfaces, each as the field of its own subtree. Only when something is
        // rounded: with every join hard there is no radius to taper and no reason to pay for a
        // second field at every sample.
        let rounds = ops.iter().any(|op| match op {
            Op::Blend { radius, .. } => may_round(radius),
            Op::Pattern { radius, .. } => *radius > 0.0,
            _ => false,
        });
        let mut protect = Vec::new();
        for mark in graph.critical().iter().filter(|_| rounds) {
            let Some(alone) = graph.subtree(mark.node) else { continue };
            let clearance = resolve(&mark.clearance)
                .ok_or(CompileError::UnresolvedClearance)?
                .max(0.0);
            // No cache handed down: a leaf cache is keyed for one compile's borrow, and a mark
            // is a handful of nodes.
            let own = Self::build(document, &alone, tolerance_mm, None)?;
            protect.push((std::sync::Arc::new(own), clearance));
        }
        let mut field = Self {
            ops,
            output,
            protect,
            bodies,
            morph_divisors: rim_scales,
        };
        field.bound_driven_morphs(document, graph, &driven_morphs);
        Ok(field)
    }

    /// Replaces every driven morph's divisor with one read from its operands' intervals.
    ///
    /// **A second pass, because the bound needs the finished ops.** `∇M` for a driven `t` is
    /// `(1 − t)·∇A + t·∇B + (B − A)·∇t`, so the value can rise at `1 + sup|B − A|·L_t` per
    /// millimetre and must be divided by that to stay a distance. `sup|B − A|` is what the two
    /// operands' intervals give, and an interval is evaluated over the op list.
    ///
    /// **This is the number `morph-spec.md` §4.3 asks for, and the reason it is worth the second
    /// pass is that the box-based bound beside it is not close.** Bounding each operand by the
    /// diagonal of the box enclosing both is exact only when the two shapes are as far apart as
    /// that box allows — and a morph is *for* shapes in the same place, which is the case it is
    /// worst on. Reported from a model that would not draw: a mesh morphed to that same mesh
    /// intersected with a gyroid, over a 2 mm band, priced the box way at **×291.7** when the
    /// two fields are never more than a lattice cell apart. Every step is that much shorter, so
    /// the viewport asked for thirty-two times the steps and stopped being usable.
    ///
    /// Over the box enclosing both operands, which is where the morph's surface can be at all:
    /// outside it both fields are positive and there is nothing to trace.
    fn bound_driven_morphs(
        &mut self,
        document: &Document,
        graph: &Graph,
        driven: &[(fieldform_scene::NodeId, usize)],
    ) {
        for (node, index) in driven.iter().copied() {
            let Op::Morph { a, b, t, .. } = self.ops[index] else { continue };
            let Amount::Driven(at) = t else { continue };
            let Some(Node::Morph { a: na, b: nb, .. }) = graph.node(node) else { continue };
            let Some((lo, hi)) = operand_extent(document, graph, na)
                .union(operand_extent(document, graph, nb))
                .as_box()
            else {
                continue;
            };
            let (a_lo, a_hi) = self.interval_at(a, lo, hi);
            let (b_lo, b_hi) = self.interval_at(b, lo, hi);
            // `B − A` over the box, and the largest magnitude it reaches.
            let spread = (b_lo - a_hi).abs().max((b_hi - a_lo).abs());
            let rate = ramp_lipschitz(&self.ops, at);
            // Never below one: `(1 − t)·∇A + t·∇B` alone reaches one, so a divisor under it
            // would claim more room than there is.
            let divisor = (1.0 + spread * rate).max(1.0);
            if !divisor.is_finite() {
                continue;
            }
            if let Op::Morph { divisor: slot, .. } = &mut self.ops[index] {
                *slot = divisor;
            }
            self.morph_divisors.push((node, divisor));
        }
    }

    /// What each driven morph was divided by, for the shader to divide by the same.
    pub fn morph_divisors(&self) -> &[(fieldform_scene::NodeId, f64)] {
        &self.morph_divisors
    }

    /// How far `p` is from the nearest critical surface, less that surface's clearance.
    ///
    /// **Asked once per query, at the world point, and handed down unchanged.** A blend under a
    /// placement or inside a pattern sees a local point, and the exactness a critical surface
    /// is promised is pointwise in *world* space (`blend-integrity-spec.md` §6.4) — so this is
    /// never recomputed from whatever point a nested op happens to hold. Infinite when nothing
    /// is marked, which makes every fillet its full radius with no branch taken.
    fn gap_at(&self, p: Point3<f64>) -> f64 {
        self.protect
            .iter()
            .map(|(field, clearance)| field.distance(p).abs() - clearance)
            .fold(f64::INFINITY, f64::min)
    }

    fn eval(&self, index: usize, p: Point3<f64>) -> f64 {
        self.eval_in(index, p, self.gap_at(p))
    }

    fn normal_at(&self, index: usize, p: Point3<f64>) -> Option<Vector3<f64>> {
        self.normal_in(index, p, self.gap_at(p))
    }

    /// Evaluates one node and, recursively, whatever it consumes.
    ///
    /// Recursive rather than a scratch buffer so a distance query allocates nothing — this is
    /// called once per sample point when meshing, which is millions of times. A node feeding
    /// two consumers is evaluated twice; v1 graphs are trees, so that costs nothing today, and
    /// it is the thing to change first if diamond-shaped graphs become common.
    fn eval_in(&self, index: usize, p: Point3<f64>, gap: f64) -> f64 {
        match &self.ops[index] {
            Op::Leaf(field) => field.distance(p),
            // The fixed lattice of whatever size the ramp says here, over the bound that keeps
            // the pair honest.
            Op::Lattice { lattice, sizing, divisor } => {
                lattice.sized(self.eval_in(*sizing, p, gap)).distance(p) / divisor
            }
            // A loop, which is the whole point: the same `min` a nest of pairs would have
            // computed, without a stack frame per operand.
            Op::Boolean { op, inputs } => {
                let mut values = inputs.iter().map(|i| self.eval_in(*i, p, gap));
                let first = values.next().unwrap_or(f64::INFINITY);
                match op {
                    BooleanOp::Union => values.fold(first, f64::min),
                    // Subtracting an operand means intersecting with its complement, and the
                    // complement of a distance field is its negation.
                    BooleanOp::Difference => values.fold(first, |kept, d| kept.max(-d)),
                    BooleanOp::Intersection => values.fold(first, f64::max),
                }
            }
            Op::Blend {
                op,
                a,
                b,
                radius,
                exponent,
                taper,
                divisor,
                second,
            } => {
                let da = self.eval_in(*a, p, gap);
                let db = self.eval_in(*b, p, gap);
                // The radius this point gets, and the division that keeps a varying one a
                // bound: one for a fixed radius, so a fixed blend reads what it always did.
                let k = self.amount_in(radius, p, gap);
                // A chamfer of two legs is the equal-legged one over scaled operands: the
                // bevel leaves `a` at `s_a` from `b` and `b` at `s_b` from `a`, so `a·m/s_b`
                // and `b·m/s_a` with `m = min(s_a, s_b)` put its plane where `a' + b' = m` is.
                // Each scale is at most one, so the bound holds as it did.
                let (da, db, k) = two_legs(da, db, k, *second);
                blend(*op, da, db, k, *exponent, tapered(k, gap), *taper) / divisor
            }
            // The whole operation. Subtracting a constant slides every level set outward by
            // that much, and the result is the exact distance to the moved boundary rather
            // than an approximation of it.
            // Divided by the divisor, which is 1 unless the distance varies. See [`Amount`]:
            // the division is what keeps a driven offset from claiming more room than it has,
            // and it leaves the surface exactly where it was because a positive constant does
            // not move a zero crossing.
            Op::Morph {
                a,
                b,
                t,
                divisor,
            } => {
                // Clamped at evaluation as well as refused at compile, because a *driven* `t`
                // is a ramp's output and a ramp is only clamped to its own range.
                let t = self.amount_in(t, p, gap).clamp(0.0, 1.0);
                (self.eval_in(*a, p, gap) * (1.0 - t) + self.eval_in(*b, p, gap) * t) / divisor
            }
            Op::Offset {
                input,
                distance,
                divisor,
            } => (self.eval_in(*input, p, gap) - self.amount_in(distance, p, gap)) / divisor,
            Op::Ramp {
                input,
                in_min,
                in_span,
                out_min,
                out_span,
                shape,
            } => {
                let t = (self.eval_in(*input, p, gap) - in_min) / in_span;
                out_min + out_span * shape.apply(t)
            }
            // Unwound, then measured, then divided by how much the winding can stretch.
            // Unlike a reflection this is *not* an isometry, so the division is what keeps the
            // result an honest lower bound rather than an overstated distance.
            Op::Twist {
                input,
                radians_per_mm,
                stretch,
                axis,
                at,
            } => self.eval_in(*input, unwind(p, *radians_per_mm, *axis, *at), gap) / stretch,
            // Pushed away from the curve, then measured, then divided by how far the push can
            // stretch material — the same three steps a twist takes, with a curve in place of
            // an axis. See the `attract` module for which way it pushes and why.
            //
            // An empty spine has no nearest point and so no warp: the body passes through
            // untouched, which is the honest answer to being drawn toward nothing. Not an
            // error, because a curve can legitimately be a single point mid-edit.
            Op::Attract { input, spine, strength, falloff, stretch } => {
                let warped = match spine.nearest(p) {
                    Some(n) => crate::attract::warp_point(p, n.point, *strength, *falloff),
                    None => p,
                };
                self.eval_in(*input, warped, gap) / stretch
            }
            // The nearest of the instances. Each is a rigid motion, so no factor has to be put
            // back the way a transform's scale does, and the `min` is the union that used to be
            // a boolean per copy.
            // The instances whose cells could hold the nearest material, which for a body
            // that fits its own pitch is three of them however many there are. See
            // `Op::Pattern::reach` for why that is the whole answer and not a shortcut.
            Op::Pattern { input, inverse, fold, reach, excluded, left_out, radius, exponent, taper } => {
                let n = inverse.len() as isize;
                // **Some copies are not the block's to hold**: the ones the skip list names, and
                // the one standing on the body when the block does not hold the body (see
                // `Node::Pattern::join`). `excluded` says which, once, at compile.
                // **Clamped, for a row.** Past the last copy the cell index keeps counting, and
                // a window around a cell that does not exist finds nothing at all -- the field
                // came back infinite beyond the end of the row, which a test caught at once.
                // The nearest instance to a point past the end is the end one, so that is the
                // cell to look around. A turn has no end to be past, which is what `wraps` is.
                let cell = fold.cell(p);
                let cell = if fold.wraps() { cell } else { cell.clamp(0, n - 1) };
                let mut best = f64::INFINITY;
                // **The n-ary smooth minimum, accumulated, not a pairwise fold.**
                // `smooth_min` is `max(k, min aᵢ) − (Σ (k−aᵢ)₊ⁿ)^(1/n)`, and over more than two
                // operands the sum runs over all of them. Blending pairwise would be a
                // different answer *and* an order-dependent one — the form is not associative,
                // so which copy folded first would show wherever three of them meet. Two
                // running numbers say it exactly: the least value, and that sum.
                let mut inside = 0.0f64;
                let rounded = *radius > 0.0;
                // The radius this point gets, which is the whole radius unless a critical
                // surface is near. See [`tapered`].
                let k_eff = tapered(*radius, gap);
                // **A window as wide as the pattern walks the instances instead.** Folding is
                // an optimisation and stops being one here: past this width the offsets land on
                // the same instance more than once, which a minimum shrugs off and the sum
                // above does not — the copy would be counted twice and the surface pushed out.
                // The shader decides it from the same two numbers.
                let whole = 2 * *reach + 1 >= n as usize;
                let window: Vec<isize> = if let Fold::Grid { .. } = fold {
                    fold.grid_window(p)
                } else if whole {
                    (0..n).collect()
                } else {
                    (-(*reach as isize)..=(*reach as isize))
                        .map(|offset| cell + offset)
                        .map(|i| if fold.wraps() { i.rem_euclid(n) } else { i })
                        .collect()
                };
                for i in window {
                    if i < 0 || i >= n || excluded[i as usize] {
                        continue;
                    }
                    let here = self.eval_in(*input, inverse[i as usize] * p, gap);
                    best = best.min(here);
                    if rounded {
                        inside += (k_eff - here).max(0.0).powf(*exponent);
                    }
                }
                if rounded {
                    // **Divided by W^(1/n), W being the instances this window sums**
                    // (blend-integrity-spec §3, option A). With W copies inside `k` at once and
                    // their gradients parallel — coplanar tops of overlapping copies — the form
                    // above rises W^(1/n) per millimetre (Hölder), so undivided it claims more
                    // room than there is. A positive constant moves no zero crossing, so the
                    // surface and its fillets are exactly where they were. W is the window's
                    // size rather than a count of who is inside `k` at `p`: a divisor that
                    // varied with position would change the value's own rate and need its own
                    // proof (option B), and the window is the most that can ever be summed.
                    // Pessimistic on purpose — seven in the window costs √7 in tracer steps.
                    (k_eff.max(best) - inside.powf(1.0 / *exponent))
                        / pattern_join_divisor(*reach, inverse.len(), *left_out, *exponent, *taper)
                } else {
                    best
                }
            }
            // The reflected point, and nothing else. A reflection preserves distance, so there
            // is no factor to put back the way a transform's scale has to be.
            //
            // Joined, the answer is the nearer of the body and its reflection — which is a
            // union, done here rather than as a separate node so that one operation is one
            // block. The input is evaluated at two points, which the CPU does by simply asking
            // twice; shader generation expands the node instead, since it gives each node a
            // single point.
            Op::Mirror {
                input,
                at,
                normal,
                join,
                one_side,
                // Where the body is evaluated is this arm's business; what it claims is
                // `normalization_of`'s.
                reaches_its_plane: _,
            } => {
                // **One reading when the body is all on one side.** The `min` below is choosing
                // between the query's own half and the far one, and for a body that never
                // crosses the plane the query's own half always wins: for any body point `b`
                // and any `p` on the body's side,
                // `|p − mirror(b)|² − |p − b|² = 4 (p·n − c)(b·n − c) ≥ 0`, both factors
                // carrying the same sign. Measured over a grid on five bodies, two of them
                // understating, the two forms agree to the bit — `mirror_fold_probe`.
                match one_side.filter(|_| *join) {
                    Some(side) => {
                        let towards = if (p - *at).dot(normal) * side < 0.0 {
                            reflect(p, *at, *normal)
                        } else {
                            p
                        };
                        self.eval_in(*input, towards, gap)
                    }
                    None => {
                        let mirrored = self.eval_in(*input, reflect(p, *at, *normal), gap);
                        if *join {
                            mirrored.min(self.eval_in(*input, p, gap))
                        } else {
                            mirrored
                        }
                    }
                }
            }
            // Evaluate the inner field at the point this one came from, then scale the
            // distance back out. Dividing the argument by `s` and multiplying the result by
            // `s` is the pair that keeps the answer a true distance instead of the unscaled
            // body's — either half alone would be wrong.
            Op::Transform {
                input,
                rotation,
                translation,
                scale,
            } => *scale * self.eval_in(*input, to_local(p, rotation, translation, *scale), gap),
        }
    }

    /// The exact analytic normal at `p`, if every field contributing to the boundary there
    /// has one — mirrors [`Self::eval`]'s recursion, but instead of combining two values it
    /// picks which operand's value *defined* the result and delegates to that operand alone.
    ///
    /// That mirroring has to be exact: a boolean's gradient at a point is the gradient of
    /// whichever operand is currently active there, not some combination of both, so the
    /// winner here must be the same operand `eval` effectively used to produce the value.
    ///
    /// A blend mixes both fields continuously near the seam — there is no single "active"
    /// operand to delegate to — so this returns `None` there and lets the caller fall back to
    /// finite-differencing the whole field, exactly as it does for any leaf with no analytic
    /// normal of its own.
    fn normal_in(&self, index: usize, p: Point3<f64>, gap: f64) -> Option<Vector3<f64>> {
        match &self.ops[index] {
            Op::Leaf(field) => field.normal(p),
            // As a driven offset answers: the ramp's own gradient is part of this surface's
            // direction, and the finite difference the caller falls back to includes it.
            Op::Lattice { .. } => None,
            // Whichever operand defined the value here owns the normal: the nearest for a
            // union, the farthest for an intersection, and for a difference the kept body
            // unless a subtracted one's complement is farther — whose gradient is the
            // negation of its own outward normal, since the composed value there is `-d`.
            // Ties go to the earlier operand, as a pair's went to its first.
            Op::Boolean { op, inputs } => {
                let (first, rest) = inputs.split_first()?;
                let mut chosen = *first;
                let mut best = self.eval_in(*first, p, gap);
                let mut negated = false;
                for &i in rest {
                    let d = self.eval_in(i, p, gap);
                    match op {
                        BooleanOp::Union if d < best => {
                            best = d;
                            chosen = i;
                        }
                        BooleanOp::Intersection if d > best => {
                            best = d;
                            chosen = i;
                        }
                        BooleanOp::Difference if -d > best => {
                            best = -d;
                            chosen = i;
                            negated = true;
                        }
                        _ => {}
                    }
                }
                let normal = self.normal_in(chosen, p, gap)?;
                Some(if negated { -normal } else { normal })
            }
            Op::Blend { .. } => None,
            // Unchanged. Subtracting a constant does not change a gradient, which is exactly
            // why an offset preserves normalization and stays blendable.
            // A fixed offset moves the surface without turning it, so the normal is the
            // inner one exactly. A driven one does turn it — the true normal is
            // `∇F − ∇d` — and there is no analytic `∇d` here without differentiating the
            // ramp's whole input, so the honest answer is that there isn't one. `None` sends
            // the caller to a central difference, which is right rather than merely close.
            // **Analytic, and exactly the gradient rather than a plausible vector.** For a
            // constant `t` the gradient of `(1 − t)·A + t·B` is `(1 − t)·∇A + t·∇B`, so both
            // operands' own normals compose without differencing anything.
            //
            // The degenerate case is handled rather than assumed away: where the two gradients
            // oppose, the combination is near zero and normalizing it returns rounding noise —
            // the failure that cost the attractor two wrong versions. Below a threshold this
            // falls back to whichever operand is nearer its own surface, which is the one
            // actually deciding the value there.
            Op::Morph { a, b, t, .. } => {
                let Amount::Fixed(t) = t else { return None };
                let (na, nb) = (self.normal_in(*a, p, gap)?, self.normal_in(*b, p, gap)?);
                let mixed = na * (1.0 - *t) + nb * *t;
                match mixed.try_normalize(1e-6) {
                    Some(n) => Some(n),
                    None => match self.eval_in(*a, p, gap).abs() <= self.eval_in(*b, p, gap).abs() {
                        true => Some(na),
                        false => Some(nb),
                    },
                }
            }
            Op::Offset {
                input, distance, ..
            } => match distance {
                Amount::Fixed(_) => self.normal_in(*input, p, gap),
                Amount::Driven(_) => None,
            },
            // A ramp has no surface to have a normal to.
            Op::Ramp { .. } => None,
            // None, so the caller differences the composed field instead. The warp's Jacobian
            // would have to be applied to the inner normal, and it is not orthonormal — so the
            // rotated normal would not be the true one, only a plausible one. Differencing is
            // slower and right, and it is what a blend already falls back to.
            Op::Twist { .. } => None,
            // **Turned, not carried through, and not refused either.** A twist gives up here
            // and lets the caller difference the whole field; an attractor does not have to,
            // because its Jacobian has a closed form -- see `attract::warp_normal`. That is
            // worth the arm rather than the `None`: the shader cannot difference anything, so
            // whatever this returns is the rule the GPU has to match, and a `None` here left
            // the two sides free to disagree about which way a surface faced.
            Op::Attract { input, spine, strength, falloff, .. } => {
                let nearest = spine.nearest(p)?;
                let warped = crate::attract::warp_point(p, nearest.point, *strength, *falloff);
                let inner = self.normal_in(*input, warped, gap)?;
                Some(crate::attract::warp_normal(
                    p,
                    nearest.point,
                    nearest.along,
                    *strength,
                    *falloff,
                    inner,
                ))
            }
            // Whichever instance is nearest owns the normal, and it is carried back out
            // through that instance's own rotation — the same rule a union follows, over as
            // many operands as there are copies.
            // A rounded fold has no single instance that owns the answer -- the value near a
            // seam is a mix of several -- so there is no exact normal to carry out, and `None`
            // sends the caller to a central difference. Exactly what `Op::Blend` answers, and
            // for the same reason.
            Op::Pattern { radius, .. } if *radius > 0.0 => None,
            Op::Pattern { input, inverse, excluded, .. } => {
                let best = inverse
                    .iter()
                    .zip(excluded)
                    .filter(|(_, left_out)| !**left_out)
                    .map(|(placement, _)| placement)
                    .min_by(|a, b| {
                        self.eval_in(*input, *a * p, gap)
                            .total_cmp(&self.eval_in(*input, *b * p, gap))
                    })?;
                self.normal_in(*input, best * p, gap)
                    .map(|n| best.rotation.inverse() * n)
            }
            // Reflected back out, from whichever half is nearer. The reflection of a direction
            // is the direction minus twice its component along the normal, and the result is
            // still unit length because a reflection is an isometry.
            Op::Mirror {
                input,
                at,
                normal,
                join,
                one_side,
                reaches_its_plane: _,
            } => {
                // The same half the value picked, picked the same way — by side when the body
                // is all on one, by which reads nearer when it is not. Choosing differently
                // here would light a surface from the wrong reflection.
                if let Some(side) = one_side.filter(|_| *join) {
                    return if (p - *at).dot(normal) * side < 0.0 {
                        self.normal_in(*input, reflect(p, *at, *normal), gap)
                            .map(|n| n - normal * (2.0 * n.dot(normal)))
                    } else {
                        self.normal_in(*input, p, gap)
                    };
                }
                let q = reflect(p, *at, *normal);
                let mirrored = self.eval_in(*input, q, gap);
                if *join && self.eval_in(*input, p, gap) <= mirrored {
                    return self.normal_in(*input, p, gap);
                }
                self.normal_in(*input, q, gap)
                    .map(|n| n - normal * (2.0 * n.dot(normal)))
            }
            // Rotate the inner normal out; the scale does not appear. Dividing the argument
            // by `s` and multiplying the value by `s` cancel in the gradient, so the result
            // is already unit length whenever the inner normal was — which is what keeps a
            // scaled body shading correctly instead of subtly flattening.
            Op::Transform {
                input,
                rotation,
                translation,
                scale,
            } => self
                .normal_at(*input, to_local(p, rotation, translation, *scale))
                .map(|n| rotation * n),
        }
    }

    /// What the graph's magnitude means, from what its leaves' magnitudes mean.
    ///
    /// One lower-bound leaf anywhere below a node makes that node's value a lower bound too,
    /// since neither `min`, `max`, negation nor a blend can recover a distance from a value
    /// that never was one.
    ///
    /// Booleans over true distance fields stay [`Normalization::Distance`]. They do
    /// underestimate near a crease — `min` inside a concave join, `max` outside a convex
    /// one — but that is inherent to composing distance fields, is local and bounded, and long
    /// predates this distinction. Treating it as equivalent to a lattice's global 40% shortfall
    /// would refuse `blend(union(a, b), c)`, which is ordinary modelling that has always
    /// worked.
    fn normalization_at(&self, index: usize) -> Normalization {
        normalization_of(&self.ops, index)
    }

    /// The largest Lipschitz constant anywhere in the graph.
    ///
    /// Every operation here is built from `min`, `max`, negation and a smooth blend, and none
    /// of those can make a value change faster with position than its fastest operand already
    /// does. So the maximum over the leaves bounds the whole composition.
    ///
    /// This has to be an override rather than the trait's default of `1.0`: the moment one
    /// leaf is a lattice or any other pseudo-distance, a graph claiming to be 1-Lipschitz is
    /// understating how fast it moves, and every consumer that trusts the claim steps too far.
    fn lipschitz_at(&self, index: usize) -> f64 {
        lipschitz_of(&self.ops, index)
    }

    /// What a driven or fixed amount comes to at one point.
    fn amount_in(&self, amount: &Amount, p: Point3<f64>, gap: f64) -> f64 {
        match amount {
            Amount::Fixed(v) => *v,
            Amount::Driven(at) => self.eval_in(*at, p, gap),
        }
    }

    /// The range an amount can take over a box, which for a ramp is its output range.
    ///
    /// A ramp is clamped at both ends, so its output never leaves `[out_min, out_max]` whatever
    /// its input does. That is loose — the input over a small box may only reach part of the
    /// range — and it is loose *safely*, which is the only direction an interval may err.
    /// Bounding the input and mapping it would be tighter, but the shape is not monotonic in
    /// general once it is composed with an arbitrary field, so this stays the honest answer.
    fn amount_interval(&self, amount: &Amount) -> (f64, f64) {
        match amount {
            Amount::Fixed(v) => (*v, *v),
            Amount::Driven(at) => match &self.ops[*at] {
                Op::Ramp {
                    out_min, out_span, ..
                } => (
                    out_min.min(out_min + out_span),
                    out_min.max(out_min + out_span),
                ),
                // Unreachable: only a ramp is ever driven, refused at compile time otherwise.
                _ => (f64::NEG_INFINITY, f64::INFINITY),
            },
        }
    }

    /// How many ops this field evaluates, which is what pruning shortens.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// What a region holds: nothing at all, or a field shortened to what can matter inside it.
    ///
    /// **Backlog item 17, in this codebase's own terms.** The item describes the mechanism as
    /// Fidget's — interval evaluation, tracing and `Shape::simplify` — because that is where it
    /// was measured. But nothing compiles our graph into a Fidget tree: the one bridge that did
    /// was the M0 spike, since deleted, and `GraphField` is its own op list with its own
    /// [`Self::interval`]. Building item 17 on Fidget would mean a *third* compiler for one
    /// composition, which is the thing this codebase refuses everywhere else. So it is written
    /// against the op list we have, and the mechanism is unchanged: bound each op over the box,
    /// delete what cannot contribute, evaluate the rest.
    ///
    /// **Where the saving comes from.** Not from shortening a leaf — a pipe still walks its
    /// spine. From not evaluating the leaf *at all* where it cannot matter, which is why the gain
    /// grows with what a leaf costs. `fieldform-field/examples/brick_timing.rs` measures it at
    /// 3.0x to 9.4x on the clock, pruning pass included, and 82% to 95% of bricks rejected by one
    /// interval evaluation each.
    ///
    /// Unions and intersections are what get shortened, because they are what a wide graph is
    /// made of: a pattern, a part's bodies, a scatter. A blend is left alone — its operands
    /// contribute within the radius whatever their intervals say, and a bound that ignored that
    /// would delete material.
    pub fn for_region(&self, lo: Point3<f64>, hi: Point3<f64>) -> Region {
        let (low, high) = self.interval(lo, hi);
        if low > 0.0 {
            return Region::Empty;
        }
        if high < 0.0 {
            return Region::Solid;
        }
        let mut out = Vec::with_capacity(self.ops.len());
        let mut moved = std::collections::HashMap::new();
        let output = self.copy_pruned(self.output, lo, hi, &mut out, &mut moved);
        Region::Field(GraphField {
            ops: out,
            output,
            // A body index is a fact about the whole part, and this field is a window on to part
            // of it -- so the mapping is deliberately not carried across. Whoever wants to know
            // which body a point belongs to asks the field it was pruned from.
            bodies: Vec::new(),
            morph_divisors: self.morph_divisors.clone(),
            // The window is still a window on to a protected part.
            protect: self.protect.clone(),
        })
    }

    /// Copies one op and everything it still needs, dropping the operands a region rules out.
    fn copy_pruned(
        &self,
        index: usize,
        lo: Point3<f64>,
        hi: Point3<f64>,
        out: &mut Vec<Op>,
        moved: &mut std::collections::HashMap<usize, usize>,
    ) -> usize {
        if let Some(already) = moved.get(&index) {
            return *already;
        }
        let op = match &self.ops[index] {
            Op::Boolean { op, inputs } => {
                let ranges: Vec<(f64, f64)> =
                    inputs.iter().map(|i| self.interval_at(*i, lo, hi)).collect();
                // **What cannot be the answer anywhere in this box.** A union takes the least, so
                // an operand whose *smallest* value here is above another's largest can never be
                // that least -- and the same argument upside down for an intersection. A
                // difference is a max over the first and the negated rest, so only the first is
                // safe to reason about this way; it is left whole.
                let keep: Vec<usize> = match op {
                    fieldform_scene::BooleanOp::Union => {
                        let best = ranges.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
                        (0..inputs.len()).filter(|n| ranges[*n].0 <= best).collect()
                    }
                    fieldform_scene::BooleanOp::Intersection => {
                        let worst =
                            ranges.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
                        (0..inputs.len()).filter(|n| ranges[*n].1 >= worst).collect()
                    }
                    fieldform_scene::BooleanOp::Difference => (0..inputs.len()).collect(),
                };
                // One operand left is that operand: the boolean has nothing to combine.
                if keep.len() == 1 {
                    let only = self.copy_pruned(inputs[keep[0]], lo, hi, out, moved);
                    moved.insert(index, only);
                    return only;
                }
                let kept: Vec<usize> = keep
                    .iter()
                    .map(|n| self.copy_pruned(inputs[*n], lo, hi, out, moved))
                    .collect();
                Op::Boolean { op: *op, inputs: kept }
            }
            other => {
                let mut copy = other.clone();
                for input in copy.inputs_mut() {
                    *input = self.copy_pruned(*input, lo, hi, out, moved);
                }
                copy
            }
        };
        out.push(op);
        let at = out.len() - 1;
        moved.insert(index, at);
        at
    }

    /// Bounds the composed field over a box by bounding each operand and combining the
    /// intervals — mirroring [`Self::eval`]'s recursion the way [`Self::normal_at`] does.
    ///
    /// Composing is worth the recursion rather than falling back on the trait's centre-and-
    /// radius default, because the default's slack is set by the *whole* graph's Lipschitz
    /// constant. One fast-moving leaf would widen the bound on every other node in the graph,
    /// including nodes that node never touches. Bounding each operand under its own constant
    /// and combining keeps that slack local to where it was earned.
    fn interval_at(&self, index: usize, lo: Point3<f64>, hi: Point3<f64>) -> (f64, f64) {
        match &self.ops[index] {
            Op::Leaf(field) => field.interval(lo, hi),
            // One-Lipschitz by its divisor, so the centre's value and the box's half-diagonal
            // bound it. Looser than a leaf's own interval, and loose in the safe direction.
            Op::Lattice { .. } => {
                let centre = nalgebra::center(&lo, &hi);
                let reach = (hi - lo).norm() * 0.5;
                let v = self.eval_in(index, centre, self.gap_at(centre));
                (v - reach, v + reach)
            }
            Op::Boolean { op, inputs } => {
                let mut ranges = inputs.iter().map(|i| self.interval_at(*i, lo, hi));
                let first = ranges.next().unwrap_or((f64::INFINITY, f64::INFINITY));
                match op {
                    // min and max are monotonic in every argument, so they carry straight
                    // through to the endpoints.
                    BooleanOp::Union => ranges.fold(first, |(a_lo, a_hi), (b_lo, b_hi)| {
                        (a_lo.min(b_lo), a_hi.min(b_hi))
                    }),
                    BooleanOp::Intersection => {
                        ranges.fold(first, |(a_lo, a_hi), (b_lo, b_hi)| {
                            (a_lo.max(b_lo), a_hi.max(b_hi))
                        })
                    }
                    // Negating an operand swaps its endpoints, matching `eval`'s `kept.max(-d)`.
                    BooleanOp::Difference => ranges.fold(first, |(a_lo, a_hi), (b_lo, b_hi)| {
                        (a_lo.max(-b_hi), a_hi.max(-b_lo))
                    }),
                }
            }
            Op::Blend {
                op,
                a,
                b,
                radius,
                exponent,
                taper,
                divisor,
                second,
            } => {
                // **Bounded at the full radius, tapered or not.** The slack below grows with
                // `k` and the upper end does not depend on it, so the interval at `k` encloses
                // the blend at every `k_eff` in `[0, k]` — which is all a taper can choose.
                // A driven radius is bounded at the largest its ramp can give, for the same
                // reason: every radius the box sees is in `[0, k]` then too.
                let radius = &self.amount_interval(radius).1.max(0.0);
                let a_range = self.interval_at(*a, lo, hi);
                let b_range = self.interval_at(*b, lo, hi);
                // A two-legged chamfer's operands are scaled before the join, as `eval` scales
                // them: an interval scaled by a positive number is its ends scaled.
                let (sa, sb, m) = two_legs(1.0, 1.0, *radius, *second);
                let a_range = (a_range.0 * sa, a_range.1 * sa);
                let b_range = (b_range.0 * sb, b_range.1 * sb);
                let radius = &m;
                // Each blended boolean is the smooth minimum of possibly-negated operands,
                // exactly as `blend` evaluates it, so the bound is derived once and the signs
                // are applied around it. Negating an interval swaps its endpoints.
                let negate = |(lo, hi): (f64, f64)| (-hi, -lo);
                let (lo, hi) = match op {
                    BooleanOp::Union => smooth_min_interval(a_range, b_range, *radius, *exponent),
                    BooleanOp::Difference => negate(smooth_min_interval(
                        negate(a_range),
                        b_range,
                        *radius,
                        *exponent,
                    )),
                    BooleanOp::Intersection => negate(smooth_min_interval(
                        negate(a_range),
                        negate(b_range),
                        *radius,
                        *exponent,
                    )),
                };
                // What `blend` reports lies between the raw form and the raw form over
                // 2^(1/n), with the raw form's sign -- divided by a constant where the part
                // tapers, bounded by [`blend_bound`] where it does not. So each end is widened
                // to the hull of the two, which moves no sign: a positive end shrinks toward
                // nought at most, and so does a negative one.
                let least = 2f64.powf(1.0 / *exponent).max(blend_divisor(*radius, *exponent, *taper));
                let hull = |v: f64| (v.min(v / least), v.max(v / least));
                // Over the driven divisor as the value is: a positive constant, so the ends
                // shrink toward nought and no sign moves.
                let (lo, hi) = if *radius > 0.0 { (hull(lo).0, hull(hi).1) } else { (lo, hi) };
                (lo / divisor, hi / divisor)
            }
            // Shifted whole, endpoints and all: subtracting a constant is monotonic. A
            // driven distance is not constant, so the widest shift either way is taken — the
            // lower end loses the largest amount and the upper end the smallest.
            // Exact interval arithmetic on a bilinear expression: the hull of the four corner
            // combinations. Tighter than the default centre-and-radius bound, and pruning is
            // what an interval is for — one that is too wide prunes nothing and one that is too
            // narrow prunes away geometry that is there.
            Op::Morph {
                a,
                b,
                t,
                divisor,
            } => {
                let (a_lo, a_hi) = self.interval_at(*a, lo, hi);
                let (b_lo, b_hi) = self.interval_at(*b, lo, hi);
                let (t_lo, t_hi) = self.amount_interval(t);
                let (t_lo, t_hi) = (t_lo.clamp(0.0, 1.0), t_hi.clamp(0.0, 1.0));
                let corners = [
                    a_lo * (1.0 - t_lo) + b_lo * t_lo,
                    a_lo * (1.0 - t_hi) + b_lo * t_hi,
                    a_hi * (1.0 - t_lo) + b_hi * t_lo,
                    a_hi * (1.0 - t_hi) + b_hi * t_hi,
                ];
                let out_lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
                let out_hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (out_lo / divisor, out_hi / divisor)
            }
            Op::Offset {
                input,
                distance,
                divisor,
            } => {
                let (lo, hi) = self.interval_at(*input, lo, hi);
                let (d_lo, d_hi) = self.amount_interval(distance);
                ((lo - d_hi) / divisor, (hi - d_lo) / divisor)
            }
            // Clamped at both ends by construction, so its output never leaves its range.
            Op::Ramp {
                out_min, out_span, ..
            } => (
                out_min.min(out_min + out_span),
                out_min.max(out_min + out_span),
            ),
            // A twisted box is not a box. Bounded by the cylinder the winding sweeps the
            // region through — same height, and a radius that covers every corner — which is
            // conservative and axis-aligned, so the interval machinery can carry on.
            Op::Twist {
                input,
                stretch,
                axis,
                at,
                ..
            } => {
                // The cylinder the winding sweeps the box through, about whichever axis: the
                // same extent along it, and a radius across it that covers every corner.
                let across = |i: usize| (lo[i] - at[i]).abs().max((hi[i] - at[i]).abs());
                let (u, v) = match axis {
                    0 => (1, 2),
                    1 => (2, 0),
                    _ => (0, 1),
                };
                let r = across(u).hypot(across(v));
                let (mut a, mut b) = (lo, hi);
                a[u] = at[u] - r;
                b[u] = at[u] + r;
                a[v] = at[v] - r;
                b[v] = at[v] + r;
                let (i_lo, i_hi) = self.interval_at(*input, a, b);
                (i_lo / stretch, i_hi / stretch)
            }
            // The warped region, which is this box grown by the furthest the warp can move a
            // point — the same displacement bound the extent walk uses, asked over a box
            // instead of over a body. Axis-aligned by construction, so nothing downstream has
            // to learn a new shape.
            Op::Attract { input, strength, falloff, stretch, .. } => {
                let pad = crate::attract::max_pull(*strength, *falloff);
                let a = Point3::new(lo.x - pad, lo.y - pad, lo.z - pad);
                let b = Point3::new(hi.x + pad, hi.y + pad, hi.z + pad);
                let (i_lo, i_hi) = self.interval_at(*input, a, b);
                (i_lo / stretch, i_hi / stretch)
            }
            // Each instance's box, combined the way a union combines them. A rotated box is
            // not axis-aligned, so its corners are bounded rather than used directly.
            Op::Pattern { input, inverse, excluded, left_out, radius, reach, exponent, taper, .. } => {
                let (least_lo, least_hi) = inverse
                .iter()
                .zip(excluded)
                .filter(|(_, left_out)| !**left_out)
                .map(|(m, _)| m)
                .map(|m| {
                    let (mut min, mut max) = (
                        Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
                        Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
                    );
                    for i in 0..8 {
                        let corner = Point3::new(
                            if i & 1 == 0 { lo.x } else { hi.x },
                            if i & 2 == 0 { lo.y } else { hi.y },
                            if i & 4 == 0 { lo.z } else { hi.z },
                        );
                        let q = m * corner;
                        min = Point3::new(min.x.min(q.x), min.y.min(q.y), min.z.min(q.z));
                        max = Point3::new(max.x.max(q.x), max.y.max(q.y), max.z.max(q.z));
                    }
                    self.interval_at(*input, min, max)
                })
                .fold(
                    (f64::INFINITY, f64::INFINITY),
                    |(a_lo, a_hi), (b_lo, b_hi)| (a_lo.min(b_lo), a_hi.min(b_hi)),
                );
                // A rounded fold sits at most one radius below the hard one and never above
                // it, so the lower end drops by the radius and the upper end does not move.
                // Loose on purpose: an interval that is too narrow prunes away geometry that
                // is there, which is the one failure an interval must not have.
                if *radius > 0.0 {
                    // Both ends by the divisor `eval` applies: positive, so the order holds.
                    let divisor =
                        pattern_join_divisor(*reach, inverse.len(), *left_out, *exponent, *taper);
                    ((least_lo - radius) / divisor, least_hi / divisor)
                } else {
                    (least_lo, least_hi)
                }
            }
            // A box reflected in a general plane is not axis-aligned, so the reflected corners
            // are bounded rather than used directly — the same thing the transform arm does for
            // a rotated box. Joined, the answer covers both halves.
            Op::Mirror {
                input,
                at,
                normal,
                join,
                // An interval is taken over a *box*, which can straddle the plane whatever the
                // body does, so both halves are still bounded here.
                one_side: _,
                reaches_its_plane: _,
            } => {
                let (mut min, mut max) = (
                    Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
                    Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
                );
                for i in 0..8 {
                    let corner = Point3::new(
                        if i & 1 == 0 { lo.x } else { hi.x },
                        if i & 2 == 0 { lo.y } else { hi.y },
                        if i & 4 == 0 { lo.z } else { hi.z },
                    );
                    let q = reflect(corner, *at, *normal);
                    min = Point3::new(min.x.min(q.x), min.y.min(q.y), min.z.min(q.z));
                    max = Point3::new(max.x.max(q.x), max.y.max(q.y), max.z.max(q.z));
                }
                let reflected = self.interval_at(*input, min, max);
                if *join {
                    let here = self.interval_at(*input, lo, hi);
                    (reflected.0.min(here.0), reflected.1.min(here.1))
                } else {
                    reflected
                }
            }
            // A rotated box is not axis-aligned, so the inner field is bounded over the
            // axis-aligned box enclosing the rotated one — the bounds of the eight mapped
            // corners. That encloses the true image rather than matching it, so the interval
            // is wider than it strictly needs to be, which is the safe direction: the contract
            // asks the interval to *contain* every value the field takes over the box, not to
            // be tight. Scaling by a positive `s` is monotonic, so the endpoints keep order.
            Op::Transform {
                input,
                rotation,
                translation,
                scale,
            } => {
                let mut l = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut h = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for i in 0..8 {
                    let corner = Point3::new(
                        if i & 1 == 0 { lo.x } else { hi.x },
                        if i & 2 == 0 { lo.y } else { hi.y },
                        if i & 4 == 0 { lo.z } else { hi.z },
                    );
                    let q = to_local(corner, rotation, translation, *scale);
                    for axis in 0..3 {
                        l[axis] = l[axis].min(q[axis]);
                        h[axis] = h[axis].max(q[axis]);
                    }
                }
                let (inner_lo, inner_hi) = self.interval_at(*input, l, h);
                (*scale * inner_lo, *scale * inner_hi)
            }
        }
    }

    /// Which body's surface passes through `p`, if any is close enough to be the one meant.
    ///
    /// A point on the composed surface lies on exactly one body's surface — the others are
    /// elsewhere — so the body whose own field reads nearest zero there is the one that was
    /// clicked. Nothing is returned when none is within `tolerance`, since a click that landed
    /// on nothing should select nothing rather than the least distant thing on screen.
    ///
    /// A body that a boolean has cut away still reads zero along the cut, which is correct:
    /// the surface there *is* that body's, and it is what a user pointing at it means.
    pub fn body_at(&self, p: Point3<f64>, tolerance: f64) -> Option<fieldform_scene::NodeId> {
        self.bodies
            .iter()
            .map(|(id, op)| (*id, self.eval(*op, p).abs()))
            .filter(|(_, d)| *d <= tolerance)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id)
    }

    /// Number of compiled steps, for diagnostics.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl ScalarField for GraphField {
    fn distance(&self, p: Point3<f64>) -> f64 {
        self.eval(self.output, p)
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        self.normal_at(self.output, p)
    }

    fn normalization(&self) -> Normalization {
        self.normalization_at(self.output)
    }

    /// See [`Self::for_region`], which is where the work is; this is how a caller that holds a
    /// `dyn ScalarField` gets at it without knowing what it holds.
    fn pruned_to(
        &self,
        lo: Point3<f64>,
        hi: Point3<f64>,
    ) -> Option<std::sync::Arc<dyn ScalarField + Send + Sync>> {
        match self.for_region(lo, hi) {
            Region::Field(field) => Some(std::sync::Arc::new(field)),
            // Uniform: the caller has already decided what to do with a region holding no
            // surface, and handing it a field to sample would be inviting it to sample one.
            Region::Empty | Region::Solid => None,
        }
    }

    fn lipschitz(&self) -> f64 {
        self.lipschitz_at(self.output)
    }

    fn interval(&self, lo: Point3<f64>, hi: Point3<f64>) -> (f64, f64) {
        self.interval_at(self.output, lo, hi)
    }

    /// The smallest any leaf declares. A boolean or a blend can only cut a leaf's features
    /// thinner where it trims them, which the mesher's own floor already bounds; what it cannot
    /// do is make a wall thinner everywhere, which is the case this exists for.
    fn feature_size(&self) -> Option<f64> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                Op::Leaf(leaf) => leaf.feature_size(),
                Op::Lattice { lattice, .. } => lattice.feature_size(),
                _ => None,
            })
            .min_by(f64::total_cmp)
    }
}

/// A point turned back about the twist's axis by its own rate, so the inner field sees where
/// the material came from.
///
/// The *inverse* of the warp, which is what evaluating a domain-warped field means: the value at
/// `p` is the inner field's value wherever `p` was carried from.
///
/// Done in a frame where the chosen axis is the third coordinate, so one piece of arithmetic
/// serves all three. The frames are the cyclic permutations, which are rotations rather than
/// reflections — a swap would turn the body inside out on two of the three axes, and that is
/// the kind of mistake that looks like a modelling error rather than a sign error.
fn unwind(p: Point3<f64>, radians_per_mm: f64, axis: usize, at: Point3<f64>) -> Point3<f64> {
    let d = p - at.coords;
    // Into the axis's frame: the chosen axis becomes the third coordinate.
    let local = match axis {
        0 => Point3::new(d.y, d.z, d.x),
        1 => Point3::new(d.z, d.x, d.y),
        _ => Point3::new(d.x, d.y, d.z),
    };
    let angle = -radians_per_mm * local.z;
    let (s, c) = angle.sin_cos();
    let turned = Point3::new(
        local.x * c - local.y * s,
        local.x * s + local.y * c,
        local.z,
    );
    // And back out of it.
    let world = match axis {
        0 => Point3::new(turned.z, turned.x, turned.y),
        1 => Point3::new(turned.y, turned.z, turned.x),
        _ => turned,
    };
    world + at.coords
}

/// A point reflected in the plane through `at` with unit `normal`.
///
/// Its own function because four arms need it and they must agree exactly: the value, the
/// normal, the interval and the extent all reflect the same way, and a sign that differs
/// between them puts the surface and its shading in different places.
fn reflect(p: Point3<f64>, at: Point3<f64>, normal: Vector3<f64>) -> Point3<f64> {
    p - normal * (2.0 * (p - at).dot(&normal))
}

/// A block's rims, resolved: the reach as a number and the shape as its exponent. A reach that
/// names a parameter without a value is refused as every other unresolved number is.
pub(crate) fn resolved_rim(
    rims: &fieldform_scene::Rims,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<super::Rim, CompileError> {
    let reach = resolve(&rims.radius).ok_or(CompileError::UnresolvedRadius)?;
    // A second leg is a chamfer's alone (`Fillet::second_leg`): one left on rims switched back to
    // a round is not read, so switching the shape never moves the seam.
    let second = match rims.fillet().second_leg() {
        Some(s) => Some(resolve(s).ok_or(CompileError::UnresolvedRadius)?.max(0.0)),
        None => None,
    };
    Ok(super::Rim {
        reach: reach.max(0.0),
        exponent: rims.exponent(),
        ends: rims.at,
        square: None,
        second,
    })
}

/// The further of a rim's two legs, which is how far the rounding reaches into either face and
/// what a room check against a solid's depth compares with.
pub(crate) fn rim_second_leg(rim: &super::Rim) -> f64 {
    rim.legs().1
}

/// A box's or a cylinder's edge, resolved: `None` for a hard one, which keeps the plain leaf.
pub(crate) fn resolved_edge(
    edge: &fieldform_scene::Fillet,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<Option<crate::Edge>, CompileError> {
    // A box's, a cylinder's or a cone's faces have no first and second, so its edges take one
    // number only (`round-chamfer-spec.md` §9.2).
    if edge.second_leg().is_some() {
        return Err(CompileError::SecondLegOnAnEdge);
    }
    let reach = resolve(&edge.radius).ok_or(CompileError::UnresolvedRadius)?;
    Ok((reach > 0.0).then(|| crate::Edge { reach, exponent: edge.exponent() }))
}

/// Refuses an edge cut back further than `room` -- the spec's `r > min(b)`, so a reach of exactly
/// the half-side is allowed, the face shrinking to a line.
pub(crate) fn check_edge(
    reach: f64,
    room: f64,
    what: &'static str,
    block: &'static str,
) -> Result<(), CompileError> {
    if reach > room {
        return Err(CompileError::EdgeTooLarge { reach, room: room.max(0.0), what, block });
    }
    Ok(())
}

/// Refuses a rim cut back further than `room`, naming what ran out.
pub(crate) fn check_rim(
    reach: f64,
    room: f64,
    what: &'static str,
    bigger: &'static str,
) -> Result<(), CompileError> {
    if reach >= room {
        return Err(CompileError::RimTooLarge { reach, room: room.max(0.0), what, bigger });
    }
    Ok(())
}
