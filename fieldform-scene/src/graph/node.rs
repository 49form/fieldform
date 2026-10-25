// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! One operation in the graph: every kind of node, and what each reads, consumes and holds.

use super::{
    Attractor, BooleanOp, CellMap, Fillet, LatticeCell, LatticePhase, ParamRef, PatternKind,
    PatternOptions, RampShape, Rims, Turn, UnknownNode,
};
use crate::ends::Ends;
use crate::id::{CageId, CurveId, MeshId, NodeId};
use crate::graph::Operand;
use crate::Frame;
use crate::{Corners, LoftSmooth, Outline, Section, SweepLaw};
#[cfg(doc)] use super::{pattern_count, CountFit, Graph};

/// One operation in the graph.
///
/// Serialized as a JSON object tagged by `kind`. Anything this build cannot fully account for
/// becomes [`Node::Unknown`] rather than an error — see that variant.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// The limit surface of a cage, as a field.
    Cage(CageId),
    /// An imported triangle mesh, as a field.
    ///
    /// Beside [`Node::Cage`] rather than folded into it, even though both become the same
    /// kind of field underneath. What they are is different: a cage is scaffolding this
    /// document owns and edits, and a mesh is finished geometry it references. Only one of
    /// them has control points to drag.
    Mesh(MeshId),
    /// A sphere, by centre and radius.
    ///
    /// The centre is three parameter references rather than three numbers for the same reason
    /// the radius is one: a position that a named parameter can drive is the difference
    /// between a document that re-evaluates and one that has to be re-entered. It also keeps
    /// every number in the graph out of the generated shader, which depends on structure alone
    /// so that changing a value costs a buffer write rather than a recompile.
    Sphere { centre: crate::Place, radius: ParamRef },
    /// An axis-aligned box, by centre and the length of each side.
    ///
    /// Three extents rather than one because the field always computed per-axis anyway — a
    /// cube was never a simpler shape here, only a narrower way of asking for it. A cube is
    /// this with three equal sides, and the Create menu still offers one under that name.
    ///
    /// Written as `"cuboid"`. A `"cube"` in an older file reads back into this with its one
    /// size in all three slots; nothing writes that kind any more. A newer file opened by a
    /// build that predates this reports a kind it cannot account for and refuses, rather than
    /// reading a 40 x 10 x 10 bar as a 40 mm cube — which is the whole of why D3 refuses
    /// instead of guessing.
    ///
    /// `edge` rounds or chamfers every edge alike (`round-chamfer-spec.md` §3): the same
    /// [`Fillet`] a blend takes (D9), its number the reach -- how far back from an edge each face
    /// starts to turn. Hard is what every one was. Named `edge`, as the spec names it, and not
    /// `edges`, which is the skewed box's three edge vectors.
    Cuboid { centre: crate::Place, size: [ParamRef; 3], turn: Turn, edge: Fillet },
    /// A box whose faces are not square to one another, by its centre and its three edge vectors.
    ///
    /// **Three vectors rather than a box and a shear**, because the vectors *are* the shape: the
    /// first edge along X, the second leaning, the third leaning further gives any parallelepiped,
    /// and a box that is merely sheared is the case where two of the three stay on their axes.
    /// There is no `turn` field: a turn is already expressible by turning the vectors, and a stored
    /// turn beside them would describe the orientation twice (the rule `Cuboid` follows for its own
    /// reason). Each vector is a *full edge*, not a half, as a cuboid's size is a full side.
    ///
    /// Exact, which is the point of its being a node of its own — `Transform` is rigid plus uniform
    /// scale, and shearing a `Cuboid` would make its value a lower bound (`SkewedBox` in
    /// `fieldform-field`). Three edges in one plane make no box and are refused at compile, naming
    /// the number to change.
    SkewedBox { centre: crate::Place, edges: [[ParamRef; 3]; 3] },
    /// A cylinder between two points.
    ///
    /// The three shapes below are defined by *where their ends are* rather than by a centre
    /// and an extent, because that is how one gets placed: you know where it starts and where
    /// it stops. The direction falls out of the two points, so there is no separate axis field
    /// that could fall out of step with them — the same reason a curve stores its control
    /// points rather than a position and a heading. A [`Node::Transform`] still places the
    /// whole thing, exactly as it does for a cage or a curve.
    ///
    /// All three say it with an [`Ends`], which is what lets those two points be *a line in the
    /// document* rather than six typed numbers: draw where the strut goes, aim a cylinder at it,
    /// and dragging the line moves the cylinder. They held two triples each before, in two
    /// different spellings, and could not be pointed at anything.
    ///
    /// `edge` rounds or chamfers both circular edges alike (`round-chamfer-spec.md` §3): the same
    /// [`Fillet`] a blend takes (D9), its number the reach -- how far back from an edge each face
    /// starts to turn. Hard is what every one was. Named `edge`, as the spec names it, and not
    /// `edges`, which is the skewed box's three edge vectors.
    Cylinder { ends: Ends, radius: ParamRef, edge: Fillet },
    /// A cone between two points, with a radius at each and flat ends.
    ///
    /// One radius at zero gives the plain cone the word usually means; two equal radii give a
    /// cylinder. One kind rather than three, because they are one shape with different numbers
    /// in it.
    ///
    /// `edge` rounds or chamfers its circular edges alike, and its apex where a radius is nought
    /// (`round-chamfer-spec.md` §3.2): the same [`Fillet`] a box's edges take (D9). Hard is what
    /// every cone was.
    Cone {
        ends: Ends,
        base_radius: ParamRef,
        top_radius: ParamRef,
        edge: Fillet,
    },
    /// A lattice: a repeating cell, tiled through space, as a field.
    ///
    /// Infinite by construction, like [`Node::Plane`] and for the same reason it is fine —
    /// a lattice becomes a part by being intersected with one. **Boolean is the exact
    /// composition** — it depends only on sign, and the field is `LowerBound`. A blend over it
    /// mixes by magnitude against a field that reads short, so the seam sits somewhat toward the
    /// lattice; that was refused at graph-compile time until D8 was reversed on 2026-09-02, and
    /// is now allowed and reported by `approximate_blends`.
    ///
    /// Three independent choices, and keeping them apart is what makes the catalogue multiply
    /// rather than add: [`LatticeCell`] is *what the cell is*, [`CellMap`] is *how cells tile
    /// space*, and [`LatticePhase`] is *which side of the surface is material*. Swapping any one
    /// of them is a swap of one field rather than a different node kind.
    Lattice {
        cell: LatticeCell,
        /// Where the cells are measured from and which way they point. See [`Frame`].
        frame: Frame,
        /// How the cells tile space, and how big they are. See [`CellMap`] for why the size
        /// lives here rather than beside the phase's own number.
        map: CellMap,
        /// Sheet or solid, and the one number that sizes it. See [`LatticePhase`] for why the
        /// number is in there rather than beside this field.
        phase: LatticePhase,
    },
    /// A planar outline given thickness along its own normal.
    ///
    /// The first block that turns a *curve* into a solid rather than sweeping something along
    /// one. A profile must be closed, because a field's sign is a statement about being inside
    /// and an open curve has no inside — the same rule that stops an open mesh becoming a
    /// field, one dimension down. What an open curve can be given instead is a thickness of
    /// its own, across itself, and then it is a wall: [`Outline`] is the choice between the
    /// two, and a thin extrude is this block reading the second.
    Extrude {
        /// What is thickened: a closed profile, or a curve given a thickness. Read in **its own
        /// plane**, fitted to its points — which is why there is no plane stored here to
        /// disagree with it.
        outline: Outline,
        /// Which way it is swept. `Normal` is exact; anything else is a shear and says so.
        sweep: crate::Sweep,
        /// How far along the sweep from the profile's plane the material begins, in
        /// millimetres. Nought is the plane itself, which is what every extrude written before
        /// this existed means, so it is written only when it is not nought.
        ///
        /// A translation along the sweep and nothing else: the start plane is where the solid
        /// begins and so the base a draft is measured from, but the profile stays where it was
        /// drawn and the section is read there.
        start: ParamRef,
        /// The whole depth of side one, measured along the sweep from the start plane. Signed:
        /// a negative depth runs the other way out of the plane.
        height: ParamRef,
        /// Which way the depth lies about the plane: one side, half each way, or two sides
        /// with a depth and a draft of their own. See [`crate::Sides`], which says why two
        /// sides are one block and not two extrudes and a union.
        ///
        /// Was a `symmetric: bool`, which is still what the file says for the middle case:
        /// `"symmetric": true` is written for `Symmetric` and nothing for `One`, so nothing
        /// saved before has to be migrated, and `Two` writes `height_2` and `draft_2`, which a
        /// build before this met as unknown and refused.
        sides: crate::Sides,
        /// How far the walls lean out, in **degrees**, as they run along the sweep.
        ///
        /// A draft angle, which is what a moulding drawing carries and so what somebody making a
        /// part will already have in mind — rather than a scale at the far end, which is the same
        /// number in a form nobody quotes. The depth runs out of one side of the plane, so the
        /// profile as drawn is the *base* of the frustum: a positive angle opens away from it,
        /// a negative one closes in, and neither does anything on the other side.
        ///
        /// Zero is a prism, and costs nothing — the field checks for it and keeps the exact
        /// corner it can only take while the walls are square.
        draft: ParamRef,
        /// Whether that draft rounds the section's corners or mitres them. See [`Corners`]:
        /// a draft is an offset, an offset rounds a convex corner by its own amount, and a
        /// moulding drawing usually means the walls to meet at an edge instead. Both ship.
        ///
        /// On the block rather than beside each draft, because an extrude has two of them and
        /// a part is moulded one way. Nothing when there is no draft.
        corners: Corners,
        /// How the rims are rounded where the side wall meets its ends -- both, or the start or
        /// the end alone -- as a fillet of the reach typed or a chamfer of that leg
        /// (`round-chamfer-spec.md` §4). Hard is what every extrude was.
        ///
        /// **A [`Fillet`]'s shape and number, as a blend and a pattern take them (D9)**, with
        /// which ends beside them ([`Rims`]): the same shapes, the same number meaning the same
        /// thing, the same editor. Named `rims` and not the spec's
        /// `ends`, because `ends` already means two other things here -- a cylinder's two points
        /// ([`crate::Ends`]) and how a wall stops ([`crate::WallEnds`]) -- and the console's
        /// keys are field names.
        rims: Rims,
    },
    /// A section swept about the frame's axis: a closed profile, or a curve given a thickness.
    ///
    /// The same section an [`Node::Extrude`] reads, asked in `(r, z)` instead — which is why the
    /// two were built together. No planarity check and no roll: a revolve is symmetric about
    /// its axis, so which half-plane the section was drawn in cannot matter.
    ///
    /// A wall here is a thin revolve: the shell a curve turns out, a vase from its silhouette.
    Revolve {
        /// What is turned. See [`Outline`], which is the type an extrude and a sweep take too.
        outline: Outline,
        /// The axis it turns about, and a point on it.
        frame: Frame,
        /// How far round it turns, in **degrees**, side one. Three hundred and sixty is all the
        /// way round, which is what every revolve was and what a document that says nothing
        /// means, so it is written only when it is less.
        ///
        /// Measured from **the half-plane the profile lies in** -- the side of the axis it sits
        /// on -- positive by the right-hand rule about the frame's axis, and not from a world
        /// direction: a profile drawn anywhere starts its revolve where it is drawn, for the
        /// reason `axis_beside` exists.
        angle: ParamRef,
        /// Which way the angle lies about that half-plane: one way, half each way, or two ways
        /// with an angle of their own. See [`crate::AngleSides`], a sibling of an extrude's
        /// `Sides` -- side two of a revolve is an angle and has no draft.
        sides: crate::AngleSides,
        /// How the rims are cut where the turned side meets each flat end of a partial revolve
        /// -- the same [`Rims`] an extrude takes (D9), its start the cap the angle is measured
        /// from and its end the cap it reaches. A full revolve has no ends, and the compile
        /// refuses a rim on one by name rather than rounding nothing.
        rims: Rims,
    },
    /// A half-space: everything below the plane through `origin`, facing +Z.
    ///
    /// The one leaf with no bounds of its own, which is what it waited on. It is worth having
    /// anyway because it is almost never a body: a half-space is the standard trimming
    /// primitive, and intersecting a finite part with an infinite plane gives a finite answer.
    ///
    /// Faces +Z and is turned by a placement rather than storing a normal. Unlike the solids
    /// drawn between two points, a plane has no second point to read a direction from, so the
    /// choice is between a stored normal and a placement — and a stored normal *beside* a
    /// placement is the orientation-described-twice problem exactly.
    /// `flipped` swaps which side is solid. A bool rather than a rotation because it is not
    /// an orientation: turning a plane over is the same plane facing the other way, and
    /// expressing it as 180° in a placement would fight the manipulator for the same control.
    Plane { origin: crate::Place, flipped: bool, turn: Turn },
    /// A single point, as a field: the distance to it.
    ///
    /// Encloses no volume, so as a body it draws nothing — which is not a defect. Its use is
    /// as something to measure from: the distance-to-a-point a falloff is built on, and the
    /// seed a scatter or a lattice is placed against. Distinct from a zero-radius sphere
    /// because the two mean different things: that is a solid that happens to be empty, this
    /// is a source that was never meant to be solid.
    Point { at: [ParamRef; 3] },
    /// A torus lying in the XY plane, about +Z, and turned by a placement.
    ///
    /// The one round solid still defined about an axis rather than between two points: a
    /// torus has no two ends to name.
    Torus {
        centre: crate::Place,
        major_radius: ParamRef,
        minor_radius: ParamRef,
        turn: Turn,
    },
    /// A capsule between two points, with a radius at each.
    ///
    /// The points are the *cap centres*, so the shape reaches a radius beyond each of them and
    /// growing a radius fattens it without moving where it starts. Spherical ends, which is
    /// the whole difference from [`Node::Cone`].
    Capsule {
        ends: Ends,
        start_radius: ParamRef,
        end_radius: ParamRef,
    },
    /// A curve swept into a round tube of the given radius — the field generator.
    ///
    /// The radius is a [`ParamRef`] rather than a bare number because that is the whole point
    /// of the parametric layer: a pipe whose radius is a named parameter re-evaluates when the
    /// parameter changes, without anything being replayed or re-edited.
    ///
    /// It references a **curve**, not a face or an edge, which keeps it inside the boundary
    /// rule that makes topological naming never arise.
    Pipe { curve: CurveId, radius: ParamRef },
    /// A section carried along a curve, held square to it at every point: a closed profile, or
    /// a curve given a thickness — which is how an open profile is swept at all.
    ///
    /// The general case of [`Node::Pipe`], which sweeps a circle and so never has to ask which
    /// way is up. Any other profile has to be oriented continuously along the spine, and that
    /// orientation is a rotation-minimising frame computed from the spine — never stored, so
    /// there is no second description of it to disagree with the curve.
    /// the sweep-spec design note.
    ///
    /// **The section is carried by the point the path sets off from**, dropped onto the
    /// section's plane, and turns about that point as the frame turns
    /// (`fieldform_field::sweep_outline_section`, ruled 2026-10-02). A path that sets off below
    /// the plane carries the solid up to start at the section (`sweep_offset`). It was the
    /// section's centroid that rode the path before, and a path drawn from a corner of a large
    /// outline swung the outline clear of the curve.
    ///
    /// `section` says how the profile is *held* on the way: square to the path, flush with its
    /// own plane at the start and tilted from there, or parallel to itself throughout. See
    /// [`Section`]. `law` says how it changes on the way — taper, twist, and the stretch of
    /// path it runs over; see [`SweepLaw`]. Plain for every sweep that has not asked.
    ///
    /// A straight spine is an extrude and should be one: this always reports a lower bound, and
    /// an extrude along its normal is exact.
    ///
    /// `rims` rounds or chamfers where the side wall meets each end of an open path, as an
    /// extrude's do; a closed path has no ends, and a rim on one is refused by name.
    Sweep { outline: Outline, spine: CurveId, section: Section, law: SweepLaw, rims: Rims },
    /// Two or more closed sections interpolated along a centerline: a solid that passes
    /// through each and changes smoothly between them. the loft-spec design note.
    ///
    /// **A centerline is not optional**, because a loft is interpolation and interpolating two
    /// distance fields shrinks a shape rather than moving it where the two do not overlap
    /// (`morph-spec.md` §2). Every section is read relative to the centerline at the query's own
    /// station, which keeps them overlapping by construction. `spine` names one; `None` builds
    /// a cubic from the first section's centroid to the last's at compile, never stored
    /// (`fieldform_field::derived_centerline`), so there is no second description of where the
    /// sections are to drift.
    ///
    /// `sections` are listed in the order the solid passes through them, and each must station
    /// after the last along the centerline or the block is refused naming the two that crossed.
    /// Closed outlines only in V1; a wall as a section is refused by name.
    ///
    /// `rims` rounds or chamfers where the side meets the first and the last section, as an
    /// extrude's do -- the ends ruled on 2026-10-03 to arrive with `round-chamfer-spec.md` part B,
    /// named `rims` for the reason the extrude's field gives.
    Loft { sections: Vec<Outline>, spine: Option<CurveId>, smooth: LoftSmooth, rims: Rims },
    /// Smooth blend of two fields — the operation a mesh boolean cannot do without leaving a
    /// crease, and the reason for the whole field representation.
    ///
    /// The same three operations a [`Node::Boolean`] has, with a fillet of `radius` along the
    /// join instead of an edge. A blend and a boolean stay separate node kinds because they
    /// demand different things of their operands: a boolean's surface depends only on sign,
    /// while a blend mixes by magnitude, so a blend requires true distance fields and a
    /// boolean does not.
    Blend {
        op: BooleanOp,
        a: Operand,
        b: Operand,
        /// How the seam is rounded. See [`Fillet`] — the same type a pattern's fold takes, so
        /// the number means the same thing in both and there is one editor for it.
        fillet: Fillet,
    },
    /// Hard boolean over a list of operands.
    ///
    /// A list rather than a pair, for two reasons that turn out to be one. A person combining
    /// five bodies means one act — "these, joined" — and a nest of four pairs is four blocks,
    /// each hiding an operand inside another, which is not what was asked and cannot be edited
    /// as what was asked. And a nest evaluates as deep as it is long: the part's own union of
    /// its bodies met that at ten thousand solids and overflowed the stack, and was made a list
    /// for the same reason.
    ///
    /// `Union` and `Intersection` treat every operand alike. `Difference` keeps the first and
    /// subtracts each of the rest, so order matters there and nowhere else. One operand is that
    /// operand; none is refused at compile, since there is nothing to combine.
    Boolean {
        op: BooleanOp,
        inputs: Vec<Operand>,
    },
    /// Two bodies averaged into a third — one becoming the other.
    ///
    /// **Not a boolean and not a blend.** Those keep both operands and differ only in how the
    /// seam between them is drawn; a morph at `t = 0.4` contains neither shape. It is the
    /// convex combination `(1 − t)·A + t·B` of the two fields themselves.
    ///
    /// **What it is good for, which is not what it sounds like.** A lerp of two distance fields
    /// does *not* slide a shape from one place to another. Two spheres of radius `R` separated
    /// by `D` interpolate at `t = ½` to an ellipsoid whose major axis is `2R` — which does not
    /// exist once `D > 2R`, so the shape shrinks to nothing and reappears at the far end. A
    /// constant `t` therefore wants the two bodies in the same place, becoming one another. See
    /// `morph-spec.md` §2, which proves it rather than asserting it.
    ///
    /// The value costs nothing to keep honest: every field here already divides by its own
    /// Lipschitz bound, so a convex combination of two of them is 1-Lipschitz for free and
    /// there is no divisor. That is what makes dragging `t` a buffer write with no shader
    /// rebuild.
    Morph {
        a: Operand,
        b: Operand,
        /// 0 is `a` exactly and 1 is `b` exactly, clamped in between.
        t: ParamRef,
    },
    /// Another field, wound about the frame's own Z axis: the further along the axis, the more
    /// it is turned.
    ///
    /// The first **domain warp** here — an operation that changes where the field is asked
    /// rather than what it answers — and the first one that is not an isometry. A rotation that
    /// varies with height stretches material sideways, by more the further it sits from the
    /// axis, so it does *not* preserve distance and cannot pretend to.
    ///
    /// That has a consequence worth stating where it will be read: a twisted body reports
    /// [`crate::Normalization::LowerBound`], so **it can be booleaned and cannot be blended**.
    /// A blend mixes by magnitude, and a field that understates itself would pull the seam
    /// toward itself — a fillet that is not the radius asked for. The graph refuses it at
    /// compile time rather than producing it.
    Twist {
        input: Operand,
        /// How far the body turns per millimetre travelled along the axis, in degrees.
        ///
        /// Per millimetre rather than "over the whole body", because a body's height is not
        /// something this node knows: it wraps a field, and a field has no ends. The panel
        /// converts, since the number a person wants is usually the total.
        degrees_per_mm: ParamRef,
        /// The axis it winds about, and a point that axis passes through.
        ///
        /// Only the two components of the origin *across* the axis matter; the third slides
        /// along it and changes nothing.
        frame: Frame,
    },
    /// Another field, drawn toward a curve — by more the nearer it already is, and not at all
    /// beyond `falloff`.
    ///
    /// The second **domain warp**, after [`Node::Twist`], and it is worth reading the two
    /// together: both change *where* the field is asked rather than what it answers, and both
    /// pay for that in the same currency. A twist winds about an axis; this one pushes the
    /// query point away from a curve, which is what draws the *material* toward it.
    ///
    /// **The sample point moves away and the geometry moves toward, and that is not a typo.**
    /// `g` is the inverse of the visible displacement: to know what the pulled shape looks like
    /// at `p`, ask what the original looked like at the point that got pulled in to occupy `p`
    /// — which is further out. Every SDF deformation is built this way (a twist here already
    /// is: it *un*winds), and getting the sign backwards fails silently, because the result is
    /// still a deformation and still looks deliberate. `a_sphere_leans_toward_the_curve_not_away`
    /// is the test that holds it.
    ///
    /// Only an approximation to a true inverse — it reads the falloff and the closest point at
    /// `p` rather than at the pre-image, which has no closed form for an arbitrary warp. That
    /// is what every implementation of this technique ships, and the cost of it is measured
    /// rather than assumed: `fieldform-field/examples/attractor_profile.rs`.
    ///
    /// Like a twist it reports [`crate::Normalization::LowerBound`], having divided by how far
    /// it can stretch material — `1 + strength`, exactly, which is one number rather than a
    /// bound taken from the body's extent. So a boolean over an attracted body is exact and a
    /// blend over one is approximate and says so.
    ///
    /// It names a **curve**, like [`Node::Pipe`], rather than a face or an edge — the boundary
    /// rule that keeps topological naming from arising.
    Attract {
        input: Operand,
        /// What it is drawn toward: a curve, or a point. See [`Attractor`].
        toward: Attractor,
        /// How hard the pull is, as a fraction: 0 is no warp at all and 1 doubles the distance
        /// the sample point is pushed out, which is as far as the material can be drawn in.
        ///
        /// Unitless rather than in millimetres because how far anything actually moves depends
        /// on `falloff` as well — the largest displacement is about `0.26 · strength · falloff`,
        /// and the panel says so in millimetres where the number is typed.
        ///
        /// Folding — the warp mapping two places onto one — begins at `16/11`, which is a
        /// property of the falloff curve rather than of the geometry, so the panel warns from
        /// there rather than guessing per model. See §8 of the spec and
        /// `folding_begins_where_the_falloff_says_it_does`.
        strength: ParamRef,
        /// How far the pull reaches, in millimetres. Zero beyond it, exactly.
        ///
        /// **Bounded on purpose.** A gravitational falloff never quite reaches zero, which
        /// makes the warp's support the whole of space — and a field whose effect has no edge
        /// cannot be given a bounding box, which is what both the mesher and the ray marcher
        /// size themselves from. A bounded falloff keeps this node's cost local, and the box it
        /// reports honest.
        falloff: ParamRef,
    },
    /// Another field, repeated — every copy unioned into one body.
    ///
    /// **One node for what used to be two blocks per instance.** Repeating something six times
    /// left twelve blocks in the list: a placement and a boolean each, for one act. The count
    /// is a number on this node instead, so changing six to eight is typing a digit rather
    /// than rebuilding the pattern.
    ///
    /// Every instance is a rigid motion of the same body — a translation, or a rotation about
    /// an axis — so distances are preserved and the union of them is still a true distance
    /// field. A pattern therefore stays blendable, which the placement-and-boolean form also
    /// managed but for a reason that had to be re-derived each time.
    Pattern {
        input: Operand,
        /// How many copies **including the original**.
        ///
        /// Including it is the off-by-one every pattern tool has to decide about out loud: six
        /// means six bodies, not the original plus six.
        ///
        /// **A `ParamRef`, so the count can be a variable — the first integer-valued one in the
        /// graph.** Asked for by the owner on 2026-09-30: "in pattern the number of copies should
        /// also be able to be declared as a var". Every other count here is a bare `u32`
        /// ([`CellMap::Cylindrical::arc_count`] and its siblings), so this sets the convention, and
        /// [`crate::pattern_count`] is the whole of it — nothing may read this field and convert it
        /// itself.
        ///
        /// **Why `arc_count`'s reasoning does not forbid it.** That field is whole because a
        /// fractional count breaks a *wrapping* coordinate: the cell formula is `2π`-periodic, so
        /// the two sides of the seam agree only for an integer, and a fraction puts a discontinuity
        /// down the cylinder. A pattern enumerates `0..n` instances and has no seam, so a fractional
        /// count there has no meaning rather than a broken one. What does carry over is that
        /// field's second clause — the number typed and the geometry built may not disagree without
        /// saying so — and [`CountFit`] is how that is honoured.
        ///
        /// **Always a whole number, including when a variable drives it.** The first ruling of
        /// 2026-09-30 refused a fraction instead, on the ground that `floor()` and `round()` are in
        /// the expression language so the system need not guess. The owner reversed it the same day
        /// on what that meant in use: "the decision we made was that it may break if it's a decimal,
        /// and obviously that doesn't work." A variable is a thing people drive from elsewhere, and
        /// much of what drives one passes through a fraction on the way, so refusing blanks the part
        /// for a value nobody typed and nobody meant to stop at.
        ///
        /// So [`pattern_count`] rounds, to the nearest, and reports that it did. It rounds **on
        /// read and never on write**: a parameter is document-wide and the same name may drive a
        /// radius, so rounding what is stored would move that radius to suit this count. What stays
        /// refused is only what cannot be rounded into geometry at all — a value that will not
        /// resolve, is not a number, rounds to nothing, or comes from a ramp. The 512 ceiling caps,
        /// announced on the row for the same reason the rounding is.
        count: ParamRef,
        kind: PatternKind,
        /// Whether the block holds instance zero — the body in its own place — as well as the
        /// copies.
        ///
        /// **The same field, spelled the same way, as [`Node::Mirror`]'s (D9.)** Both blocks
        /// make something out of a body and then have to say whether the body is part of what
        /// they made. A mirror that holds both halves swallows the body it reflected; so does a
        /// pattern that holds instance zero, and for the same reason — a consumed node is not a
        /// root, so it leaves the body list and cannot be shown, hidden or handed to a boolean
        /// on its own.
        ///
        /// So `false` is what the tool now builds: the block holds copies one to `count - 1`,
        /// and instance zero stays a body of its own. **The picture does not change** — the
        /// same `count` instances are on screen — the body list simply gains the row that used
        /// to disappear, and hiding it now leaves a gap in the ring rather than doing nothing.
        ///
        /// `true` is what every pattern written before this existed meant, which is why it is
        /// what a document missing the field is read as.
        join: bool,
        /// How the copies meet each other.
        ///
        /// **The fold is a union, and a union can be rounded.** Hard is the default and the
        /// only thing a pattern could do before: copies that touch met in a seam with no way to
        /// blend them, which is what ribs, fins and lugs on a hub all want. A radius here is
        /// the *same* [`Fillet`] a blend block takes, applied to every join in the ring at once
        /// rather than one pair at a time.
        ///
        /// It is the n-ary form of the same expression, not repeated pairwise: pairwise
        /// blending is not associative, so which copy was folded first would show at a junction
        /// where three meet.
        ///
        /// **Ruled 2026-10-02: a pattern folds by union by default.** Raised 2026-09-13 as "the
        /// pattern block is unioning them by default — feels like maybe it should not be", and
        /// answered *yes, it should*: the fold is what makes a count affordable (12 and 40 copies
        /// cost the same per reading), `join` keeps the original a body of its own, and this is how
        /// the copies are rounded when they should be. The tool builds `join: false` and
        /// [`Fillet::hard`]; `a_pattern_holding_only_its_copies_leaves_the_body_out` pins it.
        fillet: Fillet,
        /// Spacing, symmetry, a partial angle, a second direction and the copies to leave out —
        /// see [`PatternOptions`]. Last, so the fields above keep their places.
        options: PatternOptions,
    },
    /// Another field, reflected — and by default joined to what it was reflected from, which
    /// is what "mirror" means when a person says it.
    ///
    /// **One node, not a reflection beside a boolean.** It was two blocks and a stray Union in
    /// the list, which is one operation described twice and reads as though something else
    /// happened. `join` is what folds them together; turning it off gives the bare reflection,
    /// which is occasionally what you want when mirroring something in order to subtract it.
    ///
    /// A reflection is an **isometry**, so unlike almost everything else that touches the
    /// argument it costs nothing: distances are preserved exactly, `Distance` normalization
    /// carries through, and a mirrored body is still blendable. Evaluating the inner field at
    /// the reflected point is the whole implementation, with no factor to multiply back — which
    /// is the difference between this and `Transform`, where the scale has to be undone.
    Mirror {
        input: Operand,
        /// Where the mirror plane is, as a [`Frame`]: its origin is a point on the plane and
        /// its axis is the plane's normal.
        ///
        /// The same type a lattice, a twist and a radial pattern take (D9), so a plane set up
        /// for one can be handed to any of them. This replaced a `MirrorPlane` that said
        /// exactly this and could be said to nothing else.
        frame: Frame,
        /// Keep what was reflected as well, so the result is symmetric.
        ///
        /// True is the ordinary meaning of the word. False leaves the body only in its
        /// reflected position, which is a move rather than a mirror and is worth being able to
        /// ask for separately.
        join: bool,
    },
    /// A body hollowed out to a wall of a given thickness.
    ///
    /// One block, though what it means is two operations: the body, minus an inward offset of
    /// itself. It compiles to exactly that — there is no separate shell arithmetic — but a
    /// person doing it is doing one thing and answering one question, so the block asks one
    /// question. It used to be an `Offset` and a `Boolean` side by side, which put the wall
    /// thickness on a different block from the one named for the operation, and offered a
    /// boolean's union/intersect and a blend radius to something that is neither.
    ///
    /// The wall is measured **inward**, so the outside dimensions stay the ones the body was
    /// built to — which is what a wall thickness means on a drawing.
    Shell {
        input: Operand,
        thickness: ParamRef,
    },
    /// A field read as a *number*, mapped onto a range — the one thing that may drive a
    /// parameter.
    ///
    /// Below `in_min` the output is `out_min`, above `in_max` it is `out_max`, and between
    /// them it interpolates with the chosen shape. `in_max` below `in_min` is allowed and
    /// runs the mapping backwards, which is how "thicker where the field is *smaller*" is
    /// said without a second node.
    ///
    /// **It is not a body**, and [`Graph::roots`] leaves it out of the bodies for that reason:
    /// its value is a thickness or a radius, not a distance to a surface, so drawing its zero
    /// set would put a shape on screen that means nothing. Anything it reads is consumed like
    /// any other operand.
    ///
    /// The point of routing every driven parameter through this one node rather than allowing
    /// any field is the bound. See [`ParamRef::Field`].
    Ramp {
        /// The field read at each point, or `None` while nothing has been chosen yet.
        ///
        /// Any node — a plane gives a linear gradient, a point a radial one, a pipe the
        /// distance to a curve.
        ///
        /// **Optional because "placed but not yet wired" is a real state**, which
        /// `ramp-decision.md` §7.2 already describes: a ramp put down from the Create menu is a
        /// root that draws nothing until it is told what to read. Making this a bare `NodeId`
        /// meant a fresh ramp had to invent something to read, so placing one silently added a
        /// point to the part — reported as exactly that. A ramp with nothing to read is refused
        /// only when something tries to read *it*.
        input: Operand,
        in_min: ParamRef,
        in_max: ParamRef,
        out_min: ParamRef,
        out_max: ParamRef,
        shape: RampShape,
    },
    /// Another field's boundary moved by a distance: outward when positive, inward when
    /// negative.
    ///
    /// One subtraction, `F(p) − d`, and that is not an approximation of an offset — it *is*
    /// the offset. The set a field describes is where `F ≤ 0`, so the set offset by `d` is
    /// where `F ≤ d`, and the distance to *that* boundary is `F − d` exactly. Every property
    /// carries through untouched: the gradient is unchanged, so `Distance` normalization
    /// survives and an offset body stays blendable, which is the whole reason this is worth
    /// having as a node rather than as a special case inside something larger.
    ///
    /// This is what B-rep kernels fail at and the reason the architecture argues for fields:
    /// offsetting a boundary surface there means re-intersecting every face and rebuilding the
    /// topology, and self-intersections have to be detected and trimmed. Here there is nothing
    /// to trim, because a self-intersecting offset is simply a level set that happens to have
    /// merged.
    ///
    /// **Not thicken.** Offset moves an existing boundary; thicken builds a solid around a
    /// surface that has no thickness to begin with. Another CAD package keeps a support article
    /// about users
    /// conflating the two, which is reason enough to name them apart from the start.
    ///
    /// A large enough negative distance erodes the body to nothing. That is not refused: `F −
    /// d` stays a perfectly well-formed field that is positive everywhere, an empty solid is a
    /// real answer to "shrink this by more than its own thickness", and the alternative is
    /// deciding on the user's behalf how much shrinking is too much.
    Offset {
        input: Operand,
        /// Signed, in millimetres. Positive grows, negative shrinks.
        distance: ParamRef,
    },
    /// Another field, placed: translated, rotated, and scaled uniformly.
    ///
    /// This is what the manipulator writes when a whole body is dragged, and it wraps a node
    /// rather than sitting on [`crate::Part`] because a body — a root of the part's graph —
    /// is what the viewport selects. Moving one body of a two-body part has to be
    /// expressible, and a transform on the part could not say it.
    ///
    /// The scale is one factor rather than three, deliberately. Translation and rotation are
    /// isometries and uniform scale is a similarity, so evaluating the inner field at the
    /// inverse-transformed point and multiplying the result back by the factor is still an
    /// exact Euclidean distance. Three independent factors are not: a squashed field reports
    /// the unsquashed one's distance, which *overestimates* along the shortened axis and so
    /// breaks the `ScalarField` contract outright — and note that declaring
    /// `Normalization::LowerBound` does not by itself mend it, since a lower bound is precisely
    /// what an overestimate is not. Scaling the result by the *smallest* of the three factors
    /// would, at the cost of reading short by their ratio everywhere else. Either way per-axis
    /// scale would quietly produce bodies whose fillets are no longer the radius asked for,
    /// discovered at the blend rather than at the drag. Uniform scale keeps every body exact.
    ///
    /// Rotation is intrinsic XYZ Euler angles in **degrees**, not radians: it is a number the
    /// user types and reads back, and the file is meant to stay legible.
    Transform {
        input: Operand,
        translation: [ParamRef; 3],
        rotation: [ParamRef; 3],
        scale: ParamRef,
        /// The point the turn and the scale happen about. The origin unless it is set.
        ///
        /// **Where "scale it by two" needs somewhere to scale from.** Without this a placement
        /// turns and scales about its own local origin, which is the right answer only when the
        /// body happens to be centred there — anything else swings away as it grows, and the
        /// way to stop it was to work out a translation that cancels the swing. That is
        /// arithmetic a person should not be doing, and it changes every time the factor does.
        ///
        /// A [`Place`], so it is the same "where" every other block takes and can sit on a
        /// point: put the pivot on a point block and several placements turn about the one
        /// thing (D9). Read in the frame the placement's own move is read in, which is the
        /// frame its parent establishes — the one frame this placement is not yet part of.
        ///
        /// Written to a file only when it is not the origin, so a placement that does not use
        /// it writes what it always wrote and nothing saved before this needs migrating.
        about: crate::Place,
    },
    /// A node this build cannot account for, kept exactly as the file wrote it.
    ///
    /// A newer build — or a plugin — writes a node kind, or a field on a kind, that this one
    /// has never heard of. Dropping it would destroy someone else's work the moment one
    /// person on a team lacks one module, so the graph carries it through unchanged and
    /// writes it back verbatim.
    ///
    /// It does not evaluate. A field that silently ignores a node it does not understand
    /// produces geometry that is confidently the wrong shape, which is the failure this whole
    /// format argues against, so compilation refuses and names what it could not read.
    Unknown(UnknownNode),
}

impl Node {
    /// Rewrites every input this node names, for whichever of them the map mentions.
    ///
    /// Beside [`Node::inputs`] deliberately: the two have to agree about which fields are
    /// edges, and a node kind added to one and forgotten in the other produces a copy that
    /// still points at the original — which looks like it worked and is a second body sharing
    /// one subtree.
    pub(crate) fn remap_inputs(&mut self, map: &std::collections::HashMap<NodeId, NodeId>) {
        // **One line for every operand, because one match knows where they are.** This was a
        // second exhaustive match beside `for_each_operand`'s, kept "beside [`Node::inputs`]
        // deliberately: the two have to agree about which fields are edges, and a node kind added
        // to one and forgotten in the other produces a copy that still points at the original".
        // They agree by construction now — [`Self::operand_slots_mut`] is the only place that
        // knows, so there is nothing left to forget. It also carried a dead statement,
        // `{ let _ = &input; }` on the mirror arm, left over from the migration of a mirror's
        // plane into `Frame`.
        let redirect = |slot: &mut NodeId| {
            if let Some(fresh) = map.get(slot) {
                *slot = *fresh;
            }
        };
        for slot in self.operand_slots_mut() {
            slot.remap(map);
        }
        // An attractor's target is not an operand (see [`Self::for_each_operand`]), so it is
        // remapped here, where its own shape is known.
        if let Node::Attract { toward: Attractor::Point(id), .. } = self {
            redirect(id);
        }
        // Parameter edges, after the arms above, because a driven parameter is an edge on
        // whatever node carries it rather than a property of one kind. Doing it here is what
        // keeps this from having to be remembered per arm — the mistake the note above warns
        // about, made once for every node that ever gains a driveable input.
        for r in self.param_refs_mut() {
            if let ParamRef::Field(id) = r {
                redirect(id);
            }
        }
        // And frames, for the same reason and by the same arrangement — D9. Every block a
        // frame names, since one between two points names two.
        for f in self.frames_mut() {
            for id in f.nodes_mut() {
                redirect(id);
            }
        }
        // And places. `Place::remap` takes the pair rather than handing out a `&mut NodeId`,
        // because a place has only one thing to redirect and no reason to expose it.
        for place in self.places_mut() {
            if let Some(from) = place.source() {
                let mut to = from;
                redirect(&mut to);
                place.remap(from, to);
            }
        }
        // And a facing taken from a block, the same way and for the same reason. Every block
        // it names, since one between two points names two.
        if let Some(turn) = self.turn_mut() {
            for id in turn.nodes_mut() {
                redirect(id);
            }
        }
        // And a pair of ends standing on one.
        for ends in self.ends_mut() {
            if let Some(from) = ends.node() {
                let mut to = from;
                redirect(&mut to);
                ends.remap(from, to);
            }
        }
    }

    /// Every parameter this node carries, in a fixed order.
    ///
    /// Beside [`Self::param_refs_mut`], and the two must list the same things. This exists so
    /// that "which nodes does this one read through its parameters" has one answer rather than
    /// one per caller — [`Self::inputs`] and [`Self::remap_inputs`] both go through it, so a
    /// node kind that gains a driveable parameter cannot end up an edge to one and not the
    /// other.
    pub fn param_refs(&self) -> Vec<&ParamRef> {
        let mut out: Vec<&ParamRef> = Vec::new();
        self.for_each_param_ref(&mut |r| out.push(r));
        out
    }

    /// [`Self::param_refs`], handed over one at a time and in the same order, without the list.
    ///
    /// **The one spelling of which numbers a node holds**, so the list and the walk cannot differ
    /// by an index — and the walk is where [`Self::for_each_input`] finds a node's drivers, with
    /// nothing allocated. Each arm still reads as the list it was written as: `out` hands each
    /// number to `visit` as the list's `push` and `extend` stored it, in the same order.
    fn for_each_param_ref<'a>(&'a self, visit: &mut impl FnMut(&'a ParamRef)) {
        let mut out = AsList(visit);
        match self {
            Node::Sphere { centre, radius } => {
                out.extend(centre.refs());
                out.push(radius);
            }
            Node::Cuboid { centre, size, turn, edge } => {
                out.extend(centre.refs());
                out.extend(size);
                out.extend(turn.params());
                out.extend(edge.params());
            }
            Node::SkewedBox { centre, edges } => {
                out.extend(centre.refs());
                out.extend(edges.iter().flatten());
            }
            Node::Cylinder { ends, radius, edge } => {
                out.extend(ends.params());
                out.push(radius);
                out.extend(edge.params());
            }
            Node::Cone { ends, base_radius, top_radius, edge } => {
                out.extend(ends.params());
                out.push(base_radius);
                out.push(top_radius);
                out.extend(edge.params());
            }
            Node::Torus { centre, major_radius, minor_radius, turn } => {
                out.extend(centre.refs());
                out.push(major_radius);
                out.push(minor_radius);
                out.extend(turn.params());
            }
            Node::Capsule { ends, start_radius, end_radius } => {
                out.extend(ends.params());
                out.push(start_radius);
                out.push(end_radius);
            }
            // A typed origin's three numbers are still numbers, so `fx` names them like any
            // other. One taken from a block has none to name, and `Place::refs` says so.
            Node::Plane { origin, turn, .. } => {
                out.extend(origin.refs());
                out.extend(turn.params());
            }
            Node::Point { at } => out.extend(at),
            Node::Pipe { radius, .. } => out.push(radius),
            // Two curves, a wall's thickness, and what the law carries: a taper, a twist and
            // the two ends of its extent.
            Node::Sweep { outline, law, rims, .. } => {
                out.extend(outline.params());
                out.extend(law.params());
                out.extend(rims.params());
            }
            Node::Loft { sections, rims, .. } => {
                for section in sections {
                    out.extend(section.params());
                }
                out.extend(rims.params());
            }
            Node::Lattice { cell, phase, frame, map } => {
                // The phase's own number first, which is the slot a ramp is wired into.
                out.push(phase.param());
                out.extend(phase.offset());
                out.extend(cell.jitter());
                // A frame's numbers resolve like any other. Listed here beside the node's own
                // rather than gathered afterwards, because the arm has the frame in hand and a
                // second pass over `self` cannot borrow it again.
                out.extend(frame.params());
                // Last, so every index a socket may already hold stays where it was — see
                // `CellMap::params` for why the map's numbers are here at all.
                out.extend(map.params());
            }
            // The draft belongs here or it is not a parameter at all: this is what lets a
            // number be named, driven and dragged, and a field left out of it is one the panel
            // can show and nothing else can reach.
            // The outline's last, so the three an extrude always had keep the places they had.
            Node::Extrude { outline, start, height, draft, sweep, sides, corners: _, rims } => {
                out.push(height);
                out.push(draft);
                out.extend(sweep.params());
                out.extend(outline.params());
                // After the outline's, which were last, so every number keeps the place it had.
                out.push(start);
                out.extend(sides.params());
                // Last of all, for the same reason.
                out.extend(rims.params());
            }
            // The outline's last, as an extrude's is, so the frame's keep the places they had.
            Node::Revolve { outline, frame, angle, sides, rims } => {
                out.extend(frame.params());
                out.extend(outline.params());
                // After the outline's, as an extrude's start is, so no number moved.
                out.push(angle);
                out.extend(sides.params());
                // Last of all, as an extrude's are.
                out.extend(rims.params());
            }
            Node::Blend { fillet, .. } => out.extend(fillet.params()),
            Node::Morph { t, .. } => out.push(t),
            Node::Offset { distance, .. } => out.push(distance),
            Node::Shell { thickness, .. } => out.push(thickness),
            Node::Twist { degrees_per_mm, frame, .. } => {
                out.push(degrees_per_mm);
                out.extend(frame.params());
            }
            Node::Attract { strength, falloff, .. } => {
                out.push(strength);
                out.push(falloff);
            }
            Node::Transform { translation, rotation, scale, .. } => {
                out.extend(translation);
                out.extend(rotation);
                out.push(scale);
            }
            Node::Ramp { in_min, in_max, out_min, out_max, .. } => {
                out.push(in_min);
                out.push(in_max);
                out.push(out_min);
                out.push(out_max);
            }
            Node::Mirror { frame, .. } => out.extend(frame.params()),
            Node::Pattern { kind, fillet, count, options, .. } => {
                match kind {
                    PatternKind::Linear { step } => out.extend(step),
                    PatternKind::Radial { frame } => out.extend(frame.params()),
                }
                out.extend(fillet.params());
                // **Last, and that position is load-bearing.** A slot's index in this list is used
                // as a handle: `wire_driver` reaches a socket by `param_refs_mut().get_mut(slot)`
                // and `release_drivers` writes back by enumeration. Appending leaves every existing
                // index where it was; pushing the count first would silently renumber a pattern's
                // step and fillet sockets under whatever is holding one.
                //
                // Here at all because this is the wiring D9 asks a shared type to carry: being in
                // this list is what gives the count its cycle check, its rename propagation and its
                // remap on paste, without anyone having to remember. A field added to the struct and
                // not to this list is a parameter in name only.
                out.push(count);
                // The options' numbers after the count, for the same reason: appending leaves every
                // index a socket may be holding where it was. Only the one that applies to this
                // kind — a row's second direction, a ring's angle — since the other is not a
                // number this block reads.
                match kind {
                    PatternKind::Linear { .. } => {
                        if let Some(second) = &options.second {
                            out.extend(&second.step);
                            out.push(&second.count);
                        }
                    }
                    PatternKind::Radial { .. } => out.push(&options.angle),
                }
            }
            Node::Boolean { .. } | Node::Cage(_) | Node::Mesh(_) | Node::Unknown(_) => {}
        }
    }

    /// Every parameter this node carries, to change. See [`Self::param_refs`].
    pub fn param_refs_mut(&mut self) -> Vec<&mut ParamRef> {
        let mut out: Vec<&mut ParamRef> = Vec::new();
        match self {
            Node::Sphere { centre, radius } => {
                out.extend(centre.refs_mut());
                out.push(radius);
            }
            Node::Cuboid { centre, size, turn, edge } => {
                out.extend(centre.refs_mut());
                out.extend(size.iter_mut());
                out.extend(turn.params_mut());
                out.extend(edge.params_mut());
            }
            Node::SkewedBox { centre, edges } => {
                out.extend(centre.refs_mut());
                out.extend(edges.iter_mut().flatten());
            }
            Node::Cylinder { ends, radius, edge } => {
                out.extend(ends.params_mut());
                out.push(radius);
                out.extend(edge.params_mut());
            }
            Node::Cone { ends, base_radius, top_radius, edge } => {
                out.extend(ends.params_mut());
                out.push(base_radius);
                out.push(top_radius);
                out.extend(edge.params_mut());
            }
            Node::Torus { centre, major_radius, minor_radius, turn } => {
                out.extend(centre.refs_mut());
                out.push(major_radius);
                out.push(minor_radius);
                out.extend(turn.params_mut());
            }
            Node::Capsule { ends, start_radius, end_radius } => {
                out.extend(ends.params_mut());
                out.push(start_radius);
                out.push(end_radius);
            }
            Node::Plane { origin, turn, .. } => {
                out.extend(origin.refs_mut());
                out.extend(turn.params_mut());
            }
            Node::Point { at } => out.extend(at.iter_mut()),
            Node::Pipe { radius, .. } => out.push(radius),
            Node::Sweep { outline, law, rims, .. } => {
                out.extend(outline.params_mut());
                out.extend(law.params_mut());
                out.extend(rims.params_mut());
            }
            Node::Loft { sections, rims, .. } => {
                for section in sections {
                    out.extend(section.params_mut());
                }
                out.extend(rims.params_mut());
            }
            Node::Lattice { cell, phase, frame, map } => {
                // Both of a sheet's numbers are wanted at once, and an accessor for each would
                // borrow the whole phase twice, so the two are taken in one match; every other
                // phase has only the one `param_mut` hands out.
                match phase {
                    LatticePhase::Sheet { thickness, offset } => {
                        out.push(thickness);
                        out.push(offset);
                    }
                    other => out.push(other.param_mut()),
                }
                out.extend(cell.jitter_mut());
                out.extend(frame.params_mut());
                // Last, in step with `param_refs`.
                out.extend(map.params_mut());
            }
            Node::Extrude { outline, start, height, draft, sweep, sides, corners: _, rims } => {
                out.push(height);
                out.push(draft);
                out.extend(sweep.params_mut());
                out.extend(outline.params_mut());
                out.push(start);
                out.extend(sides.params_mut());
                out.extend(rims.params_mut());
            }
            Node::Revolve { outline, frame, angle, sides, rims } => {
                out.extend(frame.params_mut());
                out.extend(outline.params_mut());
                out.push(angle);
                out.extend(sides.params_mut());
                out.extend(rims.params_mut());
            }
            Node::Blend { fillet, .. } => out.extend(fillet.params_mut()),
            Node::Morph { t, .. } => out.push(t),
            Node::Offset { distance, .. } => out.push(distance),
            Node::Shell { thickness, .. } => out.push(thickness),
            Node::Twist { degrees_per_mm, frame, .. } => {
                out.push(degrees_per_mm);
                out.extend(frame.params_mut());
            }
            Node::Attract { strength, falloff, .. } => {
                out.push(strength);
                out.push(falloff);
            }
            Node::Transform { translation, rotation, scale, .. } => {
                out.extend(translation.iter_mut());
                out.extend(rotation.iter_mut());
                out.push(scale);
            }
            Node::Ramp { in_min, in_max, out_min, out_max, .. } => {
                out.push(in_min);
                out.push(in_max);
                out.push(out_min);
                out.push(out_max);
            }
            Node::Mirror { frame, .. } => out.extend(frame.params_mut()),
            Node::Pattern { kind, fillet, count, options, .. } => {
                let row = matches!(kind, PatternKind::Linear { .. });
                match kind {
                    PatternKind::Linear { step } => out.extend(step.iter_mut()),
                    PatternKind::Radial { frame } => out.extend(frame.params_mut()),
                }
                out.extend(fillet.params_mut());
                // Last, in step with `param_refs` — see the note there.
                out.push(count);
                if row {
                    if let Some(second) = &mut options.second {
                        out.extend(second.step.iter_mut());
                        out.push(&mut second.count);
                    }
                } else {
                    out.push(&mut options.angle);
                }
            }
            Node::Boolean { .. } | Node::Cage(_) | Node::Mesh(_) | Node::Unknown(_) => {}
        }
        out
    }

    /// The nodes this one consumes.
    ///
    /// Empty for [`Node::Unknown`]: its inputs are precisely what this build cannot read.
    /// Nothing depends on that being complete, because a graph holding one refuses to
    /// compile at all.
    ///
    /// The list, for a caller that keeps one or compares two. A walk over a part asks
    /// [`Self::for_each_input`], which this collects.
    pub fn inputs(&self) -> Vec<NodeId> {
        let mut edges = Vec::new();
        self.for_each_input(|id| edges.push(id));
        edges
    }

    /// [`Self::inputs`], handed to `visit` one at a time and in the same order, with nothing
    /// allocated: the operands, then whatever drives a parameter, then what the node is pointed at.
    ///
    /// **Because the whole part is asked on every edit.** [`Graph::roots`], the evaluation order,
    /// [`Graph::consumed`], who reads a block and `Document::settle_ownership` each walk every
    /// node's inputs, and they run on every graph edit, every frame of a drag included. The list
    /// was a fresh allocation every time, and so were the lists it was gathered from: four for a
    /// sphere standing on a block, eight for a two-sided extrude of a wall.
    ///
    /// A visitor rather than an iterator because the answer comes from a match over every kind of
    /// node, each arm of which would need the same iterator type.
    pub fn for_each_input(&self, mut visit: impl FnMut(NodeId)) {
        self.for_each_operand(&mut visit);
        // And whatever drives a parameter, which is an edge for exactly the same reasons: it
        // has to be evaluated first, it must not be a body of its own, and a cycle through it
        // is still a cycle.
        self.for_each_param_ref(&mut |r| {
            if let Some(driver) = r.driver() {
                visit(driver);
            }
        });
        self.for_each_reference(&mut visit);
    }

    /// Whether `id` is among [`Self::inputs`], asked without building the list.
    pub(crate) fn reads(&self, id: NodeId) -> bool {
        let mut found = false;
        self.for_each_input(|input| found |= input == id);
        found
    }

    /// The nodes this one is *made of*: a boolean's operands, the body a twist warps.
    ///
    /// **As against the ones it is merely pointed at** — see `for_each_reference`. Both are
    /// inputs, and for the same three reasons; the difference is what a placement means. A block
    /// this one is built from sits *inside* whatever placement wraps this one, so a walk composing
    /// placements must descend through it. A block it is aimed at does not: the plane a cuboid
    /// takes its facing from is a block of its own, standing where it stands, and carrying the
    /// cuboid's placement down onto it applies that rotation twice.
    ///
    /// Reported as exactly that arithmetic: "I rotate the plane by 45 degrees, it looks like it
    /// was rotated by 90, but the cube is rotated 45". The cube was right; the square drawn for
    /// the plane had the plane's own turn composed onto the cube's copy of it.
    pub fn operands(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.for_each_operand(&mut |id| out.push(id));
        out
    }

    /// [`Self::operands`], one at a time and in the same order, without the list. The match lives
    /// here, so the list and the walk cannot disagree.
    /// Every slot this node reads a field through, filled or empty.
    ///
    /// **The single place that knows a node has an operand at all** (D9). The exhaustive match
    /// lives here and nowhere else, so a block that gains a slot gains its graph edge, its remap
    /// and its cycle check at once — the arrangement [`Self::frames`] already has. It moved here
    /// from `for_each_operand`, which is now a reading of it, because the hole has to be visible
    /// to `remap_inputs` as well as to the edge walk and two matches would have drifted.
    /// [`Self::for_each_operand_slot`] for other modules of the crate, so `Graph::incomplete` can
    /// ask where the holes are without the walk becoming part of the public surface.
    pub(crate) fn for_each_operand_slot_public<'a>(
        &'a self,
        visit: &mut impl FnMut(&'a Operand),
    ) {
        self.for_each_operand_slot(visit);
    }

    fn for_each_operand_slot<'a>(&'a self, visit: &mut impl FnMut(&'a Operand)) {
        let mut out = AsList(visit);
        match self {
            // A profile is a *curve*, which lives in the document rather than the graph — the
            // same arrangement a pipe's spine has. So neither of these has a node input of its
            // own, and both are leaves as far as the graph is concerned.
            Node::Cage(_)
            | Node::Pipe { .. }
            | Node::Sweep { .. }
            | Node::Loft { .. }
            | Node::Extrude { .. }
            | Node::Revolve { .. }
            | Node::Sphere { .. }
            | Node::Cuboid { .. }
            | Node::SkewedBox { .. }
            | Node::Cylinder { .. }
            | Node::Cone { .. }
            | Node::Torus { .. }
            | Node::Capsule { .. }
            | Node::Plane { .. }
            | Node::Point { .. }
            | Node::Lattice { .. }
            | Node::Mesh(_)
            | Node::Unknown(_) => {}
            Node::Blend { a, b, .. } | Node::Morph { a, b, .. } => out.extend([a, b]),
            Node::Boolean { inputs, .. } => out.extend(inputs.iter()),
            Node::Transform { input, .. }
            | Node::Offset { input, .. }
            | Node::Twist { input, .. }
            | Node::Shell { input, .. }
            | Node::Pattern { input, .. }
            | Node::Attract { input, .. }
            // An unwired ramp has no edge, which is what makes it a root that draws nothing
            // rather than a node pointing at something that is not there — and is why every slot
            // can now be that without a new state having to be invented for it.
            | Node::Ramp { input, .. }
            // The plane a mirror reads is a frame now, so its edge comes from
            // `for_each_reference` with everything else's — see D9.
            | Node::Mirror { input, .. } => out.push(input),
        }
    }

    /// Every slot, to write to: the mutable twin of [`Self::for_each_operand_slot`], which is what
    /// `remap_inputs` and §7.2's drag-out write through. A `Vec` rather than a visitor because a
    /// caller holding two slots at once is the ordinary case (a blend's pair), and a closure
    /// taking `&mut` one at a time cannot.
    /// Every operand slot of this node, in the order its rows read, holes included.
    ///
    /// **Indices, because one gesture needs to tell two slots apart.** §7.3's move -- a block
    /// dragged from one input to another -- empties the slot it left, and when both slots belong
    /// to the *same* block the only thing that distinguishes them is where they sit:
    /// [`Self::stop_reading`] would empty the one just filled along with the one being left.
    /// `Graph::first_empty_operand` already answers in `(NodeId, usize)`, so an index is a word
    /// this vocabulary has.
    ///
    /// The exhaustive match still lives in one place ([`Self::operand_slots_mut`]): this is a
    /// reading of it, so a block that gains a slot appears here without being added to anything.
    pub fn operand_slots(&self) -> Vec<Operand> {
        let mut out = Vec::new();
        self.for_each_operand_slot(&mut |slot| out.push(*slot));
        out
    }

    /// Empties the slot at `index`, and answers whether there was one to empty.
    ///
    /// The indexed half of [`Self::stop_reading`], for the one caller that has to say *which*
    /// slot: §7.3's move out of a sibling slot of the same block. Everything else means "this
    /// block stops reading that one" and should say that instead.
    pub fn empty_operand_slot(&mut self, index: usize) -> bool {
        match self.operand_slots_mut().get_mut(index) {
            Some(slot) => {
                slot.clear();
                true
            }
            None => false,
        }
    }

    /// Empties every slot of this node that reads `block`, and answers whether any did.
    ///
    /// **§7.2's drag-out, said as what it is**: this block stops reading that one. The gesture
    /// takes a nested row out of the block that holds it, which leaves the holder waiting to be
    /// told what to read instead — and that is the whole edit, so the caller writes the node back
    /// and settling does the rest (a block nothing reads any more is a row of its own again).
    ///
    /// **The intent is public and the walk is not** (D9). [`Self::operand_slots_mut`] is the one
    /// place that knows a node has slots at all, and it stays private so a block that gains one
    /// gains its edge, its remap and its cycle check without anyone remembering. What the
    /// application needs is not the walk but this one question, and asking it here means a block
    /// added later answers it without being added to anything.
    ///
    /// **Every slot naming it, not the first.** A holder that reads one block twice has *one* row
    /// in the list — the row is per holder, since `listed_blocks` clones a shared row per owner —
    /// so the drag off that row means the holder stops reading it, and emptying one of the two
    /// would leave a gesture that has to be repeated to finish. Reading one block into two slots
    /// of the same block is the pathology `blocks::operands::same_block_twice` already warns
    /// about, so there is no case here worth splitting the row for.
    pub fn stop_reading(&mut self, block: NodeId) -> bool {
        let mut emptied = false;
        for slot in self.operand_slots_mut() {
            if slot.source() == Some(block) {
                slot.clear();
                emptied = true;
            }
        }
        emptied
    }

    pub(crate) fn operand_slots_mut(&mut self) -> Vec<&mut Operand> {
        match self {
            Node::Cage(_)
            | Node::Pipe { .. }
            | Node::Sweep { .. }
            | Node::Loft { .. }
            | Node::Extrude { .. }
            | Node::Revolve { .. }
            | Node::Sphere { .. }
            | Node::Cuboid { .. }
            | Node::SkewedBox { .. }
            | Node::Cylinder { .. }
            | Node::Cone { .. }
            | Node::Torus { .. }
            | Node::Capsule { .. }
            | Node::Plane { .. }
            | Node::Point { .. }
            | Node::Lattice { .. }
            | Node::Mesh(_)
            | Node::Unknown(_) => Vec::new(),
            Node::Blend { a, b, .. } | Node::Morph { a, b, .. } => vec![a, b],
            Node::Boolean { inputs, .. } => inputs.iter_mut().collect(),
            Node::Transform { input, .. }
            | Node::Offset { input, .. }
            | Node::Twist { input, .. }
            | Node::Shell { input, .. }
            | Node::Pattern { input, .. }
            | Node::Attract { input, .. }
            | Node::Ramp { input, .. }
            | Node::Mirror { input, .. } => vec![input],
        }
    }

    /// The blocks this one is made of, as ids: [`Self::for_each_operand_slot`] with the empty slots
    /// left out, since an empty slot is not an edge.
    ///
    /// **And an attractor's target, which is not a slot.** `Attract::toward` is an
    /// [`crate::Attractor`], whose other arm is a curve — it names what the warp pulls toward, not
    /// what the block is built from — so by D9's test it is not the same kind of thing as an
    /// operand and keeps its own refusal. Visited after `input`, which is the order
    /// `remove_cascading` and `listed_blocks` read.
    fn for_each_operand(&self, visit: &mut impl FnMut(NodeId)) {
        self.for_each_operand_slot(&mut |slot| {
            if let Some(id) = slot.source() {
                visit(id);
            }
        });
        if let Node::Attract { toward, .. } = self
            && let Some(id) = toward.node()
        {
            visit(id);
        }
    }

    /// The nodes this one is *pointed at*: where it sits, which way it faces, what it stands on.
    ///
    /// Every one of them is an input — it has to be evaluated first, it must not be a body of its
    /// own, and a cycle through it is still a cycle — which is why [`Self::inputs`] is the two
    /// lists together and why everything about deletion, cycles and bodies reads that.
    ///
    /// What they are not is *ingredients*. See [`Self::operands`] for the one thing that turns
    /// on the difference.
    ///
    /// Gathered here rather than per arm, so a node that gains a frame, a place, a turn or a
    /// pair of ends gains its edge without anybody remembering — D9, and the note in `operands`
    /// about the mistake that is otherwise made once per node kind.
    fn for_each_reference(&self, visit: &mut impl FnMut(NodeId)) {
        let mut out = AsList(visit);
        self.for_each_frame(|f| out.extend(f.nodes()));
        self.for_each_place(|p| out.extend(p.source()));
        out.extend(self.turn().into_iter().flat_map(Turn::nodes));
        self.for_each_ends(|e| out.extend(e.node()));
    }

    /// The places this node holds, so its graph edge comes with it.
    ///
    /// **The same arrangement `frames` has, for the same reason (D9).** A node that gains a
    /// place gains its edge, its remapping and its cycle-checking without anybody remembering to
    /// add it in three more functions — which is the mistake that was made once per node kind
    /// before frames carried their own wiring.
    pub fn places(&self) -> Vec<&crate::Place> {
        let mut out = Vec::new();
        self.for_each_place(|place| out.push(place));
        out
    }

    /// [`Self::places`], one at a time, without the list — where the match lives.
    fn for_each_place<'a>(&'a self, mut visit: impl FnMut(&'a crate::Place)) {
        match self {
            Node::Plane { origin, .. } => visit(origin),
            Node::Sphere { centre, .. }
            | Node::Cuboid { centre, .. }
            | Node::SkewedBox { centre, .. }
            | Node::Torus { centre, .. } => visit(centre),
            // Two, and gathered through `Ends` rather than reached into, so a block holding a
            // pair of places says so once.
            Node::Cylinder { ends, .. }
            | Node::Cone { ends, .. }
            | Node::Capsule { ends, .. } => ends.places().for_each(visit),
            // A placement's pivot, so putting it on a point makes the graph edge that keeps
            // the point alive and moves the placement when the point moves -- without anybody
            // having to remember `inputs` and `remap_inputs` separately. That is the whole
            // reason this list exists.
            Node::Transform { about, .. } => visit(about),
            _ => {}
        }
    }

    /// The same, to edit.
    pub fn places_mut(&mut self) -> Vec<&mut crate::Place> {
        match self {
            Node::Plane { origin, .. } => vec![origin],
            Node::Sphere { centre, .. }
            | Node::Cuboid { centre, .. }
            | Node::SkewedBox { centre, .. }
            | Node::Torus { centre, .. } => vec![centre],
            Node::Cylinder { ends, .. }
            | Node::Cone { ends, .. }
            | Node::Capsule { ends, .. } => ends.places_mut(),
            Node::Transform { about, .. } => vec![about],
            _ => Vec::new(),
        }
    }

    /// The turn this block carries as its own, for the blocks that have one.
    ///
    /// `None` for a block with no facing of its own — a sphere, which has none to turn, and the
    /// two-ended shapes, whose direction already falls out of where their ends are.
    pub fn turn(&self) -> Option<&Turn> {
        match self {
            Node::Cuboid { turn, .. } | Node::Torus { turn, .. } | Node::Plane { turn, .. } => {
                Some(turn)
            }
            _ => None,
        }
    }

    /// The same, to edit.
    pub fn turn_mut(&mut self) -> Option<&mut Turn> {
        match self {
            Node::Cuboid { turn, .. } | Node::Torus { turn, .. } | Node::Plane { turn, .. } => {
                Some(turn)
            }
            _ => None,
        }
    }

    /// The fillet this node rounds with, if it is a kind that rounds.
    ///
    /// One reader for "which blocks carry a radius", so hardening a part — every fillet to
    /// nought, which is what a critical surface is measured against — cannot miss a kind that
    /// gains one later the way a `match` written at the call site would.
    pub fn fillet(&self) -> Option<&Fillet> {
        match self {
            Node::Blend { fillet, .. } | Node::Pattern { fillet, .. } => Some(fillet),
            _ => None,
        }
    }

    /// The same, to edit.
    pub fn fillet_mut(&mut self) -> Option<&mut Fillet> {
        match self {
            Node::Blend { fillet, .. } | Node::Pattern { fillet, .. } => Some(fillet),
            _ => None,
        }
    }

    /// The frames this node carries, which is where every "where and which way" lives.
    ///
    /// The single place that knows a node has a frame at all. `inputs`, `remap_inputs` and
    /// parameter resolution all read this, so adding a frame to a node is one line rather than
    /// four scattered ones — D9.
    pub fn frames(&self) -> Vec<&crate::Frame> {
        let mut out = Vec::new();
        self.for_each_frame(|frame| out.push(frame));
        out
    }

    /// [`Self::frames`], one at a time, without the list — where the match lives.
    fn for_each_frame<'a>(&'a self, mut visit: impl FnMut(&'a crate::Frame)) {
        match self {
            Node::Lattice { frame, .. }
            | Node::Mirror { frame, .. }
            | Node::Twist { frame, .. }
            | Node::Revolve { frame, .. } => visit(frame),
            Node::Pattern { kind: PatternKind::Radial { frame }, .. } => visit(frame),
            _ => {}
        }
    }

    /// Every sweep direction this node carries.
    ///
    /// One today, and written the way `frames` and `ends` are so that a second block wanting a
    /// direction gains its curve edge without anybody remembering to come back here.
    pub(crate) fn sweeps(&self) -> Vec<&crate::Sweep> {
        match self {
            Node::Extrude { sweep, .. } => vec![sweep],
            _ => Vec::new(),
        }
    }

    /// The same, to be edited or redirected.
    pub(crate) fn sweeps_mut(&mut self) -> Vec<&mut crate::Sweep> {
        match self {
            Node::Extrude { sweep, .. } => vec![sweep],
            _ => Vec::new(),
        }
    }

    /// [`Self::frames`], to be edited or redirected.
    pub fn frames_mut(&mut self) -> Vec<&mut crate::Frame> {
        match self {
            Node::Lattice { frame, .. }
            | Node::Mirror { frame, .. }
            | Node::Twist { frame, .. }
            | Node::Revolve { frame, .. } => vec![frame],
            Node::Pattern { kind: PatternKind::Radial { frame }, .. } => vec![frame],
            _ => Vec::new(),
        }
    }

    /// The section this node is built on, if it is one of the blocks that has one.
    ///
    /// The counterpart of [`Node::frames`] for an [`Outline`], and for the same reason: three
    /// blocks hold one, and whatever asks "is this a wall, and of what" asks it here rather
    /// than matching the three by name — which is the list a fourth block would be left out of.
    pub fn outline(&self) -> Option<&Outline> {
        match self {
            Node::Extrude { outline, .. }
            | Node::Revolve { outline, .. }
            | Node::Sweep { outline, .. } => Some(outline),
            _ => None,
        }
    }

    /// The same, to be edited or exchanged for the other kind.
    pub fn outline_mut(&mut self) -> Option<&mut Outline> {
        match self {
            Node::Extrude { outline, .. }
            | Node::Revolve { outline, .. }
            | Node::Sweep { outline, .. } => Some(outline),
            _ => None,
        }
    }

    /// The curve this node is aimed by or carried along, which is never part of its own
    /// section: an extrude's sweep line, a revolve's axis line, a sweep's spine.
    ///
    /// **What the walk that finds a section leaves out**, said once. A line drawn from a
    /// profile's corner to say which way to pull it shares an end with the profile, and a walk
    /// that did not know to skip it would wander off down it. The compile, the extent and the
    /// viewport each pass this to the walk; a panel that wants to know what a block's section
    /// *is* has to pass the same curve, and this is where it gets it.
    pub fn aimed_by(&self) -> Option<CurveId> {
        match self {
            Node::Extrude { sweep, .. } => sweep.source(),
            Node::Revolve { frame, .. } => frame.curve_source(),
            Node::Sweep { spine, .. } => Some(*spine),
            Node::Loft { spine, .. } => *spine,
            _ => None,
        }
    }

    /// Every pair of ends this node carries.
    ///
    /// The counterpart of [`Node::frames`], and it exists for the same reason: a block that
    /// gains an [`Ends`] gains everything that follows from having one — its curve reference
    /// included — without anybody remembering to add it here.
    pub fn ends(&self) -> Vec<&Ends> {
        let mut out = Vec::new();
        self.for_each_ends(|ends| out.push(ends));
        out
    }

    /// [`Self::ends`], one at a time, without the list — where the match lives.
    fn for_each_ends<'a>(&'a self, mut visit: impl FnMut(&'a Ends)) {
        match self {
            Node::Cylinder { ends, .. }
            | Node::Cone { ends, .. }
            | Node::Capsule { ends, .. } => visit(ends),
            _ => {}
        }
    }

    /// The same, to be edited or redirected.
    pub fn ends_mut(&mut self) -> Vec<&mut Ends> {
        match self {
            Node::Cylinder { ends, .. }
            | Node::Cone { ends, .. }
            | Node::Capsule { ends, .. } => vec![ends],
            _ => Vec::new(),
        }
    }

    /// Every curve this node depends on.
    ///
    /// **Written generically because the hand-written version had already gone stale.** Deleting
    /// a curve is supposed to be refused while something is built on it, and the check matched
    /// `Node::Pipe` alone — so a curve driving an extrude or a revolve could be deleted out from
    /// under it, and those two had been added since the check was written. That is the mistake
    /// D9 describes as being made once per node kind, and this is the arrangement that stops it:
    /// one place that knows, consulted by everything that asks.
    pub fn curves(&self) -> Vec<CurveId> {
        let mut out: Vec<CurveId> = match self {
            Node::Pipe { curve, .. } => vec![*curve],
            Node::Sweep { outline, spine, .. } => vec![outline.curve(), *spine],
            Node::Loft { sections, spine, .. } => {
                let mut out: Vec<CurveId> = sections.iter().map(|s| s.curve()).collect();
                out.extend(*spine);
                out
            }
            // Only when the target is one. A point target names no curve, and reporting one
            // that is not there would keep a deleted curve alive in every check that asks.
            Node::Attract { toward, .. } => toward.curve().into_iter().collect(),
            // Through the outline, so a wall's curve is held as a closed profile's is.
            Node::Extrude { outline, .. } | Node::Revolve { outline, .. } => vec![outline.curve()],
            _ => Vec::new(),
        };
        out.extend(self.ends().iter().filter_map(|e| e.source()));
        // A frame can name a curve too, since a line is a way of saying where an axis is.
        out.extend(self.frames().iter().filter_map(|f| f.curve_source()));
        // And a sweep names one when a profile is thickened along a line.
        out.extend(self.sweeps().iter().filter_map(|s| s.source()));
        // And a turn names one when a block is aimed along a line.
        out.extend(self.turn().and_then(|t| t.curve()));
        // And a place names one when a block stands where a curve starts.
        out.extend(self.places().iter().filter_map(|p| p.curve()));
        out
    }

    /// The same, to be redirected when curves are renumbered — which a paste does, giving a
    /// copied subtree fresh ids for the curves it brought with it.
    pub(crate) fn curves_mut(&mut self) -> Vec<&mut CurveId> {
        // One match rather than two passes: a node names a curve directly *or* through its ends
        // and never both, so there is nothing to concatenate — and concatenating would mean
        // holding a borrow of one field across a borrow of the whole node, which is what the
        // two-pass version was refused for.
        match self {
            Node::Pipe { curve, .. } => vec![curve],
            Node::Sweep { outline, spine, .. } => vec![outline.curve_mut(), spine],
            Node::Loft { sections, spine, .. } => {
                let mut out: Vec<&mut CurveId> =
                    sections.iter_mut().map(|s| s.curve_mut()).collect();
                out.extend(spine.as_mut());
                out
            }
            Node::Attract { toward: Attractor::Curve(curve), .. } => vec![curve],
            // The three that carry both a place and a turn, each of which may name a curve.
            // Matched by name rather than through the generic arms below, because those borrow
            // the whole node to ask and so can answer for one field or the other and never both.
            Node::Cuboid { centre, turn, .. } | Node::Torus { centre, turn, .. } => {
                let mut out: Vec<&mut CurveId> = centre.curve_mut().into_iter().collect();
                out.extend(turn.curve_mut());
                out
            }
            Node::Plane { origin, turn, .. } => {
                let mut out: Vec<&mut CurveId> = origin.curve_mut().into_iter().collect();
                out.extend(turn.curve_mut());
                out
            }
            Node::Extrude { outline, .. } => vec![outline.curve_mut()],
            Node::Revolve { outline, .. } => vec![outline.curve_mut()],
            // **Ends or a frame, never both**, which is what lets this be one borrow rather
            // than two. Asked with the immutable `ends` first so that borrow is over before the
            // mutable one begins; collecting from each in turn is refused outright, because the
            // references handed back would both have to outlive the node they came from.
            other if !other.ends().is_empty() => {
                other.ends_mut().into_iter().filter_map(|e| e.source_mut()).collect()
            }
            other if !other.sweeps().is_empty() => other
                .sweeps_mut()
                .into_iter()
                .filter_map(|s| s.source_mut())
                .collect(),
            // A turn, for the three blocks that carry one. Asked immutably in the guard first,
            // so that borrow is over before the mutable one begins -- the same shape the ends
            // arm above uses and for the same reason.
            other if other.turn().is_some() => {
                other.turn_mut().and_then(|t| t.curve_mut()).into_iter().collect()
            }
            other => other
                .frames_mut()
                .into_iter()
                .filter_map(|f| f.curve_source_mut())
                .collect(),
        }
    }

    /// A copy of this node with every id it names pointing at the copy instead of the original.
    ///
    /// An input this does not know is left alone, which cannot happen for a subtree collected root
    /// last: everything a node consumes was inserted before it.
    ///
    /// **The edges go through the node's own lists, not a second copy of them.** This used to match
    /// on every kind and rebuild it field by field, with `other => other.clone()` underneath — so a
    /// kind whose edges it had not been told about was copied still pointing at the original.
    /// Places were exactly that: paste a sphere sitting on a point and the copy sat on the *first*
    /// sphere's point, so moving its own point did nothing to it. [`Self::remap_inputs`] and
    /// [`Self::curves_mut`] are the lists that get kept up to date, because they are written beside
    /// `inputs` and `curves` and have to agree with them (D9) — which no second spelling elsewhere
    /// ever will.
    pub fn remapped(
        &self,
        nodes: &std::collections::HashMap<NodeId, NodeId>,
        cages: &std::collections::HashMap<CageId, CageId>,
        curves: &std::collections::HashMap<CurveId, CurveId>,
    ) -> Node {
        let mut out = self.clone();
        // A cage and a curve are named by document id rather than by graph edge, so they are the
        // two this still has to say out loud.
        if let Node::Cage(c) = &mut out {
            *c = cages.get(c).copied().unwrap_or(*c);
        }
        for id in out.curves_mut() {
            *id = curves.get(id).copied().unwrap_or(*id);
        }
        out.remap_inputs(nodes);
        out
    }

    /// Whether this node is a body: a distance to a surface, of which the part is the union.
    ///
    /// [`Node::Ramp`] is not, because its value is a plain number: its zero set is wherever
    /// the mapped value happens to cross zero, which is not a shape anybody asked for.
    ///
    /// **Nor is [`Node::Point`]** -- the owner's decision, 2026-09-26. A point's field is the
    /// distance to it, never negative, so it adds nothing to a union and a traced ray has
    /// nothing to stop on; it is drawn as an overlay dot and is never picked. As a body it only
    /// cost: a body index, a term in the shader's union, and -- bodies being ordered by id -- a
    /// re-rank of every body already there each time one was made. A profile's joints are
    /// points nothing consumes, so every `join`, `ungroup` and paste was a cold shader compile
    /// for geometry that had not moved (`profile-group-spec.md` §4.5). A point a block reads --
    /// a place, a blend to a point -- was never a body either way.
    pub fn is_body(&self) -> bool {
        !matches!(self, Node::Ramp { .. } | Node::Point { .. })
    }

    /// Whether this node's field has an inside — somewhere it reads negative.
    ///
    /// **What a boolean's operands both have to have.** A point's field is the distance to it,
    /// so it is never negative and encloses nothing; combining one with a solid does one of
    /// three useless things, measured in `fieldform-field/examples/boolean_with_nothing.rs`.
    /// Union and difference leave the volume at 33764 mm3, exactly what it was, because `min`
    /// and `max` against a field that never goes negative cannot add or remove material — all
    /// either adds to the zero set is the single point, which is smaller than a pixel.
    /// Intersection reads 0 mm3: the body disappears, because a solid and something with no
    /// volume have none in common.
    ///
    /// It is not the same question as "is it a solid". A plane encloses a half-space, so
    /// intersecting with one trims a part, which is the whole reason `Node::Plane` exists and
    /// must keep working — the same measurement leaves 23159 mm3 of the ball.
    ///
    /// A ramp is a number rather than a distance, so it has no inside either, and `is_body`
    /// already says so for its own reasons.
    pub fn encloses_material(&self) -> bool {
        !matches!(self, Node::Point { .. } | Node::Ramp { .. })
    }
}

/// A visitor wearing the list it replaced: `push` hands one item over and `extend` each of several,
/// in order.
///
/// So a walk that used to build a list keeps its arms word for word, and with them the order it
/// handed things out in — which, for parameters, is an index something else may be holding.
struct AsList<'v, F>(&'v mut F);

impl<F> AsList<'_, F> {
    fn push<T>(&mut self, item: T)
    where
        F: FnMut(T),
    {
        (self.0)(item);
    }

    fn extend<T>(&mut self, items: impl IntoIterator<Item = T>)
    where
        F: FnMut(T),
    {
        items.into_iter().for_each(&mut *self.0);
    }
}
