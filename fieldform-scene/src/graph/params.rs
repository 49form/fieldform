// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The values a node's fields hold: parameter references, attractors, ramps, fillets, booleans.

use crate::id::{CurveId, NodeId, ParamId};
use crate::turn::Turn;
use serde::{Deserialize, Serialize};
#[cfg(doc)] use super::Node;

/// A value a node parameter can take: a fixed number, a reference to a named document
/// parameter so one dimension can drive several features, or a value that varies through space.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamRef {
    Literal(f64),
    Named(ParamId),
    /// A [`Node::Ramp`], read at whatever point the field is being evaluated at.
    ///
    /// **Only a ramp.** Any field could supply a number, but a ramp is the one construction
    /// whose rate of change is known in closed form — `k·|Δout|/|Δin|` times its input's own
    /// Lipschitz constant, with `k` fixed by the ramp's shape. That bound is not decoration:
    /// a parameter that varies with position stops the node it drives from being a distance
    /// field, and `Normalization::LowerBound` is only statable if there is a number to divide
    /// by. An arbitrary field carries no such number, so a driven parameter would have to
    /// report a bound it could not justify. Compilation refuses anything else and says so.
    ///
    /// The node is consumed like any other operand, so a ramp is not a body.
    Field(NodeId),
}

impl ParamRef {
    /// Resolves against a lookup of named parameter values.
    ///
    /// A field-driven reference has no single value, so it resolves to `None` — the same
    /// answer an unresolvable name gives. Callers that could accept one check for it first;
    /// the rest already refuse `None`, which is the behaviour that keeps a driven parameter
    /// from being silently read as zero.
    pub fn resolve(&self, lookup: &dyn Fn(ParamId) -> Option<f64>) -> Option<f64> {
        match self {
            ParamRef::Literal(v) => Some(*v),
            ParamRef::Named(id) => lookup(*id),
            ParamRef::Field(_) => None,
        }
    }

    /// The node this reference reads, if it is field-driven.
    pub fn driver(&self) -> Option<NodeId> {
        match self {
            ParamRef::Field(id) => Some(*id),
            _ => None,
        }
    }
}

/// What an attractor draws a body toward.
///
/// **One thing, not a list.** A body is pulled toward *something*, and that something is a curve
/// or a point — never both at once, which is what makes this a choice rather than a collection.
/// Two targets would need a rule for combining two pulls, and there is no rule anybody would
/// guess.
///
/// Both reach the warp as a [`crate::Spine`]: a polyline for a curve, and a single vertex for a
/// point. The search that finds the nearest place on a spine already answers a one-vertex spine
/// with that vertex, on both the CPU and in the shader, so a point is not a special case in the
/// arithmetic — only in what the picker offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Attractor {
    /// A curve in the document, tessellated into the spine.
    Curve(CurveId),
    /// A [`Node::Point`], whose position is the whole spine.
    Point(NodeId),
}

impl Attractor {
    /// The block this names, when it names one — so [`Node::inputs`] can make the graph edge.
    pub(crate) fn node(self) -> Option<NodeId> {
        match self {
            Attractor::Point(id) => Some(id),
            Attractor::Curve(_) => None,
        }
    }

    /// The curve this names, when it names one — so [`Node::curves`] can report it.
    pub(crate) fn curve(self) -> Option<CurveId> {
        match self {
            Attractor::Curve(id) => Some(id),
            Attractor::Point(_) => None,
        }
    }
}

/// Whether a turn is the identity, which is what decides if it is written to a file at all.
///
/// Zero in all three slots, and only when they are literals: a named or driven angle is a fact
/// about the document even when it currently reads zero. A turn taken from a block or a curve is
/// never none — it says something, whatever it currently resolves to.
pub(crate) fn turn_is_none(turn: &Turn) -> bool {
    turn.angles()
        .is_some_and(|a| a.iter().all(|r| matches!(r, ParamRef::Literal(v) if *v == 0.0)))
}

/// A turn of nothing, for a block that has not been turned.
pub fn no_turn() -> Turn {
    Turn::By([ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)])
}

/// How a [`Node::Ramp`] gets from one end of its output range to the other.
///
/// The three differ in how fast they can move, and that number is the reason the choice is
/// exposed rather than fixed: a ramp's slope with respect to its input is at most
/// `slope_factor()` times `|Δout| / |Δin|`, and that bound is what lets a driven parameter
/// declare an honest `LowerBound` instead of claiming a distance it does not have.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RampShape {
    /// Straight between the ends, clamped outside them. Steepest for a given range is 1.
    Linear,
    /// `3t² − 2t³` — flat at both ends, so a driven thickness meets its limits smoothly.
    Smooth,
    /// `6t⁵ − 15t⁴ + 10t³` — flat in curvature as well, at the cost of being steeper in the
    /// middle.
    Smoother,
}

impl RampShape {
    pub const ALL: [RampShape; 3] = [RampShape::Linear, RampShape::Smooth, RampShape::Smoother];

    /// What to call each shape in a panel.
    ///
    /// **The continuity each one delivers, which is what they differ by.** Linear reaches its
    /// ends with a slope, so the value is continuous and its rate is not: G0. Smoothstep's
    /// derivative is zero at both ends, so the rate is continuous too: G1. Smootherstep's second
    /// derivative is zero as well: G2. "Smooth" and "Smoother" were a comparative with nothing
    /// to compare against, and this application already spells continuity G0/G1/G2 where a blend
    /// asks for it -- one vocabulary for one idea (D9).
    pub fn label(self) -> &'static str {
        match self {
            RampShape::Linear => "G0",
            RampShape::Smooth => "G1",
            RampShape::Smoother => "G2",
        }
    }

    /// The steepest this shape gets, as a multiple of the straight line through the same ends.
    ///
    /// Exact, not estimated. `s(t) = 3t² − 2t³` has `s′ = 6t(1 − t)`, which peaks at `t = ½`
    /// with value **1.5**; `s(t) = 6t⁵ − 15t⁴ + 10t³` has `s′ = 30t²(1 − t)²`, peaking at the
    /// same place with value **1.875**. Both are checked by a test that differences the shape
    /// itself, because a hand-derived constant that is too small is a bound that does not
    /// bound and every consumer of it steps too far.
    pub fn slope_factor(self) -> f64 {
        match self {
            RampShape::Linear => 1.0,
            RampShape::Smooth => 1.5,
            RampShape::Smoother => 1.875,
        }
    }

    /// Maps `t` in `[0, 1]` onto `[0, 1]`, clamping outside.
    pub fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            RampShape::Linear => t,
            RampShape::Smooth => t * t * (3.0 - 2.0 * t),
            RampShape::Smoother => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        }
    }
}

/// How smoothly a blend meets the surfaces it joins.
///
/// The names are the surfacing convention: G0 is position only (a hard boolean — not a blend
/// at all, which is why it has no variant here), G1 adds a common tangent, G2 adds matching
/// curvature.
///
/// One function covers the whole family:
///
/// ```text
/// f(a, b) = max(k, min(a, b)) − ( (k−a)₊ⁿ + (k−b)₊ⁿ )^(1/n)
/// ```
///
/// n = 2 is a quarter circle of radius exactly k — measured constant curvature 1/k along the
/// whole profile — meeting each wall at a curvature step from 0 to 1/k, which is what makes a
/// circular fillet tangent-continuous and no more. n > 2 sends the curvature to zero at the
/// join instead (0.0002 against 0.1 at n = 3, on a k = 10 corner), so G2 is this same
/// expression with a different exponent rather than a different kind of blend.
///
/// That is why the exponent is not spelled out in the file: the vocabulary grows by value, and a
/// build that does not know one preserves the node and refuses to evaluate it, like any other
/// node it cannot read.
///
/// **What the number typed means, for every member: reach.** It is how far from the *other* face
/// the rounding leaves this one — `f(a, b) = 0` passes through `(a, b) = (0, k)` and `(k, 0)`
/// whatever the exponent and whatever angle the faces meet at, so changing the shape never moves
/// where the rounding starts. Ruled 2026-10-02, on a measurement: at a right angle the tightest
/// bend inside is `k` for G1 and 0.561·k for G2 ([`Continuity::apex_radius_ratio`]), so a typed
/// *tightest radius* would move the footprint 1.78× on switching from one to the other; and off a
/// right angle even G1 is not a circle (apex radius 0.47·k at 60°, 2.45·k at 120°), so "radius"
/// was only ever the reach read at a right angle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Continuity {
    /// Tangent-continuous: an exact circular fillet of the radius asked for, at a right angle.
    G1,
    /// Curvature-continuous: curvature reaches nought at the join instead of stepping to 1/k,
    /// at the price of a tighter bend in the middle — 0.561·k at a right angle against G1's k.
    G2,
    /// A flat bevel: each face cut back `k` from the other, and a plane between the two cuts.
    ///
    /// **The same form at exponent one, not a form of its own** (`round-chamfer-spec.md` §2).
    /// `max(k, min(a, b)) − ((k − a)₊ⁿ + (k − b)₊ⁿ)^(1/n)` at n = 1 is `a + b − k` inside the
    /// join and the nearer operand outside it, which is the spec's bevel `(a + b − k)` to the
    /// same zero set; the reach is the bevel's leg by the property every member shares; and the
    /// divisor `2^(1/n)` the family already carries is the spec's halving. So everything that
    /// reads an exponent reads this one, and the shader's existing functions draw it.
    ///
    /// **Not a continuity, and the name is a wart recorded rather than fixed** (ruled
    /// 2026-09-30): a chamfer is not continuous at all. Renaming the serialized field would be a
    /// format change for a word, so it is written `"chamfer"` under the existing `"continuity"`
    /// key, and a build that has not heard of it keeps the node and refuses it (D3). The panel
    /// calls the row *Shape*.
    Chamfer,
}

impl Continuity {
    /// The exponent this continuity blends with.
    pub fn exponent(self) -> f64 {
        match self {
            Continuity::G1 => 2.0,
            Continuity::G2 => 3.0,
            Continuity::Chamfer => 1.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Continuity::G1 => "G1",
            Continuity::G2 => "G2",
            Continuity::Chamfer => "Chamfer",
        }
    }

    /// Every shape, in the order a picker offers them.
    pub const ALL: [Continuity; 3] = [Continuity::G1, Continuity::G2, Continuity::Chamfer];

    /// The tightest bend inside a rounding of reach `k` between two faces at a right angle, as a
    /// multiple of `k`: `2^(1/2 − 1/n) / (n − 1)`.
    ///
    /// Derived from the curvature of `(k−a)ⁿ + (k−b)ⁿ = kⁿ` at the diagonal, where `a = b`:
    /// `κ = (n − 1)·2^(1/n) / (√2·k)`. One at G1 — the circle — and 0.5612 at G2, 0.3964 at
    /// n = 4; `fieldform-field/examples/blend_profile.rs` measures the same figures from the
    /// field itself. **Only a right angle**: the number is a property of the corner as well as
    /// of the exponent, so the panel says "at a right angle" beside it.
    ///
    /// **Nought for a chamfer**, whose two creases are where the bevel meets each face: the
    /// tightest bend there is a corner, and the formula's `n − 1` would divide by it.
    pub fn apex_radius_ratio(self) -> f64 {
        if self == Continuity::Chamfer {
            return 0.0;
        }
        let n = self.exponent();
        2f64.powf(0.5 - 1.0 / n) / (n - 1.0)
    }
}

/// A rounded join: the radius to round it by, and the continuity to round it at.
///
/// **One type, because two blocks are asking the same question (D9).** A blend rounds where two
/// bodies meet; a pattern rounds where its copies meet. Both are "make this join a fillet of
/// radius k", and the numbers mean the same thing in both — so they are the same type, edited by
/// one editor, rather than two spellings that drift.
///
/// A radius of zero is a hard join and costs nothing: [`smooth_min`] returns the plain minimum,
/// so there is no `Option` here and no branch in the generated shader.
///
/// **The rule that travels with it.** An approximate blend must announce itself where its radius
/// is typed — `approximate_blends` lists the joins whose radius will not be the radius asked
/// for, and a block gaining a `Fillet` gains that obligation. Silently allowing one is the
/// failure the old refusal guarded against.
///
/// [`smooth_min`]: https://example.invalid
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fillet {
    /// How far back from the join the rounding starts, in millimetres. Zero is a hard join.
    ///
    /// With a chamfer of two legs this is the first face's leg: how far along face `a` (a
    /// blend's first operand, a rim's side wall) the bevel reaches from the other face.
    pub radius: ParamRef,
    /// Which continuity the rounding has, which is the exponent the smooth minimum uses.
    pub continuity: Continuity,
    /// A chamfer's second leg, where its two legs differ: how far along face `b` (a blend's
    /// second operand, a rim's end) the bevel reaches from the first. `None` is the one number
    /// both ways, which is every fillet written before this existed and every rounded one
    /// still -- a round has one radius, and only a chamfer may take two
    /// (`round-chamfer-spec.md` §9.2).
    pub second: Option<ParamRef>,
}

/// A hard join, which is what every block carrying a `Fillet` was before it had one.
impl Default for Fillet {
    fn default() -> Self {
        Fillet::hard()
    }
}

impl Fillet {
    /// A hard join: no rounding at all.
    pub const fn hard() -> Self {
        Fillet {
            radius: ParamRef::Literal(0.0),
            continuity: Continuity::G1,
            second: None,
        }
    }

    /// A rounded join of this radius.
    pub const fn of(radius: ParamRef) -> Self {
        Fillet {
            radius,
            continuity: Continuity::G1,
            second: None,
        }
    }

    /// The exponent the smooth minimum raises its terms to.
    pub fn exponent(&self) -> f64 {
        self.continuity.exponent()
    }

    /// Whether this join is hard as far as the graph can tell without resolving anything.
    ///
    /// A literal zero is hard; a radius that names a parameter is not answerable here, and is
    /// treated as rounded — which is the safe direction, since the only cost of being wrong is
    /// one more number in a buffer.
    pub fn is_plainly_hard(&self) -> bool {
        matches!(self.radius, ParamRef::Literal(r) if r <= 0.0)
    }

    /// The second leg a chamfer of two legs has, and nothing for any other shape: a `second`
    /// left on a fillet whose shape was switched back to a round is not read, so switching
    /// the shape never moves the seam.
    pub fn second_leg(&self) -> Option<&ParamRef> {
        match self.continuity {
            Continuity::Chamfer => self.second.as_ref(),
            _ => None,
        }
    }

    /// The parameters this fillet reads, for the graph's wiring: the radius, and the second leg
    /// where there is one. An iterator and not a list, because the walk that asks who reads a
    /// block runs on every edit and allocates nothing.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        std::iter::once(&self.radius).chain(self.second.as_ref())
    }

    /// The same, to edit.
    pub(crate) fn params_mut(&mut self) -> impl Iterator<Item = &mut ParamRef> {
        std::iter::once(&mut self.radius).chain(self.second.as_mut())
    }
}

/// Which ends of a profile solid its rims round: both, or one.
///
/// **Start and end are the solid's own, along the direction it was made in.** An extrude of one
/// side starts at the profile's plane and ends at its far face, whichever way a negative depth
/// flips it; one of two sides, or taken both ways, starts at side two's face and ends at side
/// one's; a sweep starts where its path starts; a loft at its first section; a partial revolve
/// at the cap its angle is measured from. The owner asked
/// for it on 2026-10-17, having found both ends rounded with no way to keep one sharp.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RimEnds {
    /// Both ends alike, which is what every rimmed solid was before this existed.
    #[default]
    Both,
    /// Only the end the solid starts from.
    Start,
    /// Only the end it finishes at.
    End,
}

impl RimEnds {
    /// Every value, in the order a panel offers them.
    pub const ALL: [RimEnds; 3] = [RimEnds::Both, RimEnds::Start, RimEnds::End];

    /// What a panel calls it.
    pub fn label(self) -> &'static str {
        match self {
            RimEnds::Both => "Both",
            RimEnds::Start => "Start",
            RimEnds::End => "End",
        }
    }
}

/// A profile solid's rims: how far back and in what shape each end is cut, and which ends.
///
/// **A fillet's two numbers and one more**, kept flat rather than nesting a [`Fillet`] so that
/// everything which reads a rim's radius or shape reads it as it reads a fillet's. Extrude, sweep
/// and loft all take this one type (D9), and one editor shows it.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rims {
    /// How far back from each end the side starts to turn, in millimetres. Zero is hard.
    pub radius: ParamRef,
    /// The shape of the cut: G1, G2 or a chamfer.
    pub continuity: Continuity,
    /// Which ends are cut.
    pub at: RimEnds,
    /// A chamfer's second leg: how far in from the side each end is cut, where that differs from
    /// how far back from the end the side is. `None` is one number both ways
    /// ([`Fillet::second`]).
    pub second: Option<ParamRef>,
}

/// Hard, which is what every profile solid was before it had rims.
impl Default for Rims {
    fn default() -> Self {
        Rims::hard()
    }
}

impl From<Fillet> for Rims {
    /// The fillet's cut at both ends.
    fn from(fillet: Fillet) -> Self {
        Rims {
            radius: fillet.radius,
            continuity: fillet.continuity,
            at: RimEnds::Both,
            second: fillet.second,
        }
    }
}

impl Rims {
    /// No rounding at either end.
    pub const fn hard() -> Self {
        Rims {
            radius: ParamRef::Literal(0.0),
            continuity: Continuity::G1,
            at: RimEnds::Both,
            second: None,
        }
    }

    /// The radius and shape alone, as a blend's fillet would carry them.
    pub fn fillet(&self) -> Fillet {
        Fillet { radius: self.radius, continuity: self.continuity, second: self.second }
    }

    /// The exponent the cut's shape blends with.
    pub fn exponent(&self) -> f64 {
        self.continuity.exponent()
    }

    /// Whether these rims are hard as far as the graph can tell without resolving anything --
    /// [`Fillet::is_plainly_hard`]'s answer, since which ends are cut changes nothing of it.
    pub fn is_plainly_hard(&self) -> bool {
        self.fillet().is_plainly_hard()
    }

    /// The parameters these rims read, for the graph's wiring: the radius, and the second leg
    /// where there is one. An iterator, as [`Fillet::params`] is, so the walk allocates nothing.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        std::iter::once(&self.radius).chain(self.second.as_ref())
    }

    /// The same, to edit.
    pub(crate) fn params_mut(&mut self) -> impl Iterator<Item = &mut ParamRef> {
        std::iter::once(&mut self.radius).chain(self.second.as_mut())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BooleanOp {
    Union,
    Difference,
    Intersection,
}
