// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The blend: the smooth minimum a fillet is made of, its bound and taper, and what it divides by.

use super::graph_lipschitz;
use fieldform_scene::{BooleanOp, Document, Graph, NodeId};

/// Smooth minimum: a fillet of radius `k` where two surfaces meet, at the continuity the
/// exponent selects.
///
/// ```text
/// f(a, b) = max(k, min(a, b)) − ( (k−a)₊ⁿ + (k−b)₊ⁿ )^(1/n)
/// ```
///
/// # Why the radius is the radius
///
/// At n = 2 this is a quarter circle: on two perpendicular walls the profile has constant
/// curvature 1/k along its whole length, and the surface sits where a ball of radius k rolled
/// into the corner would touch. Both measurements agree, which is what makes it honest to
/// call the number a radius.
///
/// The polynomial smooth minimum this replaced did not have that property. The same k
/// measured 0.707 k by curvature at the apex and 0.854 k by where the surface sat, because
/// the profile is not an arc at all — so there was no radius to report, only a blend size.
/// It also perturbed each surface across a band 2k wide, where on two walls meeting
/// transversally this one stops k from the join. `blend_profile` in fieldform-field's examples
/// reproduces both tables.
///
/// # The region it changes
///
/// ```text
/// Z = { p : a(p) < k  and  b(p) < k }
/// ```
///
/// Wherever *not both* operands read under `k`, one of `(k−a)₊`, `(k−b)₊` is zero, the n-norm
/// is the other gap, and the expression collapses to `min(a, b)`: the zero set is the hard
/// boolean's there, exactly.
///
/// **Z is not "within k of the join".** For two surfaces crossing transversally it is a band
/// about the join near the surface, which is the case the radius is named for. But it also
/// holds the whole overlap interior of two unioned bodies, and *any face the operands share or
/// nearly share*, however far from a join: on coplanar tops a = b, and the blended top sits
/// `k(1 − 2^(−1/n))` above the hard one — 0.293 k at G1 — over the whole overlap.
/// `a_shared_face_under_a_blend_rises_by_the_documented_amount` holds that number, and
/// the blend-integrity-spec design note §1 is the statement this paragraph follows.
///
/// # What this returns is not what a blend reports
///
/// This is the raw form, and inside Z its slope reaches 2^(1/n) where the operand gradients
/// are parallel -- and above a shared face it overstates *outside* Z as well, by the height
/// the face rose. [`blend`] reports [`blend_bound`] instead, which has this form's zero set and
/// never overstates; nothing but that, the tapering path and the tests that measure the
/// profile should call this directly.
pub(crate) fn smooth_min(a: f64, b: f64, k: f64, exponent: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let x = (k - a).max(0.0);
    let y = (k - b).max(0.0);
    let n = exponent;
    k.max(a.min(b)) - (x.powf(n) + y.powf(n)).powf(1.0 / n)
}

/// Bounds [`smooth_min`] over intervals of its operands.
///
/// The blend never exceeds a hard `min` — a fillet only ever adds material — so the upper end
/// is the hard union's. For the lower end, write `m` for `min(a, b)`: below `k` the blend is
/// `m` minus the amount the n-norm exceeds the larger of the two gaps, which is at most
/// `(2^(1/n) − 1)(k − m)`, and at or above `k` it is exactly `m`. That bound grows with depth
/// rather than being the constant `k/4` the polynomial blend allowed, which is the price of
/// the exponent family being exact at the surface — see [`smooth_min`].
pub(crate) fn smooth_min_interval(
    (a_lo, a_hi): (f64, f64),
    (b_lo, b_hi): (f64, f64),
    k: f64,
    exponent: f64,
) -> (f64, f64) {
    let hi = a_hi.min(b_hi);
    let lo = a_lo.min(b_lo);
    if k <= 0.0 {
        return (lo, hi);
    }
    let slack = (2f64.powf(1.0 / exponent) - 1.0) * (k - lo).max(0.0);
    (lo - slack, hi)
}

/// The blended form of each boolean.
///
/// Subtraction and intersection are the same fillet with signs flipped: subtracting `b` is
/// intersecting with its complement, and the complement of a distance field is its negation,
/// so a smooth maximum is a smooth minimum of the negated operands.
///
/// **Divided by [`blend_divisor`], which moves no surface.** In its zone the smooth minimum's
/// gradient is the two operand gradients under weights of unit conjugate norm, and where ∇a
/// and ∇b are parallel — a face the operands share — its magnitude is 2^(1/n), so the
/// undivided value overstated the distance by that factor on both sides of the raised face
/// (blend-integrity-spec §2.2). A positive constant leaves every zero crossing, and so every
/// fillet radius, exactly where it was.
///
/// `k_eff` is the radius *this point* gets — `k` itself unless a critical surface is near, see
/// [`tapered`] — and `taper` is what that costs the divisor.
pub(crate) fn blend(op: BooleanOp, a: f64, b: f64, k: f64, exponent: f64, k_eff: f64, taper: f64) -> f64 {
    // A part with critical marks tapers its fillets through a form of its own, and that form
    // keeps the constant divisor its proof was made for. Everything else is bounded exactly.
    let tapering = taper > 0.0 || k_eff < k;
    let form = |a: f64, b: f64| {
        if !tapering {
            blend_bound(a, b, k, exponent)
        } else if k_eff < k {
            smooth_min_tapering(a, b, k_eff, exponent)
        } else {
            smooth_min(a, b, k, exponent)
        }
    };
    let raw = match op {
        BooleanOp::Union => form(a, b),
        BooleanOp::Difference => -form(-a, b),
        BooleanOp::Intersection => -form(-a, -b),
    };
    if tapering { raw / blend_divisor(k, exponent, taper) } else { raw }
}

/// What a blend reports: the blend's own surface, and a value that never overstates the
/// distance to it, **without dividing anything**.
///
/// Option A of blend-integrity-spec §3 divided [`smooth_min`] by 2^(1/n) everywhere. It is safe,
/// and it compounds: a blend of a blend of a blend is short by that factor each time, so eight
/// bodies chained by seven fillets read 2^(7/2) = 11 times short, and the tracer took 27 million
/// readings of a scene that took 4 million unioned. Option B as the spec words it -- "exact
/// outside the zone Z, divided inside" -- is not sound: on the spec's own shared face, a point
/// 2k above it is outside Z, where the blend returns `min(a, b) = 2k`, and the raised surface
/// is 1.707k away. The undivided value overstates outside Z too.
///
/// This is what both were reaching for. **Read the blend as a function of the two operand
/// values.** Its zero set is a curve Γ in the (a, b) plane -- the ray `a = 0, b ≥ k`, the ray
/// `b = 0, a ≥ k`, and between them the arc `(k−a)ⁿ + (k−b)ⁿ = kⁿ` -- and this returns the
/// signed distance from `(a, b)` to Γ **in the max-norm**. Two lines make it a bound:
///
/// 1. A distance to a set, in any norm, changes by at most the norm of the step:
///    `|h(P) − h(Q)| ≤ ‖P − Q‖∞`.
/// 2. `a` and `b` each change by at most a millimetre a millimetre, so between two points of
///    space `‖(Δa, Δb)‖∞ ≤ |Δp|`.
///
/// So `h(a(p), b(p))` changes by at most a millimetre a millimetre, whatever the operands'
/// gradients are doing -- parallel, opposed or square -- and it is nought exactly on the
/// blended surface, so its size is never more than the distance to that surface. It is the
/// *largest* function of `(a, b)` with that property, so nothing built this way reads longer.
/// No divisor, so nothing to compound.
///
/// What it buys, against the raw form `f` and option A's `f / 2^(1/n)`:
///
/// - always `|f| / 2^(1/n) ≤ |h| ≤ |f|`, with `f`'s sign -- never worse than A, and the pair is
///   what [`smooth_min_interval`]'s hull below rests on;
/// - `h` **is the nearer operand** wherever the other is further by `k` or more
///   (`b ≥ |a| + k`), so away from a join a blend hands on the distance it was given, and a
///   chain of them is as cheap to march as a union;
/// - on a shared face it is exact: `a = b = 2k` reads 1.707k, which is the distance to the
///   raised face, where `f` says 2k and A says 1.41k.
///
/// The nearest point of Γ in the max-norm is found along the diagonal `(1, 1)`, since the arc
/// falls as it runs and a square growing about `(a, b)` first touches it with a corner. With
/// `x = k − a`, `y = k − b`, the diagonal keeps `x − y` and at the arc `x + y = √(2k² − (x−y)²)`
/// for n = 2 -- one square root. That closed form is why this is built for the circular (G1)
/// fillet only: n = 3 and n = 4 have their own, a cube root and two square roots, and each is
/// a function to add here when G2 chooses its exponent. **Until then any other exponent gets
/// option A's division**, which is sound and is what it had.
///
/// Still a `LowerBound` and reported as one: inside the join's reach it understates, by up to
/// 2^(1/n) across the middle of a fillet between square walls.
///
/// **A chamfer has a closed form too, and a simpler one** (exponent one). Its zero set's middle
/// piece is the segment `x + y = k` between the two rays, so the diagonal through the point
/// meets it where the gaps' sum is `k`: the max-norm distance is `|k − (x + y)| / 2`, crossing
/// exactly when the two gaps differ by no more than `k`. The spec's halving is this distance on
/// the diagonal and short of it off it, so the bevel is bounded here rather than divided.
pub(crate) fn blend_bound(a: f64, b: f64, k: f64, exponent: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    if exponent == 1.0 {
        let (x, y) = (k - a, k - b);
        let mut nearest = a.abs().max(y.max(0.0)).min(b.abs().max(x.max(0.0)));
        if (x - y).abs() <= k {
            nearest = nearest.min((k - (x + y)).abs() * 0.5);
        }
        let inside = x.max(0.0) + y.max(0.0) > k;
        return if inside { -nearest } else { nearest };
    }
    if exponent != 2.0 {
        return smooth_min(a, b, k, exponent) / 2f64.powf(1.0 / exponent);
    }
    let (x, y) = (k - a, k - b);
    // The two straight pieces of Γ: a distance along one axis, and how far short of the
    // piece's end the point is along the other.
    let mut nearest = a.abs().max(y.max(0.0)).min(b.abs().max(x.max(0.0)));
    // The arc, where the diagonal through the point crosses it -- which it does exactly when
    // the two gaps differ by no more than the radius.
    let across = x - y;
    if across.abs() <= k {
        let along = (2.0 * k * k - across * across).sqrt();
        nearest = nearest.min((along - (x + y)).abs() * 0.5);
    }
    // Inside the blended solid is past Γ: the gaps' norm exceeds the radius.
    let inside = x.max(0.0).hypot(y.max(0.0)) > k;
    if inside { -nearest } else { nearest }
}

/// How a profile solid's rims are cut, resolved: the reach typed, the shape's exponent and which
/// ends (`round-chamfer-spec.md` §4). A reach of nought is a hard rim and leaves every solid as
/// it was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rim {
    pub reach: f64,
    pub exponent: f64,
    pub ends: fieldform_scene::RimEnds,
    /// `Some(s)` where the two readings a rim joins have gradients square to each other, or
    /// nearly: the join is then the exact rounded corner of the plane (`cut`), read through
    /// `s ≥ 1`, which is what the angle between them can cost. `None` keeps the max-norm bound,
    /// which holds at any angle. The compile decides, since only it knows the solid.
    pub square: Option<f64>,
}

impl Default for Rim {
    fn default() -> Self {
        Rim { reach: 0.0, exponent: 2.0, ends: fieldform_scene::RimEnds::Both, square: None }
    }
}

impl Rim {
    pub fn is_hard(&self) -> bool {
        !(self.reach > 0.0)
    }

    /// Where the side wall meets an end, rounded or chamfered: the section's reading `x` and the
    /// slab's `y` intersected through the blend's own bound.
    ///
    /// **The spec's `prism(x + r, y + r) − r` and `max(x, y, (x + y + c)/2)` are this, to the
    /// same zero set** -- the rounded intersection is the blend's union with both signs turned,
    /// at n = 2 for the circle, 3 for G2, 1 for the bevel -- so a rim is one more reader of
    /// [`blend_bound`] rather than a second fillet written beside the first.
    ///
    /// **And the bound is why no new divisor is needed** where the spec's own argument would
    /// have wanted one. Its claim that the combination is 1-Lipschitz holds only while the two
    /// readings' gradients are square to each other -- true of an upright extrude, false under a
    /// draft, a lean, a taper or a loft, which is why every one of those takes the larger of the
    /// two rather than the corner. `blend_bound` is a distance in the max-norm of its two
    /// arguments, so it moves no faster than the faster of them *whatever angle their gradients
    /// meet at*: a solid already divided so each reading is a bound stays a bound, by the same
    /// divisor, with its rims rounded.
    pub fn join(&self, x: f64, y: f64) -> f64 {
        -blend_bound(-x, -y, self.reach, self.exponent)
    }

    /// The rim's reading of `x` against `y`: the exact corner where the readings are square
    /// (`square`), the max-norm bound where they are not.
    ///
    /// **Why the bound was not enough** (2026-10-18, the owner: "the occlusion looks wrong").
    /// [`blend_bound`] is a max-norm distance, and along the arc's own normal at forty-five
    /// degrees the max-norm reads `1/√2` of the Euclidean distance -- `0.707`, measured on every
    /// rim and every shape by `examples/rim_ao_reading.rs`. The still's occlusion steps out
    /// along the normal and takes a reading short of the step for nearby material, so every
    /// rounded rim wore a shadow of about forty per cent that nothing cast. Where the two
    /// readings' gradients are square to each other the exact rounded corner of the plane,
    /// composed with them, is a true distance along that normal and still 1-Lipschitz, so it
    /// reads `1.000` there; where they lean by `θ` its rate is at most `√(1 + cos θ)`, and
    /// dividing by that keeps it a bound. The bound stays for the solids whose readings lean
    /// far: a leaning or drafted extrude, a sweep, a loft.
    pub fn cut(&self, x: f64, y: f64) -> f64 {
        match self.square {
            None => self.join(x, y),
            Some(scale) => self.join_square(x, y, scale),
        }
    }

    /// The exact rounded corner of the `(x, y)` plane -- [`crate::primitive::edged`]'s two-axis
    /// form, which is the inset box grown by a ball at G1, the bevel between square faces at a
    /// chamfer, the cube norm at G2 -- read through `scale`, and never under the hard corner.
    ///
    /// **`y` is clamped at the reach before the corner is read, and what was clamped off is
    /// taken back.** The angle between the readings is only bounded near the cap -- a point
    /// `y` out from a cap plane at radius `r` leans by `y/r` -- so the corner is read as if the
    /// point stood no further out than the reach, where the lean is at most `k/r`, and the
    /// rest of the way out is subtracted; a distance can only grow by what a point moves, so
    /// that is still a bound. Beyond the reach the hard corner reads exactly anyway, and wins
    /// the `max`.
    fn join_square(&self, x: f64, y: f64, scale: f64) -> f64 {
        let k = self.reach;
        if !(k > 0.0) {
            return x.max(y);
        }
        let edge = crate::primitive::Edge { reach: k, exponent: self.exponent };
        // Clamped only where the readings lean (`scale > 1`): square readings are exact at any
        // distance, and clamping them would dull the far end for nothing.
        let yy = if scale > 1.0 { y.min(k) } else { y };
        let near = crate::primitive::edged(&[x, yy], edge) - (y - yy);
        (near / scale).max(x.max(y))
    }

    /// The same where only one end may be rounded: `slab` the solid's reading along its length,
    /// `|toward| − half`, and `toward` how far past the middle of that length the point sits,
    /// positive toward the end.
    ///
    /// **The rounded end's own slab joined, and the other end's taken hard**:
    /// `max(join(x, s_rounded), s_other)`. The slab is the larger of the two ends' readings,
    /// `toward − half` and `−toward − half`, so this is the same solid at the end it rounds and
    /// the hard one at the end it leaves. It is a bound by the same argument as [`Rim::join`],
    /// since the larger of two bounds is one. At the hard end the join reads exactly the section,
    /// because the rounded end's slab is the whole length away and the arc only reaches `k` --
    /// which is what the compile's refusal of a rim longer than the solid keeps true. Both ends is
    /// [`Rim::join`] itself, so a solid rimmed at both ends reads what it always did.
    ///
    /// The hard end does not take the corner outside both readings that a hard solid takes where
    /// they are square, so it reads short there: a sphere trace spends a few steps it would not
    /// have, and the rounded solid was a bound already.
    pub fn join_ends(&self, x: f64, slab: f64, toward: f64) -> f64 {
        let half = toward.abs() - slab;
        match self.ends {
            fieldform_scene::RimEnds::Both => self.cut(x, slab),
            fieldform_scene::RimEnds::End => self.cut(x, toward - half).max(-toward - half),
            fieldform_scene::RimEnds::Start => self.cut(x, -toward - half).max(toward - half),
        }
    }

    /// How much of the length a rim may take: half of it with both ends rounded, where the two
    /// roundings would otherwise meet, and all of it with one.
    pub fn room_along(&self, length: f64) -> f64 {
        match self.ends {
            fieldform_scene::RimEnds::Both => length * 0.5,
            _ => length,
        }
    }
}

/// The radius a fillet gets at a point `gap` millimetres clear of the nearest critical surface:
///
/// ```text
/// k_eff = clamp(gap, 0, k)          gap = min over marks ( |F_crit| − clearance )
/// ```
///
/// Nought on the critical surface and across its clearance band, the whole radius once the gap
/// exceeds it, and in between a fillet that thins as it approaches (`blend-integrity-spec.md`
/// §6). `gap` is infinite in a part with no marks, so this is `k` and nothing is tapered.
pub(crate) fn tapered(k: f64, gap: f64) -> f64 {
    gap.max(0.0).min(k)
}

/// [`smooth_min`] for a radius that is on its way to nought, without the jump.
///
/// `smooth_min` answers `min(a, b)` for `k ≤ 0`, and that is *not* the limit of its own form:
/// where both operands are negative — the overlap interior, which is inside Z however small
/// `k` is — the form tends to `−‖(a, b)‖ₙ` and `min` is `−max(|a|, |b|)`. For a radius that
/// is simply typed as nought that does not matter, since nothing is on either side of it. For
/// one that *reaches* nought at the edge of a clearance band it is a step in the value across
/// that edge, up to 0.414 of the depth at G1 — invisible from outside, where the value is
/// positive, and a ledge in the cavity wall of anything that shells or insets the part.
///
/// So a tapering blend keeps the form all the way down. At `k_eff = 0` it reads
///
/// ```text
/// f₀(a, b) = max(0, min(a, b)) − ‖((−a)₊, (−b)₊)‖ₙ
/// ```
///
/// which is `min(a, b)` exactly wherever the operands are not both negative, is negative
/// exactly where `min(a, b)` is, and is zero exactly where `min(a, b)` is. **Its surface is the
/// hard boolean's surface, point for point**; only the magnitude deep inside the overlap
/// differs. That is the sense in which a protected face is exact, and it is the sense that
/// matters — see the proof beside [`taper_ratio`].
fn smooth_min_tapering(a: f64, b: f64, k_eff: f64, exponent: f64) -> f64 {
    let x = (k_eff - a).max(0.0);
    let y = (k_eff - b).max(0.0);
    k_eff.max(a.min(b)) - (x.powf(exponent) + y.powf(exponent)).powf(1.0 / exponent)
}

/// What a tapering fillet's divisor has to allow for, as a ratio: how fast the gap to the
/// critical surfaces can change over how fast this fillet's own operands can. Nought in a part
/// with no critical marks, which is what leaves [`blend_divisor`] at its §3 value.
///
/// **Asked of the graph, by the field and the shader alike**, the way a driven offset's divisor
/// is: two paths that worked it out separately and differed would put the surface in the same
/// place and scale the field around it, which no residual shows.
///
/// # What is proved, and what "exact" means (spec §6.1 and §6.2)
///
/// Write `κ = clamp(g, 0, k)` for the radius at a point, `g` the gap, and
/// `f(a, b, κ) = max(κ, m) − ‖((κ−a)₊, (κ−b)₊)‖ₙ` with `m = min(a, b)`.
///
/// 1. **On the clearance band the surface is the hard one.** There `g ≤ 0`, so `κ = 0` and
///    `f = f₀`, whose sign and zero set are those of `m` ([`smooth_min_tapering`]). Every
///    rounded blend in the part is in that state at that point, so by induction up the graph
///    each one's sign is the hard boolean's of its operands' signs, and the part's surface
///    within the band is the hard part's — *all* of it, the air side included. Nothing enters
///    the band that the hard part does not put there.
/// 2. **Beyond the band a fillet intrudes by at most `(2^(1/n) − 1)·d`**, `d` the distance
///    past the band's edge. Material is added where `f < 0 ≤ m`. For `m ≥ κ`, `f = m`. For
///    `0 ≤ m < κ` the larger of the two gaps is `κ − m` and the n-norm is at most `2^(1/n)`
///    times it, so `m − f ≤ (2^(1/n) − 1)(κ − m) ≤ (2^(1/n) − 1)κ`, and `κ ≤ g ≤ d·L_g`.
///    So added material lies where the hard part reads under `0.414·d` at G1 — a wedge that
///    closes to nothing at the band. Per blend; nested blends add, each by its own `κ`.
/// 3. **The value still never overstates.** In Z, `∂f/∂κ = 1 − s` with
///    `s = (xⁿ⁻¹ + yⁿ⁻¹)/‖(x, y)‖ₙⁿ⁻¹ ∈ [1, 2^(1/n)]`, so `|∂f/∂κ| ≤ 2^(1/n) − 1`; outside Z
///    it is nought. `κ` is a clamp of `g` and moves no faster than `g` does, `L_g`. With the
///    operands moving at most `L` — and *continuity in κ all the way to nought*, which is what
///    [`smooth_min_tapering`] is for —
///
///    ```text
///    |∇f| ≤ 2^(1/n)·L + (2^(1/n) − 1)·L_g
///    ```
///
///    Divided by `2^(1/n) + (2^(1/n) − 1)·L_g/L`, the blend moves no faster than its steeper
///    operand, which is what `lipschitz_of` reports for it. `L_g/L` is this function; at one,
///    the divisor is `2^(1+1/n) − 1 ≈ 1.83` at G1, the spec's figure.
pub fn taper_ratio(document: &Document, graph: &Graph, node: NodeId) -> f64 {
    let gap_rate = graph
        .critical()
        .iter()
        .map(|mark| graph_lipschitz(document, graph, mark.node))
        .fold(0.0, f64::max);
    if gap_rate <= 0.0 {
        return 0.0;
    }
    let own = graph
        .node(node)
        .map(|n| n.inputs())
        .unwrap_or_default()
        .into_iter()
        .map(|input| graph_lipschitz(document, graph, input))
        .fold(0.0, f64::max);
    // An operand that does not move at all cannot be what bounds the blend; one is the rate
    // every leaf here reports, and the honest floor.
    gap_rate / own.max(1e-9)
}

/// What a rounded blend's value is divided by: 2^(1/n), or 1 for a hard join.
///
/// Option A of blend-integrity-spec §3 — one constant, everywhere, including far outside the
/// zone where the undivided value was already exact. That is the known cost, and the zone-local
/// divisor that buys it back (option B) has to be proved before it is substituted.
///
/// `taper` is [`taper_ratio`], nought in a part with no critical marks.
pub(crate) fn blend_divisor(k: f64, exponent: f64, taper: f64) -> f64 {
    // [`blend_bound`] divides by nothing: a circular fillet in a part with no critical marks
    // costs the tracer no steps, and a chain of them costs none either.
    if k <= 0.0 || (taper <= 0.0 && exponent == 2.0) {
        return 1.0;
    }
    let steepest = 2f64.powf(1.0 / exponent);
    steepest + (steepest - 1.0) * taper
}

/// How fast a blend's reading can change per millimetre of *radius*, which is what a radius
/// following a ramp costs ([`super::lipschitz::driven_blend_divisor`]).
///
/// [`blend_bound`] at G1 and at the chamfer is the max-norm distance to the zero set Γ(k), and
/// Γ moves by at most `dk` when `k` does: its two rays end at `(0, k)` and `(k, 0)`, and the arc
/// between them is `(k − k·u, k − k·v)` over the unit curve `uⁿ + vⁿ = 1`, whose rate in `k` is
/// `(1 − u, 1 − v)` with both parts in `[0, 1]`. A distance to a set moves by no more than the
/// set does, so the rate is **one**.
///
/// Every other form is [`smooth_min`] over its divisor: `max(k, min(a, b)) − ‖((k−a)₊, (k−b)₊)‖ₙ`,
/// whose first term rises at most one in `k` and whose norm rises at most `2^(1/n)` (Hölder:
/// `x^(n−1) + y^(n−1) ≤ 2^(1/n) · ‖(x, y)‖ₙ^(n−1)`), so `1 + 2^(1/n)` before the division by
/// `2^(1/n)` and **`1 + 2^(−1/n)`** after it -- 1.71 at G1's tapering form, 1.79 at G2. The
/// tapering form keeps the constant divisor its proof was made for, so a part with critical
/// marks takes this rate at every exponent.
pub(crate) fn blend_radius_rate(exponent: f64, tapers: bool) -> f64 {
    if !tapers && (exponent == 1.0 || exponent == 2.0) {
        1.0
    } else {
        1.0 + 2f64.powf(-1.0 / exponent)
    }
}

/// What a rounded pattern's value is divided by: W^(1/n), W the instances its window sums.
///
/// The n-ary version of [`blend_divisor`]. `2·reach + 1` cells are walked unless that covers
/// the pattern, in which case every instance is; and the copies a block leaves out — skipped, or
/// the body's own when the block does not hold it — are not summed. The shader computes the same
/// number from the same three slots.
///
/// Tapered, it is the same argument as [`taper_ratio`]'s third part with `W^(1/n)` for
/// `2^(1/n)`: the form's rate in `κ` is at most `W^(1/n) − 1`.
pub(crate) fn pattern_join_divisor(
    reach: usize,
    instances: usize,
    left_out: usize,
    exponent: f64,
    taper: f64,
) -> f64 {
    let held = instances - left_out;
    let summed = (2 * reach + 1).min(held).max(1);
    let steepest = (summed as f64).powf(1.0 / exponent);
    steepest + (steepest - 1.0) * taper
}
