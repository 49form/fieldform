// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The attractor warp: geometry drawn toward a curve, fading out with distance.
//!
//! One place for the arithmetic, because four things read it and they must agree exactly: the
//! compiled op evaluates it, the extent walk sizes a box from it, the shader repeats it in
//! WGSL, and the panel quotes its constants back to the user in millimetres. A falloff written
//! twice is a CPU picture and a GPU picture of different shapes, which is the failure
//! `gpu_matches_cpu` exists to catch and the reason not to write it twice in the first place.
//!
//! # Which way it pushes
//!
//! [`warp_point`] moves the sample point **away** from the curve, and the material therefore
//! moves **toward** it. That is the standard construction for every SDF deformation — a twist
//! unwinds, a bend straightens — and it is the inverse of the visible displacement: to know
//! what the pulled shape looks like at `p`, ask what the original shape looked like at the
//! point that got pulled in to occupy `p`, which lies further out.
//!
//! # Why this falloff
//!
//! `A(d) = 1 − 3x² + 2x³` for `x = d/R`, the standard smoothstep run backwards: exactly 1 on
//! the curve, exactly 0 at `R`, and flat at both ends so there is no crease where the influence
//! stops. An inverse-square law reads more like the word "attractor" and was rejected for two
//! reasons that are not aesthetic: it is singular on the curve itself, and it never reaches
//! zero — so the warp's support would be the whole of space, and a field whose effect has no
//! edge cannot report a bounding box. Both the mesher and the ray marcher size themselves from
//! that box.
//!
//! Choosing it fixes two numbers that would otherwise have to be guessed per model, and both
//! are properties of the polynomial rather than of any geometry — see [`MAX_PULL_FRACTION`] and
//! [`FOLD_STRENGTH`].

use nalgebra::Point3;

/// The largest displacement the warp can produce, as a fraction of `strength · falloff`.
///
/// A point at `d` is pushed out to `d · (1 + s·A(d))`, so it moves by `s · R · x·A(x)` for
/// `x = d/R`. That peaks where `d/dx [x·A(x)] = 1 − 9x² + 8x³` first vanishes, at
/// `x = (1 + √33)/16 ≈ 0.4215` — a little under halfway out — and the peak is this number.
///
/// **This is what makes the node's bounding box tight rather than roughly right.** The spec
/// suggested reporting the target's box grown by the whole falloff radius; nothing ever moves
/// that far, and growing by `0.26 · strength · falloff` is a box about four times smaller in
/// each direction at `strength = 1`. `an_attracted_body_stays_inside_the_box_it_reports` marches
/// the field to confirm the smaller one still holds.
pub(crate) const MAX_PULL_FRACTION: f64 = 0.259_973_837_079_779;

/// Where that peak is, as a fraction of the falloff radius: `(1 + √33)/16`.
///
/// Exported because the panel quotes it — "moves the surface up to 6 mm, at about 17 mm out" —
/// and a second copy of `0.4215` written there is the thing the module note at the top warns
/// against. One number, one place.
pub const PEAK_PULL_AT: f64 = 0.421_535_165_408_626_8;

/// The strength at which a *pull* starts folding — mapping two places onto one.
///
/// Radially the warp scales lengths by `1 + s·(1 − 9x² + 8x³)`, whose bracket bottoms out at
/// `−11/16` at `x = 3/4`. So the factor reaches zero at `s = 16/11 ≈ 1.4545`, and beyond that
/// the shell of material three-quarters of the way out is turned inside out.
///
/// **A property of the falloff, not of the model**, which is why it can be a constant at all:
/// it does not depend on the curve, the body, or the radius. §8 of the spec expected the
/// strength slider to be clamped to a guessed range instead; there is a real number here, so
/// the panel warns at it rather than guessing. Folding is not refused — every comparable
/// deformer ships with it, and a strong pull that folds somewhere out of sight is a legitimate
/// thing to ask for. It is announced.
pub const FOLD_STRENGTH: f64 = 16.0 / 11.0;

/// The same, for a *push* — a negative strength.
///
/// Not the same number, which is the reason it is written down rather than assumed symmetric.
/// A push shrinks lengths by `1 + s·(1 − 9x² + 8x³)` with `s` negative, and that bracket is
/// largest at the curve itself, where it is 1 — so the factor reaches zero at `s = −1`, well
/// before a pull's `16/11`. Folding a push means the material closest to the curve arriving
/// from the wrong side.
pub const FOLD_PUSH_STRENGTH: f64 = -1.0;

/// Whether the warp maps two places onto one at this strength.
///
/// Between [`FOLD_PUSH_STRENGTH`] and [`FOLD_STRENGTH`] it does not, whatever the falloff, the
/// curve or the body — the bound is a fact about the polynomial. Outside them it does
/// *somewhere*, which is not the same as visibly: the fold may lie inside solid material or
/// outside the body entirely. So this is what the panel warns from, not what anything refuses.
pub fn folds(strength: f64) -> bool {
    strength > FOLD_STRENGTH || strength < FOLD_PUSH_STRENGTH
}

/// How much of the pull is felt at distance `d` from the curve: 1 on it, 0 at `radius` and
/// beyond.
///
/// A non-positive radius reaches nothing, rather than dividing by zero.
pub(crate) fn falloff(d: f64, radius: f64) -> f64 {
    if radius <= 0.0 {
        return 0.0;
    }
    let x = (d / radius).clamp(0.0, 1.0);
    1.0 - x * x * (3.0 - 2.0 * x)
}

/// Where to sample the unwarped field, given the point asked about and the nearest point on the
/// curve to it.
///
/// Away from the curve, which draws the material toward it — see the module note. Returns `p`
/// unchanged when there is no warp to apply: zero strength, no reach, or a point already
/// beyond the falloff. Exactly unchanged, not nearly: `strength = 0` has to be a no-op that a
/// comparison can see, since a body under a switched-off attractor is one somebody is about to
/// compare against.
pub fn warp_point(p: Point3<f64>, closest: Point3<f64>, strength: f64, radius: f64) -> Point3<f64> {
    let away = p - closest;
    let w = strength * falloff(away.norm(), radius);
    if w == 0.0 {
        return p;
    }
    p + away * w
}

/// The warped surface's normal, from the unwarped body's normal at the warped point.
///
/// **A warp moves a surface and turns it, and those are not the same transformation.** The
/// field is `F(g(p))`, so its gradient is `Dg^T` times the inner gradient. For this warp `Dg`
/// is symmetric and has a closed form, with **three** distinct factors:
///
/// - `phi'(d)` along the radius out from the curve,
/// - `1 + w` around it,
/// - and **1** along the curve itself.
///
/// That third one is the trap. The nearest point is not fixed as `p` moves: slide `p` parallel
/// to a straight run of curve and the nearest point slides with it, so the difference between
/// them is unchanged and nothing is scaled at all. Treating the nearest point as locally fixed
/// gives `1 + w` there instead of 1, and the resulting normal is several degrees out — measured
/// at 5.13 degrees for a pull of only 0.4, which is a visibly mis-lit surface.
///
/// `along` is that direction, and `None` at a joint or an end. There the nearest point really
/// *is* fixed — it is a single vertex — so every direction across the radius scales by `1 + w`
/// and the two-factor form is the right one. Both cases are real geometry: a cylinder about the
/// segments, a sphere about the joints.
///
/// Carrying the inner normal through unchanged is wrong wherever the two factors differ, and
/// wrong worst where it shows most. Approaching the fold the radial factor collapses toward
/// zero, so the true normal swings almost entirely tangential — and an unchanged normal then
/// points somewhere the surface does not face, which lights it as though something were in the
/// way. It was reported as a shadow in the wrong place, which is exactly what it was.
///
/// Written once here because the shader repeats it: `attract_normal` in `prelude.wgsl` is the
/// same six lines, and the two disagreeing would be a body lit one way on the CPU and another
/// on the GPU — which is a difference `gpu_matches_cpu` does not look for, since it compares
/// where the surface is rather than which way it faces.
pub(crate) fn warp_normal(
    p: Point3<f64>,
    closest: Point3<f64>,
    along: Option<nalgebra::Vector3<f64>>,
    strength: f64,
    radius: f64,
    inner: nalgebra::Vector3<f64>,
) -> nalgebra::Vector3<f64> {
    let away = p - closest;
    let d = away.norm();
    // On the curve there is no radial direction, and beyond the reach the warp is the identity.
    // Neither is a failure: both are places the inner normal is already the answer.
    if radius <= 0.0 || d < 1e-12 || d >= radius {
        return inner;
    }
    let x = d / radius;
    let a = 1.0 - x * x * (3.0 - 2.0 * x);
    // `dA/dd` scaled by `d`, which is `6x²(x − 1)` — the term that makes the radial factor
    // differ from the circumferential one, and the whole reason this function exists.
    let radial = 1.0 + strength * (1.0 - 9.0 * x * x + 8.0 * x * x * x);
    let across = 1.0 + strength * a;
    let n_hat = away / d;
    let mut turned = inner * across + n_hat * (inner.dot(&n_hat) * (radial - across));
    // And the axial term, which undoes the `across` factor along the curve. Absent at a joint,
    // where there is no axis to undo it along.
    if let Some(t_hat) = along {
        turned += t_hat * (inner.dot(&t_hat) * (1.0 - across));
    }
    // At the fold the radial factor passes through zero, and a purely radial normal has nothing
    // left to normalize. The inner normal is the honest fallback: wrong, but a direction — and
    // the alternative is a NaN, which paints the pixel black and is the artefact this removes.
    turned.try_normalize(1e-12).unwrap_or(inner)
}

/// What an attracted field's value must be divided by to stay a lower bound on the true
/// distance.
///
/// For a pull it is exactly `1 + strength`: on the curve itself the falloff is 1 and lengths
/// grow by that factor both radially and around, while further out the radial factor
/// `1 + s·(1 − 9x² + 8x³)` and the circumferential factor `1 + s·A(x)` are both smaller. Along
/// the curve nothing stretches at all.
///
/// **A push is not the mirror image of a pull, and assuming it was is a silent contract
/// break.** With `s` negative the radial factor is largest where the bracket is *most*
/// negative — `−11/16`, at three-quarters of the way out — so lengths there grow by
/// `1 + (11/16)|s|`, while on the curve they shrink. Dividing a push by `1` because "shrinking
/// cannot stretch" would leave the field overstating its distance by up to 69% of the strength,
/// which is a ray stepping through a wall rather than a cosmetic error.
///
/// **One number either way, unlike a twist's**, which has to be taken from the body's own
/// extent because a winding carries material further the further out it starts. Nothing here
/// depends on the body, so the divisor can be a live parameter rather than a constant baked
/// into the shader — which is what lets the strength slider re-evaluate instead of recompiling.
///
/// Away from the curve's cut locus, where the nearest point is not unique and the warp is
/// genuinely discontinuous. `attractor_profile.rs` measures what that costs.
pub fn stretch(strength: f64) -> f64 {
    if strength >= 0.0 {
        1.0 + strength
    } else {
        1.0 - strength * (11.0 / 16.0)
    }
}

/// The furthest anything moves, in millimetres, at this strength and reach.
///
/// The number the panel shows beside a unitless strength, and the amount the extent walk grows
/// a box by. See [`MAX_PULL_FRACTION`].
pub fn max_pull(strength: f64, radius: f64) -> f64 {
    // The magnitude, so a push grows the box by as much as a pull of the same size does — it
    // moves the material the other way, not less far.
    (MAX_PULL_FRACTION * strength.abs() * radius.max(0.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two ends are what make the support bounded, and they have to be exact rather than
    /// close: a falloff that is 0.001 at the radius is a warp with no edge, which is the whole
    /// thing the bounded shape was chosen for.
    #[test]
    fn the_falloff_is_one_on_the_curve_and_zero_at_the_radius() {
        assert_eq!(falloff(0.0, 10.0), 1.0);
        assert_eq!(falloff(10.0, 10.0), 0.0);
        assert_eq!(falloff(11.0, 10.0), 0.0);
        assert_eq!(falloff(1e6, 10.0), 0.0);
        // And flat at both ends, so nothing creases where the influence stops.
        let h = 1e-4;
        assert!((falloff(h, 10.0) - 1.0).abs() < 1e-8);
        assert!(falloff(10.0 - h, 10.0) < 1e-8);
    }

    /// Not "indistinguishable from" the unwarped point — identical to it. A body with the
    /// attractor turned off is what somebody compares the warped one against.
    #[test]
    fn zero_strength_leaves_the_point_exactly_where_it_was() {
        let p = Point3::new(3.0, -7.5, 1.25);
        let c = Point3::new(0.0, 0.0, 0.0);
        assert_eq!(warp_point(p, c, 0.0, 25.0), p);
        // And so does a reach of nothing, at any strength.
        assert_eq!(warp_point(p, c, 5.0, 0.0), p);
        // And so does sitting beyond the reach.
        assert_eq!(warp_point(p, c, 5.0, 1.0), p);
    }

    /// The sample point goes *out*; the material comes *in*. Stated as a test because the sign
    /// is the one thing about this technique that fails silently.
    #[test]
    fn the_sample_point_is_pushed_away_from_the_curve() {
        let c = Point3::origin();
        let p = Point3::new(4.0, 0.0, 0.0);
        let g = warp_point(p, c, 1.0, 10.0);
        assert!(g.x > p.x, "expected the sample to move outward, got {g:?}");
        // On the ray from the curve through the point, not off it.
        assert!(g.y.abs() < 1e-12 && g.z.abs() < 1e-12);
    }

    /// The constants are derived rather than fitted, so a search over the falloff has to find
    /// the same two numbers. If the falloff polynomial is ever changed, this is what says the
    /// constants beside it went stale.
    #[test]
    fn the_pull_and_fold_constants_match_the_falloff_they_came_from() {
        let (r, s) = (10.0, 1.0);
        let mut worst: f64 = 0.0;
        let mut fold_factor = f64::INFINITY;
        let n = 2_000_000;
        for i in 0..=n {
            let d = r * i as f64 / n as f64;
            let moved = (warp_point(Point3::new(d, 0.0, 0.0), Point3::origin(), s, r).x - d).abs();
            worst = worst.max(moved);
            // The radial scaling, differenced rather than derived, so this checks the algebra
            // in the doc comment instead of restating it.
            let h = 1e-5;
            let f = |t: f64| warp_point(Point3::new(t, 0.0, 0.0), Point3::origin(), s, r).x;
            if d > h && d < r - h {
                fold_factor = fold_factor.min((f(d + h) - f(d - h)) / (2.0 * h));
            }
        }
        assert!(
            (worst / (s * r) - MAX_PULL_FRACTION).abs() < 1e-6,
            "searched {}, constant says {MAX_PULL_FRACTION}",
            worst / (s * r)
        );
        // At s = 1 the factor is 1 - 11/16 = 5/16, and it reaches zero at s = 16/11.
        assert!((fold_factor - 5.0 / 16.0).abs() < 1e-4, "radial factor {fold_factor}");
        assert!((FOLD_STRENGTH - 16.0 / 11.0).abs() < 1e-12);
    }

    /// The claim [`stretch`] rests on: nothing anywhere grows by more than `1 + strength`.
    /// Sampled across a plane cut through a straight curve, which is where every direction of
    /// stretch is visible at once.
    #[test]
    fn nothing_stretches_by_more_than_the_divisor_in_either_direction() {
        for s in [0.9, 1.4, -0.5, -0.95] {
            measure_stretch_against_divisor(s);
        }
    }

    fn measure_stretch_against_divisor(s: f64) {
        let r = 10.0;
        // A straight curve along Z, so the closest point is the projection onto that axis and
        // the analysis in the doc comment applies exactly.
        let closest = |p: Point3<f64>| Point3::new(0.0, 0.0, p.z);
        let g = |p: Point3<f64>| warp_point(p, closest(p), s, r);
        let h = 1e-4;
        let mut worst: f64 = 0.0;
        for i in 0..400 {
            let d = r * 1.4 * i as f64 / 400.0;
            let p = Point3::new(d, 0.0, 0.0);
            for axis in 0..3 {
                let mut step = nalgebra::Vector3::zeros();
                step[axis] = h;
                worst = worst.max((g(p + step) - g(p - step)).norm() / (2.0 * h));
            }
        }
        assert!(
            worst <= stretch(s) + 1e-3,
            "strength {s}: measured stretch {worst}, divisor {}",
            stretch(s)
        );
        // And it is not slack either — the bound is reached. That matters as much as holding:
        // a divisor larger than the true stretch is a field that reads short everywhere, which
        // costs every ray a step and every fillet its radius.
        assert!(
            worst > stretch(s) - 1e-3,
            "strength {s}: divisor {} is loose, measured {worst}",
            stretch(s)
        );
    }
}
