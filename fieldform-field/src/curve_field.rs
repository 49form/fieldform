// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A curve swept into a solid: the pipe field.
//!
//! Distance to a curve, minus a radius. That single line is a **true** signed distance field
//! for a round tube following the curve — not an approximation of one — which matters because
//! everything downstream (booleans, blends, Fidget's interval pruning) assumes true distances
//! and misbehaves subtly when given something merely monotonic.
//!
//! # Where the approximation actually is
//!
//! The curve is sampled to a polyline first, and distance is then measured to that polyline.
//! So the field is exactly the distance field of a tube around the *polyline*, and it differs
//! from a tube around the true curve by at most the tessellation tolerance — which is stated,
//! passed in, and defaulted conservatively rather than hidden.
//!
//! That is the same bargain the cage field makes, and it is deliberate: the tessellation is an
//! accelerator whose resolution bounds accuracy, not a representation change.
//!
//! # Why distance-to-polyline is exact rather than approximate
//!
//! A minimum of exact point-to-segment distances is itself an exact distance — to the union of
//! those segments. Subtracting a constant radius shifts every level set by that constant and
//! leaves the gradient magnitude at 1, so the result is still a true SDF. The sign comes free:
//! inside the tube the distance to the spine is less than the radius, so the value is negative.

use crate::ScalarField;
use fieldform_scene::Curve;
use nalgebra::{Point3, Vector3};

/// Default tessellation tolerance, in millimetres.
///
/// Ten microns is finer than any display need and finer than most machining, so the pipe's
/// deviation from the true curve is not the limiting error in practice. Callers wanting a
/// tighter guarantee can say so; nothing here silently decides accuracy on its own.
pub const DEFAULT_TOLERANCE_MM: f64 = 0.01;

/// A tube of constant radius following a curve.
pub struct CurveField {
    /// The curve's spine, sampled to within the tolerance it was built with of the true curve.
    spine: crate::Spine,
    radius: f64,
}

impl CurveField {
    /// Builds a pipe of `radius_mm` around `curve`, with the spine sampled to
    /// `tolerance_mm`.
    ///
    /// A closed curve's spine is closed too, so the tube joins rather than showing two caps.
    pub(crate) fn new(curve: &Curve, radius_mm: f64, tolerance_mm: f64) -> Self {
        Self {
            spine: crate::Spine::of_curve(curve, tolerance_mm),
            radius: radius_mm.max(0.0),
        }
    }

    /// A pipe with the default tessellation tolerance.
    ///
    /// The graph builds its pipes through [`Self::new`] at the compile's own tolerance, so only
    /// the tests build one at the default, and only the tests compile this.
    #[cfg(test)]
    pub(crate) fn with_radius(curve: &Curve, radius_mm: f64) -> Self {
        Self::new(curve, radius_mm, DEFAULT_TOLERANCE_MM)
    }

    /// How many segments the spine was sampled into. Only a test asks, to know that the curve it
    /// measures joints on has enough of them, so only the tests compile it.
    #[cfg(test)]
    fn segment_count(&self) -> usize {
        self.spine.points().len().saturating_sub(1)
    }

    /// Axis-aligned bounds of the pipe: the spine's extent grown by the radius.
    ///
    /// Meshing needs somewhere to start, and a box derived from the geometry beats a guessed
    /// one — too small silently clips the model, too large wastes resolution on empty space.
    ///
    /// The graph reads a pipe's extent from its node rather than from a built pipe, so only the
    /// tests here mesh against this box, and only the tests compile it (decision 1.4).
    #[cfg(test)]
    fn bounds(&self) -> (Point3<f64>, Point3<f64>) {
        let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in self.spine.points() {
            lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        let r = self.radius;
        (
            Point3::new(lo.x - r, lo.y - r, lo.z - r),
            Point3::new(hi.x + r, hi.y + r, hi.z + r),
        )
    }

    /// The tessellated spine itself, for callers that need to measure against the curve
    /// rather than against the tube around it.
    ///
    /// Exposed because "how far is this from the curve, and how far along" is a question more
    /// than the pipe asks — a radius that varies along the curve needs the position, an
    /// attractor pulling geometry toward one needs both — and answering it here would mean
    /// each of them either rebuilding a pipe they do not want or writing the search again.
    ///
    /// Those readers build a `Spine` of their own from the curve, so only the tests here look
    /// at a pipe's, and only the tests compile this (decision 1.4).
    #[cfg(test)]
    fn spine(&self) -> &crate::Spine {
        &self.spine
    }

    /// Unsigned distance from `p` to the spine.
    pub(crate) fn distance_to_spine(&self, p: Point3<f64>) -> f64 {
        self.spine.distance(p)
    }

    /// The point on the spine nearest `p`, which is what the tube's surface is measured from.
    pub(crate) fn closest_on_spine(&self, p: Point3<f64>) -> Option<Point3<f64>> {
        self.spine.nearest(p).map(|n| n.point)
    }
}

impl ScalarField for CurveField {
    /// Negative inside the tube, positive outside, and a true Euclidean distance either way.
    fn distance(&self, p: Point3<f64>) -> f64 {
        self.distance_to_spine(p) - self.radius
    }

    /// A tube's outward normal needs no differencing: the surface is every point at `radius`
    /// from the spine, so it faces directly away from the spine point it was measured from.
    ///
    /// Worth stating because the absence of this was a real defect rather than an omission of
    /// polish. Returning `None` does not fall back to something merely slower — it falls back
    /// to a *different* answer, and a composed model then has its cage shaded one way and its
    /// pipe another, which shows up as tearing exactly along the seam where they meet.
    ///
    /// Still `None` directly on the spine, where "away from the axis" genuinely has no
    /// direction. That is a real singularity rather than a gap, and it lies a full radius
    /// inside the material where no surface point can be.
    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let closest = self.closest_on_spine(p)?;
        (p - closest).try_normalize(1e-12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_scene::CurveId;

    fn straight_line(length: f64) -> Curve {
        Curve::polyline(
            CurveId::new(),
            "line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(length, 0.0, 0.0)],
            false,
        )
        .unwrap()
    }

    /// The defining property: on the surface the field is zero, inside it is negative, outside
    /// positive — and the magnitude is the actual distance, not merely something with the
    /// right sign.
    #[test]
    fn the_field_is_the_true_distance_to_the_tube() {
        let curve = straight_line(100.0);
        let field = CurveField::with_radius(&curve, 5.0);

        // Straight out from the middle of the spine, where the answer is exact by inspection.
        let mid = Point3::new(50.0, 0.0, 0.0);
        for offset in [0.0, 2.5, 5.0, 7.5, 20.0] {
            let p = mid + nalgebra::Vector3::new(0.0, offset, 0.0);
            let expected = offset - 5.0;
            assert!(
                (field.distance(p) - expected).abs() < 1e-9,
                "at offset {offset} expected {expected}, got {}",
                field.distance(p)
            );
        }
    }

    #[test]
    fn the_radius_is_in_millimetres() {
        let curve = straight_line(100.0);
        for radius in [0.5, 3.0, 12.5] {
            let field = CurveField::with_radius(&curve, radius);
            let on_surface = Point3::new(50.0, radius, 0.0);
            assert!(
                field.distance(on_surface).abs() < 1e-9,
                "radius {radius} mm: surface point read {}",
                field.distance(on_surface)
            );
        }
    }

    /// A true SDF has gradient magnitude 1 everywhere. Booleans, blends and Fidget's interval
    /// pruning all assume that; a field that merely has the right sign misbehaves in ways that
    /// are hard to trace back. Checked by finite differences away from the spine, where the
    /// field is smooth.
    #[test]
    fn the_gradient_has_unit_magnitude() {
        let curve = straight_line(100.0);
        let field = CurveField::with_radius(&curve, 5.0);
        let h = 1e-6;

        for p in [
            Point3::new(50.0, 20.0, 0.0),
            Point3::new(50.0, 0.0, 30.0),
            Point3::new(130.0, 10.0, 10.0),
            Point3::new(-20.0, 4.0, 3.0),
        ] {
            let grad = nalgebra::Vector3::new(
                field.distance(p + nalgebra::Vector3::new(h, 0.0, 0.0))
                    - field.distance(p - nalgebra::Vector3::new(h, 0.0, 0.0)),
                field.distance(p + nalgebra::Vector3::new(0.0, h, 0.0))
                    - field.distance(p - nalgebra::Vector3::new(0.0, h, 0.0)),
                field.distance(p + nalgebra::Vector3::new(0.0, 0.0, h))
                    - field.distance(p - nalgebra::Vector3::new(0.0, 0.0, h)),
            ) / (2.0 * h);
            assert!(
                (grad.norm() - 1.0).abs() < 1e-5,
                "at {p:?} the gradient magnitude was {}",
                grad.norm()
            );
        }
    }


    /// A quarter circle of `radius_mm` in the XY plane, centred on the origin, as an exact
    /// rational quadratic.
    ///
    /// Exact rather than merely close, because the probe filter in the test below is stated in
    /// terms of that circle: every spine vertex has to lie *on* it for "outside it" to mean "on
    /// the convex side of every joint".
    fn quarter_arc(radius_mm: f64) -> Curve {
        Curve::new(
            CurveId::new(),
            "quarter arc",
            2,
            vec![
                Point3::new(radius_mm, 0.0, 0.0),
                Point3::new(radius_mm, radius_mm, 0.0),
                Point3::new(0.0, radius_mm, 0.0),
            ],
            // A conic arc's middle weight is the cosine of half the angle it spans, which for a
            // quarter circle is one over root two.
            vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a clamped quadratic rational Bezier is a curve")
    }

    /// D8's property on a spine that actually turns, which is the only place the claim could
    /// fail.
    ///
    /// `the_gradient_has_unit_magnitude` above checks a straight line, where the field is the
    /// distance to one segment and unit gradient is true by inspection. A curve is flattened
    /// into sixty-odd segments and the field is the `min` over all of them — and a `min` of
    /// smooth functions is smooth only where one of them strictly wins. That is what makes
    /// [`Normalization::Distance`], which this field claims by taking the trait's default,
    /// worth testing here rather than on a line.
    ///
    /// Three regions are genuinely singular and are avoided on purpose rather than by luck:
    ///
    /// - **The spine itself**, where "away from the axis" has no direction. It is the same
    ///   singularity [`CurveField::normal`] returns `None` at, and it lies a full radius inside
    ///   the material; every probe here is at least 6 mm off it.
    /// - **Inside the bend.** Each joint of the polyline throws a bisector sheet *inward*: the
    ///   locus where the two segments meeting there are exactly equally near. The gradient turns
    ///   across it by that joint's kink angle, so a central difference straddling one reads
    ///   short. The falsification at the end plants a probe on that sheet and requires it to
    ///   fail, so the exclusion is known to be removing something real rather than padding the
    ///   pass.
    /// - **The far side, beyond the arc's ends**, where the two end caps are equidistant on the
    ///   symmetry plane between them. Avoided by keeping every probe inside the arc's own
    ///   angular span.
    ///
    /// The surface is deliberately *not* avoided. Subtracting a constant radius shifts every
    /// level set and leaves the gradient alone, so a tube's surface is an ordinary smooth place;
    /// probes sit inside the wall and outside it both.
    ///
    /// What is left — outside the circle the spine is inscribed in, within its angular span —
    /// has a unique nearest point: on the convex side of a joint *both* segments report the
    /// distance to the vertex they share, so the `min` of the two is that one smooth function.
    #[test]
    fn a_curved_pipe_has_a_unit_gradient_away_from_the_inside_of_its_bend() {
        const SPINE_RADIUS: f64 = 100.0;
        const TUBE_RADIUS: f64 = 20.0;

        let curve = quarter_arc(SPINE_RADIUS);
        let field = CurveField::new(&curve, TUBE_RADIUS, 0.01);

        assert_eq!(
            field.normalization(),
            crate::Normalization::Distance,
            "the claim this test exists to check",
        );
        // Enough joints that they are the thing under test, rather than one incidental corner.
        assert!(
            field.segment_count() > 32,
            "a spine of {} segments is not a curve",
            field.segment_count(),
        );
        // And the spine really is inscribed in the circle the filter below is written against.
        for i in 0..=20 {
            let p = curve.evaluate(i as f64 / 20.0);
            assert!(
                (p.coords.norm() - SPINE_RADIUS).abs() < 1e-9 && p.z.abs() < 1e-12,
                "the arc is not the circle the probe filter assumes: {p:?}",
            );
        }

        let mut probes = Vec::new();
        for i in 1..16 {
            let angle = std::f64::consts::FRAC_PI_2 * i as f64 / 16.0;
            // Outward of the spine's circle, so the convex side of every joint; several
            // distances from the spine, straddling the wall; and several heights out of the
            // arc's plane, which a planar probe set would never leave.
            for outward in [6.0, 12.0, 18.0, 25.0, 45.0] {
                for z in [0.0, 7.0, -7.0, 30.0, -30.0] {
                    let rho = SPINE_RADIUS + outward;
                    probes.push(Point3::new(rho * angle.cos(), rho * angle.sin(), z));
                }
            }
        }
        let worst = crate::gradient_deviation(&field, &probes, 1e-4);
        assert!(
            worst < 1e-6,
            "unit gradient off by {worst} over {} probes",
            probes.len(),
        );

        // The falsification. At the middle joint, the direction perpendicular to the sum of the
        // two segment directions is where those two segments are exactly equally near; on the
        // concave side that is the bisector sheet named above. Planted rather than searched
        // for: a probe has to fall within `h` of the sheet to straddle it, and an angular sweep
        // of the inside reads clean for that reason alone.
        let points = field.spine().points();
        let joint = points.len() / 2;
        let (a, b, c) = (points[joint - 1], points[joint], points[joint + 1]);
        let along = ((b - a).normalize() + (c - b).normalize()).normalize();
        let mut inward = Vector3::new(-along.y, along.x, 0.0);
        // Toward the arc's centre. On the other side both segments report the distance to the
        // vertex they share, so the `min` is smooth and there is nothing to catch.
        if inward.dot(&(Point3::origin() - b)) < 0.0 {
            inward = -inward;
        }
        let crease = crate::gradient_deviation(&field, &[b + inward * 10.0], 1e-4);
        assert!(
            crease > 1e-5,
            "the inside of the bend should fail this check, and read {crease}",
        );
    }

    /// Ends are hemispherical caps, so a point beyond the end is `distance to the endpoint
    /// minus radius` — the swept-sphere definition, not a flat cut.
    #[test]
    fn the_ends_are_round_caps() {
        let curve = straight_line(100.0);
        let field = CurveField::with_radius(&curve, 5.0);

        let beyond = Point3::new(120.0, 0.0, 0.0);
        assert!((field.distance(beyond) - 15.0).abs() < 1e-9);

        // Diagonally beyond the end: distance to the end point, minus the radius.
        let diagonal = Point3::new(103.0, 4.0, 0.0);
        let expected = 5.0_f64 - 5.0;
        assert!(
            (field.distance(diagonal) - expected).abs() < 1e-9,
            "got {}",
            field.distance(diagonal)
        );
    }

    #[test]
    fn bounds_contain_the_tube_and_its_radius() {
        let curve = straight_line(100.0);
        let field = CurveField::with_radius(&curve, 5.0);
        let (lo, hi) = field.bounds();

        assert!((lo.x - -5.0).abs() < 1e-9, "{lo:?}");
        assert!((hi.x - 105.0).abs() < 1e-9, "{hi:?}");
        assert!(
            (lo.y - -5.0).abs() < 1e-9 && (hi.y - 5.0).abs() < 1e-9,
            "{lo:?} {hi:?}"
        );

        // Every corner of the box must be outside the solid, or meshing would clip it.
        for &x in &[lo.x, hi.x] {
            for &y in &[lo.y, hi.y] {
                for &z in &[lo.z, hi.z] {
                    assert!(field.distance(Point3::new(x, y, z)) >= -1e-9);
                }
            }
        }
    }

    /// A curved spine is where the tessellation tolerance shows up. The pipe around the
    /// polyline must stay within tolerance of a pipe around the true curve, which is testable
    /// directly: sample the true curve and check each sample sits `radius` from the surface to
    /// within the stated bound.
    #[test]
    fn the_pipe_honours_its_stated_tolerance_on_a_curved_spine() {
        let curve = Curve::new(
            CurveId::new(),
            "s-curve",
            3,
            vec![
                Point3::new(-1000.0, 0.0, 0.0),
                Point3::new(-333.333333, 500.0, 0.0),
                Point3::new(333.333333, -500.0, 0.0),
                Point3::new(1000.0, 0.0, 0.0),
            ],
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            false,
        )
        .unwrap();

        let tolerance = 0.01;
        let field = CurveField::new(&curve, 25.0, tolerance);
        let (lo, hi) = curve.domain();
        for i in 0..=200 {
            let t = lo + (hi - lo) * i as f64 / 200.0;
            // A point on the true curve should read exactly -radius if the polyline passed
            // through it; the tolerance bounds how far off it can be.
            let d = field.distance(curve.evaluate(t));
            assert!(
                (d + 25.0).abs() <= tolerance,
                "at t={t} the spine read {d}, which is more than {tolerance} mm from -25"
            );
        }
    }

    #[test]
    fn a_closed_curve_gives_a_closed_tube() {
        // A unit square as a closed polyline.
        let curve = Curve::polyline(
            CurveId::new(),
            "square",
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(100.0, 0.0, 0.0),
                Point3::new(100.0, 100.0, 0.0),
                Point3::new(0.0, 100.0, 0.0),
            ],
            true,
        )
        .unwrap();
        let field = CurveField::with_radius(&curve, 5.0);

        // The closing segment runs from (0,100) back to (0,0); its midpoint must be inside.
        let on_closing_segment = Point3::new(0.0, 50.0, 0.0);
        assert!(
            field.distance(on_closing_segment) < 0.0,
            "the closing segment is missing: {}",
            field.distance(on_closing_segment)
        );
    }
}
