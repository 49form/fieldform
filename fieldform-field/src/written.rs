// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A field written as arithmetic -- the Expression block's value on the CPU (backlog item 40,
//! `expression-field-spec.md` §3).
//!
//! **The contract is the spec.** `ScalarField::distance` must never overestimate the distance,
//! and an arbitrary expression honours that by construction nowhere, so the value is the
//! expression over its divisor `k` -- the bound on how fast it changes per millimetre that the
//! person supplies, one for a field written in the distance idiom -- and cut to its region by a
//! box, which is what bounds the trace and what makes `x` alone finite enough to draw. What
//! it claims, distance or lower bound, is the person's claim too, and the verify action is how
//! it is checked ([`crate::gradient_deviation`], [`crate::overstatement`]).
//!
//! **Emitted as written, nothing clamped.** `log` of a negative is not a number here, exactly as
//! in the shader; a value that is not finite is reported as the largest distance there is, so a
//! ray meets no surface in it rather than a NaN that blanks the viewport, and the verify action
//! counts such points so the person is told where their formula has no value.

use crate::{Normalization, ScalarField};
use fieldform_scene::{Expression, ParamId, WrittenClaim};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// The Expression block's field, in its own frame: the region's box is about the origin.
#[derive(Clone, Debug)]
pub struct WrittenField {
    expression: Expression,
    /// The value of every parameter the text reads, resolved once at compile time: a slot in
    /// the shader, a number here.
    values: HashMap<ParamId, f64>,
    /// The Lipschitz divisor, over nought.
    divisor: f64,
    /// Half the region's extent along each axis.
    half: Vector3<f64>,
    claim: WrittenClaim,
}

impl WrittenField {
    /// Refused when the divisor is not over nought, when a side of the region is not, or when
    /// a parameter the text reads has no value.
    pub fn new(
        expression: Expression,
        values: HashMap<ParamId, f64>,
        divisor: f64,
        size: Vector3<f64>,
        claim: WrittenClaim,
    ) -> Result<Self, String> {
        if !(divisor > 0.0) || !divisor.is_finite() {
            return Err(format!("the divisor must be a number over nought, not {divisor}"));
        }
        if size.iter().any(|s| !(*s > 0.0) || !s.is_finite()) {
            return Err("the region needs a size over nought on every axis".into());
        }
        for id in expression.references() {
            if !values.get(&id).is_some_and(|v| v.is_finite()) {
                return Err("a parameter the text reads has no value".into());
            }
        }
        Ok(Self { expression, values, divisor, half: size * 0.5, claim })
    }

    /// The expression's own value at `p`, before the divisor and the region: `None` where a
    /// step of it is not a number.
    pub fn raw(&self, p: Point3<f64>) -> Option<f64> {
        self.expression
            .evaluate_at(&|id| self.values.get(&id).copied(), [p.x, p.y, p.z])
            .ok()
            .filter(|v| v.is_finite())
    }

    /// The region's own distance: an axis-aligned box about the origin.
    pub fn region(&self, p: Point3<f64>) -> f64 {
        crate::Cuboid { centre: Point3::origin(), size: self.half * 2.0 }.distance(p)
    }

    pub fn divisor(&self) -> f64 {
        self.divisor
    }

    pub fn half(&self) -> Vector3<f64> {
        self.half
    }
}

impl ScalarField for WrittenField {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let region = self.region(p);
        match self.raw(p) {
            // The written value over its divisor, cut to the region: `max` with a distance
            // keeps a lower bound a lower bound, since `max` reads only which is nearer.
            Some(v) => (v / self.divisor).max(region),
            // No value here: no surface here. Far rather than NaN, so a ray steps past.
            None => f64::MAX.sqrt(),
        }
    }

    fn normalization(&self) -> Normalization {
        match self.claim {
            WrittenClaim::Distance => Normalization::Distance,
            WrittenClaim::LowerBound => Normalization::LowerBound,
        }
    }

    // One by the person's own claim about `k`; the region is a distance.
    fn lipschitz(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_scene::Expression;

    fn written(text: &str, k: f64, size: f64, claim: WrittenClaim) -> WrittenField {
        let radius = ParamId::new();
        let expression = Expression::parse_field(text, &|n| (n == "radius").then_some(radius)).unwrap();
        let mut values = HashMap::new();
        values.insert(radius, 5.0);
        WrittenField::new(expression, values, k, Vector3::repeat(size), claim).unwrap()
    }

    /// Scattered through the region, off every lattice, and away from the box's own faces.
    fn probes() -> Vec<Point3<f64>> {
        (0..400)
            .map(|i| {
                let t = i as f64 + 0.37;
                Point3::new(7.0 * (t * 0.618).sin(), 7.0 * (t * 0.271).cos(), 7.0 * (t * 0.173).sin())
            })
            .collect()
    }

    /// The D8 property, applied to a source whose correctness is the person's rather than ours:
    /// a sphere written in the distance idiom is a distance, and a raw gyroid claiming the same
    /// is not -- its gradient reaches well past one.
    #[test]
    fn a_written_sphere_is_a_distance_and_a_raw_gyroid_claiming_one_is_not() {
        let sphere = written("sqrt(x*x + y*y + z*z) - radius", 1.0, 30.0, WrittenClaim::Distance);
        assert_eq!(sphere.normalization(), Normalization::Distance);
        let d = crate::gradient_deviation(&sphere, &probes(), 1e-4);
        assert!(d < 1e-6, "a written sphere's gradient is unit: {d}");
        assert!((sphere.distance(Point3::new(8.0, 0.0, 0.0)) - 3.0).abs() < 1e-12);

        let gyroid = written(
            "sin(x * 36) * cos(y * 36) + sin(y * 36) * cos(z * 36) + sin(z * 36) * cos(x * 36)",
            1.0,
            30.0,
            WrittenClaim::Distance,
        );
        let d = crate::gradient_deviation(&gyroid, &probes(), 1e-4);
        assert!(d > 0.3, "a raw gyroid reads like a distance: {d}");
    }

    /// The contract itself: a written sphere with `k = 1` never claims more room than there is,
    /// a raw gyroid with `k = 1` does, and divided by its own gradient bound it does not.
    #[test]
    fn overstatement_is_nought_for_a_written_sphere_and_not_for_a_raw_gyroid_until_it_is_divided() {
        let sphere = written("sqrt(x*x + y*y + z*z) - radius", 1.0, 30.0, WrittenClaim::LowerBound);
        let outside: Vec<Point3<f64>> = probes().into_iter().filter(|p| sphere.distance(*p) > 0.0).collect();
        let over = crate::overstatement(&sphere, &outside, &crate::probe_directions(), 0.01, 1.2);
        assert!(over < 1e-6, "the written sphere claimed {over} mm too much");

        let text = "sin(x * 36) * cos(y * 36) + sin(y * 36) * cos(z * 36) + sin(z * 36) * cos(x * 36) + 0.3";
        let raw = written(text, 1.0, 30.0, WrittenClaim::LowerBound);
        let outside: Vec<Point3<f64>> = probes().into_iter().filter(|p| raw.distance(*p) > 0.0).collect();
        let over = crate::overstatement(&raw, &outside, &crate::probe_directions(), 0.01, 1.2);
        assert!(over > 1e-3, "a raw gyroid at k = 1 must overstate: {over}");
        // The gradient of each term is 36 degrees a millimetre as radians, times the sqrt(6)
        // the three products reach together: the provable bound, and anything above it holds.
        let k = 36f64.to_radians() * 6f64.sqrt();
        let divided = written(text, k * 1.01, 30.0, WrittenClaim::LowerBound);
        let outside: Vec<Point3<f64>> = probes().into_iter().filter(|p| divided.distance(*p) > 0.0).collect();
        let over = crate::overstatement(&divided, &outside, &crate::probe_directions(), 0.01, 1.2);
        assert!(over < 1e-6, "divided by its bound the gyroid still claimed {over} mm too much");
    }

    #[test]
    fn a_step_with_no_value_is_no_surface_and_the_region_cuts_the_field() {
        let field = written("log(x)", 1.0, 20.0, WrittenClaim::LowerBound);
        assert!(field.raw(Point3::new(-1.0, 0.0, 0.0)).is_none(), "log of a negative has no value");
        assert!(field.distance(Point3::new(-1.0, 0.0, 0.0)) > 1e6);
        // `x` alone is a half-space; the region's box is what bounds it.
        let plane = written("x", 1.0, 20.0, WrittenClaim::Distance);
        assert!((plane.distance(Point3::new(-3.0, 0.0, 0.0)) + 3.0).abs() < 1e-12);
        assert!(plane.distance(Point3::new(-3.0, 0.0, 40.0)) > 0.0, "outside the region is outside");
        assert!(WrittenField::new(
            Expression::parse_field("x", &|_| None).unwrap(),
            HashMap::new(),
            0.0,
            Vector3::repeat(1.0),
            WrittenClaim::Distance
        )
        .is_err());
    }
}
