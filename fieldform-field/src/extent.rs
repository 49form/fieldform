// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a body's surface can be — and the two different reasons it might be nowhere.
//!
//! # Why this is not `Option<(lo, hi)>`
//!
//! It used to be, and that shape cannot tell two opposite situations apart. `None` was read as
//! "contributes nothing", which is right for a body that is empty and wrong for one that has no
//! bound at all. The two differ exactly where it matters: a union with an *empty* operand is
//! bounded by the other one, and a union with an *unbounded* operand is unbounded. Under one
//! sentinel, a half-space unioned into a part silently reported the part's own box — and
//! `body_bounds`' own comment said too small clips the model, where too large only costs bake
//! resolution. So the sentinel was the unsafe reading of the two.
//!
//! Three states, decided at the node that knows:
//!
//! - [`Extent::Bounded`] — the surface lies inside this box.
//! - [`Extent::Empty`] — there is no surface.
//! - [`Extent::Unbounded`] — the surface is not confined to any box.
//!
//! # Why an unbounded field is worth having at all
//!
//! Because it is almost always an *operand*, not a body. A half-space is the standard trimming
//! primitive, and `Intersection(part, plane)` is bounded by the part however infinite the plane
//! is — the intersection rule below is what makes that fall out rather than needing a special
//! case. `brick_pruning.rs` measured the same effect from the other end: a bare gyroid straddles
//! zero in every region, and the same gyroid confined to a part rejects 82% of them.
//!
//! That is the general rule this type exists to express: **unboundedness is contagious through a
//! union and killed by an intersection.**

use nalgebra::Point3;

/// Where a field's surface can be.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Extent {
    /// No surface anywhere — nothing to draw, nothing to bound.
    Empty,
    /// The surface lies within this axis-aligned box.
    ///
    /// Enclosing rather than tight: a rotated box is not axis-aligned, and a conservative
    /// answer costs bake resolution where an optimistic one clips the model.
    Bounded { lo: Point3<f64>, hi: Point3<f64> },
    /// The surface is not confined to any box — a half-space, or anything built on one.
    Unbounded,
}

impl Extent {
    /// A box around a single point, which is what a zero-radius source comes to.
    pub(crate) fn at(p: Point3<f64>) -> Self {
        Extent::Bounded { lo: p, hi: p }
    }

    /// A box around a point, grown by `radius` in every direction.
    pub(crate) fn ball(centre: Point3<f64>, radius: f64) -> Self {
        Extent::Bounded {
            lo: Point3::new(centre.x - radius, centre.y - radius, centre.z - radius),
            hi: Point3::new(centre.x + radius, centre.y + radius, centre.z + radius),
        }
    }

    /// A box enclosing two ball-ended stations — every solid drawn between two points.
    pub(crate) fn between(a: Point3<f64>, b: Point3<f64>, ra: f64, rb: f64) -> Self {
        Extent::ball(a, ra).union(Extent::ball(b, rb))
    }

    /// Builds from an iterator of points, each grown by `margin`. [`Extent::Empty`] if the
    /// iterator yields nothing, which is the honest answer for geometry with no points in it.
    pub(crate) fn around(points: impl IntoIterator<Item = Point3<f64>>, margin: f64) -> Self {
        let mut out = Extent::Empty;
        for p in points {
            out = out.union(Extent::ball(p, margin));
        }
        out
    }

    /// The box, if there is one. `None` for both [`Extent::Empty`] and [`Extent::Unbounded`] —
    /// callers that genuinely cannot tell them apart, like "frame this on screen", but callers
    /// that *combine* extents must match on the variants instead.
    pub fn as_box(self) -> Option<(Point3<f64>, Point3<f64>)> {
        match self {
            Extent::Bounded { lo, hi } => Some((lo, hi)),
            Extent::Empty | Extent::Unbounded => None,
        }
    }

    pub fn is_unbounded(self) -> bool {
        matches!(self, Extent::Unbounded)
    }

    /// The smallest box holding both — and unbounded if either is.
    ///
    /// The contagion is the whole point: a part unioned with a half-space really is unbounded,
    /// and reporting the part's own box there is how an infinite operand used to disappear.
    pub fn union(self, other: Self) -> Self {
        match (self, other) {
            (Extent::Unbounded, _) | (_, Extent::Unbounded) => Extent::Unbounded,
            (Extent::Empty, x) | (x, Extent::Empty) => x,
            (Extent::Bounded { lo: alo, hi: ahi }, Extent::Bounded { lo: blo, hi: bhi }) => {
                Extent::Bounded {
                    lo: Point3::new(alo.x.min(blo.x), alo.y.min(blo.y), alo.z.min(blo.z)),
                    hi: Point3::new(ahi.x.max(bhi.x), ahi.y.max(bhi.y), ahi.z.max(bhi.z)),
                }
            }
        }
    }

    /// The part inside both — so one bounded operand bounds the result however large the other.
    ///
    /// This is what makes a half-space usable: an infinite plane cutting a finite part gives a
    /// finite answer, with no special case for the plane.
    pub(crate) fn intersection(self, other: Self) -> Self {
        match (self, other) {
            (Extent::Empty, _) | (_, Extent::Empty) => Extent::Empty,
            (Extent::Unbounded, x) | (x, Extent::Unbounded) => x,
            (Extent::Bounded { lo: alo, hi: ahi }, Extent::Bounded { lo: blo, hi: bhi }) => {
                let lo = Point3::new(alo.x.max(blo.x), alo.y.max(blo.y), alo.z.max(blo.z));
                let hi = Point3::new(ahi.x.min(bhi.x), ahi.y.min(bhi.y), ahi.z.min(bhi.z));
                // Boxes that miss each other entirely: the shapes cannot overlap either.
                if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
                    Extent::Empty
                } else {
                    Extent::Bounded { lo, hi }
                }
            }
        }
    }

    /// What is left of `self` after cutting `other` out of it: `self`, unchanged.
    ///
    /// A cut cannot reach outside the thing being cut, so the right operand contributes no
    /// extent at all — taking a union of both would stretch the box down the whole length of a
    /// pipe that contributes nothing but a hole.
    pub(crate) fn difference(self) -> Self {
        self
    }

    /// Grown outward by `by` in every direction, for an operation whose surface can bulge past
    /// its operands — a blend's fillet being the one that does.
    pub(crate) fn grown(self, by: f64) -> Self {
        match self {
            Extent::Bounded { lo, hi } => Extent::Bounded {
                lo: Point3::new(lo.x - by, lo.y - by, lo.z - by),
                hi: Point3::new(hi.x + by, hi.y + by, hi.z + by),
            },
            other => other,
        }
    }

    /// The extent under a placement, as the box around the placed corners.
    ///
    /// Encloses rather than matches, because a rotated box is not axis-aligned. Unboundedness
    /// survives any placement: no rigid motion makes a half-space finite.
    pub(crate) fn placed(self, place: impl Fn(Point3<f64>) -> Point3<f64>) -> Self {
        let Extent::Bounded { lo, hi } = self else {
            return self;
        };
        let corners = [
            Point3::new(lo.x, lo.y, lo.z),
            Point3::new(hi.x, lo.y, lo.z),
            Point3::new(lo.x, hi.y, lo.z),
            Point3::new(hi.x, hi.y, lo.z),
            Point3::new(lo.x, lo.y, hi.z),
            Point3::new(hi.x, lo.y, hi.z),
            Point3::new(lo.x, hi.y, hi.z),
            Point3::new(hi.x, hi.y, hi.z),
        ];
        Extent::around(corners.into_iter().map(place), 0.0)
    }

    /// The centre of the box, or `None` when there is not one.
    pub fn centre(self) -> Option<Point3<f64>> {
        self.as_box()
            .map(|(lo, hi)| Point3::from((lo.coords + hi.coords) * 0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn boxed(lo: f64, hi: f64) -> Extent {
        Extent::Bounded {
            lo: Point3::new(lo, lo, lo),
            hi: Point3::new(hi, hi, hi),
        }
    }

    /// The distinction the old `Option` could not draw, and the reason this type exists.
    #[test]
    fn a_union_is_unbounded_if_either_side_is_but_ignores_an_empty_one() {
        let part = boxed(-1.0, 1.0);
        assert_eq!(
            part.union(Extent::Empty),
            part,
            "an empty operand adds nothing"
        );
        assert_eq!(
            part.union(Extent::Unbounded),
            Extent::Unbounded,
            "an unbounded operand makes the whole thing unbounded"
        );
        assert_eq!(
            Extent::Unbounded.union(part),
            Extent::Unbounded,
            "either way round"
        );
    }

    /// What makes a half-space usable as a trimming primitive: cutting a finite part with an
    /// infinite plane gives a finite answer, and no special case says so.
    #[test]
    fn an_intersection_is_bounded_when_either_side_is() {
        let part = boxed(-1.0, 1.0);
        assert_eq!(part.intersection(Extent::Unbounded), part);
        assert_eq!(Extent::Unbounded.intersection(part), part);
        assert_eq!(
            Extent::Unbounded.intersection(Extent::Unbounded),
            Extent::Unbounded,
            "two half-spaces still bound nothing"
        );
        assert_eq!(part.intersection(Extent::Empty), Extent::Empty);
    }

    #[test]
    fn an_intersection_of_boxes_that_miss_each_other_is_empty() {
        assert_eq!(
            boxed(-2.0, -1.0).intersection(boxed(1.0, 2.0)),
            Extent::Empty
        );
        // Touching is not missing.
        assert_eq!(
            boxed(-2.0, 0.0).intersection(boxed(0.0, 2.0)),
            Extent::Bounded {
                lo: Point3::origin(),
                hi: Point3::origin()
            },
        );
    }

    /// A cut cannot reach outside the thing being cut, so the right operand contributes
    /// nothing — including when it is the unbounded one.
    #[test]
    fn a_difference_keeps_only_the_left_operands_extent() {
        let part = boxed(-1.0, 1.0);
        assert_eq!(part.difference(), part);
        assert_eq!(Extent::Unbounded.difference(), Extent::Unbounded);
    }

    #[test]
    fn neither_growing_nor_placing_makes_an_unbounded_extent_finite() {
        assert_eq!(Extent::Unbounded.grown(10.0), Extent::Unbounded);
        assert_eq!(
            Extent::Unbounded.placed(|p| p + Vector3::new(1.0, 2.0, 3.0)),
            Extent::Unbounded
        );
        assert_eq!(Extent::Empty.grown(10.0), Extent::Empty);
    }

    /// A rotated box is not axis-aligned, so the placed extent has to enclose the corners
    /// rather than move the box — too small clips the model, too large only costs resolution.
    #[test]
    fn a_placed_box_encloses_its_rotated_corners() {
        use nalgebra::Rotation3;
        let unit = boxed(-1.0, 1.0);
        let turn = Rotation3::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_4);
        let Some((lo, hi)) = unit.placed(|p| turn * p).as_box() else {
            panic!("a placed box is still a box")
        };
        // The corners swing out to sqrt(2) across, and Z is untouched.
        let root2 = 2f64.sqrt();
        assert!((hi.x - root2).abs() < 1e-12, "got {}", hi.x);
        assert!((lo.x + root2).abs() < 1e-12);
        assert!(
            (hi.z - 1.0).abs() < 1e-12,
            "the axis turned about does not grow"
        );
    }

    #[test]
    fn an_extent_around_no_points_is_empty() {
        assert_eq!(Extent::around(std::iter::empty(), 1.0), Extent::Empty);
    }
}
