// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The nearest point on a tessellated curve, and where along that curve it is.
//!
//! A curve arrives here already flattened into a polyline — its *spine* — because everything
//! downstream measures against the tessellation rather than against the analytic curve, and a
//! shared answer that disagreed with what the pipe actually sweeps would be worse than no
//! shared answer at all.
//!
//! # Why the parameter is here and not left to the caller
//!
//! Asking "how far is this point from the curve" and asking "where along the curve is it" are
//! the same search: both walk the segments, both keep whichever came closest. A pipe wants the
//! first, a radius profile along a curve wants the second, and an attractor wants both. Split
//! across two functions they would be the same loop written twice, free to disagree about
//! which segment won a near-tie — and a radius sampled from one segment while the distance
//! came from its neighbour is a defect that appears only where two segments are equidistant,
//! which is to say along the inside of every bend.
//!
//! # Arc length, not curve parameter
//!
//! [`Nearest::t`] runs 0 to 1 along the spine's *length*, not along the curve's own knot
//! parameter. A NURBS is not parameterised at constant speed — knots crowd where the curve
//! turns — so "halfway along" in knots is not halfway along in millimetres, and a radius
//! profile that used it would bunch its variation wherever the curve happened to be
//! knot-dense. Length is what a person means by "halfway".

use nalgebra::{Point3, Vector3};

/// A tessellated curve, with the arc length to each of its points precomputed.
#[derive(Clone, Debug, Default)]
pub struct Spine {
    points: Vec<Point3<f64>>,
    /// Arc length from the start to each point. Same length as `points`, and `[0]` is zero.
    ///
    /// Precomputed because the alternative is summing a prefix of the segments inside the
    /// search, which turns a linear walk into a quadratic one for an answer that never
    /// changes.
    cumulative: Vec<f64>,
}

/// The nearest point on a spine, and where along it that is.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Nearest {
    pub point: Point3<f64>,
    /// Distance from the queried point to [`Nearest::point`].
    pub distance: f64,
    /// Arc length from the start of the spine, in millimetres.
    pub arc: f64,
    /// The same, as a fraction of the whole length — 0 at the start, 1 at the end.
    ///
    /// Zero for a spine of no length at all, where every point is both ends and the fraction
    /// has no meaning; that is the one case a caller reading this as "where along" has to be
    /// prepared for, and it cannot be an error because a curve of zero length is a legitimate
    /// degenerate rather than a mistake.
    pub t: f64,
    /// Which way the curve runs where it was met, or `None` at a joint or an end.
    ///
    /// **Reported from the same search, because the alternative is two answers about one
    /// point.** A caller that found the nearest point here and then worked out a direction
    /// itself would be re-deciding which segment won, and near a joint the two searches can
    /// disagree — which is a surface lit as though it faced somewhere it does not, along
    /// exactly the seam where two segments meet.
    ///
    /// `None` where the nearest point is a joint or an end, and that is a real distinction
    /// rather than a missing value: there the curve has no single direction, and the geometry
    /// around it is a sphere rather than a cylinder. An attractor's normal needs to know which
    /// of the two it is, because the axial direction is scaled in one case and not the other.
    pub along: Option<Vector3<f64>>,
}

impl Spine {
    /// Flattens a curve to within `tolerance_mm` of itself and measures the result.
    ///
    /// **The one place a curve becomes a spine**, so that everything measuring against a curve
    /// measures against the same polyline. A pipe and an attractor reading the same curve
    /// through two flattenings would disagree about where its surface is by up to the coarser
    /// tolerance — and disagree only in the places the curve turns hardest, which is where
    /// anybody looking would be looking.
    ///
    /// A closed curve's spine is closed too, which is the detail worth having in one place: a
    /// tube around a curve missing that last segment shows two caps at the join, and an
    /// attractor missing it has a gap in its pull exactly there.
    pub fn of_curve(curve: &fieldform_scene::Curve, tolerance_mm: f64) -> Self {
        let mut points = curve.tessellate(tolerance_mm);
        if curve.is_closed() {
            if let (Some(&first), Some(&last)) = (points.first(), points.last()) {
                if (last - first).norm() > f64::EPSILON {
                    points.push(first);
                }
            }
        }
        Self::new(points)
    }

    /// Builds a spine from tessellated points, measuring as it goes.
    pub fn new(points: Vec<Point3<f64>>) -> Self {
        let mut cumulative = Vec::with_capacity(points.len());
        let mut total = 0.0;
        for (i, p) in points.iter().enumerate() {
            if i > 0 {
                total += (p - points[i - 1]).norm();
            }
            cumulative.push(total);
        }
        Self { points, cumulative }
    }

    pub fn points(&self) -> &[Point3<f64>] {
        &self.points
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Total arc length.
    pub fn length(&self) -> f64 {
        self.cumulative.last().copied().unwrap_or(0.0)
    }

    /// Unsigned distance from `p` to the spine, and nothing else.
    ///
    /// Kept separate from [`Spine::nearest`] rather than derived from it, because this one is
    /// the pipe's inner loop — called once per sample point, which is millions of times when
    /// meshing — and it can stay entirely in squared distances with a single square root at
    /// the end. `nearest` cannot: it has to interpolate the arc length of whichever segment
    /// won, which is arithmetic this does not need and should not pay for.
    ///
    /// Brute force over the segments. At a few hundred that is fine; it is the obvious place
    /// for a bounding-volume hierarchy once a measurement says so — the same note the cage
    /// field carries, for the same reason.
    pub fn distance(&self, p: Point3<f64>) -> f64 {
        if self.points.len() < 2 {
            return self
                .points
                .first()
                .map_or(f64::INFINITY, |&a| (p - a).norm());
        }
        let mut best = f64::INFINITY;
        for w in self.points.windows(2) {
            let d = point_segment_distance_squared(p, w[0], w[1]);
            if d < best {
                best = d;
            }
        }
        best.sqrt()
    }

    /// Which segment an arc length falls in, and how far along it, as a fraction.
    ///
    /// The inverse of the arithmetic `nearest` does to report `arc`, so a caller holding an arc
    /// length from there lands on the segment it came from — which is what lets a frame be read
    /// beside a nearest point without a second search deciding the segment differently.
    pub(crate) fn segment_at(&self, arc: f64) -> (usize, f64) {
        let last = self.points.len().saturating_sub(2);
        let i = match self.cumulative.binary_search_by(|c| c.partial_cmp(&arc).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
        .min(last);
        let length = self.cumulative.get(i + 1).copied().unwrap_or(0.0) - self.cumulative[i];
        let fraction = if length > 0.0 { ((arc - self.cumulative[i]) / length).clamp(0.0, 1.0) } else { 0.0 };
        (i, fraction)
    }

    /// The nearest point on the spine to `p`, how far it is, and how far along.
    ///
    /// `None` only for an empty spine, which has no points to be near.
    pub fn nearest(&self, p: Point3<f64>) -> Option<Nearest> {
        let first = *self.points.first()?;
        if self.points.len() == 1 {
            return Some(Nearest {
                along: None,
                point: first,
                distance: (p - first).norm(),
                arc: 0.0,
                t: 0.0,
            });
        }

        let mut best_squared = f64::INFINITY;
        let mut best_point = first;
        let mut best_arc = 0.0;
        let mut best_along = None;
        for (i, w) in self.points.windows(2).enumerate() {
            let (along, point) = closest_on_segment(p, w[0], w[1]);
            let squared = (p - point).norm_squared();
            // Strictly less, so the earliest segment wins a tie. Which one wins matters less
            // than that the same one always does: an arbitrary choice between two equidistant
            // segments would make the reported position along the curve flicker between two
            // values as the query point moves along the inside of a bend.
            if squared < best_squared {
                best_squared = squared;
                best_point = point;
                let segment = self.cumulative[i + 1] - self.cumulative[i];
                best_arc = self.cumulative[i] + segment * along;
                // Only where the segment was met along its length. At either end the nearest
                // point is a joint or a cap, shared with a neighbour or with nothing, and the
                // curve has no one direction there -- see `Nearest::along`.
                best_along = (along > 1e-9 && along < 1.0 - 1e-9)
                    .then(|| (w[1] - w[0]).try_normalize(1e-12))
                    .flatten();
            }
        }

        let length = self.length();
        Some(Nearest {
            along: best_along,
            point: best_point,
            distance: best_squared.sqrt(),
            arc: best_arc,
            t: if length > 0.0 { best_arc / length } else { 0.0 },
        })
    }
}

/// Where along `a`-`b` the nearest point to `p` lies, as a fraction in 0..=1, and that point.
///
/// The fraction comes back as well as the point because the caller needs it to interpolate the
/// segment's share of the arc length, and recovering it afterwards from the point would mean
/// dividing by a length that may be zero.
fn closest_on_segment(p: Point3<f64>, a: Point3<f64>, b: Point3<f64>) -> (f64, Point3<f64>) {
    let ab = b - a;
    let length_squared = ab.norm_squared();
    if length_squared <= f64::EPSILON {
        return (0.0, a);
    }
    let along = ((p - a).dot(&ab) / length_squared).clamp(0.0, 1.0);
    (along, a + ab * along)
}

/// Squared distance from `p` to the segment `a`-`b`.
///
/// Squared throughout so the inner loop has no square root; the caller takes one at the end.
fn point_segment_distance_squared(p: Point3<f64>, a: Point3<f64>, b: Point3<f64>) -> f64 {
    let ab = b - a;
    let length_squared = ab.norm_squared();
    if length_squared < f64::EPSILON {
        return (p - a).norm_squared();
    }
    let along = ((p - a).dot(&ab) / length_squared).clamp(0.0, 1.0);
    (p - (a + ab * along)).norm_squared()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight spine along X, so every answer is checkable by hand.
    fn line() -> Spine {
        Spine::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(20.0, 0.0, 0.0),
        ])
    }

    #[test]
    fn the_length_is_the_sum_of_the_segments() {
        assert!((line().length() - 20.0).abs() < 1e-12);
    }

    /// The whole point of lifting this out: the position along the curve comes back with the
    /// distance, from the same search, so the two cannot disagree about which segment won.
    #[test]
    fn a_point_beside_the_spine_reports_where_along_it_that_is() {
        let spine = line();
        let near = spine.nearest(Point3::new(5.0, 3.0, 0.0)).unwrap();
        assert!((near.point - Point3::new(5.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((near.distance - 3.0).abs() < 1e-12);
        assert!((near.arc - 5.0).abs() < 1e-12, "{}", near.arc);
        assert!((near.t - 0.25).abs() < 1e-12, "{}", near.t);
    }

    /// Arc length has to run across the join, not restart at each segment — the failure that
    /// would make a radius profile repeat itself once per tessellation segment.
    #[test]
    fn arc_length_accumulates_across_segments() {
        let spine = line();
        let near = spine.nearest(Point3::new(15.0, -2.0, 0.0)).unwrap();
        assert!((near.arc - 15.0).abs() < 1e-12, "{}", near.arc);
        assert!((near.t - 0.75).abs() < 1e-12, "{}", near.t);
    }

    /// Beyond either end the nearest point is that end, and the parameter pins to 0 or 1
    /// rather than running past them.
    #[test]
    fn a_point_past_an_end_clamps_to_it() {
        let spine = line();
        let before = spine.nearest(Point3::new(-8.0, 0.0, 0.0)).unwrap();
        assert!((before.point - Point3::new(0.0, 0.0, 0.0)).norm() < 1e-12);
        assert!(before.t.abs() < 1e-12);
        assert!((before.distance - 8.0).abs() < 1e-12);

        let after = spine.nearest(Point3::new(28.0, 0.0, 0.0)).unwrap();
        assert!((after.point - Point3::new(20.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((after.t - 1.0).abs() < 1e-12);
    }

    /// `t` runs along the length, not along the segment count. A spine whose segments differ
    /// in length is the case that tells the two apart: the midpoint by count is not the
    /// midpoint by distance, and a caller asking for "halfway" means the second.
    #[test]
    fn the_parameter_measures_length_and_not_segments() {
        let spine = Spine::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(101.0, 0.0, 0.0),
        ]);
        // The joint sits at 1 of 101 mm — one segment of two, and a hundredth of the length.
        let near = spine.nearest(Point3::new(1.0, 5.0, 0.0)).unwrap();
        assert!((near.t - 1.0 / 101.0).abs() < 1e-12, "{}", near.t);
    }

    /// The fast path and the full one have to agree, or the pipe's surface would sit somewhere
    /// the attractor's falloff does not.
    #[test]
    fn the_distance_only_path_agrees_with_the_full_one() {
        let spine = Spine::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 8.0, 0.0),
            Point3::new(20.0, -4.0, 3.0),
            Point3::new(35.0, 0.0, 0.0),
        ]);
        for i in 0..50 {
            let t = i as f64;
            let p = Point3::new(t * 0.8 - 5.0, (t * 0.3).sin() * 9.0, (t * 0.2).cos() * 6.0);
            let fast = spine.distance(p);
            let full = spine.nearest(p).unwrap().distance;
            assert!((fast - full).abs() < 1e-9, "{fast} vs {full} at {p:?}");
        }
    }

    /// A spine of one point is a legitimate degenerate — a curve tessellated to nothing — and
    /// answers rather than panicking or dividing by a zero length.
    #[test]
    fn a_degenerate_spine_answers_without_dividing_by_its_length() {
        let dot = Spine::new(vec![Point3::new(1.0, 2.0, 3.0)]);
        let near = dot.nearest(Point3::new(1.0, 2.0, 6.0)).unwrap();
        assert!((near.distance - 3.0).abs() < 1e-12);
        assert_eq!(near.t, 0.0);
        assert!(near.t.is_finite());

        assert!(Spine::new(Vec::new()).nearest(Point3::origin()).is_none());
    }
}
