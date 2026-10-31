// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A drilled hole as a field: its section in the axial half-plane, revolved, read in closed form.
//!
//! The section is a polyline from the entry down the wall -- out to each radius at each step --
//! and closed by the drill's point or a flat bottom, or, for a hole that goes through, left open:
//! the last wall runs on along the axis without end. Every point of a revolved section's solid
//! that is nearest a query lies in the query's own half-plane, and the section touches the axis,
//! so the distance in the half-plane *is* the distance in space: a plain hole is exact, and a
//! blend into one is a true fillet.
//!
//! A tapped hole is its tap drill and a [`crate::Thread`] in one, by `min` -- sign-only, so the
//! thread's bound and the drill's distance combine without a blend question, and the whole is a
//! bound.

use nalgebra::{Point3, Vector3};

use crate::{Normalization, ScalarField, Thread};

/// A hole's section in `(r, z)`, `z` running into the material from the entry at nought.
///
/// **It starts above the entry, by a lead-in.** A cutter whose mouth lies exactly in the body's
/// face is the one join a difference cannot describe: `max(A, −H)` reads nought all across the
/// mouth, so a ray stops there and the hole is drawn capped by a film that is not material. So
/// the wall carries on straight up from the entry by [`HoleSection::lead_in`] -- above a convex
/// entry that is air, and on a curved one it clears the rim the tangent plane would leave.
#[derive(Debug, Clone, PartialEq)]
pub struct HoleSection {
    /// The wall from the top of the lead-in down, `(r, z)`, each step's `z` no less than the
    /// last's; the first is at `z = −lead`, the second at the entry. A blind hole's last point is
    /// where its full diameter ends.
    pub wall: Vec<(f64, f64)>,
    /// Where the point of a blind hole meets the axis, below the last wall point: at the same
    /// depth for a flat bottom. `None` for a hole that goes through.
    pub tip: Option<f64>,
}

impl HoleSection {
    /// The radius of the material removed at depth `z`, or `None` above the entry or below the
    /// bottom. Taken along the wall and down the point; a horizontal step is passed over, so a
    /// counterbore's shoulder belongs to the narrower radius below it.
    pub fn radius_at(&self, z: f64) -> Option<f64> {
        if z < self.wall[0].1 {
            return None;
        }
        let last = *self.wall.last()?;
        let tip = self.tip.map(|t| (0.0, t));
        let points = self.wall.iter().copied().chain(tip);
        let mut previous: Option<(f64, f64)> = None;
        for (r, at) in points {
            if let Some((r0, z0)) = previous
                && at > z0
                && z <= at
            {
                return Some(r0 + (r - r0) * (z - z0) / (at - z0));
            }
            previous = Some((r, at));
        }
        // Past the last point: still the wall for a through hole, nothing for a blind one.
        (self.tip.is_none() && z >= last.1).then_some(last.0)
    }

    /// The signed distance in the half-plane to the material removed.
    pub fn distance(&self, r: f64, z: f64) -> f64 {
        let seg = |a: (f64, f64), b: (f64, f64)| -> f64 {
            let (dx, dz) = (b.0 - a.0, b.1 - a.1);
            let len2 = dx * dx + dz * dz;
            let t = if len2 > 0.0 { (((r - a.0) * dx + (z - a.1) * dz) / len2).clamp(0.0, 1.0) } else { 0.0 };
            (r - (a.0 + t * dx)).hypot(z - (a.1 + t * dz))
        };
        let first = self.wall[0];
        // The mouth, across the top of the lead-in.
        let mut best = seg((0.0, first.1), first);
        for pair in self.wall.windows(2) {
            best = best.min(seg(pair[0], pair[1]));
        }
        let last = *self.wall.last().expect("a hole has a wall");
        best = best.min(match self.tip {
            Some(t) => seg(last, (0.0, t)),
            // Through: the last wall runs on without end.
            None => {
                if z >= last.1 { (r - last.0).abs() } else { (r - last.0).hypot(z - last.1) }
            }
        });
        let inside = self.radius_at(z).is_some_and(|radius| r <= radius);
        if inside { -best } else { best }
    }

    /// How far down the hole reaches along its axis, `None` for one that goes through.
    pub fn bottom(&self) -> Option<f64> {
        self.tip.map(|t| t.max(self.wall.last().map_or(0.0, |p| p.1)))
    }

    /// The widest radius anywhere in it.
    pub fn widest(&self) -> f64 {
        self.wall.iter().map(|p| p.0).fold(0.0, f64::max)
    }

    /// How far above the entry the cutter starts, for a mouth of radius `rim`: the rim itself,
    /// with a floor of a millimetre. Shared with the shader, which works the section out again.
    pub fn lead_in(rim: f64) -> f64 {
        rim.max(1.0)
    }

    /// How far above the entry this one starts.
    pub fn lead(&self) -> f64 {
        -self.wall[0].1
    }
}

/// The frame a hole is read in: the entry, the axis into the material, and two directions across
/// it. The across pair is made the same way on the CPU and in the shader -- the axis crossed with
/// `+Z`, or with `+X` where the axis is near `Z` -- so a tapped hole's thread starts at the same
/// angle on both.
pub fn hole_basis(axis: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let helper = if axis.z.abs() < 0.9 { Vector3::z() } else { Vector3::x() };
    let e1 = axis.cross(&helper).normalize();
    let e2 = axis.cross(&e1);
    (e1, e2)
}

/// Which way is into the material at a point on a body's surface: the field's gradient there,
/// turned round and **normalised**, since it is unit length only where the field is a true
/// distance and most bodies worth drilling are bounds. `None` where the gradient vanishes.
///
/// What the hole tool stores as the second of its two points, one millimetre along this.
pub fn into_material(field: &dyn ScalarField, at: Point3<f64>) -> Option<Vector3<f64>> {
    let h = 1e-4;
    let d = |o: Vector3<f64>| field.distance(at + o) - field.distance(at - o);
    let g = Vector3::new(d(Vector3::x() * h), d(Vector3::y() * h), d(Vector3::z() * h));
    (-g).try_normalize(1e-12)
}

/// A hole, placed: its section about the axis from `entry` along `axis`, and its thread if it is
/// tapped.
#[derive(Debug, Clone)]
pub struct Hole {
    pub entry: Point3<f64>,
    /// Unit, into the material.
    pub axis: Vector3<f64>,
    pub section: HoleSection,
    /// In the hole's own frame, about `+Z` from the entry, read through [`hole_basis`].
    pub thread: Option<Thread>,
}

impl Hole {
    /// The query in the hole's frame: across the axis twice and along it.
    pub fn local(&self, p: Point3<f64>) -> Point3<f64> {
        let q = p - self.entry;
        let (e1, e2) = hole_basis(self.axis);
        Point3::new(q.dot(&e1), q.dot(&e2), q.dot(&self.axis))
    }
}

impl ScalarField for Hole {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let l = self.local(p);
        let drilled = self.section.distance(l.x.hypot(l.y), l.z);
        match &self.thread {
            // Started where the lead-in starts, for the reason the section does, and as much
            // longer: the thread is read from the top of the lead-in down.
            Some(thread) => drilled.min(thread.distance(Point3::new(l.x, l.y, l.z + self.section.lead()))),
            None => drilled,
        }
    }

    fn normalization(&self) -> Normalization {
        if self.thread.is_some() { Normalization::LowerBound } else { Normalization::Distance }
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple(radius: f64, depth: f64, tip: Option<f64>) -> HoleSection {
        HoleSection { wall: vec![(radius, 0.0), (radius, depth)], tip }
    }

    /// A flat blind hole's section reads the distance to a rectangle against the axis: down the
    /// wall, across the bottom, across the mouth, and through the corners.
    #[test]
    fn a_flat_hole_reads_the_rectangle_it_is() {
        let s = simple(3.0, 10.0, Some(10.0));
        assert_eq!(s.distance(5.0, 5.0), 2.0);
        assert_eq!(s.distance(1.0, 12.0), 2.0);
        assert_eq!(s.distance(1.0, -2.0), 2.0);
        assert!((s.distance(6.0, 14.0) - 5.0).abs() < 1e-12);
        assert_eq!(s.distance(0.0, 5.0), -3.0);
        assert_eq!(s.distance(2.0, 9.5), -0.5);
    }

    /// A through hole has no bottom: as deep as you like, the wall is the nearest thing.
    #[test]
    fn a_through_hole_has_no_bottom() {
        let s = HoleSection { wall: vec![(3.0, 0.0)], tip: None };
        // Below the mouth by more than the radius, so the mouth is not the nearer.
        for z in [3.5, 10.0, 1.0e4] {
            assert_eq!(s.distance(0.0, z), -3.0);
            assert_eq!(s.distance(4.0, z), 1.0);
        }
        assert_eq!(s.distance(1.0, -1.0), 1.0);
        assert_eq!(s.bottom(), None);
    }

    /// A counterbore's shoulder is a step: the wide radius above it and the narrow one below.
    #[test]
    fn a_counterbore_steps_at_its_depth() {
        let s = HoleSection { wall: vec![(5.5, 0.0), (5.5, 6.5), (3.3, 6.5), (3.3, 20.0)], tip: Some(20.0) };
        assert_eq!(s.radius_at(3.0), Some(5.5));
        assert_eq!(s.radius_at(10.0), Some(3.3));
        // Under the shoulder, beside the narrow bore: the shoulder is above.
        assert!((s.distance(4.5, 7.5) - 1.0).abs() < 1e-12);
        assert!(s.distance(4.5, 5.5) < 0.0);
    }

    /// A drill's point has the included angle asked for: its zero set at radius `r` sits
    /// `(R − r)/tan(α/2)` below the shoulder.
    #[test]
    fn a_drill_point_has_its_angle() {
        let (radius, depth, angle) = (3.0f64, 10.0, 118.0f64);
        let tip = depth + radius / (angle.to_radians() * 0.5).tan();
        let s = simple(radius, depth, Some(tip));
        for r in [0.5, 1.5, 2.5] {
            let z = depth + (radius - r) / (angle.to_radians() * 0.5).tan();
            assert!(s.distance(r, z).abs() < 1e-9, "at r {r}: {}", s.distance(r, z));
            assert!(s.distance(r, z - 0.01) < 0.0 && s.distance(r, z + 0.01) > 0.0);
        }
    }

    /// The basis is right-handed and square, so a thread read in it winds the hand it says.
    #[test]
    fn the_basis_is_square_and_right_handed() {
        for axis in [Vector3::z(), -Vector3::z(), Vector3::new(0.3, -0.8, 0.2).normalize(), Vector3::x()] {
            let (e1, e2) = hole_basis(axis);
            assert!(e1.dot(&axis).abs() < 1e-12 && e2.dot(&axis).abs() < 1e-12 && e1.dot(&e2).abs() < 1e-12);
            assert!((e1.cross(&e2) - axis).norm() < 1e-12);
        }
    }
}
