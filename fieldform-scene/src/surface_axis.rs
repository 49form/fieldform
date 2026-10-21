// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! An axis stood on a body's surface, placed by a U and a V and given a length.
//!
//! **Why this is a record and not just a line.** A plain axis is two typed points, and that is right
//! for one drawn through space. One stood on a surface should not be: its start is *where the body
//! is*, so it has to move when the body does, and the only numbers it owns are the place on the
//! surface and how far it reaches. Typing X, Y and Z for it would be editing something the body
//! decides. So the curve it draws is still an ordinary two-point line — everything that takes a
//! line takes it, and an older build draws the stored points identically — and this record, beside
//! it in the document, says how those two points are *derived* (`Document::surface_axes`).
//!
//! **Two ways to say where on the surface, and the default is the one an SDF is good at.**
//!
//! A **seed** is a point, held 0 to 1 across the body's box, and the axis stands at the surface
//! point *nearest* it. That is one gradient walk on a field whose whole job is to answer "how far
//! to the surface", and it works on every body: no face is unreachable, there are no poles, and a
//! torus or a C-shaped shell is no harder than a ball. Held across the box rather than in world
//! coordinates so that it still follows the body, which is the whole reason this is a record and
//! not two typed points. Its one weakness is honest and local: near the medial axis, where two
//! walls are equally near, a small move of the seed swaps which wall the axis stands on.
//!
//! A **ray** is the older way and is kept, because it is the one that gives a predictable grid on a
//! flat face: run a ray along one axis through the point `(u, v)` of the two across the box, and
//! stand the axis where it first meets the surface. `u` and `v` each run 0 to 1. It is a planar
//! projection, so it stretches near a sphere's pole and **cannot reach a face parallel to the ray**
//! — on a cube with the ray along Z, the four sides are unreachable at every `(u, v)`.
//!
//! A spherical projection was weighed and not built. It would cast from the body's centroid, which
//! reaches every face of anything star-shaped — but for a torus or a C-shape the centroid is in
//! empty space and the ray can miss the material or cross it twice, so it trades "cannot reach some
//! faces" for "cannot describe some bodies". The nearest point has neither failure.
//!
//! Either way an implicit surface has no U/V of its own; only a cage face does, and a parametric
//! operation names parts, never faces (the boundary rule in `graph/mod.rs`).

use serde::{Deserialize, Serialize};

use crate::graph::Axis;
use crate::id::NodeId;

/// Where an axis stands on a body, and how long it is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceAxis {
    /// The body it stands on — a block of the active part's graph, referred to by id as every
    /// reference here is, never by one of its faces.
    pub body: NodeId,
    /// The axis the ray runs along. `u` and `v` run across the other two.
    pub direction: Axis,
    /// Whether the ray comes from the minus side. By default it comes from the plus side, the way
    /// you look down at the top of something.
    pub flipped: bool,
    /// 0 to 1 across the body's box, along the first of the two axes across the ray.
    pub u: f64,
    /// The same along the second.
    pub v: f64,
    /// How far the line reaches from the surface, in millimetres, along the field's direction.
    pub length: f64,
    /// Where to look from, 0 to 1 across the body's box in each axis — and then the surface point
    /// *nearest* it is where the axis stands.
    ///
    /// **`None` is the ray**, which is what every file written before this field existed means, so
    /// an older document opens unchanged and a reader that does not know the field still gets the
    /// two stored points. Written only when it is there, which is the rule every side entry here
    /// follows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<[f64; 3]>,
}

impl SurfaceAxis {
    /// What an axis reaches when nothing says: long enough to see on a default-sized body.
    pub const DEFAULT_LENGTH_MM: f64 = 50.0;

    /// The index (0 = X) of the axis the ray runs along.
    pub fn direction_index(&self) -> usize {
        axis_index(self.direction)
    }

    /// The indices of the axes `u` and `v` run along, for this direction: X looks across (Y, Z),
    /// Y across (Z, X), Z across (X, Y) — cyclic, so a right-handed look down each.
    pub fn across(&self) -> (usize, usize) {
        let d = self.direction_index();
        ((d + 1) % 3, (d + 2) % 3)
    }

    /// The ray that finds this axis's place on a body whose box is `lo..hi`: where it starts and
    /// which way it runs (a unit vector). It starts `margin` clear of the box on the side it comes
    /// from, so it begins outside whatever the body is.
    pub fn ray(&self, lo: [f64; 3], hi: [f64; 3], margin: f64) -> ([f64; 3], [f64; 3]) {
        let d = self.direction_index();
        let (a, b) = self.across();
        let mut origin = [0.0; 3];
        origin[a] = lo[a] + self.u * (hi[a] - lo[a]);
        origin[b] = lo[b] + self.v * (hi[b] - lo[b]);
        let mut dir = [0.0; 3];
        if self.flipped {
            origin[d] = lo[d] - margin;
            dir[d] = 1.0;
        } else {
            origin[d] = hi[d] + margin;
            dir[d] = -1.0;
        }
        (origin, dir)
    }

    /// The `u` and `v` that put the ray through `point`, for a box `lo..hi` and this direction. The
    /// inverse of [`Self::ray`]'s across-coordinates, so an axis made from a clicked point can be
    /// reproduced from its numbers.
    ///
    /// A box with no width across an axis has no meaningful position along it, and reads as the
    /// middle rather than dividing by nothing.
    pub fn place_of(
        direction: Axis,
        point: [f64; 3],
        lo: [f64; 3],
        hi: [f64; 3],
    ) -> (f64, f64) {
        let probe = SurfaceAxis {
            body: NodeId::from_raw(0),
            direction,
            flipped: false,
            u: 0.0,
            v: 0.0,
            length: 0.0,
            seed: None,
        };
        let (a, b) = probe.across();
        let along = |i: usize| {
            let width = hi[i] - lo[i];
            if width.abs() < 1e-12 { 0.5 } else { (point[i] - lo[i]) / width }
        };
        (along(a), along(b))
    }
}

impl SurfaceAxis {
    /// Where the seed is in space, for a body whose box is `lo..hi`.
    ///
    /// Across the box rather than in world coordinates, so a body that moves carries its axis with
    /// it — the same bargain `u` and `v` make, in three axes instead of two.
    pub fn seed_point(&self, lo: [f64; 3], hi: [f64; 3]) -> Option<[f64; 3]> {
        let seed = self.seed?;
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = lo[i] + seed[i] * (hi[i] - lo[i]);
        }
        Some(out)
    }

    /// The seed that puts [`Self::seed_point`] back at `point`. The inverse of it, so an axis made
    /// from a clicked point can be reproduced from its numbers.
    ///
    /// A box with no width along an axis reads as the middle there, which is `place_of`'s rule for
    /// the same reason: there is no position to have along an axis of no extent.
    pub fn seed_of(point: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for i in 0..3 {
            let width = hi[i] - lo[i];
            out[i] = if width.abs() < 1e-12 { 0.5 } else { (point[i] - lo[i]) / width };
        }
        out
    }
}

/// The index of an axis, X first.
pub(crate) fn axis_index(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(direction: Axis, flipped: bool, u: f64, v: f64) -> SurfaceAxis {
        SurfaceAxis {
            body: NodeId::from_raw(1),
            direction,
            flipped,
            u,
            v,
            length: 50.0,
            seed: None,
        }
    }

    const LO: [f64; 3] = [-10.0, -20.0, -30.0];
    const HI: [f64; 3] = [30.0, 40.0, 50.0];

    /// X looks across (Y, Z), Y across (Z, X), Z across (X, Y): the ray's start lands at the right
    /// fraction of the right two axes, and clear of the box on the third.
    #[test]
    fn the_ray_runs_through_the_place_u_and_v_name_across_the_other_two_axes() {
        let (origin, dir) = at(Axis::Z, false, 0.25, 0.75).ray(LO, HI, 5.0);
        assert_eq!(origin, [-10.0 + 0.25 * 40.0, -20.0 + 0.75 * 60.0, 55.0]);
        assert_eq!(dir, [0.0, 0.0, -1.0], "from above, looking down");

        let (origin, dir) = at(Axis::X, false, 0.5, 0.5).ray(LO, HI, 5.0);
        assert_eq!(origin, [35.0, 10.0, 10.0], "u is Y and v is Z for a ray along X");
        assert_eq!(dir, [-1.0, 0.0, 0.0]);

        let (origin, dir) = at(Axis::Y, false, 0.0, 1.0).ray(LO, HI, 5.0);
        assert_eq!(origin, [30.0, 45.0, -30.0], "u is Z and v is X for a ray along Y");
        assert_eq!(dir, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn flipping_comes_from_the_other_side_and_keeps_the_place() {
        let (up, _) = at(Axis::Z, false, 0.3, 0.6).ray(LO, HI, 5.0);
        let (down, dir) = at(Axis::Z, true, 0.3, 0.6).ray(LO, HI, 5.0);
        assert_eq!((up[0], up[1]), (down[0], down[1]), "the same place across the ray");
        assert_eq!(down[2], -35.0, "starting below the box");
        assert_eq!(dir, [0.0, 0.0, 1.0]);
    }

    /// `place_of` is the inverse of the across part of `ray`: what a clicked point reads as is what
    /// puts the ray back through it.
    #[test]
    fn the_place_of_a_point_puts_the_ray_back_through_it() {
        let point = [12.5, 3.0, -7.0];
        for direction in Axis::ALL {
            let (u, v) = SurfaceAxis::place_of(direction, point, LO, HI);
            let (origin, _) = SurfaceAxis { u, v, ..at(direction, false, 0.0, 0.0) }.ray(LO, HI, 1.0);
            let d = axis_index(direction);
            for i in 0..3 {
                if i != d {
                    assert!((origin[i] - point[i]).abs() < 1e-9, "{direction:?} axis {i}");
                }
            }
        }
    }

    #[test]
    fn a_box_with_no_width_across_an_axis_reads_as_the_middle_there() {
        let flat_lo = [0.0, 0.0, 5.0];
        let flat_hi = [10.0, 10.0, 5.0];
        // Looking along X, u runs along Y and v along Z; Z has no width.
        let (u, v) = SurfaceAxis::place_of(Axis::X, [0.0, 2.5, 5.0], flat_lo, flat_hi);
        assert_eq!((u, v), (0.25, 0.5));
    }

    /// The seed is held across the box, so it goes back to the point it was taken from — and a
    /// body that moves carries it along.
    #[test]
    fn a_seed_is_held_across_the_box_so_it_moves_with_the_body() {
        let point = [12.5, 3.0, -7.0];
        let seed = SurfaceAxis::seed_of(point, LO, HI);
        let axis = SurfaceAxis { seed: Some(seed), ..at(Axis::Z, false, 0.0, 0.0) };
        let back = axis.seed_point(LO, HI).expect("a seed");
        for i in 0..3 {
            assert!((back[i] - point[i]).abs() < 1e-9, "axis {i}");
        }

        // The same body, moved ten along X: the seed lands ten further along too. This is the
        // whole reason the seed is a fraction of the box and not a point in space.
        let moved_lo = [LO[0] + 10.0, LO[1], LO[2]];
        let moved_hi = [HI[0] + 10.0, HI[1], HI[2]];
        let carried = axis.seed_point(moved_lo, moved_hi).expect("a seed");
        assert!((carried[0] - (point[0] + 10.0)).abs() < 1e-9, "it follows the body");
        assert!((carried[1] - point[1]).abs() < 1e-9);
    }

    #[test]
    fn a_box_with_no_width_along_an_axis_seeds_the_middle_there() {
        let flat_lo = [0.0, 0.0, 5.0];
        let flat_hi = [10.0, 10.0, 5.0];
        let seed = SurfaceAxis::seed_of([2.5, 5.0, 5.0], flat_lo, flat_hi);
        assert_eq!(seed, [0.25, 0.5, 0.5]);
    }

    /// An older file has no seed, and means the ray. The field is left out when it is not there,
    /// so a document that never used one is byte-for-byte what it was.
    #[test]
    fn a_file_without_a_seed_reads_as_the_ray_and_writes_no_field_for_it() {
        let axis = at(Axis::Y, true, 0.125, 0.875);
        let json = serde_json::to_string(&axis).unwrap();
        assert!(!json.contains("seed"), "nothing written for a seed that is not there: {json}");

        // The older file *is* this one: a record with no seed writes no key for it, so what an
        // older build wrote is byte-for-byte what is written here. Read back rather than
        // handwritten, so the case cannot pass against a spelling the format does not use.
        let read: SurfaceAxis = serde_json::from_str(&json).unwrap();
        assert_eq!(read.seed, None, "an older file means the ray");
        assert_eq!(read.u, 0.125);
    }

    #[test]
    fn a_surface_axis_round_trips_through_json() {
        let axis = at(Axis::Y, true, 0.125, 0.875);
        let json = serde_json::to_string(&axis).unwrap();
        assert_eq!(serde_json::from_str::<SurfaceAxis>(&json).unwrap(), axis);
        assert!(json.contains("\"direction\":\"y\""), "an axis is written in the vocabulary a pattern's is: {json}");
    }
}
