// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The helical coordinate, and the two solids built on it: a coil and a thread.
//!
//! # A helix is a coordinate, not a curve
//!
//! Read a point about an axis as `(r, φ, z)` and then as `(r, ζ)`, `ζ = z − P·φ/2π`: any shape
//! drawn in the axial half-plane, read at `(r, ζ)`, is that shape carried helically about the
//! axis with pitch `P`. A thread is *defined* in the axial plane -- the standards draw the basic
//! profile there -- so this is exact by construction rather than an approximation of a swept
//! tube. A coil's wire section is defined in the same plane here, which is said so nobody reads
//! it as the section square to the wire.
//!
//! # The bound, once for both
//!
//! The map stretches: two points a small angle apart at radius `r` are `r·Δφ` apart in space and
//! `P·Δφ/2π` apart in `ζ`, so a distance read in the plane can exceed the distance in space. The
//! usual bound is the map's local stretch `√(1 + (P/2πr)²)` at the smallest radius the material
//! reaches, which holds along a straight segment that stays outside that radius -- and a query
//! inside a coil's hollow, or inside a thread's core, has its nearest material across the axis
//! from it, where no such segment exists and the bound is simply false (a point on a thread's
//! axis reads the profile at its own `ζ`, `d/2` away, where the root a half-turn round is
//! `d₁/2` away). So the divisor here is the **global** one: for any two points at radii at
//! least `r_min`, the law of cosines gives `|p − q|² ≥ Δr² + Δz² + 4·r_p·r_q·sin²(Δφ/2)`, and
//! with `sin(x/2) ≥ x/π` on `|x| ≤ π` the least of that over the angle is
//! `Δr² + Δζ²/(1 + P²/(16·r_p·r_q))`. Hence, for both radii at least `r_c`,
//!
//! ```text
//! |p − q| ≥ √(Δr² + Δζ²) / √(1 + P² / (16 r_c²)),
//! ```
//!
//! with no condition on the segment between them. That one inequality is both halves of the
//! claim: with `q` on the surface it says the divided reading never overstates, and with `p` and
//! `q` any two queries it says the divided reading changes at most one per millimetre.
//!
//! # The core, and why the bound is taken inside `r_min`
//!
//! Every surface lies at or beyond `r_min`, the least radius the material reaches; nothing holds
//! the angle near the axis, where the reading in the plane is not even continuous. So inside a
//! **handover radius** `r_c = ¾·r_min` the solids read only `r_min − r` -- the step out to the
//! nearest radius any surface has, honest and blind to the angle -- and beyond it the divided
//! reading, with the inequality taken at `r_c`. Between `r_c` and `r_min` the two are joined by
//! a ramp that equals the radial reading at `r_c` and stays below it inside, so the join is
//! continuous and each piece is 1-Lipschitz: `max(r_min − r, min(B, r + r_min − 2r_c))` for the
//! coil's empty hollow, the same mirrored for the thread's solid core. Taking the inequality at
//! `r_min` itself, with only the radial step inside, would have left a jump at `r_min` between
//! turns, where the radial reading is nought and the helical one is not.
//!
//! [`helix_stretch`] is the divisor, `√(1 + P²/(9·r_min²))`: `1.008` on an M8, `1.024` on a coil
//! of pitch six at an inner radius of nine, `1.20` on one of pitch ten at an inner radius of
//! five -- the cost is real only for steep, tight coils, as the design note expected.
//!
//! Both solids say `LowerBound` with a Lipschitz constant of one.

use std::f64::consts::{FRAC_PI_2, TAU};

use nalgebra::Point3;

use crate::profile::cap_distance;
use crate::{Normalization, ScalarField};

/// The handover radius, as a fraction of the least radius the material reaches.
const HANDOVER: f64 = 0.75;

/// The global divisor of the helical map beyond the handover radius `¾·r_min`: see the module
/// notes.
pub fn helix_stretch(pitch: f64, r_min: f64) -> f64 {
    let rc = HANDOVER * r_min;
    (1.0 + pitch * pitch / (16.0 * rc * rc)).sqrt()
}

/// The reading of a solid whose material lies at `r ≥ r_min`, from its divided helical reading
/// `helical` (asked only where `r ≥ r_c`), for an empty core (`hollow`, a coil) or a solid one
/// (a thread). See the module notes for why each piece is what it is.
fn about_the_core(r: f64, r_min: f64, hollow: bool, helical: impl FnOnce() -> f64) -> f64 {
    let rc = HANDOVER * r_min;
    // Out to the least radius, which is a distance where the core is empty and a depth where
    // it is solid.
    let radial = r_min - r;
    if r < rc {
        return if hollow { radial } else { -radial };
    }
    // Equal to the radial reading at `r_c`, below it inside, and climbing outward.
    let ramp = r + r_min - 2.0 * rc;
    let b = helical();
    match (hollow, r < r_min) {
        (true, true) => radial.max(b.min(ramp)),
        (true, false) => b.min(ramp),
        (false, true) => (-radial).min(b.max(-ramp)),
        (false, false) => b.max(-ramp),
    }
}

/// The section of a coil's wire in the axial half-plane, centred on the mean radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WireSection {
    /// A round wire of this radius.
    Circle { radius: f64 },
    /// A square wire of this half-side.
    Square { half: f64 },
}

impl WireSection {
    /// Signed distance in the plane from `(dr, dz)` off the section's centre.
    fn distance(self, dr: f64, dz: f64) -> f64 {
        match self {
            WireSection::Circle { radius } => dr.hypot(dz) - radius,
            WireSection::Square { half } => {
                let (qx, qy) = (dr.abs() - half, dz.abs() - half);
                qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0)
            }
        }
    }

    /// How far the section reaches from its centre, radially and axially alike.
    pub fn half_extent(self) -> f64 {
        match self {
            WireSection::Circle { radius } => radius,
            WireSection::Square { half } => half,
        }
    }
}

/// A wire wound about `+Z` from the `+X` half-plane at the origin, climbing with each turn.
///
/// Right-handed as built; a left-handed coil is this one read at the query mirrored across the
/// `xz`-plane (`hand = −1`), which turns `φ` around and nothing else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coil {
    /// To the section's centre.
    pub mean_radius: f64,
    pub pitch: f64,
    /// Need not be whole.
    pub turns: f64,
    pub section: WireSection,
    /// `+1` right-handed, `−1` left.
    pub hand: f64,
}

impl Coil {
    /// The smallest radius the wire reaches: what the bound is taken at.
    pub fn inner_radius(&self) -> f64 {
        self.mean_radius - self.section.half_extent()
    }

    fn stretch(&self) -> f64 {
        helix_stretch(self.pitch, self.inner_radius())
    }

    /// The box the coil lies in, in its own frame.
    pub fn bounds(&self) -> (Point3<f64>, Point3<f64>) {
        let reach = self.mean_radius + self.section.half_extent();
        let a = self.section.half_extent();
        (
            Point3::new(-reach, -reach, -a),
            Point3::new(reach, reach, self.pitch * self.turns + a),
        )
    }

    /// The reading at radius `r ≥ r_min`, angle `phi` and height `z`.
    ///
    /// **Three copies, chosen by the nearest turn and clamped to the turns that exist.** With
    /// `τ = k + φ/2π` the turn coordinate of copy `k`, the wire's material is read at
    /// `(r − R, z − P·τ)`. A point of material is *seen* by the one copy for which the angle
    /// between it and the query, taken the short way round, is the angle the copy uses -- so
    /// each copy answers for the material within half a turn of the query along the helix, and
    /// the two neighbours of the nearest copy are what the branch cut at `φ = ±π` needs. Copies
    /// beyond those read farther in `ζ`, and a round or square section reads farther with `|ζ|`,
    /// so they cannot be nearer than the neighbour on their side -- provided that neighbour is
    /// live, which is what clamping the middle copy to the last turn that exists guarantees.
    ///
    /// **A copy past an end answers for less than a turn, and says so.** Where the copy's `τ`
    /// is past `N` (or under nought) the material it answers for lies behind the end's
    /// half-plane, within a dihedral wedge from `ψ = 2π(τ − N)` to a half-turn behind the
    /// query, so its distance is at least `r·sin(min(ψ, 90°))` as well as the plane reading
    /// divided by the stretch; the larger is kept, and the exact distance to the flat end face
    /// (the revolve's `cap_distance`, Pythagoras over the landing) is taken as a ceiling, since
    /// the face is part of the solid. A copy whose material would lie wholly past the end is
    /// left out.
    fn outer(&self, r: f64, phi: f64, z: f64) -> f64 {
        let (big_r, p, n) = (self.mean_radius, self.pitch, self.turns);
        let s = self.stretch();
        let t = z / p - phi / TAU;
        // The copies whose material is not wholly past an end, for this query's angle.
        let k_hi = (n + 0.5 - phi / TAU).floor();
        let k_lo = (-0.5 - phi / TAU).ceil();
        let k0 = t.round().clamp(k_lo, k_hi);
        let mut best = f64::INFINITY;
        for k in [k0 - 1.0, k0, k0 + 1.0] {
            if k < k_lo || k > k_hi {
                continue;
            }
            let tau = k + phi / TAU;
            let mut d = self.section.distance(r - big_r, z - p * tau) / s;
            // **Inside, no deeper than the end faces are near.** Each end's face lies in a
            // half-plane through the axis, so a point of the wire a small angle in from it is no
            // further from the surface than from that plane, `r·sin(min(ψ, 90°))` -- the
            // partial revolve's depth, for the reason it has one.
            if d < 0.0 && (0.0..=n).contains(&tau) {
                let to_start = r * (TAU * tau).min(FRAC_PI_2).sin();
                let to_end = r * (TAU * (n - tau)).min(FRAC_PI_2).sin();
                d = d.max(-to_start).max(-to_end);
            }
            // Past an end: `psi` the angle past its half-plane, `z_end` where its face sits.
            let past = if tau > n {
                Some((TAU * (tau - n), p * n))
            } else if tau < 0.0 {
                Some((-TAU * tau, 0.0))
            } else {
                None
            };
            if let Some((psi, z_end)) = past {
                d = d.max(r * psi.min(FRAC_PI_2).sin());
                let (u, h) = (r * psi.cos(), r * psi.sin());
                d = d.min(cap_distance(self.section.distance(u - big_r, z - z_end), h));
            }
            best = best.min(d);
        }
        best
    }
}

impl ScalarField for Coil {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let (x, y, z) = (p.x, p.y * self.hand, p.z);
        let r = x.hypot(y);
        let phi = y.atan2(x);
        let near = about_the_core(r, self.inner_radius(), true, || self.outer(r, phi, z));
        // And within the slab the turns climb through, which is exact for the slab and lifts
        // the hollow's reading above and below the coil, where the radial step alone says little.
        let a = self.section.half_extent();
        near.max(-a - z).max(z - (self.pitch * self.turns + a))
    }

    fn normalization(&self) -> Normalization {
        Normalization::LowerBound
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

/// The metric basic profile (the classic 60° form) over one pitch, in the axial half-plane:
/// `r` against `ζ`, with the material at `r ≤ R(ζ)`.
///
/// With `H = (√3/2)·P` the fundamental triangle's height: a flat crest of width `P/8` at the
/// major radius, a flat root of width `P/4` at the minor radius `d₁/2 = d/2 − (5/8)·H`, and
/// flanks at 30° from the radial direction between them, each `5P/16` long axially --
/// `P/8 + P/4 + 2·5P/16 = P`. The crest is centred on `ζ = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreadProfile {
    pub major_radius: f64,
    pub pitch: f64,
}

impl ThreadProfile {
    /// The height of the fundamental triangle.
    pub fn triangle_height(&self) -> f64 {
        0.5 * 3f64.sqrt() * self.pitch
    }

    /// The minor radius, `d₁/2`.
    pub fn minor_radius(&self) -> f64 {
        self.major_radius - 0.625 * self.triangle_height()
    }

    /// One period's five vertices as `(ζ, r)`, from the crest's start; the next period's crest
    /// starts where the last flank ends.
    fn vertices(&self) -> [(f64, f64); 5] {
        let (p, hi, lo) = (self.pitch, self.major_radius, self.minor_radius());
        [
            (-p / 16.0, hi),
            (p / 16.0, hi),
            (6.0 * p / 16.0, lo),
            (10.0 * p / 16.0, lo),
            (15.0 * p / 16.0, hi),
        ]
    }

    /// The material's radius at `zeta`.
    pub fn radius_at(&self, zeta: f64) -> f64 {
        let p = self.pitch;
        let v = self.vertices();
        // Into the period that starts at the crest's start.
        let z = (zeta - v[0].0).rem_euclid(p) + v[0].0;
        for pair in v.windows(2) {
            let ((z0, r0), (z1, r1)) = (pair[0], pair[1]);
            if z <= z1 {
                return if z1 > z0 { r0 + (r1 - r0) * (z - z0) / (z1 - z0) } else { r0 };
            }
        }
        // Past the last flank: the next crest.
        self.major_radius
    }

    /// Signed distance in the plane to the periodic boundary: positive outside the material.
    ///
    /// Over three periods, which is enough because the nearest point of a periodic boundary
    /// lies within one period of the query.
    pub fn distance(&self, r: f64, zeta: f64) -> f64 {
        let p = self.pitch;
        let v = self.vertices();
        let z = (zeta - v[0].0).rem_euclid(p) + v[0].0;
        let mut best = f64::INFINITY;
        for shift in [-p, 0.0, p] {
            for pair in v.windows(2) {
                let ((z0, r0), (z1, r1)) = (pair[0], pair[1]);
                best = best.min(segment_distance(z, r, z0 + shift, r0, z1 + shift, r1));
            }
        }
        if r > self.radius_at(zeta) { best } else { -best }
    }
}

/// Distance from `(x, y)` to the segment `(ax, ay)`–`(bx, by)`.
fn segment_distance(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((x - ax) * dx + (y - ay) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (x - (ax + t * dx)).hypot(y - (ay + t * dy))
}

/// A threaded rod about `+Z` from `z = 0` to `z = length`, the profile's crest centred on the
/// `+X` half-plane at the origin, with flat ends.
///
/// The clearance offsets the profile *in the plane*, where the profile's distance is exact and
/// an offset is an exact offset; the helical bound is then taken at the offset profile's own
/// minor radius. Positive opens the rod out (an internal cutter), negative shrinks it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thread {
    pub profile: ThreadProfile,
    pub length: f64,
    pub clearance: f64,
    /// `+1` right-handed, `−1` left.
    pub hand: f64,
}

impl Thread {
    /// The smallest radius the surface reaches: the minor radius plus the clearance.
    pub fn inner_radius(&self) -> f64 {
        self.profile.minor_radius() + self.clearance
    }

    /// The largest.
    pub fn outer_radius(&self) -> f64 {
        self.profile.major_radius + self.clearance
    }

    /// The box the rod lies in, in its own frame.
    pub fn bounds(&self) -> (Point3<f64>, Point3<f64>) {
        let reach = self.outer_radius();
        (Point3::new(-reach, -reach, 0.0), Point3::new(reach, reach, self.length))
    }

    /// The helical part at radius `r ≥ r_min`: the plane's reading of the offset profile,
    /// divided by the stretch.
    fn flanks(&self, r: f64, zeta: f64) -> f64 {
        let s = helix_stretch(self.profile.pitch, self.inner_radius());
        (self.profile.distance(r, zeta) - self.clearance) / s
    }
}

impl ScalarField for Thread {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let (x, y, z) = (p.x, p.y * self.hand, p.z);
        let r = x.hypot(y);
        let phi = y.atan2(x);
        let zeta = z - self.profile.pitch * phi / TAU;
        let helical = about_the_core(r, self.inner_radius(), false, || self.flanks(r, zeta));
        // Cut to its length: the intersection with a slab, which is exact for the slab and keeps
        // the other side's bound.
        helical.max((-z).max(z - self.length))
    }

    fn normalization(&self) -> Normalization {
        Normalization::LowerBound
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn coil() -> Coil {
        Coil { mean_radius: 10.0, pitch: 6.0, turns: 2.5, section: WireSection::Circle { radius: 1.0 }, hand: 1.0 }
    }

    fn thread_m8() -> Thread {
        Thread { profile: ThreadProfile { major_radius: 4.0, pitch: 1.25 }, length: 20.0, clearance: 0.0, hand: 1.0 }
    }

    /// How far `p` is from a round-wire coil, by brute force: the solid is the union of the
    /// wire's disks in the axial half-planes, so this is the least distance to one of them,
    /// sampled finely along the turns. Sampling can only miss the nearest disk, so it reads a
    /// hair long, never short.
    fn coil_true_distance(c: &Coil, p: Point3<f64>) -> f64 {
        let WireSection::Circle { radius } = c.section else { unreachable!() };
        let steps = 40_000;
        let mut best = f64::INFINITY;
        for i in 0..=steps {
            let tau = c.turns * i as f64 / steps as f64;
            let ang = TAU * tau * c.hand;
            let (cs, sn) = (ang.cos(), ang.sin());
            // In the disk's own half-plane: across the axis and up it, and off the plane.
            let u = p.x * cs + p.y * sn;
            let off = -p.x * sn + p.y * cs;
            let in_plane = ((u - c.mean_radius).hypot(p.z - c.pitch * tau) - radius).max(0.0);
            // A point behind the axis is in the opposite half-plane, which the disk is not.
            let d = if u >= 0.0 { in_plane.hypot(off) } else { (u.hypot(off) + (c.mean_radius - radius)).max(in_plane.hypot(off)) };
            best = best.min(d);
        }
        best
    }

    /// Outside the material the coil never reads past the true distance, and reads close to it
    /// off the axis; on the axis it reads the inner radius.
    #[test]
    fn a_coil_never_overstates_its_distance_and_reads_it_closely_off_the_axis() {
        let c = coil();
        let mut worst_short = (1.0f64, Point3::origin());
        let mut checked = 0;
        for i in 0..400 {
            let f = i as f64;
            let p = Point3::new(
                14.0 * ((f * 0.37).sin()),
                14.0 * ((f * 0.61).cos()),
                -3.0 + 21.0 * ((f * 0.13).sin().abs()),
            );
            let read = c.distance(p);
            if read <= 0.0 {
                continue;
            }
            let truth = coil_true_distance(&c, p);
            assert!(read <= truth + 1e-3, "at {p:?}: read {read}, true {truth}");
            checked += 1;
            if truth > 0.5 && p.x.hypot(p.y) > c.inner_radius() && read / truth < worst_short.0 {
                worst_short = (read / truth, p);
            }
        }
        assert!(checked > 200, "only {checked} points outside");
        assert!(worst_short.0 > 0.6, "reads as short as {worst_short:?} of the truth off the axis");
        // On the axis the nearest wire is the inner radius away at every height.
        for z in [0.0, 3.0, 7.5, 15.0] {
            let read = c.distance(Point3::new(0.0, 0.0, z));
            assert!((read - c.inner_radius()).abs() < 1e-9, "axis at {z}: {read}");
        }
    }

    /// The wire is where it should be: inside at the section's centre on the first and last
    /// turns, outside a hair past each end, and the zero set climbs by the pitch each turn.
    #[test]
    fn a_coil_is_solid_along_its_wire_and_ends_where_its_turns_do() {
        let c = coil();
        // Away from the end faces, which are a quarter turn off here: at an end the centre lies
        // on the face, and reads nought.
        for tau in [0.25, 1.0, 1.7, 2.25] {
            let ang = TAU * tau;
            let centre = Point3::new(10.0 * ang.cos(), 10.0 * ang.sin(), 6.0 * tau);
            let d = c.distance(centre);
            assert!((d + 1.0 / c.stretch()).abs() < 1e-6, "at turn {tau}: {d}");
        }
        // Just past the last turn's end face, in the material's shadow: outside by about the
        // small angle times the radius.
        let past = TAU * 2.5 + 0.05;
        let p = Point3::new(10.0 * past.cos(), 10.0 * past.sin(), 15.0 + 0.05 * 6.0 / TAU);
        let d = c.distance(p);
        assert!(d > 0.0 && d < 0.6, "past the end: {d}");
        // And before the start.
        let before: f64 = -0.05;
        let p = Point3::new(10.0 * before.cos(), 10.0 * before.sin(), -0.05 * 6.0 / TAU);
        let d = c.distance(p);
        assert!(d > 0.0 && d < 0.6, "before the start: {d}");
        // Along a line parallel to the axis at the mean radius, a little round from the start
        // face, the wire's centre is met once a pitch.
        let a: f64 = 0.3;
        let lift = 6.0 * a / TAU;
        let crossings: Vec<f64> = (0..=1500)
            .map(|i| -2.0 + i as f64 * 0.01)
            .filter(|z| c.distance(Point3::new(10.0 * a.cos(), 10.0 * a.sin(), *z)) < -0.99 / c.stretch())
            .collect();
        for expected in [lift, 6.0 + lift, 12.0 + lift] {
            assert!(crossings.iter().any(|z| (z - expected).abs() < 0.02), "no wire centre at {expected}");
        }
    }

    /// A left-handed coil is the right-handed one's mirror image, pointwise.
    #[test]
    fn a_left_handed_coil_is_the_mirror_of_a_right_handed_one() {
        let right = coil();
        let left = Coil { hand: -1.0, ..right };
        for i in 0..200 {
            let f = i as f64;
            let p = Point3::new(13.0 * (f * 0.7).sin(), 13.0 * (f * 1.1).cos(), 16.0 * (f * 0.3).sin());
            let mirrored = Point3::new(p.x, -p.y, p.z);
            assert!((left.distance(p) - right.distance(mirrored)).abs() < 1e-12);
        }
    }

    /// A square wire reads its own section: flat on the outside at the mean radius plus half a
    /// side.
    #[test]
    fn a_square_wire_coil_reads_its_flat() {
        let c = Coil { section: WireSection::Square { half: 1.0 }, ..coil() };
        // On the +X half-plane the first turn's centre is at (10, 0, 0): the flat's outer face
        // is at r = 11 for z within the half-side.
        for z in [-0.9, 0.0, 0.9] {
            let d = c.distance(Point3::new(11.5, 0.0, z));
            assert!((d - 0.5 / c.stretch()).abs() < 1e-9, "at z {z}: {d}");
        }
    }

    /// On the axial plane the thread's field is the basic profile's exact planar distance over
    /// the stretch, and the profile's radii are the major and minor ones where they should be.
    #[test]
    fn a_thread_reads_the_basic_profile_in_its_axial_plane() {
        let t = thread_m8();
        let prof = t.profile;
        let s = helix_stretch(1.25, prof.minor_radius());
        assert!((prof.minor_radius() - 3.3234).abs() < 1e-3, "{}", prof.minor_radius());
        // Crest and root flats.
        assert!((prof.radius_at(0.0) - 4.0).abs() < 1e-12);
        assert!((prof.radius_at(0.5 * 1.25) - prof.minor_radius()).abs() < 1e-12);
        // Period.
        assert!((prof.radius_at(0.3) - prof.radius_at(0.3 + 1.25)).abs() < 1e-12);
        // Along +X (φ = 0, ζ = z) the field is the planar reading over the stretch, inside the
        // length.
        for i in 0..100 {
            let z = 2.0 + i as f64 * 0.0125;
            // From the least radius out: inside it the reading is the core's, by design.
            for r in [3.4, 3.5, 3.9, 4.3, 6.0] {
                let p = Point3::new(r, 0.0, z);
                let expected = prof.distance(r, z) / s;
                assert!((t.distance(p) - expected).abs() < 1e-9, "at r {r} z {z}");
            }
        }
        // The flanks are at 30° from radial: on a flank, moving radially inward by `x` moves
        let on_flank = |zeta: f64| prof.radius_at(zeta);
        let zeta = 0.2 * 1.25;
        let r0 = on_flank(zeta);
        let d_in = prof.distance(r0 - 0.1, zeta);
        // the planar distance by `x·sin 30°`, the flank's normal leaning 60° off the radius.
        assert!((d_in + 0.1 * (30f64.to_radians()).sin()).abs() < 1e-9, "{d_in}");
    }

    /// The zero set at `φ = 0` and at `φ = 2π − ε` differ in `z` by the pitch: the thread climbs
    /// one pitch a turn, the other way for a left hand.
    #[test]
    fn a_thread_climbs_one_pitch_a_turn_and_the_left_hand_climbs_the_other_way() {
        let t = thread_m8();
        let p = 1.25;
        // Where the crest sits along a line parallel to the axis at angle `phi`, as a phase in
        // one pitch: the circular mean of the heights at which a radius just under the major one
        // is inside, over one pitch.
        let crest = |t: &Thread, phi: f64| -> f64 {
            let (c, s) = (phi.cos(), phi.sin());
            let (mut sx, mut sy) = (0.0, 0.0);
            for i in 0..2000 {
                let z = 8.0 + p * i as f64 / 2000.0;
                if t.distance(Point3::new(3.98 * c, 3.98 * s, z)) < 0.0 {
                    let a = TAU * z / p;
                    sx += a.cos();
                    sy += a.sin();
                }
            }
            (sy.atan2(sx) / TAU * p).rem_euclid(p)
        };
        // A quarter turn on, the crest is a quarter pitch higher for a right hand and a quarter
        // lower for a left; a whole turn on, it is back where it was, one pitch up.
        let shift = |t: &Thread, phi: f64| (crest(t, phi) - crest(t, 0.0)).rem_euclid(p);
        assert!((shift(&t, FRAC_PI_2) - p / 4.0).abs() < 0.01, "right: {}", shift(&t, FRAC_PI_2));
        let left = Thread { hand: -1.0, ..t };
        assert!((shift(&left, FRAC_PI_2) - 3.0 * p / 4.0).abs() < 0.01, "left: {}", shift(&left, FRAC_PI_2));
        // Just short of a full turn the crest has climbed just short of a pitch.
        let near_full = shift(&t, TAU - 0.01);
        assert!((near_full - p * (1.0 - 0.01 / TAU)).abs() < 0.01, "a turn on: {near_full}");
    }

    /// Left and right are mirror images pointwise, and the rod ends flat at its length.
    #[test]
    fn a_left_handed_thread_is_the_mirror_of_a_right_handed_one_and_both_end_flat() {
        let right = thread_m8();
        let left = Thread { hand: -1.0, ..right };
        for i in 0..200 {
            let f = i as f64;
            let p = Point3::new(6.0 * (f * 0.7).sin(), 6.0 * (f * 1.1).cos(), 25.0 * (f * 0.3).sin());
            let mirrored = Point3::new(p.x, -p.y, p.z);
            assert!((left.distance(p) - right.distance(mirrored)).abs() < 1e-12);
        }
        assert!((right.distance(Point3::new(0.0, 0.0, 21.0)) - 1.0).abs() < 1e-12);
        assert!((right.distance(Point3::new(0.0, 0.0, -2.0)) - 2.0).abs() < 1e-12);
        // On the axis inside: the minor radius, which the plane alone would overstate.
        let axis = right.distance(Point3::new(0.0, 0.0, 10.0));
        assert!((axis + right.profile.minor_radius()).abs() < 1e-9, "{axis}");
    }

    /// Neither solid's gradient exceeds one, which with the sign tests above is what the
    /// `LowerBound` claim rests on; and the march across both never steps through a wall.
    #[test]
    fn a_coil_and_a_thread_never_read_steeper_than_one() {
        let c = coil();
        let t = thread_m8();
        let steep = Coil { pitch: 10.0, mean_radius: 6.0, ..coil() };
        for (name, field, bounds) in [
            ("coil", &c as &(dyn ScalarField + Sync), (Point3::new(-14.0, -14.0, -4.0), Point3::new(14.0, 14.0, 20.0))),
            ("steep coil", &steep, (Point3::new(-9.0, -9.0, -4.0), Point3::new(9.0, 9.0, 30.0))),
            ("thread", &t, (Point3::new(-6.0, -6.0, -2.0), Point3::new(6.0, 6.0, 22.0))),
        ] {
            if let Some(over) = crate::find_overstep(field, bounds, 24) {
                assert!(over.rate <= 1.0 + 1e-3, "{name}: rate {} at {:?}", over.rate, over.at);
            }
        }
        // Sphere-trace a bundle of rays at each; a step through a wall ends inside.
        let fields: [(&dyn ScalarField, Point3<f64>, f64); 2] =
            [(&c, Point3::new(0.0, 0.0, 7.5), 14.0), (&t, Point3::new(0.0, 0.0, 10.0), 8.0)];
        for (field, from, radius) in fields {
            for i in 0..300 {
                let f = i as f64;
                let dir = Vector3::new((f * 0.7).sin(), (f * 1.3).cos(), (f * 0.37).sin() * 0.7).normalize();
                let start = from - dir * radius;
                // A ray has to start outside to be able to step through anything.
                if field.distance(start) <= 0.0 {
                    continue;
                }
                let mut p = start;
                let mut hit = false;
                for _ in 0..400 {
                    let d = field.distance(p);
                    if d < 1e-4 {
                        hit = d > -1e-3;
                        break;
                    }
                    p += dir * d;
                    if (p - start).norm() > 2.0 * radius {
                        hit = true;
                        break;
                    }
                }
                assert!(hit, "ray {i} stepped through a wall");
            }
        }
    }
}
