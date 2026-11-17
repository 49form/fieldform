// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The oloid and its family: the convex hull of two equal circles in perpendicular planes.
//!
//! In its own frame one circle lies in the XY plane about `(−d/2, 0, 0)` and the other in the XZ
//! plane about `(+d/2, 0, 0)`, both of radius `r`, with `d = spread · r` between their centres.
//! A spread of one puts each circle through the other's centre, which is the oloid; √2 is the
//! two-circle roller, whose centre of mass keeps its height as it rolls. Every spread is a
//! convex solid bounded by one developable surface and two circular creases.
//!
//! # Why this is an exact distance and not a bound
//!
//! A plane that touches both circles contains a tangent line of each, and two coplanar lines
//! in perpendicular planes meet on the line the planes share -- the X axis. Naming that
//! meeting point by `τ = r / x` gives every tangent plane of the surface in one line:
//!
//! ```text
//! τ·x + a(τ)·y + b(τ)·z = r,   a² = (1 + kτ/2)² − τ²,   b² = (1 − kτ/2)² − τ²
//! ```
//!
//! for `τ` in `±2/(2 + k)`, `k` the spread, and the other three quarters by the signs of `y`
//! and `z`. The plane touches the solid along the segment between its two points of contact.
//!
//! - **Inside**, the depth of a convex solid is the distance to its nearest supporting plane,
//!   and the largest ball about an interior point can only touch the surface where it is
//!   smooth, never on a crease. So the largest of the planes above *is* the signed distance.
//! - **Outside**, the same maximum is the distance wherever the foot of the perpendicular lands
//!   on the plane's own segment. Where it does not, the nearest point is on a crease, and a
//!   crease is an arc of a circle, whose distance is closed-form.
//!
//! The maximum over `τ` is one-dimensional and smooth, so it is found by a coarse walk and a
//! golden-section refinement -- the one search here, written the same way in the shader. `τ` is
//! walked as `−τₘ·cos 2u`, which takes the square roots' ends out of `a` and `b`: both are
//! smooth in `u` from one symmetry plane to the other.

use crate::primitive::round_detail;
use crate::{Normalization, ScalarField};
use nalgebra::{Point3, Vector3};

/// Steps of the coarse walk over a quarter of the surface. The function walked is a sum of
/// three slow waves in `u`, so it has a maximum or two and a dozen steps cannot straddle one.
const WALK: usize = 12;
/// Golden-section steps: two walk steps narrowed by 0.618⁴⁰, far below what an `f64` resolves.
const REFINE: usize = 40;

/// The convex hull of two perpendicular circles -- see the module's notes.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Oloid {
    /// Midway between the two circles' centres.
    pub centre: Point3<f64>,
    /// Both circles' radius.
    pub radius: f64,
    /// The distance between the circles' centres, in radii. One is the oloid.
    pub spread: f64,
}

/// One tangent plane and its segment of contact, in the folded frame `y, z ≥ 0`.
struct Tangent {
    /// The plane's unit normal.
    normal: Vector3<f64>,
    /// `normal · x − offset` is the signed distance to the plane.
    offset: f64,
    /// Where it touches the XY circle and the XZ circle.
    contact: [Vector3<f64>; 2],
}

impl Oloid {
    fn shape(&self) -> (f64, f64) {
        (self.radius.abs(), self.spread.abs())
    }

    /// The tangent plane at `u` in `[0, π/2]`, from the XZ symmetry plane round to the XY one.
    fn tangent(&self, u: f64) -> Tangent {
        let (r, k) = self.shape();
        let (s, c) = u.sin_cos();
        let tau = -2.0 / (2.0 + k) * (c * c - s * s);
        let lean = tau * (0.5 * k - 1.0);
        let a = std::f64::consts::SQRT_2 * s * (1.0 + lean).max(0.0).sqrt();
        let b = std::f64::consts::SQRT_2 * c * (1.0 - lean).max(0.0).sqrt();
        let scale = 1.0 / (tau * tau + a * a + b * b).sqrt();
        let (near, far) = (1.0 + 0.5 * k * tau, 1.0 - 0.5 * k * tau);
        let half = 0.5 * k * r;
        Tangent {
            normal: Vector3::new(tau, a, b) * scale,
            offset: r * scale,
            contact: [
                Vector3::new(-half + r * tau / near, r * a / near, 0.0),
                Vector3::new(half + r * tau / far, 0.0, r * b / far),
            ],
        }
    }

    /// The tangent plane standing furthest out from `x`, a folded point.
    fn furthest_plane(&self, x: Vector3<f64>) -> (f64, Tangent) {
        let reading = |u: f64| {
            let t = self.tangent(u);
            t.normal.dot(&x) - t.offset
        };
        let step = std::f64::consts::FRAC_PI_2 / WALK as f64;
        let mut best = (f64::NEG_INFINITY, 0);
        for i in 0..=WALK {
            let v = reading(i as f64 * step);
            if v > best.0 {
                best = (v, i);
            }
        }
        let mut lo = (best.1.saturating_sub(1)) as f64 * step;
        let mut hi = (best.1 + 1).min(WALK) as f64 * step;
        const GOLD: f64 = 0.618_033_988_749_894_9;
        let (mut m1, mut m2) = (hi - GOLD * (hi - lo), lo + GOLD * (hi - lo));
        let (mut v1, mut v2) = (reading(m1), reading(m2));
        for _ in 0..REFINE {
            if v1 > v2 {
                hi = m2;
                (m2, v2) = (m1, v1);
                m1 = hi - GOLD * (hi - lo);
                v1 = reading(m1);
            } else {
                lo = m1;
                (m1, v1) = (m2, v2);
                m2 = lo + GOLD * (hi - lo);
                v2 = reading(m2);
            }
        }
        let u = 0.5 * (lo + hi);
        (reading(u), self.tangent(u))
    }

    /// The nearest point of the two creases to `x`, a folded point: each is an arc of its
    /// circle, ending where the surface leaves it for the other circle's symmetry plane.
    fn nearest_on_creases(&self, x: Vector3<f64>) -> Vector3<f64> {
        let (r, k) = self.shape();
        let half = 0.5 * k * r;
        // The arcs stop where the tangent line reaches the far circle's tip: cos = 1/(1 + k).
        let stop = 1.0 / (1.0 + k);
        let rise = (1.0 - stop * stop).max(0.0).sqrt();
        // The XY circle, about (−half, 0, 0), keeps the arc facing away from the other.
        let (ax, ay) = (x.x + half, x.y);
        let rho = ax.hypot(ay);
        let on_xy = if ax <= rho * stop && rho > 0.0 {
            Vector3::new(-half + r * ax / rho, r * ay / rho, 0.0)
        } else {
            Vector3::new(-half + r * stop, r * rise, 0.0)
        };
        // And the XZ circle, about (+half, 0, 0), the same mirrored.
        let (bx, bz) = (x.x - half, x.z);
        let rho = bx.hypot(bz);
        let on_xz = if -bx <= rho * stop && rho > 0.0 {
            Vector3::new(half + r * bx / rho, 0.0, r * bz / rho)
        } else {
            Vector3::new(half - r * stop, 0.0, r * rise)
        };
        if (x - on_xy).norm_squared() <= (x - on_xz).norm_squared() { on_xy } else { on_xz }
    }

    /// The signed distance at a folded point, and the direction it rises in.
    fn folded(&self, x: Vector3<f64>) -> (f64, Option<Vector3<f64>>) {
        let (r, _) = self.shape();
        if !(r > 0.0) {
            let d = x.norm();
            return (d, (d > 1e-12).then(|| x / d));
        }
        let (reading, plane) = self.furthest_plane(x);
        if reading <= 0.0 {
            return (reading, Some(plane.normal));
        }
        let [p, q] = plane.contact;
        let along = q - p;
        let t = (x - p).dot(&along) / along.norm_squared().max(1e-300);
        if (0.0..=1.0).contains(&t) {
            return (reading, Some(plane.normal));
        }
        let offset = x - self.nearest_on_creases(x);
        let d = offset.norm();
        // A crease point is on the solid, so it is never nearer than the plane says.
        (d.max(reading), (d > 1e-12).then(|| offset / d))
    }

    fn fold(&self, p: Point3<f64>) -> (Vector3<f64>, Vector3<f64>) {
        let q = p - self.centre;
        let sign = Vector3::new(1.0, if q.y < 0.0 { -1.0 } else { 1.0 }, if q.z < 0.0 { -1.0 } else { 1.0 });
        (Vector3::new(q.x, q.y.abs(), q.z.abs()), sign)
    }
}

impl ScalarField for Oloid {
    fn distance(&self, p: Point3<f64>) -> f64 {
        self.folded(self.fold(p).0).0
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        self.distance_and_normal(p).1
    }

    fn distance_and_normal(&self, p: Point3<f64>) -> (f64, Option<Vector3<f64>>) {
        let (x, sign) = self.fold(p);
        let (d, n) = self.folded(x);
        (d, n.map(|n| n.component_mul(&sign)))
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }

    fn detail_within(&self, lo: Point3<f64>, hi: Point3<f64>) -> Option<f64> {
        round_detail(self, lo, hi, self.radius.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probes(seed: u64, n: usize, reach: f64) -> Vec<Point3<f64>> {
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        (0..n).map(|_| Point3::new(next() * reach, next() * reach, next() * reach)).collect()
    }

    fn solid(spread: f64) -> Oloid {
        Oloid { centre: Point3::new(3.0, -2.0, 1.0), radius: 10.0, spread }
    }

    const SPREADS: [f64; 5] = [0.4, 1.0, std::f64::consts::SQRT_2, 2.0, 3.0];

    /// The surface as points, made with nothing the field uses: for each point of the XY
    /// circle, the point of the XZ circle whose tangent line meets its own on the X axis, and
    /// the segment between them -- then mirrored into all four quarters.
    fn surface_points(o: &Oloid, rulings: usize, along: usize) -> Vec<Point3<f64>> {
        let (r, k) = o.shape();
        let half = 0.5 * k * r;
        let mut out = Vec::new();
        let start = (1.0 / (1.0 + k)).acos();
        for i in 0..=rulings {
            let theta = start + (std::f64::consts::PI - start) * i as f64 / rulings as f64;
            // r/cos θ − r/cos φ = d, both measured from each circle's own centre.
            let cos_phi = (theta.cos() / (1.0 - k * theta.cos())).clamp(-1.0, 1.0);
            let p = Vector3::new(-half + r * theta.cos(), r * theta.sin(), 0.0);
            let q = Vector3::new(half + r * cos_phi, 0.0, r * (1.0 - cos_phi * cos_phi).sqrt());
            for j in 0..=along {
                let at = p + (q - p) * (j as f64 / along as f64);
                for (sy, sz) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                    out.push(o.centre + Vector3::new(at.x, at.y * sy, at.z * sz));
                    // And the same segment seen from the other circle, which the solid's own
                    // symmetry swaps with this one: even steps round one circle are crowded
                    // steps round the other, and sparse where the other's arc ends.
                    out.push(o.centre + Vector3::new(-at.x, at.z * sz, at.y * sy));
                }
            }
        }
        out
    }

    /// **Every point of the surface reads nought**, at every spread: the surface built from the
    /// two circles alone, with none of the field's own planes in it.
    #[test]
    fn the_segments_between_matched_tangents_lie_on_the_surface() {
        for k in SPREADS {
            let o = solid(k);
            for p in surface_points(&o, 400, 16) {
                assert!(o.distance(p).abs() < 1e-7, "spread {k} at {p:?}: {}", o.distance(p));
            }
        }
    }

    /// **And the reading is the distance to that surface**, inside and out, against the nearest
    /// of a dense cloud of its points -- which can only read long, and by no more than the
    /// cloud's own spacing.
    #[test]
    fn the_reading_is_the_distance_to_the_nearest_surface_point() {
        for k in SPREADS {
            let o = solid(k);
            let cloud = surface_points(&o, 1500, 120);
            for p in probes(17, 300, 28.0) {
                let nearest = cloud.iter().map(|c| (c - p).norm()).fold(f64::INFINITY, f64::min);
                let read = o.distance(p).abs();
                assert!(read <= nearest + 1e-9, "spread {k} at {p:?}: reads {read}, a point is {nearest} away");
                assert!(nearest - read < 0.05, "spread {k} at {p:?}: reads {read} against {nearest}");
            }
        }
    }

    /// **Inside is the hull**: a point between a point of one disc and a point of the other is
    /// in the solid, and a point pushed out past a tangent plane is not.
    #[test]
    fn the_sign_is_the_convex_hull_of_the_two_circles() {
        for k in SPREADS {
            let o = solid(k);
            let (r, half) = (10.0, 0.5 * k * 10.0);
            let mut s = probes(23, 600, 1.0).into_iter();
            while let (Some(a), Some(b)) = (s.next(), s.next()) {
                let (ta, tb) = (a.x * std::f64::consts::PI, b.x * std::f64::consts::PI);
                let on_xy = Vector3::new(-half + r * a.y.abs() * ta.cos(), r * a.y.abs() * ta.sin(), 0.0);
                let on_xz = Vector3::new(half + r * b.y.abs() * tb.cos(), 0.0, r * b.y.abs() * tb.sin());
                let within = o.centre + on_xy + (on_xz - on_xy) * (0.5 + 0.5 * a.z);
                assert!(o.distance(within) < 1e-9, "spread {k}: {within:?} is in the hull");
            }
            for p in surface_points(&o, 60, 6) {
                let (_, n) = o.distance_and_normal(p + (p - o.centre) * 1e-3);
                let out = p + n.expect("a direction off the surface") * 0.5;
                assert!(o.distance(out) > 0.4, "spread {k}: {out:?} is outside");
            }
        }
    }

    /// **Every new field source gets the unit-gradient property test**: a convex solid's
    /// distance is smooth everywhere outside it, creases and all.
    #[test]
    fn an_oloid_has_unit_gradient_outside() {
        for k in SPREADS {
            let o = solid(k);
            let away: Vec<Point3<f64>> =
                probes(5, 3000, 40.0).into_iter().filter(|p| o.distance(*p) > 0.5).collect();
            assert!(away.len() > 1000);
            let worst = crate::gradient_deviation(&o, &away, 1e-5);
            assert!(worst < 1e-6, "spread {k}: {worst}");
        }
    }

    /// **And the direction it gives is that gradient**, on both sides of the surface.
    #[test]
    fn the_normal_is_the_gradient_of_the_distance() {
        let h = 1e-5;
        for k in SPREADS {
            let o = solid(k);
            let mut checked = 0;
            for p in probes(29, 1200, 26.0) {
                let (d, n) = o.distance_and_normal(p);
                // Deep inside, two planes can stand equally near; a step from that ridge the
                // direction is one plane's and the difference is the other's.
                if d < -0.5 && (o.distance(p + Vector3::x() * 0.05) - d).abs() > 0.0499 {
                    continue;
                }
                let g = Vector3::new(
                    o.distance(p + Vector3::x() * h) - o.distance(p - Vector3::x() * h),
                    o.distance(p + Vector3::y() * h) - o.distance(p - Vector3::y() * h),
                    o.distance(p + Vector3::z() * h) - o.distance(p - Vector3::z() * h),
                ) / (2.0 * h);
                if (g.norm() - 1.0).abs() > 1e-4 {
                    continue; // on a ridge of the inside, where there is no one gradient
                }
                assert!((n.unwrap() - g).norm() < 1e-4, "spread {k} at {p:?}: {n:?} against {g:?}");
                checked += 1;
            }
            assert!(checked > 800, "spread {k}: only {checked} points had a gradient to compare");
        }
    }

    /// **The oloid's own numbers**: it is as long as three radii and as wide as two, and its
    /// far tips are the far points of the circles.
    #[test]
    fn an_oloid_is_three_radii_long_and_two_across() {
        let o = Oloid { centre: Point3::origin(), radius: 10.0, spread: 1.0 };
        for tip in [Point3::new(15.0, 0.0, 0.0), Point3::new(-15.0, 0.0, 0.0), Point3::new(-5.0, 10.0, 0.0), Point3::new(5.0, 0.0, -10.0)] {
            assert!(o.distance(tip).abs() < 1e-9, "{tip:?}");
        }
        assert!((o.distance(Point3::new(17.0, 0.0, 0.0)) - 2.0).abs() < 1e-9);
        // Each circle passes through the other's centre, which is inside the solid.
        assert!(o.distance(Point3::new(5.0, 0.0, 0.0)) < -1.0);
    }
}
