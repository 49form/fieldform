// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Questions asked of a field's surface: where a line crosses it, the surface point nearest a
//! seed, and which way it faces.
//!
//! **In the kernel so every caller asks the same function.** The console, the application and a
//! tool driving them from outside must not come to different answers about the same body, and
//! these were application code until a point on a surface needed them from three places.

use crate::ScalarField;
use nalgebra::{Point3, Vector3};

/// Samples a single line may take before the search gives up.
pub const CROSSING_SAMPLES: usize = 8192;
/// Crossings a single line may report before the search gives up.
pub const CROSSING_LIMIT: usize = 256;
/// The angle between two surface normals, in degrees, past which a point is called a crease.
pub const CREASE_ANGLE_DEG: f64 = 20.0;

/// One place a line meets the surface.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Crossing {
    /// Distance along the line from its origin.
    pub t: f64,
    pub point: Point3<f64>,
    /// The value goes from positive to negative along the line here: the line enters material.
    pub entering: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SearchError {
    /// More samples or more crossings than a line may have. A partial list would be an answer
    /// that is silently missing a wall, so there is none.
    #[error("the line crosses the surface too often to list")]
    TooDense,
}

/// Every place the line `origin + t·direction` meets the surface of `field`, in order of `t`.
///
/// `direction` is normalised here. `bounds` is the box the surface can be in, which clips the
/// search; it is grown by a tenth of its diagonal and a millimetre so a surface lying on the box
/// is not clipped off. **The box only bounds the search, it never places anything.**
///
/// **Why it steps by the value.** A field that bounds its distance cannot be crossed in a step of
/// `|f|`, divided by its Lipschitz constant when that is more than one. So the only crossings
/// skipped are pairs closer together than `tolerance`, below the field's own resolution. Where the
/// sign changes between two samples the bracket is bisected down to `tolerance·1e-3`. A touch with
/// no sign change is not a crossing.
///
/// Not `trace_field`, which stops at the first confirmed hit and steps past near-misses; that is
/// right for picking and wrong for choosing among walls.
pub fn crossings(
    field: &dyn ScalarField,
    origin: Point3<f64>,
    direction: Vector3<f64>,
    bounds: (Point3<f64>, Point3<f64>),
    tolerance: f64,
) -> Result<Vec<Crossing>, SearchError> {
    let Some(direction) = direction.try_normalize(1e-12) else { return Ok(Vec::new()) };
    let tolerance = tolerance.max(1e-9);
    let margin = (bounds.1 - bounds.0).norm() * 0.1 + 1.0;
    let Some((t0, t1)) = clip_to_box(
        origin,
        direction,
        bounds.0 - Vector3::repeat(margin),
        bounds.1 + Vector3::repeat(margin),
    ) else {
        return Ok(Vec::new());
    };
    let slow = field.lipschitz().max(1.0);
    let at = |t: f64| field.distance(origin + direction * t);

    let mut out = Vec::new();
    let mut t = t0;
    let mut f = at(t);
    for _ in 0..CROSSING_SAMPLES {
        if t >= t1 {
            return Ok(out);
        }
        let next = (t + (f.abs() / slow).max(tolerance)).min(t1);
        let g = at(next);
        if (f < 0.0) != (g < 0.0) {
            let (mut lo, mut hi, mut flo) = (t, next, f);
            for _ in 0..200 {
                if hi - lo <= tolerance * 1e-3 {
                    break;
                }
                let mid = 0.5 * (lo + hi);
                let fm = at(mid);
                if (fm < 0.0) == (flo < 0.0) {
                    lo = mid;
                    flo = fm;
                } else {
                    hi = mid;
                }
            }
            let t_hit = 0.5 * (lo + hi);
            out.push(Crossing { t: t_hit, point: origin + direction * t_hit, entering: f >= 0.0 });
            if out.len() > CROSSING_LIMIT {
                return Err(SearchError::TooDense);
            }
        }
        t = next;
        f = g;
    }
    // Ran out of samples before the end of the line.
    if t < t1 { Err(SearchError::TooDense) } else { Ok(out) }
}

/// The parameter interval of a line inside a box, or `None` when it misses.
fn clip_to_box(
    origin: Point3<f64>,
    direction: Vector3<f64>,
    lo: Point3<f64>,
    hi: Point3<f64>,
) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for i in 0..3 {
        if direction[i].abs() < 1e-15 {
            if origin[i] < lo[i] || origin[i] > hi[i] {
                return None;
            }
        } else {
            let (a, b) = ((lo[i] - origin[i]) / direction[i], (hi[i] - origin[i]) / direction[i]);
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
    }
    // A line that starts inside is searched from where it starts only forward: a ray, not a
    // line. One that starts past the box has nothing ahead of it.
    let t0 = t0.max(0.0);
    (t1 > t0).then_some((t0, t1))
}

/// Which way the field increases at `p`, as a unit vector.
///
/// **Normalised explicitly.** A gradient is unit length only for a `Distance` field, and most of
/// the interesting targets -- a lattice, a lean extrude -- are `LowerBound`. The block's own
/// analytic normal is taken when it has one; otherwise a central difference of its distance, at
/// a step the caller sizes to the part.
pub fn surface_normal(field: &dyn ScalarField, p: Point3<f64>, h: f64) -> Option<Vector3<f64>> {
    if let Some(n) = field.normal(p).and_then(|n| n.try_normalize(1e-9)) {
        return Some(n);
    }
    let at = |dx: f64, dy: f64, dz: f64| field.distance(p + Vector3::new(dx, dy, dz));
    let gradient = Vector3::new(
        at(h, 0.0, 0.0) - at(-h, 0.0, 0.0),
        at(0.0, h, 0.0) - at(0.0, -h, 0.0),
        at(0.0, 0.0, h) - at(0.0, 0.0, -h),
    );
    gradient.try_normalize(1e-9)
}

/// The surface point nearest `from`, found by walking the field's own gradient.
///
/// **What an SDF is for.** The value at a point *is* the distance to the surface, so stepping
/// `-f(p)` along the unit gradient lands on it in one move for a true distance field, and creeps in
/// for one that understates (a twist, a lattice, a blend) -- which is why this iterates rather than
/// stepping once. Converged when the reading is inside `tolerance`, which is the same tolerance the
/// field was compiled at.
///
/// It works where a projection cannot: every face is reachable because there is no ray to be
/// parallel to one, and a torus or a C-shaped shell is no harder than a ball. The one place it is
/// awkward is the medial axis, where two walls are equally near and a small move of the seed swaps
/// which one wins; the caller that cares remembers where the point stood.
///
/// `None` when the walk does not settle, which a caller reports rather than papering over: a seed
/// exactly on the medial axis of a symmetric body has no nearest point to give.
pub fn walk_to_surface(
    field: &dyn ScalarField,
    from: Point3<f64>,
    tolerance: f64,
    step: f64,
) -> Option<Point3<f64>> {
    let mut at = from;
    let mut best: Option<(f64, Point3<f64>)> = None;
    for _ in 0..64 {
        let reading = field.distance(at);
        if reading.abs() <= tolerance {
            return Some(at);
        }
        // Kept, so a walk that wanders is still answered by the closest it ever came rather than
        // by nothing. A field that understates can step short for ever and never cross.
        if best.is_none_or(|(d, _)| reading.abs() < d) {
            best = Some((reading.abs(), at));
        }
        let down = surface_normal(field, at, step)?;
        at -= down * reading;
    }
    // Within a few tolerances is a surface point for this purpose; further and the walk has not
    // found one and the caller should say so.
    best.filter(|(d, _)| *d <= tolerance * 8.0).map(|(_, p)| p)
}

/// The widest angle, in degrees, between the normals at four points `4·tolerance` around `p`
/// across its tangent plane, each walked back onto the surface.
///
/// Past [`CREASE_ANGLE_DEG`] the point is on a crease and its normal is an average of faces: not
/// a fault of the point, which is where it was asked to be, but a normal that means little. Points
/// whose walk back does not settle are left out, and `None` is answered when fewer than two are
/// left to compare.
pub fn crease_angle(
    field: &dyn ScalarField,
    p: Point3<f64>,
    normal: Vector3<f64>,
    tolerance: f64,
    step: f64,
) -> Option<f64> {
    let normal = normal.try_normalize(1e-12)?;
    let helper = if normal.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
    let u = normal.cross(&helper).normalize();
    let v = normal.cross(&u);
    let reach = 4.0 * tolerance;
    let normals: Vec<Vector3<f64>> = [u, -u, v, -v]
        .iter()
        .filter_map(|d| {
            let q = walk_to_surface(field, p + d * reach, tolerance, step)?;
            surface_normal(field, q, step)
        })
        .collect();
    if normals.len() < 2 {
        return None;
    }
    let mut widest = 0.0f64;
    for (i, a) in normals.iter().enumerate() {
        for b in &normals[i + 1..] {
            widest = widest.max(a.dot(b).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    Some(widest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cuboid, Sphere};

    fn ball(r: f64) -> Sphere {
        Sphere { centre: Point3::origin(), radius: r }
    }

    /// **A line through a ball crosses twice, entering and then leaving**, at the radius either
    /// side of the centre.
    #[test]
    fn a_line_through_a_ball_enters_and_leaves_at_its_radius() {
        let b = ball(10.0);
        let hits = crossings(
            &b,
            Point3::new(-30.0, 0.0, 0.0),
            Vector3::x(),
            (Point3::new(-10.0, -10.0, -10.0), Point3::new(10.0, 10.0, 10.0)),
            1e-3,
        )
        .expect("a ball is sparse");
        assert_eq!(hits.len(), 2);
        assert!(hits[0].entering && !hits[1].entering);
        assert!((hits[0].point.x + 10.0).abs() < 1e-5, "{:?}", hits[0]);
        assert!((hits[1].point.x - 10.0).abs() < 1e-5, "{:?}", hits[1]);
    }

    /// **A line past the box has no crossings and a line that points away from it has none**,
    /// rather than the far side of the box seen backwards.
    #[test]
    fn a_line_that_misses_or_points_away_has_no_crossings() {
        let b = ball(10.0);
        let bounds = (Point3::new(-10.0, -10.0, -10.0), Point3::new(10.0, 10.0, 10.0));
        let miss = crossings(&b, Point3::new(-30.0, 50.0, 0.0), Vector3::x(), bounds, 1e-3).unwrap();
        assert!(miss.is_empty());
        let away = crossings(&b, Point3::new(-30.0, 0.0, 0.0), -Vector3::x(), bounds, 1e-3).unwrap();
        assert!(away.is_empty());
    }

    /// **A line grazing no wall at all reports nothing**: a touch with no sign change is not a
    /// crossing.
    #[test]
    fn a_touch_without_a_sign_change_is_not_a_crossing() {
        let b = ball(10.0);
        let bounds = (Point3::new(-10.0, -10.0, -10.0), Point3::new(10.0, 10.0, 10.0));
        let hits = crossings(&b, Point3::new(-30.0, 10.0 + 1e-3, 0.0), Vector3::x(), bounds, 1e-3)
            .unwrap();
        assert!(hits.is_empty(), "{hits:?}");
    }

    /// **A slab hit end-on from outside gives the entering face first**, with two walls both
    /// listed in order -- the list a facing filter chooses from.
    #[test]
    fn a_slab_lists_both_faces_in_order() {
        let slab = Cuboid { centre: Point3::origin(), size: Vector3::new(40.0, 40.0, 10.0) };
        let hits = crossings(
            &slab,
            Point3::new(3.0, 4.0, 50.0),
            -Vector3::z(),
            (Point3::new(-20.0, -20.0, -5.0), Point3::new(20.0, 20.0, 5.0)),
            1e-3,
        )
        .unwrap();
        assert_eq!(hits.len(), 2);
        assert!((hits[0].point.z - 5.0).abs() < 1e-5 && hits[0].entering);
        assert!((hits[1].point.z + 5.0).abs() < 1e-5 && !hits[1].entering);
    }

    /// **The walk from a seed lands on the sphere along the radius.**
    #[test]
    fn the_walk_from_a_seed_lands_on_the_nearest_point_of_a_ball() {
        let b = ball(10.0);
        let p = walk_to_surface(&b, Point3::new(3.0, 4.0, 0.0), 1e-6, 1e-6).expect("settles");
        assert!((p.coords - Vector3::new(6.0, 8.0, 0.0)).norm() < 1e-5, "{p:?}");
    }

    /// **A normal off a flat face is that face's, and a point on a face is not a crease.**
    #[test]
    fn a_flat_face_has_its_own_normal_and_is_no_crease() {
        let c = Cuboid { centre: Point3::origin(), size: Vector3::new(40.0, 40.0, 10.0) };
        let p = Point3::new(3.0, 4.0, 5.0);
        let n = surface_normal(&c, p, 1e-6).expect("has one");
        assert!((n - Vector3::z()).norm() < 1e-6, "{n:?}");
        let angle = crease_angle(&c, p, n, 1e-3, 1e-6).expect("four neighbours");
        assert!(angle < 1.0, "{angle}");
    }

    /// **On an edge, the neighbours disagree by a right angle**, which is what makes it a crease.
    #[test]
    fn a_point_on_an_edge_is_a_crease() {
        let c = Cuboid { centre: Point3::origin(), size: Vector3::new(40.0, 40.0, 10.0) };
        let p = Point3::new(3.0, 20.0, 5.0);
        let n = Vector3::new(0.0, 1.0, 1.0).normalize();
        let angle = crease_angle(&c, p, n, 1e-2, 1e-6).expect("four neighbours");
        assert!(angle > CREASE_ANGLE_DEG, "{angle}");
    }
}

#[cfg(test)]
mod dense_tests {
    use super::*;

    /// A bound on distance with a wall every `pi` millimetres along X, the shape of a sheet
    /// lattice met end-on.
    struct Waves;
    impl ScalarField for Waves {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.x.sin()
        }
    }

    /// **Every crossing is found, in order, at the position sampling finds**, which is the test
    /// that the step by value skips nothing a fine sampling sees.
    #[test]
    fn every_wall_a_fine_sampling_finds_is_found_in_count_and_order() {
        let bounds = (Point3::new(0.5, -1.0, -1.0), Point3::new(40.0, 1.0, 1.0));
        let hits = crossings(&Waves, Point3::new(-5.0, 0.0, 0.0), Vector3::x(), bounds, 1e-3)
            .expect("under budget");
        let tol = 1e-3;
        // Sampled at tol/10 over the same span the search covers: x from the grown box's start.
        let mut expect = Vec::new();
        let step = tol / 10.0;
        let (lo, hi) = (bounds.0.x - (bounds.1 - bounds.0).norm() * 0.1 - 1.0, bounds.1.x + (bounds.1 - bounds.0).norm() * 0.1 + 1.0);
        let mut x = lo;
        while x < hi {
            if (x.sin() < 0.0) != ((x + step).sin() < 0.0) {
                expect.push(x + 0.5 * step);
            }
            x += step;
        }
        assert_eq!(hits.len(), expect.len(), "{} vs {}", hits.len(), expect.len());
        for (h, e) in hits.iter().zip(&expect) {
            assert!((h.point.x - e).abs() < tol, "{} vs {e}", h.point.x);
        }
        assert!(hits.windows(2).all(|w| w[0].t < w[1].t && w[0].entering != w[1].entering));
    }

    /// **Too many walls is refused, not truncated**: a partial list is a wall silently missing.
    #[test]
    fn too_many_walls_is_refused_rather_than_cut_short() {
        let bounds = (Point3::new(0.0, -1.0, -1.0), Point3::new(2000.0, 1.0, 1.0));
        let r = crossings(&Waves, Point3::new(-5.0, 0.0, 0.0), Vector3::x(), bounds, 1e-3);
        assert_eq!(r, Err(SearchError::TooDense));
    }
}
