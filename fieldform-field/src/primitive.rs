// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Closed-form solids: sphere, cube, cylinder, cone, torus and capsule.
//!
//! The first geometry in this codebase that is neither swept from a curve nor derived from a
//! cage — raw shapes, exact everywhere, and cheap enough that nothing about them ever needs
//! baking. They exist because a field kernel that can only make what someone imported is
//! awkward to try anything in: a sphere and a cube are how you check a blend radius, a
//! boolean, or a bake without first drawing a curve.
//!
//! Every one is an exact signed distance function rather than a bound, so they compose with
//! blends (which need true distances) as well as with booleans.
//!
//! # Why none of them carries an axis
//!
//! Everything here with a direction points along **+Z**, and takes any other orientation from
//! a `Node::Transform` wrapped around it. That is deliberate, and it is the one design
//! decision in this module worth arguing for: a cylinder storing its own axis *and* sitting
//! under a placement would describe its orientation twice, and two descriptions of one piece
//! of state can disagree. Rotating the body would leave the stored axis stale, and whichever
//! one a given consumer happened to read would decide what the user saw. A primitive with no
//! opinion about direction cannot have that bug.

use crate::{Normalization, ScalarField};
use nalgebra::{Matrix3, Point2, Point3, Vector2, Vector3};

/// A sphere: the simplest exact distance field there is.
#[derive(Copy, Clone, Debug)]
pub struct Sphere {
    pub centre: Point3<f64>,
    pub radius: f64,
}

impl ScalarField for Sphere {
    fn distance(&self, p: Point3<f64>) -> f64 {
        (p - self.centre).norm() - self.radius
    }

    /// Exact everywhere except the centre, where the surface is equidistant in every
    /// direction and there is no normal to give.
    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let offset = p - self.centre;
        let length = offset.norm();
        (length > 1e-12).then(|| offset / length)
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// An axis-aligned box, by the length of its sides.
///
/// Three extents rather than one, and named for the general shape rather than for the cube it
/// was first asked for: the field below always computed per-axis, so a cube was never a
/// simpler case here — only a narrower signature over the same arithmetic. A cube is this with
/// three equal sides, and the graph builds one from its node's three sizes like any other box.
#[derive(Copy, Clone, Debug)]
pub struct Cuboid {
    pub centre: Point3<f64>,
    /// Full side lengths, not half-extents: it is what a person types off a drawing.
    pub size: Vector3<f64>,
}

impl Cuboid {
    /// The equal-sided case, which is what a cube is.
    ///
    /// The graph builds a cube from its node's three sizes like any other box, so only the tests
    /// ask for one this way, and only the tests compile it (decision 1.4). `pub(crate)` because
    /// `sdf`'s tests build one as well as this module's.
    #[cfg(test)]
    pub(crate) fn cube(centre: Point3<f64>, size: f64) -> Self {
        Self {
            centre,
            size: Vector3::new(size, size, size),
        }
    }

    /// Distance along each axis to the box's slab, negative inside.
    fn axis_distances(&self, p: Point3<f64>) -> Vector3<f64> {
        let half = self.size * 0.5;
        let offset = p - self.centre;
        Vector3::new(
            offset.x.abs() - half.x,
            offset.y.abs() - half.y,
            offset.z.abs() - half.z,
        )
    }
}

impl ScalarField for Cuboid {
    /// The standard exact box distance: outside, the length of the positive part; inside, the
    /// least negative axis, which is the distance to the nearest face.
    ///
    /// Exact in both regions, which matters here — a blend mixes by magnitude, so a box that
    /// reported the axis distance outside as well (the common shortcut) would fillet wrongly
    /// near its corners and look plausible doing it.
    fn distance(&self, p: Point3<f64>) -> f64 {
        let d = self.axis_distances(p);
        let outside = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
        outside.norm() + d.x.max(d.y).max(d.z).min(0.0)
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let d = self.axis_distances(p);
        let offset = p - self.centre;
        let sign = Vector3::new(
            if offset.x < 0.0 { -1.0 } else { 1.0 },
            if offset.y < 0.0 { -1.0 } else { 1.0 },
            if offset.z < 0.0 { -1.0 } else { 1.0 },
        );
        if d.x.max(d.y).max(d.z) > 0.0 {
            // Outside: the gradient of the length of the positive part.
            let outside = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
            let length = outside.norm();
            return (length > 1e-12).then(|| {
                Vector3::new(
                    sign.x * outside.x / length,
                    sign.y * outside.y / length,
                    sign.z * outside.z / length,
                )
            });
        }
        // Inside: the nearest face is the least negative axis, and the normal is its own.
        let axis = if d.x >= d.y && d.x >= d.z {
            Vector3::new(sign.x, 0.0, 0.0)
        } else if d.y >= d.z {
            Vector3::new(0.0, sign.y, 0.0)
        } else {
            Vector3::new(0.0, 0.0, sign.z)
        };
        Some(axis)
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// The direction straight out from the +Z axis, or `None` on the axis itself where there is
/// no such direction.
///
/// Shared by everything below that is round about Z, so the degenerate case is decided once.
fn radial_direction(q: Vector3<f64>) -> Option<Vector3<f64>> {
    let rho = radius_about_z(q);
    (rho > 1e-12).then(|| Vector3::new(q.x / rho, q.y / rho, 0.0))
}

/// The distance from the +Z axis.
fn radius_about_z(q: Vector3<f64>) -> f64 {
    (q.x * q.x + q.y * q.y).sqrt()
}

/// The frame in which a shape drawn between two points becomes one drawn about +Z.
///
/// Everything below that spans a segment works this way: build the frame once, evaluate the
/// axis-aligned form in it, and rotate the answer back. That keeps one implementation of each
/// shape's actual distance — the axis-aligned one, which the tests below exercise directly —
/// rather than a second arbitrary-orientation copy that could drift from it.
///
/// The two perpendiculars are chosen arbitrarily, which is sound because every shape using
/// this is rotationally symmetric about the segment: a different choice rotates the local
/// point and rotates the normal back by the same amount, so the world-space answer is
/// unchanged. That is also why the shader can pick its perpendicular differently and still
/// agree.
#[derive(Copy, Clone, Debug)]
struct Segment {
    midpoint: Point3<f64>,
    /// `u` and `v` span the plane across the segment; `w` runs along it.
    u: Vector3<f64>,
    v: Vector3<f64>,
    w: Vector3<f64>,
    length: f64,
}

impl Segment {
    /// `None` when the two ends coincide, where there is no direction to build a frame from.
    /// Callers degenerate to a sphere there rather than dividing by a zero length.
    fn new(from: Point3<f64>, to: Point3<f64>) -> Option<Self> {
        let along = to - from;
        let length = along.norm();
        if length < 1e-12 {
            return None;
        }
        let w = along / length;
        // Cross with whichever axis `w` leans on least, so the product is never near zero.
        let least = if w.x.abs() <= w.y.abs() && w.x.abs() <= w.z.abs() {
            Vector3::x()
        } else if w.y.abs() <= w.z.abs() {
            Vector3::y()
        } else {
            Vector3::z()
        };
        let u = w.cross(&least).normalize();
        Some(Segment {
            midpoint: Point3::from((from.coords + to.coords) * 0.5),
            u,
            v: w.cross(&u),
            w,
            length,
        })
    }

    /// The query point, in the frame where the segment is centred on the origin along +Z.
    fn to_local(&self, p: Point3<f64>) -> Point3<f64> {
        let q = p - self.midpoint;
        Point3::new(q.dot(&self.u), q.dot(&self.v), q.dot(&self.w))
    }

    /// A direction computed in the local frame, back in world space.
    fn to_world(&self, n: Vector3<f64>) -> Vector3<f64> {
        self.u * n.x + self.v * n.y + self.w * n.z
    }
}

/// A box whose faces are not square to one another — a parallelepiped, by its three edge vectors.
///
/// **Its own exact distance, because shear cannot be a transform here.** `Node::Transform` is rigid
/// motion plus a uniform scale on purpose: those keep a distance a distance. A general linear map
/// does not — the distance in the sheared space is not the distance in the world, so a sheared
/// `Cuboid` would have to report a lower bound and every blend against it would be marked
/// approximate (`cad-toolset-map.md` §1). This is the shape itself, so it is exact in both regions.
///
/// # The arithmetic
///
/// Write a point as `p = centre + t₀·e₀ + t₁·e₁ + t₂·e₂`, so the box is `|tᵢ| ≤ ½`. The gradient of
/// `tᵢ` is the *i*-th row of the inverse of the edge matrix, which makes the signed distance from `p`
/// to the plane `tᵢ = c` simply `(tᵢ − c)/|∇tᵢ|`.
///
/// - **Inside**, the distance to the surface is the nearest face plane: `max ᵢ (|tᵢ| − ½)/|∇tᵢ|`,
///   which is negative. The foot of the perpendicular to the nearest plane lies inside the box (the
///   ball of that radius is inside it), so this is exact.
/// - **Outside**, the nearest surface point is either inside a face — the foot of the perpendicular
///   to that face's plane, if it falls within the parallelogram — or on an edge or at a corner. So the
///   distance is the least over the six faces (taken only where the foot is inside) and the twelve
///   edges as segments. Every candidate is a real surface point, so the least is never below the
///   truth, and the truth is always among them: exact, with no square-box shortcut that would be
///   wrong at every skewed edge and corner.
#[derive(Copy, Clone, Debug)]
pub struct SkewedBox {
    centre: Point3<f64>,
    edges: [Vector3<f64>; 3],
    /// `t = inverse · (p − centre)`: row *i* is the gradient of the *i*-th edge coordinate.
    inverse: Matrix3<f64>,
}

impl SkewedBox {
    /// A skewed box about `centre` with these three full edge vectors, or `None` when they lie in a
    /// plane — then there is no volume to be inside, and nothing here would be a distance.
    ///
    /// "In a plane" is judged against the edges' own lengths, not against nought: edges a hair off
    /// coplanar invert to enormous numbers and read as a box of vanishing thickness, which is the
    /// same failure by a different road.
    pub fn new(centre: Point3<f64>, edges: [Vector3<f64>; 3]) -> Option<Self> {
        let matrix = Matrix3::from_columns(&edges);
        let scale = edges.iter().map(Vector3::norm).product::<f64>();
        if !(matrix.determinant().abs() > 1e-9 * scale) {
            return None;
        }
        Some(Self { centre, edges, inverse: matrix.try_inverse()? })
    }

    /// The eight corners, in no order a caller should rely on.
    ///
    /// The extent and the shader read the node's centre and edges themselves, so only the tests
    /// here ask for the corners, and only the tests compile this (decision 1.4).
    #[cfg(test)]
    fn corners(&self) -> [Point3<f64>; 8] {
        let mut out = [self.centre; 8];
        for (i, corner) in out.iter_mut().enumerate() {
            let sign = |bit: usize| if i >> bit & 1 == 1 { 0.5 } else { -0.5 };
            *corner = self.centre
                + self.edges[0] * sign(0)
                + self.edges[1] * sign(1)
                + self.edges[2] * sign(2);
        }
        out
    }

    /// The gradient of edge coordinate `i`: the direction it rises fastest, with its own length.
    fn gradient_of(&self, i: usize) -> Vector3<f64> {
        self.inverse.row(i).transpose()
    }

    /// Signed distance to each pair of faces, positive beyond the slab.
    fn slabs(&self, t: &Vector3<f64>) -> [f64; 3] {
        [0, 1, 2].map(|i| (t[i].abs() - 0.5) / self.gradient_of(i).norm())
    }

    /// The nearest point of the surface to `p`, for a point outside — see the type's doc.
    fn nearest_on_surface(&self, p: Point3<f64>) -> Point3<f64> {
        let q = p - self.centre;
        let t = self.inverse * q;
        let mut best = (f64::INFINITY, self.centre);
        let mut offer = |at: Vector3<f64>| {
            let d = (q - at).norm();
            if d < best.0 {
                best = (d, self.centre + at);
            }
        };
        // A face: the foot of the perpendicular, if it lands inside the parallelogram.
        for i in 0..3 {
            let g = self.gradient_of(i);
            for c in [-0.5, 0.5] {
                let foot = q - g * ((t[i] - c) / g.norm_squared());
                let tf = self.inverse * foot;
                if (0..3).all(|j| j == i || tf[j].abs() <= 0.5 + 1e-12) {
                    offer(foot);
                }
            }
        }
        // An edge: the nearest point of a segment, which includes both its ends — the corners.
        for k in 0..3 {
            let (j, l) = ((k + 1) % 3, (k + 2) % 3);
            for (a, b) in [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)] {
                let start = self.edges[k] * -0.5 + self.edges[j] * a + self.edges[l] * b;
                let along = self.edges[k];
                let u = ((q - start).dot(&along) / along.norm_squared()).clamp(0.0, 1.0);
                offer(start + along * u);
            }
        }
        self.centre + (best.1 - self.centre)
    }
}

impl ScalarField for SkewedBox {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let t = self.inverse * (p - self.centre);
        let s = self.slabs(&t);
        let nearest_face = s[0].max(s[1]).max(s[2]);
        if nearest_face <= 0.0 {
            return nearest_face;
        }
        (p - self.nearest_on_surface(p)).norm()
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let t = self.inverse * (p - self.centre);
        let s = self.slabs(&t);
        let (axis, nearest_face) = (0..3)
            .map(|i| (i, s[i]))
            .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
        if nearest_face <= 0.0 {
            // Inside: the way out through the nearest face.
            let g = self.gradient_of(axis);
            let out = if t[axis] < 0.0 { -g } else { g };
            return out.try_normalize(1e-12);
        }
        (p - self.nearest_on_surface(p)).try_normalize(1e-12)
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// A cylinder about +Z, by radius and full height.
///
/// The axis-aligned core of [`Cylinder`], and the same shape as [`Cuboid`] in a different
/// cross-section, computed the same way: a two-dimensional problem in (distance from the axis,
/// distance along it), because every query reduces to that pair. Exact inside as well as
/// outside, for the reason the cube's distance spells out — the outside-only shortcut fillets
/// wrongly near the rim.
#[derive(Copy, Clone, Debug)]
struct AxisCylinder {
    radius: f64,
    height: f64,
}

impl AxisCylinder {
    /// Distance past the side wall and past the end caps, negative inside each.
    fn wall_distances(&self, p: Point3<f64>) -> (f64, f64) {
        (
            radius_about_z(p.coords) - self.radius,
            p.z.abs() - self.height * 0.5,
        )
    }

    fn distance(&self, p: Point3<f64>) -> f64 {
        let (radial, axial) = self.wall_distances(p);
        // Outside, the length of the positive part, so the rim measures diagonally; inside,
        // the least negative wall, which is the distance to the nearest one.
        let (ox, oy) = (radial.max(0.0), axial.max(0.0));
        (ox * ox + oy * oy).sqrt() + radial.max(axial).min(0.0)
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let q = p.coords;
        let (radial, axial) = self.wall_distances(p);
        let axis = Vector3::new(0.0, 0.0, if p.z < 0.0 { -1.0 } else { 1.0 });
        if radial > 0.0 || axial > 0.0 {
            // Outside: the gradient of the length of the positive part, lifted out of the
            // (radial, axial) plane. A zero radial part needs no radial direction, which is
            // what makes a point straight off an end cap answerable while on the axis.
            let (ox, oy) = (radial.max(0.0), axial.max(0.0));
            let len = (ox * ox + oy * oy).sqrt();
            if len < 1e-12 {
                return None;
            }
            let radial_part = if ox > 0.0 {
                radial_direction(q)? * (ox / len)
            } else {
                Vector3::zeros()
            };
            return Some(radial_part + axis * (oy / len));
        }
        // Inside: whichever wall is nearest.
        if radial >= axial {
            radial_direction(q)
        } else {
            Some(axis)
        }
    }
}

/// A truncated cone about +Z, the axis-aligned core of [`Cone`].
#[derive(Copy, Clone, Debug)]
struct AxisCone {
    /// Radius at `z = -height / 2`.
    bottom_radius: f64,
    /// Radius at `z = +height / 2`. Zero gives a plain cone, apex up.
    top_radius: f64,
    height: f64,
}

impl AxisCone {
    /// The offsets to the nearest point on the caps and on the slanted side, in the
    /// (distance from the axis, distance along it) plane, and the sign of the result.
    ///
    /// Returned together because the distance is the shorter of the two and the normal is the
    /// direction of whichever one that was — computing them apart would decide the same
    /// comparison twice, and the two decisions could disagree.
    fn features(&self, p: Point3<f64>) -> ([f64; 2], [f64; 2], f64) {
        let half = self.height * 0.5;
        let (rho, z) = (radius_about_z(p.coords), p.z);

        // Offset from whichever cap the point is nearer, measured in the folded frame where
        // the two caps are one.
        let cap_radius = if z < 0.0 {
            self.bottom_radius
        } else {
            self.top_radius
        };
        let ca = [rho - rho.min(cap_radius), z.abs() - half];

        // Offset from the slanted side: a segment from (bottom_radius, -half) to
        // (top_radius, +half), projected onto, clamped to its ends, and subtracted.
        let k1 = [self.top_radius, half];
        let k2 = [self.top_radius - self.bottom_radius, 2.0 * half];
        let along = k2[0] * k2[0] + k2[1] * k2[1];
        let t = if along > 0.0 {
            (((k1[0] - rho) * k2[0] + (k1[1] - z) * k2[1]) / along).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let cb = [rho - k1[0] + k2[0] * t, z - k1[1] + k2[1] * t];

        // Inside is the axis side of the slant, and between the caps.
        let sign = if cb[0] < 0.0 && ca[1] < 0.0 {
            -1.0
        } else {
            1.0
        };
        (ca, cb, sign)
    }

    fn distance(&self, p: Point3<f64>) -> f64 {
        let (ca, cb, sign) = self.features(p);
        let (da, db) = (ca[0] * ca[0] + ca[1] * ca[1], cb[0] * cb[0] + cb[1] * cb[1]);
        sign * da.min(db).sqrt()
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let (ca, cb, sign) = self.features(p);
        let (da, db) = (ca[0] * ca[0] + ca[1] * ca[1], cb[0] * cb[0] + cb[1] * cb[1]);
        // The nearest feature is a fixed point, so the gradient is the unit offset from it,
        // and the sign carries the inside case.
        let cap_is_nearer = da < db;
        let (offset, len) = if cap_is_nearer {
            (ca, da.sqrt())
        } else {
            (cb, db.sqrt())
        };
        if len < 1e-12 {
            return None;
        }
        // The cap offset was measured with the two caps folded together, so its axial part
        // has to be unfolded back onto the side the point is actually on.
        let axial = if cap_is_nearer && p.z < 0.0 {
            -offset[1]
        } else {
            offset[1]
        };
        let radial = if offset[0].abs() > 1e-12 {
            radial_direction(p.coords)? * offset[0]
        } else {
            Vector3::zeros()
        };
        Some((radial + Vector3::new(0.0, 0.0, axial)) * (sign / len))
    }
}

/// A capsule about +Z with a radius at each end — the axis-aligned core of [`Capsule`].
///
/// Two radii rather than one, so the ends can differ: the surface is the two end spheres plus
/// the cone tangent to both, which is why this is exact rather than a blend of them. Equal
/// radii give an ordinary capsule, and the tangent term degenerates to a plain tube.
#[derive(Copy, Clone, Debug)]
struct AxisCapsule {
    /// Radius at `z = -height / 2`.
    start_radius: f64,
    /// Radius at `z = +height / 2`.
    end_radius: f64,
    /// The distance between the two cap centres.
    height: f64,
}

/// Which part of a two-radius capsule's surface is nearest, in the (radial, axial) plane.
enum CapsulePart {
    /// The sphere at the low end, with the offset from its centre.
    StartCap([f64; 2]),
    /// The sphere at the high end, with the offset from its centre.
    EndCap([f64; 2]),
    /// The cone tangent to both, with its outward direction — already unit length.
    Tangent([f64; 2]),
}

impl AxisCapsule {
    /// Whether one end sphere swallows the other, in which case there is no tangent cone and
    /// the shape is simply the larger sphere.
    fn one_end_swallows_the_other(&self) -> bool {
        (self.start_radius - self.end_radius).abs() >= self.height
    }

    /// The nearest part of the surface, measured from the low cap's centre.
    ///
    /// `q` is (distance from the axis, distance along it from the low cap), which is the frame
    /// the tangent construction below is written in.
    fn nearest_part(&self, p: Point3<f64>) -> CapsulePart {
        let h = self.height;
        let q = [radius_about_z(p.coords), p.z + h * 0.5];
        // The tangent line's slope, and the cosine that goes with it. `b` is how much the
        // radius shrinks per unit along the axis, so `a` is real exactly when neither sphere
        // contains the other — the caller checks that first.
        let b = (self.start_radius - self.end_radius) / h;
        let a = (1.0 - b * b).max(0.0).sqrt();
        // Where along the tangent this point falls: before the first cap, past the second, or
        // against the slant between them.
        let k = -b * q[0] + a * q[1];
        if k < 0.0 {
            CapsulePart::StartCap(q)
        } else if k > a * h {
            CapsulePart::EndCap([q[0], q[1] - h])
        } else {
            CapsulePart::Tangent([a, b])
        }
    }

    fn distance(&self, p: Point3<f64>) -> f64 {
        if self.height < 1e-12 || self.one_end_swallows_the_other() {
            // No tangent cone to speak of: the shape is the union of the two end spheres, and
            // that union *is* the larger one whenever it contains the smaller.
            let h = self.height * 0.5;
            let low = (p - Point3::new(0.0, 0.0, -h)).norm() - self.start_radius;
            let high = (p - Point3::new(0.0, 0.0, h)).norm() - self.end_radius;
            return low.min(high);
        }
        let q = [radius_about_z(p.coords), p.z + self.height * 0.5];
        match self.nearest_part(p) {
            CapsulePart::StartCap(o) => (o[0] * o[0] + o[1] * o[1]).sqrt() - self.start_radius,
            CapsulePart::EndCap(o) => (o[0] * o[0] + o[1] * o[1]).sqrt() - self.end_radius,
            // The distance to the tangent line, which is a plane in the lifted frame.
            CapsulePart::Tangent(d) => d[0] * q[0] + d[1] * q[1] - self.start_radius,
        }
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let lift = |radial: f64, axial: f64| -> Option<Vector3<f64>> {
            let across = if radial.abs() > 1e-12 {
                radial_direction(p.coords)? * radial
            } else {
                Vector3::zeros()
            };
            Some(across + Vector3::new(0.0, 0.0, axial))
        };
        if self.height < 1e-12 || self.one_end_swallows_the_other() {
            let h = self.height * 0.5;
            let low = (p - Point3::new(0.0, 0.0, -h)).norm() - self.start_radius;
            let high = (p - Point3::new(0.0, 0.0, h)).norm() - self.end_radius;
            let centre = Point3::new(0.0, 0.0, if low <= high { -h } else { h });
            let offset = p - centre;
            return (offset.norm() > 1e-12).then(|| offset.normalize());
        }
        match self.nearest_part(p) {
            CapsulePart::StartCap(o) | CapsulePart::EndCap(o) => {
                let len = (o[0] * o[0] + o[1] * o[1]).sqrt();
                if len < 1e-12 {
                    return None;
                }
                lift(o[0] / len, o[1] / len)
            }
            // Already unit length by construction: `a² + b² = 1`.
            CapsulePart::Tangent(d) => lift(d[0], d[1]),
        }
    }
}

/// A torus lying in the XY plane, about +Z.
///
/// The cheapest exact distance in this module: fold the query onto the ring circle, and what
/// is left is a sphere problem in the plane through the axis.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Torus {
    pub centre: Point3<f64>,
    /// From the centre to the middle of the tube.
    pub major_radius: f64,
    /// The tube's own radius.
    pub minor_radius: f64,
}

impl ScalarField for Torus {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let q = p - self.centre;
        let ring = radius_about_z(q) - self.major_radius;
        (ring * ring + q.z * q.z).sqrt() - self.minor_radius
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let q = p - self.centre;
        let ring = radius_about_z(q) - self.major_radius;
        let len = (ring * ring + q.z * q.z).sqrt();
        if len < 1e-12 {
            // On the ring circle itself, where every direction around the tube is equally
            // near — the torus's version of a sphere's centre.
            return None;
        }
        let radial = radial_direction(q)? * (ring / len);
        Some(radial + Vector3::new(0.0, 0.0, q.z / len))
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// A cylinder between two points.
///
/// Defined by where its ends actually are rather than by a centre and a height, because that
/// is how one gets placed: you know where it starts and where it stops. The direction falls
/// out of the two points, so there is no separate axis to keep in step with them.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Cylinder {
    pub base: Point3<f64>,
    pub top: Point3<f64>,
    pub radius: f64,
}

impl Cylinder {
    fn core(&self) -> Option<(Segment, AxisCylinder)> {
        let segment = Segment::new(self.base, self.top)?;
        Some((
            segment,
            AxisCylinder {
                radius: self.radius,
                height: segment.length,
            },
        ))
    }
}

impl ScalarField for Cylinder {
    fn distance(&self, p: Point3<f64>) -> f64 {
        match self.core() {
            Some((segment, core)) => core.distance(segment.to_local(p)),
            // Ends in the same place: no height, so what is left is the flat disc's rim —
            // and the honest distance to that is the distance to its centre less the radius,
            // which is a sphere. Degenerate, but never NaN.
            None => (p - self.base).norm() - self.radius,
        }
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let (segment, core) = self.core()?;
        Some(segment.to_world(core.normal(segment.to_local(p))?))
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// A cone between two points, with a radius at each: flat ends, and either may be zero.
///
/// A radius of zero at one end gives the plain cone the word usually means; two equal radii
/// give a cylinder. One node covers all three because they are one shape with different
/// numbers in it, and a separate node per case would be three descriptions of one thing.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Cone {
    pub base: Point3<f64>,
    pub top: Point3<f64>,
    pub base_radius: f64,
    pub top_radius: f64,
}

impl Cone {
    fn core(&self) -> Option<(Segment, AxisCone)> {
        let segment = Segment::new(self.base, self.top)?;
        Some((
            segment,
            AxisCone {
                bottom_radius: self.base_radius,
                top_radius: self.top_radius,
                height: segment.length,
            },
        ))
    }
}

impl ScalarField for Cone {
    fn distance(&self, p: Point3<f64>) -> f64 {
        match self.core() {
            Some((segment, core)) => core.distance(segment.to_local(p)),
            None => (p - self.base).norm() - self.base_radius.max(self.top_radius),
        }
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let (segment, core) = self.core()?;
        Some(segment.to_world(core.normal(segment.to_local(p))?))
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// A capsule between two points, with a radius at each: a sphere at either end and the
/// surface tangent to both.
///
/// The radii are independent, so this covers the plain rod, the tapered one, and — with the
/// two points together — a sphere. The ends are *spheres*, not discs, which is the whole
/// difference from [`Cone`]: the endpoints name the cap centres, so the shape reaches a radius
/// beyond each of them and growing a radius fattens it without shifting where it starts.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Capsule {
    pub start: Point3<f64>,
    pub end: Point3<f64>,
    pub start_radius: f64,
    pub end_radius: f64,
}

impl Capsule {
    fn core(&self) -> Option<(Segment, AxisCapsule)> {
        let segment = Segment::new(self.start, self.end)?;
        Some((
            segment,
            AxisCapsule {
                start_radius: self.start_radius,
                end_radius: self.end_radius,
                height: segment.length,
            },
        ))
    }
}

impl ScalarField for Capsule {
    fn distance(&self, p: Point3<f64>) -> f64 {
        match self.core() {
            Some((segment, core)) => core.distance(segment.to_local(p)),
            // Both cap centres in one place: exactly a sphere of the larger radius.
            None => (p - self.start).norm() - self.start_radius.max(self.end_radius),
        }
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        match self.core() {
            Some((segment, core)) => Some(segment.to_world(core.normal(segment.to_local(p))?)),
            None => {
                let offset = p - self.start;
                (offset.norm() > 1e-12).then(|| offset.normalize())
            }
        }
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// A half-space: everything below the plane through `origin`, facing +Z.
///
/// The one primitive here with no bounds at all, which is why it waited on
/// [`crate::Extent`] — a body's box could not previously say "there isn't one", only
/// "there's nothing here", and a union silently read the second as the first.
///
/// It is worth having despite that, because it is almost never a body. A half-space is the
/// standard trimming primitive, and intersecting a finite part with an infinite plane gives a
/// finite answer; `Extent::intersection` is what makes that fall out with no special case.
///
/// Facing +Z with no normal of its own, turned by a placement, for the same reason [`Torus`]
/// is: unlike the solids drawn between two points, a plane has no second point that would give
/// its direction as geometry, so the choice is between a stored normal and a placement — and a
/// stored normal beside a placement is the orientation-described-twice problem exactly.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Plane {
    /// A point the plane passes through. Only its Z matters before a placement turns it.
    pub origin: Point3<f64>,
    /// Which side is solid. `false` fills below the plane; `true` fills above it.
    pub flipped: bool,
}

impl Plane {
    fn facing(&self) -> f64 {
        if self.flipped { -1.0 } else { 1.0 }
    }
}

impl ScalarField for Plane {
    /// Signed height above the plane: exact, and unit-gradient by inspection.
    fn distance(&self, p: Point3<f64>) -> f64 {
        (p.z - self.origin.z) * self.facing()
    }

    fn normal(&self, _p: Point3<f64>) -> Option<Vector3<f64>> {
        Some(Vector3::z() * self.facing())
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// A single point, as a field: the distance to it.
///
/// Encloses no volume — the value is zero only at the point itself and positive everywhere
/// else — so as a *body* it draws nothing, and that is not a defect. Its use is as a field to
/// measure from: the distance-to-a-point that a falloff or a graded parameter is built on, and
/// the seed a scatter or a lattice is placed against.
///
/// Kept as its own kind rather than a zero-radius [`Sphere`] because the two mean different
/// things. A sphere of radius zero is a solid that happens to be empty; this is a source that
/// was never meant to be solid, and reading it as geometry is the mistake.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Point {
    pub at: Point3<f64>,
}

impl ScalarField for Point {
    fn distance(&self, p: Point3<f64>) -> f64 {
        (p - self.at).norm()
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let offset = p - self.at;
        // At the point itself every direction is equally near, the same as a sphere's centre.
        (offset.norm() > 1e-12).then(|| offset.normalize())
    }

    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }
}

/// How far a box's or a cylinder's edges are cut back, and in what shape: the reach of the
/// [`fieldform_scene::Fillet`] typed and its exponent -- 2 a circle, 3 G2, 1 a chamfer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Edge {
    pub reach: f64,
    pub exponent: f64,
}

/// A solid's reading from its per-axis distances `d` (negative inside each slab), with every edge
/// cut back `edge.reach`: the rounded form of `‖d₊‖ + min(max d, 0)` (`round-chamfer-spec.md` §3).
///
/// **Rounded, at any exponent above one, is the same formula on the inner solid with the n-norm:**
/// `q = d + k`, `‖q₊‖ₙ + min(max q, 0) − k`. At n = 2 that is the rounded box exactly -- the
/// inner box grown by a ball of the reach -- so it is a true distance. At n = 3 the zero set
/// across each edge is `((k + x)₊³ + (k + y)₊³)^(1/3) = k`, which is the G2 blend's own curve
/// (`Rim::join`'s, the blend's), so a box's G2 edge and a fillet's G2 join are one shape; the
/// n-norm's gradient has length at most one there, so it reads short and never long.
///
/// **A chamfer is the solid intersected with a bevel across each edge**: the plane cutting each
/// face back `k`, `(dᵢ + dⱼ + k)/√2`, which is the plane's true distance because the faces are
/// square to each other. Inside, the larger of the face and bevel readings is exactly the depth
/// of the convex solid they bound; outside, the larger of two bounds is one. The n-norm form at
/// n = 1 would also cut every corner back -- a triangle the spec does not ask for -- and reads
/// long across an edge, which is why it is not used.
pub(crate) fn edged(d: &[f64], edge: Edge) -> f64 {
    let exact = |d: &[f64]| {
        let outside = d.iter().map(|x| x.max(0.0).powi(2)).sum::<f64>().sqrt();
        outside + d.iter().copied().fold(f64::NEG_INFINITY, f64::max).min(0.0)
    };
    let k = edge.reach;
    if !(k > 0.0) {
        return exact(d);
    }
    if edge.exponent == 1.0 {
        let mut best = exact(d);
        for i in 0..d.len() {
            for j in i + 1..d.len() {
                best = best.max((d[i] + d[j] + k) * std::f64::consts::FRAC_1_SQRT_2);
            }
        }
        return best;
    }
    let n = edge.exponent;
    let q: Vec<f64> = d.iter().map(|x| x + k).collect();
    let norm = q.iter().map(|x| x.max(0.0).powf(n)).sum::<f64>().powf(1.0 / n);
    norm + q.iter().copied().fold(f64::NEG_INFINITY, f64::max).min(0.0) - k
}

/// A box with every edge cut back alike -- [`Cuboid`]'s centre and full sides, and an [`Edge`].
#[derive(Copy, Clone, Debug)]
pub struct EdgedCuboid {
    pub cuboid: Cuboid,
    pub edge: Edge,
}

impl ScalarField for EdgedCuboid {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let d = self.cuboid.axis_distances(p);
        edged(&[d.x, d.y, d.z], self.edge)
    }

    /// A true distance only for a round of the circle's exponent.
    fn normalization(&self) -> Normalization {
        if self.edge.exponent == 2.0 { Normalization::Distance } else { Normalization::LowerBound }
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

/// A cylinder with both circular edges cut back alike: [`Cylinder`]'s two points and radius, and
/// an [`Edge`], read in the meridian half-plane `(ρ − R, |h| − H)` -- which is exact for a solid
/// of revolution, since a nearest point always lies in the point's own meridian.
#[derive(Copy, Clone, Debug)]
pub struct EdgedCylinder {
    pub cylinder: Cylinder,
    pub edge: Edge,
}

impl ScalarField for EdgedCylinder {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let Some((segment, core)) = self.cylinder.core() else {
            return self.cylinder.distance(p);
        };
        let q = segment.to_local(p);
        let d = [q.x.hypot(q.y) - core.radius, q.z.abs() - core.height * 0.5];
        edged(&d, self.edge)
    }

    fn normalization(&self) -> Normalization {
        if self.edge.exponent == 2.0 { Normalization::Distance } else { Normalization::LowerBound }
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod edged_tests {
    use super::*;

    fn probes(seed: u64, n: usize, reach: f64) -> Vec<Point3<f64>> {
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        (0..n).map(|_| Point3::new(next() * reach, next() * reach, next() * reach)).collect()
    }

    fn boxed(edge: Edge) -> EdgedCuboid {
        EdgedCuboid { cuboid: Cuboid { centre: Point3::new(1.0, -2.0, 3.0), size: Vector3::new(40.0, 24.0, 16.0) }, edge }
    }

    /// **A rounded box is the inner box grown by a ball of the reach** -- the spec's acceptance,
    /// against an answer computed another way: the inner box's nearest point by clamping, less
    /// the reach outside, and the depth to its nearest face plus the reach inside.
    #[test]
    fn a_rounded_box_is_its_inner_box_grown_by_a_ball() {
        let k = 3.0;
        let solid = boxed(Edge { reach: k, exponent: 2.0 });
        let half = solid.cuboid.size * 0.5 - Vector3::repeat(k);
        for p in probes(7, 4000, 30.0) {
            let local = p - solid.cuboid.centre;
            let clamped = Vector3::new(
                local.x.clamp(-half.x, half.x),
                local.y.clamp(-half.y, half.y),
                local.z.clamp(-half.z, half.z),
            );
            let inside_inner = clamped == local;
            let expected = if inside_inner {
                -(half - local.abs()).min() - k
            } else {
                (local - clamped).norm() - k
            };
            assert!((solid.distance(p) - expected).abs() < 1e-9, "at {p:?}");
        }
        assert_eq!(solid.normalization(), Normalization::Distance);
    }

    /// **And a rounded cylinder is the revolved rounded rectangle**: the same check in the
    /// meridian half-plane, and the same box's formula across the cylinder's two edges.
    #[test]
    fn a_rounded_cylinder_is_the_revolved_rounded_rectangle() {
        let k = 2.0;
        let cylinder = Cylinder { base: Point3::new(0.0, 0.0, -10.0), top: Point3::new(0.0, 0.0, 10.0), radius: 8.0 };
        let solid = EdgedCylinder { cylinder, edge: Edge { reach: k, exponent: 2.0 } };
        let (r, h) = (8.0 - k, 10.0 - k);
        for p in probes(11, 4000, 16.0) {
            let (rho, z) = (p.x.hypot(p.y), p.z);
            let clamped = (rho.min(r), z.clamp(-h, h));
            let expected = if clamped == (rho, z) {
                -(r - rho).min(h - z.abs()) - k
            } else {
                (rho - clamped.0).hypot(z - clamped.1) - k
            };
            assert!((solid.distance(p) - expected).abs() < 1e-9, "at {p:?}");
        }
    }

    /// **Every new field source gets the unit-gradient property test** (the project notes, D8): the two
    /// rounded solids are true distances, away from the inner solid's medial creases.
    #[test]
    fn a_rounded_box_and_cylinder_have_unit_gradient() {
        let solid = boxed(Edge { reach: 3.0, exponent: 2.0 });
        let away: Vec<Point3<f64>> = probes(3, 3000, 40.0)
            .into_iter()
            .filter(|p| solid.distance(*p) > 0.5)
            .collect();
        assert!(away.len() > 1000);
        assert!(crate::gradient_deviation(&solid, &away, 1e-5) < 1e-6, "the box");
        let cylinder = EdgedCylinder {
            cylinder: Cylinder { base: Point3::new(0.0, 0.0, -10.0), top: Point3::new(3.0, 4.0, 10.0), radius: 8.0 },
            edge: Edge { reach: 2.0, exponent: 2.0 },
        };
        let away: Vec<Point3<f64>> = probes(5, 3000, 30.0)
            .into_iter()
            .filter(|p| cylinder.distance(*p) > 0.5)
            .collect();
        assert!(crate::gradient_deviation(&cylinder, &away, 1e-5) < 1e-6, "the cylinder");
    }

    /// **Each face is left where the reach says, at every shape**: on a face, a point the reach
    /// back from the edge is on the surface, and a point closer to the edge is cut away -- the
    /// one number meaning what a blend's and a rim's mean.
    #[test]
    fn every_shape_leaves_each_face_the_reach_back_from_the_edge() {
        let k = 3.0;
        for n in [2.0, 3.0, 1.0] {
            let solid = boxed(Edge { reach: k, exponent: n });
            let c = solid.cuboid.centre;
            // On the top face z = c.z + 8, along x toward the edge at x = c.x + 20.
            let on = |x: f64| solid.distance(Point3::new(c.x + x, c.y, c.z + 8.0));
            assert!(on(20.0 - k - 0.5).abs() < 1e-9, "n = {n}: flat before the reach");
            assert!(on(20.0 - k * 0.5) > 1e-6, "n = {n}: cut away inside the reach");
            // The chamfer is straight between its legs.
            if n == 1.0 {
                let mid = Point3::new(c.x + 20.0 - k * 0.5, c.y, c.z + 8.0 - k * 0.5);
                assert!(solid.distance(mid).abs() < 1e-9, "the bevel's midpoint is on it");
            }
        }
    }

    /// **And never overstates**: G2 and the chamfer read short, never long. A walk from every
    /// probe as far as it claims, in six directions, crosses no surface.
    #[test]
    fn a_g2_or_chamfered_box_or_cylinder_never_overstates() {
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        for n in [3.0, 1.0] {
            let fields: [Box<dyn ScalarField>; 2] = [
                Box::new(boxed(Edge { reach: 5.0, exponent: n })),
                Box::new(EdgedCylinder {
                    cylinder: Cylinder { base: Point3::new(0.0, 0.0, -10.0), top: Point3::new(3.0, 4.0, 10.0), radius: 8.0 },
                    edge: Edge { reach: 4.0, exponent: n },
                }),
            ];
            for field in &fields {
                assert_eq!(field.normalization(), Normalization::LowerBound);
                for p in probes(13, 1500, 35.0) {
                    let claimed = field.distance(p);
                    if claimed <= 0.0 {
                        continue;
                    }
                    for d in dirs {
                        for step in 1..=10 {
                            let at = p + d * (claimed * step as f64 / 10.0);
                            assert!(field.distance(at) > -1e-9, "n = {n}: from {p:?} along {d:?} within {claimed}");
                        }
                    }
                }
            }
        }
    }
}

/// A convex polygon in the meridian half-plane `(ρ, z)` of a solid of revolution, symmetric about
/// the axis, kept as the chain of its right-hand vertices from the axis at the bottom round to the
/// axis at the top. Its distance at `(ρ, z)`, `ρ ≥ 0`, *is* the solid's distance in space: a solid
/// of revolution's nearest point lies in the query's own meridian, and the mirror half is never
/// nearer to a point on this side.
///
/// Only the chain's segments and their half-planes are read. The axis is not an edge, being
/// interior to the full polygon, and the mirrored edges' half-planes are implied for `ρ ≥ 0`,
/// since every right-hand edge's outward normal points away from the axis. Inside a convex
/// polygon the depth is the least distance to an edge's line, the foot lying on the edge.
#[derive(Clone, Debug)]
struct Meridian {
    chain: Vec<Point2<f64>>,
}

impl Meridian {
    fn distance(&self, rho: f64, z: f64) -> f64 {
        let q = Point2::new(rho, z);
        let mut nearest = f64::INFINITY;
        let mut depth = f64::INFINITY;
        let mut inside = true;
        for pair in self.chain.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let d = b - a;
            let len = d.norm();
            if len < 1e-12 {
                nearest = nearest.min((q - a).norm());
                continue;
            }
            let t = ((q - a).dot(&d) / (len * len)).clamp(0.0, 1.0);
            nearest = nearest.min((q - (a + d * t)).norm());
            // Outward: the polygon lies to the left of a chain that runs anticlockwise.
            let s = (q - a).dot(&Vector2::new(d.y, -d.x)) / len;
            if s > 0.0 {
                inside = false;
            }
            depth = depth.min(-s);
        }
        if inside { -depth } else { nearest }
    }
}

/// A truncated cone's meridian section -- the isosceles trapezoid between its caps and its slanted
/// side, a triangle where a radius is nought -- and the cuts a reach makes of it.
#[derive(Copy, Clone, Debug)]
struct Trapezoid {
    half: f64,
    r_bottom: f64,
    r_top: f64,
}

impl Trapezoid {
    fn slant_length(&self) -> f64 {
        (self.r_bottom - self.r_top).hypot(2.0 * self.half)
    }

    /// The slant's outward unit normal and its offset: `n · q = c` on the slant.
    fn slant(&self) -> (Vector2<f64>, f64) {
        let n = Vector2::new(2.0 * self.half, self.r_bottom - self.r_top) / self.slant_length();
        (n, n.dot(&Vector2::new(self.r_bottom, -self.half)))
    }

    fn hard(&self) -> Meridian {
        Meridian {
            chain: vec![
                Point2::new(0.0, -self.half),
                Point2::new(self.r_bottom, -self.half),
                Point2::new(self.r_top, self.half),
                Point2::new(0.0, self.half),
            ],
        }
    }

    /// How far in every edge can move before the section closes: its inradius, the largest ball
    /// that fits -- `min(half, n_ρ · R / (1 + |n_z|))` with `R` the wider radius, which for a
    /// pointed cone is `R·H / (R + L)`, the triangle's.
    fn inradius(&self) -> f64 {
        let (n, _) = self.slant();
        self.half.min(n.x * self.r_bottom.max(self.r_top) / (1.0 + n.y.abs()))
    }

    /// The section with every edge moved in by `k`, or `None` when nothing is left. A trapezoid,
    /// or a triangle where the moved slants meet on the axis before the moved cap.
    fn inset(&self, k: f64) -> Option<Meridian> {
        let (n, c) = self.slant();
        let (zb, zt) = (-self.half + k, self.half - k);
        if zb > zt + 1e-12 {
            return None;
        }
        let rho_at = |z: f64| (c - k - n.y * z) / n.x;
        let (ra, rb) = (rho_at(zb), rho_at(zt));
        if ra < 0.0 && rb < 0.0 {
            return None;
        }
        // Where the moved slant crosses the axis, read only when one end has gone past it,
        // which needs the slant to lean.
        let apex = || (c - k) / n.y;
        let chain = if ra >= 0.0 && rb >= 0.0 {
            vec![Point2::new(0.0, zb), Point2::new(ra, zb), Point2::new(rb, zt), Point2::new(0.0, zt)]
        } else if rb < 0.0 {
            vec![Point2::new(0.0, zb), Point2::new(ra, zb), Point2::new(0.0, apex())]
        } else {
            vec![Point2::new(0.0, apex()), Point2::new(rb, zt), Point2::new(0.0, zt)]
        };
        Some(Meridian { chain })
    }

    /// How far a chamfer's legs can reach before two meet: the shorter of each radius that is
    /// not nought and half the slant.
    fn chamfer_room(&self) -> f64 {
        let mut room = self.slant_length() * 0.5;
        for r in [self.r_bottom, self.r_top] {
            if r > 0.0 {
                room = room.min(r);
            }
        }
        room
    }

    /// The section with each corner cut back `k` along both of its edges: a bevel across each
    /// circular edge, and a flat across an apex.
    fn clipped(&self, k: f64) -> Meridian {
        let (half, rb, rt) = (self.half, self.r_bottom, self.r_top);
        let a = Point2::new(rb, -half);
        let b = Point2::new(rt, half);
        let t = (b - a) / self.slant_length();
        let mut chain = Vec::with_capacity(6);
        if rb > 0.0 {
            chain.push(Point2::new(0.0, -half));
            chain.push(Point2::new(rb - k, -half));
        } else {
            chain.push(Point2::new(0.0, (a + t * k).y));
        }
        chain.push(a + t * k);
        chain.push(b - t * k);
        if rt > 0.0 {
            chain.push(Point2::new(rt - k, half));
            chain.push(Point2::new(0.0, half));
        } else {
            chain.push(Point2::new(0.0, (b - t * k).y));
        }
        Meridian { chain }
    }

    /// The pairs of edge lines meeting at each corner, as `(normal, offset)` with the line's
    /// signed distance `n · q − c` positive outside: the slant with a cap, or with its own
    /// mirror image at an apex.
    fn corners(&self) -> Vec<[(Vector2<f64>, f64); 2]> {
        let (n, c) = self.slant();
        let slant = (n, c);
        let mirror = (Vector2::new(-n.x, n.y), c);
        let bottom = (Vector2::new(0.0, -1.0), self.half);
        let top = (Vector2::new(0.0, 1.0), self.half);
        vec![
            if self.r_bottom > 0.0 { [bottom, slant] } else { [slant, mirror] },
            if self.r_top > 0.0 { [slant, top] } else { [slant, mirror] },
        ]
    }
}

/// How an [`EdgedCone`]'s section is cut, built once.
#[derive(Clone, Debug)]
enum ConeCut {
    Round { inset: Meridian, reach: f64 },
    Chamfer(Meridian),
    Family { hard: Meridian, corners: Vec<[(Vector2<f64>, f64); 2]>, reach: f64, exponent: f64 },
}

/// A cone with its circular edges cut back alike -- and its apex, where a radius is nought:
/// [`Cone`]'s two points and radii, and an [`Edge`] (`round-chamfer-spec.md` §3.2).
///
/// **Read in the meridian half-plane as a convex polygon**, which is what makes two of the three
/// shapes exact where a box's chamfer is a bound: a solid of revolution's nearest point lies in
/// the query's own meridian, so a distance exact in the plane is exact in space.
///
/// - **Round, at G1**: the section with every edge moved in by the reach, less the reach -- the
///   inset polygon grown by a ball. Exact.
/// - **Chamfer**: the section with each corner cut back the reach along both its edges; an apex
///   is cut flat across. Exact.
/// - **G2**: the hard section held against the blend family's join at each corner, at exponent
///   three -- `Rim::join`'s form, so the curve across an edge is the one a G2 fillet makes. A
///   bound, as a box's G2 edge is: the larger of two bounds.
pub struct EdgedCone {
    segment: Segment,
    cut: ConeCut,
}

impl EdgedCone {
    /// The section's room for a cut of this shape, and what it is named by: the inradius for a
    /// round or a G2, the shortest edge's half for a chamfer. Nought for a cone with no length.
    pub(crate) fn room(cone: &Cone, exponent: f64) -> (f64, &'static str) {
        let Some((_, core)) = cone.core() else { return (0.0, "cone's length") };
        let t = Trapezoid { half: core.height * 0.5, r_bottom: core.bottom_radius, r_top: core.top_radius };
        if exponent == 1.0 {
            (t.chamfer_room(), "cone's shortest edge")
        } else {
            (t.inradius(), "cone's section")
        }
    }

    /// `None` when the cone has no length, or the reach leaves nothing of the section -- which
    /// the compile refuses by name before it gets here.
    pub(crate) fn new(cone: Cone, edge: Edge) -> Option<Self> {
        let (segment, core) = cone.core()?;
        let t = Trapezoid { half: core.height * 0.5, r_bottom: core.bottom_radius, r_top: core.top_radius };
        let cut = if edge.exponent == 1.0 {
            ConeCut::Chamfer(t.clipped(edge.reach))
        } else if edge.exponent == 2.0 {
            ConeCut::Round { inset: t.inset(edge.reach)?, reach: edge.reach }
        } else {
            ConeCut::Family { hard: t.hard(), corners: t.corners(), reach: edge.reach, exponent: edge.exponent }
        };
        Some(Self { segment, cut })
    }
}

impl ScalarField for EdgedCone {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let q = self.segment.to_local(p);
        let (rho, z) = (q.x.hypot(q.y), q.z);
        match &self.cut {
            ConeCut::Round { inset, reach } => inset.distance(rho, z) - reach,
            ConeCut::Chamfer(clipped) => clipped.distance(rho, z),
            ConeCut::Family { hard, corners, reach, exponent } => {
                let at = Vector2::new(rho, z);
                let mut f = hard.distance(rho, z);
                for [(na, ca), (nb, cb)] in corners {
                    let (la, lb) = (na.dot(&at) - ca, nb.dot(&at) - cb);
                    f = f.max(-crate::graph_field::blend_bound(-la, -lb, *reach, *exponent));
                }
                f
            }
        }
    }

    fn normalization(&self) -> Normalization {
        match self.cut {
            ConeCut::Round { .. } | ConeCut::Chamfer(_) => Normalization::Distance,
            ConeCut::Family { .. } => Normalization::LowerBound,
        }
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod edged_cone_tests {
    use super::*;

    fn probes(seed: u64, n: usize, reach: f64) -> Vec<Point3<f64>> {
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        (0..n).map(|_| Point3::new(next() * reach, next() * reach, next() * reach)).collect()
    }

    fn frustum(rb: f64, rt: f64) -> Cone {
        Cone { base: Point3::new(0.0, 0.0, -10.0), top: Point3::new(0.0, 0.0, 10.0), base_radius: rb, top_radius: rt }
    }

    fn tilted(rb: f64, rt: f64) -> Cone {
        Cone { base: Point3::new(1.0, -2.0, -8.0), top: Point3::new(4.0, 3.0, 9.0), base_radius: rb, top_radius: rt }
    }

    /// **A rounded cone is its inset cone grown by a ball** -- held against the hard `Cone` of
    /// the inset's own radii and height, less the reach, which is a different code path: the
    /// frustum's inset is a frustum, and a pointed cone's inset a pointed cone whose apex sits
    /// where the moved slants meet.
    #[test]
    fn a_rounded_cone_is_its_inset_cone_grown_by_a_ball() {
        let k = 1.5;
        for cone in [frustum(8.0, 5.0), frustum(8.0, 0.0), frustum(3.0, 7.0), frustum(0.0, 6.0)] {
            let solid = EdgedCone::new(cone, Edge { reach: k, exponent: 2.0 }).expect("room");
            let ConeCut::Round { inset, .. } = &solid.cut else { panic!("a round") };
            // The inset chain read back as a cone between its two axis points.
            let (lo, hi) = (inset.chain[0], inset.chain[inset.chain.len() - 1]);
            let radius_at = |z: f64| inset.chain.iter().filter(|v| (v.y - z).abs() < 1e-9).map(|v| v.x).fold(0.0, f64::max);
            let inner = Cone {
                base: Point3::new(0.0, 0.0, lo.y),
                top: Point3::new(0.0, 0.0, hi.y),
                base_radius: radius_at(lo.y),
                top_radius: radius_at(hi.y),
            };
            for p in probes(3, 3000, 16.0) {
                let expected = inner.distance(p) - k;
                assert!((solid.distance(p) - expected).abs() < 1e-9, "{cone:?} at {p:?}: {} vs {expected}", solid.distance(p));
            }
            assert_eq!(solid.normalization(), Normalization::Distance);
        }
    }

    /// **A chamfered cone is the clipped section, exactly**: held against a reading of the
    /// clipped chain that shares nothing with the leaf's -- each edge's nearest point by its own
    /// projection, and the sign from the chain's turning rather than from any normal.
    #[test]
    fn a_chamfered_cone_is_its_clipped_section() {
        let k = 1.2;
        for cone in [frustum(8.0, 5.0), frustum(8.0, 0.0), frustum(0.0, 6.0)] {
            let solid = EdgedCone::new(cone, Edge { reach: k, exponent: 1.0 }).expect("room");
            let ConeCut::Chamfer(clipped) = &solid.cut else { panic!("a chamfer") };
            let chain = &clipped.chain;
            for p in probes(5, 2000, 14.0) {
                let q = Point2::new(p.x.hypot(p.y), p.z);
                let mut nearest = f64::INFINITY;
                let mut left_of_all = true;
                for pair in chain.windows(2) {
                    let (a, b) = (pair[0], pair[1]);
                    let along = ((q.x - a.x) * (b.x - a.x) + (q.y - a.y) * (b.y - a.y))
                        / ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).max(1e-18);
                    let at = a + (b - a) * along.clamp(0.0, 1.0);
                    nearest = nearest.min((q - at).norm());
                    let cross = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
                    if cross < 0.0 {
                        left_of_all = false;
                    }
                }
                let expected = if left_of_all { -nearest } else { nearest };
                assert!((solid.distance(p) - expected).abs() < 1e-9, "{cone:?} at {p:?}: {} vs {expected}", solid.distance(p));
            }
            assert_eq!(solid.normalization(), Normalization::Distance);
        }
    }

    /// **Every new field source gets the unit-gradient property test** (D8): the round and the
    /// chamfer are true distances, away from the medial creases, on a tilted cone.
    #[test]
    fn a_rounded_and_a_chamfered_cone_have_unit_gradient() {
        for n in [2.0, 1.0] {
            let solid = EdgedCone::new(tilted(7.0, 3.0), Edge { reach: 1.5, exponent: n }).expect("room");
            let away: Vec<Point3<f64>> = probes(7, 4000, 20.0).into_iter().filter(|p| solid.distance(*p) > 0.5).collect();
            assert!(away.len() > 1000);
            assert!(crate::gradient_deviation(&solid, &away, 1e-5) < 1e-6, "n = {n}");
        }
    }

    /// **Each face is left where the reach says, at every shape**: on the base, a point the reach
    /// in from the rim is on the surface and one nearer the rim is cut away.
    #[test]
    fn every_shape_leaves_each_face_the_reach_back_from_the_edge() {
        let k = 2.0;
        for n in [2.0, 3.0, 1.0] {
            let solid = EdgedCone::new(frustum(8.0, 5.0), Edge { reach: k, exponent: n }).expect("room");
            let on_base = |rho: f64| solid.distance(Point3::new(rho, 0.0, -10.0));
            assert!(on_base(8.0 - k - 0.5).abs() < 1e-9, "n = {n}: flat inside the reach");
            assert!(on_base(8.0 - k * 0.5) > 1e-6, "n = {n}: cut away within it");
            let on_top = |rho: f64| solid.distance(Point3::new(rho, 0.0, 10.0));
            assert!(on_top(5.0 - k - 0.5).abs() < 1e-9, "n = {n}: the top too");
            assert!(on_top(5.0 - k * 0.5) > 1e-6, "n = {n}");
        }
    }

    /// **A G2 cone never overstates**, pointed or truncated, upright or tilted.
    #[test]
    fn a_g2_cone_never_overstates() {
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        for cone in [frustum(8.0, 5.0), frustum(8.0, 0.0), tilted(7.0, 3.0), tilted(0.0, 6.0)] {
            let solid = EdgedCone::new(cone, Edge { reach: 2.0, exponent: 3.0 }).expect("room");
            assert_eq!(solid.normalization(), Normalization::LowerBound);
            for p in probes(11, 1500, 20.0) {
                let claimed = solid.distance(p);
                if claimed <= 0.0 {
                    continue;
                }
                for d in dirs {
                    for step in 1..=10 {
                        let at = p + d * (claimed * step as f64 / 10.0);
                        assert!(solid.distance(at) > -1e-9, "{cone:?}: from {p:?} along {d:?} within {claimed}");
                    }
                }
            }
        }
    }

    /// **The room is the inradius for a round and the shortest edge's half for a chamfer**: a
    /// pointed cone of base 8 and height 20 has an inradius of 8·20/(8 + √464) ≈ 5.40, and its
    /// chamfer may reach half its slant or its base, whichever is less.
    #[test]
    fn the_room_is_the_inradius_or_the_shortest_edge() {
        let pointed = frustum(8.0, 0.0);
        let (room, what) = EdgedCone::room(&pointed, 2.0);
        assert!((room - 8.0 * 20.0 / (8.0 + 464f64.sqrt())).abs() < 1e-9, "{room}");
        assert_eq!(what, "cone's section");
        let (room, what) = EdgedCone::room(&pointed, 1.0);
        assert!((room - 8.0).abs() < 1e-9, "{room}: the base is shorter than half the slant");
        assert_eq!(what, "cone's shortest edge");
        assert!(EdgedCone::new(pointed, Edge { reach: 5.5, exponent: 2.0 }).is_none(), "past the inradius nothing is left");
        assert_eq!(EdgedCone::room(&frustum(8.0, 8.0), 2.0).0, 8.0, "a cylinder's: the lesser of radius and half-length");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient_deviation;

    /// Points spread over and around a shape of the given size, avoiding the exact centre and
    /// the surface itself — one has no normal and the other is where a finite difference
    /// straddles a crease.
    fn probes(size: f64) -> Vec<Point3<f64>> {
        let mut points = Vec::new();
        for i in -6..=6 {
            for j in -6..=6 {
                for k in -6..=6 {
                    let p = Point3::new(
                        i as f64 * size * 0.37,
                        j as f64 * size * 0.41,
                        k as f64 * size * 0.43,
                    );
                    if p.coords.norm() > size * 0.05 {
                        points.push(p);
                    }
                }
            }
        }
        points
    }

    /// The property every field source in this codebase has to hold: the distance changes at
    /// one unit per unit of travel. A source that fails it makes every blend against it
    /// subtly wrong, with the symptom appearing nowhere near the cause.
    #[test]
    fn a_sphere_has_a_unit_gradient() {
        let sphere = Sphere {
            centre: Point3::new(3.0, -2.0, 1.0),
            radius: 7.0,
        };
        let worst = gradient_deviation(&sphere, &probes(7.0), 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
    }

    /// Same for the cube, away from its creases: the diagonal planes inside where the nearest
    /// face changes, and the edges and corners outside, are where the gradient genuinely turns
    /// and a finite difference across one reads less than unit length.
    #[test]
    fn a_cube_has_a_unit_gradient_away_from_its_creases() {
        let cube = Cuboid::cube(Point3::origin(), 10.0);
        let points: Vec<Point3<f64>> = probes(10.0)
            .into_iter()
            .filter(|p| {
                let d = cube.axis_distances(*p);
                // Off the inside diagonals and away from the outside edges: no two axes may be
                // within a finite difference of each other in the region that decides.
                let near = |a: f64, b: f64| (a - b).abs() < 0.01;
                !(near(d.x, d.y) || near(d.y, d.z) || near(d.x, d.z))
            })
            .collect();
        let worst = gradient_deviation(&cube, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
    }

    /// D8's property on a box whose three sides all differ, which is the case a cube cannot
    /// catch: every wrong half-extent is the *same* wrong half-extent on a cube, so a
    /// per-axis mistake stays invisible until the sides disagree.
    #[test]
    fn an_unequal_box_has_a_unit_gradient_away_from_its_creases() {
        let boxed = Cuboid {
            centre: Point3::new(1.0, -2.0, 0.5),
            size: Vector3::new(40.0, 10.0, 3.0),
        };
        let points: Vec<Point3<f64>> = probes(40.0)
            .into_iter()
            .filter(|p| {
                let d = boxed.axis_distances(*p);
                let near = |a: f64, b: f64| (a - b).abs() < 0.01;
                !(near(d.x, d.y) || near(d.y, d.z) || near(d.x, d.z))
            })
            .collect();
        let worst = gradient_deviation(&boxed, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
    }

    /// Each face sits at its own half-extent, and the shortest way out is the *thinnest*
    /// axis — not whichever one a cube would have used.
    ///
    /// A box 40 x 10 x 3 about a centre that is not the origin, so an implementation that used
    /// one side length for all three, or forgot the centre on one axis, has nowhere to hide.
    #[test]
    fn a_box_puts_each_face_at_its_own_half_extent() {
        let c = Point3::new(1.0, -2.0, 0.5);
        let boxed = Cuboid {
            centre: c,
            size: Vector3::new(40.0, 10.0, 3.0),
        };

        for (axis, half) in [
            (Vector3::x(), 20.0),
            (Vector3::y(), 5.0),
            (Vector3::z(), 1.5),
        ] {
            assert!(
                (boxed.distance(c + axis * half)).abs() < 1e-12,
                "face at {half}"
            );
            assert!(
                (boxed.distance(c + axis * (half + 4.0)) - 4.0).abs() < 1e-12,
                "4 mm out from the face at {half}",
            );
        }
        // Deepest inside is half the *shortest* side, since that face is the nearest one.
        assert!(
            (boxed.distance(c) + 1.5).abs() < 1e-12,
            "the centre is half the thinnest side"
        );
    }

    /// A box with three equal sides is a cube, which is the vocabulary claim this generalisation
    /// rests on: one shape, not two.
    #[test]
    fn a_box_with_three_equal_sides_is_a_cube() {
        let cube = Cuboid::cube(Point3::new(2.0, 3.0, -1.0), 10.0);
        let spelled_out = Cuboid {
            centre: Point3::new(2.0, 3.0, -1.0),
            size: Vector3::new(10.0, 10.0, 10.0),
        };
        for p in probes(10.0) {
            assert!((cube.distance(p) - spelled_out.distance(p)).abs() < 1e-12);
        }
    }

    #[test]
    fn a_sphere_is_its_radius_from_its_centre() {
        let sphere = Sphere {
            centre: Point3::new(1.0, 2.0, 3.0),
            radius: 5.0,
        };
        assert!(
            (sphere.distance(sphere.centre) + 5.0).abs() < 1e-12,
            "the centre is -r"
        );
        assert!(
            sphere.distance(Point3::new(6.0, 2.0, 3.0)).abs() < 1e-12,
            "and the surface is where the radius says"
        );
        assert!((sphere.distance(Point3::new(11.0, 2.0, 3.0)) - 5.0).abs() < 1e-12);
    }

    /// The exact box distance, checked where the shortcut would be wrong: diagonally off a
    /// corner, where the answer is the length of the offset rather than its largest axis.
    #[test]
    fn a_cube_measures_diagonally_off_its_corners() {
        let cube = Cuboid::cube(Point3::origin(), 10.0);
        assert!(
            (cube.distance(Point3::origin()) + 5.0).abs() < 1e-12,
            "half a side deep"
        );
        assert!(
            cube.distance(Point3::new(5.0, 0.0, 0.0)).abs() < 1e-12,
            "on a face"
        );

        // Three units out along each axis from the corner: 3*sqrt(3), not 3.
        let p = Point3::new(8.0, 8.0, 8.0);
        assert!(
            (cube.distance(p) - 3.0 * 3f64.sqrt()).abs() < 1e-12,
            "got {}",
            cube.distance(p)
        );
    }

    /// The analytic normal against a central difference, on the unequal box.
    ///
    /// A cube cannot catch a swapped axis in the normal — every axis looks like every other —
    /// so the shape whose sides differ is the one that can.
    #[test]
    fn an_unequal_boxs_normal_agrees_with_a_central_difference() {
        let boxed = Cuboid {
            centre: Point3::new(1.0, -2.0, 0.5),
            size: Vector3::new(40.0, 10.0, 3.0),
        };
        for p in probes(40.0) {
            let d = boxed.axis_distances(p);
            let near = |a: f64, b: f64| (a - b).abs() < 0.05;
            if near(d.x, d.y) || near(d.y, d.z) || near(d.x, d.z) {
                continue;
            }
            let Some(analytic) = boxed.normal(p) else {
                continue;
            };
            let numeric = finite_difference_normal(&boxed, p, 1e-5);
            assert!(
                (analytic - numeric).norm() < 1e-4,
                "at {p:?}: analytic {analytic:?} against {numeric:?}",
            );
        }
    }

    #[test]
    fn a_cubes_normal_points_out_of_the_nearest_face() {
        let cube = Cuboid::cube(Point3::origin(), 10.0);
        let n = cube.normal(Point3::new(4.0, 0.5, 0.5)).unwrap();
        assert!(
            (n - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12,
            "inside, nearest +X"
        );
        let n = cube.normal(Point3::new(0.0, -4.5, 0.0)).unwrap();
        assert!(
            (n - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-12,
            "inside, nearest -Y"
        );
        let n = cube.normal(Point3::new(9.0, 0.0, 0.0)).unwrap();
        assert!(
            (n - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12,
            "outside a face"
        );
    }

    /// The central difference of `distance`, which is what the analytic normal has to agree
    /// with. Written here rather than reused from `gradient_deviation` because that one
    /// measures the gradient's *length*, and a normal can have unit length and still point
    /// the wrong way — which is exactly the mistake a hand-derived gradient makes.
    fn finite_difference_normal(field: &dyn ScalarField, p: Point3<f64>, h: f64) -> Vector3<f64> {
        let axis =
            |d: Vector3<f64>| (field.distance(p + d * h) - field.distance(p - d * h)) / (2.0 * h);
        Vector3::new(axis(Vector3::x()), axis(Vector3::y()), axis(Vector3::z()))
    }

    /// Every analytic normal in this module, checked against the gradient of its own distance.
    ///
    /// The property that matters and the one an eyeball cannot check: a normal derived by hand
    /// can be unit length, vary smoothly, and still point somewhere wrong, and the symptom is
    /// shading that looks merely odd. `probes` are pre-filtered by the caller to stay off the
    /// shape's creases, where the gradient genuinely turns.
    fn assert_normals_match_the_gradient(
        field: &dyn ScalarField,
        probes: &[Point3<f64>],
        what: &str,
    ) {
        let mut worst = 0.0f64;
        let mut checked = 0;
        for p in probes {
            let Some(n) = field.normal(*p) else { continue };
            let g = finite_difference_normal(field, *p, 1e-5);
            if g.norm() < 0.5 {
                continue; // a crease the filter missed; the gradient test already catches those
            }
            worst = worst.max((n - g.normalize()).norm());
            checked += 1;
        }
        assert!(
            checked > 100,
            "{what}: only {checked} points had an analytic normal"
        );
        assert!(
            worst < 1e-5,
            "{what}: worst normal disagreement {worst} over {checked} points"
        );
    }

    /// Probes for a shape round about +Z, with the creases the caller names filtered out.
    ///
    /// A finite difference straddling a crease reads short no matter how small the step, so
    /// every such test has to say where its own creases are rather than pretend it has none.
    fn round_probes(size: f64, is_near_crease: impl Fn(Point3<f64>) -> bool) -> Vec<Point3<f64>> {
        probes(size)
            .into_iter()
            .filter(|p| !is_near_crease(*p))
            .collect()
    }

    /// The distance from the +Z axis, for the crease filters below.
    fn rho(p: Point3<f64>) -> f64 {
        (p.x * p.x + p.y * p.y).sqrt()
    }

    /// A cylinder standing on Z, so its creases can be described in the frame it is built in.
    fn upright_cylinder(radius: f64, height: f64) -> Cylinder {
        Cylinder {
            base: Point3::new(0.0, 0.0, -height * 0.5),
            top: Point3::new(0.0, 0.0, height * 0.5),
            radius,
        }
    }

    #[test]
    fn a_cylinder_has_a_unit_gradient_away_from_its_creases() {
        let cylinder = upright_cylinder(4.0, 10.0);
        let core = AxisCylinder {
            radius: 4.0,
            height: 10.0,
        };
        // Two creases: the rim, where the side wall meets a cap, and the surface inside where
        // the nearest wall changes from the side to a cap.
        let points = round_probes(10.0, |p| {
            let (radial, axial) = core.wall_distances(p);
            (radial - axial).abs() < 0.05 || rho(p) < 1e-9
        });
        let worst = gradient_deviation(&cylinder, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
        assert_normals_match_the_gradient(&cylinder, &points, "cylinder");
    }

    #[test]
    fn a_cone_has_a_unit_gradient_away_from_its_creases() {
        let cone = Cone {
            base: Point3::new(0.0, 0.0, -4.5),
            top: Point3::new(0.0, 0.0, 4.5),
            base_radius: 5.0,
            top_radius: 2.0,
        };
        let core = AxisCone {
            bottom_radius: 5.0,
            top_radius: 2.0,
            height: 9.0,
        };
        // The creases are wherever the cap feature and the slant feature are equally near.
        let points = round_probes(9.0, |p| {
            let (ca, cb, _) = core.features(p);
            let da = (ca[0] * ca[0] + ca[1] * ca[1]).sqrt();
            let db = (cb[0] * cb[0] + cb[1] * cb[1]).sqrt();
            (da - db).abs() < 0.05 || rho(p) < 0.05
        });
        let worst = gradient_deviation(&cone, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
        assert_normals_match_the_gradient(&cone, &points, "cone");
    }

    /// The apex case, which the two-radius form has to get right or a plain cone is unusable.
    #[test]
    fn a_cone_with_no_top_radius_comes_to_a_point() {
        let cone = Cone {
            base: Point3::new(0.0, 0.0, -4.0),
            top: Point3::new(0.0, 0.0, 4.0),
            base_radius: 4.0,
            top_radius: 0.0,
        };
        assert!(
            cone.distance(Point3::new(0.0, 0.0, 4.0)).abs() < 1e-9,
            "the apex is on the surface"
        );
        assert!(
            cone.distance(Point3::origin()) < 0.0,
            "the middle of the axis is inside"
        );
        assert!(
            cone.distance(Point3::new(0.0, 0.0, 6.0)) > 0.0,
            "and above the apex is outside"
        );
        // Halfway up, the radius has halved.
        assert!(
            cone.distance(Point3::new(1.9, 0.0, 0.0)) < 0.0,
            "inside at half height"
        );
        assert!(
            cone.distance(Point3::new(2.1, 0.0, 0.0)) > 0.0,
            "outside at half height"
        );
    }

    #[test]
    fn a_torus_has_a_unit_gradient_away_from_its_ring() {
        let torus = Torus {
            centre: Point3::origin(),
            major_radius: 6.0,
            minor_radius: 2.0,
        };
        // Two creases: the ring circle itself, and the axis, where folding onto the ring turns.
        let points = round_probes(8.0, |p| {
            let ring = rho(p) - 6.0;
            (ring * ring + p.z * p.z).sqrt() < 0.05 || rho(p) < 0.05
        });
        let worst = gradient_deviation(&torus, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
        assert_normals_match_the_gradient(&torus, &points, "torus");
    }

    #[test]
    fn a_capsule_has_a_unit_gradient_away_from_its_spine() {
        let capsule = Capsule {
            start: Point3::new(0.0, 0.0, -4.0),
            end: Point3::new(0.0, 0.0, 4.0),
            start_radius: 3.0,
            end_radius: 3.0,
        };
        let points = round_probes(8.0, |p| {
            let z = p.z.clamp(-4.0, 4.0);
            (p - Point3::new(0.0, 0.0, z)).norm() < 0.05
        });
        let worst = gradient_deviation(&capsule, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
        assert_normals_match_the_gradient(&capsule, &points, "capsule");
    }

    /// The tapered capsule, where the surface between the caps is the cone tangent to both.
    ///
    /// The case an equal-radius test cannot reach: with `r1 == r2` the tangent term collapses
    /// to a plain tube and a wrong slope would never show. Here the ends differ by 4 over a
    /// length of 8, so the tangent is genuinely slanted.
    #[test]
    fn a_tapered_capsule_has_a_unit_gradient_along_its_tangent() {
        let capsule = Capsule {
            start: Point3::new(0.0, 0.0, -4.0),
            end: Point3::new(0.0, 0.0, 4.0),
            start_radius: 5.0,
            end_radius: 1.0,
        };
        let points = round_probes(8.0, |p| {
            let z = p.z.clamp(-4.0, 4.0);
            (p - Point3::new(0.0, 0.0, z)).norm() < 0.05 || rho(p) < 0.05
        });
        let worst = gradient_deviation(&capsule, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
        assert_normals_match_the_gradient(&capsule, &points, "tapered capsule");
    }

    /// The endpoints are the cap *centres*, so the shape reaches a radius beyond each of them.
    /// The one thing about a capsule that is easy to get backwards.
    #[test]
    fn a_capsules_endpoints_are_its_cap_centres() {
        let capsule = Capsule {
            start: Point3::new(0.0, 0.0, -4.0),
            end: Point3::new(0.0, 0.0, 4.0),
            start_radius: 3.0,
            end_radius: 3.0,
        };
        // The cap centre is a whole radius inside, and the surface is a radius past it.
        assert!((capsule.distance(Point3::new(0.0, 0.0, 4.0)) + 3.0).abs() < 1e-12);
        assert!(
            capsule.distance(Point3::new(0.0, 0.0, 7.0)).abs() < 1e-12,
            "the end cap"
        );
        assert!((capsule.distance(Point3::new(0.0, 0.0, 9.0)) - 2.0).abs() < 1e-12);
        assert!(
            capsule.distance(Point3::new(3.0, 0.0, 2.0)).abs() < 1e-12,
            "the side wall"
        );
    }

    /// Both ends in one place is exactly a sphere — the cheapest check that the degenerate
    /// segment falls back cleanly instead of dividing by its own length.
    #[test]
    fn a_capsule_with_both_ends_together_is_a_sphere() {
        let at = Point3::new(1.0, 2.0, 3.0);
        let capsule = Capsule {
            start: at,
            end: at,
            start_radius: 5.0,
            end_radius: 5.0,
        };
        let sphere = Sphere {
            centre: at,
            radius: 5.0,
        };
        for p in probes(5.0) {
            assert!((capsule.distance(p) - sphere.distance(p)).abs() < 1e-12);
        }
    }

    /// A cone with equal radii is a cylinder, and a capsule whose caps swallow each other is
    /// a sphere. Both are the boundaries of the two-radius forms, and both are where a
    /// division by a vanishing quantity would surface.
    #[test]
    fn the_two_radius_forms_degenerate_to_the_shapes_they_contain() {
        let base = Point3::new(0.0, 0.0, -5.0);
        let top = Point3::new(0.0, 0.0, 5.0);
        let cone = Cone {
            base,
            top,
            base_radius: 3.0,
            top_radius: 3.0,
        };
        let cylinder = Cylinder {
            base,
            top,
            radius: 3.0,
        };
        for p in probes(10.0) {
            let (a, b) = (cone.distance(p), cylinder.distance(p));
            assert!((a - b).abs() < 1e-9, "cone {a} vs cylinder {b} at {p:?}");
        }

        // One cap entirely inside the other: the shape is the larger sphere, and there is no
        // tangent cone to compute. Left unguarded this is a square root of a negative number.
        let swallowed = Capsule {
            start: Point3::origin(),
            end: Point3::new(0.0, 0.0, 1.0),
            start_radius: 8.0,
            end_radius: 1.0,
        };
        let sphere = Sphere {
            centre: Point3::origin(),
            radius: 8.0,
        };
        for p in probes(8.0) {
            let d = swallowed.distance(p);
            assert!(d.is_finite(), "distance went non-finite at {p:?}");
            assert!(
                (d - sphere.distance(p)).abs() < 1e-9,
                "{d} vs {}",
                sphere.distance(p)
            );
        }
    }

    /// The whole point of defining these between two points: the shape has to be the same
    /// shape whichever way the segment runs, and its distances have to follow it there.
    ///
    /// Checked against the upright case by rotating the query point instead of comparing to a
    /// second hand-written formula — the frame is the thing under test, so the reference has
    /// to come from somewhere the frame is not involved.
    #[test]
    fn a_two_point_solid_is_the_same_shape_whichever_way_it_lies() {
        use nalgebra::{Rotation3, Unit, Vector3};
        let upright = upright_cylinder(3.0, 8.0);
        // An axis sharing no plane with X, Y or Z, so a transposed frame cannot pass.
        let axis = Unit::new_normalize(Vector3::new(0.37, -0.81, 0.45));
        let rotation = Rotation3::from_axis_angle(&axis, 1.13);
        let tilted = Cylinder {
            base: rotation * upright.base,
            top: rotation * upright.top,
            radius: upright.radius,
        };
        for p in probes(8.0) {
            let here = rotation * p;
            let (a, b) = (upright.distance(p), tilted.distance(here));
            assert!((a - b).abs() < 1e-9, "upright {a} vs tilted {b}");
            if let Some(n) = upright.normal(p) {
                let m = tilted
                    .normal(here)
                    .expect("the tilted one answers wherever the upright does");
                assert!(
                    (rotation * n - m).norm() < 1e-9,
                    "normal did not follow the rotation"
                );
            }
        }
    }

    #[test]
    fn a_cylinder_measures_diagonally_off_its_rim() {
        let cylinder = upright_cylinder(4.0, 10.0);
        assert!(
            (cylinder.distance(Point3::origin()) + 4.0).abs() < 1e-12,
            "the radius is nearest"
        );
        assert!(
            cylinder.distance(Point3::new(4.0, 0.0, 0.0)).abs() < 1e-12,
            "on the side"
        );
        assert!(
            cylinder.distance(Point3::new(0.0, 0.0, 5.0)).abs() < 1e-12,
            "on a cap"
        );
        // Three out and four up from the rim at (4, 0, 5): 5, not 4.
        let d = cylinder.distance(Point3::new(7.0, 0.0, 9.0));
        assert!((d - 5.0).abs() < 1e-12, "got {d}");
    }

    /// A torus is a sphere problem once folded onto its ring, so its extremes are exact.
    #[test]
    fn a_torus_is_its_minor_radius_from_its_ring() {
        let torus = Torus {
            centre: Point3::origin(),
            major_radius: 6.0,
            minor_radius: 2.0,
        };
        assert!(
            (torus.distance(Point3::new(6.0, 0.0, 0.0)) + 2.0).abs() < 1e-12,
            "on the ring"
        );
        assert!(
            torus.distance(Point3::new(8.0, 0.0, 0.0)).abs() < 1e-12,
            "the outer surface"
        );
        assert!(
            torus.distance(Point3::new(4.0, 0.0, 0.0)).abs() < 1e-12,
            "the inner surface"
        );
        // The hole: on the axis, the nearest ring point is a whole major radius away.
        assert!(
            (torus.distance(Point3::origin()) - 4.0).abs() < 1e-12,
            "the middle of the hole"
        );
    }
}

#[cfg(test)]
mod skewed_box_tests {
    use super::*;
    use crate::gradient_deviation;

    /// Deterministic points, spread through a region, with no dependency on a random generator.
    fn scatter(n: usize, lo: f64, hi: f64) -> Vec<Point3<f64>> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((state >> 11) as f64 / (1u64 << 53) as f64)
        };
        (0..n).map(|_| Point3::new(next(), next(), next())).collect()
    }

    fn leaning() -> SkewedBox {
        SkewedBox::new(
            Point3::new(2.0, -1.0, 3.0),
            [Vector3::new(10.0, 0.0, 0.0), Vector3::new(4.0, 8.0, 0.0), Vector3::new(3.0, 2.0, 6.0)],
        )
        .expect("three independent edges")
    }

    /// The surface of the box, sampled finely: each face as a grid of its parallelogram.
    fn surface_samples(b: &SkewedBox, per_side: usize) -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for i in 0..3 {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            for face in [-0.5, 0.5] {
                for a in 0..=per_side {
                    for c in 0..=per_side {
                        let (u, v) = (a as f64 / per_side as f64 - 0.5, c as f64 / per_side as f64 - 0.5);
                        out.push(b.centre + b.edges[i] * face + b.edges[j] * u + b.edges[k] * v);
                    }
                }
            }
        }
        out
    }

    /// **The control**: with square edges it is the box it looks like, to rounding, at points
    /// inside, outside and beside every edge and corner.
    #[test]
    fn a_skewed_box_with_square_edges_is_the_cuboid_it_looks_like() {
        let centre = Point3::new(1.0, 2.0, -3.0);
        let size = Vector3::new(10.0, 6.0, 4.0);
        let skewed = SkewedBox::new(
            centre,
            [Vector3::new(size.x, 0.0, 0.0), Vector3::new(0.0, size.y, 0.0), Vector3::new(0.0, 0.0, size.z)],
        )
        .unwrap();
        let boxed = Cuboid { centre, size };
        let mut worst = 0.0f64;
        for p in scatter(4000, -14.0, 14.0) {
            worst = worst.max((skewed.distance(p) - boxed.distance(p)).abs());
        }
        assert!(worst < 1e-9, "square edges should be a cuboid, differing by {worst}");
    }

    /// Exact against the surface itself: no sample of the surface is nearer than the distance
    /// reported (never overstated), and the nearest sample is within the sampling's own spacing of
    /// it (not understated) — outside and inside both.
    #[test]
    fn a_skewed_box_is_exact_against_the_nearest_point_of_its_own_surface() {
        let b = leaning();
        let surface = surface_samples(&b, 90);
        // The sampling grid's spacing bounds how far the nearest sample can be from the nearest point.
        let spacing = b.edges.iter().map(Vector3::norm).fold(0.0, f64::max) / 90.0;
        let mut understated = 0.0f64;
        for p in scatter(300, -12.0, 18.0) {
            let nearest = surface.iter().map(|s| (p - s).norm()).fold(f64::INFINITY, f64::min);
            let d = b.distance(p).abs();
            assert!(d <= nearest + 1e-9, "{p:?}: reports {d}, but a surface point is {nearest} away");
            understated = understated.max(nearest - d);
        }
        assert!(
            understated < spacing,
            "the nearest sample is {understated} further than reported, past the {spacing} spacing"
        );
    }

    #[test]
    fn a_skewed_boxs_gradient_is_unit_outside_and_its_normal_agrees_with_a_central_difference() {
        let b = leaning();
        let outside: Vec<Point3<f64>> =
            scatter(800, -20.0, 28.0).into_iter().filter(|p| b.distance(*p) > 0.5).collect();
        assert!(outside.len() > 300);
        let deviation = gradient_deviation(&b, &outside, 1e-4);
        assert!(deviation < 1e-6, "|∇d| strays from 1 by {deviation} outside a convex solid");

        let h = 1e-5;
        for p in outside.iter().take(200) {
            let analytic = b.normal(*p).expect("a normal outside");
            let at = |d: Vector3<f64>| b.distance(p + d);
            let numeric = Vector3::new(
                at(Vector3::x() * h) - at(-Vector3::x() * h),
                at(Vector3::y() * h) - at(-Vector3::y() * h),
                at(Vector3::z() * h) - at(-Vector3::z() * h),
            ) / (2.0 * h);
            assert!((analytic - numeric).norm() < 1e-5, "{p:?}: {analytic:?} against {numeric:?}");
        }
    }

    /// Inside, the gradient is the unit normal of the nearest face, and the field falls to nought at
    /// the surface: a point on a face reads nought, the centre reads the inradius of the nearest pair.
    #[test]
    fn a_skewed_box_reads_nought_on_its_faces_and_negative_inside() {
        let b = leaning();
        for corner in b.corners() {
            assert!(b.distance(corner).abs() < 1e-9, "a corner is on the surface");
        }
        assert!(b.distance(b.centre) < 0.0);
        let mid_face = b.centre + b.edges[1] * 0.5;
        assert!(b.distance(mid_face).abs() < 1e-9, "the middle of a face is on the surface");
        // The inside is a field of its own: 1-Lipschitz, so never faster than a millimetre a millimetre.
        let mut steepest = 0.0f64;
        let pts = scatter(600, -4.0, 8.0);
        for pair in pts.chunks(2) {
            if let [a, c] = pair {
                let moved = (a - c).norm();
                if moved > 1e-6 {
                    steepest = steepest.max((b.distance(*a) - b.distance(*c)).abs() / moved);
                }
            }
        }
        assert!(steepest <= 1.0 + 1e-9, "rises {steepest} mm per millimetre");
    }

    #[test]
    fn three_edges_in_a_plane_make_no_box() {
        let flat = SkewedBox::new(
            Point3::origin(),
            [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 0.0)],
        );
        assert!(flat.is_none(), "dependent edges have no volume");
        let almost = SkewedBox::new(
            Point3::origin(),
            [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 1e-12)],
        );
        assert!(almost.is_none(), "a hair off flat is judged against the edges' own size");
    }
}
