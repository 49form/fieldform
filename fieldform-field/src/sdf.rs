// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Signed distance to a closed triangle mesh — a cage's limit surface, or an imported one.
//!
//! One type, two ways in. The machinery here was written for a cage and turned out never to
//! have been about cages: it wants positions, triangles and a normal per vertex, and where
//! those came from is not its business. [`MeshField::from_cage`] tessellates a limit surface
//! into them; [`MeshField::from_mesh`] takes them from a file. `import.md` called this "the
//! wall" and estimated a new leaf; the leaf was already here behind a cage-shaped door.

use crate::bvh::Bvh;
use fieldform_cage::{Cage, CageError};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// Result of a closest-point query.
#[derive(Copy, Clone, Debug)]
pub struct ClosestPoint {
    pub point: Point3<f64>,
    /// Signed distance: negative inside the surface, positive outside.
    pub signed_distance: f64,
    /// The limit surface's exact analytic normal at `point`, barycentric-interpolated from
    /// the triangle it landed on — not the triangle's own flat facet normal, and not derived
    /// by finite-differencing this field. See [`CageField`]'s docs for why that distinction
    /// is the point.
    pub normal: Vector3<f64>,
}

/// Which part of a triangle a closest point landed on. This matters for sign
/// determination: a point closest to an edge or vertex is equidistant from several
/// triangles, and picking any one of their face normals gives the wrong answer in concave
/// regions. The pseudonormal for the actual feature is what's provably correct.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Feature {
    Face(usize),
    Edge(u32, u32),
    Vertex(u32),
}

/// Beyond this, two faces meeting at an edge are a *crease* and are shaded separately.
///
/// 30\u{b0} is the usual choice and it holds here for a reason worth writing down: the STEP
/// importer's angular deflection decides how far apart two adjacent facets on a curved face can
/// be, and the "normal" preset is 0.2 rad = 11.5\u{b0}, so neighbours differ by at most about
/// 23\u{b0} — under the threshold, so a tessellated cylinder stays smooth by construction. A
/// machined edge is 90\u{b0} and is nowhere near it.
///
/// The coarse preset is 0.5 rad = 28.6\u{b0}, where neighbours can exceed 30\u{b0} and a curved
/// face begins to shade as facets. That is honest rather than a failure: at that deflection the
/// facets are visible in the silhouette anyway, and shading them smooth would be painting over a
/// tessellation the user asked for.
const CREASE_ANGLE_DEG: f64 = 30.0;

/// A closed triangle mesh, as a signed distance field.
///
/// Closed is a requirement, not a preference: the sign comes from which side of the surface a
/// point is on, and an open mesh has no inside to be on the wrong side of.
///
/// The sign test is the angle-weighted pseudonormal of Bærentzen & Aanæs, which is what makes
/// it *provably* right rather than usually right. Picking the nearest triangle's own normal
/// fails wherever the closest point lands on an edge or a vertex — there it is equidistant from
/// several triangles, and in a concave region their normals disagree about the answer.
pub struct MeshField {
    positions: Vec<Point3<f64>>,
    triangles: Vec<[u32; 3]>,
    /// The best normal available at each vertex, and where it came from depends on the door
    /// used: [`MeshField::from_cage`] supplies the limit surface's *exact analytic* normals,
    /// which is why raising the tessellation level does not trade shading against position.
    /// [`MeshField::from_mesh`] has no analytic surface to ask, so it supplies the same
    /// angle-weighted average the sign test builds — the best a triangle soup can offer.
    shading_normals: Vec<Vector3<f64>>,
    /// One normal per triangle *corner*, when the mesh came from a file rather than a cage.
    ///
    /// Empty for a cage, whose `shading_normals` are the limit surface's own analytic normals
    /// and are genuinely per-vertex — a subdivision surface is smooth everywhere, so there are
    /// no creases to preserve and a second set would say the same thing twice.
    ///
    /// Per corner rather than per vertex because a crease needs one vertex to carry a different
    /// normal for each face meeting there, which a per-vertex array cannot express. See
    /// [`MeshField::from_mesh`] for what goes in it.
    corner_normals: Vec<[Vector3<f64>; 3]>,
    face_normals: Vec<Vector3<f64>>,
    /// Angle-weighted vertex pseudonormals (Bærentzen & Aanæs). The angle weighting is
    /// what makes the inside/outside test correct rather than merely usually-right.
    vertex_pseudonormals: Vec<Vector3<f64>>,
    /// Sum of the two adjacent faces' normals, keyed by the edge's ordered endpoints.
    edge_pseudonormals: HashMap<(u32, u32), Vector3<f64>>,
    /// Spatial index over `triangles`, so a query does not scan all of them.
    bvh: Bvh,
}

impl std::fmt::Debug for MeshField {
    /// A summary, not the contents. An imported mesh can be millions of triangles, and a
    /// `Debug` that printed them would be unreadable in exactly the situation it is reached
    /// for — a failing assertion, in a terminal.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshField")
            .field("vertices", &self.positions.len())
            .field("triangles", &self.triangles.len())
            .finish()
    }
}

/// A cage's limit surface, as a field.
///
/// The same type as [`MeshField`], under the name the cage side of the codebase calls it. Not a
/// wrapper and not a copy: sampling a limit surface produces a closed triangle mesh, so there
/// was never a second thing here to be kept in step with the first.
pub type CageField = MeshField;

/// Why a triangle mesh could not become a field.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// No triangles at all.
    Empty,
    /// A triangle names a vertex the position list does not have.
    BadIndex(u32),
    /// An edge shared by other than exactly two triangles, so the mesh is not a closed
    /// surface and has no inside for a sign to refer to.
    NotClosed { edge: (u32, u32), uses: usize },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Empty => write!(f, "the mesh has no triangles"),
            MeshError::BadIndex(v) => {
                write!(
                    f,
                    "a triangle refers to vertex {v}, which is not in the mesh"
                )
            }
            MeshError::NotClosed { edge, uses } => write!(
                f,
                "the mesh is not closed: the edge between vertices {} and {} is used by {uses} \
                 triangles rather than 2, so the mesh has no inside",
                edge.0, edge.1,
            ),
        }
    }
}

impl std::error::Error for MeshError {}

impl MeshField {
    /// Builds the field from a control cage: sample the limit surface on a uniform
    /// `2^levels`-resolution grid per face, welding shared edges and corners into one
    /// watertight mesh — see [`fieldform_subd::LimitSurface::tessellate`].
    ///
    /// Positions and normals both come from that one evaluation, so raising `levels` no
    /// longer trades position accuracy against shading smoothness: normals are exact
    /// regardless of level, and `levels` only controls how closely the *distance* field
    /// tracks the true surface between sample points.
    pub fn from_cage(cage: &Cage, levels: usize) -> Result<Self, CageError> {
        let mut limit_surface = fieldform_subd::LimitSurface::new(cage);
        let fieldform_subd::Tessellation {
            positions,
            normals,
            triangles,
            // Where on the surface each triangle came from. A distance field does not need it —
            // it answers "how far", and the parametric position is a different question, asked
            // by the conformal cell map through `SurfaceMap`.
            ..
        } = limit_surface.tessellate(levels);
        Ok(Self::from_triangles(positions, normals, triangles))
    }

    /// Builds the field from an arbitrary closed triangle mesh — an import, rather than a cage.
    ///
    /// The door `import.md` said had to exist before a STEP, STL or triangulated OBJ importer
    /// had anywhere to put its result. A `Cage` refuses anything that is not closed, manifold
    /// and all-quad, which is right for scaffolding a limit surface and wrong for finished
    /// geometry that is only ever going to be referenced, cut against or blended with.
    ///
    /// **Closed is still required**, and checked rather than assumed. It is the one property
    /// the sign test cannot do without: an open mesh has no inside, so every "distance" it
    /// produced would carry a sign invented from whichever triangle happened to be nearest.
    /// Refusing is the honest answer, and it is the same refusal `Cage::from_faces` makes for
    /// the same underlying reason.
    ///
    /// Normals are the angle-weighted vertex averages the sign test already computes, since
    /// there is no analytic surface to ask for better ones.
    pub fn from_mesh(
        positions: Vec<Point3<f64>>,
        triangles: Vec<[u32; 3]>,
    ) -> Result<Self, MeshError> {
        if triangles.is_empty() {
            return Err(MeshError::Empty);
        }
        let limit = positions.len() as u32;
        for t in &triangles {
            for &v in t {
                if v >= limit {
                    return Err(MeshError::BadIndex(v));
                }
            }
        }

        // Every edge used by exactly two triangles is what closed and manifold means here, and
        // one pass counts both failures at once: a count of one is a hole, and three or more is
        // a seam where sheets meet and "inside" stops being a single region.
        let mut uses: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &triangles {
            for k in 0..3 {
                *uses.entry(edge_key(t[k], t[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        // The lowest offending edge rather than whichever the hash happens to yield first, so
        // the same bad file names the same edge every time it is opened. An error a user is
        // going to quote back has to be reproducible.
        if let Some((&edge, &count)) = uses
            .iter()
            .filter(|&(_, &c)| c != 2)
            .min_by_key(|(e, _)| **e)
        {
            return Err(MeshError::NotClosed { edge, uses: count });
        }

        // Placeholder normals: `from_triangles` computes the angle-weighted pseudonormals it
        // needs for the sign test anyway. They are the right answer for the *sign*, and they
        // were once used for shading too — which is the bug this replaces.
        let placeholder = vec![Vector3::zeros(); positions.len()];
        let mut field = Self::from_triangles(positions, placeholder, triangles);
        field.shading_normals = field.vertex_pseudonormals.clone();
        field.corner_normals = field.crease_corner_normals();
        Ok(field)
    }

    /// The angle triangle `face` subtends at vertex `v`, or zero if it does not use it.
    ///
    /// The same weighting the vertex pseudonormals use, and for the same reason: it makes the
    /// result independent of how the neighbourhood happens to be triangulated, so a face split
    /// into two long slivers does not outvote one kept whole.
    fn corner_angle(&self, face: usize, v: u32) -> f64 {
        let t = &self.triangles[face];
        let Some(k) = t.iter().position(|&x| x == v) else {
            return 0.0;
        };
        let p = self.positions[v as usize];
        let e0 = (self.positions[t[(k + 1) % 3] as usize] - p).try_normalize(1e-12);
        let e1 = (self.positions[t[(k + 2) % 3] as usize] - p).try_normalize(1e-12);
        match (e0, e1) {
            (Some(e0), Some(e1)) => e0.dot(&e1).clamp(-1.0, 1.0).acos(),
            _ => 0.0,
        }
    }

    /// A shading normal per triangle corner, averaged only across neighbours that are not
    /// separated from it by a crease.
    ///
    /// **Why this is not the vertex pseudonormals.** Those average every face meeting a vertex,
    /// which is correct for the sign test and wrong for shading a machined part: at a box's
    /// corner the average of three perpendicular faces points diagonally, and since a box face
    /// is two triangles, barycentric interpolation carries that error across the entire face.
    /// Measured before this existed, on a 200 x 120 x 40 box: the normal on the flat +Z face
    /// swung from 0° at the centre to **45° at the edge** — a face with a gradient painted on
    /// it, which is what "always makes things look rounder than they actually are" describes.
    /// `fieldform-gpu/examples/baked_edge_rounding.rs` prints the table, and shows that raising
    /// the bake resolution does not touch it: the bake was faithfully reproducing a normal that
    /// was already wrong.
    ///
    /// **Why not simply the face normal.** That is flat shading, and it fixes the machined case
    /// by breaking the other one: a tessellated cylinder is a real curved surface approximated
    /// by facets, and flat-shading it shows every facet. Both complaints have been made about
    /// this viewport, a week apart.
    ///
    /// So: average across a neighbour only when the two faces differ by less than
    /// [`CREASE_ANGLE_DEG`]. A machined edge is 90° and stays sharp; a tessellated curve's
    /// facets are a few degrees apart and stay smooth.
    fn crease_corner_normals(&self) -> Vec<[Vector3<f64>; 3]> {
        // Triangles meeting each vertex, as one flat array with an index into it — a counting
        // sort rather than a `Vec` per vertex. An imported assembly is 100 000 triangles and
        // 50 000 vertices, and a `Vec<Vec<_>>` there is 50 000 separate allocations for a
        // structure read once and dropped.
        let vertex_count = self.positions.len();
        let mut starts = vec![0u32; vertex_count + 1];
        for t in &self.triangles {
            for &v in t {
                starts[v as usize + 1] += 1;
            }
        }
        for i in 0..vertex_count {
            starts[i + 1] += starts[i];
        }
        let mut filled = starts.clone();
        let mut incident = vec![0u32; self.triangles.len() * 3];
        for (i, t) in self.triangles.iter().enumerate() {
            for &v in t {
                incident[filled[v as usize] as usize] = i as u32;
                filled[v as usize] += 1;
            }
        }
        let at_vertex = |v: u32| -> &[u32] {
            &incident[starts[v as usize] as usize..starts[v as usize + 1] as usize]
        };

        let cos_crease = CREASE_ANGLE_DEG.to_radians().cos();
        self.triangles
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let own = self.face_normals[i];
                let mut corners = [own; 3];
                for (k, &v) in t.iter().enumerate() {
                    let mut sum = Vector3::zeros();
                    for &j in at_vertex(v) {
                        let other = self.face_normals[j as usize];
                        // Weighted by the angle the face subtends at this vertex, the same
                        // weighting the pseudonormals use and for the same reason: it makes the
                        // result independent of how the neighbourhood happens to be
                        // triangulated.
                        if own.dot(&other) >= cos_crease {
                            sum += other * self.corner_angle(j as usize, v);
                        }
                    }
                    corners[k] = sum.try_normalize(1e-12).unwrap_or(own);
                }
                corners
            })
            .collect()
    }

    fn from_triangles(
        positions: Vec<Point3<f64>>,
        shading_normals: Vec<Vector3<f64>>,
        triangles: Vec<[u32; 3]>,
    ) -> Self {
        let face_normals: Vec<Vector3<f64>> = triangles
            .iter()
            .map(|t| {
                let a = positions[t[0] as usize];
                let b = positions[t[1] as usize];
                let c = positions[t[2] as usize];
                // Guarded, where a cage's own tessellation never needed it: a sliver
                // triangle is normal in an imported mesh and its cross product is zero, and
                // an unguarded normalize turns that into NaN that spreads through every
                // pseudonormal touching it. A degenerate face subtends no angle and so
                // contributes nothing, which is also the geometrically right answer.
                (b - a)
                    .cross(&(c - a))
                    .try_normalize(1e-12)
                    .unwrap_or_else(Vector3::zeros)
            })
            .collect();

        let mut vertex_pseudonormals = vec![Vector3::zeros(); positions.len()];
        let mut edge_pseudonormals: HashMap<(u32, u32), Vector3<f64>> = HashMap::new();

        for (i, t) in triangles.iter().enumerate() {
            let n = face_normals[i];
            for k in 0..3 {
                let cur = t[k];
                let prev = t[(k + 2) % 3];
                let next = t[(k + 1) % 3];

                // Weight each incident face normal by the angle it subtends at this
                // vertex — the specific weighting that makes the resulting pseudonormal
                // valid for inside/outside tests.
                let p = positions[cur as usize];
                let e0 = (positions[next as usize] - p).try_normalize(1e-12);
                let e1 = (positions[prev as usize] - p).try_normalize(1e-12);
                // A repeated vertex has no direction to subtend an angle between, so it
                // weighs nothing rather than poisoning the sum.
                let angle = match (e0, e1) {
                    (Some(e0), Some(e1)) => e0.dot(&e1).clamp(-1.0, 1.0).acos(),
                    _ => 0.0,
                };
                vertex_pseudonormals[cur as usize] += n * angle;

                // Edge pseudonormal: the sum of both adjacent faces' normals. Keyed
                // undirected so both triangles sharing the edge accumulate into one entry.
                let key = edge_key(cur, next);
                *edge_pseudonormals.entry(key).or_insert_with(Vector3::zeros) += n;
            }
        }

        for n in &mut vertex_pseudonormals {
            let len = n.norm();
            if len > 1e-12 {
                *n /= len;
            }
        }

        let bvh = Bvh::build(&positions, &triangles);
        Self {
            positions,
            triangles,
            shading_normals,
            corner_normals: Vec::new(),
            face_normals,
            vertex_pseudonormals,
            edge_pseudonormals,
            bvh,
        }
    }

    /// Signed distance from `p` to the limit surface: negative inside, positive outside.
    pub fn distance(&self, p: Point3<f64>) -> f64 {
        self.closest(p).signed_distance
    }

    /// Closest point on the surface to `p`, with its signed distance.
    ///
    /// Searched through a bounding volume hierarchy rather than by scanning every triangle.
    /// That is what makes a finer triangulation affordable, and a finer triangulation is what
    /// makes composed surfaces shade smoothly — see [`crate::bvh`] for the measurements.
    pub fn closest(&self, p: Point3<f64>) -> ClosestPoint {
        let mut best_dist_sq = f64::INFINITY;
        let mut best_point = Point3::origin();
        let mut best_feature = Feature::Face(0);

        self.bvh.nearest(p, |i| {
            let i = i as usize;
            let t = &self.triangles[i];
            let a = self.positions[t[0] as usize];
            let b = self.positions[t[1] as usize];
            let c = self.positions[t[2] as usize];
            let (q, feature) = closest_point_on_triangle(p, a, b, c, i, t);
            let d_sq = (q - p).norm_squared();
            if d_sq < best_dist_sq {
                best_dist_sq = d_sq;
                best_point = q;
                best_feature = feature;
            }
            d_sq
        });

        let sign_normal = self.pseudonormal(best_feature);
        let offset = p - best_point;
        let sign = if offset.dot(&sign_normal) < 0.0 {
            -1.0
        } else {
            1.0
        };

        ClosestPoint {
            point: best_point,
            signed_distance: sign * best_dist_sq.sqrt(),
            normal: self.shading_normal(best_point, best_feature),
        }
    }

    /// The pseudonormal used for the inside/outside sign test — a flat-facet or
    /// angle-weighted average over the triangulation, not the true surface normal. See
    /// [`Self::shading_normal`] for the one shading actually wants.
    fn pseudonormal(&self, feature: Feature) -> Vector3<f64> {
        match feature {
            Feature::Face(i) => self.face_normals[i],
            Feature::Vertex(v) => self.vertex_pseudonormals[v as usize],
            Feature::Edge(a, b) => self
                .edge_pseudonormals
                .get(&edge_key(a, b))
                .copied()
                .unwrap_or_else(Vector3::zeros),
        }
    }

    /// The limit surface's exact analytic normal at `point`, interpolated from
    /// `shading_normals` rather than from `face_normals`/the pseudonormals above — those
    /// exist only to get the *sign* right near edges and vertices, and using them for shading
    /// is exactly the flat-facet artifact `shading_normals` was built to avoid.
    fn shading_normal(&self, point: Point3<f64>, feature: Feature) -> Vector3<f64> {
        match feature {
            Feature::Vertex(v) => self.shading_normals[v as usize],
            Feature::Edge(a, b) => {
                let pa = self.positions[a as usize];
                let pb = self.positions[b as usize];
                let len_sq = (pb - pa).norm_squared();
                let t = if len_sq > 1e-18 {
                    ((point - pa).dot(&(pb - pa)) / len_sq).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let n = self.shading_normals[a as usize] * (1.0 - t)
                    + self.shading_normals[b as usize] * t;
                n.try_normalize(1e-12)
                    .unwrap_or(self.shading_normals[a as usize])
            }
            Feature::Face(i) => {
                let t = &self.triangles[i];
                let a = self.positions[t[0] as usize];
                let b = self.positions[t[1] as usize];
                let c = self.positions[t[2] as usize];
                let (u, v, w) = barycentric(point, a, b, c);
                // Per corner when the mesh carries them, which is the case that has creases to
                // keep. A cage has none and interpolates its own analytic normals instead.
                let (na, nb, nc) = match self.corner_normals.get(i) {
                    Some([na, nb, nc]) => (*na, *nb, *nc),
                    None => (
                        self.shading_normals[t[0] as usize],
                        self.shading_normals[t[1] as usize],
                        self.shading_normals[t[2] as usize],
                    ),
                };
                let n = na * u + nb * v + nc * w;
                n.try_normalize(1e-12).unwrap_or(self.face_normals[i])
            }
        }
    }

    /// Number of triangles in the sampled surface, for diagnostics.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Barycentric weights of `p` with respect to triangle `abc`. Assumes `p` already lies in the
/// triangle's plane (true of every point [`closest_point_on_triangle`] returns as a face hit).
fn barycentric(p: Point3<f64>, a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> (f64, f64, f64) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-18 {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w, v, w)
}

/// Closest point on triangle `abc` to `p`, plus which feature of the triangle it landed on.
///
/// The standard Voronoi-region method (Ericson, *Real-Time Collision Detection*): check the
/// regions outside each vertex and edge in turn, and fall through to the face interior.
fn closest_point_on_triangle(
    p: Point3<f64>,
    a: Point3<f64>,
    b: Point3<f64>,
    c: Point3<f64>,
    tri_index: usize,
    tri: &[u32; 3],
) -> (Point3<f64>, Feature) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(tri[0]));
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(tri[1]));
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Feature::Edge(tri[0], tri[1]));
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(tri[2]));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Feature::Edge(tri[0], tri[2]));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, Feature::Edge(tri[1], tri[2]));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Feature::Face(tri_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn cube_field() -> CageField {
        CageField::from_cage(&Cage::cube(), 3).unwrap()
    }

    /// The hierarchy must not change any answer, on real cage geometry and including the
    /// inside/outside test — pruning that occasionally flipped a sign would read as a speck of
    /// inverted surface rather than as an obvious failure.
    ///
    /// The comparison is on the **signed distance**, not on which triangle won. Those are not
    /// the same requirement: a point equidistant from two triangles has no single right answer
    /// for *which*, and the two search orders legitimately disagree there (this test found
    /// exactly that at [200, 600, 200] when it demanded matching closest points). The signed
    /// distance is what the rest of the system consumes, and it is identical either way.
    #[test]
    fn bvh_queries_agree_with_a_full_scan_on_a_real_cage() {
        let field = CageField::from_cage(&Cage::cube_of(1000.0), 3).unwrap();

        let mut checked = 0;
        for k in 0..400 {
            let p = Point3::new(
                ((k * 37) % 1600) as f64 - 800.0,
                ((k * 71) % 1600) as f64 - 800.0,
                ((k * 13) % 1600) as f64 - 800.0,
            );

            // The same search, scanning every triangle in index order.
            let mut best_sq = f64::INFINITY;
            let mut best_point = Point3::origin();
            let mut best_feature = Feature::Face(0);
            for (i, t) in field.triangles.iter().enumerate() {
                let a = field.positions[t[0] as usize];
                let b = field.positions[t[1] as usize];
                let c = field.positions[t[2] as usize];
                let (q, feature) = closest_point_on_triangle(p, a, b, c, i, t);
                let d_sq = (q - p).norm_squared();
                if d_sq < best_sq {
                    best_sq = d_sq;
                    best_point = q;
                    best_feature = feature;
                }
            }
            let normal = field.pseudonormal(best_feature);
            let sign = if (p - best_point).dot(&normal) < 0.0 {
                -1.0
            } else {
                1.0
            };
            let expected = sign * best_sq.sqrt();

            let got = field.closest(p).signed_distance;
            assert!(
                (got - expected).abs() < 1e-9,
                "at {p:?}: hierarchy {got} vs full scan {expected}"
            );
            checked += 1;
        }
        assert_eq!(checked, 400);
    }

    #[test]
    fn origin_is_inside_and_surface_points_are_on_it() {
        let field = cube_field();

        // The centre of a closed surface must be inside, i.e. negative.
        let at_origin = field.distance(Point3::origin());
        assert!(
            at_origin < 0.0,
            "origin should be inside the surface, got {at_origin}"
        );

        // Every vertex of the sampled surface lies on it, so distance there is ~0.
        for &p in &field.positions {
            assert_relative_eq!(field.distance(p), 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn distance_grows_linearly_moving_away_along_an_axis() {
        let field = cube_field();
        // Far outside along +X, the distance should track how far we've moved: adding 1.0
        // to the query's distance from the surface adds ~1.0 to the reported distance.
        let d1 = field.distance(Point3::new(2.0, 0.0, 0.0));
        let d2 = field.distance(Point3::new(3.0, 0.0, 0.0));
        assert!(d1 > 0.0 && d2 > 0.0, "points outside should be positive");
        assert_relative_eq!(d2 - d1, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn sign_flips_exactly_once_crossing_the_surface() {
        // Walk a ray from deep inside to well outside; the sign must change exactly once.
        // More than one flip would mean the inside/outside test is unstable — the failure
        // mode a naive closest-face-normal test produces near edges and vertices.
        let field = cube_field();
        let mut flips = 0;
        let mut prev = field.distance(Point3::origin()).signum();
        for i in 1..=200 {
            let t = i as f64 / 200.0 * 2.0;
            let d = field.distance(Point3::new(t, t * 0.3, t * 0.7));
            let s = d.signum();
            if s != prev {
                flips += 1;
                prev = s;
            }
        }
        assert_eq!(flips, 1, "expected exactly one inside->outside transition");
    }

    #[test]
    fn cube_limit_surface_is_roughly_spherical() {
        // The whole premise: a cube cage's limit surface is a rounded, sphere-like form.
        // Measure it by how far the surface sits from the origin along several directions.
        let field = cube_field();
        let dirs = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0).normalize(),
            Vector3::new(1.0, 1.0, 0.0).normalize(),
        ];
        let radii: Vec<f64> = dirs
            .iter()
            .map(|d| {
                // distance(origin) is negative; the surface along `d` is that far out.
                let inside = -field.distance(Point3::origin());
                // Refine by walking out until the sign flips.
                let mut lo = 0.0f64;
                let mut hi = inside * 3.0 + 1.0;
                for _ in 0..40 {
                    let mid = (lo + hi) * 0.5;
                    if field.distance(Point3::from(d * mid)) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                (lo + hi) * 0.5
            })
            .collect();

        let min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = radii.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            max / min < 1.2,
            "limit surface should be near-spherical; radii {radii:?}"
        );
        // And strictly inside the original cage, whose corners sit at radius sqrt(3)/2.
        assert!(max < 0.5 * 3.0f64.sqrt());
    }

    #[test]
    fn gradient_points_outward() {
        // Central differences on the field should give an outward-pointing normal; check
        // it roughly agrees with the radial direction on a near-spherical surface.
        let field = cube_field();
        let p = Point3::new(0.45, 0.1, 0.1);
        let h = 1e-3;
        let grad = Vector3::new(
            field.distance(p + Vector3::new(h, 0.0, 0.0))
                - field.distance(p - Vector3::new(h, 0.0, 0.0)),
            field.distance(p + Vector3::new(0.0, h, 0.0))
                - field.distance(p - Vector3::new(0.0, h, 0.0)),
            field.distance(p + Vector3::new(0.0, 0.0, h))
                - field.distance(p - Vector3::new(0.0, 0.0, h)),
        ) / (2.0 * h);

        assert_relative_eq!(grad.norm(), 1.0, epsilon = 0.1);
        assert!(
            grad.dot(&p.coords.normalize()) > 0.8,
            "gradient {grad:?} should point outward at {p:?}"
        );
    }

    use crate::ScalarField;

    /// A closed box as a triangle soup: eight corners, twelve triangles, outward winding.
    fn box_mesh(half: f64) -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
        let h = half;
        let positions = vec![
            Point3::new(-h, -h, -h),
            Point3::new(h, -h, -h),
            Point3::new(h, h, -h),
            Point3::new(-h, h, -h),
            Point3::new(-h, -h, h),
            Point3::new(h, -h, h),
            Point3::new(h, h, h),
            Point3::new(-h, h, h),
        ];
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        (positions, triangles)
    }

    /// The test that says the whole thing works: a mesh of a cube has to agree with the cube.
    ///
    /// It checks the sign and the distance together against a field whose answer is known
    /// exactly and computed a completely different way, which no amount of self-consistency
    /// could. Corners and edges are included deliberately — they are where the nearest point is
    /// equidistant from several triangles, and where a naive face-normal sign test is wrong.
    #[test]
    fn a_meshed_cube_agrees_with_the_analytic_one() {
        let (positions, triangles) = box_mesh(5.0);
        let mesh = MeshField::from_mesh(positions, triangles).expect("a closed box");
        let exact = crate::Cuboid::cube(Point3::origin(), 10.0);

        let mut worst = 0.0f64;
        for i in -9..=9 {
            for j in -9..=9 {
                for k in -9..=9 {
                    let p = Point3::new(i as f64 * 1.1, j as f64 * 1.1, k as f64 * 1.1);
                    let got = mesh.distance(p);
                    let want = exact.distance(p);
                    assert_eq!(
                        got < 0.0,
                        want < 0.0,
                        "sign disagrees at {p:?}: mesh {got}, cube {want}",
                    );
                    worst = worst.max((got - want).abs());
                }
            }
        }
        assert!(worst < 1e-9, "worst distance disagreement {worst}");
    }

    /// D8's standing requirement for any new field source. A box is flat-sided, so the
    /// gradient is unit everywhere except on the creases where it genuinely turns.
    #[test]
    fn a_mesh_field_has_a_unit_gradient_away_from_its_creases() {
        let (positions, triangles) = box_mesh(5.0);
        let mesh = MeshField::from_mesh(positions, triangles).expect("a closed box");
        let exact = crate::Cuboid::cube(Point3::origin(), 10.0);
        let points: Vec<Point3<f64>> = box_probes()
            .into_iter()
            .filter(|p| {
                // Off the diagonals inside and away from the edges outside, the same filter
                // the analytic cube's own gradient test uses and for the same reason.
                let d = [p.x.abs() - 5.0, p.y.abs() - 5.0, p.z.abs() - 5.0];
                let near = |a: f64, b: f64| (a - b).abs() < 0.05;
                !(near(d[0], d[1]) || near(d[1], d[2]) || near(d[0], d[2]))
            })
            .collect();
        assert!(points.len() > 200, "not enough probes survived the filter");
        let worst = crate::gradient_deviation(&mesh, &points, 1e-4);
        assert!(worst < 1e-6, "worst gradient deviation {worst}");
        let _ = exact;
    }

    fn box_probes() -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for i in -8..=8 {
            for j in -8..=8 {
                for k in -8..=8 {
                    out.push(Point3::new(i as f64 * 1.3, j as f64 * 1.7, k as f64 * 1.9));
                }
            }
        }
        out
    }

    /// An open mesh is refused rather than answered, because the sign it would produce is
    /// invented. This is the whole reason `from_mesh` returns a `Result`.
    #[test]
    fn an_open_mesh_is_refused_rather_than_given_an_inside() {
        let (positions, mut triangles) = box_mesh(5.0);
        triangles.pop();
        let err = MeshField::from_mesh(positions, triangles).expect_err("a box with a hole");
        assert!(matches!(err, MeshError::NotClosed { .. }), "got {err:?}");
        // And the message has to name the problem rather than the symptom.
        assert!(err.to_string().contains("not closed"), "{err}");
    }

    /// A fin welded to a closed box has no single inside either, and the same one pass catches
    /// it: the shared edge is now used three times and the fin's own two edges once each.
    ///
    /// Asserted as the *kind* of failure rather than a particular edge, because a fin creates
    /// several at once and which one is reported is a detail. That it is reported
    /// deterministically is not a detail, and is checked below.
    #[test]
    fn a_non_manifold_seam_is_refused_too() {
        let (mut positions, mut triangles) = box_mesh(5.0);
        positions.push(Point3::new(0.0, 0.0, 20.0));
        triangles.push([0, 1, 8]);
        let err =
            MeshField::from_mesh(positions.clone(), triangles.clone()).expect_err("a fin on a box");
        assert!(matches!(err, MeshError::NotClosed { .. }), "got {err:?}");

        // The same file has to name the same edge every time, or a user cannot report it.
        for _ in 0..8 {
            let again = MeshField::from_mesh(positions.clone(), triangles.clone())
                .expect_err("still a fin");
            assert_eq!(again, err, "the reported edge moved between runs");
        }
    }

    #[test]
    fn a_mesh_with_no_triangles_or_a_bad_index_is_refused() {
        assert!(matches!(
            MeshField::from_mesh(Vec::new(), Vec::new()).err(),
            Some(MeshError::Empty),
        ));
        let (positions, _) = box_mesh(1.0);
        assert!(matches!(
            MeshField::from_mesh(positions, vec![[0, 1, 99]]).err(),
            Some(MeshError::BadIndex(99)),
        ));
    }

    /// A flat face on an imported mesh shades flat, and a gentle one still shades smooth.
    ///
    /// Both halves matter and they pull against each other. `from_mesh` used to hand shading
    /// the angle-weighted vertex pseudonormals — right for the sign test, wrong for a machined
    /// part: at a box's corner they average three perpendicular faces and point diagonally, and
    /// a box face is two triangles, so barycentric interpolation carried that across the whole
    /// face. It measured **45°** of tilt on a face whose true normal is constant.
    ///
    /// Flat shading would fix that and break the other case, which is why there is a crease
    /// angle rather than a switch: a tessellated cylinder is a genuinely curved surface and
    /// must not come out faceted. Both complaints have been made about this viewport.
    #[test]
    fn a_flat_face_shades_flat_and_a_tessellated_curve_still_shades_smooth() {
        // A box: every edge is 90 degrees, so every corner is a crease.
        let boxed = creased_box([40.0, 24.0, 8.0]);
        for i in 0..9 {
            let y = 12.0 * i as f64 / 10.0;
            let n = boxed
                .normal(Point3::new(0.0, y, 4.0))
                .expect("a mesh carries normals");
            let tilt = n.dot(&Vector3::z()).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(
                tilt < 1e-6,
                "the flat face tilted {tilt} degrees at y = {y}"
            );
        }

        // A cylinder tessellated at 64 sides: neighbours are 5.6 degrees apart, well inside the
        // crease angle, so the barrel must still interpolate rather than facet.
        let tube = tessellated_cylinder(20.0, 40.0, 64);
        let mut worst_step: f64 = 0.0;
        let mut previous: Option<Vector3<f64>> = None;
        for i in 0..64 {
            let a = std::f64::consts::TAU * i as f64 / 64.0;
            let p = Point3::new(20.0 * a.cos(), 20.0 * a.sin(), 0.0);
            let n = tube.normal(p).expect("a mesh carries normals");
            if let Some(prev) = previous {
                worst_step = worst_step.max(prev.dot(&n).clamp(-1.0, 1.0).acos().to_degrees());
            }
            previous = Some(n);
        }
        // A faceted barrel jumps by a whole facet between one facet and the next; a smooth one
        // turns by that much continuously, so consecutive samples never step further than the
        // sampling itself.
        assert!(
            worst_step < 8.0,
            "the barrel stepped {worst_step} degrees between samples, which is faceting",
        );
    }

    /// A cage keeps its own analytic normals, which the crease rule must not touch.
    ///
    /// The two doors into `MeshField` want opposite things: a subdivision surface is smooth
    /// everywhere and has no creases to preserve, so applying the mesh rule to it would replace
    /// exact normals with an average of facet normals — undoing the accuracy the limit surface
    /// was evaluated for.
    #[test]
    fn a_cage_keeps_its_analytic_normals_rather_than_creased_ones() {
        let cage = fieldform_cage::Cage::cube_of(40.0);
        let field = MeshField::from_cage(&cage, 4).expect("the cage builds");
        assert!(
            field.corner_normals.is_empty(),
            "a cage has no creases and must not carry corner normals",
        );
    }

    fn creased_box(size: [f64; 3]) -> MeshField {
        let h = Vector3::new(size[0], size[1], size[2]) * 0.5;
        let corners: Vec<Point3<f64>> = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        .iter()
        .map(|(x, y, z)| Point3::new(x * h.x, y * h.y, z * h.z))
        .collect();
        let faces: [[usize; 4]; 6] = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ];
        let mut triangles = Vec::new();
        for [a, b, c, d] in faces {
            triangles.push([a as u32, b as u32, c as u32]);
            triangles.push([a as u32, c as u32, d as u32]);
        }
        MeshField::from_mesh(corners, triangles).expect("a closed box")
    }

    /// A closed cylinder as triangles, which is what a curved CAD face arrives as.
    fn tessellated_cylinder(radius: f64, height: f64, sides: usize) -> MeshField {
        let h = height * 0.5;
        let mut positions = vec![Point3::new(0.0, 0.0, -h), Point3::new(0.0, 0.0, h)];
        for i in 0..sides {
            let a = std::f64::consts::TAU * i as f64 / sides as f64;
            positions.push(Point3::new(radius * a.cos(), radius * a.sin(), -h));
            positions.push(Point3::new(radius * a.cos(), radius * a.sin(), h));
        }
        let lo = |i: usize| (2 + 2 * (i % sides)) as u32;
        let hi = |i: usize| (3 + 2 * (i % sides)) as u32;
        let mut triangles = Vec::new();
        for i in 0..sides {
            triangles.push([lo(i), lo(i + 1), hi(i + 1)]);
            triangles.push([lo(i), hi(i + 1), hi(i)]);
            triangles.push([0, lo(i + 1), lo(i)]);
            triangles.push([1, hi(i), hi(i + 1)]);
        }
        MeshField::from_mesh(positions, triangles).expect("a closed cylinder")
    }
}
