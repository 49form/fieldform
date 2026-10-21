// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a point in space lands *on* a surface, in the surface's own coordinates.
//!
//! **The thing a conformal lattice needs and a distance field does not.** `MeshField` answers how
//! far a point is from a surface; this answers which face it is over, where in that face, and how
//! far off along the normal — `(face, s, t, w)`. Backlog item 46's map is those four numbers fed
//! to a cell function, so that the cells follow the surface instead of a box around it.
//!
//! It works on an **open** mesh, which is the whole reason item 45 came first. A panel or a shell
//! is exactly the open surface somebody wants reinforced, and it can never be a `Cage` because an
//! open surface has no inside for a sign to refer to. The offset here is signed by which side of
//! the surface the point is on rather than by enclosure, which is well defined everywhere except
//! on the surface itself and on the medial axis of the offset — the ordinary shell construction.

use fieldform_cage::{CageError, FaceId, QuadMesh};
use nalgebra::{Point3, Vector3};

use crate::bvh::Bvh;

/// Where a point landed on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    /// The face of the original quad mesh the closest point lies on.
    pub face: FaceId,
    /// Where in that face, in its own `(s, t)`, both in `[0, 1]`.
    ///
    /// **Per face, because there is no global grid over a quad mesh.** That is what a seamless
    /// parametrization would give and it is deliberately not what this is: each face carries its
    /// own square, and continuity across an edge comes from the meshing rather than from a shared
    /// coordinate. See the spec's §6.1.
    pub st: [f64; 2],
    /// How far off the surface, signed by which side. What a shell is banded about.
    pub offset: f64,
    /// The same displacement taken *along the surface's normal* — the lattice's third argument.
    ///
    /// **Equal to `offset` over the surface, and continuous where `offset` is not.** Over the
    /// interior the closest point is reached along the normal, so the two agree. Past the rim of
    /// an open surface the closest point is on the boundary, the displacement is no longer along
    /// the normal, and `offset`'s sign flips across the surface's extension while the point is
    /// still millimetres from the edge: the third argument jumped by twice that distance, the
    /// cells jumped with it, and the field read 3.9 mm of room a hundredth of a millimetre from
    /// material 8 mm deep. That was `gpu_matches_cpu`'s unexplained 8.35 mm — both sides had the
    /// same discontinuity, so they agreed to the last bit while the marcher stepped across it.
    /// The projection passes through zero there instead of jumping past it.
    pub along: f64,
    /// The limit surface's own normal where the closest point landed.
    pub normal: Vector3<f64>,
    /// How fast `s` changes with world position there, as a vector — `∇_world s`.
    ///
    /// **This is what a conformal map's `local_frequency` is made of.** A cell argument runs
    /// `2πN·s`, so its world gradient magnitude is `2πN·‖∇s‖`, and that is the number the
    /// thickness correction needs. A quad mesh's parametrization has shear, so `∇s` and `∇t`
    /// are not orthogonal and neither is aligned with anything convenient — which is exactly
    /// the case `cell-map-family.md` §1 shows only the row *norms* survive into the formula.
    ///
    /// Exact for the tessellation: the parametric coordinates are linear across a triangle, so
    /// their gradients are constant on it and come out of the corners in closed form.
    pub ds: Vector3<f64>,
    /// The same for `t` — `∇_world t`.
    pub dt: Vector3<f64>,
    /// The closest point itself.
    pub point: Point3<f64>,
    /// Which tessellated triangle answered.
    ///
    /// **Exposed because one of the answers above is discontinuous across a triangle edge and
    /// the rest are not.** `s`, `t`, the offset and the normal are all continuous — the corners
    /// agree along a shared edge — but `ds` and `dt` are piecewise constant by construction, so
    /// two triangles meeting at a point genuinely report different ones. Anything comparing this
    /// query against another implementation has to know which facet each answered from, or it is
    /// measuring that discontinuity and calling it an error.
    pub triangle: u32,
}

/// Everything a shader needs to ask a surface where a point sits, as two flat arrays.
///
/// **Two buffers rather than six, because a binding is not free and a layout that has to be
/// agreed twice is a layout that can be agreed differently.** Every float the query reads is in
/// `floats` and every index in `uints`, each surface's data laid end to end, and the offsets
/// below say where one surface's run begins. The shader is handed the same six numbers it would
/// have had as six bindings, as six integers instead.
///
/// The layout is stated here once and transcribed into WGSL exactly. `a_packed_surface_answers_
/// as_the_surface_does` walks these arrays with nothing else in scope, which is what makes the
/// transcription checkable before any of it reaches a GPU.
#[derive(Clone, Debug, Default)]
pub struct PackedSurface {
    /// Vertex positions (3 each), then vertex normals (3 each), then triangle `(s, t)` corners
    /// (6 each), then triangle corner gradients (18 each), then BVH node bounds (6 each) — in
    /// that order, at the offsets below.
    pub floats: Vec<f32>,
    /// Triangle vertex indices (3 each), then triangle face ids (1 each), then BVH node links
    /// (3 each), then the leaf order (1 each).
    pub uints: Vec<u32>,
    /// Where each run starts, in elements of its own array.
    pub layout: SurfaceLayout,
}

/// Where each run starts within a [`PackedSurface`], in elements.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceLayout {
    pub positions: u32,
    /// **The surface's own normals, not the triangles'.** A facet normal would be one fewer
    /// array to carry and it is the wrong vector twice over: it signs the offset differently
    /// from `SurfaceMap::at` wherever the surface curves, and a conformal lattice uses this
    /// direction as its third argument's gradient — so the shader would shade a normal the
    /// field does not have. Measured on a dome, the two disagree by up to 4.3°.
    pub normals: u32,
    pub uvs: u32,
    /// Three corners of `(∇s, ∇t)` per triangle: 18 floats, `∇s` then `∇t` per corner.
    pub gradients: u32,
    pub node_bounds: u32,
    pub triangles: u32,
    pub faces: u32,
    pub node_links: u32,
    pub order: u32,
    /// How many triangles, so a shader can bound its own loops.
    pub triangle_count: u32,
    /// [`SurfaceMap::worst_parametric_gradient`], carried across so a shader can bound its own
    /// stepping without a second buffer for one number.
    pub worst_gradient: f32,
}

/// The radius of the tightest fold in a quad mesh, in millimetres, read off the control mesh
/// itself — infinite where it is flat.
///
/// **The same question [`SurfaceMap::tightest_fold_mm`] answers, asked where there is no
/// tessellation to ask.** A panel showing a conformal lattice needs the number while the surface
/// is being dragged, and tessellating a large mesh per frame to get it is not affordable; two
/// quads sharing an edge turn by the same angle over the same span as the limit surface between
/// their middles, which is the same ratio. `the_tightest_fold_is_the_surfaces_own_radius` holds
/// the two together, because two ways of answering one question is how they come to disagree.
pub fn mesh_fold_mm(mesh: &QuadMesh) -> f64 {
    let centre = |face: fieldform_cage::FaceId| {
        let vs = mesh.face_vertices(face);
        let sum = vs
            .iter()
            .fold(Vector3::zeros(), |acc, v| acc + mesh.vertex_position(*v).coords);
        Point3::from(sum / vs.len() as f64)
    };
    // From its diagonals, which is right for a quad that is not planar — the same reading the
    // application's own picking takes.
    let normal = |face: fieldform_cage::FaceId| {
        let vs = mesh.face_vertices(face);
        let p: Vec<Point3<f64>> = vs.iter().map(|v| mesh.vertex_position(*v)).collect();
        (p[2] - p[0]).cross(&(p[3] - p[1])).try_normalize(1e-12)
    };
    let mut tightest = f64::INFINITY;
    for face in mesh.face_ids() {
        let Some(here) = normal(face) else { continue };
        for he in mesh.face_half_edges(face) {
            if mesh.is_boundary(he) {
                continue;
            }
            let across = mesh.half_edge(mesh.half_edge(he).twin).face;
            // Each shared edge is met from both sides; a minimum cannot notice.
            let Some(there) = normal(across) else { continue };
            let span = (centre(across) - centre(face)).norm();
            let turn = here.dot(&there).clamp(-1.0, 1.0).acos();
            if turn <= 1e-9 || span <= 0.0 {
                continue;
            }
            tightest = tightest.min(span / turn);
        }
    }
    tightest
}

/// The radius of the tightest fold in a tessellation carrying its own normals.
///
/// Each triangle edge is an arc: its ends' normals differ by `θ` over a chord of `s`, which is a
/// circle of radius `s/θ`. The smallest such radius anywhere is what bounds the offset a
/// conformal map stays single-valued over. Infinite for a flat surface, where no pair of normals
/// differs at all.
///
/// A shared edge is visited twice, which costs a little and cannot change a minimum.
fn tightest_fold(
    positions: &[Point3<f64>],
    normals: &[Vector3<f64>],
    triangles: &[[u32; 3]],
) -> f64 {
    let mut tightest = f64::INFINITY;
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k] as usize, t[(k + 1) % 3] as usize);
            let (Some(na), Some(nb)) = (normals.get(a), normals.get(b)) else {
                continue;
            };
            let chord = (positions[b] - positions[a]).norm();
            // The angle between two unit normals, guarded because a dot product can leave the
            // domain of `acos` by a rounding error and `NaN` would win every minimum.
            let turn = na.dot(nb).clamp(-1.0, 1.0).acos();
            // Below this the arc is straight as far as anything can tell, and dividing by it
            // reports a radius of `1e16` mm, which is the same answer as "flat" said badly.
            if turn <= 1e-9 || chord <= 0.0 {
                continue;
            }
            tightest = tightest.min(chord / turn);
        }
    }
    tightest
}

/// A surface that can be asked where a point sits on it.
pub struct SurfaceMap {
    positions: Vec<Point3<f64>>,
    normals: Vec<Vector3<f64>>,
    triangles: Vec<[u32; 3]>,
    /// Which face each triangle came from, and its corners in that face's own `(u, v)`.
    triangle_faces: Vec<FaceId>,
    triangle_uvs: Vec<[[f64; 2]; 3]>,
    /// Each triangle's corners' `(∇s, ∇t)` in world space, from the limit surface's own grid.
    ///
    /// **Interpolated rather than taken from the triangle**, which is not a refinement but the
    /// difference between a lattice that steps along every facet and one that does not: a
    /// convex surface puts a wedge of real volume nearest a shared *edge*, where two triangles'
    /// own gradients disagree — measured at 9.9% apart on a coarse dome, which is 9.9% of wall
    /// thickness appearing as a visible ledge.
    triangle_gradients: Vec<[[Vector3<f64>; 2]; 3]>,
    bvh: Bvh,
    /// The radius of the tightest fold anywhere on the surface, in millimetres, or infinity
    /// where it is flat. See [`SurfaceMap::tightest_fold_mm`].
    tightest_fold: f64,
    /// The largest `‖∇s‖` or `‖∇t‖` anywhere on the surface — the smallest face, in effect.
    ///
    /// **Wanted as a bound rather than as a fact about a point.** A tracer asks how fast the
    /// pattern can possibly vary so it knows how far it may safely step; asking at the point it
    /// is standing on would let it step into a finer region and through a wall. Computed once,
    /// because it is a property of the mesh and the mesh does not move.
    worst_gradient: f64,
}

/// By its size rather than its contents: a hundred thousand triangles in a debug line is
/// nobody's idea of help, and what a reader wants to know is which surface and how fine.
impl std::fmt::Debug for SurfaceMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceMap")
            .field("triangles", &self.triangles.len())
            .field("worst_gradient", &self.worst_gradient)
            .finish()
    }
}

impl SurfaceMap {
    /// Tessellates the limit surface of `mesh` and indexes it for closest-point queries.
    ///
    /// `levels` is the same subdivision-depth vocabulary the rest of the crate uses. It controls
    /// how closely the tessellation tracks the true limit surface, and so how accurately `(s, t)`
    /// is resolved between sample points — the normals are exact at every level, because the
    /// tessellation carries the surface's analytic ones rather than the triangles'.
    pub fn new(mesh: &QuadMesh, levels: usize) -> Result<Self, CageError> {
        let mut surface = fieldform_subd::LimitSurface::over(mesh);
        let fieldform_subd::Tessellation {
            positions,
            normals,
            triangles,
            triangle_faces,
            triangle_uvs,
            triangle_gradients,
        } = surface.tessellate(levels);
        let bvh = Bvh::build(&positions, &triangles);
        // The worst anywhere, over every corner rather than every facet — the bound has to hold
        // at the points the query can actually return, and those are interpolations of these.
        // An interpolation never exceeds the largest of what it blends, so a maximum over the
        // corners bounds the maximum over the surface.
        let worst_gradient = triangle_gradients
            .iter()
            .flatten()
            .map(|[ds, dt]| ds.norm().max(dt.norm()))
            .fold(0.0f64, f64::max);
        Ok(Self {
            tightest_fold: tightest_fold(&positions, &normals, &triangles),
            worst_gradient,
            positions,
            normals,
            triangles,
            triangle_faces,
            triangle_uvs,
            triangle_gradients,
            bvh,
        })
    }

    /// The radius of the tightest fold anywhere on this surface, in millimetres — infinite
    /// where it is flat.
    ///
    /// **What a conformal lattice may not reach past.** The map answers "where on the surface is
    /// this point nearest to", and that question has one answer only within the surface's own
    /// curvature radius: further in, on the inside of a fold, the nearest place is two places at
    /// once — measured at 1.0 apart in `(s, t)`, the maximum possible, by
    /// `examples/conformal_agreement.rs`. A shell reaching past this is not a thicker lattice,
    /// it is one whose cells cross themselves, and a mesh of it has holes exactly there.
    ///
    /// Read off the limit surface's own normals rather than the control mesh's corners: the
    /// tessellation carries the analytic normal at every sample, so a turn of `θ` over an arc of
    /// `s` is a radius of `s/θ` and nothing has to be differenced. Both directions of curvature
    /// count — a band is two-sided, so a fold either way is a fold.
    pub fn tightest_fold_mm(&self) -> f64 {
        self.tightest_fold
    }

    /// Where `p` lands on the surface.
    ///
    /// The closest point is found against the tessellation, and the face and `(s, t)` come from
    /// the triangle it landed on — barycentrically, because a triangle's three corners each carry
    /// the `(u, v)` of the face they were sampled from. That is why the tessellation records the
    /// pair per *triangle* rather than per vertex: a welded vertex belongs to several faces and
    /// has a different `(u, v)` in each, while a triangle belongs to exactly one.
    pub fn at(&self, p: Point3<f64>) -> SurfacePoint {
        let mut best = (f64::INFINITY, usize::MAX, Point3::origin(), [0.0; 3]);
        self.bvh.nearest(p, |i| {
            let i = i as usize;
            let t = &self.triangles[i];
            let (a, b, c) = (
                self.positions[t[0] as usize],
                self.positions[t[1] as usize],
                self.positions[t[2] as usize],
            );
            let (q, bary) = closest_point_with_barycentric(p, a, b, c);
            let d_sq = (q - p).norm_squared();
            // **A tie is broken by face id, not by whichever triangle the tree reached first.**
            // A point on the seam between two faces is genuinely equidistant from both, and the
            // order a BVH visits its leaves in is an accident of how it was built. Letting that
            // decide would return different cells for the same point on different runs, which
            // is the determinism D7 asks for -- and it would show as a lattice that reshuffles
            // along every shared edge when nothing about the model changed.
            let nearer = match d_sq.partial_cmp(&best.0) {
                Some(std::cmp::Ordering::Less) => true,
                Some(std::cmp::Ordering::Equal) if best.1 != usize::MAX => {
                    (self.triangle_faces[i].index(), i)
                        < (self.triangle_faces[best.1].index(), best.1)
                }
                Some(std::cmp::Ordering::Equal) => true,
                _ => false,
            };
            if nearer {
                best = (d_sq, i, q, bary);
            }
            d_sq
        });

        let (_, index, point, bary) = best;
        let t = &self.triangles[index];
        let uv = &self.triangle_uvs[index];
        let st = [
            uv[0][0] * bary[0] + uv[1][0] * bary[1] + uv[2][0] * bary[2],
            uv[0][1] * bary[0] + uv[1][1] * bary[1] + uv[2][1] * bary[2],
        ];
        let normal = (self.normals[t[0] as usize] * bary[0]
            + self.normals[t[1] as usize] * bary[1]
            + self.normals[t[2] as usize] * bary[2])
        .try_normalize(1e-12)
        .unwrap_or_else(Vector3::z);

        // **Signed by side, not by enclosure.** An open surface has no inside, so the sign is
        // which way off the surface the point is — `sign(dot(p − q, n(q)))`. On a closed mesh
        // this agrees with enclosure everywhere except across the medial axis, and a lattice is
        // banded about `|w|` rather than reading the sign directly, so that is the right trade
        // for the map. Anything wanting true enclosure asks `MeshField`, which is built for it.
        let away = p - point;
        let along = away.dot(&normal);
        let offset = away.norm() * if along < 0.0 { -1.0 } else { 1.0 };

        // **Interpolated, not taken from the triangle.** See `triangle_gradients` for what
        // that buys; the short version is that a facet gradient steps at every edge and a
        // convex surface puts real volume nearest one.
        let g = &self.triangle_gradients[index];
        let ds = g[0][0] * bary[0] + g[1][0] * bary[1] + g[2][0] * bary[2];
        let dt = g[0][1] * bary[0] + g[1][1] * bary[1] + g[2][1] * bary[2];

        SurfacePoint {
            ds,
            dt,
            triangle: index as u32,
            face: self.triangle_faces[index],
            st: [st[0].clamp(0.0, 1.0), st[1].clamp(0.0, 1.0)],
            offset,
            along,
            normal,
            point,
        }
    }

    /// The largest rate at which `s` or `t` changes anywhere on the surface, per millimetre.
    ///
    /// The bound a tracer needs: see the field it comes from.
    pub fn worst_parametric_gradient(&self) -> f64 {
        self.worst_gradient
    }

    /// The surface as a shader holds it. See [`PackedSurface`] for the layout.
    pub fn packed(&self) -> PackedSurface {
        let (node_bounds, node_links) = self.bvh.packed();
        let mut floats = Vec::new();
        let mut uints = Vec::new();

        let positions = 0;
        for p in &self.positions {
            floats.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
        }
        let normals_at = floats.len() as u32;
        for n in &self.normals {
            floats.extend_from_slice(&[n.x as f32, n.y as f32, n.z as f32]);
        }
        let uvs = floats.len() as u32;
        for uv in &self.triangle_uvs {
            for corner in uv {
                floats.extend_from_slice(&[corner[0] as f32, corner[1] as f32]);
            }
        }
        let gradients_at = floats.len() as u32;
        for corners in &self.triangle_gradients {
            for [ds, dt] in corners {
                floats.extend_from_slice(&[
                    ds.x as f32,
                    ds.y as f32,
                    ds.z as f32,
                    dt.x as f32,
                    dt.y as f32,
                    dt.z as f32,
                ]);
            }
        }
        let bounds_at = floats.len() as u32;
        floats.extend_from_slice(&node_bounds);

        let triangles = 0;
        for t in &self.triangles {
            uints.extend_from_slice(t);
        }
        let faces = uints.len() as u32;
        for f in &self.triangle_faces {
            uints.push(f.index() as u32);
        }
        let links_at = uints.len() as u32;
        uints.extend_from_slice(&node_links);
        let order_at = uints.len() as u32;
        uints.extend_from_slice(self.bvh.order());

        PackedSurface {
            floats,
            uints,
            layout: SurfaceLayout {
                positions,
                normals: normals_at,
                uvs,
                gradients: gradients_at,
                node_bounds: bounds_at,
                triangles,
                faces,
                node_links: links_at,
                order: order_at,
                triangle_count: self.triangles.len() as u32,
                worst_gradient: self.worst_gradient as f32,
            },
        }
    }

    /// How many triangles the surface was tessellated into — for cost measurements.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

/// The query a shader will run, in Rust, reading **only** the packed arrays.
///
/// **This exists to be transcribed, and to be checked before it is.** The WGSL version cannot be
/// stepped through or unit-tested in isolation; this one can, and
/// `a_packed_surface_answers_as_the_surface_does` holds it against `SurfaceMap::at`, which was
/// itself held against a hand-checkable panel. So a fault in the shader is a fault in the
/// transcription rather than in the algorithm, which is a much smaller thing to go looking for.
///
/// Returns everything the lattice reads, which is everything [`SurfacePoint`] carries except
/// the closest point itself.
pub fn query_packed(packed: &PackedSurface, p: Point3<f64>) -> PackedHit {
    let l = packed.layout;
    let float_at = |i: u32| packed.floats[i as usize] as f64;
    let point_at = |v: u32| {
        let base = l.positions + v * 3;
        Point3::new(float_at(base), float_at(base + 1), float_at(base + 2))
    };
    let corner = |tri: u32, k: u32| {
        let base = l.triangles + tri * 3 + k;
        packed.uints[base as usize]
    };
    let uv_at = |tri: u32, k: u32| {
        let base = l.uvs + tri * 6 + k * 2;
        [float_at(base), float_at(base + 1)]
    };
    let normal_at = |v: u32| {
        let base = l.normals + v * 3;
        Vector3::new(float_at(base), float_at(base + 1), float_at(base + 2))
    };
    // `(∇s, ∇t)` at one corner of one triangle: six floats, `∇s` first.
    let gradients_at = |tri: u32, k: u32| {
        let base = l.gradients + tri * 18 + k * 6;
        (
            Vector3::new(float_at(base), float_at(base + 1), float_at(base + 2)),
            Vector3::new(float_at(base + 3), float_at(base + 4), float_at(base + 5)),
        )
    };
    // Squared distance from `p` to a node's box, which is what prunes the walk.
    let box_distance = |node: u32| {
        let base = l.node_bounds + node * 6;
        let mut d = 0.0;
        for axis in 0..3 {
            let lo = float_at(base + axis);
            let hi = float_at(base + 3 + axis);
            let c = p.coords[axis as usize];
            let out = (lo - c).max(c - hi).max(0.0);
            d += out * out;
        }
        d
    };

    let mut best = f64::INFINITY;
    let mut best_tri = u32::MAX;
    let mut best_point = Point3::origin();
    let mut best_bary = [0.0; 3];

    // An explicit stack, bounded — a shader has no growable one, and a hierarchy over any mesh
    // this side of absurd is nowhere near this deep.
    let mut stack = [0u32; 64];
    let mut depth = 0usize;
    if !packed.uints.is_empty() && l.triangle_count > 0 {
        stack[0] = 0;
        depth = 1;
    }
    while depth > 0 {
        depth -= 1;
        let node = stack[depth];
        if box_distance(node) >= best {
            continue;
        }
        let links = (l.node_links + node * 3) as usize;
        let (first, count, right) = (
            packed.uints[links],
            packed.uints[links + 1],
            packed.uints[links + 2],
        );
        if count > 0 {
            for k in 0..count {
                let tri = packed.uints[(l.order + first + k) as usize];
                let (a, b, c) = (
                    point_at(corner(tri, 0)),
                    point_at(corner(tri, 1)),
                    point_at(corner(tri, 2)),
                );
                let (q, bary) = closest_point_with_barycentric(p, a, b, c);
                let d = (q - p).norm_squared();
                let nearer = d < best
                    || (d == best
                        && (best_tri == u32::MAX
                            || (packed.uints[(l.faces + tri) as usize], tri)
                                < (packed.uints[(l.faces + best_tri) as usize], best_tri)));
                if nearer {
                    best = d;
                    best_tri = tri;
                    best_point = q;
                    best_bary = bary;
                }
            }
            continue;
        }
        // The nearer child is pushed last so it is popped first, which tightens `best` early
        // and is what makes the test above actually prune.
        let (dl, dr) = (box_distance(first), box_distance(right));
        let (near, far) = if dl <= dr { (first, right) } else { (right, first) };
        if depth + 2 <= stack.len() {
            stack[depth] = far;
            stack[depth + 1] = near;
            depth += 2;
        }
    }

    if best_tri == u32::MAX {
        return PackedHit {
            st: [0.0; 2],
            offset: 0.0,
            along: 0.0,
            face: 0,
            triangle: 0,
            ds: Vector3::zeros(),
            dt: Vector3::zeros(),
            normal: Vector3::z(),
        };
    }

    let uv = [uv_at(best_tri, 0), uv_at(best_tri, 1), uv_at(best_tri, 2)];
    let s = uv[0][0] * best_bary[0] + uv[1][0] * best_bary[1] + uv[2][0] * best_bary[2];
    let t = uv[0][1] * best_bary[0] + uv[1][1] * best_bary[1] + uv[2][1] * best_bary[2];

    // **The surface's normal, interpolated, and not the triangle's.** A facet normal was the
    // first cut here and it is wrong in two places at once: it signs the offset differently
    // from `SurfaceMap::at` wherever the surface curves, and a conformal lattice carries this
    // direction into its gradient — so a shader using it would draw a normal the field does not
    // have, which is exactly what `gpu_matches_cpu` measures. A flat panel hides all of this,
    // which is why the test below curves.
    let normal = (normal_at(corner(best_tri, 0)) * best_bary[0]
        + normal_at(corner(best_tri, 1)) * best_bary[1]
        + normal_at(corner(best_tri, 2)) * best_bary[2])
    .try_normalize(1e-12)
    .unwrap_or_else(Vector3::z);
    let away = p - best_point;
    let along = away.dot(&normal);
    let offset = away.norm() * if along < 0.0 { -1.0 } else { 1.0 };
    let (ds0, dt0) = gradients_at(best_tri, 0);
    let (ds1, dt1) = gradients_at(best_tri, 1);
    let (ds2, dt2) = gradients_at(best_tri, 2);
    let ds = ds0 * best_bary[0] + ds1 * best_bary[1] + ds2 * best_bary[2];
    let dt = dt0 * best_bary[0] + dt1 * best_bary[1] + dt2 * best_bary[2];

    PackedHit {
        st: [s.clamp(0.0, 1.0), t.clamp(0.0, 1.0)],
        offset,
        along,
        face: packed.uints[(l.faces + best_tri) as usize],
        triangle: best_tri,
        ds,
        dt,
        normal,
    }
}

/// What [`query_packed`] answers — [`SurfacePoint`] by another name, with the face as a plain
/// integer because that is what a shader can hold.
///
/// **All six, not the four the first cut returned.** A conformal lattice reads `ds`, `dt` and the
/// normal as the world gradients of its three arguments, so leaving them out would have left the
/// only interesting half of the transcription unchecked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedHit {
    pub st: [f64; 2],
    pub offset: f64,
    /// See [`SurfacePoint::along`].
    pub along: f64,
    pub face: u32,
    /// Which tessellated triangle answered — see [`SurfacePoint::triangle`].
    pub triangle: u32,
    pub ds: Vector3<f64>,
    pub dt: Vector3<f64>,
    pub normal: Vector3<f64>,
}

/// Closest point on triangle `abc` to `p`, with the barycentric weights that produced it.
///
/// The same Voronoi-region method `sdf.rs` uses, returning the weights as well: the parametric
/// coordinates and the normal are both interpolated with them, so they have to come from the same
/// calculation that found the point rather than be recovered from it afterwards.
fn closest_point_with_barycentric(
    p: Point3<f64>,
    a: Point3<f64>,
    b: Point3<f64>,
    c: Point3<f64>,
) -> (Point3<f64>, [f64; 3]) {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(&ap), ac.dot(&ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }

    let bp = p - b;
    let (d3, d4) = (ab.dot(&bp), ac.dot(&bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }

    let cp = p - c;
    let (d5, d6) = (ab.dot(&cp), ac.dot(&cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [0.0, 1.0 - w, w]);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}

#[cfg(test)]
mod tests {
    /// A tube of `n` quads round, radius `r`, running `length` along Z — a surface whose
    /// curvature is known exactly everywhere.
    fn tube(n: usize, r: f64, length: f64) -> fieldform_cage::QuadMesh {
        let mut positions = Vec::new();
        for k in 0..=2 {
            let z = -length * 0.5 + length * k as f64 * 0.5;
            for i in 0..n {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                positions.push(Point3::new(r * a.cos(), r * a.sin(), z));
            }
        }
        let at = |ring: usize, i: usize| (ring * n + i % n) as u32;
        let mut faces = Vec::new();
        for ring in 0..2 {
            for i in 0..n {
                faces.push(vec![
                    at(ring, i),
                    at(ring, i + 1),
                    at(ring + 1, i + 1),
                    at(ring + 1, i),
                ]);
            }
        }
        fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a tube of quads")
    }

    /// **The tightest fold is a radius, and it is the radius the shape has.** A conformal
    /// lattice's shell may not reach past it — beyond that the nearest place on the surface is
    /// two places at once, and the panel now says so where the thickness is typed, using this
    /// number. So it has to be the real one: measured here against a tube, whose limit surface
    /// is a cylinder of a radius that can be worked out rather than eyeballed.
    ///
    /// Catmull-Clark pulls a ring of `n` points onto a circle a little smaller than the one they
    /// sit on, so the limit radius is below the control radius — which is why this brackets
    /// rather than equals. The point of the test is that it is a *radius*: proportional to the
    /// shape's own size, and nowhere near the face span the panel used to stand in for it.
    #[test]
    fn the_tightest_fold_is_the_surfaces_own_radius() {
        for r in [10.0f64, 40.0] {
            let map = SurfaceMap::new(&tube(16, r, r * 2.0), 3).expect("a tube tessellates");
            let fold = map.tightest_fold_mm();
            assert!(
                fold > r * 0.75 && fold < r * 1.1,
                "a tube of radius {r} folds at {fold} mm, which is not its own radius"
            );
        }

        // Twice as many faces round the same tube is the same surface, so the same radius —
        // the measure must not be reading the tessellation's own step.
        let coarse = SurfaceMap::new(&tube(16, 10.0, 20.0), 3).expect("tessellates");
        let fine = SurfaceMap::new(&tube(32, 10.0, 20.0), 3).expect("tessellates");
        let (a, b) = (coarse.tightest_fold_mm(), fine.tightest_fold_mm());
        assert!(
            (a - b).abs() < 0.15 * a.max(b),
            "the same tube measured {a} mm at 16 faces and {b} mm at 32"
        );

        // **The two measures answer the same question**, which is the only thing that makes it
        // safe to have both: the panel reads the control mesh because it cannot tessellate per
        // frame, and the kernel reads the limit surface because that is what the map uses.
        for r in [10.0f64, 40.0] {
            let mesh = tube(16, r, r * 2.0);
            let over_mesh = crate::mesh_fold_mm(&mesh);
            let over_limit = SurfaceMap::new(&mesh, 3).expect("tessellates").tightest_fold_mm();
            assert!(
                (over_mesh - over_limit).abs() < 0.25 * over_limit,
                "a tube of radius {r}: the control mesh says {over_mesh} mm, the limit surface \
                 says {over_limit} mm"
            );
        }
        assert!(crate::mesh_fold_mm(&panel(4, 40.0)) > 1e6, "a flat panel has no fold");

        // And a flat panel folds nowhere, so no shell can reach past it.
        let flat = SurfaceMap::new(&panel(4, 40.0), 3).expect("a panel tessellates");
        assert!(
            flat.tightest_fold_mm() > 1e6,
            "a flat panel has no fold, got {} mm",
            flat.tightest_fold_mm()
        );
    }
    use super::*;

    /// A flat grid of `n` by `n` quads in the XY plane, `size` across, open all the way round.
    fn panel(n: usize, size: f64) -> QuadMesh {
        let step = size / n as f64;
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                positions.push(Point3::new(i as f64 * step, j as f64 * step, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let mut faces = Vec::new();
        for j in 0..n {
            for i in 0..n {
                faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        QuadMesh::from_faces(positions, &faces).expect("a flat panel")
    }

    /// The same panel bent into a dome, `rise` millimetres at the middle.
    ///
    /// **A flat panel cannot tell a facet normal from the surface's own**, and that difference is
    /// the whole of what the packed query has to get right beyond finding the triangle. So the
    /// transcription is checked on something curved.
    fn dome(n: usize, size: f64, rise: f64) -> QuadMesh {
        let step = size / n as f64;
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let (x, y) = (i as f64 * step, j as f64 * step);
                let (u, v) = (x / size * 2.0 - 1.0, y / size * 2.0 - 1.0);
                positions.push(Point3::new(x, y, rise * (1.0 - u * u) * (1.0 - v * v)));
            }
        }
        let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let mut faces = Vec::new();
        for j in 0..n {
            for i in 0..n {
                faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        QuadMesh::from_faces(positions, &faces).expect("a domed panel")
    }

    /// **An open mesh can be asked where a point sits on it**, which a `Cage` could never be
    /// asked because an open surface cannot be one. This is what items 45 and 46 are for.
    #[test]
    fn a_point_above_an_open_panel_lands_directly_below_itself() {
        let size = 4.0;
        let map = SurfaceMap::new(&panel(4, size), 3).expect("the panel tessellates");

        for (x, y, height) in [(1.0, 1.0, 2.0), (2.5, 0.75, -1.5), (3.2, 3.4, 0.25)] {
            let hit = map.at(Point3::new(x, y, height));
            assert!(
                (hit.point - Point3::new(x, y, 0.0)).norm() < 1e-6,
                "a point over a flat panel should land under itself, got {:?}",
                hit.point
            );
            assert!(
                (hit.offset - height).abs() < 1e-6,
                "the offset should be the height, signed: wanted {height}, got {}",
                hit.offset
            );
            assert!(hit.normal.z.abs() > 0.999, "a flat panel faces along Z");
        }
    }

    /// **The face and `(s, t)` name where on the surface it landed**, which is the whole of what
    /// distinguishes this from a distance query. A unit-step panel makes the answer checkable:
    /// the point over `(2.25, 1.75)` is three quarters across face `(2, 1)` and a quarter up it.
    #[test]
    fn the_map_says_which_face_and_where_in_it() {
        let map = SurfaceMap::new(&panel(4, 4.0), 4).expect("the panel tessellates");

        let hit = map.at(Point3::new(2.25, 1.75, 1.0));
        assert_eq!(
            hit.face,
            FaceId::from_index(1 * 4 + 2),
            "column 2, row 1 of a four-by-four panel"
        );
        assert!(
            (hit.st[0] - 0.25).abs() < 0.02 && (hit.st[1] - 0.75).abs() < 0.02,
            "wanted about (0.25, 0.75) within the face, got {:?}",
            hit.st
        );

        // Every face gets asked, and each answers with itself: a point over the middle of each
        // face lands in that face at its centre.
        for j in 0..4 {
            for i in 0..4 {
                let p = Point3::new(i as f64 + 0.5, j as f64 + 0.5, 0.5);
                let hit = map.at(p);
                assert_eq!(hit.face, FaceId::from_index(j * 4 + i), "over face ({i}, {j})");
                assert!(
                    (hit.st[0] - 0.5).abs() < 0.02 && (hit.st[1] - 0.5).abs() < 0.02,
                    "the middle of a face should be its middle, got {:?}",
                    hit.st
                );
            }
        }
    }

    /// **Past the edge, the closest point is on the edge** — the panel does not extend to meet
    /// it. The offset then measures to the boundary rather than along the normal, which is what
    /// makes the shell construction well defined off the end of an open surface.
    #[test]
    fn a_point_beyond_the_panel_lands_on_its_boundary() {
        let size = 4.0;
        let map = SurfaceMap::new(&panel(4, size), 3).expect("the panel tessellates");

        let hit = map.at(Point3::new(-2.0, 2.0, 0.0));
        assert!(
            hit.point.x.abs() < 1e-6,
            "the closest point should be on the x = 0 edge, got {:?}",
            hit.point
        );
        assert!(
            (hit.offset.abs() - 2.0).abs() < 1e-6,
            "two millimetres from the edge, got {}",
            hit.offset
        );
        assert!(
            hit.st[0] < 0.02,
            "and at the start of its face in s, got {:?}",
            hit.st
        );
    }

    /// **`s` and `t` advance at the rate the faces are sized at**, which is what makes cell
    /// size in millimetres follow quad size — the mesh *is* the grading.
    ///
    /// On a panel whose faces are one millimetre across, `s` runs nought to one over that
    /// millimetre, so `‖∇s‖` is one per millimetre. Double the panel and halve it.
    #[test]
    fn the_parametric_gradient_is_one_over_the_face_size() {
        for (n, size) in [(4usize, 4.0f64), (4, 8.0), (2, 1.0)] {
            let face_size = size / n as f64;
            let map = SurfaceMap::new(&panel(n, size), 3).expect("the panel tessellates");
            let hit = map.at(Point3::new(size * 0.5, size * 0.5, 1.0));
            let wanted = 1.0 / face_size;
            assert!(
                (hit.ds.norm() - wanted).abs() < 1e-6,
                "{n} faces over {size} mm: wanted {wanted} per mm in s, got {}",
                hit.ds.norm()
            );
            assert!(
                (hit.dt.norm() - wanted).abs() < 1e-6,
                "and in t, got {}",
                hit.dt.norm()
            );
            // Both lie in the surface: moving off it does not change where on it you are.
            assert!(hit.ds.z.abs() < 1e-9 && hit.dt.z.abs() < 1e-9);
        }
    }

    /// **The packed arrays answer what the surface answers.**
    ///
    /// The whole point of `query_packed`: it reads nothing but the two flat arrays a shader
    /// will hold, so holding it against `SurfaceMap::at` checks the *layout* -- every offset,
    /// every stride, the BVH links and the leaf order -- on the CPU, where a wrong index is a
    /// panic or a visible number rather than a silently wrong picture.
    ///
    /// A fault in the shader is then a fault in the transcription, which is a much smaller
    /// thing to go looking for than a fault somewhere in an algorithm and a layout together.
    #[test]
    fn a_packed_surface_answers_as_the_surface_does() {
        // Flat and curved both. The flat ones check the walk; the domes check everything the
        // walk carries out of it, since a facet normal and the surface's own agree exactly on a
        // plane and only there.
        let meshes = [
            ("flat 4x4 over 4mm", panel(4, 4.0), 4.0),
            ("flat 3x3 over 9mm", panel(3, 9.0), 9.0),
            ("one face over 2mm", panel(1, 2.0), 2.0),
            ("domed 4x4, 1.5mm rise", dome(4, 4.0, 1.5), 4.0),
            ("domed 3x3, 3mm rise", dome(3, 9.0, 3.0), 9.0),
        ];
        for (name, mesh, size) in meshes {
            let map = SurfaceMap::new(&mesh, 3).expect("tessellates");
            let packed = map.packed();

            let mut worst_st = 0.0f64;
            let mut worst_offset = 0.0f64;
            let mut worst_normal = 0.0f64;
            let mut worst_gradient = 0.0f64;
            let mut split = 0;
            let mut checked = 0;
            for i in 0..400 {
                let t = f64::from(i) / 400.0;
                // Over the panel, past its edges, and both sides of it.
                let p = Point3::new(
                    -size * 0.2 + t * size * 1.4,
                    size * (t * 7.0).fract(),
                    size * 0.4 * (t * 11.0).cos(),
                );
                let hit = map.at(p);
                let packed_hit = query_packed(&packed, p);

                assert_eq!(
                    packed_hit.face,
                    hit.face.index() as u32,
                    "{name}: different face at {p:?}"
                );
                worst_st = worst_st
                    .max((packed_hit.st[0] - hit.st[0]).abs())
                    .max((packed_hit.st[1] - hit.st[1]).abs());
                worst_offset = worst_offset.max((packed_hit.offset - hit.offset).abs());
                worst_normal = worst_normal
                    .max(packed_hit.normal.angle(&hit.normal).to_degrees());
                // **Only where both answered from the same facet.** `ds` and `dt` are piecewise
                // constant across the tessellation, so two triangles meeting at an edge report
                // genuinely different ones -- measured at 9.9% apart on the coarse dome. A point
                // whose closest point lands on a shared edge is equidistant from both, and the
                // packed arrays hold positions in `f32`, so the two implementations can land on
                // either side of a tie that is exact in neither. That is a property of a
                // tessellated parametrization and not a transcription error; counting it as one
                // would mean loosening this to 10%, which would check nothing.
                if packed_hit.triangle == hit.triangle {
                    // Relative, because `‖∇s‖` is `1/face size` and those differ by a factor of
                    // several across these meshes.
                    let scale = hit.ds.norm().max(hit.dt.norm()).max(1e-9);
                    worst_gradient = worst_gradient
                        .max((packed_hit.ds - hit.ds).norm() / scale)
                        .max((packed_hit.dt - hit.dt).norm() / scale);
                } else {
                    split += 1;
                }
                checked += 1;
            }
            assert!(checked > 0);
            assert!(
                worst_st < 1e-6,
                "{name}: the packed parametric answer differs by {worst_st}"
            );
            assert!(worst_offset < 1e-6, "{name}: and the offset by {worst_offset} mm");
            assert!(worst_normal < 1e-3, "{name}: and the normal by {worst_normal} degrees");
            assert!(
                worst_gradient < 1e-4,
                "{name}: and the parametric gradients by {worst_gradient} relative"
            );
            // And the ties themselves stay rare. If the two walks were choosing differently for
            // any reason other than an edge tie, this is where it would show -- a transcription
            // that visited the tree in a different order would split on a large fraction rather
            // than on the handful of samples that land exactly on a shared edge.
            assert!(
                split * 50 < checked,
                "{name}: the two walks answered from different triangles {split} times in \
                 {checked}, which is too often to be edge ties"
            );
        }
    }

    /// **A point equidistant from two faces picks the same one every time.**
    ///
    /// D7: the same model must give the same geometry. A point on the seam between two faces is
    /// genuinely equidistant from both, and the order a BVH reaches its leaves in is an accident
    /// of how it was built -- so the tie is broken by face id instead. Without that, the lattice
    /// would reshuffle along every shared edge between one run and the next with nothing about
    /// the model having changed.
    #[test]
    fn a_point_on_a_seam_picks_the_same_face_every_time() {
        let map = SurfaceMap::new(&panel(4, 4.0), 3).expect("the panel tessellates");
        // Exactly on the edge between face (0, 0) and face (1, 0), and above it.
        let seam = Point3::new(1.0, 0.5, 1.0);
        let first = map.at(seam);
        for _ in 0..32 {
            let again = map.at(seam);
            assert_eq!(again.face, first.face, "the seam changed its mind about the face");
            assert_eq!(again.st, first.st, "and about where in it");
        }
        // Rebuilt from scratch, which is the case that a BVH visit order could differ in.
        let rebuilt = SurfaceMap::new(&panel(4, 4.0), 3).expect("the panel tessellates");
        assert_eq!(rebuilt.at(seam).face, first.face, "a rebuild changed the answer");
    }

    /// A closed cage answers too, by the same route — the map does not care which it has, and
    /// a cube's faces are where a cube's faces are.
    #[test]
    fn a_closed_cage_is_a_surface_like_any_other() {
        let cube = fieldform_cage::Cage::cube_of(20.0);
        let map = SurfaceMap::new(cube.mesh(), 3).expect("a cube tessellates");

        // Well outside, along +X: the closest point is on the +X side and the offset is
        // positive, because the normal there points the way the point is.
        let hit = map.at(Point3::new(40.0, 0.0, 0.0));
        assert!(hit.offset > 0.0, "outside should read positive, got {}", hit.offset);
        assert!(hit.point.x > 0.0, "and land on the +X side, got {:?}", hit.point);

        // Inside, at the centre: every side is equally close, so only the sign is asserted.
        let inside = map.at(Point3::origin());
        assert!(
            inside.offset < 0.0,
            "inside should read negative, got {}",
            inside.offset
        );
    }
}
