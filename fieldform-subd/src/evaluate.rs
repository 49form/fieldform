// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Evaluation of the Catmull-Clark limit surface at arbitrary `(u, v)` on a face.
//!
//! This replaces "subdivide a fixed number of times and treat the result as the surface" with
//! evaluation of the actual limit surface, so accuracy stops being a hidden constant.
//!
//! # How it works
//!
//! Away from extraordinary vertices the Catmull-Clark limit surface **is** a uniform bicubic
//! B-spline patch over the face's 4x4 control neighbourhood. Those points are evaluated in
//! closed form, exactly — no iteration, no approximation.
//!
//! Near an extraordinary vertex (valence != 4) there is no single bicubic patch. The classical
//! answer is Stam's eigenbasis method, which reaches the limit in O(1) by diagonalising the
//! subdivision matrix. This module instead **subdivides and descends**: subdividing once splits
//! the face into four, and only the sub-face touching the extraordinary vertex is still
//! irregular, so following `(u, v)` into its sub-face resolves to a regular patch in a few
//! steps for all but a vanishing neighbourhood of the vertex itself.
//!
//! The subdivision is **local**: only the neighbourhood around an irregular face is extracted
//! and refined (see [`Patch`]), never the whole cage, so descent costs track the handful of
//! extraordinary vertices in a model rather than its size.
//!
//! That is the same limit Stam's method computes, reached iteratively rather than in closed
//! form. It is chosen deliberately: it reuses the already-tested [`subdivide`] rules, needs no
//! eigensolver, and avoids the defective-matrix special cases that make direct eigenbasis
//! implementations delicate. The cost is `O(depth)` per query instead of `O(1)`, and depth is
//! 2-3 for the overwhelming majority of the domain. If profiling ever demands it, Stam's method
//! is a drop-in replacement for this one function.
//!
//! # Accuracy, stated rather than implied
//!
//! - **Regular patches: exact** to floating-point precision.
//! - **Within `2^-max_descent` of an extraordinary vertex in parameter space:** evaluation
//!   returns that vertex's limit position, which is itself exact in closed form (see
//!   [`limit_position`]). The error is therefore bounded by how far the surface moves across
//!   that final, geometrically tiny patch, and shrinks by roughly half per extra level.
//!
//! Both are bounded and documented, which is the point — the previous behaviour had an
//! accuracy floor set by a hardcoded subdivision level that nothing justified.

use crate::patch::{Patch, DROPPED};
use fieldform_cage::{Cage, CageError, FaceId, HalfEdgeId, QuadMesh, VertexId};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// Default descent limit before snapping to the extraordinary vertex's exact limit position.
///
/// Three is not a guess. Descent is only needed to resolve *irregular* faces, and a sample at
/// parameter distance `d` from the extraordinary corner resolves in about `log2(1/d)` levels.
/// For an `n x n` tessellation the closest non-corner sample sits at `1/(n-1)`, so depth
/// `ceil(log2(n-1))` covers every sample — and the one sample that never resolves is the
/// corner itself, whose limit position is known in closed form. Depth 3 therefore evaluates
/// an 8x8 patch grid exactly.
///
/// Depth used to be ruinous: descent subdivided the **whole cage** per level, so each extra
/// level cost 4x across the entire model — on a 24k-face cage, depth 6 took over five minutes
/// to warm up where depth 3 took under a second, for identical results at the resolutions
/// actually used. Descent is local now (see [`Patch`]), so an extra level costs 4x on the
/// irregular *neighbourhoods* instead, and raising this is affordable. Three is still the
/// right default because it is what the tessellation resolutions need, not because deeper is
/// unaffordable. See [`LimitSurface::with_max_descent_over`].
const DEFAULT_MAX_DESCENT: usize = 3;

/// Uniform cubic B-spline basis functions at `t`.
fn bspline_basis(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    [
        mt * mt * mt / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Derivatives of [`bspline_basis`], for exact surface tangents.
fn bspline_deriv(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let mt = 1.0 - t;
    [
        -mt * mt / 2.0,
        (3.0 * t2 - 4.0 * t) / 2.0,
        (-3.0 * t2 + 2.0 * t + 1.0) / 2.0,
        t2 / 2.0,
    ]
}

/// Evaluates a bicubic B-spline patch. `grid[i][j]` is indexed by `u` then `v`.
#[cfg_attr(not(test), allow(dead_code))]
fn eval_bicubic(grid: &[[Point3<f64>; 4]; 4], u: f64, v: f64) -> Point3<f64> {
    let bu = bspline_basis(u);
    let bv = bspline_basis(v);
    let mut acc = Vector3::zeros();
    for (i, &wu) in bu.iter().enumerate() {
        for (j, &wv) in bv.iter().enumerate() {
            acc += grid[i][j].coords * (wu * wv);
        }
    }
    Point3::from(acc)
}

/// Evaluates a patch along with its exact unit normal, from the analytic tangents.
///
/// Deriving the normal from the surface rather than by averaging adjacent triangle normals
/// matters for per-face tessellation: each face is meshed independently, so averaged normals
/// would disagree across a shared edge and leave a visible seam. Analytic normals agree by
/// construction because they depend only on the surface, not on the tessellation.
fn eval_bicubic_with_normal(
    grid: &[[Point3<f64>; 4]; 4],
    u: f64,
    v: f64,
) -> (Point3<f64>, Vector3<f64>) {
    let bu = bspline_basis(u);
    let bv = bspline_basis(v);
    let du = bspline_deriv(u);
    let dv = bspline_deriv(v);

    let mut pos = Vector3::zeros();
    let mut tan_u = Vector3::zeros();
    let mut tan_v = Vector3::zeros();
    for i in 0..4 {
        for j in 0..4 {
            let p = grid[i][j].coords;
            pos += p * (bu[i] * bv[j]);
            tan_u += p * (du[i] * bv[j]);
            tan_v += p * (bu[i] * dv[j]);
        }
    }

    // A degenerate tangent frame can occur only where the surface itself is degenerate; fall
    // back to +Z rather than emitting a NaN normal that would poison shading downstream.
    let n = tan_u
        .cross(&tan_v)
        .try_normalize(1e-12)
        .unwrap_or_else(Vector3::z);
    (Point3::from(pos), n)
}

/// The traversal `regular_grid` needs, shared by the closed [`Cage`] and the open
/// neighbourhood patches that local descent subdivides.
///
/// It exists so the control-grid walk below is written **once**. That walk is orientation
/// sensitive — it rotates around corners and steps across edges — and getting it wrong is
/// silent: the surface stays smooth while sampling the wrong place. A second copy for
/// patches would be a second chance to get it wrong, so patches implement this instead.
///
/// The difference from `Cage` is that a patch has a boundary, so `twin` and `valence` can
/// fail. Both return `Option`, and `regular_grid` propagates: a face whose control grid runs
/// off the edge of the patch is simply "not regular here", which is exactly how the caller
/// should treat it.
pub(crate) trait QuadTopology {
    fn face_vertices(&self, face: FaceId) -> [VertexId; 4];
    fn face_half_edges(&self, face: FaceId) -> [HalfEdgeId; 4];
    fn origin(&self, he: HalfEdgeId) -> VertexId;
    fn next(&self, he: HalfEdgeId) -> HalfEdgeId;
    /// `None` where the adjacent face is outside the patch.
    fn twin(&self, he: HalfEdgeId) -> Option<HalfEdgeId>;
    /// Whether a missing twin is the **mesh** ending rather than the patch being cut.
    ///
    /// **The two look identical to [`Self::twin`] and must not be treated alike.** A patch is a
    /// neighbourhood lifted out of a larger mesh, so a half-edge with no twin there has a face
    /// on the other side in the mesh it came from — completing its grid by reflection would
    /// invent a surface where a real one already exists. A half-edge on an open mesh's boundary
    /// has nothing on the other side at all, and reflection is the whole rule for it.
    ///
    /// `false` by default, which is the safe half: a mesh that does not distinguish them is
    /// treated as having no boundary, and a face whose grid runs off it is refused exactly as
    /// it was before this existed.
    fn on_boundary(&self, _he: HalfEdgeId) -> bool {
        false
    }
    /// Whether this vertex lies on the mesh's boundary, by the same distinction.
    fn vertex_on_boundary(&self, _v: VertexId) -> bool {
        false
    }
    /// `None` where the vertex's ring is incomplete, so its valence is unknowable from the
    /// patch alone. Never `None` for a `Cage`, which is closed by construction.
    fn valence(&self, v: VertexId) -> Option<usize>;
    fn vertex_position(&self, v: VertexId) -> Point3<f64>;
    fn adjacent_faces(&self, v: VertexId) -> Vec<FaceId>;
    fn adjacent_vertices(&self, v: VertexId) -> Vec<VertexId> {
        // Every mesh here is all-quad, so a vertex's edge-neighbours are exactly the two
        // corners flanking it in each incident face.
        let mut out = Vec::new();
        for f in self.adjacent_faces(v) {
            let corners = self.face_vertices(f);
            let k = corners
                .iter()
                .position(|&c| c == v)
                .expect("an incident face contains this vertex");
            for n in [corners[(k + 1) % 4], corners[(k + 3) % 4]] {
                if !out.contains(&n) {
                    out.push(n);
                }
            }
        }
        out
    }
}

/// **The open mesh answers for itself, and a cage answers through it.**
///
/// Written here rather than only for `Cage` because a boundary is the whole point of the open
/// case: `on_boundary` is what tells `regular_grid` that a missing twin is the mesh ending
/// rather than a patch being cut, and only a real mesh knows the difference.
impl QuadTopology for QuadMesh {
    fn face_vertices(&self, face: FaceId) -> [VertexId; 4] {
        QuadMesh::face_vertices(self, face)
    }
    fn face_half_edges(&self, face: FaceId) -> [HalfEdgeId; 4] {
        QuadMesh::face_half_edges(self, face)
    }
    fn origin(&self, he: HalfEdgeId) -> VertexId {
        self.half_edge(he).origin
    }
    fn next(&self, he: HalfEdgeId) -> HalfEdgeId {
        self.half_edge(he).next
    }
    /// `None` on the boundary, where a self-twin means there is nothing on the other side —
    /// which `on_boundary` below is what distinguishes from a patch's cut edge.
    fn twin(&self, he: HalfEdgeId) -> Option<HalfEdgeId> {
        (!self.is_boundary(he)).then(|| self.half_edge(he).twin)
    }
    fn on_boundary(&self, he: HalfEdgeId) -> bool {
        self.is_boundary(he)
    }
    fn vertex_on_boundary(&self, v: VertexId) -> bool {
        QuadMesh::vertex_on_boundary(self, v)
    }
    fn valence(&self, v: VertexId) -> Option<usize> {
        Some(QuadMesh::valence(self, v))
    }
    fn vertex_position(&self, v: VertexId) -> Point3<f64> {
        QuadMesh::vertex_position(self, v)
    }
    fn adjacent_faces(&self, v: VertexId) -> Vec<FaceId> {
        QuadMesh::adjacent_faces(self, v)
    }
    fn adjacent_vertices(&self, v: VertexId) -> Vec<VertexId> {
        QuadMesh::adjacent_vertices(self, v)
    }
}

impl QuadTopology for Cage {
    // **Every one of these goes through `mesh()`, and that is not stylistic.** They used to be
    // spelled `Cage::face_vertices(self, face)`, which disambiguated the inherent method from
    // the trait method of the same name while the two lived on the same type. The inherent ones
    // are on `QuadMesh` now, so that spelling resolves to the *trait* method — this one — and
    // every line here becomes an infinite recursion. The compiler caught exactly one of them,
    // `valence`, and only because its signature differs by an `Option`.
    fn face_vertices(&self, face: FaceId) -> [VertexId; 4] {
        self.mesh().face_vertices(face)
    }
    fn face_half_edges(&self, face: FaceId) -> [HalfEdgeId; 4] {
        self.mesh().face_half_edges(face)
    }
    fn origin(&self, he: HalfEdgeId) -> VertexId {
        self.mesh().half_edge(he).origin
    }
    fn next(&self, he: HalfEdgeId) -> HalfEdgeId {
        self.mesh().half_edge(he).next
    }
    /// Never `None`: a cage is closed, so every half-edge has a face on the other side.
    fn twin(&self, he: HalfEdgeId) -> Option<HalfEdgeId> {
        Some(self.mesh().half_edge(he).twin)
    }
    fn valence(&self, v: VertexId) -> Option<usize> {
        Some(self.mesh().valence(v))
    }
    fn vertex_position(&self, v: VertexId) -> Point3<f64> {
        self.mesh().vertex_position(v)
    }
    fn adjacent_faces(&self, v: VertexId) -> Vec<FaceId> {
        self.mesh().adjacent_faces(v)
    }
    fn adjacent_vertices(&self, v: VertexId) -> Vec<VertexId> {
        self.mesh().adjacent_vertices(v)
    }
}

/// Gathers the 4x4 control grid for a face, or `None` if no single bicubic patch describes
/// it — either because a corner is extraordinary, or, on a patch, because the grid reaches
/// outside the extracted neighbourhood.
///
/// Layout: the face's own corners sit at `[1][1] [2][1] [2][2] [1][2]`, so `(u, v) = (0, 0)`
/// is corner 0, `u` runs toward corner 1 and `v` toward corner 3. The surrounding ring comes
/// from the neighbouring faces.
///
/// # A face on the mesh's boundary
///
/// It has no surrounding ring on the open side, and the rule is the standard one: **a boundary
/// loop is a cubic B-spline curve**, got by reflecting the row two in from the boundary through
/// the boundary row itself — `P₋₁ = 2·P₀ − P₁`. A boundary face then has a full 4×4 grid and is
/// an ordinary regular patch; this is exact rather than an approximation.
///
/// **Why this rule and not another.** The limit surface *interpolates* the boundary curve, so
/// the mesh's edge is where the surface's edge lands. The alternatives pull inward — a partial
/// grid evaluated anyway shrinks the surface by roughly a third of a face, looks entirely
/// plausible, and passes every topological test. For a reinforcement feature, where the panel
/// edge is a real edge somebody drew, that is the difference between a lattice on the panel and
/// a lattice near it. `the_limit_surface_meets_the_boundary_it_was_drawn_on` is the check, and
/// it is geometric because no structural test can see the difference.
///
/// A mesh **corner** — one face, two boundary edges — gets the reflection twice, once each way,
/// which lands on `4·P₁₁ − 2·P₂₁ − 2·P₁₂ + P₂₂` and interpolates the corner point exactly.
/// That falls out of applying the edge rule in both directions rather than being a third case.
pub(crate) fn regular_grid<M: QuadTopology + ?Sized>(
    mesh: &M,
    face: FaceId,
) -> Option<[[Point3<f64>; 4]; 4]> {
    let corners = mesh.face_vertices(face);
    // **Regular means regular for the kind of vertex it is.** An interior vertex needs four
    // faces; a vertex on the boundary needs two, which is what makes its reflected fan regular,
    // and a mesh corner has one. Anything else is extraordinary and descends instead.
    let regular = |v: VertexId| match mesh.valence(v) {
        Some(4) => true,
        Some(1) | Some(2) => mesh.vertex_on_boundary(v),
        _ => false,
    };
    if !corners.iter().copied().all(regular) {
        return None;
    }

    let h = mesh.face_half_edges(face);
    let pos = |v: VertexId| mesh.vertex_position(v);

    // The two outer vertices across the edge `he`, returned as
    // (adjacent to he's origin, adjacent to he's destination).
    let across = |he: HalfEdgeId| -> Option<(VertexId, VertexId)> {
        let t = mesh.twin(he)?;
        let n1 = mesh.next(t); // origin = he's origin
        let n2 = mesh.next(n1);
        let n3 = mesh.next(n2);
        Some((mesh.origin(n2), mesh.origin(n3)))
    };

    // `next(twin(he))` is the next half-edge sharing he's origin, so rotating twice lands on
    // the face diagonally opposite this one at that corner.
    let rotate = |he: HalfEdgeId| -> Option<HalfEdgeId> { Some(mesh.next(mesh.twin(he)?)) };
    let diagonal = |he: HalfEdgeId| -> Option<VertexId> {
        let r = rotate(rotate(he)?)?;
        let n1 = mesh.next(r);
        let n2 = mesh.next(n1);
        Some(mesh.origin(n2))
    };

    let mut g: [[Option<Point3<f64>>; 4]; 4] = [[None; 4]; 4];

    g[1][1] = Some(pos(corners[0]));
    g[2][1] = Some(pos(corners[1]));
    g[2][2] = Some(pos(corners[2]));
    g[1][2] = Some(pos(corners[3]));

    // Each edge of the face brings in the two points beyond it. A miss is only allowed where
    // the mesh itself ends; a patch running out of neighbourhood still refuses, because the
    // face it cannot see is really there.
    let edge = |slots: [(usize, usize); 2], he: HalfEdgeId, g: &mut [[Option<Point3<f64>>; 4]; 4]| -> Option<()> {
        match across(he) {
            Some((a, b)) => {
                g[slots[0].0][slots[0].1] = Some(pos(a));
                g[slots[1].0][slots[1].1] = Some(pos(b));
                Some(())
            }
            None => mesh.on_boundary(he).then_some(()),
        }
    };
    edge([(1, 0), (2, 0)], h[0], &mut g)?; // v = 0 edge
    edge([(3, 1), (3, 2)], h[1], &mut g)?; // u = 1 edge
    edge([(2, 3), (1, 3)], h[2], &mut g)?; // v = 1 edge
    edge([(0, 2), (0, 1)], h[3], &mut g)?; // u = 0 edge

    for (slot, he) in [((0, 0), h[0]), ((3, 0), h[1]), ((3, 3), h[2]), ((0, 3), h[3])] {
        match diagonal(he) {
            Some(v) => g[slot.0][slot.1] = Some(pos(v)),
            // A diagonal is missing whenever either edge beside it is, and both of those have
            // already been allowed or refused above. Left empty for the reflection below.
            None => {}
        }
    }

    // **Reflect what is still missing, rows before corners.** `P₋₁ = 2·P₀ − P₁` about the
    // boundary row. Doing the rows first is what lets a corner fall out of the same rule
    // applied twice rather than needing one of its own.
    let reflect = |g: &mut [[Option<Point3<f64>>; 4]; 4], into: (usize, usize), about: (usize, usize), from: (usize, usize)| {
        if g[into.0][into.1].is_some() {
            return;
        }
        if let (Some(p0), Some(p1)) = (g[about.0][about.1], g[from.0][from.1]) {
            g[into.0][into.1] = Some(Point3::from(p0.coords * 2.0 - p1.coords));
        }
    };
    for k in 1..3 {
        reflect(&mut g, (k, 0), (k, 1), (k, 2)); // outward in v, below
        reflect(&mut g, (k, 3), (k, 2), (k, 1)); // outward in v, above
        reflect(&mut g, (0, k), (1, k), (2, k)); // outward in u, before
        reflect(&mut g, (3, k), (2, k), (1, k)); // outward in u, after
    }
    for (corner, about, from) in [
        ((0, 0), (0, 1), (0, 2)),
        ((3, 0), (3, 1), (3, 2)),
        ((3, 3), (3, 2), (3, 1)),
        ((0, 3), (0, 2), (0, 1)),
    ] {
        reflect(&mut g, corner, about, from);
    }

    let mut out = [[Point3::origin(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = g[i][j]?;
        }
    }
    Some(out)
}

/// Maps `(u, v)` on a face into the sub-face containing it after one subdivision, returning
/// `(sub-face index, u', v')`.
///
/// Sub-face `i` is emitted by [`subdivide`] as `[corner_i, edge(i, i+1), face_point,
/// edge(i-1, i)]`, so each sub-face's local frame is rotated a quarter turn from its parent's.
/// Getting this wrong is silent — the surface stays smooth but samples the wrong place — so it
/// is pinned by `descent_agrees_across_sub_faces` below.
fn descend(u: f64, v: f64) -> (usize, f64, f64) {
    if u < 0.5 && v < 0.5 {
        (0, 2.0 * u, 2.0 * v)
    } else if v < 0.5 {
        (1, 2.0 * v, 2.0 * (1.0 - u))
    } else if u >= 0.5 {
        (2, 2.0 * (1.0 - u), 2.0 * (1.0 - v))
    } else {
        (3, 2.0 * (1.0 - v), 2.0 * u)
    }
}

/// Evaluates the limit surface of a cage, caching subdivided levels between queries.
///
/// Build one per cage and reuse it: the cached levels are what make repeated queries cheap,
/// and they are only built as deep as some query actually needs.
/// Exact limit position of a vertex, on any [`QuadMesh`] — the closed-form mask from
/// [`crate::limit_position`], generalised so it also works on a patch.
///
/// `None` where the vertex's ring is incomplete, since the mask needs every neighbour.
fn limit_position_on<M: QuadTopology + ?Sized>(mesh: &M, v: VertexId) -> Option<Point3<f64>> {
    let n = mesh.valence(v)?;
    let n_f = n as f64;
    let p = mesh.vertex_position(v);

    let adjacent_sum = mesh
        .adjacent_vertices(v)
        .into_iter()
        .fold(Vector3::zeros(), |acc, a| acc + mesh.vertex_position(a).coords);

    // The diagonal vertex of each incident quad: the one sharing no edge with `v`.
    let diagonal_sum = mesh
        .adjacent_faces(v)
        .into_iter()
        .fold(Vector3::zeros(), |acc, f| {
            let corners = mesh.face_vertices(f);
            let i = corners
                .iter()
                .position(|&c| c == v)
                .expect("an incident face contains this vertex");
            acc + mesh.vertex_position(corners[(i + 2) % 4]).coords
        });

    Some(Point3::from(
        (p.coords * n_f * n_f + adjacent_sum * 4.0 + diagonal_sum) / (n_f * (n_f + 5.0)),
    ))
}

/// Area-weighted normal at a vertex, on any [`QuadMesh`] — used only at extraordinary
/// vertices, where no single bicubic patch covers the point.
fn vertex_normal_on<M: QuadTopology + ?Sized>(mesh: &M, v: VertexId) -> Vector3<f64> {
    let mut acc = Vector3::zeros();
    for f in mesh.adjacent_faces(v) {
        let c = mesh.face_vertices(f);
        let p: Vec<_> = c.iter().map(|&x| mesh.vertex_position(x)).collect();
        acc += (p[1] - p[0]).cross(&(p[2] - p[0]));
    }
    acc.try_normalize(1e-12).unwrap_or_else(Vector3::z)
}

/// The subdivided neighbourhoods that irregular faces descend through.
///
/// One [`Patch`] holds *every* irregular face's neighbourhood at once rather than one patch
/// per face. Irregular faces cluster — they surround the same extraordinary vertices — so
/// their neighbourhoods overlap heavily, and a single extraction shares that work instead of
/// repeating it.
struct Descent {
    levels: Vec<Patch>,
    /// `children[level][face]` gives the four sub-faces at `level + 1`, or [`DROPPED`].
    children: Vec<Vec<[u32; 4]>>,
    /// Where each irregular cage face sits in level 0 of the patch.
    seed_of: HashMap<FaceId, FaceId>,
}

impl Descent {
    fn build(cage: &QuadMesh, rings: usize) -> Self {
        // The seeds are exactly the faces the fast path rejects, so the two can never
        // disagree about which faces need descending.
        let seeds: Vec<FaceId> = cage
            .face_ids()
            .filter(|&f| regular_grid(cage, f).is_none())
            .collect();
        let (patch, seed_of) = Patch::extract(cage, &seeds, rings);
        Self {
            levels: vec![patch],
            children: Vec::new(),
            seed_of,
        }
    }

    /// Subdivides the patch as deep as `level`, and no deeper.
    fn ensure_level(&mut self, level: usize) {
        while self.levels.len() <= level {
            let (next, children) = self
                .levels
                .last()
                .expect("level 0 always present")
                .subdivide();
            self.levels.push(next);
            self.children.push(children);
        }
    }
}

/// Evaluates the limit surface of a cage.
///
/// Regular faces — the overwhelming majority — are evaluated in closed form straight off the
/// cage, touching none of the machinery below. Irregular faces descend through subdivided
/// *neighbourhoods* (see [`Patch`]), built lazily and only around the faces that need them.
pub struct LimitSurface {
    /// **A quad mesh rather than a cage**, because a limit surface is a property of the mesh and
    /// not of the check that closed it. An open panel has one — that is what the boundary rule
    /// in `regular_grid` is for — and a cage's is the same surface it always was, since a cage
    /// *is* a quad mesh. Nothing here ever needed the closure; `CageField` is what needs it, and
    /// it still takes a `Cage`.
    cage: QuadMesh,
    max_descent: usize,
    /// Built on the first irregular query; a cage with no extraordinary vertices never
    /// builds one at all.
    descent: Option<Descent>,
}

impl LimitSurface {
    pub fn new(cage: &Cage) -> Self {
        Self::over(cage.mesh())
    }

    /// The same, over a mesh that may be open.
    pub fn over(mesh: &QuadMesh) -> Self {
        Self::with_max_descent_over(mesh, DEFAULT_MAX_DESCENT)
    }

    /// Builds an evaluator with an explicit descent limit, over a mesh that may be open — see
    /// [`DEFAULT_MAX_DESCENT`].
    ///
    /// Depth is no longer paid across the whole model: descent subdivides only the
    /// neighbourhoods of irregular faces, so raising this costs 4x on those neighbourhoods
    /// rather than 4x on the cage. Accuracy near an extraordinary vertex is what it buys.
    pub(crate) fn with_max_descent_over(mesh: &QuadMesh, max_descent: usize) -> Self {
        Self {
            cage: mesh.clone(),
            max_descent,
            descent: None,
        }
    }

    /// The limit-surface point at `(u, v)` on `face`, with `u` and `v` in `[0, 1]`.
    ///
    /// `(0, 0)` is the face's first corner, `u` runs toward the second and `v` toward the
    /// fourth — matching [`Cage::face_vertices`] order.
    pub fn evaluate(&mut self, face: FaceId, u: f64, v: f64) -> Result<Point3<f64>, CageError> {
        self.evaluate_with_normal(face, u, v).map(|(p, _)| p)
    }

    /// The limit-surface point and its exact unit normal at `(u, v)`.
    ///
    /// The descent's parameter remapping is a rotation by a quarter turn per level, which is
    /// orientation-preserving, so the normal computed on the resolved sub-patch is already
    /// correct for the original face — no chain-rule correction needed. (Checked in
    /// `normals_are_consistent_across_the_descent`.)
    pub(crate) fn evaluate_with_normal(
        &mut self,
        face: FaceId,
        u: f64,
        v: f64,
    ) -> Result<(Point3<f64>, Vector3<f64>), CageError> {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        // Fast path: a regular face is a bicubic patch over the cage itself, exactly. That is
        // almost the entire model, and it touches neither a patch nor a subdivision.
        if let Some(grid) = regular_grid(&self.cage, face) {
            return Ok(eval_bicubic_with_normal(&grid, u, v));
        }

        // Irregular. Descend inside the neighbourhood patch, widening the padding if a
        // descent ever runs off its edge. By the ring accounting in `patch.rs` the first
        // attempt always suffices — `every_sample_on_an_irregular_face_resolves_in_the_patch`
        // pins that — so the retry guards against a wrong constant, not a normal path.
        for attempt in 0..3 {
            if self.descent.is_none() {
                let rings = self.max_descent + 2 + attempt * 2;
                self.descent = Some(Descent::build(&self.cage, rings));
            }
            if let Some(result) = self.descend(face, u, v) {
                return Ok(result);
            }
            debug_assert!(
                attempt > 0,
                "patch padding of {} rings was too thin for a descent of {}",
                self.max_descent + 2,
                self.max_descent
            );
            self.descent = None;
        }

        // Unreachable given that padding, but a renderer must not panic: fall back to the
        // nearest cage corner's exact limit position.
        let corner = self.cage.face_vertices(face)[nearest_corner(u, v)];
        Ok((
            limit_position_on(&self.cage, corner)
                .unwrap_or_else(|| self.cage.vertex_position(corner)),
            vertex_normal_on(&self.cage, corner),
        ))
    }

    /// Walks an irregular face down through the patch levels, building them if needed.
    ///
    /// Levels are ensured up front rather than as the walk goes, so that the walk itself
    /// needs only `&Descent` and can be shared with the parallel path below. The eagerness
    /// costs nothing in practice: a tessellation samples at `1/(n-1)` from each corner, which
    /// needs full depth anyway, and the patch is small by construction.
    fn descend(&mut self, face: FaceId, u: f64, v: f64) -> Option<(Point3<f64>, Vector3<f64>)> {
        let max_descent = self.max_descent;
        self.descent.as_mut()?.ensure_level(max_descent);
        walk(self.descent.as_ref()?, max_descent, face, u, v)
    }

    /// How many subdivision levels are currently cached.
    ///
    /// Counts levels of the *neighbourhood* patch now, not of the whole cage. Level `i` of
    /// the patch is level `i` of the cage restricted to the irregular neighbourhoods, so the
    /// number still means what it always did: how deep anything has had to descend.
    ///
    /// Only the tests ask now, so only the tests compile it. The import-cost example read it
    /// too while it timed the limit-surface tessellation, and stopped when that mesh went.
    #[cfg(test)]
    fn cached_levels(&self) -> usize {
        self.descent.as_ref().map_or(1, |d| d.levels.len())
    }

    /// Total faces held across all cached patch levels — the memory the descent actually
    /// costs, which is the number the whole-cage version made unaffordable. Test-only now, for
    /// the same reason as `cached_levels`.
    #[cfg(test)]
    fn cached_faces(&self) -> usize {
        self.descent
            .as_ref()
            .map_or(0, |d| d.levels.iter().map(|l| l.face_count()).sum())
    }

    /// Builds every subdivision level the irregular faces could need, so that evaluation
    /// afterwards needs only `&self`.
    ///
    /// This is what makes a parallel tessellation possible. The lazy cache is the only reason
    /// [`Self::evaluate_with_normal`] takes `&mut self`; warm it once, up front, and the
    /// per-face work becomes read-only and so trivially shareable across threads. It is cheap
    /// for the same reason local descent is cheap — it only ever touches the neighbourhoods
    /// of extraordinary vertices, not the model.
    pub(crate) fn prewarm(&mut self) {
        if self.descent.is_none() {
            // A cage with no extraordinary vertices needs no descent at all, and must not be
            // given an empty one — `cached_levels` would then overstate the work done.
            let has_irregular = self
                .cage
                .face_ids()
                .any(|f| regular_grid(&self.cage, f).is_none());
            if !has_irregular {
                return;
            }
            self.descent = Some(Descent::build(&self.cage, self.max_descent + 2));
        }
        let max_descent = self.max_descent;
        if let Some(d) = self.descent.as_mut() {
            d.ensure_level(max_descent);
        }
    }

    /// The read-only twin of [`Self::evaluate_with_normal`], for use after [`Self::prewarm`].
    ///
    /// `None` only if the cache is not warm enough — which is to say, if `prewarm` was not
    /// called. It never silently returns a worse answer: same fast path, same walk, same
    /// results.
    pub(crate) fn evaluate_with_normal_ref(
        &self,
        face: FaceId,
        u: f64,
        v: f64,
    ) -> Option<(Point3<f64>, Vector3<f64>)> {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);
        if let Some(grid) = regular_grid(&self.cage, face) {
            return Some(eval_bicubic_with_normal(&grid, u, v));
        }
        walk(self.descent.as_ref()?, self.max_descent, face, u, v)
    }

    /// Writes an `n x n` grid of surface samples for one face into `out`, as
    /// `(position, normal)` in row-major `(u, v)` order.
    ///
    /// Per-face rather than whole-cage on purpose: Catmull-Clark's support is local, so an
    /// edit only invalidates the faces near the moved vertex. Tessellating face-at-a-time is
    /// what lets a redraw touch those faces alone instead of rebuilding the model.
    pub fn tessellate_face(
        &mut self,
        face: FaceId,
        n: usize,
        out: &mut Vec<(Point3<f64>, Vector3<f64>)>,
    ) -> Result<(), CageError> {
        debug_assert!(n >= 2, "a patch needs at least its corners");
        let step = 1.0 / (n - 1) as f64;
        for j in 0..n {
            for i in 0..n {
                let sample = self.evaluate_with_normal(face, i as f64 * step, j as f64 * step)?;
                out.push(sample);
            }
        }
        Ok(())
    }

    /// A single closed, watertight triangulation of the whole limit surface, with each
    /// vertex's exact analytic normal.
    ///
    /// Unlike [`Self::tessellate_face`], grid points on a shared original edge or corner are
    /// evaluated once and shared by every face touching them, rather than duplicated per face.
    /// [`Self::tessellate_face`] can afford duplication because analytic normals already agree
    /// across a shared boundary with nothing to weld — but a duplicated boundary is a mesh
    /// *boundary* edge to anything that needs real topology, and [`crate`]-adjacent code that
    /// classifies closest points by feature (face / edge / vertex) for a sign test needs
    /// exactly that: one triangle mesh where a shared edge really is shared.
    ///
    /// Each face is sampled on a uniform `(2^levels + 1)`-point grid in its own `(u, v)`
    /// domain. `levels` matches the vocabulary used elsewhere for cage subdivision depth, even
    /// though nothing here actually subdivides — the grid resolution it implies is what
    /// carries over.
    pub fn tessellate(&mut self, levels: usize) -> Tessellation {
        self.prewarm();
        let n = 1usize << levels;

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut triangles = Vec::new();
        // **Where each triangle came from, in the surface's own terms.** A welded vertex belongs
        // to every face that touches it and so has no single `(u, v)`; a triangle corner has
        // exactly one, because a triangle belongs to one face. Recorded per triangle for that
        // reason, and recorded here rather than by a second walk, because a second walk over
        // the same grid is a second chance to disagree with this one about which corner is
        // which.
        let mut triangle_faces: Vec<FaceId> = Vec::new();
        let mut triangle_uvs: Vec<[[f64; 2]; 3]> = Vec::new();
        let mut triangle_gradients: Vec<[[Vector3<f64>; 2]; 3]> = Vec::new();
        // A grid corner is shared by every face incident to that original vertex.
        let mut corners_seen: HashMap<usize, u32> = HashMap::new();
        // A grid edge point is shared by the (at most) two faces on either side of that
        // original edge — keyed by the edge's two vertex indices (lower first) and how far
        // along that direction the point sits, so either face resolves to the same key.
        let mut edges_seen: HashMap<(usize, usize, usize), u32> = HashMap::new();

        for face in self.cage.face_ids() {
            let c = self.cage.face_vertices(face);
            let mut grid = vec![0u32; (n + 1) * (n + 1)];

            for i in 0..=n {
                for j in 0..=n {
                    let u = i as f64 / n as f64;
                    let v = j as f64 / n as f64;

                    let idx = match (i == 0, i == n, j == 0, j == n) {
                        (true, _, true, _) => weld_corner(self, &mut positions, &mut normals, &mut corners_seen, face, c[0], 0.0, 0.0),
                        (_, true, true, _) => weld_corner(self, &mut positions, &mut normals, &mut corners_seen, face, c[1], 1.0, 0.0),
                        (_, true, _, true) => weld_corner(self, &mut positions, &mut normals, &mut corners_seen, face, c[2], 1.0, 1.0),
                        (true, _, _, true) => weld_corner(self, &mut positions, &mut normals, &mut corners_seen, face, c[3], 0.0, 1.0),
                        (_, _, true, _) => weld_edge(self, &mut positions, &mut normals, &mut edges_seen, face, c[0], c[1], i, n, u, v),
                        (_, true, _, _) => weld_edge(self, &mut positions, &mut normals, &mut edges_seen, face, c[1], c[2], j, n, u, v),
                        (_, _, _, true) => weld_edge(self, &mut positions, &mut normals, &mut edges_seen, face, c[3], c[2], i, n, u, v),
                        (true, _, _, _) => weld_edge(self, &mut positions, &mut normals, &mut edges_seen, face, c[0], c[3], j, n, u, v),
                        _ => {
                            let (p, nrm) = self
                                .evaluate_with_normal_ref(face, u, v)
                                .expect("prewarmed above");
                            let idx = positions.len() as u32;
                            positions.push(p);
                            normals.push(nrm);
                            idx
                        }
                    };
                    grid[i * (n + 1) + j] = idx;
                }
            }

            // **How fast `u` and `v` change with world position, at every grid point.**
            //
            // Per point rather than per triangle, and that distinction is the whole reason this
            // is here. A triangle's own gradient is constant on it and *jumps* at its edges —
            // and on a convex surface a wedge of space of real volume has its nearest point on
            // one of those edges, where the two triangles disagree. A conformal lattice sizes
            // its walls by these numbers, so a jump is a visible step in wall thickness along
            // every facet. Measured before and after on a domed panel: 1.32 mm of disagreement
            // between the two paths, and 0.013 mm once these vary continuously.
            //
            // Central differences along the face's own grid, one-sided at its border. That is
            // O(h^2) against the surface's analytic tangents and — the property actually being
            // bought — continuous, where a facet gradient is neither. Across a *face* boundary
            // they still jump, and must: two faces have unrelated parametrizations, which is
            // what "cells are per face" already says.
            let position_at = |i: usize, j: usize| positions[grid[i * (n + 1) + j] as usize];
            let mut grads = vec![[Vector3::zeros(); 2]; (n + 1) * (n + 1)];
            for i in 0..=n {
                for j in 0..=n {
                    let (i0, i1) = (i.saturating_sub(1), (i + 1).min(n));
                    let (j0, j1) = (j.saturating_sub(1), (j + 1).min(n));
                    let du = (i1 - i0) as f64 / n as f64;
                    let dv = (j1 - j0) as f64 / n as f64;
                    let tan_u = (position_at(i1, j) - position_at(i0, j)) / du.max(1e-12);
                    let tan_v = (position_at(i, j1) - position_at(i, j0)) / dv.max(1e-12);
                    // The first fundamental form inverted: `grad u` and `grad v` are the rows of
                    // the pseudo-inverse of `[tan_u tan_v]`, which is what makes them right
                    // under the shear a quad parametrization always has.
                    let (e, f_, g) = (tan_u.dot(&tan_u), tan_u.dot(&tan_v), tan_v.dot(&tan_v));
                    let det = e * g - f_ * f_;
                    grads[i * (n + 1) + j] = if det.abs() > 1e-18 {
                        [
                            (tan_u * g - tan_v * f_) / det,
                            (tan_v * e - tan_u * f_) / det,
                        ]
                    } else {
                        [Vector3::zeros(); 2]
                    };
                }
            }

            // Winding matches `regular_grid`'s convention: `(i, j)`, `(i+1, j)`, `(i+1, j+1)`,
            // `(i, j+1)` plays the same corner-role pattern as a face's own corners 0..3, which
            // carry validated outward winding — so a sub-cell inherits it.
            for i in 0..n {
                for j in 0..n {
                    let a = grid[i * (n + 1) + j];
                    let b = grid[(i + 1) * (n + 1) + j];
                    let cc = grid[(i + 1) * (n + 1) + j + 1];
                    let d = grid[i * (n + 1) + j + 1];
                    triangles.push([a, b, cc]);
                    triangles.push([a, cc, d]);
                    let at = |i: usize, j: usize| [i as f64 / n as f64, j as f64 / n as f64];
                    let (p00, p10) = (at(i, j), at(i + 1, j));
                    let (p11, p01) = (at(i + 1, j + 1), at(i, j + 1));
                    let g = |i: usize, j: usize| grads[i * (n + 1) + j];
                    let (g00, g10) = (g(i, j), g(i + 1, j));
                    let (g11, g01) = (g(i + 1, j + 1), g(i, j + 1));
                    triangle_faces.push(face);
                    triangle_uvs.push([p00, p10, p11]);
                    triangle_gradients.push([g00, g10, g11]);
                    triangle_faces.push(face);
                    triangle_uvs.push([p00, p11, p01]);
                    triangle_gradients.push([g00, g11, g01]);
                }
            }
        }

        Tessellation {
            positions,
            normals,
            triangles,
            triangle_faces,
            triangle_uvs,
            triangle_gradients,
        }
    }
}

/// One watertight triangle mesh sampled off a [`LimitSurface`], with an exact analytic normal
/// per vertex rather than one recovered from the triangles themselves.
pub struct Tessellation {
    pub positions: Vec<Point3<f64>>,
    pub normals: Vec<Vector3<f64>>,
    pub triangles: Vec<[u32; 3]>,
    /// Which face of the original mesh each triangle was sampled from, parallel to `triangles`.
    pub triangle_faces: Vec<FaceId>,
    /// Each triangle's three corners in that face's own `(u, v)`, parallel to `triangles`.
    ///
    /// **Per triangle rather than per vertex, because a welded vertex has no single answer.**
    /// A grid corner is shared by every face touching it and each of those faces names it by a
    /// different `(u, v)`; a triangle belongs to one face, so its corners do too. Carrying the
    /// pair here is what lets a closest-point query come back with *where on the surface* it
    /// landed and not only how far away it was — which is the whole of what a conformal cell
    /// map needs and what item 46 is built on.
    pub triangle_uvs: Vec<[[f64; 2]; 3]>,
    /// Each triangle's three corners' `(∇u, ∇v)` in world space, parallel to `triangles`.
    ///
    /// **Per corner, so they can be interpolated rather than stepped.** How fast the face's own
    /// coordinates change per millimetre is what sizes a conformal lattice's cells and its
    /// walls; taken per triangle it is constant on each and discontinuous at every edge between
    /// them, and a convex surface puts a region of real volume on the wrong side of that. Per
    /// corner and interpolated, it varies the way the surface does.
    ///
    /// Alongside `triangle_uvs` and per triangle for the same reason: a welded vertex belongs to
    /// several faces and has different coordinates — and so different gradients — in each.
    pub triangle_gradients: Vec<[[Vector3<f64>; 2]; 3]>,
}

/// Resolves a face-grid corner to a global vertex index, evaluating and caching it once per
/// original cage vertex regardless of how many faces touch it.
#[allow(clippy::too_many_arguments)]
fn weld_corner(
    surface: &LimitSurface,
    positions: &mut Vec<Point3<f64>>,
    normals: &mut Vec<Vector3<f64>>,
    cache: &mut HashMap<usize, u32>,
    face: FaceId,
    vertex: VertexId,
    u: f64,
    v: f64,
) -> u32 {
    if let Some(&idx) = cache.get(&vertex.index()) {
        return idx;
    }
    let (p, n) = surface
        .evaluate_with_normal_ref(face, u, v)
        .expect("prewarmed above");
    let idx = positions.len() as u32;
    positions.push(p);
    normals.push(n);
    cache.insert(vertex.index(), idx);
    idx
}

/// Resolves a face-grid edge-interior point to a global vertex index. `k` (in `1..n`) is how
/// far along `start -> end` this point sits; keying by physical position rather than by which
/// face asked is what lets the neighbouring face — which walks the same edge in reverse —
/// resolve to the same vertex.
#[allow(clippy::too_many_arguments)]
fn weld_edge(
    surface: &LimitSurface,
    positions: &mut Vec<Point3<f64>>,
    normals: &mut Vec<Vector3<f64>>,
    cache: &mut HashMap<(usize, usize, usize), u32>,
    face: FaceId,
    start: VertexId,
    end: VertexId,
    k: usize,
    n: usize,
    u: f64,
    v: f64,
) -> u32 {
    let (lo, hi) = if start.index() < end.index() {
        (start.index(), end.index())
    } else {
        (end.index(), start.index())
    };
    let step = if start.index() == lo { k } else { n - k };
    let key = (lo, hi, step);
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let (p, nrm) = surface
        .evaluate_with_normal_ref(face, u, v)
        .expect("prewarmed above");
    let idx = positions.len() as u32;
    positions.push(p);
    normals.push(nrm);
    cache.insert(key, idx);
    idx
}

/// The descent walk, over already-built patch levels.
///
/// One copy, shared by the `&mut` and `&self` entry points. It is orientation sensitive — it
/// rotates around corners and steps across edges — and a mistake here is silent, so a second
/// copy for the parallel path would be a second chance to get it wrong.
fn walk(
    descent: &Descent,
    max_descent: usize,
    face: FaceId,
    u: f64,
    v: f64,
) -> Option<(Point3<f64>, Vector3<f64>)> {
    let mut pf = *descent.seed_of.get(&face)?;
    let (mut u, mut v) = (u, v);

    // Inclusive bound: `max_descent` descents means `max_descent + 1` levels get a regularity
    // check. Checking one fewer silently snaps points that were about to resolve — which
    // showed up as a normal discontinuity, not as a wrong position.
    for level in 0..=max_descent {
        let current = descent.levels.get(level)?;
        if let Some(grid) = regular_grid(current, pf) {
            return Some(eval_bicubic_with_normal(&grid, u, v));
        }
        if level == max_descent {
            break;
        }
        let (quadrant, nu, nv) = descend(u, v);
        let child = *descent.children.get(level)?.get(pf.index())?.get(quadrant)?;
        if child == DROPPED {
            return None;
        }
        pf = FaceId::from_index(child as usize);
        u = nu;
        v = nv;
    }

    // Still irregular at the descent limit: the query is inside the shrinking patch that hugs
    // the extraordinary vertex, which is always local corner 0 of the sub-face we keep
    // descending into. Its limit position is known exactly; the normal comes from the
    // surrounding geometry, since no single patch covers the vertex.
    let deepest = descent.levels.get(max_descent)?;
    let corner = deepest.face_vertices(pf)[0];
    Some((
        limit_position_on(deepest, corner)?,
        vertex_normal_on(deepest, corner),
    ))
}

/// Which corner of a face `(u, v)` is nearest, in `face_vertices` order.
fn nearest_corner(u: f64, v: f64) -> usize {
    match (u >= 0.5, v >= 0.5) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

/// The cage faces whose limit-surface patches move when `vertex` moves.
///
/// A face's shape depends on its 4x4 control grid, which reaches two steps out from the face
/// — so a vertex influences every face within two rings of it, and no others. This is the
/// bound that makes editing cost independent of model size, so it is deliberately computed
/// from that grid's reach rather than guessed at.
pub fn faces_influenced_by(cage: &Cage, vertex: VertexId) -> Vec<FaceId> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();

    // Ring 1: faces touching the vertex. Ring 2: faces touching any of those faces' vertices.
    for f in cage.adjacent_faces(vertex) {
        for v in cage.face_vertices(f) {
            for g in cage.adjacent_faces(v) {
                if seen.insert(g) {
                    out.push(g);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{limit_position, subdivide, subdivide_n};
    use approx::assert_relative_eq;
    use fieldform_cage::Cage;

    /// A flat grid of `n` by `n` quads in the XY plane, open all the way round.
    fn open_grid(n: usize, size: f64) -> fieldform_cage::QuadMesh {
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
        fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a flat grid")
    }

    /// **The limit surface meets the boundary it was drawn on.**
    ///
    /// The one check in section 4 of the spec that catches a silently-wrong rule. A boundary
    /// face has no control ring on the open side, and the three things an implementation can do
    /// about it are: refuse (loud), fall back to bilinear (a visible crease), or evaluate on a
    /// partial grid — which pulls the surface inward from the mesh edge by roughly a third of a
    /// face, looks entirely plausible, and passes every topological test there is.
    ///
    /// So this is geometric. A flat grid's limit surface is the plane it was drawn in, and its
    /// boundary is the square the corner points describe: sample along the parametric edge of a
    /// boundary face and the points have to land *on that square*, not inside it.
    #[test]
    fn the_limit_surface_meets_the_boundary_it_was_drawn_on() {
        let size = 4.0;
        let mesh = open_grid(4, size);

        // The face at the origin corner: its v = 0 edge and its u = 0 edge are both on the
        // mesh's boundary, so its grid is reflected in both directions.
        let face = FaceId::from_index(0);
        let grid = regular_grid(&mesh, face)
            .expect("a boundary face is regular under the reflection rule");

        // Along v = 0, which is the mesh's own bottom edge: y must be nought, not pulled up.
        for k in 0..=8 {
            let u = k as f64 / 8.0;
            let p = eval_bicubic_with_normal(&grid, u, 0.0).0;
            assert!(
                p.y.abs() < 1e-9,
                "the surface left the boundary at u = {u}: y = {}, which is {:.1}% of a face",
                p.y,
                p.y / (size / 4.0) * 100.0
            );
            assert!(p.z.abs() < 1e-9, "a flat mesh has a flat limit surface");
        }

        // And the corner is interpolated exactly, which is the case the double reflection has
        // to get right on its own.
        let corner = eval_bicubic_with_normal(&grid, 0.0, 0.0).0;
        assert!(
            (corner - Point3::new(0.0, 0.0, 0.0)).norm() < 1e-9,
            "the mesh's corner should be the surface's corner, got {corner:?}"
        );
    }

    /// **A patch's cut edge is not a boundary, and must not be reflected.**
    ///
    /// The distinction the whole rule rests on. A patch is a neighbourhood lifted out of a
    /// larger mesh, so a half-edge with no twin there has a face on the other side in the mesh
    /// it came from — reflecting would invent a surface where a real one already exists. A
    /// closed cage has no boundary at all, so every one of its faces answers the way it always
    /// did and the reflection can never fire.
    #[test]
    fn a_closed_cage_reflects_nothing_and_answers_as_it_always_did() {
        let cage = Cage::cube_of(2.0);
        assert!(
            cage.face_ids().all(|f| !QuadTopology::on_boundary(&cage, cage.face_half_edges(f)[0])),
            "a cube has no boundary half-edges"
        );
        // Every face of a cube is irregular — all its corners are valence 3 — so every one of
        // them descends, exactly as before. If the reflection had started firing here, these
        // would have become regular and the surface would have changed shape.
        assert!(
            cage.face_ids().all(|f| regular_grid(&cage, f).is_none()),
            "a cube's faces are all irregular and none of them is rescued by reflection"
        );
    }

    /// The interior of an open mesh is unchanged: a face whose ring is complete is the same
    /// bicubic patch it would be in a closed mesh, reflection or no reflection.
    #[test]
    fn an_open_meshs_interior_face_is_the_patch_it_always_was() {
        let mesh = open_grid(5, 5.0);
        // The middle face of a 5x5 grid has a full ring.
        let middle = FaceId::from_index(2 * 5 + 2);
        let grid = regular_grid(&mesh, middle).expect("a full ring is regular");
        for (i, row) in grid.iter().enumerate() {
            for (j, p) in row.iter().enumerate() {
                assert!(p.z.abs() < 1e-12, "grid[{i}][{j}] left the plane");
            }
        }
        // Face (2, 2) of a unit-step grid spans x and y from 2 to 3, so its middle is 2.5.
        let centre = eval_bicubic_with_normal(&grid, 0.5, 0.5).0;
        assert!(
            (centre - Point3::new(2.5, 2.5, 0.0)).norm() < 1e-9,
            "the middle of the middle face should be where it was drawn, got {centre:?}"
        );
    }

    /// A cage with regular faces. A cube has none — every corner is valence 3 — so two
    /// subdivisions are needed before interior faces are surrounded entirely by valence-4
    /// vertices.
    fn regular_cage() -> Cage {
        subdivide_n(&Cage::cube(), 2).unwrap()
    }

    fn first_regular_face(cage: &Cage) -> FaceId {
        cage.face_ids()
            .find(|&f| regular_grid(cage, f).is_some())
            .expect("two subdivisions of a cube must produce regular faces")
    }

    #[test]
    fn a_cube_has_no_regular_faces_but_subdivision_produces_them() {
        // Pins the reasoning above: the descent exists because of this, not by accident.
        let cube = Cage::cube();
        assert!(
            cube.face_ids().all(|f| regular_grid(&cube, f).is_none()),
            "every cube corner is valence 3, so no face can be a bicubic patch"
        );

        let two = subdivide_n(&cube, 2).unwrap();
        let regular = two.face_ids().filter(|&f| regular_grid(&two, f).is_some()).count();
        assert!(regular > 0, "expected regular faces after two subdivisions");
    }

    #[test]
    fn regular_patch_corner_matches_the_closed_form_limit_position() {
        // The strongest available check: two independent derivations must agree. Evaluating
        // the bicubic patch at (0,0) reduces to weights 1/6, 4/6, 1/6 in each direction,
        // which is exactly the valence-4 limit mask (16P + 4*sumA + sumD)/36 computed by
        // `limit_position` through completely separate code.
        let cage = regular_cage();
        let face = first_regular_face(&cage);
        let grid = regular_grid(&cage, face).unwrap();

        let corners = cage.face_vertices(face);
        let expected = limit_position(&cage, corners[0]);
        let got = eval_bicubic(&grid, 0.0, 0.0);

        assert_relative_eq!(got.x, expected.x, epsilon = 1e-6);
        assert_relative_eq!(got.y, expected.y, epsilon = 1e-6);
        assert_relative_eq!(got.z, expected.z, epsilon = 1e-6);
    }

    #[test]
    fn every_corner_of_a_regular_patch_matches_its_limit_position() {
        // The same check at all four corners, which also pins the grid's orientation: a
        // transposed or mirrored grid would still pass at (0,0) alone.
        let cage = regular_cage();
        let face = first_regular_face(&cage);
        let grid = regular_grid(&cage, face).unwrap();
        let corners = cage.face_vertices(face);

        for (i, &(u, v)) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .enumerate()
        {
            let expected = limit_position(&cage, corners[i]);
            let got = eval_bicubic(&grid, u, v);
            assert_relative_eq!(got.x, expected.x, epsilon = 1e-6);
            assert_relative_eq!(got.y, expected.y, epsilon = 1e-6);
            assert_relative_eq!(got.z, expected.z, epsilon = 1e-6);
        }
    }

    #[test]
    fn face_centre_matches_the_face_point_of_the_next_level() {
        // (u,v) = (0.5, 0.5) is exactly where the next level's face point sits, so that
        // vertex's closed-form limit position is the ground truth for the patch centre.
        // Independent of the bicubic path.
        let cage = Cage::cube();
        let refined = subdivide(&cage).unwrap();
        let mut surface = LimitSurface::new(&cage);

        for f in cage.face_ids() {
            // `subdivide` lays out [updated originals | face points | edge points].
            let face_point = VertexId::from_index(cage.vertex_count() + f.index());
            let expected = limit_position(&refined, face_point);
            let got = surface.evaluate(f, 0.5, 0.5).unwrap();

            assert_relative_eq!(got.x, expected.x, epsilon = 1e-5);
            assert_relative_eq!(got.y, expected.y, epsilon = 1e-5);
            assert_relative_eq!(got.z, expected.z, epsilon = 1e-5);
        }
    }

    #[test]
    fn edge_midpoints_match_their_subdivided_edge_points() {
        // The sharpest test of `descend`'s rotation. Each of the four edge midpoints is
        // reached through a *different* quadrant — (0.5,0) via sub-face 1, (1,0.5) and
        // (0.5,1) via sub-face 2, (0,0.5) via sub-face 3 — and each has an independently
        // known ground truth: the corresponding edge point's closed-form limit position after
        // one subdivision. A wrong quarter-turn in any quadrant fails here while still
        // producing a perfectly smooth (and wrong) surface.
        let cage = Cage::cube();
        let refined = subdivide(&cage).unwrap();
        let mut surface = LimitSurface::new(&cage);

        // `subdivide` lays out [updated originals | face points | edge points], with edge
        // points in `cage.edges()` order.
        let edge_point_base = cage.vertex_count() + cage.face_count();
        let edge_list: Vec<_> = cage.edges().collect();
        let edge_point = |a: VertexId, b: VertexId| -> VertexId {
            let i = edge_list
                .iter()
                .position(|&(x, y)| (x == a && y == b) || (x == b && y == a))
                .expect("edge must exist");
            VertexId::from_index(edge_point_base + i)
        };

        for f in cage.face_ids() {
            let c = cage.face_vertices(f);
            let cases = [
                ((0.5, 0.0), edge_point(c[0], c[1])),
                ((1.0, 0.5), edge_point(c[1], c[2])),
                ((0.5, 1.0), edge_point(c[2], c[3])),
                ((0.0, 0.5), edge_point(c[3], c[0])),
            ];
            for ((u, v), ep) in cases {
                let expected = limit_position(&refined, ep);
                let got = surface.evaluate(f, u, v).unwrap();
                assert_relative_eq!(got.x, expected.x, epsilon = 1e-5);
                assert_relative_eq!(got.y, expected.y, epsilon = 1e-5);
                assert_relative_eq!(got.z, expected.z, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn normals_are_consistent_across_the_descent() {
        // The descent rotates the parameter frame a quarter turn per level. Those rotations
        // are orientation-preserving, so a normal computed on the resolved sub-patch needs no
        // correction — but if that reasoning were wrong, normals would flip between quadrants
        // and shading would show hard creases across sub-face boundaries on a smooth surface.
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let f = cage.face_ids().next().unwrap();

        let mut previous: Option<Vector3<f64>> = None;
        for step in 0..=8 {
            let t = 0.1 + 0.8 * step as f64 / 8.0;
            let (_, n) = surface.evaluate_with_normal(f, t, t).unwrap();
            assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-5);
            if let Some(p) = previous {
                assert!(
                    p.dot(&n) > 0.9,
                    "normal flipped or lurched across a sub-face boundary at t={t}"
                );
            }
            previous = Some(n);
        }
    }

    #[test]
    fn normals_point_outward_on_a_cube() {
        // A cube cage's limit surface is convex, so every normal must point away from the
        // centre. Catches a globally inverted cross product, which the continuity check above
        // would not.
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        for f in cage.face_ids() {
            for &(u, v) in &[(0.25, 0.25), (0.5, 0.5), (0.75, 0.25)] {
                let (p, n) = surface.evaluate_with_normal(f, u, v).unwrap();
                assert!(
                    p.coords.dot(&n) > 0.0,
                    "normal at {p:?} on face {f:?} points inward: {n:?}"
                );
            }
        }
    }

    #[test]
    fn influence_is_local_and_covers_the_control_grid() {
        // The claim that makes editing cost independent of model size. On a large cage the
        // influenced set must stay small, and must include every face whose 4x4 grid contains
        // the vertex — verified by moving the vertex and checking nothing outside the set
        // changed.
        let cage = subdivide_n(&Cage::cube(), 3).unwrap();
        let v = VertexId::from_index(cage.vertex_count() / 2);
        let influenced = faces_influenced_by(&cage, v);

        assert!(
            influenced.len() < cage.face_count() / 4,
            "influence should be local: {} of {} faces",
            influenced.len(),
            cage.face_count()
        );

        // Everything outside the influenced set must be unmoved by an edit to `v`.
        let mut before = LimitSurface::new(&cage);
        let baseline: Vec<_> = cage
            .face_ids()
            .map(|f| before.evaluate(f, 0.5, 0.5).unwrap())
            .collect();

        let mut moved = cage.clone();
        let p = moved.vertex_position(v);
        moved.set_vertex_position(v, p + Vector3::new(0.3, 0.2, 0.1));
        let mut after = LimitSurface::new(&moved);

        let influenced_set: std::collections::HashSet<_> = influenced.into_iter().collect();
        for f in cage.face_ids() {
            let now = after.evaluate(f, 0.5, 0.5).unwrap();
            let delta = (now - baseline[f.index()]).norm();
            if !influenced_set.contains(&f) {
                assert!(
                    delta < 1e-6,
                    "face {f:?} moved by {delta} but was outside the influenced set"
                );
            }
        }
    }

    #[test]
    fn tessellate_face_fills_a_grid_with_unit_normals() {
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let f = cage.face_ids().next().unwrap();

        let mut out = Vec::new();
        surface.tessellate_face(f, 4, &mut out).unwrap();
        assert_eq!(out.len(), 16);
        for (_, n) in &out {
            assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-5);
        }

        // Corners of the grid must land on the face's corner limit positions.
        let corners = cage.face_vertices(f);
        let expected = limit_position(&cage, corners[0]);
        let got = out[0].0;
        assert_relative_eq!(got.x, expected.x, epsilon = 1e-4);
        assert_relative_eq!(got.y, expected.y, epsilon = 1e-4);
        assert_relative_eq!(got.z, expected.z, epsilon = 1e-4);
    }

    #[test]
    fn descent_agrees_across_sub_faces() {
        // The parameter remapping in `descend` rotates each sub-face's frame. If that is
        // wrong the surface still looks smooth but samples the wrong point, so check that a
        // parameter on a shared sub-face boundary evaluates identically from either side.
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let f = cage.face_ids().next().unwrap();

        // Points on the internal cross where sub-faces meet.
        for &(u, v) in &[(0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)] {
            let direct = surface.evaluate(f, u, v).unwrap();
            // Nudge across the boundary; the surface is continuous, so a tiny step must give
            // a tiny change rather than jumping to a different part of the patch.
            let nudged = surface.evaluate(f, u + 1e-4, v).unwrap();
            assert!(
                (direct - nudged).norm() < 1e-2,
                "discontinuity at ({u}, {v}): {direct:?} vs {nudged:?}"
            );
        }
    }

    #[test]
    fn evaluation_is_continuous_across_a_shared_edge() {
        // Sampling along one face's edge and the neighbouring face's matching edge must trace
        // the same curve — the limit surface is C2 away from extraordinary vertices, so any
        // mismatch means the grid or the descent is misoriented.
        let cage = regular_cage();
        let face = first_regular_face(&cage);
        let corners = cage.face_vertices(face);

        // The neighbour across the v = 0 edge (corners[0] -> corners[1]).
        let (f0, f1) = cage.edge_faces(corners[0], corners[1]).unwrap();
        let neighbour = if f0 == face { f1 } else { f0 };
        let n_corners = cage.face_vertices(neighbour);

        // Find where the shared edge sits in the neighbour's winding.
        let start = (0..4)
            .find(|&i| {
                let a = n_corners[i];
                let b = n_corners[(i + 1) % 4];
                (a == corners[1] && b == corners[0]) || (a == corners[0] && b == corners[1])
            })
            .expect("the neighbour must contain the shared edge");

        let mut surface = LimitSurface::new(&cage);
        for step in 0..=4 {
            let t = step as f64 / 4.0;
            // Along this face: the v = 0 edge runs corners[0] -> corners[1] as u goes 0 -> 1.
            let here = surface.evaluate(face, t, 0.0).unwrap();

            // The same point from the neighbour, walking its edge in whichever direction it
            // is wound.
            let forward = n_corners[start] == corners[0];
            let s = if forward { t } else { 1.0 - t };
            let there = match start {
                0 => surface.evaluate(neighbour, s, 0.0).unwrap(),
                1 => surface.evaluate(neighbour, 1.0, s).unwrap(),
                2 => surface.evaluate(neighbour, 1.0 - s, 1.0).unwrap(),
                _ => surface.evaluate(neighbour, 0.0, 1.0 - s).unwrap(),
            };

            assert!(
                (here - there).norm() < 1e-5,
                "edge mismatch at t={t}: {here:?} vs {there:?}"
            );
        }
    }

    #[test]
    fn evaluation_converges_to_deep_subdivision() {
        // Sanity against the old approach: a densely subdivided, limit-projected mesh should
        // hug the evaluated surface. Not a proof of exactness, but it would catch a gross
        // error in either path.
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let dense = crate::limit_cage(&subdivide_n(&cage, 4).unwrap()).unwrap();

        let f = cage.face_ids().next().unwrap();
        let centre = surface.evaluate(f, 0.5, 0.5).unwrap();

        let nearest = dense
            .vertex_ids()
            .map(|v| (dense.vertex_position(v) - centre).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest < 0.02,
            "evaluated centre sits {nearest} from the densest sampled mesh"
        );
    }

    #[test]
    fn descent_only_goes_deep_near_extraordinary_vertices() {
        // The performance claim in this module's docs: most of the domain resolves shallowly.
        // A cube is the worst realistic case, with all eight corners extraordinary.
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let f = cage.face_ids().next().unwrap();

        surface.evaluate(f, 0.5, 0.5).unwrap();
        assert!(
            surface.cached_levels() <= 4,
            "the patch centre should not need a deep descent, used {}",
            surface.cached_levels()
        );
    }
    /// The read-only path exists so tessellation can run in parallel, which is only safe if
    /// it is the *same* evaluation. Not "close enough" — identical, bit for bit, since both
    /// run the same code over the same cache. A cube subdivided twice keeps extraordinary
    /// vertices, so this covers the descent as well as the closed-form fast path.
    #[test]
    fn the_read_only_path_is_identical_to_the_mutating_one() {
        let cage = subdivide_n(&Cage::cube(), 2).unwrap();
        let mut surface = LimitSurface::new(&cage);
        surface.prewarm();

        let n = 9;
        let step = 1.0 / (n - 1) as f64;
        let mut compared = 0;
        for f in cage.face_ids() {
            for j in 0..n {
                for i in 0..n {
                    let (u, v) = (i as f64 * step, j as f64 * step);
                    let want = surface.evaluate_with_normal(f, u, v).unwrap();
                    let got = surface
                        .evaluate_with_normal_ref(f, u, v)
                        .expect("prewarm makes every face evaluable read-only");
                    assert_eq!(want.0, got.0, "position at {f:?} ({u}, {v})");
                    assert_eq!(want.1, got.1, "normal at {f:?} ({u}, {v})");
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, cage.face_count() * n * n);
    }

    /// Without `prewarm` the read-only path must refuse rather than guess. Silently returning
    /// a coarser answer would make a missing prewarm invisible except as a subtly wrong
    /// surface.
    #[test]
    fn the_read_only_path_refuses_when_the_cache_is_cold() {
        let cage = Cage::cube(); // every face irregular, so every face needs the descent
        let surface = LimitSurface::new(&cage);
        let f = cage.face_ids().next().unwrap();
        assert!(surface.evaluate_with_normal_ref(f, 0.25, 0.25).is_none());
    }

    /// The whole point of local descent, as an assertion: the mesh it caches is set by the
    /// irregular faces, not by the model. A subdivided cube keeps exactly 24 irregular faces
    /// however deep it goes, so the cache must come out the *same size* at 1.5k faces and at
    /// 98k — where the old whole-cage descent would have grown 64-fold.
    ///
    /// Note this is a constant, not a small number: on a cage smaller than the constant, local
    /// descent caches more mesh than the cage has. That is the right trade for a tool whose
    /// requirement is 10k-100k faces, but it is a trade, not a free win.
    #[test]
    fn descent_cost_is_set_by_irregular_faces_not_by_cage_size() {
        let mut sizes = Vec::new();
        for k in 4..=6 {
            let cage = subdivide_n(&Cage::cube(), k).unwrap();
            let mut surface = LimitSurface::new(&cage);
            surface.prewarm();
            sizes.push((cage.face_count(), surface.cached_faces()));
        }

        let first = sizes[0].1;
        for &(faces, cached) in &sizes {
            assert_eq!(
                cached, first,
                "cage grew to {faces} faces but the descent cache moved from {first} to \
                 {cached}; it should track the 24 irregular faces, which do not grow"
            );
        }
        assert!(first > 0, "a cube has extraordinary vertices, so something must be cached");
    }

    #[test]
    fn tessellate_welds_shared_boundaries_into_one_manifold_mesh() {
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let levels = 2;
        let Tessellation { positions, normals, triangles, .. } = surface.tessellate(levels);

        let n = 1usize << levels;
        let unwelded_upper_bound = cage.face_count() * (n + 1) * (n + 1);
        assert!(
            positions.len() < unwelded_upper_bound,
            "expected shared boundary vertices to be welded, got {} vertices for {} faces at \
             n={n}",
            positions.len(),
            cage.face_count()
        );
        assert_eq!(positions.len(), normals.len());

        // Every undirected edge of a closed manifold mesh belongs to exactly two triangles.
        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &triangles {
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }
        assert!(
            edge_count.values().all(|&c| c == 2),
            "a closed surface's tessellation must have exactly two triangles per edge"
        );
    }

    #[test]
    fn tessellate_normals_are_unit_length_and_outward_facing() {
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let Tessellation { positions, normals, .. } = surface.tessellate(2);

        for (p, n) in positions.iter().zip(&normals) {
            assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-9);
            // Cage::cube() is centred at the origin and its limit surface is convex, so the
            // outward normal and the position vector from the centre must point the same way.
            assert!(
                n.dot(&p.coords) > 0.0,
                "normal at {p:?} should point outward, got {n:?}"
            );
        }
    }

    #[test]
    fn tessellate_normals_vary_across_a_triangle_rather_than_repeating_the_facet_normal() {
        // The whole point of this method: a triangle's three analytic normals should disagree,
        // tracking the surface's curvature, instead of all equalling that triangle's flat facet
        // normal the way finite-differencing a coarse triangulation does.
        let cage = Cage::cube();
        let mut surface = LimitSurface::new(&cage);
        let Tessellation { normals, triangles, .. } = surface.tessellate(2);

        let varying = triangles
            .iter()
            .filter(|t| {
                let a = normals[t[0] as usize];
                let b = normals[t[1] as usize];
                let c = normals[t[2] as usize];
                let max_angle = a.angle(&b).max(a.angle(&c)).max(b.angle(&c));
                max_angle > 1e-4
            })
            .count();

        assert!(
            varying > triangles.len() / 2,
            "expected most triangles on a curved surface to have non-facet-flat normals, got \
             {varying} of {}",
            triangles.len()
        );
    }
}
