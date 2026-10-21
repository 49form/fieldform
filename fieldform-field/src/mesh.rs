// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Meshes any [`ScalarField`] into triangles, so a composed field can be drawn.
//!
//! # Why this exists rather than using `fidget-mesh`
//!
//! Fidget can mesh, but it drives its own sampling of a `Tree`, and our fields are not
//! trees: a cage limit surface answers by closest-point search against geometry, which no
//! expression can be written for. So the mesher has to be able to call our own `distance`,
//! and that means meshing here — which is the same reason the viewport generates its own
//! WGSL rather than handing an expression to somebody else's evaluator.
//!
//! # Dual contouring, not marching cubes
//!
//! Marching cubes places vertices *on* cell edges, which produces slivers wherever the surface
//! grazes a cell, and needs a 256-case table to stay manifold. The dual family places one
//! vertex per straddling cell and connects them across grid edges: manifold by construction,
//! better-shaped triangles, and the whole algorithm is the two loops below.
//!
//! Which dual method is the difference between a clean boolean and a chewed one. Plain Surface
//! Nets puts that vertex at the *average* of the cell's edge crossings, which is right on a
//! smooth patch and badly wrong on a crease: where a subtracted pipe meets the outer surface,
//! the crossings belong to two differently-oriented surfaces and their mean sits off both, in
//! mid-air inside the notch. This uses Dual Contouring instead (Ju et al., *Dual Contouring of
//! Hermite Data*), which fits the crossings' tangent *planes* rather than averaging their
//! positions — see [`solve_qef`]. A crease is where two families of planes intersect, so the
//! solution lands on it rather than between the two surfaces.
//!
//! Normals come from the field's gradient rather than from face normals. That matters for the
//! same reason it did on the limit surface: gradient normals depend only on the field, so they
//! are smooth across cell boundaries, where averaged face normals would show the sampling grid.

use crate::ScalarField;
use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

/// A triangle mesh with per-vertex normals, in canonical millimetres.
pub struct FieldMesh {
    pub positions: Vec<Point3<f64>>,
    pub normals: Vec<Vector3<f64>>,
    pub triangles: Vec<[u32; 3]>,
}

impl FieldMesh {
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

/// The 12 edges of a cell, as pairs of corner indices in the `(x, y, z)` bit order used below.
const CELL_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7), // along x
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7), // along y
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // along z
];

/// Meshes the zero level set of `field` inside `bounds`.
///
/// `resolution` is the number of cells along the longest axis; the others get proportionally
/// many, so cells stay cubic and a long thin part is not sampled coarsely across its thickness.
///
/// Returns an empty mesh if the surface does not cross the box at all, which is a legitimate
/// answer — an empty boolean result, say — and not an error.
pub fn mesh_field(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    resolution: usize,
) -> FieldMesh {
    let (lo, hi) = bounds;
    let extent = hi - lo;
    let longest = extent.x.max(extent.y).max(extent.z);
    if longest <= 0.0 || resolution == 0 {
        return FieldMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            triangles: Vec::new(),
        };
    }

    // Cell size from the longest axis, then at least one cell on every axis.
    let step = longest / resolution as f64;
    // The finite-difference step for fields with no analytic normal, used both when placing
    // vertices and when shading them. A half cell: small enough to track curvature, large
    // enough that a field with no analytic normal blends across more than one triangle of its
    // own underlying tessellation rather than returning a single facet.
    let h = step * 0.5;
    let dims = [
        ((extent.x / step).ceil() as usize).max(1),
        ((extent.y / step).ceil() as usize).max(1),
        ((extent.z / step).ceil() as usize).max(1),
    ];
    let corners = [dims[0] + 1, dims[1] + 1, dims[2] + 1];

    let at = |i: usize, j: usize, k: usize| {
        Point3::new(
            lo.x + i as f64 * step,
            lo.y + j as f64 * step,
            lo.z + k as f64 * step,
        )
    };
    let corner_index = |i: usize, j: usize, k: usize| (k * corners[1] + j) * corners[0] + i;

    // --- decide which blocks can contain a surface at all ---
    //
    // Sampling every corner is the whole cost of meshing, and most of it is wasted: a surface
    // occupies a thin shell, while the grid covers a volume. `ScalarField::interval` bounds the
    // field over a box, so a block whose bound does not straddle zero is provably all-inside or
    // all-outside and has no crossing anywhere in it.
    //
    // The bound is taken over the block **expanded by one cell**, and that margin is what makes
    // skipping safe rather than merely plausible. A skipped block's corners are filled with a
    // constant of the right sign, so the sign is exact but the magnitude is not — and magnitude
    // decides where along an edge a crossing is placed. Corners on the seam between a skipped
    // block and a live one are read by cells belonging to both. Expanding by a cell means any
    // block that passes the test also has no crossing in the ring of cells touching it, so the
    // wrong magnitudes are never read by anything that interpolates.
    //
    // Concretely, the margin excludes exactly the risky blocks. A skipped block has
    // `bound_lo > 0`, and a 1-Lipschitz field cannot fall below zero within one cell of it
    // unless `bound_lo < step`. Widening the tested box by a cell drops `bound_lo` by about
    // that much, so those marginal blocks stop qualifying and the ones that still do have no
    // crossing anywhere near them.
    //
    // Stated honestly: that argument is reasoning, not measurement. Removing the margin does
    // not fail any test here, because the configuration it guards against needs a block whose
    // bound lands inside that one-cell band, and none of these fixtures produce one. It is kept
    // because it costs a few percent of the sampling it saves and closes the gap by
    // construction.
    //
    // One interval call replaces up to `(BLOCK + 1)^3` evaluations, so the test pays for itself
    // even when it fails most of the time. Block size measured on the app's default scene
    // (`mesh_rebuild_cost`): 4 cells beats 8 and 16, giving 1.55x at 64 cells and 2.15x at 128.
    const BLOCK: usize = 4;
    let block_counts = [
        dims[0].div_ceil(BLOCK).max(1),
        dims[1].div_ceil(BLOCK).max(1),
        dims[2].div_ceil(BLOCK).max(1),
    ];
    let block_index =
        |bi: usize, bj: usize, bk: usize| (bk * block_counts[1] + bj) * block_counts[0] + bi;

    // `None` where the block may contain a surface; `Some(v)` where it provably cannot, with
    // `v` a value of the correct sign to fill it with. Beside it, for the blocks that may, the
    // field **shortened to that block** — backlog item 17.
    //
    // Rejection and pruning are the same interval walk read twice: the bound says whether there
    // is a surface, and the trace of which branches could have produced it says what may be
    // deleted. A block that survives rejection is the one that still costs something, and this
    // is what makes it cost less: a pattern's other instances and a part's other bodies cannot
    // reach it, so they are not evaluated at its 125 corners. Built once per block and reused
    // for all of them, which is the amortisation `brick_timing` measures at 28x to 32x.
    let blocks: Vec<(Option<f64>, Option<std::sync::Arc<dyn ScalarField + Send + Sync>>)> =
        (0..block_counts[0] * block_counts[1] * block_counts[2])
        .into_par_iter()
        .map(|b| {
            let bi = b % block_counts[0];
            let bj = (b / block_counts[0]) % block_counts[1];
            let bk = b / (block_counts[0] * block_counts[1]);

            let lo_corner = at(bi * BLOCK, bj * BLOCK, bk * BLOCK);
            let hi_corner = at(
                ((bi + 1) * BLOCK).min(dims[0]),
                ((bj + 1) * BLOCK).min(dims[1]),
                ((bk + 1) * BLOCK).min(dims[2]),
            );
            let margin = Vector3::repeat(step);
            let (bound_lo, bound_hi) = field.interval(lo_corner - margin, hi_corner + margin);

            if bound_lo > 0.0 {
                (Some(bound_lo), None)
            } else if bound_hi < 0.0 {
                (Some(bound_hi), None)
            } else {
                // Shortened against the *padded* box, which is the one every corner this block
                // fills lies inside -- the same margin the rejection is taken over, and for the
                // same reason.
                (None, field.pruned_to(lo_corner - margin, hi_corner + margin))
            }
        })
        .collect();

    // --- sample the grid, skipping what cannot matter ---
    //
    // Parallel across z-slabs, which are independent and large enough that the split pays for
    // itself.
    let mut values = vec![0.0f64; corners[0] * corners[1] * corners[2]];
    values
        .par_chunks_mut(corners[0] * corners[1])
        .enumerate()
        .for_each(|(k, slab)| {
            let bk = (k / BLOCK).min(block_counts[2] - 1);
            for j in 0..corners[1] {
                let bj = (j / BLOCK).min(block_counts[1] - 1);
                for i in 0..corners[0] {
                    let bi = (i / BLOCK).min(block_counts[0] - 1);
                    let (fill, shortened) = &blocks[block_index(bi, bj, bk)];
                    slab[j * corners[0] + i] = match fill {
                        Some(v) => *v,
                        // The shortened field when there is one, and the whole field when the
                        // kind of field being meshed has no branches to drop.
                        None => match shortened {
                            Some(local) => local.distance(at(i, j, k)),
                            None => field.distance(at(i, j, k)),
                        },
                    };
                }
            }
        });

    // --- one vertex per cell the surface crosses ---
    //
    // Parallel across cells, because a cell's vertex depends only on the sampled corners around
    // it and on the field's normal at its own crossings — nothing a neighbouring cell decides.
    // The compaction afterwards is sequential only to keep vertex numbering deterministic.
    let cell_count = dims[0] * dims[1] * dims[2];
    let candidates: Vec<Option<Point3<f64>>> = (0..cell_count)
        .into_par_iter()
        .map(|cell| {
            let i = cell % dims[0];
            let j = (cell / dims[0]) % dims[1];
            let k = cell / (dims[0] * dims[1]);

            // Corner c has bit 0 = +x, bit 1 = +y, bit 2 = +z.
            let mut corner_values = [0.0f64; 8];
            for (c, value) in corner_values.iter_mut().enumerate() {
                *value = values[corner_index(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))];
            }

            let inside = corner_values.iter().filter(|v| **v <= 0.0).count();
            if inside == 0 || inside == 8 {
                return None; // wholly outside or wholly inside: no surface here
            }

            // **And this pass is where the evaluations are** -- backlog item 17, at the place it
            // actually pays. Sampling the corners is one reading each; placing a vertex bisects
            // every crossing edge and then differences a normal at it, which is a dozen readings
            // per cell and only ever on the shell. So the shortened field matters more here than
            // it does above, and measurably: wiring only the corner pass moved nothing.
            let (_, shortened) = &blocks[block_index(
                (i / BLOCK).min(block_counts[0] - 1),
                (j / BLOCK).min(block_counts[1] - 1),
                (k / BLOCK).min(block_counts[2] - 1),
            )];
            let field: &(dyn ScalarField + Sync) = match shortened {
                Some(local) => local.as_ref(),
                None => field,
            };

            // Each sign-changing edge contributes one point on the surface and the surface's
            // orientation there — the "Hermite data" dual contouring is named for. The mean of
            // the points is kept too, as the fallback for directions the planes leave free.
            let mut a = Matrix3::zeros();
            let mut b = Vector3::zeros();
            let mut mass = Vector3::zeros();
            let mut count = 0.0;
            for &(ea, eb) in CELL_EDGES.iter() {
                let (va, vb) = (corner_values[ea], corner_values[eb]);
                if (va <= 0.0) == (vb <= 0.0) {
                    continue;
                }
                let pa = at(i + (ea & 1), j + ((ea >> 1) & 1), k + ((ea >> 2) & 1));
                let pb = at(i + (eb & 1), j + ((eb >> 1) & 1), k + ((eb >> 2) & 1));
                let p = refine_crossing(field, pa, pb, va, vb);

                let n = surface_normal(field, p, h);
                a += n * n.transpose();
                b += n * n.dot(&p.coords);
                mass += p.coords;
                count += 1.0;
            }
            if count == 0.0 {
                return None;
            }

            let mass_point = Point3::from(mass / count);
            let vertex = solve_qef(a, b, mass_point);

            // Dual contouring's solution is unbounded — on a near-degenerate crease it can
            // land far outside the cell that produced it. That has to be clamped, and clamped
            // to the cell itself rather than to a generous margin around it: the quads are
            // stitched from the vertices of four *adjacent* cells, so a vertex that strays past
            // its own cell can overtake its neighbour's and turn the quad inside out, which is
            // the spray of stray triangles a crease otherwise grows along its rim.
            let lo_cell = at(i, j, k);
            Some(Point3::new(
                vertex.x.clamp(lo_cell.x, lo_cell.x + step),
                vertex.y.clamp(lo_cell.y, lo_cell.y + step),
                vertex.z.clamp(lo_cell.z, lo_cell.z + step),
            ))
        })
        .collect();

    let mut cell_vertex = vec![u32::MAX; cell_count];
    let mut positions: Vec<Point3<f64>> = Vec::new();
    for (cell, candidate) in candidates.iter().enumerate() {
        if let Some(p) = candidate {
            cell_vertex[cell] = positions.len() as u32;
            positions.push(*p);
        }
    }

    // --- normals, before the triangles rather than after ---
    //
    // They used to be computed at the end, since only the positions feed them. The winding
    // choice below needs them: a quad's two halves can face opposite ways, and the field's own
    // normal is the only thing that says which way is right. Moving them costs nothing -- the
    // positions are final by here.
    let normals: Vec<Vector3<f64>> = positions
        .par_iter()
        .map(|p| surface_normal(field, *p, h))
        .collect();

    // --- connect vertices across every grid edge that changes sign ---
    //
    // A sign-changing edge is shared by exactly four cells, and their vertices form the quad
    // that crosses it. Walking edges rather than cells is what keeps the result manifold
    // without a case table.
    let mut triangles = Vec::new();
    let cell_at = |i: usize, j: usize, k: usize| cell_vertex[(k * dims[1] + j) * dims[0] + i];
    // A Surface Nets quad joins four cell vertices that need not be coplanar, so its two halves
    // can face opposite ways. Which diagonal is chosen decides whether they do.
    //
    // **Chosen against the field's own normals rather than by shape.** Splitting along the
    // shorter diagonal is the usual advice and it is only a proxy: it fixed the worst of a fold
    // and left 4350 triangles wound inward on a 585 892-triangle export, which a flat-shaded
    // reader lights from the inside — the bright specks scattered over an otherwise smooth
    // surface. The normals say directly what the diagonal can only imply, and they are already
    // to hand.
    //
    // Both splits are scored by how many of their halves agree with the field, and the shorter
    // diagonal breaks a tie — so where the quad is nearly flat and the question does not
    // matter, this still does what it did before.
    let mut quad = |a: u32, b: u32, c: u32, d: u32, flip: bool| {
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || d == u32::MAX {
            return;
        }
        let p = |i: u32| positions[i as usize];
        // The ring in the order it should be wound, so the two splits below differ only in
        // which diagonal they cut and not in orientation.
        let r = if flip { [a, d, c, b] } else { [a, b, c, d] };

        // Positive when the triangle faces the way the field's gradient does at its corners.
        let agrees = |t: &[u32; 3]| {
            let (pa, pb, pc) = (p(t[0]), p(t[1]), p(t[2]));
            let facing = normals[t[0] as usize] + normals[t[1] as usize] + normals[t[2] as usize];
            (pb - pa).cross(&(pc - pa)).dot(&facing) > 0.0
        };

        let split_02 = [[r[0], r[1], r[2]], [r[0], r[2], r[3]]];
        let split_13 = [[r[0], r[1], r[3]], [r[1], r[2], r[3]]];
        let score = |s: &[[u32; 3]; 2]| s.iter().filter(|t| agrees(t)).count();
        let (s02, s13) = (score(&split_02), score(&split_13));

        let tris = if s02 != s13 {
            if s02 > s13 { split_02 } else { split_13 }
        } else if (p(r[2]) - p(r[0])).norm_squared() <= (p(r[3]) - p(r[1])).norm_squared() {
            split_02
        } else {
            split_13
        };

        // Emitted whole, both halves, whatever their area.
        //
        // A zero-area half used to be dropped, to stop a facet with no computable normal
        // reaching an STL. That was the wrong trade: a collapsed half still carries the
        // diagonal it shares with the other one, so dropping it left that edge used once and
        // opened a hole — measured at about two open edges per triangle dropped, on a mesh
        // that already had holes. A degenerate triangle rasterises to nothing and is invisible;
        // a hole is what stops a slicer.
        triangles.extend_from_slice(&tris);
    };

    // An edge only yields a quad when all four of its surrounding cells exist. Each guard
    // therefore bounds *three* axes, not one: the edge's own axis, and both perpendicular axes
    // at each end, since the loops run over corners (`dims + 1`) while cells run over `dims`.
    //
    // Bounding only the lower end — which is what these guards used to do — is wrong in a way
    // that hides. A surface strictly inside the sampled box never changes sign in the final
    // row, so the out-of-range index is never reached and every ordinary model meshes fine. A
    // surface that reaches the boundary crosses it immediately and panics, and a space-filling
    // field like a lattice reaches the boundary always. Tested below in both directions.
    //
    // Skipping leaves the mesh open where the surface leaves the box, which is what Surface
    // Nets does at any clipped region and what the `u32::MAX` guard above already implies.
    for k in 0..corners[2] {
        for j in 0..corners[1] {
            for i in 0..corners[0] {
                let v = values[corner_index(i, j, k)];

                // +x edge: shared by the four cells around it in the y/z plane.
                if i < dims[0] && j > 0 && j < dims[1] && k > 0 && k < dims[2] {
                    let n = values[corner_index(i + 1, j, k)];
                    if (v <= 0.0) != (n <= 0.0) {
                        quad(
                            cell_at(i, j - 1, k - 1),
                            cell_at(i, j, k - 1),
                            cell_at(i, j, k),
                            cell_at(i, j - 1, k),
                            v > 0.0,
                        );
                    }
                }
                // +y edge.
                if j < dims[1] && i > 0 && i < dims[0] && k > 0 && k < dims[2] {
                    let n = values[corner_index(i, j + 1, k)];
                    if (v <= 0.0) != (n <= 0.0) {
                        quad(
                            cell_at(i - 1, j, k - 1),
                            cell_at(i, j, k - 1),
                            cell_at(i, j, k),
                            cell_at(i - 1, j, k),
                            v <= 0.0,
                        );
                    }
                }
                // +z edge.
                if k < dims[2] && i > 0 && i < dims[0] && j > 0 && j < dims[1] {
                    let n = values[corner_index(i, j, k + 1)];
                    if (v <= 0.0) != (n <= 0.0) {
                        quad(
                            cell_at(i - 1, j - 1, k),
                            cell_at(i, j - 1, k),
                            cell_at(i, j, k),
                            cell_at(i - 1, j, k),
                            v > 0.0,
                        );
                    }
                }
            }
        }
    }

    // --- normals: analytic where the field can give one, finite-differenced otherwise ---
    //
    // Finite-differencing was the *only* option here, and it silently mimics whatever the
    // field's gradient actually is: for a field built from a triangulation — CageField — that
    // gradient is piecewise constant across each triangle, so any step small enough to track
    // curvature also lands inside one triangle and returns exactly its flat facet normal. No
    // step size fixes that; it needed a field able to say "here is my exact normal" instead of
    // being probed for one. ScalarField::normal is that path — see its docs, and
    // shading_normal_accuracy for the measured before/after.
    //
    // The same `surface_normal` the vertex placement above used, so a vertex is shaded by the
    // orientation it was positioned to satisfy rather than by a second, differently-derived
    // answer to the same question.
    FieldMesh {
        positions,
        normals,
        triangles,
    }
}

/// Where the field actually crosses zero along the cell edge `a`-`b`, found by querying it
/// rather than by interpolating between the two endpoint samples.
///
/// Interpolating assumes the field varies linearly along the edge. Away from features it very
/// nearly does, and the two agree. At a crease it does not, and the difference is not small:
/// on the edge running out of a box's corner, the distance stays pinned at whatever the
/// *nearest* face gives until the edge clears that face, so a straight line between the
/// endpoints crosses zero well before the surface does. That error lands directly in the
/// vertex, and no amount of care in [`solve_qef`] recovers from being told the wrong point —
/// which is why sharpening vertex placement without this barely moved the corner it was meant
/// to sharpen.
///
/// Regula falsi from the interpolated guess, keeping the sign change bracketed throughout, so
/// this can only improve on the guess and never wander off the edge. Twelve iterations: enough
/// for the pinned-then-rising shape above, where the first few only walk the bracket in.
pub(crate) fn refine_crossing(
    field: &(dyn ScalarField + Sync),
    a: Point3<f64>,
    b: Point3<f64>,
    va: f64,
    vb: f64,
) -> Point3<f64> {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let (mut v_lo, mut v_hi) = (va, vb);

    for _ in 0..12 {
        let t = lo + (hi - lo) * secant(v_lo, v_hi);
        let v = field.distance(a + (b - a) * t);
        if v == 0.0 {
            return a + (b - a) * t;
        }
        if (v <= 0.0) == (v_lo <= 0.0) {
            lo = t;
            v_lo = v;
        } else {
            hi = t;
            v_hi = v;
        }
    }

    a + (b - a) * (lo + (hi - lo) * secant(v_lo, v_hi))
}

/// Where a straight line from `v_lo` to `v_hi` crosses zero, as a fraction of the interval.
fn secant(v_lo: f64, v_hi: f64) -> f64 {
    if (v_hi - v_lo).abs() < f64::EPSILON {
        0.5
    } else {
        (-v_lo / (v_hi - v_lo)).clamp(0.0, 1.0)
    }
}

/// The surface's outward direction at `p`: the field's own analytic answer when it has one,
/// and a central difference of `distance` when it does not.
///
/// The fallback is not equivalent, only adequate — see [`ScalarField::normal`] for why a field
/// built from a triangulation cannot be finite-differenced into a smooth normal at any step
/// size. Fields that can answer exactly should, and this routes to that answer wherever it
/// exists.
pub(crate) fn surface_normal(field: &(dyn ScalarField + Sync), p: Point3<f64>, h: f64) -> Vector3<f64> {
    if let Some(n) = field.normal(p) {
        return n;
    }
    let d = |o: Vector3<f64>| field.distance(p + o);
    Vector3::new(
        d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
        d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
        d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
    )
    .try_normalize(1e-12)
    .unwrap_or_else(Vector3::z)
}

/// The point minimising its total squared distance to the planes gathered from a cell's edge
/// crossings — dual contouring's quadratic error function, in normal-equation form
/// `a = sum(n nᵀ)`, `b = sum(n (n·p))`.
///
/// The catch is that `a` is usually rank-deficient, and deliberately so: on a flat patch every
/// plane is parallel, which pins the vertex along the normal and says nothing about where it
/// sits *within* the surface. Inverting that is meaningless, and a naive solve throws the
/// vertex to infinity along the unconstrained directions. So the singular values below a
/// relative cutoff are dropped and `bias` — the mean of the crossings — supplies those
/// directions instead. The result is that a flat patch reproduces plain Surface Nets exactly,
/// a crease is pinned to the line where its two plane families meet, and a corner to the point
/// where three do, each getting only as much as the data actually determines.
pub(crate) fn solve_qef(a: Matrix3<f64>, b: Vector3<f64>, bias: Point3<f64>) -> Point3<f64> {
    let svd = a.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return bias;
    };
    let largest = svd.singular_values.max();
    if largest <= 0.0 {
        return bias;
    }

    // A tenth of a percent of the largest singular value. Anything below this is a direction
    // the crossings agree on only to within noise, and treating it as real is what produces
    // the wild vertices that clamping then has to catch.
    let cutoff = largest * 1e-3;
    let mut inverted = Matrix3::zeros();
    for i in 0..3 {
        if svd.singular_values[i] > cutoff {
            inverted[(i, i)] = 1.0 / svd.singular_values[i];
        }
    }

    // Solved as a correction to `bias` rather than from the origin, which is what confines the
    // pseudo-inverse's reach to the directions that survived the cutoff.
    let pseudo_inverse = v_t.transpose() * inverted * u.transpose();
    bias + pseudo_inverse * (b - a * bias.coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere, as the simplest field whose exact answer is known everywhere.
    struct Sphere {
        radius: f64,
    }
    impl ScalarField for Sphere {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.coords.norm() - self.radius
        }
    }

    fn sphere_mesh(radius: f64, resolution: usize) -> FieldMesh {
        let r = radius * 1.5;
        mesh_field(
            &Sphere { radius },
            (Point3::new(-r, -r, -r), Point3::new(r, r, r)),
            resolution,
        )
    }

    #[test]
    fn a_sphere_meshes_to_something() {
        let mesh = sphere_mesh(10.0, 24);
        assert!(!mesh.is_empty());
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        for t in &mesh.triangles {
            for &i in t {
                assert!((i as usize) < mesh.positions.len(), "index out of range");
            }
        }
    }

    /// A cube built as three slabs intersected, whose creases run along grid planes.
    ///
    /// The alignment is the point. A tilted cube produces almost no collapsed quads; this one
    /// produces thousands, and a lattice sitting on the export grid is this case rather than
    /// the tilted one.
    struct SlabCube;
    impl ScalarField for SlabCube {
        fn distance(&self, p: Point3<f64>) -> f64 {
            (p.x.abs() - 10.0)
                .max(p.y.abs() - 10.0)
                .max(p.z.abs() - 10.0)
        }
        fn normalization(&self) -> crate::Normalization {
            crate::Normalization::LowerBound
        }
    }

    /// A crease along a grid plane must not tear the surface.
    ///
    /// Two cells straddling such a crease both clamp their vertex onto the plane between them,
    /// so a quad's two halves can collapse to a line. Zero-area triangles used to be dropped
    /// here, on the grounds that a facet with no computable normal has no business in an STL.
    /// That was the wrong trade and this test is what it became: a collapsed half still carries
    /// the diagonal it shares with the other one, so dropping it left that edge used once and
    /// tore a hole — measured at about two open edges per triangle dropped. A degenerate
    /// triangle rasterises to nothing; a hole is what stops a slicer.
    ///
    /// So the invariant asserted is the one that matters. Non-manifold edges are *not* checked
    /// here: one vertex per cell cannot represent two sheets crossing one cell, and that is a
    /// separate, known problem with its own fix.
    #[test]
    fn a_crease_along_a_grid_plane_leaves_no_hole() {
        let r = 16.0;
        let mesh = mesh_field(
            &SlabCube,
            (Point3::new(-r, -r, -r), Point3::new(r, r, r)),
            48,
        );
        assert!(!mesh.is_empty(), "the cube should mesh to something");

        let mut uses: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        for t in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let torn = uses.values().filter(|n| **n == 1).count();
        assert_eq!(torn, 0, "{torn} edges belong to only one triangle");
    }

    /// A plane through the middle of the box, so the surface reaches the sampling boundary on
    /// every side. The simplest field that does.
    struct Plane;
    impl ScalarField for Plane {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.x
        }
        fn normal(&self, _p: Point3<f64>) -> Option<Vector3<f64>> {
            Some(Vector3::x())
        }
    }

    /// A field that fills the whole sampled region, in the manner of a lattice.
    struct Slabs;
    impl ScalarField for Slabs {
        fn distance(&self, p: Point3<f64>) -> f64 {
            // Parallel walls, so sign changes occur right up to and through every boundary
            // face rather than only on one of them.
            (p.x.sin() * 0.3).abs() - 0.1
        }
    }

    /// Meshing must not depend on the surface staying clear of the sampled box.
    ///
    /// This panicked at every resolution before the edge guards bounded all three axes, and it
    /// is not an edge case for the geometry this kernel exists to handle: a lattice fills its
    /// domain by definition, so it reaches the boundary always.
    #[test]
    fn a_surface_reaching_the_sampling_boundary_meshes_instead_of_panicking() {
        let bounds = (Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));

        for resolution in [8, 16, 33, 64] {
            let plane = mesh_field(&Plane, bounds, resolution);
            assert!(
                plane.triangle_count() > 0,
                "a plane through the box should produce a surface at resolution {resolution}"
            );

            let slabs = mesh_field(&Slabs, bounds, resolution);
            assert!(
                slabs.triangle_count() > 0,
                "a space-filling field should produce a surface at resolution {resolution}"
            );
        }
    }

    /// The fix skips boundary edges rather than clamping them, so a clipped surface is left
    /// open where it leaves the box. Asserting that keeps a later "tidy up the boundary"
    /// change from quietly reintroducing the out-of-range index it was hiding.
    #[test]
    fn a_clipped_surface_is_left_open_where_it_leaves_the_box() {
        use std::collections::HashMap;

        let bounds = (Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let mesh = mesh_field(&Plane, bounds, 16);

        let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &mesh.triangles {
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edge_uses
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_insert(0) += 1;
            }
        }
        let boundary = edge_uses.values().filter(|&&c| c == 1).count();
        let folded = edge_uses.values().filter(|&&c| c > 2).count();

        assert!(boundary > 0, "a plane clipped by the box has an open rim");
        assert_eq!(
            folded, 0,
            "no edge may be shared by more than two triangles"
        );
    }

    /// Empty-space skipping must not change the answer, only the time taken to reach it.
    ///
    /// A small sphere in a large box is the case where nearly every block is skipped, so the
    /// fast path is doing almost all the work — and it is exactly the case the existing tests
    /// do not cover, since they use surfaces that fill their box. Vertices are checked against
    /// the true surface, so a skipped block that wrongly suppressed a crossing shows up as a
    /// hole and a skipped block that wrongly kept one shows up as a stray vertex far off the
    /// sphere.
    #[test]
    fn skipping_empty_space_does_not_move_the_surface() {
        let radius = 6.0;
        let sphere = Sphere { radius };
        // 40x the radius across, so the sphere occupies well under a thousandth of the volume.
        let bounds = (
            Point3::new(-120.0, -120.0, -120.0),
            Point3::new(120.0, 120.0, 120.0),
        );

        let mesh = mesh_field(&sphere, bounds, 96);
        assert!(mesh.triangle_count() > 0, "the sphere must still be found");

        let step = 240.0 / 96.0;
        let worst = mesh
            .positions
            .iter()
            .map(|p| sphere.distance(*p).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < step,
            "every vertex should sit within a cell of the sphere, worst was {worst:.3} \
             against a {step:.3} mm cell"
        );

        // A closed surface has no boundary edge. A block wrongly skipped mid-surface would
        // punch a hole and show up here rather than in the position check above.
        use std::collections::HashMap;
        let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &mesh.triangles {
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edge_uses
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_insert(0) += 1;
            }
        }
        assert_eq!(
            edge_uses.values().filter(|&&c| c != 2).count(),
            0,
            "a sphere well inside its box should mesh closed"
        );
    }

    /// A box, whose edges and corners are the sharp features a dual mesher either keeps or
    /// rounds off. Deliberately not axis-aligned with anything: the half-extents differ per
    /// axis so no face lands on a tidy multiple of the cell size.
    struct Box3 {
        half: Vector3<f64>,
    }
    impl ScalarField for Box3 {
        fn distance(&self, p: Point3<f64>) -> f64 {
            let q = self.corner_offset(p);
            let outside = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
            outside + q.x.max(q.y).max(q.z).min(0.0)
        }

        /// Given exactly, because dual contouring is only as good as the normals it is fed —
        /// see the corner test below for what happens without this.
        fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
            let q = self.corner_offset(p);
            let sign = Vector3::new(
                if p.x < 0.0 { -1.0 } else { 1.0 },
                if p.y < 0.0 { -1.0 } else { 1.0 },
                if p.z < 0.0 { -1.0 } else { 1.0 },
            );
            if q.x > 0.0 || q.y > 0.0 || q.z > 0.0 {
                // Outside: away from the closest point on the box, which for a corner region
                // is the corner itself and for a face region is that face's axis.
                let outward = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
                return outward.component_mul(&sign).try_normalize(1e-12);
            }
            // On or inside: the nearest face, which is the axis whose offset is largest.
            let axis = if q.x >= q.y && q.x >= q.z {
                Vector3::x()
            } else if q.y >= q.z {
                Vector3::y()
            } else {
                Vector3::z()
            };
            Some(axis.component_mul(&sign))
        }
    }

    impl Box3 {
        /// Per-axis distance outside the slab, negative inside it.
        fn corner_offset(&self, p: Point3<f64>) -> Vector3<f64> {
            Vector3::new(
                p.x.abs() - self.half.x,
                p.y.abs() - self.half.y,
                p.z.abs() - self.half.z,
            )
        }
    }

    /// The test that separates dual contouring from plain Surface Nets, and the one that would
    /// have caught the chewed boolean rims directly.
    ///
    /// A box's corner is where three faces meet. Averaging a cell's edge crossings puts the
    /// vertex at their mean, which on a corner cell sits well inside the corner — the feature
    /// gets rounded off by something on the order of the cell size, and no refinement removes
    /// it, since the error is always a fixed fraction of a cell. Fitting the crossing planes
    /// instead has a unique solution at a corner: the point where all three meet, which is the
    /// corner itself. So the meshed box should reach its true corners to far better than a
    /// cell, and that is what fails if the vertex placement ever regresses to averaging.
    #[test]
    fn a_box_keeps_its_corners_rather_than_rounding_them_off() {
        let half = Vector3::new(7.31, 5.14, 9.07);
        let bound = 12.0;
        let resolution = 24;
        let cell = 2.0 * bound / resolution as f64;

        let mesh = mesh_field(
            &Box3 { half },
            (
                Point3::new(-bound, -bound, -bound),
                Point3::new(bound, bound, bound),
            ),
            resolution,
        );

        let corner = Point3::new(half.x, half.y, half.z);
        let reached = mesh
            .positions
            .iter()
            .map(|p| (p - corner).norm())
            .fold(f64::INFINITY, f64::min);

        assert!(
            reached < cell * 0.01,
            "the nearest vertex to the corner {corner:?} was {reached:.4} mm away, more than a \
             hundredth of a {cell:.3} mm cell — the corner is being rounded off. Both halves of \
             the reconstruction have to be right for this to pass: solving for the vertex rather \
             than averaging (averaging misses the corner by about 0.3 of a cell), fed with true \
             crossing points rather than interpolated ones (which miss by about 0.15)"
        );
    }

    /// The real accuracy test: every vertex should sit on the surface. A dual mesher places one
    /// vertex per straddling cell, so they land near but not exactly on it — the error should
    /// scale with cell size, and shrink when the grid is refined.
    #[test]
    fn vertices_lie_on_the_surface_to_within_a_cell() {
        for (resolution, allowed) in [(16, 1.2), (32, 0.6), (64, 0.3)] {
            let mesh = sphere_mesh(10.0, resolution);
            let worst = mesh
                .positions
                .iter()
                .map(|p| (p.coords.norm() - 10.0).abs())
                .fold(0.0f64, f64::max);
            assert!(
                worst <= allowed,
                "at resolution {resolution} the worst vertex was {worst} mm off the sphere"
            );
        }
    }

    /// Gradient normals must point outward and be genuinely normal — on a sphere the normal at
    /// any point is the radial direction, which makes this checkable exactly.
    #[test]
    fn normals_point_outward_along_the_radius() {
        let mesh = sphere_mesh(10.0, 32);
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            let radial = p.coords.normalize();
            assert!(
                n.dot(&radial) > 0.99,
                "normal {n:?} disagrees with the radial direction {radial:?}"
            );
        }
    }

    /// Manifold-ness, stated as the property that actually matters downstream: every triangle
    /// edge is shared by exactly two triangles. Surface Nets gives this by construction, so a
    /// failure here means the edge-walk or the winding is wrong.
    #[test]
    fn every_interior_edge_is_shared_by_exactly_two_triangles() {
        let mesh = sphere_mesh(10.0, 20);
        let mut counts: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        for t in &mesh.triangles {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = if e.0 < e.1 { e } else { (e.1, e.0) };
                *counts.entry(key).or_default() += 1;
            }
        }
        let bad = counts.values().filter(|c| **c != 2).count();
        assert_eq!(
            bad, 0,
            "{bad} edges were not shared by exactly two triangles"
        );
    }

    /// Winding has to be consistent, or the renderer's back-face culling shows holes. With
    /// outward normals and counter-clockwise winding, each triangle's geometric normal should
    /// agree with the field gradient there.
    ///
    /// Degenerate triangles are excluded, and that exclusion is the interesting part. Two
    /// *different* cells can place their vertices at the identical position where the grid is
    /// symmetric about the surface — at a sphere's poles, exactly. The resulting triangle has
    /// three distinct indices but zero area, so it has no winding to be right or wrong about,
    /// and it renders as nothing. It cannot simply be dropped: the indices are distinct, so
    /// removing the triangle would leave its neighbours' edges used once and break the
    /// manifold property the test above checks.
    #[test]
    fn triangle_winding_agrees_with_the_outward_normal() {
        let mesh = sphere_mesh(10.0, 24);
        let mut checked = 0;
        let mut degenerate = 0;

        for t in &mesh.triangles {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            let geometric = (b - a).cross(&(c - a));
            if geometric.norm() < 1e-12 {
                degenerate += 1;
                continue;
            }
            let outward = (a.coords + b.coords + c.coords).normalize();
            assert!(
                geometric.dot(&outward) > 0.0,
                "triangle {t:?} is wound inward: {a:?} {b:?} {c:?}"
            );
            checked += 1;
        }

        assert!(checked > 0, "no triangles were checked");
        // A handful at the poles is expected; a large share would mean the vertex placement
        // is collapsing, which is a real defect rather than a symmetry artefact.
        assert!(
            degenerate * 100 < mesh.triangles.len() * 5,
            "{degenerate} of {} triangles are degenerate, which is more than symmetry explains",
            mesh.triangles.len()
        );
    }

    /// A field that never changes sign inside the box has no surface in it. That is an answer,
    /// not a failure — an empty boolean result must not panic or emit stray geometry.
    #[test]
    fn a_field_with_no_surface_in_the_box_meshes_to_nothing() {
        let mesh = mesh_field(
            &Sphere { radius: 1.0 },
            (Point3::new(50.0, 50.0, 50.0), Point3::new(60.0, 60.0, 60.0)),
            16,
        );
        assert!(mesh.is_empty());
    }
}
