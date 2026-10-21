// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What an exact NURBS closest-point costs, against the two approximations in use today.
//!
//! # The question
//!
//! Backlog item 19 wants an imported STEP solid to *be* its trimmed NURBS surfaces rather than
//! a triangle mesh sampled from them. Whether that is a week or a month turns on one number:
//! how long it takes to ask a surface "how far away are you?", against how long the current
//! path takes to answer the same question approximately.
//!
//! # Why this measures our own evaluator rather than OCCT's
//!
//! OCCT can answer this and is already linked in for the importer, so timing its projection
//! looks like the obvious thing. It is the wrong number. The viewport is field-traced on the
//! **GPU**, where OCCT cannot go at all — so a shipped exact leaf would have to be an
//! evaluator of ours, in Rust and then in WGSL. Timing OCCT would measure a library that could
//! never be in the hot path.
//!
//! # What is deliberately left out
//!
//! **Trimming.** A real STEP face is a surface plus trim curves, and a point outside the trim
//! must measure to the trimmed boundary instead. That is work this does not model, so the
//! projection cost here is a *floor* rather than an estimate. It is the right floor: the inner
//! loop is the same, and if the floor is already too slow the rest cannot save it.
//!
//! **One patch, not a part.** A solid is hundreds of faces, and a query would test the ones
//! whose boxes are near — the same pruning the mesh path does over triangles. So the per-patch
//! number below is what a query costs *after* pruning, times however many patches survive it.

use fieldform_field::MeshField;
use nalgebra::{Point3, Vector3};

/// A bicubic B-spline surface over a uniform knot vector, which is what a STEP face usually is.
///
/// Uniform rather than general, so the basis functions are written out instead of going through
/// de Boor, and unweighted rather than rational. That makes this *faster* than a real NURBS
/// evaluator, which is the right direction for a floor: a number that says "too slow" here says
/// it for every real implementation too.
struct BSplineSurface {
    /// Control net, `nu` by `nv`.
    net: Vec<Point3<f64>>,
    nu: usize,
    nv: usize,
}

impl BSplineSurface {
    /// A patch with real curvature in both directions, so the projection has to iterate rather
    /// than landing on a plane immediately.
    fn bumpy(nu: usize, nv: usize, size: f64, amplitude: f64) -> Self {
        let mut net = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            for j in 0..nv {
                let x = (i as f64 / (nu - 1) as f64 - 0.5) * size;
                let y = (j as f64 / (nv - 1) as f64 - 0.5) * size;
                // Two incommensurate waves, so no row or column of the net comes out flat.
                let z = amplitude
                    * ((x / size * std::f64::consts::TAU * 0.9).sin()
                        + 0.6 * (y / size * std::f64::consts::TAU * 1.7).cos());
                net.push(Point3::new(x, y, z));
            }
        }
        Self { net, nu, nv }
    }

    /// The four cubic B-spline basis functions on a span, and their derivatives.
    fn basis(t: f64) -> ([f64; 4], [f64; 4]) {
        let s = 1.0 - t;
        let b = [
            s * s * s / 6.0,
            (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0,
            (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
            t * t * t / 6.0,
        ];
        let d = [
            -s * s / 2.0,
            (9.0 * t * t - 12.0 * t) / 6.0,
            (-9.0 * t * t + 6.0 * t + 3.0) / 6.0,
            t * t / 2.0,
        ];
        (b, d)
    }

    /// Spans in each direction. A cubic needs four control points per span.
    fn spans(&self) -> (usize, usize) {
        (self.nu - 3, self.nv - 3)
    }

    /// Position and both first derivatives at a parameter, which is what Newton needs.
    fn eval(&self, u: f64, v: f64) -> (Point3<f64>, Vector3<f64>, Vector3<f64>) {
        let (su, sv) = self.spans();
        let uu = (u.clamp(0.0, 1.0) * su as f64).min(su as f64 - 1e-12);
        let vv = (v.clamp(0.0, 1.0) * sv as f64).min(sv as f64 - 1e-12);
        let (iu, iv) = (uu.floor() as usize, vv.floor() as usize);
        let (bu, du) = Self::basis(uu - iu as f64);
        let (bv, dv) = Self::basis(vv - iv as f64);

        let mut p = Vector3::zeros();
        let mut d_u = Vector3::zeros();
        let mut d_v = Vector3::zeros();
        for a in 0..4 {
            for b in 0..4 {
                let c = self.net[(iu + a) * self.nv + (iv + b)].coords;
                p += c * (bu[a] * bv[b]);
                d_u += c * (du[a] * bv[b]);
                d_v += c * (bu[a] * dv[b]);
            }
        }
        // Chain rule for the span scaling, so the derivatives are with respect to u and v over
        // the whole patch rather than to the span-local parameter.
        (Point3::from(p), d_u * su as f64, d_v * sv as f64)
    }

    /// Closest point to `target`: Newton from the best of a coarse seed grid.
    ///
    /// The seed grid is not optional. Distance to a curved surface has local minima, and Newton
    /// from a fixed start converges to whichever one it began nearest — a confident answer that
    /// is not the closest point. Seeding is most of the cost, which is itself part of the
    /// finding.
    fn project(&self, target: Point3<f64>, seeds: usize) -> (f64, u32) {
        let mut best = (f64::INFINITY, 0.5, 0.5);
        for i in 0..seeds {
            for j in 0..seeds {
                let u = i as f64 / (seeds - 1) as f64;
                let v = j as f64 / (seeds - 1) as f64;
                let d = (self.eval(u, v).0 - target).norm_squared();
                if d < best.0 {
                    best = (d, u, v);
                }
            }
        }

        let (mut u, mut v) = (best.1, best.2);
        let mut iterations = 0;
        for _ in 0..24 {
            iterations += 1;
            let (p, du, dv) = self.eval(u, v);
            let r = p - target;
            // Minimising the squared distance: its gradient is 2*(r.Su, r.Sv), and Gauss-Newton
            // approximates the Hessian by dropping the second-derivative term.
            let (gu, gv) = (r.dot(&du), r.dot(&dv));
            let (a, b, c) = (du.dot(&du), du.dot(&dv), dv.dot(&dv));
            let det = a * c - b * b;
            if det.abs() < 1e-18 {
                break;
            }
            let step_u = (gu * c - gv * b) / det;
            let step_v = (gv * a - gu * b) / det;
            let (was_u, was_v) = (u, v);
            u = (u - step_u).clamp(0.0, 1.0);
            v = (v - step_v).clamp(0.0, 1.0);
            // The *achieved* move, not the proposed one. A point whose nearest spot is on the
            // patch edge has its step clamped away every iteration, so testing the proposed
            // step never terminates — it ran to the cap on nearly every probe, and the first
            // version of this spike reported 23.73 iterations against a limit of 24 while
            // looking like a converged answer.
            if (u - was_u).abs() + (v - was_v).abs() < 1e-10 {
                break;
            }
        }
        ((self.eval(u, v).0 - target).norm(), iterations)
    }

    /// The patch as a closed slab of triangles, which is what the importer produces today.
    ///
    /// Closed rather than a bare sheet because `MeshField` requires it, and requires it for a
    /// real reason: an open mesh has no inside for a sign to refer to. So the patch is given a
    /// flat underside and four walls. The extra triangles are a rounding error against the
    /// grid, and the query being measured is a nearest-triangle search either way.
    fn tessellate_slab(&self, n: usize, thickness: f64) -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
        let rows = n + 1;
        let mut positions = Vec::with_capacity(rows * rows * 2);
        // Top: the patch itself.
        for i in 0..=n {
            for j in 0..=n {
                positions.push(self.eval(i as f64 / n as f64, j as f64 / n as f64).0);
            }
        }
        // Bottom: the same grid dropped, so the walls are vertical.
        let base = positions[..rows * rows]
            .iter()
            .map(|p| Point3::new(p.x, p.y, p.z - thickness))
            .collect::<Vec<_>>();
        positions.extend(base);

        let top = |i: usize, j: usize| (i * rows + j) as u32;
        let bot = |i: usize, j: usize| (rows * rows + i * rows + j) as u32;
        let mut triangles = Vec::new();
        for i in 0..n {
            for j in 0..n {
                triangles.push([top(i, j), top(i + 1, j), top(i + 1, j + 1)]);
                triangles.push([top(i, j), top(i + 1, j + 1), top(i, j + 1)]);
                // Wound the other way, so the underside faces down.
                triangles.push([bot(i, j), bot(i + 1, j + 1), bot(i + 1, j)]);
                triangles.push([bot(i, j), bot(i, j + 1), bot(i + 1, j + 1)]);
            }
        }
        // The four walls, each a strip along one edge of the grid.
        for k in 0..n {
            let edges = [
                (top(0, k), top(0, k + 1), bot(0, k), bot(0, k + 1)),
                (top(n, k + 1), top(n, k), bot(n, k + 1), bot(n, k)),
                (top(k + 1, 0), top(k, 0), bot(k + 1, 0), bot(k, 0)),
                (top(k, n), top(k + 1, n), bot(k, n), bot(k + 1, n)),
            ];
            for (a, b, c, d) in edges {
                triangles.push([a, b, d]);
                triangles.push([a, d, c]);
            }
        }
        (positions, triangles)
    }
}

/// A narrow-band grid with trilinear lookup: what the GPU samples per march step.
struct Baked {
    values: Vec<f32>,
    n: usize,
    lo: Point3<f64>,
    step: f64,
}

impl Baked {
    fn sample(&self, p: Point3<f64>) -> f64 {
        let g = (p - self.lo) / self.step;
        let clamp = |v: f64| v.clamp(0.0, (self.n - 2) as f64);
        let (x, y, z) = (clamp(g.x), clamp(g.y), clamp(g.z));
        let (i, j, k) = (x as usize, y as usize, z as usize);
        let (fx, fy, fz) = (x - i as f64, y - j as f64, z - k as f64);
        let at = |i: usize, j: usize, k: usize| self.values[(i * self.n + j) * self.n + k] as f64;
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let c00 = lerp(at(i, j, k), at(i + 1, j, k), fx);
        let c01 = lerp(at(i, j, k + 1), at(i + 1, j, k + 1), fx);
        let c10 = lerp(at(i, j + 1, k), at(i + 1, j + 1, k), fx);
        let c11 = lerp(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), fx);
        lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz)
    }
}

/// Deterministic probe points spread through the box around the patch.
///
/// Deterministic on purpose: a spike whose numbers move between runs cannot be rechecked, which
/// is the whole value of writing one down.
fn probes(n: usize, extent: f64) -> Vec<Point3<f64>> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            Point3::new(
                (t * 0.7).sin() * extent,
                (t * 1.3).cos() * extent,
                (t * 2.1).sin() * extent * 0.5,
            )
        })
        .collect()
}

fn time<T>(label: &str, queries: usize, mut f: impl FnMut() -> T) -> f64 {
    let started = std::time::Instant::now();
    for _ in 0..queries {
        std::hint::black_box(f());
    }
    let per = started.elapsed().as_secs_f64() * 1e6 / queries as f64;
    println!("  {label:<44}{per:>9.3} us/query");
    per
}

fn main() {
    const SIZE: f64 = 100.0;
    let surface = BSplineSurface::bumpy(12, 12, SIZE, 8.0);
    let points = probes(2_000, SIZE * 0.8);

    println!("A bicubic B-spline patch, 12x12 control net, 100 mm across.\n");

    println!("Exact projection (Gauss-Newton, seeded on an NxN grid):");
    let mut exact_cost = 0.0;
    for seeds in [4usize, 8, 16] {
        let mut i = 0;
        let per = time(&format!("seed {seeds}x{seeds}"), points.len(), || {
            let r = surface.project(points[i % points.len()], seeds);
            i += 1;
            r
        });
        if seeds == 8 {
            exact_cost = per;
        }
    }

    // How wrong the cheapest seeding is, which is what decides whether it may be used at all.
    let mut worst_gap: f64 = 0.0;
    for p in points.iter().take(500) {
        worst_gap = worst_gap.max((surface.project(*p, 4).0 - surface.project(*p, 24).0).abs());
    }
    let iters: u32 = points
        .iter()
        .take(500)
        .map(|p| surface.project(*p, 8).1)
        .sum();
    println!("  worst disagreement, seed 4 against seed 24: {worst_gap:>12.4} mm");
    println!(
        "  mean Newton iterations after seeding:       {:>12.2}\n",
        iters as f64 / 500.0
    );

    println!("Today, for the same patch:");
    let (positions, triangles) = surface.tessellate_slab(64, 20.0);
    println!("  (tessellated to {} triangles)", triangles.len());
    let mesh = MeshField::from_mesh(positions, triangles).expect("the slab is closed");
    let mut i = 0;
    let mesh_cost = time("triangle mesh, BVH nearest", points.len(), || {
        let r = mesh.distance(points[i % points.len()]);
        i += 1;
        r
    });

    let n = 128;
    let lo = Point3::new(-SIZE, -SIZE, -SIZE * 0.6);
    let step = (SIZE * 2.0) / (n - 1) as f64;
    let mut values = vec![0f32; n * n * n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let p = lo + Vector3::new(a as f64, b as f64, c as f64) * step;
                values[(a * n + b) * n + c] = mesh.distance(p) as f32;
            }
        }
    }
    let baked = Baked {
        values,
        n,
        lo,
        step,
    };
    let mut i = 0;
    let baked_cost = time("baked narrow band, trilinear fetch", points.len(), || {
        let r = baked.sample(points[i % points.len()]);
        i += 1;
        r
    });

    println!("\nAgainst what the viewport samples per march step:");
    println!(
        "  exact projection is {:>9.0}x a baked fetch",
        exact_cost / baked_cost
    );
    println!(
        "  the BVH mesh is     {:>9.0}x a baked fetch",
        mesh_cost / baked_cost
    );
    println!(
        "  exact is            {:>9.1}x the BVH mesh",
        exact_cost / mesh_cost
    );

    // The two things the ratio above is actually deciding between, in seconds.
    //
    // Per pixel per frame is the viewport: a sphere trace takes tens of steps per ray, and
    // every step is one query. Per bake is the alternative — sample the true surface once into
    // the narrow band the GPU already reads, so the exactness is paid for at settle time and
    // the trace stays a texture fetch.
    let pixels = 1920.0 * 1080.0;
    let steps = 60.0;
    println!(
        "
What that buys, for one patch:"
    );
    println!(
        "  tracing exactly, 1920x1080 at {steps:.0} steps: {:>9.1} s/frame (one core)",
        pixels * steps * exact_cost * 1e-6,
    );
    let cells: f64 = 128.0 * 128.0 * 128.0;
    println!(
        "  baking exactly into a 128^3 narrow band:    {:>9.2} s (one core)",
        cells * exact_cost * 1e-6,
    );
    println!(
        "  the same bake from triangles, as today:     {:>9.2} s (one core)",
        cells * mesh_cost * 1e-6,
    );
}
