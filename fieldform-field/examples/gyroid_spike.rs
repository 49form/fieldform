// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Asks whether a 1 m^3 gyroid lattice can be sphere-traced directly, and what a step policy
//! that assumes a true distance field does to it.
//!
//! The lattice is 20x20x20 cells of 50 mm with 1 mm walls: roughly 124 m^2 of surface inside a
//! one-metre box, which is order 10^9 triangles at a wall-resolving edge length and order 10^10
//! narrow-band voxels. Neither is storable. But the field that produces it is six trig calls,
//! so nothing here stores anything - every image is traced straight at the expression.
//!
//! # The contract this example deliberately breaks
//!
//! [`ScalarField`] requires a **true** signed distance with unit gradient. A gyroid cannot
//! supply one. `sin(X)cos(Y) + sin(Y)cos(Z) + sin(Z)cos(X)` is a bounded-Lipschitz
//! pseudo-distance: its zero set is the surface and its sign is right, but its magnitude is in
//! no particular units and its gradient is nowhere near unit length. Dividing by a bound on
//! that gradient converts it into a *lower bound* on Euclidean distance, which is what sphere
//! tracing and interval pruning actually require - they need to never overestimate, not to be
//! exact.
//!
//! So this file traces the same lattice under two policies:
//!
//!   `Naive`      the raw value, treated as millimetres because the trait says it is.
//!   `Lipschitz`  the raw value divided by a proven bound on |grad F|.
//!
//! and renders both at two cell sizes, because whether the naive policy is merely slow or
//! silently wrong depends on the cell size - see [`Gyroid::lipschitz`].
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example gyroid_spike [-- <resolution> [--mesh-res N]]`

use fieldform_field::{Normalization, ScalarField, TpmsKind};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::f64::consts::TAU;

/// The case from the design discussion: a one-metre cube of 50 mm cells with 1 mm walls.
const BLOCK_MM: f64 = 1000.0;
const CELL_MM: f64 = 50.0;
const WALL_MM: f64 = 1.0;

/// A second lattice, taken from the measured crossover in [`sweep_tunnelling_threshold`]
/// rather than from the algebra.
///
/// What breaks the naive step policy is not a cell size but a *ratio*: a wall thin relative to
/// its cell. At 5 mm cells the naive policy is clean with 1 mm and 0.5 mm walls, tunnels on
/// 5.4% of rays at 0.25 mm, and on 24% at 0.1 mm. This pair is the far end of that, set
/// against the 50 mm / 1 mm case where the same policy is merely slow.
const FINE_CELL_MM: f64 = 5.0;
const FINE_WALL_MM: f64 = 0.1;

/// Typical |grad F| at the surface, as a multiple of `k`, used to convert a wall thickness in
/// millimetres into a threshold on |F|.
///
/// This is an approximation - the gradient varies over the surface - so the thickness it
/// produces is nominal, and `report_wall_thickness` measures what actually came out rather
/// than trusting it.
const SURFACE_GRADIENT_FACTOR: f64 = 1.5;

/// How many sphere-tracing steps a ray may take before it is abandoned. Matches the budget in
/// `raymarch_reference`, so step counts here are comparable to the ones there.
const STEP_BUDGET: u32 = 512;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: usize = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .and_then(|a| a.parse().ok())
        .unwrap_or(600);
    let height = width;
    let mesh_res: usize = args
        .iter()
        .position(|a| a == "--mesh-res")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    println!("block:  {BLOCK_MM} mm cube");
    println!(
        "cells:  {CELL_MM} mm ({0}x{0}x{0} = {1} cells), {WALL_MM} mm walls",
        (BLOCK_MM / CELL_MM) as usize,
        ((BLOCK_MM / CELL_MM) as usize).pow(3)
    );
    println!("image:  {width}x{height}, step budget {STEP_BUDGET}\n");

    let lattice = Lattice::new(CELL_MM, WALL_MM);

    // --- does the field describe what it claims to describe? ---
    report_gradient(&lattice.gyroid);
    report_wall_thickness(&lattice.gyroid);

    // --- the headline question: can it be traced at all, and what does it cost? ---
    println!("\n--- sphere tracing, Lipschitz-bounded steps ---");
    for view in View::all() {
        let camera = view.camera(lattice.half, width, height);
        render(
            &format!("gyroid-{}.png", view.slug()),
            &camera,
            width,
            height,
            &lattice,
            StepPolicy::Lipschitz,
            view.label(),
        );
    }

    // --- where the naive policy breaks, measured rather than asserted ---
    sweep_tunnelling_threshold(width, height);

    // --- and what that looks like, in a safe regime and a broken one ---
    println!("\n--- step policy, side by side ---");
    let mut tiles: Vec<Vec<[u8; 3]>> = Vec::new();
    for (cell, wall) in [(CELL_MM, WALL_MM), (FINE_CELL_MM, FINE_WALL_MM)] {
        let lattice = Lattice::new(cell, wall);
        // The detail lens, because a whole-block frame aliases the walls away and the whole
        // question here is whether individual walls are being hit or skipped.
        let camera = View::Detail.camera(lattice.half, width, height);
        for policy in [StepPolicy::Naive, StepPolicy::Lipschitz] {
            let label = format!("{cell} mm cells / {wall} mm walls, {}", policy.label());
            tiles.push(render(
                &format!("gyroid-{}mm-{}.png", cell as usize, policy.slug()),
                &camera,
                width,
                height,
                &lattice,
                policy,
                &label,
            ));
        }
    }
    let refs: Vec<&[[u8; 3]]> = tiles.iter().map(|t| t.as_slice()).collect();
    tile("gyroid-step-policy.png", &refs, width, height);
    println!(
        "\ngyroid-step-policy.png   {CELL_MM}/{WALL_MM} naive | {CELL_MM}/{WALL_MM} bounded | \
         {FINE_CELL_MM}/{FINE_WALL_MM} naive | {FINE_CELL_MM}/{FINE_WALL_MM} bounded"
    );

    // --- what the alternative would have cost ---
    count_meshing_cost(&lattice.gyroid, mesh_res);
}

/// A gyroid wall (a TPMS sheet of finite thickness), as a pseudo-distance field.
#[derive(Clone, Copy)]
struct Gyroid {
    /// Angular frequency: one full period per cell.
    k: f64,
    /// The threshold on |F| that produces the requested wall thickness.
    c: f64,
    /// A proven upper bound on |grad F|, the constant that converts the raw value into a
    /// lower bound on Euclidean distance.
    ///
    /// **Taken from `TpmsKind::gradient_bound` rather than restated here**, which is the whole
    /// reason this field is worth a comment. It used to carry its own `sqrt(6) k`, written from
    /// the same argument the kernel used — so when that argument was replaced by a tighter one
    /// the spike went on measuring the old policy and reported, correctly and uselessly, that
    /// nothing had changed. A spike that keeps its own copy of the number it is measuring
    /// cannot measure a change to it.
    ///
    /// This is also what makes the naive policy's safety depend on cell size. The raw value
    /// understates the true distance by a factor of at least `1 / (bound · k)`, so treating it
    /// as millimetres is conservative only while `bound · k <= 1`, i.e. while the cell is
    /// larger than `tau · bound`. Below that the raw value *exceeds* the distance to the
    /// nearest wall and the tracer steps straight through it — which is why the threshold moves
    /// with the bound, and why it is quoted from the bound rather than written out.
    lipschitz: f64,
}

impl Gyroid {
    fn new(cell_mm: f64, wall_mm: f64) -> Self {
        let k = TAU / cell_mm;
        Self {
            k,
            c: SURFACE_GRADIENT_FACTOR * k * wall_mm * 0.5,
            lipschitz: TpmsKind::Gyroid.gradient_bound() * k,
        }
    }

    fn f(&self, p: Point3<f64>) -> f64 {
        let (x, y, z) = (p.x * self.k, p.y * self.k, p.z * self.k);
        x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()
    }

    fn grad_f(&self, p: Point3<f64>) -> Vector3<f64> {
        let (x, y, z) = (p.x * self.k, p.y * self.k, p.z * self.k);
        Vector3::new(
            x.cos() * y.cos() - z.sin() * x.sin(),
            y.cos() * z.cos() - x.sin() * y.sin(),
            z.cos() * x.cos() - y.sin() * z.sin(),
        ) * self.k
    }

    /// The wall's pseudo-distance, in whatever units |F| happens to be in.
    fn raw(&self, p: Point3<f64>) -> f64 {
        self.f(p).abs() - self.c
    }

    /// A lower bound on the Euclidean distance to the wall, in millimetres.
    fn bounded(&self, p: Point3<f64>) -> f64 {
        self.raw(p) / self.lipschitz
    }

    fn normal(&self, p: Point3<f64>) -> Vector3<f64> {
        let g = self.grad_f(p) * self.f(p).signum();
        g.try_normalize(1e-18).unwrap_or_else(Vector3::z)
    }
}

/// The gyroid wall intersected with the block - the whole model.
struct Lattice {
    gyroid: Gyroid,
    half: Vector3<f64>,
    /// Kept so the checks can express their tolerances in wall thicknesses, which is the only
    /// scale at which "passed through a wall" means anything.
    wall: f64,
}

impl Lattice {
    fn new(cell_mm: f64, wall_mm: f64) -> Self {
        Self {
            gyroid: Gyroid::new(cell_mm, wall_mm),
            half: Vector3::repeat(BLOCK_MM * 0.5),
            wall: wall_mm,
        }
    }

    /// The composed distance under a given step policy. The box term is a true millimetre
    /// distance either way; only the gyroid term's units are in question.
    fn distance_under(&self, p: Point3<f64>, policy: StepPolicy) -> f64 {
        let g = match policy {
            StepPolicy::Naive => self.gyroid.raw(p),
            StepPolicy::Lipschitz => self.gyroid.bounded(p),
        };
        g.max(box_sdf(p, self.half))
    }

    fn normal(&self, p: Point3<f64>) -> Vector3<f64> {
        // Whichever term is active at `p` owns the surface there.
        if self.gyroid.bounded(p) >= box_sdf(p, self.half) {
            self.gyroid.normal(p)
        } else {
            box_normal(p, self.half)
        }
    }
}

/// So the lattice can be handed to `mesh_field` for the cost comparison. The magnitude is a
/// lower bound rather than a true distance, which is exactly the trait gap this spike exists
/// to expose; Surface Nets only needs the sign and the zero crossing, so it is unharmed.
impl ScalarField for Lattice {
    fn distance(&self, p: Point3<f64>) -> f64 {
        self.distance_under(p, StepPolicy::Lipschitz)
    }

    /// The declaration this whole example exists to justify. Dividing by the provable bound
    /// leaves the value at 58-71% of the true distance, measured above, so it is safe to trace
    /// and to prune with, and wrong to blend against.
    fn normalization(&self) -> Normalization {
        Normalization::LowerBound
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        Some(Lattice::normal(self, p))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum StepPolicy {
    Naive,
    Lipschitz,
}

impl StepPolicy {
    fn label(self) -> &'static str {
        match self {
            StepPolicy::Naive => "naive (raw value as mm)",
            StepPolicy::Lipschitz => "Lipschitz-bounded",
        }
    }
    fn slug(self) -> &'static str {
        match self {
            StepPolicy::Naive => "naive",
            StepPolicy::Lipschitz => "bounded",
        }
    }
}

#[derive(Clone, Copy)]
enum View {
    /// Three-quarter, off-axis, whole block: most rays meet a wall within the first cell.
    Corner,
    /// Straight down an axis, where the lattice's channels line up with the ray and a ray can
    /// travel far without meeting anything. The worst case for step count.
    Axis,
    /// The same corner direction through a long lens, framing about a cell and a half.
    ///
    /// This is the control on the other two. At whole-block framing a 1 mm wall is a third of
    /// a pixel wide, so what those images show is a point-sampling artefact of a subpixel
    /// periodic structure, not the surface. Only a framing that resolves the wall shows what
    /// the field actually contains.
    Detail,
}

impl View {
    fn all() -> [View; 3] {
        [View::Corner, View::Axis, View::Detail]
    }
    fn slug(self) -> &'static str {
        match self {
            View::Corner => "corner",
            View::Axis => "axis",
            View::Detail => "detail",
        }
    }
    fn label(self) -> &'static str {
        match self {
            View::Corner => "corner view (off-axis, whole block)",
            View::Axis => "axis view (down the channels)",
            View::Detail => "detail view (long lens, walls resolved)",
        }
    }
    fn direction(self) -> Vector3<f64> {
        match self {
            View::Corner | View::Detail => Vector3::new(1.0, 0.7, 0.55).normalize(),
            View::Axis => Vector3::x(),
        }
    }
    /// A narrow field of view zooms in without moving the eye inside the lattice.
    fn fov_deg(self) -> f64 {
        match self {
            View::Corner | View::Axis => 45.0,
            View::Detail => 5.0,
        }
    }
    fn camera(self, half: Vector3<f64>, width: usize, height: usize) -> Camera {
        Camera::framing(half, self.direction(), self.fov_deg(), width, height)
    }
}

/// Exact signed distance to an origin-centred box.
fn box_sdf(p: Point3<f64>, half: Vector3<f64>) -> f64 {
    let q = Vector3::new(p.x.abs() - half.x, p.y.abs() - half.y, p.z.abs() - half.z);
    let outside = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

fn box_normal(p: Point3<f64>, half: Vector3<f64>) -> Vector3<f64> {
    let q = Vector3::new(p.x.abs() - half.x, p.y.abs() - half.y, p.z.abs() - half.z);
    // The face whose plane `p` is closest to is the one it lies on.
    if q.x >= q.y && q.x >= q.z {
        Vector3::new(p.x.signum(), 0.0, 0.0)
    } else if q.y >= q.z {
        Vector3::new(0.0, p.y.signum(), 0.0)
    } else {
        Vector3::new(0.0, 0.0, p.z.signum())
    }
}

/// Reports |grad F| over the surface, as a multiple of `k`.
///
/// Two numbers matter. The maximum checks the analytic bound in [`Gyroid::lipschitz`] is
/// actually an upper bound - if a sample exceeds it, the bound is wrong and every traced image
/// is untrustworthy. The mean is the constant that [`SURFACE_GRADIENT_FACTOR`] approximates.
///
/// The bound is asked for rather than written down here, for the reason [`Gyroid::lipschitz`]
/// gives: a check that carries its own copy of the number checks the copy.
fn report_gradient(g: &Gyroid) {
    let n = 120;
    let period = TAU / g.k;

    // Everywhere, not just at the surface: a sphere tracer evaluates the field far from the
    // zero set, so the Lipschitz constant has to bound the gradient over the whole cell.
    let mut everywhere = 0.0f64;
    // At the surface, which is the narrower population that sets the wall thickness.
    let mut at_surface: Vec<f64> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let p = Point3::new(
                    i as f64 / n as f64 * period,
                    j as f64 / n as f64 * period,
                    l as f64 / n as f64 * period,
                );
                let m = g.grad_f(p).norm() / g.k;
                everywhere = everywhere.max(m);
                if g.f(p).abs() < 0.02 {
                    at_surface.push(m);
                }
            }
        }
    }

    let max = at_surface.iter().cloned().fold(0.0f64, f64::max);
    let min = at_surface.iter().cloned().fold(f64::MAX, f64::min);
    let mean = at_surface.iter().sum::<f64>() / at_surface.len() as f64;

    println!(
        "gradient over the whole cell ({} samples): max {everywhere:.3}k",
        n * n * n
    );
    let bound = TpmsKind::Gyroid.gradient_bound();
    // A hair of tolerance, because the bound is *attained*: the maximum is exactly sqrt(3) at
    // the cell corner, so the sampled value and the constant differ only by however the two
    // arithmetic paths rounded, and a bare `<=` is a coin flip on the last bit. Reporting
    // "VIOLATED" on that would be crying wolf about the one check that must be believed.
    // The tolerance is far below any real violation, which would be a whole percent.
    let slack = bound * 1e-9;
    println!(
        "  analytic bound = {bound:.3}k {}",
        if everywhere <= bound + slack {
            "holds"
        } else {
            "VIOLATED - every image below is unsafe"
        }
    );
    // The bound only has to be an upper bound to be safe, but every unit of slack in it is a
    // proportional slowdown, since it divides every step the tracer takes.
    println!(
        "  measured maximum is sqrt(3) = {:.3}k, so the proven bound costs {:.0}% in step size",
        3f64.sqrt(),
        (6f64.sqrt() / everywhere - 1.0) * 100.0
    );
    println!(
        "gradient at the surface ({} samples): min {min:.3}k, mean {mean:.3}k, max {max:.3}k \
         (SURFACE_GRADIENT_FACTOR is {SURFACE_GRADIENT_FACTOR})",
        at_surface.len()
    );
}

/// Measures the wall thickness that actually came out, rather than the one that was asked for.
///
/// The threshold `c` was derived from an approximation of the surface gradient, so the
/// thickness it produces is nominal and needs checking.
///
/// Thickness is measured **along the surface normal**, which is the only direction in which it
/// means anything. Marching an arbitrary straight line through the lattice and timing the
/// solid runs — the obvious approach — measures `t / cos(theta)` for whatever angle the line
/// happens to meet each wall at, and the average of `sec(theta)` over random incidence is both
/// large and heavy-tailed. That method reported a 2.27 mm mean for a wall that is actually
/// 0.97 mm.
fn report_wall_thickness(g: &Gyroid) {
    let n = 40;
    let period = TAU / g.k;
    let step = WALL_MM / 2000.0;

    let mut thicknesses: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let p = Point3::new(
                    i as f64 / n as f64 * period,
                    j as f64 / n as f64 * period,
                    l as f64 / n as f64 * period,
                );
                // Only points lying on the wall's mid-sheet have a full thickness to measure.
                if g.f(p).abs() > 0.005 {
                    continue;
                }
                let normal = g.normal(p);
                // Walk out both ways until the wall ends. The mid-sheet is the centre by
                // construction, so the two half-thicknesses should agree.
                let mut half = [0.0f64; 2];
                for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
                    let mut t = 0.0;
                    while t < WALL_MM * 4.0 && g.raw(p + normal * sign * t) < 0.0 {
                        t += step;
                    }
                    half[side] = t;
                }
                thicknesses.push(half[0] + half[1]);
            }
        }
    }

    if thicknesses.is_empty() {
        println!("wall thickness: no surface points found - the field is empty");
        return;
    }
    let max = thicknesses.iter().cloned().fold(0.0f64, f64::max);
    let min = thicknesses.iter().cloned().fold(f64::MAX, f64::min);
    let mean = thicknesses.iter().sum::<f64>() / thicknesses.len() as f64;
    println!(
        "wall thickness along the normal, over {} surface points: {min:.3} to {max:.3} mm, \
         mean {mean:.3} mm (asked for {WALL_MM})",
        thicknesses.len()
    );
}

/// Counts what meshing one unit cell at a wall-resolving resolution would produce, and
/// extrapolates to the full block — a measured number to set against the traced images.
///
/// This counts sign-changing grid edges rather than calling [`mesh_field`], for two reasons.
/// It is mesher-independent: Surface Nets emits exactly one quad per sign-changing edge, so
/// two triangles per crossing is the count by construction rather than by trusting an
/// implementation. And `mesh_field` cannot run this case at all — it panics at mesh.rs:215 on
/// any surface that reaches the sampling boundary, which a space-filling lattice always does.
fn count_meshing_cost(g: &Gyroid, resolution: usize) {
    let cell = TAU / g.k;
    let grid_mm = cell / resolution as f64;
    let n = resolution + 1;

    println!("\n--- what meshing this would have cost ---");
    println!(
        "one {cell:.0} mm cell at resolution {resolution} ({grid_mm:.3} mm grid, \
         {:.1} samples across a {WALL_MM} mm wall)",
        WALL_MM / grid_mm
    );

    let start = std::time::Instant::now();
    let inside: Vec<bool> = (0..n * n * n)
        .into_par_iter()
        .map(|idx| {
            let (i, j, k) = (idx % n, (idx / n) % n, idx / (n * n));
            let p = Point3::new(i as f64 * grid_mm, j as f64 * grid_mm, k as f64 * grid_mm);
            g.raw(p) < 0.0
        })
        .collect();

    let at = |i: usize, j: usize, k: usize| inside[(k * n + j) * n + i];
    let crossings: u64 = (0..n)
        .into_par_iter()
        .map(|k| {
            let mut c = 0u64;
            for j in 0..n {
                for i in 0..n {
                    let here = at(i, j, k);
                    if i + 1 < n && here != at(i + 1, j, k) {
                        c += 1;
                    }
                    if j + 1 < n && here != at(i, j + 1, k) {
                        c += 1;
                    }
                    if k + 1 < n && here != at(i, j, k + 1) {
                        c += 1;
                    }
                }
            }
            c
        })
        .sum();
    let elapsed = start.elapsed().as_secs_f64();

    let per_cell = crossings * 2;
    let cells = (BLOCK_MM / cell).powi(3);
    let total = per_cell as f64 * cells;
    // Indexed triangles at their most compact: three u32 indices, plus a shared vertex
    // carrying a position and a normal. Roughly one vertex per triangle for a closed surface.
    const BYTES_PER_TRI: f64 = 12.0 + 12.0 + 12.0;
    println!(
        "  {crossings} sign-changing edges -> {per_cell} triangles per cell, in {elapsed:.2}s\n  \
         x {cells:.0} cells = {total:.2e} triangles for the block, ~{:.0} GB indexed\n  \
         and ~{:.0} s of field evaluation at this rate",
        total * BYTES_PER_TRI / 1e9,
        elapsed * cells
    );
    println!(
        "  the grid for this one cell is {:.1}e6 field evaluations; the whole block is \
         {:.2}e9",
        (n * n * n) as f64 / 1e6,
        (n * n * n) as f64 * cells / 1e9
    );
}

struct Camera {
    eye: Point3<f64>,
    forward: Vector3<f64>,
    right: Vector3<f64>,
    up: Vector3<f64>,
    /// Half the image plane's height at unit distance, from the vertical field of view.
    half_h: f64,
    aspect: f64,
}

impl Camera {
    fn framing(
        half: Vector3<f64>,
        dir: Vector3<f64>,
        fov_deg: f64,
        width: usize,
        height: usize,
    ) -> Self {
        let radius = half.norm();
        let half_h = (fov_deg.to_radians() * 0.5).tan();
        // Backed off by the block's own radius regardless of lens, so a long lens zooms in
        // rather than pushing the eye inside the lattice.
        let eye = Point3::origin() + dir * radius * 1.05 / (45f64.to_radians() * 0.5).tan();
        let forward = (Point3::origin() - eye).normalize();
        // An axis view can put `forward` parallel to Z, which would collapse the cross
        // product, so pick a different reference there.
        let reference = if forward.cross(&Vector3::z()).norm() > 1e-6 {
            Vector3::z()
        } else {
            Vector3::y()
        };
        let right = forward.cross(&reference).normalize();
        let up = right.cross(&forward);
        Self {
            eye,
            forward,
            right,
            up,
            half_h,
            aspect: width as f64 / height as f64,
        }
    }

    fn ray(&self, px: usize, py: usize, width: usize, height: usize) -> Vector3<f64> {
        let u = (px as f64 + 0.5) / width as f64 * 2.0 - 1.0;
        let v = 1.0 - (py as f64 + 0.5) / height as f64 * 2.0;
        (self.forward + self.right * (u * self.half_h * self.aspect) + self.up * (v * self.half_h))
            .normalize()
    }
}

/// One ray's outcome, kept so the cost of tracing can be reported rather than guessed at.
struct Hit {
    point: Option<Point3<f64>>,
    steps: u32,
    exhausted: bool,
}

fn trace(eye: Point3<f64>, dir: Vector3<f64>, lattice: &Lattice, policy: StepPolicy) -> Hit {
    let half = lattice.half;
    let Some((t_near, t_far)) = slab_intersect(eye, dir, half) else {
        return Hit {
            point: None,
            steps: 0,
            exhausted: false,
        };
    };
    let epsilon = half.norm() * 2.0 * 1e-5;

    let mut t = t_near.max(0.0);
    for step in 0..STEP_BUDGET {
        if t > t_far {
            return Hit {
                point: None,
                steps: step,
                exhausted: false,
            };
        }
        let p = eye + dir * t;
        let d = lattice.distance_under(p, policy);
        if d < epsilon {
            return Hit {
                point: Some(p),
                steps: step,
                exhausted: false,
            };
        }
        // The same 0.8 the reference tracer uses, for the same reason: an intersection is a
        // max of two distances and can overestimate near a convex corner.
        t += (d * 0.8).max(epsilon);
    }
    Hit {
        point: None,
        steps: STEP_BUDGET,
        exhausted: true,
    }
}

/// Ray/AABB slab test against an origin-centred box, returning entry and exit parameters.
fn slab_intersect(eye: Point3<f64>, dir: Vector3<f64>, half: Vector3<f64>) -> Option<(f64, f64)> {
    let mut t_near = f64::NEG_INFINITY;
    let mut t_far = f64::INFINITY;
    for axis in 0..3 {
        let (o, d) = (eye[axis], dir[axis]);
        let (lo, hi) = (-half[axis], half[axis]);
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (mut t0, mut t1) = ((lo - o) / d, (hi - o) / d);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_near = t_near.max(t0);
        t_far = t_far.min(t1);
        if t_near > t_far {
            return None;
        }
    }
    Some((t_near, t_far))
}

/// Checks a step policy against a brute-force scan of the same rays.
///
/// Whether a policy tunnels cannot be judged from the images: at whole-block framing the
/// lattice aliases into moire, and a ray that skips the first wall and lands on the second
/// still produces a plausible-looking pixel. So this ignores the pixels and asks the only
/// question that matters — is the surface the tracer found *the first one along the ray*?
///
/// Ground truth is a fine linear scan at a twentieth of the wall thickness, which cannot skip
/// a wall. A tracer hit deeper than that first crossing has passed through solid material.
fn verify_step_policy(
    lattice: &Lattice,
    camera: &Camera,
    policy: StepPolicy,
    side: usize,
) -> Verdict {
    let step = lattice.wall / 20.0;
    // Half a wall thickness: deeper than that and the tracer is past the wall it should have
    // stopped at, rather than merely a little way inside it.
    let tolerance = lattice.wall * 0.5;

    let outcomes: Vec<(bool, bool, f64)> = (0..side * side)
        .into_par_iter()
        .filter_map(|i| {
            // A coarse spread across the frame, since this is far costlier per ray than
            // rendering is.
            let (px, py) = ((i % side) * 600 / side, (i / side) * 600 / side);
            let dir = camera.ray(px, py, 600, 600);
            let (t_near, t_far) = slab_intersect(camera.eye, dir, lattice.half)?;

            // The first genuinely solid point along the ray, found without stepping logic.
            let mut truth = None;
            let mut t = t_near.max(0.0);
            while t <= t_far {
                let p = camera.eye + dir * t;
                if lattice.gyroid.raw(p) < 0.0 && box_sdf(p, lattice.half) < 0.0 {
                    truth = Some(t);
                    break;
                }
                t += step;
            }
            let truth = truth?;

            let hit = trace(camera.eye, dir, lattice, policy);
            match hit.point {
                Some(p) => {
                    let traced_t = (p - camera.eye).norm();
                    Some((true, false, traced_t - truth))
                }
                None => Some((false, true, 0.0)),
            }
        })
        .collect();

    let mut v = Verdict {
        checked: outcomes.len(),
        ..Verdict::default()
    };
    let mut total_overshoot = 0.0;
    for (traced, missed, overshoot) in &outcomes {
        if *traced && *overshoot > tolerance {
            v.tunnelled += 1;
            total_overshoot += overshoot;
            v.worst_mm = v.worst_mm.max(*overshoot);
        }
        if *missed {
            v.missed += 1;
        }
    }
    if v.tunnelled > 0 {
        v.mean_mm = total_overshoot / v.tunnelled as f64;
    }
    v
}

/// What a brute-force scan says about a step policy's hits.
#[derive(Default)]
struct Verdict {
    checked: usize,
    /// Rays whose traced hit lies deeper than the first solid point along the ray - they
    /// passed through a wall.
    tunnelled: usize,
    /// Rays that should have hit something and reported nothing.
    missed: usize,
    mean_mm: f64,
    worst_mm: f64,
}

impl Verdict {
    fn rate(&self) -> f64 {
        self.tunnelled as f64 / self.checked.max(1) as f64 * 100.0
    }
}

/// Finds where the naive policy starts passing through walls, by measuring rather than by
/// trusting the algebra.
///
/// # Why this is swept in two dimensions
///
/// The obvious prediction is one-dimensional. The naive step is `0.8(|F| - c)` while the true
/// distance is at least `(|F| - c) / max|grad F|`, so the policy should be safe exactly while
/// `0.8 max|grad F| <= 1` - a threshold on cell size alone, at 8.7 mm using the bound now in
/// use. It was quoted as 12.3 mm against the older `sqrt(6) k`, which was true of that bound
/// and 41% pessimistic: `sqrt(3)` is the real maximum and is now proved, so the threshold moved
/// with it. Both are printed, since the pair is the clearest statement of what the slack cost.
///
/// That prediction is wrong, and sweeping cell size alone shows it: with 1 mm walls, nothing
/// tunnels anywhere from 50 mm down to 4 mm. Overstepping the surface is not the same as
/// passing through it. A step that overshoots by 74% lands *inside* the wall, where the field
/// is negative and the hit test fires anyway — so the wall is still found, just marginally
/// deep. Tunnelling needs the overshoot to clear the whole wall, which makes wall thickness
/// the second variable and the ratio of the two the thing that actually matters.
fn sweep_tunnelling_threshold(width: usize, height: usize) {
    println!("\n--- where the naive policy starts passing through walls ---");
    println!(
        "one-dimensional prediction: unsafe below {:.1} mm at the bound in use, against \
         {:.1} mm at the sqrt(6) it replaced.",
        TAU * TpmsKind::Gyroid.gradient_bound() * 0.8,
        TAU * 6f64.sqrt() * 0.8
    );
    println!("naive tunnelling rate, as a percentage of rays that should have hit:\n");

    let walls = [1.0, 0.5, 0.25, 0.1];
    print!("  cell \\ wall ");
    for w in walls {
        print!("{:>10}", format!("{w} mm"));
    }
    println!();

    for cell in [50.0, 20.0, 10.0, 5.0] {
        print!("  {cell:>7.0} mm  ");
        for wall in walls {
            let lattice = Lattice::new(cell, wall);
            let camera = View::Detail.camera(lattice.half, width, height);
            let naive = verify_step_policy(&lattice, &camera, StepPolicy::Naive, 80);
            print!("{:>10}", format!("{:.2}%", naive.rate()));
        }
        println!();
    }

    // The bounded policy is the control: if it ever tunnels, the Lipschitz constant is wrong
    // and nothing else in this file can be believed.
    let mut worst = 0.0f64;
    for cell in [50.0, 20.0, 10.0, 5.0] {
        for wall in walls {
            let lattice = Lattice::new(cell, wall);
            let camera = View::Detail.camera(lattice.half, width, height);
            worst =
                worst.max(verify_step_policy(&lattice, &camera, StepPolicy::Lipschitz, 80).rate());
        }
    }
    println!("\n  Lipschitz-bounded, worst case over the same 16 combinations: {worst:.2}%");
}

/// Sphere-traces the lattice and shades each hit, reporting the cost as well as writing the
/// image - the step statistics are the measurement this example exists for.
fn render(
    name: &str,
    camera: &Camera,
    width: usize,
    height: usize,
    lattice: &Lattice,
    policy: StepPolicy,
    label: &str,
) -> Vec<[u8; 3]> {
    const BACKGROUND: [u8; 3] = [24, 26, 30];
    let light = (camera.forward * -1.0 + camera.up * 0.6 + camera.right * 0.35).normalize();
    let start = std::time::Instant::now();

    let results: Vec<([u8; 3], u32, bool, bool)> = (0..width * height)
        .into_par_iter()
        .map(|i| {
            let dir = camera.ray(i % width, i / width, width, height);
            let hit = trace(camera.eye, dir, lattice, policy);
            let colour = match hit.point {
                Some(p) => {
                    let n = lattice.normal(p);
                    let shade = 0.15 + 0.85 * n.dot(&light).max(0.0);
                    let c = (shade.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                    [c, c, c]
                }
                None => BACKGROUND,
            };
            (colour, hit.steps, hit.point.is_some(), hit.exhausted)
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64();
    let pixels: Vec<[u8; 3]> = results.iter().map(|r| r.0).collect();
    let hits = results.iter().filter(|r| r.2).count();
    let exhausted = results.iter().filter(|r| r.3).count();
    let total_steps: u64 = results.iter().map(|r| r.1 as u64).sum();
    let max_steps = results.iter().map(|r| r.1).max().unwrap_or(0);

    let mut buf = Vec::with_capacity(width * height * 3);
    for p in &pixels {
        buf.extend_from_slice(p);
    }
    image::save_buffer(
        name,
        &buf,
        width as u32,
        height as u32,
        image::ColorType::Rgb8,
    )
    .expect("writable path");

    println!(
        "{name:<32} {elapsed:>6.2}s  {label}\n  \
         {:.1}% hit, {:.0} steps mean, {max_steps} max, {exhausted} rays exhausted the budget",
        hits as f64 / (width * height) as f64 * 100.0,
        total_steps as f64 / (width * height) as f64,
    );
    pixels
}

/// Tiles the renders side by side, separated by a hairline so adjacent dark backgrounds do not
/// read as one image.
fn tile(name: &str, images: &[&[[u8; 3]]], width: usize, height: usize) {
    const GAP: usize = 2;
    const DIVIDER: [u8; 3] = [70, 74, 82];
    let total = images.len() * width + (images.len() - 1) * GAP;
    let mut buf = Vec::with_capacity(total * height * 3);
    for row in 0..height {
        for (i, image) in images.iter().enumerate() {
            if i > 0 {
                for _ in 0..GAP {
                    buf.extend_from_slice(&DIVIDER);
                }
            }
            for px in &image[row * width..(row + 1) * width] {
                buf.extend_from_slice(px);
            }
        }
    }
    image::save_buffer(
        name,
        &buf,
        total as u32,
        height as u32,
        image::ColorType::Rgb8,
    )
    .expect("writable path");
}
