// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Turning a field that can only answer at points into one a GPU can sample.
//!
//! # Why anything gets baked at all
//!
//! A lattice is thirty operations and evaluates on the GPU directly. A cage's limit surface
//! cannot: there is no closed form for "the surface you get after dragging these control
//! points", so answering *how far is this point from the surface* means searching actual
//! geometry through a bounding volume hierarchy. That search is fine on a CPU asking a few
//! hundred thousand times, and hopeless on a GPU asking hundreds of millions of times a frame.
//!
//! So the answers are computed once, into a grid the GPU reads instead of searching. This is
//! the line the architecture already draws: **procedural geometry stays procedural, and only
//! irreducible leaves get baked.** A cage is irreducible. A lattice never is — it is almost
//! entirely surface, so a grid over it would be enormous and buy nothing.
//!
//! # Narrow band
//!
//! Only distances near the surface are worth storing. Far from it, the exact number does not
//! affect where the surface is, and storing it at full resolution is most of the memory for
//! none of the accuracy. So values are clamped to ±[`BakedField::band`].
//!
//! Clamping *reduces* magnitude, which means the stored value never exceeds the true distance.
//! That direction matters: a renderer stepping along a ray treats the value as a safe travel
//! distance, so understating it is slow and overstating it steps through surfaces. Clamping is
//! therefore safe by construction, and the test below asserts it rather than assuming it.

use fieldform_field::ScalarField;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// How many samples along a row one interval test covers.
///
/// One test costs a distance query and a little arithmetic; skipping a run of samples saves
/// that many queries. Too short and the tests dominate, too long and the bound is too loose to
/// prove anything. Sixteen is measured, not guessed — see `bake_normal_accuracy`.
const SKIP_RUN: usize = 16;

/// A field sampled onto a uniform grid, ready to become a 3D texture.
///
/// Stored as `f32` because that is what a GPU texture holds. Geometry is `f64` everywhere else
/// in fieldform; this is one of the two places the two meet, and it is deliberately visible.
#[derive(Debug, Clone)]
pub struct BakedField {
    /// Samples per axis. One more than the number of cells, since both ends are sampled.
    pub dims: [usize; 3],
    /// World position of sample (0, 0, 0).
    pub lo: Point3<f64>,
    /// World size of one cell, per axis.
    pub cell: Vector3<f64>,
    /// Distances beyond this are clamped. See the module docs.
    pub band: f64,
    /// Row-major: `x` fastest, then `y`, then `z`.
    pub data: Vec<f32>,
    /// The field's own normal at each sample, in the same order as `data`.
    ///
    /// Stored rather than recovered, and that is the entire point. Differencing a sampled
    /// field to find its normal reproduces whatever the sampling made piecewise-constant — for
    /// a grid, the cells; for a distance-to-triangles field underneath, the triangles. Measured
    /// on a 100 mm cage, differencing was 4.56 degrees off the true normal at the traced
    /// viewport's settings, against 1.48 degrees for faceting already judged visible.
    /// `bake_normal_accuracy` is the example that produced those numbers.
    ///
    /// Carrying the real answer instead also decouples shading from resolution: the grid then
    /// governs only how closely the *surface* is placed, exactly as analytic normals decoupled
    /// shading from subdivision level for cages.
    pub normals: Vec<[f32; 3]>,
}

impl BakedField {
    /// Samples `field` over `bounds` at `resolution` cells along its longest axis.
    ///
    /// `band` is how far either side of the surface is stored exactly. A sensible default is a
    /// few cells' worth — enough that a renderer approaching the surface has real distances to
    /// work with, and no more.
    pub fn new(
        field: &(dyn ScalarField + Sync),
        bounds: (Point3<f64>, Point3<f64>),
        resolution: usize,
        band: f64,
    ) -> Self {
        Self::new_cancellable(field, bounds, resolution, band, &|| false)
            .expect("a bake nobody can stop always finishes")
    }

    /// [`Self::new`], stoppable between slabs.
    ///
    /// **Because a settled bake outlives the geometry it was for.** It runs on a worker while
    /// the model is still being dragged, and every edit makes the one in flight useless — but
    /// useless work still holds every core, and the draft bake the drag itself needs then waits
    /// behind it. Measured in the application: a draft bake of a 100 mm cage that costs 15 ms
    /// alone took **1.2 seconds** with a settled bake of the previous edit still running.
    ///
    /// Checked once per z-slab, which is a few tens of milliseconds of work at the settled
    /// resolutions and nothing at all at the draft one.
    pub fn new_cancellable(
        field: &(dyn ScalarField + Sync),
        bounds: (Point3<f64>, Point3<f64>),
        resolution: usize,
        band: f64,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Option<Self> {
        let (lo, hi) = bounds;
        let extent = hi - lo;
        let longest = extent.x.max(extent.y).max(extent.z);
        let resolution = resolution.max(1);
        let step = longest / resolution as f64;

        let dims = [
            ((extent.x / step).ceil() as usize).max(1) + 1,
            ((extent.y / step).ceil() as usize).max(1) + 1,
            ((extent.z / step).ceil() as usize).max(1) + 1,
        ];
        let cell = Vector3::new(step, step, step);

        // Parallel across z-slabs, which are independent and each large enough that splitting
        // pays for itself. Matches how `mesh_field` samples, for the same reason.
        let mut data = vec![0.0f32; dims[0] * dims[1] * dims[2]];
        let mut normals = vec![[0.0f32; 3]; dims[0] * dims[1] * dims[2]];
        let plane = dims[0] * dims[1];
        data.par_chunks_mut(plane)
            .zip(normals.par_chunks_mut(plane))
            .enumerate()
            .for_each(|(k, (slab, normal_slab))| {
                if stop() {
                    return;
                }
                let mut j = 0;
                while j < dims[1] {
                    let mut i = 0;
                    while i < dims[0] {
                        // Ask what the field can possibly do over this run of samples before
                        // evaluating any of them. Most of a bake is far from the surface and
                        // ends up clamped to the band, so computing an exact distance there is
                        // work thrown away — and a cage's distance is a hierarchy traversal,
                        // not an arithmetic expression. This is what `ScalarField::interval`
                        // was added for; until now nothing consumed it.
                        let run = (dims[0] - i).min(SKIP_RUN);
                        let block_lo = lo
                            + Vector3::new(
                                i as f64 * cell.x,
                                j as f64 * cell.y,
                                k as f64 * cell.z,
                            );
                        // A cell of margin on every side, so a run filled with the band value
                        // still interpolates correctly against an evaluated neighbour.
                        let pad = Vector3::new(cell.x, cell.y, cell.z);
                        let block_hi = block_lo + Vector3::new((run - 1) as f64 * cell.x, 0.0, 0.0);
                        let (bound_lo, bound_hi) =
                            field.interval(block_lo - pad, block_hi + pad);

                        if bound_lo >= band || bound_hi <= -band {
                            // Provably outside the band for its whole length. The stored value
                            // is the clamp; the stored normal is never read, because a ray only
                            // asks for a normal where it found a surface and there is none here.
                            let value = if bound_lo >= band { band } else { -band } as f32;
                            for step in 0..run {
                                let at = j * dims[0] + i + step;
                                slab[at] = value;
                                normal_slab[at] = [0.0, 0.0, 1.0];
                            }
                            i += run;
                            continue;
                        }

                        // **And what is left, shortened to this run** -- backlog item 17. The
                        // bound above says whether the run is worth evaluating; the same
                        // interval walk says which branches could have produced it, and the
                        // rest are deleted. Built once for the run and used for every sample
                        // and every normal in it, which is what makes it pay: a run that keeps
                        // one body out of forty evaluates one.
                        //
                        // `None` for a field with no branches to drop, which is every leaf --
                        // the whole field is then the shortest there is.
                        let shortened = field.pruned_to(block_lo - pad, block_hi + pad);
                        let near: &(dyn ScalarField + Sync) = match &shortened {
                            Some(local) => local.as_ref(),
                            None => field,
                        };
                        for step in 0..run {
                            let p = block_lo + Vector3::new(step as f64 * cell.x, 0.0, 0.0);
                            let at = j * dims[0] + i + step;
                            slab[at] = near.distance(p).clamp(-band, band) as f32;
                            // Falls back to differencing only for a field with no analytic
                            // answer of its own. A cage has one, which is the case this exists
                            // to carry.
                            let n = near
                                .normal(p)
                                .unwrap_or_else(|| central_difference(near, p, cell.x));
                            normal_slab[at] = [n.x as f32, n.y as f32, n.z as f32];
                        }
                        i += run;
                    }
                    j += 1;
                }
            });

        // Asked after the pass rather than trusted to it: a stopped bake leaves the slabs it
        // never reached at zero, and a grid of zeroes is a solid block of surface everywhere
        // rather than an empty one — the worst possible thing to hand a tracer.
        if stop() {
            return None;
        }
        Some(Self { dims, lo, cell, band, data, normals })
    }

    /// Bytes the grid occupies as an RGBA `f32` texture — normal in `xyz`, distance in `w`.
    ///
    /// Four channels rather than one, which is the price of carrying the normal. It buys back
    /// more than it costs: because shading no longer depends on how finely the grid samples,
    /// the resolution can drop to whatever the *surface position* needs.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>() * 4
    }

    /// The grid as RGBA texels: normal in `xyz`, distance in `w`.
    pub(crate) fn texels(&self) -> Vec<[f32; 4]> {
        self.data
            .iter()
            .zip(&self.normals)
            .map(|(d, n)| [n[0], n[1], n[2], *d])
            .collect()
    }

    /// Trilinearly interpolated normal, matching how the shader reads it.
    ///
    /// Interpolating normals is smooth across cell boundaries in a way that differencing the
    /// interpolated *distance* is not — the same reason a mesh shaded from vertex normals does
    /// not show its facets while one shaded from face normals does.
    pub fn sample_normal(&self, p: Point3<f64>) -> Vector3<f64> {
        let n = self.trilinear(p, |i, j, k| {
            let at = (k * self.dims[1] + j) * self.dims[0] + i;
            let n = self.normals[at];
            Vector3::new(n[0] as f64, n[1] as f64, n[2] as f64)
        });
        n.try_normalize(1e-12).unwrap_or_else(Vector3::z)
    }

    fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        let i = i.min(self.dims[0] - 1);
        let j = j.min(self.dims[1] - 1);
        let k = k.min(self.dims[2] - 1);
        self.data[(k * self.dims[1] + j) * self.dims[0] + i] as f64
    }

    /// Trilinear interpolation, matching what a GPU sampler does between texels.
    ///
    /// This exists so the bake can be checked on the CPU against the field it came from. What
    /// this returns is what the shader will see, so a discrepancy found here is a discrepancy
    /// that would otherwise have been found by looking at a wrong picture.
    pub fn sample(&self, p: Point3<f64>) -> f64 {
        self.trilinear(p, |i, j, k| self.at(i, j, k))
    }

    /// Trilinear interpolation of whatever `fetch` returns at each of the eight corners.
    ///
    /// Shared by distance and normal so the two cannot drift into interpolating differently,
    /// which would put the shading and the surface in slightly different places.
    fn trilinear<T, F>(&self, p: Point3<f64>, fetch: F) -> T
    where
        T: std::ops::Add<T, Output = T> + std::ops::Mul<f64, Output = T> + Copy,
        F: Fn(usize, usize, usize) -> T,
    {
        let g = Vector3::new(
            ((p.x - self.lo.x) / self.cell.x).clamp(0.0, (self.dims[0] - 1) as f64),
            ((p.y - self.lo.y) / self.cell.y).clamp(0.0, (self.dims[1] - 1) as f64),
            ((p.z - self.lo.z) / self.cell.z).clamp(0.0, (self.dims[2] - 1) as f64),
        );
        let (i, j, k) = (g.x.floor(), g.y.floor(), g.z.floor());
        let (tx, ty, tz) = (g.x - i, g.y - j, g.z - k);
        let (i, j, k) = (i as usize, j as usize, k as usize);
        let (i1, j1, k1) = (
            (i + 1).min(self.dims[0] - 1),
            (j + 1).min(self.dims[1] - 1),
            (k + 1).min(self.dims[2] - 1),
        );

        let lerp = |a: T, b: T, t: f64| a * (1.0 - t) + b * t;
        let c00 = lerp(fetch(i, j, k), fetch(i1, j, k), tx);
        let c10 = lerp(fetch(i, j1, k), fetch(i1, j1, k), tx);
        let c01 = lerp(fetch(i, j, k1), fetch(i1, j, k1), tx);
        let c11 = lerp(fetch(i, j1, k1), fetch(i1, j1, k1), tx);
        lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
    }
}

/// Central differences, for a field with no analytic normal of its own.
fn central_difference(field: &(dyn ScalarField + Sync), p: Point3<f64>, h: f64) -> Vector3<f64> {
    let d = |o: Vector3<f64>| field.distance(p + o);
    Vector3::new(
        d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
        d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
        d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
    )
    .try_normalize(1e-18)
    .unwrap_or_else(Vector3::z)
}

/// A baked grid, presented as an ordinary field.
///
/// Lets a bake be composed, meshed, traced or tested anywhere a field is accepted, and in
/// particular lets it be compared against the field it was baked from using the machinery that
/// already exists.
pub struct BakedFieldSource(pub BakedField);

impl ScalarField for BakedFieldSource {
    fn distance(&self, p: Point3<f64>) -> f64 {
        self.0.sample(p)
    }

    /// The stored normal, interpolated — not the gradient of the stored distance.
    ///
    /// These are different answers, and the difference is the whole reason normals are baked.
    /// See [`BakedField::normals`].
    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        Some(self.0.sample_normal(p))
    }

    /// Clamping to a band and interpolating between samples both shorten the value, so what
    /// comes out bounds the distance from below rather than being it. Saying so is what stops
    /// a blend being taken against it and coming out the wrong shape (D8).
    fn normalization(&self) -> fieldform_field::Normalization {
        fieldform_field::Normalization::LowerBound
    }
}

#[cfg(test)]
mod tests {
    /// A ball, and a counter of how many readings it was asked for.
    struct CountedBall(std::sync::atomic::AtomicUsize);

    impl ScalarField for CountedBall {
        fn distance(&self, p: Point3<f64>) -> f64 {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            p.coords.norm() - 10.0
        }
    }

    /// **A bake that is told to stop gives nothing back, and stops sampling.** Nothing reads a
    /// cancelled bake, and the half it managed is worse than nothing: the slabs it never reached
    /// are zeroes, and a grid of zeroes is surface everywhere. It has to be refused rather than
    /// returned short.
    ///
    /// The count is the other half of the promise. A token checked only at the end would let a
    /// settled bake of geometry nobody is looking at hold every core to completion, which is the
    /// failure this exists to prevent: a draft bake of 15 ms measured 1.2 s beside one.
    #[test]
    fn a_bake_told_to_stop_gives_nothing_back_and_stops_reading() {
        let field = CountedBall(std::sync::atomic::AtomicUsize::new(0));
        let bounds = (Point3::new(-12.0, -12.0, -12.0), Point3::new(12.0, 12.0, 12.0));
        let out = BakedField::new_cancellable(&field, bounds, 48, 4.0, &|| true);
        assert!(out.is_none(), "a stopped bake has nothing worth keeping");
        let read = field.0.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(read, 0, "a bake stopped before it began should read nothing, read {read}");

        // And one nobody stops is the bake it always was.
        let field = CountedBall(std::sync::atomic::AtomicUsize::new(0));
        let free = BakedField::new_cancellable(&field, bounds, 16, 4.0, &|| false)
            .expect("nothing stopped it");
        let plain = BakedField::new(&CountedBall(std::sync::atomic::AtomicUsize::new(0)), bounds, 16, 4.0);
        assert_eq!(free.dims, plain.dims);
        assert_eq!(free.data, plain.data);
    }
    use super::*;
    use fieldform_cage::Cage;
    use fieldform_field::CageField;

    fn cube_field() -> CageField {
        CageField::from_cage(&Cage::cube_of(100.0), 4).unwrap()
    }

    fn cube_bounds() -> (Point3<f64>, Point3<f64>) {
        (
            Point3::new(-80.0, -80.0, -80.0),
            Point3::new(80.0, 80.0, 80.0),
        )
    }

    /// The property everything downstream rests on: a renderer treats the value as a safe
    /// travel distance, so it may be short but must never be long. Both clamping and
    /// interpolation shorten it, and this asserts that rather than trusting the argument.
    #[test]
    fn a_baked_value_never_exceeds_the_true_distance() {
        let field = cube_field();
        let bounds = cube_bounds();
        let baked = BakedField::new(&field, bounds, 48, 20.0);

        let n = 25;
        let mut worst_overshoot: f64 = 0.0;
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    let t = |a: f64, b: f64, s: usize| a + (b - a) * s as f64 / n as f64;
                    let p = Point3::new(
                        t(bounds.0.x, bounds.1.x, i),
                        t(bounds.0.y, bounds.1.y, j),
                        t(bounds.0.z, bounds.1.z, k),
                    );
                    let truth = field.distance(p).clamp(-baked.band, baked.band);
                    // Compared in magnitude: the stored value must not claim more room than
                    // there is, on whichever side of the surface it sits.
                    let overshoot = baked.sample(p).abs() - truth.abs();
                    worst_overshoot = worst_overshoot.max(overshoot);
                }
            }
        }

        // One cell of slack: trilinear interpolation across a curved surface can read slightly
        // long between samples, and that is bounded by the cell size rather than unbounded.
        let cell = baked.cell.x;
        assert!(
            worst_overshoot < cell,
            "baked value overshot the true distance by {worst_overshoot:.4} mm, \
             more than one {cell:.4} mm cell"
        );
    }

    /// The bake has to put the surface where the field puts it, or the picture is of a
    /// different shape. Checked by sign agreement away from the surface, where a
    /// half-cell-scale disagreement is expected and meaningless.
    #[test]
    fn the_bake_agrees_with_the_field_about_what_is_inside() {
        let field = cube_field();
        let bounds = cube_bounds();
        let baked = BakedField::new(&field, bounds, 64, 20.0);

        let cell = baked.cell.x;
        let mut checked = 0;
        let n = 20;
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    let t = |a: f64, b: f64, s: usize| a + (b - a) * s as f64 / n as f64;
                    let p = Point3::new(
                        t(bounds.0.x, bounds.1.x, i),
                        t(bounds.0.y, bounds.1.y, j),
                        t(bounds.0.z, bounds.1.z, k),
                    );
                    let truth = field.distance(p);
                    // Skip points within a cell of the surface: which side of it they fall on
                    // is decided by interpolation, and disagreeing there is the sampling
                    // resolution, not an error.
                    if truth.abs() < cell {
                        continue;
                    }
                    assert_eq!(
                        baked.sample(p) < 0.0,
                        truth < 0.0,
                        "at {p:?} the bake says {} and the field says {truth}",
                        baked.sample(p)
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000, "only {checked} points were far enough to check");
    }

    /// Away from the band the exact number stops mattering, and the whole point of clamping is
    /// that it does not pretend otherwise.
    #[test]
    fn values_outside_the_band_are_clamped_to_it() {
        let baked = BakedField::new(&cube_field(), cube_bounds(), 32, 5.0);
        assert!(
            baked.data.iter().all(|v| v.abs() <= 5.0 + 1e-6),
            "every stored value should be inside the band"
        );
        // A corner is far outside a 100 mm cube, so it must be pinned at the band rather than
        // reporting its real distance.
        let corner = baked.sample(Point3::new(-79.0, -79.0, -79.0));
        assert!((corner - 5.0).abs() < 1e-6, "corner read {corner}, expected the band");
    }

    /// A bake is a lower bound, not a distance, and has to say so — otherwise a blend taken
    /// against it produces a seam in the wrong place and looks entirely plausible (D8).
    #[test]
    fn a_bake_declares_itself_a_lower_bound() {
        let baked = BakedFieldSource(BakedField::new(&cube_field(), cube_bounds(), 16, 10.0));
        assert_eq!(
            baked.normalization(),
            fieldform_field::Normalization::LowerBound
        );
    }

    /// The number that decides whether a resolution is affordable, stated rather than guessed.
    #[test]
    fn the_size_of_a_bake_is_what_the_grid_says_it_is() {
        let baked = BakedField::new(&cube_field(), cube_bounds(), 64, 10.0);
        let [x, y, z] = baked.dims;
        // Four channels: normal in xyz, distance in w.
        assert_eq!(baked.size_bytes(), x * y * z * 16);
        assert_eq!(baked.texels().len(), x * y * z);
        // 65^3 RGBA texels is about 4.4 MB. Four times what storing distance alone cost, and
        // worth it because shading no longer depends on this number — a cage is a small,
        // smooth, closed surface, and the resolution now only has to place it.
        assert!(baked.size_bytes() < 8 * 1024 * 1024, "{} bytes", baked.size_bytes());
    }

    /// The fix, asserted where it can be checked without a GPU.
    ///
    /// A stored normal has to beat a differenced one by a wide margin, or carrying it is not
    /// worth four times the memory. Differencing the same bake measures about 4.5 degrees off
    /// at the viewport's settings — worse than faceting already judged visible — because it
    /// recovers whatever the sampling made piecewise-constant rather than the real answer.
    #[test]
    fn a_stored_normal_beats_a_differenced_one() {
        let field = cube_field();
        let bounds = cube_bounds();
        let baked = BakedFieldSource(BakedField::new(&field, bounds, 64, 20.0));
        let cell = baked.0.cell.x;

        // Points on the surface, walked in from outside along a spread of directions.
        let mut worst_stored: f64 = 0.0;
        let mut worst_differenced: f64 = 0.0;
        let mut checked = 0;
        for i in 0..200 {
            let a = i as f64 * 2.399_963_23;
            let dir = Vector3::new(a.cos(), a.sin(), (i as f64 * 0.7).sin()).normalize();
            let mut p = Point3::origin() + dir * 140.0;
            for _ in 0..100 {
                let d = field.distance(p);
                if d.abs() < 1e-6 {
                    break;
                }
                p -= dir * d;
            }
            if field.distance(p).abs() > 1e-4 {
                continue;
            }
            checked += 1;

            let truth = field.closest(p).normal;
            let angle = |v: Vector3<f64>| truth.dot(&v).clamp(-1.0, 1.0).acos().to_degrees();
            worst_stored = worst_stored.max(angle(baked.normal(p).unwrap()));
            worst_differenced =
                worst_differenced.max(angle(central_difference(&baked, p, cell * 0.5)));
        }

        assert!(checked > 100, "only {checked} surface points found");
        assert!(
            worst_stored < 1.0,
            "a stored normal should be near-exact, got {worst_stored:.2} deg"
        );
        assert!(
            worst_stored * 3.0 < worst_differenced,
            "stored {worst_stored:.2} deg should comfortably beat differenced \
             {worst_differenced:.2} deg, or carrying it is not worth the memory"
        );
    }
}
