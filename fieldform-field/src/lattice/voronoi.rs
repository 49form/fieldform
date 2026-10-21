// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The Voronoi cells: a jittered seed grid hashed alike on CPU and GPU, and the measure over it.

use fieldform_scene::Banding;
use nalgebra::{Point3, Vector3};

/// A cell whose geometry comes from a jittered grid of seeds. `voronoi-lattice-spec.md` §4.
///
/// **A fourth family beside surfaces, beams and prisms**, and it is one for the reason the
/// others are: what its parameter means, what its gradient bound is, and whether a blend
/// against it is exact are all different answers. A surface cell is a formula banded by an
/// isovalue; a beam is a distance to segments; a prism is a distance to walls in a plane; this
/// is a *difference of order statistics* over a point set that is generated, not stored -- so it
/// is a lower bound under every map, with no configuration that makes it exact.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StochasticKind {
    /// The walls between the Voronoi cells of the seeds: `(d2 - d1)/2`.
    VoronoiFoam,
    /// Beams along the edges where three cells meet: `(d3 - d1)/2`.
    VoronoiBeam,
}

impl StochasticKind {
    pub const ALL: [StochasticKind; 2] = [StochasticKind::VoronoiFoam, StochasticKind::VoronoiBeam];

    pub fn label(self) -> &'static str {
        match self {
            StochasticKind::VoronoiFoam => "Voronoi foam",
            StochasticKind::VoronoiBeam => "Voronoi beam",
        }
    }

    /// Which order statistic the measure reads, as an index into the three smallest distances.
    ///
    /// `S_k`, the set of points equidistant to at least `k` seeds, is the wall complex at `k=2`
    /// and the edge network at `k=3`. The spec's §4.2 proof is one statement covering both:
    /// every `d_i` is a `k`-th smallest of 1-Lipschitz functions and so is itself 1-Lipschitz,
    /// which gives `(d_k - d_1)/2 <= d(p, S_k)` for **any** `k >= 2`. So one pass tracking three
    /// distances answers both kinds, and the kind is only which one it subtracts.
    pub(crate) fn order(self) -> usize {
        match self {
            StochasticKind::VoronoiFoam => 1,
            StochasticKind::VoronoiBeam => 2,
        }
    }

    pub(crate) fn supported_bandings(self) -> &'static [Banding] {
        match self {
            // A wall has two sides; an edge beam has one. `Solid` is offered by neither, and
            // that is a measurement gap rather than an oversight: the density-to-isovalue table
            // would need a second axis in `jitter`. `voronoi-lattice-spec.md` §4.5.
            StochasticKind::VoronoiFoam => &[Banding::Sheet],
            StochasticKind::VoronoiBeam => &[Banding::Strut],
        }
    }

    /// How wide the wall typically comes out **at this jitter**, as a multiple of the number
    /// typed. What the panel says beside the number, and the reason there is no slack constant
    /// to divide out.
    ///
    /// **It takes the jitter because the answer depends on it and the panel knows it.** The
    /// foam runs from 1.23x at a regular grid to 2.11x at full jitter — nearly a factor of two
    /// across the row — so quoting the whole range beside a Jitter field set to a quarter would
    /// say something true about the cell and something misleading about the part. Interpolated
    /// between the measured quarters rather than fitted: the beam's row is not monotonic in
    /// jitter, and a line through it would be a worse answer than the numbers are.
    ///
    /// `voronoi-lattice-spec.md` §4.5 asked for a per-kind constant, calibrated the way
    /// `tpms-infill-spec.md` §5.2 calibrated `g_kind`, so that the wall *measured* comes out the
    /// width asked for on average. Measurement said not to, and the number that said so is the
    /// **thinnest** chord rather than the mean one. `voronoi_profile` reports, over 900 lines at
    /// two-micron steps:
    ///
    /// | jitter | thinnest wall | median wall |
    /// |---|---|---|
    /// | 0.00 | 1.002 | 1.226 |
    /// | 0.50 | 1.002 | 1.504 |
    /// | 1.00 | 1.010 | 2.108 |
    ///
    /// So the wall is never thinner than the number typed and is often thicker -- the same
    /// one-directional error a `LowerBound` operand gives a fillet, and safe in the same
    /// direction. Dividing out the *median* would centre the average on the number typed and
    /// put the thinnest places at half of it, which for a **minimum wall thickness** is the one
    /// direction that is not allowed to be wrong: it is a manufacturer's constraint, and a part
    /// that reads 1 mm and builds 0.5 mm in places does not print.
    ///
    /// The calibration constant that keeps a wall at least as thick as asked is therefore one,
    /// and a multiply by one is not worth carrying. What is owed instead is the marking §4.5
    /// also asks for, which is this number said where the thickness is entered.
    pub fn wall_spread(self, jitter: f64) -> f64 {
        // The median chord over the typed thickness at jitter 0, a quarter, a half, three
        // quarters and one -- `voronoi_profile`'s last table, and the spec's §14.4.
        let row: [f64; 5] = match self {
            StochasticKind::VoronoiFoam => [1.226, 1.278, 1.504, 1.886, 2.116],
            StochasticKind::VoronoiBeam => [1.381, 1.328, 1.293, 1.351, 1.459],
        };
        let t = (jitter.clamp(0.0, 1.0) * 4.0).min(3.999_999);
        let i = t.floor() as usize;
        row[i] + (row[i + 1] - row[i]) * (t - i as f64)
    }
}

/// Three 32-bit values from an integer cell index and a seed, the same on both sides of the
/// device boundary.
///
/// **PCG3D**, Jarzynski and Olano, *Hash Functions for GPU Rendering*, JCGT 9(3) 2020, table 1 --
/// named and written down because `voronoi-lattice-spec.md` §4.1 disqualifies the usual
/// `sin(dot(p, k)) * 43758.5453` on two counts: it is not reproducible across drivers, which D7
/// forbids outright, and it has visible structure. Every operation here is 32-bit integer and
/// wraps, so `prelude.wgsl`'s copy is the same arithmetic rather than a near-enough one --
/// `a_hash_is_the_same_number_wherever_it_is_computed` pins it to a table of values.
///
/// The seed is mixed into all three lanes through two odd constants (the golden-ratio and
/// xxHash mixers), so that re-rolling changes every axis rather than shifting one.
pub fn voronoi_hash3(c: [i32; 3], seed: u32) -> [u32; 3] {
    let mut v = [
        (c[0] as u32) ^ seed,
        (c[1] as u32) ^ seed.wrapping_mul(0x9E37_79B9),
        (c[2] as u32) ^ seed.wrapping_mul(0x85EB_CA6B),
    ];
    for k in 0..3 {
        v[k] = v[k].wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    for k in 0..3 {
        v[k] ^= v[k] >> 16;
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v
}

/// The `u32` a node's stored `u64` seed is hashed with.
///
/// **Folded rather than truncated**, so that two seeds differing only above the low word are
/// different arrangements. Shared with the shader deliberately: the codegen writes this number,
/// halved into two sixteen-bit slots because a params slot is an `f32` and cannot hold 32 bits
/// of integer exactly -- so the fold has to happen on one side only, and this is it.
pub fn voronoi_seed(stored: u64) -> u32 {
    (stored ^ (stored >> 32)) as u32
}

/// One seed found by the ring pass: where it is, in cell units, and how far away.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Seed {
    pub distance: f64,
    pub at: Point3<f64>,
}

/// The three nearest seeds to `u`, searching the `(2R+1)^3` cells about it.
///
/// `period` wraps the *hash index* on axes that close on themselves, leaving the seed's
/// *position* where it actually is. The two differ by design: on a cylindrical map the cell at
/// argument `-1` and the cell at `N-1` are the same cell seen round the other way, so they must
/// hash alike or the pattern tears down the full height of the cylinder -- and a tear is not
/// merely ugly, it is a discontinuity in the field, which is a ray stepping through a wall.
pub(crate) fn nearest_seeds(
    u: Point3<f64>,
    jitter: f64,
    seed: u32,
    period: [Option<u32>; 3],
    radius: i32,
) -> [Seed; 3] {
    let base = [
        u.x.floor() as i32,
        u.y.floor() as i32,
        u.z.floor() as i32,
    ];
    let far = Seed { distance: f64::INFINITY, at: Point3::origin() };
    let mut best = [far; 3];
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let c = [base[0] + dx, base[1] + dy, base[2] + dz];
                let mut index = c;
                for k in 0..3 {
                    if let Some(n) = period[k] {
                        index[k] = c[k].rem_euclid(n as i32);
                    }
                }
                let h = voronoi_hash3(index, seed);
                let unit = |w: u32| w as f64 * (1.0 / 4_294_967_296.0);
                let at = Point3::new(
                    c[0] as f64 + 0.5 + jitter * (unit(h[0]) - 0.5),
                    c[1] as f64 + 0.5 + jitter * (unit(h[1]) - 0.5),
                    c[2] as f64 + 0.5 + jitter * (unit(h[2]) - 0.5),
                );
                let found = Seed { distance: (u - at).norm(), at };
                if found.distance < best[0].distance {
                    best[2] = best[1];
                    best[1] = best[0];
                    best[0] = found;
                } else if found.distance < best[1].distance {
                    best[2] = best[1];
                    best[1] = found;
                } else if found.distance < best[2].distance {
                    best[2] = found;
                }
            }
        }
    }
    best
}

/// The measure and its gradient, both in cell units: `max(0, (min(d_k, l) - d_1) / 2)`.
///
/// **The ring radius is a correctness parameter and one fixed ceiling is what makes it one.**
/// `voronoi-lattice-spec.md` §4.3: a seed in a cell at Chebyshev distance two or more is at least
/// `l = 1 + (1 - jitter)/2` away, while the query's own-cell seed can be as far as
/// `sqrt(3)(1 + jitter)/2`. Setting the second below the first gives
/// `jitter <= (3 - sqrt(3))/(1 + sqrt(3)) ~ 0.464` as the condition under which one ring provably
/// holds the true nearest seed. Above it, it may not.
///
/// Three facts, and together they are the whole of why one ring is enough:
///
/// - **Every seed nearer than `l` was searched**, so `min(d_k, l)` over what was found is
///   `min(d_k, l)` over every seed there is -- exactly, not approximately.
/// - **So is `d_1`, whenever `d_1 < l`.** And where `d_1 >= l`, the subtraction is already
///   negative and the `max` returns nought, which understates -- the only safe direction.
/// - **Therefore the value is `max(0, (min(d_k^true, l) - d_1^true)/2)` everywhere**, with `l` a
///   constant. A `min` against a constant is 1-Lipschitz, a difference of two such is
///   2-Lipschitz, the half makes it one again, and `max` with nought keeps it -- which is the
///   `gradient_bound` of 1 this family reports, and what lets a tracer step by it.
///
/// **§4.3's own prescription was to widen to `R = 2` and repeat, and that is unsound.** The
/// ceiling is `R + (1 - jitter)/2`, so widening raises it by a whole cell and the value *steps
/// upward* across the surface where `d_1 = l` -- nought on one side and the unclamped truth on
/// the other. Measured at jitter 1: a 10 mm foam with a 1 mm wall read -0.49 mm one side of that
/// surface and +0.52 mm an epsilon away, which is half a millimetre of room reported beside
/// material, and a ray steps through it. Found by review after the fact; `a_measure_does_not_step_across_the_surface_where_one_ring_stops_answering` is what holds it now.
///
/// What the fixed ceiling costs is the geometry where `d_1 >= l`, which reads solid rather than
/// measured. That region is where the query is far from every seed, which is a Voronoi *vertex* --
/// where the walls already meet and the material already is. §14.3 measures how much it adds.
pub fn voronoi_measure(
    kind: StochasticKind,
    u: Point3<f64>,
    raw_jitter: f64,
    seed: u32,
    period: [Option<u32>; 3],
) -> (f64, Vector3<f64>) {
    // **Clamped here as well as in [`Lattice::with_scatter`]**, and this is the copy that
    // matters for agreement: the shader has no install step, so `prelude.wgsl`'s
    // `voronoi_measure` clamps at exactly this point, and `voronoi_probe` compares the two
    // here. Above one a seed leaves its own cell and the ring search can miss the true nearest,
    // which collapses every bound above.
    let jitter = raw_jitter.clamp(0.0, 1.0);
    let ceiling = 1.0 + (1.0 - jitter) * 0.5;
    let seeds = nearest_seeds(u, jitter, seed, period, 1);
    let near = seeds[0];
    let far = seeds[kind.order()];
    let reach = far.distance.min(ceiling);
    let value = (reach - near.distance) * 0.5;
    if value <= 0.0 {
        // Nothing may be concluded here and nought is what says so.
        return (0.0, Vector3::zeros());
    }
    // The gradient of a distance is the unit vector away from the point it is measured to, and
    // a clamped term contributes none. Discontinuous exactly on the wall complex, where the
    // gradient is genuinely undefined -- §4.3.
    let away = |s: Seed| {
        if s.distance > 1e-12 {
            (u - s.at) / s.distance
        } else {
            Vector3::zeros()
        }
    };
    let gradient = if far.distance > ceiling {
        -away(near) * 0.5
    } else {
        (away(far) - away(near)) * 0.5
    };
    (value, gradient)
}
