// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A profile's distance, sampled onto a grid once instead of walked at every query.
//!
//! # Why this exists
//!
//! [`Profile::distance`](crate::Profile::distance) walks every segment of the boundary for every
//! query, and the shader does the same — so the cost of an extrude or a revolve is linear in how
//! finely its profile was sampled. Measured by `fieldform-gpu/examples/traced_cost.rs` at 720 px,
//! a four-point square profile renders in 7.6 ms and a four-hundred-point one in 74.7.
//!
//! That put speed and smoothness on one dial: fewer points is faster and visibly faceted, more
//! points is smooth and slow. A grid takes the dial away. The cost of a lookup does not depend on
//! how many points the boundary has, so **the bake can be made from a far finer polyline than
//! anything could afford to walk** — which is the part worth noticing. The sampling density and
//! the per-pixel cost stop being the same number.
//!
//! It is the rule the project notes already states for cages, one dimension down: *procedural geometry
//! stays procedural; only irreducible leaves get baked*. A hand-drawn outline is as irreducible
//! as a control cage.
//!
//! # Why it is a band and not a square
//!
//! A distance field only has to be *accurate near its surface*. Far away a ray is asking "how big
//! a step is safe", and a rough answer costs it one extra step; near the surface the answer is
//! where the surface *is*. A dense square spends the same precision everywhere, which is most of
//! it spent where nobody looks — and it is what stops a fine bake being affordable at all. On a
//! 100 mm profile, a dense grid fine enough for a **0.05 mm** margin is 18 MB.
//!
//! So there are two levels. A coarse grid over the whole box carries the sign and a safe bound
//! everywhere; fine blocks are stored only where the surface passes. The fine part then follows
//! the boundary rather than filling the box, and a boundary is one dimension smaller than the
//! area it encloses — which is the whole saving. That is the shape of a cage's narrow band, for
//! the same reason.
//!
//! Each fine block carries a one-sample apron, so interpolating inside a block never needs to
//! reach into its neighbour — which is what keeps the lookup a lookup rather than a search for
//! which block to read next.
//!
//! # Why it never overstates
//!
//! Bilinear interpolation of a sampled distance **overestimates across a crease**. A distance
//! field is convex near a corner of the boundary — locally the larger of two planes — and a
//! straight line drawn between two samples of a convex function lies *above* it. Overstating is
//! the one thing a `ScalarField` may never do: a ray would step past the surface it was looking
//! for.
//!
//! So a margin is subtracted. For a function that changes no faster than one millimetre per
//! millimetre — which a distance is, by definition — the worst a bilinear cell can overshoot is
//! half its own diagonal, reached when a crease runs exactly through the middle of the cell. Each
//! level carries its own, since a coarse cell is wider and can overshoot further.
//!
//! # And never from inside either, which it did until 2026-10-02
//!
//! "Subtracted" is the right word only outside the section. Inside, the reading is negative,
//! and a margin subtracted from a negative number makes it *deeper* — so the section claimed
//! more depth than it had, by the coarse level's margin over everything the fine band does not
//! cover. Nothing marches a section from its own inside, so nothing showed it, until the
//! section is **cut out of something**: the slot is then the section's inside, a ray crosses it
//! by the depth it is told, and it was told up to eight fine margins too much.
//! `a_baked_section_cut_out_of_a_block_claims_no_more_room_in_the_slot_than_there_is` measured
//! 3.48 mm of that at a two-millimetre tolerance, on a plain square profile, and
//! `gpu_matches_cpu` drew a slot's sides two millimetres into the block they were cut from.
//!
//! So the coarse level's margin is taken **toward nought**, off whichever sign the reading
//! has: a bound on the distance outside and on the depth inside. A coarse-only block cannot
//! straddle the surface — its corners all sit a block's diagonal clear of the band, on one
//! side — so there is a sign to read and it is the same across the block.
//!
//! **The fine level is left as it was, a margin down on both sides**, and that is deliberate.
//! It is what gives the baked section one surface: the place the reading crosses nought, a
//! margin outside the true boundary all the way round. Taking the fine margin toward nought too
//! would make the reading nought across a band two margins wide, which is a surface with a
//! thickness and a solid that overlaps its own complement. Inside the band the depth is
//! therefore long by up to two fine margins — a quarter of the tolerance — and continuous
//! through nought, so a march converges on the surface instead of stepping over it.

use nalgebra::Point2;
use rayon::prelude::*;

use crate::Profile;

/// Fine cells along the edge of one block.
///
/// The two costs pull opposite ways: the page table is one entry per block, so large blocks make
/// it small, while a block is stored whole wherever the surface touches it, so large blocks waste
/// more. Eight is near the bottom of that curve for the profiles this is used on, and keeps a
/// block's apron cheap at eighty-one samples.
const BLOCK: usize = 8;

/// How far from a fine block, in margins, a segment is listed for the block's exact read, and so
/// how near the outline a point has to be for that read to answer (backlog item 115).
///
/// **Four margins, which is where the exact read is asked.** A ray's finishing reads sit within
/// two margins of the outline -- its candidate is where the baked reading dropped under the
/// threshold, a margin outside the outline at most, and the interpolant is at most a margin off
/// either way -- and its look-ahead and settle stay about as close. Listed to twice that, a block's
/// list answers every one of them exactly: a point whose nearest listed segment is within this of
/// it can have no nearer unlisted one, since everything unlisted is further than this from the
/// whole block. Beyond it the baked reading stands, a bound where nothing is drawn. The shader's
/// `LIST_REACH` is the same number, and `cpu_march` pins the two together.
pub const LIST_REACH: f64 = 4.0;

/// No fine data for this block; read the coarse level instead.
const NO_BLOCK: u32 = u32::MAX;

/// A profile's signed distance on a two-level grid, with margins that keep it a lower bound.
#[derive(Clone, Debug)]
pub struct BakedProfile {
    /// The lower corner of the grid, in the profile's own plane.
    lo: Point2<f64>,
    /// The spacing between fine samples, in millimetres.
    cell: f64,
    /// Fine samples across and up, counting both edges.
    wide: usize,
    high: usize,
    blocks_wide: usize,
    blocks_high: usize,
    /// One sample at every block corner, over the whole box: sign and a safe bound everywhere.
    coarse: Vec<f32>,
    /// Where each block's fine samples start, or [`NO_BLOCK`].
    page: Vec<u32>,
    /// `(BLOCK + 1)²` samples per stored block, packed end to end.
    fine: Vec<f32>,
    margin: f64,
    coarse_margin: f64,
    /// For every block cell, where its segment list starts in `seglist` and how long it is,
    /// interleaved: the outline's segments within [`LIST_REACH`] margins of the block, by which an
    /// exact read inside the block walks a few segments rather than the whole outline. Empty for
    /// a bake of anything that is not an outline, such as a wall's folded distance.
    segs: Vec<u32>,
    seglist: Vec<u32>,
    /// Whether the outline turns anticlockwise, which is what the exact read's sign is taken from.
    ccw: bool,
}

/// What can answer from a grid: a profile, and a wall.
///
/// **So that one rule bakes both** — `bake_to_tolerance` in `graph_field.rs` — which is the
/// point of saying what they share. When a section is baked, how finely, and under what key a
/// grid is kept are each decided once; a wall baked by a rule of its own would be a second set
/// of all three for the shader and the harness to match, and D9 is the name for that.
pub(crate) trait Bakes: Sized {
    /// The box the zero set lies in, in the section's own plane.
    fn bounds(&self) -> (Point2<f64>, Point2<f64>);
    /// The identity of its bake: a hash of everything its distance is a function of.
    fn fingerprint(&self) -> u64;
    /// The same thing, answering from a grid it builds.
    fn baked(self, cells: usize, reads: (f64, f64)) -> Self;
    /// The same thing, answering from a grid already built for it.
    fn with_bake(self, bake: BakedProfile) -> Self;
    fn bake(&self) -> Option<&BakedProfile>;
}

impl Bakes for Profile {
    fn bounds(&self) -> (Point2<f64>, Point2<f64>) {
        Profile::bounds(self)
    }
    fn fingerprint(&self) -> u64 {
        Profile::fingerprint(self)
    }
    fn baked(self, cells: usize, reads: (f64, f64)) -> Self {
        Profile::baked(self, cells, reads)
    }
    fn with_bake(self, bake: BakedProfile) -> Self {
        Profile::with_bake(self, bake)
    }
    fn bake(&self) -> Option<&BakedProfile> {
        Profile::bake(self)
    }
}

impl Bakes for crate::Wall {
    fn bounds(&self) -> (Point2<f64>, Point2<f64>) {
        crate::Wall::bounds(self)
    }
    fn fingerprint(&self) -> u64 {
        crate::Wall::fingerprint(self)
    }
    fn baked(self, cells: usize, reads: (f64, f64)) -> Self {
        crate::Wall::baked(self, cells, reads)
    }
    fn with_bake(self, bake: BakedProfile) -> Self {
        crate::Wall::with_bake(self, bake)
    }
    fn bake(&self) -> Option<&BakedProfile> {
        crate::Wall::bake(self)
    }
}

/// A baked profile laid out for a shader, which cannot follow a structure of its own.
pub struct ShaderBake<'a> {
    pub lo: Point2<f64>,
    pub cell: f64,
    pub wide: usize,
    pub high: usize,
    pub block: usize,
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub margin: f64,
    pub coarse_margin: f64,
    pub coarse: &'a [f32],
    pub page: &'a [u32],
    pub fine: &'a [f32],
    /// Per block cell, (start, count) into `seglist`; see `BakedProfile::index_segments`.
    pub segs: &'a [u32],
    pub seglist: &'a [u32],
    pub ccw: bool,
}

/// The distance from the segment `a`-`b` to the box `lo`-`hi`: nought where they meet, and
/// otherwise attained at an end of the segment or a corner of the box, both being convex.
fn segment_box_distance(a: Point2<f64>, b: Point2<f64>, lo: Point2<f64>, hi: Point2<f64>) -> f64 {
    let inside = |p: Point2<f64>| p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y;
    if inside(a) || inside(b) {
        return 0.0;
    }
    let corners = [lo, Point2::new(hi.x, lo.y), hi, Point2::new(lo.x, hi.y)];
    // Crossing an edge of the box is meeting it.
    for e in 0..4 {
        let (c, d) = (corners[e], corners[(e + 1) % 4]);
        let cross = |u: nalgebra::Vector2<f64>, v: nalgebra::Vector2<f64>| u.x * v.y - u.y * v.x;
        let (ab, cd) = (b - a, d - c);
        let denom = cross(ab, cd);
        if denom.abs() > 1e-18 {
            let t = cross(c - a, cd) / denom;
            let u = cross(c - a, ab) / denom;
            if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                return 0.0;
            }
        }
    }
    let to_box = |p: Point2<f64>| {
        let dx = (lo.x - p.x).max(0.0).max(p.x - hi.x);
        let dy = (lo.y - p.y).max(0.0).max(p.y - hi.y);
        (dx * dx + dy * dy).sqrt()
    };
    let to_segment = |p: Point2<f64>| {
        let ab = b - a;
        let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
        (p - (a + ab * t)).norm()
    };
    to_box(a).min(to_box(b)).min(corners.iter().map(|c| to_segment(*c)).fold(f64::INFINITY, f64::min))
}

pub(crate) fn bake_padding(span_mm: f64, reads: (f64, f64)) -> f64 {
    // Only outward reading pushes the *box* out; reading inward stays within the profile's own
    // bounds and needs no more room than the quarter already there.
    span_mm * 0.25 + band_range(span_mm, reads).1.max(0.0)
}

/// The contour range the fine band actually covers, rounded outward in steps.
///
/// **Rounded, because everything that feeds it is something a user drags** -- a depth, a draft
/// angle, a fillet radius. An unrounded band would be a different grid on every frame of a
/// drag, and a grid is rebuilt by walking the boundary once per sample, which is the most
/// expensive thing here by an order of magnitude. In sixteenths of the profile's own span, so
/// the steps scale with the part rather than being a millimetre figure that means something
/// different on a watch case and on a wing spar.
///
/// Always rounds *outward*, so the band still covers everything asked for.
pub fn band_range(span_mm: f64, reads: (f64, f64)) -> (f64, f64) {
    let round = |v: f64| {
        if !v.is_finite() || v == 0.0 || !span_mm.is_finite() {
            return 0.0;
        }
        let step = (span_mm.abs() * 0.0625).max(1e-9);
        v.signum() * (v.abs() / step).ceil() * step
    };
    let lo = round(reads.0.min(0.0));
    let hi = round(reads.1.max(0.0));
    (lo, hi)
}

/// How many fine blocks a bake will store before it coarsens its grid instead.
///
/// **A budget rather than a refusal.** A fillet reads its operand out to its own radius, so a
/// 10 mm fillet asks for a 10 mm band, and on a fine tolerance that is most of the box -- tens
/// of megabytes, and seconds of boundary walking, on something a user is dragging. Refusing to
/// bake at all is worse than it sounds: the exact walk is what the bake exists to avoid, at
/// 1628 ms per million queries on a tessellated profile, and it would arrive as the viewport
/// falling over exactly when a part gets interesting.
///
/// So the grid coarsens until the band fits. That keeps the field *smooth everywhere it is
/// read*, which is what a blend needs, and spends the precision it can afford. The alternative
/// -- a fine grid with a band that stops short -- is the failure this whole parameter exists to
/// fix: a step in the operand, which a fillet turns into a terrace.
const MAX_BLOCKS: usize = 32_768;

/// How many cells a bake needs to stay inside a tolerance, or `None` if it cannot.
///
/// **The rule that keeps a bake a display convenience rather than a quiet loss of accuracy.** A
/// grid is short by half a cell's diagonal, so a tolerance fixes the cell size, and the span
/// fixes how many of them.
///
/// A *quarter* of the tolerance, which the band paid for. A dense grid could only afford half —
/// twice as fine is four times the area — so the bake was allowed to eat half the error budget of
/// whatever asked for it. A band is a boundary rather than an area, so twice as fine is a little
/// under twice the memory: measured on a 200 x 140 mm ring by `examples/profile_bake_cost.rs`,
/// **0.37 mm for 181 KB where the dense grid spent 220 KB to manage 0.74 mm**. Better and
/// smaller at once, which is rare enough to be worth the sentence.
///
/// `None` when the answer would need a finer grid than is worth holding — which is what export
/// asks for, and where the exact boundary is the right answer anyway. So "tolerance only ever
/// affects output" keeps meaning what it says: a fine tolerance simply does not bake.
///
/// The ceiling is generous now that the fine level follows the boundary rather than filling the
/// box: what a finer bake costs is a longer band, not a larger area.
pub fn bake_cells(span_mm: f64, tolerance_mm: f64, reads: (f64, f64)) -> Option<usize> {
    if !(span_mm.is_finite() && tolerance_mm.is_finite()) || span_mm <= 0.0 || tolerance_mm <= 0.0 {
        return None;
    }
    if !(reads.0.is_finite() && reads.1.is_finite()) {
        return None;
    }
    // Wide enough for the profile *and* for whatever is read off it, because the two travel
    // together: sizing the grid for one and laying the box out for the other would make the cell
    // coarser than the tolerance asked for, silently.
    let across = span_mm + 2.0 * bake_padding(span_mm, reads);
    // A *quarter*, and measured rather than assumed: halving the precision was tried and made
    // no difference to a frame at all -- 25.5 ms against 24.7 on a patterned revolve, with the
    // control moving as much. What the band costs is the dependent read, one per lookup, not the
    // number of samples behind it. So the finer bake is free and there is nothing to trade.
    let cell = tolerance_mm * 0.25 * std::f64::consts::SQRT_2;
    let cells = (across / cell).ceil();
    if !cells.is_finite() || cells > 8192.0 {
        return None;
    }
    Some((cells as usize).clamp(16, 8192))
}

impl BakedProfile {
    /// Samples `profile` onto a grid of about `cells` along its longer side.
    ///
    /// The grid is padded beyond the profile's own bounds so that the region a solid actually
    /// occupies falls inside it, rather than immediately into the fallback.
    ///
    /// `reads_out_to` is **how far off the profile's own boundary the caller will ask**, in
    /// millimetres, and it is the difference between a band that works and one that quietly
    /// does not. A plain extrude reads the zero set and nothing else, which is what the band is
    /// built around. A *drafted* one does not: at height `z` its wall is the `d = z*tan(draft)`
    /// contour, so it reads out to `depth * tan(draft)` -- 3.2 mm at 12 mm and 15 degrees,
    /// against a band four tolerances wide. Everything past the band is answered by the coarse
    /// level, and the step between the two levels' margins is a *cliff*: measured at 174 mm per
    /// millimetre, where a distance may rise at most one. A ray told it has that much room goes
    /// straight through the surface, which is the speckle the drafted walls showed.
    ///
    /// **Signed**, because a draft reads one side and not both: outward for a positive angle,
    /// inward for a negative one, never the other. Banding both ways would double the block
    /// count to store contours nothing asks for. Zero for anything that only wants the
    /// boundary. `examples/draft_limits.rs` measures the cliff, the cure and what it costs.
    pub fn new(profile: &Profile, cells: usize, reads: (f64, f64)) -> Self {
        let mut baked = Self::of(profile.bounds(), |q| profile.walk(q), cells, reads);
        baked.index_segments(profile.points());
        baked
    }

    /// Lists, for each stored block, the outline's segments within [`LIST_REACH`] margins of it.
    fn index_segments(&mut self, points: &[Point2<f64>]) {
        let n = points.len();
        if n < 3 {
            return;
        }
        let area: f64 = (0..n).map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        }).sum();
        self.ccw = area > 0.0;
        let reach = LIST_REACH * self.margin;
        let side = BLOCK as f64 * self.cell;
        // Each segment against the few blocks its own box, widened by the reach, covers -- not
        // every stored block against every segment, which cost the lens's bake sixty times itself
        // (3.2 to 195 ms at Full, `knife_scale`): a segment is about a cell long and the reach a
        // few cells, so this is a handful of box tests a segment, in the same order as before.
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); self.page.len()];
        let (wide, high) = (self.blocks_wide as i64, self.blocks_high as i64);
        let block_of = |v: f64, origin: f64| ((v - origin) / side).floor() as i64;
        for i in 0..n {
            let (a, b) = (points[i], points[(i + 1) % n]);
            if (b - a).norm_squared() == 0.0 {
                continue;
            }
            let (bi0, bi1) = (block_of(a.x.min(b.x) - reach, self.lo.x).max(0), block_of(a.x.max(b.x) + reach, self.lo.x).min(wide - 1));
            let (bj0, bj1) = (block_of(a.y.min(b.y) - reach, self.lo.y).max(0), block_of(a.y.max(b.y) + reach, self.lo.y).min(high - 1));
            for bj in bj0..=bj1 {
                for bi in bi0..=bi1 {
                    let k = (bj * wide + bi) as usize;
                    if self.page[k] == NO_BLOCK {
                        continue;
                    }
                    let lo = Point2::new(self.lo.x + bi as f64 * side, self.lo.y + bj as f64 * side);
                    let hi = Point2::new(lo.x + side, lo.y + side);
                    if segment_box_distance(a, b, lo, hi) <= reach {
                        lists[k].push(i as u32);
                    }
                }
            }
        }
        let mut segs = vec![0u32; 2 * self.page.len()];
        let mut seglist: Vec<u32> = Vec::new();
        for (k, list) in lists.iter().enumerate() {
            segs[2 * k] = seglist.len() as u32;
            segs[2 * k + 1] = list.len() as u32;
            seglist.extend_from_slice(list);
        }
        self.segs = segs;
        self.seglist = seglist;
    }

    /// The segment lists, for a test that holds them to the outline.
    pub fn listed_for_block(&self, bi: usize, bj: usize) -> &[u32] {
        let k = bj * self.blocks_wide + bi;
        let (start, count) = (self.segs[2 * k] as usize, self.segs[2 * k + 1] as usize);
        &self.seglist[start..start + count]
    }

    /// The block grid, as (blocks wide, blocks high, the block's side in mm, the grid's corner).
    pub fn blocks(&self) -> (usize, usize, f64, Point2<f64>) {
        (self.blocks_wide, self.blocks_high, BLOCK as f64 * self.cell, self.lo)
    }

    /// The same grid over any exact distance in a plane, of which a profile's is the first and
    /// a wall's the second.
    ///
    /// `bounds` is the box the zero set lies in, and `distance` the exact reading at a point.
    /// **What the grid needs of it is what the margins were worked out from, and nothing about
    /// a boundary of points**: that it is exact where it is sampled, and that it rises no faster
    /// than a millimetre a millimetre. The first is what makes the band's choice of blocks
    /// tight, the second what makes half a cell's diagonal enough to subtract, and the box is
    /// what the reading outside the grid leans on. A sampled function that is itself a grid, or
    /// a bound rather than a distance, would satisfy the type and none of the three.
    pub(crate) fn of(
        bounds: (Point2<f64>, Point2<f64>),
        distance: impl Fn(Point2<f64>) -> f64 + Sync,
        cells: usize,
        reads: (f64, f64),
    ) -> Self {
        // Coarsen until the band fits the budget, rather than storing a band that stops short.
        // Laying the grid out and choosing the blocks is cheap -- one walk per block corner --
        // and each halving makes the next attempt four times cheaper, so this settles fast.
        let mut cells = cells.max(2);
        loop {
            let plan = Plan::lay_out(bounds, &distance, cells, reads);
            if plan.kept.len() <= MAX_BLOCKS || cells <= 32 {
                return plan.fill(&distance);
            }
            cells /= 2;
        }
    }

    /// The margin subtracted from a reading near the surface, in millimetres.
    ///
    /// Reported because it is the accuracy of this bake where accuracy matters — the coarse
    /// level's own margin is wider, and applies only where the surface is not.
    pub fn margin(&self) -> f64 {
        self.margin
    }

    /// How many numbers it holds, for anyone weighing a resolution against memory.
    pub fn samples(&self) -> usize {
        self.coarse.len() + self.fine.len() + self.page.len()
    }

    /// How many blocks carry fine data, against how many there are.
    pub fn stored_blocks(&self) -> (usize, usize) {
        (
            self.page.iter().filter(|p| **p != NO_BLOCK).count(),
            self.page.len(),
        )
    }

    /// The whole thing laid out for a shader.
    pub fn for_shader(&self) -> ShaderBake<'_> {
        ShaderBake {
            lo: self.lo,
            cell: self.cell,
            wide: self.wide,
            high: self.high,
            block: BLOCK,
            blocks_wide: self.blocks_wide,
            blocks_high: self.blocks_high,
            margin: self.margin,
            coarse_margin: self.coarse_margin,
            coarse: &self.coarse,
            page: &self.page,
            fine: &self.fine,
            segs: &self.segs,
            seglist: &self.seglist,
            ccw: self.ccw,
        }
    }

    /// The distance to the profile, never more than the truth.
    pub fn distance(&self, q: Point2<f64>) -> f64 {
        let last = (self.wide - 1) as f64;
        let tall = (self.high - 1) as f64;
        let x = (q.x - self.lo.x) / self.cell;
        let y = (q.y - self.lo.y) / self.cell;

        let cx = x.clamp(0.0, last);
        let cy = y.clamp(0.0, tall);
        let here = self.sample(cx, cy);

        // **Outside the grid, Pythagoras against the nearest point on it.** The profile lies
        // inside the box and the clamp is componentwise, so on every axis where the query is
        // outside, its gap to any interior point is the gap to the clamp *plus* the clamp's own,
        // with the two the same sign — squaring and summing gives the bound.
        if x < 0.0 || y < 0.0 || x > last || y > tall {
            let away = ((x - cx) * self.cell).hypot((y - cy) * self.cell);
            return away.hypot(here.max(0.0));
        }
        here
    }

    /// Bilinear at a position already on the grid: from the block if there is one, else coarse.
    fn sample(&self, x: f64, y: f64) -> f64 {
        let bi = ((x / BLOCK as f64).floor() as usize).min(self.blocks_wide - 1);
        let bj = ((y / BLOCK as f64).floor() as usize).min(self.blocks_high - 1);
        let start = self.page[bj * self.blocks_wide + bi];
        if start != NO_BLOCK {
            let lx = x - (bi * BLOCK) as f64;
            let ly = y - (bj * BLOCK) as f64;
            let i = (lx.floor() as usize).min(BLOCK - 1);
            let j = (ly.floor() as usize).min(BLOCK - 1);
            let stride = BLOCK + 1;
            let base = start as usize + j * stride + i;
            let read = |k: usize| self.fine[k] as f64;
            let (fx, fy) = (lx - i as f64, ly - j as f64);
            let top = read(base) * (1.0 - fx) + read(base + 1) * fx;
            let bottom = read(base + stride) * (1.0 - fx) + read(base + stride + 1) * fx;
            return top * (1.0 - fy) + bottom * fy - self.margin;
        }

        // No block here, so the surface is not either: the coarse reading is enough, with the
        // wider margin a wider cell owes.
        let stride = self.blocks_wide + 1;
        let (gx, gy) = (x / BLOCK as f64, y / BLOCK as f64);
        let i = (gx.floor() as usize).min(self.blocks_wide - 1);
        let j = (gy.floor() as usize).min(self.blocks_high - 1);
        let base = j * stride + i;
        let read = |k: usize| self.coarse[k] as f64;
        let (fx, fy) = (gx - i as f64, gy - j as f64);
        let top = read(base) * (1.0 - fx) + read(base + 1) * fx;
        let bottom = read(base + stride) * (1.0 - fx) + read(base + stride + 1) * fx;
        let here = top * (1.0 - fy) + bottom * fy;
        // **Toward nought, not down.** Outside that is the margin subtracted, as it always was.
        // Inside it is the margin taken off the *depth*, where subtracting it made the section
        // deeper than it is — see the head of this file for what that cost a cut.
        if here < 0.0 {
            here + self.coarse_margin
        } else {
            here - self.coarse_margin
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every segment within reach of a block is in that block's list, and the lists are short**
    /// (backlog item 115). The lens of the owner's wing, with its three sharp corners, at the bake
    /// Full gives it: for every stored block, every segment of the outline within `LIST_REACH`
    /// margins of the block's box is listed -- checked against a brute-force distance from each
    /// segment to the box -- and no block lists more than a few dozen, which is what makes the
    /// exact read cheap. The outline's turn is read as anticlockwise, which is how it was drawn.
    #[test]
    fn every_segment_within_reach_of_a_block_is_in_its_list() {
        let lens = Profile::new(vec![
            Point2::new(-20.0, 0.0),
            Point2::new(0.0, -4.35),
            Point2::new(20.0, 0.0),
            Point2::new(10.0, 5.0),
            Point2::new(0.0, 12.3),
            Point2::new(-10.0, 5.0),
        ])
        .unwrap();
        let baked = BakedProfile::new(&lens, 1765, (0.0, 3.0));
        assert!(baked.ccw, "drawn anticlockwise");
        let (wide, high, side, lo) = baked.blocks();
        let points = lens.points();
        let n = points.len();
        let reach = LIST_REACH * baked.margin();
        let mut stored = 0;
        let mut longest = 0;
        for bj in 0..high {
            for bi in 0..wide {
                if baked.page[bj * wide + bi] == NO_BLOCK {
                    assert!(baked.listed_for_block(bi, bj).is_empty());
                    continue;
                }
                stored += 1;
                let listed = baked.listed_for_block(bi, bj);
                longest = longest.max(listed.len());
                let blo = Point2::new(lo.x + bi as f64 * side, lo.y + bj as f64 * side);
                let bhi = Point2::new(blo.x + side, blo.y + side);
                for i in 0..n {
                    let (a, b) = (points[i], points[(i + 1) % n]);
                    // Brute force: the least distance from the box to the segment over a fine
                    // sampling of both, which can only overstate the true distance.
                    let mut least = f64::INFINITY;
                    for s in 0..=64 {
                        let q = a + (b - a) * (s as f64 / 64.0);
                        let dx = (blo.x - q.x).max(0.0).max(q.x - bhi.x);
                        let dy = (blo.y - q.y).max(0.0).max(q.y - bhi.y);
                        least = least.min((dx * dx + dy * dy).sqrt());
                    }
                    if least <= reach {
                        assert!(listed.contains(&(i as u32)), "block ({bi}, {bj}) misses segment {i} at {least:.3} mm");
                    }
                }
            }
        }
        assert!(stored > 100, "{stored} blocks stored");
        assert!(longest <= 48, "the longest list is {longest} segments");
    }

    fn square(half: f64) -> Profile {
        Profile::new(vec![
            Point2::new(-half, -half),
            Point2::new(half, -half),
            Point2::new(half, half),
            Point2::new(-half, half),
        ])
        .expect("a square is a profile")
    }

    /// A baked profile never says a point is further from the boundary than it is.
    ///
    /// **The property everything else depends on.** A ray marches by the distance it is told, so
    /// a reading that is too large steps straight through the surface — the failure is a hole in
    /// the model, not a rough edge. Interpolation overstates across a crease by its nature, which
    /// is why the margins exist, and this is what says they are enough. Over both levels at once:
    /// near the boundary, where the fine blocks answer, and far from it, where the coarse grid
    /// does — and the join between them, which is the part a two-level scheme gets wrong.
    ///
    /// **From outside, and from inside.** This asserted `told <= truth` for every point until
    /// 2026-10-02, which reads as "never overstates" and for a point inside the profile means
    /// the opposite: a told depth of 24 mm against a true one of 20 passes it. A section cut
    /// out of a block is marched by exactly that depth. So it is said per side: outside, the
    /// reading is no more than the distance; inside, the depth is no more than the true depth
    /// wherever the coarse level answers, and within two fine margins of it in the band, where
    /// the margin is kept one-sided so the section has one surface.
    #[test]
    fn a_baked_profile_never_overstates_the_distance() {
        let exact = square(20.0);
        // Twice, because at the first resolution a block is a fifth of the square across and the
        // band is the whole of its inside: nothing in there is answered by the coarse level, so
        // the coarse level's depth went untested for as long as this was the only grid tried.
        for (cells, wants_deep) in [(64, false), (256, true)] {
            let baked = BakedProfile::new(&exact, cells, (0.0, 0.0));

            let (mut over_outside, mut over_inside, mut over_deep) = (0.0f64, 0.0f64, 0.0f64);
            let mut worst_near: f64 = 0.0;
            // Deeper than the fine band can reach: a block's diagonal past the surface, and a
            // second for the block the probe itself is in.
            let band = baked.margin() * 2.0 * (BLOCK as f64) * 2.0;
            let mut deep_points = 0;
            for i in 0..200 {
                for j in 0..200 {
                    let q = Point2::new(-40.0 + i as f64 * 0.4, -40.0 + j as f64 * 0.4);
                    let told = baked.distance(q);
                    let truth = exact.distance(q);
                    if truth >= 0.0 {
                        over_outside = over_outside.max(told - truth);
                    } else {
                        over_inside = over_inside.max(told.abs() - truth.abs());
                        if truth < -band {
                            deep_points += 1;
                            over_deep = over_deep.max(told.abs() - truth.abs());
                        }
                    }
                    // Near the surface is where the fine level answers and where the accuracy
                    // claim applies; further out the coarse level is deliberately looser.
                    if truth.abs() < 3.0 {
                        worst_near = worst_near.max(truth - told);
                    }
                }
            }
            assert!(over_outside <= 0.0, "{cells} cells: overstated by {over_outside:.4} mm outside");
            assert!(
                !wants_deep || deep_points > 100,
                "{cells} cells: only {deep_points} probes were past the band, so the coarse \
                 level's depth was not tested",
            );
            assert!(
                over_deep <= 0.0,
                "{cells} cells: claimed {over_deep:.4} mm more depth than there is, past the band",
            );
            assert!(
                over_inside <= baked.margin() * 2.0,
                "{cells} cells: claimed {over_inside:.4} mm more depth than there is, against a \
                 margin of {:.4}",
                baked.margin(),
            );
            assert!(
                worst_near < baked.margin() * 3.0,
                "{cells} cells: understated by {worst_near:.4} mm near the surface, against a \
                 margin of {:.4}",
                baked.margin(),
            );
        }
    }

    /// The sign survives baking, which is what says a solid is still solid.
    #[test]
    fn a_baked_profile_agrees_about_inside_and_outside() {
        let exact = square(20.0);
        let baked = BakedProfile::new(&exact, 128, (0.0, 0.0));
        for (q, inside) in [
            (Point2::new(0.0, 0.0), true),
            (Point2::new(10.0, 10.0), true),
            (Point2::new(30.0, 0.0), false),
            (Point2::new(0.0, -35.0), false),
        ] {
            assert_eq!(baked.distance(q) < 0.0, inside, "at {q:?}");
        }
    }

    /// The fine level follows the boundary, so most of the box holds nothing.
    ///
    /// **The whole reason for two levels.** A dense grid fine enough to matter is unaffordable
    /// and a band of the same fineness is not, but only while the band stays a band. If this ever
    /// stops holding, the bake has quietly become a square again — and the memory would say so
    /// long before anybody noticed the pictures.
    #[test]
    fn only_the_blocks_the_boundary_passes_through_are_stored() {
        // At a fine resolution, which is the regime a band exists for. A block is wide enough
        // to be a fair share of a coarse grid, so at 128 cells the band *is* most of the box and
        // the two levels buy nothing -- the saving arrives as the grid gets finer, because the
        // band's width stays a few blocks while the box grows as the square of the resolution.
        let baked = BakedProfile::new(&square(20.0), 1024, (0.0, 0.0));
        let (stored, total) = baked.stored_blocks();
        assert!(
            stored * 5 < total,
            "stored {stored} blocks of {total}: the band is not a band",
        );
    }
}

/// A grid laid out and its band chosen, before the expensive part.
///
/// Separate from filling it because the choice of how fine to be depends on how many blocks the
/// band comes to, and that is only known once the coarse level has been walked. Walking the
/// coarse level is one sample per block corner; filling the fine level is eighty-one per block,
/// so trying a layout costs a small fraction of committing to it.
struct Plan {
    lo: Point2<f64>,
    cell: f64,
    wide: usize,
    high: usize,
    blocks_wide: usize,
    blocks_high: usize,
    coarse: Vec<f32>,
    page: Vec<u32>,
    kept: Vec<usize>,
    margin: f64,
    coarse_margin: f64,
}

impl Plan {
    fn lay_out(
        bounds: (Point2<f64>, Point2<f64>),
        distance: &(impl Fn(Point2<f64>) -> f64 + Sync),
        cells: usize,
        reads: (f64, f64),
    ) -> Self {
        let (lo, hi) = bounds;
        let span = (hi - lo).max().max(1e-6);
        let pad = bake_padding(span, reads);
        // The same rounding the padding used, so the band and the box it sits in are laid out
        // from one number rather than two that agree until they do not.
        let (want_lo, want_hi) = band_range(span, reads);
        let lo = Point2::new(lo.x - pad, lo.y - pad);
        let hi = Point2::new(hi.x + pad, hi.y + pad);

        let cells = cells.max(2);
        let cell = ((hi - lo).max()) / cells as f64;
        let blocks_wide = (((hi.x - lo.x) / cell).ceil() as usize).div_ceil(BLOCK).max(1);
        let blocks_high = (((hi.y - lo.y) / cell).ceil() as usize).div_ceil(BLOCK).max(1);
        let wide = blocks_wide * BLOCK + 1;
        let high = blocks_high * BLOCK + 1;

        let at = |i: usize, j: usize| Point2::new(lo.x + i as f64 * cell, lo.y + j as f64 * cell);

        // The coarse level first: one sample per block corner, over the whole box.
        //
        // **In parallel, because every sample is an independent walk of the boundary** and the
        // walk is linear in the segment count -- so a tessellated curve makes this the whole
        // cost of a bake. Measured on a 500-segment loop at the finest display setting: 2445 ms
        // in one thread. Nothing here reads another sample, so there is no order to keep.
        let across = blocks_wide + 1;
        let mut coarse = vec![0.0f32; across * (blocks_high + 1)];
        coarse.par_iter_mut().enumerate().for_each(|(k, v)| {
            *v = distance(at((k % across) * BLOCK, (k / across) * BLOCK)) as f32;
        });

        let margin = cell * std::f64::consts::SQRT_2 * 0.5;
        let coarse_margin = margin * BLOCK as f64;

        // Then the fine level, only where a contour the caller reads can pass. A block's own
        // corners settle that exactly: every point in a block is within one diagonal of a
        // corner, so a block whose values all sit a diagonal clear of the wanted range cannot
        // contain any of it. Tight rather than generous, and it can be, because these samples
        // are the boundary walked exactly -- `coarse_margin` is what interpolating *between*
        // them costs at read time and has no business in this test. Adding it here made the
        // band two-thirds of the box.
        let reach = (BLOCK as f64 * cell) * std::f64::consts::SQRT_2;
        let per_block = (BLOCK + 1) * (BLOCK + 1);
        let mut page = vec![NO_BLOCK; blocks_wide * blocks_high];
        let mut kept: Vec<usize> = Vec::new();
        for bj in 0..blocks_high {
            for bi in 0..blocks_wide {
                let corners = [
                    coarse[bj * across + bi] as f64,
                    coarse[bj * across + bi + 1] as f64,
                    coarse[(bj + 1) * across + bi] as f64,
                    coarse[(bj + 1) * across + bi + 1] as f64,
                ];
                let low = corners.iter().fold(f64::INFINITY, |a, v| a.min(*v));
                let high = corners.iter().fold(f64::NEG_INFINITY, |a, v| a.max(*v));
                if low - reach > want_hi || high + reach < want_lo {
                    continue;
                }
                page[bj * blocks_wide + bi] = (kept.len() * per_block) as u32;
                kept.push(bj * blocks_wide + bi);
            }
        }

        Self {
            lo,
            cell,
            wide,
            high,
            blocks_wide,
            blocks_high,
            coarse,
            page,
            kept,
            margin,
            coarse_margin,
        }
    }

    fn fill(self, distance: &(impl Fn(Point2<f64>) -> f64 + Sync)) -> BakedProfile {
        let stride = BLOCK + 1;
        let per_block = stride * stride;
        let at = |i: usize, j: usize| {
            Point2::new(self.lo.x + i as f64 * self.cell, self.lo.y + j as f64 * self.cell)
        };
        // One sample past the block's own edge in each direction, so interpolating inside a
        // block never has to reach into the next one.
        let mut fine = vec![0.0f32; self.kept.len() * per_block];
        let blocks_wide = self.blocks_wide;
        fine.par_chunks_mut(per_block).zip(self.kept.par_iter()).for_each(
            |(block, at_block)| {
                let (bi, bj) = (at_block % blocks_wide, at_block / blocks_wide);
                for j in 0..=BLOCK {
                    for i in 0..=BLOCK {
                        block[j * stride + i] =
                            distance(at(bi * BLOCK + i, bj * BLOCK + j)) as f32;
                    }
                }
            },
        );

        BakedProfile {
            lo: self.lo,
            cell: self.cell,
            wide: self.wide,
            high: self.high,
            blocks_wide: self.blocks_wide,
            blocks_high: self.blocks_high,
            coarse: self.coarse,
            page: self.page,
            fine,
            segs: Vec::new(),
            seglist: Vec::new(),
            ccw: true,
            margin: self.margin,
            coarse_margin: self.coarse_margin,
        }
    }
}
