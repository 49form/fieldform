// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A body turned into a finished mesh — the whole export pipeline, stoppable.
//!
//! **One function, because it was one closure in the application.** Mesh, weld, reduce, relax and
//! reshape ran in that order inside `App::begin_export`, where nothing else could call them and
//! nothing could stop them. Backlog item 53 makes a tessellation a block that can be cancelled
//! and re-run by itself when the geometry changes, and both of those have to reach into every
//! stage — so the stages are here, with the cancel checked inside each one rather than between
//! them. A token checked only between stages would let a cancelled Fine mesh finish its half
//! minute of meshing before noticing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use nalgebra::Point3;

use crate::{FieldMesh, MeshTolerance, ScalarField};

/// Asks a running tessellation to stop — and is how it stops itself when it runs out of memory.
///
/// **Cooperative, because a thread cannot be killed.** Every stage looks at this at points where
/// giving up is cheap — each octree cell as the mesher descends, each pass of the passes after
/// it — and returns without finishing. A clone is the same token, so the block that started a
/// run keeps one and the run takes the other.
///
/// **Memory goes through the same token**, because running out is the other reason a run has to
/// stop part way, and every stage already looks here. The large buffers are charged against a
/// limit as they grow (`charge`), and grown with fallible allocation; a run that passes its limit,
/// or that the allocator refuses, trips the token exactly as a cancel does, and the stage that
/// noticed returns. The difference is only in what the run then reports. An allocation the
/// standard library makes infallibly aborts the process when it fails — it does not unwind — so
/// the limit is what keeps a run from reaching that point: it stops while the allocator still
/// says yes.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<Token>);

#[derive(Debug, Default)]
struct Token {
    /// [`RUNNING`], [`CANCELLED`] or [`OUT_OF_MEMORY`].
    state: AtomicU8,
    /// Bytes the run may hold before it stops itself. Zero is no limit, which is the default, so
    /// a token made by `new` behaves as it always did.
    limit: AtomicUsize,
    used: AtomicUsize,
}

const RUNNING: u8 = 0;
const CANCELLED: u8 = 1;
const OUT_OF_MEMORY: u8 = 2;

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    /// A token for a run that may hold at most `bytes` in the buffers it charges.
    pub fn with_memory_limit(bytes: usize) -> Self {
        let token = Self::default();
        token.0.limit.store(bytes.max(1), Ordering::Relaxed);
        token
    }

    /// Asks every holder of this token to stop. Idempotent, and cannot be taken back: a run that
    /// has been told to stop is over, and the next one gets a fresh token.
    pub fn cancel(&self) {
        let _ = self.0.state.compare_exchange(RUNNING, CANCELLED, Ordering::Relaxed, Ordering::Relaxed);
    }

    /// Relaxed, because nothing is published through it — a stage that sees it a cell late
    /// stops a cell late, which is the whole cost of being wrong.
    pub fn is_cancelled(&self) -> bool {
        self.0.state.load(Ordering::Relaxed) != RUNNING
    }

    /// Whether the run stopped because it ran out of memory rather than because it was told to.
    pub fn ran_out_of_memory(&self) -> bool {
        self.0.state.load(Ordering::Relaxed) == OUT_OF_MEMORY
    }

    /// Stops the run as out of memory: the limit was passed, or an allocation was refused.
    pub(crate) fn out_of_memory(&self) {
        let _ =
            self.0.state.compare_exchange(RUNNING, OUT_OF_MEMORY, Ordering::Relaxed, Ordering::Relaxed);
    }

    /// Records `bytes` more held by the run, and answers whether it may go on. Past the limit
    /// the token is tripped and every stage stops at its next look.
    pub(crate) fn charge(&self, bytes: usize) -> bool {
        let used = self.0.used.fetch_add(bytes, Ordering::Relaxed).saturating_add(bytes);
        let limit = self.0.limit.load(Ordering::Relaxed);
        if limit != 0 && used > limit {
            self.out_of_memory();
            return false;
        }
        true
    }

    /// Records `bytes` given back.
    pub(crate) fn release(&self, bytes: usize) {
        let _ = self.0.used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |u| {
            Some(u.saturating_sub(bytes))
        });
    }

    /// What the run holds now, by its own charges.
    pub fn used(&self) -> usize {
        self.0.used.load(Ordering::Relaxed)
    }

    /// The most the run may hold, or `None` for no limit.
    pub fn memory_limit(&self) -> Option<usize> {
        match self.0.limit.load(Ordering::Relaxed) {
            0 => None,
            limit => Some(limit),
        }
    }

    /// Charges `bytes` for as long as the guard lives — what a stage holds while it runs and gives
    /// back when it returns, by any path. `None`, with the token tripped, when that is past the
    /// limit.
    pub(crate) fn hold(&self, bytes: usize) -> Option<Held<'_>> {
        if self.charge(bytes) {
            Some(Held { cancel: self, bytes })
        } else {
            self.release(bytes);
            None
        }
    }

    /// A vector of `n` copies of `value`, allocated fallibly: `None`, with the token tripped,
    /// when the allocator refuses. Not charged — a stage charges its whole working set at once.
    pub(crate) fn filled<T: Clone>(&self, n: usize, value: T) -> Option<Vec<T>> {
        let mut out = Vec::new();
        if out.try_reserve_exact(n).is_err() {
            self.out_of_memory();
            return None;
        }
        out.resize(n, value);
        Some(out)
    }

    /// Grows `buffer` to take `more` items, charged and fallibly: `false`, with the token
    /// tripped, when either the limit or the allocator says no. Grows by doubling, as `push`
    /// would, so a buffer grown an item at a time is not reallocated an item at a time.
    pub(crate) fn reserve<T>(&self, buffer: &mut Vec<T>, more: usize) -> bool {
        let free = buffer.capacity() - buffer.len();
        if free >= more {
            return true;
        }
        let want = more.max(buffer.capacity()).max(64);
        if !self.charge(want * std::mem::size_of::<T>()) {
            return false;
        }
        if buffer.try_reserve_exact(want).is_err() {
            self.out_of_memory();
            return false;
        }
        true
    }
}

/// Which stage a tessellation is in, so a block can say more than "working".
///
/// **Stages rather than a percentage**, because nothing here can predict one honestly: the
/// mesher's cost is decided by the surface as it descends, and a relaxation pass that finds
/// nothing to move ends early. Saying which stage is running is true; a bar would be a guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    Meshing = 0,
    Reducing = 1,
    Improving = 2,
    Done = 3,
    /// Working out what the run will hold, before anything is meshed.
    Estimating = 4,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Meshing => "meshing",
            Stage::Reducing => "reducing",
            Stage::Improving => "improving shape",
            Stage::Done => "done",
            Stage::Estimating => "estimating memory",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Stage::Meshing,
            1 => Stage::Reducing,
            2 => Stage::Improving,
            4 => Stage::Estimating,
            _ => Stage::Done,
        }
    }
}

/// A stage's working set, charged to its run's token until it is dropped.
pub(crate) struct Held<'a> {
    cancel: &'a Cancel,
    bytes: usize,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.cancel.release(self.bytes);
    }
}

/// Where a tessellation has got to, readable from another thread while it runs.
#[derive(Clone, Debug, Default)]
pub struct Progress(Arc<AtomicU8>);

impl Progress {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stage(&self) -> Stage {
        Stage::from_u8(self.0.load(Ordering::Relaxed))
    }

    fn set(&self, stage: Stage) {
        self.0.store(stage as u8, Ordering::Relaxed);
    }
}

/// A finished tessellation.
pub struct Tessellated {
    pub mesh: FieldMesh,
    /// Triangles as meshed and welded, before any reduction — so a report can say what
    /// reducing bought rather than only what came out.
    pub meshed: usize,
}

/// Meshes `field` inside `bounds`, then welds, reduces and reshapes as asked.
///
/// `bounds` is where the body *is* — its extent, as `graph_extent` gives it — and the mesher is
/// handed a box a little larger. See [`meshing_box`] for why that is done here and not by callers.
///
/// `None` when `cancel` was set before the end; a cancelled run has nothing worth keeping, since
/// a half-reduced or half-relaxed mesh is a mesh nobody asked for.
///
/// The order is the one the export has always used, with its reasons: weld unconditionally
/// (degenerate triangles where a wall thins to nothing have no consumer), count after the weld
/// (so "reduced from" compares like with like), reduce at the chord tolerance (the mesh is
/// already promised to be within it), and relax *after* reducing (a collapse moves a vertex the
/// relaxation would have to settle again) with the flip pass last.
pub fn tessellate(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tolerance: &MeshTolerance,
    improve: bool,
    reduce: bool,
    cancel: &Cancel,
    progress: &Progress,
) -> Option<Tessellated> {
    progress.set(Stage::Meshing);
    let mut mesh = crate::adaptive::mesh_field_adaptive_cancellable(
        field,
        meshing_box(bounds, tolerance),
        tolerance,
        cancel,
    )?;
    {
        // The weld's union-find and the triangles it rewrites into.
        let _held = cancel.hold(
            mesh.positions.len() * WELD_BYTES_PER_VERTEX + mesh.triangles.len() * WELD_BYTES_PER_TRIANGLE,
        )?;
        crate::simplify::weld(&mut mesh, tolerance.min_cell_mm * 1e-6);
    }
    let meshed = mesh.triangles.len();
    if cancel.is_cancelled() {
        return None;
    }
    if reduce {
        progress.set(Stage::Reducing);
        crate::simplify::simplify_cancellable(&mut mesh, field, tolerance.chord_mm, cancel)?;
    }
    if improve {
        progress.set(Stage::Improving);
        crate::simplify::relax_cancellable(&mut mesh, field, tolerance.chord_mm, cancel)?;
        crate::simplify::improve_shape_cancellable(&mut mesh, field, tolerance.chord_mm, cancel)?;
    }
    progress.set(Stage::Done);
    Some(Tessellated { mesh, meshed })
}

/// The weld's working set: a parent per vertex, and the triangles rewritten.
const WELD_BYTES_PER_VERTEX: usize = 4;
const WELD_BYTES_PER_TRIANGLE: usize = 12;

/// Why a run gave nothing back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stopped {
    /// It was told to stop.
    Cancelled,
    /// Its estimate was more than its limit, so nothing was meshed.
    WouldNotFit { estimate: Estimate, limit: usize },
    /// It outgrew its limit, or the allocator refused it, part way through. `held` is what it
    /// had charged when it stopped.
    OutOfMemory { held: usize, limit: Option<usize> },
}

/// [`tessellate`], estimated first and refused before anything is meshed when the estimate is
/// more than the token's memory limit — and stopped part way, rather than ending the process,
/// if it outgrows that limit anyway.
///
/// `known` is an estimate already made of exactly this run, which is used rather than made
/// again. Without a limit on the token nothing is refused, and the estimate is still made and
/// returned, since the caller is told what the run was expected to hold beside what it did.
#[allow(clippy::too_many_arguments)]
pub fn tessellate_within(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tolerance: &MeshTolerance,
    improve: bool,
    reduce: bool,
    known: Option<Estimate>,
    cancel: &Cancel,
    progress: &Progress,
) -> Result<(Tessellated, Estimate), Stopped> {
    let stopped = |cancel: &Cancel| {
        if cancel.ran_out_of_memory() {
            Stopped::OutOfMemory { held: cancel.used(), limit: cancel.memory_limit() }
        } else {
            Stopped::Cancelled
        }
    };
    progress.set(Stage::Estimating);
    let estimate = match known {
        Some(estimate) => estimate,
        None => estimate(field, bounds, tolerance, improve, reduce, cancel)
            .ok_or_else(|| stopped(cancel))?,
    };
    if let Some(limit) = cancel.memory_limit()
        && estimate.bytes > limit
    {
        return Err(Stopped::WouldNotFit { estimate, limit });
    }
    let done = tessellate(field, bounds, tolerance, improve, reduce, cancel, progress)
        .ok_or_else(|| stopped(cancel))?;
    Ok((done, estimate))
}

/// What a tessellation is expected to hold at its peak, worked out before it runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// The tree's surface leaves before balancing, as sampled.
    pub leaves: crate::adaptive::LeafEstimate,
    /// The most the run is expected to hold at once, in bytes — on the high side.
    pub bytes: usize,
}

/// Estimates what [`tessellate`] will hold at its peak, without meshing anything.
///
/// **Memory follows the tree's leaves, and the leaves are what has to be found.** Measured over
/// a ball, a cage and a gyroid block at every detail (`examples/tessellation_memory.rs`), a run
/// holds a steady few hundred bytes per surface leaf whatever the body — the tree, then the
/// mesh, then the working set of whichever pass after it is largest — while the leaves
/// themselves range over three orders of magnitude and are decided by the surface: a ball at the
/// finest detail has thirty-five thousand, a cube cage half a million. Nothing read off the
/// body's box, the tolerance or its feature size predicts that, because what refines a cage is
/// not its size but where its limit surface bends hardest. So the leaves are counted, by
/// walking the tree's own rule over a thinned copy of it (`estimate_surface_leaves`), and the
/// bytes follow from them.
///
/// It costs a few percent of the run: under 0.3 s against a gyroid block's 50 s to 200 s, and
/// 1 s against a cage's minute at the finest detail.
///
/// **Errs high, by measurement.** The leaf count lands within 8% of the exact count on every
/// measured body, so it is taken 15% high; each pass's bytes per leaf is the largest measured on
/// a mesh of a megabyte or more, rounded up. Across the table that reads 1.3× to 2.4× the peak
/// the run reached, and never under it. A body far from the measured three can still surprise
/// it, which is why a run is also held to its limit as it goes (`Cancel::with_memory_limit`)
/// rather than trusted to the estimate.
pub fn estimate(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tolerance: &MeshTolerance,
    improve: bool,
    reduce: bool,
    cancel: &Cancel,
) -> Option<Estimate> {
    let leaves = crate::adaptive::estimate_surface_leaves(
        field,
        meshing_box(bounds, tolerance),
        tolerance,
        cancel,
    )?;
    let per_leaf = if reduce {
        BYTES_PER_LEAF_REDUCED
    } else if improve {
        BYTES_PER_LEAF_IMPROVED
    } else {
        BYTES_PER_LEAF_MESHED
    };
    let bytes = (leaves.expected * SAMPLING_MARGIN * per_leaf).ceil() as usize + FIXED_BYTES;
    Some(Estimate { leaves, bytes })
}

/// How much more than the sampled count of leaves is allowed for: the worst measured reading
/// was 6% short, of a cage at the finest detail.
const SAMPLING_MARGIN: f64 = 1.15;
/// Peak bytes per leaf before balancing, the most measured on a mesh of a megabyte or more and
/// rounded up: 377 meshing only, 612 improving shape and 612 reducing as well — the flip pass's
/// table of edges is the largest working set after the mesh, and reducing's is no larger. The
/// balance step's extra leaves are in these numbers, since the leaves they divide by are counted
/// before it.
const BYTES_PER_LEAF_MESHED: f64 = 400.0;
const BYTES_PER_LEAF_IMPROVED: f64 = 650.0;
const BYTES_PER_LEAF_REDUCED: f64 = 650.0;
/// A floor, for a mesh so small that its bookkeeping outweighs its leaves — a ball at Coarse
/// holds 0.3 MB on 416 leaves.
const FIXED_BYTES: usize = 1 << 20;

/// Exactly how many surface leaves the tree has before balancing — the number [`estimate`]
/// samples for, counted by building the tree, so as dear as meshing.
pub fn surface_leaves(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tolerance: &MeshTolerance,
    cancel: &Cancel,
) -> Option<usize> {
    crate::adaptive::surface_leaf_count(field, meshing_box(bounds, tolerance), tolerance, cancel)
}

/// The box the mesher is given for a body whose extent is `bounds`: that extent, grown by one of
/// its finest cells on every side.
///
/// **A flat face lying on the mesher's boundary is not meshed at all.** Dual contouring makes a
/// quad for each edge the surface crosses, from the four cells around that edge — and an edge on
/// the outside of the box has fewer than four, so no quad. A body's extent is exactly where its
/// flat faces are, so a gyroid cut by a cube, meshed inside the cube's own box, came out as walls
/// ending in open frayed edges with every cut face missing: in the preview, and in the STL.
///
/// The exporter this replaced had the margin, in the application where `pick_bounds` sized its
/// box, and the Tessellate block read the extent directly and lost it. It lives here now because
/// every caller needs it and none should be able to forget it. One finest cell is enough: the
/// box's own faces are then outside the body by that much, so no edge on them is crossed.
fn meshing_box(
    (lo, hi): (Point3<f64>, Point3<f64>),
    tolerance: &MeshTolerance,
) -> (Point3<f64>, Point3<f64>) {
    let e = hi - lo;
    // A leaf is smaller than twice `min_cell_mm`, which is the cell the mesher stops at. The
    // floor is for a tolerance so fine it rounds away against the body's own size.
    let margin = (tolerance.min_cell_mm * 2.0).max(e.x.max(e.y).max(e.z) * 1e-4);
    let grow = nalgebra::Vector3::repeat(margin);
    (lo - grow, hi + grow)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ball;
    impl ScalarField for Ball {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.coords.norm() - 10.0
        }
    }

    fn bounds() -> (Point3<f64>, Point3<f64>) {
        (Point3::new(-12.0, -12.0, -12.0), Point3::new(12.0, 12.0, 12.0))
    }

    /// **A token set before the run starts gives nothing back, and quickly.** The weaker half of
    /// what cancelling promises — the stronger is below — but the one every stage has to honour
    /// before any of it is worth having.
    #[test]
    fn a_cancelled_tessellation_gives_nothing_back() {
        let cancel = Cancel::new();
        cancel.cancel();
        let started = std::time::Instant::now();
        let out = tessellate(
            &Ball,
            bounds(),
            &MeshTolerance { chord_mm: 0.001, angle_deg: 2.0, max_edge_mm: 0.0, min_cell_mm: 0.001 },
            true,
            true,
            &cancel,
            &Progress::new(),
        );
        assert!(out.is_none());
        assert!(
            started.elapsed().as_secs_f64() < 0.5,
            "a cancelled run should stop at once, took {:?}",
            started.elapsed()
        );
    }

    /// **Cancelling a run that is already under way stops it, not merely the next one.** The
    /// block cancels a stale run the moment its geometry changes, so a token that only worked
    /// before the start would leave the old run churning beside the new one. Set from another
    /// thread a little way in, at a tolerance fine enough that the whole run takes far longer.
    #[test]
    fn a_tessellation_cancelled_part_way_stops_part_way() {
        let tolerance =
            MeshTolerance { chord_mm: 0.0005, angle_deg: 1.0, max_edge_mm: 0.0, min_cell_mm: 0.0005 };
        let cancel = Cancel::new();
        let stopper = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            stopper.cancel();
        });
        let started = std::time::Instant::now();
        let out =
            tessellate(&Ball, bounds(), &tolerance, true, true, &cancel, &Progress::new());
        let took = started.elapsed().as_secs_f64();
        assert!(out.is_none(), "the run finished before it was told to stop — make it finer");
        assert!(took < 1.5, "told to stop at 0.15 s, stopped at {took:.2} s");
    }

    /// A 20 mm cube, as exact a distance as there is.
    struct Cube;
    impl ScalarField for Cube {
        fn distance(&self, p: Point3<f64>) -> f64 {
            let q = p.coords.map(|c| c.abs() - 10.0);
            q.map(|c| c.max(0.0)).norm() + q.max().min(0.0)
        }
    }

    /// **A body whose faces lie on the box it is meshed in still comes out closed, faces and
    /// all.** The Tessellate block hands over the body's extent, which for anything cut by a cube
    /// *is* the cube — and meshed there, every flat face went missing and the part was open on
    /// all six sides. Counted two ways, because either alone could pass on a wrong mesh: no edge
    /// with one triangle, and triangles actually lying on each of the six faces.
    #[test]
    fn a_body_whose_faces_lie_on_its_bounds_comes_out_closed() {
        let exact = (Point3::new(-10.0, -10.0, -10.0), Point3::new(10.0, 10.0, 10.0));
        let out = tessellate(
            &Cube,
            exact,
            &MeshTolerance { chord_mm: 0.05, angle_deg: 15.0, max_edge_mm: 0.0, min_cell_mm: 0.5 },
            false,
            false,
            &Cancel::new(),
            &Progress::new(),
        )
        .expect("not cancelled");
        let mesh = out.mesh;

        let mut edges = std::collections::HashMap::new();
        for t in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0u32) += 1;
            }
        }
        let open = edges.values().filter(|&&n| n == 1).count();
        assert_eq!(open, 0, "{open} edges have a triangle on one side only");

        for axis in 0..3 {
            for plane in [-10.0, 10.0] {
                let on_face = mesh
                    .triangles
                    .iter()
                    .filter(|t| {
                        t.iter().all(|&v| (mesh.positions[v as usize][axis] - plane).abs() < 1e-3)
                    })
                    .count();
                assert!(on_face > 0, "nothing on the face at {plane} along axis {axis}");
            }
        }
    }

    fn fine() -> MeshTolerance {
        MeshTolerance { chord_mm: 0.002, angle_deg: 4.0, max_edge_mm: 0.0, min_cell_mm: 0.002 }
    }

    /// **A run whose estimate is past its limit is refused before anything is meshed.** The
    /// stage it ends on says so: it never reached Meshing.
    #[test]
    fn a_run_estimated_past_its_limit_is_refused_and_meshes_nothing() {
        let progress = Progress::new();
        let cancel = Cancel::with_memory_limit(64 * 1024);
        let out = tessellate_within(&Ball, bounds(), &fine(), true, false, None, &cancel, &progress);
        match out {
            Err(Stopped::WouldNotFit { estimate, limit }) => {
                assert!(estimate.bytes > limit, "{} within {limit}", estimate.bytes);
                assert_eq!(limit, 64 * 1024);
            }
            Err(other) => panic!("refused for the wrong reason: {other:?}"),
            Ok(_) => panic!("a run past its limit was meshed"),
        }
        assert_eq!(progress.stage(), Stage::Estimating, "it went on past the estimate");
        assert_eq!(cancel.used(), 0, "a refused run charged memory it never held");
    }

    /// **A run that outgrows its limit stops and says so, rather than taking the process down.**
    /// Handed an estimate that reads nothing, so the refusal before the run cannot catch it, and
    /// a quarter of a megabyte to do a job of several: the allocator would have said yes, and
    /// what stops it is the limit the run is held to as its buffers grow. The control is the same
    /// run with room, which finishes.
    #[test]
    fn a_run_that_outgrows_its_limit_stops_and_reports_rather_than_aborting() {
        let wrong = Estimate {
            leaves: crate::adaptive::LeafEstimate { expected: 0.0, cells_decided: 0 },
            bytes: 0,
        };
        let limit = 256 * 1024;
        let cancel = Cancel::with_memory_limit(limit);
        let out = tessellate_within(
            &Ball, bounds(), &fine(), true, false, Some(wrong), &cancel, &Progress::new(),
        );
        match out {
            Err(Stopped::OutOfMemory { held, limit: Some(l) }) => {
                assert_eq!(l, limit);
                assert!(held > limit, "stopped at {held}, within its {limit}");
            }
            Err(other) => panic!("stopped for the wrong reason: {other:?}"),
            Ok(_) => panic!("a run four times its limit finished"),
        }
        assert!(cancel.ran_out_of_memory() && cancel.is_cancelled());

        let room = Cancel::with_memory_limit(1 << 30);
        let out = tessellate_within(
            &Ball, bounds(), &fine(), true, false, Some(wrong), &room, &Progress::new(),
        );
        assert!(out.is_ok(), "the same run with room did not finish: {:?}", out.err());
    }

    /// **Every pass after the mesher honours the limit too**, not only the mesher: each one,
    /// handed a finished mesh and a token with no room, stops as out of memory rather than
    /// allocating its working set.
    #[test]
    fn each_pass_after_the_mesher_stops_at_the_limit() {
        let tolerance = fine();
        let mesh = || crate::adaptive::mesh_field_adaptive(&Ball, bounds(), &tolerance);
        let full = || Cancel::with_memory_limit(1);

        let cancel = full();
        assert!(crate::simplify::simplify_cancellable(&mut mesh(), &Ball, 0.01, &cancel).is_none());
        assert!(cancel.ran_out_of_memory(), "reducing");
        let cancel = full();
        assert!(crate::simplify::relax_cancellable(&mut mesh(), &Ball, 0.01, &cancel).is_none());
        assert!(cancel.ran_out_of_memory(), "relaxing");
        let cancel = full();
        assert!(crate::simplify::improve_shape_cancellable(&mut mesh(), &Ball, 0.01, &cancel).is_none());
        assert!(cancel.ran_out_of_memory(), "flipping");
    }

    /// **Small enough never to be thinned, the walk is the tree**: its count is the count.
    #[test]
    fn the_estimate_counts_a_small_tree_exactly() {
        let tolerance =
            MeshTolerance { chord_mm: 0.05, angle_deg: 15.0, max_edge_mm: 0.0, min_cell_mm: 0.05 };
        let guess = estimate(&Ball, bounds(), &tolerance, false, false, &Cancel::new()).unwrap();
        let exact = surface_leaves(&Ball, bounds(), &tolerance, &Cancel::new()).unwrap();
        assert_eq!(guess.leaves.expected, exact as f64);
    }

    /// And thinned, it stays near the count — within the margin the bytes are taken with.
    #[test]
    fn the_estimate_of_a_thinned_tree_is_within_its_margin() {
        // Fine enough that the levels near the surface hold several times what the walk keeps.
        let tolerance =
            MeshTolerance { chord_mm: 0.0005, angle_deg: 2.0, max_edge_mm: 0.0, min_cell_mm: 0.0005 };
        let guess = estimate(&Ball, bounds(), &tolerance, false, false, &Cancel::new()).unwrap();
        let exact = surface_leaves(&Ball, bounds(), &tolerance, &Cancel::new()).unwrap() as f64;
        assert!(exact > 4.0 * crate::adaptive::ESTIMATE_LEVEL as f64, "too small to be thinned: {exact}");
        let ratio = guess.leaves.expected / exact;
        assert!(
            (1.0 / SAMPLING_MARGIN..=SAMPLING_MARGIN).contains(&ratio),
            "{} estimated for {exact}",
            guess.leaves.expected
        );
    }

    /// And an uncancelled run is the pipeline it replaced: a closed ball comes out, at Done.
    #[test]
    fn an_uncancelled_tessellation_finishes_and_says_so() {
        let progress = Progress::new();
        let out = tessellate(
            &Ball,
            bounds(),
            &MeshTolerance { chord_mm: 0.05, angle_deg: 15.0, max_edge_mm: 0.0, min_cell_mm: 0.05 },
            true,
            false,
            &Cancel::new(),
            &progress,
        )
        .expect("not cancelled, so it finishes");
        assert!(out.mesh.triangles.len() > 100);
        assert_eq!(progress.stage(), Stage::Done);
    }
}
