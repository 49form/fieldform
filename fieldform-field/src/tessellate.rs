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
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use nalgebra::Point3;

use crate::{FieldMesh, MeshTolerance, ScalarField};

/// Asks a running tessellation to stop.
///
/// **Cooperative, because a thread cannot be killed.** Every stage looks at this at points where
/// giving up is cheap — each octree cell as the mesher descends, each pass of the passes after
/// it — and returns without finishing. A clone is the same token, so the block that started a
/// run keeps one and the run takes the other.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Asks every holder of this token to stop. Idempotent, and cannot be taken back: a run that
    /// has been told to stop is over, and the next one gets a fresh token.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Relaxed, because nothing is published through it — a stage that sees it a cell late
    /// stops a cell late, which is the whole cost of being wrong.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
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
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Meshing => "meshing",
            Stage::Reducing => "reducing",
            Stage::Improving => "improving shape",
            Stage::Done => "done",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Stage::Meshing,
            1 => Stage::Reducing,
            2 => Stage::Improving,
            _ => Stage::Done,
        }
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
    crate::simplify::weld(&mut mesh, tolerance.min_cell_mm * 1e-6);
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
