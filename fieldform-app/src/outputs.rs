// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Tessellate and Export blocks, and the jobs that run them — backlog item 53.
//!
//! **Asked for as "the stl export needs to be its own block - this way i can run multiple
//! exports at once and also be able to stop/cancel them"**, and ruled against how other CAD
//! software does it, as the owner uses it: meshing and writing are two blocks, a running block stops from the block, several
//! run at once, and an active block re-runs by itself when what it reads changes.
//!
//! The blocks live in the document (`Document::tessellations`, `Document::exports`). What lives
//! here is everything about *running* them, none of which is saved: the jobs in flight, what each
//! last produced, and the fingerprint of what each tessellation reads.
//!
//! **One job per block, each on its own thread with its own cancel token.** The single
//! `pending_export` slot this replaces is why a second export used to be refused.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use fieldform_field::{Cancel, FieldMesh, GraphField, MeshTolerance, Progress, Tessellated};
use fieldform_scene::{
    Export, ExportId, Graph, MeshDetail, Node, NodeId, PartId, Tessellation, TessellationId,
};
use eframe::egui;
use nalgebra::Point3;

use fieldform_ops::{Args, Command};

use super::{
    App, GESTURE_WINDOW, LengthUnit, ObjReading, RowKey, file_name, row_label,
};

/// A block that is not a node of the graph — something a row can be selected as, renamed, and
/// deleted with the Delete key or its menu, like every other block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockRef {
    Tessellation(TessellationId),
    Export(ExportId),
    Import(fieldform_scene::id::SourceId),
    /// A profile row, by its first curve. Selecting it selects every curve of the loop, so
    /// Delete takes them all, as one entry.
    Profile(fieldform_scene::CurveId),
}

/// How long what a tessellation reads has to stay still before an active one re-runs.
///
/// **Waiting for the edit to settle, not restarting on every change.** A drag is dozens of
/// revisions a second and a Fine mesh is half a minute; restarting on each would mesh nothing to
/// completion and keep a core busy doing it. The same quiet the viewport's settled tier waits for
/// before it spends anything.
const SETTLE: Duration = Duration::from_millis(600);

/// Everything about running the output blocks, none of it saved.
#[derive(Default)]
pub(crate) struct Outputs {
    tessellating: HashMap<TessellationId, Tessellating>,
    tessellated: HashMap<TessellationId, Outcome>,
    /// What each tessellation reads, fingerprinted — and when that last changed.
    reads: HashMap<TessellationId, Reads>,
    /// The document revision `reads` was worked out at. Fingerprinting builds a graph per block,
    /// so it is redone when the document changes and not every frame.
    reads_at: Option<u64>,
    /// The fingerprint a tessellation was *stopped* at, by the person rather than by a change.
    /// An active block does not start again until what it reads moves on from here: Stop means
    /// stop, and a block that restarted itself 0.6 s later would make the button a lie.
    stopped_at: HashMap<TessellationId, u64>,
    writing: HashMap<ExportId, Writing>,
    written: HashMap<ExportId, Written>,
    /// Each finished tessellation's mesh gets the next of these, and an export records which it
    /// wrote. A number rather than the fingerprint, because re-running an unchanged tessellation
    /// is still a new mesh that an export has not written.
    next_generation: u64,
    /// The block being edited and when, with the blocks as they were when the edit began — so a
    /// drag on a tolerance goes on the history once, as the value it settled at, rather than
    /// sixty times a second. The rule `set_part_graph` follows for the graph, and like the
    /// graph's it ends on undo and redo.
    pub(crate) gesture: Option<(BlockRef, Instant, (Vec<Tessellation>, Vec<Export>))>,
    /// Tessellations whose mesh is not being shown — the eye on the row, off. View state, like a
    /// hidden body's eye, so not saved and not on the history.
    pub(crate) hidden: std::collections::HashSet<TessellationId>,
    /// What the preview is drawing this frame: each mesh's key and the bodies it stands in for.
    pub(crate) showing: Vec<u64>,
    /// Bodies of the part being edited that are drawn as their mesh rather than traced, and
    /// whether that is every body in it — a whole-part tessellation on show.
    pub(crate) instead_of: std::collections::HashSet<NodeId>,
    pub(crate) instead_of_all: bool,
}

struct Reads {
    key: Result<u64, String>,
    changed: Instant,
}

struct Tessellating {
    cancel: Cancel,
    progress: Progress,
    started: Instant,
    key: u64,
    tolerance: MeshTolerance,
    /// `None` when it was cancelled; a disconnect with nothing sent is a panic.
    result: Receiver<Option<Tessellated>>,
}

/// What a tessellation last finished with.
struct Outcome {
    key: u64,
    generation: u64,
    tolerance: MeshTolerance,
    took: f64,
    mesh: Result<(Arc<FieldMesh>, usize), String>,
}

struct Writing {
    cancel: Cancel,
    started: Instant,
    generation: u64,
    triangles: usize,
    result: Receiver<Result<(), fieldform_io::stl::StlError>>,
}

struct Written {
    generation: u64,
    took: f64,
    result: Result<(usize, usize), String>,
}

/// What a tessellation meshes, worked out from the document.
struct Input {
    graph: Graph,
    bounds: (Point3<f64>, Point3<f64>),
    tolerance: MeshTolerance,
    key: u64,
}

impl Outputs {
    fn busy(&self) -> bool {
        !self.tessellating.is_empty() || !self.writing.is_empty()
    }
}

impl App {
    /// Runs the output blocks for one frame: notices what changed, finishes what finished, and
    /// starts what is due. Called before anything draws, beside the import and export polls.
    pub(crate) fn drive_outputs(&mut self, ctx: &egui::Context) {
        let started = std::time::Instant::now();
        if self.doc.tessellations.is_empty()
            && self.doc.exports.is_empty()
            && !self.outputs.busy()
            && self.outputs.showing.is_empty()
            && !self.outputs.instead_of_all
            && self.outputs.instead_of.is_empty()
        {
            return;
        }
        self.forget_removed_outputs();
        self.refresh_reads();
        self.poll_tessellations();
        self.poll_writes();
        self.start_due_tessellations();
        self.start_due_writes();
        self.sync_previews();
        self.note_stage("the output blocks", started);

        // Kept repainting while anything runs or waits to, so the seconds on a row move and a
        // settled edit starts without the mouse having to move first.
        let waiting = self.doc.tessellations.iter().any(|t| {
            t.active && self.outputs.reads.get(&t.id).is_some_and(|r| r.key.is_ok())
                && !self.tessellation_is_current(t.id)
        });
        if self.outputs.busy() || waiting {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }

    // ---------------------------------------------------------------------------------------
    // The preview
    // ---------------------------------------------------------------------------------------

    /// Works out which meshes are on show, puts any new ones on the GPU, and tells the trace to
    /// redraw without the bodies they stand in for when that set changes.
    ///
    /// **Only a current mesh is shown.** While a tessellation re-runs, its body is traced as
    /// usual, so an edit is visible as it happens; the mesh returns when it has caught up.
    pub(crate) fn sync_previews(&mut self) {
        let started = std::time::Instant::now();
        let part = self.active_part;
        let mut showing: Vec<u64> = Vec::new();
        let mut instead_of = std::collections::HashSet::new();
        let mut instead_of_all = false;
        let mut uploads: Vec<(u64, u64, Arc<FieldMesh>)> = Vec::new();
        for t in &self.doc.tessellations {
            if t.part != part || self.outputs.hidden.contains(&t.id) {
                continue;
            }
            if !self.tessellation_is_current(t.id) {
                continue;
            }
            let Some(Outcome { mesh: Ok((mesh, _)), generation, .. }) =
                self.outputs.tessellated.get(&t.id)
            else {
                continue;
            };
            // **An empty mesh is not drawn, and the body stays.** A storage binding of zero
            // bytes is a validation error, and wgpu's answer to one is to end the process — which
            // is how a Tessellate block that found nothing took the application down with it
            // (2026-09-19). There is nothing to show in the body's place anyway.
            if mesh.triangles.is_empty()
                || mesh.triangles.len() > crate::preview::MAX_PREVIEW_TRIANGLES
            {
                continue;
            }
            let key = t.id.raw();
            showing.push(key);
            uploads.push((key, *generation, Arc::clone(mesh)));
            match t.body {
                Some(body) => {
                    instead_of.insert(body);
                }
                None => instead_of_all = true,
            }
        }
        if !uploads.is_empty() || !self.outputs.showing.is_empty() {
            let device = self.wgpu_render_state.device.clone();
            let format = self.wgpu_render_state.target_format;
            let mut renderer = self.wgpu_render_state.renderer.write();
            if renderer
                .callback_resources
                .get::<crate::preview::PreviewResources>()
                .is_none()
            {
                renderer
                    .callback_resources
                    .insert(crate::preview::PreviewResources::new(&device, format));
            }
            let resources = renderer
                .callback_resources
                .get_mut::<crate::preview::PreviewResources>()
                .expect("inserted above");
            for (key, generation, mesh) in &uploads {
                if !resources.has(*key, *generation) {
                    resources.upload(&device, *key, *generation, &mesh.positions, &mesh.triangles);
                }
            }
            resources.retain(&showing);
        }
        let changed = instead_of != self.outputs.instead_of
            || instead_of_all != self.outputs.instead_of_all;
        self.outputs.showing = showing;
        self.note_stage("uploading a mesh to draw", started);
        self.outputs.instead_of = instead_of;
        self.outputs.instead_of_all = instead_of_all;
        if changed {
            // What the trace draws has changed, and the trace only redraws when told.
            self.rebuild_render_resources();
        }
    }

    /// Whether a body of the part being edited is drawn as its mesh rather than traced.
    pub(crate) fn drawn_as_mesh(&self, body: NodeId) -> bool {
        self.outputs.instead_of_all || self.outputs.instead_of.contains(&body)
    }

    /// Whether the part being edited has a body anything could be made from, whatever is being
    /// shown — the question File → Export and Tessellate ask, which is not what is on screen.
    pub(crate) fn has_solid_body(&self) -> bool {
        let graph = self.part_graph();
        graph.roots().into_iter().any(|r| !self.is_reference_in(graph, r))
    }

    // ---------------------------------------------------------------------------------------
    // What a tessellation reads
    // ---------------------------------------------------------------------------------------

    /// The graph, bounds and tolerances a tessellation would mesh, and a fingerprint of them.
    ///
    /// **Every body except reference geometry, whatever the eyes say.** The export this replaces
    /// meshed what was on screen; a block that is saved with the document cannot depend on the
    /// viewport's visibility state, so the choice of body is the block's own, made explicitly.
    fn tessellation_input(&self, t: &Tessellation) -> Result<Input, String> {
        let part = self
            .doc
            .parts
            .get(&t.part)
            .ok_or_else(|| "its part has been deleted".to_string())?;
        let mut graph = part.graph.clone();
        let roots = graph.roots();
        match t.body {
            Some(body) => {
                if !roots.contains(&body) {
                    return Err(if graph.node(body).is_some() {
                        "its body has become part of another block; pick one again".to_string()
                    } else {
                        "its body has been deleted; pick one again".to_string()
                    });
                }
                for other in roots.into_iter().filter(|r| *r != body) {
                    graph.remove_body_and_its_own(other);
                }
            }
            None => {
                for root in roots {
                    if self.is_reference_in(&part.graph, root) {
                        graph.remove_body_and_its_own(root);
                    }
                }
            }
        }
        let roots = graph.roots();
        if roots.is_empty() {
            return Err("there is nothing to mesh".to_string());
        }
        let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for root in &roots {
            let Some((blo, bhi)) =
                fieldform_field::graph_extent(&self.doc, &graph, *root).as_box()
            else {
                return Err(
                    "it reaches to infinity — intersect it with a body so there is an \
                     outside to mesh"
                        .to_string(),
                );
            };
            lo = lo.inf(&blo);
            hi = hi.sup(&bhi);
        }
        let e = hi - lo;
        let tolerance = t.detail.tolerance(e.x.max(e.y).max(e.z));
        let key = self.fingerprint(&graph, &tolerance, t);
        Ok(Input { graph, bounds: (lo, hi), tolerance, key })
    }

    /// A number that changes whenever anything a tessellation reads does, and not otherwise.
    ///
    /// **Not the document revision**, which moves on a rename or a colour change as well, and
    /// would re-mesh half a minute of lattice because a block was renamed. So it hashes what the
    /// compile actually reads: the graph node by node, every parameter's value, every curve's
    /// points, every cage's and every surface's vertices, in a settled order. Imported meshes are
    /// named by ids that never change what they hold, so the graph covers them; cages and
    /// surfaces are edited in place, so their ids do not.
    fn fingerprint(&self, graph: &Graph, tolerance: &MeshTolerance, t: &Tessellation) -> u64 {
        let mut h = std::hash::DefaultHasher::new();
        let mut nodes: Vec<(NodeId, &Node)> = graph.nodes().map(|(id, n)| (id, n)).collect();
        nodes.sort_by_key(|(id, _)| id.raw());
        for (id, node) in nodes {
            id.raw().hash(&mut h);
            format!("{node:?}").hash(&mut h);
        }
        let mut params: Vec<_> = self.doc.parameters.keys().copied().collect();
        params.sort_by_key(|id| id.raw());
        for id in params {
            id.raw().hash(&mut h);
            self.doc.parameter_value(id).map(f64::to_bits).hash(&mut h);
        }
        let mut curves: Vec<_> = self.doc.curves.values().collect();
        curves.sort_by_key(|c| c.id.raw());
        for curve in curves {
            curve.id.raw().hash(&mut h);
            curve.degree().hash(&mut h);
            curve.is_closed().hash(&mut h);
            for p in curve.control_points() {
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()].hash(&mut h);
            }
        }
        let mut cages: Vec<_> = self.doc.cages.iter().collect();
        cages.sort_by_key(|(id, _)| id.raw());
        for (id, cage) in cages {
            id.raw().hash(&mut h);
            cage.geometry_fingerprint().hash(&mut h);
        }
        // A surface to follow is edited in place like a cage, so its id no longer says what it
        // holds — moving one of its points has to re-mesh a conformal lattice on it.
        let mut surfaces: Vec<_> = self.doc.quad_meshes.iter().collect();
        surfaces.sort_by_key(|(id, _)| id.raw());
        for (id, surface) in surfaces {
            id.raw().hash(&mut h);
            surface.geometry_fingerprint().hash(&mut h);
        }
        format!("{tolerance:?}").hash(&mut h);
        (t.improve, t.reduce).hash(&mut h);
        h.finish()
    }

    /// Re-fingerprints every tessellation when the document has changed, and cancels any run
    /// whose inputs moved under it.
    fn refresh_reads(&mut self) {
        if self.outputs.reads_at == Some(self.revision) {
            return;
        }
        self.outputs.reads_at = Some(self.revision);
        let blocks = self.doc.tessellations.clone();
        for t in &blocks {
            let key = self.tessellation_input(t).map(|input| input.key);
            let moved = self
                .outputs
                .reads
                .get(&t.id)
                .is_none_or(|r| r.key != key);
            if moved {
                self.outputs.reads.insert(t.id, Reads { key: key.clone(), changed: Instant::now() });
            }
            // **A run on inputs that have since moved is stopped at once**, not left to finish.
            // It would produce a mesh of geometry that no longer exists, and it holds a core the
            // run that replaces it needs.
            if let Some(job) = self.outputs.tessellating.get(&t.id)
                && key.as_ref().ok() != Some(&job.key)
            {
                job.cancel.cancel();
            }
        }
    }

    fn tessellation_is_current(&self, id: TessellationId) -> bool {
        let Some(Reads { key: Ok(key), .. }) = self.outputs.reads.get(&id) else {
            return false;
        };
        self.outputs.tessellated.get(&id).is_some_and(|o| o.key == *key)
    }

    // ---------------------------------------------------------------------------------------
    // Running tessellations
    // ---------------------------------------------------------------------------------------

    fn start_due_tessellations(&mut self) {
        let due: Vec<TessellationId> = self
            .doc
            .tessellations
            .iter()
            .filter(|t| {
                let reads = self.outputs.reads.get(&t.id);
                starts_by_itself(
                    t.active,
                    self.outputs.tessellating.contains_key(&t.id),
                    reads.map(|r| (r.key.as_ref().ok().copied(), r.changed.elapsed())),
                    self.outputs.tessellated.get(&t.id).map(|o| o.key),
                    self.outputs.stopped_at.get(&t.id).copied(),
                )
            })
            .map(|t| t.id)
            .collect();
        for id in due {
            self.start_tessellation(id);
        }
    }

    /// Starts one tessellation now, whatever its state — the Run button, and the settled path.
    fn start_tessellation(&mut self, id: TessellationId) {
        let started = std::time::Instant::now();
        let Some(t) = self.doc.tessellations.iter().find(|t| t.id == id).cloned() else {
            return;
        };
        if let Some(job) = self.outputs.tessellating.remove(&id) {
            job.cancel.cancel();
        }
        self.outputs.stopped_at.remove(&id);
        let input = match self.tessellation_input(&t) {
            Ok(input) => input,
            Err(why) => {
                self.warn(format!("Cannot tessellate {}: {why}", t.title()));
                return;
            }
        };
        // Compiled here and *moved* there. The compile reads the document, so it has to happen
        // on this thread; meshing does not touch the document at all, which is what makes
        // handing it over safe without a lock. A `GraphField` owns its leaves through `Arc`, so
        // what crosses is a handle rather than a copy of the geometry.
        let field = match GraphField::compile_graph(
            &self.doc,
            &input.graph,
            (input.tolerance.min_cell_mm * 0.5).max(1e-4),
        ) {
            Ok(field) => field,
            Err(e) => {
                let why = self.explain_compile(&e);
                self.outputs.tessellated.insert(
                    id,
                    Outcome {
                        key: input.key,
                        generation: 0,
                        tolerance: input.tolerance,
                        took: 0.0,
                        mesh: Err(why),
                    },
                );
                return;
            }
        };
        self.note_stage("compiling a Tessellate block's body", started);
        let cancel = Cancel::new();
        let progress = Progress::new();
        let (tx, rx) = std::sync::mpsc::channel();
        let (run_cancel, run_progress) = (cancel.clone(), progress.clone());
        let (bounds, tolerance) = (input.bounds, input.tolerance);
        let spawned = std::thread::Builder::new()
            .name(format!("tessellate {}", t.title()))
            .spawn(move || {
                let out = fieldform_field::tessellate(
                    &field,
                    bounds,
                    &tolerance,
                    t.improve,
                    t.reduce,
                    &run_cancel,
                    &run_progress,
                );
                // Nothing to do if the receiver is gone: the block was removed or the
                // application is closing, and either way nobody wants the mesh.
                let _ = tx.send(out);
            });
        match spawned {
            Ok(_) => {
                self.outputs.tessellating.insert(
                    id,
                    Tessellating {
                        cancel,
                        progress,
                        started: Instant::now(),
                        key: input.key,
                        tolerance: input.tolerance,
                        result: rx,
                    },
                );
            }
            Err(e) => self.warn(format!("could not start tessellating: {e}")),
        }
    }

    fn stop_tessellation(&mut self, id: TessellationId) {
        if let Some(job) = self.outputs.tessellating.remove(&id) {
            job.cancel.cancel();
            self.outputs.stopped_at.insert(id, job.key);
        }
    }

    fn poll_tessellations(&mut self) {
        let ids: Vec<TessellationId> = self.outputs.tessellating.keys().copied().collect();
        for id in ids {
            let job = &self.outputs.tessellating[&id];
            let finished = match job.result.try_recv() {
                Err(TryRecvError::Empty) => continue,
                Ok(Some(done)) => Some(Ok(done)),
                // Cancelled. Whatever cancelled it has already decided what happens next.
                Ok(None) => None,
                // The worker panicked. The application survives it, which is most of what a
                // separate process was wanted for; an out-of-memory would not be survived, and
                // is the one failure this arrangement still shares with the export it replaced.
                Err(TryRecvError::Disconnected) => Some(Err(
                    "stopped unexpectedly — the document is unchanged".to_string(),
                )),
            };
            let job = self.outputs.tessellating.remove(&id).expect("listed above");
            let Some(finished) = finished else { continue };
            let took = job.started.elapsed().as_secs_f64();
            let name = self.tessellation_name(id);
            let mesh = finished.map(|done| {
                self.say(format!(
                    "Tessellated {name}: {} triangles in {took:.1}s",
                    done.mesh.triangles.len()
                ));
                (Arc::new(done.mesh), done.meshed)
            });
            self.outputs.next_generation += 1;
            self.outputs.tessellated.insert(
                id,
                Outcome {
                    key: job.key,
                    generation: self.outputs.next_generation,
                    tolerance: job.tolerance,
                    took,
                    mesh,
                },
            );
        }
    }

    // ---------------------------------------------------------------------------------------
    // Writing files
    // ---------------------------------------------------------------------------------------

    /// The mesh an export would write now, if its tessellation has one that is current.
    ///
    /// **Only a finished, current mesh.** While its tessellation is running or its inputs have
    /// moved on, an export waits rather than writing the previous mesh under a name that now
    /// describes different geometry.
    fn mesh_for(&self, e: &Export) -> Option<(u64, Arc<FieldMesh>)> {
        if self.outputs.tessellating.contains_key(&e.source) || !self.tessellation_is_current(e.source)
        {
            return None;
        }
        let outcome = self.outputs.tessellated.get(&e.source)?;
        let (mesh, _) = outcome.mesh.as_ref().ok()?;
        Some((outcome.generation, Arc::clone(mesh)))
    }

    fn start_due_writes(&mut self) {
        let due: Vec<ExportId> = self
            .doc
            .exports
            .iter()
            .filter(|e| e.active && !self.outputs.writing.contains_key(&e.id))
            .filter(|e| {
                self.mesh_for(e).is_some_and(|(generation, _)| {
                    self.outputs.written.get(&e.id).is_none_or(|w| w.generation != generation)
                })
            })
            .map(|e| e.id)
            .collect();
        for id in due {
            self.start_write(id);
        }
    }

    fn start_write(&mut self, id: ExportId) {
        let Some(e) = self.doc.exports.iter().find(|e| e.id == id).cloned() else { return };
        let Some((generation, mesh)) = self.mesh_for(&e) else {
            self.warn(format!(
                "{} has no finished mesh to write yet — run {} first",
                e.title(),
                self.tessellation_name(e.source)
            ));
            return;
        };
        // An STL of nothing is a file that looks like an export and holds no part. Refused where
        // it would be written, so the block says why rather than leaving an empty file behind.
        if mesh.triangles.is_empty() {
            self.outputs.written.insert(
                id,
                Written {
                    generation,
                    took: 0.0,
                    result: Err(format!(
                        "{} is empty \u{2014} nothing to write",
                        self.tessellation_name(e.source)
                    )),
                },
            );
            return;
        }
        if let Some(job) = self.outputs.writing.remove(&id) {
            job.cancel.cancel();
        }
        let path = std::path::PathBuf::from(&e.path);
        // STL only, said plainly rather than written under the wrong extension: a file named
        // `.obj` that holds binary STL is worse than no file.
        let is_stl = path
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("stl"));
        if !is_stl {
            self.outputs.written.insert(
                id,
                Written {
                    generation,
                    took: 0.0,
                    result: Err("only STL is written so far — name the file .stl".to_string()),
                },
            );
            return;
        }
        let cancel = Cancel::new();
        let run_cancel = cancel.clone();
        let triangles = mesh.triangles.len();
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name(format!("export {}", e.title()))
            .spawn(move || {
                let out = fieldform_io::stl::write_stl_cancellable(
                    &path,
                    &mesh.positions,
                    &mesh.triangles,
                    &|| run_cancel.is_cancelled(),
                );
                let _ = tx.send(out);
            });
        match spawned {
            Ok(_) => {
                self.outputs.writing.insert(
                    id,
                    Writing { cancel, started: Instant::now(), generation, triangles, result: rx },
                );
            }
            Err(e) => self.warn(format!("could not start the export: {e}")),
        }
    }

    fn stop_write(&mut self, id: ExportId) {
        if let Some(job) = self.outputs.writing.remove(&id) {
            job.cancel.cancel();
        }
    }

    fn poll_writes(&mut self) {
        let ids: Vec<ExportId> = self.outputs.writing.keys().copied().collect();
        for id in ids {
            let job = &self.outputs.writing[&id];
            let outcome = match job.result.try_recv() {
                Err(TryRecvError::Empty) => continue,
                Ok(Err(fieldform_io::stl::StlError::Cancelled)) => None,
                Ok(Ok(())) => Some(Ok((
                    job.triangles,
                    fieldform_io::stl::stl_size_bytes(job.triangles),
                ))),
                Ok(Err(e)) => Some(Err(e.to_string())),
                Err(TryRecvError::Disconnected) => {
                    Some(Err("stopped unexpectedly; nothing was left half written".to_string()))
                }
            };
            let job = self.outputs.writing.remove(&id).expect("listed above");
            let Some(result) = outcome else { continue };
            let took = job.started.elapsed().as_secs_f64();
            if let (Ok((triangles, bytes)), Some(e)) =
                (&result, self.doc.exports.iter().find(|e| e.id == id))
            {
                self.say(format!(
                    "Exported {} \u{2014} {triangles} triangles, {:.1} MB, {took:.1}s",
                    file_name(std::path::Path::new(&e.path)),
                    *bytes as f64 / (1024.0 * 1024.0),
                ));
            }
            self.outputs
                .written
                .insert(id, Written { generation: job.generation, took, result });
        }
    }

    /// Drops the running state of blocks the document no longer has, cancelling their jobs.
    fn forget_removed_outputs(&mut self) {
        let tess: std::collections::HashSet<TessellationId> =
            self.doc.tessellations.iter().map(|t| t.id).collect();
        let exports: std::collections::HashSet<ExportId> =
            self.doc.exports.iter().map(|e| e.id).collect();
        self.outputs.tessellating.retain(|id, job| {
            let keep = tess.contains(id);
            if !keep {
                job.cancel.cancel();
            }
            keep
        });
        self.outputs.writing.retain(|id, job| {
            let keep = exports.contains(id);
            if !keep {
                job.cancel.cancel();
            }
            keep
        });
        self.outputs.tessellated.retain(|id, _| tess.contains(id));
        self.outputs.reads.retain(|id, _| tess.contains(id));
        self.outputs.stopped_at.retain(|id, _| tess.contains(id));
        self.outputs.written.retain(|id, _| exports.contains(id));
    }

    // ---------------------------------------------------------------------------------------
    // Making the blocks
    // ---------------------------------------------------------------------------------------

    /// File → Export, and the `export` console command: chooses a file, then adds an Export
    /// block — wired to the part's Tessellate block if it has one, or to a new one made with it.
    /// Both run as soon as the edit settles.
    ///
    /// **The menu builds blocks rather than exporting on the side.** Everything File → Export
    /// used to ask in its dialog is on the Tessellate block now, where it stays and can be
    /// changed and run again; the menu is a shortcut for making the pair. **A Tessellate block
    /// already there is used rather than joined by another**, asked for by the owner: the mesh
    /// being looked at is the mesh meant, and a second one beside it would mesh the same part
    /// twice. [`reusable_tessellation`] says which one.
    ///
    /// `export file="C:/out/part.stl" detail=full` does the same with no dialog, so an export
    /// can be typed or scripted. `detail` is one of the presets, and naming one is a request for
    /// that detail: an existing block at another detail is passed over for one at this detail,
    /// or a new one. Without it, a new block takes the last detail chosen, as the menu does.
    pub(crate) fn export_command(&mut self, args: &Args) {
        if !self.has_solid_body() {
            self.warn("Nothing to export");
            return;
        }
        let asked = match args.get("detail").and_then(|v| v.as_str()) {
            None => None,
            Some(word) => match detail_named(word) {
                Some(d) => Some(d),
                None => {
                    self.warn(format!(
                        "{word:?} is not a detail \u{2014} coarse, normal, full or enchilada"
                    ));
                    return;
                }
            },
        };
        let path = match args.get("file").and_then(|v| v.as_str()) {
            Some(file) => std::path::PathBuf::from(file),
            None => {
                let name = self
                    .file
                    .as_ref()
                    .and_then(|(p, _)| p.file_stem().map(|s| s.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "part".to_string());
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("STL", &["stl"])
                    .set_file_name(format!("{name}.stl"))
                    .save_file()
                else {
                    return;
                };
                path
            }
        };

        let part = self.active_part;
        // The selected body when there is one, the whole part otherwise — what a person means
        // by "export this" in each case.
        let body = self.selected_body.filter(|b| self.part_graph().roots().contains(b));
        let (improve, reduce) = (self.export_relax, self.export_simplify);
        let selected = match self.selected_block {
            Some(BlockRef::Tessellation(id)) => Some(id),
            _ => None,
        };
        let mut tessellations = self.doc.tessellations.clone();
        let mut exports = self.doc.exports.clone();
        let existing = reusable_tessellation(&tessellations, part, selected, body, asked);
        let detail = asked.unwrap_or(self.export_detail);
        self.export_detail = detail;
        let source = match existing {
            Some(id) => id,
            None => {
                let id = TessellationId::new();
                tessellations.push(Tessellation {
                    id,
                    name: None,
                    part,
                    body,
                    detail,
                    improve,
                    reduce,
                    active: true,
                });
                id
            }
        };
        let export = ExportId::new();
        exports.push(Export {
            id: export,
            name: None,
            source,
            path: path.to_string_lossy().to_string(),
            active: true,
        });
        // One entry on the history for the pair: undoing an export undoes the blocks it made.
        self.set_outputs(tessellations, exports, None);
        self.select_block(BlockRef::Export(export));
        match existing {
            Some(id) => {
                let detail = self
                    .doc
                    .tessellations
                    .iter()
                    .find(|t| t.id == id)
                    .map(|t| t.detail.label())
                    .unwrap_or_default();
                self.say(format!(
                    "Added an Export Mesh block writing {} from {} ({detail})",
                    file_name(&path),
                    self.tessellation_name(id)
                ));
            }
            None => self.say(format!(
                "Added blocks to write {}: they run once the edit settles, and again whenever the \
                 geometry changes while they are active",
                file_name(&path)
            )),
        }
    }

    /// Create → Tessellate, and the `tess` console command: a Tessellate block on its own.
    ///
    /// **Independent of any export**, asked for directly: the mesh is worth having to look at —
    /// to see whether a detail is fine enough — before there is a file to write it to. On the
    /// selected body, or the whole part when none is selected, at the detail last chosen unless
    /// `detail=` says otherwise. Active, so it runs as soon as it is made and follows the
    /// geometry after that.
    pub(crate) fn tessellate_command(&mut self, args: &Args) {
        if !self.has_solid_body() {
            self.warn("Nothing to tessellate");
            return;
        }
        let detail = match args.get("detail").and_then(|v| v.as_str()) {
            None => self.export_detail,
            Some(word) => match detail_named(word) {
                Some(d) => d,
                None => {
                    self.warn(format!(
                        "{word:?} is not a detail \u{2014} coarse, normal, full or enchilada"
                    ));
                    return;
                }
            },
        };
        let body = self.selected_body.filter(|b| self.part_graph().roots().contains(b));
        let id = TessellationId::new();
        let mut tessellations = self.doc.tessellations.clone();
        tessellations.push(Tessellation {
            id,
            name: None,
            part: self.active_part,
            body,
            detail,
            improve: self.export_relax,
            reduce: self.export_simplify,
            active: true,
        });
        self.set_outputs(tessellations, self.doc.exports.clone(), None);
        self.select_block(BlockRef::Tessellation(id));
        let what = match body {
            Some(b) => self.body_label(b),
            None => "the whole part".to_string(),
        };
        self.say(format!("Tessellating {what} at {}", detail.label()));
    }

    /// The `import` console command: `import` alone opens the dialog, as File → Import does;
    /// `import file="C:/in/panel.obj" as=surface unit=m` imports with no dialog at all.
    ///
    /// `as` is `cage`, `model` or `surface` and `unit` one of `mm`, `cm`, `m` or `in` — the two
    /// things an OBJ cannot say about itself. Both are remembered as the dialog's settings too,
    /// so the next import from the menu starts where the typed one left off.
    pub(crate) fn import_command(&mut self, args: &Args) {
        if let Some(word) = args.get("as").and_then(|v| v.as_str()) {
            self.obj_as = match word.to_ascii_lowercase().as_str() {
                "cage" => ObjReading::Cage,
                "model" | "mesh" => ObjReading::Model,
                "surface" => ObjReading::Surface,
                other => {
                    self.warn(format!("{other:?} is not a reading \u{2014} cage, model or surface"));
                    return;
                }
            };
        }
        if let Some(word) = args.get("unit").and_then(|v| v.as_str()) {
            self.obj_import_unit = match word.to_ascii_lowercase().as_str() {
                "mm" => LengthUnit::Millimetre,
                "cm" => LengthUnit::Centimetre,
                "m" => LengthUnit::Metre,
                "in" | "inch" => LengthUnit::Inch,
                other => {
                    self.warn(format!("{other:?} is not a unit \u{2014} mm, cm, m or in"));
                    return;
                }
            };
        }
        let Some(file) = args.get("file").and_then(|v| v.as_str()) else {
            self.import_file();
            return;
        };
        let path = std::path::PathBuf::from(file);
        if !path.is_file() {
            self.warn(format!("There is no file at {file}"));
            return;
        }
        match path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .as_deref()
        {
            Some("obj") => self.import_obj_from(&path),
            Some("stp" | "step") => self.import_step_from(&path),
            Some("igs" | "iges") => self.import_iges_from(&path),
            _ => self.warn(format!("{file} is not a format this build imports \u{2014} STEP, OBJ or IGES")),
        }
    }

    pub(crate) fn tessellation_name(&self, id: TessellationId) -> String {
        self.doc
            .tessellations
            .iter()
            .find(|t| t.id == id)
            .map(|t| t.title().to_string())
            .unwrap_or_else(|| "a tessellation that has been removed".to_string())
    }

    // ---------------------------------------------------------------------------------------
    // Changing the blocks, undoably
    // ---------------------------------------------------------------------------------------

    /// Replaces the output blocks, on the history.
    ///
    /// `gesture` names the block being edited when the edit can continue — a drag on a number,
    /// a run of clicks on one row — so it folds into one entry; `None` for an act that stands on
    /// its own, like creating or deleting.
    ///
    /// **It only ever replaces its own entry.** Folding works by replacing the top of the
    /// history, and if anything else had gone on in between, replacing the top would throw that
    /// edit's undo away. So it continues only when the top is still an output edit of this
    /// block's.
    pub(crate) fn set_outputs(
        &mut self,
        tessellations: Vec<Tessellation>,
        exports: Vec<Export>,
        gesture: Option<BlockRef>,
    ) {
        let now = (self.doc.tessellations.clone(), self.doc.exports.clone());
        if now.0 == tessellations && now.1 == exports {
            return;
        }
        let continues = gesture.is_some()
            && self.stack.peek_undo() == Some("outputs.set")
            && self
                .outputs
                .gesture
                .as_ref()
                .is_some_and(|(b, at, _)| Some(*b) == gesture && at.elapsed() < GESTURE_WINDOW);
        let before = match (&self.outputs.gesture, continues) {
            (Some((_, _, before)), true) => before.clone(),
            _ => now,
        };
        self.outputs.gesture =
            gesture.map(|b| (b, Instant::now(), before.clone()));
        // An output edit ends a graph gesture, for the reason a material edit does: the history
        // is one line, and the next graph edit must not fold into an entry that is not its own.
        self.gesture = None;
        let cmd: Box<dyn Command> =
            Box::new(fieldform_ops::SetOutputs::new(before, (tessellations, exports)));
        let done = if continues {
            self.stack.replace_top(&mut self.doc, cmd)
        } else {
            self.stack.execute(&mut self.doc, cmd)
        };
        match done {
            Ok(()) => self.mark_changed(),
            Err(e) => self.warn(e.to_string()),
        }
    }

    pub(crate) fn select_block(&mut self, block: BlockRef) {
        self.select_body(None);
        self.selected_curve = None;
        self.clear_point_selection();
        self.selection.clear();
        self.selected_block = Some(block);
    }

    /// Deletes a block that is not a node — undoably, as one entry.
    pub(crate) fn delete_block(&mut self, block: BlockRef) {
        match block {
            BlockRef::Tessellation(id) => {
                let name = self.tessellation_name(id);
                let mut tessellations = self.doc.tessellations.clone();
                tessellations.retain(|t| t.id != id);
                // Its exports stay, saying they have nothing to write: they are blocks of their
                // own with their own file, and taking them too would delete more than was
                // selected. Pointing them at another tessellation keeps them.
                self.set_outputs(tessellations, self.doc.exports.clone(), None);
                self.say(format!("Deleted {name}"));
            }
            BlockRef::Export(id) => {
                let name = self
                    .doc
                    .exports
                    .iter()
                    .find(|e| e.id == id)
                    .map(|e| e.title().to_string())
                    .unwrap_or_default();
                let mut exports = self.doc.exports.clone();
                exports.retain(|e| e.id != id);
                self.set_outputs(self.doc.tessellations.clone(), exports, None);
                self.say(format!("Deleted {name}"));
            }
            BlockRef::Import(source) => self.delete_import(source),
            BlockRef::Profile(key) => self.delete_profile(key),
        }
        if self.selected_block == Some(block) {
            self.selected_block = None;
        }
    }

    /// Deletes an import and everything it brought in: its bodies, surfaces and curves, and its
    /// record — one act, so one entry on the history.
    ///
    /// **Refused while anything outside it still reads what it made**, the rule `delete_curve`
    /// follows: a conformal lattice whose surface vanished, or a pipe whose curve did, stops
    /// evaluating, and deleting that as well would take a block nobody selected.
    fn delete_import(&mut self, source: fieldform_scene::id::SourceId) {
        let Some(record) = self.doc.sources.get(&source).cloned() else { return };
        let members = self.source_members(source);
        let member_set: std::collections::HashSet<NodeId> = members.iter().copied().collect();
        let mut users: Vec<String> = Vec::new();
        for (_, part) in &self.doc.parts {
            for (id, node) in part.graph.nodes() {
                if member_set.contains(&id) {
                    continue;
                }
                let reads_surface = matches!(
                    node,
                    Node::Lattice {
                        map: fieldform_scene::CellMap::Conformal { surface, .. },
                        ..
                    } if record.surfaces.contains(surface)
                );
                let reads_curve = node.curves().iter().any(|c| record.curves.contains(c));
                if reads_surface || reads_curve {
                    users.push(self.body_label(id));
                }
            }
        }
        if !users.is_empty() {
            users.sort();
            users.dedup();
            self.say(format!(
                "{} is used by {} \u{2014} delete that first",
                record.name,
                users.join(", ")
            ));
            return;
        }

        let part = self.active_part;
        let before = self.part_graph().clone();
        let mut after = before.clone();
        for id in &members {
            after.remove_body_and_its_own(*id);
        }
        let mut commands: Vec<Box<dyn Command>> = Vec::new();
        if !members.is_empty() {
            commands.push(Box::new(fieldform_ops::SetPartGraph::new(part, before, after)));
        }
        for id in &record.surfaces {
            if let Some(mesh) = self.doc.quad_meshes.get(id) {
                let name = self.doc.surface_names.get(id).cloned();
                commands.push(Box::new(fieldform_ops::SetSurface::new(
                    *id,
                    Some((mesh.clone(), name)),
                    None,
                )));
            }
        }
        for id in &record.curves {
            if let Some(curve) = self.doc.curve(*id).cloned() {
                commands.push(Box::new(fieldform_ops::SetCurve::remove(curve)));
            }
        }
        commands.push(Box::new(fieldform_ops::SetSource::new(source, Some(record.clone()), None)));
        let batch = Box::new(fieldform_ops::Batch::new("import.delete", commands));
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.after_document_change();
        self.rebuild_render_resources();
        self.say(format!("Deleted {} and everything it brought in", record.name));
    }

    /// A new name for an output block, on the history. Empty puts the block's own name back.
    pub(crate) fn rename_output(&mut self, block: BlockRef, text: &str) {
        let name = (!text.is_empty()).then(|| text.to_string());
        match block {
            BlockRef::Tessellation(id) => {
                let mut tessellations = self.doc.tessellations.clone();
                if let Some(t) = tessellations.iter_mut().find(|t| t.id == id) {
                    t.name = name;
                }
                self.set_outputs(tessellations, self.doc.exports.clone(), None);
            }
            BlockRef::Export(id) => {
                let mut exports = self.doc.exports.clone();
                if let Some(e) = exports.iter_mut().find(|e| e.id == id) {
                    e.name = name;
                }
                self.set_outputs(self.doc.tessellations.clone(), exports, None);
            }
            BlockRef::Profile(key) => self.rename_profile(key, text),
            BlockRef::Import(source) => {
                // An import is called what its file is called until renamed, and an empty name
                // leaves it as it was rather than calling it nothing.
                let Some(name) = name else { return };
                let Some(before) = self.doc.sources.get(&source).cloned() else { return };
                let mut after = before.clone();
                after.name = name;
                self.gesture = None;
                match self.stack.execute(
                    &mut self.doc,
                    Box::new(fieldform_ops::SetSource::new(source, Some(before), Some(after))),
                ) {
                    Ok(()) => self.mark_changed(),
                    Err(e) => self.warn(e.to_string()),
                }
            }
        }
    }

    /// The right-click menu of a block that is not a node.
    pub(crate) fn block_menu(&mut self, ui: &mut egui::Ui, block: BlockRef) {
        match block {
            BlockRef::Tessellation(id) => {
                if self.outputs.tessellating.contains_key(&id) {
                    if ui.button("Stop").clicked() {
                        self.stop_tessellation(id);
                        ui.close();
                    }
                } else if ui.button("Run").clicked() {
                    self.start_tessellation(id);
                    ui.close();
                }
            }
            BlockRef::Export(id) => {
                if self.outputs.writing.contains_key(&id) {
                    if ui.button("Stop").clicked() {
                        self.stop_write(id);
                        ui.close();
                    }
                } else if ui.button("Run").clicked() {
                    self.start_write(id);
                    ui.close();
                }
            }
            BlockRef::Import(_) => {}
            BlockRef::Profile(key) => {
                let members = self.profiles.members(key).map(|m| m.to_vec()).unwrap_or_default();
                let (held, total) = self.profile_hold(&members);
                if ui
                    .add_enabled(held < total, egui::Button::new("Join"))
                    .on_hover_text(
                        "Puts a point at every corner and sits both ends on it, so moving a \
                         corner moves both curves",
                    )
                    .on_disabled_hover_text("Every corner already has its point")
                    .clicked()
                {
                    self.join_row(key);
                    ui.close();
                }
                if ui
                    .add_enabled(held > 0, egui::Button::new("Ungroup"))
                    .on_hover_text(
                        "Takes the ends off their points, keeping every curve exactly where \
                         it is",
                    )
                    .on_disabled_hover_text("No joints to take out")
                    .clicked()
                {
                    self.ungroup_row(key);
                    ui.close();
                }
            }
        }
        if ui.button("Rename").clicked() {
            let (key, current) = match block {
                BlockRef::Tessellation(id) => (
                    RowKey::Tessellation(id),
                    self.doc
                        .tessellations
                        .iter()
                        .find(|t| t.id == id)
                        .and_then(|t| t.name.clone())
                        .unwrap_or_default(),
                ),
                BlockRef::Export(id) => (
                    RowKey::Export(id),
                    self.doc
                        .exports
                        .iter()
                        .find(|e| e.id == id)
                        .and_then(|e| e.name.clone())
                        .unwrap_or_default(),
                ),
                BlockRef::Import(source) => (
                    RowKey::Import(source),
                    self.doc.sources.get(&source).map(|s| s.name.clone()).unwrap_or_default(),
                ),
                BlockRef::Profile(key) => (
                    RowKey::Profile(key),
                    self.doc.profile_names.get(&key).cloned().unwrap_or_default(),
                ),
            };
            self.renaming = Some((key, current));
            ui.close();
        }
        ui.separator();
        if ui.button("Delete\tDel").clicked() {
            self.delete_block(block);
            ui.close();
        }
    }

    /// Whether a root of `graph` is reference geometry rather than a solid — the rule the
    /// viewport uses, asked of any part's graph rather than only the one being edited.
    fn is_reference_in(&self, graph: &Graph, id: NodeId) -> bool {
        let mut id = id;
        loop {
            match graph.node(id) {
                Some(Node::Plane { .. }) => return true,
                Some(Node::Lattice { .. }) => {
                    return fieldform_field::graph_extent(&self.doc, graph, id).is_unbounded();
                }
                Some(Node::Mesh(mesh)) => {
                    return self.doc.mesh(*mesh).is_none_or(|m| !m.is_closed());
                }
                Some(Node::Transform { input, .. }) => match input.source() {
                    Some(below) => id = below,
                    // A hole ends the walk: there is no leaf down there to find.
                    None => return false,
                },
                _ => return false,
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // The rows
    // ---------------------------------------------------------------------------------------

    /// The one line a row says about where its block is — shown in the header, so a collapsed
    /// block still says whether it is running.
    fn tessellation_status(&self, t: &Tessellation) -> String {
        if let Some(job) = self.outputs.tessellating.get(&t.id) {
            return format!(
                "{} {}s",
                job.progress.stage().label(),
                job.started.elapsed().as_secs()
            );
        }
        if let Some(Reads { key: Err(why), .. }) = self.outputs.reads.get(&t.id) {
            return format!("cannot run: {why}");
        }
        let current = self.tessellation_is_current(t.id);
        match self.outputs.tessellated.get(&t.id) {
            Some(Outcome { mesh: Err(why), .. }) if current => format!("failed: {why}"),
            // With the tolerance it was made at, because "how long did that take, and at what
            // setting" is the pair a person needs to choose the next one.
            Some(Outcome { mesh: Ok((mesh, meshed)), took, tolerance, .. }) if current => {
                let reduced = if *meshed > mesh.triangles.len() {
                    format!(" (from {meshed})")
                } else {
                    String::new()
                };
                // Said on the block, because otherwise the only sign is the smooth body still
                // on screen where the triangles were expected — which reads as the mesh having
                // failed, or as the preview being broken.
                let undrawn = if mesh.triangles.is_empty() {
                    " \u{2014} the mesher found no surface, so the body is shown instead"
                        .to_string()
                } else if mesh.triangles.len() > crate::preview::MAX_PREVIEW_TRIANGLES {
                    format!(
                        " \u{2014} too many to draw (the viewport takes {} million), so the body \
                         is shown instead",
                        crate::preview::MAX_PREVIEW_TRIANGLES / 1_000_000
                    )
                } else {
                    String::new()
                };
                format!(
                    "{} triangles{reduced}, {took:.1}s at {:.3} mm / {:.0}\u{00b0}{undrawn}",
                    mesh.triangles.len(),
                    tolerance.chord_mm,
                    tolerance.angle_deg,
                )
            }
            _ if self.outputs.stopped_at.contains_key(&t.id) => "stopped".to_string(),
            _ if t.active => "waiting for the edit to settle".to_string(),
            Some(_) => "out of date \u{2014} press Run".to_string(),
            None => "not run yet \u{2014} press Run".to_string(),
        }
    }

    fn export_status(&self, e: &Export) -> String {
        if let Some(job) = self.outputs.writing.get(&e.id) {
            return format!("writing {}s", job.started.elapsed().as_secs());
        }
        let fresh = self.mesh_for(e).map(|(g, _)| g);
        match self.outputs.written.get(&e.id) {
            Some(Written { result: Err(why), .. }) => format!("failed: {why}"),
            Some(Written { result: Ok((triangles, bytes)), generation, took })
                if Some(*generation) == fresh =>
            {
                format!(
                    "written: {triangles} triangles, {:.1} MB, {took:.1}s",
                    *bytes as f64 / (1024.0 * 1024.0)
                )
            }
            _ if fresh.is_none() => {
                format!("waiting for {}", self.tessellation_name(e.source))
            }
            _ if e.active => "about to write".to_string(),
            _ => "ready \u{2014} press Run".to_string(),
        }
    }

    pub(crate) fn tessellation_block(&mut self, ui: &mut egui::Ui, id: TessellationId) {
        let Some(before) = self.doc.tessellations.iter().find(|t| t.id == id).cloned() else {
            return;
        };
        let mut t = before.clone();
        let running = self.outputs.tessellating.contains_key(&id);
        let status = self.tessellation_status(&t);
        let selected = self.selected_block == Some(BlockRef::Tessellation(id));
        let mut run = false;
        let mut stop = false;

        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("tessellation", id.raw())),
            true,
        );
        state
            .show_header(ui, |ui| {
                // The mesh's eye, in the column every block's eye is in. Shown, the body is
                // drawn as its triangles instead of traced; hidden, it is traced as usual.
                let showing = !self.outputs.hidden.contains(&id);
                let icon = egui::RichText::new("\u{1f441}").color(if showing {
                    ui.visuals().strong_text_color()
                } else {
                    ui.visuals().weak_text_color()
                });
                if ui
                    .add_sized(
                        egui::vec2(18.0, ui.spacing().interact_size.y),
                        egui::Button::new(icon).frame(false),
                    )
                    .on_hover_text("Show the triangles in the viewport, in place of the body")
                    .clicked()
                {
                    if showing {
                        self.outputs.hidden.insert(id);
                    } else {
                        self.outputs.hidden.remove(&id);
                    }
                }
                ui.checkbox(&mut t.active, "").on_hover_text(
                    "Active: meshes again by itself once the geometry it reads has changed and \
                     the edit has settled.",
                );
                if running {
                    stop = ui.small_button("Stop").clicked();
                } else {
                    run = ui.small_button("Run").clicked();
                }
                // The name alone. What it produced goes underneath, where a number that changes
                // every second does not make the row's own name hard to read.
                let title = row_title(before.name.as_deref(), Tessellation::DEFAULT_NAME);
                let Some(row) = self.list_row(ui, RowKey::Tessellation(id), &title, selected)
                else {
                    return;
                };
                if row.clicked() || row.secondary_clicked() {
                    self.select_block(BlockRef::Tessellation(id));
                }
                row.context_menu(|ui| self.block_menu(ui, BlockRef::Tessellation(id)));
            })
            .body(|ui| {
                ui.label(egui::RichText::new(&status).small().weak());
                ui.separator();

                // Which body. The whole part by default, one of its bodies by choice.
                //
                // In the block list's own order, so the dropdown reads like the list above it
                // rather than reshuffling whenever an unrelated block is created.
                let bodies: Vec<NodeId> = self
                    .doc
                    .parts
                    .get(&t.part)
                    .map(|p| {
                        self.doc
                            .listed_bodies(&p.graph)
                            .into_iter()
                            .filter(|r| !self.is_reference_in(&p.graph, *r))
                            .collect()
                    })
                    .unwrap_or_default();
                let label = |app: &App, body: Option<NodeId>| match body {
                    Some(b) => app.body_label(b),
                    None => "Whole part".to_string(),
                };
                ui.horizontal(|ui| {
                    row_label(ui, "Body:");
                    egui::ComboBox::from_id_salt(("tessellation body", id.raw()))
                        .selected_text(label(self, t.body))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut t.body, None, "Whole part");
                            for b in &bodies {
                                ui.selectable_value(&mut t.body, Some(*b), label(self, Some(*b)));
                            }
                        });
                });

                let longest = self
                    .tessellation_input(&t)
                    .map(|i| {
                        let e = i.bounds.1 - i.bounds.0;
                        e.x.max(e.y).max(e.z)
                    })
                    .unwrap_or(100.0);
                mesh_detail_controls(ui, &mut t.detail, longest);

                ui.checkbox(&mut t.improve, "Improve triangle shape").on_hover_text(
                    "Slides vertices along the surface until the triangles are more even, leaving \
                     creases where they are. Same file, same worst deviation, roughly three times \
                     as long: on the reference document, slivers dropped from 6.3% to 1.6%.",
                );
                ui.checkbox(&mut t.reduce, "Reduce triangles").on_hover_text(
                    "Collapses edges the surface does not need, keeping every triangle within the \
                     chord tolerance — measured against the field itself. Took a third off the \
                     reference document with its worst deviation unchanged, and roughly ten times \
                     as long.",
                );
            });

        // Written back only if the block is still there: its own menu may have deleted it this
        // frame, and writing the edited copy back would bring it back from the dead.
        if t != before && self.doc.tessellations.iter().any(|x| x.id == id) {
            // Remembered as the setting the next File → Export starts from, so the menu keeps
            // offering what was last chosen rather than resetting to Normal.
            self.export_detail = t.detail;
            self.export_relax = t.improve;
            self.export_simplify = t.reduce;
            let tessellations = self
                .doc
                .tessellations
                .iter()
                .map(|x| if x.id == id { t.clone() } else { x.clone() })
                .collect();
            self.set_outputs(tessellations, self.doc.exports.clone(), Some(BlockRef::Tessellation(id)));
        }
        if stop {
            self.stop_tessellation(id);
        }
        if run {
            self.start_tessellation(id);
        }
    }

    pub(crate) fn export_block(&mut self, ui: &mut egui::Ui, id: ExportId) {
        let Some(before) = self.doc.exports.iter().find(|e| e.id == id).cloned() else { return };
        let mut e = before.clone();
        let running = self.outputs.writing.contains_key(&id);
        let status = self.export_status(&e);
        let selected = self.selected_block == Some(BlockRef::Export(id));
        let mut run = false;
        let mut stop = false;

        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("export", id.raw())),
            true,
        );
        state
            .show_header(ui, |ui| {
                ui.checkbox(&mut e.active, "").on_hover_text(
                    "Active: writes the file again each time its tessellation finishes a new mesh.",
                );
                if running {
                    stop = ui.small_button("Stop").clicked();
                } else {
                    run = ui.small_button("Run").clicked();
                }
                let title = row_title(before.name.as_deref(), Export::DEFAULT_NAME);
                let Some(row) = self.list_row(ui, RowKey::Export(id), &title, selected) else {
                    return;
                };
                if row.clicked() || row.secondary_clicked() {
                    self.select_block(BlockRef::Export(id));
                }
                row.context_menu(|ui| self.block_menu(ui, BlockRef::Export(id)));
            })
            .body(|ui| {
                ui.label(egui::RichText::new(&status).small().weak());
                ui.separator();
                ui.horizontal(|ui| {
                    row_label(ui, "Mesh:");
                    egui::ComboBox::from_id_salt(("export source", id.raw()))
                        .selected_text(self.tessellation_name(e.source))
                        .show_ui(ui, |ui| {
                            for t in &self.doc.tessellations {
                                ui.selectable_value(&mut e.source, t.id, t.title().to_string());
                            }
                        });
                });
                ui.horizontal(|ui| {
                    row_label(ui, "File:");
                    ui.label(file_name(std::path::Path::new(&e.path))).on_hover_text(&e.path);
                    if ui.small_button("Choose\u{2026}").clicked() {
                        let current = std::path::PathBuf::from(&e.path);
                        let mut dialog = rfd::FileDialog::new().add_filter("STL", &["stl"]);
                        if let Some(dir) = current.parent() {
                            dialog = dialog.set_directory(dir);
                        }
                        if let Some(name) = current.file_name() {
                            dialog = dialog.set_file_name(name.to_string_lossy());
                        }
                        if let Some(path) = dialog.save_file() {
                            e.path = path.to_string_lossy().to_string();
                        }
                    }
                });
            });

        if e != before && self.doc.exports.iter().any(|x| x.id == id) {
            // A new destination or source is a file that has not been written: forget what the
            // old one wrote, so an active export writes to the new place.
            if e.path != before.path || e.source != before.source {
                self.outputs.written.remove(&id);
            }
            let exports = self
                .doc
                .exports
                .iter()
                .map(|x| if x.id == id { e.clone() } else { x.clone() })
                .collect();
            self.set_outputs(self.doc.tessellations.clone(), exports, Some(BlockRef::Export(id)));
        }
        if stop {
            self.stop_write(id);
        }
        if run {
            self.start_write(id);
        }
    }

    /// The output blocks belonging to the part being edited, in the order they were made:
    /// its tessellations, then the exports that write them.
    pub(crate) fn output_blocks(&self) -> (Vec<TessellationId>, Vec<ExportId>) {
        let part: PartId = self.active_part;
        let tess: Vec<TessellationId> = self
            .doc
            .tessellations
            .iter()
            .filter(|t| t.part == part)
            .map(|t| t.id)
            .collect();
        // An export whose tessellation has gone is still listed, so it can be pointed somewhere
        // else or removed rather than lingering invisibly in the file.
        let exports = self
            .doc
            .exports
            .iter()
            .filter(|e| {
                self.doc
                    .tessellations
                    .iter()
                    .find(|t| t.id == e.source)
                    .is_none_or(|t| t.part == part)
            })
            .map(|e| e.id)
            .collect();
        (tess, exports)
    }
}

/// A detail by the word someone typed for it. "fine" is still understood, having been the
/// name until it matched the Display quality ladder's "Full".
fn detail_named(word: &str) -> Option<MeshDetail> {
    if word.eq_ignore_ascii_case("fine") {
        return Some(MeshDetail::Full);
    }
    MeshDetail::PRESETS.into_iter().find(|d| d.label().eq_ignore_ascii_case(word))
}

/// The Tessellate block an export should read from, when there is one worth using.
///
/// In order: the Tessellate block selected, since picking one and exporting says which; one of
/// the selected body, since that is what "export this" means with a body picked; and with nothing
/// picked, the part's whole-part block, then its newest. A block of *another* body is never used
/// for a selected one — that would write the wrong geometry to the file and say nothing.
///
/// `detail`, when the command named one, narrows every step to blocks at that detail: typing
/// `detail=full` beside a Coarse block asks for a Full mesh, and quietly writing Coarse would
/// ignore what was typed. `None` means make a new block.
fn reusable_tessellation(
    tessellations: &[Tessellation],
    part: PartId,
    selected: Option<TessellationId>,
    body: Option<NodeId>,
    detail: Option<MeshDetail>,
) -> Option<TessellationId> {
    let candidates: Vec<&Tessellation> = tessellations
        .iter()
        .filter(|t| t.part == part && detail.is_none_or(|d| t.detail == d))
        .collect();
    if let Some(id) = selected.filter(|id| candidates.iter().any(|t| t.id == *id)) {
        return Some(id);
    }
    let newest = |want: Option<NodeId>| candidates.iter().rev().find(|t| t.body == want).map(|t| t.id);
    match body {
        Some(_) => newest(body),
        None => newest(None).or_else(|| candidates.last().map(|t| t.id)),
    }
}

/// A row's title: the name given to it and what it is, or only what it is.
///
/// The same shape every body's row has — "Left bearing · Sphere" — so a renamed block still says
/// which kind of block it is, which is what the list is for reading.
fn row_title(name: Option<&str>, kind: &str) -> String {
    match name {
        Some(name) => format!("{name}  \u{00b7}  {kind}"),
        None => kind.to_string(),
    }
}

/// Whether an active tessellation should start by itself now.
///
/// **Five conditions, and each one is a promise the block makes.** It is active; it is not
/// already running; what it reads can be meshed at all (an error waits for the geometry to
/// change rather than failing again every frame); that has been still for [`SETTLE`]; and it is
/// not what the last mesh was made from. And one more that is easy to lose: if the person pressed
/// Stop, it does not start again until what it reads moves on — otherwise Stop would last 0.6 s.
///
/// `reads` is the current fingerprint (or `None` when it cannot be meshed) and how long it has
/// been that; `last` is the fingerprint the last finished mesh came from; `stopped` the one it
/// was stopped at.
fn starts_by_itself(
    active: bool,
    running: bool,
    reads: Option<(Option<u64>, Duration)>,
    last: Option<u64>,
    stopped: Option<u64>,
) -> bool {
    let Some((Some(key), still_for)) = reads else { return false };
    active && !running && still_for >= SETTLE && last != Some(key) && stopped != Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(part: PartId, body: Option<NodeId>, detail: MeshDetail) -> Tessellation {
        Tessellation {
            id: TessellationId::new(),
            name: None,
            part,
            body,
            detail,
            improve: false,
            reduce: false,
            active: true,
        }
    }

    /// **`export` beside a Tessellate block adds only the Export block**, asked for by the owner:
    /// with nothing picked it takes the part's block, whatever its detail — the mesh on screen
    /// is the mesh meant — and with nothing there it makes one.
    #[test]
    fn an_export_uses_the_tessellate_block_already_there() {
        let part = PartId::new();
        let there = block(part, None, MeshDetail::Coarse);
        let blocks = vec![there.clone()];
        assert_eq!(reusable_tessellation(&blocks, part, None, None, None), Some(there.id));
        assert_eq!(reusable_tessellation(&[], part, None, None, None), None);
        // Another part's block is not this part's mesh.
        assert_eq!(reusable_tessellation(&blocks, PartId::new(), None, None, None), None);
    }

    /// The block picked wins over the rest, and a picked body is only ever exported from a
    /// block of that body — never from another body's, which would write the wrong geometry.
    #[test]
    fn an_export_follows_what_is_picked() {
        let part = PartId::new();
        let (a, b) = (NodeId::new(), NodeId::new());
        let whole = block(part, None, MeshDetail::Normal);
        let of_a = block(part, Some(a), MeshDetail::Normal);
        let blocks = vec![whole.clone(), of_a.clone()];
        assert_eq!(reusable_tessellation(&blocks, part, Some(of_a.id), None, None), Some(of_a.id));
        assert_eq!(reusable_tessellation(&blocks, part, None, Some(a), None), Some(of_a.id));
        assert_eq!(reusable_tessellation(&blocks, part, None, Some(b), None), None);
        // Nothing picked: the whole part's block rather than a body's, though it is older.
        assert_eq!(reusable_tessellation(&blocks, part, None, None, None), Some(whole.id));
    }

    /// A detail typed is a detail asked for: a block at another one is passed over.
    #[test]
    fn an_export_that_names_a_detail_gets_that_detail() {
        let part = PartId::new();
        let coarse = block(part, None, MeshDetail::Coarse);
        let full = block(part, None, MeshDetail::Full);
        let blocks = vec![coarse.clone(), full.clone()];
        let only_coarse = vec![coarse.clone()];
        assert_eq!(
            reusable_tessellation(&only_coarse, part, None, None, Some(MeshDetail::Full)),
            None
        );
        assert_eq!(
            reusable_tessellation(&blocks, part, Some(coarse.id), None, Some(MeshDetail::Full)),
            Some(full.id)
        );
    }

    const STILL: Duration = Duration::from_secs(1);
    const MOVING: Duration = Duration::from_millis(100);

    #[test]
    fn an_active_block_starts_once_its_edit_has_settled_and_not_before() {
        assert!(starts_by_itself(true, false, Some((Some(7), STILL)), None, None));
        assert!(!starts_by_itself(true, false, Some((Some(7), MOVING)), None, None));
    }

    #[test]
    fn an_inactive_or_running_block_does_not_start_by_itself() {
        assert!(!starts_by_itself(false, false, Some((Some(7), STILL)), None, None));
        assert!(!starts_by_itself(true, true, Some((Some(7), STILL)), None, None));
    }

    /// A mesh already made from these inputs is not made again — which is what lets an idle
    /// document with a Fine tessellation in it cost nothing.
    #[test]
    fn a_block_whose_mesh_is_current_does_not_start_again() {
        assert!(!starts_by_itself(true, false, Some((Some(7), STILL)), Some(7), None));
        assert!(starts_by_itself(true, false, Some((Some(8), STILL)), Some(7), None));
    }

    /// **Stop means stop**, until the geometry moves. Without this an active block would restart
    /// itself the moment the edit counted as settled, and the button would do nothing visible.
    #[test]
    fn a_stopped_block_stays_stopped_until_what_it_reads_changes() {
        assert!(!starts_by_itself(true, false, Some((Some(7), STILL)), None, Some(7)));
        assert!(starts_by_itself(true, false, Some((Some(8), STILL)), None, Some(7)));
    }

    /// Inputs that cannot be meshed wait for the geometry to change, rather than trying and
    /// failing again every frame.
    #[test]
    fn a_block_that_cannot_be_meshed_does_not_keep_trying() {
        assert!(!starts_by_itself(true, false, Some((None, STILL)), None, None));
        assert!(!starts_by_itself(true, false, None, None, None));
    }
}

/// The detail a tessellation is meshed at: four presets and the numbers behind them.
///
/// **One editor for "how finely to mesh"** (D9), the one the export dialog used to carry. Picking
/// a preset stores the preset, so it keeps meaning the same thing if the body changes size;
/// touching any of the numbers makes the detail `Custom` with exactly those numbers.
fn mesh_detail_controls(ui: &mut egui::Ui, detail: &mut MeshDetail, longest: f64) {
    ui.horizontal(|ui| {
        row_label(ui, "Detail:");
        for preset in MeshDetail::PRESETS {
            let chosen = *detail == preset;
            if ui
                .selectable_label(chosen, preset.label())
                .on_hover_text(match preset {
                    MeshDetail::Coarse => {
                        "A tenth the triangles of Normal. For checking a shape, or for a part \
                         with nothing small in it."
                    }
                    MeshDetail::Normal => {
                        "The default. On a 185 mm part with a gyroid in it: about 460 000 \
                         triangles, 22 MB, four seconds."
                    }
                    MeshDetail::Full => {
                        "Five times the triangles of Normal, and about seven times the wait. \
                         Worth it when a lattice wall or a fillet is small next to the part."
                    }
                    _ => {
                        "As fine as the presets go, and slow. Check that Full is not already \
                         enough before reaching for it."
                    }
                })
                .clicked()
            {
                *detail = preset;
            }
        }
        if matches!(detail, MeshDetail::Custom(_)) {
            ui.label(egui::RichText::new("Custom").weak());
        }
    });
    // The numbers the detail means for this body, still editable. Touching any of them makes it
    // Custom rather than leaving a preset's name on numbers it no longer describes.
    let mut tol = detail.tolerance(longest);
    let mut edited = false;
    ui.horizontal(|ui| {
        ui.label("Chord:");
        edited |= ui
            .add(egui::DragValue::new(&mut tol.chord_mm).speed(0.005).range(0.001..=5.0).suffix(" mm"))
            .on_hover_text(
                "How far the mesh may sit from the true surface — the sag between a chord and \
                 the arc it cuts. Compare it against the layer height you will print at.",
            )
            .changed();
        ui.label("Angle:");
        edited |= ui
            .add(egui::DragValue::new(&mut tol.angle_deg).speed(0.5).range(1.0..=60.0).suffix("\u{00b0}"))
            .on_hover_text(
                "How far the surface may turn within one cell. This is the control that \
                 sharpens hard edges: a crease has no sag, so the chord cannot see one.",
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Max edge:");
        // **Zero is "no cap", and it says so.** Showing it as "0.0 mm" reads as edges of no
        // length, which was the question the first person to see it asked. No preset caps it:
        // a flat face meshing as a few large triangles is the whole saving for a slicer.
        edited |= ui
            .add(
                egui::DragValue::new(&mut tol.max_edge_mm)
                    .speed(0.5)
                    .range(0.0..=1000.0)
                    .custom_formatter(|v, _| {
                        if v <= 0.0 { "no cap".to_string() } else { format!("{v:.1} mm") }
                    })
                    .custom_parser(|text| {
                        let t = text.trim().trim_end_matches("mm").trim();
                        if t.eq_ignore_ascii_case("no cap") || t.eq_ignore_ascii_case("none") {
                            Some(0.0)
                        } else {
                            t.parse().ok()
                        }
                    }),
            )
            .on_hover_text(
                "The longest a triangle may be however flat the surface is. Zero means no cap, \
                 which is right for a slicer; set one for analysis, where long thin triangles \
                 integrate badly.",
            )
            .changed();
        ui.label("Min:");
        edited |= ui
            .add(egui::DragValue::new(&mut tol.min_cell_mm).speed(0.01).range(0.001..=10.0).suffix(" mm"))
            .on_hover_text(
                "The smallest cell the mesher may reach — the floor on the whole process, and \
                 what bounds its cost. Set it to the smallest feature that matters.",
            )
            .changed();
    });
    if edited {
        *detail = MeshDetail::Custom(tol);
    }
}
