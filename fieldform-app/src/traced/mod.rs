// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The viewport drawn by evaluating the field, rather than by meshing it.
//!
//! Nothing is uploaded and nothing is read back: the composed graph is compiled to a shader,
//! every pixel asks the field directly where the surface is, a second pass lights what the first
//! found, and egui samples the result where it was written. What that buys is in
//! the commercial-readiness design note — the short version is that meshing cost scales
//! with the volume sampled and is paid on every change, while this scales with pixels and is
//! paid per frame whether anything changed or not.
//!
//! # Two tiers, and what each spends its budget on
//!
//! `shading-spec.md` §9: while anything is moving — the camera, a slider, a cage — the frame is
//! traced at a reduced size with no occlusion or shadows, so it keeps up. Once the input has
//! been quiet for [`SETTLE_AFTER`], the frame is traced at the viewport's own size with
//! occlusion and shadows on, and then traced again a handful of times with the rays nudged by a
//! fraction of a pixel, each frame averaged into the last. That average is the anti-aliased
//! image: a traced silhouette has no triangle edges for multisampling to find, so the only way to
//! soften one is to sample it more than once.
//!
//! **A still converges and then stops.** Once [`STILL_FRAMES`] have been averaged nothing is
//! traced until something changes — not a tick, not a frame. Idle means idle, or a laptop's
//! fans say otherwise an hour into a session (§9 again).
//!
//! **A still goes up in bands.** Each jittered frame is several submissions of a band of rows,
//! sized so that none costs more than [`BAND_BUDGET_MS`] by the last measurement
//! (`performance-stability-loop-spec.md` S2): a card that cannot preempt a compute dispatch
//! is reset by Windows when one packet runs past about two seconds, and a heavy still on a
//! Quadro M4000 did exactly that. Bands also make input between them cancel the rest, and the
//! draft's picture stays up until the still's first whole frame is in.
//!
//! # What is rebuilt, and when
//!
//! The renderer is rebuilt only when [`invalidate`](TracedView::invalidate) is called, which
//! `rebuild_render_resources` does after a cage edit or a change to the composed graph.
//! Parameter values are rewritten every frame because doing so costs a handful of bytes — that
//! is the whole point of compiling them as buffer reads rather than constants. The lighting pass
//! and its environment are built once and outlive every renderer, since neither depends on the
//! document.

use eframe::egui;
use eframe::egui_wgpu;
use fieldform_field::{CageField, MeshField};
use fieldform_gpu::{
    halton_jitter, BakedLeaf, Camera, Draft, EnvironmentChoice, EnvironmentMaps, Look,
    Occlusion, Renderer, ShaderProgram, ShadingResources, SubmissionCost, SubmissionTicket,
};
use fieldform_scene::{CurveId, Graph, Material, ParamId};
use nalgebra::{Point3, Vector3};
use std::time::{Duration, Instant};

mod draw;
mod rebuild;

/// A band's shape, with the resolution taken out: two bakes of one shape that differ only in
/// how finely they sample it describe the same geometry in the same place.
fn shape_of(key: &BakeKey) -> BakeKey {
    BakeKey { resolution: 0, ..*key }
}

/// Whether a shader the driver has just finished is still the one wanted.
///
/// **The rule that keeps a hidden body hidden.** A compile takes seconds, and the graph can
/// change while it runs — most sharply when the change is "there is nothing to draw at all",
/// which leaves no program to install it beside. The rebuild that asked for it stamped it; if
/// another rebuild has happened since, what came back is a picture of a shape that has gone.
fn compile_is_current(serial: u64, asked: u64) -> bool {
    serial == asked
}

/// Whether the compile already running is the one this rebuild wants, so it can be adopted
/// rather than replaced.
///
/// **The rule that lets a slow shader land at all.** Every rebuild takes a new serial and
/// [`compile_is_current`] throws away anything older, so a shader that takes longer to compile
/// than the gap between two rebuilds can never arrive. On the M4000 under DX12 the patterned
/// pipe's shader takes five to nine seconds; orbiting a freshly opened document asked for a
/// rebuild about twice a second; and in 340 seconds the march never ran once, because every
/// compile was abandoned mid-flight for an identical one. The perf log held its two header
/// lines and nothing else.
///
/// A rebuild that wants a source already being compiled therefore keeps that worker and makes
/// its serial the current one. Nothing is installed early: the source being equal is exactly
/// the condition under which the pipelines are interchangeable, which is the same test
/// `Renderer::rebuilt_from` makes to keep a pipeline.
fn compile_already_wanted(in_flight: &str, wanted: &str) -> bool {
    in_flight == wanted
}

/// Whether a settled bake in flight is still worth finishing.
///
/// **The rule that decides whether seconds of work are thrown away.** A rebuild knows every leaf
/// it asked for; a band whose shape is not among them is of geometry that has since moved, and
/// only that is worth stopping. Stopping on anything else — a section dragged, a body placed, a
/// parameter turned — is how a scene touched every few seconds never left its 32-cell stand-in.
fn bake_is_obsolete(
    pending: &[BakeKey],
    identities: &std::collections::HashSet<BakeKey>,
) -> bool {
    pending.iter().any(|key| !identities.contains(&shape_of(key)))
}

/// Which device made the GPU resources a view is holding, as far as a frame can tell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MadeOn {
    /// None are held: the first frame of a session, or the frame after a loss.
    Nothing,
    /// The device this frame is drawing with.
    ThisDevice,
    /// A device that is not this one, so everything held belongs to that one.
    AnotherDevice,
}

/// What a frame must do before it asks a device for anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MustDo {
    /// Draw: what is held was made on the device in hand, or nothing is held and the ordinary
    /// rebuild is about to make it.
    CarryOn,
    /// Let go of everything held, because the device that made it has gone — lost, or replaced
    /// by another.
    LetGo,
    /// Ask the device nothing at all: it is the one that was lost, and nothing is held to draw
    /// with in any case.
    AskNothing,
}

/// Decides it from the two facts a frame has: whether the device in hand has been reported
/// lost, and which device made what the view is holding.
///
/// **A rule of its own, because a real device loss cannot be provoked to test this against.**
/// Windows resets a display driver that runs past its timeout and a suspend invalidates a
/// device outright (backlog 35), both of which happen to somebody once and to this machine
/// never, so the only thing that can be held to a property here is the decision itself.
fn what_a_frame_must_do(lost: bool, made_on: MadeOn) -> MustDo {
    match (lost, made_on) {
        // Nothing was made on it and nothing can be: a submission to a lost device fails
        // rather than waits, and a pipeline built on one is refused.
        (true, MadeOn::Nothing) => MustDo::AskNothing,
        // Everything held is of a device that has gone. The same situation arrived at from
        // either side -- a loss reported, or a frame drawing with another device -- and the
        // same answer, since what makes a resource unusable is the device and not the reason.
        (true, _) | (false, MadeOn::AnotherDevice) => MustDo::LetGo,
        (false, _) => MustDo::CarryOn,
    }
}

/// What the viewport says once the device is lost, in the place every other reason for an empty
/// viewport is said (`TracedView::error`).
const LOST_DEVICE_REASON: &str = "The GPU was lost, so nothing more can be drawn — start the \
                                  application again. The line above says where the document is.";

/// Each curve's spine as the view last tessellated it, kept against everything the tessellation
/// read: the curve itself, the angle and the floor.
///
/// **Keyed by the curve's whole value, so a kept spine is the fresh one rather than probably
/// it.** `Curve::tessellate_to_angle` is a function of those three and nothing else, and a curve
/// is a handful of control points, weights and knots, so comparing one is comparing a handful of
/// numbers where walking it is measuring its length and then tessellating it as many as a dozen
/// times. A rebuild for a number, a settle and a band arriving move no curve, and each used to
/// walk every one.
///
/// The floor is the scene's extent times a constant, so a drag that grows the part moves it and
/// every spine is walked again. That is the key being exact, not a miss to be tuned away: the
/// floor decides where the halving stops, and a spine walked to another floor can be another
/// spine.
#[derive(Default)]
struct KeptSpines {
    kept: std::collections::HashMap<CurveId, KeptSpine>,
}

/// Each curve's tessellated spine by id, as [`Scene::curves`] holds them.
type Spines = Vec<(CurveId, Vec<Point3<f64>>)>;

/// One curve's spine and what it was tessellated from. The two numbers are kept as bits, so a
/// floor that moved in its last place is a different floor, as it is to the tessellation.
struct KeptSpine {
    curve: fieldform_scene::Curve,
    degrees: u64,
    floor_mm: u64,
    points: Vec<Point3<f64>>,
}

impl KeptSpines {
    /// The spine of each of `ids` that `curves` holds, in `ids`' order, tessellated to turn at
    /// most `degrees` a joint with `floor_mm` as the finest tolerance -- and how many of them had
    /// to be tessellated now.
    ///
    /// Only what was asked for this time is kept, so a curve nothing reads any more is let go
    /// rather than held for the life of the view.
    fn spines(
        &mut self,
        curves: &std::collections::HashMap<CurveId, fieldform_scene::Curve>,
        ids: impl IntoIterator<Item = CurveId>,
        degrees: f64,
        floor_mm: f64,
    ) -> (Spines, usize) {
        let mut before = std::mem::take(&mut self.kept);
        let mut spines = Vec::new();
        let mut made = 0;
        for id in ids {
            let Some(curve) = curves.get(&id) else { continue };
            let spine = match before.remove(&id) {
                Some(kept)
                    if kept.curve == *curve
                        && kept.degrees == degrees.to_bits()
                        && kept.floor_mm == floor_mm.to_bits() =>
                {
                    kept
                }
                _ => {
                    made += 1;
                    KeptSpine {
                        curve: curve.clone(),
                        degrees: degrees.to_bits(),
                        floor_mm: floor_mm.to_bits(),
                        points: curve.tessellate_to_angle(degrees, floor_mm),
                    }
                }
            };
            spines.push((id, spine.points.clone()));
            self.kept.insert(id, spine);
        }
        (spines, made)
    }
}

/// A generated program and what it was generated from besides the document, which
/// [`TracedView::invalidate`] answers for. See [`TracedView::program_for`].
struct KeptProgram {
    graph: Graph,
    morph_divisors: Vec<(fieldform_scene::NodeId, f64)>,
    program: ShaderProgram,
}

/// The three numbers the march reads that belong to a graph rather than to a camera.
///
/// Held together because they are applied together: with the renderer they describe, and never
/// before it.
#[derive(Clone, Copy)]
struct Marching {
    scene_bounded: bool,
    step_budget: Option<u32>,
    value_scale: Option<f64>,
}

/// A shader being compiled elsewhere, and what it is for.
struct Compiling {
    /// Which rebuild asked for it. Installed only if that is still the newest.
    serial: u64,
    /// The numbers to apply when it lands.
    marching: Marching,
    /// The finished renderer, or the driver's refusal.
    done: std::sync::mpsc::Receiver<Result<Renderer, fieldform_gpu::RenderError>>,
    /// What it is being built for. Installed with the renderer, because a program and the
    /// pipelines built from it are one thing: the parameter slots are read by position.
    program: Option<fieldform_gpu::ShaderProgram>,
    /// The source, so a later rebuild can tell whether this compile is still the one wanted.
    source: String,
    started: Instant,
    /// Whether the wait has been said out loud yet.
    announced: bool,
    /// The uploads a later rebuild of the same source brought while this compile was in
    /// flight -- its bakes, curves and surfaces -- for the finished shader's buffers to be
    /// built from, since the worker built them from the uploads it set off with.
    ///
    /// A slot's value and a curve's polyline are refreshed only by a rebuild, and adopting the
    /// worker used to drop the rebuild that brought them: a loft's G0/G1 toggle during its
    /// first compile drew G0 under a panel saying G1 until the next edit, and a section
    /// released mid-compile drew where it had been. A sweep's `closed` slot and spine had the
    /// same exposure for longer, unseen. Found in review.
    uploads: Option<(
        Vec<std::sync::Arc<fieldform_gpu::BakedField>>,
        Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)>,
        Vec<fieldform_field::PackedSurface>,
    )>,
}

/// How long a shader compile has to run before the wait is worth a word in the console.
///
/// Under this nobody notices; over it, the viewport is showing the *previous* shape and the
/// reason has to be somewhere. One line per compile, not per frame.
const SAY_COMPILE_AFTER: Duration = Duration::from_millis(1200);

/// Settled bands being computed elsewhere, and what they are for.
struct Baking {
    /// Set when the geometry this bake was for has been edited again. The worker reads it
    /// between slabs and stops, which is what keeps the cores free for the draft bake the
    /// drag itself needs.
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The bands it has not delivered yet.
    ///
    /// **So a rebuild can tell "still coming" from "never asked for".** Without this, every
    /// arriving band started a rebuild that rebuilt the remaining leaves' fields, killed the
    /// worker mid-band and started it again — N bodies cost N² subdivisions and the settle took
    /// longer than the frozen version it replaced.
    pending: Vec<BakeKey>,
    /// Each finished band as it arrives, so a scene of several bodies sharpens body by body
    /// rather than all at once at the end.
    arrivals: std::sync::mpsc::Receiver<(BakeKey, std::sync::Arc<fieldform_gpu::BakedField>)>,
    /// How many are still to come, so the caller can say what is happening.
    outstanding: usize,
}

/// Everything a baked narrow band depends on, so one can be reused when none of it changed.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BakeKey {
    leaf: BakedLeaf,
    resolution: usize,
    /// The leaf's own box and the band, by bits. Exact comparison is what is wanted here: a box
    /// that moved at all is a different bake, and there is no tolerance that would be right.
    box_bits: [u64; 6],
    band_bits: u64,
    /// The cage's `geometry_fingerprint`, and nought for a mesh, whose id names one mesh for
    /// ever. **The box was standing in for this and cannot**: a cage vertex moved without moving
    /// the box — a cube's corner pulled inwards leaves three corners on every face — kept the
    /// old bake, so the view went on showing the shape before the edit.
    geometry: u64,
    /// The level a cage's field was subdivided to, and nought for a mesh. **Read off the
    /// scene's tolerance, which nothing else here describes**: Cage detail and the size of
    /// everything else on screen move it while the cage, its box and its band stay put, so a
    /// band made at one level went on being served when the tolerance asked for another. The
    /// level rather than the tolerance, because the tolerance moves with every frame of a drag
    /// of another body while the level moves only when the field built would differ.
    levels: usize,
}

/// How finely a settled rebuild samples the model: the cells across a cage's bake, and how far
/// a spine's and a profile's polylines may turn at a joint. `Display quality` in the app sets
/// it, as numbers rather than as a rung.
///
/// **Compared whole, because two rungs can agree on one number and not on the others.** This
/// used to be the bake's resolution alone, and the top two rungs both bake 192 cells, so moving
/// between them asked for nothing: the spines and profiles kept the turn they had been walked at
/// until an edit happened to rebuild. A rung's name would have caught that pair and made two
/// rungs that sample alike rebuild for nothing, so the key is the numbers the rebuild reads.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Fineness {
    /// Samples across a cage's settled bake. Nought reads as [`BAKE_RESOLUTION`].
    pub resolution: usize,
    /// How far a spine's polyline may turn at a joint in a settled rebuild, in degrees; a draft
    /// walks at [`DRAFT_SPINE_DEGREES`] where that is coarser. The app hands the angle to
    /// [`TracedView::spines`] from the same reading, so here it is only compared.
    pub spine_degrees: f64,
    /// How far a profile's polyline may turn at a joint, in degrees: what the rebuild is handed
    /// as [`Scene::profile_degrees`], from the same reading of the setting.
    pub profile_degrees: f64,
}

/// Longest edge of the frame, in pixels, while something is moving — the most a view may ask
/// for. `Display quality` asks for less on a slower card ([`View::interactive_pixels`]).
///
/// This used to be the cap on every frame, and it existed because the image was read back to the
/// CPU and handed to egui as a texture: the dearest field cost what the cheapest one did, to
/// within 5%, because the copy was the cost. The copy is gone. What remains is the trace itself,
/// which does scale with the field, so the cap stays for the frames that have to keep up with a
/// hand and lifts for the ones that do not.
///
/// What a frame cost *with* the readback, measured on master by
/// `fieldform-gpu/examples/traced_cost.rs`, square, per frame — kept because its fourth column
/// is the reason a drag has a cap at all:
///
/// | longest edge | an extruded profile | a gyroid-filled cube | a revolve | that revolve x6 |
/// |---|---|---|---|---|
/// | 720 | 6.7 ms | 7.3 ms | 7.8 ms | 15.4 ms |
/// | 1080 | 15.1 ms | 17.1 ms | 16.0 ms | 31.1 ms |
/// | 1440 | 24.8 ms | 27.4 ms | 26.5 ms | 49.2 ms |
///
/// A gyroid costs about what a cube does, but a pattern re-evaluates its input per copy, so
/// six of them double the frame at every size. The readback is gone and the trace is a few
/// milliseconds on these scenes (`shading_cost.rs`), but the pattern's multiplier is real, and
/// master's answer to it — shrinking the frame from a measured cost — cannot be carried here
/// as it was: it timed the frame around a readback that blocked on the GPU, and nothing blocks
/// now. Sizing from GPU time again wants timestamp queries; backlog item 34.
const INTERACTIVE_PIXELS: u32 = 720;

/// Longest edge of a settled frame. Larger than any viewport on this machine, so in practice a
/// still is traced at the viewport's own size; a bound at all so a wall of monitors cannot ask
/// for a gigabyte of G-buffer.
const STILL_PIXELS: u32 = 2560;

/// How many jittered frames a still averages before it is called converged — the most a view
/// may ask for. `Display quality` asks for fewer on a slower card ([`View::still_frames`]).
///
/// Eight is where the silhouettes stop visibly changing; the Halton sequence covers a pixel
/// about as evenly as eight samples can. Beyond that a frame costs what a frame costs and buys
/// nothing anyone can see.
const STILL_FRAMES: u32 = 8;


/// How many samples across the cage's bake, once an edit has settled.
///
/// This governs only how closely the *surface* is placed, not how it shades: the bake stores
/// the cage's own normal alongside the distance, so a coarse bake shades to 0.04 degrees, the
/// same as a fine one.
///
/// At 128 the atlas is about 34 MB as RGBA floats and takes roughly 0.9 seconds to build.
const BAKE_RESOLUTION: usize = 128;

/// How long the input has to be quiet before a draft is replaced by a settled bake, and before
/// an interactive frame gives way to a still.
///
/// A drag invalidates every frame, so "quiet" is the only signal that a gesture has finished
/// which does not require every caller to remember to say so — and three of them did not,
/// which is what left the viewport coarse indefinitely. Long enough not to fire between two
/// frames of a drag, short enough that letting go feels like it settles rather than like
/// something happened later.
const SETTLE_AFTER: Duration = Duration::from_millis(250);

/// The resolution used while a cage is being dragged.
///
/// A full bake is far too slow to sit inside a drag loop — measured at 926 ms against 32 ms
/// here, on the same cage. So the surface is placed coarsely while it is moving and rebuilt
/// properly the moment the gesture ends, which is the usual trade of a proxy during
/// interaction. Shading is unaffected either way, because the normals are carried rather than
/// derived from the grid, so the coarse version looks like the model rather than like a
/// low-resolution artefact of it.
const DRAFT_BAKE_RESOLUTION: usize = 32;

/// How far a spine's polyline may turn at a joint in a draft rebuild, in degrees, whatever the
/// rung asks of a settled one -- and never finer than it does.
///
/// **The draft bake's trade, made for the spine** (loop spec P9). A draft is at most
/// [`INTERACTIVE_PIXELS`] across, and at that size a joint of one degree and one of two cannot be
/// told apart; only a still can show the difference. Yet the shader walks the spine at every
/// sample, and a draft rebuild walks again every spine whose curve moved, so every frame of a
/// gesture paid for a fineness only the still after it can show. On a 310 mm cubic two degrees
/// is 97 segments where one is 193 (`fieldform-field/examples/spine_angle.rs`).
///
/// What it buys a draft frame at 720 px, against Balanced's one degree, by
/// `fieldform-gpu/examples/sweep_cost.rs` (medians of seven runs, 2026-10-12): a pipe round a
/// tight bend 11.6 → 9.5 ms, a large glyph swept round a gentle bend 15.6 → 12.4 and round the
/// tight one 84.8 → 73.2. Less than half, because the spine index already made the walk
/// sub-linear in its segments. The example's control, a straight spine that is one segment at
/// any angle, reads 0.4 ms apart between its two tables, which is the noise in those figures. A
/// settled rebuild walks the rung's own angle, so the still is the picture it was.
const DRAFT_SPINE_DEGREES: f64 = 2.0;

/// The step budget of a trace through a field that reports true distance.
///
/// `gpu_matches_cpu`'s sweep showed 512 and 4096 give byte-identical results on such a field, so
/// nothing runs out of steps at this number — **and that sentence stopped being the whole story
/// when a parameter could be driven.** A driven morph or offset divides its value to keep it a
/// lower bound, so the tracer reads short and steps short, and a fixed budget covers that much
/// less of the ray. See [`fieldform_field::step_multiplier`], which is what this is multiplied
/// by, and `fieldform-field/examples/grazing_rays.rs`, which measures the holes it leaves.
const STEP_BUDGET: u32 = 512;

/// Spare steps beyond what the multiplier accounts for.
///
/// **Because the multiplier bounds the field's rate, not the ray's path.** `step_multiplier` says
/// how much shorter each step is; it says nothing about how many steps a particular ray takes,
/// and a ray that runs nearly tangent to the surface takes far more of them than one that meets
/// it head on. That is why the holes appear along the silhouette and nowhere else.
///
/// Two, measured: `fieldform-gpu/examples/driven_morph_budget.rs` on the reported scene leaves
/// one pixel of 228345 missing at ×8.5 exactly and none at twice that. It is free — the same
/// example holds a true-distance scene at 5.8 ms flat from 512 steps to 16384 — because a ray
/// that converges stops, so the budget is a ceiling rather than an amount of work.
const STEP_HEADROOM: f64 = 2.0;

/// As many steps as any frame may take, however large the divisor.
///
/// **A cap, because the alternative to a slow frame is not a fast one.** A budget that scales
/// without limit meets the display driver's own timeout, and a reset driver loses the window
/// rather than a few pixels. Thirty-two times is past the ×20 the morph panel already warns at,
/// so a document that reaches this has been told twice; and a ray still short of the surface
/// after sixteen thousand steps is one whose band wants widening, not one more step.
const MAX_STEP_BUDGET: u32 = 16_384;

/// How good the current bake needs to be.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Quality {
    /// Mid-gesture: fast enough to keep up with a drag.
    Draft,
    /// Settled: what the model actually looks like.
    Final,
}

/// What the shader and the bake are built from. Gathered only when something changed.
pub struct Scene<'a> {
    /// The document the graph's nodes point into.
    ///
    /// Needed by shader generation and not only by the bake: a twist divides its value by how
    /// far the body it wraps reaches from the axis, and that reach comes from the document's
    /// own cages, meshes and parameters. The CPU field computes the same number, and the two
    /// have to agree or a ray stops short on one path and not the other.
    pub document: &'a fieldform_scene::Document,
    pub graph: &'a Graph,
    /// What each driven morph was divided by, from the field the app already compiled.
    ///
    /// **Handed over rather than derived here.** The bound wants the two operands' intervals and
    /// an interval needs a compiled op list, which the app has and shader generation does not.
    /// Deriving it twice, differently, is one surface in two places -- see
    /// `fieldform_gpu::generate_with`. Empty is legal and means "use the box bound", which both
    /// sides then do.
    pub morph_divisors: &'a [(fieldform_scene::NodeId, f64)],
    /// How closely a cage's limit surface has to be placed, which decides each one's
    /// subdivision depth.
    ///
    /// The tolerance rather than a depth, because the depth depends on the cage: a 10 mm one
    /// and a metre one need different numbers of levels to land within the same distance, and
    /// a document can now hold both.
    pub tolerance_mm: f64,
    /// How far off every surface the display itself reads, in millimetres: the occlusion's
    /// radius, `fieldform_gpu::Occlusion::reach` of the scene the view will be drawn with.
    ///
    /// **A profile's bake has to cover it, and the field cannot know it.** The band of fine
    /// samples about a baked boundary is sized to what the *field* reads off it -- a draft's far
    /// contour, a fillet's radius -- and beyond it the coarse level answers, short by seven
    /// fine margins. The occlusion's outer samples fell past that edge at some heights and not
    /// others, by where the surface sat in the block grid, and every baked profile came out
    /// striped at the block's period. `fieldform_field::reads_widened` has the measurement;
    /// here is the one place that knows both the bake and the display.
    pub reads_mm: f64,
    /// How much finer than `tolerance_mm` a settled profile or wall is baked: the rung's
    /// `DisplayQuality::profile_bake_fineness`, one everywhere but Enchilada. The draft tier
    /// ignores it and bakes coarse, as it always did.
    pub profile_fineness: f64,
    /// How far a profile's polyline may turn at a joint, in degrees.
    ///
    /// **The viewport's half of a trade, and the only caller that has to care.** The shader
    /// scans every point of a profile at every sample and a pattern multiplies that by its
    /// count, so this number is the frame time; it is also what banding is, since shading is
    /// the normal and a facet holds one normal across its width. See `profile_loop_turning`
    /// for the measured table. Everything off the hot path uses the finest sensible answer
    /// instead, because there the count costs nothing.
    pub profile_degrees: f64,
    /// Tessellated spine per curve, keyed by id and in any order.
    ///
    /// Ordering them is this module's job, not the caller's: the shader binds curves in the
    /// order compilation happened to intern them, and that order is only known here, after
    /// `generate`. A caller that reproduced it would be duplicating an assumption that is
    /// invisible until a second curve exists.
    pub curves: Vec<(CurveId, Vec<Point3<f64>>)>,
    /// The imported meshes the document holds, by id and in any order.
    ///
    /// Ordered here rather than by the caller for the same reason the curves are: the shader
    /// binds baked leaves in whatever order compilation interned them, and that order is only
    /// known after `generate`.
    pub meshes: Vec<(fieldform_scene::MeshId, &'a fieldform_scene::MeshData)>,
    /// The box the cage is baked over — the cage's own extent with a margin.
    ///
    /// Deliberately not `bounds`. Only the cage is baked; the pipe is closed-form and
    /// evaluated exactly in the shader. Baking over the composed bounds stretched the grid to
    /// cover a curve running well outside the cage, spending most of it on empty space and
    /// coarsening the one thing actually in it.
    pub cage_bounds: (Point3<f64>, Point3<f64>),
}

/// What changes every frame. Deliberately cheap to build: no geometry, no cloning.
///
/// The split between this and [`Scene`] is the three-tier cost model made concrete. Anything
/// in here can change without touching the shader or the bake, which is why the camera and the
/// parameter values live here and the cage does not.
pub struct View<'a> {
    pub eye: Point3<f64>,
    pub target: Point3<f64>,
    /// The app camera's own up vector, which carries whatever roll it has accumulated.
    ///
    /// Passed rather than re-derived: deriving one here disagrees with everything else the app
    /// draws through its own view matrix, and the disagreement is worst looking straight down,
    /// where a world-axis reference degenerates.
    pub up: Vector3<f64>,
    /// The middle of what is on screen and how far it reaches, which is what the ray budget
    /// and the hit threshold are derived from.
    pub scene: (Point3<f64>, f64),
    pub fov_deg: f64,
    /// Whether the viewport is orthographic. The traced view has to be told, because it builds
    /// its own rays rather than going through a projection matrix.
    pub orthographic: bool,
    /// The half-spaces to keep, each as a unit normal and an offset along it; empty for no
    /// section. What is past any of them is cut away.
    ///
    /// Part of the *view* request rather than of the graph, which is the whole design of the
    /// section tool: it changes what is drawn and nothing about what the part is.
    pub sections: Vec<(Vector3<f64>, f64)>,
    /// Resolves a named parameter, so a value can change without recompiling anything.
    pub lookup: &'a dyn Fn(ParamId) -> Option<f64>,
    /// Which body is selected, by its position among the part's bodies — the same order the
    /// shader's union fold uses, so a pixel's owner compares directly against it.
    ///
    /// Part of the view rather than of the graph: highlighting changes what is *drawn* and
    /// nothing about the part, which is the same rule the section tool follows. In particular
    /// it must not invalidate the shader, or selecting would cost a recompile.
    pub selected: Option<u32>,
    /// What the part is made of, for the lighting pass. The part's own, or the default when it
    /// has none.
    ///
    /// Here rather than in the shader for the same reason a parameter value is: a material is
    /// eight numbers in a uniform, and picking chrome must cost a buffer write, not a recompile.
    pub material: Material,
    /// Whether a settled frame measures occlusion and the key light's shadow. A view setting,
    /// so it is here beside the section rather than in the document.
    pub occlusion: bool,
    /// Whether the still's rays finish on a profile's outline rather than its bake: the rung's
    /// `DisplayQuality::exact_finish`. The draft never does.
    pub exact_finish: bool,
    /// The room the part sits in. A view setting like the section: it changes what is seen and
    /// nothing about the part, and two views of one document may light it differently.
    pub environment: EnvironmentChoice,
    /// A draft-angle read laid over the shading, or none. A view tool, like the section.
    pub draft: Option<Draft>,
    /// Longest edge a settled frame is drawn at, in pixels — what `Display quality`
    /// asks for. Zero means the viewport's own size, up to `STILL_PIXELS`.
    ///
    /// Carried rather than read from a constant because it is the one part of that
    /// setting a body with neither a cage nor a pipe can feel: an analytic solid is
    /// neither baked nor tessellated, so the image's own size is the only sharpness the
    /// setting has to offer it. The drag frame ignores it and keeps to its cap.
    pub max_pixels: u32,
    /// Longest edge of a frame drawn while something is moving, in pixels — what
    /// `Display quality` asks for. [`INTERACTIVE_PIXELS`] is the most it can be.
    pub interactive_pixels: u32,
    /// How many jittered frames a still averages before it is called converged. One means the
    /// still is the draft at the still's size, written once: no jitter, no accumulation.
    /// [`STILL_FRAMES`] is the most it can be.
    pub still_frames: u32,
}

/// Everything about a frame that, if it changed, makes the last frame's image wrong.
///
/// Compared whole against the previous frame's. Equal means the still may keep accumulating;
/// different means it starts over at the interactive tier. Every field is here because leaving
/// one out is a way for a stale image to stay on screen — a slider drag that did not restart the
/// average would blend eight frames of eight different radii.
#[derive(PartialEq)]
struct ViewKey {
    eye: [f64; 3],
    target: [f64; 3],
    up: [f64; 3],
    scene: ([f64; 3], f64),
    fov_deg: f64,
    orthographic: bool,
    sections: Vec<([f64; 3], f64)>,
    selected: Option<u32>,
    material: Material,
    occlusion: bool,
    exact_finish: bool,
    environment: EnvironmentChoice,
    draft: Option<Draft>,
    max_pixels: u32,
    interactive_pixels: u32,
    /// In the key so that choosing a quality restarts the still at the new count, rather than
    /// leaving one converged at the old.
    still_frames: u32,
    /// Every parameter value the shader reads, in slot order.
    values: Vec<f32>,
    /// The viewport's own size in pixels, so a resize starts over rather than stretching.
    width: u32,
    height: u32,
}

impl ViewKey {
    /// The key of a frame of `view`, `width` by `height` pixels on screen, whose parameter values
    /// are `values` in slot order.
    ///
    /// **`view` is taken apart whole**, so a field added to [`View`] does not compile here until
    /// it is keyed or set aside by name. Read field by field, a new one would compile without
    /// reaching the key, and that is a stale image nothing reports. The lookup is the one set
    /// aside: it reaches the key as the `values` it resolved.
    fn of(view: &View<'_>, values: Vec<f32>, width: u32, height: u32) -> Self {
        let &View {
            eye,
            target,
            up,
            scene,
            fov_deg,
            orthographic,
            ref sections,
            lookup: _,
            selected,
            ref material,
            occlusion,
            exact_finish,
            environment,
            draft,
            max_pixels,
            interactive_pixels,
            still_frames,
        } = view;
        ViewKey {
            eye: eye.coords.into(),
            target: target.coords.into(),
            up: up.into(),
            scene: (scene.0.coords.into(), scene.1),
            fov_deg,
            orthographic,
            sections: sections.iter().map(|&(n, d)| (n.into(), d)).collect(),
            selected,
            material: material.clone(),
            occlusion,
            exact_finish,
            environment,
            draft,
            max_pixels,
            interactive_pixels,
            still_frames,
            values,
            width,
            height,
        }
    }
}

/// This frame's value of every slot the shader reads, in slot order, written into the buffer
/// `spare` held. See [`note_view`] for where that buffer comes from.
fn resolve_values(
    spare: &mut Vec<f32>,
    params: &[fieldform_scene::ParamRef],
    lookup: &dyn Fn(ParamId) -> Option<f64>,
) -> Vec<f32> {
    let mut values = std::mem::take(spare);
    values.clear();
    values.extend(params.iter().map(|p| p.resolve(lookup).unwrap_or(0.0) as f32));
    values
}

/// Keeps `key` as the last frame's when it differs from it, and says whether it did.
///
/// **Whichever of the two keys is not kept gives its values back as `spare`**, the buffer the
/// next frame's are resolved into. Collected afresh, the values made one buffer and dropped
/// another every frame the viewport drew, moving or not (performance plan §0 G). Traded like
/// this, once the two exist no frame makes or drops one until a rebuild forgets the last key,
/// and its buffer with it; the second frame after that makes one again. A value drag rebuilds
/// every frame and so still pays one a frame, but beside the rebuild, which takes milliseconds
/// to the buffer's nanoseconds. The comparison is still the whole key's, every value exactly as
/// before: a value it skipped, or a hash that could collide, is a slider drag that leaves the
/// old image averaging on screen.
fn note_view(last: &mut Option<ViewKey>, spare: &mut Vec<f32>, key: ViewKey) -> bool {
    let moved = last.as_ref() != Some(&key);
    let free = if moved { last.replace(key) } else { Some(key) };
    *spare = free.map(|key| key.values).unwrap_or_default();
    moved
}

/// One submission the viewport made, waiting for the GPU to say what it cost.
struct Submitted {
    ticket: SubmissionTicket,
    /// Which call of [`TracedView::draw`] made it, so the application can put it beside that
    /// frame's own CPU time.
    frame: u64,
    still: bool,
    /// The frame traced, in pixels.
    width: u32,
    height: u32,
    /// How many rows this submission covered: the whole frame for a draft, one band of a still.
    rows: u32,
    /// For a still, which band of the frame this was, of how many at the band size chosen.
    band: Option<(u32, u32)>,
    /// For a still, which jittered frame this was, of how many.
    still_frame: Option<(u32, u32)>,
    /// For a draft, how many drafts were skipped before it because one was still running.
    skipped: u32,
    /// A warm-up band: work for the card's clocks, which measures nothing about the view.
    warm: bool,
}

/// What one [`TracedView::rebuild`] cost the interface's thread, in the three parts that want
/// different fixes. Plan §5 V2 asks for exactly this before anything is built: "first measure
/// the in-app per-edit cost, which is still unmeasured".
#[derive(Clone, Copy, Debug)]
pub struct RebuildCost {
    /// Graph to WGSL.
    pub generate_ms: f32,
    /// Bakes, spines re-sampled, profiles flattened, surfaces: everything handed to the card.
    pub prepare_ms: f32,
    /// The renderer: the driver turning the WGSL into a pipeline, and the uploads. **The
    /// renderer's own four parts, which `fieldform_gpu::BuildCost` names, and nothing else** —
    /// letting go of the one it replaces is `release_ms` below.
    pub pipeline_ms: f32,
    /// **Letting go of the renderer this one replaces**: its skip grid and, where the source
    /// changed, its pipelines, and where the bands changed, its atlas texture.
    ///
    /// Its own number because it used to be inside `pipeline_ms` by accident — the old renderer
    /// was bound in the arm that built the new one, so it was freed before the clock was read —
    /// and on the M4000 a settled rebuild spent 1.4 s that none of `BuildCost`'s parts owned up
    /// to. A driver may wait for the card to finish with a resource before freeing it, which is
    /// work that belongs to the frame it happens on however little it belongs to the rebuild.
    pub release_ms: f32,
    /// How big the shader this rebuild compiled was — see [`fieldform_gpu::ShaderSize`].
    ///
    /// **On the line beside the time, because the time alone cannot be acted on** (plan step 13).
    /// The driver's compile is a power of about 2 to 2.5 in the inlined size of one entry point
    /// (`union-compile-8fa06b7.md`), so a slow rebuild is a question about the graph — and
    /// `shader_size_cost` puts the limit at 200 bodies or 400 stacked ops on the A500. A person
    /// reading "pipeline 8148" learns nothing; reading it beside "273 KB, 200 bodies" they know
    /// which way to go.
    pub size: fieldform_gpu::ShaderSize,
}

impl RebuildCost {
    pub fn line(&self) -> String {
        format!(
            "rebuild {:.1} ms (generate {:.1}, prepare {:.1}, pipeline {:.1}, release {:.1}) \
             [{}]",
            self.generate_ms + self.prepare_ms + self.pipeline_ms + self.release_ms,
            self.generate_ms,
            self.prepare_ms,
            self.pipeline_ms,
            self.release_ms,
            self.size,
        )
    }
}

/// A submission and what it cost, for the perf log.
///
/// The loop spec's S8: the application says what each frame cost, in the GPU's own time where
/// the card can give it, so a machine without cargo can fill the ledger's second column.
pub struct PerfReport {
    /// The [`TracedView::frame`] that submitted it.
    pub frame: u64,
    /// `still` for the settled tier, `draft` for the one that keeps up with a hand.
    pub still: bool,
    /// Longest edge of the frame traced, in pixels.
    pub pixels: u32,
    pub band: Option<(u32, u32)>,
    pub still_frame: Option<(u32, u32)>,
    /// Drafts skipped ahead of this one because the previous was still running.
    pub skipped: u32,
    /// A warm-up band rather than a frame: see `CLOCKS_HOLD`. Its cost is of a cold card by
    /// design, so nothing is sized from it, and it is logged so that it can be seen to run.
    pub warm: bool,
    pub cost: SubmissionCost,
}

impl PerfReport {
    /// The line the loop spec asks for, given the CPU time of the frame that submitted it:
    /// `draft 720px 43.1 ms (march 38.2, shade 0.6, cpu 2.1)` and
    /// `still 2/8 band 3/12 1920px 41.2 ms (…)`, or `(wall, cpu 2.1)` on a device that cannot
    /// write timestamps — so an excerpt never passes a wall time off as a GPU one.
    pub fn line(&self, cpu_ms: Option<f32>) -> String {
        let tier = match (self.still, self.still_frame) {
            (true, Some((k, n))) => format!("still {k}/{n}"),
            (true, None) => "still".to_string(),
            (false, _) if self.warm => "warm".to_string(),
            (false, _) => "draft".to_string(),
        };
        let tier = match self.band {
            Some((k, n)) => format!("{tier} band {k}/{n}"),
            None => tier,
        };
        let cpu = cpu_ms.map_or(String::new(), |ms| format!(", cpu {ms:.1}"));
        let skipped = if self.skipped > 0 { format!(", skipped {}", self.skipped) } else { String::new() };
        let breakdown = match (self.cost.march_ms, self.cost.shade_ms) {
            (Some(march), Some(shade)) => {
                format!("march {march:.1}, shade {shade:.1}{cpu}{skipped}")
            }
            _ => format!("wall{cpu}{skipped}"),
        };
        format!("{tier} {}px {:.1} ms ({breakdown})", self.pixels, self.cost.ms())
    }
}

/// The draft tier's measurements of one shader, and which shader they are of (item 105).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LearnedCosts {
    /// A hash of the shader's source: the costs describe that program and no other.
    pub(crate) source: u64,
    pub(crate) draft_rung: usize,
    pub(crate) draft_fast_streak: u32,
    pub(crate) draft_ms_per_pixel: Option<f32>,
    pub(crate) draft_ms_per_row: Option<(u32, u32, f32)>,
    pub(crate) last_draft_band_rows: Option<u32>,
    pub(crate) still_ms_per_row: Option<(u32, u32, f32)>,
}

/// A shader source's identity for [`LearnedCosts`].
///
/// A hash rather than the text, which runs to hundreds of kilobytes. A collision would hand one
/// shader another's starting rung, which the ladder corrects within a few frames — a cost, never
/// a wrong picture, so a 64-bit hash is ample.
pub(crate) fn source_identity(source: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

/// The costs to carry into a newly built shader: what was learned, if it was learned of this
/// same source, and nothing otherwise.
///
/// **A new shape starts over; the same shape does not.** The first half is the rule the reset
/// always had, and is right: a different program costs differently. The second is item 105.
pub(crate) fn costs_to_keep(learned: Option<LearnedCosts>, source: &str) -> Option<LearnedCosts> {
    learned.filter(|l| l.source == source_identity(source))
}

#[derive(Default)]
pub struct TracedView {
    /// The device every GPU resource in hand was made on, or `None` while none is held.
    ///
    /// **One device for all of it, and asked of the view rather than of the renderer.** A
    /// device that has gone takes everything made on it: the march's pipelines and buffers,
    /// its atlas and its skip grid, the lighting pass, the handles egui samples the written
    /// textures through, and the submissions it has not reported on. `Renderer::device`
    /// answered for the march's pipeline alone, so a frame drawing with another device kept
    /// the lighting pass of the one before and handed egui a texture of a device that could
    /// not sample it -- a validation error, and wgpu's answer to one is to end the process.
    /// See [`TracedView::ready_to_draw_on`].
    built_on: Option<wgpu::Device>,
    renderer: Option<Renderer>,
    program: Option<ShaderProgram>,
    /// The lighting pass, its environment and the frame targets. Built on the first draw and
    /// kept across every renderer rebuild: none of it depends on the document.
    shading: Option<ShadingResources>,
    /// Every environment baked so far, so switching back to one is an upload and not a bake.
    /// A few megabytes each on the CPU, and there are three.
    environments: std::collections::HashMap<EnvironmentChoice, EnvironmentMaps>,
    /// The environment the lighting pass currently holds.
    environment: Option<EnvironmentChoice>,
    /// egui's handle for each set of targets still allocated, by generation: the drag frame
    /// and the still are two textures, and a handle whose set has been freed is freed with it.
    display: Vec<(u64, egui::TextureId)>,
    /// What the last frame drew, so an identical request accumulates instead of starting over.
    last_view: Option<ViewKey>,
    /// The buffer the next frame's parameter values are resolved into. See [`note_view`].
    spare_values: Vec<f32>,
    /// When the view last changed, for the tier switch.
    view_changed_at: Option<Instant>,
    /// How many jittered frames of the current still are averaged so far.
    accumulated: u32,
    /// Reported in the status bar rather than swallowed: a viewport that silently shows
    /// nothing is indistinguishable from a model that is not there.
    error: Option<String>,
    /// The quality this needs rebuilding at, if it needs rebuilding.
    stale: Option<Quality>,
    /// The quality it was last built at, so a draft can be replaced by a final one.
    ///
    /// This was written and never read, and the promotion its own comment describes did not
    /// exist: the view reached final quality only when a caller happened to ask, and every
    /// settle path that returned early — a drag that ended without moving anything — left a
    /// 32³ bake on screen permanently. Reading it here is what makes that class of mistake
    /// impossible rather than fixed three times.
    built: Option<Quality>,
    /// When something last said the view was out of date, for [`SETTLE_AFTER`].
    last_invalidated: Option<Instant>,
    /// How finely a settled rebuild samples the model. See `DisplayQuality` in the app.
    fineness: Fineness,
    /// How many steps a ray gets, when the graph has been looked at.
    ///
    /// `None` until the first rebuild, and read as [`STEP_BUDGET`] then: the field defaults to
    /// zero and a budget of zero draws an empty viewport, which is the failure this whole field
    /// exists to prevent, so it is never a bare number.
    step_budget: Option<u32>,
    /// What a field reading is multiplied by before occlusion and shadow treat it as a distance.
    ///
    /// `None` until the first rebuild and read as one then, the way [`Self::step_budget`] is: the
    /// field defaults to zero and a scale of zero shades everything black, which is the failure
    /// this exists to prevent.
    value_scale: Option<f64>,
    /// What the last rebuild's three parts cost, and at which quality — read by the slow-frame
    /// line, which is trying to answer "why did that hang" and must not consume the perf log to
    /// do it.
    last_rebuild: Option<(Quality, RebuildCost)>,
    /// What the last renderer's own four parts cost — see `fieldform_gpu::BuildCost`. Read by
    /// the slow-frame line, which needs to say which of them a freeze was.
    last_build: Option<fieldform_gpu::BuildCost>,
    /// What each leaf of the last rebuild cost to make ready: building its field, and baking
    /// it. Split because "preparing the bands" covers both and they fail differently — a field
    /// rebuilt per frame is a cache that is not working, a slow bake is a resolution too high.
    last_bakes: Vec<(String, f32, f32)>,

    /// The worker's thread pool, made once and kept.
    ///
    /// **Not per bake.** A named parameter's slider calls `rebuild_render_resources` per frame,
    /// which asks for a settled rebuild per frame, and a pool built and dropped each time is
    /// (cores − 2) threads created and destroyed at the frame rate.
    pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    /// How many rebuilds have been asked for. A compile carries the number it was started
    /// for, and one that comes back older than this is for a shape that has since changed —
    /// installing it anyway put a hidden body back on screen.
    asked: u64,
    /// A shader being turned into pipelines off the interface thread, if one is.
    ///
    /// **Because the driver can take half a minute.** Measured in the application, on the
    /// owner's wing: including a mirror in a blend changed the shader's structure, and
    /// `create_compute_pipeline` took **29.2 seconds** with everything else in that frame under
    /// 50 ms. Nothing about it needs the interface thread — the device and the queue are shared
    /// handles — so it goes to a worker and the picture already on screen stays up, and stays
    /// interactive, until the new one is ready.
    ///
    /// Only a *new shader* goes this way. A rebuild whose source is unchanged keeps its
    /// pipelines (`Renderer::rebuilt_from`), which is every drag of a number, and those stay on
    /// this thread where they have always been.
    compiling: Option<Compiling>,
    /// A settled bake being computed off the interface thread, if one is.
    ///
    /// **Because the bake was the one thing that stopped the application.** A finished edit asks
    /// for a settled rebuild, and on the owner's part that measured 1.3 to 2.8 seconds in a
    /// single frame — every one of them inside `BakedField::new`, with the compile and the
    /// drawing resources at 3 ms between them. Nothing about it needs the interface thread: it
    /// samples a field into two arrays and hands them over. So it goes to a worker, the coarse
    /// band already on screen stands in meanwhile, and the fine one is swapped in when it
    /// lands — the same draft-then-settle the image itself has always used, applied to the
    /// thing the draft was standing in for.
    ///
    /// What each tier costs on the starting cage, measured by `fieldform-gpu/examples/bake_cost.rs`
    /// on 2026-09-19:
    ///
    /// | cells | bake | held |
    /// |---|---|---|
    /// | 32 (draft) | 14.5 ms | under a megabyte |
    /// | 64 | 129 ms | 4 MB |
    /// | 128 | 806 ms | 34 MB |
    /// | 192 | 2.54 s | 115 MB |
    /// | 256 | 5.68 s | 271 MB |
    ///
    /// The draft is what a hand can be answered with, and everything above it is what a worker
    /// is for. Display quality picks the settled row, so the wait this removes grows with it.
    baking: Option<Baking>,
    /// Narrow bands already baked, kept across rebuilds.
    ///
    /// A bake is a pure function of one leaf's geometry, its own box, the resolution and the
    /// band — and a show/hide changes none of those for the bodies it did not touch. Rebuilding
    /// them anyway was **890 ms on thirty small solids** (`hide_cost.rs`), against 3.8 ms to
    /// generate and validate the shader: hiding one body re-baked the twenty-nine that had not
    /// changed.
    ///
    /// Keyed by everything the bake depends on, so a stale entry is impossible rather than
    /// unlikely. The box and the band are `f64` and are keyed by their bits, which is exact:
    /// two boxes that differ in the last place really are different bakes.
    bakes: std::collections::HashMap<BakeKey, std::sync::Arc<fieldform_gpu::BakedField>>,
    /// The spines the last settled rebuild was handed, kept against what they were tessellated
    /// from. See [`KeptSpines`] and [`Self::spines`].
    kept_spines: KeptSpines,
    /// The same for the last draft, which walks at [`DRAFT_SPINE_DEGREES`].
    ///
    /// **Apart from the settled ones, because the two tiers walk at different angles.** One kept
    /// set would be walked afresh at every change of tier, so a gesture and the settle after it
    /// would each walk every spine in the part again, moved or not -- the walks keeping them was
    /// for.
    draft_spines: KeptSpines,
    /// The program the last rebuild generated, until something invalidates the view. See
    /// [`Self::program_for`].
    kept_program: Option<KeptProgram>,
    /// How many times [`Self::draw`] has been called. Names a frame in a [`PerfReport`].
    frame: u64,
    /// Submissions the GPU has not yet reported on.
    submitted: Vec<Submitted>,
    /// The next row of the still frame being traced, since a still is traced in bands.
    still_cursor: u32,
    /// How many bands of the current still frame have been submitted.
    still_band: u32,
    /// The dearest band measured of this scene, as milliseconds per row at the still's size
    /// `(width, height)` — what the next band is sized from.
    ///
    /// The dearest rather than the latest, because a band across the background costs almost
    /// nothing and a band through the model costs everything: sizing from a cheap one would
    /// put the next band several times over its budget. The dearest can only shrink the band,
    /// which is the safe direction. Kept across view changes and dropped on a rebuild: the
    /// dearest rows of a scene cost about the same from any angle, and the first live run
    /// showed the estimate starting from nothing at every camera move was what let a band
    /// grow past the ceiling.
    still_ms_per_row: Option<(u32, u32, f32)>,
    /// What the last draft cost per pixel, as a floor under the still's estimate: a still does
    /// everything a draft does and more, so its rows can never be cheaper than the draft's
    /// pixels say — which is what a band across the background would otherwise claim.
    draft_ms_per_pixel: Option<f32>,
    /// Rows in the last band submitted of this view, which the next may exceed by at most
    /// [`BAND_GROWTH`]. `None` until a band of this view has gone up.
    ///
    /// **Taken where the band is submitted, never from a report, and that is the staleness
    /// guard.** A cost arrives a frame or more after its band and the view can have changed in
    /// between, so a report cannot be trusted to describe the current view. Reading the rows at
    /// submission and clearing them when the view changes makes a stale band impossible here
    /// rather than merely unlikely.
    ///
    /// It is also why the two estimates above may safely be a frame or more out of date:
    /// however wrong they are, what actually runs is at most [`BAND_GROWTH`] times a band this
    /// view really submitted. A field recording which view each cost came from was carried here
    /// for a while and never read, because there was nothing left for it to decide — the cap is
    /// the answer, and `still_ms_per_row` keeps a previous view's band on purpose.
    last_band_rows: Option<u32>,
    /// The next row of the draft being traced, when a draft goes up in bands.
    ///
    /// **Deliberately not cleared when the view changes, which the still's cursor is.** A still
    /// accumulates one fixed image and a new camera voids it; a banded draft is a rolling
    /// shutter over a camera that is still moving, and restarting it at row nought every frame
    /// of a drag would refresh the top band for ever and leave the rest of the picture stale.
    /// So the sweep carries on across views and the cost is that the strips of one picture are
    /// traced from cameras a frame or two apart — which is the shear P10 trades for the
    /// interface's own cadence, and only on documents that miss [`INTERFACE_BUDGET_MS`] as a
    /// whole frame.
    draft_cursor: u32,
    /// How many bands of the current draft have been submitted.
    draft_band: u32,
    /// The dearest band measured of this scene, as milliseconds per row at the draft's size.
    ///
    /// **The dearest rather than the mean, because the mean is what makes a band miss.** The
    /// cost of a frame is concentrated in the rows the model covers; a mean over the whole
    /// frame understates those rows by up to seventeen times (`interface_latency`), and a band
    /// sized from it lands on the model at several budgets.
    draft_ms_per_row: Option<(u32, u32, f32)>,
    /// Rows in the last band of a draft submitted at this size, which the next may exceed by at
    /// most [`BAND_GROWTH`] — the same guard, and for the same reason, as [`Self::last_band_rows`].
    ///
    /// Its own field rather than that one, because the two tiers alternate: a cap is only worth
    /// anything if it names a band of the tier it is capping.
    last_draft_band_rows: Option<u32>,
    /// The size, cost and rows of the draft sweep being measured, while one is incomplete.
    ///
    /// **Because the ladder's question is about a frame and a band is not one.** `next_rung`
    /// asks whether a picture of this size can be drawn inside [`DRAFT_BUDGET_MS`], and every
    /// band is cheap by construction — feeding it one would climb the ladder back to 720 and
    /// hold it there while a sweep took a fifth of a second. So the bands of a sweep are summed
    /// and the ladder is told once, about the frame. A sweep interrupted by a resize starts
    /// over, since half of one size and half of another is a measurement of neither.
    draft_sweep: Option<(u32, u32, f32, u32)>,
    /// Whether the still's texture holds a complete frame of the current view. Until it does,
    /// the draft's texture stays on screen — a still is written band by band, and showing it
    /// before its first frame is complete would show the previous view in the rows not yet
    /// reached.
    still_shown: bool,
    /// The targets the last draft was drawn into, so they can stay on screen while a still's
    /// first frame is being traced.
    draft_generation: Option<u64>,
    /// Which rung of [`DRAFT_LADDER`] a draft is drawn at. Not in the view key: changing it
    /// must not restart anything, and the still tier ignores it.
    draft_rung: usize,
    /// How many drafts running have come in under [`DRAFT_CLIMB_MS`].
    draft_fast_streak: u32,
    /// What the draft tier had learned about the shader on screen when a rebuild began, kept
    /// until the new shader is known so it can be handed back if the shader is the same one.
    ///
    /// **Item 105.** A rebuild used to send the ladder to its top rung and forget every cost it
    /// had measured, on the grounds that "the new graph may be cheaper". True of a new shape —
    /// and false of a rebuild that regenerated the same source and adopted the pipeline it already
    /// had, which is what `snap`'s `invalidate`, a bake moving from draft to final on a document
    /// with no cage, and anything else that asks for a rebuild without changing the shape all do.
    /// Measured on 48 gyroid-filled shells over eight view changes: eight rebuilds, seven of them
    /// 0.5 ms no-ops, and all sixteen drafts left at 517 px — the ladder never got one rung down,
    /// on the document whose drafts most need it.
    learned: Option<LearnedCosts>,
    /// How many drafts were not submitted because the previous one was still running, since
    /// the last one that was. Reported on that one's perf line, so the effect is visible.
    draft_skipped: u32,
    /// Whether every body drawn has a finite extent, so the scene's sphere encloses all of
    /// them and a ray may start where it enters it (`Camera::start_at_scene`). A plane makes
    /// this false, and the march then starts at the eye as it always did.
    scene_bounded: bool,
    /// The camera, frame size and step budget of the last frame drawn: what the warm-up marches,
    /// and what a march-stats capture traces so that it is what is on screen.
    last_frame: Option<(Camera, u32, u32, u32)>,
    /// When the card was last given anything to do, which is what says its clocks have dropped.
    last_gpu_work: Option<Instant>,
    /// Until when the warm-up runs, while one is running. See [`CLOCKS_HOLD`].
    warm_until: Option<Instant>,
    /// What each rebuild since the perf log last asked cost, for that log.
    rebuilds: Vec<RebuildCost>,
    /// What the renderer's skip grid was filled from -- which renderer, which parameter values,
    /// which cube -- or `None` while it has none. See [`SKIP_CELLS`].
    skip_key: Option<(u64, Vec<f32>, [u64; 4], u32)>,
    /// Whether the document is dear enough for the grid to pay. See [`SKIP_WORTH_MS`].
    skip_wanted: bool,
    /// The parameter values the last frame was drawn with, for the same capture. Feature only.
    #[cfg(feature = "march-stats")]
    last_values: Vec<f32>,
}

/// What one submission of a still may cost, in milliseconds.
///
/// A fifth of the loop spec's 500 ms ceiling per submission on the Quadro M4000, which is
/// itself a four-times margin under the two-second reset — and small enough that input
/// between bands feels answered at once, since a band is the unit that can be cancelled.
const BAND_BUDGET_MS: f32 = 100.0;

/// The fewest pixels a band may hold: enough threads to fill the card.
///
/// **Measured on the first live run, and it reversed an assumption.** The floor was eight
/// rows, and on a six-copy pipe pattern the bands shrank to it and stayed there: an 8 × 1080
/// band is 8 600 threads, a quarter of what an RTX A500 runs at once, so each band ran at the
/// speed of its slowest ray and the perf log showed 80–130 ms for eight rows against 633 ms
/// for the whole 720-row frame in one submission — seven times the work, in bands. Thirty-two
/// thousand pixels is about one full wave on that card; a band of fewer would be paying latency
/// rather than doing work. It is also the size of a still's first band of a view, since a band
/// that small is the cheapest honest measurement there is.
const MIN_BAND_PIXELS: u32 = 1 << 15;

/// Rows in one tile of the per-tile body mask, which is the march's workgroup height.
///
/// A still's bands are rounded to a whole number of these, so that each one after the first
/// starts on a tile row. `march.wgsl` declares the workgroup as `8, 8, 1`; if that changes, this
/// must change with it, and the renderer stays correct either way -- it ANDs the masks of the two
/// tile rows a straddling workgroup covers -- but culls less.
const TILE_ROWS: u32 = 8;

/// A still band's row count, rounded to a whole number of tile rows and no taller than the frame.
///
/// **So every band after the first starts on a tile row.** The march's workgroup is 8 × 8, and a
/// band whose first row is not a multiple of 8 puts one workgroup across two tile rows, where the
/// per-tile body mask has to take the conservative AND of both (`tile-mask-design-01941c5.md`
/// Fault 1). That is still correct -- the renderer must be, for any band, since this is not its
/// only caller -- but it culls less, and rounding here makes the AND a no-op in the application.
/// At most seven rows over the budget, against a band of dozens.
///
/// Only the *count* is rounded, because a band's first row is the previous band's last: whole
/// counts from nought give whole starts. The last band ends at the frame's height, which need not be
/// a multiple of 8, and nothing follows it.
fn tile_aligned_rows(rows: u32, height: u32) -> u32 {
    rows.max(1).next_multiple_of(TILE_ROWS).min(height)
}

/// How much larger than the last band of this view the next may be.
///
/// The cost per row changes smoothly down a frame, from nothing across the background to the
/// model's own rows, so the last band is the best predictor of the next — and an unbounded
/// jump from a band of cheap rows into the model would be a band of the whole frame. Half
/// again is the most a band may grow, so entering the model costs at most half a budget more
/// than the band before it.
const BAND_GROWTH: f32 = 1.5;

/// How much dearer a still's dearest rows are than the draft's mean pixel, as a floor under the
/// band estimate before any dense band of a scene has been measured.
///
/// The draft's cost is a mean over the whole frame, background included, and a still adds
/// five occlusion samples and a shadow march to every hit pixel. On the first live run the
/// dearest still rows of a six-copy pipe pattern measured five times the draft's mean, and a
/// floor of one let the band grow across the cheap background into a 157-row band that took
/// 504 ms on the model — over the 500 ms ceiling bands exist to hold. Four keeps that band to
/// about a budget and a quarter; the dearest band measured takes over from there.
const DRAFT_TO_STILL: f32 = 4.0;

/// What a draft frame may cost, in milliseconds, on the slowest machine the display is held to.
///
/// The loop spec's §3: twenty frames a second is the floor at which an orbit reads as an orbit.
/// Stated on the Quadro M4000, so the laptop simply has to be faster.
pub const DRAFT_BUDGET_MS: f32 = 50.0;

/// Cells a side of the viewport's skip grid (plan §5 V5; `march.wgsl` says what it is).
///
/// Measured by `fieldform-io/examples/framed_capture.rs`, the model framed, a 720 px draft's
/// march on the card's own clock:
///
/// | | none | 16³ | 32³ | 64³ | 96³ |
/// |---|---|---|---|---|---|
/// | patterned pipe, march | 71.0 ms | 61.3 | 59.7 | 58.4 | 58.8 |
/// | patterned pipe, filling the grid | -- | 0.4 ms | 0.9 | 3.2 | 9.1 |
/// | attractor-infill, march | 8.3 ms | 6.7 | 6.3 | 5.9 | 5.7 |
/// | a bare sphere, march | 0.1 ms | 0.3 | 0.3 | 0.4 | 0.4 |
///
/// Thirty-two takes nearly all there is and fills in under a millisecond on the dearest
/// document here, which is what lets it be refilled on every step of a parameter drag rather
/// than waiting for the view to settle.
const SKIP_CELLS: u32 = 32;

/// The same, once the field has stood still for [`SETTLE_AFTER`]. With a pattern's copies
/// masked cell by cell (plan §5 V6) the grid's fineness is worth more than it was for skipping
/// alone, because a smaller cell rules out more copies: the patterned pipe's draft marches in
/// 22.7 ms over 32³ and 15.3 over 64³, against 59.7 with the grid and no masks and 71.0 with
/// neither. Filling 64³ is 6 ms on that document, which is too much to pay on every step of a
/// drag and nothing to pay once.
const SKIP_CELLS_SETTLED: u32 = 64;

/// The draft below which the grid is not worth asking. The last row above: a field that costs
/// a nanosecond a reading is cheaper to read than to look up.
const SKIP_WORTH_MS: f32 = 4.0;

/// How long the card's clocks stay up after its queue empties, as far as it is safe to assume.
///
/// Plan §5 V4, measured by `fieldform-io/examples/idle_clocks.rs` on the RTX A500 with the
/// patterned pipe, whose steady 720 px draft is 82 ms:
///
/// | idle before the draft | first draft | against steady |
/// |---|---|---|
/// | 1 s | 80 ms | 0.9 |
/// | 2 s | 242 ms | 2.8 |
/// | 5 s | 257 ms | 3.1 |
/// | 5 s, 8 to 32 rows re-traced every 33 to 250 ms throughout | 254 to 371 ms | 3.1 to 4.5 |
/// | 5 s, first draft at 240 px instead of 720 | 169 ms (steady: about 9) | -- |
/// | 5 s, then 200 ms of bands back to back | 98 ms | 1.1 |
/// | 5 s, then a sixth of the draft each 16 ms frame for 250 ms | 104 ms | 1.3 |
///
/// So the clocks answer to a *full* queue and to nothing less. A trickle through the idle is
/// worse than leaving the card alone, and would cost power for as long as the window sat; a
/// smaller first draft waits for the clocks all the same, because the wait is the clocks and
/// not the pixels. What works is a fifth of a second of the real march, flat out, just before
/// it is needed -- and a pointer moving over the viewport is that notice. A drag begun with a
/// pointer that was already resting there gets no warm-up and is as slow as it was; so is a
/// number typed into a panel.
const CLOCKS_HOLD: Duration = Duration::from_secs(1);

/// How long one warm-up runs. The measured knee is 200 ms flat out; paced by the interface's
/// frames it is a little less than flat out, so a little longer.
const WARM_FOR: Duration = Duration::from_millis(300);

/// A warm-up band is this fraction of a draft's pixels: long enough cold to outlast an
/// interface frame, so the queue stays full, and short enough that a press waits on one band
/// for less than the slow draft it is there to prevent. A sixth measured 1.3 times steady.
const WARM_SHARE: f32 = 6.0;

/// A view is worth warming for when its steady draft is at least this fraction of the draft
/// budget: a slow first draft is about three steady ones, so below a third of the budget even
/// the slow one is inside it.
const WORTH_WARMING: f32 = 3.0;

/// Longest edge of a draft frame, in pixels, from the size a drag wants down to the rung that
/// keeps a Quadro M4000 inside the budget on a patterned pipe. The shading spec's §9 tiers with
/// one more rung under them (plan step 3, loop spec P8).
const DRAFT_LADDER: [u32; 4] = [720, 480, 360, 240];

/// What one submission may cost if the interface is to keep its own cadence while a march is
/// queued ahead of the presentation — the loop spec's P10.
///
/// **This sizes a band; it does not decide whether to band.** That decision is
/// [`draft_bands`], against the draft tier's own budget, and conflating the two was item 104.
///
/// Two thirds of a 16.7 ms frame, leaving egui its own share of one.
///
/// **It is a budget, not a promise, and the measurement says why.** A band cannot be smaller
/// than [`MIN_BAND_PIXELS`], and the cost of a frame is not spread evenly down it: on a radial
/// pattern of forty-eight gyroid-filled shells at 720 px, `fieldform-gpu/examples/interface_latency.rs`
/// measures 3.6 ms for a band across the background and 61.9 ms for one across the model —
/// seventeen times, for the same thirty-eight rows. So the dearest band of a heavy document is
/// several budgets whatever this number says, and banding buys the difference between the
/// dearest band and the whole frame (203 ms there, so 3.3x) rather than a 16 ms frame. Reaching
/// one would need fewer steps per ray, which is a different lever.
const INTERFACE_BUDGET_MS: f32 = 11.0;

/// A draft under this many milliseconds, this many frames running, climbs a rung.
///
/// Hysteresis is the point: a cap that steps down over 50 ms and up under 50 would oscillate
/// between two rungs at every frame the scene straddled it, and an image that changes size
/// every frame is worse than a small one. Down at once, up only when it has been cheap for a
/// while.
const DRAFT_CLIMB_MS: f32 = 20.0;
const DRAFT_CLIMB_STREAK: u32 = 5;

/// The draft ladder's rule: where the next draft is drawn, given where the last one was, how
/// many cheap ones preceded it, and what it cost.
///
/// Down a rung at once when a draft misses the budget; up a rung only after a run of cheap
/// ones. Between the two thresholds the streak starts over, so a scene that sits at 30 ms
/// stays where it is rather than creeping up to a size it cannot afford.
fn next_rung(rung: usize, streak: u32, ms: f32) -> (usize, u32) {
    if ms > DRAFT_BUDGET_MS {
        ((rung + 1).min(DRAFT_LADDER.len() - 1), 0)
    } else if ms < DRAFT_CLIMB_MS {
        if streak + 1 >= DRAFT_CLIMB_STREAK && rung > 0 {
            (rung - 1, 0)
        } else {
            (rung, streak + 1)
        }
    } else {
        (rung, 0)
    }
}

/// Whether a draft of this size is too dear to go up as one submission — **the last resort,
/// after the ladder has run out of smaller pictures to draw.**
///
/// Both halves are the point, and getting the second one wrong is backlog item 104: this read
/// the frame against [`INTERFACE_BUDGET_MS`] and ignored the rung, so it fired on any draft
/// dearer than 11 ms. That is most real documents, including every one the ladder was already
/// fitting in a single frame — and a healthy 20 ms draft became seven bands, because a band
/// cannot go below [`MIN_BAND_PIXELS`]. A picture that arrived in one frame arrived over seven
/// with the camera moving between them, which the owner met as the viewport banding and skipping
/// frames. The shear P10 traded for a bounded interface frame was being paid by documents that
/// were never slow.
///
/// So the trigger is the draft tier's own budget at the ladder's last rung:
///
/// - **The rung**, because shrinking the picture is what the ladder is for and it is the cheaper
///   answer — it costs resolution, where a band costs the picture its coherence. Banding is what
///   is left when there is no smaller rung.
/// - **[`DRAFT_BUDGET_MS`]**, because that is the budget a draft actually has. A draft inside it
///   is a draft that was never a problem; [`INTERFACE_BUDGET_MS`] still *sizes* the bands once
///   the decision to band has been taken, which is a different question.
///
/// Asked of the mean pixel of the *whole* frame, which is what [`TracedView::draft_ms_per_pixel`]
/// holds, so the question is about the picture and not about whichever part of it was traced
/// last. With nothing measured yet the answer is no: the first draft of a view goes up whole,
/// which is both what the ladder measures and the cheapest honest measurement there is.
fn draft_bands(rung: usize, ms_per_pixel: Option<f32>, width: u32, height: u32) -> bool {
    rung + 1 == DRAFT_LADDER.len()
        && ms_per_pixel
            .is_some_and(|ms| ms * (width as f32) * (height as f32) > DRAFT_BUDGET_MS)
}

/// What a draft sweep has cost and how many rows it has covered once this band is added.
///
/// A band of a different size starts the sum over, since half a sweep at one size and half at
/// another measures neither — see [`TracedView::draft_sweep`].
fn sweep_with(
    sweep: Option<(u32, u32, f32, u32)>,
    width: u32,
    height: u32,
    ms: f32,
    rows: u32,
) -> (f32, u32) {
    match sweep {
        Some((w, h, was_ms, was_rows)) if w == width && h == height => {
            (was_ms + ms, was_rows + rows)
        }
        _ => (ms, rows),
    }
}

/// How many rows the next band of a still may cover, given the dearest cost per row this view
/// has shown at this size, the rows of the band before it, and the frame's size.
///
/// **The cap is what bounds a wrong estimate, and that is why no report needs to say which view
/// it measured.** A band's cost arrives a frame or more after it was submitted, so `per_row` may
/// describe a view that has since moved on. `last` cannot: it is read where the band goes up and
/// cleared when the view changes. So the worst a stale cost can do is ask for [`BAND_GROWTH`]
/// times a band this view really ran — which is the whole of the guard.
///
/// With no estimate yet, the answer is the floor: the smallest band that fills the card, which
/// is the cheapest honest measurement there is.
fn band_rows(
    budget: f32,
    per_row: Option<f32>,
    last: Option<u32>,
    width: u32,
    height: u32,
) -> u32 {
    let floor = MIN_BAND_PIXELS.div_ceil(width.max(1)).clamp(1, height);
    match (per_row, last) {
        (Some(ms), Some(last)) => {
            let wanted = (budget / ms.max(1e-6)) as u32;
            wanted.min(((last as f32) * BAND_GROWTH) as u32)
        }
        _ => floor,
    }
    .clamp(floor, height)
}

impl TracedView {
    /// Traces the view on screen once more through the counting march, at `pixels` on its
    /// longest edge, and hands back the counts with the generated source and the parameter
    /// values it ran with. The camera is the last frame's own, so a capture at 240 and one at
    /// 1080 are the same rays at two densities — which is what a draft and a still are.
    #[cfg(feature = "march-stats")]
    pub fn capture_march_stats(
        &self,
        pixels: u32,
    ) -> Result<(fieldform_gpu::stats::MarchCapture, String, Vec<f32>), String> {
        let (Some(renderer), Some(program)) = (&self.renderer, &self.program) else {
            return Err("nothing has been drawn yet".to_string());
        };
        let Some((mut camera, wide, high, budget)) = self.last_frame else {
            return Err("nothing has been drawn yet".to_string());
        };
        let shrink = pixels as f32 / wide.max(high).max(1) as f32;
        let width = ((wide as f32) * shrink).round().max(1.0) as u32;
        let height = ((high as f32) * shrink).round().max(1.0) as u32;
        camera.jitter = [0.0, 0.0];
        camera.occlusion = None;
        renderer
            .march_stats(&camera, width, height, budget)
            .map(|capture| (capture, program.source.clone(), self.last_values.clone()))
            .map_err(|e| e.to_string())
    }

    /// Which call of [`Self::draw`] this is, so a caller can pair its own timing of the frame
    /// with the reports that name it.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Whether the costs reported are the GPU's own timestamps, or `None` before the first
    /// frame has built the lighting pass and it is not yet known.
    pub fn timestamps_available(&self) -> Option<bool> {
        self.shading.as_ref().map(|s| s.timestamps_available())
    }

    /// The rebuilds since this was last asked, oldest first.
    pub fn take_rebuilds(&mut self) -> Vec<RebuildCost> {
        std::mem::take(&mut self.rebuilds)
    }

    /// What the GPU has finished since this was last asked. Never waits.
    ///
    /// Asked every frame by the application whether or not it is logging, because the draft
    /// tier sizes itself from what the last one cost — so the costs have to be collected for
    /// the number to exist, not only when someone is reading the log.
    pub fn take_reports(&mut self) -> Vec<PerfReport> {
        let Some(shading) = &mut self.shading else { return Vec::new() };
        let mut out = Vec::new();
        for (ticket, cost) in shading.costs() {
            let Some(i) = self.submitted.iter().position(|s| s.ticket == ticket) else {
                continue;
            };
            let done = self.submitted.remove(i);
            // A cost of nothing is a measurement that failed, not a band that was free, and
            // a band sized from it would be the whole frame.
            let measured = cost.ms() > 0.0;
            if done.warm {
                // Deliberately a measurement of nothing.
            } else if done.still {
                // The dearest band so far at this size, and only at this size.
                if measured {
                    let per_row = cost.ms() / done.rows.max(1) as f32;
                    self.still_ms_per_row = Some(match self.still_ms_per_row {
                        Some((w, h, was)) if w == done.width && h == done.height => {
                            (w, h, was.max(per_row))
                        }
                        _ => (done.width, done.height, per_row),
                    });
                }
            } else if measured {
                // The dearest band so far at this size, as the still does, and for the reason
                // in [`TracedView::draft_ms_per_row`].
                let per_row = cost.ms() / done.rows.max(1) as f32;
                self.draft_ms_per_row = Some(match self.draft_ms_per_row {
                    Some((w, h, was)) if w == done.width && h == done.height => {
                        (w, h, was.max(per_row))
                    }
                    _ => (done.width, done.height, per_row),
                });
                // The ladder and the still's floor are both told about a whole frame; see
                // [`TracedView::draft_sweep`] for why a band will not do.
                let (ms, rows) =
                    sweep_with(self.draft_sweep, done.width, done.height, cost.ms(), done.rows);
                if rows >= done.height {
                    let pixels = (done.width as f32) * (done.height as f32);
                    self.draft_ms_per_pixel = Some(ms / pixels.max(1.0));
                    (self.draft_rung, self.draft_fast_streak) =
                        next_rung(self.draft_rung, self.draft_fast_streak, ms);
                    self.draft_sweep = None;
                } else {
                    self.draft_sweep = Some((done.width, done.height, ms, rows));
                }
            }
            out.push(PerfReport {
                frame: done.frame,
                still: done.still,
                pixels: done.width.max(done.height),
                band: done.band,
                still_frame: done.still_frame,
                skipped: done.skipped,
                warm: done.warm,
                cost,
            });
        }
        // A submission that found every timing slot busy is never reported, so the list would
        // otherwise grow by one for each of those. Nothing waits this long.
        if self.submitted.len() > 16 {
            let over = self.submitted.len() - 16;
            self.submitted.drain(..over);
        }
        out
    }

    /// Whether this frame may ask `render_state`'s device for anything — letting go of
    /// everything a device that has gone made, first (loop spec S4).
    ///
    /// `lost` is the flag the device-lost callback raises. wgpu runs that callback on whichever
    /// thread made the call that found the device gone — a shader compile on a worker as
    /// readily as a submit in the frame — so this is the interface thread answering it for the
    /// picture, as `App::keep_the_document_if_the_gpu_was_lost` answers it for the document.
    /// Until a loss it costs one comparison of two handles.
    ///
    /// **Everything the view holds on the GPU is derived, so letting go of it costs only the
    /// rebuild it asks for**: the pipelines come from the document's graph, the uploads from
    /// the bakes, and the ordinary rebuild path builds both on whatever device the next frame
    /// has in hand. What is *not* let go of is the CPU side — the baked bands, the baked
    /// environments, the walked spines — because none of it is a function of a device, and it
    /// is what makes the rebuild a rebuild rather than a cold start.
    pub fn ready_to_draw_on(&mut self, render_state: &egui_wgpu::RenderState, lost: bool) -> bool {
        let may_ask = match what_a_frame_must_do(lost, self.made_on(&render_state.device)) {
            MustDo::CarryOn => true,
            MustDo::LetGo => {
                self.let_go_of_a_lost_device(render_state);
                !lost
            }
            MustDo::AskNothing => false,
        };
        // An empty viewport is indistinguishable from an empty document, so the reason is said
        // where every other reason for one is said -- and said in whichever state the loss
        // arrives in, since a loss during the first shader compile has nothing to let go of and
        // leaves the same unexplained emptiness. Where the work is stays the application's to
        // say, since the document is its and it says it once, before any of this
        // (`App::keep_the_document_if_the_gpu_was_lost`). Compared before it is written, so a
        // session that ends lost does not allocate the same sentence once a frame.
        if lost && self.error.as_deref() != Some(LOST_DEVICE_REASON) {
            self.error = Some(LOST_DEVICE_REASON.to_string());
        }
        may_ask
    }

    /// Which device made what the view is holding. See [`TracedView::built_on`].
    fn made_on(&self, device: &wgpu::Device) -> MadeOn {
        match &self.built_on {
            None => MadeOn::Nothing,
            Some(made) if made == device => MadeOn::ThisDevice,
            Some(_) => MadeOn::AnotherDevice,
        }
    }

    /// Lets go of every GPU resource the view holds, because the device that made them has
    /// gone, and leaves nothing of them on screen.
    ///
    /// **A stale image is the failure to avoid here, not a leak.** Dropping a resource of a
    /// dead device frees what the driver has already taken back, which costs nothing and says
    /// nothing; what a person sees is the picture, and a picture of a device that has gone is
    /// one that cannot be redrawn, cannot be added to, and must not be presented as the model.
    /// So the key a frame compares itself against goes too: left behind, the first frame on a
    /// new device would find the view unchanged and accumulate onto an image that no longer
    /// exists.
    ///
    /// **This view's own resources, and not every resource the viewport draws with.** The
    /// control cage's buffers and the preview outputs' live in egui's `callback_resources`
    /// (`CageRenderResources`, `PreviewResources`), are of the same device, and are remade only
    /// when a mesh changes — so polygon and overlay mode would still hand a new device what the
    /// old one made. Harmless while the device cannot be replaced at all, there being no second
    /// device for a session to meet, and the piece the other half of S4 owes.
    fn let_go_of_a_lost_device(&mut self, render_state: &egui_wgpu::RenderState) {
        // egui's handles first, while what they stand for is still named here. A handle is this
        // side's bookkeeping rather than a call on the device, so freeing one is safe with the
        // device gone -- and a handle left behind is one the painter would be given next frame
        // for a texture that is not there.
        {
            let mut egui_renderer = render_state.renderer.write();
            for (_, id) in self.display.drain(..) {
                egui_renderer.free_texture(&id);
            }
        }
        // The work in flight, before the things it would be installed into: a compile worker
        // holds a clone of the device that has gone, and a renderer built on one is built on
        // nothing. The bake is told to stop, as every other reason to abandon one does, so its
        // cores go back to the rebuild that follows.
        self.stop_baking();
        self.compiling = None;
        // No cost of theirs will ever arrive: a submission is reported on a poll of the device
        // it was given to.
        self.submitted.clear();
        self.renderer = None;
        self.program = None;
        self.kept_program = None;
        self.shading = None;
        self.environment = None;
        self.skip_key = None;
        self.built_on = None;
        // Nothing of the picture, and nothing that reads as already drawn.
        self.last_view = None;
        self.still_shown = false;
        self.accumulated = 0;
        self.still_cursor = 0;
        self.still_band = 0;
        self.draft_cursor = 0;
        self.draft_band = 0;
        self.draft_generation = None;
        self.draft_sweep = None;
        self.last_frame = None;
        self.last_gpu_work = None;
        self.warm_until = None;
        // **The band sizes go with the device that measured them**, and that is not tidiness: a
        // band is sized from the dearest row measured, and the loss this recovers from is most
        // often a card reset by one submission running past the driver's timeout (S2). Carrying
        // that number over would size the first submission on the next device from the one that
        // killed the last.
        self.still_ms_per_row = None;
        self.draft_ms_per_pixel = None;
        self.draft_ms_per_row = None;
        self.last_band_rows = None;
        self.last_draft_band_rows = None;
        // Rebuilt from the document at the settled quality, not a draft's: nothing else is
        // about to invalidate the view, so a draft here is a 32-cell bake left on screen for
        // the rest of the session.
        self.built = None;
        self.invalidate(Quality::Final);
    }

    /// Marks the compiled shader and baked cage as out of date, to be rebuilt at `quality`.
    ///
    /// A `Final` request always wins: asking for a draft while one is already pending keeps
    /// the pending draft, but a settle after a drag must not be downgraded by a stray draft
    /// request or the model would stay coarse until something else happened to rebuild it.
    /// **Deliberately does not stop the bake in flight.** Most invalidations change nothing a
    /// bake depends on — a section plane, a placement, a curve point, a parameter — and stopping
    /// on those threw away seconds of finished work and started again from nothing, so a scene
    /// touched every few seconds never left the 32-cell stand-in. The rebuild that follows knows
    /// every leaf's key, so it is what decides; see `rebuild`.
    pub fn invalidate(&mut self, quality: Quality) {
        self.stale = match (self.stale, quality) {
            (Some(Quality::Final), _) | (_, Quality::Final) => Some(Quality::Final),
            _ => Some(Quality::Draft),
        };
        self.last_invalidated = Some(Instant::now());
        // Whatever is out of date may be the document, which the kept program was generated
        // from -- see `program_for`.
        self.kept_program = None;
    }

    /// The program `scene` is drawn with, and whether it had to be generated for it.
    ///
    /// **Generated once for each time the document can have moved, and kept between.** A
    /// generated program is a function of the document, the graph and the divisors, and nothing
    /// else. A rebuild nothing has invalidated since the last one -- a settle, a settled band
    /// arriving, a frame while the first shader compiles -- is handed the same three, and
    /// generated the same program again. The document is held by the contract the view already
    /// rests on, that whatever changes it calls [`Self::invalidate`]: one that did not would
    /// never be redrawn at all. The graph and the divisors are what the caller hands over, and
    /// cost microseconds to compare, so they are compared rather than trusted.
    ///
    /// **Not kept across a value, and that is not caution.** It was the plan (performance plan
    /// V2, refactor plan step 3.6): skip generating when nothing structural changed. But a value
    /// does not reach the program through its own slot alone. Slots worked out from values at
    /// generation -- each body's cull extents and guard box, the divisors, frames resolved to
    /// numbers, a pattern's numbers, a loft's stations -- move with them while the source stands,
    /// and the only code that works them out is generation's own walk. A program kept across a
    /// value would draw with the old ones, which is missing material where the cull extents
    /// were. So a value invalidates, and generates, like everything else.
    fn program_for(
        &mut self,
        scene: &Scene<'_>,
    ) -> Result<(ShaderProgram, bool), fieldform_gpu::CodegenError> {
        if let Some(kept) = &self.kept_program
            && kept.graph == *scene.graph
            && kept.morph_divisors == scene.morph_divisors
        {
            return Ok((kept.program.clone(), false));
        }
        // A graph that cannot be generated leaves the kept program as it was: it is still the
        // program of the graph it was kept against.
        let program =
            fieldform_gpu::generate_with(scene.document, scene.graph, scene.morph_divisors)?;
        self.kept_program = Some(KeptProgram {
            graph: scene.graph.clone(),
            morph_divisors: scene.morph_divisors.to_vec(),
            program: program.clone(),
        });
        Ok((program, true))
    }

    /// The tessellated spine of each curve in `ids` that `curves` holds, as [`Scene::curves`]
    /// wants them, walked to turn at most `degrees` a joint -- the rung's angle, which a settled
    /// rebuild walks at, and a draft at [`DRAFT_SPINE_DEGREES`] where that is coarser -- with
    /// `floor_mm` as the finest tolerance. Which of the two the next rebuild is, only the view
    /// knows, so it is decided here.
    ///
    /// A curve its tier walked last time to the same angle and floor, and unchanged since, is
    /// handed back rather than walked again ([`KeptSpines`]). Here rather than in the caller
    /// because the view is what outlives one rebuild and the caller's gathering does not.
    pub fn spines(
        &mut self,
        curves: &std::collections::HashMap<CurveId, fieldform_scene::Curve>,
        ids: impl IntoIterator<Item = CurveId>,
        degrees: f64,
        floor_mm: f64,
    ) -> Spines {
        self.walk_spines(curves, ids, degrees, floor_mm).0
    }

    /// [`Self::spines`], and how many of them had to be walked now.
    fn walk_spines(
        &mut self,
        curves: &std::collections::HashMap<CurveId, fieldform_scene::Curve>,
        ids: impl IntoIterator<Item = CurveId>,
        degrees: f64,
        floor_mm: f64,
    ) -> (Spines, usize) {
        match self.pending() {
            Quality::Draft => {
                let draft = degrees.max(DRAFT_SPINE_DEGREES);
                self.draft_spines.spines(curves, ids, draft, floor_mm)
            }
            Quality::Final => self.kept_spines.spines(curves, ids, degrees, floor_mm),
        }
    }

    /// Sets how finely a settled rebuild samples the model, and asks for one if that changed
    /// anything it reads.
    pub fn set_fineness(&mut self, fineness: Fineness) {
        if self.fineness != fineness {
            self.fineness = fineness;
            self.invalidate(Quality::Final);
        }
    }

    /// How long until a draft should be replaced by a settled bake, if one should.
    ///
    /// The caller uses this to ask for a repaint then: nothing else would, because the
    /// settle is driven by time rather than by input, and an idle application draws no frames
    /// in which to notice.
    pub fn settle_due_in(&self) -> Option<Duration> {
        if self.built != Some(Quality::Draft) || self.stale.is_some() {
            return None;
        }
        let since = self.last_invalidated?.elapsed();
        Some(SETTLE_AFTER.saturating_sub(since))
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Whether a settled bake is still being computed, so the caller can keep asking for frames
    /// and say what is going on.
    pub fn is_baking(&self) -> bool {
        self.baking.is_some()
    }

    /// Tells any settled bake in flight to stop, and forgets it.
    fn stop_baking(&mut self) {
        if let Some(baking) = self.baking.take() {
            baking.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The last rebuild's parts, for a caller reporting a slow frame. Taken, so the same
    /// rebuild is not blamed for two frames.
    pub fn take_rebuild_parts(&mut self) -> Option<(Quality, RebuildCost)> {
        self.last_rebuild.take()
    }

    /// What the last renderer's own parts cost: the shader through the driver, the pipelines it
    /// became, the curve and profile packing, and the baked atlas.
    pub fn take_build_parts(&mut self) -> Option<fieldform_gpu::BuildCost> {
        self.last_build.take()
    }

    /// Each leaf the last rebuild had to make ready: what it is, what building its field cost,
    /// and what baking it cost.
    pub fn take_bake_parts(&mut self) -> Vec<(String, f32, f32)> {
        std::mem::take(&mut self.last_bakes)
    }

    /// Gives up what is on screen, because there is nothing to draw.
    ///
    /// **A picture is only honest while its graph is.** Hiding the last body leaves nothing to
    /// generate, and leaving the renderer in place drew the body that had just been hidden —
    /// beside a panel saying the part was empty. Any compile in flight goes too: it is for the
    /// shape that has just gone.
    fn nothing_to_draw(&mut self) {
        self.program = None;
        self.renderer = None;
        self.compiling = None;
    }

    /// Puts a finished renderer and the program it belongs to in place together.
    ///
    /// **Together, because they are one thing**: the parameter slots are read by position, so a
    /// renderer drawing with the wrong program's numbers is a body in the wrong place.
    /// Restores the draft tier's measurements if the shader just installed is the one they were
    /// measured on, and forgets them either way (item 105). Called wherever a program is
    /// installed, so no path can keep stale costs or drop good ones.
    fn hand_back_learned_costs(&mut self) {
        let Some(program) = self.program.as_ref() else { return };
        if let Some(kept) = costs_to_keep(self.learned.take(), &program.source) {
            self.draft_rung = kept.draft_rung;
            self.draft_fast_streak = kept.draft_fast_streak;
            self.draft_ms_per_pixel = kept.draft_ms_per_pixel;
            self.draft_ms_per_row = kept.draft_ms_per_row;
            self.last_draft_band_rows = kept.last_draft_band_rows;
            self.still_ms_per_row = kept.still_ms_per_row;
        }
    }

    fn install(
        &mut self,
        built: Result<Renderer, fieldform_gpu::RenderError>,
        program: fieldform_gpu::ShaderProgram,
        marching: Marching,
        took: Duration,
    ) {
        match built {
            Ok(r) => {
                self.scene_bounded = marching.scene_bounded;
                self.step_budget = marching.step_budget;
                self.value_scale = marching.value_scale;
                self.last_build = Some(r.build_cost());
                // Whichever device the worker that built it had, which is the device every
                // resource in hand is now of. See [`TracedView::built_on`].
                self.built_on = Some(r.device().clone());
                self.renderer = Some(r);
                self.program = Some(program);
                self.hand_back_learned_costs();
                let cost = RebuildCost {
                    generate_ms: 0.0,
                    prepare_ms: 0.0,
                    pipeline_ms: took.as_secs_f32() * 1e3,
                    release_ms: 0.0,
                    size: self.program.as_ref().map(|p| p.size).unwrap_or_default(),
                };
                self.last_rebuild = self.last_rebuild.map(|(q, _)| (q, cost)).or(Some((
                    Quality::Final,
                    cost,
                )));
                self.rebuilds.push(cost);
                if self.rebuilds.len() > 64 {
                    self.rebuilds.remove(0);
                }
            }
            Err(e) => {
                self.error = Some(format!("could not build the field renderer: {e}"));
                self.program = None;
            }
        }
    }

    /// Whether a shader is being compiled elsewhere, so the caller keeps asking for frames and
    /// can say what the wait is.
    pub fn is_compiling(&self) -> Option<Duration> {
        self.compiling.as_ref().map(|c| c.started.elapsed())
    }

    /// Whether the wait is worth saying out loud, once per compile.
    pub fn compile_worth_saying(&mut self) -> Option<Duration> {
        let c = self.compiling.as_mut()?;
        let waited = c.started.elapsed();
        (!c.announced && waited >= SAY_COMPILE_AFTER).then(|| {
            c.announced = true;
            waited
        })
    }

    /// Takes in a shader the driver has finished with, and draws with it.
    ///
    /// `true` when one arrived, which is when the caller wants another frame.
    pub fn collect_compile(&mut self) -> bool {
        let Some(mut compiling) = self.compiling.take() else { return false };
        match compiling.done.try_recv() {
            Ok(built) => {
                // **For the shape that was on screen when it started, which may not be this
                // one.** A rebuild that found nothing to draw — the last body hidden — leaves no
                // program, and installing a compile started before it put the hidden body back
                // on screen beside a panel correctly saying the part was empty.
                if !compile_is_current(compiling.serial, self.asked) {
                    return false;
                }
                let took = compiling.started.elapsed();
                let program = compiling.program.take().expect("kept until installed");
                // Built from the uploads it set off with; a rebuild adopted since brought newer
                // ones, and the same source makes its pipelines reusable -- the cheap path a
                // same-shader rebuild takes.
                let built = match (built, compiling.uploads.take()) {
                    (Ok(renderer), Some((bakes, spines, surfaces))) => {
                        Renderer::rebuilt_from(&renderer, &program, &bakes, &spines, &surfaces)
                    }
                    (built, _) => built,
                };
                self.install(built, program, compiling.marching, took);
                // A new renderer draws a new image, so nothing averaged of the old one holds.
                self.last_view = None;
                self.accumulated = 0;
                self.still_shown = false;
                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.compiling = Some(compiling);
                false
            }
            // The worker is gone without an answer, which can only be a panic in it. Said
            // rather than left as a viewport that never changes again.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.error = Some("the shader compile did not finish".to_string());
                true
            }
        }
    }

    /// Takes in whatever the settled bake has finished, and asks for the rebuild that will draw
    /// with it. `true` when something arrived, which is when the caller wants another frame.
    ///
    /// The rebuild it asks for is cheap by construction: every band it needs is now in hand, so
    /// it recompiles nothing and re-bakes nothing.
    pub fn collect_bakes(&mut self) -> bool {
        let Some(baking) = self.baking.take() else { return false };
        let mut arrived = Vec::new();
        let mut done = false;
        loop {
            match baking.arrivals.try_recv() {
                Ok(pair) => arrived.push(pair),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    done = true;
                    break;
                }
            }
        }
        let outstanding = baking.outstanding.saturating_sub(arrived.len());
        let settled = !arrived.is_empty();
        let mut pending = baking.pending.clone();
        for (key, band) in arrived {
            pending.retain(|k| *k != key);
            self.bakes.insert(key, band);
        }
        if !done && outstanding > 0 {
            self.baking = Some(Baking { outstanding, pending, ..baking });
        }
        // **Drawn with once they are all in.** A rebuild re-uploads every band the shader binds,
        // so asking for one per arrival costs an assembly's whole atlas per body — ten bodies at
        // 128 cells is some three gigabytes of texture traffic to show the same picture ten
        // times. They arrive in seconds of each other and the last one is what makes the view
        // settled, so the rebuild waits for it.
        let all_in = settled && self.baking.is_none();
        if all_in {
            self.stale = Some(Quality::Final);
        }
        all_in
    }

    /// Whether the shader and bake need rebuilding before the next draw.
    ///
    /// Asked by the caller so that gathering a [`Scene`] — which means cloning a cage and
    /// tessellating curves — only happens when it will be used, rather than every frame.
    pub fn is_stale(&self) -> bool {
        self.stale.is_some()
            // No renderer yet, and no failure explaining why not. **A rebuild that failed is
            // not stale until something changes**: it leaves no renderer behind, and asking
            // again every frame only fails again at the same cost -- 75 ms a frame on a
            // document whose extrude could not evaluate, which was a viewport that never
            // settled and a frame rate nothing could be done at. The failure is reported where
            // it is, and `invalidate` is what asks for another try, when there is a change to
            // try with.
            || (self.renderer.is_none() && self.error.is_none())
            // A draft that nothing has touched for a while is a gesture that has finished.
            || self.settle_due_in() == Some(Duration::ZERO)
    }

    /// Whether what is on screen is the finished picture: built at final quality, nothing stale,
    /// and every jittered frame of the still averaged in.
    ///
    /// **For a snapshot, which must not catch a still half way.** A still averages its frames one
    /// a frame, so the first of them is a visibly noisier image than the last; a caller writing a
    /// file wants the one the person would have waited for. The same question the viewport already
    /// answers for itself by stopping — asked out loud, so nothing has to guess at a frame count.
    pub fn is_settled(&self) -> bool {
        !self.is_stale()
            && self.built == Some(Quality::Final)
            // How many this still wants is on the key the last frame was drawn from, which is
            // also the only place that knows what  asked for.
            && self
                .last_view
                .as_ref()
                .is_some_and(|key| self.accumulated >= key.still_frames.clamp(1, STILL_FRAMES))
    }

    /// The finished picture, read back from the texture the shading pass wrote.
    ///
    /// `None` until there is one: before the first still, or where the renderer could not be
    /// built. Reads the same texture egui samples, so a file written from it is the picture that
    /// was on screen rather than a second rendering of the same scene.
    pub fn read_display(&self) -> Option<fieldform_gpu::DisplayImage> {
        let (renderer, shading) = (self.renderer.as_ref()?, self.shading.as_ref()?);
        renderer.read_display(shading).ok()
    }

    /// The quality the next rebuild should use.
    fn pending(&self) -> Quality {
        self.stale.unwrap_or(Quality::Final)
    }
}

#[cfg(test)]
mod tests;
