// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Running a generated field shader on a GPU.
//!
//! Everything here is plumbing around [`crate::codegen`]: create the buffers the prelude
//! declares, bake each cage into the atlas, dispatch one thread per pixel, read back normals
//! and depths. The interesting decisions were made in the shader; this is what feeds it.
//!
//! # What is rebuilt, and when
//!
//! Three tiers, and keeping them apart is the whole reason for the exercise:
//!
//! | Change | Cost |
//! |---|---|
//! | A parameter value | write a few bytes to a buffer |
//! | A cage edit | rebake that one cage, reupload the atlas |
//! | The graph's shape | regenerate and recompile the shader |
//!
//! That mirrors the two-layer model exactly. Above the line — parameters, blends, booleans,
//! curve radii — nothing is rebuilt at all. Below the line, where editing is destructive and
//! comparatively rare, one leaf is resampled.

use crate::bake::BakedField;
use crate::codegen::ShaderProgram;
use fieldform_field::ScalarField;
use nalgebra::{Point3, Vector3};
use wgpu::util::DeviceExt;

/// How many section planes a frame may cut with at once: the length of the camera's array.
pub const MAX_SECTIONS: usize = 8;

/// The section planes a frame is cut by: each a unit normal and an offset along it, and the kept
/// region is where `dot(p, n) - offset <= 0` holds for *every* one — the intersection of their
/// half-spaces, so two planes make a corner cutaway. Empty is no section.
///
/// A fixed array rather than a `Vec`, because the camera is `Copy` and the shader's uniform is a
/// fixed array anyway; `MAX_SECTIONS` is the one number both read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sections {
    planes: [(Vector3<f64>, f64); MAX_SECTIONS],
    count: usize,
}

impl Sections {
    /// No section: nothing is cut.
    pub fn none() -> Self {
        Self { planes: [(Vector3::zeros(), 0.0); MAX_SECTIONS], count: 0 }
    }

    /// The first `MAX_SECTIONS` of these planes. Any past that are not cut with; a caller that
    /// wants to say so checks its own count against `MAX_SECTIONS` before it gets here.
    pub fn of(planes: &[(Vector3<f64>, f64)]) -> Self {
        let mut out = Self::none();
        for plane in planes.iter().take(MAX_SECTIONS) {
            out.planes[out.count] = *plane;
            out.count += 1;
        }
        out
    }

    pub fn planes(&self) -> &[(Vector3<f64>, f64)] {
        &self.planes[..self.count]
    }
}

/// Where the camera is and what it can see.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub eye: Point3<f64>,
    pub forward: Vector3<f64>,
    pub right: Vector3<f64>,
    pub up: Vector3<f64>,
    /// Half the image plane's height: at unit distance under perspective, and in world units
    /// under orthographic, where it does not vary with distance.
    pub half_height: f64,
    /// Whether rays are parallel rather than diverging from `eye`.
    ///
    /// Orthographic keeps parallel edges parallel, which is what makes a face-on view
    /// measurable — the whole reason a mechanical tool offers it.
    pub orthographic: bool,
    pub aspect: f64,
    /// How far a ray travels before giving up.
    pub far: f64,
    /// How close counts as a hit.
    pub epsilon: f64,
    /// The half-spaces to keep, each as a unit normal and an offset along it; none for no
    /// section. Everything where `dot(p, n) - offset > 0` for *any* of them is cut away.
    ///
    /// On the camera rather than on the field because a section is a way of looking at the
    /// part: the graph, the save and the export are all unaffected, and two views of one
    /// document can cut it differently without either being wrong.
    pub sections: Sections,
    /// Ambient occlusion and the key light's shadow, or `None` for neither this frame.
    ///
    /// On the camera because it is a per-frame choice about *this* trace rather than a fact
    /// about the scene: a drag turns both off to spend its budget on keeping up, and the still
    /// that follows turns them on. The lighting pass reads what the march measured.
    pub occlusion: Option<Occlusion>,
    /// Sub-pixel offset of every ray, in pixels. Zero for a single frame; a low-discrepancy
    /// sequence while a still accumulates, which is what anti-aliases traced silhouettes.
    pub jitter: [f64; 2],
    /// Whether a ray, once it has a candidate, finishes on a profile's outline rather than its
    /// bake (`exact_reads` in `prelude.wgsl`, backlog item 115): the bake's margin draws every
    /// baked body fat and a knife edge as a chamfer, and this is what takes it back to the
    /// outline for the frames somebody looks at. Off by default and off for the draft, since
    /// it costs a few walks of the outline per hit pixel; the still tier turns it on.
    pub exact_finish: bool,
    /// What to multiply a field reading by before treating it as a distance, for occlusion and
    /// shadow only.
    ///
    /// **Because those two read the value as "how far away is anything", and a divided field is
    /// no longer that.** A driven morph or offset divides its value to keep it a lower bound;
    /// the number stays honest, so the march is safe, but it stops estimating distance. Ambient
    /// occlusion accumulates how much less clearance it found than it travelled and the shadow
    /// keeps the smallest ratio of clearance to distance — so every sample reading short makes
    /// the body shade as though it were at the bottom of a hole. Reported as a model going from
    /// light to dark the moment a morph's amount started varying.
    ///
    /// `fieldform-field/examples/driven_morph_shading.rs` measures it: on the reported scene the
    /// occlusion at the pole reads **0.000** against the constant version's 1.000, and putting
    /// the divisor back reads 1.000 again.
    ///
    /// One number for the whole graph, because the shader has no way to ask which node defined
    /// the value at a point: `fieldform_field::value_scale_with`, which is the budget's
    /// `step_multiplier` but for a sweep read in its tube -- that one reads its walls truly and
    /// falls short only inside its caps, where the shading never samples. It is therefore too
    /// large for a body in the scene that was never divided, which reads as *less* occlusion
    /// than it deserves — a mild artefact against a black one, and the same global-for-local
    /// trade the divisor itself makes.
    ///
    /// One by default: a field that reports true distance needs no correction.
    pub value_scale: f64,
    /// The sphere the scene was said to lie in, as [`Self::looking_at`] was told it.
    pub scene_centre: Point3<f64>,
    pub scene_radius: f64,
    /// Whether the primary ray may start where it enters that sphere rather than at the eye.
    ///
    /// Exact when the sphere encloses everything drawn, and off by default because that is a
    /// promise only the caller can make: the application turns it on when every body has a
    /// finite extent, and leaves it off for a scene with a plane in it. The examples leave it
    /// off, so `gpu_matches_cpu` measures the march from the eye as it always has.
    pub start_at_scene: bool,
    /// Whether this program's cull extents still describe the document being drawn.
    ///
    /// **On by default, and the one caller that must turn it off is the application mid-compile**
    /// (`tile-mask-design-01941c5.md` §1.6, review point 5). A cull extent is a *literal* in the
    /// parameter buffer, so it is fixed when the program is generated -- where a named value is
    /// resolved from the document on every `set_params`. While a new shader compiles, which was
    /// measured at 1.8 s for a hundred bodies and seen at 29, the old program keeps drawing with
    /// current numbers and *old* extents: a body the user is dragging larger would be culled out
    /// of the tiles it has grown into, which is missing material.
    ///
    /// Off by default would have been the other way to write it, and is worse: every example and
    /// the readback path build their program from the document they then draw, so they are never
    /// stale, and defaulting off would leave the cull untested by `gpu_matches_cpu` -- which is
    /// the only thing that checks it against the CPU's own field.
    pub cull: bool,
}

/// What the march measures beyond the surface: how open it is, and how much of one light
/// reaches it. Both scale with the model, because a 10 mm part and a metre one occlude alike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Occlusion {
    /// Unit direction toward the key light, which casts the shadow.
    pub light: Vector3<f64>,
    /// How far out along the normal ambient occlusion looks.
    pub radius: f64,
    /// How far toward the light the shadow march goes before giving up.
    pub shadow_reach: f64,
    /// How sharp the shadow's edge is; larger is sharper.
    pub softness: f64,
    /// Whether both are marched for one pixel in each two-by-two block, with the lighting pass
    /// filling in the rest where it can -- or for every pixel.
    ///
    /// **Every pixel is how the viewport draws a still, and what [`Self::for_scene`] gives.** One
    /// in four was built for loop spec P5 and is off until the owner rules on it, because on the
    /// owner's documents it is cheaper only where it can be seen. Filling every other pixel from
    /// its neighbours took 7-30 % off a still frame, and moved the beam lattices further from
    /// every pixel's picture than a still stopped at four frames is from one run to eight.
    /// Filling only where two neighbours on the pixel's own surface agree, and marching the rest
    /// -- which is what this does -- keeps every one of them within a quarter of a percent of its
    /// pixels of every pixel's picture, but leaves a pixel to march in nearly every wave that
    /// touches the model, and a wave pays for its busiest lane: from 10 % cheaper to 8 % dearer
    /// than every pixel. The loop spec's P5 has both tables, and
    /// `fieldform-io/examples/still_cost.rs` measures them.
    pub half_resolution: bool,
}

impl Occlusion {
    /// Sized to a scene of `radius`: occlusion reaches a few percent of the model, the shadow
    /// march crosses all of it.
    pub fn for_scene(light: Vector3<f64>, radius: f64) -> Self {
        let radius = radius.max(1e-6);
        Self {
            light: light.try_normalize(1e-12).unwrap_or_else(Vector3::z),
            radius: Self::reach(radius),
            shadow_reach: radius * 3.0,
            softness: 12.0,
            half_resolution: false,
        }
    }

    /// How far off every surface the occlusion reads, for a scene of `radius`.
    ///
    /// **Said on its own because a bake has to know it.** A baked profile keeps fine samples
    /// only in a band about its boundary and answers from a coarse level beyond, with a margin
    /// eight times wider -- a step down in the reading at the band's edge. The occlusion's five
    /// samples go this far out along the normal, and the band as the field alone wants it
    /// stops a third of the way: the outer samples read the coarse level at some heights and
    /// the fine one at others, by where the surface happens to sit in the block grid, and a
    /// surface open everywhere came out striped at the block's period, 0.75 to 1.00 in three
    /// flat levels where the exact field reads 1.00 throughout --
    /// `fieldform-field/examples/bake_normal_bands.rs`. So whatever bakes for the display
    /// widens its band by this (`fieldform_field::reads_widened`).
    pub fn reach(scene_radius: f64) -> f64 {
        scene_radius.max(1e-6) * 0.06
    }
}

impl Camera {
    /// A three-quarter view framing a bounding box, matching how the CPU examples frame theirs
    /// so images from the two paths can be compared directly.
    pub fn framing(bounds: (Point3<f64>, Point3<f64>), aspect: f64) -> Self {
        let centre = Point3::from((bounds.0.coords + bounds.1.coords) * 0.5);
        let radius = (bounds.1 - bounds.0).norm() * 0.5;
        let half_height = (45f64.to_radians() * 0.5).tan();
        let dir = Vector3::new(1.0, 0.7, 0.55).normalize();
        let eye = centre + dir * (radius / half_height) * 1.05;
        Self::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, aspect)
    }

    /// A camera placed where an application's own camera is.
    ///
    /// `up_hint` must be the caller's **own** up vector, not a world axis. An orbit camera that
    /// supports twist carries an up vector that has rolled away from world up, and re-deriving
    /// one here would silently disagree with whatever else the application draws — overlays,
    /// gizmos, curves — putting them in a different place from the geometry.
    ///
    /// It is also the fix for a discontinuity. Deriving `right` as `forward x world_up` fails
    /// when the camera looks straight down, and any fallback to a different reference axis makes
    /// the basis *jump* as the view crosses that threshold. Taking the caller's up removes the
    /// degeneracy rather than papering over it, because a camera's own up is never parallel to
    /// its own forward.
    ///
    /// This mirrors `Matrix4::look_at_rh` exactly: `right = forward x up`, then up is
    /// re-squared as `right x forward`.
    ///
    /// `scene_radius` is only used to size the far plane and the hit threshold, both of which
    /// have to scale with the model: an absolute epsilon is either too coarse for a 1 mm part
    /// or so fine that a 10 m one never converges, and the viewport is meant to be scale-free.
    ///
    /// A camera looking from `eye` at `target`, marching far enough to cross `scene`.
    ///
    /// The scene is given as its own centre and radius rather than inferred from the target,
    /// because the target is wherever the user last pivoted — which may be a corner of the
    /// model, or a point in mid-air. A ray budget measured from there stops short of geometry
    /// that is plainly on screen.
    #[allow(clippy::too_many_arguments)]
    pub fn looking_at(
        eye: Point3<f64>,
        target: Point3<f64>,
        up_hint: Vector3<f64>,
        scene: (Point3<f64>, f64),
        fov_deg: f64,
        aspect: f64,
    ) -> Self {
        let (scene_centre, scene_radius) = scene;
        let forward = (target - eye).try_normalize(1e-12).unwrap_or_else(Vector3::x);
        // Only if the caller handed us an up parallel to forward, which an orbit camera cannot
        // produce. Picking any perpendicular is as good an answer as exists.
        let right = forward
            .cross(&up_hint)
            .try_normalize(1e-9)
            .unwrap_or_else(|| {
                let axis = if forward.x.abs() < 0.9 {
                    Vector3::x()
                } else {
                    Vector3::y()
                };
                forward.cross(&axis).normalize()
            });
        let up = right.cross(&forward);
        let radius = scene_radius.max(1e-9);

        Self {
            eye,
            forward,
            right,
            up,
            half_height: (fov_deg.to_radians() * 0.5).tan(),
            // Perspective by default: the callers that build a camera this way are the
            // comparison examples, which frame a scene the way a person looks at one.
            orthographic: false,
            aspect,
            far: (eye - scene_centre).norm() + radius * 2.0,
            // A hundred-thousandth of the model, and the factor is measured rather than
            // chosen. A ray stops when the field reads below this, so **the hit threshold is
            // the surface error** — not a floor the tracer converges to, which is what this
            // number was described as for a while. `gpu_matches_cpu`'s sweep
            // (`FIELDFORM_EPSILON_SWEEP=1`) shows them equal to four figures across two orders
            // of magnitude, and on a field that reports true distance the step budget makes no
            // difference at all: 512 steps and 4096 give byte-identical results.
            //
            // **That last clause held only while every field reported true distance.** A driven
            // morph or offset divides its value to stay a lower bound, and a tracer steps by
            // what it reads, so the same divisor shortens every step and a fixed budget covers
            // that much less of the ray -- which is a hole in the silhouette, not a slower
            // frame. `fieldform-gpu/examples/driven_morph_budget.rs` measures both halves: the
            // true-distance scene really is identical from 512 steps to 16384 and flat at
            // 5.8 ms, while a morph at x8.5 loses 2580 pixels of 228345 at 512 and none at the
            // budget `traced/rebuild.rs` now picks, for the same 11 ms. The budget lives there
            // because it depends on the graph; the epsilon here does not.
            //
            // At `1e-3` a gyroid-filled sphere measured 1.299 mm of error, which is a third of
            // a 3 mm wall and showed as seams and banding on the surface. At `1e-5` the same
            // scene measures 0.013 mm and renders in 2.1 ms against 2.4 — *faster*, because a
            // ray that converges decisively stops sooner than one that dithers.
            //
            // Scenes with a cage improve from 1.429 mm to 0.210 mm and cost 20% more (11.2 ms
            // to 13.4). They stop at 0.210 because that is the baked narrow band's own grid,
            // not the tracer — a separate lever, and the one View > Display quality moves.
            //
            // Not finer than this: at `1e-6` the threshold approaches what `f32` can resolve at
            // the far end of a large model, and the trace would start measuring its own
            // arithmetic.
            epsilon: radius * 1e-5,
            // One: the callers that build a camera this way are the comparison examples, and
            // a scene whose field is divided says so by setting this itself.
            value_scale: 1.0,
            // Sectioning is something a viewer turns on, never a default.
            sections: Sections::none(),
            // Likewise occlusion: a caller that wants a still asks for one.
            occlusion: None,
            jitter: [0.0, 0.0],
            exact_finish: false,
            scene_centre,
            scene_radius: radius,
            start_at_scene: false,
            // The common case: a program generated from the document it is about to draw.
            cull: true,
        }
    }
}

/// The `Camera` block in `camera.wgsl`, field for field.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    lens: [f32; 4],
    frame: [u32; 4],
    sections: [[f32; 4]; MAX_SECTIONS],
    key_light: [f32; 4],
    occlusion: [f32; 4],
    jitter: [f32; 4],
    band: [u32; 4],
    scene: [f32; 4],
}

/// The camera as the shaders read it. Shared by the readback path and the presented one, so
/// the two cannot disagree about which pixel a ray goes through.
///
/// `rows` is the band of the frame this dispatch covers, as its first row and its count — the
/// whole frame is `(0, height)`. See `Camera.band` in `camera.wgsl` for why a frame is bands.
///
/// `culled` is whether *this* submission recorded the per-tile cull into its own encoder, for
/// this camera and this size. **It is a promise about the submission, not about the program**
/// (`tile-mask-design-01941c5.md` §1.6, review point 4): the mask buffer outlives frames, so a
/// warm-up, a readback or a frame drawn while a new shader compiles would otherwise read the bits
/// the last `present` left -- a different camera, possibly a different width. False keeps the
/// polarity rule, since a mask nothing reads rules nothing out.
///
/// `measured` is which pixels a still marches its occlusion and shadow for, from
/// [`occlusion_pixels`]; nought for every pixel, which is what anything without a still frame
/// to draw passes.
pub(crate) fn camera_uniform(
    camera: &Camera,
    width: u32,
    height: u32,
    max_steps: u32,
    rows: (u32, u32),
    culled: bool,
    measured: u32,
) -> CameraUniform {
    CameraUniform {
        eye: [
            camera.eye.x as f32,
            camera.eye.y as f32,
            camera.eye.z as f32,
            0.0,
        ],
        forward: vec4(camera.forward),
        right: vec4(camera.right),
        up: vec4(camera.up),
        lens: [
            camera.half_height as f32,
            camera.aspect as f32,
            camera.far as f32,
            camera.epsilon as f32,
        ],
        frame: [width, height, max_steps, u32::from(camera.orthographic)],
        sections: {
            // Packed from the front: a zero normal is how the shader reads "no section", and
            // the first one ends its loop. See `section_distance`.
            let mut packed = [[0.0; 4]; MAX_SECTIONS];
            for (slot, (n, offset)) in packed.iter_mut().zip(camera.sections.planes()) {
                *slot = [n.x as f32, n.y as f32, n.z as f32, *offset as f32];
            }
            packed
        },
        key_light: match camera.occlusion {
            Some(o) => [o.light.x as f32, o.light.y as f32, o.light.z as f32, 1.0],
            None => [0.0, 0.0, 1.0, 0.0],
        },
        occlusion: match camera.occlusion {
            Some(o) => [
                o.radius as f32,
                o.shadow_reach as f32,
                o.softness as f32,
                camera.value_scale as f32,
            ],
            // The scale is written in both arms rather than left at zero with the rest: a
            // frame with no occlusion still has a `w`, and a zero there would be a trap for
            // whatever reads it next.
            None => [0.0, 0.0, 0.0, camera.value_scale as f32],
        },
        jitter: [camera.jitter[0] as f32, camera.jitter[1] as f32, f32::from(camera.exact_finish), 0.0],
        band: [rows.0, rows.1, u32::from(culled && camera.cull), measured],
        scene: [
            camera.scene_centre.x as f32,
            camera.scene_centre.y as f32,
            camera.scene_centre.z as f32,
            // Zero is how the shader reads "start at the eye" -- see `Camera.scene`.
            if camera.start_at_scene { camera.scene_radius as f32 } else { 0.0 },
        ],
    }
}

/// The pixel of each two-by-two block a still measures, by the frame it is in the still: a
/// corner, then the one diagonally across, then the other two.
///
/// **It moves so that the average holds every pixel's own reading.** The lighting pass fills a
/// pixel from the measured ones about it on its own surface where two or more agree, and
/// `occlude` marches the rest; what a fill still misses is a reading that varies within the two
/// pixels between its neighbours. Moved round the block, every pixel is measured itself in one
/// frame of each four, so the accumulation holds some of each pixel's own reading there too; and
/// diagonal first, so that two frames already measure the block evenly.
const MEASURED_CORNERS: [(u32, u32); 4] = [(0, 0), (1, 1), (1, 0), (0, 1)];

/// Which pixels the still's occlusion and shadow are marched for, as `Camera.band.w` carries
/// them to both passes (`occlusion_pixels` in `camera.wgsl`): nought for every pixel, otherwise
/// bit 2 and the measured corner in bits 0 and 1. `frame` is how many frames of this still are
/// already accumulated, which is what moves the corner.
pub(crate) fn occlusion_pixels(camera: &Camera, frame: u32) -> u32 {
    match camera.occlusion {
        Some(o) if o.half_resolution => {
            let (x, y) = MEASURED_CORNERS[(frame % 4) as usize];
            4 | x | (y << 1)
        }
        _ => 0,
    }
}

/// How many workgroups of `occlude` cover the measured pixels of a band `rows` high.
///
/// Rounded up on the band's own rows rather than on its measured ones: a band that does not
/// start on a measured row has one fewer of them, never one more, and the shader drops the
/// threads past the band.
pub(crate) fn occlusion_workgroups(width: u32, rows: u32, measured: u32) -> (u32, u32) {
    let stride = if measured & 4 != 0 { 2 } else { 1 };
    (width.div_ceil(stride).div_ceil(8), rows.div_ceil(stride).div_ceil(8))
}

/// How many workgroups of `occlude`'s second dispatch at one in four cover the blocks of a band
/// `rows` high, each deciding the three pixels of its block that were not measured: three deep,
/// one slice for each.
///
/// A block answers for the pixels left of its measured one and above it, so the image's last row
/// and column, where they are unmeasured, need a block one past the edge: hence half the width
/// and half the rows, plus one, rather than rounded up. That is never too few, and the shader
/// drops the threads past the band and past that one block.
pub(crate) fn unmeasured_workgroups(width: u32, rows: u32) -> (u32, u32, u32) {
    ((width / 2 + 1).div_ceil(8), (rows / 2 + 1).div_ceil(8), 3)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CurveSpan {
    start: u32,
    count: u32,
    /// Where this profile's baked samples begin, and how big the grid is.
    ///
    /// `bake_wide` of zero means there is none, and the shader walks the boundary instead. Every
    /// field is a four-byte scalar so the layout needs no thought about alignment — a mismatch
    /// here reads the wrong numbers rather than failing, which is the worst kind.
    coarse_start: u32,
    page_start: u32,
    fine_start: u32,
    blocks_wide: u32,
    blocks_high: u32,
    bake_wide: u32,
    bake_high: u32,
    lo_x: f32,
    lo_y: f32,
    cell: f32,
    margin: f32,
    coarse_margin: f32,
    /// The segment tree over this curve, as node pairs in `curve_points`. See
    /// [`segment_tree`]; zero nodes and the shader walks the segments one by one.
    bvh_start: u32,
    bvh_count: u32,
    /// The bake's segment lists (`BakedProfile::index_segments`): per block cell a (start, count)
    /// pair from `segs_start`, the lists themselves from `seglist_start`, both as `f32` in the
    /// bake buffer like the page, and whether the outline turns anticlockwise.
    segs_start: u32,
    seglist_start: u32,
    ccw: u32,
}

/// How many segments a leaf of the segment tree holds, at most. Eight, the top of the plan's
/// range: a leaf's tests are the cheap, coherent part of the walk and a node is the dear
/// part, so leaves are as full as the packing below allows.
const SEGMENT_LEAF: u32 = 8;

/// A tree over a polyline's segments, as node pairs for `curve_points`, given the polyline's
/// points as they are uploaded.
///
/// **Built over runs of consecutive segments rather than by sorting**, because a curve's
/// segments are already ordered along it: a run is a stretch of curve, its box is compact, and
/// the split is the middle of the run. **Laid out depth first, for a walk with no stack**: each
/// node is two entries — the box's low corner with, in `w`, the index of the node after its
/// whole subtree, so a walk that rules the box out jumps straight past it; then the high corner
/// with, in `w`, minus one for an inner node or, for a leaf, its first segment times sixteen
/// plus its count. Whole numbers stored as floats, exact well past any curve; nothing here is a
/// NaN payload for a driver to canonicalise. Indices are relative to the tree's own first node.
/// Empty for fewer than two segments, which the shader walks.
fn segment_tree(points: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let segments = points.len().saturating_sub(1) as u32;
    if segments < 2 {
        return Vec::new();
    }
    fn bounds(points: &[[f32; 4]], first: u32, count: u32) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for p in &points[first as usize..=(first + count) as usize] {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo, hi)
    }
    fn build(nodes: &mut Vec<[f32; 4]>, points: &[[f32; 4]], first: u32, count: u32) {
        let index = nodes.len();
        nodes.push([0.0; 4]);
        nodes.push([0.0; 4]);
        let (lo, hi) = bounds(points, first, count);
        let leaf = if count <= SEGMENT_LEAF {
            (first * 16 + count) as f32
        } else {
            let half = count / 2;
            build(nodes, points, first, half);
            build(nodes, points, first + half, count - half);
            -1.0
        };
        let skip = (nodes.len() / 2) as f32;
        nodes[index] = [lo[0], lo[1], lo[2], skip];
        nodes[index + 1] = [hi[0], hi[1], hi[2], leaf];
    }
    let mut nodes = Vec::with_capacity(4 * segments.div_ceil(SEGMENT_LEAF) as usize);
    build(&mut nodes, points, 0, segments);
    nodes
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CageSlab {
    lo: [f32; 4],
    inv_cell: [f32; 4],
    dims: [u32; 4],
    layer_offset: u32,
    band: f32,
    _pad: [f32; 2],
}

#[derive(thiserror::Error, Debug)]
pub enum RenderError {
    #[error("no GPU adapter was available")]
    NoAdapter,
    #[error("could not create a GPU device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("the generated shader was rejected by the driver: {0}")]
    Shader(String),
    /// A pipeline the driver would not make, with the graph's size so the message says what was
    /// too big rather than only that something was.
    ///
    /// **Its own variant rather than [`RenderError::Shader`]** because the two have different
    /// causes and different answers. `Shader` is our own front end refusing source it generated,
    /// which is this build's fault and a bug to fix. This is the driver refusing source that is
    /// valid WGSL, which is a limit to work within — and the numbers are what tell a person
    /// whether to hide some bodies or simplify the graph.
    #[error(
        "the driver would not compile this part's shader ({size}) — hide some bodies or simplify \
         the graph, and see the console log for what it said. Driver: {detail}"
    )]
    ShaderRefused {
        size: crate::codegen::ShaderSize,
        detail: String,
    },
    #[error("reading the rendered image back failed")]
    Readback,
    #[error(
        "the baked bodies need a {width}x{height}x{depth} texture and this GPU allows {limit}          in each direction — choose a lower View > Display quality, or hide some bodies"
    )]
    AtlasTooLarge {
        width: usize,
        height: usize,
        depth: usize,
        limit: usize,
    },
}

/// What the `fields` group binds, as data rather than inline, so that a test can hold
/// `STORAGE_BUFFERS_NEEDED` -- the number the device is asked for -- to what is actually bound.
fn fields_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    vec![
        storage(0, true),
        storage(1, true),
        storage(2, true),
        storage(3, true),
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage(5, false),
        storage(6, false),
        storage(7, true),
        storage(8, false),
        storage(9, true),
        storage(10, true),
        storage(11, true),
    ]
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    /// `STORAGE_BUFFERS_NEEDED` is what the device is asked for and `fields_layout_entries` is
    /// what is bound. A binding added to the layout and not to the number is the abort this
    /// arrangement exists to prevent -- eight were bound and eight allowed until the surface's
    /// three arrived -- so the two are held together here rather than beside each other.
    #[test]
    fn the_device_is_asked_for_exactly_the_storage_buffers_the_field_layout_binds() {
        let entries = fields_layout_entries();
        let bound = entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.ty,
                    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { .. }, .. }
                )
            })
            .count() as u32;
        assert_eq!(bound, STORAGE_BUFFERS_NEEDED);
        assert_eq!(entries.len(), FIELD_BINDINGS, "the layout and the binding list agree");
        // Contiguous from nought, since the bind group supplies them by position.
        let mut bindings: Vec<u32> = entries.iter().map(|e| e.binding).collect();
        bindings.sort_unstable();
        assert_eq!(bindings, (0..bindings.len() as u32).collect::<Vec<_>>());
    }
}

/// The most cells a bake may use when `leaves` of them share one atlas.
///
/// Every baked leaf becomes a stack of layers in **one** 3D texture, so their depths add up
/// while their widths and heights only take a maximum. A GPU caps a 3D texture at 2048 in each
/// direction on this hardware, and wgpu treats exceeding it as fatal — so an assembly of a dozen
/// solids at 256 cells crashed the application outright, which is what an imported STEP assembly
/// is. Reported: "Dimension Z value 2203 exceeds the limit of 2048".
///
/// The resolution is the longest edge, so a leaf's depth is at most that: dividing the limit by
/// the number of leaves is sufficient rather than merely likely to fit. A floor of 8 keeps a
/// hundred-solid assembly coarse rather than degenerate.
pub fn bake_resolution_limit(device: &wgpu::Device, leaves: usize, wanted: usize) -> usize {
    let limit = device.limits().max_texture_dimension_3d as usize;
    wanted.min((limit / leaves.max(1)).max(8))
}

/// One pixel of a render: the surface normal, and how far along the ray it was found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub normal: Vector3<f64>,
    /// Negative where the ray hit nothing.
    pub depth: f64,
    /// Which of the part's bodies owns the surface here, by position in the graph's roots —
    /// `u32::MAX` where the ray hit nothing.
    ///
    /// The union fold already finds the nearest body to return its value; this is that answer
    /// kept rather than discarded, which is what lets a selected body be tinted without a
    /// second trace or a shader that knows about selection.
    pub body: u32,
}

impl Hit {
    pub fn is_hit(&self) -> bool {
        self.depth >= 0.0
    }
}

/// One surface's offsets, as the shader reads them — the same numbers `SurfaceLayout` carries,
/// shifted by where that surface's run begins in the shared buffers.
///
/// Field for field with `SurfaceLayout` in `prelude.wgsl`, in order: every member is four bytes
/// so the struct is 40 with no padding either side, and a mismatch would not fail to compile —
/// it would read one array's offsets as another's.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSurfaceLayout {
    positions: u32,
    normals: u32,
    uvs: u32,
    gradients: u32,
    node_bounds: u32,
    triangles: u32,
    faces: u32,
    node_links: u32,
    order: u32,
    triangle_count: u32,
    /// Not an offset: the surface's own worst parametric gradient, which the shader needs to
    /// bound its stepping and which would otherwise want a buffer of its own for one float.
    worst_gradient: f32,
}

/// How many storage buffers the field shader binds, and so what a device has to allow.
///
/// **Eleven, against a default of eight.** Parameters, curve points, curve spans, cage slabs,
/// the three G-buffer outputs, profile bakes, and a conformal surface's three arrays. The
/// budget was exactly full before surfaces arrived, which is worth saying plainly: the next
/// thing that wants a buffer of its own will have to share one.
///
/// wgpu's default limits are the WebGPU baseline, which every desktop adapter exceeds by orders
/// of magnitude — this machine's reports far more than eleven. Asking for more than the
/// baseline does narrow what could ever run this: a browser would refuse. That is a trade
/// already made elsewhere in this crate, which requires compute shaders at all.
pub(crate) const STORAGE_BUFFERS_NEEDED: u32 = 11;

/// What the device is asked for: what the field layout binds, and under `march-stats` the five
/// buffers the counting march and the grid evaluator use, which live in a group of their own so the
/// field layout — and the test that pins `STORAGE_BUFFERS_NEEDED` to it — are untouched.
fn storage_buffers_wanted() -> u32 {
    // And one for the skip grid, which rides in the atlas's group for the same reason.
    #[cfg(feature = "march-stats")]
    {
        STORAGE_BUFFERS_NEEDED + 1 + 5
    }
    #[cfg(not(feature = "march-stats"))]
    {
        STORAGE_BUFFERS_NEEDED + 1
    }
}

/// How far past a reading the primary march steps, as a multiple of the reading (plan §5 V8;
/// `march.wgsl` has the argument for why a long step is safe once it has been vouched for).
///
/// Measured by `fieldform-io/examples/framed_capture.rs`, a 720 px draft's march on the card's
/// own clock, over the 64³ grid with its masks:
///
/// | | plain | 1.2 | 1.4 | 1.6 |
/// |---|---|---|---|---|
/// | patterned pipe | 15.6 ms | 12.6 | 11.2 | 10.8 |
/// | attractor-infill (a lattice) | 5.7 ms | 4.6 | 4.6 | 5.0 |
/// | fcc-cubic (beams) | 4.2 ms | 3.4 | 3.4 | 3.3 |
/// | eight pipes chained by blends | 60.2 ms | 46.3 | 40.2 | 36.7 |
///
/// A field that reads short -- divided by a blend's or a driven radius's bound -- wants the
/// longest step, because its readings understate the room by that divisor and a long step is
/// seldom refused. A lattice is refused more often, and past 1.4 the withdrawn steps cost more
/// than the long ones save. 1.4 is at or beside the best in every row.
pub(crate) const RELAX: f32 = 1.4;

/// The most cells a side a skip grid may have, which is what its buffer is sized for.
pub(crate) const SKIP_GRID_MAX: u32 = 64;

/// How many tiles the per-tile cull's mask holds, and the widest frame it therefore serves.
///
/// 512 by 288 tiles of eight pixels is a frame of **4096 by 2304**, which covers a 4K viewport
/// full screen. Eight words a tile (`MASK_WORDS_MAX` in `march.wgsl`, which is `CULL_CAP / 32`)
/// at four bytes is 4.7 MB, on top of the skip grid's 3.1 MB in the same buffer. Allocated for
/// the cap rather than for the program, so a document that gains a body needs no reallocation in
/// the middle of a drag.
///
/// A frame larger than that **draws unculled** rather than wrapping: `load_tile_mask` refuses a
/// tile index past this, because an out-of-range read is not zero -- it is some other tile's
/// bits, and those would rule out a body that is on screen.
pub(crate) const TILE_MASK_TILES: u32 = (4096 / TILE_PIXELS) * (2304 / TILE_PIXELS);

/// Words a tile, `MASK_WORDS_MAX` in `march.wgsl`.
pub(crate) const TILE_MASK_WORDS: u32 = 8;

/// A tile's side in pixels, `TILE` in `march.wgsl` -- which is the primary march's own workgroup,
/// so a tile is a workgroup and the threads that share a mask word are the ones that load it.
const TILE_PIXELS: u32 = 8;

/// `tile_cull`'s workgroup, one invocation a tile.
const CULL_WORKGROUP: u32 = 64;

/// Floats ahead of the cells in the side tables' buffer. See `march.wgsl`.
const SKIP_HEADER: u64 = 8;

/// Where the per-tile body mask starts in the same buffer, in words: after the header and the
/// three planes of cells at the *largest* grid, so the offset does not move when the resolution
/// does. `TILE_MASK_BASE` in `march.wgsl`, which must say the same number.
pub(crate) const TILE_MASK_BASE: u64 = SKIP_HEADER + 3 * (SKIP_GRID_MAX as u64).pow(3);

/// One fill of the skip grid, read back. See [`Renderer::read_skip_grid`].
pub struct SkipGridReading {
    /// How far the march may step from each cell's centre before it must ask the field again.
    pub clearance: Vec<f32>,
    /// Which operands each cell ruled out, a bit each, indexed as `ShaderProgram::masked` is.
    pub ruled_out: Vec<u32>,
}

/// One frame's per-tile body mask, read back. See [`Renderer::read_tile_mask`].
pub struct TileMaskReading {
    /// Tiles across and down the frame it was recorded for.
    pub tiles: (u32, u32),
    /// Words a tile, as `tile_cull` strides them: `MASK_WORDS` of the program, which is the
    /// bodies in 32s -- not `MASK_WORDS_MAX`, which is only the buffer's capacity.
    pub words: u32,
    /// `tiles.0 * tiles.1 * words` words, row by row of tiles; a set bit is a body ruled out.
    pub bits: Vec<u32>,
}

impl TileMaskReading {
    /// Whether `tile_cull` ruled `body` out of `tile`.
    pub fn ruled_out(&self, tile: (u32, u32), body: u32) -> bool {
        let at = (tile.1 * self.tiles.0 + tile.0) * self.words + body / 32;
        self.bits.get(at as usize).is_some_and(|w| w & (1 << (body % 32)) != 0)
    }
}


/// The limits the field renderer needs, on top of whatever the platform's defaults are.
///
/// **Shared so the two devices cannot be asked for different things.** The application's device
/// comes from eframe and an example's comes from `Renderer::new`, and a shader that validates on
/// one and not the other is a bug that only appears in the binary nobody is testing in.
pub fn required_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let base = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::default()
    };
    let available = adapter.limits();
    wgpu::Limits {
        // Never above what the adapter offers: asking for more than a device has is refused
        // outright, where asking for exactly what it has always succeeds.
        max_storage_buffers_per_shader_stage: available
            .max_storage_buffers_per_shader_stage
            .min(storage_buffers_wanted().max(base.max_storage_buffers_per_shader_stage)),
        // eframe's own reason, kept because this replaces the descriptor it would have used:
        // a depth buffer has to cover the whole surface, and 4k displays exist.
        max_texture_dimension_2d: available.max_texture_dimension_2d.min(8192).max(
            base.max_texture_dimension_2d,
        ),
        ..base
    }
}

/// The features a device is asked for so the frame can time itself: GPU timestamps, and only
/// where the adapter offers them.
///
/// Asked for by intersection rather than outright, because asking for a feature a device
/// lacks refuses the device — and the whole point of the measurement is to work on the
/// machine that has no cargo. A device without them times submissions by the wall clock
/// instead, and every line printed from that says so (`ShadingResources::costs`).
pub fn timing_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features()
        & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
}

/// What building a renderer cost, part by part.
///
/// **Because "the driver's pipeline and uploads" is four things.** A frozen application named
/// that stage and nothing narrower, and the very same shader measures 21 ms outside the
/// application — so the number and the guess disagreed, and the only way past that is to say
/// which of the four it was.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildCost {
    /// Baked bands packed into the atlas texture and uploaded, with the slabs that say where each
    /// one landed: next to nothing for an atlas kept from the previous renderer, which is what
    /// keeping it is for.
    pub atlas_ms: f32,
    /// Curve points, spine trees and baked profile grids packed and uploaded.
    pub buffers_ms: f32,
    /// The driver parsing and validating the WGSL.
    pub module_ms: f32,
    /// The driver turning that module into the compute pipelines — three of them.
    pub pipelines_ms: f32,
    /// **Everything else `build` does**, by subtraction rather than by a stopwatch of its own:
    /// the bind group layouts, the pipeline layouts, the bind groups and the skip grid's own
    /// buffer, which is `SKIP_GRID_MAX`³ cells whatever grid the frame will ask for.
    ///
    /// By subtraction because the four above were each timed for a named suspect, and the
    /// Z840's 1.4 s a rebuild was in none of them — a stopwatch only ever covers what someone
    /// thought to put it around. This one cannot miss: it is the total less the named parts.
    pub rest_ms: f32,
}

impl BuildCost {
    /// The dearest parts, named, for a slow-frame line.
    pub fn parts(&self) -> [(&'static str, f32); 5] {
        [
            ("the shader through the driver", self.module_ms),
            ("the driver's pipelines", self.pipelines_ms),
            ("packing curves and profiles", self.buffers_ms),
            ("the baked atlas", self.atlas_ms),
            ("the layouts and the skip grid", self.rest_ms),
        ]
    }
}

/// A compiled program with its buffers, ready to render repeatedly.
pub struct Renderer {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) pipeline: wgpu::ComputePipeline,
    /// Fills the skip grid from the field. Same module, so kept and made with the march's.
    skip_pipeline: wgpu::ComputePipeline,
    /// The still tier's occlusion and shadow, made on the first still frame -- see
    /// [`Pipelines::occlude`].
    occlude_pipeline: std::sync::Arc<std::sync::OnceLock<Option<wgpu::ComputePipeline>>>,
    /// Held so that pipeline can be made when it is first wanted.
    module: wgpu::ShaderModule,
    march_layout: wgpu::PipelineLayout,
    /// The march's side tables: the skip grid -- a header and up to `SKIP_GRID_MAX`³ cells --
    /// and after them the per-tile body mask, `TILE_MASK_TILES` runs of `TILE_MASK_WORDS`.
    ///
    /// **One buffer because a thirteenth storage binding refused the pipeline**; see
    /// `TILE_MASK_BASE` in `march.wgsl` for the whole of that. The mask is never cleared between
    /// frames and does not need to be: whether its bits mean anything is said by `cull_live` in
    /// the camera uniform, which only a submission that recorded the cull into its own encoder
    /// sets, so a frame that did not cull reads the region not at all rather than reading it
    /// stale.
    pub(crate) skip_grid: wgpu::Buffer,
    /// How far past a reading the march steps, as it was last set. See [`RELAX`]. Kept here
    /// because it rides in the skip grid's header, which a fill rewrites.
    relax: std::sync::atomic::AtomicU32,
    /// How big the program was, for the message a refused still march logs.
    size: crate::codegen::ShaderSize,
    /// What each part of building this renderer cost, for a caller reporting a slow frame.
    /// See [`BuildCost`].
    cost: BuildCost,
    /// What the pipeline was compiled from, so the next rebuild can tell whether it still
    /// holds. See [`Renderer::rebuilt_from`].
    source: std::sync::Arc<str>,
    pub(crate) params: wgpu::Buffer,
    pub(crate) curve_points: wgpu::Buffer,
    pub(crate) curve_spans: wgpu::Buffer,
    /// Every conformal surface's floats, integers and layout — see `PackedSurface`.
    pub(crate) surface_floats: wgpu::Buffer,
    pub(crate) surface_uints: wgpu::Buffer,
    pub(crate) surface_layouts: wgpu::Buffer,
    /// Every baked profile's samples, packed end to end and indexed through `curve_spans`.
    pub(crate) profile_bakes: wgpu::Buffer,
    pub(crate) cage_slabs: wgpu::Buffer,
    pub(crate) atlas: wgpu::TextureView,
    /// The bands the atlas and its slabs were packed from, in their order, so the next rebuild
    /// can tell whether they still hold. See [`atlas_holds`].
    bands: Vec<std::sync::Weak<BakedField>>,
    /// The counting copy of the primary march, built beside the frame's own from the same
    /// module. Only under `march-stats`.
    #[cfg(feature = "march-stats")]
    stats_pipeline: wgpu::ComputePipeline,
    /// The field on a lattice of points, from the same module. Only under `march-stats`.
    #[cfg(feature = "march-stats")]
    grid_pipeline: wgpu::ComputePipeline,
    param_count: usize,
    /// Where this program's cull extents start in the parameter buffer, or `None` where it does
    /// not cull. From [`crate::ShaderProgram::cull_base`].
    ///
    /// **Kept rather than re-derived**, because the only other way to ask is to parse `CULL_BASE`
    /// back out of the source we generated -- reading our own output to learn what we decided.
    cull_base: Option<usize>,
    /// Words a tile in the mask `tile_cull` writes: the program's bodies in 32s, which is the
    /// stride the shader reads at and so the one a readback must use.
    cull_words: u32,
    /// `tile_cull`'s pipeline, from the same module as the march, or `None` where this program
    /// does not cull and the entry point would do nothing.
    cull_pipeline: Option<wgpu::ComputePipeline>,
    /// A number no other renderer has had, so a cache keyed by it — the lighting pass keeps
    /// one frame's bind groups across the bands of that frame — cannot serve one renderer's
    /// groups to the next after a rebuild.
    identity: u64,
}

/// Everything a renderer compiles from its source, so it can be kept or made as one thing.
struct Pipelines {
    /// What making these cost: nought for a set kept from the previous renderer, which is the
    /// whole point of keeping them.
    module_ms: f32,
    pipelines_ms: f32,
    march: wgpu::ComputePipeline,
    skip: wgpu::ComputePipeline,
    /// `tile_cull`, or `None` where the program does not cull.
    ///
    /// **Built with the march rather than lazily like the still's.** It is a dozen lines of fixed
    /// WGSL over no composition at all, so it costs the driver nothing measurable -- where the
    /// still's `occlude` inlines the whole field twice more, which is what made that one lazy.
    cull: Option<wgpu::ComputePipeline>,
    /// The still tier's occlusion and shadow, **built on the first still frame and not before**.
    ///
    /// This is where the draft-tier split actually pays. `occlude` is the entry point that
    /// carries the occlusion and the shadow, and compiling it beside the draft's would put two
    /// more copies of the composition in front of the first picture. Built lazily, the first
    /// picture costs the draft's compile alone, which `draft_tier_cost.rs` measures at about
    /// half; the still's is paid later, off the path to that picture, and only by a document
    /// that settles. It used to be a second primary march with the two marches after it; since
    /// those became a dispatch of their own (loop spec P5) the still traces with the draft's
    /// march, and this is the two marches alone.
    ///
    /// `OnceLock` rather than a `Cell`, because a renderer is handed out behind `&self` and must
    /// stay `Sync`.
    occlude: std::sync::Arc<std::sync::OnceLock<Option<wgpu::ComputePipeline>>>,
    /// Kept so the still's pipeline can be made later: a module and a layout are cheap to hold
    /// and impossible to recover once dropped.
    module: wgpu::ShaderModule,
    march_layout: wgpu::PipelineLayout,
    #[cfg(feature = "march-stats")]
    stats: wgpu::ComputePipeline,
    #[cfg(feature = "march-stats")]
    grid: wgpu::ComputePipeline,
}

/// Hands each renderer its [`Renderer::identity`].
static NEXT_RENDERER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Whether an atlas packed from `packed` is the atlas `bands` would pack, so a rebuild may keep
/// it rather than pack and upload every sample again.
///
/// **The same bands in the same order, and nothing weaker.** A band sits behind an `Arc` that
/// nothing writes through -- `get_mut` refuses while the renderer's weak reference stands, and
/// `make_mut` moves the value to a new allocation -- so the same allocation is the same samples,
/// and the order is where each slab's layer starts. A band made afresh with equal samples is
/// declined rather than compared: comparing them would cost about what packing them does, and the
/// view hands back the very `Arc` it kept whenever a band's key is unchanged, which is every
/// rebuild where the geometry under it did not move.
///
/// `packed` is weak so that a renderer does not keep a band's samples in memory after the view
/// has let the band go. A weak reference still holds the allocation, so nothing made later can
/// sit at a gone band's address and be taken for it.
fn atlas_holds(
    packed: &[std::sync::Weak<BakedField>],
    bands: &[std::sync::Arc<BakedField>],
) -> bool {
    packed.len() == bands.len()
        && packed
            .iter()
            .zip(bands)
            .all(|(kept, band)| std::ptr::eq(kept.as_ptr(), std::sync::Arc::as_ptr(band)))
}

impl Renderer {
    /// What every march binds as its second group: the atlas and the side tables, read-only.
    pub(crate) fn atlas_bindings(&self) -> [wgpu::BindGroupEntry<'_>; 2] {
        [
            entry(0, wgpu::BindingResource::TextureView(&self.atlas)),
            entry(1, self.skip_grid.as_entire_binding()),
        ]
    }

    /// The same group as a pass that *writes* the side tables sees it: the skip grid's own fill,
    /// and the per-tile cull, which writes the region after it.
    pub(crate) fn writing_bindings(&self) -> [wgpu::BindGroupEntry<'_>; 2] {
        [
            entry(0, wgpu::BindingResource::TextureView(&self.atlas)),
            entry(2, self.skip_grid.as_entire_binding()),
        ]
    }

    /// `tile_cull`'s pipeline, or `None` where this program does not cull.
    pub(crate) fn cull_pipeline(&self) -> Option<&wgpu::ComputePipeline> {
        self.cull_pipeline.as_ref()
    }

    /// How many workgroups of `tile_cull` cover the tile rows a band of `rows` touches.
    ///
    /// Nought where there is nothing to cull, which is what the caller reads as "do not dispatch,
    /// and do not set `cull_live`".
    pub(crate) fn cull_workgroups(&self, width: u32, height: u32, rows: (u32, u32)) -> u32 {
        if self.cull_base.is_none() || rows.1 == 0 {
            return 0;
        }
        // The band must start on a tile row, or a tile would be half in one band and half in
        // another with only one of them having written it. The caller arranges that; a band that
        // does not is drawn unculled rather than drawn wrong.
        if rows.0 % TILE_PIXELS != 0 {
            return 0;
        }
        let tiles_x = width.div_ceil(TILE_PIXELS);
        let first = rows.0 / TILE_PIXELS;
        let last = ((rows.0 + rows.1).div_ceil(TILE_PIXELS)).min(height.div_ceil(TILE_PIXELS));
        if first >= last || tiles_x == 0 {
            return 0;
        }
        // A frame wider than the mask draws unculled -- see `TILE_MASK_TILES`. Checked on the
        // whole frame rather than on this band's tiles, so every band of one frame agrees.
        if tiles_x * height.div_ceil(TILE_PIXELS) > TILE_MASK_TILES {
            return 0;
        }
        ((last - first) * tiles_x).div_ceil(CULL_WORKGROUP)
    }

    /// Fills the skip grid over the cube at `lo`, `edge` along each side, `cells` a side, from
    /// the field **as the parameter buffer stands now** -- so after [`Renderer::set_params`],
    /// and again whenever they change. One compute pass and nothing read back; every march
    /// submitted after this uses it. See `march.wgsl` for what the grid is and why it is sound.
    ///
    /// `copy_masks` is whether the march is also to use what the same pass records about a
    /// pattern's copies (plan §5 V6). The record is made either way; this only says whether it
    /// is read, which is how the two are measured apart.
    pub fn build_skip_grid(&self, lo: Point3<f64>, edge: f64, cells: u32, copy_masks: bool) {
        let cells = cells.clamp(1, SKIP_GRID_MAX);
        let cell = (edge / f64::from(cells)) as f32;
        if !(cell.is_finite() && cell > 0.0) {
            self.clear_skip_grid();
            return;
        }
        // Stale while it is being filled: a march queued ahead of the fill must not read it.
        // Read as words by the shader, where any float but nought is "yes".
        let masks = if copy_masks { 1.0 } else { 0.0 };
        let relax = f32::from_bits(self.relax.load(std::sync::atomic::Ordering::Relaxed));
        let header = [lo.x as f32, lo.y as f32, lo.z as f32, cells as f32, cell, 0.0, masks, relax];
        self.queue.write_buffer(&self.skip_grid, 0, bytemuck::cast_slice(&header));
        // The field group binds a frame's buffers; the fill reads none of them, but they have
        // to be there, and an all-zero camera is "no section".
        let stand_in = |size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skip-build stand-in"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let camera = stand_in(
            std::mem::size_of::<CameraUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let output = stand_in(16, wgpu::BufferUsages::STORAGE);
        let owners = stand_in(4, wgpu::BufferUsages::STORAGE);
        let extras = stand_in(4, wgpu::BufferUsages::STORAGE);
        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fields"),
            layout: &self.skip_pipeline.get_bind_group_layout(0),
            entries: &self.fields_bindings(&camera, &output, &owners, &extras),
        });
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas, filling the skip grid"),
            layout: &self.skip_pipeline.get_bind_group_layout(1),
            entries: &self.writing_bindings(),
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("skip-build") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skip-build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.skip_pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            let groups = cells.div_ceil(4);
            pass.dispatch_workgroups(groups, groups, groups);
        }
        self.queue.submit(Some(encoder.finish()));
        // Ordered after the fill on the queue, and ahead of whatever is submitted next.
        self.queue.write_buffer(&self.skip_grid, 5 * 4, bytemuck::bytes_of(&1.0f32));
    }

    /// How far past a reading the march steps, as a multiple of it: see [`RELAX`] and
    /// `march.wgsl`. One or less is the plain march. For measuring; a renderer starts at
    /// [`RELAX`] and nothing in the application changes it.
    pub fn set_relaxation(&self, relax: f32) {
        self.relax.store(relax.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.queue.write_buffer(&self.skip_grid, 7 * 4, bytemuck::bytes_of(&relax));
    }

    /// What the last fill recorded, cell by cell: each cell's clearance, and which operands it
    /// ruled out.
    ///
    /// **For measuring whether a mask ever fires, which no frame time can answer.** A masked
    /// operand that is never ruled out anywhere costs a WGSL function and a branch and buys
    /// nothing, and from the outside that looks exactly like a mask that fires and does not pay.
    /// The clearance comes with it because a mask firing in a cell the march never enters buys
    /// nothing either, and only the clearance says which cells those are.
    ///
    /// A set bit is "ruled out"; a cell that never reached the join has its bits clear, which
    /// reads the same as kept -- deliberately, since neither is a skip. Blocks on the queue, so
    /// nothing on a frame path may call it.
    pub fn read_skip_grid(&self, cells: u32) -> Result<SkipGridReading, RenderError> {
        let cells = cells.clamp(1, SKIP_GRID_MAX);
        let count = u64::from(cells).pow(3);
        // **Packed by the grid it was filled at, not by the buffer.** `skip_build` indexes
        // every plane by its own `n`, so a 32-cell fill leaves the 64-cell offsets holding
        // whatever was there before -- which reads as a mask that never fired.
        let stride = count;
        let back = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skip grid back"),
            size: count * 4 * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("skip grid readback") },
        );
        encoder.copy_buffer_to_buffer(&self.skip_grid, SKIP_HEADER * 4, &back, 0, count * 4);
        encoder.copy_buffer_to_buffer(
            &self.skip_grid,
            (SKIP_HEADER + 2 * stride) * 4,
            &back,
            count * 4,
            count * 4,
        );
        self.queue.submit(Some(encoder.finish()));
        back.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;
        let reading = {
            let view = back.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            let n = count as usize;
            SkipGridReading {
                clearance: words[..n].iter().map(|w| f32::from_bits(*w)).collect(),
                ruled_out: words[n..].to_vec(),
            }
        };
        back.unmap();
        Ok(reading)
    }

    /// The per-tile body mask as the last submission's `tile_cull` left it, for a frame of
    /// `width` by `height`.
    ///
    /// **For holding the cull to its own replica, which a hit count cannot do.** `cpu_march_gate`
    /// marches the CPU with this mask rather than with its own five-plane verdict, so a
    /// disagreement between the two marches is the *march's* and nothing else's -- and then
    /// compares this mask against `ruled_out_on_the_cpu` tile by tile, which is the only test that
    /// sees a plane oriented the wrong way (§1.7's first fault) without a body having to stand
    /// behind it. Nothing says which camera filled it: the caller reads it straight after the
    /// `render` whose camera it wants, as `read_skip_grid` is read after its fill. Blocks on the
    /// queue, so nothing on a frame path may call it.
    ///
    /// Reads nothing where the program does not cull, since the region was never written.
    pub fn read_tile_mask(&self, width: u32, height: u32) -> Result<TileMaskReading, RenderError> {
        let tiles = (width.div_ceil(TILE_PIXELS), height.div_ceil(TILE_PIXELS));
        let words = self.cull_words;
        if self.cull_base.is_none() || tiles.0 * tiles.1 > TILE_MASK_TILES {
            return Ok(TileMaskReading { tiles, words, bits: Vec::new() });
        }
        let count = u64::from(tiles.0) * u64::from(tiles.1) * u64::from(words);
        let back = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile mask back"),
            size: count * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tile mask readback") },
        );
        encoder.copy_buffer_to_buffer(&self.skip_grid, TILE_MASK_BASE * 4, &back, 0, count * 4);
        self.queue.submit(Some(encoder.finish()));
        back.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;
        let bits = {
            let view = back.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        back.unmap();
        Ok(TileMaskReading { tiles, words, bits })
    }

    /// Says the skip grid no longer describes the field, so the march stops asking it.
    pub fn clear_skip_grid(&self) {
        self.queue.write_buffer(&self.skip_grid, 5 * 4, bytemuck::bytes_of(&0.0f32));
    }

    /// The still's occlusion and shadow for `camera`, made on first use, or `None` for a frame
    /// that does not ask for them.
    ///
    /// **Chosen from `camera.occlusion`, which is the same field the uniform's `key_light.w` is
    /// built from** — so whether this runs and the branch the shader takes cannot disagree
    /// about which tier this is. Deriving it from anything else would be a second description of
    /// one fact, and the two would drift.
    ///
    /// The pipeline is made the first time a still frame asks for it, which is the whole
    /// point: the first picture of a document pays the draft's compile alone. See
    /// [`Pipelines::occlude`] and `draft_tier_cost.rs`.
    /// **Refused here, it is left out rather than panicking.** The build path wraps
    /// pipeline creation in error scopes so a shader the driver will not take becomes
    /// [`RenderError::ShaderRefused`] — and this one, made later and behind `&self`, had none, so a
    /// still the driver refused would have panicked the frame thread. Exactly the failure the
    /// scopes were added to remove, reintroduced by making a pipeline somewhere else.
    ///
    /// There is no `Result` to return from a dispatch, so the answer is the draft's march alone:
    /// the still then draws without its ambient term and its shadow, which is a visibly flatter
    /// picture rather than a closed window. The error goes to the log, once, because
    /// `OnceLock::get_or_init` runs the closure a single time.
    pub(crate) fn occlude_pipeline(&self, camera: &Camera) -> Option<&wgpu::ComputePipeline> {
        camera.occlusion?;
        self.occlude_pipeline
            .get_or_init(|| {
                let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
                let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
                let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
                let occlude =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("occlude"),
                            layout: Some(&self.march_layout),
                            module: &self.module,
                            entry_point: Some("occlude"),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                let refused = pollster::block_on(validation.pop())
                    .or_else(|| pollster::block_on(memory.pop()))
                    .or_else(|| pollster::block_on(internal.pop()));
                match refused {
                    Some(error) => {
                        eprintln!(
                            "the driver would not compile this part's still occlusion ({}) — the \
                             viewport will shade it without occlusion or shadow. Driver: {error}",
                            self.size,
                        );
                        None
                    }
                    None => Some(occlude),
                }
            })
            .as_ref()
    }

    /// Compiles the still's `occlude` entry cold, from this renderer's source with `surgery`
    /// applied to the text, and answers the milliseconds the module and the pipeline took.
    ///
    /// **A measurement, not a path the application takes.** The pipeline the viewport uses is
    /// [`Renderer::occlude_pipeline`]'s, made once from the module the draft was; this one makes a
    /// fresh module each call so the driver cannot answer from the draft's work, and takes the
    /// source through a caller's edit so a piece of the shader can be cut out and its share of the
    /// time read off -- which is how `document_cost` lays a partial revolve's minutes at the outline
    /// walk's door (spec `round-chamfer-spec.md` §4.4). The module is dropped on return.
    pub fn time_occlude_pipeline(&self, surgery: &dyn Fn(&str) -> String) -> (f64, f64) {
        let source = surgery(&self.source);
        let ms = |since: std::time::Instant| since.elapsed().as_secs_f64() * 1000.0;
        let making = std::time::Instant::now();
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field (timed)"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let module_ms = ms(making);
        let compiling = std::time::Instant::now();
        let _occlude = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("occlude (timed)"),
                layout: Some(&self.march_layout),
                module: &module,
                entry_point: Some("occlude"),
                compilation_options: Default::default(),
                cache: None,
            });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        (module_ms, ms(compiling))
    }

    /// Records the march of a band `rows` high, and after it the still's occlusion and shadow
    /// where `camera` asks for them, into `pass`: at the measured pixels, and at one in four a
    /// second dispatch for the rest, which marches those the lighting pass could not fill in.
    ///
    /// **One pass for all of it, so that whatever brackets the pass brackets all of it.** A band
    /// is sized from what the last one cost by the GPU's timestamps (loop spec S2), and a second
    /// pass between the march's and the lighting's would be work no timestamp saw -- bands sized
    /// from an undercount, on exactly the slow card the bands exist for. Dispatches in one pass
    /// are still ordered: each is its own usage scope, so `occlude` reads what `march` wrote, and
    /// its second dispatch the readings its first wrote.
    pub(crate) fn record_march(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        groups: [&wgpu::BindGroup; 2],
        camera: &Camera,
        width: u32,
        rows: u32,
        measured: u32,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, groups[0], &[]);
        pass.set_bind_group(1, groups[1], &[]);
        pass.dispatch_workgroups(width.div_ceil(8), rows.div_ceil(8), 1);
        if let Some(occlude) = self.occlude_pipeline(camera) {
            let (across, down) = occlusion_workgroups(width, rows, measured);
            pass.set_pipeline(occlude);
            pass.set_bind_group(0, groups[0], &[]);
            pass.set_bind_group(1, groups[1], &[]);
            pass.dispatch_workgroups(across, down, 1);
            if measured & 4 != 0 {
                // Three deep, which is how `occlude` knows this dispatch from the first.
                let (across, down, deep) = unmeasured_workgroups(width, rows);
                pass.dispatch_workgroups(across, down, deep);
            }
        }
    }

    fn pipelines(&self) -> Pipelines {
        Pipelines {
            // Kept, not made: the whole reason a rebuild whose source has not changed is cheap.
            module_ms: 0.0,
            pipelines_ms: 0.0,
            march: self.pipeline.clone(),
            skip: self.skip_pipeline.clone(),
            cull: self.cull_pipeline.clone(),
            // Shared, so a rebuild that keeps the pipelines keeps a still that was already made
            // rather than paying for it again on the next still frame.
            occlude: self.occlude_pipeline.clone(),
            module: self.module.clone(),
            march_layout: self.march_layout.clone(),
            #[cfg(feature = "march-stats")]
            stats: self.stats_pipeline.clone(),
            #[cfg(feature = "march-stats")]
            grid: self.grid_pipeline.clone(),
        }
    }

    /// Builds a renderer for `program`, baking each cage it references.
    ///
    /// `cage_fields` must be in the program's binding order — `program.baked[i]` is described
    /// by `cage_fields[i]`. `curves` likewise. The caller resolves those from the document,
    /// which is what keeps this module free of any knowledge of one.
    pub async fn new(
        program: &ShaderProgram,
        cage_fields: &[(&(dyn ScalarField + Sync), (Point3<f64>, Point3<f64>))],
        curves: &[(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)],
        // Every conformal surface the graph names, already packed. See `PackedSurface`.
        surfaces: &[fieldform_field::PackedSurface],
        bake_resolution: usize,
        bake_band: f64,
    ) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| RenderError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: required_limits(&adapter),
                // So an example's `shading_cost` row is the GPU's own number where the card
                // can give one, the same as the application's perf line.
                required_features: timing_features(&adapter),
                ..Default::default()
            })
            .await?;
        // Bakes here, where an example or a test hands over fields rather than bakes. The
        // application does it itself and keeps them — see `with_device`.
        let resolution = bake_resolution_limit(&device, cage_fields.len(), bake_resolution);
        let bakes: Vec<std::sync::Arc<BakedField>> = cage_fields
            .iter()
            .map(|(field, bounds)| {
                std::sync::Arc::new(BakedField::new(*field, *bounds, resolution, bake_band))
            })
            .collect();
        Self::with_device(device, queue, program, &bakes, curves, surfaces)
    }

    /// Builds a renderer on a device someone else owns.
    ///
    /// This is the constructor an application uses. An eframe app already has a device and a
    /// queue, and creating a second one would mean a second adapter, duplicate memory, and no
    /// way to share a texture with the surface it eventually has to draw into. [`Self::new`]
    /// exists for examples and tests, where there is no device to borrow.
    ///
    /// Builds a renderer from bakes the caller already holds.
    ///
    /// The bakes are passed in rather than made here because they are what a rebuild actually
    /// costs, and they are *reusable*: a bake is a pure function of one leaf's geometry, its own
    /// box, the resolution and the band, none of which a show/hide changes. Measured on thirty
    /// small solids, `hide_cost.rs`: 890 ms to rebuild them all, against a shader that
    /// generates and validates in 3.8 ms. Hiding a body was re-baking twenty-nine bodies that
    /// had not changed.
    #[allow(clippy::too_many_arguments)]
    pub fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        program: &ShaderProgram,
        bakes: &[std::sync::Arc<BakedField>],
        curves: &[(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)],
        surfaces: &[fieldform_field::PackedSurface],
    ) -> Result<Self, RenderError> {
        Self::build(device, queue, program, bakes, curves, surfaces, None)
    }

    /// A renderer for `program` on `previous`'s device, **keeping `previous`'s pipeline when the
    /// source is the one it was compiled from.**
    ///
    /// Plan §5 V2. Every number in a generated shader is a buffer read, so most edits -- a
    /// dragged radius, a moved control point, a typed count -- generate the source they
    /// generated last time, to the byte. Handing that to the driver again is a lookup in its
    /// cache rather than a compile, but it is still a module parsed and validated and a
    /// pipeline asked for, once a drag step, on the interface's thread
    /// (`fieldform-io/examples/edit_step_cost.rs` has the figures). A pipeline is a function of
    /// its source and nothing else, so the one in hand is the one that would come back.
    ///
    /// **And its atlas, when the bands are the ones it was packed from** ([`atlas_holds`]): the
    /// same reasoning one level down, since an atlas is a function of its bands and nothing else.
    /// Everything else is made afresh: the buffers are what the edit changed.
    pub fn rebuilt_from(
        previous: &Renderer,
        program: &ShaderProgram,
        bakes: &[std::sync::Arc<BakedField>],
        curves: &[(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)],
        surfaces: &[fieldform_field::PackedSurface],
    ) -> Result<Self, RenderError> {
        Self::build(
            previous.device.clone(),
            previous.queue.clone(),
            program,
            bakes,
            curves,
            surfaces,
            Some(previous),
        )
    }

    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        program: &ShaderProgram,
        bakes: &[std::sync::Arc<BakedField>],
        // Each curve as the shader needs it: the polyline, and the grid its distance was baked
        // onto when one was worth making. A pipe's spine never has one; a profile has one
        // whenever the tolerance it was compiled at allows it.
        curves: &[(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)],
        // Every conformal surface the graph names, already packed. See `PackedSurface`.
        surfaces: &[fieldform_field::PackedSurface],
        // A renderer whose pipeline may be kept, if its source is this program's.
        previous: Option<&Renderer>,
    ) -> Result<Self, RenderError> {
        let began = std::time::Instant::now();
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f32() * 1e3;
        // **The one edit the driver's text takes beyond the generator's.** The normal pass reads
        // every section through `profile_distance_bake`, whose fallback for a section the
        // tolerance could not bake is the walk the value pass takes -- a loop the driver inlines
        // into every differenced normal, and the whole of what made four partial revolves' compile
        // 9.7 s where the bake alone makes it 2.5 s. Only this renderer knows whether every
        // section has its bake, so it cuts the walk here when none is missing. The cache below
        // compares the edited text, so a drag that costs a section its bake is a new pipeline and
        // never a stale one reading a grid it does not have.
        let every_section_baked = program
            .curves
            .iter()
            .zip(curves.iter())
            .all(|(use_, (_, bake))| {
                !matches!(use_, crate::codegen::CurveUse::Profile(_) | crate::codegen::CurveUse::Wall(..)) || bake.is_some()
            });
        let source: String = if every_section_baked {
            program.source.replacen(
                "    return profile_distance(index, q); // unbaked-section-walk",
                "    return 1e30; // every section has its bake; see the renderer",
                1,
            )
        } else {
            program.source.clone()
        };
        // **Kept when it would be packed from the same bands** (`atlas_holds`). Every sample of
        // every band is copied twice on this thread and then uploaded -- a 192-cell band is
        // 115 MB -- and a rebuild that changed a number, a curve or which shader is drawn
        // leaves every band where it was. The slabs go with it: they say where each band's
        // layer starts, so they are the atlas's own description and are kept or made with it.
        let (atlas, cage_slabs) = match previous.filter(|p| atlas_holds(&p.bands, bakes)) {
            Some(kept) => (kept.atlas.clone(), kept.cage_slabs.clone()),
            None => {
                let (atlas, slabs) = Self::build_atlas(&device, &queue, bakes)?;
                let cage_slabs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cage_slabs"),
                    contents: bytemuck::cast_slice(&slabs),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                (atlas, cage_slabs)
            }
        };
        let atlas_ms = ms(began);
        let packing = std::time::Instant::now();

        // A runtime-sized storage array cannot be empty, so every buffer gets at least one
        // element. A graph with no pipes still has to bind something for `curve_points`.
        let mut points: Vec<[f32; 4]> = Vec::new();
        let mut spans: Vec<CurveSpan> = Vec::new();
        let mut bake_values: Vec<f32> = Vec::new();
        for (spine, bake) in curves.iter() {
            let mut span = CurveSpan {
                start: points.len() as u32,
                count: spine.len() as u32,
                coarse_start: 0,
                page_start: 0,
                fine_start: 0,
                blocks_wide: 0,
                blocks_high: 0,
                bake_wide: 0,
                bake_high: 0,
                lo_x: 0.0,
                lo_y: 0.0,
                cell: 1.0,
                margin: 0.0,
                coarse_margin: 0.0,
                bvh_start: 0,
                bvh_count: 0,
                segs_start: 0,
                seglist_start: 0,
                ccw: 1,
            };
            if let Some(baked) = bake {
                // All three parts in one buffer, in this order, with the span saying where each
                // begins. The page is written as `f32` alongside the samples rather than as a
                // buffer of its own: an index survives a round trip through `f32` exactly at
                // these sizes, and "no block" is any negative number, which an index never is.
                let b = baked.for_shader();
                span.coarse_start = bake_values.len() as u32;
                bake_values.extend_from_slice(b.coarse);
                span.page_start = bake_values.len() as u32;
                bake_values.extend(b.page.iter().map(|p| {
                    if *p == u32::MAX { -1.0 } else { *p as f32 }
                }));
                span.fine_start = bake_values.len() as u32;
                bake_values.extend_from_slice(b.fine);
                span.segs_start = bake_values.len() as u32;
                bake_values.extend(b.segs.iter().map(|v| *v as f32));
                span.seglist_start = bake_values.len() as u32;
                bake_values.extend(b.seglist.iter().map(|v| *v as f32));
                span.ccw = u32::from(b.ccw);
                span.blocks_wide = b.blocks_wide as u32;
                span.blocks_high = b.blocks_high as u32;
                span.bake_wide = b.wide as u32;
                span.bake_high = b.high as u32;
                span.lo_x = b.lo.x as f32;
                span.lo_y = b.lo.y as f32;
                span.cell = b.cell as f32;
                span.margin = b.margin as f32;
                span.coarse_margin = b.coarse_margin as f32;
            }
            // Each point with the arc length from the curve's start in `w`, summed in `f64`
            // from the points as given; then the tree over the segments, so the shader can
            // find the nearest one without visiting them all.
            let first_point = points.len();
            let mut arc = 0.0f64;
            for (k, p) in spine.iter().enumerate() {
                if k > 0 {
                    arc += (p - spine[k - 1]).norm();
                }
                points.push([p.x as f32, p.y as f32, p.z as f32, arc as f32]);
            }
            // `FIELDFORM_NO_SPINE_INDEX` leaves the tree out, for a measurement of what it is
            // worth against the walk on the same scene; nothing else reads it.
            let tree = if std::env::var_os("FIELDFORM_NO_SPINE_INDEX").is_some() {
                Vec::new()
            } else {
                segment_tree(&points[first_point..])
            };
            if !tree.is_empty() {
                span.bvh_start = points.len() as u32;
                span.bvh_count = (tree.len() / 2) as u32;
                points.extend(tree);
            }
            spans.push(span);
        }
        if bake_values.is_empty() {
            bake_values.push(0.0);
        }
        if points.is_empty() {
            points.push([0.0; 4]);
        }
        if spans.is_empty() {
            spans.push(CurveSpan {
                start: 0,
                count: 0,
                coarse_start: 0,
                page_start: 0,
                fine_start: 0,
                blocks_wide: 0,
                blocks_high: 0,
                bake_wide: 0,
                bake_high: 0,
                lo_x: 0.0,
                lo_y: 0.0,
                cell: 1.0,
                margin: 0.0,
                coarse_margin: 0.0,
                bvh_start: 0,
                bvh_count: 0,
                segs_start: 0,
                seglist_start: 0,
                ccw: 1,
            });
        }

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: (program.params.len().max(1) * 4) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let curve_points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("curve_points"),
            contents: bytemuck::cast_slice(&points),
            usage: storage,
        });
        let curve_spans = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("curve_spans"),
            contents: bytemuck::cast_slice(&spans),
            usage: storage,
        });
        let profile_bakes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("profile_bakes"),
            contents: bytemuck::cast_slice(&bake_values),
            usage: storage,
        });

        // Each surface's runs concatenated, with its layout shifted by where its run begins.
        // The shader indexes one array per scalar type and one layout per surface, which is
        // three bindings however many surfaces a graph has.
        let mut surface_floats: Vec<f32> = Vec::new();
        let mut surface_uints: Vec<u32> = Vec::new();
        let mut surface_layouts: Vec<GpuSurfaceLayout> = Vec::new();
        for packed in surfaces {
            let float_base = surface_floats.len() as u32;
            let uint_base = surface_uints.len() as u32;
            let l = packed.layout;
            surface_layouts.push(GpuSurfaceLayout {
                positions: float_base + l.positions,
                normals: float_base + l.normals,
                uvs: float_base + l.uvs,
                gradients: float_base + l.gradients,
                node_bounds: float_base + l.node_bounds,
                triangles: uint_base + l.triangles,
                faces: uint_base + l.faces,
                node_links: uint_base + l.node_links,
                order: uint_base + l.order,
                triangle_count: l.triangle_count,
                worst_gradient: l.worst_gradient,
            });
            surface_floats.extend_from_slice(&packed.floats);
            surface_uints.extend_from_slice(&packed.uints);
        }
        // A graph with no surfaces still has to bind something, exactly as `curve_points` does.
        if surface_floats.is_empty() {
            surface_floats.push(0.0);
        }
        if surface_uints.is_empty() {
            surface_uints.push(0);
        }
        if surface_layouts.is_empty() {
            surface_layouts.push(GpuSurfaceLayout::default());
        }
        let surface_floats = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface_floats"),
            contents: bytemuck::cast_slice(&surface_floats),
            usage: storage,
        });
        let surface_uints = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface_uints"),
            contents: bytemuck::cast_slice(&surface_uints),
            usage: storage,
        });
        let surface_layouts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface_layouts"),
            contents: bytemuck::cast_slice(&surface_layouts),
            usage: storage,
        });

        let buffers_ms = ms(packing);
        // **A pipeline the driver refuses must not take the application down** (plan step 13).
        //
        // wgpu's default answer to an error is to panic on whatever thread raised it, and a
        // rebuild happens on the frame's own thread — so a graph one size too large for the
        // driver closed the window, with `fieldform-crash.log` naming `create_compute_pipeline`
        // and nothing naming the graph. An error scope captures the error instead and hands it
        // back, and `TracedView` already keeps the previous renderer when a build returns `Err`,
        // so the last good image stays on screen while the message says what was too big.
        //
        // Three scopes, nested, because the three filters are exclusive and a driver's refusal
        // does not announce which kind it will be: a shader too large has come back as each of
        // them on different backends. Popped innermost first, which is the order wgpu requires.
        //
        // **This catches a refusal, not slowness.** A driver that takes two minutes rather than
        // failing is a compile that succeeds, and no scope will report it; that is what the size
        // report and `shader_size_cost` are for.
        let compile = || -> Result<Pipelines, RenderError> {
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let compiling = std::time::Instant::now();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });

        // Declared explicitly rather than derived from the shader with `layout: None`.
        //
        // A derived layout contains only the bindings the shader actually *references*, and
        // the generated shader varies: a graph with no pipe never calls `pipe_distance`, and
        // one with no numbers at all never reads `params`. Those bindings then vanish from the
        // layout while the binding code still supplies them, and wgpu treats the mismatch as
        // fatal. A bare cage produced exactly that — three bindings in the layout against six
        // supplied — and crashed the app.
        //
        // Unused entries in an explicit layout are harmless, so this is stable across every
        // shape of generated program.
        let fields_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fields"),
            entries: &fields_layout_entries(),
        });
        let atlas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atlas"),
            entries: &[atlas_texture_entry(), skip_grid_entry(1, true)],
        });
        // The pass that fills the grid sees the same buffer writable, at a binding of its own.
        let skip_build_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("atlas, filling the skip grid"),
                entries: &[atlas_texture_entry(), skip_grid_entry(2, false)],
            });
        let skip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skip-build"),
                bind_group_layouts: &[Some(&fields_layout), Some(&skip_build_layout)],
                immediate_size: 0,
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("march"),
            bind_group_layouts: &[Some(&fields_layout), Some(&atlas_layout)],
            immediate_size: 0,
        });
        let module_ms = ms(compiling);
        let shaping = std::time::Instant::now();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("march"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("march"),
            compilation_options: Default::default(),
            cache: None,
        });
        // The counting march: the same two groups, and a third of its own for the two buffers
        // it writes, so nothing the frame's pipeline binds has changed.
        #[cfg(feature = "march-stats")]
        let (stats_pipeline, grid_pipeline) = {
            let counted = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let stats_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("march-stats"),
                entries: &[
                    counted(0, false),
                    counted(1, false),
                    counted(2, false),
                    counted(3, false),
                    counted(4, true),
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("march-stats"),
                bind_group_layouts: &[Some(&fields_layout), Some(&atlas_layout), Some(&stats_layout)],
                immediate_size: 0,
            });
            let make = |entry: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            (make("march_stats"), make("grid_eval"))
        };
            let skip = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("skip-build"),
                layout: Some(&skip_pipeline_layout),
                module: &module,
                entry_point: Some("skip_build"),
                compilation_options: Default::default(),
                cache: None,
            });
            // **The same layout the skip grid's own fill uses**, because the mask is part of the
            // same buffer: the cull writes it through `skip_out`, at binding 2.
            let cull = program.cull_base.map(|_| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("tile-cull"),
                    layout: Some(&skip_pipeline_layout),
                    module: &module,
                    entry_point: Some("tile_cull"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            });
            let pipelines_ms = ms(shaping);
            // Innermost first. Each pop takes effect immediately, so the pipelines above are
            // already out of every scope by the time these resolve.
            let refused = pollster::block_on(validation.pop())
                .or_else(|| pollster::block_on(out_of_memory.pop()))
                .or_else(|| pollster::block_on(internal.pop()));
            if let Some(error) = refused {
                return Err(RenderError::ShaderRefused {
                    size: program.size,
                    detail: error.to_string(),
                });
            }
            Ok(Pipelines {
                module_ms,
                pipelines_ms,
                march: pipeline,
                skip,
                cull,
                occlude: std::sync::Arc::new(std::sync::OnceLock::new()),
                module,
                march_layout: pipeline_layout,
                #[cfg(feature = "march-stats")]
                stats: stats_pipeline,
                #[cfg(feature = "march-stats")]
                grid: grid_pipeline,
            })
        };
        // Compared whole, not by a hash: a wrong pipeline is a wrong picture, and comparing a
        // hundred kilobytes is microseconds.
        let pipelines = match previous.filter(|p| *p.source == *source) {
            Some(kept) => kept.pipelines(),
            None => compile()?,
        };

        // Zeroed, which is a grid that says it describes nothing: `[5]` is nought.
        let skip_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("side-tables"),
            // A clearance, a copy mask and an operand mask for every cell, then the per-tile
            // body mask. See `march.wgsl`, and `TILE_MASK_BASE` for why the mask is in here.
            size: (TILE_MASK_BASE + u64::from(TILE_MASK_TILES) * u64::from(TILE_MASK_WORDS)) * 4,
            // `COPY_SRC` so `read_operand_masks` can say how often a mask actually fires --
            // the question no frame time can answer on its own.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // The one word of its header that is not about the grid: see `RELAX`.
        queue.write_buffer(&skip_grid, 7 * 4, bytemuck::bytes_of(&RELAX));

        Ok(Self {
            device,
            queue,
            size: program.size,
            pipeline: pipelines.march,
            skip_pipeline: pipelines.skip,
            occlude_pipeline: pipelines.occlude,
            module: pipelines.module,
            march_layout: pipelines.march_layout,
            skip_grid,
            relax: std::sync::atomic::AtomicU32::new(RELAX.to_bits()),
            source: std::sync::Arc::from(source.as_str()),
            params,
            curve_points,
            curve_spans,
            surface_floats,
            surface_uints,
            surface_layouts,
            profile_bakes: profile_bakes_buffer,
            cage_slabs,
            atlas,
            bands: bakes.iter().map(std::sync::Arc::downgrade).collect(),
            #[cfg(feature = "march-stats")]
            stats_pipeline: pipelines.stats,
            #[cfg(feature = "march-stats")]
            grid_pipeline: pipelines.grid,
            cost: BuildCost {
                atlas_ms,
                buffers_ms,
                module_ms: pipelines.module_ms,
                pipelines_ms: pipelines.pipelines_ms,
                // Never below nought: the four are taken on one clock and the whole on the
                // same one, but a subtraction of floats has no obligation to be kind.
                rest_ms: (ms(began)
                    - atlas_ms
                    - buffers_ms
                    - pipelines.module_ms
                    - pipelines.pipelines_ms)
                    .max(0.0),
            },
            param_count: program.params.len().max(1),
            cull_base: program.cull_base,
            // `MASK_WORDS` as `cull_decls` emits it: one body a bit, 32 to the word.
            cull_words: (program.body_boxes.len() as u32).div_ceil(32).max(1),
            cull_pipeline: pipelines.cull,
            identity: NEXT_RENDERER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// Stacks every leaf's bake into one 3D texture and records where each one landed, or says
    /// it cannot.
    ///
    /// Bakes of different sizes are padded to the widest, and `sample_baked_raw` clamps to each
    /// bake's own dimensions, so the padding is allocated but never read.
    ///
    /// The caller already scales the resolution to the device's limit; this is the check that
    /// the arithmetic held, because the alternative to a `Result` here is a process that exits
    /// with a validation error naming a texture dimension rather than a part.
    fn build_atlas(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bakes: &[std::sync::Arc<BakedField>],
    ) -> Result<(wgpu::TextureView, Vec<CageSlab>), RenderError> {
        let width = bakes.iter().map(|b| b.dims[0]).max().unwrap_or(1).max(1);
        let height = bakes.iter().map(|b| b.dims[1]).max().unwrap_or(1).max(1);
        let depth: usize = bakes.iter().map(|b| b.dims[2]).sum::<usize>().max(1);

        // RGBA: the cage's own normal in xyz, its distance in w.
        let mut data = vec![[0.0f32; 4]; width * height * depth];
        let mut slabs = Vec::new();
        let mut layer = 0usize;
        for bake in bakes {
            let texels = bake.texels();
            for k in 0..bake.dims[2] {
                for j in 0..bake.dims[1] {
                    for i in 0..bake.dims[0] {
                        let src = (k * bake.dims[1] + j) * bake.dims[0] + i;
                        let dst = ((layer + k) * height + j) * width + i;
                        data[dst] = texels[src];
                    }
                }
            }
            slabs.push(CageSlab {
                lo: [bake.lo.x as f32, bake.lo.y as f32, bake.lo.z as f32, 0.0],
                inv_cell: [
                    (1.0 / bake.cell.x) as f32,
                    (1.0 / bake.cell.y) as f32,
                    (1.0 / bake.cell.z) as f32,
                    0.0,
                ],
                dims: [
                    bake.dims[0] as u32,
                    bake.dims[1] as u32,
                    bake.dims[2] as u32,
                    0,
                ],
                layer_offset: layer as u32,
                band: bake.band as f32,
                _pad: [0.0; 2],
            });
            layer += bake.dims[2];
        }
        if slabs.is_empty() {
            slabs.push(CageSlab {
                lo: [0.0; 4],
                inv_cell: [1.0, 1.0, 1.0, 0.0],
                dims: [1, 1, 1, 0],
                layer_offset: 0,
                band: 0.0,
                _pad: [0.0; 2],
            });
        }

        let limit = device.limits().max_texture_dimension_3d as usize;
        if width > limit || height > limit || depth > limit {
            return Err(RenderError::AtlasTooLarge {
                width,
                height,
                depth,
                limit,
            });
        }

        let size = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: depth as u32,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cage_atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((width * 16) as u32),
                rows_per_image: Some(height as u32),
            },
            size,
        );

        Ok((texture.create_view(&Default::default()), slabs))
    }

    /// Updates the parameter values. This is the cheap path: no rebuild, no recompile.
    pub fn set_params(&self, values: &[f32]) {
        let mut padded = values.to_vec();
        padded.resize(self.param_count, 0.0);
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&padded));
    }

    /// Renders one frame, returning a normal and depth per pixel in row-major order.
    pub fn render(
        &self,
        camera: &Camera,
        width: u32,
        height: u32,
        max_steps: u32,
    ) -> Result<Vec<Hit>, RenderError> {
        // **The readback path culls too**, which is what puts the mask under `gpu_matches_cpu`:
        // that test compares this function's hits against the CPU's field, so a cull that ruled
        // out a body with material in it shows up there as a changed depth rather than as a
        // report from a user. The whole frame is one band, so it starts on a tile row.
        let cull_groups = self.cull_workgroups(width, height, (0, height));
        let measured = occlusion_pixels(camera, 0);
        let uniform = camera_uniform(
            camera,
            width,
            height,
            max_steps,
            (0, height),
            cull_groups > 0,
            measured,
        );
        let camera_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("camera"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pixels = (width as u64) * (height as u64);
        let bytes = pixels * 16;
        let owner_bytes = pixels * 4;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Which body each pixel belongs to. A quarter the size of the normals, and read back
        // beside them so a highlight costs a comparison per pixel rather than a second trace.
        let owners = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("owners"),
            size: owner_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let owners_back = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("owners_readback"),
            size: owner_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Occlusion and shadow, which this path binds and never reads: a `Hit` is geometry,
        // and the comparison against the CPU field is about where the surface is.
        let extras = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("extras"),
            size: owner_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fields"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &self.fields_bindings(&camera_buf, &output, &owners, &extras),
        });
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &self.atlas_bindings(),
        });
        // The cull's own second group: the same buffer, writable, at its own binding. Group 0 is
        // shared with the march -- both pipelines' layouts were made from one `fields_layout`,
        // which is what already lets the still's `occlude` bind the group the march made.
        let cull = self.cull_pipeline.as_ref().filter(|_| cull_groups > 0).map(|pipeline| {
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile-cull"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &self.writing_bindings(),
            });
            (pipeline, group)
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Before the march, in the same encoder: compute passes of one encoder run in the order
        // they were recorded, so the march reads what this wrote.
        if let Some((pipeline, group)) = &cull {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile-cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, group, &[]);
            pass.dispatch_workgroups(cull_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("march"),
                timestamp_writes: None,
            });
            // With the still's occlusion and shadow where the camera asks for them, though nothing
            // here reads them back: a readback with a still's camera compiles and runs what a still
            // frame does, which is what `draft_tier_cost` times it for.
            self.record_march(&mut pass, [&group0, &group1], camera, width, height, measured);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        encoder.copy_buffer_to_buffer(&owners, 0, &owners_back, 0, owner_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let owner_slice = owners_back.slice(..);
        owner_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;

        let view = slice.get_mapped_range();
        let owner_view = owner_slice.get_mapped_range();
        let raw: &[[f32; 4]] = bytemuck::cast_slice(&view);
        let owned: &[u32] = bytemuck::cast_slice(&owner_view);
        let hits = raw
            .iter()
            .zip(owned)
            .map(|(p, body)| Hit {
                normal: Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64),
                depth: p[3] as f64,
                body: *body,
            })
            .collect();
        drop(view);
        drop(owner_view);
        readback.unmap();
        owners_back.unmap();

        Ok(hits)
    }
}

#[cfg(feature = "march-stats")]
impl Renderer {
    /// Traces the primary march of one frame with a counter in it, and reads back how many
    /// steps each ray took and why it stopped. See [`crate::stats`].
    ///
    /// Not a timing measurement — it reads back, and it goes up in bands of rows, each its own
    /// submission, because a whole 1080-pixel frame of a patterned pipe in one packet is the
    /// thing the loop spec's S2 forbids and an instrument should not be what loses the device.
    pub fn march_stats(
        &self,
        camera: &Camera,
        width: u32,
        height: u32,
        max_steps: u32,
    ) -> Result<crate::stats::MarchCapture, RenderError> {
        let (width, height) = (width.max(1), height.max(1));
        let pixels = u64::from(width) * u64::from(height);
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let camera_buf = buffer(
            "stats-camera",
            std::mem::size_of::<CameraUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        // Bound because the field group binds them; the counting march never writes them.
        let output = buffer("stats-output", pixels * 16, storage);
        let owners = buffer("stats-owners", pixels * 4, storage);
        let extras = buffer("stats-extras", pixels * 4, storage);
        let steps = buffer("stats-steps", pixels * 4, storage | wgpu::BufferUsages::COPY_SRC);
        let exits = buffer("stats-exits", pixels * 4, storage | wgpu::BufferUsages::COPY_SRC);
        let calls = buffer("stats-calls", pixels * 4, storage | wgpu::BufferUsages::COPY_SRC);
        // The grid evaluator's two bindings share the group, and have to be bound to something.
        let grid_none = buffer("stats-no-grid", 16, storage);
        let spec_none = buffer("stats-no-spec", 32, storage);
        let back = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let steps_back = buffer("stats-steps-back", pixels * 4, back);
        let exits_back = buffer("stats-exits-back", pixels * 4, back);
        let calls_back = buffer("stats-calls-back", pixels * 4, back);

        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fields"),
            layout: &self.stats_pipeline.get_bind_group_layout(0),
            entries: &self.fields_bindings(&camera_buf, &output, &owners, &extras),
        });
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas"),
            layout: &self.stats_pipeline.get_bind_group_layout(1),
            entries: &self.atlas_bindings(),
        });
        let group2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("march-stats"),
            layout: &self.stats_pipeline.get_bind_group_layout(2),
            entries: &[
                entry(0, steps.as_entire_binding()),
                entry(1, exits.as_entire_binding()),
                entry(2, calls.as_entire_binding()),
                entry(3, grid_none.as_entire_binding()),
                entry(4, spec_none.as_entire_binding()),
            ],
        });

        // **The counting march culls too** (review point 9). It is a second copy of the primary
        // march, and left alone it would never cull -- so every steps-per-ray and calls-per-pixel
        // figure taken under `march-stats` would describe a renderer the viewport does not run,
        // which is the one thing these numbers exist not to do.
        let cull = self.cull_pipeline.as_ref().map(|pipeline| {
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile-cull"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &self.writing_bindings(),
            });
            (pipeline, group)
        });

        // A multiple of the tile, so every band starts on a tile row.
        const BAND: u32 = 48;
        let mut first = 0;
        while first < height {
            let rows = BAND.min(height - first);
            let cull_groups = self.cull_workgroups(width, height, (first, rows));
            self.queue.write_buffer(
                &camera_buf,
                0,
                bytemuck::bytes_of(&camera_uniform(
                    camera,
                    width,
                    height,
                    max_steps,
                    (first, rows),
                    cull_groups > 0,
                    // The primary march's count, which no still's measured pixels touch.
                    0,
                )),
            );
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("march-stats") });
            if let (Some((pipeline, group)), true) = (&cull, cull_groups > 0) {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tile-cull"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group0, &[]);
                pass.set_bind_group(1, group, &[]);
                pass.dispatch_workgroups(cull_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("march-stats"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.stats_pipeline);
                pass.set_bind_group(0, &group0, &[]);
                pass.set_bind_group(1, &group1, &[]);
                pass.set_bind_group(2, &group2, &[]);
                pass.dispatch_workgroups(width.div_ceil(8), rows.div_ceil(8), 1);
            }
            self.queue.submit(Some(encoder.finish()));
            // Waited for band by band, so the queue never holds more than one.
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|_| RenderError::Readback)?;
            first += rows;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&steps, 0, &steps_back, 0, pixels * 4);
        encoder.copy_buffer_to_buffer(&exits, 0, &exits_back, 0, pixels * 4);
        encoder.copy_buffer_to_buffer(&calls, 0, &calls_back, 0, pixels * 4);
        self.queue.submit(Some(encoder.finish()));
        steps_back.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        exits_back.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        calls_back.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;
        let read = |b: &wgpu::Buffer| -> Vec<u32> {
            let view = b.slice(..).get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        let capture = crate::stats::MarchCapture {
            width,
            height,
            cap: max_steps,
            steps: read(&steps_back),
            exits: read(&exits_back),
            calls: read(&calls_back),
        };
        steps_back.unmap();
        exits_back.unmap();
        calls_back.unmap();
        Ok(capture)
    }

    /// The field at `n`³ points spread evenly over a box, corners included, as the GPU reads
    /// it: `values[(z·n + y)·n + x]`. For holding one generated program to another bit for bit
    /// — the pruned pattern loop against the unpruned one — since a generated field runs
    /// nowhere but here. In slabs of the grid, each its own submission and waited for.
    pub fn field_on_grid(
        &self,
        lo: Point3<f64>,
        hi: Point3<f64>,
        n: u32,
    ) -> Result<Vec<f32>, RenderError> {
        let n = n.max(2);
        let count = u64::from(n).pow(3);
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        // A camera is bound because the field group binds one; the grid never reads it, but
        // `scene`'s section plane does, and an all-zero camera is "no section".
        let camera_buf = buffer(
            "grid-camera",
            std::mem::size_of::<CameraUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let output = buffer("grid-output", 16, storage);
        let owners = buffer("grid-owners", 4, storage);
        let extras = buffer("grid-extras", 4, storage);
        let none = [buffer("grid-a", 4, storage), buffer("grid-b", 4, storage), buffer("grid-c", 4, storage)];
        let values = buffer("grid-values", count * 4, storage | wgpu::BufferUsages::COPY_SRC);
        let spec = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid-spec"),
            contents: bytemuck::cast_slice(&[
                lo.x as f32, lo.y as f32, lo.z as f32, n as f32,
                hi.x as f32, hi.y as f32, hi.z as f32, 0.0,
            ]),
            usage: storage,
        });
        let back = buffer(
            "grid-back",
            count * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fields"),
            layout: &self.grid_pipeline.get_bind_group_layout(0),
            entries: &self.fields_bindings(&camera_buf, &output, &owners, &extras),
        });
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas"),
            layout: &self.grid_pipeline.get_bind_group_layout(1),
            entries: &self.atlas_bindings(),
        });
        let group2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid"),
            layout: &self.grid_pipeline.get_bind_group_layout(2),
            entries: &[
                entry(0, none[0].as_entire_binding()),
                entry(1, none[1].as_entire_binding()),
                entry(2, none[2].as_entire_binding()),
                entry(3, values.as_entire_binding()),
                entry(4, spec.as_entire_binding()),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grid") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.grid_pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_bind_group(2, &group2, &[]);
            pass.dispatch_workgroups(n.div_ceil(4), n.div_ceil(4), n.div_ceil(4));
        }
        encoder.copy_buffer_to_buffer(&values, 0, &back, 0, count * 4);
        self.queue.submit(Some(encoder.finish()));
        back.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;
        let out: Vec<f32> = {
            let view = back.slice(..).get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        back.unmap();
        Ok(out)
    }
}

impl Renderer {
    /// What building this renderer cost, part by part.
    pub fn build_cost(&self) -> BuildCost {
        self.cost
    }

    /// The WGSL its pipelines were compiled from, so a caller can tell whether a new program
    /// needs the driver again or is the same shader with different numbers.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The device this renderer draws with, so the lighting pass can be built on the same one
    /// and share its buffers.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Which renderer this is, for a cache that must not outlive it.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

fn vec4(v: Vector3<f64>) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}

/// The baked leaves' texture, as the second group declares it.
fn atlas_texture_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            // Read with `textureLoad` and interpolated by hand, so the texture is never
            // filtered by hardware and a 32-bit float format needs no feature.
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

/// The skip grid: read-only to every march, writable to the pass that fills it.
fn skip_grid_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn entry(binding: u32, resource: wgpu::BindingResource) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry { binding, resource }
}

/// How many bindings the `fields` group has -- the length of `fields_bindings` and of
/// `fields_layout_entries`, which the test in `limit_tests` holds equal.
pub(crate) const FIELD_BINDINGS: usize = 12;

impl Renderer {
    /// Every binding of the `fields` group, in one place, for the two passes that build it:
    /// the plain march above and the shaded frame in `shading.rs`.
    ///
    /// **They used to list the bindings separately**, and when the surface's three buffers
    /// arrived they were added to one list and not the other -- so every example passed, since
    /// the examples march plainly, and the application aborted on its first frame, since it
    /// shades. The four buffers that differ per pass are the arguments; everything the renderer
    /// owns is read here.
    pub(crate) fn fields_bindings<'a>(
        &'a self,
        camera: &'a wgpu::Buffer,
        output: &'a wgpu::Buffer,
        owners: &'a wgpu::Buffer,
        extras: &'a wgpu::Buffer,
    ) -> [wgpu::BindGroupEntry<'a>; FIELD_BINDINGS] {
        [
            entry(0, self.params.as_entire_binding()),
            entry(1, self.curve_points.as_entire_binding()),
            entry(2, self.curve_spans.as_entire_binding()),
            entry(3, self.cage_slabs.as_entire_binding()),
            entry(4, camera.as_entire_binding()),
            entry(5, output.as_entire_binding()),
            entry(6, owners.as_entire_binding()),
            entry(7, self.profile_bakes.as_entire_binding()),
            entry(8, extras.as_entire_binding()),
            entry(9, self.surface_floats.as_entire_binding()),
            entry(10, self.surface_uints.as_entire_binding()),
            entry(11, self.surface_layouts.as_entire_binding()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **An atlas is kept exactly when it would be packed from the same bands, in the same
    /// order.** The same bands, not equal ones: the order is the slabs' layers, and a band made
    /// afresh with the same samples is one this rule declines rather than compares, since
    /// comparing 115 MB of samples a rebuild would cost what keeping saves.
    #[test]
    fn an_atlas_is_kept_exactly_when_its_bands_are_the_same_bands_in_the_same_order() {
        use std::sync::{Arc, Weak};
        let band = |value: f32| {
            Arc::new(BakedField {
                dims: [2, 2, 2],
                lo: Point3::origin(),
                cell: nalgebra::Vector3::repeat(1.0),
                band: 1.0,
                data: vec![value; 8],
                normals: vec![[0.0, 0.0, 1.0]; 8],
            })
        };
        let (a, b) = (band(1.0), band(2.0));
        let packed: Vec<Weak<BakedField>> = [&a, &b].into_iter().map(Arc::downgrade).collect();

        assert!(
            atlas_holds(&packed, &[Arc::clone(&a), Arc::clone(&b)]),
            "the same bands"
        );
        assert!(
            atlas_holds(&[], &[]),
            "no bands either side, which is the one-texel stand-in"
        );
        assert!(
            !atlas_holds(&packed, &[Arc::clone(&b), Arc::clone(&a)]),
            "the same bands reordered, which moves every slab's layer"
        );
        assert!(!atlas_holds(&packed, &[Arc::clone(&a)]), "one band fewer");
        assert!(
            !atlas_holds(&packed, &[Arc::clone(&a), Arc::clone(&b), Arc::clone(&a)]),
            "one band more"
        );
        assert!(
            !atlas_holds(&packed, &[Arc::new((*a).clone()), Arc::clone(&b)]),
            "a band made afresh with the same samples"
        );
        // A band that has gone since: what the kept reference still holds is its allocation, so
        // nothing made later can sit at its address and be taken for it.
        let gone = {
            let c = band(3.0);
            Arc::downgrade(&c)
        };
        assert!(
            !atlas_holds(&[gone], &[band(3.0)]),
            "a band that has gone, and a new one"
        );
    }

    /// The shader's walk of the tree, in Rust, so the tree can be held to the linear walk
    /// without a GPU: the same boxes, the same pruning, the same tie rule.
    fn nearest_by_tree(points: &[[f32; 4]], nodes: &[[f32; 4]], p: [f32; 3]) -> (u32, f32) {
        let hit = |i: u32| -> (f32, f32) {
            let a = points[i as usize];
            let b = points[i as usize + 1];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let denom = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
            let mut t = 0.0;
            if denom > 1e-18 {
                let pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                t = ((pa[0] * ab[0] + pa[1] * ab[1] + pa[2] * ab[2]) / denom).clamp(0.0, 1.0);
            }
            let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
            let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            (t, d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        };
        let mut best = (0u32, 0.0f32, 1e30f32);
        // The greedy descent to one leaf for a bound, as the shader makes it.
        let outside_of = |n: usize| -> f32 {
            let (lo, hi) = (nodes[2 * n], nodes[2 * n + 1]);
            let mut inside = f32::MIN;
            let mut out = 0.0f32;
            for k in 0..3 {
                let d = (lo[k] - p[k]).max(p[k] - hi[k]);
                inside = inside.max(d);
                out += d.max(0.0) * d.max(0.0);
            }
            out.sqrt() + inside.min(0.0)
        };
        let mut g = 0usize;
        loop {
            let hi = nodes[2 * g + 1];
            if hi[3] >= 0.0 {
                let packed = hi[3] as u32;
                for i in (packed >> 4)..(packed >> 4) + (packed & 15) {
                    let (t, d2) = hit(i);
                    if d2 < best.2 || (d2 == best.2 && i < best.0) {
                        best = (i, t, d2);
                    }
                }
                break;
            }
            let left = g + 1;
            let right = nodes[2 * left][3] as usize;
            g = if outside_of(right) < outside_of(left) { right } else { left };
        }
        let mut n = 0usize;
        while n < nodes.len() / 2 {
            let lo = nodes[2 * n];
            let hi = nodes[2 * n + 1];
            let mut outside = 0.0f32;
            for k in 0..3 {
                let d = (lo[k] - p[k]).max(p[k] - hi[k]).max(0.0);
                outside += d * d;
            }
            if outside > best.2 * (1.0 + 1e-6) {
                n = lo[3] as usize;
                continue;
            }
            if hi[3] >= 0.0 {
                let packed = hi[3] as u32;
                let first = packed >> 4;
                for i in first..first + (packed & 15) {
                    let (t, d2) = hit(i);
                    if d2 < best.2 || (d2 == best.2 && i < best.0) {
                        best = (i, t, d2);
                    }
                }
            }
            n += 1;
        }
        (best.0, best.2)
    }

    /// The tree finds the segment the walk finds — the same one, with the same tie broken the
    /// same way — on a curve that doubles back on itself, so equidistant segments are common,
    /// and every segment sits in exactly one leaf.
    #[test]
    fn the_segment_tree_finds_the_segment_the_walk_finds() {
        // A zigzag in a plane, so a point between two rows is equidistant from two segments.
        let points: Vec<[f32; 4]> = (0..37)
            .map(|i| {
                let x = (i % 2) as f32 * 10.0;
                [x, i as f32 * 3.0, (i as f32 * 0.7).sin(), 0.0]
            })
            .collect();
        let nodes = segment_tree(&points);
        assert!(!nodes.is_empty() && nodes.len() % 2 == 0);

        // Every segment in exactly one leaf, and every leaf's box holds its segments.
        let mut seen = vec![0u32; points.len() - 1];
        for n in 0..nodes.len() / 2 {
            let (lo, hi) = (nodes[2 * n], nodes[2 * n + 1]);
            if hi[3] >= 0.0 {
                let packed = hi[3] as usize;
                let first = packed >> 4;
                let count = packed & 15;
                for i in first..first + count {
                    seen[i] += 1;
                    for p in [points[i], points[i + 1]] {
                        for k in 0..3 {
                            assert!(lo[k] <= p[k] && p[k] <= hi[k], "leaf {n} misses segment {i}");
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s == 1), "{seen:?}");

        // Against the walk, on a grid of query points that lands on the ties.
        let mut rng = 12345u64;
        for _ in 0..2000 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = |shift: u32| ((rng >> shift) & 0xffff) as f32 / 65535.0;
            let p = [f(8) * 30.0 - 10.0, f(24) * 130.0 - 10.0, f(40) * 4.0 - 2.0];
            let mut walk = (0u32, 1e30f32);
            for i in 0..points.len() as u32 - 1 {
                let a = points[i as usize];
                let b = points[i as usize + 1];
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let denom = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                let pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let t = ((pa[0] * ab[0] + pa[1] * ab[1] + pa[2] * ab[2]) / denom).clamp(0.0, 1.0);
                let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
                let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 < walk.1 {
                    walk = (i, d2);
                }
            }
            let tree = nearest_by_tree(&points, &nodes, p);
            assert_eq!(tree, walk, "at {p:?}");
        }
        // Too few segments for a tree: the shader walks.
        assert!(segment_tree(&points[..2]).is_empty());
    }

    /// The traced view has to agree with whatever else the application draws, and the
    /// application draws through `look_at_rh`. Anything else puts the geometry in a different
    /// place from the overlays — which is how this was noticed: a curve floating clear of the
    /// pipe swept along it.
    #[test]
    fn the_basis_matches_the_view_matrix_an_application_would_build() {
        let cases = [
            // An ordinary three-quarter view.
            (
                Point3::new(300.0, 200.0, 150.0),
                Point3::origin(),
                Vector3::z(),
            ),
            // Straight down, where deriving `right` from world up degenerates.
            (Point3::new(0.0, 0.0, 400.0), Point3::origin(), Vector3::y()),
            // Rolled: an orbit camera that has accumulated twist carries an up like this, and
            // re-deriving one from a world axis would quietly ignore the roll.
            (
                Point3::new(0.0, -300.0, 0.0),
                Point3::origin(),
                Vector3::new(0.3, 0.0, 0.95).normalize(),
            ),
        ];

        for (eye, target, up) in cases {
            let camera = Camera::looking_at(eye, target, up, (target, 100.0), 45.0, 1.0);
            let view = nalgebra::Matrix4::look_at_rh(&eye, &target, &up);

            // `look_at_rh` holds the basis in its upper-left 3x3, by rows.
            let expect_right = Vector3::new(view[(0, 0)], view[(0, 1)], view[(0, 2)]);
            let expect_up = Vector3::new(view[(1, 0)], view[(1, 1)], view[(1, 2)]);
            let expect_forward = -Vector3::new(view[(2, 0)], view[(2, 1)], view[(2, 2)]);

            assert!(
                (camera.right - expect_right).norm() < 1e-9,
                "right: {:?} vs {expect_right:?}",
                camera.right
            );
            assert!(
                (camera.up - expect_up).norm() < 1e-9,
                "up: {:?} vs {expect_up:?}",
                camera.up
            );
            assert!(
                (camera.forward - expect_forward).norm() < 1e-9,
                "forward: {:?} vs {expect_forward:?}",
                camera.forward
            );
        }
    }

    /// Looking almost straight down must not be a special case.
    ///
    /// The bug this replaces derived `right` from world up and switched to a different
    /// reference axis when that degenerated, so the basis *jumped* as the view crossed the
    /// threshold and the model appeared to leap sideways. Two cameras a thousandth of a degree
    /// apart must produce two nearly identical bases.
    #[test]
    fn the_basis_does_not_jump_as_the_view_crosses_straight_down() {
        let up = Vector3::y();
        let mut previous: Option<Camera> = None;

        // Sweep through exactly vertical rather than stopping either side of it.
        for i in -5..=5 {
            let tilt = i as f64 * 1e-5;
            let eye = Point3::new(tilt * 400.0, 0.0, 400.0);
            let camera = Camera::looking_at(eye, Point3::origin(), up, (Point3::origin(), 100.0), 45.0, 1.0);

            if let Some(prev) = previous {
                let moved = (camera.right - prev.right).norm()
                    + (camera.up - prev.up).norm()
                    + (camera.forward - prev.forward).norm();
                assert!(
                    moved < 1e-3,
                    "the basis moved by {moved} across a 1e-5 change in view direction"
                );
            }
            previous = Some(camera);
        }
    }
}
