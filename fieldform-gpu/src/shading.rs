// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The lighting pass, and the frame it lights.
//!
//! [`Renderer::render`] traces a frame and reads it back to the CPU, which is what the
//! comparison examples and tests want. This is the other consumer: the application, which wants
//! the frame *lit* and *shown* and never wants to see a pixel of it on the CPU. The march writes
//! the G-buffer into storage buffers that stay on the GPU, [`SHADE`] turns them into a
//! display-encoded image in a texture, and egui samples that texture. Nothing is read back.
//!
//! That is what lifted the viewport's 720-pixel cap: the cap existed because the old path copied
//! every frame down and up again, and `traced/mod.rs` measured that copy as the whole cost — the
//! dearest field cost what the cheapest one did, to within 5%.
//!
//! # What persists and what does not
//!
//! [`ShadingResources`] outlives any one [`Renderer`]. The lighting pipeline and the environment
//! it samples depend on nothing in the document, and a renderer is rebuilt on every change to
//! the graph's shape; rebuilding the environment with it would bake the studio again for every
//! edit. The frame targets live here for the same reason — a 1080p G-buffer is fifty megabytes,
//! and reallocating it per edit would be the pause you noticed on every keystroke.

use crate::environment::{EnvironmentChoice, EnvironmentMaps};
use crate::flake::FlakeMap;
use crate::render::{camera_uniform, Camera, RenderError, Renderer};
use fieldform_scene::Material;
use nalgebra::Vector3;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The lighting pass: the shared camera, then the shader that reads what the march wrote.
pub(crate) const SHADE: &str = concat!(include_str!("camera.wgsl"), include_str!("shade.wgsl"));

/// One directional light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Unit direction *toward* the light, in world space.
    pub direction: Vector3<f64>,
    /// Linear RGB.
    pub color: [f32; 3],
    pub intensity: f32,
}

/// A draft-angle read of the surface, laid over the finished image.
///
/// The first surface-analysis overlay, and the shape the rest take: a function of the geometric
/// normal, evaluated in the lighting pass and composited in display space so exposure cannot
/// reach it. Each face is coloured by whether it has enough draft to leave a mould pulled along
/// `pull` — `pass` where it does, `fail` where it does not, vertical or undercut — and the line
/// between them is blurred over `tolerance_deg` either side of `angle_deg`, so on a curved face
/// it is a soft band rather than a contour that would read as a crease. The other mould half is
/// the same read with the pull reversed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draft {
    /// Unit direction the mould pulls along, in world space.
    pub pull: Vector3<f64>,
    /// The draft a face needs, in degrees from vertical along the pull.
    pub angle_deg: f32,
    /// How far either side of the angle the boundary blurs, in degrees.
    pub tolerance_deg: f32,
    /// Display-encoded colours for a face with enough draft and one without, and how strongly
    /// they are laid over the shading — soft, so the material still reads through.
    pub pass: [f32; 3],
    pub fail: [f32; 3],
    pub opacity: f32,
}

impl Draft {
    pub fn new(pull: Vector3<f64>, angle_deg: f32, tolerance_deg: f32) -> Self {
        Self {
            pull,
            angle_deg,
            tolerance_deg,
            pass: [0.35, 0.6, 1.0],
            fail: [1.0, 0.4, 0.35],
            opacity: 0.55,
        }
    }
}

/// How a frame is lit and shown: everything about the picture that is not the geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    pub material: Material,
    /// Key first — it is the one the march casts a shadow for.
    pub lights: [Light; 3],
    /// Which body is tinted as selected, by its position among the part's roots.
    pub selected: Option<u32>,
    /// Scene-referred scale applied before the tonemap. One is neutral.
    pub exposure: f32,
    /// Scale on the environment's contribution, so the dome and the analytic lights can be
    /// balanced without rebaking either.
    pub environment_intensity: f32,
    /// What a miss shows, in linear RGB.
    pub background: [f32; 3],
    /// The selection outline's colour, **display-encoded**: it is interface, drawn over the
    /// tonemapped image, so exposure never dims it and it matches the panel's own yellow.
    pub outline: [f32; 3],
    /// The outline's radius in texels of the frame being presented. The caller sets it from
    /// how much the frame is scaled to the screen, so the line is the same width on screen
    /// whether the frame is a 720-pixel drag or a full-size still.
    pub outline_radius: u32,
    /// A draft-angle read over the shading, or none.
    pub draft: Option<Draft>,
    /// Paint each pixel's normal instead of lighting it (backlog item 116).
    ///
    /// Read by the lighting pass in place of the whole shade, so the lights, the dome, the
    /// clearcoat, the exposure and the tonemap are all skipped rather than configured away. See
    /// `shade.wgsl` for why the tonemap in particular has to be skipped and not merely neutral.
    pub normals: bool,
}

impl Look {
    /// The three-light studio rig, placed relative to the camera so the model is lit from
    /// wherever it is being looked at — which is how the viewport has always been lit and what
    /// keeps a part legible from every side without moving a light.
    ///
    /// Key over the viewer's left shoulder, a dimmer fill low on the right, and a rim behind and
    /// above so the silhouette catches an edge. The directions point *toward* each light.
    pub(crate) fn studio_lights(
        forward: Vector3<f64>,
        right: Vector3<f64>,
        up: Vector3<f64>,
    ) -> [Light; 3] {
        let key = (-forward * 0.6 + up * 0.75 - right * 0.5).normalize();
        let fill = (-forward * 0.55 + up * 0.1 + right * 0.85).normalize();
        let rim = (forward * 0.7 + up * 0.6 + right * 0.25).normalize();
        [
            Light { direction: key, color: [1.0, 0.98, 0.95], intensity: 1.8 },
            Light { direction: fill, color: [0.9, 0.94, 1.0], intensity: 0.45 },
            Light { direction: rim, color: [1.0, 1.0, 1.0], intensity: 1.1 },
        ]
    }

    /// The look for an environment: the studio's lamps where it has them, the room alone
    /// where it does not.
    pub fn for_environment(
        choice: EnvironmentChoice,
        material: Material,
        forward: Vector3<f64>,
        right: Vector3<f64>,
        up: Vector3<f64>,
    ) -> Self {
        let mut look = Self::studio(material, forward, right, up);
        if !choice.has_lamps() {
            for light in &mut look.lights {
                light.intensity = 0.0;
            }
        }
        // Set here rather than by the caller, so the one function that turns a choice into a look
        // answers for all of it and a new call site cannot forget this one.
        look.normals = choice.paints_normals();
        look
    }

    /// A look with the studio rig and the viewport's own defaults, for a camera basis.
    pub fn studio(
        material: Material,
        forward: Vector3<f64>,
        right: Vector3<f64>,
        up: Vector3<f64>,
    ) -> Self {
        Self {
            material,
            lights: Self::studio_lights(forward, right, up),
            selected: None,
            exposure: 1.0,
            environment_intensity: 1.0,
            // The dark neutral the viewport has always had, decoded from its display value
            // (24, 26, 30) so the background does not shift when the shading does.
            background: [0.0093, 0.0107, 0.0137],
            outline: [1.0, 0.85, 0.25],
            outline_radius: 2,
            draft: None,
            normals: false,
        }
    }

    /// `rows` is the band the lighting pass is to light: its first row and how many. See
    /// `present_rows` for why that is not the band the march has just written.
    fn uniform(&self, accumulated: u32, rows: (u32, u32)) -> ShadingUniform {
        let m = self.material.clone().normalised();
        let light_dirs = self.lights.map(|l| {
            let d = l.direction.normalize();
            [d.x as f32, d.y as f32, d.z as f32, l.intensity]
        });
        let light_colors = self.lights.map(|l| [l.color[0], l.color[1], l.color[2], 0.0]);
        ShadingUniform {
            base: [
                m.base_color[0] as f32,
                m.base_color[1] as f32,
                m.base_color[2] as f32,
                m.metallic as f32,
            ],
            surface: [
                m.roughness as f32,
                m.clearcoat as f32,
                m.clearcoat_roughness as f32,
                m.anisotropy as f32,
            ],
            // `w` was the spare, which is what this flag is for: no new binding, no new uniform,
            // and a pass that already reads `look` every pixel.
            look: [
                m.flake as f32,
                self.exposure,
                self.environment_intensity,
                if self.normals { 1.0 } else { 0.0 },
            ],
            light_dirs,
            light_colors,
            background: [self.background[0], self.background[1], self.background[2], 1.0],
            outline: [self.outline[0], self.outline[1], self.outline[2], 0.0],
            frame: [
                self.selected.unwrap_or(u32::MAX),
                accumulated,
                self.outline_radius,
                // Sixteen bits each, which a frame's height fits in with room to spare.
                (rows.0 & 0xffff) | (rows.1.min(0xffff) << 16),
            ],
            draft_dir: match self.draft {
                Some(d) => {
                    let p = d.pull.try_normalize(1e-12).unwrap_or_else(Vector3::z);
                    [p.x as f32, p.y as f32, p.z as f32, 1.0]
                }
                None => [0.0, 0.0, 1.0, 0.0],
            },
            draft_params: match self.draft {
                Some(d) => [d.angle_deg, d.tolerance_deg, d.opacity, 0.0],
                None => [0.0; 4],
            },
            draft_pass: match self.draft {
                Some(d) => [d.pass[0], d.pass[1], d.pass[2], 0.0],
                None => [0.0; 4],
            },
            draft_fail: match self.draft {
                Some(d) => [d.fail[0], d.fail[1], d.fail[2], 0.0],
                None => [0.0; 4],
            },
        }
    }
}

/// The `Shading` block in `shade.wgsl`, field for field.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadingUniform {
    base: [f32; 4],
    surface: [f32; 4],
    look: [f32; 4],
    light_dirs: [[f32; 4]; 3],
    light_colors: [[f32; 4]; 3],
    background: [f32; 4],
    outline: [f32; 4],
    frame: [u32; 4],
    draft_dir: [f32; 4],
    draft_params: [f32; 4],
    draft_pass: [f32; 4],
    draft_fail: [f32; 4],
}

/// The sub-pixel offset for the `frame`th sample of a still, in pixels.
///
/// The Halton sequence in bases 2 and 3: low-discrepancy, so eight frames cover the pixel about
/// as evenly as eight frames can, and deterministic, so the same still converges to the same
/// image. Frame zero is the pixel centre — the frame a drag shows must not be shifted.
pub fn halton_jitter(frame: u32) -> [f64; 2] {
    if frame == 0 {
        return [0.0, 0.0];
    }
    fn halton(mut index: u32, base: u32) -> f64 {
        let mut f = 1.0;
        let mut r = 0.0;
        while index > 0 {
            f /= base as f64;
            r += f * (index % base) as f64;
            index /= base;
        }
        r
    }
    [halton(frame, 2) - 0.5, halton(frame, 3) - 0.5]
}

/// What one submission of [`Renderer::present`] cost, reported once the GPU has finished it.
///
/// Two clocks, because the exact one is not always there. The GPU's own timestamps bracket
/// each pass and are the number every budget in
/// the performance-stability-loop-spec design note is stated in; they need
/// `wgpu::Features::TIMESTAMP_QUERY`, which the application requests only when the adapter
/// offers it. The wall clock runs from the submit to the queue reporting the work done, which
/// every device can do and which always overstates: the report arrives when the device is next
/// polled, so it carries up to a frame of waiting the GPU never spent. A line printed from it
/// says `(wall)` so nobody reads it as a GPU time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmissionCost {
    /// Submit to work-done, on the CPU's clock. Always measured.
    pub wall_ms: f32,
    /// The march pass alone, from the GPU's timestamps; `None` where the device has none.
    pub march_ms: Option<f32>,
    /// The lighting pass alone, likewise.
    pub shade_ms: Option<f32>,
}

impl SubmissionCost {
    /// The GPU's own total, when it was measured.
    pub(crate) fn gpu_ms(&self) -> Option<f32> {
        Some(self.march_ms? + self.shade_ms?)
    }

    /// The best number there is: the GPU's when it was measured, the wall clock otherwise.
    pub fn ms(&self) -> f32 {
        self.gpu_ms().unwrap_or(self.wall_ms)
    }
}

/// Names one submission, so its cost can be matched back to it when the GPU reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionTicket(u64);

/// Milliseconds between two timestamps `period` nanoseconds apart per tick, or `None` for a
/// pair the GPU wrote out of order — which a reset or a wrapped counter can produce, and which
/// would otherwise read as a frame that took two hundred years.
fn ms_between(begin: u64, end: u64, period_ns: f32) -> Option<f32> {
    if end < begin {
        return None;
    }
    Some(((end - begin) as f64 * f64::from(period_ns) / 1e6) as f32)
}

/// What a timing slot is waiting on. Written by wgpu's callbacks, on whichever thread polls.
#[derive(Clone, Copy, Default)]
struct Arrival {
    /// Set by `on_submitted_work_done`, with the wall time since the submit.
    wall_ms: Option<f32>,
    /// Set by `map_async`: mapped, or the map failed and the wall clock is all there is.
    mapped: Option<bool>,
}

/// One submission's timestamps, on their way back to the CPU.
struct TimingSlot {
    /// This slot's own four queries, or `None` on a device without `TIMESTAMP_QUERY`.
    ///
    /// **One set per slot, not one shared.** Two submissions in flight at once — a draft and
    /// the still band that followed it — both wrote and resolved the same four queries, and a
    /// queue does not promise that one command buffer's resolve runs before the next one's
    /// reset: a band came back as 0.0 ms, and the band that was sized from that number was
    /// the whole frame in one 634 ms submission, which is the thing bands exist to prevent.
    queries: Option<wgpu::QuerySet>,
    /// The four timestamps, resolved out of the query set.
    resolve: wgpu::Buffer,
    /// The same four, in a buffer the CPU can map.
    readback: wgpu::Buffer,
    /// The submission this slot is measuring, or `None` while it is free.
    ticket: Option<SubmissionTicket>,
    arrival: Arc<Mutex<Arrival>>,
}

/// How many submissions may be in flight with their timestamps unread. A ring rather than one
/// slot because a still is several submissions before any of them is polled for, and a
/// submission that finds no free slot goes unmeasured rather than waited for.
const TIMING_SLOTS: usize = 4;
/// March begin and end, then shade begin and end.
const TIMESTAMPS: u32 = 4;

/// Measures what each submission costs without ever waiting for one.
///
/// **Asynchronous by construction.** The application path may never block on the GPU — a
/// stall here would be the very thing the measurement exists to remove — so every readback is
/// mapped with a callback and read on a later call, one frame of latency being nothing against
/// a number that was previously not there at all.
struct Timing {
    /// The nanoseconds one timestamp tick is worth, or `None` on a device that has no
    /// `TIMESTAMP_QUERY` — the wall clock alone then, and every line says so.
    period: Option<f32>,
    slots: Vec<TimingSlot>,
    next_ticket: u64,
    completed: Vec<(SubmissionTicket, SubmissionCost)>,
}

impl Timing {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let period = timestamps.then(|| queue.get_timestamp_period());
        let bytes = u64::from(TIMESTAMPS) * 8;
        let slots = (0..TIMING_SLOTS)
            .map(|_| TimingSlot {
                queries: timestamps.then(|| {
                    device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("frame-timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: TIMESTAMPS,
                    })
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamps-resolve"),
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamps-readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                ticket: None,
                arrival: Arc::default(),
            })
            .collect();
        Self { period, slots, next_ticket: 1, completed: Vec::new() }
    }

    /// A ticket for the submission about to be encoded, and the slot that will time it if one
    /// is free.
    fn begin(&mut self) -> (SubmissionTicket, Option<usize>) {
        let ticket = SubmissionTicket(self.next_ticket);
        self.next_ticket += 1;
        let slot = self.slots.iter().position(|s| s.ticket.is_none());
        if let Some(i) = slot {
            self.slots[i].ticket = Some(ticket);
            *self.slots[i].arrival.lock().expect("no panic holds this") = Arrival::default();
        }
        (ticket, slot)
    }

    /// The timestamp writes for the pass whose begin and end are query indices `begin` and
    /// `begin + 1`, or `None` when nothing is timing this submission.
    fn writes(&self, slot: Option<usize>, begin: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let set = self.slots[slot?].queries.as_ref()?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        })
    }

    /// Copies the timestamps into the slot's own buffers, after the passes and before the
    /// submit. Nothing when the device cannot write timestamps.
    fn resolve(&self, slot: Option<usize>, encoder: &mut wgpu::CommandEncoder) {
        let Some(s) = slot.map(|i| &self.slots[i]) else { return };
        let Some(set) = s.queries.as_ref() else { return };
        encoder.resolve_query_set(set, 0..TIMESTAMPS, &s.resolve, 0);
        encoder.copy_buffer_to_buffer(&s.resolve, 0, &s.readback, 0, s.resolve.size());
    }

    /// Registers what will tell this slot the submission has finished. Called straight after
    /// the submit, since both callbacks attach to the most recent one.
    fn arm(&mut self, slot: Option<usize>, queue: &wgpu::Queue) {
        let Some(i) = slot else { return };
        let started = Instant::now();
        let s = &self.slots[i];
        let arrival = Arc::clone(&s.arrival);
        queue.on_submitted_work_done(move || {
            if let Ok(mut a) = arrival.lock() {
                a.wall_ms = Some(started.elapsed().as_secs_f32() * 1e3);
            }
        });
        if s.queries.is_some() {
            let arrival = Arc::clone(&s.arrival);
            s.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if let Ok(mut a) = arrival.lock() {
                    a.mapped = Some(result.is_ok());
                }
            });
        }
    }

    /// Reads every slot whose submission has finished, freeing it.
    fn collect(&mut self) {
        let period = self.period;
        for slot in &mut self.slots {
            let Some(ticket) = slot.ticket else { continue };
            let arrival = *slot.arrival.lock().expect("no panic holds this");
            let Some(wall_ms) = arrival.wall_ms else { continue };
            let (mut march_ms, mut shade_ms) = (None, None);
            match (period, arrival.mapped) {
                (Some(period), Some(true)) => {
                    {
                        let view = slot.readback.slice(..).get_mapped_range();
                        let t: &[u64] = bytemuck::cast_slice(&view);
                        march_ms = ms_between(t[0], t[1], period);
                        shade_ms = ms_between(t[2], t[3], period);
                    }
                    slot.readback.unmap();
                }
                // The map failed — a device on its way out — and the wall clock is all there is.
                (Some(_), Some(false)) => {}
                // The work is done and the map is not yet: next time.
                (Some(_), None) => continue,
                (None, _) => {}
            }
            self.completed.push((ticket, SubmissionCost { wall_ms, march_ms, shade_ms }));
            slot.ticket = None;
        }
    }
}

/// The buffers one frame is traced into and the texture it is shown from.
struct FrameTargets {
    /// Which allocation this is, so a caller holding a handle to the texture can tell it went
    /// stale. Never reused.
    generation: u64,
    width: u32,
    height: u32,
    gbuffer: wgpu::Buffer,
    owners: wgpu::Buffer,
    extras: wgpu::Buffer,
    hdr: wgpu::Buffer,
    present: wgpu::Texture,
    /// The texture as the lighting pass writes it and as egui reads it: plain 8-bit holding
    /// display-encoded values.
    ///
    /// **Not an sRGB view**, although the bytes are sRGB-encoded. egui's texture shader treats
    /// every texture as already encoded and passes the sample straight through to its gamma
    /// framebuffer (`egui.wgsl`: "we expect normal textures that are NOT sRGB-aware"), and its
    /// own textures are `Rgba8Unorm` for that reason. An sRGB view here made the sampler decode
    /// the bytes once and egui encode them never, so the whole viewport — background included —
    /// displayed at the *linear* value of its colour: (24, 26, 30) came out as (2, 3, 3). That
    /// was reported as the background having changed, which it had.
    view: wgpu::TextureView,
}

impl FrameTargets {
    fn new(device: &wgpu::Device, generation: u64, width: u32, height: u32) -> Self {
        let pixels = u64::from(width) * u64::from(height);
        let storage = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let present = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("present"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = present.create_view(&wgpu::TextureViewDescriptor {
            label: Some("present"),
            ..Default::default()
        });
        Self {
            generation,
            width,
            height,
            gbuffer: storage("gbuffer", pixels * 16),
            owners: storage("owners", pixels * 4),
            extras: storage("extras", pixels * 4),
            hdr: storage("hdr", pixels * 16),
            present,
            view,
        }
    }
}

/// One frame's bind groups: the march's, the atlas's and the lighting pass's, for one renderer
/// and one set of targets.
///
/// Made once per frame rather than once per submission. A frame is several submissions now —
/// one per band of rows — and nothing in these groups changes between bands: the camera is one
/// persistent buffer that is rewritten, not rebound, so the band's rows travel as a uniform
/// value and the groups stay as they were. Keyed by the renderer and the targets' generation,
/// which are the two things that can change what they bind.
struct FrameBindings {
    renderer: u64,
    generation: u64,
    march: wgpu::BindGroup,
    atlas: wgpu::BindGroup,
    /// The second group as the per-tile cull sees it -- the same mask buffer, writable, at its
    /// own binding. `None` where the program does not cull.
    cull: Option<wgpu::BindGroup>,
    shade: wgpu::BindGroup,
}

/// The lighting pipeline, the environment it samples, and the frame it lights.
pub struct ShadingResources {
    pipeline: wgpu::ComputePipeline,
    environment_layout: wgpu::BindGroupLayout,
    /// The flake normal map and its repeating sampler, shared by every environment's bind
    /// group: the flake is a property of the paint, not of the room.
    flake: (wgpu::TextureView, wgpu::Sampler),
    environment: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    /// The camera, written each frame rather than allocated each frame.
    camera: wgpu::Buffer,
    /// One set of targets per frame size in use — the drag size and the still size — so
    /// switching tiers costs nothing.
    ///
    /// **This used to be one set, reallocated on every switch**, and the soak (`soak.rs`)
    /// measured what that did: live allocations stayed flat at ~95 MB while the allocator's
    /// reserved memory climbed from 201 MB to 470 MB, because freeing and re-carving 90 MB of
    /// buffers every interaction fragments its blocks until it opens another. Nothing leaked and
    /// the card still filled. A session now allocates its two sets once per window size and
    /// never again.
    targets: Vec<FrameTargets>,
    /// Which of `targets` the last frame used.
    current: usize,
    /// Handed to each allocation in turn, so no two sets ever share a generation.
    next_generation: u64,
    /// What each submission cost, read back without waiting.
    timing: Timing,
    /// The bind groups of the frame being presented, kept across its bands.
    bindings: Option<FrameBindings>,
}

impl ShadingResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        maps: &EnvironmentMaps,
    ) -> Result<Self, RenderError> {
        crate::codegen::validated(SHADE).map_err(|e| RenderError::Shader(e.to_string()))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade"),
            source: wgpu::ShaderSource::Wgsl(SHADE.into()),
        });

        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shade-frame"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let texture_entry = |binding: u32, dimension: wgpu::TextureViewDimension| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: dimension,
                    multisampled: false,
                },
                count: None,
            }
        };
        let environment_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shade-environment"),
                entries: &[
                    texture_entry(0, wgpu::TextureViewDimension::Cube),
                    texture_entry(1, wgpu::TextureViewDimension::Cube),
                    texture_entry(2, wgpu::TextureViewDimension::D2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    texture_entry(4, wgpu::TextureViewDimension::D2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shade"),
            bind_group_layouts: &[Some(&frame_layout), Some(&environment_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shade"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("shade"),
            compilation_options: Default::default(),
            cache: None,
        });

        let flake = upload_flake(device, queue, &FlakeMap::generate(256, 4, 6.0, 7));
        let environment =
            environment_bind_group(device, queue, &environment_layout, maps, &flake);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shading"),
            size: std::mem::size_of::<ShadingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<crate::render::CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            environment_layout,
            flake,
            environment,
            uniform,
            camera,
            targets: Vec::new(),
            current: 0,
            next_generation: 1,
            timing: Timing::new(device, queue),
            bindings: None,
        })
    }

    /// Whether a submission this timed is still running on the GPU.
    ///
    /// For the caller that must not queue a second draft behind a first (loop spec P10): one
    /// march in flight is a frame of latency, several is a drag that keeps moving after the
    /// hand has stopped. Fails open — a submission that found every timing slot busy is not
    /// known about — so the worst case is the old behaviour, never a viewport that stops.
    pub fn busy(&self) -> bool {
        self.timing.slots.iter().any(|s| {
            s.ticket.is_some()
                && s.arrival.lock().map_or(false, |a| a.wall_ms.is_none())
        })
    }

    /// Whether the device writes GPU timestamps, so a cost is the GPU's own rather than the
    /// wall clock's. Decided by the features the device was created with.
    pub fn timestamps_available(&self) -> bool {
        self.timing.period.is_some()
    }

    /// What the GPU has finished since this was last asked, each cost against the ticket
    /// [`Renderer::present`] handed out for it. Never waits: a submission still running is
    /// reported the next time, and one that found every timing slot busy is never reported.
    pub fn costs(&mut self) -> Vec<(SubmissionTicket, SubmissionCost)> {
        self.timing.collect();
        std::mem::take(&mut self.timing.completed)
    }

    /// Lights every frame from now on with `maps` instead. The pipeline and the frame targets
    /// are untouched — an environment is three textures, and swapping one is an upload.
    pub fn set_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        maps: &EnvironmentMaps,
    ) {
        self.environment =
            environment_bind_group(device, queue, &self.environment_layout, maps, &self.flake);
    }

    /// Selects the targets of `width` by `height`, allocating them if no set has that size.
    ///
    /// Two sets are kept: the drag size and the still size. A third size — a window resize —
    /// replaces whichever set was not used last, so a resize costs one allocation and a
    /// session at a steady window costs none after its first two frames.
    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(i) = self.targets.iter().position(|t| t.width == width && t.height == height)
        {
            self.current = i;
            return;
        }
        let fresh = FrameTargets::new(device, self.next_generation, width, height);
        self.next_generation += 1;
        if self.targets.len() < 2 {
            self.targets.push(fresh);
            self.current = self.targets.len() - 1;
        } else {
            let stale = 1 - self.current.min(1);
            self.targets[stale] = fresh;
            self.current = stale;
        }
    }

    fn targets(&self) -> Option<&FrameTargets> {
        self.targets.get(self.current)
    }

    /// The lit frame as a texture to sample, once a frame has been presented: 8-bit,
    /// display-encoded, in the plain format egui expects its textures in.
    pub fn display_view(&self) -> Option<&wgpu::TextureView> {
        self.targets().map(|t| &t.view)
    }

    /// Identifies the texture [`Self::display_view`] currently returns. Two frames with the
    /// same generation show the same texture; a generation no longer in
    /// [`Self::live_generations`] names a texture that has been freed.
    pub fn generation(&self) -> u64 {
        self.targets().map_or(0, |t| t.generation)
    }

    /// The generations of every set still allocated, so a caller can drop handles to the rest.
    pub fn live_generations(&self) -> impl Iterator<Item = u64> + '_ {
        self.targets.iter().map(|t| t.generation)
    }
}

/// The flake normal map with its mips, and a sampler that repeats it.
fn upload_flake(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    map: &FlakeMap,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("flake"),
        size: wgpu::Extent3d { width: map.size, height: map.size, depth_or_array_layers: 1 },
        mip_level_count: map.levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (mip, level) in map.levels.iter().enumerate() {
        let size = map.size >> mip;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(level),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
    }
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("flake"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    (texture.create_view(&Default::default()), sampler)
}

/// The three environment lookups the lighting pass samples, uploaded and bound with the flake.
fn environment_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    maps: &EnvironmentMaps,
    flake: &(wgpu::TextureView, wgpu::Sampler),
) -> wgpu::BindGroup {
    let specular = upload_cubemap(device, queue, "specular", &maps.specular);
    let irradiance =
        upload_cubemap(device, queue, "irradiance", std::slice::from_ref(&maps.irradiance));
    let lut = upload_lut(device, queue, maps);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("environment"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shade-environment"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&specular),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&irradiance),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&lut),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&flake.0),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&flake.1),
            },
        ],
    })
}

/// One cube texture with a level per entry of `levels`, as 16-bit floats.
fn upload_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    levels: &[crate::environment::CubeLevel],
) -> wgpu::TextureView {
    let base = levels[0].size;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: base, height: base, depth_or_array_layers: 6 },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (mip, level) in levels.iter().enumerate() {
        for (face, texels) in level.faces.iter().enumerate() {
            let data: Vec<u16> = texels
                .iter()
                .flat_map(|t| t.iter().map(|&v| half::f16::from_f32(v).to_bits()))
                .collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level.size * 8),
                    rows_per_image: Some(level.size),
                },
                wgpu::Extent3d { width: level.size, height: level.size, depth_or_array_layers: 1 },
            );
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

fn upload_lut(device: &wgpu::Device, queue: &wgpu::Queue, maps: &EnvironmentMaps) -> wgpu::TextureView {
    let size = maps.lut_size;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brdf-lut"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let data: Vec<u16> = maps
        .brdf_lut
        .iter()
        .flat_map(|t| t.iter().map(|&v| half::f16::from_f32(v).to_bits()))
        .collect();
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
    );
    texture.create_view(&Default::default())
}

/// A displayed frame read back for inspection, as sRGB-encoded RGBA rows.
pub struct DisplayImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Renderer {
    /// Traces and lights one frame into textures that stay on the GPU.
    ///
    /// `accumulated` is how many jittered frames of this same view are already in the HDR
    /// buffer; zero writes afresh. The caller decides the tier — a drag passes zero with the
    /// camera's occlusion off, a still counts up with it on — and this only does what it is
    /// told, so the policy lives in one place.
    ///
    /// The ticket names this submission in [`ShadingResources::costs`], which says what it
    /// cost once the GPU has finished it.
    ///
    /// The whole frame in one submission. The application traces a still in bands instead —
    /// [`Self::present_rows`] — because one packet for a whole frame is what a slow card is
    /// reset over; this is the form the examples and a drag use.
    #[allow(clippy::too_many_arguments)]
    pub fn present(
        &self,
        shading: &mut ShadingResources,
        camera: &Camera,
        width: u32,
        height: u32,
        max_steps: u32,
        look: &Look,
        accumulated: u32,
    ) -> Result<SubmissionTicket, RenderError> {
        self.present_rows(shading, camera, width, height, max_steps, look, accumulated, 0..height)
    }

    /// Traces and lights the rows `rows` of a `width` by `height` frame, as one submission.
    ///
    /// **The unit the loop spec's S2 schedules on.** A Maxwell card cannot preempt a compute
    /// dispatch, and Windows resets a driver whose packet runs past about two seconds — so a
    /// frame is several submissions of a band of rows each, sized by the caller from what the
    /// last one cost, and no single packet approaches the limit. Everything about the frame
    /// other than the rows is the same across its bands: the bind groups are made once per
    /// frame and reused, and the band travels in the camera uniform, which is rewritten — not
    /// rebound — before each submission. Rows past the frame are ignored.
    #[allow(clippy::too_many_arguments)]
    pub fn present_rows(
        &self,
        shading: &mut ShadingResources,
        camera: &Camera,
        width: u32,
        height: u32,
        max_steps: u32,
        look: &Look,
        accumulated: u32,
        rows: std::ops::Range<u32>,
    ) -> Result<SubmissionTicket, RenderError> {
        let (width, height) = (width.max(1), height.max(1));
        let first = rows.start.min(height);
        let count = rows.end.min(height).saturating_sub(first);
        // **The lighting trails the march by the outline's reach.** The outline looks at the
        // owners of the pixels around it, and at the foot of a band those are rows the march
        // has not written this frame: they hold whatever frame was there before. When that was
        // another view -- zoom into the selected body, let it settle, zoom out -- the rows
        // below every band's edge said "selected" where the rows above said "nothing", and a
        // line of outline was drawn along each edge, spaced as the bands are. It was left in
        // once as "below what the accumulation shows"; at four frames to a still it is a
        // quarter-strength line across the viewport, which is how it was reported.
        //
        // So a band is marched whole and lit short of its foot by the radius, and the rows
        // held back are lit by the next band, once the rows beneath them exist. The last band
        // lights to the end. Every row is lit exactly once a frame, as before, and a frame in
        // bands is the frame whole whatever lay in the buffers -- `overlay_smoke` holds that
        // with another view underneath.
        //
        // **And by a row at least where the occlusion is measured one pixel in four**, since a
        // pixel's fill reads the measured row beneath it, which is the next band's when the pixel
        // is the last of this one (loop spec P5).
        let measured = crate::render::occlusion_pixels(camera, accumulated);
        let reach = look.outline_radius.max(u32::from(measured != 0));
        let lit_first = first.saturating_sub(if first == 0 { 0 } else { reach });
        let lit_last = if first + count >= height { height } else { (first + count).saturating_sub(reach) };
        let lit = (lit_first, lit_last.saturating_sub(lit_first));
        shading.ensure_targets(&self.device, width, height);
        let (ticket, slot) = shading.timing.begin();
        let generation = shading.generation();

        let fresh = !matches!(
            &shading.bindings,
            Some(b) if b.renderer == self.identity() && b.generation == generation
        );
        if fresh {
            let targets = shading.targets().expect("ensured above");
            use crate::render::entry;
            // The same list the plain march binds -- see `Renderer::fields_bindings` for what
            // listing it twice cost.
            let march = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fields"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &self.fields_bindings(
                    &shading.camera,
                    &targets.gbuffer,
                    &targets.owners,
                    &targets.extras,
                ),
            });
            let atlas = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("atlas"),
                layout: &self.pipeline.get_bind_group_layout(1),
                entries: &self.atlas_bindings(),
            });
            // Group 0 is not made again for the cull: both pipelines' layouts come from one
            // `fields_layout`, which is already what lets the still's `occlude` bind the group
            // the march made.
            let cull = self.cull_pipeline().map(|pipeline| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tile-cull"),
                    layout: &pipeline.get_bind_group_layout(1),
                    entries: &self.writing_bindings(),
                })
            });
            let shade = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shade-frame"),
                layout: &shading.pipeline.get_bind_group_layout(0),
                entries: &[
                    entry(0, shading.camera.as_entire_binding()),
                    entry(1, shading.uniform.as_entire_binding()),
                    entry(2, targets.gbuffer.as_entire_binding()),
                    entry(3, targets.owners.as_entire_binding()),
                    entry(4, targets.extras.as_entire_binding()),
                    entry(5, targets.hdr.as_entire_binding()),
                    entry(6, wgpu::BindingResource::TextureView(&targets.view)),
                ],
            });
            shading.bindings = Some(FrameBindings {
                renderer: self.identity(),
                generation,
                march,
                atlas,
                cull,
                shade,
            });
        }
        let bindings = shading.bindings.as_ref().expect("set above");

        // **The per-tile cull, for this band's tile rows, into this band's own encoder.** How
        // many workgroups is also the answer to "did we cull": nought where the program does not,
        // where the band does not start on a tile row, or where the frame is wider than the mask.
        // Whatever it says here is what the uniform tells the march, so the two cannot disagree.
        let cull_groups = self.cull_workgroups(width, height, (first, count));

        // Two writes and no allocations: what a submission costs in memory is nothing, by
        // design. Ordered on the queue ahead of the submit below, so each band's dispatch
        // reads its own rows even when several bands are queued.
        self.queue.write_buffer(
            &shading.camera,
            0,
            bytemuck::bytes_of(&camera_uniform(
                camera,
                width,
                height,
                max_steps,
                (first, count),
                cull_groups > 0,
                measured,
            )),
        );
        self.queue
            .write_buffer(&shading.uniform, 0, bytemuck::bytes_of(&look.uniform(accumulated, lit)));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        // Before the march and in the same encoder, so the march reads what it wrote: the compute
        // passes of one encoder run in the order they were recorded. Not timed on its own slot --
        // the march's timestamps bracket the march, and the cull is inside the band's wall time,
        // which is the number `SubmissionCost` is about.
        if let (Some(pipeline), Some(group), true) =
            (self.cull_pipeline(), bindings.cull.as_ref(), cull_groups > 0)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile-cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings.march, &[]);
            pass.set_bind_group(1, group, &[]);
            pass.dispatch_workgroups(cull_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("march"),
                timestamp_writes: shading.timing.writes(slot, 0),
            });
            // The march, and the still's occlusion and shadow after it where the camera carries
            // them -- see `Renderer::record_march`. A still frame is what first makes the
            // pipeline for those.
            self.record_march(
                &mut pass,
                [&bindings.march, &bindings.atlas],
                camera,
                width,
                count,
                measured,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade"),
                timestamp_writes: shading.timing.writes(slot, 2),
            });
            pass.set_pipeline(&shading.pipeline);
            pass.set_bind_group(0, &bindings.shade, &[]);
            pass.set_bind_group(1, &shading.environment, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), lit.1.div_ceil(8).max(1), 1);
        }
        shading.timing.resolve(slot, &mut encoder);
        self.queue.submit(Some(encoder.finish()));
        shading.timing.arm(slot, &self.queue);
        Ok(ticket)
    }

    /// Marches about `pixels` pixels of whatever frame was presented last, in whole rows about
    /// its middle, and lights nothing: work for the
    /// card's clocks, not for the screen. `None` when there is no such frame to march.
    ///
    /// The first draft after the viewport has sat converged runs three to four times slower
    /// than the ones after it -- the card has clocked down -- and
    /// `fieldform-io/examples/idle_clocks.rs` measured what brings it back: a fifth of a
    /// second of a *full* queue. A light trickle does not, and leaves the card slower than
    /// leaving it alone; a smaller first draft does not either, because the wait is the clocks
    /// and not the pixels. So this is the real march, on the real field, in a band big enough
    /// to outlast an interface frame while the card is slow.
    ///
    /// It writes the G-buffer rows it marches and nothing else. The picture on screen is the
    /// lit texture, which only the lighting pass writes, and the next frame that is presented
    /// marches every row again before lighting any -- so nothing reads what this leaves.
    /// No targets are made and no bindings built: if the last frame's are not standing, there
    /// is nothing worth warming for.
    pub fn warm_rows(
        &self,
        shading: &mut ShadingResources,
        camera: &Camera,
        max_steps: u32,
        pixels: u32,
    ) -> Option<SubmissionTicket> {
        let generation = shading.generation();
        let (width, height) = shading.targets().map(|t| (t.width, t.height))?;
        let bindings = shading
            .bindings
            .as_ref()
            .filter(|b| b.renderer == self.identity() && b.generation == generation)?;
        let count = pixels.div_ceil(width.max(1)).clamp(1, height);
        // About the middle of the frame, where the model is. The first version marched the top
        // rows, which on a framed model all miss the scene's sphere and return at once: twenty
        // bands of a tenth of a millisecond each, and a card as cold after them as before.
        let first = (height - count) / 2;
        // **The warm-up does not cull.** This frame exists to put the card's clocks up and is
        // thrown away, so recording a cull for it would be work for nothing -- and `cull_live`
        // false is what keeps it from reading a mask some earlier frame left, for a camera and a
        // band that are not these.
        self.queue.write_buffer(
            &shading.camera,
            0,
            bytemuck::bytes_of(&camera_uniform(
                camera,
                width,
                height,
                max_steps,
                (first, count),
                false,
                // It marches no occlusion, so measures none.
                0,
            )),
        );
        let (ticket, slot) = shading.timing.begin();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("warm") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("march"),
                timestamp_writes: shading.timing.writes(slot, 0),
            });
            // **The march alone, whatever this camera carries.** This frame exists to put the
            // card's clocks up and is thrown away -- it lights nothing, as the comment above says.
            // Asking for the still's occlusion here would compile it during a warm-up, which is
            // exactly the wait the lazy build exists to move out of the way.
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings.march, &[]);
            pass.set_bind_group(1, &bindings.atlas, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), count.div_ceil(8), 1);
        }
        {
            // An empty pass, for its two timestamps: the slot resolves all four, and resolving
            // a query nothing wrote is a wait on the driver that need never end.
            let _ = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade, not run"),
                timestamp_writes: shading.timing.writes(slot, 2),
            });
        }
        shading.timing.resolve(slot, &mut encoder);
        self.queue.submit(Some(encoder.finish()));
        shading.timing.arm(slot, &self.queue);
        Some(ticket)
    }

    /// The last presented frame, read back as display-encoded bytes — for the examples that
    /// write an image and the tests that check one. The application never calls this.
    pub fn read_display(&self, shading: &ShadingResources) -> Result<DisplayImage, RenderError> {
        let targets = shading.targets().ok_or(RenderError::Readback)?;
        let (width, height) = (targets.width, targets.height);
        // Rows of a texture copied to a buffer are padded to 256 bytes.
        let padded = (width * 4).div_ceil(256) * 256;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display-readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            targets.present.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;
        let view = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * padded) as usize;
            rgba.extend_from_slice(&view[start..start + (width * 4) as usize]);
        }
        drop(view);
        readback.unmap();
        Ok(DisplayImage { width, height, rgba })
    }

    /// The accumulated linear image, for tests that check energy rather than appearance.
    pub fn read_hdr(&self, shading: &ShadingResources) -> Result<Vec<[f32; 4]>, RenderError> {
        let targets = shading.targets().ok_or(RenderError::Readback)?;
        let bytes = targets.hdr.size();
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdr-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&targets.hdr, 0, &readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;
        let view = slice.get_mapped_range();
        let out: Vec<[f32; 4]> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback.unmap();
        Ok(out)
    }

    /// What the last frame's marches measured beyond the surface, per pixel: how open it is and
    /// how much of the key light reaches it, as the lighting pass reads them -- `[1, 1]` where a
    /// frame measures nothing, as a draft does, and `[-1, -1]` where a still at one in four left
    /// the pixel for the lighting pass to fill in. For the examples and tests that check which
    /// pixels a still spends its occlusion and shadow on, which the lit image cannot say.
    pub fn read_extras(&self, shading: &ShadingResources) -> Result<Vec<[f32; 2]>, RenderError> {
        let targets = shading.targets().ok_or(RenderError::Readback)?;
        let bytes = targets.extras.size();
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("extras-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&targets.extras, 0, &readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::Readback)?;
        let view = slice.get_mapped_range();
        // `pack2x16float` puts its first half in the low sixteen bits.
        let out: Vec<[f32; 2]> = bytemuck::cast_slice::<u8, u32>(&view)
            .iter()
            .map(|&word| {
                [
                    half::f16::from_bits((word & 0xffff) as u16).to_f32(),
                    half::f16::from_bits((word >> 16) as u16).to_f32(),
                ]
            })
            .collect();
        drop(view);
        readback.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Caught here rather than by wgpu, whose answer to a shader it cannot compile is to panic
    /// the process.
    #[test]
    fn the_lighting_shader_is_valid_wgsl() {
        crate::codegen::validated(SHADE).expect("shade.wgsl validates");
    }

    /// The uniform is read by the shader field for field, so its size is a fact about both.
    #[test]
    fn the_shading_uniform_is_the_size_the_shader_declares() {
        assert_eq!(std::mem::size_of::<ShadingUniform>(), 16 * 16);
    }

    /// A timestamp pair is scaled by the tick, and one the GPU wrote backwards is refused
    /// rather than read as an enormous frame.
    #[test]
    fn a_timestamp_pair_is_milliseconds_or_nothing() {
        // One tick per nanosecond: a million ticks is a millisecond.
        assert_eq!(ms_between(1_000, 1_001_000, 1.0), Some(1.0));
        // A 40 ns tick, the kind an NVIDIA card reports.
        let ms = ms_between(0, 25_000, 40.0).expect("in order");
        assert!((ms - 1.0).abs() < 1e-6, "{ms}");
        assert_eq!(ms_between(10, 5, 1.0), None);
    }

    /// The best number is the GPU's own when there is one, and says so by being there.
    #[test]
    fn a_cost_prefers_the_gpu_clock_and_falls_back_to_the_wall() {
        let timed = SubmissionCost { wall_ms: 20.0, march_ms: Some(3.0), shade_ms: Some(0.5) };
        assert_eq!(timed.gpu_ms(), Some(3.5));
        assert_eq!(timed.ms(), 3.5);
        let wall = SubmissionCost { wall_ms: 20.0, march_ms: None, shade_ms: None };
        assert_eq!(wall.gpu_ms(), None);
        assert_eq!(wall.ms(), 20.0);
    }

    /// Frame zero is centred and the rest stay inside the pixel, so a still converges to the
    /// image the drag showed and not to one shifted beside it.
    #[test]
    fn jitter_starts_centred_and_never_leaves_the_pixel() {
        assert_eq!(halton_jitter(0), [0.0, 0.0]);
        for frame in 1..64 {
            let [x, y] = halton_jitter(frame);
            assert!(x.abs() < 0.5 && y.abs() < 0.5, "frame {frame}: {x} {y}");
        }
    }

    #[test]
    fn the_studio_rig_points_its_lights_at_the_lit_side() {
        let forward = Vector3::new(0.0, 1.0, 0.0);
        let right = Vector3::new(1.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 0.0, 1.0);
        let [key, fill, rim] = Look::studio_lights(forward, right, up);
        // Key and fill come from the viewer's side; the rim from behind the model.
        assert!(key.direction.dot(&forward) < 0.0);
        assert!(fill.direction.dot(&forward) < 0.0);
        assert!(rim.direction.dot(&forward) > 0.0);
        // All from above, and all unit length.
        for light in [key, fill, rim] {
            assert!(light.direction.z > 0.0);
            assert!((light.direction.norm() - 1.0).abs() < 1e-9);
        }
    }
}
