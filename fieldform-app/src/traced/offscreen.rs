// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A picture of the field at a size of its own, drawn beside the viewport and never shown.
//!
//! The Save Image block's "Still": the same march, lights, environment and quality rung the
//! viewport uses, into targets of its own, at any size, converged the way the viewport's still
//! converges and read back when it is. The viewport is untouched while it runs -- its targets,
//! its accumulation and its camera are all its own -- which is what lets somebody keep looking
//! at a converged view while a print of it is drawn.
//!
//! **One projection, in tiles.** An image larger than one set of targets may hold -- the device
//! binds a storage buffer of 128 MiB by default, which is eight million pixels of HDR -- is
//! drawn as tiles, each a sub-frustum of the whole image's one camera (`Camera::tile`), so the
//! seams are not there to be seen. The jitter is per frame and the same for every pixel, and
//! the occlusion's measured corner is chosen by the frame number, so a pixel receives the same
//! samples tiled or whole; the tile origins are multiples of eight so the two-by-two occlusion
//! blocks never straddle an edge.
//!
//! **Supersampling is more samples, not a bigger buffer.** A still already averages jittered
//! frames in linear light before the tonemap; `supersample` k asks for k² times as many, over
//! the same low-discrepancy sequence, which is the box average of a k-times-larger render done
//! without the k² memory. The hit threshold is the scene's (a hundred-thousandth of it) and not
//! the pixel's, so a larger picture is the same surface drawn with more rays, and no step budget
//! has to grow with it.

use super::*;

/// How much the targets of one tile may hold, over every buffer and the texture: the lid on a
/// tile's size beside the device's own limits. Forty bytes a pixel (`FrameTargets`), so a tile
/// is about thirteen million pixels at most, and in practice the device's storage-binding limit
/// is the tighter of the two.
const TILE_BUDGET_BYTES: u64 = 512 << 20;

/// Bytes a pixel costs across a tile's targets: the G-buffer, owners, extras and HDR buffers
/// and the lit texture. The half-float texture is four more.
const BYTES_PER_PIXEL: u64 = 16 + 4 + 4 + 16 + 4;

/// What a saved image asks of the traced view.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTarget {
    pub width: u32,
    pub height: u32,
    /// One, two or four: how many times the samples per pixel of the viewport's still.
    pub supersample: u32,
    /// Coverage in the alpha channel, a miss contributing nothing -- see `Look::transparent`.
    pub transparent: bool,
    /// What a miss shows, linear, or the viewport's own backdrop when `None`.
    pub background: Option<[f32; 3]>,
    /// Half floats in the lit texture, for a 16-bit file.
    pub deep: bool,
}

/// The view as the picture is to be taken from it: everything [`View`] carries but the lookup,
/// owned, so a job spanning many frames draws the view it was asked for.
#[derive(Clone, Debug)]
pub struct ViewSnapshot {
    pub eye: Point3<f64>,
    pub target: Point3<f64>,
    pub up: Vector3<f64>,
    pub scene: (Point3<f64>, f64),
    pub fov_deg: f64,
    pub orthographic: bool,
    pub sections: Vec<(Vector3<f64>, f64)>,
    pub material: Material,
    pub occlusion: bool,
    pub reflections: bool,
    pub exact_finish: bool,
    pub environment: EnvironmentChoice,
    pub still_frames: u32,
}

impl View<'_> {
    /// The view, owned, with the selection and the draft read left out: a saved image draws no
    /// interface over the part.
    pub fn snapshot(&self) -> ViewSnapshot {
        ViewSnapshot {
            eye: self.eye,
            target: self.target,
            up: self.up,
            scene: self.scene,
            fov_deg: self.fov_deg,
            orthographic: self.orthographic,
            sections: self.sections.clone(),
            material: self.material.clone(),
            occlusion: self.occlusion,
            reflections: self.reflections,
            exact_finish: self.exact_finish,
            environment: self.environment,
            still_frames: self.still_frames,
        }
    }
}

/// The camera a frame of `view` is traced with, `width` by `height` pixels.
///
/// **The one derivation**, shared by the viewport's draw and a saved image: the two must frame
/// the scene identically or an export is a different picture from the one on screen. Perspective
/// keeps the vertical field of view, so a wider image extends sideways and a narrower one crops;
/// orthographic keeps the vertical extent the same way. The caller adds what is a per-frame
/// choice -- jitter, occlusion, the exact finish, a tile.
pub(super) fn frame_camera(
    view: &ViewSnapshot,
    width: u32,
    height: u32,
    scene_bounded: bool,
    cull: bool,
    value_scale: f64,
) -> Camera {
    let mut camera = Camera::looking_at(
        view.eye,
        view.target,
        view.up,
        view.scene,
        view.fov_deg,
        width as f64 / height.max(1) as f64,
    );
    camera.sections = fieldform_gpu::Sections::of(&view.sections);
    // The app's sphere is the union of every bounded body's extent, so it encloses all of
    // them exactly when there is no unbounded one -- which the rebuild checked.
    camera.start_at_scene = scene_bounded;
    camera.cull = cull;
    // The same number the step budget is bought with, for the same reason: a divided field
    // is honest and is not a distance estimate. The budget is what stops it losing the
    // silhouette; this is what stops it shading black. See `Camera::value_scale`.
    camera.value_scale = value_scale;
    if view.orthographic {
        // The same half-height the app's own projection matrix uses: the perspective one
        // taken out to the distance it is being viewed from. Sharing the derivation keeps
        // the traced surface the same size as the cage drawn over it in overlay mode -- two
        // framings of one scene would show as the wireframe sitting a little inside or
        // outside the surface it describes.
        let distance = (view.eye - view.target).norm();
        camera.orthographic = true;
        camera.half_height = (distance * (view.fov_deg.to_radians() * 0.5).tan()).max(1e-9);
        // Parallel rays start on a plane through the eye rather than at a point, so a ray
        // at the edge of the frame begins further out than the centre one does. The budget
        // has to cover that, or the model clips along its own silhouette.
        camera.far += camera.half_height * 2.0;
    }
    camera
}

/// The look a frame of `view` is lit with, for `camera`'s basis: the studio rig follows the
/// camera, as the viewport's one light always has, and the environment does not.
pub(super) fn frame_look(view: &ViewSnapshot, camera: &Camera) -> Look {
    Look::for_environment(view.environment, view.material.clone(), camera.forward, camera.right, camera.up)
}

/// What a still frame adds to the camera: the jitter of its sample, the exact finish on the
/// rung that asks, and occlusion with the key light's shadow where the view wants them.
pub(super) fn still_frame(camera: &mut Camera, look: &Look, view: &ViewSnapshot, accumulated: u32) {
    camera.jitter = halton_jitter(accumulated);
    // The frames somebody looks at finish on the outline (`Camera::exact_finish`), on the rung
    // that asks (`DisplayQuality::exact_finish`, every rung since the read is local), and by the
    // env var for an example or a rung pulled back. `FIELDFORM_BAKE_FINISH=1` holds every still
    // to its bake, the control arm for a picture that may or may not be the finish's doing.
    camera.exact_finish = (view.exact_finish || std::env::var_os("FIELDFORM_EXACT_FINISH").is_some())
        && std::env::var_os("FIELDFORM_BAKE_FINISH").is_none();
    if view.occlusion && view.environment.wants_occlusion() {
        camera.occlusion = Some(Occlusion::for_scene(look.lights[0].direction, view.scene.1));
    }
    // The reflected march, where the view wants one and the room shows it (item 54). Whether
    // this material's lobe can be traced, and as a mirror or a glossy sample, is decided here
    // once a frame and never per pixel: the material is a uniform.
    if view.reflections && view.environment.wants_reflections() {
        camera.reflections = Reflections::for_material(&view.material, view.still_frames);
    }
}

/// The picture, once it is whole.
pub enum ImageOutcome {
    Rgba8(fieldform_gpu::DisplayImage),
    RgbaF32(fieldform_gpu::DisplayImageF32),
}

/// Where a job has got to, for a status line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageProgress {
    /// Which tile is being drawn, of how many.
    pub tile: (usize, usize),
    /// Which sample of the tile is being accumulated, of how many.
    pub frame: (u32, u32),
}

impl ImageProgress {
    /// Nought to one, by tiles and samples done.
    pub fn fraction(&self) -> f32 {
        let per_tile = 1.0 / self.tile.1.max(1) as f32;
        let within = (self.frame.0 as f32 / self.frame.1.max(1) as f32).min(1.0);
        ((self.tile.0 as f32 + within) * per_tile).min(1.0)
    }
}

/// What one frame of driving the job found.
pub enum ImageStep {
    /// A band submitted or waited for; `TracedView::image_in_progress` says how far along.
    Running,
    Done(ImageOutcome, ImageCost),
    Failed(String),
}

/// What a finished image cost: how the GPU time was spent, and how it was cut up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImageCost {
    pub gpu_ms: f32,
    pub bands: u32,
    pub tiles: u32,
    pub frames_per_tile: u32,
}

/// The pixels of the whole image, filled tile by tile.
enum Assembled {
    Rgba8(Vec<u8>),
    RgbaF32(Vec<f32>),
}

/// A saved image being drawn: its own targets, its tiles, and where it has got to.
pub(super) struct ImageJob {
    target: ImageTarget,
    view: ViewSnapshot,
    shading: ShadingResources,
    tiles: Vec<fieldform_gpu::ImageTile>,
    tile: usize,
    /// Samples per tile: the still's count times the supersample squared.
    frames: u32,
    accumulated: u32,
    cursor: u32,
    /// The rows of the band in flight, or `None` for none: one submission at a time, as the
    /// viewport keeps to, so a band is sized from what the last one cost.
    in_flight: Option<(SubmissionTicket, u32)>,
    last_band_rows: Option<u32>,
    /// The dearest row cost seen on this job's own bands, in milliseconds.
    ms_per_row: Option<f32>,
    assembled: Assembled,
    cost: ImageCost,
}

impl TracedView {
    /// Begins drawing `target` from `view`, on the program in hand.
    ///
    /// Refused while there is nothing built, while a shader is still compiling -- the renderer
    /// in hand draws the document as it was before the edit -- and while another image is being
    /// drawn, since the two would share the one renderer's parameter values.
    pub fn begin_image(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        target: ImageTarget,
        view: ViewSnapshot,
    ) -> Result<(), String> {
        if self.image.is_some() {
            return Err("an image is already being drawn".into());
        }
        if self.renderer.is_none() || self.program.is_none() {
            return Err("there is no field drawn to picture yet".into());
        }
        if self.is_stale() || self.compiling.is_some() {
            return Err("the view is still being rebuilt".into());
        }
        let (width, height) = (target.width, target.height);
        if width == 0 || height == 0 {
            return Err("an image needs a width and a height over nought".into());
        }
        let device = &render_state.device;
        let limits = device.limits();
        let per_pixel = BYTES_PER_PIXEL + if target.deep { 4 } else { 0 };
        let max_pixels = (self.image_tile_pixels.unwrap_or(TILE_BUDGET_BYTES / per_pixel))
            .min(limits.max_storage_buffer_binding_size / 16)
            .min(limits.max_buffer_size / 16)
            .max(64);
        let tiles = tiles_for(width, height, limits.max_texture_dimension_2d, max_pixels);
        let maps = self.environments.entry(view.environment).or_insert_with(|| view.environment.bake());
        let format = if target.deep {
            fieldform_gpu::PresentFormat::Rgba16Float
        } else {
            fieldform_gpu::PresentFormat::Rgba8
        };
        let shading = ShadingResources::with_present_format(device, &render_state.queue, maps, format)
            .map_err(|e| format!("could not build the lighting pass for the image: {e}"))?;
        let pixels = (width as usize) * (height as usize) * 4;
        let assembled = if target.deep {
            Assembled::RgbaF32(vec![0.0; pixels])
        } else {
            Assembled::Rgba8(vec![0; pixels])
        };
        let frames = view.still_frames.clamp(1, STILL_FRAMES) * target.supersample.clamp(1, 4).pow(2);
        let tiles_count = tiles.len() as u32;
        self.image = Some(ImageJob {
            target,
            view,
            shading,
            tiles,
            tile: 0,
            frames,
            accumulated: 0,
            cursor: 0,
            in_flight: None,
            last_band_rows: None,
            ms_per_row: None,
            assembled,
            cost: ImageCost { gpu_ms: 0.0, bands: 0, tiles: tiles_count, frames_per_tile: frames },
        });
        Ok(())
    }

    /// How many tiles an image of this size would be drawn in on this device, and the longest
    /// side one of them may have: what the panel says under the size once tiling is needed.
    pub fn image_tiles(&self, device: &wgpu::Device, width: u32, height: u32, deep: bool) -> usize {
        let limits = device.limits();
        let per_pixel = BYTES_PER_PIXEL + if deep { 4 } else { 0 };
        let max_pixels = (TILE_BUDGET_BYTES / per_pixel)
            .min(limits.max_storage_buffer_binding_size / 16)
            .min(limits.max_buffer_size / 16)
            .max(64);
        tiles_for(width, height, limits.max_texture_dimension_2d, max_pixels).len()
    }

    /// Holds a tile to at most this many pixels, for the test that proves a tiled render is the
    /// untiled one byte for byte; `None` is the budget.
    #[cfg(test)]
    pub(crate) fn limit_image_tile(&mut self, pixels: Option<u64>) {
        self.image_tile_pixels = pixels;
    }

    /// Whether an image is being drawn.
    pub fn image_in_progress(&self) -> Option<ImageProgress> {
        self.image.as_ref().map(|job| ImageProgress {
            tile: (job.tile, job.tiles.len()),
            frame: (job.accumulated, job.frames),
        })
    }

    /// Drops the image being drawn, if one is. Its targets go with it.
    pub fn cancel_image(&mut self) {
        self.image = None;
    }

    /// One frame's worth of the image: a band submitted if the card is free, a tile read back
    /// once its samples are in, the picture handed over once its tiles are.
    ///
    /// Never while the viewport's own march is in flight, so a print never adds to the latency
    /// of a hand on the view; the viewport draws nothing while it is converged, which is when
    /// this gets the card to itself.
    pub fn drive_image(&mut self, render_state: &egui_wgpu::RenderState) -> Option<ImageStep> {
        let job = self.image.as_mut()?;
        let Some(renderer) = self.renderer.as_ref() else {
            self.image = None;
            return Some(ImageStep::Failed("the renderer went away while the image was drawn".into()));
        };
        if self.compiling.is_some() || self.stale.is_some() {
            self.image = None;
            return Some(ImageStep::Failed(
                "the document changed while the image was being drawn, so it was not written".into(),
            ));
        }
        let _ = render_state.device.poll(wgpu::PollType::Poll);
        // What the band in flight cost, once the GPU says. Fails open: a band whose timing
        // slot was busy reports nothing and the next band is sized from the one before it.
        if let Some((ticket, rows)) = job.in_flight {
            let mut done = false;
            for (t, cost) in job.shading.costs() {
                if t == ticket {
                    done = true;
                    if cost.ms() > 0.0 {
                        let per_row = cost.ms() / rows.max(1) as f32;
                        job.ms_per_row = Some(job.ms_per_row.map_or(per_row, |was| was.max(per_row)));
                        job.cost.gpu_ms += cost.ms();
                    }
                }
            }
            if done || !job.shading.busy() {
                job.in_flight = None;
            }
        }
        if job.in_flight.is_some() {
            return Some(ImageStep::Running);
        }
        if self.shading.as_ref().is_some_and(|s| s.busy()) {
            return Some(ImageStep::Running);
        }

        let Some(tile) = job.tiles.get(job.tile).copied() else {
            // Nothing left to draw: finished on an earlier frame and not yet taken.
            return None;
        };
        let (tw, th) = (tile.width, tile.height);

        let mut camera = frame_camera(
            &job.view,
            job.target.width,
            job.target.height,
            self.scene_bounded,
            // The cull's tile mask is sized for a frame no wider than the viewport's; an
            // image is drawn unculled, which costs time and never material.
            false,
            self.value_scale.unwrap_or(1.0),
        );
        let mut look = frame_look(&job.view, &camera);
        still_frame(&mut camera, &look, &job.view, job.accumulated);
        // The shader is told where this tile sits in the whole image and how large the image
        // is; a frame that is the whole image is told nothing, as the viewport's is.
        camera.tile = (job.tiles.len() > 1).then_some(fieldform_gpu::ImageTile {
            x: tile.x,
            y: tile.y,
            width: job.target.width,
            height: job.target.height,
        });
        look.selected = None;
        look.draft = None;
        look.transparent = job.target.transparent;
        if let Some(background) = job.target.background {
            look.background = background;
        }

        // A still goes up in bands, each its own submission, sized from the dearest row this
        // job has drawn and floored by what the viewport's draft cost per pixel -- the same rule
        // the viewport's still follows, for the same driver timeout.
        let from_draft = self.draft_ms_per_pixel.map(|ms| ms * tw as f32 * DRAFT_TO_STILL);
        let per_row = match (job.ms_per_row, from_draft) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let rows = tile_aligned_rows(band_rows(BAND_BUDGET_MS, per_row, job.last_band_rows, tw, th), th);
        let first = job.cursor.min(th);
        let last = (first + rows).min(th);
        let budget = self.step_budget.unwrap_or(STEP_BUDGET);
        match renderer.present_rows(&mut job.shading, &camera, tw, th, budget, &look, job.accumulated, first..last) {
            Ok(ticket) => {
                job.in_flight = Some((ticket, last - first));
                job.last_band_rows = Some(last - first);
                job.cost.bands += 1;
            }
            Err(e) => {
                self.image = None;
                return Some(ImageStep::Failed(format!("the image's march failed: {e}")));
            }
        }
        job.cursor = last;
        if last < th {
            return Some(ImageStep::Running);
        }
        job.cursor = 0;
        job.accumulated += 1;
        if job.accumulated < job.frames {
            return Some(ImageStep::Running);
        }
        // The tile's samples are all in: read it back into its place in the picture. The read
        // waits for the band just submitted, which is the last of the tile.
        let placed = match &mut job.assembled {
            Assembled::Rgba8(whole) => renderer.read_display(&job.shading).map(|img| {
                place_rows(whole, job.target.width, &tile, tw, th, &img.rgba, 4);
            }),
            Assembled::RgbaF32(whole) => renderer.read_display_f32(&job.shading).map(|img| {
                place_rows(whole, job.target.width, &tile, tw, th, &img.rgba, 4);
            }),
        };
        if let Err(e) = placed {
            self.image = None;
            return Some(ImageStep::Failed(format!("the image could not be read back: {e}")));
        }
        // The read waited for the tile's last band, whose cost is now in: add it, or the
        // figure said for a one-band picture is nought.
        for (t, cost) in job.shading.costs() {
            if job.in_flight.is_some_and(|(ticket, _)| ticket == t) && cost.ms() > 0.0 {
                job.cost.gpu_ms += cost.ms();
            }
        }
        job.in_flight = None;
        job.tile += 1;
        job.accumulated = 0;
        job.last_band_rows = None;
        if job.tile < job.tiles.len() {
            return Some(ImageStep::Running);
        }
        let job = self.image.take().expect("held above");
        let (width, height) = (job.target.width, job.target.height);
        let outcome = match job.assembled {
            Assembled::Rgba8(rgba) => ImageOutcome::Rgba8(fieldform_gpu::DisplayImage { width, height, rgba }),
            Assembled::RgbaF32(rgba) => ImageOutcome::RgbaF32(fieldform_gpu::DisplayImageF32 { width, height, rgba }),
        };
        Some(ImageStep::Done(outcome, job.cost))
    }
}

/// Copies a tile's rows, `channels` values a pixel, into their place in the whole picture.
fn place_rows<T: Copy>(
    whole: &mut [T],
    width: u32,
    tile: &fieldform_gpu::ImageTile,
    tw: u32,
    th: u32,
    rows: &[T],
    channels: usize,
) {
    let stride = width as usize * channels;
    let span = tw as usize * channels;
    for row in 0..th as usize {
        let from = row * span;
        let to = (tile.y as usize + row) * stride + tile.x as usize * channels;
        whole[to..to + span].copy_from_slice(&rows[from..from + span]);
    }
}

/// How an image of `width` by `height` is cut into tiles no side of which exceeds `max_side`
/// and no one of which holds more than `max_pixels`: a grid of equal steps, multiples of
/// eight so the occlusion's two-by-two blocks never straddle a seam, with the last column and
/// row taking what is left. One tile covering the whole image when it fits.
///
/// **Each tile carries its own size in `width`/`height`**, not the whole image's, because the
/// caller sets `Camera::tile` from the whole image's size separately and what the job needs
/// per tile is the step; `drive_image` writes the camera's tile with the image's size and the
/// tile's origin, which is what the shader reads.
pub(super) fn tiles_for(width: u32, height: u32, max_side: u32, max_pixels: u64) -> Vec<fieldform_gpu::ImageTile> {
    let side = max_side.max(8);
    let mut tw = width.min(side);
    if tw < width {
        tw = (tw / 8 * 8).max(8);
    }
    let rows_allowed = (max_pixels / u64::from(tw.max(1))).clamp(1, u64::from(u32::MAX)) as u32;
    let mut th = height.min(side).min(rows_allowed);
    if th < height {
        th = (th / 8 * 8).max(8);
    }
    let mut out = Vec::new();
    let mut y = 0;
    while y < height {
        let h = th.min(height - y);
        let mut x = 0;
        while x < width {
            let w = tw.min(width - x);
            out.push(fieldform_gpu::ImageTile { x, y, width: w, height: h });
            x += w;
        }
        y += h;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_that_fits_is_one_tile_the_size_of_the_image() {
        let tiles = tiles_for(1920, 1080, 8192, 1 << 24);
        assert_eq!(tiles, vec![fieldform_gpu::ImageTile { x: 0, y: 0, width: 1920, height: 1080 }]);
    }

    #[test]
    fn tiles_cover_the_image_exactly_once_on_eight_pixel_origins() {
        for (w, h, side, px) in [(10_000, 7_000, 8192, 4_000_000u64), (5001, 3001, 4096, 1_000_000), (16_384, 16_384, 8192, 8_000_000), (9, 9, 8, 64)] {
            let tiles = tiles_for(w, h, side, px);
            let mut covered = vec![0u8; (w as usize) * (h as usize)];
            for t in &tiles {
                assert!(t.width <= side && t.height <= side, "{t:?} within the side limit");
                assert!(u64::from(t.width) * u64::from(t.height) <= px.max(64), "{t:?} within the pixel budget");
                assert!(t.x % 8 == 0 && t.y % 8 == 0, "{t:?} starts on a multiple of eight");
                for y in t.y..t.y + t.height {
                    for x in t.x..t.x + t.width {
                        covered[(y * w + x) as usize] += 1;
                    }
                }
            }
            assert!(covered.iter().all(|c| *c == 1), "{w}x{h}: every pixel in exactly one tile");
        }
    }

    #[test]
    fn progress_runs_from_nought_to_one_over_tiles_and_samples() {
        let p = |tile, tiles, frame, frames| ImageProgress { tile: (tile, tiles), frame: (frame, frames) }.fraction();
        assert_eq!(p(0, 1, 0, 8), 0.0);
        assert!((p(0, 1, 4, 8) - 0.5).abs() < 1e-6);
        assert!((p(1, 2, 0, 8) - 0.5).abs() < 1e-6);
        assert!((p(3, 4, 8, 8) - 1.0).abs() < 1e-6);
    }
}
