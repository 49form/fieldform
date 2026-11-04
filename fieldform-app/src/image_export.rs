// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The viewport as an image file -- backlog item 117.
//!
//! **One request, one render path, one encoder, three ways in.** The console's `snap`, the
//! Save Image block's *Save now* and the palette's *Save image again* all build an
//! [`ImageRequest`] and hand it to [`App::begin_image`]; the frame loop drives it through
//! [`App::drive_snap`] -- the viewport's own picture read back once it has settled
//! (`Displayed`), or a still of its own at any size drawn beside the viewport (`Still`,
//! `traced::offscreen`) -- and a worker thread encodes and writes the file. There is no
//! second path to a frame: `snap` used to read the viewport's texture on its own, and that
//! reading is now the `Displayed` arm of this one.
//!
//! **What a file carries.** A PNG is written with an sRGB chunk and three text chunks: the
//! console line that reproduces it, the application and version, and the document's name. A
//! JPEG carries the line in a comment segment. So a render found on disk months later says
//! what made it, and pasting the line makes it again.

use crate::*;
use fieldform_scene::{ImageBackground, ImageFormat, ImageQuality, SaveImage, SaveImageId};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long a Still may take to draw before it is given up as a view that will not finish:
/// a named constant, as the design asks, and generous because a print is slow by nature.
pub(crate) const IMAGE_RENDER_MAX: Duration = Duration::from_secs(120);

/// Everything one picture is made from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageRequest {
    /// A view as `snap view=` takes it, or `None` for the camera as it stands.
    pub(crate) view: Option<String>,
    /// Closer by this factor about the same target, or none.
    pub(crate) zoom: Option<f32>,
    /// Output pixels. Nought for either means the viewport's own.
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) quality: ImageQuality,
    pub(crate) supersample: u8,
    pub(crate) background: ImageBackground,
    pub(crate) format: ImageFormat,
    pub(crate) bits: u8,
    pub(crate) jpeg_quality: u8,
    pub(crate) path: PathBuf,
    pub(crate) overwrite: bool,
    /// The console line that reproduces this picture, written into the file.
    pub(crate) line: String,
    /// The block this ran for, if one did: its result line and counter are updated.
    pub(crate) block: Option<SaveImageId>,
    /// An automatic capture: nothing said in the console, only the block's own log.
    pub(crate) silent: bool,
}

impl ImageRequest {
    /// A request as `snap file=` makes one with no other key: the viewport's picture, as PNG.
    pub(crate) fn displayed(path: PathBuf) -> Self {
        Self {
            view: None,
            zoom: None,
            width: 0,
            height: 0,
            quality: ImageQuality::Displayed,
            supersample: 1,
            background: ImageBackground::Environment,
            format: ImageFormat::Png,
            bits: 8,
            jpeg_quality: 92,
            path,
            overwrite: true,
            line: String::new(),
            block: None,
            silent: false,
        }
    }

    /// The settings of a block, with its file name resolved. `counter` is the `{n}` to use.
    pub(crate) fn of_block(block: &SaveImage, path: PathBuf, silent: bool) -> Self {
        Self {
            view: block.view.clone(),
            zoom: None,
            width: block.width,
            height: block.height,
            quality: block.quality,
            supersample: block.supersample,
            background: block.background,
            format: block.format,
            bits: block.bits,
            jpeg_quality: block.jpeg_quality,
            path,
            overwrite: block.overwrite,
            line: String::new(),
            block: Some(block.id),
            silent,
        }
    }

    /// Why this request cannot be honoured, if it cannot. Every refusal names the key and the
    /// value that would be accepted, since each is a thing a person typed or chose.
    pub(crate) fn refusal(&self, viewport: (u32, u32)) -> Option<String> {
        if self.format == ImageFormat::Jpeg && self.background == ImageBackground::Transparent {
            return Some("a JPEG cannot carry alpha, so `bg=transparent` needs `format=png`".into());
        }
        if self.bits != 8 && self.bits != 16 {
            return Some(format!("`bits={}` is neither 8 nor 16", self.bits));
        }
        if self.bits == 16 && self.format != ImageFormat::Png {
            return Some("16-bit depth is PNG only: `format=png`".into());
        }
        if self.bits == 16 && self.quality != ImageQuality::Still {
            return Some("16-bit depth is for a Still: the displayed picture is 8-bit already".into());
        }
        if ![1, 2, 4].contains(&self.supersample) {
            return Some(format!("`ss={}` is not 1, 2 or 4", self.supersample));
        }
        if !(1..=100).contains(&self.jpeg_quality) {
            return Some(format!("`jpgq={}` is outside 1 to 100", self.jpeg_quality));
        }
        if self.width > SaveImage::MAX_SIDE || self.height > SaveImage::MAX_SIDE {
            return Some(format!(
                "a side of {} pixels is past the {} an image may have",
                self.width.max(self.height),
                SaveImage::MAX_SIDE
            ));
        }
        if self.quality == ImageQuality::Displayed {
            let asked = (self.width, self.height);
            if asked != (0, 0) && asked != viewport {
                return Some(format!(
                    "the displayed picture is the viewport's own size, {} by {}; another size is a Still: `quality=still`",
                    viewport.0, viewport.1
                ));
            }
            if self.supersample != 1 || self.background != ImageBackground::Environment {
                return Some("supersampling and a background are for a Still: `quality=still`".into());
            }
        }
        if let Some(ext) = self.path.extension().and_then(|e| e.to_str()) {
            match ImageFormat::from_extension(ext) {
                Some(f) if f != self.format => {
                    return Some(format!(
                        "`format={}` does not match the file's `.{ext}`",
                        format_word(self.format)
                    ));
                }
                None => return Some(format!("`.{ext}` is neither .png nor .jpg")),
                _ => {}
            }
        }
        None
    }

    /// The size this request renders at, given the viewport's.
    pub(crate) fn size(&self, viewport: (u32, u32)) -> (u32, u32) {
        match (self.width, self.height) {
            (0, 0) => viewport,
            // One side given: the other from the viewport's aspect.
            (w, 0) => (w, ((w as f64) * viewport.1 as f64 / viewport.0.max(1) as f64).round().max(1.0) as u32),
            (0, h) => (((h as f64) * viewport.0 as f64 / viewport.1.max(1) as f64).round().max(1.0) as u32, h),
            (w, h) => (w, h),
        }
    }
}

/// The word a format is typed as.
pub(crate) fn format_word(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
    }
}

/// The word a background is typed as.
pub(crate) fn background_word(bg: ImageBackground) -> String {
    match bg {
        ImageBackground::Environment => "env".into(),
        ImageBackground::Transparent => "transparent".into(),
        ImageBackground::Colour([r, g, b]) => format!("color:#{r:02x}{g:02x}{b:02x}"),
    }
}

/// A background as typed: `env`, `transparent` or `color:#rrggbb`.
pub(crate) fn parse_background(text: &str) -> Result<ImageBackground, String> {
    let t = text.trim().to_ascii_lowercase();
    match t.as_str() {
        "env" | "environment" => return Ok(ImageBackground::Environment),
        "transparent" | "alpha" | "none" => return Ok(ImageBackground::Transparent),
        _ => {}
    }
    let hex = t
        .strip_prefix("color:")
        .or_else(|| t.strip_prefix("colour:"))
        .map(|h| h.trim_start_matches('#'))
        .ok_or_else(|| format!("`bg={text}` is not env, transparent or color:#rrggbb"))?;
    if hex.len() != 6 {
        return Err(format!("`bg={text}` wants six hex digits after the #"));
    }
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| format!("`bg={text}` is not hex"));
    Ok(ImageBackground::Colour([byte(0)?, byte(2)?, byte(4)?]))
}

/// The linear colour the lighting pass is handed for a flat background: the sRGB bytes decoded,
/// so that the pass's own encode gives the bytes back. A flat background is "not lit, not
/// tonemapped" exactly because a miss keeps the encode and skips both.
pub(crate) fn linear_of(srgb: [u8; 3]) -> [f32; 3] {
    let decode = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    [decode(srgb[0]), decode(srgb[1]), decode(srgb[2])]
}

/// Where a view is aimed: a cube region, or any direction toward the eye.
pub(crate) enum ViewAim {
    Region(CubeRegion),
    Toward(nalgebra::Vector3<f32>),
}

/// `view=` as `snap` takes it: the ViewCube's own names, a corner by signs, or a direction.
pub(crate) fn parse_view(name: &str) -> Result<ViewAim, String> {
    let wanted = name.trim();
    let found = view_cube_faces()
        .into_iter()
        .find(|f| f.label().is_some_and(|l| l.eq_ignore_ascii_case(wanted)))
        .or_else(|| {
            // The corner the cube calls home, which is what every tool calls iso.
            ["iso", "home", "isometric"]
                .contains(&wanted.to_ascii_lowercase().as_str())
                .then_some(CubeRegion { sign: [-1, -1, 1] })
        })
        .or_else(|| {
            // Any other face, edge or corner of the cube, by the signs of where the eye goes:
            // `view=1,-1,1` is the corner across the model from iso.
            let signs: Vec<i8> = wanted.split(',').map(|v| v.trim().parse().ok()).collect::<Option<_>>()?;
            let sign: [i8; 3] = signs.try_into().ok()?;
            (sign.iter().all(|v| (-1..=1).contains(v)) && sign != [0, 0, 0]).then_some(CubeRegion { sign })
        });
    if let Some(region) = found {
        return Ok(ViewAim::Region(region));
    }
    // Or any direction at all, from the target toward the eye: `view=0.8,-0.3,0.5` is the orbit
    // somebody's screenshot was taken from, near enough to compare.
    let toward = wanted
        .split(',')
        .map(|v| v.trim().parse().ok())
        .collect::<Option<Vec<f32>>>()
        .and_then(|parts| <[f32; 3]>::try_from(parts).ok())
        .map(|v| nalgebra::Vector3::new(v[0], v[1], v[2]))
        .filter(|dir| dir.norm() > 1e-6);
    toward.map(ViewAim::Toward).ok_or_else(|| {
        format!(
            "`view=` takes the ViewCube's own names — front, back, left, right, top, bottom or iso — \
             a corner by signs, `view=1,-1,1`, or a direction, `view=0.8,-0.3,0.5`, not {wanted:?}"
        )
    })
}

/// Aims `camera` at once rather than swung: nobody is watching a scripted run, and an animation
/// would only be frames spent arriving. The same swing the ViewCube starts, finished at once.
pub(crate) fn aim_camera(camera: &mut OrbitCamera, aim: &ViewAim, zoom: Option<f32>) {
    match aim {
        ViewAim::Region(region) => camera.begin_snap(*region),
        ViewAim::Toward(dir) => camera.begin_snap_toward(*dir, nalgebra::Vector3::z()),
    }
    while camera.tick_snap(SNAP_SECONDS) {}
    if let Some(zoom) = zoom {
        camera.zoom_in(zoom);
    }
}

/// The tokens a file name pattern is filled from.
pub(crate) struct NameFill<'a> {
    pub(crate) doc: &'a str,
    pub(crate) block: &'a str,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) view: &'a str,
    /// `(YYYY-MM-DD, HHMMSS)` as the clock reads now.
    pub(crate) when: (String, String),
}

/// `pattern` with its tokens filled in and `n` as the counter. `{n:04}` pads to four digits;
/// a token nothing knows is left as typed, so a mistyped one shows in the file name rather
/// than vanishing.
pub(crate) fn fill_pattern(pattern: &str, fill: &NameFill<'_>, n: u32) -> String {
    let mut out = String::with_capacity(pattern.len() + 16);
    let mut rest = pattern;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            out.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let token = &after[..end];
        let filled = match token {
            "doc" => Some(fill.doc.to_string()),
            "block" => Some(fill.block.to_string()),
            "date" => Some(fill.when.0.clone()),
            "time" => Some(fill.when.1.clone()),
            "w" => Some(fill.width.to_string()),
            "h" => Some(fill.height.to_string()),
            "view" => Some(fill.view.to_string()),
            "n" => Some(n.to_string()),
            _ => token
                .strip_prefix("n:0")
                .and_then(|digits| digits.parse::<usize>().ok())
                .map(|digits| format!("{n:0digits$}")),
        };
        match filled {
            Some(text) => out.push_str(&text),
            None => out.push_str(&rest[start..start + end + 2]),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

/// A file name no other character of which the file system refuses: the tokens may bring a
/// document name in, and a document may be called anything.
pub(crate) fn safe_file_name(name: &str) -> String {
    name.chars()
        .map(|c| if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
        .collect()
}

/// The clock as `{date}` and `{time}` read it, local time where the platform says what that is.
pub(crate) fn now_tokens() -> (String, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let local = secs + local_offset_seconds();
    let (days, rem) = (local.div_euclid(86_400), local.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    (format!("{y:04}-{m:02}-{d:02}"), format!("{:02}{:02}{:02}", rem / 3600, (rem / 60) % 60, rem % 60))
}

/// The offset of local time from UTC, in seconds, where the platform can say; nought otherwise.
#[cfg(windows)]
fn local_offset_seconds() -> i64 {
    #[repr(C)]
    #[derive(Default)]
    struct TimeZoneInformation {
        bias: i32,
        standard_name: [u16; 32],
        standard_date: [u16; 8],
        standard_bias: i32,
        daylight_name: [u16; 32],
        daylight_date: [u16; 8],
        daylight_bias: i32,
    }
    unsafe extern "system" {
        fn GetTimeZoneInformation(info: *mut TimeZoneInformation) -> u32;
    }
    let mut info = TimeZoneInformation::default();
    // The bias is minutes *behind* UTC; 2 means daylight time is in force.
    let kind = unsafe { GetTimeZoneInformation(&mut info) };
    let bias = info.bias + if kind == 2 { info.daylight_bias } else { info.standard_bias };
    -(bias as i64) * 60
}

#[cfg(not(windows))]
fn local_offset_seconds() -> i64 {
    0
}

/// Civil date from days since 1970-01-01, by the usual proleptic-Gregorian arithmetic.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The path a block writes next, and the counter it used: the first free name from `from`
/// upward when overwriting is off, `from`'s own name when it is on.
pub(crate) fn next_path(block: &SaveImage, folder: &Path, fill: &NameFill<'_>, from: u32) -> (PathBuf, u32) {
    let mut n = from.max(1);
    loop {
        let mut name = safe_file_name(&fill_pattern(&block.pattern, fill, n));
        let has_extension = Path::new(&name)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| ImageFormat::from_extension(e).is_some());
        if !has_extension {
            name.push('.');
            name.push_str(block.format.extension());
        }
        let path = folder.join(name);
        if block.overwrite || !path.exists() || n == u32::MAX {
            return (path, n);
        }
        n += 1;
    }
}

/// One finished file.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageWritten {
    pub(crate) path: PathBuf,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes: u64,
    pub(crate) took: Duration,
    pub(crate) cost: Option<traced::ImageCost>,
}

impl ImageWritten {
    /// The line the console says for it.
    pub(crate) fn said(&self) -> String {
        let cost = match self.cost {
            Some(c) if c.tiles > 1 => format!(", {:.1} s on the card in {} tiles", c.gpu_ms / 1000.0, c.tiles),
            Some(c) => format!(", {:.1} s on the card", c.gpu_ms / 1000.0),
            None => String::new(),
        };
        format!(
            "Wrote {} — {} by {}, {} in {:.1} s{cost}",
            file_name(&self.path),
            self.width,
            self.height,
            size_text(self.bytes),
            self.took.as_secs_f32()
        )
    }
}

/// A file's size as a person reads one: kilobytes below a megabyte, a decimal of megabytes above.
pub(crate) fn size_text(bytes: u64) -> String {
    if bytes < 1 << 20 {
        format!("{} KB", bytes.div_ceil(1024))
    } else {
        format!("{:.1} MB", bytes as f64 / f64::from(1u32 << 20))
    }
}

/// Where a snapshot in flight has got to. See [`PendingSnap`].
pub(crate) enum SnapStage {
    /// Waiting for the viewport's own still to settle, then reading its texture.
    Viewport,
    /// A still of its own: begun on the traced view once it is current, driven each frame.
    Offscreen { begun: Option<Instant> },
    /// The picture is on a worker thread being encoded and written.
    Encoding(mpsc::Receiver<Result<ImageWritten, String>>),
}

/// A picture being made: the request, when it began, and which stage it is in.
///
/// A still converges over several frames, so the verb asks for one and this sits here until
/// the frame loop has written the file. A script **stops** while one is pending, so two
/// snapshots in a row cannot write each other's image, and the agent's `snapshot` tool waits
/// on the same thing.
pub(crate) struct PendingSnap {
    pub(crate) request: ImageRequest,
    pub(crate) since: Instant,
    /// The longest the driver has been seen compiling a shader during this wait, which does
    /// not count against the limit -- see `SNAP_WAIT_MAX`.
    pub(crate) compiling: Duration,
    pub(crate) stage: SnapStage,
    pub(crate) started: Instant,
}

impl App {
    /// Begins a picture: refuses what cannot be made, aims the camera where a Displayed
    /// picture asks, and leaves the rest to [`Self::drive_snap`].
    pub(crate) fn begin_image(&mut self, request: ImageRequest) -> Result<(), String> {
        if self.pending_snap.is_some() {
            return Err("a picture is already being made; it is written before the next begins".into());
        }
        let viewport = self.viewport_pixels();
        if let Some(why) = request.refusal(viewport) {
            return Err(why);
        }
        if let Some(parent) = request.path.parent().filter(|p| !p.as_os_str().is_empty())
            && !parent.is_dir()
        {
            return Err(format!("there is no folder {} to write the image into", parent.display()));
        }
        if !request.overwrite && request.path.exists() {
            return Err(format!("{} is already there, and overwrite is off", file_name(&request.path)));
        }
        let aim = match &request.view {
            Some(name) => Some(parse_view(name)?),
            None => None,
        };
        if let Some(zoom) = request.zoom
            && zoom.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        {
            return Err("`zoom=` wants a number over nought: 3 is three times closer".into());
        }
        let stage = match request.quality {
            ImageQuality::Displayed => {
                // The viewport's own picture, so the view is moved to take it: set rather than
                // swung, and the still asked for at final quality so a snapshot taken
                // mid-gesture is still the finished picture.
                if let Some(aim) = &aim {
                    aim_camera(&mut self.camera, aim, request.zoom);
                    self.traced.invalidate(traced::Quality::Final);
                } else if let Some(zoom) = request.zoom {
                    self.camera.zoom_in(zoom);
                    self.traced.invalidate(traced::Quality::Final);
                }
                self.traced.invalidate(traced::Quality::Final);
                SnapStage::Viewport
            }
            ImageQuality::Still => SnapStage::Offscreen { begun: None },
        };
        self.pending_snap = Some(PendingSnap {
            request,
            since: Instant::now(),
            compiling: Duration::ZERO,
            stage,
            started: Instant::now(),
        });
        Ok(())
    }

    /// The viewport's size in pixels, as the last frame drew it.
    ///
    /// **The still's size, not the window's.** A still is capped at the longest edge the display
    /// quality asks for, and the displayed picture is that still read back -- so this is the
    /// size `quality=displayed` writes and the size a Still defaults to.
    pub(crate) fn viewport_pixels(&self) -> (u32, u32) {
        let scale = self.egui_ctx.pixels_per_point();
        let rect = self.viewport_rect.unwrap_or(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 720.0)));
        traced::still_frame_size(rect.width() * scale, rect.height() * scale, self.display_quality.still_pixels())
    }

    /// The view a Still is drawn from: the camera as it stands or aimed where the request
    /// says, on a copy, so the person's own view is never moved for a picture.
    fn image_view(&self, request: &ImageRequest) -> Result<traced::ViewSnapshot, String> {
        let mut camera = self.camera.clone();
        if let Some(name) = &request.view {
            aim_camera(&mut camera, &parse_view(name)?, request.zoom);
        } else if let Some(zoom) = request.zoom {
            camera.zoom_in(zoom);
        }
        let material = self
            .doc
            .parts
            .get(&self.active_part)
            .and_then(|p| p.material.clone())
            .unwrap_or_default();
        Ok(traced::ViewSnapshot {
            eye: camera.eye.cast::<f64>(),
            target: camera.target.cast::<f64>(),
            up: camera.up.cast::<f64>(),
            scene: (
                Point3::new(
                    camera.scene_centre.x as f64,
                    camera.scene_centre.y as f64,
                    camera.scene_centre.z as f64,
                ),
                camera.scene_radius as f64,
            ),
            fov_deg: 45.0,
            orthographic: camera.projection == Projection::Orthographic,
            sections: self.section_half_spaces(),
            material,
            occlusion: self.idle_occlusion && self.display_quality.occlusion(),
            exact_finish: self.display_quality.exact_finish(),
            environment: self.environment,
            still_frames: self.display_quality.still_frames(),
        })
    }

    /// Drives the picture in flight: one stage a frame, the file written at the end.
    pub(crate) fn drive_snap(&mut self, ctx: &egui::Context) {
        // A shader the driver is still compiling is progress, not a stall, so the wait is
        // measured against everything but that.
        let compiling = self.traced.is_compiling();
        let viewport = self.viewport_pixels();
        let Some(snap) = &mut self.pending_snap else { return };
        if let Some(so_far) = compiling {
            snap.compiling = snap.compiling.max(so_far);
        }
        let waited = snap.since.elapsed().saturating_sub(snap.compiling);
        let limit = match &snap.stage {
            SnapStage::Offscreen { begun: Some(_) } => IMAGE_RENDER_MAX,
            _ => SNAP_WAIT_MAX,
        };
        if waited > limit && !matches!(snap.stage, SnapStage::Encoding(_)) {
            let name = file_name(&snap.request.path);
            let what = match &snap.stage {
                SnapStage::Offscreen { begun: Some(_) } => format!("the image did not finish drawing in {} s", limit.as_secs()),
                _ => format!("the view did not finish a still in {} s of drawing", limit.as_secs()),
            };
            let silent = snap.request.silent;
            let block = snap.request.block;
            self.pending_snap = None;
            self.traced.cancel_image();
            self.image_failed(block, silent, format!("{what}, so {name} was not written"));
            return;
        }
        match &mut snap.stage {
            SnapStage::Viewport => {
                if !self.traced.is_settled() {
                    // Keep the frames coming: a still averages one a frame and stops when it is done.
                    ctx.request_repaint();
                    return;
                }
                let Some(image) = self.traced.read_display() else {
                    let (block, silent, name) = (snap.request.block, snap.request.silent, file_name(&snap.request.path));
                    self.pending_snap = None;
                    self.image_failed(block, silent, format!("there is no picture to write to {name}"));
                    return;
                };
                let outcome = traced::ImageOutcome::Rgba8(image);
                self.start_encoding(outcome, None);
            }
            SnapStage::Offscreen { begun } => {
                if begun.is_none() {
                    let request = snap.request.clone();
                    let (width, height) = request.size(viewport);
                    let view = match self.image_view(&request) {
                        Ok(v) => v,
                        Err(why) => {
                            self.pending_snap = None;
                            self.image_failed(request.block, request.silent, why);
                            return;
                        }
                    };
                    let target = traced::ImageTarget {
                        width,
                        height,
                        supersample: u32::from(request.supersample),
                        transparent: request.background == ImageBackground::Transparent,
                        background: match request.background {
                            ImageBackground::Colour(rgb) => Some(linear_of(rgb)),
                            _ => None,
                        },
                        deep: request.bits == 16,
                    };
                    // Refused while a rebuild or a compile is in hand: ask for another frame
                    // and try again, counted against the wait.
                    if self.traced.begin_image(&self.wgpu_render_state, target, view).is_ok()
                        && let Some(snap) = &mut self.pending_snap
                    {
                        snap.stage = SnapStage::Offscreen { begun: Some(Instant::now()) };
                        snap.since = Instant::now();
                        snap.compiling = Duration::ZERO;
                    }
                    ctx.request_repaint();
                    return;
                }
                match self.traced.drive_image(&self.wgpu_render_state) {
                    Some(traced::ImageStep::Running) | None => ctx.request_repaint(),
                    Some(traced::ImageStep::Done(outcome, cost)) => self.start_encoding(outcome, Some(cost)),
                    Some(traced::ImageStep::Failed(why)) => {
                        let (block, silent) = (snap.request.block, snap.request.silent);
                        self.pending_snap = None;
                        self.image_failed(block, silent, why);
                    }
                }
            }
            SnapStage::Encoding(rx) => match rx.try_recv() {
                Ok(Ok(written)) => {
                    let request = snap.request.clone();
                    self.pending_snap = None;
                    self.image_written(&request, written);
                }
                Ok(Err(why)) => {
                    let (block, silent) = (snap.request.block, snap.request.silent);
                    self.pending_snap = None;
                    self.image_failed(block, silent, why);
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint(),
                Err(mpsc::TryRecvError::Disconnected) => {
                    let (block, silent) = (snap.request.block, snap.request.silent);
                    self.pending_snap = None;
                    self.image_failed(block, silent, "the encoder stopped without writing the file".into());
                }
            },
        }
    }

    /// Hands a finished picture to a worker to encode and write, and moves the snapshot to its
    /// last stage. The interface never waits on a PNG's compression.
    fn start_encoding(&mut self, outcome: traced::ImageOutcome, cost: Option<traced::ImageCost>) {
        let document = self.document_name();
        let Some(snap) = &mut self.pending_snap else { return };
        let request = snap.request.clone();
        let started = snap.started;
        let (tx, rx) = mpsc::channel();
        snap.stage = SnapStage::Encoding(rx);
        std::thread::Builder::new()
            .name("image-export".into())
            .spawn(move || {
                let result = encode_and_write(outcome, &request, &document).map(|(bytes, width, height)| ImageWritten {
                    path: request.path.clone(),
                    width,
                    height,
                    bytes,
                    took: started.elapsed(),
                    cost,
                });
                let _ = tx.send(result);
            })
            .expect("a thread for the encoder");
    }

    /// The document's name for a file's metadata and the `{doc}` token: its file's stem, or
    /// "untitled".
    pub(crate) fn document_name(&self) -> String {
        self.file
            .as_ref()
            .and_then(|(p, _)| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .unwrap_or_else(|| "untitled".to_string())
    }

    /// The folder a block's files go into: its folder under the document's, or under the
    /// Pictures folder's `FieldForm` for a document that has no file yet. Made if it is not there.
    pub(crate) fn image_folder(&self, block: &SaveImage) -> Result<PathBuf, String> {
        let base = match self.file.as_ref().and_then(|(p, _)| p.parent().map(Path::to_path_buf)) {
            Some(dir) => dir,
            None => pictures_folder().join("FieldForm"),
        };
        let folder = if block.folder.trim().is_empty() { base } else { base.join(block.folder.trim()) };
        std::fs::create_dir_all(&folder).map_err(|e| format!("could not make {}: {e}", folder.display()))?;
        Ok(folder)
    }
}

/// The user's Pictures folder, or the home folder, or where the program runs.
fn pictures_folder() -> PathBuf {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let pictures = home.join("Pictures");
    if pictures.is_dir() { pictures } else { home }
}

/// Encodes the picture as the request asks and writes it. Written to a `.part` beside the
/// file and renamed into place, so a cancel or a failure leaves nothing behind that looks
/// like a finished image. Returns the bytes written and the picture's size.
pub(crate) fn encode_and_write(
    outcome: traced::ImageOutcome,
    request: &ImageRequest,
    document: &str,
) -> Result<(u64, u32, u32), String> {
    let transparent = request.background == ImageBackground::Transparent;
    let text = metadata(request, document);
    let (bytes, width, height) = match (&outcome, request.format) {
        (traced::ImageOutcome::Rgba8(img), ImageFormat::Png) => {
            (encode_png8(img.width, img.height, &img.rgba, transparent, &text)?, img.width, img.height)
        }
        (traced::ImageOutcome::RgbaF32(img), ImageFormat::Png) => {
            (encode_png16(img.width, img.height, &img.rgba, transparent, &text)?, img.width, img.height)
        }
        (traced::ImageOutcome::Rgba8(img), ImageFormat::Jpeg) => {
            (encode_jpeg(img.width, img.height, &img.rgba, request.jpeg_quality, &request.line)?, img.width, img.height)
        }
        (traced::ImageOutcome::RgbaF32(_), ImageFormat::Jpeg) => {
            return Err("a JPEG is 8-bit; 16-bit depth is PNG only".into());
        }
    };
    let part = request.path.with_extension(format!("{}.part", request.format.extension()));
    std::fs::write(&part, &bytes).map_err(|e| format!("{} could not be written: {e}", file_name(&request.path)))?;
    if let Err(e) = std::fs::rename(&part, &request.path) {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{} could not be written: {e}", file_name(&request.path)));
    }
    Ok((bytes.len() as u64, width, height))
}

/// The text a file carries: the line that reproduces it, who wrote it, and from what.
fn metadata(request: &ImageRequest, document: &str) -> Vec<(String, String)> {
    vec![
        ("Comment".to_string(), request.line.clone()),
        ("Software".to_string(), format!("FieldForm {}", env!("CARGO_PKG_VERSION"))),
        ("Source".to_string(), document.to_string()),
    ]
}

/// An 8-bit PNG, RGB or RGBA, with an sRGB chunk and the text chunks.
pub(crate) fn encode_png8(width: u32, height: u32, rgba: &[u8], alpha: bool, text: &[(String, String)]) -> Result<Vec<u8>, String> {
    let channels = if alpha { 4 } else { 3 };
    let mut pixels = Vec::with_capacity(rgba.len() / 4 * channels);
    for px in rgba.chunks_exact(4) {
        pixels.extend_from_slice(&px[..channels]);
    }
    write_png(width, height, &pixels, alpha, png::BitDepth::Eight, text)
}

/// A 16-bit PNG from display-encoded values in `[0, 1]`: each quantised to sixteen bits from
/// the float, never expanded from eight.
pub(crate) fn encode_png16(width: u32, height: u32, rgba: &[f32], alpha: bool, text: &[(String, String)]) -> Result<Vec<u8>, String> {
    let channels = if alpha { 4 } else { 3 };
    let mut pixels = Vec::with_capacity(rgba.len() / 4 * channels * 2);
    for px in rgba.chunks_exact(4) {
        for &v in &px[..channels] {
            let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            pixels.extend_from_slice(&q.to_be_bytes());
        }
    }
    write_png(width, height, &pixels, alpha, png::BitDepth::Sixteen, text)
}

fn write_png(width: u32, height: u32, pixels: &[u8], alpha: bool, depth: png::BitDepth, text: &[(String, String)]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(if alpha { png::ColorType::Rgba } else { png::ColorType::Rgb });
        encoder.set_depth(depth);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        for (key, value) in text {
            if value.is_empty() {
                continue;
            }
            // Latin-1 keeps it a `tEXt`; anything else is an `iTXt`, which the crate chooses.
            if value.is_ascii() {
                encoder.add_text_chunk(key.clone(), value.clone()).map_err(|e| e.to_string())?;
            } else {
                encoder.add_itxt_chunk(key.clone(), value.clone()).map_err(|e| e.to_string())?;
            }
        }
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(pixels).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// A JPEG at `quality`, with `comment` in a COM segment after the start-of-image marker.
pub(crate) fn encode_jpeg(width: u32, height: u32, rgba: &[u8], quality: u8, comment: &str) -> Result<Vec<u8>, String> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    let mut out = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100));
    use image::ImageEncoder as _;
    encoder
        .write_image(&rgb, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| e.to_string())?;
    Ok(with_jpeg_comment(out, comment))
}

/// `jpeg` with a COM segment holding `comment` spliced in after SOI. A comment too long for
/// one segment is cut, since the line it carries is a sentence and not a document.
fn with_jpeg_comment(jpeg: Vec<u8>, comment: &str) -> Vec<u8> {
    if comment.is_empty() || jpeg.len() < 2 || jpeg[..2] != [0xFF, 0xD8] {
        return jpeg;
    }
    let bytes = comment.as_bytes();
    let bytes = &bytes[..bytes.len().min(65_533)];
    let len = (bytes.len() + 2) as u16;
    let mut out = Vec::with_capacity(jpeg.len() + bytes.len() + 4);
    out.extend_from_slice(&jpeg[..2]);
    out.extend_from_slice(&[0xFF, 0xFE]);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// Reads the picture keys a line may carry into `request`, refusing a value by name. The keys
/// are the Save Image block's own fields (`console_keys::VERBS` lists them for `snap` and
/// `saveimage` alike): `view`, `zoom`, `width`, `height`, `quality`, `ss`, `bg`, `format`,
/// `bits`, `jpgq`, `overlays`, `overwrite`.
pub(crate) fn read_image_keys(request: &mut ImageRequest, args: &Args) -> Result<(), String> {
    let number = |key: &str| -> Result<Option<f64>, String> {
        match args.get(key) {
            None => Ok(None),
            Some(v) => v.as_f64().map(Some).ok_or_else(|| format!("`{key}=` wants a number, not {:?}", v.text())),
        }
    };
    let whole = |key: &str| -> Result<Option<u32>, String> {
        Ok(number(key)?.map(|n| n.round().max(0.0) as u32))
    };
    if let Some(v) = args.get("view") {
        request.view = Some(v.text());
    }
    if let Some(z) = number("zoom")? {
        request.zoom = Some(z as f32);
    }
    if let Some(w) = whole("width")? {
        request.width = w;
    }
    if let Some(h) = whole("height")? {
        request.height = h;
    }
    if let Some(q) = args.get("quality") {
        request.quality = match q.text().to_ascii_lowercase().as_str() {
            "still" => ImageQuality::Still,
            "displayed" | "screen" => ImageQuality::Displayed,
            other => return Err(format!("`quality={other}` is neither still nor displayed")),
        };
    }
    if let Some(s) = whole("ss")? {
        request.supersample = s.min(255) as u8;
    }
    if let Some(bg) = args.get("bg") {
        request.background = parse_background(&bg.text())?;
    }
    if let Some(f) = args.get("format") {
        request.format = match f.text().to_ascii_lowercase().as_str() {
            "png" => ImageFormat::Png,
            "jpg" | "jpeg" => ImageFormat::Jpeg,
            other => return Err(format!("`format={other}` is neither png nor jpg")),
        };
    }
    if let Some(b) = whole("bits")? {
        request.bits = b.min(255) as u8;
    }
    if let Some(q) = whole("jpgq")? {
        request.jpeg_quality = q.min(255) as u8;
    }
    if let Some(o) = args.get("overlays") {
        match o.text().to_ascii_lowercase().as_str() {
            "none" => {}
            other => {
                return Err(format!(
                    "`overlays={other}` is not drawn: the grid, the cube and the handles are the interface's and the \
                     march knows nothing of them, so a saved image is the part alone (`overlays=none`)"
                ))
            }
        }
    }
    if let Some(o) = args.get("overwrite") {
        request.overwrite = match o {
            Value::Bool(b) => *b,
            other => match other.text().to_ascii_lowercase().as_str() {
                "true" | "yes" | "on" | "1" => true,
                "false" | "no" | "off" | "0" => false,
                _ => return Err(format!("`overwrite=` wants true or false, not {:?}", other.text())),
            },
        };
    }
    Ok(())
}

/// The `snap` line that reproduces `request`: every key whose value is not the default, so the
/// line is as short as the picture lets it be and says everything that made it.
pub(crate) fn snap_line(request: &ImageRequest) -> String {
    let mut line = format!("snap file={}", quoted(&request.path.to_string_lossy().replace('\\', "/")));
    if let Some(view) = &request.view {
        line.push_str(&format!(" view={view}"));
    }
    if let Some(zoom) = request.zoom {
        line.push_str(&format!(" zoom={zoom}"));
    }
    if request.quality == ImageQuality::Still {
        line.push_str(" quality=still");
        if request.width != 0 {
            line.push_str(&format!(" width={}", request.width));
        }
        if request.height != 0 {
            line.push_str(&format!(" height={}", request.height));
        }
        if request.supersample != 1 {
            line.push_str(&format!(" ss={}", request.supersample));
        }
        if request.background != ImageBackground::Environment {
            line.push_str(&format!(" bg={}", background_word(request.background)));
        }
        if request.bits != 8 {
            line.push_str(&format!(" bits={}", request.bits));
        }
    }
    if request.format == ImageFormat::Jpeg && request.jpeg_quality != 92 {
        line.push_str(&format!(" jpgq={}", request.jpeg_quality));
    }
    if !request.overwrite {
        line.push_str(" overwrite=false");
    }
    line
}

/// A value quoted as the console reads one back: in double quotes, with a quote inside
/// escaped. A path with a space in it would otherwise split into two words.
pub(crate) fn quoted(text: &str) -> String {
    format!("\"{}\"", text.replace('"', "\\\""))
}

/// The text chunks of a PNG, for a test that reads a file back, as `(key, value)` pairs.
#[cfg(test)]
pub(crate) fn png_text_chunks(bytes: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut at = 8;
    while at + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
        let kind = &bytes[at + 4..at + 8];
        let data = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        match kind {
            b"tEXt" => {
                if let Some(split) = data.iter().position(|b| *b == 0) {
                    out.push((
                        String::from_utf8_lossy(&data[..split]).to_string(),
                        String::from_utf8_lossy(&data[split + 1..]).to_string(),
                    ));
                }
            }
            b"iTXt" => {
                if let Some(split) = data.iter().position(|b| *b == 0) {
                    let key = String::from_utf8_lossy(&data[..split]).to_string();
                    // keyword NUL compression-flag compression-method language NUL translated NUL text
                    let rest = &data[split + 1..];
                    if rest.len() >= 2 {
                        let rest = &rest[2..];
                        let after_language = rest.iter().position(|b| *b == 0).map(|p| &rest[p + 1..]).unwrap_or(&[]);
                        let text = after_language.iter().position(|b| *b == 0).map(|p| &after_language[p + 1..]).unwrap_or(&[]);
                        out.push((key, String::from_utf8_lossy(text).to_string()));
                    }
                }
            }
            b"IEND" => break,
            _ => {}
        }
        at += 12 + len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill<'a>() -> NameFill<'a> {
        NameFill { doc: "bracket", block: "hero", width: 3840, height: 2160, view: "iso", when: ("2026-11-02".into(), "161500".into()) }
    }

    #[test]
    fn every_token_of_a_pattern_is_filled_and_an_unknown_one_is_left_as_typed() {
        let f = fill();
        assert_eq!(fill_pattern("{doc}_{block}_{n:04}", &f, 7), "bracket_hero_0007");
        assert_eq!(fill_pattern("{date}-{time}-{w}x{h}-{view}-{n}", &f, 12), "2026-11-02-161500-3840x2160-iso-12");
        assert_eq!(fill_pattern("x{nope}y{n:02}", &f, 3), "x{nope}y03");
        assert_eq!(fill_pattern("unclosed{n", &f, 3), "unclosed{n");
    }

    #[test]
    fn a_file_name_keeps_nothing_the_file_system_refuses() {
        assert_eq!(safe_file_name("a:b/c\\d*e?f\"g<h>i|j"), "a_b_c_d_e_f_g_h_i_j");
    }

    #[test]
    fn the_next_path_advances_past_a_file_that_is_there_unless_overwriting() {
        let dir = std::env::temp_dir().join(format!("fieldform-next-path-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut block = SaveImage::new(fieldform_scene::PartId::new(), 100, 50);
        std::fs::write(dir.join("untitled_Save Image_0003.png"), b"x").unwrap();
        let f = NameFill { doc: "untitled", block: "Save Image", width: 100, height: 50, view: "", when: ("".into(), "".into()) };
        let (path, n) = next_path(&block, &dir, &f, 3);
        assert_eq!(n, 4, "the counter advanced past the file that is there");
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), "untitled_Save Image_0004.png");
        block.overwrite = true;
        let (path, n) = next_path(&block, &dir, &f, 3);
        assert_eq!((n, path.file_name().unwrap().to_str().unwrap()), (3, "untitled_Save Image_0003.png"));
        block.pattern = "shot.jpg".into();
        block.format = ImageFormat::Jpeg;
        let (path, _) = next_path(&block, &dir, &f, 1);
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), "shot.jpg", "an extension typed is kept, not doubled");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_background_is_read_as_typed_and_decodes_to_the_bytes_it_will_encode_back_to() {
        assert_eq!(parse_background("env").unwrap(), ImageBackground::Environment);
        assert_eq!(parse_background("transparent").unwrap(), ImageBackground::Transparent);
        assert_eq!(parse_background("color:#202020").unwrap(), ImageBackground::Colour([32, 32, 32]));
        assert_eq!(parse_background("colour:FF8000").unwrap(), ImageBackground::Colour([255, 128, 0]));
        assert!(parse_background("color:#12").is_err());
        assert!(parse_background("blue").is_err());
        let encode = |c: f32| {
            let v = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
            (v * 255.0).round() as u8
        };
        for byte in [0u8, 1, 32, 127, 128, 200, 254, 255] {
            let lin = linear_of([byte, byte, byte]);
            assert_eq!(encode(lin[0]), byte, "sRGB {byte} survives the decode and the encode");
        }
    }

    #[test]
    fn a_png_carries_its_line_in_a_text_chunk_and_a_jpeg_in_a_comment() {
        let rgba = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
        let text = vec![("Comment".to_string(), "snap file=\"a.png\"".to_string()), ("Software".to_string(), "FieldForm 0.3.0".to_string())];
        let png = encode_png8(2, 1, &rgba, false, &text).unwrap();
        let chunks = png_text_chunks(&png);
        assert!(chunks.contains(&("Comment".to_string(), "snap file=\"a.png\"".to_string())), "{chunks:?}");
        let decoded = image::load_from_memory(&png).unwrap().to_rgb8();
        assert_eq!(decoded.get_pixel(1, 0).0, [40, 50, 60], "the pixels are the bytes given, through the sRGB chunk");

        let deep = encode_png16(2, 1, &[0.5, 0.25, 1.0, 1.0, 0.0, 0.5, 0.75, 1.0], true, &text).unwrap();
        let decoded = image::load_from_memory(&deep).unwrap().to_rgba16();
        assert_eq!(decoded.get_pixel(0, 0).0, [32768, 16384, 65535, 65535]);

        let jpeg = encode_jpeg(2, 1, &rgba, 90, "snap file=\"a.jpg\"").unwrap();
        assert_eq!(&jpeg[2..4], &[0xFF, 0xFE], "a COM segment follows SOI");
        let len = u16::from_be_bytes([jpeg[4], jpeg[5]]) as usize;
        assert_eq!(&jpeg[6..4 + len], b"snap file=\"a.jpg\"");
        assert!(image::load_from_memory(&jpeg).is_ok(), "the JPEG still decodes with the comment in it");
    }

    #[test]
    fn a_request_is_refused_for_the_reasons_the_design_names() {
        let mut r = ImageRequest::displayed(PathBuf::from("out.png"));
        assert_eq!(r.refusal((800, 600)), None);
        r.width = 1000;
        assert!(r.refusal((800, 600)).unwrap().contains("quality=still"), "another size is a Still");
        r.quality = ImageQuality::Still;
        assert_eq!(r.refusal((800, 600)), None);
        r.background = ImageBackground::Transparent;
        r.format = ImageFormat::Jpeg;
        r.path = PathBuf::from("out.jpg");
        assert!(r.refusal((800, 600)).unwrap().contains("alpha"));
        r.format = ImageFormat::Png;
        assert!(r.refusal((800, 600)).unwrap().contains("does not match"), "the extension and the format disagree");
        r.path = PathBuf::from("out.png");
        r.bits = 16;
        assert_eq!(r.refusal((800, 600)), None);
        r.supersample = 3;
        assert!(r.refusal((800, 600)).unwrap().contains("ss=3"));
        r.supersample = 2;
        r.width = SaveImage::MAX_SIDE + 1;
        assert!(r.refusal((800, 600)).unwrap().contains("past"));
        assert_eq!(ImageRequest::displayed(PathBuf::from("x.png")).size((800, 600)), (800, 600));
        let mut half = ImageRequest::displayed(PathBuf::from("x.png"));
        half.width = 400;
        assert_eq!(half.size((800, 600)), (400, 300), "one side given keeps the viewport's aspect");
    }

    #[test]
    fn the_calendar_arithmetic_agrees_with_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_759), (2026, 11, 2));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }
}
