// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Save Image block in the application: its verb, its runs, its panel and its clock
//! (backlog item 117, `image-export-spec.md` §6 and §7).
//!
//! The block is a document thing (`fieldform_scene::SaveImage`); everything here about
//! *running* one -- the result of the last save, the per-block log, the `{n}` counter, the
//! automatic capture's timers and whether it is armed -- is the application's and is not saved,
//! for the reason the tessellation's meshes are not: a run is derived from the block, and a
//! file opened later must not start writing pictures on its own.

use crate::image_export::{
    self, next_path, quoted, ImageRequest, ImageWritten, NameFill,
};
use crate::outputs::row_title;
use crate::*;
use fieldform_ops::Command;
use fieldform_scene::{Capture, ImageBackground, ImageFormat, ImageQuality, SaveImage, SaveImageId};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Automatic capture stops when the drive has less than this left.
const DISK_FLOOR_BYTES: u64 = 1 << 30;

/// Everything about running Save Image blocks, none of it saved.
#[derive(Default)]
pub(crate) struct ImageRuns {
    /// What the last save of each block came to.
    pub(crate) results: HashMap<SaveImageId, Result<ImageWritten, String>>,
    /// The next `{n}` each block names, from one each session.
    pub(crate) counters: HashMap<SaveImageId, u32>,
    /// Blocks whose automatic capture is running. **A block opened from a file is not here**:
    /// a document with capture configured opens paused, and *Resume capture* is what arms it.
    pub(crate) armed: HashSet<SaveImageId>,
    pub(crate) captured: HashMap<SaveImageId, u32>,
    pub(crate) skipped: HashMap<SaveImageId, u32>,
    pub(crate) last_capture_at: HashMap<SaveImageId, Instant>,
    /// The document revision each block last captured at or passed over.
    pub(crate) revision_seen: HashMap<SaveImageId, u64>,
    /// Blocks whose size is being edited with the aspect unlocked.
    pub(crate) aspect_free: HashSet<SaveImageId>,
    /// The block being edited and when, with the blocks as they were when the edit began --
    /// the rule `set_outputs` follows, so a scrub on a number is one entry on the history.
    pub(crate) gesture: Option<(SaveImageId, Instant, Vec<SaveImage>)>,
    /// The last picture asked for, which *Save image again* repeats.
    pub(crate) last_request: Option<ImageRequest>,
}

/// The size presets the panel offers; each writes explicit pixels into the block.
const PRESETS: &[(&str, Preset)] = &[
    ("Viewport (1×)", Preset::Viewport(1)),
    ("2× viewport", Preset::Viewport(2)),
    ("4K UHD (3840 wide)", Preset::Wide(3840)),
    ("5000 wide", Preset::Wide(5000)),
    ("8K (7680 wide)", Preset::Wide(7680)),
];

#[derive(Clone, Copy, PartialEq)]
enum Preset {
    Viewport(u32),
    Wide(u32),
}

impl App {
    /// `saveimage [width= height= quality= ss= bg= format= bits= jpgq= folder= pattern=
    /// overwrite= view= capture= every= max=] ['name']`: a Save Image block, selected, beside
    /// the other outputs. With no size it takes the viewport's. File › Save Image... is this
    /// with no keys.
    pub(crate) fn saveimage_command(&mut self, args: &Args) {
        let viewport = self.viewport_pixels();
        let mut block = SaveImage::new(self.active_part, viewport.0, viewport.1);
        // **Where the picture goes, asked first** (the owner, 2026-11-07: File › Save Image "should
        // pop" a dialog that sets "the directory and name", and the block stays for saving again).
        // It used to add the block and say so in the console, which from the menu looked like
        // nothing at all. `file=` is the same answer typed, and a script, which cannot answer a
        // dialog, keeps the old quiet block when it names nothing.
        let file = match args.get("file").map(|v| v.text()) {
            Some(text) if text.trim().is_empty() => {
                self.warn("`file=` wants a path, such as `file=\"C:/renders/shot.png\"`");
                return;
            }
            Some(text) => Some(std::path::PathBuf::from(text.trim())),
            None if args.iter().next().is_none() && self.script.is_none() && !self.script_running && !cfg!(test) => {
                let start = self.image_folder(&block).ok();
                let mut dialog = rfd::FileDialog::new()
                    .set_title("Save Image")
                    .add_filter("PNG image", &["png"])
                    .add_filter("JPEG image", &["jpg", "jpeg"])
                    .set_file_name(format!("{}.png", self.document_name()));
                if let Some(dir) = start {
                    dialog = dialog.set_directory(dir);
                }
                match dialog.save_file() {
                    Some(path) => Some(path),
                    None => return, // cancelled: nothing made
                }
            }
            None => None,
        };
        // The picture keys read the way `snap` reads them, so the two cannot drift.
        let mut request = ImageRequest::displayed(std::path::PathBuf::new());
        request.quality = ImageQuality::Still;
        if let Err(why) = image_export::read_image_keys(&mut request, args) {
            self.warn(why);
            return;
        }
        if args.get("quality").is_none() {
            request.quality = ImageQuality::Still;
        }
        if request.width != 0 || request.height != 0 {
            let (w, h) = request.size(viewport);
            block.width = w;
            block.height = h;
        }
        block.view = request.view;
        block.quality = request.quality;
        block.supersample = request.supersample;
        block.background = request.background;
        block.format = request.format;
        block.bits = request.bits;
        block.jpeg_quality = request.jpeg_quality;
        block.overwrite = request.overwrite;
        if let Some(name) = args.get("name").map(|v| v.text()) {
            let name = name.trim().to_string();
            if !name.is_empty() {
                block.name = Some(name);
            }
        }
        if let Some(folder) = args.get("folder").map(|v| v.text()) {
            block.folder = folder;
        }
        if let Some(pattern) = args.get("pattern").map(|v| v.text()) {
            if pattern.trim().is_empty() {
                self.warn("`pattern=` wants a file name, with `{doc}`, `{block}`, `{n:04}` and the rest to fill in");
                return;
            }
            block.pattern = pattern;
        }
        let every = match args.get("every") {
            None => None,
            Some(v) => match v.as_f64() {
                Some(n) if n >= 1.0 => Some(n.round() as u32),
                _ => {
                    self.warn(format!("`every=` wants a whole number over nought, not {:?}", v.text()));
                    return;
                }
            },
        };
        if let Some(mode) = args.get("capture").map(|v| v.text().to_ascii_lowercase()) {
            block.capture = match mode.as_str() {
                "off" => Capture::Off,
                "change" | "onchange" => Capture::OnChange,
                "seconds" | "time" => Capture::EverySeconds(every.unwrap_or(60).max(SaveImage::MIN_CAPTURE_SECONDS)),
                "commits" | "edits" => Capture::EveryCommits(every.unwrap_or(10).max(1)),
                other => {
                    self.warn(format!("`capture={other}` is not off, change, seconds or commits"));
                    return;
                }
            };
        } else if let Some(n) = every {
            block.capture = Capture::EverySeconds(n.max(SaveImage::MIN_CAPTURE_SECONDS));
        }
        if let Some(v) = args.get("max") {
            match v.as_f64() {
                Some(n) if n >= 1.0 => block.max_count = n.round() as u32,
                _ => {
                    self.warn(format!("`max=` wants a whole number over nought, not {:?}", v.text()));
                    return;
                }
            }
        }
        if let Some(path) = &file
            && let Err(why) = aim_block_at(&mut block, path)
        {
            self.warn(why);
            return;
        }
        let probe = std::path::PathBuf::from(format!("x.{}", block.format.extension()));
        if let Some(why) = ImageRequest::of_block(&block, probe, false)
            .refusal((block.width, block.height))
        {
            self.warn(why);
            return;
        }
        let id = block.id;
        let title = block.title().to_string();
        let armed = block.capture != Capture::Off;
        let mut images = self.doc.images.clone();
        images.push(block);
        self.set_images(images, None);
        if armed {
            // Typed on, so running: only a block read from a file waits to be resumed.
            self.images.armed.insert(id);
        }
        self.select_block(outputs::BlockRef::Image(id));
        if file.is_some() {
            // Asked where, so saved there now; the block is what saves it again.
            self.say(format!(
                "Added {title}, saving the picture now: its block saves it again, or `snap block={}`",
                quoted(&title)
            ));
            self.run_image_block(id, false);
            return;
        }
        self.say(format!(
            "Added {title}: Save now writes the picture, and a line can run it with `snap block={}`",
            quoted(&title)
        ));
    }

    /// The Save Image block called `name`, by its title, case aside.
    pub(crate) fn image_block_named(&self, name: &str) -> Option<SaveImageId> {
        let wanted = name.trim().trim_matches('"').trim_matches('\'');
        self.doc.images.iter().find(|b| b.title().eq_ignore_ascii_case(wanted)).map(|b| b.id)
    }

    /// Sets every Save Image block at once, undoably -- one entry per edit, or one per gesture
    /// on the same block within the window, as `set_outputs` keeps its lists.
    pub(crate) fn set_images(&mut self, images: Vec<SaveImage>, gesture: Option<SaveImageId>) {
        let now = self.doc.images.clone();
        if now == images {
            return;
        }
        let continues = gesture.is_some()
            && self.stack.peek_undo() == Some("images.set")
            && self
                .images
                .gesture
                .as_ref()
                .is_some_and(|(b, at, _)| Some(*b) == gesture && at.elapsed() < GESTURE_WINDOW);
        let before = match (&self.images.gesture, continues) {
            (Some((_, _, before)), true) => before.clone(),
            _ => now,
        };
        self.images.gesture = gesture.map(|b| (b, Instant::now(), before.clone()));
        self.gesture = None;
        let cmd: Box<dyn Command> = Box::new(fieldform_ops::SetImages::new(before, images));
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

    /// Runs a block's settings: resolves its file name and begins the picture. `silent` is an
    /// automatic capture, which says nothing in the console.
    pub(crate) fn run_image_block(&mut self, id: SaveImageId, silent: bool) {
        let Some(block) = self.doc.images.iter().find(|b| b.id == id).cloned() else { return };
        let folder = match self.image_folder(&block) {
            Ok(f) => f,
            Err(why) => {
                self.image_failed(Some(id), silent, why);
                return;
            }
        };
        let doc = self.document_name();
        let title = block.title().to_string();
        let fill = NameFill {
            doc: &doc,
            block: &title,
            width: block.width,
            height: block.height,
            view: block.view.as_deref().unwrap_or("current"),
            when: image_export::now_tokens(),
        };
        let from = self.images.counters.get(&id).copied().unwrap_or(1);
        let (path, n) = next_path(&block, &folder, &fill, from);
        self.images.counters.insert(id, n + 1);
        let mut request = ImageRequest::of_block(&block, path, silent);
        // What the file says made it: the block by name, which is how a line runs it, and the
        // line that would make the same picture without the block.
        request.line = format!("snap block={}", quoted(&title));
        match self.begin_image(request.clone()) {
            Ok(()) => {
                self.images.last_request = Some(request);
                self.images.results.remove(&id);
            }
            Err(why) => self.image_failed(Some(id), silent, why),
        }
    }

    /// *Save image again*: the last picture asked for, with a fresh file name where it came
    /// from a block and the same file where it was a `snap`.
    pub(crate) fn save_image_again(&mut self) {
        match self.images.last_request.clone() {
            Some(request) => match request.block {
                Some(id) if self.doc.images.iter().any(|b| b.id == id) => self.run_image_block(id, false),
                _ => {
                    let mut again = request;
                    again.overwrite = true;
                    if let Err(why) = self.begin_image(again) {
                        self.warn(why);
                    }
                }
            },
            None => self.warn("No image saved yet to save again — File › Save Image... makes a block"),
        }
    }

    /// A picture written: said in the console, automatic captures included, and its result kept
    /// on its block.
    ///
    /// **The console is the one log** (the owner, 2026-11-07: "remove the 'log' from the save
    /// image block - that output should be shown in the console"). A block used to keep twenty
    /// lines of its own and say nothing of an automatic capture, on the reading that twenty an
    /// hour in the console would bury what was typed; the owner ruled the other way, so what a
    /// block did is found where everything else is said.
    pub(crate) fn image_written(&mut self, request: &ImageRequest, written: ImageWritten) {
        self.say(written.said());
        if let Some(id) = request.block {
            self.images.results.insert(id, Ok(written));
            if request.silent {
                *self.images.captured.entry(id).or_default() += 1;
            }
        }
    }

    /// A picture not written: said in the console, and kept on its block as its last result. An
    /// automatic capture's failure is said as a line rather than a warning, since a warning stops
    /// a script that was not told to keep going and the capture is not a line of it.
    pub(crate) fn image_failed(&mut self, block: Option<SaveImageId>, silent: bool, why: String) {
        if silent {
            self.say(format!("Save Image: {why}"));
        } else {
            self.warn(why.clone());
        }
        if let Some(id) = block {
            self.images.results.insert(id, Err(why));
        }
    }

    /// Automatic capture: each frame, every armed block is asked whether it is due, and one
    /// that is runs -- never during a drag or a tool session, never while the view is still
    /// converging, never while another picture is in flight (that one is skipped and counted,
    /// not queued), and never past its count or a drive with less than a gibibyte left.
    pub(crate) fn drive_captures(&mut self, ctx: &egui::Context) {
        if self.images.armed.is_empty() {
            return;
        }
        let interacting = ctx.input(|i| i.pointer.any_down() || i.pointer.is_moving() || !i.events.is_empty());
        let quiet = self.session.is_none() && self.cage_session.is_none() && !interacting;
        let revision = self.document_moves;
        let ids: Vec<SaveImageId> = self.images.armed.iter().copied().collect();
        for id in ids {
            let Some(block) = self.doc.images.iter().find(|b| b.id == id).cloned() else {
                self.images.armed.remove(&id);
                continue;
            };
            let due = match block.capture {
                Capture::Off => {
                    self.images.armed.remove(&id);
                    continue;
                }
                Capture::EverySeconds(s) => self
                    .images
                    .last_capture_at
                    .get(&id)
                    .is_none_or(|at| at.elapsed().as_secs() >= u64::from(s.max(SaveImage::MIN_CAPTURE_SECONDS))),
                Capture::OnChange => self.images.revision_seen.get(&id).copied() != Some(revision),
                Capture::EveryCommits(n) => {
                    let seen = self.images.revision_seen.get(&id).copied().unwrap_or(revision);
                    revision.saturating_sub(seen) >= u64::from(n.max(1))
                }
            };
            if !due {
                continue;
            }
            if block.capture == Capture::OnChange || matches!(block.capture, Capture::EveryCommits(_)) {
                // Wait for the edit to settle: a drag is dozens of revisions a second.
                if !quiet || !self.traced.is_settled() {
                    continue;
                }
            } else if !quiet || !self.traced.is_settled() {
                // A timed capture that falls in a drag waits for the next idle, as the design
                // says, and does not count the wait as a skip.
                continue;
            }
            let count = self.images.captured.get(&id).copied().unwrap_or(0);
            if count >= block.max_count {
                self.images.armed.remove(&id);
                self.say(format!("{} stopped after {} captures", block.title(), block.max_count));
                continue;
            }
            if let Ok(folder) = self.image_folder(&block)
                && let Some(free) = free_disk_bytes(&folder)
                && free < DISK_FLOOR_BYTES
            {
                self.images.armed.remove(&id);
                self.say(format!("{} stopped: only {} free on the drive", block.title(), memory::describe(free)));
                continue;
            }
            // Due now, whatever happens next: a skip is counted against this moment, not
            // retried every frame until the other picture finishes.
            self.images.last_capture_at.insert(id, Instant::now());
            self.images.revision_seen.insert(id, revision);
            if self.pending_snap.is_some() {
                *self.images.skipped.entry(id).or_default() += 1;
                continue;
            }
            self.run_image_block(id, true);
        }
    }

    /// The Save Image block's panel: the request top to bottom, then Save now and what the
    /// last save came to.
    pub(crate) fn image_block(&mut self, ui: &mut egui::Ui, id: SaveImageId) {
        let Some(before) = self.doc.images.iter().find(|b| b.id == id).cloned() else { return };
        let mut b = before.clone();
        let selected = self.selected_block == Some(outputs::BlockRef::Image(id));
        let running = self.pending_snap.as_ref().is_some_and(|s| s.request.block == Some(id));
        let viewport = self.viewport_pixels();
        let mut save = false;
        let mut open_folder = false;
        let mut arm: Option<bool> = None;

        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("image", id.raw())),
            true,
        );
        state
            .show_header(ui, |ui| {
                if running {
                    ui.add_enabled(false, egui::Button::new("Saving…").small());
                } else {
                    save = ui.small_button("Save now").on_hover_text("Writes the picture as these settings say.").clicked();
                }
                let title = row_title(before.name.as_deref(), SaveImage::DEFAULT_NAME);
                let Some(row) = self.list_row(ui, RowKey::Image(id), &title, selected) else {
                    return;
                };
                if row.clicked() || row.secondary_clicked() {
                    self.select_block(outputs::BlockRef::Image(id));
                }
                row.context_menu(|ui| self.block_menu(ui, outputs::BlockRef::Image(id)));
            })
            .body(|ui| {
                // Camera.
                ui.horizontal(|ui| {
                    row_label(ui, "Camera:");
                    let mut current = b.view.clone().unwrap_or_default();
                    egui::ComboBox::from_id_salt(("image camera", id.raw()))
                        .selected_text(if current.is_empty() { "Current view".to_string() } else { current.clone() })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut current, String::new(), "Current view");
                            for name in ["iso", "front", "back", "left", "right", "top", "bottom"] {
                                ui.selectable_value(&mut current, name.to_string(), name);
                            }
                        });
                    b.view = (!current.is_empty()).then_some(current);
                })
                .response
                .on_hover_text("Current view is the camera as it stands when the picture is taken; a named view aims a copy of it, and your own view does not move.");
                // Size.
                ui.horizontal(|ui| {
                    row_label(ui, "Size:");
                    let preset_now = PRESETS.iter().find(|(_, p)| preset_size(*p, viewport) == (b.width, b.height)).map(|(n, _)| *n).unwrap_or("Custom");
                    egui::ComboBox::from_id_salt(("image preset", id.raw()))
                        .selected_text(preset_now)
                        .show_ui(ui, |ui| {
                            for (name, preset) in PRESETS {
                                if ui.selectable_label(preset_now == *name, *name).clicked() {
                                    (b.width, b.height) = preset_size(*preset, viewport);
                                }
                            }
                        });
                });
                ui.horizontal(|ui| {
                    row_label(ui, "");
                    let locked = !self.images.aspect_free.contains(&id);
                    let aspect = before.width as f64 / before.height.max(1) as f64;
                    let mut w = b.width as f64;
                    let mut h = b.height as f64;
                    let wr = ui.add(egui::DragValue::new(&mut w).speed(8.0).suffix(" px"));
                    ui.label("×");
                    let hr = ui.add(egui::DragValue::new(&mut h).speed(8.0).suffix(" px"));
                    if wr.changed() {
                        b.width = w.round().clamp(1.0, SaveImage::MAX_SIDE as f64) as u32;
                        if locked {
                            b.height = (b.width as f64 / aspect).round().clamp(1.0, SaveImage::MAX_SIDE as f64) as u32;
                        }
                    } else if hr.changed() {
                        b.height = h.round().clamp(1.0, SaveImage::MAX_SIDE as f64) as u32;
                        if locked {
                            b.width = (b.height as f64 * aspect).round().clamp(1.0, SaveImage::MAX_SIDE as f64) as u32;
                        }
                    }
                    let mut lock = locked;
                    if ui.toggle_value(&mut lock, "Aspect").on_hover_text("Locked: the picture frames exactly as the viewport, only with more pixels. Unlocked: the vertical view is kept and the sides extend or crop.").changed() {
                        if lock {
                            self.images.aspect_free.remove(&id);
                        } else {
                            self.images.aspect_free.insert(id);
                        }
                    }
                });
                let tiles = self.traced.image_tiles(&self.wgpu_render_state.device, b.width, b.height, b.bits == 16);
                if tiles > 1 && b.quality == ImageQuality::Still {
                    ui.label(egui::RichText::new(format!("Drawn in {tiles} tiles: more than one set of targets holds.")).small().weak());
                }
                // Quality.
                ui.horizontal(|ui| {
                    row_label(ui, "Quality:");
                    egui::ComboBox::from_id_salt(("image quality", id.raw()))
                        .selected_text(match b.quality { ImageQuality::Still => "Still", ImageQuality::Displayed => "Displayed" })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut b.quality, ImageQuality::Still, "Still").on_hover_text("A fresh render at the size above, converged as the viewport's still is.");
                            ui.selectable_value(&mut b.quality, ImageQuality::Displayed, "Displayed").on_hover_text("The pixels on screen, at the viewport's own size, with nothing re-rendered.");
                        });
                    if b.quality == ImageQuality::Still {
                        ui.label("Supersample");
                        egui::ComboBox::from_id_salt(("image ss", id.raw()))
                            .selected_text(format!("{}×", b.supersample))
                            .show_ui(ui, |ui| {
                                for s in [1u8, 2, 4] {
                                    ui.selectable_value(&mut b.supersample, s, format!("{s}×"));
                                }
                            })
                            .response
                            .on_hover_text("Times the samples per pixel, averaged in linear light before the tonemap.");
                    }
                });
                if b.quality == ImageQuality::Displayed {
                    (b.width, b.height) = viewport;
                    b.supersample = 1;
                    b.background = ImageBackground::Environment;
                    b.bits = 8;
                }
                // Background.
                ui.horizontal(|ui| {
                    row_label(ui, "Background:");
                    let word = |bg: ImageBackground| match bg {
                        ImageBackground::Environment => "Environment".to_string(),
                        ImageBackground::Colour(_) => "Colour".to_string(),
                        ImageBackground::Transparent => "Transparent".to_string(),
                    };
                    let jpeg = b.format == ImageFormat::Jpeg;
                    egui::ComboBox::from_id_salt(("image bg", id.raw()))
                        .selected_text(word(b.background))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut b.background, ImageBackground::Environment, "Environment");
                            if !matches!(b.background, ImageBackground::Colour(_)) && ui.selectable_label(false, "Colour").clicked() {
                                b.background = ImageBackground::Colour([32, 32, 32]);
                            } else if matches!(b.background, ImageBackground::Colour(_)) {
                                let _ = ui.selectable_label(true, "Colour");
                            }
                            ui.add_enabled_ui(!jpeg, |ui| {
                                ui.selectable_value(&mut b.background, ImageBackground::Transparent, "Transparent")
                                    .on_disabled_hover_text("A JPEG cannot carry alpha: choose PNG for a transparent background.")
                                    .on_hover_text("Coverage in the alpha channel; the part looks the same and the backdrop goes. Any ground shadow goes with it.");
                            });
                        });
                    if let ImageBackground::Colour(rgb) = &mut b.background {
                        let mut c = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            *rgb = [c.r(), c.g(), c.b()];
                        }
                    }
                });
                // Format.
                ui.horizontal(|ui| {
                    row_label(ui, "Format:");
                    let transparent = b.background == ImageBackground::Transparent;
                    egui::ComboBox::from_id_salt(("image format", id.raw()))
                        .selected_text(match (b.format, b.bits) { (ImageFormat::Png, 16) => "PNG 16-bit", (ImageFormat::Png, _) => "PNG 8-bit", (ImageFormat::Jpeg, _) => "JPEG" })
                        .show_ui(ui, |ui| {
                            if ui.selectable_label(b.format == ImageFormat::Png && b.bits == 8, "PNG 8-bit").clicked() {
                                (b.format, b.bits) = (ImageFormat::Png, 8);
                            }
                            ui.add_enabled_ui(b.quality == ImageQuality::Still, |ui| {
                                if ui.selectable_label(b.format == ImageFormat::Png && b.bits == 16, "PNG 16-bit").on_disabled_hover_text("16-bit depth is for a Still.").clicked() {
                                    (b.format, b.bits) = (ImageFormat::Png, 16);
                                }
                            });
                            ui.add_enabled_ui(!transparent, |ui| {
                                if ui.selectable_label(b.format == ImageFormat::Jpeg, "JPEG").on_disabled_hover_text("A JPEG cannot carry alpha: the background is transparent.").clicked() {
                                    (b.format, b.bits) = (ImageFormat::Jpeg, 8);
                                }
                            });
                        });
                    if b.format == ImageFormat::Jpeg {
                        ui.label("Quality");
                        let mut q = b.jpeg_quality as f64;
                        if ui.add(egui::DragValue::new(&mut q).speed(1.0)).changed() {
                            b.jpeg_quality = q.round().clamp(1.0, 100.0) as u8;
                        }
                    }
                });
                // Where.
                ui.horizontal(|ui| {
                    row_label(ui, "Folder:");
                    ui.add(egui::TextEdit::singleline(&mut b.folder).hint_text("beside the document").desired_width(160.0))
                        .on_hover_text("Relative to the document's folder; an unsaved document writes under Pictures\\FieldForm.");
                    if ui.small_button("Choose…").clicked()
                        && let Some(dir) = rfd::FileDialog::new().pick_folder()
                    {
                        b.folder = dir.to_string_lossy().to_string();
                    }
                });
                ui.horizontal(|ui| {
                    row_label(ui, "Name:");
                    ui.add(egui::TextEdit::singleline(&mut b.pattern).desired_width(160.0))
                        .on_hover_text("{doc} {block} {date} {time} {n} or {n:04} {w} {h} {view}");
                    ui.checkbox(&mut b.overwrite, "Overwrite")
                        .on_hover_text("Off: the counter advances until the name is free; nothing is replaced.");
                });
                // Automatic capture.
                ui.horizontal(|ui| {
                    row_label(ui, "Capture:");
                    let word = match b.capture { Capture::Off => "Off", Capture::EverySeconds(_) => "Every N seconds", Capture::OnChange => "On change", Capture::EveryCommits(_) => "Every N edits" };
                    egui::ComboBox::from_id_salt(("image capture", id.raw()))
                        .selected_text(word)
                        .show_ui(ui, |ui| {
                            if ui.selectable_label(b.capture == Capture::Off, "Off").clicked() { b.capture = Capture::Off; }
                            if ui.selectable_label(matches!(b.capture, Capture::EverySeconds(_)), "Every N seconds").clicked() { b.capture = Capture::EverySeconds(60); }
                            if ui.selectable_label(b.capture == Capture::OnChange, "On change").on_hover_text("After any committed edit, once the view has converged.").clicked() { b.capture = Capture::OnChange; }
                            if ui.selectable_label(matches!(b.capture, Capture::EveryCommits(_)), "Every N edits").clicked() { b.capture = Capture::EveryCommits(10); }
                        });
                    match &mut b.capture {
                        Capture::EverySeconds(s) => {
                            let mut v = *s as f64;
                            if ui.add(egui::DragValue::new(&mut v).speed(1.0).suffix(" s")).changed() {
                                *s = v.round().max(SaveImage::MIN_CAPTURE_SECONDS as f64) as u32;
                            }
                        }
                        Capture::EveryCommits(n) => {
                            let mut v = *n as f64;
                            if ui.add(egui::DragValue::new(&mut v).speed(1.0).suffix(" edits")).changed() {
                                *n = v.round().max(1.0) as u32;
                            }
                        }
                        _ => {}
                    }
                    if b.capture != Capture::Off {
                        ui.label("Max");
                        let mut m = b.max_count as f64;
                        if ui.add(egui::DragValue::new(&mut m).speed(1.0)).changed() {
                            b.max_count = m.round().max(1.0) as u32;
                        }
                        let armed = self.images.armed.contains(&id);
                        if armed {
                            if ui.small_button("Pause capture").clicked() {
                                arm = Some(false);
                            }
                        } else if ui.small_button("Resume capture").on_hover_text("A document opens with capture paused: a file never starts writing pictures on its own.").clicked() {
                            arm = Some(true);
                        }
                        let captured = self.images.captured.get(&id).copied().unwrap_or(0);
                        let skipped = self.images.skipped.get(&id).copied().unwrap_or(0);
                        ui.label(egui::RichText::new(if skipped > 0 { format!("{captured} captured, {skipped} skipped") } else { format!("{captured} captured") }).small().weak());
                    }
                });
                // The last result, and the folder.
                ui.horizontal(|ui| {
                    if ui.small_button("Open folder").clicked() {
                        open_folder = true;
                    }
                    if let Some(progress) = self.traced.image_in_progress().filter(|_| running) {
                        ui.label(egui::RichText::new(format!("Drawing… {:.0} %", progress.fraction() * 100.0)).small());
                    } else if running {
                        ui.label(egui::RichText::new("Saving…").small());
                    }
                });
                match self.images.results.get(&id) {
                    Some(Ok(w)) => {
                        ui.label(egui::RichText::new(w.said()).small().weak()).on_hover_text(w.path.to_string_lossy());
                    }
                    Some(Err(why)) => {
                        ui.label(egui::RichText::new(format!("!! {why}")).small().color(ui.visuals().warn_fg_color));
                    }
                    None => {}
                }
            });

        if b != before && self.doc.images.iter().any(|x| x.id == id) {
            if b.capture != before.capture && b.capture != Capture::Off {
                // Turned on by hand, so running.
                self.images.armed.insert(id);
            }
            let images = self.doc.images.iter().map(|x| if x.id == id { b.clone() } else { x.clone() }).collect();
            self.set_images(images, Some(id));
        }
        if let Some(on) = arm {
            if on {
                self.images.armed.insert(id);
            } else {
                self.images.armed.remove(&id);
            }
        }
        if open_folder {
            match self.image_folder(&before) {
                Ok(folder) => {
                    let _ = std::process::Command::new(if cfg!(windows) { "explorer" } else { "xdg-open" }).arg(folder).spawn();
                }
                Err(why) => self.warn(why),
            }
        }
        if save {
            // The echo: a click says the line that does the same (`interaction-spec.md` §6).
            self.echo_line(format!("snap block={}", quoted(before.title())));
            self.run_image_block(id, false);
        }
    }

    /// The Save Image blocks of the part being edited, in the order they were made.
    pub(crate) fn image_blocks(&self) -> Vec<SaveImageId> {
        let part = self.active_part;
        self.doc.images.iter().filter(|b| b.part == part).map(|b| b.id).collect()
    }

    /// Deletes a Save Image block, undoably.
    pub(crate) fn delete_image_block(&mut self, id: SaveImageId) {
        let name = self.doc.images.iter().find(|b| b.id == id).map(|b| b.title().to_string()).unwrap_or_default();
        let mut images = self.doc.images.clone();
        images.retain(|b| b.id != id);
        self.set_images(images, None);
        self.images.armed.remove(&id);
        self.say(format!("Deleted {name}"));
    }

    /// Renames a Save Image block, undoably; an empty name puts the default back.
    pub(crate) fn rename_image_block(&mut self, id: SaveImageId, text: &str) {
        let name = (!text.trim().is_empty()).then(|| text.trim().to_string());
        let mut images = self.doc.images.clone();
        if let Some(b) = images.iter_mut().find(|b| b.id == id) {
            b.name = name;
        }
        self.set_images(images, None);
    }
}

/// The pixels a preset means for a viewport of this size: the viewport's aspect, kept.
fn preset_size(preset: Preset, viewport: (u32, u32)) -> (u32, u32) {
    let (vw, vh) = (viewport.0.max(1), viewport.1.max(1));
    match preset {
        Preset::Viewport(k) => ((vw * k).min(SaveImage::MAX_SIDE), (vh * k).min(SaveImage::MAX_SIDE)),
        Preset::Wide(w) => (w, ((w as f64) * vh as f64 / vw as f64).round().clamp(1.0, SaveImage::MAX_SIDE as f64) as u32),
    }
}

/// The bytes free on the drive `path` is on, where the platform says.
#[cfg(windows)]
fn free_disk_bytes(path: &std::path::Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(dir: *const u16, free: *mut u64, total: *mut u64, total_free: *mut u64) -> i32;
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let (mut free, mut total, mut total_free) = (0u64, 0u64, 0u64);
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, &mut total, &mut total_free) };
    (ok != 0).then_some(free)
}

#[cfg(not(windows))]
fn free_disk_bytes(_path: &std::path::Path) -> Option<u64> {
    None
}

/// What a block's settings say as a `saveimage` line: every key whose value is not the default.
/// Points a block at one file: its folder, its name and its format from the path chosen, and
/// written over when saved again, since choosing a name in a save dialog is choosing that file
/// (the dialog asks before replacing one that is there).
///
/// The name has its braces taken out, because the block's name is a pattern and a brace there
/// would be read as a field to fill.
pub(crate) fn aim_block_at(block: &mut SaveImage, path: &std::path::Path) -> Result<(), String> {
    let format = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(fieldform_scene::ImageFormat::from_extension)
        .ok_or_else(|| format!("{} is not a .png or a .jpg", path.display()))?;
    let name: String = path
        .file_stem()
        .map(|s| s.to_string_lossy().replace(['{', '}'], ""))
        .unwrap_or_default();
    if name.trim().is_empty() {
        return Err(format!("{} names no file", path.display()));
    }
    block.folder = path.parent().map(|p| p.display().to_string()).unwrap_or_default();
    block.pattern = name;
    block.format = format;
    block.overwrite = true;
    Ok(())
}

#[cfg(test)]
pub(crate) fn saveimage_line(block: &SaveImage, viewport: (u32, u32)) -> String {
    let mut line = String::from("saveimage");
    if (block.width, block.height) != viewport {
        line.push_str(&format!(" width={} height={}", block.width, block.height));
    }
    if let Some(view) = &block.view {
        line.push_str(&format!(" view={view}"));
    }
    if block.quality == ImageQuality::Displayed {
        line.push_str(" quality=displayed");
    }
    if block.supersample != 1 {
        line.push_str(&format!(" ss={}", block.supersample));
    }
    if block.background != ImageBackground::Environment {
        line.push_str(&format!(" bg={}", image_export::background_word(block.background)));
    }
    if block.format != ImageFormat::Png {
        line.push_str(" format=jpg");
    }
    if block.bits != 8 {
        line.push_str(&format!(" bits={}", block.bits));
    }
    if block.jpeg_quality != 92 {
        line.push_str(&format!(" jpgq={}", block.jpeg_quality));
    }
    if !block.folder.is_empty() {
        line.push_str(&format!(" folder={}", quoted(&block.folder)));
    }
    if block.pattern != SaveImage::DEFAULT_PATTERN {
        line.push_str(&format!(" pattern={}", quoted(&block.pattern)));
    }
    if block.overwrite {
        line.push_str(" overwrite=true");
    }
    match block.capture {
        Capture::Off => {}
        Capture::EverySeconds(s) => line.push_str(&format!(" capture=seconds every={s}")),
        Capture::OnChange => line.push_str(" capture=change"),
        Capture::EveryCommits(n) => line.push_str(&format!(" capture=commits every={n}")),
    }
    if block.max_count != SaveImage::DEFAULT_MAX_COUNT {
        line.push_str(&format!(" max={}", block.max_count));
    }
    if let Some(name) = &block.name {
        line.push_str(&format!(" '{name}'"));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preset_keeps_the_viewports_aspect_and_the_lid_on_a_side() {
        assert_eq!(preset_size(Preset::Viewport(2), (1280, 720)), (2560, 1440));
        assert_eq!(preset_size(Preset::Wide(3840), (1280, 720)), (3840, 2160));
        assert_eq!(preset_size(Preset::Wide(7680), (1000, 1000)), (7680, 7680));
        assert_eq!(preset_size(Preset::Viewport(16), (2000, 1000)), (SaveImage::MAX_SIDE, 16_000));
    }

    #[test]
    fn a_blocks_line_says_only_what_differs_from_the_defaults() {
        let mut b = SaveImage::new(fieldform_scene::PartId::new(), 1280, 720);
        assert_eq!(saveimage_line(&b, (1280, 720)), "saveimage");
        b.width = 5000;
        b.height = 2812;
        b.supersample = 2;
        b.background = ImageBackground::Transparent;
        b.capture = Capture::EverySeconds(30);
        b.name = Some("hero".into());
        assert_eq!(
            saveimage_line(&b, (1280, 720)),
            "saveimage width=5000 height=2812 ss=2 bg=transparent capture=seconds every=30 'hero'"
        );
    }
}
