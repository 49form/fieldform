// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The display's settings: the quality ladder and its frame budgets, the projection, the view mode.

use super::*;

/// What a draft frame may cost, in milliseconds — the traced view's own number, so the startup
/// correction and the draft ladder hold the display to one budget.
pub(crate) const DRAFT_BUDGET_MS: f32 = traced::DRAFT_BUDGET_MS;

/// A frame this long is a hang rather than a slow frame, and says what it spent itself on.
///
/// Well above the draft budget on purpose: a still frame, a bake and a shader rebuild are all
/// allowed to be slow, and a line for each of those would be noise. What this catches is the
/// interface stopping for long enough that a hand notices.
pub(crate) const SLOW_FRAME_MS: f32 = 150.0;

/// How much of the display this machine is asked to pay for: one ladder, every dial.
///
/// **A tier policy, not a shader** (performance-plan-of-action.md §0 A and step 1). The lighting
/// pass is a fixed few tenths of a millisecond whatever the field — `shading_cost` measures it —
/// so there is no cheaper shading to fall back to. What costs is field evaluation per ray, and
/// a still pays it once per pixel per jittered frame with five occlusion samples and a shadow
/// march on top; and what the model costs per reading is how finely its cage is baked and its
/// curves are sampled. Two settings used to share this — a *resolution* for the model's
/// sampling and a *quality* for what the card is asked — and the still's size sat in both.
/// Nobody wants two axes for one question, so they are one ladder (the owner, 2026-09-17), and
/// each rung sets every number:
///
/// | | drag px | still px | still frames | occlusion + shadow | bake cells | profile turn | spine turn |
/// |---|---|---|---|---|---|---|---|
/// | Fast | 480 | 720 | 1 | off | 64 | 4° | 2° |
/// | Balanced | 720 | 1080 | 4 | on | 128 | 2° | 1° |
/// | Full | 720 | 1440 | 8 | on | 192 | 1° | 0.5° |
/// | Enchilada | 720 | the viewport's, to 2560 | 8 | on | 192 | 0.5° | 0.25° |
///
/// `Fast` with one frame is the draft at the still's size, written once: the pre-lighting feel,
/// at the current image quality per pixel. `Enchilada` is everything the viewport can be asked
/// for; nothing chooses it by itself. A rung somebody chose is kept for the next session and
/// starts it ([`starting_quality`]); without one the default is guessed from the adapter at
/// startup and corrected once by the first measured draft — see `App::note_frame_cost`.
/// `Shadows and occlusion when idle` stays the switch for the still's marches; `Fast` only holds
/// it off.
///
/// **Bake cells.** Only the *settled* bake — a drag always uses the 32-cell draft, so this
/// cannot make interaction slower. Measured on the starting cage by
/// `fieldform-gpu/examples/bake_cost.rs`, dev profile, which is the build a person runs:
///
/// | | resolution | bake | memory |
/// |---|---|---|---|
/// | (drag) | 32 | 13.5 ms | <1 MB |
/// | Fast | 64 | 82 ms | 4 MB |
/// | Balanced | 128 | 523 ms | 34 MB |
/// | Full, Enchilada | 192 | 1.62 s | 115 MB |
///
/// 256 is deliberately not offered: it measured 3.74 s and 271 MB, which is 2.3 times the time
/// and 2.4 times the memory of 192 for a third more resolution. Recorded rather than merely
/// omitted, so the next person weighing it has the number instead of the question.
///
/// **Still size.** The part of this setting a body without a cage or a pipe can feel: an
/// analytic solid is neither baked nor tessellated, so the image's own size is the only
/// sharpness on offer to it. Per frame by `traced_cost.rs` when the image was still read back:
/// 6.7 ms at 720, 15.1 at 1080, 24.8 at 1440; 2160 measured 63.4 ms and is offered only as the
/// viewport's own size on the top rung, where a still is traced in bands anyway.
///
/// **Profile turn.** How much a profile's polyline may turn at a joint. Shading is the normal,
/// and a facet holds one normal across its width, so this is what banding *is* — and the walk
/// it used to cost is gone since the profile is baked on the display path (item 33). Measured
/// on a closed cubic ring: 1° is 433 points, 2° 217, 4° 109; 10° bands visibly.
///
/// **Spine turn.** The same for a pipe's spine, which the shader still walks per sample. On a
/// 310 mm cubic (`fieldform-field/examples/spine_angle.rs`): 2° gives 97 segments, 1° 193,
/// 0.5° 385; 6.35° per joint was visible bands across the tube. Only the *settled* walk, as
/// with the bake: while an edit is dragged every spine is walked at 2° whatever the rung
/// (`traced::DRAFT_SPINE_DEGREES`), since a drag frame is at most 720 px and cannot show finer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum DisplayQuality {
    Fast,
    Balanced,
    Full,
    Enchilada,
}

impl DisplayQuality {
    pub(crate) const ALL: [Self; 4] = [Self::Fast, Self::Balanced, Self::Full, Self::Enchilada];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::Full => "Full",
            Self::Enchilada => "Enchilada",
        }
    }

    pub(crate) fn hint(self) -> &'static str {
        match self {
            Self::Fast => {
                "480 px while dragging, a 720 px still drawn once, no shadows or occlusion, a \
                 64-cell bake, 4° profiles and 2° spines — for a slow or old card"
            }
            Self::Balanced => {
                "720 px while dragging, a 1080 px still refined over four frames with shadows \
                 and occlusion, a 128-cell bake, 2° profiles and 1° spines"
            }
            Self::Full => {
                "720 px while dragging, a 1440 px still refined over eight frames with shadows \
                 and occlusion, a 192-cell bake, 1° profiles and 0.5° spines"
            }
            Self::Enchilada => {
                "the whole thing: a still at the viewport's own size refined over eight \
                 frames, a 192-cell bake with every profile baked twice as fine, 0.5° profiles and 0.25° spines — dear on a patterned \
                 pipe, and the still goes up in bands so a slow card survives it"
            }
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|q| q.label().eq_ignore_ascii_case(name.trim()))
    }

    /// Longest edge of a frame drawn while something moves, in pixels.
    pub(crate) fn drag_pixels(self) -> u32 {
        match self {
            Self::Fast => 480,
            _ => 720,
        }
    }

    /// Longest edge of a settled frame, in pixels; zero is the viewport's own size, up to the
    /// traced view's cap.
    pub(crate) fn still_pixels(self) -> u32 {
        match self {
            Self::Fast => 720,
            Self::Balanced => 1080,
            Self::Full => 1440,
            Self::Enchilada => 0,
        }
    }

    pub(crate) fn still_frames(self) -> u32 {
        match self {
            Self::Fast => 1,
            Self::Balanced => 4,
            Self::Full | Self::Enchilada => 8,
        }
    }

    /// Whether a still's rays finish on a profile's outline rather than its bake
    /// (`Camera::exact_finish`, backlog item 115): every sharp edge drawn where it is, 0.01 px at
    /// every size. Every rung's, since the read became local to the bake's block: measured at
    /// the 720 still by `traced_cost`, 6.4 to 6.5 ms on an extruded profile, 5.8 to 6.2 on a
    /// revolve, 119 to 160 on a hundred patterned and 565 to 761 on five hundred, and a gyroid
    /// untouched; the whole outline walked had been 51 ms and 4.4 s. A rung that cannot pay a
    /// third more on a pattern's still is pulled back to its bake here. The draft never asks.
    pub(crate) fn exact_finish(self) -> bool {
        true
    }

    /// Whether a settled frame may measure occlusion and the shadow at all.
    pub(crate) fn occlusion(self) -> bool {
        !matches!(self, Self::Fast)
    }

    /// Samples across the cage's settled bake.
    pub(crate) fn cells(self) -> usize {
        match self {
            Self::Fast => 64,
            Self::Balanced => 128,
            Self::Full | Self::Enchilada => 192,
        }
    }

    /// How much a profile's polyline may turn at a joint, in degrees.
    /// How much finer than the cage bake's rule a settled profile or wall is baked: the margin a
    /// baked outline is drawn fat by shrinks with it, and a knife edge shows that margin as a
    /// chamfer `margin / tan(half the edge's angle)` wide (backlog item 115).
    ///
    /// **Two on Enchilada, and two is the ceiling.** `knife_scale` measures a 39-degree knife
    /// fitted to a 720 px frame at 0.54 px of error on Full and 0.27 at twice as fine, the same
    /// at 100 mm, 1 m and 50 m, for 3.5 to 12.8 ms of bake per profile at a settled rebuild
    /// and no change to the frame; asked for four, the bake's block ceiling hands back two at
    /// four times the cost. The draft keeps its coarse bake, so nothing moving pays.
    pub(crate) fn profile_bake_fineness(self) -> f64 {
        match self {
            Self::Enchilada => 2.0,
            _ => 1.0,
        }
    }

    pub(crate) fn profile_degrees(self) -> f64 {
        match self {
            Self::Fast => 4.0,
            Self::Balanced => 2.0,
            Self::Full => 1.0,
            Self::Enchilada => 0.5,
        }
    }

    /// How much a pipe's spine may turn at a joint, in degrees.
    pub(crate) fn spine_degrees(self) -> f64 {
        match self {
            Self::Fast => 2.0,
            Self::Balanced => 1.0,
            Self::Full => 0.5,
            Self::Enchilada => 0.25,
        }
    }

    /// Every number of this rung a settled rebuild reads, which is what the traced view
    /// compares to decide whether a change of rung needs one.
    pub(crate) fn fineness(self) -> traced::Fineness {
        traced::Fineness {
            resolution: self.cells(),
            spine_degrees: self.spine_degrees(),
            profile_degrees: self.profile_degrees(),
        }
    }
}

/// Where the display starts, and whether that is somebody's choice.
///
/// A rung chosen in an earlier session wins, on any adapter. Without one it is the guess:
/// `Full` on a discrete card, `Balanced` on anything else. The flag is what
/// `App::note_frame_cost` reads before correcting the start by the first measured draft, so a
/// saved rung, being a choice, is never demoted: a slow first draft under a rung somebody picked
/// is a heavy document, not a reason to second-guess them.
pub(crate) fn starting_quality(
    saved: Option<DisplayQuality>,
    discrete: bool,
) -> (DisplayQuality, bool) {
    match saved {
        Some(quality) => (quality, true),
        None if discrete => (DisplayQuality::Full, false),
        None => (DisplayQuality::Balanced, false),
    }
}

/// The settings file's name for the display quality.
const QUALITY_KEY: &str = "display_quality";

/// Where this person's settings are kept between sessions, if the platform says where.
///
/// **A file of the application's own, not the document and not eframe's store.** The rung is a
/// fact about the machine rather than the model, so a document carrying it would hand the next
/// person's card the last one's choice. eframe's store needs its `persistence` feature, which
/// brings in crates this build does not otherwise carry and starts remembering the window's
/// size and egui's own state as well, which nobody asked for. One JSON file, through the
/// `serde_json` the binary already links, holds the one thing that was.
///
/// **Not the temp directory, where the logs and the autosave are.** Those are one session's
/// record and a crash's recovery, and a temp directory gets swept; a setting that quietly reverts
/// whenever the sweep comes round is not a setting. The per-person application data directory is
/// where eframe's own store would have put it.
pub(crate) fn settings_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("APPDATA")
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME"))
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::Path::new(&home).join(".config"))
        })?;
    Some(base.join("fieldform").join("settings.json"))
}

/// The rung a settings file saved, if it saved one this build offers.
///
/// Anything else — an empty or unparseable file, a rung by a name no longer offered — is `None`,
/// and the display starts on the adapter's guess: a setting is never a reason not to start.
pub(crate) fn saved_quality(settings: &str) -> Option<DisplayQuality> {
    let read: serde_json::Value = serde_json::from_str(settings).ok()?;
    DisplayQuality::from_name(read.get(QUALITY_KEY)?.as_str()?)
}

/// `settings` with the display quality set to `quality` and every other entry kept, so a setting
/// kept beside it later is not erased each time this one is chosen. A file that is not a JSON
/// object is replaced: what cannot be read cannot be kept, and is no reason to refuse a choice.
pub(crate) fn with_saved_quality(settings: Option<&str>, quality: DisplayQuality) -> String {
    let mut entries = match settings.and_then(|text| serde_json::from_str(text).ok()) {
        Some(serde_json::Value::Object(entries)) => entries,
        _ => serde_json::Map::new(),
    };
    entries.insert(QUALITY_KEY.to_string(), quality.label().into());
    serde_json::to_string_pretty(&serde_json::Value::Object(entries))
        .expect("a map of strings to JSON values always serializes")
}

/// The rung an earlier session saved, read from [`settings_path`].
pub(crate) fn load_saved_quality() -> Option<DisplayQuality> {
    saved_quality(&std::fs::read_to_string(settings_path()?).ok()?)
}

/// Keeps `quality` as the rung the next session starts on.
pub(crate) fn save_quality(quality: DisplayQuality) -> Result<(), String> {
    let path = settings_path().ok_or("this platform names no per-person settings directory")?;
    let before = std::fs::read_to_string(&path).ok();
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, with_saved_quality(before.as_deref(), quality))
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// How the viewport projects.
///
/// Presentation state, like the view mode beside it: it changes what you see, never what the
/// document is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Projection {
    /// What an eye does. Converging edges, so depth reads immediately.
    Perspective,
    /// What a drawing does. Parallel edges stay parallel and equal lengths stay equal, which
    /// is what makes a face-on view something you can measure off.
    Orthographic,
}

impl Projection {
    pub(crate) fn toggled(self) -> Self {
        match self {
            Projection::Perspective => Projection::Orthographic,
            Projection::Orthographic => Projection::Perspective,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Projection::Perspective => "Perspective",
            Projection::Orthographic => "Orthographic",
        }
    }
}

/// Which representation the viewport is showing. Tab cycles through all three.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum ViewMode {
    /// The control cage alone — flat-shaded, so its facets read as facets.
    Polygon,
    /// The composed field — cage, curves and booleans together — evaluated per pixel on the
    /// GPU rather than meshed.
    Field,
    /// The composed field with the control cage translucent over it, so cage edits can be
    /// made while watching the surface they drive.
    Overlay,
}

impl ViewMode {
    pub(crate) fn next(self) -> Self {
        match self {
            ViewMode::Polygon => ViewMode::Field,
            ViewMode::Field => ViewMode::Overlay,
            ViewMode::Overlay => ViewMode::Polygon,
        }
    }

    /// The mode's own name, for a menu entry that sets it rather than narrating it.
    pub(crate) fn menu_label(self) -> &'static str {
        match self {
            ViewMode::Polygon => "Cage",
            ViewMode::Field => "Field",
            ViewMode::Overlay => "Field + Cage",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            ViewMode::Polygon => "Mode: Cage (Tab to cycle)",
            ViewMode::Field => "Mode: Field (Tab to cycle)",
            ViewMode::Overlay => "Mode: Field + Cage (Tab to cycle)",
        }
    }

    /// Whether cage elements are selectable/editable and their highlights drawn.
    pub(crate) fn cage_is_editable(self) -> bool {
        matches!(self, ViewMode::Polygon | ViewMode::Overlay)
    }
}
