// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! fieldform: cage-driven, field-native modeler.
//!
//! M1 in progress: the control cage (currently the default cube) renders in a POI-anchored
//! orbit-camera viewport (a point-of-interest camera):
//! LMB-drag tumbles about whatever point is under the
//! cursor, not a fixed target. Editing tools and OBJ import land next; field mode after that.
//! See the project plan for the full milestone sequence.

mod blocks;
mod gizmo;
mod cage_mesh;
mod render;
mod traced;
mod outputs;
mod menu_hover;
mod preview;
mod split_axis;
mod console_keys;
mod commands;
mod block_edit;
mod echo;
mod palette;
mod keybindings;
mod marking;
mod settings;
mod expect;
mod names;
mod lattice_readout;
mod camera;
mod view_cube;
mod overlay_draw;
mod script;
mod console;
mod selection;
mod grid;
mod curve_edits;
mod operands;
mod pick;
mod display_quality;
mod app;

use split_axis::AxisKind;

use cage_mesh::CageMesh;
use eframe::{egui, egui_wgpu};
use fieldform_cage::{Cage, FaceId, QuadMesh, VertexId};
use fieldform_ops::{
    Args, CommandStack, Document, Keymap, Modifiers, OpRegistry, SetControlPoints,
    SetControlWeights, SetCurve, SetParameter, SetPartGraph, SetPartMaterial, SetRelation,
    SetShown, SetSurfacePositions, SetVertexPositions, SetVisible, Shortcut,
    OwnershipSnapshot, Value,
};
use fieldform_field::{Extent, GraphField, ScalarField};
// nalgebra also has a `Unit`, and it means something completely different (a normalised
// vector). Aliased so the two cannot be confused at a glance.
use fieldform_scene::Unit as LengthUnit;
use fieldform_scene::{Fillet, 
    Axis, PatternKind, RampShape,
    Banding, CellMap, LatticeCell, LatticePhase,
    BooleanOp, CageId, Continuity, CurveId, Ends, Graph, Material, Node, NodeId, ParamId,
    ParamRef, PartId, Turn,
    BlockKey, EyeFlag, Ownership, OwnershipChange,
    draws_as_overlay_in, eye_flag_by, eye_flag_in, is_visible_among, is_visible_in,
    shown_ingredients_in,
};
#[cfg(test)]
use fieldform_scene::drawn_graph;
use nalgebra::{Matrix4, Point3, Rotation3, Similarity3, Unit, UnitQuaternion, Vector3, Vector4};
use render::{CageDrawCallback, CageRenderResources};
use traced::TracedView;
use camera::OrbitCamera;
use view_cube::{show_view_cube, view_cube_controls_rect, view_cube_faces, CubeRegion, SNAP_SECONDS};
#[cfg(test)]
use view_cube::{
    view_cube_cells, view_cube_grid_rect, view_cube_home_rect, view_cube_projection_rect,
    view_cube_rect, VIEWCUBE_MIN_VIEWPORT,
};
use overlay_draw::{
    draw_axis_seed, draw_brep_edges, draw_curves, draw_dashed_polyline, draw_highlight,
    draw_poi_marker, draw_surface_wire, CONTROL_POLYGON_COLOR, CURVE_COLOR, CURVE_SELECTED_COLOR,
    GUIDE_COLOR, GUIDE_FILL, HOVER_COLOR, HOVER_FILL, PLANE_PREVIEW_FILL, SELECTION_COLOR,
    SELECTION_FILL,
};
use script::{
    ExpectRecord, PendingSnap, SaveAs, Script, ScriptTally, SCRIPT_EXIT_CODE, SNAP_WAIT_MAX,
};
use console::{
    console_arguments, console_commands, console_words, ends_from_word, parse_value,
    pattern_options_from, side_from_word, split_degree_suffix, wall_thickness_asked, wall_wanted,
    EXTRUDE_ALIASES, REVOLVE_ALIASES,
};
use selection::{ControlMesh, Picked, Selection};
use grid::{
    grid_label_at, grid_pitch, snap_to_grid, PlaneGrid, GRID_AXIS, GRID_LABEL, GRID_MAJOR,
    GRID_MINOR,
};
#[cfg(test)]
use grid::{GRID_FINEST_MM, GRID_LABEL_INSET, GRID_MIN_CELL_PX};
use curve_edits::{
    bezier_between, copied_body, curve_shapes_a_body_in, delete_loop_member_commands,
    delete_row_commands, drag_targets, group_by_curve, join_loop_commands, move_point_node,
    moved_to_start_at, newly_closed_rows, paste_body, point_literals, points_in_box,
    readers_in_other_parts, realign_commands, rederived_commands, rows_not_in, set_targets,
    settle_curves_on_points, single_span, single_span_bezier, split_selection, toggle_in,
    ungroup_commands, CURVE_DEGREES,
};
#[cfg(test)]
use curve_edits::delete_curves_commands;
use operands::{
    operand_choices_in, operand_refusal, outermost_in, pair_candidates_in, starting_pair_among,
    twist_would_take,
};
#[cfg(test)]
use operands::starting_pair_in;
use blocks::list::{
    branch_starts, curve_v_elsewhere, drop_gutter, ends_on_point, DraggedBlock,
    filtered_choices, folded_curve_row_in, grip_mark, joint_choices, joint_points_in,
    joint_relation_between, make_variable, nest_rows, points_hidden_with_rows,
    names_in_a_row, profile_v_curve_in, put_back_inside, put_number_back, rename_block,
    variable_item, Block, BlockRow, Clipboard, ProfileRows, Row, RowKey,
};
use pick::pick;
#[cfg(test)]
use pick::occluded;
use display_quality::{
    starting_quality, DisplayQuality, Projection, ViewMode, DRAFT_BUDGET_MS, SLOW_FRAME_MS,
};

const FOVY: f32 = 45.0 * std::f32::consts::PI / 180.0;

/// How much slack the clip planes leave around the scene, as a fraction of its radius.
///
/// The planes are fitted to the *scene* — see [`OrbitCamera::clip_planes`] — rather than to a
/// fixed distance or a fraction of one, which is the third and last attempt at this.
///
/// They were constants first (0.05 and 100, in millimetres, because that is what world units
/// are here). That worked while the only model was a 1 mm cube; a metre-scale part put the
/// whole model past the far plane and it simply stopped being drawn.
///
/// Then they were fractions of the distance from the eye to the *target*. Scale-free, so no
/// size of model could fall out of range — but the target is the pivot, and the pivot is
/// wherever you last clicked. Pivot on one corner of a part and zoom out, and the near plane
/// grows with the distance to that corner until it is in front of the geometry, which is the
/// clipping this replaces.
///
/// Fitting to the scene has neither failure: the planes bracket what is actually there,
/// whatever its size, wherever the camera and the pivot happen to be.
const CLIP_MARGIN: f32 = 0.5;

/// One line of the console: what was said, whether it went well, and how many times running.
struct ConsoleLine {
    ok: bool,
    text: String,
    /// How many times in a row this exact line was reported. One, unless something repeated.
    said: u32,
}

/// How many lines the console keeps. Old ones fall off the top.
const CONSOLE_LINES: usize = 4000;

/// How large the console's file may get before it is rewritten from what is still in memory.
const CONSOLE_FILE_BYTES: u64 = 1 << 20;

/// Starts this session's console log, in the temp directory beside the crash log.
///
/// Truncated rather than appended, so the file is one session and reading it cannot turn up a
/// message from yesterday that looks like it happened just now. `None` if it cannot be opened,
/// which is not worth complaining about — the console on screen is unaffected.
fn open_console_log() -> Option<std::path::PathBuf> {
    let path = std::env::temp_dir().join("fieldform-console.log");
    std::fs::write(&path, "").ok()?;
    Some(path)
}

/// A local to take the address of, which is all `install_crash_log` wants from it.
fn path_base() -> usize {
    0
}

/// Flattens anything that would break the one-entry-per-line shape of the log file.
///
/// Console text is written by this application rather than by a document, so this is tidiness
/// rather than a defence — but a file name comes from the filesystem, and an error message can
/// carry whatever a library put in it, so neither is guaranteed to be one line of printable text.
fn one_line(text: &str) -> String {
    text.chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}

/// Roughly where the main thread's stack starts, recorded at startup.
///
/// Zero until `install_crash_log` sets it, which is the first thing `main` does.
static STACK_BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How far the stack may grow before a walk is treated as a runaway rather than as deep work.
///
/// Half a megabyte. Nothing here recurses honestly beyond a few hundred frames, and the main
/// thread has megabytes — so this trips long before the guard page does, which is the whole
/// point: past the guard page there is no stack left to write a report with.
const STACK_RUNAWAY_BYTES: usize = 512 * 1024;

/// Aborts with a backtrace when the stack has run away, rather than letting it overflow.
///
/// **Because a stack overflow leaves nothing behind.** It aborts without unwinding, so the panic
/// hook never runs and `fieldform-crash.log` stays empty — which is exactly what two crash
/// reports looked like: the window vanished, the log was empty, and the only evidence was the
/// one line Windows prints to a stderr nobody sees. This turns the next one into a named
/// backtrace in the file the hook already writes.
///
/// Called from the cheap functions nearly every walk goes through, so whichever recursion is
/// running away has this on its stack and names itself.
fn guard_stack(what: &str) {
    let here = &what as *const _ as usize;
    let base = STACK_BASE.load(std::sync::atomic::Ordering::Relaxed);
    // Stacks grow down on every platform this runs on, and a base of zero means startup has not
    // recorded one yet -- in both cases there is nothing to compare against.
    if base == 0 || here >= base || base - here < STACK_RUNAWAY_BYTES {
        return;
    }
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::temp_dir().join("fieldform-crash.log"))
    {
        let _ = writeln!(file, "\n=== fieldform runaway walk ===");
        let _ = writeln!(file, "in {what}, {} bytes of stack deep", base - here);
        let _ = writeln!(file, "{}", std::backtrace::Backtrace::force_capture());
        let _ = file.flush();
    }
    std::process::abort();
}

/// How often the document is written to [`autosave_path`] while it has unsaved changes.
///
/// A minute is the most work a sudden death may cost. **Measured before it was chosen:** the
/// whole document of the heaviest scene to hand — a radial pattern of 48 gyroid-filled shells —
/// writes through the archive in **11 ms**, which is why this runs on the interface thread and
/// not a worker. A document is a graph, so its size follows the number of blocks rather than
/// how heavy they are to draw; the autosave logs its own cost every time precisely so the day
/// that stops being true is visible.
const AUTOSAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(60);

/// Whether the autosave is owed, given the document's revision, the one last written, whether
/// there is anything unsaved, and how long since the last write.
///
/// All four conditions, and each is there for its own reason: nothing unsaved means the
/// document's own file is already the record; an unchanged revision means the autosave already
/// *is* this document, which matters because `is_dirty` stays true after an autosave (an
/// autosave is not a save); and the interval is what keeps a heavy document's write off every
/// frame.
fn autosave_due(revision: u64, autosaved: u64, dirty: bool, since: std::time::Duration) -> bool {
    dirty && revision != autosaved && since >= AUTOSAVE_EVERY
}

/// How long ago, in the roundest words that are still true: a recovery offer wants "about an
/// hour ago", not a duration to parse.
fn how_long_ago(d: std::time::Duration) -> String {
    let minutes = d.as_secs() / 60;
    match minutes {
        0 => "less than a minute ago".to_string(),
        1 => "a minute ago".to_string(),
        2..=59 => format!("{minutes} minutes ago"),
        60..=119 => "about an hour ago".to_string(),
        _ => {
            let hours = minutes / 60;
            if hours < 24 {
                format!("about {hours} hours ago")
            } else {
                let days = hours / 24;
                format!("{days} day{} ago", if days == 1 { "" } else { "s" })
            }
        }
    }
}

/// Where **this process's** autosave is written: beside the console and crash logs, not beside
/// the document, and named by the process so it is never another session's file.
///
/// **One name for every session was how a crashed session's work was deleted, 2026-10-15.** The
/// owner's test instance was killed, leaving its autosave; the next instance — a scripted run —
/// exited cleanly and removed "the" autosave, which was not its own. One shared name also let a
/// second live instance overwrite the first one's. So each process writes and removes only the file
/// named by its own id, and [`other_autosaves`] is how the others are found: offered, never removed.
fn autosave_path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "fieldform-autosave-{}.{DOCUMENT_EXTENSION}",
        std::process::id()
    ))
}

/// Which files in the temp directory are autosaves of sessions other than `own`, newest first.
///
/// Pure over a listing so the rule can be held to a property. Matches this build's per-process
/// names and the single shared name earlier builds wrote, so the crash of an older build is still
/// offered. **Never a reason to delete anything:** a file listed here may belong to a session that is
/// still running, and only that session knows when it is finished with it.
fn other_autosaves(
    listing: &[(String, std::time::SystemTime)],
    own: &str,
) -> Vec<(String, std::time::SystemTime)> {
    let ext = format!(".{DOCUMENT_EXTENSION}");
    let mut out: Vec<(String, std::time::SystemTime)> = listing
        .iter()
        .filter(|(name, _)| {
            name != own
                && name.starts_with("fieldform-autosave")
                && name.ends_with(&ext)
                && {
                    // Exactly the shared name, or the shared name with a process id: nothing a
                    // person happened to call "fieldform-autosave backup" or similar.
                    let middle = &name["fieldform-autosave".len()..name.len() - ext.len()];
                    middle.is_empty()
                        || (middle.len() > 1
                            && middle.starts_with('-')
                            && middle[1..].chars().all(|c| c.is_ascii_digit()))
                }
        })
        .cloned()
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

/// Where the document is kept when the GPU is lost: `<name>.recovered.fform` beside the file it
/// was opened from or last saved to, or beside the logs for one never saved (loop spec S3).
///
/// **Beside the document, unlike the autosave.** The autosave is one file for whatever was being
/// edited, overwritten every minute; this is the newest state of one particular document, and the
/// place to look for it is next to that document. **Never the document's own file**, for the
/// autosave's reason: a dying GPU must not overwrite the last state somebody deliberately saved.
/// In the current extension whatever the file was opened as, since that is what this build writes.
fn recovered_path(file: Option<&std::path::Path>) -> std::path::PathBuf {
    match file {
        Some(path) => {
            let name = path
                .file_stem()
                .map_or_else(|| "untitled".into(), |stem| stem.to_string_lossy());
            path.with_file_name(format!("{name}.recovered.{DOCUMENT_EXTENSION}"))
        }
        // Prefixed like the logs and the autosave, so the temp directory lists them together.
        None => std::env::temp_dir()
            .join(format!("fieldform-untitled.recovered.{DOCUMENT_EXTENSION}")),
    }
}

/// Writes `doc` to [`recovered_path`] and answers where it went.
///
/// **Through the archive the document was opened from**, for `save_document`'s reason: a fresh
/// archive drops the `ext/` and `sources/` entries the file carried, and a kept document somebody
/// later saves over their own would lose them. A clone of it, so the application's copy still
/// matches the file on disk.
///
/// **The temp directory, under the same name, when the document's folder refuses.** A share that
/// dropped or a drive pulled out is no reason to lose the work, and the logs' folder is the one
/// place this application always writes.
fn write_recovered(
    doc: &Document,
    file: Option<&(std::path::PathBuf, fieldform_io::Archive)>,
) -> Result<std::path::PathBuf, fieldform_io::FormatError> {
    let archive = match file {
        Some((_, opened)) => {
            let mut archive = opened.clone();
            fieldform_io::update_archive(&mut archive, doc)?;
            archive
        }
        None => fieldform_io::to_archive(doc)?,
    };
    let beside = recovered_path(file.map(|(path, _)| path.as_path()));
    match archive.write(&beside) {
        Ok(()) => Ok(beside),
        Err(refused) => {
            let instead = std::env::temp_dir().join(beside.file_name().unwrap_or_default());
            if instead == beside {
                return Err(refused);
            }
            archive.write(&instead).map(|()| instead)
        }
    }
}

/// Appends a headed note to `fieldform-crash.log`, the file a person is sent to when the window
/// has gone. Best effort: a note that cannot be written must not become a second failure.
fn note_in_crash_log(heading: &str, said: &str) {
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::temp_dir().join("fieldform-crash.log"))
    {
        let _ = writeln!(file, "\n=== {heading} ===\n{said}");
    }
}

/// Writes any panic, with a backtrace, to `fieldform-crash.log` in the temp directory.
///
/// A windowed application's stderr goes nowhere the user can see — launched from Explorer
/// there is no console at all — so without this a crash is just the window vanishing, and the
/// one piece of information needed to fix it is the piece that gets lost. The default hook
/// still runs afterwards, so a terminal launch keeps printing as before.
fn install_crash_log() -> std::path::PathBuf {
    // Whatever the stack pointer is here, which is near enough the top of it: this runs from
    // `main` before anything else.
    let here = &path_base() as *const _ as usize;
    STACK_BASE.store(here, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join("fieldform-crash.log");
    let target = path.clone();
    let previous = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target)
        {
            let _ = writeln!(file, "\n=== fieldform panic ===");
            let _ = writeln!(file, "{info}");
            // force_capture rather than capture: RUST_BACKTRACE is almost never set for a
            // double-clicked application, and a backtrace-less crash report is close to
            // useless.
            let _ = writeln!(file, "{}", std::backtrace::Backtrace::force_capture());
        }
        previous(info);
        // Then die, rather than unwind. A panic raised inside a Windows message callback -- which
        // is where every frame of this application runs -- does not unwind back out of it: the
        // window stays on screen, stops answering, and holds the GPU while it does. The log
        // above is what a crash is for; a hung window on top of it is not. Seen on 2026-09-06,
        // when an egui allocation failed and the machine had to be restarted to be rid of it.
        std::process::abort();
    }));

    path
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod app_tests;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let crash_log = install_crash_log();
    log::info!("crashes will be logged to {}", crash_log.display());

    let mut options = eframe::NativeOptions::default();
    // Maximised, because the still is drawn at the viewport's size and a script's `snap` writes
    // it at that size: the default window gave a 424-pixel still, which is no picture of a seam.
    options.viewport = egui::ViewportBuilder::default()
        .with_maximized(true)
        .with_icon(window_icon());
    options.wgpu_options = wgpu_configuration();
    let ran = eframe::run_native(
        "fieldform",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    );
    // `--run … --exit`: the count of what did not pass is the process's answer, so a caller can
    // test it without reading the log.
    let code = SCRIPT_EXIT_CODE.load(std::sync::atomic::Ordering::SeqCst);
    if code >= 0 {
        std::process::exit(code);
    }
    ran
}

/// The window's icon: the emblem, white on black, for the title bar and the taskbar.
///
/// Embedded, like the splash, because the binary is what gets run and it is often run from
/// somewhere other than the source tree. Without one eframe puts up its own white `e` on black,
/// which on a taskbar reads as some other program's window. `window-icon.png` is
/// `window-icon.svg` rasterised at 256 pixels; the SVG is the source, so the PNG can be remade
/// at another size rather than redrawn.
///
/// An icon that will not decode is not worth stopping for: an empty `IconData` asks for the
/// operating system's own, which is a worse icon and a working window.
fn window_icon() -> egui::IconData {
    const BYTES: &[u8] = include_bytes!("../assets/window-icon.png");
    match image::load_from_memory(BYTES) {
        Ok(decoded) => {
            let rgba = decoded.to_rgba8();
            egui::IconData { width: rgba.width(), height: rgba.height(), rgba: rgba.into_raw() }
        }
        Err(e) => {
            log::warn!("the embedded window icon does not decode ({e}); using the system's");
            egui::IconData::default()
        }
    }
}

/// The device fieldform asks for, shared by the window and the tests' headless `App`.
///
/// It asks for what the field shader binds -- `fieldform_gpu::required_limits`, the same function
/// an example's device asks. eframe's own request allows eight storage buffers per stage and
/// `fields` binds more than that; left at the default, the first frame of any document aborted
/// the process.
fn wgpu_configuration() -> eframe::egui_wgpu::WgpuConfiguration {
    let mut config = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut config.wgpu_setup {
        setup.device_descriptor = std::sync::Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("fieldform device"),
            required_limits: fieldform_gpu::required_limits(adapter),
            // GPU timestamps, where the adapter has them, so the `perf` console verb reports
            // the card's own time for each frame rather than the wall clock's guess. Asked for
            // by intersection: a feature the adapter lacks must not refuse the device.
            required_features: fieldform_gpu::timing_features(adapter),
            ..Default::default()
        });
    }
    config
}

/// Which translator a file needs, and so which options it has.
///
/// Read from the extension rather than asked for. The file already says what it is, and a
/// format picker that can disagree with the file is a second description of one fact — the
/// shape of bug this codebase keeps meeting. What the dialog asks for is the part the file
/// genuinely does *not* carry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ImportFormat {
    Step,
    Obj,
    Iges,
}

impl ImportFormat {
    fn of(path: &std::path::Path) -> Option<Self> {
        match path.extension()?.to_string_lossy().to_lowercase().as_str() {
            "stp" | "step" => Some(Self::Step),
            "obj" => Some(Self::Obj),
            "igs" | "iges" => Some(Self::Iges),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Step => "STEP",
            Self::Obj => "Wavefront OBJ",
            Self::Iges => "IGES",
        }
    }
}

/// A file chosen for import, waiting on its options.
struct ImportPrompt {
    path: std::path::PathBuf,
    format: ImportFormat,
}

/// The importer's file name, without the platform's executable suffix.
///
/// A separate program rather than this one re-invoked with a flag, and the reason is size as
/// much as safety. OpenCASCADE is statically linked and measures 29.4 MB; while the worker was
/// this binary under another name, every copy of the application carried a CAD kernel most
/// sessions never open. Apart, `fieldform` is ~16 MB and the importer is the component that can
/// be shipped separately, fetched on demand, or left out for anyone who never opens a STEP
/// file.
///
/// The safety half stands unchanged: OCCT reports some failures by *aborting*, which no
/// `catch_unwind` can catch, so the import has to be something allowed to die on its own.
const IMPORT_WORKER: &str = "fieldform-import";

/// Where the importer is, if it is anywhere.
///
/// Beside this executable, which is where a build puts it and where an installer would. Not
/// searched for on `PATH` deliberately: picking up a `fieldform-import` from elsewhere on the
/// system would run some other build's importer against this build's `mesh_wire` format.
fn import_worker_path() -> Option<std::path::PathBuf> {
    let beside = std::env::current_exe()
        .ok()?
        .parent()?
        .join(format!("{IMPORT_WORKER}{}", std::env::consts::EXE_SUFFIX));
    beside.is_file().then_some(beside)
}

/// An import running in another process, and what the application needs to finish it.
struct PendingImport {
    name: String,
    /// The file being read, kept so the import it produces can name and re-read it.
    path: std::path::PathBuf,
    /// The import this is a new revision of, when it is one.
    ///
    /// A re-import replaces what an earlier read produced *in place* rather than adding a
    /// second copy beside it, which is the whole point of an import being a block: point it at
    /// a newer file and everything built on it re-evaluates.
    replacing: Option<fieldform_scene::id::SourceId>,
    /// Where the worker will write the mesh. Removed when the import finishes, however it
    /// finishes — a cancelled or crashed import must not leave a file behind.
    output: std::path::PathBuf,
    child: std::process::Child,
    started: std::time::Instant,
}

/// Document geometry is `f64` (D2); the camera, picking and highlight code work in `f32`,
/// where screen precision is a few thousand pixels and `f64` would buy nothing. This is the
/// single read-side cast, so the narrowing is explicit at every use rather than implicit.
fn vertex_pos(cage: &QuadMesh, placement: &Similarity3<f64>, v: VertexId) -> Point3<f32> {
    (placement * cage.vertex_position(v)).cast::<f32>()
}

/// A number as a block shows it: `23` rather than `23.0000`, and at most four decimals,
/// which is a tenth of a micron on a millimetre and more than any field here resolves.
fn readout(v: f64) -> String {
    let text = format!("{v:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    // A value that rounds to nothing is nothing, not "-0".
    if text.is_empty() || text == "-" || text == "-0" { "0".to_string() } else { text.to_string() }
}

/// Rounds a length to a tidy value of similar magnitude — 52.6 becomes 53, 0.0417 becomes
/// 0.042. Used for defaults offered to the user, where a suspiciously precise number implies a
/// precision that was never meant.
fn tidy_mm(v: f64) -> f64 {
    if !v.is_finite() || v <= 0.0 {
        return 1.0;
    }
    let magnitude = 10f64.powf(v.log10().floor());
    (v / magnitude * 10.0).round() / 10.0 * magnitude
}

/// Bounding-box center and radius (half the diagonal) of a cage's vertices, for `/`'s
/// re-center-on-object behavior.
fn cage_bounds(cage: &Cage) -> (Point3<f32>, f32) {
    let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in cage.vertices() {
        let p = v.position.cast::<f32>();
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }
    let center = Point3::new(
        (min.x + max.x) * 0.5,
        (min.y + max.y) * 0.5,
        (min.z + max.z) * 0.5,
    );
    let radius = ((max - min).norm() * 0.5).max(0.1);
    (center, radius)
}

// ---------------------------------------------------------------------------
// Selection and move: click a vertex/edge/face, drag to translate it.
// ---------------------------------------------------------------------------

/// How a boolean meets: a hard edge, or a fillet of some continuity.
///
/// `None` is G0 — position continuity and nothing more, which is a plain boolean rather than a
/// blend of radius zero. That is the default because it is what a boolean *is*; a fillet is
/// something you ask for.
type BlendMode = Option<Continuity>;

/// The continuity choices the tool offers, in the order they are shown.
const BLEND_MODES: [(BlendMode, &str); 4] = [
    (None, "G0"),
    (Some(Continuity::G1), "G1"),
    (Some(Continuity::G2), "G2"),
    (Some(Continuity::Chamfer), "Chamfer"),
];

/// What the row holding a fillet's number is called.
///
/// **It is a radius only at G1, and only at a right angle** — the zero set of the form leaves each
/// face where the *other* is `k` away, whatever the exponent and the angle, which is a reach. G1
/// keeps the word people expect and is a circle of that radius between square faces; G2 says what
/// the number is, because its tightest bend is not it (0.56 × at a right angle). Ruled 2026-10-02,
/// on a measurement (`Continuity`'s doc): the number stays the same one when the shape changes, so
/// switching never moves where the rounding starts.
fn fillet_label(continuity: Continuity) -> &'static str {
    match continuity {
        Continuity::G1 => "Fillet:",
        Continuity::G2 => "Reach:",
        // The leg, which is the reach under its own name: how far along each face the cut
        // runs. A chamfer's size is said that way on every drawing.
        Continuity::Chamfer => "Chamfer:",
    }
}

/// The read-only line under a fillet's number saying what it does to the bend, or nothing at G1,
/// where the tightest bend *is* the number between square faces.
///
/// Derived and never stored — the number is the one thing the node holds — so the line cannot
/// disagree with it. "At a right angle" is in the text because it is a property of the corner as
/// well as of the shape: at 60° the same G2 reach bends at 0.26 ×, at 120° at 1.37 ×.
fn fillet_bend_note(continuity: Continuity, reach_mm: f64) -> Option<String> {
    if continuity == Continuity::Chamfer {
        return Some(format!(
            "A flat bevel, cut back {reach_mm:.2} mm along each face from the other; at a right \
             angle a 45\u{b0} chamfer of that leg."
        ));
    }
    (continuity != Continuity::G1).then(|| {
        let ratio = continuity.apex_radius_ratio();
        format!(
            "Leaves each face {reach_mm:.2} mm from the other. Tightest bend \u{2248} {:.2} mm \
             at a right angle ({ratio:.2} \u{d7} the reach).",
            ratio * reach_mm
        )
    })
}

/// What the G1 and G2 buttons say when hovered, on a blend and on a pattern's fold alike.
fn fillet_shape_hover(continuity: Continuity) -> &'static str {
    match continuity {
        Continuity::G1 => {
            "A circular fillet: tangent to both faces, and between square faces a circle of the \
             number below."
        }
        Continuity::G2 => {
            "A curvature-continuous fillet: it leaves each face the same reach from the other \
             as G1 does, but eases in with curvature rising from nought rather than stepping to \
             1/k, so the bend in the middle is tighter."
        }
        Continuity::Chamfer => {
            "A flat bevel: each face cut back the number below from the other, with a plane \
             between the two cuts -- the same footprint as G1 and G2, so switching moves nothing \
             but the shape between them. Its value reads short across the bevel, so the march \
             takes a few more steps there; the surface is exact."
        }
    }
}

/// How big anything added from the Create menu starts.
///
/// A sphere gets this as its diameter and a cube as its side, so a new body is always about the
/// same size as any other new body, whatever else is on screen.
const DEFAULT_BODY_MM: f64 = 100.0;

/// How long after a graph edit another one still counts as the same gesture.
///
/// Long enough to bridge the gap between two frames of a drag on a loaded machine, short enough
/// that two deliberate clicks are never merged. It is a *supporting* condition rather than the
/// test itself — see `set_part_graph`, where the structural comparison does the real work.
const GESTURE_WINDOW: std::time::Duration = std::time::Duration::from_millis(400);

/// What the part looked like when the graph edit in progress began.
struct Gesture {
    part: PartId,
    before: Graph,
    /// Who owned what when it began, which is what goes with `before` (`block-ownership-spec.md`
    /// §4.3). Taken at the first instant and handed to every entry the gesture builds, never
    /// taken again: by a later instant the first one has settled, and can have put blocks inside
    /// one that `before` does not hold. Each later instant settles from it too, since an entry
    /// settles from the maps it is handed (`SetPartGraph`).
    ownership: OwnershipSnapshot,
    at: std::time::Instant,
}

/// The same, for a material being dragged: the channels the gesture started from.
struct MaterialGesture {
    part: PartId,
    before: Option<Material>,
    at: std::time::Instant,
}

/// What the shared direction editor calls its choices, which depends on what the direction is
/// for. An extrude *sweeps along a line*; a draft read has a *vector*, and the curve you pick is
/// that vector. Same editor, same type, different nouns — the nouns are the only thing that
/// differs, and they are the thing a person reads.
#[derive(Clone, Copy)]
struct DirectionWords {
    /// The choice that takes a curve.
    along: &'static str,
    along_hint: &'static str,
    /// The row that names the curve, once that choice is made.
    curve_row: &'static str,
    /// Why the choice is disabled, when no curve can be picked.
    none_hint: &'static str,
}

const SWEEP_WORDS: DirectionWords = DirectionWords {
    along: "Line",
    along_hint: "Lean it the way a curve runs, end to end. Where the curve sits does not \
                 matter — only which way it points — so it need not touch the \
                 profile or be anywhere near it.",
    curve_row: "Line:",
    none_hint: "Needs a curve that is not part of the profile. Every curve here is one the \
                profile is made of, and a profile cannot be swept along itself — draw \
                another line first.",
};

const DRAFT_WORDS: DirectionWords = DirectionWords {
    along: "Vector",
    along_hint: "A curve whose direction, start to end, is the draft vector. A line — a \
                 two-point curve — is the natural pick. Where it sits does not matter, \
                 only which way it points.",
    curve_row: "Curve:",
    none_hint: "No curves in the document yet. Add a line and this can point along it.",
};

/// The draft-angle overlay's settings. A view tool, like the section: it changes what is drawn
/// and nothing about the part, so it is application state and not the document's.
struct DraftAnalysis {
    enabled: bool,
    /// Which way the mould pulls — the same type an extrude is aimed with, edited with the same
    /// controls, so a line picked here is a pick an extrude could make (D9).
    pull: fieldform_scene::Sweep,
    angle_deg: f64,
    tolerance_deg: f64,
    /// Pull the other way: the read for the other half of the mould.
    flipped: bool,
}

impl Default for DraftAnalysis {
    fn default() -> Self {
        Self {
            enabled: false,
            // Straight up, which is where a mould opens until someone says otherwise.
            pull: fieldform_scene::Sweep::Toward {
                xyz: [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(1.0)],
            },
            // The degree or two most mouldings ask for, with a degree of blur about it.
            angle_deg: 3.0,
            tolerance_deg: 1.0,
            flipped: false,
        }
    }
}

/// Whether two graphs differ only in their numbers.
///
/// The test that separates a drag from an act. Dragging a radius rewrites a parameter and
/// leaves every node and every edge exactly where it was; adding a body, deleting one, or
/// wrapping one in a placement does not. So this asks the structural question and ignores the
/// values, which is precisely the line between "still the same gesture" and "a new one".
///
/// **The curves a node reads are left out on purpose**, though they are edges to who owns what
/// (`block-ownership-spec.md` I2). A console line builds its block on defaults and then gives it
/// the keys it was told, and it is one undo step because those keys continue the line's gesture:
/// `revolve profile= axis=` points the frame at a curve after the revolve is made. Comparing curves
/// would split such a line in two, and its first entry would settle the default for good. A curve
/// re-pointed inside a gesture is owned as one edit from the gesture's start owns it, because
/// every later instant settles from there (`SetPartGraph`).
fn same_structure(a: &Graph, b: &Graph) -> bool {
    if a.nodes().count() != b.nodes().count() {
        return false;
    }
    // Marking a surface critical is a deliberate act, never an instant of a drag, so it takes
    // its own undo entry however soon after another edit it comes.
    if a.critical() != b.critical() {
        return false;
    }
    a.nodes().all(|(id, node)| match b.node(id) {
        // Edges, and the kind. A node whose *kind* changed is a different act even at the same
        // id — switching a boolean from subtract to union is not a drag.
        Some(other) => {
            node.inputs() == other.inputs()
                && std::mem::discriminant(node) == std::mem::discriminant(other)
        }
        None => false,
    })
}

/// Puts a graph edit on the history: a new entry, or the next instant of the gesture in progress.
///
/// The history half of `App::set_part_graph`, which says what a continuation is and why. Free of
/// the `App` so a test can hold the rule, since an `App` needs a window and a device.
fn record_graph_edit(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    part: PartId,
    graph: Graph,
) -> Result<(), fieldform_ops::OpError> {
    let current = doc
        .parts
        .get(&part)
        .expect("active_part must name a part in the document")
        .graph
        .clone();
    let continues = gesture.as_ref().is_some_and(|g| {
        g.part == part && g.at.elapsed() < GESTURE_WINDOW && same_structure(&current, &graph)
    });
    // A continuation reaches back to where the gesture began, for who owned what as for the
    // graph: add a blend and drag its radius, and the drag's entry undoes to a graph with no
    // blend, so the ownership it puts back must be from before the blend adopted its operands.
    // And it settles from there, so the entry it leaves is the one an edit straight from the start
    // makes, while saying only what moved since the instant before, which the last frame said
    // (`SetPartGraph`, refactor plan 2.22). Shared rather than copied, so building each instant's
    // entry copies nothing.
    let (before, ownership) = match (gesture.as_ref(), continues) {
        (Some(g), true) => (g.before.clone(), g.ownership.clone()),
        _ => (current, OwnershipSnapshot::of(doc)),
    };
    *gesture = Some(Gesture {
        part,
        before: before.clone(),
        ownership: ownership.clone(),
        at: std::time::Instant::now(),
    });
    let cmd = Box::new(SetPartGraph::with_ownership(part, before, graph, Some(ownership)));
    if continues {
        stack.replace_top(doc, cmd)
    } else {
        stack.execute(doc, cmd)
    }
}

/// Undoes one entry of the history (`back`) or redoes one, and ends the graph gesture in
/// progress.
///
/// **A gesture may only replace an entry of its own, and undo and redo both move the top.** A
/// gesture folds each new instant into the entry on top of the history, so after an undo it would
/// fold into the entry *below* the one taken back: a same-structure edit inside `GESTURE_WINDOW`
/// replaced an edit nobody had undone, and nothing could bring it back. Ending it here is the
/// whole fix, because a new gesture can only start with a new entry. Free of the `App` for the
/// reason `record_graph_edit` is.
fn step_history(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    back: bool,
) -> Option<String> {
    *gesture = None;
    if back { stack.undo(doc) } else { stack.redo(doc) }
}

/// What a person is told of the moves settling made (`block-ownership-spec.md` §6.3): one line,
/// for a block nothing reads any more.
///
/// **It used to say two things, and item 112 took one away.** The other line was "is now a
/// variable", for the promotion a second reader caused; settling never declares now
/// (`notebook-spec.md` §12, R5), so a second reader makes a shared instance and there is nothing
/// to announce. Adoption is still not said, for the reason it never was: adopting is what every
/// consume does, so it would be a line under every boolean. What is left is the one move nobody
/// asked for in so many words — the block that held one deleted.
///
/// **Once, and only while it is still true.** `changes` is everything settled since the frame
/// last asked, so it can hold a block twice — let go of, then adopted again by the next edit —
/// and can hold a move an undo has taken back, since an undo restores who owned what without
/// settling. So each block is judged by its last change, and only if the document bears it out.
///
/// Readers are the ones reading the block as the line is said, named in the order they were made
/// rather than by id, which is random, so the same edits always say the same line. Free of the
/// `App`, which supplies what to call a block, so a test can hold the rule.
fn ownership_news(
    doc: &Document,
    changes: &[OwnershipChange],
    label: impl Fn(BlockKey) -> String,
) -> Vec<String> {
    let block_of = |change: &OwnershipChange| change.block();
    let still_here = |block| match block {
        BlockKey::Node(id) => doc.parts.values().any(|p| p.graph.node(id).is_some()),
        BlockKey::Curve(id) => doc.curves.contains_key(&id),
    };
    let mut heard = std::collections::HashSet::new();
    let mut last: Vec<&OwnershipChange> =
        changes.iter().rev().filter(|c| heard.insert(block_of(c))).collect();
    last.reverse();
    last.into_iter()
        .filter(|c| still_here(block_of(c)))
        .filter_map(|change| match change {
            // **Only a release is news, since 112.** The other line this said was "is now a
            // variable — Boolean and Shell both use it", for the promotion a second reader used
            // to cause; settling never declares now (`notebook-spec.md` §12, R5), so a block
            // gaining a second reader is simply drawn in both and there is nothing to announce.
            // Adoption was silent before and stays silent: a block moving into the one thing that
            // reads it is what the person just asked for by wiring it.
            OwnershipChange::Released { block } if doc.ownership(*block) == Ownership::Free => {
                Some(format!("{} is on its own again", label(*block)))
            }
            _ => None,
        })
        .collect()
}

/// The blocks a drain names, in the order it names them (`block-ownership-spec.md` §4.2, §9.2).
///
/// **Nothing, in a frame, since 112.** This named the variables a settle had declared, and
/// settling no longer declares anything (`notebook-spec.md` §12, R5) — every variable now comes
/// from the person asking, which names it in the same command. The load keeps its rule, below,
/// because the rule a file from before 3.15 is placed by still makes variables directly.
///
/// **After a load, the unnamed variables two or more blocks read**, reported or not: the rule a
/// file from before 3.15 is placed by makes such a block a variable directly, which no settle
/// reports, and those are the blocks §9.2 has the load's drain name. Not every unnamed variable:
/// one with fewer readers is the same cleared name, saved, and naming it on opening lifted it
/// back out of the reader it had been put back into, so its row moved between a save and the
/// next open.
///
/// Free of the `App`, which supplies the names, so a test can hold which blocks get one.
fn variables_to_name(doc: &Document, after_load: bool) -> Vec<NodeId> {
    if after_load {
        let mut unnamed = doc.unnamed_variables_among(doc.ownership_nodes.keys().copied());
        // Counted only for the few with no name, since each count walks the document.
        unnamed.retain(|id| doc.block_readers(BlockKey::Node(*id)).len() >= 2);
        return unnamed;
    }
    Vec::new()
}

/// Parameter edits a panel made this frame, held for the graph edit they go with.
///
/// **Naming a value, and the × that puts it back to a number, are each one act with two halves:**
/// the parameter, and the graph edit pointing a field at it or away from it. The parameter half
/// used to be written straight into the document while the graph half went on the history, so
/// Ctrl+Z took back one half: undoing a × restored a `ParamRef::Named` to a parameter that was
/// gone, and undoing a naming left the parameter behind with nothing reading it (refactor plan
/// §3.3, root cause 7). Held here, the two go on the history as one entry
/// ([`record_graph_edit_then`]); what no graph edit takes is settled by
/// [`record_parameter_edits`].
#[derive(Default)]
struct ParameterEdits {
    /// Parameters made this frame. Already in the document, so the rest of the frame reads the
    /// value the field now names; putting them on the history writes them again, unchanged.
    made: Vec<SetParameter>,
    /// Parameters a field stopped reading. Whether one goes is decided against the graph the edit
    /// writes, since until then the field's own block is still a reader.
    released: Vec<ParamId>,
}

/// Makes the parameter a value is named as: in the document now, and on the history with the
/// graph edit that points the field at it, if one does.
fn name_value(doc: &mut Document, edits: &mut ParameterEdits, name: String, value: f64) -> ParamId {
    let id = ParamId::new();
    let parameter = fieldform_scene::Parameter {
        id,
        name,
        value: fieldform_scene::ParamValue::Literal(value),
    };
    let mut made = SetParameter::insert(parameter);
    // An insert writes a map entry and cannot fail.
    let _ = fieldform_ops::Command::execute(&mut made, doc);
    edits.made.push(made);
    id
}

/// [`record_graph_edit`], with the parameter edits that came with it, the `shown` flags it
/// leaves stale ([`stale_shown`]), and `then`, in the same entry.
///
/// The parameters made go in front of the graph edit and the ones let go of behind it, so undo
/// puts a released parameter back before the graph reads it again, and takes a made one out only
/// once the graph has stopped reading it.
///
/// **`then` is the rest of an act the graph edit is only half of**, as the eye Pattern and Mirror
/// turn on for the body they consume is (`block-ownership-spec.md` §7.3). It goes after the graph
/// edit, which settles who owns what inside its own execute, so `then` acts on the settled state, a
/// redo replays the two in that order, and an undo takes both back. As two entries, taking a
/// pattern back took two presses of Ctrl+Z, and the one Ctrl+Y after them brought the pattern back
/// with its source no longer drawn: the eye was the entry nothing had redone. After the stale flags
/// too, so what the caller asks for is the last word on a flag.
///
/// **Never the next instant of a gesture, and the end of one.** A gesture folds each instant into
/// the entry on top by replacing it with a bare graph edit, so continuing one here would drop the
/// parameter half, or `then`, from the history while the document kept it, and a drag straight
/// after would do the same to this entry. A flag is let go of only by an edit that changes what
/// reads a block, which a gesture's next instant never does. With nothing held, nothing let go of
/// and nothing to follow, this is `record_graph_edit` exactly.
fn record_graph_edit_then(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    edits: &mut ParameterEdits,
    part: PartId,
    graph: Graph,
    then: Vec<Box<dyn fieldform_ops::Command>>,
) -> Result<(), fieldform_ops::OpError> {
    let mut released = std::mem::take(&mut edits.released);
    released.sort();
    released.dedup();
    let removed: Vec<SetParameter> = released
        .into_iter()
        .filter(|&id| !doc.parameter_is_read(id, Some((part, &graph))))
        .filter_map(|id| doc.parameters.get(&id).cloned().map(SetParameter::remove))
        .collect();
    let made = std::mem::take(&mut edits.made);
    let stale = stale_shown(
        doc,
        &doc.parts.get(&part).expect("active_part must name a part in the document").graph,
        &graph,
    );
    if made.is_empty() && removed.is_empty() && stale.is_empty() && then.is_empty() {
        return record_graph_edit(stack, doc, gesture, part, graph);
    }
    *gesture = None;
    let before = doc
        .parts
        .get(&part)
        .expect("active_part must name a part in the document")
        .graph
        .clone();
    // A delete that lets go of a block reads "Undo: part.set_graph" as any other graph edit does;
    // the flag it clears is not an act of its own, and nor is the eye a pattern turns on.
    let name = match (made.is_empty(), removed.is_empty()) {
        (false, _) => "parameter.name_value",
        (true, false) => "parameter.unname_value",
        (true, true) => "part.set_graph",
    };
    let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
    for edit in made {
        commands.push(Box::new(edit));
    }
    commands.push(Box::new(SetPartGraph::new(part, before, graph)));
    for flag in stale {
        commands.push(Box::new(flag));
    }
    commands.extend(then);
    for edit in removed {
        commands.push(Box::new(edit));
    }
    stack.execute(doc, Box::new(fieldform_ops::Batch::new(name, commands)))
}

/// [`record_graph_edit_then`] with nothing to follow the graph edit: what `App::set_part_graph`
/// runs, for the tests that hold it.
#[cfg(test)]
fn record_graph_edit_with(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    edits: &mut ParameterEdits,
    part: PartId,
    graph: Graph,
) -> Result<(), fieldform_ops::OpError> {
    record_graph_edit_then(stack, doc, gesture, edits, part, graph, Vec::new())
}

/// The `shown` flags a graph edit from `before` to `after` leaves stale, each as the command that
/// clears it: one on every block `before` consumed that nothing in `after` reads.
///
/// **Let go of, it is a body again and draws as one,** so the flag that drew it as an ingredient
/// has outlived its reason. Left standing, it drew the block beside whatever consumed it next as
/// though that had been asked for: pattern a sphere, delete the pattern, union the sphere, and the
/// sphere stood beside the union (refactor plan §3.3, root cause 9; cleared, as decided in §7).
/// Clearing it changes nothing on screen at the time: a body draws unless hidden, whatever this
/// flag says.
///
/// Only `shown`: `hidden` is what a body's eye means, and every consume clears it on purpose so a
/// block hidden and then consumed comes back when let go of (`block-ownership-spec.md` §7.2).
///
/// The flag is asked first, and few blocks hold it, so a drag's every frame walks the graph only
/// for those.
fn stale_shown(doc: &Document, before: &Graph, after: &Graph) -> Vec<SetShown> {
    let mut ids: Vec<NodeId> = doc
        .node_meta
        .iter()
        .filter(|(id, meta)| {
            meta.shown
                && after.node(**id).is_some()
                && before.is_consumed(**id)
                && !after.is_consumed(**id)
        })
        .map(|(id, _)| *id)
        .collect();
    // In a fixed order, so the history holds the same entry however the map iterates.
    ids.sort();
    ids.into_iter().map(|id| SetShown::new(id, false)).collect()
}

/// What is left of a frame's parameter edits once no graph edit has taken them. A parameter let go
/// of that nothing reads goes on the history as an entry of its own; whether anything did.
///
/// **A parameter made for a reference no graph edit wrote stays off the history.** The draft
/// analysis's pull offers "Name this value" on X, Y and Z, and the pull is the panel's own state,
/// not the document's. On the history alone, the parameter was the next Ctrl+Z's to take, and the
/// pull went on reading it: the field said `missing` and the pull resolved to nought. A parameter
/// goes on the history with the reference to it or not at all, so this one stays in the document as
/// every named value did before naming could be undone. Removal is the other way round, because
/// undoing it puts back a parameter that at worst nothing reads.
fn record_parameter_edits(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    edits: &mut ParameterEdits,
) -> Result<bool, fieldform_ops::OpError> {
    edits.made.clear();
    let mut recorded = false;
    let mut released = std::mem::take(&mut edits.released);
    released.sort();
    released.dedup();
    for id in released {
        recorded |= remove_unread_parameter(stack, doc, gesture, id)?;
    }
    Ok(recorded)
}

/// Removes parameter `id` on the history, if nothing reads it; whether it did.
///
/// Ends the graph's gesture, since the next instant of a drag would otherwise replace this entry
/// and the removal would be off the history again.
fn remove_unread_parameter(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    id: ParamId,
) -> Result<bool, fieldform_ops::OpError> {
    if doc.parameter_is_read(id, None) {
        return Ok(false);
    }
    let Some(parameter) = doc.parameters.get(&id).cloned() else {
        return Ok(false);
    };
    *gesture = None;
    stack.execute(doc, Box::new(SetParameter::remove(parameter)))?;
    Ok(true)
}

/// The numbers a typed direction's X, Y and Z read now, held across a step through the history for
/// [`keep_typed_direction`]; `None` for a direction with nothing typed.
fn typed_direction(doc: &Document, sweep: &fieldform_scene::Sweep) -> Option<[f64; 3]> {
    match sweep {
        fieldform_scene::Sweep::Toward { xyz } => Some(xyz.map(|r| doc.resolve(&r).unwrap_or(0.0))),
        _ => None,
    }
}

/// Puts each of a typed direction's X, Y and Z that names a parameter the document no longer holds
/// back to the number it read before the step, `was`.
///
/// **For the draft pull, the one reader of a parameter the history cannot see.** The pull is the
/// analysis panel's own state, so it is not on the history, and its "or use" can point it at a
/// parameter a naming made. Undoing that naming takes the parameter out once nothing in the
/// document reads it, and the pull read `missing` and resolved to nought. It goes back to the
/// number instead, which is what the × beside a named value does.
fn keep_typed_direction(doc: &Document, sweep: &mut fieldform_scene::Sweep, was: Option<[f64; 3]>) {
    let (fieldform_scene::Sweep::Toward { xyz }, Some(was)) = (sweep, was) else {
        return;
    };
    for (r, number) in xyz.iter_mut().zip(was) {
        if let ParamRef::Named(id) = *r
            && !doc.parameters.contains_key(&id)
        {
            *r = ParamRef::Literal(number);
        }
    }
}

/// How much of the window the inspector may take, and the least it may be squeezed to.
///
/// A fraction rather than a width, so the viewport keeps the same share of a laptop screen as
/// of a monitor. The floor is what a number and its `fx` button need to stay usable — below
/// that the panel is present without being editable, which is worse than narrow.
///
/// **The ceiling was 0.34 and stopped the user reading their own block names.** It was put there
/// when the panel sized itself to its content, where the danger was real: a row of operations
/// squeezed the viewport to a strip, and the thing being modelled lost room every time it gained
/// an operation. That danger is gone — the panel truncates rather than demanding room, so nothing
/// in it can move the divider any more — and what was left was a cap on how wide somebody is
/// *allowed to drag it themselves*, which is a different thing and not one worth defending. What
/// the ceiling is for now is only that the viewport should not be draggable away to nothing.
const INSPECTOR_MAX_FRACTION: f32 = 0.7;
const INSPECTOR_MIN_WIDTH: f32 = 180.0;

/// The width the inspector opens at, before anyone drags it. Unchanged by the ceiling above:
/// wanting the *option* of a wide panel is not the same as wanting one by default.
const INSPECTOR_DEFAULT_WIDTH: f32 = 360.0;

/// Cells along the longest axis when measuring a body.
///
/// 128 is where the accuracy stops being the limiting factor and the wait starts to be:
/// `measure_accuracy.rs` reports 0.012% on a sphere at 24 ms, against 0.049% at 64 and 0.006%
/// for 69 ms at 192. Doubling the cells quarters the error and roughly triples the time, so
/// this is the knee rather than a round number.
const MEASURE_CELLS: usize = 128;

/// A fresh phase of the given banding, sized for a cell of `cell_mm`.
///
/// One function because the panel seeds a phase from two places — switching banding, and
/// switching to a cell that cannot take the banding it had — and two copies of "what a new solid
/// starts at" would drift. Nothing carries across from the old phase, and nothing can: there is
/// no conversion between a wall of 2 mm, a density of 40% and a beam radius.
fn seeded_phase(banding: Banding, cell_mm: f64) -> LatticePhase {
    match banding {
        // A tenth of the cell, which is a lattice rather than a block of cheese.
        Banding::Sheet => {
            LatticePhase::sheet(ParamRef::Literal((cell_mm * 0.1).max(0.001)))
        }
        // What a part is usually printed at.
        Banding::Solid => LatticePhase::Solid { density: ParamRef::Literal(0.4) },
        // Thinner than a wall for the same cell, because a strut is material in every direction
        // where a wall is thin in one.
        Banding::Strut => {
            LatticePhase::Strut { radius: ParamRef::Literal((cell_mm * 0.06).max(0.001)) }
        }
    }
}

/// The range a fresh grade runs over: half to half again of the number the lattice had, so the
/// switch reads as a grade of the lattice on screen rather than as a resize -- and, for a
/// density, kept inside what a fraction can mean.
fn grade_range(banding: Banding, was: f64) -> (f64, f64) {
    match banding {
        Banding::Solid => {
            let was = if was > 1e-9 && was <= 1.0 { was } else { 0.4 };
            ((was * 0.5).clamp(0.05, 0.9), (was * 1.5).clamp(0.1, 0.95))
        }
        Banding::Sheet | Banding::Strut => {
            let was = if was > 1e-9 { was } else { DEFAULT_BODY_MM * 0.01 };
            (was * 0.5, was * 1.5)
        }
    }
}

/// A lattice cell's name, in one place so the block, the menu and the console agree.
fn lattice_cell_label(cell: LatticeCell) -> &'static str {
    match cell {
        LatticeCell::Gyroid => "Gyroid",
        LatticeCell::SchwarzP => "Schwarz P",
        LatticeCell::SchwarzD => "Schwarz D",
        LatticeCell::SchoenIwp => "Schoen IWP",
        LatticeCell::Neovius => "Neovius",
        LatticeCell::FischerKochS => "Fischer-Koch S",
        LatticeCell::Cubic => "Cubic",
        LatticeCell::Bcc => "BCC",
        LatticeCell::Fcc => "FCC",
        LatticeCell::Octet => "Octet",
        LatticeCell::Diamond => "Diamond",
        LatticeCell::Honeycomb => "Honeycomb",
        LatticeCell::VoronoiFoam { .. } => "Voronoi foam",
        LatticeCell::VoronoiBeam { .. } => "Voronoi beam",
    }
}

/// Every lattice cell, in the order Create lists them.
const LATTICE_CELLS: [LatticeCell; 14] = [
    LatticeCell::Gyroid,
    LatticeCell::SchwarzP,
    LatticeCell::SchwarzD,
    LatticeCell::SchoenIwp,
    LatticeCell::Neovius,
    LatticeCell::FischerKochS,
    // The beams last, after the six surfaces: two families in one list, in the order they were
    // built rather than interleaved, so the list reads as what it is.
    LatticeCell::Cubic,
    LatticeCell::Bcc,
    LatticeCell::Fcc,
    LatticeCell::Octet,
    LatticeCell::Diamond,
    // A prism: hexagons in the plane, straight through the third axis. Its cell size is the
    // hexagon's flat-to-flat width and its wall is exact.
    LatticeCell::Honeycomb,
    // The two built from a jittered seed grid, and the first cells carrying numbers of their
    // own. Their defaults are what `LatticeCell::voronoi_*` says, so a document that leaves
    // either out reads back as the cell this list offers.
    LatticeCell::VoronoiFoam { jitter: ParamRef::Literal(0.75), seed: 0 },
    LatticeCell::VoronoiBeam { jitter: ParamRef::Literal(0.75), seed: 0 },
];

/// The analytic solids Create offers, and the one place their default proportions live.
///
/// An enum rather than the boolean this used to be, which only ever had a sphere and a cube to
/// choose between. Keeping the default size here — instead of at each call site — is what
/// makes "every new body arrives at the same size" a property of the list rather than a habit
/// each entry has to remember.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrimitiveKind {
    Sphere,
    Cube,
    SkewedBox,
    Cylinder,
    Cone,
    Torus,
    Capsule,
    Plane,
    Point,
}

impl PrimitiveKind {
    /// Every primitive, in the order they are offered.
    ///
    /// **Reference geometry first, then solids.** A point locates a plane and a plane locates a
    /// curve, so these come before the thing they are reached for; the solids after them run
    /// simplest idea first -- a ball, a box, then the ones taking two ends and a radius. It used
    /// to be solids first with the plane and the point trailing, on the argument that neither is
    /// a solid. True, and beside the point: what a menu orders is the order they are reached in.
    ///
    /// **The menu and the console read the command registry now** (`commands.rs`), which lists
    /// each primitive in this order; this is what the tests walk to hold the two together —
    /// `every_primitive_is_reachable_from_the_console` asks that every kind has an entry with its
    /// word and its operation, in this order, in Create › Primitives.
    #[cfg(test)]
    const ALL: [PrimitiveKind; 9] = [
        PrimitiveKind::Point,
        PrimitiveKind::Plane,
        PrimitiveKind::Sphere,
        PrimitiveKind::Cube,
        PrimitiveKind::SkewedBox,
        PrimitiveKind::Cylinder,
        PrimitiveKind::Cone,
        PrimitiveKind::Torus,
        PrimitiveKind::Capsule,
    ];

    /// Where the reference geometry ends and the solids begin, which is where the menu puts a
    /// curve and a separator.
    #[cfg(test)]
    const SOLIDS_START: usize = 2;

    fn label(self) -> &'static str {
        match self {
            PrimitiveKind::Sphere => "Sphere",
            PrimitiveKind::Cube => "Cube",
            PrimitiveKind::SkewedBox => "Skewed box",
            PrimitiveKind::Cylinder => "Cylinder",
            PrimitiveKind::Cone => "Cone",
            PrimitiveKind::Torus => "Torus",
            PrimitiveKind::Capsule => "Capsule",
            PrimitiveKind::Plane => "Plane",
            PrimitiveKind::Point => "Point",
        }
    }

    /// The one word the console knows this by: its label lowercased, with the spaces closed up,
    /// because a console word is a single token and "skewed box" would read as a word and an
    /// argument.
    fn word(self) -> String {
        self.label().to_lowercase().replace(' ', "")
    }

    /// The operation name `dispatch` knows this by: the inverse of `dispatch`'s own arms, which
    /// the tests hold against the registry.
    #[cfg(test)]
    fn command(self) -> &'static str {
        match self {
            PrimitiveKind::Sphere => "create.sphere",
            PrimitiveKind::Cube => "create.cube",
            PrimitiveKind::SkewedBox => "create.skewbox",
            PrimitiveKind::Cylinder => "create.cylinder",
            PrimitiveKind::Cone => "create.cone",
            PrimitiveKind::Torus => "create.torus",
            PrimitiveKind::Capsule => "create.capsule",
            PrimitiveKind::Plane => "create.plane",
            PrimitiveKind::Point => "create.point",
        }
    }

    /// The node this makes, sized so every kind arrives inside the same `size` envelope.
    ///
    /// Same size on screen whichever one is asked for, which is why the sphere takes half of
    /// `size` as a radius and the torus splits it between its two: a Create menu where each
    /// entry appeared at a different scale would make the first thing you do after every one
    /// of them a resize.
    fn node(self, centre: [ParamRef; 3], size: f64) -> Node {
        let literal = ParamRef::Literal;
        // An end offset up or down Z from the centre, resolved to a number: a new body has
        // literal coordinates, and ends that were parameter references to a centre the shape
        // no longer has would be unreadable.
        let along_z = |c: [ParamRef; 3], dz: f64| {
            let at = |r: ParamRef| match r {
                ParamRef::Literal(v) => v,
                ParamRef::Named(_) | ParamRef::Field(_) => 0.0,
            };
            [literal(at(c[0])), literal(at(c[1])), literal(at(c[2]) + dz)]
        };
        match self {
            PrimitiveKind::Sphere => Node::Sphere {
                centre: fieldform_scene::Place::At(centre),
                radius: literal(size * 0.5),
            },
            PrimitiveKind::Cube => Node::Cuboid {
                centre: fieldform_scene::Place::At(centre),
                size: [literal(size); 3],
                turn: fieldform_scene::no_turn(),
                edge: fieldform_scene::Fillet::hard(),
            },
            // The two-point solids arrive standing on Z and centred like the rest, so no
            // Create entry lands somewhere different from its neighbours. Where they go after
            // that is a drag of either end.
            // Leaning on every axis at once, so what arrives is plainly not a cube and a
            // click on Create shows what the entry is for. The first edge stays on X so the
            // lean reads as a lean rather than as a turn, and each edge is a full edge, as a
            // cube's `size` is a full side. About the same volume as the cube beside it.
            PrimitiveKind::SkewedBox => Node::SkewedBox {
                centre: fieldform_scene::Place::At(centre),
                edges: [
                    [literal(size), literal(0.0), literal(0.0)],
                    [literal(size * 0.3), literal(size * 0.85), literal(0.0)],
                    [literal(size * 0.15), literal(size * 0.1), literal(size * 0.85)],
                ],
            },
            PrimitiveKind::Cylinder => Node::Cylinder {
                ends: Ends::between(
                    fieldform_scene::Place::At(along_z(centre, -size * 0.5)),
                    fieldform_scene::Place::At(along_z(centre, size * 0.5))),
                radius: literal(size * 0.5),
                edge: fieldform_scene::Fillet::hard(),
            },
            // A plain cone, apex up: the shape the word names. A top radius above zero
            // truncates it, which is why there is no separate entry for that.
            PrimitiveKind::Cone => Node::Cone {
                ends: Ends::between(
                    fieldform_scene::Place::At(along_z(centre, -size * 0.5)),
                    fieldform_scene::Place::At(along_z(centre, size * 0.5))),
                base_radius: literal(size * 0.5),
                top_radius: literal(0.0),
                edge: fieldform_scene::Fillet::hard(),
            },
            // Split so the outer radius is still half the envelope.
            PrimitiveKind::Torus => Node::Torus {
                centre: fieldform_scene::Place::At(centre),
                major_radius: literal(size * 0.35),
                minor_radius: literal(size * 0.15),
                turn: fieldform_scene::no_turn(),
            },
            // The points are cap centres, so the ends reach a radius past them: a quarter of
            // the envelope each way leaves the whole capsule `size` long.
            PrimitiveKind::Capsule => Node::Capsule {
                ends: Ends::between(
                    fieldform_scene::Place::At(along_z(centre, -size * 0.25)),
                    fieldform_scene::Place::At(along_z(centre, size * 0.25))),
                start_radius: literal(size * 0.25),
                end_radius: literal(size * 0.25),
            },
            // Both arrive at the centre rather than offset like the solids: there is no
            // envelope to fill, since one has no edges and the other no volume.
            PrimitiveKind::Plane => Node::Plane {
                origin: fieldform_scene::Place::At(centre),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            },
            PrimitiveKind::Point => Node::Point { at: centre },
        }
    }
}

/// The depth an extrude has once its arrow has been pulled `along` the sweep from where it was.
///
/// The face being pulled stands at half the depth of a symmetric extrude, so it moves the depth
/// twice as far; side two grows as its face goes the other way, against the sweep. Ctrl rounds to
/// whole millimetres.
fn dragged_depth(side: usize, symmetric: bool, was: f64, along: f64, snap: bool) -> f64 {
    let depth = match (side, symmetric) {
        (0, false) => was + along,
        (0, true) => was + 2.0 * along,
        _ => was - along,
    };
    if snap { depth.round() } else { depth }
}

/// Whether nothing above `id` moves where its points are drawn.
///
/// What a handle placed from a block's own coordinates needs to know: a placement, a pattern, a
/// mirror, a twist, an attractor or a morph above it would draw the solid somewhere else, and an
/// arrow standing where the *extrude* is would stand over nothing. A boolean, a blend, an offset or
/// a shell above it leaves the face where it is.
fn drawn_where_it_is(graph: &Graph, id: NodeId) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![id];
    while let Some(below) = stack.pop() {
        for (consumer, node) in graph.nodes() {
            if !node.inputs().contains(&below) {
                continue;
            }
            if matches!(
                node,
                Node::Transform { .. }
                    | Node::Pattern { .. }
                    | Node::Mirror { .. }
                    | Node::Twist { .. }
                    | Node::Attract { .. }
                    | Node::Morph { .. }
            ) {
                return false;
            }
            if seen.insert(consumer) {
                stack.push(consumer);
            }
        }
    }
    true
}

/// The most dots a pattern's skip row draws. A pattern may hold five hundred copies, and a row of
/// that many buttons is more than anyone can aim at; what is past this is still kept and still
/// skipped, and the console (`skip=`) reaches all of them.
const SKIP_DOTS_MAX: u32 = 120;

/// One dot per copy, filled while the copy is there and open once it is skipped; a click toggles it.
///
/// **The copy standing on the body has no dot to click.** Leaving it out would delete the body from
/// a pattern that is meant to include it, and the layout refuses a skip list that names it — so the
/// dot is drawn, because it is a copy, and cannot be pressed, with the reason on hover.
fn skip_dots(
    ui: &mut egui::Ui,
    total: u32,
    body: Option<u32>,
    skip: &[u32],
    per_line: Option<u32>,
) -> Vec<u32> {
    let mut out: Vec<u32> = skip.to_vec();
    let shown = total.min(SKIP_DOTS_MAX);
    // One line of dots for a row or a ring; for a grid, one line a row of copies, so the dots sit
    // where the copies do.
    let lines: Vec<std::ops::Range<u32>> = match per_line {
        Some(n) if n > 0 => (0..shown.div_ceil(n)).map(|r| r * n..((r + 1) * n).min(shown)).collect(),
        _ => vec![0..shown],
    };
    ui.vertical(|ui| {
        for (line, range) in lines.into_iter().enumerate() {
            ui.horizontal_wrapped(|ui| {
                if line == 0 {
                    row_label(ui, "Skip:");
                } else {
                    row_label(ui, "");
                }
                for i in range {
                    let left_out = out.contains(&i);
                    let on_body = body == Some(i);
                    let glyph = if left_out { "\u{25cb}" } else { "\u{25cf}" };
                    let dot = egui::Button::new(egui::RichText::new(glyph).small())
                        .frame(false)
                        .min_size(egui::vec2(12.0, 14.0));
                    let hover = if on_body {
                        format!("Copy {i} stands where the body does, so it cannot be skipped.")
                    } else if left_out {
                        format!("Copy {i} is left out — click to put it back.")
                    } else {
                        format!("Copy {i} — click to leave it out.")
                    };
                    if ui.add_enabled(!on_body, dot).on_hover_text(hover).clicked() {
                        if left_out {
                            out.retain(|&n| n != i);
                        } else {
                            out.push(i);
                        }
                    }
                }
            });
        }
        if total > SKIP_DOTS_MAX {
            ui.label(
                egui::RichText::new(format!("… {total} copies; `skip=` reaches them all"))
                    .small()
                    .weak(),
            );
        }
    });
    out.sort_unstable();
    out.dedup();
    out
}

/// How wide the V mark is, so a row without one still lines up with a row that has one.
const MARK_W: f32 = 14.0;

/// A row's label, right-aligned in the column every row in a block shares.
///
/// Allocated even for an empty label: a row that skips its label would start its numbers at the
/// panel edge while every labelled row starts them further in, and the two would not line up —
/// which is the whole thing the width exists for.
fn grid_gaps(ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing.x = 4.0;
}

/// The smallest and largest face of a surface, as a span in millimetres.
///
/// **What a conformal lattice's cell size is actually set by**, which is the one thing about
/// that map a user cannot work out from the controls: there is no cell size to type, because
/// `s` and `t` run nought to one across every face however big it is. So the panel says the
/// range the mesh produces and lets the number be checked rather than believed.
///
/// A face's span is its mean edge length — not its diagonal and not its area, because the cells
/// are laid out along the edges and that is the direction they are counted in.
fn face_span_mm(mesh: &fieldform_cage::QuadMesh) -> Option<(f64, f64)> {
    let mut small = f64::INFINITY;
    let mut large: f64 = 0.0;
    for face in mesh.face_ids() {
        let v = mesh.face_vertices(face);
        let mut total = 0.0;
        for k in 0..4 {
            let a = mesh.vertex_position(v[k]);
            let b = mesh.vertex_position(v[(k + 1) % 4]);
            total += (b - a).norm();
        }
        let span = total / 4.0;
        small = small.min(span);
        large = large.max(span);
    }
    (small.is_finite() && large > 0.0).then_some((small, large))
}

/// What an OBJ is taken to be, which the file itself cannot say.
///
/// **Three readings of the same triangles and quads, and they are not variations on each
/// other.** A cage is scaffolding for a subdivision surface and has to be closed; a model is
/// finished geometry to reference; a surface is an *open* sheet that nothing draws at all and
/// only a conformal lattice names. One control, because it is one question -- what is this
/// file -- and a second checkbox beside the first would let two answers be given at once.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ObjReading {
    Cage,
    Model,
    Surface,
}

impl ObjReading {
    /// In the order they are offered, with what each one means. The strings live here rather
    /// than in the panel so the list and its explanations cannot come apart.
    const ALL: [(ObjReading, &'static str, &'static str); 3] = [
        (
            ObjReading::Cage,
            "Subdivision cage",
            "The file is a control mesh and the body is the smooth surface it describes, \
             which you can then push and pull. Needs the file to be closed, manifold and all \
             quads.",
        ),
        (
            ObjReading::Model,
            "Model",
            "The triangles are the geometry. Takes any OBJ, including one with holes in it, \
             and the result is referenced rather than edited.",
        ),
        (
            ObjReading::Surface,
            "Surface to follow",
            "An open quad sheet -- a panel or a shell -- that a conformal lattice lays its \
             cells over. Nothing draws it and it is not a body; it is geometry for another \
             block to reference. Quads, but open where a cage may not be.",
        ),
    ];
}

/// What an import brought in, said under its name — a count per kind, and nothing for a kind
/// it did not bring.
fn import_contents(bodies: usize, surfaces: usize, curves: usize) -> String {
    let plural = |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    let mut parts = Vec::new();
    if bodies > 0 {
        parts.push(plural(bodies, "body", "bodies"));
    }
    if surfaces > 0 {
        parts.push(plural(surfaces, "surface", "surfaces"));
    }
    if curves > 0 {
        parts.push(plural(curves, "curve", "curves"));
    }
    if parts.is_empty() {
        "nothing left from this import".to_string()
    } else {
        parts.join(", ")
    }
}

/// What an import that did not make meshes produced, for [`App::record_import`].
#[derive(Default)]
struct ImportMade {
    cages: Vec<CageId>,
    surfaces: Vec<fieldform_scene::QuadMeshId>,
    curves: Vec<CurveId>,
}

fn made_cages(cages: Vec<CageId>) -> ImportMade {
    ImportMade { cages, ..Default::default() }
}

fn made_surfaces(surfaces: Vec<fieldform_scene::QuadMeshId>) -> ImportMade {
    ImportMade { surfaces, ..Default::default() }
}

fn made_curves(curves: Vec<CurveId>) -> ImportMade {
    ImportMade { curves, ..Default::default() }
}

/// The label column every row shares, right-aligned against what follows it.
///
/// Allocated at its full width and painted, rather than laid out inside a child ui: a label
/// laid out right-to-left reports only the text's own width to whatever placed it, which a
/// horizontal row never noticed (its cursor lands on the text's right edge either way) and a
/// grid did -- the column came out as wide as "Start", the text sat at the far right of the
/// space it was given, and the next column started on top of it. That is how a curve's
/// "Start" and "End" were hidden under their X fields, and under a point's name when the end
/// sat on one.
fn row_label(ui: &mut egui::Ui, text: &str) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(AXIS_LABEL_W, ui.spacing().interact_size.y),
        egui::Sense::hover(),
    );
    if !text.is_empty() {
        ui.painter().text(
            rect.right_center(),
            egui::Align2::RIGHT_CENTER,
            text,
            egui::TextStyle::Body.resolve(ui.style()),
            ui.visuals().text_color(),
        );
    }
}

/// Where a number's own label goes.
///
/// **Because a row and a cell are different places.** A row's label belongs in the label column
/// every row shares, right-aligned against the numbers; a cell's is the axis letter, and belongs
/// inside the cell beside its own number. The same widget draws both, so it has to be told which
/// — guessing from the text would be one more rule to know.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Labelled {
    /// In the shared label column: `Radius:`, `At:`, `From:`.
    Row,
    /// Inside the cell, ahead of the number: `X`, `Y`, `Z`.
    Cell,
}

/// One candidate for a reference field: what it is called, and what it may be *typed* as.
///
/// **Two strings, because a label and an address are not the same fact.** The label is what the field
/// shows and the list offers; the address is the text that resolves to this choice, and a choice may
/// have none. A block gets an address only once somebody names it, since its label is what it *is*
/// and eight pipes on spokes all read "Pipe"; a curve's label is already unique, because
/// `unique_curve_name` makes it so on creation. Keeping them apart here is what lets one widget serve
/// both without knowing which it is holding.
struct RefChoice<K> {
    key: K,
    label: String,
    address: Option<String>,
    /// Why this block may not go in the field, if it may not (`operand_refusal`). A refused choice
    /// is still *known* — typing its name or dropping it finds it — but is never suggested and never
    /// taken, and what the person is told is this.
    refusal: Option<&'static str>,
}

impl<K> RefChoice<K> {
    /// A choice whose label is also its address — a curve, or anything else born with a unique name.
    fn named(key: K, name: &str) -> Self {
        Self { key, label: name.to_string(), address: Some(name.to_string()), refusal: None }
    }

    /// A choice that can only be typed once it has been named, which is every block.
    fn declared(key: K, label: String, address: Option<&str>) -> Self {
        Self { key, label, address: address.map(str::to_string), refusal: None }
    }
}

/// A row its V can make a variable or put back inside.
///
/// **Three kinds of row, and one act on each** (`block-ownership-spec.md` §10): the V writes the
/// stored ownership of the block the row stands for, and a name is a label on all three. They
/// were declared three ways until the curves' V wrote the curve ownership (refactor plan 2.19) —
/// a block by its name, a profile by a name of its own (`Document::profile_names`), and a plain
/// curve, born named, by a flag (`Document::declared_curves`) — and the two names moved a row
/// whenever one was typed for any other reason. Kept apart only for what the row is called when
/// the V says what it did.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Declarable {
    Block(NodeId),
    /// A profile row: `key`, its loop's first curve, which the row is drawn under and called by,
    /// and `curve`, the one whose ownership its V writes — the curve that puts the row where it
    /// is ([`profile_v_curve_in`]). Not always the key: an extrude names whichever curve was
    /// picked, and the row goes inside the block that holds any of them.
    Profile { key: CurveId, curve: CurveId },
    Curve(CurveId),
}

impl Declarable {
    /// The block whose ownership the row's V writes.
    fn block(self) -> BlockKey {
        match self {
            Declarable::Block(id) => BlockKey::Node(id),
            Declarable::Profile { curve, .. } | Declarable::Curve(curve) => BlockKey::Curve(curve),
        }
    }
}

/// The rows a reference field offers while it has the focus: everything but what it already
/// holds, narrowed by what has been typed once the text is no longer just the settled name, and at
/// most ten.
///
/// **A refused row is offered and marked, not left out** — the owner's ruling when the gate was
/// built, and the reading it rests on: "a row that is missing answers nothing, and the question
/// somebody has while looking down this list is why their body is not an option." So each row
/// comes back with its refusal beside it, and the caller draws those weak, unclickable and with
/// the reason on hover. Leaving them out put the explanation behind an action — a person had to
/// type the name and submit it to be told why it would not go in, which is the moment they have
/// stopped looking for it.
///
/// **And the row the field already holds is kept too, for the same reason.** It used to be left
/// out, on the reading that offering what is already chosen is noise. Reported from the other
/// side: a twist is created over the selected body, so its `Of:` already held the cube — and the
/// list, which omitted exactly that one, read as "it will not let me pick the cube". The held row
/// is marked instead, so a list is never missing the thing somebody is looking for.
///
/// `typed` is lowercased by the caller. `current` is optional because a field may hold nothing yet,
/// and then there is nothing to leave out.
fn suggested<K: Copy + PartialEq>(
    choices: &[RefChoice<K>],
    typed: &str,
    showing: bool,
) -> Vec<(K, String, Option<&'static str>)> {
    choices
        .iter()
        .filter(|c| !showing || typed.is_empty() || c.label.to_lowercase().contains(typed))
        .take(10)
        .map(|c| (c.key, c.label.clone(), c.refusal))
        .collect()
}

/// Which dragged rows a reference field can take, and how each becomes its kind of reference.
///
/// **A pair rather than two arguments**, because the two are one fact about the field — what it
/// accepts — and a field that took them separately would let a caller pass a block mapper and forget
/// the curve one without anything reading oddly.
///
/// A mapper rather than a bare flag because the key is not always the id: an `Attractor` wraps a
/// `CurveId` one way and a `NodeId` the other, and that wrapping is exactly what the caller knows and
/// the widget does not.
#[derive(Clone, Copy)]
struct DropsFrom<K> {
    block: Option<fn(NodeId) -> K>,
    curve: Option<fn(CurveId) -> K>,
}

impl<K> DropsFrom<K> {
    /// Takes a dragged block.
    fn block(f: fn(NodeId) -> K) -> Self {
        Self { block: Some(f), curve: None }
    }

    /// Takes a dragged curve.
    fn curve(f: fn(CurveId) -> K) -> Self {
        Self { block: None, curve: Some(f) }
    }

    /// Takes either, which only an attractor's target does.
    fn either(block: fn(NodeId) -> K, curve: fn(CurveId) -> K) -> Self {
        Self { block: Some(block), curve: Some(curve) }
    }
}

/// What kind of thing a reference field is asking for, which decides only what it *says*.
///
/// The behaviour is the same for all three — type a name, or pick from the list — so this carries
/// the wording rather than any logic. Separate from the key type because `Attractor` is a curve or a
/// block and could not say which from its type alone.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RefKind {
    /// A block: a body, or any named block in the part.
    Block,
    /// A curve in the document.
    Curve,
    /// Either, which is what an attractor takes.
    Target,
    /// A ramp: what a driveable number is driven by. A block, but one with its own words, since
    /// "No block is called" is the wrong thing to say to someone typing a ramp's name.
    Ramp,
}

impl RefKind {
    /// The empty field's prompt, which is also where a curve field says that a drag will not work.
    fn hint(self) -> &'static str {
        match self {
            RefKind::Block => "name, or drop a block",
            RefKind::Curve => "name, or drop a curve",
            RefKind::Target => "name, or drop either",
            RefKind::Ramp => "name, or drop a ramp",
        }
    }

    /// Why a dropped block was refused — only reachable where a block can be dropped at all.
    fn why_not(self) -> &'static str {
        match self {
            RefKind::Block => {
                "cannot go here — a block cannot be made from something that is made from it"
            }
            RefKind::Curve => "cannot go here — this field takes a curve",
            RefKind::Target => "cannot be attracted to — it is made from this block",
            RefKind::Ramp => "cannot drive this — only a ramp can, and it must not read this block",
        }
    }

    fn nothing(self) -> &'static str {
        match self {
            RefKind::Block => "No block is called",
            RefKind::Curve => "No curve is called",
            RefKind::Target => "Nothing here is called",
            RefKind::Ramp => "No ramp is called",
        }
    }

    /// How to make one that can be typed.
    fn how(self) -> &'static str {
        match self {
            RefKind::Block => {
                "Click the dim V on a block's row to name it, and it can be referred to here."
            }
            RefKind::Curve => "Rename a curve on its own row to call it something you will remember.",
            RefKind::Target => "Name the block with the dim V on its row, or rename the curve.",
            RefKind::Ramp => "Name the ramp with the dim V on its row, or pick it from the list.",
        }
    }
}

/// Whether a number is continuous or counts things.
///
/// **Because a field showing 4 while holding 3.5 is the disagreement the count exists to avoid.**
/// A pattern's copy count is a `ParamRef`, and `fieldform_scene::pattern_count` rounds what it reads
/// — so the number built is always whole, and the *stored* number should be too wherever somebody
/// typed it here rather than drove it from elsewhere. Whole numbers are a property of the quantity,
/// which is the same reasoning the range already travels by: both are facts about what the number
/// *is*, and both belong with the caller that knows.
///
/// It is a grain rather than a flag so the third case has somewhere to go when one turns up.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Grain {
    /// Any value in range: a length, an angle, a ratio.
    Continuous,
    /// Whole numbers only, stepped one at a time and shown without decimals.
    Whole,
}

/// Whether a number offers to become a variable, and the one case where it should not.
///
/// **A position is one thing, so it gets one V.** Three coordinates that each offered to be
/// named put four marks on a row describing a single place, and the row-level one -- the one
/// that matters, because it is how a block comes to sit on a point -- was the least visible of
/// them. Asked for directly: "lets remove that and keep only the XYZ combined parameter as a V".
///
/// A number that is *already* named still shows its mark. Withholding it there would leave a
/// value driven from elsewhere looking like a plain number, which is worse than clutter; the
/// dim offer is what goes, not the bright fact.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Naming {
    /// A dim V inviting this number to become a variable. What almost every number gets.
    Offered,
    /// The mark only when there is already something to mark. For the coordinates of a place,
    /// whose naming happens once for the position rather than three times for its parts.
    OnlyWhenSet,
}

/// How wide one coordinate's field is, everywhere in the application.
///
/// **Fixed, so columns line up across panels and not merely within one.** A position reads as a
/// position when its three numbers are on one line, and two positions read as a pair when the
/// X of one sits above the X of the other. Rows that size to their own content cannot promise
/// that — the same two positions come out ragged whenever their numbers differ in length,
/// which is most of the time. A grid aligns a column within one widget; a shared width aligns
/// it between a curve's ends, a sphere's centre and a transform's move.
const AXIS_FIELD_W: f32 = 98.0;

/// And the label ahead of them, so the fields themselves start at the same place.
///
/// **Right-aligned in it**, which is the half that was missing. The label used to be centred in
/// this width, so a row called "At:" started its text in a different place from one called
/// "From radius:" and the left edge of a block was ragged — asked for as "organise the inside of
/// the blocks better, more on a grid". Right-aligned, every label ends where the numbers begin
/// and the eye has one line to run down.
///
/// Wide enough for the longest label a block row uses. The long ones in the import and export
/// dialogs are plain `ui.label` calls and do not come through here.
const AXIS_LABEL_W: f32 = 84.0;

/// One number's own box, inside its cell.
///
/// Fixed, so the boxes are the same size down the panel and across it. A `DragValue` sizes
/// itself to whatever it is currently showing otherwise, so a column of them steps in and out as
/// the numbers change length — which is what stops a panel reading as a grid even when its
/// columns are aligned.
const NUMBER_W: f32 = 64.0;

/// How fast a number scrubs, given how big it currently is.
///
/// **A fixed rate cannot serve both ends of a CAD document.** The position rows moved at half a
/// millimetre a pixel, which is right for a fillet and useless for a span: a two-metre dimension
/// wants four thousand pixels of travel and there is no screen that wide, while a 0.2 mm fillet
/// moves in steps larger than itself. So the rate follows the magnitude, with the row's own speed
/// as a floor — so a value sitting at zero still moves, and a small one keeps the feel it had.
///
/// Four parts in a thousand per pixel, which puts any number within about two hundred and fifty
/// pixels of doubling: a drag of a couple of inches, whatever it currently reads.
///
/// The other two halves of this gesture are egui's and are not repeated here: holding shift
/// divides the rate by ten, and a `DragValue` already shows the horizontal-resize cursor, so the
/// field says it can be dragged before anybody tries. Ctrl's coarse steps are not egui's, and
/// [`scrub_rate`] adds them.
fn scrub_speed(value: f64, base: f64) -> f64 {
    (value.abs() * 0.004).max(base)
}

/// [`scrub_speed`] as the drag stands this frame: ten times coarser while Ctrl is held
/// (`interaction-spec.md` §4.4), so a number field in the notebook steps as the tool panel's does.
/// Read every frame, so pressing Ctrl part-way through a drag changes the step from there on.
fn scrub_rate(ui: &egui::Ui, value: f64, base: f64) -> f64 {
    let coarse = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
    scrub_speed(value, base) * if coarse { 10.0 } else { 1.0 }
}

/// The axis letter ahead of a number's box ([`NUMBER_W`]), for the rows that have one.
const AXIS_LETTER_W: f32 = 11.0;

/// The green a variable wears when it is one.
///
/// **Two shades, not two marks.** The same V sits on a field that *could* be a variable and on
/// one that *is*, because they are one idea at two stages — so the difference is brightness
/// rather than a different symbol, and the eye reads "available" and "on" without reading a
/// legend. A dim V is an offer; a bright one is a fact.
const VARIABLE_ON: egui::Color32 = egui::Color32::from_rgb(90, 190, 110);

/// The same green, dimmed: this value could be named and has not been.
const VARIABLE_OFF: egui::Color32 = egui::Color32::from_rgb(96, 128, 104);

/// The console entry field's own id, so the Tab hook can tell it from every other text field.
const CONSOLE_INPUT_ID: egui::Id = egui::Id::NULL;

/// What a refusal of a loop that does not close gains, now that there is a second way on.
///
/// Joining the ends was the only answer while a block could be built on a closed profile and
/// nothing else. A curve can be given a thickness instead, and that is a checkbox on the very
/// block the refusal is about — so the refusal says so (`outline-wall-spec.md` §6).
const OR_A_WALL: &str = "Or turn on Wall in the block to give the curve a thickness instead.";

/// Which of the three blocks built on a section a wall is being made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WallBlock {
    Extrude,
    Revolve,
    /// With the spine it was told, if it was told one, which it is built on from the start
    /// (`App::add_sweep_body` says why). Carried here because a sweep is the one of the three that
    /// has a spine, and every way to a wall sweep has to hand it over. `sweep curve= spine=` with
    /// no closed profile anywhere came by a way that did not, and swept along a curve of its own
    /// picking (refactor plan 2.22).
    Sweep { spine: Option<CurveId> },
}

impl WallBlock {
    /// What was done, for the line that says so.
    fn did(self) -> &'static str {
        match self {
            WallBlock::Extrude => "Extruded",
            WallBlock::Revolve => "Revolved",
            WallBlock::Sweep { .. } => "Swept",
        }
    }

}

/// The other side of a curve: left for right, and the middle for itself.
fn across(side: fieldform_scene::WallSide) -> fieldform_scene::WallSide {
    use fieldform_scene::WallSide;
    match side {
        WallSide::Centre => WallSide::Centre,
        WallSide::Left => WallSide::Right,
        WallSide::Right => WallSide::Left,
    }
}

/// The three sides a wall is offered, each as the side it stores, the word it is shown by and
/// what the word means.
///
/// **In the words the curve under it makes true.** Along an open curve there is a left and a
/// right. Round a loop a person means inside and outside, and which of left and right that is
/// depends on the way the loop happens to run — so `inside` is the side the field says is the
/// inside (`Wall::inside`), and the same three buttons store whichever word is true of it.
/// `None` is a curve that closes no loop.
fn side_choices(
    inside: Option<fieldform_scene::WallSide>,
) -> [(fieldform_scene::WallSide, &'static str, &'static str); 3] {
    use fieldform_scene::WallSide;
    let centre = (
        WallSide::Centre,
        "Centre",
        "Half the thickness to each side of the curve.",
    );
    match inside {
        None => [
            centre,
            (
                WallSide::Left,
                "Left",
                "All of it to the left of the curve, walking along it the way it was drawn. \
                 Which way that is on screen is quickest read off the picture.",
            ),
            (
                WallSide::Right,
                "Right",
                "All of it to the right of the curve, walking along it the way it was drawn.",
            ),
        ],
        Some(inside) => [
            centre,
            (inside, "Inside", "All of it inside the loop, so the loop is the wall's outer face."),
            (
                across(inside),
                "Outside",
                "All of it outside the loop, so the loop is the wall's inner face.",
            ),
        ],
    }
}

/// A size: positive, and bounded well past anything anyone models.
///
/// Zero is excluded because a radius of zero has no surface — the shape would simply not be
/// there, which reads as the tool being broken rather than as the number being wrong.
const POSITIVE_MM: std::ops::RangeInclusive<f64> = 0.001..=1.0e6;

/// A join's radius where nought is an answer: a pattern's fillet has no switch beside it, so
/// zero is how it is turned off. With [`POSITIVE_MM`] the field refused the zero its own
/// caption offered -- "Nought is a hard join" -- and left 0.001, which is a rounded join nobody
/// can see and which costs everything a visible one does.
const FROM_NOUGHT_MM: std::ops::RangeInclusive<f64> = 0.0..=1.0e6;

/// A coordinate: anywhere, including zero and the negative half of every axis.
const ANYWHERE_MM: std::ops::RangeInclusive<f64> = -1.0e6..=1.0e6;

/// Rotations are unbounded rather than wrapped to a turn: a manipulator drag accumulates past
/// 360 degrees on purpose (§6.5), and clamping the readout there would make a continuous drag
/// appear to stop.
const DEGREES: std::ops::RangeInclusive<f64> = -1.0e5..=1.0e5;

/// How far round a revolve turns: a full turn is the whole solid, and the widget stops there
/// rather than letting a drag run on past it, since past it nothing changes. **A degree at
/// the least, not nought**: nought is a solid of nothing, which the compile refuses by the
/// number, and a drag that could reach it took the part away under the hand and brought it
/// back a few degrees later -- see the compile error's slot in the part panel for what that
/// did to the drag. The console can still ask for nought and be told why not.
const REVOLVE_DEGREES: std::ops::RangeInclusive<f64> = 1.0..=360.0;

/// The other way round, which may be nothing: side two of a revolve turned only one way.
const REVOLVE_OTHER_DEGREES: std::ops::RangeInclusive<f64> = 0.0..=360.0;

/// A draft angle, which is a wall's lean and so cannot reach a right angle.
///
/// Stopped short of ninety on both sides rather than left open: at ninety the wall is a plane and
/// past it the solid folds back through itself, which is a shape nobody drew. The compile refuses
/// beyond this too, so the two agree; this is what stops a drag reaching it in the first place.
const DRAFT_DEGREES: std::ops::RangeInclusive<f64> = -89.0..=89.0;

/// How big a swept section may end up, as a multiple of the drawn one. **Never nought**: a
/// scale of nothing is a section closed to a point before the end of the path, which the
/// kernel refuses by name -- so the widget stops a hundredth short of it rather than letting
/// a drag take the part away and bring it back, which is what a revolve's angle did.
const TAPER_SCALE: std::ops::RangeInclusive<f64> = 0.01..=20.0;

/// How far round a swept section turns over its extent. Unbounded in the sense a draft is
/// not: more than a full turn is a helix of the section and a perfectly ordinary thing to ask
/// for, and the only cost is the divisor, which the panel states.
const TWIST_DEGREES: std::ops::RangeInclusive<f64> = -3600.0..=3600.0;

/// Where a block starts and ends along its path, as a percentage of the path's length.
const PATH_PERCENT: std::ops::RangeInclusive<f64> = 0.0..=100.0;

/// What a new attractor pulls at: half, which is a visible effect well clear of folding in
/// either direction. The block's panel is where it is changed afterwards.
const DEFAULT_ATTRACT_STRENGTH: f64 = 0.5;

/// An attractor's strength: a fraction, and wide enough on both sides to reach a fold.
///
/// **Not clamped short of folding**, which §8 of the spec expected. The fold threshold is a
/// real number rather than a guess -- `16/11` pulling, `-1` pushing -- so the panel can say
/// where it is and let somebody go past it deliberately. A slider that stopped at 1 would be
/// hiding a limit it knows exactly, and folding out of sight is a legitimate thing to ask for.
const ATTRACT_STRENGTH: std::ops::RangeInclusive<f64> = -4.0..=4.0;

/// A direction's components. Wide rather than clamped to a unit ball, because a direction of
/// any length means the same thing once it is normalized, and silently rewriting a typed 2 to a
/// 1 would look like the field refusing to accept what was typed.
const DIRECTION: std::ops::RangeInclusive<f64> = -1.0e6..=1.0e6;

/// What to call a boolean operation in the interface.
///
/// "Subtract" rather than "Difference" because the tool is a verb: you subtract one body from
/// another. The graph's own name for it stays `BooleanOp::Difference`.
fn boolean_label(op: BooleanOp) -> &'static str {
    match op {
        BooleanOp::Union => "Union",
        BooleanOp::Difference => "Subtract",
        BooleanOp::Intersection => "Intersect",
    }
}

/// One appearance channel: a number dragged within its range. Returns whether it changed.
fn channel(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    hover: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let response = ui.add(
            egui::DragValue::new(value)
                .speed(0.005)
                .range(range)
                .max_decimals(3),
        );
        changed = response.changed();
        response.on_hover_text(hover);
    });
    changed
}

/// The extension a native document is written with.
const DOCUMENT_EXTENSION: &str = "fform";

/// Every extension a native document is *opened* from, newest first.
///
/// **Renamed from `.fieldform` on 2026-09-13, and the old name still opens.** Nothing inside the
/// container changed and nothing reads the extension to decide how to parse — `fieldform_io::open`
/// reads the archive itself — so an older document is not an older *format*, only an older name.
/// Dropping the old one from this list would make files people already have invisible in the
/// dialog while being perfectly readable, which is the worst of both.
const DOCUMENT_EXTENSIONS: &[&str] = &["fform", "fieldform"];

/// A parameter name derived from the field it was promoted from, made unique.
///
/// "Radius:" becomes `radius`, then `radius_2` if that is taken — a document with two named
/// radii is normal, and two parameters with the same name is a document nobody can read.
///
/// "Start radius:" becomes `start_radius`, not `start radius`: an expression has to be able
/// to say the name, so anything that is not a letter, a digit or an underscore becomes an
/// underscore, and a name that would start with a digit gets one in front.
fn unique_parameter_name(doc: &Document, label: &str) -> String {
    let mut base = String::new();
    for c in label.trim_end_matches(':').trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            base.push(c);
        } else if !base.ends_with('_') {
            base.push('_');
        }
    }
    let mut base = base.trim_matches('_').to_string();
    if base.is_empty() {
        base = "value".to_string();
    }
    if base.starts_with(|c: char| c.is_ascii_digit()) {
        base.insert(0, '_');
    }
    let taken = |name: &str| doc.parameters.values().any(|p| p.name == name);
    if !taken(&base) {
        return base;
    }
    (2..).map(|n| format!("{base}_{n}")).find(|name| !taken(name)).expect("always terminates")
}

/// What to call a benchmark in the console and in the report: its folder's name where the file is
/// the suite's own `scene.ffc`, and the file's name otherwise.
///
/// A folder of benchmarks is `*/scene.ffc`, so every one of them is called `scene.ffc` and a
/// report naming them that identifies none. The folder is what the suite calls each one.
fn benchmark_name(path: &std::path::Path) -> String {
    if path.file_name().is_some_and(|n| n == "scene.ffc") {
        if let Some(folder) = path.parent().and_then(|p| p.file_name()) {
            return folder.to_string_lossy().into_owned();
        }
    }
    file_name(path)
}

/// A path's file name, for status lines. The full path is too long for the bar and the name
/// is what the user recognises.
fn file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// A cutaway view: everything past a plane is not drawn, and the cut face is shown solid.
///
/// **View state, not document state.** It changes what is on screen and nothing about the
/// part — the graph, the save and any export are untouched, and two windows on one document
/// could section it differently without either being wrong. That is why it lives here rather
/// than as a `Node`, and why there is no block for it in the list.
///
/// Kept as an origin and a rotation rather than as a normal and an offset, even though the
/// shader wants the latter, because the manipulator moves a *frame*: a stored normal beside a
/// rotation would be the same orientation written down twice, which is the failure this
/// codebase keeps meeting. The normal is derived, never stored.
#[derive(Clone, Copy, Debug)]
struct Section {
    on: bool,
    origin: Point3<f64>,
    rotation: Rotation3<f64>,
    /// Which half is kept. Turning a plane over is not a rotation — it is the same plane
    /// facing the other way — so spending 180 degrees of the frame on it would fight the
    /// manipulator for one control. Same reasoning as `Node::Plane`.
    flipped: bool,
}

impl Default for Section {
    fn default() -> Self {
        Self {
            on: false,
            origin: Point3::origin(),
            rotation: Rotation3::identity(),
            flipped: false,
        }
    }
}

impl Section {
    /// The outward normal: the half-space *past* this is what gets cut away.
    fn normal(&self) -> Vector3<f64> {
        let facing = if self.flipped { -1.0 } else { 1.0 };
        self.rotation * Vector3::z() * facing
    }

    /// This section, with a manipulator motion applied about `pivot`.
    ///
    /// Scale is ignored rather than refused: a plane has no extent for a factor to act on, and
    /// the handles for it are not offered. Translation along the normal is what "move the
    /// section through the part" means, and rotation turns the frame the normal is read from.
    fn with(self, motion: gizmo::Motion, pivot: Point3<f64>) -> Self {
        match motion {
            gizmo::Motion::Translate(d) => Self { origin: self.origin + d, ..self },
            gizmo::Motion::Rotate { axis, degrees } => {
                let Some(unit) = nalgebra::Unit::try_new(axis, 1e-9) else { return self };
                let turn = Rotation3::from_axis_angle(&unit, degrees.to_radians());
                Self {
                    origin: turn * (self.origin - pivot.coords) + pivot.coords,
                    rotation: turn * self.rotation,
                    ..self
                }
            }
            gizmo::Motion::Scale { .. } => self,
        }
    }

    /// The plane as the shader wants it, or `None` when there is nothing to cut.
    fn half_space(&self) -> Option<(Vector3<f64>, f64)> {
        if !self.on {
            return None;
        }
        let n = self.normal();
        Some((n, n.dot(&self.origin.coords)))
    }
}

/// The command a block's eye issues to draw it or stop: `App::set_visible`'s, a function of the
/// graph so a test can run what the eye runs.
fn eye_command(graph: &Graph, id: NodeId, on: bool) -> SetVisible {
    SetVisible::new(id, eye_flag_in(graph, id), on)
}

/// One frozen point, with a manipulator motion applied about `pivot`.
///
/// The cage and curve half of what [`Placement::with`] does for a body: those targets have no
/// placement to compose with, so the motion is applied to the positions themselves.
fn moved_point(start: Point3<f64>, motion: gizmo::Motion, pivot: Point3<f64>) -> Point3<f64> {
    match motion {
        gizmo::Motion::Translate(d) => start + d,
        gizmo::Motion::Rotate { axis, degrees } => {
            match nalgebra::Unit::try_new(axis, 1e-9) {
                Some(unit) => {
                    let turn = Rotation3::from_axis_angle(&unit, degrees.to_radians());
                    pivot + turn * (start - pivot)
                }
                None => start,
            }
        }
        // A named axis scales along that axis alone, leaving the two perpendicular components
        // untouched; the uniform handle scales the whole offset. Unlike a body, a cage
        // selection may do either — moving vertices cannot break the distance field, because
        // there is no field being wrapped.
        gizmo::Motion::Scale { axis: Some(a), factor } => {
            let offset = start - pivot;
            let along = offset.dot(&a);
            pivot + offset + a * (along * (factor - 1.0))
        }
        gizmo::Motion::Scale { axis: None, factor } => pivot + (start - pivot) * factor,
    }
}

/// Intersection of a ray with a plane (point + normal), or `None` if the ray is parallel to
/// the plane.
fn ray_plane_intersect(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    plane_point: Point3<f32>,
    plane_normal: Vector3<f32>,
) -> Option<Point3<f32>> {
    let denom = dir.dot(&plane_normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_point - origin).dot(&plane_normal) / denom;
    Some(origin + dir * t)
}

/// How near edge-on a drawing plane may be seen and still take a click.
///
/// The cosine of the angle between the ray and the plane's normal, so this is a ray within
/// about five degrees of the plane itself. Nearer than that, a plane half a metre away puts the
/// point six metres along the ray — `ray_plane_intersect` would happily do it — and the spec's
/// acceptance is that such a click is declined and told why, not placed far away.
const EDGE_ON: f32 = 0.087;

/// Where a click in the viewport lands in space.
///
/// Three answers, in an order that is the whole design (`sketcher-spec.md` §7):
///
/// 1. **A drawing plane, if one is set.** Points land on it even where the model is behind
///    them — that is what setting one means, and it is what keeps a profile drawn over a body
///    flat rather than pulled onto the body's surface. Seen edge-on it declines (`None`).
/// 2. **The surface under the cursor** otherwise: the field, the cage or a curve, by what the
///    mode draws, which is the rule the whole picking path follows.
/// 3. **The plane through the pivot facing the camera**, when the ray hits nothing. Predictable,
///    and in a view square to the model it is the view plane; the point is dragged from there.
///
/// Free of the application so it can be tested without a GPU: `surface` is whatever the caller
/// can hit-test, asked only when it is needed.
fn landing_point(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    plane: Option<(Point3<f32>, Vector3<f32>)>,
    surface: impl FnOnce() -> Option<Point3<f32>>,
    pivot: Point3<f32>,
    forward: Vector3<f32>,
) -> Option<Point3<f32>> {
    if let Some((point, normal)) = plane {
        if dir.dot(&normal).abs() < EDGE_ON {
            return None;
        }
        return ray_plane_intersect(origin, dir, point, normal);
    }
    if let Some(hit) = surface() {
        return Some(hit);
    }
    ray_plane_intersect(origin, dir, pivot, forward)
}

/// A small red cross, for the buttons that take something out.
///
/// **A red cross, and `U+00D7` rather than `U+2715`.** The multiplication X was not in the
/// font, so the button rendered as an empty box -- reported as a box inside a box. The times
/// sign is, and is already used elsewhere in this panel for the same job. Red because the
/// button removes something, which is the one class of button worth colouring.
fn remove_button(ui: &mut egui::Ui, tip: &str) -> bool {
    ui.add(
        egui::Button::new(
            egui::RichText::new("\u{d7}").color(egui::Color32::from_rgb(220, 80, 80)).strong(),
        )
        .small(),
    )
    .on_hover_text(tip)
    .clicked()
}

/// A toggle that looks like a button whether or not it is on.
///
/// **`Button::selectable` draws no frame at all when it is off.** It sets
/// `frame_when_inactive(selected)`, so an unselected toggle has no outline for the style's
/// `inactive.bg_stroke` to be drawn with — which is why setting that stroke did not give the
/// mode switches their border, and why a toggle still read as a label until the pointer crossed
/// it. Asked for twice: a button should look like a button before it is clicked.
///
/// One function rather than `.frame_when_inactive(true)` at twenty call sites, because the
/// twenty-first will be written without it.
fn mode_button(selected: bool, text: impl Into<egui::WidgetText>) -> egui::Button<'static> {
    egui::Button::selectable(selected, text.into()).frame_when_inactive(true)
}

/// Which of three numbers was dragged, and by what factor, for [`App::size_row`]'s lock.
///
/// Free of the app so the arithmetic can be asserted: an `App` needs a window and a device
/// before it will exist, which is the arrangement `operand_choices_in` and `starting_pair_in`
/// already use.
///
/// `None` when nothing moved, when more than one did — a frame that changed two is not a drag of
/// one and guessing which led would be a guess — or when the one that moved started at zero,
/// where every factor is zero and the ratio has no meaning to preserve.
fn leading_factor(before: [f64; 3], after: [f64; 3]) -> Option<(usize, f64)> {
    let mut moved = (0..3).filter(|&i| before[i] != after[i]);
    let led = moved.next()?;
    if moved.next().is_some() || before[led].abs() < 1e-9 {
        return None;
    }
    Some((led, after[led] / before[led]))
}

/// Where new curves are built, or the world if there is none.
///
/// **Application state, not document state.** It is where the next thing you draw goes — the same
/// category as which body is selected or where the camera is — and nothing saved refers to it.
/// `sketcher-spec.md` §4 is explicit that this is what keeps the whole feature off the file
/// format: the plane a curve *remembers* is a different thing, and is §6's stamp.
#[derive(Clone, Copy, PartialEq)]
enum DrawingPlane {
    /// Free placement, which is what every curve has had until now.
    None,
    /// One of the world planes, named by the axis it faces along.
    World { normal: Axis, offset_mm: f64 },
    /// A plane block in the document. Its origin and facing are read through `Frame::From`, so
    /// this is the same reading every aimed block already gets.
    Node(NodeId),
}

impl DrawingPlane {
    /// How this reads in the console and on the row.
    fn label(self, app: &App) -> String {
        match self {
            DrawingPlane::None => "the world".to_string(),
            DrawingPlane::World { normal, offset_mm } if offset_mm.abs() < 1e-9 => {
                format!("the {} plane", plane_word(normal))
            }
            DrawingPlane::World { normal, offset_mm } => {
                format!("the {} plane at {offset_mm:.3} mm", plane_word(normal))
            }
            DrawingPlane::Node(id) => app.body_label(id),
        }
    }
}

/// The [`Frame`] a drawing plane names, or `None` for the world.
///
/// Free rather than a method so it can be tested without an application around it — the same
/// reason `operand_choices_in` is one. An `App` needs a GPU device; this needs an enum.
fn drawing_frame_of(plane: DrawingPlane) -> Option<fieldform_scene::Frame> {
    match plane {
        DrawingPlane::None => None,
        DrawingPlane::World { normal, offset_mm } => {
            // The origin sits *along the normal*, so `at=25` on the XY plane means 25 mm up
            // rather than 25 mm sideways. That is the only reading of "offset a plane" anybody
            // means, and it is the one that makes `plane on=xy at=25` do what it looks like.
            let along = match normal {
                Axis::X => [offset_mm, 0.0, 0.0],
                Axis::Y => [0.0, offset_mm, 0.0],
                Axis::Z => [0.0, 0.0, offset_mm],
            };
            Some(fieldform_scene::Frame::at_along(
                normal,
                along.map(ParamRef::Literal),
            ))
        }
        DrawingPlane::Node(id) => Some(fieldform_scene::Frame::From { node: id }),
    }
}

/// The distance from a point to a line segment, in screen pixels.
///
/// A segment rather than an infinite line: the nearest point on a *line* through two ends can
/// lie well beyond both of them, which would make a curve pickable along the whole of its own
/// extension. Clamping the parameter to `[0, 1]` is the difference.
fn distance_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let length2 = ab.length_sq();
    // A segment of no length is a point, and the projection below would divide by zero.
    if length2 < 1e-12 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / length2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// How near the cursor an end has to be, in pixels, for a snap to take it.
///
/// **Measured on screen and generously.** A world-space reach means the same gesture snaps from
/// a metre away when zoomed out and demands surgery when zoomed in, which is not what "near it"
/// means to the person doing it — the target is what they can see. Bigger than the grab radius
/// for a handle, because a snap is a deliberate act with the modifier held: there is no other
/// thing the click could have meant.
const SNAP_PIXELS: f32 = 22.0;

/// How big the target drawn over a snappable end is, in pixels.
///
/// Matched to the reach above rather than chosen separately: the box is a promise about where
/// the snap will take, and a box smaller than the reach would understate it while a bigger one
/// would lie.
const SNAP_BOX_PIXELS: f32 = SNAP_PIXELS;

/// Whether setting a drawing plane should also turn the camera to look down it.
#[derive(Clone, Copy, PartialEq)]
enum Swing {
    /// The usual answer: choosing a plane from a menu or the console means "put me there".
    Yes,
    /// For the ViewCube, which has already turned the camera — and turned it to the face's own
    /// sign, which the plane's unsigned axis cannot express.
    No,
}

/// The world drawing plane a ViewCube region names, for the six that name one.
///
/// A face has exactly one non-zero component and that is its normal. An edge has two and a
/// corner three: they name a direction to look from, not a plane to draw on, so they get `None`
/// and keep orienting the camera and nothing else.
fn plane_of_face(region: CubeRegion) -> Option<DrawingPlane> {
    let nonzero: Vec<usize> = (0..3).filter(|&i| region.sign[i] != 0).collect();
    let [axis] = nonzero[..] else { return None };
    let normal = match axis {
        0 => Axis::X,
        1 => Axis::Y,
        _ => Axis::Z,
    };
    Some(DrawingPlane::World { normal, offset_mm: 0.0 })
}

/// Which way is up when looking squarely along `normal`.
///
/// Generalises `CubeRegion::up`, which answers the same question for the twenty-six cube
/// regions: looking down the world Z there is no sensible "up" along Z, so Y serves; from
/// anywhere else Z is up, which is what makes a front view look like a front view.
///
/// `an_arbitrary_plane_is_stood_up_the_way_a_cube_face_is` holds the two together, because two
/// rules for one question is how a plane picked from the menu ends up rolled differently from
/// the same plane picked off the ViewCube.
fn up_for_normal(normal: Vector3<f32>) -> Vector3<f32> {
    if normal.x.abs() < 1e-6 && normal.y.abs() < 1e-6 {
        Vector3::y()
    } else {
        Vector3::z()
    }
}

/// The plane facing along an axis, said the way a person names it.
///
/// The XY plane is the one whose *normal* is Z, which is the reading everybody means and nobody
/// says out loud — so it is written down once here rather than inverted at each call site.
fn plane_word(normal: Axis) -> &'static str {
    match normal {
        Axis::X => "YZ",
        Axis::Y => "XZ",
        Axis::Z => "XY",
    }
}

/// The axis a named plane faces along. The inverse of [`plane_word`], and deliberately beside it.
fn plane_normal(word: &str) -> Option<Axis> {
    match word.to_lowercase().as_str() {
        "yz" | "zy" => Some(Axis::X),
        "xz" | "zx" => Some(Axis::Y),
        "xy" | "yx" => Some(Axis::Z),
        _ => None,
    }
}

/// What a loft's panel says about the loft as built -- see `App::loft_readings` for why it is
/// kept rather than asked each frame.
#[derive(Clone, Debug)]
struct LoftReadings {
    /// What the field divides by to stay a bound; said as "reads N× short" past 1.05.
    stretch: f64,
    /// How far each section's plane is from square to the centerline where it sits, in degrees.
    tilts: Vec<f64>,
    /// The narrowest the solid gets between two sections as a share of the smaller, and
    /// which span -- `loft-spec.md` §6.
    neck: Option<(usize, f64)>,
    /// The centerline the loft built for itself, when it was given none, for the viewport to
    /// draw dashed.
    derived: Option<Vec<Point3<f32>>>,
}

/// What `loft`, `loft sections=a,b,c spine=d --g1` asks for, read against the document: the
/// sections by name in the order given -- or, with none named, every closed curve and every
/// closed loop in the document in list order -- the centerline by name or none, and the
/// smoothness.
///
/// **`sections=a,b,c` rather than the spec's bare `s1 s2 s3`**, because this console is
/// `key=value` and a bare word is reported rather than read (item 70's note made the same
/// call). A name that matches no curve is refused by name, not skipped: a loft of the wrong
/// sections is a solid that looks nearly right.
fn loft_asked(
    doc: &fieldform_scene::Document,
    loops: &[Vec<CurveId>],
    args: &Args,
) -> Result<(Vec<CurveId>, Option<CurveId>, fieldform_scene::LoftSmooth), String> {
    let by_name = |name: &str| -> Result<CurveId, String> {
        doc.curves
            .iter()
            .find(|(_, c)| c.name.eq_ignore_ascii_case(name.trim()))
            .map(|(id, _)| *id)
            .ok_or_else(|| format!("No curve is named {name:?}"))
    };
    // `spine=none` is the centerline built from the end sections, said out loud: the one word that
    // empties a slot everywhere else (`console_keys::NOTHING`), and the echo's spelling of a loft
    // with no spine of its own.
    let spine = match args.get("spine").and_then(Value::as_str) {
        Some(name) if name.trim().eq_ignore_ascii_case(console_keys::NOTHING) => None,
        Some(name) => Some(by_name(name)?),
        None => None,
    };
    let sections = match args.get("sections").and_then(Value::as_str) {
        Some(list) => list
            .split(',')
            .filter(|w| !w.trim().is_empty())
            .map(by_name)
            .collect::<Result<Vec<_>, _>>()?,
        None => {
            // A closed curve is its own section; a loop of several is one section, named by
            // the curve that heads it -- the key `profile_partition` gives the row.
            let mut found: Vec<CurveId> = doc
                .curves
                .iter()
                .filter(|(_, c)| c.is_closed())
                .map(|(id, _)| *id)
                .chain(loops.iter().filter_map(|l| l.first().copied()))
                .filter(|id| Some(*id) != spine)
                .collect();
            found.sort_by_key(|id| id.raw());
            found.dedup();
            found
        }
    };
    let flagged = |key: &str| args.get(key).is_some_and(|v| !matches!(v, Value::Bool(false)));
    // `smooth=g0|g1` is the field's own name (the project notes); `--g1` is the flag this was written
    // with and still reads, so a script already using it keeps working.
    let smooth = match args.get("smooth").map(console_keys::loft_smooth_named).transpose()? {
        Some(smooth) => smooth,
        None if flagged("g1") => fieldform_scene::LoftSmooth::G1,
        None => fieldform_scene::LoftSmooth::G0,
    };
    Ok((sections, spine, smooth))
}

/// An expression being typed for a parameter, kept until it is accepted.
///
/// **Committed on Enter or on leaving the field, not per keystroke.** A parse error reported
/// while `radius *` is half typed is noise; the same error after Enter is the answer. And a
/// text that failed stays here with its error beside it rather than being replaced by what
/// the document still holds — losing what was typed to a typo would be the worse outcome.
struct ExpressionDraft {
    text: String,
    /// Why the last attempt was refused, shown under the row until the next one.
    error: Option<String>,
    /// Take the keyboard on the next frame: set when `=` opens the field, so typing can start
    /// without a second click.
    focus: bool,
}

/// A parameter's name while it is being typed, and why the last attempt to apply it was
/// refused.
///
/// **Applied on Enter or on leaving the field, never per keystroke.** The rows are sorted by
/// name, so renaming on each character moved the row under the cursor; the field's identity
/// was its position in the list, so the keyboard stayed where the row had been and the next
/// character went into whichever row had moved there -- "the text from the field below
/// populates the field above". And a name typed one character at a time passes through
/// every prefix of itself, so `cap1` beside an existing `cap` was refused at `cap` and came
/// out as `ca1`. Kept until it is applied or abandoned, with the refusal under the row.
struct NameDraft {
    text: String,
    error: Option<String>,
}

/// A body's placement, as the manipulator thinks of it.
///
/// The same three fields [`Node::Transform`] carries, in a form that composes: reading them
/// out, applying a motion and writing them back is easier to get right than editing three
/// arrays of parameter references in place.
#[derive(Copy, Clone, Debug)]
struct Placement {
    translation: Vector3<f64>,
    rotation: Rotation3<f64>,
    scale: f64,
}

impl Placement {
    fn identity() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: Rotation3::identity(),
            scale: 1.0,
        }
    }

    /// Reads a placement node, resolving its parameters. Anything else reads as the identity,
    /// which is exactly right: an unplaced body sits where it was built.
    fn of(app: &App, node: Option<&Node>) -> Self {
        let Some(Node::Transform { translation, rotation, scale, .. }) = node else {
            return Self::identity();
        };
        let r = app.resolve_point(rotation);
        Self {
            translation: app.resolve_point(translation).coords,
            rotation: Rotation3::from_euler_angles(
                r.x.to_radians(),
                r.y.to_radians(),
                r.z.to_radians(),
            ),
            scale: app.resolve(*scale).max(1e-6),
        }
    }

    /// The three fields, as the literals a placement node stores.
    fn into_node(self, input: impl Into<fieldform_scene::Operand>) -> Node {
        let (rx, ry, rz) = self.rotation.euler_angles();
        Node::Transform {
            // Whatever the caller holds: a manipulator wrapping a body has an id, and one
            // re-wrapping a block that is already placed has that block's slot, which may be
            // empty. Neither is this function's business to open.
            input: input.into(),
            translation: [
                ParamRef::Literal(self.translation.x),
                ParamRef::Literal(self.translation.y),
                ParamRef::Literal(self.translation.z),
            ],
            rotation: [
                ParamRef::Literal(rx.to_degrees()),
                ParamRef::Literal(ry.to_degrees()),
                ParamRef::Literal(rz.to_degrees()),
            ],
            scale: ParamRef::Literal(self.scale),
            about: fieldform_scene::Place::origin(),
        }
    }

    /// This placement with a manipulator motion applied about `pivot`.
    ///
    /// The pivot is the body's bounding-box centre, not its own origin, so every motion has to
    /// move the translation as well as the field it names. Rotating without that would swing
    /// the body around the part's origin instead of turning it in place — the single most
    /// obvious way for a gizmo to feel broken.
    fn with(self, motion: gizmo::Motion, pivot: Point3<f64>) -> Self {
        match motion {
            gizmo::Motion::Translate(d) => Self {
                translation: self.translation + d,
                ..self
            },
            gizmo::Motion::Rotate { axis, degrees } => {
                let Some(unit) = nalgebra::Unit::try_new(axis, 1e-9) else { return self };
                let turn = Rotation3::from_axis_angle(&unit, degrees.to_radians());
                Self {
                    translation: turn * (self.translation - pivot.coords) + pivot.coords,
                    rotation: turn * self.rotation,
                    scale: self.scale,
                }
            }
            // A body only ever receives the uniform factor — the per-axis cubes are not
            // offered for one, because a non-uniform scale would stop it being a distance
            // field. See `Handles::per_axis_scale`.
            gizmo::Motion::Scale { factor, .. } => Self {
                translation: (self.translation - pivot.coords) * factor + pivot.coords,
                rotation: self.rotation,
                scale: self.scale * factor,
            },
        }
    }
}

/// What a live manipulator drag is editing, frozen at mouse-down.
///
/// The two layers the architecture separates, as two variants: a cage selection is a direct
/// edit that reverses against exact prior positions, and a body is a parametric placement that
/// reverses by restoring the graph. One widget, two commit paths, and they cannot be confused
/// because the target is decided before the first frame of the drag.
enum GizmoTarget {
    Cage {
        /// The cage or a surface. Named here rather than read from `App::control` each frame,
        /// because a cage *body* is moved through this arm in Field mode while a surface may
        /// be the control mesh — the two are different meshes and the drag must not swap them.
        mesh: ControlMesh,
        vertices: Vec<VertexId>,
        before: Vec<Point3<f64>>,
    },
    Body {
        /// The placement node. Created at drag start if the body had none, so that every
        /// later frame changes numbers rather than structure — which is the difference
        /// between a buffer write and a shader recompile, once per drag instead of per frame.
        placement: NodeId,
        before_graph: Graph,
        before: Placement,
    },
    /// The section plane. Reverses against where it was, like the direct layer's other
    /// targets, because a section is not in the document and has no parameter to name.
    Section { before: Section },
    /// Control points of one curve.
    ///
    /// A curve is a parametric-layer object — it is serialized, and `Node::Pipe` reads it —
    /// but its control points are internal state the way a cage's vertex array is. So this is
    /// the direct layer's shape, reversing against exact prior positions, rather than a named
    /// parameter for each point.
    Curve {
        curve: CurveId,
        indices: Vec<usize>,
        before: Vec<Point3<f64>>,
        gesture: CurveGesture,
    },
    /// A profile row: every curve of the loop, moved as one piece.
    ///
    /// Selecting the row selects its curves, and a selection has a manipulator. Each curve
    /// moves through the same gesture a single one does, so a held corner moves its block
    /// once however many curves sit on it, and the lot commits as one entry.
    Profile {
        moves: Vec<CurveMove>,
        /// The graph before any block moved, taken once for the whole row.
        graph_before: Option<Graph>,
        shapes_field: bool,
    },
    /// A block's own XYZ, moved directly rather than wrapped in a placement.
    ///
    /// **Because a block that already says where it is should say it once.** A sphere carries a
    /// centre and a cuboid carries a centre; dragging one used to wrap it in a `Node::Transform`
    /// and write the motion there, so the block's own coordinates stayed at whatever they were
    /// built with and the number on screen no longer described where the shape was. Two places
    /// held one fact and only one of them moved.
    ///
    /// Translation only. A rotation or a scale is not something a centre can express, so those
    /// still wrap -- see `begin_gizmo`, which decides from the handle that was grabbed.
    Anchors {
        node: NodeId,
        before: Vec<Point3<f64>>,
        /// The block's own turn as it was, for the blocks that carry one.
        ///
        /// A rotate writes here for the same reason a translate writes the centre: the block
        /// says which way it faces, so the gesture should move the number the block owns rather
        /// than wrap a second one around it.
        before_turn: Option<Rotation3<f64>>,
        before_graph: Graph,
    },
    /// An extrude's depth, pulled by the arrow on its far face.
    ///
    /// **One number, written as a literal, and refused when the number is not one.** The depth is a
    /// uniform, so a drag is a buffer write and never a recompile. If a variable or a ramp drives it,
    /// the drag never begins: overwriting the expression with a literal would be the app deciding that
    /// the variable was in the way.
    Depth {
        node: NodeId,
        /// `0` is side one, `1` the second side of an extrude that has one.
        side: usize,
        /// Whether the extrude runs half each way, where the face being pulled stands at half the
        /// depth and so moves the depth by twice as much.
        symmetric: bool,
        /// The depth as it was at mouse-down, in millimetres.
        was: f64,
        /// The way a positive depth runs, a unit vector, which the motion is read along.
        direction: Vector3<f64>,
        before_graph: Graph,
    },
}

/// A manipulator drag in progress.
struct GizmoDrag {
    drag: gizmo::Drag,
    target: GizmoTarget,
    /// The pivot the gizmo sat on when the drag began. Every motion is about this, and it must
    /// not follow the body it is moving.
    pivot: Point3<f64>,
    /// What to show beside the gizmo, refreshed each frame.
    readout: String,
}

/// Frozen state for an in-progress move: the selected vertices' starting positions, and the
/// plane (through their centroid, facing the camera) that screen-space drag motion is
/// projected onto to produce a 3D translation.
struct MoveState {
    /// Whose vertices these are, fixed at the press so the gesture commits to the mesh it began
    /// on whatever else happens during it.
    mesh: ControlMesh,
    /// Document positions (f64), not the f32 the drag math runs in: these are written back
    /// to the cage and handed to the undo command, so they must be the exact prior values.
    start_positions: Vec<(VertexId, Point3<f64>)>,
    plane_point: Point3<f32>,
    plane_normal: Vector3<f32>,
    /// Offset between the plane's true center and where the initial click actually landed
    /// on the plane, so the dragged geometry doesn't jump to re-align with the cursor the
    /// instant the drag starts.
    click_offset: Vector3<f32>,
}

/// A control point being dragged directly, without the manipulator.
///
/// **The manipulator cannot do this job in the view where it is most wanted.** Looking straight
/// down a plane — which is the whole point of picking one to draw on — the axis toward the camera
/// is a dot and the two plane handles are the only things that work, so placing a point meant
/// aiming at a handle rather than at the point. Dragging the point itself is what everybody
/// expects of a control point and is what makes the endpoint snap reachable, since a snap needs
/// a drag to end.
struct PointDrag {
    curve: CurveId,
    /// Every point moving, so a multi-point selection drags together the way the cage's does.
    indices: Vec<usize>,
    start_positions: Vec<Point3<f64>>,
    plane_point: Point3<f32>,
    plane_normal: Vector3<f32>,
    /// Where on the plane the click actually landed, relative to the point, so the geometry
    /// does not jump to meet the cursor the instant the drag starts.
    click_offset: Vector3<f32>,
    /// The curve whose end this is currently snapped onto, while it is.
    ///
    /// Live rather than decided at the end: a snap you cannot see until you let go is one you
    /// cannot aim, and the whole difficulty was not knowing whether it had taken.
    snapped_to: Option<CurveId>,
    /// What the gesture needs to know about the points it moves. See [`CurveGesture`].
    gesture: CurveGesture,
    /// Points picked on other curves, moving by the same offset and committed with these as
    /// one entry.
    others: Vec<CurveMove>,
}

/// One curve's part in a gesture that moves several: which of its points, where they started,
/// and what the gesture knows about them.
#[derive(Clone)]
struct CurveMove {
    curve: CurveId,
    indices: Vec<usize>,
    before: Vec<Point3<f64>>,
    gesture: CurveGesture,
}

/// What a gesture on a curve's control points knows before it starts, whichever hand is on
/// them: the point drag and the manipulator share this, so a held end behaves the same way
/// under both.
///
/// **A dragged point that sits on a block moves the block, not the control point**
/// (`profile-group-spec.md` §4.3). The point is where the end is, so moving either moves both,
/// and every other end on that block follows through `settle_curves_on_points`. Writing the
/// control point instead was what phase 2 found in the point drag -- the end moved, and snapped
/// straight back to its block on release -- and what the owner then found in the manipulator,
/// which had its own copy of the same mistake.
#[derive(Clone)]
struct CurveGesture {
    /// The moving points that sit on blocks, with the block each sits on.
    bound: Vec<(usize, NodeId)>,
    /// The graph before any block moved, so the gesture lands on the history as one edit from
    /// where it started. `None` when nothing moving sits on a block.
    graph_before: Option<Graph>,
    /// Whether a block that moves has a body reading it, in which case the traced view is
    /// rebuilt per frame so the body follows as the curve does.
    bodies_follow: bool,
    /// Whether moving these points changes any field at all -- the curve is read by a block,
    /// or a block the ends sit on is, or another curve on those blocks is. When nothing is,
    /// the traced view is left exactly as it was: dropping it to draft on every frame and
    /// re-settling it on release, for a picture that could not have changed, is what made
    /// editing a loose curve beside an extrude feel slow.
    shapes_field: bool,
}

/// A curve being placed by clicks: the start, then the end (`sketcher-spec.md` §7).
///
/// **A gesture in progress, not a mode.** It is the same category as a drag: it shows itself
/// while it waits (the start is drawn, a dashed run follows the cursor, the console says what
/// the next click does), it ends by itself after the second click, and Escape cancels only
/// this. It never becomes "keep placing until you stop" — that would be a tool that changes
/// what every click means until something is remembered, which is the thing the application
/// does not have. Between the two clicks the view still turns, because a drag is still a drag;
/// only the press-time picks stand aside, so a click cannot both place a point and select.
///
/// Where a click lands is decided by [`landing_point`], never by which view you happen to be
/// in. The proposal that placing should switch on when the view is square to a plane was made
/// and answered (`sketcher-spec.md` §5.1): the plane is the state, set on purpose and visible,
/// and the view is something you change for other reasons.
/// What ending the curve tool says, given what it had done (item 111).
///
/// Three cases, because the tool now stays on: a curve half-placed is lost; curves already placed
/// are kept, and saying otherwise is the one thing that must not happen; and nothing at all.
fn curve_tool_ended(pending: &PendingCurve, how: &str) -> String {
    let kept = match pending.placed {
        0 => None,
        1 => Some("the curve placed is kept".to_string()),
        n => Some(format!("the {n} curves placed are kept")),
    };
    match (pending.first.is_some(), kept) {
        (true, Some(kept)) => format!("{how} — the one in progress was not drawn; {kept}"),
        (true, None) => format!("{how} — nothing was drawn"),
        (false, Some(kept)) => format!("{how} — {kept}"),
        (false, None) => format!("{how} — nothing was drawn"),
    }
}

struct PendingCurve {
    degree: usize,
    /// The start, once it has been clicked. `None` while the first click is still awaited.
    first: Option<Point3<f64>>,
    /// Where the next click would land, for the preview. See [`Landing`].
    preview: Option<Landing>,
    /// Set when what is being placed is an axis rather than a line: it is named `Axis n`, and a
    /// normal-to-surface one takes a single click on the field's surface (`split_axis`).
    axis: Option<AxisKind>,
    /// The axis being stood somewhere new, when this click is re-pointing one rather than making
    /// one. `None` makes a new axis, which is what the Create menu does.
    ///
    /// **Here rather than beside it on the `App`**, so the target cannot outlive the gesture that
    /// set it: one object is begun, finished and cancelled, and `Esc` clearing `pending_curve`
    /// clears this with it.
    restanding: Option<CurveId>,
    /// Curves placed since the tool was begun, which the tool now stays on across (item 111).
    ///
    /// **So every way out says something true.** Before the tool stayed on, ending it always meant
    /// abandoning the one curve in progress, and every exit said "nothing was drawn". Once four
    /// lines can be placed in one session, that sentence is false at exactly the moment someone
    /// reads it to check their work survived.
    placed: u32,
}

/// Where the next click would land, kept against the cursor and camera it was computed for.
///
/// A cache rather than a per-frame computation because the landing can be a field trace, and
/// tracing sixty times a second for a cursor that has not moved is a cost with no reader.
struct Landing {
    cursor: egui::Pos2,
    eye: Point3<f32>,
    snapping: bool,
    grid: bool,
    at: Point3<f64>,
}

/// The STEP reader's own parameters, wherever they are being edited.
///
/// Shared between the import dialog and an import block, because they are the same settings:
/// the dialog says how to read a file, and the block says how one *was* read and re-reads it.
/// Two copies of these controls would drift, and the pair that drifted would disagree about
/// what a document had been built from.
fn step_option_controls(ui: &mut egui::Ui, o: &mut fieldform_io::StepOptions) {
        // Presets first, as buttons that *fill in* the fields below rather than
        // as a mode of their own. A preset that were its own state would be a
        // second description of the settings, and the two would disagree the
        // moment a number was nudged.
        ui.horizontal(|ui| {
            ui.label("Preset:");
            let current = o.preset_name();
            for (name, preset) in fieldform_io::StepOptions::presets() {
                if ui.selectable_label(current == Some(name), name).clicked() {
                    *o = preset;
                }
            }
        })
        .response
        .on_hover_text(
            "Fills in the settings below. Change any of them and no preset is \
             shown, because none of them describes what you have.",
        );
        ui.separator();

        let o = &mut *o;
        ui.horizontal(|ui| {
            ui.label("Linear deflection:");
            ui.add(
                egui::DragValue::new(&mut o.linear_deflection_mm)
                    .speed(0.005)
                    .range(0.001..=10.0)
                    .suffix(" mm"),
            );
        })
        .response
        .on_hover_text(
            "How far a facet may sit from the true surface. Finer costs \
             triangles and time.",
        );
        ui.horizontal(|ui| {
            ui.label("Angular deflection:");
            ui.add(
                egui::DragValue::new(&mut o.angular_deflection_deg)
                    .speed(0.25)
                    .range(3.0..=60.0)
                    .suffix("\u{b0}"),
            );
        })
        .response
        .on_hover_text(
            "How far a facet's normal may turn from its neighbour's. This is \
             what faceting on a curved face actually is — OCCT's own \
             default is 28.6\u{b0}. Below about 3\u{b0} the mesher starts \
             emitting slivers and the result gets worse, not better.",
        );
        ui.horizontal(|ui| {
            ui.label("Sewing tolerance:");
            ui.add(
                egui::DragValue::new(&mut o.sew_tolerance_mm)
                    .speed(0.005)
                    .range(0.0..=5.0)
                    .suffix(" mm"),
            );
        })
        .response
        .on_hover_text(
            "Joins faces that meet but are not joined, which is what makes an \
             exported part a solid. Zero takes the file at its word. Too large \
             welds genuinely separate faces together and can lose a thin wall.",
        );
        ui.checkbox(&mut o.shape_fix, "Repair before sewing")
            .on_hover_text(
                "Runs OCCT's general repair pass first — wires out of \
                 order, edges that do not meet their curves, faces the wrong \
                 way round. A different kind of damage from an unsewn seam.",
            );
        ui.checkbox(&mut o.parallel, "Mesh faces in parallel")
            .on_hover_text(
                "Faces tessellate independently, so this is close to free. The \
                 first thing to turn off if one particular file misbehaves.",
            );
}

/// How coarsely `verify` looks for a step in the field, per axis.
///
/// **A detector's reach, chosen by what a person will wait for.** 64 steps is about 800 000
/// readings of the composed field — a second or two on a cage, longer on a lattice. It does not
/// have to resolve a step, only to land a sample pair across one: the refinement that follows
/// does the finding, and it is what tells a step from the creases a boolean is full of. A fault
/// that fits entirely between two samples is still missed, which is why the spacing is reported
/// with a clean answer rather than calling it proof.
const PROMISE_SAMPLES: usize = 64;

struct App {
    camera: OrbitCamera,
    /// The interface's context, for what an operation does to the window rather than the
    /// document: Exit closes it from `dispatch`, which has no `Ui` to ask.
    egui_ctx: egui::Context,
    /// Where the echo last wrote up to (`app::echoing`), and every line it has written, in order:
    /// the record T1 replays (`interaction-spec.md` §6.3).
    echo_base: app::echoing::EchoBase,
    echo_log: Vec<String>,
    /// The tool session pending, if one is (`app::sessions`), and what each command was last kept
    /// with this run of the application, which the next session of it starts from (§3.2).
    session: Option<app::sessions::Session>,
    last_values: std::collections::HashMap<&'static str, Args>,
    /// The command palette while it is open, and the commands run lately, most recent first,
    /// which it ranks by (`interaction-spec.md` §5).
    palette: Option<app::palette_ui::PaletteState>,
    recent: Vec<&'static str>,
    /// The person's own bindings over the registry's defaults (`keybindings`), what collides, the
    /// command waiting for keys to be pressed, and what the last rebind said (§9.2).
    bindings: Vec<keybindings::Line>,
    binding_conflicts: Vec<keybindings::Conflict>,
    capturing: Option<&'static str>,
    bindings_note: Option<String>,
    /// The marking menu while it is open (§7), and the last command kept or run, which Shift+R
    /// runs again (§8).
    marking: Option<app::marking_ui::MarkingMenu>,
    last_command: Option<&'static commands::Command>,
    /// The shelf's commands (§10.2), where each was drawn last frame, and how many times the
    /// palette has been opened, which the first-run hint goes by (§10.3).
    shelf: Vec<&'static str>,
    shelf_drawn: Vec<(&'static str, egui::Rect)>,
    /// Where the viewport was drawn last frame, so the tool panel can keep clear of the block it is
    /// making (`interaction-spec.md` §4.1).
    viewport_rect: Option<egui::Rect>,
    /// Whether a pending session shows the floating tool panel. **Off by default, at the owner's
    /// asking (2026-10-17):** the panel and the block's own rows in the notebook said the same
    /// numbers twice, so a pending block now opens in the notebook with its first number waiting
    /// for the keyboard, and the panel is View › Floating tool panel for whoever wants it back.
    floating_panel: bool,
    /// Set while a pending block's rows are being drawn for the first time: the first number field
    /// drawn takes the keyboard, and clears it (`number_field`).
    focus_first_number: bool,
    /// Set while the pending block's rows are being drawn, so a number confirmed there with Enter
    /// can say the block is done (`keep_pending`).
    drawing_pending: bool,
    /// Enter confirmed a number of the pending block: it is kept at the end of the frame, once the
    /// number has been written.
    keep_pending: bool,
    /// The pending block's number fields as last drawn, in order, for Tab to move between.
    pending_numbers: Vec<egui::Id>,
    palette_opens: u64,
    /// The document every edit goes through — see `fieldform-scene`.
    doc: Document,
    /// Which cage the UI is currently editing.
    ///
    /// The document model supports several parts and cages; the v1 interface edits one at a
    /// time, so the app tracks the active one rather than the model pretending there is only
    /// ever one. Adding a part list later means changing this field, not the document.
    active_cage: CageId,
    /// Which quad mesh the cage tools are editing — the cage above, or a surface. See
    /// [`ControlMesh`]; read it through `control` rather than reaching for the cage.
    control: ControlMesh,
    /// Which part the UI is composing. Its graph is the composition — see `part_graph`.
    active_part: PartId,
    /// Where this document was opened from or last saved to, and the archive it came out of.
    ///
    /// The archive is kept, not just the path, because it carries the entries this build does
    /// not interpret — a plugin's `ext/` data, the `sources/` an import came from. Saving
    /// through it preserves them; saving a freshly built one would drop exactly what the
    /// container exists to protect.
    file: Option<(std::path::PathBuf, fieldform_io::Archive)>,
    /// How many times the document has changed, and what that count was when it was last
    /// written or opened. Unequal means there is something in memory the file does not have.
    ///
    /// A counter rather than a comparison against the saved bytes, because the marker is
    /// wanted every frame and serializing a hundred-thousand-vertex cage per frame is not a
    /// price a title bar can pay. It is conservative in one direction only: undoing back to
    /// the state that was saved still counts as changed, so the app can offer to save
    /// something identical to what is on disk. It can never call a changed document clean,
    /// which is the direction that loses work.
    revision: u64,
    saved_revision: u64,
    /// Moves whenever the document may have, recorded or not: on every edit
    /// [`Self::mark_changed`] records, every write to the part's graph or to a control mesh, every
    /// change to a curve, and every rebuild of what is drawn, which is what opening a file or
    /// starting a new one ends with. What the overlay's readings and the block list's are kept
    /// against ([`Self::overlay_readings`], [`Self::list_readings`]).
    ///
    /// **Not `revision`**, which answers whether the file is behind and so stands still exactly
    /// where this has to move: a drag writes the document every frame and is recorded once, when
    /// the hand comes off, and opening a file replaces all of it and leaves nothing unsaved.
    document_moves: u64,
    /// The overlay's readings as of [`Self::document_moves`] and the view they were read for. In a
    /// cell because the overlay, picking and the trace all read them through `&self`, and
    /// whichever reads first after something moves is the one that keeps them.
    kept_readings: std::cell::RefCell<fieldform_field::KeptReadings<OverlayKey>>,
    /// The block list's readings as of [`Self::document_moves`] and the part they were read off,
    /// in a cell for the same reason.
    kept_list: std::cell::RefCell<
        fieldform_field::KeptReadings<ListKey, fieldform_scene::ListReadings>,
    >,
    /// What the window title was last set to, so it is only sent when it actually changes.
    title: String,
    /// The splash, until it is dismissed.
    ///
    /// Decoded once, on the first frame that needs it, rather than at startup: it is a
    /// two-megapixel texture that most sessions look at for a second and never again, and
    /// uploading it before the window is up would delay the window.
    splash: Option<egui::TextureHandle>,
    splash_shown: bool,
    /// The two Help windows. Windows rather than menu text: the shortcut list is six lines
    /// that want to stay open while you try them, which a menu that closes on click cannot do.
    show_shortcuts: bool,
    show_about: bool,
    /// The console: what is being typed, and what came back.
    ///
    /// Every line goes through the same named dispatch a menu item does, so a command and a
    /// click cannot mean different things. That is the whole reason the app has had named
    /// operations from the start.
    console_input: String,
    console_log: Vec<ConsoleLine>,
    /// Where new curves are built. See [`DrawingPlane`].
    drawing_plane: DrawingPlane,
    /// How many things have been reported, ever. Not `console_log.len()`, which stops growing
    /// when a line repeats and is capped besides — see [`App::say`].
    console_reports: u64,
    /// The viewport error already reported, so a standing one is not reported again next frame.
    traced_error_said: Option<String>,
    /// Where the console is being mirrored, and how much of it is there.
    console_file: Option<std::path::PathBuf>,
    console_file_bytes: u64,
    started: std::time::Instant,
    /// What has been typed before, newest last, walked with the up and down arrows.
    console_history: Vec<String>,
    console_history_at: Option<usize>,
    /// Whether the unsaved-changes prompt has already been answered in favour of closing.
    ///
    /// Discarding does not make the document clean, so without this the re-issued close asks
    /// again, and again. See `guard_close`.
    closing: bool,
    stack: CommandStack,
    registry: OpRegistry,
    keymap: Keymap,
    /// The control cage, triangulated — what polygon mode draws and what picking hits.
    cage_mesh: CageMesh,
    /// Imported geometry, as triangles ready to draw. See `rebuild_imported_mesh`.
    imported_mesh: CageMesh,
    wgpu_render_state: egui_wgpu::RenderState,
    /// What a long-running job is doing *right now*, shown as the console's last line and
    /// replaced in place rather than appended.
    ///
    /// Separate from the log because it is not an event. An export ticking its elapsed seconds
    /// would otherwise write a line a second, and the history the console exists to keep would
    /// be a column of near-identical stopwatch readings. What it says when it *finishes* is an
    /// event, and goes through `say` like everything else.
    console_progress: Option<String>,
    view_mode: ViewMode,
    /// What each named stage cost this frame, for [`Self::note_frame_cost`] to name when the
    /// frame was slow enough to be felt. Cleared every frame.
    stages: Vec<(&'static str, f32)>,
    selection: Picked,
    hover: Selection,
    move_state: Option<MoveState>,
    /// A control point being dragged directly. See [`PointDrag`].
    point_drag: Option<PointDrag>,
    /// A curve waiting for its clicks. See [`PendingCurve`].
    pending_curve: Option<PendingCurve>,
    /// Why an axis on a surface cannot stand where it says, by curve, when its ray no longer meets
    /// its body; shown in its block. Derived on every rebuild, never saved (`split_axis`).
    surface_axis_notes: std::collections::HashMap<CurveId, String>,
    /// An axis's record as it was when a drag of its U, V or length began, so that the drag lands on
    /// the history as one entry when it ends.
    surface_axis_edit_before: Option<(CurveId, fieldform_scene::SurfaceAxis)>,
    /// The manipulator drag in progress, if any.
    gizmo_drag: Option<GizmoDrag>,
    /// The handle under the cursor, for the hover highlight.
    gizmo_hover: Option<gizmo::Handle>,
    /// Whether the ViewCube is shown. Presentation state — it is not saved with the document
    /// and nothing about it is undoable, for the same reason the camera is not: the view is
    /// how you are looking at the model, not part of it.
    show_view_cube: bool,
    /// The block whose name is being typed on its own row, and what has been typed so far.
    ///
    /// **A name is edited where it is read.** It used to be a field inside the block, so naming
    /// a sphere meant opening it, finding the row, typing, and closing it again — four actions
    /// for a label, and the label was already on screen the whole time. Double-clicking the row
    /// edits it in place.
    ///
    /// Presentation state, like the camera: which row is mid-edit is not part of the document,
    /// and a half-typed name is not a name until it is committed. The buffer lives here rather
    /// than in the node so that Escape can abandon it without having written anything.
    renaming: Option<(RowKey, String)>,
    /// The reference field being typed in, and what is in it so far.
    ///
    /// Presentation state for the reason `renaming` is: half a block's name is not a reference, so
    /// nothing is written to the graph until the text resolves. One slot rather than one per field,
    /// because only the focused field is being typed in — and a field that loses focus with text
    /// that resolves to nothing puts the settled name back, so there is nothing to keep.
    ref_edit: Option<(egui::Id, String)>,
    /// A parameter whose name field should take the keyboard on the next frame.
    ///
    /// **Set where a variable is *made* from a field, not where one is edited.** Naming a
    /// value from the V menu invents a name from the field's label -- an X becomes `x`, then
    /// `x_2` -- and the only place to change it was the variables panel, which means finding
    /// the row you just made among the ones you did not. Reported as "when i name the x y z
    /// of a point as a VAR - i need to be able to give it a name". The same arrangement
    /// `ExpressionDraft::focus` already uses for the expression field, for the same reason:
    /// the next thing typed is the thing that was missing.
    naming_parameter: Option<ParamId>,
    /// Names being typed, by parameter. See [`NameDraft`].
    name_drafts: std::collections::HashMap<ParamId, NameDraft>,
    /// The boxes whose three sides are being kept in proportion while they are edited.
    ///
    /// **An editing aid, not part of the shape, and deliberately not saved.** The three sides
    /// of a box are three independent numbers — that is what the node carries and what a file
    /// records. A lock stored *in* the document would be a fourth thing describing the same
    /// three, and the two could disagree: a file written with a lock, opened and edited by
    /// anything that did not honour it, would come back with the ratio silently broken. The
    /// objection the old code raised against a link between the sides was exactly this, and it
    /// is answered by keeping the lock out here, where it can only ever affect the next drag.
    ///
    /// So it behaves like the camera: it is how you are working on the model, not part of it,
    /// and reopening a file starts unlocked.
    size_locked: std::collections::HashSet<NodeId>,
    clipboard: Option<Clipboard>,
    /// A Tab taken out of the raw input before egui could route it to focus, waiting to be
    /// dispatched through the keymap like any other shortcut. See [`App::raw_input_hook`].
    stolen_tab: Option<Modifiers>,
    /// A Tab pressed while the console field had the keyboard, waiting to complete a word.
    ///
    /// Taken in `raw_input_hook` for the same reason the mode cycle's is, and kept apart from
    /// it so that completing a command neither cycles the viewport nor walks the focus away
    /// from the field being typed in.
    console_complete: bool,
    /// How many of those reports were warnings, so a script can tell whether the line it just
    /// ran was refused. See [`Script`].
    console_warnings: usize,
    /// A script waiting to run, from `--run` on the command line or the `run` verb.
    script: Option<Script>,
    /// Whether a script's lines are being run this frame, so `run` inside one is refused.
    script_running: bool,
    /// Every `expect` this session, in order, for a script's report.
    expect_results: Vec<ExpectRecord>,
    /// A snapshot waiting for the view to finish its still. See [`PendingSnap`].
    pending_snap: Option<PendingSnap>,
    /// What a run has tallied so far, kept across the frames a snapshot waits for. See
    /// [`App::drive_script`].
    script_tally: Option<ScriptTally>,
    /// Whether the press that is currently down landed on a manipulator handle. Decided at the
    /// press for the same reason `press_on_selection` is: by the time egui calls it a drag,
    /// the pointer has left the handle and where the gesture *started* can no longer be
    /// asked.
    press_on_gizmo: bool,
    /// The same, for a control point: it claims the press so the click does not also land on
    /// the cage or the body drawn behind it.
    press_on_control_point: bool,
    /// Control points picked on curves other than `selected_curve`, so one selection spans
    /// curves: Shift+click adds across them, Ctrl+drag boxes them, and a drag from any picked
    /// point moves the lot as one entry. `selected_points` stays the selected curve's own,
    /// which is what every single-curve reader already means by it.
    other_points: Vec<(CurveId, usize)>,
    /// A press on a point already picked: the selection is kept so a drag moves all of it,
    /// and narrowed to that one point if the press turns out to be a click.
    pending_narrow: Option<(CurveId, usize)>,
    /// A box selection being dragged: where it started, and whether it adds (Ctrl+Shift).
    marquee: Option<(egui::Pos2, bool)>,
    /// The running press started a box, which the click handlers stand aside for.
    press_on_marquee: bool,
    /// Every visible curve's control points drawn and clickable, not only the selected one's.
    show_control_points: bool,
    /// Whether the press that is in flight landed on a curve.
    ///
    /// Needed because the two halves of a click are handled at different moments: a curve is
    /// picked on the *press*, and the body pick runs on the *release* and clears the selected
    /// curve as part of meaning "you clicked the model, not a curve". Without this the curve was
    /// selected and then unselected within one click, which looks exactly like the pick not
    /// working.
    press_on_curve: bool,
    /// Where a curve's points were before the numeric fields started being dragged, so the
    /// whole gesture lands on the history as one entry rather than one per frame.
    curve_edit_before: Option<(CurveId, Vec<Point3<f64>>)>,
    /// The same, for the weights: which points, and what they pulled with before the drag.
    weight_edit_before: Option<(CurveId, Vec<usize>, Vec<f64>)>,
    /// Curves in the document, tessellated once for drawing. Rebuilt on import, not per
    /// frame — a 0.05 mm polyline of a long spline is thousands of points.
    curve_display: Vec<(CurveId, Vec<Point3<f32>>)>,
    /// Moves every time [`Self::rebuild_curve_display`] runs, which is every time a curve
    /// changes; what the profile rows are cached against. Not the document's `revision`, which
    /// moves on every edit of anything and would re-walk the loops on each frame of a slider
    /// drag that touched no curve.
    curve_revision: u64,
    /// The profile rows, as of [`Self::curve_revision`]. See [`ProfileRows`].
    profiles: ProfileRows,
    /// What a loft's panel and the viewport read of it -- its bound, each section's tilt, the
    /// neck and the derived centerline -- as of one node at one [`Self::curve_revision`], or
    /// `None` inside for a loft that cannot be built. Building a loft walks and resamples
    /// every section and matches their corners, which a panel drawn every frame need not pay
    /// per frame; the sweep's panel gets away with building its leaf each frame because a
    /// sweep resamples nothing.
    loft_readings: Option<(NodeId, Node, u64, Option<LoftReadings>)>,
    /// Loops that closed while a gesture was in flight, by key, waiting to be held with
    /// joints once the hand comes off. See [`Self::refresh_profiles_with`].
    pending_joins: Vec<CurveId>,
    /// Where the cursor is over the viewport this frame, with the aspect and rect a pick
    /// needs, so Ctrl+V can land a curve under it (`profile-group-spec.md` §4.4). `None`
    /// whenever a menu, the console or a panel is under the pointer instead.
    viewport_hover: Option<(f32, egui::Rect, egui::Pos2)>,
    /// Where the viewport's context menu was opened, which is where "Paste here" pastes.
    viewport_menu_at: Option<egui::Pos2>,
    selected_curve: Option<CurveId>,
    /// Which of [`Self::selected_curve`]'s control points are picked, by index.
    ///
    /// Indices rather than ids, and that is the point rather than a shortcut. A control point
    /// has no stable identity and is not given one: naming N of them per curve is per-element
    /// identity by another name, which is what D1 declined to take on for cage vertices — the
    /// same argument one level up. Undo reverses against exact prior positions instead, so
    /// nothing needs to survive being renumbered.
    selected_points: Vec<usize>,
    /// Field generator settings, live even before anything is applied so the radius can be
    /// dialled in and then committed.
    pipe_radius: ParamRef,
    /// Which body the inspector is bound to, and which two the boolean tool would combine.
    ///
    /// Bodies are graph nodes, and a node id outlives the edits around it — unlike a cage
    /// vertex index, which is why selection here can be persistent state and selection down in
    /// the cage cannot.
    selected_body: Option<NodeId>,
    /// A selected block that is not a node — a tessellation, an export or an import — so the
    /// Delete key and a row's menu act on it the way they act on a body.
    selected_block: Option<outputs::BlockRef>,
    /// Whether the left button went down on the current selection, which is what makes the
    /// drag that may follow a move rather than a tumble.
    ///
    /// Answered at the press and remembered, because by the time egui calls it a drag the
    /// pointer has moved and where the gesture started can no longer be asked.
    press_on_selection: bool,
    /// Curves the viewport leaves out. Same idea as `hidden`, and drawn from the same eye.
    hidden_curves: std::collections::HashSet<CurveId>,
    hidden_surfaces: std::collections::HashSet<fieldform_scene::QuadMeshId>,
    /// Each surface's edges in world space, for the overlay — rebuilt when the document's
    /// surfaces change: on import, on load, and whenever one of their points moves.
    surface_display: Vec<(fieldform_scene::QuadMeshId, Vec<[Point3<f32>; 2]>)>,
    /// What `surface_display` was built from: each surface's id and `geometry_fingerprint`.
    surface_prints: Vec<(fieldform_scene::QuadMeshId, u64)>,

    // `hidden` used to live here as a set of node ids. It is a property of the *document* now,
    // in the one metadata slot a body has, reached through `is_visible` and `set_visible`.
    // It was app state on the reasoning that hiding is a way of looking at a part rather than a
    // fact about it — true of the camera, and not of this: hiding half an assembly to work on
    // the other half is work, and reopening the file put it all back. The slot it needed is the
    // one naming a body needed too, which is why they arrived together.

    /// Which two bodies the boolean tool would combine.
    boolean_a: Option<NodeId>,
    boolean_b: Option<NodeId>,
    boolean_op: BooleanOp,
    /// Fillet settings for the boolean, live before anything is applied so they can be dialled
    /// in the way the radius can.
    blend_mode: BlendMode,
    blend_radius: ParamRef,
    /// The GPU-traced viewport. Holds a compiled shader and a baked cage; see `traced/mod.rs`.
    traced: TracedView,
    /// The composed field, kept for picking.
    ///
    /// Picking used to raycast whichever *mesh* was on screen, which made the pivot depend on
    /// the display mode and on how finely that mesh happened to be sampled — in traced mode it
    /// tested a coarse stand-in for the surface actually being looked at, so clicks near a
    /// silhouette missed and silently fell back to the previous pivot. Tracing the field
    /// instead is mode-independent by construction.
    pick_field: Option<GraphField>,
    /// Nominal cells along the longest axis, used to derive a modelling tolerance.
    ///
    /// It no longer meshes anything for display. The viewport used to mesh the composed field
    /// with it, and that could not be made to look right: silhouette faceting on a swept tube
    /// measured 50 degrees at 64 cells and stopped improving after about 124, because a
    /// uniform grid over a box stretched by a curve running through empty space is never more
    /// than a handful of cells across the tube. Raising it added triangles to the sphere and
    /// left the tube a polygon.
    ///
    /// What it still does is set the tolerance the traced view bakes its cage at, and the
    /// tolerance export will mesh at. Worth renaming once export exists, since "cells" now
    /// describes neither.
    mesh_resolution: usize,
    /// How closely an export must follow the field: the detail the next File → Export starts
    /// its Tessellate block at, being the last one chosen, on a block or here.
    ///
    /// **A tolerance rather than a resolution, and the mesher is adaptive because of it.** A
    /// cell count is a statement about the sampling grid, not about the answer, and the two
    /// only agree while the part stays the same size. Adding an offset cube to a 105 mm sphere
    /// grew its box to 185 mm, and "256 cells" silently went from 0.41 mm cells to 0.72 mm --
    /// the same setting, nearly double the error, and 2 mm lattice walls down from five samples
    /// across to under three. Nothing said so, because from the control's point of view nothing
    /// had changed. A tolerance says what the *answer* may be wrong by, which is the thing a
    /// person has an opinion about, and the mesher spends whatever cells each region needs.
    ///
    /// Measured on a sphere-and-gyroid document, adaptive against uniform at a matched floor:
    /// 321 946 triangles instead of 2 425 796, 15.4 MB instead of 115.7 MB, in less time.
    ///
    /// **Separate from the display's setting, which is capped and should be.** An export is not
    /// interactive: it runs on a worker thread, it is allowed to take a minute, and the file it
    /// writes cannot be looked at again at a finer setting the way the viewport can. Sharing one
    /// number meant every export was silently clamped by a frame-rate concern that does not
    /// apply to it -- a gyroid inside a sphere came out at 64 cells across the whole part,
    /// roughly six cells per wall, and looked faceted for that reason and no other.
    ///
    /// A preset is kept as a preset, so it scales with whatever is meshed.
    export_detail: fieldform_scene::MeshDetail,
    /// Whether an export runs the simplifier. Off by default - see
    /// `fieldform_scene::Tessellation::reduce`.
    export_simplify: bool,
    /// Whether an export improves triangle shape. On by default, for the reason given on
    /// `fieldform_scene::Tessellation::improve`.
    export_relax: bool,
    /// What one unit in an OBJ file means. OBJ has no unit field, so this cannot be read
    /// from the file and has to be told to the importer.
    obj_import_unit: LengthUnit,
    /// A file picked for import, waiting for its options to be confirmed.
    ///
    /// How finely a STEP solid is tessellated is asked here too, and lives in `step_options`
    /// rather than in a field of its own.
    import_prompt: Option<ImportPrompt>,
    /// How many copies Modify > Pattern makes, counting the original.
    pattern_count: u32,
    /// Whether the pattern goes round the Z axis or along a line.
    pattern_radial: bool,
    /// How far apart a linear pattern's copies sit, along X.
    pattern_step_mm: f64,
    /// Which axis Modify > Twist winds about. Remembered between uses, like the mirror's.
    twist_axis: Axis,
    /// Which plane Modify > Mirror reflects across. Remembered between uses, like the offset
    /// distance: a part that wants mirroring in X usually wants it again.
    mirror_axis: Axis,
    /// The cutaway view, if one is on. See [`Section`].
    section: Section,
    /// Whether the manipulator is currently aimed at the section rather than at geometry.
    ///
    /// Explicit rather than inferred from "the section is on and nothing else is selected":
    /// two things claiming one gizmo is exactly the shape of bug this codebase keeps meeting,
    /// and the fix each time was to decide where the claim is *made*. Turning the section on,
    /// or clicking its row, makes the claim; selecting any geometry drops it.
    section_selected: bool,
    /// The graph edit in progress, so a drag lands on the undo stack once.
    ///
    /// Holds what the part looked like when the gesture *started*, which is what the entry has
    /// to reach back to however many frames the drag runs for.
    gesture: Option<Gesture>,
    /// What `verify` last found, and what the part was when it found it. See [`Verification`].
    verification: Option<Verification>,
    /// A material drag in progress, so a slider goes on the history once. See `set_part_material`.
    material_gesture: Option<MaterialGesture>,
    /// Whether a settled frame measures ambient occlusion and the key light's shadow.
    ///
    /// A view setting rather than a document one, like the section: it changes how the part
    /// is looked at and nothing about the part. On by default; the switch exists for the
    /// machine where a settled frame takes too long with it.
    idle_occlusion: bool,
    /// Whether the drawing plane shows its grid, which is also what Shift snaps to.
    ///
    /// A view setting, on by default. Off, there is nothing to snap to — deliberately, since a
    /// snap to something unseen cannot be aimed.
    show_grid: bool,
    /// The room the part is lit by. A view setting like the section; `View > Environment`.
    environment: fieldform_gpu::EnvironmentChoice,
    /// The draft-angle overlay. Under Display in the part panel, and `View > Draft analysis`.
    draft: DraftAnalysis,
    /// The boolean whose list is taking bodies from viewport clicks, if one is.
    ///
    /// The way other CAD software fills a list input: turn the list on, click the bodies, watch
    /// block update. While it is set a click in the viewport adds instead of selecting, so it
    /// is deliberately hard to leave on by accident — collapsing the block or clicking the
    /// control again clears it.
    collecting: Option<NodeId>,
    /// Imports whose original CAD is drawn over the field derived from it.
    ///
    /// App state rather than document state, like body visibility and for the same reason: it
    /// is what you are *looking at*, not what the part is. It inherits that item's unsolved
    /// "visibility is not saved" problem rather than inventing a second answer to it.
    showing_original: std::collections::HashSet<fieldform_scene::id::SourceId>,
    /// Whether an OBJ is read as a subdivision cage rather than as a model.
    ///
    /// Defaulted to the cage, which is what an OBJ import has always meant here: the file is a
    /// control mesh and the body is the surface it describes. A triangle mesh from anywhere
    /// else is the other reading and the file cannot say which it is.
    obj_as: ObjReading,
    /// Compiled cage and mesh leaves, reused across compiles. See `fieldform_field::LeafCache`.
    leaves: fieldform_field::LeafCache,
    /// Why the display graph would not compile, recorded when it was last compiled.
    ///
    /// Cached rather than asked for, because asking means compiling — see `compile_error`.
    compile_error: Option<String>,
    /// Parameters a panel row made or stopped referring to this frame, waiting for the graph edit
    /// they go on the history with — see [`ParameterEdits`] and [`App::parametric_field`].
    parameter_edits: ParameterEdits,
    /// Expressions being typed, by parameter. See [`ExpressionDraft`].
    expression_drafts: std::collections::HashMap<ParamId, ExpressionDraft>,
    /// How far Modify > Offset moves a surface, and how thick Modify > Shell leaves a wall.
    /// Kept on the app rather than asked for in a dialog, the same as the blend radius: the
    /// block it produces carries the number and changing it there is one drag.
    offset_distance: f64,
    shell_thickness: f64,
    /// How thick the next wall starts out — a curve given a thickness, on an extrude, a revolve
    /// or a sweep. The last thickness typed for one, for the reason the two above are kept:
    /// turning Wall on, or `extrude --wall` with no number, repeats it rather than asking.
    wall_thickness: f64,
    /// Everything the STEP reader is told. See [`fieldform_io::StepOptions`].
    step_options: fieldform_io::StepOptions,
    /// An import running in another process, if one is.
    pending_import: Option<PendingImport>,
    /// The Tessellate and Export blocks' running state — jobs in flight and what each last
    /// produced. The blocks themselves are in the document; see `outputs.rs`.
    outputs: outputs::Outputs,
    /// The `Drawing on …` line said at startup, kept so the perf log can open with it: a
    /// timing without the adapter that produced it is not a measurement (loop spec §2).
    adapter_line: String,
    /// Where per-frame timings are being written, while `perf` is on.
    ///
    /// Its own file rather than the console's, because the console's is capped and rewritten
    /// from what is in memory once it grows past the cap — and these lines are deliberately
    /// not in memory, at one per traced frame. They would vanish from the file at the first
    /// rewrite, which is exactly when a long session's numbers are wanted.
    perf_file: Option<std::path::PathBuf>,
    /// The document revision last written to the autosave, so an unchanged document is not
    /// written again.
    autosaved_revision: u64,
    /// When the autosave was last written, or the application started. See [`App::autosave`].
    autosaved_at: std::time::Instant,
    /// Raised by the device-lost callback, from whichever thread found the loss, and answered on
    /// this one. See [`App::keep_the_document_if_the_gpu_was_lost`].
    gpu_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether the loss has been answered, so the document is kept once rather than on every
    /// frame the process limps on for; the autosave goes on covering anything after it.
    kept_after_loss: bool,
    /// How long each recent frame's `ui` took, by the traced view's frame number, so a GPU cost
    /// that arrives a frame late can still be printed beside the CPU time of the frame that
    /// submitted it.
    frame_cpu_ms: std::collections::VecDeque<(u64, f32)>,
    /// How much of the display this machine pays for. See [`DisplayQuality`].
    display_quality: DisplayQuality,
    /// Whether somebody chose it, this session or an earlier one, as against the startup guess
    /// from the adapter. The first measured draft may correct a guess; it never overrides a
    /// choice.
    quality_chosen: bool,
}

/// What the overlay's readings are read for: which part, the application's own view of it, and
/// the count that moves whenever the document may have ([`App::document_moves`]).
///
/// **Everything the readings read but the document is a field here, and is read from here.**
/// [`Self::read`] is handed this and the document, so no view setting can reach a reading without
/// being part of what it is kept against. A field left out would be a reading left on screen after
/// what it was read from had moved, with nothing to say so.
#[derive(Clone, Debug, PartialEq)]
struct OverlayKey {
    moves: u64,
    part: PartId,
    /// The meshing resolution, one cell of which is the pick box's margin.
    resolution: usize,
    /// The joint points that go unseen with their profile row (`points_hidden_with_rows`).
    hidden_with_rows: Vec<NodeId>,
    /// The part's bodies the application draws as their mesh rather than traced
    /// ([`App::drawn_as_mesh`]), sorted.
    meshed: Vec<NodeId>,
}

impl OverlayKey {
    /// The readings, from the document and this. `numbers` is the application's reading of a
    /// number for drawing and sizing ([`App::resolve`]), which asks the document and nothing else.
    fn read(
        &self,
        doc: &Document,
        numbers: &dyn Fn(&ParamRef) -> Option<f64>,
    ) -> fieldform_field::OverlayReadings {
        let part = &doc
            .parts
            .get(&self.part)
            .expect("active_part must name a part in the document")
            .graph;
        // Asked only of the part's bodies (`drawn_graph`), which are what `meshed` was read from.
        let drawn_as_mesh = |id: NodeId| self.meshed.contains(&id);
        let inputs = fieldform_field::OverlayInputs {
            drawn_as_mesh: &drawn_as_mesh,
            hidden_with_rows: &self.hidden_with_rows,
            resolution: self.resolution,
            empty_mm: DEFAULT_BODY_MM,
            numbers,
        };
        fieldform_field::OverlayReadings::build(doc, part, &inputs)
    }
}

/// What the block list's readings are read for: which part, and the count that moves whenever the
/// document may have ([`App::document_moves`]).
///
/// **Only those two, because the list's readings read nothing else.** Where each block is drawn
/// and the order of the rows are the document's own: no view setting reaches either, so the key
/// is the overlay's without its view, and [`Self::read`] is handed nothing but this and the
/// document, so none can.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ListKey {
    moves: u64,
    part: PartId,
}

impl ListKey {
    /// The readings, from the document and this.
    fn read(&self, doc: &Document) -> fieldform_scene::ListReadings {
        let part = &doc
            .parts
            .get(&self.part)
            .expect("active_part must name a part in the document")
            .graph;
        fieldform_scene::ListReadings::build(doc, part)
    }
}

/// The block one curve control point sits on, if it sits on one.
///
/// A free function so the row can ask while `self` is already borrowed by the closure drawing it.
fn placed_on(app: &App, id: CurveId, index: usize) -> Option<NodeId> {
    app.doc.curves.get(&id).and_then(|c| c.placed_on(index))
}

/// The point blocks in the part, which are what a control point can sit on.
fn point_blocks(app: &App) -> Vec<(NodeId, String)> {
    let mut out: Vec<(NodeId, String)> = app
        .part_graph()
        .nodes()
        .filter(|(_, n)| matches!(n, Node::Point { .. }))
        .map(|(n, _)| (n, app.body_label(n)))
        .collect();
    out.sort_by_key(|(n, _)| n.raw());
    out
}

/// The last `verify`, kept with the state it was true of.
///
/// **Not the document, and not a second copy of it**: a report is derived, and the only thing
/// remembered about the part is a fingerprint to tell whether the report still describes it.
/// Any edit since makes it stale, and a stale report reads as *unverified* — never as the pass
/// it may once have been (blend-integrity-spec §5, the rule the Expression block's verify
/// follows too). Dated, because "held" is a claim about a moment.
struct Verification {
    part: PartId,
    fingerprint: String,
    /// The day it ran, UTC, as `2026-09-18`.
    when: String,
    reports: Vec<fieldform_field::CriticalReport>,
}

/// Today, UTC, as `YYYY-MM-DD`. Days-from-civil run backwards, so the application does not
/// take on a date library for one line of a report.
fn utc_date(seconds_since_epoch: u64) -> String {
    let z = (seconds_since_epoch / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Every named value with a number, for the menus that offer them.
///
/// A computed one is offered at what it currently computes to; one whose expression has
/// failed has no number to show beside its name and is left out until it has.
fn named_values(app: &App) -> Vec<(ParamId, String, f64)> {
    let mut all: Vec<(ParamId, String, f64)> = app
        .doc
        .parameters
        .values()
        .filter_map(|p| app.doc.parameter_value(p.id).map(|v| (p.id, p.name.clone(), v)))
        .collect();
    all.sort_by(|a, b| a.1.cmp(&b.1));
    all
}


impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let wgpu_render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("fieldform-app requires the wgpu backend (eframe's \"wgpu\" feature)")
            .clone();

        // **A lost GPU names itself in the crash log, before anything else trips over it.**
        //
        // Windows resets a display driver whose work runs longer than about two seconds on one
        // packet (TDR), and a Maxwell-era card cannot preempt a compute dispatch, so a heavy
        // frame on one is killed rather than paused. wgpu then marks the device invalid and
        // every later call fails -- and the first to notice, on 2026-09-15 on a Quadro M4000,
        // was egui-wgpu asking for a staging buffer for its own index data, which panicked
        // about buffer sizes that were fine. That report went looking at the wrong crate. This
        // writes the real cause first, so the next one reads "GPU lost" rather than "egui".
        // And it says where the work is, since the next thing the reader of that note wants is
        // their work rather than the reason.
        //
        // **It does not write the document itself.** wgpu runs this on whichever thread made
        // the call that found the device gone -- a shader compile on a worker as readily as a
        // submit here -- and the document belongs to the interface thread, behind no lock. So
        // it raises `gpu_lost` and asks for a frame, and the interface thread keeps the
        // document (`App::keep_the_document_if_the_gpu_was_lost`) and adds where to this note.
        let gpu_lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let raised = std::sync::Arc::clone(&gpu_lost);
        let interface = cc.egui_ctx.clone();
        wgpu_render_state.device.set_device_lost_callback(move |reason, message| {
            let said = format!(
                "GPU device lost ({reason:?}): {message}\n\
                 Windows resets a display driver that takes longer than about 2 s on one \
                 frame, and a heavy document on a slow or old GPU can do that. Every GPU call \
                 after this fails, so the application aborts on the next one.\n\
                 The interface keeps the document as it stands in <name>.recovered.\
                 {DOCUMENT_EXTENSION} beside its file, or beside this log if it was never saved, \
                 and says where below if a frame is left to do it in. Failing that, the document \
                 as of the last minute is at {}, which `open file=` reads.",
                autosave_path().display()
            );
            log::error!("{said}");
            note_in_crash_log("fieldform GPU lost", &said);
            raised.store(true, std::sync::atomic::Ordering::Release);
            interface.request_repaint();
        });

        // **Which GPU is actually doing the work, said once at startup.**
        //
        // This machine has six adapters — an AMD 780M and an NVIDIA RTX A500, each through both
        // Vulkan and DX12, plus a software rasteriser and a GL fallback — and wgpu's choice
        // among them is not guaranteed to be the intended one. Windows' own graphics preference
        // can override it silently, which means a timing that halves after a driver update looks
        // like a regression in this code and is not.
        //
        // `shading-spec.md` §10.8 asks for exactly this and calls it the cheapest of three
        // things worth doing, because every number measured on this application is ambiguous
        // without it. egui_wgpu already logs the adapters it *found*; nothing said which one it
        // took.
        // **Every button wears an outline, so a button looks like one before it is clicked.**
        //
        // egui draws an inactive widget with no stroke by default, which makes a toggle read as
        // a label until the pointer is over it — reported of the box's Ratio toggle, and true of
        // every mode switch in every block. The rule is one grey outline when idle and the
        // accent when the thing it names is on, said here once rather than per widget: a style
        // set at the widget is a style the next widget will not have.
        //
        // `inactive` is the resting state and `open` is a menu button whose menu is up; both get
        // the grey. `hovered` and `active` keep egui's own brighter strokes, which are the
        // feedback that the pointer is on it. `selected` is the blue.
        let mut style = (*cc.egui_ctx.global_style()).clone();
        {
            let w = &mut style.visuals.widgets;
            let outline = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(90));
            // **`inactive` and `open` only.** `noninteractive.bg_stroke` is not a button's at
            // all -- it is the stroke every *frame* uses, so the outline around each block, the
            // divider between the viewport and the block list, and the one between Display and
            // the console are all drawn with it. Clearing it to make buttons the only outlined
            // thing removed every one of those instead. Reported as the UI losing its lines.
            w.inactive.bg_stroke = outline;
            w.open.bg_stroke = outline;
            // **`selection.stroke` is left alone: it is the selected *text*, not an outline.**
            // `Style::interact_selectable` assigns it to `fg_stroke`, so setting it to the
            // selection's own fill painted blue text on a blue background and every selected
            // row and toggle went blank. Reported as text disappearing on click. egui already
            // fills a selected widget with the accent, which is the whole of what "blue when it
            // is on" needs.
        }
        cc.egui_ctx.set_global_style(style);

        let info = wgpu_render_state.adapter.get_info();
        log::info!(
            "rendering on {} ({:?}, {:?}) — driver {} {}",
            info.name,
            info.device_type,
            info.backend,
            info.driver,
            info.driver_info,
        );
        // **And in the console, where somebody can see it.** The log line above goes to stderr,
        // which a windowed application launched from Explorer does not have — so the one fact
        // that makes every timing in this application interpretable was written somewhere nobody
        // would ever read. A laptop with two GPUs can be quietly drawing on the slower one, and
        // the symptom is "it feels sluggish", which is indistinguishable from the model being
        // heavy. Said once, at startup, beside everything else the application says.
        let adapter_said = format!(
            "Drawing on {} ({}), {:?} — driver {} {}; GPU timestamps {}",
            info.name,
            match info.device_type {
                wgpu::DeviceType::DiscreteGpu => "discrete",
                wgpu::DeviceType::IntegratedGpu => "integrated",
                wgpu::DeviceType::VirtualGpu => "virtual",
                wgpu::DeviceType::Cpu => "software — expect it to be very slow",
                wgpu::DeviceType::Other => "unknown kind",
            },
            info.backend,
            info.driver,
            info.driver_info,
            // Whether `perf` will report the card's own time or the wall clock's. Said here
            // because it is a fact about the device the line already describes, and the
            // ledger wants the two together.
            if wgpu_render_state
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
            {
                "on"
            } else {
                "unavailable, so `perf` times by the wall clock"
            },
        );

        // The same size as anything the Create menu makes, so the first thing on screen and
        // the second are comparable. It was a metre for a while — real parts arrive at real
        // sizes, and starting three orders of magnitude below anything importable made every
        // scale-relative default wrong on arrival — but that argument was about *not* starting
        // at 1 mm, and 100 mm satisfies it while matching everything else.
        let cage = Cage::cube_of(DEFAULT_BODY_MM);
        let cage_mesh = CageMesh::from_cage(&cage);
        let mut doc = Document::with_cage(cage.clone());
        let active_cage = *doc.cages.keys().next().expect("with_cage inserts one");
        let active_part = *doc.parts.keys().next().expect("with_cage inserts one");

        // Open on an empty part, not on a cube nobody asked for.
        //
        // The cage itself stays in the document; only the body that draws it is taken out. That
        // is the same state deleting the cage block already produces, and it is what keeps
        // `active_cage` naming something real — the app leans on that everywhere — while
        // leaving the viewport empty for the splash to sit over. Create > Cage puts *this*
        // cage back rather than making a second one, so nothing has been lost by not drawing
        // it.
        doc.parts
            .get_mut(&active_part)
            .expect("with_cage inserts one")
            .graph = Graph::empty();
        // Settled as every graph written past the history is (`block-ownership-spec.md` §4.1),
        // as `new_document` does after the same write.
        doc.settle_ownership();

        // A test's `App` draws on Fast whatever the owner chose: the tests are about what the
        // interface does, and the owner's own rung would make them as slow as Enchilada is.
        let saved_quality = if cfg!(test) { Some(DisplayQuality::Fast) } else { display_quality::load_saved_quality() };
        let (start_quality, start_chosen) = starting_quality(
            saved_quality,
            info.device_type == wgpu::DeviceType::DiscreteGpu,
        );

        let echo_base = app::echoing::EchoBase {
            snapshot: echo::Snapshot::of(&doc, &doc.parts[&active_part].graph),
            part: active_part,
            moves: 0,
        };
        let mut app = Self {
            egui_ctx: cc.egui_ctx.clone(),
            echo_base,
            echo_log: Vec::new(),
            session: None,
            last_values: std::collections::HashMap::new(),
            palette: None,
            recent: Vec::new(),
            bindings: Vec::new(),
            binding_conflicts: Vec::new(),
            capturing: None,
            bindings_note: None,
            marking: None,
            last_command: None,
            shelf: Vec::new(),
            shelf_drawn: Vec::new(),
            viewport_rect: None,
            floating_panel: false,
            focus_first_number: false,
            drawing_pending: false,
            keep_pending: false,
            pending_numbers: Vec::new(),
            palette_opens: 0,
            doc,
            active_cage,
            control: ControlMesh::Cage,
            active_part,
            file: None,
            revision: 0,
            saved_revision: 0,
            document_moves: 0,
            kept_readings: Default::default(),
            kept_list: Default::default(),
            title: String::new(),
            splash: None,
            splash_shown: true,
            show_shortcuts: false,
            show_about: false,
            console_progress: None,
            console_input: String::new(),
            console_log: vec![ConsoleLine {
                ok: true,
                text: "Type a command, or `help`. Everything here is the same operation a menu \
                       item runs."
                    .to_string(),
                said: 1,
            }],
            drawing_plane: DrawingPlane::None,
            console_reports: 0,
            console_warnings: 0,
            script: Script::from_command_line(),
            script_running: false,
            expect_results: Vec::new(),
            pending_snap: None,
            script_tally: None,
            traced_error_said: None,
            // **Never from a test's `App`.** The file is one name in the temp folder and this
            // truncates it, so a test run would empty the log of whatever instance the owner has
            // open (`tests::headless_app`).
            console_file: if cfg!(test) { None } else { open_console_log() },
            console_file_bytes: 0,
            started: std::time::Instant::now(),
            console_history: Vec::new(),
            console_history_at: None,
            closing: false,
            stack: CommandStack::default(),
            registry: OpRegistry::default(),
            keymap: commands::default_keymap(),
            wgpu_render_state,
            camera: OrbitCamera::default(),
            cage_mesh,
            imported_mesh: CageMesh::default(),
            // **Field, because that is what the model looks like.** Polygon draws the
            // control cage and nothing else, which is right while editing a cage and wrong as
            // a starting point: the cage is scaffolding, and a part built from primitives has
            // no cage at all, so the first thing a new user saw was an empty viewport with no
            // hint that a mode was the reason. Every Create command already switches here, so
            // this only changes what is true before the first one.
            view_mode: ViewMode::Field,
            stages: Vec::new(),
            selection: Picked::default(),
            hover: Selection::None,
            move_state: None,
            point_drag: None,
            pending_curve: None,
            surface_axis_notes: std::collections::HashMap::new(),
            surface_axis_edit_before: None,
            gizmo_drag: None,
            gizmo_hover: None,
            show_view_cube: true,
            clipboard: None,
            stolen_tab: None,
            console_complete: false,
            press_on_gizmo: false,
            press_on_control_point: false,
            other_points: Vec::new(),
            pending_narrow: None,
            marquee: None,
            press_on_marquee: false,
            show_control_points: true,
            press_on_curve: false,
            curve_edit_before: None,
            weight_edit_before: None,
            curve_display: Vec::new(),
            curve_revision: 0,
            profiles: ProfileRows::default(),
            loft_readings: None,
            pending_joins: Vec::new(),
            viewport_hover: None,
            viewport_menu_at: None,
            selected_curve: None,
            selected_points: Vec::new(),
            // 5 mm is a plausible starting pipe for hardware-scale work and is visible at
            // the default framing; it is a starting point, not a constraint.
            pipe_radius: ParamRef::Literal(5.0),
            selected_body: None,
            selected_block: None,
            press_on_selection: false,
            hidden_curves: std::collections::HashSet::new(),
            hidden_surfaces: std::collections::HashSet::new(),
            surface_display: Vec::new(),
            surface_prints: Vec::new(),
            renaming: None,
            ref_edit: None,
            naming_parameter: None,
            name_drafts: std::collections::HashMap::new(),
            size_locked: std::collections::HashSet::new(),
            boolean_a: None,
            boolean_b: None,
            boolean_op: BooleanOp::Difference,
            blend_mode: None,
            blend_radius: ParamRef::Literal(2.0),
            traced: TracedView::default(),
            pick_field: None,
            // 64 cells across the longest axis: fast enough to feel interactive on this
            // brute-force field evaluator, fine enough to read the shape.
            mesh_resolution: 64,
            export_detail: fieldform_scene::MeshDetail::Normal,
            export_simplify: false,
            export_relax: true,
            // Metres, because most DCC tools write most of the OBJs that reach a modeller and
            // treat one unit as one metre, and their default cube is 1 unit. A CAD exporter
            // more often means millimetres, which is why this is a visible setting and not a
            // constant.
            obj_import_unit: LengthUnit::Metre,
            pattern_count: 6,
            pattern_radial: true,
            pattern_step_mm: DEFAULT_BODY_MM * 0.5,
            twist_axis: Axis::Z,
            mirror_axis: Axis::X,
            import_prompt: None,
            section: Section::default(),
            section_selected: false,
            gesture: None,
            verification: None,
            material_gesture: None,
            idle_occlusion: true,
            show_grid: true,
            environment: fieldform_gpu::EnvironmentChoice::Studio,
            draft: DraftAnalysis::default(),
            collecting: None,
            showing_original: std::collections::HashSet::new(),
            obj_as: ObjReading::Cage,
            leaves: fieldform_field::LeafCache::default(),
            compile_error: None,
            parameter_edits: ParameterEdits::default(),
            expression_drafts: std::collections::HashMap::new(),
            offset_distance: 1.0,
            shell_thickness: 2.0,
            // A shell's own starting wall, since it is the same question asked of a curve.
            wall_thickness: 2.0,
            step_options: fieldform_io::StepOptions::default(),
            pending_import: None,
            outputs: Default::default(),
            adapter_line: adapter_said.clone(),
            perf_file: None,
            autosaved_revision: 0,
            // From launch rather than from the first edit, so a document is not written the
            // instant it is touched — which on a heavy one is a stall for no gain, the file
            // having just been opened.
            autosaved_at: std::time::Instant::now(),
            gpu_lost,
            kept_after_loss: false,
            frame_cpu_ms: std::collections::VecDeque::new(),
            // The rung chosen last session if there is one; otherwise a guess from the adapter,
            // corrected once by the first draft it measures: a discrete card is offered
            // everything and demoted if its first draft misses the budget, and anything else
            // starts in the middle.
            display_quality: start_quality,
            quality_chosen: start_chosen,
        };
        // Before anything else it says, because it is the frame of reference for all of it.
        app.say(adapter_said);
        // A kept rung is said once, because it is the one setting that outlives a session: a
        // still that looks coarse for a reason chosen weeks ago should say where the reason is.
        if let Some(saved) = saved_quality {
            app.say(format!(
                "Display quality: {}, as chosen in an earlier session. View > Display quality \
                 changes it.",
                saved.label()
            ));
        }
        // A test's `App` offers nothing it did not make: the temp folder is the owner's.
        if !cfg!(test) {
            app.offer_the_autosave();
        }
        app.rebuild_render_resources();
        // Frame the starting cage rather than trusting the default distance, which is fitted
        // to a default-sized body and so is right only for one. A real cage is whatever size
        // it is.
        let (center, radius) = cage_bounds(app.cage());
        app.camera.frame(center, radius);
        // And a starting pipe radius that suits the starting cage, for the same reason the
        // imported one is derived from the imported curve: a radius unrelated to the model is
        // a radius the mesher cannot resolve.
        app.pipe_radius = ParamRef::Literal(tidy_mm(radius as f64 * 0.05));
        app.blend_radius = ParamRef::Literal(tidy_mm(radius as f64 * 0.01));
        // The document a session opens on is said already: the echo starts from it.
        app.echo_rebase();
        // The keys: the registry's, with the person's file over them. Not in a test, whose keys
        // are the defaults whatever file the owner keeps.
        app.rebuild_keymap();
        if !cfg!(test) {
            app.load_bindings();
        }
        app.load_shelf();
        app
    }

    /// What the camera is looking at, for the clip planes and the marcher's reach.
    ///
    /// **Its own step, because a drag needs it without the rest of the rebuild.** The far plane
    /// is the scene's centre plus twice its radius, and both came only from the full rebuild --
    /// so a body dragged past where the part *was* when that last ran met the far plane and was
    /// cut off along a straight line, until the hand came off and the rebuild moved the plane.
    /// One extent walk over the bodies is cheap enough for every frame of a drag; the cage bake
    /// and the picking compile beside it in the rebuild are not, and are what the drag skips.
    fn refresh_scene_bounds(&mut self) -> (Point3<f64>, Point3<f64>) {
        let (lo, hi) = self.pick_bounds();
        let centre = Point3::from(((lo.coords + hi.coords) * 0.5).cast::<f32>());
        let radius = ((hi - lo).norm() * 0.5).max(1e-6) as f32;
        self.camera.set_scene(centre, radius);
        (lo, hi)
    }

    /// Brings everything derived from the part up to date after it changes: curve ends settle
    /// onto their points, the cage is re-triangulated, the scene bounds are refreshed, the traced
    /// view is marked stale, the field that picking traces is compiled, axes on a surface follow
    /// their body, and the drawn meshes are uploaded again.
    ///
    /// Called whenever the part, or what is visible of it, changes. A cage drag does not come here
    /// each frame — `update_after_vertex_move` rewrites only the faces it moved — and `end_move`
    /// calls it once the gesture is committed.
    fn rebuild_render_resources(&mut self) {
        let whole = std::time::Instant::now();
        // Whatever asked for this changed the part or what is visible of it, which is what the
        // overlay reads; opening a file and starting a new one reach the readings this way.
        self.document_moved();
        // **Curve ends catch up here, because this is what runs when the part changes.**
        // It was called on binding an end, on editing a variable and on opening a document —
        // and not when the *point itself* moved, which is the one case anybody would try first.
        // Dragging a point's X went through the block panel, which rebuilds but never asked a
        // curve to follow, so the end stayed where it was and the feature looked inert.
        //
        // Before the cage mesh and the bounds below, since both read geometry this may move.
        if self.settle_curves_on_points() {
            self.rebuild_curve_display();
        }
        self.cage_mesh = CageMesh::from_cage(self.control());

        // What the camera is looking at, for the clip planes and the marcher's reach. Here
        // because this runs whenever the part, or what is visible of it, changes -- and on its
        // own for the drags that move a body without running the rest of this.
        let (lo, hi) = self.refresh_scene_bounds();

        // The traced view holds a compiled shader and a baked cage, both of which this rebuild
        // has just invalidated. Marking rather than rebuilding here keeps the cost off cage
        // edits that are never looked at in that mode.
        self.traced.invalidate(traced::Quality::Final);

        // The field picking traces. Built here rather than per click, because compiling it
        // subdivides the cage and a click should not pay for that.
        //
        // And it is the *only* compile: the panel used to call `compile_error`, which compiled
        // the whole graph again to ask whether it could be compiled — from inside the inspector,
        // so once a frame. On a 109 000-triangle STEP import that is four BVH builds and four
        // crease passes sixty times a second, for an answer that only changes when the document
        // does. One compile, both consumers.
        //
        // Through the leaf cache, so hiding a body does not rebuild every other body's BVH.
        // What a show/hide changes is which roots are in the graph; the meshes are the same
        // meshes, and rebuilding them was 90 ms of the pause on a 109 000-triangle import.
        // From the box measured above, which is the one the overlay's readings keep.
        let tolerance = (hi - lo).norm() * 1e-3;
        let graph = self.display_graph();
        let compiling = std::time::Instant::now();
        let compiled = GraphField::compile_cached(
            &self.doc,
            &graph,
            tolerance.max(1e-4),
            &mut self.leaves,
        );
        self.note_stage("compiling the part for picking", compiling);
        match compiled {
            Ok(field) => {
                self.pick_field = Some(field);
                self.compile_error = None;
            }
            Err(e) => {
                self.pick_field = None;
                // A raw node id means nothing in a panel that lists blocks by name, so the
                // errors that carry one say which block instead. This is the difference
                // between a document that looks broken and one that says what to change:
                // a blend against a lattice draws nothing and exports nothing, and until it
                // named the block there was no way to tell which of several was at fault.
                self.compile_error = Some(self.explain_compile(&e));
            }
        }

        // The control cage is the only mesh the viewport draws. Everything smooth — the
        // limit surface included — is the traced view's job, evaluated per pixel.
        // **An axis on a surface follows its body**, and this is where it catches up: after the
        // field is compiled, because the rule needs the body's field, and before what is drawn is
        // rebuilt. Only the curve display is refreshed — never this function — so it cannot loop.
        if self.refresh_surface_axes() {
            self.rebuild_curve_display();
        }
        self.refresh_drawn_meshes();
        self.note_stage("rebuilding what is drawn", whole);
    }

    /// Rebuilds the triangle buffers the viewport draws and uploads them.
    ///
    /// Separate from the rest of the rebuild because the *mode* changes what is in them and
    /// nothing else about the part: cycling to Polygon needs the imported solids as triangles,
    /// where in Field mode the tracer is already drawing them. Pressing Tab should not pay for
    /// a compile, and an edit should not skip this.
    fn refresh_drawn_meshes(&mut self) {
        let started = std::time::Instant::now();
        self.rebuild_imported_mesh();
        let resources = CageRenderResources::new(
            &self.wgpu_render_state.device,
            self.wgpu_render_state.target_format,
            &self.cage_mesh,
            &self.imported_mesh,
        );
        self.wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);
        self.note_stage("the drawing resources", started);
    }

    /// Runs a named operation. Every key binding funnels through here, and the eventual
    /// Python API will too — so a scripted call and a key press cannot diverge in behaviour.
    ///
    /// Operations split into two kinds. *Document* operations mutate `doc` and go through
    /// the registry and the undo stack. *Application* operations (undo itself, view changes)
    /// act on transient UI state that isn't part of the saved document and isn't undoable —
    /// handled here directly, since putting them on the undo stack would mean undoing an
    /// undo.
    fn dispatch(&mut self, op: &str, args: &Args) {
        // Starting something else ends a placement in progress, and says so. A curve left
        // waiting under another tool would be a click that means something nobody can see,
        // which is the trap the "no modes" rule exists to avoid.
        //
        // `modify.*` too, now that the tool stays on between curves (item 111): drawing a profile
        // and extruding it is the ordinary sequence, and an extrude that left the tool armed would
        // make the next click in the viewport a line nobody meant.
        if (op.starts_with("create.") || op.starts_with("modify."))
            && !matches!(op, "create.line" | "create.curve")
            && let Some(pending) = self.pending_curve.take()
        {
            self.say(curve_tool_ended(&pending, "Stopped placing curves"));
        }
        match op {
            "param.set" => self.param_command(args),
            "create.number" => self.add_number(args),
            "create.tessellate" => self.tessellate_command(args),
            "edit.undo" => {
                // A material drag and an output block's fold into the top of the history the
                // same way the graph's does, so they end with it (`step_history`).
                self.material_gesture = None;
                self.outputs.gesture = None;
                let pull = typed_direction(&self.doc, &self.draft.pull);
                match step_history(&mut self.stack, &mut self.doc, &mut self.gesture, true) {
                    Some(name) => {
                        keep_typed_direction(&self.doc, &mut self.draft.pull, pull);
                        self.say(format!("Undo: {name}"));
                        // Undoing a refine is the same renumbering run backwards.
                        if name == "cage.refine" {
                            self.forget_cage_selection();
                        }
                        self.after_document_change();
                    }
                    None => self.warn("Nothing to undo"),
                }
            }
            "edit.redo" => {
                self.material_gesture = None;
                self.outputs.gesture = None;
                let pull = typed_direction(&self.doc, &self.draft.pull);
                match step_history(&mut self.stack, &mut self.doc, &mut self.gesture, false) {
                    Some(name) => {
                        keep_typed_direction(&self.doc, &mut self.draft.pull, pull);
                        self.say(format!("Redo: {name}"));
                        if name == "cage.refine" {
                            self.forget_cage_selection();
                        }
                        self.after_document_change();
                    }
                    None => self.warn("Nothing to redo"),
                }
            }
            // Each reads its block's own numbers -- `sphere radius=10 centre=40,0,0` -- and a
            // menu, passing none, gets the default (`console_keys::primitive_with_keys`).
            "create.sphere" => self.add_primitive_with(PrimitiveKind::Sphere, args),
            "create.cube" => self.add_primitive_with(PrimitiveKind::Cube, args),
            "create.skewbox" => self.add_primitive_with(PrimitiveKind::SkewedBox, args),
            "create.cylinder" => self.add_primitive_with(PrimitiveKind::Cylinder, args),
            "create.cone" => self.add_primitive_with(PrimitiveKind::Cone, args),
            "create.torus" => self.add_primitive_with(PrimitiveKind::Torus, args),
            "create.capsule" => self.add_primitive_with(PrimitiveKind::Capsule, args),
            "create.plane" => self.add_primitive_with(PrimitiveKind::Plane, args),
            // Gyroid on a rectangular map when nothing says otherwise, and changeable in the
            // block afterwards — the same reasoning as Create > Curve, which stopped asking for
            // a degree up front once the degree could be changed without losing the shape. The
            // cell, the map and its numbers, the phase and its own number, and the frame are all
            // read by their own names (`console_keys::lattice_with_keys`); a cell, map or phase
            // that does not exist is refused by name rather than quietly made a gyroid.
            "create.tpms" => self.add_lattice_with(args),
            "create.point" => self.add_primitive_with(PrimitiveKind::Point, args),
            // Every argument optional, so `ramp` alone places the default one and the console
            // and the menu are the same act. `in` and `out` take a pair because a range is one
            // idea: two arguments for it would let a file say half of one.
            // The ranges are pairs because a range is one idea, and a half-written one is
            // refused rather than half-read (`console_keys::ramp_with_keys`).
            "create.ramp" => self.add_ramp_with(args),
            "create.cage" => self.add_cage_body(),
            // `extrude --wall 2 --side left`, `revolve --wall 3 --side centre`,
            // `sweep --wall 1.5 --ends round`: a wall is asked for by name, on the word of the
            // block it is a kind of, rather than by three more words (`outline-wall-spec.md`
            // §6). The same arguments in the console's own spelling: `extrude wall=2`.
            // And an extrude's depth, after it is made: `extrude --start 5 --two 10 --draft2 3`,
            // `extrude --symmetric`, `extrude --depth 30 --draft 5`
            // (`extrude-revolve-options-spec.md` §4). Applied to the block the command left
            // selected, so a wall and a profile are told the same way.
            // `extrude profile=Profile1 height=30 draft=2`, `extrude curve=l1 wall=2`: the curve
            // named rather than the one in hand (A6). The field names and the older spellings
            // both work -- `height`/`depth`, `height_2`/`two`, `draft_2`/`draft2`,
            // `thickness`/`wall` -- through one alias table.
            "create.extrude" => {
                let Some(args) = self.curve_keys_in_hand("extrude", args, EXTRUDE_ALIASES) else { return };
                if wall_wanted(&args) {
                    self.add_wall_solid(WallBlock::Extrude, &args);
                } else {
                    self.add_profile_solid(false);
                }
                self.apply_extrude_options(&args);
            }
            // `revolve --angle 120 --symmetric`, `revolve --angle 90 --two 45`: how far round
            // and which way, applied to the block the command made, wall or profile alike.
            // And `revolve profile=p axis=l_axis`, or `axis=z origin=20,0,0`: the axis a curve
            // (the frame follows it) or a direction and a point.
            "create.revolve" => {
                let Some(args) = self.curve_keys_in_hand("revolve", args, REVOLVE_ALIASES) else { return };
                if let Err(why) = self.revolve_frame(&args, &fieldform_scene::Frame::default()) {
                    self.warn(why);
                    return;
                }
                if wall_wanted(&args) {
                    self.add_wall_solid(WallBlock::Revolve, &args);
                } else {
                    self.add_profile_solid(true);
                }
                self.apply_revolve_options(&args);
            }
            // `sweep profile=p spine=path section=skewed`, `sweep curve=l1 wall=1 spine=path`.
            // Its own arm, ahead of the two below, so the curves are named before either runs.
            "create.sweep" if console_keys::names_a_curve(args) => self.sweep_with_keys(args),
            "create.sweep" if wall_wanted(args) => {
                self.add_wall_solid(WallBlock::Sweep { spine: None }, args)
            }
            "create.sweep" => {
                self.add_sweep_body(None);
                self.apply_sweep_options(args);
            }
            "create.loft" => self.add_loft_body(args),
            // Both are placed by clicking — the start, then the end (`sketcher-spec.md` §7).
            // The line a pipe or a draft vector makes for itself still comes from `add_line`,
            // because those need a curve to exist now rather than after two clicks.
            // `line p0=0,0,0 p1=40,0,0` places one now; `line` alone starts clicking (A6).
            "create.line" => self.place_or_begin_curve("line", 1, args),
            "create.curve" => {
                // Degree one when nothing says otherwise -- a straight line, which is what
                // most drawing starts from and the thing you least often have to undo. It was
                // cubic on the argument that a curve should look like its degree from the moment
                // it exists; that is true and it is the wrong default, because raising a degree
                // is exact and lowering one is not.
                let degree = args
                    .get("degree")
                    .and_then(|v| v.as_f64())
                    .map(|d| d.round().max(1.0) as usize)
                    .unwrap_or(1);
                self.place_or_begin_curve("curve", degree, args);
            }
            "file.export_stl" => self.export_command(args),
            // `pipe curve=l1 radius=3`; a ramp's name drives the radius.
            "create.pipe" => self.pipe_with_keys(args),
            "create.boolean" => self.apply_boolean(args),
            // The distance is an argument rather than a prompt, matching `curve degree=5`:
            // it is remembered afterwards, so `offset` alone repeats the last one.
            // Every modifier reads `target=` by name, else the selection as before, and its
            // numbers by their own field names (`console_keys`). What a tool remembers between
            // uses -- the last distance, the last axis -- it still remembers, from the number it
            // was given.
            "modify.pattern" => self.apply_pattern(args),
            "modify.twist" => self.apply_twist(args),
            "modify.attract" => self.apply_attract(args),
            "modify.mirror" => self.apply_mirror(args),
            "modify.offset" => self.apply_offset(args),
            "modify.morph" => self.apply_morph(args),
            "modify.move" => self.apply_placing(console_keys::Placing::Move, args),
            "modify.turn" => self.apply_placing(console_keys::Placing::Turn, args),
            "modify.scale" => self.apply_placing(console_keys::Placing::Scale, args),
            "modify.shell" => {
                self.apply_shell(args);
            }
            // `delete target=Sphere_0` names what goes, as the echo of a delete has to: a line that
            // deleted "the selection" would delete whatever the replay happened to have selected.
            "edit.delete_body" => match args.get("target").map(|v| v.text()) {
                None => self.delete_selection(),
                Some(name) => match (self.node_named(&name), self.curve_named(&name)) {
                    (Some(id), _) => {
                        self.select_body(Some(id));
                        self.delete_selected_body();
                    }
                    (None, Some(curve)) => self.delete_curve(curve),
                    (None, None) => self.warn(format!("nothing is called {name:?} to delete")),
                },
            },
            // The V typed, `variable 'Sphere'` and `inline 'Sphere'`: the row by what it is
            // called rather than the selection, since a nested row is often not the one selected.
            "block.make_variable" => self.variable_from_console(args),
            // `rename 'Sphere_0' to=ball`, and with no name the selected row's field opened.
            "block.rename" => self.rename_from_console(args),
            "block.put_back_inside" => self.inline_from_console(args),
            "edit.copy" => self.copy_selection(true),
            "edit.cut" => self.cut_selection(),
            "edit.duplicate" => self.duplicate_selection(),
            "edit.paste" => self.paste(true),
            "edit.paste_here" => self.paste_at_hover(),
            "profile.join" => self.join_selected(),
            "profile.ungroup" => self.ungroup_selected(),
            "profile.align" => self.align_command(args),
            "profile.relate" => self.relate_command(args),
            "profile.break" => self.break_command(args),
            "profile.realign" => self.realign_command(args),
            "curve.set_points" => self.set_command(args),
            "view.measure" => self.measure_selection(),
            "view.estimate" => self.estimate_from_console(args),
            // A default that means something on the part in front of you rather than a fixed
            // number: a hundredth of the longest side is about where a wall stops being a wall
            // on anything printed. `thickness 0.8` says it outright.
            "view.thickness" => {
                let (lo, hi) = self.pick_bounds();
                let e = hi - lo;
                let default = e.x.max(e.y).max(e.z) * 0.01;
                let threshold = args
                    .get("min")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(default)
                    .max(1e-4);
                self.measure_thickness(threshold);
            }
            "view.toggle_visibility" => self.toggle_selected_visibility(),
            "view.cycle_mode" => {
                self.view_mode = self.view_mode.next();
                // Which imported solids are drawn as triangles depends on the mode, and
                // nothing else about the part has changed — so this refreshes the buffers
                // without paying for a compile.
                self.refresh_drawn_meshes();
                // Said at the moment the question arises — this is the keystroke someone
                // presses *because* they want the cage — rather than left as a viewport that
                // is silently empty and looks broken.
                if self.view_mode.cage_is_editable() && !self.control_visible() {
                    self.say(cage_absence_reason(self));
                }
            }
            "view.frame_selection" => match self.framing_bounds() {
                Some((center, radius)) => self.camera.frame(center, radius),
                None => self.warn("Nothing to frame"),
            },
            "draw.plane" => self.set_drawing_plane(args),
            "part.set_material" => self.set_material_from_console(args),
            "part.critical" => self.critical_from_console(args),
            "part.verify" => self.verify_from_console(),
            "view.environment" => self.set_environment_from_console(args),
            "view.draft" => self.draft_from_console(args),
            "view.grid" => self.grid_from_console(args),
            "view.perf" => self.perf_from_console(args),
            "view.display" => self.display_from_console(args),
            // The View menu's settings that had no operation of their own: a tick box flipped a
            // field and nothing else could reach it, which P3 calls a bug -- the GUI could do
            // something the console could not (`interaction-spec.md` §1).
            "view.control_points" => {
                self.show_control_points = !self.show_control_points;
                self.say(if self.show_control_points {
                    "Every curve's control points are shown"
                } else {
                    "Only the selected curve's control points are shown"
                });
            }
            "view.view_cube" => {
                self.show_view_cube = !self.show_view_cube;
                self.say(if self.show_view_cube { "ViewCube shown" } else { "ViewCube hidden" });
            }
            "view.tool_panel" => {
                self.floating_panel = !self.floating_panel;
                self.say(if self.floating_panel {
                    "A pending block shows the floating tool panel"
                } else {
                    "A pending block opens in the list, its first number waiting to be typed"
                });
            }
            "view.idle_occlusion" => {
                if !self.display_quality.occlusion() {
                    self.warn(
                        "Display quality is Fast, which never traces occlusion or shadows — choose                          Balanced or Full first",
                    );
                } else {
                    self.idle_occlusion = !self.idle_occlusion;
                    self.traced.invalidate(traced::Quality::Final);
                    self.say(if self.idle_occlusion {
                        "Shadows and occlusion once the view is still"
                    } else {
                        "No shadows or occlusion, even when still"
                    });
                }
            }
            #[cfg(feature = "march-stats")]
            "view.march_stats" => self.capture_march_stats(&[240, 1080]),
            "file.exit" => self.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            "help.shortcuts" => self.show_shortcuts = true,
            "palette.open" => self.open_palette(None),
            "edit.repeat" => match self.last_repeatable() {
                Some(command) => self.repeat_last(command),
                None => self.warn("Nothing to repeat yet"),
            },
            "view.show_all" => {
                let hidden: Vec<NodeId> =
                    self.part_graph().roots().into_iter().filter(|id| !self.is_visible(*id)).collect();
                if hidden.is_empty() {
                    self.say("Nothing is hidden");
                } else {
                    for id in &hidden {
                        self.set_visible(*id, true);
                    }
                    self.say(format!("Showing {} hidden", hidden.len()));
                }
            }
            "help.about" => self.show_about = true,
            // The console's two built-ins, run from the palette or a menu as typed.
            "console.help" => self.run_command("help"),
            "console.clear" => self.console_log.clear(),
            "view.section" => self.toggle_section(),
            "view.section_flip" => {
                self.section.flipped = !self.section.flipped;
                self.traced.invalidate(traced::Quality::Final);
                self.say("Section flipped");
            }
            "file.new" => self.new_document(),
            // `open file="C:/in/part.fform"` opens it without the dialog; `open` alone asks.
            "file.open" => {
                let asked = args.get("file").and_then(|v| v.as_str()).map(std::path::PathBuf::from);
                self.open_document(asked);
            }
            "file.save" => {
                self.save_document(SaveAs::SamePath);
            }
            // `saveas file="C:/out/part.fform"` saves there without the dialog; `saveas` alone
            // asks, as it always did.
            "file.save_as" => {
                let mode = match args.get("file").and_then(|v| v.as_str()) {
                    Some(path) => SaveAs::To(std::path::PathBuf::from(path)),
                    None => SaveAs::AskForPath,
                };
                self.save_document(mode);
            }
            "script.run" => self.run_script_command(args),
            "check.expect" => self.expect_command(args),
            "view.snap" => self.snap_command(args),
            "file.import" => self.import_command(args),
            "file.import_obj" => self.import_obj(),
            "file.import_iges" => self.import_iges(),
            "file.import_step" => self.import_step(),
            "cage.refine" => self.refine_cage(args),
            "modify.split" => self.split_body(args),
            "construct.axis" => self.axis_command(args),
            _ => match self.registry.build(op, args, &self.doc) {
                Ok(cmd) => {
                    if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                        self.warn(e.to_string());
                    } else {
                        self.after_document_change();
                    }
                }
                Err(e) => self.warn(e.to_string()),
            },
        }
    }

    /// Records that the document now differs from the file, and returns nothing else — the
    /// rebuild paths call this rather than touching `revision`, so a new kind of edit cannot
    /// be added without passing through here.
    fn mark_changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.document_moved();
        // Not everything that adds something goes through `set_part_graph` — an import adds
        // curves and no nodes at all — but everything that changes the document says so here.
        self.stamp_new_nodes();
    }

    /// Says the document may have moved, so the overlay reads it again
    /// ([`Self::document_moves`]).
    ///
    /// **Called from the few places every change already passes through, not from each change:**
    /// marking an edit, writing the part's graph, handing out the graph or a control mesh to be
    /// written, redrawing the curves and rebuilding what is drawn. Each is there for a reason of
    /// its own that a missed call already shows — a file that does not ask to be saved, an edit
    /// undo cannot reach, a line left where it was, a picture left as it was — so a change that
    /// reaches none of them is wrong on screen before the readings are. The two accessors are the
    /// drags' way in, since a drag is recorded only when the hand comes off.
    fn document_moved(&mut self) {
        self.document_moves = self.document_moves.wrapping_add(1);
    }

    /// Whether there is anything in memory the file on disk does not have.
    fn is_dirty(&self) -> bool {
        self.revision != self.saved_revision
    }

    /// Called after the document has been written or read, so the two agree again.
    fn mark_saved(&mut self) {
        self.saved_revision = self.revision;
    }

    /// Anything that changes the document invalidates the derived meshes, and may have
    /// removed the geometry the current selection referred to.
    fn after_document_change(&mut self) {
        self.mark_changed();
        // **The curves are drawn from a copy, and this is where it is refreshed.** Undoing a
        // curve move reverted the document and left the drawn line exactly where it was, which
        // is indistinguishable from undo not working — and it was not only the move: every
        // command that touches a curve went the same way. One line, and it is here rather than
        // at each call site because "the document changed" is precisely when the drawing is
        // stale.
        self.rebuild_curve_display();
        let vertex_count = self.control().vertex_count();
        let face_count = self.control().face_count();
        self.selection.keep_valid_for(vertex_count, face_count);
        self.hover = Selection::None;
        self.rebuild_render_resources();
    }

    /// Drops whatever is picked on the control mesh, and the gesture that was using it.
    ///
    /// A selection names vertices and faces by index, and a refine renumbers both: the
    /// originals keep their indices but have moved under the vertex-point rule, and every face
    /// is new. A selection that survived would keep the manipulator on indices that now mean
    /// something else and move the wrong points without a word — the one thing in
    /// `cage-refine-spec.md` §7 that fails quietly. It is also why this runs on undo and redo:
    /// they put the other numbering back.
    fn forget_cage_selection(&mut self) {
        self.selection.clear();
        self.hover = Selection::None;
        self.move_state = None;
    }

    /// `refine`: one Catmull-Clark step on the cage the tools are editing.
    ///
    /// Through the registry like any document operation, so it is one entry on the undo stack
    /// and the console and the menu cannot differ — what this adds is the two things the
    /// registry cannot know. *Which* cage: the registry takes the only one and refuses to
    /// guess among several, but the app has an answer, the one on screen. And the selection,
    /// above.
    ///
    /// Refused while a surface is the thing being edited. `subdivide` is the closed cage's
    /// rule, a surface to follow is open, and refining "the cage" while a surface is
    /// highlighted would change something other than what is selected.
    fn refine_cage(&mut self, args: &Args) {
        if self.control != ControlMesh::Cage {
            self.warn(
                "Refine works on the cage, and a surface is selected — pick the cage's body first",
            );
            return;
        }
        let args = match args.get("cage") {
            Some(_) => args.clone(),
            None => args
                .clone()
                .with("cage", Value::Str(self.active_cage.to_string())),
        };
        let command = match self.registry.build("cage.refine", &args, &self.doc) {
            Ok(command) => command,
            Err(e) => {
                self.warn(e.to_string());
                return;
            }
        };
        if let Err(e) = self.stack.execute(&mut self.doc, command) {
            self.warn(e.to_string());
            return;
        }
        self.forget_cage_selection();
        self.after_document_change();
        let (points, faces) = (self.cage().vertex_count(), self.cage().face_count());
        self.say(format!(
            "Refined the cage: {points} control points, {faces} faces. The surface has not moved."
        ));
    }

    /// The cage the UI is editing.
    ///
    /// Panics only if the active id has been removed without updating it, which would be a
    /// bug in this file rather than bad input — every path that replaces the document sets
    /// `active_cage` in the same step.
    fn cage(&self) -> &Cage {
        self.doc
            .cage(self.active_cage)
            .expect("active_cage must name a cage in the document")
    }

    /// The mesh `mesh` names, if the document still holds it.
    fn mesh_of(&self, mesh: ControlMesh) -> Option<&QuadMesh> {
        match mesh {
            ControlMesh::Cage => self.doc.cage(self.active_cage).map(|c| &**c),
            ControlMesh::Surface(id) => self.doc.quad_meshes.get(&id),
        }
    }

    fn mesh_of_mut(&mut self, mesh: ControlMesh) -> Option<&mut QuadMesh> {
        // A cage's extent is a body's, which the pick box is sized to, on every frame of a drag.
        self.document_moved();
        match mesh {
            ControlMesh::Cage => self.doc.cage_mut(self.active_cage).map(|c| &mut **c),
            ControlMesh::Surface(id) => self.doc.quad_meshes.get_mut(&id),
        }
    }

    /// The mesh the cage tools are editing — see [`ControlMesh`].
    ///
    /// The cage when the chosen surface has gone, which lasts only until the start of the next
    /// frame: `refresh_surface_display` notices and hands the tools back to the cage properly.
    fn control(&self) -> &QuadMesh {
        match self.mesh_of(self.control) {
            Some(mesh) => mesh,
            None => self.cage(),
        }
    }

    /// Where `mesh` is drawn from its own coordinates. A cage can sit under a placement in the
    /// graph; a surface is not a node, and is drawn, picked and followed where its points are.
    fn mesh_placement(&self, mesh: ControlMesh) -> nalgebra::Similarity3<f64> {
        match mesh {
            ControlMesh::Cage => self.cage_placement(),
            ControlMesh::Surface(_) => nalgebra::Similarity3::identity(),
        }
    }

    fn control_placement(&self) -> nalgebra::Similarity3<f64> {
        self.mesh_placement(self.control)
    }

    /// Whether the control mesh is on screen at all — the cage by the rules
    /// `visible_cage_node` gives, a surface by its own eye.
    fn control_visible(&self) -> bool {
        match self.control {
            ControlMesh::Cage => self.visible_cage_node().is_some(),
            ControlMesh::Surface(id) => {
                self.doc.quad_meshes.contains_key(&id) && !self.hidden_surfaces.contains(&id)
            }
        }
    }

    /// Whether a click in the viewport picks on the control mesh: a mode that draws it, and it
    /// there to be drawn.
    fn control_editable(&self) -> bool {
        self.view_mode.cage_is_editable() && self.control_visible()
    }

    /// Hands the cage tools to another mesh.
    ///
    /// The selection goes with it — it names vertices by index, and index 3 of a surface is not
    /// index 3 of the cage — and the drawn control mesh is rebuilt from the new one. No compile:
    /// nothing about the field has changed, only which of its inputs is being pointed at.
    fn set_control(&mut self, mesh: ControlMesh) {
        if self.control == mesh {
            return;
        }
        self.control = mesh;
        self.selection.clear();
        self.hover = Selection::None;
        self.move_state = None;
        self.cage_mesh = CageMesh::from_cage(self.control());
        self.refresh_drawn_meshes();
    }

    /// Makes a surface the control mesh, from its row.
    fn edit_surface(&mut self, id: fieldform_scene::QuadMeshId) {
        let was = self.control;
        self.set_control(ControlMesh::Surface(id));
        // Whatever else was selected would otherwise keep the manipulator in Field mode, over
        // something that is not what was just clicked.
        self.selected_body = None;
        self.selected_curve = None;
        self.clear_point_selection();
        self.selected_block = None;
        self.section_selected = false;
        if was != self.control {
            let name = self.doc.surface_name(id).to_string();
            if self.view_mode.cage_is_editable() {
                self.say(format!(
                    "Editing {name}: click a point, edge or face and drag it, or use the handles"
                ));
            } else {
                self.say(format!(
                    "Editing {name}: its points are picked in Polygon or Overlay view (Tab)"
                ));
            }
        }
    }

    /// The command that records a finished move of `vertices` on `mesh`.
    fn positions_command(
        &self,
        mesh: ControlMesh,
        vertices: Vec<VertexId>,
        before: Vec<Point3<f64>>,
        after: Vec<Point3<f64>>,
    ) -> Box<dyn fieldform_ops::Command> {
        match mesh {
            ControlMesh::Cage => {
                Box::new(SetVertexPositions::new(self.active_cage, vertices, before, after))
            }
            ControlMesh::Surface(id) => {
                Box::new(SetSurfacePositions::new(id, vertices, before, after))
            }
        }
    }

    /// The selected body's place among the ones being drawn, which is the order the shader's
    /// union fold uses — so a pixel's `Hit::body` compares directly against it.
    ///
    /// Derived rather than stored, like every other read of the graph: a stored index goes
    /// stale the moment a body is added, removed or hidden. Counted without building the
    /// display graph, because this is read once a frame and that clones — the mistake that put
    /// a whole graph compile in the frame loop, made smaller.
    fn selected_body_index(&self) -> Option<u32> {
        let id = self.selected_body?;
        // Read from `drawn_bodies` rather than recounted here. The two used to be the same
        // walk written twice, which was safe while the order was simply `roots()`; a drawn
        // variable is a copy under a random id, so it is not any more.
        self.drawn_bodies()
            .iter()
            .position(|drawn| *drawn == id)
            .map(|i| i as u32)
    }

    /// The part's graph with the hidden bodies taken out — what the viewport draws, picks
    /// against, and sizes its bake to: [`fieldform_field::display_graph_with_stand_ins`], told
    /// which bodies this application is drawing as a mesh.
    fn display_graph_with_stand_ins(&self) -> (Graph, std::collections::HashMap<NodeId, NodeId>) {
        fieldform_field::display_graph_with_stand_ins(&self.doc, self.part_graph(), &|id| {
            self.drawn_as_mesh(id)
        })
    }

    /// What the viewport draws, and nothing else. Most callers want only this.
    fn display_graph(&self) -> Graph {
        self.display_graph_with_stand_ins().0
    }

    /// The bodies the viewport draws, in the order it draws them, named by the block each is.
    ///
    /// **The one reading of that order.** `selected_body_index` turns it into the number the
    /// shader is given to highlight one, and `display_graph` is what actually gets drawn -- so
    /// the two have to be the same list or the wrong body lights up. It used to be safe to
    /// recount it by hand because the order was just `roots()`; it is not any more, because a
    /// drawn variable is a *copy* under a new id and **ids are random, not sequential**
    /// (`id.rs` says why). Counting the copies as "after the bodies" looked obviously right and
    /// would have highlighted the wrong body at random.
    ///
    /// Through `Document::folded_bodies`, which is the same function the shader's fold asks, so the
    /// index handed to the shader and the index the shader assigned are one number by
    /// construction rather than by two sorts agreeing. A stand-in has no stamp of its own --
    /// its id is derived, not minted -- and sorts where the block it stands for sorts, which is
    /// `folded_bodies`'s job and stated there.
    fn drawn_bodies(&self) -> Vec<NodeId> {
        // **Nothing shown, nothing to build.** This is read once a frame, and building the
        // display graph clones the part's -- which is what the note in `selected_body_index`
        // has always asked callers not to do per frame. With no variable on screen the drawn
        // list is exactly the roots that pass the two exclusions, which needs no clone and is
        // what this cost before the feature existed. Every document that does not use it pays
        // about what it paid.
        //
        // That holds only while each body's eye is a lookup, so the eye is read from the part's
        // consumed set built once here, never asked of the graph a body at a time
        // (`is_visible_among`): one more pass over the graph a frame, where asking per body was
        // a pass per body. The flags are the part's in both branches, as they always were.
        let shown = self.shown_ingredients();
        let part = self.part_graph();
        let consumed = part.consumed();
        if shown.is_empty() {
            // **A body with an empty slot is pruned, the fourth term of the same predicate**
            // (`notebook-spec.md` §7.2, the `is_reference_body` precedent: one body that
            // cannot compile used to take the whole viewport with it). Said here as well as in
            // `display_graph_with_stand_ins` because this is the hand-written copy of that
            // predicate, and the two lists have to be the same list or the wrong body lights
            // up -- the copy is the whole hazard, and it has already been wrong once: this term
            // was added here while the predicate it copies still had three. Closed downstream by
            // `incomplete`, so a block reading a holed one goes with it; one pass over the graph
            // a frame, beside the `consumed` one above.
            let holed = part.incomplete();
            return self
                .doc
                .folded_bodies(part)
                .into_iter()
                .filter(|id| {
                    is_visible_among(&self.doc, part, &consumed, *id)
                        && !self.is_reference_body(*id)
                        && !self.drawn_as_mesh(*id)
                        && !holed.contains(id)
                })
                .collect();
        }
        let _ = shown;
        let (graph, stand_ins) = self.display_graph_with_stand_ins();
        self.doc
            .folded_bodies(&graph)
            .into_iter()
            .filter(|id| {
                // A stand-in is drawn because it was asked for; the exclusions below are about
                // bodies, and the block it stands for has already passed them.
                stand_ins.contains_key(id)
                    || (is_visible_among(&self.doc, part, &consumed, *id)
                        && !self.is_reference_body(*id)
                        && !self.drawn_as_mesh(*id))
            })
            .map(|id| stand_ins.get(&id).copied().unwrap_or(id))
            .collect()
    }

    /// Gathers the imported meshes the tracer is *not* drawing, as triangles.
    ///
    /// A mesh the tessellator left holes in cannot be a signed field — there is no inside for
    /// a sign to refer to — so it is left out of the display graph and would not appear at all
    /// without this. Seeing what arrived is most of why it was imported.
    ///
    /// **A closed one is left out here in field mode, and used to be included.** The reason
    /// given was that an import should look the same whether or not it is solid — but it did
    /// not: a closed mesh is in the display graph, so the tracer draws it *and* this drew a
    /// translucent copy on top of it. Two descriptions of one solid, the second washing out
    /// the first at 75% alpha, and every show/hide rebuilding and re-uploading a buffer of
    /// every triangle in the document to draw it.
    ///
    /// In polygon mode nothing is traced, so there it is the only way a closed import appears
    /// and it is gathered as before.
    ///
    /// Placements are baked into the vertices rather than carried as a matrix, because several
    /// imports can sit under different ones and a single MVP cannot say that.
    fn rebuild_imported_mesh(&mut self) {
        let traced = self.view_mode != ViewMode::Polygon;
        let graph = self.part_graph().clone();
        let mut positions: Vec<Point3<f64>> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        // Once for every root below, which is every rebuild (`is_visible_among`).
        let consumed = graph.consumed();
        for root in graph.roots() {
            if !is_visible_among(&self.doc, &graph, &consumed, root) {
                continue;
            }
            // Down through any placements to the mesh itself, so a placed import draws where
            // it is rather than where it was built.
            let mut id = root;
            let mesh = loop {
                match graph.node(id) {
                    Some(Node::Mesh(mesh)) => break Some(*mesh),
                    // An empty slot ends the walk the same way a block that is not a mesh
                    // does: there is no leaf under a slot nothing has been put in.
                    Some(Node::Transform { input, .. }) => match input.source() {
                        Some(next) => id = next,
                        None => break None,
                    },
                    _ => break None,
                }
            };
            let Some(mesh) = mesh.and_then(|m| self.doc.mesh(m)) else { continue };
            // Already on screen, drawn per pixel by the thing that owns the surface.
            if traced && mesh.is_closed() {
                continue;
            }
            let place = self.placement_above(id);
            let base = positions.len() as u32;
            positions.extend(
                mesh.positions.iter().map(|p| place * Point3::new(p[0], p[1], p[2])),
            );
            triangles.extend(mesh.triangles.iter().map(|t| [base + t[0], base + t[1], base + t[2]]));
        }
        self.imported_mesh = CageMesh::from_triangles(&positions, &triangles);
    }

    /// Whether this block is drawn over the model rather than traced as a solid.
    ///
    /// A plane's square and a point's dot. Neither has a surface to trace — a half-space fills
    /// the screen and a point encloses nothing — so both are painted by the viewport overlay,
    /// and what takes one off screen is the `hidden` flag that overlay reads. Which is why the
    /// eye on their rows means `hidden` even when something consumes them: they are on screen
    /// either way, so "draw this as well" would be answering a question nobody asked.
    fn draws_as_overlay(&self, id: NodeId) -> bool {
        draws_as_overlay_in(self.part_graph(), id)
    }

    /// Whether this body of the active part is reference geometry rather than a part:
    /// [`fieldform_field::is_reference_body`], which says why.
    fn is_reference_body(&self, id: NodeId) -> bool {
        fieldform_field::is_reference_body(&self.doc, self.part_graph(), id)
    }

    /// The flag this block's eye reads and writes, and never anything else
    /// (`block-ownership-spec.md` §7.2).
    ///
    /// `hidden` while it is a body, since a body draws unless hidden; `shown` once something
    /// consumes it, since an ingredient draws only when asked for. So a body's eye goes off as it
    /// is consumed and comes back as it is let go of, which is what has a union draw rather than
    /// its parts. And `hidden` for a plane or a point whatever consumes it, for the reason
    /// [`Self::draws_as_overlay`] gives.
    ///
    /// **One rule, because the eye spoke to the two flags from several places that each chose
    /// one.** The H key wrote `hidden` on a consumed block, which nothing draws by until the block
    /// is let go of, while the row beside it wrote `shown`. Where a block sits in the list is not
    /// asked: neither flag moves a row, and nothing here reads ownership.
    fn eye_flag(&self, id: NodeId) -> EyeFlag {
        eye_flag_in(self.part_graph(), id)
    }

    /// Whether a block is drawn, as its eye says: a body unless hidden, a block something
    /// consumes only when shown.
    ///
    /// Everything that decides what draws reads the flags through this, so the eye and the
    /// picture cannot disagree about which flag counts.
    fn is_visible(&self, id: NodeId) -> bool {
        is_visible_in(&self.doc, self.part_graph(), id)
    }

    /// Draws a block or stops drawing it, through the flag its eye means: **the one write every
    /// eye makes**, and an undo step (`fieldform_ops::SetVisible`).
    ///
    /// The flag is asked of the graph as it is now. An operation that consumes a body and wants it
    /// kept on screen does not call this after its graph edit, which would be an entry of its own
    /// that a redo of the edit does not reach: it hands [`Self::set_part_graph_with`] the
    /// [`eye_command`] asked of the graph it writes, and reaches `shown` in the edit's own entry.
    fn set_visible(&mut self, id: NodeId, on: bool) {
        self.set_visible_all(&[id], on);
    }

    /// [`Self::set_visible`] for several blocks as one undo step and one rebuild: the import's
    /// eye, where forty bodies turned off by one click come back with one Ctrl+Z.
    ///
    /// A block already as asked is left out, so a click that changes nothing leaves no entry.
    fn set_visible_all(&mut self, ids: &[NodeId], on: bool) {
        let graph = self.part_graph();
        let mut commands: Vec<Box<dyn fieldform_ops::Command>> = ids
            .iter()
            .filter(|id| is_visible_in(&self.doc, graph, **id) != on)
            .map(|id| Box::new(eye_command(graph, *id, on)) as Box<dyn fieldform_ops::Command>)
            .collect();
        let cmd: Box<dyn fieldform_ops::Command> = match commands.len() {
            0 => return,
            1 => commands.remove(0),
            _ => Box::new(fieldform_ops::Batch::new("block.set_visible", commands)),
        };
        // Not a graph edit, so it ends any gesture a value drag had open rather than folding into
        // it, as a rename does: undoing the drag must not take the eye with it. **All three**, as
        // undo ends them: a material or an output block's drag folds by replacing the top of the
        // history, so its next step inside the window would have replaced this entry, leaving the
        // eye's change in the document with nothing on the stack to take it back.
        self.gesture = None;
        self.material_gesture = None;
        self.outputs.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
        // **Every eye rebuilds, for one reason.** This changes what is drawn, picked and baked,
        // and none of it happens until the render resources are built again. Without it the
        // flag flipped and the picture did not, until some *other* edit rebuilt for its own
        // reasons -- which reached the owner as an eye that worked only after toggling a
        // different block a couple of times. One of the two eyes this replaced had left it out.
        self.rebuild_render_resources();
    }

    /// The named blocks something else consumes that are drawn as well, oldest first.
    ///
    /// **One reading, because two would disagree.** `display_graph` draws these and
    /// `selected_body_index` counts them, and the count is what tells the shader which body to
    /// highlight -- so a body counted against a different list highlights the wrong one, which
    /// is the failure the comment in `selected_body_index` has always warned about.
    fn shown_ingredients(&self) -> Vec<NodeId> {
        shown_ingredients_in(&self.doc, self.part_graph())
    }

    /// Where the cage sits in the part, as against where it stores itself.
    ///
    /// The cage keeps its own coordinates — it is the direct layer, and they are the truth —
    /// while a [`Node::Transform`] anywhere above it moves the *field* those coordinates
    /// describe. This is the one place the two are reconciled, and everything that puts the
    /// cage on screen or reads a click off it goes through here, so that what is drawn, what
    /// is picked, and where the manipulator sits are the same points rather than three
    /// descriptions that can drift apart.
    ///
    /// The identity for a cage nothing has placed, which is the ordinary case.
    fn cage_placement(&self) -> nalgebra::Similarity3<f64> {
        let graph = self.part_graph();
        let Some(target) = graph
            .nodes()
            .find(|(_, n)| matches!(n, Node::Cage(c) if *c == self.active_cage))
            .map(|(id, _)| id)
        else {
            return nalgebra::Similarity3::identity();
        };
        self.placement_above(target)
    }

    /// Every placement standing above `target`, composed: where the block sits and which way it
    /// is turned, after everything above it.
    ///
    /// A node's own numbers say where it was built; the placements above it say where it is
    /// drawn. Anything that positions something in the viewport has to ask this or it draws
    /// the first and calls it the second — the disagreement the cage hit first, and then the
    /// point marker and the plane preview hit again the moment the block list let an *inner*
    /// node be selected.
    ///
    /// **One walk, and it is the kernel's.** This used to be a second implementation that read
    /// the unexpanded graph and composed each block's own turn as it went -- the same idea said
    /// twice, which D9 exists to stop and which cost three rounds of one bug being fixed in one
    /// copy and not the other. Asked for directly: "please make sure we are re-using code where
    /// possible -- it feels like some bugs are repeating."
    ///
    /// The two differed only in whether turns had been expanded, and that difference is itself
    /// one function: a turned block *is* that block inside a placement. So the graph goes
    /// through `with_turns_flattened` and the kernel's walk answers -- which also means the
    /// square drawn for a plane and the field built from it cannot come to different answers
    /// about where it is, because there is nothing left to disagree.
    fn placement_above(&self, target: NodeId) -> nalgebra::Similarity3<f64> {
        self.placement_in(&self.placed_graph(), target)
    }

    /// The graph with every turn rewritten as the placement it means, for
    /// [`Self::placement_above`]'s single block. The overlay's points and planes are placed by the
    /// one its readings keep ([`Self::overlay_readings`]), since making it clones the graph.
    fn placed_graph(&self) -> Graph {
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::with_turns_flattened(&self.doc, self.part_graph(), &numbers)
    }

    /// [`Self::placement_above`], over a graph whose turns are already placements.
    fn placement_in(&self, placed: &Graph, target: NodeId) -> nalgebra::Similarity3<f64> {
        guard_stack("App::placement_above");
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::placement_above(&self.doc, placed, target, None, &numbers)
    }

    /// Which point block the selection is, whether it is the point or a placement holding one.
    ///
    /// A `Node::Point` encloses no volume, so the field tracer draws nothing for it and there
    /// would otherwise be no sign of it in the viewport at all — a block in the list and empty
    /// space beside it. [`Self::point_positions`] is what gives every one of them a position
    /// you can see; this says which of them is being edited.
    fn selected_point_node(&self) -> Option<NodeId> {
        let mut id = self.selected_body?;
        // Through any placements wrapping it: what is selected is the outermost node.
        loop {
            match self.part_graph().node(id)? {
                Node::Point { .. } => return Some(id),
                // `?`, so an empty slot answers what the arm below answers: there is no point
                // under a hole.
                Node::Transform { input, .. } => id = input.source()?,
                _ => return None,
            }
        }
    }

    /// What the overlay and the panels read off the active part: every point and plane that draws,
    /// the pick box and every parameter's value ([`fieldform_field::OverlayReadings`]), read once
    /// and kept until something they read moves.
    ///
    /// **Kept, because in most frames nothing they read has moved.** An orbit moves the camera and
    /// nothing else, and read afresh, the pick box asked every drawn body for its extent on every
    /// frame of one: 2.0 ms a frame for one sweep, 17 ms for a wing of nine blocks
    /// (`fieldform-io/examples/overlay_cost.rs`). Asked through [`Self::overlay_key`], which names
    /// everything they read but the document, and the count of the document's moves.
    fn overlay_readings(&self) -> std::rc::Rc<fieldform_field::OverlayReadings> {
        let key = self.overlay_key();
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        self.kept_readings.borrow_mut().get(key, |key| key.read(&self.doc, &numbers))
    }

    /// The application's view of the active part as the overlay reads it, with the count of the
    /// document's moves: what [`Self::overlay_readings`] is kept against.
    fn overlay_key(&self) -> OverlayKey {
        // Asked of the part's bodies only while something is drawn as a mesh, which is rarely, so
        // most frames make this without walking the graph.
        let meshed = if self.outputs.instead_of_all || !self.outputs.instead_of.is_empty() {
            let mut ids: Vec<NodeId> = self
                .part_graph()
                .roots()
                .into_iter()
                .filter(|id| self.drawn_as_mesh(*id))
                .collect();
            ids.sort();
            ids
        } else {
            Vec::new()
        };
        OverlayKey {
            moves: self.document_moves,
            part: self.active_part,
            resolution: self.mesh_resolution,
            // A joint goes unseen with its row, by the rule its row's eye reads: hiding the
            // profile hides everything in it, the points that hold it included.
            hidden_with_rows: points_hidden_with_rows(
                &self.doc,
                self.active_part,
                &self.hidden_curves,
                &self.profiles.loops,
            ),
            meshed,
        }
    }

    /// What the block list reads off the active part: where each block is drawn, which blocks are
    /// bodies, and the order of the rows ([`fieldform_scene::ListReadings`]), read once and kept
    /// until the document moves or another part is shown.
    ///
    /// **Kept for the reason the overlay's readings are:** the list is drawn every frame, and in
    /// most frames nothing has moved. Read afresh, the order of the rows was worked out three times
    /// a frame — for the rows, and for the summary above them and its hover — and once more for
    /// every import's row, and every row asked the graph for its bodies
    /// (`fieldform-io/examples/list_cost.rs`).
    fn list_readings(&self) -> std::rc::Rc<fieldform_scene::ListReadings> {
        let key = ListKey { moves: self.document_moves, part: self.active_part };
        self.kept_list.borrow_mut().get(key, |key| key.read(&self.doc))
    }

    /// Every point block in the part that is not hidden, with where it is:
    /// [`fieldform_field::point_positions`], which says why every one is drawn, as the overlay's
    /// readings keep it.
    fn point_positions(&self) -> Vec<(NodeId, Point3<f64>)> {
        self.overlay_readings().points.clone()
    }

    /// Every plane block in the part that is not hidden, as a point on it and the way it faces:
    /// [`fieldform_field::plane_frames`], as the overlay's readings keep it.
    /// [`Self::selected_plane`] stays: the gizmo needs to know which one is being edited, which is
    /// a different question from which ones exist.
    fn plane_frames(&self) -> Vec<(NodeId, Point3<f64>, Rotation3<f64>)> {
        self.overlay_readings().planes.clone()
    }

    /// The selected body's plane, as the point it passes through and the direction it faces,
    /// both after any placements above it.
    ///
    /// A half-space has no edges, so the tracer fills the screen with it and there is nothing
    /// to see *of* it — which is true to the geometry and useless for aiming one. The preview
    /// this feeds is a finite stand-in for an infinite thing, drawn at a size taken from the
    /// scene rather than from the plane, since the plane has no size to take.
    fn selected_plane(&self) -> Option<(NodeId, Point3<f64>, Vector3<f64>)> {
        let mut id = self.selected_body?;
        loop {
            match self.part_graph().node(id)? {
                Node::Plane { origin, flipped, .. } => {
                    let (origin, flipped) = (*origin, *flipped);
                    let facing = if flipped { -1.0 } else { 1.0 };
                    // Composed from the roots down, so this is right whether the selection is
                    // the outermost placement or the plane inside it.
                    let place = self.placement_above(id);
                    return Some((
                        id,
                        place * self.resolve_place(&origin),
                        place.isometry.rotation * Vector3::z() * facing,
                    ));
                }
                // As above: nothing is under an empty slot, which is the `None` the arm below
                // gives for anything else.
                Node::Transform { input, .. } => id = input.source()?,
                _ => return None,
            }
        }
    }

    /// The curves whose control points are drawn and can be clicked: the selected one always,
    /// and every visible curve while View > Show control points is on -- asked for as "points
    /// should stay turned on", so a point can be picked without first finding its curve.
    fn handle_curves(&self) -> Vec<CurveId> {
        let mut out: Vec<CurveId> = Vec::new();
        if let Some(id) = self.selected_curve
            && !self.hidden_curves.contains(&id)
            && self.doc.curve(id).is_some()
        {
            out.push(id);
        }
        if self.show_control_points {
            let mut rest: Vec<CurveId> = self
                .doc
                .curves
                .keys()
                .copied()
                .filter(|id| Some(*id) != self.selected_curve && !self.hidden_curves.contains(id))
                .collect();
            rest.sort_by_key(|id| id.raw());
            out.extend(rest);
        }
        out
    }

    /// Every drawn handle's place on screen, curve by curve, the selected curve's first.
    fn handles_on_screen(&self, aspect: f32, rect: egui::Rect) -> Vec<(CurveId, usize, egui::Pos2)> {
        let mut out = Vec::new();
        for id in self.handle_curves() {
            let Some(c) = self.doc.curve(id) else { continue };
            for (i, p) in c.control_points().iter().enumerate() {
                if let Some(at) = self.camera.project_to_screen(aspect, rect, p.cast::<f32>()) {
                    out.push((id, i, at));
                }
            }
        }
        out
    }

    /// Which control point is under `pos`, if one is within reach: the nearest on any curve
    /// whose handles are drawn. The selected curve's come first, so a tie is theirs.
    fn control_point_at(
        &self,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Option<(CurveId, usize)> {
        const THRESHOLD: f32 = 10.0;
        let mut best: Option<((CurveId, usize), f32)> = None;
        for (id, i, at) in self.handles_on_screen(aspect, rect) {
            let distance = at.distance(pos);
            if distance <= THRESHOLD && best.is_none_or(|(_, best)| distance < best) {
                best = Some(((id, i), distance));
            }
        }
        best.map(|(hit, _)| hit)
    }

    /// Every picked control point, the selected curve's first.
    fn picked_points(&self) -> Vec<(CurveId, usize)> {
        let mut out: Vec<(CurveId, usize)> = match self.selected_curve {
            Some(id) => self.selected_points.iter().map(|i| (id, *i)).collect(),
            None => Vec::new(),
        };
        out.extend(self.other_points.iter().copied());
        out
    }

    fn is_picked(&self, curve: CurveId, index: usize) -> bool {
        (self.selected_curve == Some(curve) && self.selected_points.contains(&index))
            || self.other_points.contains(&(curve, index))
    }

    /// This point alone, its curve taking the selection.
    fn pick_only(&mut self, curve: CurveId, index: usize) {
        if self.selected_curve != Some(curve) {
            self.select_curve(curve);
        }
        self.selected_points = vec![index];
        self.other_points.clear();
    }

    /// Adds this point to what is picked, or takes it away -- on the selected curve or on
    /// another, which is what lets one selection span curves.
    fn toggle_point(&mut self, curve: CurveId, index: usize) {
        match self.selected_curve {
            None => self.pick_only(curve, index),
            Some(id) if id == curve => toggle_in(&mut self.selected_points, index),
            Some(_) => toggle_in(&mut self.other_points, (curve, index)),
        }
    }

    /// Makes `picked` the point selection, keeping the selected curve as the one that holds
    /// the handles when any point is on it.
    fn set_point_selection(&mut self, picked: &[(CurveId, usize)]) {
        match split_selection(picked, self.selected_curve) {
            Some((primary, indices, others)) => {
                if self.selected_curve != Some(primary) {
                    self.select_curve(primary);
                }
                self.selected_points = indices;
                self.other_points = others;
            }
            None => self.clear_point_selection(),
        }
    }

    fn clear_point_selection(&mut self) {
        self.selected_points.clear();
        self.other_points.clear();
    }

    /// Picks every drawn handle inside the box from `from` to `to` -- replacing what was
    /// picked, or adding to it -- and says how many. A box too small to have been meant
    /// changes nothing, since that press was a click.
    fn finish_marquee(
        &mut self,
        aspect: f32,
        rect: egui::Rect,
        from: egui::Pos2,
        to: egui::Pos2,
        adding: bool,
    ) {
        let bx = egui::Rect::from_two_pos(from, to);
        if bx.width() < 3.0 && bx.height() < 3.0 {
            return;
        }
        let inside = points_in_box(&self.handles_on_screen(aspect, rect), bx);
        let mut picked = if adding { self.picked_points() } else { Vec::new() };
        for p in inside {
            if !picked.contains(&p) {
                picked.push(p);
            }
        }
        self.set_point_selection(&picked);
        self.selection.clear();
        let n = self.picked_points().len();
        self.say(format!("{n} control point{} picked", if n == 1 { "" } else { "s" }));
    }

    /// The curve under the cursor, if one is close enough to have been aimed at.
    ///
    /// **A curve could not be picked in the viewport at all until this.** Control points are
    /// clickable, but only once their curve is selected, and a curve could only be selected from
    /// the block list — so the way to reach a curve you were looking at was to find its row in a
    /// panel, which is exactly backwards for the one kind of geometry you draw by hand.
    ///
    /// Measured against the *drawn* polyline rather than the control polygon, because the drawn
    /// line is what the click is aimed at. A degree-9 curve's control points can sit a long way
    /// from the curve itself, and picking against them would mean clicking empty space to select
    /// something and clicking the visible line to miss it.
    ///
    /// Hidden curves are not offered: a click cannot mean something you cannot see.
    fn curve_at(&self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) -> Option<CurveId> {
        // Generous, because a curve is one pixel wide and nobody can hit that. The same figure
        // a control point uses, which keeps the two feeling like one target.
        const THRESHOLD: f32 = 10.0;
        let mut best: Option<(CurveId, f32)> = None;

        for curve in self.doc.curves.values() {
            if self.hidden_curves.contains(&curve.id) {
                continue;
            }
            // At a tolerance proportional to the curve, so a big one is not tessellated into
            // thousands of segments for a hit test. Picking wants shape, not precision.
            let drawn = curve.tessellate((curve.length(0.5) * 1e-2).max(1e-3));
            let mut previous: Option<egui::Pos2> = None;
            for p in &drawn {
                let screen = self.camera.project_to_screen(aspect, rect, p.cast::<f32>());
                // A point behind the camera projects to nothing, and the segment leading to it
                // cannot be measured — so the chain restarts rather than joining across it.
                let Some(screen) = screen else {
                    previous = None;
                    continue;
                };
                if let Some(from) = previous {
                    let distance = distance_to_segment(pos, from, screen);
                    if distance <= THRESHOLD && best.is_none_or(|(_, b)| distance < b) {
                        best = Some((curve.id, distance));
                    }
                }
                previous = Some(screen);
            }
        }
        best.map(|(id, _)| id)
    }

    /// The end of another curve that `p` should snap to, if one is near enough.
    ///
    /// **Ends only, and never the curve's own.** A profile is closed by bringing two ends
    /// together, so those are the points worth landing on exactly; an interior control point is
    /// not somewhere anything meets. Excluding `moving`'s own curve is what stops a two-point
    /// line collapsing onto itself the moment it is dragged with the modifier down.
    ///
    /// The reach is a fraction of what is on screen rather than a fixed distance in millimetres,
    /// so it means the same thing on a bracket and on an airframe — the same reasoning `Home`
    /// already uses for framing.
    fn endpoint_near(
        &self,
        aspect: f32,
        rect: egui::Rect,
        cursor: egui::Pos2,
        moving: Option<CurveId>,
    ) -> Option<(CurveId, Point3<f64>)> {
        let mut best: Option<(CurveId, Point3<f64>, f32)> = None;
        for curve in self.doc.curves.values() {
            // `moving` is `None` for a curve still being placed: it has no end of its own yet,
            // so every end in reach is a target.
            if Some(curve.id) == moving || self.hidden_curves.contains(&curve.id) {
                continue;
            }
            let points = curve.control_points();
            let (Some(&first), Some(&last)) = (points.first(), points.last()) else {
                continue;
            };
            for end in [first, last] {
                let Some(screen) = self.camera.project_to_screen(aspect, rect, end.cast::<f32>())
                else {
                    continue;
                };
                let distance = screen.distance(cursor);
                if distance <= SNAP_PIXELS && best.is_none_or(|(_, _, b)| distance < b) {
                    best = Some((curve.id, end, distance));
                }
            }
        }
        best.map(|(id, at, _)| (id, at))
    }

    /// The end a snap would take right now, if the modifier is held and something is in reach.
    ///
    /// Shared by the drawing and by the drag, so the box cannot promise a snap the drag would
    /// not make: one question, asked once, answered in one place.
    fn snap_target(
        &self,
        aspect: f32,
        rect: egui::Rect,
        ui: &egui::Ui,
    ) -> Option<Point3<f64>> {
        if !ui.input(|i| i.modifiers.ctrl || i.modifiers.command) {
            return None;
        }
        let cursor = ui.input(|i| i.pointer.hover_pos())?;
        if !rect.contains(cursor) {
            return None;
        }
        // A curve being placed has no end of its own yet, so every end in reach is a target —
        // and the box is the same box, so what a click would take is what the drag would.
        if self.pending_curve.is_some() {
            return self.endpoint_near(aspect, rect, cursor, None).map(|(_, at)| at);
        }
        // Mid-drag the search runs from the moving point rather than the cursor, so the box has
        // to as well — a target drawn where the drag will not look is worse than none.
        let from = match &self.point_drag {
            Some(state) => self
                .doc
                .curve(state.curve)
                .and_then(|c| c.control_points().get(state.indices[0]).copied())
                .and_then(|p| self.camera.project_to_screen(aspect, rect, p.cast::<f32>()))
                .unwrap_or(cursor),
            None => cursor,
        };
        // Which curve is moving, so its own ends are not offered as targets. Mid-drag that is
        // the curve being dragged; otherwise it is whatever is selected, since that is what a
        // drag would move if one began now.
        let moving = match &self.point_drag {
            Some(state) => state.curve,
            None => self.selected_curve?,
        };
        // Only an end can be snapped, so only offer a target when an end is what would move.
        let count = self.doc.curve(moving).map(|c| c.control_points().len())?;
        let lone_end = self.other_points.is_empty()
            && match self.selected_points.as_slice() {
                [i] if *i == 0 || *i + 1 == count => true,
                _ => false,
            };
        if !lone_end {
            return None;
        }
        self.endpoint_near(aspect, rect, from, Some(moving)).map(|(_, at)| at)
    }

    /// The XYZ a block carries as its own position, when it has one and it is its own.
    ///
    /// **This is what tells a block that knows where it is from one that does not.** A sphere,
    /// a cuboid and a torus carry a centre; a cylinder, a cone and a capsule carry the pair of
    /// ends they are drawn between; a plane carries its origin and a point *is* one. A boolean,
    /// a lattice, a mesh and a cage carry none — their position is wherever their operands put
    /// them, so there is nothing here to write and a placement is the only way to move them.
    ///
    /// `None` too when any of those places is taken *from* another block. Such a place follows
    /// what it sits on, by design; overwriting it would quietly detach the block from the point
    /// it was put on, which is the opposite of what dragging it should mean.
    ///
    /// A `Node::Transform` is excluded although it carries a place: `about` is the pivot it
    /// turns around, not where it is. Writing a drag there would move the pivot and leave the
    /// body still.
    fn own_anchors(&self, id: NodeId) -> Option<Vec<Point3<f64>>> {
        let node = self.part_graph().node(id)?;
        if let Node::Point { at } = node {
            return Some(vec![self.resolve_point(at)]);
        }
        if matches!(node, Node::Transform { .. }) {
            return None;
        }
        let places = node.places();
        if places.is_empty() || places.iter().any(|p| p.source().is_some()) {
            return None;
        }
        Some(places.iter().map(|p| self.resolve_place(p)).collect())
    }

    /// The turn a block carries as its own, resolved.
    ///
    /// `None` for a block with no facing of its own, which is what sends a rotate back to the
    /// placement path.
    fn own_turn(&self, id: NodeId) -> Option<Rotation3<f64>> {
        let turn = self.part_graph().node(id)?.turn()?;
        // **Only a turn this block owns.** One aimed at a plane or a line is not the block's to
        // set: dragging the rotate handle would have to either break the aim or move the thing
        // it is aimed at, and neither is what the handle promises. `None` sends the drag to the
        // placement path, which wraps the block in one — the aim is kept, and the extra turn is
        // a block of its own that can be deleted.
        turn.angles()?;
        Some(self.aim_of(turn))
    }

    /// Which way a turn points, whether it is typed or taken from something.
    ///
    /// Through the kernel's `turn_rotation`, which is what both compilers read, so the square
    /// drawn for a plane and the field built from it cannot disagree about where it faces.
    fn aim_of(&self, turn: &Turn) -> Rotation3<f64> {
        guard_stack("aim_of");
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::turn_rotation(&self.doc, self.part_graph(), turn, &numbers)
    }

    /// Writes one back, as the degrees the block stores.
    fn set_own_turn(graph: &mut Graph, id: NodeId, turn: Rotation3<f64>) {
        let (x, y, z) = turn.euler_angles();
        if let Some(node) = graph.node_mut(id)
            && let Some(slot) = node.turn_mut()
            // An aimed turn is not written over: `own_turn` already refused to offer it, and
            // the two have to agree about which turns a drag may change.
            && slot.angles().is_some()
        {
            *slot = Turn::By([
                ParamRef::Literal(x.to_degrees()),
                ParamRef::Literal(y.to_degrees()),
                ParamRef::Literal(z.to_degrees()),
            ]);
        }
    }

    /// Which block a drag on `id` should actually move.
    ///
    /// **Itself, unless its position is somebody else's.** A block whose place is taken *from*
    /// another block has no coordinates of its own — the numbers in its panel are that block's,
    /// read through. So dragging it moves the block it sits on, and the numbers on screen are the
    /// numbers that change.
    ///
    /// The alternative was writing literals into the place, which updates the panel by cutting
    /// the link the panel is showing: the point stays in the list, nothing reads it, and the V
    /// that said "this is a variable" has quietly stopped being true. Reported as a sphere whose
    /// coordinates would not move once they had been made a variable.
    ///
    /// One place with one source, because that is the only case with one answer. A cylinder with
    /// its two ends on two different points has no single block to move, so it keeps the
    /// placement path — and a block sitting on a *plane* keeps it too, since a plane is a facing
    /// as much as a position and dragging one is not the same gesture.
    fn drag_moves(&self, id: NodeId) -> NodeId {
        let Some(node) = self.part_graph().node(id) else { return id };
        if matches!(node, Node::Transform { .. }) {
            return id;
        }
        let places = node.places();
        let [place] = places.as_slice() else { return id };
        match place.source() {
            Some(source)
                if matches!(self.part_graph().node(source), Some(Node::Point { .. })) =>
            {
                source
            }
            _ => id,
        }
    }

    /// Writes those positions back, as the literals the block stores.
    fn set_own_anchors(graph: &mut Graph, id: NodeId, points: &[Point3<f64>]) {
        let at_of = |p: &Point3<f64>| {
            [
                ParamRef::Literal(p.x),
                ParamRef::Literal(p.y),
                ParamRef::Literal(p.z),
            ]
        };
        let Some(node) = graph.node_mut(id) else { return };
        if let Node::Point { at } = node {
            if let Some(p) = points.first() {
                *at = at_of(p);
            }
            return;
        }
        for (slot, p) in node.places_mut().into_iter().zip(points) {
            *slot = fieldform_scene::Place::At(at_of(p));
        }
    }

    /// Selects a body, releasing whatever else was claiming the manipulator.
    ///
    /// One thing at a time, because the manipulator can only be on one of them and something
    /// has to decide which. Without this a curve selected in the panel — and left selected,
    /// which is the normal way to keep its points on screen — kept the handles forever: a
    /// click on a sphere updated the selection and changed nothing visible, so the sphere
    /// looked unselectable rather than second in a queue.
    fn select_body(&mut self, id: Option<NodeId>) {
        self.selected_body = id;
        // **Picking the cage's body is how the cage gets its tools back** from a surface — the
        // same gesture that makes it the thing being worked on everywhere else. Any other body
        // leaves a surface as the control mesh: clicking the lattice to look at it is not a
        // request to stop editing what it follows.
        let cage_node = self
            .part_graph()
            .nodes()
            .find(|(_, n)| matches!(n, Node::Cage(c) if *c == self.active_cage))
            .map(|(node, _)| node);
        if let (Some(body), Some(cage_node)) = (id, cage_node)
            && self.control != ControlMesh::Cage
            && (body == cage_node || self.body_containing(cage_node) == Some(body))
        {
            self.set_control(ControlMesh::Cage);
        }
        if id.is_some() {
            self.selected_block = None;
            self.selected_curve = None;
            self.clear_point_selection();
            // Picking geometry is a more deliberate act than leaving the section selected, so
            // it takes the manipulator. The section stays *on* — only its claim on the handles
            // goes, which is the distinction the flag exists to make.
            self.section_selected = false;
        }
    }

    /// The same, the other way round.
    fn select_curve(&mut self, id: CurveId) {
        self.selected_curve = Some(id);
        self.selected_block = None;
        // Indices name points on *a* curve, so they mean something else the moment the curve
        // does.
        self.clear_point_selection();
        self.selected_body = None;
        self.selection.clear();
        self.section_selected = false;
    }

    /// Whether `node` draws the cage the UI edits.
    ///
    /// The manipulator treats such a body as the direct layer rather than as something to
    /// place — see [`App::begin_gizmo`] for why that distinction has to exist.
    fn is_active_cage_node(&self, node: NodeId) -> bool {
        matches!(self.part_graph().node(node), Some(Node::Cage(c)) if *c == self.active_cage)
    }

    /// The body — the root — whose composition reaches `node`, if any still does.
    ///
    /// A node stops being a body the moment something consumes it, but it is still *part* of
    /// one, and for anything the user can see or hide it is that containing body that decides.
    fn body_containing(&self, node: NodeId) -> Option<NodeId> {
        let graph = self.part_graph();
        self.doc
            .listed_bodies(graph)
            .into_iter()
            .find(|root| graph.subtree_of(*root).contains(&node))
    }

    /// The node that draws the cage the UI edits, if the part still has one and it is shown.
    ///
    /// Searched across the whole graph rather than among the bodies, and that distinction is
    /// the whole point. A cage stops being a body as soon as a boolean consumes it — but it is
    /// still the thing being edited, and the reason to boolean it against something was to
    /// shape the result by pushing its points around. Looking only at the roots meant that the
    /// moment a cage joined anything it vanished from the viewport, could not be picked, and
    /// the manipulator would not appear for it: the edit that motivated the boolean became
    /// impossible the instant the boolean was made.
    ///
    /// Hidden still means hidden, in either of the two ways it can be. The cage's own node may
    /// be hidden; or the body that now contains it may be, and drawing a control mesh in front
    /// of a solid nobody can see is a wireframe floating in empty space.
    fn visible_cage_node(&self) -> Option<NodeId> {
        let id = self
            .part_graph()
            .nodes()
            .find(|(_, n)| matches!(n, Node::Cage(c) if *c == self.active_cage))
            .map(|(id, _)| id)?;
        // A node's own eye only means anything while it is a body, because a body row is the
        // only place an eye is drawn. Hide a cage, then boolean it, and the flag outlives the
        // row that set it: the cage is held hidden by a switch that no longer exists anywhere
        // in the interface. Honouring that is how the cage becomes unrecoverable.
        //
        // Once something consumes it, what decides is the body that contains it — which does
        // still have a row, and so is still something the user can act on.
        if self.part_graph().roots().contains(&id) && !self.is_visible(id) {
            return None;
        }
        // No containing body means nothing draws it at all, which is not the same as it being
        // hidden — but it is equally not on screen.
        self.body_containing(id)
            .filter(|body| self.is_visible(*body))
            .map(|_| id)
    }


    /// The composition being edited: the active part's graph.
    ///
    /// The app used to hold the applied generator in a field of its own and rebuild a graph
    /// from it for each consumer, which meant the thing on screen was not the thing in the
    /// document — so saving the document would have written out a part that was missing the
    /// composition the user was looking at. The graph in the document is the only copy now,
    /// and the UI derives its controls from it rather than the other way round.
    fn part_graph(&self) -> &Graph {
        &self
            .doc
            .parts
            .get(&self.active_part)
            .expect("active_part must name a part in the document")
            .graph
    }

    /// The active part's graph, to write during a gesture.
    ///
    /// Not a way round [`Self::set_part_graph`]: a drag moves the document live for feedback
    /// and puts *one* entry on the history when the hand comes off, which is what every drag
    /// here does, and what it settles at is committed from the graph this returns.
    fn part_graph_mut(&mut self) -> &mut Graph {
        self.document_moved();
        &mut self
            .doc
            .parts
            .get_mut(&self.active_part)
            .expect("active_part must name a part in the document")
            .graph
    }

    /// Puts the active part's graph back to `was`, off the history — a drag abandoned, or one
    /// put back to where it began so the command that commits it can redo it — and settles who
    /// owns what.
    ///
    /// **Every graph write settles** (`block-ownership-spec.md` §4.1), and these are the ones no
    /// command makes. What they put back differs from the live graph only in the numbers a drag
    /// moved, so the settle finds nothing to do today; it is here so that a restore which ever
    /// differs in structure cannot leave a block stored inside one that no longer reads it.
    fn restore_part_graph(&mut self, was: Graph) {
        *self.part_graph_mut() = was;
        self.doc.settle_ownership();
    }

     /// Replaces the active part's graph, and records it for undo.
    ///
    /// **The one gate every graph edit passes through**, which is what makes this the place to
    /// record them. Adding a body, combining two, unapplying an operation and dragging a radius
    /// all end up here, and until now none of them was undoable: Ctrl+Z reversed the last cage
    /// vertex you moved and skipped straight past everything else you had done since.
    ///
    /// A drag lands as **one** entry, not one per frame. A `DragValue` writes a new graph every
    /// frame it is held, and thirty entries for one gesture makes undo useless in the other
    /// direction — you would press it thirty times to get back where you started. So a
    /// continuation of a gesture replaces the entry describing the previous instant of it,
    /// keeping the graph the gesture *started* from as its "before".
    ///
    /// A continuation is: soon after the last edit, and **structurally identical** to it. The
    /// second half is what keeps the rule honest — dragging a number changes values, so the
    /// nodes and the edges between them are the same graph, while adding or deleting a body
    /// changes the shape and starts a new entry however fast you were clicking. Time alone
    /// would merge two deliberate acts that happened to be quick. And an undo or a redo ends the
    /// gesture, since after either the entry on top is not its own (`step_history`).
    fn set_part_graph(&mut self, graph: Graph) {
        self.set_part_graph_with(graph, Vec::new());
    }

    /// [`Self::set_part_graph`], with `then` in the same undo entry after the graph edit
    /// ([`record_graph_edit_then`]): for an act the graph edit is only half of, as the eye Pattern
    /// and Mirror turn on for the body they consume is (`block-ownership-spec.md` §7.3).
    ///
    /// The bookkeeping is every graph edit's, since this is one: the splash, the end of a material
    /// drag, and the stamp on what it made. With anything to follow, never the next instant of a
    /// gesture, and the end of one. With nothing, this is `set_part_graph` exactly, and a drag's
    /// next instant folds into the entry on top as that says: an empty `then` is not how an edit
    /// asks for an entry of its own.
    fn set_part_graph_with(&mut self, graph: Graph, then: Vec<Box<dyn fieldform_ops::Command>>) {
        // Any change to what the part is made of means the splash has been read.
        self.splash_shown = false;
        // A graph edit ends a material drag: the two must not fold into one undo entry, or
        // undoing the one would take back the other.
        self.material_gesture = None;
        // Every instant of a manipulator drag comes through here, and only its end is marked.
        self.document_moved();

        let part = self.active_part;
        // With whatever a field's V or × did to a parameter on the way here, so the two are one
        // undo (`ParameterEdits`).
        let done = record_graph_edit_then(
            &mut self.stack,
            &mut self.doc,
            &mut self.gesture,
            &mut self.parameter_edits,
            part,
            graph,
            then,
        );
        if let Err(e) = done {
            // The command only fails when the part is gone, which would mean the application
            // asked to edit something it had already removed. Said rather than swallowed.
            self.warn(e.to_string());
            return;
        }
        self.stamp_new_nodes();
    }

    /// Acts on what settling moved since this was last asked (`block-ownership-spec.md` §4.2):
    /// a block that became a variable with no name is given one, and what moved is said.
    ///
    /// **Here and not in the commands that settle**, because the name a variable should carry
    /// ("Cube", not "Cuboid") resolves parameters, which is the application's business, so
    /// settling writes no name. Asked at the top of every frame, so the list and the console have
    /// caught up before anything draws; after every line of a script, which runs its lines in one
    /// frame, so a line can name what the line before it made a variable; and before every save,
    /// so no file holds a variable that cannot be typed (I4).
    ///
    /// **Which blocks are named is `variables_to_name`'s**, in the order they were made
    /// (`unnamed_variables_among`), so of two spheres the first made is `Sphere`. **Off the
    /// history**: undoing the edit that made a variable leaves the name it was given, since a
    /// name is a label (§4.3).
    ///
    /// `quietly` after a load (§9.2): nothing is said and the document is not marked changed,
    /// since nothing moved and what is in memory is what was opened.
    fn drain_ownership_changes(&mut self, quietly: bool) {
        // Most frames: nothing settled since the last one, so nothing was declared either.
        if self.doc.pending_ownership_changes.is_empty() && !quietly {
            return;
        }
        let changes = std::mem::take(&mut self.doc.pending_ownership_changes);
        let unnamed = variables_to_name(&self.doc, quietly);
        for id in &unnamed {
            let name = self.unique_variable_name_in(self.graph_holding_or_active(*id), *id);
            self.doc.node_meta.entry(*id).or_default().name = Some(name);
        }
        if quietly {
            return;
        }
        if !unnamed.is_empty() {
            self.mark_changed();
        }
        let news = ownership_news(&self.doc, &changes, |block| match block {
            BlockKey::Node(id) => self.body_label_in(self.graph_holding_or_active(id), id),
            BlockKey::Curve(id) => {
                self.doc.curve(id).map_or_else(|| "A curve".to_string(), |c| c.name.clone())
            }
        });
        for line in news {
            self.say(line);
        }
    }

    /// The graph of whichever part holds `id`, or the active part's when none does: settling
    /// reaches every part, so what it moved need not be on screen.
    fn graph_holding_or_active(&self, id: NodeId) -> &Graph {
        self.graph_holding(id).unwrap_or_else(|| self.part_graph())
    }


    /// Sets the active part's material, folding a drag into one undo entry.
    ///
    /// `None` clears it back to the default. The same gesture rule `set_part_graph` uses: an
    /// edit within `GESTURE_WINDOW` of the last one continues it, so a slider drag goes on the
    /// history once, as the material it settled at, rather than sixty times a second.
    fn set_part_material(&mut self, material: Option<Material>) {
        let part = self.active_part;
        let current = self.doc.parts.get(&part).and_then(|p| p.material.clone());
        if current == material {
            return;
        }
        // And a material edit ends a graph gesture, for the same reason the reverse holds.
        self.gesture = None;
        let continues = self
            .material_gesture
            .as_ref()
            .is_some_and(|g| g.part == part && g.at.elapsed() < GESTURE_WINDOW);
        let before = match (&self.material_gesture, continues) {
            (Some(g), true) => g.before.clone(),
            _ => current,
        };
        self.material_gesture = Some(MaterialGesture {
            part,
            before: before.clone(),
            at: std::time::Instant::now(),
        });
        let cmd = Box::new(SetPartMaterial::new(part, before, material));
        let done = if continues {
            self.stack.replace_top(&mut self.doc, cmd)
        } else {
            self.stack.execute(&mut self.doc, cmd)
        };
        if let Err(e) = done {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
    }

    /// Everything a verify's answer depends on, in an order that does not vary: the blocks, the
    /// marks, and every number a block might read. Compared, never parsed.
    fn verification_fingerprint(&self) -> String {
        let graph = self.part_graph();
        let mut nodes: Vec<(NodeId, &Node)> = graph.nodes().collect();
        nodes.sort_by_key(|(id, _)| id.raw());
        format!("{nodes:?}|{:?}|{:?}", graph.critical(), named_values(self))
    }

    /// The last verify, if it still describes the part on screen.
    fn current_verification(&self) -> Option<&Verification> {
        let found = self.verification.as_ref()?;
        (found.part == self.active_part && found.fingerprint == self.verification_fingerprint())
            .then_some(found)
    }

    /// What to say beside a blend's radius about the surfaces it may be moving, if anything.
    ///
    /// Nothing at all for a part with no critical marks. With marks: the blame, if a current
    /// verify lays one on this blend; or that nobody has looked, which is not the same as fine.
    fn critical_note(&self, blend: NodeId) -> Option<(bool, String, String)> {
        if self.part_graph().critical().is_empty() {
            return None;
        }
        let Some(found) = self.current_verification() else {
            return Some((
                false,
                "unverified".to_string(),
                "This part has critical surfaces and has changed since they were last measured \
                 — or never was. Type `verify` in the console."
                    .to_string(),
            ));
        };
        let lines: Vec<String> = found
            .reports
            .iter()
            .filter(|r| r.blames(blend))
            .flat_map(|r| {
                let this = self.body_label(r.node);
                r.causes.iter().filter(|c| c.blend == blend).map(move |c| {
                    format!("moves \u{2018}{this}\u{2019} by up to {:.3} mm", c.movement)
                })
            })
            .collect();
        (!lines.is_empty()).then(|| {
            (
                true,
                lines.join("; "),
                format!(
                    "Measured by `verify` on {}: with only this fillet rounded, the critical \
                     surface sits that far from where the hard boolean puts it. Reduce the \
                     radius, or change what it joins.",
                    found.when
                ),
            )
        })
    }

    /// `critical node=<name> [tol=<mm>] [clearance=<mm>]`, `critical node=<name> -off`, or
    /// `critical` alone to list the marks.
    fn critical_from_console(&mut self, args: &Args) {
        let Some(wanted) = args.get("node").map(|v| match v.as_str() {
            Some(text) => text.to_string(),
            // A block called `7` parses as a number on the way in; it is still a name.
            None => v.as_f64().map(|n| n.to_string()).unwrap_or_default(),
        }) else {
            let marks = self.part_graph().critical().to_vec();
            if marks.is_empty() {
                self.say(
                    "No critical surfaces. Say `critical node=<block> tol=0.01` to mark one, \
                     then `verify`.",
                );
            }
            for mark in marks {
                self.say(format!(
                    "critical node={:?} tol={} clearance={}",
                    self.body_label(mark.node),
                    mark.tolerance,
                    self.resolve(mark.clearance),
                ));
            }
            return;
        };
        let Some(node) = self.node_named(&wanted) else {
            self.warn(format!("No block called {wanted:?} in this part"));
            return;
        };
        let mut graph = self.part_graph().clone();
        let off = args.get("off").is_some_and(|v| !matches!(v, Value::Bool(false)));
        if off {
            if !graph.unmark_critical(node) {
                self.warn(format!("{wanted:?} is not marked critical"));
                return;
            }
            self.set_part_graph(graph);
            self.mark_changed();
            // A mark tapers every fillet near it, so taking one off changes the shape drawn.
            self.rebuild_render_resources();
            self.say(format!("{wanted:?} is no longer critical; fillets near it are whole again"));
            return;
        }
        // Said again, a mark keeps what it was not told to change.
        let mut mark = graph
            .critical()
            .iter()
            .find(|m| m.node == node)
            .cloned()
            .unwrap_or_else(|| fieldform_scene::Critical::new(node));
        if let Some(tol) = args.get("tol").or(args.get("tolerance")).and_then(|v| v.as_f64()) {
            if !(tol > 0.0 && tol.is_finite()) {
                self.warn("A tolerance is a positive number of millimetres");
                return;
            }
            mark.tolerance = tol;
        }
        if let Some(c) = args.get("clearance").and_then(|v| v.as_f64()) {
            mark.clearance = ParamRef::Literal(c.max(0.0));
        }
        let (tol, clearance) = (mark.tolerance, self.resolve(mark.clearance));
        graph.mark_critical(mark);
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!(
            "{wanted:?} is critical: every fillet now thins to nothing {}, and held means within \
             {tol} mm. Unverified until `verify` runs.",
            if clearance > 0.0 {
                format!("{clearance} mm short of it")
            } else {
                "at it".to_string()
            },
        ));
    }

    /// Sweeps the part's field for the fastest it changes and says whether that is what it
    /// promises. A detector rather than a proof: a grid sees a jump only if it is wider than the
    /// grid's own step, which is why the step is said with the answer.
    fn verify_the_promise(&mut self) {
        // The third of the three statements about the part, asking the one rule the other two
        // ask (`no_statement_while_a_slot_is_empty`). Asked of the *part* rather than of the
        // display graph, which is the point: the pruning is what would hide the hole from the
        // sweep.
        if let Some(why) = crate::app::no_statement_while_a_slot_is_empty(
            self.part_graph(),
            |id| self.body_label(id),
            "say whether the field keeps its promise",
        ) {
            self.warn(why);
            return;
        }
        let graph = self.display_graph();
        if graph.roots().is_empty() {
            return;
        }
        let Some((lo, hi)) = fieldform_field::part_extent(&self.doc, &graph).as_box() else {
            self.say("The part reaches to infinity, so there is no box to sweep for a promise.");
            return;
        };
        let tolerance = ((hi - lo).norm() * 1e-3).max(1e-4);
        let field = match GraphField::compile_graph(&self.doc, &graph, tolerance) {
            Ok(field) => field,
            Err(e) => {
                let why = self.explain_compile(&e);
                self.warn(format!("Could not verify the field: {why}"));
                return;
            }
        };
        // A margin, so a surface lying on the box's own face is swept from both sides.
        let margin = (hi - lo) * 0.02;
        let started = std::time::Instant::now();
        let found =
            fieldform_field::find_overstep(&field, (lo - margin, hi + margin), PROMISE_SAMPLES);
        let took = started.elapsed().as_secs_f64();
        let spacing = (hi + margin - (lo - margin)).norm() / PROMISE_SAMPLES as f64;
        match found {
            Some(over) => {
                let at = over.at;
                let body = field
                    .body_at(at, over.moved.max(spacing))
                    .map(|node| format!(" in \u{2018}{}\u{2019}", self.body_label(node)))
                    .unwrap_or_default();
                // **A step and a slope are the same fault and read differently.** Where the
                // refinement closed to nothing and the value still moved, a rate is a number
                // with eleven digits and no meaning; what is worth saying is that the value
                // steps at all. Where it settled at a finite span, the rate is the thing.
                let what = if over.across <= spacing * 1e-4 {
                    format!("steps {:.3} mm across nothing at all", over.moved)
                } else {
                    format!(
                        "changes {:.2} mm per mm ({:.3} mm across {:.3} mm)",
                        over.rate, over.moved, over.across
                    )
                };
                self.warn(format!(
                    "The field {what} at ({:.1}, {:.1}, {:.1}){body}, where it promises {:.2} \
                     per mm. A mesh of it has holes there and rays step through it: the mesher \
                     cuts cells away by that promise. Looked in {took:.1} s.",
                    at.x, at.y, at.z, over.allowed,
                ));
            }
            None => self.say(format!(
                "The field keeps its promise: nothing changes faster than it says, from a grid \
                 of {spacing:.1} mm in {took:.1} s. A fault between two of those points would \
                 not be seen.",
            )),
        }
    }

    /// `verify`: measures every critical surface of the active part and says what it found, in
    /// the sentences the panel and a transcript share.
    fn verify_from_console(&mut self) {
        // **The field's own promise first, because nothing else is worth checking if it is
        // broken.** Every consumer believes `lipschitz`: the mesher prunes a cell by it and the
        // tracer steps by it. A part that breaks it meshes with holes nowhere near anything that
        // looks wrong — which is how a conformal lattice on a spiked surface failed, at 2.86 mm
        // per millimetre where a distance promises 1.0. It takes no marks and applies to every
        // part, so it runs whether or not anything is critical.
        self.verify_the_promise();
        if self.part_graph().critical().is_empty() {
            self.say(
                "Nothing more to verify: no surface is marked critical. `critical node=<block>`.",
            );
            return;
        }
        let started = std::time::Instant::now();
        let reports =
            match fieldform_field::verify_critical(&self.doc, self.part_graph(), 1e-3) {
                Ok(reports) => reports,
                Err(e) => {
                    // Through `explain_compile`, because this refusal reaches a person like
                    // every other one: an empty slot's own message carries the holed block's
                    // id, and a raw id names nothing anybody can go and look at.
                    let why = self.explain_compile(&e);
                    self.warn(format!("Could not verify: {why}"));
                    return;
                }
            };
        let when = utc_date(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        );
        let mut lines: Vec<(bool, String)> = Vec::new();
        for report in &reports {
            let fine = report.verdict() == fieldform_field::Verdict::Held;
            for sentence in report.sentences(&|id| self.body_label(id)) {
                lines.push((fine, sentence));
            }
        }
        let held = reports
            .iter()
            .filter(|r| r.verdict() == fieldform_field::Verdict::Held)
            .count();
        for (fine, sentence) in lines {
            if fine { self.say(sentence) } else { self.warn(sentence) }
        }
        // What the marks are costing: the same surfaces with the protection taken off. Said
        // only when it is something, and as information — the part on screen is the one above.
        if let Ok(bare) = fieldform_field::verify_unprotected(&self.doc, self.part_graph(), 1e-3) {
            for report in bare.iter().filter(|r| r.verdict() == fieldform_field::Verdict::Moved) {
                for cause in &report.causes {
                    self.say(format!(
                        "Unprotected, blend \u{2018}{}\u{2019} would move \u{2018}{}\u{2019} by up \
                         to {:.3} mm; its fillet is given up there instead.",
                        self.body_label(cause.blend),
                        self.body_label(report.node),
                        cause.movement,
                    ));
                }
            }
        }
        self.say(format!(
            "Verified {when}: {held} of {} held, in {:.1} s. Any edit makes this stale.",
            reports.len(),
            started.elapsed().as_secs_f64(),
        ));
        self.verification = Some(Verification {
            part: self.active_part,
            fingerprint: self.verification_fingerprint(),
            when,
            reports,
        });
    }

    /// `material preset=<name>` from the console, or `material` alone to hear the choices.
    fn set_material_from_console(&mut self, args: &Args) {
        let names: Vec<&str> = Material::presets().iter().map(|p| p.name).collect();
        let Some(wanted) = args.get("preset").and_then(|v| v.as_str().map(str::to_string)) else {
            self.say(format!(
                "Presets: {}. Say `material preset=<name>`, or `preset=default` to clear.",
                names.join(", ")
            ));
            return;
        };
        if wanted.eq_ignore_ascii_case("default") || wanted.eq_ignore_ascii_case("none") {
            self.set_part_material(None);
            self.say("Material: default");
            return;
        }
        match Material::preset(&wanted) {
            Some(m) => {
                let name = m.preset_name().unwrap_or("custom");
                self.set_part_material(if m == Material::default() { None } else { Some(m) });
                self.say(format!("Material: {name}"));
            }
            None => self.warn(format!(
                "No material called {wanted}. Presets: {}",
                names.join(", ")
            )),
        }
    }

    /// Lights the part by another room, and says what the room is for.
    fn set_environment(&mut self, choice: fieldform_gpu::EnvironmentChoice) {
        if self.environment == choice {
            return;
        }
        self.environment = choice;
        // Said when it is chosen: the zebra in particular does nothing visible on clay, and
        // the one sentence that explains why is only interesting once it has been picked.
        self.say(format!("Environment: {} — {}", choice.label(), choice.hint()));
    }

    /// `environment name=<studio|showroom|zebra>` from the console, or `environment` alone to
    /// hear the choices.
    fn set_environment_from_console(&mut self, args: &Args) {
        let names: Vec<&str> = fieldform_gpu::EnvironmentChoice::ALL
            .iter()
            .map(|c| c.label())
            .collect();
        let Some(wanted) = args.get("name").and_then(|v| v.as_str().map(str::to_string)) else {
            self.say(format!(
                "Environments: {}. Say `environment name=<which>`.",
                names.join(", ")
            ));
            return;
        };
        match fieldform_gpu::EnvironmentChoice::from_name(&wanted) {
            Some(choice) => self.set_environment(choice),
            None => self.warn(format!(
                "No environment called {wanted}. Environments: {}",
                names.join(", ")
            )),
        }
    }

    /// The draft overlay as the lighting pass wants it, or `None` when it is off or its pull
    /// cannot be read — a line that has been deleted, a typed direction of zero length.
    fn draft_overlay(&self) -> Option<fieldform_gpu::Draft> {
        if !self.draft.enabled {
            return None;
        }
        // The same function an extrude's sweep is read by, so "along that line" means the same
        // direction here that it means there.
        let mut pull = fieldform_field::sweep_direction(&self.doc, &self.draft.pull, &|r| {
            Some(self.resolve(*r))
        })?;
        if self.draft.flipped {
            pull = -pull;
        }
        Some(fieldform_gpu::Draft::new(
            pull,
            self.draft.angle_deg as f32,
            self.draft.tolerance_deg as f32,
        ))
    }

    /// `draft` toggles the overlay; `draft angle=3 tolerance=1 vector=<curve>` sets it up and
    /// turns it on; `draft off` turns it off. `line=` is accepted for `vector=`.
    fn draft_from_console(&mut self, args: &Args) {
        let mut set_up = false;
        if let Some(angle) = args.get("angle").and_then(Value::as_f64) {
            self.draft.angle_deg = angle.clamp(0.0, 45.0);
            set_up = true;
        }
        if let Some(tolerance) = args.get("tolerance").and_then(Value::as_f64) {
            self.draft.tolerance_deg = tolerance.clamp(0.01, 10.0);
            set_up = true;
        }
        let named = args
            .get("vector")
            .or_else(|| args.get("line"))
            .and_then(|v| v.as_str().map(str::to_string));
        if let Some(name) = named {
            match self.doc.curves.values().find(|c| c.name.eq_ignore_ascii_case(&name)) {
                Some(c) => {
                    self.draft.pull = fieldform_scene::Sweep::Along { curve: c.id };
                    set_up = true;
                }
                None => {
                    self.warn(format!("No curve called {name} to pull along"));
                    return;
                }
            }
        }
        if args.get("off").is_some() {
            self.draft.enabled = false;
        } else {
            self.draft.enabled = set_up || !self.draft.enabled;
        }
        if self.draft.enabled {
            self.say(format!(
                "Draft analysis on: blue has at least {:.1}\u{b0} of draft along the pull, red \
                 has not, blurred {:.1}\u{b0} either side. Flip the pull for the other half.",
                self.draft.angle_deg, self.draft.tolerance_deg
            ));
        } else {
            self.say("Draft analysis off");
        }
    }

    /// The draft-angle overlay's controls: on or off, which way the mould pulls, the angle a
    /// face needs and how softly the line is drawn.
    fn draft_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.draft.enabled, "Draft analysis").on_hover_text(
                "Colour each face by whether it has enough draft to leave a mould pulled in \
                 the chosen direction: soft blue where it does, soft red where it does not \
                 (vertical, or undercut). The line between blurs over the tolerance, so a \
                 curved face shows a band rather than a contour.",
            );
            if self.draft.enabled
                && ui
                    .small_button(if self.draft.flipped { "Flipped" } else { "Flip" })
                    .on_hover_text("Pull the other way: the read for the other half of the mould.")
                    .clicked()
            {
                self.draft.flipped = !self.draft.flipped;
            }
        });
        if !self.draft.enabled {
            return;
        }
        let pull = self.draft.pull;
        self.draft.pull = self.direction_controls(
            ui,
            "draft_pull",
            "Pull:",
            pull,
            None,
            DRAFT_WORDS,
            &[],
            None,
        );
        ui.horizontal(|ui| {
            ui.label("Angle:");
            ui.add(
                egui::DragValue::new(&mut self.draft.angle_deg)
                    .speed(0.1)
                    .range(0.0..=45.0)
                    .suffix("\u{b0}"),
            )
            .on_hover_text("The draft a face needs, in degrees from vertical along the pull.");
            ui.label("Tolerance:");
            ui.add(
                egui::DragValue::new(&mut self.draft.tolerance_deg)
                    .speed(0.05)
                    .range(0.01..=10.0)
                    .suffix("\u{b0}"),
            )
            .on_hover_text(
                "How far either side of the angle the boundary blurs. Wider is softer on a \
                 curved face.",
            );
        });
    }

}

/// Distinct stamps for one batch of new things, in the order given.
///
/// **Free of the app because the property is worth asserting on its own**, the same arrangement
/// `operand_choices_in` and `pair_candidates_in` have: an `App` needs a window and a device before
/// it will exist, and what matters here is arithmetic. The property is that no two things in one
/// write share a stamp — a shared stamp orders them by nothing, exactly as their random ids did,
/// measured at the same rate by `fieldform-gpu/examples/body_order_probe.rs`.
fn stamps_from(now: u64, count: usize) -> Vec<u64> {
    (0..count).map(|i| now + i as u64).collect()
}

impl App {
    /// The clock the stamps are taken from, in microseconds since the epoch.
    ///
    /// Microseconds because a stamp two nodes share orders them by nothing — see
    /// `Document::created`, which carries the whole argument and the format consequence.
    fn stamp_clock() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    /// Records when anything new was made, so the block list can be in the order you made it.
    ///
    /// Done here rather than at each of the half-dozen places that add a node, because this is
    /// the one gate every graph edit already passes through — and a stamp that has to be
    /// remembered at each call site is a stamp that will be forgotten at the next one.
    ///
    /// Existing entries are never overwritten, so a node keeps the moment it was made however
    /// often the graph around it is rewritten. That is also what lets a path with an order of its
    /// own declare it first, through [`Self::stamp_in_order`], and leave this to fill the rest.
    ///
    /// **A distinct stamp per node, not one reading shared by the batch.** This used to write one
    /// clock reading to every fresh node, and bodies that tie fall back to their ids — which is
    /// the random order the whole arrangement exists to replace, measured at the same rate by
    /// `fieldform-gpu/examples/body_order_probe.rs`. A write that adds one node is the common
    /// case and is unaffected; a write that adds several used to be silently ordered by nothing.
    ///
    /// The batch is sorted by id before the offsets are handed out, which is arbitrary and is
    /// *not* nothing: the alternative is `HashMap` order, which is randomly seeded per process,
    /// so the same edit made twice would stamp two different orders and neither would be
    /// reproducible in a test. Arbitrary-but-fixed is what this can promise on its own; a path
    /// that knows better says so through [`Self::stamp_in_order`].
    fn stamp_new_nodes(&mut self) {
        let now = Self::stamp_clock();
        let mut fresh: Vec<NodeId> = self
            .part_graph()
            .nodes()
            .map(|(id, _)| id)
            .filter(|id| !self.doc.created.contains_key(id))
            .collect();
        fresh.sort_by_key(|id| id.raw());
        let stamps = stamps_from(now, fresh.len());
        for (id, at) in fresh.into_iter().zip(stamps) {
            self.doc.created.insert(id, at);
        }
        // Curves as well, and by the same clock. A curve is a block in the same list as the
        // bodies, so an import that arrives without a time sorts against them by its random id
        // — which is how an imported curve landed at the top of a tree it had just been added
        // to the bottom of.
        let mut fresh: Vec<CurveId> = self
            .doc
            .curves
            .keys()
            .copied()
            .filter(|id| !self.doc.curves_created.contains_key(id))
            .collect();
        fresh.sort_by_key(|id| id.raw());
        let stamps = stamps_from(now, fresh.len());
        for (id, at) in fresh.into_iter().zip(stamps) {
            self.doc.curves_created.insert(id, at);
        }
    }

    /// Stamps nodes in an order the caller knows and the graph does not.
    ///
    /// **For the one kind of path that has an order to state.** An import builds an ordered list
    /// and numbers it — `file (1)`, `file (2)` — and a re-import pairs the solids up *by that
    /// position* (`cad-import-block-spec.md` §4). The gate above cannot recover that: it reads
    /// the batch out of a `HashMap`, so it can promise each node a distinct stamp and nothing
    /// about their order. Called before the edit settles, this writes the order the file had, and
    /// the gate then fills in anything not named here.
    ///
    /// Not a duty spread across every call site, which is the thing `stamp_new_nodes`'s own
    /// comment rightly warns against: a path that forgets to call this still gets stamped, just
    /// arbitrarily, so forgetting costs an order rather than a stamp. The half-dozen paths with
    /// no order to state should keep saying nothing.
    ///
    /// Never overwrites, for the same reason the gate does not: a node keeps the moment it was
    /// made however often the graph around it is rewritten.
    fn stamp_in_order(&mut self, ids: &[NodeId]) {
        let now = Self::stamp_clock();
        for (id, at) in ids.iter().zip(stamps_from(now, ids.len())) {
            self.doc.created.entry(*id).or_insert(at);
        }
    }

    /// The same, for curves — an IGES import brings them in an order too.
    fn stamp_curves_in_order(&mut self, ids: &[CurveId]) {
        let now = Self::stamp_clock();
        for (id, at) in ids.iter().zip(stamps_from(now, ids.len())) {
            self.doc.curves_created.entry(*id).or_insert(at);
        }
    }

    /// Clears `hidden` on a block an operation has just consumed, so it draws if it is ever let go
    /// of.
    ///
    /// **Not the eye, and never to become it** (`block-ownership-spec.md` §7.2). Every operation
    /// that consumes a block calls this right after, and it names its flag rather than asking
    /// [`Self::eye_flag`]: the block is no longer a body, so the eye's flag is `shown` by then,
    /// and writing that would draw every operand beside its result. A body hidden by hand and
    /// then consumed therefore comes back visible when released.
    ///
    /// Off the undo stack, as it always was; the eye is on it. The flag lives in the document
    /// rather than the app, so hiding half an assembly survives a reopen.
    fn clear_hidden(&mut self, id: NodeId) {
        let was = self.doc.node_meta.get(&id).is_some_and(|m| m.hidden);
        if let Some(m) = self.doc.node_meta.get_mut(&id) {
            m.hidden = false;
        }
        if was {
            self.mark_changed();
        }
    }

    /// The name a body has been given, if it has been given one.
    fn body_name(&self, id: NodeId) -> Option<&str> {
        self.doc.node_meta.get(&id).and_then(|m| m.name.as_deref())
    }

    /// A name nothing else in the document is using, built from what the block already is.
    ///
    /// Derived rather than generic — `Sphere`, then `Sphere 2` — because the name goes straight
    /// into the From and With lists of every other block, and a list of `variable_1` through
    /// `variable_6` is a list you have to click through to read. The block's own description is
    /// the best first guess at what someone would have typed, and it stays editable.
    fn unique_variable_name(&self, id: NodeId) -> String {
        self.unique_variable_name_in(self.part_graph(), id)
    }

    /// The same, for a block in `graph`: settling reaches every part, and a block it made a
    /// variable in a part not on screen is described by that part's graph, not this one's.
    fn unique_variable_name_in(&self, graph: &Graph, id: NodeId) -> String {
        // The names blocks that exist hold, for the reason `set_body_name` gives.
        let in_use = names::in_use(&self.doc);
        let taken: std::collections::HashSet<&str> = in_use.iter().map(|(_, n)| n.as_str()).collect();
        let base = self.derived_label_in(graph, id);
        if !taken.contains(base.as_str()) {
            return base;
        }
        // **Suffixed with an underscore, not a space.** A variable's name is what an expression
        // refers to it by, so the invented one is spelled the way an identifier is spelled:
        // `Sphere`, then `Sphere_1`. It used to be `Sphere 2`, which read as prose in the one
        // place the text is a name rather than a label. Asked for by the owner while testing the
        // V handle, which is now the commonest way a name gets invented at all.
        //
        // Bounded: there cannot be more collisions than there are names already taken.
        (1..=taken.len() + 1)
            .map(|n| format!("{base}_{n}"))
            .find(|candidate| !taken.contains(candidate.as_str()))
            .unwrap_or(base)
    }

    /// A curve name nothing else in the document is using.
    ///
    /// **Because curve names are now read, not just displayed.** Every block that takes a curve
    /// offers them by name — a cylinder's ends, a pipe's spine, an extrude's profile — and until
    /// this every line was called "Line". Three of them made a dropdown of three identical rows,
    /// which is a list you have to click through to read. Same shape as `unique_variable_name`
    /// and the same reason.
    fn unique_curve_name(&self, base: &str) -> String {
        let taken: std::collections::HashSet<&str> =
            self.doc.curves.values().map(|c| c.name.as_str()).collect();
        if !taken.contains(base) {
            return base.to_string();
        }
        (2..=taken.len() + 2)
            .map(|n| format!("{base} {n}"))
            .find(|candidate| !taken.contains(candidate.as_str()))
            .unwrap_or_else(|| base.to_string())
    }

    /// Renames a curve, keeping the empty case meaning "back to what it was called".
    ///
    /// **One undo step, through [`fieldform_ops::SetName`], as renaming a body is.** `SetCurve`
    /// was the obvious command and the wrong one: it replaces the whole curve, so a text field
    /// writing it character by character would push an undo step per keystroke. `SetName` writes
    /// only the name, and the row's field writes it once, on Enter or when it loses focus. Curves,
    /// bodies and profile rows went onto the stack together, so no kind of name is undoable while
    /// another is not.
    fn rename_curve(&mut self, id: CurveId, name: &str) {
        self.set_name(fieldform_ops::Named::Curve(id), name);
    }

    /// Writes a name through the undo stack. See [`fieldform_ops::SetName`] for why a rename is
    /// an undo step at all.
    fn set_name(&mut self, what: fieldform_ops::Named, name: &str) {
        // A rename is not a graph edit, so it ends any gesture a value drag had open rather than
        // folding into it: undoing the drag must not take the name with it.
        self.gesture = None;
        let cmd = Box::new(fieldform_ops::SetName::new(what, name));
        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
    }

    /// Puts a block where `cmd` says, through the undo stack — make variable, put back inside, or
    /// a rename (`fieldform_ops::SetOwnership`). Answers whether it went.
    fn set_ownership(&mut self, cmd: fieldform_ops::SetOwnership) -> bool {
        // Not a graph edit, so it ends any gesture a value drag had open rather than folding into
        // it, as a rename always has: undoing the drag must not take this with it.
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(cmd)) {
            self.warn(e.to_string());
            return false;
        }
        self.mark_changed();
        true
    }

    /// Names a block, or clears the name when given an empty one. Answers whether the name stuck.
    ///
    /// Cleared rather than stored empty, so a name typed and then deleted leaves the body back
    /// where it started — described by what it *is* — rather than showing a blank row.
    ///
    /// **A name is a label, and writing one moves no row** (`block-ownership-spec.md` §5.3). It
    /// goes on the history as a `SetOwnership` carrying the ownership the block already has, so a
    /// rename and the V are one kind of entry. It used to be what lifted a block out of the block
    /// it was drawn inside, so a name typed for any reason moved the row. **A variable's name may
    /// not be cleared**, since other blocks type it to refer to it (I4): putting it back inside
    /// is the way to stop it being one, and the refusal says so.
    ///
    /// **A name two blocks share is refused, because a name is an address.** A reference field
    /// resolves typed text to one block (`reference_field`), so two blocks called `Sphere` make one
    /// of them unreachable by typing and the field quietly picks the other. Refused on the
    /// keystroke rather than repaired afterwards, which is the arrangement
    /// `set_parameter_expression` already uses for a cycle and for the same reason: the moment the
    /// name is typed is the only moment the person knows what they meant by it.
    ///
    /// Auto-suffixing was the alternative and is worse here — it renames under the person, so the
    /// block they just called `Shell` is called `Shell_1` and the name they typed belongs to
    /// something else. Invented names still suffix (`unique_variable_name`), because nobody chose
    /// those.
    ///
    /// One place, so every path that renames gets the same answer: the row's own edit, a name
    /// given in the console, and the halves a split names.
    fn set_body_name(&mut self, id: NodeId, name: &str) -> bool {
        let name = name.trim();
        if let Some(why) = self.body_name_refusal(id, name) {
            self.say(why);
            return false;
        }
        match rename_block(&self.doc, id, name) {
            Ok(cmd) => self.set_ownership(cmd),
            Err(why) => {
                self.say(why);
                false
            }
        }
    }

    /// Why block `id` may not be called `name`, or `None` where it may: [`Self::set_body_name`]'s
    /// refusals, askable first by a caller that has to warn rather than say -- a typed `rename`,
    /// which must stop a script where a row's field only has to tell the person.
    fn body_name_refusal(&self, id: NodeId, name: &str) -> Option<String> {
        let name = name.trim();
        // **An empty name is a name being cleared, and a clear never clashes.** Comparing it
        // like any other would match every block that has no name — `node_meta` holds entries for
        // `hidden` and `shown` too — so clearing one would report a clash with an unrelated block.
        //
        // **Among the blocks that exist**, not among the table's entries: a deleted block keeps
        // its entry so undo can bring the name back, and the first version read that entry as a
        // clash -- delete `ball`, type `'ball'` again, and the name was "taken by a sphere" nobody
        // could see (`names.rs`).
        if let Some(clash) = (!name.is_empty()).then(|| names::holder_of(&self.doc, name, id)).flatten() {
            let what = self.derived_label(clash);
            return Some(format!(
                "\u{201c}{name}\u{201d} is already taken by a {what} — names are what blocks refer \
                 to each other by, so two cannot share one"
            ));
        }
        rename_block(&self.doc, id, name).err().map(str::to_string)
    }

    /// The file a body came from, if it is an imported mesh.
    ///
    /// Down through any placements, so a body that has been moved is still the body that
    /// arrived — the same walk `rebuild_imported_mesh` makes for the same reason.
    fn body_source(&self, root: NodeId) -> Option<fieldform_scene::id::SourceId> {
        let graph = self.part_graph();
        let mut id = root;
        loop {
            match graph.node(id) {
                Some(Node::Mesh(mesh)) => return self.doc.mesh(*mesh).and_then(|m| m.source),
                Some(Node::Cage(cage)) => {
                    return self.doc.sources.values().find(|s| s.cages.contains(cage)).map(|s| s.id);
                }
                // Nothing imported is under an empty slot, which is the `None` anything else
                // already answers.
                Some(Node::Transform { input, .. }) => id = input.source()?,
                _ => return None,
            }
        }
    }

    /// The mesh a body draws, down through any placements.
    fn mesh_under(&self, root: NodeId) -> Option<&fieldform_scene::MeshData> {
        let graph = self.part_graph();
        let mut id = root;
        loop {
            match graph.node(id) {
                Some(Node::Mesh(mesh)) => return self.doc.mesh(*mesh),
                // The same end to the walk: no mesh is under a hole.
                Some(Node::Transform { input, .. }) => id = input.source()?,
                _ => return None,
            }
        }
    }

    /// The bodies one import produced, in the list's own order.
    fn source_members(&self, source: fieldform_scene::id::SourceId) -> Vec<NodeId> {
        self.ordered_nodes()
            .into_iter()
            .filter(|id| self.body_source(*id) == Some(source))
            .collect()
    }

    /// What a parameter reference is worth right now.
    ///
    /// An unresolvable one — a name the document does not hold, or an expression that fails to
    /// evaluate (`Document::parameter_value` answers `None` for both) — reads as zero *here
    /// only*, for sizing a bounding box. Evaluation refuses it rather than substituting anything,
    /// which is where that has to be caught.
    fn resolve(&self, r: ParamRef) -> f64 {
        guard_stack("App::resolve");
        r.resolve(&|id| {
            self.doc.parameter_value(id)
        })
        .unwrap_or(0.0)
    }

    /// A primitive's centre as a point, for sizing a box around it.
    fn resolve_point(&self, centre: &[ParamRef; 3]) -> Point3<f64> {
        Point3::new(
            self.resolve(centre[0]),
            self.resolve(centre[1]),
            self.resolve(centre[2]),
        )
    }

    /// Writes a number through a reference: into the literal, or into the parameter it names.
    ///
    /// Returns the reference to store, which only changes for a literal — a named one keeps
    /// its name and gets a new value, which is the whole point of naming it.
    fn set_value(&mut self, r: ParamRef, value: f64) -> ParamRef {
        match r {
            ParamRef::Literal(_) => ParamRef::Literal(value),
            ParamRef::Named(id) => {
                // Only a typed number takes the write. A computed one is left alone: the number
                // a drag would put here is the expression's own output, and overwriting the
                // expression with it would silently turn a relationship into a constant. The
                // field is disabled for a computed value, so nothing routes a drag here from
                // the panel; this is the guard for the manipulator.
                if let Some(param) = self.doc.parameters.get_mut(&id) {
                    if param.value.as_literal().is_some() {
                        param.value = fieldform_scene::ParamValue::Literal(value);
                        self.mark_changed();
                    }
                }
                r
            }
            // A driven reference has no single value to write. Nothing routes a drag here —
            // the panel shows a ramp's own controls instead of one number — so this is the
            // "leave it alone" answer rather than a case to handle.
            ParamRef::Field(_) => r,
        }
    }

    /// What to call a body in a list: what it is, not what it is called.
    ///
    /// Bodies have no names of their own. Naming them would mean a persistent per-node label
    /// in the file, and nothing yet needs one — the graph is small enough that "Pipe" and
    /// "Subtract" say which is which.
    fn body_label(&self, id: NodeId) -> String {
        self.body_label_in(self.part_graph(), id)
    }

    /// The same, for a block in `graph` — another part's, when a refusal names a block there.
    /// Asked of the part on screen, a block it does not hold reads "gone".
    fn body_label_in(&self, graph: &Graph, id: NodeId) -> String {
        guard_stack("body_label");
        // A given name wins. What a body *is* is the right answer until two of them are the
        // same thing — two pipes are two rows both saying "Pipe", and the only way to tell them
        // apart was to click one and watch the viewport.
        match self.body_name(id) {
            Some(name) => name.to_string(),
            None => self.derived_label_in(graph, id),
        }
    }

    /// What a block's slot reads, for a row that names it — or that it reads nothing yet.
    ///
    /// **One wording for the hole, because these rows are read by scanning the list's left
    /// edge.** A slot is empty while a block waits to be told what to read
    /// (`notebook-spec.md` §7.2), five of the labels below name what their block is applied
    /// to, and each spelling that state its own way would make one state look like five. The
    /// row is not the place to explain why it is empty: the block says, where the slot is.
    fn operand_label_in(&self, graph: &Graph, slot: &fieldform_scene::Operand) -> String {
        match slot.source() {
            Some(id) => self.body_label_in(graph, id),
            None => "nothing yet".to_string(),
        }
    }

    /// What a body is, ignoring any name it has been given.
    ///
    /// Kept separate because the name box needs it as its placeholder: the hint shows what
    /// clearing the box would give back, which is the only way that box can say what it does.
    fn derived_label(&self, id: NodeId) -> String {
        self.derived_label_in(self.part_graph(), id)
    }

    /// The same, for a block in `graph`, and what it is applied to looked up there too.
    fn derived_label_in(&self, graph: &Graph, id: NodeId) -> String {
        match graph.node(id) {
            Some(Node::Cage(_)) => "Cage".to_string(),
            // **"Pipe", asked for by name.** It read "Along <curve>", which was an attempt to
            // answer a different complaint -- a ramp offering "Pipe" as the field it reads says
            // the mechanism and not the intent. That put the input where every other block puts
            // the operation, which is the thing the rule below exists to stop. Which curve it
            // follows is in its own block, and naming the pipe is what puts a word of your own
            // on the row.
            Some(Node::Pipe { .. }) => "Pipe".to_string(),
            // **The operation, not the curve.** These used to read "Outline extruded", which
            // put the input where every other block puts what it does — so the same fact
            // appeared in two orders depending on the block, and the row could not be scanned
            // down its left edge. The profile is still shown: it is listed as an input, beside
            // every other curve the block names, which is where a pipe's curve already was.
            Some(Node::Extrude { .. }) => "Extrude".to_string(),
            Some(Node::Revolve { .. }) => "Revolve".to_string(),
            Some(Node::Sweep { .. }) => "Sweep".to_string(),
            Some(Node::Loft { .. }) => "Loft".to_string(),
            Some(Node::Sphere { .. }) => "Sphere".to_string(),
            // Named for the shape it is rather than for the menu entry that made it: three
            // equal sides is a cube, and anything else is a box. Stretching one in the panel
            // renames its own row, which is the only signal that it stopped being a cube.
            Some(Node::Cuboid { size, .. }) => {
                let [x, y, z] = size.map(|s| self.resolve(s));
                if (x - y).abs() < 1e-9 && (y - z).abs() < 1e-9 {
                    "Cube".to_string()
                } else {
                    "Box".to_string()
                }
            }
            Some(Node::SkewedBox { .. }) => "Skewed box".to_string(),
            Some(Node::Cylinder { .. }) => "Cylinder".to_string(),
            Some(Node::Cone { .. }) => "Cone".to_string(),
            Some(Node::Torus { .. }) => "Torus".to_string(),
            Some(Node::Plane { .. }) => "Plane".to_string(),
            Some(Node::Point { .. }) => "Point".to_string(),
            // Its own name, because an import is a thing the user brought in and will look for
            // by the name it arrived under.
            Some(Node::Mesh(id)) => self
                .doc
                .mesh(*id)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| "Missing mesh".to_string()),
            // Named for the cell, because that is what distinguishes one lattice from another.
            Some(Node::Lattice { cell, .. }) => lattice_cell_label(*cell).to_string(),
            Some(Node::Capsule { .. }) => "Capsule".to_string(),
            Some(Node::Boolean { op, .. }) => boolean_label(*op).to_string(),
            Some(Node::Blend { op, .. }) => format!("{} (filleted)", boolean_label(*op)),
            // The amount is in the name because it is the whole of what the block does, and a
            // list of three morphs that all read "Morph" says nothing about which is which.
            Some(Node::Morph { t, .. }) => {
                format!("Morph {:.0}%", self.resolve(*t).clamp(0.0, 1.0) * 100.0)
            }
            // **The operation first, then what it is applied to.** These read the other way
            // round -- "Morph 50% (attracted)" -- so the block's own identity was in the
            // brackets and the thing it was built on was the headline. Reported of an attractor
            // over a morph, and the same complaint the extrude arm above already records: the
            // same fact appeared in two orders depending on the block, and the list could not be
            // scanned down its left edge.
            //
            // What each one is applied to stays, in brackets, because a placement or a twist is
            // not a body of its own -- it is a body changed -- and losing which one would lose
            // the part somebody is looking for.
            Some(Node::Transform { input, .. }) => {
                format!("Placed ({})", self.operand_label_in(graph, input))
            }
            Some(Node::Twist { input, .. }) => {
                format!("Twist ({})", self.operand_label_in(graph, input))
            }
            Some(Node::Attract { input, .. }) => {
                format!("Attract ({})", self.operand_label_in(graph, input))
            }
            // **The number it builds, not the name of the variable behind it.** A row's label is
            // read by scanning the list's left edge, and `Pattern ×copies_per_hub` makes that a
            // thing to decode where `Pattern ×6` is a thing to see. The name is on the count's own
            // field, where it is being set. A count that will not resolve shows `×?` rather than a
            // number it is not building — the row is not the place to explain why, and the field says.
            Some(Node::Pattern { input, count, kind, join, .. }) => format!(
                "Pattern \u{d7}{} {} ({}){}",
                fieldform_scene::pattern_count(&self.doc, count)
                    .map_or_else(|_| "?".to_string(), |(n, _)| n.to_string()),
                match kind {
                    PatternKind::Linear { .. } => "in a row",
                    PatternKind::Radial { .. } => "around",
                },
                self.operand_label_in(graph, input),
                // Said on the row, because it is the difference between this block being all
                // of the pattern and being all but one of it -- and the body list is where a
                // person looks to find the one that is missing.
                if *join { "" } else { ", copies only" },
            ),
            Some(Node::Mirror { input, .. }) => {
                format!("Mirror ({})", self.operand_label_in(graph, input))
            }
            // **"Offset", which is what the menu calls it.** The row said "Grow" or "Shrink"
            // -- which way the number went -- so the menu entry somebody clicked and the row
            // that appeared did not share a word, and there was nothing to search the list for.
            // Reported as exactly that. Which way it went is in the number, one line inside the
            // block; whether it varies is not, so that much stays.
            Some(Node::Offset { input, distance }) => format!(
                "{} ({})",
                if distance.driver().is_some() { "Offset varying" } else { "Offset" },
                self.operand_label_in(graph, input),
            ),
            Some(Node::Shell { input, .. }) => {
                format!("Shell ({})", self.operand_label_in(graph, input))
            }
            // Never listed as a body — `roots` leaves it out — so this is only reached when a
            // panel names the node driving something.
            Some(Node::Ramp { .. }) => "Ramp".to_string(),
            Some(Node::Unknown(_)) => "Unreadable".to_string(),
            None => "gone".to_string(),
        }
    }

    // `visible_mesh` used to live here    // `visible_mesh` used to live here: it chose whichever mesh was on screen for pivot
    // picking to raycast. That choice was the bug. It made the pivot depend on the display
    // mode, and in traced mode it tested a coarse mesh against a surface the user was seeing
    // drawn per pixel, so clicks near a silhouette missed. `pick_pivot` traces the field
    // instead, which is the same in every mode, and nothing else ever wanted this.

    /// Draws the splash centred in the viewport, scaled to fit without distorting it.
    fn draw_splash(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let texture = match &self.splash {
            Some(texture) => texture,
            None => {
                // Embedded rather than read from disk: the binary is the thing that gets run,
                // often from somewhere other than the source tree, and a splash that depends
                // on the working directory is a splash that is usually missing.
                const BYTES: &[u8] = include_bytes!("../assets/splash-screen.png");
                let Ok(decoded) = image::load_from_memory(BYTES) else {
                    // A splash that will not decode is not worth a word to the user, but it
                    // should not be retried every frame either.
                    self.splash_shown = false;
                    return;
                };
                let rgba = decoded.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                self.splash.insert(ui.ctx().load_texture(
                    "splash",
                    image,
                    egui::TextureOptions::LINEAR,
                ))
            }
        };

        let native = texture.size_vec2();
        // Fits inside the viewport with a margin, and never enlarged past its own size — an
        // upscaled splash on a large monitor looks like a mistake.
        let scale = (rect.width() * 0.8 / native.x)
            .min(rect.height() * 0.8 / native.y)
            .min(1.0);
        let size = native * scale;
        let at = egui::Rect::from_center_size(rect.center(), size);
        egui::Image::new((texture.id(), size)).paint_at(ui, at);
    }

    /// Keeps the window title showing which document this is and whether it is saved.
    ///
    /// Sent only when the string changes: a viewport command every frame is a message to the
    /// windowing system sixty times a second to say nothing.
    fn sync_window_title(&mut self, ctx: &egui::Context) {
        let name = match &self.file {
            Some((path, _)) => file_name(path),
            None => "untitled".to_string(),
        };
        let marker = if self.is_dirty() { "*" } else { "" };
        let title = format!("{marker}{name} — fieldform");
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    /// Intercepts the window's close button while there are unsaved changes.
    ///
    /// The close is cancelled first and re-issued after the prompt, because the prompt is
    /// modal and the answer arrives too late to stop a close already in flight.
    ///
    /// The answer has to be *remembered*, which is the whole reason for the flag. Choosing to
    /// discard leaves the document dirty — that is what discarding means — so the close this
    /// method then issues arrives back here one frame later, meets the same unsaved changes,
    /// and is cancelled and asked about again. The window flickered and would not close, and
    /// only saving escaped it, by making the second pass see a clean document.
    fn guard_close(&mut self, ctx: &egui::Context) {
        if !ctx.input(|i| i.viewport().close_requested()) {
            return;
        }
        if self.closing || !self.is_dirty() {
            // Nothing to ask about, so this is the exit itself: the autosave goes with it.
            self.forget_autosave();
            return;
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        if self.may_discard("closing") {
            self.closing = true;
            self.forget_autosave();
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    /// Why the current graph does not evaluate, if it does not.
    ///
    /// Compiling is how this is answered rather than inspecting the graph, so the reason the
    /// user is given is the same one the viewport acted on.
    fn compile_error(&self) -> Option<String> {
        self.compile_error.clone()
    }

    /// A compile refusal, with any node id in it replaced by what the block is called.
    ///
    /// **The difference between a document that looks broken and one that says what to
    /// change.** A blend against a lattice is refused by D8, which is correct — the radius
    /// would not be the radius asked for — but the consequence is that the viewport traces
    /// nothing and the export dialog never opens, and neither of those looks like a refusal.
    /// Until this named the block there was no way to tell which of several was at fault.
    ///
    /// Shared by the panel and by export so the two cannot describe the same refusal
    /// differently, which is the failure this codebase keeps meeting: one question, answered
    /// in two places.
    fn explain_compile(&self, e: &fieldform_field::CompileError) -> String {
        match e {
            fieldform_field::CompileError::BlendNeedsDistanceFields(id) => {
                let name = self
                    .body_name(*id)
                    .map(str::to_string)
                    .unwrap_or_else(|| self.derived_label(*id));
                e.to_string()
                    .replace(&format!("{id:?}"), &format!("\u{201c}{name}\u{201d}"))
            }
            // The same substitution, for the same reason: the refusal reaches a person, and
            // "node NodeId(2f1c..)'s slot 0 is empty" names nothing they can go and look at.
            // The word `node` goes with the id -- the way `curve` does in the `BadWall` arm
            // below -- so what is left is the block's own name owning the slot. The slot number
            // stays: it is the half of the message that says which hole to fill, and the block
            // it names is the holed one rather than whatever reads it, because that is the one
            // with a slot to fill.
            fieldform_field::CompileError::EmptySlot { node, .. } => {
                let name = self
                    .body_name(*node)
                    .map(str::to_string)
                    .unwrap_or_else(|| self.derived_label(*node));
                e.to_string()
                    .replace(&format!("node {node:?}"), &format!("\u{201c}{name}\u{201d}"))
            }
            // **The other answer to a loop that does not close**, said beside the first. Until
            // a block could be built on a wall, joining the ends was the only way on; now
            // giving the curve a thickness is one too, and it is a checkbox on the block the
            // refusal is about (`outline-wall-spec.md` §6).
            fieldform_field::CompileError::BadLoop(
                _,
                fieldform_field::LoopError::NotClosed { .. },
            ) => format!("{e} {OR_A_WALL}"),
            // By the curve's name, since an id is nothing a person can go and look for.
            fieldform_field::CompileError::BadWall(id, _) => {
                let text = e.to_string();
                match self.doc.curves.get(id) {
                    Some(curve) => text.replace(
                        &format!("curve {id:?}"),
                        &format!("\u{201c}{}\u{201d}", curve.name),
                    ),
                    None => text,
                }
            }
            _ => e.to_string(),
        }
    }

    // `compile_error` used to have a body here: it compiled the display graph on every call,
    // and the panel called it once a frame. That is a full compile — cage subdivision, mesh
    // BVHs, crease normals — to answer a question whose answer changes only when the document
    // does, and on a 109 000-triangle import it was what made the whole interface slow.
    // `rebuild_render_resources` already compiles the same graph for picking, so the answer is
    // recorded there and this reads it.


    /// Re-tessellates every curve for drawing.
    ///
    /// 0.05 mm is far finer than a screen pixel at any sane framing, so the drawn curve is
    /// smooth rather than faceted, and this runs on import rather than per frame.
    fn rebuild_curve_display(&mut self) {
        let started = std::time::Instant::now();
        // Every path that changes a curve comes through here, which is what makes this the
        // counter the profile rows are cached against.
        self.curve_revision = self.curve_revision.wrapping_add(1);
        // And the overlay's readings, for the same reason: a sweep's extent is its curve's.
        self.document_moved();
        self.curve_display = self
            .doc
            .curves
            .values()
            .map(|c| {
                let points = c
                    .tessellate(0.05)
                    .into_iter()
                    .map(|p| Point3::new(p.x as f32, p.y as f32, p.z as f32))
                    .collect();
                (c.id, points)
            })
            .collect();
        // HashMap order is arbitrary; sort so the curve list in the UI does not reshuffle
        // between frames.
        self.curve_display.sort_by_key(|(id, _)| id.raw());
        self.note_stage("the curve overlay", started);
    }

    /// Bounds over everything in the scene — the cage and every curve — as centre and radius.
    ///
    /// `None` only when there is nothing at all to frame.
    fn scene_bounds(&self) -> Option<(Point3<f32>, f32)> {
        let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        fn grow(
            min: &mut Point3<f32>,
            max: &mut Point3<f32>,
            lo: Point3<f32>,
            hi: Point3<f32>,
        ) {
            *min = Point3::new(min.x.min(lo.x), min.y.min(lo.y), min.z.min(lo.z));
            *max = Point3::new(max.x.max(hi.x), max.y.max(hi.y), max.z.max(hi.z));
        }

        // Every visible body, from the graph — which is the thing the viewport draws, and so
        // the only description of the scene that cannot disagree with it. This used to be the
        // cage and the curves alone, so a part made of primitives contributed *nothing*: a
        // 300 mm sphere framed to the 100 mm cage sitting invisibly behind it, and an import
        // framed to the same. The cage is one body among several, not the scene.
        let part = self.part_graph();
        let consumed = part.consumed();
        for body in part.roots() {
            if !is_visible_among(&self.doc, part, &consumed, body) {
                continue;
            }
            // An unbounded body — a half-space, a lattice — contributes no box and does not
            // veto the others', the same rule `pick_bounds` follows. A plane has no extent to
            // frame, and letting it erase everything else would leave the camera nowhere.
            if let Some((lo, hi)) = self.body_extent(body).as_box() {
                grow(&mut min, &mut max, lo.cast::<f32>(), hi.cast::<f32>());
            }
        }

        // Curves are drawn but are not bodies, so they are their own contribution.
        for (id, points) in &self.curve_display {
            if self.hidden_curves.contains(id) {
                continue;
            }
            for p in points {
                grow(&mut min, &mut max, *p, *p);
            }
        }

        // And surfaces, for the same reason. Framing has to reach an imported surface or a
        // document holding nothing else opens looking at empty space.
        for (id, edges) in &self.surface_display {
            if self.hidden_surfaces.contains(id) {
                continue;
            }
            for [a, b] in edges {
                grow(&mut min, &mut max, *a, *a);
                grow(&mut min, &mut max, *b, *b);
            }
        }

        // The cage last and only if nothing else spoke. It is in the graph when a body draws
        // it, so it is already counted above in the ordinary case; this is the empty document,
        // where framing the cage that Create > Cage would bring back beats framing nothing.
        if !min.x.is_finite() {
            let (centre, radius) = cage_bounds(self.cage());
            grow(
                &mut min,
                &mut max,
                centre - Vector3::repeat(radius),
                centre + Vector3::repeat(radius),
            );
        }
        if !min.x.is_finite() {
            return None;
        }
        let center = Point3::from((min.coords + max.coords) * 0.5);
        Some((center, ((max - min).norm() * 0.5).max(0.1)))
    }

    /// What framing should aim at: the selected body if there is one, else the whole scene.
    ///
    /// "Frame selection" has to mean the selection when there is one — framing the document
    /// while one body is selected is the command declining to do what it is called.
    fn framing_bounds(&self) -> Option<(Point3<f32>, f32)> {
        // A block something consumes frames whatever its own eye says, as the manipulator keeps
        // its handles (`gizmo_frame`): that eye only draws it beside its consumer as well, and
        // framing a selection never asked whether the selection was drawn through something.
        if let Some(body) = self.selected_body
            && !(self.eye_flag(body) == EyeFlag::Hidden && !self.is_visible(body))
            && let Some((lo, hi)) = self.body_extent(body).as_box()
        {
            let (lo, hi) = (lo.cast::<f32>(), hi.cast::<f32>());
            let centre = Point3::from((lo.coords + hi.coords) * 0.5);
            return Some((centre, ((hi - lo).norm() * 0.5).max(0.1)));
        }
        self.scene_bounds()
    }

    /// The selected curve's bounding-box diagonal in millimetres, for scaling defaults to it.
    fn selected_curve_size(&self) -> Option<f64> {
        let curve = self.doc.curves.get(&self.selected_curve?)?;
        let points = curve.tessellate(0.5);
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &points {
            min = Point3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Point3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        }
        min.x.is_finite().then(|| (max - min).norm())
    }

    /// The box the model occupies, for sizing picking tolerances: [`fieldform_field::pick_bounds`]
    /// over what the viewport draws and the points it paints, as the overlay's readings keep it.
    fn pick_bounds(&self) -> (Point3<f64>, Point3<f64>) {
        self.overlay_readings().pick_bounds
    }

    /// Where one body's surface can be.
    ///
    /// Delegates to `fieldform_field::graph_extent` rather than walking the graph here. This
    /// used to be its own copy of that walk, which meant the viewport and anything else asking
    /// the same question of the same graph could answer it differently — the failure this
    /// codebase keeps meeting. One walk, three states: see `Extent` for why an `Option` could
    /// not say what a half-space needs it to.
    fn body_extent(&self, id: NodeId) -> Extent {
        fieldform_field::graph_extent(&self.doc, self.part_graph(), id)
    }

    /// Selects the body whose surface is under the cursor, if any is.
    ///
    /// Traces the composed field for a hit and then asks which body owns that point, rather
    /// than tracing each body separately: the bodies are already compiled as steps of the same
    /// field, so this is one trace and a handful of evaluations rather than one trace each.
    fn select_body_under(&mut self, origin: Point3<f32>, dir: Vector3<f32>) {
        // No field to trace is a click that hit nothing, not a click to ignore. Returning early
        // here meant that with nothing visible — everything hidden, or a graph that would not
        // compile — clicking could not clear a selection either, so the one gesture for "none
        // of this" stopped working exactly when there was nothing to select.
        let Some(field) = &self.pick_field else {
            self.selected_body = None;
            return;
        };
        let bounds = self.pick_bounds();
        let radius = ((bounds.1 - bounds.0).norm() * 0.5).max(1e-6);
        let eye = Point3::new(origin.x as f64, origin.y as f64, origin.z as f64);
        let d = Vector3::new(dir.x as f64, dir.y as f64, dir.z as f64).normalize();
        let reach = self.trace_reach(eye, bounds);
        let Some(hit) = trace_field(field, eye, d, reach, radius * 1e-4, 512) else {
            // A click on empty space clears the selection, which is what makes it possible to
            // select nothing on purpose.
            if self.selected_body.take().is_some() {
                self.say("Cleared the selection");
            }
            return;
        };
        // The trace stops within its own epsilon of the surface, so the body that owns the
        // point is within the same distance of zero. A tolerance any tighter would reject the
        // hit the trace just accepted.
        // **Only when it changes.** Every click through a body runs this, so reporting each one
        // would fill the log with the same line while someone turns a part around looking at it.
        let picked = field.body_at(hit, radius * 1e-3);
        // A list that is collecting takes the click. Only a body that is drawn can be clicked,
        // and a drawn body is one nothing consumes yet, so what arrives here is exactly what a
        // list can take — the one exception being the boolean's own result, which is also drawn.
        if let Some(block) = self.collecting
            && let Some(body) = picked
        {
            self.add_operand(block, body);
            return;
        }
        let was = self.selected_body;
        self.selected_body = picked;
        if let Some(id) = self.selected_body.filter(|&id| Some(id) != was) {
            let label = self.body_label(id);
            self.say(format!("Selected {label}"));
        }
    }

    /// Adds a body to a boolean's list, from a click in the viewport while the list collects.
    ///
    /// Through `replace_node`, so a body that would close a loop is refused with the reason and
    /// the graph is left as it was.
    fn add_operand(&mut self, block: NodeId, body: NodeId) {
        let Some(Node::Boolean { op, inputs }) = self.part_graph().node(block).cloned() else {
            // The block stopped being a boolean — turned into a fillet, or was deleted — so
            // there is no list to add to.
            self.collecting = None;
            return;
        };
        let what = self.body_label(block);
        if body == block {
            self.say(format!("That is {what} itself — click a body to add it"));
            return;
        }
        // Over the slots' sources rather than over the slots: an empty one in the list is
        // not this body, and a filled one is only when it reads it.
        if inputs.iter().any(|slot| slot.source() == Some(body)) {
            self.say(format!("{} is already in {what}", self.body_label(body)));
            return;
        }
        let added = self.body_label(body);
        let mut inputs = inputs;
        inputs.push(body.into());
        self.replace_node(block, Node::Boolean { op, inputs });
        self.say(format!("Added {added} to {what}"));
    }

    /// Shows or hides the selected body.
    fn toggle_selected_visibility(&mut self) {
        let Some(id) = self.selected_body else {
            self.warn("Nothing selected");
            return;
        };
        self.toggle_visibility(id);
    }

    /// Turns a block's eye over: the eye on its row, the row menu's item, and H.
    ///
    /// Said in the words of what its flag does: a body is shown or hidden, a block something
    /// consumes is drawn beside the bodies or not.
    fn toggle_visibility(&mut self, id: NodeId) {
        let label = self.body_label(id);
        let on = !self.is_visible(id);
        let said = match (self.eye_flag(id), on) {
            (EyeFlag::Hidden, true) => format!("Showing {label}"),
            (EyeFlag::Hidden, false) => format!("Hiding {label}"),
            (EyeFlag::Shown, true) => format!("Drawing {label} as well"),
            (EyeFlag::Shown, false) => format!("Stopped drawing {label}"),
        };
        self.set_visible(id, on);
        self.say(said);
    }

    /// Deletes whatever is selected — a curve, or a body.
    ///
    /// A curve first, because it is the more specific selection: selecting one clears the body,
    /// so if a curve is selected it is what the user is pointing at.
    fn delete_selection(&mut self) {
        // A selected block is what was clicked last: selecting a body or a curve clears it.
        if let Some(block) = self.selected_block {
            self.delete_block(block);
            return;
        }
        if let Some(id) = self.selected_curve {
            self.delete_curve(id);
            return;
        }
        self.delete_selected_body();
    }

    /// Deletes a curve, unless a pipe is sweeping it or a block in another part reads it.
    ///
    /// Refused rather than cascaded. A `Node::Pipe` naming a curve that is not there does not
    /// evaluate at all, so deleting one out from under a pipe would blank the viewport — and
    /// silently deleting the pipe as well would throw away a body the user did not ask about.
    /// Naming the pipe is the answer that leaves them in control of both.
    fn delete_curve(&mut self, id: CurveId) {
        // **`Node::curves` rather than a match on `Node::Pipe`.** The hand-written version was
        // right when a pipe was the only thing that referenced a curve, and had been wrong since
        // extrude and revolve were added: a curve driving either could be deleted out from under
        // it. That is the mistake D9 describes as being made once per node kind, and asking the
        // node is what stops the next one making it.
        // **Asking the node is not enough, because an extrude names one curve and reads a loop.**
        // `Node::curves` returns the single `profile` an extrude, revolve or sweep stores; the boundary
        // it actually walks is `profile_loop` from that curve. So deleting any *other* member of the
        // same loop passed this test and opened the loop, and the solid stopped compiling with nothing
        // said at the moment the damage was done. The row path over-refused and this path
        // under-refused, in opposite directions, on the same four curves.
        //
        // So a member is tested by its loop: whatever consumes any curve of the loop this one is in
        // consumes this one in effect.
        let mut loop_mates =
            fieldform_field::loop_from(&self.doc, id, None).unwrap_or_else(|_| vec![id]);
        if !loop_mates.contains(&id) {
            loop_mates.push(id);
        }
        // **A block in another part refuses it outright, before anything below is asked.**
        // Everything below reads the part on screen, where a delete can open a loop into a wall
        // because the block is in front of the person deleting; `readers_in_other_parts` says why
        // another part's is not. Asked of this curve, not over `loop_mates`: the function walks
        // each block's own section, so a block built on the loop is found from any side of it,
        // and a pipe along a neighbouring side — which reads that side alone — is not.
        let elsewhere = readers_in_other_parts(&self.doc, self.active_part, &[id], |g, n| {
            self.body_label_in(g, n)
        });
        if !elsewhere.is_empty() {
            let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
            let readers = elsewhere.readers.join(", ");
            let read = if elsewhere.through_sections {
                format!("is one of the curves {readers} is made from")
            } else {
                format!("is used by {readers}")
            };
            self.say(format!("{name} {read} — {}", elsewhere.built_on()));
            return;
        }
        let used_by: Vec<String> = self
            .part_graph()
            .nodes()
            .filter(|(_, n)| n.curves().iter().any(|c| loop_mates.contains(c)))
            .map(|(node, _)| self.body_label(node))
            .collect();
        if !used_by.is_empty() {
            // **A member of a loop that blocks are built on can go, and the blocks follow.**
            // What is left is an open chain, which a block can be built on now, as a wall. So
            // the member goes, the joints at its ends go, and each block that read the loop
            // reads the chain. Only where every reader of *this* curve reads it as its section
            // — a pipe along it, or a sweep carrying something along it, still refuses below.
            let only_as_a_section = self.part_graph().nodes().all(|(_, n)| {
                let names = n.curves().iter().filter(|c| **c == id).count();
                names == 0 || (names == 1 && n.outline().map(|o| o.curve()) == Some(id))
            });
            if loop_mates.len() > 1 && only_as_a_section {
                self.delete_loop_member(id, &loop_mates);
                return;
            }
            let mut used_by = used_by;
            used_by.sort();
            used_by.dedup();
            let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
            // Said differently for a loop member, because "X is used by Extrude" is baffling when
            // nothing names X: what reads it is the loop it closes.
            if loop_mates.len() > 1 {
                self.say(format!(
                    "{name} closes the loop {} is made from, so deleting it would open that loop \
                     — delete {} first, or separate the row and keep the curves",
                    used_by.join(", "),
                    used_by.join(", "),
                ));
            } else {
                self.say(format!("{name} is used by {} — delete that first", used_by.join(", ")));
            }
            return;
        }
        let Some(curve) = self.doc.curve(id).cloned() else { return };
        let name = curve.name.clone();
        // **The same builder the row path uses, because a lone curve leaves joints too.** This went
        // through `delete_curves_commands`, which knows nothing about points, so deleting one member of
        // a held loop left its corner dots behind — the bug `deleting_a_row_takes_its_joints_with_it`
        // was written for, fixed on the row path and not here. `delete_row_commands` removes a point
        // only when nothing reads it and no *remaining* curve sits on it, so a shared corner stays for
        // the curve still using it.
        let commands = delete_row_commands(&self.doc, self.active_part, &[id]);
        let relations = commands.len().saturating_sub(1);
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("curve.delete", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.selected_curve = None;
        self.clear_point_selection();
        self.mark_changed();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        self.say(match relations {
            0 => format!("Deleted {name}"),
            1 => format!("Deleted {name} and the relation naming it"),
            n => format!("Deleted {name} and the {n} relations naming it"),
        });
    }

    /// The commands that take one curve out of a held loop, and what they did; or, refused, what
    /// to say instead.
    ///
    /// Free of the `App` for the reason `after_deleting` is: it is the whole of the delete that a
    /// test has to see, the commands and the check.
    pub(crate) fn loop_member_deletion(
        doc: &Document,
        part: PartId,
        id: CurveId,
        mates: &[CurveId],
        wall_thickness: f64,
        explain: &dyn Fn(&fieldform_field::CompileError) -> String,
    ) -> Result<(Vec<Box<dyn fieldform_ops::Command>>, curve_edits::LoopMemberDeleted), String> {
        let name = doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
        let taken = delete_loop_member_commands(doc, part, id, mates, wall_thickness);
        let (Some((commands, after, done)), Some(before)) = (taken, doc.parts.get(&part)) else {
            return Err(format!("{name} could not be taken out of its loop"));
        };
        // Refused here rather than committed into a part that cannot be drawn, by the rule and in
        // the words a block's delete is refused by: only for breaking a part that builds. A
        // drafted extrude with sharp corners, say, cannot be the wall this makes it. Asked of the
        // graph left alone, a part already broken elsewhere refused every curve out of every loop.
        if let Some(why) = Self::delete_breaks(doc, &before.graph, &after, explain) {
            return Err(Self::delete_refused(&name, &why));
        }
        Ok((commands, done))
    }

    /// Deletes one curve out of a loop that blocks are built on, and says what followed from it.
    /// See `delete_loop_member_commands`; this is the part that needs the window.
    fn delete_loop_member(&mut self, id: CurveId, mates: &[CurveId]) {
        let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
        let deleting = Self::loop_member_deletion(
            &self.doc,
            self.active_part,
            id,
            mates,
            self.wall_thickness,
            &|e| self.explain_compile(e),
        );
        let (commands, done) = match deleting {
            Ok(deleting) => deleting,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let walled: Vec<String> = done.walled.iter().map(|n| self.body_label(*n)).collect();
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("curve.delete", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.selected_curve = None;
        self.clear_point_selection();
        self.mark_changed();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        let joints = match done.joints {
            0 => String::new(),
            1 => " and the joint at its end".to_string(),
            n => format!(" and the {n} joints at its ends"),
        };
        self.say(if walled.is_empty() {
            format!("Deleted {name}{joints}; the wall runs along what is left")
        } else {
            format!(
                "Deleted {name}{joints}. The loop is open now, so {} is a wall {} mm thick \
                 along what is left — tick Wall off in the block once it closes again",
                walled.join(" and "),
                self.wall_thickness,
            )
        });
    }

    /// Puts the selection on the clipboard: a curve, or a body with everything it is made of
    /// ([`copied_body`] says what that takes).
    fn copy_selection(&mut self, announce: bool) {
        if let Some(id) = self.selected_curve
            && let Some(curve) = self.doc.curve(id).cloned()
        {
            self.clipboard = Some(Clipboard::Curve(curve));
            if announce {
                self.say("Copied the curve");
            }
            return;
        }
        let Some(root) = self.selected_body else {
            self.warn("Nothing selected");
            return;
        };
        let label = self.body_label(root);
        self.clipboard = Some(copied_body(&self.doc, self.part_graph(), root));
        if announce {
            self.say(format!("Copied {label}"));
        }
    }

    /// Copies, then deletes.
    fn cut_selection(&mut self) {
        self.copy_selection(true);
        if self.clipboard.is_some() {
            self.delete_selection();
        }
    }

    /// A second, independent copy of what is selected, beside it.
    ///
    /// **Copy and paste with the clipboard put back**, rather than a third traversal of the
    /// graph. The pair already does everything a duplicate has to: fresh ids for every node,
    /// cage and curve underneath, so the copy shares nothing with what it came from, and the
    /// whole of it in one undo step. Writing it again would be a second answer to the question
    /// "what does copying a body mean", and the two would drift.
    ///
    /// What it must not do is spend the clipboard. That belongs to whoever put something in
    /// it, and losing it to a menu item used for something else is the kind of small theft
    /// nobody reports and everybody notices.
    fn duplicate_selection(&mut self) {
        let held = self.clipboard.take();
        self.copy_selection(false);
        if self.clipboard.is_none() {
            // `copy_selection` has already said why there was nothing to take.
            self.clipboard = held;
            return;
        }
        self.paste(false);
        self.clipboard = held;
        // Said once, here, rather than twice by the two halves this is made of.
        match (self.selected_curve, self.selected_body) {
            (Some(_), _) => self.say("Duplicated the curve"),
            (_, Some(id)) => {
                let label = self.body_label(id);
                self.say(format!("Duplicated as {label} — drag it off the original"));
            }
            _ => {}
        }
    }

    /// Pastes whatever was copied, as something independent of it.
    ///
    /// In place rather than offset by a guess. The copy is selected on arrival and the
    /// manipulator is already on it, so moving it is the next gesture rather than a number this
    /// has to invent.
    fn paste(&mut self, announce: bool) {
        let Some(clipboard) = self.clipboard.clone() else {
            self.warn("Nothing copied");
            return;
        };
        match clipboard {
            Clipboard::Curve(curve) => {
                let id = CurveId::new();
                let Ok(fresh) = curve.copy_as(id, format!("{} copy", curve.name)) else {
                    self.warn("Could not copy that curve");
                    return;
                };
                self.insert_pasted_curve(fresh, announce.then(|| "Pasted the curve".to_string()));
            }
            Clipboard::Body { nodes, root, cages, curves } => {
                // What `set_part_graph` does around a graph edit, which this is as well, though it
                // goes on the history as a batch with the curves ([`paste_body`] says why). The
                // gestures end because the entry on top is no longer theirs to fold into.
                self.splash_shown = false;
                self.material_gesture = None;
                self.gesture = None;
                let part = self.active_part;
                let pasted =
                    paste_body(&mut self.stack, &mut self.doc, part, &nodes, root, &cages, &curves);
                let new_root = match pasted {
                    Ok(id) => id,
                    Err(why) => {
                        self.warn(why);
                        return;
                    }
                };
                self.stamp_new_nodes();
                self.select_body(Some(new_root));
                self.mark_changed();
                self.rebuild_curve_display();
                self.rebuild_render_resources();
                // **Loose, as a pasted curve is** (`insert_pasted_curve`). The copies carry no
                // places, and a loop they close is found here without being held, so no join
                // lands on the history after the paste as an entry of its own.
                self.refresh_profiles_with(false);
                if announce {
                    self.say("Pasted — drag it to move it off the original");
                }
            }
        }
    }

    /// Puts a pasted curve into the document and says where it landed.
    ///
    /// **Loose, whatever it touches.** A copy carries no places, so its ends sit on nothing: a
    /// cut curve pasted straight back touches its loop again and the row re-forms, but nothing
    /// rejoins behind your back (`profile-group-spec.md` §4.4). The row marks it "not held",
    /// and `join` holds it. The loop it may have re-formed is reported here, before the block
    /// list would find it, so that the finding is the one that does not join.
    fn insert_pasted_curve(&mut self, fresh: fieldform_scene::Curve, said: Option<String>) {
        let id = fresh.id;
        // Not a graph edit, so it ends any gesture a value drag had open, as a placed curve does.
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(SetCurve::insert(fresh))) {
            self.warn(e.to_string());
            return;
        }
        self.select_curve(id);
        self.mark_changed();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        self.refresh_profiles_with(false);
        if let Some(mut line) = said {
            if let Some(key) = self.profiles.key_of(id) {
                line.push_str(&format!(
                    " — it touches {} but is not held; join holds it",
                    self.profile_label(key)
                ));
            }
            self.say(line);
        }
    }

    /// Pastes the copied curve with its start under the cursor, moved as one piece.
    ///
    /// The viewport's **Paste here**, and Ctrl+V while the cursor is over the viewport
    /// (`profile-group-spec.md` §4.4). A click is a place the person chose, which is what
    /// [`Self::paste`] rightly refuses to guess at; where it lands is decided by
    /// `landing_point`, exactly as a placed curve's clicks are.
    fn paste_here(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let Some(Clipboard::Curve(curve)) = self.clipboard.clone() else {
            self.warn("Nothing copied that can be put down at a point — copy a curve first");
            return;
        };
        let at = match self.landing_under(aspect, rect, pos) {
            Ok(at) => at,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let id = CurveId::new();
        let Ok(fresh) = moved_to_start_at(&curve, id, format!("{} copy", curve.name), at) else {
            self.warn("Could not copy that curve");
            return;
        };
        self.insert_pasted_curve(
            fresh,
            Some(format!(
                "Pasted the curve with its start at {:.1}, {:.1}, {:.1}",
                at.x, at.y, at.z
            )),
        );
    }

    /// Ctrl+V: a curve lands under the cursor when the cursor is over the viewport; anything
    /// else, or anywhere else, pastes in place as it always did.
    fn paste_at_hover(&mut self) {
        match (self.viewport_hover, &self.clipboard) {
            (Some((aspect, rect, pos)), Some(Clipboard::Curve(_))) => {
                self.paste_here(aspect, rect, pos)
            }
            _ => self.paste(true),
        }
    }

    /// Deletes whichever body is selected, from the tree or the viewport.
    ///
    /// A key rather than another button: the panel already has a Delete inside each block, and
    /// the thing missing was a way to do it from where the selection was made.
    fn delete_selected_body(&mut self) {
        let Some(id) = self.selected_body else {
            self.warn("Nothing selected");
            return;
        };
        let label = self.body_label(id);
        // A refused delete has said why, and a "Deleted" after it would be the line a reader
        // trusts and the wrong one.
        if self.delete_body(id) {
            self.say(format!("Deleted {label}"));
        }
    }

    /// How far a pick has to march before giving up: to the model, and then across it.
    ///
    /// It used to be a multiple of the model's own size, which is wrong whenever the camera is
    /// further away than the model is wide. Hide a metre cage, add a hundred-millimetre sphere
    /// in the middle of it, and the camera is still framed for the cage — so the budget was a
    /// few hundred millimetres and the sphere was two metres away. Every click missed, the
    /// pivot stayed where it was, and it looked as though the pivot were stuck to the geometry
    /// that had just been hidden.
    fn trace_reach(&self, eye: Point3<f64>, bounds: (Point3<f64>, Point3<f64>)) -> f64 {
        let centre = Point3::from((bounds.0.coords + bounds.1.coords) * 0.5);
        let across = (bounds.1 - bounds.0).norm();
        (eye - centre).norm() + across.max(1.0) * 2.0
    }

    /// Finds the point to tumble about, under the cursor.
    ///
    /// Traces the field rather than raycasting a mesh of it. A mesh is a mode-dependent
    /// approximation of what is on screen — in traced mode a noticeably coarser one — so
    /// picking against it made the pivot land differently depending on which view happened to
    /// be showing, and miss entirely near a silhouette. The field is the same in every mode.
    ///
    /// Curves are picked too, in screen space, because a curve has no surface to hit but is
    /// still something worth pivoting around.
    fn pick_pivot(
        &self,
        origin: Point3<f32>,
        dir: Vector3<f32>,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Option<Point3<f32>> {
        let bounds = self.pick_bounds();
        let radius = ((bounds.1 - bounds.0).norm() * 0.5).max(1e-6);

        // Only what the mode actually draws can be aimed at, and a cage may not stand in for
        // its field or the reverse. A cage strictly *encloses* its limit surface — a
        // Catmull-Clark cube's limit is roughly a sphere well inside the box — so the two are
        // wrong for each other in opposite directions. Picking the field in polygon mode
        // misses every click except near the middle of the cage; picking the cage in field
        // mode lands the pivot in empty space in front of the surface being looked at.
        //
        // Falling back from one to the other is therefore not a kindness, it is the bug. Only
        // overlay draws both, and there the field is what reads as solid.
        let (may_pick_cage, may_pick_field) = match self.view_mode {
            ViewMode::Polygon => (true, false),
            ViewMode::Field => (false, true),
            ViewMode::Overlay => (true, true),
        };

        if may_pick_field {
            if let Some(field) = &self.pick_field {
                let eye = Point3::new(origin.x as f64, origin.y as f64, origin.z as f64);
                let d = Vector3::new(dir.x as f64, dir.y as f64, dir.z as f64).normalize();
                // Generous budget: a click happens once, and giving up early looks exactly
                // like hitting nothing.
                let reach = self.trace_reach(eye, bounds);
                if let Some(hit) = trace_field(field, eye, d, reach, radius * 1e-4, 512) {
                    return Some(Point3::new(hit.x as f32, hit.y as f32, hit.z as f32));
                }
            }
        }

        // Only if the cage is actually on screen. Hiding a body has to take it out of every
        // answer the viewport gives, not only the drawn one — a pivot landing on geometry that
        // is not there reads as the pivot being broken, which is how this was noticed.
        if may_pick_cage && self.control_visible() {
            if let Some((hit, _)) = self.cage_mesh.raycast(origin, dir) {
                return Some(hit);
            }
        }

        // No surface under the cursor: try the curves, which are drawn but have no solid to
        // hit. Screen-space, because that is how the user aimed at them.
        const CURVE_PICK_PIXELS: f32 = 10.0;
        let mut best: Option<(f32, Point3<f32>)> = None;
        for curve in self
            .doc
            .curves
            .values()
            .filter(|c| !self.hidden_curves.contains(&c.id))
        {
            for p in curve.tessellate((radius * 2e-3).max(1e-3)) {
                let world = Point3::new(p.x as f32, p.y as f32, p.z as f32);
                let Some(screen) = self.camera.project_to_screen(aspect, rect, world) else {
                    continue;
                };
                let d = screen.distance(pos);
                if d <= CURVE_PICK_PIXELS && best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, world));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// Draws the composed field by evaluating it per pixel on the GPU.
    ///
    /// The expensive half — compiling the shader and baking the cage — happens only when
    /// something has actually changed, which is why staleness is checked before a `Scene` is
    /// gathered rather than after.
    fn draw_traced(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        // **First, before anything is asked of the device: whether it is the device everything
        // was made on** (loop spec S4). A lost device fails every call rather than waiting on
        // one, and everything the view holds — the march's pipelines and buffers, the lighting
        // pass, the textures egui samples — belongs to it. The callback that found the loss may
        // have run on any thread, so this is the frame answering it for the picture, as
        // `keep_the_document_if_the_gpu_was_lost` answers it for the document — that one runs
        // first and last in every frame, so the work is kept before this says where it is.
        if !self.traced.ready_to_draw_on(
            &self.wgpu_render_state,
            self.gpu_lost.load(std::sync::atomic::Ordering::Acquire),
        ) {
            // Nothing is traced, nothing is submitted and nothing of the old picture is left on
            // screen; what is said is the view's own reason for an empty viewport.
            self.say_why_nothing_is_traced();
            return;
        }
        // Told every frame rather than on change, because it is the one place that knows both
        // the setting and the view, and setting it to what it already is does nothing. The
        // rebuild below is handed its walks from this same reading, so what the view compares
        // is what it was built from.
        let fineness = self.display_quality.fineness();
        self.traced.set_fineness(fineness);
        // A settled band that has finished being computed elsewhere comes in here, and asks for
        // the rebuild that draws with it. While one is still being computed the viewport keeps
        // asking for frames — nothing else would, since the news arrives from another thread.
        if self.traced.collect_bakes() || self.traced.is_baking() {
            ui.ctx().request_repaint();
        }
        // The same for a shader the driver is still chewing on. Until it lands the viewport
        // keeps drawing the shape it had, which is why the wait is said out loud: what is on
        // screen is honest about the *old* document, and nothing else would explain that.
        if self.traced.collect_compile() || self.traced.is_compiling().is_some() {
            ui.ctx().request_repaint();
        }
        if let Some(waited) = self.traced.compile_worth_saying() {
            self.say(format!(
                "The driver is still compiling this shape's shader ({:.0} s so far) — \
                 what is drawn is the shape before the change until it lands.",
                waited.as_secs_f32()
            ));
        }
        let tracing = std::time::Instant::now();
        if self.traced.is_stale() {
            // With no field generator applied there is still a cage to trace, and drawing
            // nothing was indistinguishable from the renderer being broken. `pick_bounds`
            // already answers for both cases.
            // What the kernel refuses, the viewport does not draw.
            //
            // These used to disagree: `fieldform_gpu::generate` checks for unknown nodes and
            // cycles but says nothing about normalization, so a blend against a lattice --
            // refused by D8 in `GraphField::compile_graph` -- was lowered to WGSL and traced
            // anyway. The result was a document you could see and could not export, measure or
            // click on, and the shape on screen was the one D8 exists to stop you trusting: a
            // blend mixes by magnitude, so an operand that understates its distance pulls the
            // seam toward itself and the radius is not the radius asked for.
            //
            // The CPU compile is the authority rather than a second opinion, because it is the
            // one that owns the rule. Drawing nothing and explaining why in the panel is worse
            // to look at than a wrong shape and better than one.
            let graph = if self.compile_error.is_some() {
                Graph::empty()
            } else {
                self.display_graph()
            };
            let bounds = self.pick_bounds();
            let extent = bounds.1 - bounds.0;
            let cell = extent.x.max(extent.y).max(extent.z) / self.mesh_resolution as f64;
            let tolerance = (cell * 0.5).max(1e-4);

            // An angle, not a tolerance. The tube's normal turns a corner at every joint, and
            // shading *is* the normal, so what shows is the size of that corner — where a
            // tolerance only says how far the spine strays from the curve. A thousandth of the
            // extent sounds tight and measured 6.35° per joint on a 310 mm cubic, which is
            // bands across the tube and, through a blend, across whatever it was cut into.
            // One degree is the threshold where they stop being visible; see
            // `Curve::tessellate_to_angle` for the measurements.
            //
            // The shader walks these points per sample, so the cost is linear in their number
            // and a fine spine is cheap next to being visibly wrong.
            let spine_degrees = fineness.spine_degrees;
            let spine_floor = (extent.norm() * 1e-5).max(1e-4);
            // Tessellated by id and handed over unordered: the shader binds curves in the
            // order its own compilation interned them, and `traced` is where that order is
            // known. Producing the list in what this side guessed that order to be worked
            // only while there was one curve.
            // **`Node::curves` rather than a match on `Node::Pipe`.** A hand-written match
            // is a list of node kinds that name a curve, kept in one more place than the one
            // that knows -- and the failure when it falls behind is silent in the worst way:
            // the curve never reaches the shader, the span arrives with a count of zero, and
            // the body renders as though nothing had been asked of it. An attractor drawn
            // toward nothing looks exactly like an attractor set to zero. The same omission
            // in `gpu_matches_cpu` is what it cost to find this one.
            //
            // Extra entries are free -- a profile's curve is keyed by node rather than by id
            // on the other side and simply goes unread -- so over-collecting is the safe
            // direction and under-collecting is not.
            //
            // Through the view, which keeps each spine against the curve, angle and floor it was
            // walked from: a rebuild that moved no curve walks none. The angle here is the
            // rung's, which a settled rebuild walks at; a draft walks at two degrees, which is
            // all a drag frame can show, and the view is what knows which rebuild comes next.
            let read: std::collections::BTreeSet<CurveId> =
                graph.nodes().flat_map(|(_, node)| node.curves()).collect();
            let curves = self.traced.spines(&self.doc.curves, read, spine_degrees, spine_floor);
            // Cloned because `Scene` borrows it while `self.traced` is borrowed mutably. A
            // control cage is scaffolding and small by design, and this happens on a rebuild
            // rather than per frame.
            let cage = self.cage().clone();
            let device = self.wgpu_render_state.device.clone();
            let queue = self.wgpu_render_state.queue.clone();

            self.traced.rebuild(
                &device,
                &queue,
                &traced::Scene {
                    document: &self.doc,
                    meshes: self.doc.meshes.iter().map(|(id, m)| (*id, m)).collect(),
                    graph: &graph,
                    // From the field compiled for picking, over this same graph, so the shader
                    // divides by exactly what the CPU divided by.
                    morph_divisors: self
                        .pick_field
                        .as_ref()
                        .map(|f| f.morph_divisors())
                        .unwrap_or(&[]),
                    tolerance_mm: tolerance,
                    // The occlusion's radius for the scene the view will be drawn with -- the
                    // same box, so the same radius `refresh_scene_bounds` hands the camera.
                    reads_mm: fieldform_gpu::Occlusion::reach(extent.norm() * 0.5),
                    profile_fineness: self.display_quality.profile_bake_fineness(),
                    profile_degrees: fineness.profile_degrees,
                    curves,
                    cage_bounds: cage_box(&cage),
                },
            );
        }

        // Every parameter's value, as the overlay's readings keep the table: the closure outlives
        // this borrow of the document, so it carries the table rather than the document. A failed
        // expression is simply absent, which the compile reports the way it reports an unknown
        // name.
        let readings = self.overlay_readings();
        let lookup = move |id: fieldform_scene::ParamId| readings.values.get(&id).copied();

        let section = self.section.half_space();
        let selected = self.selected_body_index();
        // The part's own material, or the default when it has none. Read every frame rather
        // than kept, so an undo that puts a material back is drawn without anyone remembering
        // to say so.
        let material = self
            .doc
            .parts
            .get(&self.active_part)
            .and_then(|p| p.material.clone())
            .unwrap_or_default();
        let view = traced::View {
            selected,
            section,
            eye: self.camera.eye.cast::<f64>(),
            target: self.camera.target.cast::<f64>(),
            up: self.camera.up.cast::<f64>(),
            scene: (
                Point3::new(
                    self.camera.scene_centre.x as f64,
                    self.camera.scene_centre.y as f64,
                    self.camera.scene_centre.z as f64,
                ),
                self.camera.scene_radius as f64,
            ),
            fov_deg: 45.0,
            orthographic: self.camera.projection == Projection::Orthographic,
            lookup: &lookup,
            material,
            occlusion: self.idle_occlusion && self.display_quality.occlusion(),
            exact_finish: self.display_quality.exact_finish(),
            environment: self.environment,
            draft: self.draft_overlay(),
            max_pixels: self.display_quality.still_pixels(),
            interactive_pixels: self.display_quality.drag_pixels(),
            still_frames: self.display_quality.still_frames(),
        };
        self.traced.draw(ui, rect, &view, &self.wgpu_render_state);

        self.note_stage("rebuilding the traced view", tracing);
        // Its three parts, so a slow one says which: generating the shader, preparing what the
        // card is handed (bands, spines, surfaces), or the driver's own pipeline and uploads.
        if let Some((quality, cost)) = self.traced.take_rebuild_parts() {
            let tier = match quality {
                traced::Quality::Draft => "draft",
                traced::Quality::Final => "settled",
            };
            let mut parts = vec![
                (format!("{tier}: generating the shader"), cost.generate_ms),
                (format!("{tier}: preparing the bands"), cost.prepare_ms),
                (format!("{tier}: the driver's pipeline and uploads"), cost.pipeline_ms),
                (format!("{tier}: letting the old renderer go"), cost.release_ms),
            ];
            // The renderer's own four, so "the driver's pipeline and uploads" says which.
            if let Some(build) = self.traced.take_build_parts() {
                for (what, ms) in build.parts() {
                    parts.push((format!("{tier}: {what}"), ms));
                }
            }
            for (leaf, field_ms, bake_ms) in self.traced.take_bake_parts() {
                parts.push((format!("building {leaf}'s field"), field_ms));
                parts.push((format!("baking {leaf}"), bake_ms));
            }
            for (what, ms) in parts {
                if ms >= 1.0 {
                    self.stages.push((Box::leak(what.into_boxed_str()), ms));
                }
            }
        }
        self.say_why_nothing_is_traced();
    }

    /// Says the traced view's error in the console, once, when it changes.
    ///
    /// A viewport that silently shows nothing is indistinguishable from a model that is not
    /// there, so a shader that failed to compile, a device that refused a buffer or a device
    /// that has gone has to be said rather than stopping at the field that recorded it.
    ///
    /// **Only when it changes.** This is called every frame and the error it reads is a stored
    /// field, not an event — so a part with no bodies is un-compilable for as long as it stays
    /// that way, and saying so each time filled the console with one sentence at the frame
    /// rate. The condition is continuous; the report is not.
    fn say_why_nothing_is_traced(&mut self) {
        let error = self.traced.error().map(|e| e.to_string());
        if error != self.traced_error_said {
            if let Some(why) = &error {
                self.warn(why.clone());
            }
            self.traced_error_said = error;
        }
    }

    // `build_composed_mesh` lived here. It was the viewport's only use of `mesh_field`, and
    // its output could not be made to look right: the grid is uniform over a box stretched by
    // the swept curve, so the tube was 3.1 cells across at the default and 9.3 at triple the
    // resolution, with silhouette faceting that stopped improving after about 124 cells.
    // Raising the setting added triangles to the sphere and left the tube a polygon. The
    // traced view draws the same field per pixel, with no grid, no resolution setting and no
    // such floor. Meshing stays for export, where the cost is paid once and offline.

    /// The grid on the drawing plane for this view, or `None` when there is no plane, the grid
    /// is off, or the plane cannot be seen.
    ///
    /// The pitch is measured where the grid is looked at: a millimetre along the plane's X at
    /// the window's centre, projected. In perspective that varies across the plane and the
    /// centre is the honest single answer; in the square view the plane was set from, it is
    /// exact everywhere. Asked by the drawing and by every snap, so what a snap takes is
    /// exactly what is on screen.
    fn drawing_grid(&self, aspect: f32, rect: egui::Rect) -> Option<PlaneGrid> {
        // Orthographic views only. In perspective the far lines crowd together and the grid
        // read as noise over the model (owner, 2026-09-06); a grid is for the view you measure
        // off, and that is the square one. Not drawn means not snapped to, as everywhere.
        if !self.show_grid || self.camera.projection != Projection::Orthographic {
            return None;
        }
        let frame = self.drawing_frame()?;
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        let (translation, rotation) =
            fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers).ok()?;
        // The window follows the pivot: its foot on the plane is the centre.
        let poi = self.camera.last_poi.cast::<f64>();
        let local = rotation.inverse() * (poi.coords - translation);
        let on_plane = |x: f64, y: f64| {
            Point3::from(rotation * Vector3::new(x, y, 0.0) + translation).cast::<f32>()
        };
        let at = self.camera.project_to_screen(aspect, rect, on_plane(local.x, local.y))?;
        let one_mm =
            self.camera.project_to_screen(aspect, rect, on_plane(local.x + 1.0, local.y))?;
        let px_per_mm = at.distance(one_mm) as f64;
        let pitch = grid_pitch(px_per_mm);
        let centre = ((local.x / pitch).round() as i64, (local.y / pitch).round() as i64);
        // Enough lines to cover the viewport in the square view, and a cap so an oblique
        // perspective — where the far lines crowd together anyway — does not draw hundreds.
        let diagonal = (rect.width().powi(2) + rect.height().powi(2)).sqrt() as f64;
        let cells = (diagonal / (pitch * px_per_mm)).ceil() as i64;
        let reach = (cells / 2 + 2).clamp(4, 60);
        Some(PlaneGrid { translation, rotation, pitch, px_per_mm, centre, reach })
    }

    /// Draws the grid, line by line, projected from the plane.
    ///
    /// A line on a plane projects to a line, so each is exact between its two ends however the
    /// plane is seen. An end behind the camera drops its line rather than streaking it across
    /// the viewport — the same rule the curves follow.
    fn draw_grid(&self, painter: &egui::Painter, aspect: f32, rect: egui::Rect, grid: &PlaneGrid) {
        let on_plane = |x: f64, y: f64| {
            Point3::from(grid.rotation * Vector3::new(x, y, 0.0) + grid.translation).cast::<f32>()
        };
        let project = |x: f64, y: f64| self.camera.project_to_screen(aspect, rect, on_plane(x, y));
        let colour = |index: i64| {
            if index == 0 {
                GRID_AXIS
            } else if index % 5 == 0 {
                GRID_MAJOR
            } else {
                GRID_MINOR
            }
        };
        let (cx, cy) = grid.centre;
        let span = |c: i64| ((c - grid.reach) as f64 * grid.pitch, (c + grid.reach) as f64 * grid.pitch);
        let (x0, x1) = span(cx);
        let (y0, y1) = span(cy);
        for k in (cx - grid.reach)..=(cx + grid.reach) {
            let x = k as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x, y0), project(x, y1)) {
                painter.line_segment([a, b], egui::Stroke::new(1.0_f32, colour(k)));
            }
        }
        for l in (cy - grid.reach)..=(cy + grid.reach) {
            let y = l as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x0, y), project(x1, y)) {
                painter.line_segment([a, b], egui::Stroke::new(1.0_f32, colour(l)));
            }
        }

        // **The numbers, which are the point of a grid.** Lines alone say that something is
        // regular; they do not say how big it is, and a part on screen has no size until
        // something on screen is labelled. Asked for directly: "I want actual numbers displayed
        // on it -- either every 10 mm or 100 or 1000 -- so users know the size."
        //
        // Every line when there is room to write between them, every fifth when there is not.
        // The pitch is a decade, so a fifth is not a number anybody thinks in -- but it is a
        // *label* rather than a step, and a crowded margin is unreadable at any spacing.
        let cell_px = grid.pitch * grid.px_per_mm;
        let step = if cell_px >= 64.0 { 1 } else { 5 };
        let font = egui::FontId::proportional(11.0);
        // Whole millimetres always, since a decade pitch times an integer is one.
        let say = |v: f64| format!("{}", v.round() as i64);
        for k in (cx - grid.reach)..=(cx + grid.reach) {
            if k % step != 0 {
                continue;
            }
            let x = k as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x, y0), project(x, y1))
                && let Some(at) = grid_label_at(rect, a, b)
            {
                painter.text(at, egui::Align2::CENTER_CENTER, say(x), font.clone(), GRID_LABEL);
            }
        }
        for l in (cy - grid.reach)..=(cy + grid.reach) {
            if l % step != 0 {
                continue;
            }
            let y = l as f64 * grid.pitch;
            if let (Some(a), Some(b)) = (project(x0, y), project(x1, y))
                && let Some(at) = grid_label_at(rect, a, b)
            {
                painter.text(at, egui::Align2::CENTER_CENTER, say(y), font.clone(), GRID_LABEL);
            }
        }

        // And what one cell is, once, in the corner -- so the numbers have a unit without every
        // one of them carrying it.
        painter.text(
            egui::pos2(rect.right() - 8.0, rect.bottom() - 8.0),
            egui::Align2::RIGHT_BOTTOM,
            format!("grid {} mm", say(grid.pitch)),
            egui::FontId::proportional(11.0),
            GRID_LABEL,
        );
    }

    /// `point` moved to the nearest intersection of the grid that is drawn, if one is.
    fn grid_snap(&self, aspect: f32, rect: egui::Rect, point: Point3<f64>) -> Option<Point3<f64>> {
        let grid = self.drawing_grid(aspect, rect)?;
        Some(snap_to_grid(point, grid.translation, grid.rotation, grid.pitch))
    }

    /// Sets the display quality and says what that means. `chosen` records that a person
    /// picked it, which the startup correction then leaves alone, and keeps it for the next
    /// session.
    fn set_display_quality(&mut self, quality: DisplayQuality, chosen: bool) {
        self.quality_chosen |= chosen;
        // **A person's choice is kept; a script's is not.** A run is the application driven with
        // no hand on it, and a benchmark that asks for `Fast` to time it would otherwise leave
        // the next session on `Fast` with nothing to say why. Kept even when the rung is the one
        // already showing, because picking the guess is what makes it a choice. Said rather than
        // warned if it cannot be written: the rung has still been set, for this session.
        if chosen
            && !self.script_running
            && !cfg!(test)
            && let Err(why) = display_quality::save_quality(quality)
        {
            self.say(format!(
                "Display quality is {}, but could not be kept for the next session: {why}",
                quality.label()
            ));
        }
        if self.display_quality == quality {
            return;
        }
        self.display_quality = quality;
        self.say(format!(
            "Display quality: {} — {}.",
            quality.label(),
            quality.hint()
        ));
    }

    /// `display quality=<fast|balanced|full|enchilada>`, or `display` alone to hear which.
    fn display_from_console(&mut self, args: &Args) {
        // `display mode=field`: the View menu's Display mode, which a click could set and a line
        // could only cycle. Beside `quality` because the menu puts the two side by side.
        if let Some(wanted) = args.get("mode").and_then(|v| v.as_str().map(str::to_string)) {
            let mode = match wanted.to_ascii_lowercase().as_str() {
                "polygon" | "cage" => ViewMode::Polygon,
                "field" => ViewMode::Field,
                "overlay" | "both" => ViewMode::Overlay,
                _ => {
                    self.warn(format!(
                        "No display mode called {wanted} — polygon, field or overlay"
                    ));
                    return;
                }
            };
            self.view_mode = mode;
            // Which imported solids are drawn as triangles depends on the mode.
            self.refresh_drawn_meshes();
            self.say(format!("Display mode: {}", mode.menu_label()));
            if args.get("quality").is_none() {
                return;
            }
        }
        let Some(wanted) = args.get("quality").and_then(|v| v.as_str().map(str::to_string)) else {
            self.say(format!(
                "Display quality: {} ({}). Say `display quality=<{}>`.",
                self.display_quality.label(),
                if self.quality_chosen { "chosen" } else { "guessed from the adapter" },
                DisplayQuality::ALL.map(|q| q.label().to_lowercase()).join("|"),
            ));
            return;
        };
        match DisplayQuality::from_name(&wanted) {
            Some(quality) => self.set_display_quality(quality, true),
            None => self.warn(format!(
                "No display quality called {wanted}. Qualities: {}",
                DisplayQuality::ALL.map(|q| q.label()).join(", ")
            )),
        }
    }

    /// Captures steps per ray for the view on screen, at each size asked for, and writes the
    /// tables, the static counts and the generated source beside the perf log.
    #[cfg(feature = "march-stats")]
    fn capture_march_stats(&mut self, sizes: &[u32]) {
        let stem = self
            .file
            .as_ref()
            .and_then(|(path, _)| path.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled".to_string());
        let dir = std::env::temp_dir().join("fieldform-march-stats");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.warn(format!("Could not make {}: {e}", dir.display()));
            return;
        }
        for &px in sizes {
            match self.traced.capture_march_stats(px) {
                Ok((capture, source, values)) => {
                    let converged = capture.class(1).len();
                    let capped = capture.class(3).len();
                    let total = capture.steps.len().max(1);
                    let body = format!(
                        "# {stem}, {px} px\n\n{}\n{}\n",
                        self.adapter_line,
                        capture.report()
                    );
                    let _ = std::fs::write(dir.join(format!("{stem}-{px}px.md")), body);
                    let _ = std::fs::write(dir.join(format!("{stem}.wgsl")), &source);
                    let _ = std::fs::write(
                        dir.join(format!("{stem}-static.md")),
                        format!(
                            "# {stem}, static\n\n{}\n## Copies visited per reading\n\n{}",
                            fieldform_gpu::stats::function_report(&source),
                            fieldform_gpu::stats::pattern_widths(&source, &values)
                        ),
                    );
                    self.say(format!(
                        "March stats at {px} px: {} rays, {converged} converged, {capped} out of \
                         steps ({:.2}%). Written to {}",
                        capture.steps.len(),
                        100.0 * capped as f64 / total as f64,
                        dir.display()
                    ));
                }
                Err(why) => self.warn(format!("March stats at {px} px: {why}")),
            }
        }
    }

    /// `perf` toggles per-frame timing lines; `perf on=true` and `perf on=false` set it.
    ///
    /// The loop spec's S6/S8: the application measures itself, so the machine without cargo
    /// can fill the ledger. The file opens with the adapter line, and each traced frame adds
    /// `draft 720px 43.1 ms (march 38.2, shade 0.6, cpu 2.1)` — the GPU's own time where the
    /// card writes timestamps, `(wall …)` where it does not, so an excerpt is self-identifying.
    fn perf_from_console(&mut self, args: &Args) {
        let on = match args.get("on") {
            Some(Value::Bool(on)) => *on,
            Some(other) => {
                other.as_f64().is_some_and(|v| v != 0.0)
                    || other.as_str().is_some_and(|s| matches!(s, "on" | "yes"))
            }
            None => self.perf_file.is_none(),
        };
        if !on {
            if self.perf_file.take().is_some() {
                self.say("Perf lines off.");
            } else {
                self.say("Perf lines were not on. `perf` starts them.");
            }
            return;
        }
        if self.perf_file.is_some() {
            self.say("Perf lines are already on.");
            return;
        }
        let path = std::env::temp_dir().join("fieldform-perf.log");
        let clock = match self.traced.timestamps_available() {
            Some(true) => "GPU timestamps",
            Some(false) => "the wall clock — this device writes no GPU timestamps",
            None => "GPU timestamps if the device writes them, the wall clock otherwise",
        };
        let head = format!("{}\nTimed by {clock}.\n", self.adapter_line);
        match std::fs::write(&path, head) {
            Ok(()) => {
                self.perf_file = Some(path.clone());
                self.say(format!(
                    "Perf lines on: one per traced frame into {}, timed by {clock}. `perf` \
                     again turns them off.",
                    path.display()
                ));
            }
            Err(e) => self.warn(format!("Could not open {}: {e}", path.display())),
        }
    }

    /// Files this frame's CPU time and writes the perf line for every submission the GPU has
    /// since finished. Every frame, not only while logging: the draft tier sizes itself from
    /// the last draft's cost, and that number exists only if the costs are collected.
    fn note_frame_cost(&mut self, cpu_ms: f32) {
        use std::io::Write;
        // **A frame nobody could mistake for interactive says what it spent itself on.** Said in
        // the console rather than only in the perf log, because it is the answer to a question
        // the person is asking at that moment — "why did that hang?" — and because a perf log
        // has to be switched on before the hang rather than after it.
        if cpu_ms >= SLOW_FRAME_MS && !self.stages.is_empty() {
            let mut worst = std::mem::take(&mut self.stages);
            worst.sort_by(|a, b| b.1.total_cmp(&a.1));
            worst.truncate(3);
            let named: Vec<String> =
                worst.iter().map(|(what, ms)| format!("{what} {ms:.0} ms")).collect();
            self.say(format!(
                "That frame took {cpu_ms:.0} ms: {}",
                named.join(", ")
            ));
        }
        self.stages.clear();
        self.frame_cpu_ms.push_back((self.traced.frame(), cpu_ms));
        while self.frame_cpu_ms.len() > 32 {
            self.frame_cpu_ms.pop_front();
        }
        let reports = self.traced.take_reports();
        // **The startup guess, corrected once.** A discrete card was offered `Full` on the
        // strength of being discrete; the first draft it measures is the first fact about how
        // fast it actually is, and a draft over the loop spec's 50 ms budget is a card that
        // should start in the middle. Once only, and never over a choice somebody made: a
        // heavy document later is a heavy document, not a slow machine. A rung kept from an
        // earlier session is such a choice (`starting_quality`), and the correction is set here
        // rather than through `set_display_quality`, so it is never kept: the next session
        // measures its own first draft.
        if !self.quality_chosen
            && let Some(first) = reports.iter().find(|r| !r.still && !r.warm)
        {
            self.quality_chosen = true;
            if first.cost.ms() > DRAFT_BUDGET_MS && self.display_quality == DisplayQuality::Full {
                self.display_quality = DisplayQuality::Balanced;
                self.say(format!(
                    "The first draft frame took {:.0} ms, over the {DRAFT_BUDGET_MS:.0} ms \
                     budget, so Display quality starts on Balanced. View > Display quality \
                     changes it.",
                    first.cost.ms()
                ));
            }
        }
        let rebuilds = self.traced.take_rebuilds();
        let Some(path) = &self.perf_file else { return };
        if reports.is_empty() && rebuilds.is_empty() {
            return;
        }
        let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) else {
            return;
        };
        let elapsed = self.started.elapsed().as_secs_f64();
        for rebuild in rebuilds {
            let _ = writeln!(file, "[{elapsed:7.2}s] {}", rebuild.line());
        }
        for report in reports {
            let cpu = self
                .frame_cpu_ms
                .iter()
                .find(|(frame, _)| *frame == report.frame)
                .map(|(_, ms)| *ms);
            let _ = writeln!(file, "[{elapsed:7.2}s] {}", report.line(cpu));
        }
    }

    /// `grid` toggles the drawing plane's grid; `grid on=true` and `grid on=false` set it.
    fn grid_from_console(&mut self, args: &Args) {
        self.show_grid = match args.get("on") {
            Some(Value::Bool(on)) => *on,
            Some(other) => {
                other.as_f64().is_some_and(|v| v != 0.0)
                    || other.as_str().is_some_and(|s| matches!(s, "on" | "yes"))
            }
            None => !self.show_grid,
        };
        self.say_grid();
    }

    /// Says where the grid stands, and what is still needed to see it.
    ///
    /// One sentence for three switches — the console, the View menu and the button under the
    /// ViewCube — so they cannot describe the same state three ways. "On" is not the same as
    /// "shown": the grid also needs a drawing plane and an orthographic view, and a switch that
    /// says "on" while nothing appears reads as broken, so the missing piece is named.
    fn say_grid(&mut self) {
        if !self.show_grid {
            self.say("Grid off — nothing to snap to");
        } else if self.drawing_plane == DrawingPlane::None {
            self.say(
                "Grid on — it appears once a drawing plane is set and the view is \
                 orthographic. Hold Shift to snap to it",
            );
        } else if self.camera.projection != Projection::Orthographic {
            self.say(
                "Grid on — it shows in an orthographic view; the toggle is under the \
                 ViewCube, left of this one",
            );
        } else {
            self.say(
                "Grid on — hold Shift while dragging a point or placing a line to snap to it",
            );
        }
    }

    /// Translates this frame's key presses through the keymap and dispatches whatever they
    /// are bound to. Nothing here knows what Ctrl+Z *means* — that lives in the keymap, which
    /// is why it can be remapped from user preferences later without touching this code.
    fn handle_keys(&mut self, ui: &egui::Ui) {
        // **Keys being captured for a binding are the panel's, and nothing else's** (§9.2): the
        // next chord pressed is the one given, whatever it is bound to now.
        if self.capturing.is_some() {
            let pressed = ui.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Key { key, pressed: true, modifiers, .. } => Some((*key, *modifiers)),
                    _ => None,
                })
            });
            if let Some((key, modifiers)) = pressed {
                ui.ctx().input_mut(|i| i.consume_key(modifiers, key));
                self.captured(commands::Chord {
                    key: key.name().to_string(),
                    ctrl: modifiers.ctrl || modifiers.command,
                    shift: modifiers.shift,
                    alt: modifiers.alt,
                });
            }
            return;
        }
        // A field with the keyboard owns the plain keys, because there they are text: Tab
        // moves between entries, and cycling the viewport behind them as well is what typing a
        // coordinate and tabbing to the next one used to do.
        //
        // Only the *unmodified* ones. Ctrl+S while a number is selected is still a save — no
        // text entry means anything by it, and a shortcut that stops working depending on
        // where the caret happens to be is its own kind of confusing.
        let typing = ui.ctx().egui_wants_keyboard_input();

        // Key events, kept alongside the egui key they came from so a binding can *consume*
        // its key. Consuming is the point: Tab is also egui's own focus traversal, so a Tab
        // this app acts on has to be taken off the queue, or one press both cycles the mode
        // and walks the focus into the menu bar — which is exactly what it did.
        let pressed: Vec<(egui::Key, egui::Modifiers, Shortcut)> = ui.input(|i| {
            let m = Modifiers {
                ctrl: i.modifiers.ctrl || i.modifiers.command,
                shift: i.modifiers.shift,
                alt: i.modifiers.alt,
            };
            i.events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Key { key, pressed: true, modifiers, .. } => {
                        Some((*key, *modifiers, Shortcut::new(key.name(), m)))
                    }
                    _ => None,
                })
                .collect()
        });

        // **Typing into a pending session's panel needs no click** (`interaction-spec.md` §4.3):
        // a digit or the first letter of a parameter's name starts it, and the plain keys this
        // frame are then text, not bindings -- `h` starting `height` must not also hide.
        let typed_into_panel = self.panel_typed(ui.ctx());

        // The Tab that `raw_input_hook` took off the queue, dispatched here so it still goes
        // through the keymap and can still be rebound — the interception is about *when* egui
        // sees the key, not about who decides what it means. While a session is pending it moves
        // between the panel's rows instead (§4.2).
        if let Some(modifiers) = self.stolen_tab.take() {
            if self.session.is_some() && !modifiers.ctrl && !modifiers.alt {
                // With no panel, Tab with nothing focused goes back to the pending block's first
                // number, so the keyboard can always find its way to the block being made.
                if self.floating_panel {
                    self.panel_tab(modifiers.shift);
                } else if let Some(session) = self.session.as_mut() {
                    session.awaiting = true;
                }
            } else if let Some(binding) = self.keymap.lookup(&Shortcut::new("Tab", modifiers)) {
                let (op, args) = (binding.op.clone(), binding.args.clone());
                self.run_binding(&op, &args);
            }
        }

        for (key, egui_modifiers, shortcut) in pressed {
            if (typing || typed_into_panel) && !shortcut.modifiers.any() {
                continue;
            }
            // **Enter keeps a pending session and Esc takes it back** (`interaction-spec.md` §3.4,
            // §3.5), ahead of any binding: they are what a session is left by, and nothing else
            // is blocked while one is pending.
            if self.session.is_some() && !shortcut.modifiers.any() && matches!(key, egui::Key::Enter | egui::Key::Escape) {
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                if key == egui::Key::Enter {
                    self.commit_session();
                } else {
                    self.cancel_session();
                }
                continue;
            }
            if let Some(binding) = self.keymap.lookup(&shortcut) {
                let (op, args) = (binding.op.clone(), binding.args.clone());
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                let op = self.shortcut_op(op);
                self.run_binding(&op, &args);
            }
        }

        // **Ctrl+C, Ctrl+X and Ctrl+V never arrive as keys.** egui-winit turns them into
        // `Copy`, `Cut` and `Paste` events and drops the key event on the floor
        // (`on_keyboard_input` returns before pushing it), so a keymap bound to the key alone
        // was unreachable from the keyboard: Paste worked from the Edit menu and from nowhere
        // else, and so did Copy and Cut. Read back as the shortcuts they are, so a rebinding
        // in the keymap still decides what they do. A field with the keyboard keeps them,
        // since there they are text.
        if !typing {
            let clipboard: Vec<Shortcut> = ui.input(|i| {
                i.events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::Copy => Some(Shortcut::new("C", Modifiers::ctrl())),
                        egui::Event::Cut => Some(Shortcut::new("X", Modifiers::ctrl())),
                        egui::Event::Paste(_) => Some(Shortcut::new("V", Modifiers::ctrl())),
                        _ => None,
                    })
                    .collect()
            });
            for shortcut in clipboard {
                if let Some(binding) = self.keymap.lookup(&shortcut) {
                    let (op, args) = (binding.op.clone(), binding.args.clone());
                    let op = self.shortcut_op(op);
                    self.run_binding(&op, &args);
                }
            }
        }
    }

    /// What a shortcut means where the cursor is: Ctrl+V lands the copied curve under the
    /// cursor when the cursor is over the viewport (`profile-group-spec.md` §4.4). The same
    /// word from the Edit menu, a row's menu or the console pastes in place, because none of
    /// those has a cursor over the viewport when it fires.
    fn shortcut_op(&self, op: String) -> String {
        if op == "edit.paste" && self.viewport_hover.is_some() {
            "edit.paste_here".to_string()
        } else {
            op
        }
    }
}

impl eframe::App for App {
    /// Takes Tab out of the input before egui can route it to focus.
    ///
    /// This has to happen here and cannot happen anywhere else. egui decides which widget the
    /// focus moves to in `Memory::begin_pass`, reading the raw events *before* an app's `ui`
    /// runs — so consuming the key during `handle_keys`, which is what this used to do, is
    /// always exactly one frame too late. The mode cycled and the focus walked into the menu
    /// bar anyway, and consuming it looked like it should have worked. `raw_input_hook` is the
    /// only point that runs earlier than egui's own reading of the events.
    ///
    /// Only when nothing has the keyboard. In the block list Tab means "next field", which is
    /// exactly what it should mean there — typing a coordinate and tabbing to the next one must
    /// not cycle the viewport behind the panel.
    ///
    /// The key is stashed rather than dropped, because Tab is still a binding: it goes through
    /// the keymap a moment later like every other shortcut, so it stays rebindable.
    fn raw_input_hook(&mut self, ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        // **Anything focused, not specifically a text edit**, and the narrower question was
        // wrong in a way the owner met: "sometimes there is a weird error where it keeps trying
        // to cycle the different view modes". A `DragValue` — which is what every coordinate and
        // size in the block list is — has focus without being a text edit until it is actually
        // being typed into. So the *first* Tab was taken as "next field", egui moved the focus
        // onto the next `DragValue`, and the *second* Tab found no text edit focused and cycled
        // the viewport. Every time, which is what "sometimes" is when the cause is the second
        // press rather than the first.
        //
        // The broader question was rejected here once, on the grounds that a click can leave
        // focus on a button and Tab would then stop cycling. That is now the wanted behaviour
        // rather than the feared one: focus on a button means the pointer is working in the
        // panel, and the panel is where Tab means "next". Clicking the viewport surrenders
        // focus — it senses clicks and is not focusable — so working in the view still cycles
        // modes, which is the only place that gesture was for.
        //
        // The console's own field is the exception, and it has to be taken here for exactly the
        // reason the mode cycle does: egui decides where focus goes before any of this app's ui
        // runs, so a Tab left in the queue has already walked away by the time the console
        // could look at it. Stashed as a completion rather than as a binding, so it neither
        // cycles the viewport nor moves the focus.
        let focused = ctx.memory(|m| m.focused());
        // **A pending block's numbers, with no floating panel: Tab moves between them alone**
        // (`interaction-spec.md` §4.2), as the panel's rows did. egui's own traversal also stops on
        // every V beside a number, so from Z it reached the position's V and not the radius, and
        // what was typed next went nowhere. Taken here because egui has moved the focus by the
        // time any of this app's ui runs; wrapping, so Tab from the last comes back to the first.
        let hop_from = focused
            .filter(|_| self.session.is_some() && !self.floating_panel)
            .and_then(|f| self.pending_numbers.iter().position(|n| *n == f));
        if let Some(at) = hop_from {
            let mut back = None;
            raw_input.events.retain(|event| match event {
                egui::Event::Key { key: egui::Key::Tab, pressed, modifiers, .. }
                    if !modifiers.ctrl && !modifiers.alt && !modifiers.command =>
                {
                    if *pressed {
                        back = Some(modifiers.shift);
                    }
                    false
                }
                _ => true,
            });
            if let Some(back) = back {
                let n = self.pending_numbers.len();
                let next = self.pending_numbers[if back { (at + n - 1) % n } else { (at + 1) % n }];
                ctx.data_mut(|d| d.remove_temp::<String>(self.pending_numbers[at]));
                ctx.memory_mut(|m| m.request_focus(next));
                app::inspector::select_whole_number(ctx, next);
            }
            return;
        }
        let in_console = focused == Some(CONSOLE_INPUT_ID.with("console"));
        // A pending session's panel field is the other exception: Tab there confirms the row and
        // moves to the next (`interaction-spec.md` §4.3), which egui's own traversal would not.
        let in_panel = self
            .session
            .as_ref()
            .is_some_and(|s| focused == Some(app::tool_panel::field_id(s.command.id)));
        if focused.is_some() && !in_console && !in_panel {
            return;
        }
        let mut stolen = None;
        raw_input.events.retain(|event| {
            // Plain Tab and Shift+Tab are the two egui reads as focus movement; a Tab with any
            // other modifier it ignores, so there is nothing to take.
            if let egui::Event::Key { key: egui::Key::Tab, pressed: true, modifiers, .. } = event
                && (!modifiers.any() || modifiers.shift_only())
            {
                stolen = Some(Modifiers {
                    ctrl: modifiers.ctrl || modifiers.command,
                    shift: modifiers.shift,
                    alt: modifiers.alt,
                });
                return false;
            }
            true
        });
        if stolen.is_some() {
            if in_console {
                self.console_complete = true;
            } else {
                self.stolen_tab = stolen;
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // The whole frame's CPU time, for the perf line: what the interface itself costs,
        // separately from what the trace does.
        let frame_started = std::time::Instant::now();
        // First, before this frame asks anything of a GPU that may be gone: a loss found by
        // egui's own submit is answered here, before the next paint uploads anything, and one
        // found on a worker by whichever of this and the call at the end comes first.
        self.keep_the_document_if_the_gpu_was_lost();
        // What the last frame's edits moved, named and said before anything draws or saves.
        self.drain_ownership_changes(false);
        // Before anything draws, so a finished import is on screen this frame rather than the
        // next one, and so a running one keeps the window repainting.
        self.poll_import(ui.ctx());
        self.drive_outputs(ui.ctx());
        self.refresh_surface_display();
        self.sync_window_title(ui.ctx());
        self.guard_close(ui.ctx());
        self.autosave();
        self.drive_snap(ui.ctx());
        self.drive_script(ui.ctx());
        self.help_windows(ui.ctx());
        self.keys_panel(ui.ctx());
        self.import_options_window(ui.ctx());
        self.handle_keys(ui);
        // After the keys, so a row typing began into this frame shows what was typed, and the
        // palette Space opened this frame has the keyboard at once.
        self.tool_panel(ui.ctx());
        self.palette_ui(ui.ctx());
        self.marking_ui(ui.ctx());

        // Controls live in their own top panel, and the viewport gets the central one.
        //
        // They used to share a single panel, with the viewport taking `available_size()` —
        // whatever the controls left over. Every row added to the header therefore ate the
        // viewport, and once the leftover height approached zero the `max(1.0)` below turned
        // the aspect ratio into width-over-one-pixel: the model rendered violently squished
        // laterally. A top panel cannot do that, because the central panel is sized by egui
        // rather than by subtraction.
        egui::Panel::top("fieldform_controls").show_inside(ui, |ui| {
            self.menu_bar(ui);
            self.shelf_strip(ui);
            ui.horizontal(|ui| {
                ui.heading("fieldform");
                // Which build this is. It used to be a warning — an unoptimized bake ran
                // ~70x slower and looked exactly like a hang — and it is now only a label,
                // because the workspace compiles the numeric crates optimized even in a dev
                // build. Kept because knowing which binary is on screen is worth a word, and
                // because this crate's own code really is unoptimized: picking and the cage
                // triangulation live here.
                if cfg!(debug_assertions) {
                    ui.label(egui::RichText::new("dev build").weak()).on_hover_text(
                        "This crate is unoptimized; the field, bake and subdivision crates \
                         are not, so the viewport runs at release speed. See the profile \
                         settings in Cargo.toml.",
                    );
                }
            });

            // One row, one thing: which mode the viewport is in, and whatever the last
            // operation had to say.
            //
            // Save, Undo and Redo were here as one-click items. They read as a second, smaller
            // menu sitting under the real one — the same three words, twice, a centimetre
            // apart — and both are already a keystroke away. A toolbar earns its height by
            // showing something a menu cannot, and the mode is that: it is state, not an
            // action, and it has to be legible without opening anything.
            ui.horizontal(|ui| {
                if ui
                    .button(self.view_mode.label())
                    .on_hover_text("Click or press Tab to cycle")
                    .clicked()
                {
                    self.dispatch("view.cycle_mode", &Args::new());
                }
            });

        });

        self.inspector(ui);

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let available = ui.available_size();
            // A viewport this small cannot be drawn meaningfully, and dividing by its height
            // is what produced the squished render before. Say so instead.
            if available.x < 32.0 || available.y < 32.0 {
                ui.centered_and_justified(|ui| ui.label("Window too small for the viewport"));
                return;
            }

            let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
            let aspect = rect.width() / rect.height();
            // Kept for what places itself against the picture between frames: the tool panel.
            self.viewport_rect = Some(rect);

            // Said on the picture, because the picture is what is wrong. See `compiling_notice`.
            self.compiling_notice(ui, rect);

            // Where the cursor is over the viewport, for a paste that lands under it
            // (`profile-group-spec.md` §4.4). `None` whenever a menu, the console or a panel
            // is under the pointer instead, which is what makes those paste in place.
            self.viewport_hover = response
                .hover_pos()
                .filter(|p| rect.contains(*p))
                .map(|p| (aspect, rect, p));
            // **A right-click opens the marking menu** (`interaction-spec.md` §7) -- except on a
            // corner of a held loop, which keeps the list of its continuities it always offered:
            // that is a question about the corner, not a command, and the corner is what was aimed
            // at. Paste here, which this menu also held, is the empty table's North-West slot.
            if response.secondary_clicked() {
                self.viewport_menu_at = response.interact_pointer_pos();
                let corner = self.viewport_menu_at.and_then(|p| self.joint_at(aspect, rect, p));
                if corner.is_none()
                    && let Some(at) = self.viewport_menu_at
                {
                    self.open_marking_menu(at);
                }
            }
            if self.viewport_menu_at.and_then(|p| self.joint_at(aspect, rect, p)).is_some() {
                response.context_menu(|ui| {
                    if let Some(point) = self.viewport_menu_at.and_then(|p| self.joint_at(aspect, rect, p)) {
                        self.joint_menu(ui, point);
                    }
                });
            }

            // The ViewCube overlay sits inside this rect (upper-right corner) and has its
            // own click handling below; a press starting there shouldn't also begin a
            // tumble/dolly/select on the main viewport underneath it.
            // The cube *and* the buttons beside and under it: a press on any of them is theirs,
            // not the viewport's — see `view_cube_controls_rect`.
            let cube_rect = view_cube_controls_rect(rect);
            let press_over_cube = response
                .interact_pointer_pos()
                .is_some_and(|p| cube_rect.contains(p));

            // Navigation used to require Alt+Shift, which kept the plain buttons free for
            // editing. It is now the other way round: the buttons navigate on their own, and
            // the one gesture that still edits is a drag that *starts on what is already
            // selected*. That is how a modeller with a gizmo behaves — you drag the handle,
            // everything else orbits — and it costs no modifier on either side.
            //
            // Alt+Shift still works, because it costs nothing to keep and hands already know
            // it.
            let nav_mods = ui.input(|i| i.modifiers.alt && i.modifiers.shift);
            // Ctrl on a press that lands on no handle and no manipulator starts a box, which
            // picks every control point inside it (Ctrl+Shift adds). A plain drag there is the
            // tumble and Shift already means "add", so Ctrl was the modifier left.
            let boxing = ui.input(|i| i.modifiers.ctrl || i.modifiers.command) && !nav_mods;

            // Whether a curve is waiting for a click. While one is, the press-time picks below
            // stand aside so a click can only mean "here" — and a drag still turns the view,
            // which is what lets the second point be aimed in perspective at all.
            let placing = self.pending_curve.is_some();

            // A drag that begins on the current selection is a move, not a tumble. Decided at
            // the press, before egui has decided whether the pointer moved far enough to be a
            // drag at all — by the time it has, the pointer is elsewhere and the question of
            // where the gesture *started* can no longer be answered.
            let editable = self.control_editable();

            // The manipulator sees the press before anything else does, and that ordering is
            // the whole of how it coexists with everything already bound to the left button. A
            // handle is drawn on top of what it manipulates, so a click that lands on one is
            // never also a click on the geometry behind it, never a tumble, and never a
            // re-pick of the selection — the three things this same button otherwise means.
            let gizmo_frame = self.gizmo_frame();
            // A control point takes the press ahead of the cage, the bodies and the
            // manipulator. It is a small, deliberate target that is only on offer while its
            // curve is selected, so it cannot be in the way of anything — and a click that
            // lands on one is never also a click on whatever it is drawn over.
            //
            // Asked *here*, above the manipulator, because the manipulator has to know whether
            // to stand aside; the block further down is what acts on it.
            let mut pressed_control_point = None;
            if !press_over_cube
                && !nav_mods
                && !placing
                && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
                && let Some(pos) = response.interact_pointer_pos()
            {
                pressed_control_point = self.control_point_at(aspect, rect, pos);
            }

            // **A control point pressed this frame takes the click off the manipulator.**
            // The manipulator's pivot sits at the middle of what it moves, and on a curve with
            // an odd number of control points that is exactly where one of them is — most
            // visibly on a degree-2 curve seen flat on. The handle drew over the point and took
            // every click aimed at it, so the point could not be selected at all.
            //
            // Priority rather than moving the pivot: a nudge is a guess that fails again as soon
            // as some other point lands where the pivot was nudged to, and "the smaller, more
            // specific target wins" is the rule the rest of the picking path already follows.
            // The manipulator is still there — its axes are clear of the centre — and dragging
            // the point it just selected is the next click.
            if !press_over_cube
                && !nav_mods
                && !placing
                && pressed_control_point.is_none()
                && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
                && let Some(frame) = gizmo_frame
                && let Some(pos) = response.interact_pointer_pos()
            {
                let eye = self.camera.eye.cast::<f64>();
                let grabbed = {
                    let project = self.gizmo_project(aspect, rect);
                    gizmo::hit_test(&frame, pos, &project, eye)
                };
                if let Some(handle) = grabbed {
                    self.press_on_gizmo = true;
                    self.begin_gizmo(&frame, handle, aspect, rect, pos);
                }
            }

            if !press_over_cube
                && !nav_mods
                && !placing
                && editable
                && !self.press_on_gizmo
                && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
            {
                self.press_on_selection = response.interact_pointer_pos().is_some_and(|pos| {
                    let under = pick(
                        self.control(),
                        &self.control_placement(),
                        &self.cage_mesh,
                        &self.camera,
                        aspect,
                        rect,
                        pos,
                    );
                    // Any part of it: pressing one of four picked faces drags all four.
                    self.selection.contains(under)
                });
            }
            if pressed_control_point.is_some() {
                self.press_on_control_point = true;
            }
            // **A curve takes the press ahead of the cage and the bodies, and behind its own
            // control points.** Behind them because a point is the more precise thing to have
            // aimed at and sits on the curve anyway, so the curve would otherwise swallow every
            // click meant for a point. Ahead of the bodies because a curve is reference geometry
            // drawn over the model: it is one pixel wide, it is usually in front of whatever it
            // was drawn against, and a click that lands on it cannot plausibly have meant the
            // solid behind.
            let mut pressed_curve = None;
            if pressed_control_point.is_none()
                && !press_over_cube
                && !nav_mods
                && !placing
                && !self.press_on_gizmo
                && !boxing
                && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
                && let Some(pos) = response.interact_pointer_pos()
            {
                pressed_curve = self.curve_at(aspect, rect, pos);
            }
            if let Some(id) = pressed_curve {
                self.press_on_curve = true;
                let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
                if ui.input(|i| i.modifiers.shift) && self.selected_curve.is_some_and(|s| s != id) {
                    // **Shift+click on another curve adds every point of it**, so several
                    // curves move as one piece under the manipulator. A selected curve with no
                    // point picked stands for all of its points, which is what it moves.
                    let mut picked = if self.selected_points.is_empty() && self.other_points.is_empty() {
                        let s = self.selected_curve.expect("checked");
                        let n = self.doc.curve(s).map_or(0, |c| c.control_points().len());
                        (0..n).map(|i| (s, i)).collect()
                    } else {
                        self.picked_points()
                    };
                    let n = self.doc.curve(id).map_or(0, |c| c.control_points().len());
                    for i in 0..n {
                        if !picked.contains(&(id, i)) {
                            picked.push((id, i));
                        }
                    }
                    self.set_point_selection(&picked);
                    self.say(format!("Added {name}"));
                } else {
                    // Selecting it is what puts its control points on offer and hands it the
                    // manipulator — `select_curve` is the one place that decides who holds those.
                    self.select_curve(id);
                    self.say(format!("Selected {name}"));
                }
            }

            if let Some((curve, index)) = pressed_control_point {
                // **Shift adds or takes away, across curves**; a plain press on a point already
                // picked keeps the lot, so a drag moves all of it, and narrows to that point if
                // it turns out to be a click; a plain press anywhere else picks it alone.
                let add = ui.input(|i| i.modifiers.shift);
                if add {
                    self.toggle_point(curve, index);
                } else if self.is_picked(curve, index) {
                    self.pending_narrow = Some((curve, index));
                } else {
                    self.pick_only(curve, index);
                }
                // Whatever else was selected is not what the handles are on now.
                self.selection.clear();
            }
            if boxing
                && !press_over_cube
                && !placing
                && pressed_control_point.is_none()
                && !self.press_on_gizmo
                && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
                && let Some(pos) = response.interact_pointer_pos()
            {
                self.marquee = Some((pos, ui.input(|i| i.modifiers.shift)));
                self.press_on_marquee = true;
            }

            let on_gizmo = self.press_on_gizmo
                || self.gizmo_drag.is_some()
                || self.press_on_control_point;
            let moving = self.press_on_selection
                || self.move_state.is_some()
                || on_gizmo
                || self.press_on_marquee;

            if !press_over_cube && !moving {
                // The pivot is chosen on the *press*, not on the drag that may follow.
                //
                // egui only reports a drag once the pointer has moved past its threshold, so
                // aiming the pivot with a click — which is how one aims it — set nothing at
                // all, and whether a press happened to count came down to hand tremor. That
                // was the "sometimes it doesn't remember where I clicked", and it is why the
                // crosshair stayed where it was: nothing had asked it to move.
                //
                // Any of the three navigation buttons aims it. They differ in what they then
                // *do* with the camera — tumble, dolly, pan — but all three are ways of saying
                // "this is the part I am looking at", and having only one of them set the
                // pivot meant re-aiming with the left button before every other gesture.
                let nav_pressed = ui.input(|i| {
                    i.pointer.button_pressed(egui::PointerButton::Primary)
                        || i.pointer.button_pressed(egui::PointerButton::Secondary)
                        || i.pointer.button_pressed(egui::PointerButton::Middle)
                });
                if nav_pressed {
                    if let Some(press_pos) = response.interact_pointer_pos() {
                        let (origin, dir) = self.camera.pick_ray(aspect, press_pos, rect);
                        // A miss leaves the previous pivot alone rather than flinging it at
                        // `camera.target`, which POI-anchored tumbling drifts away from the
                        // model over repeated orbits. (the reference camera reuses
                        // `_current_poi` for the same reason.)
                        if let Some(poi) = self.pick_pivot(origin, dir, aspect, rect, press_pos) {
                            self.camera.last_poi = poi;
                        }
                    }
                }
                if response.drag_started_by(egui::PointerButton::Primary) {
                    self.camera.begin_tumble(self.camera.last_poi);
                }
                if response.drag_started_by(egui::PointerButton::Secondary) {
                    self.camera.begin_dolly();
                }
                if response.drag_started_by(egui::PointerButton::Middle) {
                    self.camera.begin_pan();
                }

                let delta = response.drag_delta();
                if response.dragged_by(egui::PointerButton::Primary) {
                    self.camera.tumble_drag(delta);
                }
                if response.dragged_by(egui::PointerButton::Secondary) {
                    self.camera.dolly_drag(delta);
                }
                if response.dragged_by(egui::PointerButton::Middle) {
                    self.camera.pan(delta);
                }
            }

            // Ending a gesture is *not* gated on the modifiers, deliberately. It used to be,
            // so letting go of Alt+Shift before the mouse button left the tumble latched: the
            // camera stayed in a drag it would never be told had finished, and the crosshair
            // sat at that pivot indefinitely.
            if !response.dragged_by(egui::PointerButton::Primary) {
                self.camera.end_tumble();
            }
            if !response.dragged_by(egui::PointerButton::Secondary) {
                self.camera.end_dolly();
            }
            if !response.dragged_by(egui::PointerButton::Middle) {
                self.camera.end_pan();
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera.zoom(scroll);
                }
            }

            // **A curve being placed takes the click ahead of everything else the button
            // means.** A click is a point. A drag is still a tumble, so the view turns between
            // the two clicks; the press-time picks above stood aside so that a press cannot
            // select a curve or grab a handle on the way to placing a point.
            if placing
                && !press_over_cube
                && response.clicked_by(egui::PointerButton::Primary)
                && let Some(pos) = response.interact_pointer_pos()
            {
                let snapping = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
                let grid = ui.input(|i| i.modifiers.shift);
                self.place_curve_point(aspect, rect, pos, snapping, grid);
            }
            // Escape ends the placement and nothing else. What has been drawn so far is
            // nothing, so there is nothing to put back.
            if self.pending_curve.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.cancel_curve_placement();
            }

            // Field mode has no cage to aim at, so a plain click there means the body under
            // the cursor instead — the same gesture, aimed at whatever the mode is drawing,
            // which is the rule the whole picking path already follows.
            if !press_over_cube
                && !on_gizmo
                && !placing
                && !self.press_on_curve
                && !self.press_on_marquee
                && !self.view_mode.cage_is_editable()
                && response.clicked_by(egui::PointerButton::Primary)
            {
                if let Some(pos) = response.interact_pointer_pos() {
                    let (origin, dir) = self.camera.pick_ray(aspect, pos, rect);
                    self.clear_point_selection();
                    self.selected_curve = None;
                    // `select_body_under` already clears the body when the trace hits nothing,
                    // which is the same rule: a click on empty space selects empty space.
                    self.select_body_under(origin, dir);
                }
            }

            // Selecting and moving. A click selects whatever is under it; a drag moves it,
            // but only when the drag began on the thing already selected — otherwise the same
            // drag is a tumble, decided above.
            //
            // **Shift adds to what is picked rather than replacing it**, so four faces can be
            // moved as one piece. Read here rather than at the press: it is the click that
            // decides, and holding Shift down through a drag means nothing else in this view.
            let adding = ui.input(|i| i.modifiers.shift) && !nav_mods;
            if !press_over_cube && editable && !on_gizmo {
                if !placing
                    && !self.press_on_marquee
                    && response.clicked_by(egui::PointerButton::Primary)
                {
                    if let Some(pos) = response.interact_pointer_pos() {
                        // Picking on the cage is picking somewhere else, so the handles leave
                        // the curve rather than two selections both claiming them.
                        self.clear_point_selection();
                        self.selected_curve = None;
                        let under = pick(
                            self.control(),
                            &self.control_placement(),
                            &self.cage_mesh,
                            &self.camera,
                            aspect,
                            rect,
                            pos,
                        );
                        // **Shift adds and takes away; a plain click replaces.** The one rule
                        // worth keeping across every list and viewport there is, and the same
                        // one a curve's control points already follow here.
                        if adding {
                            self.selection.toggle(under);
                        } else {
                            self.selection = Picked::one(under);
                            // A click that hit nothing means nothing is selected, and the
                            // manipulator goes with it. Leaving it on whatever was last touched
                            // makes clicking empty space the one gesture with no way to say
                            // "none of this" — and the handles then sit over geometry the user
                            // has deliberately stopped pointing at. Shift is exempt: a miss
                            // while adding is a miss, not a request to drop what is held.
                            if self.selection.is_empty() {
                                self.selected_body = None;
                            }
                        }
                    }
                }
                if self.press_on_selection {
                    if response.drag_started_by(egui::PointerButton::Primary) {
                        if let Some(pos) = response.interact_pointer_pos() {
                            self.begin_move(aspect, rect, pos);
                        }
                    }
                    if response.dragged_by(egui::PointerButton::Primary) {
                        if let Some(pos) = response.interact_pointer_pos() {
                            self.continue_move(aspect, rect, pos);
                        }
                    }
                }
                if response.drag_stopped_by(egui::PointerButton::Primary) {
                    self.end_move();
                }
            }
            // **A control point's own drag**, which is what makes a point placeable in a view
            // where the manipulator cannot help: looking straight down a plane, its axis toward
            // the camera is a dot and only the plane handles work, so the point had to be moved
            // by aiming at a handle rather than at the point. Mutually exclusive with the
            // manipulator's by construction -- the press that starts this is the one the
            // manipulator now stands aside for.
            if self.press_on_control_point {
                if response.drag_started_by(egui::PointerButton::Primary)
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    self.begin_point_drag(aspect, rect, pos);
                }
                if response.dragged_by(egui::PointerButton::Primary)
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    let snapping = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
                    let grid = ui.input(|i| i.modifiers.shift);
                    self.continue_point_drag(aspect, rect, pos, snapping, grid);
                }
            }
            if self.press_on_marquee
                && let Some((from, _)) = self.marquee
                && let Some(to) = response.interact_pointer_pos().or(response.hover_pos())
            {
                let bx = egui::Rect::from_two_pos(from, to);
                let blue = egui::Color32::from_rgb(120, 170, 255);
                ui.painter().rect_filled(bx, 0.0, blue.gamma_multiply(0.12));
                ui.painter().rect_stroke(
                    bx,
                    0.0,
                    egui::Stroke::new(1.0_f32, blue),
                    egui::StrokeKind::Inside,
                );
            }
            if self.point_drag.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                // Abandoned: put every point back where it started and forget the gesture --
                // the blocks too, or a held end would be put back and then settled onto where
                // its block was dragged, which is what `cancel_gizmo` already does.
                if let Some(state) = self.point_drag.take() {
                    if let Some(was) = state
                        .gesture
                        .graph_before
                        .clone()
                        .or_else(|| state.others.iter().find_map(|m| m.gesture.graph_before.clone()))
                    {
                        self.restore_part_graph(was);
                    }
                    if let Some(c) = self.doc.curve_mut(state.curve) {
                        for (i, p) in state.indices.iter().zip(&state.start_positions) {
                            c.set_control_point(*i, *p);
                        }
                    }
                    for m in &state.others {
                        if let Some(c) = self.doc.curve_mut(m.curve) {
                            for (i, p) in m.indices.iter().zip(&m.before) {
                                c.set_control_point(*i, *p);
                            }
                        }
                    }
                }
                self.rebuild_curve_display();
            }

            // The manipulator's drag. Mutually exclusive with the cage's by construction:
            // `press_on_gizmo` is only set when the press landed on a handle, and the cage's
            // `press_on_selection` is not even evaluated when it did.
            if self.press_on_gizmo
                && response.dragged_by(egui::PointerButton::Primary)
                && let Some(pos) = response.interact_pointer_pos()
            {
                // Ctrl rounds a depth to whole millimetres, the way it rounds a curve's points.
                let snap = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
                self.continue_gizmo(aspect, rect, pos, snap);
            }
            // Escape abandons the gesture. Checked before the release below, so a drag
            // cancelled with the button still down is not then committed by letting go.
            if self.gizmo_drag.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.cancel_gizmo();
                self.press_on_gizmo = false;
            }

            // The press is over once the button is up, whatever it turned out to be.
            //
            // The manipulator commits here rather than on `drag_stopped`, because a press that
            // never moved far enough to count as a drag still has to be closed out — otherwise
            // a click on a handle leaves a drag live until the next one begins.
            if ui.input(|i| i.pointer.button_released(egui::PointerButton::Primary)) {
                // A press on a picked point that never became a drag was a click on it.
                if let Some((curve, index)) = self.pending_narrow.take()
                    && self.point_drag.is_none()
                {
                    self.pick_only(curve, index);
                }
                if self.point_drag.is_some() {
                    self.end_point_drag();
                }
                if self.gizmo_drag.is_some() {
                    self.end_gizmo();
                }
                if let Some((from, adding)) = self.marquee.take()
                    && let Some(to) = response.interact_pointer_pos().or(response.hover_pos())
                {
                    self.finish_marquee(aspect, rect, from, to, adding);
                }
                self.press_on_gizmo = false;
                self.press_on_control_point = false;
                self.press_on_curve = false;
                self.press_on_selection = false;
                self.press_on_marquee = false;
            }

            // Hover preview: what would be selected if the user clicked right now. Only
            // while idle (not navigating, not mid-move) — otherwise it'd be recomputing a
            // pick against a mouse that's busy doing something else.
            // What the manipulator would grab, which also suppresses the cage's own hover
            // below: a handle sits over the geometry, so highlighting both at once would
            // promise a pick that cannot happen.
            self.gizmo_hover = match (self.gizmo_drag.is_none(), gizmo_frame, response.hover_pos())
            {
                (true, Some(frame), Some(pos)) => {
                    let eye = self.camera.eye.cast::<f64>();
                    let project = self.gizmo_project(aspect, rect);
                    gizmo::hit_test(&frame, pos, &project, eye)
                }
                _ => None,
            };

            self.hover = if !press_over_cube
                && editable
                && self.move_state.is_none()
                && self.gizmo_hover.is_none()
                && self.gizmo_drag.is_none()
            {
                response
                    .hover_pos()
                    .map(|pos| {
                        pick(
                            self.control(),
                            &self.control_placement(),
                            &self.cage_mesh,
                            &self.camera,
                            aspect,
                            rect,
                            pos,
                        )
                    })
                    .unwrap_or(Selection::None)
            } else {
                Selection::None
            };

            let mvp = self.camera.mvp(aspect);
            // The cage draws through its placement, exactly as picking now reads through it.
            // These two must move together or they say different things about where the same
            // control mesh is — which is the whole failure the picking rule names.
            let cage_mvp = mvp
                * self
                    .control_placement()
                    .to_homogeneous()
                    .cast::<f32>();
            const CAGE_RGB: [f32; 3] = [0.55, 0.62, 0.75];

            // Draw order matters: without a depth buffer, later draws paint over earlier
            // ones, so the translucent cage has to be issued after the field it sits over.
            // Imported geometry, in every mode and before the cage. Translucent because
            // there is no depth buffer here — an opaque draw would paint over the traced
            // field whatever was actually in front — and being able to see through it is
            // the honest way to show something the tracer is not drawing.
            if !self.imported_mesh.indices.is_empty() {
                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    CageDrawCallback {
                        mvp,
                        color: [0.62, 0.66, 0.70, 0.75],
                        target: render::DrawTarget::Imported,
                    },
                ));
            }

            match self.view_mode {
                ViewMode::Polygon => {
                    if self.control_visible() {
                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            CageDrawCallback {
                                mvp: cage_mvp,
                                color: [CAGE_RGB[0], CAGE_RGB[1], CAGE_RGB[2], 1.0],
                                target: render::DrawTarget::Cage,
                            },
                        ));
                    }
                }
                ViewMode::Field => self.draw_traced(ui, rect),
                ViewMode::Overlay => {
                    self.draw_traced(ui, rect);
                    if self.control_visible() {
                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            CageDrawCallback {
                                mvp: cage_mvp,
                                color: [0.45, 0.72, 0.95, 0.5],
                                target: render::DrawTarget::Cage,
                            },
                        ));
                    }
                }
            }

            // Tessellations on show, over whatever the mode drew — each in place of the body it
            // meshes, which the trace has already left out.
            if !self.outputs.showing.is_empty() {
                let ppp = ui.ctx().pixels_per_point();
                let towards_eye = (self.camera.eye - self.camera.target)
                    .try_normalize(1e-9)
                    .unwrap_or_else(nalgebra::Vector3::z);
                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    preview::PreviewCallback {
                        mvp,
                        towards_eye,
                        size: [
                            (rect.width() * ppp).round().max(1.0) as u32,
                            (rect.height() * ppp).round().max(1.0) as u32,
                        ],
                        meshes: self
                            .outputs
                            .showing
                            .iter()
                            .map(|k| (*k, [0.74, 0.76, 0.80]))
                            .collect(),
                    },
                ));
            }

            // The splash, over everything the viewport drew. It goes on the first click in
            // the viewport or the first body added, because either one means the person is
            // past looking at it — and it is not modal, so the menus and the panel work
            // underneath it and dismiss it as soon as they change what is on screen.
            if self.splash_shown {
                self.draw_splash(ui, rect);
                if response.clicked() || response.dragged() {
                    self.splash_shown = false;
                }
            }
            // The two fast paths, said once on an empty document until the palette has been
            // opened three times (`interaction-spec.md` §10.3).
            if self.first_run_hint_due() {
                self.draw_first_run_hint(ui, rect);
            }

            // The pivot marker belongs to the camera, not to cage editing, so it is drawn in
            // every mode. `active_poi` is Some only while tumbling or dollying, which is
            // exactly when the pivot is worth showing and why panning does not raise it.
            if let Some(poi) = self.camera.active_poi() {
                if let Some(screen) = self.camera.project_to_screen(aspect, rect, poi) {
                    draw_poi_marker(ui.painter(), screen);
                }
            }

            // Gated on the cage actually being drawn, not merely on the mode being one that
            // would draw it. A highlight is a mark *on* the cage; with no cage on screen it
            // becomes a coloured dot floating in the viewport at coordinates nothing else is
            // using — which is precisely what a suppressed cage produced, and it looked like a
            // stray artefact rather than like the selection it was.
            if self.control_editable() {
                if !self.selection.contains(self.hover) {
                    draw_highlight(
                        ui.painter(),
                        self.control(),
                        &self.control_placement(),
                        &self.camera,
                        aspect,
                        rect,
                        self.hover,
                        HOVER_COLOR,
                        HOVER_FILL,
                        2.0,
                    );
                }
                for part in self.selection.parts() {
                    draw_highlight(
                        ui.painter(),
                        self.control(),
                        &self.control_placement(),
                        &self.camera,
                        aspect,
                        rect,
                        *part,
                        SELECTION_COLOR,
                        SELECTION_FILL,
                        3.0,
                    );
                }
            }

            // The drawing plane's grid, over the model and under the curves. Drawn whenever a
            // plane is active: it is the one visible sign that clicks now land there, and Shift
            // snaps to it, so it has to be seen to be aimed at.
            if let Some(grid) = self.drawing_grid(aspect, rect) {
                self.draw_grid(ui.painter(), aspect, rect, &grid);
            }

            // Curves draw over the surface deliberately: when subtracting, the curve is
            // inside the solid it is cutting, and hiding it would hide the thing being aimed.
            let visible_curves: Vec<(CurveId, Vec<Point3<f32>>)> = self
                .curve_display
                .iter()
                .filter(|(id, _)| !self.hidden_curves.contains(id))
                .cloned()
                .collect();
            // Every curve of a selected profile row, not only the one curve a click picks:
            // selecting the row selects its curves, and the viewport says so.
            let highlighted = self.highlighted_curves();
            // The axes whose own note says they could not be re-derived — the same map the panel
            // reads, so the two cannot say different things about one line.
            let stale: Vec<CurveId> = self.surface_axis_notes.keys().copied().collect();
            draw_curves(
                ui.painter(),
                &self.camera,
                aspect,
                rect,
                &visible_curves,
                &highlighted,
                &stale,
            );
            // **The centerline a selected loft built for itself, dashed** -- the same mark a
            // stale axis gets, and for a related reason: it is a line the document does not
            // hold. Dashed is what says "derived, not drawn"; the panel offers to make it real.
            if let Some(derived) = self.selected_derived_centerline() {
                draw_dashed_polyline(
                    ui.painter(),
                    &self.camera,
                    aspect,
                    rect,
                    &derived,
                    egui::Stroke::new(1.5_f32, CURVE_COLOR),
                );
            }
            for (id, edges) in &self.surface_display {
                if self.hidden_surfaces.contains(id) {
                    continue;
                }
                draw_surface_wire(ui.painter(), &self.camera, aspect, rect, edges);
            }

            // **The seed of whichever axis is selected**, and only that one: every seed at once
            // would be a field of rings over a part that has several, and the question the drawing
            // answers — "where is the number I am dragging" — is about the one in the panel.
            if let Some((seed, stands)) = self.selected_axis_seed() {
                draw_axis_seed(ui.painter(), &self.camera, aspect, rect, seed, stands);
            }

            // The curve being placed, over the curves that exist. A gesture that waited
            // invisibly would be the trap the "no modes" rule is about, so it shows where the
            // next click lands and what is already down.
            if self.pending_curve.is_some() {
                // Not refreshed while dragging: the view is turning, the cursor is not aiming,
                // and a field trace per frame would be spent on a preview nobody is reading.
                if !response.dragged() {
                    let snapping = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
                    let grid = ui.input(|i| i.modifiers.shift);
                    self.refresh_placement_preview(
                        aspect,
                        rect,
                        response.hover_pos(),
                        snapping,
                        grid,
                    );
                }
                self.draw_placement(ui.painter(), aspect, rect);
            }

            // The original CAD, over the field derived from it. Gathered per frame rather than
            // cached because it is only walked when the toggle is on, and the toggle is a thing
            // a person turns on to look at something rather than a state they model in.
            if !self.showing_original.is_empty() {
                let edges = self.original_cad_edges();
                if !edges.is_empty() {
                    draw_brep_edges(ui.painter(), &self.camera, aspect, rect, &edges);
                }
            }

            // **Control points on every curve**, while View > Show control points is on -- it
            // is unless turned off, asked for as "points should stay turned on". The selected
            // curve's are drawn as they were, with its control polygon; every other visible
            // curve's are smaller and dimmer, without the polygon, so a point on any curve can
            // be clicked, Shift+clicked into the selection or boxed with Ctrl+drag. A picked
            // point is drawn picked on whichever curve it is.
            let hover = response
                .hover_pos()
                .and_then(|pos| self.control_point_at(aspect, rect, pos));
            for curve_id in self.handle_curves() {
                let Some(curve) = self.doc.curve(curve_id) else { continue };
                let selected = self.selected_curve == Some(curve_id);
                let weights = curve.weights().to_vec();
                let points = curve.control_points().to_vec();

                // The control polygon: the points a curve is pulled by, joined in the order it
                // runs, so the shape being edited is legible before it is dragged. Above degree
                // one only -- a degree-one curve *is* its control polygon -- and only on the
                // selected curve, dashed and dim, because it is scaffolding for the thing being
                // edited and must not compete with the curve it explains.
                if selected && curve.degree() > 1 {
                    let projected: Vec<Option<egui::Pos2>> = points
                        .iter()
                        .map(|p| self.camera.project_to_screen(aspect, rect, p.cast::<f32>()))
                        .collect();
                    // Segment by segment, so a point behind the camera breaks the polygon
                    // rather than joining two points that are not neighbours.
                    for pair in projected.windows(2) {
                        let (Some(a), Some(b)) = (pair[0], pair[1]) else { continue };
                        for shape in egui::Shape::dashed_line(
                            &[a, b],
                            egui::Stroke::new(1.0_f32, CONTROL_POLYGON_COLOR),
                            3.0,
                            3.0,
                        ) {
                            ui.painter().add(shape);
                        }
                    }
                }

                for (i, p) in points.iter().enumerate() {
                    let Some(at) = self.camera.project_to_screen(aspect, rect, p.cast::<f32>())
                    else {
                        continue;
                    };
                    let (base, colour) = if self.is_picked(curve_id, i) {
                        (4.5, SELECTION_COLOR)
                    } else if hover == Some((curve_id, i)) {
                        (4.0, HOVER_COLOR)
                    } else if selected {
                        (3.0, CURVE_SELECTED_COLOR)
                    } else {
                        (2.5, CONTROL_POLYGON_COLOR)
                    };
                    // Heavier points draw bigger, so what a curve is pulled by is visible on the
                    // curve. The fourth root keeps a weight of 16 twice the size of a weight of 1.
                    let weight = weights.get(i).copied().unwrap_or(1.0);
                    let size = base * weight.max(1e-6).powf(0.25).clamp(0.6, 2.2) as f32;
                    // Square, where a cage vertex is round: two kinds of point that move under
                    // the same handles should still be tellable apart at a glance.
                    ui.painter().rect_filled(
                        egui::Rect::from_center_size(at, egui::vec2(size * 2.0, size * 2.0)),
                        1.0,
                        colour,
                    );
                }
            }

            // Every point body, which the tracer cannot draw because a point encloses no
            // volume. Round where a curve's control points are square, so the two do not read
            // as the same kind of thing: those are handles that shape something else, and this
            // is a body in its own right.
            //
            // Drawn with the gizmo rather than with the model, and outlined, because it has no
            // surface to be occluded by and no size to be judged against — a flat dot at a
            // fixed radius is the honest way to show something with no extent.
            //
            // The unselected ones are the size a curve's control points are, which is what was
            // asked for: the two are the same size of thing on screen, and the selected one is
            // bigger because it is the one being dragged.
            let selected_point = self.selected_point_node();
            for (id, at) in self.point_positions() {
                if let Some(screen) =
                    self.camera.project_to_screen(aspect, rect, at.cast::<f32>())
                {
                    let here = Some(id) == selected_point;
                    let (radius, colour) =
                        if here { (5.0, SELECTION_COLOR) } else { (3.0, GUIDE_COLOR) };
                    ui.painter().circle_filled(screen, radius, colour);
                    ui.painter().circle_stroke(
                        screen,
                        radius,
                        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(60, 50, 10)),
                    );
                }
            }

            // **The end a snap would take, drawn while the modifier is held.**
            //
            // Snapping was reported as not working three times, and the reason it could not be
            // diagnosed from the outside is that there was nothing to see: a snap that finds no
            // target and a snap that is not looking are the same picture. So the target is
            // drawn. A box appears over the end that would be taken, and if none appears then
            // nothing is within reach — which is the answer, rather than a mystery.
            //
            // It is drawn whenever the modifier is down and a curve end could move, not only
            // mid-drag, so the reach can be learned by waving the cursor around rather than by
            // committing to a drag and finding out.
            if let Some(at) = self.snap_target(aspect, rect, ui) {
                if let Some(screen) =
                    self.camera.project_to_screen(aspect, rect, at.cast::<f32>())
                {
                    let half = SNAP_BOX_PIXELS * 0.5;
                    let box_ = egui::Rect::from_center_size(
                        screen,
                        egui::vec2(SNAP_BOX_PIXELS, SNAP_BOX_PIXELS),
                    );
                    ui.painter().rect_stroke(
                        box_,
                        0.0,
                        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(90, 190, 110)),
                        egui::StrokeKind::Middle,
                    );
                    // A cross through it, so it reads as a target rather than as a selection.
                    let ink = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(90, 190, 110));
                    ui.painter().line_segment(
                        [
                            egui::pos2(screen.x - half, screen.y),
                            egui::pos2(screen.x + half, screen.y),
                        ],
                        ink,
                    );
                    ui.painter().line_segment(
                        [
                            egui::pos2(screen.x, screen.y - half),
                            egui::pos2(screen.x, screen.y + half),
                        ],
                        ink,
                    );
                }
            }

            // Every plane, as a finite square standing in for an infinite one.
            let selected_plane = self.selected_plane().map(|(id, _, _)| id);
            // Once, not once per plane: every plane wants the same answer -- how big the scene is.
            let (lo, hi) = self.pick_bounds();
            for (id, origin, facing) in self.plane_frames() {
                let here = Some(id) == selected_plane;
                let scene = (hi - lo).norm();
                // Sized from the scene, and from the view when the scene has no size to give
                // — a part that is nothing but a plane, or an empty document. Either way the
                // square is a drawing aid, so it is deliberately bigger than what it cuts.
                let span = if scene > 1e-6 {
                    scene * 0.6
                } else {
                    self.camera.view_half_height(origin.cast::<f32>()) as f64
                };
                // **Centred on the plane's own origin.** It used to be centred on the scene,
                // projected onto the plane -- so the square stayed over the model wherever the
                // origin went, and the origin was the one thing it did not show. Two reports,
                // and they are the same one: turning a plane looked like nothing happened,
                // because the square pivoted about the scene rather than about the point the
                // plane actually turns about; and a plane sitting on a point slid along itself
                // when the point moved, because a half-space genuinely does not change when its
                // origin moves within it, and the only thing that could have shown the origin
                // moving was the square, which was not drawn there.
                //
                // The origin is what the plane turns about and what it sits on, so it is the
                // middle of the square.
                let centre = origin;
                // The plane's own X and Y, so the square is drawn the way the plane is turned
                // rather than the way an arbitrary choice of "some direction not along the
                // normal" happens to fall. That choice is what made a roll invisible, and it
                // also made the square jump as a plane was tilted past the axis it was picked
                // against.
                let u = facing * Vector3::x() * span;
                let v = facing * Vector3::y() * span;
                let corners = [centre + u + v, centre - u + v, centre - u - v, centre + u - v];
                let screen: Vec<egui::Pos2> = corners
                    .iter()
                    .filter_map(|p| {
                        self.camera.project_to_screen(aspect, rect, p.cast::<f32>())
                    })
                    .collect();
                // All four or none: a partly-projected quad closes across the viewport rather
                // than running off it, which reads as a shape that is not there.
                if screen.len() == 4 {
                    ui.painter().add(egui::Shape::convex_polygon(
                        screen.clone(),
                        if here { PLANE_PREVIEW_FILL } else { GUIDE_FILL },
                        egui::Stroke::NONE,
                    ));
                    let edge = if here {
                        egui::Stroke::new(1.5_f32, SELECTION_COLOR)
                    } else {
                        egui::Stroke::new(1.0_f32, GUIDE_COLOR)
                    };
                    for i in 0..4 {
                        ui.painter().line_segment([screen[i], screen[(i + 1) % 4]], edge);
                    }
                }
            }

            // Last, so it is on top of everything: there is no depth buffer, and a handle
            // that can be hidden by the model is a handle that cannot be clicked.
            if let Some(frame) = self.gizmo_frame() {
                let live = self.gizmo_drag.as_ref().map(|d| d.drag.handle);
                {
                    let project = self.gizmo_project(aspect, rect);
                    gizmo::paint(ui.painter(), &frame, self.gizmo_hover, live, &project);
                }
                // The number, beside the gizmo, while a drag is live. Engineers need the value
                // as well as the picture — the same reason cage editing shows one.
                if let Some(drag) = &self.gizmo_drag
                    && !drag.readout.is_empty()
                    && let Some(at) = self.camera.project_to_screen(
                        aspect,
                        rect,
                        frame.pivot.cast::<f32>(),
                    )
                {
                    let text = drag.readout.clone();
                    let where_at = at + egui::vec2(14.0, -22.0);
                    let galley = ui.painter().layout_no_wrap(
                        text,
                        egui::FontId::monospace(13.0),
                        egui::Color32::WHITE,
                    );
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(where_at, galley.size()).expand(4.0),
                        3.0,
                        egui::Color32::from_black_alpha(190),
                    );
                    ui.painter().galley(where_at, galley, egui::Color32::WHITE);
                }
            }

            // Advanced before the cube is drawn, so the cube shows the orientation this frame
            // actually renders with rather than the previous one's — the widget is a read-out
            // of the camera, and a frame of lag in it reads as the cube lagging the model.
            // The settle is driven by time, not by input, so nothing else would bring about
            // the frame it happens in: an application nobody is touching draws nothing, and a
            // draft would sit on screen until the next unrelated click.
            if let Some(wait) = self.traced.settle_due_in() {
                ui.ctx().request_repaint_after(wait);
            }

            let dt = ui.input(|i| i.stable_dt).min(0.1);
            if self.camera.tick_snap(dt) {
                // A snap is time-driven, not input-driven, so nothing else would ask for the
                // next frame and the animation would advance only when the mouse moved.
                ui.ctx().request_repaint();
            }
            if self.show_view_cube {
                // **A face click sets the plane you draw on, as well as the view.** Picking
                // a view and picking a plane are one act — you turned to look at the back
                // because that is where you are about to draw — and `sketcher-spec.md` §5 is
                // built around that being one gesture.
                //
                // Faces only. An edge or a corner names no single plane, so it keeps orienting
                // and nothing else.
                let grid_before = self.show_grid;
                if let Some(region) =
                    show_view_cube(ui, &mut self.camera, rect, &mut self.show_grid)
                {
                    if let Some(plane) = plane_of_face(region) {
                        // The cube has already swung the camera, and it did so with the face's
                        // own *sign*: clicking BOTTOM looks up from below. Swinging again from
                        // the plane would use the unsigned axis and flip that to a top view, so
                        // this sets the plane and leaves the camera exactly where the cube put
                        // it.
                        self.use_drawing_plane(plane, Swing::No);
                    }
                }
                if self.show_grid != grid_before {
                    self.say_grid();
                }
            }
        });

        // And again once the trace has been submitted: a loss one of its submits found leaves
        // the frame to egui's paint, which panics on its first upload to a dead device (it
        // asks for a staging buffer and is given none) before any next frame begins. This is
        // the only chance such a frame gets.
        self.keep_the_document_if_the_gpu_was_lost();

        // A pending session is kept once its block is no longer what is selected (§3.4); then what
        // the panels, the handles and the notebook changed this frame is written as console lines,
        // once nothing is mid-gesture (§6.2).
        self.keep_when_selection_moves(ui.ctx());
        // **Enter in one of the pending block's numbers keeps the block** -- the value first, which
        // its field wrote while it was drawn. Tab is what moves between them, so Enter can mean
        // done, which is what a single press meant with the floating panel too.
        if std::mem::take(&mut self.keep_pending) {
            self.commit_session();
        }
        self.echo_when_quiet(ui.ctx());

        // Last, so the number is the whole frame's.
        self.note_frame_cost(frame_started.elapsed().as_secs_f32() * 1e3);
    }
}

/// Why no cage is on screen in a mode that would otherwise draw one.
///
/// Four different situations look identical — an empty viewport — and each has a different
/// answer, so the one thing not to do is say "no cage".
fn cage_absence_reason(app: &App) -> String {
    let graph = app.part_graph();
    let Some(node) = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Cage(c) if *c == app.active_cage))
        .map(|(id, _)| id)
    else {
        return "This part has no cage block — Create > Cage adds one".to_string();
    };
    if graph.roots().contains(&node) && !app.is_visible(node) {
        return "The cage block is hidden — switch its eye back on".to_string();
    }
    match app.body_containing(node) {
        Some(body) if !app.is_visible(body) => {
            "The block containing the cage is hidden — switch its eye back on".to_string()
        }
        _ => "Nothing draws the cage at the moment".to_string(),
    }
}

/// The cage's own extent, with a margin, for sizing its bake.
///
/// The margin matters rather than being tidiness: outside its bake the shader bounds the
/// distance using the distance to the box, and that bound is only useful while the surface
/// keeps clear of the faces.
fn cage_box(cage: &Cage) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    if !lo.x.is_finite() {
        return (Point3::origin(), Point3::origin());
    }
    let margin = (hi - lo).norm() * 0.08;
    let m = Vector3::new(margin, margin, margin);
    (lo - m, hi + m)
}

/// How far past a candidate hit `trace_field` looks, in hit thresholds. `CONFIRM` in
/// `march.wgsl`, and the two are the same number for the same reason.
const TRACE_CONFIRM: f64 = 4.0;

/// Sphere-traces a field, for picking.
///
/// Separate from the GPU marcher because this answers one ray on the CPU rather than a
/// million on a GPU, and because a click can afford to be careful where a frame cannot.
///
/// The step takes the full reported distance rather than a fraction of it. Every field in
/// fieldform undertakes never to overestimate, so a full step cannot pass through a surface,
/// and shortening it only costs iterations. What *did* cause misses was giving up too early:
/// a ray that runs out of budget looks exactly like a ray that hit nothing, and the caller
/// then silently reused the previous pivot.
///
/// A candidate hit is confirmed the same way `march.wgsl` confirms one, and for the same
/// reason: a small reading says the bound is small, not that there is a surface, and where two
/// solids share a face `min` reads nought through solid material. Picking has to agree with
/// what is drawn — the rule this file already holds to for cage against field — so if the
/// viewport goes through that sheet, a click must go through it too, or the pivot lands on a
/// surface nobody can see. `fieldform-field/examples/opposed_extrudes.rs` measures it.
fn trace_field(
    field: &GraphField,
    origin: Point3<f64>,
    dir: Vector3<f64>,
    far: f64,
    epsilon: f64,
    budget: u32,
) -> Option<Point3<f64>> {
    let mut t = 0.0;
    for _ in 0..budget {
        if t > far {
            return None;
        }
        let p = origin + dir * t;
        let d = field.distance(p);
        if d < epsilon {
            let beyond = field.distance(origin + dir * (t + TRACE_CONFIRM * epsilon));
            // Negative past it is a boundary; still falling is a loose bound closing on one.
            // Only a value that dips and comes back up is refused, and no surface does that.
            if beyond < 0.0 || beyond < d {
                return Some(p);
            }
            t += TRACE_CONFIRM * epsilon;
            continue;
        }
        t += d.max(epsilon);
    }
    None
}
