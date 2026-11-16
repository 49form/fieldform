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
mod image_export;
mod image_block;
mod comb_block;
mod memory;
mod menu_hover;
mod preview;
mod split_axis;
mod sketch;
mod surface_point;
mod hole_tool;
mod console_keys;
mod commands;
#[cfg_attr(not(test), allow(dead_code))] // carried to by the transport, which comes next
mod agent;
mod key_types;
mod key_defaults;
mod examples;
#[cfg_attr(not(test), allow(dead_code))] // read by the agent host, which a later phase adds
mod catalogue;
#[cfg_attr(not(test), allow(dead_code))]
mod outcome;
mod block_edit;
mod echo;
mod palette;
mod keybindings;
mod marking;
mod settings;
mod expect;
mod names;
mod infill_presets;
mod lattice_readout;
mod camera;
mod view_cube;
mod overlay_draw;
mod cage_select;
mod script;
mod console;
mod file_picker;
mod selection;
mod grid;
mod curve_edits;
mod operands;
mod pick;
mod display_quality;
mod app;
mod graph_view;
mod theme;
mod app_state;
mod console_log;
mod stack_guard;
mod recovery;
mod import_prompt;
mod cage_helpers;
mod number_fields;
mod block_words;
mod gestures;
mod draft_analysis;
mod graph_edits;
mod primitive_kind;
mod row_widgets;
mod references;
mod curve_blocks;
mod file_names;
mod section;
mod drawing_plane;
mod placement;
mod view_keys;
mod verification;
mod pick_trace;

use app_state::*;
use console_log::*;
use stack_guard::*;
use recovery::*;
use import_prompt::*;
use cage_helpers::*;
use number_fields::*;
use block_words::*;
use gestures::*;
use draft_analysis::*;
use graph_edits::*;
use primitive_kind::*;
use row_widgets::*;
use references::*;
use curve_blocks::*;
use file_names::*;
use section::*;
use drawing_plane::*;
use placement::*;
use view_keys::*;
use verification::*;
use pick_trace::*;

use split_axis::AxisKind;
use outcome::LineError;

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
    draw_axis_seed, draw_brep_edges, draw_cage_edges, draw_curves, draw_dashed_polyline, draw_highlight,
    draw_poi_marker, draw_surface_wire, CONTROL_POLYGON_COLOR, CURVE_COLOR, CURVE_SELECTED_COLOR,
    GUIDE_COLOR, GUIDE_FILL, HOVER_COLOR, HOVER_FILL, PLANE_PREVIEW_FILL, SELECTION_COLOR,
    SELECTION_FILL,
};
use script::{ExpectRecord, SaveAs, Script, ScriptTally, SCRIPT_EXIT_CODE, SNAP_WAIT_MAX};
use image_export::PendingSnap;
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
    delete_held_curve_commands,
    bezier_between, copied_body, curve_shapes_a_body_in, delete_loop_member_commands, hold_join_commands,
    release_join_commands,
    delete_row_commands, drag_targets, group_by_curve, join_loop_commands, move_point_node,
    moved_to_start_at, newly_closed_rows, paste_body, point_literals, points_in_box,
    readers_in_other_parts, realign_commands, rederived_commands, rows_not_in, set_targets,
    settle_curves_on_points, single_span, single_span_bezier, split_selection, toggle_in,
    ungroup_commands, CURVE_DEGREES,
};
#[cfg(test)]
use curve_edits::delete_curves_commands;
#[cfg(test)]
use operands::pair_candidates_in;
use operands::{
    operand_choices_in, operand_refusal, outermost_in, starting_pair_among,
    twist_would_take,
};
#[cfg(test)]
use operands::starting_pair_in;
use blocks::list::{
    branch_starts, curve_v_elsewhere, drop_gutter, ends_on_point, DraggedBlock,
    filtered_choices, folded_curve_row_in, grip_mark, join_of, joins_in, joint_choices,
    joint_points_in, joint_relation_between, make_variable, Join, nest_rows, points_hidden_with_rows,
    names_in_a_row, profile_v_curve_in, put_back_inside, put_number_back, rename_block,
    variable_item, Block, BlockRow, Clipboard, ProfileRows, Reveal, Row, RowKey,
    REVEAL_FRAMES_PER_ROW, REVEAL_SETTLE_FRAMES,
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

#[cfg(test)]
mod tests;
#[cfg(test)]
mod app_tests;
#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod written_tests;
#[cfg(test)]
mod family_tests;

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
    // `on_exit` has joined them already; this catches a close that never reached it, since
    // `process::exit` below would tear the driver down under a worker still inside it.
    traced::join_shader_workers();
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
    // A frame the display would not take within its second is, by default, dropped and nothing
    // more: the window then shows the frame before it until some later event asks for another,
    // and while the cause lasts every attempt costs the window's thread that second again. On a
    // laptop whose discrete card powers down when idle that is how a save looked like a hang —
    // the frames drawn as the card woke were never handed back. Reconfiguring gives the surface
    // fresh images to draw into and asks for the frame again, so the window catches up as soon
    // as the card is awake.
    // **On a Wayland desktop frames are not paced to the display.** A paced surface waits for the
    // compositor to say it is ready for a frame, and one that does not say so holds every frame
    // for its full second: the window stalled for a minute or more at one refused frame a
    // second, on its first frames after sitting untouched or after a change of workspace, and
    // reconfiguring the surface did not end it. Measured rather than argued: three of four such
    // returns stalled paced, a 66 minute run unpaced had none, and the next paced run stalled
    // again. Unpaced, a frame does not wait to be asked for. The interface draws only when
    // something changes, so this is not a loop spinning; what it gives up is the pacing of a
    // drag. `FIELDFORM_PRESENT=vsync` asks for pacing back and `=novsync` asks for this anywhere.
    let asked = std::env::var("FIELDFORM_PRESENT").ok();
    let unpaced = match asked.as_deref() {
        Some("novsync") => true,
        Some("vsync") => false,
        _ => cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_some(),
    };
    if unpaced {
        config.present_mode = wgpu::PresentMode::AutoNoVsync;
        log::info!("frames are not paced to the display (FIELDFORM_PRESENT=vsync paces them)");
    }
    let otherwise = config.on_surface_status.clone();
    config.on_surface_status = std::sync::Arc::new(move |status| match status {
        wgpu::CurrentSurfaceTexture::Timeout => {
            log::warn!("the display took no frame for a second; reconfiguring the surface");
            eframe::egui_wgpu::SurfaceErrorAction::RecreateSurface
        }
        other => otherwise(other),
    });
    config
}

/// How big anything added from the Create menu starts.
///
/// A sphere gets this as its diameter and a cube as its side, so a new body is always about the
/// same size as any other new body, whatever else is on screen.
const DEFAULT_BODY_MM: f64 = 100.0;

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

/// The tolerance a field is compiled at to be measured or sampled rather than drawn.
///
/// **Fine enough that no section answers from its display grid.** A bake reads a quarter of the
/// tolerance outside its outline on every wall, and whether a section bakes depends on its size:
/// at 0.05 mm every extrude under about a hundred millimetres across was, so `sample` put a
/// 17 mm pad 0.0125 mm proud while a 100 mm plate beside it read exact. The picture finishes on
/// the outline itself, so it never showed. At a thousandth only a section under about two
/// millimetres bakes, and then a quarter of a micron proud. The number `expect` already
/// measured at, so a script's check and the console's reading are of one field.
const MEASURE_TOLERANCE_MM: f64 = 1e-3;

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
                    | Node::Scale { .. }
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

/// The console entry field's own id, so the Tab hook can tell it from every other text field.
const CONSOLE_INPUT_ID: egui::Id = egui::Id::NULL;

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

/// What a new document's part is made of: chrome (the owner, 2026-11-07: "lets make horizon with
/// chrome the default"), which with the Horizon environment shows a surface's curvature as one
/// clean highlight line from the first thing drawn.
///
/// **Written into the new part, not into what no material means.** `Part::material` being `None`
/// is the neutral preset, and every document saved without a material reads that way; making
/// `None` mean chrome would turn every one of them to chrome on opening. So the choice is stored
/// on the part a new document starts with, and an older file looks as it did.
fn starting_look(doc: &mut Document, part: PartId) {
    if let Some(p) = doc.parts.get_mut(&part) {
        p.material = fieldform_scene::Material::preset("Chrome");
    }
}

/// The command a block's eye issues to draw it or stop: `App::set_visible`'s, a function of the
/// graph so a test can run what the eye runs.
fn eye_command(graph: &Graph, id: NodeId, on: bool) -> SetVisible {
    SetVisible::new(id, eye_flag_in(graph, id), on)
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
    /// The console: what is being typed, what came back, the file it is mirrored to, and the echo.
    console: ConsoleState,
    /// A command on its way to being kept: the pending session, the palette, the marking menu and the shelf.
    tools: ToolState,
    /// The keymap, the person's own bindings over it, and a rebind in progress.
    keys: KeyState,
    /// The block list's own state: a name or a reference being typed, the row it is on its way to, and what one frame's drawing hands from a block to its rows.
    notebook: NotebookState,
    /// Where the viewport was drawn last frame, so the tool panel can keep clear of the block it is
    /// making (`interaction-spec.md` §4.1).
    viewport_rect: Option<egui::Rect>,
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
    /// Where the document lives on disk, whether it is ahead of that file, and what is being asked about it.
    saving: SaveState,
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
    /// Readings of the document kept until it next moves, so that no view walks the graph every frame.
    kept: Kept,
    /// Every row frame the list has drawn since the record was last cleared, in the order drawn,
    /// for the tests that hold the frames to their nesting: a window is not needed to ask whether
    /// one encloses another.
    #[cfg(test)]
    drawn_frames: Vec<blocks::list::DrawnFrame>,
    /// The frames being drawn, innermost last, so a frame knows which one it is drawn inside.
    #[cfg(test)]
    open_frames: Vec<usize>,
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
    /// Check Model: its window, its filter, and the checks in flight.
    check: CheckState,
    /// The cage tool armed, if one is, and the extrude or inset session pending, if one is
    /// (`app/cage_tools.rs`).
    cage_tool: Option<app::cage_tools::CageTool>,
    cage_session: Option<app::cage_tools::CageSession>,
    /// Each infill block's cavity report, worked out off the interface thread.
    infill_cavities: std::collections::HashMap<NodeId, blocks::infill::CavityRun>,
    /// Where new curves are built. See [`DrawingPlane`].
    drawing_plane: DrawingPlane,
    started: std::time::Instant,
    stack: CommandStack,
    registry: OpRegistry,
    /// The control cage, triangulated — what polygon mode draws and what picking hits.
    cage_mesh: CageMesh,
    /// Imported geometry, as triangles ready to draw. See `rebuild_imported_mesh`.
    imported_mesh: CageMesh,
    wgpu_render_state: egui_wgpu::RenderState,
    /// How the part is looked at, which is no part of the document: the mode, the quality and what is drawn over the picture.
    view: ViewSettings,
    /// What a frame cost, and where that is being written.
    perf: PerfState,
    selection: Picked,
    hover: Selection,
    move_state: Option<MoveState>,
    /// Curves as the viewport and the panel work on them: a drag or a draw in progress, what is drawn, and the profile rows.
    curve_ui: CurveState,
    /// What a hole waiting for its click was asked to be: the line's keys, kept until the
    /// surface it goes into is picked (`hole_tool.rs`).
    pending_hole: Option<Args>,
    /// Points on a surface: what each one's last solve said, and one being edited or slid.
    surface_points: SurfacePointState,
    /// What the press that is down landed on. Decided at the press, because by the time it is a drag the pointer has moved off whatever it started on.
    presses: Presses,
    /// The projection the view had before Sketch on plane made it orthographic, given back when
    /// the sketch stops. `None` outside a sketch.
    sketch_projection_before: Option<Projection>,
    /// The manipulator drag in progress, if any.
    gizmo_drag: Option<GizmoDrag>,
    /// The handle under the cursor, for the hover highlight.
    gizmo_hover: Option<gizmo::Handle>,
    clipboard: Option<Clipboard>,
    /// The agent endpoint: whether one is published, who is listening, and the session attached to it.
    agent_link: AgentLink,
    /// A script being run, and what its `expect` lines have found.
    scripting: ScriptState,
    /// A snapshot waiting for the view to finish its still. See [`PendingSnap`].
    pending_snap: Option<PendingSnap>,
    /// Everything about running the Save Image blocks, none of it saved (`image_block`).
    images: image_block::ImageRuns,
    /// Everything about drawing and editing the curvature combs, none of it saved (`comb_block`).
    comb_runs: comb_block::CombRuns,
    /// The Expression blocks' running state, none of it saved.
    written: WrittenState,
    /// Where the cursor is over the viewport this frame, with the aspect and rect a pick
    /// needs, so Ctrl+V can land a curve under it (`profile-group-spec.md` §4.4). `None`
    /// whenever a menu, the console or a panel is under the pointer instead.
    viewport_hover: Option<(f32, egui::Rect, egui::Pos2)>,
    /// Where the viewport's context menu was opened, which is where "Paste here" pastes.
    viewport_menu_at: Option<egui::Pos2>,
    selected_curve: Option<CurveId>,
    /// The notebook's second view (`notebook-graph-view-spec.md`): camera, expansion and pins only.
    graph_view: graph_view::GraphView,
    /// Which of [`Self::selected_curve`]'s control points are picked, by index.
    ///
    /// Indices rather than ids, and that is the point rather than a shortcut. A control point
    /// has no stable identity and is not given one: naming N of them per curve is per-element
    /// identity by another name, which is what D1 declined to take on for cage vertices — the
    /// same argument one level up. Undo reverses against exact prior positions instead, so
    /// nothing needs to survive being renumbered.
    selected_points: Vec<usize>,
    /// What the next block of each kind starts with: the last value chosen for one, remembered instead of asked for.
    next: NextBlock,
    /// Which body the inspector is bound to, and which two the boolean tool would combine.
    ///
    /// Bodies are graph nodes, and a node id outlives the edits around it — unlike a cage
    /// vertex index, which is why selection here can be persistent state and selection down in
    /// the cage cannot.
    selected_body: Option<NodeId>,
    /// A selected block that is not a node — a tessellation, an export or an import — so the
    /// Delete key and a row's menu act on it the way they act on a body.
    selected_block: Option<outputs::BlockRef>,
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
    /// An import: what the reader is told, the file waiting for its options, and the one running.
    importing: ImportState,
    /// A section's edits in progress, so typing into its rows lands on the undo stack once: the
    /// section, when the gesture began, and the list as it was then. `set_images`' arrangement.
    section_gesture: Option<(fieldform_scene::SectionCutId, std::time::Instant, Vec<fieldform_scene::SectionCut>)>,
    /// Which section the manipulator is currently aimed at, rather than at geometry.
    ///
    /// Explicit rather than inferred from "a section is on and nothing else is selected":
    /// two things claiming one gizmo is exactly the shape of bug this codebase keeps meeting,
    /// and the fix each time was to decide where the claim is *made*. Turning a section on,
    /// or clicking its row, makes the claim; selecting any geometry drops it.
    section_selected: Option<fieldform_scene::SectionCutId>,
    /// The graph edit in progress, so a drag lands on the undo stack once.
    ///
    /// Holds what the part looked like when the gesture *started*, which is what the entry has
    /// to reach back to however many frames the drag runs for.
    gesture: Option<Gesture>,
    /// What `verify` last found, and what the part was when it found it. See [`Verification`].
    verification: Option<Verification>,
    /// A material drag in progress, so a slider goes on the history once. See `set_part_material`.
    material_gesture: Option<MaterialGesture>,
    /// Imports whose original CAD is drawn over the field derived from it.
    ///
    /// App state rather than document state, like body visibility and for the same reason: it
    /// is what you are *looking at*, not what the part is. It inherits that item's unsolved
    /// "visibility is not saved" problem rather than inventing a second answer to it.
    showing_original: std::collections::HashSet<fieldform_scene::id::SourceId>,
    /// Compiled cage and mesh leaves, reused across compiles. See `fieldform_field::LeafCache`.
    leaves: fieldform_field::LeafCache,
    /// Why the display graph would not compile, recorded when it was last compiled.
    ///
    /// Cached rather than asked for, because asking means compiling — see `compile_error`.
    compile_error: Option<String>,
    /// The block whose draft the last compile refused, so its panel can say why beside the
    /// number that was typed rather than only on the line above every block.
    draft_refused: Option<NodeId>,
    /// Parameters a panel row made or stopped referring to this frame, waiting for the graph edit
    /// they go on the history with — see [`ParameterEdits`] and [`App::parametric_field`].
    parameter_edits: ParameterEdits,
    /// The Tessellate and Export blocks' running state — jobs in flight and what each last
    /// produced. The blocks themselves are in the document; see `outputs.rs`.
    outputs: outputs::Outputs,
    /// Raised by the device-lost callback, from whichever thread found the loss, and answered on
    /// this one. See [`App::keep_the_document_if_the_gpu_was_lost`].
    gpu_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether the loss has been answered, so the document is kept once rather than on every
    /// frame the process limps on for; the autosave goes on covering anything after it.
    kept_after_loss: bool,
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

/// An `App` that goes without `on_exit` still waits for its shader compiles.
///
/// The window's close reaches `on_exit`; an `App` built without a window, as every test that
/// drives the interface builds one, is simply dropped, and the process then ended with a worker
/// still inside the driver. That was a segfault in `create_pipeline_layout` after every test had
/// passed, which a test run reports as a failure of the whole binary. A join that finds nothing
/// running costs nothing, so the window's own close, which has joined already, pays nothing here.
impl Drop for App {
    fn drop(&mut self) {
        traced::join_shader_workers();
    }
}

impl eframe::App for App {
    /// Waits for any shader still compiling before the window and the GPU device go.
    ///
    /// A compile runs on a worker of its own and holds the device; closing under it crashed in
    /// the driver, which turned every `--exit` that closed mid-compile into a segfault.
    fn on_exit(&mut self) {
        traced::join_shader_workers();
    }

    /// Answers the agent while the window is minimised.
    ///
    /// **A minimised window gets no `ui` pass**, and `ui` is where [`App::drive_agent`] runs, so a
    /// host whose window was minimised answered nothing: a call sat until the client gave up, with
    /// the process alive and idle. eframe still runs this hook when the window is hidden and a
    /// repaint is asked for, and the agent's listener asks for one with every call. While the
    /// window is shown it does nothing, so the order the frame keeps for everything else stands.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.agent_link.window_hidden = ctx.input(|i| i.viewport().minimized == Some(true));
        if self.agent_link.window_hidden {
            self.drive_agent();
        }
    }

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
            .filter(|_| self.tools.session.is_some() && !self.tools.floating_panel)
            .and_then(|f| self.tools.pending_numbers.iter().position(|n| *n == f));
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
                let n = self.tools.pending_numbers.len();
                let next = self.tools.pending_numbers[if back { (at + n - 1) % n } else { (at + 1) % n }];
                ctx.data_mut(|d| d.remove_temp::<String>(self.tools.pending_numbers[at]));
                ctx.memory_mut(|m| m.request_focus(next));
                app::inspector::select_whole_number(ctx, next);
            }
            return;
        }
        let in_console = focused == Some(CONSOLE_INPUT_ID.with("console"));
        // A pending session's panel field is the other exception: Tab there confirms the row and
        // moves to the next (`interaction-spec.md` §4.3), which egui's own traversal would not.
        let in_panel = self
            .tools.session
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
                self.console.complete = true;
            } else {
                self.keys.stolen_tab = stolen;
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
        self.show_discard_prompt(ui.ctx());
        self.show_file_picker_wait(ui.ctx());
        // An experiment, off unless asked for: `FIELDFORM_KEEP_AWAKE=1` draws a frame every two
        // seconds while nothing else does, so a discrete card that powers down when unused never
        // gets the chance. It is here to find out whether the run of frames the display refuses
        // after an idle spell is the card waking, and is not a decision to keep it awake.
        if std::env::var_os("FIELDFORM_KEEP_AWAKE").is_some() {
            ui.ctx().request_repaint_after(std::time::Duration::from_secs(2));
        }
        self.autosave();
        self.drive_snap(ui.ctx());
        self.drive_captures(ui.ctx());
        self.drive_script(ui.ctx());
        self.drive_agent();
        self.help_windows(ui.ctx());
        self.keys_panel(ui.ctx());
        self.check_model_panel(ui.ctx());
        self.poll_surface_checks();
        self.import_options_window(ui.ctx());
        self.handle_keys(ui);
        // After the keys, so a row typing began into this frame shows what was typed, and the
        // palette Space opened this frame has the keyboard at once.
        self.tool_panel(ui.ctx());
        self.cage_panel(ui.ctx());
        self.cage_tool_panel(ui.ctx());
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
                    .button(self.view.mode.label())
                    .on_hover_text("Click or press Tab to cycle")
                    .clicked()
                {
                    self.dispatch("view.cycle_mode", &Args::new());
                }
                // **Visible and stoppable** (`mcp-server-spec.md` §9.1): while a client is
                // connected the person sees it, and one click puts it out.
                if let Some(host) = &self.agent_link.host
                    && host.clients() > 0
                    && ui
                        .button(egui::RichText::new("agent connected").color(egui::Color32::from_rgb(230, 160, 40)))
                        .on_hover_text("An agent is connected and can edit this document. Click to disconnect it.")
                        .clicked()
                {
                    host.disconnect_all();
                    self.say("Agent disconnected.");
                }
            });

        });

        self.inspector(ui);

        egui::CentralPanel::default().show_inside(ui, |ui| self.viewport(ui));

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
        if std::mem::take(&mut self.tools.keep_pending) {
            self.commit_session();
        }
        self.echo_when_quiet(ui.ctx());

        // Last, so the number is the whole frame's.
        self.note_frame_cost(frame_started.elapsed().as_secs_f32() * 1e3);
    }
}
