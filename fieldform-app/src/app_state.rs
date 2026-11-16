// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The application's state, grouped by what it serves. Each of these is one field of `App`.

use crate::*;

/// A command on its way to being kept: the pending session, the palette, the marking menu and the shelf.
pub(crate) struct ToolState {
    /// The tool session pending, if one is (`app::sessions`), and what each command was last kept
    /// with this run of the application, which the next session of it starts from (§3.2).
    pub(crate) session: Option<app::sessions::Session>,
    pub(crate) last_values: std::collections::HashMap<&'static str, Args>,
    /// The command palette while it is open, and the commands run lately, most recent first,
    /// which it ranks by (`interaction-spec.md` §5).
    pub(crate) palette: Option<app::palette_ui::PaletteState>,
    pub(crate) recent: Vec<&'static str>,
    /// The marking menu while it is open (§7), and the last command kept or run, which Shift+R
    /// runs again (§8).
    pub(crate) marking: Option<app::marking_ui::MarkingMenu>,
    /// A join's continuity shown before it is chosen, while a lead is hovered in the join's list.
    /// See [`app::marking_ui::JoinPreview`].
    pub(crate) join_preview: Option<app::marking_ui::JoinPreview>,
    pub(crate) last_command: Option<&'static commands::Command>,
    /// The shelf's commands (§10.2), where each was drawn last frame, and how many times the
    /// palette has been opened, which the first-run hint goes by (§10.3).
    pub(crate) shelf: Vec<&'static str>,
    pub(crate) shelf_drawn: Vec<(&'static str, egui::Rect)>,
    /// Whether a pending session shows the floating tool panel. **Off by default, at the owner's
    /// asking (2026-10-17):** the panel and the block's own rows in the notebook said the same
    /// numbers twice, so a pending block now opens in the notebook with its first number waiting
    /// for the keyboard, and the panel is View › Floating tool panel for whoever wants it back.
    pub(crate) floating_panel: bool,
    /// Set while a pending block's rows are being drawn for the first time: the first number field
    /// drawn takes the keyboard, and clears it (`number_field`).
    pub(crate) focus_first_number: bool,
    /// Set while the pending block's rows are being drawn, so a number confirmed there with Enter
    /// can say the block is done (`tools.keep_pending`).
    pub(crate) drawing_pending: bool,
    /// Enter confirmed a number of the pending block: it is kept at the end of the frame, once the
    /// number has been written.
    pub(crate) keep_pending: bool,
    /// The pending block's number fields as last drawn, in order, for Tab to move between.
    pub(crate) pending_numbers: Vec<egui::Id>,
    pub(crate) palette_opens: u64,
}

/// The keymap, the person's own bindings over it, and a rebind in progress.
pub(crate) struct KeyState {
    /// The person's own bindings over the registry's defaults (`keybindings`), what collides, the
    /// command waiting for keys to be pressed, and what the last rebind said (§9.2).
    pub(crate) bindings: Vec<keybindings::Line>,
    pub(crate) conflicts: Vec<keybindings::Conflict>,
    pub(crate) capturing: Option<&'static str>,
    pub(crate) note: Option<String>,
    pub(crate) keymap: Keymap,
    /// A Tab taken out of the raw input before egui could route it to focus, waiting to be
    /// dispatched through the keymap like any other shortcut. See [`App::raw_input_hook`].
    pub(crate) stolen_tab: Option<Modifiers>,
}

/// Where the document lives on disk, whether it is ahead of that file, and what is being asked about it.
pub(crate) struct SaveState {
    /// Where this document was opened from or last saved to, and the archive it came out of.
    ///
    /// The archive is kept, not just the path, because it carries the entries this build does
    /// not interpret — a plugin's `ext/` data, the `sources/` an import came from. Saving
    /// through it preserves them; saving a freshly built one would drop exactly what the
    /// container exists to protect.
    pub(crate) file: Option<(std::path::PathBuf, fieldform_io::Archive)>,
    /// How many times the document has changed, and what that count was when it was last
    /// written or opened. Unequal means there is something in memory the file does not have.
    ///
    /// A counter rather than a comparison against the saved bytes, because the marker is
    /// wanted every frame and serializing a hundred-thousand-vertex cage per frame is not a
    /// price a title bar can pay. It is conservative in one direction only: undoing back to
    /// the state that was saved still counts as changed, so the app can offer to save
    /// something identical to what is on disk. It can never call a changed document clean,
    /// which is the direction that loses work.
    pub(crate) revision: u64,
    pub(crate) saved_revision: u64,
    /// Whether the unsaved-changes prompt has already been answered in favour of closing.
    ///
    /// Discarding does not make the document clean, so without this the re-issued close asks
    /// again, and again. See `guard_close`.
    pub(crate) closing: bool,
    /// The unsaved-changes prompt, while one waits for an answer. See `may_discard`.
    pub(crate) discard_prompt: Option<app::document_io::DiscardPrompt>,
    /// Set by the prompt's answer for the one call it then makes, so that call's own
    /// `may_discard` lets it through instead of asking again.
    pub(crate) discard_answered: bool,
    /// A file picker that is open on a thread of its own, and what its answer is for.
    pub(crate) file_picker: Option<file_picker::Waiting>,
    /// The document revision last written to the autosave, so an unchanged document is not
    /// written again.
    pub(crate) autosaved_revision: u64,
    /// When the autosave was last written, or the application started. See [`App::autosave`].
    pub(crate) autosaved_at: std::time::Instant,
}

/// Readings of the document kept until it next moves, so that no view walks the graph every frame.
pub(crate) struct Kept {
    /// The overlay's readings as of [`Self::document_moves`] and the view they were read for. In a
    /// cell because the overlay, picking and the trace all read them through `&self`, and
    /// whichever reads first after something moves is the one that keeps them.
    pub(crate) readings: std::cell::RefCell<fieldform_field::KeptReadings<OverlayKey>>,
    /// The block list's readings as of [`Self::document_moves`] and the part they were read off,
    /// in a cell for the same reason.
    pub(crate) list: std::cell::RefCell<
        fieldform_field::KeptReadings<ListKey, fieldform_scene::ListReadings>,
    >,
    /// What the notebook says is wrong with each block of the part, kept against the same count
    /// as the list, since both views draw it and asking per row per frame is a walk of the graph.
    pub(crate) statuses: std::cell::RefCell<
        fieldform_field::KeptReadings<ListKey, graph_view::BlockStatuses>,
    >,
}

/// Check Model: its window, its filter, and the checks in flight.
pub(crate) struct CheckState {
    /// Check Model: the window, its filter, the self-intersection check in flight, when the amber
    /// marks stop showing on their own, and when the next check may start (`app/check_model.rs`).
    pub(crate) window_open: bool,
    pub(crate) only_errors: bool,
    pub(crate) surface_check: Option<app::check_model::SurfaceCheck>,
    pub(crate) amber_until: Option<std::time::Instant>,
    pub(crate) due: std::time::Instant,
}

/// The console: what is being typed, what came back, the file it is mirrored to, and the echo.
pub(crate) struct ConsoleState {
    /// Where the echo last wrote up to (`app::echoing`), and every line it has written, in order:
    /// the record T1 replays (`interaction-spec.md` §6.3).
    pub(crate) echo_base: app::echoing::EchoBase,
    pub(crate) echo_log: Vec<String>,
    /// The console: what is being typed, and what came back.
    ///
    /// Every line goes through the same named dispatch a menu item does, so a command and a
    /// click cannot mean different things. That is the whole reason the app has had named
    /// operations from the start.
    pub(crate) input: String,
    pub(crate) log: Vec<ConsoleLine>,
    /// How many things have been reported, ever. Not `console.log.len()`, which stops growing
    /// when a line repeats and is capped besides — see [`App::say`].
    pub(crate) reports: u64,
    /// The viewport error already reported, so a standing one is not reported again next frame.
    pub(crate) traced_error_said: Option<String>,
    /// Where the console is being mirrored, and how much of it is there.
    pub(crate) file: Option<std::path::PathBuf>,
    pub(crate) file_bytes: u64,
    /// What has been typed before, newest last, walked with the up and down arrows.
    pub(crate) history: Vec<String>,
    pub(crate) history_at: Option<usize>,
    /// What a long-running job is doing *right now*, shown as the console's last line and
    /// replaced in place rather than appended.
    ///
    /// Separate from the log because it is not an event. An export ticking its elapsed seconds
    /// would otherwise write a line a second, and the history the console exists to keep would
    /// be a column of near-identical stopwatch readings. What it says when it *finishes* is an
    /// event, and goes through `say` like everything else.
    pub(crate) progress: Option<String>,
    /// A Tab pressed while the console field had the keyboard, waiting to complete a word.
    ///
    /// Taken in `raw_input_hook` for the same reason the mode cycle's is, and kept apart from
    /// it so that completing a command neither cycles the viewport nor walks the focus away
    /// from the field being typed in.
    pub(crate) complete: bool,
    /// How many of those reports were warnings, so a script can tell whether the line it just
    /// ran was refused. See [`Script`].
    pub(crate) warnings: usize,
    /// While a line runs through [`App::run_line`], everything it reported, in order.
    pub(crate) line_capture: Option<Vec<(bool, String)>>,
    /// The structured form of the refusal the line in flight met, set where it is refused.
    pub(crate) line_error: Option<LineError>,
    /// The console line a cage verb just run from a menu, a key or the palette amounts to, with the
    /// elements it read from the selection spelled out -- what the echo writes for it. A topology
    /// edit is not something the document's diff can spell, and the verb's bare word would replay
    /// as a refusal for want of the edges or faces it was given.
    pub(crate) cage_line: Option<String>,
}

/// A script being run, and what its `expect` lines have found.
pub(crate) struct ScriptState {
    /// A script waiting to run, from `--run` on the command line or the `run` verb.
    pub(crate) script: Option<Script>,
    /// Whether a script's lines are being run this frame, so `run` inside one is refused.
    pub(crate) running: bool,
    /// Every `expect` this session, in order, for a script's report.
    pub(crate) expect_results: Vec<ExpectRecord>,
    /// What a run has tallied so far, kept across the frames a snapshot waits for. See
    /// [`App::drive_script`].
    pub(crate) tally: Option<ScriptTally>,
}

/// The agent endpoint: whether one is published, who is listening, and the session attached to it.
pub(crate) struct AgentLink {
    pub(crate) state: agent::AgentState,
    /// The MCP listener, when `--agent-host` started one (`mcp-server-spec.md` §3.2).
    pub(crate) host: Option<agent::server::Host>,
    /// Whether the window is minimised, as the last pass saw it. A hidden window gets no `ui`
    /// pass, only [`eframe::App::logic`], so the agent is answered from there while it is set.
    pub(crate) window_hidden: bool,
    /// Allow agent (`mcp-server-spec.md` §9.1): whether this session publishes an endpoint a
    /// client may attach to. Off by default and kept per person.
    pub(crate) allowed: bool,
    /// An endpoint other than the account's own, so a test never meets a running application's.
    pub(crate) endpoint: Option<String>,
}

/// What the press that is down landed on. Decided at the press, because by the time it is a drag the pointer has moved off whatever it started on.
pub(crate) struct Presses {
    /// Whether the press that is down landed on a surface point's dot, which it then slides.
    pub(crate) surface_point: bool,
    /// Whether the press that is currently down landed on a manipulator handle. Decided at the
    /// press for the same reason `presses.selection` is: by the time egui calls it a drag,
    /// the pointer has left the handle and where the gesture *started* can no longer be
    /// asked.
    pub(crate) gizmo: bool,
    /// The same, for the knife's active point: a press that lands on it, while the knife is
    /// armed, slides the point along its edge instead of tumbling the camera.
    pub(crate) cage_tool: bool,
    /// The same, for a control point: it claims the press so the click does not also land on
    /// the cage or the body drawn behind it.
    pub(crate) control_point: bool,
    /// The running press started a box, which the click handlers stand aside for.
    pub(crate) marquee: bool,
    /// Whether the press that is in flight landed on a curve.
    ///
    /// Needed because the two halves of a click are handled at different moments: a curve is
    /// picked on the *press*, and the body pick runs on the *release* and clears the selected
    /// curve as part of meaning "you clicked the model, not a curve". Without this the curve was
    /// selected and then unselected within one click, which looks exactly like the pick not
    /// working.
    pub(crate) curve: bool,
    /// Whether the left button went down on the current selection, which is what makes the
    /// drag that may follow a move rather than a tumble.
    ///
    /// Answered at the press and remembered, because by the time egui calls it a drag the
    /// pointer has moved and where the gesture started can no longer be asked.
    pub(crate) selection: bool,
}

/// Points on a surface: what each one's last solve said, and one being edited or slid.
pub(crate) struct SurfacePointState {
    /// What each point on a surface's last solve said: why it is red, a warning, and which of a
    /// ray's crossings it took. Derived, and read by the panel and the viewport.
    pub(crate) states: std::collections::HashMap<NodeId, surface_point::PointState>,
    /// The record a drag of one of a point's numbers began from, until the hand comes off.
    pub(crate) edit_before: Option<(NodeId, fieldform_scene::SurfacePoint)>,
    /// A point on a surface being slid, by its dot or by a handle: its body, asked once for the
    /// gesture (`surface_point.rs`).
    pub(crate) slide: Option<surface_point::Slide>,
}

/// Curves as the viewport and the panel work on them: a drag or a draw in progress, what is drawn, and the profile rows.
pub(crate) struct CurveState {
    /// A control point being dragged directly. See [`PointDrag`].
    pub(crate) point_drag: Option<PointDrag>,
    /// A curve waiting for its clicks. See [`PendingCurve`].
    pub(crate) pending: Option<PendingCurve>,
    /// Why an axis on a surface cannot stand where it says, by curve, when its ray no longer meets
    /// its body; shown in its block. Derived on every rebuild, never saved (`split_axis`).
    pub(crate) surface_axis_notes: std::collections::HashMap<CurveId, String>,
    /// An axis's record as it was when a drag of its U, V or length began, so that the drag lands on
    /// the history as one entry when it ends.
    pub(crate) surface_axis_edit_before: Option<(CurveId, fieldform_scene::SurfaceAxis)>,
    /// Control points picked on curves other than `selected_curve`, so one selection spans
    /// curves: Shift+click adds across them, Ctrl+drag boxes them, and a drag from any picked
    /// point moves the lot as one entry. `selected_points` stays the selected curve's own,
    /// which is what every single-curve reader already means by it.
    pub(crate) other_points: Vec<(CurveId, usize)>,
    /// A press on a point already picked: the selection is kept so a drag moves all of it,
    /// and narrowed to that one point if the press turns out to be a click.
    pub(crate) pending_narrow: Option<(CurveId, usize)>,
    /// A box selection being dragged: where it started, and whether it adds (Ctrl+Shift).
    pub(crate) marquee: Option<(egui::Pos2, bool)>,
    /// Every visible curve's control points drawn and clickable, not only the selected one's.
    pub(crate) show_control_points: bool,
    /// Where a curve's points were before the numeric fields started being dragged, so the
    /// whole gesture lands on the history as one entry rather than one per frame.
    pub(crate) edit_before: Option<(CurveId, Vec<Point3<f64>>)>,
    /// The same, for the weights: which points, and what they pulled with before the drag.
    pub(crate) weight_edit_before: Option<(CurveId, Vec<usize>, Vec<f64>)>,
    /// Curves in the document, tessellated once for drawing. Rebuilt on import, not per
    /// frame — a 0.05 mm polyline of a long spline is thousands of points.
    pub(crate) display: Vec<(CurveId, Vec<Point3<f32>>)>,
    /// Moves every time [`Self::rebuild_curve_display`] runs, which is every time a curve
    /// changes; what the profile rows are cached against. Not the document's `revision`, which
    /// moves on every edit of anything and would re-walk the loops on each frame of a slider
    /// drag that touched no curve.
    pub(crate) revision: u64,
    /// The profile rows, as of [`Self::curve_ui.revision`]. See [`ProfileRows`].
    pub(crate) profiles: ProfileRows,
    /// What a loft's panel and the viewport read of it -- its bound, each section's tilt, the
    /// neck and the derived centerline -- as of one node at one [`Self::curve_ui.revision`], or
    /// `None` inside for a loft that cannot be built. Building a loft walks and resamples
    /// every section and matches their corners, which a panel drawn every frame need not pay
    /// per frame; the sweep's panel gets away with building its leaf each frame because a
    /// sweep resamples nothing.
    pub(crate) loft_readings: Option<(NodeId, Node, u64, Option<LoftReadings>)>,
    /// Loops that closed while a gesture was in flight, by key, waiting to be held with
    /// joints once the hand comes off. See [`Self::refresh_profiles_with`].
    pub(crate) pending_joins: Vec<CurveId>,
    /// Curves the viewport leaves out. Same idea as `hidden`, and drawn from the same eye.
    pub(crate) hidden: std::collections::HashSet<CurveId>,
}

/// The block list's own state: a name or a reference being typed, the row it is on its way to, and what one frame's drawing hands from a block to its rows.
pub(crate) struct NotebookState {
    /// The text box the last reference field drew, for the row around it to hang a right-click
    /// menu on: the field answers only with what was picked, and the menu is the row's.
    pub(crate) last_reference_field: Option<egui::Response>,
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
    pub(crate) renaming: Option<(RowKey, String)>,
    /// The block list on its way to a row chosen somewhere else. See [`Reveal`].
    pub(crate) reveal: Option<Reveal>,
    /// The selection the block list has already followed, so that it follows each one once and
    /// a row chosen in the list itself is not scrolled out from under the pointer.
    pub(crate) followed: Option<RowKey>,
    /// A curve clicked in the viewport to be edited, waiting for the list to open its row.
    ///
    /// **Its own row, not only the rows above it**, which is what the reveal opens for any other
    /// selection (the owner: "when you click on a curve to edit it, its block should expand").
    /// Set by the click rather than read off the selection, because a curve also becomes selected
    /// when a line or a paste makes one, and a script of twenty lines should not leave twenty
    /// rows open behind it.
    pub(crate) curve_picked: Option<CurveId>,
    /// The same for the graph view, whose node for a curve is not the curve's row.
    pub(crate) graph_followed: Option<NodeId>,
    /// The reference field being typed in, and what is in it so far.
    ///
    /// Presentation state for the reason `renaming` is: half a block's name is not a reference, so
    /// nothing is written to the graph until the text resolves. One slot rather than one per field,
    /// because only the focused field is being typed in — and a field that loses focus with text
    /// that resolves to nothing puts the settled name back, so there is nothing to keep.
    pub(crate) ref_edit: Option<(egui::Id, String)>,
    /// The rows a block holds, while that block's own controls are being drawn, so the field that
    /// reads each one can draw it **in place** (`notebook-spec.md` §7.1, §17).
    ///
    /// **Why the rows travel rather than the drawing being moved.** A nested block used to be
    /// drawn in a list under its holder's parameters, which said twice what it is made of: the
    /// field named the block and the list below held it again. Containment is one place — the slot
    /// *holds* its block — and the slot is drawn by whichever row owns it, several layers inside
    /// the per-kind controls. Handing the rows to the drawing rather than threading a parameter
    /// through every kind's controls is what makes every slot host its block at once.
    ///
    /// A row is **claimed** as it is drawn ([`App::host_nested_block`]), so what is left when the
    /// controls finish is what no field hosted — a profile's curves, a frame's source, a loft's
    /// sections — and those are drawn below as they always were, until their own fields host them.
    /// Nothing can be drawn twice and nothing can vanish.
    ///
    /// Saved and restored around the controls, because a hosted block draws its own controls and
    /// sets this for its own rows. One frame's drawing state, in the sense `notebook.ref_edit` is.
    pub(crate) hosted: Option<Hosted>,
    /// The other half of a move begun by a drop: which block has to let go, and of what
    /// (`notebook-spec.md` §7.3).
    ///
    /// **Why it is carried rather than done where it is decided.** A block dragged from one input
    /// to another is one gesture and two slot writes -- the destination filled, the source emptied
    /// -- and the two are owned by different code: the drop knows where the block came from
    /// (`DraggedBlock::from`), while the destination slot is written by whichever panel row the
    /// pointer was over, which then hands its whole node to [`App::replace_node`]. So the drop
    /// leaves the second half here and `replace_node` completes it in the same graph edit, which
    /// is what makes the move one undo step instead of two.
    ///
    /// Armed only when the drop was *taken*, so a refused drop moves nothing, and taken by the
    /// first write that actually reads the block -- `replace_node` checks that the node it is
    /// writing now names it, so an unrelated edit in the same frame cannot consume it.
    ///
    /// Presentation state in the sense `notebook.ref_edit` is: it lives for part of one frame and nothing
    /// in the document depends on it.
    pub(crate) completing_a_move: Option<(NodeId, NodeId)>,
    /// A parameter whose name field should take the keyboard on the next frame.
    ///
    /// **Set where a variable is *made* from a field, not where one is edited.** Naming a
    /// value from the V menu invents a name from the field's label -- an X becomes `x`, then
    /// `x_2` -- and the only place to change it was the variables panel, which means finding
    /// the row you just made among the ones you did not. Reported as "when i name the x y z
    /// of a point as a VAR - i need to be able to give it a name". The same arrangement
    /// `ExpressionDraft::focus` already uses for the expression field, for the same reason:
    /// the next thing typed is the thing that was missing.
    pub(crate) naming_parameter: Option<ParamId>,
    /// Names being typed, by parameter. See [`NameDraft`].
    pub(crate) name_drafts: std::collections::HashMap<ParamId, NameDraft>,
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
    pub(crate) size_locked: std::collections::HashSet<NodeId>,
    /// The boolean whose list is taking bodies from viewport clicks, if one is.
    ///
    /// The way other CAD software fills a list input: turn the list on, click the bodies, watch
    /// block update. While it is set a click in the viewport adds instead of selecting, so it
    /// is deliberately hard to leave on by accident — collapsing the block or clicking the
    /// control again clears it.
    pub(crate) collecting: Option<NodeId>,
    /// Expressions being typed, by parameter. See [`ExpressionDraft`].
    pub(crate) expression_drafts: std::collections::HashMap<ParamId, ExpressionDraft>,
}

/// The Expression blocks' running state, none of it saved.
pub(crate) struct WrittenState {
    /// What verify measured for each Expression block, the text being typed into one, where
    /// its last parse stopped, and a notice about a paste (`app::written_panel`). None saved.
    pub(crate) verdicts: std::collections::HashMap<NodeId, app::written_panel::WrittenVerdict>,
    pub(crate) drafts: std::collections::HashMap<NodeId, String>,
    pub(crate) errors: std::collections::HashMap<NodeId, (usize, String)>,
    pub(crate) notices: std::collections::HashMap<NodeId, String>,
}

/// What the next block of each kind starts with: the last value chosen for one, remembered instead of asked for.
pub(crate) struct NextBlock {
    /// Field generator settings, live even before anything is applied so the radius can be
    /// dialled in and then committed.
    pub(crate) pipe_radius: ParamRef,
    /// Which two bodies the boolean tool would combine.
    pub(crate) boolean_a: Option<NodeId>,
    pub(crate) boolean_b: Option<NodeId>,
    pub(crate) boolean_op: BooleanOp,
    /// Fillet settings for the boolean, live before anything is applied so they can be dialled
    /// in the way the radius can.
    pub(crate) blend_mode: BlendMode,
    pub(crate) blend_radius: ParamRef,
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
    pub(crate) export_detail: fieldform_scene::MeshDetail,
    /// Whether an export runs the simplifier. Off by default - see
    /// `fieldform_scene::Tessellation::reduce`.
    pub(crate) export_simplify: bool,
    /// Whether an export improves triangle shape. On by default, for the reason given on
    /// `fieldform_scene::Tessellation::improve`.
    pub(crate) export_relax: bool,
    /// How many copies Modify > Pattern makes, counting the original.
    pub(crate) pattern_count: u32,
    /// Whether the pattern goes round the Z axis or along a line.
    pub(crate) pattern_radial: bool,
    /// How far apart a linear pattern's copies sit, along X.
    pub(crate) pattern_step_mm: f64,
    /// Which axis Modify > Twist winds about. Remembered between uses, like the mirror's.
    pub(crate) twist_axis: Axis,
    /// Which plane Modify > Mirror reflects across. Remembered between uses, like the offset
    /// distance: a part that wants mirroring in X usually wants it again.
    pub(crate) mirror_axis: Axis,
    /// How far Modify > Offset moves a surface, and how thick Modify > Shell leaves a wall.
    /// Kept on the app rather than asked for in a dialog, the same as the blend radius: the
    /// block it produces carries the number and changing it there is one drag.
    pub(crate) offset_distance: f64,
    pub(crate) shell_thickness: f64,
    /// How thick the next wall starts out — a curve given a thickness, on an extrude, a revolve
    /// or a sweep. The last thickness typed for one, for the reason the two above are kept:
    /// turning Wall on, or `extrude --wall` with no number, repeats it rather than asking.
    pub(crate) wall_thickness: f64,
}

/// An import: what the reader is told, the file waiting for its options, and the one running.
pub(crate) struct ImportState {
    /// What one unit in an OBJ file means. OBJ has no unit field, so this cannot be read
    /// from the file and has to be told to the importer.
    pub(crate) obj_unit: LengthUnit,
    /// A file picked for import, waiting for its options to be confirmed.
    ///
    /// How finely a STEP solid is tessellated is asked here too, and lives in `importing.step_options`
    /// rather than in a field of its own.
    pub(crate) prompt: Option<ImportPrompt>,
    /// Whether an OBJ is read as a subdivision cage rather than as a model.
    ///
    /// Defaulted to the cage, which is what an OBJ import has always meant here: the file is a
    /// control mesh and the body is the surface it describes. A triangle mesh from anywhere
    /// else is the other reading and the file cannot say which it is.
    pub(crate) obj_as: ObjReading,
    /// Everything the STEP reader is told. See [`fieldform_io::StepOptions`].
    pub(crate) step_options: fieldform_io::StepOptions,
    /// An import running in another process, if one is.
    pub(crate) pending: Option<PendingImport>,
}

/// How the part is looked at, which is no part of the document: the mode, the quality and what is drawn over the picture.
pub(crate) struct ViewSettings {
    pub(crate) mode: ViewMode,
    /// Whether the ViewCube is shown. Presentation state — it is not saved with the document
    /// and nothing about it is undoable, for the same reason the camera is not: the view is
    /// how you are looking at the model, not part of it.
    pub(crate) show_view_cube: bool,
    /// Whether a settled frame measures ambient occlusion and the key light's shadow.
    ///
    /// A view setting rather than a document one, like the section: it changes how the part
    /// is looked at and nothing about the part. On by default; the switch exists for the
    /// machine where a settled frame takes too long with it.
    pub(crate) idle_occlusion: bool,
    /// Whether a settled frame traces reflections (backlog item 54): a view setting like the
    /// one above, on by default, and off on Fast whatever it says. What a frame then traces is
    /// the material's to decide, so a matte part costs nothing either way.
    pub(crate) idle_reflections: bool,
    /// Whether the drawing plane shows its grid, which is also what Shift snaps to.
    ///
    /// A view setting, on by default. Off, there is nothing to snap to — deliberately, since a
    /// snap to something unseen cannot be aimed.
    pub(crate) show_grid: bool,
    /// The room the part is lit by. A view setting like the section; `View > Environment`.
    pub(crate) environment: fieldform_gpu::EnvironmentChoice,
    /// The draft-angle overlay. Under Display in the part panel, and `View > Draft analysis`.
    pub(crate) draft: DraftAnalysis,
    /// How much of the display this machine pays for. See [`DisplayQuality`].
    pub(crate) quality: DisplayQuality,
    /// Whether somebody chose it, this session or an earlier one, as against the startup guess
    /// from the adapter. The first measured draft may correct a guess; it never overrides a
    /// choice.
    pub(crate) quality_chosen: bool,
}

/// What a frame cost, and where that is being written.
pub(crate) struct PerfState {
    /// What each named stage cost this frame, for [`Self::note_frame_cost`] to name when the
    /// frame was slow enough to be felt. Cleared every frame.
    pub(crate) stages: Vec<(&'static str, f32)>,
    /// The `Drawing on …` line said at startup, kept so the perf log can open with it: a
    /// timing without the adapter that produced it is not a measurement (loop spec §2).
    pub(crate) adapter_line: String,
    /// Where per-frame timings are being written, while `perf` is on.
    ///
    /// Its own file rather than the console's, because the console's is capped and rewritten
    /// from what is in memory once it grows past the cap — and these lines are deliberately
    /// not in memory, at one per traced frame. They would vanish from the file at the first
    /// rewrite, which is exactly when a long session's numbers are wanted.
    pub(crate) file: Option<std::path::PathBuf>,
    /// How long each recent frame's `ui` took, by the traced view's frame number, so a GPU cost
    /// that arrives a frame late can still be printed beside the CPU time of the frame that
    /// submitted it.
    pub(crate) frame_cpu_ms: std::collections::VecDeque<(u64, f32)>,
}
