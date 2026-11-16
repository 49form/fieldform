// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A drag in progress: what is being moved, where it started, and the undo step it will become.

use crate::*;

/// How long after a graph edit another one still counts as the same gesture.
///
/// Long enough to bridge the gap between two frames of a drag on a loaded machine, short enough
/// that two deliberate clicks are never merged. It is a *supporting* condition rather than the
/// test itself — see `set_part_graph`, where the structural comparison does the real work.
pub(crate) const GESTURE_WINDOW: std::time::Duration = std::time::Duration::from_millis(400);

/// What the part looked like when the graph edit in progress began.
pub(crate) struct Gesture {
    pub(crate) part: PartId,
    pub(crate) before: Graph,
    /// Who owned what when it began, which is what goes with `before` (`block-ownership-spec.md`
    /// §4.3). Taken at the first instant and handed to every entry the gesture builds, never
    /// taken again: by a later instant the first one has settled, and can have put blocks inside
    /// one that `before` does not hold. Each later instant settles from it too, since an entry
    /// settles from the maps it is handed (`SetPartGraph`).
    pub(crate) ownership: OwnershipSnapshot,
    pub(crate) at: std::time::Instant,
}

/// The same, for a material being dragged: the channels the gesture started from.
pub(crate) struct MaterialGesture {
    pub(crate) part: PartId,
    pub(crate) before: Option<Material>,
    pub(crate) at: std::time::Instant,
}

/// What a live manipulator drag is editing, frozen at mouse-down.
///
/// The two layers the architecture separates, as two variants: a cage selection is a direct
/// edit that reverses against exact prior positions, and a body is a parametric placement that
/// reverses by restoring the graph. One widget, two commit paths, and they cannot be confused
/// because the target is decided before the first frame of the drag.
pub(crate) enum GizmoTarget {
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
    /// A section cut's plane: moved live in the document, then put back and written as one
    /// undo step when the hand comes off, as a body's placement is.
    Section { id: fieldform_scene::SectionCutId, before: fieldform_scene::SectionCut, resolved: Section },
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
pub(crate) struct GizmoDrag {
    pub(crate) drag: gizmo::Drag,
    pub(crate) target: GizmoTarget,
    /// The pivot the gizmo sat on when the drag began. Every motion is about this, and it must
    /// not follow the body it is moving.
    pub(crate) pivot: Point3<f64>,
    /// What to show beside the gizmo, refreshed each frame.
    pub(crate) readout: String,
}

/// Frozen state for an in-progress move: the selected vertices' starting positions, and the
/// plane (through their centroid, facing the camera) that screen-space drag motion is
/// projected onto to produce a 3D translation.
pub(crate) struct MoveState {
    /// Whose vertices these are, fixed at the press so the gesture commits to the mesh it began
    /// on whatever else happens during it.
    pub(crate) mesh: ControlMesh,
    /// Document positions (f64), not the f32 the drag math runs in: these are written back
    /// to the cage and handed to the undo command, so they must be the exact prior values.
    pub(crate) start_positions: Vec<(VertexId, Point3<f64>)>,
    pub(crate) plane_point: Point3<f32>,
    pub(crate) plane_normal: Vector3<f32>,
    /// Offset between the plane's true center and where the initial click actually landed
    /// on the plane, so the dragged geometry doesn't jump to re-align with the cursor the
    /// instant the drag starts.
    pub(crate) click_offset: Vector3<f32>,
}

/// A control point being dragged directly, without the manipulator.
///
/// **The manipulator cannot do this job in the view where it is most wanted.** Looking straight
/// down a plane — which is the whole point of picking one to draw on — the axis toward the camera
/// is a dot and the two plane handles are the only things that work, so placing a point meant
/// aiming at a handle rather than at the point. Dragging the point itself is what everybody
/// expects of a control point and is what makes the endpoint snap reachable, since a snap needs
/// a drag to end.
pub(crate) struct PointDrag {
    pub(crate) curve: CurveId,
    /// Every point moving, so a multi-point selection drags together the way the cage's does.
    pub(crate) indices: Vec<usize>,
    pub(crate) start_positions: Vec<Point3<f64>>,
    pub(crate) plane_point: Point3<f32>,
    pub(crate) plane_normal: Vector3<f32>,
    /// Where on the plane the click actually landed, relative to the point, so the geometry
    /// does not jump to meet the cursor the instant the drag starts.
    pub(crate) click_offset: Vector3<f32>,
    /// The curve whose end this is currently snapped onto, while it is.
    ///
    /// Live rather than decided at the end: a snap you cannot see until you let go is one you
    /// cannot aim, and the whole difficulty was not knowing whether it had taken.
    pub(crate) snapped_to: Option<CurveId>,
    /// What the gesture needs to know about the points it moves. See [`CurveGesture`].
    pub(crate) gesture: CurveGesture,
    /// Points picked on other curves, moving by the same offset and committed with these as
    /// one entry.
    pub(crate) others: Vec<CurveMove>,
}

/// One curve's part in a gesture that moves several: which of its points, where they started,
/// and what the gesture knows about them.
#[derive(Clone)]
pub(crate) struct CurveMove {
    pub(crate) curve: CurveId,
    pub(crate) indices: Vec<usize>,
    pub(crate) before: Vec<Point3<f64>>,
    pub(crate) gesture: CurveGesture,
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
pub(crate) struct CurveGesture {
    /// The moving points that sit on blocks, with the block each sits on.
    pub(crate) bound: Vec<(usize, NodeId)>,
    /// The graph before any block moved, so the gesture lands on the history as one edit from
    /// where it started. `None` when nothing moving sits on a block.
    pub(crate) graph_before: Option<Graph>,
    /// Whether a block that moves has a body reading it, in which case the traced view is
    /// rebuilt per frame so the body follows as the curve does.
    pub(crate) bodies_follow: bool,
    /// Whether moving these points changes any field at all -- the curve is read by a block,
    /// or a block the ends sit on is, or another curve on those blocks is. When nothing is,
    /// the traced view is left exactly as it was: dropping it to draft on every frame and
    /// re-settling it on release, for a picture that could not have changed, is what made
    /// editing a loose curve beside an extrude feel slow.
    pub(crate) shapes_field: bool,
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
pub(crate) fn curve_tool_ended(pending: &PendingCurve, how: &str) -> String {
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

pub(crate) struct PendingCurve {
    pub(crate) degree: usize,
    /// The start, once it has been clicked. `None` while the first click is still awaited.
    pub(crate) first: Option<Point3<f64>>,
    /// Where the next click would land, for the preview. See [`Landing`].
    pub(crate) preview: Option<Landing>,
    /// Set when what is being placed is an axis rather than a line: it is named `Axis n`, and a
    /// normal-to-surface one takes a single click on the field's surface (`split_axis`).
    pub(crate) axis: Option<AxisKind>,
    /// The axis being stood somewhere new, when this click is re-pointing one rather than making
    /// one. `None` makes a new axis, which is what the Create menu does.
    ///
    /// **Here rather than beside it on the `App`**, so the target cannot outlive the gesture that
    /// set it: one object is begun, finished and cancelled, and `Esc` clearing `curve_ui.pending`
    /// clears this with it.
    pub(crate) restanding: Option<CurveId>,
    /// Curves placed since the tool was begun, which the tool now stays on across (item 111).
    ///
    /// **So every way out says something true.** Before the tool stayed on, ending it always meant
    /// abandoning the one curve in progress, and every exit said "nothing was drawn". Once four
    /// lines can be placed in one session, that sentence is false at exactly the moment someone
    /// reads it to check their work survived.
    pub(crate) placed: u32,
}

/// Where the next click would land, kept against the cursor and camera it was computed for.
///
/// A cache rather than a per-frame computation because the landing can be a field trace, and
/// tracing sixty times a second for a cursor that has not moved is a cost with no reader.
pub(crate) struct Landing {
    pub(crate) cursor: egui::Pos2,
    pub(crate) eye: Point3<f32>,
    pub(crate) snapping: bool,
    pub(crate) grid: bool,
    pub(crate) at: Point3<f64>,
}
