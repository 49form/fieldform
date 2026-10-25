// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The triad manipulator: one widget that moves, turns and scales whatever is selected.
//!
//! Three axis handles, three planar handles, three rotation arcs and four scale handles, all
//! shown at once at the selection's pivot — the combined manipulator
//! rather than separate move/rotate/scale tools the user has to switch between.
//!
//! # What this module does and does not know
//!
//! It knows world-space geometry and screen-space pixels, and nothing else. The camera arrives
//! as two closures — one that projects a world point, one that builds a ray through a pixel —
//! so nothing here depends on `OrbitCamera`, and the drag mathematics can be tested without a
//! camera, a window or a GPU. What the manipulator is *pointed at* is likewise not its
//! business: it reports a [`Motion`], and the caller decides whether that becomes a placement
//! on a body, a cage command, or a curve command.
//!
//! # Drag mathematics
//!
//! Every drag is computed against the state frozen at mouse-down, never accumulated frame to
//! frame — with one deliberate exception. Accumulating deltas drifts under floating-point
//! error over a long drag, and the drift is invisible until someone measures the result. The
//! exception is rotation, which *must* accumulate: a single absolute angle from drag-start
//! flips sign as the drag crosses the half-turn, so the angle between consecutive frames is
//! summed instead. That is why [`Drag`] carries both a frozen start and a running total.
//!
//! # Overlay, not scene
//!
//! Painted with egui after the viewport, so every handle stays visible and clickable whatever
//! geometry is in front of it. That is the same treatment the ViewCube gets, and it is free
//! here: the renderer has no depth buffer, so drawing later *is* drawing on top.

use eframe::egui;
use nalgebra::{Point3, Rotation3, Vector3};

/// How much of the viewport's half-height a full axis handle spans.
///
/// The gizmo's world size is recomputed every frame from the camera distance so this fraction
/// stays fixed: it must not shrink to invisibility on a far body or dwarf a near one.
const REACH_FRACTION: f64 = 0.34;

/// Where the planar handles sit along their two axes, as a fraction of the reach.
const PLANE_INSET: f64 = 0.26;
/// Half-width of a planar handle's quad, as a fraction of the reach.
const PLANE_HALF: f64 = 0.09;
/// Radius of the rotation arcs, as a fraction of the reach. Just outside the axis handles, so
/// an arc and an arrowhead never compete for the same pixel.
const ARC_RADIUS: f64 = 1.16;
/// Where an axis's scale cube sits, as a fraction of the reach — past the arrowhead.
const SCALE_AT: f64 = 1.0;
/// Where the arrowhead begins, as a fraction of the reach.
const ARROW_AT: f64 = 0.84;
/// The inner gap, so the three axes do not collide visually at the pivot.
const INNER_GAP: f64 = 0.14;

/// Segments per rotation arc. Enough that the arc reads as a circle and that hit-testing it as
/// a polyline is indistinguishable from hit-testing a torus.
const ARC_SEGMENTS: usize = 48;

/// How near the cursor must come, in pixels, to grab a handle.
const GRAB_PIXELS: f32 = 7.0;
/// An arrow's reach, a little wider than a ring's: it is the handle most often aimed at, and in
/// a true view it shares its pixels with a ring.
const ARROW_PIXELS: f32 = 9.0;
/// The same for the centre cube, which is small and sits under everything else.
const CENTRE_PIXELS: f32 = 9.0;

/// How long a depth arrow is, as a fraction of the reach.
const EXTRUDE_ARROW: f64 = 0.55;

/// How long an axis must appear on screen before it can be aimed at, in pixels.
///
/// **An axis pointing at the camera projects to a dot**, and a dot sitting on the pivot is not
/// something anybody chose — it is on top of the centre handle and on top of the two axes you
/// can actually see. Looking down the back view and reaching for Y or Z, the invisible X was
/// winning the click.
///
/// The drag already refused this case (`drag_plane_normal` returns nothing when the axis and the
/// view are parallel) but only within a rounding error of exactly end-on, which is the right
/// threshold for arithmetic and far too tight for aiming: a degree off is mathematically fine
/// and still a three-pixel stub. Picking is a screen-space question, so it is asked in screen
/// space — which also means it needs no view vector and holds under both projections.
const MIN_AXIS_PIXELS: f32 = 12.0;

/// One of the thirteen handles. The `usize` is the axis: 0 = X, 1 = Y, 2 = Z.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Handle {
    /// Translate along one axis.
    Axis(usize),
    /// Translate in the plane *normal* to one axis — so `Plane(0)` is the YZ plane.
    Plane(usize),
    /// Rotate about one axis.
    Arc(usize),
    /// Scale along one axis.
    ScaleAxis(usize),
    /// Scale all three together.
    ScaleUniform,
    /// An extrude's depth: `0` is side one's far face and `1` side two's. Not one of the thirteen —
    /// a frame offers it only over a selected extrude — but routed exactly as they are, through
    /// this module's hit test and drag, so that there is not a fourth click-routing path.
    Extrude(usize),
}

impl Handle {
    /// The axis this handle belongs to, for colouring. `None` for the uniform-scale cube,
    /// which belongs to no axis.
    pub fn axis(self) -> Option<usize> {
        match self {
            Handle::Axis(a) | Handle::Plane(a) | Handle::Arc(a) | Handle::ScaleAxis(a) => Some(a),
            Handle::ScaleUniform | Handle::Extrude(_) => None,
        }
    }
}

/// Which handles a target offers.
///
/// Not every selection can meaningfully use all of them, and a control that visibly does
/// nothing when dragged is worse than one that is absent.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Handles {
    /// A single point has no extent, so rotating or scaling about it is a no-op.
    pub rotate_and_scale: bool,
    /// Whether the three per-axis scale cubes are offered, or only the uniform one.
    ///
    /// False for a placed body. A non-uniform scale would stop that body being a true distance
    /// field, and blends refuse those at compile time — so per-axis cubes there would quietly
    /// manufacture bodies that can never be filleted again, discovered at some later blend
    /// rather than at the drag that caused it. Cage and curve selections keep them: those move
    /// vertices, they do not wrap a field.
    pub per_axis_scale: bool,
}

impl Handles {
    /// A whole body: everything except per-axis scale.
    pub fn body() -> Self {
        Self { rotate_and_scale: true, per_axis_scale: false }
    }

    /// A cage or curve selection with extent — an edge, a face, several points.
    pub fn extended() -> Self {
        Self { rotate_and_scale: true, per_axis_scale: true }
    }

    /// A single vertex or control point: translate only.
    pub fn single_point() -> Self {
        Self { rotate_and_scale: false, per_axis_scale: false }
    }

    /// Every handle on offer, nearest-to-the-pivot last so painting order reads correctly.
    pub fn all(self) -> Vec<Handle> {
        let mut out = Vec::with_capacity(13);
        if self.rotate_and_scale {
            out.extend((0..3).map(Handle::Arc));
            if self.per_axis_scale {
                out.extend((0..3).map(Handle::ScaleAxis));
            }
            // The uniform cube sits at the centre whenever there is extent to scale: a loop or
            // a ring of the cage is scaled as a whole far more often than along one axis (the
            // owner, 2026-10-24, scaling a selected loop), and a body never had the per-axis
            // cubes to reach for instead.
            out.push(Handle::ScaleUniform);
        }
        out.extend((0..3).map(Handle::Axis));
        out.extend((0..3).map(Handle::Plane));
        out
    }
}

/// Where an extrude's depth arrows stand: one on each face whose depth can be dragged.
///
/// **Standing on the far face and pointing away from the solid**, which is where a person reaches
/// to pull the material out. Dragging one along the sweep is a motion like any axis handle's, and
/// the caller reads it as a change of depth.
#[derive(Copy, Clone, Debug)]
pub struct ExtrudeArrows {
    /// The sweep direction, a unit vector: the way a positive depth runs.
    pub direction: Vector3<f64>,
    /// Where side one's far face is and, when the extrude has a second side, side two's.
    pub tips: [Option<Point3<f64>>; 2],
}

/// Where the manipulator is, how it is oriented, and how big it should be drawn.
#[derive(Copy, Clone, Debug)]
pub struct Frame {
    pub pivot: Point3<f64>,
    /// The manipulator's own axes. Identity is world-aligned; a face selection turns it so one
    /// axis leaves the surface, because moving along a face's own normal is more often what
    /// was meant than moving along a world axis.
    pub basis: Rotation3<f64>,
    /// World length of a full axis handle. Recomputed each frame — see [`Frame::reach_for`].
    pub reach: f64,
    pub handles: Handles,
    /// The depth arrows of a selected extrude, beside the triad rather than instead of it.
    pub extrude: Option<ExtrudeArrows>,
}

impl Frame {
    /// The world length that keeps the gizmo a constant size on screen.
    ///
    /// Pinned to a fraction of the world height the viewport covers, whatever produced that
    /// height. Under perspective it is `d · tan(fovy/2)` and grows with distance; under
    /// orthographic it is the view's own half-height and does not depend on distance at all.
    /// Using the perspective formula in an orthographic view would shrink the handles as the
    /// camera pulled back while the model stayed exactly the same size — the one thing this is
    /// here to prevent.
    pub fn reach_for(half_height: f64) -> f64 {
        half_height.max(1e-6) * REACH_FRACTION
    }

    fn axis(&self, i: usize) -> Vector3<f64> {
        self.basis * Vector3::ith(i, 1.0)
    }

    /// The two axes spanning the plane normal to axis `i`.
    fn plane_axes(&self, i: usize) -> (Vector3<f64>, Vector3<f64>) {
        (self.axis((i + 1) % 3), self.axis((i + 2) % 3))
    }

    /// The endpoints of an axis handle's shaft, in world space.
    fn axis_segment(&self, i: usize) -> (Point3<f64>, Point3<f64>) {
        let a = self.axis(i);
        (self.pivot + a * self.reach * INNER_GAP, self.pivot + a * self.reach * ARROW_AT)
    }

    /// The four corners of a planar handle's quad.
    fn plane_corners(&self, i: usize) -> [Point3<f64>; 4] {
        let (u, v) = self.plane_axes(i);
        let c = self.pivot + (u + v) * self.reach * PLANE_INSET;
        let (du, dv) = (u * self.reach * PLANE_HALF, v * self.reach * PLANE_HALF);
        [c - du - dv, c + du - dv, c + du + dv, c - du + dv]
    }

    /// A rotation arc, as a closed ring of points in the plane normal to axis `i`.
    fn arc_points(&self, i: usize) -> Vec<Point3<f64>> {
        let (u, v) = self.plane_axes(i);
        let r = self.reach * ARC_RADIUS;
        (0..=ARC_SEGMENTS)
            .map(|s| {
                let t = s as f64 / ARC_SEGMENTS as f64 * std::f64::consts::TAU;
                self.pivot + (u * t.cos() + v * t.sin()) * r
            })
            .collect()
    }

    /// The depth arrows on offer.
    fn extrude_handles(&self) -> Vec<Handle> {
        match &self.extrude {
            Some(arrows) => (0..2).filter(|i| arrows.tips[*i].is_some()).map(Handle::Extrude).collect(),
            None => Vec::new(),
        }
    }

    /// An extrude arrow, from its tip out along the way it points: away from the solid, which is
    /// the sweep for the first side and against it for the second.
    fn extrude_segment(&self, side: usize) -> Option<(Point3<f64>, Point3<f64>)> {
        let arrows = self.extrude.as_ref()?;
        let tip = arrows.tips.get(side).copied().flatten()?;
        let out = if side == 0 { arrows.direction } else { -arrows.direction };
        Some((tip, tip + out * self.reach * EXTRUDE_ARROW))
    }

    /// Where a handle's grab point sits, for the cubes and for depth ordering.
    fn handle_anchor(&self, handle: Handle) -> Point3<f64> {
        match handle {
            Handle::Extrude(i) => self.extrude_segment(i).map_or(self.pivot, |s| s.1),
            Handle::Axis(i) => self.axis_segment(i).1,
            Handle::Plane(i) => {
                let (u, v) = self.plane_axes(i);
                self.pivot + (u + v) * self.reach * PLANE_INSET
            }
            Handle::Arc(i) => self.pivot + self.axis(i) * self.reach * ARC_RADIUS,
            Handle::ScaleAxis(i) => self.pivot + self.axis(i) * self.reach * SCALE_AT,
            Handle::ScaleUniform => self.pivot,
        }
    }
}

/// What a drag has produced so far, relative to the state at mouse-down.
///
/// Always relative to drag-start, never to the previous frame, so the caller applies it to the
/// positions it froze when the drag began and a dropped frame changes nothing.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Motion {
    Translate(Vector3<f64>),
    /// A turn about `axis` through the gizmo's pivot.
    Rotate { axis: Vector3<f64>, degrees: f64 },
    /// A factor, along `axis` if the handle was a per-axis cube, or uniform if it was the
    /// centre one.
    Scale { axis: Option<Vector3<f64>>, factor: f64 },
}

/// An in-progress drag: everything frozen at mouse-down, plus the running total a rotation
/// needs.
#[derive(Clone, Debug)]
pub struct Drag {
    pub handle: Handle,
    /// The frame as it was when the drag began. The live frame moves as the body does, and
    /// computing against a moving frame feeds the drag its own output.
    frame: Frame,
    plane_normal: Vector3<f64>,
    /// A point the drag plane passes through: the pivot for every handle but a depth arrow, whose
    /// line does not pass through it. A plane through the pivot would be measured against a line
    /// beside it, and the arrow would creep as the view turned.
    plane_point: Point3<f64>,
    /// Where along the constrained axis the drag started, for axis moves and axis scales.
    start_param: f64,
    /// Where on the drag plane the cursor first landed, for planar moves.
    start_hit: Point3<f64>,
    /// Offset between the handle and where the click actually landed, so the body does not
    /// jump to meet the cursor the instant the drag starts.
    grab_offset: f64,
    /// The previous frame's in-plane direction, for the rotation delta.
    last_direction: Option<Vector3<f64>>,
    /// Degrees turned so far, summed across frames — see this module's header.
    turned: f64,
    /// Screen position at mouse-down, for the uniform-scale handle, which is the one drag
    /// with no world-space constraint to project onto.
    start_screen: egui::Pos2,
}

impl Drag {
    /// Begins a drag, or returns `None` if the cursor ray gives no usable constraint — which
    /// happens when the view is looking almost exactly along the handle's own axis.
    pub fn begin(
        frame: &Frame,
        handle: Handle,
        origin: Point3<f64>,
        direction: Vector3<f64>,
        screen: egui::Pos2,
    ) -> Option<Self> {
        let plane_normal = drag_plane_normal(frame, handle, direction)?;
        let plane_point = match handle {
            Handle::Extrude(i) => frame.extrude_segment(i)?.0,
            _ => frame.pivot,
        };
        let hit = ray_plane(origin, direction, plane_point, plane_normal)?;

        let (start_param, grab_offset) = match handle {
            Handle::Extrude(_) => {
                let axis = frame.extrude?.direction;
                ((hit - plane_point).dot(&axis), 0.0)
            }
            Handle::Axis(i) | Handle::ScaleAxis(i) => {
                let axis = frame.axis(i);
                let param = (hit - frame.pivot).dot(&axis);
                // What the handle itself is at, so the offset is how far off the handle the
                // click landed rather than how far from the pivot.
                let anchor = (frame.handle_anchor(handle) - frame.pivot).dot(&axis);
                (param, param - anchor)
            }
            _ => (0.0, 0.0),
        };

        Some(Self {
            handle,
            frame: *frame,
            plane_normal,
            plane_point,
            start_param,
            start_hit: hit,
            grab_offset,
            last_direction: (hit - frame.pivot).try_normalize(1e-9),
            turned: 0.0,
            start_screen: screen,
        })
    }

    /// Advances the drag to the cursor's current position and reports the motion so far.
    ///
    /// Returns `None` for a frame whose ray misses the drag plane — a grazing view — which the
    /// caller should treat as "no change this frame" rather than as the end of the drag.
    pub fn update(
        &mut self,
        origin: Point3<f64>,
        direction: Vector3<f64>,
        screen: egui::Pos2,
    ) -> Option<Motion> {
        let frame = self.frame;

        if self.handle == Handle::ScaleUniform {
            // The one handle with no world constraint: it is dragged in any direction, so the
            // measure is screen distance. Rightward and upward grow, which is the convention
            // every tool in this family uses.
            let delta = (screen.x - self.start_screen.x) - (screen.y - self.start_screen.y);
            let factor = (1.0 + delta as f64 / 200.0).max(MIN_SCALE);
            return Some(Motion::Scale { axis: None, factor });
        }

        let hit = ray_plane(origin, direction, self.plane_point, self.plane_normal)?;

        Some(match self.handle {
            Handle::Extrude(_) => {
                let axis = frame.extrude?.direction;
                let now = (hit - self.plane_point).dot(&axis);
                Motion::Translate(axis * (now - self.start_param))
            }
            Handle::Axis(i) => {
                let axis = frame.axis(i);
                let now = (hit - frame.pivot).dot(&axis);
                Motion::Translate(axis * (now - self.start_param))
            }
            Handle::Plane(_) => Motion::Translate(hit - self.start_hit),
            Handle::Arc(i) => {
                let axis = frame.axis(i);
                let direction_now = (hit - frame.pivot).try_normalize(1e-9)?;
                if let Some(previous) = self.last_direction {
                    // Signed angle between consecutive frames, about this arc's own axis.
                    // Summing these is what lets a drag pass the half-turn without the sign
                    // flip an absolute angle from drag-start would show.
                    let cross = previous.cross(&direction_now);
                    let step = cross.dot(&axis).atan2(previous.dot(&direction_now));
                    self.turned += step.to_degrees();
                }
                self.last_direction = Some(direction_now);
                Motion::Rotate { axis, degrees: self.turned }
            }
            Handle::ScaleAxis(i) => {
                let axis = frame.axis(i);
                let now = (hit - frame.pivot).dot(&axis) - self.grab_offset;
                let start = self.start_param - self.grab_offset;
                // Clamped, never allowed through zero: a handle dragged past the pivot would
                // otherwise flip the sign and turn the body inside out, which reads as a bug
                // even where it is arithmetically what was asked for.
                let factor = if start.abs() < 1e-9 {
                    1.0
                } else {
                    (now / start).max(MIN_SCALE)
                };
                Motion::Scale { axis: Some(axis), factor }
            }
            Handle::ScaleUniform => unreachable!("handled above, before the plane cast"),
        })
    }

    /// What to show next to the gizmo while this drag is live.
    pub fn readout(&self, motion: Motion) -> String {
        match motion {
            Motion::Translate(d) => format!("{:+.3}, {:+.3}, {:+.3} mm", d.x, d.y, d.z),
            Motion::Rotate { degrees, .. } => format!("{degrees:+.2}°"),
            Motion::Scale { factor, .. } => format!("×{factor:.4}"),
        }
    }
}

/// A scale may approach zero but never reach or pass it — see [`Drag::update`].
const MIN_SCALE: f64 = 1e-3;

/// The plane a handle's drag is measured on.
///
/// For an axis handle that is the plane containing the axis and facing the camera as squarely
/// as possible, which is `axis × (axis × view)`. For a planar handle it is that plane itself.
/// For an arc it is the plane the arc lies in.
///
/// `None` when the construction collapses: looking almost exactly along an axis leaves
/// `axis × view` near zero, and normalising that is how a handle starts jittering. Refusing
/// the drag is better than accepting one whose plane is noise.
fn drag_plane_normal(frame: &Frame, handle: Handle, view: Vector3<f64>) -> Option<Vector3<f64>> {
    match handle {
        Handle::Axis(i) | Handle::ScaleAxis(i) => {
            let axis = frame.axis(i);
            axis.cross(&axis.cross(&view)).try_normalize(1e-6)
        }
        Handle::Plane(i) | Handle::Arc(i) => Some(frame.axis(i)),
        Handle::ScaleUniform => Some(-view),
        Handle::Extrude(_) => {
            let axis = frame.extrude?.direction;
            axis.cross(&axis.cross(&view)).try_normalize(1e-6)
        }
    }
}

/// Where a ray meets a plane, or `None` if it runs parallel to it.
fn ray_plane(
    origin: Point3<f64>,
    direction: Vector3<f64>,
    plane_point: Point3<f64>,
    plane_normal: Vector3<f64>,
) -> Option<Point3<f64>> {
    let denominator = direction.dot(&plane_normal);
    if denominator.abs() < 1e-9 {
        return None;
    }
    let t = (plane_point - origin).dot(&plane_normal) / denominator;
    // Behind the eye is not a hit: without this a handle stays draggable after the cursor
    // passes the vanishing point, and the body leaps to the far side of the pivot.
    (t > 0.0).then(|| origin + direction * t)
}

/// Which handle the cursor is over, if any.
///
/// Hit-tested in screen space against each handle's projected outline rather than in 3D
/// against analytic solids. The two agree wherever it matters and this one is honest about
/// what the user is actually aiming at: a pixel. It is also how the cage's own vertex and edge
/// picking already works, so the tolerances mean the same thing throughout.
///
/// Ties go to whichever handle is nearest the camera, since that is the one visibly on top.
pub fn hit_test(
    frame: &Frame,
    cursor: egui::Pos2,
    project: &dyn Fn(Point3<f64>) -> Option<egui::Pos2>,
    eye: Point3<f64>,
) -> Option<Handle> {
    let mut best: Option<(Handle, u8, f64)> = None;

    // **The centre is the uniform cube's, outright.** Every arrow and every plane leaves the
    // pivot, so a click there is within reach of all of them and the arrows would rank first;
    // the cube at the centre is what a click on the centre means.
    if frame.handles.rotate_and_scale
        && let Some(centre) = project(frame.pivot)
        && centre.distance(cursor) <= CENTRE_PIXELS
    {
        return Some(Handle::ScaleUniform);
    }

    // **An arrow beats a ring where they overlap.** Seen straight down an axis -- a true view --
    // that axis's rotation ring lies flat in the view and crosses the two arrows in it, and the
    // ring's anchor is often the nearer, so the depth rule handed the ring every click aimed at
    // an arrow. Asked for by the owner, 2026-10-24: the arrows rank first, the rings last, and
    // depth decides only within a rank.
    let rank = |handle: Handle| -> u8 {
        match handle {
            Handle::Axis(_) | Handle::Extrude(_) => 0,
            Handle::Plane(_) | Handle::ScaleAxis(_) | Handle::ScaleUniform => 1,
            Handle::Arc(_) => 2,
        }
    };
    for handle in frame.handles.all().into_iter().chain(frame.extrude_handles()) {
        let Some(distance) = handle_screen_distance(frame, handle, cursor, project) else {
            continue;
        };
        let threshold = match handle {
            Handle::ScaleUniform => CENTRE_PIXELS,
            Handle::Axis(_) | Handle::Extrude(_) => ARROW_PIXELS,
            _ => GRAB_PIXELS,
        };
        if distance > threshold {
            continue;
        }
        let depth = (frame.handle_anchor(handle) - eye).norm();
        let candidate = (rank(handle), depth);
        let better = match best {
            None => true,
            Some((_, best_rank, best_depth)) => candidate < (best_rank, best_depth),
        };
        if better {
            best = Some((handle, candidate.0, candidate.1));
        }
    }
    best.map(|(handle, _, _)| handle)
}

/// How far the cursor is, in pixels, from a handle's projected outline.
fn handle_screen_distance(
    frame: &Frame,
    handle: Handle,
    cursor: egui::Pos2,
    project: &dyn Fn(Point3<f64>) -> Option<egui::Pos2>,
) -> Option<f32> {
    match handle {
        Handle::Extrude(i) => {
            let (a, b) = frame.extrude_segment(i)?;
            let (a, b) = (project(a)?, project(b)?);
            // Seen end-on it is a dot on the face, not an arrow anybody aimed at.
            if (b - a).length() < MIN_AXIS_PIXELS {
                return None;
            }
            Some(segment_distance(cursor, a, b))
        }
        Handle::Axis(i) => {
            let (a, b) = frame.axis_segment(i);
            let (a, b) = (project(a)?, project(b)?);
            // Too foreshortened to have been aimed at. `None` rather than a large distance,
            // because the handle is not merely far from the cursor — it is not on offer.
            if (b - a).length() < MIN_AXIS_PIXELS {
                return None;
            }
            Some(segment_distance(cursor, a, b))
        }
        Handle::Plane(i) => {
            let corners: Option<Vec<egui::Pos2>> =
                frame.plane_corners(i).iter().map(|p| project(*p)).collect();
            let corners = corners?;
            if point_in_polygon(cursor, &corners) {
                Some(0.0)
            } else {
                Some(
                    (0..corners.len())
                        .map(|k| {
                            segment_distance(cursor, corners[k], corners[(k + 1) % corners.len()])
                        })
                        .fold(f32::INFINITY, f32::min),
                )
            }
        }
        Handle::Arc(i) => {
            let points: Option<Vec<egui::Pos2>> =
                frame.arc_points(i).iter().map(|p| project(*p)).collect();
            let points = points?;
            Some(
                points
                    .windows(2)
                    .map(|w| segment_distance(cursor, w[0], w[1]))
                    .fold(f32::INFINITY, f32::min),
            )
        }
        Handle::ScaleAxis(i) => {
            // Same rule, measured along the axis it sits on: a scale grip whose axis has
            // collapsed to a dot is under the uniform-scale handle at the centre, and would
            // take clicks meant for it.
            let (a, b) = frame.axis_segment(i);
            if (project(b)? - project(a)?).length() < MIN_AXIS_PIXELS {
                return None;
            }
            Some(project(frame.handle_anchor(handle))?.distance(cursor))
        }
        Handle::ScaleUniform => Some(project(frame.handle_anchor(handle))?.distance(cursor)),
    }
}

/// Distance from a point to a 2D segment.
fn segment_distance(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let length_squared = ab.length_sq();
    if length_squared < 1e-9 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / length_squared).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Whether a point is inside a convex polygon, by consistent winding.
fn point_in_polygon(p: egui::Pos2, polygon: &[egui::Pos2]) -> bool {
    let mut sign = 0.0_f32;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        let edge = b - a;
        let to_p = p - a;
        let cross = edge.x * to_p.y - edge.y * to_p.x;
        if cross.abs() > 1e-6 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Red for X, green for Y, blue for Z — the convention this family of tools has used long
/// enough that departing from it would be the surprising choice.
fn axis_colour(axis: Option<usize>) -> egui::Color32 {
    match axis {
        Some(0) => egui::Color32::from_rgb(226, 74, 74),
        Some(1) => egui::Color32::from_rgb(96, 196, 96),
        Some(2) => egui::Color32::from_rgb(86, 138, 232),
        _ => egui::Color32::from_gray(210),
    }
}

/// Orange for a depth arrow, which belongs to no axis and is not the triad's.
const EXTRUDE_COLOUR: egui::Color32 = egui::Color32::from_rgb(240, 150, 60);

/// Yellow for whatever is live, whichever axis it belongs to, so "what will move if I drag
/// now" is never ambiguous.
const LIVE: egui::Color32 = egui::Color32::from_rgb(255, 214, 64);

/// Paints the manipulator.
///
/// `live` is the handle being dragged, if any; `hovered` the one under the cursor. A drag wins:
/// once the pointer is down the cursor may wander off the handle entirely, and the highlight
/// has to stay on what is actually moving.
pub fn paint(
    painter: &egui::Painter,
    frame: &Frame,
    hovered: Option<Handle>,
    live: Option<Handle>,
    project: &dyn Fn(Point3<f64>) -> Option<egui::Pos2>,
) {
    let highlight = live.or(hovered);
    let colour_of = |handle: Handle| {
        if Some(handle) == highlight {
            LIVE
        } else if matches!(handle, Handle::Extrude(_)) {
            EXTRUDE_COLOUR
        } else {
            axis_colour(handle.axis())
        }
    };
    // While a handle is live the others are dimmed rather than hidden: they say what the
    // gizmo still is, without competing with the one that is moving.
    let dimmed = |c: egui::Color32| {
        if live.is_some() {
            egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 70)
        } else {
            c
        }
    };
    let tint = |handle: Handle| {
        if Some(handle) == highlight {
            LIVE
        } else {
            dimmed(colour_of(handle))
        }
    };

    for handle in frame.handles.all() {
        match handle {
            Handle::Arc(i) => {
                let points: Vec<egui::Pos2> =
                    frame.arc_points(i).iter().filter_map(|p| project(*p)).collect();
                for w in points.windows(2) {
                    painter.line_segment([w[0], w[1]], egui::Stroke::new(1.6_f32, tint(handle)));
                }
            }
            Handle::Plane(i) => {
                let corners: Vec<egui::Pos2> =
                    frame.plane_corners(i).iter().filter_map(|p| project(*p)).collect();
                if corners.len() == 4 {
                    let c = tint(handle);
                    painter.add(egui::Shape::convex_polygon(
                        corners.clone(),
                        egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 60),
                        egui::Stroke::new(1.2_f32, c),
                    ));
                }
            }
            Handle::Axis(i) => {
                let (a, b) = frame.axis_segment(i);
                if let (Some(pa), Some(pb)) = (project(a), project(b)) {
                    painter.line_segment([pa, pb], egui::Stroke::new(2.4_f32, tint(handle)));
                    // The arrowhead is drawn as a short thick stub rather than a cone: it is
                    // three pixels of screen at this size, and a projected cone would be more
                    // arithmetic for a shape nobody can resolve.
                    let tip = frame.pivot + frame.axis(i) * frame.reach;
                    if let Some(pt) = project(tip) {
                        painter.line_segment([pb, pt], egui::Stroke::new(5.0_f32, tint(handle)));
                    }
                }
            }
            // Drawn below, with the arrows: it is not one of the triad's handles.
            Handle::Extrude(_) => {}
            Handle::ScaleAxis(_) | Handle::ScaleUniform => {
                if let Some(p) = project(frame.handle_anchor(handle)) {
                    let half = if handle == Handle::ScaleUniform { 4.5 } else { 3.5 };
                    painter.rect_filled(
                        egui::Rect::from_center_size(p, egui::vec2(half * 2.0, half * 2.0)),
                        1.0,
                        tint(handle),
                    );
                }
            }
        }
    }

    // The depth arrows: a shaft out of the face, and a short thick head, as the triad's axes are.
    for handle in frame.extrude_handles() {
        let Handle::Extrude(side) = handle else { continue };
        let Some((tip, out)) = frame.extrude_segment(side) else { continue };
        if let (Some(a), Some(b)) = (project(tip), project(out)) {
            let colour = tint(handle);
            painter.line_segment([a, b], egui::Stroke::new(2.4_f32, colour));
            // The head: the last fifth of the shaft, thickened.
            painter.line_segment([a + (b - a) * 0.8, b], egui::Stroke::new(6.0_f32, colour));
            // A ring on the face, where the arrow stands, so what it belongs to is plain.
            painter.circle_stroke(a, 4.0, egui::Stroke::new(1.4_f32, colour));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_at(pivot: Point3<f64>) -> Frame {
        Frame {
            pivot,
            basis: Rotation3::identity(),
            reach: 10.0,
            handles: Handles::extended(),
            extrude: None,
        }
    }

    /// Every screen point the painter would draw for `frame` under `camera`, by handle.
    ///
    /// Mirrors `paint`'s own list — an arc's points, a plane's corners, an axis's segment and its
    /// tip, a scale handle's anchor — so a property over this is a property over what is drawn.
    fn drawn_points(
        frame: &Frame,
        camera: &crate::camera::OrbitCamera,
        aspect: f32,
        rect: egui::Rect,
    ) -> Vec<(Handle, egui::Pos2)> {
        let project =
            |p: Point3<f64>| camera.project_to_screen(aspect, rect, p.cast::<f32>());
        let mut out = Vec::new();
        for handle in frame.handles.all() {
            let world: Vec<Point3<f64>> = match handle {
                Handle::Arc(i) => frame.arc_points(i),
                Handle::Plane(i) => frame.plane_corners(i).to_vec(),
                Handle::Axis(i) => {
                    let (a, b) = frame.axis_segment(i);
                    vec![a, b, frame.pivot + frame.axis(i) * frame.reach]
                }
                Handle::ScaleAxis(_) | Handle::ScaleUniform => vec![frame.handle_anchor(handle)],
                Handle::Extrude(_) => Vec::new(),
            };
            out.extend(world.into_iter().filter_map(project).map(|p| (handle, p)));
        }
        out
    }

    /// **No part of the manipulator is drawn far from its pivot, at any ViewCube face** (item 106).
    ///
    /// The owner's screenshot: a RIGHT view with a red line the full height of the viewport
    /// through the pivot, the rest of the manipulator its ordinary size. This puts the real camera
    /// on every face, the way `snap view=` does, under both projections, with the pivot at the
    /// target and off to one side, and projects every point the painter would draw. A drawn point
    /// further from the pivot than the viewport is tall is the streak.
    #[test]
    fn the_manipulator_stays_near_its_pivot_at_every_view_cube_face() {
        use crate::camera::OrbitCamera;
        use crate::display_quality::Projection;
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(424.0, 517.0));
        let aspect = rect.width() / rect.height();
        // **Bounded by the widget's own size, not the viewport's.** The farthest thing drawn is
        // the rings, at `ARC_RADIUS` reaches, and a reach is `REACH_FRACTION` of the half-height
        // — about 102 px here, the size of the ring in the owner's screenshot. Twice that is room
        // for perspective to enlarge whatever leans toward the eye. A first version bounded by the
        // whole viewport height, and could not have caught the report: a line from the top edge
        // to the bottom through a centred pivot has its ends only half a viewport away.
        let widget_bound =
            (2.0 * ARC_RADIUS * REACH_FRACTION * rect.height() as f64 / 2.0) as f32;
        let mut worst: Vec<String> = Vec::new();
        for projection in [Projection::Perspective, Projection::Orthographic] {
            // Every region the cube offers — six faces, twelve edges, eight corners — and each one
            // tilted a little, since the screenshot's cube showed RIGHT with a strip of BOTTOM.
            let mut views: Vec<(String, Vector3<f32>)> = Vec::new();
            for x in -1i8..=1 {
                for y in -1i8..=1 {
                    for z in -1i8..=1 {
                        if (x, y, z) == (0, 0, 0) {
                            continue;
                        }
                        let d = Vector3::new(x as f32, y as f32, z as f32).normalize();
                        views.push((format!("[{x},{y},{z}]"), d));
                        for by in [1.0f32, 5.0, 12.0] {
                            let side = if d.z.abs() < 0.9 { Vector3::z() } else { Vector3::x() };
                            let r = nalgebra::UnitQuaternion::from_axis_angle(
                                &nalgebra::Unit::new_normalize(d.cross(&side)),
                                by.to_radians(),
                            );
                            views.push((format!("[{x},{y},{z}] +{by} deg"), r * d));
                        }
                    }
                }
            }
            for (face, toward) in &views {
                for pivot in [Point3::origin(), Point3::new(20.0, -12.0, 8.0)] {
                    let mut camera = OrbitCamera::default();
                    camera.projection = projection;
                    camera.frame(Point3::origin(), 50.0);
                    let up = if toward.z.abs() > 0.9 { Vector3::y() } else { Vector3::z() };
                    camera.begin_snap_toward(*toward, up);
                    while camera.tick_snap(10.0) {}
                    let frame = Frame {
                        pivot,
                        basis: Rotation3::identity(),
                        reach: Frame::reach_for(
                            camera.view_half_height(pivot.cast::<f32>()) as f64,
                        ),
                        handles: Handles::extended(),
                        extrude: None,
                    };
                    let Some(centre) =
                        camera.project_to_screen(aspect, rect, pivot.cast::<f32>())
                    else {
                        continue;
                    };
                    for (handle, p) in drawn_points(&frame, &camera, aspect, rect) {
                        let away = (p - centre).length();
                        if away > widget_bound {
                            worst.push(format!(
                                "{projection:?} view {face} pivot {pivot:?}: {handle:?} drawn {away:.0} px from the pivot"
                            ));
                        }
                    }
                }
            }
        }
        assert!(worst.is_empty(), "{} drawn points streak:
{}", worst.len(), worst.join("
"));
    }

    /// **The control for the sweep above: it really does look down each axis.** A sweep that
    /// never reached the views it names would pass vacuously, so this asks the one thing an
    /// on-axis view must show — the axis pointing at the camera foreshortened to a dot, and the
    /// other two at their full length.
    #[test]
    fn looking_down_an_axis_makes_that_axis_a_dot() {
        use crate::camera::OrbitCamera;
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(424.0, 517.0));
        let aspect = rect.width() / rect.height();
        for (i, toward) in [Vector3::x(), Vector3::y(), Vector3::z()].into_iter().enumerate() {
            let mut camera = OrbitCamera::default();
            camera.frame(Point3::origin(), 50.0);
            let up = if i == 2 { Vector3::y() } else { Vector3::z() };
            camera.begin_snap_toward(toward, up);
            while camera.tick_snap(10.0) {}
            let pivot = Point3::origin();
            let frame = Frame {
                pivot,
                basis: Rotation3::identity(),
                reach: Frame::reach_for(camera.view_half_height(pivot.cast::<f32>()) as f64),
                handles: Handles::extended(),
                extrude: None,
            };
            let on_screen = |axis: usize| {
                let (a, b) = frame.axis_segment(axis);
                let p = |q: Point3<f64>| camera.project_to_screen(aspect, rect, q.cast::<f32>());
                (p(b).unwrap() - p(a).unwrap()).length()
            };
            assert!(on_screen(i) < 1.0, "axis {i} viewed head on is {} px, not a dot", on_screen(i));
            for other in (0..3).filter(|&o| o != i) {
                assert!(on_screen(other) > 20.0, "axis {other} beside the view is {} px", on_screen(other));
            }
        }
    }

    /// A ray from a camera looking down -Z at the given world point.
    fn ray_at(target: Point3<f64>) -> (Point3<f64>, Vector3<f64>) {
        let eye = Point3::new(target.x, target.y, target.z + 100.0);
        (eye, Vector3::new(0.0, 0.0, -1.0))
    }

    /// An axis drag must move along its own axis and no other, whatever the cursor does off to
    /// the side — that constraint is the entire reason to have a single-axis handle.
    #[test]
    fn an_axis_drag_moves_only_along_that_axis() {
        let frame = frame_at(Point3::origin());
        let (origin, direction) = ray_at(Point3::new(5.0, 0.0, 0.0));
        let mut drag = Drag::begin(
            &frame,
            Handle::Axis(0),
            origin,
            direction,
            egui::pos2(0.0, 0.0),
        )
        .unwrap();

        // Move the cursor 3 mm along X and 7 mm along Y. Only the X part may survive.
        let (o2, d2) = ray_at(Point3::new(8.0, 7.0, 0.0));
        let Some(Motion::Translate(delta)) = drag.update(o2, d2, egui::pos2(0.0, 0.0)) else {
            panic!("an axis drag should translate");
        };
        assert!((delta.x - 3.0).abs() < 1e-9, "{delta:?}");
        assert!(delta.y.abs() < 1e-9, "{delta:?}");
        assert!(delta.z.abs() < 1e-9, "{delta:?}");
    }

    /// Relative to drag-start, not to the previous frame. A drag that reaches the same place by
    /// a long wandering route must report the same motion as one that goes straight there,
    /// which is what makes a dropped frame harmless.
    #[test]
    fn a_drag_reports_the_same_motion_however_it_got_there() {
        let frame = frame_at(Point3::origin());
        let (origin, direction) = ray_at(Point3::new(5.0, 0.0, 0.0));

        let mut direct = Drag::begin(&frame, Handle::Axis(0), origin, direction, egui::Pos2::ZERO)
            .unwrap();
        let (o, d) = ray_at(Point3::new(9.0, 0.0, 0.0));
        let straight = direct.update(o, d, egui::Pos2::ZERO).unwrap();

        let mut wandering =
            Drag::begin(&frame, Handle::Axis(0), origin, direction, egui::Pos2::ZERO).unwrap();
        for x in [40.0, -25.0, 3.0, 9.0] {
            let (o, d) = ray_at(Point3::new(x, 0.0, 0.0));
            wandering.update(o, d, egui::Pos2::ZERO);
        }
        let (o, d) = ray_at(Point3::new(9.0, 0.0, 0.0));
        let wandered = wandering.update(o, d, egui::Pos2::ZERO).unwrap();

        assert_eq!(straight, wandered);
    }

    /// The bug this whole accumulation scheme exists to avoid: a single absolute angle from
    /// drag-start flips sign as the drag passes the half-turn, so the readout jumps from +179
    /// to -179 and the body snaps backwards. Summed per-frame deltas pass straight through.
    #[test]
    fn a_rotation_past_the_half_turn_keeps_climbing_instead_of_flipping_sign() {
        let frame = frame_at(Point3::origin());
        // Look along +Z so the XY plane — the plane of the Z arc — faces the camera squarely.
        let eye = Point3::new(0.0, 0.0, 100.0);
        let cast = |angle_degrees: f64| {
            let a = angle_degrees.to_radians();
            let target = Point3::new(20.0 * a.cos(), 20.0 * a.sin(), 0.0);
            (eye, (target - eye).normalize())
        };

        let (o, d) = cast(0.0);
        let mut drag = Drag::begin(&frame, Handle::Arc(2), o, d, egui::Pos2::ZERO).unwrap();

        let mut last = 0.0;
        // Round the circle and well past it, in steps small enough to be unambiguous.
        for step in 1..=100 {
            let angle = step as f64 * 5.0;
            let (o, d) = cast(angle);
            let Some(Motion::Rotate { degrees, .. }) = drag.update(o, d, egui::Pos2::ZERO) else {
                panic!("an arc drag should rotate");
            };
            assert!(degrees > last, "angle went backwards at {angle}: {degrees} after {last}");
            last = degrees;
        }
        // 100 steps of 5 degrees, accumulated without a wrap.
        assert!((last - 500.0).abs() < 1e-6, "{last}");
    }

    /// Dragging a scale handle back through the pivot must not invert the body. The factor
    /// bottoms out instead of going negative.
    #[test]
    fn a_scale_dragged_through_the_pivot_clamps_instead_of_inverting() {
        let frame = frame_at(Point3::origin());
        let (origin, direction) = ray_at(Point3::new(10.0, 0.0, 0.0));
        let mut drag =
            Drag::begin(&frame, Handle::ScaleAxis(0), origin, direction, egui::Pos2::ZERO)
                .unwrap();

        for x in [5.0, 0.0, -5.0, -40.0] {
            let (o, d) = ray_at(Point3::new(x, 0.0, 0.0));
            let Some(Motion::Scale { factor, .. }) = drag.update(o, d, egui::Pos2::ZERO) else {
                panic!("a scale handle should scale");
            };
            assert!(factor > 0.0, "factor went non-positive at x = {x}: {factor}");
        }
    }

    /// Grabbing a handle off-centre must not teleport the body so the handle meets the cursor.
    #[test]
    fn grabbing_a_handle_off_centre_does_not_jump() {
        let frame = frame_at(Point3::origin());
        // The X handle's shaft ends at 8.4 mm; grab at 6, well off its anchor.
        let (origin, direction) = ray_at(Point3::new(6.0, 0.0, 0.0));
        let mut drag =
            Drag::begin(&frame, Handle::Axis(0), origin, direction, egui::Pos2::ZERO).unwrap();

        let Some(Motion::Translate(delta)) = drag.update(origin, direction, egui::Pos2::ZERO)
        else {
            panic!("an axis drag should translate");
        };
        assert!(delta.norm() < 1e-9, "a drag that has not moved reported {delta:?}");
    }

    /// A single point has no extent to turn or scale about, so those handles are absent rather
    /// than present and inert.
    #[test]
    fn a_single_point_offers_only_the_translation_handles() {
        let handles = Handles::single_point().all();
        assert_eq!(handles.len(), 6);
        assert!(handles.iter().all(|h| matches!(h, Handle::Axis(_) | Handle::Plane(_))));
    }

    /// The architectural constraint, asserted where it is easy to break: a body gets the
    /// uniform cube and no per-axis ones.
    #[test]
    fn a_body_is_offered_uniform_scale_and_never_per_axis() {
        let handles = Handles::body().all();
        assert!(handles.contains(&Handle::ScaleUniform));
        assert!(!handles.iter().any(|h| matches!(h, Handle::ScaleAxis(_))));
    }

    /// A cage selection with extent gets both: the per-axis cubes and, at the centre, the
    /// uniform one -- all thirteen -- and a click on the centre is the uniform cube's whatever
    /// else passes through the pivot.
    #[test]
    fn a_cage_selection_is_offered_uniform_scale_at_the_centre_as_well_as_per_axis() {
        let handles = Handles::extended().all();
        assert_eq!(handles.len(), 13);
        assert!(handles.contains(&Handle::ScaleUniform));
        assert!(handles.iter().any(|h| matches!(h, Handle::ScaleAxis(_))));
        let frame = frame_at(Point3::origin());
        let project = |p: Point3<f64>| Some(egui::pos2(400.0 + p.x as f32 * 10.0, 300.0 - p.y as f32 * 10.0));
        let eye = Point3::new(0.0, 0.0, 100.0);
        assert_eq!(hit_test(&frame, egui::pos2(403.0, 298.0), &project, eye), Some(Handle::ScaleUniform));
    }

    /// Looking straight down an axis leaves its drag plane undefined, and a normalised
    /// near-zero vector is how a handle starts jittering. Refused rather than accepted as
    /// noise.
    #[test]
    fn an_axis_seen_end_on_refuses_the_drag_rather_than_jittering() {
        let frame = frame_at(Point3::origin());
        let along_x = Vector3::new(1.0, 0.0, 0.0);
        assert!(drag_plane_normal(&frame, Handle::Axis(0), along_x).is_none());
        // A degree off is enough to be usable again.
        let nearly = Vector3::new(1.0, 0.02, 0.0).normalize();
        assert!(drag_plane_normal(&frame, Handle::Axis(0), nearly).is_some());
    }

    fn frame_with_arrows() -> Frame {
        Frame {
            extrude: Some(ExtrudeArrows {
                direction: Vector3::z(),
                tips: [Some(Point3::new(30.0, 0.0, 50.0)), Some(Point3::new(30.0, 0.0, -20.0))],
            }),
            ..frame_at(Point3::origin())
        }
    }

    /// **A depth arrow is dragged along the sweep and nowhere else, from any viewpoint.** The arrow
    /// stands on a face thirty millimetres from the pivot, so a drag plane through the *pivot* would
    /// be measured against a line beside it and the arrow would creep as the view turned. The plane
    /// goes through the arrow's own tip, and the cursor's ray, from an oblique eye, to a point on the
    /// arrow's line reads exactly that point.
    #[test]
    fn a_depth_arrow_follows_the_sweep_exactly_from_an_oblique_view() {
        let frame = frame_with_arrows();
        let eye = Point3::new(95.0, -200.0, 140.0);
        let through = |p: Point3<f64>| (eye, (p - eye).normalize());

        // The arrow's end is 5.5 mm above the tip, which is 50 up.
        let (o, d) = through(Point3::new(30.0, 0.0, 55.5));
        let mut drag = Drag::begin(&frame, Handle::Extrude(0), o, d, egui::Pos2::ZERO).unwrap();
        let (o, d) = through(Point3::new(30.0, 0.0, 65.5));
        let Some(Motion::Translate(delta)) = drag.update(o, d, egui::Pos2::ZERO) else {
            panic!("a depth arrow translates");
        };
        assert!((delta.z - 10.0).abs() < 1e-9, "{delta:?}");
        assert!(delta.x.abs() < 1e-9 && delta.y.abs() < 1e-9, "{delta:?}");
    }

    /// The second side's arrow points *against* the sweep, out of its own face, and both are offered
    /// only when the frame carries them.
    #[test]
    fn the_second_arrow_points_away_from_the_solid_and_a_plain_frame_has_none() {
        let frame = frame_with_arrows();
        let (tip, out) = frame.extrude_segment(1).unwrap();
        assert!(out.z < tip.z, "side two's arrow runs against the sweep: {tip:?} to {out:?}");
        let (tip, out) = frame.extrude_segment(0).unwrap();
        assert!(out.z > tip.z, "side one's runs with it");
        assert_eq!(frame.extrude_handles(), vec![Handle::Extrude(0), Handle::Extrude(1)]);
        assert!(frame_at(Point3::origin()).extrude_handles().is_empty());
        let one_sided = Frame {
            extrude: Some(ExtrudeArrows {
                direction: Vector3::z(),
                tips: [Some(Point3::new(0.0, 0.0, 5.0)), None],
            }),
            ..frame_at(Point3::origin())
        };
        assert_eq!(one_sided.extrude_handles(), vec![Handle::Extrude(0)]);
    }

    /// The arrow is aimed at on the screen: a cursor on its shaft finds it, and one well off it does
    /// not — through the same hit test the triad's handles go through.
    #[test]
    fn a_cursor_on_a_depth_arrow_finds_it_and_one_off_it_does_not() {
        // A reach of forty, so the arrow is twenty-two pixels long at a pixel to the millimetre —
        // longer than the foreshortening floor, as it is at the size the app draws it.
        let frame = Frame { reach: 40.0, ..frame_with_arrows() };
        // Looking straight along +Y: x is across the screen and z up it, one pixel to the mm.
        let project = |p: Point3<f64>| Some(egui::pos2(p.x as f32, -(p.z as f32)));
        let eye = Point3::new(30.0, -500.0, 50.0);
        // Side one's arrow runs from z = 50 to 72, so its shaft is on the screen at y = -50 to -72.
        let found = hit_test(&frame, egui::pos2(30.0, -60.0), &project, eye);
        assert_eq!(found, Some(Handle::Extrude(0)));
        assert_eq!(hit_test(&frame, egui::pos2(200.0, -60.0), &project, eye), None);
    }

    /// The gizmo must not shrink as the view widens: twice the world height on screen means
    /// twice the world reach, which is what holds its size in pixels fixed. Stated in terms of
    /// the height rather than the distance because those are the same thing under perspective
    /// and are not under orthographic, where distance does not change what is visible.
    #[test]
    fn the_reach_tracks_what_the_viewport_covers_so_the_screen_size_holds() {
        let near = Frame::reach_for(10.0);
        let far = Frame::reach_for(20.0);
        assert!((far / near - 2.0).abs() < 1e-9, "{near} then {far}");
    }
}
