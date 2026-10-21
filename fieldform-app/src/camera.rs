// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The viewport's orbit camera, and the tumble, dolly and snap gestures that move it.

use super::*;

/// Frozen state for an in-progress POI-anchored tumble drag. Rotation is recomputed from
/// these frozen start values plus accumulated totals every frame (rather than repeatedly
/// composing the previous frame's result), which avoids the slow drift that chaining many
/// small quaternion multiplications produces — same rationale as `_tumble` in
/// the reference camera.
struct TumbleState {
    poi: Point3<f32>,
    start_eye: Point3<f32>,
    start_forward: Vector3<f32>,
    /// Frozen once, at drag-start, rather than recomputed every frame from the live
    /// (possibly near-vertical) forward vector. Recomputing `forward.cross(world_up)` every
    /// frame degenerates whenever forward swings close to the poles, which is exactly what
    /// caused the "camera jumps/inverts" bug during a sustained vertical drag.
    start_right: Vector3<f32>,
    start_up: Vector3<f32>,
    target_distance: f32,
    total_yaw: f32,
    total_pitch: f32,
}

/// Which axis an in-progress dolly/twist drag has committed to, decided once per drag by
/// whichever of the first few pixels of motion is larger. Mirrors `_dolly`'s
/// `DOLLY_AXIS_LOCK_THRESHOLD` in the reference camera: without this, ordinary hand tremor
/// during a vertical-only zoom drag leaks tiny horizontal deltas into an unwanted twist
/// wobble (and vice versa).
enum DollyAxis {
    Zoom,
    Twist,
}

/// Frozen state for an in-progress RMB dolly/twist drag: vertical motion zooms toward/away
/// from the POI, horizontal motion twists (rolls) the camera around the world-space axis
/// running through the eye and the POI, so the POI is the fixed point of the roll instead
/// of the screen centre.
struct DollyState {
    poi: Point3<f32>,
    start_eye: Point3<f32>,
    start_forward: Vector3<f32>,
    start_up: Vector3<f32>,
    target_distance: f32,
    axis_lock: Option<DollyAxis>,
    cum_delta: egui::Vec2,
    total_zoom_factor: f32,
    total_twist: f32,
    /// Frozen once `axis_lock` is decided; meaningless before then.
    twist_axis: Vector3<f32>,
}

/// A stable "right" vector for `forward`, using `up_hint` as the reference — falling back
/// to a different reference axis if `forward` is (near-)parallel to `up_hint`, so this never
/// degenerates the way a bare `forward.cross(up_hint)` does near the poles.
fn safe_right(forward: Vector3<f32>, up_hint: Vector3<f32>) -> Vector3<f32> {
    let r = forward.cross(&up_hint);
    if r.norm_squared() > 1e-8 {
        r.normalize()
    } else {
        let alt = if forward.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::z()
        };
        forward.cross(&alt).normalize()
    }
}

/// Camera state is `eye` + `target` directly (not yaw/pitch/distance around a fixed
/// target), because POI-anchored tumbling needs to orbit around an arbitrary pivot that
/// generally isn't `target` — mirroring the reference camera's `view_rotation`/
/// `view_location`-from-eye model rather than a simple look-at-a-fixed-point camera.
pub(crate) struct OrbitCamera {
    pub(crate) eye: Point3<f32>,
    pub(crate) target: Point3<f32>,
    /// Persistent camera-space up vector. NOT re-derived from world up every frame — doing
    /// so silently forces the camera to auto-level and makes twist/roll invisible (that was
    /// the "RMB twist barely does anything" bug: the view matrix was always rebuilt against
    /// a fixed world-up reference, so no accumulated roll could ever show up). Tumbling yaws
    /// around world up still (turntable-style, deliberate), but both tumble and twist rotate
    /// this vector so the view matrix reflects whatever roll has actually accumulated.
    pub(crate) up: Vector3<f32>,
    /// The most recent point of interest set by a tumble click, reused by RMB dolly (which
    /// doesn't raycast under its own cursor — it dollies toward whatever the last tumble
    /// pivoted around, same as the reference camera's dolly mode).
    pub(crate) last_poi: Point3<f32>,
    tumble: Option<TumbleState>,
    dolly: Option<DollyState>,
    /// The pivot to show while panning.
    ///
    /// Panning does not rotate about a point, so unlike tumble and dolly it needs no state to
    /// do its job — but the marker should still be up while the camera is being moved, and
    /// where it is is still worth seeing. Held separately rather than folded into a shared
    /// "gesture" so that pan keeps having no effect on the pivot itself.
    pan_poi: Option<Point3<f32>>,
    /// Roughly how big the thing being looked at is, in millimetres. Set by [`Self::frame`].
    ///
    /// Zoom limits have to be *relative* to it. They were absolute — distance clamped to
    /// 0.5..50 mm — which is invisible while the only model is a 1 mm cube and catastrophic
    /// the moment one is a metre: framing puts the camera ~3400 mm back, so the first scroll
    /// snapped it to 50 mm, inside the model.
    pub(crate) scene_radius: f32,
    /// The middle of what is on screen, kept beside its radius so the clip planes can be
    /// fitted to the scene rather than to the pivot.
    pub(crate) scene_centre: Point3<f32>,
    /// A ViewCube snap in flight, if one is.
    pub(crate) snap: Option<SnapAnimation>,
    pub(crate) projection: Projection,
}

/// A camera swinging from one orientation to another.
///
/// Orientation is carried as a quaternion here and nowhere else in the camera, which stores an
/// eye, a target and an up vector day to day. That is deliberate: interpolating those three
/// directly does not produce a rotation — the eye would slide through the inside of the model
/// on a half-turn, and the up vector would shear rather than turn. Converting at this one
/// boundary and interpolating the quaternion is what makes the camera *swing around* the model
/// instead of the model appearing to teleport.
///
/// The target and the distance are untouched throughout. Only where you are standing changes.
pub(crate) struct SnapAnimation {
    from: UnitQuaternion<f32>,
    pub(crate) to: UnitQuaternion<f32>,
    /// Seconds elapsed, against [`SNAP_SECONDS`].
    elapsed: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        // Z-up world (CAD convention, matching the ViewCube): pitch is measured from
        // +Z, and the horizontal ring swept by yaw lies in the XY plane.
        //
        // Framed for a default-sized body at the origin, rather than at a distance of 3 — a
        // unitless number from before geometry was canonically millimetres, which put the eye
        // 3 mm from the target and inside anything real. Startup used to work around it by
        // framing the starting cage, so the bug survived anywhere that reset the camera
        // without geometry to frame: File > New left the eye 3 mm out, and the first body
        // created afterwards filled the screen.
        let target = Point3::origin();
        let yaw = 0.6_f32;
        let pitch = 1.1_f32;
        let scene_radius = DEFAULT_BODY_MM as f32 * 0.5;
        let distance = (scene_radius / (FOVY * 0.5).sin()) * 1.3;
        let eye = target
            + Vector3::new(
                distance * pitch.sin() * yaw.cos(),
                distance * pitch.sin() * yaw.sin(),
                distance * pitch.cos(),
            );
        Self {
            eye,
            target,
            up: Vector3::z(),
            last_poi: target,
            pan_poi: None,
            scene_radius,
            scene_centre: Point3::origin(),
            tumble: None,
            dolly: None,
            snap: None,
            projection: Projection::Perspective,
        }
    }
}

impl OrbitCamera {
    pub(crate) fn forward(&self) -> Vector3<f32> {
        (self.target - self.eye).normalize()
    }

    fn distance(&self) -> f32 {
        (self.target - self.eye).norm()
    }

    /// Clip planes fitted around the scene — see [`CLIP_MARGIN`].
    ///
    /// The far plane clears the back of the scene; the near plane sits in front of it, or as
    /// close to the eye as precision allows once the eye is inside. Both are derived from
    /// where the geometry *is*, so no amount of zooming, panning or pivoting on a far corner
    /// can put the model outside them.
    fn clip_planes(&self) -> (f32, f32) {
        let radius = self.scene_radius.max(1e-6);
        let slack = radius * CLIP_MARGIN;
        let to_centre = (self.eye - self.scene_centre).norm();

        let far = (to_centre + radius + slack).max(radius * 1e-3);
        // Half the gap to the front of the scene, so geometry nearer than expected — a body
        // added off to one side, a pan that brings the camera in close — still draws.
        let near = ((to_centre - radius - slack) * 0.5)
            // Never so small next to `far` that the depth range is unusable, and never zero.
            .max(far * 1e-5)
            .max(1e-6);
        (near, far)
    }

    /// View-projection in OpenGL clip-space convention (z in [-1, 1]), used for picking-ray
    /// unprojection. Kept separate from [`Self::mvp`], which additionally remaps z to wgpu's
    /// [0, 1] clip space for actual rendering.
    fn view_proj_gl(&self, aspect: f32) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(&self.eye, &self.target, &self.up);
        let (near, far) = self.clip_planes();
        let proj = match self.projection {
            Projection::Perspective => Matrix4::new_perspective(aspect, FOVY, near, far),
            Projection::Orthographic => {
                let half_h = self.ortho_half_height();
                let half_w = half_h * aspect;
                // The near plane goes *behind* the eye, which a perspective projection cannot
                // do and an orthographic one has no reason not to. Its rays are parallel, so
                // there is no apex to stay in front of, and clipping at the eye would slice
                // away the near half of anything the camera has been pushed inside of.
                Matrix4::new_orthographic(-half_w, half_w, -half_h, half_h, -far, far)
            }
        };
        proj * view
    }

    /// Half the world height the viewport covers at `at`.
    ///
    /// The one question anything sizing itself against the screen actually wants to ask, and
    /// the only place the two projections differ in answering it: under perspective what is
    /// visible widens with distance, and under orthographic it does not.
    pub(crate) fn view_half_height(&self, at: Point3<f32>) -> f32 {
        match self.projection {
            Projection::Perspective => {
                ((at - self.eye).norm().max(1e-6)) * (FOVY * 0.5).tan()
            }
            Projection::Orthographic => self.ortho_half_height(),
        }
    }

    /// Half the world height the viewport covers, orthographically.
    ///
    /// Derived from the distance and the same field of view the perspective mode uses, so that
    /// toggling between the two leaves the model the same size on screen. A fixed extent would
    /// make the switch a jump-cut in zoom as well as in projection, and only one of those was
    /// asked for.
    fn ortho_half_height(&self) -> f32 {
        (self.distance() * (FOVY * 0.5).tan()).max(1e-6)
    }

    /// Full model-view-projection matrix for rendering (model is identity: the cage sits at
    /// the origin).
    pub(crate) fn mvp(&self, aspect: f32) -> Matrix4<f32> {
        render::opengl_to_wgpu_projection() * self.view_proj_gl(aspect)
    }

    /// Builds a world-space picking ray through a screen-space point (in the viewport
    /// rect's local pixel coordinates, Y-down as egui reports it).
    pub(crate) fn pick_ray(&self, aspect: f32, local_pos: egui::Pos2, rect: egui::Rect) -> (Point3<f32>, Vector3<f32>) {
        let ndc_x = (local_pos.x - rect.min.x) / rect.width() * 2.0 - 1.0;
        // Flip Y: screen space increases downward, NDC increases upward.
        let ndc_y = 1.0 - (local_pos.y - rect.min.y) / rect.height() * 2.0;

        let inv = self
            .view_proj_gl(aspect)
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        let unproject = |z: f32| {
            let p = inv * Vector4::new(ndc_x, ndc_y, z, 1.0);
            Point3::new(p.x / p.w, p.y / p.w, p.z / p.w)
        };
        let far_world = unproject(1.0);
        match self.projection {
            Projection::Perspective => (self.eye, (far_world - self.eye).normalize()),
            // Every ray leaves a different point, so the origin has to be unprojected too —
            // taking `eye` here would aim every pick at the middle of the screen, which reads
            // as picking working only in the centre of the viewport.
            Projection::Orthographic => {
                let near_world = unproject(-1.0);
                (near_world, (far_world - near_world).normalize())
            }
        }
    }

    /// Projects a world point to screen-space pixel coordinates within `rect`, for drawing
    /// the POI marker. `None` if the point is behind the camera.
    pub(crate) fn project_to_screen(
        &self,
        aspect: f32,
        rect: egui::Rect,
        world: Point3<f32>,
    ) -> Option<egui::Pos2> {
        let clip = self.view_proj_gl(aspect) * Vector4::new(world.x, world.y, world.z, 1.0);
        if clip.w <= 1e-4 {
            return None;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let x = rect.min.x + (ndc_x * 0.5 + 0.5) * rect.width();
        // Flip Y back: NDC increases upward, screen space increases downward.
        let y = rect.min.y + (1.0 - (ndc_y * 0.5 + 0.5)) * rect.height();
        Some(egui::pos2(x, y))
    }

    /// The pivot currently being visualized, if a tumble or dolly is in progress.
    pub(crate) fn active_poi(&self) -> Option<Point3<f32>> {
        self.tumble
            .as_ref()
            .map(|t| t.poi)
            .or_else(|| self.dolly.as_ref().map(|d| d.poi))
            .or(self.pan_poi)
    }

    /// Raises the pivot marker for the duration of a pan.
    pub(crate) fn begin_pan(&mut self) {
        self.pan_poi = Some(self.last_poi);
    }

    pub(crate) fn end_pan(&mut self) {
        self.pan_poi = None;
    }

    /// Begins a POI-anchored tumble: `poi` is the raycast hit (or fallback) that becomes
    /// the fixed pivot of rotation for the rest of the drag, and is remembered as
    /// `last_poi` for a subsequent RMB dolly.
    pub(crate) fn begin_tumble(&mut self, poi: Point3<f32>) {
        self.last_poi = poi;
        let start_forward = self.forward();
        self.tumble = Some(TumbleState {
            poi,
            start_eye: self.eye,
            start_forward,
            start_right: safe_right(start_forward, self.up),
            start_up: self.up,
            target_distance: self.distance(),
            total_yaw: 0.0,
            total_pitch: 0.0,
        });
    }

    /// Accumulates drag motion into the active tumble and re-derives `eye`/`target`/`up`
    /// from the frozen drag-start state, matching the reference camera's tumble. The
    /// pitch axis is the frozen `start_right` rotated by yaw only — not recomputed from the
    /// live forward vector every frame — so it stays well-defined even as the camera swings
    /// through near-vertical during a long drag.
    pub(crate) fn tumble_drag(&mut self, delta: egui::Vec2) {
        const SENSITIVITY: f32 = 0.01;
        let Some(t) = &mut self.tumble else { return };

        // Both axes inverted relative to raw cursor delta (matches the "invert tumble"
        // defaults in the reference camera's preferences).
        t.total_yaw -= delta.x * SENSITIVITY;
        t.total_pitch -= delta.y * SENSITIVITY;

        let yaw_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), t.total_yaw);
        let right = Unit::new_normalize(yaw_q * t.start_right);
        let pitch_q = UnitQuaternion::from_axis_angle(&right, t.total_pitch);
        let rot = pitch_q * yaw_q;

        self.eye = t.poi + rot * (t.start_eye - t.poi);
        self.target = self.eye + (rot * t.start_forward) * t.target_distance;
        self.up = rot * t.start_up;
    }

    pub(crate) fn end_tumble(&mut self) {
        self.tumble = None;
    }

    /// Begins an RMB dolly/twist drag, pivoting on the last tumble's POI (not a fresh
    /// raycast — see `last_poi`).
    pub(crate) fn begin_dolly(&mut self) {
        self.dolly = Some(DollyState {
            poi: self.last_poi,
            start_eye: self.eye,
            start_forward: self.forward(),
            start_up: self.up,
            target_distance: self.distance(),
            axis_lock: None,
            cum_delta: egui::Vec2::ZERO,
            total_zoom_factor: 1.0,
            total_twist: 0.0,
            twist_axis: Vector3::zeros(),
        });
    }

    /// Continues an RMB dolly/twist drag, matching the reference camera's dolly:
    /// commits to zoom-or-twist once the first few pixels of motion decide it, then applies
    /// only that axis for the rest of the drag.
    pub(crate) fn dolly_drag(&mut self, delta: egui::Vec2) {
        const AXIS_LOCK_THRESHOLD: f32 = 4.0;
        const ZOOM_SENSITIVITY: f32 = 0.005;
        const TWIST_SENSITIVITY: f32 = 0.01;

        let Some(d) = &mut self.dolly else { return };
        d.cum_delta += delta;

        if d.axis_lock.is_none() {
            if d.cum_delta.length() < AXIS_LOCK_THRESHOLD {
                return;
            }
            d.axis_lock = Some(if d.cum_delta.x.abs() > d.cum_delta.y.abs() {
                DollyAxis::Twist
            } else {
                DollyAxis::Zoom
            });
            // Roll around the axis running through the eye and the POI (not the camera's
            // own view axis) so the POI is the fixed point of the twist.
            let axis = d.poi - d.start_eye;
            d.twist_axis = if axis.norm_squared() < 1e-10 {
                d.start_forward
            } else {
                axis.normalize()
            };
        }

        match d.axis_lock {
            Some(DollyAxis::Zoom) => {
                d.total_zoom_factor =
                    (d.total_zoom_factor * (1.0 + delta.y * ZOOM_SENSITIVITY)).max(0.02);
                self.eye = d.poi + (d.start_eye - d.poi) * d.total_zoom_factor;
                // Under perspective the image comes from where the eye is, so sliding it
                // toward the POI *is* the zoom and the target rides along at a fixed distance.
                // Under orthographic that does nothing at all: the rays are parallel and the
                // near plane sits behind the eye, so moving along the view axis changes
                // neither what is in frame nor what is clipped. What sets an orthographic zoom
                // is `ortho_half_height`, which reads the eye-to-target distance — the one
                // quantity the line above deliberately holds constant. So the target has to
                // close on the eye by the same factor, which also walks it toward the POI as
                // you zoom in, making the POI the fixed point under both projections.
                let distance = match self.projection {
                    Projection::Perspective => d.target_distance,
                    Projection::Orthographic => d.target_distance * d.total_zoom_factor,
                };
                self.target = self.eye + d.start_forward * distance;
            }
            Some(DollyAxis::Twist) => {
                d.total_twist += delta.x * TWIST_SENSITIVITY;
                let axis = Unit::new_normalize(d.twist_axis);
                let rot = UnitQuaternion::from_axis_angle(&axis, d.total_twist);
                // `start_eye` lies exactly on the twist axis by construction, so rotating it
                // around that axis leaves it unchanged; only the look direction swings.
                self.eye = d.start_eye;
                let new_forward = rot * d.start_forward;
                self.target = self.eye + new_forward * d.target_distance;
                // Also rotate `up`, not just `forward`/`target`: the view matrix is built
                // from `up` directly now (see `OrbitCamera::up`'s docs), so without this the
                // twist barely appeared to do anything — forward alone barely changes when
                // the twist axis is close to the view direction (the usual case, since the
                // POI is normally roughly where you were already looking), and the old
                // world-up-locked view matrix silently discarded any roll anyway.
                self.up = rot * d.start_up;
            }
            None => {} // still within the axis-lock threshold; no motion yet
        }
    }

    pub(crate) fn end_dolly(&mut self) {
        self.dolly = None;
    }

    /// MMB drag: pan `eye` and `target` together in the camera's local right/up plane,
    /// scaled by distance so panning feels proportional whether zoomed in or out.
    pub(crate) fn pan(&mut self, delta: egui::Vec2) {
        const SPEED: f32 = 0.0015;
        let forward = self.forward();
        let right = safe_right(forward, self.up);
        let up = right.cross(&forward).normalize();
        let amount = self.distance() * SPEED;
        let offset = right * (-delta.x * amount) + up * (delta.y * amount);
        self.eye += offset;
        self.target += offset;
    }

    pub(crate) fn zoom(&mut self, scroll_y: f32) {
        self.zoom_by(1.0 - scroll_y * 0.001);
    }

    /// Closer by `times`, about the target, as a script asks for a detail the wheel would be
    /// turned toward: `snap zoom=3`.
    pub(crate) fn zoom_in(&mut self, times: f32) {
        self.zoom_by(1.0 / times.max(1e-3));
    }

    fn zoom_by(&mut self, factor: f32) {
        // Relative to the scene, not to millimetres: close enough to inspect a detail, far
        // enough to lose the model off the far clip plane, whatever size it is.
        let min = self.scene_radius * 0.01;
        let max = self.scene_radius * 100.0;
        let distance = (self.distance() * factor).clamp(min, max);
        self.eye = self.target - self.forward() * distance;
    }

    /// The camera's orientation, as a rotation whose Z points from the target toward the eye.
    fn orientation(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::face_towards(&(self.eye - self.target), &self.up)
    }

    /// Which way is up for a view along `direction`, keeping the roll the camera already has
    /// where the fixed rule would throw it away.
    ///
    /// **Straight down and straight up are the two views with no natural up**, and the fixed
    /// answer -- Y -- is what made Left-then-Top land the model the other way round. Tipped up
    /// out of a Left view, the camera's up has already begun swinging toward where it is
    /// looking, and the orbit that would reach Top smoothly keeps swinging it there; snapping
    /// to Y instead turned the model under the person's hands. So the up is taken from the
    /// camera: its current up projected onto the plane of the view, or -- when that is nothing,
    /// as it is from any square side view -- its forward, which is where the up ends after
    /// orbiting over the top, and its opposite after orbiting under the bottom.
    ///
    /// **Snapped to the nearest axis**, which is what keeps repeated Top clicks from turning the
    /// view a little each time. The drift `CubeRegion::up` was fixed against comes from a *raw*
    /// projection feeding the next click; a snapped one is its own fixed point, so the second
    /// click asks for exactly what the first produced. Every other direction keeps the hint it
    /// was given: Z up is the right answer for a side view whatever the camera was doing.
    fn up_keeping_roll(&self, direction: Vector3<f32>, up_hint: Vector3<f32>) -> Vector3<f32> {
        if direction.x.abs() > 1e-6 || direction.y.abs() > 1e-6 {
            return up_hint;
        }
        let flat = |v: Vector3<f32>| Vector3::new(v.x, v.y, 0.0);
        let mut want = flat(self.up);
        if want.norm() < 1e-3 {
            // `direction` points from the target toward the eye, so +Z is the top view.
            want = flat(self.forward()) * direction.z.signum();
        }
        if want.norm() < 1e-3 {
            return up_hint;
        }
        if want.x.abs() >= want.y.abs() {
            Vector3::new(want.x.signum(), 0.0, 0.0)
        } else {
            Vector3::new(0.0, want.y.signum(), 0.0)
        }
    }

    /// Begins swinging to a ViewCube region's view, keeping the target and distance exactly.
    ///
    /// A snap already in flight is *replaced*, starting from wherever the camera has got to
    /// rather than from where the previous one began. Queueing them would leave the camera
    /// finishing a journey nobody is waiting for any more, and cutting to the new target
    /// instantly would throw away the reason for animating at all.
    pub(crate) fn begin_snap(&mut self, region: CubeRegion) {
        self.begin_snap_toward(region.direction(), region.up());
    }

    /// The same, toward any direction at all.
    ///
    /// The ViewCube's twenty-six regions are not the only things worth looking at squarely: a
    /// drawing plane taken from a plane block faces wherever that block faces, and setting one
    /// should swing the camera to look down it exactly as clicking a cube face does. One
    /// animation path rather than two, so the two cannot come to feel different.
    pub(crate) fn begin_snap_toward(&mut self, direction: Vector3<f32>, up_hint: Vector3<f32>) {
        // A direction with no length names no view; leave the camera where it is.
        let Some(direction) = direction.try_normalize(1e-6) else { return };
        let up_hint = self.up_keeping_roll(direction, up_hint);
        let to = UnitQuaternion::face_towards(&direction, &up_hint);
        self.snap = Some(SnapAnimation {
            from: self.orientation(),
            to,
            elapsed: 0.0,
        });
        // A snap and a hand-driven gesture cannot both be moving the camera.
        self.tumble = None;
        self.dolly = None;
    }

    /// Advances a snap by one frame. Returns whether one is still running, so the caller knows
    /// to ask for another frame.
    pub(crate) fn tick_snap(&mut self, dt: f32) -> bool {
        let Some(snap) = &mut self.snap else { return false };
        snap.elapsed += dt;
        let t = (snap.elapsed / SNAP_SECONDS).clamp(0.0, 1.0);
        // Ease in and out. Linear reads as mechanical — the camera starts and stops abruptly,
        // which is the difference between a swing and a slide.
        let eased = if t < 0.5 {
            4.0 * t * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
        };
        let (from, to) = (snap.from, snap.to);
        let done = t >= 1.0;

        let orientation = from.slerp(&to, eased);
        let distance = self.distance();
        self.eye = self.target + (orientation * Vector3::z()) * distance;
        self.up = orientation * Vector3::y();

        if done {
            self.snap = None;
        }
        !done
    }

    /// Frames whatever the camera was last told is on screen — what the Home button reaches
    /// for, so Home means the same thing on a bracket and on an airframe.
    pub(crate) fn frame_scene(&mut self) {
        let (centre, radius) = (self.scene_centre, self.scene_radius);
        self.frame(centre, radius);
    }

    /// Tells the camera what it is looking at, which is what the clip planes are fitted to.
    ///
    /// Called whenever the document or what is visible changes, rather than per frame: it
    /// walks the bodies, and the answer only moves when they do.
    pub(crate) fn set_scene(&mut self, centre: Point3<f32>, radius: f32) {
        self.scene_centre = centre;
        self.scene_radius = radius.max(1e-6);
    }

    /// Re-centers on `center`, keeping the current view direction and up vector but setting
    /// the distance so a sphere of `radius` comfortably fills the frame. Bound to `/`.
    pub(crate) fn frame(&mut self, center: Point3<f32>, radius: f32) {
        let view_dir = -self.forward(); // unit vector from target toward eye, preserved
        const PADDING: f32 = 1.3;
        let distance = (radius / (FOVY * 0.5).sin()).max(0.5) * PADDING;
        self.set_scene(center, radius);
        self.target = center;
        self.eye = center + view_dir * distance;
        // Re-anchor the pivot too, so a later click that misses the model falls back to
        // something near what's on screen rather than a stale point.
        self.last_poi = center;
        self.tumble = None;
        self.dolly = None;
    }
}
