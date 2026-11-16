// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The viewport: what a press, a drag and a hover in it do, and what is drawn over the picture.

use crate::*;

impl App {
    /// The central panel. Input is read before anything is drawn, so a frame shows what its own
    /// press did rather than the frame before's.
    pub(crate) fn viewport(&mut self, ui: &mut egui::Ui) {
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

        let response = &response;
        self.viewport_input(ui, rect, response, aspect);
        self.draw_scene(ui, rect, response, aspect);
        self.draw_overlays(ui, rect, response, aspect);
        self.draw_view_cube(ui, rect);
    }

    /// Presses, drags, releases and hovers: selection, the manipulator, curve handles and the
    /// camera, in the order that decides which of them a press belongs to.
    fn viewport_input(&mut self, ui: &mut egui::Ui, rect: egui::Rect, response: &egui::Response, aspect: f32) {
        // Where the cursor is over the viewport, for a paste that lands under it
        // (`profile-group-spec.md` §4.4). `None` whenever a menu, the console or a panel
        // is under the pointer instead, which is what makes those paste in place.
        self.viewport_hover = response
            .hover_pos()
            .filter(|p| rect.contains(*p))
            .map(|p| (aspect, rect, p));
        // **A right-click opens the marking menu** (`interaction-spec.md` §7) -- and on a join,
        // where two curve ends meet, the join's own ring of G0, G1 and G2, whatever is
        // selected: the join is what was aimed at (the owner: "when you right-click on top of
        // a curve intersection you should get a marking menu that allows you to set
        // tangency"). It replaced a list that only a held loop's corners offered, so an open
        // chain's corners have it too. Paste here is the empty table's North-West slot.
        if response.secondary_clicked() {
            self.viewport_menu_at = response.interact_pointer_pos();
            if let Some(at) = self.viewport_menu_at {
                match self.join_at(aspect, rect, at) {
                    Some(join) => self.open_join_menu(at, join.ends),
                    None => self.open_marking_menu(at),
                }
            }
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
        let placing = self.curve_ui.pending.is_some();

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
        // **A surface point's dot takes the press the same way**, behind a control point and
        // ahead of the manipulator: pressing it and dragging slides the point over its surface,
        // which is the one gesture the point is for. Its own manipulator still works for a
        // move along one axis; the dot is how it is moved by hand.
        if pressed_control_point.is_none()
            && !press_over_cube
            && !nav_mods
            && !placing
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
            && let Some(pos) = response.interact_pointer_pos()
            && let Some(point) = self.surface_point_at(aspect, rect, pos)
        {
            self.presses.surface_point = true;
            self.grab_surface_point(point);
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
            && !self.presses.surface_point
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
                self.presses.gizmo = true;
                self.begin_gizmo(&frame, handle, aspect, rect, pos);
            }
        }

        // **A press on what an armed tool drags -- the knife's active point, the edge Insert
        // Edge slides a ring along -- is the tool's, not the camera's** (`cage-knife-spec.md`
        // §3): decided at the press, for the same reason the gizmo's own grab is -- by the time
        // egui calls it a drag the pointer has moved on.
        if !press_over_cube
            && !self.presses.gizmo
            && self.cage_tool.is_some()
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
            && let Some(pos) = response.interact_pointer_pos()
            && self.cage_tool_press(aspect, rect, pos)
        {
            self.presses.cage_tool = true;
        }

        if !press_over_cube
            && !nav_mods
            && !placing
            && editable
            && !self.presses.gizmo
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
        {
            self.presses.selection = response.interact_pointer_pos().is_some_and(|pos| {
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
            self.presses.control_point = true;
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
            && !self.presses.gizmo
            && !boxing
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
            && let Some(pos) = response.interact_pointer_pos()
        {
            pressed_curve = self.curve_at(aspect, rect, pos);
        }
        // **A plane's outline takes the press behind a curve and ahead of the bodies**, for the
        // reason a curve does: reference geometry drawn over the model, one pixel wide. Its
        // area is the body pick's to decline first (see the click below).
        if pressed_curve.is_none()
            && pressed_control_point.is_none()
            && !press_over_cube
            && !nav_mods
            && !placing
            && !self.presses.gizmo
            && !self.presses.surface_point
            && !boxing
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
            && let Some(pos) = response.interact_pointer_pos()
            && let Some(plane) = self.plane_at(aspect, rect, pos, false)
        {
            self.presses.curve = true;
            self.clear_point_selection();
            self.selected_curve = None;
            self.select_body(Some(plane));
            self.say(format!("Selected {}", self.body_label(plane)));
        }
        if let Some(id) = pressed_curve {
            self.presses.curve = true;
            let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
            if ui.input(|i| i.modifiers.shift) && self.selected_curve.is_some_and(|s| s != id) {
                // **Shift+click on another curve adds every point of it**, so several
                // curves move as one piece under the manipulator. A selected curve with no
                // point picked stands for all of its points, which is what it moves.
                let mut picked = if self.selected_points.is_empty() && self.curve_ui.other_points.is_empty() {
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
                self.notebook.curve_picked = Some(id);
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
                self.curve_ui.pending_narrow = Some((curve, index));
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
            && !self.presses.gizmo
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
            && let Some(pos) = response.interact_pointer_pos()
        {
            self.curve_ui.marquee = Some((pos, ui.input(|i| i.modifiers.shift)));
            self.presses.marquee = true;
        }

        let on_gizmo = self.presses.gizmo
            || self.gizmo_drag.is_some()
            || self.presses.control_point
            || self.presses.surface_point;
        let moving = self.presses.selection
            || self.move_state.is_some()
            || on_gizmo
            || self.presses.marquee
            || self.presses.cage_tool;

        // What the tool took the press for, dragged: the knife's point slides along its own
        // edge, Insert Edge's ring along the edge pressed, rather than the camera tumbling.
        if self.presses.cage_tool
            && response.dragged_by(egui::PointerButton::Primary)
            && let Some(pos) = response.interact_pointer_pos()
        {
            self.cage_tool_drag(aspect, rect, pos);
        }

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
        if self.curve_ui.pending.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.cancel_curve_placement();
        }

        // Field mode has no cage to aim at, so a plain click there means the body under
        // the cursor instead — the same gesture, aimed at whatever the mode is drawing,
        // which is the rule the whole picking path already follows.
        if !press_over_cube
            && !on_gizmo
            && !placing
            && !self.presses.curve
            && !self.presses.marquee
            && !self.view.mode.cage_is_editable()
            && response.clicked_by(egui::PointerButton::Primary)
        {
            if let Some(pos) = response.interact_pointer_pos() {
                let (origin, dir) = self.camera.pick_ray(aspect, pos, rect);
                self.clear_point_selection();
                self.selected_curve = None;
                // `select_body_under` already clears the body when the trace hits nothing,
                // which is the same rule: a click on empty space selects empty space.
                self.select_body_under(origin, dir);
                // A click on no body, inside a plane's square, is a click on the plane.
                if self.selected_body.is_none()
                    && let Some(pos) = response.interact_pointer_pos()
                    && let Some(plane) = self.plane_at(aspect, rect, pos, true)
                {
                    self.select_body(Some(plane));
                    self.say(format!("Selected {}", self.body_label(plane)));
                }
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
        // **A double-click on an edge selects its loop even where the manipulator sits over
        // it.** A selected edge's handles hide the edge from the hover, so the second click
        // of a double-click landed on a handle and did nothing (the owner, 2026-10-24). The
        // cage is picked here through the handles, and the drag the first click began on a
        // handle is let go: nothing has moved.
        if !press_over_cube
            && editable
            && on_gizmo
            && !placing
            && !self.presses.marquee
            && response.double_clicked_by(egui::PointerButton::Primary)
            && let Some(pos) = response.interact_pointer_pos()
        {
            let under = pick(self.control(), &self.control_placement(), &self.cage_mesh, &self.camera, aspect, rect, pos);
            if matches!(under, Selection::Edge(..)) {
                self.cancel_gizmo();
                self.presses.gizmo = false;
                let op = if ui.input(|i| i.modifiers.alt) { "select.ring" } else { "select.loop" };
                if !self.selection.contains(under) {
                    self.selection.toggle(under);
                }
                self.select_along(op);
            }
        }
        if !press_over_cube && editable && !on_gizmo {
            // **A double-click on an edge selects its loop, and with Alt its ring**
            // (instructions §0.4) -- the same gesture every quad modeller has.
            if !placing
                && !self.presses.marquee
                && response.double_clicked_by(egui::PointerButton::Primary)
                && matches!(self.hover, Selection::Edge(..))
            {
                let op = if ui.input(|i| i.modifiers.alt) { "select.ring" } else { "select.loop" };
                if !self.selection.contains(self.hover) {
                    self.selection.toggle(self.hover);
                }
                self.select_along(op);
            } else if !placing
                && !self.presses.marquee
                && response.clicked_by(egui::PointerButton::Primary)
                && self.cage_tool.is_some()
                && self.cage_tool_click(ui.input(|i| i.modifiers.shift), response.interact_pointer_pos(), aspect, rect)
            {
                // The click was the tool's: its line has run, and nothing is picked by it.
            } else if !placing
                && !self.presses.marquee
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
                    // Picking the cage takes the manipulator from the section, as picking
                    // a body does: the section stays on, only its claim on the handles goes
                    // (the owner, 2026-10-24: a face picked with the section on still moved
                    // the section).
                    if !matches!(under, Selection::None) {
                        self.section_selected = None;
                    }
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
            if self.presses.selection {
                if response.drag_started_by(egui::PointerButton::Primary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.begin_move(aspect, rect, pos);
                    }
                }
                if response.dragged_by(egui::PointerButton::Primary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let ends = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
                        let grid = ui.input(|i| i.modifiers.shift);
                        self.continue_move(aspect, rect, pos, grid, ends);
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
        if self.presses.control_point {
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
        // **A surface point's dot, dragged, slides the point under the cursor** -- over its own
        // body only, and it waits where it is while the cursor is over anything else. Escape
        // puts it back.
        if self.presses.surface_point {
            if response.dragged_by(egui::PointerButton::Primary)
                && let Some(pos) = response.interact_pointer_pos()
            {
                self.continue_surface_point_grab(aspect, rect, pos);
            }
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.cancel_surface_point_grab();
                self.presses.surface_point = false;
            }
        }
        if self.presses.marquee
            && let Some((from, _)) = self.curve_ui.marquee
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
        if self.curve_ui.point_drag.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            // Abandoned: put every point back where it started and forget the gesture --
            // the blocks too, or a held end would be put back and then settled onto where
            // its block was dragged, which is what `cancel_gizmo` already does.
            if let Some(state) = self.curve_ui.point_drag.take() {
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
        // `presses.gizmo` is only set when the press landed on a handle, and the cage's
        // `presses.selection` is not even evaluated when it did.
        if self.presses.gizmo
            && response.dragged_by(egui::PointerButton::Primary)
            && let Some(pos) = response.interact_pointer_pos()
        {
            // Ctrl rounds a depth to whole millimetres, the way it rounds a curve's points.
            let snap = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
            let grid = ui.input(|i| i.modifiers.shift);
            self.continue_gizmo(aspect, rect, pos, snap, grid);
        }
        // Escape abandons the gesture. Checked before the release below, so a drag
        // cancelled with the button still down is not then committed by letting go.
        if self.gizmo_drag.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.cancel_gizmo();
            self.presses.gizmo = false;
        }

        // The press is over once the button is up, whatever it turned out to be.
        //
        // The manipulator commits here rather than on `drag_stopped`, because a press that
        // never moved far enough to count as a drag still has to be closed out — otherwise
        // a click on a handle leaves a drag live until the next one begins.
        if ui.input(|i| i.pointer.button_released(egui::PointerButton::Primary)) {
            // A press on a picked point that never became a drag was a click on it.
            if let Some((curve, index)) = self.curve_ui.pending_narrow.take()
                && self.curve_ui.point_drag.is_none()
            {
                self.pick_only(curve, index);
            }
            if self.curve_ui.point_drag.is_some() {
                self.end_point_drag();
            }
            if self.gizmo_drag.is_some() {
                self.end_gizmo();
            }
            if self.presses.surface_point {
                self.end_surface_point_grab();
                self.presses.surface_point = false;
            }
            if let Some((from, adding)) = self.curve_ui.marquee.take()
                && let Some(to) = response.interact_pointer_pos().or(response.hover_pos())
            {
                self.finish_marquee(aspect, rect, from, to, adding);
            }
            self.presses.gizmo = false;
            if self.presses.cage_tool {
                self.cage_tool_release();
            }
            self.presses.cage_tool = false;
            self.presses.control_point = false;
            self.presses.curve = false;
            self.presses.selection = false;
            self.presses.marquee = false;
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
        // An armed cage tool reads the hover: where along the edge the pointer is, and what
        // a click there would make (`app/cage_tools.rs`).
        if self.cage_tool.is_some() {
            let (snap_half, ctrl, alt) =
                ui.input(|i| (i.modifiers.shift, i.modifiers.ctrl || i.modifiers.command, i.modifiers.alt));
            let pointer = response.hover_pos().or(if self.presses.cage_tool { response.interact_pointer_pos() } else { None });
            self.cage_tool_hover(aspect, rect, pointer, snap_half, ctrl, alt);
        }
    }

    /// The model itself: imported geometry, the traced field, the control cage and the grid.
    fn draw_scene(&mut self, ui: &mut egui::Ui, rect: egui::Rect, response: &egui::Response, aspect: f32) {
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

        match self.view.mode {
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

        // What is wrong with the cage, in every mode: a red mark is about the cage whether
        // or not the mode draws it, and the ghost is what stands where the solid was.
        // The cage's edges, wherever the cage is drawn: a cage of flat faces shows no seam
        // between two coplanar quads, and how dense the cage is cannot be read without them.
        if self.control_editable() && self.control == ControlMesh::Cage {
            draw_cage_edges(ui.painter(), &self.camera, aspect, rect, self.cage(), &self.control_placement());
        }
        if self.control == ControlMesh::Cage {
            self.draw_cage_faults(ui.painter(), aspect, rect);
            self.draw_cage_tool_preview(ui.painter(), aspect, rect);
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
    }

    /// What is drawn over the model: curves and their handles, points, planes, the snap mark and
    /// the manipulator.
    fn draw_overlays(&mut self, ui: &mut egui::Ui, rect: egui::Rect, response: &egui::Response, aspect: f32) {
        // Curves draw over the surface deliberately: when subtracting, the curve is
        // inside the solid it is cutting, and hiding it would hide the thing being aimed.
        let visible_curves: Vec<(CurveId, Vec<Point3<f32>>)> = self
            .curve_ui.display
            .iter()
            .filter(|(id, _)| !self.curve_ui.hidden.contains(id))
            .cloned()
            .collect();
        // Every curve of a selected profile row, not only the one curve a click picks:
        // selecting the row selects its curves, and the viewport says so.
        let highlighted = self.highlighted_curves();
        // The axes whose own note says they could not be re-derived — the same map the panel
        // reads, so the two cannot say different things about one line.
        let stale: Vec<CurveId> = self.curve_ui.surface_axis_notes.keys().copied().collect();
        draw_curves(
            ui.painter(),
            &self.camera,
            aspect,
            rect,
            &visible_curves,
            &highlighted,
            &stale,
        );
        // The combs over their curves, so a tooth is read against the line it measures.
        let combs = self.comb_readings();
        comb_block::draw_combs(ui.painter(), &self.camera, aspect, rect, &combs);
        self.draw_surface_points(ui.painter(), aspect, rect);
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
        if self.curve_ui.pending.is_some() {
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
        for (id, origin, facing) in self.plane_frames() {
            let here = Some(id) == selected_plane;
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
            // middle of the square, and the plane's own X and Y are its sides, so the square
            // is drawn the way the plane is turned rather than the way an arbitrary choice of
            // "some direction not along the normal" happens to fall. That choice is what made
            // a roll invisible, and it also made the square jump as a plane was tilted past
            // the axis it was picked against. One geometry with the picking: `plane_square`.
            let corners = self.plane_square(origin, facing);
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
    }

    /// The view cube, and the repaints a settling picture and a turning camera ask for.
    fn draw_view_cube(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
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
        if self.view.show_view_cube {
            // **A face click sets the plane you draw on, as well as the view.** Picking
            // a view and picking a plane are one act — you turned to look at the back
            // because that is where you are about to draw — and `sketcher-spec.md` §5 is
            // built around that being one gesture.
            //
            // Faces only. An edge or a corner names no single plane, so it keeps orienting
            // and nothing else.
            let grid_before = self.view.show_grid;
            if let Some(region) =
                show_view_cube(ui, &mut self.camera, rect, &mut self.view.show_grid)
            {
                if let Some(plane) = plane_of_face(region) {
                    // **A face click leaves a sketch, and says so.** It used to swap the
                    // plane being sketched on for a world plane through the origin without
                    // a word, so the next curve, and the grid and its snap, were quietly on
                    // that one (the owner, 2026-10-31: "snaps to some other grid"). Leaving
                    // is what clicking a face means — "look at the world from here" — and
                    // Sketch on turns square to the plane again.
                    if self.sketching() {
                        self.leave_sketch_keeping_view();
                    }
                    // The cube has already swung the camera, and it did so with the face's
                    // own *sign*: clicking BOTTOM looks up from below. Swinging again from
                    // the plane would use the unsigned axis and flip that to a top view, so
                    // this sets the plane and leaves the camera exactly where the cube put
                    // it.
                    self.use_drawing_plane(plane, Swing::No);
                }
            }
            if self.view.show_grid != grid_before {
                self.say_grid();
            }
        }
    }
}
