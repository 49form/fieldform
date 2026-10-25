// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The manipulator: dragging a selection, the gizmo, and the depth arrows an extrude draws.

use crate::*;

impl App {
    /// Starts moving the current selection: captures its vertices' starting positions and a
    /// drag plane (through their centroid, facing the camera) that subsequent screen-space
    /// drag motion is projected onto.
    pub(crate) fn begin_move(&mut self, aspect: f32, rect: egui::Rect, click_pos: egui::Pos2) {
        let mesh = self.control;
        let placement = self.control_placement();
        let verts = self.selection.vertices(self.control());
        let Some(centroid) = self.selection.centroid(self.control(), &placement) else {
            self.move_state = None;
            return;
        };
        let plane_normal = self.camera.forward();
        let start_positions = verts
            .iter()
            .map(|&v| (v, self.control().vertex_position(v)))
            .collect();

        let (origin, dir) = self.camera.pick_ray(aspect, click_pos, rect);
        let click_offset = ray_plane_intersect(origin, dir, centroid, plane_normal)
            .map(|p| p - centroid)
            .unwrap_or_else(Vector3::zeros);

        self.move_state = Some(MoveState {
            mesh,
            start_positions,
            plane_point: centroid,
            plane_normal,
            click_offset,
        });
    }

    /// Continues an in-progress move: projects the current cursor position onto the drag
    /// plane and translates every selected vertex by the resulting offset from its starting
    /// position.
    pub(crate) fn continue_move(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let Some(state) = self.move_state.take() else {
            return;
        };
        let (origin, dir) = self.camera.pick_ray(aspect, pos, rect);
        if let Some(hit) = ray_plane_intersect(origin, dir, state.plane_point, state.plane_normal)
        {
            let delta = (hit - state.click_offset) - state.plane_point;
            // The drag is measured in the part, where the cursor is; the cage stores its own
            // coordinates. Carrying the offset back through the placement is what keeps a
            // placed cage moving with the pointer rather than by some rotated fraction of it.
            let placement = self.mesh_placement(state.mesh);
            let local = placement.inverse_transform_vector(&delta.cast::<f64>());
            if let Some(target) = self.mesh_of_mut(state.mesh) {
                for (id, start_pos) in &state.start_positions {
                    target.set_vertex_position(*id, start_pos + local);
                }
            }
            let moved: Vec<VertexId> = state.start_positions.iter().map(|(id, _)| *id).collect();
            self.update_after_vertex_move(state.mesh, &moved);
        }
        self.move_state = Some(state);
    }

    /// Refreshes only what a vertex move actually changed.
    ///
    /// Catmull-Clark's support is local: a vertex influences the limit surface across the
    /// faces within two rings of it, and nowhere else. Rebuilding the whole model per frame
    /// made drag cost scale with model size — measured at seconds per frame on a cage of a
    /// few thousand faces, which is the failure mode this avoids.
    fn update_after_vertex_move(&mut self, mesh: ControlMesh, moved: &[VertexId]) {
        let started = std::time::Instant::now();
        // The drawn control mesh is only this mesh when this mesh is the control — a cage body
        // dragged in Field mode while a surface holds the tools has nothing drawn to refresh,
        // and writing its faces into the surface's buffer would scribble over it.
        if mesh != self.control {
            self.traced.invalidate(traced::Quality::Draft);
            return;
        }
        // Only faces *containing* a moved vertex change, since a cage face is its own four
        // corners and nothing more. The limit surface used to be re-tessellated here too, over
        // a wider neighbourhood — a patch reaches two steps out — but nothing draws it now.
        let cage = self.control().clone();
        let mut dirty_cage: Vec<FaceId> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &v in moved {
            for f in cage.adjacent_faces(v) {
                if seen.insert(f) {
                    dirty_cage.push(f);
                }
            }
        }

        self.cage_mesh.update_cage_faces(&cage, &dirty_cage);
        self.upload_dirty_faces(&dirty_cage);

        self.note_stage("following the drag", started);
        // The traced view has to be rebaked, and a full bake takes about 0.9 seconds — far too
        // slow to sit inside a drag. A draft one takes about 32 ms, which keeps up, and looks
        // like the model rather than like a low-resolution artefact of it because the normals
        // are carried through the bake rather than derived from its grid. `end_move` asks for
        // a final bake when the gesture finishes.
        self.traced.invalidate(traced::Quality::Draft);
    }

    /// Uploads only the changed vertices, instead of rebuilding every GPU buffer.
    ///
    /// This is what makes the O(1) tessellation actually O(1) in the app: without it, a drag
    /// frame re-uploaded the entire model to change a handful of vertices, so cost scaled
    /// with model size no matter how little geometry moved.
    ///
    /// Falls back to a full rebuild if a write lands out of range, which can only mean the
    /// vertex *count* changed — the buffers are fixed-size, so that needs new ones.
    fn upload_dirty_faces(&mut self, dirty_cage: &[FaceId]) {
        let queue = self.wgpu_render_state.queue.clone();
        let renderer = self.wgpu_render_state.renderer.write();
        let Some(resources) = renderer.callback_resources.get::<CageRenderResources>() else {
            drop(renderer);
            self.rebuild_render_resources();
            return;
        };

        let mut ok = true;
        for &face in dirty_cage {
            let range = CageMesh::cage_face_vertex_range(face);
            ok &= resources.write_vertices(
                &queue,
                range.start,
                &self.cage_mesh.vertices()[range.clone()],
            );
        }

        drop(renderer);
        if !ok {
            self.rebuild_render_resources();
        }
    }

    /// Ends a move, recording the whole gesture as a single undo entry.
    ///
    /// The drag mutates the cage directly every frame so the user sees live feedback; going
    /// through the command stack per frame would instead bury the history under hundreds of
    /// one-pixel entries. So on release we rewind to where the drag started and re-apply the
    /// net movement as one command — same final geometry, one undo step.
    pub(crate) fn end_move(&mut self) {
        let started = std::time::Instant::now();
        let Some(state) = self.move_state.take() else {
            return;
        };

        let vertices: Vec<VertexId> = state.start_positions.iter().map(|(v, _)| *v).collect();
        let before: Vec<Point3<f64>> = state.start_positions.iter().map(|(_, p)| *p).collect();
        let Some(moved_mesh) = self.mesh_of(state.mesh) else {
            return;
        };
        let after: Vec<Point3<f64>> =
            vertices.iter().map(|&v| moved_mesh.vertex_position(v)).collect();

        // A click that never actually moved anything shouldn't clutter the undo history.
        let moved = before
            .iter()
            .zip(&after)
            .any(|(b, a)| (a - b).norm_squared() > 1e-12);
        if !moved {
            return;
        }

        // Rewind, then let the command apply the move properly through the stack.
        if let Some(target) = self.mesh_of_mut(state.mesh) {
            for (&v, &p) in vertices.iter().zip(&before) {
                target.set_vertex_position(v, p);
            }
        }
        let cmd = self.positions_command(state.mesh, vertices, before, after);
        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
            self.warn(e.to_string());
        }
        self.mark_changed();
        self.rebuild_render_resources();
        self.note_stage("finishing the move", started);
    }

    // ---------------------------------------------------------------- the manipulator

    /// Where the manipulator should sit right now, or `None` when nothing manipulable is
    /// selected.
    ///
    /// Which target wins follows the same rule the rest of the app already obeys: picking
    /// follows whatever the current mode draws. In a cage mode the cage selection is what the
    /// user is aiming at; in field mode there is no cage on screen, so the selected body is.
    /// Letting one stand in for the other would put the gizmo somewhere nothing is drawn.
    pub(crate) fn gizmo_frame(&self) -> Option<gizmo::Frame> {
        // The section first, and only when it is *explicitly* what is selected. Deciding it
        // here — where the claim is made — is what keeps it from competing with geometry: the
        // flag is set by turning the section on and cleared by selecting anything else, so
        // there is never a moment when two things could both be answering this.
        //
        // Its own basis, not world-aligned, because unlike a body the plane *is* its frame:
        // the normal is read off the rotation, so the handles have to be the thing being
        // turned. That is the opposite of `Node::Plane`'s manipulator, where the placement is
        // separate from the geometry and world-aligned handles are what stop them fighting.
        if let Some(section) = self.selected_section() {
            let pivot = section.origin;
            return Some(gizmo::Frame {
                extrude: None,
                pivot,
                basis: section.rotation,
                reach: gizmo::Frame::reach_for(
                    self.camera.view_half_height(pivot.cast::<f32>()) as f64,
                ),
                // Move and turn, no scale: a plane has no extent for a factor to act on.
                handles: gizmo::Handles::body(),
            });
        }

        // A selected profile row: its curves as one piece, pivoted on their centroid.
        if let Some(outputs::BlockRef::Profile(key)) = self.selected_block
            && let Some(members) = self.profiles.members(key)
        {
            let points: Vec<Point3<f64>> = members
                .iter()
                .filter_map(|id| self.doc.curve(*id))
                .flat_map(|c| c.control_points().iter().copied())
                .collect();
            if !points.is_empty() {
                let sum = points.iter().fold(Vector3::zeros(), |acc, p| acc + p.coords);
                let pivot = Point3::from(sum / points.len() as f64);
                return Some(gizmo::Frame {
                    extrude: None,
                    pivot,
                    basis: Rotation3::identity(),
                    reach: gizmo::Frame::reach_for(
                        self.camera.view_half_height(pivot.cast::<f32>()) as f64,
                    ),
                    handles: gizmo::Handles::extended(),
                });
            }
        }

        let editable = self.control_editable();

        // Control points come first when any are picked. Curves are drawn in every mode — a
        // subtracting curve is inside the solid it cuts, and hiding it would hide the thing
        // being aimed — so there is no mode where handles on one would sit over nothing.
        if !self.selected_points.is_empty() || !self.other_points.is_empty() {
            // Every picked point, on whichever curve it is.
            let picked: Vec<Point3<f64>> = self
                .picked_points()
                .iter()
                .filter_map(|(c, i)| self.doc.curve(*c).and_then(|c| c.control_points().get(*i).copied()))
                .collect();
            if !picked.is_empty() {
                let sum = picked.iter().fold(Vector3::zeros(), |acc, p| acc + p.coords);
                let pivot = Point3::from(sum / picked.len() as f64);
                return Some(gizmo::Frame {
                    extrude: None,
                    pivot,
                    basis: Rotation3::identity(),
                    reach: gizmo::Frame::reach_for(
                        self.camera.view_half_height(pivot.cast::<f32>()) as f64,
                    ),
                    // One point has no extent to turn or scale about; several do.
                    handles: if picked.len() == 1 {
                        gizmo::Handles::single_point()
                    } else {
                        gizmo::Handles::extended()
                    },
                });
            }
        }

        // A whole curve, when one is selected and no point on it is. Selecting a curve and
        // getting handles is what "move this curve" has to mean — moving every control point
        // by hand is the same edit done N times, and rotating one that way is not reachable at
        // all.
        //
        // Behind the cage, though. A cage selection is made *in the viewport*, deliberately and
        // just now, where a curve is often selected in the panel and left that way to keep its
        // points on screen; letting the older, more passive selection win would put handles
        // over the cage the user is actually working on.
        let cage_claimed = editable
            && self
                .selection
                .centroid(self.control(), &self.control_placement())
                .is_some();
        if !cage_claimed
            && let Some(curve) = self.selected_curve
            && !self.hidden_curves.contains(&curve)
            && let Some(points) = self.doc.curve(curve).map(|c| c.control_points().to_vec())
            && !points.is_empty()
        {
            let sum = points.iter().fold(Vector3::zeros(), |acc, p| acc + p.coords);
            let pivot = Point3::from(sum / points.len() as f64);
            return Some(gizmo::Frame {
                extrude: None,
                pivot,
                basis: Rotation3::identity(),
                reach: gizmo::Frame::reach_for(
                    self.camera.view_half_height(pivot.cast::<f32>()) as f64,
                ),
                // A curve has extent unless it is a single point, so turning and scaling it
                // both mean something.
                handles: if points.len() == 1 {
                    gizmo::Handles::single_point()
                } else {
                    gizmo::Handles::extended()
                },
            });
        }

        // The manipulator appears over what the current mode actually draws, and over nothing
        // else. Cage mode draws no field, so it offers no body handles however tempting the
        // fallback is — that is the rule two previous simplifications broke. Overlay draws
        // both, so with nothing picked on the cage it can still offer the body.
        let draws_field = matches!(self.view_mode, ViewMode::Field | ViewMode::Overlay);
        let placement = self.control_placement();
        let on_cage = editable && self.selection.centroid(self.control(), &placement).is_some();
        if !on_cage && !draws_field {
            return None;
        }

        let (pivot, basis, handles) = if on_cage {
            let cage = self.control();
            let pivot = self.selection.centroid(cage, &placement)?.cast::<f64>();
            // A face turns the gizmo so one axis leaves the surface: moving along a face's own
            // normal is more often what was meant than moving along a world axis. A vertex or
            // an edge has no such direction, so those stay world-aligned.
            let basis = self
                .selection
                .normal(cage, &placement)
                .and_then(|n| {
                    Rotation3::rotation_between(&Vector3::z(), &n.cast::<f64>())
                })
                .unwrap_or_else(Rotation3::identity);
            let handles = if self.selection.is_single_vertex() {
                gizmo::Handles::single_point()
            } else {
                gizmo::Handles::extended()
            };
            (pivot, basis, handles)
        } else {
            let body = self.selected_body?;
            // A body whose eye is off has no handles. A block something consumes keeps them
            // whatever its own eye says, as it did when this read `hidden` alone: that eye only
            // draws it beside what consumes it as well, and was never what decided whether it
            // can be moved. Nor is whether that consumer is drawn, which this has never asked.
            if self.eye_flag(body) == EyeFlag::Hidden && !self.is_visible(body) {
                return None;
            }
            // No box means no handles: an unbounded body has no extent to turn or scale
            // about, and a gizmo drawn at a guessed size would move it by a guessed amount.
            // A plane has no box to centre handles in, so it says where its own are: at the
            // point it passes through, turned to face the way it does. Without this the
            // manipulator refused a plane outright, which left the one primitive that most
            // needs aiming as the only one that could not be aimed.
            if let Some((_, origin, _)) = self.selected_plane() {
                // World-aligned, exactly like every other body, and deliberately *not*
                // turned to the plane.
                //
                // Two attempts got this wrong in the same way. Rebuilding the basis from the
                // normal was worst — a normal fixes two of three degrees of freedom, so the
                // roll about it had to be invented and was invented differently as the plane
                // turned. Taking it from the placement fixed the jumping but kept the deeper
                // problem: the handles moved *while being dragged*, because a rotation changes
                // the very thing the axes were read from. Turning about Z then walked the
                // other two arcs out from under the cursor.
                //
                // A handle has to stay where it was grabbed. The cage face precedent does not
                // apply, because a face's basis is read from geometry the drag is not turning.
                return Some(gizmo::Frame {
                    extrude: None,
                    pivot: origin,
                    basis: Rotation3::identity(),
                    reach: gizmo::Frame::reach_for(
                        self.camera.view_half_height(origin.cast::<f32>()) as f64,
                    ),
                    // Move and turn, but not scale: a half-space has no size to scale.
                    handles: gizmo::Handles::body(),
                });
            }
            let (lo, hi) = self.body_extent(body).as_box()?;
            // `graph_extent` reads a cage node's own coordinates, which are not where it is
            // drawn if something above it has been placed. Every other body kind already
            // accounts for its placement there, because the placement *is* a node on the path.
            let (lo, hi) = if self.is_active_cage_node(body) {
                let p = self.cage_placement();
                (p * lo, p * hi)
            } else {
                (lo, hi)
            };
            // A cage or a pipe is moved by moving points — the cage's own, or the curve the
            // pipe is swept along — so both get the per-axis scale cubes. Squashing points
            // along one axis cannot cost the result its distance property, which is the only
            // reason a *placed* body is denied them.
            let moves_points = self.is_active_cage_node(body)
                || matches!(self.part_graph().node(body), Some(Node::Pipe { .. }));
            // **A point block has nothing to turn or scale, and said otherwise.**
            // `Handles::single_point` already carries the reason -- "a single point has no
            // extent, so rotating or scaling about it is a no-op" -- and a cage vertex and a
            // curve's control point both use it. A `Node::Point` standing as a body is the same
            // thing and was getting the body set: three arcs and a scale cube drawn around a
            // dot, every one of which does nothing a person can see. Reported as "when moving a
            // point the gizmo rotate handle should disappear, they are superfluous".
            //
            // `selected_point_node` rather than a match here, because it walks the placements a
            // point may be wrapped in, and a placed point is still a point: rotating its
            // placement turns it about the pivot the handles sit on, which is the point itself.
            let handles = if self.selected_point_node().is_some() {
                gizmo::Handles::single_point()
            } else if moves_points {
                gizmo::Handles::extended()
            } else {
                gizmo::Handles::body()
            };
            // **On the block's own origin when it has one, and the box centre when it does
            // not.** A sphere's centre *is* the middle of its box, so nothing changes there; a
            // cone's is its base, which is where its numbers say it is and where turning it
            // about is what "rotate this cone" means. The handles sitting somewhere the block
            // has no name for is what made the gizmo and the panel disagree about where a shape
            // was.
            //
            // The first anchor, for the two-ended shapes that carry a pair: a cylinder is drawn
            // between its ends and the first is the one its block lists first. The box centre
            // remains for everything with no position of its own -- a boolean, a lattice, a
            // mesh, a cage -- which is every block a placement is the only way to move.
            let anchor = self
                .own_anchors(body)
                .and_then(|a| a.first().copied())
                .unwrap_or_else(|| nalgebra::center(&lo, &hi));
            (anchor, Rotation3::identity(), handles)
        };

        Some(gizmo::Frame {
            pivot,
            basis,
            reach: gizmo::Frame::reach_for(
                self.camera.view_half_height(pivot.cast::<f32>()) as f64,
            ),
            handles,
            // Beside the triad, over a selected extrude only, and not over a cage: the arrows
            // are on the field's solid, which a cage mode does not draw.
            extrude: if on_cage { None } else { self.selected_body.and_then(|b| self.extrude_arrows(b)) },
        })
    }

    /// The depth arrows of an extrude, or `None` where they would stand somewhere the solid is not.
    ///
    /// **A straight sweep only**, because a leaning one is sheared and its far face is not over the
    /// section's centre; and **only where nothing above moves it**, because the arrow is placed from
    /// the extrude's own coordinates, and a placement, a pattern, a mirror, a twist, an attractor or
    /// a morph above it would draw the solid somewhere else. In those cases the depth is typed in the
    /// block, as it always was.
    fn extrude_arrows(&self, id: NodeId) -> Option<gizmo::ExtrudeArrows> {
        let graph = self.part_graph();
        let Node::Extrude { outline, sweep, start, height, sides, .. } = graph.node(id)? else {
            return None;
        };
        if !matches!(sweep, fieldform_scene::Sweep::Normal) || !drawn_where_it_is(graph, id) {
            return None;
        }
        // From the function the field lays the section down with, so the arrow is on the solid
        // that is drawn — a wall's as much as a profile's.
        let section = fieldform_field::extrude_section(
            &self.doc,
            outline,
            sweep,
            fieldform_field::PROFILE_TURN_DEGREES,
        )
        .ok()?;
        let (start, h1) = (self.resolve(*start), self.resolve(*height));
        // Where the material ends on each side, along the sweep from the profile's own plane.
        let (far_one, far_two) = match sides {
            fieldform_scene::Sides::One => (start + h1, None),
            fieldform_scene::Sides::Symmetric => (start + h1 * 0.5, None),
            fieldform_scene::Sides::Two { height: h2, .. } => {
                (start + h1, Some(start - self.resolve(*h2)))
            }
        };
        let at = |z: f64| section.origin + section.normal * z;
        Some(gizmo::ExtrudeArrows {
            direction: section.normal,
            tips: [Some(at(far_one)), far_two.map(at)],
        })
    }

    /// What drives a depth, in words, when something does: the variable's name, or a ramp.
    fn depth_driver(&self, r: ParamRef) -> Option<String> {
        match r {
            ParamRef::Literal(_) => None,
            ParamRef::Named(id) => Some(format!("\u{201c}{}\u{201d}", self.parameter_label(id))),
            ParamRef::Field(_) => Some("a ramp".to_string()),
        }
    }

    /// The depth `side` of an extrude holds now, as a number. Zero for what is not one.
    fn depth_of(&self, node: NodeId, side: usize) -> f64 {
        match self.part_graph().node(node) {
            Some(Node::Extrude { height, sides, .. }) => match (side, sides) {
                (0, _) => self.resolve(*height),
                (_, fieldform_scene::Sides::Two { height: h2, .. }) => self.resolve(*h2),
                _ => 0.0,
            },
            _ => 0.0,
        }
    }

    /// Projects for the manipulator, in the doubles it works in.
    pub(crate) fn gizmo_project(
        &self,
        aspect: f32,
        rect: egui::Rect,
    ) -> impl Fn(Point3<f64>) -> Option<egui::Pos2> + '_ {
        move |p: Point3<f64>| self.camera.project_to_screen(aspect, rect, p.cast::<f32>())
    }

    /// A cursor ray, in doubles.
    fn gizmo_ray(
        &self,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> (Point3<f64>, Vector3<f64>) {
        let (origin, direction) = self.camera.pick_ray(aspect, pos, rect);
        (origin.cast::<f64>(), direction.cast::<f64>().normalize())
    }

    /// Starts a manipulator drag on `handle`, freezing whatever it is about to edit.
    pub(crate) fn begin_gizmo(&mut self, frame: &gizmo::Frame, handle: gizmo::Handle, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let (origin, direction) = self.gizmo_ray(aspect, rect, pos);
        let Some(drag) = gizmo::Drag::begin(frame, handle, origin, direction, pos) else {
            // The view is looking almost exactly along the handle's axis, so there is no
            // usable drag plane. Saying so beats a handle that jitters.
            self.warn("Turn the view a little — that handle is edge-on");
            return;
        };

        // An extrude's depth arrow, which the frame offered over the selected extrude and which
        // belongs to nothing else.
        if let gizmo::Handle::Extrude(side) = drag.handle {
            let Some(body) = self.selected_body else { return };
            let Some(Node::Extrude { height, sides, .. }) = self.part_graph().node(body).cloned()
            else {
                return;
            };
            // The number this arrow edits, refused if it is not a plain one: overwriting a
            // variable's expression with a literal is a decision that is the person's.
            let edited = match (side, &sides) {
                (0, _) => height,
                (_, fieldform_scene::Sides::Two { height: h2, .. }) => *h2,
                _ => return,
            };
            if let Some(why) = self.depth_driver(edited) {
                self.warn(format!("This depth is driven by {why}; edit it there"));
                return;
            }
            let Some(direction) = frame.extrude.map(|e| e.direction) else { return };
            self.gizmo_drag = Some(GizmoDrag {
                drag,
                target: GizmoTarget::Depth {
                    node: body,
                    side,
                    symmetric: matches!(sides, fieldform_scene::Sides::Symmetric),
                    was: self.depth_of(body, side),
                    direction,
                    before_graph: self.part_graph().clone(),
                },
                pivot: frame.pivot,
                readout: String::new(),
            });
            return;
        }

        // The section, matching `gizmo_frame`'s order exactly. The two must agree about what
        // the handles belong to, or a drag moves something other than what was drawn.
        if let Some((index, section)) = self.section_selected.zip(self.selected_section()) {
            self.gizmo_drag = Some(GizmoDrag {
                drag,
                target: GizmoTarget::Section { index, before: section },
                pivot: frame.pivot,
                readout: String::new(),
            });
            return;
        }

        // A selected profile row moves every curve of its loop as one piece.
        if let Some(outputs::BlockRef::Profile(key)) = self.selected_block
            && let Some(members) = self.profiles.members(key).map(|m| m.to_vec())
        {
            let mut moves = Vec::new();
            for curve in members {
                let Some(points) = self.doc.curve(curve).map(|c| c.control_points().to_vec())
                else {
                    continue;
                };
                let indices: Vec<usize> = (0..points.len()).collect();
                let Some(gesture) = self.begin_curve_gesture(curve, &indices) else { return };
                moves.push(CurveMove { curve, indices, before: points, gesture });
            }
            if moves.is_empty() {
                return;
            }
            // The first gesture's graph is the one from before anything moved.
            let graph_before = moves.iter().find_map(|m| m.gesture.graph_before.clone());
            let shapes_field = moves.iter().any(|m| m.gesture.shapes_field);
            self.gizmo_drag = Some(GizmoDrag {
                drag,
                target: GizmoTarget::Profile { moves, graph_before, shapes_field },
                pivot: frame.pivot,
                readout: String::new(),
            });
            return;
        }

        // A curve, whatever the mode is drawing: the points picked on it, or all of them when
        // none is — which is the same drag either way, over a different set.
        let cage_claimed = {
            let placement = self.control_placement();
            self.control_editable()
                && self.selection.centroid(self.control(), &placement).is_some()
        };
        // Points picked across several curves move as one piece, the way a profile row's do.
        if !self.other_points.is_empty() {
            let mut moves = Vec::new();
            for (curve, indices) in group_by_curve(&self.picked_points()) {
                let Some(points) = self.doc.curve(curve).map(|c| c.control_points().to_vec())
                else {
                    continue;
                };
                let before: Vec<Point3<f64>> =
                    indices.iter().filter_map(|i| points.get(*i).copied()).collect();
                if before.len() != indices.len() {
                    continue;
                }
                let Some(gesture) = self.begin_curve_gesture(curve, &indices) else { return };
                moves.push(CurveMove { curve, indices, before, gesture });
            }
            if !moves.is_empty() {
                let graph_before = moves.iter().find_map(|m| m.gesture.graph_before.clone());
                let shapes_field = moves.iter().any(|m| m.gesture.shapes_field);
                self.gizmo_drag = Some(GizmoDrag {
                    drag,
                    target: GizmoTarget::Profile { moves, graph_before, shapes_field },
                    pivot: frame.pivot,
                    readout: String::new(),
                });
                return;
            }
        }
        if (!self.selected_points.is_empty() || !cage_claimed)
            && let Some(curve) = self.selected_curve
            && !self.hidden_curves.contains(&curve)
            && let Some(points) = self.doc.curve(curve).map(|c| c.control_points().to_vec())
        {
            let indices: Vec<usize> = if self.selected_points.is_empty() {
                (0..points.len()).collect()
            } else {
                self.selected_points
                    .iter()
                    .copied()
                    .filter(|i| *i < points.len())
                    .collect()
            };
            if !indices.is_empty() {
                let before = indices.iter().map(|i| points[*i]).collect();
                // The same gesture the point drag begins, so a held end moves its block
                // under the handles too rather than snapping back on release.
                let Some(gesture) = self.begin_curve_gesture(curve, &indices) else { return };
                self.gizmo_drag = Some(GizmoDrag {
                    drag,
                    target: GizmoTarget::Curve { curve, indices, before, gesture },
                    pivot: frame.pivot,
                    readout: String::new(),
                });
                return;
            }
        }

        let editable = self.control_editable();
        let target = if editable {
            let cage = self.control();
            let vertices = self.selection.vertices(cage);
            if vertices.is_empty() {
                return;
            }
            let before = vertices.iter().map(|&v| cage.vertex_position(v)).collect();
            GizmoTarget::Cage { mesh: self.control, vertices, before }
        } else {
            let Some(body) = self.selected_body else { return };

            // Moving a pipe moves the curve it is swept along.
            //
            // A pipe has no position of its own. It *is* a curve thickened — `Node::Pipe` holds
            // a curve and a radius and nothing else — so asking where the pipe is is asking
            // where its curve is, and the honest answer to "move this pipe" is to move that.
            // Placing it instead would move the field while leaving the curve drawn where it
            // was built, which is the same two-descriptions-disagreeing failure a placed cage
            // had, and the drag handles would end up nowhere near the curve they were really
            // editing.
            //
            // Two pipes sharing a curve therefore move together, which is correct rather than
            // surprising: they are the same curve, and it moved.
            //
            // Nothing about the graph changes here. `Node::Transform` stays general and can
            // still wrap a pipe — this is the manipulator choosing the edit that matches what
            // the user pointed at, which is exactly what it already does for a cage body.
            if let Some(Node::Pipe { curve, .. }) = self.part_graph().node(body).cloned()
                && let Some(points) = self.doc.curve(curve).map(|c| c.control_points().to_vec())
                && !points.is_empty()
            {
                let indices: Vec<usize> = (0..points.len()).collect();
                // Moving the whole curve is the same gesture as moving its points: an end
                // held on a block moves the block, so a pipe along a held loop stays held.
                let Some(gesture) = self.begin_curve_gesture(curve, &indices) else { return };
                self.gizmo_drag = Some(GizmoDrag {
                    drag,
                    target: GizmoTarget::Curve { curve, indices, before: points, gesture },
                    pivot: frame.pivot,
                    readout: String::new(),
                });
                return;
            }

            // Moving a cage body moves the cage, not a placement wrapping it.
            //
            // A placement transforms the *field*, and the control mesh is drawn straight from
            // the cage's own coordinates with no model matrix — so placing a cage body moved
            // the surface and left the cage it is edited through sitting where it started.
            // Two descriptions of where the same thing is, disagreeing, which is the failure
            // the picking rule exists to prevent: the handles would be over there and the
            // points to grab still back here.
            //
            // Moving the points instead is also the honest reading of the layer split. The
            // cage is the direct layer — its current state *is* the truth, nothing is replayed
            // — so a gesture that moves a cage belongs below the line, reversing against the
            // exact prior positions like every other cage edit.
            if self.is_active_cage_node(body) {
                let cage = self.cage();
                let vertices: Vec<VertexId> = cage.vertex_ids().collect();
                if vertices.is_empty() {
                    return;
                }
                let before = vertices.iter().map(|&v| cage.vertex_position(v)).collect();
                self.gizmo_drag = Some(GizmoDrag {
                    drag,
                    target: GizmoTarget::Cage { mesh: ControlMesh::Cage, vertices, before },
                    pivot: frame.pivot,
                    readout: String::new(),
                });
                return;
            }

            let before_graph = self.part_graph().clone();

            // **A block that says where it is has its own numbers moved, not a wrapper added.**
            // Translation only, decided from the handle that was grabbed: a centre can express
            // where a sphere is and cannot express how it is turned, so an arc or a scale cube
            // still wraps. Deciding here rather than when the first motion arrives is what keeps
            // the whole drag one kind of edit -- switching halfway would leave a placement
            // holding part of the gesture and the block's own coordinates the rest.
            let translating = matches!(
                drag.handle,
                gizmo::Handle::Axis(_) | gizmo::Handle::Plane(_)
            );
            // A turn goes the same way, for the blocks that carry one: a box, a torus and a
            // plane each say which way they face, so an arc should move that number rather than
            // wrap a placement around the block to hold it. A block with no turn of its own --
            // a sphere, or anything drawn between two ends -- still wraps, and so does a scale,
            // which no primitive carries.
            // A block sitting on a point is moved by moving that point -- see `drag_moves`.
            // A turn is always the block's own: sitting on something says where, not which way.
            let moves = self.drag_moves(body);
            let own_turn = self.own_turn(body);
            let rotating = matches!(drag.handle, gizmo::Handle::Arc(_)) && own_turn.is_some();
            let anchors = match rotating {
                true => self.own_anchors(body),
                false => self.own_anchors(moves),
            };
            if (translating || rotating)
                && let Some(before) = anchors
            {
                self.gizmo_drag = Some(GizmoDrag {
                    drag,
                    target: GizmoTarget::Anchors {
                        node: match rotating {
                            true => body,
                            false => moves,
                        },
                        before,
                        before_turn: own_turn,
                        before_graph,
                    },
                    pivot: frame.pivot,
                    readout: String::new(),
                });
                return;
            }

            // Wrapped once, here, rather than on every frame of the drag. Adding a node is a
            // structural change and costs a shader recompile; changing its numbers afterwards
            // is a buffer write. Doing this per frame would recompile per frame.
            let mut graph = before_graph.clone();
            let placement = if matches!(graph.node(body), Some(Node::Transform { .. })) {
                body
            } else if let Some(existing) = self.part_graph().enclosing_transform(body) {
                // Already placed, and selected by its inner row in the block list. Move the
                // placement it has rather than giving it another.
                existing
            } else {
                let node = graph.add(Placement::identity().into_node(body));
                // The placement takes the body's place in whatever consumed it, rather than
                // sitting beside it. Dragging a sphere that was one side of a blend used to add
                // a second consumer of that sphere: the part gained a stray body drawing an
                // unblended copy, and the shader generator refused the graph outright because
                // one node would have to be sampled at two points — the viewport went blank.
                graph.rewire_consumers(body, node);
                // The selection follows the placement, because the placement is now what the
                // part draws in that spot and what a click there will hit.
                self.selected_body = Some(node);
                node
            };
            let before = Placement::of(self, graph.node(placement));
            self.set_part_graph(graph);
            // Draft, not the settle. Wrapping the body did change the shader's structure, so
            // this rebuild recompiles — but the geometry is identical (the new placement is the
            // identity), so paying 0.9 s for a final bake here would be a hitch at the start of
            // every drag in exchange for a picture that has not changed.
            self.traced.invalidate(traced::Quality::Draft);
            GizmoTarget::Body { placement, before_graph, before }
        };

        self.gizmo_drag = Some(GizmoDrag {
            drag,
            target,
            pivot: frame.pivot,
            readout: String::new(),
        });
    }

    /// Advances a manipulator drag and applies it live.
    ///
    /// Applied to the state frozen at mouse-down every frame, never to the previous frame's
    /// result — the same rule the cage's own drag already follows, and what makes a dropped
    /// frame or a wandering cursor leave no trace in the final position.
    pub(crate) fn continue_gizmo(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2, snap: bool) {
        let Some(mut live) = self.gizmo_drag.take() else { return };
        let (origin, direction) = self.gizmo_ray(aspect, rect, pos);
        let Some(motion) = live.drag.update(origin, direction, pos) else {
            // A ray that missed the drag plane is a frame with no news, not the end of the
            // drag: put the state back and wait for the next one.
            self.gizmo_drag = Some(live);
            return;
        };
        live.readout = live.drag.readout(motion);
        let pivot = live.pivot;
        // What a target says in place of the motion's own readout, for the ones whose result is a
        // number the motion is not: a depth, not the distance the arrow was pulled.
        let mut said: Option<String> = None;

        // Neither arm may call `rebuild_render_resources` — that is the settle path. It
        // rebuilds the whole cage mesh, re-uploads every GPU buffer, recompiles the picking
        // field (which subdivides the cage), and asks for a *final* bake at about 0.9 s. Once,
        // on release, that is right; sixty times a second it is why a drag crawls.
        match &live.target {
            GizmoTarget::Cage { mesh, vertices, before } => {
                // Out through the placement, moved in the part, and back again. The motion and
                // the pivot are both where the user is pointing; the positions written are
                // where the cage keeps them.
                let mesh = *mesh;
                let placement = self.mesh_placement(mesh);
                let written: Vec<(VertexId, Point3<f64>)> = vertices
                    .iter()
                    .zip(before)
                    .map(|(&v, &start)| {
                        let placed = moved_point(placement * start, motion, pivot);
                        (v, placement.inverse_transform_point(&placed))
                    })
                    .collect();
                if let Some(target) = self.mesh_of_mut(mesh) {
                    for (v, p) in &written {
                        target.set_vertex_position(*v, *p);
                    }
                }
                let moved: Vec<VertexId> = written.iter().map(|(v, _)| *v).collect();
                // The same local refresh the cage's own drag uses: only the faces touching a
                // moved vertex, only the vertices that moved, and a draft bake.
                self.update_after_vertex_move(mesh, &moved);
            }
            GizmoTarget::Curve { curve, indices, before, gesture } => {
                // The gesture's shared path: a held end moves its block here exactly as it
                // does under the control point's own drag, and the field is redrawn at draft
                // only when the curve shapes one.
                let (curve, gesture) = (*curve, gesture.clone());
                let to: Vec<(usize, Point3<f64>)> = indices
                    .iter()
                    .zip(before)
                    .map(|(&i, &start)| (i, moved_point(start, motion, pivot)))
                    .collect();
                self.move_curve_points_live(curve, &to, &gesture);
            }
            // Nothing in the document changes, so nothing but the trace needs refreshing —
            // and the trace is where the cut is applied. A draft rebake, like the others.
            GizmoTarget::Profile { moves, .. } => {
                let moves = moves.clone();
                for m in &moves {
                    let to: Vec<(usize, Point3<f64>)> = m
                        .indices
                        .iter()
                        .zip(&m.before)
                        .map(|(&i, &start)| (i, moved_point(start, motion, pivot)))
                        .collect();
                    self.move_curve_points_live(m.curve, &to, &m.gesture);
                }
            }
            GizmoTarget::Section { index, before } => {
                if let Some(slot) = self.sections.get_mut(*index) {
                    *slot = before.with(motion, pivot);
                }
                self.traced.invalidate(traced::Quality::Draft);
            }
            GizmoTarget::Anchors { node, before, before_turn, .. } => {
                let mut graph = self.part_graph().clone();
                match motion {
                    // Every anchor moves by the same vector, so a cylinder keeps its length and
                    // a cone keeps its taper: this is the block being moved, not reshaped.
                    gizmo::Motion::Translate(d) => {
                        let moved: Vec<Point3<f64>> = before.iter().map(|p| p + d).collect();
                        Self::set_own_anchors(&mut graph, *node, &moved);
                    }
                    // Turned about the gizmo's own pivot, which is the block's origin -- so a
                    // box spins where it stands and its centre does not move. That is why the
                    // anchors are left alone here.
                    gizmo::Motion::Rotate { axis, degrees } => {
                        let Some(was) = *before_turn else { return };
                        let Some(unit) = nalgebra::Unit::try_new(axis, 1e-9) else { return };
                        let turned =
                            Rotation3::from_axis_angle(&unit, degrees.to_radians()) * was;
                        Self::set_own_turn(&mut graph, *node, turned);
                    }
                    gizmo::Motion::Scale { .. } => return,
                }
                self.set_part_graph(graph);
                self.refresh_scene_bounds();
                self.traced.invalidate(traced::Quality::Draft);
            }
            GizmoTarget::Depth { node, side, symmetric, was, direction, .. } => {
                if let gizmo::Motion::Translate(d) = motion {
                    let depth = dragged_depth(*side, *symmetric, *was, d.dot(direction), snap);
                    let mut graph = self.part_graph().clone();
                    if let Some(Node::Extrude { height, sides, .. }) = graph.node_mut(*node) {
                        match (side, sides) {
                            (0, _) => *height = ParamRef::Literal(depth),
                            (_, fieldform_scene::Sides::Two { height: h2, .. }) => {
                                *h2 = ParamRef::Literal(depth)
                            }
                            _ => {}
                        }
                    }
                    self.set_part_graph(graph);
                    self.refresh_scene_bounds();
                    self.traced.invalidate(traced::Quality::Draft);
                    said = Some(format!(
                        "{} {depth:.2} mm{}",
                        if *side == 0 { "Depth" } else { "Depth 2" },
                        if snap { "  (whole mm)" } else { "" },
                    ));
                }
            }
            GizmoTarget::Body { placement, before, .. } => {
                let placed = before.with(motion, pivot);
                let mut graph = self.part_graph().clone();
                // Only the numbers change here — the wrapper node and its input were settled at
                // drag start — so the shader's structure is untouched and the cage underneath
                // is not edited at all. A draft rebake is all this needs; the cage mesh and the
                // picking field describe geometry this drag has not altered.
                if let Some(Node::Transform { input, .. }) = graph.node(*placement).cloned() {
                    if let Some(slot) = graph.node_mut(*placement) {
                        *slot = placed.into_node(input);
                    }
                    self.set_part_graph(graph);
                    // The reach follows the body, or it is cut off at where it was.
                    self.refresh_scene_bounds();
                    self.traced.invalidate(traced::Quality::Draft);
                }
            }
        }
        if let Some(text) = said {
            live.readout = text;
        }
        self.gizmo_drag = Some(live);
    }

    /// Ends a manipulator drag, committing it as exactly one undo step.
    ///
    /// The same rewind the cage's own drag does, for the same reason: the document was mutated
    /// live for feedback, so it is put back to where the drag started and the whole gesture is
    /// re-applied through the stack as one entry. A drag that moved nothing commits nothing.
    pub(crate) fn end_gizmo(&mut self) {
        let Some(live) = self.gizmo_drag.take() else { return };
        match live.target {
            // Nothing to commit: the section is not in the document, so there is nothing for
            // undo to reverse and nothing a save would have written. It was moved live and it
            // stays where it was left — the same as the camera, which is what it belongs to.
            GizmoTarget::Section { .. } => {
                self.traced.invalidate(traced::Quality::Final);
            }
            GizmoTarget::Cage { mesh, vertices, before } => {
                let Some(moved_mesh) = self.mesh_of(mesh) else { return };
                let after: Vec<Point3<f64>> =
                    vertices.iter().map(|&v| moved_mesh.vertex_position(v)).collect();
                let moved = before
                    .iter()
                    .zip(&after)
                    .any(|(b, a)| (a - b).norm_squared() > 1e-12);
                if !moved {
                    return;
                }
                if let Some(target) = self.mesh_of_mut(mesh) {
                    for (&v, &p) in vertices.iter().zip(&before) {
                        target.set_vertex_position(v, p);
                    }
                }
                let cmd = self.positions_command(mesh, vertices, before, after);
                if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                    self.warn(e.to_string());
                }
            }
            GizmoTarget::Curve { curve, indices, before, gesture } => {
                // No snapping here. It lives on the control point's own drag, which is where
                // it can be *seen* — the manipulator's version could only decide at the end,
                // which is a snap you cannot aim. The commit is the gesture's shared path:
                // rewound and re-applied through the stack as one entry, the blocks a held
                // end sits on included, and the field rebuilt only when the curve shapes one.
                let shapes_field = gesture.shapes_field;
                if !self.commit_curve_gesture(curve, &indices, &before, gesture) {
                    return;
                }
                self.after_curve_gesture(shapes_field);
            }
            GizmoTarget::Profile { moves, graph_before, shapes_field } => {
                if !self.commit_curves_gesture(moves, graph_before) {
                    return;
                }
                self.after_curve_gesture(shapes_field);
            }
            GizmoTarget::Anchors { node, before, before_turn, before_graph } => {
                let after = self.own_anchors(node).unwrap_or_default();
                let turned = match (before_turn, self.own_turn(node)) {
                    (Some(b), Some(a)) => a.angle_to(&b) > 1e-9,
                    _ => false,
                };
                let moved = turned
                    || before
                        .iter()
                        .zip(&after)
                        .any(|(b, a)| (a - b).norm_squared() > 1e-18);
                if !moved {
                    self.set_part_graph(before_graph);
                    self.rebuild_render_resources();
                    return;
                }
                let after_graph = self.part_graph().clone();
                self.set_part_graph(before_graph.clone());
                let cmd = Box::new(SetPartGraph::new(self.active_part, before_graph, after_graph));
                if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                    self.warn(e.to_string());
                }
            }
            // One undo step, and none for a drag that did not change the number: the graph is put
            // back to where it began and the whole gesture re-applied through the stack, as the
            // block's own numbers are.
            GizmoTarget::Depth { node, side, was, before_graph, .. } => {
                if (self.depth_of(node, side) - was).abs() < 1e-12 {
                    self.set_part_graph(before_graph);
                    self.rebuild_render_resources();
                    return;
                }
                let after = self.part_graph().clone();
                self.set_part_graph(before_graph.clone());
                let cmd = Box::new(SetPartGraph::new(self.active_part, before_graph, after));
                if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                    self.warn(e.to_string());
                }
            }
            GizmoTarget::Body { placement, before_graph, before } => {
                // Compared as a placement rather than as two graphs, because that is the
                // actual question. A press that never moved still *wrapped* the body, so a
                // graph comparison would call that a change and leave an identity placement
                // behind — a node in the file, and an undo step, for a click.
                let after_placement = Placement::of(self, self.part_graph().node(placement));
                let moved = (after_placement.translation - before.translation).norm() > 1e-9
                    || (after_placement.scale - before.scale).abs() > 1e-12
                    || after_placement.rotation.angle_to(&before.rotation) > 1e-9;
                if !moved {
                    self.put_back_pressed_graph(before_graph);
                    self.rebuild_render_resources();
                    return;
                }
                let after = self.part_graph().clone();
                self.set_part_graph(before_graph.clone());
                let cmd = Box::new(SetPartGraph::new(self.active_part, before_graph, after));
                if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                    self.warn(e.to_string());
                }
            }
        }
        self.mark_changed();
        self.rebuild_render_resources();
    }

    /// Puts the active part's graph back to where a manipulator press found it, through
    /// `set_part_graph` as before, and says nothing of what putting it back settles.
    ///
    /// A press on a body with no placement wraps it in one (`begin_gizmo`), and the wrap adopts
    /// the body, which is not said. Taking the wrap away lets the body go, which would be: a frame
    /// after "Cancelled", "Sphere is on its own again", about a block the person never put
    /// anywhere. The wrap was the press's, and so is its undoing, so what this settles is dropped
    /// before the drain hears of it. No name is owed by anything dropped: the wrap took over the
    /// body's readers and putting it back hands them back, so the body is let go or re-homed, or
    /// stays the variable it was, and nothing else is read by anything new.
    fn put_back_pressed_graph(&mut self, before_graph: Graph) {
        let heard = self.doc.pending_ownership_changes.len();
        self.set_part_graph(before_graph);
        self.doc.pending_ownership_changes.truncate(heard);
    }

    /// Abandons a manipulator drag, restoring what it found and recording nothing.
    ///
    /// Nothing goes on the undo stack — not even a command that undoes to the same place. A
    /// cancelled gesture did not happen, and an entry saying otherwise would make the next
    /// Ctrl+Z appear to do nothing.
    pub(crate) fn cancel_gizmo(&mut self) {
        let Some(live) = self.gizmo_drag.take() else { return };
        match live.target {
            GizmoTarget::Cage { mesh, vertices, before } => {
                if let Some(target) = self.mesh_of_mut(mesh) {
                    for (&v, &p) in vertices.iter().zip(&before) {
                        target.set_vertex_position(v, p);
                    }
                }
            }
            GizmoTarget::Curve { curve, indices, before, gesture } => {
                // The blocks go back too, or a held end would be put back and then settled
                // onto a block still where the gesture left it.
                if let Some(was) = gesture.graph_before {
                    self.restore_part_graph(was);
                }
                if let Some(c) = self.doc.curve_mut(curve) {
                    for (&i, &p) in indices.iter().zip(&before) {
                        c.set_control_point(i, p);
                    }
                }
                self.rebuild_curve_display();
            }
            GizmoTarget::Profile { moves, graph_before, .. } => {
                if let Some(was) = graph_before {
                    self.restore_part_graph(was);
                }
                for m in moves {
                    if let Some(c) = self.doc.curve_mut(m.curve) {
                        for (&i, &p) in m.indices.iter().zip(&m.before) {
                            c.set_control_point(i, p);
                        }
                    }
                }
                self.rebuild_curve_display();
            }
            GizmoTarget::Anchors { before_graph, .. }
            | GizmoTarget::Depth { before_graph, .. }
            | GizmoTarget::Body { before_graph, .. } => self.put_back_pressed_graph(before_graph),
            // Put back where the gesture started, like the others. Nothing was committed, so
            // there is nothing on the stack to disagree with this.
            GizmoTarget::Section { index, before } => {
                if let Some(slot) = self.sections.get_mut(index) {
                    *slot = before;
                }
                self.traced.invalidate(traced::Quality::Final);
            }
        }
        self.rebuild_render_resources();
        self.say("Cancelled");
    }
}
