// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The one door every command comes through, typed or clicked.

use crate::*;

impl App {
    /// Runs a named operation. Every key binding funnels through here, and the eventual
    /// Python API will too — so a scripted call and a key press cannot diverge in behaviour.
    ///
    /// Operations split into two kinds. *Document* operations mutate `doc` and go through
    /// the registry and the undo stack. *Application* operations (undo itself, view changes)
    /// act on transient UI state that isn't part of the saved document and isn't undoable —
    /// handled here directly, since putting them on the undo stack would mean undoing an
    /// undo.
    pub(crate) fn dispatch(&mut self, op: &str, args: &Args) {
        // Starting something else ends a placement in progress, and says so. A curve left
        // waiting under another tool would be a click that means something nobody can see,
        // which is the trap the "no modes" rule exists to avoid.
        //
        // `modify.*` too, now that the tool stays on between curves (item 111): drawing a profile
        // and extruding it is the ordinary sequence, and an extrude that left the tool armed would
        // make the next click in the viewport a line nobody meant.
        if (op.starts_with("create.") || op.starts_with("modify."))
            && !matches!(op, "create.line" | "create.curve")
            && let Some(pending) = self.curve_ui.pending.take()
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
                let pull = typed_direction(&self.doc, &self.view.draft.pull);
                match step_history(&mut self.stack, &mut self.doc, &mut self.gesture, true) {
                    Some(name) => {
                        keep_typed_direction(&self.doc, &mut self.view.draft.pull, pull);
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
                let pull = typed_direction(&self.doc, &self.view.draft.pull);
                match step_history(&mut self.stack, &mut self.doc, &mut self.gesture, false) {
                    Some(name) => {
                        keep_typed_direction(&self.doc, &mut self.view.draft.pull, pull);
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
            "create.expression" => self.expression_command(args),
            "create.sphere" => self.add_primitive_with(PrimitiveKind::Sphere, args),
            "create.cube" => self.add_primitive_with(PrimitiveKind::Cube, args),
            "create.skewbox" => self.add_primitive_with(PrimitiveKind::SkewedBox, args),
            "create.cylinder" => self.add_primitive_with(PrimitiveKind::Cylinder, args),
            "create.cone" => self.add_primitive_with(PrimitiveKind::Cone, args),
            "create.torus" => self.add_primitive_with(PrimitiveKind::Torus, args),
            "create.capsule" => self.add_primitive_with(PrimitiveKind::Capsule, args),
            "create.coil" => self.add_primitive_with(PrimitiveKind::Coil, args),
            "create.thread" => self.add_primitive_with(PrimitiveKind::Thread, args),
            "create.plane" => self.add_primitive_with(PrimitiveKind::Plane, args),
            // Gyroid on a rectangular map when nothing says otherwise, and changeable in the
            // block afterwards — the same reasoning as Create > Curve, which stopped asking for
            // a degree up front once the degree could be changed without losing the shape. The
            // cell, the map and its numbers, the phase and its own number, and the frame are all
            // read by their own names (`console_keys::lattice_with_keys`); a cell, map or phase
            // that does not exist is refused by name rather than quietly made a gyroid.
            "create.tpms" => self.add_lattice_with(args),
            // `point on=<body>` stands on a surface; any other `point` is a place in space.
            "create.point" if args.get("on").is_some() => self.point_on_surface_command(args),
            "create.point" => self.add_primitive_with(PrimitiveKind::Point, args),
            "create.surfacepoint" => self.begin_axis_placement_for(AxisKind::SurfacePoint, None),
            "create.surfaceplane" => self.begin_axis_placement_for(AxisKind::SurfacePlane, None),
            "point.fix" => self.surface_point_live_command(args, false),
            "point.live" => self.surface_point_live_command(args, true),
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
                // `extrude profile=none`, or nothing in the document to build on: the block
                // is made and waits (`App::add_waiting`).
                if self.waiting_asked(&args) {
                    self.add_waiting(CurveBlock::Extrude, &args);
                } else if wall_wanted(&args) {
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
                if self.waiting_asked(&args) {
                    self.add_waiting(CurveBlock::Revolve, &args);
                } else if wall_wanted(&args) {
                    self.add_wall_solid(WallBlock::Revolve, &args);
                } else {
                    self.add_profile_solid(true);
                }
                self.apply_revolve_options(&args);
            }
            // `sweep profile=p spine=path section=skewed`, `sweep curve=l1 wall=1 spine=path`.
            // Its own arm, ahead of the two below, so the curves are named before either runs.
            "create.sweep" if console_keys::names_a_curve(args) => self.sweep_with_keys(args),
            // Nothing drawn to sweep or to sweep along: the block is made and waits.
            "create.sweep" if self.doc.curves.is_empty() => {
                self.add_waiting(CurveBlock::Sweep, args);
                self.apply_sweep_options(args);
            }
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
            "image.block" => self.saveimage_command(args),
            "evaluate.comb" => self.comb_command(args),
            "image.save_again" => self.save_image_again(),
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
            "modify.infill" => self.apply_infill(args),
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
            // `lock target=Sphere_0` and `unlock target=Sphere_0` state it, so a line run twice
            // leaves the block as asked. Alone, `lock` turns the selected block's lock over, as
            // `hide` does its eye.
            "block.lock" => match (args.get("target").map(|v| v.text()), self.selected_body) {
                (Some(name), _) => self.set_locked_named("lock", &name, true),
                (None, Some(id)) => self.set_locked(id, !self.doc.is_locked(id)),
                (None, None) => self.warn("`lock` names what to lock: `lock target=Sphere_0`"),
            },
            "block.unlock" => match (args.get("target").map(|v| v.text()), self.selected_body) {
                (Some(name), _) => self.set_locked_named("unlock", &name, false),
                (None, Some(id)) => self.set_locked(id, false),
                (None, None) => self.warn("`unlock` names what to unlock: `unlock target=Sphere_0`"),
            },
            "edit.copy" => self.copy_selection(true),
            "edit.cut" => self.cut_selection(),
            "edit.duplicate" => self.duplicate_selection(),
            "edit.paste" => self.paste(true),
            "edit.paste_here" => self.paste_at_hover(),
            "profile.join" => self.join_selected(),
            "profile.ungroup" => self.ungroup_selected(),
            "profile.align" => self.align_command(args),
            "curve.continuity" => self.continuity_command(args),
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
            // `hide target=Sphere_0` hides that block, and `show target=…` draws it: by name, so a
            // script does not depend on what happens to be selected, and stated rather than turned
            // over, so running the line twice leaves it as asked. Alone, `hide` is the H key's
            // toggle of the selection, as it always was.
            "view.toggle_visibility" => match args.get("target").map(|v| v.text()) {
                None => self.toggle_selected_visibility(),
                Some(name) => self.set_visibility_named("hide", &name, false),
            },
            "view.show" => match args.get("target").map(|v| v.text()) {
                Some(name) => self.set_visibility_named("show", &name, true),
                None => match self.selected_body {
                    Some(id) if !self.is_visible(id) => self.toggle_visibility(id),
                    Some(id) => self.say(format!("{} is already drawn", self.body_label(id))),
                    None => self.warn("`show` names what to draw: `show target=Sphere_0`"),
                },
            },
            "view.cycle_mode" => {
                self.view.mode = self.view.mode.next();
                // Which imported solids are drawn as triangles depends on the mode, and
                // nothing else about the part has changed — so this refreshes the buffers
                // without paying for a compile.
                self.refresh_drawn_meshes();
                // Said at the moment the question arises — this is the keystroke someone
                // presses *because* they want the cage — rather than left as a viewport that
                // is silently empty and looks broken.
                if self.view.mode.cage_is_editable() && !self.control_visible() {
                    self.say(cage_absence_reason(self));
                }
            }
            // One command for the viewport and the graph: whichever the panel shows is framed.
            "view.frame_selection" if self.doc.graph_layout.shown => {
                self.graph_view.request_frame(self.selected_body);
            }
            "view.frame_all" if self.doc.graph_layout.shown => self.graph_view.request_frame(None),
            "view.frame_all" => match self.scene_bounds() {
                Some((center, radius)) => self.camera.frame(center, radius),
                None => self.warn("Nothing to frame"),
            },
            "view.panel" => self.panel_from_console(args),
            "view.find" => self.find_from_console(args),
            "view.graph_layout" => {
                self.doc.graph_layout.pins.clear();
                self.say("Graph laid out again");
            }
            "view.frame_selection" => match self.framing_bounds() {
                Some((center, radius)) => self.camera.frame(center, radius),
                None => self.warn("Nothing to frame"),
            },
            "draw.plane" => self.set_drawing_plane(args),
            "draw.sketch" => self.sketch_on_command(args),
            "curve.degree" => self.curve_degree_command(args),
            "curve.degree_menu" => self.open_degree_menu(),
            "part.set_material" => self.set_material_from_console(args),
            "part.critical" => self.critical_from_console(args),
            "part.verify" => self.verify_from_console(),
            "cage.check" => self.check_from_console(),
            "select.loop" | "select.ring" | "select.grow" | "select.shrink" => self.select_along(op),
            "view.environment" => self.set_environment_from_console(args),
            "view.draft" => self.draft_from_console(args),
            "view.grid" => self.grid_from_console(args),
            "view.perf" => self.perf_from_console(args),
            "view.display" => self.display_from_console(args),
            // The View menu's settings that had no operation of their own: a tick box flipped a
            // field and nothing else could reach it, which P3 calls a bug -- the GUI could do
            // something the console could not (`interaction-spec.md` §1).
            "view.control_points" => {
                self.curve_ui.show_control_points = !self.curve_ui.show_control_points;
                self.say(if self.curve_ui.show_control_points {
                    "Every curve's control points are shown"
                } else {
                    "Only the selected curve's control points are shown"
                });
            }
            "edit.allow_agent" => self.toggle_allow_agent(),
            "view.view_cube" => {
                self.view.show_view_cube = !self.view.show_view_cube;
                self.say(if self.view.show_view_cube { "ViewCube shown" } else { "ViewCube hidden" });
            }
            "view.tool_panel" => {
                self.tools.floating_panel = !self.tools.floating_panel;
                self.say(if self.tools.floating_panel {
                    "A pending block shows the floating tool panel"
                } else {
                    "A pending block opens in the list, its first number waiting to be typed"
                });
            }
            "view.idle_occlusion" => {
                if !self.view.quality.occlusion() {
                    self.warn(
                        "Display quality is Fast, which never traces occlusion or shadows — choose                          Balanced or Full first",
                    );
                } else {
                    self.view.idle_occlusion = !self.view.idle_occlusion;
                    self.traced.invalidate(traced::Quality::Final);
                    self.say(if self.view.idle_occlusion {
                        "Shadows and occlusion once the view is still"
                    } else {
                        "No shadows or occlusion, even when still"
                    });
                }
            }
            "view.idle_reflections" => {
                if !self.view.quality.reflections() {
                    self.warn(
                        "Display quality is Fast, which never traces reflections — choose Balanced or \
                         Full first",
                    );
                } else {
                    self.view.idle_reflections = !self.view.idle_reflections;
                    self.traced.invalidate(traced::Quality::Final);
                    self.say(if self.view.idle_reflections {
                        "Reflections traced once the view is still"
                    } else {
                        "No traced reflections, even when still"
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
            "console.clear" => self.console.log.clear(),
            "view.section" => self.toggle_section(),
            "view.section_new" => self.section_new_command(args),
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
            "create.hole" => self.hole_command(args),
            _ => {
                // A cage verb acts on the cage on screen unless told which, as `refine` does: the
                // registry takes the only cage and refuses to guess among several, and the app
                // has the answer it lacks.
                // And on the elements selected, where a key or a menu ran it without naming any.
                let with_cage;
                let args = if op.starts_with("cage.") {
                    let mut filled = self.cage_selection_args(op, args);
                    if filled.get("cage").is_none() {
                        filled = filled.with("cage", Value::Str(self.active_cage.to_string()));
                    }
                    with_cage = filled;
                    &with_cage
                } else {
                    args
                };
                // Said in the user's terms, rather than left to a refusal for a missing key.
                if matches!(op, "cage.crease" | "cage.uncrease") && args.get("edges").is_none() {
                    let what = if op == "cage.crease" { "crease" } else { "uncrease" };
                    self.warn(format!("select the edges to {what} first: click an edge, or double-click one for its whole loop"));
                    return;
                }
                match self.registry.build(op, args, &self.doc) {
                    Ok(cmd) => {
                        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                            self.warn(e.to_string());
                        } else {
                            if op.starts_with("cage.") {
                                self.console.cage_line = app::cage_tools::cage_console_line(op, args, self.doc.cages.len() > 1);
                            }
                            // A topology edit renumbers what a selection names, as a refine does.
                            // A crease keeps every number, and the edges stay picked so the
                            // next step -- uncrease, crease the next loop -- can read them.
                            if op.starts_with("cage.") && !matches!(op, "cage.set_positions" | "cage.crease" | "cage.uncrease") {
                                self.forget_cage_selection();
                            }
                            self.after_document_change();
                        }
                    }
                    Err(e) => {
                        self.console.line_error = Some(outcome::from_op_error(&e));
                        self.warn(e.to_string());
                    }
                }
            }
        }
    }
}
