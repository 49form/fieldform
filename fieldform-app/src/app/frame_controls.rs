// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Where and which way: frames, placement, direction, outline and ends, and the drawing plane.

use crate::*;

impl App {
    /// Three labelled numeric fields, for a placement's move or turn.
    ///
    /// Separate from [`Self::centre_fields`] because a rotation is neither a length nor
    /// bounded like one: it wants degrees, its own step, and its own range.
    pub(crate) fn placement_fields(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        values: [ParamRef; 3],
        speed: f64,
        range: std::ops::RangeInclusive<f64>,
    ) -> [ParamRef; 3] {
        // Degrees when the range says so, which is the one place the two are told apart —
        // the three cells are otherwise identical, and a rotation was reading as millimetres.
        let suffix = if range == DEGREES { "\u{b0}" } else { " mm" };
        self.xyz_row(ui, label, values, speed, range, suffix)
    }

    /// One position, on one line, in columns every other position shares.
    ///
    /// **Horizontal because a position is one thing.** These were three stacked rows, one per
    /// axis, which reads as three numbers that happen to be near each other — and made a curve's
    /// two ends six rows deep with nothing lining up. On one line with fixed columns, a position
    /// reads as a position and two of them read as a pair.
    ///
    /// Each cell is still a full `parametric_field`, so any single coordinate can be named and
    /// driven exactly as before; the V simply sits inside its column instead of after a row of
    /// its own.
    pub(crate) fn xyz_row(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        values: [ParamRef; 3],
        speed: f64,
        range: std::ops::RangeInclusive<f64>,
        suffix: &str,
    ) -> [ParamRef; 3] {
        self.xyz_row_naming(ui, label, values, speed, range, suffix, Naming::Offered)
    }

    /// The same, saying whether each coordinate offers to be named. See [`Naming`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn xyz_row_naming(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        values: [ParamRef; 3],
        speed: f64,
        range: std::ops::RangeInclusive<f64>,
        suffix: &str,
        naming: Naming,
    ) -> [ParamRef; 3] {
        self.xyz_row_locked(ui, label, values, speed, range, suffix, naming, None).0
    }

    /// The same, with an optional lock at the end of the row. See [`Self::size_row`].
    ///
    /// **One layout, not two.** The lock is a fourth cell in the row every other triple already
    /// draws, so a locked row and a plain one line up with each other and with Move, Turn and
    /// About. Writing a second row builder for the one triple that wanted a toggle is how two
    /// spellings of one thing start (D9).
    #[allow(clippy::too_many_arguments)]
    fn xyz_row_locked(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        values: [ParamRef; 3],
        speed: f64,
        range: std::ops::RangeInclusive<f64>,
        suffix: &str,
        naming: Naming,
        lock: Option<bool>,
    ) -> ([ParamRef; 3], bool) {
        let mut out = values;
        let mut locked = lock.unwrap_or(false);
        ui.horizontal(|ui| {
            grid_gaps(ui);
            row_label(ui, label);
            for (slot, axis) in out.iter_mut().zip(["X", "Y", "Z"]) {
                ui.allocate_ui(
                    egui::vec2(AXIS_FIELD_W, ui.spacing().interact_size.y),
                    |ui| {
                        let (edited, _) = self.number_field(
                            ui,
                            axis,
                            *slot,
                            speed,
                            range.clone(),
                            "",
                            suffix,
                            naming,
                            Labelled::Cell,
                            // A coordinate is a length, whatever the quantity around it is.
                            Grain::Continuous,
                        );
                        *slot = edited;
                    },
                );
            }
            if lock.is_some() {
                // `mode_button`, not `toggle_value`: a toggle draws no frame when it is off, so
                // this read as the word "Ratio" until it was clicked. Reported twice.
                if ui
                    .add(mode_button(locked, "Ratio"))
                    .on_hover_text(
                        "Keeps the three in proportion: drag one and the other two follow. An \
                         editing aid only — it is not saved with the document, because the \
                         three sides of a box really are three independent numbers and a lock \
                         recorded beside them would be a fourth thing describing the same three.",
                    )
                    .clicked()
                {
                    locked = !locked;
                }
            }
        });
        (out, locked)
    }

    /// A box's three sides, with a lock that keeps them in proportion.
    ///
    /// **The ratio is applied here rather than stored anywhere.** Locking reads the three
    /// numbers as they stand and, when one is dragged, moves the other two by the same factor —
    /// so the proportion is whatever it was when the drag began, and nothing has to be kept in
    /// step between edits. The node still carries three independent numbers, which is what it
    /// carries in the file.
    ///
    /// A side at zero cannot be scaled — every factor leaves it at zero — but `POSITIVE_MM`
    /// starts at a thousandth of a millimetre, so the guard is against arithmetic rather than
    /// against a shape anyone can build.
    pub(crate) fn size_row(&mut self, ui: &mut egui::Ui, id: NodeId, size: [ParamRef; 3]) -> [ParamRef; 3] {
        let was_locked = self.size_locked.contains(&id);
        let before = [self.resolve(size[0]), self.resolve(size[1]), self.resolve(size[2])];
        let (mut out, locked) = self.xyz_row_locked(
            ui,
            "Size:",
            size,
            0.5,
            POSITIVE_MM,
            " mm",
            Naming::Offered,
            Some(was_locked),
        );
        match locked {
            true => self.size_locked.insert(id),
            false => self.size_locked.remove(&id),
        };
        if !was_locked {
            return out;
        }
        // Comparing the resolved numbers rather than the references, since a named side that is
        // dragged keeps the same reference and changes the value behind it.
        let after = [self.resolve(out[0]), self.resolve(out[1]), self.resolve(out[2])];
        let Some((led, factor)) = leading_factor(before, after) else {
            return out;
        };
        for other in (0..3).filter(|&i| i != led) {
            let wanted = (before[other] * factor).clamp(*POSITIVE_MM.start(), *POSITIVE_MM.end());
            out[other] = self.set_value(out[other], wanted);
        }
        out
    }

    /// Three labelled numeric fields for a direction: which way, not where or how far.
    ///
    /// **Separate from [`Self::centre_fields`] for the reason a rotation is**, and found the
    /// same way. A direction was being edited as three millimetre lengths, so it read
    /// "1.00 mm" for what is a ratio, and dragged at half a unit per pixel -- a unit vector
    /// swung right past every direction worth having before the pointer had moved a centimetre.
    /// A sweep and a frame's axis are the same kind of thing and now take the same editor,
    /// which is what stops the next one being a third spelling.
    fn direction_fields(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        values: [ParamRef; 3],
    ) -> [ParamRef; 3] {
        // No suffix: a direction has no unit. A hundredth per pixel puts the whole span from
        // one axis to another inside a comfortable drag.
        self.xyz_row(ui, label, values, 0.01, DIRECTION, "")
    }

    /// The active drawing plane as a [`Frame`], or `None` for the world.
    ///
    /// Converting to a `Frame` rather than resolving here is what makes this cost nothing:
    /// `resolve_frame` already knows how to read a typed plane and how to read one taken from a
    /// block, and going through it means the plane you draw on is read by exactly the same code
    /// that reads the plane a revolve turns about. Two readings would be two answers.
    pub(crate) fn drawing_frame(&self) -> Option<fieldform_scene::Frame> {
        drawing_frame_of(self.drawing_plane)
    }

    /// A point given in the drawing plane's own coordinates, put where it belongs in the world.
    ///
    /// The identity when no plane is active, which is why every caller can apply it
    /// unconditionally: a curve built with no plane set comes out exactly where it did before
    /// this existed.
    pub(crate) fn into_drawing_plane(&self, local: Point3<f64>) -> Point3<f64> {
        let Some(frame) = self.drawing_frame() else { return local };
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        match fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers) {
            Ok((translation, rotation)) => Point3::from(rotation * local.coords + translation),
            // A plane block that has gone away leaves the point where it was rather than
            // dropping it at the origin. The op that set it is what reports the problem.
            Err(_) => local,
        }
    }

    /// Which way the active drawing plane faces, in world coordinates.
    ///
    /// Read through the same `resolve_frame` the geometry is built with, so the direction the
    /// camera swings to and the plane the points land on cannot part company.
    pub(crate) fn drawing_normal(&self) -> Option<Vector3<f64>> {
        let frame = self.drawing_frame()?;
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        let (_, rotation) =
            fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers).ok()?;
        Some(rotation * Vector3::z())
    }

    /// Sets or clears the plane new curves are built on.
    ///
    /// Refuses rather than guesses: a word that is not a plane, or `on=selected` with no plane
    /// block selected, says so and changes nothing. Setting the plane changes where everything
    /// you draw next ends up, so silently doing something else is the expensive failure.
    pub(crate) fn set_drawing_plane(&mut self, args: &Args) {
        let on = args
            .get("on")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "xy".to_string());
        let offset_mm = args.get("at").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let chosen = match on.to_lowercase().as_str() {
            "none" | "off" | "world" => Some(DrawingPlane::None),
            "selected" => match self.selected_body {
                Some(id)
                    if matches!(self.part_graph().node(id), Some(Node::Plane { .. })) =>
                {
                    Some(DrawingPlane::Node(id))
                }
                Some(_) => {
                    self.warn("That block is not a plane — select a plane, or say `draw on=xy`");
                    return;
                }
                None => {
                    self.warn("Nothing selected — select a plane block, or say `draw on=xy`");
                    return;
                }
            },
            word => plane_normal(word).map(|normal| DrawingPlane::World { normal, offset_mm }),
        };

        let Some(chosen) = chosen else {
            self.warn(format!(
                "{on:?} is not a plane — try xy, xz, yz, selected, or none"
            ));
            return;
        };

        self.use_drawing_plane(chosen, Swing::Yes);
    }

    /// Sets the plane and says so, turning the camera to look down it unless told not to.
    ///
    /// The two callers want opposite things about the camera and the same thing about
    /// everything else, which is why this is one function with a flag rather than two that
    /// would drift apart.
    pub(crate) fn use_drawing_plane(&mut self, chosen: DrawingPlane, swing: Swing) {
        self.drawing_plane = chosen;
        match chosen {
            DrawingPlane::None => {
                // Nothing to look at squarely, so the camera stays where the user left it.
                // Clearing the plane is not a request to move the view.
                self.say("Drawing in the world again — new curves go where they always did")
            }
            other => {
                // **Picking a plane and orienting to it is one act, not two** — you turned to
                // look at the back because that is where you are about to draw
                // (`sketcher-spec.md` §5). Drawing on a plane you are looking at edge-on is a
                // way of placing points you cannot see.
                let turned = if swing == Swing::Yes
                    && let Some(normal) = self.drawing_normal()
                {
                    let normal = normal.cast::<f32>();
                    self.camera.begin_snap_toward(normal, up_for_normal(normal));
                    " and the view has turned to look down it"
                } else {
                    ""
                };
                let where_ = other.label(self);
                self.say(format!(
                    "Drawing on {where_} — new curves are built in it{turned}"
                ))
            }
        }
    }

    /// Which way a frame points, resolved for the panel's own arithmetic.
    ///
    /// Through the kernel's `resolve_frame`, so the number the panel reasons about is the one
    /// the geometry is built from. A second reading of a frame is a second answer, and the two
    /// would part company the moment a frame is taken from a plane rather than typed.
    pub(crate) fn frame_axis(&self, frame: &fieldform_scene::Frame) -> nalgebra::Vector3<f64> {
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::resolve_frame(&self.doc, self.part_graph(), frame, &numbers)
            .map(|(_, rotation)| rotation * nalgebra::Vector3::z())
            .unwrap_or_else(|_| nalgebra::Vector3::z())
    }

    /// The editor for a [`Frame`], used by every block that has one.
    ///
    /// **One widget, not one per block (D9).** Two panels showing the same thing drift, and
    /// teach two habits for one idea — so the block that gains a frame gains its controls too,
    /// with the same words in the same order and the same shortcuts.
    ///
    /// `axis_label` varies, because the direction means something different per block: a revolve
    /// turns about an *axis*, a spherical lattice radiates from a *pole*, an extrude runs *along*
    /// a direction. The same number, named for the job.
    ///
    /// `typed` is what the **Typed** option should produce when it is chosen.
    ///
    /// **Because "typed" does not mean the same thing to every block.** For a revolve or a twist
    /// any aim is a legitimate aim, so switching back to typed should keep whatever the frame was
    /// already pointing along — snapping it to the world would throw away an aim somebody set. An
    /// extrude is different: its profile is *read* in this frame, so the only typed frame that
    /// works is the profile's own plane, and keeping the last aim keeps a broken one. Going back
    /// and forth between the two options has to leave the block working, which is what the owner
    /// asked for and is only true if the caller decides what going back means.
    pub(crate) fn frame_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        axis_label: &str,
        frame: fieldform_scene::Frame,
        typed: Option<fieldform_scene::Frame>,
    ) -> fieldform_scene::Frame {
        // Blocks a frame can be taken *from*. Offered only when there is one: an empty picker
        // is a promise the document cannot keep.
        //
        // **A point as well as a plane.** A plane carries a place and a facing, which is a whole
        // frame; a point carries the place and leaves the facing at `+Z`. Asked for directly --
        // draw a point and put another block's origin on it, so moving the point moves
        // everything aimed at it. `resolve_frame` is where that is decided, and the field and
        // the shader both read it there, so this list only has to agree with one answer.
        let sources: Vec<(NodeId, String)> = self.sources_for(id);

        // Every curve, not only the straight ones. A curve's ends are what a frame reads, and
        // refusing a curved one would be refusing an answer that is perfectly well defined —
        // it aims from where the curve starts to where it stops. Whether that is what somebody
        // wanted is their business; the tooltip says what it does.
        // Every curve outside this block's own profile -- see `axis_candidates` for the one
        // thing that is still withheld and the one that stopped being.
        let lines: Vec<(CurveId, String)> = self.axis_candidates(id);

        let mut out = frame;
        let mut make_axis_line = false;
        let mut make_aim_plane = false;
        // **Always, empty document or not.** The whole row used to appear only once there was
        // something to point at, so a block in a new document had no aim control at all and
        // nothing said one existed. Made a rule at the owner's request: "it appears you hide
        // features in blocks until geometry is available -- please don't do that ever ... user
        // needs to see what's possible inside blocks". Greyed out with a reason on hover is what
        // an unavailable option looks like from here on.
        {
            ui.horizontal(|ui| {
                row_label(ui, "Aim:");
                if ui
                    .add(mode_button(matches!(out, fieldform_scene::Frame::At { .. }), "Manual"))
                    .on_hover_text("A point and a direction, entered here.")
                    .clicked()
                    && !matches!(out, fieldform_scene::Frame::At { .. })
                {
                    // What the caller says typed means; failing that, wherever the frame is
                    // pointing now. Never `Frame::default()`, which is the world XY plane and
                    // was how an extrude drawn in any other view became unrecoverable: look at
                    // the typed option, look away, and the plane it was fitted to was gone.
                    let numbers = |r: &ParamRef| Some(self.resolve(*r));
                    let here = fieldform_field::resolve_frame(
                        &self.doc,
                        self.part_graph(),
                        &out,
                        &numbers,
                    )
                    .map(|(origin, rotation)| {
                        let axis = rotation * Vector3::z();
                        fieldform_scene::Frame::At {
                            origin: [
                                ParamRef::Literal(origin.x),
                                ParamRef::Literal(origin.y),
                                ParamRef::Literal(origin.z),
                            ],
                            axis: [
                                ParamRef::Literal(axis.x),
                                ParamRef::Literal(axis.y),
                                ParamRef::Literal(axis.z),
                            ],
                        }
                    })
                    .unwrap_or_default();
                    out = typed.unwrap_or(here);
                }
                let taken = matches!(out, fieldform_scene::Frame::From { .. });
                // Offered and disabled rather than absent when there is nothing to read, and
                // clicking it with nothing there *makes* one -- the same answer the Line button
                // below has given since it was written, and for the same reason: an option that
                // leads nowhere teaches that the block cannot do it.
                let picked_plane = ui
                    .add(mode_button(taken, "Plane"))
                    .on_hover_text(
                        "Read from a plane in the document. Move that plane and this \
                         follows it — several blocks aimed at one move together, \
                         which is what copying the numbers cannot do. A plane gives the \
                         place and the facing; a point gives the place and \
                         leaves the facing upright. With no plane in the document, this \
                         makes one facing the way this block already points.",
                    )
                    .clicked();
                if picked_plane && !taken {
                    let a_plane = sources.iter().find(|(n, _)| {
                        matches!(self.part_graph().node(*n), Some(Node::Plane { .. }))
                    });
                    match a_plane.or_else(|| sources.first()) {
                        Some((node, _)) => out = fieldform_scene::Frame::From { node: *node },
                        None => make_aim_plane = true,
                    }
                }
                let along = matches!(out, fieldform_scene::Frame::Along { .. });
                // Offered whether or not there is a line: with none, clicking makes one along
                // the axis this block already has, so the option always leads somewhere.
                let picked = ui
                    .add(mode_button(along, "Line"))
                    .on_hover_text(
                        "From a curve: it starts where the curve starts and points the \
                         way the curve runs, end to end. Draw the line an axis should be \
                         on, and every block aimed at it moves when the line moves.",
                    )
                    .clicked();
                // **Between two blocks**, which is how a direction is said when there is no
                // line to draw: "I should be able to do that with a vector or a straight curve
                // by supplying either a curve or two points". Backlog items 18 and 25, both of
                // which asked for it and both of which parked on settling the reference once --
                // so it is a `Frame` variant, and every aimed block has it at the same moment.
                //
                // Needs two blocks that can answer a place. Offered and disabled when there are
                // not two, with the reason on hover, like everything else here.
                let between = matches!(out, fieldform_scene::Frame::Between { .. });
                if ui
                    .add_enabled(sources.len() >= 2 || between, mode_button(between, "Points"))
                    .on_hover_text(
                        "From one block to another: it sits at the first and points at the \
                         second. Move either and this re-aims — which is how you say a \
                         direction there is no line for.",
                    )
                    .on_disabled_hover_text(
                        "Needs two blocks to point between. Add a second point or plane and \
                         this becomes available.",
                    )
                    .clicked()
                    && !between
                    && sources.len() >= 2
                {
                    out = fieldform_scene::Frame::Between {
                        from: sources[0].0,
                        to: sources[1].0,
                    };
                }
                // **A line that works, not the first one.** Clicking lands on whichever line
                // this picks, so landing on one the compile refuses puts the option straight
                // back where it was, before the list below can be reached at all. When none of
                // them works, switching at all would do exactly that -- so it says why instead,
                // which is the one outcome that leaves somebody knowing what to do next.
                if picked && !along {
                    match lines.first() {
                        Some((curve, _)) => {
                            out = fieldform_scene::Frame::Along { curve: *curve }
                        }
                        // **Make one rather than do nothing.** With no line in the document
                        // there is nothing to point at, and the option would either be dead or
                        // absent. Building one along the axis the block already has keeps it
                        // pointing exactly where it was pointing; what it gains is ends that
                        // can be dragged.
                        None => make_axis_line = true,
                    }
                }
            });
        }

        // Outside the closure, where `self` can be borrowed again to add a block.
        if make_aim_plane {
            let plane = self.plane_facing(out);
            out = fieldform_scene::Frame::From { node: plane };
        }
        if make_axis_line {
            match self.axis_as_line(id, out) {
                Some(curve) => {
                    out = fieldform_scene::Frame::Along { curve };
                    self.say(
                        "Made a line along this axis — drag its ends to aim the block",
                    );
                }
                None => self.warn(
                    "Could not read this block's axis, so there is nothing to make a line \
                     from. Type an origin and a direction first.",
                ),
            }
        }

        match out {
            fieldform_scene::Frame::At { origin, axis } => {
                let origin = self.centre_fields(ui, "Origin:", origin);
                let mut axis = self.direction_fields(ui, axis_label, axis);
                ui.horizontal(|ui| {
                    // An empty label, which allocates the column without writing in it. These
                    // buttons are shortcuts for the row directly above, so they belong under the
                    // fields they fill in — and every other row in a block reaches the field
                    // column through this same call. Skipping it put them against the panel's
                    // left edge, which is the one place nothing else in a block starts.
                    grid_gaps(ui);
                    row_label(ui, "");
                    for (label, v) in [
                        ("X", [1.0, 0.0, 0.0]),
                        ("Y", [0.0, 1.0, 0.0]),
                        ("Z", [0.0, 0.0, 1.0]),
                    ] {
                        if ui
                            .small_button(label)
                            .on_hover_text("Point it down this axis")
                            .clicked()
                        {
                            axis = v.map(ParamRef::Literal);
                        }
                    }
                })
                .response
                .on_hover_text(
                    "The three the answer usually is. Anything else can be typed above, and the \
                     vector need not be unit — it is normalised where it is used.",
                );
                out = fieldform_scene::Frame::At { origin, axis };
            }
            fieldform_scene::Frame::From { node } => {
                ui.horizontal(|ui| {
                    row_label(ui, "Block:");
                    let offered = self.block_choices(&sources);
                    let shown = self.shown_label(&offered, node, "—");
                    if let Some(picked) = self.reference_field(
                        ui,
                        (id, "frame_plane"),
                        node,
                        &shown,
                        &offered,
                        RefKind::Block,
                        DropsFrom::block(|n| n),
                    ) {
                        out = fieldform_scene::Frame::From { node: picked };
                    }
                });
                // The block the frame is taken from, under the row that names it (§7.1, §17).
                self.host_nested_block(ui, node);
            }
            fieldform_scene::Frame::Between { from, to } => {
                // Two rows rather than one with two boxes, so each end is picked the way every
                // other operand in the application is picked.
                for (label, current, first) in
                    [("From:", from, true), ("Toward:", to, false)]
                {
                    ui.horizontal(|ui| {
                        row_label(ui, label);
                        let offered = self.block_choices(&sources);
                        let shown = self.shown_label(&offered, current, "—");
                        if let Some(picked) = self.reference_field(
                            ui,
                            (id, label),
                            current,
                            &shown,
                            &offered,
                            RefKind::Block,
                            DropsFrom::block(|n| n),
                        ) {
                            out = if first {
                                fieldform_scene::Frame::Between { from: picked, to }
                            } else {
                                fieldform_scene::Frame::Between { from, to: picked }
                            };
                        }
                    });
                    // Each end under its own row, which is why the two ends are two rows.
                    self.host_nested_block(ui, current);
                }
                // Where it actually points, since the answer comes from geometry rather than
                // from anything typed here -- and two ends in the same place have no direction
                // between them, which is worth saying before it reads as the block ignoring you.
                let numbers = |r: &ParamRef| Some(self.resolve(*r));
                let at = |node| {
                    fieldform_field::resolve_place(
                        &self.doc,
                        self.part_graph(),
                        &fieldform_scene::Place::From { node },
                        &numbers,
                    )
                    .unwrap_or_else(|_| Point3::origin())
                };
                let (a, b) = (at(from), at(to));
                let aim = b - a;
                if aim.norm() < 1e-9 {
                    ui.label(
                        egui::RichText::new(
                            "Those two are in the same place, so there is no direction between \
                             them and this points up.",
                        )
                        .color(ui.visuals().warn_fg_color),
                    );
                } else {
                    ui.label(
                        egui::RichText::new(format!(
                            "from ({:.1}, {:.1}, {:.1}) toward ({:.2}, {:.2}, {:.2})",
                            a.x, a.y, a.z, aim.x, aim.y, aim.z
                        ))
                        .weak(),
                    );
                }
            }
            fieldform_scene::Frame::Tangent { curve } => {
                ui.horizontal(|ui| {
                    row_label(ui, "Across:");
                    let offered = self.curve_choices(&lines);
                    let shown = self.shown_label(&offered, curve, "—");
                    if let Some(picked) = self.reference_field(
                        ui,
                        (id, "frame_tangent"),
                        curve,
                        &shown,
                        &offered,
                        RefKind::Curve,
                        DropsFrom::curve(|c| c),
                    ) {
                        out = fieldform_scene::Frame::Tangent { curve: picked };
                    }
                });
                // The curve itself, under the row that names it (§7.1, §17).
                self.host_nested_curve(ui, curve);
                ui.horizontal(|ui| {
                    row_label(ui, "Read:");
                    if ui
                        .add(mode_button(false, "Along it"))
                        .on_hover_text("The line it spans, end to end")
                        .clicked()
                    {
                        out = fieldform_scene::Frame::Along { curve };
                    }
                    ui.add(mode_button(true, "Across it"))
                        .on_hover_text("Square across it, where it starts");
                });
                ui.label(
                    egui::RichText::new(
                        "Square across the curve where it starts. The angles are read from the \
                         curve, not typed — redraw it and this follows.",
                    )
                    .small()
                    .weak(),
                );
            }
            fieldform_scene::Frame::Along { curve } => {
                ui.horizontal(|ui| {
                    row_label(ui, "Line:");
                    let offered = self.curve_choices(&lines);
                    let shown = self.shown_label(&offered, curve, "—");
                    if let Some(picked) = self.reference_field(
                        ui,
                        (id, "frame_line"),
                        curve,
                        &shown,
                        &offered,
                        RefKind::Curve,
                        DropsFrom::curve(|c| c),
                    ) {
                        out = fieldform_scene::Frame::Along { curve: picked };
                    }
                });
                // The line itself, under the row that names it (§7.1, §17).
                self.host_nested_curve(ui, curve);
                ui.horizontal(|ui| {
                    row_label(ui, "Read:");
                    ui.add(mode_button(true, "Along it"))
                        .on_hover_text("The line it spans, end to end");
                    if ui
                        .add(mode_button(false, "Across it"))
                        .on_hover_text(
                            "Square across it, where it starts — which is what a plane \
                             standing over a curved path wants, since the two part company the \
                             moment the curve is not straight",
                        )
                        .clicked()
                    {
                        out = fieldform_scene::Frame::Tangent { curve };
                    }
                });
                // Where it actually points, since the answer comes from geometry rather than
                // from a number typed here, and a line drawn the other way round is the first
                // thing to check when a pattern turns the wrong way.
                if let Some(c) = self.doc.curves.get(&curve) {
                    let (from, to) = c.ends();
                    let aim = to - from;
                    ui.label(
                        egui::RichText::new(format!(
                            "from ({:.1}, {:.1}, {:.1}) toward ({:.2}, {:.2}, {:.2})",
                            from.x, from.y, from.z, aim.x, aim.y, aim.z
                        ))
                        .weak(),
                    );
                }
            }
        }
        out
    }

    /// Which way an extrude sweeps its profile.
    ///
    /// **Not the frame editor.** An extrude used to take a `Frame`, which set the plane its
    /// profile was read in *and* the direction it was thickened along — one control for two
    /// things, so aiming it moved the plane out from under the profile and the block stopped
    /// compiling. The plane is the profile's own now and is not editable because it is not a
    /// choice; what is left is a direction, and that is what this edits.
    ///
    /// The exact case is a distinct option rather than a direction that happens to be the
    /// normal, and it says so, because that is the difference between a field that is exact and
    /// one that is only nearly.
    pub(crate) fn sweep_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        sweep: fieldform_scene::Sweep,
    ) -> fieldform_scene::Sweep {
        // Typed starts at the profile's own normal, so choosing it changes nothing until a
        // number is edited — the same reason the frame editor keeps where it is.
        let normal = self.profile_normal(id).unwrap_or_else(Vector3::z);
        // **Not every curve: not the profile's own.** The walk that finds the profile loop is
        // told to ignore whichever curve the sweep names, so naming one of the loop's own
        // curves deletes the profile it is describing, and the block refuses with "that curve
        // has no length to build a profile from". Clicking "Along a line" offered whichever
        // curve the map yielded first, which for a single-curve profile is the profile itself
        // -- so the option sprang straight back off every time it was clicked.
        let in_profile = self.profile_curves(id);
        self.direction_controls(
            ui,
            (id, "sweep"),
            "Sweep:",
            sweep,
            Some(normal),
            SWEEP_WORDS,
            &in_profile,
            Some(id),
        )
    }

    /// The one editor for "which way": an extrude's sweep and the draft tool's pull both come
    /// through here, so a line picked for one is the pick the other makes — D9's "one editor,
    /// not one per block". `straight_out` is what `Sweep::Normal` would mean, offered only where
    /// there is a profile for it to mean anything; a pull direction has no such thing.
    /// `exclude` are curves that may not be picked — a profile's own, for the sweep that
    /// would otherwise run along itself — and `lean_of` is the block whose lean is reported
    /// when the direction it chose is not exact.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn direction_controls(
        &mut self,
        ui: &mut egui::Ui,
        salt: impl std::hash::Hash + Copy,
        label: &str,
        sweep: fieldform_scene::Sweep,
        straight_out: Option<Vector3<f64>>,
        words: DirectionWords,
        exclude: &[CurveId],
        lean_of: Option<NodeId>,
    ) -> fieldform_scene::Sweep {
        let lines: Vec<(CurveId, String)> = self
            .doc
            .curves
            .values()
            .filter(|c| !exclude.contains(&c.id))
            .map(|c| (c.id, c.name.clone()))
            .collect();

        let mut out = sweep;
        ui.horizontal(|ui| {
            ui.label(label);
            if straight_out.is_some()
                && ui
                    .add(mode_button(
                        matches!(out, fieldform_scene::Sweep::Normal),
                        "Normal",
                    ))
                    .on_hover_text(
                        "Along the profile's own normal. The only sweep that is an exact \
                         distance field.",
                    )
                    .clicked()
            {
                out = fieldform_scene::Sweep::Normal;
            }
            let along = matches!(out, fieldform_scene::Sweep::Along { .. });
            // Offered and disabled rather than absent when there is nothing to pick: a
            // control that vanishes leaves nothing to hover, so the reason cannot be read
            // anywhere — and hidden, the draft tool looked as though it took typed numbers
            // only. The way to make a line stands beside it.
            let picked = ui
                .add_enabled(!lines.is_empty(), mode_button(along, words.along))
                .on_hover_text(words.along_hint)
                .on_disabled_hover_text(words.none_hint)
                .clicked();
            if lines.is_empty()
                && ui
                    .small_button("New line")
                    .on_hover_text(
                        "Adds a line — from the origin, along X — and points this \
                         along it. Drag its ends to set the direction.",
                    )
                    .clicked()
                && let Some(id) = self.add_line()
            {
                out = fieldform_scene::Sweep::Along { curve: id };
            }
            if picked && !along {
                out = fieldform_scene::Sweep::Along { curve: lines[0].0 };
            }
            let toward = matches!(out, fieldform_scene::Sweep::Toward { .. });
            if ui
                .add(mode_button(toward, "Manual"))
                .on_hover_text("A direction entered here. Need not be unit length.")
                .clicked()
                && !toward
            {
                let normal = straight_out.unwrap_or_else(Vector3::z);
                out = fieldform_scene::Sweep::Toward {
                    xyz: [
                        ParamRef::Literal(normal.x),
                        ParamRef::Literal(normal.y),
                        ParamRef::Literal(normal.z),
                    ],
                };
            }
        });

        match out {
            fieldform_scene::Sweep::Normal => {}
            fieldform_scene::Sweep::Along { curve } => {
                ui.horizontal(|ui| {
                    ui.label(words.curve_row);
                    let offered = self.curve_choices(&lines);
                    let shown = self.shown_label(&offered, curve, "—");
                    if let Some(picked) = self.reference_field(
                        ui,
                        (salt, "line"),
                        curve,
                        &shown,
                        &offered,
                        RefKind::Curve,
                        DropsFrom::curve(|c| c),
                    ) {
                        out = fieldform_scene::Sweep::Along { curve: picked };
                    }
                });
                // The curve it is swept along, under the row that names it (§7.1, §17).
                self.host_nested_curve(ui, curve);
            }
            fieldform_scene::Sweep::Toward { xyz } => {
                let edited = self.direction_fields(ui, "Toward:", xyz);
                out = fieldform_scene::Sweep::Toward { xyz: edited };
            }
        }

        // **Said where the sweep is chosen**, which is the rule D8's reversal left standing: an
        // approximation announces itself where it is caused, not where it is eventually noticed.
        //
        // With a number, because "approximate" on its own is not something anybody can act on:
        // the first question it provokes is how far off, and the answer was nowhere. The lean is
        // this block's own and is recomputed as it is edited; what the lean costs is measured
        // once, by `fieldform-field/examples/lean_error.rs`, and read out of the table
        // `lean_shortfall` carries.
        if let Some(id) = lean_of
            && !out.is_exact()
        {
            let lean = fieldform_field::sweep_lean_degrees(&self.doc, self.part_graph(), id);
            let text = match lean {
                Some(degrees) => {
                    // **Says the consequence, not the ratio.** This read "reads as little as
                    // 49% of the true distance", which is true of the field and was taken --
                    // reasonably -- to mean the solid came out 49% of the depth typed, since
                    // that is the number the sentence sits beside. The size is exact at every
                    // lean, and `lean_error.rs` measures that alongside the shortfall.
                    //
                    // It then said "about 2x the radius typed", which was also wrong, just
                    // less obviously: a leaning operand does not *scale* a fillet. The blend
                    // is tangent where the reading reaches the radius, so a body reading short
                    // is tangent further out on one side only -- the arc becomes an ellipse
                    // with semi-axes `k/s` and `k`. It spreads, and it stops having a radius.
                    // `fillet_under_lean` carries that, and `blend_over_a_bound.rs` measures
                    // the real `smooth_min` against it.
                    let (spread, _, _) = fieldform_field::fillet_under_lean(1.0, degrees);
                    format!(
                        "\u{2248} leans {degrees:.0}\u{b0} — size exact, but a fillet \
                         here spreads {spread:.1}\u{d7} and is no longer one radius"
                    )
                }
                None => "\u{2248} approximate".to_string(),
            };
            let detail = match lean {
                Some(degrees) => {
                    let keeps = fieldform_field::lean_shortfall(degrees);
                    let (spread, tight, flat) = fieldform_field::fillet_under_lean(5.0, degrees);
                    format!(
                        "**The solid is the size you asked for.** Its depth is measured along \
                         the sweep and it is exactly that deep at every lean — \
                         measured, not assumed, by `lean_error.rs`. The surface is exactly \
                         where it should be, so cutting, joining, patterning, mirroring and \
                         measuring this body are all exact: they ask only which side of the \
                         surface a point is on.\n\n\
                         What leans is the *field around* it. A shear makes a distance read \
                         short — as little as {:.0}% of the room there really is \
                         — which is safe, since a field may understate and never \
                         overstate, but it is what a fillet, an offset and a shell measure.\n\n\
                         **A fillet here is not simply larger; it is a different shape.** A 5 \
                         mm one spreads {:.1} mm across the body it joins instead of 5, stays 5 \
                         on this one, and turns through radii from {:.1} mm to {:.1} mm rather \
                         than 5 throughout. Note the tight end is *under* what was asked, which \
                         matters if the number was chosen for a tool or a stress \
                         concentration.\n\n\
                         The lean is the angle between the sweep and the profile's own normal, \
                         so a profile drawn square to the line has none — straight out \
                         is exact.",
                        keeps * 100.0,
                        spread,
                        tight,
                        flat,
                    )
                }
                None => "The solid is the size you asked for. A leaning sweep is a shear, \
                         though, and a sheared distance reads short — so this body \
                         reports less room beside it than there really is, and blends against \
                         it come out larger than the radius asked for. Straight out is exact."
                    .to_string(),
            };
            ui.label(egui::RichText::new(text).color(ui.visuals().warn_fg_color))
                .on_hover_text(detail);
        }
        out
    }

    /// Turns a block's current axis into a line in the document, and aims the block at it.
    ///
    /// **The missing step between a typed axis and a picked one.** A revolve's axis has to be
    /// coplanar with its profile and clear of it -- that is inherent, since the profile is read
    /// in `(r, z)` and so must lie in a half-plane containing the axis -- and a line drawn from
    /// the Create menu lands at the drawing plane's origin running across it. So "draw a line
    /// and pick it" failed for a reason no message could fix: there was no way to get a line
    /// *where the axis already is*.
    ///
    /// This makes one there. The block keeps pointing exactly where it pointed, so nothing about
    /// the shape changes; what changes is that the axis is now a thing in the viewport with ends
    /// that can be dragged, and the block follows it. That is the whole point of aiming a block
    /// at a line, and it was unreachable.
    ///
    /// Sized to the profile it belongs to, with a margin, so it is visible beside the body
    /// rather than a dot in the middle of it or a line off the edge of the world.
    fn axis_as_line(&mut self, id: NodeId, frame: fieldform_scene::Frame) -> Option<CurveId> {
        let numbers = |r: &ParamRef| {
            r.resolve(&|pid| self.doc.parameter_value(pid))
        };
        let (translation, rotation) =
            fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers).ok()?;
        let origin = Point3::from(translation);
        let direction = rotation * Vector3::z();

        // As long as the profile it belongs to, so both ends are reachable beside the body
        // rather than buried inside it.
        // Through the chain rather than the loop, so a wall's open curve has a reach too; for a
        // loop the two are the same walk.
        let seed = match self.part_graph().node(id) {
            Some(Node::Revolve { outline, .. }) | Some(Node::Extrude { outline, .. }) => {
                Some(outline.curve())
            }
            _ => None,
        };
        let reach = seed
            .and_then(|curve| {
                fieldform_field::outline_chain(&self.doc, curve, frame.curve_source()).ok()
            })
            .map(|found| {
                found.points.iter().map(|p| (p - origin).norm()).fold(0.0f64, f64::max)
            });
        let half = reach.unwrap_or(DEFAULT_BODY_MM * 0.5).max(DEFAULT_BODY_MM * 0.25) * 1.2;

        let curve = fieldform_scene::Curve::polyline(
            CurveId::new(),
            self.unique_curve_name("Axis"),
            vec![origin - direction * half, origin + direction * half],
            false,
        )
        .ok()?;
        let curve_id = curve.id;
        self.doc.curves.insert(curve_id, curve);
        // Placed now, as everything written past the history is (`block-ownership-spec.md`
        // §4.1), rather than reading as free until some later edit settles.
        self.doc.settle_ownership();
        self.rebuild_curve_display();
        Some(curve_id)
    }

    /// The one editor for an [`Outline`](fieldform_scene::Outline): which curve, whether it is
    /// a wall, and for a wall how thick, to which side and how it stops.
    ///
    /// **One editor, not one per block** (D9). An extrude, a revolve and a sweep are built on
    /// the same thing, so their three panels show it through here, and a change to how a wall
    /// is entered is a change to all three at once. It replaces the picker each of them called
    /// and a sentence that described a wall nothing in the application could yet make.
    ///
    /// **Side is three choices in a row, not the flip toggle the spec drew**, as a sweep's
    /// Section is: one click reaches any of them and the one in force is the one lit. Round a
    /// loop the three are Centre, Inside and Outside, because that is what a person means
    /// there. The document still holds left and right, and which of those is the inside is the
    /// loop's winding — asked of the wall the field builds ([`Self::inside_side`]), so the
    /// word shown is true of the wall drawn.
    ///
    /// **Ends is greyed on a loop rather than hidden.** The spec shows it only for an open
    /// curve; the owner's later rule is that a block does not hide what it can do, so the row
    /// stays and says why it does nothing there.
    pub(crate) fn outline_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        outline: fieldform_scene::Outline,
    ) -> fieldform_scene::Outline {
        use fieldform_scene::{Outline, WallEnds, WallSide};
        let mut out = outline;

        // Which curve. A profile's loop or a wall's chain is walked from it either way, and
        // changing it leaves the kind alone: a wall pointed at another curve is still a wall.
        let curve = out.curve();
        let label = if out.is_wall() { "Curve:" } else { "Profile:" };
        let picked = self.pick_profile(ui, id, curve, label);
        if picked != curve {
            // **Inside stays inside.** Which of left and right is a loop's inside depends on
            // the way the loop is walked, and that on which of its curves the block names. So
            // a wall shown as Inside is asked which side it was really on before its curve is
            // changed, and is put on that side of the new one — or the same click that picks a
            // curve would flip the wall to the other face of the loop. The picker itself hands
            // back a loop's first curve for any of its others, so without this, opening the
            // block of a wall read from a file could flip it.
            let stood = match out {
                Outline::Wall { side, .. } if side != WallSide::Centre => {
                    self.inside_side(id, curve).map(|inside| side == inside)
                }
                _ => None,
            };
            *out.curve_mut() = picked;
            if let (Some(was_inside), Outline::Wall { side, .. }) = (stood, &mut out)
                && let Some(inside) = self.inside_side(id, picked)
            {
                *side = if was_inside { inside } else { across(inside) };
            }
        }

        let mut walled = out.is_wall();
        ui.horizontal(|ui| {
            row_label(ui, "");
            ui.checkbox(&mut walled, "Wall").on_hover_text(
                "Gives the curve a thickness instead of filling the loop it closes — so the \
                 curve need not close. A thin extrude, a shell of revolution, an open section \
                 carried along a path.",
            );
        });
        if walled != out.is_wall() {
            out = if walled {
                Outline::Wall {
                    curve: out.curve(),
                    // The last thickness a wall was given, as an offset repeats its distance.
                    thickness: ParamRef::Literal(self.wall_thickness),
                    side: WallSide::Centre,
                    ends: WallEnds::Square,
                }
            } else {
                Outline::closed(out.curve())
            };
        }

        if let Outline::Wall { curve, thickness, side, ends } = out {
            let (thickness, _) = self.parametric_field(
                ui,
                "Thickness:",
                thickness,
                0.5,
                POSITIVE_MM,
                "The whole thickness of the band, whichever side of the curve it is on.",
            );
            // `None` for a curve that closes no loop, which is what makes the row below say
            // Left and Right, and what leaves Ends something to choose.
            let inside = self.inside_side(id, curve);
            let mut side = side;
            ui.horizontal(|ui| {
                row_label(ui, "Side:");
                for (choice, word, hover) in side_choices(inside) {
                    ui.selectable_value(&mut side, choice, word).on_hover_text(hover);
                }
            });
            let mut ends = ends;
            ui.horizontal(|ui| {
                row_label(ui, "Ends:");
                ui.add_enabled_ui(inside.is_none(), |ui| {
                    for choice in [WallEnds::Square, WallEnds::Round] {
                        ui.selectable_value(&mut ends, choice, choice.label())
                            .on_hover_text(match choice {
                                WallEnds::Square => "Cut square to the curve, at its end.",
                                WallEnds::Round => {
                                    "A half-round of the wall's own thickness beyond each end."
                                }
                            })
                            .on_disabled_hover_text(
                                "A loop has no ends: its wall runs all the way round.",
                            );
                    }
                });
            });
            out = Outline::Wall { curve, thickness, side, ends };
            // Remembered for the next wall, once it is a number somebody typed here.
            if let ParamRef::Literal(typed) = thickness
                && out != outline
            {
                self.wall_thickness = typed;
            }
        }
        out
    }

    /// Which of left and right is the inside of this block's section, were it built on
    /// `curve` — or `None` where that curve closes no loop, and so has a left and a right and
    /// two ends.
    ///
    /// **Asked of the wall the field would build**, on the block as it would be, because the
    /// answer is not the curve's alone: a loop is laid in an extrude's plane, in a sweep's, or
    /// in the half-plane about a revolve's axis, and its winding is read there. A probe wall
    /// whose own numbers do not matter stands in for whatever the block holds.
    ///
    /// By the curves' ends first, which tessellates nothing. Most walls are on open curves, and
    /// this is asked every frame a wall's block is open; only a loop pays for a walk.
    fn inside_side(&self, id: NodeId, curve: CurveId) -> Option<fieldform_scene::WallSide> {
        self.inside_of(self.part_graph().node(id)?, curve)
    }

    /// The same, of a block that need not be in the graph yet — which is where a command that
    /// is about to make one has it.
    pub(crate) fn inside_of(&self, block: &Node, curve: CurveId) -> Option<fieldform_scene::WallSide> {
        use fieldform_scene::{Outline, WallEnds, WallSide};
        let mut block = block.clone();
        let closes = self.doc.curves.get(&curve)?.is_closed()
            || fieldform_field::loop_from(&self.doc, curve, block.aimed_by()).is_ok();
        if !closes {
            return None;
        }
        *block.outline_mut()? = Outline::Wall {
            curve,
            thickness: ParamRef::Literal(1.0),
            side: WallSide::Left,
            ends: WallEnds::Square,
        };
        fieldform_field::block_wall(
            &self.doc,
            self.part_graph(),
            &block,
            fieldform_field::PROFILE_TURN_DEGREES,
        )?
        .inside()
    }

    /// How far a revolve's profile spreads around its axis, said where the axis is chosen.
    ///
    /// **Reported rather than refused, which is a reversal.** A profile used to have to lie in
    /// one half-plane about its axis, and anything else was refused at compile time. That rule
    /// was doing more harm than good: nudging the axis line a millimetre out of the profile's
    /// plane stopped the block compiling, and coplanarity is not something anybody can see or
    /// hold by hand.
    ///
    /// What is left is worth saying, though, because it is the one thing about a revolve you
    /// cannot read off the screen: past a few degrees the solid's cross-section stops being the
    /// profile and becomes the profile's *shadow* in cylindrical coordinates. Near zero they are
    /// the same thing. So this is a number that grows quietly rather than a wall.
    pub(crate) fn revolve_spread(&mut self, ui: &mut egui::Ui, id: NodeId) {
        let Some(Node::Revolve { outline, frame, .. }) = self.part_graph().node(id).cloned() else {
            return;
        };
        // The chain rather than the loop, so a wall's open curve is measured too; for a profile
        // the two are the same walk.
        let Ok(found) =
            fieldform_field::outline_chain(&self.doc, outline.curve(), frame.curve_source())
        else {
            return;
        };
        let numbers = |r: &ParamRef| {
            r.resolve(&|pid| self.doc.parameter_value(pid))
        };
        let Ok((translation, rotation)) =
            fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers)
        else {
            return;
        };
        let into = move |p: Point3<f64>| {
            Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
        };
        let spread = fieldform_field::Profile::azimuth_span(&found.points, into);

        // A degree of slack for a curve drawn by hand: below that it is in the axis's plane as
        // far as anybody could tell, and saying so would be noise on every well-formed revolve.
        if spread < 1.0 {
            return;
        }
        ui.label(
            egui::RichText::new(format!("\u{2248} profile spreads {spread:.0}\u{b0} about the axis"))
                .color(ui.visuals().warn_fg_color),
        )
        .on_hover_text(
            "The profile is not in the axis's own plane, so what is revolved is its shadow in \
             cylindrical coordinates rather than the shape as drawn — a section held out \
             to the side turns into a ring.\n\n\
             A few degrees is indistinguishable from none. The further this goes, the less the \
             cross-section resembles the profile, and past about a half turn the far side folds \
             onto the near one. The field stays an exact distance to whatever solid results; \
             what drifts is which solid that is.\n\n\
             Move the axis into the plane the profile was drawn on to take this to zero.",
        );
    }

    /// The curves offered as an axis for a block.
    ///
    /// Only one filter now: the curves the block's own profile is made of, because naming one
    /// deletes the profile it is describing -- that is not a choice with a bad outcome, it is
    /// not a choice.
    ///
    /// There was a second, and it was wrong twice over. It withheld any line a revolve's profile
    /// straddled, which emptied the list and made the option read as missing; and the rule it
    /// was enforcing has since been dropped, because insisting a profile stay coplanar with its
    /// axis refused ordinary work -- see `Profile::about_axis`. An axis off the plane now builds
    /// a solid, and the panel says how far off it is rather than the picker hiding the choice.
    pub(crate) fn axis_candidates(&self, id: NodeId) -> Vec<(CurveId, String)> {
        let in_profile = self.profile_curves(id);
        self.doc
            .curves
            .values()
            .filter(|c| !in_profile.contains(&c.id))
            .map(|c| (c.id, c.name.clone()))
            .collect()
    }

    /// The curves an extrude's profile loop is made of.
    ///
    /// Empty when the loop cannot be walked at all, which is the right answer for the one thing
    /// this is for: a list of curves not to offer as the line to sweep along.
    fn profile_curves(&self, id: NodeId) -> Vec<CurveId> {
        // Both blocks that are built from a profile, and each excludes the curve it is aimed
        // by from the walk that finds it -- so for both, the curve it is aimed by must not be
        // one of the profile's own.
        let (profile, aimed_by) = match self.part_graph().node(id) {
            Some(Node::Extrude { outline, sweep, .. }) => (outline.curve(), sweep.source()),
            Some(Node::Revolve { outline, frame, .. }) => (outline.curve(), frame.curve_source()),
            _ => return Vec::new(),
        };
        // The chain, which for a profile is its loop and for a wall is whatever runs on from
        // its curve end to end — every curve of which is equally not a line to sweep along.
        fieldform_field::outline_chain(&self.doc, profile, aimed_by)
            .map(|found| found.curves)
            .unwrap_or_default()
    }

    /// Which way an extrude's profile faces, if it has one that can be read.
    fn profile_normal(&self, id: NodeId) -> Option<Vector3<f64>> {
        let Some(Node::Extrude { outline, sweep, .. }) = self.part_graph().node(id) else {
            return None;
        };
        // From the function the field lays the section down with, so the answer is the plane
        // the solid is actually in — a wall's as much as a profile's.
        fieldform_field::extrude_section(
            &self.doc,
            outline,
            sweep,
            fieldform_field::PROFILE_TURN_DEGREES,
        )
        .ok()
        .map(|section| section.normal)
    }

    /// Which curve a profile block starts from, as a picker over every curve in the document.
    ///
    /// **It used to offer only the closed ones**, reasoning that an open curve has no inside and
    /// would be refused the moment it was chosen. That was true while a profile was one curve.
    /// It is not any more: `profile_loop` walks outward from whichever curve is named here
    /// through ends that meet, so three open curves forming a triangle are a profile and any one
    /// of them will do as the seed. Filtering them out hid the whole feature.
    ///
    /// A curve that leads nowhere is still refused — by the compile, which can say how big the
    /// gap is and where it is, which is something to go and fix rather than a rule. Unless the
    /// block is built on a wall, which is what an open curve is for: `label` is what the row is
    /// called, "Profile:" for a loop that is filled and "Curve:" for one given a thickness.
    fn pick_profile(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        current: CurveId,
        label: &str,
    ) -> CurveId {
        // **The rows first, since a row is what a profile is.** This listed every curve by
        // name, which was right while a loop of four lines was four rows and wrong the moment
        // it became one: "Line, Line 2, Line 3, Line 4" is a list you cannot pick a profile
        // from. A row is offered as itself and hands the block its first curve, which is what
        // the walk starts from; the curves outside any row follow, as they were.
        self.refresh_profiles();
        let mut closed: Vec<(CurveId, String)> = self
            .profiles
            .loops
            .iter()
            .map(|l| (l[0], format!("{} ({} curves)", self.profile_label(l[0]), l.len())))
            .collect();
        let mut loose: Vec<(CurveId, String)> = self
            .doc
            .curves
            .values()
            .filter(|c| self.profiles.key_of(c.id).is_none())
            .map(|c| (c.id, c.name.clone()))
            .collect();
        loose.sort_by_key(|(id, _)| {
            (self.doc.curves_created.get(id).copied().unwrap_or(0), id.raw())
        });
        closed.extend(loose);
        // A member picked before the rows existed still reads its loop, so it is shown as the
        // row it is in.
        let current = self.profiles.key_of(current).unwrap_or(current);

        let mut out = current;
        ui.horizontal(|ui| {
            row_label(ui, label);
            if closed.is_empty() {
                ui.label(
                    egui::RichText::new("no curves in this document")
                        .weak(),
                )
                .on_hover_text(
                    "A profile is a closed loop — one closed curve, or several open \
                     ones whose ends meet. Draw one and it will appear here.",
                );
                return;
            }
            let offered = self.curve_choices(&closed);
            let shown = self.shown_label(&offered, current, "—");
            if let Some(picked) = self.reference_field(
                ui,
                (id, "profile"),
                current,
                &shown,
                &offered,
                RefKind::Curve,
                DropsFrom::curve(|c| c),
            ) {
                out = picked;
            }
        });
        // **The profile itself, under the row that names it** (§7.1, §17) -- the row for the
        // whole loop, claimed by the loop rather than by the curve this block happens to name
        // ([`App::host_nested_curve`]).
        self.host_nested_curve(ui, current);
        out
    }

    /// The one editor for a pair of [`Ends`], used by every block that has one.
    ///
    /// **One editor, not one per block**, which is the second of D9's three requirements and the
    /// one most easily skipped: two panels showing the same thing drift apart, and teach two
    /// habits for one idea. The cylinder, the cone and the capsule all come through here, so a
    /// change to how ends are entered is a change to all three at once.
    ///
    /// It moved the radii below the ends on the cone, which used to interleave them — base
    /// point, base radius, top point, top radius. That reads well and is exactly the shape that
    /// cannot be shared, so it went: the ends are one thing now, and the radii follow.
    /// `id` is the block these ends belong to, so each end can be offered the blocks it may
    /// sit on without being offered anything that consumes it.
    pub(crate) fn ends_controls(&mut self, ui: &mut egui::Ui, id: NodeId, ends: Ends) -> Ends {
        // Every line, and every block that has a facing. Both lists are *always shown*, empty or
        // not, and an empty one says so in a disabled entry rather than vanishing: a control that
        // is not there cannot say why, and what a block can do should be readable from the block.
        // Made a rule at the owner's request -- "it appears you hide features in blocks until
        // geometry is available; please don't do that ever, just grey it out".
        let curves: Vec<(fieldform_scene::CurveId, String)> = self.axis_candidates(id);
        let planes: Vec<(NodeId, String)> = self.sources_for(id);

        let mut out = ends;
        ui.horizontal(|ui| {
            // **"Axis", because that is what these two points are for.** A cylinder, a cone and
            // a capsule are each a shape about a line, and "Ends" named the storage rather than
            // the thing being set.
            row_label(ui, "Axis:");
            let selected = match out {
                // **"Manual", not "typed".** One word for this option in every block that has
                // it, because it is one idea: you are entering the value by hand.
                Ends::At { .. } => "Manual".to_string(),
                Ends::From { curve } => self
                    .doc
                    .curves
                    .get(&curve)
                    .map(|c| c.name.clone())
                    // A curve that has gone missing still has to say something, and saying so is
                    // more use than saying nothing.
                    .unwrap_or_else(|| "missing line".to_string()),
                Ends::On { node, .. } => self.body_label(node),
            };
            egui::ComboBox::from_id_salt("ends_source")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    // Switching back to manual keeps where the block *is* rather than snapping it
                    // to a default: the numbers start as whatever it was following, so nothing
                    // moves at the moment it stops following.
                    let manual = match out {
                        Ends::At { .. } => out,
                        _ => {
                            let (a, b) = self.resolved_ends(&out);
                            Ends::between(
                                fieldform_scene::Place::At(Self::as_literals(a)),
                                fieldform_scene::Place::At(Self::as_literals(b)),
                            )
                        }
                    };
                    ui.selectable_value(&mut out, manual, "Manual")
                        .on_hover_text("Two points, entered here");

                    ui.separator();
                    ui.label(egui::RichText::new("on a plane").weak());
                    if planes.is_empty() {
                        ui.add_enabled(false, egui::Button::new("no plane yet").frame(false))
                            .on_disabled_hover_text(
                                "Add a plane and it appears here. This block would then stand on \
                                 its origin and point along its normal, and follow it when it \
                                 moves.",
                            );
                    }
                    for (n, name) in &planes {
                        let length = {
                            let (a, b) = self.resolved_ends(&out);
                            let span = (b - a).norm();
                            // Its current length, so standing it on a plane re-aims it without
                            // also resizing it. A block with no length yet gets one it can be
                            // seen at rather than nothing.
                            ParamRef::Literal(if span > 1e-9 { span } else { DEFAULT_BODY_MM })
                        };
                        let here = matches!(out, Ends::On { node, .. } if node == *n);
                        if ui
                            .selectable_label(here, name)
                            .on_hover_text(
                                "Stands on this block: from its origin, along the way it faces. \
                                 Move or turn it and this follows.",
                            )
                            .clicked()
                        {
                            out = Ends::On { node: *n, length };
                        }
                    }

                    ui.separator();
                    ui.label(egui::RichText::new("along a line").weak());
                    if curves.is_empty() {
                        ui.add_enabled(false, egui::Button::new("no line yet").frame(false))
                            .on_disabled_hover_text(
                                "Draw a line and it appears here. This block would then run \
                                 between its two ends.",
                            );
                    }
                    for (c, name) in &curves {
                        ui.selectable_value(&mut out, Ends::From { curve: *c }, name)
                            .on_hover_text(
                                "The two ends of this curve. The shape between them is ignored \
                                 — to follow a curve, sweep a pipe along it.",
                            );
                    }
                });
        });
        ui.separator();

        match out {
            Ends::At { from, to } => {
                let new_from = self.place_fields(ui, id, "From:", from);
                let new_to = self.place_fields(ui, id, "To:", to);
                // Radius and length are what these shapes are usually given as, so both are
                // typeable even though only the ends are stored.
                //
                // **Only while both ends are typed.** Typing a length moves the far end, and an
                // end sitting on a point is not this block's to move -- the point is where it
                // is, and quietly rewriting it here would move every other body on that point.
                // The field is left out rather than shown doing nothing, because the ends
                // themselves say plainly where they are taken from.
                let new_to = match (new_from.typed(), new_to.typed()) {
                    (Some(a), Some(b)) => {
                        fieldform_scene::Place::At(self.length_field(ui, a, b))
                    }
                    _ => new_to,
                };
                Ends::At { from: new_from, to: new_to }
            }
            Ends::On { node, length } => {
                // The one number that is still this block's own. Where it starts and which way
                // it runs belong to the plane; how far it goes belongs here.
                let (new_length, _) = self.parametric_field(
                    ui,
                    "Length:",
                    length,
                    0.5,
                    POSITIVE_MM,
                    "Along the plane's normal, from its origin",
                );
                let (a, b) = self.resolved_ends(&Ends::On { node, length: new_length });
                ui.label(
                    egui::RichText::new(format!(
                        "from ({:.2}, {:.2}, {:.2})  to ({:.2}, {:.2}, {:.2})",
                        a.x, a.y, a.z, b.x, b.y, b.z
                    ))
                    .weak(),
                );
                Ends::On { node, length: new_length }
            }
            Ends::From { curve } => {
                // Read-only, because the curve is where these numbers live now. Shown at all
                // because "which way round is it" is the first question, and because a block
                // that has quietly followed its line somewhere unexpected says so here.
                let (a, b) = self.resolved_ends(&out);
                ui.label(
                    egui::RichText::new(format!(
                        "from ({:.2}, {:.2}, {:.2})  to ({:.2}, {:.2}, {:.2})",
                        a.x, a.y, a.z, b.x, b.y, b.z
                    ))
                    .weak(),
                );
                if (b - a).norm() < 1e-9 {
                    ui.label(
                        egui::RichText::new(
                            "That curve's ends are in the same place, so this has no length.",
                        )
                        .color(ui.visuals().warn_fg_color),
                    );
                }
                Ends::From { curve }
            }
        }
    }

    /// Where a pair of ends actually is, with parameters and curves resolved.
    pub(crate) fn resolved_ends(&self, ends: &Ends) -> (Point3<f64>, Point3<f64>) {
        match ends {
            Ends::At { from, to } => (self.resolve_place(from), self.resolve_place(to)),
            Ends::From { curve } => self
                .doc
                .curves
                .get(curve)
                .map(|c| c.ends())
                .unwrap_or((Point3::origin(), Point3::origin())),
            // Through the kernel, which is the one reading of where a plane is and which way it
            // faces -- so the numbers the panel prints are the ones the shape is built from.
            Ends::On { .. } => {
                let numbers = |r: &ParamRef| self.resolve(*r);
                fieldform_field::ends_at(ends, &self.doc, self.part_graph(), &numbers)
            }
        }
    }

/// A resolved point, back as three literals.
///
/// What "switch this block from following a line to being typed" needs: the numbers start where
/// the block already is, so it does not jump at the moment it stops following.
pub(crate) fn as_literals(p: Point3<f64>) -> [ParamRef; 3] {
    [
        ParamRef::Literal(p.x),
        ParamRef::Literal(p.y),
        ParamRef::Literal(p.z),
    ]
}

    /// The distance between a shape's two ends, as a number you can type.
    ///
    /// Derived on the way in and applied by *moving a point* on the way out, rather than
    /// stored: a length kept beside the two ends would be a second description of the same
    /// thing, and dragging an end would leave it stale. So this reads the ends, and typing
    /// into it slides the far one along the axis they already define.
    ///
    /// Returns the far end, changed or not.
    fn length_field(
        &mut self,
        ui: &mut egui::Ui,
        from: [ParamRef; 3],
        to: [ParamRef; 3],
    ) -> [ParamRef; 3] {
        let (a, b) = (self.resolve_point(&from), self.resolve_point(&to));
        let mut length = (b - a).norm();
        let before = length;
        ui.horizontal(|ui| {
            row_label(ui, "Length:");
            ui.add(egui::DragValue::new(&mut length).speed(0.5).suffix(" mm"))
                .on_hover_text("Between the two ends — moves the far one along the axis");
        });
        if (length - before).abs() < 1e-12 {
            return to;
        }
        // No axis to grow along when the ends sit together; +Z is where a new one arrives.
        let direction = (b - a).try_normalize(1e-12).unwrap_or_else(Vector3::z);
        let end = a + direction * length.max(0.001);
        [ParamRef::Literal(end.x), ParamRef::Literal(end.y), ParamRef::Literal(end.z)]
    }

    /// A primitive's centre, as three parametric numbers.
    ///
    /// Three fields rather than one, because each is a reference in its own right: a document
    /// can drive one axis from a named parameter and leave the others as plain numbers.
    pub(crate) fn centre_fields(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        centre: [ParamRef; 3],
    ) -> [ParamRef; 3] {
        self.xyz_row(ui, label, centre, 0.5, ANYWHERE_MM, " mm")
    }
}
