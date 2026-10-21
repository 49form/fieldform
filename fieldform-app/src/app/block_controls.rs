// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A block's panel: what `node_block_controls` draws for each kind, and the drivers it wires.

use crate::*;
use fieldform_scene::Operand;

impl App {
    /// Turns a fixed offset into a varying one, building the ramp and its driver.
    ///
    /// The driver is a plane, because that is the one field whose reading is obvious before
    /// you have seen it: it is the signed distance along an axis, so "thicker toward the top"
    /// is a plane facing +Z and an output range running the way you want it. Anything else in
    /// the graph can be swapped in afterwards.
    ///
    /// The plane is placed at the *bottom* of the body and the range runs to its top, so the
    /// default already varies across the body rather than sitting flat at one end of the ramp
    /// — a first result that does nothing is indistinguishable from one that is broken.
    fn drive_offset(&mut self, offset: NodeId, input: Operand) {
        // The one left standing by a previous switch-off, before making a second. See
        // `spare_ramp` -- leaving a ramp behind and then building another every time is what
        // piled them up.
        if let Some(spare) = self.spare_ramp() {
            self.wire_driver(offset, 0, spare);
            return;
        }
        // Through `operand_extent` rather than unwrapping the slot here, so a hole answers
        // `Empty` in the one place that decides what a hole's extent is and this cannot reach a
        // different conclusion about the same slot than the extent walk does.
        let extent = fieldform_field::operand_extent(&self.doc, self.part_graph(), &input);
        let (lo, hi) = match extent.as_box() {
            Some(box_) => box_,
            // No box to measure -- an unbounded body, or a slot with nothing in it yet -- so the
            // range is the default body size about the origin. Still a working ramp, just not
            // one fitted to anything, which is the honest answer for a block with nothing to fit
            // to; switching back to one number is how it is undone either way.
            None => (
                Point3::new(0.0, 0.0, -DEFAULT_BODY_MM * 0.5),
                Point3::new(0.0, 0.0, DEFAULT_BODY_MM * 0.5),
            ),
        };
        let height = (hi.z - lo.z).abs().max(1.0);
        let was = self.resolve(match self.part_graph().node(offset) {
            Some(Node::Offset { distance, .. }) => *distance,
            _ => ParamRef::Literal(0.0),
        });
        // A range that is only ever zero would be an empty output, which is a ramp that does
        // nothing while looking like it should.
        let reach = if was.abs() < 1e-9 { DEFAULT_BODY_MM * 0.05 } else { was };

        let mut graph = self.part_graph().clone();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(lo.z),
            ]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(height),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(reach),
            shape: RampShape::Smooth,
        });
        if let Some(Node::Offset { distance, .. }) = graph.node_mut(offset) {
            *distance = ParamRef::Field(ramp);
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say("Offset varies along Z — edit the ramp to change what drives it");
    }

    /// Makes a pipe's radius vary along the curve it sweeps.
    ///
    /// **A plane facing the way the curve runs**, so the ramp reads distance *along* the pipe
    /// and the radius is a gradient from one end to the other — the offset's reasoning rather
    /// than the morph's, for the same reason: a radius profile is a gradient along an axis, and
    /// the axis a pipe has is its own spine.
    ///
    /// The plane's facing is `Turn::Along` that same curve, so re-drawing the curve re-aims the
    /// ramp with it rather than leaving the gradient pointing where the curve used to go. That
    /// is the whole reason a facing can name a curve.
    ///
    /// Re-point the ramp at a *point* and the profile becomes a peak instead — fat where the
    /// point is, thin away from it — which is the other half of what the backlog asked for and
    /// needs no second mechanism.
    fn drive_pipe(&mut self, pipe: NodeId, curve: CurveId) {
        // The one left standing by a previous switch-off, before making a second. The same rule
        // the offset and the morph follow.
        if let Some(spare) = self.spare_ramp() {
            self.wire_driver(pipe, 0, spare);
            return;
        }
        let Some(c) = self.doc.curves.get(&curve) else {
            self.warn("That pipe has no curve to vary along");
            return;
        };
        let (start, end) = c.ends();
        let along = (end - start).norm().max(1.0);
        let was = self.resolve(match self.part_graph().node(pipe) {
            Some(Node::Pipe { radius, .. }) => *radius,
            _ => ParamRef::Literal(0.0),
        });
        // A range that is only ever one number would be a ramp that does nothing while looking
        // as though it should. Quarter to whole of what the pipe already was, so the switch
        // reads as a taper of the pipe on screen rather than as a resize.
        let fat = if was.abs() < 1e-9 { DEFAULT_BODY_MM * 0.05 } else { was };

        let mut graph = self.part_graph().clone();
        let plane = graph.add(Node::Plane {
            // **Where the curve begins, not where it began.** This wrote the start's
            // coordinates, so the plane's facing followed the curve and its position did not:
            // drag that end and the angle updated while the plane stayed put. Reported as
            // exactly that -- "the plane should be tied to the point of the curve".
            origin: fieldform_scene::Place::StartOf { curve },
            flipped: false,
            // **Square across the curve, not along the line between its ends.** `Turn::Along`
            // reads the chord, which is right for "where is this axis" and wrong here: a
            // degree-five spine sets off one way and ends up going another, so the plane came
            // out tilted by an angle nobody had typed and the ramp measured along the wrong
            // direction. Reported as exactly that -- "it should just take the normal of
            // whatever the curve is doing".
            turn: fieldform_scene::Turn::Tangent { curve },
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(along),
            out_min: ParamRef::Literal(fat * 0.25),
            out_max: ParamRef::Literal(fat),
            shape: RampShape::Smooth,
        });
        if let Some(Node::Pipe { radius, .. }) = graph.node_mut(pipe) {
            *radius = ParamRef::Field(ramp);
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(
            "The radius varies along the pipe — edit the ramp to change the profile, or \
             point it at a block to vary by distance from that instead",
        );
    }

    /// Makes a blend's radius vary through space, driven by a ramp off a plane
    /// (`round-chamfer-spec.md` §9.1).
    ///
    /// The offset's reasoning, for the offset's reason: a plane facing +Z at the bottom of what
    /// the blend reaches, its range running to the top, so the first result is a fillet that
    /// grows up the part -- visibly a grade rather than a resize -- and anything else in the
    /// graph can be pointed at afterwards. A ramp aimed along the seam is this ramp turned.
    pub(crate) fn drive_blend(&mut self, blend: NodeId) {
        if let Some(spare) = self.spare_ramp() {
            self.wire_driver(blend, 0, spare);
            return;
        }
        let extent = fieldform_field::graph_extent(&self.doc, self.part_graph(), blend);
        let (lo, hi) = match extent.as_box() {
            Some(box_) => box_,
            None => (
                Point3::new(0.0, 0.0, -DEFAULT_BODY_MM * 0.5),
                Point3::new(0.0, 0.0, DEFAULT_BODY_MM * 0.5),
            ),
        };
        let height = (hi.z - lo.z).abs().max(1.0);
        let was = self.resolve(match self.part_graph().node(blend) {
            Some(Node::Blend { fillet, .. }) => fillet.radius,
            _ => ParamRef::Literal(0.0),
        });
        // Half to half again what the radius was, so the switch reads as the same fillet
        // graded rather than as a different one; a radius of nought has nothing to grade from
        // and takes the default body's twentieth.
        let reach = if was.abs() < 1e-9 { DEFAULT_BODY_MM * 0.05 } else { was };

        let mut graph = self.part_graph().clone();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(lo.z),
            ]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(height),
            out_min: ParamRef::Literal(reach * 0.5),
            out_max: ParamRef::Literal(reach * 1.5),
            shape: RampShape::Smooth,
        });
        if let Some(Node::Blend { fillet, .. }) = graph.node_mut(blend) {
            fillet.radius = ParamRef::Field(ramp);
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(
            "The radius varies along Z — edit the ramp to change the grade, or point it at a \
             block to vary by distance from that instead",
        );
    }

    /// Says, in the fillet's colour, that a wall this thick for its cell has met its neighbour.
    /// Silent below the cell's own limit.
    pub(crate) fn wall_limit_note(
        &self,
        ui: &mut egui::Ui,
        kind: fieldform_field::TpmsKind,
        wall_mm: f64,
        cell_mm: f64,
    ) {
        let limit_mm = kind.wall_limit() * cell_mm;
        if !(cell_mm > 0.0 && wall_mm > limit_mm) {
            return;
        }
        ui.horizontal(|ui| {
            row_label(ui, "");
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!("\u{2248} past {limit_mm:.1} mm this cell's walls meet"),
            )
            .on_hover_text(format!(
                "A {} wall thicker than about {:.0}% of its cell has merged with the wall next \
                 to it, so the passages between them close. Measured, not estimated. A larger \
                 cell or a thinner wall keeps them apart.",
                kind.label(),
                kind.wall_limit() * 100.0
            ));
        });
    }

    /// Switches a *graded* lattice's phase and keeps it graded: the new phase takes the same
    /// ramp, and the ramp's Out range is re-seeded for the new number, since a wall in
    /// millimetres and a density as a fraction share nothing.
    ///
    /// `false`, having changed nothing, when the ramp drives something else as well -- its
    /// range is not this block's to rewrite -- and the caller falls back to a typed number.
    pub(crate) fn regrade_lattice(
        &mut self,
        lattice: NodeId,
        ramp: NodeId,
        banding: Banding,
        cell_mm: f64,
    ) -> bool {
        let shared = self
            .part_graph()
            .nodes()
            .filter(|(id, n)| *id != lattice && n.inputs().contains(&ramp))
            .count()
            > 0;
        if shared {
            self.warn("That ramp drives something else too, so the new phase starts from a typed number");
            return false;
        }
        let seeded = seeded_phase(banding, cell_mm);
        let (low, high) = grade_range(banding, self.resolve(*seeded.param()));
        let mut graph = self.part_graph().clone();
        if let Some(Node::Ramp { out_min, out_max, .. }) = graph.node_mut(ramp) {
            *out_min = ParamRef::Literal(low);
            *out_max = ParamRef::Literal(high);
        }
        if let Some(Node::Lattice { phase, .. }) = graph.node_mut(lattice) {
            let mut next = seeded;
            *next.param_mut() = ParamRef::Field(ramp);
            *phase = next;
        }
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return false;
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!("Still graded — the ramp now runs {low:.2} to {high:.2}"));
        true
    }

    /// Makes a lattice's wall, density or strut vary through the part, along Z.
    ///
    /// **A plane under the part and a ramp up through it**, which is the grade people draw
    /// first: thin at the bottom and thick at the top, or the other way round by swapping the
    /// ramp's two Out numbers. The plane is a block of its own, so dragging or turning it turns
    /// the grade, and the ramp's input can be re-pointed at a point or a body to grade by
    /// distance from that instead.
    ///
    /// Half to half again of what the lattice already was, so the switch reads as a grade of
    /// the lattice on screen rather than as a resize -- and, for a density, kept inside what the
    /// table can mean.
    pub(crate) fn drive_lattice(&mut self, lattice: NodeId) {
        let Some(Node::Lattice { phase, .. }) = self.part_graph().node(lattice).cloned() else {
            return;
        };
        // The one left standing by a previous switch-off, before making a second -- and
        // **re-ranged**, which the offset and the pipe never needed: their numbers are all
        // millimetres, and a lattice's may be a fraction. A spare drives nothing, so changing
        // its range changes nothing else.
        if let Some(spare) = self.spare_ramp() {
            let (low, high) = grade_range(phase.banding(), self.resolve(*phase.param()));
            let mut graph = self.part_graph().clone();
            if let Some(Node::Ramp { out_min, out_max, .. }) = graph.node_mut(spare) {
                *out_min = ParamRef::Literal(low);
                *out_max = ParamRef::Literal(high);
            }
            self.set_part_graph(graph);
            self.wire_driver(lattice, 0, spare);
            return;
        }
        let was = self.resolve(*phase.param());
        let (low, high) = grade_range(phase.banding(), was);
        // From the floor of whatever the part occupies to its top; a bare lattice fills space
        // and has neither, so it gets the default body's height about the origin.
        let (floor, rise) = match fieldform_field::part_extent(&self.doc, self.part_graph()) {
            fieldform_field::Extent::Bounded { lo, hi } if hi.z > lo.z => {
                (Point3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, lo.z), hi.z - lo.z)
            }
            _ => (Point3::new(0.0, 0.0, -DEFAULT_BODY_MM * 0.5), DEFAULT_BODY_MM),
        };

        let mut graph = self.part_graph().clone();
        let plane = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([
                ParamRef::Literal(floor.x),
                ParamRef::Literal(floor.y),
                ParamRef::Literal(floor.z),
            ]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let ramp = graph.add(Node::Ramp {
            input: plane.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(rise),
            out_min: ParamRef::Literal(low),
            out_max: ParamRef::Literal(high),
            shape: RampShape::Linear,
        });
        if let Some(Node::Lattice { phase, .. }) = graph.node_mut(lattice) {
            *phase.param_mut() = ParamRef::Field(ramp);
        }
        // An aimed lattice is refused by name, and says how to get the same part another way.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(
            "The lattice is graded from the bottom of the part to the top — edit the ramp \
             to change by how much, or move its plane to change which way",
        );
    }

    /// Makes a morph's amount vary through space, driven by distance from a point.
    ///
    /// **A point rather than the plane an offset gets**, because the two blocks are asked for
    /// different reasons. A driven offset usually wants a gradient along an axis; a morph wants
    /// a *region* — this shape here, that shape away from here — and the smallest thing that
    /// says "here" is a point. The point is a block of its own, so it can be dragged in the
    /// viewport and the transition follows it, and the ramp's input can be re-pointed at a
    /// curve or any other body if a line or a surface is the better centre.
    ///
    /// `Smooth` easing by default, which is the ease-in and ease-out that makes a morph read as
    /// a blend rather than as a switch. `Linear` is a visible crease at both ends of the band.
    pub(crate) fn drive_morph(&mut self, morph: NodeId, a: Operand, b: Operand) {
        // The same reach-first rule the offset follows, and for the same reason.
        if let Some(spare) = self.spare_ramp() {
            self.wire_driver(morph, 0, spare);
            return;
        }
        // And a field to read, before building one of those either. The band below is still
        // fitted to the two bodies, so a reused point gives a transition that does something
        // rather than one sitting flat at an end of its range.
        let found = self.field_to_read(Some(morph));
        // `operand_extent` for each half rather than unwrapping either slot: an empty one
        // answers `Empty` and `Empty` unions away, so the band is still fitted to whichever of
        // the two shapes is there, and to neither when neither is.
        let both = fieldform_field::operand_extent(&self.doc, self.part_graph(), &a)
            .union(fieldform_field::operand_extent(&self.doc, self.part_graph(), &b));
        let (centre, reach) = match both.as_box() {
            Some((lo, hi)) => (
                nalgebra::center(&lo, &hi),
                ((hi - lo).norm() * 0.5).max(1.0),
            ),
            None => (Point3::origin(), DEFAULT_BODY_MM * 0.5),
        };

        let mut graph = self.part_graph().clone();
        let point = match found {
            Some(existing) => existing,
            None => graph.add(Node::Point {
                at: [
                    ParamRef::Literal(centre.x),
                    ParamRef::Literal(centre.y),
                    ParamRef::Literal(centre.z),
                ],
            }),
        };
        // Nought at the point and one at the far edge of the two bodies, so the band covers the
        // shape rather than a sliver of it: a narrow band is what makes the divisor large, and
        // a default that starts there would make the viewport slow for no reason anybody chose.
        let ramp = graph.add(Node::Ramp {
            input: point.into(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(reach),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(1.0),
            shape: RampShape::Smooth,
        });
        if let Some(Node::Morph { t, .. }) = graph.node_mut(morph) {
            *t = ParamRef::Field(ramp);
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(
            "The amount now varies with distance from a point — drag the point to move \
             the transition, or edit the ramp to change its band or what drives it",
        );
    }

    /// Puts a driven number back to one number, whichever socket it is.
    ///
    /// **Whatever in this block reads that ramp, not `Offset.distance` by name.** It matched
    /// only an offset, which was true when an offset was the only driveable socket and silently
    /// wrong the moment a morph became the second: the morph panel called this, nothing matched,
    /// and "One number" left `t` driven while reporting that it had not. One function for every
    /// socket §7 marks driveable is the same rule the picker beside it follows (D9), and it is
    /// what stops the third socket repeating this.
    pub(crate) fn drop_driver(&mut self, holder: NodeId, ramp: NodeId) {
        let mut graph = self.part_graph().clone();
        let last = self.resolve_driven_at_centre(ramp);
        if let Some(node) = graph.node_mut(holder) {
            for reference in node.param_refs_mut() {
                if reference.driver() == Some(ramp) {
                    *reference = ParamRef::Literal(last);
                }
            }
        }
        // **The ramp stays.** It used to be swept away here along with the field it read,
        // because a ramp was not in the block list and one left behind was invisible and
        // permanent. `ramp-decision.md` §7.2 ends that premise: a ramp is a placeable block
        // with a row of its own, and a placed one that drives nothing is a legal state rather
        // than litter. Deleting a visible block as a side effect of a toggle is the destructive
        // behaviour the delete cascade was already changed to stop doing, and it would throw
        // away a ramp a second socket may still want. Delete it from its own row instead.
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!("{} is {last:.2} everywhere", self.body_label(holder)));
    }

    /// A ramp's midpoint output, which is the number a fixed offset inherits when the ramp goes.
    ///
    /// The midpoint rather than either end, so switching back lands between what the varying
    /// offset was doing at its extremes instead of at whichever one the code happened to pick.
    fn resolve_driven_at_centre(&self, ramp: NodeId) -> f64 {
        match self.part_graph().node(ramp) {
            Some(Node::Ramp { out_min, out_max, .. }) => {
                (self.resolve(*out_min) + self.resolve(*out_max)) * 0.5
            }
            _ => 0.0,
        }
    }

    /// A ramp's own controls, nested wherever the parameter it drives is edited.
    ///
    /// Nested rather than given the ramp a row of its own, because a ramp is not a body and a
    /// list of bodies is what the block list is. The thing a person is editing is "how this
    /// offset varies", and that is one idea in one place.
    fn ramp_controls(&mut self, ui: &mut egui::Ui, ramp: NodeId) {
        let Some(Node::Ramp { input, in_min, in_max, out_min, out_max, shape }) =
            self.part_graph().node(ramp).cloned()
        else {
            ui.label("The ramp driving this is gone.");
            return;
        };

        // **One list, because it is one question.** This was two controls — an "Of:" operand
        // picker and a separate "Or a curve:" beside it — so "what does this read" was asked
        // twice and answered in whichever of them happened to hold it, with the other showing a
        // prompt. A point, a plane, a pipe and a curve are all answers to the same question.
        // `.source()`, because the row already has somewhere to put the hole: "nothing yet"
        // is an entry rather than an absence, which a ramp's input has needed since
        // `ramp-decision.md` §7.2 and every other slot now needs too.
        self.pick_field_to_read(ui, ramp, input.source());
        ui.label(
            egui::RichText::new(
                "A ramp reads one field as a number. Below the first value the output is the \
                 first, above the second it is the second, and it eases between.",
            )
            .small()
            .weak(),
        );

        let mut new = (in_min, in_max, out_min, out_max, shape);
        ui.label("Field reads");
        let (a, _) = self.parametric_field(ui, "From:", in_min, 0.5, ANYWHERE_MM, "");
        let (b, _) = self.parametric_field(ui, "To:", in_max, 0.5, ANYWHERE_MM, "");
        // **Nought to one when this drives a morph's amount, and anything at all otherwise.**
        // A morph's `t` is a fraction between two shapes: at 7.6 the result is
        // `(1 - 7.6)*A + 7.6*B`, an extrapolation past both, which is what a *typed* amount has
        // always been refused for. The same number arriving from a ramp went unchecked, and it
        // is expensive as well as meaningless -- the divisor is `1 + sup|B - A|*L_t` and `L_t`
        // is this span over the input span, so a range seven times wider than it may be makes
        // every step seven times shorter. Reported as a viewport that stopped being usable.
        //
        // The range is what the row offers rather than a warning after the fact, so the state
        // cannot be built here at all; the compile check behind it is for a file that was
        // hand-edited.
        let drives_a_morph = self.part_graph().nodes().any(|(_, n)| {
            matches!(n, Node::Morph { .. })
                && n.param_refs().iter().any(|r| r.driver() == Some(ramp))
        });
        // **The output's unit is whatever it drives, and one of them has none.** Every number in
        // this block used to read " mm", which is right for the input side — a ramp reads a
        // *field*, and a field is a distance in millimetres whatever it is a distance to. It is
        // wrong for the output side whenever the thing being driven is not a length: a morph's
        // amount is a fraction of the way from one shape to the other, and "0.35 mm" is a
        // millimetre reading on a number that has no millimetres in it. The same mistake a
        // twist's rate and a placement's rotation were already found making.
        let (range, speed, suffix, what) = match drives_a_morph {
            true => (
                0.0..=1.0,
                0.01,
                "",
                "A fraction between the two shapes: 0 is the first, 1 is the second. It has no \
                 unit, and a morph has no meaning outside that range — which is why this \
                 row stops there.",
            ),
            false => (ANYWHERE_MM, 0.1, " mm", ""),
        };
        ui.label(match drives_a_morph {
            true => "Amount becomes",
            false => "Distance becomes",
        });
        let (c, _) =
            self.parametric_field_in(ui, "From:", out_min, speed, range.clone(), what, suffix);
        let (d, _) = self.parametric_field_in(ui, "To:", out_max, speed, range, what, suffix);
        new.0 = a;
        new.1 = b;
        new.2 = c;
        new.3 = d;

        ui.horizontal(|ui| {
            row_label(ui, "Ease:");
            for option in RampShape::ALL {
                if ui.add(mode_button(new.4 == option, option.label())).clicked() {
                    new.4 = option;
                }
            }
        });

        // An empty range has no bound at all — the mapping becomes a step — so the part stops
        // evaluating. Said here, in the panel where the two numbers are, rather than left to
        // surface as a shader that will not compile. Dragging From and To together is an
        // ordinary thing to do while scrubbing, and it used to crash the application.
        if (self.resolve(new.1) - self.resolve(new.0)).abs() < f64::EPSILON {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "The two field values are the same, so there is no range to map across.                  Move one of them.",
            );
        }

        // The bound, shown rather than hidden, because it is what the whole mechanism turns
        // on and because the user can make it large by dragging. Past 1 the surface can outrun
        // the geometry and fold through itself — not an error, and not something a divisor
        // fixes, so it is said here instead of refused.
        let slope = self.ramp_slope(ramp);
        let text = format!("Varies at up to {slope:.2} mm per mm");
        if slope >= 1.0 {
            ui.label(
                egui::RichText::new(format!(
                    "{text} — past 1 the surface can outrun the body and fold through itself"
                ))
                .small()
                .color(egui::Color32::from_rgb(210, 150, 60)),
            );
        } else {
            ui.label(egui::RichText::new(text).small().weak());
        }
        ui.label(
            egui::RichText::new("A varying offset cannot be filleted — blends need a true distance")
                .small()
                .weak(),
        );

        if new != (in_min, in_max, out_min, out_max, shape) {
            self.replace_node(
                ramp,
                Node::Ramp {
                    input,
                    in_min: new.0,
                    in_max: new.1,
                    out_min: new.2,
                    out_max: new.3,
                    shape: new.4,
                },
            );
        }
    }

    /// How fast a ramp's output can change per unit of space, which is the number a driven
    /// parameter's bound is built from.
    fn ramp_slope(&self, ramp: NodeId) -> f64 {
        fieldform_field::graph_lipschitz(&self.doc, self.part_graph(), ramp)
    }

    /// The controls for one node: what it is, in its own terms.
    pub(crate) fn node_block_body(&mut self, ui: &mut egui::Ui, id: NodeId) {
        // Scoped by the node, so a widget that comes and goes inside one block cannot renumber
        // the widgets in the blocks after it.
        //
        // egui derives a widget's id from its position within an id scope, so a *conditional*
        // widget -- the fillet row, or the "approximate" note beside it -- shifts the id of
        // everything following it the moment it appears. Between passes the ids then disagree,
        // egui says so ("changed id between passes") and paints its markers, which is the
        // stray orange rectangles in the corner. Expanding the last block was enough to do it.
        //
        // A scope per node is the fix rather than making each widget unconditional: blocks
        // legitimately show different controls, and the numbering only has to be stable
        // *within* the block it belongs to.
        ui.push_id(id, |ui| self.node_block_controls(ui, id));
    }

    fn node_block_controls(&mut self, ui: &mut egui::Ui, id: NodeId) {
        let Some(node) = self.part_graph().node(id).cloned() else { return };

        // **No name field here: the name is edited on the row.** It was the first thing in
        // every block, which meant naming a sphere took opening it, finding this, typing, and
        // closing it again -- four actions for a label that was already on screen above.
        // Double-clicking the row does it in place; see `list_row`.
        //
        // A name is a label (`block-ownership-spec.md` §6). Making a block a variable — a row of
        // its own that other blocks refer to rather than own — is the V on the row and the
        // Variable checkbox in the row's own menu, and those are the places it is written.
        ui.separator();
        match node {
            // Reachable only by selecting it deliberately — a ramp is not a body, so it is
            // not in the block list. The controls are the same ones the offset panel nests,
            // rather than a second set that could drift from them.
            Node::Ramp { .. } => self.ramp_controls(ui, id),
            Node::Cage(_) => {
                ui.label("The control cage. Edit it in the viewport.");
                // Deletable like anything else now that Create > Cage exists to put one back —
                // and it puts *this* one back, since the cage itself stays in the document
                // when the body that drew it goes.
            }
            Node::Extrude { outline, sweep, start, height, sides, draft, corners, rims } => self
                .extrude_controls(ui, id, outline, sweep, start, height, sides, draft, corners, rims),
            Node::Sweep { outline, spine, section, law, rims } => {
                self.sweep_block_controls(ui, id, outline, spine, section, law, rims)
            }
            Node::Revolve { outline, frame, angle, sides, rims } => {
                self.revolve_controls(ui, id, outline, frame, angle, sides, rims)
            }
            Node::Sphere { centre, radius } => {
                self.sphere_controls(ui, id, centre, radius)
            }
            Node::Cuboid { centre, size, turn, edge } => {
                self.cuboid_controls(ui, id, centre, size, turn, edge)
            }
            // **Three edges, each a row of the columns `At:` uses**, because that is what the
            // shape is: a corner-to-corner leaning box is three vectors from its middle, and a
            // "shear" would be a fourth number describing the same three. An edge may point any
            // way, so the range allows negatives, and a left-handed set is a box like any other.
            Node::SkewedBox { centre, edges } => {
                self.skewed_box_controls(ui, id, centre, edges)
            }
            // The two-point solids are edited as the ends they are drawn between, not as a
            // centre and an extent — the same numbers the shape is defined by, so there is
            // nothing here that has to be kept in step with anything else.
            Node::Cylinder { ends, radius, edge } => {
                self.cylinder_controls(ui, id, ends, radius, edge)
            }
            Node::Cone { ends, base_radius, top_radius, edge } => {
                self.cone_controls(ui, id, ends, base_radius, top_radius, edge)
            }
            Node::Torus { centre, major_radius, minor_radius, turn } => {
                self.torus_controls(ui, id, centre, major_radius, minor_radius, turn)
            }
            Node::Mesh(mesh) => {
                self.mesh_controls(ui, mesh)
            }
            Node::Lattice { cell, map, phase, frame } => {
                self.lattice_controls(ui, id, cell, frame, map, phase)
            }
            // Neither of these is a solid you would frame on its own, and both blocks say so
            // rather than leaving the viewport to explain it.
            Node::Plane { origin, flipped, turn } => {
                self.plane_controls(ui, id, origin, flipped, turn)
            }
            Node::Point { at } => {
                self.point_controls(ui, id, at)
            }
            Node::Capsule { ends, start_radius, end_radius } => {
                self.capsule_controls(ui, id, ends, start_radius, end_radius)
            }
            Node::Twist { input, degrees_per_mm, frame } => {
                self.twist_controls(ui, id, input, degrees_per_mm, frame)
            }
            Node::Attract { input, toward, strength, falloff } => {
                self.attract_controls(ui, id, input, toward, strength, falloff)
            }
            Node::Pattern { input, count, kind, join, fillet, options } => {
                self.pattern_controls(ui, id, input, count, kind, join, fillet, options)
            }
            Node::Mirror { input, frame, join } => {
                self.mirror_controls(ui, id, input, frame, join)
            }
            Node::Shell { input, thickness } => {
                self.shell_controls(ui, id, input, thickness)
            }
            Node::Offset { input, distance } => {
                self.offset_controls(ui, id, input, distance)
            }
            // The same fields the manipulator writes, which is the point: dragging a handle
            // and typing a number here are two ways into one value, not two mechanisms.
            Node::Transform { input, translation, rotation, scale, about } => {
                self.transform_controls(ui, id, input, translation, rotation, scale, about)
            }
            Node::Pipe { curve, radius } => {
                self.pipe_controls(ui, id, curve, radius)
            }
            Node::Morph { a, b, t } => {
                self.morph_controls(ui, id, node, a, b, t)
            }
            Node::Boolean { .. } | Node::Blend { .. } => {
                self.boolean_controls(ui, id, node)
            }
            Node::Loft { ref sections, spine, smooth, rims } => {
                self.loft_controls(ui, id, &node, sections, spine, smooth, rims)
            }
            Node::Unknown(unknown) => {
                ui.label(egui::RichText::new(unknown.to_string()).italics());
            }
        }
        self.body_actions(ui, id);
    }

    /// The Sweep block: the section, the spine it is carried along, how it is held, and how
    /// it changes on the way.
    ///
    /// Not `sweep_controls`, which edits an extrude's `sweep`: which way it pulls its profile.
    fn sweep_block_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        outline: fieldform_scene::Outline,
        spine: CurveId,
        section: fieldform_scene::Section,
        law: fieldform_scene::SweepLaw,
        rims: fieldform_scene::Rims,
    ) {
        ui.label(if outline.is_wall() {
            "A curve given a thickness and carried along another, held square to it \
             the whole way: a section that need not close. It starts where the \
             section is drawn and follows the path's shape, turning about the middle \
             of the section's two ends — or, round a loop, about the loop's own \
             middle. This always reads a little short, so a fillet against it is \
             marked approximate."
        } else {
            "A closed profile carried along a curve, held square to it the whole way. \
             It starts where the profile is and follows the curve's shape, so where the \
             curve was drawn does not matter, and the section turns only as much as the \
             curve makes it — there is no up direction to set. A straight curve \
             is what Extrude is for and reads exactly; this always reads a little \
             short, so a fillet against it is marked approximate."
        });
        ui.separator();
        let new_outline = self.outline_controls(ui, id, outline);
        // Any curve but the section's, open or closed: the path is not a section. The
        // same chooser a pipe's curve has, since it is the same question.
        let section_curve = new_outline.curve();
        let mut curves: Vec<(CurveId, String)> = self
            .doc
            .curves
            .iter()
            .filter(|(cid, _)| **cid != section_curve)
            .map(|(cid, c)| (*cid, c.name.clone()))
            .collect();
        curves.sort_by_key(|(cid, _)| cid.raw());
        let mut chosen = spine;
        let offered = self.curve_choices(&curves);
        let shown = self.shown_label(&offered, spine, "missing");
        ui.horizontal(|ui| {
            row_label(ui, "Spine:");
            if let Some(picked) = self.reference_field(
                ui,
                (id, "sweep_spine"),
                chosen,
                &shown,
                &offered,
                RefKind::Curve,
                DropsFrom::curve(|c| c),
            ) {
                chosen = picked;
            }
        });
        // **The cost, said where the curve is chosen.** A sweep divides its value by
        // `1 / (1 - k r)` to stay a lower bound, and the tracer takes that many times
        // the steps. Measured by `fieldform-gpu/examples/sweep_cost.rs`: a 270 mm glyph
        // round a 330 mm bend reads 6.6x short and draws at 150 ms a frame where the
        // same glyph extruded draws at 8. Nothing else on the block said so, and a
        // viewport that crawled with no reason on screen is the report that put this
        // here -- the same rule as the fillet's "approximate": the number is stated
        // where it is entered.
        //
        // **Not where the section reads in its tube** (`sweep-spec.md` §4): a square section
        // under the plain law, round a path in one plane, reads its walls truly however the
        // path bends, so there is no factor to state and no crawl to explain.
        let stretch = fieldform_field::sweep_stretch(&self.doc, &new_outline, chosen, &law);
        let in_its_tube =
            fieldform_field::sweep_reads_true(&self.doc, &new_outline, chosen, &section, &law);
        if stretch > 1.05 && !in_its_tube {
            ui.horizontal(|ui| {
                row_label(ui, "");
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("\u{2248} reads {stretch:.1}\u{d7} short"),
                )
                .on_hover_text(format!(
                    "The section is wide for the tightest bend of the path, so the field \
                     reads short by this factor to stay honest, and the viewport takes \
                     {stretch:.1}\u{d7} the steps to draw it. A gentler path or a smaller \
                     section brings it down; past about 16\u{d7} the viewport runs out of \
                     steps and the surface starts to go missing."
                ));
            });
        }
        // **How the section is held, where the path is chosen.** The two answers
        // differ only when the path meets the profile at an angle, which is exactly when
        // somebody is looking at this row — so the choice sits under the spine rather
        // than in a menu somewhere.
        let mut held = section;
        ui.horizontal(|ui| {
            row_label(ui, "Section:");
            for choice in [
                fieldform_scene::Section::Square,
                fieldform_scene::Section::Skewed,
                fieldform_scene::Section::Parallel,
            ] {
                ui.selectable_value(&mut held, choice, choice.label())
                    .on_hover_text(match choice {
                        fieldform_scene::Section::Square => {
                            "Perpendicular to the path everywhere, turning with it \
                             — what a pipe does, and what a round section wants."
                        }
                        fieldform_scene::Section::Skewed => {
                            "The start face lies flush in the profile's own plane, at \
                             whatever angle the path meets it, and the section carries \
                             that tilt the whole way."
                        }
                        fieldform_scene::Section::Parallel => {
                            "The section never turns: it stays parallel to the profile's \
                             own plane all the way along, carried by the path's shape. \
                             A draughtsman's parallel sweep."
                        }
                    });
            }
        });
        if held != fieldform_scene::Section::Square {
            ui.label(
                egui::RichText::new(
                    "Held at an angle, the path has to cross the section rather than \
                     run along it — a path within a few degrees of the profile's \
                     own plane is refused.",
                )
                .small()
                .weak(),
            );
        }
        ui.separator();
        let new_law = self.sweep_law_controls(ui, id, &law, held, outline.is_wall());
        ui.separator();
        let new_rims = self.rims_controls(ui, rims);
        if new_outline != outline
            || chosen != spine
            || held != section
            || new_law != law
            || new_rims != rims
        {
            self.replace_node(
                id,
                Node::Sweep {
                    outline: new_outline,
                    spine: chosen,
                    section: held,
                    law: new_law,
                    rims: new_rims,
                },
            );
        }
    }

    /// The Revolve block: the section, the axis, and how far round on each side.
    fn revolve_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        outline: fieldform_scene::Outline,
        frame: fieldform_scene::Frame,
        angle: ParamRef,
        sides: fieldform_scene::AngleSides,
        rims: fieldform_scene::Rims,
    ) {
        ui.label(if outline.is_wall() {
            "A curve given a thickness and swept about an axis: a shell of revolution, \
             from a curve that need not close. Left and right are as the curve is \
             seen beside the axis, with the axis running up and the radius out to \
             the right, and where the wall reaches across the axis it is solid \
             through."
        } else {
            "A closed profile swept about an axis. It turns fastest where the profile \
             is furthest from the axis, so where the axis sits decides the shape as \
             much as the profile does."
        });
        ui.separator();
        let new_outline = self.outline_controls(ui, id, outline);
        let new_frame = self.frame_controls(ui, id, "Axis:", frame, None);
        self.revolve_spread(ui, id);
        ui.separator();
        // How far round, from the half-plane the section lies in. A full turn is what
        // every revolve was and reads exactly; anything less is a wedge with two flat
        // caps, which the field says is a bound inside, so the panel says so beside it.
        let (new_angle, _) = self.parametric_field_in(
            ui,
            "Angle:",
            angle,
            1.0,
            REVOLVE_DEGREES,
            "How far round the axis, in degrees, starting from where the profile is \
             drawn. 360 is all the way round; less leaves two flat ends.",
            "\u{b0}",
        );
        // Which way about the profile: three in a row, as an extrude's Sides are.
        let mut new_sides = sides.clone();
        ui.horizontal(|ui| {
            row_label(ui, "Sides:");
            let same_kind = |a: &fieldform_scene::AngleSides, b: &fieldform_scene::AngleSides| {
                std::mem::discriminant(a) == std::mem::discriminant(b)
            };
            let mut lit_up = |ui: &mut egui::Ui, choice: fieldform_scene::AngleSides, word, hover| {
                if ui.selectable_label(same_kind(&new_sides, &choice), word).on_hover_text(hover).clicked() {
                    new_sides = choice;
                }
            };
            lit_up(ui, fieldform_scene::AngleSides::One, "One", "The whole angle one way from the profile.");
            lit_up(
                ui,
                fieldform_scene::AngleSides::Symmetric,
                "Both",
                "Half the angle each way, so the profile is the middle of the wedge.",
            );
            // Side two starts out as side one's mirror, so choosing it moves nothing
            // until its angle is typed.
            let other = match &sides {
                fieldform_scene::AngleSides::Two { .. } => sides.clone(),
                _ => fieldform_scene::AngleSides::Two { angle },
            };
            lit_up(ui, other, "Two", "The whole angle one way and another angle the other.");
        });
        if let fieldform_scene::AngleSides::Two { angle: other } = new_sides.clone() {
            let (other, _) = self.parametric_field_in(
                ui,
                "Angle 2:",
                other,
                1.0,
                REVOLVE_OTHER_DEGREES,
                "How far round the other way from the profile.",
                "\u{b0}",
            );
            new_sides = fieldform_scene::AngleSides::Two { angle: other };
        }
        let partial = fieldform_field::revolve_wedge(&self.doc, &new_angle, &new_sides)
            .is_ok_and(|(_, total)| total < 360.0 - 1e-9);
        if partial {
            ui.label(
                egui::RichText::new("\u{2248} a partial revolve reads short inside")
                    .color(ui.visuals().warn_fg_color),
            )
            .on_hover_text(
                "Its surface is exactly where it should be, and from outside the field is \
                 exact too. Inside the material the depth is held against each flat end's \
                 plane rather than the end itself, so it reads a little shallow there, and \
                 a fillet against it is marked approximate.",
            );
        }
        // **The rims, where there are ends to round**: a partial revolve's two caps, with the
        // editor every profile solid shares. All the way round there are none, and rims a file
        // brought are said to be refused rather than shown as a row that does nothing.
        let new_rims = if partial {
            ui.separator();
            self.rims_controls(ui, rims)
        } else {
            if !rims.is_plainly_hard() {
                ui.label(
                    egui::RichText::new("\u{26a0} rims need flat ends -- turn less than 360")
                        .color(ui.visuals().warn_fg_color),
                );
            }
            rims
        };
        if new_outline != outline || new_frame != frame || new_angle != angle || new_sides != sides || new_rims != rims {
            self.replace_node(
                id,
                Node::Revolve { outline: new_outline, frame: new_frame, angle: new_angle, sides: new_sides, rims: new_rims },
            );
        }
    }

    /// The Sphere block: where it is, and its radius.
    fn sphere_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        centre: fieldform_scene::Place,
        radius: ParamRef,
    ) {
        let new_centre = self.place_fields(ui, id, "At:", centre);
        ui.separator();
        let (new_radius, _) =
            self.parametric_field(ui, "Radius:", radius, 0.5, POSITIVE_MM, "");
        if new_centre != centre || new_radius != radius {
            self.replace_node(id, Node::Sphere { centre: new_centre, radius: new_radius });
        }
    }

    /// The Cuboid block: where it is, which way it faces, and its three sides.
    fn cuboid_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        centre: fieldform_scene::Place,
        size: [ParamRef; 3],
        turn: Turn,
        edge: fieldform_scene::Fillet,
    ) {
        let new_centre = self.place_fields(ui, id, "At:", centre);
        // Three sides rather than one, and no link between them: a cube is this with
        // three equal numbers. A checkbox holding them together would be state the node
        // does not carry, so it could only be remembered by the app and would disagree
        // with a saved file.
        //
        // **One row, in the columns `At:` above it uses.** These were three stacked
        // full-width rows named `Width (X)`, `Depth (Y)` and `Height (Z)`, so a box's
        // size took three rows to say and lined up with nothing — while the position
        // directly above it was already one row of three cells. The axis letters carry
        // the width/depth/height reading, and they are the letters every other row
        // uses: a second spelling of X, Y and Z would be a second thing to learn for
        // one idea (D9).
        // **A box has a facing, so it says so here.** It used to have one only once a
        // placement had been wrapped around it, which meant a freshly made box had no
        // orientation to type and gained one by being dragged — the block that owns the
        // shape not holding the fact about the shape. Same three angles, same columns
        // and same editor a placement's Turn uses (D9).
        //
        // **Where it is, then which way it faces, then what it is.** One order in every
        // block, so the two rows every block shares sit in the same place and the eye
        // can go straight past them to the part that differs.
        let new_turn = self.turn_fields(ui, id, turn);
        ui.separator();
        let new_size = self.size_row(ui, id, size);
        ui.separator();
        let new_edge = self.edge_controls(ui, edge);
        if new_centre != centre || new_size != size || new_turn != turn || new_edge != edge {
            self.replace_node(
                id,
                Node::Cuboid { centre: new_centre, size: new_size, turn: new_turn, edge: new_edge },
            );
        }
    }

    /// The Skewed Box block: its middle, and the three edges that lean it.
    fn skewed_box_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        centre: fieldform_scene::Place,
        edges: [[ParamRef; 3]; 3],
    ) {
        let new_centre = self.place_fields(ui, id, "At:", centre);
        ui.separator();
        let mut new_edges = edges;
        for (i, edge) in new_edges.iter_mut().enumerate() {
            *edge = self.xyz_row(ui, &format!("Edge {}:", i + 1), *edge, 0.5, ANYWHERE_MM, " mm");
        }
        // **Said where the numbers are typed**, not only when the compile refuses them:
        // three edges in one plane enclose nothing, and the viewport would otherwise go
        // on showing the last box that did. Judged by the same test the field applies.
        let vectors = new_edges.map(|e| {
            nalgebra::Vector3::new(self.resolve(e[0]), self.resolve(e[1]), self.resolve(e[2]))
        });
        if fieldform_field::SkewedBox::new(nalgebra::Point3::origin(), vectors).is_none() {
            ui.label(
                egui::RichText::new("\u{26a0} the three edges lie in one plane")
                    .color(ui.visuals().warn_fg_color),
            )
            .on_hover_text(
                "A box needs the third edge to leave the plane of the other two; \
                 otherwise it has no volume. Change one number.",
            );
        }
        if new_centre != centre || new_edges != edges {
            self.replace_node(id, Node::SkewedBox { centre: new_centre, edges: new_edges });
        }
    }

    /// The Cylinder block: the two ends it is drawn between, and its radius.
    fn cylinder_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        ends: Ends,
        radius: ParamRef,
        edge: fieldform_scene::Fillet,
    ) {
        let new_ends = self.ends_controls(ui, id, ends);
        ui.separator();
        let (new_radius, _) =
            self.parametric_field(ui, "Radius:", radius, 0.5, POSITIVE_MM, "");
        ui.separator();
        let new_edge = self.edge_controls(ui, edge);
        if new_ends != ends || new_radius != radius || new_edge != edge {
            self.replace_node(
                id,
                Node::Cylinder { ends: new_ends, radius: new_radius, edge: new_edge },
            );
        }
    }

    /// The Cone block: the two ends, and a radius at each that may be nought.
    fn cone_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        ends: Ends,
        base_radius: ParamRef,
        top_radius: ParamRef,
        edge: fieldform_scene::Fillet,
    ) {
        let new_ends = self.ends_controls(ui, id, ends);
        ui.separator();
        // Zero is allowed on either radius where one normally is not: it is what
        // brings that end to a point, which is the shape most people mean by "cone".
        let (new_base_r, _) = self.parametric_field(
            ui,
            "From radius:",
            base_radius,
            0.5,
            ANYWHERE_MM,
            "Zero brings this end to a point",
        );
        let (new_top_r, _) = self.parametric_field(
            ui,
            "To radius:",
            top_radius,
            0.5,
            ANYWHERE_MM,
            "Zero comes to a point; above zero truncates",
        );
        ui.separator();
        let new_edge = self.edge_controls(ui, edge);
        if new_ends != ends || new_base_r != base_radius || new_top_r != top_radius || new_edge != edge {
            self.replace_node(
                id,
                Node::Cone {
                    ends: new_ends,
                    base_radius: new_base_r,
                    top_radius: new_top_r,
                    edge: new_edge,
                },
            );
        }
    }

    /// The Torus block: where it is, which way it faces, and its ring and tube.
    fn torus_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        centre: fieldform_scene::Place,
        major_radius: ParamRef,
        minor_radius: ParamRef,
        turn: Turn,
    ) {
        let new_centre = self.place_fields(ui, id, "At:", centre);
        // Where it is, then which way it faces, then what it is -- the order every
        // block follows. A torus is built lying in XY about +Z, so its facing is the
        // whole of how it is aimed, and it is the one round solid with no pair of ends
        // to say that for it.
        let new_turn = self.turn_fields(ui, id, turn);
        ui.separator();
        let (new_major, _) = self.parametric_field(
            ui,
            "Ring:",
            major_radius,
            0.5,
            POSITIVE_MM,
            "Centre to the middle of the tube",
        );
        let (new_minor, _) = self.parametric_field(
            ui,
            "Tube:",
            minor_radius,
            0.5,
            POSITIVE_MM,
            "The tube's own radius",
        );
        if new_centre != centre
            || new_major != major_radius
            || new_minor != minor_radius
            || new_turn != turn
        {
            self.replace_node(
                id,
                Node::Torus {
                    centre: new_centre,
                    major_radius: new_major,
                    minor_radius: new_minor,
                    turn: new_turn,
                },
            );
        }
    }

    /// An imported mesh: what it is and whether it closes, said rather than edited, since
    /// nothing about it is this document's to change.
    fn mesh_controls(&self, ui: &mut egui::Ui, mesh: fieldform_scene::MeshId) {
        match self.doc.mesh(mesh) {
            Some(data) => {
                ui.label(
                    "Imported geometry. Nothing here is editable — it is a solid \
                     someone else authored, and this document references it.",
                );
                ui.separator();
                ui.label(format!(
                    "{} triangles, {} vertices",
                    data.triangle_count(),
                    data.vertex_count(),
                ));
                if !data.is_closed() {
                    // Said plainly, and said as what it means rather than as what
                    // failed. The user's next question is "can I cut with it", and the
                    // answer is the thing to lead with.
                    ui.label(
                        egui::RichText::new(format!(
                            "{} open edges — not a solid, so it can be seen and \
                             measured but not cut or blended. The exporter's \
                             tessellator could not close some faces.",
                            data.open_edges,
                        ))
                        .color(egui::Color32::from_rgb(230, 180, 90)),
                    );
                }
                // Shown as information, never as a knob. D2: no modelling tolerance is
                // exposed to the user, and this one is the source's rather than ours
                // anyway — it says how faithful the sample is, which is worth knowing
                // and not worth pretending can be changed after the fact.
                if let Some(tolerance) = data.tolerance_mm {
                    ui.label(format!("Tessellated at {tolerance} mm"));
                }
                if data.is_closed() {
                    ui.label("Cut it, blend it, or place it with the manipulator.");
                }
            }
            None => {
                ui.label("This mesh is not in the document. The file may be damaged.");
            }
        }
    }

    /// The Plane block: where it is, which way it faces, and which half is solid.
    fn plane_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        origin: fieldform_scene::Place,
        flipped: bool,
        turn: Turn,
    ) {
        ui.label(
            "A half-space. Drawn as a square so you can aim it, but it has no edges \
             — cut something with it and the result is finite.",
        );
        ui.separator();
        let new_origin = self.place_fields(ui, id, "At:", origin);
        // Where it is, then which way it faces, then what it is -- the order every
        // block follows. Typed as well as dragged: the manipulator still turns it and
        // the normal below still follows, and this is the same three angles written
        // down, for when the answer is a number off a drawing.
        let new_turn = self.turn_fields(ui, id, turn);
        ui.separator();
        let mut new_flipped = flipped;
        // The direction itself rather than a name for which half is kept. It stays
        // readable once a placement has turned the plane, where "above" and "below"
        // stop meaning anything; it is the number a person would check against a
        // drawing; and it is where binding a normal to a vector belongs when that
        // lands (backlog item 18), so the panel already has the shape for it.
        let facing = if new_flipped { -1.0 } else { 1.0 };
        let n = self.placement_above(id).isometry.rotation * Vector3::z() * facing;
        ui.horizontal(|ui| {
            ui.label(format!("Normal:  {:.3}, {:.3}, {:.3}", n.x, n.y, n.z));
            // Turning a plane over is not a rotation: it is the same plane facing the
            // other way, and spending 180 degrees of the placement on it would fight
            // the manipulator for the same control.
            if ui
                .button("Flip")
                .on_hover_text("Reverses the normal, so the other half is the solid one")
                .clicked()
            {
                new_flipped = !new_flipped;
            }
        });

        if new_origin != origin || new_flipped != flipped || new_turn != turn {
            self.replace_node(
                id,
                Node::Plane {
                    origin: new_origin,
                    flipped: new_flipped,
                    turn: new_turn,
                },
            );
        }
    }

    /// The Point block: where it is.
    fn point_controls(&mut self, ui: &mut egui::Ui, id: NodeId, at: [ParamRef; 3]) {
        ui.label(
            "A position, as something to measure from. It encloses no volume, so it \
             draws nothing — that is what it is for, not a fault.",
        );
        ui.separator();
        let new_at = self.centre_fields(ui, "At:", at);
        if new_at != at {
            self.replace_node(id, Node::Point { at: new_at });
        }
    }

    /// The Capsule block: the two cap centres, and a radius at each.
    fn capsule_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        ends: Ends,
        start_radius: ParamRef,
        end_radius: ParamRef,
    ) {
        // The points are the cap *centres*, so the capsule reaches a radius past each
        // of them and its overall length is the distance between them plus two radii.
        // Said here because it is the one thing about a capsule that is easy to read
        // the other way, and it changes what a typed number does.
        ui.label(
            egui::RichText::new(
                "The ends are the cap centres - the shape reaches a radius past each.",
            )
            .weak(),
        );
        let new_ends = self.ends_controls(ui, id, ends);
        ui.separator();
        let (new_start_r, _) =
            self.parametric_field(ui, "From radius:", start_radius, 0.5, POSITIVE_MM, "");
        let (new_end_r, _) =
            self.parametric_field(ui, "To radius:", end_radius, 0.5, POSITIVE_MM, "");
        if new_ends != ends || new_start_r != start_radius || new_end_r != end_radius {
            self.replace_node(
                id,
                Node::Capsule {
                    ends: new_ends,
                    start_radius: new_start_r,
                    end_radius: new_end_r,
                },
            );
        }
    }

    /// The Twist block: the body wound, the axis it winds about, and how fast.
    fn twist_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
        degrees_per_mm: ParamRef,
        frame: fieldform_scene::Frame,
    ) {
        ui.label(
            "Winds this body about an axis — the further along it, the more the \
             body is turned.",
        );
        ui.separator();
        let picked = self.pick_input(ui, id, input);

        // The same frame editor every other aimable block gets (D9). A twist used
        // to hold an `Axis` and an origin, which said "about this axis through this
        // point" and could say it to nothing else — so an axis set up for a
        // twist could not be handed to a mirror or a lattice. It can now, and a twist
        // can wind about a direction that is not one of the three.
        let new_frame = self.frame_controls(ui, id, "About:", frame, None);
        ui.separator();

        let (new_rate, _) = self.parametric_field_in(
            ui,
            "Rate:",
            degrees_per_mm,
            0.1,
            DEGREES,
            "Degrees per millimetre along the axis. Negative winds the other way.",
            "\u{b0}/mm",
        );
        // The total, which is the number a person actually has in mind — "a quarter
        // turn over its length" rather than a rate. Derived rather than stored, so it
        // cannot drift from the rate it is computed from.
        if let Some((lo, hi)) = self.body_extent(id).as_box() {
            // Measured along the frame's own axis, which is any direction now rather
            // than one of three — so the reach is the box projected onto it
            // rather than the side it happens to name.
            let aim = self.frame_axis(&new_frame);
            let half = (hi - lo) * 0.5;
            let along = 2.0
                * (half.x * aim.x.abs() + half.y * aim.y.abs() + half.z * aim.z.abs());
            let total = self.resolve(new_rate) * along;
            ui.label(
                egui::RichText::new(format!(
                    "{total:.0}\u{b0} over its {along:.1} mm along that axis",
                ))
                .small()
                .weak(),
            );
        }
        ui.label(
            egui::RichText::new(
                "A twist stretches material, so its field is a lower bound and a blend \
                 against one comes out larger than the radius asked for — allowed, \
                 and marked approximate where the radius is typed.",
            )
            .small()
            .weak(),
        );
        if new_rate != degrees_per_mm || new_frame != frame || picked != input {
            self.replace_node(
                id,
                Node::Twist {
                    input: picked,
                    degrees_per_mm: new_rate,
                    frame: new_frame,
                },
            );
        }
    }

    /// The Mirror block: the body reflected, the plane it is reflected in, and whether the
    /// original is part of the block.
    fn mirror_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
        frame: fieldform_scene::Frame,
        join: bool,
    ) {
        ui.label(
            "The reflection of this body in a plane. The body itself is still a block \
             of its own, so each can be shown, hidden and moved separately — \
             union the two to make something symmetric.",
        );
        ui.separator();
        let picked = self.pick_input(ui, id, input);

        // Which planes in the document could serve as a mirror. Offered only when
        // there are any, so the control does not appear as a dead end in a document
        // with no plane in it.
        // The frame editor, the same one a lattice, a twist and a radial pattern get
        // (D9). A mirror plane *is* a frame — a point and a normal — and it used to be
        // spelled `MirrorPlane`, which said exactly that and could be said to nothing
        // else. Its two cases were this type's two cases all along.
        let new_frame = self.frame_controls(ui, id, "Normal:", frame, None);

        let mut new_join = join;
        // **Off is what a new mirror gets**, and this is here for the ones that are not
        // new: a file written before the block held only the reflection says `join`, and
        // a control that can read a document has to be able to show what it says. On, it
        // is the old behaviour -- both halves in one block -- which also means the body
        // is drawn twice over if it is still beside it.
        ui.checkbox(&mut new_join, "Include the original in this block")
            .on_hover_text(
                "Off, this block is the reflection and the body it came from is its own \
                 block beside it. On, the two are one block, which is how mirrors worked \
                 before and is what a file written then will show here.",
            );

        if new_frame != frame || new_join != join || picked != input {
            self.replace_node(
                id,
                Node::Mirror { input: picked, frame: new_frame, join: new_join },
            );
        }
    }

    /// The Shell block: the body hollowed, and its wall.
    fn shell_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
        thickness: ParamRef,
    ) {
        ui.label(
            "Hollows this body out to a wall. The wall is measured inward, so the \
             outside dimensions do not change.",
        );
        ui.separator();
        let picked = self.pick_input(ui, id, input);
        let (new_thickness, _) = self.parametric_field(
            ui,
            "Wall:",
            thickness,
            0.05,
            POSITIVE_MM,
            "Thicker than the body is anywhere and nothing is left to hollow, which \
             is a real answer rather than an error",
        );
        if new_thickness != thickness || picked != input {
            self.replace_node(
                id,
                Node::Shell { input: picked, thickness: new_thickness },
            );
        }
    }

    /// The Offset block: the body moved, and a distance that is one number or a ramp.
    fn offset_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
        distance: ParamRef,
    ) {
        ui.label(
            "Moves this body's surface by a distance. Positive grows it, negative \
             shrinks it.",
        );
        ui.separator();
        let picked = self.pick_input(ui, id, input);

        let driver = distance.driver();
        ui.horizontal(|ui| {
            row_label(ui, "Distance:");
            if ui.add(mode_button(driver.is_none(), "One number")).clicked() {
                if let Some(ramp) = driver {
                    // The ramp goes with the switch back, and so does whatever it was
                    // reading — otherwise a body that was only ever a driver reappears
                    // in the viewport as soon as nothing consumes it.
                    self.drop_driver(id, ramp);
                }
            }
            if ui.add(mode_button(driver.is_some(), "Ramp")).clicked()
                && driver.is_none()
            {
                self.drive_offset(id, input);
            }
        });
        // The same picker the morph has, because it is the same question. See
        // `pick_driver` — one of these ramps may already be driving that morph, which is
        // the arrangement `ParamRef::Field` was built for.
        if let Some(choice) = self.pick_driver(ui, "offset_driver", driver) {
            match choice {
                Some(ramp) => self.wire_driver(id, 0, ramp),
                None => {
                    if let Some(ramp) = driver {
                        self.drop_driver(id, ramp);
                    }
                }
            }
        }

        // **A ramp is edited in the ramp, not here.** Its controls used to be repeated
        // inside whatever it drove, from when a ramp had no row of its own and could
        // not be reached any other way. It has one now, so this was one block shown in
        // two places -- which is what D9 says drifts apart -- and a ramp driving two
        // sockets appeared twice over. The picker above says which ramp; its own row is
        // where it is changed. Asked for directly, and the morph lost the same
        // duplication earlier for the same reason.
        if driver.is_none() {
            // Signed, and the range says so: shrinking is half of what this is for, and
            // a control that refused negatives would need a second control beside it
            // saying which way — two ways to express one number, which is how they
            // disagree.
            let (new_distance, _) = self.parametric_field(
                ui,
                "By:",
                distance,
                0.1,
                ANYWHERE_MM,
                "Negative shrinks. Shrink by more than the body is thick and it \
                 goes away, which is a real answer rather than an error",
            );
            if new_distance != distance || picked != input {
                self.replace_node(
                    id,
                    Node::Offset { input: picked, distance: new_distance },
                );
            }
        }
    }

    /// The Transform block: the body placed, where it sits, how it is turned and scaled, and
    /// what it turns and scales about.
    ///
    /// Takes the node's five fields by value, as the lattice panel takes its four: the caller has
    /// already cloned the node, and the panel writes a whole replacement when anything changed.
    #[allow(clippy::too_many_arguments)]
    fn transform_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
        translation: [ParamRef; 3],
        rotation: [ParamRef; 3],
        scale: ParamRef,
        about: fieldform_scene::Place,
    ) {
        ui.label(
            "Where this body sits, how it is turned, and how big it is. Drag it in \
             the viewport, or type it here.",
        );
        ui.separator();
        let picked = self.pick_input(ui, id, input);
        let new_translation =
            self.placement_fields(ui, "Move:", translation, 0.5, ANYWHERE_MM);
        let new_rotation = self.placement_fields(ui, "Angle:", rotation, 1.0, DEGREES);
        let (new_scale, _) = self.parametric_field(
            ui,
            "Scale:",
            scale,
            0.01,
            POSITIVE_MM,
            "One factor for all three axes — a per-axis scale would stop this body \
             being a true distance field, and blends refuse those",
        );
        // **How big it is, where the panel already says it says so.** The heading above
        // promises "how big it is" and only Scale answered — which is one factor for
        // all three axes and has to be, since a per-axis scale would stop the body
        // being a true distance field. The three numbers a box is actually sized by
        // live on the box, so this row edits them *there* rather than inventing a
        // second and unsound way to say the same thing. Same `xyz_row` the box's own
        // panel uses, so it lands in the columns Move, Turn and About already use.
        //
        // Only for a cuboid, because it is the only shape whose size is three lengths.
        // A sphere's radius is one number and a cylinder is a pair of ends, and putting
        // either into three columns would be a row that lies about its shape.
        //
        // **And only when the slot names a block.** An empty one has no box to size, so this
        // row draws nothing -- the same nothing it already drew for an input that is some other
        // shape, rather than a second way of saying "not a cuboid".
        if let Some(below) = picked.source()
            && let Some(Node::Cuboid { centre, size, turn, edge }) =
                self.part_graph().node(below).cloned()
        {
            // Keyed by the box, not by the placement, so the lock follows the thing
            // whose sides it holds in proportion — the same box edited from its own
            // panel is the same lock.
            let new_size = self.size_row(ui, below, size);
            // Said only when it is true, and it is true rarely: a shared box resized
            // from one of its placements changes the others too, which is what sharing
            // is for and is invisible from here.
            let users = self.part_graph().consumers_of(below).count();
            if users > 1 {
                ui.label(
                    egui::RichText::new(format!(
                        "This box is used {users} times — its size is shared"
                    ))
                    .weak(),
                )
                .on_hover_text(
                    "One block, used more than once. Resizing it here resizes every \
                     use of it, which is what sharing it is for; to size one on its \
                     own, give that one a box of its own.",
                );
            }
            if new_size != size {
                // The box's turn and edges are carried through: this row edits its size
                // and nothing else, and rebuilding the node without them would quietly
                // straighten a box that had been turned, or sharpen one that was rounded.
                self.replace_node(
                    below,
                    Node::Cuboid { centre, size: new_size, turn, edge },
                );
            }
        }
        // **What "scale it by two" needs, to have somewhere to scale from.** Without
        // a pivot the turn and the scale happen about the local origin, so a body that
        // is not centred there swings away as it grows, and the way to hold it still
        // was to work out a translation cancelling the swing -- arithmetic that has to
        // be redone every time the factor changes. A place rather than three numbers,
        // so a pivot can sit on a point and several placements turn about one thing.
        let new_about = self.place_fields(ui, id, "About:", about);
        if picked != input
            || new_translation != translation
            || new_rotation != rotation
            || new_scale != scale
            || new_about != about
        {
            self.replace_node(
                id,
                Node::Transform {
                    input: picked,
                    translation: new_translation,
                    rotation: new_rotation,
                    scale: new_scale,
                    about: new_about,
                },
            );
        }
    }

    /// The Pipe block: the curve it sweeps, and a radius that is one number or a ramp.
    fn pipe_controls(&mut self, ui: &mut egui::Ui, id: NodeId, curve: CurveId, radius: ParamRef) {
        // Which curve it sweeps is an input of the pipe, editable here — the same
        // shape a boolean's two operands take. Sweeping a different curve is a change
        // to this block, not a reason to delete it and make another.
        let mut curves: Vec<(CurveId, String)> = self
            .doc
            .curves
            .iter()
            .map(|(id, c)| (*id, c.name.clone()))
            .collect();
        curves.sort_by_key(|(id, _)| id.raw());
        let mut chosen = curve;
        let offered = self.curve_choices(&curves);
        let shown = self.shown_label(&offered, curve, "missing");
        ui.horizontal(|ui| {
            row_label(ui, "Curve:");
            if let Some(picked) = self.reference_field(
                ui,
                (id, "pipe_curve"),
                chosen,
                &shown,
                &offered,
                RefKind::Curve,
                DropsFrom::curve(|c| c),
            ) {
                chosen = picked;
            }
        });
        // **The same switch an offset's distance has**, because it is the same
        // question: one number, or a number that varies through space. Backlog item 2,
        // and the third socket `ramp-decision.md` said each further consumer would have
        // to earn its own bound for -- which a pipe does, since a varying radius is its
        // own spine offset by that radius and costs the offset's `1 + L`.
        let driver = radius.driver();
        ui.horizontal(|ui| {
            row_label(ui, "Radius:");
            if ui.add(mode_button(driver.is_none(), "One number")).clicked()
                && let Some(ramp) = driver
            {
                self.drop_driver(id, ramp);
            }
            if ui
                .add(mode_button(driver.is_some(), "Ramp"))
                .on_hover_text(
                    "A radius that varies along the pipe — thin at one end and \
                     fat at the other, or fat in the middle if the ramp is pointed at a \
                     point instead of a plane.",
                )
                .clicked()
                && driver.is_none()
            {
                self.drive_pipe(id, curve);
            }
        });
        if let Some(choice) = self.pick_driver(ui, "pipe_driver", driver) {
            match choice {
                Some(ramp) => self.wire_driver(id, 0, ramp),
                None => {
                    if let Some(ramp) = driver {
                        self.drop_driver(id, ramp);
                    }
                }
            }
        }
        let (changed_radius, changed) = match driver {
            Some(_) => (radius, false),
            None => self.parametric_field(ui, "By:", radius, 0.5, POSITIVE_MM, ""),
        };
        if changed || chosen != curve {
            self.replace_node(id, Node::Pipe { curve: chosen, radius: changed_radius });
        }
    }
}
