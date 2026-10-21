// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The inspector panel, and the number, count and parameter fields its blocks are built from.

use crate::*;

impl App {
    /// One parametric number: a drag field, and a V mark that names it.
    ///
    /// The mark is the whole difference between the two layers made clickable. A literal is
    /// just a number in the graph; naming it promotes it to a document `Parameter` that other
    /// features can reference, which is what makes one dimension able to drive several — and a
    /// named one can then be computed from others, by an expression typed on the parameter's
    /// own row (`fieldform_scene::expression` parses it).
    ///
    /// Returns the reference to store, and whether the value changed.
    /// **One editor for every block's rims** (D9, `round-chamfer-spec.md` §7): Hard, the three
    /// shapes a blend offers in the same words, and the number under the label a blend's fillet
    /// uses for the same shape, with the line under it saying what it does. An extrude, a sweep
    /// and a loft each call this and get back the rims to write, so the three cannot drift.
    ///
    /// Hard is a reach of nought, which is what a hard rim *is* -- there is no flag beside the
    /// number to disagree with it. Picking a shape from hard starts at a millimetre, which is
    /// visible on anything and refused by name on a solid too small for it.
    pub(crate) fn rims_controls(
        &mut self,
        ui: &mut egui::Ui,
        rims: fieldform_scene::Rims,
    ) -> fieldform_scene::Rims {
        use fieldform_scene::{RimEnds, Rims};
        let shaped = cut_shape_row(ui, "Rims:", "The side wall meets each end at an edge, as drawn.", rims.fillet());
        let mut new = if shaped.is_plainly_hard() {
            Rims::hard()
        } else {
            Rims { radius: shaped.radius, continuity: shaped.continuity, at: rims.at }
        };
        if !new.is_plainly_hard() {
            // **Which ends**, asked by the owner (2026-10-17) after finding both ends rounded
            // with no way to keep one sharp. The solid's own start and end, along the way it was
            // made -- `RimEnds` says which face each is for every kind of solid.
            ui.horizontal(|ui| {
                row_label(ui, "At:");
                for at in RimEnds::ALL {
                    if ui
                        .add(mode_button(new.at == at, at.label()))
                        .on_hover_text(match at {
                            RimEnds::Both => "Both ends rounded alike.",
                            RimEnds::Start => {
                                "Only the end the solid starts from: an extrude's profile face, \
                                 or side two's face when it has two; a sweep's path start; a \
                                 loft's first section."
                            }
                            RimEnds::End => {
                                "Only the end the solid finishes at: an extrude's far face, or \
                                 side one's; a sweep's path end; a loft's last section."
                            }
                        })
                        .clicked()
                    {
                        new.at = at;
                    }
                }
            });
            new.radius = self.cut_number_rows(
                ui,
                new.fillet(),
                "How far back from the end the side wall starts to turn, and how far in from \
                 the side the end does. Nought is a hard rim.",
            );
        }
        new
    }

    /// **A box's or a cylinder's edges, in the rims' own rows** (D9, `round-chamfer-spec.md` §7):
    /// Hard, the three shapes, and the number under the label a fillet of that shape uses.
    /// Every edge alike, so there is no row saying which.
    pub(crate) fn edge_controls(
        &mut self,
        ui: &mut egui::Ui,
        edge: fieldform_scene::Fillet,
    ) -> fieldform_scene::Fillet {
        let mut new = cut_shape_row(ui, "Edges:", "Every face meets the next at a sharp edge, as drawn.", edge);
        if !new.is_plainly_hard() {
            new.radius = self.cut_number_rows(
                ui,
                new,
                "How far back from each edge the faces start to turn. Nought is a sharp edge.",
            );
        }
        new
    }

    /// The number a cut is made with, and the line under it saying what that shape does there --
    /// the second half of the rims' and the edges' one editor.
    fn cut_number_rows(
        &mut self,
        ui: &mut egui::Ui,
        cut: fieldform_scene::Fillet,
        hover: &str,
    ) -> fieldform_scene::ParamRef {
        let (edited, _changed) =
            self.parametric_field(ui, fillet_label(cut.continuity), cut.radius, 0.1, FROM_NOUGHT_MM, hover);
        if let Some(note) = fillet_bend_note(cut.continuity, self.resolve(edited)) {
            ui.label(egui::RichText::new(note).small().weak());
        }
        edited
    }

    pub(crate) fn parametric_field(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        current: ParamRef,
        speed: f64,
        range: std::ops::RangeInclusive<f64>,
        hover: &str,
    ) -> (ParamRef, bool) {
        // Millimetres, which almost every number in this application is. The two that are not
        // go through `parametric_field_in` and say so.
        self.parametric_field_in(ui, label, current, speed, range, hover, " mm")
    }

    #[allow(clippy::too_many_arguments)]
    /// How a draft cuts the section's corners, shown wherever a draft is typed.
    ///
    /// **One row for both blocks** (D9), as the type is one type: an extrude's draft and a
    /// sweep's are the same construction, so the question and its words are the same. Shown
    /// only when there *is* a draft, since with upright walls the two answers are one solid
    /// and a control that changes nothing is a control that teaches nothing.
    ///
    /// A wall is told that an open chain scales whole, which is the one case where the answer
    /// the panel gives is not the one the words suggest.
    pub(crate) fn corner_controls(
        &mut self,
        ui: &mut egui::Ui,
        corners: fieldform_scene::Corners,
        drafted: bool,
    ) -> fieldform_scene::Corners {
        if !drafted {
            return corners;
        }
        let mut new = corners;
        ui.horizontal(|ui| {
            row_label(ui, "Corners:");
            for choice in fieldform_scene::Corners::ALL {
                if ui
                    .add(mode_button(new == choice, choice.label()))
                    .on_hover_text(match choice {
                        fieldform_scene::Corners::Round => {
                            "A draft moves every wall out along its own normal, and at a corner \
                             there is no wall to move \u{2014} so the corner sweeps an arc of \
                             however far the walls have gone, and rounds further as the draft \
                             grows. Geometrically what an offset is."
                        }
                        fieldform_scene::Corners::Sharp => {
                            "The two walls stay planes and meet at an edge, which is what a \
                             moulding drawing usually means. A very sharp corner's mitre is \
                             held to four times the offset, past which it keeps the round \
                             answer rather than throwing a spike."
                        }
                    })
                    .clicked()
                {
                    new = choice;
                }
            }
        });
        new
    }

    /// What a swept section does on the way: its taper, its twist, the law that distributes
    /// them, and the stretch of path it runs over. `sweep-v2-spec.md` section 9.
    ///
    /// **One editor, not one per block** (D9): a loft between two sections will take the same
    /// `SweepLaw` and show it here. Each row says what it costs or refuses where its number is
    /// typed, which is the rule the fillet's "approximate" and the sweep's "reads short"
    /// already follow -- a refusal a user meets as a blank viewport is a refusal said in the
    /// wrong place.
    pub(crate) fn sweep_law_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        law: &fieldform_scene::SweepLaw,
        held: fieldform_scene::Section,
        is_wall: bool,
    ) -> fieldform_scene::SweepLaw {
        use fieldform_scene::Taper;
        let mut new = law.clone();

        // **Taper: none, a scale or a draft, never both.** They answer different questions --
        // a scale keeps the section's shape, a draft leans every wall at one angle -- and a
        // block that carried both would be two answers to how its section grows.
        ui.horizontal(|ui| {
            row_label(ui, "Taper:");
            let now = std::mem::discriminant(&new.taper);
            for choice in [
                Taper::None,
                Taper::Scale(ParamRef::Literal(0.5)),
                Taper::Draft(ParamRef::Literal(3.0)),
            ] {
                if ui
                    .add(mode_button(std::mem::discriminant(&choice) == now, choice.label()))
                    .on_hover_text(match choice {
                        Taper::None => "The section is the size it was drawn, the whole way.",
                        Taper::Scale(_) => {
                            "The section grows or shrinks about the point the path carries it \
                             by, keeping its shape — horns, ducts, tapered handles. The wall \
                             angle varies round a section that is not a circle, so this is not \
                             a draft."
                        }
                        Taper::Draft(_) => {
                            "Every wall leans out at this one angle, which is what a moulded \
                             or cast part wants: the same construction an extrude's draft is, \
                             so a drafted sweep along a straight path is the drafted extrude. \
                             Growing rounds the convex corners; shrinking can close a thin \
                             feature."
                        }
                    })
                    .clicked()
                {
                    new.taper = choice;
                }
            }
        });
        match &new.taper {
            Taper::Scale(end) => {
                let (v, _) = self.parametric_field_in(
                    ui,
                    "End size:",
                    end.clone(),
                    0.01,
                    TAPER_SCALE,
                    "How big the section is where the extent ends, as a multiple of the drawn \
                     one. A section closed to nothing before the end of the path is refused.",
                    "\u{d7}",
                );
                new.taper = Taper::Scale(v);
            }
            Taper::Draft(angle) => {
                let (v, _) = self.parametric_field_in(
                    ui,
                    "Draft:",
                    angle.clone(),
                    0.1,
                    DRAFT_DEGREES,
                    "How far every wall leans out, in degrees. Negative leans them in, and a \
                     draft that closes the section before the end of the path is refused.",
                    "\u{b0}",
                );
                let leaning = self.resolve(v.clone()) != 0.0;
                new.taper = Taper::Draft(v);
                new.corners = self.corner_controls(ui, new.corners, leaning);
            }
            Taper::None => {}
        }
        // **A draft pushes a section's walls out from its inside, and an open wall has none.**
        // Said here rather than met as a viewport that will not draw.
        if is_wall && matches!(new.taper, Taper::Draft(_)) {
            ui.horizontal(|ui| {
                row_label(ui, "");
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "An open wall has no inside to draft from — scale it instead.",
                );
            });
        }
        if is_wall && matches!(new.taper, Taper::Scale(_)) {
            ui.label(
                egui::RichText::new(
                    "An open wall scales whole, thickness and all; a wall round a closed loop \
                     keeps its thickness.",
                )
                .small()
                .weak(),
            );
        }

        let (twist, _) = self.parametric_field_in(
            ui,
            "Twist:",
            new.twist.clone(),
            1.0,
            TWIST_DEGREES,
            "How far the section turns about the path over its extent. A section held parallel \
             never turns, so it has no axis to twist about and is refused.",
            "\u{b0}",
        );
        new.twist = twist;
        if matches!(held, fieldform_scene::Section::Parallel)
            && self.resolve(new.twist.clone()) != 0.0
        {
            ui.horizontal(|ui| {
                row_label(ui, "");
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "A section held parallel never turns, so there is no axis to twist about.",
                );
            });
        }

        // The same three a ramp has, and the same word for them: one vocabulary for one idea.
        ui.horizontal(|ui| {
            row_label(ui, "Law:");
            for option in RampShape::ALL {
                if ui
                    .add(mode_button(new.shape == option, option.label()))
                    .on_hover_text(
                        "How the taper and the twist are spread along the extent: G0 is \
                         straight, G1 and G2 ease in and out — steeper in the middle, which \
                         costs a little more in the divisor below.",
                    )
                    .clicked()
                {
                    new.shape = option;
                }
            }
        });

        let (from, _) = self.parametric_field_in(
            ui,
            "Starts at:",
            new.extent[0].clone(),
            1.0,
            PATH_PERCENT,
            "Where along the path the solid begins, as a percentage of its length. The section \
             is carried to that station, so the law begins there too.",
            "%",
        );
        let (to, _) = self.parametric_field_in(
            ui,
            "Ends at:",
            new.extent[1].clone(),
            1.0,
            PATH_PERCENT,
            "Where along the path the solid ends. The start must come before the end.",
            "%",
        );
        new.extent = [from, to];
        if self.resolve(new.extent[0].clone()) >= self.resolve(new.extent[1].clone()) {
            ui.horizontal(|ui| {
                row_label(ui, "");
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "The extent starts where it ends, so there is nothing of the path to sweep \
                     along.",
                );
            });
        }
        let _ = id;
        new
    }

    /// [`Self::parametric_field`], for a number that is not a length.
    ///
    /// The suffix used to be hardcoded to " mm" for everything, so a twist's rate read
    /// "2.812 mm" when it is degrees per millimetre, and a placement's rotation read "45 mm"
    /// when it is degrees. Both are the kind of wrong a person reads past once and then
    /// mistrusts the whole panel for.
    pub(crate) fn parametric_field_in(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        current: ParamRef,
        speed: f64,
        range: std::ops::RangeInclusive<f64>,
        hover: &str,
        suffix: &str,
    ) -> (ParamRef, bool) {
        self.number_field(
            ui,
            label,
            current,
            speed,
            range,
            hover,
            suffix,
            Naming::Offered,
            Labelled::Row,
            Grain::Continuous,
        )
    }

    /// A line under a count saying what became of it — capped, or refused and why.
    ///
    /// **Under the field rather than in the console**, because this is the half of the whole-number
    /// rule that is about the interface: the count a pattern builds and the expression typed for it
    /// may differ only where the difference is said, and the only honest place to say it is beside
    /// the number. The same arrangement an approximate blend's `≈` uses, and for the same reason.
    pub(crate) fn count_note(
        &mut self,
        ui: &mut egui::Ui,
        text: &str,
        colour: egui::Color32,
        hover: Option<&str>,
    ) {
        ui.horizontal(|ui| {
            row_label(ui, "");
            let line = ui.label(egui::RichText::new(text).small().color(colour));
            if let Some(hover) = hover {
                line.on_hover_text(hover);
            }
        });
    }

    /// A whole-number count, with the same variable affordance every other number has.
    ///
    /// **Here rather than as a count widget of its own (D9).** A pattern's copy count is told the
    /// same kind of thing as a radius — a number, or the name of a variable — so it is told it the
    /// same way: the dim V that names it, the bright V carrying the parameter's name, the `×` back to
    /// a plain number, and the disabled box reading `Computed: …` when an expression drives it. The
    /// only difference is the grain, which is why that is the only argument this adds.
    pub(crate) fn count_field(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        current: ParamRef,
        range: std::ops::RangeInclusive<f64>,
        hover: &str,
        suffix: &str,
    ) -> (ParamRef, bool) {
        self.number_field(
            ui,
            label,
            current,
            // One per drag step. A count has no sub-unit to scrub through, so the speed that reads
            // well for millimetres would make a drag feel like it was sticking.
            1.0,
            range,
            hover,
            suffix,
            Naming::Offered,
            Labelled::Row,
            Grain::Whole,
        )
    }

    /// [`Self::parametric_field_in`], saying whether this number offers to be named, where its
    /// label goes and whether it counts things. See [`Naming`], [`Labelled`] and [`Grain`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn number_field(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        current: ParamRef,
        speed: f64,
        range: std::ops::RangeInclusive<f64>,
        hover: &str,
        suffix: &str,
        naming: Naming,
        where_: Labelled,
        grain: Grain,
    ) -> (ParamRef, bool) {
        let mut reference = current;
        let mut changed = false;
        // A parameter this row stopped referring to, dropped after the graph has been told.
        let mut release: Option<ParamId> = None;
        ui.horizontal(|ui| {
            grid_gaps(ui);
            match where_ {
                Labelled::Row => row_label(ui, label),
                // Allocated at a fixed width even for the empty label a lone number passes, so
                // that number starts where the X of every triple above it starts.
                Labelled::Cell => {
                    ui.add_sized(
                        egui::vec2(AXIS_LETTER_W, ui.spacing().interact_size.y),
                        egui::Label::new(label),
                    );
                }
            }
            let mut value = self.resolve(current);
            // A computed parameter shows its number and does not take a drag: the number is
            // the expression's output, and the place to change it is the expression, on the
            // parameter's own row.
            let computed = match current {
                ParamRef::Named(id) => self
                    .doc
                    .parameters
                    .get(&id)
                    .and_then(|p| p.value.expression())
                    .map(|e| e.source().to_string()),
                _ => None,
            };
            // A fixed box rather than one sized to what it happens to be showing: a column of
            // numbers that step in and out as their lengths change does not read as a column.
            let response = ui
                .add_enabled_ui(computed.is_none(), |ui| {
                    ui.add_sized(
                        egui::vec2(NUMBER_W, ui.spacing().interact_size.y),
                        {
                            let rate = scrub_rate(ui, value, speed);
                            let field =
                                egui::DragValue::new(&mut value).speed(rate).suffix(suffix);
                            match grain {
                                Grain::Continuous => field,
                                // **The range goes through to the widget here, not only to the
                                // commit.** A continuous value is clamped where it is written and
                                // the overshoot is invisible; a count dragged below one is a
                                // refusal, so the floor has to be felt in the hand. `fixed_decimals`
                                // keeps the drag from showing a fraction it would then reject.
                                //
                                // **`clamp_existing_to_range(false)`, and leaving it out was a real
                                // bug.** egui defaults it to *true*, and its own doc says what that
                                // means: "all values are clamped to this range, even when not
                                // interacted with". So the widget rewrote the value it was handed on
                                // every frame it drew, `set` it, and marked itself changed — with no
                                // pointer anywhere near it, and whether or not the ui was enabled.
                                // Downstream of that, `set_value` writes a `ParamRef::Named`
                                // count's number into the *parameter*, so a variable the whole
                                // document reads was silently rewritten to 512 and every other
                                // expression reading it moved. It also made the `capped at` note
                                // unreachable for exactly the two cases it exists for, because the
                                // value was already 512 by the time the note was computed.
                                //
                                // The clamp belongs on the commit below, where it only runs when
                                // somebody actually changed something.
                                Grain::Whole => field
                                    .fixed_decimals(0)
                                    .range(range.clone())
                                    .clamp_existing_to_range(false),
                            }
                        },
                    )
                })
                .inner;
            // The first number of a block a session just made, waiting for the keyboard: focused,
            // a `DragValue` shows as text with its value selected, so typing replaces it.
            if self.focus_first_number && computed.is_none() {
                self.focus_first_number = false;
                response.request_focus();
                select_whole_number(ui.ctx(), response.id);
            }
            if self.drawing_pending && computed.is_none() {
                self.pending_numbers.push(response.id);
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    self.keep_pending = true;
                }
            }
            if let Some(expression) = &computed {
                response.clone().on_disabled_hover_text(format!(
                    "Computed: {expression}. Change it on the parameter's own row at the top                      of the list."
                ));
            } else if !hover.is_empty() {
                response.clone().on_hover_text(hover);
            }
            if response.changed() {
                // The range comes from the caller because it is a property of the quantity: a
                // radius has no surface at zero or below, while a coordinate is perfectly
                // happy there and negative besides. One clamp for both is how typing 0 into a
                // centre used to produce 0.001.
                // **Rounded for a whole-number quantity, because `fixed_decimals` is a formatter and
                // nothing more.** egui rounds the drag path and the arrow keys, and does *not* round
                // the typed-text path — so typing `3.5` into a count committed 3.5 while the box
                // displayed `3`, which is the stored value and the shown value disagreeing in the one
                // widget built to stop that.
                //
                // Rounded here *and* on the way out in `pattern_count`, which is not duplication:
                // this one keeps the stored number whole where somebody typed it, so the box and the
                // document agree; that one keeps the *built* count whole however a variable drives
                // it. A value typed here is the only one this line ever sees.
                let committed = match grain {
                    Grain::Continuous => value,
                    Grain::Whole => value.round(),
                };
                reference =
                    self.set_value(current, committed.clamp(*range.start(), *range.end()));
                changed = true;
                // A named value is *in* the document, so a graph already referencing it is
                // showing something out of date the moment this changes — that is the whole
                // difference from a literal, which is only the tool's pending state until
                // Apply. Rebuilding per drag frame costs what a cage drag costs.
                if matches!(current, ParamRef::Named(_)) {
                    self.rebuild_render_resources();
                }
            }

            match current {
                // Nothing to mark and nothing offered, so the column holds a blank of the
                // mark's width rather than closing up -- a row whose numbers shift sideways
                // depending on whether the one beside them is named would not line up with
                // anything.
                ParamRef::Literal(_) if naming == Naming::OnlyWhenSet => {
                    ui.add_space(MARK_W);
                }
                ParamRef::Literal(v) => {
                    // **Name a new one, or join one that exists.** `fx` only ever made a *new*
                    // parameter, so every value that was named got a name of its own and there
                    // was no way to point a second block at the first -- which is the whole
                    // point of naming a number. One radius shared by fifty spheres was not
                    // expressible: fifty `fx` clicks made fifty parameters.
                    //
                    // A menu rather than a second button, because these are one decision with
                    // two answers: this value is a variable, and here is which.
                    let existing = named_values(self);
                    // **A green V, the same mark a named block wears in the list.** This said
                    // `fx`, which is a spreadsheet's word for a formula and not this
                    // application's word for anything -- while the block list has marked a
                    // variable with a green V since naming a block started meaning something.
                    // One idea was wearing two badges, and the one on the *field* was the one
                    // nobody could guess.
                    // Dim: an offer, not a fact. It brightens once this value is a variable.
                    let mark = egui::RichText::new("V").color(VARIABLE_OFF);
                    ui.menu_button(mark, |ui| {
                        if ui
                            .button("Name this value")
                            .on_hover_text(
                                "Makes a new parameter from this number, so other features can \
                                 be driven by it",
                            )
                            .clicked()
                        {
                            let name = unique_parameter_name(&self.doc, label);
                            // In the document now and on the history with the graph edit that
                            // writes `reference` back, so one undo takes back both -- or off the
                            // history, with a reference no graph edit writes (the draft pull).
                            let id =
                                name_value(&mut self.doc, &mut self.parameter_edits, name, v);
                            reference = ParamRef::Named(id);
                            changed = true;
                            // **Focused, because the invented name is a placeholder.** `x` and
                            // then `x_2` say nothing about what the number is for, and the row
                            // that holds them is in another panel among all the others. The
                            // next thing typed replaces it.
                            self.naming_parameter = Some(id);
                            self.mark_changed();
                            ui.close();
                        }
                        if existing.is_empty() {
                            return;
                        }
                        ui.separator();
                        ui.label(egui::RichText::new("or use").weak());
                        for (pid, name, value) in &existing {
                            // The value beside the name, because taking one *replaces* the
                            // number in this field and the person is entitled to know with
                            // what before they click. A computed one shows its expression
                            // too, so `radius + 10` is recognisably the Number just made.
                            let computed = self
                                .doc
                                .parameters
                                .get(pid)
                                .and_then(|p| p.value.expression())
                                .map(|e| format!(" = {}", e.source()))
                                .unwrap_or_default();
                            if ui
                                .button(format!("{name}{computed}  ({})", readout(*value)))
                                .on_hover_text(
                                    "Drive this by a parameter that already exists — \
                                     change it once and everything using it follows",
                                )
                                .clicked()
                            {
                                reference = ParamRef::Named(*pid);
                                changed = true;
                                self.mark_changed();
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_hover_text(
                        "Make this value a named parameter, or drive it by one that already \
                         exists",
                    );
                }
                // A driven value has no number to name, and its own controls are shown
                // instead of this row — so nothing routes here. Named for completeness rather
                // than reached.
                ParamRef::Field(_) => {}
                ParamRef::Named(id) => {
                    let name = self
                        .doc
                        .parameters
                        .get(&id)
                        .map(|p| p.name.clone())
                        .unwrap_or_else(|| "missing".to_string());
                    ui.label(egui::RichText::new("V").color(VARIABLE_ON).strong())
                        .on_hover_text(
                            "A variable: this value is a named parameter, and has a block of                              its own at the top of the list",
                        );
                    ui.label(egui::RichText::new(name).italics())
                        .on_hover_text(
                            "Driven by a document parameter — change it once and every field                              using it follows. See Parameters below.",
                        );
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("\u{d7}")
                                    .color(egui::Color32::from_rgb(220, 80, 80))
                                    .strong(),
                            )
                            .small(),
                        )
                        .on_hover_text("Back to a plain number, keeping the current value")
                        .clicked()
                    {
                        reference = ParamRef::Literal(self.resolve(current));
                        changed = true;
                        // The parameter goes too, if this was the last thing reading it. It
                        // used to stay in the document forever, so naming a value and changing
                        // your mind left an entry nothing referred to and nothing could remove
                        // — press fx and x a few times and the panel fills with x, x_2, x_3.
                        //
                        // Deferred, because `reference` has not been written back into the
                        // node yet: dropping it here would count this node as a reader and
                        // keep the parameter alive. It goes with the graph edit that writes the
                        // number back, as one undo, so undoing the × brings the parameter back
                        // with the reference to it.
                        release = Some(id);
                    }
                }
            }
        });
        if let Some(id) = release {
            self.parameter_edits.released.push(id);
        }
        (reference, changed)
    }

    /// Settles what this frame's fields did to parameters that no graph edit took: one let go of
    /// that nothing reads now goes on the history, and one named for a reference no graph edit
    /// wrote stays off it (see the free function of the same name).
    ///
    /// Called once every block has been drawn, so "nothing reads it" is the truth about the
    /// document as it now stands rather than as it stood a moment ago.
    fn record_parameter_edits(&mut self) {
        let done = record_parameter_edits(
            &mut self.stack,
            &mut self.doc,
            &mut self.gesture,
            &mut self.parameter_edits,
        );
        self.after_parameter_entry(done);
    }

    /// What follows a parameter going on the history in an entry of its own.
    ///
    /// The material's gesture ends with it as the graph's has, since a drag on a slider folds into
    /// whatever entry is on top and would take this one off the history.
    pub(crate) fn after_parameter_entry(&mut self, done: Result<bool, fieldform_ops::OpError>) {
        match done {
            Ok(true) => {
                self.material_gesture = None;
                self.mark_changed();
            }
            Ok(false) => {}
            Err(e) => self.warn(e.to_string()),
        }
    }

    /// Whether anything in the document reads this parameter — see [`Document::parameter_is_read`].
    pub(crate) fn parameter_is_used(&self, id: ParamId) -> bool {
        self.doc.parameter_is_read(id, None)
    }


    /// The part's material: a preset to pick, with the channels underneath as the escape hatch.
    ///
    /// Under the part's name rather than in a block, because a material is a fact about the
    /// part (D1): the whole part is aluminium, the way the whole part has a name. The preset
    /// list is what a designer reaches for; the channels are for whoever knows what roughness
    /// means, and are shown closed so they do not become the primary interface by accident
    /// (`shading-spec.md` §7).
    fn material_row(&mut self, ui: &mut egui::Ui) {
        let material = self
            .doc
            .parts
            .get(&self.active_part)
            .and_then(|p| p.material.clone())
            .unwrap_or_default();
        // Derived from the channels, never stored — see `fieldform_scene::material`.
        let label = material.preset_name().unwrap_or("Custom");
        let mut next: Option<Material> = None;
        ui.horizontal(|ui| {
            ui.label("Preset:");
            egui::ComboBox::from_id_salt("part_material")
                .selected_text(label)
                .show_ui(ui, |ui| {
                    for preset in Material::presets() {
                        let is_this = material.preset_name() == Some(preset.name);
                        if ui
                            .selectable_label(is_this, preset.name)
                            .on_hover_text(preset.hint)
                            .clicked()
                        {
                            next = Some(Material::from(preset.material));
                        }
                    }
                });
        });
        egui::CollapsingHeader::new("Appearance")
            .id_salt("part_appearance")
            .show(ui, |ui| {
                let mut m = material.clone();
                let mut edited = false;
                // **The kind is not offered here, only obeyed.** It decides which rows appear
                // — clay has no flake, chrome has no lacquer — and it comes from the preset,
                // which is how anyone actually chooses one: you pick "car paint", not "paint,
                // and then these seven numbers". Offered as a control of its own it was a
                // second way to say what the preset had already said, and the two could
                // disagree on screen. `Material::with_kind` is still there for when there is a
                // reason to expose it; nothing calls it today, and that is the honest state.
                ui.horizontal(|ui| {
                    // What the colour *is* depends on the kind: a metal's is its reflectance.
                    ui.label(m.kind.color_label());
                    // egui's picker works in linear light, which is what the material holds.
                    let mut rgb = m.base_color.map(|c| c as f32);
                    if egui::color_picker::color_edit_button_rgb(ui, &mut rgb).changed() {
                        m.base_color = rgb.map(f64::from);
                        edited = true;
                    }
                });
                for slot in m.kind.channels() {
                    let mut value = slot.get(&m);
                    if channel(ui, slot.label(), &mut value, slot.range(), slot.hint()) {
                        slot.set(&mut m, value);
                        edited = true;
                    }
                }
                if edited {
                    next = Some(m.normalised());
                }
            });
        if let Some(m) = next {
            // The default is what `None` means; storing a copy of it would be a second
            // description of one appearance.
            self.set_part_material(if m == Material::default() { None } else { Some(m) });
        }
    }

    /// Which room the part is lit by, as a row beside the material.
    ///
    /// **Beside the material because they are one question.** What a surface looks like is the
    /// finish and the light together — chrome in a featureless room is a grey blob, and the
    /// zebra environment exists precisely to be looked at *through* a chrome finish. It was in
    /// the View menu, which meant judging a material was a trip across the window and back for
    /// every comparison.
    ///
    /// Chosen through `set_environment` rather than by assignment, so the first use of a room
    /// still bakes its maps and says so.
    fn environment_row(&mut self, ui: &mut egui::Ui) {
        let current = self.environment;
        let mut next = None;
        ui.horizontal(|ui| {
            ui.label("Lighting:");
            egui::ComboBox::from_id_salt("part_environment")
                .selected_text(current.label())
                .show_ui(ui, |ui| {
                    for choice in fieldform_gpu::EnvironmentChoice::ALL {
                        if ui
                            .selectable_label(current == choice, choice.label())
                            .on_hover_text(choice.hint())
                            .clicked()
                            && choice != current
                        {
                            next = Some(choice);
                        }
                    }
                });
        });
        if let Some(choice) = next {
            self.set_environment(choice);
        }
    }

    /// The parameter panel: docked, non-modal, and bound to the part being edited.
    ///
    /// This is the *above the line* treatment from the interface-modality design note.
    /// A pipe's radius and a blend's fillet are parametric — serialized, re-evaluated, and part
    /// of what a save writes — so they belong in a panel bound to the part rather than in the
    /// toolbar's chrome. Non-modal and always visible while it applies, because the viewport
    /// updates as the numbers change and a dialog would hide the thing being adjusted.
    ///
    /// Cage editing is deliberately not here. That is below the line: destructive, current
    /// state, no node and nothing to re-evaluate, so it is edited in the viewport directly.
    pub(crate) fn inspector(&mut self, ui: &mut egui::Ui) {
        // Left, not right: mainstream CAD tools keep the tree and its parameters on that
        // side, and a modeller coming from any of them looks there first.
        // Dragged by its edge, and bounded at both ends. Unbounded, the panel sized itself
        // to its widest label — and a label is the whole chain of what made a body, so
        // shelling, offsetting, placing, mirroring and patterning one cube produced a row of
        // text wide enough to squeeze the viewport to a strip. The thing being modelled lost
        // room every time it gained an operation, which is precisely backwards.
        //
        // **That is fixed by the truncation set below, not by the ceiling.** The two were
        // introduced together and it is worth keeping them apart: nothing in the panel can move
        // the divider any more, so the ceiling is no longer protecting the viewport from the
        // content — only from the person dragging, who is entitled to a wide panel if they want
        // to read a long block name. It is a fraction of the window rather than a fixed width so
        // the same limit holds on a laptop and on a monitor, and it is there now only so the
        // viewport cannot be dragged away to nothing.
        let full = ui.ctx().content_rect().width();
        let widest = (full * INSPECTOR_MAX_FRACTION).max(INSPECTOR_MIN_WIDTH);
        egui::Panel::left("fieldform_inspector")
            .resizable(true)
            .default_size(INSPECTOR_DEFAULT_WIDTH.min(widest))
            .size_range(INSPECTOR_MIN_WIDTH..=widest)
            .show_inside(ui, |ui| {
                // Text truncates rather than demanding room. A panel sized to its content moves
                // its own divider whenever any string in it changes width — and strings in here
                // change width constantly: a curve's summary carries its length, so dragging a
                // control point sideways takes it through 99.9 to 100.0 and the divider jumps
                // once per digit. Reported as the divider "jumping around", and only for curves,
                // because a body's rows do not recompute a number as it is dragged.
                //
                // Set on the panel rather than per label, since the next label to carry a live
                // number would bring the bug back with it.
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                // The console sits under the blocks and keeps its place while they scroll,
                // which is what a command line is for: always there, never in the way.
                egui::Panel::bottom("fieldform_console")
                    // Draggable along its top edge, so the log can be grown when there is
                    // something to read back through and shrunk when the blocks matter more.
                    .resizable(true)
                    .default_size(170.0)
                    .size_range(64.0..=560.0)
                    .show_inside(ui, |ui| self.console(ui));
                let part_name = self
                    .doc
                    .parts
                    .get(&self.active_part)
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| "part".to_string());
                ui.heading(&part_name);
                // A one-line read of the part's graph, which is a preview of the same
                // structure the graph UI will eventually draw rather than a separate concept.
                // It is the closest thing to a history cue that does not reintroduce a feature
                // tree — the direct layer's undo depends on commands applying against current
                // state, not a replayed log, and D4 already rejected the tree.
                ui.label(egui::RichText::new(self.graph_summary()).weak())
                    .on_hover_text(self.graph_summary());
                // **One line, always, whether or not there is anything to say.** This label
                // appeared only when the part would not compile, and everything below it moved
                // down a line when it did. A field being dragged sits below it: dragging a
                // revolve's angle through nought made the part uncompilable, the label
                // appeared, and the next frame the angle climbed back to three or five degrees
                // the part compiled again, the label went, and the field jumped back up from
                // under the pointer -- egui ends a drag whose widget has moved, so the angle
                // stuck there. Reported as "it gets stuck at three or five degrees, I think it's
                // trying to display this error message". Truncated to the line, with the whole
                // message on hover; the console has it in full as well.
                let line = ui.spacing().interact_size.y;
                let (slot, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), line),
                    egui::Sense::hover(),
                );
                if let Some(why) = self.compile_error() {
                    ui.put(
                        slot,
                        egui::Label::new(egui::RichText::new(&why).color(ui.visuals().warn_fg_color))
                            .truncate(),
                    )
                    .on_hover_text(why);
                }
                ui.separator();
                // Pinned under the list rather than drawn after it.
                //
                // These used to sit inside the scroll area, below every block, so expanding
                // one moved them down the panel. egui runs more than one pass while a block
                // animates open, and it checks each pass against the last by *rectangle* — so
                // a row that moves leaves its old rectangle to some other widget, which is
                // reported as "changed id between passes" and painted over. Three widgets
                // here, three markers in the corner.
                //
                // Two earlier attempts scoped the ids instead. The ids were never the problem:
                // nothing was colliding, the row was moving. Reading the rectangles out of the
                // log named the three widgets in a minute, which is what the backlog item said
                // to do and what should have happened before either guess.
                egui::Panel::bottom("display_settings").show_inside(ui, |ui| {
                ui.separator();
                // **One heading over all of it, and shut until it is wanted.** These are the
                // settings for *looking* at the part rather than for what the part is — the
                // finish, the room it is lit by, the mesh, the draft overlay — and open they
                // stood permanently between the block list and the bottom of the panel. Behind
                // one expander they cost a line until somebody is judging a surface, which is
                // the only time all four are interesting at once.
                //
                // Material is inside rather than beside it for the same reason: a finish and
                // the light on it are one question, and two headings a click apart taught that
                // they were separate settings.
                egui::CollapsingHeader::new("Display")
                    .id_salt("display_settings_section")
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Material").weak());
                        self.material_row(ui);
                        self.environment_row(ui);
                        ui.separator();
                        ui.horizontal(|ui| {
                            // **"Cage detail", because that is what it sets.** Nothing displays
                            // a limit-surface mesh -- the viewport is field-traced -- so "Mesh"
                            // read as a setting for the model, when what it sets is the
                            // tolerance the traced view bakes the *control cage* at, and the one
                            // export will mesh at once export exists. Asked what it did; the
                            // answer was not in the word. The field's own doc has said it was
                            // worth renaming since the tracer landed.
                            ui.label("Cage detail:").on_hover_text(
                                "How finely the control cage is baked for display, in cells \
                                 across the part. Bodies are traced from the field itself and \
                                 are not affected.",
                            );
                            let mut resolution = self.mesh_resolution as u32;
                            ui.add(
                                egui::DragValue::new(&mut resolution).speed(1.0).suffix(" cells"),
                            );
                            // Below about 8 cells nothing is recognisable; above ~256 the
                            // brute-force field evaluation stops being interactive on this
                            // machine.
                            let resolution = (resolution as usize).clamp(8, 256);
                            // Changing this used to store the number and stop there, so nothing
                            // was rebuilt until some *other* edit happened to trigger one. The
                            // control looked like it did nothing, which — until the next cage
                            // edit — was true.
                            if resolution != self.mesh_resolution {
                                self.mesh_resolution = resolution;
                                self.rebuild_render_resources();
                            }
                        });
                        self.draft_controls(ui);
                    });
                });
                egui::ScrollArea::vertical().id_salt("blocks").show(ui, |ui| {

                // --- what this part is made of ---
                //
                // One row per thing, in dependency order: curves first, then the nodes that
                // use them, then whatever uses those. Each row expands in place to show its
                // own parameters, and names its inputs rather than nesting under them — a
                // boolean has two operands and only one of them could be its parent, so a
                // strict tree would have to invent a hierarchy that the graph does not have.
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Blocks").weak());
                    // **Only once there is something to undo.** A button offering to reset an
                    // order nobody has changed is a button that never does anything, and it would
                    // sit there in every document. Its presence is also the only indication that
                    // the list *is* arranged rather than simply in the order things were made,
                    // which is worth saying where the list is read.
                    if self.is_arranged() {
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui
                                    .small_button("Reset order")
                                    .on_hover_text(
                                        "Put the blocks back in the order they were made",
                                    )
                                    .clicked()
                                {
                                    self.clear_arrangement();
                                }
                            },
                        );
                    }
                });

                // Curves and bodies in one list, in the order they arrived — see
                // `ordered_blocks`, and `ordered_nodes` for why the bodies' own order needs
                // more than a sort. The profile rows are brought up to date first, which is a
                // counter compare when no curve has changed.
                self.refresh_profiles();
                // A tree now rather than a flat list: a block inside the one block that owns it is
                // drawn inside that block's row. `nested_blocks` arranges it, and
                // `Document::nesting_in` reads it from the stored ownership — kept from one frame
                // to the next by `list_readings`, and taken once here for the arrangement and for
                // the rows drawn from it.
                let list = self.list_readings();
                let nesting = &list.nesting;
                let rows = self.nested_blocks(nesting);
                // **A line between the rows, not a target on them.** A drop means "put it in front
                // of that", and only a body has a rank to take: a parameter, a curve, a surface and
                // an output sit in fixed sections of the list (`ordered_blocks` says why each is
                // where it is), and a nested row is pinned above whatever consumes it. So a gutter
                // is allocated at the top of each body's branch and nowhere else, which is what
                // makes the insertion line appear only where it means something and only where the
                // block will actually land.
                let starts = branch_starts(&rows, self.part_graph());
                // **What is being dragged decides which gutters answer.** A nested block is pulled
                // out by a drop anywhere on the list, so every gutter is a target for it; a body
                // may only take another body's rank, so its targets are the branch starts below.
                // Read once per frame rather than per row.
                //
                // **The test is "is drawn inside something", not "is not a body".** Every row is a
                // drag source now, so a variable that is not a body is draggable too -- and a
                // list-drop does nothing for it, because its row is at the top level and §7.2's
                // drag-out finds no slot to empty. Asking the nesting directly is what keeps the insertion lines
                // honest: they appear only where letting go would actually move something.
                let dragging_nested = egui::DragAndDrop::payload::<DraggedBlock>(ui.ctx())
                    .is_some_and(|d| nesting.is_nested(d.id));
                for (i, row) in rows.iter().enumerate() {
                    let first_of_branch =
                        starts[i].is_some() && (i == 0 || starts[i - 1] != starts[i]);
                    let onto = starts[i].filter(|_| first_of_branch);
                    if let Some(dragged) =
                        drop_gutter(ui, onto.is_some() || dragging_nested)
                    {
                        self.drop_on_the_list(dragged, onto);
                    }
                    self.block_row(ui, row, nesting, None);
                }

                // The empty space under the rows deselects, the same as empty space in the
                // viewport does. Without it the only way to stop pointing at something was to
                // point at something else, which makes "none of this" the one intention the
                // list cannot express — and leaves the manipulator sitting on geometry that is
                // no longer being worked on.
                //
                // Allocated rather than merely left blank: egui gives a click to whatever
                // claimed the rectangle under it, and nothing had.
                let below = ui.available_size_before_wrap();
                if below.y > 1.0 {
                    // **One allocation answering both.** A drop here means the end of the list
                    // (see `arrange_block_last`), and a click here still deselects -- which is
                    // what the paragraph above is about. Allocating twice would give the second
                    // rectangle no room, since the first has already taken the space: the click
                    // target would be a hairline and "none of this" would stop being expressible.
                    //
                    // **Allocated and painted by hand, because `dnd_drop_zone` paints a filled
                    // box.** egui 0.34's `Ui::dnd_drop_zone` assigns `frame.fill = style.bg_fill`
                    // unconditionally (`egui/src/ui.rs`), falling back to `widgets.inactive` when
                    // nothing is being dragged -- so the `Frame::new()` handed to it was
                    // overwritten and the whole empty remainder of the panel sat there as a grey
                    // slab under the blocks. Reported by the owner: "there's a gray box underneath
                    // the blocks in the work area ... I don't think it's really necessary." The
                    // region *is* necessary; its fill never was. So this does what `drop_gutter`
                    // does -- allocate, take the payload off the response, and draw only while a
                    // drag is in flight.
                    let (rect, empty) = ui.allocate_exact_size(below, egui::Sense::click());
                    if egui::DragAndDrop::has_payload_of_type::<DraggedBlock>(ui.ctx())
                        && ui.is_rect_visible(rect)
                    {
                        // At the top edge, because this stands for the place after the last row --
                        // the same insertion line a gutter draws, in the one spot no gutter covers.
                        let under = empty.contains_pointer();
                        ui.painter().rect_filled(
                            egui::Rect::from_center_size(
                                egui::pos2(rect.center().x, rect.top()),
                                egui::vec2(rect.width(), if under { 3.0 } else { 1.0 }),
                            ),
                            1.0,
                            if under {
                                ui.visuals().selection.bg_fill
                            } else {
                                ui.visuals().widgets.noninteractive.bg_stroke.color
                            },
                        );
                    }
                    if let Some(dragged) = empty.dnd_release_payload::<DraggedBlock>() {
                        self.drop_on_the_list(*dragged, None);
                    }
                    if empty.clicked() {
                        self.select_body(None);
                        self.selected_block = None;
                        self.selected_curve = None;
                        self.clear_point_selection();
                        self.selection.clear();
                    }
                }

                // After every block has been drawn, so the node that stopped reading a
                // parameter has already been written back and the "does anything still use
                // this" question has the right answer.
                self.record_parameter_edits();



                });
            });
    }
}

/// Selects the whole of number field `id`'s text, so what is typed next replaces it.
///
/// egui selects a `DragValue`'s text itself only on the frame it *gains* focus, and it decides that
/// against the previous frame's focus -- so focus handed over by the application, rather than by a
/// click or egui's own Tab, arrives with the caret at the end and a typed 5 makes "0.05". Any text
/// it was holding is dropped too, so the field starts from the number as it is now. The range runs
/// past the end and is clamped to the text when the field is drawn.
pub(crate) fn select_whole_number(ctx: &egui::Context, id: egui::Id) {
    ctx.data_mut(|d| d.remove_temp::<String>(id));
    let mut state = egui::TextEdit::load_state(ctx, id).unwrap_or_default();
    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
        egui::text::CCursor::new(0),
        egui::text::CCursor::new(1 << 16),
    )));
    state.store(ctx, id);
}

/// Hard and the three shapes a blend offers, in the same words: the first row of the one editor
/// rims and edges share. Hard is a reach of nought, which is what a hard cut *is* -- there is no
/// flag beside the number to disagree with it -- and picking a shape from hard starts at a
/// millimetre, visible on anything and refused by name on a solid too small for it.
fn cut_shape_row(
    ui: &mut egui::Ui,
    label: &str,
    hard_hover: &str,
    cut: fieldform_scene::Fillet,
) -> fieldform_scene::Fillet {
    use fieldform_scene::{Continuity, Fillet, ParamRef};
    let hard = cut.is_plainly_hard();
    let mut new = cut;
    ui.horizontal(|ui| {
        row_label(ui, label);
        if ui.add(mode_button(hard, "Hard")).on_hover_text(hard_hover).clicked() {
            new = Fillet::hard();
        }
        for shape in Continuity::ALL {
            if ui
                .add(mode_button(!hard && cut.continuity == shape, shape.label()))
                .on_hover_text(fillet_shape_hover(shape))
                .clicked()
            {
                new.continuity = shape;
                if hard {
                    new.radius = ParamRef::Literal(1.0);
                }
            }
        }
    });
    new
}
