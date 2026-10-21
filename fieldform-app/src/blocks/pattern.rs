// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Pattern block's panel.
//!
//! Lifted out of `node_block_controls` whole, as the lattice's was. A body repeated in a row,
//! a grid or round an axis, with a count that may be a variable and copies that may be
//! centred, skipped or rounded into each other, is a screen of controls for one kind.

use crate::*;
use fieldform_scene::Operand;

impl App {
    /// The Pattern block: what is repeated, how many times and along what, whether the
    /// original is held, which copies are left out, and the fillet where copies meet.
    ///
    /// Takes the node's six fields by value, as the lattice panel takes its four: the caller has
    /// already cloned the node, and the panel writes a whole replacement when anything changed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pattern_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
        count: ParamRef,
        kind: PatternKind,
        join: bool,
        fillet: Fillet,
        options: fieldform_scene::PatternOptions,
    ) {
        ui.label(
            "Repeats this body and unions the copies. One block, however many there \
             are — change the count and the pattern rebuilds.",
        );
        ui.separator();
        // **The same row a boolean's operands get (D9), rather than `pick_input`'s.** The two draw
        // the one question — which block does this read — and only this one has the empty state the
        // slot can now be in; `operand_input` is `pick_input`'s three lines with an `Operand` in
        // place of its `NodeId`, so the hint and the separator under the row are unchanged.
        let picked = self.operand_input(ui, id, input);

        let mut new_kind = kind;
        ui.horizontal(|ui| {
            row_label(ui, "Repeat:");
            let radial = matches!(new_kind, PatternKind::Radial { .. });
            if ui.add(mode_button(!radial, "In a row")).clicked() && radial {
                new_kind = PatternKind::Linear {
                    step: [
                        ParamRef::Literal(DEFAULT_BODY_MM * 0.5),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                    ],
                };
            }
            if ui.add(mode_button(radial, "Around an axis")).clicked() && !radial {
                new_kind = PatternKind::Radial { frame: fieldform_scene::Frame::at_along(Axis::Z, [ParamRef::Literal(0.0); 3]) };
            }
        });

        // The count includes the original, which is the off-by-one every pattern tool
        // has to settle: six means six bodies.
        //
        // **The same field every other number gets, so it can be a variable.** Asked for by
        // the owner: "in pattern the number of copies should also be able to be declared as
        // a var". `count_field` is `number_field` with a whole-number grain, so the V, the
        // naming menu and the `Computed:` readout all arrive without a second spelling.
        let (new_count, count_changed) = self.count_field(
            ui,
            "Copies:",
            count.clone(),
            1.0..=f64::from(fieldform_scene::PATTERN_COUNT_MAX),
            "Including the original. The body's expression is written out once per \
             copy, so a large count is a large shader — 512 copies of a \
             sphere measure 478 KB of it, and a real body is more, which is why \
             that is the ceiling. A variable may drive this, and a count is always a \
             whole number of copies: a value that is not one is rounded to the nearest \
             and the row says so. Use floor() or ceil() in the expression to say which \
             way you want it instead.",
            " total",
        );
        // **Said where the number is typed, because quantising in silence is the one thing
        // `arc_count`'s doc forbids.** Its clause is that the number typed and the geometry
        // built may not disagree *without saying so* -- so once a fraction rounds rather
        // than refusing, this line is what keeps the rounding honest. Same for the ceiling,
        // which caps a count that reaches past what a driver will compile.
        match fieldform_scene::pattern_count(&self.doc, &new_count) {
            Ok((built, fit)) => {
                if let Some(note) = fit.say(built) {
                    let capped = matches!(fit, fieldform_scene::CountFit::Capped(_));
                    let colour = if capped {
                        ui.visuals().warn_fg_color
                    } else {
                        ui.visuals().weak_text_color()
                    };
                    self.count_note(
                        ui,
                        &note,
                        colour,
                        Some(if capped {
                            "The expression works out to more copies than a pattern will \
                             build. The body is written into the shader once per copy, so \
                             the ceiling is what the driver can compile rather than a \
                             modelling limit."
                        } else {
                            "A count is always a whole number of copies, so a value that \
                             is not one is rounded to the nearest. The variable itself is \
                             untouched — anything else reading it still sees what it says \
                             — and floor() or ceil() in the expression will say which way \
                             you want it instead."
                        }),
                    );
                }
            }
            Err(why) => {
                let colour = ui.visuals().error_fg_color;
                self.count_note(ui, &why.say(), colour, None);
            }
        }

        let mut new_options = options.clone();
        new_kind = match new_kind {
            PatternKind::Linear { step } => {
                let whole_run = new_options.spacing == fieldform_scene::Spacing::Extent;
                let new_step = self.centre_fields(
                    ui,
                    if whole_run { "Run:" } else { "Step:" },
                    step,
                );
                // **What the vector means is said beside it**, because the same three numbers
                // are a gap in one reading and the whole length in the other, and nothing
                // about the number says which it is.
                ui.horizontal(|ui| {
                    row_label(ui, "Spacing:");
                    for (spacing, text, hover) in [
                        (
                            fieldform_scene::Spacing::Step,
                            "Gap",
                            "The vector is the gap between neighbours, so the last copy is \
                             one gap fewer than the count away from the first.",
                        ),
                        (
                            fieldform_scene::Spacing::Extent,
                            "Whole run",
                            "The vector is the whole run from the first copy to the last, \
                             so the gap is the run over one less than the count. It needs \
                             at least two copies.",
                        ),
                    ] {
                        if ui
                            .add(mode_button(new_options.spacing == spacing, text))
                            .on_hover_text(hover)
                            .clicked()
                        {
                            new_options.spacing = spacing;
                        }
                    }
                });
                // **A second direction makes a grid**: the row repeated along another step,
                // so `count × rows` copies, row after row. Off by default, and a plain row is
                // what a pattern has always been.
                let mut has_second = new_options.second.is_some();
                if ui
                    .checkbox(&mut has_second, "Second direction")
                    .on_hover_text(
                        "Repeat the row along another step, making rows and columns of \
                         copies. Copies are numbered along the first step, row after row.",
                    )
                    .changed()
                {
                    new_options.second = has_second.then(|| {
                        self.default_second_direction(new_step)
                    });
                }
                if let Some(second) = new_options.second.as_mut() {
                    second.step = self.centre_fields(
                        ui,
                        if whole_run { "Run 2:" } else { "Step 2:" },
                        second.step,
                    );
                    let (rows, _) = self.count_field(
                        ui,
                        "Rows:",
                        second.count,
                        1.0..=f64::from(fieldform_scene::PATTERN_COUNT_MAX),
                        "How many copies along the second step, including the row they \
                         repeat. The columns times the rows is what the ceiling of 512 \
                         counts.",
                        " rows",
                    );
                    second.count = rows;
                }
                PatternKind::Linear { step: new_step }
            }
            PatternKind::Radial { frame } => {
                let new_frame = self.frame_controls(ui, id, "About:", frame, None);
                let (angle, _) = self.parametric_field_in(
                    ui,
                    "Angle:",
                    new_options.angle,
                    1.0,
                    REVOLVE_DEGREES,
                    "How far round the copies spread. A full turn, 360\u{b0}, puts them \
                     evenly all the way round; less puts the first and last at the two \
                     ends of the angle.",
                    "\u{b0}",
                );
                new_options.angle = angle;
                let total = fieldform_scene::pattern_count(&self.doc, &new_count)
                    .map_or(1, |(n, _)| n);
                let degrees = self.resolve(new_options.angle);
                let note = if degrees >= fieldform_scene::FULL_TURN_DEGREES - 1e-9 {
                    format!(
                        "{:.1}\u{b0} apart, evenly around a full turn",
                        360.0 / f64::from(total),
                    )
                } else if total > 1 {
                    format!(
                        "{:.1}\u{b0} apart, over {degrees:.0}\u{b0}",
                        degrees / (f64::from(total) - 1.0),
                    )
                } else {
                    "one copy has nothing to spread over an angle".to_string()
                };
                ui.label(egui::RichText::new(note).small().weak());
                PatternKind::Radial { frame: new_frame }
            }
        };

        // **The same choice Mirror offers, spelled the same way (D9).** A block that
        // holds the body it patterns consumes it, and a consumed node is not a root --
        // so the body leaves the list and cannot be shown, hidden or handed to a
        // boolean on its own. Off is what the tool now builds; on is what documents
        // written before this existed mean, and they keep meaning it.
        let mut new_join = join;
        ui.horizontal(|ui| {
            row_label(ui, "Holds:");
            if ui.add(mode_button(!new_join, "Copies only")).clicked() {
                new_join = false;
            }
            if ui.add(mode_button(new_join, "With the original")).clicked() {
                new_join = true;
            }
        });
        ui.label(
            egui::RichText::new(if new_join {
                "The body is part of this block, so it is drawn here and not \
                 separately."
            } else {
                "The body stays a block of its own — hide it to leave a gap."
            })
            .small()
            .weak(),
        );

        // **Centred, and which copies to leave out.** Both are options on the node and both
        // change where a copy is or whether it exists, so they are said where the count is.
        ui.checkbox(&mut new_options.symmetric, "Centred on the body").on_hover_text(
            "Spread the copies evenly either side of the body instead of starting at it. \
             The body is one of the copies only when the count is odd — an even count puts \
             the middle between two of them.",
        );
        let columns = fieldform_scene::pattern_count(&self.doc, &new_count)
            .map_or(1, |(n, _)| n);
        // A grid's copies are numbered along the first step, row after row, so its dots are
        // drawn in rows of that many and the copy on the body is the grid's middle.
        let rows = match (&new_kind, &new_options.second) {
            (PatternKind::Linear { .. }, Some(second)) => {
                Some(fieldform_scene::pattern_count(&self.doc, &second.count)
                    .map_or(1, |(n, _)| n))
            }
            _ => None,
        };
        let (total, body_copy, per_line) = match rows {
            Some(rows) => (
                columns.saturating_mul(rows),
                fieldform_field::pattern_body_copy_grid(columns, rows, new_options.symmetric),
                Some(columns),
            ),
            None => (
                columns,
                fieldform_field::pattern_body_copy(columns, new_options.symmetric),
                None,
            ),
        };
        new_options.skip = skip_dots(ui, total, body_copy, &new_options.skip, per_line);
        // **The reason, where the numbers are typed.** A pattern the layout refuses draws
        // nothing, and the viewport would otherwise go on showing the last one that
        // worked. The count's own failures are said beside the count already.
        let mut trial = self.part_graph().clone();
        if let Some(slot) = trial.node_mut(id) {
            *slot = Node::Pattern {
                input: picked,
                count: new_count.clone(),
                kind: new_kind,
                join: new_join,
                fillet,
                options: new_options.clone(),
            };
        }
        if let Err(why) = fieldform_field::PatternLayout::of(&self.doc, &trial, id)
            && !matches!(why, fieldform_field::PatternRefusal::Count(_))
        {
            ui.colored_label(ui.visuals().error_fg_color, why.say());
        }

        // **The same fillet a blend takes, and the same editor (D9).** A fold is a
        // union, and a union can be rounded: copies that touch met in a hard seam
        // before this, which is what ribs, fins and lugs on a hub all want rounded.
        // Zero is a hard join and costs nothing, so there is no toggle beside it.
        ui.separator();
        let mut new_fillet = fillet;
        // **The shape a blend offers, in the same words.** A fold's fillet is the same
        // type and carried one all along, but the panel never let anyone choose it — so a
        // pattern could only ever be G1 however a blend beside it was set.
        ui.horizontal(|ui| {
            row_label(ui, "Shape:");
            // **No Chamfer here, by the owner's ruling (2026-10-17)** -- not offered disabled,
            // which is this panel's usual rule, but left out. A chamfer has no order-free form
            // across many copies (`round-chamfer-spec.md` §2): where three meet, the bevel
            // between two of them is cut again by the third, and which pair is cut first changes
            // the shape. The compile still refuses one by name, for a file or a line that asks.
            for shape in Continuity::ALL.into_iter().filter(|s| *s != Continuity::Chamfer) {
                if ui
                    .add(mode_button(new_fillet.continuity == shape, shape.label()))
                    .on_hover_text(fillet_shape_hover(shape))
                    .clicked()
                {
                    new_fillet.continuity = shape;
                }
            }
        });
        let (edited, _changed) = self.parametric_field(
            ui,
            fillet_label(new_fillet.continuity),
            fillet.radius,
            0.25,
            FROM_NOUGHT_MM,
            "",
        );
        new_fillet.radius = edited;
        if let Some(note) =
            fillet_bend_note(new_fillet.continuity, self.resolve(new_fillet.radius))
        {
            ui.label(egui::RichText::new(note).small().weak());
        }
        // Beside the number, because that is the only place saying so is honest --
        // the rule that travels with the type. See `approximate_blends`.
        // An empty slot has no normalization to be approximate about, so `source` is what is
        // asked and a hole contributes nothing — the row above it carries its own error.
        if let Some(why) = self.approximate_join(input.source()) {
            ui.colored_label(ui.visuals().warn_fg_color, "\u{2248} approximate")
                .on_hover_text(why);
        }
        ui.label(
            egui::RichText::new(
                "Rounds where the copies meet. Nought is a hard join.",
            )
            .small()
            .weak(),
        );

        if new_kind != kind
            || count_changed
            || new_count != count
            || picked != input
            || new_join != join
            || new_fillet != fillet
            || new_options != options
        {
            self.replace_node(
                id,
                Node::Pattern {
                    input: picked,
                    count: new_count.clone(),
                    kind: new_kind,
                    join: new_join,
                    fillet: new_fillet,
                    options: new_options,
                },
            );
        }
    }
}
