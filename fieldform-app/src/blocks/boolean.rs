// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Boolean and Blend blocks' panel.
//!
//! One panel for the two kinds, as it was one arm of `node_block_controls`: the Blend row is
//! what turns a boolean into a blend and back, so the panel is about the join rather than
//! about either kind. Lifted out whole, as the lattice's was; see `lattice.rs` for why the
//! larger arms move at all.

use crate::*;
use crate::blocks::operands::same_block_twice;
use fieldform_scene::Operand;

impl App {
    /// A boolean's list of bodies or a blend's pair: the operands, the operation, and the
    /// fillet that turns one kind into the other.
    ///
    /// **The whole node, where the other panels take its fields**: what this rebuilds may be
    /// the other kind, so it is compared against the node it was given rather than field by
    /// field. The caller has already cloned it, and only the arm that matched these two kinds
    /// calls this, which is what the `unreachable!` below leans on.
    pub(crate) fn boolean_controls(&mut self, ui: &mut egui::Ui, id: NodeId, node: Node) {
        // One panel for both, because the join control turns one into the other. A
        // boolean is a list; a blend is the pair it always was, since a fillet rolls
        // between two surfaces and no more.
        let (op, operands, fillet, continuity, second) = match &node {
            Node::Boolean { op, inputs } => (*op, inputs.clone(), None, None, None),
            Node::Blend { op, a, b, fillet } => {
                (*op, vec![*a, *b], Some(fillet.radius), Some(fillet.continuity), fillet.second)
            }
            _ => unreachable!("matched above"),
        };
        let is_list = fillet.is_none();

        // Operands first: which body is cut from which is the thing a subtraction gets
        // wrong, and it is not visible from the result. The first row of a subtraction
        // is the body kept and every other row is taken from it; a union or an
        // intersection treats its rows alike, and the labels say so.
        //
        // **A boolean takes no point.** A point encloses no volume, so `min` and `max`
        // have nothing of it to combine; a blend can reach *to* one, which is why the
        // rule is the boolean's and not the chooser's. A held loop's joints are points
        // and roots, so thirty of them in this list is what asked for it.
        let choices: Vec<(NodeId, String)> = self
            .operand_choices(id)
            .into_iter()
            .filter(|(n, _)| {
                !is_list || !matches!(self.part_graph().node(*n), Some(Node::Point { .. }))
            })
            .collect();
        // Each block the rows may read, with its refusal already on it, built once for the whole
        // list rather than once per row — it asks the graph a question per candidate.
        let offered = self.operand_offers(id, &choices);
        let mut new_operands = operands.clone();
        let mut removed: Option<usize> = None;
        // **Two edits at the end of a row, and they are different edits.** Emptying a slot in
        // place is §7.2's drag-out and §6.5's *Remove contents*: the row stays, the block keeps
        // its operation and its radius, and it is left *incomplete* — pruned from the viewport
        // until it is told what to read. Removing the row is the list's own older edit, and it
        // leaves the block **complete** with the operands that remain, which is why it needs a
        // list with a row to spare and emptying needs nothing. `RowEdits` holds the pair; this is
        // only the half that depends on the list.
        let over_two = is_list && operands.len() > 2;
        // The list's own header: what it holds, and the two ways to fill it. A pair
        // (a fillet) has neither, since a pair is all it can be.
        if is_list {
            let collecting = self.collecting == Some(id);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("Bodies ({})", operands.len())).weak(),
                );
                if ui
                    .add(mode_button(collecting, "Pick in viewport"))
                    .on_hover_text(
                        "Click bodies in the viewport to add them to this list, one \
                         after another. Click again, or close the block, to stop.",
                    )
                    .clicked()
                {
                    self.collecting = if collecting { None } else { Some(id) };
                    if !collecting {
                        self.say(format!(
                            "Picking bodies for {} — click them in the viewport",
                            self.body_label(id)
                        ));
                    }
                }
            });
        }
        for i in 0..operands.len() {
            let label = match (op, i) {
                (BooleanOp::Difference, 0) => "From:",
                (BooleanOp::Difference, _) => "Subtract:",
                (BooleanOp::Union, 0) => "Join:",
                (BooleanOp::Union, _) => "With:",
                (BooleanOp::Intersection, 0) => "Keep:",
                (BooleanOp::Intersection, _) => "Within:",
            };
            // **The one operand row (D9).** One widget for a blend's pair and a boolean's list,
            // so the empty state is drawn once and in one vocabulary — and a morph's two and a
            // pattern's one are the same row. Two is the least a boolean means anything with, so
            // the last two rows can be emptied but not removed.
            if self.operand_row(
                ui,
                label,
                (id, "operand", i),
                &mut new_operands[i],
                &offered,
                over_two,
            ) {
                removed = Some(i);
            }
        }
        if is_list {
            // Adding by name, for the bodies the viewport cannot show — a named block
            // already consumed elsewhere — and for anyone who would rather read a list
            // than click a model. Never a button that silently picks something.
            let eligible: Vec<(NodeId, String)> = choices
                .iter()
                .filter(|(n, _)| !new_operands.iter().any(|slot| slot.source() == Some(*n)))
                .cloned()
                .collect();
            let verb = match op {
                BooleanOp::Difference => "Subtract another body",
                BooleanOp::Union => "Join another body",
                BooleanOp::Intersection => "Keep only what is also within another body",
            };
            ui.horizontal(|ui| {
                if eligible.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "Every body in the part is in the list. Create another, \
                             or pick one in the viewport once there is one.",
                        )
                        .weak()
                        .italics(),
                    );
                } else {
                    // **`\u{203a}` rather than a down-pointing triangle.** egui's
                    // proportional face is Ubuntu-Light, which has no triangle at any
                    // codepoint -- `\u{25be}` was drawn as the empty box that means "no
                    // such glyph", seen the first time the block list was run under a
                    // display. Hack has one and is the *monospace* face, which a button
                    // label never reaches. `grip_mark` is the same discovery.
                    ui.menu_button("Add \u{203a}", |ui| {
                        for (n, name) in &eligible {
                            if ui.button(name).on_hover_text(verb).clicked() {
                                new_operands.push(Operand::filled(*n));
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_hover_text(verb);
                }
            });
        }
        if let Some(i) = removed
            && new_operands.len() > 2
        {
            new_operands.remove(i);
        }
        // **Said where the body is looked for, not where the name is typed.** This
        // list holds the bodies and the *named* blocks, because offering every inner
        // node would fill it with the insides of things nobody meant to share. So a
        // block already consumed by something else is missing from "Add" until it is
        // named, and this is where somebody finds that out.
        // **The state a deletion can leave behind**, said where it can be fixed.
        // Losing an operand no longer deletes the block: a pair keeps its survivor in
        // both slots and a longer list simply drops the row, so a definition stays
        // repairable instead of gone. A pair holding one body twice is the case worth
        // naming, because a difference of a body with itself is empty and a body that
        // has vanished for a reason nobody can see is what this panel exists to prevent.
        // **About a block chosen twice, not about a slot with nothing in it.** `same_block_twice`
        // is why that is not the plain comparison it reads as: an `Operand` holding nothing equals
        // another holding nothing, so this used to say "this subtracts a body from itself" about a
        // block reading neither. A hole has its own error, on its own row, where it is filled.
        if new_operands.len() == 2 && same_block_twice(new_operands[0], new_operands[1]) {
            let (text, colour) = match op {
                BooleanOp::Difference => (
                    "both rows are the same block, so this subtracts a body from \
                     itself and leaves nothing — pick something for one of them",
                    ui.visuals().error_fg_color,
                ),
                _ => (
                    "both rows are the same block, so this does nothing — pick \
                     something for one of them",
                    ui.visuals().warn_fg_color,
                ),
            };
            ui.label(egui::RichText::new(text).color(colour)).on_hover_text(
                "A deleted operand leaves the block behind with what it has left, so \
                 the operation, its radius and its name survive and only the missing \
                 side has to be chosen again.",
            );
        }
        self.reuse_hint(ui, id);

        let mut new_op = op;
        ui.horizontal(|ui| {
            for option in
                [BooleanOp::Difference, BooleanOp::Union, BooleanOp::Intersection]
            {
                // `mode_button`, not `selectable_value`: a selectable label draws no
                // frame at all, so Union and Intersect read as text until clicked.
                if ui.add(mode_button(new_op == option, boolean_label(option))).clicked() {
                    new_op = option;
                }
            }
        });

        // Whether a fillet on these operands will be the radius asked for, said where
        // the radius is typed rather than discovered afterwards.
        //
        // D8 used to *refuse* a blend whose operand only bounds distance from below, and
        // the only way to learn that was to pick G1 and have the edit silently decline --
        // the button appeared to ignore the click. Since the reversal on 2026-09-02 the
        // blend is allowed and this text is what replaced the refusal, so it is the whole
        // of the user's warning rather than a courtesy ahead of one. The operand at fault
        // is usually not the one being looked at: a sphere intersected with a lattice is a
        // body like any other in the list, and nothing about the row it occupies says its
        // fillet will run large.
        //
        // Read from the graph rather than by compiling, because this runs per frame
        // for an expanded block; `carries_true_distance` is the walk that exists for
        // it, and a test holds it to the same answer the compiler gives.
        // **An operand with no volume forces a blend**, because a hard boolean has
        // nothing to do with one: `min` and `max` depend only on sign, and a point's
        // field never goes negative. `smooth_min` mixes by magnitude instead, so it
        // reaches the point and pulls material out to it. Asked for directly: "when u
        // pick a sdf and a point it auto changes to g1 and u can blend it".
        //
        // G1 is the only continuity there is, so "make it a blend" and "set it to G1"
        // are the same instruction.
        //
        // **Read through `source`, so an empty slot is nothing rather than something.** A hole
        // neither encloses material nor fails to: it has no field at all, so it cannot force a
        // blend, cannot be the point a gap is measured to, and cannot be the operand whose
        // normalization makes a fillet approximate. The row it sits on says what is wrong with it.
        let hollow = new_operands.iter().filter_map(|slot| slot.source()).find(|id| {
            self.part_graph()
                .node(*id)
                .map(|n| !n.encloses_material())
                .unwrap_or(false)
        });
        // The distance from that point to the other operand's surface, which is exactly
        // the radius the blend has to beat. Only meaningful for a pair, which is also
        // the only case a fillet applies to, and read only when there is a point, since
        // it compiles a field to answer.
        let gap = hollow.and_then(|hollow| {
            let solid = new_operands
                .iter()
                .filter_map(|slot| slot.source())
                .find(|n| *n != hollow)?;
            (new_operands.len() == 2).then(|| self.gap_between(hollow, solid)).flatten()
        });

        let approximate =
            self.approximate_join(new_operands.iter().filter_map(|slot| slot.source()));

        // A fillet rolls between two surfaces. With more operands than that the join
        // stays hard, and the reason is on the control rather than in a refusal.
        let two = new_operands.len() == 2;
        let mut new_continuity = continuity;
        // Switched before the row is drawn, so it shows the state it is about to be in
        // rather than the one the click left behind for a frame.
        // Made a blend if it is not one — and left at the shape it has, since forcing G1
        // every frame would undo a G2 pick the moment it was made.
        if hollow.is_some() && new_continuity.is_none() {
            new_continuity = Some(Continuity::G1);
        }
        // **Titled as the spec names the tool, and the picker as every other panel names
        // it** (`round-chamfer-spec.md` §7): the section is *Fillet / Chamfer* and the row is
        // *Shape*, the word a pattern's fold and a profile solid's rims already use for the same
        // three shapes (D9). G0 is the plain boolean -- no fillet -- and sits first.
        ui.label(egui::RichText::new("Fillet / Chamfer").strong());
        ui.horizontal(|ui| {
            row_label(ui, "Shape:");
            for (mode, label) in BLEND_MODES {
                // Offered and disabled rather than absent, which is this panel's rule:
                // an option that vanishes cannot say why it went. Two reasons it can be
                // unavailable, and they are different reasons.
                let allowed = match mode {
                    // A plain boolean does nothing with an operand that encloses
                    // nothing -- but only worth forcing when a fillet is possible at
                    // all, which is a pair.
                    None => hollow.is_none() || new_operands.len() != 2,
                    // A fillet rolls between two surfaces.
                    Some(_) => two,
                };
                ui.add_enabled_ui(allowed, |ui| {
                    let picked = ui.add(mode_button(new_continuity == mode, label));
                    if picked.clicked() {
                        new_continuity = mode;
                    }
                    picked
                        .on_hover_text(match (label, &approximate) {
                            ("G0", _) => "A plain boolean: the surfaces meet at a \
                                          crease. Exact for any body, a lattice included."
                                .to_string(),
                            (_, Some(why)) => why.clone(),
                            ("G2", _) => fillet_shape_hover(Continuity::G2).to_string(),
                            ("Chamfer", _) => fillet_shape_hover(Continuity::Chamfer).to_string(),
                            _ => "A rolling-ball fillet of the radius below, exact for \
                                  these two bodies."
                                .to_string(),
                        })
                        .on_disabled_hover_text(match mode {
                            None => "One of these encloses nothing, and a plain \
                                     boolean depends only on sign — it would \
                                     leave the body exactly as it is, or empty it. A \
                                     blend reaches the point instead.",
                            Some(_) => "A fillet joins two bodies. Take this down to \
                                        two operands to fillet it, or nest booleans.",
                        });
                });
            }
        });
        // A fillet radius the node already carries stays with it; one being added
        // starts from the tool's current setting.
        let mut new_fillet = fillet.unwrap_or(self.blend_radius);
        // A chamfer's second leg, kept through a switch of shape: the kernel reads it only on a
        // chamfer (`Fillet::second_leg`), so switching to a round and back loses nothing.
        let mut new_second = second;
        // **Half again the gap**, when a blend is being made against a point and had no
        // radius of its own. The tool's usual default is a fixed number and would land
        // in the regime where the blend touches the point and does nothing visible --
        // an offered control that moves nothing, which is worse than none.
        if fillet.is_none()
            && let Some(gap) = gap
            && gap > 0.0
        {
            new_fillet = ParamRef::Literal(gap * 1.5);
        }
        if let Some(shape) = new_continuity {
            // **One number, or a ramp** (`round-chamfer-spec.md` §9.1): the offset's two
            // buttons and the one picker every driveable socket shares, so a radius that
            // grows along a seam is wired the way a thickness that grows along a body is.
            // Each switch edits the graph itself and returns, leaving the rest of this panel
            // to the next frame, so the node rebuilt below cannot write a stale radius over it.
            let driver = new_fillet.driver();
            let mut switched = false;
            ui.horizontal(|ui| {
                row_label(ui, "Radius:");
                if ui.add(mode_button(driver.is_none(), "One number")).clicked() {
                    if let Some(ramp) = driver {
                        self.drop_driver(id, ramp);
                        switched = true;
                    }
                }
                if ui.add(mode_button(driver.is_some(), "Ramp")).clicked() && driver.is_none() {
                    self.drive_blend(id);
                    switched = true;
                }
            });
            if switched {
                return;
            }
            if let Some(choice) = self.pick_driver(ui, "blend_driver", driver) {
                match choice {
                    Some(ramp) => self.wire_driver(id, 0, ramp),
                    None => {
                        if let Some(ramp) = driver {
                            self.drop_driver(id, ramp);
                        }
                    }
                }
                return;
            }
            if driver.is_some() {
                // The ramp is edited in the ramp, as an offset's is; the number below is for a
                // radius that is one number, and the notes under it read that number.
                ui.label(
                    egui::RichText::new(
                        "The radius is what the ramp reads at each point of the seam.",
                    )
                    .small()
                    .weak(),
                );
            }
            let (edited, _changed) = if driver.is_none() {
                self.parametric_field(ui, fillet_label(shape), new_fillet, 0.25, POSITIVE_MM, "")
            } else {
                (new_fillet, false)
            };
            new_fillet = edited;
            if driver.is_none()
                && let Some(note) = fillet_bend_note(shape, self.resolve(new_fillet))
            {
                ui.label(egui::RichText::new(note).small().weak());
            }
            // A chamfer's two legs (`round-chamfer-spec.md` §9.2): the first is the number
            // above, along A from B; the second along B from A. The rims' editor shows the
            // same rows (D9). Not with a radius a ramp drives, which the kernel refuses.
            if driver.is_none() {
                new_second = self.second_leg_rows(
                    ui,
                    shape,
                    new_fillet,
                    new_second,
                    "On B:",
                    "How far along the second body the bevel reaches from the first. The number \
                     above is how far along the first it reaches from the second.",
                );
            }
            // Beside the number, because that is the only place saying so is honest.
            // A blend against a lower-bound operand is allowed now, and what makes
            // that defensible is the radius announcing where it is nominal -- without
            // it this is the silent wrong shape D8 was written to prevent.
            if let Some(why) = &approximate {
                ui.colored_label(ui.visuals().warn_fg_color, "\u{2248} approximate")
                    .on_hover_text(why);
            }
            // And what this radius does to a surface somebody said must not move —
            // an error when measured and blamed, a caution when nobody has looked.
            if let Some((blamed, text, why)) = self.critical_note(id) {
                let colour = if blamed {
                    ui.visuals().error_fg_color
                } else {
                    ui.visuals().warn_fg_color
                };
                ui.colored_label(colour, text).on_hover_text(why);
            }
            // Said wherever the radius ends up below the gap, not only when this
            // block set it: the number is draggable and there is nothing about a
            // radius of 5 that looks wrong beside a gap of 40.
            if driver.is_none()
                && let Some(gap) = gap
                && self.resolve(new_fillet) <= gap
            {
                ui.colored_label(
                    ui.visuals().error_fg_color,
                    format!("touches only — needs more than {gap:.2} mm"),
                )
                .on_hover_text(
                    "Below the distance from the point to the other surface, the blend \
                     meets the point at a single tangent touch: the field reads nought \
                     there and no material moves. Above it, the blend swallows the \
                     point. Measured in `boolean_with_nothing.rs`.",
                );
            }
        }

        let rebuilt = match new_continuity {
            Some(continuity) if new_operands.len() == 2 => Node::Blend {
                op: new_op,
                a: new_operands[0],
                b: new_operands[1],
                fillet: fieldform_scene::Fillet { radius: new_fillet, continuity, second: new_second },
            },
            _ => Node::Boolean { op: new_op, inputs: new_operands },
        };
        if rebuilt != node {
            self.replace_node(id, rebuilt);
        }
    }
}
