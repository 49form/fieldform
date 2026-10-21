// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The block list: its rows and their order, and the markers, menus and reference fields on them.

use crate::console_keys::RowCalled;
use crate::*;


/// Whether a reference field's suggestion list is drawn this frame (item 109).
///
/// While the field has focus, or while the pointer is over a list **that was open last frame** —
/// the second half spans the press and the release of a click on a row, which take the focus
/// off the text box in between. A list that was closed is never reopened by the pointer alone,
/// however exactly the pointer sits where it used to be.
pub(crate) fn show_reference_list(has_focus: bool, was_open: bool, pointer_in_last_rect: bool) -> bool {
    has_focus || (was_open && pointer_in_last_rect)
}

/// Whether the list counts as open on the next frame: only if it was drawn, and not if a row was
/// chosen from it. A choice closes the list even with the pointer still on the row it clicked,
/// or the reopening of item 109 returns by a shorter route.
pub(crate) fn reference_list_open_next(drawn: bool, chosen: bool) -> bool {
    drawn && !chosen
}
impl App {
    /// The row of a closed loop of curves: one line collapsed, its curves in walk order when
    /// opened.
    ///
    /// The header says what an extrude would find — how many curves, and whether they lie on
    /// one plane — because those are the two things that decide whether the loop can become a
    /// solid, and a row that hid them would send the person to the extrude's refusal to find
    /// out. Everything inside is the ordinary curve row: rename, select and delete work as
    /// they do anywhere else.
    ///
    /// The open/closed state is egui's own collapsing state keyed by the row's first curve,
    /// the arrangement an import's row already uses, so there is no second store of what is
    /// open (`profile-group-spec.md` §9.5).
    ///
    /// `nesting` is the one the list was laid out from, which carries every curve's reader count:
    /// the row's V and its members' rows read theirs from it rather than walking the parts again
    /// for each row.
    pub(crate) fn profile_block(
        &mut self,
        ui: &mut egui::Ui,
        key: CurveId,
        nesting: &fieldform_scene::Nesting,
    ) {
        let Some(members) = self.profiles.members(key).map(|m| m.to_vec()) else { return };
        let flat = fieldform_field::loop_is_flat(&self.doc, &members);
        // Whether the loop is held together by joints (§4): every end on a point, some, or
        // none. Said in the header because it decides what a drag on a corner does.
        let (held, total) = self.profile_hold(&members);
        let hold = if held == 0 {
            ""
        } else if held == total {
            ", held"
        } else {
            ", partly held"
        };
        let joints = self.joint_points_of(&members);
        let header = format!(
            "{}  \u{00b7}  {} curves, {}{hold}",
            self.profile_label(key),
            members.len(),
            if flat { "flat" } else { "3D" }
        );
        let all_hidden = members.iter().all(|id| self.hidden_curves.contains(id));
        let selected = self.selected_block == Some(outputs::BlockRef::Profile(key));

        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("profile", key.raw())),
            false,
        );
        state
            .show_header(ui, |ui| {
                // **A grip, like a curve's, and keyed the same way.** A profile row *is* a
                // `CurveId` — the loop's first curve, which is the key the row is drawn under — and
                // that is exactly what an extrude's profile field holds, so dragging this row into
                // that field is the gesture rather than an approximation of it.
                ui.dnd_drag_source(
                    ui.make_persistent_id(("profile_grip", key.raw())),
                    key,
                    |ui| grip_mark(ui),
                )
                .response
                .on_hover_text("Drag onto a block's profile field to extrude or revolve this loop");
                // **The real state, and clickable — it used to be hardcoded bright.** The mark said
                // "variable" about every profile regardless, and offered no way to make one, so an
                // extruded profile looked declared and could not be taken out of the extrude. That
                // is the same gap the block row's V had, fixed the same way: the stored ownership.
                // It was the row's name until the ownership was stored, and naming a row for any
                // other reason lifted it out.
                //
                // **Of the curve that puts the row where it is, which is not always the key.** An
                // extrude names whichever curve was picked, and the row goes inside whatever holds
                // any of them, so a V on the key alone wrote a curve nothing read: it turned
                // bright, said the profile had a row of its own, and the row stayed in the extrude.
                let curve = profile_v_curve_in(&self.doc, nesting, &members).unwrap_or(key);
                let is_variable = self.doc.ownership(BlockKey::Curve(curve)) == Ownership::Variable;
                let v = Ok(Declarable::Profile { key, curve });
                self.variable_marker(ui, is_variable, v);
                // One eye for the lot, in the column the curves' own eyes sit in below it.
                let icon = egui::RichText::new("\u{1f441}").color(if all_hidden {
                    ui.visuals().weak_text_color()
                } else {
                    ui.visuals().strong_text_color()
                });
                let size = egui::vec2(18.0, ui.spacing().interact_size.y);
                if ui
                    .add_sized(size, egui::Button::new(icon).frame(false))
                    .on_hover_text(if all_hidden { "Show the profile" } else { "Hide the profile" })
                    .clicked()
                {
                    for id in &members {
                        if all_hidden {
                            self.hidden_curves.remove(id);
                        } else {
                            self.hidden_curves.insert(*id);
                        }
                    }
                }
                let Some(row) = self.list_row(ui, RowKey::Profile(key), &header, selected) else {
                    return;
                };
                if row.clicked() || row.secondary_clicked() {
                    self.select_block(outputs::BlockRef::Profile(key));
                }
                row.context_menu(|ui| self.block_menu(ui, outputs::BlockRef::Profile(key)));
            })
            .body(|ui| {
                ui.label(
                    egui::RichText::new(if flat {
                        "A closed loop on one plane — can be extruded or revolved"
                    } else {
                        "A closed loop, not flat — can be revolved or swept, not extruded"
                    })
                    .small()
                    .weak(),
                );
                for &id in &members {
                    // **Asked of `folded_curve_row`, not decided again here, and deciding it again
                    // made a curve vanish.** This read `declared_curves` directly, which disagrees
                    // with that function for exactly one member: the loop's *key*. The key's row **is**
                    // the profile row (`Some(key) if key == id => Block::Profile`), so declaring it
                    // never gives it a row of its own — and skipping it here left it drawn nowhere at
                    // all. Reported: "I nested a profile in an extrude then went to the V on the first
                    // curve and now the curve is gone completely."
                    //
                    // One rule, asked in both places: a member is drawn inside the row unless it took
                    // a top-level row of its own.
                    if matches!(self.folded_curve_row(id, nesting), Some(Block::Curve(_))) {
                        continue;
                    }
                    self.curve_block(ui, id, nesting.readers_of_curve(id));
                    // A member whose ends sit on nothing while the rest are held — a curve
                    // pasted back in, most likely — is marked, since dragging it moves it
                    // alone and takes it out of the loop (§4.4).
                    let loose = held > 0
                        && self.doc.curve(id).is_some_and(|c| {
                            let n = c.control_points().len();
                            n >= 2 && c.placed_on(0).is_none() && c.placed_on(n - 1).is_none()
                        });
                    if loose {
                        ui.label(
                            egui::RichText::new("not held — join to hold it")
                                .small()
                                .color(ui.visuals().warn_fg_color),
                        );
                    }
                }
                // The joints, inside the row rather than in the list: a thirty-curve loop
                // adds thirty points and zero visible rows (§4.3).
                if !joints.is_empty() {
                    ui.separator();
                    ui.label(egui::RichText::new("Joints").weak());
                    for point in joints {
                        self.node_block(ui, point, &[], nesting.references_of(point), false, nesting, None);
                        self.joint_controls(ui, point);
                    }
                }
                // Then the relations naming a member (§3.2), each marked when it was found not
                // to hold on open, with the reason on hover and `realign` as the way back.
                let relations: Vec<(fieldform_scene::RelationId, String, Option<String>)> = self
                    .doc
                    .relations
                    .iter()
                    .filter(|(_, r)| members.iter().any(|m| r.relation.names(*m)))
                    .map(|(id, r)| (*id, r.relation.describe(&self.doc), r.broken.clone()))
                    .collect();
                if !relations.is_empty() {
                    ui.separator();
                    ui.label(egui::RichText::new("Relations").weak());
                    // Break and Realign beside each: the console's `break` and `realign`,
                    // where the relation is read.
                    let mut to_break = None;
                    let mut to_realign = None;
                    for (id, what, broken) in relations {
                        ui.horizontal(|ui| {
                            match &broken {
                                None => {
                                    ui.label(egui::RichText::new(what).small());
                                }
                                Some(why) => {
                                    ui.label(
                                        egui::RichText::new(format!("{what} — broken"))
                                            .small()
                                            .color(ui.visuals().warn_fg_color),
                                    )
                                    .on_hover_text(format!("{why}. Realign re-derives it."));
                                }
                            }
                            if broken.is_some()
                                && ui
                                    .small_button("Realign")
                                    .on_hover_text(
                                        "Re-derives it from the master's current state, as \
                                         one undoable edit",
                                    )
                                    .clicked()
                            {
                                to_realign = Some(id);
                            }
                            if ui
                                .small_button("Break")
                                .on_hover_text("Deletes the relation. Nothing moves.")
                                .clicked()
                            {
                                to_break = Some(id);
                            }
                        });
                    }
                    if let Some(id) = to_realign {
                        self.realign_relations(&[id]);
                    }
                    if let Some(id) = to_break {
                        self.break_relations(&[id]);
                    }
                }
            });
    }

    /// One row of the block list: a curve, or a node.
    ///
    /// They are listed together because they are the same kind of thing to the person reading
    /// the list — the parts this part is made of, in the order they arrived. Keeping them in
    /// two lists is what put every curve above every body regardless of when either was made.
    ///
    /// `nesting` is the list's, whose curve reader counts fold the loops.
    fn ordered_blocks(&self, nesting: &fieldform_scene::Nesting) -> Vec<Block> {
        // Variables first, by name. They are inputs to everything and belong to nothing, so
        // there is no creation order that would put them anywhere more meaningful — and a list
        // that reshuffles between frames is worse than one whose order is merely a convention.
        // The document holds them in a hash map, whose order is nobody's idea of stable.
        let mut variables: Vec<ParamId> = self.doc.parameters.keys().copied().collect();
        variables.sort_by_key(|id| {
            self.doc.parameters.get(id).map(|p| p.name.clone()).unwrap_or_default()
        });

        let nodes = self.ordered_nodes();
        let mut curves: Vec<CurveId> = self.doc.curves.keys().copied().collect();
        // By time, falling back to the id so that curves a file never stamped still come out in
        // a stable order rather than a different one each run.
        curves.sort_by_key(|id| (self.doc.curves_created.get(id).copied().unwrap_or(0), id.raw()));

        // Merged rather than sorted, because the node list is already in an order that has to
        // survive: it satisfies dependency as well as time, and re-sorting the two together
        // would lose that. Walking two ordered lists keeps it.
        let at = |b: &Block| match b {
            Block::Curve(id) => self.doc.curves_created.get(id).copied().unwrap_or(0),
            Block::Node(id) => self.doc.created.get(id).copied().unwrap_or(0),
            // Never reached: the grouping runs after this merge, on its result, and the
            // variables and surfaces were placed before it — none of the three is in the two
            // lists being merged.
            Block::Import(_)
            | Block::Profile(_)
            | Block::Parameter(_)
            | Block::Surface(_)
            | Block::Tessellation(_)
            | Block::Export(_) => 0,
        };
        let mut out: Vec<Block> = variables.into_iter().map(Block::Parameter).collect();
        // **With the inputs, above the things that read them.** A surface is not built from
        // anything in the part -- it arrived whole from a file -- so there is no creation order
        // that would put it anywhere more meaningful than the top, which is the argument the
        // variables above already make. Ordered by id so the list does not reshuffle.
        out.extend(self.doc.quad_meshes_ordered().into_iter().map(Block::Surface));
        out.reserve(nodes.len() + curves.len());
        let (mut c, mut n) = (curves.into_iter().peekable(), nodes.into_iter().peekable());
        loop {
            match (c.peek().copied(), n.peek().copied()) {
                (None, None) => break,
                (Some(id), None) => {
                    c.next();
                    out.push(Block::Curve(id));
                }
                (None, Some(id)) => {
                    n.next();
                    out.push(Block::Node(id));
                }
                (Some(curve), Some(node)) => {
                    // A tie goes to the curve: the usual reason they share a moment is that one
                    // was made to be swept by the other, and the thing swept comes first.
                    if at(&Block::Curve(curve)) <= at(&Block::Node(node)) {
                        c.next();
                        out.push(Block::Curve(curve));
                    } else {
                        n.next();
                        out.push(Block::Node(node));
                    }
                }
            }
        }
        // A closed loop of open curves collapses into one row, at the place its first curve
        // would have been — before the imports fold, so that a loop whose first curve came from
        // a file folds into that file's row along with it. `folded_curve_row` is the rule.
        let joints = crate::blocks::list::held_joint_points(&self.doc, self.active_part, &self.profiles);
        out = out
            .into_iter()
            .filter_map(|block| match block {
                Block::Curve(id) => self.folded_curve_row(id, nesting),
                // A loop's joint points are listed inside its row (§4.3): thirty of them for
                // a thirty-curve loop, and zero rows out here.
                Block::Node(id) if joints.contains(&id) => None,
                other => Some(other),
            })
            .collect();
        // Bodies from one file collapse into one row, at the place the first of them would
        // have been — so an import sits where it arrived rather than jumping to the top or the
        // bottom.
        //
        // **However few, including one.** This used to group only two or more, on the reasoning
        // that a group of one is a disclosure triangle over nothing. That is true of the
        // *nesting* and false of the block, which also carries the file it came from, the
        // tessellation it was read at, and the button that reads it again — so a single-solid
        // STEP put all of that out of reach, with no way even to see which file the body came
        // from. Reported as the CAD block not showing up on import: it shows for an assembly,
        // and never did for one solid.
        let mut seen: std::collections::HashSet<fieldform_scene::id::SourceId> =
            std::collections::HashSet::new();
        // Bodies, surfaces and curves alike: whichever of an import's things comes first in
        // the list is where its block goes, and the rest are inside it.
        let source_of = |block: &Block| match block {
            Block::Node(id) => self.body_source(*id),
            Block::Surface(id) => self.doc.sources.values().find(|s| s.surfaces.contains(id)).map(|s| s.id),
            Block::Curve(id) => self.doc.sources.values().find(|s| s.curves.contains(id)).map(|s| s.id),
            // A profile row goes where its first curve would have: inside the import that
            // curve came from, if one did.
            Block::Profile(key) => self.doc.sources.values().find(|s| s.curves.contains(key)).map(|s| s.id),
            _ => None,
        };
        out.retain(|block| {
            let Some(source) = source_of(block) else { return true };
            seen.insert(source)
        });
        for block in out.iter_mut() {
            if let Some(source) = source_of(block) {
                *block = Block::Import(source);
            }
        }
        // **Outputs last, below everything they read.** A tessellation consumes a body and an
        // export consumes a tessellation, so the list reads top to bottom the way the data
        // flows: inputs, the blocks built from them, and what leaves the program.
        let (tessellations, exports) = self.output_blocks();
        out.extend(tessellations.into_iter().map(Block::Tessellation));
        out.extend(exports.into_iter().map(Block::Export));
        out
    }

    /// The same rows, with each one nested inside the block it belongs to.
    ///
    /// The arrangement is [`nest_rows`], which is a free function so the shape of the tree can be
    /// asserted — an `App` needs a window and a device before it will exist, the same reason
    /// `operand_choices_in` and `Document::listed_blocks` are where they are.
    ///
    /// `nesting` is the caller's, `Document::nesting_in` of the part on screen, because the rows
    /// are drawn from it too ([`Self::block_row`]): one walk a frame for the lot.
    pub(crate) fn nested_blocks(&self, nesting: &fieldform_scene::Nesting) -> Vec<BlockRow> {
        nest_rows(
            self.ordered_blocks(nesting),
            nesting,
            &|seed| fieldform_field::loop_from(&self.doc, seed, None).unwrap_or_default(),
        )
    }

    /// The blocks, in the order they were made, with nothing above what it is made from.
    ///
    /// Moved to [`fieldform_scene::Document::listed_blocks`], which is where the stamps it reads live
    /// and where it can be tested: this was a method on `App`, so the block list's order was the
    /// one order in the codebase no test could reach — an `App` needs a window and a device.
    /// `block-order-spec.md` §2.1 wanted it measured before it is changed, and that is why.
    ///
    /// As the list's readings keep it ([`Self::list_readings`]): the rows, the summary above them
    /// and every import's row ask for it each frame.
    pub(crate) fn ordered_nodes(&self) -> Vec<NodeId> {
        self.list_readings().listed.clone()
    }

    /// Moves a block in the list so it sits where `onto` sits, and renumbers the rows.
    ///
    /// **Renumbered rather than given a fractional rank.** A part holds tens of bodies, not
    /// thousands, so rewriting every rank on a drop costs nothing and keeps the numbers in the file
    /// readable — where inserting at the midpoint between two neighbours never renumbers but drifts
    /// toward needing it anyway. The whole map for this part's bodies is written, so it is total
    /// over what is on screen: a body created afterwards has no entry, which is what puts it at the
    /// bottom where a new body belongs.
    ///
    /// **Only this part's bodies are touched.** `arrangement` is keyed by node across the whole
    /// document, so replacing the map wholesale would forget how every other part was arranged.
    ///
    /// Not on the undo stack, which is a gap rather than a decision. `rename_curve` and
    /// `set_body_name` shared it until both went onto the stack through `fieldform_ops::SetName`,
    /// so a reorder is what is left of it, and backlog item 60 lists it as left to do. A reorder
    /// changes no geometry, which is why it was always the cheapest of the three to leave.
    fn arrange_block(&mut self, dragged: NodeId, onto: NodeId) {
        if dragged == onto {
            return;
        }
        // The rows as the person is looking at them, which is the only order a drop means
        // anything against.
        let mut rows = self.doc.listed_bodies(self.part_graph());
        let Some(from) = rows.iter().position(|id| *id == dragged) else { return };
        if !rows.contains(&onto) {
            return;
        }
        rows.remove(from);
        // **The target's index is read *after* the removal, and that is not a detail.** Dropping on
        // a row means "put it where that row is", which is before it — and taken before the
        // removal, dragging *downward* lands one place too far, because the target has already
        // shifted up by one. An off-by-one that only shows in one direction is exactly the kind
        // that ships.
        let to = rows.iter().position(|id| *id == onto).unwrap_or(rows.len());
        rows.insert(to, dragged);
        self.write_arrangement(&rows);
    }

    /// Moves a block to the end of the list.
    ///
    /// **Because dropping on a row always means "before it", and nothing is below the last one.**
    /// Without this the bottom of the list is the one position a body cannot be dragged to, which
    /// is not a position a person would expect to be refused. The empty space under the rows is
    /// where it lands — the same rectangle that already deselects on a click, now answering a drop
    /// as well.
    fn arrange_block_last(&mut self, dragged: NodeId) {
        let mut rows = self.doc.listed_bodies(self.part_graph());
        let Some(from) = rows.iter().position(|id| *id == dragged) else { return };
        if from + 1 == rows.len() {
            return;
        }
        rows.remove(from);
        rows.push(dragged);
        self.write_arrangement(&rows);
    }

    /// Writes a rank per row, for this part's bodies only.
    fn write_arrangement(&mut self, rows: &[NodeId]) {
        for (i, id) in rows.iter().enumerate() {
            self.doc.arrangement.insert(*id, i as u32);
        }
        self.mark_changed();
    }

    /// Puts the list back in the order the blocks were made.
    ///
    /// Clearing the map rather than writing creation order into it, because an absent entry
    /// already *means* "wherever it was made" — writing the same answer down would leave a
    /// document that reads as arranged when nobody has arranged it, and would pin blocks against a
    /// later creation instead of letting it go to the bottom.
    pub(crate) fn clear_arrangement(&mut self) {
        let mine: Vec<NodeId> = self.part_graph().nodes().map(|(id, _)| id).collect();
        let before = self.doc.arrangement.len();
        for id in mine {
            self.doc.arrangement.remove(&id);
        }
        if self.doc.arrangement.len() != before {
            self.mark_changed();
        }
    }

    /// Whether this part's list has been arranged, so the menu can offer to put it back.
    pub(crate) fn is_arranged(&self) -> bool {
        self.part_graph().nodes().any(|(id, _)| self.doc.arranged_at(id).is_some())
    }

    /// The Number block: a name, a number typed or an expression computing one, and — for an
    /// expression — what it currently computes to, or why it computes nothing.
    ///
    /// The value under the expression is the reason this is a block on the canvas rather than
    /// a row in a dialog: `boss_r`, the text `radius * 5 - 2`, and `23` beside it. The
    /// expression is committed on Enter or on leaving the field (see [`ExpressionDraft`]); a
    /// parse error or a cycle is shown here, under the row, and the document keeps what it had.
    ///
    /// The same controls the Parameters section used to hold at the foot of the panel. That
    /// section is gone rather than kept beside this, because two lists of one thing drift and
    /// the question "where do I edit a variable" should have one answer.
    fn parameter_block(&mut self, ui: &mut egui::Ui, id: ParamId) {
        let Some(param) = self.doc.parameters.get(&id) else { return };
        // The draft while one is being typed, else the name the document holds.
        let mut name = self
            .name_drafts
            .get(&id)
            .map(|d| d.text.clone())
            .unwrap_or_else(|| param.name.clone());
        let literal = param.value.as_literal();
        let source = param.value.expression().map(|e| e.source().to_string());
        let used = self.parameter_is_used(id);
        let worth = self.doc.evaluate_parameter(id);
        let mut rename: Option<String> = None;
        let mut abandon_name = false;
        let mut typed: Option<f64> = None;
        let mut start_expression = false;
        let mut commit: Option<String> = None;
        let mut discard = false;
        let mut remove = false;
        let mut put_back = false;

        ui.horizontal(|ui| {
            // Bright, because this *is* a variable -- the dim one on a field is the offer to
            // make one. Same mark either way, so the two read as one idea.
            //
            // **And a click puts it back, as a block's bright V does** (`block-ownership-spec.md`
            // §11): the one field using it holds the number again, and this row goes. Whether it
            // can is asked only under the pointer, since asking walks every part and the list is
            // drawn every frame; a click it refuses says why on the status line, as the hover did.
            let mark = egui::RichText::new("V").color(VARIABLE_ON).strong();
            let hit = ui.add(egui::Label::new(mark).sense(egui::Sense::click()));
            let hit = if hit.hovered() {
                let what = "A variable. Every field driven by it follows when this changes, and \
                            the fields themselves show a bright V.";
                match self.number_put_back(id) {
                    Ok(_) => hit.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(
                        format!(
                            "{what} Click to put it back: the field that uses it holds the \
                             number again, and this row goes."
                        ),
                    ),
                    Err(why) => hit.on_hover_text(format!("{what} {why}")),
                }
            } else {
                hit
            };
            if hit.clicked() {
                put_back = true;
            }
            // Salted with the parameter, not placed by position: the rows are sorted by
            // name, so a row moves when its name is applied, and a field whose identity was
            // its position kept the keyboard where the row had been.
            let field = ui.add(
                egui::TextEdit::singleline(&mut name)
                    .desired_width(110.0)
                    .id_salt(("parameter_name", id)),
            );
            // Once, and the whole of it selected, so typing replaces the invented name rather
            // than appending to it.
            if self.naming_parameter == Some(id) {
                self.naming_parameter = None;
                field.request_focus();
                if let Some(mut state) =
                    egui::TextEdit::load_state(ui.ctx(), field.id)
                {
                    let all = egui::text::CCursor::new(0)
                        ..=egui::text::CCursor::new(name.chars().count());
                    state
                        .cursor
                        .set_char_range(Some(egui::text_selection::CCursorRange::two(
                            *all.start(),
                            *all.end(),
                        )));
                    state.store(ui.ctx(), field.id);
                }
            }
            if field.changed() {
                let draft = self
                    .name_drafts
                    .entry(id)
                    .or_insert_with(|| NameDraft { text: String::new(), error: None });
                draft.text = name.clone();
                draft.error = None;
            }
            // Escape puts the document's name back; Enter, or leaving the field, applies
            // what was typed. `lost_focus` is true on Enter as well, since a single-line
            // field gives the keyboard up on it -- and on Escape, which egui answers by
            // clearing the focus before the frame draws, so "lost focus and Escape" is the
            // test and "has focus and Escape" is never true.
            let escaped = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape));
            if escaped {
                abandon_name = true;
            } else if field.lost_focus() && self.name_drafts.contains_key(&id) {
                rename = Some(name.clone());
            }
            let editing = literal.is_none() || self.expression_drafts.contains_key(&id);
            if !editing {
                // A typed number, and nobody is typing an expression over it.
                let mut value = literal.unwrap_or(0.0);
                    // A variable has no unit and no range anybody declared, so it is the field
                    // that most needs a rate taken from what it reads.
                    let rate = scrub_rate(ui, value, 0.25);
                    if ui.add(egui::DragValue::new(&mut value).speed(rate)).changed() {
                        typed = Some(value);
                    }
                    // `=`, as a spreadsheet starts a formula: compute this instead of typing it.
                    if ui
                        .add(egui::Button::new("=").small())
                        .on_hover_text(
                            "Compute it from other parameters — type an expression such \
                             as radius * 5 - 2. A name with a dash or a space goes in square \
                             brackets, [cap-1] * 2. Typing a plain number puts it back to one.",
                        )
                        .clicked()
                    {
                        start_expression = true;
                    }
            } else {
                // An expression, stored or being typed.
                    let draft = self.expression_drafts.entry(id).or_insert_with(|| ExpressionDraft {
                        text: source.clone().unwrap_or_default(),
                        error: None,
                        focus: false,
                    });
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut draft.text)
                            .desired_width(180.0)
                            .font(egui::TextStyle::Monospace)
                            .hint_text("radius * 5 - 2")
                            .id_salt(("parameter_expression", id)),
                    );
                    // Focused with the whole of it selected, so typing replaces the number
                    // the field opened with rather than landing in front of it: `=` then
                    // `radius * 2` used to read `radius * 20`.
                    if std::mem::take(&mut draft.focus) {
                        edit.request_focus();
                        if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), edit.id) {
                            state.cursor.set_char_range(Some(
                                egui::text_selection::CCursorRange::two(
                                    egui::text::CCursor::new(0),
                                    egui::text::CCursor::new(draft.text.chars().count()),
                                ),
                            ));
                            state.store(ui.ctx(), edit.id);
                        }
                    }
                    let escaped = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape));
                    if escaped {
                        discard = true;
                    } else if edit.lost_focus() {
                        commit = Some(draft.text.clone());
                    }
                    // The value, or the reason there is none. Beside the expression rather than
                    // in a tooltip, because the number is what the block is for.
                    match &worth {
                        Ok(v) => {
                            ui.label(egui::RichText::new(format!("= {}", readout(*v))).strong())
                                .on_hover_text("What the expression computes to, in millimetres where it is a length");
                        }
                        Err(_) if source.is_some() => {
                            ui.colored_label(ui.visuals().warn_fg_color, "= ?");
                        }
                        // A number being turned into an expression has a value still.
                        Err(_) => {}
                    }
            }
            // Only when nothing reads it. Removing one that is still driving geometry -- or
            // that another parameter computes from -- would leave those resolving to nothing,
            // which refuses to compile: the row would be offering a way to break the part.
            if ui
                .add_enabled(
                    !used,
                    egui::Button::new(
                        egui::RichText::new("\u{d7}")
                            .color(egui::Color32::from_rgb(220, 80, 80))
                            .strong(),
                    )
                    .small(),
                )
                .on_hover_text("Delete this variable")
                .on_disabled_hover_text("Still driving something — clear it there first")
                .clicked()
            {
                remove = true;
            }
        });

        // Under the row: what stopped the last attempt, or why the stored expression computes
        // nothing. Both in the same place, since either is the thing to fix.
        let stuck = self
            .name_drafts
            .get(&id)
            .and_then(|d| d.error.clone())
            .or_else(|| self.expression_drafts.get(&id).and_then(|d| d.error.clone()))
            .or_else(|| match (&worth, &source) {
                (Err(why), Some(_)) => Some(why.to_string()),
                _ => None,
            });
        if let Some(why) = stuck {
            ui.horizontal(|ui| {
                ui.add_space(MARK_W);
                ui.colored_label(ui.visuals().warn_fg_color, why);
            });
        }

        if remove {
            self.expression_drafts.remove(&id);
            self.name_drafts.remove(&id);
            // On the history, so undoing the delete that left it unread cannot bring back a block
            // reading a parameter that is gone: this undoes first.
            let done = remove_unread_parameter(&mut self.stack, &mut self.doc, &mut self.gesture, id);
            self.after_parameter_entry(done);
            return;
        }
        if abandon_name {
            self.name_drafts.remove(&id);
        }
        if let Some(name) = rename {
            match self.doc.rename_parameter(id, name.trim()) {
                Ok(()) => {
                    self.name_drafts.remove(&id);
                    self.mark_changed();
                }
                // Refused: the draft stays, with the reason under the row, so what was typed
                // can be fixed rather than typed again.
                Err(why) => {
                    self.warn(why.to_string());
                    if let Some(draft) = self.name_drafts.get_mut(&id) {
                        draft.error = Some(why.to_string());
                    }
                }
            }
        }
        if let Some(value) = typed {
            self.doc.set_parameter_literal(id, value);
            self.after_parameter_change();
        }
        if start_expression {
            self.expression_drafts.insert(
                id,
                ExpressionDraft {
                    text: readout(literal.unwrap_or(0.0)),
                    error: None,
                    focus: true,
                },
            );
        }
        if discard {
            self.expression_drafts.remove(&id);
        }
        if let Some(text) = commit {
            match self.commit_expression(id, &text) {
                Ok(()) => {
                    self.expression_drafts.remove(&id);
                }
                Err(why) => {
                    self.warn(format!("`{}`: {why}", self.parameter_label(id)));
                    if let Some(draft) = self.expression_drafts.get_mut(&id) {
                        draft.error = Some(why);
                    }
                }
            }
        }
        // Last: the click on the V is what took the keyboard from a name or a number being typed,
        // and that is committed above, so what goes back is what the row last said.
        if put_back {
            self.put_parameter_back(id);
        }
    }

    /// [`put_number_back`] for parameter `id`, with the blocks a refusal names called what the
    /// list calls them, in whichever part each is.
    fn number_put_back(&self, id: ParamId) -> Result<fieldform_ops::Batch, String> {
        put_number_back(&self.doc, id, |part, node| match self.doc.parts.get(&part) {
            Some(held) => self.body_label_in(&held.graph, node),
            None => self.body_label(node),
        })
    }

    /// *Put back inside* for a parameter (`block-ownership-spec.md` §11), the bright V's act on
    /// its row: the one field using it holds its number again and the parameter goes, as one undo
    /// step ([`put_number_back`]). Refused, saying why, wherever there is not one field for it.
    ///
    /// Not a gesture's next instant, so it ends any drag the graph or a material had open, as a
    /// graph edit carrying a parameter does: folded into one, the parameter's half would come off
    /// the history while the document kept it.
    fn put_parameter_back(&mut self, id: ParamId) {
        let name = self.parameter_label(id);
        let cmd = match self.number_put_back(id) {
            Ok(cmd) => cmd,
            Err(why) => {
                self.say(format!("\u{201c}{name}\u{201d}: {why}"));
                return;
            }
        };
        let value = self.doc.parameters.get(&id).and_then(|p| p.value.as_literal());
        self.gesture = None;
        self.material_gesture = None;
        // The draft pull is the analysis panel's own, so nothing the history sees reads it: it
        // keeps the number it showed, as it does when an undo takes a parameter it reads.
        let pull = typed_direction(&self.doc, &self.draft.pull);
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(cmd)) {
            self.warn(e.to_string());
            return;
        }
        keep_typed_direction(&self.doc, &mut self.draft.pull, pull);
        self.expression_drafts.remove(&id);
        self.name_drafts.remove(&id);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!(
            "\u{201c}{name}\u{201d} is back inside the field that used it, as {}",
            readout(value.unwrap_or(0.0))
        ));
    }

    /// Stores what was typed for a parameter: a plain number as a number, anything else as an
    /// expression. Nothing changes on an error, and the error is what the row shows.
    pub(crate) fn commit_expression(&mut self, id: ParamId, text: &str) -> Result<(), String> {
        let text = text.trim();
        // Nothing typed is not a mistake: a fresh Number left blank stays 0, and an expression
        // whose text was cleared stays what it was.
        if text.is_empty() {
            return Ok(());
        }
        if let Ok(v) = text.parse::<f64>() {
            if v.is_finite() {
                self.doc.set_parameter_literal(id, v);
                self.after_parameter_change();
                return Ok(());
            }
        }
        self.doc.set_parameter_expression(id, text).map_err(|e| e.to_string())?;
        self.after_parameter_change();
        Ok(())
    }

    /// What follows any change to a parameter's number or expression.
    ///
    /// Curves first: a driven coordinate is derived from the value, and everything that reads
    /// a curve -- the display, a pipe, a profile -- reads the derived point. Then the viewport,
    /// because a named value is *in* the document and every block reading it is out of date.
    fn after_parameter_change(&mut self) {
        self.mark_changed();
        if self.settle_curves_on_points() {
            self.rebuild_curve_display();
        }
        self.rebuild_render_resources();
    }

    pub(crate) fn parameter_label(&self, id: ParamId) -> String {
        self.doc.parameters.get(&id).map(|p| p.name.clone()).unwrap_or_else(|| id.to_string())
    }

    /// A parameter as the console says it: `boss_r = radius * 5 - 2 = 23`, `wall = 2.4`, or
    /// the expression and why it computes nothing.
    fn describe_parameter(&self, id: ParamId) -> String {
        let Some(param) = self.doc.parameters.get(&id) else {
            return format!("no parameter has id {id}");
        };
        match (&param.value, self.doc.evaluate_parameter(id)) {
            (fieldform_scene::ParamValue::Literal(v), _) => {
                format!("{} = {}", param.name, readout(*v))
            }
            (fieldform_scene::ParamValue::Expression(e), Ok(v)) => {
                format!("{} = {} = {}", param.name, e.source(), readout(v))
            }
            (fieldform_scene::ParamValue::Expression(e), Err(why)) => {
                format!("{} = {} — {why}", param.name, e.source())
            }
        }
    }

    /// A fresh Number block: a new parameter, opened on its row for an expression to be typed.
    ///
    /// **Separate from the variables it reads.** Naming a sphere's radius makes `radius`; this
    /// makes a second parameter that can say `radius + 10` and leaves `radius` at what it was.
    /// The `=` on an existing row edits that parameter in place, which is the other thing, and
    /// the one a person reaching for "add ten to it" does not mean.
    ///
    /// `number` alone names it `number` (then `number_2`) and waits for the expression;
    /// `number 'boss_r'` names it; `number expr="radius + 10"` fills it in one line.
    pub(crate) fn add_number(&mut self, args: &Args) {
        let name = match args.get("name").and_then(|v| v.as_str()).map(str::trim) {
            Some(wanted) if !Document::is_parameter_name(wanted) => {
                self.warn(fieldform_scene::RenameError::NotAName(wanted.to_string()).to_string());
                return;
            }
            Some(wanted) if self.doc.parameter_named(wanted).is_some() => {
                self.warn(fieldform_scene::RenameError::Taken(wanted.to_string()).to_string());
                return;
            }
            Some(wanted) => wanted.to_string(),
            None => unique_parameter_name(&self.doc, "number"),
        };
        let id = self.doc.add_parameter(name.clone(), 0.0);
        self.mark_changed();
        let expr = args.get("expr").and_then(|v| match v {
            Value::Str(s) => Some(s.clone()),
            Value::Int(i) => Some(i.to_string()),
            Value::Float(f) => Some(f.to_string()),
            _ => None,
        });
        match expr {
            Some(text) => {
                if let Err(why) = self.commit_expression(id, &text) {
                    self.warn(format!("`{name}`: {why}"));
                    self.doc.parameters.remove(&id);
                    return;
                }
                let line = self.describe_parameter(id);
                self.say(line);
            }
            // Opened empty and focused, so the next thing typed is the expression. The row
            // shows the hint until then.
            None => {
                self.expression_drafts.insert(
                    id,
                    ExpressionDraft { text: String::new(), error: None, focus: true },
                );
                self.say(format!(
                    "Number \u{201c}{name}\u{201d} — type an expression on its row, such as \
                     radius + 10"
                ));
            }
        }
    }

    /// `param` lists every parameter; `param name=wall value=2.4` makes or sets one;
    /// `param name=boss_r expr="radius * 5 - 2"` computes one; `param name=wall` reads it back.
    pub(crate) fn param_command(&mut self, args: &Args) {
        let name = args.get("name").and_then(|v| v.as_str()).map(|s| s.trim().to_string());
        let Some(name) = name else {
            let mut ids: Vec<ParamId> = self.doc.parameters.keys().copied().collect();
            ids.sort_by_key(|id| self.parameter_label(*id));
            if ids.is_empty() {
                self.say("No parameters. `param name=wall value=2.4` makes one.");
            }
            for id in ids {
                let line = self.describe_parameter(id);
                self.say(line);
            }
            return;
        };
        // An expression is a string, but `expr=5` arrives as a number: the console reads every
        // argument as the narrowest type that fits, and a number is an expression too.
        let expr = args.get("expr").and_then(|v| match v {
            Value::Str(s) => Some(s.clone()),
            Value::Int(i) => Some(i.to_string()),
            Value::Float(f) => Some(f.to_string()),
            _ => None,
        });
        let value = args.get("value").and_then(|v| v.as_f64());
        let (id, fresh) = match self.doc.parameter_named(&name) {
            Some(id) => (id, false),
            None if expr.is_none() && value.is_none() => {
                self.warn(format!("No parameter is called `{name}`"));
                return;
            }
            None => {
                if !Document::is_parameter_name(&name) {
                    self.warn(fieldform_scene::RenameError::NotAName(name).to_string());
                    return;
                }
                (self.doc.add_parameter(name.clone(), value.unwrap_or(0.0)), true)
            }
        };
        if let Some(text) = expr {
            if let Err(why) = self.commit_expression(id, &text) {
                self.warn(format!("`{name}`: {why}"));
                if fresh {
                    self.doc.parameters.remove(&id);
                }
                return;
            }
        } else if let Some(v) = value {
            self.doc.set_parameter_literal(id, v);
            self.after_parameter_change();
        }
        let line = self.describe_parameter(id);
        self.say(line);
    }

    /// One surface's row: what it is, how big its faces are, and whether it is drawn.
    ///
    /// Deliberately the same shape as `curve_block`. Both are geometry the document holds and
    /// other blocks read by id, both are drawn as reference rather than traced as a body, and
    /// both hide from the same column — so they should be learned once, not twice.
    pub(crate) fn surface_block(&mut self, ui: &mut egui::Ui, id: fieldform_scene::QuadMeshId) {
        let Some(mesh) = self.doc.quad_meshes.get(&id) else { return };
        let faces = mesh.face_count();
        let points = mesh.vertex_count();
        let open = mesh.is_open();
        let span = face_span_mm(mesh);
        let fold = fieldform_field::mesh_fold_mm(mesh);
        let name = self.doc.surface_name(id).to_string();

        let header = format!("{name}  \u{00b7}  surface");
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("surface", id.raw())),
            false,
        );
        state
            .show_header(ui, |ui| {
                let showing = !self.hidden_surfaces.contains(&id);
                let icon = egui::RichText::new("\u{1f441}").color(if showing {
                    ui.visuals().strong_text_color()
                } else {
                    ui.visuals().weak_text_color()
                });
                // Always marked, exactly as a curve is: every conformal lattice offers every
                // surface, because a surface exists to be built on. The mark is a fact here
                // rather than a state.
                let why = "A surface is always a row of its own, and every lattice that takes \
                           one is offered it.";
                self.variable_marker(ui, true, Err(why));
                let size = egui::vec2(18.0, ui.spacing().interact_size.y);
                if ui.add_sized(size, egui::Button::new(icon).frame(false)).clicked() {
                    if showing {
                        self.hidden_surfaces.insert(id);
                    } else {
                        self.hidden_surfaces.remove(&id);
                    }
                }
                // Highlighted while it is the control mesh, which is the one piece of state
                // that says what a click in the viewport will pick on.
                let editing = self.control == ControlMesh::Surface(id);
                if let Some(row) = self.list_row(ui, RowKey::Surface(id), &header, editing)
                    && row.clicked()
                {
                    self.edit_surface(id);
                }
            })
            .body(|ui| {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!(
                        "{faces} quads, {points} points, {}",
                        if open { "open" } else { "closed" },
                    ))
                    .weak(),
                );
                if let Some((small, large)) = span {
                    ui.label(
                        egui::RichText::new(format!("faces {small:.1} to {large:.1} mm across"))
                            .weak(),
                    );
                }
                // **How sharply it turns, because that is what bounds what can be built on it.**
                // A conformal lattice's shell may not reach past this radius on either side; a
                // surface with no fold at all has no such limit.
                if fold.is_finite() {
                    ui.label(
                        egui::RichText::new(format!(
                            "tightest fold {fold:.1} mm radius — a shell on this may be \
                             up to {:.1} mm thick",
                            fold * 2.0
                        ))
                        .weak(),
                    )
                    .on_hover_text(
                        "Past its own fold radius a surface has no single nearest point, so a \
                         lattice laid on it crosses itself there.",
                    );
                } else {
                    ui.label(egui::RichText::new("flat — no fold to reach past").weak());
                }
                ui.label(
                    egui::RichText::new(
                        "Reference geometry, not a body — it has no inside. A lattice \
                         set to Conformal tiling lays its cells over this.",
                    )
                    .small()
                    .weak(),
                );
            });
    }

    /// One curve, as a block: reference geometry, and where a pipe comes from.
    ///
    /// Curves are in this list although they are not graph nodes, because they are what the
    /// first block in any real part is made from, and a list that started at the pipe would
    /// leave the question of *which* curve answered in a combo box somewhere else.
    ///
    /// `readers` is how many blocks read it, in every part, which its V asks before offering to
    /// put it back. Where the row is drawn is not asked: its V reads the stored ownership, which
    /// is what put the row there.
    pub(crate) fn curve_block(&mut self, ui: &mut egui::Ui, id: CurveId, readers: usize) {
        let Some(curve) = self.doc.curves.get(&id) else { return };
        let name = curve.name.clone();
        let positions: Vec<Point3<f64>> = curve.control_points().to_vec();
        let points = positions.len();
        let degree = curve.degree();
        let length = curve.length(0.5);
        // An axis stood on a surface: derived from its body, so its block offers U, V and length in
        // place of the coordinates, and no change of degree.
        let surface_axis = self.doc.surface_axes.get(&id).copied();

        let header = format!("{name}  ·  curve");
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("curve", id.raw())),
            false,
        );
        state
            .show_header(ui, |ui| {
                // A curve is drawn as reference geometry, so it hides like anything else that
                // is drawn — and the eye sits in the same column as the bodies' below it.
                let showing = !self.hidden_curves.contains(&id);
                let icon = egui::RichText::new("\u{1f441}").color(if showing {
                    ui.visuals().strong_text_color()
                } else {
                    ui.visuals().weak_text_color()
                });
                // **Bright only when it has a row of its own.** A node has to be named before other blocks will
                // offer it, because a dropdown of every node in the part would list the insides
                // of blocks nobody meant to share. Curves have no such filter and never needed
                // one: every block that takes a curve offers all of them, because a curve is
                // reference geometry — it exists to be built on. So the mark is not a state
                // here, it is a fact, and it is shown for the same reason it is shown on a
                // named node: to say at a glance what can be plugged into something else.
                // **A grip, the same as a block row's, carrying a `CurveId`.** Asked for on the
                // ground that settles it: "this is consistent with blocks." A curve is reference
                // geometry — it exists to be built on — so every field that takes one should take it
                // the way an operand field takes a block, and until this the one drag source in the
                // panel was a block row's, which is why a curve could only be typed.
                //
                // It does not reorder. A curve has no rank among the bodies and no consumer to be
                // pulled out of, so `drop_on_the_list` never sees this payload: the gutters and the
                // empty area below both ask for a `NodeId` and a curve drag leaves them dark, which
                // is the honest reading — there is nowhere on the list for it to land.
                ui.dnd_drag_source(
                    ui.make_persistent_id(("curve_grip", id.raw())),
                    id,
                    |ui| grip_mark(ui),
                )
                .response
                .on_hover_text(
                    "Drag onto a block's curve field — a sweep's spine, a pipe's curve, a \
                     profile — to make it read this curve",
                );
                // **A real toggle, where it was once a dim marker.** Dimming it stopped the glyph
                // claiming a state the curve did not have, and that was ruled enough — until the
                // case the ruling named as the one to reopen for turned up: an extrude names one
                // curve of its four-curve loop, so when the loop opens the named one stays inside
                // the extrude with no way out (backlog item 73). Make variable is that way out, and
                // put back inside the way back, as on a block's row: the stored ownership, which
                // the list is drawn from (`block-ownership-spec.md` §10). A curve on a row of its
                // own because nothing reads it is free, not a variable, so its V is dim too, as a
                // body's is.
                // **A curve whose row is its loop's has no V of its own to click** — the curve that
                // heads it, and one a block reads ([`curve_v_elsewhere`] says which and why). The
                // row above carries the gesture for the whole loop; this one shows the state and
                // stops, and the row menu's Variable item does the same.
                let is_variable = self.doc.ownership(BlockKey::Curve(id)) == Ownership::Variable;
                let toggle = match curve_v_elsewhere(&self.profiles, id, readers) {
                    Some(why) => Err(why),
                    None => Ok(Declarable::Curve(id)),
                };
                self.variable_marker(ui, is_variable, toggle);
                let size = egui::vec2(18.0, ui.spacing().interact_size.y);
                if ui.add_sized(size, egui::Button::new(icon).frame(false)).clicked() {
                    if showing {
                        self.hidden_curves.insert(id);
                    } else {
                        self.hidden_curves.remove(&id);
                    }
                }
                let selected = self.selected_curve == Some(id);
                let Some(row) = self.list_row(ui, RowKey::Curve(id), &header, selected) else {
                    return;
                };
                if row.clicked() || row.secondary_clicked() {
                    self.select_curve(id);
                }
                row.context_menu(|ui| self.row_menu(ui, Row::Curve { id, readers }));
            })
            .body(|ui| {
                // First, exactly where a body's name is. A curve has carried a name since it
                // was created — "Line", "Curve" — and until now there was no way to change it,
                // which mattered more once curves started being *referenced*: a block's Ends
                // list is a list of curve names, and six lines all called "Line" is a list you
                // cannot use.
                // Renamed on its own row, like a body's -- double-click it. A curve's name is
                // what every list of curves shows, so it was worth two clicks less to change.
                ui.separator();
                ui.label(
                    egui::RichText::new(format!(
                        "degree {degree}, {points} control points, {length:.1} mm long"
                    ))
                    .weak(),
                );
                if let Some(axis) = surface_axis {
                    self.surface_axis_controls(ui, id, axis);
                }
                // **Whether this curve is part of a closed loop, said where it is asked.**
                // A profile is a loop — one curve or several meeting end to end — and until this
                // the only way to find out whether yours closed was to try to extrude it and read
                // the refusal. Two ends a fraction of a millimetre apart look closed at any zoom
                // that shows the whole shape, so the answer has to be computed rather than
                // eyeballed.
                //
                // **Not on an axis.** An axis is two points standing on a surface and is never a
                // profile, so "these curves do not close: there is a 50.000 mm gap" is an answer
                // to a question nobody asked — it read as a fault in the axis, and its gap is just
                // the axis's own length.
                // On the ends alone: this runs every frame the row is open, and walking the
                // polylines cost 20 ms a frame on seven curves.
                match fieldform_field::loop_from(&self.doc, id, None) {
                    _ if surface_axis.is_some() => {}
                    Ok(curves) => {
                        let n = curves.len();
                        let made_of = if n == 1 {
                            "closed on its own".to_string()
                        } else {
                            let names: Vec<String> = curves
                                .iter()
                                .filter_map(|c| self.doc.curve(*c).map(|c| c.name.clone()))
                                .collect();
                            format!("a closed loop of {n}: {}", names.join(", "))
                        };
                        ui.label(
                            egui::RichText::new(format!("{made_of} — can be extruded"))
                                .color(egui::Color32::from_rgb(90, 190, 110)),
                        );
                    }
                    Err(e) => {
                        ui.label(
                            egui::RichText::new(e.to_string()).color(ui.visuals().warn_fg_color),
                        )
                        .on_hover_text(
                            "A profile has to be a closed loop. Drag an end onto another \
                             curve's end with Ctrl held to join them exactly.",
                        );
                    }
                }
                // The degree, changeable after the fact.
                //
                // Offered only for a single span — a Bézier — because that is the case where
                // raising the degree is exact. Raising adds control points without moving the
                // curve at all, so going to 9 to get finer control costs nothing; lowering
                // discards shape it cannot hold, keeps the two ends, and says so.
                let single_span = surface_axis.is_none()
                    && self.doc.curve(id).is_some_and(|c| c.is_single_span());
                if single_span {
                    let mut chosen = degree;
                    ui.horizontal(|ui| {
                        row_label(ui, "Degree:");
                        egui::ComboBox::from_id_salt((id.raw(), "curve_degree"))
                            .selected_text(degree.to_string())
                            .show_ui(ui, |ui| {
                                for d in CURVE_DEGREES {
                                    ui.selectable_value(&mut chosen, d, d.to_string());
                                }
                            });
                        if chosen < degree {
                            ui.label(
                                egui::RichText::new("lowering approximates")
                                    .weak()
                                    .small(),
                            );
                        }
                    });
                    if chosen != degree
                        && let Some(before) = self.doc.curve(id).cloned()
                    {
                        match before.with_degree(chosen) {
                            Ok(after) => {
                                let raised = chosen > degree;
                                let cmd = Box::new(SetCurve::new(id, Some(before), Some(after)));
                                // Not a graph edit, so it ends any gesture a value drag had
                                // open rather than becoming the entry that drag replaces next.
                                self.gesture = None;
                                if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                                    self.warn(e.to_string());
                                } else {
                                    // Indices outlive nothing here: the count changed, so what
                                    // was picked names different points now.
                                    self.clear_point_selection();
                                    self.say(if raised {
                                        format!("Degree {chosen} — same shape, more points")
                                    } else {
                                        format!("Degree {chosen} — shape approximated")
                                    });
                                }
                                self.mark_changed();
                                self.rebuild_curve_display();
                                self.rebuild_render_resources();
                            }
                            Err(e) => self.warn(e.to_string()),
                        }
                    }
                }

                // What the picked points pull with. Shown only when some are picked, because a
                // weight belongs to a point rather than to the curve, and a field with no point
                // to apply to would have to invent one.
                let picked: Vec<usize> = if self.selected_curve == Some(id) {
                    self.selected_points.iter().copied().filter(|i| *i < points).collect()
                } else {
                    Vec::new()
                };
                // Weights are hidden for now — the control they give is real and the vocabulary
                // for it is not something to put in front of someone still learning what a
                // control point does. The model, the command and the drawing all keep working,
                // so this is one `false` away from coming back.
                const SHOW_WEIGHTS: bool = false;
                if SHOW_WEIGHTS && !picked.is_empty() {
                    let weights: Vec<f64> = self
                        .doc
                        .curve(id)
                        .map(|c| picked.iter().map(|i| c.weights()[*i]).collect())
                        .unwrap_or_default();
                    // One field for the lot when several are picked, showing their shared value
                    // or the first: pulling three points harder is one intention, not three.
                    let mut weight = weights.first().copied().unwrap_or(1.0);
                    let label = if picked.len() == 1 {
                        "Weight:".to_string()
                    } else {
                        format!("Weight ({} points):", picked.len())
                    };
                    let mut began = false;
                    let mut touched = false;
                    let mut settled = false;
                    ui.horizontal(|ui| {
                        ui.label(label);
                        let response = ui
                            .add(
                                egui::DragValue::new(&mut weight)
                                    .speed(0.02)
                                    .range(0.01..=100.0)
                                    .max_decimals(3),
                            )
                            .on_hover_text(
                                "How hard this point pulls the curve toward itself. 1 is \
                                 neutral; more pulls it nearer, less lets it drift away.",
                            );
                        began = response.drag_started() || response.gained_focus();
                        touched = response.changed();
                        settled = response.drag_stopped() || response.lost_focus();
                    });

                    // Live while dragging, one entry on release — the rule every drag here
                    // follows.
                    if began && self.weight_edit_before.is_none() {
                        self.weight_edit_before = Some((id, picked.clone(), weights.clone()));
                    }
                    if touched {
                        if let Some(c) = self.doc.curve_mut(id) {
                            for i in &picked {
                                c.set_weight(*i, weight);
                            }
                        }
                        self.rebuild_curve_display();
                        if self.curve_shapes_a_body(id) {
                            self.traced.invalidate(traced::Quality::Draft);
                        }
                        self.mark_changed();
                    }
                    if settled
                        && let Some((edited, indices, before)) = self.weight_edit_before.take()
                        && edited == id
                        && before.iter().any(|w| (w - weight).abs() > 1e-12)
                    {
                        if let Some(c) = self.doc.curve_mut(id) {
                            for (i, w) in indices.iter().zip(&before) {
                                c.set_weight(*i, *w);
                            }
                        }
                        let after = vec![weight; indices.len()];
                        let cmd = Box::new(SetControlWeights::new(id, indices, before, after));
                        // Ends a graph gesture, as the degree above does.
                        self.gesture = None;
                        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                            self.warn(e.to_string());
                        }
                        self.rebuild_curve_display();
                        self.rebuild_render_resources();
                    }
                }

                // A radius and a "sweep into a pipe" button used to live here, and they were
                // the pipe's, not the curve's. A curve has no thickness — that is the whole
                // difference between the two — and a block carrying a control for something it
                // is not says the two are one thing. Create > Pipe makes a pipe, and the pipe's
                // own block says which curve it sweeps and how wide.
                //
                // Two control points is a line, and a line is a span between two places, so
                // that is what it offers. Anything with more is a shape rather than a span, and
                // three numbers per point would be a spreadsheet instead of a block.
                // Which points get numbers: the ones picked, whatever the curve's degree.
                //
                // This used to be the two ends of a two-point curve and nothing else, which
                // covered a line and left every other curve — every imported one, and every
                // Bézier above degree one — with a shape you could drag and could not type.
                // Picking is what says which point is meant, so a curve of any length shows
                // exactly as much as was asked for rather than a column of every point it has.
                //
                // **The two ends are always here, whatever the degree.** They used to appear
                // only on a two-point curve, so a cubic showed nothing until a point was picked
                // — and where a curve starts and stops is the thing people reach for most,
                // regardless of how many control points sit between them. Everything picked is
                // shown as well, and the whole list runs in curve order so the numbers read the
                // way the curve does.
                let mut editing: Vec<usize> = vec![0];
                if points > 1 {
                    editing.push(points - 1);
                }
                for &i in &picked {
                    if !editing.contains(&i) {
                        editing.push(i);
                    }
                }
                editing.sort_unstable();
                // No coordinates for an axis on a surface: the body decides them.
                if points == 0 || surface_axis.is_some() {
                    editing.clear();
                }
                if !editing.is_empty() {
                    let mut edited = positions.clone();
                    let (mut touched, mut began, mut settled) = (false, false, false);
                    // Which coordinate was just bound or unbound, applied after the rows are
                    // drawn -- egui hands out `&mut self` for the duration of the closure, and
                    // a document edit inside it would be a second borrow.
                    let mut bind: Option<(usize, NodeId)> = None;
                    let mut unbind: Option<usize> = None;
                    // Which control point asked for a point block of its own, acted on below
                    // where the graph can be edited.
                    let mut declare: Option<usize> = None;
                    // **A grid, so the columns line up.** Each end wants its three
                    // coordinates on one line — a position reads as a position that way — and
                    // Start's X has to sit directly above End's X, which plain rows cannot
                    // promise: they size to their own content, so two rows of different numbers
                    // come out ragged. A grid gives every column one width.
                    egui::Grid::new((id, "curve_points"))
                        .num_columns(5)
                        .spacing([6.0, 4.0])
                        .show(ui, |ui| {
                            for &i in &editing {
                                // The ends are named for what they are; anything between them
                                // is an index, which is what it actually is and matches the
                                // order the curve runs in.
                                // The same label width every other position uses, so a
                                // curve's ends line up with a sphere's centre rather than
                                // merely with each other.
                                row_label(
                                    ui,
                                    &(if i == 0 {
                                        "Start".to_string()
                                    } else if i + 1 == points {
                                        "End".to_string()
                                    } else {
                                        format!("{}", i + 1)
                                    }),
                                );

                                if let Some(node) = placed_on(self, id, i) {
                                    // The point says where this is, so its name stands where
                                    // the numbers would be rather than beside them.
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new("V").color(VARIABLE_ON).strong(),
                                        );
                                        ui.label(
                                            egui::RichText::new(self.body_label(node)).italics(),
                                        )
                                        .on_hover_text(
                                            "Sitting on a point — move that point and \
                                             this end follows it",
                                        );
                                        if remove_button(
                                            ui,
                                            "Take it off, keeping where it is",
                                        ) {
                                            unbind = Some(i);
                                        }
                                    });
                                    ui.label("");
                                    ui.label("");
                                    ui.label("");
                                    ui.end_row();
                                    continue;
                                }

                                for (axis, axis_label) in ["X", "Y", "Z"].iter().enumerate() {
                                    let mut value = edited[i][axis];
                                    let response = ui.add_sized(
                                        egui::vec2(AXIS_FIELD_W, ui.spacing().interact_size.y),
                                        egui::DragValue::new(&mut value)
                                            .speed(0.5)
                                            .max_decimals(3)
                                            .prefix(format!("{axis_label} ")),
                                    );
                                    if response.drag_started() || response.gained_focus() {
                                        began = true;
                                    }
                                    if response.changed() {
                                        edited[i][axis] = value;
                                        touched = true;
                                    }
                                    if response.drag_stopped() || response.lost_focus() {
                                        settled = true;
                                    }
                                }

                                // **Always offered, even with no point to sit on.** It was
                                // shown only when the part already held one, so a document
                                // without a point had no V at all and nothing to say why —
                                // the mistake the revolve axis made once, and the reason the
                                // rule here is to mark rather than withhold.
                                let available = point_blocks(self);
                                let mark = egui::RichText::new("V").color(VARIABLE_OFF);
                                ui.menu_button(mark, |ui| {
                                    // The same offer a place gets: make the point rather than
                                    // only choose among points that happen to exist.
                                    if ui
                                        .button("Make a point here")
                                        .on_hover_text(
                                            "Makes a named point block where this end is and \
                                             sits the end on it",
                                        )
                                        .clicked()
                                    {
                                        declare = Some(i);
                                        ui.close();
                                    }
                                    if available.is_empty() {
                                        ui.label(
                                            egui::RichText::new(
                                                "nothing to sit on yet — making one is \
                                                 the offer above",
                                            )
                                            .weak(),
                                        );
                                        return;
                                    }
                                    ui.separator();
                                    ui.label(egui::RichText::new("sit on").weak());
                                    for (node, name) in &available {
                                        if ui.button(name).clicked() {
                                            bind = Some((i, *node));
                                            ui.close();
                                        }
                                    }
                                })
                                .response
                                .on_hover_text(
                                    "Sit this end on a point, so moving the point moves it",
                                );
                                ui.end_row();
                            }
                        });

                    // The same rule every drag in this app follows: move the document live so
                    // the pipe around it follows, and put *one* entry on the history when the
                    // hand comes off. Committing per frame would bury an undo stack under a
                    // hundred one-tenth-of-a-millimetre steps.
                    // Made first, then bound by the same path a picked point takes -- so there
                    // is one way an end comes to sit on something, however the point arrived.
                    if let Some(i) = declare {
                        let at = Point3::new(edited[i][0], edited[i][1], edited[i][2]);
                        let point = self.point_block_at(at);
                        bind = Some((i, point));
                    }
                    if let Some((i, node)) = bind {
                        // A point a relation derives is that relation's to move, not a
                        // block's (`profile-group-spec.md` §6): refused, naming the owner.
                        let owners = fieldform_scene::relation::owners_of(&self.doc, id, i, None);
                        if let Some(owner) = owners.first().and_then(|o| self.doc.relations.get(o)) {
                            let what = owner.relation.describe(&self.doc);
                            self.warn(format!(
                                "point {} is derived by {what} — break it to place this point",
                                i + 1
                            ));
                        } else {
                            if let Some(c) = self.doc.curve_mut(id) {
                                c.place_on(i, Some(node));
                            }
                            self.settle_curves_on_points();
                            self.mark_changed();
                            self.rebuild_curve_display();
                            self.rebuild_render_resources();
                        }
                    }
                    if let Some(i) = unbind {
                        if let Some(c) = self.doc.curve_mut(id) {
                            c.place_on(i, None);
                        }
                        self.mark_changed();
                    }
                    if began && self.curve_edit_before.is_none() {
                        self.curve_edit_before = Some((id, positions.clone()));
                    }
                    if touched {
                        if let Some(c) = self.doc.curve_mut(id) {
                            for (i, p) in edited.iter().enumerate() {
                                c.set_control_point(i, *p);
                            }
                        }
                        self.rebuild_curve_display();
                        // Only when a field moved: a loose curve's numbers redraw the
                        // overlay and nothing else.
                        if self.curve_shapes_a_body(id) {
                            self.traced.invalidate(traced::Quality::Draft);
                        }
                        self.mark_changed();
                    }
                    if settled
                        && let Some((edited_curve, before)) = self.curve_edit_before.take()
                        && edited_curve == id
                        && before != edited
                    {
                        for (i, p) in before.iter().enumerate() {
                            if let Some(c) = self.doc.curve_mut(id) {
                                c.set_control_point(i, *p);
                            }
                        }
                        let cmd = Box::new(SetControlPoints::new(
                            id,
                            (0..before.len()).collect(),
                            before,
                            edited,
                        ));
                        // Ends a graph gesture, as the degree above does.
                        self.gesture = None;
                        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
                            self.warn(e.to_string());
                        }
                        self.rebuild_curve_display();
                        self.rebuild_render_resources();
                    }
                }
                self.relation_controls(ui, id);
            });
    }

    /// One graph node, as a block: what it is, its own parameters, and the blocks it is made of.
    ///
    /// `inside` is the rows drawn within this one — the blocks only this one reads and nobody has
    /// named. `nested` is whether *this* row is itself drawn within another, which decides whether
    /// its grip means "move it among the bodies" or "pull it out". [`Self::nested_blocks`] works
    /// both out, from `nesting`, which the rows inside are drawn from too.
    pub(crate) fn node_block(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        inside: &[BlockRow],
        references: &[BlockKey],
        nested: bool,
        nesting: &fieldform_scene::Nesting,
        holder: Option<NodeId>,
    ) {
        // Asked only to know the block is real: the row reads nothing else off it now, since
        // both the name and the kind come from the labels.
        if self.part_graph().node(id).is_none() {
            return;
        }
        // Looked up in the nesting the rows were laid out from rather than asked of the graph,
        // which walks every node and made the list quadratic in the part.
        let is_body = nesting.is_root(id);
        // **The name, then what it is, and nothing else.** Asked for: a row read
        // "TEST3 · TEST, TEST2 · body", which is a name followed by a list of the bodies
        // it combines -- so the thing the row is *about* was one item in a list of the things it
        // was made from, and a boolean of forty bodies was a row forty names long.
        //
        // The derived label already says which operation it is, so a union reads "Union" and a
        // subtraction "Subtract" -- more than "Boolean" would have said. What a block is made
        // *from* lives in the block, where every operand has a picker of its own.
        //
        // Only when there is a name to put first: an unnamed block's label already *is* what it
        // is, and "Sphere · Sphere" says it twice.
        let mut header = self.body_label(id);
        if self.body_name(id).is_some() {
            header.push_str(&format!("  \u{b7}  {}", self.derived_label(id)));
        }
        // The eye says whether it is shown, so the row only has to say whether it is a body
        // at all -- everything else is inside one.
        if is_body {
            header.push_str("  \u{b7}  body");
        }
        // A session's block is marked until it is kept or taken back (`interaction-spec.md` §3.2).
        if self.is_pending(id) {
            header.push_str("  \u{b7}  pending");
        }
        // **And how many blocks it holds, because a closed row hides them.** A union of a sphere
        // and a cube reads "Union" with both of its ingredients behind a triangle, and a row that
        // said nothing about them would leave the two blocks somebody had just made looking
        // deleted. The same thing an import's row says about its solids, for the same reason.
        if !inside.is_empty() {
            header.push_str(&format!("  \u{b7}  {} inside", inside.len()));
        }
        // **And the variables it reads, for the same reason** (item 108): a sweep with one of its
        // profiles made a variable used to read one profile short, closed or open, and nothing on
        // the row said where the other had gone.
        if !references.is_empty() {
            let n = references.len();
            header.push_str(&format!(
                "  \u{b7}  {n} variable{}",
                if n == 1 { "" } else { "s" }
            ));
        }

        // The row is built by hand rather than with a plain collapsing header, because the
        // eye has to sit *in* it — a visibility toggle two clicks deep, inside the thing it
        // controls, is not a visibility toggle. Every layer panel from Photoshop to every CAD
        // tree puts it on the row, and for the same reason: you are usually toggling several.
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("node", id.raw())),
            false,
        );
        // **A block being made opens here, waiting for its numbers**, when there is no floating
        // panel to type them into (the owner's trial, 2026-10-17): the panel showed the same
        // numbers the block's own rows do, and two places for one value is one too many.
        let awaiting = !self.floating_panel && self.session.as_ref().is_some_and(|s| s.block == id && s.awaiting);
        if awaiting {
            state.set_open(true);
            state.store(ui.ctx());
        }
        // A list collects only while its block is open. Closing the block is the natural end of
        // the act, and a viewport that keeps adding to a list nobody can see is a trap.
        if self.collecting == Some(id) && !state.is_open() {
            self.collecting = None;
        }
        let header_response = state
            .show_header(ui, |ui| {
                // **A grip, not the whole row.** Making the row itself the drag source would put
                // a drag in front of every click it already answers -- select, rename on a double
                // click, and the context menu -- so the one gesture that reorders gets a handle of
                // its own. Every layer panel that reorders does this, and for the same reason.
                //
                // **Every block, because a grip now means three things and the third applies to
                // all of them.** A body's reorders it among the bodies; a nested row's takes it out
                // of the block that holds it (`drop_on_the_list` is where those two part); and any
                // block at all can be dropped onto a reference field to become what another block
                // is made from, which is the gesture the owner asked for -- "you could click and
                // drag on the sphere and drag it into the block and it would load."
                //
                // It was `is_body || nested`, on the reasoning that everything else is pinned in the
                // list by dependency or by its section. That is still true of where a row can *go*,
                // and it is no longer the only thing a drag is for: the one row it left without a
                // grip was a variable that is not a body, which is precisely the row somebody
                // wants to drag into a field.
                //
                // Dropping a variable on the list is harmless rather than special-cased: its
                // row is at the top level, so the drop finds no slot to take it out of and does
                // nothing ([`Self::take_out_of_its_slot`]).
                // **Keyed by the holder as well as the block**, because a shared instance has
                // a row in every reader and egui keeps one drag state per id: with the block's
                // id alone the two rows of one shared block were one drag source, and §7.2's
                // "out of *that* slot only" is exactly the distinction that erased.
                ui.dnd_drag_source(
                    ui.make_persistent_id(("grip", id.raw(), holder.map(|h| h.raw()))),
                    DraggedBlock { id, from: holder },
                    |ui| grip_mark(ui),
                )
                .response
                .on_hover_text(if is_body {
                    "Drag to move this block in the list, or onto a block's From or Subtract field"
                } else if nested {
                    "Drag onto the list to take this block out of the block that holds it, \
                     leaving that slot waiting to be told what to read, or onto another block's \
                     From or Subtract field"
                } else {
                    "Drag onto another block's From or Subtract field to make it read this block"
                });
                let is_variable = self.doc.ownership(BlockKey::Node(id)) == Ownership::Variable;
                self.variable_marker(ui, is_variable, Ok(Declarable::Block(id)));
                self.visibility_toggle(ui, id, is_body);
                let selected = self.selected_body == Some(id);
                // Truncated, with the whole thing on hover. A label names what a body is
                // made of, and what a body is made of grows with every operation applied to
                // it — so an untruncated row demanded width without limit and took it from
                // the viewport. The name is the part anybody reads; the chain is reference.
                // **The whole width of the row, with the name against its left edge.**
                //
                // Asked for twice, and a wider `Button` is not the way: a button centres its
                // text, so giving one the panel's width put the name off the visible edge. What
                // the row wants is a left-aligned label drawn on a row-wide background, which
                // is a different widget rather than a bigger one — so it is painted here.
                //
                // No stroke on the background: this sits inside the block's own frame, and a
                // second outline a pixel inside the first reads as a rendering fault. The fill
                // is the accent when selected, which `interact_selectable` supplies along with
                // the hover states, so the row responds like every other clickable thing.
                let Some(row) = self.list_row(ui, RowKey::Node(id), &header, selected) else {
                    return;
                };
                // Either button selects. The right one selects *before* its menu opens, because
                // every item in that menu acts on the selection -- so without this, right-clicking
                // one row and choosing Delete would delete whichever row was selected before.
                if row.clicked() || row.secondary_clicked() {
                    self.select_body(Some(id));
                }
                row.context_menu(|ui| self.row_menu(ui, Row::Node { id, is_body }));
            })
            .body(|ui| {
                self.focus_first_number = awaiting;
                self.drawing_pending = self.is_pending(id) && !self.floating_panel;
                if self.drawing_pending {
                    self.pending_numbers.clear();
                }
                self.node_block_body(ui, id);
                self.drawing_pending = false;
                if awaiting {
                    // Done once the first number has the keyboard, or once the block has none.
                    self.focus_first_number = false;
                    if let Some(session) = self.session.as_mut() {
                        session.awaiting = false;
                    }
                    ui.scroll_to_cursor(Some(egui::Align::Center));
                }
                // **What this block is made of, inside it.** One row that opens, which is the
                // shape an import's solids and a profile's curves already have -- and the reason
                // the contents share the block's own disclosure rather than getting a second one:
                // a block put away should put away what it is made of, or closing it leaves its
                // ingredients standing where the block used to be.
                if !inside.is_empty() || !references.is_empty() {
                    ui.separator();
                }
                for row in inside {
                    // This block is their holder, which is what makes a drag off one of them
                    // able to empty the slot it came from (§7.2).
                    self.block_row(ui, row, nesting, Some(id));
                }
                // After what it owns, so the rows it holds and the variables it only refers to
                // read as two different things in one place.
                for key in references {
                    self.reference_slot(ui, *key, nesting);
                }
            });
        let _ = header_response;
    }

    /// One input of a block that lives outside it as a variable: named, marked as living
    /// elsewhere, and leading to it (backlog item 108).
    ///
    /// **A reference, not a copy, so it shows no numbers.** The variable's own row is where it is
    /// edited; drawing its fields here too would be one block edited from two places, which is
    /// what "one editor, not one per block" (D9) exists to prevent. A click selects the variable,
    /// which is the way to it, and the hover says who else shares it — the durable form of the
    /// sentence settle composes once in the status line.
    ///
    /// The same green V the variable's own row wears, so the two are recognisably one thing.
    fn reference_slot(&mut self, ui: &mut egui::Ui, key: BlockKey, nesting: &fieldform_scene::Nesting) {
        // A curve that is one side of a closed loop is shown as the profile it belongs to, which
        // is what the person made a variable and what its row is called.
        let label = match key {
            BlockKey::Node(n) => self.body_label(n),
            BlockKey::Curve(c) => {
                self.refresh_profiles();
                match self.profiles.key_of(c) {
                    Some(loop_key) => self.profile_label(loop_key),
                    None => self.doc.curve(c).map(|c| c.name.clone()).unwrap_or_default(),
                }
            }
        };
        let others: Vec<String> = nesting
            .readers_of_variable(key)
            .into_iter()
            .map(|reader| self.body_label(reader))
            .collect();
        let shared = match others.len() {
            0 | 1 => "Nothing else reads it yet".to_string(),
            _ => format!("Read by {}", others.join(", ")),
        };
        let response = ui
            .horizontal(|ui| {
                row_label(ui, "");
                ui.label(egui::RichText::new("V").color(VARIABLE_ON).strong());
                ui.add(
                    egui::Button::selectable(
                        false,
                        egui::RichText::new(format!("↗ {label}")).italics(),
                    )
                    .frame(false),
                )
            })
            .inner
            .on_hover_text(format!(
                "{label} is a variable: a row of its own at the top of the list, which this block \
                 refers to rather than holds. Edit it there. {shared}. Click to go to it."
            ));
        if response.clicked() {
            match key {
                BlockKey::Node(n) => self.select_body(Some(n)),
                BlockKey::Curve(c) => self.select_curve(c),
            }
        }
    }

    /// Draws one row of the block list, and whatever is nested inside it.
    ///
    /// **One place a block is boxed, so a curve, a body, an import and a row nested three deep are
    /// all bounded the same way** — and a fifth kind is too without anyone remembering. It used to
    /// be the loop in the panel; it is a function because the list is a tree now and a row inside a
    /// row comes back through here.
    ///
    /// **The frame is no longer a drop zone, and that is what the nesting required.**
    /// `dnd_drop_zone` highlights the whole box it is given, so with contents in that box the
    /// feedback said "goes inside this block" while the drop meant "goes before it". Dropping
    /// moved to the gutters between the rows ([`drop_gutter`]), which can only mean one thing. Worth knowing for
    /// anyone comparing screenshots: that zone also *replaced* the frame's fill and stroke with
    /// egui's own, so the box had no border at all until this moved, although the code has asked
    /// for one since it was written.
    ///
    /// `nesting` is the one `row` was laid out from ([`Self::nested_blocks`]).
    pub(crate) fn block_row(
        &mut self,
        ui: &mut egui::Ui,
        row: &BlockRow,
        nesting: &fieldform_scene::Nesting,
        holder: Option<NodeId>,
    ) {
        // **A box around each block, closed at the bottom.** Expanding one used to leave its
        // contents running straight into the next row: the collapsing header draws a line down its
        // left and stops, so where one block ended and the next began was something you worked out
        // from indentation. Enclosing the whole thing -- header, parameters and contents together
        // -- means the boundary is drawn rather than inferred, and it grows and shrinks with the
        // block because the frame wraps the collapsing state rather than sitting beside it.
        let frame = egui::Frame::new()
            .stroke(egui::Stroke::new(
                1.0_f32,
                ui.visuals().widgets.noninteractive.bg_stroke.color,
            ))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(4, 3));
        frame.show(ui, |ui| {
            // Full width, so the right-hand edge lines up down the list instead of following
            // whichever label happens to be longest.
            ui.set_width(ui.available_width());
            match row.block {
                Block::Parameter(id) => self.parameter_block(ui, id),
                Block::Curve(id) => self.curve_block(ui, id, row.readers),
                Block::Surface(id) => self.surface_block(ui, id),
                Block::Tessellation(id) => self.tessellation_block(ui, id),
                Block::Export(id) => self.export_block(ui, id),
                Block::Node(id) => {
                    self.node_block(ui, id, &row.inside, &row.references, row.nested, nesting, holder)
                }
                Block::Import(source) => self.import_block(ui, source, nesting),
                Block::Profile(key) => self.profile_block(ui, key, nesting),
            }
        });
    }

    /// A block let go over the list: `onto` is the row it landed above, or `None` for the end.
    ///
    /// **Two meanings on one target, because to the hand it is one gesture** -- take this row and
    /// put it out here. What a drop can *say* differs. A body has a rank among the bodies, so it
    /// takes the one the gutter stands for. A nested block has no rank to take -- dependency pins
    /// it directly above whatever consumes it (`block-order-spec.md` §2) -- so what it gets is
    /// what §7.2 says: it comes out of the slot it was in, as a **plain** block.
    ///
    /// **It does not declare** (`notebook-spec.md` R5, §7.2). This used to reach *Make variable*,
    /// which left the block wired to its holder and named it -- the pre-112 model, where a row of
    /// one's own and a declaration were the same thing. They are two things now, and the section
    /// names the difference in a sentence: "To pull a block out *and* keep it wired, use *Make
    /// variable*." A drop that declared would leave that sentence with nothing to distinguish.
    pub(crate) fn drop_on_the_list(&mut self, dragged: DraggedBlock, onto: Option<NodeId>) {
        if !self.part_graph().roots().contains(&dragged.id) {
            self.take_out_of_its_slot(dragged);
            return;
        }
        match onto {
            Some(onto) => self.arrange_block(dragged.id, onto),
            None => self.arrange_block_last(dragged.id),
        }
    }

    /// §7.2's drag-out: the slot the row came from is emptied and the block stays as it is.
    ///
    /// **One edit, so one undo step**, through [`Self::replace_node`] -- the same writer the
    /// operand row's *Clear* goes through, which is the other way to this state and says so in its
    /// own hover text. Nothing is written about the block that comes out: a block nothing reads is
    /// a row of its own once the edit settles, and `Ownership::Free` is what settling gives it. So
    /// the gesture is the holder letting go, and the block being plain follows from it rather than
    /// being a second thing to remember.
    ///
    /// **A shared instance leaves that slot only** (§7.2): the holder is the one whose row was
    /// grabbed, so the other readers are not touched and the block is still nested in them.
    /// Settling sees a block that is still read and leaves it nested, with no special case here.
    ///
    /// A row with no holder cannot be dragged out of anything -- `drop_on_the_list` sends only a
    /// non-root here, and a non-root is read by something, so this is a payload from an older
    /// frame rather than a state to handle. Said rather than silently ignored, because a grip that
    /// does nothing is the bug this section exists to fix.
    fn take_out_of_its_slot(&mut self, dragged: DraggedBlock) {
        // **A row with no holder is already at the top level**, so there is nothing to take it
        // out of and the drag is harmless rather than special-cased: a variable that is not a body
        // is draggable, and dropping it on the list is precisely this case. `from` is `None`
        // exactly when the row was drawn at the top level, since every nested row is drawn from
        // its holder's own `inside` list, so this needs no state to be sure of.
        let Some(holder) = dragged.from else { return };
        let Some(mut node) = self.part_graph().node(holder).cloned() else { return };
        if !node.stop_reading(dragged.id) {
            // The graph moved under the drag: the holder no longer reads it, so the gesture has
            // already happened or was undone. Nothing to do, and nothing to say about it.
            return;
        }
        let what = self.body_label(dragged.id);
        let from = self.body_label(holder);
        self.replace_node(holder, node);
        // **Said only if it went.** `replace_node` refuses an edit that would leave a graph it
        // cannot draw and warns in its own words, so announcing the drag-out before asking would
        // put a success and a refusal in the status bar together -- and the success would be the
        // one left on screen. Asked of the graph rather than of a flag, since the graph is what
        // was or was not written.
        if self.part_graph().node(holder).is_some_and(|n| !n.operands().contains(&dragged.id)) {
            self.say(format!(
                "\u{201c}{what}\u{201d} is out of {from}, which is waiting to be told what to read instead"
            ));
        }
    }

    /// *Make variable* for any row with a V: a block's, a curve's, or a profile's, which makes the
    /// curve of its loop that puts the row where it is one (`block-ownership-spec.md` §10, §5.1).
    /// The block takes a row of its own, at its consumer's branch, and a name if it has none. One
    /// command and one undo step ([`make_variable`]) whichever it is, and nothing for a row that
    /// already is one. A curve is born named, so only a block can need a name invented.
    ///
    /// **One operation however it is reached** — the dim V, the row menu's Variable tick. **A drop
    /// on the list is not one of them** since item 112: that gesture takes a block out of its slot
    /// and declares nothing ([`Self::take_out_of_its_slot`], `notebook-spec.md` §7.2, R5). It had
    /// a method of its own for the drop to call, `pull_out`, which is gone with the drop that
    /// called it rather than left dormant.
    ///
    /// Declaring used to be a name, because the list was laid out by names and *"to pull it out
    /// you rename it"* was the owner's ruling; a name typed for any other reason then pulled a
    /// block out too, which is what storing the ownership ended. No edge moves: the block is still
    /// consumed by whatever consumed it, and it is still not a body — which is the whole of what
    /// separates this from §7.2's drag-out, where the edge is exactly what goes.
    ///
    /// **It names silently and does not take the keyboard.** This used to open the name for
    /// editing, on the reasoning that an invented name is a placeholder and the next thing typed
    /// is the thing that was missing. Reversed by the owner on 2026-09-30, once the V handle made
    /// this the ordinary way a block becomes a variable rather than a rare one: a text cursor
    /// appearing on every click is a modal state to dismiss, and pulling a block out is frequently
    /// all that was wanted. `unique_variable_name` carries the weight instead — the invented name
    /// is unique and spelled as an identifier — and a double click on the row still renames it.
    ///
    /// **It does not start drawing the block.** Showing an ingredient costs a compile and puts it
    /// on screen, which is a question about the viewport with its own control — the eye, on the
    /// row this just made.
    fn make_variable_of(&mut self, what: Declarable) {
        let invented = || match what {
            Declarable::Block(id) => self.unique_variable_name(id),
            Declarable::Profile { .. } | Declarable::Curve(_) => self.declarable_label(what),
        };
        let Some(cmd) = make_variable(&self.doc, what.block(), invented) else { return };
        if !self.set_ownership(cmd) {
            return;
        }
        let name = self.declarable_label(what);
        self.say(format!(
            "\u{201c}{name}\u{201d} has a row of its own now — double-click the row to rename it"
        ));
    }

    /// What a row with a V is called, as the line its V says names it.
    fn declarable_label(&self, what: Declarable) -> String {
        match what {
            Declarable::Block(id) => self.body_label(id),
            Declarable::Profile { key, .. } => self.profile_label(key),
            Declarable::Curve(id) => {
                self.doc.curve(id).map_or_else(|| "That curve".to_string(), |c| c.name.clone())
            }
        }
    }

    /// One row of the block list: its name, selectable across the whole width, renamed in place.
    ///
    /// **The name is edited where it is read.** It used to be a field inside the block, so
    /// naming a sphere meant opening it, finding the row, typing and closing it again -- four
    /// actions for a label that was already on screen. Double-clicking the row edits it, which
    /// is the gesture every file list and layer panel uses for the same thing.
    ///
    /// **Painted rather than a wide `Button`**, because a button centres its text: given the
    /// row's width it put the name off the visible edge, which is a mistake this made twice
    /// before it was written down. A left-aligned label on a row-wide background is a different
    /// widget, not a bigger one. No stroke of its own -- the row sits inside the block's frame,
    /// and a second outline a pixel inside the first reads as a rendering fault.
    ///
    /// `None` while the row is being renamed: there is a text box where the label was, and
    /// nothing for the caller to attach a click or a context menu to.
    ///
    /// One builder for a node and a curve, because both are rows with a name on them (D9).
    pub(crate) fn list_row(
        &mut self,
        ui: &mut egui::Ui,
        key: RowKey,
        header: &str,
        selected: bool,
    ) -> Option<egui::Response> {
        if self.renaming.as_ref().is_some_and(|(being, _)| *being == key) {
            let mut text = self.renaming.take().map(|(_, t)| t).unwrap_or_default();
            let hint = match key {
                RowKey::Node(id) => self.derived_label(id),
                RowKey::Curve(_) => "Curve".to_string(),
                RowKey::Surface(_) => "Surface".to_string(),
                RowKey::Tessellation(_) => fieldform_scene::Tessellation::DEFAULT_NAME.to_string(),
                RowKey::Export(_) => fieldform_scene::Export::DEFAULT_NAME.to_string(),
                RowKey::Import(_) => "Import".to_string(),
                RowKey::Profile(key) => {
                    format!("Profile {}", self.profiles.number(key).unwrap_or(0))
                }
            };
            // **Salted with the row, not placed by position.** A row can move while it is being
            // renamed -- a block made a variable or let go of takes a row of its own -- and a
            // field whose identity is its position keeps the keyboard where the row *was*. The
            // variables panel learned this when applying a name re-sorted the rows out from under
            // the field.
            let edit = ui.add(
                egui::TextEdit::singleline(&mut text)
                    .hint_text(hint)
                    .id_salt(("row_name", key))
                    .desired_width(ui.available_width()),
            );
            // Focused on the first frame only. Asking every frame would take the keyboard back
            // from anything else that wanted it, and hold it until the row was closed.
            if !edit.has_focus() && !edit.lost_focus() {
                edit.request_focus();
            }
            let escaped = ui.input(|i| i.key_pressed(egui::Key::Escape));
            let entered = ui.input(|i| i.key_pressed(egui::Key::Enter));
            if escaped {
                // Nothing was written, so there is nothing to undo.
            } else if entered || edit.lost_focus() {
                // Empty puts the derived label back rather than leaving a blank row -- the same
                // thing clearing the old field did, and what makes "no name" a state you can
                // return to. Not for a variable, which other blocks type by its name:
                // `set_body_name` refuses that clear, as it refuses it from anywhere.
                match key {
                    RowKey::Node(id) => {
                        self.set_body_name(id, text.trim());
                    }
                    RowKey::Curve(id) => self.rename_curve(id, text.trim()),
                    RowKey::Surface(id) => {
                        let name = text.trim();
                        if name.is_empty() {
                            self.doc.surface_names.remove(&id);
                        } else {
                            self.doc.surface_names.insert(id, name.to_string());
                        }
                        self.mark_changed();
                    }
                    // Through the history, like every other change to a block.
                    RowKey::Tessellation(id) => {
                        self.rename_output(outputs::BlockRef::Tessellation(id), text.trim())
                    }
                    RowKey::Export(id) => {
                        self.rename_output(outputs::BlockRef::Export(id), text.trim())
                    }
                    RowKey::Import(source) => {
                        self.rename_output(outputs::BlockRef::Import(source), text.trim())
                    }
                    RowKey::Profile(key) => self.rename_profile(key, text.trim()),
                }
            } else {
                self.renaming = Some((key, text));
            }
            return None;
        }

        let (rect, row) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
            egui::Sense::click(),
        );
        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact_selectable(&row, selected);
            ui.painter().rect(
                rect,
                visuals.corner_radius,
                visuals.weak_bg_fill,
                egui::Stroke::NONE,
                egui::StrokeKind::Inside,
            );
            // Clipped to the row rather than given an ellipsis: the whole thing is on the
            // hover, and the chain of operands after the name is reference rather than
            // something to read across.
            ui.painter().with_clip_rect(rect).text(
                rect.left_center() + egui::vec2(6.0, 0.0),
                egui::Align2::LEFT_CENTER,
                header,
                egui::TextStyle::Button.resolve(ui.style()),
                visuals.text_color(),
            );
        }
        if row.double_clicked() {
            let current = match key {
                RowKey::Node(id) => self.body_name(id).unwrap_or_default().to_string(),
                RowKey::Curve(id) => {
                    self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default()
                }
                RowKey::Surface(id) => self.doc.surface_name(id).to_string(),
                // The given name, or empty so the box shows the block's own as its hint.
                RowKey::Tessellation(id) => self
                    .doc
                    .tessellations
                    .iter()
                    .find(|t| t.id == id)
                    .and_then(|t| t.name.clone())
                    .unwrap_or_default(),
                RowKey::Export(id) => self
                    .doc
                    .exports
                    .iter()
                    .find(|e| e.id == id)
                    .and_then(|e| e.name.clone())
                    .unwrap_or_default(),
                RowKey::Import(source) => {
                    self.doc.sources.get(&source).map(|s| s.name.clone()).unwrap_or_default()
                }
                RowKey::Profile(key) => {
                    self.doc.profile_names.get(&key).cloned().unwrap_or_default()
                }
            };
            self.renaming = Some((key, current));
        }
        Some(row.on_hover_text(header.to_owned()))
    }

    /// The green **V** beside a variable, and a dim one beside a block that is not.
    ///
    /// **Being a variable does something, and until this nothing on the row said so.** A variable
    /// is a row of its own that any number of blocks refer to — that is how one sphere is cut
    /// from two lattices rather than modelled twice — and the only place that was written was the
    /// hover text of the field you typed a name into. Which is the right place to explain it and
    /// the wrong place to be reminded of it: you read that once, and afterwards the fact lives on a
    /// row you are only glancing at.
    ///
    /// `on` is whether the row is a variable: the stored ownership (`block-ownership-spec.md`
    /// §2.1) of the block it stands for — a node, a curve, or the curve that puts a profile row
    /// where it is. `toggle` is what a click acts on, or why the row's V is somewhere else.
    ///
    /// Dim rather than absent when it is not, for the reason the eye beside it is: a column that
    /// appears and disappears moves every label after it, and a list whose text does not line up
    /// is a list you read rather than scan.
    fn variable_marker(
        &mut self,
        ui: &mut egui::Ui,
        on: bool,
        toggle: Result<Declarable, &str>,
    ) {
        let size = egui::vec2(14.0, ui.spacing().interact_size.y);
        // **Dimmed rather than absent, and clickable, for a block that could be a variable.** The
        // slot was `allocate_space` — dead pixels — so a block drawn inside another had no way at
        // all to become a variable: the row menu was the named block's, and the drag needed a
        // gutter that only a body's branch has. Reported by the owner while testing a sphere and a
        // cube in a boolean: "you also don't have a way to declare it as a variable ... maybe if
        // the little V icon was grayed out, maybe you can just click on it."
        //
        // Both clicks are the operations the row menu and a drop reach too (`declare`,
        // `undeclare`), so this is a handle on them, not a second mechanism.
        let id = match toggle {
            Ok(id) => id,
            // **The state, and no handle, for a row whose V is somewhere else**, saying where. A
            // curve drawn in its loop's row is one case (`curve_v_elsewhere`): the profile row's V
            // carries the gesture for the whole loop, and a click here would be a second handle on
            // it or one that moved nothing. A surface is the other, always a row of its own. Dim
            // rather than blank when it is not a variable, because a column that comes and goes
            // moves every label after it.
            Err(why) => {
                let (colour, text) = if on {
                    (
                        VARIABLE_ON,
                        format!("A variable: every other block can refer to it by name. {why}"),
                    )
                } else {
                    (VARIABLE_OFF, why.to_string())
                };
                let mark = egui::RichText::new("V").color(colour).strong();
                ui.add_sized(size, egui::Label::new(mark).sense(egui::Sense::hover()))
                    .on_hover_text(text);
                return;
            }
        };
        if !on {
            let mark = egui::RichText::new("V").color(VARIABLE_OFF).strong();
            let hit = ui.add_sized(size, egui::Label::new(mark).sense(egui::Sense::click()));
            if hit
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(
                    // Neutral about what it is making, because a profile reaches this too and
                    // "this block" would be wrong about the one row that most needed the handle.
                    "Not a variable yet. Click to make it one: it takes a row of its own, and a \
                     name if it has none, and every other block can refer to it by that name.",
                )
                .clicked()
            {
                self.declare(id);
            }
            return;
        }
        // **It unmakes a variable too, and only where that is a question with one answer.** Asked
        // for by the owner, in exactly the terms the rule is written in: "you should be able to
        // undeclare it as a variable, as long as it's not linked to anything else ... if it's used
        // in more than one place, you can't undeclare it. Just like in programming."
        //
        // The test is the count of blocks that read this one, and it is not a policy invented for
        // the button -- it is the count *put back inside* goes by (`block-ownership-spec.md`
        // §5.2). One reader is a block it *can* go back inside, so the move has a destination.
        // Two have no single owner to return to, so there is nowhere for the row to go. Refusing
        // is the honest answer, and the readers are named on the hover so it says *why* rather
        // than just going dead.
        //
        // **Nothing goes dead here since 112.** A bright V was unclickable where two or more
        // blocks read the variable, there being no single block to go back inside; it goes back
        // inside every reader now (`notebook-spec.md` §6.6), so the click always has an answer.
        let mark = egui::RichText::new("V").color(VARIABLE_ON).strong();
        let hit = ui.add_sized(size, egui::Label::new(mark).sense(egui::Sense::click()));
        if hit
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text(
                "A variable: it has a row of its own and other blocks can refer to it by name. \
                 Click to turn that off — it goes back inside the block that reads it.",
            )
            .clicked()
        {
            self.undeclare(id);
        }
    }

    /// Block candidates as reference choices: labelled by what they are, addressed by their name.
    ///
    /// One place, so every block field answers "what may I type here" the same way. The address is
    /// `body_name` and not `body_label`, which is the whole of the block-versus-curve difference —
    /// see [`Self::reference_field`].
    ///
    /// Takes the labelled list each caller already built, so a field goes on showing exactly what
    /// its dropdown showed; only the address is added.
    pub(crate) fn block_choices(&self, listed: &[(NodeId, String)]) -> Vec<RefChoice<NodeId>> {
        listed
            .iter()
            .map(|(n, label)| RefChoice::declared(*n, label.clone(), self.body_name(*n)))
            .collect()
    }

    /// Curve candidates as reference choices, where the label **is** the address.
    ///
    /// The other half of the difference `block_choices` carries. A curve is born named, and
    /// `unique_curve_name` keeps those names apart — "Line", then "Line 2" — so the text on the row
    /// is already something that resolves to exactly one curve. A block has to be named before it
    /// has an address at all.
    pub(crate) fn curve_choices(&self, listed: &[(CurveId, String)]) -> Vec<RefChoice<CurveId>> {
        listed.iter().map(|(c, name)| RefChoice::named(*c, name)).collect()
    }

    /// What a field shows for the reference it currently holds.
    ///
    /// `missing` is what the dropdown each of these replaced showed when the current value was not
    /// among the choices, and it differs per field — "none", "missing", an em dash — so it stays the
    /// caller's word rather than becoming one answer for all of them.
    pub(crate) fn shown_label<K: Copy + PartialEq>(
        &self,
        choices: &[RefChoice<K>],
        current: K,
        missing: &str,
    ) -> String {
        choices
            .iter()
            .find(|c| c.key == current)
            .map_or_else(|| missing.to_string(), |c| c.label.clone())
    }

    /// Takes a reference a drag just let go of, or says why it could not go here.
    ///
    /// One place for both payload kinds, so a dropped block and a dropped curve are refused in the
    /// same words and neither can start saying something the other does not.
    fn take_dropped_reference<K: Copy + PartialEq>(
        &mut self,
        as_ref: K,
        choices: &[RefChoice<K>],
        kind: RefKind,
        what: &str,
    ) -> Option<K> {
        self.ref_edit = None;
        match choices.iter().find(|c| c.key == as_ref) {
            Some(c) => match c.refusal {
                Some(why) => {
                    self.say(format!("{what} cannot go here — {why}"));
                    None
                }
                None => Some(as_ref),
            },
            None => {
                self.say(format!("{what} {}", kind.why_not()));
                None
            }
        }
    }

    /// A field naming the thing a block is made from: type the name, or drop the block on it.
    ///
    /// **One widget for every "which block" and "which curve" in the panel (D9).** Asked for as the
    /// way the whole interface should work: "you should be able to just drop the variables or the
    /// SDFs or the blocks into that area ... it doesn't need to be a drop-down menu, it can just be
    /// a text input menu, and if it's a text input menu you should be able to just type in the name
    /// of the variable." A boolean's operands, a sweep's spine, a pipe's curve, a frame's source, an
    /// attractor's target and a ramp's input are all one question, and a `ComboBox` apiece was
    /// eleven spellings of it.
    ///
    /// Generic over the key for the reason [`filtered_choices`] is: the three kinds of reference in
    /// the panel are a [`NodeId`], a [`CurveId`] and an `Attractor` (which is either), and a widget
    /// that knew which would be three widgets.
    ///
    /// **Typing resolves against [`RefChoice::address`] and nothing else**, which is where the
    /// block-versus-curve difference lives and why it is the caller's to state. A block's *label* is
    /// what it is — "Pipe" — and eight pipes on spokes are eight rows all labelled "Pipe", so
    /// matching that text would take the first of them and silently repoint one operand at a
    /// different pipe. Only a name somebody gave is unique (`set_body_name` refuses a duplicate), so
    /// only that is an address. A curve is the other way round: `unique_curve_name` makes its name
    /// unique on creation, so a curve's label *is* its address.
    ///
    /// **The candidate list is `choices` and nothing wider**, so the eligibility rule each caller
    /// already had — `operand_choices_in`'s "nothing that consumes this block", a frame's own
    /// sources — is still the only answer to what may go here. A field that searched the document
    /// for a name would be a second, looser rule, and would let somebody type their way into a cycle.
    ///
    /// **A block inside another may be chosen** (`block-ownership-spec.md` §6.2), by its name, from
    /// the list or by a drop. That gives it a second reader, so the settle the edit runs makes it
    /// a variable, and the next frame says so ("Sphere is now a variable — Union and Shell both use
    /// it"): the one way a row moves as a side effect, and it is moved by the person pointing a
    /// second block at it.
    ///
    /// Returns the reference to switch to, if this frame chose one.
    pub(crate) fn reference_field<K: Copy + PartialEq>(
        &mut self,
        ui: &mut egui::Ui,
        salt: impl std::hash::Hash,
        current: K,
        shown: &str,
        choices: &[RefChoice<K>],
        kind: RefKind,
        drops: DropsFrom<K>,
    ) -> Option<K> {
        self.reference_field_in(ui, salt, Some(current), shown, choices, kind, drops)
    }

    /// [`Self::reference_field`] for a field that may hold nothing yet.
    ///
    /// The only difference is that `current` is optional, and all it was ever used for is leaving
    /// the held reference out of the suggestions — so a field with nothing offers every choice. It
    /// answers with the reference chosen and never with "clear": where a field can be emptied
    /// there is a control for that beside it, not a convention about an empty box (an empty box
    /// already means *nothing typed yet*, which is why the two cannot share one).
    pub(crate) fn reference_field_in<K: Copy + PartialEq>(
        &mut self,
        ui: &mut egui::Ui,
        salt: impl std::hash::Hash,
        current: Option<K>,
        shown: &str,
        choices: &[RefChoice<K>],
        kind: RefKind,
        drops: DropsFrom<K>,
    ) -> Option<K> {
        let fid = ui.make_persistent_id(salt);
        let settled = shown.to_string();
        let mut text = match self.ref_edit.as_ref().filter(|(at, _)| *at == fid) {
            Some((_, buffer)) => buffer.clone(),
            None => settled.clone(),
        };

        let response = ui.add(
            egui::TextEdit::singleline(&mut text)
                .id(fid)
                .desired_width(118.0)
                .hint_text(kind.hint()),
        );
        let mut picked = None;

        // **The drop is offered on the field itself.** A `TextEdit` senses the pointer for its own
        // selection, and `dnd_release_payload` asks the same response whether the pointer was over
        // it when the button came up, so the two do not collide.
        //
        // Only where a dragged *block* can become this kind of reference. Nothing carries a
        // `CurveId` payload yet — the one drag source in the panel is a block row's grip — so a
        // curve field offers typing and the list, and says so in its hint rather than lighting up
        // for a drag it cannot accept.
        // **Both kinds of payload, because a curve row has a grip now too.** The only drag source in
        // the panel used to be a block row's, which is why a curve field could offer typing and
        // nothing else; curve rows carry a `CurveId` the same way a block row carries its id, so a
        // curve field behaves like a block field instead of explaining why it cannot. Asked for on
        // that ground: "this is consistent with blocks."
        //
        // **A drag of the wrong kind draws the refusing outline rather than nothing at all**, since a
        // field that ignores a drag looks broken in the one way a field that visibly refuses one does
        // not.
        let dragging_block = egui::DragAndDrop::has_payload_of_type::<DraggedBlock>(ui.ctx());
        let dragging_curve = egui::DragAndDrop::has_payload_of_type::<CurveId>(ui.ctx());
        if response.contains_pointer() && (dragging_block || dragging_curve) {
            let welcome = if dragging_block {
                drops.block.is_some_and(|f| {
                    egui::DragAndDrop::payload::<DraggedBlock>(ui.ctx())
                        .is_some_and(|d| choices.iter().any(|c| c.key == f(d.id)))
                })
            } else {
                drops.curve.is_some_and(|f| {
                    egui::DragAndDrop::payload::<CurveId>(ui.ctx())
                        .is_some_and(|id| choices.iter().any(|c| c.key == f(*id)))
                })
            };
            ui.painter().rect_stroke(
                response.rect.expand(1.0),
                2.0_f32,
                egui::Stroke::new(
                    2.0_f32,
                    if welcome {
                        ui.visuals().selection.bg_fill
                    } else {
                        ui.visuals().error_fg_color
                    },
                ),
                egui::StrokeKind::Outside,
            );
        }
        if let Some(f) = drops.block
            && let Some(dropped) = response.dnd_release_payload::<DraggedBlock>()
        {
            let what = self.body_label(dropped.id);
            return self.take_dropped_reference(f(dropped.id), choices, kind, &what);
        }
        if let Some(f) = drops.curve
            && let Some(dropped) = response.dnd_release_payload::<CurveId>()
        {
            let what = self
                .doc
                .curves
                .get(&dropped)
                .map_or_else(|| "That curve".to_string(), |c| c.name.clone());
            return self.take_dropped_reference(f(*dropped), choices, kind, &what);
        }

        if response.changed() {
            self.ref_edit = Some((fid, text.clone()));
        }

        // **The list is drawn while the field has focus, and before anything commits.** Clicking a
        // row in it takes the focus off the text box, so a version of this that only drew the list
        // in an `else` branch of "has the field finished" removed the list on the very frame the
        // click landed, and no suggestion could ever be chosen. It is also the only way to pick a
        // thing that has no address — an unnamed block — so it is not a convenience beside the
        // dropdown, it is what replaces it.
        let mut chosen_from_list = None;
        // **Drawn while the field has focus, or while the pointer is over the list itself.**
        //
        // The second half is what makes the list clickable at all, and without it only a drag
        // worked. A click is a press and a release, and pressing a row in the list takes the
        // focus off the text field -- so on the release frame `has_focus()` was already false,
        // the `Area` was not built, the row did not exist, and `clicked()` never fired. The
        // suggestion a person pressed simply did nothing, which is exactly how it was reported:
        // "I could click into the input field and try to pick the cube, but it wouldn't let me."
        //
        // Keeping the list while the pointer is inside its own rectangle spans those two frames
        // and nothing else: moving the pointer off it closes the list as before.
        //
        // **"Its own rectangle" means the list that is open, not one that was** (item 109). egui
        // remembers an `Area`'s rectangle after the `Area` stops being drawn, so asking only
        // whether the pointer was inside it answered yes for ever: pick a profile, the list
        // closes, move the pointer down over the rows where the list had been, and it drew itself
        // again — which kept its rectangle alive, which kept it open. Reported as "the dropdown
        // keeps popping back up". So the pointer may only *keep* a list open that was open on the
        // frame before, which is exactly the press-to-release span the rule was written for.
        let area_id = fid.with("near");
        let open_key = area_id.with("open last frame");
        // Read and cleared at once, so a frame that leaves early cannot leave a stale "open".
        let was_open = ui.ctx().data_mut(|d| {
            let was = d.get_temp::<bool>(open_key).unwrap_or(false);
            d.insert_temp(open_key, false);
            was
        });
        let pointer_in_last_rect = ui
            .ctx()
            .memory(|m| m.area_rect(area_id))
            .zip(ui.ctx().pointer_latest_pos())
            .is_some_and(|(rect, at)| rect.contains(at));
        let mut drawn = false;
        if show_reference_list(response.has_focus(), was_open, pointer_in_last_rect) {
            let typed = text.trim().to_lowercase();
            let showing = text.trim() != settled;
            let near = suggested(choices, &typed, showing);
            if !near.is_empty() {
                drawn = true;
                egui::Area::new(area_id)
                    .order(egui::Order::Foreground)
                    .fixed_pos(response.rect.left_bottom() + egui::vec2(0.0, 2.0))
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.set_max_width(response.rect.width().max(160.0));
                            for (key, label, refusal) in near {
                                // What the field already holds, said on the row rather than by
                                // the row's absence. Still clickable, which costs nothing and is
                                // what somebody re-picking it expects.
                                let held = Some(key) == current;
                                let label = if held {
                                    format!("{label}  \u{00b7}  held")
                                } else {
                                    label
                                };
                                match refusal {
                                    // Weak, disabled and saying why on hover. The click is
                                    // refused by the widget rather than by a check after it, so
                                    // there is no path where a refused row is taken.
                                    Some(why) => {
                                        ui.add_enabled(
                                            false,
                                            egui::Button::selectable(
                                                false,
                                                egui::RichText::new(label).weak(),
                                            ),
                                        )
                                        .on_disabled_hover_text(why);
                                    }
                                    None => {
                                        if ui.selectable_label(held, label).clicked() {
                                            chosen_from_list = Some(key);
                                        }
                                    }
                                }
                            }
                        });
                    });
            }
        }
        ui.ctx().data_mut(|d| {
            d.insert_temp(open_key, reference_list_open_next(drawn, chosen_from_list.is_some()))
        });
        if let Some(key) = chosen_from_list {
            self.ref_edit = None;
            return Some(key);
        }

        // Committed on Enter or on leaving the field, which are the two ways a person says they
        // have finished typing. Nothing is written while the text is half a name.
        let finished = response.lost_focus()
            || (response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
        if finished {
            let typed = text.trim().to_string();
            self.ref_edit = None;
            if !typed.is_empty() && typed != settled {
                match choices.iter().find(|c| {
                    c.address.as_deref().is_some_and(|a| a.eq_ignore_ascii_case(&typed))
                }) {
                    Some(c) => match c.refusal {
                        Some(why) => self.say(format!("\u{201c}{typed}\u{201d} cannot go here — {why}")),
                        None => picked = Some(c.key),
                    },
                    None => {
                        // A label that matched but had no address is the case worth spelling out:
                        // the row plainly reads "Pipe", so "nothing is called Pipe" would be a
                        // baffling answer. What is true is that a label describes and a name
                        // addresses.
                        let described = choices
                            .iter()
                            .any(|c| c.label.eq_ignore_ascii_case(&typed) && c.address.is_none());
                        self.unresolved_reference(&typed, kind, described);
                    }
                }
            }
        }
        picked
    }

    /// Says why a name typed into a reference field did not name anything usable.
    ///
    /// One place, so the answer does not depend on which field was typed into, and so it can tell
    /// the cases apart: a name nothing has, a *label* rather than a name, and — for a block — a name
    /// belonging to something that may not go here. The last is the one worth spelling out, because
    /// the block exists and is visible on screen, so "unknown" would read as a fault in the field
    /// rather than as the rule against cycles that it is.
    fn unresolved_reference(&mut self, typed: &str, kind: RefKind, described: bool) {
        if described {
            self.say(format!(
                "\u{201c}{typed}\u{201d} is what those blocks are, not what one of them is called \
                 — click the dim V on the row you mean to name it, or drag the row in"
            ));
            return;
        }
        if kind == RefKind::Block {
            let known = self
                .doc
                .node_meta
                .iter()
                .any(|(_, m)| m.name.as_deref().is_some_and(|n| n.eq_ignore_ascii_case(typed)));
            if known {
                self.say(format!(
                    "\u{201c}{typed}\u{201d} cannot go here — it is made from this block, directly \
                     or further down"
                ));
                return;
            }
        }
        self.say(format!("{} \u{201c}{typed}\u{201d}. {}", kind.nothing(), kind.how()));
    }


    /// What the blocks reading a row are called, in the order they were made: what a refused
    /// *put back inside* says (`block-ownership-spec.md` §5.2).
    ///
    /// Its readers as [`fieldform_scene::Document::block_readers`] counts them (spec I2) — a
    /// curve's in every part, as the count the guard went by is — each called what the part
    /// holding it calls it.
    fn reader_names(&self, what: Declarable) -> Vec<String> {
        let mut readers = self.doc.block_readers(what.block());
        readers.sort_by_key(|id| (self.doc.made_at(*id), id.raw()));
        readers
            .into_iter()
            .map(|id| self.body_label_in(self.graph_holding_or_active(id), id))
            .collect()
    }

    /// Makes a row a variable, whichever kind of row it is ([`Self::make_variable_of`]): the dim
    /// V's act.
    fn declare(&mut self, what: Declarable) {
        self.make_variable_of(what);
    }

    /// Puts a row back inside whatever reads it, keeping its name ([`Self::nest_back`]): the
    /// bright V's act.
    fn undeclare(&mut self, what: Declarable) {
        self.nest_back(what);
    }

    /// *Put back inside* (`notebook-spec.md` §6.6): a variable goes back inside whatever reads
    /// it, or stands on its own when nothing does, **keeping its name** — a block, a curve, or a
    /// profile row by the curve of its loop its V writes.
    ///
    /// The inverse of [`Self::make_variable_of`], and one undo step ([`put_back_inside`]). It used
    /// to clear the name, because the list was laid out by names and removing one was the whole
    /// move, so a name somebody typed was lost to the gesture — a profile row's as much as a
    /// block's — and a curve's flag was written off the history, where Ctrl+Z passed it by. No
    /// edge changes, and the block is consumed by exactly what consumed it a moment before.
    ///
    /// **Never refused for reader count, since 112.** Two readers used to refuse, there being no
    /// single block to go back inside; with no parent stored the block goes inside both and is one
    /// block in each — a shared instance (§7.3) — so the gesture always has an answer and the line
    /// names every block it went into.
    fn nest_back(&mut self, what: Declarable) {
        let name = self.declarable_label(what);
        let readers = self.reader_names(what);
        let cmd = put_back_inside(&self.doc, what.block());
        if !self.set_ownership(cmd) {
            return;
        }
        match self.doc.ownership(what.block()) {
            Ownership::Nested => {
                let inside = names_in_a_row(&readers);
                self.say(format!("\u{201c}{name}\u{201d} is part of {inside} again"));
            }
            _ => self.say(format!("\u{201c}{name}\u{201d} is not a variable any more")),
        }
    }

    /// `variable 'Sphere'`: the dim V on the row called that ([`Self::make_variable_of`]), so a
    /// script can say what a click does and get the same one undo step.
    ///
    /// **A refusal is warned, where the V's hover only says it**, because a typed line that did
    /// nothing must stop a script, and a success is said so it never does. A row that is a variable
    /// already, and a parameter, which always is one, are answered rather than refused: what the
    /// line asked for is already true.
    pub(crate) fn variable_from_console(&mut self, args: &Args) {
        let what = match self.row_from_console("variable", args) {
            Some(Ok(what)) => what,
            Some(Err(id)) => {
                let name = self.parameter_label(id);
                let line = console_keys::line_naming("inline", &name);
                self.say(format!(
                    "\u{201c}{name}\u{201d} is a parameter, and a parameter is a variable already \
                     \u{2014} `{line}` puts its number back"
                ));
                return;
            }
            None => return,
        };
        if self.doc.ownership(what.block()) == Ownership::Variable {
            self.say(format!("\u{201c}{}\u{201d} is a variable already", self.declarable_label(what)));
            return;
        }
        self.make_variable_of(what);
    }

    /// `inline 'Sphere'`: the bright V on the row called that ([`Self::nest_back`]), or on a
    /// parameter's row, whose number goes back into the one field holding it
    /// ([`Self::put_parameter_back`]). One undo step either way.
    ///
    /// Asked the guard the V asks before acting, so a refusal can be warned: the V's own path says
    /// it, which is right under a pointer and wrong in a script, where a line that moved nothing
    /// has to stop the run. A row that is not a variable has nothing to put back, and is told so.
    pub(crate) fn inline_from_console(&mut self, args: &Args) {
        let what = match self.row_from_console("inline", args) {
            Some(Ok(what)) => what,
            Some(Err(id)) => {
                if let Err(why) = self.number_put_back(id) {
                    self.warn(format!("\u{201c}{}\u{201d}: {why}", self.parameter_label(id)));
                    return;
                }
                self.put_parameter_back(id);
                return;
            }
            None => return,
        };
        let name = self.declarable_label(what);
        if self.doc.ownership(what.block()) != Ownership::Variable {
            self.say(format!("\u{201c}{name}\u{201d} is not a variable, so there is nothing to put back"));
            return;
        }
        self.nest_back(what);
    }

    /// The row a console line's `name` calls, as the V on it acts — `Ok` a block, a curve or a
    /// profile, `Err` a parameter — or `None` once a refusal is warned.
    ///
    /// **Found as the list shows it**: a block of the part on screen that has a row, so not a held
    /// loop's joint ([`crate::blocks::list::blocks_a_line_can_call`]), by its name or, unnamed, its
    /// label; a curve by its name; a profile by its name or "Profile N"; a parameter by its name
    /// ([`console_keys::row_called`] decides between them). The profiles are found again first,
    /// since a script runs every line in one frame and the line before may have closed one.
    ///
    /// **A curve whose V is its loop's is refused, naming the loop**, as its row menu greys the
    /// item ([`curve_v_elsewhere`]): making that curve a variable could be a click that moves
    /// nothing, and the profile row's V is the one that moves it.
    fn row_from_console(&mut self, word: &str, args: &Args) -> Option<Result<Declarable, ParamId>> {
        let wanted = args.get("name").map(|v| v.text()).unwrap_or_default().trim().to_string();
        let found = self.row_called_from_console(word, &wanted)?;
        let nesting = self.doc.nesting_in(self.part_graph());
        let what = match found {
            RowCalled::Block(id) => Declarable::Block(id),
            RowCalled::Parameter(id) => return Some(Err(id)),
            RowCalled::Curve(id) => match curve_v_elsewhere(&self.profiles, id, nesting.readers_of_curve(id)) {
                None => Declarable::Curve(id),
                Some(why) => {
                    let row = self.profiles.key_of(id).map(|key| self.profile_label(key)).unwrap_or_default();
                    let line = console_keys::line_naming(word, &row);
                    self.warn(format!(
                        "\u{201c}{wanted}\u{201d}: {why} From the console that row is `{line}`."
                    ));
                    return None;
                }
            },
            RowCalled::Profile(key) => {
                let members = self.profiles.members(key).map(|m| m.to_vec()).unwrap_or_default();
                let curve = profile_v_curve_in(&self.doc, &nesting, &members).unwrap_or(key);
                Declarable::Profile { key, curve }
            }
        };
        Some(Ok(what))
    }

    /// The one row in the list called `wanted`, or `None` once the refusal is warned: what
    /// [`Self::row_from_console`] and `rename` both find, so a line names a row one way whatever
    /// it then does to it.
    fn row_called_from_console(&mut self, word: &str, wanted: &str) -> Option<RowCalled> {
        let wanted = wanted.trim().to_string();
        self.refresh_profiles();
        let mut named = Vec::new();
        let mut labelled = Vec::new();
        let blocks = crate::blocks::list::blocks_a_line_can_call(&self.doc, self.active_part, &self.profiles);
        for id in blocks {
            match self.body_name(id) {
                Some(n) if n.eq_ignore_ascii_case(&wanted) => named.push(RowCalled::Block(id)),
                Some(_) => {}
                None if self.derived_label(id).eq_ignore_ascii_case(&wanted) => {
                    labelled.push(RowCalled::Block(id))
                }
                None => {}
            }
        }
        if let Some(id) = self.curve_named(&wanted) {
            named.push(RowCalled::Curve(id));
        }
        for key in self.profiles.loops.iter().map(|l| l[0]) {
            match self.doc.profile_names.get(&key) {
                Some(n) if n.eq_ignore_ascii_case(&wanted) => named.push(RowCalled::Profile(key)),
                Some(_) => {}
                None if self.profile_label(key).eq_ignore_ascii_case(&wanted) => {
                    labelled.push(RowCalled::Profile(key))
                }
                None => {}
            }
        }
        if let Some(id) = self.doc.parameter_named(&wanted) {
            named.push(RowCalled::Parameter(id));
        }
        match console_keys::row_called(word, &wanted, &named, &labelled) {
            Ok(found) => Some(found),
            Err(why) => {
                self.warn(why);
                None
            }
        }
    }

    /// `rename 'Sphere_0' to=ball`: a new name for the row called that -- a block, a curve, a
    /// profile or a parameter, found as `variable` finds one -- written through the path the row's
    /// own field takes, so a line is refused where the field would be and is the same one step.
    ///
    /// **The echo's spelling of a rename, which is why it exists** (`interaction-spec.md` §6): a
    /// name typed into a row changed what every later line calls that row, and with no line for it
    /// the record went on using a name the replay had never been given. A refusal is warned, not
    /// said, so a script stops on a rename that did not happen.
    ///
    /// **Without a name, the selected row's field opens for typing**: the menu, the palette and F2
    /// have no name to give until one is typed, and the name typed is written when it lands.
    pub(crate) fn rename_from_console(&mut self, args: &Args) {
        if args.get("name").is_none() && args.get("to").is_none() {
            self.begin_renaming_selection();
            return;
        }
        let Some(to) = args.get("to").map(|v| v.text()) else {
            self.warn("`rename` wants the new name too \u{2014} `rename 'Sphere_0' to=ball`, or `to=\"\"` to clear a block's");
            return;
        };
        let to = to.trim().to_string();
        let wanted = args.get("name").map(|v| v.text()).unwrap_or_default();
        let Some(found) = self.row_called_from_console("rename", &wanted) else { return };
        match found {
            RowCalled::Block(id) => {
                if let Some(why) = self.body_name_refusal(id, &to) {
                    self.warn(why);
                    return;
                }
                self.set_body_name(id, &to);
            }
            RowCalled::Curve(id) => self.rename_curve(id, &to),
            RowCalled::Profile(key) => self.rename_profile(key, &to),
            RowCalled::Parameter(id) => match self.doc.rename_parameter(id, &to) {
                Ok(()) => {
                    self.name_drafts.remove(&id);
                    self.mark_changed();
                }
                Err(why) => self.warn(why.to_string()),
            },
        }
    }

    /// Opens the selected row's name for typing, as a double-click on its name does: the block's,
    /// or the curve's where a curve is what is selected.
    fn begin_renaming_selection(&mut self) {
        let key = match (self.selected_body, self.selected_curve) {
            (Some(id), _) => RowKey::Node(id),
            (None, Some(curve)) => RowKey::Curve(curve),
            (None, None) => {
                self.warn("Select a row to rename, or name it: `rename 'Sphere_0' to=ball`");
                return;
            }
        };
        let current = match key {
            RowKey::Node(id) => self.body_name(id).unwrap_or_default().to_string(),
            RowKey::Curve(id) => self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default(),
            _ => String::new(),
        };
        self.renaming = Some((key, current));
    }

    /// The eye, on every block's row: a body's, a plane's or a point's, and a block's that
    /// something else consumes, named or not. A ramp's is shown disabled, saying why, since it
    /// has nothing to draw; no row is left blank.
    ///
    /// **A block inside another can be looked at.** The ingredient you are reusing is worth
    /// seeing while you place what you are cutting it from. It is off until asked for, because an
    /// ingredient is not a body, and drawing it moves no row.
    ///
    /// The two states are separate flags and read separately: a body draws unless `hidden`, an
    /// ingredient draws only if `shown`. See `NodeMeta::shown` for why one field could not carry
    /// both without changing meaning under the user. Which of them this eye means is not decided
    /// here: every eye reads through the rule `App::eye_flag` asks and writes through
    /// `set_visible`. The branches below differ only in what they say.
    fn visibility_toggle(&mut self, ui: &mut egui::Ui, id: NodeId, is_body: bool) {
        let size = egui::vec2(18.0, ui.spacing().interact_size.y);
        // **A plane and a point answer to `hidden` whether or not anything consumes them**,
        // because both are painted by the overlay rather than traced, and the overlay reads that
        // flag. Reported: a plane the ramp block made had no eye at all -- it is nobody's body,
        // so the old rule gave it a blank -- and the square it draws could not be got rid of.
        let overlay = self.draws_as_overlay(id);
        // Anything else that is a body and is consumed by something: the eye draws it beside the
        // bodies. **A name is no longer required.** It was, on the reasoning that a named block
        // is a thing in its own right -- but naming says a block is *reusable*, which is a
        // different question from whether it can be looked at, and an unnamed operand is exactly
        // the one somebody has lost track of. Reported as a capsule fed into an offset and not
        // seen again.
        let variable = !is_body
            && !overlay
            && self.part_graph().node(id).is_some_and(|n| n.is_body());
        // A ramp is the one block with nothing to draw: its value is a number, not a distance,
        // so its zero set is not a surface. Shown disabled rather than left blank, because a
        // control that is missing cannot say why it is missing.
        if !is_body && !variable && !overlay {
            ui.add_enabled(
                false,
                egui::Button::new(
                    egui::RichText::new("\u{1f441}").color(ui.visuals().weak_text_color()),
                )
                .frame(false),
            )
            .on_disabled_hover_text(
                "A ramp is a number rather than a shape, so there is nothing of it to draw. \
                 What it drives is drawn in the block that reads it.",
            );
            return;
        }
        // Through the eye's rule with the one answer it would otherwise walk the whole graph for,
        // whether anything reads this block, taken from the row's own `is_body`: a block the rule
        // asks that of is a body kind, and one of those is a root exactly when nothing reads it.
        // The list draws every row every frame, and asking the graph made each row a second pass
        // over it.
        let showing = self.doc.visible_by(id, eye_flag_by(self.part_graph(), id, || !is_body));
        if variable {
            let icon = egui::RichText::new("\u{1f441}").color(if showing {
                ui.visuals().strong_text_color()
            } else {
                ui.visuals().weak_text_color()
            });
            let response = ui.add_sized(size, egui::Button::new(icon).frame(false));
            if response.clicked() {
                self.toggle_visibility(id);
            }
            response.on_hover_text(if showing {
                "Stop drawing this block beside the bodies"
            } else {
                "Draw this block too — something else uses it, so it is not otherwise on \
                 screen. Its row stays where it is: drawing a block never moves it in the list."
            });
            return;
        }
        // Dimmed rather than absent when hidden: the eye is what you click to get the body
        // back, so it has to still be there and still be in the same place.
        let icon = egui::RichText::new("\u{1f441}").color(if showing {
            ui.visuals().strong_text_color()
        } else {
            ui.visuals().weak_text_color()
        });
        let response = ui.add_sized(size, egui::Button::new(icon).frame(false));
        if response.clicked() {
            self.toggle_visibility(id);
        }
        response.on_hover_text(if showing { "Hide (H)" } else { "Show (H)" });
    }

    /// The right-click menu on a row of the block list.
    ///
    /// **This replaced a delete cross on every row.** The cross was put there because deleting
    /// had been two clicks deep inside the expanded block and missing entirely from the rows most
    /// likely to be deleted — a real problem, fixed by a control that then sat permanently beside
    /// every row in the document, in red, one misclick from destroying work, to offer the single
    /// operation that already had a key. It also fixed the wrong half: what was hard was not
    /// *deleting*, it was that a row offered no actions at all, so copy, cut, paste and visibility
    /// were each somewhere else.
    ///
    /// So the row now offers all of them, in the place a tree row conventionally offers them, and
    /// deleting is one item in a list rather than a permanent red button. The items are the ones
    /// the Edit menu already runs, which is the point: right-clicking a row is a way of reaching
    /// them, not a second set of them.
    ///
    /// Everything here acts on the *selection*, because those operations do. The right-click
    /// selects the row before the menu opens, so the two agree.
    fn row_menu(&mut self, ui: &mut egui::Ui, row: Row) {
        // **A selected block counts as something, which it did not.** `select_block` clears both the
        // body and the curve selection, so Cut, Copy and Delete went grey the moment a profile row was
        // picked -- while `delete_selection` tests `selected_block` *first* and would have handled it.
        // Half of "cant delete the profile" was the menu refusing to offer what the handler supports.
        let something = self.selected_body.is_some()
            || self.selected_curve.is_some()
            || self.selected_block.is_some();
        if ui
            .add_enabled(something, egui::Button::new("Cut\tCtrl+X"))
            .clicked()
        {
            self.dispatch("edit.cut", &Args::new());
            ui.close();
        }
        if ui
            .add_enabled(something, egui::Button::new("Copy\tCtrl+C"))
            .clicked()
        {
            self.dispatch("edit.copy", &Args::new());
            ui.close();
        }
        if ui
            .add_enabled(self.clipboard.is_some(), egui::Button::new("Paste\tCtrl+V"))
            .clicked()
        {
            self.dispatch("edit.paste", &Args::new());
            ui.close();
        }
        // Right-clicking a row selects it first, which is what makes this act on the row the
        // menu was opened on rather than on whatever happened to be selected before.
        if ui
            .add_enabled(something, egui::Button::new("Duplicate"))
            .on_hover_text(
                "A second copy beside this one, sharing nothing with it. Leaves the clipboard \
                 alone.",
            )
            .clicked()
        {
            self.dispatch("edit.duplicate", &Args::new());
            ui.close();
        }

        ui.separator();

        // **Make variable and put back inside, from the menu** (`block-ownership-spec.md` §5).
        // Ticking is the dim V's act ([`Self::make_variable_of`]): a row of its own, and a
        // name derived from what it is if it has none, which the row itself is where to change.
        // Unticking is the bright V's ([`Self::nest_back`]): back inside the one block reading it,
        // keeping its name. Each is one undo step. It used to be the name that did both, ticking
        // naming and unticking clearing it, because the list was laid out by names.
        //
        // The untick is the bright V's act, so `variable_item` refuses it where the V refuses, by
        // asking the V's own guard, with the readers named only where they refuse: naming them
        // is a walk the menu need not make otherwise.
        //
        // A curve's row has the same item, since its V is the same act (§10). It was shown ticked
        // and unclickable, saying a curve is always available to other blocks — which is still
        // true of what a field offers, and stopped being an answer to where its row is drawn.
        //
        // **And it follows the V where the V is the loop's** (`curve_v_elsewhere`): greyed, with
        // the V's hover. Offered there, it made the curve heading a loop a variable — which moved
        // the whole profile row when that curve was the one placing it, from a menu on a row whose
        // V says it cannot.
        if let Row::Curve { id, .. } = row {
            // **The way back from a separated row, which otherwise has none.** Dismissing a grouping
            // stores every member in `ungrouped_curves`, and without this that is a trapdoor: the
            // curves could never be a profile again, because nothing would take them out of the set.
            // Offered on a curve's own row because once a row is separated there is no profile row
            // left to carry the gesture.
            if self.doc.ungrouped_curves.contains(&id)
                && ui
                    .button("Group as a profile again")
                    .on_hover_text(
                        "These curves were separated. Gather them back into one row \u{2014} the \
                         loop itself never changed.",
                    )
                    .clicked()
            {
                self.regroup_curve(id);
                ui.close();
            }
        }
        let (what, elsewhere) = match row {
            Row::Node { id, .. } => (Declarable::Block(id), None),
            Row::Curve { id, readers } => (
                Declarable::Curve(id),
                curve_v_elsewhere(&self.profiles, id, readers),
            ),
        };
        let is_variable = self.doc.ownership(what.block()) == Ownership::Variable;
        if let Some(make) = variable_item(ui, is_variable, elsewhere) {
            if make {
                self.make_variable_of(what);
            } else {
                self.nest_back(what);
            }
            ui.close();
        }

        ui.separator();

        // The same toggle the eye on the row runs. Offered here as well because the eye is 18
        // pixels wide and this is the menu you are already in.
        match row {
            Row::Node { id, is_body: true, .. } => {
                let showing = self.is_visible(id);
                if ui
                    .button(if showing { "Hide\tH" } else { "Show\tH" })
                    .clicked()
                {
                    self.toggle_visibility(id);
                    ui.close();
                }
            }
            Row::Curve { id, .. } => {
                let showing = !self.hidden_curves.contains(&id);
                if ui
                    .button(if showing { "Hide\tH" } else { "Show\tH" })
                    .clicked()
                {
                    if showing {
                        self.hidden_curves.insert(id);
                    } else {
                        self.hidden_curves.remove(&id);
                    }
                    ui.close();
                }
            }
            // **Everything else has one too, and it means what its eye means.** This used to
            // say that a node inside a body has no eye of its own -- which stopped being true
            // when a plane gained a square, a point gained a dot, and a consumed body gained
            // the option of being drawn beside the bodies. A menu that silently lacks the item
            // the row's own eye offers is two answers to one question.
            Row::Node { id, .. } => {
                if self.draws_as_overlay(id) {
                    let showing = self.is_visible(id);
                    if ui.button(if showing { "Hide	H" } else { "Show	H" }).clicked() {
                        self.toggle_visibility(id);
                        ui.close();
                    }
                } else if self.part_graph().node(id).is_some_and(|n| n.is_body()) {
                    let showing = self.is_visible(id);
                    if ui
                        .button(if showing { "Stop drawing" } else { "Draw this as well" })
                        .on_hover_text(
                            "Something else consumes this block, so it is not otherwise on \
                             screen. This draws it beside the bodies, and gives it a row of its \
                             own while it is drawn.",
                        )
                        .clicked()
                    {
                        self.toggle_visibility(id);
                        ui.close();
                    }
                }
            }
        }
        if ui.button("Frame	/").clicked() {
            self.dispatch("view.frame_selection", &Args::new());
            ui.close();
        }

        ui.separator();

        if ui
            .add_enabled(something, egui::Button::new("Delete\tDel"))
            .clicked()
        {
            self.dispatch("edit.delete_body", &Args::new());
            ui.close();
        }
    }

    /// The foot of an expanded block, which offers nothing now. It held a block's delete, which is
    /// in the row's right-click menu (`row_menu`) beside the eye on the row that hides it, so
    /// neither needs the block opened to reach it.
    pub(crate) fn body_actions(&mut self, _ui: &mut egui::Ui, _id: NodeId) {}
}

#[cfg(test)]
mod reference_list_tests {
    use super::{reference_list_open_next, show_reference_list};

    /// **A list that was closed is never reopened by the pointer alone** (item 109).
    ///
    /// egui keeps an `Area`'s rectangle after the `Area` stops being drawn, so a rule that only
    /// asked whether the pointer was inside it reopened the list every time the pointer crossed
    /// the strip where it had been — "the dropdown keeps popping back up". Every combination,
    /// and the one that was wrong is the third assertion.
    #[test]
    fn a_closed_reference_list_is_not_reopened_by_the_pointer() {
        // Focus always shows it, wherever the pointer is.
        for was_open in [false, true] {
            for inside in [false, true] {
                assert!(show_reference_list(true, was_open, inside));
            }
        }
        // An open list survives the press-to-release of a click on one of its rows.
        assert!(show_reference_list(false, true, true));
        // The regression: closed, unfocused, pointer where the list used to be — stays shut.
        assert!(!show_reference_list(false, false, true));
        // And the pointer leaving closes it, as it always did.
        assert!(!show_reference_list(false, true, false));
    }

    /// A choice closes the list, even though the pointer is still on the row it clicked.
    #[test]
    fn choosing_from_a_reference_list_closes_it() {
        assert!(!reference_list_open_next(true, true), "chosen: closed next frame");
        assert!(reference_list_open_next(true, false), "drawn, nothing chosen: still open");
        assert!(!reference_list_open_next(false, false), "not drawn: closed");
        // Closed after a choice, then the pointer over the old rows: shut (the reported sequence).
        let next = reference_list_open_next(true, true);
        assert!(!show_reference_list(false, next, true));
    }
}
