// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Measuring a body: volume and area, thin walls, a lattice's estimate, and the section view.

use crate::*;

/// Why a statement about the part cannot be made yet, or `None` when it can.
///
/// **One rule for the three places that make one** — *Measure*, wall thickness and the promise
/// check. It was pasted into each, which is the arrangement that has already gone wrong once in
/// this change: `drawn_bodies` grew a term its own doc said it shared with
/// `display_graph_with_stand_ins`, while that predicate still had three. Three copies of a
/// refusal drift the same way, so this is the one copy and they ask it.
///
/// **Refused rather than answered, which is the owner's ruling and not a shortcut**
/// (`notebook-spec.md` §7.2). A body with an empty slot is *pruned* from what is drawn, so the
/// display graph these three would otherwise sweep is a part with a body silently missing.
/// Measuring that answers confidently about something nobody has: wall thickness is "a statement
/// about the part that someone is about to make", and a number that gets copied into a quote has
/// to be about the whole of it. Hiding a body is something somebody chose; a hole is not.
///
/// Free of the `App`, which supplies what to call a block, so a test can hold the rule — the
/// arrangement `ownership_news` has for the same reason.
pub(crate) fn no_statement_while_a_slot_is_empty(
    graph: &fieldform_scene::Graph,
    label: impl Fn(fieldform_scene::NodeId) -> String,
    doing: &str,
) -> Option<String> {
    let (node, slot) = graph.first_empty_operand()?;
    Some(format!(
        "Cannot {doing} while a block is waiting to be told what to read: \u{201c}{}\u{201d}'s \
         slot {slot} is empty, so the part would be read with that body missing. Put a block in \
         it, or take out what reads it.",
        label(node),
    ))
}

impl App {
    /// Reports the selected body's volume, surface area and balance point.
    ///
    /// **Measured, not looked up.** There is no stored volume anywhere in this kernel — a body
    /// is a function, so the only way to know how much of it there is, is to go and find out.
    /// That is why this is a command rather than a number sitting in the panel: it costs a mesh
    /// of the field, which is 24 ms at the resolution used here and not something to spend
    /// sixty times a second on a readout nobody asked for.
    ///
    /// Approximate, and it says so. The mesh converges on the true surface as the square of the
    /// cell size — 0.012% on a sphere at this resolution — but a number that gets copied into a
    /// quote deserves to carry its own accuracy rather than to look exact.
    pub(crate) fn measure_selection(&mut self) {
        if let Some(why) =
            no_statement_while_a_slot_is_empty(self.part_graph(), |id| self.body_label(id), "measure")
        {
            self.warn(why);
            return;
        }
        let bodies = self.doc.listed_bodies(self.part_graph());
        let target = self.selected_body.filter(|id| bodies.contains(id));
        let what = match target {
            Some(id) => self.body_label(id),
            None => "the part".to_string(),
        };

        // The chosen body alone, or everything when nothing is chosen — the same rule framing
        // follows, so "what is selected" means one thing across the application.
        // The chosen body alone is the part with its siblings taken out, rather than a
        // separately built graph: a body *is* a root of this graph, so removing the others is
        // the operation that says "only this one" without describing it a second way.
        let mut graph = self.display_graph();
        if let Some(id) = target {
            for other in graph.roots() {
                if other != id {
                    graph.remove_body(other);
                }
            }
        }
        let field = GraphField::compile_graph(&self.doc, &graph, 0.05).ok();
        let Some(field) = field else {
            self.warn(match self.compile_error() {
                Some(why) => format!("Nothing to measure: {why}"),
                None => "Nothing to measure".to_string(),
            });
            return;
        };

        // Grown, so a body that touches its own bounding box is not measured as though the box
        // were a wall — the mesher closes the surface where it leaves the region.
        let Some((lo, hi)) = self.measure_bounds(target) else {
            self.warn(format!("{what} has no bounds to measure within"));
            return;
        };

        let started = std::time::Instant::now();
        let measured = fieldform_field::MassProperties::of(&field, (lo, hi), MEASURE_CELLS);
        let secs = started.elapsed().as_secs_f64();

        match measured {
            Some(m) => {
                let c = m.centre_of_mass;
                self.say(format!(
                    "{what}: {:.1} mm3, {:.1} mm2 of surface, balances at \
                     ({:.2}, {:.2}, {:.2}) mm — measured in {secs:.2}s, within about \
                     0.02%",
                    m.volume_mm3, m.surface_area_mm2, c.x, c.y, c.z,
                ));
            }
            None => self.warn(format!("{what} has no surface inside its own bounds")),
        }
    }

    /// `estimate`: what the lattice panel's readout says, for every lattice inside the selected
    /// body or `node=`, and with `stiffness=5` what each source would need (item 94, ruling 18).
    ///
    /// **Reads and never edits** -- `lattice_readout` takes the document by reference -- so its
    /// cautions are said, not warned: a warning stops a script, and a reading outside a source's
    /// range is not a refusal. What is refused is only what cannot be asked: no body, a key that
    /// is not a percentage, a body with no lattice in it.
    pub(crate) fn estimate_from_console(&mut self, args: &Args) {
        let named = match console_keys::block_named("estimate", "node", args, &|t| self.lookup(t)) {
            Ok(named) => named,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let Some(start) = named.or(self.selected_body) else {
            self.warn("`estimate` reads the selected body — select one, or say `estimate node=<name>`");
            return;
        };
        let target = match args.get("stiffness") {
            None => None,
            Some(v) => match v.as_f64() {
                Some(percent) if percent.is_finite() && percent > 0.0 && percent <= 100.0 => {
                    Some(percent / 100.0)
                }
                _ => {
                    self.warn(
                        "`estimate stiffness=` is a percentage of the solid's modulus, above \
                         nought and at most 100 — like stiffness=5",
                    );
                    return;
                }
            },
        };
        let report = lattice_readout::estimate_report(
            &self.doc,
            self.part_graph(),
            start,
            &|id| self.body_label(id),
            target,
        );
        match report {
            Ok(lines) => {
                for line in lines {
                    self.say(line);
                }
            }
            Err(why) => self.warn(why),
        }
    }


    /// Reports where the part is thinner than `threshold_mm`, and how thin it gets.
    ///
    /// **The answer a knife edge needs.** Where a lattice wall is cut by the body it infills,
    /// the solid tapers to zero and the mesh there frays — but no triangulation of a wall
    /// thinner than a printer's line width is printable, however tidy. So the useful question is
    /// not how to mesh it but whether it is there, and where.
    ///
    /// Meshed at the export tolerance rather than a display one: this is a statement about the
    /// part that someone is about to make, and the sampling should be the sampling that
    /// produced the file they are about to send.
    pub(crate) fn measure_thickness(&mut self, threshold_mm: f64) {
        if let Some(why) =
            no_statement_while_a_slot_is_empty(self.part_graph(), |id| self.body_label(id), "measure")
        {
            self.warn(why);
            return;
        }
        let graph = self.display_graph();
        if graph.roots().is_empty() {
            self.warn("Nothing to measure");
            return;
        }
        // At the detail the next export would use, scaled to what is on screen.
        let bounds = self.pick_bounds();
        let extent = bounds.1 - bounds.0;
        let tolerance = self.export_detail.tolerance(extent.x.max(extent.y).max(extent.z));
        let field = match GraphField::compile_graph(
            &self.doc,
            &graph,
            (tolerance.min_cell_mm * 0.5).max(1e-4),
        ) {
            Ok(f) => f,
            Err(e) => {
                let why = self.explain_compile(&e);
                self.warn(format!("Nothing to measure: {why}"));
                return;
            }
        };

        let started = std::time::Instant::now();
        let mesh = fieldform_field::mesh_field_adaptive(&field, bounds, &tolerance);
        let report = fieldform_field::thinness(&field, &mesh, threshold_mm);
        let secs = started.elapsed().as_secs_f64();

        let Some(report) = report else {
            self.warn("No surface to measure");
            return;
        };
        let at = report.thinnest_at;
        self.say(if report.below == 0 {
            format!(
                "Nothing thinner than {threshold_mm} mm. The thinnest place measures \
                 {:.3} mm, at ({:.1}, {:.1}, {:.1}) — {} points in {secs:.1}s",
                report.thinnest_mm, at.x, at.y, at.z, report.sampled,
            )
        } else {
            format!(
                "{:.1}% of the surface is thinner than {threshold_mm} mm. The thinnest \
                 measures {:.3} mm, at ({:.1}, {:.1}, {:.1}) — {} of {} points in \
                 {secs:.1}s",
                report.fraction_below() * 100.0,
                report.thinnest_mm,
                at.x,
                at.y,
                at.z,
                report.below,
                report.sampled,
            )
        });
    }

    /// A box to measure within: the target's own extent with room around it.
    ///
    /// An unbounded body — one unioned with a half-space or a bare lattice — has no volume to
    /// report, and saying so beats measuring whatever happens to fall inside an arbitrary box.
    fn measure_bounds(&self, target: Option<NodeId>) -> Option<(Point3<f64>, Point3<f64>)> {
        let extent = match target {
            Some(id) => self.body_extent(id),
            None => {
                let part = self.part_graph();
                let consumed = part.consumed();
                part.roots()
                    .into_iter()
                    .filter(|id| is_visible_among(&self.doc, part, &consumed, *id))
                    .map(|id| self.body_extent(id))
                    .fold(fieldform_field::Extent::Empty, |acc, e| acc.union(e))
            }
        };
        let (lo, hi) = extent.as_box()?;
        let margin = ((hi - lo).norm() * 0.05).max(0.5);
        Some((
            lo - Vector3::repeat(margin),
            hi + Vector3::repeat(margin),
        ))
    }

    /// The sections that are on, in the order made.
    pub(crate) fn sections_on(&self) -> impl Iterator<Item = usize> + '_ {
        self.sections.iter().enumerate().filter(|(_, s)| s.on).map(|(i, _)| i)
    }

    /// The planes the shader cuts with: one per section that is on, none when none is. What is
    /// past *any* of them is cut away — the kept region is the intersection of their kept halves,
    /// so two make a corner cutaway.
    ///
    /// **Several at once, because the field allows it** (the owner, 2026-10-24, asking whether it
    /// did). The shader's cut is `max(field, plane)`, and over several planes it is the same `max`,
    /// so there was nothing to build but the array. `fieldform_gpu::MAX_SECTIONS` is its length,
    /// and `set_section_on` refuses the one past it by name rather than cutting with a plane the
    /// shader never sees.
    pub(crate) fn section_half_spaces(&self) -> Vec<(Vector3<f64>, f64)> {
        self.sections_on().filter_map(|i| self.sections[i].half_space()).collect()
    }

    /// The section whose handles are showing: selected *and* on. A selected section that is off
    /// has nothing drawn to move, so it offers nothing — which is what keeps the manipulator off
    /// a plane that is not there.
    pub(crate) fn selected_section(&self) -> Option<Section> {
        self.section_selected.and_then(|i| self.sections.get(i)).filter(|s| s.on).copied()
    }

    /// The section a command with no row of its own acts on: the selected one, else the last made.
    fn current_section(&self) -> Option<usize> {
        self.section_selected
            .filter(|&i| i < self.sections.len())
            .or_else(|| self.sections.len().checked_sub(1))
    }

    /// Where a new section goes: the middle of **the selected body** if there is one, and of the
    /// whole scene otherwise. Both are the same rule applied to different scopes: a section plane
    /// at the world origin cuts nothing when the part is somewhere else, and a tool that appears to
    /// do nothing reads as broken. Selecting a body first is how you say which thing you wanted to
    /// look inside.
    ///
    /// Read once, when the section is made — never on a later selection. Re-centring while the
    /// cut is live would snatch the plane back from wherever it had been dragged to, which is the
    /// same tool appearing to do something nobody asked for.
    fn section_origin(&self) -> Option<Point3<f64>> {
        let selected = self
            .selected_body
            .filter(|id| self.part_graph().roots().contains(id))
            .and_then(|id| self.body_extent(id).as_box())
            .map(|(lo, hi)| nalgebra::center(&lo, &hi));
        selected.or_else(|| self.scene_bounds().map(|(centre, _)| centre.cast::<f64>()))
    }

    /// Makes another section where the selection is, switches it on beside the others, and aims
    /// the manipulator at it. `view.section_new`; and what `view.section` does when there is none
    /// yet, so the first X does what it always did.
    pub(crate) fn add_section(&mut self) {
        self.sections_made += 1;
        let mut section = Section { number: self.sections_made, ..Section::default() };
        if let Some(origin) = self.section_origin() {
            section.origin = origin;
        }
        self.sections.push(section);
        let index = self.sections.len() - 1;
        // Selected, because making it is the act of reaching for it — the same reason creating a
        // body selects it. Selected even when the switch is refused, so the row it made is the
        // row that is highlighted and the refusal reads against it.
        let on = self.set_section_on(index, true);
        self.claim_section(index);
        if on {
            self.say(format!("Section {} on — drag the handles to move the cut", self.sections_made));
        }
    }

    /// Switches one section on or off, and says whether it did.
    ///
    /// The others are left as they are: any number may be on at once (see `section_half_spaces`),
    /// up to the shader's array. The one past that is refused by name rather than switched on and
    /// not cut with, which would be a row saying "on" about a plane nothing reads.
    pub(crate) fn set_section_on(&mut self, index: usize, on: bool) -> bool {
        let Some(section) = self.sections.get(index).copied() else { return false };
        if on && !section.on && self.sections_on().count() >= fieldform_gpu::MAX_SECTIONS {
            self.warn(format!(
                "{} sections are on already, which is as many as the view cuts with — switch one \
                 off first",
                fieldform_gpu::MAX_SECTIONS
            ));
            return false;
        }
        self.sections[index].on = on;
        self.traced.invalidate(traced::Quality::Final);
        true
    }

    /// The claim on the manipulator, with nothing else left selected to compete for it.
    fn claim_section(&mut self, index: usize) {
        self.section_selected = Some(index);
        self.selected_body = None;
        self.selected_curve = None;
        self.selected_block = None;
        self.clear_point_selection();
        self.selection.clear();
    }

    /// `view.section` — X, the View menu, `section`: the current section on or off, or the first
    /// one made.
    ///
    /// The current section is the selected one, else the last made; with none made at all this
    /// makes one, so a document's first X is a cut and not a row to find. Switching on also takes
    /// the handles, because switching on is the act of reaching for it; switching off leaves the
    /// row selected, with nothing drawn to move.
    pub(crate) fn toggle_section(&mut self) {
        let Some(index) = self.current_section() else {
            self.add_section();
            return;
        };
        let on = !self.sections[index].on;
        if !self.set_section_on(index, on) {
            return;
        }
        let number = self.sections[index].number;
        if on {
            self.claim_section(index);
            self.say(format!("Section {number} on — drag the handles to move the cut"));
        } else {
            self.say(format!("Section {number} off"));
        }
    }

    /// `view.section_flip`: the selected section faces the other way if it is on, else the last
    /// one that is. The command refuses when none is on (`section_on`), so the guard here is only
    /// against a row's button racing it.
    pub(crate) fn flip_section(&mut self) {
        let selected = self.section_selected.filter(|&i| self.sections.get(i).is_some_and(|s| s.on));
        let Some(index) = selected.or_else(|| self.sections_on().last()) else { return };
        self.flip_section_at(index);
    }

    /// A row's Flip: *its* section, whichever is selected.
    fn flip_section_at(&mut self, index: usize) {
        let Some(section) = self.sections.get_mut(index) else { return };
        section.flipped = !section.flipped;
        let number = section.number;
        self.traced.invalidate(traced::Quality::Final);
        self.say(format!("Section {number} flipped"));
    }

    /// Aims the manipulator at a section from its row, switching it on if it is off.
    ///
    /// Clicking the row of a cut that is off switches it on rather than selecting nothing
    /// visible: a selected plane with no handles and no cut would be a row that highlights and
    /// does nothing. Clicking the row of a cut that is on only takes the handles back, from a body
    /// or a face picked since, and moves nothing, for the reason `section_origin` gives against
    /// re-centring. The other sections are left as they are.
    pub(crate) fn select_section(&mut self, index: usize) {
        if index >= self.sections.len() {
            return;
        }
        if !self.sections[index].on && self.set_section_on(index, true) {
            self.say(format!("Section {} on", self.sections[index].number));
        }
        self.claim_section(index);
    }

    /// Takes a section out of the list: the row's menu, or Delete with the row selected. Nothing
    /// to undo: it was never in the document, and the others keep their numbers, so the rows that
    /// remain read as they did.
    pub(crate) fn remove_section(&mut self, index: usize) {
        if index >= self.sections.len() {
            return;
        }
        let removed = self.sections.remove(index);
        self.section_selected = match self.section_selected {
            Some(i) if i == index => None,
            Some(i) if i > index => Some(i - 1),
            other => other,
        };
        if removed.on {
            self.traced.invalidate(traced::Quality::Final);
        }
        self.say(format!("Section {} removed", removed.number));
    }

    /// The sections' rows in the block list, one each and none until one is made.
    ///
    /// Framed like a block's row and drawn after them, because the list is where the owner looks
    /// for everything the part is seen through — but with no grip, no eye and no V, and a number
    /// for a name, since a section is not in the document and none of those would mean anything
    /// (see [`Section`]). A row's switch and Flip act on *its* section directly rather than
    /// through `view.section`, which acts on the current one: with several rows the two are not
    /// the same section, and a switch that switched some other row would be the bug the rows exist
    /// to make impossible. Removing one is in the row's menu and on the Delete key, not a button on
    /// the row (the owner, 2026-10-24: a ✕ in the notebook "is not good"). The row is highlighted
    /// exactly when the handles are its.
    pub(crate) fn section_rows(&mut self, ui: &mut egui::Ui) {
        let mut index = 0;
        while index < self.sections.len() {
            let count = self.sections.len();
            self.section_row(ui, index);
            // A row that removed itself shifts the ones below it; they are drawn next frame
            // rather than under the wrong index now.
            if self.sections.len() < count {
                break;
            }
            index += 1;
        }
    }

    fn section_row(&mut self, ui: &mut egui::Ui, index: usize) {
        let Some(section) = self.sections.get(index).copied() else { return };
        let frame = egui::Frame::new()
            .stroke(egui::Stroke::new(
                1.0_f32,
                ui.visuals().widgets.noninteractive.bg_stroke.color,
            ))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(4, 3));
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.push_id(("section_row", section.number), |ui| {
                ui.horizontal(|ui| {
                    let mut on = section.on;
                    if ui
                        .checkbox(&mut on, "")
                        .on_hover_text(
                            "Cuts away everything past this plane so you can see inside. Changes \
                             the view only — the part is not modified, and it stays on while you \
                             work. With several on, what is past any of them is cut away.",
                        )
                        .changed()
                        && self.set_section_on(index, on)
                        && on
                    {
                        self.claim_section(index);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(section.on, egui::Button::new("Flip").small())
                            .on_hover_text("Keeps the other half")
                            .clicked()
                        {
                            self.flip_section_at(index);
                        }
                        // The rest of the row is the row: drawn as `list_row` draws a block's, so
                        // the highlight reads the same down the list.
                        let header = format!(
                            "Section {}  \u{00b7}  view, {}",
                            section.number,
                            if section.on { "on" } else { "off" }
                        );
                        let selected = section.on && self.section_selected == Some(index);
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
                            ui.painter().with_clip_rect(rect).text(
                                rect.left_center() + egui::vec2(6.0, 0.0),
                                egui::Align2::LEFT_CENTER,
                                header,
                                egui::TextStyle::Button.resolve(ui.style()),
                                visuals.text_color(),
                            );
                        }
                        // Right-clicking selects first, as a block's row does, so Delete on the
                        // key and Delete in the menu agree about which section goes.
                        if row.secondary_clicked() {
                            self.claim_section(index);
                        }
                        row.context_menu(|ui| {
                            if ui.button("Delete\tDel").clicked() {
                                self.remove_section(index);
                                ui.close();
                            }
                        });
                        if row
                            .on_hover_text(if section.on {
                                "Click to take the handles back to this section"
                            } else {
                                "Click to switch this section on"
                            })
                            .clicked()
                        {
                            self.select_section(index);
                        }
                    });
                });
            });
        });
    }
}
