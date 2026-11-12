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
        let field = GraphField::compile_graph(&self.doc, &graph, crate::MEASURE_TOLERANCE_MM).ok();
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
    pub(crate) fn measure_bounds(&self, target: Option<NodeId>) -> Option<(Point3<f64>, Point3<f64>)> {
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

    /// The active part's section cuts, each read to where its plane is now, in the order made.
    ///
    /// Indices below are into this list. A section on another part waits in the document for that
    /// part to be shown, and is neither listed nor cut with meanwhile.
    pub(crate) fn sections(&self) -> Vec<Section> {
        self.doc
            .sections
            .iter()
            .filter(|c| c.part == self.active_part)
            .map(|c| self.resolve_section(c))
            .collect()
    }

    /// Where a cut's plane is: its place and its turn read as a plane block's are -- the same two
    /// functions, through its part's graph -- so a cut can sit on a point or face the way a plane
    /// faces, and follows when that moves.
    pub(crate) fn resolve_section(&self, cut: &fieldform_scene::SectionCut) -> Section {
        let graph = self.doc.parts.get(&cut.part).map(|p| &p.graph).unwrap_or_else(|| self.part_graph());
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        let origin = fieldform_field::resolve_place(&self.doc, graph, &cut.origin, &numbers)
            .unwrap_or_else(|_| Point3::origin());
        let rotation = fieldform_field::turn_rotation(&self.doc, graph, &cut.turn, &numbers);
        Section { id: cut.id, number: cut.number, on: cut.on, origin, rotation, flipped: cut.flipped }
    }

    /// The indices of the sections that are on, in the order made.
    pub(crate) fn sections_on(&self) -> Vec<usize> {
        self.sections().iter().enumerate().filter(|(_, s)| s.on).map(|(i, _)| i).collect()
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
        self.sections().iter().filter_map(Section::half_space).collect()
    }

    /// The section whose handles are showing: selected *and* on. A selected section that is off
    /// has nothing drawn to move, so it offers nothing — which is what keeps the manipulator off
    /// a plane that is not there.
    pub(crate) fn selected_section(&self) -> Option<Section> {
        let id = self.section_selected?;
        self.sections().into_iter().find(|s| s.id == id && s.on)
    }

    /// The section a command with no row of its own acts on: the selected one, else the last made.
    fn current_section(&self) -> Option<usize> {
        let sections = self.sections();
        self.section_selected
            .and_then(|id| sections.iter().position(|s| s.id == id))
            .or_else(|| sections.len().checked_sub(1))
    }

    /// The record of the active part's section at `index`.
    fn section_record(&self, index: usize) -> Option<fieldform_scene::SectionCut> {
        let id = self.sections().get(index)?.id;
        self.doc.sections.iter().find(|c| c.id == id).copied()
    }

    /// Sets every section cut at once, undoably: one entry per edit, or one per gesture on the
    /// same section within the window, which is what lets typing a number into its rows land on
    /// the stack once. The arrangement `set_images` keeps for the Save Image blocks.
    pub(crate) fn set_sections(
        &mut self,
        sections: Vec<fieldform_scene::SectionCut>,
        gesture: Option<fieldform_scene::SectionCutId>,
    ) {
        let now = self.doc.sections.clone();
        if now == sections {
            return;
        }
        let continues = gesture.is_some()
            && self.stack.peek_undo() == Some("sections.set")
            && self
                .section_gesture
                .as_ref()
                .is_some_and(|(s, at, _)| Some(*s) == gesture && at.elapsed() < GESTURE_WINDOW);
        let before = match (&self.section_gesture, continues) {
            (Some((_, _, before)), true) => before.clone(),
            _ => now,
        };
        self.section_gesture = gesture.map(|s| (s, std::time::Instant::now(), before.clone()));
        self.gesture = None;
        let cmd: Box<dyn fieldform_ops::Command> = Box::new(fieldform_ops::SetSections::new(before, sections));
        let done = if continues {
            self.stack.replace_top(&mut self.doc, cmd)
        } else {
            self.stack.execute(&mut self.doc, cmd)
        };
        match done {
            Ok(()) => self.mark_changed(),
            Err(e) => self.warn(e.to_string()),
        }
        self.traced.invalidate(traced::Quality::Final);
    }

    /// One section changed by `edit`, as one undoable step (or part of a gesture's).
    fn edit_section(
        &mut self,
        index: usize,
        gesture: bool,
        edit: impl FnOnce(&mut fieldform_scene::SectionCut),
    ) {
        let Some(record) = self.section_record(index) else { return };
        let mut changed = record;
        edit(&mut changed);
        let sections =
            self.doc.sections.iter().map(|c| if c.id == record.id { changed } else { *c }).collect();
        self.set_sections(sections, gesture.then_some(record.id));
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
    /// yet, so the first X does what it always did. One undo step, which takes it away again.
    ///
    /// `origin`, `turn` and `flip` are the console's: where it passes through, which way it faces
    /// and which half it keeps, as a plane block takes them.
    pub(crate) fn add_section_with(
        &mut self,
        origin: Option<Point3<f64>>,
        turn: Option<[f64; 3]>,
        flipped: bool,
    ) {
        let number = self.doc.sections.iter().map(|c| c.number).max().unwrap_or(0) + 1;
        let at = origin.or_else(|| self.section_origin()).unwrap_or_else(Point3::origin);
        let mut cut = fieldform_scene::SectionCut {
            id: fieldform_scene::SectionCutId::new(),
            number,
            part: self.active_part,
            on: true,
            origin: fieldform_scene::Place::At([
                ParamRef::Literal(at.x),
                ParamRef::Literal(at.y),
                ParamRef::Literal(at.z),
            ]),
            turn: match turn {
                Some([x, y, z]) => fieldform_scene::Turn::By([ParamRef::Literal(x), ParamRef::Literal(y), ParamRef::Literal(z)]),
                None => fieldform_scene::Turn::default(),
            },
            flipped,
        };
        // Made off when the view is cutting with as many as it can, and said so: the row is made,
        // which is what was asked for, and the refusal reads against it.
        let full = self.sections_on().len() >= fieldform_gpu::MAX_SECTIONS;
        if full {
            cut.on = false;
        }
        let mut sections = self.doc.sections.clone();
        sections.push(cut);
        self.section_gesture = None;
        self.set_sections(sections, None);
        // Selected, because making it is the act of reaching for it — the same reason creating a
        // body selects it.
        self.claim_section_id(cut.id);
        if full {
            self.warn(format!(
                "Section {number} made off: {} sections are on already, which is as many as the \
                 view cuts with — switch one off first",
                fieldform_gpu::MAX_SECTIONS
            ));
        } else {
            self.say(format!("Section {number} on — drag the handles to move the cut, or open its row to type where"));
        }
    }

    /// `view.section_new` with nothing said: where the selection is, facing up.
    pub(crate) fn add_section(&mut self) {
        self.add_section_with(None, None, false);
    }

    /// `addsection origin=x,y,z turn=a,b,c flip=true`: a section from the console, with every
    /// number its row takes (`console_keys::VERBS`). A place or a facing taken from another block
    /// has no console spelling, as for every block (the recorded gap there).
    pub(crate) fn section_new_command(&mut self, args: &Args) {
        let triple = |key: &str| -> Result<Option<[f64; 3]>, String> {
            let Some(v) = args.get(key) else { return Ok(None) };
            let parts: Vec<f64> = v
                .text()
                .split(',')
                .map(|p| p.trim().parse::<f64>())
                .collect::<Result<_, _>>()
                .map_err(|_| format!("`{key}=` wants three numbers, such as `{key}=0,0,10`"))?;
            match parts.as_slice() {
                [x, y, z] => Ok(Some([*x, *y, *z])),
                _ => Err(format!("`{key}=` wants three numbers, such as `{key}=0,0,10`")),
            }
        };
        let (origin, turn) = match (triple("origin"), triple("turn")) {
            (Ok(o), Ok(t)) => (o, t),
            (Err(why), _) | (_, Err(why)) => {
                self.warn(why);
                return;
            }
        };
        let flipped = crate::console_keys::flag(args, "flip");
        self.add_section_with(origin.map(|[x, y, z]| Point3::new(x, y, z)), turn, flipped);
    }

    /// Switches one section on or off, and says whether it did.
    ///
    /// The others are left as they are: any number may be on at once (see `section_half_spaces`),
    /// up to the shader's array. The one past that is refused by name rather than switched on and
    /// not cut with, which would be a row saying "on" about a plane nothing reads.
    pub(crate) fn set_section_on(&mut self, index: usize, on: bool) -> bool {
        let Some(section) = self.sections().get(index).copied() else { return false };
        if on && !section.on && self.sections_on().len() >= fieldform_gpu::MAX_SECTIONS {
            self.warn(format!(
                "{} sections are on already, which is as many as the view cuts with — switch one \
                 off first",
                fieldform_gpu::MAX_SECTIONS
            ));
            return false;
        }
        self.edit_section(index, false, |c| c.on = on);
        true
    }

    /// The claim on the manipulator, with nothing else left selected to compete for it.
    fn claim_section(&mut self, index: usize) {
        if let Some(id) = self.sections().get(index).map(|s| s.id) {
            self.claim_section_id(id);
        }
    }

    fn claim_section_id(&mut self, id: fieldform_scene::SectionCutId) {
        self.section_selected = Some(id);
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
        let Some(section) = self.sections().get(index).copied() else { return };
        let on = !section.on;
        if !self.set_section_on(index, on) {
            return;
        }
        if on {
            self.claim_section(index);
            self.say(format!("Section {} on — drag the handles to move the cut", section.number));
        } else {
            self.say(format!("Section {} off", section.number));
        }
    }

    /// A row's Flip: *its* section, whichever is selected. The only way to flip one since the
    /// menu's Flip section went (the owner, 2026-11-08): the row says which section it is, where a
    /// menu entry had to guess.
    pub(crate) fn flip_section_at(&mut self, index: usize) {
        let Some(number) = self.sections().get(index).map(|s| s.number) else { return };
        self.edit_section(index, false, |c| c.flipped = !c.flipped);
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
        let Some(section) = self.sections().get(index).copied() else { return };
        if !section.on && self.set_section_on(index, true) {
            self.say(format!("Section {} on", section.number));
        }
        self.claim_section(index);
    }

    /// Takes a section out of the list: the row's menu, or Delete with the row selected. One undo
    /// step, which puts it back; the others keep their numbers, so the rows that remain read as
    /// they did.
    pub(crate) fn remove_section(&mut self, index: usize) {
        let Some(record) = self.section_record(index) else { return };
        let sections = self.doc.sections.iter().filter(|c| c.id != record.id).copied().collect();
        self.section_gesture = None;
        self.set_sections(sections, None);
        if self.section_selected == Some(record.id) {
            self.section_selected = None;
        }
        self.say(format!("Section {} removed — undo puts it back", record.number));
    }

    /// The sections' rows in the block list, one each and none until one is made.
    ///
    /// Framed like a block's row and drawn after them. A row's switch and Flip act on *its*
    /// section directly rather than through `view.section`, which acts on the current one: with
    /// several rows the two are not the same section, and a switch that switched some other row
    /// would be the bug the rows exist to make impossible. Removing one is in the row's menu and on
    /// the Delete key, not a button on the row (the owner, 2026-10-24: a ✕ in the notebook "is not
    /// good"). The row is highlighted exactly when the handles are its.
    ///
    /// **And it opens, like a block** (the owner, 2026-11-07: "you should be able to expand the
    /// block and input its orientation, same machinery as the plane block"): inside are the plane
    /// block's own rows -- where it passes through, which way it faces, and its normal -- drawn
    /// by the same two functions, so a section can sit on a point or face the way a plane faces.
    pub(crate) fn section_rows(&mut self, ui: &mut egui::Ui) {
        let count = self.sections().len();
        for index in 0..count {
            // A row that removed itself shortens the list; the rest are drawn next frame rather
            // than under the wrong index now.
            if self.sections().len() != count {
                break;
            }
            self.section_row(ui, index);
        }
    }

    fn section_row(&mut self, ui: &mut egui::Ui, index: usize) {
        let Some(section) = self.sections().get(index).copied() else { return };
        let Some(record) = self.section_record(index) else { return };
        let frame = egui::Frame::new()
            .stroke(egui::Stroke::new(
                1.0_f32,
                ui.visuals().widgets.noninteractive.bg_stroke.color,
            ))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(4, 3));
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.push_id(("section_row", section.id.raw()), |ui| {
                let state = egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    ui.make_persistent_id(("section", section.id.raw())),
                    false,
                );
                state
                    .show_header(ui, |ui| {
                        let mut on = section.on;
                        if ui
                            .checkbox(&mut on, "")
                            .on_hover_text(
                                "Cuts away everything past this plane so you can see inside. \
                                 Changes the view only — the part is not modified — and is saved \
                                 with the document. With several on, what is past any of them is \
                                 cut away.",
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
                            // The rest of the row is the row: drawn as `list_row` draws a block's,
                            // so the highlight reads the same down the list.
                            let header = format!(
                                "Section {}  \u{00b7}  view, {}",
                                section.number,
                                if section.on { "on" } else { "off" }
                            );
                            let selected = section.on && self.section_selected == Some(section.id);
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
                            // Right-clicking selects first, as a block's row does, so Delete on
                            // the key and Delete in the menu agree about which section goes.
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
                    })
                    .body(|ui| self.section_body(ui, index, record, section));
            });
        });
    }

    /// Inside a section's row: the plane block's rows, for the plane the section cuts at.
    fn section_body(
        &mut self,
        ui: &mut egui::Ui,
        index: usize,
        record: fieldform_scene::SectionCut,
        section: Section,
    ) {
        // The rows take the block they are drawn for, to keep it out of its own lists. A section
        // is not a block of the graph, so it is named by an id the graph does not hold: nothing is
        // left out, and none of the planes and points offered can read the section back.
        let stand_in = NodeId::from_raw(record.id.raw());
        let new_origin = self.place_fields(ui, stand_in, "At:", record.origin);
        let new_turn = self.turn_fields(ui, stand_in, record.turn);
        // The normal, read out; Flip is on the row itself, once (the owner, 2026-11-07: a second
        // one down here was one too many).
        ui.separator();
        let n = section.normal();
        ui.label(format!("Normal:  {:.3}, {:.3}, {:.3}", n.x, n.y, n.z));
        if new_origin != record.origin || new_turn != record.turn {
            self.edit_section(index, true, |c| {
                c.origin = new_origin;
                c.turn = new_turn;
            });
        }
    }
}
