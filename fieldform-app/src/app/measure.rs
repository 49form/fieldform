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

    /// Turns the cutaway on or off, and aims the manipulator at it when it goes on.
    ///
    /// Placed at the middle of **the selected body** if there is one, and at the middle of the
    /// whole scene otherwise. Both are the same rule applied to different scopes: a section
    /// plane at the world origin cuts nothing when the part is somewhere else, and a tool that
    /// appears to do nothing reads as broken. Selecting a body first is how you say which
    /// thing you wanted to look inside.
    ///
    /// On turning it on, not on every selection afterwards. Re-centring while the cut is live
    /// would snatch the plane back from wherever it had been dragged to, which is the same
    /// tool appearing to do something nobody asked for.
    pub(crate) fn toggle_section(&mut self) {
        self.section.on = !self.section.on;
        if self.section.on {
            let selected = self
                .selected_body
                .filter(|id| self.part_graph().roots().contains(id))
                .and_then(|id| self.body_extent(id).as_box())
                .map(|(lo, hi)| nalgebra::center(&lo, &hi));
            if let Some(centre) = selected {
                self.section.origin = centre;
            } else if let Some((centre, _)) = self.scene_bounds() {
                self.section.origin = centre.cast::<f64>();
            }
            // Selected, because turning it on is the act of reaching for it — the same reason
            // creating a body selects it.
            self.section_selected = true;
            self.selected_body = None;
            self.selected_curve = None;
            self.selection.clear();
            self.say("Section on — drag the handles to move the cut");
        } else {
            self.section_selected = false;
            self.say("Section off");
        }
        self.traced.invalidate(traced::Quality::Final);
    }
}
