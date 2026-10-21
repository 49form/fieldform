// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Morph block's panel.
//!
//! Lifted out of `node_block_controls` whole, as the lattice's was. Its driver,
//! `drive_morph`, stays in `app/block_controls.rs` beside the offset's and the pipe's, as
//! `drive_lattice` did: the drivers are one family, whichever panel calls them.

use crate::*;
use fieldform_scene::Operand;

impl App {
    /// The Morph block: the two bodies, and an amount that is one number or a ramp.
    ///
    /// The node comes whole beside its three fields, because the panel compares what it
    /// rebuilds against all of it. The caller has already cloned it.
    pub(crate) fn morph_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        node: Node,
        a: Operand,
        b: Operand,
        t: ParamRef,
    ) {
        ui.label(
            "One body becoming another. Not a boolean — at anything between 0 and \
             100% the result contains neither shape, it is the average of the two.",
        );
        // **Said here because it decides whether the block does anything at all.** A
        // morph of two bodies that are apart is empty in the middle: the average of two
        // fields that are both far from their surfaces is far from any surface. It
        // looks like the shape flies apart, and the answer is not a setting.
        ui.label(
            egui::RichText::new(
                "Works when the two are in the same place. Two bodies apart from each \
                 other vanish in the middle rather than travelling.",
            )
            .weak(),
        );
        ui.separator();

        // **The same row a boolean's operands get (D9), and the slots themselves rather than a
        // buffer.** The `Option` these two were edited through was only ever an editing buffer —
        // "none" was what a picker showed for an input that was not among its choices, and the
        // write at the bottom put the old value back — so an emptied slot could not have survived
        // it. An `Operand` *is* the state, so the row writes what the node keeps.
        let choices = self.operand_choices(id);
        let offered = self.operand_offers(id, &choices);
        let (mut new_a, mut new_b) = (a, b);
        // A morph mixes two fields and no more, so neither row can be removed — only emptied.
        self.operand_row(ui, "From:", (id, "morph_a"), &mut new_a, &offered, false);
        self.operand_row(ui, "To:", (id, "morph_b"), &mut new_b, &offered, false);
        self.reuse_hint(ui, id);

        // The same switch a driven offset has, and deliberately the same words: it is
        // one idea — a number, or a number that varies through space — and two blocks
        // spelling it differently would teach two habits for it.
        let driver = t.driver();
        // **The switch stays live while a slot is empty**, and the slots go to `drive_morph`
        // whole: it fits the band through `operand_extent`, where a hole answers `Empty` and
        // `Empty` unions away, so the band is fitted to whichever shape is there and to the
        // default size when neither is. A block with a hole is a block still being built, and
        // disabling its other controls would make the hole cost more than the hole.
        //
        // The slots as this frame edited them rather than the node's, so filling a slot and
        // switching to a ramp in one gesture fits the band to what was just put in.
        ui.horizontal(|ui| {
            row_label(ui, "Amount:");
            if ui.add(mode_button(driver.is_none(), "One number")).clicked() {
                if let Some(ramp) = driver {
                    self.drop_driver(id, ramp);
                }
            }
            if ui.add(mode_button(driver.is_some(), "Ramp")).clicked()
                && driver.is_none()
            {
                self.drive_morph(id, new_a, new_b);
            }
        });
        // The switch makes a ramp; this picks one that exists. Both write the same
        // `ParamRef::Field`, so a ramp reached either way is the same ramp and one of
        // them can drive this morph and an offset at once (`ramp-decision.md` §7.2).
        if let Some(choice) = self.pick_driver(ui, "morph_driver", driver) {
            match choice {
                Some(ramp) => self.wire_driver(id, 0, ramp),
                None => {
                    if let Some(ramp) = driver {
                        self.drop_driver(id, ramp);
                    }
                }
            }
        }

        let mut new_t = t;
        match driver {
            None => {
                // Nought to one in the document and a percentage here, because nobody
                // says "morph it to 0.35".
                let mut percent = self.resolve(t).clamp(0.0, 1.0) * 100.0;
                let response = ui.add(
                    egui::Slider::new(&mut percent, 0.0..=100.0)
                        .text("Amount")
                        .suffix(" %"),
                );
                if response.changed() {
                    new_t = self.set_value(t, percent / 100.0);
                    if matches!(t, ParamRef::Named(_)) {
                        self.rebuild_render_resources();
                    }
                }
                ui.label(
                    egui::RichText::new(
                        "Dragging this is a buffer write — no shader rebuild, \
                         so it is live.",
                    )
                    .weak(),
                );
            }
            Some(ramp) => {
                // **What it costs, said plainly.** This is the one node whose tracing
                // cost is both large and chosen by the user — the band they type on the
                // ramp is what sets it — so hiding the number means a narrow band makes
                // the viewport slow with no visible cause.
                // The number the tracer actually divides by, from the compiled field,
                // not a second derivation of it. The box bound this used to show reads
                // high -- x42.5 against the intervals' x24.5 on the reported scene --
                // so the panel was quoting a cost the viewport was not paying, and
                // would have kept quoting it after the divisor got tighter.
                let divisor = self
                    .pick_field
                    .as_ref()
                    .and_then(|f| {
                        f.morph_divisors().iter().find(|(n, _)| *n == id).map(|(_, d)| *d)
                    })
                    .unwrap_or_else(|| {
                        fieldform_field::driven_morph_divisor(&self.doc, self.part_graph(), id)
                    });
                let text = format!("tracing cost \u{00d7}{divisor:.1}");
                match divisor > 20.0 {
                    true => {
                        ui.colored_label(ui.visuals().warn_fg_color, text)
                            .on_hover_text(
                                "The transition is sharp relative to how far apart \
                                 these bodies are. Widen the ramp's input range, or \
                                 move the bodies closer.",
                            );
                    }
                    false => {
                        ui.label(egui::RichText::new(text).weak()).on_hover_text(
                            "Every ray step is divided by this to stay inside the \
                             surface, so the viewport takes about this many times as \
                             many steps.",
                        );
                    }
                }
                // **The ramp's own numbers are edited in the ramp's own block.** They
                // were repeated here, which put one set of values in two panels — and a
                // ramp can drive two sockets, so the same numbers appeared under two
                // different blocks and each looked like it owned them. The block list
                // has a row for the ramp; this says which ramp and what it costs, and
                // stops there.
                ui.label(
                    egui::RichText::new(format!(
                        "Edit its band and easing in {}",
                        self.body_label(ramp)
                    ))
                    .weak(),
                );
            }
        }

        let rebuilt = Node::Morph { a: new_a, b: new_b, t: new_t };
        if rebuilt != node {
            self.replace_node(id, rebuilt);
        }
    }
}
