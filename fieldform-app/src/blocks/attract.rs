// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Attract block's panel.
//!
//! Lifted out of `node_block_controls` whole, as the lattice's was: two inputs, one of
//! which may be either a curve or a point, and two numbers that only mean something
//! together, so the panel says what they come to.

use crate::*;

impl App {
    /// The Attract block: which body is warped, the curve or point it is drawn toward, and
    /// how hard and how far, with what those two come to in millimetres.
    pub(crate) fn attract_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: fieldform_scene::Operand,
        toward: fieldform_scene::Attractor,
        strength: ParamRef,
        falloff: ParamRef,
    ) {
        ui.label(
            "Draws this body toward a curve or a point — hardest on it, fading \
             to nothing at the reach, and untouched beyond it.",
        );
        ui.separator();

        // **Both inputs, because an attractor has two.** Which body is warped was
        // chosen when the block was made, and choosing it there is right -- but it was
        // then fixed, so pointing the block at a different body meant deleting it and
        // making another. That is the same argument the pipe's curve chooser already
        // won: an input is an input, and changing one is an edit to this block.
        let choices = self.operand_choices(id);
        let mut new_input = input.source();
        self.pick_operand(ui, "Body:", "attract_body", id, &choices, &mut new_input);
        // **One list, because an attractor is drawn toward one thing.** A curve and a
        // point are two answers to the same question, so two controls would ask it
        // twice and leave the answer in whichever held it — the mistake the ramp's
        // input was making. Two targets at once is not offered at all: it would need a
        // rule for combining two pulls, and there is none anybody would guess.
        let mut curves: Vec<(CurveId, String)> = self
            .doc
            .curves
            .iter()
            .map(|(id, c)| (*id, c.name.clone()))
            .collect();
        curves.sort_by_key(|(id, _)| id.raw());
        // Points only. Every other block has a position somewhere, but "toward a
        // sphere" would mean toward its centre and would read as toward its surface —
        // a point is the one block whose position is the whole of what it is.
        let mut points: Vec<NodeId> = self
            .part_graph()
            .nodes()
            .filter(|(n, node)| *n != id && matches!(node, Node::Point { .. }))
            .map(|(n, _)| n)
            .collect();
        points.sort_by_key(|n| n.raw());
        let points: Vec<(NodeId, String)> =
            points.into_iter().map(|n| (n, self.body_label(n))).collect();
        let mut chosen = toward;
        let current = match toward {
            fieldform_scene::Attractor::Curve(cid) => curves
                .iter()
                .find(|(c, _)| *c == cid)
                .map(|(_, name)| name.clone())
                .unwrap_or_else(|| "missing".to_string()),
            fieldform_scene::Attractor::Point(n) => self.body_label(n),
        };
        // One list of targets rather than a section each, so the filter covers both -- and
        // so a curve and a point read as two answers to one question, which is what they
        // are. A point says so beside its name, since the two kinds can share one.
        //
        // **The only field holding a reference that is either kind**, which is why
        // `reference_field` is generic over its key rather than over a node: an `Attractor`
        // cannot say from its type which of the two it is. A curve's name is its address
        // and a point's is whatever it has been named, so the two halves of this list are
        // built the two different ways and the widget does not have to know.
        let targets: Vec<RefChoice<fieldform_scene::Attractor>> = curves
            .iter()
            .map(|(cid, name)| {
                RefChoice::named(fieldform_scene::Attractor::Curve(*cid), name)
            })
            .chain(points.iter().map(|(n, name)| {
                RefChoice::declared(
                    fieldform_scene::Attractor::Point(*n),
                    format!("{name}  (point)"),
                    self.body_name(*n),
                )
            }))
            .collect();
        ui.horizontal(|ui| {
            row_label(ui, "Toward:");
            if let Some(picked) = self.reference_field(
                ui,
                (id, "attract_target"),
                chosen,
                &current,
                &targets,
                RefKind::Target,
                // Either kind, and the only field in the panel where that is true: a
                // dragged block becomes the point it is, a dragged curve the curve it is.
                DropsFrom::either(
                    fieldform_scene::Attractor::Point,
                    fieldform_scene::Attractor::Curve,
                ),
            ) {
                chosen = picked;
            }
        });
        // **Whichever kind it is, under the row that names it** (`notebook-spec.md` §7.1, §17).
        // The one field in the panel that takes either, so it is the one host call that has to
        // ask which: a point is a block and a curve is a curve, and the two have their own rows.
        match chosen {
            fieldform_scene::Attractor::Point(point) => self.host_nested_block(ui, point),
            fieldform_scene::Attractor::Curve(curve) => self.host_nested_curve(ui, curve),
        }

        let (new_strength, changed_strength) = self.parametric_field_in(
            ui,
            "Pull:",
            strength,
            0.01,
            ATTRACT_STRENGTH,
            "How hard the curve pulls. Negative pushes the body away instead.",
            "",
        );
        let (new_falloff, changed_falloff) = self.parametric_field(
            ui,
            "Reach:",
            falloff,
            0.5,
            POSITIVE_MM,
            "How far the pull carries. Exactly nothing beyond this.",
        );

        // What the two unitless-and-millimetre numbers actually come to, in
        // millimetres, because neither of them alone says how far anything moves.
        // Derived rather than stored, so it cannot drift from the pair it reads.
        let s = self.resolve(new_strength);
        let r = self.resolve(new_falloff);
        let pull = fieldform_field::max_pull(s, r);
        ui.label(
            egui::RichText::new(format!(
                "moves the surface up to {pull:.2} mm, at about {:.1} mm out",
                fieldform_field::PEAK_PULL_AT * r,
            ))
            .small()
            .weak(),
        );
        // Said where the number is typed, which is the only place saying it is honest
        // -- the same rule the approximate-fillet note follows.
        if fieldform_field::folds(s) {
            ui.label(
                egui::RichText::new(format!(
                    "This pull folds the body through itself somewhere. Between \
                     {:.2} and {:.2} it cannot.",
                    fieldform_field::FOLD_PUSH_STRENGTH,
                    fieldform_field::FOLD_STRENGTH,
                ))
                .small()
                .color(egui::Color32::from_rgb(0xd0, 0x8a, 0x3a)),
            );
        }
        ui.label(
            egui::RichText::new(
                "Booleans over this are exact. A fillet is approximate and is marked \
                 where its radius is typed — a warp stretches material, so the \
                 field reads short.",
            )
            .small()
            .weak(),
        );

        // An operand the list no longer offers is one that would close a loop, so the
        // block keeps the one it had rather than losing it -- the same rule
        // `operand_choices` states.
        // `pick_operand` shows nothing for an input not among its choices, so an unchanged
        // pick reads as `None` and must not be mistaken for the slot having been emptied --
        // emptying is the row's own act (§7.2), never a side effect of a picker.
        let body = new_input.map_or(input, Into::into);
        if changed_strength || changed_falloff || chosen != toward || body != input {
            self.replace_node(
                id,
                Node::Attract {
                    input: body,
                    toward: chosen,
                    strength: new_strength,
                    falloff: new_falloff,
                },
            );
        }
    }
}
