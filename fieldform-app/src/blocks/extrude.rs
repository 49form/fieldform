// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Extrude block's panel.
//!
//! Lifted out of `node_block_controls` whole, as the lattice's was. A signed depth, a start,
//! one side or two each with a draft of its own, and the corners a draft cuts are as many
//! rows as any block has.

use crate::*;

impl App {
    /// The Extrude block: the section, the way it is pulled, how far and from where, to one
    /// side or two, and the draft with the corners it cuts.
    ///
    /// Takes the node's seven fields by value, as the lattice panel takes its four: the caller has
    /// already cloned the node, and the panel writes a whole replacement when anything changed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn extrude_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        outline: fieldform_scene::Outline,
        sweep: fieldform_scene::Sweep,
        start: ParamRef,
        height: ParamRef,
        sides: fieldform_scene::Sides,
        draft: ParamRef,
        corners: fieldform_scene::Corners,
        rims: fieldform_scene::Rims,
    ) {
        // Said for the kind it is, since the two are different things to have drawn.
        ui.label(if outline.is_wall() {
            "A curve given a thickness, and then a depth: a wall standing where the \
             curve was drawn, which need not close. The plane is the curve's own — \
             a straight line, which has none, stands upright, or squares itself to \
             the direction it is pulled along — and the depth runs out of one side \
             of it, from where Start puts the material. Flip chooses which side; \
             Sides puts the curve in the middle, or gives the other side a depth of \
             its own."
        } else {
            "A closed profile given thickness. The plane is the profile's own — \
             computed from its points, so there is nothing here to disagree with it \
             — and the depth runs out of one side of it, from where Start puts the \
             material. The profile as drawn is the base; Flip chooses which side, and \
             Sides puts the profile in the middle, or gives the other side a depth of \
             its own."
        });
        ui.separator();
        // The section, through the one editor all three blocks built on one share.
        let new_outline = self.outline_controls(ui, id, outline);
        let new_sweep = self.sweep_controls(ui, id, sweep);
        ui.separator();
        // Signed, with a button that negates it. Which side of the plane the solid
        // grows toward is a real question now that it starts there rather than
        // straddling it -- and a minus sign on the depth answers it without a second
        // piece of state that could disagree with the first.
        let (mut new_height, _) = self.parametric_field(
            ui,
            "Depth:",
            height,
            0.5,
            ANYWHERE_MM,
            "How far out of the plane, measured along the sweep. The profile as drawn \
             is the base. Negative runs the other way, which is what Flip does.",
        );
        if ui
            .button("Flip")
            .on_hover_text("Grow out of the other side of the profile's plane")
            .clicked()
        {
            new_height = match new_height {
                ParamRef::Literal(v) => ParamRef::Literal(-v),
                // A named or driven depth is not this block's number to negate: the
                // sign belongs to whatever supplies it, and flipping here would put
                // the two out of step at the next evaluation.
                other => {
                    self.warn(
                        "This depth comes from a parameter, so flip it where it is set",
                    );
                    other
                }
            };
        }
        // Where the material begins, along the sweep from the profile's plane. A
        // translation and nothing more; the draft then opens from there, since the
        // start plane is where the solid begins and so the base a drawing measures from.
        let (new_start, _) = self.parametric_field(
            ui,
            "Start:",
            start,
            0.5,
            ANYWHERE_MM,
            "How far along the sweep from the profile's plane the material begins. \
             Nought starts at the profile. The depth is measured from here, and so is \
             the draft.",
        );
        // **One block rather than two and a union**, which is what this was being built
        // out of. Two extrudes meeting at a shared face are the join `min` cannot
        // describe: it reads nought straight through the material, and subtracting the
        // pair leaves a sheet of surface in the middle of the cut. One extrude with two
        // sides is a single closed form with no join in it. Three choices in a row, as
        // a sweep's Section is: the one in force is the one lit.
        let mut new_sides = sides.clone();
        ui.horizontal(|ui| {
            row_label(ui, "Sides:");
            let two_like = |sides: &fieldform_scene::Sides| match sides {
                fieldform_scene::Sides::Two { .. } => fieldform_scene::Sides::Two {
                    height: ParamRef::Literal(0.0),
                    draft: ParamRef::Literal(0.0),
                },
                other => other.clone(),
            };
            let mut lit_up = |ui: &mut egui::Ui, choice: fieldform_scene::Sides, word, hover| {
                if ui
                    .selectable_label(two_like(&new_sides) == two_like(&choice), word)
                    .on_hover_text(hover)
                    .clicked()
                {
                    new_sides = choice;
                }
            };
            lit_up(
                ui,
                fieldform_scene::Sides::One,
                "One",
                "The whole depth out of one side of the plane, from the start.",
            );
            lit_up(
                ui,
                fieldform_scene::Sides::Symmetric,
                "Both",
                "Half the depth each way, so the profile is the middle of the solid \
                 rather than one end. A draft then opens away from the plane on both \
                 sides, which is what a parting line means.",
            );
            // Side two starts out as side one's mirror, which is what Both was, so that
            // switching to Two changes nothing until one of its numbers is touched.
            let other = match &sides {
                fieldform_scene::Sides::Two { .. } => sides.clone(),
                _ => fieldform_scene::Sides::Two { height, draft },
            };
            lit_up(
                ui,
                other,
                "Two",
                "The whole depth one way and another depth the other, each with a draft \
                 of its own — a boss standing proud of a plate by one amount and sunk \
                 into it by another, as one solid.",
            );
        });
        // In degrees, because that is what a moulding drawing carries and so what
        // somebody making a part already has in mind. The same number as an end scale,
        // in the form people quote it.
        let (new_draft, _) = self.parametric_field_in(
            ui,
            "Draft:",
            draft,
            0.1,
            DRAFT_DEGREES,
            "How far the walls lean out as they run along the sweep. The start plane is \
             the base, so a positive angle opens away from it and a negative one closes \
             in. Zero is a prism.",
            "\u{b0}",
        );
        // How that draft cuts the section's corners, under the number that makes the
        // question exist. Side two's draft counts too: a part is moulded one way.
        let leaning = self.resolve(new_draft.clone()) != 0.0
            || matches!(&new_sides, fieldform_scene::Sides::Two { draft, .. }
                if self.resolve(draft.clone()) != 0.0);
        let new_corners = self.corner_controls(ui, corners, leaning);
        // Side two's own two numbers, shown where there is a side two.
        if let fieldform_scene::Sides::Two { height: other_height, draft: other_draft } =
            new_sides.clone()
        {
            let (other_height, _) = self.parametric_field(
                ui,
                "Depth 2:",
                other_height,
                0.5,
                POSITIVE_MM,
                "How far the other side runs out of the start plane, the other way.",
            );
            let (other_draft, _) = self.parametric_field_in(
                ui,
                "Draft 2:",
                other_draft,
                0.1,
                DRAFT_DEGREES,
                "The other side's draft, from the start plane, opening away from it.",
                "\u{b0}",
            );
            new_sides =
                fieldform_scene::Sides::Two { height: other_height, draft: other_draft };
        }
        let drafted = new_draft != ParamRef::Literal(0.0)
            || matches!(&new_sides, fieldform_scene::Sides::Two { draft, .. } if *draft != ParamRef::Literal(0.0));
        if drafted {
            // Said where the number is typed, the same as a leaning sweep says it.
            ui.label(
                egui::RichText::new("\u{2248} a drafted wall reads short")
                    .color(ui.visuals().warn_fg_color),
            )
            .on_hover_text(
                "The surface is exactly where it should be: offsetting a distance field \
                 is exact, and that is what a draft is. What reads short is the distance \
                 *to* it, by the secant of the angle — 0.4% at five degrees, 15% \
                 at thirty. Blends against a drafted wall come out a little larger than \
                 the radius asked for.",
            );
        }
        ui.separator();
        let new_rims = self.rims_controls(ui, rims);
        if new_outline != outline
            || new_sweep != sweep
            || new_start != start
            || new_height != height
            || new_draft != draft
            || new_sides != sides
            || new_corners != corners
            || new_rims != rims
        {
            self.replace_node(
                id,
                Node::Extrude {
                    outline: new_outline,
                    sweep: new_sweep,
                    start: new_start,
                    height: new_height,
                    draft: new_draft,
                    sides: new_sides,
                    corners: new_corners,
                    rims: new_rims,
                },
            );
        }
    }
}
