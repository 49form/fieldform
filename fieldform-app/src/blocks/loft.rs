// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Loft block's panel.
//!
//! Lifted out of `node_block_controls` whole, as the lattice's was: an ordered list of
//! sections, a centerline that is a curve or derived from them, and the readings that say
//! what to redraw.

use crate::*;

impl App {
    /// The Loft block: its sections in order, its centerline, how smoothly it passes
    /// through them, and what the field reads as a result.
    ///
    /// The sections are borrowed from the node the caller matched, and the node comes too,
    /// because the panel compares what it rebuilds against the whole of it.
    pub(crate) fn loft_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        node: &Node,
        sections: &[fieldform_scene::Outline],
        spine: Option<CurveId>,
        smooth: fieldform_scene::LoftSmooth,
        rims: fieldform_scene::Rims,
    ) {
        ui.label(
            "Closed curves passed through in turn along a centerline, the solid \
             changing from one to the next. Each is read square to the centerline \
             where it passes nearest the curve's middle, so the list is the order the \
             path meets them. An interpolation always reads a little short, so a \
             fillet against it is marked approximate.",
        );
        ui.separator();
        // **The sections, as a list.** Each row is a curve field like a boolean's
        // operand row, with the two arrows that are the order, and a remove while
        // more than two remain -- two being the least a loft means anything with.
        let mut new_sections: Vec<fieldform_scene::Outline> = sections.to_vec();
        let mut curves: Vec<(CurveId, String)> = self
            .doc
            .curves
            .iter()
            .filter(|(cid, _)| Some(**cid) != spine)
            .map(|(cid, c)| (*cid, c.name.clone()))
            .collect();
        curves.sort_by_key(|(cid, _)| cid.raw());
        let offered = self.curve_choices(&curves);
        let mut removed: Option<usize> = None;
        let mut moved: Option<(usize, usize)> = None;
        let count = new_sections.len();
        for i in 0..count {
            let current = new_sections[i].curve();
            let shown = self.shown_label(&offered, current, "missing");
            ui.horizontal(|ui| {
                row_label(ui, if i == 0 { "Sections:" } else { "" });
                if let Some(picked) = self.reference_field(
                    ui,
                    (id, "loft_section", i),
                    current,
                    &shown,
                    &offered,
                    RefKind::Curve,
                    DropsFrom::curve(|c| c),
                ) {
                    new_sections[i] = fieldform_scene::Outline::closed(picked);
                }
                // `\u{2b06}` and `\u{2b07}`, which egui's icon font has -- its text
                // face has no triangles at all; `grip_mark` records the discovery.
                ui.add_enabled_ui(i > 0, |ui| {
                    if ui
                        .small_button("\u{2b06}")
                        .on_hover_text("Earlier along the centerline")
                        .clicked()
                    {
                        moved = Some((i, i - 1));
                    }
                });
                ui.add_enabled_ui(i + 1 < count, |ui| {
                    if ui
                        .small_button("\u{2b07}")
                        .on_hover_text("Later along the centerline")
                        .clicked()
                    {
                        moved = Some((i, i + 1));
                    }
                });
                if count > 2
                    && remove_button(ui, "Take this section out of the loft. The curve stays.")
                {
                    removed = Some(i);
                }
            });
            // Each section under its own row, which is what makes a loft of four sections read
            // as four things it is made of rather than a list and a heap (§7.1, §17).
            self.host_nested_curve(ui, current);
        }
        if let Some((from, to)) = moved {
            new_sections.swap(from, to);
        }
        if let Some(i) = removed
            && new_sections.len() > 2
        {
            new_sections.remove(i);
        }
        // Adding by name, from the curves not yet in the list. Whether a curve closes
        // is the compile's to say, by name, when the edit is refused.
        let eligible: Vec<(CurveId, String)> = curves
            .iter()
            .filter(|(c, _)| !new_sections.iter().any(|o| o.curve() == *c))
            .cloned()
            .collect();
        ui.horizontal(|ui| {
            row_label(ui, "");
            if eligible.is_empty() {
                ui.label(
                    egui::RichText::new("Every curve in the document is in the list.")
                        .weak()
                        .italics(),
                );
            } else {
                ui.menu_button("Add \u{203a}", |ui| {
                    for (c, name) in &eligible {
                        if ui
                            .button(name)
                            .on_hover_text("Add this curve as the last section")
                            .clicked()
                        {
                            new_sections.push(fieldform_scene::Outline::closed(*c));
                            ui.close();
                        }
                    }
                })
                .response
                .on_hover_text("Add a section after the last");
            }
        });
        ui.separator();
        // **The centerline: a curve, or "derived".** The field never answers "clear",
        // so going back to a derived one is a button beside it; and a derived one can
        // be written into the document as the cubic it is, after which it is dragged
        // like any curve and the loft follows it.
        let mut new_spine = spine;
        let mut make_editable = false;
        let section_ids: Vec<CurveId> = new_sections.iter().map(|o| o.curve()).collect();
        let mut paths: Vec<(CurveId, String)> = self
            .doc
            .curves
            .iter()
            .filter(|(cid, _)| !section_ids.contains(cid))
            .map(|(cid, c)| (*cid, c.name.clone()))
            .collect();
        paths.sort_by_key(|(cid, _)| cid.raw());
        let offered_paths = self.curve_choices(&paths);
        let shown = match spine {
            Some(s) => self.shown_label(&offered_paths, s, "missing"),
            None => "derived".to_string(),
        };
        ui.horizontal(|ui| {
            row_label(ui, "Centerline:");
            if let Some(picked) = self.reference_field_in(
                ui,
                (id, "loft_spine"),
                spine,
                &shown,
                &offered_paths,
                RefKind::Curve,
                DropsFrom::curve(|c| c),
            ) {
                new_spine = Some(picked);
            }
            match spine {
                Some(_) => {
                    if ui
                        .small_button("derive")
                        .on_hover_text(
                            "Let the loft build its own centerline from the first and \
                             last sections again. The curve stays in the document.",
                        )
                        .clicked()
                    {
                        new_spine = None;
                    }
                }
                None => {
                    if ui
                        .small_button("make editable")
                        .on_hover_text(
                            "Write the derived centerline into the document as the \
                             cubic curve it is, and point the loft at it. Drag its \
                             control points to steer the solid; it no longer moves \
                             with the sections on its own.",
                        )
                        .clicked()
                    {
                        make_editable = true;
                    }
                }
            }
        });
        // The centerline, when there is one in the document: a derived one is no curve and has
        // no row (§7.1, §17).
        if let Some(spine) = spine {
            self.host_nested_curve(ui, spine);
        }
        if spine.is_none() {
            ui.label(
                egui::RichText::new(
                    "Built from the end sections: a cubic from the first's middle to \
                     the last's, setting off square to each. Drawn dashed in the \
                     viewport, moves with the sections, never saved.",
                )
                .small()
                .weak(),
            );
        }
        if make_editable {
            let flattened = fieldform_field::loft_parts(
                &self.doc,
                &new_sections,
                None,
                fieldform_field::PROFILE_TURN_DEGREES,
                fieldform_field::DEFAULT_TOLERANCE_MM,
            )
            .ok()
            .and_then(|parts| fieldform_field::derived_centerline_cubic(&parts.sections));
            match flattened {
                Some(points) => {
                    let cid = CurveId::new();
                    match fieldform_scene::Curve::new(
                        cid,
                        self.unique_curve_name("Centerline"),
                        3,
                        points.to_vec(),
                        vec![1.0; 4],
                        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        false,
                    ) {
                        Ok(curve) => {
                            self.doc.curves.insert(cid, curve);
                            // Placed now, as everything written past the history
                            // is (`block-ownership-spec.md` §4.1): the edit pointing
                            // the loft at it below can be refused.
                            self.doc.settle_ownership();
                            self.rebuild_curve_display();
                            new_spine = Some(cid);
                            self.say("The centerline is a curve now — drag it to steer the loft");
                        }
                        Err(e) => self.warn(format!("The centerline could not be made: {e}")),
                    }
                }
                None => self.warn(
                    "The centerline cannot be derived until the first and last sections \
                     are closed curves with a middle to join",
                ),
            }
        }
        ui.separator();
        let mut new_smooth = smooth;
        ui.horizontal(|ui| {
            row_label(ui, "Smooth:");
            for choice in fieldform_scene::LoftSmooth::ALL {
                if ui
                    .add(mode_button(new_smooth == choice, choice.label()))
                    .on_hover_text(match choice {
                        fieldform_scene::LoftSmooth::G0 => {
                            "Straight from each section to the next: the shape changes \
                             at a steady rate and creases where it passes through a \
                             listed one."
                        }
                        fieldform_scene::LoftSmooth::G1 => {
                            "Smooth through every section: no crease where the solid \
                             passes through a listed one. Each corner rides a smooth \
                             rail through its positions, which can swing a little wide \
                             between two alike sections after a different one."
                        }
                    })
                    .clicked()
                {
                    new_smooth = choice;
                }
            }
        });
        // **The readings, under the list** (`loft-spec.md` §3, §6): what the field
        // divides by, how far each section is from square to its path, and whether the
        // solid necks between two of them. Readings and never knobs -- each says what
        // to redraw, not what to type.
        // The rims under the shape's own rows, as the extrude and the sweep have them.
        ui.separator();
        let new_rims = self.rims_controls(ui, rims);
        let asked = Node::Loft {
            sections: new_sections.clone(),
            spine: new_spine,
            smooth: new_smooth,
            rims: new_rims,
        };
        if let Some(readings) = self.loft_readings(id, &asked).cloned() {
            if readings.stretch > 1.05 {
                ui.horizontal(|ui| {
                    row_label(ui, "");
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!("\u{2248} reads {:.1}\u{d7} short", readings.stretch),
                    )
                    .on_hover_text(format!(
                        "The sections differ, or sit wide on a bent centerline, so the \
                         field reads short by this factor to stay honest, and the \
                         viewport takes {:.1}\u{d7} the steps to draw it. Sections more \
                         alike, more of them, or a gentler centerline brings it down.",
                        readings.stretch
                    ));
                });
            }
            if readings.tilts.iter().any(|t| *t > 0.5) {
                ui.horizontal(|ui| {
                    row_label(ui, "Tilt:");
                    ui.label(
                        readings
                            .tilts
                            .iter()
                            .map(|t| format!("{t:.0}\u{b0}"))
                            .collect::<Vec<_>>()
                            .join(", "),
                    )
                    .on_hover_text(
                        "How far each section's own plane is from square to the \
                         centerline where it sits, in list order. A tilted section is \
                         read as its projection onto the square plane -- foreshortened \
                         by that angle -- which is how a sketch drawn flat across a \
                         leaning path is meant to be read. Redraw the section square to \
                         the path to read it whole.",
                    );
                });
            }
            if let Some((between, share)) = readings.neck.filter(|(_, s)| *s < 0.5) {
                ui.horizontal(|ui| {
                    row_label(ui, "");
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!(
                            "necks to {:.0}% between sections {} and {}",
                            share * 100.0,
                            between + 1,
                            between + 2
                        ),
                    )
                    .on_hover_text(
                        "Between these two the section pinches to this share of the \
                         smaller one's area: its corners were matched to corners so \
                         far round the other section that the sides cross on the way, \
                         like a square lofted to the same square turned nearly half \
                         a turn. Turn one section to match the other, or put a \
                         section between them.",
                    );
                });
            }
        }
        let rebuilt = Node::Loft {
            sections: new_sections,
            spine: new_spine,
            smooth: new_smooth,
            rims: new_rims,
        };
        if rebuilt != *node {
            self.replace_node(id, rebuilt);
        }
    }
}
