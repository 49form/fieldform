// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Infill block: the tool that makes one, and its panel.
//!
//! The block itself is one line of arithmetic the kernel already had; what is here is what a
//! person needs from it. The wall lives on this block. The lattice stays a block of its own, in
//! the slot, so the rows here that touch it write that block rather than a copy of it.

use crate::infill_presets::{self, Preset};
use crate::*;
use fieldform_scene::{Banding, Frame, LatticeCell, LatticePhase, Operand};
use std::sync::mpsc;

/// How many points the cavity report samples. Enough for a figure quoted to two places once the
/// deepest of them has been walked downhill, and a fraction good to about a percent.
const CAVITY_SAMPLES: u32 = 60_000;

/// How long a document must sit still before the cavity is measured, so a drag asks once.
const CAVITY_SETTLE: std::time::Duration = std::time::Duration::from_millis(300);

/// The cavity report of one block, computed off the interface thread.
pub(crate) struct CavityRun {
    /// The document revision this answers for, or is waiting to.
    revision: u64,
    /// When that revision was first seen, for [`CAVITY_SETTLE`].
    seen: std::time::Instant,
    /// The worker, while one is running.
    rx: Option<mpsc::Receiver<Option<fieldform_field::InfillCavity>>>,
    /// Whether a worker has been sent for this revision, so one is sent once.
    asked: bool,
    /// The latest answer, which may be a revision behind for as long as the next one takes.
    report: Option<fieldform_field::InfillCavity>,
}

#[cfg(test)]
impl CavityRun {
    /// The latest answer, for the tests to wait on.
    pub(crate) fn report(&self) -> Option<fieldform_field::InfillCavity> {
        self.report
    }
}

impl App {
    /// `infill`: the body named or selected gets a skin, and whatever fills it.
    ///
    /// Commits at once like every session command: the block exists at sensible numbers and then
    /// gets edited, so there is no state in which it is half made.
    pub(crate) fn apply_infill(&mut self, args: &Args) {
        let Ok(named) = self.named_target("infill", args) else { return };
        let bodies = self.doc.listed_bodies(self.part_graph());
        let target = Self::target_operand(named, self.selected_body, &bodies);
        let lookup = |t: &str| self.lookup(t);
        let asked_fill = match console_keys::operand_named("infill", "fill", args, &lookup) {
            Ok(asked) => asked,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        if asked_fill.is_some()
            && let Some(key) = console_keys::INFILL_LATTICE_KEYS.iter().find(|k| args.get(k).is_some())
        {
            self.warn(format!(
                "`infill fill=` names a block to fill with and `{key}=` describes a new one — say one or the other"
            ));
            return;
        }

        // The body's box, for where the cells start and how big they are. A body with no box is
        // refused by name when the block compiles, and the numbers here only have to be numbers.
        let bounds = target
            .source()
            .and_then(|id| fieldform_field::graph_extent(&self.doc, self.part_graph(), id).as_box());
        let side = bounds
            .map(|(lo, hi)| {
                let size = hi - lo;
                size.x.min(size.y).min(size.z)
            })
            .unwrap_or(DEFAULT_BODY_MM);
        let typed_wall = args.get("wall").and_then(|v| v.as_f64());
        let click = infill_presets::first_click(side, typed_wall);

        let mut graph = self.part_graph().clone();
        let fill = match asked_fill {
            Some(fill) => fill,
            None => {
                // **At the body's centre, as a number.** So the pattern sits symmetrically in the
                // part, and does not swim when the body is edited upstream.
                let frame = match bounds {
                    Some((lo, hi)) => {
                        let centre = lo + (hi - lo) * 0.5;
                        let mut frame = Frame::default();
                        if let Frame::At { origin, .. } = &mut frame {
                            *origin = [centre.x, centre.y, centre.z].map(ParamRef::Literal);
                        }
                        frame
                    }
                    None => Frame::default(),
                };
                let lattice = match self.infill_lattice(frame, click.cell_mm, args) {
                    Ok(lattice) => lattice,
                    Err(why) => {
                        self.warn(why);
                        return;
                    }
                };
                graph.add(lattice).into()
            }
        };
        let block = Node::Infill {
            input: target,
            wall: ParamRef::Literal(click.wall_mm),
            fill,
            join: Fillet::hard(),
        };
        let block = match console_keys::infill_with_keys(block, args, &|t| self.lookup(t)) {
            Ok(block) => block,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let block = graph.add(block);
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        if let Some(target) = target.source() {
            self.clear_hidden(target);
        }
        self.select_body(Some(block));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(if target.is_empty() {
            "An infill with an empty slot \u{2014} put a block in it".to_string()
        } else {
            format!("Infilled behind a {:.3} mm wall", click.wall_mm)
        });
    }

    /// The lattice a new infill owns: the preset for the cell asked for, at `cell_mm`, with the
    /// same keys `tpms` reads laid over it.
    ///
    /// **A graded phase keeps the world's frame.** A lattice whose phase follows a ramp cannot be
    /// aimed, which the shader generator refuses by name, and a frame moved to the body's centre
    /// counts. An origin typed on the line is still asked for and still refused.
    fn infill_lattice(&self, frame: Frame, cell_mm: f64, args: &Args) -> Result<Node, String> {
        let gyroid = Preset::of(LatticeCell::Gyroid).expect("the table holds a gyroid");
        let node = Node::Lattice {
            frame,
            cell: LatticeCell::Gyroid,
            map: CellMap::Rectangular { cell: ParamRef::Literal(cell_mm) },
            phase: gyroid.phase(cell_mm),
        };
        let offered = |cell: &LatticeCell| fieldform_field::CellKind::from(*cell).supported_bandings();
        let seed = |banding: Banding| seeded_phase(banding, cell_mm);
        let mut node =
            console_keys::lattice_with_keys(node, args, &offered, &seed, &|t| self.lookup(t))
                // Its refusals name the verb they were written for, and this is the one typed.
                .map_err(|why| why.replace("`tpms", "`infill"))?;
        let Node::Lattice { frame, cell, map, phase } = &mut node else {
            return Err("`infill` fills with a lattice".to_string());
        };
        if phase.param().driver().is_some() && args.get("origin").is_none() {
            *frame = Frame::default();
        }
        // `cell=` alone applies that cell's own start. Any phase key typed overrides it, since
        // what was typed is what was meant.
        let typed = ["thickness", "offset", "density", "radius"].iter().any(|k| args.get(k).is_some());
        if !typed && let Some(preset) = Preset::of(*cell) {
            let size = self.infill_cell_mm(map, cell_mm);
            let start = preset.phase(size);
            *phase = if start.banding() == phase.banding() {
                start
            } else {
                seeded_phase(phase.banding(), size)
            };
        }
        Ok(node)
    }

    /// The number a preset's proportion is of: a map's first size, or `otherwise`.
    fn infill_cell_mm(&self, map: &CellMap, otherwise: f64) -> f64 {
        map.params()
            .next()
            .and_then(|size| self.doc.resolve(size))
            .filter(|size| *size > 0.0)
            .unwrap_or(otherwise)
    }

    /// The Infill block's panel. The wall, what fills it, and how the two meet, in the order they
    /// are decided, with what the block has to say about itself underneath.
    pub(crate) fn infill_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
        wall: ParamRef,
        fill: Operand,
        join: Fillet,
    ) {
        ui.label(
            "Gives this body a solid skin and fills the inside. The wall is measured inward, so \
             the outside dimensions do not change.",
        );
        ui.separator();
        let picked = self.pick_input(ui, id, input);
        let (new_wall, _) = self.parametric_field(
            ui,
            "Wall:",
            wall,
            0.05,
            FROM_NOUGHT_MM,
            "How thick the skin is. Nought is plain infill: no skin, and the lattice cut by the \
             outer surface.",
        );
        self.wall_minimum_note(ui, picked, new_wall);
        let wall_mm = self.resolve(new_wall);

        let choices = self.operand_choices(id);
        let mut chosen = fill.source();
        self.pick_operand(ui, "Fill:", "infill_fill", id, &choices, &mut chosen);
        let new_fill = chosen.map_or(fill, Operand::filled);

        // Written to the lattice block itself, in the same frame the rows below draw it.
        if let Some(lattice) = new_fill.source() {
            self.infill_lattice_rows(ui, lattice);
        }

        ui.separator();
        let new_join = self.infill_join_rows(ui, id, join, picked, new_fill, wall_mm);

        if let Some(lattice) = new_fill.source()
            && matches!(self.part_graph().node(lattice), Some(Node::Lattice { .. }))
        {
            let lines =
                lattice_readout::lattice_readout(&self.doc, self.part_graph(), lattice, None);
            self.lattice_readout_rows(ui, &lines);
        }
        self.infill_notes(ui, id, wall_mm);

        if picked != input || new_wall != wall || new_fill != fill || new_join != join {
            self.replace_node(
                id,
                Node::Infill { input: picked, wall: new_wall, fill: new_fill, join: new_join },
            );
        }
    }

    /// The preset, the cell size and the phase's number of the lattice in the slot. What else a
    /// lattice has is on its own panel, one click in.
    fn infill_lattice_rows(&mut self, ui: &mut egui::Ui, id: NodeId) {
        let Some(Node::Lattice { cell, frame, map, phase }) = self.part_graph().node(id).cloned()
        else {
            return;
        };
        let mut new_cell = cell;
        let mut new_phase = phase;
        let size = self.infill_cell_mm(&map, DEFAULT_BODY_MM * 0.2);
        ui.horizontal(|ui| {
            row_label(ui, "Lattice:");
            egui::ComboBox::from_id_salt((id, "infill_preset"))
                .selected_text(lattice_cell_label(cell))
                .show_ui(ui, |ui| {
                    let mut group = "";
                    for preset in infill_presets::PRESETS {
                        if preset.group != group {
                            group = preset.group;
                            ui.label(egui::RichText::new(group).small().weak());
                        }
                        let option = preset.cell();
                        // A pairing the compile refuses is greyed with the reason, as the
                        // lattice panel's own list does.
                        let refusal = blocks::lattice::cell_refuses_map(option, map);
                        let response = ui.add_enabled(
                            refusal.is_none(),
                            egui::Button::selectable(cell.name() == option.name(), preset.label),
                        );
                        if let Some(why) = refusal {
                            response.on_hover_text(why);
                            continue;
                        }
                        // Resets the phase's number from the proportion and today's cell size.
                        // Changing the size afterwards does not rescale it: the number stays in
                        // millimetres, as it does on a lattice block.
                        if response.clicked() {
                            new_cell = option;
                            new_phase = preset.phase(size);
                        }
                    }
                });
        });

        let mut new_map = map;
        if let CellMap::Rectangular { cell: size } = map {
            let (edited, _) = self.parametric_field(
                ui,
                "Cell size:",
                size,
                0.25,
                POSITIVE_MM,
                "How far the lattice repeats. The wall or strut below stays in millimetres.",
            );
            new_map = CellMap::Rectangular { cell: edited };
        }

        let number = *new_phase.param();
        if number.driver().is_some() {
            ui.label(
                egui::RichText::new("Its number follows a ramp, edited on the lattice's own panel.")
                    .small()
                    .weak(),
            );
        } else {
            let edited = match &new_phase {
                LatticePhase::Sheet { .. } => {
                    self.parametric_field(ui, "Thickness:", number, 0.05, POSITIVE_MM, "").0
                }
                LatticePhase::Strut { .. } => {
                    self.parametric_field(ui, "Strut radius:", number, 0.05, POSITIVE_MM, "").0
                }
                LatticePhase::Solid { .. } => {
                    self.parametric_field_in(ui, "Density:", number, 0.01, 0.0..=1.0, "", "").0
                }
            };
            match &mut new_phase {
                LatticePhase::Sheet { thickness, .. } => *thickness = edited,
                LatticePhase::Strut { radius } => *radius = edited,
                LatticePhase::Solid { density } => *density = edited,
            }
        }

        if new_cell != cell || new_map != map || new_phase != phase {
            self.replace_node(id, Node::Lattice { cell: new_cell, frame, map: new_map, phase: new_phase });
        }
    }

    /// How the fill meets the skin: the shapes a blend offers in the words it offers them, and
    /// greyed with the reason at a wall of nought, where there is no skin to meet.
    fn infill_join_rows(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        join: Fillet,
        body: Operand,
        fill: Operand,
        wall_mm: f64,
    ) -> Fillet {
        const NO_SKIN: &str = "With no skin there is nothing for the fill to meet, and rounding \
                               would flare every lattice wall where the outer surface cuts it.";
        let mut new_join = join;
        let skinned = wall_mm > 0.0;
        ui.add_enabled_ui(skinned, |ui| {
            ui.horizontal(|ui| {
                row_label(ui, "Join:");
                if ui
                    .add(mode_button(join.is_plainly_hard(), "Hard"))
                    .on_hover_text("The fill stops dead at the skin.")
                    .on_disabled_hover_text(NO_SKIN)
                    .clicked()
                {
                    new_join = Fillet::hard();
                }
                for shape in Continuity::ALL {
                    let on = !join.is_plainly_hard() && join.continuity == shape;
                    if ui
                        .add(mode_button(on, shape.label()))
                        .on_hover_text(fillet_shape_hover(shape))
                        .on_disabled_hover_text(NO_SKIN)
                        .clicked()
                    {
                        new_join.continuity = shape;
                        if new_join.is_plainly_hard() {
                            new_join.radius = ParamRef::Literal((wall_mm * 0.5).max(0.1));
                        }
                    }
                }
            });
        });
        if !skinned || new_join.is_plainly_hard() {
            return new_join;
        }
        if new_join.radius.driver().is_some() {
            ui.label(
                egui::RichText::new("The join is what the ramp reads along the seam.").small().weak(),
            );
            return new_join;
        }
        let (edited, _) = self.parametric_field(
            ui,
            fillet_label(new_join.continuity),
            new_join.radius,
            0.05,
            FROM_NOUGHT_MM,
            "",
        );
        new_join.radius = edited;
        if let Some(note) = fillet_bend_note(new_join.continuity, self.resolve(new_join.radius)) {
            ui.label(egui::RichText::new(note).small().weak());
        }
        new_join.second = self.second_leg_rows(
            ui,
            new_join.continuity,
            new_join.radius,
            new_join.second,
            "On lattice:",
            "How far along the fill the bevel reaches from the skin. The number above is how far \
             along the skin it reaches from the fill.",
        );
        // Beside the number, which is the rule that travels with a `Fillet`.
        if self.part_graph().node(id).is_some()
            && let Some(why) = self.approximate_join(body.source().into_iter().chain(fill.source()))
        {
            ui.colored_label(ui.visuals().warn_fg_color, "\u{2248} approximate").on_hover_text(why);
        }
        new_join
    }

    /// What a closed skin, an empty cavity or a misplaced lattice mean, said on the block. None of
    /// them is an error: each is a part that would otherwise look like a failed tool.
    fn infill_notes(&mut self, ui: &mut egui::Ui, id: NodeId, wall_mm: f64) {
        let report = self.infill_cavity(ui.ctx(), id);
        let note = |ui: &mut egui::Ui, text: String| {
            ui.label(egui::RichText::new(text).small().weak());
        };
        let warn = ui.visuals().warn_fg_color;
        match report {
            Some(cavity) if !cavity.has_room(wall_mm) => {
                ui.colored_label(
                    warn,
                    format!(
                        "The wall fills the body, so there is nothing to infill. Thickest wall \
                         that leaves a cavity: {:.2} mm.",
                        // Rounded down, so the figure typed back in does leave one.
                        (cavity.deepest * 100.0).floor() / 100.0,
                    ),
                );
                return;
            }
            Some(cavity) if !cavity.fill_lands => {
                ui.colored_label(
                    warn,
                    "No lattice cell lands in the cavity; smaller cells, or move the lattice's frame.",
                );
            }
            Some(cavity) => note(ui, format!("About {:.0} % of the solid.", cavity.solid_share * 100.0)),
            None => {}
        }
        if wall_mm > 0.0 {
            ui.colored_label(warn, "Closed skin: the inside has no way out for powder or resin.")
                .on_hover_text(
                    "The skin is closed, so on a powder-bed or vat process what is inside stays \
                     inside. A hole cut through this block opens it, since a hole cuts the skin \
                     and the fill together.",
                );
        }
    }

    /// The cavity report for `id` at this revision, once it has settled and been worked out.
    ///
    /// **Never on the interface thread.** It compiles the body and samples it, which is a few
    /// milliseconds for a primitive and a bake for a cage, and a panel is drawn every frame.
    fn infill_cavity(
        &mut self,
        ctx: &egui::Context,
        id: NodeId,
    ) -> Option<fieldform_field::InfillCavity> {
        let revision = self.revision;
        let run = self.infill_cavities.entry(id).or_insert_with(|| CavityRun {
            revision,
            seen: std::time::Instant::now(),
            rx: None,
            asked: false,
            report: None,
        });
        if run.revision != revision {
            // The old answer stays up until the new one lands: a figure a moment stale beats one
            // that flickers through every step of a drag.
            run.revision = revision;
            run.seen = std::time::Instant::now();
            run.rx = None;
            run.asked = false;
        }
        if let Some(rx) = &run.rx {
            match rx.try_recv() {
                Ok(report) => {
                    run.report = report;
                    run.rx = None;
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(CAVITY_SETTLE),
                Err(mpsc::TryRecvError::Disconnected) => run.rx = None,
            }
        }
        let report = run.report;
        if run.asked {
            return report;
        }
        if run.seen.elapsed() < CAVITY_SETTLE {
            ctx.request_repaint_after(CAVITY_SETTLE);
            return report;
        }
        run.asked = true;
        let (tx, rx) = mpsc::channel();
        let graph = self.part_graph().clone();
        let doc = self.doc.clone();
        let spawned = std::thread::Builder::new().name("infill cavity".into()).spawn(move || {
            let _ = tx.send(fieldform_field::infill_cavity(&doc, &graph, id, CAVITY_SAMPLES));
        });
        if spawned.is_ok()
            && let Some(run) = self.infill_cavities.get_mut(&id)
        {
            run.rx = Some(rx);
            ctx.request_repaint_after(CAVITY_SETTLE);
        }
        report
    }

    /// **The wall is at least what was typed.** On a body whose field only bounds its distance
    /// the inner face sits at or past the number, never short of it, and that is said where the
    /// number is. A shell and an infill hollow a body the same way, so both panels call this.
    pub(crate) fn wall_minimum_note(&self, ui: &mut egui::Ui, body: Operand, wall: ParamRef) {
        let Some(body) = body.source() else { return };
        if fieldform_field::carries_true_distance(&self.doc, self.part_graph(), body) {
            return;
        }
        ui.colored_label(
            ui.visuals().warn_fg_color,
            format!("\u{2248} \u{2265} {:.2} mm", self.resolve(wall)),
        )
        .on_hover_text(
            "This body's field only bounds its distance, so the wall is measured along that \
             bound: it is never thinner than typed, and sometimes thicker.",
        );
    }
}
