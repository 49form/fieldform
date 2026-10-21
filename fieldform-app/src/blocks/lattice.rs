// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Lattice block's panel.
//!
//! Lifted out of `node_block_controls` whole. That function is a `match` over every kind of
//! block, and this one arm was 976 of its 2 629 lines — two thirds of a file that is already
//! the largest in the workspace, for one of twenty-six kinds. The arm is a panel, not a
//! decision the match needs to see, so it moves and the match keeps a line.
//!
//! **Why a module rather than another method in `main.rs`.** Fourteen cells over four cell maps,
//! each with its own numbers, is the largest block there is and the one most likely to grow
//! again; and privacy costs nothing here, because a child of the crate root can see the root's
//! private items. `use crate::*` is the whole of the wiring — no item had to be made
//! `pub(crate)` to be moved beside its neighbours.
//!
//! The other large arms are the same shape and follow it here, each a module of its own and
//! each its own move: one at a time is how a 2 629-line function gets read.

use crate::*;

/// Why this cell cannot be laid over this map, or `None` where it can.
///
/// **One place, because two sides of the panel ask it**: the cell list greys the cells a
/// conformal map cannot carry, and the tiling row greys the maps a cell cannot take. The
/// refusals themselves live in the field and the codegen (`CellCannotTakeThatMap`,
/// `CellCannotTakeAConformalMap`) and are by name there; this is the same fact said early
/// enough to stop the edit rather than to explain it afterwards. A lattice cell swap is not a
/// structural edit, so `replace_node` commits it without consulting `refuse_reason`, and the
/// document would then hold a block the viewport cannot draw.
pub(crate) fn cell_refuses_map(cell: LatticeCell, map: CellMap) -> Option<&'static str> {
    match (cell, map) {
        // A prism's walls are measured in millimetres across a flat period, and only a
        // rectangular map is one.
        (c, m) if c.is_prism() && !matches!(m, CellMap::Rectangular { .. }) => Some(
            "A honeycomb tiles a plane with a flat period, and only a rectangular map has one. \
             Use a rectangular map, and turn or place the lattice to aim it.",
        ),
        // A conformal map gives each face its own coordinates, so the integer cell a seed is
        // hashed from means something different either side of a face edge.
        (c, CellMap::Conformal { .. }) if c.is_stochastic() => Some(
            "A Voronoi cell finds its seeds by searching the cells around a point, and a \
             conformal map gives each face its own coordinates — so the search would stop \
             at every face edge. Rectangular, cylindrical and spherical all work.",
        ),
        _ => None,
    }
}

impl App {
    /// The Lattice block: which cell, how it tiles, how thick, and where it is measured from.
    ///
    /// Takes the node's four fields by value because the caller has already cloned the node —
    /// the panel writes a whole replacement `Node::Lattice` at the end rather than mutating in
    /// place, so nothing here borrows the graph across the drawing.
    pub(crate) fn lattice_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        cell: LatticeCell,
        frame: fieldform_scene::Frame,
        map: CellMap,
        phase: LatticePhase,
    ) {
        ui.label(
            "A lattice, tiled through space. On its own it fills the view; intersect \
             it with a body to infill that body.",
        );
        ui.separator();
        let mut new_cell = cell;
        ui.horizontal(|ui| {
            row_label(ui, "Cell:");
            // A list rather than a row of buttons. Three cells fitted across the panel;
            // six do not, and the row crowded and wrapped at the narrower widths the
            // divider allows. The tiling below stays as three buttons, because three is
            // what a row of them is good for — this is a judgement about how many, not a
            // rule that a vocabulary must be a list.
            egui::ComboBox::from_id_salt((id, "lattice_cell"))
                .selected_text(lattice_cell_label(new_cell))
                .show_ui(ui, |ui| {
                    for option in LATTICE_CELLS {
                        // **By name, not by value.** Two cells carry numbers now, so a
                        // foam jittered three-quarters is not equal to the foam in this
                        // list and `selectable_value` would tick the wrong row and
                        // overwrite the numbers on every click.
                        let chosen = new_cell.name() == option.name();
                        // **A pairing the compile refuses is not one to offer.** The
                        // refusal is by name on both sides, but a cell swap is not a
                        // structural edit, so `replace_node` commits it without asking
                        // `refuse_reason` -- and the document then holds a lattice the
                        // viewport cannot draw. Greyed with the reason rather than
                        // hidden: a cell missing from the list reads as a build that
                        // does not have it.
                        let refusal = cell_refuses_map(option, map);
                        let response = ui.add_enabled(
                            refusal.is_none(),
                            egui::Button::selectable(
                                chosen,
                                lattice_cell_label(option),
                            ),
                        );
                        if let Some(why) = refusal {
                            response.on_hover_text(why);
                            continue;
                        }
                        if response.clicked() && !chosen {
                            let mut picked = option;
                            // A cell's own numbers survive a change between two cells
                            // that both have them: swapping a foam for its beams draws
                            // something else from the same seeds, and re-rolling there
                            // would lose the arrangement the user was looking at.
                            if picked.is_stochastic()
                                && let (Some(j), Some(s)) =
                                    (new_cell.jitter().copied(), new_cell.seed())
                            {
                                *picked.jitter_mut().unwrap() = j;
                                *picked.seed_mut().unwrap() = s;
                            }
                            new_cell = picked;
                        }
                    }
                });
        });

        // **The first numbers a cell has had of its own**, and they are here beside the
        // cell rather than below with the sizes because that is what they are about:
        // the map says where the cells go, these say where the seed sits inside one.
        if new_cell.is_stochastic() {
            let (new_jitter, _) = self.parametric_field_in(
                ui,
                "Jitter:",
                *new_cell.jitter().unwrap(),
                0.01,
                0.0..=1.0,
                "How far a seed may wander from its cell's centre, as a fraction of the \
                 cell. Nought is a regular grid — boxes for a foam. One is as \
                 irregular as it goes, and no further: a seed never leaves its own \
                 cell, which is what keeps the field sound.",
                "",
            );
            *new_cell.jitter_mut().unwrap() = new_jitter;
            ui.horizontal(|ui| {
                row_label(ui, "Seed:");
                let seed = new_cell.seed().unwrap();
                ui.monospace(format!("{seed}"));
                // **Stepped rather than drawn from the clock.** Re-rolling until it
                // looks right is the workflow, and D7 asks that an arrangement be a
                // function of what was saved -- so the next seed is a function of this
                // one, and reopening a document and pressing this again walks the same
                // sequence. Knuth's MMIX multiplier; any full-period step would do.
                if ui
                    .button("Re-roll")
                    .on_hover_text(
                        "The next arrangement. The seed is saved, so what you settle on \
                         is what reopens.",
                    )
                    .clicked()
                {
                    *new_cell.seed_mut().unwrap() = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                }
            });
        }

        // Switching maps keeps the sizes it can and derives the ones it cannot, rather
        // than resetting to a default: the user picked those numbers for this part, and
        // a map change is a change of how it tiles, not of how big the cells are.
        let mut new_map = map;
        // Switching keeps the radial size across all three, since that is the one
        // number every map has: a change of tiling is a change of how the cells sit,
        // not of how big they are.
        let radial = match new_map {
            CellMap::Rectangular { cell } => cell,
            CellMap::Cylindrical { cell_radius, .. } => cell_radius,
            CellMap::Spherical { cell_radius, .. } => cell_radius,
            // A conformal map has no radial size at all -- its across-the-surface sizes
            // come from the mesh. The depth is its one length, so that is what carries.
            CellMap::Conformal { depth, .. } => depth,
        };
        ui.horizontal(|ui| {
            row_label(ui, "Tiling:");
            let is_rect = matches!(new_map, CellMap::Rectangular { .. });
            let is_cyl = matches!(new_map, CellMap::Cylindrical { .. });
            let is_sph = matches!(new_map, CellMap::Spherical { .. });
            if ui.add(mode_button(is_rect, "Rectangular")).clicked() && !is_rect {
                new_map = CellMap::Rectangular { cell: radial };
            }
            let sph_refused = cell_refuses_map(new_cell, CellMap::Spherical {
                cell_radius: radial,
                azimuth_count: 1,
                polar_count: 1,
            });
            let sph = ui.add_enabled(sph_refused.is_none(), mode_button(is_sph, "Spherical"));
            if let Some(why) = sph_refused {
                sph.on_hover_text(why);
            } else if sph.clicked() && !is_sph {
                let r_ref = self
                    .body_extent(id)
                    .as_box()
                    .map(|(lo, hi)| ((hi - lo).norm() * 0.25).max(1.0))
                    .unwrap_or(DEFAULT_BODY_MM * 0.5);
                let size = self.resolve(radial).max(0.001);
                let around = CellMap::arc_count_for(size, r_ref);
                new_map = CellMap::Spherical {
                    cell_radius: radial,
                    azimuth_count: around,
                    // Half as many pole to pole as around, since that arc is half as
                    // long — which is what keeps cells roughly square at the equator.
                    polar_count: (around / 2).max(1),
                };
            }
            // **Offered only when there is a surface to follow**, because the map
            // names one and a lattice with nothing to follow is not a state worth being
            // able to reach. The hover says where surfaces come from rather than
            // leaving a dead button to be puzzled over.
            let is_conf = matches!(new_map, CellMap::Conformal { .. });
            let surfaces = self.doc.quad_meshes_ordered();
            // A cell that cannot follow a surface, for the reason the cell list greys
            // the other way round.
            let conformal_refused = cell_refuses_map(new_cell, CellMap::Conformal {
                surface: surfaces.first().copied().unwrap_or_default(),
                cells_per_face: 2,
                depth: radial,
                thickness: None,
            });
            let response = ui.add_enabled(
                !surfaces.is_empty() && conformal_refused.is_none(),
                mode_button(is_conf, "Conformal"),
            );
            if let Some(why) = conformal_refused {
                response.on_hover_text(why);
            } else if surfaces.is_empty() {
                response.on_hover_text(
                    "Needs a surface to follow. File \u{2192} Import surface reads an \
                     open quad OBJ -- a panel or a shell -- which a cage import refuses \
                     because a cage has to be closed.",
                );
            } else if response.clicked() && !is_conf {
                new_map = CellMap::Conformal {
                    surface: surfaces[0],
                    // Two across a face, which shows the mesh is driving the layout
                    // without making the first result too fine to see.
                    cells_per_face: 2,
                    depth: radial,
                    // **Confined from the first click**, because the unconfined form is
                    // a tiling and not a part: it repeats away from the surface
                    // forever, so a first result with no shell is whatever box it
                    // happens to be cut by, which reads as the surface not being
                    // followed at all. One cell thick is the shallowest shell that
                    // shows a whole cell.
                    thickness: Some(radial),
                };
            }
            if ui.add(mode_button(is_cyl, "Cylindrical")).clicked() && !is_cyl {
                if true {
                    // A count that puts circumferential cells nearest the size already
                    // chosen, measured at the part's own radius so the first result is
                    // in the right neighbourhood rather than at some arbitrary default.
                    let r_ref = self
                        .body_extent(id)
                        .as_box()
                        .map(|(lo, hi)| ((hi - lo).norm() * 0.25).max(1.0))
                        .unwrap_or(DEFAULT_BODY_MM * 0.5);
                    let size = self.resolve(radial).max(0.001);
                    new_map = CellMap::Cylindrical {
                        cell_radius: radial,
                        cell_height: radial,
                        arc_count: CellMap::arc_count_for(size, r_ref),
                    };
                }
            }
        })
        .response
        .on_hover_text(
            "Rectangular fills a box. Cylindrical wraps the frame's Z axis, so the \
             pattern follows a bore or a shell. Spherical wraps its origin.",
        );

        new_map = match new_map {
            CellMap::Rectangular { cell } => {
                let (size, _) = self.parametric_field(
                    ui,
                    "Cell size:",
                    cell,
                    0.5,
                    POSITIVE_MM,
                    "One full period of the pattern",
                );
                CellMap::Rectangular { cell: size }
            }
            CellMap::Spherical { cell_radius, azimuth_count, polar_count } => {
                let (radius, _) = self.parametric_field(
                    ui,
                    "Cell radius:",
                    cell_radius,
                    0.5,
                    POSITIVE_MM,
                    "One period outward from the centre",
                );
                let mut around = azimuth_count as i32;
                let mut over = polar_count as i32;
                ui.horizontal(|ui| {
                    row_label(ui, "Around:");
                    ui.add(
                        egui::DragValue::new(&mut around)
                            .speed(0.25)
                            .range(1..=4096)
                            .suffix(" cells"),
                    )
                    .on_hover_text("Cells around the equator");
                });
                ui.horizontal(|ui| {
                    row_label(ui, "Pole to pole:");
                    ui.add(
                        egui::DragValue::new(&mut over)
                            .speed(0.25)
                            .range(1..=4096)
                            .suffix(" cells"),
                    )
                    .on_hover_text(
                        "Cells from one pole to the other. Whole for the same reason \
                         the equatorial count is: a fraction tears the seam.",
                    );
                });
                // The feature, not the wall: a solid lattice has no wall and still
                // closes up. Read through the same function the field uses so the
                // readout and the geometry cannot drift apart.
                let feature = fieldform_field::feature_size_mm(
                    phase.banding(),
                    self.resolve(*phase.param()),
                    self.resolve(radius),
                )
                .max(0.0);
                let r_min = around.max(over) as f64 * feature / std::f64::consts::TAU;
                ui.label(
                    egui::RichText::new(format!(
                        "Solid within {r_min:.2} mm of the centre, and cells close up \
                         toward the poles"
                    ))
                    .small()
                    .weak(),
                );
                CellMap::Spherical {
                    cell_radius: radius,
                    azimuth_count: around.max(1) as u32,
                    polar_count: over.max(1) as u32,
                }
            }
            CellMap::Conformal { surface, cells_per_face, depth, thickness } => {
                // Which surface, by name where the file gave one. A single surface still
                // gets the row: what a lattice is following is the first thing to want
                // to know about it, and a row that appears only sometimes is a row
                // nobody learns to look at.
                let ids = self.doc.quad_meshes_ordered();
                let mut chosen = surface;
                // By name, because the blocks above it are named and a picker that
                // says "36 faces" where the list says "panel" is two vocabularies for
                // one thing. The face count stays, after the name, since it is what
                // distinguishes two surfaces from the same file.
                let label_for = |doc: &Document, id: fieldform_scene::QuadMeshId| {
                    match doc.quad_meshes.get(&id) {
                        Some(m) => {
                            format!("{} ({} faces)", doc.surface_name(id), m.face_count())
                        }
                        None => "missing".to_string(),
                    }
                };
                ui.horizontal(|ui| {
                    row_label(ui, "Surface:");
                    egui::ComboBox::from_id_salt(("conformal surface", id))
                        .selected_text(label_for(&self.doc, chosen))
                        .show_ui(ui, |ui| {
                            for candidate in &ids {
                                ui.selectable_value(
                                    &mut chosen,
                                    *candidate,
                                    label_for(&self.doc, *candidate),
                                );
                            }
                        });
                });

                // A count and a whole number, for the reason `arc_count` is one: two
                // faces sharing an edge divide it this many times each, and only a whole
                // number makes their cells meet along it.
                let mut across = cells_per_face as i32;
                ui.horizontal(|ui| {
                    row_label(ui, "Across:");
                    ui.add(
                        egui::DragValue::new(&mut across)
                            .speed(0.1)
                            .range(1..=256)
                            .suffix(" per face"),
                    )
                    .on_hover_text(
                        "How many cells across every face of the surface, both ways. \
                         One count for the whole mesh, so cells meet across every \
                         shared edge without being solved for.",
                    );
                });
                let (through, _) = self.parametric_field(
                    ui,
                    "Cell depth:",
                    depth,
                    0.5,
                    POSITIVE_MM,
                    "One period through the surface — the only length here",
                );

                // **What makes this a part rather than a tiling.** The cells repeat
                // away from the surface as well as across it, so without a shell the
                // shape is whatever the lattice is intersected with -- a box gives a
                // box. Off is still worth having, and is not the same as a very large
                // number: it is what you want when the lattice is infilling a body that
                // already has a shape and the surface is only orienting the cells.
                let mut shelled = thickness.is_some();
                ui.horizontal(|ui| {
                    row_label(ui, "Shell:");
                    ui.checkbox(&mut shelled, "Follow the surface")
                        .on_hover_text(
                            "Keeps a shell of the surface this thick and discards the \
                             rest. Without it the pattern tiles outward forever and \
                             the shape is whatever you intersect it with.",
                        );
                });
                let shell = match (shelled, thickness) {
                    (true, Some(t)) => {
                        let (t, _) = self.parametric_field(
                            ui,
                            "Thickness:",
                            t,
                            0.5,
                            POSITIVE_MM,
                            "How thick a shell of the surface to keep",
                        );
                        Some(t)
                    }
                    // Just switched on: one cell through, which is the shallowest
                    // shell that contains a whole cell.
                    (true, None) => Some(through),
                    (false, _) => None,
                };
                // The limit the map has and the cylindrical map's degenerate radius do
                // not have a common name, but they are the same kind of fact: past it
                // the block stops describing what its controls say. A closest point is
                // ambiguous beyond the surface's own curvature radius -- two answers
                // the same distance away in different directions -- so a shell reaching
                // past it is not a thicker lattice, it is a confused one.
                //
                // **Measured rather than stood in for.** This used to compare the shell
                // against the smallest face span, on the reasoning that a quad's size is
                // the scale its curvature works at -- which is true of a mesh whose faces
                // shrink where it bends and says nothing about one whose faces are all
                // the same size. `mesh_fold_mm` reads the turn between neighbouring faces
                // over the span between them, which is a radius, and the kernel's own
                // `SurfaceMap::tightest_fold_mm` reads the same thing off the limit
                // surface it actually maps against.
                let fold = self
                    .doc
                    .quad_meshes
                    .get(&chosen)
                    .map(fieldform_field::mesh_fold_mm)
                    .filter(|f| f.is_finite());
                if let (Some(t), Some(fold)) = (shell, fold) {
                    let half = self.resolve(t).abs() * 0.5;
                    if half > fold {
                        ui.label(
                            egui::RichText::new(format!(
                                "Reaches {half:.1} mm either side, past the surface's \
                                 tightest fold at {fold:.1} mm — the cells cross \
                                 themselves there, and a mesh of it has holes at the fold",
                            ))
                            .small()
                            .color(ui.visuals().warn_fg_color),
                        )
                        .on_hover_text(
                            "Inside a fold, the nearest place on the surface is two \
                             places at once, so there is no single cell to lay down. \
                             Keep the thickness under twice the fold radius, or smooth \
                             the surface where it turns most sharply.",
                        );
                    }
                }
                // **The mesh is the grading**, which is the sentence that explains why
                // there is no cell size to type. Said with the number it produces, so it
                // can be checked rather than believed.
                if let Some(mesh) = self.doc.quad_meshes.get(&chosen) {
                    let count = across.max(1) as f64;
                    if let Some((small, large)) = face_span_mm(mesh) {
                        ui.label(
                            egui::RichText::new(format!(
                                "Cells across: {:.2} to {:.2} mm — the mesh sets \
                                 them",
                                small / count,
                                large / count,
                            ))
                            .small()
                            .weak(),
                        );
                    }
                    // **It repeats through the surface; it is not a shell of it.**
                    // The third cell argument is `2*pi*d/depth` and the cell function is
                    // `2*pi`-periodic, so the pattern at `d` and at `d + depth` is the
                    // same pattern -- it tiles outward forever, exactly as a rectangular
                    // lattice does. This row read "banded about it, one depth either
                    // side", which was wrong and made a flat result look like a fault.
                    // Said here because it decides what you have to intersect it with.
                    ui.label(
                        egui::RichText::new(
                            "Tiles through the surface as well as across it, so the \
                             shape is whatever you intersect it with.",
                        )
                        .small()
                        .weak(),
                    );
                }
                CellMap::Conformal {
                    surface: chosen,
                    cells_per_face: across.max(1) as u32,
                    depth: through,
                    thickness: shell,
                }
            }
            CellMap::Cylindrical { cell_radius, cell_height, arc_count } => {
                let (radius, _) = self.parametric_field(
                    ui,
                    "Cell radius:",
                    cell_radius,
                    0.5,
                    POSITIVE_MM,
                    "One period outward from the axis",
                );
                let (height, _) = self.parametric_field(
                    ui,
                    "Cell height:",
                    cell_height,
                    0.5,
                    POSITIVE_MM,
                    "One period along the axis",
                );
                // A count, not a length, and a drag rather than a text field so it
                // cannot be given a fraction. Going once around advances the pattern by
                // exactly this many cells, and the two sides of the seam only meet when
                // that is a whole number — a fraction puts a split down the full height
                // of the cylinder.
                let mut count = arc_count as i32;
                ui.horizontal(|ui| {
                    row_label(ui, "Around:");
                    ui.add(
                        egui::DragValue::new(&mut count)
                            .speed(0.25)
                            .range(1..=4096)
                            .suffix(" cells"),
                    )
                    .on_hover_text(
                        "How many cells go around the circumference. A whole number by \
                         necessity — a fraction would split the seam.",
                    );
                });
                let arc_count = count.max(1) as u32;

                // The readouts that make a cylindrical map predictable. Cells stretch
                // with radius — that is inherent to the map, not a defect — so the one
                // number that says whether the result will look right is the ratio
                // across the part, and it cannot be worked out by eye.
                let map_now = CellMap::Cylindrical {
                    cell_radius: radius,
                    cell_height: height,
                    arc_count,
                };
                if let Some((lo, hi)) = self.body_extent(id).as_box() {
                    let r_outer = lo.x.abs().max(hi.x.abs()).hypot(lo.y.abs().max(hi.y.abs()));
                    if let Some(outer) = map_now.arc_cell_mm(r_outer) {
                        ui.label(
                            egui::RichText::new(format!(
                                "Cell around: {outer:.2} mm at r = {r_outer:.1}"
                            ))
                            .small()
                            .weak(),
                        );
                    }
                }
                // A cylindrical map's two fixed extents are its cell radius and its
                // cell height; the circumferential one is the one that shrinks, which is
                // what this readout is about.
                let feature = fieldform_field::feature_size_mm(
                    phase.banding(),
                    self.resolve(*phase.param()),
                    self.resolve(radius).min(self.resolve(height)),
                )
                .max(0.0);
                let r_min = arc_count as f64 * feature / std::f64::consts::TAU;
                ui.label(
                    egui::RichText::new(format!(
                        "Solid within {r_min:.2} mm of the axis — cells have \
                         closed up by there"
                    ))
                    .small()
                    .weak(),
                );
                if arc_count <= 2 {
                    ui.label(
                        egui::RichText::new(
                            "One or two cells around the whole circumference is legal \
                             and almost certainly not what you meant.",
                        )
                        .small()
                        .color(egui::Color32::from_rgb(220, 170, 80)),
                    );
                }
                map_now
            }
        };

        // **A cell change can invalidate the phase, and the compiler cannot catch it.**
        // Swapping a gyroid for a beam leaves a wall thickness on something with no wall;
        // the document would be refused at compile with a clear message, but the panel
        // would have let it happen and shown a field that means nothing. So the phase
        // moves with the cell: kept when the new cell supports it, and seeded from the
        // cell's own first banding when it does not.
        let supported_by_new = fieldform_field::CellKind::from(new_cell)
            .supported_bandings();
        let mut new_phase = phase;
        if !supported_by_new.contains(&phase.banding()) {
            new_phase = seeded_phase(supported_by_new[0], self.resolve(radial));
        }

        // Which side of the surface is material. Offered from what the *cell* says it
        // supports rather than from a fixed pair, so a cell with only one side — a beam
        // lattice, when one arrives — is not offered a sheet it cannot make.
        //
        // One editor for the phase, not one per banding: the row below changes which
        // number it asks for, which is the whole visible difference between the two.
        let supported = fieldform_field::CellKind::from(new_cell).supported_bandings();
        if supported.len() > 1 {
            ui.horizontal(|ui| {
                row_label(ui, "Phase:");
                for option in supported {
                    let selected = new_phase.banding() == *option;
                    let label = match option {
                        Banding::Sheet => "Sheet",
                        Banding::Solid => "Solid",
                        Banding::Strut => "Strut",
                    };
                    if ui.add(mode_button(selected, label)).clicked() && !selected {
                        // Nothing carries across, and nothing can: there is no
                        // conversion between a wall of 2 mm and a density of 40%. So
                        // each side seeds its own start — a wall at a tenth of the cell,
                        // which is a lattice rather than a block of cheese, and a
                        // density a part is usually printed at.
                        //
                        // **A graded lattice stays graded.** The ramp's Out range is
                        // the number that does not carry across, so it is re-seeded
                        // the same way. This used to drop the ramp and leave it spare,
                        // and the Ramp button then picked it back up with a range in
                        // millimetres driving a density, which clamps to one: reported
                        // as "solid gyroid graded with ramp did not work".
                        let cell_mm = self.resolve(radial);
                        match new_phase.param().driver() {
                            Some(ramp) if self.regrade_lattice(id, ramp, *option, cell_mm) => {}
                            _ => new_phase = seeded_phase(*option, cell_mm),
                        }
                    }
                }
            });
        }

        // **One number, or a number that varies through space** -- the switch an
        // offset's distance and a pipe's radius have, because it is the same question.
        // What it drives is the phase's own number, which is what makes a graded wall
        // follow its ramp: grading through an offset grows a surface cell's wall by the
        // offset times the field's shortfall, and `graded-lattice-spec.md` measures a
        // gyroid asked for 5 mm reading 7.4.
        let driver = new_phase.param().driver();
        ui.horizontal(|ui| {
            row_label(ui, "Sized by:");
            if ui.add(mode_button(driver.is_none(), "One number")).clicked()
                && let Some(ramp) = driver
            {
                self.drop_driver(id, ramp);
            }
            if ui
                .add(mode_button(driver.is_some(), "Ramp"))
                .on_hover_text(
                    "A wall, density or strut that varies through the part — thin \
                     at one end and thick at the other, or thick near a point if the \
                     ramp is aimed at one.",
                )
                .clicked()
                && driver.is_none()
            {
                self.drive_lattice(id);
            }
        });
        if let Some(choice) = self.pick_driver(ui, "lattice_driver", driver) {
            match choice {
                Some(ramp) => self.wire_driver(id, 0, ramp),
                None => {
                    if let Some(ramp) = driver {
                        self.drop_driver(id, ramp);
                    }
                }
            }
        }
        if let Some(ramp) = driver
            && let Some(Node::Ramp { input, out_min, out_max, .. }) =
                self.part_graph().node(ramp).cloned()
        {
            // **Which block the grade is measured from, and a way to get to it.** The
            // ramp's own row is where that is chosen, and the block itself is what is
            // dragged or turned to aim the grade -- asked for as "pick the ramp's plane".
            // Shown here rather than repeated here, which is the rule the offset's
            // panel states: one block in two places drifts.
            ui.horizontal(|ui| {
                row_label(ui, "Measured from:");
                match input.source() {
                    Some(source) => {
                        ui.label(self.body_label(source));
                        if ui
                            .button("Select")
                            .on_hover_text(
                                "Selects the block the ramp measures from, so it can be \
                                 moved or turned to aim the grade. To measure from a \
                                 different block, open the ramp's own row and change \
                                 what it reads.",
                            )
                            .clicked()
                        {
                            self.select_body(Some(source));
                        }
                    }
                    None => {
                        ui.weak("nothing yet — set it in the ramp's own row");
                    }
                }
            });
            // A density is a fraction. A ramp left over from a wall, or picked from the
            // list, may run 1 to 3, which clamps to one and fills the cell.
            let (low, high) = (self.resolve(out_min), self.resolve(out_max));
            // A graded wall is a typed wall at every place, limit included: its thick
            // end is what decides whether the walls meet.
            if new_phase.banding() == fieldform_scene::Banding::Sheet
                && let fieldform_field::CellKind::Tpms(kind) =
                    fieldform_field::CellKind::from(new_cell)
            {
                self.wall_limit_note(ui, kind, low.max(high), self.resolve(radial));
            }
            if new_phase.banding() == fieldform_scene::Banding::Solid
                && (low.min(high) < 0.0 || low.max(high) > 1.0)
            {
                ui.horizontal(|ui| {
                    row_label(ui, "");
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!(
                            "\u{2248} the ramp's Out range is {low:.2} to {high:.2}; a \
                             density is 0 to 1"
                        ),
                    );
                });
            }
            // Each Out end is a density a part is built at somewhere, so each is held to the
            // floor a typed one is (ruling 11) -- by name, since a grade may run either way.
            if new_phase.banding() == fieldform_scene::Banding::Solid
                && let fieldform_field::CellKind::Tpms(kind) =
                    fieldform_field::CellKind::from(new_cell)
            {
                for (end, number) in [("Out-min", out_min), ("Out-max", out_max)] {
                    if let Some(typed) = number.resolve(&|p| self.doc.parameter_value(p))
                        && let Some(note) = lattice_readout::solid_floor_note(kind, typed)
                    {
                        self.solid_floor_row(ui, &format!("at the {end} end, {note}"));
                    }
                }
            }
        }
        if driver.is_some() {
            ui.horizontal(|ui| {
                row_label(ui, "");
                ui.weak(match new_phase.banding() {
                    fieldform_scene::Banding::Sheet => "The wall is the ramp's Out range, in mm",
                    fieldform_scene::Banding::Solid => "The density is the ramp's Out range, 0 to 1",
                    fieldform_scene::Banding::Strut => "The radius is the ramp's Out range, in mm",
                });
            });
        }

        // Both numbers are approximate and both say so where they are typed, which is
        // the rule the fillet radius already follows: where a number is entered is the
        // only place saying it is approximate is honest.
        //
        // Skipped while a ramp drives it: the number is the ramp's to state, and a field
        // here would be a second place to type it.
        new_phase = if driver.is_some() { new_phase } else { match new_phase {
            LatticePhase::Sheet { thickness, offset } => {
                // Nominal, and by how much is the cell's own business: the conversion
                // to an isovalue goes through an average gradient, and the measured
                // band (`TpmsKind::wall_band`) runs from a few per cent for a gyroid to
                // three times the number typed for a Neovius. This said "roughly a
                // tenth either way" for all six, which was true of two of them.
                let band = match fieldform_field::CellKind::from(new_cell) {
                    fieldform_field::CellKind::Tpms(kind) => Some(kind.wall_band()),
                    _ => None,
                };
                // A Voronoi wall errs one way only, which is a different sentence from
                // a band and wants saying differently -- and at *this* jitter, which is
                // the row two above this one and moves the answer by nearly a factor of
                // two across its range.
                let spread = match fieldform_field::CellKind::from(new_cell) {
                    fieldform_field::CellKind::Stochastic { kind, .. } => {
                        Some(kind.wall_spread(
                            new_cell.jitter().map_or(0.0, |j| self.resolve(*j)),
                        ))
                    }
                    _ => None,
                };
                let hover = match (band, spread) {
                    (Some((low, high)), _) => format!(
                        "Nominal — measured across the cell, the wall comes out \
                         between {:.0}% and {:.0}% of this, for a wall a tenth of the \
                         cell size. A thicker wall for its cell varies more.",
                        low * 100.0,
                        high * 100.0
                    ),
                    (_, Some(typical)) => format!(
                        "At least this, never less — the measure a Voronoi cell \
                         is built from understates the distance, so the wall comes out \
                         about {typical:.2}\u{d7} the number typed at this jitter, and \
                         thicker still at the corners where walls meet. Measured across \
                         the wall rather than estimated. More jitter, thicker wall."
                    ),
                    // A prism's wall comes off a distance, so there is no band to
                    // state: what is typed is what is built.
                    _ => "Exact — a honeycomb's wall is measured from a \
                          distance, so the thickness typed is the thickness built. The \
                          cell size is the hexagon's width across its flats."
                        .to_string(),
                };
                let (new_thickness, _) = self.parametric_field(
                    ui,
                    "Wall:",
                    thickness,
                    0.1,
                    POSITIVE_MM,
                    &hover,
                );
                // On the panel rather than only on hover when it is this far out: a
                // wall that reaches half again what was typed is a different part.
                let widest = match (band, spread) {
                    (Some((_, high)), _) => Some((
                        high,
                        "Where this surface flattens, its gradient collapses and the \
                         wall fattens. Measured by the chord across the wall rather \
                         than estimated from the slope, which understated it.",
                    )),
                    (_, Some(typical)) => Some((
                        typical,
                        "The measure understates the distance most where a point sits \
                         off the line joining two seeds, which is toward the edges of \
                         each wall — so the walls fatten where they meet. Never \
                         the other way: nowhere is thinner than the number typed.",
                    )),
                    _ => None,
                };
                if let Some((high, why)) = widest
                    && high > 1.5
                {
                    ui.horizontal(|ui| {
                        row_label(ui, "");
                        ui.colored_label(
                            ui.visuals().warn_fg_color,
                            format!("\u{2248} reaches {high:.1}\u{d7} this in places"),
                        )
                        .on_hover_text(why);
                    });
                }
                // **Where the wall stops being one.** Past a fraction of its cell a wall
                // has met its neighbour, and that fraction is the cell's own
                // (`TpmsKind::wall_limit`): a quarter for a gyroid, a tenth for a
                // Fischer-Koch S. Found when a graded one "went solid" and a typed one
                // turned out to have done the same, with nothing on the panel saying so.
                if let fieldform_field::CellKind::Tpms(kind) =
                    fieldform_field::CellKind::from(new_cell)
                {
                    self.wall_limit_note(
                        ui,
                        kind,
                        self.resolve(new_thickness),
                        self.resolve(radial),
                    );
                }
                // **The wall's middle, moved off the cell's own surface** -- backlog item 48,
                // what other CAD software calls a walled TPMS. A plain sheet divides the cell
                // two passages of the same size; an offset makes one larger than the
                // other, which is the point of it for two fluids. Nominal in the same
                // way the wall is, and the hover says where it runs out: past the
                // cell's own extremes there is nothing for the wall to sit on.
                let runs_out = match (band, new_map) {
                    (Some(_), CellMap::Rectangular { cell: size }) => {
                        let size = self.resolve(size);
                        match fieldform_field::CellKind::from(new_cell) {
                            fieldform_field::CellKind::Tpms(kind) if size > 0.0 => {
                                let table = kind.density_isovalues();
                                let scale = kind.argument_gradient().norm()
                                    * std::f64::consts::TAU
                                    / size;
                                let (a, b) = (table[0] / scale, table[10] / scale);
                                Some((a.min(b), a.max(b)))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                let offset_hover = match runs_out {
                    Some((low, high)) => format!(
                        "Moves the wall's middle off the cell's surface, so the two                                  passages it separates differ in size. Nought is a plain                                  sheet. Nominal, like the wall. Past about {low:.1} to                                  {high:.1} mm the wall leaves the cell and the lattice empties."
                    ),
                    None => "Moves the wall's middle off the cell's surface, so the two                                      passages it separates differ in size. Nought is a plain                                      sheet. Nominal, like the wall."
                        .to_string(),
                };
                // Offered only while a number sizes the wall; a graded wall keeps the
                // offset it had, which the row above already says is the ramp's turn.
                // A surface cell's, and nobody else's: a prism's wall has no
                // surface of the cell's own to sit off.
                let new_offset = match band {
                    Some(_) => {
                        self.parametric_field(
                            ui,
                            "Offset:",
                            offset,
                            0.1,
                            ANYWHERE_MM,
                            &offset_hover,
                        )
                        .0
                    }
                    None => offset,
                };
                LatticePhase::Sheet { thickness: new_thickness, offset: new_offset }
            }
            LatticePhase::Solid { density } => {
                // A fraction of each cell rather than a length, so it carries no unit and
                // does not scale with the cell size. Read through a measured per-cell
                // table, which is where the approximation comes from -- and the hover says
                // how far, from that table, rather than "within about 1%", which was false
                // below a tenth for all six cells and between tenths for three (item 94).
                let tpms = match fieldform_field::CellKind::from(new_cell) {
                    fieldform_field::CellKind::Tpms(kind) => Some(kind),
                    _ => None,
                };
                let hover = match tpms {
                    Some(kind) => lattice_readout::density_hover(
                        kind,
                        density.resolve(&|p| self.doc.parameter_value(p)),
                    ),
                    None => "Fraction of each cell that is material — 0.4 is 40%.".to_string(),
                };
                let (new_density, _) = self.parametric_field_in(
                    ui,
                    "Density:",
                    density,
                    0.005,
                    0.01..=0.99,
                    &hover,
                    "",
                );
                // **Below its saddle a solid is separate pieces**, on every map, since a
                // density is the same fraction whatever the map (ruling 11). Asked of the
                // document, so a parameter that fails says nothing rather than reading as
                // nought and claiming the part falls apart.
                if let Some(kind) = tpms
                    && let Some(typed) = new_density.resolve(&|p| self.doc.parameter_value(p))
                    && let Some(note) = lattice_readout::solid_floor_note(kind, typed)
                {
                    self.solid_floor_row(ui, &note);
                }
                LatticePhase::Solid { density: new_density }
            }
            LatticePhase::Strut { radius } => {
                // The one lattice number that is exact, and the hover says so rather
                // than leaving the absence of a warning to be noticed. A wall goes
                // through an average gradient and a density through a measured table;
                // this is subtracted from a distance.
                // A Voronoi beam is the exception: its measure understates, so the
                // beams come out thicker than asked and never thinner.
                let radius_hover = match fieldform_field::CellKind::from(new_cell) {
                    fieldform_field::CellKind::Stochastic { kind, .. } => {
                        let typical = kind.wall_spread(
                            new_cell.jitter().map_or(0.0, |j| self.resolve(*j)),
                        );
                        // **Stated against the radius, because that is the number
                        // in the box.** The measurement is a chord over the typed
                        // *diameter*, so quoting it beside a field labelled radius
                        // invites exactly the factor of two `LatticePhase::Strut`'s own
                        // doc warns about.
                        format!(
                            "At least this, never less — the measure a Voronoi \
                             cell is built from understates the distance, so a beam \
                             comes out about {:.1}\u{d7} this radius across at this \
                             jitter, rather than the 2\u{d7} an exact one would. \
                             Radius, not diameter.",
                            typical * 2.0,
                        )
                    }
                    _ => "Exact — the beams come out at the radius asked for. \
                          Radius, not diameter."
                        .to_string(),
                };
                let (new_radius, _) = self.parametric_field(
                    ui,
                    "Beam radius:",
                    radius,
                    0.05,
                    POSITIVE_MM,
                    &radius_hover,
                );
                LatticePhase::Strut { radius: new_radius }
            }
        } };
        // **What these cells are estimated to do** (item 94): the density they build, then
        // stiffness and strength read from published fits at it -- after the numbers it is read
        // from, and from the block as it is being edited this frame, so a dragged wall moves
        // the estimate under the hand rather than a frame later.
        let editing = Node::Lattice { frame: frame.clone(), cell: new_cell, map: new_map, phase: new_phase };
        self.lattice_estimate(ui, id, &editing, &mut new_phase);
        ui.label(
            egui::RichText::new(
                "A lattice's field is a lower bound, so a blend against one comes out \
                 larger than the radius asked for — allowed, and marked \
                 approximate where the radius is typed. Booleans are exact either way.",
            )
            .small()
            .weak(),
        );
        // Where the pattern is measured from and which way it wraps.
        //
        // The cylindrical map wraps this axis and the spherical map centres on this
        // origin, so for those two this is the control that decides what the lattice is
        // for — a bore, a boss, a shell around a point. Until it existed the only way to
        // aim one was a placement above it, in Euler degrees: the right mechanism and
        // the wrong vocabulary, since nobody knows a bore as three angles.
        ui.separator();
        let axis_label = match map {
            CellMap::Cylindrical { .. } => "Axis:",
            CellMap::Spherical { .. } => "Pole:",
            CellMap::Rectangular { .. } | CellMap::Conformal { .. } => "Up:",
        };
        // **A conformal map has nothing to aim.** Its cells are laid out on a surface
        // that carries its own position, so there is no direction left for a frame to
        // decide -- a frame here would move the whole pattern, which is what a
        // transform above the block already does and says more clearly. The row is
        // hidden rather than disabled: an empty control implies there is something
        // here to set.
        let new_frame = match map {
            CellMap::Conformal { .. } => frame,
            _ => self.frame_controls(ui, id, axis_label, frame, None),
        };
        if matches!(map, CellMap::Cylindrical { .. } | CellMap::Spherical { .. }) {
            ui.label(
                egui::RichText::new(
                    "Re-aiming changes which radius the cells wrap at, so check the                              count.",
                )
                .weak(),
            );
        }

        if new_cell != cell || new_map != map || new_phase != phase || new_frame != frame
        {
            self.replace_node(
                id,
                Node::Lattice {
                    frame: new_frame,
                    cell: new_cell,
                    map: new_map,
                    phase: new_phase,
                },
            );
        }
    }

    /// One warning row under a density, in the colour the wall limit uses: what is drawn is
    /// separate pieces, not the lattice it looks like.
    fn solid_floor_row(&self, ui: &mut egui::Ui, note: &str) {
        ui.horizontal(|ui| {
            row_label(ui, "");
            ui.add(
                egui::Label::new(
                    egui::RichText::new(format!("\u{2248} {note}"))
                        .small()
                        .color(ui.visuals().warn_fg_color),
                )
                .wrap(),
            )
            .on_hover_text(
                "A solid is one piece only once its isovalue passes the cell's saddle; below \
                 it the material is separate blobs, each touching nothing, with no stiffness \
                 to estimate and nothing to print as one part. Measured by tpms_profile.rs \
                 and pinned by a flood fill either side of the saddle.",
            );
        });
    }

    /// The estimate rows (item 94): what `lattice_readout` says for the block as it is being
    /// edited, and -- where a published fit describes this cell -- a target stiffness that
    /// writes the phase's own number once.
    ///
    /// **The same lines `estimate` prints**, from the one App-free function, so a script asks
    /// for exactly what this shows.
    fn lattice_estimate(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        editing: &Node,
        new_phase: &mut LatticePhase,
    ) {
        let lines = lattice_readout::lattice_readout(&self.doc, self.part_graph(), id, Some(editing));
        if lines.is_empty() {
            return;
        }
        let warn = ui.visuals().warn_fg_color;
        for line in &lines {
            ui.horizontal_top(|ui| {
                row_label(ui, line.label);
                let text = egui::RichText::new(&line.text).small();
                let text = match line.tone {
                    lattice_readout::Tone::Plain => text,
                    lattice_readout::Tone::Weak => text.weak(),
                    lattice_readout::Tone::Warn => text.color(warn),
                };
                let response = ui.add(egui::Label::new(text).wrap());
                if !line.hover.is_empty() {
                    response.on_hover_text(&line.hover);
                }
            });
            let under = |ui: &mut egui::Ui, text: egui::RichText| {
                ui.horizontal_top(|ui| {
                    row_label(ui, "");
                    ui.add(egui::Label::new(text).wrap());
                });
            };
            if let Some(detail) = &line.detail {
                under(ui, egui::RichText::new(detail).small().weak());
            }
            for caution in &line.warn {
                under(ui, egui::RichText::new(format!("\u{2248} {caution}")).small().color(warn));
            }
            for others in &line.others {
                under(ui, egui::RichText::new(others).small().weak());
            }
            if let Some(note) = &line.note {
                under(ui, egui::RichText::new(note).small().weak());
            }
        }
        // Offered wherever a published fit describes this cell and phase on a map whose cells
        // are cubes -- including a solid below its floor or a wall past its limit, which is
        // exactly when a number that does build is worth being offered.
        let offered = match editing {
            Node::Lattice { cell, map: CellMap::Rectangular { .. }, phase, .. } => {
                fieldform_field::default_row(
                    fieldform_field::CellKind::from(*cell),
                    phase.banding(),
                    fieldform_field::PropertyClass::Modulus,
                )
                .is_some()
            }
            _ => false,
        };
        if offered {
            self.lattice_target_rows(ui, id, editing, new_phase);
        }
    }

    /// A target stiffness, per end: what the default source needs, what the others need, and a
    /// Set that writes the default's number once through the ordinary edit path (ruling 13).
    ///
    /// **Never a stored drive, and never a rewrite of a parameter or a shared ramp**: Set is
    /// greyed with the reason instead. The target itself lives in the panel's memory, not the
    /// document -- it is a question, not a property of the part.
    fn lattice_target_rows(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        editing: &Node,
        new_phase: &mut LatticePhase,
    ) {
        let Node::Lattice { phase, .. } = editing else { return };
        let open_id = egui::Id::new((id, "lattice_target_open"));
        let mut open = ui.data(|d| d.get_temp::<bool>(open_id)).unwrap_or(false);
        ui.horizontal(|ui| {
            row_label(ui, "");
            if ui
                .add(mode_button(open, "Target stiffness\u{2026}"))
                .on_hover_text(
                    "Asks the published fits what the wall, density or radius would need for \
                     a stiffness you give, and writes it once when you press Set. Never stored \
                     as a drive: a corrected source would otherwise reshape every saved part \
                     sized by it.",
                )
                .clicked()
            {
                open = !open;
            }
        });
        ui.data_mut(|d| d.insert_temp(open_id, open));
        if !open {
            return;
        }
        let mut set: Option<(lattice_readout::SetWrite, String)> = None;
        for (slot, end) in lattice_readout::ends_of(phase).into_iter().enumerate() {
            let value_id = egui::Id::new((id, "lattice_target", slot));
            // A plain number in the panel's memory, not a `ParamRef`: nothing here is written to
            // the document until Set, so there is no shared value for a widget to rewrite.
            let mut percent = ui.data(|d| d.get_temp::<f64>(value_id)).unwrap_or(5.0);
            let answer = lattice_readout::number_for_target(
                &self.doc,
                self.part_graph(),
                id,
                Some(editing),
                end,
                percent / 100.0,
            );
            ui.horizontal(|ui| {
                row_label(
                    ui,
                    match end {
                        lattice_readout::End::Typed => "Target:",
                        lattice_readout::End::OutMin => "Target, Out-min:",
                        lattice_readout::End::OutMax => "Target, Out-max:",
                    },
                );
                ui.add(egui::DragValue::new(&mut percent).speed(0.05).suffix(" %"))
                    .on_hover_text("Percent of the solid's own modulus");
                match &answer {
                    Ok(a) => match &a.default.number {
                        Ok(n) => {
                            ui.label(
                                egui::RichText::new(format!(
                                    "of the solid's modulus \u{2192} {}",
                                    lattice_readout::number_text(a.word, *n, a.unit)
                                ))
                                .small(),
                            );
                            let button = ui.add_enabled(a.write.is_ok(), egui::Button::new("Set"));
                            // The caution goes with the number wherever it is offered, the
                            // button's own hover included: Set writes a number the source was
                            // never fitted at as readily as one it was.
                            let caution = match &a.default.caution {
                                Some(c) => format!(" It {c}."),
                                None => String::new(),
                            };
                            let button = match &a.write {
                                Ok(_) => button.on_hover_text(format!(
                                    "Writes it, once, as one undo step: {} by {}, an estimate.{caution}",
                                    lattice_readout::number_text(a.word, *n, a.unit),
                                    a.default.row.source.short
                                )),
                                Err(why) => button.on_disabled_hover_text(why.as_str()),
                            };
                            if button.clicked()
                                && let Ok(write) = &a.write
                            {
                                set = Some((
                                    write.clone(),
                                    format!(
                                        "Set the {} for {}% of the solid's modulus by {} — an estimate",
                                        lattice_readout::number_text(a.word, *n, a.unit),
                                        lattice_readout::typed_percent(percent / 100.0),
                                        a.default.row.source.short
                                    ),
                                ));
                            }
                        }
                        Err(why) => {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(format!(
                                        "of the solid's modulus: {} — {why}",
                                        a.default.row.source.short
                                    ))
                                    .small()
                                    .color(ui.visuals().warn_fg_color),
                                )
                                .wrap(),
                            );
                        }
                    },
                    Err(why) => {
                        ui.add(egui::Label::new(egui::RichText::new(why).small().weak()).wrap());
                    }
                }
            });
            // Beside the number, as the readout's own cautions are drawn under its lines.
            if let Ok(a) = &answer
                && let (Ok(_), Some(caution)) = (&a.default.number, &a.default.caution)
            {
                ui.horizontal_top(|ui| {
                    row_label(ui, "");
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!(
                                "\u{2248} {} {caution}",
                                a.default.row.source.short
                            ))
                            .small()
                            .color(ui.visuals().warn_fg_color),
                        )
                        .wrap(),
                    );
                });
            }
            if let Ok(a) = &answer
                && let Some(span) = lattice_readout::others_span(a)
            {
                ui.horizontal_top(|ui| {
                    row_label(ui, "");
                    ui.add(egui::Label::new(egui::RichText::new(span).small().weak()).wrap())
                        .on_hover_text(lattice_readout::offers_hover(a));
                });
            }
            let percent = if percent.is_finite() { percent.max(0.0) } else { 5.0 };
            ui.data_mut(|d| d.insert_temp(value_id, percent));
        }
        if let Some((write, said)) = set {
            match write {
                // The phase's own literal goes out with the rest of the block's edit at the end
                // of the panel: one replacement, one undo step.
                lattice_readout::SetWrite::Phase(phase) => *new_phase = phase,
                // A ramp's end is its own block, replaced on its own.
                lattice_readout::SetWrite::Ramp(ramp, node) => self.replace_node(ramp, node),
            }
            self.say(said);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_scene::QuadMeshId;

    /// The four maps, one of each kind. Their numbers do not matter to `cell_refuses_map`,
    /// which reads only which kind a map is.
    fn every_map() -> Vec<(&'static str, CellMap)> {
        vec![
            ("rectangular", CellMap::Rectangular { cell: ParamRef::Literal(20.0) }),
            (
                "cylindrical",
                CellMap::Cylindrical {
                    cell_radius: ParamRef::Literal(20.0),
                    cell_height: ParamRef::Literal(20.0),
                    arc_count: 24,
                },
            ),
            (
                "spherical",
                CellMap::Spherical {
                    cell_radius: ParamRef::Literal(20.0),
                    azimuth_count: 24,
                    polar_count: 12,
                },
            ),
            (
                "conformal",
                CellMap::Conformal {
                    surface: QuadMeshId::new(),
                    cells_per_face: 4,
                    depth: ParamRef::Literal(20.0),
                    thickness: None,
                },
            ),
        ]
    }

    /// A honeycomb's wall is measured across a flat period, and only the rectangular map has
    /// one — so it is the one map a prism can take.
    #[test]
    fn a_prism_cell_is_refused_by_every_map_but_the_rectangular_one() {
        for (name, map) in every_map() {
            let refusal = cell_refuses_map(LatticeCell::Honeycomb, map);
            if name == "rectangular" {
                assert!(refusal.is_none(), "a honeycomb should take a rectangular map");
            } else {
                let why = refusal.unwrap_or_else(|| panic!("a honeycomb should refuse {name}"));
                assert!(
                    why.contains("rectangular"),
                    "the refusal should say what to use instead, and said: {why}",
                );
            }
        }
    }

    /// A Voronoi cell hashes an integer cell index, and a conformal map gives each face its own
    /// coordinates — so that one map, and only that one, is refused.
    #[test]
    fn a_stochastic_cell_is_refused_by_the_conformal_map_and_no_other() {
        for cell in [
            LatticeCell::VoronoiFoam { jitter: ParamRef::Literal(0.75), seed: 0 },
            LatticeCell::VoronoiBeam { jitter: ParamRef::Literal(0.75), seed: 7 },
        ] {
            for (name, map) in every_map() {
                let refusal = cell_refuses_map(cell, map);
                assert_eq!(
                    refusal.is_some(),
                    name == "conformal",
                    "{cell:?} on a {name} map: {refusal:?}",
                );
            }
        }
    }

    /// The complement, over the whole list the panel offers: every cell that is neither a prism
    /// nor stochastic is taken by all four maps.
    ///
    /// Exhaustive on purpose. A cell added to `LATTICE_CELLS` without a rule here is either
    /// carried by every map — which this asserts — or needs a rule, and then this fails and
    /// says which cell.
    #[test]
    fn every_other_cell_the_panel_offers_is_taken_by_every_map() {
        let mut checked = 0;
        for cell in LATTICE_CELLS {
            if cell.is_prism() || cell.is_stochastic() {
                continue;
            }
            checked += 1;
            for (name, map) in every_map() {
                assert!(
                    cell_refuses_map(cell, map).is_none(),
                    "{cell:?} should take a {name} map, and refused it",
                );
            }
        }
        assert_eq!(
            checked,
            LATTICE_CELLS.len() - 3,
            "the panel offers one prism and two stochastic cells; if that changed, so has this",
        );
    }
}
