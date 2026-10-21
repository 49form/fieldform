// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Bringing other formats in: the OBJ, STEP and IGES readers, and the rows an import leaves.

use crate::*;

impl App {
    /// Adds an import's solids as bodies of the active part, **in the file's own order**.
    ///
    /// One function for both importers rather than the same loop written twice (D9): STEP and OBJ
    /// were character-for-character identical here, so an order fixed in one would have been an
    /// order still random in the other.
    ///
    /// The solids are stamped before the graph is set, so the gate inside `set_part_graph` finds
    /// them already stamped and leaves them alone. That is what puts `file (1)` above `file (2)`
    /// in the block list and in the shader's own fold, and it is the position a re-import matches
    /// them by.
    ///
    /// Returns the first body, which is what the callers select: not all of them, because the
    /// manipulator moves one body and a selection of four hundred is a selection of nothing in
    /// particular.
    fn add_imported_bodies(
        &mut self,
        meshes: Vec<fieldform_scene::MeshData>,
        source: fieldform_scene::id::SourceId,
    ) -> Option<NodeId> {
        let mut graph = self.part_graph().clone();
        let mut added: Vec<NodeId> = Vec::with_capacity(meshes.len());
        for mut mesh in meshes {
            mesh.source = Some(source);
            let id = self.doc.add_mesh(mesh);
            added.push(graph.add(Node::Mesh(id)));
        }
        self.stamp_in_order(&added);
        self.set_part_graph(graph);
        added.first().copied()
    }

    /// One import, as a row that opens into the bodies it brought in.
    ///
    /// The eye and the delete act on the whole import, because that is what a person means by
    /// them here: forty solids from one file are one thing that was added and one thing to take
    /// away. Inside, each body is an ordinary row with its own eye, so a single part of an
    /// assembly can still be picked out — drawn, as every row is, from the list's `nesting`.
    pub(crate) fn import_block(
        &mut self,
        ui: &mut egui::Ui,
        source: fieldform_scene::id::SourceId,
        nesting: &fieldform_scene::Nesting,
    ) {
        let members = self.source_members(source);
        let record = self.doc.sources.get(&source).cloned();
        let name = record
            .as_ref()
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "Import".to_string());
        let surfaces: Vec<fieldform_scene::QuadMeshId> = record
            .as_ref()
            .map(|r| {
                r.surfaces
                    .iter()
                    .filter(|s| self.doc.quad_meshes.contains_key(s))
                    .copied()
                    .collect()
            })
            .unwrap_or_default();
        let curves: Vec<CurveId> = record
            .as_ref()
            .map(|r| r.curves.iter().filter(|c| self.doc.curves.contains_key(c)).copied().collect())
            .unwrap_or_default();
        // **The eye is the eye of what the import put on screen: its members that are bodies.** A
        // member something consumes -- the mesh under a placement, an import booleaned into a
        // part -- is drawn by its consumer, and its own eye means `shown`, so turning that on
        // would draw a second copy beside the first, where the placement is not. This eye used
        // to write `hidden` on every member, which never stopped a consumed one drawing either.
        //
        // A body is a root, which is a body-kind block nothing reads, so the part's consumed set
        // answers that and the eye's reading both, built once for a row drawn every frame
        // (`is_visible_among`).
        let part = self.part_graph();
        let consumed = part.consumed();
        let bodies: Vec<NodeId> = members
            .iter()
            .copied()
            .filter(|id| !consumed.contains(id) && part.node(*id).is_some_and(Node::is_body))
            .collect();
        let all_hidden = !bodies.is_empty()
            && bodies.iter().all(|id| !is_visible_among(&self.doc, part, &consumed, *id));
        let selected = self.selected_block == Some(outputs::BlockRef::Import(source));

        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("import", source.raw())),
            false,
        );
        state
            .show_header(ui, |ui| {
                // The original CAD, over the field it became. Offered only when there is
                // something to draw: an OBJ arrives as triangles with no B-rep behind them, and
                // a button that does nothing is worse than one that is absent.
                let has_edges = members
                    .iter()
                    .any(|id| self.mesh_under(*id).is_some_and(|m| !m.edges.is_empty()));
                if has_edges {
                    let on = self.showing_original.contains(&source);
                    if ui
                        .add(mode_button(on, "CAD"))
                        .on_hover_text(
                            "Draws the solid's own edges over the field derived from it: the \
                             geometry as the file has it, against what this build made of it.",
                        )
                        .clicked()
                    {
                        if on {
                            self.showing_original.remove(&source);
                        } else {
                            self.showing_original.insert(source);
                        }
                    }
                }
                // Only while there is a body for it to act on. Booleaned, patterned or mirrored into
                // something, every member is drawn by what consumes it and has its own eye in its
                // row; an eye here would read "Hide the import" and do nothing, which is what the
                // CAD button above is absent rather than offered to avoid.
                if !bodies.is_empty() {
                    let eye = if all_hidden { "\u{2013}" } else { "\u{1f441}" };
                    let tip = if all_hidden { "Show the import" } else { "Hide the import" };
                    if ui.small_button(eye).on_hover_text(tip).clicked() {
                        // One undo step for the whole import, as it is one click.
                        self.set_visible_all(&bodies, all_hidden);
                    }
                }
                // The folder, which is what makes this a block rather than a record of
                // something that happened: point it at a newer file and every body it produced
                // is replaced in place, so whatever was built on them re-evaluates.
                if ui
                    .small_button("\u{1f4c1}")
                    .on_hover_text("Load a different file in place of this one")
                    .clicked()
                {
                    self.reload_import(source);
                }
                // The name alone, which renames on a double-click and deletes from its menu or
                // with Delete, like every other block. What it brought in is underneath.
                let Some(row) = self.list_row(ui, RowKey::Import(source), &name, selected) else {
                    return;
                };
                if row.clicked() || row.secondary_clicked() {
                    self.select_block(outputs::BlockRef::Import(source));
                }
                row.context_menu(|ui| self.block_menu(ui, outputs::BlockRef::Import(source)));
            })
            .body(|ui| {
                ui.label(
                    egui::RichText::new(import_contents(members.len(), surfaces.len(), curves.len()))
                        .small()
                        .weak(),
                );
                // The file, how it was read, and what came out — one block, one story. The
                // settings are the ones this import was made with rather than whatever the
                // dialog is set to now, which is what makes re-reading it repeatable.
                if let Some(mut record) = record {
                    ui.label(
                        egui::RichText::new(
                            record.path.clone().unwrap_or_else(|| "(no path)".into()),
                        )
                        .small()
                        .weak(),
                    );
                    if let fieldform_scene::ImportSettings::Step(mut options) = record.settings {
                        let before = options;
                        ui.separator();
                        step_option_controls(ui, &mut options);
                        if options != before {
                            record.settings = fieldform_scene::ImportSettings::Step(options);
                            self.doc.sources.insert(source, record.clone());
                            self.mark_changed();
                        }
                        // Changing a tessellation setting *is* a re-read — the triangles are
                        // what it decides — but not on every frame of a drag, which is why it
                        // is a button rather than an effect of the number changing.
                        let known = record.path.clone().map(std::path::PathBuf::from);
                        let here = known.as_ref().is_some_and(|p| p.exists());
                        if ui
                            .add_enabled(here, egui::Button::new("Re-read with these"))
                            .on_hover_text(
                                "Reads the same file again at these settings, replacing the \
                                 bodies it produced. Anything built on them re-evaluates.",
                            )
                            .on_disabled_hover_text(
                                "The file is not where it was last read from — use the \
                                 folder to point at it",
                            )
                            .clicked()
                        {
                            if let Some(path) = known {
                                self.step_options = options;
                                self.import_step_into(&path, Some(source));
                            }
                        }
                    }
                    ui.separator();
                }
                for id in members {
                    self.node_block(ui, id, &[], nesting.references_of(id), false, nesting, None);
                }
                for id in surfaces {
                    self.surface_block(ui, id);
                }
                // Folded the way the list above is: a loop of the file's curves is one row
                // here too, and a curve of a loop headed elsewhere is shown there, not twice.
                for id in curves {
                    match self.folded_curve_row(id, nesting) {
                        Some(Block::Profile(key)) => self.profile_block(ui, key, nesting),
                        Some(_) => self.curve_block(ui, id, nesting.readers_of_curve(id)),
                        None => {}
                    }
                }
            });
    }

    /// Records an import that did not make meshes — a cage, a surface to follow, an IGES file's
    /// curves — so it is a block with its file and settings like every other import.
    fn record_import(&mut self, path: &std::path::Path, made: ImportMade) {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Imported".to_string());
        let settings = match ImportFormat::of(path) {
            Some(ImportFormat::Obj) => fieldform_scene::ImportSettings::Obj {
                unit_mm: self.obj_import_unit.in_millimetres(),
            },
            Some(ImportFormat::Iges) => fieldform_scene::ImportSettings::Iges,
            _ => fieldform_scene::ImportSettings::None,
        };
        let id = fieldform_scene::id::SourceId::new();
        self.doc.sources.insert(
            id,
            fieldform_scene::ImportSource {
                id,
                name,
                path: Some(path.display().to_string()),
                settings,
                cages: made.cages,
                surfaces: made.surfaces,
                curves: made.curves,
            },
        );
    }

    /// Reads a new file in place of an import, keeping the bodies it produced.
    ///
    /// Only STEP for now: an OBJ arrives as bodies too, but through a different reader, and the
    /// matching would need its own pass. Said rather than silently doing nothing.
    fn reload_import(&mut self, source: fieldform_scene::id::SourceId) {
        let Some(record) = self.doc.sources.get(&source).cloned() else { return };
        if !matches!(record.settings, fieldform_scene::ImportSettings::Step(_)) {
            self.warn("only a STEP import can be reloaded in place yet");
            return;
        }
        if !self.may_open_a_dialog("reload", "import file=<path>") {
            return;
        }
        let mut dialog =
            rfd::FileDialog::new().add_filter("STEP", &["step", "stp", "STEP", "STP"]);
        // Starting where it was last read from, which is usually where the revision is too.
        if let Some(dir) = record
            .path
            .as_ref()
            .map(std::path::PathBuf::from)
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.pick_file() else { return };
        // The settings the block carries, so a reload repeats how the file was read rather than
        // however the import dialog happens to be set now.
        if let fieldform_scene::ImportSettings::Step(options) = record.settings {
            self.step_options = options;
        }
        self.import_step_into(&path, Some(source));
    }

    /// The B-rep edges of every import currently being shown as original CAD, placed.
    ///
    /// Placed here rather than carried as a matrix, for the reason the imported triangles are:
    /// several imports can sit under different placements and a single MVP cannot say that.
    pub(crate) fn original_cad_edges(&self) -> Vec<Vec<Point3<f32>>> {
        let graph = self.part_graph();
        let mut out = Vec::new();
        for root in graph.roots() {
            let Some(source) = self.body_source(root) else { continue };
            if !self.showing_original.contains(&source) {
                continue;
            }
            // Down through any placements to the mesh, and back up for where it now sits.
            let mut id = root;
            let mesh = loop {
                match graph.node(id) {
                    Some(Node::Mesh(mesh)) => break Some(*mesh),
                    Some(Node::Transform { input, .. }) => match input.source() {
                        Some(below) => id = below,
                        // A hole ends the walk: there is no leaf down there to find.
                        None => break None,
                    },
                    _ => break None,
                }
            };
            let Some(mesh) = mesh.and_then(|m| self.doc.mesh(m)) else { continue };
            let place = self.placement_above(id);
            for line in &mesh.edges {
                out.push(
                    line.iter()
                        .map(|p| (place * Point3::new(p[0], p[1], p[2])).cast::<f32>())
                        .collect(),
                );
            }
        }
        out
    }

    /// Picks a file, then asks for that format's options before translating it — the format
    /// decided by the file rather than by the menu.
    ///
    /// Two steps rather than one because the options are per format and only the file says
    /// which format it is. Asking first would mean asking about all three, and throwing most
    /// of the answer away.
    ///
    /// Two menu items asking for two file types is a question the file already answers, and
    /// answering it wrongly only produced an error. The importers stay separate operations
    /// underneath, since they do genuinely different things, and a script may want to name one
    /// directly.
    pub(crate) fn import_file(&mut self) {
        if !self.may_open_a_dialog("import", "import file=<path>") {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Geometry", &["stp", "step", "obj", "igs", "iges"])
            .add_filter("STEP", &["stp", "step"])
            .add_filter("Wavefront OBJ", &["obj"])
            .add_filter("IGES", &["igs", "iges"])
            .pick_file()
        else {
            return; // user cancelled
        };
        match ImportFormat::of(&path) {
            Some(format) => self.import_prompt = Some(ImportPrompt { path, format }),
            None => {
                let what = path.extension().map(|e| e.to_string_lossy().to_string());
                self.say(match what {
                    Some(ext) => format!(
                        "{ext:?} is not a format this build imports — STEP, OBJ or IGES"
                    ),
                    None => "That file has no extension to read a format from".to_string(),
                });
            }
        }
    }

    /// The options for the file just chosen, and the button that starts the translation.
    ///
    /// Every setting here is one the *file does not carry*: an OBJ has no unit, and a STEP has
    /// no opinion about how finely to facet it or how hard to try to close it. Anything a file
    /// does state is read from it and is not offered.
    pub(crate) fn import_options_window(&mut self, ctx: &egui::Context) {
        let Some(prompt) = &self.import_prompt else { return };
        let (path, format) = (prompt.path.clone(), prompt.format);
        let mut go = false;
        let mut cancel = false;

        egui::Window::new(format!("{} import options", format.label()))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(file_name(&path)).strong());
                ui.separator();
                match format {
                    ImportFormat::Step => step_option_controls(ui, &mut self.step_options),
                    ImportFormat::Obj => {
                        ui.horizontal(|ui| {
                            ui.label("Unit:");
                            for unit in [
                                LengthUnit::Millimetre,
                                LengthUnit::Centimetre,
                                LengthUnit::Metre,
                                LengthUnit::Inch,
                            ] {
                                ui.selectable_value(
                                    &mut self.obj_import_unit,
                                    unit,
                                    unit.abbreviation(),
                                );
                            }
                        })
                        .response
                        .on_hover_text(
                            "What one unit in the file means. An OBJ does not say, and guessing \
                             wrong is a factor of a thousand that looks entirely plausible on \
                             screen. Most DCC tools write metres.",
                        );
                        ui.separator();
                        // The two things an OBJ can be, and the file cannot say which. A quad
                        // mesh from a subdivision modeller is a *control cage* and the surface
                        // is what it describes; a triangle mesh from anywhere else is the
                        // geometry itself. Reading one as the other is not a small difference
                        // — a cage is smoothed and editable, a model is neither.
                        ui.label("Read as:");
                        ui.horizontal(|ui| {
                            for (reading, label, hover) in ObjReading::ALL {
                                if ui
                                    .selectable_label(self.obj_as == reading, label)
                                    .on_hover_text(hover)
                                    .clicked()
                                {
                                    self.obj_as = reading;
                                }
                            }
                        });
                    }
                    // Nothing to ask. Said rather than left as an empty panel, which reads as
                    // something having failed to load.
                    ImportFormat::Iges => {
                        ui.label("IGES declares its own units, so there is nothing to set.");
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Import").clicked() {
                        go = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if go || cancel {
            self.import_prompt = None;
        }
        if go {
            match format {
                ImportFormat::Step => self.import_step_from(&path),
                ImportFormat::Obj => self.import_obj_from(&path),
                ImportFormat::Iges => self.import_iges_from(&path),
            }
        }
    }

    /// Opens a native file picker for an OBJ file and, if the user picks one, imports it as
    /// [`Self::import_obj_from`] does. Errors are surfaced in the UI rather than panicking,
    /// since a bad import is an expected user-facing outcome, not a bug.
    pub(crate) fn import_obj(&mut self) {
        if !self.may_open_a_dialog("importobj", "import file=<path>") {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Wavefront OBJ", &["obj"])
            .pick_file()
        else {
            return; // user cancelled
        };
        self.import_obj_from(&path);
    }

    /// Imports an OBJ as bodies, beside whatever else the document holds.
    ///
    /// **It used to replace the whole document**, and prompted to save first — the same discard
    /// an Open is. That was defensible when an OBJ could only arrive as *the* cage, and became
    /// destructive nonsense once anything else could be in the document: importing an OBJ after
    /// a STEP assembly asked to save, then deleted the assembly.
    ///
    /// It arrives as a mesh body now, which is what the import design note argues an
    /// import is: finished geometry to be referenced, cut against or blended with, not
    /// scaffolding to be pushed and pulled. That also widens what can be imported at all — a
    /// cage must be closed, manifold and all-quad, and most OBJs are none of those.
    ///
    /// Reading one *as a cage* is still possible and is now an explicit choice in the import
    /// options, where the fact that it replaces the document is stated rather than discovered.
    pub(crate) fn import_obj_from(&mut self, path: &std::path::Path) {
        match self.obj_as {
            ObjReading::Cage => return self.import_obj_as_cage(path),
            ObjReading::Surface => return self.import_obj_as_surface(path),
            ObjReading::Model => {}
        }
        let meshes = match fieldform_io::import_meshes_in(path, self.obj_import_unit) {
            Ok(meshes) if !meshes.is_empty() => meshes,
            Ok(_) => {
                self.say("that OBJ has no geometry in it");
                return;
            }
            Err(e) => {
                self.warn(e.to_string());
                return;
            }
        };

        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Imported".to_string());
        let count = meshes.len();
        let triangles: usize = meshes.iter().map(|m| m.triangles.len()).sum();
        let closed = meshes.iter().filter(|m| m.is_closed()).count();

        // One import, however many objects were in it — the same grouping a STEP assembly
        // gets, and for the same reason.
        let source = fieldform_scene::id::SourceId::new();
        self.doc.sources.insert(
            source,
            fieldform_scene::ImportSource {
                id: source,
                name: name.clone(),
                path: Some(path.display().to_string()),
                // The unit is the only thing an OBJ leaves open: the file declares none, so
                // what one unit meant is a reading rather than a fact, and it is the reading a
                // re-import has to repeat.
                settings: fieldform_scene::ImportSettings::Obj {
                    unit_mm: self.obj_import_unit.in_millimetres(),
                },
                cages: Vec::new(),
                surfaces: Vec::new(),
                curves: Vec::new(),
            },
        );

        let first = self.add_imported_bodies(meshes, source);
        self.select_body(first);
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        if let Some((center, radius)) = self.scene_bounds() {
            self.camera.frame(center, radius);
        }

        let bodies = if count == 1 { "body".to_owned() } else { format!("{count} bodies") };
        let mut said = format!(
            "Imported {name}: {bodies}, {triangles} triangles (as {})",
            self.obj_import_unit.abbreviation(),
        );
        // The unit cannot be read from the file, so the only way to know the right one was
        // chosen is to see the size it produced — 1.7 mm and 1679 mm are both plausible cubes.
        if let Some((lo, hi)) = fieldform_field::part_extent(&self.doc, self.part_graph()).as_box()
        {
            let size = hi - lo;
            said.push_str(&format!(
                " — {:.1} x {:.1} x {:.1} mm in all",
                size.x, size.y, size.z,
            ));
        }
        if closed < count {
            said.push_str(&format!(
                " — {closed} of {count} are solid; the rest can be seen but not cut with",
            ));
        }
        self.say(said);
    }

    /// Imports an OBJ as **surfaces to follow** — open quad sheets a conformal lattice can
    /// lay cells over.
    ///
    /// **Adds no body and no node**, which is what separates this from the other two doors. A
    /// surface is not geometry the document shows; it is geometry another block *references*,
    /// the way a curve is. So the import ends with something in `Document::quad_meshes` and the
    /// viewport unchanged — which would read as nothing having happened, and is why the log line
    /// says where to go next rather than only what arrived.
    ///
    /// The refusal a cage import gives is still the right one for a cage: closed, manifold and
    /// all quads is what Catmull-Clark needs to produce a solid. This asks for all of that
    /// except closed, because a panel or a shell is exactly the open surface somebody wants
    /// reinforced and it can never be a cage — an open surface has no inside for a sign to
    /// refer to.
    fn import_obj_as_surface(&mut self, path: &std::path::Path) {
        let surfaces = match fieldform_io::import_surfaces_in(path, self.obj_import_unit) {
            Ok(surfaces) if !surfaces.is_empty() => surfaces,
            Ok(_) => {
                self.say("that OBJ has no objects in it");
                return;
            }
            Err(e) => {
                self.warn(e.to_string());
                return;
            }
        };

        let count = surfaces.len();
        let faces: usize = surfaces.iter().map(|(_, m)| m.face_count()).sum();
        let open = surfaces.iter().filter(|(_, m)| m.is_open()).count();
        let names: Vec<String> = surfaces.iter().map(|(n, _)| n.clone()).collect();
        let mut made = Vec::new();
        for (name, mesh) in surfaces {
            made.push(self.doc.add_quad_mesh(mesh, name));
        }
        // **An import is a block, whatever it was read as.** The surfaces are listed inside it,
        // with the file they came from and how it was read, rather than as loose rows.
        self.record_import(path, made_surfaces(made));
        self.rebuild_surface_display();
        self.mark_changed();
        // Framed, because a surface is the only thing an import can add that the viewport was
        // not already looking at -- and an import you cannot see is what this whole block
        // exists to stop being.
        if let Some((centre, radius)) = self.scene_bounds() {
            self.camera.frame(centre, radius);
        }

        // Nothing changed on screen, so the line has to say both what arrived and what to do
        // with it. The closed count is worth reporting rather than hiding: a closed quad mesh is
        // a perfectly good thing to follow, and it is also a sign the file was probably meant to
        // be a cage.
        let what = if count == 1 {
            format!("Imported {} as a surface", names[0])
        } else {
            format!("Imported {count} surfaces")
        };
        let shape = match open {
            0 => format!("{faces} faces, all closed"),
            n if n == count => format!("{faces} faces, open"),
            n => format!("{faces} faces, {n} of {count} open"),
        };
        self.say(format!(
            "{what}: {shape} (as {}). Add a lattice and set its tiling to Conformal to lay \
             cells over it.",
            self.obj_import_unit.abbreviation(),
        ));
    }

    /// Imports an OBJ as subdivision cages, beside whatever else the document holds.
    ///
    /// This is what an OBJ import is *for*: the file is a control mesh, and what fieldform makes
    /// of it is the limit surface it describes — a body you can then push and pull, which is the
    /// direct layer the whole architecture is built around.
    ///
    /// **It used to replace the whole document**, and prompted to save first, because the
    /// document held exactly one cage and an imported one had to be it. Importing an OBJ after a
    /// STEP assembly therefore asked to save and then deleted the assembly. A document holds as
    /// many cages as it likes now — the traced view resolves each `BakedLeaf::Cage` against the
    /// document by id and bakes it over its own box at its own subdivision depth, which is what
    /// the single-cage assumption had been standing in the way of.
    fn import_obj_as_cage(&mut self, path: &std::path::Path) {
        let cages = match fieldform_io::import_cages_in(path, self.obj_import_unit) {
            Ok(cages) if !cages.is_empty() => cages,
            Ok(_) => {
                self.say("that OBJ has no objects in it");
                return;
            }
            Err(e) => {
                self.warn(e.to_string());
                return;
            }
        };

        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Imported".to_string());
        let count = cages.len();
        // Counted here because the loop below moves them into the document. Faces rather than
        // triangles: a cage is a quad control mesh, and its faces are what an OBJ's faces became.
        let faces: usize = cages.iter().map(|c| c.face_count()).sum();
        let points: usize = cages.iter().map(|c| c.vertex_count()).sum();

        let mut graph = self.part_graph().clone();
        let mut first = None;
        let mut made = Vec::new();
        for cage in cages {
            let id = self.doc.add_cage(cage);
            made.push(id);
            let node = graph.add(Node::Cage(id));
            first.get_or_insert((node, id));
        }
        self.record_import(path, made_cages(made));
        self.set_part_graph(graph);
        // The imported one becomes the cage being edited, because that is what a person has
        // just reached for. The others are bodies like any other until one of them is picked.
        if let Some((node, id)) = first {
            self.active_cage = id;
            self.select_body(Some(node));
        }
        self.selection.clear();
        self.hover = Selection::None;
        self.move_state = None;
        self.mark_changed();
        self.rebuild_render_resources();
        if let Some((centre, radius)) = self.scene_bounds() {
            self.camera.frame(centre, radius);
        }

        // The size that resulted, because the import unit cannot be read from the file: the
        // only way to know the right one was chosen is to see the number it produced, and
        // 1.7 mm and 1679 mm are both plausible-looking cubes on screen.
        let mut said = if count == 1 {
            format!(
                "Imported {name} as a cage: {faces} faces, {points} points (as {})",
                self.obj_import_unit.abbreviation(),
            )
        } else {
            format!(
                "Imported {name}: {count} cages, {faces} faces, {points} points (as {})",
                self.obj_import_unit.abbreviation(),
            )
        };
        let (lo, hi) = cage_box(self.cage());
        let size = hi - lo;
        said.push_str(&format!(
            " — {:.1} x {:.1} x {:.1} mm",
            size.x, size.y, size.z,
        ));
        self.say(said);
    }

    pub(crate) fn import_step(&mut self) {
        if !self.may_open_a_dialog("importstep", "import file=<path>") {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("STEP", &["stp", "step"])
            .pick_file()
        else {
            return; // user cancelled
        };
        self.import_step_from(&path);
    }

    /// Reads a STEP solid in as a body: a mesh in the document, and a node that references it.
    ///
    /// A body rather than a replacement for anything, unlike the OBJ path which takes over the
    /// cage being edited. An imported solid is finished geometry that arrives *beside* what is
    /// already there — the same rule as a swept pipe, and the reason a part holds bodies.
    pub(crate) fn import_step_from(&mut self, path: &std::path::Path) {
        self.import_step_into(path, None);
    }

    /// Reads a STEP file, replacing what `replacing` produced if it names an earlier import.
    fn import_step_into(
        &mut self,
        path: &std::path::Path,
        replacing: Option<fieldform_scene::id::SourceId>,
    ) {
        if self.pending_import.is_some() {
            self.say("an import is already running");
            return;
        }
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Imported".to_string());

        let Some(worker) = import_worker_path() else {
            self.say(format!(
                "the importer is missing — {IMPORT_WORKER} should sit beside this program",
            ));
            return;
        };
        // Named for the process, so two applications importing at once cannot collide, and
        // cleaned up when the import finishes however it finishes.
        let output = std::env::temp_dir()
            .join(format!("fieldform-import-{}.mesh", std::process::id()));

        let child = std::process::Command::new(&worker)
            .arg(format!("in={}", path.display()))
            .arg(format!("out={}", output.display()))
            .arg(format!("linear={}", self.step_options.linear_deflection_mm))
            .arg(format!("angular={}", self.step_options.angular_deflection_deg))
            .arg(format!("sew={}", self.step_options.sew_tolerance_mm))
            .arg(format!("fix={}", self.step_options.shape_fix))
            .arg(format!("parallel={}", self.step_options.parallel))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        match child {
            Ok(child) => {
                self.say(format!("importing {name}\u{2026}"));
                self.pending_import = Some(PendingImport {
                    name,
                    path: path.to_path_buf(),
                    replacing,
                    output,
                    child,
                    started: std::time::Instant::now(),
                });
            }
            Err(e) => self.warn(format!("could not start the importer: {e}")),
        }
    }

    /// Checks on a running import, and finishes it if it is done.
    ///
    /// Polled from `update` rather than waited on: the whole point is that the window keeps
    /// drawing, so this must never block. `try_wait` is what makes that true.
    pub(crate) fn poll_import(&mut self, ctx: &egui::Context) {
        let Some(pending) = &mut self.pending_import else { return };

        match pending.child.try_wait() {
            // Still going. Say how long, because a 30-second import with nothing on screen is
            // indistinguishable from one that has hung. Rewritten in place rather than appended:
            // a line a second would bury the history the console exists to keep.
            Ok(None) => {
                let seconds = pending.started.elapsed().as_secs();
                self.console_progress =
                    Some(format!("importing {}\u{2026} {seconds}s", pending.name));
                // Without this the window only redraws on input, so the elapsed time would
                // freeze and the application would look hung while working perfectly.
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
                return;
            }
            Ok(Some(status)) => {
                self.console_progress = None;
                let pending = self.pending_import.take().expect("checked above");
                self.finish_import(pending, status);
            }
            Err(e) => {
                self.console_progress = None;
                let pending = self.pending_import.take().expect("checked above");
                let _ = std::fs::remove_file(&pending.output);
                self.say(format!("lost track of the importer: {e}"));
            }
        }
    }

    /// Turns a finished worker process into a body, or into an explanation.
    fn finish_import(&mut self, mut pending: PendingImport, status: std::process::ExitStatus) {
        let elapsed = pending.started.elapsed().as_secs_f64();
        let mut stderr = String::new();
        if let Some(mut pipe) = pending.child.stderr.take() {
            use std::io::Read;
            let _ = pipe.read_to_string(&mut stderr);
        }

        if !status.success() {
            let _ = std::fs::remove_file(&pending.output);
            let reason = stderr.trim();
            // A worker that died without saying anything was killed rather than refused — an
            // abort inside OCCT, which is exactly the case a thread could not have survived.
            self.say(if reason.is_empty() {
                format!(
                    "{}: the importer stopped unexpectedly ({status}). The file is unchanged \
                     and so is your document.",
                    pending.name,
                )
            } else {
                format!("{}: {reason}", pending.name)
            });
            return;
        }

        let payload = std::fs::File::open(&pending.output)
            .map_err(fieldform_io::mesh_wire::WireError::from)
            .and_then(|f| fieldform_io::mesh_wire::read_payload(&mut std::io::BufReader::new(f)));
        let _ = std::fs::remove_file(&pending.output);

        // **Curves count as geometry.** A STEP file of construction lines holds no solid at all
        // and is exactly what somebody meant to import, so a payload with curves and no mesh is
        // a successful read rather than an empty one.
        let (meshes, imported_curves) = match payload {
            Ok((meshes, curves)) if !meshes.is_empty() || !curves.is_empty() => (meshes, curves),
            Ok(_) => {
                self.say(format!("{}: no geometry in that file", pending.name));
                return;
            }
            Err(e) => {
                self.say(format!("{}: {e}", pending.name));
                return;
            }
        };
        let curve_count = imported_curves.len();

        // One body per solid, which is what a STEP assembly is. Merging them would put every
        // touching face's edge into four triangles instead of two, and nothing in the result
        // would be closed — the whole reason the importer splits them.
        let count = meshes.len();
        let triangles: usize = meshes.iter().map(|m| m.triangle_count()).sum();
        let closed = meshes.iter().filter(|m| m.is_closed()).count();
        let tolerance = meshes.first().and_then(|m| m.tolerance_mm).unwrap_or_default();

        // A re-import replaces what an earlier read produced, in place. That is the whole point
        // of an import being a block: point it at a newer file and everything built on the
        // bodies it produced re-evaluates, rather than a second copy appearing beside the
        // first.
        //
        // **Matched by position**, which is what the importer's own naming allows: solids come
        // back as "file (1)", "file (2)" — numbered rather than named, because reading an
        // assembly's own solid names means its XCAF labels and the importer does not read them
        // yet. Position is right for a revision of the same assembly and wrong for one that
        // reordered its solids, so all three counts are reported and none of it is silent.
        if let Some(source) = pending.replacing {
            let existing = self.source_members(source);
            let mut graph = self.part_graph().clone();
            let (mut replaced, mut added) = (0usize, 0usize);

            for (i, mut mesh) in meshes.into_iter().enumerate() {
                mesh.source = Some(source);
                match existing.get(i) {
                    Some(root) => {
                        // A **new** mesh id under the **same** node. The node keeps its
                        // identity, so a shell or a boolean built on this body still points at
                        // it and simply re-evaluates. The id is new so that both caches — the
                        // compiled leaves and the baked bands, each keyed by mesh id — cannot
                        // hand back the geometry that was just replaced. Reusing the id would
                        // mean remembering to invalidate two caches, which is the kind of thing
                        // that rots.
                        let fresh = self.doc.add_mesh(mesh);
                        let mut at = *root;
                        let target = loop {
                            match graph.node(at) {
                                Some(Node::Mesh(old)) => break Some((at, *old)),
                                Some(Node::Transform { input, .. }) => match input.source() {
                                    Some(below) => at = below,
                                    None => break None,
                                },
                                _ => break None,
                            }
                        };
                        match target {
                            Some((node, old)) => {
                                if let Some(slot) = graph.node_mut(node) {
                                    *slot = Node::Mesh(fresh);
                                }
                                self.doc.meshes.remove(&old);
                                replaced += 1;
                            }
                            None => {
                                graph.add(Node::Mesh(fresh));
                                added += 1;
                            }
                        }
                    }
                    None => {
                        // The new file has solids the old one did not.
                        let fresh = self.doc.add_mesh(mesh);
                        graph.add(Node::Mesh(fresh));
                        added += 1;
                    }
                }
            }

            // Bodies the new file has no counterpart for. Removed when nothing reads them, and
            // **kept when something does**: leaving a graph that refers to nothing is worse
            // than leaving a body the file no longer has, and only the user can say which they
            // meant. Either way it is reported rather than discovered later.
            let mut dropped = 0usize;
            let mut orphaned = 0usize;
            for root in existing.iter().skip(count) {
                let consumed = graph.is_consumed(*root);
                if consumed {
                    orphaned += 1;
                } else {
                    graph.remove_body_and_its_own(*root);
                    dropped += 1;
                }
            }

            self.set_part_graph(graph);
            if let Some(record) = self.doc.sources.get_mut(&source) {
                record.name = pending.name.clone();
                record.path = Some(pending.path.display().to_string());
                record.settings = fieldform_scene::ImportSettings::Step(self.step_options);
            }
            self.mark_changed();
            self.view_mode = ViewMode::Field;
            self.rebuild_render_resources();

            let mut said = format!("Reloaded {}: {replaced} replaced", pending.name);
            if added > 0 {
                said.push_str(&format!(", {added} new"));
            }
            if dropped > 0 {
                said.push_str(&format!(", {dropped} gone"));
            }
            if orphaned > 0 {
                said.push_str(&format!(
                    " — {orphaned} kept because something is built on {}",
                    if orphaned == 1 { "it" } else { "them" },
                ));
            }
            self.say(said);
            return;
        }

        // One import, however many solids came out of it. Every mesh carries the id, which is
        // what lets the block list show them as one block, a re-import find what to replace,
        // and the settings the file was read with have somewhere to live.
        let source = fieldform_scene::id::SourceId::new();
        self.doc.sources.insert(
            source,
            fieldform_scene::ImportSource {
                id: source,
                name: pending.name.clone(),
                path: Some(pending.path.display().to_string()),
                settings: fieldform_scene::ImportSettings::Step(self.step_options),
                cages: Vec::new(),
                surfaces: Vec::new(),
                curves: Vec::new(),
            },
        );

        let first = self.add_imported_bodies(meshes, source);

        // Curves are document geometry rather than graph nodes, so they go straight in — the
        // same place a drawn curve lives, and immediately usable as a profile or a sweep. Given
        // fresh ids here rather than honouring the importer's, so importing one file twice
        // makes two sets instead of the second silently replacing the first.
        let mut made_curves: Vec<CurveId> = Vec::with_capacity(imported_curves.len());
        for curve in imported_curves {
            let id = fieldform_scene::id::CurveId::new();
            match curve.copy_as(id, curve.name.clone()) {
                Ok(fresh) => {
                    self.doc.curves.insert(id, fresh);
                    made_curves.push(id);
                }
                Err(_) => continue,
            }
        }
        // Placed now, as everything written past the history is (`block-ownership-spec.md`
        // §4.1).
        self.doc.settle_ownership();
        // In the file's order, like the solids above: a curve is a row in the same list.
        self.stamp_curves_in_order(&made_curves);
        if curve_count > 0 {
            self.rebuild_curve_display();
        }
        // The first body, so something is selected and the panel has something to show. Not
        // all of them: the manipulator moves one body, and a selection of four hundred would
        // be a selection of nothing in particular.
        self.select_body(first);
        if let Some(node) = first {
            self.boolean_b = Some(node);
            if self.boolean_a.is_none() || self.boolean_a == Some(node) {
                self.boolean_a =
                    self.doc.listed_bodies(self.part_graph()).into_iter().find(|b| *b != node);
            }
        }
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();

        // Framed, for the reason the IGES path already gives: a 2000 mm import arriving into a
        // view framed on a 100 mm cube is off screen, and that reads as the application being
        // broken rather than as a scale mismatch.
        if let Some((center, radius)) = self.scene_bounds() {
            self.camera.frame(center, radius);
        }

        let bodies = if count == 1 { "body".to_owned() } else { format!("{count} bodies") };
        // Said, because a file can hold curves and no solids and the count is then the whole of
        // what arrived -- and because curves landing silently in the curve list is how somebody
        // concludes the importer still ignores them.
        let bodies = match curve_count {
            0 => bodies,
            1 => format!("{bodies} and 1 curve"),
            n => format!("{bodies} and {n} curves"),
        };
        let mut said = format!(
            "Imported {}: {bodies}, {triangles} triangles at {tolerance:.3} mm in {elapsed:.1}s",
            pending.name,
        );
        // Which of them are solids, since that is what decides whether they can be cut with,
        // and the answer can differ between parts of one assembly.
        if closed < count {
            said.push_str(&format!(
                " — {} of {count} are solid; the rest can be seen but not cut with",
                closed,
            ));
        }
        self.say(said);
    }

    pub(crate) fn import_iges(&mut self) {
        if !self.may_open_a_dialog("importiges", "import file=<path>") {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("IGES", &["igs", "iges"])
            .pick_file()
        else {
            return; // user cancelled
        };
        self.import_iges_from(&path);
    }

    pub(crate) fn import_iges_from(&mut self, path: &std::path::Path) {
        match fieldform_io::import_iges(path) {
            Ok(import) => {
                let summary = import.summary();
                let first = import.curves.first().map(|c| c.id);
                let made: Vec<CurveId> = import.curves.iter().map(|c| c.id).collect();
                for curve in import.curves {
                    self.doc.curves.insert(curve.id, curve);
                }
                // Placed now, as everything written past the history is
                // (`block-ownership-spec.md` §4.1).
                self.doc.settle_ownership();
                // In the file's order, which is the order `made` is already in.
                self.stamp_curves_in_order(&made);
                self.record_import(path, made_curves(made));
                if self.selected_curve.is_none() {
                    self.selected_curve = first;
                }
                self.mark_changed();
                self.rebuild_curve_display();

                // Frame the imported geometry. Without this, a 2000 mm curve arrives into a
                // view framed on the 1.7 mm starting cube: the curve is off screen, the cube
                // is a speck, and every camera pivot raycast misses it. That reads as the
                // application being broken rather than as a scale mismatch.
                if let Some((center, radius)) = self.scene_bounds() {
                    self.camera.frame(center, radius);
                }

                // Offer a radius that suits what was just imported. A fixed 5 mm default is
                // 0.3 of a mesh cell on a 2 m curve, which meshes to nothing at all — the
                // generator looks like it does not work when it is only being asked for
                // something invisible.
                if let Some(size) = self.selected_curve_size() {
                    self.set_value(self.pipe_radius, tidy_mm(size * 0.025));
                    // A fillet is a detail on a join, so it starts well under the pipe it
                    // fillets — for the same reason the pipe radius is derived at all.
                    self.set_value(self.blend_radius, tidy_mm(size * 0.005));
                }

                self.say(summary);
            }
            Err(e) => self.warn(e.to_string()),
        }
    }

    /// Rebuilds the surface overlay when the document's surfaces are not the ones it was built
    /// from.
    ///
    /// **Checked every frame rather than called at the places that change the document**,
    /// because the first version was called at one of them — import — and File → Open left the
    /// previous document's panel drawn over the new one. The curve overlay is rebuilt by hand at
    /// a dozen call sites and has survived that by care; surfaces are immutable once imported, so
    /// their ids are the whole of what can change, and comparing a short sorted list costs less
    /// than remembering every door a document comes in through.
    ///
    /// **And by what they hold, since a surface's points can now be moved.** Ids alone were the
    /// whole of what could change while surfaces were immutable; a drag, an undo or a redo moves
    /// points under the same id, and the overlay would go on drawing where they were. A
    /// fingerprint per surface is a pass over its points each frame, which for a panel a lattice
    /// follows is nothing beside drawing its edges.
    pub(crate) fn refresh_surface_display(&mut self) {
        let mut prints: Vec<(fieldform_scene::QuadMeshId, u64)> = self
            .doc
            .quad_meshes
            .iter()
            .map(|(id, mesh)| (*id, mesh.geometry_fingerprint()))
            .collect();
        prints.sort_by_key(|(id, _)| id.raw());
        if prints != self.surface_prints {
            self.rebuild_surface_display();
            self.surface_prints = prints;
            // A hidden surface from the last document is not a thing to carry into this one.
            self.hidden_surfaces
                .retain(|id| self.doc.quad_meshes.contains_key(id));
        }
        // Deleted, undone out of existence, or a document opened over it: the tools go back to
        // the cage rather than editing a mesh nobody can see.
        if let ControlMesh::Surface(id) = self.control
            && !self.doc.quad_meshes.contains_key(&id)
        {
            self.set_control(ControlMesh::Cage);
        }
    }

    /// The wireframe drawn for each surface: every edge once.
    ///
    /// **The control mesh's edges, not the limit surface's.** What a conformal lattice divides
    /// is the quad, so the quad is what is worth seeing — the same argument that makes the cage
    /// rather than its limit surface the thing drawn in cage mode. Deduplicated by vertex pair,
    /// since every interior edge belongs to two faces.
    fn rebuild_surface_display(&mut self) {
        let started = std::time::Instant::now();
        let mut out = Vec::new();
        for (id, mesh) in &self.doc.quad_meshes {
            let mut seen = std::collections::HashSet::new();
            let mut edges = Vec::new();
            for face in mesh.face_ids() {
                let v = mesh.face_vertices(face);
                for k in 0..4 {
                    let (a, b) = (v[k], v[(k + 1) % 4]);
                    let key = (a.index().min(b.index()), a.index().max(b.index()));
                    if !seen.insert(key) {
                        continue;
                    }
                    let (pa, pb) = (mesh.vertex_position(a), mesh.vertex_position(b));
                    edges.push([
                        Point3::new(pa.x as f32, pa.y as f32, pa.z as f32),
                        Point3::new(pb.x as f32, pb.y as f32, pb.z as f32),
                    ]);
                }
            }
            out.push((*id, edges));
        }
        // HashMap order is arbitrary; sorted so the overlay does not reshuffle between frames.
        out.sort_by_key(|(id, _)| id.raw());
        self.surface_display = out;
        self.note_stage("the surface overlay", started);
    }
}
