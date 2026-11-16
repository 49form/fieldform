// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The active part's graph: reading it, replacing it as an undo step, and its material.

use crate::*;

impl App {
    /// The composition being edited: the active part's graph.
    ///
    /// The app used to hold the applied generator in a field of its own and rebuild a graph
    /// from it for each consumer, which meant the thing on screen was not the thing in the
    /// document — so saving the document would have written out a part that was missing the
    /// composition the user was looking at. The graph in the document is the only copy now,
    /// and the UI derives its controls from it rather than the other way round.
    pub(crate) fn part_graph(&self) -> &Graph {
        &self
            .doc
            .parts
            .get(&self.active_part)
            .expect("active_part must name a part in the document")
            .graph
    }

    /// The active part's graph, to write during a gesture.
    ///
    /// Not a way round [`Self::set_part_graph`]: a drag moves the document live for feedback
    /// and puts *one* entry on the history when the hand comes off, which is what every drag
    /// here does, and what it settles at is committed from the graph this returns.
    pub(crate) fn part_graph_mut(&mut self) -> &mut Graph {
        self.document_moved();
        &mut self
            .doc
            .parts
            .get_mut(&self.active_part)
            .expect("active_part must name a part in the document")
            .graph
    }

    /// Puts the active part's graph back to `was`, off the history — a drag abandoned, or one
    /// put back to where it began so the command that commits it can redo it — and settles who
    /// owns what.
    ///
    /// **Every graph write settles** (`block-ownership-spec.md` §4.1), and these are the ones no
    /// command makes. What they put back differs from the live graph only in the numbers a drag
    /// moved, so the settle finds nothing to do today; it is here so that a restore which ever
    /// differs in structure cannot leave a block stored inside one that no longer reads it.
    pub(crate) fn restore_part_graph(&mut self, was: Graph) {
        *self.part_graph_mut() = was;
        self.doc.settle_ownership();
    }

     /// Replaces the active part's graph, and records it for undo.
    ///
    /// **The one gate every graph edit passes through**, which is what makes this the place to
    /// record them. Adding a body, combining two, unapplying an operation and dragging a radius
    /// all end up here, and until now none of them was undoable: Ctrl+Z reversed the last cage
    /// vertex you moved and skipped straight past everything else you had done since.
    ///
    /// A drag lands as **one** entry, not one per frame. A `DragValue` writes a new graph every
    /// frame it is held, and thirty entries for one gesture makes undo useless in the other
    /// direction — you would press it thirty times to get back where you started. So a
    /// continuation of a gesture replaces the entry describing the previous instant of it,
    /// keeping the graph the gesture *started* from as its "before".
    ///
    /// A continuation is: soon after the last edit, and **structurally identical** to it. The
    /// second half is what keeps the rule honest — dragging a number changes values, so the
    /// nodes and the edges between them are the same graph, while adding or deleting a body
    /// changes the shape and starts a new entry however fast you were clicking. Time alone
    /// would merge two deliberate acts that happened to be quick. And an undo or a redo ends the
    /// gesture, since after either the entry on top is not its own (`step_history`).
    pub(crate) fn set_part_graph(&mut self, graph: Graph) {
        self.set_part_graph_with(graph, Vec::new());
    }

    /// [`Self::set_part_graph`], with `then` in the same undo entry after the graph edit
    /// ([`record_graph_edit_then`]): for an act the graph edit is only half of, as the eye Mirror
    /// turns on for the body it consumes is (`block-ownership-spec.md` §7.3).
    ///
    /// The bookkeeping is every graph edit's, since this is one: the splash, the end of a material
    /// drag, and the stamp on what it made. With anything to follow, never the next instant of a
    /// gesture, and the end of one. With nothing, this is `set_part_graph` exactly, and a drag's
    /// next instant folds into the entry on top as that says: an empty `then` is not how an edit
    /// asks for an entry of its own.
    pub(crate) fn set_part_graph_with(&mut self, graph: Graph, then: Vec<Box<dyn fieldform_ops::Command>>) {
        // A lock holds here whatever the edit was and whether or not it asked first: this is
        // the one place every one of them arrives.
        if let Some(why) = self.lock_refusal(&graph) {
            self.warn(why);
            return;
        }
        // Any change to what the part is made of means the splash has been read.
        self.splash_shown = false;
        // A graph edit ends a material drag: the two must not fold into one undo entry, or
        // undoing the one would take back the other.
        self.material_gesture = None;
        // Every instant of a manipulator drag comes through here, and only its end is marked.
        self.document_moved();

        let part = self.active_part;
        // With whatever a field's V or × did to a parameter on the way here, so the two are one
        // undo (`ParameterEdits`).
        let done = record_graph_edit_then(
            &mut self.stack,
            &mut self.doc,
            &mut self.gesture,
            &mut self.parameter_edits,
            part,
            graph,
            then,
        );
        if let Err(e) = done {
            // The command only fails when the part is gone, which would mean the application
            // asked to edit something it had already removed. Said rather than swallowed.
            self.warn(e.to_string());
            return;
        }
        self.stamp_new_nodes();
        // Every graph edit comes through here, a cage block deleted among them: the cage the
        // tools act on is re-read against the part as it now is.
        self.resync_active_cage();
    }

    /// Acts on what settling moved since this was last asked (`block-ownership-spec.md` §4.2):
    /// a block that became a variable with no name is given one, and what moved is said.
    ///
    /// **Here and not in the commands that settle**, because the name a variable should carry
    /// ("Cube", not "Cuboid") resolves parameters, which is the application's business, so
    /// settling writes no name. Asked at the top of every frame, so the list and the console have
    /// caught up before anything draws; after every line of a script, which runs its lines in one
    /// frame, so a line can name what the line before it made a variable; and before every save,
    /// so no file holds a variable that cannot be typed (I4).
    ///
    /// **Which blocks are named is `variables_to_name`'s**, in the order they were made
    /// (`unnamed_variables_among`), so of two spheres the first made is `Sphere`. **Off the
    /// history**: undoing the edit that made a variable leaves the name it was given, since a
    /// name is a label (§4.3).
    ///
    /// `quietly` after a load (§9.2): nothing is said and the document is not marked changed,
    /// since nothing moved and what is in memory is what was opened.
    pub(crate) fn drain_ownership_changes(&mut self, quietly: bool) {
        // Most frames: nothing settled since the last one, so nothing was declared either.
        if self.doc.pending_ownership_changes.is_empty() && !quietly {
            return;
        }
        let changes = std::mem::take(&mut self.doc.pending_ownership_changes);
        let unnamed = variables_to_name(&self.doc, quietly);
        for id in &unnamed {
            let name = self.unique_variable_name_in(self.graph_holding_or_active(*id), *id);
            self.doc.node_meta.entry(*id).or_default().name = Some(name);
        }
        if quietly {
            return;
        }
        if !unnamed.is_empty() {
            self.mark_changed();
        }
        let news = ownership_news(&self.doc, &changes, |block| match block {
            BlockKey::Node(id) => self.body_label_in(self.graph_holding_or_active(id), id),
            BlockKey::Curve(id) => {
                self.doc.curve(id).map_or_else(|| "A curve".to_string(), |c| c.name.clone())
            }
        });
        for line in news {
            self.say(line);
        }
    }

    /// The graph of whichever part holds `id`, or the active part's when none does: settling
    /// reaches every part, so what it moved need not be on screen.
    pub(crate) fn graph_holding_or_active(&self, id: NodeId) -> &Graph {
        self.graph_holding(id).unwrap_or_else(|| self.part_graph())
    }


    /// Sets the active part's material, folding a drag into one undo entry.
    ///
    /// `None` clears it back to the default. The same gesture rule `set_part_graph` uses: an
    /// edit within `GESTURE_WINDOW` of the last one continues it, so a slider drag goes on the
    /// history once, as the material it settled at, rather than sixty times a second.
    pub(crate) fn set_part_material(&mut self, material: Option<Material>) {
        let part = self.active_part;
        let current = self.doc.parts.get(&part).and_then(|p| p.material.clone());
        if current == material {
            return;
        }
        // And a material edit ends a graph gesture, for the same reason the reverse holds.
        self.gesture = None;
        let continues = self
            .material_gesture
            .as_ref()
            .is_some_and(|g| g.part == part && g.at.elapsed() < GESTURE_WINDOW);
        let before = match (&self.material_gesture, continues) {
            (Some(g), true) => g.before.clone(),
            _ => current,
        };
        self.material_gesture = Some(MaterialGesture {
            part,
            before: before.clone(),
            at: std::time::Instant::now(),
        });
        let cmd = Box::new(SetPartMaterial::new(part, before, material));
        let done = if continues {
            self.stack.replace_top(&mut self.doc, cmd)
        } else {
            self.stack.execute(&mut self.doc, cmd)
        };
        if let Err(e) = done {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
    }
}
