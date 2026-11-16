// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Which bodies are drawn, and showing and hiding them.

use crate::*;

impl App {
    /// The selected body's place among the ones being drawn, which is the order the shader's
    /// union fold uses — so a pixel's `Hit::body` compares directly against it.
    ///
    /// Derived rather than stored, like every other read of the graph: a stored index goes
    /// stale the moment a body is added, removed or hidden. Counted without building the
    /// display graph, because this is read once a frame and that clones — the mistake that put
    /// a whole graph compile in the frame loop, made smaller.
    pub(crate) fn selected_body_index(&self) -> Option<u32> {
        let id = self.selected_body?;
        // Read from `drawn_bodies` rather than recounted here. The two used to be the same
        // walk written twice, which was safe while the order was simply `roots()`; a drawn
        // variable is a copy under a random id, so it is not any more.
        self.drawn_bodies()
            .iter()
            .position(|drawn| *drawn == id)
            .map(|i| i as u32)
    }

    /// The part's graph with the hidden bodies taken out — what the viewport draws, picks
    /// against, and sizes its bake to: [`fieldform_field::display_graph_with_stand_ins`], told
    /// which bodies this application is drawing as a mesh.
    pub(crate) fn display_graph_with_stand_ins(&self) -> (Graph, std::collections::HashMap<NodeId, NodeId>) {
        fieldform_field::display_graph_with_stand_ins(&self.doc, self.part_graph(), &|id| {
            self.drawn_as_mesh(id)
        })
    }

    /// What the viewport draws, and nothing else. Most callers want only this.
    pub(crate) fn display_graph(&self) -> Graph {
        self.display_graph_with_stand_ins().0
    }

    /// The bodies the viewport draws, in the order it draws them, named by the block each is.
    ///
    /// **The one reading of that order.** `selected_body_index` turns it into the number the
    /// shader is given to highlight one, and `display_graph` is what actually gets drawn -- so
    /// the two have to be the same list or the wrong body lights up. It used to be safe to
    /// recount it by hand because the order was just `roots()`; it is not any more, because a
    /// drawn variable is a *copy* under a new id and **ids are random, not sequential**
    /// (`id.rs` says why). Counting the copies as "after the bodies" looked obviously right and
    /// would have highlighted the wrong body at random.
    ///
    /// Through `Document::folded_bodies`, which is the same function the shader's fold asks, so the
    /// index handed to the shader and the index the shader assigned are one number by
    /// construction rather than by two sorts agreeing. A stand-in has no stamp of its own --
    /// its id is derived, not minted -- and sorts where the block it stands for sorts, which is
    /// `folded_bodies`'s job and stated there.
    pub(crate) fn drawn_bodies(&self) -> Vec<NodeId> {
        // **Nothing shown, nothing to build.** This is read once a frame, and building the
        // display graph clones the part's -- which is what the note in `selected_body_index`
        // has always asked callers not to do per frame. With no variable on screen the drawn
        // list is exactly the roots that pass the two exclusions, which needs no clone and is
        // what this cost before the feature existed. Every document that does not use it pays
        // about what it paid.
        //
        // That holds only while each body's eye is a lookup, so the eye is read from the part's
        // consumed set built once here, never asked of the graph a body at a time
        // (`is_visible_among`): one more pass over the graph a frame, where asking per body was
        // a pass per body. The flags are the part's in both branches, as they always were.
        let shown = self.shown_ingredients();
        let part = self.part_graph();
        let consumed = part.consumed();
        if shown.is_empty() {
            // **A body with an empty slot is pruned, the fourth term of the same predicate**
            // (`notebook-spec.md` §7.2, the `is_reference_body` precedent: one body that
            // cannot compile used to take the whole viewport with it). Said here as well as in
            // `display_graph_with_stand_ins` because this is the hand-written copy of that
            // predicate, and the two lists have to be the same list or the wrong body lights
            // up -- the copy is the whole hazard, and it has already been wrong once: this term
            // was added here while the predicate it copies still had three. Closed downstream by
            // `incomplete`, so a block reading a holed one goes with it; one pass over the graph
            // a frame, beside the `consumed` one above. Of the document, as the predicate it
            // copies is, so a red cage's body is left out here too.
            let holed = self.doc.incomplete_in(part);
            return self
                .doc
                .folded_bodies(part)
                .into_iter()
                .filter(|id| {
                    is_visible_among(&self.doc, part, &consumed, *id)
                        && !self.is_reference_body(*id)
                        && !self.drawn_as_mesh(*id)
                        && !holed.contains(id)
                })
                .collect();
        }
        let _ = shown;
        let (graph, stand_ins) = self.display_graph_with_stand_ins();
        self.doc
            .folded_bodies(&graph)
            .into_iter()
            .filter(|id| {
                // A stand-in is drawn because it was asked for; the exclusions below are about
                // bodies, and the block it stands for has already passed them.
                stand_ins.contains_key(id)
                    || (is_visible_among(&self.doc, part, &consumed, *id)
                        && !self.is_reference_body(*id)
                        && !self.drawn_as_mesh(*id))
            })
            .map(|id| stand_ins.get(&id).copied().unwrap_or(id))
            .collect()
    }

    /// Gathers the imported meshes the tracer is *not* drawing, as triangles.
    ///
    /// A mesh the tessellator left holes in cannot be a signed field — there is no inside for
    /// a sign to refer to — so it is left out of the display graph and would not appear at all
    /// without this. Seeing what arrived is most of why it was imported.
    ///
    /// **A closed one is left out here in field mode, and used to be included.** The reason
    /// given was that an import should look the same whether or not it is solid — but it did
    /// not: a closed mesh is in the display graph, so the tracer draws it *and* this drew a
    /// translucent copy on top of it. Two descriptions of one solid, the second washing out
    /// the first at 75% alpha, and every show/hide rebuilding and re-uploading a buffer of
    /// every triangle in the document to draw it.
    ///
    /// In polygon mode nothing is traced, so there it is the only way a closed import appears
    /// and it is gathered as before.
    ///
    /// Placements are baked into the vertices rather than carried as a matrix, because several
    /// imports can sit under different ones and a single MVP cannot say that.
    pub(crate) fn rebuild_imported_mesh(&mut self) {
        let traced = self.view.mode != ViewMode::Polygon;
        let graph = self.part_graph().clone();
        let mut positions: Vec<Point3<f64>> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        // Once for every root below, which is every rebuild (`is_visible_among`).
        let consumed = graph.consumed();
        for root in graph.roots() {
            if !is_visible_among(&self.doc, &graph, &consumed, root) {
                continue;
            }
            // Down through any placements to the mesh itself, so a placed import draws where
            // it is rather than where it was built.
            let mut id = root;
            let mesh = loop {
                match graph.node(id) {
                    Some(Node::Mesh(mesh)) => break Some(*mesh),
                    // An empty slot ends the walk the same way a block that is not a mesh
                    // does: there is no leaf under a slot nothing has been put in.
                    Some(Node::Transform { input, .. }) => match input.source() {
                        Some(next) => id = next,
                        None => break None,
                    },
                    _ => break None,
                }
            };
            let Some(mesh) = mesh.and_then(|m| self.doc.mesh(m)) else { continue };
            // Already on screen, drawn per pixel by the thing that owns the surface.
            if traced && mesh.is_closed() {
                continue;
            }
            let place = self.placement_above(id);
            let base = positions.len() as u32;
            positions.extend(
                mesh.positions.iter().map(|p| place * Point3::new(p[0], p[1], p[2])),
            );
            triangles.extend(mesh.triangles.iter().map(|t| [base + t[0], base + t[1], base + t[2]]));
        }
        self.imported_mesh = CageMesh::from_triangles(&positions, &triangles);
    }

    /// Whether this block is drawn over the model rather than traced as a solid.
    ///
    /// A plane's square and a point's dot. Neither has a surface to trace — a half-space fills
    /// the screen and a point encloses nothing — so both are painted by the viewport overlay,
    /// and what takes one off screen is the `hidden` flag that overlay reads. Which is why the
    /// eye on their rows means `hidden` even when something consumes them: they are on screen
    /// either way, so "draw this as well" would be answering a question nobody asked.
    pub(crate) fn draws_as_overlay(&self, id: NodeId) -> bool {
        draws_as_overlay_in(self.part_graph(), id)
    }

    /// Whether this body of the active part is reference geometry rather than a part:
    /// [`fieldform_field::is_reference_body`], which says why.
    pub(crate) fn is_reference_body(&self, id: NodeId) -> bool {
        fieldform_field::is_reference_body(&self.doc, self.part_graph(), id)
    }

    /// The flag this block's eye reads and writes, and never anything else
    /// (`block-ownership-spec.md` §7.2).
    ///
    /// `hidden` while it is a body, since a body draws unless hidden; `shown` once something
    /// consumes it, since an ingredient draws only when asked for. So a body's eye goes off as it
    /// is consumed and comes back as it is let go of, which is what has a union draw rather than
    /// its parts. And `hidden` for a plane or a point whatever consumes it, for the reason
    /// [`Self::draws_as_overlay`] gives.
    ///
    /// **One rule, because the eye spoke to the two flags from several places that each chose
    /// one.** The H key wrote `hidden` on a consumed block, which nothing draws by until the block
    /// is let go of, while the row beside it wrote `shown`. Where a block sits in the list is not
    /// asked: neither flag moves a row, and nothing here reads ownership.
    pub(crate) fn eye_flag(&self, id: NodeId) -> EyeFlag {
        eye_flag_in(self.part_graph(), id)
    }

    /// Whether a block is drawn, as its eye says: a body unless hidden, a block something
    /// consumes only when shown.
    ///
    /// Everything that decides what draws reads the flags through this, so the eye and the
    /// picture cannot disagree about which flag counts.
    pub(crate) fn is_visible(&self, id: NodeId) -> bool {
        is_visible_in(&self.doc, self.part_graph(), id)
    }

    /// Draws a block or stops drawing it, through the flag its eye means: **the one write every
    /// eye makes**, and an undo step (`fieldform_ops::SetVisible`).
    ///
    /// The flag is asked of the graph as it is now. An operation that consumes a body and wants it
    /// kept on screen does not call this after its graph edit, which would be an entry of its own
    /// that a redo of the edit does not reach: it hands [`Self::set_part_graph_with`] the
    /// [`eye_command`] asked of the graph it writes, and reaches `shown` in the edit's own entry.
    pub(crate) fn set_visible(&mut self, id: NodeId, on: bool) {
        self.set_visible_all(&[id], on);
    }

    /// [`Self::set_visible`] for several blocks as one undo step and one rebuild: the import's
    /// eye, where forty bodies turned off by one click come back with one Ctrl+Z.
    ///
    /// A block already as asked is left out, so a click that changes nothing leaves no entry.
    pub(crate) fn set_visible_all(&mut self, ids: &[NodeId], on: bool) {
        let graph = self.part_graph();
        let mut commands: Vec<Box<dyn fieldform_ops::Command>> = ids
            .iter()
            .filter(|id| is_visible_in(&self.doc, graph, **id) != on)
            .map(|id| Box::new(eye_command(graph, *id, on)) as Box<dyn fieldform_ops::Command>)
            .collect();
        let cmd: Box<dyn fieldform_ops::Command> = match commands.len() {
            0 => return,
            1 => commands.remove(0),
            _ => Box::new(fieldform_ops::Batch::new("block.set_visible", commands)),
        };
        // Not a graph edit, so it ends any gesture a value drag had open rather than folding into
        // it, as a rename does: undoing the drag must not take the eye with it. **All three**, as
        // undo ends them: a material or an output block's drag folds by replacing the top of the
        // history, so its next step inside the window would have replaced this entry, leaving the
        // eye's change in the document with nothing on the stack to take it back.
        self.gesture = None;
        self.material_gesture = None;
        self.outputs.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
        // **Every eye rebuilds, for one reason.** This changes what is drawn, picked and baked,
        // and none of it happens until the render resources are built again. Without it the
        // flag flipped and the picture did not, until some *other* edit rebuilt for its own
        // reasons -- which reached the owner as an eye that worked only after toggling a
        // different block a couple of times. One of the two eyes this replaced had left it out.
        self.rebuild_render_resources();
    }

    /// The named blocks something else consumes that are drawn as well, oldest first.
    ///
    /// **One reading, because two would disagree.** `display_graph` draws these and
    /// `selected_body_index` counts them, and the count is what tells the shader which body to
    /// highlight -- so a body counted against a different list highlights the wrong one, which
    /// is the failure the comment in `selected_body_index` has always warned about.
    pub(crate) fn shown_ingredients(&self) -> Vec<NodeId> {
        shown_ingredients_in(&self.doc, self.part_graph())
    }

    /// Shows or hides the selected body.
    pub(crate) fn toggle_selected_visibility(&mut self) {
        let Some(id) = self.selected_body else {
            self.warn("Nothing selected");
            return;
        };
        self.toggle_visibility(id);
    }

    /// `hide target=…` and `show target=…`: the named block's eye set as asked, or a word that it
    /// already was. A name that names nothing is refused, so a script stops where it was wrong.
    pub(crate) fn set_visibility_named(&mut self, verb: &str, name: &str, on: bool) {
        let Some(id) = self.node_named(name) else {
            self.warn(format!("nothing is called {name:?} to {verb}"));
            return;
        };
        if self.is_visible(id) == on {
            let label = self.body_label(id);
            self.say(if on { format!("{label} is already drawn") } else { format!("{label} is already hidden") });
            return;
        }
        self.toggle_visibility(id);
    }

    /// `lock target=…` and `unlock target=…`: the named block. A name that names nothing is
    /// refused, so a script stops where it was wrong.
    pub(crate) fn set_locked_named(&mut self, verb: &str, name: &str, locked: bool) {
        match self.node_named(name) {
            Some(id) => self.set_locked(id, locked),
            None => self.warn(format!("nothing is called {name:?} to {verb}")),
        }
    }

    /// Locks or unlocks a block: the padlock on its row, the row menu's item, and the two verbs.
    ///
    /// A locked block cannot be deleted and cannot be taken by a block that does not already
    /// read it. Its own numbers, its name and its eye are left to be changed, and whatever read
    /// it before goes on reading it. An undo step, and nothing to redraw: a lock changes what
    /// may be done next, not what is there.
    pub(crate) fn set_locked(&mut self, id: NodeId, locked: bool) {
        let label = self.body_label(id);
        if self.doc.is_locked(id) == locked {
            self.say(if locked { format!("{label} is already locked") } else { format!("{label} is not locked") });
            return;
        }
        // Not a graph edit, so it ends an open gesture rather than folding into it, as an eye does.
        self.gesture = None;
        self.material_gesture = None;
        self.outputs.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(fieldform_ops::SetLocked::new(id, locked))) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
        self.say(if locked {
            format!("Locked {label}: it cannot be deleted or used in another block until it is unlocked")
        } else {
            format!("Unlocked {label}")
        });
    }

    /// Why the active part's graph may not become `after`, when the reason is a lock.
    ///
    /// Asked by the check every structural edit makes before it commits and again by the commit
    /// itself, so an edit that skipped the first is still stopped.
    pub(crate) fn lock_refusal(&self, after: &Graph) -> Option<String> {
        use fieldform_scene::LockBroken;
        Some(match self.doc.lock_broken_by(self.part_graph(), after)? {
            LockBroken::Removed(id) => {
                format!("{} is locked, so it cannot be deleted \u{2014} unlock it first", self.body_label(id))
            }
            LockBroken::Taken { locked, .. } => {
                format!(
                    "{} is locked, so another block cannot use it \u{2014} unlock it first",
                    self.body_label(locked)
                )
            }
        })
    }

    /// Turns a block's eye over: the eye on its row, the row menu's item, and H.
    ///
    /// Said in the words of what its flag does: a body is shown or hidden, a block something
    /// consumes is drawn beside the bodies or not.
    pub(crate) fn toggle_visibility(&mut self, id: NodeId) {
        let label = self.body_label(id);
        let on = !self.is_visible(id);
        let said = match (self.eye_flag(id), on) {
            (EyeFlag::Hidden, true) => format!("Showing {label}"),
            (EyeFlag::Hidden, false) => format!("Hiding {label}"),
            (EyeFlag::Shown, true) => format!("Drawing {label} as well"),
            (EyeFlag::Shown, false) => format!("Stopped drawing {label}"),
        };
        self.set_visible(id, on);
        self.say(said);
        // A cage shown or hidden can change which cage is there to be worked on.
        self.resync_active_cage();
    }
}
