// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Names and labels: a body's, a curve's, a variable's, and what each row is called.

use crate::*;

impl App {
    /// Clears `hidden` on a block an operation has just consumed, so it draws if it is ever let go
    /// of.
    ///
    /// **Not the eye, and never to become it** (`block-ownership-spec.md` §7.2). Every operation
    /// that consumes a block calls this right after, and it names its flag rather than asking
    /// [`Self::eye_flag`]: the block is no longer a body, so the eye's flag is `shown` by then,
    /// and writing that would draw every operand beside its result. A body hidden by hand and
    /// then consumed therefore comes back visible when released.
    ///
    /// Off the undo stack, as it always was; the eye is on it. The flag lives in the document
    /// rather than the app, so hiding half an assembly survives a reopen.
    pub(crate) fn clear_hidden(&mut self, id: NodeId) {
        let was = self.doc.node_meta.get(&id).is_some_and(|m| m.hidden);
        if let Some(m) = self.doc.node_meta.get_mut(&id) {
            m.hidden = false;
        }
        if was {
            self.mark_changed();
        }
    }

    /// The name a body has been given, if it has been given one.
    pub(crate) fn body_name(&self, id: NodeId) -> Option<&str> {
        self.doc.node_meta.get(&id).and_then(|m| m.name.as_deref())
    }

    /// A name nothing else in the document is using, built from what the block already is.
    ///
    /// Derived rather than generic — `Sphere`, then `Sphere 2` — because the name goes straight
    /// into the From and With lists of every other block, and a list of `variable_1` through
    /// `variable_6` is a list you have to click through to read. The block's own description is
    /// the best first guess at what someone would have typed, and it stays editable.
    pub(crate) fn unique_variable_name(&self, id: NodeId) -> String {
        self.unique_variable_name_in(self.part_graph(), id)
    }

    /// The same, for a block in `graph`: settling reaches every part, and a block it made a
    /// variable in a part not on screen is described by that part's graph, not this one's.
    pub(crate) fn unique_variable_name_in(&self, graph: &Graph, id: NodeId) -> String {
        // The names blocks that exist hold, for the reason `set_body_name` gives.
        let in_use = names::in_use(&self.doc);
        let taken: std::collections::HashSet<&str> = in_use.iter().map(|(_, n)| n.as_str()).collect();
        let base = self.derived_label_in(graph, id);
        if !taken.contains(base.as_str()) {
            return base;
        }
        // **Suffixed with an underscore, not a space.** A variable's name is what an expression
        // refers to it by, so the invented one is spelled the way an identifier is spelled:
        // `Sphere`, then `Sphere_1`. It used to be `Sphere 2`, which read as prose in the one
        // place the text is a name rather than a label. Asked for by the owner while testing the
        // V handle, which is now the commonest way a name gets invented at all.
        //
        // Bounded: there cannot be more collisions than there are names already taken.
        (1..=taken.len() + 1)
            .map(|n| format!("{base}_{n}"))
            .find(|candidate| !taken.contains(candidate.as_str()))
            .unwrap_or(base)
    }

    /// A curve name nothing else in the document is using.
    ///
    /// **Because curve names are now read, not just displayed.** Every block that takes a curve
    /// offers them by name — a cylinder's ends, a pipe's spine, an extrude's profile — and until
    /// this every line was called "Line". Three of them made a dropdown of three identical rows,
    /// which is a list you have to click through to read. Same shape as `unique_variable_name`
    /// and the same reason.
    pub(crate) fn unique_curve_name(&self, base: &str) -> String {
        let taken: std::collections::HashSet<&str> =
            self.doc.curves.values().map(|c| c.name.as_str()).collect();
        if !taken.contains(base) {
            return base.to_string();
        }
        (2..=taken.len() + 2)
            .map(|n| format!("{base} {n}"))
            .find(|candidate| !taken.contains(candidate.as_str()))
            .unwrap_or_else(|| base.to_string())
    }

    /// Renames a curve, keeping the empty case meaning "back to what it was called".
    ///
    /// **One undo step, through [`fieldform_ops::SetName`], as renaming a body is.** `SetCurve`
    /// was the obvious command and the wrong one: it replaces the whole curve, so a text field
    /// writing it character by character would push an undo step per keystroke. `SetName` writes
    /// only the name, and the row's field writes it once, on Enter or when it loses focus. Curves,
    /// bodies and profile rows went onto the stack together, so no kind of name is undoable while
    /// another is not.
    pub(crate) fn rename_curve(&mut self, id: CurveId, name: &str) {
        self.set_name(fieldform_ops::Named::Curve(id), name);
    }

    /// Writes a name through the undo stack. See [`fieldform_ops::SetName`] for why a rename is
    /// an undo step at all.
    pub(crate) fn set_name(&mut self, what: fieldform_ops::Named, name: &str) {
        // A rename is not a graph edit, so it ends any gesture a value drag had open rather than
        // folding into it: undoing the drag must not take the name with it.
        self.gesture = None;
        let cmd = Box::new(fieldform_ops::SetName::new(what, name));
        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
    }

    /// Puts a block where `cmd` says, through the undo stack — make variable, put back inside, or
    /// a rename (`fieldform_ops::SetOwnership`). Answers whether it went.
    pub(crate) fn set_ownership(&mut self, cmd: fieldform_ops::SetOwnership) -> bool {
        // Not a graph edit, so it ends any gesture a value drag had open rather than folding into
        // it, as a rename always has: undoing the drag must not take this with it.
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(cmd)) {
            self.warn(e.to_string());
            return false;
        }
        self.mark_changed();
        true
    }

    /// Names a block, or clears the name when given an empty one. Answers whether the name stuck.
    ///
    /// Cleared rather than stored empty, so a name typed and then deleted leaves the body back
    /// where it started — described by what it *is* — rather than showing a blank row.
    ///
    /// **A name is a label, and writing one moves no row** (`block-ownership-spec.md` §5.3). It
    /// goes on the history as a `SetOwnership` carrying the ownership the block already has, so a
    /// rename and the V are one kind of entry. It used to be what lifted a block out of the block
    /// it was drawn inside, so a name typed for any reason moved the row. **A variable's name may
    /// not be cleared**, since other blocks type it to refer to it (I4): putting it back inside
    /// is the way to stop it being one, and the refusal says so.
    ///
    /// **A name two blocks share is refused, because a name is an address.** A reference field
    /// resolves typed text to one block (`reference_field`), so two blocks called `Sphere` make one
    /// of them unreachable by typing and the field quietly picks the other. Refused on the
    /// keystroke rather than repaired afterwards, which is the arrangement
    /// `set_parameter_expression` already uses for a cycle and for the same reason: the moment the
    /// name is typed is the only moment the person knows what they meant by it.
    ///
    /// Auto-suffixing was the alternative and is worse here — it renames under the person, so the
    /// block they just called `Shell` is called `Shell_1` and the name they typed belongs to
    /// something else. Invented names still suffix (`unique_variable_name`), because nobody chose
    /// those.
    ///
    /// One place, so every path that renames gets the same answer: the row's own edit, a name
    /// given in the console, and the halves a split names.
    pub(crate) fn set_body_name(&mut self, id: NodeId, name: &str) -> bool {
        let name = name.trim();
        if let Some(why) = self.body_name_refusal(id, name) {
            self.say(why);
            return false;
        }
        match rename_block(&self.doc, id, name) {
            Ok(cmd) => self.set_ownership(cmd),
            Err(why) => {
                self.say(why);
                false
            }
        }
    }

    /// Why block `id` may not be called `name`, or `None` where it may: [`Self::set_body_name`]'s
    /// refusals, askable first by a caller that has to warn rather than say -- a typed `rename`,
    /// which must stop a script where a row's field only has to tell the person.
    pub(crate) fn body_name_refusal(&self, id: NodeId, name: &str) -> Option<String> {
        let name = name.trim();
        // **An empty name is a name being cleared, and a clear never clashes.** Comparing it
        // like any other would match every block that has no name — `node_meta` holds entries for
        // `hidden` and `shown` too — so clearing one would report a clash with an unrelated block.
        //
        // **Among the blocks that exist**, not among the table's entries: a deleted block keeps
        // its entry so undo can bring the name back, and the first version read that entry as a
        // clash -- delete `ball`, type `'ball'` again, and the name was "taken by a sphere" nobody
        // could see (`names.rs`).
        if let Some(clash) = (!name.is_empty()).then(|| names::holder_of(&self.doc, name, id)).flatten() {
            let what = self.derived_label(clash);
            return Some(format!(
                "\u{201c}{name}\u{201d} is already taken by a {what} — names are what blocks refer \
                 to each other by, so two cannot share one"
            ));
        }
        rename_block(&self.doc, id, name).err().map(str::to_string)
    }

    /// The file a body came from, if it is an imported mesh.
    ///
    /// Down through any placements, so a body that has been moved is still the body that
    /// arrived — the same walk `rebuild_imported_mesh` makes for the same reason.
    pub(crate) fn body_source(&self, root: NodeId) -> Option<fieldform_scene::id::SourceId> {
        let graph = self.part_graph();
        let mut id = root;
        loop {
            match graph.node(id) {
                Some(Node::Mesh(mesh)) => return self.doc.mesh(*mesh).and_then(|m| m.source),
                Some(Node::Cage(cage)) => {
                    return self.doc.sources.values().find(|s| s.cages.contains(cage)).map(|s| s.id);
                }
                // Nothing imported is under an empty slot, which is the `None` anything else
                // already answers.
                Some(Node::Transform { input, .. }) => id = input.source()?,
                _ => return None,
            }
        }
    }

    /// The mesh a body draws, down through any placements.
    pub(crate) fn mesh_under(&self, root: NodeId) -> Option<&fieldform_scene::MeshData> {
        let graph = self.part_graph();
        let mut id = root;
        loop {
            match graph.node(id) {
                Some(Node::Mesh(mesh)) => return self.doc.mesh(*mesh),
                // The same end to the walk: no mesh is under a hole.
                Some(Node::Transform { input, .. }) => id = input.source()?,
                _ => return None,
            }
        }
    }

    /// The bodies one import produced, in the list's own order.
    pub(crate) fn source_members(&self, source: fieldform_scene::id::SourceId) -> Vec<NodeId> {
        self.ordered_nodes()
            .into_iter()
            .filter(|id| self.body_source(*id) == Some(source))
            .collect()
    }

    /// What a parameter reference is worth right now.
    ///
    /// An unresolvable one — a name the document does not hold, or an expression that fails to
    /// evaluate (`Document::parameter_value` answers `None` for both) — reads as zero *here
    /// only*, for sizing a bounding box. Evaluation refuses it rather than substituting anything,
    /// which is where that has to be caught.
    pub(crate) fn resolve(&self, r: ParamRef) -> f64 {
        guard_stack("App::resolve");
        r.resolve(&|id| {
            self.doc.parameter_value(id)
        })
        .unwrap_or(0.0)
    }

    /// A primitive's centre as a point, for sizing a box around it.
    pub(crate) fn resolve_point(&self, centre: &[ParamRef; 3]) -> Point3<f64> {
        Point3::new(
            self.resolve(centre[0]),
            self.resolve(centre[1]),
            self.resolve(centre[2]),
        )
    }

    /// Writes a number through a reference: into the literal, or into the parameter it names.
    ///
    /// Returns the reference to store, which only changes for a literal — a named one keeps
    /// its name and gets a new value, which is the whole point of naming it.
    pub(crate) fn set_value(&mut self, r: ParamRef, value: f64) -> ParamRef {
        match r {
            ParamRef::Literal(_) => ParamRef::Literal(value),
            ParamRef::Named(id) => {
                // Only a typed number takes the write. A computed one is left alone: the number
                // a drag would put here is the expression's own output, and overwriting the
                // expression with it would silently turn a relationship into a constant. The
                // field is disabled for a computed value, so nothing routes a drag here from
                // the panel; this is the guard for the manipulator.
                if let Some(param) = self.doc.parameters.get_mut(&id) {
                    if param.value.as_literal().is_some() {
                        param.value = fieldform_scene::ParamValue::Literal(value);
                        self.mark_changed();
                    }
                }
                r
            }
            // A driven reference has no single value to write. Nothing routes a drag here —
            // the panel shows a ramp's own controls instead of one number — so this is the
            // "leave it alone" answer rather than a case to handle.
            ParamRef::Field(_) => r,
        }
    }

    /// What to call a body in a list: what it is, not what it is called.
    ///
    /// Bodies have no names of their own. Naming them would mean a persistent per-node label
    /// in the file, and nothing yet needs one — the graph is small enough that "Pipe" and
    /// "Subtract" say which is which.
    pub(crate) fn body_label(&self, id: NodeId) -> String {
        self.body_label_in(self.part_graph(), id)
    }

    /// Why a row's block is not drawn, when the reason is a cage that is not a solid: the cage's
    /// own row reads the state ("open: 1 hole, 4 edges — not drawn"), and a block reading it says
    /// which cage. `None` for every row an open cage has nothing to do with, which is the usual
    /// one.
    pub(crate) fn red_cage_note(&self, id: NodeId) -> Option<String> {
        let graph = self.part_graph();
        if let Some(Node::Cage(cage)) = graph.node(id) {
            let body = self.doc.cage_body(*cage)?;
            return match body.describe_error() {
                Some(why) => Some(format!("{why} \u{2014} not drawn")),
                // Drawn from the unchecked surface while the check runs (R14); said so.
                None if body.surface_unchecked() => Some("checking\u{2026}".to_string()),
                None => None,
            };
        }
        let cage = fieldform_field::red_cage_behind(&self.doc, graph, id)?;
        let name = graph
            .nodes()
            .find(|(_, n)| matches!(n, Node::Cage(c) if *c == cage))
            .map(|(node, _)| self.body_label(node))
            .unwrap_or_else(|| "a cage".to_string());
        Some(format!("reads {name}, which is open \u{2014} not drawn"))
    }

    /// The same, for a block in `graph` — another part's, when a refusal names a block there.
    /// Asked of the part on screen, a block it does not hold reads "gone".
    pub(crate) fn body_label_in(&self, graph: &Graph, id: NodeId) -> String {
        guard_stack("body_label");
        // A given name wins. What a body *is* is the right answer until two of them are the
        // same thing — two pipes are two rows both saying "Pipe", and the only way to tell them
        // apart was to click one and watch the viewport.
        match self.body_name(id) {
            Some(name) => name.to_string(),
            None => self.derived_label_in(graph, id),
        }
    }

    /// What a block's slot reads, for a row that names it — or that it reads nothing yet.
    ///
    /// **One wording for the hole, because these rows are read by scanning the list's left
    /// edge.** A slot is empty while a block waits to be told what to read
    /// (`notebook-spec.md` §7.2), five of the labels below name what their block is applied
    /// to, and each spelling that state its own way would make one state look like five. The
    /// row is not the place to explain why it is empty: the block says, where the slot is.
    pub(crate) fn operand_label_in(&self, graph: &Graph, slot: &fieldform_scene::Operand) -> String {
        match slot.source() {
            Some(id) => self.body_label_in(graph, id),
            None => "nothing yet".to_string(),
        }
    }

    /// What a body is, ignoring any name it has been given.
    ///
    /// Kept separate because the name box needs it as its placeholder: the hint shows what
    /// clearing the box would give back, which is the only way that box can say what it does.
    pub(crate) fn derived_label(&self, id: NodeId) -> String {
        self.derived_label_in(self.part_graph(), id)
    }

    /// The same, for a block in `graph`, and what it is applied to looked up there too.
    pub(crate) fn derived_label_in(&self, graph: &Graph, id: NodeId) -> String {
        match graph.node(id) {
            Some(Node::Cage(_)) => "Cage".to_string(),
            // **"Pipe", asked for by name.** It read "Along <curve>", which was an attempt to
            // answer a different complaint -- a ramp offering "Pipe" as the field it reads says
            // the mechanism and not the intent. That put the input where every other block puts
            // the operation, which is the thing the rule below exists to stop. Which curve it
            // follows is in its own block, and naming the pipe is what puts a word of your own
            // on the row.
            Some(Node::Pipe { .. }) => "Pipe".to_string(),
            // **The operation, not the curve.** These used to read "Outline extruded", which
            // put the input where every other block puts what it does — so the same fact
            // appeared in two orders depending on the block, and the row could not be scanned
            // down its left edge. The profile is still shown: it is listed as an input, beside
            // every other curve the block names, which is where a pipe's curve already was.
            Some(Node::Extrude { .. }) => "Extrude".to_string(),
            Some(Node::Revolve { .. }) => "Revolve".to_string(),
            Some(Node::Sweep { .. }) => "Sweep".to_string(),
            Some(Node::Loft { .. }) => "Loft".to_string(),
            Some(Node::Expression { .. }) => "Expression".to_string(),
            Some(Node::Sphere { .. }) => "Sphere".to_string(),
            // Named for the shape it is rather than for the menu entry that made it: three
            // equal sides is a cube, and anything else is a box. Stretching one in the panel
            // renames its own row, which is the only signal that it stopped being a cube.
            Some(Node::Cuboid { size, .. }) => {
                let [x, y, z] = size.map(|s| self.resolve(s));
                if (x - y).abs() < 1e-9 && (y - z).abs() < 1e-9 {
                    "Cube".to_string()
                } else {
                    "Box".to_string()
                }
            }
            Some(Node::SkewedBox { .. }) => "Skewed box".to_string(),
            Some(Node::Cylinder { .. }) => "Cylinder".to_string(),
            Some(Node::Cone { .. }) => "Cone".to_string(),
            Some(Node::Torus { .. }) => "Torus".to_string(),
            Some(Node::Coil { .. }) => "Coil".to_string(),
            // Its name, where it has one, is what it is: "M8 thread" reads down the list.
            Some(Node::Thread { designation: Some(name), .. }) => format!("{name} thread"),
            Some(Node::Thread { .. }) => "Thread".to_string(),
            // What kind of hole it is, which is what reads down a list of several.
            Some(Node::Hole { kind, .. }) => match kind {
                fieldform_scene::HoleKind::Simple { .. } => "Hole".to_string(),
                fieldform_scene::HoleKind::Counterbore { .. } => "Counterbored hole".to_string(),
                fieldform_scene::HoleKind::Countersink { .. } => "Countersunk hole".to_string(),
                fieldform_scene::HoleKind::Tapped { designation, .. } => format!("{designation} tapped hole"),
            },
            Some(Node::Plane { .. }) => "Plane".to_string(),
            Some(Node::Point { .. }) => "Point".to_string(),
            // Its own name, because an import is a thing the user brought in and will look for
            // by the name it arrived under.
            Some(Node::Mesh(id)) => self
                .doc
                .mesh(*id)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| "Missing mesh".to_string()),
            // Named for the cell, because that is what distinguishes one lattice from another.
            Some(Node::Lattice { cell, .. }) => lattice_cell_label(*cell).to_string(),
            Some(Node::Capsule { .. }) => "Capsule".to_string(),
            Some(Node::Boolean { op, .. }) => boolean_label(*op).to_string(),
            Some(Node::Blend { op, .. }) => format!("{} (filleted)", boolean_label(*op)),
            // The amount is in the name because it is the whole of what the block does, and a
            // list of three morphs that all read "Morph" says nothing about which is which.
            Some(Node::Morph { t, .. }) => {
                format!("Morph {:.0}%", self.resolve(*t).clamp(0.0, 1.0) * 100.0)
            }
            // **The operation first, then what it is applied to.** These read the other way
            // round -- "Morph 50% (attracted)" -- so the block's own identity was in the
            // brackets and the thing it was built on was the headline. Reported of an attractor
            // over a morph, and the same complaint the extrude arm above already records: the
            // same fact appeared in two orders depending on the block, and the list could not be
            // scanned down its left edge.
            //
            // What each one is applied to stays, in brackets, because a placement or a twist is
            // not a body of its own -- it is a body changed -- and losing which one would lose
            // the part somebody is looking for.
            Some(Node::Transform { input, .. }) => {
                format!("Placed ({})", self.operand_label_in(graph, input))
            }
            Some(Node::Twist { input, .. }) => {
                format!("Twist ({})", self.operand_label_in(graph, input))
            }
            Some(Node::Attract { input, .. }) => {
                format!("Attract ({})", self.operand_label_in(graph, input))
            }
            // **The number it builds, not the name of the variable behind it.** A row's label is
            // read by scanning the list's left edge, and `Pattern ×copies_per_hub` makes that a
            // thing to decode where `Pattern ×6` is a thing to see. The name is on the count's own
            // field, where it is being set. A count that will not resolve shows `×?` rather than a
            // number it is not building — the row is not the place to explain why, and the field says.
            Some(Node::Pattern { input, count, kind, join, .. }) => format!(
                "Pattern \u{d7}{} {} ({}){}",
                fieldform_scene::pattern_count(&self.doc, count)
                    .map_or_else(|_| "?".to_string(), |(n, _)| n.to_string()),
                match kind {
                    PatternKind::Linear { .. } => "in a row",
                    PatternKind::Radial { .. } => "around",
                },
                self.operand_label_in(graph, input),
                // Said on the row, because it is the difference between this block being all
                // of the pattern and being all but one of it -- and the body list is where a
                // person looks to find the one that is missing.
                if *join { "" } else { ", copies only" },
            ),
            Some(Node::Mirror { input, .. }) => {
                format!("Mirror ({})", self.operand_label_in(graph, input))
            }
            Some(Node::Scale { input, .. }) => {
                format!("Scale ({})", self.operand_label_in(graph, input))
            }
            // **"Offset", which is what the menu calls it.** The row said "Grow" or "Shrink"
            // -- which way the number went -- so the menu entry somebody clicked and the row
            // that appeared did not share a word, and there was nothing to search the list for.
            // Reported as exactly that. Which way it went is in the number, one line inside the
            // block; whether it varies is not, so that much stays.
            Some(Node::Offset { input, distance }) => format!(
                "{} ({})",
                if distance.driver().is_some() { "Offset varying" } else { "Offset" },
                self.operand_label_in(graph, input),
            ),
            Some(Node::Shell { input, .. }) => {
                format!("Shell ({})", self.operand_label_in(graph, input))
            }
            Some(Node::Infill { input, .. }) => {
                format!("Infill ({})", self.operand_label_in(graph, input))
            }
            // Never listed as a body — `roots` leaves it out — so this is only reached when a
            // panel names the node driving something.
            Some(Node::Ramp { .. }) => "Ramp".to_string(),
            Some(Node::Unknown(_)) => "Unreadable".to_string(),
            None => "gone".to_string(),
        }
    }

    // `visible_mesh` used to live here    // `visible_mesh` used to live here: it chose whichever mesh was on screen for pivot
    // picking to raycast. That choice was the bug. It made the pivot depend on the display
    // mode, and in traced mode it tested a coarse mesh against a surface the user was seeing
    // drawn per pixel, so clicks near a silhouette missed. `pick_pivot` traces the field
    // instead, which is the same in every mode, and nothing else ever wanted this.
}
