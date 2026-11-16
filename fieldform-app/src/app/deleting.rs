// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Delete: a body, a curve, and a curve some block is built from.

use crate::*;

impl App {
    /// Deletes whatever is selected — a curve, or a body.
    ///
    /// A curve first, because it is the more specific selection: selecting one clears the body,
    /// so if a curve is selected it is what the user is pointing at.
    pub(crate) fn delete_selection(&mut self) {
        // A section's row, when it is the thing selected: Delete takes the section and nothing
        // else (the owner, 2026-10-24: no ✕ on the row; the menu or the key). Selecting it
        // cleared everything below, as they clear it.
        if let Some(id) = self.section_selected
            && let Some(index) = self.sections().iter().position(|s| s.id == id)
        {
            self.remove_section(index);
            return;
        }
        // A selected block is what was clicked last: selecting a body or a curve clears it.
        if let Some(block) = self.selected_block {
            self.delete_block(block);
            return;
        }
        if let Some(id) = self.selected_curve {
            self.delete_curve(id);
            return;
        }
        self.delete_selected_body();
    }

    /// Deletes a curve, unless a pipe is sweeping it or a block in another part reads it.
    ///
    /// Refused rather than cascaded. A `Node::Pipe` naming a curve that is not there does not
    /// evaluate at all, so deleting one out from under a pipe would blank the viewport — and
    /// silently deleting the pipe as well would throw away a body the user did not ask about.
    /// Naming the pipe is the answer that leaves them in control of both.
    pub(crate) fn delete_curve(&mut self, id: CurveId) {
        // **`Node::curves` rather than a match on `Node::Pipe`.** The hand-written version was
        // right when a pipe was the only thing that referenced a curve, and had been wrong since
        // extrude and revolve were added: a curve driving either could be deleted out from under
        // it. That is the mistake D9 describes as being made once per node kind, and asking the
        // node is what stops the next one making it.
        // **Asking the node is not enough, because an extrude names one curve and reads a loop.**
        // `Node::curves` returns the single `profile` an extrude, revolve or sweep stores; the boundary
        // it actually walks is `profile_loop` from that curve. So deleting any *other* member of the
        // same loop passed this test and opened the loop, and the solid stopped compiling with nothing
        // said at the moment the damage was done. The row path over-refused and this path
        // under-refused, in opposite directions, on the same four curves.
        //
        // So a member is tested by its loop: whatever consumes any curve of the loop this one is in
        // consumes this one in effect.
        let mut loop_mates =
            fieldform_field::loop_from(&self.doc, id, None).unwrap_or_else(|_| vec![id]);
        if !loop_mates.contains(&id) {
            loop_mates.push(id);
        }
        // **A block in another part refuses it outright, before anything below is asked.**
        // Everything below reads the part on screen, where a delete can open a loop into a wall
        // because the block is in front of the person deleting; `readers_in_other_parts` says why
        // another part's is not. Asked of this curve, not over `loop_mates`: the function walks
        // each block's own section, so a block built on the loop is found from any side of it,
        // and a pipe along a neighbouring side — which reads that side alone — is not.
        let elsewhere = readers_in_other_parts(&self.doc, self.active_part, &[id], |g, n| {
            self.body_label_in(g, n)
        });
        if !elsewhere.is_empty() {
            let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
            let readers = elsewhere.readers.join(", ");
            let read = if elsewhere.through_sections {
                format!("is one of the curves {readers} is made from")
            } else {
                format!("is used by {readers}")
            };
            self.say(format!("{name} {read} — {}", elsewhere.built_on()));
            return;
        }
        let used_by: Vec<String> = self
            .part_graph()
            .nodes()
            .filter(|(_, n)| n.curves().iter().any(|c| loop_mates.contains(c)))
            .map(|(node, _)| self.body_label(node))
            .collect();
        if !used_by.is_empty() {
            // **A member of a loop that blocks are built on can go, and the blocks follow.**
            // What is left is an open chain, which a block can be built on now, as a wall. So
            // the member goes, the joints at its ends go, and each block that read the loop
            // reads the chain. Only where every reader of *this* curve reads it as its section
            // — a pipe along it, or a sweep carrying something along it, still refuses below.
            //
            // **And a block built from this very curve lets go of it and waits.** A pipe along
            // it, or a sweep carrying something along it, used to refuse here -- "delete that
            // first" -- because a curve slot could not name nothing, which left the curve stuck
            // inside the block. It can be empty now (`fieldform_scene::CurveSlot`), so the delete
            // goes ahead and the block stays, waiting to be told which curve instead.
            //
            // What still refuses is a block that names the curve some way that has no empty
            // state to go to: a frame aimed along it, an end standing on it, an attractor.
            let held_beyond: Vec<String> = {
                let mut held: Vec<String> = self
                    .part_graph()
                    .nodes()
                    .filter(|(_, n)| n.holds_beyond_its_slots(id))
                    .map(|(node, _)| self.body_label(node))
                    .collect();
                held.sort();
                held.dedup();
                held
            };
            if held_beyond.is_empty() {
                match loop_mates.len() > 1 {
                    true => self.delete_loop_member(id, &loop_mates),
                    false => self.delete_held_curve(id),
                }
                return;
            }
            let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
            self.say(format!(
                "{name} is what {} is aimed along or stands on — point {} somewhere else first, \
                 or delete it",
                held_beyond.join(", "),
                if held_beyond.len() == 1 { "it" } else { "them" },
            ));
            return;
        }
        let Some(curve) = self.doc.curve(id).cloned() else { return };
        let name = curve.name.clone();
        // **The same builder the row path uses, because a lone curve leaves joints too.** This went
        // through `delete_curves_commands`, which knows nothing about points, so deleting one member of
        // a held loop left its corner dots behind — the bug `deleting_a_row_takes_its_joints_with_it`
        // was written for, fixed on the row path and not here. `delete_row_commands` removes a point
        // only when nothing reads it and no *remaining* curve sits on it, so a shared corner stays for
        // the curve still using it.
        let commands = delete_row_commands(&self.doc, self.active_part, &[id]);
        let relations = commands.len().saturating_sub(1);
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("curve.delete", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.selected_curve = None;
        self.clear_point_selection();
        self.mark_changed();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        self.say(match relations {
            0 => format!("Deleted {name}"),
            1 => format!("Deleted {name} and the relation naming it"),
            n => format!("Deleted {name} and the {n} relations naming it"),
        });
    }

    /// The commands that take one curve out of a held loop, and what they did; or, refused, what
    /// to say instead.
    ///
    /// Free of the `App` for the reason `after_deleting` is: it is the whole of the delete that a
    /// test has to see, the commands and the check.
    pub(crate) fn loop_member_deletion(
        doc: &Document,
        part: PartId,
        id: CurveId,
        mates: &[CurveId],
        wall_thickness: f64,
        explain: &dyn Fn(&fieldform_field::CompileError) -> String,
    ) -> Result<(Vec<Box<dyn fieldform_ops::Command>>, curve_edits::LoopMemberDeleted), String> {
        let name = doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
        let taken = delete_loop_member_commands(doc, part, id, mates, wall_thickness);
        let (Some((commands, after, done)), Some(before)) = (taken, doc.parts.get(&part)) else {
            return Err(format!("{name} could not be taken out of its loop"));
        };
        // Refused here rather than committed into a part that cannot be drawn, by the rule and in
        // the words a block's delete is refused by: only for breaking a part that builds. A
        // drafted extrude with sharp corners, say, cannot be the wall this makes it. Asked of the
        // graph left alone, a part already broken elsewhere refused every curve out of every loop.
        if let Some(why) = Self::delete_breaks(doc, &before.graph, &after, explain) {
            return Err(Self::delete_refused(&name, &why));
        }
        Ok((commands, done))
    }

    /// Deletes a curve blocks are built from, and says which of them are waiting now. See
    /// `delete_held_curve_commands`; this is the part that needs the window.
    pub(crate) fn delete_held_curve(&mut self, id: CurveId) {
        let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
        let part = self.active_part;
        let Some((commands, after, let_go)) = delete_held_curve_commands(&self.doc, part, id) else {
            self.warn(format!("{name} could not be taken out of the blocks that read it"));
            return;
        };
        // Refused only for breaking a part that builds, by the rule every delete is held to.
        if let Some(before) = self.doc.parts.get(&part)
            && let Some(why) =
                Self::delete_breaks(&self.doc, &before.graph, &after, &|e| self.explain_compile(e))
        {
            self.warn(Self::delete_refused(&name, &why));
            return;
        }
        let waiting = self.waiting_said(&let_go);
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("curve.delete", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.selected_curve = None;
        self.clear_point_selection();
        self.mark_changed();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        self.say(format!("Deleted {name}{waiting}"));
    }

    /// What follows a curve's going for the blocks that let go of it: nothing when none did,
    /// else which are waiting now -- said, because the block stops being drawn and nothing else
    /// on screen says why.
    pub(crate) fn waiting_said(&self, let_go: &[NodeId]) -> String {
        let labels: Vec<String> = let_go.iter().map(|n| self.body_label(*n)).collect();
        match labels.len() {
            0 => String::new(),
            1 => format!(". {} is waiting to be told which curve to build from, and is not drawn until then", labels[0]),
            _ => format!(". {} are waiting to be told which curves to build from, and are not drawn until then", labels.join(" and ")),
        }
    }

    /// Takes a curve out of every block of this part that is built from it, and keeps the curve.
    ///
    /// **The other way out a curve did not have**, beside Delete: the row menu's item on a curve
    /// drawn inside a block, and the same edit as Clear on that block's own row. One graph edit,
    /// so one undo step. Blocks that merely aim along it or stand on it are not touched.
    pub(crate) fn release_curve(&mut self, id: CurveId) {
        let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
        self.release_curves(&[id], &name);
    }

    /// [`Self::release_curve`] for a profile's row: a block names one curve of the loop it is
    /// built on, any of them, so the row is let go of by whichever of its curves the block names.
    pub(crate) fn release_curves(&mut self, curves: &[CurveId], name: &str) {
        let mut graph = self.part_graph().clone();
        let mut ids: Vec<NodeId> = graph.nodes().map(|(node, _)| node).collect();
        ids.sort_by_key(|node| node.raw());
        let let_go: Vec<NodeId> = ids
            .into_iter()
            .filter(|node| {
                graph.node_mut(*node).is_some_and(|n| {
                    // Every curve asked, not the first that answers: a loft may pass through
                    // two rows' worth of them.
                    curves.iter().fold(false, |any, curve| n.stop_building_on(*curve) || any)
                })
            })
            .collect();
        if let_go.is_empty() {
            self.say(format!("No block here is built from {name}"));
            return;
        }
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        let waiting = self.waiting_said(&let_go);
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!("Took {name} out{waiting}"));
    }

    /// The blocks of this part built from `curve` -- the ones [`Self::release_curve`] would
    /// empty -- by their labels, for the menu item that names them.
    pub(crate) fn built_from(&self, curve: CurveId) -> Vec<String> {
        self.built_from_any(&[curve])
    }

    /// The same, for a profile's row, which a block reads through any one of its curves.
    pub(crate) fn built_from_any(&self, curves: &[CurveId]) -> Vec<String> {
        let mut found: Vec<(NodeId, String)> = self
            .part_graph()
            .nodes()
            .filter(|(_, n)| curves.iter().any(|curve| n.is_built_on(*curve)))
            .map(|(node, _)| (node, self.body_label(node)))
            .collect();
        found.sort_by_key(|(node, _)| node.raw());
        found.into_iter().map(|(_, label)| label).collect()
    }

    /// Deletes one curve out of a loop that blocks are built on, and says what followed from it.
    /// See `delete_loop_member_commands`; this is the part that needs the window.
    pub(crate) fn delete_loop_member(&mut self, id: CurveId, mates: &[CurveId]) {
        let name = self.doc.curve(id).map(|c| c.name.clone()).unwrap_or_default();
        let deleting = Self::loop_member_deletion(
            &self.doc,
            self.active_part,
            id,
            mates,
            self.next.wall_thickness,
            &|e| self.explain_compile(e),
        );
        let (commands, done) = match deleting {
            Ok(deleting) => deleting,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let walled: Vec<String> = done.walled.iter().map(|n| self.body_label(*n)).collect();
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("curve.delete", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.selected_curve = None;
        self.clear_point_selection();
        self.mark_changed();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        let joints = match done.joints {
            0 => String::new(),
            1 => " and the joint at its end".to_string(),
            n => format!(" and the {n} joints at its ends"),
        };
        let waiting = self.waiting_said(&done.let_go);
        // Nothing read the loop as a section: what there is to say is what let go, if anything.
        let no_section_read_it = walled.is_empty() && !done.let_go.is_empty();
        self.say(if no_section_read_it {
            format!("Deleted {name}{joints}{waiting}")
        } else if walled.is_empty() {
            format!("Deleted {name}{joints}; the wall runs along what is left{waiting}")
        } else {
            format!(
                "Deleted {name}{joints}. The loop is open now, so {} is a wall {} mm thick \
                 along what is left — tick Wall off in the block once it closes again{waiting}",
                walled.join(" and "),
                self.next.wall_thickness,
            )
        });
    }

    /// Deletes whichever body is selected, from the tree or the viewport.
    ///
    /// A key rather than another button: the panel already has a Delete inside each block, and
    /// the thing missing was a way to do it from where the selection was made.
    pub(crate) fn delete_selected_body(&mut self) {
        let Some(id) = self.selected_body else {
            self.warn("Nothing selected");
            return;
        };
        let label = self.body_label(id);
        // A refused delete has said why, and a "Deleted" after it would be the line a reader
        // trusts and the wrong one.
        if self.delete_body(id) {
            self.say(format!("Deleted {label}"));
        }
    }
}
