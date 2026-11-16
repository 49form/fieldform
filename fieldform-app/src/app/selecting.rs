// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Selecting a body or a curve, and finding what a click landed on.

use crate::*;

impl App {
    /// Selects a body, releasing whatever else was claiming the manipulator.
    ///
    /// One thing at a time, because the manipulator can only be on one of them and something
    /// has to decide which. Without this a curve selected in the panel — and left selected,
    /// which is the normal way to keep its points on screen — kept the handles forever: a
    /// click on a sphere updated the selection and changed nothing visible, so the sphere
    /// looked unselectable rather than second in a queue.
    pub(crate) fn select_body(&mut self, id: Option<NodeId>) {
        self.selected_body = id;
        // **Picking the cage's body is how the cage gets its tools back** from a surface — the
        // same gesture that makes it the thing being worked on everywhere else. Any other body
        // leaves a surface as the control mesh: clicking the lattice to look at it is not a
        // request to stop editing what it follows.
        let cage_node = self
            .part_graph()
            .nodes()
            .find(|(_, n)| matches!(n, Node::Cage(c) if *c == self.active_cage))
            .map(|(node, _)| node);
        if let (Some(body), Some(cage_node)) = (id, cage_node)
            && self.control != ControlMesh::Cage
            && (body == cage_node || self.body_containing(cage_node) == Some(body))
        {
            self.set_control(ControlMesh::Cage);
        }
        if id.is_some() {
            self.selected_block = None;
            self.selected_curve = None;
            self.clear_point_selection();
            // Picking geometry is a more deliberate act than leaving the section selected, so
            // it takes the manipulator. The section stays *on* — only its claim on the handles
            // goes, which is the distinction the flag exists to make.
            self.section_selected = None;
            // Selecting a cage block makes its cage the one the cage tools act on.
            self.resync_active_cage();
        }
    }

    /// The same, the other way round.
    pub(crate) fn select_curve(&mut self, id: CurveId) {
        self.selected_curve = Some(id);
        self.selected_block = None;
        // Indices name points on *a* curve, so they mean something else the moment the curve
        // does.
        self.clear_point_selection();
        self.selected_body = None;
        self.selection.clear();
        self.section_selected = None;
    }

    /// Whether `node` draws the cage the UI edits.
    ///
    /// The manipulator treats such a body as the direct layer rather than as something to
    /// place — see [`App::begin_gizmo`] for why that distinction has to exist.
    pub(crate) fn is_active_cage_node(&self, node: NodeId) -> bool {
        matches!(self.part_graph().node(node), Some(Node::Cage(c)) if *c == self.active_cage)
    }

    /// The body — the root — whose composition reaches `node`, if any still does.
    ///
    /// A node stops being a body the moment something consumes it, but it is still *part* of
    /// one, and for anything the user can see or hide it is that containing body that decides.
    pub(crate) fn body_containing(&self, node: NodeId) -> Option<NodeId> {
        let graph = self.part_graph();
        self.doc
            .listed_bodies(graph)
            .into_iter()
            .find(|root| graph.subtree_of(*root).contains(&node))
    }

    /// The node that draws the cage the UI edits, if the part still has one and it is shown.
    ///
    /// Searched across the whole graph rather than among the bodies, and that distinction is
    /// the whole point. A cage stops being a body as soon as a boolean consumes it — but it is
    /// still the thing being edited, and the reason to boolean it against something was to
    /// shape the result by pushing its points around. Looking only at the roots meant that the
    /// moment a cage joined anything it vanished from the viewport, could not be picked, and
    /// the manipulator would not appear for it: the edit that motivated the boolean became
    /// impossible the instant the boolean was made.
    ///
    /// Hidden still means hidden, in either of the two ways it can be. The cage's own node may
    /// be hidden; or the body that now contains it may be, and drawing a control mesh in front
    /// of a solid nobody can see is a wireframe floating in empty space.
    pub(crate) fn visible_cage_node(&self) -> Option<NodeId> {
        let id = self
            .part_graph()
            .nodes()
            .find(|(_, n)| matches!(n, Node::Cage(c) if *c == self.active_cage))
            .map(|(id, _)| id)?;
        // A node's own eye only means anything while it is a body, because a body row is the
        // only place an eye is drawn. Hide a cage, then boolean it, and the flag outlives the
        // row that set it: the cage is held hidden by a switch that no longer exists anywhere
        // in the interface. Honouring that is how the cage becomes unrecoverable.
        //
        // Once something consumes it, what decides is the body that contains it — which does
        // still have a row, and so is still something the user can act on.
        if self.part_graph().roots().contains(&id) && !self.is_visible(id) {
            return None;
        }
        // No containing body means nothing draws it at all, which is not the same as it being
        // hidden — but it is equally not on screen.
        self.body_containing(id)
            .filter(|body| self.is_visible(*body))
            .map(|_| id)
    }

    /// Selects the body whose surface is under the cursor, if any is.
    ///
    /// Traces the composed field for a hit and then asks which body owns that point, rather
    /// than tracing each body separately: the bodies are already compiled as steps of the same
    /// field, so this is one trace and a handful of evaluations rather than one trace each.
    pub(crate) fn select_body_under(&mut self, origin: Point3<f32>, dir: Vector3<f32>) {
        // No field to trace is a click that hit nothing, not a click to ignore. Returning early
        // here meant that with nothing visible — everything hidden, or a graph that would not
        // compile — clicking could not clear a selection either, so the one gesture for "none
        // of this" stopped working exactly when there was nothing to select.
        let Some(field) = &self.pick_field else {
            self.selected_body = None;
            return;
        };
        let bounds = self.pick_bounds();
        let radius = ((bounds.1 - bounds.0).norm() * 0.5).max(1e-6);
        let eye = Point3::new(origin.x as f64, origin.y as f64, origin.z as f64);
        let d = Vector3::new(dir.x as f64, dir.y as f64, dir.z as f64).normalize();
        let reach = self.trace_reach(eye, bounds);
        let Some(hit) = trace_field(field, eye, d, reach, radius * 1e-4, 512) else {
            // A click on empty space clears the selection, which is what makes it possible to
            // select nothing on purpose.
            if self.selected_body.take().is_some() {
                self.say("Cleared the selection");
            }
            return;
        };
        // The trace stops within its own epsilon of the surface, so the body that owns the
        // point is within the same distance of zero. A tolerance any tighter would reject the
        // hit the trace just accepted.
        // **Only when it changes.** Every click through a body runs this, so reporting each one
        // would fill the log with the same line while someone turns a part around looking at it.
        let picked = field.body_at(hit, radius * 1e-3);
        // A list that is collecting takes the click. Only a body that is drawn can be clicked,
        // and a drawn body is one nothing consumes yet, so what arrives here is exactly what a
        // list can take — the one exception being the boolean's own result, which is also drawn.
        if let Some(block) = self.notebook.collecting
            && let Some(body) = picked
        {
            self.add_operand(block, body);
            return;
        }
        let was = self.selected_body;
        self.selected_body = picked;
        if let Some(id) = self.selected_body.filter(|&id| Some(id) != was) {
            let label = self.body_label(id);
            self.say(format!("Selected {label}"));
        }
    }

    /// Adds a body to a boolean's list, from a click in the viewport while the list collects.
    ///
    /// Through `replace_node`, so a body that would close a loop is refused with the reason and
    /// the graph is left as it was.
    pub(crate) fn add_operand(&mut self, block: NodeId, body: NodeId) {
        let Some(Node::Boolean { op, inputs }) = self.part_graph().node(block).cloned() else {
            // The block stopped being a boolean — turned into a fillet, or was deleted — so
            // there is no list to add to.
            self.notebook.collecting = None;
            return;
        };
        let what = self.body_label(block);
        if body == block {
            self.say(format!("That is {what} itself — click a body to add it"));
            return;
        }
        // Over the slots' sources rather than over the slots: an empty one in the list is
        // not this body, and a filled one is only when it reads it.
        if inputs.iter().any(|slot| slot.source() == Some(body)) {
            self.say(format!("{} is already in {what}", self.body_label(body)));
            return;
        }
        let added = self.body_label(body);
        let mut inputs = inputs;
        inputs.push(body.into());
        self.replace_node(block, Node::Boolean { op, inputs });
        self.say(format!("Added {added} to {what}"));
    }
}
