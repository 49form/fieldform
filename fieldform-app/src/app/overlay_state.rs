// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What the overlay and the block list read from the document each frame.

use crate::*;

impl App {
    /// Which point block the selection is, whether it is the point or a placement holding one.
    ///
    /// A `Node::Point` encloses no volume, so the field tracer draws nothing for it and there
    /// would otherwise be no sign of it in the viewport at all — a block in the list and empty
    /// space beside it. [`Self::point_positions`] is what gives every one of them a position
    /// you can see; this says which of them is being edited.
    pub(crate) fn selected_point_node(&self) -> Option<NodeId> {
        let mut id = self.selected_body?;
        // Through any placements wrapping it: what is selected is the outermost node.
        loop {
            match self.part_graph().node(id)? {
                Node::Point { .. } => return Some(id),
                // `?`, so an empty slot answers what the arm below answers: there is no point
                // under a hole.
                Node::Transform { input, .. } => id = input.source()?,
                _ => return None,
            }
        }
    }

    /// What the overlay and the panels read off the active part: every point and plane that draws,
    /// the pick box and every parameter's value ([`fieldform_field::OverlayReadings`]), read once
    /// and kept until something they read moves.
    ///
    /// **Kept, because in most frames nothing they read has moved.** An orbit moves the camera and
    /// nothing else, and read afresh, the pick box asked every drawn body for its extent on every
    /// frame of one: 2.0 ms a frame for one sweep, 17 ms for a wing of nine blocks
    /// (`fieldform-io/examples/overlay_cost.rs`). Asked through [`Self::overlay_key`], which names
    /// everything they read but the document, and the count of the document's moves.
    pub(crate) fn overlay_readings(&self) -> std::rc::Rc<fieldform_field::OverlayReadings> {
        let key = self.overlay_key();
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        self.kept.readings.borrow_mut().get(key, |key| key.read(&self.doc, &numbers))
    }

    /// The application's view of the active part as the overlay reads it, with the count of the
    /// document's moves: what [`Self::overlay_readings`] is kept against.
    pub(crate) fn overlay_key(&self) -> OverlayKey {
        // Asked of the part's bodies only while something is drawn as a mesh, which is rarely, so
        // most frames make this without walking the graph.
        let meshed = if self.outputs.instead_of_all || !self.outputs.instead_of.is_empty() {
            let mut ids: Vec<NodeId> = self
                .part_graph()
                .roots()
                .into_iter()
                .filter(|id| self.drawn_as_mesh(*id))
                .collect();
            ids.sort();
            ids
        } else {
            Vec::new()
        };
        OverlayKey {
            moves: self.document_moves,
            part: self.active_part,
            resolution: self.mesh_resolution,
            // A joint goes unseen with its row, by the rule its row's eye reads: hiding the
            // profile hides everything in it, the points that hold it included.
            hidden_with_rows: points_hidden_with_rows(
                &self.doc,
                self.active_part,
                &self.curve_ui.hidden,
                &self.curve_ui.profiles.all_rows(),
            ),
            meshed,
        }
    }

    /// What the block list reads off the active part: where each block is drawn, which blocks are
    /// bodies, and the order of the rows ([`fieldform_scene::ListReadings`]), read once and kept
    /// until the document moves or another part is shown.
    ///
    /// **Kept for the reason the overlay's readings are:** the list is drawn every frame, and in
    /// most frames nothing has moved. Read afresh, the order of the rows was worked out three times
    /// a frame — for the rows, and for the summary above them and its hover — and once more for
    /// every import's row, and every row asked the graph for its bodies
    /// (`fieldform-io/examples/list_cost.rs`).
    /// What the notebook says is wrong with each block of the active part, as both views draw it
    /// ([`graph_view::BlockStatuses`]), kept until the document moves.
    pub(crate) fn block_statuses(&self) -> std::rc::Rc<graph_view::BlockStatuses> {
        let key = ListKey { moves: self.document_moves, part: self.active_part };
        self.kept.statuses.borrow_mut().get(key, |key| {
            let graph = &self
                .doc
                .parts
                .get(&key.part)
                .expect("active_part must name a part in the document")
                .graph;
            graph_view::BlockStatuses::of(&self.doc, graph)
        })
    }

    pub(crate) fn list_readings(&self) -> std::rc::Rc<fieldform_scene::ListReadings> {
        let key = ListKey { moves: self.document_moves, part: self.active_part };
        self.kept.list.borrow_mut().get(key, |key| key.read(&self.doc))
    }

    /// Every point block in the part that is not hidden, with where it is:
    /// [`fieldform_field::point_positions`], which says why every one is drawn, as the overlay's
    /// readings keep it.
    pub(crate) fn point_positions(&self) -> Vec<(NodeId, Point3<f64>)> {
        self.overlay_readings().points.clone()
    }

    /// Every plane block in the part that is not hidden, as a point on it and the way it faces:
    /// [`fieldform_field::plane_frames`], as the overlay's readings keep it.
    /// [`Self::selected_plane`] stays: the gizmo needs to know which one is being edited, which is
    /// a different question from which ones exist.
    pub(crate) fn plane_frames(&self) -> Vec<(NodeId, Point3<f64>, Rotation3<f64>)> {
        self.overlay_readings().planes.clone()
    }

    /// The selected body's plane, as the point it passes through and the direction it faces,
    /// both after any placements above it.
    ///
    /// A half-space has no edges, so the tracer fills the screen with it and there is nothing
    /// to see *of* it — which is true to the geometry and useless for aiming one. The preview
    /// this feeds is a finite stand-in for an infinite thing, drawn at a size taken from the
    /// scene rather than from the plane, since the plane has no size to take.
    pub(crate) fn selected_plane(&self) -> Option<(NodeId, Point3<f64>, Vector3<f64>)> {
        let mut id = self.selected_body?;
        loop {
            match self.part_graph().node(id)? {
                Node::Plane { origin, flipped, .. } => {
                    let (origin, flipped) = (*origin, *flipped);
                    let facing = if flipped { -1.0 } else { 1.0 };
                    // Composed from the roots down, so this is right whether the selection is
                    // the outermost placement or the plane inside it.
                    let place = self.placement_above(id);
                    return Some((
                        id,
                        place * self.resolve_place(&origin),
                        place.isometry.rotation * Vector3::z() * facing,
                    ));
                }
                // As above: nothing is under an empty slot, which is the `None` the arm below
                // gives for anything else.
                Node::Transform { input, .. } => id = input.source()?,
                _ => return None,
            }
        }
    }
}
