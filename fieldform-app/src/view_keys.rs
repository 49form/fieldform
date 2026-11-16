// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What the overlay and the block list were last built from, so each is rebuilt only when that changes.

use crate::*;

/// What the overlay's readings are read for: which part, the application's own view of it, and
/// the count that moves whenever the document may have ([`App::document_moves`]).
///
/// **Everything the readings read but the document is a field here, and is read from here.**
/// [`Self::read`] is handed this and the document, so no view setting can reach a reading without
/// being part of what it is kept against. A field left out would be a reading left on screen after
/// what it was read from had moved, with nothing to say so.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OverlayKey {
    pub(crate) moves: u64,
    pub(crate) part: PartId,
    /// The meshing resolution, one cell of which is the pick box's margin.
    pub(crate) resolution: usize,
    /// The joint points that go unseen with their profile row (`points_hidden_with_rows`).
    pub(crate) hidden_with_rows: Vec<NodeId>,
    /// The part's bodies the application draws as their mesh rather than traced
    /// ([`App::drawn_as_mesh`]), sorted.
    pub(crate) meshed: Vec<NodeId>,
}

impl OverlayKey {
    /// The readings, from the document and this. `numbers` is the application's reading of a
    /// number for drawing and sizing ([`App::resolve`]), which asks the document and nothing else.
    pub(crate) fn read(
        &self,
        doc: &Document,
        numbers: &dyn Fn(&ParamRef) -> Option<f64>,
    ) -> fieldform_field::OverlayReadings {
        let part = &doc
            .parts
            .get(&self.part)
            .expect("active_part must name a part in the document")
            .graph;
        // Asked only of the part's bodies (`drawn_graph`), which are what `meshed` was read from.
        let drawn_as_mesh = |id: NodeId| self.meshed.contains(&id);
        let inputs = fieldform_field::OverlayInputs {
            drawn_as_mesh: &drawn_as_mesh,
            hidden_with_rows: &self.hidden_with_rows,
            resolution: self.resolution,
            empty_mm: DEFAULT_BODY_MM,
            numbers,
        };
        fieldform_field::OverlayReadings::build(doc, part, &inputs)
    }
}

/// What the block list's readings are read for: which part, and the count that moves whenever the
/// document may have ([`App::document_moves`]).
///
/// **Only those two, because the list's readings read nothing else.** Where each block is drawn
/// and the order of the rows are the document's own: no view setting reaches either, so the key
/// is the overlay's without its view, and [`Self::read`] is handed nothing but this and the
/// document, so none can.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ListKey {
    pub(crate) moves: u64,
    pub(crate) part: PartId,
}

impl ListKey {
    /// The readings, from the document and this.
    pub(crate) fn read(&self, doc: &Document) -> fieldform_scene::ListReadings {
        let part = &doc
            .parts
            .get(&self.part)
            .expect("active_part must name a part in the document")
            .graph;
        fieldform_scene::ListReadings::build(doc, part)
    }
}

/// The block one curve control point sits on, if it sits on one.
///
/// A free function so the row can ask while `self` is already borrowed by the closure drawing it.
pub(crate) fn placed_on(app: &App, id: CurveId, index: usize) -> Option<NodeId> {
    app.doc.curves.get(&id).and_then(|c| c.placed_on(index))
}

/// The point blocks in the part, which are what a control point can sit on.
pub(crate) fn point_blocks(app: &App) -> Vec<(NodeId, String)> {
    let mut out: Vec<(NodeId, String)> = app
        .part_graph()
        .nodes()
        .filter(|(_, n)| matches!(n, Node::Point { .. }))
        .map(|(n, _)| (n, app.body_label(n)))
        .collect();
    out.sort_by_key(|(n, _)| n.raw());
    out
}
