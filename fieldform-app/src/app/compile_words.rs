// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Why a part will not compile, in words a person can act on.

use crate::*;

impl App {
    /// Why the current graph does not evaluate, if it does not.
    ///
    /// Compiling is how this is answered rather than inspecting the graph, so the reason the
    /// user is given is the same one the viewport acted on.
    pub(crate) fn compile_error(&self) -> Option<String> {
        self.compile_error.clone()
    }

    /// A compile refusal, with any node id in it replaced by what the block is called.
    ///
    /// **The difference between a document that looks broken and one that says what to
    /// change.** A blend against a lattice is refused by D8, which is correct — the radius
    /// would not be the radius asked for — but the consequence is that the viewport traces
    /// nothing and the export dialog never opens, and neither of those looks like a refusal.
    /// Until this named the block there was no way to tell which of several was at fault.
    ///
    /// Shared by the panel and by export so the two cannot describe the same refusal
    /// differently, which is the failure this codebase keeps meeting: one question, answered
    /// in two places.
    pub(crate) fn explain_compile(&self, e: &fieldform_field::CompileError) -> String {
        match e {
            fieldform_field::CompileError::BlendNeedsDistanceFields(id) => {
                let name = self
                    .body_name(*id)
                    .map(str::to_string)
                    .unwrap_or_else(|| self.derived_label(*id));
                e.to_string()
                    .replace(&format!("{id:?}"), &format!("\u{201c}{name}\u{201d}"))
            }
            // The same substitution, for the same reason: the refusal reaches a person, and
            // "node NodeId(2f1c..)'s slot 0 is empty" names nothing they can go and look at.
            // The word `node` goes with the id -- the way `curve` does in the `BadWall` arm
            // below -- so what is left is the block's own name owning the slot. The slot number
            // stays: it is the half of the message that says which hole to fill, and the block
            // it names is the holed one rather than whatever reads it, because that is the one
            // with a slot to fill.
            fieldform_field::CompileError::EmptySlot { node, .. } => {
                let name = self
                    .body_name(*node)
                    .map(str::to_string)
                    .unwrap_or_else(|| self.derived_label(*node));
                e.to_string()
                    .replace(&format!("node {node:?}"), &format!("\u{201c}{name}\u{201d}"))
            }
            // The same, for a block waiting for a curve: named, not numbered.
            fieldform_field::CompileError::WaitsForCurve { node } => {
                let name = self
                    .body_name(*node)
                    .map(str::to_string)
                    .unwrap_or_else(|| self.derived_label(*node));
                e.to_string()
                    .replace(&format!("node {node:?}"), &format!("\u{201c}{name}\u{201d}"))
            }
            // **The other answer to a loop that does not close**, said beside the first. Until
            // a block could be built on a wall, joining the ends was the only way on; now
            // giving the curve a thickness is one too, and it is a checkbox on the block the
            // refusal is about (`outline-wall-spec.md` §6).
            fieldform_field::CompileError::BadLoop(
                _,
                fieldform_field::LoopError::NotClosed { .. },
            ) => format!("{e} {OR_A_WALL}"),
            // A red cage is said as what it is -- open, with its hole count -- rather than as a
            // raw id. The two arms are kept apart because the fixes differ: an open cage is
            // closed with Fill Hole, a missing one is a document fault.
            fieldform_field::CompileError::OpenCage(_, why) => {
                format!("The cage is not a solid ({why}) \u{2014} fill its holes to see the surface")
            }
            fieldform_field::CompileError::MissingCage(id) => {
                format!("The part names cage {id}, which is not in the document")
            }
            // By the curve's name, since an id is nothing a person can go and look for.
            fieldform_field::CompileError::BadWall(id, _) => {
                let text = e.to_string();
                match self.doc.curves.get(id) {
                    Some(curve) => text.replace(
                        &format!("curve {id:?}"),
                        &format!("\u{201c}{}\u{201d}", curve.name),
                    ),
                    None => text,
                }
            }
            _ => e.to_string(),
        }
    }

    // `compile_error` used to have a body here: it compiled the display graph on every call,
    // and the panel called it once a frame. That is a full compile — cage subdivision, mesh
    // BVHs, crease normals — to answer a question whose answer changes only when the document
    // does, and on a 109 000-triangle import it was what made the whole interface slow.
    // `rebuild_render_resources` already compiles the same graph for picking, so the answer is
    // recorded there and this reads it.
}
