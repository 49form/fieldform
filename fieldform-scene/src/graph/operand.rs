// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The one slot a block reads a field through.

use crate::NodeId;
use std::collections::HashMap;

/// A slot a block reads a field through: filled, or empty and waiting to be told.
///
/// **A named type rather than thirteen `Option<NodeId>`s, because D9 wants the shared type to
/// carry its own wiring.** [`crate::Node::for_each_operand_slot`] and `operand_slots_mut` are what
/// `for_each_operand` and `remap_inputs` read, so a block that gains an operand gains its graph
/// edge, its remap and its cycle check without anybody remembering — the arrangement
/// `frames`/`frames_mut` already has. The alternative, an `Option` per field, is a slot that only
/// claims to be shared: the next block to grow one has to be added by hand to each walk, and the
/// walks that use a wildcard arm would compile while answering nothing.
///
/// **And `Node` already holds an `Option` meaning something else.** `Loft::spine` is "absent on
/// purpose, derive the centreline from the sections" — not a hole, not an error, nothing to fill
/// in. A bare `Option<NodeId>` for "empty and in error" would make the two read alike at the point
/// of use, which is the kind of sameness that D9 exists to prevent.
///
/// Empty is a legal, savable state, reached by dragging a block out of a slot
/// (`notebook-spec.md` §7.2) or by *Remove contents* (§6.5) and by nothing else. A block with one
/// cannot be evaluated — there is no value a missing input could supply that does not break the
/// `ScalarField` contract — so the compile refuses it by name and the viewport prunes that body
/// while drawing its siblings. What it is *not* is a reason for the document to stop opening: a
/// ramp with nothing to read has been exactly this since `ramp-decision.md` §7.2.
///
/// No `Deref`, no `unwrap`, no `expect`: every reader asks [`Self::source`] and answers for the
/// `None`, which is the whole point of giving the hole a type.
///
/// **One sharp edge, found the day this was written: two empty slots are `==`.** The derive says
/// so, and it is the right answer for a derive — but it is almost never the question being asked.
/// A blend's panel compared its two slots to warn "both rows are the same block, so this subtracts
/// a body from itself"; with both slots emptied that warning fired about a block reading neither.
/// Code asking *"do these two slots name one block"* has to say so, which is what
/// `blocks::operands::same_block_twice` does by comparing `source()` and requiring `Some`. Equality
/// here means "these two slots are in the same state", nothing more.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Operand(Option<NodeId>);

impl Operand {
    /// A slot with nothing in it.
    pub fn empty() -> Self {
        Self(None)
    }

    /// A slot reading `id`.
    pub fn filled(id: NodeId) -> Self {
        Self(Some(id))
    }

    /// Whether the slot is waiting to be told what to read.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The block this slot names, when it names one — so `Node::inputs` can make the graph edge,
    /// and so every reader has to say what it does about the hole.
    pub fn source(&self) -> Option<NodeId> {
        self.0
    }

    /// Puts a block in the slot, over whatever was there.
    pub fn fill(&mut self, id: NodeId) {
        self.0 = Some(id);
    }

    /// Empties the slot and hands back what was in it, so §7.2's drag-out is one call: the gesture
    /// needs both halves at once — the slot emptied, and the block it held to put at the top level.
    pub fn clear(&mut self) -> Option<NodeId> {
        self.0.take()
    }

    /// Follows a paste's renumbering, the way every other edge does. An empty slot stays empty;
    /// an id the map does not mention is left alone, which is what `remap_inputs` means by it.
    pub(crate) fn remap(&mut self, map: &HashMap<NodeId, NodeId>) {
        if let Some(id) = self.0
            && let Some(moved) = map.get(&id)
        {
            self.0 = Some(*moved);
        }
    }
}

/// So the construction sites read `input: id.into()` rather than naming the type at each one.
impl From<NodeId> for Operand {
    fn from(id: NodeId) -> Self {
        Self::filled(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_operand_names_no_block() {
        let slot = Operand::empty();
        assert!(slot.is_empty());
        assert_eq!(slot.source(), None, "so every walk sees no edge at all");
        assert_eq!(Operand::default(), Operand::empty(), "and a default slot is an empty one");
    }

    /// **Filling and clearing are the only ways in and out**, which is what makes the hole
    /// something a reader can be made to answer for rather than a value it might miss.
    #[test]
    fn filling_a_slot_is_the_only_way_it_names_one() {
        let id = NodeId::new();
        let mut slot = Operand::empty();
        slot.fill(id);
        assert_eq!(slot.source(), Some(id));
        assert!(!slot.is_empty());
        assert_eq!(slot.clear(), Some(id), "clearing hands back what it held, for §7.2's drag-out");
        assert!(slot.is_empty(), "and leaves the slot empty rather than removing it");
        assert_eq!(slot.clear(), None, "clearing an empty slot is not an error");
        assert_eq!(Operand::from(id), Operand::filled(id));
    }

    #[test]
    fn remapping_follows_a_paste_and_leaves_an_empty_slot_empty() {
        let (was, now) = (NodeId::new(), NodeId::new());
        let map = HashMap::from([(was, now)]);
        let mut filled = Operand::filled(was);
        filled.remap(&map);
        assert_eq!(filled.source(), Some(now));
        let mut empty = Operand::empty();
        empty.remap(&map);
        assert!(empty.is_empty(), "nothing to renumber, and nothing invented");
        let other = NodeId::new();
        let mut untouched = Operand::filled(other);
        untouched.remap(&map);
        assert_eq!(untouched.source(), Some(other), "an id the map does not mention is left alone");
    }
}
