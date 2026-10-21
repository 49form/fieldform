// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The operand row: one slot, what it reads, and the error state while it reads nothing.
//!
//! **One row for every operand slot there is (D9).** A boolean's list, a blend's pair, a morph's
//! two and a pattern's one are all the same question — which block does this read — so they are
//! one widget, and the empty state the owner ruled on (`notebook-spec.md` §7.2, §6.5) is drawn
//! once rather than once per panel. Two panels showing the same hole would drift, and would teach
//! two habits for one idea.
//!
//! Not to be confused with `crate::operands`, which is the *rule*: what may go in a slot
//! ([`operand_choices_in`], [`operand_refusal`]). This is what the rule looks like on screen.

use crate::*;
use fieldform_scene::Operand;

/// The two edits at the end of an operand row, which are **not** the same gesture.
///
/// Kept as one type with a field apiece because the pair was a single `removable` flag, and the
/// flag could only ever offer the second of them — so a pair's slot and a two-operand boolean's
/// had no way out at all, which is the state §7.2's drag-out and §6.5's *Remove contents* both
/// reach and the panel could not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowEdits {
    /// **Empty the slot in place.** The row stays, the block keeps its operation, its radius and
    /// its name, and it waits to be told what to read. The block is *incomplete* afterwards:
    /// pruned from the viewport, and refused by name by anything that makes a statement about the
    /// part. Offered on any filled slot, whatever kind of block holds it.
    pub(crate) clear: bool,
    /// **Take the slot out of the list.** The row goes and the block is left **complete** with the
    /// operands that remain — no hole, nothing pruned. Only a boolean over two operands has this,
    /// because two is the least a boolean means anything with; a pair has no row to spare.
    pub(crate) remove: bool,
}

/// What a row's end offers, given whether it holds anything and whether its list can spare it.
///
/// Free of the `App` so the distinction can be asserted: an `App` needs a window and a device
/// before it will exist, which is the arrangement [`operand_choices_in`] already uses.
pub(crate) fn row_edits(filled: bool, over_two: bool) -> RowEdits {
    RowEdits { clear: filled, remove: over_two }
}

/// Whether a pair reads one block through both of its slots.
///
/// **Two empty slots are not two readings of one block**, and the plain `a == b` this replaced
/// said they were: [`Operand`] is `PartialEq`, so a blend with both slots emptied drew "both rows
/// are the same block, so this subtracts a body from itself" about a block reading nothing at all.
/// Each hole already has its own error on its own row; this warning is about a block *chosen*
/// twice, which is a different mistake with a different repair.
pub(crate) fn same_block_twice(a: Operand, b: Operand) -> bool {
    matches!((a.source(), b.source()), (Some(x), Some(y)) if x == y)
}

impl App {
    /// The blocks a slot of `consumer` may read, each carrying its own refusal.
    ///
    /// The addresses from [`Self::block_choices`] and the eligibility from [`operand_refusal`] in
    /// one place, because a field that offered one without the other would either suggest a block
    /// it then declines or decline one with the wrong reason. `pick_operand` attaches them the
    /// same way for the blocks still going through it.
    pub(crate) fn operand_offers(
        &self,
        consumer: NodeId,
        listed: &[(NodeId, String)],
    ) -> Vec<RefChoice<NodeId>> {
        // No clone of the graph, which `pick_operand` needs only because it holds `&mut self`:
        // this runs per frame for an expanded block and the list is as long as the part.
        let graph = self.part_graph();
        let mut offered = self.block_choices(listed);
        for choice in &mut offered {
            choice.refusal = operand_refusal(&self.doc, graph, consumer, choice.key);
        }
        offered
    }

    /// One operand slot: the block it reads, or the error state while it reads nothing.
    ///
    /// A filled slot draws as it always has — a field to type a name into or drop a block on,
    /// [`Self::reference_field_in`] carrying the whole of both. An empty one draws **in the error
    /// state**, because a block with a hole cannot be worked out: it is left out of the viewport
    /// while everything beside it carries on (`notebook-spec.md` §7.2), and *Measure*,
    /// wall-thickness and the promise check decline rather than answer about a part with a body
    /// silently missing.
    ///
    /// **Said on the row, because the row is where it is fixed.** The same reasoning the pair's
    /// duplicate-operand note already follows: a state somebody has to repair is announced beside
    /// the control that repairs it, not in a status line that has scrolled away by then.
    ///
    /// `over_two` is whether the list this slot sits in can spare a row — see [`RowEdits`], whose
    /// two edits this draws. Answers `true` when the row asked to be removed, which only its
    /// owner can do, the list's indices being its own business.
    pub(crate) fn operand_row(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        salt: impl std::hash::Hash,
        slot: &mut Operand,
        offered: &[RefChoice<NodeId>],
        over_two: bool,
    ) -> bool {
        let mut remove = false;
        ui.horizontal(|ui| {
            // **The label carries the error colour, not the box.** An empty text box already
            // means "nothing typed yet" here — which is why `reference_field_in` has no "clear"
            // and answers only with a reference — so colouring the box would make a hole and a
            // half-typed name look alike.
            match slot.is_empty() {
                true => {
                    ui.colored_label(ui.visuals().error_fg_color, label);
                }
                false => {
                    ui.label(label);
                }
            }
            // **A field, not a dropdown.** Type the name of a block or drop the block on it;
            // `reference_field` carries the whole of both and the reason. Nothing held shows the
            // field's own hint — "name, or drop a block" — which is exactly what the slot waits
            // for, so there is nothing to invent for the empty case.
            let shown = match slot.source() {
                Some(read) => self.shown_label(offered, read, "none"),
                None => String::new(),
            };
            if let Some(picked) = self.reference_field_in(
                ui,
                salt,
                slot.source(),
                &shown,
                offered,
                RefKind::Block,
                DropsFrom::block(|n| n),
            ) {
                slot.fill(picked);
            }
            // Read after the field, so the rest of the row shows the state the slot is about to
            // be in rather than the one this frame's typing left behind — the blend row's rule.
            let edits = row_edits(!slot.is_empty(), over_two);
            if slot.is_empty() {
                ui.colored_label(ui.visuals().error_fg_color, "empty — not drawn")
                    .on_hover_text(
                        "This block reads nothing here, so it cannot be worked out: it is left \
                         out of the viewport while everything beside it goes on being drawn. \
                         Type a block's name or drop a block on the field to fill it. Measure, \
                         wall thickness and the promise check decline while a slot is empty, \
                         rather than answering about a part with a body missing from it.",
                    );
            }
            if edits.clear
                && ui
                    .add(egui::Button::new("Clear").small())
                    .on_hover_text(
                        "Empty this slot. What it holds becomes a body of its own again and the \
                         block stays where it is, with its operation and its numbers, waiting to \
                         be told what to read instead — the same state dragging the block out of \
                         the slot leaves.",
                    )
                    .clicked()
            {
                // Nothing to put anywhere: a block no longer read becomes a top-level row when
                // the edit settles, which is what makes this and §7.2's drag-out one state.
                slot.clear();
            }
            if edits.remove
                && remove_button(
                    ui,
                    "Take this operand out and close the row up. It becomes a body of its own \
                     again, and the block is left complete with the operands that remain.",
                )
            {
                remove = true;
            }
        });
        remove
    }

    /// The single-input row: the slot, the hint under it, and the separator that closes the group.
    ///
    /// [`Self::pick_input`] with an [`Operand`] in place of its `NodeId`, so a one-input block gets
    /// the same row a boolean's operands get and the same empty state. **The blocks still going
    /// through `pick_input` do not**, which is a D9 debt rather than a design: this is the one place
    /// to forward it to once that signature takes a slot.
    pub(crate) fn operand_input(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: Operand,
    ) -> Operand {
        let choices = self.operand_choices(id);
        let offered = self.operand_offers(id, &choices);
        let mut slot = input;
        self.operand_row(ui, "Of:", (id, "edit_input"), &mut slot, &offered, false);
        self.reuse_hint(ui, id);
        ui.separator();
        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Emptying depends on the slot; removing depends on the list.** Collapsing the two into one
    /// flag is what left a pair with no way to empty a slot at all.
    #[test]
    fn a_filled_slot_can_always_be_emptied_and_only_a_list_over_two_can_lose_a_row() {
        assert_eq!(
            row_edits(true, false),
            RowEdits { clear: true, remove: false },
            "a pair's slot empties in place; there is no row to spare",
        );
        assert_eq!(
            row_edits(true, true),
            RowEdits { clear: true, remove: true },
            "and a list over two offers both, because they leave different states behind",
        );
        assert_eq!(
            row_edits(false, false),
            RowEdits { clear: false, remove: false },
            "nothing to empty out of an empty slot",
        );
        assert!(
            row_edits(false, true).remove,
            "but an empty row in a list over two can still be closed up, which is the repair \
             that makes the block complete again",
        );
    }

    #[test]
    fn two_empty_slots_are_not_the_same_block_read_twice() {
        let (one, other) = (NodeId::new(), NodeId::new());
        assert!(same_block_twice(one.into(), one.into()));
        assert!(!same_block_twice(one.into(), other.into()));
        assert!(
            !same_block_twice(Operand::empty(), Operand::empty()),
            "a block reading nothing at all is not a block subtracted from itself",
        );
        assert!(!same_block_twice(one.into(), Operand::empty()));
        assert!(!same_block_twice(Operand::empty(), one.into()));
    }
}
