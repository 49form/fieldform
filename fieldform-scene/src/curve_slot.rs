// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The one slot a block reads a curve through.

use crate::id::CurveId;

/// A slot a block reads a curve through: filled, or empty and waiting to be told.
///
/// **[`crate::Operand`] for a curve, and for its reasons (D9).** A block that is built *from* a
/// curve -- a pipe along one, a sweep's section and its path, an extrude's profile -- held a bare
/// [`CurveId`], so there was no state in which the block existed and the curve did not. Two
/// things followed, and neither was reported by anything: a loft could not be made before its
/// sections were drawn, so a part had to be described bottom-up; and a curve could not be taken
/// *out* of a block, so deleting one a block read was refused with "delete that first", and the
/// curve was stuck inside the block until the block itself was thrown away.
///
/// Empty is a legal, savable state. A block with one cannot be evaluated, so
/// [`crate::Graph::incomplete`] names it, the viewport prunes that body while drawing its
/// siblings, and the paths that make a statement about the part refuse by name -- exactly what an
/// empty [`crate::Operand`] gets, through the same walk.
///
/// **Only the curve a block is built from**, not every curve it may name. A frame aimed along a
/// line, an end standing on one, a loft's centreline: each of those already has a way of naming
/// nothing (`Frame::At`, a typed point, `None` for "derived"), and a second empty state beside it
/// would be the same hole spelled twice.
///
/// No `Deref` and no `unwrap`: every reader asks [`Self::source`] and answers for the `None`.
/// Two empty slots are `==`, which means "in the same state" and never "name the same curve" --
/// the sharp edge `Operand` records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CurveSlot(Option<CurveId>);

impl CurveSlot {
    /// A slot with nothing in it.
    pub fn empty() -> Self {
        Self(None)
    }

    /// A slot reading `id`.
    pub fn filled(id: CurveId) -> Self {
        Self(Some(id))
    }

    /// Whether the slot is waiting to be told what to read.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The curve this slot names, when it names one.
    pub fn source(&self) -> Option<CurveId> {
        self.0
    }

    /// The same, to be redirected when curves are renumbered -- as a paste does. An empty slot
    /// has nothing to redirect.
    pub fn source_mut(&mut self) -> Option<&mut CurveId> {
        self.0.as_mut()
    }

    /// Puts a curve in the slot, over whatever was there.
    pub fn fill(&mut self, id: CurveId) {
        self.0 = Some(id);
    }

    /// Empties the slot and hands back what was in it.
    pub fn clear(&mut self) -> Option<CurveId> {
        self.0.take()
    }
}

/// So the construction sites read `curve: id.into()` rather than naming the type at each one.
impl From<CurveId> for CurveSlot {
    fn from(id: CurveId) -> Self {
        Self::filled(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_curve_slot_names_no_curve() {
        let slot = CurveSlot::empty();
        assert!(slot.is_empty());
        assert_eq!(slot.source(), None);
        assert_eq!(CurveSlot::default(), CurveSlot::empty(), "a default slot is an empty one");
    }

    #[test]
    fn filling_and_clearing_are_the_only_ways_in_and_out() {
        let id = CurveId::new();
        let mut slot = CurveSlot::empty();
        slot.fill(id);
        assert_eq!(slot.source(), Some(id));
        assert_eq!(slot.clear(), Some(id), "clearing hands back what it held");
        assert!(slot.is_empty(), "and leaves the slot empty rather than removing it");
        assert_eq!(slot.clear(), None, "clearing an empty slot is not an error");
        assert_eq!(CurveSlot::from(id), CurveSlot::filled(id));
    }

    #[test]
    fn a_paste_redirects_a_filled_slot_and_leaves_an_empty_one_empty() {
        let (was, now) = (CurveId::new(), CurveId::new());
        let mut filled = CurveSlot::filled(was);
        *filled.source_mut().expect("filled") = now;
        assert_eq!(filled.source(), Some(now));
        assert!(CurveSlot::empty().source_mut().is_none(), "nothing to renumber, nothing invented");
    }
}
