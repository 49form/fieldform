// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Choosing what a field refers to: the candidates, how each is labelled, and why one is refused.

use crate::*;

/// Where a number's own label goes.
///
/// **Because a row and a cell are different places.** A row's label belongs in the label column
/// every row shares, right-aligned against the numbers; a cell's is the axis letter, and belongs
/// inside the cell beside its own number. The same widget draws both, so it has to be told which
/// — guessing from the text would be one more rule to know.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Labelled {
    /// In the shared label column: `Radius:`, `At:`, `From:`.
    Row,
    /// Inside the cell, ahead of the number: `X`, `Y`, `Z`.
    Cell,
}

/// One candidate for a reference field: what it is called, and what it may be *typed* as.
///
/// **Two strings, because a label and an address are not the same fact.** The label is what the field
/// shows and the list offers; the address is the text that resolves to this choice, and a choice may
/// have none. A block gets an address only once somebody names it, since its label is what it *is*
/// and eight pipes on spokes all read "Pipe"; a curve's label is already unique, because
/// `unique_curve_name` makes it so on creation. Keeping them apart here is what lets one widget serve
/// both without knowing which it is holding.
pub(crate) struct RefChoice<K> {
    pub(crate) key: K,
    pub(crate) label: String,
    pub(crate) address: Option<String>,
    /// Why this block may not go in the field, if it may not (`operand_refusal`). A refused choice
    /// is still *known* — typing its name or dropping it finds it — but is never suggested and never
    /// taken, and what the person is told is this.
    pub(crate) refusal: Option<&'static str>,
}

impl<K> RefChoice<K> {
    /// A choice whose label is also its address — a curve, or anything else born with a unique name.
    pub(crate) fn named(key: K, name: &str) -> Self {
        Self { key, label: name.to_string(), address: Some(name.to_string()), refusal: None }
    }

    /// A choice that can only be typed once it has been named, which is every block.
    pub(crate) fn declared(key: K, label: String, address: Option<&str>) -> Self {
        Self { key, label, address: address.map(str::to_string), refusal: None }
    }
}

/// A row its V can make a variable or put back inside.
///
/// **Three kinds of row, and one act on each** (`block-ownership-spec.md` §10): the V writes the
/// stored ownership of the block the row stands for, and a name is a label on all three. They
/// were declared three ways until the curves' V wrote the curve ownership (refactor plan 2.19) —
/// a block by its name, a profile by a name of its own (`Document::profile_names`), and a plain
/// curve, born named, by a flag (`Document::declared_curves`) — and the two names moved a row
/// whenever one was typed for any other reason. Kept apart only for what the row is called when
/// the V says what it did.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declarable {
    Block(NodeId),
    /// A profile row: `key`, its loop's first curve, which the row is drawn under and called by,
    /// and `curve`, the one whose ownership its V writes — the curve that puts the row where it
    /// is ([`profile_v_curve_in`]). Not always the key: an extrude names whichever curve was
    /// picked, and the row goes inside the block that holds any of them.
    Profile { key: CurveId, curve: CurveId },
    Curve(CurveId),
}

impl Declarable {
    /// The block whose ownership the row's V writes.
    pub(crate) fn block(self) -> BlockKey {
        match self {
            Declarable::Block(id) => BlockKey::Node(id),
            Declarable::Profile { curve, .. } | Declarable::Curve(curve) => BlockKey::Curve(curve),
        }
    }
}

/// The rows a reference field offers while it has the focus: everything but what it already
/// holds, narrowed by what has been typed once the text is no longer just the settled name, and at
/// most ten.
///
/// **A refused row is offered and marked, not left out** — the owner's ruling when the gate was
/// built, and the reading it rests on: "a row that is missing answers nothing, and the question
/// somebody has while looking down this list is why their body is not an option." So each row
/// comes back with its refusal beside it, and the caller draws those weak, unclickable and with
/// the reason on hover. Leaving them out put the explanation behind an action — a person had to
/// type the name and submit it to be told why it would not go in, which is the moment they have
/// stopped looking for it.
///
/// **And the row the field already holds is kept too, for the same reason.** It used to be left
/// out, on the reading that offering what is already chosen is noise. Reported from the other
/// side: a twist is created over the selected body, so its `Of:` already held the cube — and the
/// list, which omitted exactly that one, read as "it will not let me pick the cube". The held row
/// is marked instead, so a list is never missing the thing somebody is looking for.
///
/// `typed` is lowercased by the caller. `current` is optional because a field may hold nothing yet,
/// and then there is nothing to leave out.
pub(crate) fn suggested<K: Copy + PartialEq>(
    choices: &[RefChoice<K>],
    typed: &str,
    showing: bool,
) -> Vec<(K, String, Option<&'static str>)> {
    choices
        .iter()
        // **A block that cannot go here is not listed** (the owner, 2026-11-07: a tool "should not
        // list items like points and curves as input in its field"). It used to be listed greyed
        // with the reason on hover; the reason is still said when one is typed or dropped, which is
        // the only way left to try it.
        .filter(|c| c.refusal.is_none())
        .filter(|c| !showing || typed.is_empty() || c.label.to_lowercase().contains(typed))
        .take(10)
        .map(|c| (c.key, c.label.clone(), c.refusal))
        .collect()
}

/// Which dragged rows a reference field can take, and how each becomes its kind of reference.
///
/// **A pair rather than two arguments**, because the two are one fact about the field — what it
/// accepts — and a field that took them separately would let a caller pass a block mapper and forget
/// the curve one without anything reading oddly.
///
/// A mapper rather than a bare flag because the key is not always the id: an `Attractor` wraps a
/// `CurveId` one way and a `NodeId` the other, and that wrapping is exactly what the caller knows and
/// the widget does not.
#[derive(Clone, Copy)]
pub(crate) struct DropsFrom<K> {
    pub(crate) block: Option<fn(NodeId) -> K>,
    pub(crate) curve: Option<fn(CurveId) -> K>,
}

impl<K> DropsFrom<K> {
    /// Takes a dragged block.
    pub(crate) fn block(f: fn(NodeId) -> K) -> Self {
        Self { block: Some(f), curve: None }
    }

    /// Takes a dragged curve.
    pub(crate) fn curve(f: fn(CurveId) -> K) -> Self {
        Self { block: None, curve: Some(f) }
    }

    /// Takes either, which only an attractor's target does.
    pub(crate) fn either(block: fn(NodeId) -> K, curve: fn(CurveId) -> K) -> Self {
        Self { block: Some(block), curve: Some(curve) }
    }
}

/// What kind of thing a reference field is asking for, which decides only what it *says*.
///
/// The behaviour is the same for all three — type a name, or pick from the list — so this carries
/// the wording rather than any logic. Separate from the key type because `Attractor` is a curve or a
/// block and could not say which from its type alone.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefKind {
    /// A block: a body, or any named block in the part.
    Block,
    /// A curve in the document.
    Curve,
    /// Either, which is what an attractor takes.
    Target,
    /// A ramp: what a driveable number is driven by. A block, but one with its own words, since
    /// "No block is called" is the wrong thing to say to someone typing a ramp's name.
    Ramp,
}

impl RefKind {
    /// The empty field's prompt, which is also where a curve field says that a drag will not work.
    pub(crate) fn hint(self) -> &'static str {
        match self {
            RefKind::Block => "name, or drop a block",
            RefKind::Curve => "name, or drop a curve",
            RefKind::Target => "name, or drop either",
            RefKind::Ramp => "name, or drop a ramp",
        }
    }

    /// Why a dropped block was refused — only reachable where a block can be dropped at all.
    pub(crate) fn why_not(self) -> &'static str {
        match self {
            RefKind::Block => {
                "cannot go here — a block cannot be made from something that is made from it"
            }
            RefKind::Curve => "cannot go here — this field takes a curve",
            RefKind::Target => "cannot be attracted to — it is made from this block",
            RefKind::Ramp => "cannot drive this — only a ramp can, and it must not read this block",
        }
    }

    pub(crate) fn nothing(self) -> &'static str {
        match self {
            RefKind::Block => "No block is called",
            RefKind::Curve => "No curve is called",
            RefKind::Target => "Nothing here is called",
            RefKind::Ramp => "No ramp is called",
        }
    }

    /// How to make one that can be typed.
    pub(crate) fn how(self) -> &'static str {
        match self {
            RefKind::Block => {
                "Click the dim V on a block's row to name it, and it can be referred to here."
            }
            RefKind::Curve => "Rename a curve on its own row to call it something you will remember.",
            RefKind::Target => "Name the block with the dim V on its row, or rename the curve.",
            RefKind::Ramp => "Name the ramp with the dim V on its row, or pick it from the list.",
        }
    }
}

/// Whether a number is continuous or counts things.
///
/// **Because a field showing 4 while holding 3.5 is the disagreement the count exists to avoid.**
/// A pattern's copy count is a `ParamRef`, and `fieldform_scene::pattern_count` rounds what it reads
/// — so the number built is always whole, and the *stored* number should be too wherever somebody
/// typed it here rather than drove it from elsewhere. Whole numbers are a property of the quantity,
/// which is the same reasoning the range already travels by: both are facts about what the number
/// *is*, and both belong with the caller that knows.
///
/// It is a grain rather than a flag so the third case has somewhere to go when one turns up.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Grain {
    /// Any value in range: a length, an angle, a ratio.
    Continuous,
    /// Whole numbers only, stepped one at a time and shown without decimals.
    Whole,
}

/// Whether a number offers to become a variable, and the one case where it should not.
///
/// **A position is one thing, so it gets one V.** Three coordinates that each offered to be
/// named put four marks on a row describing a single place, and the row-level one -- the one
/// that matters, because it is how a block comes to sit on a point -- was the least visible of
/// them. Asked for directly: "lets remove that and keep only the XYZ combined parameter as a V".
///
/// A number that is *already* named still shows its mark. Withholding it there would leave a
/// value driven from elsewhere looking like a plain number, which is worse than clutter; the
/// dim offer is what goes, not the bright fact.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Naming {
    /// A dim V inviting this number to become a variable. What almost every number gets.
    Offered,
    /// The mark only when there is already something to mark. For the coordinates of a place,
    /// whose naming happens once for the position rather than three times for its parts.
    OnlyWhenSet,
}
