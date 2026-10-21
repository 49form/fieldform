// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a draft does to a section's corners: rounds them, or mitres them.
//!
//! **A draft is an offset, and that is the whole of why there is a choice here.** Offsetting a
//! shape outward moves every wall out along its own normal, which is exactly what a draft
//! angle means — and at a convex corner there is no wall to move, so the offset sweeps an arc
//! of its own radius. The corner rounds, and rounds further as the draft grows. That is
//! geometrically correct and is often not what was drawn: a moulding drawing usually means the
//! two walls to stay planes and meet at an edge.
//!
//! Both are right answers to different questions, so both ship and the block says which — the
//! owner's ruling, 2026-10-03.
//!
//! # One type, because an extrude's draft and a sweep's are one construction
//!
//! They already are: a drafted sweep along a straight path is the drafted extrude of the same
//! section, pointwise (`sweep-v2-spec.md` §8). So the thing that says how their corners are cut
//! is one type on both blocks rather than two spellings of one idea (D9), and a loft's draft
//! will take the same one.
//!
//! It sits on the **block** rather than inside the draft, because an extrude has two drafts —
//! one per side — and a part is moulded one way. A block with no draft carries it harmlessly
//! and the panel shows the row only when there is a draft for it to mean something about.

use serde::{Deserialize, Serialize};

/// How a draft treats the section's corners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Corners {
    /// The offset's own arc: the corner rounds by however far the wall has moved. What every
    /// drafted solid did before there was a choice, so it is what a document that says nothing
    /// means.
    #[default]
    Round,
    /// The two walls stay planes and meet at an edge. Each corner goes to `v + t·m`, with `m`
    /// the miter vector of its two edges — `fieldform-field/examples/mitred_draft.rs` measures
    /// that this is the mitre, that the walls still lean at the draft angle, and where the
    /// mitre runs away at a sharp corner.
    Sharp,
}

impl Corners {
    pub const ALL: [Corners; 2] = [Corners::Round, Corners::Sharp];

    /// What the block calls it.
    pub fn label(self) -> &'static str {
        match self {
            Corners::Round => "Round",
            Corners::Sharp => "Sharp",
        }
    }

    /// Whether this is what a document that says nothing means, and so writes nothing.
    pub fn is_default(self) -> bool {
        matches!(self, Corners::Round)
    }
}

/// How long a mitre may get, as a multiple of the offset.
///
/// **A mitre's length is `1 / sin(half the included angle)`**, so it is 1.41 times the offset
/// at a right angle, 2.00 at sixty degrees and 7.66 at fifteen — it runs away as a corner
/// sharpens, and a needle-sharp corner would throw a spike across the part. Held to four,
/// which is the limit every other tool has and for the same reason; past it the corner keeps
/// the round answer, which is the one that is always finite. Measured in
/// `fieldform-field/examples/mitred_draft.rs`.
pub const MITRE_LIMIT: f64 = 4.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// A document that says nothing means round, which is what every drafted solid was.
    #[test]
    fn round_is_what_a_document_that_says_nothing_means() {
        assert_eq!(Corners::default(), Corners::Round);
        assert!(Corners::Round.is_default() && !Corners::Sharp.is_default());
        assert_eq!(Corners::ALL.len(), 2);
    }
}
