// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How a loft gets from one section to the next. the loft-spec design note §4.

use serde::{Deserialize, Serialize};

/// The continuity of a loft's interpolation between its sections.
///
/// **Its own type, not [`crate::RampShape`]**, which spells how a *law* eases along a path.
/// This is how an interpolation passes *through* a section — a different question that happens
/// to share the G-words — and sharing the type would be D9 applied to two things that only
/// sound alike: a ramp has a G2 and a loft does not, and a loft's G1 is a Hermite through
/// neighbouring sections where a ramp's is a smoothstep between two ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoftSmooth {
    /// Straight from each section to the next: creased at every interior section, which is
    /// sometimes exactly what was drawn.
    #[default]
    G0,
    /// Through every section without a crease: Hermite, with each section's slope taken from
    /// its two neighbours and one-sided at the two ends.
    G1,
}

impl LoftSmooth {
    pub const ALL: [LoftSmooth; 2] = [LoftSmooth::G0, LoftSmooth::G1];

    /// What the block calls it — the same words a blend's continuity uses, for the same idea.
    pub fn label(self) -> &'static str {
        match self {
            LoftSmooth::G0 => "G0",
            LoftSmooth::G1 => "G1",
        }
    }

    pub(crate) fn is_default(self) -> bool {
        matches!(self, LoftSmooth::G0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document that says nothing means straight between sections.
    #[test]
    fn straight_is_what_a_loft_that_says_nothing_means() {
        assert_eq!(LoftSmooth::default(), LoftSmooth::G0);
        assert!(LoftSmooth::G0.is_default() && !LoftSmooth::G1.is_default());
        assert_eq!(LoftSmooth::ALL.map(|s| s.label()), ["G0", "G1"]);
    }
}
