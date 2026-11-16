// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The draft overlay's settings, and the words a direction is described in.

use crate::*;

/// What the shared direction editor calls its choices, which depends on what the direction is
/// for. An extrude *sweeps along a line*; a draft read has a *vector*, and the curve you pick is
/// that vector. Same editor, same type, different nouns — the nouns are the only thing that
/// differs, and they are the thing a person reads.
#[derive(Clone, Copy)]
pub(crate) struct DirectionWords {
    /// The choice that takes a curve.
    pub(crate) along: &'static str,
    pub(crate) along_hint: &'static str,
    /// The row that names the curve, once that choice is made.
    pub(crate) curve_row: &'static str,
    /// Why the choice is disabled, when no curve can be picked.
    pub(crate) none_hint: &'static str,
}

pub(crate) const SWEEP_WORDS: DirectionWords = DirectionWords {
    along: "Line",
    along_hint: "Lean it the way a curve runs, end to end. Where the curve sits does not \
                 matter — only which way it points — so it need not touch the \
                 profile or be anywhere near it.",
    curve_row: "Line:",
    none_hint: "Needs a curve that is not part of the profile. Every curve here is one the \
                profile is made of, and a profile cannot be swept along itself — draw \
                another line first.",
};

pub(crate) const DRAFT_WORDS: DirectionWords = DirectionWords {
    along: "Vector",
    along_hint: "A curve whose direction, start to end, is the draft vector. A line — a \
                 two-point curve — is the natural pick. Where it sits does not matter, \
                 only which way it points.",
    curve_row: "Curve:",
    none_hint: "No curves in the document yet. Add a line and this can point along it.",
};

/// The draft-angle overlay's settings. A view tool, like the section: it changes what is drawn
/// and nothing about the part, so it is application state and not the document's.
pub(crate) struct DraftAnalysis {
    pub(crate) enabled: bool,
    /// Which way the mould pulls — the same type an extrude is aimed with, edited with the same
    /// controls, so a line picked here is a pick an extrude could make (D9).
    pub(crate) pull: fieldform_scene::Sweep,
    pub(crate) angle_deg: f64,
    pub(crate) tolerance_deg: f64,
    /// Pull the other way: the read for the other half of the mould.
    pub(crate) flipped: bool,
}

impl Default for DraftAnalysis {
    fn default() -> Self {
        Self {
            enabled: false,
            // Straight up, which is where a mould opens until someone says otherwise.
            pull: fieldform_scene::Sweep::Toward {
                xyz: [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(1.0)],
            },
            // The degree or two most mouldings ask for, with a degree of blur about it.
            angle_deg: 3.0,
            tolerance_deg: 1.0,
            flipped: false,
        }
    }
}
