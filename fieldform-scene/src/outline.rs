// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The 2D section a profile-based block is built from, said once for every block that has one.
//!
//! # Why this is a type
//!
//! An extrude, a revolve and a sweep each read one thing in a plane and do something different
//! with it. That thing was a closed loop and nothing else, named by a bare [`CurveId`] on each of
//! the three — and an open curve was refused, correctly, because an open curve has no inside.
//! What an open curve wants is a thickness, and with one it is a rib rather than a solid.
//!
//! That is one idea, so it is one type (D9): a closed profile, or a curve given a thickness and a
//! side. Three blocks spelling "a curve and how thick" three ways would be three panels to keep
//! alike and a wall that could be handed from an extrude to a sweep only by retyping it.
//! the outline-wall-spec design note carries the decisions.
//!
//! # The shared type carries its own wiring
//!
//! [`Outline::curve`] is the graph edge and [`Outline::params`] is the number, so a block that
//! holds an `Outline` gains both without anybody remembering — the arrangement [`crate::Ends`],
//! [`crate::Frame`] and [`crate::Sweep`] already have.
//!
//! # Not `Section`, and not `Profile`
//!
//! Both names were taken: [`crate::Section`] is how a sweep holds what it carries, and a profile
//! is both the kernel's closed loop and the document object a group of curves becomes.

use serde::{Deserialize, Serialize};

use crate::graph::ParamRef;
use crate::id::CurveId;

/// Which side of its curve a wall's material lies on.
///
/// Left and right as seen walking along the curve in its own direction, looking down the plane's
/// normal. On a closed loop they are inside and outside, and which is which is read from the
/// loop's own winding where the wall is built — never stored, because a stored "inside" would be
/// a second description of something the points already say.
///
/// The words are what the file carries, so they are chosen to still make sense to a reader that
/// has never seen this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WallSide {
    /// Half the thickness each way. The default, and the one that needs no flip to be right.
    #[default]
    Centre,
    Left,
    Right,
}

/// How a wall on an open curve stops at the curve's two ends. A closed loop has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WallEnds {
    /// Cut square to the curve, at its end.
    #[default]
    Square,
    /// A half-round of the wall's own thickness beyond each end.
    Round,
}

impl WallEnds {
    /// What the block calls it.
    pub fn label(self) -> &'static str {
        match self {
            WallEnds::Square => "Square",
            WallEnds::Round => "Round",
        }
    }
}

/// The 2D section a profile-based block is built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outline {
    /// A closed loop, read as a signed distance. What every profile-based block read before
    /// there was a choice, so it is what a document written before this means.
    ///
    /// Names *one* curve of the loop, any of them; the loop is walked from it rather than
    /// stored. See `fieldform_field::profile_loop` for why.
    Closed { profile: CurveId },
    /// A curve given a thickness: the band of material along it.
    Wall {
        /// Any curve of an open chain or of a closed loop, walked the same way.
        curve: CurveId,
        /// The whole thickness, in millimetres — not half of it, whichever side it is on.
        thickness: ParamRef,
        side: WallSide,
        /// Read only on an open chain.
        ends: WallEnds,
    },
}

impl Outline {
    /// The closed profile every block had before this type existed.
    ///
    /// Offered so migrating a block that held a bare curve is a substitution.
    pub fn closed(profile: CurveId) -> Self {
        Self::Closed { profile }
    }

    /// The curve this section is read from, whichever kind it is.
    ///
    /// The whole of the reference an `Outline` creates. `Node::curves` is to go through this
    /// once a block holds one, so that nothing else has to know a section can be a wall at all;
    /// until then no block does, and the three that will still name their curve directly.
    pub fn curve(&self) -> CurveId {
        match self {
            Outline::Closed { profile } => *profile,
            Outline::Wall { curve, .. } => *curve,
        }
    }

    /// The same, to be redirected when curves are renumbered — as a paste does.
    pub fn curve_mut(&mut self) -> &mut CurveId {
        match self {
            Outline::Closed { profile } => profile,
            Outline::Wall { curve, .. } => curve,
        }
    }

    /// The numbers it carries, so they take part in parameter resolution like any other.
    ///
    /// A thickness left out of this could be shown by a panel and reached by nothing else: not
    /// named, not driven, not dragged.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let thickness = match self {
            Outline::Closed { .. } => None,
            Outline::Wall { thickness, .. } => Some(thickness),
        };
        thickness.into_iter()
    }

    /// The same, to be edited.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Outline::Closed { .. } => Vec::new(),
            Outline::Wall { thickness, .. } => vec![thickness],
        }
    }

    /// Whether this is a curve given a thickness rather than a loop with an inside.
    pub fn is_wall(&self) -> bool {
        matches!(self, Outline::Wall { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall(curve: CurveId) -> Outline {
        Outline::Wall {
            curve,
            thickness: ParamRef::Literal(2.0),
            side: WallSide::Left,
            ends: WallEnds::Round,
        }
    }

    /// The edge a block gains by holding an outline is the same edge for both kinds, which is
    /// what lets `Node::curves` ask without knowing which it has.
    #[test]
    fn an_outline_names_its_curve_whichever_kind_it_is() {
        let (a, b) = (CurveId::new(), CurveId::new());
        for mut outline in [Outline::closed(a), wall(a)] {
            assert_eq!(outline.curve(), a);
            *outline.curve_mut() = b;
            assert_eq!(outline.curve(), b, "a paste must be able to redirect it");
        }
    }

    /// A closed profile has nothing to type, and a wall's thickness is reachable by the same
    /// route every other number on a block is.
    #[test]
    fn only_a_wall_carries_a_number_and_it_can_be_edited() {
        let id = CurveId::new();
        assert!(Outline::closed(id).params().next().is_none());

        let mut outline = wall(id);
        assert_eq!(outline.params().collect::<Vec<_>>(), vec![&ParamRef::Literal(2.0)]);
        *outline.params_mut()[0] = ParamRef::Literal(3.5);
        assert!(matches!(
            outline,
            Outline::Wall { thickness: ParamRef::Literal(t), .. } if t == 3.5
        ));
    }

    /// The format names these by word, so the words are pinned: renaming a variant would
    /// otherwise change what every saved wall means without anything failing.
    #[test]
    fn the_words_a_file_carries_are_the_ones_the_spec_names() {
        let word = |v: &dyn erased::Word| v.word();
        assert_eq!(word(&WallSide::Centre), "\"centre\"");
        assert_eq!(word(&WallSide::Left), "\"left\"");
        assert_eq!(word(&WallSide::Right), "\"right\"");
        assert_eq!(word(&WallEnds::Square), "\"square\"");
        assert_eq!(word(&WallEnds::Round), "\"round\"");
        // And what is absent from a file is the default, so these are part of the format too.
        assert_eq!(WallSide::default(), WallSide::Centre);
        assert_eq!(WallEnds::default(), WallEnds::Square);
    }

    /// The one thing the test above needs of two unrelated enums: their word on the wire.
    mod erased {
        pub(crate) trait Word {
            fn word(&self) -> String;
        }
        impl<T: serde::Serialize> Word for T {
            fn word(&self) -> String {
                serde_json::to_string(self).expect("a unit variant serialises")
            }
        }
    }
}
