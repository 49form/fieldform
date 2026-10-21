// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which way out of its plane a profile solid runs, said once for every block that runs out
//! of one.
//!
//! An extrude ran out of one side of its plane, or half each way when a flag said so. A third
//! way is two sides with a depth and a draft each — a boss standing proud of a wall by one
//! amount and sunk into it by another — and a flag cannot say it. So the flag is a type (D9):
//! one of these on the block, with the one place that reads it being this file. The same
//! shape serves a revolve, whose sides are angles and have no draft, as [`AngleSides`].
//! the extrude-revolve-options-spec design note carries the decisions.
//!
//! **Why `Two` is not two extrudes and a union.** Two solids meeting at a shared face are the
//! one join `min` cannot describe: it reads nought straight through the material, and
//! subtracting the pair leaves a sheet of surface in the middle of the cut
//! (`fieldform-field/examples/opposed_extrudes.rs`). Side one and side two are one closed form
//! with the profile's plane inside the material.

use crate::graph::ParamRef;

/// How an extrude's depth lies about its plane.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Sides {
    /// From the plane, the whole depth one way. What every extrude written before there was a
    /// choice is, so it is what a document that says nothing means.
    #[default]
    One,
    /// Half the depth each way, the profile in the middle. A draft then opens away from the
    /// plane on both sides, which is what a parting line means.
    Symmetric,
    /// The whole depth one way and this much the other, each with its own draft.
    Two {
        /// How far the solid runs out of the other side of the plane, in millimetres.
        height: ParamRef,
        /// The other side's draft, in degrees, measured from the start plane as side one's is.
        draft: ParamRef,
    },
}

impl Sides {
    /// Both ways or one, from the flag this type replaced — for a caller that has a `bool`.
    pub fn symmetric_if(both: bool) -> Self {
        if both { Sides::Symmetric } else { Sides::One }
    }

    /// Whether the solid lies on both sides of its plane.
    ///
    /// Nothing outside the tests here asks, so only the tests compile this (decision 1.4).
    #[cfg(test)]
    fn both(&self) -> bool {
        !matches!(self, Sides::One)
    }

    /// The numbers side two carries, so they take part in parameter resolution like any other.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let typed = match self {
            Sides::Two { height, draft } => Some([height, draft]),
            _ => None,
        };
        typed.into_iter().flatten()
    }

    /// The same, to be edited.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Sides::Two { height, draft } => vec![height, draft],
            _ => Vec::new(),
        }
    }
}

/// How a revolve's angle lies about the half-plane its profile is drawn in. A sibling of
/// [`Sides`] rather than the same type: side two of a revolve is an angle, and has no draft.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AngleSides {
    /// From the profile's half-plane, the whole angle one way.
    #[default]
    One,
    /// Half the angle each way.
    Symmetric,
    /// The whole angle one way and this much the other.
    Two {
        /// The other way's angle, in degrees.
        angle: ParamRef,
    },
}

impl AngleSides {
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let typed = match self {
            AngleSides::Two { angle } => Some(angle),
            _ => None,
        };
        typed.into_iter()
    }

    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            AngleSides::Two { angle } => vec![angle],
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document that says nothing means one side, which is what every extrude was; and only
    /// two sides carry numbers of their own, reachable by the route every other number is.
    #[test]
    fn one_side_is_the_default_and_only_two_sides_carry_numbers() {
        assert_eq!(Sides::default(), Sides::One);
        assert_eq!(Sides::symmetric_if(false), Sides::One);
        assert_eq!(Sides::symmetric_if(true), Sides::Symmetric);
        assert!(Sides::One.params().next().is_none() && Sides::Symmetric.params().next().is_none());
        let mut two = Sides::Two { height: ParamRef::Literal(5.0), draft: ParamRef::Literal(3.0) };
        assert_eq!(
            two.params().collect::<Vec<_>>(),
            vec![&ParamRef::Literal(5.0), &ParamRef::Literal(3.0)],
        );
        *two.params_mut()[0] = ParamRef::Literal(8.0);
        assert!(matches!(two, Sides::Two { height: ParamRef::Literal(h), .. } if h == 8.0));
        assert!(!Sides::One.both() && Sides::Symmetric.both() && two.both());

        assert_eq!(AngleSides::default(), AngleSides::One);
        let mut turned = AngleSides::Two { angle: ParamRef::Literal(90.0) };
        assert_eq!(turned.params().collect::<Vec<_>>(), vec![&ParamRef::Literal(90.0)]);
        *turned.params_mut()[0] = ParamRef::Literal(45.0);
        assert!(matches!(turned, AngleSides::Two { angle: ParamRef::Literal(a) } if a == 45.0));
    }
}
