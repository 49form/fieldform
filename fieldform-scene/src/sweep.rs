// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which way a profile is swept, and whether that sweep is exact.
//!
//! # Why this is not a [`Frame`]
//!
//! An extrusion needs two things that look like one and are not: the **plane** its profile lies
//! in, and the **direction** it is swept along. They were one `Frame`, and every bug that came
//! of it was the same bug — setting the direction also moved the plane, so the profile stopped
//! being flat in the plane it was read in and the block stopped compiling. Aiming an extrusion
//! along a line could not be made to work, because aiming it was what broke it.
//!
//! The plane is not a choice. It is a fact about the profile, computed from the profile's own
//! points, and it is not stored anywhere — so it cannot be wrong and cannot go stale. What is
//! left is a direction, and a direction is not a frame: it has no origin, and a `Frame` carrying
//! one that nothing reads would be state that exists to be misleading.
//!
//! # Exactness is a variant, not a coincidence
//!
//! [`Sweep::Normal`] is its own case rather than a direction that happens to equal the profile's
//! normal. That is deliberate. Sweeping along the normal gives an **exact** distance field;
//! sweeping along anything else is a shear, and a sheared field reads short. Making the exact
//! case a variant means it is exact by construction rather than by a comparison against a
//! tolerance — and it means a profile whose plane later moves stays exact, instead of silently
//! becoming approximate because a number no longer matches.

use serde::{Deserialize, Serialize};

use crate::graph::ParamRef;
use crate::id::CurveId;

/// How a profile is held as it is carried along a curve.
///
/// **Three, because a sweep is two questions and the answer to the second is a choice.** Where
/// the solid starts is settled — at the profile, wherever the curve was drawn
/// (`fieldform_field::sweep_offset`). How the section is *held* on the way is not: holding it
/// square to the path is what a pipe does and what most sweeps want, and holding it parallel to
/// itself is what a draughtsman means by sweeping a shape along a line. Asked for by the owner,
/// who wanted the start flush with the profile's plane in both.
///
/// The names are what the file carries, so they are chosen to still make sense to a reader that
/// has never seen this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    /// Perpendicular to the path everywhere, turning with it — the original behaviour and the
    /// default, so every document written before this reads exactly as it did.
    #[default]
    Square,
    /// Starts lying in the profile's own plane, at whatever angle the path meets it, and carries
    /// that same tilt relative to the path the whole way. The start face is flush with the
    /// profile; the section still turns with the curve.
    Skewed,
    /// Never turns: the section stays parallel to the profile's own plane the whole way, carried
    /// along the path's shape. What some call a parallel or translational sweep.
    Parallel,
}

impl Section {
    /// What the block calls it.
    pub fn label(self) -> &'static str {
        match self {
            Section::Square => "Square to the path",
            Section::Skewed => "Flush start",
            Section::Parallel => "Parallel",
        }
    }
}

/// The direction a profile is swept along.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sweep {
    /// Straight out of the profile's own plane. **Exact**, and the default.
    Normal,
    /// Along a curve, end to end. The curve's position is irrelevant — only which way it runs —
    /// so it need not touch the profile or be anywhere near it.
    Along { curve: CurveId },
    /// A typed direction. Need not be unit; it is normalised where it is used.
    Toward { xyz: [ParamRef; 3] },
}

impl Default for Sweep {
    fn default() -> Self {
        Self::Normal
    }
}

impl Sweep {
    /// The curve this sweep reads, if it reads one.
    ///
    /// The whole of the graph edge a sweep creates. `Node::curves` goes through this, so a block
    /// that gains a sweep gains its curve reference without anybody remembering — the same
    /// arrangement frames and ends already have.
    pub fn source(&self) -> Option<CurveId> {
        match self {
            Sweep::Along { curve } => Some(*curve),
            _ => None,
        }
    }

    /// The same, to be redirected when curves are renumbered.
    pub(crate) fn source_mut(&mut self) -> Option<&mut CurveId> {
        match self {
            Sweep::Along { curve } => Some(curve),
            _ => None,
        }
    }

    /// The numbers it carries, so they take part in parameter resolution like any other.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let typed: &[ParamRef] = match self {
            Sweep::Toward { xyz } => xyz,
            _ => &[],
        };
        typed.iter()
    }

    /// The same, to be edited.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Sweep::Toward { xyz } => xyz.iter_mut().collect(),
            _ => Vec::new(),
        }
    }

    /// Whether this sweep gives an exact distance field.
    ///
    /// Only the normal does. Everything else is a shear, which reads short and must say so
    /// where the block is edited — the rule D8's reversal left standing: an approximation
    /// announces itself.
    pub fn is_exact(&self) -> bool {
        matches!(self, Sweep::Normal)
    }
}
