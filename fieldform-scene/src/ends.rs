// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a block starts and where it stops, said once for every block defined that way.
//!
//! # Why this is a type
//!
//! A cylinder, a cone and a capsule are each two points and some radii. Before this they said
//! so in two different spellings — `base`/`top` on the first two, `start`/`end` on the third —
//! and neither could be handed to the other. That is the situation [`Frame`] was made for, one
//! step along: D9 asks that when two blocks need to be told the same *kind* of thing, they take
//! the same type rather than a similar one spelled differently.
//!
//! The test that rule states is whether two blocks mean the same thing, and these three plainly
//! do. A frame does not cover it: a frame is a place and a direction, deliberately without a
//! length, and what these blocks need is both ends. So this is a second brick, not a use of the
//! first.
//!
//! # Both ends can be taken from a curve
//!
//! [`Ends::From`] names a curve and takes its two ends. A line is the case that asks for it —
//! draw a line where the strut goes, point a cylinder at it, and the cylinder follows the line
//! when the line is dragged. That is the difference between copying two triples and sharing a
//! reference, and it is why several blocks can be built on *one* line and all move together.
//!
//! Any curve works, not only a straight one: a curve's ends are its ends. The shape between them
//! is ignored, which is worth saying out loud because a cylinder aimed at an arc is a straight
//! cylinder between that arc's endpoints and nothing about it will look like the arc. Sweeping
//! along a curve is [`Node::Pipe`], and that is the block for it.
//!
//! The edge this creates is wired generically, the way a frame's is: [`Node::ends`] and
//! [`Node::ends_mut`] are what [`Node::curves`] consults, so a block that gains an `Ends` gains
//! its curve reference without anyone remembering — which is the arrangement that stops the
//! next one being forgotten the way extrude and revolve were forgotten by `delete_curve`.
//!
//! [`Frame`]: crate::Frame
//! [`Node::Pipe`]: crate::graph::Node::Pipe
//! [`Node::ends`]: crate::graph::Node::ends
//! [`Node::ends_mut`]: crate::graph::Node::ends_mut
//! [`Node::curves`]: crate::graph::Node::curves

use serde::{Deserialize, Serialize};

use crate::graph::ParamRef;
use crate::id::{CurveId, NodeId};

/// Two points: where a block begins and where it ends.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ends {
    /// Typed, as two points.
    At {
        /// The first end. Which end that is belongs to the block — the base of a cone, the
        /// bottom of a cylinder — and the pair is ordered, so a cone's two radii know which is
        /// which.
        ///
        /// A [`crate::Place`], so either end can sit on a point or a plane rather than only on
        /// three typed numbers. Two bars ending at one point stay ending at one point when it
        /// moves, which is what typing the same coordinates twice cannot promise.
        from: crate::Place,
        /// The second.
        to: crate::Place,
    },
    /// Taken from a curve: its first point and its last.
    ///
    /// A curve whose ends coincide — a closed one — leaves the block with no length and no
    /// direction. That is not refused here: it is a state you pass through while dragging a
    /// point, and a modelling operation that becomes an error mid-gesture is worse than one
    /// that briefly has nothing to draw.
    From { curve: CurveId },
    /// Standing on a block that has a facing: from its origin, along its normal, this long.
    ///
    /// **Because a plane is the other obvious way to say which way something points.** Asked for
    /// directly: "if you create a cylinder — or any object that requires an axis — you should be
    /// able to provide as input under Axis a plane, and that object should re-orient itself along
    /// it as its new coordinate frame." A line says where an axis is by being drawn there; a
    /// plane says it by facing that way, and blocks are already aimed at planes everywhere else
    /// ([`crate::Frame`], [`crate::Turn`]).
    ///
    /// The length stays the block's own, because a plane has none to give. So moving or turning
    /// the plane moves and re-aims the block, and how long it is remains a number in its panel.
    On { node: NodeId, length: ParamRef },
}

impl Default for Ends {
    /// A unit length along `+Z`, which is a block you can see rather than one of zero size.
    fn default() -> Self {
        Self::At {
            from: crate::Place::at(0.0, 0.0, 0.0),
            to: crate::Place::at(0.0, 0.0, 1.0),
        }
    }
}

impl Ends {
    /// Two typed points.
    ///
    /// Offered so migrating a block that held two triples is a substitution rather than a
    /// translation.
    pub fn between(from: crate::Place, to: crate::Place) -> Self {
        Self::At { from, to }
    }

    /// The curve these ends are taken from, if they are taken from one.
    ///
    /// The whole of the reference an `Ends` creates. [`Node::curves`] goes through this, so
    /// nothing else has to know that ends can be a reference at all.
    ///
    /// [`Node::curves`]: crate::graph::Node::curves
    pub(crate) fn source(&self) -> Option<CurveId> {
        match self {
            Ends::From { curve } => Some(*curve),
            _ => None,
        }
    }

    /// The block these ends stand on, if they stand on one.
    ///
    /// The other half of the wiring, for the other kind of reference: `Node::inputs` and
    /// `Node::remap_inputs` read this, so a block that gains an `Ends` gains this edge along with
    /// the curve one and without anybody remembering (D9).
    pub fn node(&self) -> Option<NodeId> {
        match self {
            Ends::On { node, .. } => Some(*node),
            _ => None,
        }
    }

    /// Points these ends at a different block, if they stand on one. For a paste.
    pub(crate) fn remap(&mut self, from: NodeId, to: NodeId) {
        if let Ends::On { node, .. } = self
            && *node == from
        {
            *node = to;
        }
    }

    /// [`Self::source`], to be redirected when curves are renumbered — as a paste does, which
    /// gives a copied subtree fresh ids for the curves it brought with it.
    pub(crate) fn source_mut(&mut self) -> Option<&mut CurveId> {
        match self {
            Ends::From { curve } => Some(curve),
            _ => None,
        }
    }

    /// The numbers these ends carry, so they take part in parameter resolution like any other.
    /// Ends taken from a curve carry none of their own.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let (from, to, length) = match self {
            Ends::At { from, to } => (Some(from), Some(to), None),
            // The length is this block's own number, so it is named and driven like any other.
            Ends::On { length, .. } => (None, None, Some(length)),
            Ends::From { .. } => (None, None, None),
        };
        from.into_iter().chain(to).flat_map(crate::Place::refs).chain(length)
    }

    /// The same, to be edited.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Ends::At { from, to } => {
                let mut out = from.refs_mut();
                out.extend(to.refs_mut());
                out
            }
            Ends::On { length, .. } => vec![length],
            Ends::From { .. } => Vec::new(),
        }
    }

    /// The places these ends are, so a block holding them carries their graph edges.
    ///
    /// The same arrangement a frame and a place already have (D9): gathered in one place so a
    /// node that holds `Ends` gains the edge, the remapping and the cycle check without anybody
    /// adding it to three more functions.
    pub(crate) fn places(&self) -> impl Iterator<Item = &crate::Place> {
        let both = match self {
            Ends::At { from, to } => Some([from, to]),
            _ => None,
        };
        both.into_iter().flatten()
    }

    /// The same, to edit.
    pub(crate) fn places_mut(&mut self) -> Vec<&mut crate::Place> {
        match self {
            Ends::At { from, to } => vec![from, to],
            _ => Vec::new(),
        }
    }
}
