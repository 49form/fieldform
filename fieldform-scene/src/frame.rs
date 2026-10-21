// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a block sits and which way it points, said once for every block that needs it.
//!
//! # Why this is a type
//!
//! Half the operations in a modeller have to be told a place and a direction, and before this
//! there were four ways to say it: an [`Axis`] enum with a separate origin (a twist, a radial
//! pattern), a `MirrorPlane` that was either one of those or a reference to a plane node, a
//! frame of two triples on a lattice, and [`Node::Plane`] itself. Four spellings of one idea,
//! none of them interchangeable — so a direction set up for a twist could not be handed to a
//! mirror, and an extrude and a revolve would have made six.
//!
//! One type instead. Any block that needs aiming takes a [`Frame`], all of them mean the same
//! thing by it, and one set up for a block can be copied to another or pointed at the same
//! source.
//!
//! # A frame can be taken from another block
//!
//! [`Frame::From`] names a node that supplies both the point and the direction — a
//! [`Node::Plane`] gives its origin and its normal. That is what makes these interchangeable
//! rather than merely identical: several blocks can be aimed at *one* plane, and moving that
//! plane moves all of them, which is the difference between copying numbers and sharing a
//! reference.
//!
//! The edge that creates is wired generically. [`Node::frames`] and [`Node::frames_mut`] are
//! what `inputs` and `remap_inputs` consult, so a node that gains a frame gains its graph
//! wiring without anybody remembering to add it — the same arrangement parameter edges already
//! have, and for the same reason: the alternative is a mistake made once per node kind.
//!
//! # What it deliberately is not
//!
//! **A full orientation.** A frame says which way its own `+Z` points and leaves the roll about
//! that axis free. That is everything a revolve, a twist, a cylindrical lattice or a plane
//! needs, because each is symmetric about its own axis. An operation that genuinely needs the
//! roll — a rectangular lattice squared up to something, a profile's orientation within its own
//! plane — needs a second direction and should say so, rather than have one invented for it.
//!
//! [`Axis`]: crate::graph::Axis
//! [`Node::Plane`]: crate::graph::Node::Plane
//! [`Node::frames`]: crate::graph::Node::frames

use serde::{Deserialize, Serialize};

use crate::graph::{Axis, ParamRef};
use crate::id::{CurveId, NodeId};

/// A place and a direction, shared by every block that needs aiming.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Frame {
    /// Typed: a point, and a direction through it.
    At {
        /// Where it sits. What that means is the block's business — the centre a spherical
        /// lattice radiates from, a point the axis of a revolve passes through, the base of an
        /// extrusion.
        origin: [ParamRef; 3],
        /// Which way it points. Need not be unit: it is normalised where it is used, so a
        /// vector read off a drawing works without being tidied first.
        ///
        /// A zero vector means `+Z` rather than a division by zero. It cannot point anywhere,
        /// and refusing it would make an intermediate state during typing into an error.
        axis: [ParamRef; 3],
    },
    /// Taken from another block, which supplies both the point and the direction.
    ///
    /// Consumed like any other operand, so a plane being used as a frame stops being a body of
    /// its own — the same rule a boolean's operands follow.
    From { node: NodeId },
    /// Taken from a curve: it starts where the curve starts and points the way the curve runs.
    ///
    /// **A line is the obvious way to say where an axis is**, and it was always meant to be one
    /// of the answers here — `resolve_frame` has carried a note since frames were introduced
    /// saying a curve could answer and that the place to add it was there, once. This is that.
    /// Draw the line a pattern turns about or a twist winds along, and every block aimed at it
    /// moves when the line moves.
    ///
    /// Start to *end*, not the tangent at the start: it is the direction the line as a whole
    /// runs, which is what somebody drawing a line to mean an axis means by it. A curve whose
    /// ends coincide has no direction, and falls back to `+Z` the way a zero vector does.
    Along { curve: CurveId },
    /// Taken from two blocks: it sits at the first and points at the second.
    ///
    /// **The way somebody says a direction when there is no line to draw.** Asked for twice, in
    /// the same words both times: "I should be able to do that with a vector or a straight curve
    /// by supplying either a curve or two points" (a lattice's axis), and "a vector … read from
    /// geometry — a line, or two points" (a plane's normal). Backlog items 25 and 18, both of
    /// which parked on settling what such a reference looks like *once* rather than growing a
    /// narrower copy per block. This is that once: every block aimed by a [`Frame`] gains it
    /// together, and [`crate::Turn`] reads it through the same resolver.
    ///
    /// Two [`crate::graph::Node::Point`]s are the ordinary case, but anything that can answer a
    /// place will do — the resolver asks each end for its position and nothing more. That is why
    /// this is cheaper than it looks: a point is already "a position, as a thing to reference",
    /// so no new kind of reference had to be invented, only a second one of the kind that exists.
    ///
    /// Two ends in the same place have no direction between them, and fall back to `+Z` the way
    /// a zero vector and a closed curve already do. It is a state you pass through while
    /// dragging one of them.
    Between { from: NodeId, to: NodeId },
    /// Taken from where a curve *starts*: at its first point, facing the way it sets off.
    ///
    /// **The tangent, not the chord — which is the whole difference from [`Frame::Along`].**
    /// Those two agree for a straight line and part company for anything else, and each is right
    /// for a different question. "Where is this axis?" is answered by the line as a whole, which
    /// is what somebody drawing a line to mean an axis means. "Which way is this curve going
    /// *here*?" is answered by the tangent, and nothing else will do: a plane meant to stand
    /// square across a pipe has to be square across it where it stands, not square across the
    /// straight line between the pipe's two ends.
    ///
    /// Reported of exactly that. A pipe swept along a degree-five curve, its radius driven by a
    /// ramp reading a plane at the curve's start: the plane came out tilted by an angle nobody
    /// had typed, because it was facing along the chord. "It should just take the normal of
    /// whatever the curve is doing."
    ///
    /// A curve with no length has no direction to read, and falls back to `+Z` the way a zero
    /// vector and a closed curve already do.
    Tangent { curve: CurveId },
}

impl Default for Frame {
    /// The world's own frame: at the origin, pointing up.
    fn default() -> Self {
        Self::At {
            origin: [ParamRef::Literal(0.0); 3],
            axis: [
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(1.0),
            ],
        }
    }
}

impl Frame {
    /// A frame at the origin pointing along one of the world's axes.
    ///
    /// The three answers most frames have. Offered as a constructor because a panel wants to
    /// write one from a button rather than have somebody type a one and two zeros.
    pub fn along(axis: Axis) -> Self {
        let v = match axis {
            Axis::X => [1.0, 0.0, 0.0],
            Axis::Y => [0.0, 1.0, 0.0],
            Axis::Z => [0.0, 0.0, 1.0],
        };
        Self::At {
            origin: [ParamRef::Literal(0.0); 3],
            axis: v.map(ParamRef::Literal),
        }
    }

    /// A frame at a given point, pointing along one of the world's axes.
    ///
    /// What every block that used to hold an `Axis` and an origin was saying. Offered so that
    /// migrating one is a substitution rather than a translation, and so that the three answers
    /// people give most often stay one call.
    pub fn at_along(axis: Axis, origin: [ParamRef; 3]) -> Self {
        let Self::At { axis, .. } = Self::along(axis) else {
            unreachable!("`along` builds an explicit frame")
        };
        Self::At { origin, axis }
    }

    /// The node this frame is taken from, if it is taken from one.
    ///
    /// **Only [`Frame::From`]'s one block — not the whole of the edges a frame makes.** A
    /// [`Frame::Between`] reads two blocks and answers `None` here, so a question about every
    /// block a frame reads asks [`Frame::nodes`]. The delete used to ask this one, and so left a
    /// frame between two points holding the one it deleted.
    pub fn source(&self) -> Option<NodeId> {
        match self {
            Frame::From { node } => Some(*node),
            _ => None,
        }
    }

    /// Every block this frame reads, which is one for [`Frame::From`] and two for
    /// [`Frame::Between`].
    ///
    /// **What the graph edges come from.** `source` answers "the block this is taken from",
    /// which was the whole story while a frame could name at most one; a frame between two points
    /// names two, and both have to be evaluated first, neither may be a body of its own, and a
    /// cycle through either is still a cycle. `Node::for_each_reference` reads this, and so does
    /// the application's delete, which lets go of a frame reading the block it deletes.
    ///
    /// Walked rather than listed, since `Node::for_each_input` asks it of every node on every edit.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> {
        let read = match self {
            Frame::From { node } => [Some(*node), None],
            Frame::Between { from, to } => [Some(*from), Some(*to)],
            _ => [None, None],
        };
        read.into_iter().flatten()
    }

    /// The same, to be redirected when the graph is renumbered — as a paste does.
    pub(crate) fn nodes_mut(&mut self) -> Vec<&mut NodeId> {
        match self {
            Frame::From { node } => vec![node],
            Frame::Between { from, to } => vec![from, to],
            _ => Vec::new(),
        }
    }

    /// The curve this frame is taken from, if it is taken from one.
    ///
    /// The counterpart of [`Frame::source`] for the other kind of reference. `Node::curves` goes
    /// through this, so a block that gains a frame gains its curve edge along with everything
    /// else a frame brings.
    pub fn curve_source(&self) -> Option<CurveId> {
        match self {
            Frame::Along { curve } | Frame::Tangent { curve } => Some(*curve),
            _ => None,
        }
    }

    /// The same, to be redirected when curves are renumbered.
    pub(crate) fn curve_source_mut(&mut self) -> Option<&mut CurveId> {
        match self {
            Frame::Along { curve } | Frame::Tangent { curve } => Some(curve),
            _ => None,
        }
    }

    /// The numbers this frame carries, so they take part in parameter resolution like any
    /// other. A frame taken from a node carries none of its own.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let (origin, axis): (&[ParamRef], &[ParamRef]) = match self {
            Frame::At { origin, axis } => (origin, axis),
            _ => (&[], &[]),
        };
        origin.iter().chain(axis)
    }

    /// The same, to be edited.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Frame::At { origin, axis } => origin.iter_mut().chain(axis.iter_mut()).collect(),
            _ => Vec::new(),
        }
    }

    /// Whether this is the world frame.
    ///
    /// Used to keep it out of the file when it says nothing: a field written only when it
    /// carries information is one an older reader never has to have heard of.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}
