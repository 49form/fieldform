// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which way a block is turned: typed, or taken from something that has a direction.
//!
//! # Why this is a type
//!
//! It was three numbers — `pub type Turn = [ParamRef; 3]` — and three numbers cannot be pointed
//! at anything. So a block could be turned only by typing angles, and the three ways of saying
//! "this way up" that [`crate::Frame`] already offers every *aimed* block were unavailable to
//! the blocks that carry their own facing: a cuboid, a torus, a plane.
//!
//! Two requests met here, and they are the same request:
//!
//! - "treat the turn the same way we treat the `at` coordinate — the whole XYZ matrix can become
//!   a V, instead of each X Y or Z being treated separately". A place has one V because a place
//!   is one thing; a facing is one thing too.
//! - "all SDF primitives should be able to be aligned along an axis — so far cone, cylinder and
//!   capsule; I want to do it with cube, sphere and torus … long term with a plane also, not
//!   just a curve". A cylinder can be aimed because [`crate::Ends`] can name a curve. This is
//!   the same ability for the blocks defined about a centre rather than between two points.
//!
//! # The three answers, and why they are these three
//!
//! [`Turn::By`] is what a turn has always been, and is still what a file holds: intrinsic XYZ
//! Euler angles in degrees, the same triple a [`crate::graph::Node::Transform`] stores. It is
//! the only one of the three that can say a *roll*, which is why it is the manual answer rather
//! than a typed axis: an axis fixes two of the three degrees of freedom and a box cares about
//! the third.
//!
//! [`Turn::From`] takes the facing from a block that has one — a plane gives its normal, a point
//! leaves it upright — and [`Turn::Along`] takes it from a curve, start to end. Both are
//! references, so moving the plane or dragging the line re-aims everything built on it. That is
//! the difference between this and a button that writes three numbers once.
//!
//! Both leave the roll at whatever `+Z` to that direction gives, which is the same thing
//! [`crate::Frame`] promises and for the same reason: a direction does not carry a roll, and
//! inventing one would be inventing it differently in each block that asked.
//!
//! # The edges are carried here
//!
//! [`Turn::nodes`] and [`Turn::curve`] are what [`crate::graph::Node::inputs`] and
//! `Node::curves` consult, so a block that gains a turn gains its graph wiring without anybody
//! remembering — the arrangement a frame, a place and a parameter already have, and the third of
//! D9's requirements.
//!
//! # The file did not have to change
//!
//! A typed turn serialises as the bare three-number array it always was, so every file already
//! written parses into [`Turn::By`] with no migration and no version bump. The two references are
//! objects, a shape those files never held. The reader tells them apart by looking, which is why
//! there is a hand-written `Deserialize` here rather than a derive — the same arrangement
//! [`crate::Place`] uses, and for the same reason.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::graph::ParamRef;
use crate::id::{CurveId, NodeId};

/// Which way a block is turned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Turn {
    /// Typed: intrinsic XYZ Euler angles, in degrees.
    By([ParamRef; 3]),
    /// Taken from a block that has a facing — a plane's normal, or a point's `+Z`.
    ///
    /// Consumed like any other operand, so a plane used as a facing stops being a body of its
    /// own. That is the rule a boolean's operands, a frame's source and a place's already follow.
    From { node: NodeId },
    /// Taken from a curve: `+Z` is turned onto the way the curve runs, start to end.
    ///
    /// The shape between the ends is ignored, exactly as it is for [`crate::Ends`]: what is being
    /// read is a direction, and a curve has one whether or not it is straight. A curve whose ends
    /// coincide has none, and leaves the block unturned rather than becoming an error mid-drag.
    Along { curve: CurveId },
    /// Taken from two blocks: `+Z` is turned onto the direction from the first to the second.
    ///
    /// **What "perpendicular to this edge" means when there is no edge to draw.** The plane
    /// half of the same request [`crate::Frame::Between`] answers for the aimed blocks — a
    /// normal "read from geometry: a line, or two points" — and read through that same arm, so
    /// there is one description of what two points mean and not a second, narrower one here.
    Between { from: NodeId, to: NodeId },
    /// A facing whose rotation has **already been written into a placement above the block**,
    /// leaving only the references it was reading.
    ///
    /// **Produced by [`crate::Graph::with_turns_expanded_using`] and by nothing else.** That
    /// rewrite is what makes "a turned block *is* that block inside a placement" true, and it
    /// has two jobs that pull against each other: the rotation has to move to the placement, so
    /// that one walk answers where everything is; and the *edge* has to stay, because the plane
    /// a block is aimed at is consumed by it and a block nothing consumes is a body — a plane
    /// that became a body again is a half-space drawn across the whole viewport.
    ///
    /// Leaving the whole turn behind did both jobs and one of them twice: `resolve_frame`'s
    /// plane arm reads a turn, so a plane aimed at another was turned once for the picture and
    /// twice for anything aimed at it — 40 degrees drawn against 80 degrees mirrored, measured
    /// by `aimed_turn_probe`. This is the half that was actually wanted: the references, with
    /// no rotation of their own.
    ///
    /// It exists only inside a lowered graph, which is never saved — a document is written from
    /// the graph the user built. Serializing one is refused rather than guessed at, so if that
    /// ever stops being true it is heard about rather than written to a file.
    Absorbed {
        /// The blocks the turn read: `From`'s one, `Between`'s two, neither for the rest.
        nodes: [Option<NodeId>; 2],
        /// The curve it read, for `Along` and `Tangent`.
        curve: Option<CurveId>,
    },
    /// Taken from where a curve starts: `+Z` is turned onto the way it sets off.
    ///
    /// The tangent rather than the chord — see [`crate::Frame::Tangent`], which this reads
    /// through, so "which way is the curve going here" has one answer whether a block is aimed
    /// by a frame or carries its own facing.
    Tangent { curve: CurveId },
}

impl Default for Turn {
    fn default() -> Self {
        Turn::By([ParamRef::Literal(0.0); 3])
    }
}

impl Turn {
    /// The angles, when it is typed.
    pub fn angles(&self) -> Option<&[ParamRef; 3]> {
        match self {
            Turn::By(xyz) => Some(xyz),
            _ => None,
        }
    }

    /// The block this facing is taken from, if it is taken from one.
    ///
    /// **One block, not the whole of the edges a turn makes**: a [`Turn::Between`] reads two and
    /// answers `None` here, so a question about every block a turn reads asks [`Turn::nodes`] —
    /// the mistake the delete made, which left a facing between two points holding a deleted one.
    pub fn source(&self) -> Option<NodeId> {
        match self {
            Turn::From { node } => Some(*node),
            Turn::Absorbed { nodes, .. } => nodes[0],
            _ => None,
        }
    }

    /// Whether this facing's rotation has already moved to a placement. See [`Turn::Absorbed`].
    pub(crate) fn is_absorbed(&self) -> bool {
        matches!(self, Turn::Absorbed { .. })
    }

    /// The same references, with the rotation taken out — what a block is left holding once its
    /// turn has been written into the placement above it.
    ///
    /// A typed turn becomes no turn at all, because it read nothing: there is no reference to
    /// keep, and leaving an empty `Absorbed` behind would make a block that plainly has no
    /// facing answer "yes, but spent" to everything that asks. A curve-borne one keeps its
    /// curve, so a curve drawn to be an axis still reads as being used for one.
    pub(crate) fn absorbed(&self) -> Turn {
        match self {
            Turn::By(_) => crate::graph::no_turn(),
            Turn::From { node } => {
                Turn::Absorbed { nodes: [Some(*node), None], curve: None }
            }
            Turn::Between { from, to } => {
                Turn::Absorbed { nodes: [Some(*from), Some(*to)], curve: None }
            }
            Turn::Along { curve } | Turn::Tangent { curve } => {
                Turn::Absorbed { nodes: [None, None], curve: Some(*curve) }
            }
            Turn::Absorbed { .. } => *self,
        }
    }

    /// Every block this turn reads: one for [`Turn::From`], two for [`Turn::Between`].
    ///
    /// What the graph edges come from — `Node::for_each_reference` reads this, so both ends of a
    /// facing between two points are evaluated first, neither is a body of its own, and a cycle
    /// through either is still a cycle. The application's delete reads it too, to let go of a
    /// facing that reads the block it deletes.
    ///
    /// Walked rather than listed, since `Node::for_each_input` asks it of every node on every edit.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> {
        let read = match self {
            Turn::From { node } => [Some(*node), None],
            Turn::Between { from, to } => [Some(*from), Some(*to)],
            Turn::Absorbed { nodes, .. } => *nodes,
            _ => [None, None],
        };
        read.into_iter().flatten()
    }

    /// The same, to be redirected when the graph is renumbered — as a paste does.
    pub(crate) fn nodes_mut(&mut self) -> Vec<&mut NodeId> {
        match self {
            Turn::From { node } => vec![node],
            Turn::Between { from, to } => vec![from, to],
            Turn::Absorbed { nodes, .. } => nodes.iter_mut().flatten().collect(),
            _ => Vec::new(),
        }
    }

    /// The curve this facing is taken from, if it is taken from one.
    pub(crate) fn curve(&self) -> Option<CurveId> {
        match self {
            Turn::Along { curve } | Turn::Tangent { curve } => Some(*curve),
            Turn::Absorbed { curve, .. } => *curve,
            _ => None,
        }
    }

    /// The same, to be redirected when curves are renumbered.
    pub(crate) fn curve_mut(&mut self) -> Option<&mut CurveId> {
        match self {
            Turn::Along { curve } | Turn::Tangent { curve } => Some(curve),
            Turn::Absorbed { curve, .. } => curve.as_mut(),
            _ => None,
        }
    }

    /// The numbers a typed turn is made of, so each can still be named and driven.
    ///
    /// Empty for a taken one: there is no number to name, because the answer is elsewhere.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let typed: &[ParamRef] = match self {
            Turn::By(xyz) => xyz,
            _ => &[],
        };
        typed.iter()
    }

    /// The same, to edit.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Turn::By(xyz) => xyz.iter_mut().collect(),
            _ => Vec::new(),
        }
    }
}

impl Serialize for Turn {
    /// A typed turn is the bare array it has always been; a taken one names what it reads.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            Turn::By(xyz) => xyz.serialize(serializer),
            Turn::From { node } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("from", node)?;
                map.end()
            }
            Turn::Along { curve } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("along", curve)?;
                map.end()
            }
            Turn::Between { from, to } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("from", from)?;
                map.serialize_entry("to", to)?;
                map.end()
            }
            Turn::Tangent { curve } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("tangent", curve)?;
                map.end()
            }
            // **Refused rather than guessed at**, which is the same rule the reader below
            // follows. This state belongs to a lowered graph and a document is written from the
            // graph the user built, so reaching here means a lowered graph has been handed to a
            // save — and a file that quietly recorded the wrong facing, or none, would be worse
            // than a save that stops and says so.
            Turn::Absorbed { .. } => Err(serde::ser::Error::custom(
                "a facing whose rotation has already moved to a placement belongs to a lowered \
                 graph and cannot be saved; save the document's own graph",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for Turn {
    /// An array is typed; an object names exactly one of `from` and `along`.
    ///
    /// Both or neither is refused rather than one being preferred: a turn that says it reads two
    /// different things was written by something this build does not understand, and D3's rule is
    /// that such a document is refused rather than guessed at.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        if value.is_array() {
            let xyz: [ParamRef; 3] = serde_json::from_value(value).map_err(D::Error::custom)?;
            return Ok(Turn::By(xyz));
        }
        let object = value.as_object().ok_or_else(|| {
            D::Error::custom("a turn is three angles, or a block or curve to take it from")
        })?;
        // A `to` beside a `from` is a facing between two blocks; a `from` alone is one taken
        // from a single block. The two are told apart by which keys are there, not by a tag.
        if let Some(curve) = object.get("tangent") {
            return Ok(Turn::Tangent {
                curve: serde_json::from_value(curve.clone()).map_err(D::Error::custom)?,
            });
        }
        match (object.get("from"), object.get("to"), object.get("along")) {
            (Some(from), Some(to), None) => Ok(Turn::Between {
                from: serde_json::from_value(from.clone()).map_err(D::Error::custom)?,
                to: serde_json::from_value(to.clone()).map_err(D::Error::custom)?,
            }),
            (Some(node), None, None) => Ok(Turn::From {
                node: serde_json::from_value(node.clone()).map_err(D::Error::custom)?,
            }),
            (None, None, Some(curve)) => Ok(Turn::Along {
                curve: serde_json::from_value(curve.clone()).map_err(D::Error::custom)?,
            }),
            (_, _, Some(_)) => Err(D::Error::custom(
                "a turn reads one thing: an \"along\" cannot sit beside a \"from\" or a \"to\"",
            )),
            (None, Some(_), None) => Err(D::Error::custom(
                "a turn between two blocks needs a \"from\" as well as a \"to\"",
            )),
            (None, None, None) => Err(D::Error::custom(
                "a turn taken from something needs a \"from\" or an \"along\"",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A typed turn is still the array it always was**, which is what makes this change cost
    /// nothing to every file already written.
    #[test]
    fn a_typed_turn_reads_and_writes_as_a_bare_array() {
        let turn = Turn::By([
            ParamRef::Literal(0.0),
            ParamRef::Literal(90.0),
            ParamRef::Literal(-45.0),
        ]);
        let text = serde_json::to_string(&turn).expect("serialises");
        assert!(text.starts_with('['), "a typed turn must still be three numbers, got {text}");
        assert_eq!(serde_json::from_str::<Turn>(&text).expect("reads back"), turn);

        // Asserted against what the array itself writes rather than against a literal, so it
        // stays true if a parameter's own spelling ever changes.
        let old = serde_json::to_string(&[
            ParamRef::Literal(0.0),
            ParamRef::Literal(90.0),
            ParamRef::Literal(-45.0),
        ])
        .expect("the old shape serialises");
        assert_eq!(text, old, "a typed turn writes exactly what the array wrote");
        assert_eq!(serde_json::from_str::<Turn>(&old).expect("an old file reads"), turn);
    }

    /// Each reference round-trips, and reports exactly its own edge.
    #[test]
    fn a_taken_turn_round_trips_and_names_what_it_reads() {
        let node = NodeId::new();
        let from = Turn::From { node };
        let text = serde_json::to_string(&from).expect("serialises");
        assert_eq!(serde_json::from_str::<Turn>(&text).expect("reads back"), from);
        assert_eq!(from.source(), Some(node));
        assert_eq!(from.curve(), None);

        let curve = CurveId::new();
        let along = Turn::Along { curve };
        let text = serde_json::to_string(&along).expect("serialises");
        assert_eq!(serde_json::from_str::<Turn>(&text).expect("reads back"), along);
        assert_eq!(along.curve(), Some(curve));
        assert_eq!(along.source(), None);
    }

    /// A turn that names two sources, or none, is refused rather than half-read.
    #[test]
    fn a_turn_that_reads_two_things_or_nothing_is_refused() {
        assert!(serde_json::from_str::<Turn>("7").is_err());
        assert!(serde_json::from_str::<Turn>("{}").is_err(), "an object needs a source");
        let both = format!(
            "{{\"from\":{},\"along\":{}}}",
            serde_json::to_string(&NodeId::new()).expect("a node id"),
            serde_json::to_string(&CurveId::new()).expect("a curve id"),
        );
        assert!(serde_json::from_str::<Turn>(&both).is_err(), "one source, not two");
    }
}
