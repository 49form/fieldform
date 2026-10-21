// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where something sits: typed, or taken from a block that has a position.
//!
//! # Why this is not a [`crate::Frame`]
//!
//! A frame is a place *and a direction*, and D9's test is whether two blocks mean the same
//! thing rather than whether they happen to hold three numbers. A sphere's centre, a plane's
//! origin and a torus's centre all mean one thing — a place — and none of them has a facing.
//! Giving them frames would store an axis nothing reads and put a control in three panels that
//! does nothing, which is a shared type in name and a lie in the file.
//!
//! So this is the smaller idea, and a frame's own origin is an instance of it. The two are
//! deliberately separate types answering separate questions: "where" and "where, and which way".
//!
//! # What it buys
//!
//! [`Place::From`] is the whole point. A point block can be *the* place several bodies sit at,
//! so moving the point moves all of them — which is what copying three numbers into three
//! panels cannot do, and what the owner asked for: "add in a point and place a plane's origin
//! on it".
//!
//! # The file format did not have to change
//!
//! A typed place serialises as the bare three-number array it always was, so every file already
//! written parses into `Place::At` with no migration step and no version bump. A taken one is an
//! object, which is a shape those files never held. The reader tells them apart by looking,
//! which is why there is a hand-written `Deserialize` here rather than a derive.

use crate::{CurveId, NodeId, ParamRef};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Where something sits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Place {
    /// Typed: three numbers, each of which can itself be a named parameter.
    At([ParamRef; 3]),
    /// Taken from another block that has a position — a point, or a plane.
    ///
    /// Consumed like any other operand, so a point used as a place stops being a body of its
    /// own. That is the same rule a boolean's operands and a frame's source already follow.
    From { node: NodeId },
    /// Where a curve begins: its first point, wherever that has been dragged to.
    ///
    /// **Because a plane standing at the start of a pipe has to stand there when the pipe
    /// moves.** Reported of exactly that: a ramp driving a pipe's radius reads a plane at the
    /// curve's start, and the plane's *facing* followed the curve — [`crate::Turn::Tangent`] —
    /// while its position did not, because a position could name a block and not a curve. "The
    /// plane should be tied to the point of the curve, so when we move the point of the curve
    /// the plane also moves with it."
    ///
    /// The start rather than either end, because that is what has been asked for and what
    /// `Frame::Tangent` reads at. The other end is a second variant if anything ever wants it;
    /// one is not worth two until then.
    ///
    /// This is why resolving a place takes the document: a curve lives there rather than in the
    /// graph, exactly as [`crate::Ends::From`] and [`crate::Frame::Along`] already do.
    StartOf { curve: CurveId },
}

impl Default for Place {
    fn default() -> Self {
        Place::At([ParamRef::Literal(0.0); 3])
    }
}

impl Place {
    /// The world origin, typed.
    ///
    /// Named rather than written out at each site: it is the default pivot, the value a file
    /// leaves out, and the value the writer compares against to decide to leave it out — three
    /// places that have to agree about what "nowhere in particular" is.
    pub fn origin() -> Self {
        Place::At([ParamRef::Literal(0.0); 3])
    }

    /// A place at three literal millimetre coordinates.
    pub fn at(x: f64, y: f64, z: f64) -> Self {
        Place::At([ParamRef::Literal(x), ParamRef::Literal(y), ParamRef::Literal(z)])
    }

    /// The block this place is taken from, if it is taken from one.
    ///
    /// `Node::inputs` reads this, so a node that gains a place gains its graph edge without
    /// anybody remembering — the arrangement `frames` and `param_refs` already have, and the
    /// reason D9 insists a shared type carries its own wiring.
    pub fn source(&self) -> Option<NodeId> {
        match self {
            Place::From { node } => Some(*node),
            _ => None,
        }
    }

    /// The curve this place is read from, if it is read from one.
    ///
    /// What `Node::curves` consults, so a block whose position comes from a curve keeps that
    /// curve alive and follows it when it is renumbered — the wiring the type carries itself.
    pub(crate) fn curve(&self) -> Option<CurveId> {
        match self {
            Place::StartOf { curve } => Some(*curve),
            _ => None,
        }
    }

    /// The same, to be redirected when curves are renumbered — as a paste does.
    pub(crate) fn curve_mut(&mut self) -> Option<&mut CurveId> {
        match self {
            Place::StartOf { curve } => Some(curve),
            _ => None,
        }
    }

    /// Points this place at a different block, if it was taken from one.
    pub(crate) fn remap(&mut self, from: NodeId, to: NodeId) {
        if let Place::From { node } = self
            && *node == from
        {
            *node = to;
        }
    }

    /// The numbers a typed place is made of, so they can be named and driven like any others.
    ///
    /// Empty for a taken one: there is no number to name, because the answer is another block's.
    ///
    /// Walked rather than listed, as every list `Node::for_each_input` reads is: it asks this of
    /// every node on every edit, and a list was an allocation each time.
    pub(crate) fn refs(&self) -> impl Iterator<Item = &ParamRef> {
        let typed: &[ParamRef] = match self {
            Place::At(xyz) => xyz,
            _ => &[],
        };
        typed.iter()
    }

    /// The same, to edit.
    pub(crate) fn refs_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Place::At(xyz) => xyz.iter_mut().collect(),
            _ => Vec::new(),
        }
    }

    /// The three numbers, when it is typed.
    pub fn typed(&self) -> Option<[ParamRef; 3]> {
        match self {
            Place::At(xyz) => Some(*xyz),
            _ => None,
        }
    }
}

impl Serialize for Place {
    /// A typed place is the bare array it has always been; a taken one is `{"from": <node>}`.
    ///
    /// **So no file changes shape unless it uses the new thing.** A document full of typed
    /// centres serialises byte-for-byte as it did before this type existed, which is what makes
    /// this a change with no migration and no version bump.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Place::At(xyz) => xyz.serialize(serializer),
            Place::From { node } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("from", node)?;
                map.end()
            }
            Place::StartOf { curve } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("start_of", curve)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Place {
    /// An array is a typed place, an object is a taken one.
    ///
    /// Hand-written rather than derived because the two shapes have to be told apart by looking:
    /// every file written before this type existed holds the array, and reading those correctly
    /// without a migration step is the whole reason the array is still what a typed place looks
    /// like.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        if value.is_array() {
            let xyz: [ParamRef; 3] =
                serde_json::from_value(value).map_err(D::Error::custom)?;
            return Ok(Place::At(xyz));
        }
        let object = value.as_object().ok_or_else(|| {
            D::Error::custom("a place is three numbers or a block to take it from")
        })?;
        if let Some(curve) = object.get("start_of") {
            return Ok(Place::StartOf {
                curve: serde_json::from_value(curve.clone()).map_err(D::Error::custom)?,
            });
        }
        let node = object.get("from").ok_or_else(|| {
            D::Error::custom("a place taken from a block needs a \"from\" or a \"start_of\"")
        })?;
        Ok(Place::From {
            node: serde_json::from_value(node.clone()).map_err(D::Error::custom)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A typed place is still the array it always was**, which is what makes this change
    /// costless to every file already written.
    #[test]
    fn a_typed_place_reads_and_writes_as_a_bare_array() {
        let place = Place::at(1.0, -2.5, 3.0);
        let text = serde_json::to_string(&place).expect("serialises");
        assert!(
            text.starts_with('['),
            "a typed place must still look like the three numbers it was, got {text}"
        );
        assert_eq!(serde_json::from_str::<Place>(&text).expect("reads back"), place);

        // And the shape a file written before this type existed holds, which is whatever
        // `[ParamRef; 3]` writes -- asserted against that rather than against a literal, so it
        // stays true if a parameter's own spelling ever changes.
        let old = serde_json::to_string(&[
            ParamRef::Literal(1.0),
            ParamRef::Literal(-2.5),
            ParamRef::Literal(3.0),
        ])
        .expect("the old shape serialises");
        assert_eq!(text, old, "a typed place writes exactly what the array wrote");
        assert_eq!(serde_json::from_str::<Place>(&old).expect("an old file reads"), place);
    }

    /// A taken place round-trips, and carries its edge.
    #[test]
    fn a_taken_place_round_trips_and_names_its_source() {
        let node = NodeId::new();
        let place = Place::From { node };
        let text = serde_json::to_string(&place).expect("serialises");
        assert_eq!(serde_json::from_str::<Place>(&text).expect("reads back"), place);
        assert_eq!(place.source(), Some(node));
        assert_eq!(Place::at(0.0, 0.0, 0.0).source(), None);
    }

    /// Neither shape is mistaken for the other, and nonsense is refused rather than guessed at.
    #[test]
    fn a_place_that_is_neither_shape_is_refused() {
        assert!(serde_json::from_str::<Place>("7").is_err());
        assert!(serde_json::from_str::<Place>("{}").is_err(), "an object needs a from");
    }
}
