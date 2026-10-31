// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a coil and a thread are told beyond their frame: which way they wind, how a coil's
//! size is given, and what its wire's section is.
//!
//! A helix is a **coordinate**, not a curve: read a point as a radius and a height reduced by
//! the pitch times the turn, and any section drawn in the axial half-plane is carried
//! helically about the axis. The coil and the thread are the two blocks built on it, and the
//! types here are the words both are told in. The helix-thread design note carries the
//! decisions; this file carries the vocabulary.

use serde::{Deserialize, Serialize};

use crate::graph::ParamRef;

/// Which way a helix winds, looking along its axis.
///
/// A right-hand helix climbs anticlockwise about its frame's `+Z`, which is what every ordinary
/// screw does; left-handed is its mirror image, and the field makes it one by mirroring the
/// query across the frame's `xz`-plane before anything else is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hand {
    #[default]
    Right,
    Left,
}

impl Hand {
    /// The word a file and the console use for it.
    pub fn name(self) -> &'static str {
        match self {
            Hand::Right => "right",
            Hand::Left => "left",
        }
    }

    /// The hand a word names, if it names one.
    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "right" | "rh" => Some(Hand::Right),
            "left" | "lh" => Some(Hand::Left),
            _ => None,
        }
    }

    /// `+1` for right, `−1` for left: what the query's `y` is multiplied by to wind the other
    /// way, on the CPU and in the shader alike.
    pub fn sign(self) -> f64 {
        match self {
            Hand::Right => 1.0,
            Hand::Left => -1.0,
        }
    }
}

/// How big a coil is, said as the two of its three numbers that were typed.
///
/// Turns, pitch and height are one relation (`height = turns · pitch`), so a block storing all
/// three would hold two descriptions of one fact and the pair could disagree — the `Ends`
/// principle. The two typed are stored; the third is derived where it is read
/// ([`CoilSize::turns_and_pitch`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoilSize {
    /// So many turns, this far apart.
    TurnsPitch { turns: ParamRef, pitch: ParamRef },
    /// This tall, with the turns this far apart.
    HeightPitch { height: ParamRef, pitch: ParamRef },
    /// So many turns, this tall altogether.
    TurnsHeight { turns: ParamRef, height: ParamRef },
}

impl Default for CoilSize {
    /// Five turns at a pitch of six: a spring that reads as one on the Create menu.
    fn default() -> Self {
        CoilSize::TurnsPitch { turns: ParamRef::Literal(5.0), pitch: ParamRef::Literal(6.0) }
    }
}

impl CoilSize {
    /// The number of turns and the pitch, from whichever pair was typed. `None` where a number
    /// did not resolve or the derivation divides by nought; the compile refuses that by name.
    pub fn turns_and_pitch(&self, resolve: &dyn Fn(&ParamRef) -> Option<f64>) -> Option<(f64, f64)> {
        Some(match self {
            CoilSize::TurnsPitch { turns, pitch } => (resolve(turns)?, resolve(pitch)?),
            CoilSize::HeightPitch { height, pitch } => {
                let pitch = resolve(pitch)?;
                if pitch == 0.0 {
                    return None;
                }
                (resolve(height)? / pitch, pitch)
            }
            CoilSize::TurnsHeight { turns, height } => {
                let turns = resolve(turns)?;
                if turns == 0.0 {
                    return None;
                }
                (turns, resolve(height)? / turns)
            }
        })
    }

    /// The word a file and the console use for the pair.
    pub fn name(&self) -> &'static str {
        match self {
            CoilSize::TurnsPitch { .. } => "turns_pitch",
            CoilSize::HeightPitch { .. } => "height_pitch",
            CoilSize::TurnsHeight { .. } => "turns_height",
        }
    }

    /// The two numbers, in the order the shader reads them: `(first, second)` for the pair's
    /// own two words in the order the name says them.
    pub fn pair(&self) -> (&ParamRef, &ParamRef) {
        match self {
            CoilSize::TurnsPitch { turns, pitch } => (turns, pitch),
            CoilSize::HeightPitch { height, pitch } => (height, pitch),
            CoilSize::TurnsHeight { turns, height } => (turns, height),
        }
    }

    /// The numbers this size carries, so they take part in parameter resolution like any other.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let (a, b) = self.pair();
        [a, b].into_iter()
    }

    /// The same, to be edited.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            CoilSize::TurnsPitch { turns, pitch } => vec![turns, pitch],
            CoilSize::HeightPitch { height, pitch } => vec![height, pitch],
            CoilSize::TurnsHeight { turns, height } => vec![turns, height],
        }
    }
}

/// The section of a coil's wire, drawn in the axial half-plane and centred on the coil's mean
/// diameter — stated so nobody reads it as the section square to the wire.
///
/// One number either way, named `wire` on both so the console has one key for it: a circle's
/// diameter, or a square's side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoilSection {
    Circle { wire: ParamRef },
    Square { wire: ParamRef },
}

impl Default for CoilSection {
    fn default() -> Self {
        CoilSection::Circle { wire: ParamRef::Literal(2.0) }
    }
}

impl CoilSection {
    pub fn name(&self) -> &'static str {
        match self {
            CoilSection::Circle { .. } => "circle",
            CoilSection::Square { .. } => "square",
        }
    }

    /// The same shape with the word changed and the number kept.
    pub fn with_name(&self, name: &str) -> Option<Self> {
        let wire = self.wire().clone();
        match name {
            "circle" | "round" => Some(CoilSection::Circle { wire }),
            "square" => Some(CoilSection::Square { wire }),
            _ => None,
        }
    }

    /// The one number: a diameter or a side.
    pub fn wire(&self) -> &ParamRef {
        match self {
            CoilSection::Circle { wire } | CoilSection::Square { wire } => wire,
        }
    }

    pub fn wire_mut(&mut self) -> &mut ParamRef {
        match self {
            CoilSection::Circle { wire } | CoilSection::Square { wire } => wire,
        }
    }

    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        std::iter::once(self.wire())
    }

    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        vec![self.wire_mut()]
    }
}

/// The ISO metric coarse series a thread may be named from: designation, major diameter and
/// pitch, in millimetres. V1 is the coarse series alone; fine pitches and unified threads are
/// deferred. Stored once, here, and read by the thread block and — when it comes — the tapped
/// hole.
pub const THREAD_TABLE: &[(&str, f64, f64)] = &[
    ("M2", 2.0, 0.4),
    ("M2.5", 2.5, 0.45),
    ("M3", 3.0, 0.5),
    ("M4", 4.0, 0.7),
    ("M5", 5.0, 0.8),
    ("M6", 6.0, 1.0),
    ("M8", 8.0, 1.25),
    ("M10", 10.0, 1.5),
    ("M12", 12.0, 1.75),
    ("M14", 14.0, 2.0),
    ("M16", 16.0, 2.0),
    ("M20", 20.0, 2.5),
    ("M24", 24.0, 3.0),
];

/// A designation's major diameter and pitch, if the table has it. Case does not matter, so a
/// console line may say `m8`.
pub fn thread_designation(name: &str) -> Option<(f64, f64)> {
    let wanted = name.trim();
    THREAD_TABLE
        .iter()
        .find(|(n, _, _)| n.eq_ignore_ascii_case(wanted))
        .map(|(_, major, pitch)| (*major, *pitch))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whichever pair was typed, the two numbers the field reads come out the same; a pair that
    /// divides by nought answers nothing rather than infinity.
    #[test]
    fn every_pair_gives_the_same_turns_and_pitch_and_a_nought_divisor_answers_nothing() {
        let lit = |v: f64| ParamRef::Literal(v);
        let resolve = |r: &ParamRef| match r {
            ParamRef::Literal(v) => Some(*v),
            _ => None,
        };
        let pairs = [
            CoilSize::TurnsPitch { turns: lit(5.0), pitch: lit(6.0) },
            CoilSize::HeightPitch { height: lit(30.0), pitch: lit(6.0) },
            CoilSize::TurnsHeight { turns: lit(5.0), height: lit(30.0) },
        ];
        for pair in &pairs {
            assert_eq!(pair.turns_and_pitch(&resolve), Some((5.0, 6.0)), "{}", pair.name());
            assert_eq!(pair.params().count(), 2);
        }
        assert_eq!(CoilSize::HeightPitch { height: lit(30.0), pitch: lit(0.0) }.turns_and_pitch(&resolve), None);
        assert_eq!(CoilSize::TurnsHeight { turns: lit(0.0), height: lit(30.0) }.turns_and_pitch(&resolve), None);
    }

    /// The table answers the coarse series by name, whatever the case, and nothing else.
    #[test]
    fn the_thread_table_answers_the_coarse_series_by_name() {
        assert_eq!(thread_designation("M8"), Some((8.0, 1.25)));
        assert_eq!(thread_designation("m8"), Some((8.0, 1.25)));
        assert_eq!(thread_designation(" M2.5 "), Some((2.5, 0.45)));
        assert_eq!(thread_designation("M7"), None);
        assert_eq!(thread_designation("1/4-20"), None);
        assert_eq!(THREAD_TABLE.len(), 13);
    }

    /// A left hand is the mirror of a right one, and the words round-trip.
    #[test]
    fn a_hand_round_trips_through_its_word_and_signs_the_mirror() {
        for hand in [Hand::Right, Hand::Left] {
            assert_eq!(Hand::parse(hand.name()), Some(hand));
        }
        assert_eq!(Hand::Right.sign(), 1.0);
        assert_eq!(Hand::Left.sign(), -1.0);
        assert_eq!(Hand::parse("up"), None);
        let section = CoilSection::Square { wire: ParamRef::Literal(3.0) };
        assert_eq!(section.with_name("circle"), Some(CoilSection::Circle { wire: ParamRef::Literal(3.0) }));
        assert_eq!(section.with_name("hexagon"), None);
    }
}
