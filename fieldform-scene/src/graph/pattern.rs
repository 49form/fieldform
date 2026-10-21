// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A pattern's kinds, options and count, and the axis a mirror, a twist and a frame each name.

use super::ParamRef;
use crate::Frame;
use serde::{Deserialize, Serialize};
#[cfg(doc)] use super::{CellMap, Node};

/// How a pattern repeats its body.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PatternKind {
    /// Each copy displaced by a further `step` from the last.
    Linear {
        /// The offset between neighbours, as a vector — so a row can run along any direction
        /// rather than only an axis, and a `Transform` around the result is not needed to aim
        /// something this simple.
        step: [ParamRef; 3],
    },
    /// Copies spread evenly around a turn about an axis — a full one unless
    /// [`PatternOptions::angle`] says otherwise.
    Radial {
        /// The axis the copies spread about, and a point it passes through.
        frame: Frame,
    },
}

/// What a row's step number means. See [`PatternOptions::spacing`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Spacing {
    /// The vector is the gap between neighbours: the last of `n` copies is `n − 1` steps away.
    #[default]
    Step,
    /// The vector is the **whole run**, so the gap is the vector over `n − 1` and the last copy is
    /// exactly one vector away however many there are.
    Extent,
}

/// A row's second direction: the copies of the first direction, repeated along another step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PatternSecond {
    pub step: [ParamRef; 3],
    /// Copies along it, **including** the row they repeat — read by [`pattern_count`] exactly as
    /// the first count is, and the product of the two is what [`PATTERN_COUNT_MAX`] caps.
    pub count: ParamRef,
}

/// What a pattern can do beyond repeating a body evenly, each part left out of the file when it is
/// what an older document meant.
///
/// **One field on the node rather than a field per option on each kind**, so that a pattern which
/// uses none of this is built, written and read exactly as it was — and so that a panel showing the
/// options once serves a row and a ring alike (D9). Which part applies to which kind is said on the
/// field: `spacing` and `second` are a row's, `angle` a ring's, the rest both.
///
/// **Default is what a pattern always did**: gaps of one step, the body at the first copy, a full
/// turn, nothing left out.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternOptions {
    /// A row's: whether its step is a gap or the whole run. Ignored by a ring.
    pub spacing: Spacing,
    /// Both: copies centred on the body rather than starting at it, so offsets run from
    /// `−(n − 1)/2` to `+(n − 1)/2` steps.
    ///
    /// **The body is a copy only when the count is odd.** At an even count the middle falls between
    /// two copies and none of them stands where the body does, which is the honest reading of
    /// "centred on the original" and is why a block that does not hold the body then has nothing to
    /// leave out of its copies.
    pub symmetric: bool,
    /// A ring's: how far round the copies spread, in degrees, 360 by default. A full turn puts `n`
    /// copies `360/n` apart; **less than a full turn puts them `angle/(n − 1)` apart**, so the
    /// first and last sit at the two ends of the angle. Ignored by a row.
    pub angle: ParamRef,
    /// A row's: a second direction, giving a grid of `count × count2` copies. Ignored by a ring.
    pub second: Option<PatternSecond>,
    /// Both: 0-based copy numbers to leave out. Numbers past the last copy are kept and mean
    /// nothing, so shrinking the count does not erase what was chosen.
    pub skip: Vec<u32>,
}

impl Default for PatternOptions {
    fn default() -> Self {
        Self {
            spacing: Spacing::Step,
            symmetric: false,
            angle: ParamRef::Literal(FULL_TURN_DEGREES),
            second: None,
            skip: Vec::new(),
        }
    }
}

/// A full turn, in the unit a ring's angle is typed in.
pub const FULL_TURN_DEGREES: f64 = 360.0;

impl PatternOptions {
    /// Whether this is what a pattern did before it had options, which is what the file leaves out.
    pub fn is_plain(&self) -> bool {
        *self == Self::default()
    }
}

/// One of the frame's three axes.
///
/// Not `MirrorAxis`, which is what it was called while a mirror was the only thing that needed
/// one. A twist turns about an axis too, and a type named for its first caller is a name that
/// misleads the second — the same drift `CageField` and `ChordHeight` went through.
///
/// A mirror reads it as the direction its plane is *perpendicular* to, which is how every CAD
/// tool asks the question: "mirror in X" reflects left to right. A twist reads it as the axis
/// it winds about.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    pub fn label(self) -> &'static str {
        match self {
            Axis::X => "X",
            Axis::Y => "Y",
            Axis::Z => "Z",
        }
    }
}

/// The largest copy count a pattern will build.
///
/// **A measured shader-size fact, not a modelling tolerance.** 512 spheres generate about 478 KB of
/// WGSL, and the count is what the window is sized from, so a count in the tens of thousands is a
/// source the driver spends a long time on. It is the one limit on a count that clamps rather than
/// refuses, and the row says when it has bitten — a cap that said nothing would be exactly the
/// disagreement [`CountFit`] exists to prevent.
pub const PATTERN_COUNT_MAX: u32 = 512;

/// Why a pattern's count could not be used.
///
/// Separate from the number so the caller can say *which* way it failed where the count is typed.
///
/// **A fraction is no longer one of these.** It was, from the owner's first ruling on 2026-09-30 —
/// refuse rather than round, since `floor()` and `round()` are in the expression language. Reversed
/// the same day, by the owner, on the consequence: "the decision we made was that it may break if
/// it's a decimal, and obviously that doesn't work." A variable is a thing people drive from
/// elsewhere, and half of what drives one lands on a fraction on the way — `copies / 2`, a ratio, an
/// interpolation. A count that refuses mid-drive blanks the part for a value nobody typed and nobody
/// meant to dwell on. So a fraction is rounded, and [`CountFit`] is how the panel says it was.
///
/// What is left here is what cannot be repaired into a usable number without inventing geometry.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CountError {
    /// The expression did not evaluate — an unknown name, a division by zero, a reference to
    /// another failure, or a parameter the document no longer holds.
    Unresolved,
    /// It evaluated to something that is not a number.
    NotFinite,
    /// It rounded to zero or less. A pattern of nothing is not the body — it is a broken expression
    /// wearing the body's clothes, which is why this is named rather than clamped to one. Carries
    /// the value before rounding, because that is the number on screen.
    NotPositive(f64),
    /// A ramp drives it. A ramp's value varies with the evaluation point and a copy count cannot,
    /// so this is refused by name rather than read as the zero [`ParamRef::resolve`] gives.
    Driven,
}

impl CountError {
    /// What to put in front of the user, naming the count so the message is about the field they
    /// typed in rather than about the block.
    pub fn say(self) -> String {
        match self {
            CountError::Unresolved => {
                "a pattern's count did not work out — check the expression driving it".to_string()
            }
            CountError::NotFinite => "a pattern's count is not a number".to_string(),
            CountError::NotPositive(v) => {
                format!("a pattern's count works out to {v}, and a pattern needs at least one copy")
            }
            CountError::Driven => {
                "a pattern's count cannot be driven by a ramp, because a ramp's value changes with \
                 position and a number of copies cannot"
                    .to_string()
            }
        }
    }
}

/// What [`pattern_count`] had to do to a number before it was a count.
///
/// **Because quantising silently is the one thing `arc_count`'s doc forbids.** Its second clause is
/// that the number the user typed and the geometry they get may not disagree *without saying so* —
/// and once a fraction is rounded rather than refused, this is the saying so. It is the arrangement
/// [`CellMap::arc_count_for`] already uses, where the count is quantised and the achieved size is
/// reported back rather than assumed, and the one an approximate blend's mark uses beside its radius.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CountFit {
    /// Already a whole number within range. Nothing to say.
    Exact,
    /// Rounded to the nearest whole number, from this value.
    Rounded(f64),
    /// Above the ceiling and capped, from this value. Reported in preference to the rounding when
    /// both happened, because a cap changes the count by more than a half and is the bigger surprise.
    Capped(f64),
}

impl CountFit {
    /// What to say beside the number, or `None` when the number is exactly what was asked for.
    ///
    /// Takes the count because every message that is worth printing names it: the point is to put
    /// the figure actually built next to the expression that did not say it.
    pub fn say(self, count: u32) -> Option<String> {
        match self {
            CountFit::Exact => None,
            CountFit::Rounded(from) => Some(format!("{count} copies, rounded from {from}")),
            CountFit::Capped(from) => Some(format!(
                "capped at {count} — the expression works out to {from}"
            )),
        }
    }
}

/// The whole number of copies a pattern builds, or why it has none.
///
/// **The only way a [`Node::Pattern::count`] becomes an integer.** Seven places read the count —
/// the field builder, the reach, the GPU's two passes, `pattern_first_of`, and the panel — and
/// before this they all read the same `u32` so they could not disagree. With a `ParamRef` between
/// the stored value and the integer, seven open-coded `resolve`-then-`as u32` conversions would be
/// seven chances to differ, and `f64 as u32` saturates silently: a NaN becomes nought and a
/// nonsense expression draws one plausible copy. That is the mistake `cells_reached` sitting beside
/// `pattern_reach` already cost once.
///
/// **Asked of the document, never read off the parameter**, which is why this takes a `Document`:
/// `parameter_value` evaluates the expression through everything it reads and answers `None` for one
/// that failed, and that `None` is the same one an unknown name gives.
///
/// **A count is always a whole number, including when a variable drives it.** Rounded here, on the
/// way out, and never written back: a parameter is document-wide and the same name may drive a
/// radius, so rounding what is *stored* would move that radius to suit this count. Quantise on read
/// is the whole of the rule, and [`CountFit`] carries what the quantising did so the panel can say
/// it where the number is typed.
///
/// Rounded rather than floored, following [`CellMap::arc_count_for`], which is the only other
/// quantised count here. `floor()` and `ceil()` remain in the expression language for anyone who
/// means one of those — this is the default, not a decision taken away.
///
/// The ceiling caps, and a value that will not round to at least one copy, will not resolve, or is
/// driven by a ramp is refused — see [`PATTERN_COUNT_MAX`] and [`CountError`].
pub fn pattern_count(
    document: &crate::Document,
    count: &ParamRef,
) -> Result<(u32, CountFit), CountError> {
    if let ParamRef::Field(_) = count {
        return Err(CountError::Driven);
    }
    let value = count
        .resolve(&|id| document.parameter_value(id))
        .ok_or(CountError::Unresolved)?;
    if !value.is_finite() {
        return Err(CountError::NotFinite);
    }
    let whole = value.round();
    // Refused on the *rounded* value, so 0.6 is one copy rather than a refusal, and only something
    // that genuinely asks for nothing is turned away. The message carries the number on screen.
    if whole < 1.0 {
        return Err(CountError::NotPositive(value));
    }
    if whole > f64::from(PATTERN_COUNT_MAX) {
        return Ok((PATTERN_COUNT_MAX, CountFit::Capped(value)));
    }
    // Compared against the value rather than tracked through the arithmetic, so `6.0` is `Exact`
    // and nothing is said about it. An exact integer rounds to itself.
    let fit =
        if whole == value { CountFit::Exact } else { CountFit::Rounded(value) };
    Ok((whole as u32, fit))
}
