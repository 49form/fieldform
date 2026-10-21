// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Units.
//!
//! Geometry is **always** stored in millimetres, as `f64`. A unit is presentation only:
//! switching a document from millimetres to inches rescales nothing, loses no precision, and
//! touches no geometry — it changes how numbers are written and read at the edges.
//!
//! Keeping one canonical internal unit is what stops conversion logic leaking into the
//! geometry code, where a missed conversion would be a silent, wrong-by-25.4 bug. Millimetres
//! because it is the CAD interop convention (STEP's default), and because `f64` has eleven
//! digits to spare below the millimetre even at airframe scale.

use serde::{Deserialize, Serialize};

/// A unit a document may be authored and displayed in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Unit {
    #[default]
    Millimetre,
    Centimetre,
    Metre,
    Inch,
    Foot,
}

impl Unit {
    /// How many millimetres one of this unit is.
    pub fn in_millimetres(self) -> f64 {
        match self {
            Unit::Millimetre => 1.0,
            Unit::Centimetre => 10.0,
            Unit::Metre => 1000.0,
            // Exact by definition since 1959, not an approximation.
            Unit::Inch => 25.4,
            Unit::Foot => 304.8,
        }
    }

    /// Converts a value expressed in this unit into canonical millimetres.
    ///
    /// Every reader outside the tests scales by [`Self::in_millimetres`] itself, so only the
    /// tests convert through this pair, and only the tests compile it.
    #[cfg(test)]
    fn to_millimetres(self, value: f64) -> f64 {
        value * self.in_millimetres()
    }

    /// Converts canonical millimetres into this unit, for display. Test-only for the same
    /// reason as `to_millimetres`.
    #[cfg(test)]
    fn from_millimetres(self, millimetres: f64) -> f64 {
        millimetres / self.in_millimetres()
    }

    pub fn abbreviation(self) -> &'static str {
        match self {
            Unit::Millimetre => "mm",
            Unit::Centimetre => "cm",
            Unit::Metre => "m",
            Unit::Inch => "in",
            Unit::Foot => "ft",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_round_trip() {
        for unit in [
            Unit::Millimetre,
            Unit::Centimetre,
            Unit::Metre,
            Unit::Inch,
            Unit::Foot,
        ] {
            for value in [0.001, 1.0, 137.5, 30_000.0] {
                let back = unit.from_millimetres(unit.to_millimetres(value));
                assert!(
                    (back - value).abs() < 1e-9,
                    "{unit:?} round trip lost {value} -> {back}"
                );
            }
        }
    }

    #[test]
    fn inch_is_exact() {
        // 25.4 exactly, by the 1959 international definition — not a rounded constant.
        assert_eq!(Unit::Inch.to_millimetres(1.0), 25.4);
        assert_eq!(Unit::Foot.to_millimetres(1.0), 304.8);
    }

    #[test]
    fn f64_holds_micron_precision_at_airframe_scale_and_f32_does_not() {
        // The D2 claim, asserted rather than argued. Note what the claim actually is: f32
        // does not *fail* to represent a micron step at 30 m — it quantises it. Asking for
        // 1 µm gets you roughly 2 µm, silently. That is the failure mode that matters for
        // manufacturing tolerance, and it is subtler than "cannot represent".
        let airframe = 30_000.0_f64; // 30 m, in canonical millimetres
        let micron = 0.001_f64;
        let target = airframe + micron;

        let f64_error = (target - airframe - micron).abs();
        assert!(
            f64_error < 1e-9,
            "f64 should hold a micron at airframe scale; error {f64_error}"
        );

        let f32_step = (target as f32) - (airframe as f32);
        assert!(
            (f32_step - micron as f32).abs() > 0.0005,
            "f32 should visibly quantise a micron at this scale; got a step of {f32_step}"
        );
    }
}
