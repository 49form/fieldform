// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! How a number is shown and scrubbed: the widths, the ranges each kind of number may take, and the rounding.

use crate::*;

/// A number as a block shows it: `23` rather than `23.0000`, and at most four decimals,
/// which is a tenth of a micron on a millimetre and more than any field here resolves.
pub(crate) fn readout(v: f64) -> String {
    let text = format!("{v:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    // A value that rounds to nothing is nothing, not "-0".
    if text.is_empty() || text == "-" || text == "-0" { "0".to_string() } else { text.to_string() }
}

/// Rounds a length to a tidy value of similar magnitude — 52.6 becomes 53, 0.0417 becomes
/// 0.042. Used for defaults offered to the user, where a suspiciously precise number implies a
/// precision that was never meant.
pub(crate) fn tidy_mm(v: f64) -> f64 {
    if !v.is_finite() || v <= 0.0 {
        return 1.0;
    }
    let magnitude = 10f64.powf(v.log10().floor());
    (v / magnitude * 10.0).round() / 10.0 * magnitude
}

/// How wide one coordinate's field is, everywhere in the application.
///
/// **Fixed, so columns line up across panels and not merely within one.** A position reads as a
/// position when its three numbers are on one line, and two positions read as a pair when the
/// X of one sits above the X of the other. Rows that size to their own content cannot promise
/// that — the same two positions come out ragged whenever their numbers differ in length,
/// which is most of the time. A grid aligns a column within one widget; a shared width aligns
/// it between a curve's ends, a sphere's centre and a transform's move.
pub(crate) const AXIS_FIELD_W: f32 = 98.0;

/// And the label ahead of them, so the fields themselves start at the same place.
///
/// **Right-aligned in it**, which is the half that was missing. The label used to be centred in
/// this width, so a row called "At:" started its text in a different place from one called
/// "From radius:" and the left edge of a block was ragged — asked for as "organise the inside of
/// the blocks better, more on a grid". Right-aligned, every label ends where the numbers begin
/// and the eye has one line to run down.
///
/// Wide enough for the longest label a block row uses. The long ones in the import and export
/// dialogs are plain `ui.label` calls and do not come through here.
pub(crate) const AXIS_LABEL_W: f32 = 84.0;

/// One number's own box, inside its cell.
///
/// Fixed, so the boxes are the same size down the panel and across it. A `DragValue` sizes
/// itself to whatever it is currently showing otherwise, so a column of them steps in and out as
/// the numbers change length — which is what stops a panel reading as a grid even when its
/// columns are aligned.
pub(crate) const NUMBER_W: f32 = 64.0;

/// How fast a number scrubs, given how big it currently is.
///
/// **A fixed rate cannot serve both ends of a CAD document.** The position rows moved at half a
/// millimetre a pixel, which is right for a fillet and useless for a span: a two-metre dimension
/// wants four thousand pixels of travel and there is no screen that wide, while a 0.2 mm fillet
/// moves in steps larger than itself. So the rate follows the magnitude, with the row's own speed
/// as a floor — so a value sitting at zero still moves, and a small one keeps the feel it had.
///
/// Four parts in a thousand per pixel, which puts any number within about two hundred and fifty
/// pixels of doubling: a drag of a couple of inches, whatever it currently reads.
///
/// The other two halves of this gesture are egui's and are not repeated here: holding shift
/// divides the rate by ten, and a `DragValue` already shows the horizontal-resize cursor, so the
/// field says it can be dragged before anybody tries. Ctrl's coarse steps are not egui's, and
/// [`scrub_rate`] adds them.
pub(crate) fn scrub_speed(value: f64, base: f64) -> f64 {
    (value.abs() * 0.004).max(base)
}

/// [`scrub_speed`] as the drag stands this frame: ten times coarser while Ctrl is held
/// (`interaction-spec.md` §4.4), so a number field in the notebook steps as the tool panel's does.
/// Read every frame, so pressing Ctrl part-way through a drag changes the step from there on.
pub(crate) fn scrub_rate(ui: &egui::Ui, value: f64, base: f64) -> f64 {
    let coarse = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
    scrub_speed(value, base) * if coarse { 10.0 } else { 1.0 }
}

/// The axis letter ahead of a number's box ([`NUMBER_W`]), for the rows that have one.
pub(crate) const AXIS_LETTER_W: f32 = 11.0;

/// A size: positive, and bounded well past anything anyone models.
///
/// Zero is excluded because a radius of zero has no surface — the shape would simply not be
/// there, which reads as the tool being broken rather than as the number being wrong.
pub(crate) const POSITIVE_MM: std::ops::RangeInclusive<f64> = 0.001..=1.0e6;

/// A join's radius where nought is an answer: a pattern's fillet has no switch beside it, so
/// zero is how it is turned off. With [`POSITIVE_MM`] the field refused the zero its own
/// caption offered -- "Nought is a hard join" -- and left 0.001, which is a rounded join nobody
/// can see and which costs everything a visible one does.
pub(crate) const FROM_NOUGHT_MM: std::ops::RangeInclusive<f64> = 0.0..=1.0e6;

/// A coordinate: anywhere, including zero and the negative half of every axis.
pub(crate) const ANYWHERE_MM: std::ops::RangeInclusive<f64> = -1.0e6..=1.0e6;

/// Rotations are unbounded rather than wrapped to a turn: a manipulator drag accumulates past
/// 360 degrees on purpose (§6.5), and clamping the readout there would make a continuous drag
/// appear to stop.
pub(crate) const DEGREES: std::ops::RangeInclusive<f64> = -1.0e5..=1.0e5;

/// A scale factor: above nought, since nought has no inverse and a negative scale is a mirror.
/// The floor is a thousandth rather than nought so the widget cannot drag a body to nothing.
pub(crate) const SCALE_FACTOR: std::ops::RangeInclusive<f64> = 1.0e-3..=1.0e4;

/// A coil's turns: above nought, since none is no wire, and not whole by necessity.
pub(crate) const COIL_TURNS: std::ops::RangeInclusive<f64> = 1.0e-3..=1.0e4;

/// How far round a revolve turns: a full turn is the whole solid, and the widget stops there
/// rather than letting a drag run on past it, since past it nothing changes. **A degree at
/// the least, not nought**: nought is a solid of nothing, which the compile refuses by the
/// number, and a drag that could reach it took the part away under the hand and brought it
/// back a few degrees later -- see the compile error's slot in the part panel for what that
/// did to the drag. The console can still ask for nought and be told why not.
pub(crate) const REVOLVE_DEGREES: std::ops::RangeInclusive<f64> = 1.0..=360.0;

/// The other way round, which may be nothing: side two of a revolve turned only one way.
pub(crate) const REVOLVE_OTHER_DEGREES: std::ops::RangeInclusive<f64> = 0.0..=360.0;

/// A draft angle, which is a wall's lean and so cannot reach a right angle.
///
/// Stopped short of ninety on both sides rather than left open: at ninety the wall is a plane and
/// past it the solid folds back through itself, which is a shape nobody drew. The compile refuses
/// beyond this too, so the two agree; this is what stops a drag reaching it in the first place.
pub(crate) const DRAFT_DEGREES: std::ops::RangeInclusive<f64> = -89.0..=89.0;

/// How big a swept section may end up, as a multiple of the drawn one. **Never nought**: a
/// scale of nothing is a section closed to a point before the end of the path, which the
/// kernel refuses by name -- so the widget stops a hundredth short of it rather than letting
/// a drag take the part away and bring it back, which is what a revolve's angle did.
pub(crate) const TAPER_SCALE: std::ops::RangeInclusive<f64> = 0.01..=20.0;

/// How far round a swept section turns over its extent. Unbounded in the sense a draft is
/// not: more than a full turn is a helix of the section and a perfectly ordinary thing to ask
/// for, and the only cost is the divisor, which the panel states.
pub(crate) const TWIST_DEGREES: std::ops::RangeInclusive<f64> = -3600.0..=3600.0;

/// Where a block starts and ends along its path, as a percentage of the path's length.
pub(crate) const PATH_PERCENT: std::ops::RangeInclusive<f64> = 0.0..=100.0;

/// What a new attractor pulls at: half, which is a visible effect well clear of folding in
/// either direction. The block's panel is where it is changed afterwards.
pub(crate) const DEFAULT_ATTRACT_STRENGTH: f64 = 0.5;

/// An attractor's strength: a fraction, and wide enough on both sides to reach a fold.
///
/// **Not clamped short of folding**, which §8 of the spec expected. The fold threshold is a
/// real number rather than a guess -- `16/11` pulling, `-1` pushing -- so the panel can say
/// where it is and let somebody go past it deliberately. A slider that stopped at 1 would be
/// hiding a limit it knows exactly, and folding out of sight is a legitimate thing to ask for.
pub(crate) const ATTRACT_STRENGTH: std::ops::RangeInclusive<f64> = -4.0..=4.0;

/// A direction's components. Wide rather than clamped to a unit ball, because a direction of
/// any length means the same thing once it is normalized, and silently rewriting a typed 2 to a
/// 1 would look like the field refusing to accept what was typed.
pub(crate) const DIRECTION: std::ops::RangeInclusive<f64> = -1.0e6..=1.0e6;

/// Which of three numbers was dragged, and by what factor, for [`App::size_row`]'s lock.
///
/// Free of the app so the arithmetic can be asserted: an `App` needs a window and a device
/// before it will exist, which is the arrangement `operand_choices_in` and `starting_pair_in`
/// already use.
///
/// `None` when nothing moved, when more than one did — a frame that changed two is not a drag of
/// one and guessing which led would be a guess — or when the one that moved started at zero,
/// where every factor is zero and the ratio has no meaning to preserve.
pub(crate) fn leading_factor(before: [f64; 3], after: [f64; 3]) -> Option<(usize, f64)> {
    let mut moved = (0..3).filter(|&i| before[i] != after[i]);
    let led = moved.next()?;
    if moved.next().is_some() || before[led].abs() < 1e-9 {
        return None;
    }
    Some((led, after[led] / before[led]))
}
