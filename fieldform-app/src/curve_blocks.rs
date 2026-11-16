// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The blocks built from curves, as the commands that create them name them.

use crate::*;

/// What a refusal of a loop that does not close gains, now that there is a second way on.
///
/// Joining the ends was the only answer while a block could be built on a closed profile and
/// nothing else. A curve can be given a thickness instead, and that is a checkbox on the very
/// block the refusal is about — so the refusal says so (`outline-wall-spec.md` §6).
pub(crate) const OR_A_WALL: &str = "Or turn on Wall in the block to give the curve a thickness instead.";

/// A block that is built from curves, for the one function that makes any of them waiting
/// (`App::add_waiting`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CurveBlock {
    Extrude,
    Revolve,
    Sweep,
    Loft,
    Pipe,
}

impl CurveBlock {
    /// What it is called, and what it is waiting for, for the line that says so.
    pub(crate) fn waiting(self) -> (&'static str, &'static str) {
        match self {
            CurveBlock::Extrude => ("an extrude", "its profile — name one in the block; it is drawn once it has one"),
            CurveBlock::Revolve => ("a revolve", "its profile — name one in the block; it is drawn once it has one"),
            CurveBlock::Sweep => ("a sweep", "its profile and its path — name them in the block; it is drawn once it has both"),
            CurveBlock::Loft => ("a loft", "its sections — add them in the block; it is drawn once it has two"),
            CurveBlock::Pipe => ("a pipe", "its curve — name one in the block; it is drawn once it has one"),
        }
    }
}

/// Whether a name typed for a slot is the word that leaves it empty (`console_keys::NOTHING`).
pub(crate) fn names_nothing(name: &str) -> bool {
    name.trim().eq_ignore_ascii_case(console_keys::NOTHING)
}

/// Which of the three blocks built on a section a wall is being made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WallBlock {
    Extrude,
    Revolve,
    /// With the spine it was told, if it was told one, which it is built on from the start
    /// (`App::add_sweep_body` says why). Carried here because a sweep is the one of the three that
    /// has a spine, and every way to a wall sweep has to hand it over. `sweep curve= spine=` with
    /// no closed profile anywhere came by a way that did not, and swept along a curve of its own
    /// picking (refactor plan 2.22).
    Sweep { spine: Option<CurveId> },
}

impl WallBlock {
    /// What was done, for the line that says so.
    pub(crate) fn did(self) -> &'static str {
        match self {
            WallBlock::Extrude => "Extruded",
            WallBlock::Revolve => "Revolved",
            WallBlock::Sweep { .. } => "Swept",
        }
    }

}

/// The other side of a curve: left for right, and the middle for itself.
pub(crate) fn across(side: fieldform_scene::WallSide) -> fieldform_scene::WallSide {
    use fieldform_scene::WallSide;
    match side {
        WallSide::Centre => WallSide::Centre,
        WallSide::Left => WallSide::Right,
        WallSide::Right => WallSide::Left,
    }
}

/// The three sides a wall is offered, each as the side it stores, the word it is shown by and
/// what the word means.
///
/// **In the words the curve under it makes true.** Along an open curve there is a left and a
/// right. Round a loop a person means inside and outside, and which of left and right that is
/// depends on the way the loop happens to run — so `inside` is the side the field says is the
/// inside (`Wall::inside`), and the same three buttons store whichever word is true of it.
/// `None` is a curve that closes no loop.
pub(crate) fn side_choices(
    inside: Option<fieldform_scene::WallSide>,
) -> [(fieldform_scene::WallSide, &'static str, &'static str); 3] {
    use fieldform_scene::WallSide;
    let centre = (
        WallSide::Centre,
        "Centre",
        "Half the thickness to each side of the curve.",
    );
    match inside {
        None => [
            centre,
            (
                WallSide::Left,
                "Left",
                "All of it to the left of the curve, walking along it the way it was drawn. \
                 Which way that is on screen is quickest read off the picture.",
            ),
            (
                WallSide::Right,
                "Right",
                "All of it to the right of the curve, walking along it the way it was drawn.",
            ),
        ],
        Some(inside) => [
            centre,
            (inside, "Inside", "All of it inside the loop, so the loop is the wall's outer face."),
            (
                across(inside),
                "Outside",
                "All of it outside the loop, so the loop is the wall's inner face.",
            ),
        ],
    }
}

/// What a loft's panel says about the loft as built -- see `App::loft_readings` for why it is
/// kept rather than asked each frame.
#[derive(Clone, Debug)]
pub(crate) struct LoftReadings {
    /// What the field divides by to stay a bound; said as "reads N× short" past 1.05.
    pub(crate) stretch: f64,
    /// How far each section's plane is from square to the centerline where it sits, in degrees.
    pub(crate) tilts: Vec<f64>,
    /// The narrowest the solid gets between two sections as a share of the smaller, and
    /// which span -- `loft-spec.md` §6.
    pub(crate) neck: Option<(usize, f64)>,
    /// The centerline the loft built for itself, when it was given none, for the viewport to
    /// draw dashed.
    pub(crate) derived: Option<Vec<Point3<f32>>>,
}

/// What `loft`, `loft sections=a,b,c spine=d --g1` asks for, read against the document: the
/// sections by name in the order given -- or, with none named, every closed curve and every
/// closed loop in the document in list order -- the centerline by name or none, and the
/// smoothness.
///
/// **`sections=a,b,c` rather than the spec's bare `s1 s2 s3`**, because this console is
/// `key=value` and a bare word is reported rather than read (item 70's note made the same
/// call). A name that matches no curve is refused by name, not skipped: a loft of the wrong
/// sections is a solid that looks nearly right.
pub(crate) fn loft_asked(
    doc: &fieldform_scene::Document,
    loops: &[Vec<CurveId>],
    args: &Args,
) -> Result<(Vec<CurveId>, Option<CurveId>, fieldform_scene::LoftSmooth), String> {
    let by_name = |name: &str| -> Result<CurveId, String> {
        doc.curves
            .iter()
            .find(|(_, c)| c.name.eq_ignore_ascii_case(name.trim()))
            .map(|(id, _)| *id)
            .ok_or_else(|| format!("No curve is named {name:?}"))
    };
    // `spine=none` is the centerline built from the end sections, said out loud: the one word that
    // empties a slot everywhere else (`console_keys::NOTHING`), and the echo's spelling of a loft
    // with no spine of its own.
    let spine = match args.get("spine").and_then(Value::as_str) {
        Some(name) if name.trim().eq_ignore_ascii_case(console_keys::NOTHING) => None,
        Some(name) => Some(by_name(name)?),
        None => None,
    };
    let sections = match args.get("sections").and_then(Value::as_str) {
        // `sections=none`: a loft with no sections yet, said out loud -- the word that empties
        // a slot everywhere else, and the echo's spelling of a loft still waiting.
        Some(list) if names_nothing(list) => Vec::new(),
        Some(list) => list
            .split(',')
            .filter(|w| !w.trim().is_empty())
            .map(by_name)
            .collect::<Result<Vec<_>, _>>()?,
        None => {
            // A closed curve is its own section; a loop of several is one section, named by
            // the curve that heads it -- the key `profile_partition` gives the row.
            let mut found: Vec<CurveId> = doc
                .curves
                .iter()
                .filter(|(_, c)| c.is_closed())
                .map(|(id, _)| *id)
                .chain(loops.iter().filter_map(|l| l.first().copied()))
                .filter(|id| Some(*id) != spine)
                .collect();
            found.sort_by_key(|id| id.raw());
            found.dedup();
            found
        }
    };
    let flagged = |key: &str| args.get(key).is_some_and(|v| !matches!(v, Value::Bool(false)));
    // `smooth=g0|g1` is the field's own name (the project notes); `--g1` is the flag this was written
    // with and still reads, so a script already using it keeps working.
    let smooth = match args.get("smooth").map(console_keys::loft_smooth_named).transpose()? {
        Some(smooth) => smooth,
        None if flagged("g1") => fieldform_scene::LoftSmooth::G1,
        None => fieldform_scene::LoftSmooth::G0,
    };
    Ok((sections, spine, smooth))
}
