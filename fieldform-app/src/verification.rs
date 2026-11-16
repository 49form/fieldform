// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The record of a part having been checked against what was promised of it.

use crate::*;

/// The last `verify`, kept with the state it was true of.
///
/// **Not the document, and not a second copy of it**: a report is derived, and the only thing
/// remembered about the part is a fingerprint to tell whether the report still describes it.
/// Any edit since makes it stale, and a stale report reads as *unverified* — never as the pass
/// it may once have been (blend-integrity-spec §5, the rule the Expression block's verify
/// follows too). Dated, because "held" is a claim about a moment.
pub(crate) struct Verification {
    pub(crate) part: PartId,
    pub(crate) fingerprint: String,
    /// The day it ran, UTC, as `2026-09-18`.
    pub(crate) when: String,
    pub(crate) reports: Vec<fieldform_field::CriticalReport>,
}

/// Today, UTC, as `YYYY-MM-DD`. Days-from-civil run backwards, so the application does not
/// take on a date library for one line of a report.
pub(crate) fn utc_date(seconds_since_epoch: u64) -> String {
    let z = (seconds_since_epoch / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Every named value with a number, for the menus that offer them.
///
/// A computed one is offered at what it currently computes to; one whose expression has
/// failed has no number to show beside its name and is left out until it has.
pub(crate) fn named_values(app: &App) -> Vec<(ParamId, String, f64)> {
    let mut all: Vec<(ParamId, String, f64)> = app
        .doc
        .parameters
        .values()
        .filter_map(|p| app.doc.parameter_value(p.id).map(|v| (p.id, p.name.clone(), v)))
        .collect();
    all.sort_by(|a, b| a.1.cmp(&b.1));
    all
}
