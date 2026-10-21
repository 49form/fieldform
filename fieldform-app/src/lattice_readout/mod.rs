// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What a lattice's stiffness and strength are estimated to be, read from the published fits in
//! `fieldform_field::property_table`, and the number a target stiffness asks of the phase.
//! Backlog item 94; `lattice-qualification-spec.md` §9 and the rulings in §9.1.
//!
//! **Free of the `App`, as `console_keys` is, so every rule here is held by a test.** The panel
//! and the console's `estimate` read the same [`lattice_readout`], so a script asks for exactly
//! what a click shows (ruling 18). It takes `&Document`, which is the whole guarantee that asking
//! changes nothing.
//!
//! **Every number is asked of the document** (`Document::parameter_value`), never read through
//! `App::resolve`, which answers nought for a parameter that fails -- and a wall of nought read as
//! a density of nought is a confident estimate of a lattice that does not exist. A phase goes
//! through `fieldform_field::phase_of`, the conversion the compile uses, so what is estimated is
//! what is built.
//!
//! What the lines say, in order of the rulings that decided it: percent of the solid's own
//! property and never GPa (6); the range across sources first, then the default's value (5);
//! a direction only where the source states one (7); strength with its definition (8);
//! rectangular maps only (9); a solid at the density it builds (10) and not at all below its
//! saddle (11); and a reshaping block downstream named (15).

use std::collections::{HashSet, VecDeque};

use fieldform_field::{
    excluded_for, other_axis_row, phase_of, ranked_rows, uniform_density, CellKind, DataKind,
    DensityBasis, DensityRange, Exponent, LoadDirection, Lower, Phase, Property, PropertyClass,
    PropertyRow, Reading, Refusal, RelativeDensity, SolidBasis, Standing, StrengthKind, TpmsKind,
    Unresolved,
};
use fieldform_scene::{
    Banding, CellMap, Document, Graph, LatticeCell, LatticePhase, Node, NodeId, ParamRef,
};

mod inverse;

pub(crate) use inverse::{
    number_for_target, number_text, offer_text, offers_hover, others_span, typed_percent, SetWrite,
    TargetAnswer,
};
use inverse::source_and_direction;

/// How a line is drawn. The text says the same thing in the console, where there is no colour.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Tone {
    Plain,
    /// Context: how a density was come by, why nothing is said.
    Weak,
    /// The number typed is not what is built, and that changes what everything else reads.
    Warn,
}

/// One row of the readout: a labelled line, and the weak and warning lines under it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReadoutLine {
    /// "Density:", "Density built:", "Stiffness:", "Strength:", "Properties:", or nothing.
    pub label: &'static str,
    pub text: String,
    pub tone: Tone,
    /// Weak: the source, the kind of data, what it divided by, the direction, and "estimate".
    pub detail: Option<String>,
    /// Weak: rows of another strength definition or direction, counted and ranged -- one line,
    /// or on a ramp one per end.
    pub others: Vec<String>,
    /// Weak: on a ramp, what the middle of it reads, which is not the mean of the ends.
    pub note: Option<String>,
    /// Warning-coloured lines: where the reading stands against the source's own range, and
    /// every other source the range or the others line rests on that is read past its own
    /// range or states none (ruling 8).
    pub warn: Vec<String>,
    pub hover: String,
}

impl ReadoutLine {
    /// A line with nothing under it.
    fn bare(label: &'static str, text: String, tone: Tone, hover: String) -> Self {
        ReadoutLine {
            label,
            text,
            tone,
            detail: None,
            others: Vec::new(),
            note: None,
            warn: Vec::new(),
            hover,
        }
    }
}

/// Which number a reading is at: the phase's own typed number, or one end of the ramp that
/// drives it -- named, so the order the user chose survives (a grade may run thick to thin).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum End {
    Typed,
    OutMin,
    OutMax,
}

impl End {
    pub(crate) fn phrase(self) -> &'static str {
        match self {
            End::Typed => "",
            End::OutMin => "at the Out-min end",
            End::OutMax => "at the Out-max end",
        }
    }

    /// " at the Out-min end", or nothing for a typed number.
    fn suffix(self) -> String {
        match self {
            End::Typed => String::new(),
            End::OutMin | End::OutMax => format!(" {}", self.phrase()),
        }
    }

    /// "At the Out-min end, ", or nothing for a typed number.
    fn prefix(self) -> String {
        match self {
            End::Typed => String::new(),
            End::OutMin => "at the Out-min end, ".to_string(),
            End::OutMax => "at the Out-max end, ".to_string(),
        }
    }
}

/// The ends a phase has a number at: one for a typed number, both of a driving ramp's Out range.
pub(crate) fn ends_of(phase: &LatticePhase) -> Vec<End> {
    match phase.param() {
        ParamRef::Field(_) => vec![End::OutMin, End::OutMax],
        ParamRef::Literal(_) | ParamRef::Named(_) => vec![End::Typed],
    }
}

/// A density as read at one end: a fraction, or a solid typed at one or more -- which is no
/// lattice, and is not read through a fit made at densities a lattice has (§6.5).
#[derive(Copy, Clone, Debug, PartialEq)]
enum Rho {
    Value(f64),
    FullySolid,
}

/// One end's number, the kernel's phase at it, and what it builds.
#[derive(Copy, Clone, Debug)]
struct EndState {
    end: End,
    number: f64,
    phase: Phase,
    density: RelativeDensity,
}

const CAVEAT: &str = "An estimate from published fits for this cell shape — not a property of your \
                      part. A printed part's stiffness and strength depend on its material, \
                      machine, build direction and finish more than on which of these sources \
                      is right; FE and theory describe the cells as drawn, and tests of printed \
                      parts read lower. Test before relying on it.";

const LEDGER: &str = "None of the sources read for this table gives one (ledger: \
                      the lattice property research ledger).";

// ---------------------------------------------------------------------------------- the readout

/// The readout for lattice `id`: its density, then its stiffness and strength as published fits
/// read at that density.
///
/// `in_flight` is the block as the panel is editing it this frame, so the readout follows a
/// dragged number in the same frame rather than the next; the console passes `None` and the
/// block is read from `graph`. Empty for a node that is not a lattice.
pub(crate) fn lattice_readout(
    doc: &Document,
    graph: &Graph,
    id: NodeId,
    in_flight: Option<&Node>,
) -> Vec<ReadoutLine> {
    let Some(Node::Lattice { cell, map, phase, .. }) = in_flight.or_else(|| graph.node(id)) else {
        return Vec::new();
    };
    let kind = CellKind::from(*cell);
    let banding = phase.banding();
    let what = what_it_is(*cell, banding);
    let numbers = match numbers_at_ends(doc, graph, phase) {
        Ok(numbers) => numbers,
        Err(why) => return vec![not_estimated(vec![why], Vec::new(), String::new())],
    };
    // **The floor is a fact of the density, whatever the map** (ruling 11), so it is said even
    // where nothing else is.
    let floor_warnings: Vec<String> = match (kind, phase) {
        (CellKind::Tpms(tpms), LatticePhase::Solid { .. }) => numbers
            .iter()
            .filter_map(|(end, typed)| {
                solid_floor_note(tpms, *typed).map(|note| format!("{}{note}", end.prefix()))
            })
            .collect(),
        _ => Vec::new(),
    };
    // **No wildcard, on purpose** (ruling 16). Item 91 adds a graded map; this stops compiling
    // until its arm is written, and the arm is decided already: "the cells change shape along
    // the grade, and every source measured cubic cells".
    let cell_size = match map {
        CellMap::Rectangular { cell } => cell,
        CellMap::Cylindrical { .. } => return vec![curved("cylindrical", floor_warnings)],
        CellMap::Spherical { .. } => return vec![curved("spherical", floor_warnings)],
        CellMap::Conformal { .. } => return vec![curved("conformal", floor_warnings)],
    };
    let Some(cell_mm) = resolve(doc, cell_size).filter(|c| *c > 0.0) else {
        return vec![not_estimated(
            vec!["the cell size does not resolve".to_string()],
            Vec::new(),
            String::new(),
        )];
    };
    let states = match states_at(doc, phase, kind, cell_mm, &numbers) {
        Ok(states) => states,
        Err(why) => return vec![not_estimated(vec![why], Vec::new(), String::new())],
    };

    let mut lines = density_lines(kind, &states);
    let blocked: Vec<String> = states.iter().filter_map(|s| blocked_reason(kind, &what, s)).collect();
    if !blocked.is_empty() {
        let mut hover = density_hover_for(kind, &states);
        // Where no row describes the cell at all, say so here too, and name the one found and
        // not used: a Voronoi cell has no density to read, and also no row (ledger row 46).
        let classes = [PropertyClass::Modulus, PropertyClass::Strength];
        if classes.iter().all(|c| ranked_rows(kind, banding, *c).is_empty()) {
            if !hover.is_empty() {
                hover.push_str("\n\n");
            }
            hover.push_str(&no_rows_hover(kind, banding, &classes));
        }
        lines.push(not_estimated(blocked, Vec::new(), hover));
        return lines;
    }
    let rhos: Vec<(End, Rho)> = states
        .iter()
        .map(|s| {
            let rho = s.density.value().expect("a blocked end returned above");
            (s.end, if rho >= 0.9995 { Rho::FullySolid } else { Rho::Value(rho) })
        })
        .collect();
    if rhos.iter().all(|(_, rho)| *rho == Rho::FullySolid) {
        lines.push(ReadoutLine::bare(
            "Properties:",
            "fully solid — not a lattice, so the solid's own modulus and strength apply"
                .to_string(),
            Tone::Weak,
            "At a density of one there are no cells, and no fit made on lattices is read there."
                .to_string(),
        ));
        return lines;
    }

    // The density at the middle of a ramp: every ramp shape passes through half way there, so
    // the phase's number at the middle is the mean of the two ends.
    let middle = match states.as_slice() {
        [a, b] => {
            let number = 0.5 * (a.number + b.number);
            phase_of(phase, |_| Some(number), |r| resolve(doc, r))
                .ok()
                .and_then(|p| uniform_density(kind, p, cell_mm).value())
        }
        _ => None,
    };

    let downstream = downstream_of(doc, graph, id);
    let mut extra: Vec<String> = Vec::new();
    if let Some(name) = downstream.joined_by.first() {
        extra.push(format!(
            "'{name}' downstream clips, places or joins these cells. Inside, they are as this \
             block makes them; where two operands meet the material is neither's, and no per-cell \
             figure describes the joins."
        ));
    }
    let word = number_word(banding).0;
    let mut modulus_extra = extra.clone();
    // A honeycomb's one row is in-plane, and the hover says what that leaves out (ruling 7), and
    // what the fit reads at the other density it might have been made against -- at each end
    // of a ramp as at a typed number.
    if let CellKind::Prism(_) = kind {
        let default = ranked_rows(kind, banding, PropertyClass::Modulus).first().copied();
        for state in &states {
            let Some(rho) = state.density.value().filter(|rho| *rho < 0.9995) else { continue };
            modulus_extra.push(sentence(
                state.end,
                &format!(
                    "out of plane a honeycomb is about ρ times the solid — here ≈ {}% — which no \
                     row describes.",
                    pct(rho)
                ),
            ));
            let thin = 2.0 * state.number / cell_mm;
            if let Some(default) = default
                && let (Some(exact), Some(thin_value)) =
                    (default.read(rho).value, default.read(thin).value)
            {
                modulus_extra.push(sentence(
                    state.end,
                    &format!(
                        "read at the thin-wall density 2t/w = {}, {} gives ≈ {}%, {:.2}× the \
                         figure shown at the exact density {}: which of the two the source \
                         fitted against is not stated.",
                        fmt_rho(thin),
                        default.source.short,
                        pct(thin_value),
                        thin_value / exact,
                        fmt_rho(rho)
                    ),
                ));
            }
        }
    }
    let stiffness = class_line(PropertyClass::Modulus, kind, banding, &rhos, middle, word, &modulus_extra);
    let strength = class_line(PropertyClass::Strength, kind, banding, &rhos, middle, word, &extra);
    match (stiffness, strength) {
        (None, None) => lines.push(no_fit_line("Properties:", kind, banding, &what, &[
            PropertyClass::Modulus,
            PropertyClass::Strength,
        ])),
        (stiffness, strength) => {
            lines.push(stiffness.unwrap_or_else(|| {
                no_fit_line("Stiffness:", kind, banding, &what, &[PropertyClass::Modulus])
            }));
            lines.push(strength.unwrap_or_else(|| {
                no_fit_line("Strength:", kind, banding, &what, &[PropertyClass::Strength])
            }));
        }
    }
    if let Some(name) = downstream.reshaped_by.first() {
        let more = match downstream.reshaped_by.len() {
            1 => String::new(),
            n => format!(" (and {} more)", n - 1),
        };
        lines.push(ReadoutLine::bare(
            "",
            format!("for the cells as this block makes them — '{name}'{more} downstream changes them"),
            Tone::Weak,
            "A block that moves the lattice's boundary changes how much of each cell is \
             material, which every figure above is read from: an offset of 0.2 mm grows a 1 mm \
             wall to 1.4, about 40% more material. The figures are for the lattice as this block \
             makes it."
                .to_string(),
        ));
    }
    lines
}

/// "Properties: not estimated — …", with each reason.
fn not_estimated(reasons: Vec<String>, warn: Vec<String>, hover: String) -> ReadoutLine {
    ReadoutLine {
        warn,
        ..ReadoutLine::bare(
            "Properties:",
            format!("not estimated — {}", reasons.join("; ")),
            Tone::Weak,
            hover,
        )
    }
}

/// The one line a curved map gets (ruling 9).
fn curved(map: &str, warn: Vec<String>) -> ReadoutLine {
    not_estimated(
        vec![format!(
            "on a {map} map the cells are not cubes, and every source measured cubic cells"
        )],
        warn,
        "A density does not change when a map stretches the cells, so a reading keyed by \
         density would quote a cubic cell's number for any stretch alike. Use a rectangular map \
         to read one."
            .to_string(),
    )
}

/// "Stiffness: no published fit for a Neovius sheet", and the hover naming any row that was
/// found and not used (§6.3). Never a neighbour's row.
fn no_fit_line(
    label: &'static str,
    kind: CellKind,
    banding: Banding,
    what: &str,
    classes: &[PropertyClass],
) -> ReadoutLine {
    ReadoutLine::bare(
        label,
        format!("no published fit for {what}"),
        Tone::Weak,
        no_rows_hover(kind, banding, classes),
    )
}

/// That no source read gives a row for this cell and phase, and each one found and not used,
/// with why (`EXCLUDED`, and the ledger).
fn no_rows_hover(kind: CellKind, banding: Banding, classes: &[PropertyClass]) -> String {
    let mut hover = LEDGER.to_string();
    for class in classes {
        for excluded in excluded_for(kind, banding, *class) {
            hover.push_str(&format!(
                "\nOne was found and not used: {} — {}",
                excluded.source, excluded.reason
            ));
        }
    }
    hover
}

/// The phase's own number at each end, by name. `Err` says why nothing can be read; a number
/// that does not resolve is never read as nought.
fn numbers_at_ends(
    doc: &Document,
    graph: &Graph,
    phase: &LatticePhase,
) -> Result<Vec<(End, f64)>, String> {
    let word = number_word(phase.banding()).0;
    match phase.param() {
        ParamRef::Field(ramp) => match graph.node(*ramp) {
            Some(Node::Ramp { out_min, out_max, .. }) => {
                let low = resolve(doc, out_min)
                    .ok_or_else(|| format!("the ramp's Out-min {word} does not resolve"))?;
                let high = resolve(doc, out_max)
                    .ok_or_else(|| format!("the ramp's Out-max {word} does not resolve"))?;
                Ok(vec![(End::OutMin, low), (End::OutMax, high)])
            }
            _ => Err(format!("the {word} is driven by a block that is not a ramp")),
        },
        own => resolve(doc, own)
            .map(|v| vec![(End::Typed, v)])
            .ok_or_else(|| format!("the {word} does not resolve")),
    }
}

/// Each end's phase, through the compile's own conversion, and what it builds.
fn states_at(
    doc: &Document,
    phase: &LatticePhase,
    kind: CellKind,
    cell_mm: f64,
    numbers: &[(End, f64)],
) -> Result<Vec<EndState>, String> {
    numbers
        .iter()
        .map(|&(end, number)| {
            let at = phase_of(phase, |_| Some(number), |r| resolve(doc, r))
                .map_err(|Unresolved { which }| format!("the {which} does not resolve"))?;
            Ok(EndState { end, number, phase: at, density: uniform_density(kind, at, cell_mm) })
        })
        .collect()
}

/// Why an end has no density a fit may be read at: separate pieces, or none quoted.
fn blocked_reason(kind: CellKind, what: &str, state: &EndState) -> Option<String> {
    match state.density {
        RelativeDensity::Tabled(_) | RelativeDensity::Exact(_) => None,
        RelativeDensity::Disconnected { built, joins_above } => {
            let typed_floor = match kind {
                CellKind::Tpms(tpms) => format!(
                    " ({:.3} typed)",
                    tpms.density_for_isovalue(tpms.solid_floor().0)
                ),
                _ => String::new(),
            };
            Some(format!(
                "{}at {} {what} is separate pieces, not one lattice; it joins up above about {} \
                 built{typed_floor}",
                state.end.prefix(),
                fmt_rho(built),
                fmt_rho(joins_above),
            ))
        }
        RelativeDensity::Unavailable(reason) => Some(format!("{}{reason}", state.end.prefix())),
    }
}

/// The density lines: a sheet's or a strut's fill, weak; a solid's built density where it is
/// not the number typed, warning-coloured (ruling 10).
fn density_lines(kind: CellKind, states: &[EndState]) -> Vec<ReadoutLine> {
    let hover = density_hover_for(kind, states);
    let mut lines = Vec::new();
    let mut quoted: Vec<String> = Vec::new();
    let mut method = "";
    for state in states {
        match state.phase {
            Phase::Solid { density: typed } => {
                let built = match state.density {
                    RelativeDensity::Tabled(b) => b,
                    RelativeDensity::Disconnected { built, .. } => built,
                    RelativeDensity::Exact(_) | RelativeDensity::Unavailable(_) => continue,
                };
                if typed > 0.0 && ((built - typed) / typed).abs() > 0.01 {
                    lines.push(ReadoutLine::bare(
                        "Density built:",
                        format!(
                            "{built:.3}{} — typed {}: the isovalue table is linear between tenths \
                             and the fill is not",
                            state.end.suffix(),
                            fmt_rho(typed)
                        ),
                        Tone::Warn,
                        hover.clone(),
                    ));
                }
            }
            Phase::Sheet { .. } | Phase::Strut { .. } => match state.density {
                RelativeDensity::Tabled(rho) => {
                    method = "by a table measured on this cell";
                    quoted.push(format!("{}{}", fmt_rho(rho), state.end.suffix()));
                }
                RelativeDensity::Exact(rho) => {
                    method = "exact for this cell";
                    quoted.push(format!("{}{}", fmt_rho(rho), state.end.suffix()));
                }
                RelativeDensity::Disconnected { .. } | RelativeDensity::Unavailable(_) => {}
            },
        }
    }
    if !quoted.is_empty() {
        let from = match states[0].phase {
            Phase::Strut { .. } => "the beam radius",
            Phase::Sheet { .. } | Phase::Solid { .. } => "the wall",
        };
        lines.push(ReadoutLine::bare(
            "Density:",
            format!("{} — from {from}, {method}", quoted.join(", ")),
            Tone::Weak,
            hover,
        ));
    }
    lines
}

/// How the density was come by, and how far it can be trusted, with the example that measured
/// it (§6.4).
fn density_hover_for(kind: CellKind, states: &[EndState]) -> String {
    let Some(first) = states.first() else { return String::new() };
    match (kind, first.phase) {
        (CellKind::Tpms(tpms), Phase::Solid { .. }) => {
            let (iso, joins_above) = tpms.solid_floor();
            let mut out = String::new();
            for state in states {
                if let Phase::Solid { density } = state.phase {
                    out.push_str(&format!(
                        "Typed {}{} builds {:.3}. ",
                        fmt_rho(density),
                        state.end.suffix(),
                        tpms.solid_density_built(density)
                    ));
                }
            }
            out.push_str(&format!(
                "What a typed density fills, from a table tpms_profile.rs measured on this cell's \
                 own field (TpmsKind::solid_density_built): within 0.05% wherever at least 2% of \
                 the cell is filled, and within 0.003% of the cell below that. A {} solid is one \
                 piece only above about {} built ({:.3} typed).",
                tpms.label(),
                fmt_rho(joins_above),
                tpms.density_for_isovalue(iso)
            ));
            out
        }
        (CellKind::Tpms(_), _) => "The fraction of each cell the wall fills, from a table \
             tpms_profile.rs measured on this cell's own field (TpmsKind::sheet_density): within \
             0.1% of what the field builds at every row. It is the wall as built — nominal \
             thickness and all — not an ideal one."
            .to_string(),
        (CellKind::Beam(beam), _) => match beam {
            fieldform_field::BeamKind::Cubic => "3πr² − 8√2r³, with r the radius over the cell: \
                 three cylinders less what they share at a node. Exact up to half a cell, where \
                 the struts meet."
                .to_string(),
            _ => format!(
                "The fraction of each cell within the beam radius of a strut, from a table \
                 beam_profile.rs measured on this cell (BeamKind::strut_density): within 0.4% at \
                 the thinnest row and 0.1% from a radius of 2% of the cell. Past {:.3} of the \
                 cell the struts meet and nothing is read.",
                beam.merge_radius()
            ),
        },
        (CellKind::Prism(_), _) => "1 − (1 − t/w)², with w the width across flats: what is \
             left open is a hexagon w − t across. Exact."
            .to_string(),
        (CellKind::Stochastic { .. }, _) => String::new(),
    }
}

// ---------------------------------------------------------------------------- the class lines

fn class_label(class: PropertyClass) -> &'static str {
    match class {
        PropertyClass::Modulus => "Stiffness:",
        PropertyClass::Strength => "Strength:",
    }
}

/// Which rows a range is taken across: one direction and, for strength, one definition.
///
/// **A build direction is grouped with a direction not stated**, because as a claim about the
/// cell it is the same: how the cell sat in the specimen is what neither says.
fn direction_group(direction: LoadDirection) -> u8 {
    match direction {
        LoadDirection::Axis100 => 0,
        LoadDirection::Diagonal111 => 1,
        LoadDirection::InPlane => 2,
        LoadDirection::NotStated | LoadDirection::BuildDirection => 3,
    }
}

fn group_phrase(group: u8) -> &'static str {
    match group {
        0 => LoadDirection::Axis100.phrase(),
        1 => LoadDirection::Diagonal111.phrase(),
        2 => LoadDirection::InPlane.phrase(),
        _ => LoadDirection::NotStated.phrase(),
    }
}

fn strength_kind(row: &PropertyRow) -> Option<StrengthKind> {
    match row.property {
        Property::Modulus => None,
        Property::Strength(kind) => Some(kind),
    }
}

fn group_key(row: &PropertyRow) -> (u8, Option<StrengthKind>) {
    (direction_group(row.direction), strength_kind(row))
}

/// The direction on the main line, where the source states one -- and only there (ruling 7).
fn direction_on_main_line(direction: LoadDirection) -> String {
    match direction {
        LoadDirection::Axis100 | LoadDirection::Diagonal111 | LoadDirection::InPlane => {
            format!(", {}", direction.phrase())
        }
        LoadDirection::NotStated | LoadDirection::BuildDirection => String::new(),
    }
}

/// "Lu 2020 · FE · over the model's own solid · direction not stated · estimate".
fn detail_for(row: &PropertyRow) -> String {
    let data = match row.range {
        DensityRange::Single(_) => format!("{}, one density", row.data.phrase()),
        DensityRange::Span { .. } | DensityRange::NotStated => row.data.phrase().to_string(),
    };
    format!(
        "{} · {data} · {} · {} · estimate",
        row.source.short,
        row.solid_basis.phrase(),
        row.direction.phrase()
    )
}

fn standing_warning(standing: Standing) -> Option<String> {
    match standing {
        Standing::Inside | Standing::Refused(_) => None,
        Standing::Outside { lo, hi } => Some(format!(
            "outside the source's density range, {}",
            range_text(lo, hi)
        )),
        Standing::LowestNotStated { surely_from } => Some(format!(
            "the source does not state its lowest density; below {} this may be an extrapolation",
            fmt_rho(surely_from)
        )),
        Standing::OnePoint { at } => Some(format!(
            "measured at one density ({}); elsewhere the exponent is assumed",
            fmt_rho(at)
        )),
        Standing::RangeNotStated => Some(
            "the source's density range is not stated, so whether this is an extrapolation \
             cannot be said"
                .to_string(),
        ),
    }
}

fn range_text(lo: Lower, hi: f64) -> String {
    match lo {
        Lower::At(lo) => format!("{}–{}", fmt_rho(lo), fmt_rho(hi)),
        Lower::NotStated { surely_from } => {
            format!("{}–{} (its lowest not stated)", fmt_rho(surely_from), fmt_rho(hi))
        }
    }
}

fn refusal_text(refusal: Refusal) -> &'static str {
    match refusal {
        Refusal::NoDensity => "there is no density to read it at",
        Refusal::NotPositive => "its fit is at or below nought here",
        Refusal::NearZero => {
            "its fit is so near nought here that a 1% error in the density would move it by \
             more than 10%"
        }
        Refusal::AboveSolidShare => {
            "its fit would give more than the solid's share here, ρ times the solid's own"
        }
    }
}

/// The few words that say a reading is past or without its source's range: what a list of
/// sources or a target's number carries beside each name. `None` inside a range.
fn standing_brief(standing: Standing) -> Option<String> {
    match standing {
        Standing::Inside | Standing::Refused(_) => None,
        Standing::Outside { lo, hi } => Some(format!("outside its range, {}", range_text(lo, hi))),
        Standing::LowestNotStated { surely_from } => Some(format!(
            "below where its data surely reach, {}",
            fmt_rho(surely_from)
        )),
        Standing::OnePoint { at } => Some(format!("measured at one density, {}", fmt_rho(at))),
        Standing::RangeNotStated => Some("its range not stated".to_string()),
    }
}

/// "of the 5 sources in the range, read outside its own range: Naghavi 2022 (0.28–0.46); no
/// range stated: Polley 2022 via Distefano 2023, Sun 2022 via Distefano 2023": every source a
/// figure on the line rests on that is read past its own range or states none, by name.
///
/// **Ruling 8 for every row a number comes from**, not only the one shown: the range a line
/// leads with is as much a reading as the default's value, and its top was once a fit read at
/// three times past its range with nothing on the line to say so. `None` where all are inside.
fn sources_named(of: &str, rows: &[(&'static PropertyRow, Reading)]) -> Option<String> {
    // In the order a reader weighs them: an extrapolation first, then what cannot be checked.
    let mut outside: Vec<String> = Vec::new();
    let mut below: Vec<String> = Vec::new();
    let mut one_point: Vec<String> = Vec::new();
    let mut not_stated: Vec<String> = Vec::new();
    for (row, reading) in rows {
        let name = row.source.short;
        match reading.standing {
            Standing::Inside | Standing::Refused(_) => {}
            Standing::Outside { lo, hi } => outside.push(format!("{name} ({})", range_text(lo, hi))),
            Standing::LowestNotStated { surely_from } => {
                below.push(format!("{name} (from {})", fmt_rho(surely_from)))
            }
            Standing::OnePoint { at } => one_point.push(format!("{name} ({})", fmt_rho(at))),
            Standing::RangeNotStated => not_stated.push(name.to_string()),
        }
    }
    let plural = |names: &[String], one: &str, many: &str| match names.len() {
        1 => one.to_string(),
        _ => many.to_string(),
    };
    let mut parts: Vec<String> = Vec::new();
    if !outside.is_empty() {
        let head = plural(&outside, "read outside its own range", "read outside their own ranges");
        parts.push(format!("{head}: {}", outside.join(", ")));
    }
    if !below.is_empty() {
        let head =
            plural(&below, "read below where its data surely reach", "read below where their data surely reach");
        parts.push(format!("{head}: {}", below.join(", ")));
    }
    if !one_point.is_empty() {
        parts.push(format!("measured at one density: {}", one_point.join(", ")));
    }
    if !not_stated.is_empty() {
        parts.push(format!("no range stated: {}", not_stated.join(", ")));
    }
    (!parts.is_empty()).then(|| format!("{of}, {}", parts.join("; ")))
}

/// The line for one class, typed or driven. `None` where no row describes this cell and phase.
fn class_line(
    class: PropertyClass,
    kind: CellKind,
    banding: Banding,
    rhos: &[(End, Rho)],
    middle: Option<f64>,
    word: &str,
    extra_hover: &[String],
) -> Option<ReadoutLine> {
    let ranked = ranked_rows(kind, banding, class);
    if ranked.is_empty() {
        return None;
    }
    match rhos {
        [(End::Typed, Rho::Value(rho))] => Some(typed_line(class, &ranked, *rho, extra_hover)),
        _ => Some(driven_line(class, &ranked, rhos, middle, word, extra_hover)),
    }
}

/// One class read at one density, before it is put on a line: everything ruling 5 asks of a
/// reading, made the same way for a typed number and for each end of a ramp.
struct AtDensity {
    /// The range across sources of one definition and direction, then the shown row's value
    /// and its axis partner; or that none gives a value.
    text: String,
    /// The row whose value is shown: the default, or the best-ranked one that gives a value
    /// where the default's fit does not.
    shown: Option<&'static PropertyRow>,
    warn: Vec<String>,
    /// Rows of another strength definition or direction, counted and ranged.
    others: Option<String>,
    /// Why nothing is shown, where nothing is: "Lu 2020: its fit is at or below nought here."
    refused: Option<String>,
}

/// The rows of a key read at `rho`. `ranked` is the key's rows best-ranked first, never empty.
fn read_at(ranked: &[&'static PropertyRow], rho: f64) -> AtDensity {
    let default = ranked[0];
    let readings: Vec<(&'static PropertyRow, Reading)> =
        ranked.iter().map(|row| (*row, row.read(rho))).collect();
    // **A refused default falls through to the best-ranked row that gives a value**, and says
    // so; a refused value is never shown.
    let Some((shown, reading)) = readings.iter().find(|(_, r)| r.value.is_some()).copied() else {
        let text = match readings.len() {
            1 => format!("— {}'s fit gives no meaningful value at this density", default.source.short),
            n => format!("— none of the {n} sources' fits gives a meaningful value at this density"),
        };
        let why = match readings[0].1.standing {
            Standing::Refused(refusal) => refusal_text(refusal),
            _ => "",
        };
        return AtDensity {
            text,
            shown: None,
            warn: Vec::new(),
            others: None,
            refused: Some(format!("{}: {why}.", default.source.short)),
        };
    };
    let value = reading.value.expect("a shown row gives a value");
    let key = group_key(shown);
    let partner = other_axis_row(shown).and_then(|p| {
        let at = p.read(rho);
        at.value.map(|v| (p, at, v))
    });
    let group: Vec<(&'static PropertyRow, Reading)> = readings
        .iter()
        .filter(|(row, reading)| group_key(row) == key && reading.value.is_some())
        .copied()
        .collect();
    let phrase = shown.property.phrase();
    let direction = direction_on_main_line(shown.direction);
    let mut text = if group.len() >= 2 {
        let (lo, hi) = min_max(group.iter().filter_map(|(_, r)| r.value));
        let differ = if group.iter().any(|(row, _)| row.solid_basis != shown.solid_basis) {
            ", which divide by different solids"
        } else {
            ""
        };
        let span = if pct(lo) == pct(hi) {
            format!("≈ {}%", pct(lo))
        } else {
            format!("{}–{}%", pct(lo), pct(hi))
        };
        format!(
            "{span} {phrase}{direction}, across {} sources{differ} · {} ≈ {}%",
            group.len(),
            shown.source.short,
            pct(value)
        )
    } else {
        format!("≈ {}% {phrase}{direction} · {}", pct(value), shown.source.short)
    };
    if let Some((partner, _, partner_value)) = partner {
        text.push_str(&format!(
            " · {}% {} ({}, {})",
            pct(partner_value),
            partner.direction.phrase(),
            partner.source.short,
            partner.data.phrase()
        ));
    }
    if shown.id != default.id {
        text.push_str(&format!(
            " ({}'s fit, the default here, gives no meaningful value at this density)",
            default.source.short
        ));
    }
    let mut warn: Vec<String> = standing_warning(reading.standing).into_iter().collect();
    if let Some((partner, at, _)) = partner
        && let Some(w) = standing_warning(at.standing)
    {
        warn.push(format!("{}: {w}", partner.source.short));
    }
    let in_range: Vec<(&'static PropertyRow, Reading)> =
        group.iter().filter(|(row, _)| row.id != shown.id).copied().collect();
    warn.extend(sources_named(&format!("of the {} sources in the range", group.len()), &in_range));

    // The others: another strength definition or another direction, counted and ranged on their
    // own line rather than dropped, which hid most of the disagreement (ruling 5).
    let mut others: Vec<(String, Vec<f64>)> = Vec::new();
    let mut counted: Vec<(&'static PropertyRow, Reading)> = Vec::new();
    let mut silent = 0usize;
    for (row, reading) in &readings {
        let is_partner = partner.is_some_and(|(p, _, _)| p.id == row.id);
        if row.id == shown.id || row.id == default.id || is_partner {
            continue;
        }
        let Some(v) = reading.value else {
            silent += 1;
            continue;
        };
        if group_key(row) == key {
            continue;
        }
        let why = if strength_kind(row) != strength_kind(shown) {
            "strength defined differently".to_string()
        } else {
            group_phrase(direction_group(row.direction)).to_string()
        };
        match others.iter_mut().find(|(reason, _)| *reason == why) {
            Some((_, values)) => values.push(v),
            None => others.push((why, vec![v])),
        }
        counted.push((*row, *reading));
    }
    let mut parts: Vec<String> = others
        .iter()
        .map(|(why, values)| {
            let (lo, hi) = min_max(values.iter().copied());
            let span = if pct(lo) == pct(hi) {
                format!("{}%", pct(lo))
            } else {
                format!("{}–{}%", pct(lo), pct(hi))
            };
            let n = values.len();
            format!("{n} other source{}, {why}: {span}", if n == 1 { "" } else { "s" })
        })
        .collect();
    if silent > 0 {
        parts.push(format!(
            "{silent} more give{} no meaningful value at this density",
            if silent == 1 { "s" } else { "" }
        ));
    }
    let of_others = match counted.len() {
        1 => "of the other source".to_string(),
        n => format!("of the {n} other sources"),
    };
    warn.extend(sources_named(&of_others, &counted));
    AtDensity {
        text,
        shown: Some(shown),
        warn,
        others: (!parts.is_empty()).then(|| parts.join("; ")),
        refused: None,
    }
}

/// A typed number's line: the range across sources of one definition and direction, then the
/// default's value; the detail; the warnings; the others counted on their own (ruling 5).
fn typed_line(
    class: PropertyClass,
    ranked: &[&'static PropertyRow],
    rho: f64,
    extra_hover: &[String],
) -> ReadoutLine {
    let at = read_at(ranked, rho);
    let shown: Vec<&str> = at.shown.map(|r| r.id).into_iter().collect();
    let mut hover = rows_hover(ranked, &[(End::Typed, Rho::Value(rho))], &shown, extra_hover);
    if let Some(refused) = &at.refused {
        hover = format!("{refused}\n\n{hover}");
    }
    ReadoutLine {
        label: class_label(class),
        text: at.text,
        tone: if at.shown.is_some() { Tone::Plain } else { Tone::Weak },
        detail: at.shown.map(detail_for),
        others: at.others.into_iter().collect(),
        note: None,
        warn: at.warn,
        hover,
    }
}

/// A driven number's line: each Out end read by name, exactly as a typed number is read -- its
/// range across sources, its axis partner, its fall-through, its others and its warnings --
/// in the order the user chose; and what the middle of the ramp reads, since a ramp even in the
/// wall is not even in stiffness (§6.5).
fn driven_line(
    class: PropertyClass,
    ranked: &[&'static PropertyRow],
    rhos: &[(End, Rho)],
    middle: Option<f64>,
    word: &str,
    extra_hover: &[String],
) -> ReadoutLine {
    let ats: Vec<(End, Option<AtDensity>)> = rhos
        .iter()
        .map(|(end, rho)| match rho {
            Rho::FullySolid => (*end, None),
            Rho::Value(r) => (*end, Some(read_at(ranked, *r))),
        })
        .collect();
    let text = ats
        .iter()
        .map(|(end, at)| match at {
            None => format!("{}fully solid, 100% of the solid's own", end.prefix()),
            Some(at) => format!("{}{}", end.prefix(), at.text),
        })
        .collect::<Vec<_>>()
        .join("; ");
    let shown: Vec<&'static PropertyRow> =
        ats.iter().filter_map(|(_, at)| at.as_ref().and_then(|a| a.shown)).collect();
    let detail = match shown.as_slice() {
        [] => None,
        [first, rest @ ..] if rest.iter().all(|r| r.id == first.id) => Some(detail_for(first)),
        _ => Some(
            ats.iter()
                .filter_map(|(end, at)| {
                    let row = at.as_ref()?.shown?;
                    Some(format!("{}: {}", end.phrase(), detail_for(row)))
                })
                .collect::<Vec<_>>()
                .join("; "),
        ),
    };
    let mut warn = Vec::new();
    let mut others = Vec::new();
    let mut refused = Vec::new();
    for (end, at) in &ats {
        let Some(at) = at else { continue };
        warn.extend(at.warn.iter().map(|w| format!("{}: {w}", end.phrase())));
        others.extend(at.others.iter().map(|o| format!("{}: {o}", end.phrase())));
        refused.extend(at.refused.iter().map(|r| sentence(*end, r)));
    }
    let note = middle_note(class, ranked, rhos, middle, word);
    let shown_ids: Vec<&str> = shown.iter().map(|r| r.id).collect();
    let mut hover = rows_hover(ranked, rhos, &shown_ids, extra_hover);
    if !refused.is_empty() {
        hover = format!("{}\n\n{hover}", refused.join("\n"));
    }
    ReadoutLine {
        label: class_label(class),
        text,
        tone: if shown.is_empty() { Tone::Weak } else { Tone::Plain },
        detail,
        others,
        note,
        warn,
        hover,
    }
}

/// "at the middle of the ramp ≈ 11% by Nguyen-Xuan 2023, not the 13% midway between the ends":
/// every ramp shape passes half way at its middle, so the phase's number there is the mean of
/// the ends, and reading it there is what shows the ramp is not even in stiffness. One row reads
/// all three -- the best-ranked that gives a value at both ends and the middle -- since a
/// comparison across sources would say more about the sources than the ramp.
fn middle_note(
    class: PropertyClass,
    ranked: &[&'static PropertyRow],
    rhos: &[(End, Rho)],
    middle: Option<f64>,
    word: &str,
) -> Option<String> {
    let ([(_, Rho::Value(a)), (_, Rho::Value(b))], Some(m)) = (rhos, middle) else {
        return None;
    };
    ranked.iter().find_map(|row| {
        let (va, vb, vm) = (row.read(*a).value?, row.read(*b).value?, row.read(m).value?);
        if pct(va) == pct(vb) {
            return None;
        }
        let what = match class {
            PropertyClass::Modulus => "stiffness",
            PropertyClass::Strength => "strength",
        };
        Some(format!(
            "at the middle of the ramp ≈ {}% by {}, not the {}% midway between the ends: the ramp \
             is even in the {word}, not in {what}",
            pct(vm),
            row.source.short,
            pct(0.5 * (va + vb))
        ))
    })
}

/// A sentence about one end, capitalised: "At the Out-min end, out of plane …", or the rest
/// alone, capitalised, for a typed number.
fn sentence(end: End, rest: &str) -> String {
    let whole = format!("{}{rest}", end.prefix());
    let mut chars = whole.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => whole,
    }
}

/// Every row of the key, best-ranked first, read at each end, with what qualifies it (§6.4).
fn rows_hover(
    ranked: &[&'static PropertyRow],
    rhos: &[(End, Rho)],
    shown: &[&str],
    extra: &[String],
) -> String {
    let mut out = String::new();
    for row in ranked {
        let marker = if shown.contains(&row.id) { " (shown)" } else { "" };
        let at: Vec<String> = rhos
            .iter()
            .map(|(end, rho)| {
                let value = match rho {
                    Rho::FullySolid => "fully solid".to_string(),
                    Rho::Value(r) => {
                        let reading = row.read(*r);
                        match (reading.value, reading.standing) {
                            (Some(v), standing) => format!("≈ {}%{}", pct(v), standing_note(standing)),
                            (None, Standing::Refused(refusal)) => {
                                format!("no value — {}", refusal_text(refusal))
                            }
                            (None, _) => "no value".to_string(),
                        }
                    }
                };
                format!("{value}{}", end.suffix())
            })
            .collect();
        out.push_str(&format!("• {}{marker}: {}\n", row.source.short, at.join(", ")));
        out.push_str(&format!("   {} — {}\n", row.source.citation, row.source.locator));
        out.push_str(&format!(
            "   {}; {}; {}; {}\n",
            match row.data {
                DataKind::Fe if row.id.starts_with("cellucci") => "FE (beam elements)",
                other => other.phrase(),
            },
            exponent_phrase(row.exponent),
            range_phrase(row.range),
            row.direction.phrase()
        ));
        out.push_str(&format!("   {}; {}\n", row.material, basis_phrase(row)));
        match row.density_basis {
            DensityBasis::Design => {}
            DensityBasis::Measured => {
                out.push_str("   fitted on measured density; read here at the density built\n")
            }
            DensityBasis::NotStated => {
                out.push_str("   which density the source fitted against is not stated\n")
            }
        }
    }
    for line in extra {
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out.push_str(CAVEAT);
    out
}

fn standing_note(standing: Standing) -> &'static str {
    match standing {
        Standing::Inside => " (inside its range)",
        Standing::Outside { .. } => " (outside its range)",
        Standing::LowestNotStated { .. } => " (below where its data surely reach)",
        Standing::OnePoint { .. } => " (away from its one density)",
        Standing::RangeNotStated => " (its range not stated)",
        Standing::Refused(_) => "",
    }
}

fn exponent_phrase(exponent: Exponent) -> &'static str {
    match exponent {
        Exponent::Fitted => "exponent fitted",
        Exponent::Theory => "exponent from theory",
        Exponent::Assumed => "exponent assumed by the authors",
        Exponent::NotStated => "whether the exponent was fitted is not stated",
    }
}

fn range_phrase(range: DensityRange) -> String {
    match range {
        DensityRange::Span { lo, hi } => format!("range {}", range_text(lo, hi)),
        DensityRange::Single(at) => format!("one density, {}", fmt_rho(at)),
        DensityRange::NotStated => "range not stated".to_string(),
    }
}

/// What the ratio was divided by, with the value where it is known or implied (ruling 17).
fn basis_phrase(row: &PropertyRow) -> String {
    match (row.solid_basis, row.solid_value_mpa) {
        (SolidBasis::Implied, Some(mpa)) => format!(
            "over a divisor the review does not state, ≈{}",
            stress(mpa)
        ),
        (basis, Some(mpa)) => format!("{} ({})", basis.phrase(), stress(mpa)),
        (basis, None) => basis.phrase().to_string(),
    }
}

fn stress(mpa: f64) -> String {
    if mpa >= 1000.0 {
        let gpa = mpa / 1000.0;
        if (gpa - gpa.round()).abs() < 0.05 {
            format!("{gpa:.0} GPa")
        } else {
            format!("{gpa:.1} GPa")
        }
    } else {
        format!("{mpa:.0} MPa")
    }
}

// --------------------------------------------------------------------------------- downstream

/// The blocks a lattice's cells pass through on the way to a body.
#[derive(Default)]
struct Downstream {
    /// Those that change the cells (ruling 15), by name.
    reshaped_by: Vec<String>,
    /// Those that clip, place or join them, by name.
    joined_by: Vec<String>,
}

/// Whether `consumer`, taking `from` as an operand, changes the cells a per-cell figure is read
/// from.
///
/// **Exhaustive, with no wildcard**, so a new kind of block must say which it is before it
/// compiles (ruling 15). The slot matters for one kind: an attractor reshapes what it warps,
/// and the block it is drawn toward is untouched.
pub(crate) fn changes_the_cells(consumer: &Node, from: NodeId) -> bool {
    match consumer {
        Node::Offset { .. } | Node::Shell { .. } | Node::Twist { .. } | Node::Morph { .. } => true,
        Node::Attract { input, .. } => input.source() == Some(from),
        // A rigid move and a uniform scale keep every ratio a density is.
        Node::Transform { .. } => false,
        // They clip, place or join: material changes only where two operands meet, which a
        // per-cell figure does not describe, and the hover says that rather than calling them
        // rigid -- a pattern's union can be rounded, and overlapping copies are not one cell.
        Node::Boolean { .. } | Node::Blend { .. } | Node::Pattern { .. } | Node::Mirror { .. } => {
            false
        }
        // It reads the lattice as a number.
        Node::Ramp { .. } => false,
        // It cannot be read, so the safe answer is that it may.
        Node::Unknown(_) => true,
        // No field operand at all; here because the match is exhaustive.
        Node::Cage(_)
        | Node::Mesh(_)
        | Node::Sphere { .. }
        | Node::Cuboid { .. }
        | Node::SkewedBox { .. }
        | Node::Cylinder { .. }
        | Node::Cone { .. }
        | Node::Lattice { .. }
        | Node::Extrude { .. }
        | Node::Revolve { .. }
        | Node::Plane { .. }
        | Node::Point { .. }
        | Node::Torus { .. }
        | Node::Capsule { .. }
        | Node::Pipe { .. }
        | Node::Sweep { .. }
        | Node::Loft { .. } => false,
    }
}

fn joins(consumer: &Node) -> bool {
    matches!(
        consumer,
        Node::Boolean { .. } | Node::Blend { .. } | Node::Pattern { .. } | Node::Mirror { .. }
    )
}

/// Every chain of consumers from the lattice up to the bodies, by what each is **made of**
/// (`Node::operands`): a block merely aimed at the lattice is not reshaping it.
fn downstream_of(doc: &Document, graph: &Graph, id: NodeId) -> Downstream {
    let mut nodes: Vec<(NodeId, &Node)> = graph.nodes().collect();
    nodes.sort_by_key(|(n, _)| n.raw());
    let mut out = Downstream::default();
    let mut seen: HashSet<NodeId> = HashSet::from([id]);
    let mut queue: VecDeque<NodeId> = VecDeque::from([id]);
    while let Some(from) = queue.pop_front() {
        for (consumer_id, consumer) in &nodes {
            if !consumer.operands().contains(&from) {
                continue;
            }
            let name = block_name(doc, *consumer_id, consumer);
            if changes_the_cells(consumer, from) {
                if !out.reshaped_by.contains(&name) {
                    out.reshaped_by.push(name);
                }
            } else if joins(consumer) && !out.joined_by.contains(&name) {
                out.joined_by.push(name);
            }
            if seen.insert(*consumer_id) {
                queue.push_back(*consumer_id);
            }
        }
    }
    out
}

/// A block's given name, or what it is.
fn block_name(doc: &Document, id: NodeId, node: &Node) -> String {
    if let Some(name) = doc.node_meta.get(&id).and_then(|m| m.name.clone()) {
        return name;
    }
    match node {
        Node::Offset { .. } => "Offset",
        Node::Shell { .. } => "Shell",
        Node::Twist { .. } => "Twist",
        Node::Morph { .. } => "Morph",
        Node::Attract { .. } => "Attract",
        Node::Boolean { .. } => "Boolean",
        Node::Blend { .. } => "Blend",
        Node::Pattern { .. } => "Pattern",
        Node::Mirror { .. } => "Mirror",
        Node::Transform { .. } => "Placement",
        Node::Unknown(_) => "Unreadable",
        _ => "block",
    }
    .to_string()
}

// ----------------------------------------------------------------------------- the console

/// Every lattice a body is made of: the body itself if it is one, and each lattice down its
/// operands -- the body a person selects is normally the boolean above the lattice (§8).
pub(crate) fn lattices_within(graph: &Graph, start: NodeId) -> Vec<NodeId> {
    let mut found = Vec::new();
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut stack = vec![start];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = graph.node(id) else { continue };
        if matches!(node, Node::Lattice { .. }) {
            found.push(id);
        }
        // Reversed so the first operand is walked first, and the report reads in the order the
        // block lists its operands.
        stack.extend(node.operands().into_iter().rev());
    }
    found
}

/// What `estimate` prints for the body `start`: a block per lattice inside it, and with a
/// target, what each source would need. `Err` where it holds no lattice.
///
/// Reads, never edits: every function it calls takes the document by reference.
pub(crate) fn estimate_report(
    doc: &Document,
    graph: &Graph,
    start: NodeId,
    label: &dyn Fn(NodeId) -> String,
    target: Option<f64>,
) -> Result<Vec<String>, String> {
    let found = lattices_within(graph, start);
    if found.is_empty() {
        return Err(format!("'{}' contains no lattice", label(start)));
    }
    let mut out = Vec::new();
    for lattice in found {
        let what = match graph.node(lattice) {
            Some(Node::Lattice { cell, phase, .. }) => what_it_is(*cell, phase.banding()),
            _ => String::new(),
        };
        out.push(format!("{} — {what}:", label(lattice)));
        out.extend(console_lines(&lattice_readout(doc, graph, lattice, None)));
        let Some(target) = target else { continue };
        let ends = match graph.node(lattice) {
            Some(Node::Lattice { phase, .. }) => ends_of(phase),
            _ => Vec::new(),
        };
        for end in ends {
            match number_for_target(doc, graph, lattice, None, end, target) {
                Ok(answer) => out.extend(target_lines(&answer, target)),
                Err(why) => out.push(format!(
                    "  Target {}% of the solid's modulus{}: none offered — {why}",
                    typed_percent(target),
                    end.suffix()
                )),
            }
        }
    }
    Ok(out)
}

/// The readout's lines as the console prints them, indented under the lattice's name.
pub(crate) fn console_lines(lines: &[ReadoutLine]) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        out.push(match line.label {
            "" => format!("  {}", line.text),
            label => format!("  {label} {}", line.text),
        });
        if let Some(detail) = &line.detail {
            out.push(format!("      {detail}"));
        }
        for warning in &line.warn {
            out.push(format!("      caution: {warning}"));
        }
        for others in &line.others {
            out.push(format!("      {others}"));
        }
        if let Some(note) = &line.note {
            out.push(format!("      {note}"));
        }
    }
    out
}

/// A target's answer as the console prints it: the default's number, each other source's, and
/// whether Set would write it -- each with its caution where the density it needs is past or
/// without its source's range. The console never writes it.
pub(crate) fn target_lines(answer: &TargetAnswer, target: f64) -> Vec<String> {
    let head = format!(
        "  Target {}% of the solid's modulus{}",
        typed_percent(target),
        answer.end.suffix()
    );
    let mut out = vec![match &answer.default.number {
        Ok(n) => format!(
            "{head} → {} by {} (the default) — estimate",
            number_text(answer.word, *n, answer.unit),
            answer.default.row.source.short
        ),
        Err(why) => format!("{head}: {} — {why}", answer.default.row.source.short),
    }];
    if let (Ok(_), Some(caution)) = (&answer.default.number, &answer.default.caution) {
        out.push(format!("      caution: {} {caution}", answer.default.row.source.short));
    }
    for offer in &answer.others {
        out.push(format!("      {}: {}", source_and_direction(offer.row), offer_text(answer, offer)));
    }
    if let Err(why) = &answer.write
        && answer.default.number.is_ok()
    {
        out.push(format!("      Set would not write it: {why}"));
    }
    out
}

// ------------------------------------------------------------------------------ the panel's

/// Where a typed solid density is separate pieces, the sentence the Density row says on every
/// map (ruling 11); `None` above the floor. Compared on isovalues, as the kernel compares them,
/// so it agrees with `uniform_density` exactly.
pub(crate) fn solid_floor_note(kind: TpmsKind, typed: f64) -> Option<String> {
    if !typed.is_finite() {
        return None;
    }
    let (iso, joins_above) = kind.solid_floor();
    if kind.isovalue_for_density(typed) > iso {
        return None;
    }
    Some(format!(
        "below about {} built ({:.3} typed) a {} solid is separate pieces, not one lattice",
        fmt_rho(joins_above),
        kind.density_for_isovalue(iso),
        kind.label()
    ))
}

/// The Density row's hover, from what the kernel measured rather than from a sentence that was
/// true of three cells (§5.7): "within about 1%" was false below a tenth, and for three of the
/// six even between 0.1 and 0.9.
pub(crate) fn density_hover(kind: TpmsKind, typed: Option<f64>) -> String {
    let off = |t: f64| (kind.solid_density_built(t) - t) / t;
    let worst_middle = (10..=90).map(|i| off(i as f64 / 100.0).abs()).fold(0.0, f64::max);
    let at_005 = kind.solid_density_built(0.05);
    let mut out = format!(
        "Fraction of each cell that is material — 0.4 is 40%. Approximate: the isovalue table is \
         linear between tenths and the fill is not, so typed and built agree at the tenths and \
         differ between them. For a {} that is at most {:.1}% of the number from 0.1 to 0.9, and \
         far more below 0.1: 0.05 builds {at_005:.3}.",
        kind.label(),
        worst_middle * 100.0
    );
    if let Some(t) = typed.filter(|t| t.is_finite() && *t > 0.0) {
        out.push_str(&format!(" This number builds {:.3}.", kind.solid_density_built(t)));
    }
    out
}

// ---------------------------------------------------------------------------------- formatting

fn resolve(doc: &Document, r: &ParamRef) -> Option<f64> {
    r.resolve(&|id| doc.parameter_value(id)).filter(|v| v.is_finite())
}

/// "wall" and " mm", "density" and nothing, "radius" and " mm": `phase_of`'s own names.
fn number_word(banding: Banding) -> (&'static str, &'static str) {
    match banding {
        Banding::Sheet => ("wall", " mm"),
        Banding::Solid => ("density", ""),
        Banding::Strut => ("radius", " mm"),
    }
}

/// "a gyroid sheet", "an octet strut lattice", "a Neovius solid".
fn what_it_is(cell: LatticeCell, banding: Banding) -> String {
    let name = match cell {
        LatticeCell::Gyroid => "gyroid",
        LatticeCell::SchwarzP => "Schwarz P",
        LatticeCell::SchwarzD => "Schwarz D",
        LatticeCell::SchoenIwp => "Schoen IWP",
        LatticeCell::Neovius => "Neovius",
        LatticeCell::FischerKochS => "Fischer-Koch S",
        LatticeCell::Cubic => "cubic",
        LatticeCell::Bcc => "BCC",
        LatticeCell::Fcc => "FCC",
        LatticeCell::Octet => "octet",
        LatticeCell::Diamond => "diamond",
        LatticeCell::Honeycomb => "honeycomb",
        LatticeCell::VoronoiFoam { .. } => "Voronoi foam",
        LatticeCell::VoronoiBeam { .. } => "Voronoi beam",
    };
    let noun = match (cell, banding) {
        (LatticeCell::Honeycomb, _) => name.to_string(),
        (_, Banding::Sheet) => format!("{name} sheet"),
        (_, Banding::Solid) => format!("{name} solid"),
        (_, Banding::Strut) => format!("{name} strut lattice"),
    };
    let article = if noun.starts_with(['a', 'e', 'i', 'o', 'u']) || noun.starts_with("FCC") {
        "an"
    } else {
        "a"
    };
    format!("{article} {noun}")
}

/// A fraction as a percentage to two significant figures (ruling 5): 7.0, 14, 0.36, 130.
pub(crate) fn pct(fraction: f64) -> String {
    sig2(fraction * 100.0)
}

fn sig2(v: f64) -> String {
    if !(v.is_finite() && v > 0.0) {
        return format!("{v:.1}");
    }
    let scale = 10f64.powf(v.log10().floor() - 1.0);
    let r = (v / scale).round() * scale;
    let decimals = (1.0 - r.log10().floor()).max(0.0) as usize;
    format!("{r:.decimals$}")
}

/// A density: two decimals from a tenth up, two significant figures below.
fn fmt_rho(x: f64) -> String {
    if x >= 0.1 {
        return format!("{x:.2}");
    }
    let text = sig2(x);
    if text.contains('.') {
        let trimmed = text.trim_end_matches('0');
        if trimmed.ends_with('.') { format!("{trimmed}0") } else { trimmed.to_string() }
    } else {
        text
    }
}

/// A number written into a box: up to four decimals, trailing zeros dropped.
fn fmt_number(x: f64) -> String {
    let text = format!("{x:.4}");
    let trimmed = text.trim_end_matches('0');
    match trimmed.strip_suffix('.') {
        Some(whole) => whole.to_string(),
        None => trimmed.to_string(),
    }
}

fn min_max(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
}

#[cfg(test)]
mod tests;
