// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What published sources say a lattice's stiffness and strength are, against its relative
//! density. Backlog item 94; `lattice-qualification-spec.md` §9, and the rulings in §9.1.
//!
//! **A lookup, not analysis, and never material data.** Every number here is a literature fit,
//! `E*/E_s` or `σ*/σ_s` against `ρ`, compiled in with its citation, the kind of data behind it, the
//! density range it was fitted over, what its ratio was divided by, the direction it was loaded
//! in, and every correction its verification made. The readout that quotes them says "estimate",
//! names its source and its direction, and leads with the range across sources -- which at one
//! density runs 1.6 to 8.4 times for the same cell (cubic struts at 0.30: FE 15.6%, tests 3.5% and
//! 1.8%), so a single number would read as more than it is.
//!
//! Seventy-five rows were researched. Forty-eight ship, in [`ROWS`]; the other twenty-seven are in
//! [`EXCLUDED`], each with the reason it does not. the lattice-property-ledger design note
//! keeps the research's numbering, 0 to 74, and a test holds this file to it, so the document and
//! the code cannot drift. What the exclusions taught is the rule each row here passed:
//!
//! - the cited source must contain the number;
//! - `C` must be a ratio to the solid, or one whose divisor can be recovered and is marked so;
//! - a fit must stay under the solid's share, `ρ`, in its own range and agree with its own plot;
//! - the cell and phase must be tied to FieldForm's by something read;
//! - metadata from a review is the review's claim, and the source line says "via";
//! - an offset in a fit is part of the fit;
//! - a density measured on a powder-filled specimen is not the design density.
//!
//! The vocabulary is the kernel's own (D9): [`CellKind`] names the cell and [`Banding`] the phase,
//! so a row is matched to a block by the same types the block is built from. [`LoadDirection`] is
//! new and is neither an `Axis` nor a `Frame`: a row's direction is a statement about a test,
//! relative to the cell, not a place or a vector.

use crate::{BeamKind, CellKind, PrismKind, StochasticKind, TpmsKind};
use fieldform_scene::Banding;

mod rows;

pub use rows::{EXCLUDED, ROWS};

/// Stiffness or strength: the two questions a reading answers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PropertyClass {
    Modulus,
    Strength,
}

/// What a row's ratio is a ratio of.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Property {
    /// `E*/E_s`.
    Modulus,
    /// `σ*/σ_s` by the source's own definition. Two definitions are two numbers, so a range is
    /// only ever taken within one.
    Strength(StrengthKind),
}

/// Which strength a source measured.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StrengthKind {
    /// 0.2% offset yield.
    Yield,
    /// The plateau, or plastic collapse, stress.
    Plateau,
    /// The source, or the review quoting it, does not say.
    NotStated,
}

/// Which way a row's specimen or model was loaded, relative to its cell.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoadDirection {
    /// Along a cube axis. Only when the source says so, or its own model fixes it.
    Axis100,
    /// Along the body diagonal.
    Diagonal111,
    /// In the plane of a prismatic cell's walls (the honeycomb).
    InPlane,
    /// Along a printed specimen's build direction. How the cell sat in it is not stated, so this
    /// is **not** [100].
    BuildDirection,
    /// Never rendered as a cube axis.
    NotStated,
}

/// A row's fit, `ρ` to the ratio.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Fit {
    /// `c·ρⁿ`.
    Power { c: f64, n: f64 },
    /// `c·ρⁿ + offset`. **The offset is part of the fit**: Lu 2020's Schwarz P reads 2.9× high
    /// without it.
    PowerOffset { c: f64, n: f64, offset: f64 },
    /// `c1·ρ^n1` up to the knee, then `c2·ρ^n2 + c3`: continuous at the knee and 1 at `ρ = 1`.
    TwoBranch { c1: f64, n1: f64, knee: f64, c2: f64, n2: f64, c3: f64 },
}

/// Where a row's exponent came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Exponent {
    Fitted,
    Theory,
    /// Fixed by the authors (usually 2, or 3/2 for strength) and `C` calibrated to it.
    Assumed,
    NotStated,
}

/// The low end of a row's density range.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Lower {
    At(f64),
    /// The source does not state its lowest density; its data surely reach down to this one.
    NotStated { surely_from: f64 },
}

/// The densities a row's fit was made over.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DensityRange {
    Span { lo: Lower, hi: f64 },
    /// One density: the exponent is assumed and the fit is a calibration through one point.
    Single(f64),
    NotStated,
}

/// What a source's density axis is. A fit made on measured density is read here at the density
/// built, and the hover says so.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DensityBasis {
    Design,
    Measured,
    NotStated,
}

/// What kind of data a fit was made to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DataKind {
    Experiment,
    Fe,
    FitToFe,
    Analytic,
}

/// Whether we read the source that produced the numbers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    Primary,
    /// Read from another source quoting them. `primary_read` is true when the primary was opened
    /// as well and agrees -- only Alabort 2019.
    Review { via: &'static str, primary_read: bool },
}

/// What a row's ratio was divided by. It decides what an absolute value could ever be: a ratio
/// is only as absolute as the `E_s` it is multiplied by, and sources disagree about `E_s` for one
/// alloy by 3× (§9.1 ruling 6).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SolidBasis {
    /// The model's own solid: geometry and ν only. FE and theory rows, and Naghavi 2022, whose fit
    /// reproduces its FE.
    ModelInput,
    /// The source's own bulk specimens, measured (Ahmadi 2015).
    Measured,
    /// A value the source states (Li 2019's resin, Maskery 2017's PA2200).
    Stated,
    /// Not stated; recovered from the primary's absolute fit (Alabort 2019 via the review).
    Implied,
    /// Not stated and not recoverable (the other review rows).
    NotStated,
}

/// Where a row's numbers were read.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Source {
    /// What the line prints: "Lu 2020", "Wang 2022 via Distefano 2023".
    pub short: &'static str,
    pub citation: &'static str,
    /// The table, figure or equation within the source.
    pub locator: &'static str,
    /// A `doi:` or an `http` address.
    pub link: &'static str,
}

/// One literature fit, with everything that qualifies it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PropertyRow {
    /// Stable: "nx23-g-sheet-e".
    pub id: &'static str,
    /// The row's number in the lattice-property-ledger design note, which keeps the
    /// research's numbering 0–74.
    pub ledger_row: u16,
    pub cell: CellKind,
    pub phase: Banding,
    pub property: Property,
    pub direction: LoadDirection,
    pub fit: Fit,
    pub exponent: Exponent,
    pub range: DensityRange,
    pub density_basis: DensityBasis,
    pub data: DataKind,
    pub provenance: Provenance,
    pub material: &'static str,
    pub solid_basis: SolidBasis,
    /// The divisor where it is known or implied, in MPa: Naghavi 35 770, Alabort ≈114 100.
    pub solid_value_mpa: Option<f64>,
    pub source: Source,
    /// The corrections applied, a sentence each.
    pub notes: &'static str,
}

/// A researched row that does not ship, and why.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExcludedRow {
    pub ledger_row: u16,
    pub cell: Option<CellKind>,
    /// `None` where the phase is not known -- which is itself why the row does not ship.
    pub phase: Option<Banding>,
    pub class: PropertyClass,
    pub source: &'static str,
    pub reason: &'static str,
}

/// A row read at one density.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reading {
    /// The ratio, never clamped; `None` exactly when the standing is [`Standing::Refused`].
    pub value: Option<f64>,
    pub standing: Standing,
}

/// Where a density stands against the row's own range, or why the row is not read there.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Standing {
    Inside,
    /// Warn: the fit is extrapolated.
    Outside { lo: Lower, hi: f64 },
    /// Warn: below the density the data surely reach, and the source does not say how far below
    /// they go (Nguyen-Xuan, below 0.15).
    LowestNotStated { surely_from: f64 },
    /// Warn away from the point -- inside within ±0.005 of it.
    OnePoint { at: f64 },
    /// Warn: nothing else can say whether this is an extrapolation.
    RangeNotStated,
    Refused(Refusal),
}

/// Why a fit is not read at a density at all.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Not finite, or `ρ ≤ 0` or `ρ > 1`.
    NoDensity,
    /// The fit is at or below nought here (Lu's P below 0.183, Lu's gyroid below 0.043).
    NotPositive,
    /// Above nought, but so near an offset fit's zero that the value is the small difference of
    /// two nearly equal numbers: a 1% error in the density would move it by more than
    /// [`MOST_SENSITIVE`] per cent (Lu's P below 0.203, Lu's gyroid below 0.048).
    NearZero,
    /// More than `ρ` times the solid's own value. For a modulus that is the Voigt bound. For a
    /// strength it is the same volume average: the lattice's stress is the mean of the solid's,
    /// which is nought in the voids, so with one solid and one definition it cannot pass `ρ·σ_s`.
    /// (Lu's D below 0.0021, Lu's FKS below 0.024; Wang's cubic strength above 0.71; Sun's gyroid
    /// sheet strength above 0.49.)
    AboveSolidShare,
}

/// The steepest a reading may move with its density, `d ln E / d ln ρ`, before it is refused as
/// [`Refusal::NearZero`].
///
/// **Why ten.** For a power fit the slope is its exponent, and none here passes 2.89, so a 1%
/// error in the density -- which the density tables hold to a few hundredths of that -- moves a
/// reading by under 3%. Only an offset fit near its zero goes past it, and there the reading is
/// the small difference of two nearly equal numbers: Lu 2020's gyroid solid at a built 0.043 has a
/// slope of 95, and half a unit in the last printed place of its offset, −0.0011, moves the value
/// there by 200%. Two significant figures of that would be a number with no digit in it that
/// means anything. Past ten -- more than three times any power fit's -- the row is refused, and
/// the readout falls through to the next one (`property_rows.rs` prints the slope at every row's
/// range ends; the refused spans are in the ledger's §4.3).
pub(crate) const MOST_SENSITIVE: f64 = 10.0;

impl Fit {
    /// The ratio at `ρ`, as fitted: no clamp and no refusal, which are [`PropertyRow::read`]'s.
    pub fn at(self, rho: f64) -> f64 {
        match self {
            Fit::Power { c, n } => c * rho.powf(n),
            Fit::PowerOffset { c, n, offset } => c * rho.powf(n) + offset,
            Fit::TwoBranch { c1, n1, knee, c2, n2, c3 } => {
                if rho <= knee {
                    c1 * rho.powf(n1)
                } else {
                    c2 * rho.powf(n2) + c3
                }
            }
        }
    }

    /// `d ln E / d ln ρ` at `ρ`: how many per cent a reading moves for a 1% error in the density.
    /// The exponent for a power law, and more -- without bound -- as an offset fit nears its zero.
    /// Not finite where the fit is at or below nought.
    pub fn log_slope(self, rho: f64) -> f64 {
        let (n, power, value) = match self {
            Fit::Power { n, .. } => return n,
            Fit::PowerOffset { c, n, .. } => (n, c * rho.powf(n), self.at(rho)),
            Fit::TwoBranch { n1, knee, .. } if rho <= knee => return n1,
            Fit::TwoBranch { c2, n2, .. } => (n2, c2 * rho.powf(n2), self.at(rho)),
        };
        if value > 0.0 { n * power / value } else { f64::INFINITY }
    }
}

impl Property {
    pub(crate) fn class(self) -> PropertyClass {
        match self {
            Property::Modulus => PropertyClass::Modulus,
            Property::Strength(_) => PropertyClass::Strength,
        }
    }

    /// What the percentage is of. Fixed here so the panel, the console and the ledger say one
    /// thing (D9's "one editor", carried to text).
    pub fn phrase(self) -> &'static str {
        match self {
            Property::Modulus => "of the solid's modulus",
            Property::Strength(StrengthKind::Yield) => {
                "of the solid's yield strength, as 0.2% offset yield"
            }
            Property::Strength(StrengthKind::Plateau) => {
                "of the solid's yield strength, as plateau stress"
            }
            Property::Strength(StrengthKind::NotStated) => {
                "of the solid's strength, definition not given"
            }
        }
    }
}

impl LoadDirection {
    /// **The only words a direction is ever rendered in.** A row that states no direction, or only
    /// a build direction, never says "[100]", "[111]" or "cube axis": a cube axis is a claim about
    /// how the cell sat in the specimen, which those sources do not make.
    pub fn phrase(self) -> &'static str {
        match self {
            LoadDirection::Axis100 => "along a cube axis [100]",
            LoadDirection::Diagonal111 => "along a body diagonal [111]",
            LoadDirection::InPlane => "in-plane",
            LoadDirection::BuildDirection => {
                "along the source's build direction (cell axes not stated)"
            }
            LoadDirection::NotStated => "direction not stated",
        }
    }
}

impl DataKind {
    pub fn phrase(self) -> &'static str {
        match self {
            DataKind::Fe => "FE",
            DataKind::FitToFe => "fit to FE",
            DataKind::Experiment => "test",
            DataKind::Analytic => "theory",
        }
    }
}

impl SolidBasis {
    pub fn phrase(self) -> &'static str {
        match self {
            SolidBasis::ModelInput => "over the model's own solid",
            SolidBasis::Measured => "over the source's measured solid",
            SolidBasis::Stated => "over the solid value the source states",
            SolidBasis::Implied => "over a divisor the review does not state",
            SolidBasis::NotStated => "over a solid value not stated",
        }
    }
}

/// How a row ranks for being the default: lower is better, compared field by field.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
struct Rank(u8, u8, u8, u8, f64, u8);

impl PropertyRow {
    pub fn class(&self) -> PropertyClass {
        self.property.class()
    }

    /// The row at `ρ`: refused first, for both classes -- no density, a fit at or below nought or
    /// so near it that the value means nothing, or more than the solid's share -- and otherwise
    /// the standing against the row's own range.
    /// The value is never clamped, and a refused value is not given at all.
    pub fn read(&self, rho: f64) -> Reading {
        let refused = |why| Reading { value: None, standing: Standing::Refused(why) };
        if !(rho.is_finite() && rho > 0.0 && rho <= 1.0) {
            return refused(Refusal::NoDensity);
        }
        let value = self.fit.at(rho);
        // `!(value > 0)` rather than `value <= 0`, so a NaN is refused rather than quoted.
        if !(value > 0.0) {
            return refused(Refusal::NotPositive);
        }
        if self.fit.log_slope(rho) > MOST_SENSITIVE {
            return refused(Refusal::NearZero);
        }
        // A part in a billion of slack, so a two-branch fit that reaches exactly 1 at ρ = 1 is
        // not refused by the last bit of its own arithmetic.
        if value > rho * (1.0 + 1e-9) {
            return refused(Refusal::AboveSolidShare);
        }
        let standing = match self.range {
            DensityRange::Span { lo, hi } => {
                if rho > hi {
                    Standing::Outside { lo, hi }
                } else {
                    match lo {
                        Lower::At(lo) if rho < lo => Standing::Outside { lo: Lower::At(lo), hi },
                        Lower::NotStated { surely_from } if rho < surely_from => {
                            Standing::LowestNotStated { surely_from }
                        }
                        _ => Standing::Inside,
                    }
                }
            }
            DensityRange::Single(at) => {
                if (rho - at).abs() <= 0.005 {
                    Standing::Inside
                } else {
                    Standing::OnePoint { at }
                }
            }
            DensityRange::NotStated => Standing::RangeNotStated,
        };
        Reading { value: Some(value), standing }
    }

    /// The density at which this row reads `target`: bisection on `(0, 1]`, eighty halvings.
    ///
    /// Sound because every fit rises (`every_fit_rises_with_density`). `None` when the target is
    /// outside what the fit reaches, or the density found would be refused. **Knows nothing of
    /// geometry**: whether a lattice can be built at the answer -- above a solid's floor, inside a
    /// wall limit or a merge radius -- is the caller's question, and asking it is not optional.
    pub fn density_for(&self, target: f64) -> Option<f64> {
        if !(target.is_finite() && target > 0.0) || self.fit.at(1.0) < target {
            return None;
        }
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if self.fit.at(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.read(hi).value.map(|_| hi)
    }

    /// Lower is better: provenance (a source we read, then a review checked against its
    /// primary, then a review); exponent (fitted or theory, then not stated, then assumed); data
    /// (FE, fit to FE or theory, then tests); range (a span, then one point, then none); a wider
    /// range; direction (a cube axis or in-plane, then not stated or a build direction, then a
    /// body diagonal). Ties go to table order.
    ///
    /// Why that order (§9.1 ruling 4): what was read can be vouched for, and FE describes the
    /// cells as drawn rather than one printer's. A one-point row keeps its place over review
    /// rows -- its weakness is known and said on the line, a review row's are not.
    fn rank(&self) -> Rank {
        let provenance = match self.provenance {
            Provenance::Primary => 0,
            Provenance::Review { primary_read: true, .. } => 1,
            Provenance::Review { primary_read: false, .. } => 2,
        };
        let exponent = match self.exponent {
            Exponent::Fitted | Exponent::Theory => 0,
            Exponent::NotStated => 1,
            Exponent::Assumed => 2,
        };
        let data = match self.data {
            DataKind::Fe | DataKind::FitToFe | DataKind::Analytic => 0,
            DataKind::Experiment => 1,
        };
        let (range, width) = match self.range {
            DensityRange::Span { lo: Lower::At(lo), hi } => (0, hi - lo),
            DensityRange::Span { lo: Lower::NotStated { surely_from }, hi } => (0, hi - surely_from),
            DensityRange::Single(_) => (1, 0.0),
            DensityRange::NotStated => (2, 0.0),
        };
        let direction = match self.direction {
            LoadDirection::Axis100 | LoadDirection::InPlane => 0,
            LoadDirection::NotStated | LoadDirection::BuildDirection => 1,
            LoadDirection::Diagonal111 => 2,
        };
        Rank(provenance, exponent, data, range, -width, direction)
    }
}

/// Whether two cells are the same cell for a row: a stochastic cell's jitter and seed are
/// ignored, since no row is keyed by them.
///
/// **The arms name each family, and anything else answers "no row".** That is the safe direction
/// for a family that does not exist yet: it gets "no published fit" until somebody decides a row
/// describes it, rather than a neighbour's.
pub(crate) fn same_cell(a: CellKind, b: CellKind) -> bool {
    match (a, b) {
        (CellKind::Tpms(x), CellKind::Tpms(y)) => x == y,
        (CellKind::Beam(x), CellKind::Beam(y)) => x == y,
        (CellKind::Prism(x), CellKind::Prism(y)) => x == y,
        (CellKind::Stochastic { kind: x, .. }, CellKind::Stochastic { kind: y, .. }) => x == y,
        _ => false,
    }
}

/// Every row for one cell, phase and class, in table order.
pub fn rows_for(
    cell: CellKind,
    phase: Banding,
    class: PropertyClass,
) -> impl Iterator<Item = &'static PropertyRow> {
    ROWS.iter()
        .filter(move |r| same_cell(r.cell, cell) && r.phase == phase && r.class() == class)
}

fn best(rows: impl Iterator<Item = &'static PropertyRow>) -> Option<&'static PropertyRow> {
    // `min_by` keeps the first of equals, which is what makes ties go to table order.
    rows.min_by(|a, b| a.rank().partial_cmp(&b.rank()).expect("ranks are finite"))
}

/// The row a key is quoted by, before anything is read: the best-ranked of [`rows_for`].
pub fn default_row(
    cell: CellKind,
    phase: Banding,
    class: PropertyClass,
) -> Option<&'static PropertyRow> {
    best(rows_for(cell, phase, class))
}

/// Every row for one key, best-ranked first and ties in table order, so the first is
/// [`default_row`].
///
/// What the readout's hover lists, and where a reading falls through to when the default's fit
/// gives nothing at a density (Lu's Schwarz P below 0.183). **Here rather than in the app**,
/// because the rank is the ruling (§9.1 ruling 4) and a second copy of it in the panel would be
/// a second place to change it.
pub fn ranked_rows(
    cell: CellKind,
    phase: Banding,
    class: PropertyClass,
) -> Vec<&'static PropertyRow> {
    let mut rows: Vec<&'static PropertyRow> = rows_for(cell, phase, class).collect();
    // A stable sort keeps table order among equals: the tie rule `best` keeps by taking the first.
    rows.sort_by(|a, b| a.rank().partial_cmp(&b.rank()).expect("ranks are finite"));
    rows
}

/// The best-ranked row along the other cubic axis from `default`'s. Only the octet has one: its
/// [100] is Cellucci 2016's and its [111] Deshpande 2001's as Cellucci quotes it. A row whose
/// direction is not a cube axis has no partner, whatever else is in its key.
pub fn other_axis_row(default: &PropertyRow) -> Option<&'static PropertyRow> {
    let wanted = match default.direction {
        LoadDirection::Axis100 => LoadDirection::Diagonal111,
        LoadDirection::Diagonal111 => LoadDirection::Axis100,
        LoadDirection::InPlane | LoadDirection::BuildDirection | LoadDirection::NotStated => {
            return None
        }
    };
    best(rows_for(default.cell, default.phase, default.class()).filter(move |r| r.direction == wanted))
}

/// The researched rows that do not ship for a key, so the hover can say one was found and why it
/// was not used. A row whose phase is unknown is offered for every phase of its cell: it ships for
/// neither, and saying so is part of saying why there is nothing.
pub fn excluded_for(
    cell: CellKind,
    phase: Banding,
    class: PropertyClass,
) -> impl Iterator<Item = &'static ExcludedRow> {
    EXCLUDED.iter().filter(move |r| {
        r.cell.is_some_and(|c| same_cell(c, cell))
            && r.phase.is_none_or(|p| p == phase)
            && r.class == class
    })
}

#[cfg(test)]
mod tests;
