// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The table itself: the forty-eight researched rows that ship, and the twenty-seven that do not.

use super::*;

// ---------------------------------------------------------------------------------------------
// The rows. Each is the research row of the same `ledger_row`, with every correction its
// verification and the design review made, said in `notes`. Fit numbers are exactly as verified
// unless the notes say otherwise.
// ---------------------------------------------------------------------------------------------

/// Distefano 2023's Table 4 is a review, and every row read only from it carries the same caveats.
const DISTEFANO: Provenance = Provenance::Review { via: "Distefano 2023, Table 4", primary_read: false };
const DISTEFANO_LINK: &str = "https://www.mdpi.com/2079-4983/14/3/125";
const TITANIUM: &str = "titanium (the review covers titanium lattices); alloy not verified";

/// The correction every Distefano row carries, with the row's own appended.
macro_rules! distefano_notes {
    ($extra:literal) => {
        concat!(
            "The table header's unit (GPa or MPa) is a label: the values are the dimensionless ",
            "constants of the review's Eqs (5) and (6), and in those units a titanium lattice's ",
            "strength would be capped at a few MPa. Read from the review only; its primary was not ",
            "opened. ",
            $extra
        )
    };
}

macro_rules! distefano_citation {
    ($primary:literal) => {
        concat!(
            $primary,
            ", as tabulated in Distefano F., Pasta S., Epasto G. (2023) Titanium Lattice ",
            "Structures Produced via Additive Manufacturing for a Bone Scaffold: A Review. ",
            "J. Funct. Biomater. 14(3):125"
        )
    };
}

/// A row read only from Distefano 2023's Table 4: test data, range, density basis, direction,
/// exponent and divisor all not stated.
#[allow(clippy::too_many_arguments)]
const fn quoted(
    id: &'static str,
    ledger_row: u16,
    cell: CellKind,
    phase: Banding,
    property: Property,
    fit: Fit,
    material: &'static str,
    short: &'static str,
    citation: &'static str,
    locator: &'static str,
    notes: &'static str,
) -> PropertyRow {
    PropertyRow {
        id,
        ledger_row,
        cell,
        phase,
        property,
        direction: LoadDirection::NotStated,
        fit,
        exponent: Exponent::NotStated,
        range: DensityRange::NotStated,
        density_basis: DensityBasis::NotStated,
        data: DataKind::Experiment,
        provenance: DISTEFANO,
        material,
        solid_basis: SolidBasis::NotStated,
        solid_value_mpa: None,
        source: Source { short, citation, locator, link: DISTEFANO_LINK },
        notes,
    }
}

const NX23: Source = Source {
    short: "Nguyen-Xuan 2023",
    citation: "Nguyen-Xuan H., Tran K.Q., Thai C.H., Lee J. (2023) Modelling of functionally graded \
               triply periodic minimal surface (FG-TPMS) plates. arXiv:2303.13838",
    locator: "Table 2, E* row",
    link: "https://arxiv.org/abs/2303.13838",
};
const NX23_MATERIAL: &str = "isotropic parent, ν_s = 0.3";
const NX23_RANGE: DensityRange =
    DensityRange::Span { lo: Lower::NotStated { surely_from: 0.15 }, hi: 0.80 };

const NAGHAVI22: Source = Source {
    short: "Naghavi 2022",
    citation: "Naghavi S.A., Tamaddon M., Marghoub A., et al., Liu C. (2022) Mechanical \
               Characterisation and Numerical Modelling of TPMS-Based Gyroid and Diamond Ti6Al4V \
               Scaffolds for Bone Implants. Bioengineering 9(10):504",
    locator: "Table 7",
    link: "doi:10.3390/bioengineering9100504",
};
const NAGHAVI22_MATERIAL: &str = "Ti-6Al-4V, SLM; the FE's solid was the source's own measured \
                                  bulk, 35.77 GPa and 1012 MPa yield, about a third of handbook \
                                  stiffness";

const LI19: Source = Source {
    short: "Li 2019",
    citation: "Li D., Liao W., Dai N., Xie Y.M. (2019) Comparison of Mechanical Properties and \
               Energy Absorption of Sheet-Based and Strut-Based Gyroid Cellular Structures with \
               Graded Densities. Materials 12(13):2183",
    locator: "Table 2, C1 (modulus) and C2 (plateau)",
    link: "doi:10.3390/ma12132183",
};

const LU20: Source = Source {
    short: "Lu 2020",
    citation: "Lu Y., Cheng L., Yang Z., Li J., Zhu H. (2020) Relationship between the \
               morphological, mechanical and permeability properties of porous bone scaffolds and \
               the underlying microstructure. PLoS ONE 15(9):e0238471",
    locator: "Table 3 (present study, Eq 5); range from Fig 5a",
    link: "doi:10.1371/journal.pone.0238471",
};
const LU20_MATERIAL: &str = "Ti-6Al-4V as the FE's input, 110 GPa";
const LU20_RANGE: DensityRange = DensityRange::Span { lo: Lower::At(0.30), hi: 0.50 };

/// The correction the verification made to every Lu 2020 row's direction.
macro_rules! lu20_axis {
    () => {
        concat!(
            "Along a cube axis [100], not 'not stated': the homogenisation loads a strain along ",
            "one cell axis (ε_x = 0.01, periodic boundaries), so its own model fixes the direction."
        )
    };
}

/// A Lu 2020 row: FE homogenisation with periodic boundaries over 0.30–0.50, its offset kept.
///
/// **Along a cube axis, because its own model fixes it** (the verification's correction to all
/// five): the homogenisation loads one cell with a strain along one of its own axes
/// (`ε_x = 0.01`, periodic boundaries), so the "effective compressive modulus" it fits is the
/// cell's `E[100]` -- the case [`LoadDirection::Axis100`] allows, as Cellucci 2016's beam model is.
/// A test of a printed block states no such thing, so Lu's rows are ranged apart from those.
const fn lu20(
    id: &'static str,
    ledger_row: u16,
    cell: CellKind,
    phase: Banding,
    fit: Fit,
    notes: &'static str,
) -> PropertyRow {
    PropertyRow {
        id,
        ledger_row,
        cell,
        phase,
        property: Property::Modulus,
        direction: LoadDirection::Axis100,
        fit,
        exponent: Exponent::Fitted,
        range: LU20_RANGE,
        density_basis: DensityBasis::Design,
        data: DataKind::Fe,
        provenance: Provenance::Primary,
        material: LU20_MATERIAL,
        solid_basis: SolidBasis::ModelInput,
        solid_value_mpa: Some(110_000.0),
        source: LU20,
        notes,
    }
}

const AHMADI15_CITATION: &str = "Ahmadi S.M., Yavari S.A., Wauthle R., Pouran B., Schrooten J., \
                                 Weinans H., Zadpoor A.A. (2015) Additively Manufactured Open-Cell \
                                 Porous Biomaterials Made from Six Different Space-Filling Unit \
                                 Cells. Materials 8(4):1871–1896";

const MASKERY17: Source = Source {
    short: "Maskery 2017",
    citation: "Maskery I., Hussey A., Panesar A., Aremu A., Tuck C., Ashcroft I., Hague R. (2017) \
               An investigation into reinforced and functionally graded lattice structures. \
               J. Cell. Plast. 53(2):151–165",
    locator: "Table 4",
    link: "doi:10.1177/0021955X16639035",
};

const ALABORT19_CITATION: &str = distefano_citation!(
    "Alabort E., Barba D., Reed R.C. (2019) Design of metallic bone by additive manufacturing. \
     Scr. Mater. 164:110–114"
);
const ALABORT19_MATERIAL: &str =
    "titanium; the implied divisors are typical of Ti-6Al-4V; alloy not verified";

/// The caveat every Distefano row carries (`distefano_notes!`), said for the four whose primary
/// was read: the primary's absolute fit is what confirms the review's numbers are ratios.
macro_rules! alabort_notes {
    ($($extra:tt)*) => {
        concat!(
            "The review's table header's unit (GPa or MPa) is a label: its values are the ",
            "dimensionless constants of its Eqs (5) and (6), which the primary's absolute fit, ",
            "read, confirms: the primary's coefficient over the review's is a titanium modulus or ",
            "yield, so the review's is a ratio to the solid. ",
            $($extra)*
        )
    };
}

/// Alabort 2019 via the review, with its primary read: the primary's exponents over the review's
/// unstated divisor.
#[allow(clippy::too_many_arguments)]
const fn alabort19(
    id: &'static str,
    ledger_row: u16,
    cell: CellKind,
    property: Property,
    fit: Fit,
    solid_value_mpa: f64,
    locator: &'static str,
    notes: &'static str,
) -> PropertyRow {
    PropertyRow {
        id,
        ledger_row,
        cell,
        phase: Banding::Solid,
        property,
        direction: LoadDirection::NotStated,
        fit,
        exponent: Exponent::Fitted,
        range: DensityRange::NotStated,
        density_basis: DensityBasis::NotStated,
        data: DataKind::Experiment,
        provenance: Provenance::Review { via: "Distefano 2023, Table 4", primary_read: true },
        material: ALABORT19_MATERIAL,
        solid_basis: SolidBasis::Implied,
        solid_value_mpa: Some(solid_value_mpa),
        source: Source {
            short: "Alabort 2019 via Distefano 2023",
            citation: ALABORT19_CITATION,
            locator,
            link: DISTEFANO_LINK,
        },
        notes,
    }
}

pub(crate) const GYROID: CellKind = CellKind::Tpms(TpmsKind::Gyroid);
pub(crate) const SCHWARZ_P: CellKind = CellKind::Tpms(TpmsKind::SchwarzP);
pub(crate) const SCHWARZ_D: CellKind = CellKind::Tpms(TpmsKind::SchwarzD);
pub(crate) const SCHOEN_IWP: CellKind = CellKind::Tpms(TpmsKind::SchoenIwp);
pub(crate) const NEOVIUS: CellKind = CellKind::Tpms(TpmsKind::Neovius);
pub(crate) const FISCHER_KOCH_S: CellKind = CellKind::Tpms(TpmsKind::FischerKochS);
pub(crate) const CUBIC: CellKind = CellKind::Beam(BeamKind::Cubic);
pub(crate) const BCC: CellKind = CellKind::Beam(BeamKind::Bcc);
pub(crate) const FCC: CellKind = CellKind::Beam(BeamKind::Fcc);
pub(crate) const OCTET: CellKind = CellKind::Beam(BeamKind::Octet);
pub(crate) const DIAMOND: CellKind = CellKind::Beam(BeamKind::Diamond);
pub(crate) const HONEYCOMB: CellKind = CellKind::Prism(PrismKind::Honeycomb);
pub(crate) const VORONOI_BEAM: CellKind =
    CellKind::Stochastic { kind: StochasticKind::VoronoiBeam, jitter: 0.0, seed: 0 };

const MODULUS: Property = Property::Modulus;
const STRENGTH: Property = Property::Strength(StrengthKind::NotStated);

/// The 48 literature rows that ship. Table order is the ledger's, and breaks ties in the rank.
pub static ROWS: &[PropertyRow] = &[
    // --- Nguyen-Xuan 2023: three sheet moduli, fitted to FE ---------------------------------
    ROW_NX23_P,
    PropertyRow {
        id: "nx23-g-sheet-e",
        ledger_row: 1,
        cell: GYROID,
        fit: Fit::TwoBranch {
            c1: 0.596,
            n1: 1.467,
            knee: 0.45,
            c2: 0.962_557_118_632_729_2,
            n2: 2.351,
            c3: 0.037_442_881_367_270_81,
        },
        notes: concat!(
            "C3 is 0.0374, not the 0.038 a rounded reading gives; C2 is recomputed from the ",
            "continuity condition and E*(1) = 1. The fit is to FE homogenisation from two earlier ",
            "studies, 0–0.15 and 0.15–0.80; the first's lowest density is not stated, so the range ",
            "does not claim nought and the line warns below 0.15. No loading direction is stated."
        ),
        ..ROW_NX23_P
    },
    PropertyRow {
        id: "nx23-iwp-sheet-e",
        ledger_row: 2,
        cell: SCHOEN_IWP,
        fit: Fit::TwoBranch {
            c1: 0.597,
            n1: 1.225,
            knee: 0.35,
            c2: 0.987_012_958_669_008_6,
            n2: 1.782,
            c3: 0.012_987_041_330_991_378,
        },
        notes: concat!(
            "The upper branch is recomputed from the continuity condition and E*(1) = 1 (C2 ",
            "0.9870, C3 0.0130). The fit is to FE homogenisation from two earlier studies, 0–0.15 ",
            "and 0.15–0.80; the first's lowest density is not stated, so the range does not claim ",
            "nought and the line warns below 0.15. No loading direction is stated."
        ),
        ..ROW_NX23_P
    },
    // --- Naghavi 2022: gyroid and diamond sheets, Ti-6Al-4V, FE ------------------------------
    ROW_NAGHAVI22_G,
    PropertyRow {
        id: "naghavi22-g-sheet-yield",
        ledger_row: 4,
        property: Property::Strength(StrengthKind::Yield),
        fit: Fit::Power { c: 0.536, n: 1.45 },
        solid_value_mpa: Some(1012.0),
        notes: concat!(
            "As naghavi22-g-sheet-e: the FE's range, 0.28–0.46; the ratio is the model's, over its ",
            "own measured bulk yield of 1012 MPa; direction not stated relative to the cell. G800: ",
            "fit 133 MPa, FE 133, test 108."
        ),
        ..ROW_NAGHAVI22_G
    },
    PropertyRow {
        id: "naghavi22-d-sheet-e",
        ledger_row: 5,
        cell: SCHWARZ_D,
        fit: Fit::Power { c: 0.713, n: 1.21 },
        range: DensityRange::Span { lo: Lower::At(0.30), hi: 0.44 },
        notes: concat!(
            "The range is the FE's, 56.4–70.3% porosity (0.30–0.44), not 0.38–0.47. FE data, a ",
            "300 µm sheet; the ratio is the model's, over its own measured bulk. Direction not ",
            "stated relative to the cell."
        ),
        ..ROW_NAGHAVI22_G
    },
    PropertyRow {
        id: "naghavi22-d-sheet-yield",
        ledger_row: 6,
        cell: SCHWARZ_D,
        property: Property::Strength(StrengthKind::Yield),
        fit: Fit::Power { c: 0.419, n: 1.14 },
        range: DensityRange::Span { lo: Lower::At(0.30), hi: 0.44 },
        solid_value_mpa: Some(1012.0),
        notes: concat!(
            "The range is the FE's, 0.30–0.44. The fit sits between FE and test (D900: 165 MPa, FE ",
            "170, test 159). Over the model's own measured bulk yield, 1012 MPa."
        ),
        ..ROW_NAGHAVI22_G
    },
    // --- Li 2019: one gyroid density in SLA resin, n assumed -------------------------------
    ROW_LI19_G_SHEET,
    PropertyRow {
        id: "li19-g-sheet-plateau",
        ledger_row: 8,
        property: Property::Strength(StrengthKind::Plateau),
        fit: Fit::Power { c: 0.179, n: 1.5 },
        material: "SLA resin (Form 2); the resin's tensile yield, 132.5 MPa",
        solid_value_mpa: Some(132.5),
        notes: concat!(
            "A compressive plateau over the solid's tensile yield; a one-point calibration at 0.30 ",
            "with m = 3/2 assumed."
        ),
        ..ROW_LI19_G_SHEET
    },
    PropertyRow {
        id: "li19-g-solid-e",
        ledger_row: 9,
        phase: Banding::Solid,
        fit: Fit::Power { c: 0.512, n: 2.0 },
        notes: concat!(
            "The paper's 'strut-based gyroid' is the level set's network solid, so the phase is ",
            "Solid. A one-point calibration at 0.30 with n = 2 assumed; compressed along the ",
            "manufacturing direction."
        ),
        ..ROW_LI19_G_SHEET
    },
    PropertyRow {
        id: "li19-g-solid-plateau",
        ledger_row: 10,
        phase: Banding::Solid,
        property: Property::Strength(StrengthKind::Plateau),
        fit: Fit::Power { c: 0.135, n: 1.5 },
        material: "SLA resin (Form 2); the resin's tensile yield, 132.5 MPa",
        solid_value_mpa: Some(132.5),
        notes: concat!(
            "The network solid, as li19-g-solid-e. A compressive plateau over the solid's tensile ",
            "yield, m = 3/2 assumed. C recomputes to 0.134; the printed 0.135 is kept."
        ),
        ..ROW_LI19_G_SHEET
    },
    // --- Lu 2020: four solids and the cubic strut, FE, offsets kept --------------------------
    lu20(
        "lu20-d-solid-e",
        11,
        SCHWARZ_D,
        Banding::Solid,
        Fit::PowerOffset { c: 0.743, n: 2.081, offset: 0.0021 },
        concat!(
            "The phase is confirmed Solid; the range is read from Fig 5a; FE homogenisation with ",
            "periodic boundaries. The offset is part of the fit and kept: below 0.0021 the fit ",
            "passes ρ and is refused. ",
            lu20_axis!()
        ),
    ),
    lu20(
        "lu20-g-solid-e",
        12,
        GYROID,
        Banding::Solid,
        Fit::PowerOffset { c: 0.919, n: 2.131, offset: -0.0011 },
        concat!(
            "The phase is confirmed Solid; the range is read from Fig 5a; FE homogenisation with ",
            "periodic boundaries. The offset is kept: the fit is at or below nought under 0.043, ",
            "and under 0.048 so near it that a 1% density error moves it by more than 10%; it is ",
            "refused there. ",
            lu20_axis!()
        ),
    ),
    lu20(
        "lu20-p-solid-e",
        13,
        SCHWARZ_P,
        Banding::Solid,
        Fit::PowerOffset { c: 0.950, n: 0.863, offset: -0.2193 },
        concat!(
            "The offset is essential: 0.117 at 0.30 with it, 0.336 without, and Fig 5a agrees with ",
            "0.117. The fit is nought at 0.183, and under 0.203 so near it that a 1% density error ",
            "moves it by more than 10%; it is refused there. FE homogenisation with periodic ",
            "boundaries. ",
            lu20_axis!()
        ),
    ),
    lu20(
        "lu20-fks-solid-e",
        14,
        FISCHER_KOCH_S,
        Banding::Solid,
        Fit::PowerOffset { c: 0.918, n: 2.322, offset: 0.0242 },
        concat!(
            "The offset is about 30% of the value at 0.30 and is kept; below 0.024 the fit passes ",
            "ρ and is refused. FE homogenisation with periodic boundaries. ",
            lu20_axis!()
        ),
    ),
    lu20(
        "lu20-cubic-strut-e",
        15,
        CUBIC,
        Banding::Strut,
        Fit::PowerOffset { c: 0.672, n: 1.216, offset: 0.0001 },
        concat!(
            "'Cube' is struts on the twelve cell edges, which is FieldForm's Cubic. The offset is ",
            "kept. FE homogenisation with periodic boundaries. ",
            lu20_axis!()
        ),
    ),
    // --- Ahmadi 2015: cubic and diamond struts, SLM Ti-6Al-4V ELI, tested ---------------------
    ROW_AHMADI15_CUBIC,
    PropertyRow {
        id: "ahmadi15-diamond-strut-e",
        ledger_row: 34,
        cell: DIAMOND,
        fit: Fit::Power { c: 0.167, n: 1.68 },
        range: DensityRange::Span { lo: Lower::At(0.11), hi: 0.36 },
        source: Source {
            short: "Ahmadi 2015",
            citation: AHMADI15_CITATION,
            locator: "Fig. 9D",
            link: "doi:10.3390/ma8041871",
        },
        notes: concat!(
            "The range is the fit's own four points, read off Fig. 9D at about 0.11, 0.21, 0.28 ",
            "and 0.36 on its axis, so both ends share one basis. Fitted on measured density. No ",
            "loading axis stated."
        ),
        ..ROW_AHMADI15_CUBIC
    },
    // --- Maskery 2017: one BCC density in PA2200, along the build direction ------------------
    ROW_MASKERY17_E,
    PropertyRow {
        id: "maskery17-bcc-strut-plateau",
        ledger_row: 41,
        property: Property::Strength(StrengthKind::Plateau),
        fit: Fit::Power { c: 0.202, n: 1.5 },
        material: "PA2200, SLS; σ_ys = 55 MPa",
        solid_value_mpa: Some(55.0),
        notes: concat!(
            "Plastic collapse strength at one density, m = 3/2 assumed; loaded along z, the build ",
            "direction."
        ),
        ..ROW_MASKERY17_E
    },
    // --- The octet's two cubic axes, by theory -------------------------------------------------
    ROW_CELLUCCI16,
    PropertyRow {
        id: "deshpande01-octet-strut-e111",
        ledger_row: 43,
        direction: LoadDirection::Diagonal111,
        fit: Fit::Power { c: 0.2, n: 1.0 },
        // Not Cellucci's 0.001–0.10: that is the range Cellucci's own sweep confirms 1/5 over,
        // and Deshpande's analysis, as quoted, states none.
        range: DensityRange::NotStated,
        data: DataKind::Analytic,
        provenance: Provenance::Review {
            via: "Cellucci 2016 (text after Eq 5c)",
            primary_read: false,
        },
        material: "a pin-jointed analysis; any isotropic solid",
        source: Source {
            short: "Deshpande 2001 via Cellucci 2016",
            citation: "Deshpande V.S., Fleck N.A., Ashby M.F. (2001) Effective properties of the \
                       octet-truss lattice material. J. Mech. Phys. Solids 49(8):1747–1769, as \
                       quoted in Cellucci D., Cheung K.C. (2016) arXiv:1512.07512",
            locator: "Cellucci 2016, text after Eq 5c",
            link: "https://arxiv.org/abs/1512.07512",
        },
        notes: concat!(
            "A quote of Deshpande's analysis whose primary was not read, so it is tagged 'via' and ",
            "its data is theory. Checked twice: Cellucci's own S-matrix (S11 = 9, S12 = −3, S44 = 12 ",
            "in 1/(ρE_s), Voigt notation with engineering shear) gives 1/E[111] = 5; and a ",
            "pin-jointed octet's bars, six <110> directions with equal share, give C11 = 1/6 and ",
            "C12 = C44 = 1/12 in ρE_s, which is the same S-matrix. The tensor-shear reading of S44 ",
            "(1/E = 17) is not the Voigt convention and contradicts the bar model. The range is ",
            "not stated: Deshpande's is a slender-strut analysis, and the 0.001–0.10 over which ",
            "Cellucci's own S-matrix gives the same 1/5 is Cellucci's sweep, not Deshpande's."
        ),
        ..ROW_CELLUCCI16
    },
    PropertyRow {
        id: "bhate19-octet-strut-strength",
        ledger_row: 44,
        cell: OCTET,
        phase: Banding::Strut,
        property: STRENGTH,
        direction: LoadDirection::NotStated,
        fit: Fit::Power { c: 1.0 / 3.0, n: 1.0 },
        exponent: Exponent::Theory,
        range: DensityRange::NotStated,
        density_basis: DensityBasis::NotStated,
        data: DataKind::Analytic,
        provenance: Provenance::Review {
            via: "Bhate et al. 2019, Designs 3(1):19, Table 1",
            primary_read: false,
        },
        material: "any solid; an idealised scaling for slender struts",
        solid_basis: SolidBasis::ModelInput,
        solid_value_mpa: None,
        source: Source {
            short: "O'Masta 2017 via Bhate 2019",
            citation: "O'Masta M.R. et al. (2017) J. Mech. Phys. Solids 98:271–289, as tabulated \
                       in Bhate D., Penick C.A., Ferry L.A., Lee C. (2019) Classification and \
                       Selection of Cellular Materials in Mechanical Design: Engineering and \
                       Biomimetic Approaches. Designs 3(1):19",
            locator: "Bhate 2019, Table 1 ('from [14]')",
            link: "https://www.mdpi.com/2411-9660/3/1/19",
        },
        notes: concat!(
            "An idealised scaling for slender struts, not a fit; no direction and no density range ",
            "are given. Read from the review only."
        ),
    },
    // --- Beigrezaee 2026: the honeycomb, in-plane, a fit to a closed form ---------------------
    PropertyRow {
        id: "beigrezaee26-honeycomb-sheet-e",
        ledger_row: 45,
        cell: HONEYCOMB,
        phase: Banding::Sheet,
        property: MODULUS,
        direction: LoadDirection::InPlane,
        fit: Fit::Power { c: 0.66, n: 2.43 },
        exponent: Exponent::Fitted,
        range: DensityRange::Span { lo: Lower::At(0.0569), hi: 0.4940 },
        density_basis: DensityBasis::NotStated,
        data: DataKind::Analytic,
        provenance: Provenance::Primary,
        material: "beam theory, Timoshenko closed form (Eqs 3a/4a); depends on ν_s, value not \
                   recorded",
        solid_basis: SolidBasis::ModelInput,
        solid_value_mpa: None,
        source: Source {
            short: "Beigrezaee 2026",
            citation: "Beigrezaee M.J., Jalali S.K., Misseroni D., Pugno N.M. (2026) A refined \
                       Gibson-Ashby model for functionally graded honeycombs with random \
                       irregularities. Int. J. Mech. Sci. 314:111297",
            locator: "Table 2 (HEX)",
            link: "doi:10.1016/j.ijmecsci.2026.111297",
        },
        notes: concat!(
            "The source states t/l 0.05–0.5, converted with the exact regular-hexagon fraction ",
            "ρ = 1 − (1 − t/(√3 l))², which is also Gibson–Ashby's corrected 2t/(√3 l) − t²/(3l²): ",
            "0.0569–0.4940. Which density the source fitted against, exact or thin-wall ",
            "2t/(√3 l), is not stated; read at the thin-wall density the fit is 1.07× higher at ",
            "t/l 0.1, 1.25× at 0.3 and 1.46× at 0.5. Fitted to a closed form, not to tests or FE. ",
            "In-plane only: out of plane a honeycomb is about ρ, 16× more at ρ = 0.19."
        ),
    },
    // --- Distefano 2023, Table 4: rows read only from the review -----------------------------
    quoted(
        "wang22-cubic-strut-e",
        47,
        CUBIC,
        Banding::Strut,
        MODULUS,
        Fit::Power { c: 0.55, n: 2.82 },
        TITANIUM,
        "Wang 2022 via Distefano 2023",
        distefano_citation!("Wang N. et al. (2022) Biomater. Adv. 137:212829"),
        "Table 4, row 'Cubic [119]'; Eq (5)",
        distefano_notes!("Wang's own paper was not read."),
    ),
    quoted(
        "wang22-cubic-strut-strength",
        48,
        CUBIC,
        Banding::Strut,
        STRENGTH,
        Fit::Power { c: 1.34, n: 1.85 },
        TITANIUM,
        "Wang 2022 via Distefano 2023",
        distefano_citation!("Wang N. et al. (2022) Biomater. Adv. 137:212829"),
        "Table 4, row 'Cubic [119]'; Eq (6)",
        distefano_notes!("The fit passes ρ above 0.71, and is refused there."),
    ),
    quoted(
        "crupi17-bcc-strut-e",
        49,
        BCC,
        Banding::Strut,
        MODULUS,
        Fit::Power { c: 0.15, n: 2.0 },
        "Ti-6Al-4V (the primary's abstract)",
        "Crupi 2017 via Distefano 2023",
        distefano_citation!("Crupi V. et al. (2017) Mater. Des. 135:246–256"),
        "Table 4, row 'BCC [120]'; Eq (5)",
        distefano_notes!(
            "The alloy is named from the primary's abstract. The review does not say whether n = 2 \
             was fitted or fixed."
        ),
    ),
    quoted(
        "crupi17-bcc-strut-strength",
        50,
        BCC,
        Banding::Strut,
        STRENGTH,
        Fit::Power { c: 0.57, n: 1.9 },
        "Ti-6Al-4V (the primary's abstract)",
        "Crupi 2017 via Distefano 2023",
        distefano_citation!("Crupi V. et al. (2017) Mater. Des. 135:246–256"),
        "Table 4, row 'BCC [120]'; Eq (6)",
        distefano_notes!("The alloy is named from the primary's abstract."),
    ),
    quoted(
        "wang22-octet-strut-e",
        53,
        OCTET,
        Banding::Strut,
        MODULUS,
        Fit::Power { c: 0.51, n: 2.33 },
        TITANIUM,
        "Wang 2022 via Distefano 2023",
        distefano_citation!("Wang N. et al. (2022) Biomater. Adv. 137:212829"),
        "Table 4, row 'Octet-truss [119]'; Eq (5)",
        distefano_notes!(
            "n = 2.33 is unlike an ideal stretch-dominated octet, and no direction is stated, so \
             it is counted apart from the octet's two axes."
        ),
    ),
    quoted(
        "wang22-octet-strut-strength",
        54,
        OCTET,
        Banding::Strut,
        STRENGTH,
        Fit::Power { c: 1.37, n: 1.37 },
        TITANIUM,
        "Wang 2022 via Distefano 2023",
        distefano_citation!("Wang N. et al. (2022) Biomater. Adv. 137:212829"),
        "Table 4, row 'Octet-truss [119]'; Eq (6)",
        distefano_notes!("The fit passes ρ above 0.43, and is refused there."),
    ),
    quoted(
        "sun22-p-sheet-e",
        55,
        SCHWARZ_P,
        Banding::Sheet,
        MODULUS,
        Fit::Power { c: 0.09, n: 1.15 },
        TITANIUM,
        "Sun 2022 via Distefano 2023",
        distefano_citation!("Sun Q., Sun J., Guo K., Wang L. (2022) Mech. Mater. 166:104241"),
        "Table 4, row 'Schwarz Primitive [123]'; Eq (5)",
        distefano_notes!(
            "Sheet by the review's grouping, and the same source's D and G rows are tagged Sheet; \
             unverified against the primary."
        ),
    ),
    quoted(
        "sun22-p-sheet-strength",
        56,
        SCHWARZ_P,
        Banding::Sheet,
        STRENGTH,
        Fit::Power { c: 0.34, n: 1.25 },
        TITANIUM,
        "Sun 2022 via Distefano 2023",
        distefano_citation!("Sun Q., Sun J., Guo K., Wang L. (2022) Mech. Mater. 166:104241"),
        "Table 4, row 'Schwarz Primitive [123]'; Eq (6)",
        distefano_notes!(
            "Sheet by the review's grouping; unverified against the primary. The only Schwarz P \
             sheet strength found, so the default for it."
        ),
    ),
    quoted(
        "sun22-d-sheet-e",
        59,
        SCHWARZ_D,
        Banding::Sheet,
        MODULUS,
        Fit::Power { c: 0.12, n: 1.06 },
        TITANIUM,
        "Sun 2022 via Distefano 2023",
        distefano_citation!("Sun Q., Sun J., Guo K., Wang L. (2022) Mech. Mater. 166:104241"),
        "Table 4, row 'Diamond Sheet [123]'; Eq (5)",
        distefano_notes!("Tagged Sheet by the review."),
    ),
    quoted(
        "sun22-d-sheet-strength",
        60,
        SCHWARZ_D,
        Banding::Sheet,
        STRENGTH,
        Fit::Power { c: 1.66, n: 1.89 },
        TITANIUM,
        "Sun 2022 via Distefano 2023",
        distefano_citation!("Sun Q., Sun J., Guo K., Wang L. (2022) Mech. Mater. 166:104241"),
        "Table 4, row 'Diamond Sheet [123]'; Eq (6)",
        distefano_notes!(
            "The fit passes ρ above 0.57 and the solid itself above 0.77; refused above 0.57."
        ),
    ),
    quoted(
        "yan15-d-solid-e",
        61,
        SCHWARZ_D,
        Banding::Solid,
        MODULUS,
        Fit::Power { c: 0.17, n: 1.64 },
        TITANIUM,
        "Yan 2015 via Distefano 2023",
        distefano_citation!(
            "Yan C., Hao L., Hussein A., Young P. (2015) J. Mech. Behav. Biomed. Mater. 51:61–73"
        ),
        "Table 4, row 'Diamond Skeletal [15]'; Eq (5)",
        distefano_notes!("'Skeletal' is the network solid."),
    ),
    quoted(
        "yan15-d-solid-strength",
        62,
        SCHWARZ_D,
        Banding::Solid,
        STRENGTH,
        Fit::Power { c: 1.39, n: 1.95 },
        TITANIUM,
        "Yan 2015 via Distefano 2023",
        distefano_citation!(
            "Yan C., Hao L., Hussein A., Young P. (2015) J. Mech. Behav. Biomed. Mater. 51:61–73"
        ),
        "Table 4, row 'Diamond Skeletal [15]'; Eq (6)",
        distefano_notes!("The fit passes ρ above 0.71, and is refused there."),
    ),
    alabort19(
        "alabort19-d-solid-e",
        63,
        SCHWARZ_D,
        MODULUS,
        Fit::Power { c: 0.70, n: 2.67 },
        114_100.0,
        "the primary's Fig. 3a; the review's Table 4, row 'Diamond Skeletal [126]'",
        alabort_notes!(concat!(
            "The primary (read) gives the absolute fit E = 79 870·ρ^2.67 MPa. The review's 0.7 is ",
            "that over an unstated ≈114.1 GPa, and its 2.7 is 2.67 rounded, so the row is the ",
            "primary's fit over the review's divisor: C = 0.70 (79 870 / 114 100) with the ",
            "primary's n = 2.67. The review's rounding read 3.5% low at 0.30 and 6.7% low at 0.10. ",
            "The divisor is the review's, not stated by it. Skeletal is confirmed in the primary."
        )),
    ),
    alabort19(
        "alabort19-d-solid-strength",
        64,
        SCHWARZ_D,
        STRENGTH,
        Fit::Power { c: 1.17, n: 2.61 },
        903.0,
        "the primary's Fig. 3b; the review's Table 4, row 'Diamond Skeletal [126]'",
        alabort_notes!(concat!(
            "The primary (read) gives σ = 1057·ρ^2.61 MPa, so the review's 1.17 implies σ_s ≈ 903 ",
            "MPa, not stated by it; the primary's n = 2.61 replaces the review's 2.6. The fit ",
            "passes ρ above 0.91, and is refused there."
        )),
    ),
    quoted(
        "polley22-g-sheet-e",
        65,
        GYROID,
        Banding::Sheet,
        MODULUS,
        Fit::Power { c: 0.2, n: 1.2 },
        TITANIUM,
        "Polley 2022 via Distefano 2023",
        distefano_citation!("Polley C. et al. (2022) J. Mech. Behav. Biomed. Mater. 125:104882"),
        "Table 4, row 'Gyroid Sheet [127]'; Eq (5)",
        distefano_notes!("'Sheet' is the review's label."),
    ),
    quoted(
        "polley22-g-sheet-strength",
        66,
        GYROID,
        Banding::Sheet,
        STRENGTH,
        Fit::Power { c: 0.67, n: 1.3 },
        TITANIUM,
        "Polley 2022 via Distefano 2023",
        distefano_citation!("Polley C. et al. (2022) J. Mech. Behav. Biomed. Mater. 125:104882"),
        "Table 4, row 'Gyroid Sheet [127]'; Eq (6)",
        distefano_notes!("'Sheet' is the review's label."),
    ),
    quoted(
        "sun22-g-sheet-e",
        67,
        GYROID,
        Banding::Sheet,
        MODULUS,
        Fit::Power { c: 0.12, n: 1.1 },
        TITANIUM,
        "Sun 2022 via Distefano 2023",
        distefano_citation!("Sun Q., Sun J., Guo K., Wang L. (2022) Mech. Mater. 166:104241"),
        "Table 4, row 'Gyroid Sheet [123]'; Eq (5)",
        distefano_notes!("Tagged Sheet by the review."),
    ),
    quoted(
        "sun22-g-sheet-strength",
        68,
        GYROID,
        Banding::Sheet,
        STRENGTH,
        Fit::Power { c: 2.07, n: 2.03 },
        TITANIUM,
        "Sun 2022 via Distefano 2023",
        distefano_citation!("Sun Q., Sun J., Guo K., Wang L. (2022) Mech. Mater. 166:104241"),
        "Table 4, row 'Gyroid Sheet [123]'; Eq (6)",
        distefano_notes!(
            "C > 1 as printed. The fit passes ρ above 0.49 and reads 133% of the solid at the \
             gyroid sheet's wall limit (fill 0.803); refused above 0.49."
        ),
    ),
    quoted(
        "yan15-g-solid-e",
        69,
        GYROID,
        Banding::Solid,
        MODULUS,
        Fit::Power { c: 0.19, n: 1.71 },
        TITANIUM,
        "Yan 2015 via Distefano 2023",
        distefano_citation!(
            "Yan C., Hao L., Hussein A., Young P. (2015) J. Mech. Behav. Biomed. Mater. 51:61–73"
        ),
        "Table 4, row 'Gyroid Skeletal [15]'; Eq (5)",
        distefano_notes!("'Skeletal' is the network solid."),
    ),
    quoted(
        "yan15-g-solid-strength",
        70,
        GYROID,
        Banding::Solid,
        STRENGTH,
        Fit::Power { c: 1.31, n: 1.83 },
        TITANIUM,
        "Yan 2015 via Distefano 2023",
        distefano_citation!(
            "Yan C., Hao L., Hussein A., Young P. (2015) J. Mech. Behav. Biomed. Mater. 51:61–73"
        ),
        "Table 4, row 'Gyroid Skeletal [15]'; Eq (6)",
        distefano_notes!("The fit passes ρ above 0.72, and is refused there."),
    ),
    quoted(
        "ataee18-g-solid-e",
        71,
        GYROID,
        Banding::Solid,
        MODULUS,
        Fit::Power { c: 0.29, n: 2.0 },
        "titanium; alloy not verified (possibly commercially pure)",
        "Ataee 2018 via Distefano 2023",
        distefano_citation!("Ataee A., Li Y., Brandt M., Wen C. (2018) Acta Mater. 158:354–368"),
        "Table 4, row 'Gyroid Skeletal [128]'; Eq (5)",
        distefano_notes!(
            "Whether n = 2 was fitted is not stated. The alloy is specifically unverified: it may \
             be commercially pure titanium."
        ),
    ),
    quoted(
        "ataee18-g-solid-strength",
        72,
        GYROID,
        Banding::Solid,
        STRENGTH,
        Fit::Power { c: 0.46, n: 1.5 },
        "titanium; alloy not verified (possibly commercially pure)",
        "Ataee 2018 via Distefano 2023",
        distefano_citation!("Ataee A., Li Y., Brandt M., Wen C. (2018) Acta Mater. 158:354–368"),
        "Table 4, row 'Gyroid Skeletal [128]'; Eq (6)",
        distefano_notes!(
            "The alloy is specifically unverified: it may be commercially pure titanium."
        ),
    ),
    alabort19(
        "alabort19-neovius-solid-e",
        73,
        NEOVIUS,
        MODULUS,
        Fit::Power { c: 0.31, n: 2.28 },
        112_900.0,
        "the primary's text and Fig. 3a; the review's Table 4, row 'Neovius [126]'",
        alabort_notes!(concat!(
            "Solid (skeletal) per the primary's text, against the row's 'not stated' and the ",
            "review's 'sheet-only'. The primary gives E = 35 013·ρ^2.28 MPa, so the review's 0.31 ",
            "implies a divisor of ≈112.9 GPa, not stated by it; the primary's n = 2.28 replaces the ",
            "review's 2.3."
        )),
    ),
    alabort19(
        "alabort19-neovius-solid-strength",
        74,
        NEOVIUS,
        STRENGTH,
        Fit::Power { c: 1.43, n: 2.89 },
        897.0,
        "the primary's text and Fig. 3b; the review's Table 4, row 'Neovius [126]'",
        alabort_notes!(concat!(
            "Solid (skeletal) per the primary's text. The primary gives σ = 1282·ρ^2.89 MPa, so the ",
            "implied σ_s is ≈897 MPa; the primary's n = 2.89 replaces the review's 2.9. The fit ",
            "passes ρ above 0.83, and is refused there."
        )),
    ),
];

// The rows other rows are written as differences from, each a row of `ROWS` in its own right.
// Named so that `..ROW_X` reads as "as that row, except", which is how the research records them.
const ROW_NX23_P: PropertyRow = PropertyRow {
    id: "nx23-p-sheet-e",
    ledger_row: 0,
    cell: SCHWARZ_P,
    phase: Banding::Sheet,
    property: MODULUS,
    direction: LoadDirection::NotStated,
    // The upper branch from the paper's continuity condition and E*(1) = 1.
    fit: Fit::TwoBranch {
        c1: 0.317,
        n1: 1.264,
        knee: 0.25,
        c2: 1.007_485_139_658_782_7,
        n2: 2.006,
        c3: -0.007_485_139_658_782_725,
    },
    exponent: Exponent::Fitted,
    range: NX23_RANGE,
    density_basis: DensityBasis::Design,
    data: DataKind::FitToFe,
    provenance: Provenance::Primary,
    material: NX23_MATERIAL,
    solid_basis: SolidBasis::ModelInput,
    solid_value_mpa: None,
    source: NX23,
    notes: concat!(
        "The upper branch is recomputed from the paper's continuity condition and E*(1) = 1 ",
        "(C2 1.0075, C3 −0.0075). The fit is to FE homogenisation from two earlier studies, ",
        "0–0.15 and 0.15–0.80; the first's lowest density is not stated, so the range does not ",
        "claim nought and the line warns below 0.15. No loading direction is stated."
    ),
};
const ROW_NAGHAVI22_G: PropertyRow = PropertyRow {
    id: "naghavi22-g-sheet-e",
    ledger_row: 3,
    cell: GYROID,
    phase: Banding::Sheet,
    property: MODULUS,
    direction: LoadDirection::NotStated,
    fit: Fit::Power { c: 0.9, n: 1.58 },
    exponent: Exponent::Fitted,
    range: DensityRange::Span { lo: Lower::At(0.28), hi: 0.46 },
    density_basis: DensityBasis::Design,
    data: DataKind::Fe,
    provenance: Provenance::Primary,
    material: NAGHAVI22_MATERIAL,
    solid_basis: SolidBasis::ModelInput,
    solid_value_mpa: Some(35_770.0),
    source: NAGHAVI22,
    notes: concat!(
        "The range is the FE's, 0.28–0.46 (about seven FE-extended designs), not three designs ",
        "at 0.33–0.46. The data is FE: the fit reproduces the FE (G800: fit 0.195, FE 0.195, ",
        "test 0.190), so the ratio is the model's, geometry and ν only, and the tests agreed ",
        "with it. The direction is not stated relative to the cell; 'vertical' described the ",
        "platens."
    ),
};
const ROW_LI19_G_SHEET: PropertyRow = PropertyRow {
    id: "li19-g-sheet-e",
    ledger_row: 7,
    cell: GYROID,
    phase: Banding::Sheet,
    property: MODULUS,
    direction: LoadDirection::BuildDirection,
    fit: Fit::Power { c: 0.817, n: 2.0 },
    exponent: Exponent::Assumed,
    range: DensityRange::Single(0.30),
    density_basis: DensityBasis::Design,
    data: DataKind::Experiment,
    provenance: Provenance::Primary,
    material: "SLA resin (Form 2); 850.1 MPa",
    solid_basis: SolidBasis::Stated,
    solid_value_mpa: Some(850.1),
    source: LI19,
    notes: concat!(
        "Compressed along the manufacturing direction (no longer 'not verified'). C is a ",
        "one-point calibration at 0.30 with n = 2 assumed by the authors."
    ),
};
const ROW_AHMADI15_CUBIC: PropertyRow = PropertyRow {
    id: "ahmadi15-cubic-strut-e",
    ledger_row: 33,
    cell: CUBIC,
    phase: Banding::Strut,
    property: MODULUS,
    direction: LoadDirection::NotStated,
    fit: Fit::Power { c: 0.106, n: 0.92 },
    exponent: Exponent::Fitted,
    range: DensityRange::Span { lo: Lower::At(0.13), hi: 0.37 },
    density_basis: DensityBasis::Measured,
    data: DataKind::Experiment,
    provenance: Provenance::Primary,
    material: "Ti-6Al-4V ELI, SLM; the solid's elastic gradient, measured",
    solid_basis: SolidBasis::Measured,
    solid_value_mpa: None,
    source: Source {
        short: "Ahmadi 2015",
        citation: AHMADI15_CITATION,
        locator: "Fig. 9C",
        link: "doi:10.3390/ma8041871",
    },
    notes: concat!(
        "The range is the fit's own four points, read off Fig. 9C at about 0.13, 0.24, 0.28 ",
        "and 0.37 on its axis (structure relative density), so both ends share one basis. ",
        "Fitted on measured density. No loading axis stated."
    ),
};
const ROW_MASKERY17_E: PropertyRow = PropertyRow {
    id: "maskery17-bcc-strut-e",
    ledger_row: 40,
    cell: BCC,
    phase: Banding::Strut,
    property: MODULUS,
    direction: LoadDirection::BuildDirection,
    fit: Fit::Power { c: 0.44, n: 2.0 },
    exponent: Exponent::Assumed,
    range: DensityRange::Single(0.19),
    density_basis: DensityBasis::NotStated,
    data: DataKind::Experiment,
    provenance: Provenance::Primary,
    material: "PA2200, SLS; 741 MPa",
    solid_basis: SolidBasis::Stated,
    solid_value_mpa: Some(741.0),
    source: MASKERY17,
    notes: "One density, n = 2 assumed by the authors; loaded along z, the build direction.",
};
const ROW_CELLUCCI16: PropertyRow = PropertyRow {
    id: "cellucci16-octet-strut-e100",
    ledger_row: 42,
    cell: OCTET,
    phase: Banding::Strut,
    property: MODULUS,
    direction: LoadDirection::Axis100,
    fit: Fit::Power { c: 1.0 / 9.0, n: 1.0 },
    exponent: Exponent::Theory,
    range: DensityRange::Span { lo: Lower::At(0.001), hi: 0.1 },
    density_basis: DensityBasis::Design,
    data: DataKind::Fe,
    provenance: Provenance::Primary,
    material: "Timoshenko-beam unit cell, periodic boundaries; any isotropic solid",
    solid_basis: SolidBasis::ModelInput,
    solid_value_mpa: None,
    source: Source {
        short: "Cellucci 2016",
        citation: "Cellucci D., Cheung K.C. (2016) Evaluation of Cellular Solids Derived from \
                   Triply Periodic Minimal Surfaces. ASME MSEC2016-8849, arXiv:1512.07512",
        locator: "Eq 5a (S11)",
        link: "https://arxiv.org/abs/1512.07512",
    },
    notes: concat!(
        "FE with beam elements, so the data phrase on hover is 'FE (beam elements)'. The ",
        "paper's 'D-Schwarz' and 'P-Schwarz' are beam networks, not the surfaces FieldForm ",
        "draws, so only its octet ships."
    ),
};

// ---------------------------------------------------------------------------------------------
// The 27 that do not ship.
// ---------------------------------------------------------------------------------------------

macro_rules! maskery18a {
    ($c:literal) => {
        concat!(
            "As reproduced in Lu 2020's Table 3 (", $c, "). The cited paper (Polymer 152:62–71, ",
            "read in full) contains none of these numbers: it tested one density (0.294) with n ",
            "imposed. The true origin (Maskery 2018b) could not be fetched, and a number whose ",
            "cited source does not contain it does not ship."
        )
    };
}

macro_rules! abou_ali {
    ($c:literal) => {
        concat!(
            "C is in absolute MPa (", $c, ") and the paper never reports the solid PA11's modulus ",
            "or yield, so no ratio to the solid can be formed from it at all. A coefficient in ",
            "absolute units with no recoverable divisor is excluded."
        )
    };
}

macro_rules! abou_ali_sheet {
    ($c:literal) => {
        concat!(
            abou_ali!($c),
            " The sheet rows were also fitted on measured density inflated by trapped powder ",
            "(designed 0.084–0.209, measured 0.242–0.342), so not even their exponents are scaling ",
            "exponents."
        )
    };
}

macro_rules! libonati {
    ($c:literal, $plot:literal) => {
        concat!(
            "The printed C (", $c, ") is not reproduced by the paper's own Fig. 7b (about ", $plot,
            " read off it); C > 1 at n ≈ 1 breaks E*/E_s ≤ ρ; the modulus is over a reduced strut ",
            "modulus; and the paper contradicts itself on its density sweep. A fit that breaks the ",
            "rule of mixtures in its own range, or disagrees with its own plot, does not ship."
        )
    };
}

const ABOU_ALI: &str = "Abou-Ali, Lee, Abu Al-Rub 2022, Polymers 14(21):4583, Table 5";
const MASKERY18A: &str = "Maskery 2018a as reproduced in Lu 2020, Table 3";
const LIBONATI: &str = "Libonati et al. 2021, ACS Biomater. Sci. Eng., Eqs 2–4";

const fn excluded(
    ledger_row: u16,
    cell: CellKind,
    phase: Option<Banding>,
    class: PropertyClass,
    source: &'static str,
    reason: &'static str,
) -> ExcludedRow {
    ExcludedRow { ledger_row, cell: Some(cell), phase, class, source, reason }
}

pub(crate) use PropertyClass::{Modulus as E, Strength as S};

/// The 27 researched rows that do not ship, each with the reason.
pub static EXCLUDED: &[ExcludedRow] = &[
    excluded(16, SCHWARZ_D, Some(Banding::Solid), E, MASKERY18A, maskery18a!("C 0.750, n 2.102")),
    excluded(17, GYROID, Some(Banding::Solid), E, MASKERY18A, maskery18a!("C 1.020, n 2.405")),
    excluded(18, SCHWARZ_P, Some(Banding::Solid), E, MASKERY18A, maskery18a!("C 0.920, n 1.001")),
    excluded(19, SCHWARZ_D, Some(Banding::Solid), E, ABOU_ALI, abou_ali!("720.7, n 2.04")),
    excluded(20, SCHWARZ_D, Some(Banding::Solid), S, ABOU_ALI, abou_ali!("52.4, n 2.46")),
    excluded(21, SCHOEN_IWP, Some(Banding::Solid), E, ABOU_ALI, abou_ali!("1070.7, n 2.54")),
    excluded(22, SCHOEN_IWP, Some(Banding::Solid), S, ABOU_ALI, abou_ali!("27.2, n 2.31")),
    excluded(23, GYROID, Some(Banding::Solid), E, ABOU_ALI, abou_ali!("1362.6, n 2.40")),
    excluded(24, GYROID, Some(Banding::Solid), S, ABOU_ALI, abou_ali!("26.3, n 2.05")),
    excluded(25, SCHWARZ_D, Some(Banding::Sheet), E, ABOU_ALI, abou_ali_sheet!("225.5, n 0.75")),
    excluded(26, SCHWARZ_D, Some(Banding::Sheet), S, ABOU_ALI, abou_ali_sheet!("17.4, n 1.52")),
    excluded(27, SCHOEN_IWP, Some(Banding::Sheet), E, ABOU_ALI, abou_ali_sheet!("232.8, n 0.93")),
    excluded(28, SCHOEN_IWP, Some(Banding::Sheet), S, ABOU_ALI, abou_ali_sheet!("17.4, n 1.57")),
    excluded(29, GYROID, Some(Banding::Sheet), E, ABOU_ALI, abou_ali_sheet!("329.7, n 1.08")),
    excluded(30, GYROID, Some(Banding::Sheet), S, ABOU_ALI, abou_ali_sheet!("8.9, n 1.01")),
    excluded(31, OCTET, Some(Banding::Strut), E, ABOU_ALI, abou_ali!("79.0, n 0.74")),
    excluded(32, OCTET, Some(Banding::Strut), S, ABOU_ALI, abou_ali!("4.5, n 0.91")),
    excluded(
        35,
        CUBIC,
        Some(Banding::Strut),
        S,
        "Ahmadi et al. 2015, Materials 8(4):1871, Fig. 12C",
        "A yield stress in MPa (C 349, n 1.21), and the paper gives no bulk yield to divide by.",
    ),
    excluded(
        36,
        DIAMOND,
        Some(Banding::Strut),
        S,
        "Ahmadi et al. 2015, Materials 8(4):1871, Fig. 12D",
        "A yield stress in MPa (C 411, n 1.83), and the paper gives no bulk yield to divide by.",
    ),
    excluded(37, CUBIC, Some(Banding::Strut), E, LIBONATI, libonati!("2.3, n 1.02", "0.36")),
    excluded(38, BCC, Some(Banding::Strut), E, LIBONATI, libonati!("1.5, n 1.06", "0.18")),
    excluded(
        39,
        FCC,
        Some(Banding::Strut),
        E,
        LIBONATI,
        concat!(libonati!("1.6, n 1.05", "0.22"), " FCC is left with no row."),
    ),
    excluded(
        46,
        VORONOI_BEAM,
        Some(Banding::Strut),
        E,
        "Roberts & Garboczi 2002, J. Mech. Phys. Solids 50(1):33–55, Eq 8",
        concat!(
            "Verified (C 0.930, n 2.04), but the source's tessellation is random and checked ",
            "isotropic, while FieldForm's jittered grid keeps cubic order below full jitter (at ",
            "jitter 0 its beams sit on cube edges); and a Voronoi block's density depends on its ",
            "jitter and seed, with no method to read it until item 88's sampler. Revisited with ",
            "item 88."
        ),
    ),
    excluded(
        51,
        BCC,
        Some(Banding::Strut),
        E,
        "Bai et al. 2021, Mater. Des. 211:110140, via Distefano 2023, Table 4",
        concat!(
            "Verified as tabulated (C 0.15, n 2), but the primary studies curved-strut BCC ",
            "variants and which one these belong to cannot be checked; the review misattributes ",
            "elsewhere in the same table. A row ships only if its cell is tied to FieldForm's by ",
            "something read, and two other BCC rows exist."
        ),
    ),
    excluded(
        52,
        BCC,
        Some(Banding::Strut),
        S,
        "Bai et al. 2021, Mater. Des. 211:110140, via Distefano 2023, Table 4",
        concat!(
            "Verified as tabulated (C 0.23, n 1.5), but the primary studies curved-strut BCC ",
            "variants and which one these belong to cannot be checked; the review misattributes ",
            "elsewhere in the same table. A row ships only if its cell is tied to FieldForm's by ",
            "something read."
        ),
    ),
    excluded(
        57,
        SCHWARZ_P,
        None,
        E,
        "Sokollu et al. 2022, Addit. Manuf. 60:103199, via Distefano 2023, Table 4",
        concat!(
            "Verified as tabulated (C 1.38, n 2), but nothing reachable says sheet or network ",
            "solid, unlike Sun 2022, whose own D and G rows are tagged Sheet; and the fit passes ",
            "the rule of mixtures above 0.725. A row whose phase is unknown is attached to ",
            "neither phase."
        ),
    ),
    excluded(
        58,
        SCHWARZ_P,
        None,
        S,
        "Sokollu et al. 2022, Addit. Manuf. 60:103199, via Distefano 2023, Table 4",
        concat!(
            "Verified as tabulated (C 0.98, n 1.5), but nothing reachable says sheet or network ",
            "solid. A row whose phase is unknown is attached to neither phase."
        ),
    ),
];
