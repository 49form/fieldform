// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The minimal-surface cells: each one's trig sum, its gradient bounds, and its measured tables.

use fieldform_scene::Banding;
use nalgebra::{Point3, Vector3};
#[cfg(doc)] use super::r3_point;

/// Which triply periodic minimal surface.
///
/// One enum rather than three node kinds: all three share the whole evaluation shape — scale
/// the point by the frequency, evaluate a trig sum, band it by an isovalue — and differ only in
/// the sum. Written so a fourth is a variant and a formula, which is what makes this the
/// canonical example of D5's plugin surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TpmsKind {
    Gyroid,
    SchwarzP,
    SchwarzD,
    SchoenIwp,
    Neovius,
    FischerKochS,
}

impl From<fieldform_scene::LatticeCell> for TpmsKind {
    /// The scene names the cell and this crate does its arithmetic, the same split
    /// `Continuity` and `BooleanOp` already live under. Two enums, but they cannot drift: the
    /// match is total, so adding a cell to the vocabulary without giving it a formula does not
    /// compile.
    fn from(cell: fieldform_scene::LatticeCell) -> Self {
        match cell {
            fieldform_scene::LatticeCell::Gyroid => TpmsKind::Gyroid,
            fieldform_scene::LatticeCell::SchwarzP => TpmsKind::SchwarzP,
            fieldform_scene::LatticeCell::SchwarzD => TpmsKind::SchwarzD,
            fieldform_scene::LatticeCell::SchoenIwp => TpmsKind::SchoenIwp,
            fieldform_scene::LatticeCell::Neovius => TpmsKind::Neovius,
            fieldform_scene::LatticeCell::FischerKochS => TpmsKind::FischerKochS,
            // Not surfaces at all, and there is no sensible TPMS to stand in for one. The
            // conversion that matters is `CellKind::from`, which routes these to `BeamKind`
            // before this impl is ever reached; this arm exists because the match is total
            // and is the loudest thing that can go here.
            cell @ (fieldform_scene::LatticeCell::Cubic
            | fieldform_scene::LatticeCell::Bcc
            | fieldform_scene::LatticeCell::Fcc
            | fieldform_scene::LatticeCell::Octet
            | fieldform_scene::LatticeCell::Diamond
            | fieldform_scene::LatticeCell::Honeycomb
            | fieldform_scene::LatticeCell::VoronoiFoam { .. }
            | fieldform_scene::LatticeCell::VoronoiBeam { .. }) => {
                panic!("{cell:?} is not a surface — convert through CellKind")
            }
        }
    }
}

impl TpmsKind {
    pub const ALL: [TpmsKind; 6] = [
        TpmsKind::Gyroid,
        TpmsKind::SchwarzP,
        TpmsKind::SchwarzD,
        TpmsKind::SchoenIwp,
        TpmsKind::Neovius,
        TpmsKind::FischerKochS,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TpmsKind::Gyroid => "Gyroid",
            TpmsKind::SchwarzP => "Schwarz P",
            TpmsKind::SchwarzD => "Schwarz D",
            TpmsKind::SchoenIwp => "Schoen IWP",
            TpmsKind::Neovius => "Neovius",
            TpmsKind::FischerKochS => "Fischer-Koch S",
        }
    }

    /// The surface's defining sum, at unit frequency.
    ///
    /// The standard trig approximations, not exact Weierstrass minimal surfaces — the ones
    /// every tool that ships these actually uses.
    pub fn f(self, p: Point3<f64>) -> f64 {
        let (x, y, z) = (p.x, p.y, p.z);
        match self {
            TpmsKind::Gyroid => x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos(),
            TpmsKind::SchwarzP => x.cos() + y.cos() + z.cos(),
            // 2(cos x cos y + cos y cos z + cos z cos x) - (cos 2x + cos 2y + cos 2z).
            TpmsKind::SchoenIwp => {
                2.0 * (x.cos() * y.cos() + y.cos() * z.cos() + z.cos() * x.cos())
                    - ((2.0 * x).cos() + (2.0 * y).cos() + (2.0 * z).cos())
            }
            // The factored form is kept rather than expanded: it is cheaper, and it is the
            // reason the bound below is 7 rather than an argument about six separate terms.
            TpmsKind::Neovius => {
                3.0 * (x.cos() + y.cos() + z.cos()) + 4.0 * x.cos() * y.cos() * z.cos()
            }
            TpmsKind::FischerKochS => {
                (2.0 * x).cos() * y.sin() * z.cos()
                    + x.cos() * (2.0 * y).cos() * z.sin()
                    + x.sin() * y.cos() * (2.0 * z).cos()
            }
            TpmsKind::SchwarzD => {
                x.sin() * y.sin() * z.sin()
                    + x.sin() * y.cos() * z.cos()
                    + x.cos() * y.sin() * z.cos()
                    + x.cos() * y.cos() * z.sin()
            }
        }
    }

    /// `∇F` at unit frequency, analytically.
    pub fn gradient(self, p: Point3<f64>) -> Vector3<f64> {
        let (x, y, z) = (p.x, p.y, p.z);
        match self {
            TpmsKind::Gyroid => Vector3::new(
                x.cos() * y.cos() - z.sin() * x.sin(),
                y.cos() * z.cos() - x.sin() * y.sin(),
                z.cos() * x.cos() - y.sin() * z.sin(),
            ),
            TpmsKind::SchwarzP => Vector3::new(-x.sin(), -y.sin(), -z.sin()),
            TpmsKind::SchoenIwp => Vector3::new(
                -2.0 * x.sin() * (y.cos() + z.cos()) + 2.0 * (2.0 * x).sin(),
                -2.0 * y.sin() * (z.cos() + x.cos()) + 2.0 * (2.0 * y).sin(),
                -2.0 * z.sin() * (x.cos() + y.cos()) + 2.0 * (2.0 * z).sin(),
            ),
            TpmsKind::Neovius => Vector3::new(
                -x.sin() * (3.0 + 4.0 * y.cos() * z.cos()),
                -y.sin() * (3.0 + 4.0 * z.cos() * x.cos()),
                -z.sin() * (3.0 + 4.0 * x.cos() * y.cos()),
            ),
            // Cyclic, not fully symmetric, so each partial is written out: the three are the
            // same expression under x -> y -> z -> x, and reading them as such is the check.
            TpmsKind::FischerKochS => Vector3::new(
                -2.0 * (2.0 * x).sin() * y.sin() * z.cos()
                    - x.sin() * (2.0 * y).cos() * z.sin()
                    + x.cos() * y.cos() * (2.0 * z).cos(),
                (2.0 * x).cos() * y.cos() * z.cos()
                    - 2.0 * (2.0 * y).sin() * x.cos() * z.sin()
                    - x.sin() * y.sin() * (2.0 * z).cos(),
                -(2.0 * x).cos() * y.sin() * z.sin() + x.cos() * (2.0 * y).cos() * z.cos()
                    - 2.0 * (2.0 * z).sin() * x.sin() * y.cos(),
            ),
            TpmsKind::SchwarzD => Vector3::new(
                x.cos() * (y.sin() * z.sin() + y.cos() * z.cos())
                    - x.sin() * (y.sin() * z.cos() + y.cos() * z.sin()),
                y.cos() * (x.sin() * z.sin() + x.cos() * z.cos())
                    - y.sin() * (x.sin() * z.cos() + x.cos() * z.sin()),
                z.cos() * (x.sin() * y.sin() + x.cos() * y.cos())
                    - z.sin() * (x.sin() * y.cos() + x.cos() * y.sin()),
            ),
        }
    }

    /// A proven upper bound on `|∇F|` at unit frequency.
    ///
    /// Safety depends on this being a true bound, so each is argued rather than fitted, and
    /// `tpms_profile.rs` checks numerically that no sample exceeds it.
    ///
    /// - **Gyroid**: `√3`, exactly, and it attains it. Each partial is an inner product with a
    ///   *unit* vector — `gx = cos a cos b − sin a sin c = ⟨(cos a, −sin a), (cos b, sin c)⟩` —
    ///   so Cauchy–Schwarz gives `|gx| ≤ √(cos²b + sin²c)`, and cyclically
    ///   `|gy| ≤ √(cos²c + sin²a)` and `|gz| ≤ √(cos²a + sin²b)`. Summing the squares pairs
    ///   every `cos²` with its own `sin²`:
    ///
    ///   ```text
    ///   |∇F|² ≤ (cos²a + sin²a) + (cos²b + sin²b) + (cos²c + sin²c) = 3
    ///   ```
    ///
    ///   Attained at the origin, where all three partials are 1 — which is the maximum
    ///   `tpms_profile.rs` measures, so the proof and the sampling agree to six figures.
    ///
    ///   The earlier argument bounded each partial separately at `√2` and multiplied up to
    ///   `√6`. That is true and 41% loose: the three partials share `a`, `b` and `c`, so they
    ///   cannot all reach `√2` at once, and it is the *sum* that has to be bounded rather than
    ///   each term on its own.
    /// - **Schwarz P**: the gradient is `(−sin x, −sin y, −sin z)`, so it is bounded by `√3`
    ///   exactly, and attains it.
    /// - **Schwarz D**: `√3`, by the same Cauchy–Schwarz step applied *twice*. Each partial
    ///   factors two ways — group by its own variable or by the third:
    ///
    ///   ```text
    ///   gx = cos x · cos(y−z) − sin x · sin(y+z)
    ///      = cos z · cos(x+y) − sin z · sin(x−y)
    ///   ```
    ///
    ///   Each gives an upper bound on the same `|∇F|²`, and summing the three terms of each
    ///   with `cos²u = (1+cos2u)/2` leaves
    ///
    ///   ```text
    ///   B₁ = 3 + S,   B₂ = 3 − S,   S = sin2x·sin2y + sin2y·sin2z + sin2z·sin2x
    ///   ```
    ///
    ///   so `B₁ + B₂ = 6`. Neither alone is enough — `S` reaches ±1, so each is only a bound of
    ///   4 — but both bound the same quantity, so `|∇F|² ≤ min(B₁, B₂) = 3 − |S| ≤ 3`.
    ///
    /// **All three are now proved and all three are attained**, so `tpms_profile.rs` reports no
    /// headroom on any of them — which is what a bound that is doing its job looks like. Every
    /// unit of slack in one is a proportional slowdown, since it divides every step the tracer
    /// takes, and the gyroid carried 41% of it for as long as the loose argument stood.
    ///
    /// The rule that got them here is worth keeping: **prove it before tightening it.** A grid
    /// showing that a maximum has not been *reached* is not a demonstration that it cannot be,
    /// and this constant is the one place where being wrong means stepping through a wall.
    pub fn gradient_bound(self) -> f64 {
        match self {
            // All three proved above, and each attains it.
            TpmsKind::Gyroid | TpmsKind::SchwarzP | TpmsKind::SchwarzD => 3f64.sqrt(),
            // The three below are bounded **termwise**, which is sound and loose. Each partial
            // is bounded by the sum of the magnitudes of its terms, and the vector by `√3` times
            // that — an argument that never asks whether the terms can be extreme together, and
            // they cannot. So these carry slack where the first three carry none, and
            // `tpms_profile.rs` prints the cost ratio that says how much.
            //
            // **Shipped loose on purpose.** The gyroid carried 41% of slack for months under a
            // `√6` argument until Cauchy–Schwarz tightened it to `√3`, and the rule that got it
            // there is the rule here: *prove it before tightening it*. A grid showing a maximum
            // has not been reached is not a demonstration that it cannot be, and this constant
            // is the one place where being wrong means a ray through a wall. Slack is a
            // proportional slowdown; a wrong bound is a hole.

            // |Fx| <= 2·(1 + 1) + 2 = 6, termwise.
            TpmsKind::SchoenIwp => 6.0 * 3f64.sqrt(),
            // |Fx| = |sin x|·|3 + 4 cos y cos z| <= 1 · 7. The factored form is what makes this
            // a single product to bound rather than four terms to add up.
            TpmsKind::Neovius => 7.0 * 3f64.sqrt(),
            // |Fx| <= 2 + 1 + 1 = 4, termwise.
            TpmsKind::FischerKochS => 4.0 * 3f64.sqrt(),
        }
    }

    /// Typical `|∇F|` on the surface at unit frequency, for converting a wall thickness into a
    /// threshold on `|F|`.
    ///
    /// **Measured, not derived** — it is the mean over the surface, and the gradient varies, so
    /// the thickness it produces is nominal rather than exact. That error is the same one D8
    /// names for distance and it goes away for free when local-gradient normalization lands.
    /// `tpms_profile.rs` prints both the mean these come from and the spread around it.
    ///
    /// The mean is measured by `tpms_profile.rs`. What it *costs* is measured the honest way by
    /// `a_wall_comes_out_within_the_band_its_doc_comment_promises`: from thousands of points
    /// inside the wall, march across it along the gradient both ways and take the chord. That
    /// test asserts the two right-hand columns, so this table cannot go stale unread.
    ///
    /// | kind | mean \|∇F\| | 2 mm asked → 10th–90th | median |
    /// |---|---|---|---|
    /// | Gyroid | 1.530 | 1.87 – 2.18 mm | 2.09 |
    /// | Schwarz P | 1.321 | 1.71 – 2.46 mm | 2.13 |
    /// | Schwarz D | 1.486 | 1.84 – 2.18 mm | 2.09 |
    /// | Schoen IWP | 4.328 | 1.86 – 2.51 mm | 2.05 |
    /// | Neovius | 2.401 | **1.36 – 6.23 mm** | 2.21 |
    /// | Fischer-Koch S | 1.759 | 1.65 – 2.73 mm | 2.25 |
    ///
    /// **Two things this table says that the previous one did not.** The earlier bands were a
    /// tangent estimate — `2·iso / |∇F|` at each surface point — which is what `tpms_profile.rs`
    /// still prints, and it understates wherever `F` is flat and curves back rather than
    /// continuing straight. For the three original cells the two methods agree within a few per
    /// cent. For the three added in item 42 they do not: Neovius's tangent estimate said a 2 mm
    /// wall reaches 3.64 mm, and the chord says **6.23** — more than three times what was typed,
    /// in the flat regions where its gradient collapses. The old rows were wrong, and the test
    /// that replaced them is what found it.
    ///
    /// And **every median sits above the number asked for**, by 2% (IWP) to 13% (Fischer-Koch S).
    /// The conversion divides by the *mean* gradient, and every one of these surfaces has a
    /// right-skewed gradient — a steep tail pulls the mean above the typical value — so the
    /// typical wall comes out fat. That is inherent to converting one number through an average,
    /// not a defect in the constant; it is what "nominal" means here, and it is the cell where
    /// local-gradient normalization (D8) would buy the most.
    pub fn surface_gradient(self) -> f64 {
        match self {
            TpmsKind::Gyroid => 1.530,
            TpmsKind::SchwarzP => 1.321,
            TpmsKind::SchwarzD => 1.486,
            TpmsKind::SchoenIwp => 4.328,
            TpmsKind::Neovius => 2.401,
            TpmsKind::FischerKochS => 1.759,
        }
    }

    /// The thickest a wall can be, as a fraction of its cell, and still be a wall.
    ///
    /// Past it the wall has met its neighbour: a march across it finds no far side, or finds
    /// one more than twice the thickness away. Measured by
    /// `a_wall_past_its_cells_limit_has_met_its_neighbours` as the largest ratio at which no
    /// more than a tenth of the probes read that way, and asserted there, so the panel's
    /// warning cannot drift from the geometry:
    ///
    /// | kind | limit | just past it |
    /// |---|---|---|
    /// | Gyroid | 25% | 92% merged at 30% |
    /// | Schwarz P | 22% | 22% at 25% |
    /// | Schwarz D | 20% | 35% at 22% |
    /// | Schoen IWP | 18% | 22% at 20% |
    /// | Fischer-Koch S | 10% | 17% at 12% |
    /// | Neovius | 6% | 30% at 8% |
    ///
    /// Found when a graded Fischer-Koch S "went solid" and a typed wall of the same thickness
    /// turned out to have done the same (`graded-lattice-spec.md` section 6). The limit is the
    /// cell's, typed or graded, and nothing said so where the number is typed.
    pub fn wall_limit(self) -> f64 {
        match self {
            TpmsKind::Gyroid => 0.25,
            TpmsKind::SchwarzP => 0.22,
            TpmsKind::SchwarzD => 0.20,
            TpmsKind::SchoenIwp => 0.18,
            TpmsKind::FischerKochS => 0.10,
            TpmsKind::Neovius => 0.06,
        }
    }

    /// How far a sheet's wall actually comes out from the number typed: the 10th and 90th
    /// percentile of the measured chord, over the thickness asked for.
    ///
    /// The right-hand column of [`Self::surface_gradient`]'s table, as ratios, so that the one
    /// set of numbers is read twice: the panel says it where the wall is typed, and
    /// `a_wall_comes_out_within_the_band_its_doc_comment_promises` holds it to the code. The
    /// panel used to say "roughly a tenth either way" for every cell, which these numbers make
    /// true of two of the six -- and an approximation announced inaccurately is the thing the
    /// announcing rule exists to prevent.
    ///
    /// Measured at a wall a tenth of the cell size. A wall thicker than that for its cell
    /// reaches further into the flat regions and varies more.
    pub fn wall_band(self) -> (f64, f64) {
        match self {
            TpmsKind::Gyroid => (0.935, 1.09),
            TpmsKind::SchwarzP => (0.855, 1.23),
            TpmsKind::SchwarzD => (0.92, 1.09),
            TpmsKind::SchoenIwp => (0.93, 1.255),
            TpmsKind::Neovius => (0.68, 3.115),
            TpmsKind::FischerKochS => (0.825, 1.365),
        }
    }

    /// Per-argument RMS gradient magnitude at unit frequency, as a vector.
    ///
    /// What a cell map needs in order to size walls under a coordinate map that does *not*
    /// scale all three axes equally — a cylindrical map, where the circumferential frequency
    /// varies with radius. `Lattice::sheet_scale` is the consumer.
    ///
    /// **A vector rather than a scalar, though all three components are equal here.** For these
    /// three surfaces they are equal for a nameable reason: Schwarz P and Schwarz D are fully
    /// symmetric under any permutation of their arguments, and the gyroid is cyclically
    /// symmetric, so the partial derivatives have identical distributions. That is a property
    /// of *these cells* and not of unit cells generally — a beam cell with different struts per
    /// axis has direction-dependent gradient statistics, and a scalar would silently mis-size
    /// its walls. The vector costs nothing and is right for both.
    ///
    /// **The `1/√3` is the whole of the convention, and getting it wrong scales every wall in
    /// the model by that factor.** [`TpmsKind::surface_gradient`] is the magnitude of the whole
    /// gradient vector, `|∇F|`; three equal components of a vector with that magnitude are each
    /// `|∇F|/√3`. The check is that the rectangular case must come back out unchanged: with
    /// every frequency equal to `ω`, `sqrt((ω·g/√3)² × 3)` is `ω·g`, which is exactly the
    /// formula this replaces.
    pub fn argument_gradient(self) -> Vector3<f64> {
        let g = self.surface_gradient() / 3f64.sqrt();
        Vector3::new(g, g, g)
    }

    /// Which bandings this cell can give.
    ///
    /// Every surface here gives both, because a minimal surface has two sides and two
    /// labyrinths. It is stated as a set anyway: the answer stops being "both" at the first
    /// family that is not a minimal surface — a beam cell is struts, so it is [`Banding::Solid`]
    /// only and a sheet of it is not a thing to ask for. A panel that asks this question offers
    /// the right choices for free when that lands, instead of offering one that does not exist.
    pub(crate) fn supported_bandings(self) -> &'static [Banding] {
        match self {
            TpmsKind::Gyroid
            | TpmsKind::SchwarzP
            | TpmsKind::SchwarzD
            | TpmsKind::SchoenIwp
            | TpmsKind::Neovius
            | TpmsKind::FischerKochS => &[Banding::Sheet, Banding::Solid],
        }
    }

    /// The isovalue at which `F <= iso` fills the requested fraction of a cell.
    ///
    /// **Measured, because there is no closed form for any of these surfaces.** The table is
    /// the quantiles of `F` over one full period, sampled by `tpms_profile.rs`, with linear
    /// interpolation between tenths.
    ///
    /// **Five of the six are antisymmetric about zero, and Schoen IWP is not.** For the five,
    /// `F <= 0` is exactly half the cell and the table's two halves are the same numbers negated
    /// — each is a trig sum that negates under a half-period shift. Schoen IWP has no such shift:
    /// its value runs from −5 to +3, and half its volume lies below **0.232**.
    ///
    /// That is the whole justification for the table, and it is worth saying which cell supplied
    /// it, because the spec predicted a different one. `lattice-cell-additions-spec.md` §4 named
    /// *Neovius* as the cell whose labyrinths are unequal; measured, Neovius is antisymmetric
    /// (±13) and Schoen IWP is the odd one. Either way a build that had assumed symmetry from
    /// the first three cells would return a different amount of material than the number typed,
    /// with nothing about the result looking wrong — which is the failure D8 exists to name.
    ///
    /// **What reading through tenths costs**, measured by `tpms_profile.rs` ("Solid density
    /// built") as the density each typed number actually builds, relative to the number typed:
    ///
    /// | kind | 0.03 | 0.05 | 0.08 | 0.15 | 0.95 | worst in 0.10–0.90 |
    /// |---|---|---|---|---|---|---|
    /// | Gyroid | 0.019 (−37%) | 0.043 (−14%) | 0.078 (−3%) | 0.151 (+1%) | 0.957 (+1%) | 0.69% at 0.14 |
    /// | Schwarz P | 0.014 (−55%) | 0.031 (−39%) | 0.067 (−16%) | 0.143 (−5%) | 0.969 (+2%) | 4.6% at 0.15 |
    /// | Schwarz D | 0.012 (−59%) | 0.028 (−44%) | 0.063 (−21%) | 0.150 (0%) | 0.972 (+2%) | 0.36% at 0.20 |
    /// | Schoen IWP | 0.013 (−58%) | 0.029 (−42%) | 0.065 (−19%) | 0.150 (0%) | 0.940 (−1%) | 0.35% at 0.85 |
    /// | Neovius | 0.011 (−63%) | 0.026 (−48%) | 0.063 (−22%) | 0.141 (−6%) | 0.974 (+3%) | 5.9% at 0.15 |
    /// | Fischer-Koch S | 0.012 (−59%) | 0.028 (−44%) | 0.064 (−20%) | 0.149 (−1%) | 0.972 (+2%) | 3.1% at 0.12 |
    ///
    /// **The table this replaced was in absolute percentage points and covered three kinds**
    /// (worst 1.18, 1.94 and 2.20 points; under 0.7 between 10% and 90%), which read as "within
    /// about one per cent" and is not: relative to a small number typed, the first tenth is off by
    /// up to two thirds. At the tenths themselves typed and built agree; between them the fill
    /// curves and the table does not. The error lives at the *ends* — Schwarz P's first step runs
    /// from −3.000 to −1.621 — and, for Neovius, Schwarz P and Fischer-Koch S, in the second tenth
    /// as well; Neovius is more than 1% off across most of 0.11–0.37.
    ///
    /// So nothing quotes a solid at the number typed: [`Self::solid_density_built`] is what the
    /// property readout reads. Splitting the end steps would close the gap, and would change what
    /// every saved low-density part builds, which makes it a decision of its own rather than a fix.
    pub fn isovalue_for_density(self, density: f64) -> f64 {
        let table = self.density_isovalues();
        let d = density.clamp(0.0, 1.0);
        let x = d / Self::DENSITY_STEP;
        let i = (x.floor() as usize).min(table.len() - 2);
        let t = x - i as f64;
        table[i] * (1.0 - t) + table[i + 1] * t
    }

    /// The steepest the isovalue moves per unit of density, anywhere in the table.
    ///
    /// What a *driven* density multiplies the ramp's own rate by: the value is
    /// `(F - iso(density)) / bound`, so a density that changes through space moves it at
    /// `iso'` times the ramp's rate, and the divisor that keeps the pair honest is built from
    /// the worst of it. See `driven_lattice_divisor`.
    pub(crate) fn density_slope(self) -> f64 {
        self.density_isovalues()
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() / Self::DENSITY_STEP)
            .fold(0.0, f64::max)
    }

    /// The spacing of [`Self::density_isovalues`], as a fraction.
    pub(crate) const DENSITY_STEP: f64 = 0.1;

    /// Isovalue at 0%, 10%, ... 100% relative density.
    ///
    /// **Measured by `tpms_profile.rs`; re-run it and paste the row back after touching any
    /// `f`.** A stale row here returns a different amount of material than the number typed,
    /// and nothing about the result looks wrong — which is the failure D8 exists to name.
    pub fn density_isovalues(self) -> [f64; 11] {
        match self {
            TpmsKind::Gyroid => DENSITY_GYROID,
            TpmsKind::SchwarzP => DENSITY_SCHWARZ_P,
            TpmsKind::SchwarzD => DENSITY_SCHWARZ_D,
            TpmsKind::SchoenIwp => DENSITY_SCHOEN_IWP,
            TpmsKind::Neovius => DENSITY_NEOVIUS,
            TpmsKind::FischerKochS => DENSITY_FISCHER_KOCH_S,
        }
    }

    /// The typed density whose isovalue is `iso`: the exact inverse of
    /// [`Self::isovalue_for_density`], which is piecewise linear and strictly rising.
    ///
    /// What turns an isovalue the geometry cares about -- a saddle, say -- back into the number a
    /// person types. Clamped to the table's own ends, as the forward reading is.
    pub fn density_for_isovalue(self, iso: f64) -> f64 {
        let table = self.density_isovalues();
        if iso <= table[0] {
            return 0.0;
        }
        if iso >= table[table.len() - 1] {
            return 1.0;
        }
        let i = table.windows(2).position(|w| iso <= w[1]).unwrap_or(table.len() - 2);
        let t = (iso - table[i]) / (table[i + 1] - table[i]);
        (i as f64 + t) * Self::DENSITY_STEP
    }

    /// The fraction of a cell a solid actually fills at [`Self::isovalue_for_density`]`(typed)`.
    ///
    /// **Not the number typed, below a tenth.** The isovalue table is linear across tenths and
    /// the fill is not, so typed 0.05 builds 0.026 (Neovius) to 0.043 (gyroid), and a fit read at
    /// the typed number would overstate a part by the ratio to the power `n`. Where a solid holds
    /// together, that is up to about 2.7× at `n = 2`: a gyroid just above its floor, typed 0.028,
    /// builds 0.017. (A Neovius typed 0.05 would read 4.4× by Alabort's `ρ^2.28`, but it is
    /// separate pieces and refused before any fit is read.) This is what the property readout
    /// and its inverse read a solid at: the density the field has, not the one asked for.
    /// Closing the gap instead, by splitting the table's end steps, would change what every saved
    /// low-density part builds, which is its own decision.
    ///
    /// Measured: `SOLID_BUILT_*`, 101 rows at typed 0.00, 0.01, … 1.00, linear between them --
    /// within each tenth both the isovalue and the fill are smooth in the typed number, so
    /// hundredths interpolate near-exactly. `the_solid_density_table_is_the_fraction_the_field_fills`
    /// holds it to `Lattice::distance`.
    pub fn solid_density_built(self, typed: f64) -> f64 {
        if !typed.is_finite() {
            return f64::NAN;
        }
        let rows = self.solid_built_rows();
        let x = typed.clamp(0.0, 1.0) * 100.0;
        let i = (x.floor() as usize).min(rows.len() - 2);
        let t = x - i as f64;
        rows[i] * (1.0 - t) + rows[i + 1] * t
    }

    fn solid_built_rows(self) -> &'static [f64; 101] {
        match self {
            TpmsKind::Gyroid => &SOLID_BUILT_GYROID,
            TpmsKind::SchwarzP => &SOLID_BUILT_SCHWARZ_P,
            TpmsKind::SchwarzD => &SOLID_BUILT_SCHWARZ_D,
            TpmsKind::SchoenIwp => &SOLID_BUILT_SCHOEN_IWP,
            TpmsKind::Neovius => &SOLID_BUILT_NEOVIUS,
            TpmsKind::FischerKochS => &SOLID_BUILT_FISCHER_KOCH_S,
        }
    }

    /// The isovalue at or below which the solid `{F ≤ t}` is separate pieces, and the fill
    /// there.
    ///
    /// **Below it a solid is not a lattice.** `{F ≤ t}` gains a loop round the period only at
    /// an index-1 critical point of `F` -- a saddle -- so until `t` passes the right one the
    /// material is blobs, each round a minimum of `F`, touching nothing: no stiffness to quote
    /// and nothing to print as one part. A Neovius solid joins only above about a third full,
    /// so a typed 0.30 Neovius solid is separate pieces however lattice-like it looks drawn.
    ///
    /// The isovalue is the saddle's own value, exact where it has a closed form -- gyroid `−√2`,
    /// Schwarz P and D `−1`, Schoen IWP `−3` (on the line `(π, 0, z)` `F = −4 − cos 2z`, whose
    /// highest point is `−3`), Neovius `−3/4` (on `(π, y, y)`, `F = −3 + 6c − 4c²` at `c = 3/4`)
    /// -- and Fischer-Koch S's by Newton's method on `∇F = 0`. Which saddle is decided by a lifted
    /// flood fill. `tpms_profile.rs` measures all of it; see `SOLID_FLOOR` for its table, and
    /// `a_solid_below_its_saddle_is_separate_pieces` pins each one either side.
    pub fn solid_floor(self) -> (f64, f64) {
        let i = TpmsKind::ALL
            .iter()
            .position(|k| *k == self)
            .expect("every kind is in ALL");
        SOLID_FLOOR[i]
    }

    /// The fraction of a cubic cell a centred wall fills, against the wall over the cell.
    ///
    /// Exactly what `Lattice::distance` calls material for a sheet on a cubic rectangular map,
    /// `|F| ≤ π·g·(t/L)` with `g` the [`Self::surface_gradient`] the wall is converted through --
    /// so it is the fill of the wall as built, nominal thickness and all, not of an ideal wall.
    ///
    /// Measured: `SHEET_DENSITY_*` holds `ρ/(t/L)` at every 1% of wall to cell from 1% up to the
    /// [`Self::wall_limit`] -- item 87's rows, and its material-fraction column: one table of
    /// one field, not two. The coefficient is nearly flat (it tends to the surface's area per
    /// cell), so 1% rows interpolate near-exactly, and below the first row its coefficient is
    /// used. `None` past the wall limit, where the walls have met and this is not a sheet.
    pub fn sheet_density(self, wall_over_cell: f64) -> Option<f64> {
        let w = wall_over_cell;
        if !(w.is_finite() && w >= 0.0) || w > self.wall_limit() + 1e-9 {
            return None;
        }
        let rows = self.sheet_rows();
        // Row `i` is at `(i + 1)/100`.
        let x = w * 100.0 - 1.0;
        let coefficient = if x <= 0.0 {
            rows[0]
        } else {
            let i = (x.floor() as usize).min(rows.len() - 2);
            let t = x - i as f64;
            rows[i] * (1.0 - t) + rows[i + 1] * t
        };
        Some(coefficient * w)
    }

    fn sheet_rows(self) -> &'static [f64] {
        match self {
            TpmsKind::Gyroid => &SHEET_DENSITY_GYROID,
            TpmsKind::SchwarzP => &SHEET_DENSITY_SCHWARZ_P,
            TpmsKind::SchwarzD => &SHEET_DENSITY_SCHWARZ_D,
            TpmsKind::SchoenIwp => &SHEET_DENSITY_SCHOEN_IWP,
            TpmsKind::Neovius => &SHEET_DENSITY_NEOVIUS,
            TpmsKind::FischerKochS => &SHEET_DENSITY_FISCHER_KOCH_S,
        }
    }
}

/// Isovalue at 0%, 10%, ... 100% relative density for each cell, measured over one full period
/// by `fieldform-field/examples/tpms_profile.rs`.
///
/// Sampled on a 128³ grid over one period in each argument, which is why the figures carry three
/// decimals and not more.
///
/// The end entries are the extremes of `F` over a cell, and they are the check that the sampling
/// is dense enough to find them — every one came back at its analytic value: ±1.5 for the gyroid,
/// ±3 for Schwarz P, ±√2 for Schwarz D and for Fischer-Koch S, ±13 for Neovius
/// (`3·3 + 4` at the origin), and **−5 to +3** for Schoen IWP, which is the asymmetry itself
/// rather than a sampling artefact.
const DENSITY_GYROID: [f64; 11] = [
    -1.500, -1.197, -0.913, -0.615, -0.310, 0.000, 0.310, 0.615, 0.913, 1.197, 1.500,
];
const DENSITY_SCHWARZ_P: [f64; 11] = [
    -3.000, -1.621, -1.051, -0.700, -0.350, 0.000, 0.350, 0.700, 1.051, 1.621, 3.000,
];
const DENSITY_SCHOEN_IWP: [f64; 11] = [
    -5.000, -3.002, -2.164, -1.339, -0.538, 0.232, 0.954, 1.618, 2.205, 2.691, 3.000,
];
const DENSITY_NEOVIUS: [f64; 11] = [
    -13.000, -4.857, -2.301, -0.980, -0.456, 0.000, 0.456, 0.980, 2.301, 4.857, 13.000,
];
const DENSITY_FISCHER_KOCH_S: [f64; 11] = [
    -1.414, -0.784, -0.598, -0.404, -0.203, 0.000, 0.203, 0.404, 0.598, 0.784, 1.414,
];
const DENSITY_SCHWARZ_D: [f64; 11] = [
    -1.414, -0.956, -0.726, -0.485, -0.243, 0.000, 0.243, 0.485, 0.726, 0.956, 1.414,
];

/// What each typed solid density builds, at typed 0.00, 0.01, ... 1.00: the fill of
/// `F ≤ isovalue_for_density(typed)` over one period, read by [`TpmsKind::solid_density_built`].
///
/// **Measured by `tpms_profile.rs`** ("Solid density built"), from 2²⁴ points of [`r3_point`]
/// over one period. Its error bound, the 2²² prefix against the whole: at most 0.047% where the
/// fill is at least 0.02 (Fischer-Koch S; gyroid 0.034%, Schwarz P 0.030%, Schwarz D 0.035%,
/// Schoen IWP 0.029%, Neovius 0.012%), and 2.7e-5 absolute below, where a relative figure says
/// nothing about a shape. The same run counts `Lattice::distance`'s own sign on 2²⁰ points of a
/// 10 mm lattice at typed 0.05, 0.50 and 0.95 and finds it identical, point for point, to the
/// formula's -- the conversion from a typed number to a field is the one this table assumes.
///
/// **Re-run it and paste the six arrays back after touching any `f` or `density_isovalues`.**
/// `the_solid_density_table_is_the_fraction_the_field_fills` is what fails first if they drift.
const SOLID_BUILT_GYROID: [f64; 101] = [
    0.000000, 0.002999, 0.009052, 0.018750, 0.030950, 0.042900, 0.054643, 0.066211,
    0.077631, 0.088905, 0.100049, 0.110412, 0.120661, 0.130864, 0.140969, 0.151000,
    0.160994, 0.170896, 0.180779, 0.190574, 0.200329, 0.210522, 0.220673, 0.230762,
    0.240829, 0.250839, 0.260818, 0.270766, 0.280686, 0.290555, 0.300417, 0.310493,
    0.320517, 0.330517, 0.340512, 0.350470, 0.360420, 0.370341, 0.380258, 0.390158,
    0.400041, 0.410065, 0.420092, 0.430100, 0.440100, 0.450093, 0.460086, 0.470078,
    0.480043, 0.490019, 0.499998, 0.509964, 0.519948, 0.529939, 0.539905, 0.549910,
    0.559893, 0.569911, 0.579904, 0.589928, 0.599955, 0.609846, 0.619729, 0.629647,
    0.639577, 0.649525, 0.659492, 0.669468, 0.679482, 0.689517, 0.699576, 0.709438,
    0.719322, 0.729243, 0.739192, 0.749154, 0.759186, 0.769236, 0.779343, 0.789483,
    0.799673, 0.809428, 0.819230, 0.829093, 0.839012, 0.848984, 0.859034, 0.869152,
    0.879324, 0.889585, 0.899952, 0.911099, 0.922354, 0.933782, 0.945351, 0.957095,
    0.969050, 0.981253, 0.990950, 0.997007, 1.000000,
];
const SOLID_BUILT_SCHWARZ_P: [f64; 101] = [
    0.000000, 0.002496, 0.007219, 0.013570, 0.021388, 0.030637, 0.041319, 0.053494,
    0.067241, 0.082691, 0.100038, 0.107814, 0.115966, 0.124543, 0.133560, 0.143053,
    0.153079, 0.163697, 0.174986, 0.187088, 0.200204, 0.209017, 0.218883, 0.229028,
    0.239164, 0.249296, 0.259416, 0.269518, 0.279623, 0.289716, 0.299811, 0.309868,
    0.319911, 0.329949, 0.339976, 0.350005, 0.360025, 0.370045, 0.380063, 0.390076,
    0.400075, 0.410087, 0.420067, 0.430074, 0.440062, 0.450062, 0.460058, 0.470021,
    0.480025, 0.490015, 0.500000, 0.509990, 0.519961, 0.529966, 0.539947, 0.549942,
    0.559935, 0.569938, 0.579924, 0.589922, 0.599925, 0.609936, 0.619937, 0.629951,
    0.639976, 0.649989, 0.660029, 0.670051, 0.680092, 0.690143, 0.700198, 0.710275,
    0.720377, 0.730475, 0.740578, 0.750699, 0.760827, 0.770972, 0.781117, 0.790982,
    0.799789, 0.812917, 0.825012, 0.836309, 0.846931, 0.856947, 0.866442, 0.875459,
    0.884024, 0.892182, 0.899961, 0.917302, 0.932753, 0.946506, 0.958684, 0.969367,
    0.978611, 0.986428, 0.992777, 0.997502, 1.000000,
];
const SOLID_BUILT_SCHWARZ_D: [f64; 101] = [
    0.000001, 0.002291, 0.006609, 0.012421, 0.019614, 0.028182, 0.038198, 0.049813,
    0.063369, 0.079821, 0.100007, 0.110117, 0.120153, 0.130172, 0.140129, 0.150076,
    0.159968, 0.169852, 0.179697, 0.189496, 0.199283, 0.209524, 0.219704, 0.229879,
    0.240028, 0.250170, 0.260267, 0.270342, 0.280405, 0.290445, 0.300472, 0.310540,
    0.320559, 0.330585, 0.340604, 0.350595, 0.360568, 0.370552, 0.380510, 0.390473,
    0.400409, 0.410414, 0.420380, 0.430350, 0.440314, 0.450258, 0.460205, 0.470171,
    0.480105, 0.490064, 0.499998, 0.509935, 0.519899, 0.529835, 0.539793, 0.549749,
    0.559686, 0.569664, 0.579631, 0.589609, 0.599580, 0.609510, 0.619480, 0.629439,
    0.639404, 0.649396, 0.659398, 0.669398, 0.679420, 0.689467, 0.699502, 0.709551,
    0.719591, 0.729664, 0.739732, 0.749853, 0.759962, 0.770100, 0.780286, 0.790479,
    0.800721, 0.810493, 0.820312, 0.830150, 0.840026, 0.849923, 0.859878, 0.869819,
    0.879847, 0.889883, 0.899999, 0.920180, 0.936625, 0.950177, 0.961809, 0.971811,
    0.980379, 0.987569, 0.993393, 0.997708, 0.999999,
];
const SOLID_BUILT_SCHOEN_IWP: [f64; 101] = [
    0.000000, 0.002318, 0.006719, 0.012666, 0.020037, 0.028842, 0.039128, 0.051031,
    0.064778, 0.080787, 0.100273, 0.110263, 0.120233, 0.130211, 0.140206, 0.150190,
    0.160189, 0.170190, 0.180198, 0.190216, 0.200256, 0.210150, 0.220061, 0.229995,
    0.239934, 0.249895, 0.259897, 0.269893, 0.279931, 0.289992, 0.300083, 0.309901,
    0.319757, 0.329622, 0.339561, 0.349511, 0.359495, 0.369527, 0.379574, 0.389684,
    0.399837, 0.409634, 0.419495, 0.429356, 0.439297, 0.449290, 0.459315, 0.469401,
    0.479575, 0.489763, 0.500052, 0.509753, 0.519484, 0.529311, 0.539197, 0.549139,
    0.559178, 0.569252, 0.579430, 0.589693, 0.600026, 0.609613, 0.619292, 0.629050,
    0.638884, 0.648853, 0.658882, 0.669036, 0.679293, 0.689667, 0.700159, 0.709553,
    0.719063, 0.728684, 0.738425, 0.748311, 0.758324, 0.768500, 0.778847, 0.789376,
    0.800098, 0.809139, 0.818335, 0.827700, 0.837272, 0.847054, 0.857060, 0.867341,
    0.877904, 0.888846, 0.900178, 0.907637, 0.915319, 0.923262, 0.931516, 0.940159,
    0.949259, 0.959000, 0.969649, 0.981883, 1.000000,
];
const SOLID_BUILT_NEOVIUS: [f64; 101] = [
    0.000000, 0.001970, 0.005796, 0.011112, 0.017893, 0.026232, 0.036283, 0.048287,
    0.062560, 0.079594, 0.099988, 0.107239, 0.114933, 0.123126, 0.131879, 0.141222,
    0.151253, 0.162036, 0.173663, 0.186248, 0.199927, 0.207492, 0.215427, 0.223787,
    0.232597, 0.241909, 0.251796, 0.262331, 0.273630, 0.285836, 0.299341, 0.305969,
    0.313460, 0.321794, 0.331258, 0.342638, 0.354356, 0.366016, 0.377634, 0.389178,
    0.400693, 0.410690, 0.420662, 0.430623, 0.440550, 0.450494, 0.460398, 0.470310,
    0.480212, 0.490106, 0.500001, 0.509880, 0.519801, 0.529696, 0.539588, 0.549510,
    0.559430, 0.569383, 0.579338, 0.589316, 0.599307, 0.610811, 0.622376, 0.633981,
    0.645642, 0.657355, 0.668746, 0.678199, 0.686554, 0.694024, 0.700663, 0.714166,
    0.726373, 0.737675, 0.748201, 0.758090, 0.767405, 0.776212, 0.784561, 0.792504,
    0.800071, 0.813755, 0.826345, 0.837965, 0.848745, 0.858765, 0.868128, 0.876861,
    0.885064, 0.892775, 0.900004, 0.920414, 0.937435, 0.951719, 0.963714, 0.973767,
    0.982103, 0.988885, 0.994199, 0.998031, 1.000000,
];
const SOLID_BUILT_FISCHER_KOCH_S: [f64; 101] = [
    0.000000, 0.002226, 0.006461, 0.012217, 0.019391, 0.028024, 0.038204, 0.050085,
    0.063979, 0.080390, 0.100450, 0.107528, 0.116315, 0.127585, 0.138433, 0.148977,
    0.159387, 0.169615, 0.179727, 0.189764, 0.199701, 0.209989, 0.220192, 0.230334,
    0.240425, 0.250427, 0.260418, 0.270375, 0.280257, 0.290127, 0.299932, 0.310103,
    0.320198, 0.330280, 0.340343, 0.350369, 0.360374, 0.370348, 0.380320, 0.390288,
    0.400207, 0.410215, 0.420227, 0.430216, 0.440214, 0.450189, 0.460140, 0.470117,
    0.480087, 0.490058, 0.499988, 0.509955, 0.519917, 0.529880, 0.539833, 0.549816,
    0.559783, 0.569783, 0.579771, 0.589760, 0.599796, 0.609738, 0.619682, 0.629649,
    0.639636, 0.649655, 0.659666, 0.669709, 0.679807, 0.689913, 0.700049, 0.709880,
    0.719741, 0.729622, 0.739555, 0.749535, 0.759592, 0.769671, 0.779801, 0.790033,
    0.800297, 0.810244, 0.820267, 0.830380, 0.840645, 0.851010, 0.861588, 0.872390,
    0.883705, 0.892461, 0.899543, 0.919619, 0.936016, 0.949906, 0.961805, 0.971977,
    0.980607, 0.987792, 0.993543, 0.997774, 0.999999,
];

/// Where each TPMS solid stops being separate pieces, in [`TpmsKind::ALL`] order: the saddle's
/// isovalue, and the fill there. Read by [`TpmsKind::solid_floor`].
///
/// **Measured by `tpms_profile.rs`** ("Solid floor"). Newton's method on `∇F = 0` from 14³
/// starts finds the index-1 critical points; a lifted flood fill of `{F ≤ t}` on a 64³ grid,
/// bisected forty times, finds where a component first wraps the period -- a little above the
/// saddle, since a neck must be a voxel wide -- and the floor is the highest index-1 value at or
/// below that. Newton landed on every closed form to 1e-12, and the 40³ flood that
/// `a_solid_below_its_saddle_is_separate_pieces` runs reads apart at 0.05 below and joined at
/// 0.05 above, for all six. The fill is from 2²⁴ points, within the solid table's bound.
///
/// | kind | saddle | the 64³ grid spans above | fill there | typed density at the saddle |
/// |---|---|---|---|---|
/// | Gyroid | −√2 | −1.4084 | 0.0167 | 0.028 |
/// | Schwarz P | −1 | −0.9988 | 0.2133 | 0.215 |
/// | Schwarz D | −1 | −0.9940 | 0.0806 | 0.090 |
/// | Schoen IWP | −3 | −2.9904 | 0.1005 | 0.100 |
/// | Neovius | −3/4 | −0.7416 | 0.3354 | 0.344 |
/// | Fischer-Koch S | −0.749168 (Newton) | −0.7471 | 0.1148 | 0.119 |
///
/// A Neovius solid typed below about 0.34 is therefore not one lattice, and a Schwarz P solid
/// below about 0.21; a gyroid holds together down to 2.8% typed, 1.7% built.
const SOLID_FLOOR: [(f64, f64); 6] = [
    (-std::f64::consts::SQRT_2, 0.01666),
    (-1.0, 0.21334),
    (-1.0, 0.08059),
    (-3.0, 0.10051),
    (-0.75, 0.33544),
    (-0.749_167_700_349, 0.11480),
];

/// A centred sheet's fill over the wall-to-cell ratio `w`, as `ρ/w`, at `w` = 0.01, 0.02, ...
/// up to the kind's [`TpmsKind::wall_limit`]: item 87's rows, and its material-fraction column.
/// Read by [`TpmsKind::sheet_density`].
///
/// **Measured by `tpms_profile.rs`** ("Sheet density"): the fraction of 2²⁴ points of
/// [`r3_point`] with `|F| ≤ π·surface_gradient·w`, which is what `Lattice::distance` calls
/// material for a centred sheet on a cubic rectangular map. Its error bound, the 2²² prefix
/// against the whole, row by row: at most 0.018% gyroid, 0.032% Schwarz P, 0.054% Schwarz D,
/// 0.039% Schoen IWP, 0.096% Neovius, 0.091% Fischer-Koch S. The field's own sign on 2²⁰ points
/// of a 10 mm lattice matches the formula's point for point at the first, middle and last rows.
///
/// **Re-run and paste back after touching any `f` or `surface_gradient`, and when item 86's exact
/// wall replaces the nominal one** -- the sheet field changes then and
/// `the_sheet_density_table_is_the_fraction_the_field_fills` fails until this is re-measured.
const SHEET_DENSITY_GYROID: [f64; 25] = [
    3.09232, 3.09432, 3.09488, 3.09618, 3.09701, 3.09915, 3.10124, 3.10329,
    3.10604, 3.10924, 3.11249, 3.11648, 3.12062, 3.12518, 3.13032, 3.13575,
    3.14178, 3.14835, 3.15536, 3.16314, 3.17142, 3.18061, 3.19043, 3.20131,
    3.21317,
];
const SHEET_DENSITY_SCHWARZ_P: [f64; 22] = [
    2.36847, 2.36795, 2.36851, 2.36896, 2.36913, 2.36897, 2.36933, 2.36969,
    2.36987, 2.37025, 2.37054, 2.37101, 2.37144, 2.37199, 2.37255, 2.37315,
    2.37374, 2.37453, 2.37515, 2.37583, 2.37666, 2.37755,
];
const SHEET_DENSITY_SCHWARZ_D: [f64; 20] = [
    3.82214, 3.82186, 3.82277, 3.82419, 3.82608, 3.82778, 3.82994, 3.83293,
    3.83605, 3.83949, 3.84347, 3.84777, 3.85277, 3.85795, 3.86384, 3.87018,
    3.87722, 3.88473, 3.89287, 3.90181,
];
const SHEET_DENSITY_SCHOEN_IWP: [f64; 18] = [
    3.57518, 3.57592, 3.57877, 3.58182, 3.58634, 3.59185, 3.59793, 3.60562,
    3.61430, 3.62446, 3.63602, 3.64938, 3.66433, 3.68142, 3.70101, 3.72376,
    3.74997, 3.78092,
];
const SHEET_DENSITY_NEOVIUS: [f64; 6] = [
    3.27436, 3.27429, 3.27592, 3.27822, 3.28097, 3.28512,
];
const SHEET_DENSITY_FISCHER_KOCH_S: [f64; 10] = [
    5.42147, 5.42479, 5.42996, 5.43542, 5.44427, 5.45515, 5.46851, 5.48442,
    5.50374, 5.52688,
];
