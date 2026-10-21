// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The two constants each TPMS kind needs, and what the second one costs.
//!
//! `lattice/tpms.rs` carries a bound on `|∇F|` and a typical `|∇F|` at the surface for each
//! kind. They do different jobs and are established differently, which is the point of
//! measuring both here rather than either one alone.
//!
//! - **The bound** is a safety property. The field's value is divided by it, which is what
//!   makes the value never overstate the distance to a wall — so a sphere tracer cannot step
//!   through one and interval pruning stays correct. It is *proved* in `lattice/tpms.rs`; this
//!   example checks no sample exceeds it, because a bound that is wrong makes every traced
//!   image and every pruned region built on it untrustworthy.
//!
//! - **The typical gradient** turns a wall thickness in millimetres into a threshold on `|F|`.
//!   It is a mean over a surface whose gradient varies, so there is nothing to prove and the
//!   thickness it produces is *nominal*. What this example adds is the size of that error,
//!   which was previously asserted to exist and never quantified.
//!
//! Extends the methodology `gyroid_spike.rs` used for one kind to all three, which is what
//! `tpms-infill-spec.md` §5.2 asks for.
//!
//! **Three fill tables follow, for item 94's property readout** (and, for the sheet, as item 87's
//! material-fraction column), each pasted back into `lattice/tpms.rs` with the bound printed here:
//!
//! - **Solid density built**: what each typed density, 0.00 to 1.00 by hundredths, actually
//!   fills -- not the number typed, below a tenth -- with the relative-error table that
//!   `isovalue_for_density`'s doc comment carries;
//! - **Solid floor**: where each solid stops being separate pieces, by Newton's method on the
//!   saddles and a lifted flood fill;
//! - **Sheet density**: a centred wall's fill at every 1% of wall to cell up to the wall limit.
//!
//! All three sample the kernel's own `r3_point` sequence -- 2²⁴ points, checked against the first
//! 2²² -- so the kernel's tests can sample the same points and be held to what this run printed.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example tpms_profile`

use fieldform_field::{r3_point, CellKind, Lattice, LatticeMap, Phase, ScalarField, TpmsKind};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::f64::consts::{FRAC_1_SQRT_2, TAU};

fn main() {
    println!("Gradient bound — must hold everywhere, or the field is unsafe to trace\n");
    println!(
        "{:>12}  {:>10}  {:>10}  {:>8}",
        "kind", "bound", "worst seen", "headroom"
    );
    for kind in TpmsKind::ALL {
        let worst = everywhere(kind).into_iter().fold(0.0f64, f64::max);
        println!(
            "{:>12}  {:>10.4}  {:>10.4}  {:>7.1}%",
            kind.label(),
            kind.gradient_bound(),
            worst,
            100.0 * (kind.gradient_bound() - worst) / kind.gradient_bound(),
        );
    }

    println!("\nGradient at the surface — the constant a wall thickness is converted through\n");
    println!(
        "{:>12}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
        "kind", "in use", "measured", "10th pct", "90th pct", "samples"
    );
    for kind in TpmsKind::ALL {
        let mut at = at_surface(kind);
        at.sort_by(f64::total_cmp);
        let mean = at.iter().sum::<f64>() / at.len() as f64;
        println!(
            "{:>12}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>10}",
            kind.label(),
            kind.surface_gradient(),
            mean,
            at[at.len() / 10],
            at[at.len() * 9 / 10],
            at.len(),
        );
    }

    println!(
        "\nWhat that costs — the wall is thinner where the gradient is steeper\n\
         (a tangent estimate, 2·iso/|grad F| per surface point; the chord measured by\n\
         `a_wall_comes_out_within_the_band_its_doc_comment_promises` is the authority, and\n\
         runs fatter wherever F is flat — up to 3x on Neovius)\n"
    );
    println!(
        "{:>12}  {:>10}  {:>12}  {:>12}",
        "kind", "asked (mm)", "thinnest", "thickest"
    );
    for kind in TpmsKind::ALL {
        let mut at = at_surface(kind);
        at.sort_by(f64::total_cmp);
        // Thickness runs inversely with the gradient: the isovalue is fixed, so where |grad F|
        // is steeper the band in |F| spans less distance. Reported against the constant
        // actually in use rather than against the measured mean, since that is what a typed
        // number is converted through.
        let asked = 2.0;
        let g = kind.surface_gradient();
        println!(
            "{:>12}  {:>10.2}  {:>12.3}  {:>12.3}",
            kind.label(),
            asked,
            asked * g / at[at.len() * 9 / 10],
            asked * g / at[at.len() / 10],
        );
    }

    println!(
        "\nAnd the thing the whole field is for: how much room it claims against how\n\
              much there is\n"
    );
    println!(
        "{:>12}  {:>10}  {:>10}  {:>10}  {:>16}",
        "kind", "worst", "mean", "best", "verdict"
    );
    for kind in TpmsKind::ALL {
        let mut ratios = claimed_against_actual(kind);
        ratios.sort_by(f64::total_cmp);
        let worst = *ratios.last().expect("some ray found a wall");
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        println!(
            "{:>12}  {:>10.3}  {:>10.3}  {:>10.3}  {:>16}",
            kind.label(),
            worst,
            mean,
            ratios[0],
            if worst <= 1.0 {
                "never overstates"
            } else {
                "OVERSTATES"
            },
        );
    }
    println!(
        "\nRatio is the value the field reported divided by the distance actually there. At or\n\
         below 1 is the contract, and the one thing that must hold. How far below is what a\n\
         *global* bound gives away: every step a tracer takes is that fraction of the step it\n\
         could have taken, so the mean is roughly the factor by which tracing a lattice is\n\
         slower than it needs to be. That is what local-gradient normalization would recover,\n\
         and the reason D8 names it as future work rather than a nicety."
    );

    println!(
        "
Per-argument gradient at the surface — whether the /sqrt(3) shortcut holds for this cell
"
    );
    println!(
        "{:>15}  {:>9}  {:>9}  {:>9}  {:>9}",
        "kind", "RMS x", "RMS y", "RMS z", "spread"
    );
    for kind in TpmsKind::ALL {
        let [rx, ry, rz] = per_argument_rms(kind);
        let hi = rx.max(ry).max(rz);
        let lo = rx.min(ry).min(rz);
        println!(
            "{:>15}  {:>9.4}  {:>9.4}  {:>9.4}  {:>8.2}%",
            kind.label(),
            rx,
            ry,
            rz,
            100.0 * (hi - lo) / hi,
        );
    }
    println!(
        "
`argument_gradient` splits the whole gradient evenly across the three arguments, which is
         only sound for a cell whose partials have the same distribution. Spread is how far
         apart the three measured values actually are, and
         `every_kinds_arguments_are_equally_steep` asserts it stays small. It is the assertion
         that will fail first when a beam cell arrives — struts differ per axis — which is
         exactly when it should, because the shortcut mis-sizes such a cell under a curved map
         and nothing else would report it."
    );

    println!(
        "
Tracing cost — what a bound looser than the field costs every ray
"
    );
    println!("{:>15}  {:>10}  {:>10}  {:>12}", "kind", "bound", "at surface", "cost ratio");
    for kind in TpmsKind::ALL {
        println!(
            "{:>15}  {:>10.4}  {:>10.3}  {:>12.2}",
            kind.label(),
            kind.gradient_bound(),
            kind.surface_gradient(),
            kind.gradient_bound() / kind.surface_gradient(),
        );
    }
    println!(
        "
The ratio is what a ray pays for the bound being looser than the field really is: every step
         is divided by the bound, so a cell whose bound sits well above its typical gradient
         traces proportionally slower. The first three are proved and attained, so theirs is as
         low as it goes. The last three are bounded *termwise* — sound, and loose — so theirs
         starts high by construction. That is a known cause with a known fix (a tighter argument,
         not a grid), which is the point of printing it rather than leaving \"this one is slow\"
         as a complaint."
    );

    println!(
        "
The solid phase: which isovalue gives which relative density
"
    );
    print!("{:>12}", "kind");
    for d in DENSITIES {
        print!("{:>8.0}%", d * 100.0);
    }
    println!();
    for kind in TpmsKind::ALL {
        // Sorted once per kind rather than once per column: the sort is the expensive half and
        // every column is a different index into the same array.
        let mut values = over_one_cell(kind);
        values.sort_by(f64::total_cmp);
        print!("{:>12}", kind.label());
        for d in DENSITIES {
            print!("{:>9.3}", quantile(&values, d));
        }
        println!();
    }
    println!(
        "
The quantile of F over one cell: the fraction of a cell's volume where F is at or below
         that value, which is exactly what `F <= iso` selects. Read the 50% column: five of the
         six are zero, because those five sums negate under a half-period shift — and **Schoen
         IWP is 0.232**, because it has no such shift and its value runs from -5 to +3 rather
         than symmetrically. That one cell is the whole reason this is a per-cell table and not
         a constant, and it is not the cell the spec expected: it named Neovius, which measured
         antisymmetric at +/-13.

         These are the numbers pasted into `TpmsKind::density_isovalues`. Re-run this after
         touching any `f`, and paste the row back — the table is what a typed density is read
         through, and a stale one silently returns a different amount of material than asked."
    );

    println!(
        "
What the table costs — asked against achieved, reading through the stored tenths, in
absolute percentage points (achieved minus asked), not per cent of the number asked
"
    );
    println!(
        "{:>12}  {:>12}  {:>12}  {:>24}",
        "kind", "worst (pts)", "at density", "worst in 10-90% (pts)"
    );
    for kind in TpmsKind::ALL {
        let mut values = over_one_cell(kind);
        values.sort_by(f64::total_cmp);
        let mut worst = (0.0f64, 0.0f64);
        let mut worst_mid = 0.0f64;
        // Between the stored entries, where interpolation is doing the work. The stored
        // densities themselves are exact by construction and would only flatter the result.
        for step in 1..80 {
            let asked = step as f64 / 80.0;
            let achieved = fraction_below(&values, kind.isovalue_for_density(asked));
            let error = (achieved - asked).abs();
            if error > worst.0 {
                worst = (error, asked);
            }
            if (0.1..=0.9).contains(&asked) && error > worst_mid {
                worst_mid = error;
            }
        }
        println!(
            "{:>12}  {:>12.2}  {:>11.0}%  {:>24.2}",
            kind.label(),
            worst.0 * 100.0,
            worst.1 * 100.0,
            worst_mid * 100.0,
        );
    }
    println!(
        "
This is the number the panel's `approximate` mark is for: type a density and this is how
         far the result can sit from it. The error is worst at the *ends* of the range, not in
         the middle, and that is where the steep first and last steps of the table are —
         Schwarz P's runs from -3.000 to -1.621, a curve far too steep for a straight line
         across a tenth. It is symmetric, so the same error sits a few per cent above empty as
         a few per cent below full; which end the table above names is down to how ties on a
         discrete grid fall and means nothing.

         **Points, not per cent of what was typed** (corrected by item 94). Under one point
         between 10% and 90% reads as \"within about one per cent\" and is not: relative to a
         small number typed, the first tenth is off by up to two thirds (a typed 0.05 builds
         0.026 to 0.043), and Neovius, Schwarz P and Fischer-Koch S are 3-6% off in the second
         tenth as well. \"Built against typed, relative\" below is the table to quote, and
         `TpmsKind::solid_density_built` is what the property readout reads. Splitting the two
         end steps would close the gap and change what every saved low-density part builds,
         which is backlog item 99, not a quiet fix."
    );

    // Item 94: what a lattice's density *is*, as Lattice::distance builds it. Three tables, each
    // pasted back into lattice/tpms.rs and each held to the field by a test in lattice/tests.rs
    // that samples sign(Lattice::distance) on the first 2^18 points of this same sequence.
    solid_density_built();
    solid_floor();
    sheet_density();
}

/// Points of one sample: every fill table below is read off 2^24 of them, and checked against
/// the first 2^22 of the same sequence.
const FULL: u64 = 1 << 24;
const PART: u64 = 1 << 22;
/// What the kernel's tests sample: the first 2^18. Printed so their tolerances can be read
/// against what that many points actually miss by.
const TEST: u64 = 1 << 18;
/// The spot check through the field itself.
const SPOT: u64 = 1 << 20;

/// `value` at the first `n` points of the kernel's own sequence over one period, sorted.
fn sorted_over_period(n: u64, value: impl Fn(Point3<f64>) -> f64 + Sync) -> Vec<f64> {
    let mut out: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| value(r3_point(i)))
        .collect();
    out.par_sort_unstable_by(f64::total_cmp);
    out
}

/// The fraction of a sample a lattice's field calls material, through `Lattice::distance`
/// itself: the spot column, which proves the conversion from number to field is the one the
/// table assumes.
fn field_fill(lattice: &Lattice, period: Vector3<f64>, n: u64) -> f64 {
    let inside: u64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let u = r3_point(i);
            let p = Point3::new(u.x * period.x, u.y * period.y, u.z * period.z);
            u64::from(lattice.distance(p) <= 0.0)
        })
        .sum();
    inside as f64 / n as f64
}

fn print_rust_array(name: &str, rows: &[f64], decimals: usize) {
    println!("const {name}: [f64; {}] = [", rows.len());
    for chunk in rows.chunks(8) {
        let line: Vec<String> = chunk.iter().map(|v| format!("{v:.decimals$}")).collect();
        println!("    {},", line.join(", "));
    }
    println!("];");
}

fn const_name(kind: TpmsKind) -> &'static str {
    match kind {
        TpmsKind::Gyroid => "GYROID",
        TpmsKind::SchwarzP => "SCHWARZ_P",
        TpmsKind::SchwarzD => "SCHWARZ_D",
        TpmsKind::SchoenIwp => "SCHOEN_IWP",
        TpmsKind::Neovius => "NEOVIUS",
        TpmsKind::FischerKochS => "FISCHER_KOCH_S",
    }
}

fn cube(cell: f64) -> Vector3<f64> {
    Vector3::new(cell, cell, cell)
}

/// **What each typed density builds.** `isovalue_for_density` is linear across tenths, so at the
/// tenths the typed number and the fill agree, and between the first two they do not: below 0.1
/// a solid is up to two thirds lighter than typed. This table is what the readout reads a solid
/// at, so a fit is read at the density the geometry has.
fn solid_density_built() {
    println!(
        "
Solid density built -- the fill at isovalue_for_density(typed), typed 0.00, 0.01, ... 1.00
({FULL} R3 points of F over one period; SOLID_BUILT_<KIND> in lattice/tpms.rs)
"
    );
    let mut relative = Vec::new();
    for kind in TpmsKind::ALL {
        let full = sorted_over_period(FULL, |u| kind.f(Point3::from(u.coords * TAU)));
        let part = sorted_over_period(PART, |u| kind.f(Point3::from(u.coords * TAU)));
        let test = sorted_over_period(TEST, |u| kind.f(Point3::from(u.coords * TAU)));
        let spot = sorted_over_period(SPOT, |u| kind.f(Point3::from(u.coords * TAU)));
        let typed = |k: usize| k as f64 / 100.0;
        let rows: Vec<f64> = (0..=100)
            .map(|k| fraction_below(&full, kind.isovalue_for_density(typed(k))))
            .collect();
        // The error bound: relative where the fill is at least 0.02, absolute below, because a
        // relative error on a fill of 0.001 says nothing about the shape.
        let (mut rel, mut abs) = (0.0f64, 0.0f64);
        for (k, row) in rows.iter().enumerate() {
            let other = fraction_below(&part, kind.isovalue_for_density(typed(k)));
            if *row >= 0.02 {
                rel = rel.max((other - row).abs() / row);
            } else {
                abs = abs.max((other - row).abs());
            }
        }
        print_rust_array(&format!("SOLID_BUILT_{}", const_name(kind)), &rows, 6);
        println!(
            "// {}: |2^22 - 2^24| at most {:.3}% where the fill is >= 0.02, {:.1e} absolute below",
            kind.label(),
            100.0 * rel,
            abs
        );
        // Every row as pasted, against what the first 2^20 points read: what the kernel's test
        // that holds every row to the formula can be held to.
        let (mut rel20, mut abs20) = (0.0f64, 0.0f64);
        for (k, row) in rows.iter().enumerate() {
            let pasted = (row * 1e6).round() / 1e6;
            let read = fraction_below(&spot, kind.isovalue_for_density(typed(k)));
            if pasted >= 0.02 {
                rel20 = rel20.max((read - pasted).abs() / pasted);
            } else {
                abs20 = abs20.max((read - pasted).abs());
            }
        }
        println!(
            "//   every row as pasted against the first 2^20 points: at most {:.3}% where the fill is >= 0.02, {:.1e} absolute below",
            100.0 * rel20,
            abs20
        );
        // What the tests see: the first 2^18 points, at the three rows they check. And the spot
        // column: the field's own sign against the formula's on the same 2^20 points, which
        // should differ by nothing but a point or two on the boundary.
        for k in [5usize, 50, 95] {
            let at_test = fraction_below(&test, kind.isovalue_for_density(typed(k)));
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Rectangular { cell: cube(10.0) },
                phase: Phase::Solid { density: typed(k) },
            };
            let through_field = field_fill(&lattice, cube(10.0), SPOT);
            let formula = fraction_below(&spot, kind.isovalue_for_density(typed(k)));
            println!(
                "//   typed {:.2}: table {:.5}; the first 2^18 read {:+.3}% off it, the first 2^20 {:+.3}%; field and formula on the 2^20 {} points apart",
                typed(k),
                rows[k],
                100.0 * (at_test - rows[k]) / rows[k],
                100.0 * (through_field - rows[k]) / rows[k],
                ((through_field - formula) * SPOT as f64).round(),
            );
        }
        relative.push((kind, rows));
    }

    println!(
        "
Built against typed, relative: the doc table of isovalue_for_density
"
    );
    let columns = [0.03, 0.05, 0.08, 0.10, 0.15, 0.50, 0.90, 0.95, 0.97];
    print!("| kind |");
    for c in columns {
        print!(" {c:.2} |");
    }
    println!(" worst in 0.10-0.90 |");
    print!("|---|");
    for _ in columns {
        print!("---|");
    }
    println!("---|");
    for (kind, rows) in &relative {
        print!("| {} |", kind.label());
        for c in columns {
            let k = (c * 100.0_f64).round() as usize;
            print!(" {:.3} ({:+.0}%) |", rows[k], 100.0 * (rows[k] - c) / c);
        }
        let worst = (10..=90)
            .map(|k| (rows[k] - k as f64 / 100.0).abs() / (k as f64 / 100.0))
            .fold(0.0f64, f64::max);
        println!(" {:.2}% |", 100.0 * worst);
    }
}

/// The critical points of `F` over one period, by Newton's method on `∇F = 0` from a grid of
/// starts, with each one's value and Morse index (how many directions it falls in).
fn critical_points(kind: TpmsKind) -> Vec<(f64, usize, Point3<f64>)> {
    let n = 14;
    let mut found: Vec<(f64, usize, Point3<f64>)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // Off the symmetric grid by an irrational nudge, so no start sits exactly on a
                // mirror plane where the Hessian may be singular.
                let mut p = Point3::new(
                    TAU * (i as f64 + 0.31) / n as f64,
                    TAU * (j as f64 + 0.57) / n as f64,
                    TAU * (k as f64 + 0.73) / n as f64,
                );
                let mut converged = false;
                for _ in 0..80 {
                    let g = kind.gradient(p);
                    if g.norm() < 1e-13 {
                        converged = true;
                        break;
                    }
                    let h = hessian(kind, p);
                    let Some(step) = h.lu().solve(&(-g)) else { break };
                    // Damped, so a start between two basins does not fling itself across the
                    // period.
                    let step = if step.norm() > 0.5 { step * (0.5 / step.norm()) } else { step };
                    p += step;
                }
                if !converged {
                    continue;
                }
                let h = hessian(kind, p);
                let sym = (h + h.transpose()) * 0.5;
                let eigen = nalgebra::SymmetricEigen::new(sym);
                if eigen.eigenvalues.iter().any(|l| l.abs() < 1e-6) {
                    // Degenerate: not a Morse point, and none of the six has one at a value
                    // that matters here.
                    continue;
                }
                let index = eigen.eigenvalues.iter().filter(|l| **l < 0.0).count();
                let value = kind.f(p);
                let wrap = |t: f64| t.rem_euclid(TAU);
                let p = Point3::new(wrap(p.x), wrap(p.y), wrap(p.z));
                if !found
                    .iter()
                    .any(|(v, ix, _)| (v - value).abs() < 1e-9 && *ix == index)
                {
                    found.push((value, index, p));
                }
            }
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    found
}

fn hessian(kind: TpmsKind, p: Point3<f64>) -> nalgebra::Matrix3<f64> {
    let h = 1e-6;
    let mut m = nalgebra::Matrix3::zeros();
    for c in 0..3 {
        let mut e = Vector3::zeros();
        e[c] = h;
        let column = (kind.gradient(p + e) - kind.gradient(p - e)) / (2.0 * h);
        m.set_column(c, &column);
    }
    m
}

/// Whether some component of `{F <= t}` wraps the period, on an `n`-cubed grid of one period.
///
/// **Lifted labels**, as item 87's window search specifies: each voxel reached carries the
/// whole-period offset it was reached at, and a component wraps when it reaches one voxel at two
/// different offsets. A plain periodic flood cannot tell wrapping from touching the boundary.
fn spans(values: &[f64], n: usize, t: f64) -> bool {
    let index = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
    let mut offset: Vec<Option<[i32; 3]>> = vec![None; n * n * n];
    let mut queue = std::collections::VecDeque::new();
    for start in 0..n * n * n {
        if values[start] > t || offset[start].is_some() {
            continue;
        }
        offset[start] = Some([0, 0, 0]);
        queue.push_back(start);
        while let Some(c) = queue.pop_front() {
            let (i, j, k) = (c / (n * n), (c / n) % n, c % n);
            let here = offset[c].expect("queued voxels are labelled");
            for (axis, step) in [(0, 1i32), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)] {
                let mut at = [i as i32, j as i32, k as i32];
                let mut lifted = here;
                at[axis] += step;
                if at[axis] < 0 {
                    at[axis] += n as i32;
                    lifted[axis] -= 1;
                } else if at[axis] >= n as i32 {
                    at[axis] -= n as i32;
                    lifted[axis] += 1;
                }
                let next = index(at[0] as usize, at[1] as usize, at[2] as usize);
                if values[next] > t {
                    continue;
                }
                match offset[next] {
                    Some(seen) if seen != lifted => return true,
                    Some(_) => {}
                    None => {
                        offset[next] = Some(lifted);
                        queue.push_back(next);
                    }
                }
            }
        }
    }
    false
}

fn grid_values(kind: TpmsKind, n: usize) -> Vec<f64> {
    let h = TAU / n as f64;
    let mut out = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                out.push(kind.f(Point3::new(
                    (i as f64 + 0.5) * h,
                    (j as f64 + 0.5) * h,
                    (k as f64 + 0.5) * h,
                )));
            }
        }
    }
    out
}

/// **Where a solid stops being one piece.** `{F <= t}` gains a loop round the period only at an
/// index-1 critical point of `F`, so the floor is a saddle value, and which saddle is decided by
/// a flood fill: the grid finds spanning a little above the true saddle, since a neck must be a
/// voxel wide, so the floor is the highest index-1 value at or below what the grid finds.
fn solid_floor() {
    println!(
        "
Solid floor -- the isovalue at or below which {{F <= t}} is separate pieces
(Newton on grad F = 0 from 14^3 starts; lifted flood fill on 64^3, 40 bisections; fill from
{FULL} R3 points; control: the 40^3 flood the kernel's test runs, at the floor -/+ 0.05)
"
    );
    println!(
        "{:>15}  {:>10}  {:>16}  {:>10}  {:>9}  {:>9}  {:>8}  {:>8}",
        "kind", "closed", "saddle (Newton)", "grid 64^3", "fill", "typed", "-0.05", "+0.05"
    );
    let mut table = Vec::new();
    for kind in TpmsKind::ALL {
        let closed: Option<f64> = match kind {
            TpmsKind::Gyroid => Some(-std::f64::consts::SQRT_2),
            TpmsKind::SchwarzP | TpmsKind::SchwarzD => Some(-1.0),
            TpmsKind::SchoenIwp => Some(-3.0),
            TpmsKind::Neovius => Some(-0.75),
            TpmsKind::FischerKochS => None,
        };
        let grid = grid_values(kind, 64);
        let ends = kind.density_isovalues();
        let (mut lo, mut hi) = (ends[0], ends[10]);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if spans(&grid, 64, mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let saddles: Vec<f64> = critical_points(kind)
            .into_iter()
            .filter(|(_, index, _)| *index == 1)
            .map(|(value, _, _)| value)
            .collect();
        let saddle = saddles
            .iter()
            .copied()
            .filter(|v| *v <= hi + 1e-9)
            .fold(f64::NEG_INFINITY, f64::max);
        let full = sorted_over_period(FULL, |u| kind.f(Point3::from(u.coords * TAU)));
        let fill = fraction_below(&full, saddle);
        let control = grid_values(kind, 40);
        println!(
            "{:>15}  {:>10}  {:>16.12}  {:>10.4}  {:>9.5}  {:>9.4}  {:>8}  {:>8}",
            kind.label(),
            closed.map_or("--".to_string(), |c| format!("{c:.6}")),
            saddle,
            hi,
            fill,
            kind.density_for_isovalue(saddle),
            if spans(&control, 40, saddle - 0.05) { "SPANS" } else { "apart" },
            if spans(&control, 40, saddle + 0.05) { "spans" } else { "APART" },
        );
        if let Some(c) = closed
            && (c - saddle).abs() > 1e-9
        {
            println!("                 ** the Newton saddle is not the closed form: {c} **");
        }
        table.push((kind, saddle, fill));
    }
    println!("\n// SOLID_FLOOR, (isovalue, fill) per kind in TpmsKind::ALL order:");
    for (kind, saddle, fill) in table {
        println!("    ({saddle:.12}, {fill:.5}), // {}", kind.label());
    }
}

/// **A sheet's fill against its wall.** Item 87's rows: every 1% of wall to cell from 1% to
/// the kind's wall_limit, and `|F| <= π·g·(t/L)` is exactly what Lattice::distance calls
/// material for a centred sheet on a cubic rectangular map. Printed as ρ/(t/L), which is nearly
/// flat and tends to the surface area per cell, so linear interpolation between 1% rows is
/// near-exact.
fn sheet_density() {
    println!(
        "
Sheet density -- the fill of |F| <= pi*g*w, w = wall/cell, every 0.01 to the wall limit
({FULL} R3 points; SHEET_DENSITY_<KIND> in lattice/tpms.rs, as rho/w)
"
    );
    for kind in TpmsKind::ALL {
        let g = kind.surface_gradient();
        let full = sorted_over_period(FULL, |u| kind.f(Point3::from(u.coords * TAU)).abs());
        let part = sorted_over_period(PART, |u| kind.f(Point3::from(u.coords * TAU)).abs());
        let test = sorted_over_period(TEST, |u| kind.f(Point3::from(u.coords * TAU)).abs());
        let spot = sorted_over_period(SPOT, |u| kind.f(Point3::from(u.coords * TAU)).abs());
        let count = (kind.wall_limit() * 100.0).round() as usize;
        let w = |i: usize| (i + 1) as f64 / 100.0;
        let iso = |i: usize| std::f64::consts::PI * g * w(i);
        let rows: Vec<f64> = (0..count).map(|i| fraction_below(&full, iso(i)) / w(i)).collect();
        let worst = (0..count)
            .map(|i| {
                let a = fraction_below(&full, iso(i));
                (fraction_below(&part, iso(i)) - a).abs() / a
            })
            .fold(0.0f64, f64::max);
        print_rust_array(&format!("SHEET_DENSITY_{}", const_name(kind)), &rows, 5);
        println!(
            "// {}: {count} rows, |2^22 - 2^24| at most {:.3}%; rho at the limit {:.4}",
            kind.label(),
            100.0 * worst,
            rows[count - 1] * w(count - 1),
        );
        // Every row as pasted, against what the first 2^20 points read; and the thinnest row
        // through the field on the first 2^22, which is how the kernel's test holds it.
        let pasted = |i: usize| (rows[i] * 1e5).round() / 1e5 * w(i);
        let off = |i: usize| (fraction_below(&spot, iso(i)) - pasted(i)).abs() / pasted(i);
        let thinnest = Lattice {
            cell: CellKind::Tpms(kind),
            map: LatticeMap::Rectangular { cell: cube(10.0) },
            phase: Phase::sheet(w(0) * 10.0),
        };
        println!(
            "//   every row as pasted against the first 2^20 points: {:.3}% at the first, at most {:.3}% elsewhere; the first row through the field on 2^22 points: {:+.3}%",
            100.0 * off(0),
            100.0 * (1..count).map(off).fold(0.0f64, f64::max),
            100.0 * (field_fill(&thinnest, cube(10.0), PART) - pasted(0)) / pasted(0),
        );
        for i in [0, count / 2, count - 1] {
            let table = rows[i] * w(i);
            let at_test = fraction_below(&test, iso(i));
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Rectangular { cell: cube(10.0) },
                phase: Phase::sheet(w(i) * 10.0),
            };
            let through_field = field_fill(&lattice, cube(10.0), SPOT);
            let formula = fraction_below(&spot, iso(i));
            println!(
                "//   w {:.2}: table {:.5}; the first 2^18 read {:+.3}% off it, the first 2^20 {:+.3}%; field and formula on the 2^20 {} points apart",
                w(i),
                table,
                100.0 * (at_test - table) / table,
                100.0 * (through_field - table) / table,
                ((through_field - formula) * SPOT as f64).round(),
            );
        }
    }
}

/// The fraction of a sorted sample that lies at or below `iso` — the achieved density, measured
/// the same way the table was built so the two are comparable.
fn fraction_below(sorted: &[f64], iso: f64) -> f64 {
    let i = sorted.partition_point(|v| *v <= iso);
    i as f64 / sorted.len() as f64
}

/// RMS of each partial derivative separately, over the surface.
///
/// `argument_gradient` claims the three are equal and splits the whole gradient evenly between
/// them. This is the measurement that claim is checked against — a vector rather than a scalar
/// exists precisely so a cell that fails it can say so instead of being silently mis-sized.
fn per_argument_rms(kind: TpmsKind) -> [f64; 3] {
    let n = 90;
    let step = TAU / n as f64;
    let mut sums = [0.0f64; 3];
    let mut count = 0usize;
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let p = Point3::new(i as f64 * step, j as f64 * step, l as f64 * step);
                if kind.f(p).abs() >= 0.05 {
                    continue;
                }
                let g = kind.gradient(p);
                sums[0] += g.x * g.x;
                sums[1] += g.y * g.y;
                sums[2] += g.z * g.z;
                count += 1;
            }
        }
    }
    let n = count.max(1) as f64;
    [
        (sums[0] / n).sqrt(),
        (sums[1] / n).sqrt(),
        (sums[2] / n).sqrt(),
    ]
}

/// The densities the stored table samples. Ten per cent apart, which is close enough that
/// linear interpolation between neighbours is well inside the error of the phrase "50% dense".
const DENSITIES: [f64; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// The isovalue at which `F <= iso` fills the requested fraction of one cell.
///
/// Measured rather than derived, because the map from isovalue to volume fraction has no closed
/// form for any of these surfaces and is *not* symmetric about zero for all of them. Sampling
/// `f` over one full period in each argument, sorting, and reading the quantile is the whole
/// method — the grid is uniform, so the sorted position of a value is the volume fraction below
/// it.
fn quantile(values: &[f64], density: f64) -> f64 {
    if density <= 0.0 {
        return values[0];
    }
    if density >= 1.0 {
        return values[values.len() - 1];
    }
    values[((values.len() - 1) as f64 * density).round() as usize]
}

/// `F` over one full cell at unit frequency.
fn over_one_cell(kind: TpmsKind) -> Vec<f64> {
    let n = 128;
    let step = TAU / n as f64;
    let mut out = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let p = Point3::new(i as f64 * step, j as f64 * step, l as f64 * step);
                out.push(kind.f(p));
            }
        }
    }
    out
}

/// `|∇F|` over a whole cell at unit frequency — everywhere, not only at the surface, because a
/// sphere tracer evaluates the field far from the zero set.
fn everywhere(kind: TpmsKind) -> Vec<f64> {
    let n = 60;
    let step = TAU / n as f64;
    let mut out = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let p = Point3::new(i as f64 * step, j as f64 * step, l as f64 * step);
                out.push(kind.gradient(p).norm());
            }
        }
    }
    out
}

/// `|∇F|` at the surface: the narrower population that sets the wall thickness.
///
/// "At the surface" is `|F|` near zero rather than exactly zero, since a grid never lands on
/// it. The band is tight enough that widening it does not move the mean.
fn at_surface(kind: TpmsKind) -> Vec<f64> {
    let n = 90;
    let step = TAU / n as f64;
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let p = Point3::new(i as f64 * step, j as f64 * step, l as f64 * step);
                if kind.f(p).abs() < 0.05 {
                    out.push(kind.gradient(p).norm());
                }
            }
        }
    }
    out
}

/// Claimed distance over actual distance, for rays from a grid of points in several
/// directions.
///
/// The march deliberately runs *well past* what the field claimed. An earlier version stopped
/// at the claimed distance and reported that no ray ever reached a wall, which is true and
/// useless: a field that understates by half can never reach a wall inside its own claim, so
/// the measurement could only ever confirm what it assumed.
fn claimed_against_actual(kind: TpmsKind) -> Vec<f64> {
    let cell = 40.0;
    let lattice = Lattice {
        cell: CellKind::Tpms(kind),
        map: fieldform_field::LatticeMap::Rectangular {
            cell: nalgebra::Vector3::new(cell, cell, cell),
        },
        phase: Phase::sheet(4.0),
    };
    let dirs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5773, 0.5773, 0.5773],
        [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0],
        [0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2],
    ];
    let n = 12;
    let step = cell / n as f64;
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let from = Point3::new(i as f64 * step, j as f64 * step, l as f64 * step);
                let claimed = lattice.distance(from);
                if claimed <= 1e-6 {
                    continue;
                }
                for d in dirs {
                    let along = |t: f64| {
                        lattice.distance(Point3::new(
                            from.x + d[0] * t,
                            from.y + d[1] * t,
                            from.z + d[2] * t,
                        ))
                    };
                    // Out to a whole cell, which always contains a wall.
                    let fine = cell / 2000.0;
                    let mut hit = None;
                    for s in 1..=2000 {
                        let t = fine * s as f64;
                        if along(t) <= 0.0 {
                            // Bisect the last interval so the answer is not step-limited.
                            let (mut lo, mut hi) = (t - fine, t);
                            for _ in 0..40 {
                                let mid = 0.5 * (lo + hi);
                                if along(mid) <= 0.0 {
                                    hi = mid;
                                } else {
                                    lo = mid;
                                }
                            }
                            hit = Some(hi);
                            break;
                        }
                    }
                    if let Some(actual) = hit {
                        out.push(claimed / actual);
                    }
                }
            }
        }
    }
    out
}
