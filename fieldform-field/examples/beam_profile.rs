// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a beam cell fills, against its strut radius: the strut tables in `lattice/beam.rs`.
//!
//! A strut lattice is everything within `r` of a set of axes, so its relative density is one
//! function of `r/L` per cell, measured once and pasted back as `STRUT_DENSITY_<KIND>` -- the
//! same "re-run and paste" rule `tpms_profile.rs` keeps for the surfaces. Item 94's property
//! readout reads it, and item 87 adds its metric columns to these same rows rather than
//! measuring a second fill of the same field (`lattice-qualification-spec.md` §2.1 as amended in
//! §9.1, and the reason this example has the name §2.1 gave it).
//!
//! The method: `axis_distance` at 2^24 points of the kernel's own R3 sequence over one cell,
//! sorted once, so the fill at any `r` is a rank. Rows every 1% of radius to cell up to where the
//! struts merge, which is exact per kind (`BeamKind::merge_radius`), printed as `ρ/r²`: that is
//! `π` times the strut length per cell less a node term nearly linear in `r`, so it interpolates
//! where `ρ` itself would not.
//!
//! Four columns say what the table is worth:
//! - **control**: the cubic cell measured the same way against its closed form `3πr² − 8√2r³`,
//!   which should show no difference -- its residual is the method's own error;
//! - **2^22**: the same rows from the first quarter of the points;
//! - **extrapolation**: rows 1 and 2 carried linearly to `r = 0`, against the exact `π·ℓ`;
//! - **spot**: the fill counted from the sign of `Lattice::distance` on a real 10 mm lattice,
//!   against the axis distance on the same points, which proves the conversion matches the field.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example beam_profile`

use fieldform_field::{
    r3_point, BeamKind, CellKind, Lattice, LatticeMap, Phase, PrismKind, ScalarField,
};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::f64::consts::PI;

const FULL: u64 = 1 << 24;
const PART: u64 = 1 << 22;
/// What the kernel's tests sample: the first 2^18 of the same points.
const TEST: u64 = 1 << 18;
const SPOT: u64 = 1 << 20;

fn main() {
    println!(
        "Strut density -- the fill of axis_distance <= r, r = radius/cell, every 0.01 to the merge
radius ({FULL} R3 points over one cell; STRUT_DENSITY_<KIND> in lattice/beam.rs, as rho/r^2)
"
    );
    for beam in BeamKind::ALL {
        let merge = beam.merge_radius();
        let length = beam.strut_length_per_cell();
        let full = sorted_over_cell(FULL, |u| beam.axis_distance(u));
        let part = sorted_over_cell(PART, |u| beam.axis_distance(u));
        let test = sorted_over_cell(TEST, |u| beam.axis_distance(u));
        let spot = sorted_over_cell(SPOT, |u| beam.axis_distance(u));

        let last_hundredth = (merge * 100.0 + 1e-9).floor() as usize;
        let mut radii: Vec<f64> = (0..=last_hundredth).map(|k| k as f64 / 100.0).collect();
        if *radii.last().expect("row 0 at least") < merge - 1e-12 {
            radii.push(merge);
        }
        let coefficient = |sorted: &[f64], k: usize| {
            if k == 0 {
                PI * length
            } else {
                fraction_below(sorted, radii[k]) / (radii[k] * radii[k])
            }
        };
        let rows: Vec<f64> = (0..radii.len()).map(|k| coefficient(&full, k)).collect();

        // 2^22 against 2^24, row by row: the first row on its own, since 1% of a cell is the
        // thinnest strut anyone types and the fewest points land on it.
        let quarter = |k: usize| (coefficient(&part, k) - rows[k]).abs() / rows[k];
        let quarter_first = quarter(1);
        let quarter_rest = (2..radii.len()).map(quarter).fold(0.0f64, f64::max);

        let name = match beam {
            BeamKind::Cubic => "STRUT_DENSITY_CUBIC (printed, not pasted: the kernel uses the closed form)",
            BeamKind::Bcc => "STRUT_DENSITY_BCC",
            BeamKind::Fcc => "STRUT_DENSITY_FCC",
            BeamKind::Octet => "STRUT_DENSITY_OCTET",
            BeamKind::Diamond => "STRUT_DENSITY_DIAMOND",
        };
        print_rust_array(name, &rows, 5);
        println!(
            "// {}: {} rows, the last at the merge radius {:.6}; |2^22 - 2^24| {:.3}% at r/L 0.01, at most {:.3}% from 0.02",
            beam.label(),
            rows.len(),
            merge,
            100.0 * quarter_first,
            100.0 * quarter_rest,
        );
        // Every row as pasted, against what the first 2^20 points read; and the thinnest row
        // through the field on the first 2^22, which is how the kernel's test holds it.
        let pasted = |k: usize| (rows[k] * 1e5).round() / 1e5 * radii[k] * radii[k];
        let off = |k: usize| (fraction_below(&spot, radii[k]) - pasted(k)).abs() / pasted(k);
        let thinnest = Lattice {
            cell: CellKind::Beam(beam),
            map: LatticeMap::Rectangular { cell: Vector3::new(10.0, 10.0, 10.0) },
            phase: Phase::Strut { radius_mm: radii[1] * 10.0 },
        };
        println!(
            "//   every row as pasted against the first 2^20 points: {:.3}% at r/L 0.01, at most {:.3}% from 0.02; r/L 0.01 through the field on 2^22 points: {:+.3}%",
            100.0 * off(1),
            100.0 * (2..radii.len()).map(off).fold(0.0f64, f64::max),
            100.0 * (field_fill(&thinnest, 10.0, PART) - pasted(1)) / pasted(1),
        );

        // The control arm: the one cell with an answer known in closed form, measured exactly as
        // the others are. Should show no difference; what it does show is the method's error.
        if let BeamKind::Cubic = beam {
            let exact = |r: f64| 3.0 * PI * r * r - 8.0 * 2f64.sqrt() * r * r * r;
            let off = |k: usize| (fraction_below(&full, radii[k]) - exact(radii[k])).abs() / exact(radii[k]);
            let first = off(1);
            let rest = (2..radii.len()).map(off).fold(0.0f64, f64::max);
            println!(
                "//   control: measured against 3 pi r^2 - 8 sqrt2 r^3, {:.3}% at r/L 0.01, at most {:.3}% from 0.02",
                100.0 * first,
                100.0 * rest,
            );
            // What `a_cubic_strut_fills_its_closed_form` sees: the field's own sign at the first
            // 2^20 points, at the four radii it checks.
            for r in [0.02, 0.10, 0.30, 0.50] {
                let lattice = Lattice {
                    cell: CellKind::Beam(beam),
                    map: LatticeMap::Rectangular { cell: Vector3::new(10.0, 10.0, 10.0) },
                    phase: Phase::Strut { radius_mm: r * 10.0 },
                };
                let through_field = field_fill(&lattice, 10.0, SPOT);
                println!(
                    "//   control through the field, first 2^20 points, r/L {r:.2}: {:+.3}% off the closed form",
                    100.0 * (through_field - exact(r)) / exact(r),
                );
            }
        }

        // Rows 1 and 2 carried to r = 0: the coefficient's node term is nearly linear in r, so
        // this should land on pi times the strut length.
        let extrapolated = 2.0 * rows[1] - rows[2];
        println!(
            "//   rows 1 and 2 extrapolated to r = 0: {:.4} against pi*l = {:.4} ({:+.3}%)",
            extrapolated,
            PI * length,
            100.0 * (extrapolated - PI * length) / (PI * length),
        );

        let recomputed = merge_from_struts(beam);
        println!(
            "//   merge radius from the unfolded struts {:.12} against {:.12}; the fill there {:.4}",
            recomputed,
            merge,
            fraction_below(&full, merge),
        );

        for k in [1usize, radii.len() / 2, radii.len() - 1] {
            let r = radii[k];
            let table = rows[k] * r * r;
            let at_test = fraction_below(&test, r);
            let lattice = Lattice {
                cell: CellKind::Beam(beam),
                map: LatticeMap::Rectangular { cell: Vector3::new(10.0, 10.0, 10.0) },
                phase: Phase::Strut { radius_mm: r * 10.0 },
            };
            let through_field = field_fill(&lattice, 10.0, SPOT);
            let formula = fraction_below(&spot, r);
            println!(
                "//   r/L {:.4}: table {:.5}; the first 2^18 read {:+.3}% off it, the first 2^20 {:+.3}%; field and axis distance on the 2^20 {} points apart",
                r,
                table,
                100.0 * (at_test - table) / table,
                100.0 * (through_field - table) / table,
                ((through_field - formula) * SPOT as f64).round(),
            );
        }
    }
    println!(
        "
The stated error of each table is the larger of the control residual and its own 2^22 column;
both are printed per row range above. Re-run this after touching `axis_distance` or a strut
list, and paste the arrays back."
    );

    // The honeycomb has no table: its fill is exact, 1 - (1 - t/across)^2. This is the same
    // check run through its field, over exactly one period -- sqrt(3) flat-to-flat widths by
    // one, which is the box `a_honeycomb_fills_one_minus_its_open_hexagon` samples.
    println!("\nHoneycomb wall -- the closed form against the field, one period, first 2^18 and 2^20 points\n");
    let across = 10.0;
    let period = Vector3::new(3f64.sqrt() * across, across, across);
    for w in [0.05, 0.15, 0.30] {
        let lattice = Lattice {
            cell: CellKind::Prism(PrismKind::Honeycomb),
            map: LatticeMap::Rectangular { cell: period },
            phase: Phase::sheet(w * across),
        };
        let exact = 1.0 - (1.0 - w) * (1.0 - w);
        let count = |n: u64| {
            let inside: u64 = (0..n)
                .into_par_iter()
                .map(|i| {
                    let u = r3_point(i);
                    let p = Point3::new(u.x * period.x, u.y * period.y, u.z * period.z);
                    u64::from(lattice.distance(p) <= 0.0)
                })
                .sum();
            inside as f64 / n as f64
        };
        println!(
            "  t/across {w:.2}: exact {exact:.5}; the first 2^18 read {:+.3}% off it, the first 2^20 {:+.3}%",
            100.0 * (count(TEST) - exact) / exact,
            100.0 * (count(SPOT) - exact) / exact,
        );
    }
}

/// `value` at the first `n` points of the kernel's own sequence over one cell, sorted.
fn sorted_over_cell(n: u64, value: impl Fn(Point3<f64>) -> f64 + Sync) -> Vec<f64> {
    let mut out: Vec<f64> = (0..n).into_par_iter().map(|i| value(r3_point(i))).collect();
    out.par_sort_unstable_by(f64::total_cmp);
    out
}

fn fraction_below(sorted: &[f64], at: f64) -> f64 {
    sorted.partition_point(|v| *v <= at) as f64 / sorted.len() as f64
}

fn field_fill(lattice: &Lattice, cell: f64, n: u64) -> f64 {
    let inside: u64 = (0..n)
        .into_par_iter()
        .map(|i| u64::from(lattice.distance(Point3::from(r3_point(i).coords * cell)) <= 0.0))
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

/// Every strut of one cell, unfolded, from the crystallographic definition -- the same honest
/// list `lattice/tests.rs` checks the folded distance against.
fn struts(beam: BeamKind) -> Vec<(Point3<f64>, Point3<f64>)> {
    let at = |x: f64, y: f64, z: f64| Point3::new(x, y, z);
    let corners: Vec<Point3<f64>> = (0..8)
        .map(|c| at((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64))
        .collect();
    let faces = [
        at(0.5, 0.5, 0.0),
        at(0.5, 0.5, 1.0),
        at(0.5, 0.0, 0.5),
        at(0.5, 1.0, 0.5),
        at(0.0, 0.5, 0.5),
        at(1.0, 0.5, 0.5),
    ];
    match beam {
        BeamKind::Cubic => vec![
            (at(0.0, 0.0, 0.0), at(1.0, 0.0, 0.0)),
            (at(0.0, 0.0, 0.0), at(0.0, 1.0, 0.0)),
            (at(0.0, 0.0, 0.0), at(0.0, 0.0, 1.0)),
        ],
        BeamKind::Fcc => {
            let mut out = struts(BeamKind::Cubic);
            for i in 0..8 {
                for j in i + 1..8 {
                    let (a, b) = (corners[i], corners[j]);
                    if ((a - b).norm() - 2f64.sqrt()).abs() < 1e-9 {
                        out.push((a, b));
                    }
                }
            }
            out
        }
        BeamKind::Octet => {
            let edge = 0.5f64.sqrt();
            let mut out = Vec::new();
            for i in 0..6 {
                for j in i + 1..6 {
                    if ((faces[i] - faces[j]).norm() - edge).abs() < 1e-9 {
                        out.push((faces[i], faces[j]));
                    }
                }
            }
            for k in &corners {
                for f in &faces {
                    if ((k - f).norm() - edge).abs() < 1e-9 {
                        out.push((*k, *f));
                    }
                }
            }
            out
        }
        BeamKind::Diamond => {
            let sites = [at(0.0, 0.0, 0.0), at(0.5, 0.5, 0.0), at(0.5, 0.0, 0.5), at(0.0, 0.5, 0.5)];
            let bond = 3f64.sqrt() / 4.0;
            let mut out = Vec::new();
            for a in sites {
                for b in sites {
                    let b = b + Vector3::new(0.25, 0.25, 0.25);
                    for o in offsets() {
                        if ((a - (b + o)).norm() - bond).abs() < 1e-9 {
                            out.push((a, b + o));
                        }
                    }
                }
            }
            out
        }
        BeamKind::Bcc => corners.iter().map(|c| (*c, at(0.5, 0.5, 0.5))).collect(),
    }
}

fn offsets() -> Vec<Vector3<f64>> {
    let mut out = Vec::new();
    for i in -1..=1 {
        for j in -1..=1 {
            for k in -1..=1 {
                out.push(Vector3::new(i as f64, j as f64, k as f64));
            }
        }
    }
    out
}

/// Half the smallest gap between a strut of the base cell and any strut of the 27 cells round
/// it that shares no node with it and does not cross it: a crossing (FCC's two diagonals on a
/// face) is a node of its own, where struts overlap from the start.
fn merge_from_struts(beam: BeamKind) -> f64 {
    let base = struts(beam);
    let all: Vec<(Point3<f64>, Point3<f64>)> = offsets()
        .into_iter()
        .flat_map(|o| base.iter().map(move |(a, b)| (a + o, b + o)))
        .collect();
    let same = |p: Point3<f64>, q: Point3<f64>| (p - q).norm() < 1e-9;
    let mut best = f64::INFINITY;
    for (a, b) in &base {
        for (c, d) in &all {
            if same(*a, *c) || same(*a, *d) || same(*b, *c) || same(*b, *d) {
                continue;
            }
            let gap = segment_distance(*a, *b, *c, *d);
            if gap < 1e-9 {
                continue;
            }
            best = best.min(gap);
        }
    }
    best * 0.5
}

/// Closest distance between two segments, in closed form (Ericson, *Real-Time Collision
/// Detection*, 5.1.9).
fn segment_distance(p1: Point3<f64>, q1: Point3<f64>, p2: Point3<f64>, q2: Point3<f64>) -> f64 {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(&d1), d2.dot(&d2), d2.dot(&r));
    let (c, b) = (d1.dot(&r), d1.dot(&d2));
    let denominator = a * e - b * b;
    let mut s = if denominator > 1e-14 {
        ((b * f - c * e) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    ((p1 + d1 * s) - (p2 + d2 * t)).norm()
}
