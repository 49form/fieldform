// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Item 94's literature rows, read: the tables the lattice-property-ledger design note
//! carries, printed from the compiled rows themselves so the document cannot say what the code
//! does not. In the ledger's order:
//!
//! - the ledger's numbered table, all 75 research rows: each shipped row with its id, what it
//!   is and every correction made to it, each excluded row with its reason;
//! - every row at ρ = 0.1 … 0.5, with its standing (inside its range, outside, refused);
//! - the default per key, and the spread across sources at ρ = 0.30;
//! - where each fit is refused: at or below nought, or above the solid's share;
//! - `d ln E / d ln ρ` at each row's range ends -- how far a 1% density error moves the reading,
//!   which is `n` for a power fit and much more for an offset fit near its zero;
//! - the inverse's worst round trip;
//! - the gyroid sheet's inverse spread at 2%, 5% and 10%, with each answer checked buildable;
//! - the honeycomb's fit read at the exact density and at the thin-wall one.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example property_rows`

use fieldform_field::property_table::{
    default_row, rows_for, DensityBasis, DensityRange, Exponent, Fit, Lower, PropertyClass,
    PropertyRow, Provenance, Refusal, Standing, EXCLUDED, ROWS,
};
use fieldform_field::{Banding, CellKind, PrismKind, TpmsKind};

fn main() {
    ledger();
    every_row_read();
    defaults_and_spread();
    refusals();
    sensitivity();
    inverse();
    gyroid_sheet_inverse();
    honeycomb_readings();
}

fn phase_word(phase: Banding) -> &'static str {
    match phase {
        Banding::Sheet => "sheet",
        Banding::Solid => "solid",
        Banding::Strut => "strut",
    }
}

fn class_word(class: PropertyClass) -> &'static str {
    match class {
        PropertyClass::Modulus => "modulus",
        PropertyClass::Strength => "strength",
    }
}

fn fit_text(fit: Fit) -> String {
    match fit {
        Fit::Power { c, n } => format!("{c:.4}·ρ^{n}"),
        Fit::PowerOffset { c, n, offset } => format!("{c}·ρ^{n} {offset:+}"),
        Fit::TwoBranch { c1, n1, knee, c2, n2, c3 } => {
            format!("{c1}·ρ^{n1} to {knee}, then {c2:.4}·ρ^{n2} {c3:+.4}")
        }
    }
}

fn range_text(range: DensityRange) -> String {
    match range {
        DensityRange::Span { lo: Lower::At(lo), hi } => format!("{lo}–{hi}"),
        DensityRange::Span { lo: Lower::NotStated { surely_from }, hi } => {
            format!("(not stated; surely from {surely_from})–{hi}")
        }
        DensityRange::Single(at) => format!("one density, {at}"),
        DensityRange::NotStated => "not stated".to_string(),
    }
}

fn reading_text(row: &PropertyRow, rho: f64) -> String {
    let reading = row.read(rho);
    let mark = match reading.standing {
        Standing::Inside => "",
        Standing::Outside { .. } => " out",
        Standing::LowestNotStated { .. } => " low?",
        Standing::OnePoint { .. } => " 1pt",
        Standing::RangeNotStated => " ?",
        Standing::Refused(Refusal::NoDensity) => return "no ρ".into(),
        Standing::Refused(Refusal::NotPositive) => return "≤ 0".into(),
        Standing::Refused(Refusal::NearZero) => return "≈ 0".into(),
        Standing::Refused(Refusal::AboveSolidShare) => return "> ρ".into(),
    };
    format!("{:.2}%{mark}", 100.0 * reading.value.expect("a value where not refused"))
}

fn every_row_read() {
    println!("\nEvery row at five densities, % of the solid's own value");
    println!("(out: outside its range; low?: below where its data surely reach; 1pt: one density;");
    println!(" ?: no range stated; ≤ 0, ≈ 0 and > ρ: refused, at or below nought, so near it that a 1%");
    println!(" density error moves it by over 10%, or above the solid's share)\n");
    // The id leads, not the row number: the ledger's test finds a row by a line that *starts*
    // with its number, and a second table led by numbers would answer for it.
    println!("| id | row | cell, phase | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |");
    println!("|---|---|---|---|---|---|---|---|");
    for r in ROWS {
        print!("| `{}` | {} | {} {} |", r.id, r.ledger_row, r.cell.label(), phase_word(r.phase));
        for rho in [0.1, 0.2, 0.3, 0.4, 0.5] {
            print!(" {} |", reading_text(r, rho));
        }
        println!();
    }
}

/// Every key, once, in table order.
fn keys() -> Vec<(CellKind, Banding, PropertyClass)> {
    let mut out: Vec<(CellKind, Banding, PropertyClass)> = Vec::new();
    for r in ROWS {
        let key = (r.cell, r.phase, r.class());
        if !out.contains(&key) {
            out.push(key);
        }
    }
    out
}

fn defaults_and_spread() {
    println!("\nThe default per key, and the spread at ρ = 0.30 across every row of the key that reads there\n");
    println!("| key | default | default at 0.30 | rows reading | spread |");
    println!("|---|---|---|---|---|");
    for (cell, phase, class) in keys() {
        let default = default_row(cell, phase, class).expect("a key with rows has a default");
        let values: Vec<f64> =
            rows_for(cell, phase, class).filter_map(|r| r.read(0.30).value).collect();
        let spread = if values.len() > 1 {
            let hi = values.iter().copied().fold(f64::MIN, f64::max);
            let lo = values.iter().copied().fold(f64::MAX, f64::min);
            format!("{:.1}×", hi / lo)
        } else {
            "--".into()
        };
        println!(
            "| {} {} {} | `{}` | {} | {} | {} |",
            cell.label(),
            phase_word(phase),
            class_word(class),
            default.id,
            reading_text(default, 0.30),
            values.len(),
            spread
        );
    }
    println!("\n{} keys.", keys().len());
}

fn refusals() {
    println!("\nWhere each fit is refused on (0, 1], to 0.001 (rows refused nowhere are left out)\n");
    println!("| id | class | refused | why |");
    println!("|---|---|---|---|");
    for r in ROWS {
        let mut spans: Vec<(f64, f64, Refusal)> = Vec::new();
        for i in 1..=1000 {
            let rho = i as f64 / 1000.0;
            if let Standing::Refused(why) = r.read(rho).standing {
                match spans.last_mut() {
                    Some(last) if last.2 == why && (last.1 - (rho - 0.001)).abs() < 1e-9 => last.1 = rho,
                    _ => spans.push((rho, rho, why)),
                }
            }
        }
        for (from, to, why) in spans {
            let why = match why {
                Refusal::NotPositive => "at or below nought",
                Refusal::NearZero => "so near nought a 1% density error moves it by over 10%",
                Refusal::AboveSolidShare => "above the solid's share",
                Refusal::NoDensity => "no density",
            };
            println!("| `{}` | {} | {from:.3}–{to:.3} | {why} |", r.id, class_word(r.class()));
        }
    }
}

fn sensitivity() {
    println!("\nd ln E / d ln ρ at each row's range ends: what a 1% density error does to the reading\n");
    println!("| id | range | at the low end | at the high end |");
    println!("|---|---|---|---|");
    // `Fit::log_slope`, the number `PropertyRow::read` refuses a reading past.
    let slope = |r: &PropertyRow, rho: f64| {
        let s = r.fit.log_slope(rho);
        if s.is_finite() { format!("{s:.2}") } else { "not positive".into() }
    };
    for r in ROWS {
        let ends = match r.range {
            DensityRange::Span { lo: Lower::At(lo), hi } => Some((lo, hi)),
            DensityRange::Span { lo: Lower::NotStated { surely_from }, hi } => Some((surely_from, hi)),
            DensityRange::Single(at) => Some((at, at)),
            DensityRange::NotStated => None,
        };
        if let Some((lo, hi)) = ends {
            println!("| `{}` | {} | {} | {} |", r.id, range_text(r.range), slope(r, lo), slope(r, hi));
        }
    }
}

fn inverse() {
    let mut worst = (0.0f64, "", 0.0);
    for r in ROWS {
        for i in 5..=95 {
            let rho = i as f64 / 100.0;
            if let Some(v) = r.read(rho).value {
                let back = r.density_for(v).expect("a value read inverts");
                if (back - rho).abs() > worst.0 {
                    worst = ((back - rho).abs(), r.id, rho);
                }
            }
        }
    }
    println!(
        "\nThe inverse's worst round trip over every row and ρ = 0.05 … 0.95: {:.1e} ({} at {})",
        worst.0, worst.1, worst.2
    );
}

/// The density a gyroid sheet's wall builds, inverted: the wall over the cell that fills `rho`,
/// or `None` past the wall limit.
fn gyroid_wall_for(rho: f64) -> Option<f64> {
    let kind = TpmsKind::Gyroid;
    let limit = kind.wall_limit();
    if kind.sheet_density(limit)? < rho {
        return None;
    }
    let (mut lo, mut hi) = (0.0, limit);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if kind.sheet_density(mid).expect("inside the limit") < rho {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(hi)
}

fn gyroid_sheet_inverse() {
    println!("\nThe gyroid sheet's inverse: the density, and the wall on a 10 mm cell, each source needs\n");
    println!("| target | source | density | wall on a 10 mm cell |");
    println!("|---|---|---|---|");
    for target in [0.02, 0.05, 0.10] {
        for r in rows_for(CellKind::Tpms(TpmsKind::Gyroid), Banding::Sheet, PropertyClass::Modulus) {
            let answer = match r.density_for(target) {
                None => ("does not reach it".to_string(), "--".to_string()),
                Some(rho) => match gyroid_wall_for(rho) {
                    Some(w) => (format!("{rho:.3}"), format!("{:.2} mm", 10.0 * w)),
                    None => (
                        format!("{rho:.3}"),
                        format!(
                            "not buildable: the walls meet at {:.3}",
                            TpmsKind::Gyroid.sheet_density(TpmsKind::Gyroid.wall_limit()).unwrap()
                        ),
                    ),
                },
            };
            println!("| {:.0}% | {} | {} | {} |", 100.0 * target, r.source.short, answer.0, answer.1);
        }
    }
}

fn honeycomb_readings() {
    println!("\nThe honeycomb's fit read at the exact density and at the thin-wall 2t/(√3 l)\n");
    println!("| t/l | exact ρ | thin-wall ρ | at exact | at thin-wall | ratio |");
    println!("|---|---|---|---|---|---|");
    let row = ROWS.iter().find(|r| r.id == "beigrezaee26-honeycomb-sheet-e").expect("the honeycomb row");
    for tl in [0.05, 0.1, 0.3, 0.5] {
        // `l` is the hexagon's side, so the flat-to-flat width is √3·l.
        let exact = PrismKind::Honeycomb.wall_density(tl / 3f64.sqrt()).expect("a fraction");
        let thin = 2.0 * tl / 3f64.sqrt();
        let (a, b) = (row.fit.at(exact), row.fit.at(thin));
        println!(
            "| {tl} | {exact:.4} | {thin:.4} | {:.3}% | {:.3}% | {:.3} |",
            100.0 * a,
            100.0 * b,
            b / a
        );
    }
}

/// The ledger's table: every research row at its own number, shipped with its id, what it is and
/// every correction made to it, or excluded with its reason.
/// `every_ledger_row_is_shipped_or_excluded_with_a_reason` reads it back.
fn ledger() {
    let caveat = review_caveat();
    println!("The ledger: all 75 research rows\n");
    println!(
        "Every row read only from Distefano 2023's Table 4 opens its corrections with the same \
         caveat, written \"the review's caveat\" in its row: {caveat}\n"
    );
    println!(
        "| row | status | id or source | cell, phase, property | fit | what it is | corrections, or why not |"
    );
    println!("|---|---|---|---|---|---|---|");
    for n in 0..=74u16 {
        if let Some(r) = ROWS.iter().find(|r| r.ledger_row == n) {
            let corrections = match r.notes.strip_prefix(caveat.as_str()) {
                Some(rest) if !caveat.is_empty() => format!("The review's caveat. {}", rest.trim()),
                _ => r.notes.to_string(),
            };
            println!(
                "| {n} | shipped | `{}` | {} {}, {} | {} | {} | {} |",
                r.id,
                r.cell.label(),
                phase_word(r.phase),
                property_word(r),
                fit_text(r.fit),
                what_it_is(r),
                corrections
            );
        } else if let Some(e) = EXCLUDED.iter().find(|e| e.ledger_row == n) {
            println!(
                "| {n} | excluded | {} | {} {}, {} | -- | -- | {} |",
                e.source,
                e.cell.map_or("--", |c| c.label()),
                e.phase.map_or("phase unknown", phase_word),
                class_word(e.class),
                e.reason
            );
        } else {
            println!("| {n} | MISSING | | | | | |");
        }
    }
}

/// The sentences every row read only from Distefano 2023's Table 4 opens its notes with -- their
/// longest common prefix, cut back to a whole sentence -- so the ledger can print the caveat once
/// and each row's own correction after it, and still print nothing the code does not hold.
fn review_caveat() -> String {
    let notes: Vec<&str> = ROWS
        .iter()
        .filter(|r| {
            matches!(r.provenance,
                Provenance::Review { via, primary_read: false } if via.starts_with("Distefano"))
        })
        .map(|r| r.notes)
        .collect();
    let Some((first, rest)) = notes.split_first() else {
        return String::new();
    };
    let mut len = first.len();
    for other in rest {
        let shared = first
            .char_indices()
            .zip(other.chars())
            .take_while(|((_, a), b)| a == b)
            .last()
            .map_or(0, |((i, c), _)| i + c.len_utf8());
        len = len.min(shared);
    }
    let prefix = &first[..len];
    prefix.rfind(". ").map_or(String::new(), |end| prefix[..=end].to_string())
}

/// What a shipped row is, in the kernel's own phrases: where it was read, the data, the exponent,
/// the range and what the density axis was, the direction, and what the ratio was divided by.
fn what_it_is(r: &PropertyRow) -> String {
    let read = match r.provenance {
        Provenance::Primary => "read",
        Provenance::Review { primary_read: true, .. } => "quoted, primary read",
        Provenance::Review { primary_read: false, .. } => "quoted, primary not read",
    };
    let exponent = match r.exponent {
        Exponent::Fitted => "n fitted",
        Exponent::Theory => "n from theory",
        Exponent::Assumed => "n assumed",
        Exponent::NotStated => "n not stated as fitted",
    };
    let basis = match r.density_basis {
        DensityBasis::Design => "",
        DensityBasis::Measured => ", on measured density",
        DensityBasis::NotStated => ", density basis not stated",
    };
    let divisor = match r.solid_value_mpa {
        Some(v) if v >= 10_000.0 => format!(" ({} GPa)", v / 1000.0),
        Some(v) => format!(" ({v} MPa)"),
        None => String::new(),
    };
    format!(
        "{}, {}; {read} · {} · {exponent} · ρ {}{basis} · {} · {}{divisor}",
        r.source.short,
        r.source.locator,
        r.data.phrase(),
        range_text(r.range),
        r.direction.phrase(),
        r.solid_basis.phrase()
    )
}

fn property_word(r: &PropertyRow) -> &'static str {
    use fieldform_field::property_table::{Property, StrengthKind};
    match r.property {
        Property::Modulus => "modulus",
        Property::Strength(StrengthKind::Yield) => "strength (0.2% yield)",
        Property::Strength(StrengthKind::Plateau) => "strength (plateau)",
        Property::Strength(StrengthKind::NotStated) => "strength (not stated)",
    }
}
