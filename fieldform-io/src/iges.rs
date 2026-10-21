// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! IGES import, for curves.
//!
//! # The format, in the parts that matter
//!
//! An IGES file is fixed-format 80-column card images. Column 73 names the section — Start,
//! Global, Directory, Parameter, Terminate — and 74-80 is a sequence number. Two consequences
//! drive the shape of this parser:
//!
//! - **Data is split across cards mid-token.** A parameter section card holds data only in
//!   columns 1-64, and a number can straddle the boundary: the test file's `1000.0` arrives as
//!   `1000` on one card and `.0` on the next. So cards are concatenated *before* tokenising,
//!   never parsed line by line.
//! - **Delimiters are data.** The parameter and record delimiters are themselves the first two
//!   global parameters, written as Hollerith strings (`1H,` and `1H;`) — which means the file
//!   declares its own punctuation using that punctuation. And a Hollerith string may contain
//!   the delimiter, so splitting on it naively corrupts any string that does. The tokeniser
//!   below consumes `nH` runs by length for that reason.
//!
//! # What this reads
//!
//! Curves. Entity 126 (rational B-spline) is the target, since that is what CAD systems
//! actually emit and what lines, arcs and conics all degrade into exactly. Also 110 (line),
//! 100 (circular arc, built as an exact rational NURBS), 102 (composite curve, expanded into
//! its constituents) and 124 (transformation matrix, applied where referenced).
//!
//! Anything else — surfaces, solids, annotation — is **counted and reported**, never silently
//! dropped. A file whose geometry did not come through should say so rather than import as an
//! empty document.

use fieldform_scene::{Curve, CurveError, CurveId};
use nalgebra::{Matrix3, Point3, Vector3};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum IgesError {
    #[error("failed to read IGES file: {0}")]
    Io(#[from] std::io::Error),
    #[error("not an IGES file: no section codes found in column 73")]
    NotIges,
    #[error("global section is malformed: {0}")]
    Global(String),
    #[error("unit flag {0} is not one this importer knows how to convert to millimetres")]
    UnknownUnits(i64),
    #[error("entity {entity} (type {entity_type}) is malformed: {reason}")]
    Entity {
        entity: usize,
        entity_type: u32,
        reason: String,
    },
    #[error("entity {entity} produced an invalid curve: {source}")]
    Curve {
        entity: usize,
        #[source]
        source: CurveError,
    },
}

/// What an IGES file yielded.
pub struct IgesImport {
    pub curves: Vec<Curve>,
    /// Entity types present but not converted, and how many of each. Reported so an import
    /// that dropped the interesting half of a file cannot look like a successful one.
    pub unsupported: BTreeMap<u32, usize>,
    /// Millimetres per file unit, from the global section's unit flag and model space scale.
    pub scale_to_mm: f64,
    /// The units name the file declared, for reporting.
    pub units: String,
}

impl IgesImport {
    /// A one-line summary for the UI: what came in, and what did not.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} curve{} in {}",
            self.curves.len(),
            if self.curves.len() == 1 { "" } else { "s" },
            self.units
        );
        if !self.unsupported.is_empty() {
            let skipped: Vec<String> = self
                .unsupported
                .iter()
                .map(|(ty, n)| format!("{n}x type {ty} ({})", entity_name(*ty)))
                .collect();
            s.push_str(&format!("; skipped {}", skipped.join(", ")));
        }
        s
    }
}

/// Human-readable names for the entity types most likely to turn up unsupported, so the
/// message says "surface" rather than only "type 128".
fn entity_name(ty: u32) -> &'static str {
    match ty {
        100 => "circular arc",
        102 => "composite curve",
        104 => "conic arc",
        106 => "copious data",
        108 => "plane",
        110 => "line",
        112 => "parametric spline curve",
        114 => "parametric spline surface",
        116 => "point",
        118 => "ruled surface",
        120 => "surface of revolution",
        122 => "tabulated cylinder",
        124 => "transformation matrix",
        126 => "rational B-spline curve",
        128 => "rational B-spline surface",
        142 => "curve on a parametric surface",
        144 => "trimmed surface",
        186 => "manifold solid B-rep",
        314 => "colour definition",
        402 => "associativity instance",
        406 => "property",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------------------
// Card and token handling
// ---------------------------------------------------------------------------------------

/// One directory entry: two cards, nine 8-column fields each.
#[derive(Debug, Clone)]
struct DirectoryEntry {
    /// The odd sequence number that other entities point at.
    number: usize,
    entity_type: u32,
    /// DE number of a type-124 transform, or 0 for none.
    transform: usize,
}

/// Splits an IGES free-format string into parameter tokens.
///
/// Hollerith runs (`5HCodex`) are consumed by their declared length rather than scanned for
/// delimiters, because the string may legitimately contain one — the file's own delimiter
/// declaration, `1H,`, is exactly that case.
fn tokenize(s: &str, param_delim: char, record_delim: char) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut current = String::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c == 'H' && !current.is_empty() && current.chars().all(|d| d.is_ascii_digit()) {
            let n: usize = current.parse().unwrap_or(0);
            let start = i + 1;
            let end = (start + n).min(chars.len());
            current = chars[start..end].iter().collect();
            i = end;
            continue;
        }
        if c == param_delim || c == record_delim {
            out.push(current.trim().to_string());
            current = String::new();
            i += 1;
            if c == record_delim {
                return out;
            }
            continue;
        }
        current.push(c);
        i += 1;
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// Parses an IGES real. Fortran-style `D` exponents (`1.0D+02`) are legal and common.
fn as_f64(token: Option<&String>) -> Option<f64> {
    let t = token?.trim();
    if t.is_empty() {
        return None;
    }
    t.replace(['D', 'd'], "e").parse().ok()
}

fn as_i64(token: Option<&String>) -> Option<i64> {
    let t = token?.trim();
    if t.is_empty() {
        return None;
    }
    t.parse().ok()
}

/// Millimetres per unit for an IGES unit flag.
fn units_to_mm(flag: i64) -> Option<(f64, &'static str)> {
    Some(match flag {
        1 => (25.4, "inches"),
        2 => (1.0, "millimetres"),
        4 => (304.8, "feet"),
        5 => (1_609_344.0, "miles"),
        6 => (1000.0, "metres"),
        7 => (1_000_000.0, "kilometres"),
        8 => (0.0254, "mils"),
        9 => (0.001, "microns"),
        10 => (10.0, "centimetres"),
        11 => (0.0000254, "microinches"),
        _ => return None, // 3 is "as declared by name", which needs a lookup we don't have
    })
}

// ---------------------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------------------

/// Reads every curve in an IGES file, converted to canonical millimetres.
pub fn import_iges(path: &Path) -> Result<IgesImport, IgesError> {
    let text = std::fs::read_to_string(path)?;
    import_iges_str(&text)
}

/// The parser proper, taking text so it can be tested without touching the filesystem.
pub(crate) fn import_iges_str(text: &str) -> Result<IgesImport, IgesError> {
    let mut global_cards = String::new();
    let mut directory_cards: Vec<(usize, String)> = Vec::new();
    // Parameter cards, grouped by the directory entry they belong to.
    let mut parameter_cards: BTreeMap<usize, String> = BTreeMap::new();
    let mut saw_section = false;

    for line in text.lines() {
        let chars: Vec<char> = line.chars().collect();
        if chars.len() < 73 {
            continue; // short or blank card; nothing addressable in it
        }
        let section = chars[72];
        let body: String = chars[..72].iter().collect();
        let sequence: usize = chars[72..]
            .iter()
            .skip(1)
            .collect::<String>()
            .trim()
            .parse()
            .unwrap_or(0);

        match section {
            'S' | 's' => saw_section = true,
            'G' | 'g' => {
                saw_section = true;
                global_cards.push_str(body.trim_end());
            }
            'D' | 'd' => {
                saw_section = true;
                directory_cards.push((sequence, body));
            }
            'P' | 'p' => {
                saw_section = true;
                // Columns 1-64 are data; 65-72 point back at the owning directory entry.
                let data: String = chars[..64.min(chars.len())].iter().collect();
                let owner: usize = chars[64..72].iter().collect::<String>().trim().parse().unwrap_or(0);
                parameter_cards
                    .entry(owner)
                    .or_default()
                    .push_str(data.trim_end());
            }
            'T' | 't' => saw_section = true,
            _ => {}
        }
    }

    if !saw_section {
        return Err(IgesError::NotIges);
    }

    // --- global section: delimiters first, since everything else depends on them ---
    let (param_delim, record_delim) = delimiters(&global_cards);
    let globals = tokenize(&global_cards, param_delim, record_delim);

    let unit_flag = as_i64(globals.get(13)).unwrap_or(2);
    let (unit_mm, unit_name) = units_to_mm(unit_flag).ok_or(IgesError::UnknownUnits(unit_flag))?;
    // Model space scale relates file coordinates to model space; 1.0 in almost every file.
    let model_scale = as_f64(globals.get(12)).filter(|s| *s > 0.0).unwrap_or(1.0);
    let scale_to_mm = unit_mm * model_scale;

    // --- directory entries: two cards each ---
    let mut entries: Vec<DirectoryEntry> = Vec::new();
    for pair in directory_cards.chunks(2) {
        if pair.len() < 2 {
            break;
        }
        let field = |card: &str, i: usize| -> String {
            card.chars()
                .skip(i * 8)
                .take(8)
                .collect::<String>()
                .trim()
                .to_string()
        };
        let entity_type: u32 = field(&pair[0].1, 0).parse().unwrap_or(0);
        entries.push(DirectoryEntry {
            number: pair[0].0,
            entity_type,
            transform: field(&pair[0].1, 6).parse().unwrap_or(0),
        });
    }

    // --- transforms, resolved before the geometry that references them ---
    let mut transforms: BTreeMap<usize, (Matrix3<f64>, Vector3<f64>)> = BTreeMap::new();
    for e in entries.iter().filter(|e| e.entity_type == 124) {
        if let Some(raw) = parameter_cards.get(&e.number) {
            let t = tokenize(raw, param_delim, record_delim);
            let v: Vec<f64> = (1..13).map(|i| as_f64(t.get(i)).unwrap_or(0.0)).collect();
            if v.len() == 12 {
                transforms.insert(
                    e.number,
                    (
                        Matrix3::new(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]),
                        Vector3::new(v[3], v[7], v[11]),
                    ),
                );
            }
        }
    }

    // --- geometry ---
    let mut curves = Vec::new();
    let mut unsupported: BTreeMap<u32, usize> = BTreeMap::new();
    for e in &entries {
        if e.entity_type == 124 {
            continue; // resolved above, into `transforms`
        }
        if e.entity_type == 102 {
            // A composite curve is a list of pointers to other entities, and those entities
            // are directory entries in their own right — so they are imported on their own
            // pass and the composite adds nothing. The consequence, worth knowing: a
            // composite arrives as several separate curves rather than one joined curve.
            continue;
        }
        let Some(raw) = parameter_cards.get(&e.number) else {
            *unsupported.entry(e.entity_type).or_default() += 1;
            continue;
        };
        let tokens = tokenize(raw, param_delim, record_delim);

        let built = match e.entity_type {
            126 => nurbs_curve(e, &tokens)?,
            110 => line(e, &tokens)?,
            100 => arc(e, &tokens)?,
            _ => {
                *unsupported.entry(e.entity_type).or_default() += 1;
                continue;
            }
        };

        // Apply the entity's transform, then the file's units, so everything downstream is in
        // canonical millimetres and no later stage has to remember a scale factor.
        let placed = place(built, e, &transforms, scale_to_mm);
        curves.push(placed.map_err(|source| IgesError::Curve {
            entity: e.number,
            source,
        })?);
    }

    Ok(IgesImport {
        curves,
        unsupported,
        scale_to_mm,
        units: unit_name.to_string(),
    })
}

/// Reads the parameter and record delimiters out of the global section.
///
/// They are the first two global parameters, in Hollerith form — so this has to read a little
/// of the section before it knows how to read the rest. Defaults are `,` and `;`, and an empty
/// first parameter means "use the default".
fn delimiters(global: &str) -> (char, char) {
    let chars: Vec<char> = global.chars().collect();
    let mut param = ',';
    let mut record = ';';

    // `1H,` — a one-character Hollerith holding the delimiter itself.
    if chars.len() >= 3 && chars[0] == '1' && chars[1] == 'H' {
        param = chars[2];
        // The record delimiter follows, after the parameter delimiter just read.
        if chars.len() >= 7 && chars[3] == param && chars[4] == '1' && chars[5] == 'H' {
            record = chars[6];
        }
    }
    (param, record)
}

/// A curve still in its entity's definition space, before transform and unit scaling.
struct Raw {
    degree: usize,
    points: Vec<Point3<f64>>,
    weights: Vec<f64>,
    knots: Vec<f64>,
    closed: bool,
    name: String,
}

fn place(
    raw: Raw,
    entry: &DirectoryEntry,
    transforms: &BTreeMap<usize, (Matrix3<f64>, Vector3<f64>)>,
    scale_to_mm: f64,
) -> Result<Curve, CurveError> {
    let points = raw
        .points
        .into_iter()
        .map(|p| {
            let p = match transforms.get(&entry.transform) {
                Some((r, t)) => Point3::from(r * p.coords + t),
                None => p,
            };
            Point3::from(p.coords * scale_to_mm)
        })
        .collect();

    Curve::new(
        CurveId::new(),
        raw.name,
        raw.degree,
        points,
        raw.weights,
        raw.knots,
        raw.closed,
    )
}

/// Entity 126 — rational B-spline curve.
fn nurbs_curve(e: &DirectoryEntry, t: &[String]) -> Result<Raw, IgesError> {
    let bad = |reason: String| IgesError::Entity {
        entity: e.number,
        entity_type: 126,
        reason,
    };

    let k = as_i64(t.get(1)).ok_or_else(|| bad("missing upper index K".into()))? as usize;
    let degree = as_i64(t.get(2)).ok_or_else(|| bad("missing degree M".into()))? as usize;
    let closed = as_i64(t.get(4)).unwrap_or(0) == 1;

    let control_count = k + 1;
    // The knot count follows from K and M: N + 2M + 1 where N = K - M + 1, i.e. K + M + 2 —
    // which is exactly `control_count + degree + 1`, the same relation `Curve` validates.
    let knot_count = k + degree + 2;

    // Everything after the six property flags is one long run of reals, in a fixed order.
    let mut i = 7;
    let mut take = |n: usize, what: &str| -> Result<Vec<f64>, String> {
        let mut v = Vec::with_capacity(n);
        for j in 0..n {
            match as_f64(t.get(i + j)) {
                Some(x) => v.push(x),
                None => return Err(format!("ran out of data reading {what}")),
            }
        }
        i += n;
        Ok(v)
    };

    let knots = take(knot_count, "knots").map_err(bad)?;
    let weights = take(control_count, "weights").map_err(bad)?;
    let coords = take(control_count * 3, "control points").map_err(bad)?;
    let range = take(2, "parameter range").map_err(bad)?;

    let points: Vec<Point3<f64>> = coords
        .chunks_exact(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();

    // V(0)/V(1) may name a subrange of the knot domain — a trimmed curve. Honouring that
    // needs knot insertion, which this importer does not do, so say so rather than silently
    // importing the untrimmed curve and calling it the same thing.
    let (d0, d1) = (knots[degree], knots[control_count]);
    let tol = (d1 - d0).abs() * 1e-9 + 1e-9;
    if (range[0] - d0).abs() > tol || (range[1] - d1).abs() > tol {
        return Err(bad(format!(
            "curve is trimmed to [{}, {}] of its [{d0}, {d1}] domain; trimmed curves are not \
             supported yet (it needs knot insertion)",
            range[0], range[1]
        )));
    }

    Ok(Raw {
        degree,
        points,
        weights,
        knots,
        closed,
        name: format!("Curve {}", e.number),
    })
}

/// Entity 110 — a straight line, as a degree-1 curve.
fn line(e: &DirectoryEntry, t: &[String]) -> Result<Raw, IgesError> {
    let bad = |reason: String| IgesError::Entity {
        entity: e.number,
        entity_type: 110,
        reason,
    };
    let v: Vec<f64> = (1..7)
        .map(|i| as_f64(t.get(i)).ok_or_else(|| bad("missing endpoint coordinate".into())))
        .collect::<Result<_, _>>()?;

    Ok(Raw {
        degree: 1,
        points: vec![
            Point3::new(v[0], v[1], v[2]),
            Point3::new(v[3], v[4], v[5]),
        ],
        weights: vec![1.0, 1.0],
        knots: vec![0.0, 0.0, 1.0, 1.0],
        closed: false,
        name: format!("Line {}", e.number),
    })
}

/// Entity 100 — circular arc, as an *exact* rational NURBS rather than a polyline.
///
/// The arc is defined in a plane at constant Z in its own definition space, counterclockwise
/// from start to end. Built from quadratic rational Bezier spans of at most 90 degrees, the
/// standard construction: the middle control point sits where the end tangents meet, at
/// `r / cos(half)` from the centre, carrying weight `cos(half)`. That reproduces the circle
/// to floating point, which a polyline approximation would not.
fn arc(e: &DirectoryEntry, t: &[String]) -> Result<Raw, IgesError> {
    let bad = |reason: String| IgesError::Entity {
        entity: e.number,
        entity_type: 100,
        reason,
    };
    let v: Vec<f64> = (1..8)
        .map(|i| as_f64(t.get(i)).ok_or_else(|| bad("missing arc parameter".into())))
        .collect::<Result<_, _>>()?;

    let (zt, cx, cy) = (v[0], v[1], v[2]);
    let (sx, sy, ex, ey) = (v[3], v[4], v[5], v[6]);

    let radius = ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt();
    if radius <= 0.0 {
        return Err(bad("arc has zero radius".into()));
    }

    let start_angle = (sy - cy).atan2(sx - cx);
    let end_angle = (ey - cy).atan2(ex - cx);
    let mut sweep = end_angle - start_angle;
    // IGES arcs run counterclockwise; a start that equals the end means a full circle.
    while sweep <= 1e-12 {
        sweep += std::f64::consts::TAU;
    }
    let closed = (sweep - std::f64::consts::TAU).abs() < 1e-9;

    let spans = (sweep / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as usize;
    let span_angle = sweep / spans as f64;
    let half = span_angle / 2.0;
    let middle_weight = half.cos();
    let middle_radius = radius / half.cos();

    let at = |angle: f64, r: f64| Point3::new(cx + r * angle.cos(), cy + r * angle.sin(), zt);

    let mut points = vec![at(start_angle, radius)];
    let mut weights = vec![1.0];
    for i in 0..spans {
        let a0 = start_angle + span_angle * i as f64;
        points.push(at(a0 + half, middle_radius));
        weights.push(middle_weight);
        points.push(at(a0 + span_angle, radius));
        weights.push(1.0);
    }

    // Clamped degree-2 knots with each interior knot doubled: 0,0,0, 1,1, 2,2, ..., n,n,n.
    let mut knots = vec![0.0, 0.0, 0.0];
    for i in 1..spans {
        knots.push(i as f64);
        knots.push(i as f64);
    }
    let last = spans as f64;
    knots.extend_from_slice(&[last, last, last]);

    Ok(Raw {
        degree: 2,
        points,
        weights,
        knots,
        closed,
        name: format!("Arc {}", e.number),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file on the user's desktop, verbatim. Kept inline so the test does not depend on a
    /// path outside the repo, and so the exact card layout — including the `1000.0` that
    /// straddles two parameter cards — is what gets exercised.
    const NURBS_2000MM: &str = concat!(
"Fieldform NURBS curve: endpoints X=-1000mm and X=1000mm                 S      1\n",
"1H,,1H;,5HCodex,32Hfieldform_nurbs_curve_2000mm.igs,6HPython,5HCodex,32,G      1\n",
"38,6,308,15,5HCodex,1.0,2,2HMM,1,0.01,15H20260823.170000,0.001,1000.0,5HG      2\n",
"Codex,9HFieldform,11,0,15H20260823.170000;                              G      3\n",
"     126       1       0       0       0       0       0       000000000D      1\n",
"     126       0       0       3       0       0       0 5HNURBS       0D      2\n",
"126,3,3,1,0,1,0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,       1P      1\n",
"-1000.0,0.0,0.0,-333.333333,500.0,0.0,333.333333,-500.0,0.0,1000       1P      2\n",
".0,0.0,0.0,0.0,1.0,0.0,0.0,1.0;                                        1P      3\n",
"S      1G      3D      2P      3                                        T      1\n",
    );

    #[test]
    fn reads_the_nurbs_curve_from_the_test_file() {
        let import = import_iges_str(NURBS_2000MM).expect("valid IGES");
        assert_eq!(import.curves.len(), 1, "{}", import.summary());
        assert!(import.unsupported.is_empty(), "{:?}", import.unsupported);
        assert_eq!(import.units, "millimetres");
        assert_eq!(import.scale_to_mm, 1.0);

        let c = &import.curves[0];
        assert_eq!(c.degree(), 3);
        assert_eq!(c.control_points().len(), 4);
        assert!(!c.is_closed());
        assert!(!c.is_rational(), "every weight is 1, so it is polynomial");
    }

    /// The headline fact about this file is in its name: the endpoints are 2000 mm apart.
    /// Checking that catches a units error, a control-point misread, and a knot-vector
    /// misalignment all at once.
    #[test]
    fn the_imported_curve_spans_2000_millimetres() {
        let import = import_iges_str(NURBS_2000MM).unwrap();
        let c = &import.curves[0];
        let (lo, hi) = c.domain();

        let start = c.evaluate(lo);
        let end = c.evaluate(hi);
        assert!((start - Point3::new(-1000.0, 0.0, 0.0)).norm() < 1e-6, "{start:?}");
        assert!((end - Point3::new(1000.0, 0.0, 0.0)).norm() < 1e-6, "{end:?}");
        assert!(((end - start).norm() - 2000.0).abs() < 1e-6);
    }

    /// `1000.0` is written as `1000` at the end of one parameter card and `.0` at the start of
    /// the next. Parsing cards individually would read it as 1000 and 0 — two values where
    /// there is one — and shift every subsequent parameter. That failure is silent, which is
    /// why it gets its own test.
    #[test]
    fn a_number_split_across_two_cards_is_read_as_one_value() {
        let import = import_iges_str(NURBS_2000MM).unwrap();
        let cp = import.curves[0].control_points();
        assert_eq!(cp.len(), 4, "a shifted parse would change the control point count");
        assert!(
            (cp[3] - Point3::new(1000.0, 0.0, 0.0)).norm() < 1e-6,
            "last control point was {:?}; the split 1000.0 did not reassemble",
            cp[3]
        );
    }

    /// A Hollerith string can contain the parameter delimiter — the file's own delimiter
    /// declaration `1H,` is precisely that. Splitting naively on commas would break the global
    /// section at its first parameter.
    #[test]
    fn hollerith_strings_containing_the_delimiter_survive_tokenising() {
        let tokens = tokenize("1H,,1H;,5HCodex,32,", ',', ';');
        assert_eq!(tokens[0], ",");
        assert_eq!(tokens[1], ";");
        assert_eq!(tokens[2], "Codex");
        assert_eq!(tokens[3], "32");
    }

    /// Builds a well-formed IGES file from card bodies.
    ///
    /// Editing a fixed-format file by string substitution is a trap: lengthening the body
    /// pushes the section code out of column 73 and the card stops parsing. (This test was
    /// written that way first, and failed for exactly that reason.) So variants get assembled
    /// with the columns enforced instead.
    fn build(cards: &[(&str, char)]) -> String {
        let mut out = String::new();
        let mut counters = std::collections::HashMap::new();
        for (body, section) in cards {
            let n = counters.entry(*section).or_insert(0usize);
            *n += 1;
            assert!(body.len() <= 72, "card body overflows column 72: {body:?}");
            out.push_str(&format!("{body:<72}{section}{n:>7}\n"));
        }
        out
    }

    #[test]
    fn units_are_converted_to_millimetres() {
        // The same curve, declared in inches (unit flag 1) instead of millimetres.
        let inches = build(&[
            ("Fieldform NURBS curve in inches", 'S'),
            ("1H,,1H;,5HCodex,8Htest.igs,6HPython,5HCodex,32,", 'G'),
            ("38,6,308,15,5HCodex,1.0,1,4HINCH,1,0.01,15H20260823.170000,0.001,", 'G'),
            ("1000.0,5HCodex,9HFieldform,11,0,15H20260823.170000;", 'G'),
            ("     126       1       0       0       0       0       0       000000000", 'D'),
            ("     126       0       0       3       0       0       0 5HNURBS       0", 'D'),
            ("126,3,3,1,0,1,0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,       1", 'P'),
            ("-1000.0,0.0,0.0,-333.333333,500.0,0.0,333.333333,-500.0,0.0,1000       1", 'P'),
            (".0,0.0,0.0,0.0,1.0,0.0,0.0,1.0;                                        1", 'P'),
            ("S      1G      3D      2P      3", 'T'),
        ]);
        let import = import_iges_str(&inches).expect("valid IGES");
        assert_eq!(import.units, "inches");
        assert_eq!(import.scale_to_mm, 25.4);

        let c = &import.curves[0];
        let (lo, hi) = c.domain();
        let span = (c.evaluate(hi) - c.evaluate(lo)).norm();
        assert!(
            (span - 2000.0 * 25.4).abs() < 1e-3,
            "2000 inches should import as {} mm, got {span}",
            2000.0 * 25.4
        );
    }

    #[test]
    fn a_file_with_no_section_codes_is_rejected() {
        assert!(matches!(
            import_iges_str("this is not an IGES file\n"),
            Err(IgesError::NotIges)
        ));
    }

    /// Unsupported entities must be counted, not dropped. An import that silently returned
    /// zero curves for a surface-only file would look like success.
    #[test]
    fn unsupported_entities_are_reported_rather_than_dropped() {
        let with_surface = NURBS_2000MM
            .replace(
"     126       1       0       0       0       0       0       000000000D      1\n",
"     128       1       0       0       0       0       0       000000000D      1\n",
            )
            .replace(
"     126       0       0       3       0       0       0 5HNURBS       0D      2\n",
"     128       0       0       3       0       0       0 5HNURBS       0D      2\n",
            );
        let import = import_iges_str(&with_surface).unwrap();
        assert!(import.curves.is_empty());
        assert_eq!(import.unsupported.get(&128), Some(&1));
        assert!(
            import.summary().contains("rational B-spline surface"),
            "{}",
            import.summary()
        );
    }

    /// An IGES circular arc must come in as an exact circle, not a polyline approximation —
    /// that is the whole reason for carrying rational weights through the importer.
    #[test]
    fn a_circular_arc_imports_as_an_exact_circle() {
        let e = DirectoryEntry {
            number: 1,
            entity_type: 100,
            transform: 0,
        };
        // Quarter arc, radius 25, centred at the origin, from (25,0) to (0,25).
        let tokens: Vec<String> = ["100", "0.0", "0.0", "0.0", "25.0", "0.0", "0.0", "25.0"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let raw = arc(&e, &tokens).expect("valid arc");
        let curve = place(raw, &e, &BTreeMap::new(), 1.0).expect("valid curve");

        let (lo, hi) = curve.domain();
        for i in 0..=32 {
            let t = lo + (hi - lo) * i as f64 / 32.0;
            let p = curve.evaluate(t);
            assert!(
                (p.coords.norm() - 25.0).abs() < 1e-9,
                "at t={t} radius was {}",
                p.coords.norm()
            );
        }
    }

    #[test]
    fn a_line_entity_imports_as_a_two_point_curve() {
        let e = DirectoryEntry {
            number: 1,
            entity_type: 110,
            transform: 0,
        };
        let tokens: Vec<String> = ["110", "0.0", "0.0", "0.0", "10.0", "20.0", "30.0"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let raw = line(&e, &tokens).unwrap();
        let curve = place(raw, &e, &BTreeMap::new(), 1.0).unwrap();
        assert_eq!(curve.degree(), 1);
        assert_eq!(curve.control_points().len(), 2);
        let (lo, hi) = curve.domain();
        assert!((curve.evaluate(hi) - Point3::new(10.0, 20.0, 30.0)).norm() < 1e-9);
        assert!((curve.evaluate(lo) - Point3::origin()).norm() < 1e-9);
    }
}
