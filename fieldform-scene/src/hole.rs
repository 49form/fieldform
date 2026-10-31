// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a hole is told beyond where it goes: its type and sizes, how deep, and how it ends.
//!
//! A hole is a cutter shaped like a drilled hole, placed by two points -- the entry and a point
//! along the axis into the material -- and made already subtracted from a body. It remembers no
//! face: the points are where it was put, exactly as a cylinder's are (the boundary rule).

use serde::{Deserialize, Serialize};

use crate::graph::ParamRef;

/// The four types of hole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoleKind {
    /// One diameter all the way down.
    Simple { diameter: ParamRef },
    /// A wider, shallower bore at the entry, for a cap screw's head.
    Counterbore { diameter: ParamRef, bore_diameter: ParamRef, bore_depth: ParamRef },
    /// A cone at the entry, for a countersunk head: its widest diameter and its included angle.
    Countersink { diameter: ParamRef, sink_diameter: ParamRef, sink_angle: ParamRef },
    /// Drilled to the thread's minor diameter and threaded to a depth of its own, from the metric
    /// coarse table, with a clearance opening the thread for fit.
    Tapped { designation: String, thread_depth: ParamRef, clearance: ParamRef },
}

impl Default for HoleKind {
    fn default() -> Self {
        HoleKind::Simple { diameter: ParamRef::Literal(6.0) }
    }
}

impl HoleKind {
    /// The word a file and the console use for the type.
    pub fn name(&self) -> &'static str {
        match self {
            HoleKind::Simple { .. } => "simple",
            HoleKind::Counterbore { .. } => "counterbore",
            HoleKind::Countersink { .. } => "countersink",
            HoleKind::Tapped { .. } => "tapped",
        }
    }

    /// The main diameter, where the type has one typed: a tapped hole's is the table's.
    pub fn diameter(&self) -> Option<&ParamRef> {
        match self {
            HoleKind::Simple { diameter }
            | HoleKind::Counterbore { diameter, .. }
            | HoleKind::Countersink { diameter, .. } => Some(diameter),
            HoleKind::Tapped { .. } => None,
        }
    }

    /// The same type under another name, keeping what carries over: the diameter between the
    /// three plain types, and beside it **the standard's head sizes where the diameter is a
    /// clearance hole the table knows** -- a 9 mm hole becoming a counterbore takes the M8 socket
    /// cap screw's 15 × 9 -- and proportions of it otherwise.
    pub fn with_name(&self, name: &str) -> Option<Self> {
        let d = self.diameter().copied().unwrap_or(ParamRef::Literal(6.6));
        let thread = match d {
            ParamRef::Literal(v) => thread_cleared_by(v).map(|t| t.to_string()),
            _ => None,
        };
        let scaled = |f: f64| match d {
            ParamRef::Literal(v) => ParamRef::Literal(v * f),
            _ => ParamRef::Literal(6.6 * f),
        };
        let lit = ParamRef::Literal;
        Some(match name {
            "simple" => HoleKind::Simple { diameter: d },
            "counterbore" => {
                let (bore, depth) = match thread.as_deref().and_then(counterbore_for) {
                    Some((b, t)) => (lit(b), lit(t)),
                    None => (scaled(1.8), scaled(1.0)),
                };
                HoleKind::Counterbore { diameter: d, bore_diameter: bore, bore_depth: depth }
            }
            "countersink" => HoleKind::Countersink {
                diameter: d,
                sink_diameter: thread.as_deref().and_then(countersink_for).map_or_else(|| scaled(2.0), lit),
                sink_angle: lit(90.0),
            },
            "tapped" => HoleKind::Tapped {
                designation: "M6".to_string(),
                thread_depth: ParamRef::Literal(12.0),
                clearance: ParamRef::Literal(0.0),
            },
            _ => return None,
        })
    }

    /// Walked rather than listed: a node's numbers are asked for on every edit, and the walk
    /// allocates nothing.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let three: [Option<&ParamRef>; 3] = match self {
            HoleKind::Simple { diameter } => [Some(diameter), None, None],
            HoleKind::Counterbore { diameter, bore_diameter, bore_depth } => {
                [Some(diameter), Some(bore_diameter), Some(bore_depth)]
            }
            HoleKind::Countersink { diameter, sink_diameter, sink_angle } => {
                [Some(diameter), Some(sink_diameter), Some(sink_angle)]
            }
            HoleKind::Tapped { thread_depth, clearance, .. } => {
                [Some(thread_depth), Some(clearance), None]
            }
        };
        three.into_iter().flatten()
    }

    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            HoleKind::Simple { diameter } => vec![diameter],
            HoleKind::Counterbore { diameter, bore_diameter, bore_depth } => {
                vec![diameter, bore_diameter, bore_depth]
            }
            HoleKind::Countersink { diameter, sink_diameter, sink_angle } => {
                vec![diameter, sink_diameter, sink_angle]
            }
            HoleKind::Tapped { thread_depth, clearance, .. } => vec![thread_depth, clearance],
        }
    }
}

/// How loose a clearance hole is: the three series of the metric clearance-hole standard
/// (ISO 273), H12, H13 and H14 when a tolerance is wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    Fine,
    Medium,
    Coarse,
}

impl Fit {
    pub fn name(self) -> &'static str {
        match self {
            Fit::Fine => "fine",
            Fit::Medium => "medium",
            Fit::Coarse => "coarse",
        }
    }

    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "fine" => Some(Fit::Fine),
            "medium" => Some(Fit::Medium),
            "coarse" => Some(Fit::Coarse),
            _ => None,
        }
    }
}

/// Clearance hole diameters for bolts and screws, by thread diameter, in the fine, medium and
/// coarse series: ISO 273:1979, every row from 1 mm to 150 mm, in millimetres. Data, kept in one
/// place beside the thread table, and read by the hole block's *Clearance for* and its console
/// key. The values result from bearing-area calculations against the ISO bolt and nut product
/// standards; a hole for a special application is chosen on its own design.
pub const CLEARANCE_TABLE: &[(f64, f64, f64, f64)] = &[
    (1.0, 1.1, 1.2, 1.3),
    (1.2, 1.3, 1.4, 1.5),
    (1.4, 1.5, 1.6, 1.8),
    (1.6, 1.7, 1.8, 2.0),
    (1.8, 2.0, 2.1, 2.2),
    (2.0, 2.2, 2.4, 2.6),
    (2.5, 2.7, 2.9, 3.1),
    (3.0, 3.2, 3.4, 3.6),
    (3.5, 3.7, 3.9, 4.2),
    (4.0, 4.3, 4.5, 4.8),
    (4.5, 4.8, 5.0, 5.3),
    (5.0, 5.3, 5.5, 5.8),
    (6.0, 6.4, 6.6, 7.0),
    (7.0, 7.4, 7.6, 8.0),
    (8.0, 8.4, 9.0, 10.0),
    (10.0, 10.5, 11.0, 12.0),
    (12.0, 13.0, 13.5, 14.5),
    (14.0, 15.0, 15.5, 16.5),
    (16.0, 17.0, 17.5, 18.5),
    (18.0, 19.0, 20.0, 21.0),
    (20.0, 21.0, 22.0, 24.0),
    (22.0, 23.0, 24.0, 26.0),
    (24.0, 25.0, 26.0, 28.0),
    (27.0, 28.0, 30.0, 32.0),
    (30.0, 31.0, 33.0, 35.0),
    (33.0, 34.0, 36.0, 38.0),
    (36.0, 37.0, 39.0, 42.0),
    (39.0, 40.0, 42.0, 45.0),
    (42.0, 43.0, 45.0, 48.0),
    (45.0, 46.0, 48.0, 52.0),
    (48.0, 50.0, 52.0, 56.0),
    (52.0, 54.0, 56.0, 62.0),
    (56.0, 58.0, 62.0, 66.0),
    (60.0, 62.0, 66.0, 70.0),
    (64.0, 66.0, 70.0, 74.0),
    (68.0, 70.0, 74.0, 78.0),
    (72.0, 74.0, 78.0, 82.0),
    (76.0, 78.0, 82.0, 86.0),
    (80.0, 82.0, 86.0, 91.0),
    (85.0, 87.0, 91.0, 96.0),
    (90.0, 93.0, 96.0, 101.0),
    (95.0, 98.0, 101.0, 107.0),
    (100.0, 104.0, 107.0, 112.0),
    (105.0, 109.0, 112.0, 117.0),
    (110.0, 114.0, 117.0, 122.0),
    (115.0, 119.0, 122.0, 127.0),
    (120.0, 124.0, 127.0, 132.0),
    (125.0, 129.0, 132.0, 137.0),
    (130.0, 134.0, 137.0, 144.0),
    (140.0, 144.0, 147.0, 155.0),
    (150.0, 155.0, 158.0, 165.0),
];

/// The clearance hole for a thread, by name (`"M8"`, case aside) or by its diameter in
/// millimetres (`"8"`), in a series; `None` for a size the table does not have.
pub fn clearance_hole(thread: &str, fit: Fit) -> Option<f64> {
    let d = thread_size(thread)?;
    CLEARANCE_TABLE.iter().find(|row| (row.0 - d).abs() < 1e-9).map(|row| match fit {
        Fit::Fine => row.1,
        Fit::Medium => row.2,
        Fit::Coarse => row.3,
    })
}

/// A thread's diameter from `"M8"` or `"8"`, case aside.
pub fn thread_size(thread: &str) -> Option<f64> {
    let t = thread.trim();
    let number = t.strip_prefix('M').or_else(|| t.strip_prefix('m')).unwrap_or(t);
    number.replace(',', ".").parse().ok()
}

/// Counterbores for hexagon socket head cap screws (ISO 4762), by thread diameter: the
/// counterbore's diameter and its depth, in millimetres -- the standard's *Type K*, the head
/// sitting flush with no washer under it. From IS 3406 (Part 2):1986, *Dimensions for
/// countersinks and counterbores, Part 2: Counterbores*, with its Amendment 1 (the M5 depth read
/// 5.7). Its clearance holes are ISO 273's, so a counterbore here is completed by the clearance
/// table's medium or fine hole for the same thread.
pub const COUNTERBORE_TABLE: &[(f64, f64, f64)] = &[
    (1.4, 2.8, 1.6),
    (1.6, 3.3, 1.8),
    (2.0, 4.3, 2.3),
    (2.5, 5.0, 2.9),
    (3.0, 6.0, 3.4),
    (4.0, 8.0, 4.6),
    (5.0, 10.0, 5.7),
    (6.0, 11.0, 6.8),
    (8.0, 15.0, 9.0),
    (10.0, 18.0, 11.0),
    (12.0, 20.0, 13.0),
    (14.0, 24.0, 15.0),
    (16.0, 26.0, 17.5),
    (18.0, 30.0, 19.5),
    (20.0, 33.0, 21.5),
    (22.0, 36.0, 23.5),
    (24.0, 40.0, 25.5),
    (27.0, 43.0, 28.5),
    (30.0, 48.0, 32.0),
    (33.0, 53.0, 35.0),
    (36.0, 57.0, 38.0),
];

/// Countersinks for countersunk head screws, 90° included, by thread diameter: the countersink's
/// diameter at the surface, in millimetres, for the medium-series hole. The standard's *Type A*,
/// for slotted and cross-recessed countersunk screws (the ISO 2009 and ISO 7046 family), from
/// IS 3406 (Part 1):1986, *Dimensions for countersinks and counterbores, Part 1: Countersinks*.
pub const COUNTERSINK_TABLE: &[(f64, f64)] = &[
    (1.0, 2.4),
    (1.2, 2.8),
    (1.4, 3.3),
    (1.6, 3.7),
    (1.8, 4.1),
    (2.0, 4.6),
    (2.5, 5.7),
    (3.0, 6.5),
    (3.5, 7.6),
    (4.0, 8.6),
    (4.5, 9.5),
    (5.0, 10.4),
    (6.0, 12.4),
    (8.0, 16.4),
    (10.0, 20.4),
    (12.0, 23.9),
    (14.0, 26.9),
    (16.0, 31.9),
    (18.0, 36.4),
    (20.0, 40.4),
];

/// The counterbore's diameter and depth for a socket head cap screw of this thread, if the table
/// has it.
pub fn counterbore_for(thread: &str) -> Option<(f64, f64)> {
    let d = thread_size(thread)?;
    COUNTERBORE_TABLE.iter().find(|r| (r.0 - d).abs() < 1e-9).map(|r| (r.1, r.2))
}

/// The 90° countersink's diameter for a countersunk screw of this thread, if the table has it.
pub fn countersink_for(thread: &str) -> Option<f64> {
    let d = thread_size(thread)?;
    COUNTERSINK_TABLE.iter().find(|r| (r.0 - d).abs() < 1e-9).map(|r| r.1)
}

/// The thread a clearance hole of `diameter` is for, in any series, if one is: what lets a type
/// switch find the standard's head sizes for a hole already sized from the table.
pub fn thread_cleared_by(diameter: f64) -> Option<f64> {
    CLEARANCE_TABLE
        .iter()
        .find(|r| [r.1, r.2, r.3].iter().any(|c| (c - diameter).abs() < 1e-9))
        .map(|r| r.0)
}

/// Every thread and series whose clearance hole is `diameter`, smallest thread first: what the
/// *Clearance for* picker shows as picked. More than one can be -- 21 mm is an M18's coarse hole
/// and an M20's fine one -- so the reader that knows more, such as a counterbore's head, narrows it.
pub fn clearances_of(diameter: f64) -> Vec<(f64, Fit)> {
    CLEARANCE_TABLE
        .iter()
        .flat_map(|r| [(r.0, Fit::Fine, r.1), (r.0, Fit::Medium, r.2), (r.0, Fit::Coarse, r.3)])
        .filter(|(_, _, c)| (c - diameter).abs() < 1e-9)
        .map(|(d, fit, _)| (d, fit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table answers by name or by number, in each series, and only for the sizes it has;
    /// every row widens from fine to coarse, and the rows climb with the thread.
    #[test]
    fn the_clearance_table_answers_each_series_and_only_its_own_sizes() {
        assert_eq!(clearance_hole("M8", Fit::Fine), Some(8.4));
        assert_eq!(clearance_hole("m8", Fit::Medium), Some(9.0));
        assert_eq!(clearance_hole("8", Fit::Coarse), Some(10.0));
        assert_eq!(clearance_hole("M6", Fit::Medium), Some(6.6));
        assert_eq!(clearance_hole("M2.5", Fit::Fine), Some(2.7));
        assert_eq!(clearance_hole("M90", Fit::Coarse), Some(101.0));
        assert_eq!(clearance_hole("M9", Fit::Fine), None);
        assert_eq!(clearance_hole("bolt", Fit::Fine), None);
        assert_eq!(CLEARANCE_TABLE.len(), 51);
        for pair in CLEARANCE_TABLE.windows(2) {
            assert!(pair[1].0 > pair[0].0);
        }
        for row in CLEARANCE_TABLE {
            assert!(row.0 < row.1 && row.1 <= row.2 && row.2 <= row.3, "{row:?}");
        }
        for fit in [Fit::Fine, Fit::Medium, Fit::Coarse] {
            assert_eq!(Fit::parse(fit.name()), Some(fit));
        }
    }

    /// A diameter names back every row and series that gives it, and only those: every entry of
    /// the table finds itself, a shared value finds both, and a size off the table finds none.
    #[test]
    fn a_clearance_diameter_names_every_bolt_it_is_the_hole_for() {
        assert_eq!(clearances_of(9.0), vec![(8.0, Fit::Medium)]);
        assert_eq!(clearances_of(21.0), vec![(18.0, Fit::Coarse), (20.0, Fit::Fine)]);
        assert!(clearances_of(9.1).is_empty());
        for row in CLEARANCE_TABLE {
            for (fit, c) in [(Fit::Fine, row.1), (Fit::Medium, row.2), (Fit::Coarse, row.3)] {
                assert!(clearances_of(c).contains(&(row.0, fit)), "{row:?} {fit:?}");
                assert_eq!(clearance_hole(&row.0.to_string(), fit), Some(c));
            }
        }
    }

    /// The head tables answer by thread, every head wider than the medium clearance hole for its
    /// thread and the counterbores deeper as they widen; and a hole sized from the clearance table
    /// takes the standard's head when its type is switched.
    #[test]
    fn the_head_tables_answer_by_thread_and_a_type_switch_takes_them() {
        assert_eq!(counterbore_for("M8"), Some((15.0, 9.0)));
        assert_eq!(counterbore_for("M5"), Some((10.0, 5.7)), "the amended depth");
        assert_eq!(countersink_for("M6"), Some(12.4));
        assert_eq!(counterbore_for("M7"), None);
        for &(d, bore, depth) in COUNTERBORE_TABLE {
            let hole = clearance_hole(&d.to_string(), Fit::Medium).expect("the clearance table has it");
            assert!(bore > hole && depth > 0.0, "M{d}");
        }
        for pair in COUNTERBORE_TABLE.windows(2) {
            assert!(pair[1].1 > pair[0].1 && pair[1].2 > pair[0].2);
        }
        for &(d, sink) in COUNTERSINK_TABLE {
            assert!(sink > clearance_hole(&d.to_string(), Fit::Medium).expect("cleared"), "M{d}");
        }
        let lit = ParamRef::Literal;
        let m8 = HoleKind::Simple { diameter: lit(9.0) };
        assert_eq!(
            m8.with_name("counterbore"),
            Some(HoleKind::Counterbore { diameter: lit(9.0), bore_diameter: lit(15.0), bore_depth: lit(9.0) })
        );
        assert_eq!(
            m8.with_name("countersink"),
            Some(HoleKind::Countersink { diameter: lit(9.0), sink_diameter: lit(16.4), sink_angle: lit(90.0) })
        );
        // Any series counts: 7 is M6's coarse hole, so it takes M6's head.
        let coarse = HoleKind::Simple { diameter: lit(7.0) };
        assert!(matches!(coarse.with_name("counterbore"), Some(HoleKind::Counterbore { bore_diameter: ParamRef::Literal(b), .. }) if b == 11.0));
        // A diameter the table does not know keeps the proportions.
        let odd = HoleKind::Simple { diameter: lit(7.1) };
        assert!(matches!(odd.with_name("counterbore"), Some(HoleKind::Counterbore { bore_diameter: ParamRef::Literal(b), .. }) if (b - 7.1 * 1.8).abs() < 1e-9));
    }
}

/// How deep a hole goes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoleDepth {
    /// This far along the axis from the entry to where the full diameter ends -- the shoulder,
    /// before any drill point.
    Blind(ParamRef),
    /// All the way. A hole only ever cuts the body it was made in, so it runs on along its axis
    /// without end: it cannot fall short when that body grows, and it reads no extent to say so.
    Through,
}

impl Default for HoleDepth {
    fn default() -> Self {
        HoleDepth::Blind(ParamRef::Literal(12.0))
    }
}

/// How a blind hole ends. Ignored by a through hole, which has no end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HoleTip {
    #[default]
    Flat,
    /// A drill's point, of this included angle in degrees (118° for a twist drill).
    Drill(ParamRef),
}
