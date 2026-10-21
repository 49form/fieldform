// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! `expect`: a measurement asserted, with an error estimate and three verdicts.
//!
//! `benchmark-suite-instructions.md` §2 A8 and `benchmark-suite-spec.md` §3–§4. The point of the
//! verb is that a script can say what a part must measure and the application can say whether it
//! does, so a scripted build checks itself instead of being read by eye.
//!
//! **A measurement is never trusted on its own.** Each quantity is measured on the grid anchored
//! at the world origin (`MassProperties::anchored`) at spacing `h` and again at `h/2`; the value
//! reported is the finer one, and the error estimate is the difference between the two. That is a
//! sound bound whenever the method converges at order one or better and is in its asymptotic
//! range (§3.4). With `w` the wanted value and `tol` the tolerance:
//!
//! - **pass** when `|Q − w| + e ≤ tol` -- right even at the worst the estimate allows;
//! - **fail** when `|Q − w| − e > tol` -- wrong even at the best;
//! - otherwise halve `h` and measure again, and at the finest grid the machine holds the verdict
//!   is **unresolved**: printed apart from a fail, counted as not passed, and the remedy is a
//!   better measurement, never a looser tolerance.
//!
//! A difference or a ratio of two blocks is taken at each level before the error is, which is how
//! the cancellation the anchored grid buys carries through to a fillet measured against its sharp
//! twin (§3.2).
//!
//! **`thickness_min=v`** is the local thickness of the thinnest place, measured on the same
//! anchored mesh and refined the same way, and it passes when the thinnest place is at or above
//! `v` even at the worst the error estimate allows. It is a bound, so it takes no tolerance.
//!
//! **It is measured over the whole surface, and a sharp edge is part of the surface.** The
//! figure is twice the radius of the largest ball inside the solid that contains the point, so
//! at an edge it is small however thick the walls are, and at a corner smaller still. Measured
//! on the smoke test's 2 mm wall, which really is 2 mm thick everywhere between its faces: the
//! answer was 0.75 mm, from the corners, and it moved by 0.75 between two grids because what a
//! mesh puts at a corner depends on the grid. That is the number the `thickness` command reports
//! and what it is for -- finding where a solid tapers to nothing (backlog item 27) -- so this is
//! the same measurement, not a second one, and a benchmark asserting a wall's own thickness is
//! asking the wrong question of it. Assert it on a body whose thinnest place is the one meant,
//! and say in the benchmark's README where that place is, which is what the spec's §3.7 asks
//! for.
//!
//! **The first grid is chosen against the smallest feature the target carries** (§3.3): a
//! lattice's own feature size, a wall or strut thickness, a shell, and a blend's radius. A sheet
//! thinner than about two cells can vanish from a marching mesh at *every* grid coarse enough to
//! miss it, so two coarse levels would agree on the wrong answer and the error estimate would
//! read zero. The floor comes first, before the body's size.
//!
//! **`equal=` is not built, and will not be**: `expect ratio=a,b volume=1 tol=0.5%` says that two
//! bodies measure the same, in the vocabulary that is already here. A third spelling of one idea
//! is what D9 is about, so the key is refused by name with that line as the remedy.
//!
//! The rules are free of the `App` so a test can hold them: an `App` needs a window and a device.

use crate::console_keys::{block_called, Lookup};
use fieldform_field::Measured;
use fieldform_ops::{Args, Value};
use fieldform_scene::{Graph, NodeId};

/// What is measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Quantity {
    Volume,
    Area,
    /// The six bounds, low corner then high.
    Bounds,
    /// The local thickness of the thinnest place: twice the radius of the largest ball inside the
    /// solid that contains the point (§3.7).
    ThicknessMin,
}

impl Quantity {
    pub fn word(self) -> &'static str {
        match self {
            Quantity::Volume => "volume",
            Quantity::Area => "area",
            Quantity::Bounds => "bbox",
            Quantity::ThicknessMin => "thickness_min",
        }
    }

    /// Whether this quantity is read off the solid's thickness rather than its mass properties,
    /// so the caller knows which measurement to take.
    pub fn is_thickness(self) -> bool {
        matches!(self, Quantity::ThicknessMin)
    }

    /// The numbers this quantity reads off one measurement.
    pub fn read(self, m: &Measured) -> Vec<f64> {
        match self {
            Quantity::Volume => vec![m.props.volume_mm3],
            Quantity::Area => vec![m.props.surface_area_mm2],
            Quantity::Bounds => vec![m.lo.x, m.lo.y, m.lo.z, m.hi.x, m.hi.y, m.hi.z],
            // Carried on the same reading, so one grid answers either.
            Quantity::ThicknessMin => vec![m.thinnest_mm],
        }
    }
}

/// Of which blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Of {
    One(NodeId),
    /// The first less the second.
    Diff(NodeId, NodeId),
    /// The first over the second.
    Ratio(NodeId, NodeId),
}

/// What the value must be.
#[derive(Clone, Debug, PartialEq)]
pub enum Wanted {
    /// Each number within `tol` of its wanted value.
    Near { values: Vec<f64>, tol: f64 },
    /// Above `v`, strictly or not (`volume=>v`, `volume=>=v`).
    Above { v: f64, strict: bool },
}

/// One `expect` line, read.
#[derive(Clone, Debug, PartialEq)]
pub enum Expect {
    Measure { of: Of, quantity: Quantity, wanted: Wanted },
    /// `approx=true|false`: the blend's own report, no measurement and no tolerance (§4.1).
    Approx { target: NodeId, wanted: bool },
}

/// The verdicts of §3.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Unresolved,
}

impl Verdict {
    pub fn word(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::Unresolved => "unresolved",
        }
    }
}

fn number(word: &str, text: &str) -> Result<f64, String> {
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("`expect {word}=` wants a number, not {text:?}"))
}

fn block(word: &str, name: &str, lookup: &dyn Fn(&str) -> Lookup) -> Result<NodeId, String> {
    // The trim goes on the lookup rather than the name, because `target=` hands the value over
    // as typed and a refusal has always quoted it that way.
    block_called("expect", word, name, &|n: &str| lookup(n.trim()))
}

fn pair(word: &str, value: &Value, lookup: &dyn Fn(&str) -> Lookup) -> Result<(NodeId, NodeId), String> {
    let text = value.text();
    let names: Vec<&str> = text.split(',').map(str::trim).collect();
    let [a, b] = names[..] else {
        return Err(format!("`expect {word}=` names two blocks, like {word}=mark,sharp — not {text:?}"));
    };
    Ok((block(word, a, lookup)?, block(word, b, lookup)?))
}

/// One `expect` line's keys, read and checked.
pub fn read(args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Expect, String> {
    if args.get("equal").is_some() {
        return Err(
            "`expect equal=` is `ratio=a,b` with the quantity at 1 — `expect ratio=a,b volume=1 tol=0.5%`              says that two bodies measure the same, and one idea does not get two spellings"
                .to_string(),
        );
    }
    let target = args.get("target").map(|v| block("target", &v.text(), lookup)).transpose()?;
    let diff = args.get("diff").map(|v| pair("diff", v, lookup)).transpose()?;
    let ratio = args.get("ratio").map(|v| pair("ratio", v, lookup)).transpose()?;
    let of = match (target, diff, ratio) {
        (Some(t), None, None) => Of::One(t),
        (None, Some((a, b)), None) => Of::Diff(a, b),
        (None, None, Some((a, b))) => Of::Ratio(a, b),
        (None, None, None) => return Err("`expect` names what it measures: target=, diff=a,b or ratio=a,b".to_string()),
        _ => return Err("`expect` measures one thing at a time: target=, diff= or ratio=, not two of them".to_string()),
    };

    if let Some(v) = args.get("approx") {
        let Of::One(target) = of else {
            return Err("`expect approx=` reads one block's own report — say target=".to_string());
        };
        let wanted = match v {
            Value::Bool(b) => *b,
            other => return Err(format!("`expect approx=` is true or false, not {}", other.text())),
        };
        if ["volume", "area", "bbox", "tol"].iter().any(|k| args.get(k).is_some()) {
            return Err("`expect approx=` is not a measurement and takes no tolerance — one check a line".to_string());
        }
        return Ok(Expect::Approx { target, wanted });
    }

    let asked: Vec<(Quantity, &Value)> =
        [Quantity::Volume, Quantity::Area, Quantity::Bounds, Quantity::ThicknessMin]
            .into_iter()
            .filter_map(|q| args.get(q.word()).map(|v| (q, v)))
            .collect();
    let (quantity, value) = match asked[..] {
        [one] => one,
        [] => return Err("`expect` says what it measures: volume=, area=, bbox= or approx=".to_string()),
        _ => return Err("`expect` checks one quantity a line — write a line for each".to_string()),
    };
    if quantity == Quantity::Bounds && !matches!(of, Of::One(_)) {
        return Err("`expect bbox=` is one block's bounds — a difference or a ratio of bounds means nothing".to_string());
    }
    if quantity == Quantity::ThicknessMin {
        if !matches!(of, Of::One(_)) {
            return Err(
                "`expect thickness_min=` is one block's thinnest place — a difference or a ratio of two is not a thickness"
                    .to_string(),
            );
        }
        if args.get("tol").is_some() {
            return Err("`expect thickness_min=` is a bound and takes no tolerance".to_string());
        }
        // A bound, read the way `volume=>=v` is: it passes when the thinnest place is at or above
        // the number even at the worst the estimate allows.
        let v = number("thickness_min", &value.text())?;
        if v <= 0.0 {
            return Err("`expect thickness_min=` is a thickness, so above nought".to_string());
        }
        return Ok(Expect::Measure { of, quantity, wanted: Wanted::Above { v, strict: false } });
    }
    let text = value.text();
    let word = quantity.word();

    // `volume=>v` and `volume=>=v`: a lower bound, with no tolerance (§4.1).
    if let Some(rest) = text.strip_prefix('>') {
        let (strict, rest) = match rest.strip_prefix('=') {
            Some(r) => (false, r),
            None => (true, rest),
        };
        if quantity == Quantity::Bounds {
            return Err("`expect bbox=` is six numbers, not a bound".to_string());
        }
        if args.get("tol").is_some() {
            return Err(format!("`expect {word}=>…` is a bound and takes no tolerance"));
        }
        return Ok(Expect::Measure { of, quantity, wanted: Wanted::Above { v: number(word, rest)?, strict } });
    }

    let values: Vec<f64> = text.split(',').map(|t| number(word, t)).collect::<Result<_, _>>()?;
    let count = if quantity == Quantity::Bounds { 6 } else { 1 };
    if values.len() != count {
        return Err(match quantity {
            Quantity::Bounds => format!("`expect bbox=` is six numbers, xmin,ymin,zmin,xmax,ymax,zmax — not {text:?}"),
            _ => format!("`expect {word}=` is one number, not {text:?}"),
        });
    }
    let Some(tol) = args.get("tol") else {
        return Err(format!("`expect {word}=` wants a tolerance — tol=0.1% or tol=0.02 (mm)"));
    };
    let tol_text = tol.text();
    let tol = match tol_text.trim().strip_suffix('%') {
        Some(percent) => {
            if quantity == Quantity::Bounds {
                return Err("`expect bbox=` takes its tolerance in millimetres, not a percentage".to_string());
            }
            number("tol", percent)? / 100.0 * values[0].abs()
        }
        None => number("tol", &tol_text)?,
    };
    if tol < 0.0 {
        return Err("`expect tol=` cannot be negative".to_string());
    }
    Ok(Expect::Measure { of, quantity, wanted: Wanted::Near { values, tol } })
}

/// The verdict on one level's numbers `q` with error estimates `e`, or `None` to refine.
pub fn judge(q: &[f64], e: &[f64], wanted: &Wanted) -> Option<Verdict> {
    match wanted {
        Wanted::Near { values, tol } => {
            let off: Vec<f64> = q.iter().zip(values).map(|(q, w)| (q - w).abs()).collect();
            if off.iter().zip(e).any(|(off, e)| off - e > *tol) {
                Some(Verdict::Fail)
            } else if off.iter().zip(e).all(|(off, e)| off + e <= *tol) {
                Some(Verdict::Pass)
            } else {
                None
            }
        }
        Wanted::Above { v, strict } => {
            let (q, e) = (q[0], e[0]);
            let passes = if *strict { q - e > *v } else { q - e >= *v };
            let fails = if *strict { q + e <= *v } else { q + e < *v };
            if passes {
                Some(Verdict::Pass)
            } else if fails {
                Some(Verdict::Fail)
            } else {
                None
            }
        }
    }
}

/// What a measurement came to.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub verdict: Verdict,
    /// The finer level's numbers, and their error estimates.
    pub got: Vec<f64>,
    pub error: Vec<f64>,
    /// The finer grid spacing they were measured at.
    pub h: f64,
    /// Why it stopped refining, for an unresolved one.
    pub why: Option<String>,
}

/// The coarsest grid a block is first measured at: a power of two, no more than a millimetre, no
/// more than a thirty-second of the body's longest side, and **no more than a quarter of the
/// smallest feature the block carries** (§3.3).
///
/// The feature floor comes first in the reasoning even though it is the same `min`: a sheet
/// thinner than about two cells can be missed by a marching mesh at every grid coarse enough to
/// miss it, so the two coarse levels agree on the wrong answer and the error estimate reads zero.
/// Size alone cannot see that; only the block's own numbers can.
pub fn first_spacing_with_feature(longest_mm: f64, feature_mm: Option<f64>) -> f64 {
    let by_feature = feature_mm.filter(|f| *f > 0.0).map_or(f64::INFINITY, |f| f / 4.0);
    let limit = (longest_mm / 32.0).min(1.0).min(by_feature).max(H_MIN);
    2f64.powf(limit.log2().floor())
}

/// The same for a block carrying no feature of its own. Read by the tests, which is where the
/// two halves of the rule are told apart.
#[cfg(test)]
pub fn first_spacing(longest_mm: f64) -> f64 {
    first_spacing_with_feature(longest_mm, None)
}

/// The finest grid any expect measures at (§3.3's `h_min`).
pub const H_MIN: f64 = 1.0 / 64.0;

/// Measures `quantity` of `of` at successive grids from `h0` until the verdict is decided.
///
/// `measure(block, h)` is one anchored measurement of one block, or why there is none at that
/// spacing (a grid too fine to hold, or nothing there). A difference or ratio is formed at each
/// level before the error is, so the anchored grid's cancellation carries through.
pub fn decide(
    of: Of,
    quantity: Quantity,
    wanted: &Wanted,
    h0: f64,
    measure: &mut dyn FnMut(NodeId, f64) -> Result<Measured, String>,
) -> Outcome {
    let level = |h: f64, measure: &mut dyn FnMut(NodeId, f64) -> Result<Measured, String>| -> Result<Vec<f64>, String> {
        let at = |id: NodeId, measure: &mut dyn FnMut(NodeId, f64) -> Result<Measured, String>| measure(id, h).map(|m| quantity.read(&m));
        Ok(match of {
            Of::One(id) => at(id, measure)?,
            Of::Diff(a, b) => vec![at(a, measure)?[0] - at(b, measure)?[0]],
            Of::Ratio(a, b) => {
                let below = at(b, measure)?[0];
                if below.abs() < 1e-12 {
                    return Err("the second block measures nothing, so there is no ratio".to_string());
                }
                vec![at(a, measure)?[0] / below]
            }
        })
    };
    let mut h = h0;
    let mut coarse = match level(h, measure) {
        Ok(q) => q,
        Err(why) => {
            return Outcome { verdict: Verdict::Unresolved, got: Vec::new(), error: Vec::new(), h, why: Some(why) };
        }
    };
    loop {
        let fine_h = h / 2.0;
        let fine = if fine_h < H_MIN {
            Err(format!("the grid is at its finest, {} mm", h))
        } else {
            level(fine_h, measure)
        };
        let fine = match fine {
            Ok(fine) => fine,
            // Out of reach: the last level decided nothing, so the verdict is that measurement
            // is the limit. Its numbers are reported with no error estimate.
            Err(why) => {
                return Outcome { verdict: Verdict::Unresolved, error: vec![f64::NAN; coarse.len()], got: coarse, h, why: Some(why) };
            }
        };
        let error: Vec<f64> = fine.iter().zip(&coarse).map(|(f, c)| (f - c).abs()).collect();
        if let Some(verdict) = judge(&fine, &error, wanted) {
            return Outcome { verdict, got: fine, error, h: fine_h, why: None };
        }
        coarse = fine;
        h = fine_h;
    }
}

/// The smallest feature the blocks under `target` ask for, in millimetres, or `None` where none
/// of them names a length that small matters at: a lattice sheet's thickness or a strut's radius,
/// a shell's wall, a wall given to an open curve, and a blend's radius (§3.3).
///
/// **Read off the typed numbers, not off a compiled field.** The question is which grid to start
/// at, asked before anything is compiled, and a number the user typed is the one a reader can
/// check against the block. A solid lattice is left out deliberately: its phase is a density, not
/// a length, and its features are bounded by the cell size rather than by a wall.
pub fn smallest_feature(doc: &fieldform_scene::Document, graph: &Graph, target: NodeId) -> Option<f64> {
    use fieldform_scene::{LatticePhase, Node, Outline};
    let alone = graph_of(graph, target);
    let number = |r: &fieldform_scene::ParamRef| r.resolve(&|id| doc.parameter_value(id));
    let mut smallest: Option<f64> = None;
    let mut note = |v: Option<f64>| {
        if let Some(v) = v.filter(|v| *v > 0.0 && v.is_finite()) {
            smallest = Some(smallest.map_or(v, |s: f64| s.min(v)));
        }
    };
    for (_, node) in alone.nodes() {
        match node {
            Node::Lattice { phase, .. } => match phase {
                LatticePhase::Sheet { thickness, .. } => note(number(thickness)),
                LatticePhase::Strut { radius } => note(number(radius)),
                LatticePhase::Solid { .. } => {}
            },
            Node::Shell { thickness, .. } => note(number(thickness)),
            Node::Blend { fillet, .. } => note(number(&fillet.radius)),
            Node::Pattern { fillet, join: true, .. } => note(number(&fillet.radius)),
            other => {
                if let Some(Outline::Wall { thickness, .. }) = other.outline() {
                    note(number(thickness));
                }
            }
        }
    }
    smallest
}

/// The graph that draws only `target` and what it reads, so a block that is an operand of
/// something else -- consumed, not a root -- can still be measured on its own.
///
/// **Not [`Graph::subtree`], which leaves the critical marks behind.** A mark tapers the fillets
/// near it, so it is part of the shape being measured. Removing everything else by id keeps the
/// marks on what stays, since [`Graph::remove`] takes only the removed block's own.
pub fn graph_of(graph: &Graph, target: NodeId) -> Graph {
    let keep = graph.subtree_of(target);
    let mut out = graph.clone();
    let gone: Vec<NodeId> = graph.nodes().map(|(id, _)| id).filter(|id| !keep.contains(id)).collect();
    for id in gone {
        out.remove(id);
    }
    out
}

/// How a verdict reads in the console: what, wanted, got, error, grid.
pub fn line(what: &str, quantity: Quantity, wanted: &Wanted, outcome: &Outcome) -> String {
    let nums = |v: &[f64]| v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",");
    let wanted_said = match wanted {
        Wanted::Near { values, tol } => format!("{} ± {tol:.3}", nums(values)),
        Wanted::Above { v, strict } => format!("{} {v:.3}", if *strict { ">" } else { "≥" }),
    };
    let got = if outcome.got.is_empty() { "nothing".to_string() } else { nums(&outcome.got) };
    let error = if outcome.error.iter().any(|e| e.is_nan()) || outcome.error.is_empty() {
        "no estimate".to_string()
    } else {
        format!("± {}", nums(&outcome.error))
    };
    let tail = outcome.why.as_ref().map(|w| format!(" — {w}")).unwrap_or_default();
    format!(
        "expect {what} {}: {} — wanted {wanted_said}, got {got} ({error}) at h = {} mm{tail}",
        quantity.word(),
        outcome.verdict.word().to_uppercase(),
        outcome.h
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_field::MassProperties;
    use nalgebra::Point3;

    fn args(pairs: &[(&str, Value)]) -> Args {
        pairs.iter().fold(Args::new(), |a, (k, v)| a.with(k, v.clone()))
    }

    fn s(t: &str) -> Value {
        Value::Str(t.to_string())
    }

    #[test]
    fn an_expect_line_reads_what_it_measures_its_wanted_value_and_its_tolerance() {
        let (a, b) = (NodeId::new(), NodeId::new());
        let lookup = |n: &str| match n {
            "box" => Lookup::Block(a),
            "plate" => Lookup::Block(b),
            _ => Lookup::Nothing,
        };
        let e = read(&args(&[("target", s("box")), ("volume", Value::Int(64000)), ("tol", s("0.1%"))]), &lookup).unwrap();
        assert_eq!(e, Expect::Measure { of: Of::One(a), quantity: Quantity::Volume, wanted: Wanted::Near { values: vec![64000.0], tol: 64.0 } });
        let e = read(&args(&[("target", s("box")), ("bbox", s("-20,-20,-20,20,20,20")), ("tol", Value::Float(0.02))]), &lookup).unwrap();
        assert!(matches!(e, Expect::Measure { quantity: Quantity::Bounds, wanted: Wanted::Near { ref values, tol }, .. } if values.len() == 6 && tol == 0.02));
        let e = read(&args(&[("ratio", s("box,plate")), ("volume", Value::Float(0.51)), ("tol", Value::Float(0.04))]), &lookup).unwrap();
        assert!(matches!(e, Expect::Measure { of: Of::Ratio(x, y), .. } if x == a && y == b));
        let e = read(&args(&[("target", s("box")), ("volume", s(">=1000"))]), &lookup).unwrap();
        assert!(matches!(e, Expect::Measure { wanted: Wanted::Above { v, strict: false }, .. } if v == 1000.0));
        let e = read(&args(&[("target", s("box")), ("approx", Value::Bool(false))]), &lookup).unwrap();
        assert_eq!(e, Expect::Approx { target: a, wanted: false });
        // Refused by name, every one.
        for (pairs, says) in [
            (vec![("target", s("box")), ("volume", Value::Int(1))], "wants a tolerance"),
            (vec![("target", s("box")), ("volume", Value::Int(1)), ("area", Value::Int(1)), ("tol", Value::Int(1))], "one quantity a line"),
            (vec![("target", s("nothing")), ("volume", Value::Int(1)), ("tol", Value::Int(1))], "no block is called"),
            (vec![("target", s("box")), ("bbox", s("1,2,3")), ("tol", Value::Int(1))], "six numbers"),
            (vec![("diff", s("box,plate")), ("bbox", s("1,2,3,4,5,6")), ("tol", Value::Int(1))], "means nothing"),
            (vec![("target", s("box")), ("thickness_min", Value::Float(0.9)), ("volume", Value::Int(1)), ("tol", Value::Int(1))], "one quantity a line"),
            (vec![("volume", Value::Int(1)), ("tol", Value::Int(1))], "names what it measures"),
        ] {
            let why = read(&args(&pairs), &lookup).unwrap_err();
            assert!(why.contains(says), "{pairs:?}: {why}");
        }
    }

    #[test]
    fn a_verdict_passes_only_at_the_worst_the_error_allows_and_fails_only_at_the_best() {
        let near = Wanted::Near { values: vec![100.0], tol: 1.0 };
        assert_eq!(judge(&[100.5], &[0.4], &near), Some(Verdict::Pass));
        assert_eq!(judge(&[100.5], &[0.6], &near), None, "could be 1.1 off: refine");
        assert_eq!(judge(&[102.0], &[0.5], &near), Some(Verdict::Fail));
        assert_eq!(judge(&[101.2], &[0.5], &near), None, "could be 0.7 off: refine");
        let above = Wanted::Above { v: 10.0, strict: true };
        assert_eq!(judge(&[10.5], &[0.4], &above), Some(Verdict::Pass));
        assert_eq!(judge(&[9.5], &[0.4], &above), Some(Verdict::Fail));
        assert_eq!(judge(&[10.2], &[0.4], &above), None);
    }

    /// Refinement on a real field: a sphere's volume converges, so a tight tolerance is decided
    /// at some grid, a wrong value fails, and one finer than the machine can hold is unresolved
    /// -- never a pass.
    #[test]
    fn refinement_decides_a_sphere_and_admits_when_it_cannot() {
        let ball = fieldform_field::Sphere { centre: Point3::new(0.0, 0.0, 0.0), radius: 10.0 };
        let id = NodeId::new();
        let mut measure = |_: NodeId, h: f64| match MassProperties::anchored(&ball, (Point3::new(-10.0, -10.0, -10.0), Point3::new(10.0, 10.0, 10.0)), h) {
            fieldform_field::Anchored::Measured(m) => Ok(m),
            other => Err(format!("{other:?}")),
        };
        let truth = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
        let ok = decide(Of::One(id), Quantity::Volume, &Wanted::Near { values: vec![truth], tol: truth * 0.01 }, 1.0, &mut measure);
        assert_eq!(ok.verdict, Verdict::Pass, "{ok:?}");
        let wrong = decide(Of::One(id), Quantity::Volume, &Wanted::Near { values: vec![truth * 1.1], tol: truth * 0.01 }, 1.0, &mut measure);
        assert_eq!(wrong.verdict, Verdict::Fail, "{wrong:?}");
        let unreachable = decide(Of::One(id), Quantity::Volume, &Wanted::Near { values: vec![truth], tol: 1e-9 }, 1.0, &mut measure);
        assert_eq!(unreachable.verdict, Verdict::Unresolved, "{unreachable:?}");
    }

    #[test]
    fn the_first_grid_is_a_power_of_two_scaled_to_the_body() {
        assert_eq!(first_spacing(40.0), 1.0);
        assert_eq!(first_spacing(20.0), 0.5);
        assert_eq!(first_spacing(3.0), 1.0 / 16.0);
        assert_eq!(first_spacing(1000.0), 1.0);
    }

    /// A thin feature pulls the first grid down to a quarter of itself, whatever the body's size
    /// says, so a sheet is not missed at both of the two levels an error estimate compares.
    #[test]
    fn a_thin_feature_sets_the_first_grid_rather_than_the_bodys_size() {
        // A 40 mm body alone starts at 1 mm; with a 1 mm wall in it, at a quarter of that.
        assert_eq!(first_spacing_with_feature(40.0, None), 1.0);
        assert_eq!(first_spacing_with_feature(40.0, Some(1.0)), 0.25);
        assert_eq!(first_spacing_with_feature(40.0, Some(0.5)), 0.125);
        // A feature bigger than the size's own limit changes nothing.
        assert_eq!(first_spacing_with_feature(40.0, Some(100.0)), 1.0);
        // And the floor still holds: nothing starts finer than the finest grid there is.
        assert_eq!(first_spacing_with_feature(40.0, Some(1e-6)), H_MIN);
        // Nonsense is ignored rather than taken as a floor of nought.
        assert_eq!(first_spacing_with_feature(40.0, Some(0.0)), 1.0);
        assert_eq!(first_spacing_with_feature(40.0, Some(f64::NAN)), 1.0);
    }

    #[test]
    fn a_thickness_is_a_bound_with_no_tolerance_and_one_block() {
        let (a, b) = (NodeId::new(), NodeId::new());
        let lookup = |n: &str| match n {
            "vent" => Lookup::Block(a),
            "other" => Lookup::Block(b),
            _ => Lookup::Nothing,
        };
        let e = read(&args(&[("target", s("vent")), ("thickness_min", Value::Float(0.9))]), &lookup).unwrap();
        assert_eq!(
            e,
            Expect::Measure {
                of: Of::One(a),
                quantity: Quantity::ThicknessMin,
                wanted: Wanted::Above { v: 0.9, strict: false },
            }
        );
        assert!(Quantity::ThicknessMin.is_thickness());
        assert!(!Quantity::Volume.is_thickness());
        for (pairs, says) in [
            (vec![("target", s("vent")), ("thickness_min", Value::Float(0.9)), ("tol", Value::Float(0.1))], "takes no tolerance"),
            (vec![("diff", s("vent,other")), ("thickness_min", Value::Float(0.9))], "is not a thickness"),
            (vec![("target", s("vent")), ("thickness_min", Value::Float(0.0))], "above nought"),
        ] {
            let why = read(&args(&pairs), &lookup).unwrap_err();
            assert!(why.contains(says), "{pairs:?}: {why}");
        }
        // It is judged as the bound it is: the thinnest place must clear the number even at the
        // worst the estimate allows.
        let bound = Wanted::Above { v: 0.9, strict: false };
        assert_eq!(judge(&[1.0], &[0.05], &bound), Some(Verdict::Pass));
        assert_eq!(judge(&[0.8], &[0.05], &bound), Some(Verdict::Fail));
        assert_eq!(judge(&[0.92], &[0.05], &bound), None, "could be under: refine");
    }

    /// `equal=` is `ratio=`, and the refusal says so rather than leaving the reader to find out.
    #[test]
    fn equal_is_refused_with_the_line_that_replaces_it() {
        let why = read(&args(&[("target", s("x")), ("equal", s("y"))]), &|_| Lookup::Nothing).unwrap_err();
        assert!(why.contains("ratio=a,b volume=1"), "{why}");
    }
}
