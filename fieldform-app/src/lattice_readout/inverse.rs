// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The inverse: the number a target stiffness asks of a lattice's phase, and what Set writes.

use super::*;

// ------------------------------------------------------------------------------- the inverse

/// One source's answer to a target: the number the phase would need, or why it cannot have it.
#[derive(Clone, Debug)]
pub(crate) struct Offer {
    pub row: &'static PropertyRow,
    pub number: Result<f64, String>,
    /// Where the density that number builds is past the source's own range, or the source
    /// states none: "needs a density of 0.45, outside its range, 0.001–0.10". **Said beside the
    /// number wherever it is offered** -- the console never writes, so a caution that waited for
    /// the readout after Set would never reach a script at all.
    pub caution: Option<String>,
}

/// What Set writes, decided here rather than in the panel so the rule is held by a test.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SetWrite {
    /// The lattice's own typed number, as the whole phase. The panel folds it into its edit of
    /// the block, so the block is replaced once: one undo step with whatever else changed.
    Phase(LatticePhase),
    /// One end of the ramp that drives the number: that ramp, replaced on its own, with only
    /// that end changed.
    Ramp(NodeId, Node),
}

/// What a target stiffness asks of one end of a lattice's phase (§7).
#[derive(Clone, Debug)]
pub(crate) struct TargetAnswer {
    pub end: End,
    /// "wall", "density" or "radius".
    pub word: &'static str,
    /// " mm", or nothing for a density.
    pub unit: &'static str,
    /// The default source's number, which is what Set writes.
    pub default: Offer,
    /// Every other source of the key, best-ranked first, each checked buildable.
    pub others: Vec<Offer>,
    /// What Set writes, or why it will not: never a named parameter, never a ramp that drives
    /// anything else.
    pub write: Result<SetWrite, String>,
}

/// The highest density the Density row takes; past it the inverse does not offer a number.
const SOLID_MOST: f64 = 0.99;

/// The number `end` of lattice `id`'s phase needs for `target` (a fraction: 0.05 is 5%) of the
/// solid's modulus, by the default source and by each other one.
///
/// **A one-shot, never a stored drive** (ruling 13): Set writes the phase's own literal once,
/// through the ordinary edit path. **Every number offered is checked buildable** -- above a
/// solid's floor, inside a wall limit or a merge radius -- before it is offered, and one that is
/// not says so by name. `Err` where nothing can be offered at all.
pub(crate) fn number_for_target(
    doc: &Document,
    graph: &Graph,
    id: NodeId,
    in_flight: Option<&Node>,
    end: End,
    target: f64,
) -> Result<TargetAnswer, String> {
    let Some(node @ Node::Lattice { cell, map, phase, .. }) = in_flight.or_else(|| graph.node(id))
    else {
        return Err("that block is not a lattice".to_string());
    };
    if !(target.is_finite() && target > 0.0) {
        return Err("a target is a percentage above nought".to_string());
    }
    let kind = CellKind::from(*cell);
    let banding = phase.banding();
    let what = what_it_is(*cell, banding);
    let (word, unit) = number_word(banding);
    // No wildcard, as in the readout.
    let cell_size = match map {
        CellMap::Rectangular { cell } => cell,
        CellMap::Cylindrical { .. } | CellMap::Spherical { .. } | CellMap::Conformal { .. } => {
            return Err("only a rectangular map's cells are cubes, which is all any source measured".to_string())
        }
    };
    let cell_mm = resolve(doc, cell_size)
        .filter(|c| *c > 0.0)
        .ok_or_else(|| "the cell size does not resolve".to_string())?;
    // The phase as it would be with a number at this end: what is offered has to be a phase the
    // tables read, whatever the number is now.
    let numbers = numbers_at_ends(doc, graph, phase)?;
    let now = numbers
        .iter()
        .find(|(e, _)| *e == end)
        .map(|(_, n)| *n)
        .ok_or_else(|| match end {
            End::Typed => format!("the {word} is driven by a ramp: set each Out end instead"),
            End::OutMin | End::OutMax => format!("the {word} is one number, not a ramp's range"),
        })?;
    let at_now = phase_of(phase, |_| Some(now), |r| resolve(doc, r))
        .map_err(|Unresolved { which }| format!("the {which} does not resolve"))?;
    match (kind, at_now) {
        (CellKind::Stochastic { .. }, _) => {
            return Err("a Voronoi cell's fill depends on its jitter and seed and is not tabled".to_string())
        }
        (CellKind::Tpms(_), Phase::Sheet { offset_mm, .. }) if offset_mm != 0.0 => {
            return Err("an offset wall divides the cell unequally; the sources measured centred sheets".to_string())
        }
        (_, Phase::Solid { density }) if density >= 1.0 => {
            return Err("this end is fully solid — not a lattice, so there is nothing to target".to_string())
        }
        _ => {}
    }
    let ranked = ranked_rows(kind, banding, PropertyClass::Modulus);
    let Some((first, rest)) = ranked.split_first() else {
        return Err(format!("no published fit for {what}"));
    };
    let offer = |row: &'static PropertyRow| {
        let number = offered_number(row, target, kind, at_now, cell_mm, &what);
        // The density the row was inverted at, read back for its standing: what the number
        // offered builds, to the bisection's precision, and past or without the row's range
        // exactly where a reading there would warn.
        let caution = match (&number, row.density_for(target)) {
            (Ok(_), Some(rho)) => standing_brief(row.read(rho).standing)
                .map(|brief| format!("needs a density of {}, {brief}", fmt_rho(rho))),
            _ => None,
        };
        Offer { row, number, caution }
    };
    let default = offer(first);
    let write = match &default.number {
        Ok(number) => write_for(doc, graph, id, node, end, *number, word, unit),
        Err(_) => Err(format!("{} cannot reach it", default.row.source.short)),
    };
    Ok(TargetAnswer {
        end,
        word,
        unit,
        default,
        others: rest.iter().map(|row| offer(row)).collect(),
        write,
    })
}

/// The number one row needs for `target`, checked buildable.
fn offered_number(
    row: &'static PropertyRow,
    target: f64,
    kind: CellKind,
    phase: Phase,
    cell_mm: f64,
    what: &str,
) -> Result<f64, String> {
    let Some(rho) = row.density_for(target) else {
        return Err(format!(
            "its fit does not reach {}% at any density it can be read at",
            pct(target)
        ));
    };
    number_for_density(kind, phase, rho, cell_mm, what).map_err(|limit| {
        let reads = match row.read(limit.at).value {
            Some(v) => format!("≈ {}% by this source", pct(v)),
            None => "no meaningful value by this source".to_string(),
        };
        format!(
            "not buildable — it needs a density of {}, {}; the {} it gets reads {reads}",
            fmt_rho(rho),
            limit.sentence,
            limit.extreme
        )
    })
}

/// Why a density cannot be built, and the nearest one that can.
struct Limit {
    sentence: String,
    /// The density at the limit, to say what the source reads there.
    at: f64,
    /// "stiffest", or "least that holds together".
    extreme: &'static str,
}

/// The phase's own number that builds `rho`, or the limit it passes.
///
/// Every map a density goes through rises (`the_fill_rises_with_the_wall_with_the_radius_and_
/// with_the_density`), so bisection is sound. The answer is rounded to a ten-thousandth -- a
/// number written into a box -- and re-checked buildable after rounding, since a density just
/// above a solid's floor can round below it.
fn number_for_density(
    kind: CellKind,
    phase: Phase,
    rho: f64,
    cell_mm: f64,
    what: &str,
) -> Result<f64, Limit> {
    let builds = |number: f64| -> bool {
        let at = match phase {
            Phase::Sheet { offset_mm, .. } => Phase::Sheet { thickness_mm: number, offset_mm },
            Phase::Solid { .. } => Phase::Solid { density: number },
            Phase::Strut { radius_mm: _ } => Phase::Strut { radius_mm: number },
        };
        uniform_density(kind, at, cell_mm).value().is_some()
    };
    let settle = |exact: f64, limit: Limit| -> Result<f64, Limit> {
        let rounded = (exact * 1e4).round() / 1e4;
        [rounded, (exact * 1e4).ceil() / 1e4, (exact * 1e4).floor() / 1e4]
            .into_iter()
            .find(|n| *n > 0.0 && builds(*n))
            .ok_or(limit)
    };
    match (kind, phase) {
        (CellKind::Tpms(tpms), Phase::Solid { .. }) => {
            let (iso, joins_above) = tpms.solid_floor();
            let floor_typed = tpms.density_for_isovalue(iso);
            let light = Limit {
                sentence: format!(
                    "and {what} this light is separate pieces — it holds together only above \
                     about {} built",
                    fmt_rho(joins_above)
                ),
                at: joins_above,
                extreme: "least that holds together",
            };
            if rho <= tpms.solid_density_built(floor_typed) {
                return Err(light);
            }
            let most = tpms.solid_density_built(SOLID_MOST);
            if rho > most {
                return Err(Limit {
                    sentence: format!("and the Density row stops at 0.99, which builds {most:.3}"),
                    at: most,
                    extreme: "stiffest",
                });
            }
            let typed = bisect(floor_typed, SOLID_MOST, |t| tpms.solid_density_built(t) >= rho);
            settle(typed, light)
        }
        (CellKind::Tpms(tpms), Phase::Sheet { .. }) => {
            let limit = tpms.wall_limit();
            let most = tpms.sheet_density(limit).unwrap_or(0.0);
            let met = Limit {
                sentence: format!("and {what}'s walls meet at {}", fmt_rho(most)),
                at: most,
                extreme: "stiffest",
            };
            if rho > most {
                return Err(met);
            }
            let w = bisect(0.0, limit, |w| tpms.sheet_density(w).is_some_and(|d| d >= rho));
            settle(w * cell_mm, met)
        }
        (CellKind::Beam(beam), Phase::Strut { radius_mm: _ }) => {
            let limit = beam.merge_radius();
            let most = beam.strut_density(limit).unwrap_or(0.0);
            let met = Limit {
                sentence: format!("and {what}'s struts meet at {}", fmt_rho(most)),
                at: most,
                extreme: "stiffest",
            };
            if rho > most {
                return Err(met);
            }
            let r = bisect(0.0, limit, |r| beam.strut_density(r).is_some_and(|d| d >= rho));
            settle(r * cell_mm, met)
        }
        // Exact: the open hexagon is `w − t` across, so `ρ = 1 − (1 − t/w)²`.
        (CellKind::Prism(_), Phase::Sheet { .. }) => {
            let full = Limit {
                sentence: "and a wall that thick fills the cell".to_string(),
                at: 1.0,
                extreme: "stiffest",
            };
            if rho >= 1.0 {
                return Err(full);
            }
            settle(cell_mm * (1.0 - (1.0 - rho).sqrt()), full)
        }
        _ => Err(Limit {
            sentence: "and this cell does not take this phase".to_string(),
            at: rho,
            extreme: "stiffest",
        }),
    }
}

/// The least `x` in `[lo, hi]` at which `reached` holds, to eighty halvings; `reached` must be
/// false at `lo`, true at `hi`, and switch once.
fn bisect(mut lo: f64, mut hi: f64, reached: impl Fn(f64) -> bool) -> f64 {
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if reached(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// What Set writes, or why it will not.
///
/// **Only a literal is ever written** (§7.2): a named parameter may be read by other blocks,
/// and a shared parameter was once silently changed by a panel. A ramp's end is written only
/// where the ramp drives nothing but this lattice, and only that end, so a grade that runs thick
/// to thin stays that way.
#[allow(clippy::too_many_arguments)]
fn write_for(
    doc: &Document,
    graph: &Graph,
    id: NodeId,
    node: &Node,
    end: End,
    number: f64,
    word: &str,
    unit: &str,
) -> Result<SetWrite, String> {
    let Node::Lattice { phase, .. } = node else {
        return Err("that block is not a lattice".to_string());
    };
    let named = |what: &str, id: fieldform_scene::ParamId| {
        let name = doc.parameters.get(&id).map(|p| p.name.clone()).unwrap_or_else(|| id.to_string());
        format!(
            "{what} is the parameter '{name}'. Set it there to {}{unit} — this button never \
             rewrites a parameter other blocks may read.",
            fmt_number(number)
        )
    };
    match end {
        End::Typed => match phase.param() {
            ParamRef::Literal(_) => {
                let mut next = *phase;
                *next.param_mut() = ParamRef::Literal(number);
                Ok(SetWrite::Phase(next))
            }
            ParamRef::Named(param) => Err(named(&format!("The {word}"), *param)),
            ParamRef::Field(_) => Err(format!("the {word} is driven by a ramp: set each Out end instead")),
        },
        End::OutMin | End::OutMax => {
            let Some(ramp) = phase.param().driver() else {
                return Err(format!("the {word} is one number, not a ramp's range"));
            };
            let Some(Node::Ramp { input, in_min, in_max, out_min, out_max, shape }) =
                graph.node(ramp).cloned()
            else {
                return Err(format!("the {word} is driven by a block that is not a ramp"));
            };
            if graph.nodes().any(|(other, n)| other != id && n.inputs().contains(&ramp)) {
                return Err("That ramp drives something else too, so its Out numbers are not this \
                            block's to rewrite — set them on the ramp, knowing what else they move."
                    .to_string());
            }
            let slot = match end {
                End::OutMin => out_min,
                _ => out_max,
            };
            match slot {
                ParamRef::Literal(_) => {}
                ParamRef::Named(param) => {
                    return Err(named(&format!("The ramp's {} number", end_word(end)), param))
                }
                ParamRef::Field(_) => return Err("that end is driven, not typed".to_string()),
            }
            let (out_min, out_max) = match end {
                End::OutMin => (ParamRef::Literal(number), out_max),
                _ => (out_min, ParamRef::Literal(number)),
            };
            Ok(SetWrite::Ramp(ramp, Node::Ramp { input, in_min, in_max, out_min, out_max, shape }))
        }
    }
}

fn end_word(end: End) -> &'static str {
    match end {
        End::Typed => "typed",
        End::OutMin => "Out-min",
        End::OutMax => "Out-max",
    }
}

/// "wall 0.60 mm", "density 0.3412".
pub(crate) fn number_text(word: &str, number: f64, unit: &str) -> String {
    format!("{word} {}{unit}", fmt_number(number))
}

/// "by 4 other sources: wall 0.52–1.46 mm, 3 of them past or without a stated range; 1 other
/// source cannot build it" -- the other sources along the default's own direction, which is
/// the readout's range seen from the other side (§7.1). A row along another direction answers
/// another question, so it is on the hover only. `None` where no other source shares the
/// direction.
pub(crate) fn others_span(answer: &TargetAnswer) -> Option<String> {
    let key = direction_group(answer.default.row.direction);
    let same: Vec<&Offer> = answer
        .others
        .iter()
        .filter(|o| direction_group(o.row.direction) == key)
        .collect();
    if same.is_empty() {
        return None;
    }
    let numbers: Vec<f64> = same.iter().filter_map(|o| o.number.as_ref().ok().copied()).collect();
    let cannot = same.len() - numbers.len();
    let unsure = same.iter().filter(|o| o.number.is_ok() && o.caution.is_some()).count();
    let mut parts = Vec::new();
    if !numbers.is_empty() {
        let (lo, hi) = min_max(numbers.iter().copied());
        let n = numbers.len();
        let span = if fmt_number(lo) == fmt_number(hi) {
            number_text(answer.word, lo, answer.unit)
        } else {
            format!("{} {}–{}{}", answer.word, fmt_number(lo), fmt_number(hi), answer.unit)
        };
        let unsure = match (unsure, n) {
            (0, _) => String::new(),
            (1, 1) => ", past or without a stated range".to_string(),
            (k, _) => format!(", {k} of them past or without a stated range"),
        };
        parts.push(format!("by {n} other source{}: {span}{unsure}", if n == 1 { "" } else { "s" }));
    }
    if cannot > 0 {
        parts.push(format!(
            "{cannot} other source{} cannot build it",
            if cannot == 1 { "" } else { "s" }
        ));
    }
    Some(parts.join("; "))
}

/// Every source's answer to a target, the default first, each with its direction and, where the
/// density it needs is past or without its range, that.
pub(crate) fn offers_hover(answer: &TargetAnswer) -> String {
    std::iter::once((&answer.default, " (the default)"))
        .chain(answer.others.iter().map(|o| (o, "")))
        .map(|(offer, marker)| {
            format!("{}{marker}: {}", source_and_direction(offer.row), offer_text(answer, offer))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// "radius 1.1172 mm — needs a density of 0.45, outside its range, 0.001–0.10", or why there is
/// no number.
pub(crate) fn offer_text(answer: &TargetAnswer, offer: &Offer) -> String {
    match (&offer.number, &offer.caution) {
        (Ok(n), None) => number_text(answer.word, *n, answer.unit),
        (Ok(n), Some(caution)) => {
            format!("{} — {caution}", number_text(answer.word, *n, answer.unit))
        }
        (Err(why), _) => why.clone(),
    }
}

/// "Li 2019, along the source's build direction (cell axes not stated)".
pub(crate) fn source_and_direction(row: &PropertyRow) -> String {
    format!("{}, {}", row.source.short, row.direction.phrase())
}

/// A target as it was typed: 5, not "5.0".
pub(crate) fn typed_percent(target: f64) -> String {
    fmt_number(target * 100.0)
}
