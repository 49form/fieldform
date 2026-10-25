// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which blend moves which critical surface, and by how much (`blend-integrity-spec.md` §5).
//!
//! A [`Critical`] mark says a node's surface must end up where the hard boolean would put it.
//! This measures whether it did. It is a **report**: derived, never stored in the document, and
//! a part that has not been verified reads as unverified — not as passing.
//!
//! # What is measured, and against what
//!
//! The **hard part** is the same graph with every fillet radius at nought
//! ([`Graph::hardened`]). The **critical patch** is where the marked node's zero set and the
//! hard part's surface coincide — not the node's whole surface, since a bore's cylinder runs out
//! past the part. Every offset, twist and warp is in both graphs, so nothing but a blend can
//! account for a difference between them.
//!
//! # Two departures from the spec's first draft, both forced by option A
//!
//! **Movement is marched, not read off the value.** §5 said `max |F_part(p)|`. Since §3 A a
//! rounded blend's value is divided by 2^(1/n), so `|F_part|` is a *lower bound* on how far the
//! surface is, and reading it as the movement understates by up to that factor per nesting — in
//! the direction that reports "held" for a face that moved past its tolerance. So from each
//! patch sample this walks along the marked surface's own normal until the part's surface is
//! met, stepping by the value, which as a lower bound can never step past it. The answer is a
//! length in millimetres, measured the way a machinist would: how far did this face move, square
//! to itself.
//!
//! **A cause is a counterfactual, not a zone test.** §5 named every blend whose zone
//! `a < k ∧ b < k` holds a sample. The zone holds the whole overlap interior of its operands
//! (§1), so a bore drilled through two blended slabs sits inside their blend's zone along its
//! whole length while that blend moves it by nothing — and the sentence wants a number per
//! blend, which a zone cannot give. Each rounded blend is instead measured *alone*: the hard
//! part with that one fillet restored. What it moves the patch by is its own doing.

use fieldform_scene::{Critical, Document, Graph, NodeId};
use nalgebra::{Point3, Vector3};

use crate::{
    mesh_field_adaptive, part_extent, CompileError, Extent, GraphField, MeshTolerance, ScalarField,
};

/// Below this a movement is arithmetic, not geometry: the march stops within it of a surface.
const FLOOR_MM: f64 = 1e-6;

/// The most patch samples measured per mark. Past this the mesh is strided rather than the
/// cost growing with it: a verify is asked for by a person who is waiting.
const MOST_SAMPLES: usize = 4000;

/// What one blend, alone, does to one critical surface.
#[derive(Clone, Debug, PartialEq)]
pub struct BlendMovement {
    pub blend: NodeId,
    /// The furthest any patch sample moved with only this blend rounded, in millimetres.
    pub movement: f64,
}

/// What became of one critical surface.
#[derive(Clone, Debug, PartialEq)]
pub struct CriticalReport {
    pub node: NodeId,
    pub tolerance: f64,
    /// How many points of the critical patch were measured. **Nought means there is no patch**
    /// — the hard part has no surface on this node — and that is neither held nor moved.
    pub samples: usize,
    /// The furthest any sample moved, square to the surface, in millimetres.
    pub movement: f64,
    /// Where that was.
    pub at: Option<Point3<f64>>,
    /// The march ran out of part before it met a surface, so `movement` is only a floor.
    pub at_least: bool,
    /// The blends that move it, worst first. Empty when nothing does.
    pub causes: Vec<BlendMovement>,
}

/// The three things a mark can turn out to be.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Held,
    Moved,
    /// The part has no surface on the marked node, so there was nothing to measure.
    NoPatch,
}

impl CriticalReport {
    pub fn verdict(&self) -> Verdict {
        if self.samples == 0 {
            Verdict::NoPatch
        } else if self.movement <= self.tolerance && !self.at_least {
            Verdict::Held
        } else {
            Verdict::Moved
        }
    }

    /// Whether `blend` is one this report lays a movement above tolerance on.
    pub fn blames(&self, blend: NodeId) -> bool {
        self.causes.iter().any(|c| c.blend == blend && c.movement > self.tolerance)
    }

    /// The report as plain sentences — the same text for the console, the panel and a model
    /// reading the transcript, which is the point of §7. `name` is how the caller refers to a
    /// node; the scene crate has no names to offer.
    pub fn sentences(&self, name: &dyn Fn(NodeId) -> String) -> Vec<String> {
        let this = name(self.node);
        match self.verdict() {
            Verdict::NoPatch => vec![format!(
                "\u{2018}{this}\u{2019} has no surface on the hard part, so there is nothing to \
                 hold. Mark the block as it is placed, not the shape inside a placement."
            )],
            Verdict::Held => vec![format!(
                "\u{2018}{this}\u{2019} is held: moved {} mm at most, within {} mm, over {} \
                 samples.",
                mm(self.movement),
                mm(self.tolerance),
                self.samples,
            )],
            Verdict::Moved => {
                let mut out = vec![format!(
                    "\u{2018}{this}\u{2019} moves by {}{} mm, against a tolerance of {} mm{}.",
                    if self.at_least { "at least " } else { "up to " },
                    mm(self.movement),
                    mm(self.tolerance),
                    self.at
                        .map(|p| format!(", worst at ({:.2}, {:.2}, {:.2})", p.x, p.y, p.z))
                        .unwrap_or_default(),
                )];
                for cause in &self.causes {
                    out.push(format!(
                        "Blend \u{2018}{}\u{2019} moves \u{2018}{this}\u{2019} by up to {} mm.",
                        name(cause.blend),
                        mm(cause.movement),
                    ));
                }
                if self.causes.is_empty() {
                    // Possible in principle: two blends that each leave the face alone and
                    // move it together. Said rather than left as a movement with no author.
                    out.push(format!(
                        "No single blend moves \u{2018}{this}\u{2019} on its own; the movement \
                         comes from blends acting together."
                    ));
                }
                out
            }
        }
    }
}

/// Enough figures to act on and no more: a hundredth of a millimetre matters, a billionth is
/// the march's own floor.
fn mm(v: f64) -> String {
    if v < 1e-4 { "0.000".to_string() } else { format!("{v:.3}") }
}

/// Measures every critical mark on `graph`. One report per mark, in the marks' own order.
///
/// `tolerance_mm` is the compile tolerance — how finely cages and curves are read — and is not
/// any mark's tolerance.
///
/// **This measures the part as it is, and the part is protected**: a mark tapers every fillet
/// to nothing at its surface (§6), so what this is expected to say is *held*, and it is the
/// check on that promise rather than a restatement of it — it shares no code with the taper.
/// [`verify_unprotected`] says what the same marks would have read without it.
pub fn verify_critical(
    document: &Document,
    graph: &Graph,
    tolerance_mm: f64,
) -> Result<Vec<CriticalReport>, CompileError> {
    verify_marks(document, graph, graph.critical(), tolerance_mm)
}

/// The same report over the part with its protection taken off: which blend *would* move each
/// critical surface, and by how much. It is what §5 described before §6 existed, and what says
/// how much fillet a mark is costing.
pub fn verify_unprotected(
    document: &Document,
    graph: &Graph,
    tolerance_mm: f64,
) -> Result<Vec<CriticalReport>, CompileError> {
    verify_marks(document, &graph.without_critical(), graph.critical(), tolerance_mm)
}

fn verify_marks(
    document: &Document,
    graph: &Graph,
    marks: &[Critical],
    tolerance_mm: f64,
) -> Result<Vec<CriticalReport>, CompileError> {
    if marks.is_empty() {
        return Ok(Vec::new());
    }
    let hard_graph = graph.hardened(None);
    let hard = GraphField::compile_graph(document, &hard_graph, tolerance_mm)?;
    let part = GraphField::compile_graph(document, graph, tolerance_mm)?;

    // Each rounded blend alone, compiled once and asked about by every mark.
    let resolves_round = |id: NodeId| {
        graph.node(id).and_then(|n| n.fillet()).is_some_and(|f| {
            f.radius.resolve(&|p| document.parameter_value(p)).is_some_and(|k| k > 0.0)
        })
    };
    let mut rounded: Vec<NodeId> = graph.nodes().map(|(id, _)| id).filter(|id| resolves_round(*id)).collect();
    rounded.sort_by_key(|id| id.raw());
    let alone: Vec<(NodeId, GraphField)> = rounded
        .into_iter()
        .map(|id| {
            GraphField::compile_graph(document, &graph.hardened(Some(id)), tolerance_mm)
                .map(|f| (id, f))
        })
        .collect::<Result<_, _>>()?;

    // Where to look. The patch lies on the hard part's surface, so the hard part's box holds
    // it; a part with no box has no patch that can be sampled.
    let Extent::Bounded { lo, hi } = part_extent(document, &hard_graph) else {
        return Ok(marks.iter().map(no_patch).collect());
    };
    let reach = (hi - lo).norm();
    let pad = Vector3::repeat(reach * 0.01 + 1e-3);
    let bounds = (lo - pad, hi + pad);

    marks
        .iter()
        .map(|mark| {
            let Some(own) = graph.subtree(mark.node) else { return Ok(no_patch(mark)) };
            let own = GraphField::compile_graph(document, &own, tolerance_mm)?;
            let samples = patch_samples(&own, &hard, bounds, reach, mark.tolerance);
            let (movement, at, at_least) = worst_movement(&samples, &hard, &part, reach);
            let mut causes: Vec<BlendMovement> = alone
                .iter()
                .map(|(blend, field)| BlendMovement {
                    blend: *blend,
                    movement: worst_movement(&samples, &hard, field, reach).0,
                })
                .filter(|c| c.movement > FLOOR_MM.max(mark.tolerance * 1e-3))
                .collect();
            causes.sort_by(|a, b| b.movement.total_cmp(&a.movement));
            Ok(CriticalReport {
                node: mark.node,
                tolerance: mark.tolerance,
                samples: samples.len(),
                movement,
                at,
                at_least,
                causes,
            })
        })
        .collect()
}

fn no_patch(mark: &Critical) -> CriticalReport {
    CriticalReport {
        node: mark.node,
        tolerance: mark.tolerance,
        samples: 0,
        movement: 0.0,
        at: None,
        at_least: false,
        causes: Vec::new(),
    }
}

/// A patch sample: a point on the marked surface where the hard part has its surface too, and
/// the unit direction there that leads *out of the hard part*.
struct Sample {
    at: Point3<f64>,
    outward: Vector3<f64>,
}

fn gradient(field: &dyn ScalarField, p: Point3<f64>, h: f64) -> Vector3<f64> {
    let d = |o: Vector3<f64>| field.distance(p + o) - field.distance(p - o);
    Vector3::new(d(Vector3::x() * h), d(Vector3::y() * h), d(Vector3::z() * h)) / (2.0 * h)
}

/// Meshes the marked node alone, settles each vertex onto its zero set, and keeps the ones the
/// hard part's surface passes through.
///
/// Meshed rather than projected from a grid, as §5 asks: a mesh puts samples *on* the surface
/// at a density that follows its curvature. The vertices are then settled by a few Newton
/// steps, which need a gradient and not a true distance — so a node that only claims a bound is
/// as good as one that claims a distance — because the keep test below compares against the
/// mark's tolerance, and a vertex left a chord's sag off the surface would fail it for the
/// mesher's reasons rather than the part's.
fn patch_samples(
    own: &GraphField,
    hard: &GraphField,
    bounds: (Point3<f64>, Point3<f64>),
    reach: f64,
    tolerance: f64,
) -> Vec<Sample> {
    let tol = MeshTolerance {
        chord_mm: (reach * 2e-4).max(0.01),
        // A flat face meets every other test at any size; capped so it is still sampled.
        max_edge_mm: reach / 48.0,
        min_cell_mm: reach / 1024.0,
        ..MeshTolerance::default()
    };
    let mesh = mesh_field_adaptive(own, bounds, &tol);
    let h = (reach * 1e-6).max(1e-6);
    let stride = mesh.positions.len().div_ceil(MOST_SAMPLES).max(1);
    let keep = tolerance.max(FLOOR_MM * 10.0);
    mesh.positions
        .iter()
        .step_by(stride)
        .filter_map(|&vertex| {
            let mut p = vertex;
            for _ in 0..4 {
                let g = gradient(own, p, h);
                let n2 = g.norm_squared();
                if n2 < 1e-12 {
                    return None;
                }
                p -= g * (own.distance(p) / n2);
            }
            if own.distance(p).abs() > FLOOR_MM || hard.distance(p).abs() >= keep {
                return None;
            }
            let normal = gradient(own, p, h).try_normalize(1e-9)?;
            // Which way along the normal leaves the hard part. A bore's own gradient points
            // into the material it was cut from, so the node's sense is not the part's; asked
            // of the hard part a little way off the surface, where a crease does not confuse it.
            let probe = (reach * 1e-4).max(1e-4);
            // **Not on a crease of the marked node itself.** The mesher puts vertices on edges
            // on purpose, and the differenced gradient there is the diagonal between two faces
            // — a march along it measured a box's top as having risen 1.73 mm where it rose
            // 1.46. A face has one normal a step to either side; an edge does not.
            // Asked *along* the surface, not along the normal: on an edge the diagonal is
            // steady along itself, and it is a step sideways that lands on one face or the other.
            let aside = if normal.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
            let t1 = normal.cross(&aside).normalize();
            let t2 = normal.cross(&t1);
            let steady = [t1, -t1, t2, -t2].into_iter().all(|t| {
                gradient(own, p + t * probe, h)
                    .try_normalize(1e-9)
                    .is_some_and(|g| g.dot(&normal) > 0.9995)
            });
            if !steady {
                return None;
            }
            let rise =hard.distance(p + normal * probe) - hard.distance(p - normal * probe);
            if rise.abs() < probe * 1e-3 {
                // An edge of the patch, where the hard part's surface turns away on one side:
                // neither direction is "out", and a face's movement is not measured at its rim.
                return None;
            }
            Some(Sample { at: p, outward: normal * rise.signum() })
        })
        .collect()
}

/// Where `field`'s surface is along a sample's normal line: millimetres from the sample,
/// positive outward. The second value says the march gave up before meeting one.
fn surface_along(field: &GraphField, s: &Sample, reach: f64) -> (f64, bool) {
    let start = field.distance(s.at);
    if start.abs() < FLOOR_MM {
        return (0.0, false);
    }
    // Buried under added material: the surface is outward. Left standing in air by material
    // taken away: it is inward.
    let sign = if start < 0.0 { 1.0 } else { -1.0 };
    let mut t = 0.0;
    for _ in 0..512 {
        let f = field.distance(s.at + s.outward * (sign * t));
        if f.abs() < FLOOR_MM || (f < 0.0) != (start < 0.0) {
            return (sign * t, false);
        }
        // The value bounds the distance to the surface in *every* direction from below, so it
        // bounds it along this one: the step cannot pass the first crossing.
        t += f.abs();
        if t > reach {
            break;
        }
    }
    (sign * t, true)
}

/// The furthest `field`'s surface sits from the hard part's, along any sample's normal.
///
/// **A difference of two marches, not one.** A sample is kept when the hard part's surface is
/// within the mark's tolerance of it, not only when it is exactly on it, so reading the part's
/// surface position alone would report that offset as a movement. Both surfaces are found along
/// the same line and the movement is the gap between them.
///
/// Returns the distance, where it was, and whether a march gave up before meeting a surface.
fn worst_movement(
    samples: &[Sample],
    hard: &GraphField,
    field: &GraphField,
    reach: f64,
) -> (f64, Option<Point3<f64>>, bool) {
    let mut worst = (0.0f64, None, false);
    for s in samples {
        let (was, _) = surface_along(hard, s, reach);
        let (is, gave_up) = surface_along(field, s, reach);
        let moved = (is - was).abs();
        if moved > worst.0 && moved > FLOOR_MM {
            worst = (moved, Some(s.at), gave_up);
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_scene::{BooleanOp, Continuity, Fillet, Node, ParamRef, Place};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn fillet(k: f64) -> Fillet {
        Fillet { radius: lit(k), continuity: Continuity::G1, second: None }
    }

    fn slab(graph: &mut Graph, x_lo: f64, x_hi: f64) -> NodeId {
        graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit((x_lo + x_hi) / 2.0), lit(0.0), lit(-10.0)]),
            size: [lit(x_hi - x_lo), lit(40.0), lit(20.0)],
            turn: fieldform_scene::no_turn(),
        })
    }

    fn names(graph: &Graph) -> impl Fn(NodeId) -> String + '_ {
        |id| match graph.node(id) {
            Some(Node::Blend { .. }) => format!("blend-{}", id.raw() % 1000),
            _ => "face".to_string(),
        }
    }

    /// §2.1 as the report sees it: the slab's top is critical, the blend raises it 0.293 k
    /// over the overlap, and the report says so in millimetres and names the blend.
    ///
    /// **The number is the geometric one.** Under option A the part's value at the hard top is
    /// −1.4645 here only because the field is linear above a flat face; the march is what makes
    /// the same hold where it is not.
    #[test]
    fn a_shared_face_is_reported_as_moved_by_the_documented_amount() {
        let k = 5.0;
        let mut graph = Graph::empty();
        let a = slab(&mut graph, -40.0, 10.0);
        let b = slab(&mut graph, -10.0, 40.0);
        let blend = graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet: fillet(k) });
        graph.mark_critical(Critical::new(a));

        let reports = verify_unprotected(&Document::new(), &graph, 1e-3).expect("compiles");
        let report = &reports[0];
        assert_eq!(report.verdict(), Verdict::Moved);
        let rise = k * (1.0 - 1.0 / 2f64.sqrt());
        assert!(
            (report.movement - rise).abs() < 1e-4,
            "the top rises {rise} and the report says {} at {:?}", 
            report.movement, report.at
        );
        assert!(report.blames(blend));
        let text = report.sentences(&names(&graph)).join(" ");
        assert!(text.contains("1.464") && text.contains("Blend"), "{text}");
    }

    /// The test §8 asks for. A body of two blended slabs, a boss blended on at one end, and a
    /// bore through the other end — **through the slabs' overlap**, so it lies inside the
    /// slab blend's zone along its whole length. The zone test of the spec's first draft names
    /// both blends here; only the bore's own fillet moves the bore.
    #[test]
    fn the_report_names_the_blend_that_moves_the_face() {
        let mut graph = Graph::empty();
        let a = slab(&mut graph, -40.0, 10.0);
        let b = slab(&mut graph, -10.0, 40.0);
        let body = graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet: fillet(3.0) });
        let boss = graph.add(Node::Sphere {
            centre: Place::At([lit(30.0), lit(0.0), lit(0.0)]),
            radius: lit(6.0),
        });
        let bossed =
            graph.add(Node::Blend { op: BooleanOp::Union, a: body.into(), b: boss.into(), fillet: fillet(2.0) });
        let bore = graph.add(Node::Cylinder { edge: Default::default(),
            ends: fieldform_scene::Ends::between(
                Place::At([lit(0.0), lit(0.0), lit(-30.0)]),
                Place::At([lit(0.0), lit(0.0), lit(10.0)]),
            ),
            radius: lit(4.0),
        });
        let drilled = graph.add(Node::Blend {
            op: BooleanOp::Difference,
            a: bossed.into(),
            b: bore.into(),
            fillet: fillet(1.5),
        });
        graph.mark_critical(Critical::new(bore));

        let reports = verify_unprotected(&Document::new(), &graph, 1e-3).expect("compiles");
        let report = &reports[0];
        assert!(report.samples > 50, "only {} samples on the bore", report.samples);
        assert_eq!(report.verdict(), Verdict::Moved);
        let named: Vec<NodeId> = report.causes.iter().map(|c| c.blend).collect();
        assert_eq!(named, vec![drilled], "exactly the bore's own fillet, of {body:?} {bossed:?}");
        assert!(report.causes[0].movement > 0.1 && report.causes[0].movement <= 1.5 + 1e-6);

        // Protected, which is what a mark means, the same bore does not move at all — measured
        // by this module's march, which knows nothing of how the taper is done.
        let protected = &verify_critical(&Document::new(), &graph, 1e-3).expect("compiles")[0];
        assert_eq!(protected.verdict(), Verdict::Held, "{:?}", protected.sentences(&names(&graph)));
        assert!(protected.movement <= FLOOR_MM, "a protected bore moved {}", protected.movement);
        assert_eq!(protected.samples, report.samples);

        // And with that one fillet hard, the same bore is held unprotected too — by the two
        // blends still there.
        graph.node_mut(drilled).unwrap().fillet_mut().unwrap().radius = lit(0.0);
        let held = &verify_unprotected(&Document::new(), &graph, 1e-3).expect("compiles")[0];
        assert_eq!(held.verdict(), Verdict::Held, "{:?}", held.sentences(&names(&graph)));
        assert!(held.causes.is_empty());
    }

    /// A mark on the shape inside a placement has no surface on the part, and says that rather
    /// than reading as held — an empty patch passing would be the unverified-reads-as-passing
    /// failure by another door.
    #[test]
    fn a_mark_with_no_surface_on_the_part_is_neither_held_nor_moved() {
        let mut graph = Graph::empty();
        let inner = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(5.0) });
        let placed = graph.add(Node::Transform {
            input: inner.into(),
            translation: [lit(100.0), lit(0.0), lit(0.0)],
            rotation: [lit(0.0); 3],
            scale: lit(1.0),
            about: Place::origin(),
        });
        let other = graph.add(Node::Sphere {
            centre: Place::At([lit(104.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        graph.add(Node::Blend { op: BooleanOp::Union, a: placed.into(), b: other.into(), fillet: fillet(1.0) });
        graph.mark_critical(Critical::new(inner));
        let report = &verify_critical(&Document::new(), &graph, 1e-3).expect("compiles")[0];
        assert_eq!(report.verdict(), Verdict::NoPatch);
        assert!(report.sentences(&|_| "ball".into())[0].contains("no surface"));
    }

    #[test]
    fn a_part_with_no_marks_has_nothing_to_report() {
        let mut graph = Graph::empty();
        slab(&mut graph, -1.0, 1.0);
        assert!(verify_critical(&Document::new(), &graph, 1e-3).unwrap().is_empty());
    }
}

// ---------------------------------------------------------------------------------------------
// The field's own promise
// ---------------------------------------------------------------------------------------------

/// Where a field changes faster than it promises — see [`find_overstep`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overstep {
    /// The fastest the value changes there, in millimetres per millimetre travelled.
    pub rate: f64,
    /// What the field says that can be — its [`ScalarField::lipschitz`], 1.0 for a distance.
    pub allowed: f64,
    /// How far apart the two readings that showed it were.
    pub across: f64,
    /// What the value did over that span, in millimetres.
    pub moved: f64,
    /// The middle of the pair.
    pub at: Point3<f64>,
}

/// How many suspicious pairs the coarse pass hands to the refinement.
const CANDIDATES: usize = 96;

/// Halvings each candidate is put through, each one looking at half the span for twice the
/// detail. Twenty takes a 3 mm pair down to three nanometres.
const HALVINGS: usize = 20;

/// Hunts for a place where the field changes faster than it says it can.
///
/// **The one promise every consumer here is built on.** The mesher prunes a cell when the value
/// at its centre exceeds the cell's own half-diagonal, and the tracer steps by the value: both
/// believe [`ScalarField::lipschitz`]. Where the field moves faster, cells that do hold surface
/// are cut away and rays step through walls — and the symptom is holes nowhere near anything
/// that looks wrong. Measured at **2.86 mm per millimetre** beside a spiked conformal surface,
/// with 132 open edges in the mesh of it, where the fold radius and the cell size both looked
/// fine (`examples/surface_edit_tessellation.rs`).
///
/// **Why a sweep alone cannot find it.** The region is thinner than any affordable grid: read
/// over a 3 mm spacing, that same fault averages to well under what an ordinary flat wall reads,
/// so it hides under the legitimate rate of 1. What gives it away is *nonlinearity* — the
/// midpoint of a pair sitting far from the average of its ends, which is nought on a plane and
/// small on a curve. So the coarse pass ranks pairs by that, and each suspect is then bisected
/// toward whichever half holds more of the change, measuring the rate at every scale down. The
/// worst rate seen anywhere on the way is the answer.
///
/// A crease — two planes meeting, which every boolean makes — stays at rate 1 under this,
/// because neither side of it ever exceeds 1. `None` means nothing was found from this grid,
/// not that the field is sound everywhere.
pub fn find_overstep(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    per_axis: usize,
) -> Option<Overstep> {
    use rayon::prelude::*;

    let (lo, hi) = bounds;
    let n = per_axis.max(2);
    let extent = hi - lo;
    let step = Vector3::new(
        extent.x / (n - 1) as f64,
        extent.y / (n - 1) as f64,
        extent.z / (n - 1) as f64,
    );
    let at = |i: usize, j: usize, k: usize| {
        Point3::new(lo.x + step.x * i as f64, lo.y + step.y * j as f64, lo.z + step.z * k as f64)
    };

    // The coarse pass, ranked by how far each pair's midpoint sits from the average of its ends.
    // A plane gives nought, a curve a little, and anything with a thin steep region rather more —
    // which is what is being looked for, since its *rate* at this spacing is unremarkable.
    let mut candidates: Vec<(f64, Point3<f64>, Point3<f64>)> = (0..n)
        .into_par_iter()
        .map(|k| {
            let mut mine: Vec<(f64, Point3<f64>, Point3<f64>)> = Vec::new();
            // **Two scores, because a fault hides from either one alone.** A thin steep band
            // read across symmetrically has its midpoint exactly where a straight line would
            // put it, so its bend is nought; a band whose total change is small against the
            // spacing leaves the rate unremarkable. Ranking by both and refining the union
            // costs one more list and catches both shapes.
            let consider = |a: Point3<f64>, b: Point3<f64>, mine: &mut Vec<_>| {
                let (va, vb) = (field.distance(a), field.distance(b));
                let m = nalgebra::center(&a, &b);
                let span = (b - a).norm();
                let bend = (field.distance(m) - (va + vb) * 0.5).abs();
                let rate = if span > 0.0 { (vb - va).abs() / span } else { 0.0 };
                // Scaled to the same units so one list can hold both: a bend is a length, and
                // dividing it by the span makes it the rate the curvature is worth.
                mine.push(((bend / span.max(f64::MIN_POSITIVE)).max(rate), a, b));
            };
            for j in 0..n {
                for i in 0..n {
                    let here = at(i, j, k);
                    if i + 1 < n {
                        consider(here, at(i + 1, j, k), &mut mine);
                    }
                    if j + 1 < n {
                        consider(here, at(i, j + 1, k), &mut mine);
                    }
                    if k + 1 < n {
                        consider(here, at(i, j, k + 1), &mut mine);
                    }
                }
            }
            mine.sort_by(|a, b| b.0.total_cmp(&a.0));
            mine.truncate(CANDIDATES);
            mine
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a.sort_by(|x, y| y.0.total_cmp(&x.0));
            a.truncate(CANDIDATES);
            a
        });
    candidates.truncate(CANDIDATES);

    let allowed = field.lipschitz();
    let worst = candidates
        .into_par_iter()
        .filter_map(|(_, mut a, mut b)| {
            let mut worst: Option<Overstep> = None;
            // **Quartered rather than halved, because a steep band has edges.** Bisecting keeps
            // the half holding more of the change, which around a thin band is the half holding
            // one of its edges — and an edge reads as the average of the steep side and the
            // gentle one. Four sub-intervals let the search step *inside* the band and stay
            // there: measured 1.56 against a true 2.5 when this halved, and 2.5 when it
            // quartered.
            const PIECES: usize = 4;
            for _ in 0..HALVINGS {
                let span = (b - a).norm() / PIECES as f64;
                if span <= 0.0 {
                    break;
                }
                let along = (b - a) / PIECES as f64;
                let mut values = [0.0f64; PIECES + 1];
                for (step, slot) in values.iter_mut().enumerate() {
                    *slot = field.distance(a + along * step as f64);
                }
                let mut steepest = (0.0f64, 0usize);
                for piece in 0..PIECES {
                    let moved = (values[piece + 1] - values[piece]).abs();
                    let rate = moved / span;
                    if rate > steepest.0 {
                        steepest = (rate, piece);
                    }
                    let mid = a + along * (piece as f64 + 0.5);
                    if worst.is_none_or(|w: Overstep| rate > w.rate) {
                        worst = Some(Overstep { rate, allowed, across: span, moved, at: mid });
                    }
                }
                let (start, piece) = (a, steepest.1);
                a = start + along * piece as f64;
                b = start + along * (piece + 1) as f64;
            }
            // A little room for the arithmetic: a flat wall reads exactly the bound, and
            // rounding puts it either side.
            worst.filter(|w| w.rate > allowed * 1.05)
        })
        .collect::<Vec<Overstep>>();
    worst.into_iter().max_by(|a, b| a.rate.total_cmp(&b.rate))
}

#[cfg(test)]
mod promise_tests {
    use super::*;

    struct Ball;
    impl ScalarField for Ball {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.coords.norm() - 10.0
        }
    }

    /// A cube, whose distance is full of creases — legal, and the thing this must not mistake
    /// for a fault: every boolean in every document makes them.
    struct Cube;
    impl ScalarField for Cube {
        fn distance(&self, p: Point3<f64>) -> f64 {
            let q = p.coords.map(|c| c.abs() - 8.0);
            q.map(|c| c.max(0.0)).norm() + q.max().min(0.0)
        }
    }

    /// A ball whose value is squeezed through a thin slab: continuous, and three times steeper
    /// than a distance may be inside 0.3 mm of `x = 1.3`. The shape of the fault measured on the
    /// owner's spiked conformal lattice — not a step, a slope too steep to be trusted.
    struct Steep;
    impl ScalarField for Steep {
        fn distance(&self, p: Point3<f64>) -> f64 {
            let base = p.coords.norm() - 10.0;
            let t = ((p.x - 1.3) / 0.3).clamp(-1.0, 1.0);
            base + t * 0.45
        }
    }

    /// **A rate no consumer can work with is found, and the creases every part has are not.**
    /// Read over any affordable grid the fault reads as *slower* than an ordinary wall — it is
    /// thinner than the grid — so this refines toward it instead of sweeping for it.
    #[test]
    fn a_field_that_changes_too_fast_is_found_and_a_crease_is_not() {
        let box_of = (Point3::new(-20.0, -20.0, -20.0), Point3::new(20.0, 20.0, 20.0));

        let caught = find_overstep(&Steep, box_of, 24).expect("2.5 mm per mm is not a distance");
        assert!(caught.rate > 2.0, "the slope is about 2.5, reported {:.2}", caught.rate);
        assert!(
            (caught.at.x - 1.3).abs() < 0.5,
            "the steep region is at x = 1.3, found at {:?}",
            caught.at
        );

        assert!(find_overstep(&Ball, box_of, 24).is_none(), "a ball is a distance everywhere");
        assert!(
            find_overstep(&Cube, box_of, 24).is_none(),
            "a cube's creases are two slopes of one meeting, and neither exceeds it"
        );
    }
}
