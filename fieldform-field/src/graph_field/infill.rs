// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A body with a skin and a filled inside, rewritten into the blocks it is made of.

use super::compile::{CompileError, GraphField};
use super::{graph_extent, operand_extent};
use crate::ScalarField;
use fieldform_scene::{BooleanOp, Document, Graph, Node, NodeId, ParamRef};
use nalgebra::{Point3, Vector3};

/// Every [`Node::Infill`] in `graph` as the three blocks it stands for, or `None` when there is
/// nothing to rewrite, so a document without one clones nothing.
///
/// **A rewrite both compilers read**, as [`crate::with_shared_faces_merged`] is. A second copy of
/// this arithmetic in the shader generator is a second place for a sign to go wrong, and a join
/// needs everything a blend carries (its taper, its bound, its second leg), so the honest way to
/// have those is for the join to be a blend. The node keeps its id, which is what lets everything
/// keyed on a body still find it, and no new op or shader code exists for it.
///
/// With `F` the body, `L` the fill and `t` the wall:
///
/// ```text
/// cavity = Offset(body, -t)                  F + t
/// voids  = Difference(cavity, fill)          max(F + t, -L), smoothly where the join rounds
/// infill = Difference(body, voids)           max(F, min(L, -F - t))
/// ```
///
/// **The outer `max` is last, and that is the whole of why no join can grow the part.** On the
/// body's own surface `F` is nought and the inner term is at most `-t`, whatever the fill does or
/// however far a fillet reaches, so the larger of the two is `F`.
///
/// **A wall of nought is not this formula.** As a set it is right, but as a field it is `|F|` over
/// every part of the surface the fill is not on: a value that falls to nought and comes back up,
/// which is a skin of no thickness. The mesher sees no sign change and is not troubled. A ray is:
/// the march refuses the crossing after slowing to reach it, and every lattice face behind it came
/// out in contour lines. So it lowers to what it means, the plain intersection, which also makes
/// the join's being ignored there a fact rather than a rule to remember: with no skin it would
/// flare every lattice wall where the outer surface cuts it.
///
/// A block that is incomplete is left as it is, since the compile passes over those by name.
///
/// # Errors
///
/// A body with no bounds, as a shell refuses one, and a wall below nought. A wall that follows a
/// ramp never arrives here: [`super::lipschitz::driven_refusal`] has it first.
pub fn with_infills_lowered(
    document: &Document,
    graph: &Graph,
) -> Result<Option<Graph>, CompileError> {
    let infills: Vec<NodeId> = graph
        .nodes()
        .filter(|(_, n)| matches!(n, Node::Infill { .. }))
        .map(|(id, _)| id)
        .collect();
    if infills.is_empty() {
        return Ok(None);
    }
    let incomplete = document.incomplete_in(graph);
    let mut out = graph.clone();
    let mut lowered_any = false;
    for id in infills {
        if incomplete.contains(&id) {
            continue;
        }
        let Some(Node::Infill { input, wall, fill, join }) = graph.node(id).cloned() else {
            continue;
        };
        // The same refusal as a shell's, for the same reason: a half-space with a skin is an
        // infinite slab, which nobody can mesh, measure or make.
        if operand_extent(document, graph, &input).as_box().is_none() {
            return Err(CompileError::InfillNeedsBounds);
        }
        let t = document.resolve(&wall).ok_or(CompileError::UnresolvedOffset)?;
        if t < 0.0 {
            return Err(CompileError::InfillWallBelowNought(t));
        }
        let lowered = if t == 0.0 {
            Node::Boolean { op: BooleanOp::Intersection, inputs: vec![input, fill] }
        } else {
            // Negated, because the cavity is the body shrunk and `Offset` subtracts.
            let cavity = out.add(Node::Offset { input, distance: ParamRef::Literal(-t) });
            let inputs = vec![cavity.into(), fill];
            let voids = out.add(if join.is_plainly_hard() {
                Node::Boolean { op: BooleanOp::Difference, inputs }
            } else {
                Node::Blend { op: BooleanOp::Difference, inputs, fillet: join }
            });
            Node::Boolean { op: BooleanOp::Difference, inputs: vec![input, voids.into()] }
        };
        if let Some(slot) = out.node_mut(id) {
            *slot = lowered;
            lowered_any = true;
        }
    }
    Ok(lowered_any.then_some(out))
}

/// What an infill's inside comes to, for a panel to say what the geometry alone cannot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InfillCavity {
    /// How far the body goes down, in its own field's millimetres. A wall of this or more leaves
    /// no cavity, because the cavity is where the field reads below minus the wall.
    pub deepest: f64,
    /// Whether any of the fill reaches the cavity. `true` when that cannot be ruled out, so a
    /// sheet too thin to have been sampled is never reported missing.
    pub fill_lands: bool,
    /// The block's material as a share of the solid body, from nought to one.
    pub solid_share: f64,
}

impl InfillCavity {
    /// Whether a wall of this thickness leaves anything inside it.
    pub fn has_room(&self, wall: f64) -> bool {
        wall < self.deepest
    }
}

/// [`InfillCavity`] for block `node`, or `None` when it is not a complete infill of a body that
/// can be measured.
///
/// **Sampled, and not cheap on a cage**, which is why the panel asks on a thread of its own. The
/// points are a low-discrepancy sequence rather than a grid, since a grid at a rational multiple of
/// a lattice's period reads the same few phases of it over and over. The deepest of them is then
/// walked downhill, which is what makes the figure good enough to quote: the sampling alone would
/// leave it short by up to a sample's spacing, and the message offers it as a number to type.
pub fn infill_cavity(
    document: &Document,
    graph: &Graph,
    node: NodeId,
    samples: u32,
) -> Option<InfillCavity> {
    let Some(Node::Infill { input, wall, fill, .. }) = graph.node(node) else {
        return None;
    };
    let (body, filler) = (input.source()?, fill.source()?);
    let wall = document.resolve(wall)?;
    let (lo, hi) = graph_extent(document, graph, body).as_box()?;
    let size = hi - lo;
    if samples == 0 || size.min() <= 0.0 {
        return None;
    }
    let tolerance = size.max() / 128.0;
    let compile = |id: NodeId| {
        GraphField::compile_graph(document, &graph.subtree(id)?, tolerance).ok()
    };
    let (body_field, fill_field, part) = (compile(body)?, compile(filler)?, compile(node)?);

    let spacing = (size.x * size.y * size.z / f64::from(samples)).cbrt();
    let (mut inside, mut material) = (0_u32, 0_u32);
    let (mut deepest, mut at) = (0.0_f64, lo);
    let mut nearest_fill = f64::INFINITY;
    for i in 1..=samples {
        let p = lo
            + Vector3::new(
                radical_inverse(i, 2) * size.x,
                radical_inverse(i, 3) * size.y,
                radical_inverse(i, 5) * size.z,
            );
        let reads = body_field.distance(p);
        if reads >= 0.0 {
            continue;
        }
        inside += 1;
        material += u32::from(part.distance(p) < 0.0);
        if reads < deepest {
            deepest = reads;
            at = p;
        }
        if reads < -wall {
            nearest_fill = nearest_fill.min(fill_field.distance(p));
        }
    }
    if inside == 0 {
        return None;
    }

    let mut step = spacing;
    while step > size.max() * 1e-5 {
        let better = neighbours(at, step).into_iter().map(|p| (body_field.distance(p), p)).fold(
            (deepest, at),
            |best, candidate| if candidate.0 < best.0 { candidate } else { best },
        );
        if better.0 < deepest {
            (deepest, at) = better;
        } else {
            step *= 0.5;
        }
    }

    Some(InfillCavity {
        deepest: -deepest,
        // A sample misses a point of the fill by at most about its spacing, and the field cannot
        // have risen by more than its rate over that, so past this the fill really is absent.
        fill_lands: !nearest_fill.is_finite()
            || nearest_fill <= fill_field.lipschitz() * spacing * 1.5,
        solid_share: f64::from(material) / f64::from(inside),
    })
}

/// The six axes and the eight corners about `at`, which is enough to walk a crease.
fn neighbours(at: Point3<f64>, step: f64) -> [Point3<f64>; 14] {
    let d = |x: f64, y: f64, z: f64| at + Vector3::new(x, y, z) * step;
    [
        d(1.0, 0.0, 0.0),
        d(-1.0, 0.0, 0.0),
        d(0.0, 1.0, 0.0),
        d(0.0, -1.0, 0.0),
        d(0.0, 0.0, 1.0),
        d(0.0, 0.0, -1.0),
        d(1.0, 1.0, 1.0),
        d(1.0, 1.0, -1.0),
        d(1.0, -1.0, 1.0),
        d(1.0, -1.0, -1.0),
        d(-1.0, 1.0, 1.0),
        d(-1.0, 1.0, -1.0),
        d(-1.0, -1.0, 1.0),
        d(-1.0, -1.0, -1.0),
    ]
}

/// `i` with its digits in `base` mirrored about the point, which is one axis of a Halton sequence.
fn radical_inverse(mut i: u32, base: u32) -> f64 {
    let (mut value, mut unit) = (0.0, 1.0 / f64::from(base));
    while i > 0 {
        value += f64::from(i % base) * unit;
        i /= base;
        unit /= f64::from(base);
    }
    value
}
