// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A pattern's fold: how many copies either side a point must look at, and where it counts from.

use super::graph_extent;
use crate::{Extent, PatternLayout, PatternShape};
use fieldform_scene::{Document, Graph, Node, NodeId};
use nalgebra::{Point3, Rotation3, Vector3};

/// How many instances either side of a point's own cell a pattern has to look at.
///
/// **Public because the shader has to agree with the field about it.** The two paths reach the
/// same surface by different routes and `gpu_matches_cpu` checks that they do; a window computed
/// twice is two chances to compute it differently, and the symptom would be material missing
/// from a viewport that the field says is there. So there is one answer and both sides ask for
/// it. See `Op::Pattern::reach` for what the number means.
pub fn pattern_reach(
    document: &Document,
    graph: &Graph,
    input: NodeId,
    kind: &fieldform_scene::PatternKind,
    count: u32,
    radius: f64,
) -> usize {
    // A bare kind and count: nothing left out, the body at the first copy. What every caller
    // asked of this before a pattern had options, so none of them has to change to keep asking.
    let count = count.clamp(1, 512);
    match PatternLayout::plain(document, graph, kind, count) {
        Ok(layout) => pattern_reach_in(document, graph, input, &layout, radius),
        Err(_) => count as usize,
    }
}

/// [`pattern_reach`] for a pattern's own layout, which is what knows how far apart its copies
/// really are: a row spaced by its whole run, or a ring that stops short of a turn, has a
/// different pitch from `step` and `360/n`, and a window sized for the wrong one is a seam that
/// moves as the point crosses a cell.
pub fn pattern_reach_in(
    document: &Document,
    graph: &Graph,
    input: NodeId,
    layout: &PatternLayout,
    radius: f64,
) -> usize {
    let count = layout.total.clamp(1, 512) as usize;
    // **Grown by the fillet's radius, because that is how far a rounded join reads.** A fillet
    // reads both its operands out to its radius -- that is what makes it a fillet -- so a copy
    // within `radius` of the point contributes even though its own surface does not reach it.
    // A window sized for the hard union would round against a subset of the copies that are
    // really there, and the seam would move as the point crossed a cell boundary.
    let extent = graph_extent(document, graph, input).grown(radius.max(0.0));
    match &layout.shape {
        PatternShape::Row { step, .. } => {
            let pitch = step.norm();
            let unit = step.unscale(pitch.max(1e-12));
            cells_reached(extent, |lo, hi| span_along(lo, hi, unit), pitch, count)
        }
        PatternShape::Ring { at, axis, step, .. } => cells_reached(
            extent,
            |lo, hi| angle_about(lo, hi, *at, *axis),
            *step,
            count,
        ),
        // The wider of the two directions: a caller that wants one number gets a safe one, and the
        // fold itself asks for both.
        PatternShape::Grid { .. } => {
            let [columns, rows] = pattern_reach_axes_in(document, graph, input, layout, radius);
            columns.max(rows)
        }
    }
}

/// How many cells either side a pattern's window reaches, **in each of its two directions**.
///
/// A row and a ring have one; a grid has a window along each step, and the product of the two is
/// what is walked. Measured in steps along each, which for steps at an angle is read off the dual
/// vectors, not off the steps themselves.
pub fn pattern_reach_axes_in(
    document: &Document,
    graph: &Graph,
    input: NodeId,
    layout: &PatternLayout,
    radius: f64,
) -> [usize; 2] {
    let PatternShape::Grid { step, count, step2, count2, .. } = &layout.shape else {
        let reach = pattern_reach_in(document, graph, input, layout, radius);
        return [reach, reach];
    };
    let (columns, rows) = ((*count).clamp(1, 512) as usize, (*count2).clamp(1, 512) as usize);
    let Some(dual) = crate::pattern_layout::grid_dual(step, step2) else {
        return [columns, rows];
    };
    // Grown by the fillet's radius for the reason the row is: a rounded join reads that far.
    let extent = graph_extent(document, graph, input).grown(radius.max(0.0));
    let Extent::Bounded { lo, hi } = extent else {
        return [columns, rows];
    };
    // **How far, in steps, the copy nearest a point can be from the cell the point is read into.**
    // A point is at most `ρ` from some copy's centre, where `ρ` is half the longer diagonal of a
    // cell — the covering radius of the lattice, which for an oblique grid is more than half a step.
    // The copy whose *surface* is nearest can stand `2S` farther still, `S` being how far the body
    // reaches. And a distance `x` is at most `x·|d|` steps along a direction whose dual is `d`, so
    // each direction's window is that many cells either side. A row's rule says the same thing for
    // one direction, where the covering radius is half a step.
    let rho = (step + step2).norm().max((step - step2).norm()) * 0.5;
    let axis = |d: Vector3<f64>, n: usize| -> usize {
        let cells = (rho * d.norm() + 2.0 * span_along(lo, hi, d)).ceil();
        if cells.is_finite() { (cells as usize).max(1).min(n) } else { n }
    };
    [axis(dual[0], columns), axis(dual[1], rows)]
}

/// Where copy 0 of a grid stands: the body's middle with the steps to it added.
///
/// The grid's counterpart of [`pattern_origin_in`], which is a coordinate along one direction. A
/// grid is read from a *point* instead, because the nearest copy to a point outside it is found by
/// minimising a distance over the box of copies, which needs the copies' positions and not only
/// their coordinates along each step.
pub fn pattern_grid_anchor_in(
    document: &Document,
    graph: &Graph,
    input: NodeId,
    layout: &PatternLayout,
) -> Point3<f64> {
    let PatternShape::Grid { step, bias, step2, bias2, .. } = &layout.shape else {
        return Point3::origin();
    };
    let Extent::Bounded { lo, hi } = graph_extent(document, graph, input) else {
        return Point3::origin();
    };
    nalgebra::center(&lo, &hi) + step * *bias + step2 * *bias2
}

/// Where the patterned body itself sits along its fold, in the fold's own units, **plus the
/// steps from the body to copy 0**, so a centred pattern is counted from its first copy.
///
/// **Public for the same reason [`pattern_reach`] is: the shader has to agree.** The fold finds
/// a point's instance by counting steps, and it has to count them from the body — instance `i`
/// is the body moved `i` steps, so a point is covered by the instance `i` steps along from where
/// the body is, not from where the world origin is. Both paths fold, so both need this number,
/// and two derivations of it would be two chances to place the window in different places.
///
/// Radians about the axis for a turn, millimetres along the step for a row. Zero when the body's
/// extent cannot be read: [`pattern_reach`] returns the whole count in that case, so every
/// instance is tried and where the window is centred stops mattering.
///
/// The body's *middle* is what is reported, and the reach is its full spread in cells, so the
/// window covers the body wherever within its own box the nearest material happens to be.
///
/// Folding the bias in here is what lets a centred row or ring reuse the fold unchanged: copy `i`
/// is the body moved `i + bias` steps, so a point is nearest copy `(x − body)/pitch − bias`, which
/// is `(x − (body + bias·pitch))/pitch`.
pub fn pattern_origin_in(
    document: &Document,
    graph: &Graph,
    input: NodeId,
    layout: &PatternLayout,
) -> f64 {
    let Extent::Bounded { lo, hi } = graph_extent(document, graph, input) else {
        return 0.0;
    };
    let middle = nalgebra::center(&lo, &hi);
    match &layout.shape {
        // A grid has two origins, in steps; see [`pattern_grid_origin_in`].
        PatternShape::Grid { .. } => 0.0,
        PatternShape::Row { step, bias, .. } => {
            let pitch = step.norm();
            if pitch <= 1e-12 {
                return 0.0;
            }
            middle.coords.dot(&step.unscale(pitch)) + bias * pitch
        }
        PatternShape::Ring { at, axis, step, bias, .. } => {
            let local = axis.inverse_transform_vector(&(middle - *at));
            // On the axis there is no angle to speak of, and no need for one: a body the fold
            // cannot separate from its neighbours has already been given the whole count.
            if local.x.hypot(local.y) <= 1e-9 {
                return 0.0;
            }
            local.y.atan2(local.x) + bias * step
        }
    }
}

/// How many cells of a pattern one body can reach into, either side of its own.
///
/// **A bound, deliberately loose, because the cost of being generous is one more evaluation and
/// the cost of being tight is a wrong answer.** The point's own cell holds the nearest instance
/// unless the body spills out of it, so the window is that spill measured in cells, plus one for
/// the half cell the point can sit off centre.
///
/// An unbounded or empty input gives up and returns the whole count: something that reaches
/// everywhere is near everything, and there is nothing to fold.
fn cells_reached(
    extent: Extent,
    spread: impl Fn(Point3<f64>, Point3<f64>) -> f64,
    pitch: f64,
    count: usize,
) -> usize {
    let Extent::Bounded { lo, hi } = extent else {
        return count;
    };
    if pitch <= 1e-12 {
        return count;
    }
    let reach = spread(lo, hi);
    if !reach.is_finite() {
        return count;
    }
    ((reach / pitch).ceil() as usize + 1).min(count)
}

/// How far a box spans along a direction.
fn span_along(lo: Point3<f64>, hi: Point3<f64>, unit: Vector3<f64>) -> f64 {
    let half = (hi - lo) * 0.5;
    (half.x * unit.x).abs() + (half.y * unit.y).abs() + (half.z * unit.z).abs()
}

/// How wide a box looks from an axis, in radians.
///
/// A box that touches the axis subtends everything, so that is what it says — and
/// [`cells_reached`] turns an infinity into "try them all", which is the honest answer for a
/// body the fold cannot separate from its neighbours.
fn angle_about(
    lo: Point3<f64>,
    hi: Point3<f64>,
    at: Point3<f64>,
    aim: Rotation3<f64>,
) -> f64 {
    let mut nearest = f64::INFINITY;
    let mut furthest: f64 = 0.0;
    for i in 0..8 {
        let corner = Point3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let local = aim.inverse_transform_vector(&(corner - at));
        let r = local.x.hypot(local.y);
        nearest = nearest.min(r);
        furthest = furthest.max(r);
    }
    if nearest <= 1e-9 {
        return f64::INFINITY;
    }
    // The whole box seen from the axis, bounded by the chord across it at the nearest radius.
    let across = (hi - lo).norm();
    2.0 * (across / (2.0 * nearest)).min(1.0).asin()
}

/// A pattern's fillet, which both its divisor and its extent read.
pub(crate) fn pattern_fillet(node: &Node) -> &fieldform_scene::Fillet {
    match node {
        Node::Pattern { fillet, .. } => fillet,
        _ => unreachable!("asked of something that is not a pattern"),
    }
}
