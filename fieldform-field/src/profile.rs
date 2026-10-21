// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A closed planar curve as a signed distance in its own plane.
//!
//! # The one piece extrude and revolve share
//!
//! Both are the same 2D field read differently. An extrude asks it in the plane and thickens
//! along the normal; a revolve asks it in the half-plane `(r, z)` and the answer is already
//! rotationally symmetric. So the work is one signed distance to a closed polyline, and the two
//! blocks are a `max` and a change of coordinates on top of it — which is why they are built
//! together rather than one after the other (D9).
//!
//! # Why the profile has to be closed
//!
//! A [`crate::ScalarField`] returns a **sign**, and a sign needs an inside. An open curve has no
//! inside, so it cannot be extruded into a solid — the same rule that stops an open mesh
//! becoming a field, one dimension down. An open profile is not useless; it wants a thickness,
//! and becomes a rib rather than a solid. That is a separate operation and is not this one.
//!
//! # The fill rule, said out loud
//!
//! A closed curve that crosses itself has no inside until somebody picks a rule. This uses
//! **even–odd**: a point is inside when a ray from it crosses the boundary an odd number of
//! times. So a figure-eight has two solid lobes and a doubly-wound circle is hollow in the
//! middle — which is the convention SVG and most drawing programs use, and the one people
//! predict. The alternative, non-zero winding, fills the middle of both. Neither is more
//! correct; what matters is that it is chosen rather than emergent.

use nalgebra::{Point2, Point3, Vector2, Vector3};

use crate::{Normalization, ScalarField};

/// A closed polyline in a plane, as a signed distance.
///
/// Built from a curve's tessellation rather than from its control points: the distance wanted is
/// to the *curve*, and a NURBS control polygon is not the curve. How finely it was tessellated
/// is the accuracy of everything downstream, which is why it is the caller's decision.
#[derive(Debug, Clone)]
pub struct Profile {
    /// The boundary, without repeating the first point at the end — the closing segment is
    /// implied, so a caller cannot half-close it.
    points: Vec<Point2<f64>>,
    /// Where each corner goes per millimetre of outward offset, when a draft mitres rather
    /// than rounds. Computed with the points and never separately; see [`Profile::miters`].
    miters: Vec<Vector2<f64>>,
    /// The same distance on a grid, when one has been asked for.
    ///
    /// **Here rather than in a type of its own**, so that a profile can be baked without every
    /// consumer learning a new name for it: an extrude, a revolve and a leaning sweep all keep
    /// holding a `Profile` and keep calling `distance`, and what changed is how it answers. The
    /// boundary is kept beside it because the shader still has to be handed the polyline, and
    /// because a bake is a display convenience that export undoes by simply not asking for one.
    baked: Option<crate::BakedProfile>,
}

/// How close two curve ends must be to count as meeting, in millimetres.
///
/// Deliberately tiny. Snapping is what brings two ends together — a drag with the modifier held
/// copies one position onto the other exactly — so this only has to survive arithmetic, not
/// stand in for aim. A generous figure here would silently join curves somebody meant to leave
/// apart, and the failure would be a solid where they wanted two open paths.
const JOINED_MM: f64 = fieldform_scene::JOINED_MM;

/// Why a set of curves did not make a closed loop, said in terms of the gap rather than the rule.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LoopError {
    #[error("curve {0:?} is not in the document")]
    MissingCurve(fieldform_scene::CurveId),
    #[error("that curve has no length to build a profile from")]
    Empty,
    #[error(
        "these curves do not close: there is a {gap:.3} mm gap at ({x:.2}, {y:.2}, {z:.2}). \
         Drag one end onto the other with Ctrl held to join them exactly."
    )]
    NotClosed { gap: f64, x: f64, y: f64, z: f64 },
}

/// What a fillet of `radius_mm` actually becomes against a body leaning `lean_degrees`.
///
/// Returns how far it spreads across the *other* body, and the tightest and flattest radius it
/// turns through. All three are `radius_mm` when there is no lean, which is what a fillet of
/// that radius means.
///
/// **Closed form, because the blend is.** `Node::Blend` equals the other operand exactly beyond
/// `k` — for `a >= k` the formula collapses to `b` — so the join is tangent where the *reading*
/// reaches `k`, and a body reading `s·a` is tangent where `a = k/s`. Inside the arc the surface
/// is `(k − s·x)² + (k − y)² = k²`: an ellipse with semi-axes `k/s` and `k`, whose extreme radii
/// of curvature are `k·s` at the end of the long axis and `k/s²` at the end of the short one.
///
/// **So a leaning operand does not scale a fillet, it stretches one.** That distinction is the
/// whole of why this is worth reporting. A fillet's *footprint* varying is ordinary — a rolling
/// ball between faces at a shallower angle covers more of them, and every CAD tool does that.
/// A fillet's *radius* varying is not: a constant-radius fillet is a ball, and this is an
/// ellipsoid. And the tight end is **smaller than the radius asked for** (`k·s`), which is the
/// direction that matters to anyone who typed a number for a reason — a minimum bend radius, a
/// tool that has to reach in, a stress concentration.
///
/// Measured against this by `examples/blend_over_a_bound.rs`, which walks the real `smooth_min`
/// rather than this arithmetic, and whose zero-lean row is the check on both.
pub fn fillet_under_lean(radius_mm: f64, lean_degrees: f64) -> (f64, f64, f64) {
    let s = lean_shortfall(lean_degrees).clamp(1e-3, 1.0);
    (radius_mm / s, radius_mm * s, radius_mm / (s * s))
}

/// How short a leaning sweep's distance can read, as a fraction of the truth.
///
/// **Measured, not derived.** The correction a shear needs is `|d·n|`, and it is tempting to
/// report that as the error — but it is not the whole of it: the profile's distance and the
/// slab's stop being perpendicular, so the `max` that combines them loses more on top. The
/// numbers here are what `fieldform-field/examples/lean_error.rs` measures, comparing the field
/// against the solid's own surface enumerated in closed form:
///
/// | lean | cos(lean) | reading / truth |
/// |---|---|---|
/// | 0° | 1.000 | 1.000 |
/// | 10° | 0.985 | 0.654 |
/// | 20° | 0.940 | 0.576 |
/// | 30° | 0.866 | 0.505 |
/// | 45° | 0.707 | 0.387 |
/// | 60° | 0.500 | 0.268 |
/// | 75° | 0.259 | 0.150 |
///
/// Straight through zero is exact and the drop after it is abrupt, which is the honest shape of
/// it: leaning at all costs the corner, and leaning further costs the shear on top.
///
/// A worst case over a sampled region, so it is a floor on the quality rather than a proof of
/// it. Reported as "as little as" for that reason, and interpolated between the rows measured.
///
/// **This is about the field, never about the size.** The solid is exactly as deep as it was
/// told to be at every lean, which `lean_error.rs` now measures beside the ratio -- because the
/// panel used to quote this percentage next to a depth, where it was read as a percentage *of*
/// the depth, and that reading is wrong.
///
/// **Re-measured 2026-09-06.** The example had been judging against a solid centred on its
/// plane, which an extrude stopped being when it started at the plane instead. Zero lean is the
/// row that caught it: the field is exact there by construction, so anything but 1.000 is the
/// reference reporting on itself -- and it had been printing 0.371. The other rows moved by
/// about a point, which is why nothing had looked wrong. A broken check that happens to agree
/// is the worst kind, so the case whose answer is known is now a row in the table rather than a
/// constant written in beside it.
pub fn lean_shortfall(lean_degrees: f64) -> f64 {
    const MEASURED: [(f64, f64); 7] = [
        (0.0, 1.0),
        (10.0, 0.654),
        (20.0, 0.576),
        (30.0, 0.505),
        (45.0, 0.387),
        (60.0, 0.268),
        (75.0, 0.150),
    ];
    let lean = lean_degrees.abs();
    if lean <= 0.0 {
        return 1.0;
    }
    for pair in MEASURED.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        if lean <= x1 {
            let t = (lean - x0) / (x1 - x0);
            return y0 + (y1 - y0) * t;
        }
    }
    // Past the last row it keeps falling; edge-on there is no solid at all, and the compile
    // refuses before it gets there.
    MEASURED[MEASURED.len() - 1].1 * (90.0 - lean).max(0.0) / 15.0
}

/// An axis a profile can actually be revolved about: in its own plane, with the profile beside it.
///
/// **A new revolve used to get the world Z axis through the world origin**, which is a fine
/// answer only for a profile drawn around the world origin. Anything drawn anywhere else
/// straddles it, and the block was refused the moment it was made -- "this one spans 101.4°
/// about it" -- for a choice nobody had made. Same shape of mistake as the extrude's frame: a
/// default that describes the world rather than the geometry it is about to be used on.
///
/// The axis lies *in* the profile's plane, because an axis out of it revolves the profile
/// through itself; it points as near to upright as that plane allows, because a lathe turns
/// about something vertical and a profile lying flat has no preference to honour; and it is slid
/// sideways until the whole profile is on one side, touching at the nearest point. Touching is
/// deliberate -- it is the ordinary lathe case, and it makes a solid with no hole down the
/// middle. Points on the axis constrain nothing and are skipped by [`Profile::about_axis`], so
/// resting against it is safe rather than marginal.
///
/// Still only a default. The axis is an input on the block and is meant to be changed; what this
/// fixes is a block that could not be built at all before it was.
pub fn axis_beside(points: &[Point3<f64>]) -> Option<(Point3<f64>, Vector3<f64>)> {
    let (origin, normal) = plane_of(points)?;
    let project = |v: Vector3<f64>| v - normal * v.dot(&normal);
    let dir = project(Vector3::z())
        .try_normalize(1e-9)
        .or_else(|| project(Vector3::x()).try_normalize(1e-9))?;
    let across = normal.cross(&dir);
    let least = points
        .iter()
        .map(|p| (p - origin).dot(&across))
        .fold(f64::INFINITY, f64::min);
    least.is_finite().then(|| (origin + across * least, dir))
}

/// Which curve to build a profile from, given the one the person had selected.
///
/// **A seed has to be a curve that actually closes, and the selected one need not be.** Drawing
/// a curve selects it, so drawing a stray line beside a finished loop and reaching for Extrude
/// asked the walk to start where there was nothing to join. It refused, naming a gap that was
/// only the new line's own length, while a perfectly good loop sat untouched beside it — and it
/// did that for any third curve, whether or not it touched anything.
///
/// Candidates are tried in the order a person would look: the preferred curve, then the rest
/// oldest-first, which is the order the block list shows them in. Falling back past an explicit
/// selection is safe because the caller says which curve it took; silently refusing while a
/// loop exists is what was not.
///
/// The refusal handed back is the *first* candidate's, so when nothing closes the gap named is
/// the one under the curve they were looking at rather than a distance measured elsewhere.
pub fn profile_seed(
    document: &fieldform_scene::Document,
    preferred: Option<fieldform_scene::CurveId>,
) -> Result<fieldform_scene::CurveId, LoopError> {
    let mut candidates: Vec<fieldform_scene::CurveId> = preferred
        .filter(|id| document.curves.contains_key(id))
        .into_iter()
        .collect();
    let mut rest: Vec<fieldform_scene::CurveId> = document
        .curves
        .keys()
        .copied()
        .filter(|id| !candidates.contains(id))
        .collect();
    // By time, falling back to the id so that curves a file never stamped still come out in a
    // stable order rather than a different one each run — the same key the block list sorts by.
    rest.sort_by_key(|id| (document.curves_created.get(id).copied().unwrap_or(0), id.raw()));
    candidates.extend(rest);

    let mut refusal = None;
    for id in candidates {
        match profile_loop(document, id, None) {
            Ok(_) => return Ok(id),
            Err(e) => refusal = refusal.or(Some(e)),
        }
    }
    Err(refusal.unwrap_or(LoopError::Empty))
}

/// The plane a set of points lies in: a point on it, and its normal.
///
/// **What an extrude has to be told, and what it was guessing.** A new extrude used to be given
/// the world XY plane, so a profile drawn in a back view — which is the whole reason for having
/// a drawing plane — was not flat in the plane it was being read in, and was refused as
/// non-planar. The profile's own points are the only honest answer to which plane it is in.
///
/// Newell's method for the normal, which is the sum of the cross products around the loop: it
/// uses every point rather than three of them, so a nearly-straight run at one corner cannot
/// decide the answer, and it degrades to a zero vector for a genuinely straight line rather than
/// producing something confident and wrong.
pub fn plane_of(points: &[Point3<f64>]) -> Option<(Point3<f64>, Vector3<f64>)> {
    if points.len() < 3 {
        return None;
    }
    let mut centroid = Vector3::zeros();
    for p in points {
        centroid += p.coords;
    }
    let centroid = Point3::from(centroid / points.len() as f64);

    let mut normal: Vector3<f64> = Vector3::zeros();
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        normal.x += (a.y - b.y) * (a.z + b.z);
        normal.y += (a.z - b.z) * (a.x + b.x);
        normal.z += (a.x - b.x) * (a.y + b.y);
    }
    let normal = normal.try_normalize(1e-12)?;
    // **The sign is settled here rather than left to the winding.** Newell's method takes its
    // direction from the order the points are in, which is a fact about how somebody happened to
    // draw the curve and not about where the plane is. That was invisible while an extrude
    // straddled its plane; now that one runs *from* it, the winding would decide which way the
    // solid grows -- so the same shape drawn the other way round would extrude backwards, with
    // nothing on screen to say why.
    //
    // Pinned to the largest component, which is stable under small movements of the points in a
    // way that "positive dot with +Z" is not: a plane nearly edge-on to Z would flip back and
    // forth as it was nudged. A profile on the ground extrudes up, one on YZ extrudes along +X,
    // and a curve redrawn the other way round behaves the same as before.
    let biggest = if normal.x.abs() >= normal.y.abs() && normal.x.abs() >= normal.z.abs() {
        normal.x
    } else if normal.y.abs() >= normal.z.abs() {
        normal.y
    } else {
        normal.z
    };
    Some((centroid, if biggest < 0.0 { -normal } else { normal }))
}

pub struct ProfileLoop {
    /// The closed polyline, in world coordinates, with no repeated point at the seam.
    pub points: Vec<Point3<f64>>,
    /// Which curves it is made of, in the order the walk found them.
    ///
    /// Reported because "did it pick up the other one?" is the first question anybody asks of a
    /// profile built from several curves, and counting the answer out of the points is not
    /// something a panel should be doing.
    pub curves: Vec<fieldform_scene::CurveId>,
}

/// How far a profile's polyline may turn at a joint, in degrees.
///
/// The pipe spine's number, for the pipe spine's reason: shading is the normal, and the normal
/// turns a corner at every joint, so what a person sees is the size of that corner. One degree
/// is where they stop being visible — measured for the spine in `Curve::tessellate_to_angle`,
/// and the geometry of the two cases is the same.
pub const PROFILE_TURN_DEGREES: f64 = 1.0;

/// The world-space polyline of the closed loop that `seed` is part of.
///
/// # Why a loop is walked rather than stored
///
/// A profile is often several curves — a line, an arc, another line — and the obvious design is
/// for the extrude to hold a list of them. That list is then a second description of something
/// the geometry already says, and the two part company the moment a curve moves: a stored list
/// still naming a curve whose end has wandered off is a profile that is closed according to the
/// document and open according to the points.
///
/// So nothing is stored. The node names *one* curve, any curve of the loop, and the loop is
/// found by walking outward from it through ends that meet. There is one description — where the
/// curves actually are — and it cannot go stale.
///
/// A single closed curve is the degenerate case and comes back unchanged, which is what keeps
/// this compatible with every profile that already works.
///
/// # What it refuses, and why it is refused rather than guessed
///
/// **A gap**, reported with its size and where it is, because "not closed" is a rule and
/// "0.42 mm apart at (12, 30, 0)" is something you can go and fix. **A branch** — three or more
/// ends at one point — was refused too, as more than one way round; now every way round is tried
/// and the first that closes wins, and `close_the_loop` says why.
///
/// Two ends meet within `fieldform_scene::JOINED_MM`, which is a fact about the geometry rather
/// than about how finely the curves are sampled, so a loop that closes at draft quality closes
/// at final too.
pub fn profile_loop(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    ignore: Option<fieldform_scene::CurveId>,
) -> Result<ProfileLoop, LoopError> {
    profile_loop_turning(document, seed, PROFILE_TURN_DEGREES, ignore)
}

/// The same, saying how much the polyline may turn at a joint.
///
/// **Separate because the two callers want different things and one of them is per-pixel.** Off
/// the hot path -- fitting a plane, deciding whether a loop closes, compiling a field to measure
/// against -- the count does not matter and the finest sensible answer is the right one. In the
/// viewport it is the whole cost: the shader scans every point of this polyline at every sample,
/// and a pattern multiplies that by its count.
///
/// Measured on a closed cubic ring by `examples/profile_points.rs`, with the frame cost from
/// `fieldform-gpu/examples/traced_cost.rs` beside it:
///
/// | max turn | points | one revolve at 720 px |
/// |---|---|---|
/// | 0.5° | 865 | ~150 ms |
/// | 1° | 433 | ~75 ms |
/// | 2° | 217 | ~37 ms |
/// | 5° | 109 | ~18 ms |
/// | 10° | 55 | 9 ms |
///
/// So it is a real trade rather than a setting with a right answer, and the viewport puts it on
/// Display resolution beside the cage cells and the pipe spine. Ten degrees is visibly banded --
/// the spine's own notes record 6.35° as bands across a tube -- and one degree is where they stop
/// being visible and where a pattern of six stops being turnable. Making both true at once wants
/// the profile scan to stop being linear; that is backlog 33.
pub fn profile_loop_turning(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    turn_degrees: f64,
    ignore: Option<fieldform_scene::CurveId>,
) -> Result<ProfileLoop, LoopError> {
    // **Not the sampling tolerance.** Two ends either meet or they do not, and that is a fact
    // about the geometry rather than about how finely it is being evaluated — so a profile must
    // not close at draft quality and open at final, which is what deriving this from a sampling
    // tolerance produced. `JOINED_MM`, which the walk reads: a tenth of a micron, far below
    // anything anybody can see, wide enough to absorb float noise, and reachable only by
    // snapping two ends together on purpose.
    //
    // **How finely the curves are sampled, which is not how close two ends must be.** These were
    // the same number, and that number was the join tolerance -- a tenth of a micron. It is the
    // right answer for "do these two ends meet", and it was never the right one for "how many
    // points describe this curve": it put **1025 points** on a 200 mm degree-two curve where
    // seventeen would be indistinguishable, measured by `examples/profile_points.rs`.
    //
    // The shader walks every one of those points at every sample, and a pattern multiplies that
    // by its count -- so six copies of a revolve carried six thousand segments per pixel and the
    // viewport could not be turned. Nothing needed it: the *ends* of a tessellation are the
    // curve's exact ends at any tolerance, which is what the join test compares, so sampling and
    // joining were only ever coupled by sharing a variable.
    //
    // An angle rather than a distance, and the same angle the pipe spine settled on for the same
    // reason: what shows on a silhouette is the corner the surface turns at each joint, and a
    // chord tolerance says nothing about that on its own. See `Curve::tessellate_to_angle`.
    //
    // There was a `tolerance_mm` beside the angle for a while, meant as the floor, and it was the
    // second half of the same mistake: a distance coarser than the refinement's own starting
    // point stopped it before it began, so the angle was never approached and the setting moved
    // nothing. It has been removed rather than left to be trusted -- every caller was passing a
    // number that changed no part of the answer. The floor is now a hundred-thousandth of the
    // curve's own length, which is a safety stop and not a quality knob.
    let sampled = |c: &fieldform_scene::Curve| {
        // **A safety stop, not a quality knob**, and passing the display tolerance here was a
        // second mistake on top of the first. `tessellate_to_angle` starts at a thousandth of
        // the curve's length and quarters until the turn is small enough *or* the floor is
        // reached -- so a floor coarser than that start makes it break before refining once,
        // and the angle asked for is never approached. The measurement said so and was misread:
        // the point count came out identical at 0.1, 0.5 and 1 mm, which is what a floor looks
        // like, not what a criterion looks like.
        //
        // A hundred-thousandth of the length is small enough never to bind on a smooth curve
        // and still stop the refinement dead at a cusp, where no amount of subdivision reduces
        // the turn. What governs is the angle, which is the thing that shows.
        //
        // In a function of its own so that a wall's chain is sampled by this rule and no other.
        sampled_to(c, turn_degrees)
    };
    // **Only the loop's own curves are tessellated.** The search is over every open curve's
    // ends, which [`loop_from`] answers from the ends alone in a tenth of a millisecond;
    // sampling every candidate finely first cost the owner's three-section loft nine
    // tessellations per section -- 48 ms a build, paid several times a rebuild, measured by
    // `fieldform-gpu/examples/document_cost.rs` (2026-10-04). A curve the ends-walk did not
    // place in the loop is offered to the same search as its two ends, which can only fail to
    // join where the full polyline would have failed too, since joining reads the ends.
    let members = loop_from(document, seed, ignore)?;
    let ends = |c: &fieldform_scene::Curve| {
        let (start, end) = c.ends();
        vec![start, end]
    };
    let member_or_ends = |c: &fieldform_scene::Curve| {
        if members.contains(&c.id) { sampled(c) } else { ends(c) }
    };
    let (points, curves) = walk(document, seed, ignore, &member_or_ends)?;
    Ok(ProfileLoop { points, curves })
}

/// The curves of the loop `seed` is part of, in walk order, found from the curves' ends alone.
///
/// **The same search as [`profile_loop`], over the same points it tests.** A piece meets the
/// chain by its ends and by nothing else, so a walk that samples only the ends finds exactly
/// the loop the polyline walk finds, in the same order and the same orientation -- and
/// tessellates nothing to do it. Only a caller that wants the polyline needs the points
/// between, and the block list, the row's "can be extruded" line and the console's gap report
/// do not.
///
/// Measured on `plane-test.fform` -- seven open curves, three loops -- by
/// `fieldform-io/examples/profile_cost.rs`, which is what turned "very slow" into a number:
///
/// | | ms |
/// |---|---|
/// | tessellate every open curve once, at 1° | 20.4 |
/// | `profile_partition` walking polylines | 86.8 |
/// | `profile_loop` from every open curve | 138.5 |
/// | `profile_partition` walking ends | 0.1 |
/// | `loop_from` from every open curve | 0.095 |
///
/// The partition re-walks on every frame of a curve drag, and an opened curve row walked
/// per frame to say whether its loop closed, so the polyline numbers were being paid sixty
/// times a second for seven curves. What remains per draft frame is the compile's own walk
/// for each extrude, revolve and sweep that reads the curves, which needs the polyline.
pub fn loop_from(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    ignore: Option<fieldform_scene::CurveId>,
) -> Result<Vec<fieldform_scene::CurveId>, LoopError> {
    let ends = |c: &fieldform_scene::Curve| {
        let (start, end) = c.ends();
        vec![start, end]
    };
    walk(document, seed, ignore, &ends).map(|(_, curves)| curves)
}

/// The search itself: from `seed`, through pieces sampled by `sample`, back to where it began.
///
/// Shared by [`profile_loop_turning`], which samples finely for a polyline, and [`loop_from`],
/// which samples only the ends. One search over pieces joined by one test is what keeps the two
/// from ever disagreeing about which curves make a loop.
fn walk(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    ignore: Option<fieldform_scene::CurveId>,
    sample: &dyn Fn(&fieldform_scene::Curve) -> Vec<Point3<f64>>,
) -> Result<(Vec<Point3<f64>>, Vec<fieldform_scene::CurveId>), LoopError> {
    let start = document
        .curves
        .get(&seed)
        .ok_or(LoopError::MissingCurve(seed))?;

    // A curve that already closes is its own loop, and asking the document about its neighbours
    // could only find ways to make it worse.
    if start.is_closed() {
        return Ok((sample(start), vec![seed]));
    }

    // Every open curve in the document is a candidate. A closed one cannot be part of a chain —
    // it has no free ends to join to — so it is left out rather than skipped later.
    let passed = passed_over(document, seed, ignore);
    let mut pieces: Vec<(fieldform_scene::CurveId, Vec<Point3<f64>>)> = document
        .curves
        .iter()
        // **The curve a block is aimed *along* is not part of what it is extruding.** An axis
        // drawn from the origin shares an end with a profile that also starts there, and the
        // walk would otherwise wander off down it. Which curve to leave out is the caller's to
        // say, because only the caller knows what the curve is being used *for*.
        .filter(|(id, _)| Some(**id) != ignore)
        // And of curves lying exactly on each other, the one this walk is not built on.
        .filter(|(id, _)| !passed.contains(*id))
        .filter(|(_, c)| !c.is_closed())
        .map(|(id, c)| (*id, sample(c)))
        .filter(|(_, points)| points.len() >= 2)
        .collect();
    // Sorted so the search does not depend on a map's iteration order: the same document must
    // give the same loop every time, or a profile would change shape between runs.
    pieces.sort_by_key(|(id, _)| id.raw());

    let seat = pieces
        .iter()
        .position(|(id, _)| *id == seed)
        .ok_or(LoopError::Empty)?;

    let mut chain = pieces[seat].1.clone();
    let mut used = vec![false; pieces.len()];
    used[seat] = true;
    let mut walked = vec![seed];
    // The furthest the search got before running out, for the message when it fails.
    let mut stranded: Option<(f64, Point3<f64>)> = None;

    if close_the_loop(
        &pieces,
        JOINED_MM,
        &mut chain,
        &mut used,
        &mut walked,
        &mut stranded,
    ) {
        // Back where it started: the repeated point goes, because a polyline that ends on its
        // own first point describes a zero-length final segment.
        chain.pop();
        return Ok((chain, walked));
    }

    let (gap, at) = stranded.unwrap_or_else(|| {
        let tail = *chain.last().expect("a piece has at least two points");
        ((tail - chain[0]).norm(), tail)
    });
    Err(LoopError::NotClosed { gap, x: at.x, y: at.y, z: at.z })
}

/// The curves a walk from `seed` passes over, because another curve lies exactly where each of
/// them does and is the one this walk takes.
///
/// # Why curves lie on each other at all
///
/// A paste lands in place, and it copies every curve the blocks it pastes are built on, so after
/// one, every side of the pasted extrude's loop lies exactly on a side of the original's. A curve
/// copied and pasted on its own does the same to the loop it came from. The walk searches every
/// open curve in the document and joins any two ends that meet, so it met each corner with two
/// ways on that were the same way, and took whichever came first by id. Pasting a four-line
/// square's extrude left the original's walk round its own four sides about one time in
/// nineteen; otherwise it went round a mix of its own sides and the copies, or closed with no
/// area, out along a side and back along its copy, which broke the extrude it was copied from.
/// An open chain fared no better: every joint was a fork, so a wall shrank to the one curve it
/// names, or closed out along the chain and back along its copy.
///
/// # Which one it takes
///
/// **One of each set of curves lying on each other.** A loop can only use two of them by going
/// out along one and back along the other, which encloses nothing, so no loop is lost.
///
/// **From every set, the one of the seed's own generation.** Each set is ranked oldest first, by
/// `curves_created` and then the id, the order the block list and [`profile_partition`] use. The
/// seed's rank in its own set is its generation, and the walk takes that rank from every other
/// set, or the youngest where a set has fewer. A copy is always younger than what it was copied
/// from, so the original, the oldest of every set it is in, is offered exactly the curves it was
/// offered before anything was pasted on it and walks the same loop, and a paste is walked
/// through its own copies whatever their ids. Where nothing lies on anything, nothing is passed
/// over and the walk is what it always was.
///
/// **What lies on the curve a walk leaves out is left out with it.** A paste copies the line an
/// aimed block leaves out as well as the block, and the copy is the same line.
fn passed_over(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    ignore: Option<fieldform_scene::CurveId>,
) -> std::collections::HashSet<fieldform_scene::CurveId> {
    let sets = lying_on_each_other(document);
    let generation = sets.iter().find_map(|set| set.iter().position(|id| *id == seed)).unwrap_or(0);
    let mut passed = std::collections::HashSet::new();
    for set in &sets {
        let kept = if set.contains(&seed) {
            Some(seed)
        } else if ignore.is_some_and(|aim| set.contains(&aim)) {
            None
        } else {
            Some(set[generation.min(set.len() - 1)])
        };
        passed.extend(set.iter().copied().filter(|id| Some(*id) != kept));
    }
    passed
}

/// Every set of two or more open curves lying exactly on each other, each ranked oldest first.
fn lying_on_each_other(document: &fieldform_scene::Document) -> Vec<Vec<fieldform_scene::CurveId>> {
    // **Sorted by the lesser x of a curve's two ends, so each is compared only with its few
    // neighbours.** Two curves lying on each other share both ends, either way round, so that x
    // differs between them by no more than the tolerance the ends are compared at. This runs on
    // every walk, and comparing every pair would make the partition's walk-per-curve cubic.
    let mut open: Vec<(f64, fieldform_scene::CurveId, &fieldform_scene::Curve)> = document
        .curves
        .iter()
        .filter(|(_, c)| !c.is_closed() && c.control_points().len() >= 2)
        .map(|(id, c)| {
            let points = c.control_points();
            (points[0].x.min(points[points.len() - 1].x), *id, c)
        })
        .collect();
    open.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut head: Vec<usize> = (0..open.len()).collect();
    fn root(head: &mut [usize], mut i: usize) -> usize {
        while head[i] != i {
            head[i] = head[head[i]];
            i = head[i];
        }
        i
    }
    let mut any = false;
    for i in 0..open.len() {
        for j in i + 1..open.len() {
            if open[j].0 - open[i].0 > JOINED_MM {
                break;
            }
            if lie_on_each_other(open[i].2, open[j].2) {
                let (a, b) = (root(&mut head, i), root(&mut head, j));
                head[a] = b;
                any = true;
            }
        }
    }
    // Nothing lying on anything is the usual document, and it costs the walk only the sort.
    if !any {
        return Vec::new();
    }
    let mut sets: std::collections::HashMap<usize, Vec<fieldform_scene::CurveId>> =
        std::collections::HashMap::new();
    for (i, (_, id, _)) in open.iter().enumerate() {
        sets.entry(root(&mut head, i)).or_default().push(*id);
    }
    let age = |id: &fieldform_scene::CurveId| {
        (document.curves_created.get(id).copied().unwrap_or(0), id.raw())
    };
    sets.into_values()
        .filter(|set| set.len() > 1)
        .map(|mut set| {
            set.sort_by_key(age);
            set
        })
        .collect()
}

/// Whether two curves are the same curve: the same degree, control points within [`JOINED_MM`],
/// weights and knots, in the same order or exactly reversed.
///
/// **The whole curve, not its ends.** Two different arcs between the same two points meet at both
/// ends and are a loop, a lens, and a profile; only a curve lying along the whole of another is
/// the same way round twice. Knots are read across the domain, since where a curve's parameter
/// starts is a choice and where the curve lies is not.
fn lie_on_each_other(a: &fieldform_scene::Curve, b: &fieldform_scene::Curve) -> bool {
    let (pa, pb) = (a.control_points(), b.control_points());
    let (wa, wb) = (a.weights(), b.weights());
    if a.degree() != b.degree()
        || pa.len() != pb.len()
        || wa.len() != wb.len()
        || a.knots().len() != b.knots().len()
    {
        return false;
    }
    let across = |knots: &[f64]| -> Vec<f64> {
        let (low, high) = (knots[0], knots[knots.len() - 1]);
        let span = high - low;
        knots.iter().map(|k| if span > 0.0 { (k - low) / span } else { 0.0 }).collect()
    };
    let (ka, kb) = (across(a.knots()), across(b.knots()));
    let near = |p: &Point3<f64>, q: &Point3<f64>| (p - q).norm() <= JOINED_MM;
    let same = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0);
    let forwards = pa.iter().zip(pb).all(|(p, q)| near(p, q))
        && wa.iter().zip(wb).all(|(x, y)| same(*x, *y))
        && ka.iter().zip(&kb).all(|(x, y)| same(*x, *y));
    let backwards = pa.iter().zip(pb.iter().rev()).all(|(p, q)| near(p, q))
        && wa.iter().zip(wb.iter().rev()).all(|(x, y)| same(*x, *y))
        && ka.iter().zip(kb.iter().rev()).all(|(x, y)| same(*x, 1.0 - *y));
    forwards || backwards
}

/// A curve as the points a profile or a wall is built from, turning no more than `turn_degrees`
/// at a joint. One rule for both, so a wall along a loop and a profile of the same loop are
/// built from the same polyline; [`profile_loop_turning`] says why the floor is what it is.
fn sampled_to(curve: &fieldform_scene::Curve, turn_degrees: f64) -> Vec<Point3<f64>> {
    let floor = (curve.length(0.5) * 1e-5).max(1e-4);
    curve.tessellate_to_angle(turn_degrees, floor)
}

/// A curve and whatever joins it end to end: the thing a wall runs along.
///
/// Closed when the way round comes back to where it started, which is exactly a profile's loop,
/// and open when it does not — which is what a profile refuses and a wall is for.
#[derive(Debug, Clone)]
pub struct Chain {
    /// The polyline, in world coordinates. A closed chain does not repeat its first point.
    pub points: Vec<Point3<f64>>,
    /// Which curves it is made of, in the order they are met along it.
    pub curves: Vec<fieldform_scene::CurveId>,
    pub closed: bool,
}

/// The chain `seed` is part of, at the fineness a profile is walked at.
pub fn outline_chain(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    ignore: Option<fieldform_scene::CurveId>,
) -> Result<Chain, LoopError> {
    outline_chain_turning(document, seed, PROFILE_TURN_DEGREES, ignore)
}

/// The chain `seed` is part of: the loop it closes if it closes one, and otherwise the open run
/// of curves that meet it end to end.
///
/// # Walked, for the reason a loop is
///
/// A wall names *one* curve and the rest is found from where the curves actually are, so there
/// is no stored list to go stale when an end is dragged away. See [`profile_loop`].
///
/// # Which way it runs
///
/// **In the direction of the curve it was asked about.** Left and right of a wall are read
/// walking along it, so the direction has to come from something the person can see and
/// change, and the named curve is the only thing they named. Curves joined before it are put in
/// front and those after it behind, each turned round if it was drawn the other way.
///
/// # Where it stops
///
/// At an end nothing else meets — and at a fork. Where three or more ends share a point there
/// is more than one way on, and a loop's walk settles that by trying each and keeping the one
/// that closes. An open chain has no such test, so it stops there rather than pick: a wall that
/// ends at a junction is something a person can see and extend, and one that wandered off down
/// the wrong branch is not something they asked for.
pub(crate) fn outline_chain_turning(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    turn_degrees: f64,
    ignore: Option<fieldform_scene::CurveId>,
) -> Result<Chain, LoopError> {
    let sample = |c: &fieldform_scene::Curve| sampled_to(c, turn_degrees);
    let start = document.curves.get(&seed).ok_or(LoopError::MissingCurve(seed))?;
    if start.is_closed() {
        return Ok(Chain { points: sample(start), curves: vec![seed], closed: true });
    }
    // The loop first, by the same search a profile makes, so that a wall along a loop and a
    // profile of it can never disagree about which curves the loop is.
    match walk(document, seed, ignore, &sample) {
        Ok((points, curves)) => return Ok(Chain { points, curves, closed: true }),
        Err(LoopError::NotClosed { .. }) => {}
        Err(other) => return Err(other),
    }

    let pieces = open_pieces(document, seed, ignore, &sample);
    let seat = pieces.iter().position(|(id, _)| *id == seed).ok_or(LoopError::Empty)?;
    let mut used = vec![false; pieces.len()];
    used[seat] = true;
    let mut points = pieces[seat].1.clone();
    let mut curves = vec![seed];

    // Onward from the far end, for as long as there is exactly one way on.
    while let Some((i, by_its_start)) =
        the_only_piece_at(&pieces, &used, *points.last().expect("a piece has two points"))
    {
        let mut added = pieces[i].1.clone();
        if !by_its_start {
            added.reverse();
        }
        // The shared point is in both; keeping it twice would leave a segment of no length.
        points.extend(added.into_iter().skip(1));
        used[i] = true;
        curves.push(pieces[i].0);
    }
    // And back from the near end, the same way, with each piece turned to arrive at it.
    while let Some((i, by_its_start)) = the_only_piece_at(&pieces, &used, points[0]) {
        let mut added = pieces[i].1.clone();
        if by_its_start {
            added.reverse();
        }
        added.pop();
        added.extend(points);
        points = added;
        used[i] = true;
        curves.insert(0, pieces[i].0);
    }
    Ok(Chain { points, curves, closed: false })
}

/// Every open curve a chain from `seed` could be made of, sampled, in an order that does not
/// depend on a map's. The same gathering [`walk`] does for a loop, for the same reasons.
fn open_pieces(
    document: &fieldform_scene::Document,
    seed: fieldform_scene::CurveId,
    ignore: Option<fieldform_scene::CurveId>,
    sample: &dyn Fn(&fieldform_scene::Curve) -> Vec<Point3<f64>>,
) -> Vec<(fieldform_scene::CurveId, Vec<Point3<f64>>)> {
    let passed = passed_over(document, seed, ignore);
    let mut pieces: Vec<(fieldform_scene::CurveId, Vec<Point3<f64>>)> = document
        .curves
        .iter()
        .filter(|(id, _)| Some(**id) != ignore)
        .filter(|(id, _)| !passed.contains(*id))
        .filter(|(_, c)| !c.is_closed())
        .map(|(id, c)| (*id, sample(c)))
        .filter(|(_, points)| points.len() >= 2)
        .collect();
    pieces.sort_by_key(|(id, _)| id.raw());
    pieces
}

/// The one unused piece with an end at `at`, and whether that end is its first — or nothing,
/// where no piece meets there or more than one does.
fn the_only_piece_at(
    pieces: &[(fieldform_scene::CurveId, Vec<Point3<f64>>)],
    used: &[bool],
    at: Point3<f64>,
) -> Option<(usize, bool)> {
    let meets = |p: &Point3<f64>| (p - at).norm() <= JOINED_MM;
    let mut found = None;
    for (i, (_, points)) in pieces.iter().enumerate() {
        if used[i] {
            continue;
        }
        let (by_start, by_end) = (meets(&points[0]), meets(points.last().expect("two points")));
        if !by_start && !by_end {
            continue;
        }
        // A second piece here is a fork, and so is one piece with both its ends here.
        if found.is_some() || (by_start && by_end) {
            return None;
        }
        found = Some((i, by_start));
    }
    found
}

/// The plane an open chain lies in, and a point on it: its own where it bends, and a chosen one
/// where it is straight.
///
/// # Why this is not [`plane_of`]
///
/// A closed profile always has a plane to give, because a loop encloses an area — and a wall
/// along a loop takes that one, so that it is laid where the profile of the loop would be. An
/// open chain need not have one: a straight line lies in every plane through it, and a rib
/// along a straight line is the plainest wall there is. So there has to be an answer for it,
/// and it has to be one a person would predict.
///
/// **A straight chain stands upright, unless it has been told where it is going.** With a sweep
/// direction to hand — `toward` — the plane is the one that direction leaves most squarely, so
/// a rib pulled along a typed direction is the rib that was asked for rather than a sheared
/// one. It is also what lets it be built at all: a line on the ground pulled sideways would
/// otherwise be swept along its own plane, which is no solid. With none, the normal is as near to world up as the line
/// allows: a line drawn on the ground is a wall standing on it, which is what drawing one
/// there means. A line that is itself vertical falls back to facing along X.
///
/// # The point is midway between the chain's two ends
///
/// An extrude does not care which point of the plane it is handed. **A sweep does**: it
/// carries the section along its path by this point and turns it about this point where the
/// path bends, so the further the section reaches from it, the wider the far side swings and
/// the more the sweep has to divide by. What it wants is the section's middle.
///
/// A profile's is the centroid of its points ([`plane_of`]), and that will not do here: it
/// moves as the same curve is sampled more finely. That is harmless while one tessellation is
/// used throughout, and a wall is not built from one — the field is compiled at one fineness
/// and the viewport's polyline at another. The two ends of a chain are the same points at
/// every fineness, and so is the point midway between them. It is also the same point
/// whichever way the curve was drawn.
///
/// It was the chain's first point until 2026-10-02, chosen for that same steadiness while only
/// an extrude took a wall. Carried round a bend, a section then swung about whichever end its
/// curve happened to start at. Midway between the ends is the middle of a line, an L, a Z, an
/// S and a half ring. Of a U it is the middle of the open side, which is the one common
/// section a box's centre would serve better — and a box's centre moves with the sampling.
///
/// # "Bends" is how the points spread, not the area they enclose
///
/// Newell's sum over the chain closed by its chord was the first answer, and it is the signed
/// area — which an S with equal lobes has none of. Such a chain was judged straight, given the
/// ground-up plane, and refused as "strays 7.2 mm from its plane" on any drawing plane but the
/// ground, where the default happened to be right. Reported from the application as *"when a
/// curve of degree 2 or more inflects the wall-based extrude tends to break"*;
/// `examples/inflecting_wall_probe.rs` is the table. So bending is how far the points stand
/// off the chord between the ends — the sag, which an S has as much of as a C — judged against
/// the chain's own length so that a line whose points stray by a rounding is not bent by the
/// rounding. The plane of a chain that bends is the points' own best-fit plane, by their
/// spread about their centroid: the direction they do not spread in is the normal. The sag
/// decides and the spread only orients, because a straight line's spread across itself is
/// nothing but the eigensolver's rounding, which is not a thing to compare a threshold to.
pub(crate) fn chain_plane(
    points: &[Point3<f64>],
    toward: Option<Vector3<f64>>,
) -> Option<(Point3<f64>, Vector3<f64>)> {
    let (first, last) = (*points.first()?, *points.last()?);
    let between = nalgebra::center(&first, &last);
    let length: f64 = points.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
    if length <= 0.0 {
        return None;
    }
    // How far the chain stands off its own chord. Ends that meet are a chain that bends.
    let chord = (last - first).try_normalize(1e-12);
    let sag = match chord {
        Some(along) => points
            .iter()
            .map(|p| (p - first).cross(&along).norm())
            .fold(0.0f64, f64::max),
        None => f64::INFINITY,
    };
    if sag > 1e-9 * length {
        // The spread of the points about their centroid, in each direction: its eigenvectors
        // are the directions the chain lies along, across, and does not lie along at all.
        let centroid = points.iter().fold(Vector3::zeros(), |sum, p| sum + p.coords)
            / points.len() as f64;
        let mut spread = nalgebra::Matrix3::<f64>::zeros();
        for p in points {
            let d = p.coords - centroid;
            spread += d * d.transpose();
        }
        let eigen = spread.symmetric_eigen();
        let least = (0..3)
            .min_by(|a, b| eigen.eigenvalues[*a].total_cmp(&eigen.eigenvalues[*b]))
            .expect("three eigenvalues");
        let normal = eigen.eigenvectors.column(least).into_owned().try_normalize(1e-12)?;
        return Some((between, faced(normal)));
    }

    // Straight. Whatever is chosen has to be square to the line itself.
    let along = chord?;
    let square_to_the_line = |v: Vector3<f64>| (v - along * v.dot(&along)).try_normalize(1e-6);
    let stated = toward.and_then(|t| t.try_normalize(1e-12)).and_then(square_to_the_line);
    let normal = match stated {
        // Kept the way it was pointing: the sweep runs out of this side of the plane.
        Some(normal) => normal,
        None => faced(
            square_to_the_line(Vector3::z()).or_else(|| square_to_the_line(Vector3::x()))?,
        ),
    };
    Some((between, normal))
}

/// A normal turned to face the way its largest component is positive — the rule [`plane_of`]
/// settles a profile's by, so that which way a curve was drawn does not decide which way its
/// solid grows.
fn faced(normal: Vector3<f64>) -> Vector3<f64> {
    let biggest = if normal.x.abs() >= normal.y.abs() && normal.x.abs() >= normal.z.abs() {
        normal.x
    } else if normal.y.abs() >= normal.z.abs() {
        normal.y
    } else {
        normal.z
    };
    if biggest < 0.0 { -normal } else { normal }
}

/// The document's open curves, partitioned into the closed loops they make, in block-list order.
///
/// **The same walk an extrude uses, so the row and the solid can never disagree about what the
/// loop is.** That is the whole reason this is a loop over [`profile_loop`] rather than a second
/// loop-finder: a partition that walked differently would show a row for a loop the extrude
/// would refuse, or the other way round, and nothing on screen would say which was right.
/// `profile-group-spec.md` §3.1 is the pseudo-code this follows.
///
/// Each loop's curves come back in walk order, and the first is the seed the walk started from
/// — the loop's earliest curve in block-list order, since any earlier member would have been
/// seeded first and claimed it. That makes the first curve a stable key for the loop while it
/// stays in the loop, which is what the block list needs to keep a row's state across frames.
///
/// **A single closed curve is not a loop here.** It is its own profile and its own row, and a
/// group of one is a disclosure triangle over nothing. **A branch follows `close_the_loop`**:
/// the first loop that closes from a seed wins, and a curve that lost the branch stays its own
/// row rather than breaking the loop it touches. A curve is in at most one loop, because a seed
/// whose loop would reuse a curve already placed is left out rather than placed twice.
///
/// Walks on the curves' ends alone ([`loop_from`]), so it costs no tessellation; callers that
/// read it per frame still cache it behind something that changes when a curve does.
pub fn profile_partition(document: &fieldform_scene::Document) -> Vec<Vec<fieldform_scene::CurveId>> {
    let mut seeds: Vec<fieldform_scene::CurveId> = document
        .curves
        .iter()
        .filter(|(_, c)| !c.is_closed())
        .map(|(id, _)| *id)
        .collect();
    // The block list's own order — by time, then id — so the key of a loop is the curve the
    // list would have shown first.
    seeds.sort_by_key(|id| (document.curves_created.get(id).copied().unwrap_or(0), id.raw()));

    let mut placed: std::collections::HashSet<fieldform_scene::CurveId> =
        std::collections::HashSet::new();
    let mut loops = Vec::new();
    for seed in seeds {
        if placed.contains(&seed) {
            continue;
        }
        // The ends-only walk: the same loop `profile_loop` finds, without a tessellation.
        let Ok(curves) = loop_from(document, seed, None) else { continue };
        if curves.len() < 2 || curves.iter().any(|c| placed.contains(c)) {
            continue;
        }
        // **A loop somebody has dismissed is not gathered into a row.** A profile row is a reading of
        // the geometry, so it cannot be deleted -- it re-forms on the next refresh. `ungrouped_curves`
        // is the stored dismissal that makes "these are separate curves" stick, and it is read here
        // and nowhere else: `profile_loop`, which an extrude walks to find its boundary, does not
        // consult it, so dismissing a row changes the list and no geometry.
        if curves.iter().any(|c| document.ungrouped_curves.contains(c)) {
            continue;
        }
        placed.extend(curves.iter().copied());
        loops.push(curves);
    }
    loops
}

/// Where two ends of a loop meet: the place, and the two ends that meet there.
#[derive(Debug, Clone, PartialEq)]
pub struct Meeting {
    pub at: Point3<f64>,
    /// The two ends as (curve, control-point index): the end the walk arrived on, then the
    /// end it left by.
    pub ends: [(fieldform_scene::CurveId, usize); 2],
}

/// The meetings of a loop where exactly two ends meet, in walk order.
///
/// `curves` is a loop as [`profile_loop`] reports it: each curve meets the next where the walk
/// stepped across, and the last meets the first. The walk's own rule decides which end is
/// which — the first curve runs forward, and every later one is joined by whichever of its
/// ends is at the chain's tail — so the ends named here are the ends the walk actually
/// crossed, not a guess from proximity.
///
/// **A meeting of three or more ends is left out.** A joint there would lock a choice the
/// walk deliberately leaves open (`profile-group-spec.md` §4.2): a stray line drawn from a
/// profile's corner shares that corner, and pinning all three ends to one point would make the
/// stray part of the profile's structure. Every end of every open curve in the document is
/// counted, not only the loop's.
///
/// Ends are the first and last control points, which is where a clamped curve — every curve
/// the tools make — starts and stops. Two ends meet within [`JOINED_MM`], the walk's own test.
pub fn loop_meetings(
    document: &fieldform_scene::Document,
    curves: &[fieldform_scene::CurveId],
) -> Vec<Meeting> {
    let meets = |a: &Point3<f64>, b: &Point3<f64>| (a - b).norm() <= JOINED_MM;
    let ends_of = |id: fieldform_scene::CurveId| {
        let points = document.curves.get(&id)?.control_points();
        Some((points[0], *points.last()?, points.len() - 1))
    };
    let mut out = Vec::new();
    if curves.len() < 2 {
        return out;
    }
    // Every open end in the document, so a branch is seen wherever its third end came from.
    let every_end: Vec<Point3<f64>> = document
        .curves
        .values()
        .filter(|c| !c.is_closed())
        .flat_map(|c| {
            let points = c.control_points();
            [points[0], points[points.len() - 1]]
        })
        .collect();
    let alone = |at: &Point3<f64>| every_end.iter().filter(|p| meets(p, at)).count() == 2;

    let Some((start, end, last)) = ends_of(curves[0]) else { return out };
    let first = (curves[0], 0usize, start);
    // The chain's tail: which curve, which of its control points, and where that is.
    let mut tail = (curves[0], last, end);
    for &next in &curves[1..] {
        let Some((start, end, last)) = ends_of(next) else { return Vec::new() };
        let (near, far, far_at) = if meets(&tail.2, &start) {
            (0, last, end)
        } else if meets(&tail.2, &end) {
            (last, 0, start)
        } else {
            // Not the loop it was said to be. Better no joints than joints in the wrong places.
            return Vec::new();
        };
        if alone(&tail.2) {
            out.push(Meeting { at: tail.2, ends: [(tail.0, tail.1), (next, near)] });
        }
        tail = (next, far, far_at);
    }
    if meets(&tail.2, &first.2) && alone(&tail.2) {
        out.push(Meeting { at: tail.2, ends: [(tail.0, tail.1), (first.0, first.1)] });
    }
    out
}

/// Whether every control point of every curve in `curves` lies on one plane.
///
/// What `sketcher-spec.md` §6 calls a stamp, asked of the geometry rather than stored: a Bézier
/// lies on a plane exactly when its control points do, so this is the fact a stamp would
/// record. Judged at [`JOINED_MM`], the one tolerance a geometric fact is judged at here — two
/// ends either meet or they do not, and a point is either on the plane or it is not — rather
/// than a second number invented for flatness (§6.2). The plane is fitted to all the points
/// at once, so a loop drawn across two planes reads as 3D even when each curve is flat.
pub fn loop_is_flat(
    document: &fieldform_scene::Document,
    curves: &[fieldform_scene::CurveId],
) -> bool {
    let points: Vec<Point3<f64>> = curves
        .iter()
        .filter_map(|id| document.curves.get(id))
        .flat_map(|c| c.control_points().iter().copied())
        .collect();
    let Some((origin, normal)) = plane_of(&points) else {
        // Nothing to fit a plane to — a loop of straight collinear runs has no normal. Such a
        // loop has no inside either, so calling it flat would promise an extrude it cannot have.
        return false;
    };
    points.iter().all(|p| (p - origin).dot(&normal).abs() <= JOINED_MM)
}

/// Extends `chain` until it closes, trying every continuation and undoing the ones that fail.
///
/// # Why this searches rather than walks
///
/// It walked, and took the only continuation it found — refusing outright when it found two,
/// because there was "more than one way round". That is true and it is not a reason to give up:
/// drawing an unrelated line that happens to start where a profile does made the profile
/// ambiguous and broke an extrude built on it, which is a graph edit reaching out and changing
/// what another block means.
///
/// A branch is only genuinely ambiguous if more than one choice *closes*. So the answer is to
/// try them. The recursion is bounded by the number of open curves in the document and every
/// branch is undone before the next is tried, so nothing accumulates.
///
/// **The first loop found wins**, and the pieces are in a fixed order, so the answer is stable
/// for a given document even when two loops genuinely exist. Deciding *which* loop somebody
/// meant, when there really are two, is a question this cannot answer and should not guess at
/// loudly — but a stable answer that draws something beats a refusal that draws nothing.
fn close_the_loop(
    pieces: &[(fieldform_scene::CurveId, Vec<Point3<f64>>)],
    joined_mm: f64,
    chain: &mut Vec<Point3<f64>>,
    used: &mut Vec<bool>,
    walked: &mut Vec<fieldform_scene::CurveId>,
    stranded: &mut Option<(f64, Point3<f64>)>,
) -> bool {
    let meets = |a: &Point3<f64>, b: &Point3<f64>| (a - b).norm() <= joined_mm;
    let tail = *chain.last().expect("a piece has at least two points");
    if chain.len() > 2 && meets(&tail, &chain[0]) {
        return true;
    }

    let mut went_anywhere = false;
    for i in 0..pieces.len() {
        if used[i] {
            continue;
        }
        let piece = &pieces[i].1;
        let forwards = meets(&tail, &piece[0]);
        let backwards = meets(&tail, piece.last().expect("two points"));
        if !forwards && !backwards {
            continue;
        }
        went_anywhere = true;

        // Reversed when joined by its far end, so the chain keeps running one way. A profile's
        // winding decides its inside, so a piece appended backwards is not a detail.
        let mut added: Vec<Point3<f64>> = piece.clone();
        if !forwards {
            added.reverse();
        }
        // The shared point is in both pieces; keeping it would leave a zero-length segment.
        let grew = added.len() - 1;
        chain.extend(added.into_iter().skip(1));
        used[i] = true;
        walked.push(pieces[i].0);

        if close_the_loop(pieces, joined_mm, chain, used, walked, stranded) {
            return true;
        }

        chain.truncate(chain.len() - grew);
        used[i] = false;
        walked.pop();
    }

    // Nowhere left to go from here. Remember the widest gap reached, since that is the end a
    // person is looking for — and prefer the deepest attempt, which is the one that got closest
    // to being a loop.
    if !went_anywhere {
        let gap = (tail - chain[0]).norm();
        if stranded.is_none_or(|(best, _)| gap < best) {
            *stranded = Some((gap, tail));
        }
    }
    false
}

/// Why a curve could not become a profile.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProfileError {
    #[error(
        "a profile must be a closed curve: an open one has no inside, and a field's sign is a \
         statement about being inside. Close the curve, or use it as a path rather than a \
         profile."
    )]
    NotClosed,
    #[error(
        "a profile needs at least three distinct points, and this one has {0} after tessellation"
    )]
    TooFewPoints(usize),
    #[error(
        "a profile must be flat: this one strays {0:.3} mm from the plane it is being read in. \
         A curve that is closed but not planar has no inside either — the region it would \
         enclose depends on which surface is imagined spanning it."
    )]
    NotPlanar(f64),
    #[error(
        "a profile being revolved has to stay on one side of its axis, and this one spans \
         {0:.1}° about it. Revolving material through the axis makes a solid that passes \
         through itself, and folding it onto one side would revolve a shape nobody drew."
    )]
    CrossesTheAxis(f64),
}

impl Profile {
    /// Builds a profile from points already in the plane's own coordinates.
    pub fn new(points: Vec<Point2<f64>>) -> Result<Self, ProfileError> {
        // Repeated endpoints are dropped, so a caller that closed the loop by repeating the
        // first point gets the same profile as one that did not. A zero-length segment is not
        // wrong, only useless, and it makes the crossing test ambiguous.
        let mut kept: Vec<Point2<f64>> = Vec::with_capacity(points.len());
        for p in points {
            if kept.last().is_none_or(|q| (p - q).norm() > 1e-12) {
                kept.push(p);
            }
        }
        if kept.len() > 1 && (kept[0] - kept[kept.len() - 1]).norm() <= 1e-12 {
            kept.pop();
        }
        if kept.len() < 3 {
            return Err(ProfileError::TooFewPoints(kept.len()));
        }
        let miters = miters_of(&kept);
        Ok(Self { points: kept, miters, baked: None })
    }

    /// Where each corner goes per millimetre a draft offsets the section, when the draft
    /// mitres its corners rather than rounding them.
    ///
    /// **A property of the boundary, computed once with it**, which is what makes a sharp
    /// draft cost the walk it always cost: the mitred offset is this same polygon with each
    /// vertex moved by `t · miter`, and `offset_by` below is that walk.
    pub(crate) fn miters(&self) -> &[Vector2<f64>] {
        &self.miters
    }

    /// The section offset outward by `t`, with its corners cut as `corners` says.
    ///
    /// **Round is `d(q) − t`**, which is the offset of a distance field and rounds every convex
    /// corner by `t` -- exact, and what a draft has always done. **Sharp is the same walk over
    /// the displaced boundary**, which is the mitre: the two walls stay planes and meet at an
    /// edge. Both are exact distances to the shape they describe;
    /// `examples/mitred_draft.rs` measures that they agree off every face, that the sharp solid
    /// never holds less than the round one, and that the walls lean at the draft angle either
    /// way.
    ///
    /// A negative `t` narrows the section, where the question reverses: an inward offset leaves
    /// a convex corner sharp of its own accord and rounds the reflex ones. The same arithmetic
    /// answers both, since the miter vectors point whichever way the offset does.
    pub(crate) fn offset_by(&self, q: Point2<f64>, t: f64, corners: fieldform_scene::Corners) -> f64 {
        if t == 0.0 || matches!(corners, fieldform_scene::Corners::Round) {
            return self.distance(q) - t;
        }
        walk_points(
            self.points.iter().zip(&self.miters).map(|(p, m)| p + m * t),
            self.points.len(),
            q,
        )
    }

    /// Builds one from a curve's tessellated points, read in a frame.
    ///
    /// `into_frame` maps a world point into the plane's coordinates: `x` and `y` across it, `z`
    /// along its normal. The `z` values are what the planarity check reads, so a caller does not
    /// have to decide what "flat" means — the frame does.
    pub fn in_plane(
        world: &[Point3<f64>],
        into_frame: impl Fn(Point3<f64>) -> Point3<f64>,
        flatness_mm: f64,
    ) -> Result<Self, ProfileError> {
        let local: Vec<Point3<f64>> = world.iter().map(|p| into_frame(*p)).collect();
        let stray = local.iter().map(|p| p.z.abs()).fold(0.0f64, f64::max);
        if stray > flatness_mm {
            return Err(ProfileError::NotPlanar(stray));
        }
        Self::new(local.iter().map(|p| Point2::new(p.x, p.y)).collect())
    }

    /// The same, read as a half-plane about the frame's axis, for a revolve.
    ///
    /// `x` becomes the distance from the axis and `y` the height along it, so a profile drawn
    /// in any half-plane about the axis produces the same solid — a revolve is symmetric, and
    /// which half-plane it was drawn in cannot matter.
    ///
    /// **The profile need not be coplanar with the axis, and this used to insist that it was.**
    /// Taking `hypot(x, y)` folds everything onto the positive side, so a profile that is not in
    /// a single half-plane comes back as a different, smaller profile — its *radial footprint*.
    /// That was refused, on the grounds that revolving it is "not what anybody drew".
    ///
    /// Reversed by the owner, deliberately, after it refused ordinary work three times running:
    /// draw a profile, draw an axis, drag the axis a millimetre off the profile's plane, and the
    /// block stops compiling. The coplanarity that satisfied the check is not something you can
    /// see or maintain by hand, and nothing in the interface was helping hold it.
    ///
    /// What is given up is smaller than it sounds, and what is kept matters more:
    ///
    /// - **The field stays exact.** The footprint is a closed curve like any other, and the
    ///   distance to the surface of revolution it generates is exact in `(r, z)` — that is a
    ///   property of the revolve, not of where the profile happened to be drawn.
    /// - **The fill rule already covers the fold.** A footprint that crosses itself fills
    ///   even–odd, the same as any other self-crossing profile, which is the documented rule and
    ///   the one a figure-eight already relies on.
    /// - What is lost is the *guarantee* that the solid's cross-section is the shape drawn. Off
    ///   the plane it is the shape's shadow in cylindrical coordinates, which is a different
    ///   shape and often the intended one — a square held out to the side revolves into a ring.
    ///
    /// So the span is no longer a refusal. [`azimuth_span`] still measures it, and the panel
    /// reports it, because "your profile is not in the axis's plane" is worth knowing and is not
    /// worth stopping for.
    pub fn about_axis(
        world: &[Point3<f64>],
        into_frame: impl Fn(Point3<f64>) -> Point3<f64>,
    ) -> Result<Self, ProfileError> {
        if world.len() < 2 {
            return Err(ProfileError::TooFewPoints(world.len()));
        }
        Self::new(Self::radial_footprint(world, true, into_frame))
    }

    /// A polyline's points read as `(r, z)` about the frame's axis: the map
    /// [`about_axis`](Self::about_axis) builds a profile from, for a chain that need not close.
    ///
    /// A wall's curve is revolved the way a profile's is, so it is laid in the half-plane by
    /// the same map — and a chain has no closing segment, which is the only difference
    /// `closed` makes. An open chain keeps its last point; a loop's is implied.
    pub(crate) fn radial_footprint(
        world: &[Point3<f64>],
        closed: bool,
        into_frame: impl Fn(Point3<f64>) -> Point3<f64>,
    ) -> Vec<Point2<f64>> {
        let local: Vec<Point3<f64>> = world.iter().map(|p| into_frame(*p)).collect();
        // **Straight in space is curved in `(r, z)`, so the segments are subdivided.** Radius is
        // `hypot(x, y)`, which is not linear along a straight edge unless that edge lies in the
        // axis's plane -- an edge held out to the side is nearest the axis at its middle. So the
        // four corners of a square off to one side map to *two* distinct radii and the footprint
        // collapses to a line, which is the shape a segment's endpoints alone can describe.
        //
        // Nothing had to do this while the profile was required to be coplanar, because there
        // the map is linear and endpoints are the whole story. Sampling by turned angle rather
        // than by length for the same reason the pipe spine does: what is being resolved is the
        // bend, and a long edge that barely turns needs no help.
        let mut flat = Vec::with_capacity(local.len());
        let segments = if closed { local.len() } else { local.len().saturating_sub(1) };
        for (i, a) in local.iter().enumerate().take(segments) {
            let b = local[(i + 1) % local.len()];
            let steps = Self::steps_across(*a, b);
            for s in 0..steps {
                let t = s as f64 / steps as f64;
                let p = a + (b - a) * t;
                flat.push(Point2::new(p.x.hypot(p.y), p.z));
            }
        }
        if !closed && let Some(last) = local.last() {
            flat.push(Point2::new(last.x.hypot(last.y), last.z));
        }
        flat
    }

    /// How many pieces a segment needs so its arc in `(r, z)` is resolved to about a degree.
    ///
    /// One when the two ends sit at the same azimuth, which is every segment of a profile drawn
    /// in the axis's own plane -- so a coplanar revolve samples exactly what it always did.
    fn steps_across(a: Point3<f64>, b: Point3<f64>) -> usize {
        let angle = |p: Point3<f64>| p.y.atan2(p.x);
        let (ra, rb) = (a.x.hypot(a.y), b.x.hypot(b.y));
        if ra <= 1e-9 || rb <= 1e-9 {
            return 1;
        }
        let mut turn = (angle(b) - angle(a)).abs();
        if turn > std::f64::consts::PI {
            turn = std::f64::consts::TAU - turn;
        }
        (turn.to_degrees().ceil() as usize).clamp(1, 180)
    }

    /// How far a profile spreads around an axis, in degrees: zero when it lies in one half-plane.
    ///
    /// Reported rather than refused — see [`Profile::about_axis`]. Points *on* the axis have no
    /// direction across it and constrain nothing, so they are skipped rather than counted as
    /// agreeing with everything.
    pub fn azimuth_span(
        world: &[Point3<f64>],
        into_frame: impl Fn(Point3<f64>) -> Point3<f64>,
    ) -> f64 {
        let local: Vec<Point3<f64>> = world.iter().map(|p| into_frame(*p)).collect();
        let mut reference: Option<(f64, f64)> = None;
        let mut worst: f64 = 0.0;
        for p in local.iter().filter(|p| p.x.hypot(p.y) > 1e-9) {
            let r = p.x.hypot(p.y);
            let (ux, uy) = (p.x / r, p.y / r);
            match reference {
                None => reference = Some((ux, uy)),
                Some((rx, ry)) => {
                    worst = worst.max((ux * rx + uy * ry).clamp(-1.0, 1.0).acos());
                }
            }
        }
        worst.to_degrees()
    }

    /// The half-plane a section lies in about the axis, as its angle from the frame's `+x`:
    /// where a partial revolve starts from.
    ///
    /// The mean direction of the points' directions across the axis, so a profile that is not
    /// quite in one half-plane -- which `about_axis` says is allowed -- starts where its middle
    /// is. Points on the axis have no direction and are skipped; a section wholly on the axis
    /// has none either, and starts at `+x`.
    pub fn azimuth(
        world: &[Point3<f64>],
        into_frame: impl Fn(Point3<f64>) -> Point3<f64>,
    ) -> f64 {
        let (mut x, mut y) = (0.0, 0.0);
        for p in world.iter().map(|p| into_frame(*p)) {
            let r = p.x.hypot(p.y);
            if r > 1e-9 {
                x += p.x / r;
                y += p.y / r;
            }
        }
        if x.hypot(y) <= 1e-9 { 0.0 } else { y.atan2(x) }
    }

    /// How far the profile reaches, as a box in its own plane.
    pub fn bounds(&self) -> (Point2<f64>, Point2<f64>) {
        let mut lo = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &self.points {
            lo = Point2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Point2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
        (lo, hi)
    }

    /// The boundary points, for a caller that has to hand them to something else.
    pub fn points(&self) -> &[Point2<f64>] {
        &self.points
    }

    /// The signed distance from `q` to the boundary: negative inside.
    ///
    /// **Exact, and a true distance.** The magnitude is the Euclidean distance to the polyline —
    /// the minimum over its segments, each of which is a closed-form point-to-segment — so the
    /// gradient is unit everywhere except on the medial axis, where a distance field's gradient
    /// is undefined anyway. The sign is a separate question answered by crossing parity, so the
    /// two cannot disagree about magnitude near a corner the way a half-plane test would.
    pub fn distance(&self, q: Point2<f64>) -> f64 {
        match &self.baked {
            Some(grid) => grid.distance(q),
            None => self.walk(q),
        }
    }

    /// The same profile, answering from a grid instead of from its boundary.
    ///
    /// **A display convenience, and the tolerance is what says whether to ask for it.** Walking
    /// the boundary is exact and costs one pass over every segment; a grid costs one read and is
    /// short by its margin. Measured on a 200 x 140 mm ring by `examples/profile_bake_cost.rs`:
    /// 1628 ms per million walks against 10.3 ms per million reads, and flat in the point count.
    ///
    /// So it is worth it wherever the answer only has to be as good as a picture, and not worth
    /// it where the answer *is* the output. `cells` is chosen from the tolerance being worked to
    /// — see [`bake_cells`] — and a caller that asks for nothing keeps the exact boundary.
    pub fn baked(mut self, cells: usize, reads: (f64, f64)) -> Self {
        self.baked = Some(crate::BakedProfile::new(&self, cells, reads));
        self
    }

    /// Whether this profile answers from a grid.
    pub(crate) fn is_baked(&self) -> bool {
        self.baked.is_some()
    }

    /// A hash of the boundary this profile actually has.
    ///
    /// **The identity of a bake, because the bake is a function of these points and nothing
    /// else.** Taken here rather than over the loop the points were walked out of: a revolve
    /// never flattens a loop into a plane at all — [`Profile::about_axis`] maps each world point
    /// into its frame's `(r, z)` — so two revolves off one sketch with different axes have the
    /// same loop and different profiles. Four of five readers set to break a key built on the
    /// loop broke it that way, and all four proposed this.
    ///
    /// Ordered, and the length is in it: the closing segment runs from the last point to the
    /// first, so the same points rotated are a different polyline. Bits rather than values, the
    /// way [`fieldform_cage::QuadMesh::geometry_fingerprint`] does it, so nothing depends on how
    /// a float compares.
    pub(crate) fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.points.len().hash(&mut hasher);
        for p in &self.points {
            p.x.to_bits().hash(&mut hasher);
            p.y.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The same boundary, answering from a grid somebody else already built for it.
    ///
    /// For a cache: the bake is a pure function of the points, the cell count and the band, so a
    /// bake taken under an equal key is *this* profile's bake, not a near one.
    pub(crate) fn with_bake(mut self, bake: crate::BakedProfile) -> Self {
        self.baked = Some(bake);
        self
    }

    /// The grid, for a caller that has to hand it to a shader.
    pub fn bake(&self) -> Option<&crate::BakedProfile> {
        self.baked.as_ref()
    }

    /// The distance to the boundary, by walking every segment of it.
    ///
    /// What [`Profile::distance`] does when nothing has been baked — and what a bake is sampled
    /// *from*, which is why it is separate rather than inlined above: sampling a grid from
    /// itself would be a circle.
    pub fn walk(&self, q: Point2<f64>) -> f64 {
        walk_points(self.points.iter().copied(), self.points.len(), q)
    }
}

/// The signed distance to a closed polygon, negative inside: nearest point and crossing parity,
/// in one pass.
///
/// **Over an iterator rather than a slice**, so that the mitred offset can be walked without
/// building its displaced boundary first -- one allocation per query is what would make a
/// sharp draft cost more than a round one, and it costs the same.
fn walk_points(
    points: impl Iterator<Item = Point2<f64>> + Clone,
    n: usize,
    q: Point2<f64>,
) -> f64 {
    let mut nearest = f64::INFINITY;
    let mut inside = false;
    let mut first = None;
    let mut previous = None;
    let mut seen = 0;
    for p in points {
        if first.is_none() {
            first = Some(p);
        }
        if let Some(a) = previous {
            edge(a, p, q, &mut nearest, &mut inside);
        }
        previous = Some(p);
        seen += 1;
    }
    debug_assert_eq!(seen, n, "the walk was handed a different number of points than it was told");
    if let (Some(a), Some(b)) = (previous, first) {
        edge(a, b, q, &mut nearest, &mut inside);
    }
    if inside { -nearest } else { nearest }
}

/// One edge's contribution to the walk: how near it is, and whether it crosses the ray.
///
/// Crossing parity counts each edge once. The `>` and `<=` pair is what stops a ray passing
/// exactly through a vertex being counted twice: an edge owns its lower endpoint and not its
/// upper, so a vertex belongs to exactly one of the two edges meeting there.
fn edge(a: Point2<f64>, b: Point2<f64>, q: Point2<f64>, nearest: &mut f64, inside: &mut bool) {
    let ab = b - a;
    let t = if ab.norm_squared() > 0.0 {
        ((q - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    *nearest = nearest.min((q - (a + ab * t)).norm());
    if (a.y > q.y) != (b.y > q.y) {
        let x = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
        if q.x < x {
            *inside = !*inside;
        }
    }
}

/// The miter vector at each vertex: where that corner goes per millimetre of outward offset.
///
/// `(n₁ + n₂) / (1 + n₁·n₂)` is where the two offset lines meet, which is the whole of the
/// mitre; its length is `1 / sin(half the included angle)`, so it runs away as a corner
/// sharpens and is held to [`fieldform_scene::MITRE_LIMIT`]. A corner past the limit keeps an
/// answer that is finite rather than a spike thrown across the part.
///
/// The outward direction is taken from the polygon's own winding, so a section drawn either way
/// round offsets outward.
fn miters_of(points: &[Point2<f64>]) -> Vec<Vector2<f64>> {
    let n = points.len();
    let twice_area: f64 = (0..n)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum();
    let facing = if twice_area < 0.0 { -1.0 } else { 1.0 };
    let outward = |a: Point2<f64>, b: Point2<f64>| {
        (b - a).try_normalize(1e-12).map(|e| Vector2::new(e.y, -e.x) * facing)
    };
    (0..n)
        .map(|i| {
            let here = points[i];
            let (Some(n1), Some(n2)) = (
                outward(points[(i + n - 1) % n], here),
                outward(here, points[(i + 1) % n]),
            ) else {
                return Vector2::zeros();
            };
            let share = 1.0 + n1.dot(&n2);
            if share < 1e-9 {
                // A corner that turns right back on itself has no mitre to speak of.
                return Vector2::zeros();
            }
            let m = (n1 + n2) / share;
            if m.norm() > fieldform_scene::MITRE_LIMIT {
                m.normalize() * fieldform_scene::MITRE_LIMIT
            } else {
                m
            }
        })
        .collect()
}

/// A profile swept along a direction that is *not* its normal — a leaning prism.
///
/// # What it is, in one line
///
/// Every point is slid back along the sweep direction until it lands on the profile's plane;
/// what it finds there decides whether it is inside, and how far it slid decides whether it is
/// within the depth. Sweeping along the normal makes the slide a straight drop and gives
/// [`Extrusion`], which is exact; sweeping along anything else shears the solid, and shearing is
/// where the exactness goes.
///
/// # Why this is `LowerBound` and cannot be made exact
///
/// A shear does not preserve distance. Two points a millimetre apart across the lean are less
/// than a millimetre apart in the sheared coordinates the profile is measured in, so the profile's
/// own reading *overstates* how far away the surface is — which is the one thing a
/// [`ScalarField`] may never do.
///
/// The map from world to sheared coordinates has gradient no larger than `1 / |d·n|`, so
/// multiplying the result by `|d·n|` brings it back to 1-Lipschitz and turns an overestimate into
/// an underestimate. That is the whole correction, and it is why the field is exactly as good as
/// the lean is small: at `d = n` the factor is 1 and this *is* [`Extrusion`]; at 60° off it reads
/// half the true distance; edge-on it reads nothing at all, which is why a direction lying in the
/// profile's plane is refused rather than divided by zero.
///
/// D8 asks a source that does this to say so, and `Sweep::is_exact` is what the panel asks so it
/// can — the same arrangement a fillet over a lattice already has.
pub(crate) struct SkewExtrusion {
    /// What is swept: a closed profile or a wall, read in its own plane. See
    /// [`Extrusion::footprint`].
    pub footprint: crate::Footprint,
    /// How far **along the sweep direction** from the plane the material begins — see
    /// [`Extrusion::start`], which it means the same way, measured along the sweep.
    pub start: f64,
    /// The whole depth of side one, measured **along the sweep direction**, from the start.
    pub height: f64,
    /// Which way the depth lies about the start — see [`Extrusion::depth`], which it means
    /// the same way, measured along the sweep instead of along the normal.
    pub depth: Depth,
    /// The sweep direction in the profile's own frame: `+Z` would be the exact case, and
    /// anything else leans. Unit length.
    pub direction: Vector3<f64>,
    /// The tangent of the draft angle — see [`Extrusion::draft_tangent`], which it means the
    /// same way. Measured along the sweep, so a leaning drafted extrude opens about the sweep
    /// rather than about the plane's normal, which is what somebody aiming one would expect.
    pub draft_tangent: f64,
    /// Whether the draft rounds the section's corners or mitres them — see
    /// [`Extrusion::corners`], which it means the same way.
    pub corners: fieldform_scene::Corners,
    /// How the rims are cut where the side meets each end -- see [`Extrusion::rims`].
    pub rims: crate::graph_field::Rim,
}

impl SkewExtrusion {
    /// How much the shear costs, as the factor the field is multiplied by.
    ///
    /// One when the sweep is straight, shrinking as it leans. Named because it appears twice —
    /// once to correct the value and once as the Lipschitz bound — and the two must be the same
    /// number.
    fn honesty(&self) -> f64 {
        self.direction.z.abs()
    }
}

impl ScalarField for SkewExtrusion {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let along = self.honesty();
        // Edge-on: the sweep never leaves the plane, so there is no solid. Refused at build
        // time; this is the belt to that pair of braces.
        if along < 1e-9 {
            return f64::INFINITY;
        }
        // How far along the direction this point sits from the plane, and where it lands on it.
        let t = p.z / self.direction.z;
        let landed = p - self.direction * t;

        // The two readings, exactly as `Extrusion` takes them -- along the sweep here, from
        // the start, which is where this one measures everything from.
        let (grown, slab) = grown_and_slab(
            t - self.start,
            self.height,
            self.draft_tangent,
            self.depth,
        );
        let plane =
            self.footprint.offset_by(Point2::new(landed.x, landed.y), grown, self.corners);
        let steepest = steepest_tangent(self.draft_tangent, self.depth);
        // **The larger, unless there is no shear to lose.** `Extrusion` may take the corner
        // outside both faces, because there the profile's distance and the slab's are measured
        // along perpendicular directions and Pythagoras is exact. A shear is precisely the loss
        // of that perpendicularity, so combining them that way overstates -- measured at 8 mm on
        // a forty-five degree lean before this line said `max`. `max` is the conservative CSG
        // answer, and it costs a sphere trace some steps near the rim, which is the right thing
        // to spend.
        //
        // But a sweep *typed as the profile's own normal* is not sheared at all, and taking the
        // larger there threw away exactness for nothing: switching a working extrude from
        // "Normal" to "Typed" without touching a number made the field read up to 36% short at
        // the corners, measured by `examples/lean_error.rs`. The shader has always taken the
        // corner in this case, so this was also a quiet disagreement between the two -- one the
        // GPU/CPU check cannot see, because scaling a field about its own zero set moves no
        // surface.
        // The corner needs the two measurements square to each other, which a lean destroys and
        // a draft tilts — so it is taken only when there is neither.
        if !self.rims.is_hard() {
            // Through the blend's bound, which holds at any angle between the two readings, so
            // the lean's and the draft's divisor is all it needs -- see `Rim::join`.
            let toward = toward_end(t - self.start, self.height, self.depth);
            return self.rims.join_ends(plane, slab, toward) / extrude_divisor(along, steepest);
        }
        if along >= 1.0 - 1e-9 && steepest == 0.0 && plane > 0.0 && slab > 0.0 {
            return plane.hypot(slab);
        }
        // Divided by what a lean and a draft together cost, which is more than each alone --
        // `extrude_divisor` says how much and why. With two sides the slope is the steeper
        // side's: the bound has to hold on both.
        plane.max(slab) / extrude_divisor(along, steepest)
    }

    fn normalization(&self) -> Normalization {
        // Never exact. `Sweep::Normal` is a different type precisely so that the exact case does
        // not depend on noticing that a direction happens to be the normal.
        Normalization::LowerBound
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

/// A profile extruded along its own normal, centred on its plane.
#[derive(Debug, Clone)]
pub(crate) struct Extrusion {
    /// What is thickened, as a distance in its own plane: a closed profile, or a wall.
    ///
    /// **Either, and nothing here knows which.** Everything below reads one 2D distance and
    /// combines it with a slab, and that is all an extrusion is; whether the distance came
    /// from a loop's inside or a curve's thickness changes nothing about the combining. So a
    /// thin extrude is this same solid, asked of a different footprint.
    pub footprint: crate::Footprint,
    /// How far along the normal from the profile's plane the material begins.
    ///
    /// Nought is the plane itself, which is what every extrude was. A translation along the
    /// sweep and nothing else: the start plane is where the solid begins, and so the base a
    /// draft is measured from, but the section is still read in the profile's own plane.
    pub start: f64,
    /// The whole depth of side one, measured from the start plane.
    ///
    /// **From the plane, not about it.** It was centred once, on the argument that a profile
    /// drawn on a construction plane wants to straddle it — which is true of a plane through the
    /// middle of a part, and false of the thing people actually do, which is draw on a face and
    /// pull material off it. Every CAD tool starts an extrude at its sketch plane and this one
    /// now does too.
    ///
    /// It also makes the draft mean what a moulding drawing means: the profile as drawn is the
    /// *base*, and the walls open away from it. Centred, the drawn shape was the waist of a
    /// frustum and neither end was the thing you dimensioned.
    ///
    /// **Signed**, so a negative depth runs the other way out of the plane. Which side a solid
    /// grows toward only became a question once it stopped straddling, and this is the answer
    /// that needs no second piece of state: a flip is a minus sign on a number that already
    /// exists, rather than a boolean beside it that could disagree with it.
    pub height: f64,
    /// Which way the depth lies about the start plane: one side, half each way, or two sides
    /// with a depth and a draft of their own.
    ///
    /// A closed form rather than two halves unioned, and that is the whole reason it lives
    /// here instead of being built in the graph: two solids sharing a face are the one join
    /// `min` cannot describe, and it reads nought right through the material. See
    /// `fieldform_scene::Sides` and `examples/opposed_extrudes.rs`.
    pub depth: Depth,
    /// How far the walls lean out per unit of height: the tangent of the draft angle.
    ///
    /// **A draft is an offset, not a scale**, and the difference matters for any profile that is
    /// not centred on its own origin. Scaling a profile by a factor moves it *toward or away
    /// from that origin* — a shape drawn off to one side would slide sideways as it tapered,
    /// which is not what a moulding drawing means and not what anybody would predict. An offset
    /// moves every wall out by the same amount along its own normal, which is exactly what a
    /// draft angle is: the wall makes that angle with the direction of pull.
    ///
    /// And an offset of a distance field is *exact* — subtracting a constant from a distance is
    /// the distance to the offset shape, which is the same reasoning `Node::Offset` already
    /// rests on. So the shape is right for free; what it costs is [`Extrusion::honesty`].
    ///
    /// Zero for a prism, which is the common case and costs nothing to check.
    pub draft_tangent: f64,
    /// Whether the draft rounds the section's corners or mitres them.
    ///
    /// **A draft is an offset, and an offset rounds a convex corner by its own amount** -- so
    /// the corners grow rounder as the draft grows, which is geometrically right and is often
    /// not what was drawn. Sharp keeps the two walls planes and meets them at an edge. Both
    /// are exact; see [`crate::Profile::offset_by`], and `fieldform_scene::Corners` for the
    /// ruling. Nothing to pay when there is no draft, since neither does anything then.
    pub corners: fieldform_scene::Corners,
    /// How the rims are cut where the side wall meets each end: hard, rounded at G1 or G2,
    /// or chamfered (`round-chamfer-spec.md` §4). Both ends alike. A hard rim is the solid
    /// this always was, to the bit.
    pub rims: crate::graph_field::Rim,
}

impl Extrusion {
    /// What the reading has to be multiplied by to stay a lower bound.
    ///
    /// **A sloped wall reads long, and by a known amount.** The value is
    /// `profile − z·tan(θ)`, whose gradient picks up that `tan(θ)` in the z direction on top of
    /// the profile's own unit gradient — so it grows at `sec(θ)` rather than at one, and a
    /// reading has to be divided by that to be a distance again. The cosine is the same thing
    /// written as a multiply.
    ///
    /// Closed form, so the error is *known* rather than bounded by hope: at five degrees of
    /// draft the field reads 0.4% long before the correction and nothing after it.
    fn honesty(&self) -> f64 {
        1.0 / extrude_divisor(1.0, steepest_tangent(self.draft_tangent, self.depth))
    }
}

/// What an extrusion divides its reading by to stay a bound, from the cosine of its lean --
/// one for an extrusion along its own normal -- and the steeper of its draft tangents.
///
/// **A lean and a draft together cost more than each alone**, and they were charged as if they
/// did not: the shear's factor times the slope's, which is right for either on its own and short
/// for both. The section is read at `landed = p − d·t`, so moving a point along the normal moves
/// where it is read by `−d_xy / d_z` across the plane, and the draft takes `tan / d_z` off the
/// reading for the same move: where the section's own gradient lies along the lean the two add,
/// and the reading's rate along the normal is `(sin φ + tan α) / cos φ`, not `tan α / cos φ`.
/// The rate across the plane is one. So the bound is `√(cos² φ + (sin φ + tan α)²) / cos φ`,
/// which is `1 / cos φ` with no draft and `√(1 + tan² α)` with no lean, and more than their
/// product with both. Measured by `examples/lean_draft_probe.rs`: a one-sided extrude along
/// (0.4, 0.2, 1) drafted 25° claimed 2.43 mm more room than there was with the product, and none
/// with this.
///
/// **One function for the field and for the step budget.** The field divides by it
/// ([`Extrusion`], [`SkewExtrusion`]) and `step_multiplier` buys steps against it
/// (`extrude_stretch`), and the two have to be the same number: the extrude was the sixth
/// consumer of the one cause the attractor, the morph, the pattern, the twist and the sweep
/// each were -- a division the walk was not told of, so the tracer stepped by a value shorter
/// than it believed and lost the surface at grazing incidence.
pub(crate) fn extrude_divisor(along: f64, steepest: f64) -> f64 {
    let along = along.abs().max(1e-9);
    let lean = (1.0 - along * along).max(0.0).sqrt();
    (along * along + (lean + steepest.abs()).powi(2)).sqrt() / along
}

/// Which way an extrusion's depth lies about its start plane: the kernel's reading of
/// [`fieldform_scene::Sides`], with side two's numbers resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Depth {
    /// From the start, the whole depth one way.
    One,
    /// Half the depth each way, the profile in the middle.
    Symmetric,
    /// The whole depth one way and this much the other, with a draft of its own.
    Two { height: f64, draft_tangent: f64 },
}

/// **How far the draft has moved the wall at `along`, and the distance out of the slab**: the
/// two readings an extrusion combines, with the section's own left to the caller.
///
/// **Measured from the start plane**, so the draft opens from where the material begins,
/// which is the base a moulding drawing measures from. The profile as drawn is that base for
/// one side; the widest section for half each way, the walls opening away on both sides,
/// which is what a parting line means; and for two sides it is inside the material, each side
/// growing from it by its own draft. `height` is side one's, signed, and a negative one runs
/// the other way: side two then runs the way side one would have.
///
/// One function for the upright extrusion and the leaning one, which differ only in what
/// `along` is measured along.
///
/// Said apart from the section's reading because a draft's offset is not always a number to
/// subtract. Rounding the corners
/// is `d − grown`, which is the offset of a distance field; mitring them is the section's own
/// walk over a displaced boundary, which needs `grown` rather than the reading it would have
/// produced. [`Footprint::offset_by`] takes it either way.
fn grown_and_slab(along: f64, height: f64, tangent: f64, depth: Depth) -> (f64, f64) {
    match depth {
        // Zero at the start and `height` at the far end, so the slab is the distance out of
        // `[0, height]` rather than out of a band centred on nothing in particular. Written
        // to work for either sign, since a negative depth is how an extrude is flipped, and
        // the draft is measured along the extrusion so a flipped solid still opens away.
        Depth::One => {
            let half = height * 0.5;
            (along * height.signum() * tangent, (along - half).abs() - half.abs())
        }
        // Folded about the start. `|along|` makes the profile the widest section rather than
        // the base. Still one expression: the fold is a reflection, and reflecting a distance
        // leaves it a distance.
        Depth::Symmetric => {
            let half = height.abs() * 0.5;
            (along.abs() * tangent, along.abs() - half)
        }
        Depth::Two { height: other, draft_tangent: other_tangent } => {
            let forward = if height < 0.0 { -1.0 } else { 1.0 };
            let u = along * forward;
            let (one, two) = (height.abs(), other.abs());
            let grown = if u >= 0.0 { u * tangent } else { -u * other_tangent };
            (grown, (u - (one - two) * 0.5).abs() - (one + two) * 0.5)
        }
    }
}

/// How far past the middle of its length a point `along` the sweep from the start sits,
/// positive toward the solid's end -- the far face of one side, side one's face of two or of a
/// symmetric pair -- so that the slab [`grown_and_slab`] gives is `|toward| − half` exactly, and
/// a rim on one end can tell which end the point is nearer ([`crate::graph_field::Rim::join_ends`]).
fn toward_end(along: f64, height: f64, depth: Depth) -> f64 {
    match depth {
        Depth::One => (along - height * 0.5) * height.signum(),
        Depth::Symmetric => along,
        Depth::Two { height: other, .. } => {
            let forward = if height < 0.0 { -1.0 } else { 1.0 };
            along * forward - (height.abs() - other.abs()) * 0.5
        }
    }
}

/// The steeper of an extrusion's drafts, which is what its bound has to hold against.
fn steepest_tangent(tangent: f64, depth: Depth) -> f64 {
    match depth {
        Depth::Two { draft_tangent, .. } => tangent.abs().max(draft_tangent.abs()),
        _ => tangent.abs(),
    }
}

impl ScalarField for Extrusion {
    fn distance(&self, p: Point3<f64>) -> f64 {
        // The 2D field and a slab, intersected. `max` of two true distances underestimates
        // near the rim where both are active — the ordinary CSG crease, bounded and local, and
        // the one this codebase has always accepted as `Distance`. `plane_and_slab` says how
        // the draft and the sides are read.
        let (grown, slab) = grown_and_slab(
            p.z - self.start,
            self.height,
            self.draft_tangent,
            self.depth,
        );
        // **The section is asked for its own offset rather than handed a number to subtract.**
        // Rounding the corners is the subtraction it always was; mitring them is the section's
        // walk over a boundary displaced by `grown`, which no amount of arithmetic on the
        // reading could produce.
        let plane = self.footprint.offset_by(Point2::new(p.x, p.y), grown, self.corners);
        // Outside both, the true distance is the diagonal rather than the larger of the two.
        // Taking the max there would understate it, which is safe but slow; the corner is
        // cheap and makes a sphere trace along the rim take the steps it should.
        //
        // Only while the walls are upright, though: the corner is Pythagoras between two
        // *perpendicular* measurements, and a draft tilts one of them out of square with the
        // other. Same reason `SkewExtrusion` takes the larger — see the note there, which cost
        // eight millimetres to learn.
        let steepest = steepest_tangent(self.draft_tangent, self.depth);
        // **A rounded or chamfered rim is the blend's own bound across the two** -- see
        // `Rim::join`, which says why it needs no divisor the draft did not already bring.
        let combined = if !self.rims.is_hard() {
            let toward = toward_end(p.z - self.start, self.height, self.depth);
            self.rims.join_ends(plane, slab, toward)
        } else if plane > 0.0 && slab > 0.0 && steepest == 0.0 {
            plane.hypot(slab)
        } else {
            plane.max(slab)
        };
        combined * self.honesty()
    }

    fn normalization(&self) -> Normalization {
        // A draft reads short once corrected, the same as a lean does, so it announces itself
        // the same way — a blend against a drafted wall comes out larger than the radius typed.
        // A rounded rim reads short inside its reach, as every fillet does.
        if steepest_tangent(self.draft_tangent, self.depth) != 0.0 || !self.rims.is_hard() {
            return Normalization::LowerBound;
        }
        // A baked profile is a bound and a walked one a distance; a wall is a distance always.
        self.footprint.normalization()
    }

    fn lipschitz(&self) -> f64 {
        // Unchanged by a bake: the stored value is a distance less a constant, and subtracting a
        // constant does not change how fast a function moves. Unchanged by a draft either, since
        // the correction is exactly what puts the rate back to one.
        1.0
    }
}

/// A section swept about the frame's own axis: a profile, or a wall.
#[derive(Debug, Clone)]
pub struct Revolution {
    /// Read in `(r, z)`: `x` is the distance from the axis, `y` the height along it.
    pub footprint: crate::Footprint,
}

impl ScalarField for Revolution {
    fn distance(&self, p: Point3<f64>) -> f64 {
        // Exactly the 2D distance, and exactly a true 3D one. The nearest point on a surface of
        // revolution lies in the same half-plane as the query, so the distance in `(r, z)` *is*
        // the distance in space — no approximation, unlike the `max` an extrude has to make.
        self.footprint.about_axis(p.x.hypot(p.y), p.z)
    }

    fn normalization(&self) -> Normalization {
        self.footprint.normalization()
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

/// A section swept **part of the way** about the frame's axis: the wedge of the solid a
/// [`Revolution`] is, between two half-planes through the axis.
///
/// # Three cases, two of them exact (`extrude-revolve-options-spec.md` §3.2)
///
/// With `φ` the angle about the axis from the wedge's start and `d(r, z)` the section's own
/// distance in the half-plane:
///
/// 1. **Inside the wedge, outside the solid**: `d(r, z)`, exactly as a full revolve reads it.
///    Rotating any point of a cap to the query's own angle keeps it in the solid and brings it
///    nearer, so the nearest point lies in the query's half-plane.
/// 2. **Outside the wedge**: the distance to the nearer of the two caps, each a planar region
///    -- the section, in its own half-plane. With `ψ` the angle to that cap's plane (the
///    smaller way round), the query stands `r·sin ψ` off the plane and lands at `u = r·cos ψ`
///    across the axis in it, and the distance to the region is `√(max(d(u, z), 0)² + h²)`.
///    Exact: a distance to a planar region.
/// 3. **Inside the wedge, inside the solid**: the depth is bounded by `d(r, z)` and by the
///    distance to each cap's plane, `r·sin(min(ψ, 90°))` -- past ninety degrees that is the
///    distance to the axis. A bound and not the depth, because the nearest point of a cap's
///    plane need not lie in the cap.
///
/// So the leaf says `LowerBound`, divisor one: it understates only inside the material, and
/// only by the width of case 3. **If case 3 is proved exact the claim upgrades to
/// `Distance`** -- the argument would be that the projection of an interior point onto a cap's
/// plane always lands in the cap region before the revolved surface intervenes. Do not upgrade
/// on the unit-gradient test alone; it is evidence to look at, not the proof.
///
/// Case 2 is the shared end-cap helper `helix-thread-spec.md` reuses for coil and thread
/// ends: [`cap_distance`].
#[derive(Debug, Clone)]
pub(crate) struct RevolvedWedge {
    /// Read in `(r, z)`, as a full revolve's is.
    pub footprint: crate::Footprint,
    /// Where the wedge begins: the angle about the axis from the frame's `+x`, in radians,
    /// counter-clockwise looking down the axis -- the right-hand rule about it.
    pub from: f64,
    /// How far it turns from there, in radians: above nought and under a full turn, which the
    /// compile holds it to. A full turn is a [`Revolution`].
    pub sweep: f64,
    /// How the rims are cut where the turned side meets each cap -- hard unless
    /// [`RevolvedWedge::with_rims`] says otherwise. See `distance` for the reading.
    pub rims: crate::graph_field::Rim,
}

impl RevolvedWedge {
    /// The same wedge with its rims cut as told: the start cap is the one at `from`, the end
    /// cap the one `sweep` on from it.
    pub fn with_rims(mut self, rims: crate::graph_field::Rim) -> Self {
        self.rims = rims;
        self
    }

    /// Whether the cap `which` -- 1 the start, 2 the end -- is rounded.
    fn rounded(&self, which: u8) -> bool {
        use fieldform_scene::RimEnds;
        !self.rims.is_hard()
            && match self.rims.ends {
                RimEnds::Both => true,
                RimEnds::Start => which == 1,
                RimEnds::End => which == 2,
            }
    }

    /// The section's reading `x` against a cap's `y`: the blend's bound where the cap is
    /// rounded, the plain intersection where it is hard -- which is what every wedge read.
    fn against(&self, which: u8, x: f64, y: f64) -> f64 {
        if self.rounded(which) { self.rims.cut(x, y) } else { x.max(y) }
    }
}

/// The distance to a planar cap region seen from `h` off its plane and landing at `(u, z)` in
/// it, where `d` is the region's own signed distance there: Pythagoras outside the region, and
/// the height alone over it. **Shared**: a partial revolve's two ends, a coil's and a thread's.
pub(crate) fn cap_distance(d: f64, h: f64) -> f64 {
    if d > 0.0 { d.hypot(h) } else { h.abs() }
}

impl ScalarField for RevolvedWedge {
    /// **With rims, each cap's reading is the section's own against that cap's half-plane**
    /// (`round-chamfer-spec.md` §4.3). Inside the wedge the three cases above become one line:
    /// the larger over the two caps of `against(x, y_c)`, with `x = d(r, z)` in the query's own
    /// meridian and `y_c = −r·sin(min(ψ_c, 90°))` the distance in to cap `c`'s half-plane --
    /// which with hard caps is exactly `max(d, to_start, to_end)` and `d` where `d ≥ 0`, so a
    /// hard wedge reads what it always did. Outside, the nearer cap: a hard cap by the landing
    /// read it always had, exact; a rounded one by `join(x, +r·sin(min(ψ_c, 90°)))`, the same
    /// line continued across the cap plane with the sign of `y` turned.
    ///
    /// **Why the rounded cap keeps the query's own radius outside too, and not the landing.**
    /// Each cap's reading, taken that way, is continuous across its own cap plane and
    /// 1-Lipschitz on either side (the distance to a half-plane, and a section distance in
    /// cylindrical coordinates, are both 1-Lipschitz), and reads at most nought on the solid;
    /// so along the straight line from any outside point to any point of the solid, which
    /// enters the wedge through exactly one cap, that cap's reading bounds the distance -- and
    /// the lesser over the two caps bounds it whichever cap the line enters by. A landing read
    /// there would not be: it is the section at a *different* radius from the one the solid's
    /// rim was cut at, and a ring section can read the two apart by up to the reach.
    ///
    /// What the compile holds the reach to, so no two readings meet: the section's distance
    /// from the axis times the sine of the angle each rounded cap has to itself, and of half
    /// the gap the revolve leaves, and the section's depth.
    fn distance(&self, p: Point3<f64>) -> f64 {
        use std::f64::consts::{FRAC_PI_2, TAU};
        let r = p.x.hypot(p.y);
        // The angle on from the wedge's start, nought to a turn.
        let rel = (p.y.atan2(p.x) - self.from).rem_euclid(TAU);
        let x = self.footprint.about_axis(r, p.z);
        if rel <= self.sweep {
            let to_start = -(r * rel.min(FRAC_PI_2).sin());
            let to_end = -(r * (self.sweep - rel).min(FRAC_PI_2).sin());
            return self.against(1, x, to_start).max(self.against(2, x, to_end));
        }
        // Outside: the nearer cap. The angle to each is taken the smaller way round, which is
        // what makes a wedge past a half turn pick the right end for every query.
        let cap = |which: u8, raw: f64| {
            let psi = raw.min(TAU - raw);
            if self.rounded(which) {
                return self.rims.cut(x, r * psi.min(FRAC_PI_2).sin());
            }
            let (u, h) = (r * psi.cos(), r * psi.sin());
            cap_distance(self.footprint.across_axis(u, p.z), h)
        };
        cap(1, TAU - rel).min(cap(2, rel - self.sweep))
    }

    fn normalization(&self) -> Normalization {
        // Case 3 is a bound whatever the section promises.
        Normalization::LowerBound
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

/// What a solid built on a profile can promise about its distance.
///
/// **A baked profile is a lower bound, and saying so is the contract rather than a courtesy.**
/// The grid's reading has a margin subtracted so that interpolation can never overstate, which
/// is exactly the definition of a lower bound — and `ScalarField` says that a field returning
/// one must report it, because a blend mixes by magnitude and an operand reading short pulls the
/// seam toward itself. A field that quietly claimed an exact distance while answering from a
/// grid is the silent wrong shape D8 exists to prevent.
///
/// Exact when nothing is baked, which is what export gets: a fine tolerance does not bake, so
/// the geometry that leaves the application is still a true distance.
///
/// A revolve's comment here used to say something else — that a profile crossing its own axis
/// was refused at compile time, so what arrived was always exact. That refusal was removed on
/// 2026-09-06, because it also refused every profile whose axis had merely been nudged. The
/// folded footprint it now builds is an exact distance to whatever solid results, so the
/// reasoning changed rather than the answer.
pub(crate) fn baked_normalization(profile: &Profile) -> Normalization {
    if profile.is_baked() {
        Normalization::LowerBound
    } else {
        Normalization::Distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(half: f64) -> Profile {
        Profile::new(vec![
            Point2::new(-half, -half),
            Point2::new(half, -half),
            Point2::new(half, half),
            Point2::new(-half, half),
        ])
        .expect("a square is a profile")
    }

    /// Two curves meeting at both ends, plus a stray line, and a document holding them.
    ///
    /// The stray is stamped newest, because that is how it arises: it is the one just drawn.
    fn two_and_a_stray() -> (fieldform_scene::Document, [fieldform_scene::CurveId; 3]) {
        let mut doc = fieldform_scene::Document::new();
        let mut add = |points: Vec<Point3<f64>>, name: &str, when: u64| {
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(id, name.to_string(), points, false)
                    .expect("an open polyline"),
            );
            doc.curves_created.insert(id, when);
            id
        };
        // Down one side and back up the other: open on their own, closed together.
        let a = add(
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
            ],
            "Line",
            1,
        );
        let b = add(
            vec![Point3::new(10.0, 10.0, 0.0), Point3::new(-10.0, 10.0, 0.0),
                 Point3::new(-10.0, -10.0, 0.0)],
            "Line 2",
            2,
        );
        // Well clear of both, touching nothing.
        let stray = add(
            vec![Point3::new(60.0, 60.0, 0.0), Point3::new(60.0, 110.0, 0.0)],
            "Line 3",
            3,
        );
        (doc, [a, b, stray])
    }

    /// A profile straddling the world Z axis still gets an axis it can be revolved about.
    ///
    /// **The case the application was reported broken on**: a curve drawn on a plane away from
    /// the world origin, revolved, and the default axis running straight through it. That is no
    /// longer refused — it revolves the folded footprint instead — but folding a profile onto
    /// itself is still not what somebody drawing a lathe section means, so the *default* axis
    /// should still be one the profile sits beside.
    #[test]
    fn a_profile_straddling_the_world_axis_still_gets_an_axis_to_turn_about() {
        // In the YZ plane, spanning y from -10 to 10, so the world Z axis runs straight
        // through the middle of it.
        let world = vec![
            Point3::new(0.0, -10.0, -5.0),
            Point3::new(0.0, 10.0, -5.0),
            Point3::new(0.0, 10.0, 5.0),
            Point3::new(0.0, -10.0, 5.0),
        ];

        // What the old default did, measured rather than asserted: straight through it, which
        // shows as a half-turn of spread about the axis.
        assert!(
            (Profile::azimuth_span(&world, |p| p) - 180.0).abs() < 1e-6,
            "the world axis cuts this profile in half"
        );

        // And what the fitted one does.
        let (origin, dir) = axis_beside(&world).expect("a planar profile has an axis beside it");
        // In the profile's plane: the plane's normal is X, so the axis must have no X in it.
        assert!(dir.x.abs() < 1e-9, "the axis left the profile's plane: {dir:?}");
        assert!(origin.x.abs() < 1e-9, "the axis origin left the plane: {origin:?}");

        let rotation = crate::rotation_from_z(dir);
        let into = move |p: Point3<f64>| {
            Point3::from(rotation.inverse_transform_vector(&(p - origin)))
        };
        assert!(
            Profile::azimuth_span(&world, into) < 1e-6,
            "the fitted axis leaves the profile in one half-plane"
        );
        Profile::about_axis(&world, into).expect("and it builds");
    }

    /// A square of four lines, a triangle of three, a closed polyline and a stray line, in a
    /// document, stamped in that order — two loops the walk has to find, one curve that is its
    /// own profile, and one that is nothing.
    fn two_loops_a_ring_and_a_stray() -> (
        fieldform_scene::Document,
        [Vec<fieldform_scene::CurveId>; 2],
        fieldform_scene::CurveId,
        fieldform_scene::CurveId,
    ) {
        let mut doc = fieldform_scene::Document::new();
        let mut when = 0;
        let mut add = |points: Vec<Point3<f64>>, closed: bool| {
            when += 1;
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(id, format!("Line {when}"), points, closed)
                    .expect("a polyline"),
            );
            doc.curves_created.insert(id, when);
            id
        };
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let square = vec![
            add(vec![p(0.0, 0.0), p(10.0, 0.0)], false),
            add(vec![p(10.0, 0.0), p(10.0, 10.0)], false),
            add(vec![p(10.0, 10.0), p(0.0, 10.0)], false),
            add(vec![p(0.0, 10.0), p(0.0, 0.0)], false),
        ];
        let triangle = vec![
            add(vec![p(30.0, 0.0), p(40.0, 0.0)], false),
            add(vec![p(40.0, 0.0), p(35.0, 8.0)], false),
            add(vec![p(35.0, 8.0), p(30.0, 0.0)], false),
        ];
        let ring = add(vec![p(60.0, 0.0), p(70.0, 0.0), p(70.0, 10.0), p(60.0, 10.0)], true);
        let stray = add(vec![p(100.0, 100.0), p(120.0, 100.0)], false);
        (doc, [square, triangle], ring, stray)
    }

    /// Every profile row is the loop an extrude of its first curve would read.
    ///
    /// The row and the solid come from one walk, and this is what holds them together: for
    /// every row the partition makes, starting `profile_loop` from the row's first curve — which
    /// is what Extrude is handed when the row is selected — finds exactly those curves in
    /// exactly that order. A partition that agreed on membership but not on order would still
    /// wind a profile the other way round.
    #[test]
    fn every_profile_row_is_the_loop_an_extrude_of_its_first_curve_would_read() {
        let (doc, [square, triangle], _ring, _stray) = two_loops_a_ring_and_a_stray();
        let rows = profile_partition(&doc);
        assert_eq!(rows.len(), 2, "a square and a triangle: {rows:?}");
        for row in &rows {
            let read = profile_loop(&doc, row[0], None).expect("the row's first curve closes");
            assert_eq!(&read.curves, row, "the extrude would read a different loop");
        }
        // And the key is the loop's earliest curve, which is the block list's order.
        assert_eq!(rows[0][0], square[0]);
        assert_eq!(rows[1][0], triangle[0]);
        // Rows follow the list too: the square was drawn first.
        assert_eq!(rows[0].len(), 4);
        assert_eq!(rows[1].len(), 3);
    }

    /// A single closed curve is never a profile row.
    ///
    /// It is its own profile already, and a group of one is a disclosure triangle over nothing
    /// — the reason an import of one solid stays one ordinary row.
    #[test]
    fn a_single_closed_curve_is_never_a_profile_row() {
        let (doc, _, ring, stray) = two_loops_a_ring_and_a_stray();
        let rows = profile_partition(&doc);
        assert!(rows.iter().all(|row| !row.contains(&ring)), "the ring is its own row");
        assert!(rows.iter().all(|row| !row.contains(&stray)), "the stray is its own row");
        // Closed on its own is still a profile: the walk says so, and so does the extrude.
        let alone = profile_loop(&doc, ring, None).expect("closed on its own");
        assert_eq!(alone.curves, vec![ring]);
    }

    /// A curve is in at most one profile row.
    ///
    /// Two loops sharing a side — a rectangle with a line down the middle — are three curves
    /// that can close two ways, and the middle line closes both. The partition places it once:
    /// the first loop to close from the earliest seed takes it, and the other side's curve is
    /// left as its own row rather than shown twice or in a row that is not a loop.
    #[test]
    fn a_curve_is_in_at_most_one_profile_row() {
        let mut doc = fieldform_scene::Document::new();
        let mut when = 0;
        let mut add = |points: Vec<Point3<f64>>| {
            when += 1;
            let id = fieldform_scene::CurveId::new();
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(id, format!("Line {when}"), points, false)
                    .expect("an open polyline"),
            );
            doc.curves_created.insert(id, when);
            id
        };
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        // A U on the left, a U on the right, and the line between them that closes either.
        let left = add(vec![p(0.0, 0.0), p(-10.0, 0.0), p(-10.0, 10.0), p(0.0, 10.0)]);
        let right = add(vec![p(0.0, 10.0), p(10.0, 10.0), p(10.0, 0.0), p(0.0, 0.0)]);
        let middle = add(vec![p(0.0, 0.0), p(0.0, 10.0)]);

        let rows = profile_partition(&doc);
        let mut seen: Vec<fieldform_scene::CurveId> = rows.iter().flatten().copied().collect();
        let total = seen.len();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), total, "a curve appears in two rows: {rows:?}");
        // Exactly one loop forms, and it is the earliest seed's: the left U, which closes
        // through whichever of the other two the walk tries first.
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0][0], left);
        assert!(rows[0].contains(&middle) || rows[0].contains(&right));
        // And whichever of them lost the branch is not in a row at all.
        let out = if rows[0].contains(&middle) { right } else { middle };
        assert!(!rows[0].contains(&out));
    }

    /// A loop's meetings are the ends the walk crossed, and never a branch.
    ///
    /// The square's four corners are four meetings, each naming the end one line arrives on
    /// and the end the next leaves by. A stray line drawn from the first corner makes three
    /// ends there, and that corner stops being a meeting while the other three stay: a joint
    /// there would make the stray part of the square (`profile-group-spec.md` §4.2).
    #[test]
    fn a_loops_meetings_are_the_ends_the_walk_crossed_and_never_a_branch() {
        let (mut doc, [square, _], _, _) = two_loops_a_ring_and_a_stray();
        let meetings = loop_meetings(&doc, &square);
        assert_eq!(meetings.len(), 4, "{meetings:?}");
        // The first line runs (0,0)→(10,0) and the second starts at (10,0): the walk arrives
        // on the first's last point and leaves by the second's first.
        assert!((meetings[0].at - Point3::new(10.0, 0.0, 0.0)).norm() < 1e-9);
        assert_eq!(meetings[0].ends, [(square[0], 1), (square[1], 0)]);
        // And the loop closes back onto the first line's start.
        assert_eq!(meetings[3].ends, [(square[3], 1), (square[0], 0)]);
        for m in &meetings {
            assert_ne!(m.ends[0].0, m.ends[1].0, "a meeting is between two curves");
        }

        let id = fieldform_scene::CurveId::new();
        doc.curves.insert(
            id,
            fieldform_scene::Curve::polyline(
                id,
                "Stray",
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(-30.0, -30.0, 0.0)],
                false,
            )
            .expect("an open line"),
        );
        let meetings = loop_meetings(&doc, &square);
        assert_eq!(meetings.len(), 3, "{meetings:?}");
        assert!(
            meetings.iter().all(|m| m.at.coords.norm() > 1.0),
            "the branched corner gets no meeting: {meetings:?}"
        );
    }

    /// A stray line from a corner does not break the row or the extrude.
    ///
    /// The case `close_the_loop` searches for: a line drawn out from a profile's corner makes a
    /// branch at that corner, and refusing the branch would take the row away from a loop that
    /// still closes. The row stays, the stray stays out of it, and an extrude seeded from the
    /// row's first curve — or from the stray, as drawing it leaves selected — reads the loop.
    #[test]
    fn a_stray_line_from_a_corner_does_not_break_the_row_or_the_extrude() {
        let (mut doc, [square, _], _, _) = two_loops_a_ring_and_a_stray();
        let id = fieldform_scene::CurveId::new();
        doc.curves.insert(
            id,
            fieldform_scene::Curve::polyline(
                id,
                "Stray",
                // From the square's first corner, off to nowhere.
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(-30.0, -30.0, 0.0)],
                false,
            )
            .expect("an open line"),
        );
        doc.curves_created.insert(id, 99);

        let rows = profile_partition(&doc);
        let row = rows.iter().find(|r| r[0] == square[0]).expect("the square is still a row");
        assert_eq!(row.len(), 4, "the stray joined the row: {row:?}");
        assert!(!row.contains(&id));
        assert!(rows.iter().all(|r| !r.contains(&id)), "the stray is in no row");

        // What the row hands an extrude, and what the selection after drawing hands it.
        let from_row = profile_loop(&doc, row[0], None).expect("the row extrudes");
        assert_eq!(&from_row.curves, row);
        let seed = profile_seed(&doc, Some(id)).expect("a loop is there to seed from");
        assert!(row.contains(&seed), "seeded from the row, not the stray");
    }

    /// A stray curve beside a finished loop does not stop the loop being found.
    ///
    /// **The bug this pins was reported from the application seven times in a row**, and the
    /// last shape of it was this: draw two curves that close, extrude — fine. Delete the solid,
    /// draw a third line off in space touching nothing, extrude — refused, naming a gap that
    /// was only the new line's own length. Drawing a curve selects it, so the walk was being
    /// asked to start from the one curve in the document that could not close.
    #[test]
    fn a_stray_curve_does_not_stop_a_closed_loop_being_found() {
        let (doc, [a, b, stray]) = two_and_a_stray();

        // Selected, exactly as the application leaves it after drawing.
        let seed = profile_seed(&doc, Some(stray)).expect("the loop is still there");
        assert!(seed == a || seed == b, "seeded from a curve of the loop");
        assert!(profile_loop(&doc, seed, None).is_ok(), "and it closes");

        // A curve *of* the loop is preferred over searching, so an explicit choice between two
        // separate loops would still be honoured.
        assert_eq!(profile_seed(&doc, Some(b)).expect("closes"), b);
    }

    /// A ten-millimetre square of four lines, and the same four again exactly on top of them and
    /// younger: what pasting an extrude leaves, since a paste lands in place and copies every
    /// curve the block is built on. The document, the originals and the copies, each in order
    /// round the square.
    fn a_square_and_its_copy_in_place() -> (
        fieldform_scene::Document,
        Vec<fieldform_scene::CurveId>,
        Vec<fieldform_scene::CurveId>,
    ) {
        let mut doc = fieldform_scene::Document::new();
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let corners = [p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0), p(0.0, 10.0)];
        let mut side = |n: usize, when: u64| {
            let id = fieldform_scene::CurveId::new();
            let ends = vec![corners[n], corners[(n + 1) % 4]];
            doc.curves.insert(
                id,
                fieldform_scene::Curve::polyline(id, format!("Line {when}"), ends, false)
                    .expect("a line"),
            );
            doc.curves_created.insert(id, when);
            id
        };
        let originals: Vec<_> = (0..4).map(|n| side(n, n as u64 + 1)).collect();
        let copies: Vec<_> = (0..4).map(|n| side(n, n as u64 + 11)).collect();
        (doc, originals, copies)
    }

    /// The area a closed polyline in the ground plane encloses.
    fn ground_area(points: &[Point3<f64>]) -> f64 {
        let twice: f64 = (0..points.len())
            .map(|i| {
                let (a, b) = (points[i], points[(i + 1) % points.len()]);
                a.x * b.y - b.x * a.y
            })
            .sum();
        (twice * 0.5).abs()
    }

    /// **A walk keeps to its own curves where a copy lies exactly on them.**
    ///
    /// The walk searches every open curve in the document, and after a paste every side of the
    /// pasted square lies on a side of the original. It took whichever came first by id, so the
    /// original's walk went round its own four sides about one paste in nineteen, and otherwise
    /// through a mix of its own and the copies or, more often than not, closed with no area, out
    /// along a side and back along its copy. So from every side of either square the walk goes
    /// round that square and measures what a square measures, and the rows are the two squares.
    /// Over many sets of random ids, since the ids were what decided it.
    #[test]
    fn a_walk_keeps_to_its_own_curves_where_a_copy_lies_exactly_on_them() {
        use std::collections::BTreeSet;
        for _ in 0..64 {
            let (doc, originals, copies) = a_square_and_its_copy_in_place();
            for own in [&originals, &copies] {
                let own: BTreeSet<_> = own.iter().copied().collect();
                for &seed in &own {
                    let walked = loop_from(&doc, seed, None).expect("the square closes");
                    let walked: BTreeSet<_> = walked.into_iter().collect();
                    assert_eq!(walked, own, "round its own square");
                    let found = profile_loop(&doc, seed, None).expect("the square closes");
                    let area = ground_area(&found.points);
                    assert!((area - 100.0).abs() < 1e-9, "a square of 100 mm², not {area}");
                }
            }
            let rows: BTreeSet<BTreeSet<_>> = profile_partition(&doc)
                .into_iter()
                .map(|row| row.into_iter().collect())
                .collect();
            let squares: BTreeSet<BTreeSet<_>> =
                [originals, copies].into_iter().map(|s| s.into_iter().collect()).collect();
            assert_eq!(rows, squares, "a row for each square");
        }
    }

    /// **A curve and a copy lying exactly on it are not a loop.**
    ///
    /// Out along one and back along the other comes back to where it started, which is all the
    /// walk asked of a loop, and encloses nothing, so an extrude of it builds nothing and a row of
    /// it is a row of nothing. Either way round, since a curve drawn back over another lies on it
    /// just the same.
    #[test]
    fn a_curve_and_a_copy_lying_exactly_on_it_are_not_a_loop() {
        let bend =
            [Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 20.0, 0.0)];
        for reversed in [false, true] {
            let mut doc = fieldform_scene::Document::new();
            let mut add = |points: Vec<Point3<f64>>, when: u64| {
                let id = fieldform_scene::CurveId::new();
                doc.curves.insert(
                    id,
                    fieldform_scene::Curve::polyline(id, format!("Bend {when}"), points, false)
                        .expect("an open polyline"),
                );
                doc.curves_created.insert(id, when);
                id
            };
            let original = add(bend.to_vec(), 1);
            let mut again = bend.to_vec();
            if reversed {
                again.reverse();
            }
            let copy = add(again, 2);
            for seed in [original, copy] {
                let walked = loop_from(&doc, seed, None);
                assert!(
                    matches!(walked, Err(LoopError::NotClosed { .. })),
                    "reversed {reversed}: {walked:?}"
                );
            }
            assert!(profile_partition(&doc).is_empty(), "and no row");
        }
    }

    /// With nothing that closes, the gap named is the selected curve's, not another's.
    ///
    /// Falling back through the whole document means the *last* refusal is measured somewhere
    /// the person never looked, so the first is the one to keep.
    #[test]
    fn a_document_with_no_loop_reports_the_gap_under_the_selected_curve() {
        let (mut doc, [a, _b, stray]) = two_and_a_stray();
        // Break the loop by removing one of its halves, leaving two curves and no loop at all.
        doc.curves.remove(&_b);

        let err = profile_seed(&doc, Some(stray)).expect_err("nothing closes now");
        let LoopError::NotClosed { gap, .. } = err else { panic!("a gap, not {err:?}") };
        // The stray is 50 long; the remaining half of the square leaves its two ends diagonally
        // opposite, 20 by 20 apart. The two numbers differ, which is the whole point: the one
        // reported says which curve it was measured on.
        assert!((gap - 50.0).abs() < 1e-9, "the selected curve's own gap, got {gap}");

        // And the other way round, to show it is the *preference* being followed and not the
        // stray happening to sort first.
        let err = profile_seed(&doc, Some(a)).expect_err("still nothing closes");
        let LoopError::NotClosed { gap, .. } = err else { panic!("a gap, not {err:?}") };
        assert!((gap - 800f64.sqrt()).abs() < 1e-9, "the other curve's gap, got {gap}");
    }

    /// The distances a square's own geometry fixes, inside and out.
    #[test]
    fn a_square_profile_measures_what_a_square_measures() {
        let p = square(10.0);
        // The centre is ten from every side.
        assert!((p.distance(Point2::new(0.0, 0.0)) + 10.0).abs() < 1e-9);
        // On an edge, nothing.
        assert!(p.distance(Point2::new(10.0, 0.0)).abs() < 1e-9);
        // Straight out from a face.
        assert!((p.distance(Point2::new(15.0, 0.0)) - 5.0).abs() < 1e-9);
        // Diagonally off a corner, where the answer is the hypotenuse rather than the larger
        // of the two — the case a half-plane test gets wrong.
        let d = p.distance(Point2::new(13.0, 14.0));
        assert!((d - 5.0).abs() < 1e-9, "expected 5, got {d}");
    }

    /// An open or degenerate curve is refused rather than guessed at.
    #[test]
    fn a_profile_needs_three_distinct_points() {
        assert!(matches!(
            Profile::new(vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)]),
            Err(ProfileError::TooFewPoints(2)),
        ));
        // A loop closed by repeating its first point is the same loop, not one point longer.
        let repeated = Profile::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.0, 0.0),
        ])
        .expect("closes");
        assert_eq!(repeated.points().len(), 3);
    }

    /// Even–odd, stated as a test because it is a choice and not a consequence.
    #[test]
    fn a_self_crossing_profile_fills_by_even_odd() {
        // A bow tie: two triangles meeting at the origin, traced so the middle is crossed twice.
        let bow = Profile::new(vec![
            Point2::new(-10.0, -5.0),
            Point2::new(-10.0, 5.0),
            Point2::new(10.0, -5.0),
            Point2::new(10.0, 5.0),
        ])
        .expect("a bow tie is closed");
        // Well inside the left lobe.
        assert!(bow.distance(Point2::new(-7.0, 0.0)) < 0.0);
        // And the crossing point itself is on the boundary, not inside it.
        assert!(bow.distance(Point2::new(0.0, 0.0)).abs() < 1e-9);
    }

    /// An extrusion is the profile in the plane and a slab along the normal.
    #[test]
    fn an_extrusion_is_the_profile_within_its_height_and_nothing_beyond() {
        let solid = Extrusion { rims: Default::default(), footprint: square(10.0).into(), start: 0.0, height: 6.0, depth: Depth::One, draft_tangent: 0.0, corners: Default::default() };
        // The middle: three to the nearest face, which is the slab rather than the profile.
        // From the plane rather than about it, so a six-deep solid spans [0, 6] and its
        // middle is at three.
        assert!((solid.distance(Point3::new(0.0, 0.0, 3.0)) + 3.0).abs() < 1e-9);
        // Above the top, straight up.
        assert!((solid.distance(Point3::new(0.0, 0.0, 8.0)) - 2.0).abs() < 1e-9);
        // Outside the profile, level with the middle of the depth.
        assert!((solid.distance(Point3::new(14.0, 0.0, 3.0)) - 4.0).abs() < 1e-9);
        // Off the rim diagonally, where the answer is the hypotenuse: three out from the
        // profile and four above a top that now sits at six rather than at three.
        let d = solid.distance(Point3::new(13.0, 0.0, 10.0));
        assert!((d - 5.0).abs() < 1e-9, "expected 5, got {d}");
    }

    /// A revolved rectangle is a tube, whose walls are where its own numbers put them.
    #[test]
    fn a_revolved_rectangle_is_a_tube_of_those_radii() {
        // From r = 10 to r = 20, and 8 tall about z = 0.
        let wall = Profile::new(vec![
            Point2::new(10.0, -4.0),
            Point2::new(20.0, -4.0),
            Point2::new(20.0, 4.0),
            Point2::new(10.0, 4.0),
        ])
        .expect("a rectangle is a profile");
        let tube = Revolution { footprint: wall.into() };

        // Mid-wall, five from either face.
        assert!((tube.distance(Point3::new(15.0, 0.0, 0.0)) + 4.0).abs() < 1e-9);
        // The bore is hollow, and the same in every direction round.
        for (x, y) in [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (3.5, 3.5)] {
            let d = tube.distance(Point3::new(x, y, 0.0));
            let r = (x as f64).hypot(y);
            assert!((d - (10.0 - r)).abs() < 1e-9, "at r={r} got {d}");
        }
        // And well outside.
        assert!((tube.distance(Point3::new(0.0, 25.0, 0.0)) - 5.0).abs() < 1e-9);
    }

    /// Every field claiming to be a distance has to move at unit rate.
    #[test]
    fn an_extrusion_and_a_revolution_both_move_at_unit_rate() {
        let solid = Extrusion { rims: Default::default(), footprint: square(10.0).into(), start: 0.0, height: 6.0, depth: Depth::One, draft_tangent: 0.0, corners: Default::default() };
        let points: Vec<Point3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64 * 0.37;
                Point3::new(14.0 + t, 3.0 + 0.5 * t, 1.0 + 0.2 * t)
            })
            .collect();
        assert!(crate::gradient_deviation(&solid, &points, 1e-5) < 1e-4);

        let wall = Profile::new(vec![
            Point2::new(10.0, -4.0),
            Point2::new(20.0, -4.0),
            Point2::new(20.0, 4.0),
            Point2::new(10.0, 4.0),
        ])
        .expect("a rectangle is a profile");
        let tube = Revolution { footprint: wall.into() };
        let points: Vec<Point3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64 * 0.29;
                Point3::new(26.0 + t, 1.0 + 0.3 * t, 0.5 + 0.1 * t)
            })
            .collect();
        assert!(crate::gradient_deviation(&tube, &points, 1e-5) < 1e-4);
    }
}

#[cfg(test)]
mod fillet_under_lean_tests {
    use super::*;

    /// The blend this codebase evaluates, copied for the same reason `blend_profile.rs` copies
    /// it: `smooth_min` lives in `graph_field.rs` and is private.
    fn smooth_min(a: f64, b: f64, k: f64) -> f64 {
        let x = (k - a).max(0.0);
        let y = (k - b).max(0.0);
        k.max(a.min(b)) - (x * x + y * y).sqrt()
    }

    /// **What `fillet_under_lean` predicts is what the blend actually does.**
    ///
    /// The function is closed form and the blend is walked, so this is the arithmetic checked
    /// against the surface rather than against itself. Zero lean is the row that matters most:
    /// a fillet of radius `k` there is tangent `k` from the corner on both sides and turns
    /// through `k` throughout, by definition, so anything else means the closed form is wrong.
    #[test]
    fn a_leaning_operand_stretches_a_fillet_rather_than_scaling_it() {
        let k = 5.0;
        for lean in [0.0f64, 10.0, 32.0, 45.0] {
            let s = lean_shortfall(lean);
            let (spread, tight, flat) = fillet_under_lean(k, lean);

            // Where the blend rejoins the body it is filleted against, walked rather than
            // derived: beyond the tangent point the formula *equals* that body.
            let f = |x: f64, y: f64| smooth_min(s * x, y, k);
            let (mut lo, mut hi) = (0.0, 200.0);
            for _ in 0..300 {
                let mid = 0.5 * (lo + hi);
                if f(mid, 0.0).abs() > 1e-9 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            assert!(
                (hi - spread).abs() < 1e-3,
                "at {lean} degrees the blend rejoins at {hi:.4} mm, not the {spread:.4} predicted"
            );

            // And the arc it turns through is the ellipse those semi-axes describe, so its
            // extreme radii are the ones reported.
            assert!(tight <= k + 1e-9 && flat >= k - 1e-9, "the range must straddle the radius");
            assert!(
                (tight * flat - k * k / s).abs() < 1e-6 * (k * k / s).max(1.0),
                "the two extremes must be k*s and k/s^2"
            );
            if lean == 0.0 {
                assert!(
                    (spread - k).abs() < 1e-9
                        && (tight - k).abs() < 1e-9
                        && (flat - k).abs() < 1e-9,
                    "with no lean a {k} mm fillet is {k} mm every way it is measured"
                );
            }
        }
    }
}

#[cfg(test)]
mod chain_tests {
    use super::*;
    use fieldform_scene::{Curve, CurveId, Document};

    fn line(doc: &mut Document, from: (f64, f64), to: (f64, f64)) -> CurveId {
        let id = CurveId::new();
        let points = vec![Point3::new(from.0, from.1, 0.0), Point3::new(to.0, to.1, 0.0)];
        doc.curves.insert(id, Curve::polyline(id, "line", points, false).expect("a line"));
        id
    }

    /// The corners a chain turns at and its two ends, in order — which is what a chain of
    /// straight lines is, however many points each line was sampled to.
    fn corners(points: &[Point3<f64>]) -> Vec<(f64, f64)> {
        let mut out: Vec<(f64, f64)> = Vec::new();
        for (i, p) in points.iter().enumerate() {
            let turns = i == 0 || i + 1 == points.len() || {
                let (a, b) = (p - points[i - 1], points[i + 1] - p);
                a.cross(&b).norm() > 1e-9
            };
            if turns {
                out.push((p.x, p.y));
            }
        }
        out
    }

    /// Left and right of a wall are read walking along it, so the walk has to have a direction
    /// somebody chose: the named curve's. Its neighbours are turned to follow, whichever way
    /// each was drawn.
    #[test]
    fn a_chain_runs_the_way_its_named_curve_does_whichever_way_the_others_were_drawn() {
        let mut doc = Document::new();
        let a = line(&mut doc, (0.0, 0.0), (10.0, 0.0));
        // Drawn from its far end back.
        let b = line(&mut doc, (20.0, 5.0), (10.0, 0.0));
        let c = line(&mut doc, (20.0, 5.0), (20.0, 15.0));

        let from_a = outline_chain(&doc, a, None).expect("a chain");
        assert!(!from_a.closed);
        assert_eq!(corners(&from_a.points), [(0.0, 0.0), (10.0, 0.0), (20.0, 5.0), (20.0, 15.0)]);
        assert_eq!(from_a.curves, [a, b, c]);

        // Named by the one drawn backwards, the same chain runs the other way.
        let from_b = outline_chain(&doc, b, None).expect("a chain");
        assert_eq!(corners(&from_b.points), [(20.0, 15.0), (20.0, 5.0), (10.0, 0.0), (0.0, 0.0)]);
        assert_eq!(from_b.curves, [c, b, a]);
    }

    /// Where three ends meet there are two ways on, and an open chain has no test for which
    /// was meant — a loop tries each and keeps the one that closes, and this has nothing to
    /// close. So it stops, from whichever side it arrives.
    #[test]
    fn a_chain_stops_at_a_fork_rather_than_pick_a_way_on() {
        let mut doc = Document::new();
        let a = line(&mut doc, (0.0, 0.0), (10.0, 0.0));
        let b = line(&mut doc, (10.0, 0.0), (20.0, 0.0));
        let c = line(&mut doc, (10.0, 0.0), (10.0, 10.0));
        for (seed, ends) in [
            (a, [(0.0, 0.0), (10.0, 0.0)]),
            (b, [(10.0, 0.0), (20.0, 0.0)]),
            (c, [(10.0, 0.0), (10.0, 10.0)]),
        ] {
            let chain = outline_chain(&doc, seed, None).expect("a chain");
            assert_eq!(corners(&chain.points), ends);
            assert_eq!(chain.curves, [seed]);
        }
    }

    /// A wall along a loop and a profile of that loop are built from one polyline, because the
    /// chain's walk hands the question to the loop's when the way round comes back.
    #[test]
    fn a_chain_that_comes_back_round_is_the_loop_a_profile_walks() {
        let mut doc = Document::new();
        let sides = [
            line(&mut doc, (0.0, 0.0), (30.0, 0.0)),
            line(&mut doc, (30.0, 0.0), (30.0, 20.0)),
            // Drawn backwards, as one side of a hand-drawn loop usually is.
            line(&mut doc, (0.0, 20.0), (30.0, 20.0)),
            line(&mut doc, (0.0, 20.0), (0.0, 0.0)),
        ];
        for seed in sides {
            let chain = outline_chain(&doc, seed, None).expect("a chain");
            let looped = profile_loop(&doc, seed, None).expect("a loop");
            assert!(chain.closed);
            assert_eq!(chain.points, looped.points);
            assert_eq!(chain.curves, looped.curves);
        }

        // And a curve that closes on its own is its own chain, whatever else touches it.
        let ring = CurveId::new();
        let square = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(5.0, 5.0, 0.0),
            Point3::new(0.0, 5.0, 0.0),
        ];
        doc.curves.insert(ring, Curve::polyline(ring, "ring", square, true).expect("closed"));
        let alone = outline_chain(&doc, ring, None).expect("a chain");
        assert!(alone.closed);
        assert_eq!(alone.curves, [ring]);
    }

    /// The curve a block is aimed along is not part of what it thickens, for a wall as for a
    /// profile: a line drawn from the wall's end to say which way to pull it would otherwise
    /// be walked into the wall.
    #[test]
    fn a_chain_leaves_out_the_curve_it_is_told_to() {
        let mut doc = Document::new();
        let wall = line(&mut doc, (0.0, 0.0), (10.0, 0.0));
        let aim = line(&mut doc, (10.0, 0.0), (10.0, 40.0));
        let walked_on = outline_chain(&doc, wall, None).expect("a chain");
        assert_eq!(walked_on.curves, [wall, aim]);
        let kept_apart = outline_chain(&doc, wall, Some(aim)).expect("a chain");
        assert_eq!(kept_apart.curves, [wall]);
        assert_eq!(corners(&kept_apart.points), [(0.0, 0.0), (10.0, 0.0)]);
    }

    /// **What lies exactly on the curve a walk leaves out is left out with it.**
    ///
    /// A paste copies the line a block is aimed along as well as the block, and lands in place, so
    /// the copy is the same line. Left in, the walk went down it, which is what leaving the aim out
    /// is for.
    #[test]
    fn what_lies_exactly_on_the_curve_a_walk_leaves_out_is_left_out_with_it() {
        let mut doc = Document::new();
        let wall = line(&mut doc, (0.0, 0.0), (10.0, 0.0));
        let aim = line(&mut doc, (10.0, 0.0), (10.0, 40.0));
        let copy = line(&mut doc, (10.0, 0.0), (10.0, 40.0));
        for (when, id) in [wall, aim, copy].into_iter().enumerate() {
            doc.curves_created.insert(id, when as u64 + 1);
        }
        let kept_apart = outline_chain(&doc, wall, Some(aim)).expect("a chain");
        assert_eq!(kept_apart.curves, [wall]);
        assert_eq!(corners(&kept_apart.points), [(0.0, 0.0), (10.0, 0.0)]);
    }

    /// **A chain and a copy lying exactly on it each run their own length.**
    ///
    /// The loop's walk is tried first, and it closed out along the chain and back along the copy,
    /// so a wall pasted in place became a loop of nothing, and so did the wall it was copied from.
    /// Kept from that, the open run would still have stopped at once: it stops at a fork, and
    /// where a chain has a copy on top of it every joint is one. Over many sets of random ids,
    /// since the ids decide the order the curves are tried in.
    #[test]
    fn a_chain_and_a_copy_lying_exactly_on_it_each_run_their_own_length() {
        let zig =
            [((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (10.0, 10.0)), ((10.0, 10.0), (20.0, 10.0))];
        for _ in 0..64 {
            let mut doc = Document::new();
            let mut drawn = Vec::new();
            for (when, (from, to)) in zig.iter().chain(zig.iter()).enumerate() {
                let id = line(&mut doc, *from, *to);
                doc.curves_created.insert(id, when as u64 + 1);
                drawn.push(id);
            }
            let (originals, copies) = drawn.split_at(zig.len());
            for own in [originals, copies] {
                for &seed in own {
                    let chain = outline_chain(&doc, seed, None).expect("a chain");
                    assert!(!chain.closed);
                    assert_eq!(chain.curves, own, "its own three, in order");
                    let along = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0)];
                    assert_eq!(corners(&chain.points), along);
                }
            }
        }
    }

    /// A bent chain has its own plane; a straight one is given one. Both are measured from
    /// midway between the chain's two ends, and neither the plane nor the point is decided by
    /// which way the curve was drawn.
    #[test]
    fn a_chains_plane_is_its_own_where_it_bends_and_upright_where_it_is_straight() {
        // An L standing in the YZ plane.
        let bent = [
            Point3::new(7.0, 0.0, 0.0),
            Point3::new(7.0, 30.0, 0.0),
            Point3::new(7.0, 30.0, 20.0),
        ];
        let reversed: Vec<Point3<f64>> = bent.iter().rev().copied().collect();
        for points in [bent.to_vec(), reversed] {
            let (origin, normal) = chain_plane(&points, None).expect("a plane");
            // Not a point of the chain at all: the L's own middle, off both its legs.
            assert_eq!(origin, Point3::new(7.0, 15.0, 10.0));
            assert!((normal - Vector3::x()).norm() < 1e-12, "{normal:?}");
        }

        let straight = [Point3::new(1.0, 2.0, 0.0), Point3::new(31.0, 42.0, 0.0)];
        let (origin, normal) = chain_plane(&straight, None).expect("a plane");
        assert_eq!(origin, Point3::new(16.0, 22.0, 0.0));
        assert!((normal - Vector3::z()).norm() < 1e-12, "{normal:?}");
        // A sweep straight down keeps its sense: the solid goes the way it was told to.
        let (_, down) = chain_plane(&straight, Some(Vector3::new(0.0, 0.0, -5.0))).expect("a plane");
        assert!((down + Vector3::z()).norm() < 1e-12, "{down:?}");
        // And one along the line itself says nothing about the plane, so the line stands up.
        let along = straight[1] - straight[0];
        let (_, fallen_back) = chain_plane(&straight, Some(along)).expect("a plane");
        assert!((fallen_back - Vector3::z()).norm() < 1e-12, "{fallen_back:?}");

        // A line whose points stray by a rounding is straight, not bent by the rounding.
        let nearly = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(20.0, 1e-11, 0.0),
            Point3::new(40.0, 0.0, 1e-11),
        ];
        let (_, normal) = chain_plane(&nearly, None).expect("a plane");
        assert!((normal - Vector3::z()).norm() < 1e-9, "{normal:?}");

        assert!(chain_plane(&[Point3::origin(), Point3::origin()], None).is_none());
    }

    /// An S whose two lobes are equal encloses no signed area, and it bends as much as any
    /// chain does: its plane is its own, on a vertical drawing plane as on the ground. Judged
    /// by the area it was straight, and was given the ground's plane and refused as not flat
    /// (`examples/inflecting_wall_probe.rs`).
    #[test]
    fn an_s_whose_lobes_cancel_still_lies_in_its_own_plane() {
        // Point-symmetric about (45, 10, 0) in the XZ plane: the signed area is exactly nought.
        let s: Vec<Point3<f64>> = (0..=40)
            .map(|i| {
                let x = 90.0 * i as f64 / 40.0;
                Point3::new(x, 10.0, 25.0 * (x / 90.0 * std::f64::consts::TAU).sin())
            })
            .collect();
        let (origin, normal) = chain_plane(&s, None).expect("a plane");
        assert!((normal - Vector3::y()).norm() < 1e-9, "{normal:?}");
        assert!((origin - Point3::new(45.0, 10.0, 0.0)).norm() < 1e-9, "{origin:?}");
        // And a sweep direction does not override a plane the chain has of its own.
        let (_, normal) = chain_plane(&s, Some(Vector3::z())).expect("a plane");
        assert!((normal - Vector3::y()).norm() < 1e-9, "{normal:?}");
    }
}
