// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a section lies: the profile or wall a block carries, laid in the plane it is read in.

use super::{
    CompileError, bake_to_tolerance, profile_reads_out_to, reads_widened, resolve_frame,
    rotation_from_z, sweep_direction, wall_bakes_under,
};
use fieldform_scene::{Document, Graph, Node, NodeId};
use nalgebra::{Point3, Vector3};
#[cfg(doc)] use super::sweep_offset;

/// Where a section lies: its polyline in the world, and the plane it is read in. An extrude's,
/// which is what named it, and a sweep's -- [`sweep_outline_section`] answers with the same.
#[derive(Debug, Clone)]
pub struct ExtrudeSection {
    /// The loop or the chain, in world coordinates.
    pub points: Vec<Point3<f64>>,
    /// Whether the last point joins back to the first. Always, for a closed profile.
    pub closed: bool,
    /// A point of the plane, which the section's own coordinates are measured from.
    pub origin: Point3<f64>,
    /// The plane's normal, which a straight sweep runs along.
    pub normal: Vector3<f64>,
}

/// Which curves an extrude is built from, and the plane they are read in.
///
/// **One answer, for everything that has to lay the section down**: the field compiles from it,
/// the extent is taken from it, the shader's placement is worked out from it and the polyline
/// the shader walks is flattened in it. Each of those used to walk the loop and fit the plane
/// for itself, which was four chances to do it differently — and a section laid in two places
/// is a solid the field puts in one and the picture in another.
///
/// A closed profile is walked and fitted exactly as it always was. A wall's chain may be open,
/// and may be straight, which is why its plane comes from [`crate::chain_plane`]: a straight
/// line has no plane of its own, and takes one from the direction it is swept in.
pub fn extrude_section(
    document: &Document,
    outline: &fieldform_scene::Outline,
    sweep: &fieldform_scene::Sweep,
    turn_degrees: f64,
) -> Result<ExtrudeSection, CompileError> {
    let resolve = |r: &fieldform_scene::ParamRef| document.resolve(r);
    let toward = sweep_direction(document, sweep, &resolve);
    section_in_its_plane(document, outline, sweep.source(), toward, turn_degrees)
}

/// The same for a section carried along a curve: which curves a sweep's section is, the plane
/// they are read in, and **the point of that plane the path carries** — its origin.
///
/// Two things differ from an extrude in how the plane is found. The spine is what is left out
/// of the walk, since a path that happens to touch its section is not part of it. And a
/// straight line, which has no plane of its own, takes the one squared to **the way the path
/// sets off** — which is the direction an extrude's stated sweep gives it, read off the curve.
///
/// **The origin is where the path sets off from, not the section's centroid — ruled by the
/// owner 2026-10-02.** A sweep carries its section by the origin and turns it about that point,
/// so the origin is the one point of the section that follows the spine; every other point
/// swings about it as the frame turns. The centroid is nobody's choice of that point: a path
/// drawn from a corner of a 500 mm outline round a sixty-degree bend had the corner leave the
/// spine after the first few millimetres and the whole outline swing 400 mm clear of the
/// curve's end, reported as *"geometry does not follow sweeping curve"* and measured by
/// `fieldform-io/examples/sweep_follows_spine.rs`. What every CAD sweep does, and what a hand
/// drawing a path through a profile means, is that the point the path was drawn from is the
/// point that rides it. So the origin is the spine's first point, dropped square onto the
/// section's plane; a path that starts on the plane is carried by exactly where it starts, and
/// one that starts off it is carried by the point beneath it, with the off-plane remainder
/// becoming [`sweep_offset`]. A path drawn away from its section carries the section on an
/// arm, which is what was drawn — and the fold refusal and the bound measure their reach from
/// this point, since it is the point the section turns about.
///
/// Two things this is better at than the centroid, beyond being right: it is a point of the
/// document rather than of a tessellation, so it does not move with how finely the loop is
/// sampled (`loop_plane` walks the loop twice to hold the centroid still); and it is the
/// natural pivot for a taper, which scales the section about the point on the path
/// (`sweep-v2-spec.md` §2).
pub fn sweep_outline_section(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    turn_degrees: f64,
) -> Result<ExtrudeSection, CompileError> {
    let start = spine_start(document, spine);
    let sets_off = start.and_then(|first| {
        let path = document.curves.get(&spine)?.tessellate(0.1);
        path.iter().map(|p| p - first).find_map(|d| d.try_normalize(1e-9))
    });
    let mut laid = section_in_its_plane(document, outline, Some(spine), sets_off, turn_degrees)?;
    if let Some(start) = start {
        let off = (start - laid.origin).dot(&laid.normal);
        laid.origin = start - laid.normal * off;
    }
    Ok(laid)
}

/// Where a spine sets off from: its first point, read once here for everything that has to
/// agree about it — the section's origin, the way the path sets off, and the offset.
pub(crate) fn spine_start(document: &Document, spine: fieldform_scene::CurveId) -> Option<Point3<f64>> {
    document.curves.get(&spine)?.tessellate(0.1).first().copied()
}

/// Where a section lies, for an extrude and for a sweep alike: walked from the curve the
/// outline names, leaving `ignore` out, and laid in its own plane — or, for a straight line,
/// the plane squared to `toward`.
fn section_in_its_plane(
    document: &Document,
    outline: &fieldform_scene::Outline,
    ignore: Option<fieldform_scene::CurveId>,
    toward: Option<Vector3<f64>>,
    turn_degrees: f64,
) -> Result<ExtrudeSection, CompileError> {
    match outline {
        fieldform_scene::Outline::Closed { profile } => {
            let points = crate::profile_loop_turning(document, *profile, turn_degrees, ignore)
                .map_err(|e| CompileError::BadLoop(*profile, e))?
                .points;
            let (origin, normal) = crate::plane_of(&points)
                .ok_or(CompileError::BadProfile(*profile, crate::ProfileError::NotClosed))?;
            Ok(ExtrudeSection { points, closed: true, origin, normal })
        }
        fieldform_scene::Outline::Wall { curve, .. } => {
            let chain = crate::outline_chain_turning(document, *curve, turn_degrees, ignore)
                .map_err(|e| CompileError::BadLoop(*curve, e))?;
            let plane = if chain.closed {
                loop_plane(document, *curve, ignore, &chain, turn_degrees)
            } else {
                crate::chain_plane(&chain.points, toward)
            };
            let (origin, normal) = plane.ok_or(CompileError::BadWall(
                *curve,
                crate::WallError::TooFewPoints(chain.points.len()),
            ))?;
            Ok(ExtrudeSection { points: chain.points, closed: chain.closed, origin, normal })
        }
    }
}

/// The plane of a wall that runs round a loop: the one a profile of that loop is laid in.
///
/// **So that a tube rides its path where the rod it was drawn from does.** A sweep carries a
/// section by its plane's origin and turns it about that point, and a loop given a thickness is
/// the same loop: turning Wall on must not move where it is carried from. A profile's origin is
/// the centroid of its points, so that is what a loop's wall takes.
///
/// **Read at the compile's fineness whatever this was asked at.** That centroid moves with how
/// finely the loop is sampled, and a wall is laid twice, at the compile's fineness for the
/// field and at the viewport's for the polylines the shader walks. Two origins would be one
/// wall in two places. So the loop is walked a second time where the two differ, which is the
/// cost of a tessellation on the path that uploads one anyway.
fn loop_plane(
    document: &Document,
    curve: fieldform_scene::CurveId,
    ignore: Option<fieldform_scene::CurveId>,
    chain: &crate::Chain,
    turn_degrees: f64,
) -> Option<(Point3<f64>, Vector3<f64>)> {
    let compiled_at = crate::profile::PROFILE_TURN_DEGREES;
    if turn_degrees == compiled_at {
        return crate::plane_of(&chain.points);
    }
    let as_compiled = crate::outline_chain_turning(document, curve, compiled_at, ignore).ok()?;
    crate::plane_of(&as_compiled.points)
}

/// A wall laid in the half-plane about an axis, from its chain in the world: a thin revolve's
/// section. `into_frame` carries a world point into the frame whose `z` is the axis.
///
/// **By the map a profile is laid by** ([`crate::Profile::radial_footprint`]), so a curve
/// revolves as a wall where it would revolve as a profile, whether or not it lies in a plane
/// through the axis. Left and right are then as the curve is seen *in that half-plane*, with
/// the axis running up and the radius out to the right: a curve drawn upward has the axis on
/// its left.
pub(crate) fn wall_about_axis(
    chain: &crate::Chain,
    curve: fieldform_scene::CurveId,
    thickness: f64,
    side: fieldform_scene::WallSide,
    ends: fieldform_scene::WallEnds,
    into_frame: impl Fn(Point3<f64>) -> Point3<f64>,
) -> Result<crate::Wall, CompileError> {
    let flat = crate::Profile::radial_footprint(&chain.points, chain.closed, into_frame);
    let built = if chain.closed {
        let profile =
            crate::Profile::new(flat).map_err(|e| CompileError::BadProfile(curve, e))?;
        crate::Wall::closed(profile, thickness, side)
    } else {
        crate::Wall::open(flat, thickness, side, ends)
    };
    built.map_err(|e| CompileError::BadWall(curve, e))
}

/// A wall built on a section already laid in its plane.
pub(crate) fn wall_in_plane(
    section: &ExtrudeSection,
    curve: fieldform_scene::CurveId,
    thickness: f64,
    side: fieldform_scene::WallSide,
    ends: fieldform_scene::WallEnds,
    flatness_mm: f64,
) -> Result<crate::Wall, CompileError> {
    let rotation = rotation_from_z(section.normal);
    let local: Vec<Vector3<f64>> = section
        .points
        .iter()
        .map(|p| rotation.inverse_transform_vector(&(p - section.origin)))
        .collect();
    // A curve that strays from its plane has no thickness *across* itself to speak of: the
    // band would be measured in a plane the curve is not in.
    let stray = local.iter().map(|p| p.z.abs()).fold(0.0f64, f64::max);
    if stray > flatness_mm {
        return Err(CompileError::BadWall(curve, crate::WallError::NotPlanar(stray)));
    }
    let flat: Vec<nalgebra::Point2<f64>> =
        local.iter().map(|p| nalgebra::Point2::new(p.x, p.y)).collect();
    let built = if section.closed {
        let profile =
            crate::Profile::new(flat).map_err(|e| CompileError::BadProfile(curve, e))?;
        crate::Wall::closed(profile, thickness, side)
    } else {
        crate::Wall::open(flat, thickness, side, ends)
    };
    built.map_err(|e| CompileError::BadWall(curve, e))
}

/// The wall a block is built on, for a caller that has to hand it to a shader: a thin
/// extrude's or a sweep's in its own plane, a thin revolve's in the half-plane about its axis.
///
/// **One function for the three blocks that take an outline**, which is the point of their
/// taking the same type: a caller binding a wall's polylines has one thing to ask, and does
/// not have to know which block it is asking for.
///
/// `None` when the block has no wall, or it cannot be built — which the compile refuses by
/// name, so a caller here has only to say that nothing could be supplied. **Not held to
/// flatness**, for that reason: the compile is, at the tolerance it is working to, and the
/// application draws nothing when it refuses.
pub fn outline_wall(
    document: &Document,
    graph: &Graph,
    node: NodeId,
    turn_degrees: f64,
) -> Option<crate::Wall> {
    block_wall(document, graph, graph.node(node)?, turn_degrees)
}

/// The miter vectors of the section a block reads: where each of its corners goes per
/// millimetre of draft, in step with the polyline that block's section is uploaded as.
///
/// **One thing for every supplier to call**, as `outline_wall` is for a wall: the viewport and
/// the shader-versus-field harness both have to hand the shader these, and a second derivation
/// would be a section displaced two ways. `None` when the block has no section to speak of.
///
/// For a wall it is the *chain's* miters, because a wall round a loop is that loop's distance
/// folded about a band -- so what moves under a draft is the loop.
pub fn outline_miters(
    document: &Document,
    graph: &Graph,
    node: NodeId,
    turn_degrees: f64,
) -> Option<Vec<Point3<f64>>> {
    let block = graph.node(node)?;
    let numbers = |r: &fieldform_scene::ParamRef| document.resolve(r);
    let laid = match block {
        Node::Extrude { outline, sweep, .. } => {
            extrude_section(document, outline, sweep, turn_degrees).ok()?
        }
        Node::Sweep { outline, spine, .. } => {
            sweep_outline_section(document, outline, *spine, turn_degrees).ok()?
        }
        _ => return None,
    };
    let _ = numbers;
    let rotation = rotation_from_z(laid.normal);
    let translation = laid.origin.coords;
    let into = move |p: Point3<f64>| {
        Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
    };
    let profile = crate::Profile::in_plane(&laid.points, into, f64::INFINITY).ok()?;
    Some(profile.miters().iter().map(|m| Point3::new(m.x, m.y, 0.0)).collect())
}

/// [`outline_wall`], for a block that need not be in the graph: the wall `block` would be built
/// on if it were put there.
///
/// **What a panel asks before it commits an edit.** Which side of a loop is its inside depends
/// on the way the loop is walked, and that on which of its curves the block names — so a panel
/// about to hand a block another curve has to ask of the block *as it would be*, and the graph
/// only holds it as it is. `graph` is still wanted, because a revolve's axis may stand on
/// another block.
pub fn block_wall(
    document: &Document,
    graph: &Graph,
    block: &Node,
    turn_degrees: f64,
) -> Option<crate::Wall> {
    let numbers = |r: &fieldform_scene::ParamRef| document.resolve(r);
    let fieldform_scene::Outline::Wall { curve, thickness, side, ends } = block.outline()? else {
        return None;
    };
    let thickness = numbers(thickness)?;
    match block {
        Node::Extrude { outline, sweep, .. } => {
            let section = extrude_section(document, outline, sweep, turn_degrees).ok()?;
            wall_in_plane(&section, *curve, thickness, *side, *ends, f64::INFINITY).ok()
        }
        Node::Sweep { outline, spine, .. } => {
            let section = sweep_outline_section(document, outline, *spine, turn_degrees).ok()?;
            wall_in_plane(&section, *curve, thickness, *side, *ends, f64::INFINITY).ok()
        }
        Node::Revolve { frame, .. } => {
            let chain =
                crate::outline_chain_turning(document, *curve, turn_degrees, frame.curve_source())
                    .ok()?;
            let (translation, rotation) = resolve_frame(document, graph, frame, &numbers).ok()?;
            let into = move |p: Point3<f64>| {
                Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
            };
            wall_about_axis(&chain, *curve, thickness, *side, *ends, into).ok()
        }
        _ => None,
    }
}

/// [`outline_wall`]'s wall, answering from a grid where `tolerance_mm` allows one: what a caller
/// uploads when the picture is to match a field compiled at that tolerance.
///
/// **Baked by the compile's own rule and nothing else** — `bake_to_tolerance`, reading out as
/// far as `profile_reads_out_to` says this node is read, and `also_mm` further for whatever the
/// caller reads that the field does not (`reads_widened`) — because a grid on one side and a
/// walk on the other, or two grids of different fineness, are a surface in two places. A
/// tolerance too fine to bake at comes back unbaked, exactly as the compile leaves it.
pub fn outline_wall_baked(
    document: &Document,
    graph: &Graph,
    node: NodeId,
    turn_degrees: f64,
    tolerance_mm: f64,
    also_mm: f64,
) -> Option<crate::Wall> {
    let wall = outline_wall(document, graph, node, turn_degrees)?;
    // A sweep under a section law is walked, as the compile walks it -- `wall_bakes_under`.
    if let Some(Node::Sweep { law, .. }) = graph.node(node)
        && !wall_bakes_under(law)
    {
        return Some(wall);
    }
    Some(bake_to_tolerance(
        wall,
        tolerance_mm,
        reads_widened(profile_reads_out_to(document, graph, node), also_mm),
        None,
    ))
}

/// Where a revolve's wedge begins and how far it turns, in degrees: the start as an angle from
/// the section's own half-plane -- nought for one way, back by half for half each way, back by
/// side two's angle for two -- and the total. **Refused by the number** when the total is
/// nothing or less: a revolve through no angle is no solid, and one through a negative angle
/// is a sign somebody typed by mistake. A total of a full turn or more is the whole solid, and
/// the compile reads it exactly as it always did.
pub fn revolve_wedge(
    document: &Document,
    angle: &fieldform_scene::ParamRef,
    sides: &fieldform_scene::AngleSides,
) -> Result<(f64, f64), CompileError> {
    let resolve = |r: &fieldform_scene::ParamRef| {
        document.resolve(r).ok_or(CompileError::UnresolvedRadius)
    };
    let one = resolve(angle)?;
    let (from, total) = match sides {
        fieldform_scene::AngleSides::One => (0.0, one),
        fieldform_scene::AngleSides::Symmetric => (-one * 0.5, one),
        fieldform_scene::AngleSides::Two { angle } => {
            let two = resolve(angle)?;
            (-two, one + two)
        }
    };
    if !(total > 0.0) {
        return Err(CompileError::RevolveTurnsNothing(total));
    }
    Ok((from, total))
}

/// How far an extrude's sweep leans off its profile's normal, in degrees.
///
/// Zero for a straight sweep, and `None` for a node that is not an extrude or whose profile
/// cannot be read. Feeds [`crate::lean_shortfall`], which turns it into the number the panel
/// shows -- the two are separate because one is a fact about *this* block and the other is a
/// measured property of the field, and only the first changes as the block is edited.
pub fn sweep_lean_degrees(document: &Document, graph: &Graph, node: NodeId) -> Option<f64> {
    let Node::Extrude { outline, sweep, .. } = graph.node(node)? else {
        return None;
    };
    if matches!(sweep, fieldform_scene::Sweep::Normal) {
        return Some(0.0);
    }
    let normal =
        extrude_section(document, outline, sweep, crate::profile::PROFILE_TURN_DEGREES).ok()?.normal;
    let resolve = |r: &fieldform_scene::ParamRef| document.resolve(r);
    let dir = sweep_direction(document, sweep, &resolve)?;
    Some(dir.dot(&normal).abs().clamp(0.0, 1.0).acos().to_degrees())
}
