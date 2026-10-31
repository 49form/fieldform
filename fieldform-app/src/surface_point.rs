// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Points on a surface: the solve that answers a `SurfacePoint`'s question, the refresh that keeps
//! every one of them answered, and the click, the drag and the panel that ask it.
//!
//! **Everything that goes wrong here is silent.** A point that slides 3 mm when an unrelated boss is
//! added, or hops to the inside of a thin wall, still draws a plausible dot on the surface, and
//! whatever is built on it inherits the error. So the two rules below are what this file is for: the
//! question is measured in a frame that knows nothing of the body's extent, and where it has several
//! answers the point keeps to the one nearest where it stood, facing the same way.
//!
//! **An error never moves a point.** It stays where it last stood and says why, so what is built on
//! it does not jump while the cause is being fixed.
//!
//! **The gesture is the whole question.** A click leaves a [`Guide::Spot`]: where it was put and the
//! way the surface faced there. A drag slides the point over the surface under the cursor and leaves
//! another. Nothing about that asks for an axis, a U and a V, or which face of a wall is meant, and
//! the panel does not offer them: the first build of this tool put the solve's every control in front
//! of a click, and the owner found it unintuitive (2026-10-30). The ray and the seed are still here
//! for a place typed as numbers from the console, and their numbers show in the panel of a point made
//! that way, and that is all.

use super::*;
use fieldform_field::{
    body_frame, crease_angle, crossings, placement_above, resolve_frame, surface_normal,
    walk_to_surface, with_turns_flattened, LeafCache, SearchError, CREASE_ANGLE_DEG,
};
use fieldform_scene::{across, Facing, Frame, Guide, GuideFrame, Solved, SurfacePoint};
use nalgebra::Similarity3;

/// How much further the second-nearest answer may be than the nearest and still count as a tie.
const TIE_FACTOR: f64 = 1.05;
/// How shallow a line may meet the surface before a tiny edit could move the answer a long way.
const GRAZE_DEGREES: f64 = 5.0;
/// How near a point's dot a press has to land to take hold of it, in pixels: a control point's reach.
const GRAB_PX: f32 = 10.0;

/// What a solve found.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Solution {
    pub at: Point3<f64>,
    pub solved: Solved,
    /// Written anyway, and said: a tie, a grazing line or a crease.
    pub warning: Option<String>,
    /// Which of the walls along the line this is, and how many there are.
    pub crossing: Option<(usize, usize)>,
    /// What "moved" is measured against when this is written back.
    pub tolerance: f64,
}

/// What the panel and the viewport say about a point, kept from its last solve.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PointState {
    /// The reason it is red: it was left where it last stood.
    pub error: Option<String>,
    pub warning: Option<String>,
    pub crossing: Option<(usize, usize)>,
}

/// Whether a click leaves a point, or a plane standing on one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stand {
    Point,
    Plane,
}

/// A unit vector along an axis.
fn axis_vector(index: usize) -> Vector3<f64> {
    let mut e = Vector3::zeros();
    e[index] = 1.0;
    e
}

/// The similarity a record's millimetres are measured in, in world.
pub(crate) fn guide_frame(doc: &Document, graph: &Graph, record: &SurfacePoint) -> Result<Similarity3<f64>, String> {
    let resolve = |r: &ParamRef| doc.resolve(r);
    match record.frame {
        GuideFrame::Body => body_frame(doc, graph, record.body, &resolve)
            .map_err(|_| "where the body stands names a variable that has no value".to_string()),
        GuideFrame::Plane { node } => {
            if graph.node(node).is_none() {
                return Err("the plane it is measured from is gone".to_string());
            }
            let flat = with_turns_flattened(doc, graph, &resolve);
            let (origin, rotation) = resolve_frame(doc, &flat, &Frame::From { node }, &resolve)
                .map_err(|_| "the plane it is measured from cannot be read".to_string())?;
            Ok(Similarity3::from_parts(
                nalgebra::Translation3::from(origin),
                nalgebra::UnitQuaternion::from_rotation_matrix(&rotation),
                1.0,
            ))
        }
    }
}

/// The frame's axes in world, unit length.
fn frame_axes(frame: &Similarity3<f64>) -> [Vector3<f64>; 3] {
    [0, 1, 2].map(|i| frame.isometry.rotation * axis_vector(i))
}

/// The roll about a normal: the frame's first across axis laid into the tangent plane, the second
/// where the first lies along the normal.
fn roll_about(normal: Vector3<f64>, axes: &[Vector3<f64>; 3], first_across: (usize, usize)) -> Vector3<f64> {
    let lay = |axis: Vector3<f64>| axis - normal * axis.dot(&normal);
    let laid = lay(axes[first_across.0]);
    let x = if laid.norm() < 1e-3 { lay(axes[first_across.1]) } else { laid };
    x.try_normalize(1e-12).unwrap_or_else(|| lay(axes[0]).normalize())
}

/// A body made ready to be asked: its field, the box that bounds the search, and the frames.
pub(crate) struct Asked {
    field: GraphField,
    bounds: (Point3<f64>, Point3<f64>),
    diagonal: f64,
    tolerance: f64,
    step: f64,
    frame: Similarity3<f64>,
    /// The placement above the body. Its field is compiled on its own, so it reads in the space
    /// under this: a question goes down through it and the answer comes back up.
    above: Similarity3<f64>,
}

fn ask(doc: &Document, graph: &Graph, record: &SurfacePoint, leaves: &mut LeafCache) -> Result<Asked, String> {
    let resolve = |r: &ParamRef| doc.resolve(r);
    let body = graph.subtree(record.body).ok_or_else(|| "the body it stands on is gone".to_string())?;
    // **The extent bounds the search and never places anything**, and that is the whole difference
    // from an axis held in fractions of a box.
    let (lo, hi) = fieldform_field::graph_extent(doc, &body, record.body).as_box().ok_or_else(|| {
        "the body it stands on has no limit to stand against — it is empty, or it reaches forever"
            .to_string()
    })?;
    let diagonal = (hi - lo).norm();
    if diagonal < 1e-9 {
        return Err("the body it stands on has no size".to_string());
    }
    let tolerance = (diagonal * 1e-3).max(1e-4);
    let field = GraphField::compile_cached(doc, &body, tolerance, leaves)
        .map_err(|e| format!("the body it stands on cannot be evaluated: {e}"))?;
    let frame = guide_frame(doc, graph, record)?;
    let flat = with_turns_flattened(doc, graph, &resolve);
    let above = placement_above(doc, &flat, record.body, None, &resolve);
    Ok(Asked { field, bounds: (lo, hi), diagonal, tolerance, step: (diagonal * 1e-4).max(1e-6), frame, above })
}

impl Asked {
    /// Every crossing of the whole line through `through` along `way`, both in the frame, that
    /// `facing` counts, in order along the line, in the body's own space, with the line's direction.
    #[allow(clippy::type_complexity)]
    fn counted_along(
        &self,
        through: Point3<f64>,
        way: Vector3<f64>,
        facing: Facing,
    ) -> Result<(Vec<Point3<f64>>, Vector3<f64>), String> {
        let below = self.above.inverse();
        let direction = (below.isometry.rotation * (self.frame.isometry.rotation * way))
            .try_normalize(1e-12)
            .ok_or_else(|| "it has no direction to look along".to_string())?;
        // **From behind the whole body, not from the frame's plane.** A frame's origin is often
        // inside the body and `crossings` searches forward, so a line started there would skip every
        // wall behind it, and the question is the whole line.
        let on_line = below * (self.frame * through);
        let centre = nalgebra::center(&self.bounds.0, &self.bounds.1);
        let origin = on_line - direction * ((on_line - centre).norm() + self.diagonal * 2.0);
        let all = crossings(&self.field, origin, direction, self.bounds, self.tolerance).map_err(|e| match e {
            SearchError::TooDense => {
                "the line crosses the surface too many times to choose — move the point to a plainer \
                 part of the surface"
                    .to_string()
            }
        })?;
        Ok((all.iter().filter(|c| facing.counts(c.entering)).map(|c| c.point).collect(), direction))
    }

    /// Every wall a line guide could stand on, in the body's own space, with the line's direction.
    #[allow(clippy::type_complexity)]
    fn counted(&self, guide: &Guide, doc: &Document) -> Result<(Vec<Point3<f64>>, Vector3<f64>), String> {
        let unnamed = || "a number of its place names a variable that has no value".to_string();
        match guide {
            Guide::Ray { along, flipped, u, v, facing } => {
                let (u, v) = (doc.resolve(u).ok_or_else(unnamed)?, doc.resolve(v).ok_or_else(unnamed)?);
                let (a, b) = across(*along);
                let mut through = Point3::origin();
                through[a] = u;
                through[b] = v;
                // What across leaves: the three indices sum to 3.
                let way = axis_vector(3 - a - b) * if *flipped { 1.0 } else { -1.0 };
                let (counted, direction) = self.counted_along(through, way, *facing)?;
                if counted.is_empty() {
                    let face = if *facing == Facing::Front { "front" } else { "back" };
                    return Err(format!(
                        "the ray at U {u:.3}, V {v:.3} meets no {face} face — move U or V, or look along another axis"
                    ));
                }
                Ok((counted, direction))
            }
            // Into the material, so the surface facing `facing` is the one the line enters. An
            // empty list is answered rather than refused: the solve has one more place to look
            // (the spot itself) before it says so.
            Guide::Spot { at, facing } => {
                let through = self.spot_of(at, doc)?;
                self.counted_along(through, -Vector3::from(*facing), Facing::Front)
            }
            Guide::Nearest { .. } => Err("a seed has no crossings to choose among".to_string()),
        }
    }

    /// A spot's numbers resolved, in the frame.
    fn spot_of(&self, at: &[ParamRef; 3], doc: &Document) -> Result<Point3<f64>, String> {
        let mut through = Point3::origin();
        for (i, n) in at.iter().enumerate() {
            through[i] = doc.resolve(n).ok_or_else(|| "a number of its place names a variable that has no value".to_string())?;
        }
        Ok(through)
    }

    /// The surface nearest `target` (in world), in world, and the way it faces there.
    fn nearest(&self, target: Point3<f64>) -> Option<(Point3<f64>, Vector3<f64>)> {
        let hit = walk_to_surface(&self.field, self.above.inverse() * target, self.tolerance, self.step)?;
        let normal = surface_normal(&self.field, hit, self.step)?;
        Some((self.above * hit, (self.above.isometry.rotation * normal).normalize()))
    }

    /// Whether a world point lies on this body's surface, to the field's own tolerance.
    fn holds(&self, at: Point3<f64>) -> bool {
        self.field.distance(self.above.inverse() * at).abs() <= 4.0 * self.tolerance
    }
}

/// The walls a line guide could stand on, in world, for the panel's buttons to step between.
pub(crate) fn surface_point_crossings(
    doc: &Document,
    graph: &Graph,
    record: &SurfacePoint,
    leaves: &mut LeafCache,
) -> Result<Vec<Point3<f64>>, String> {
    let asked = ask(doc, graph, record, leaves)?;
    let (counted, _) = asked.counted(&record.guide, doc)?;
    Ok(counted.into_iter().map(|p| asked.above * p).collect())
}

/// The answer to `record`'s question.
///
/// `stored` is where the point stands, in world, where it has stood anywhere yet: the solve's whole
/// memory, and read only to choose among answers. `None` is a point made from typed numbers, whose
/// first answer is the wall nearest its spot, or the first along its ray.
pub(crate) fn solve_surface_point(
    doc: &Document,
    graph: &Graph,
    record: &SurfacePoint,
    stored: Option<Point3<f64>>,
    leaves: &mut LeafCache,
) -> Result<Solution, String> {
    let resolve = |r: &ParamRef| doc.resolve(r);
    let asked = ask(doc, graph, record, leaves)?;
    let Asked { tolerance, step, frame, above, .. } = asked;
    let field = &asked.field;
    let below = above.inverse();
    let stored_below = stored.map(|p| below * p);

    let mut warnings: Vec<String> = Vec::new();
    let mut crossing = None;
    let mut grazed_along = None;
    let hit = match &record.guide {
        guide @ (Guide::Ray { .. } | Guide::Spot { .. }) => {
            let (counted, direction) = asked.counted(guide, doc)?;
            let spot_below = match guide {
                Guide::Spot { at, .. } => Some(below * (frame * asked.spot_of(at, doc)?)),
                _ => None,
            };
            // **A spot on an edge is its own answer.** Its facing there is one face's or an
            // average, and a line along either runs in a face rather than through it, so it
            // crosses nothing: a corner typed to the millimetre, or clicked to the pixel. The spot
            // is on the surface all the same, so it stands there and says why; a spot whose face
            // has gone from under it is the error below.
            if counted.is_empty()
                && let Some(spot) = spot_below
            {
                let on = walk_to_surface(field, spot, tolerance, step)
                    .filter(|p| (p - spot).norm() <= 4.0 * tolerance)
                    .ok_or_else(|| {
                        let world = above * spot;
                        format!(
                            "nothing faces that way under ({:.1}, {:.1}, {:.1}) any more — drag it onto the surface, \
                             or type where it should stand",
                            world.x, world.y, world.z
                        )
                    })?;
                warnings.push(
                    "it is on an edge, where its line runs along the surface, so it stands at the spot itself".to_string(),
                );
                on
            } else {
                // A spot with no memory yet remembers where it was put; a ray takes its first wall.
                let anchor = stored_below.or(spot_below);
                let chosen = match anchor {
                    None => 0,
                    Some(p) => {
                        let mut by_distance: Vec<(usize, f64)> =
                            counted.iter().enumerate().map(|(i, c)| (i, (c - p).norm())).collect();
                        by_distance.sort_by(|x, y| x.1.total_cmp(&y.1));
                        if let [(nearest, near), (second, far), ..] = by_distance[..]
                            && far <= near * TIE_FACTOR
                            && (counted[nearest] - counted[second]).norm() > 10.0 * tolerance
                        {
                            warnings.push(
                                "two walls are about equally near where it stood — it kept the nearer".to_string(),
                            );
                        }
                        by_distance[0].0
                    }
                };
                crossing = Some((chosen + 1, counted.len()));
                grazed_along = Some(direction);
                counted[chosen]
            }
        }
        Guide::Nearest { at } => {
            let unnamed = || "a number of its seed names a variable that has no value".to_string();
            let mut seed = Point3::origin();
            for (i, n) in at.iter().enumerate() {
                seed[i] = resolve(n).ok_or_else(unnamed)?;
            }
            let seed = below * (frame * seed);
            let unsettled = || {
                "no nearest surface point settles from that seed — it may be equally far from two \
                 walls; move it toward the one you mean"
                    .to_string()
            };
            let from_seed = walk_to_surface(field, seed, tolerance, step).ok_or_else(unsettled)?;
            // Near the middle of a wall the seed is about equally near both sides, and the walk from
            // where the point stands says which one it was on.
            match stored_below.and_then(|p| walk_to_surface(field, p, tolerance, step)) {
                Some(kept)
                    if (kept - from_seed).norm() > 10.0 * tolerance
                        && (seed - kept).norm() <= TIE_FACTOR * (seed - from_seed).norm() =>
                {
                    warnings.push(
                        "the seed is about equally near two walls — it stayed on the wall it was on"
                            .to_string(),
                    );
                    kept
                }
                _ => from_seed,
            }
        }
    };

    let normal = surface_normal(field, hit, step)
        .ok_or_else(|| "the field has no direction there — it is flat, or on a crease".to_string())?;
    if let Some(d) = grazed_along
        && normal.dot(&d).abs() < GRAZE_DEGREES.to_radians().sin()
    {
        warnings.push("the surface here lies almost along the way it was put, so a small edit can move it a long way".to_string());
    }
    if crease_angle(field, hit, normal, tolerance, step).is_some_and(|angle| angle > CREASE_ANGLE_DEG) {
        warnings.push("it is on a crease, so its normal is an average of the faces".to_string());
    }

    let world_normal = (above.isometry.rotation * normal).normalize();
    let first_across = match &record.guide {
        Guide::Ray { along, .. } => across(*along),
        Guide::Nearest { .. } | Guide::Spot { .. } => (0, 1),
    };
    let x = roll_about(world_normal, &frame_axes(&frame), first_across);

    Ok(Solution {
        at: above * hit,
        solved: Solved {
            normal: [world_normal.x, world_normal.y, world_normal.z],
            x: [x.x, x.y, x.z],
        },
        warning: (!warnings.is_empty()).then(|| warnings.join("; ")),
        crossing,
        tolerance,
    })
}

/// Whether what a solve found is worth writing over what is stored.
///
/// **Without a threshold the refresh could keep finding work**, since the position is part of the
/// graph and an answer one bisection different is an edit.
fn worth_writing(stored: Option<(Point3<f64>, Solved)>, found: &Solution) -> bool {
    let Some((at, solved)) = stored else { return true };
    let turned = Vector3::from(solved.normal).dot(&Vector3::from(found.solved.normal)).clamp(-1.0, 1.0).acos();
    (at - found.at).norm() > found.tolerance * 1e-2 || turned > 1e-6
}

/// Whether `point` reads itself: whether what it stands on, or what that stands on through any other
/// surface point, is built on it. Such a point has no answer to settle on.
pub(crate) fn reads_itself(doc: &Document, graph: &Graph, point: NodeId) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut walk = vec![point];
    while let Some(id) = walk.pop() {
        let Some(record) = doc.surface_points.get(&id) else { continue };
        for source in record.sources() {
            for block in graph.subtree_of(source) {
                if block == point {
                    return true;
                }
                if seen.insert(block) {
                    walk.push(block);
                }
            }
        }
    }
    false
}

/// The live points in the order they must be solved: a point on a body that stands on another
/// surface point comes after that point, so one pass is enough.
fn solve_order(doc: &Document) -> Vec<NodeId> {
    fn visit(
        doc: &Document,
        id: NodeId,
        seen: &mut std::collections::HashSet<NodeId>,
        out: &mut Vec<NodeId>,
    ) {
        if !seen.insert(id) {
            return;
        }
        if let Some(record) = doc.surface_points.get(&id) {
            let graph = doc.parts.values().map(|p| &p.graph).find(|g| g.node(id).is_some());
            if let Some(graph) = graph {
                let mut below: Vec<NodeId> = record
                    .sources()
                    .flat_map(|s| graph.subtree_of(s))
                    .filter(|b| doc.surface_points.contains_key(b))
                    .collect();
                // The set's order is arbitrary and two equal files must solve alike.
                below.sort_by_key(|b| b.raw());
                for b in below {
                    visit(doc, b, seen, out);
                }
            }
        }
        out.push(id);
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let ids: Vec<NodeId> = doc.surface_points.keys().copied().collect();
    for id in ids {
        visit(doc, id, &mut seen, &mut out);
    }
    out
}

impl App {
    /// Solves every live point on a surface, writing the ones that moved. Answers whether any did.
    ///
    /// **At the start of a rebuild, before anything is compiled**, where a surface axis catches up
    /// after: a point's position is in the graph, so whatever stands on it has to be compiled against
    /// the answer, and solving here means one rebuild sees it. Nothing here calls the rebuild, so it
    /// cannot loop, and the solve compiles each body on its own.
    pub(crate) fn refresh_surface_points(&mut self) -> bool {
        if self.doc.surface_points.is_empty() {
            self.surface_point_states.clear();
            return false;
        }
        let mut moved = false;
        for id in solve_order(&self.doc) {
            let Some(record) = self.doc.surface_points.get(&id).copied() else { continue };
            let Some(graph) = self.graph_holding(id).cloned() else {
                // The block is gone and the record waits for an undo to bring it back.
                self.surface_point_states.remove(&id);
                continue;
            };
            if !record.live {
                self.surface_point_states.remove(&id);
                continue;
            }
            if reads_itself(&self.doc, &graph, id) {
                self.fix_surface_point_in_a_loop(id, &graph);
                continue;
            }
            let at = match graph.node(id) {
                Some(Node::Point { at }) => at.map(|r| self.doc.resolve(&r)),
                _ => continue,
            };
            let at = match at {
                [Some(x), Some(y), Some(z)] => Some(Point3::new(x, y, z)),
                _ => None,
            };
            let stored = record.solved.and(at);
            match solve_surface_point(&self.doc, &graph, &record, stored, &mut self.leaves) {
                Ok(found) => {
                    self.surface_point_states.insert(
                        id,
                        PointState { error: None, warning: found.warning.clone(), crossing: found.crossing },
                    );
                    if worth_writing(at.zip(record.solved), &found) {
                        self.write_surface_point(id, &found);
                        moved = true;
                    }
                }
                Err(why) => {
                    let state = self.surface_point_states.entry(id).or_default();
                    state.error = Some(format!("{why} — it stays where it last stood"));
                    state.warning = None;
                }
            }
        }
        moved
    }

    /// Writes an answer into the point's block and its record, outside the history: it is derived,
    /// as a surface axis's two points are, so undoing the edit that moved it moves it back.
    fn write_surface_point(&mut self, id: NodeId, found: &Solution) {
        let literal = [found.at.x, found.at.y, found.at.z].map(ParamRef::Literal);
        for part in self.doc.parts.values_mut() {
            if let Some(Node::Point { at }) = part.graph.node_mut(id) {
                *at = literal;
                break;
            }
        }
        if let Some(record) = self.doc.surface_points.get_mut(&id) {
            record.solved = Some(found.solved);
        }
    }

    /// A point whose body has come to depend on it has no answer to settle on, so it is made Fixed
    /// where it stands and says why. Nothing is refused: the edit that made the loop stands.
    fn fix_surface_point_in_a_loop(&mut self, id: NodeId, graph: &Graph) {
        if let Some(record) = self.doc.surface_points.get_mut(&id) {
            record.live = false;
        }
        self.surface_point_states.remove(&id);
        let name = self.body_label_in(graph, id);
        self.say(format!("{name} is now fixed — the body it stood on is placed on it"));
    }

    /// Why a point cannot be made Live, if it cannot.
    pub(crate) fn surface_point_live_refusal(&self, id: NodeId) -> Option<String> {
        let graph = self.graph_holding(id)?;
        reads_itself(&self.doc, graph, id).then(|| {
            format!(
                "{} stays fixed — the body it stands on is placed on it, so it has no answer to settle on",
                self.body_label_in(graph, id)
            )
        })
    }
}

/// A spot guide at a world point facing a world direction.
fn spot_at(frame: &Similarity3<f64>, at: Point3<f64>, facing: Vector3<f64>) -> Guide {
    let here = frame.inverse() * at;
    let way = (frame.isometry.rotation.inverse() * facing).normalize();
    Guide::Spot { at: [here.x, here.y, here.z].map(ParamRef::Literal), facing: [way.x, way.y, way.z] }
}

/// A ray guide along frame axis `along`, through a world point.
fn ray_through(frame: &Similarity3<f64>, along: usize, flipped: bool, at: Point3<f64>) -> Guide {
    let axis = [fieldform_scene::Axis::X, fieldform_scene::Axis::Y, fieldform_scene::Axis::Z][along];
    let (a, b) = across(axis);
    let here = frame.inverse() * at;
    Guide::Ray {
        along: axis,
        flipped,
        u: ParamRef::Literal(here[a]),
        v: ParamRef::Literal(here[b]),
        facing: Facing::Front,
    }
}

/// A seed guide at a world point.
fn seed_at(frame: &Similarity3<f64>, at: Point3<f64>) -> Guide {
    let here = frame.inverse() * at;
    Guide::Nearest { at: [here.x, here.y, here.z].map(ParamRef::Literal) }
}

/// The record a click, or a drag's end, leaves: a spot in the body's frame where the surface was
/// touched, facing the way the body's own field faces there.
fn record_for_touch(
    doc: &Document,
    graph: &Graph,
    body: NodeId,
    at: Point3<f64>,
    leaves: &mut LeafCache,
) -> Result<SurfacePoint, String> {
    let mut record = SurfacePoint {
        body,
        frame: GuideFrame::Body,
        guide: Guide::Nearest { at: [ParamRef::Literal(0.0); 3] },
        live: true,
        solved: None,
    };
    let asked = ask(doc, graph, &record, leaves)?;
    let (on, facing) = asked
        .nearest(at)
        .ok_or_else(|| "the surface there has no direction to stand along — click somewhere else on it".to_string())?;
    record.guide = spot_at(&asked.frame, on, facing);
    Ok(record)
}

/// A point on a surface being slid: the body asked once, for every instant of the drag.
///
/// Compiling the body is the dear part of a solve, and a drag asks sixty times a second where the
/// surface is under the cursor; this holds the answer to the first question for the length of the
/// gesture. `before` is the record as the gesture found it, which is what its one undo entry reverses.
pub(crate) struct Slide {
    point: NodeId,
    before: SurfacePoint,
    asked: Asked,
    /// The graph as the gesture found it, when the gesture is the dot's own drag. The manipulator's
    /// drag keeps its own copy and leaves this empty.
    grabbed: Option<Graph>,
    /// Whether the dot's own drag has moved the point at all, so a press that was a click commits
    /// nothing.
    moved: bool,
}

impl App {
    /// What a click on the surface finds: the body under the cursor and the spot on it, or why not.
    fn touched_surface(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) -> Result<(NodeId, Point3<f64>), String> {
        let at = self
            .normal_axis_landing(aspect, rect, pos)
            .ok_or("No surface under the cursor — click on the body itself")?;
        let graph = self.part_graph().clone();
        let body = crate::split_axis::body_under(&self.doc, &graph, at, &mut self.leaves)
            .ok_or("No body under the cursor — click on the body itself")?;
        Ok((body, at))
    }

    /// The one click of Point on surface, or of Plane on surface: the surface under the cursor and
    /// the body it belongs to. A click on nothing keeps waiting.
    pub(crate) fn place_surface_point_click(&mut self, stand: Stand, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let (body, at) = match self.touched_surface(aspect, rect, pos) {
            Ok(found) => found,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        self.pending_curve = None;
        let graph = self.part_graph().clone();
        // A point made by a click stands where it was clicked from the start, so the first solve
        // takes the wall there and not the first one along the line.
        match record_for_touch(&self.doc, &graph, body, at, &mut self.leaves) {
            Ok(record) => {
                self.make_surface_point(record, Some(at), stand);
            }
            Err(why) => self.warn(why),
        }
    }

    /// Makes the point block and its record as one undo step, and the plane standing on it when one
    /// is asked for. A question with no answer yet is refused, since there is nowhere for the point to
    /// stand. Answers the block made: the point, or the plane.
    fn make_surface_point(&mut self, mut record: SurfacePoint, stored: Option<Point3<f64>>, stand: Stand) -> Option<NodeId> {
        let mut graph = self.part_graph().clone();
        let found = match solve_surface_point(&self.doc, &graph, &record, stored, &mut self.leaves) {
            Ok(found) => found,
            Err(why) => {
                self.warn(format!("No point made: {why}"));
                return None;
            }
        };
        record.solved = Some(found.solved);
        let point = graph.add(Node::Point { at: [found.at.x, found.at.y, found.at.z].map(ParamRef::Literal) });
        // **A plane on a surface names the point twice**, as its place and as its facing, and is
        // otherwise an ordinary plane: `Frame::From` a surface point answers the surface's own
        // frame, so nothing else has to know the plane stands on one.
        let made = match stand {
            Stand::Point => point,
            Stand::Plane => graph.add(Node::Plane {
                origin: fieldform_scene::Place::From { node: point },
                flipped: false,
                turn: Turn::From { node: point },
            }),
        };
        let command: Box<dyn fieldform_ops::Command> = Box::new(fieldform_ops::SetSurfacePoint::insert(point, record));
        self.set_part_graph_with(graph, vec![command]);
        self.select_body(Some(made));
        self.mark_changed();
        self.rebuild_render_resources();
        let warning = found.warning.map(|w| format!(" — {w}")).unwrap_or_default();
        let what = match stand {
            Stand::Point => "A point",
            Stand::Plane => "A plane",
        };
        self.say(format!(
            "{what} on {} at ({:.2}, {:.2}, {:.2}){warning}",
            self.body_label(record.body),
            found.at.x,
            found.at.y,
            found.at.z
        ));
        Some(made)
    }

    /// The record a line begins from: a ray down the frame's Z through its origin, which `at=`,
    /// `near=` or the ray's own keys then replace.
    fn default_record(&self) -> SurfacePoint {
        SurfacePoint {
            body: NodeId::from_raw(0),
            frame: GuideFrame::Body,
            guide: Guide::Ray {
                along: fieldform_scene::Axis::Z,
                flipped: false,
                u: ParamRef::Literal(0.0),
                v: ParamRef::Literal(0.0),
                facing: Facing::Front,
            },
            live: true,
            solved: None,
        }
    }

    /// `point on=<body> …`: made from typed numbers, so its first answer is the wall nearest its
    /// spot, or the first along its ray.
    pub(crate) fn point_on_surface_command(&mut self, args: &Args) {
        match self.surface_point_with_keys(self.default_record(), args) {
            Ok(record) => {
                self.make_surface_point(record, None, Stand::Point);
            }
            Err(why) => self.warn(why),
        }
    }

    /// `plane on=<point>`: a plane standing on a point, square to the surface the point is on. Or
    /// `plane on=<body> at=x,y,z`: the point is made first, and the plane on it, as one step.
    pub(crate) fn plane_on_surface_command(&mut self, args: &Args) {
        let lookup = |t: &str| self.lookup(t);
        let on = match console_keys::operand_named("plane", "on", args, &lookup) {
            Ok(on) => on.and_then(|o| o.source()),
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let Some(on) = on else {
            self.warn("`plane on=` names a point to stand on, or a body with at=x,y,z saying where on it");
            return;
        };
        if matches!(self.part_graph().node(on), Some(Node::Point { .. })) {
            let mut graph = self.part_graph().clone();
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::From { node: on },
                flipped: args.get("flipped").is_some_and(|_| console_keys::flag(args, "flipped")),
                turn: Turn::From { node: on },
            });
            self.set_part_graph(graph);
            self.select_body(Some(plane));
            self.mark_changed();
            self.rebuild_render_resources();
            self.say(format!("A plane on {}", self.body_label(on)));
            return;
        }
        if args.get("at").is_none() {
            self.warn("`plane on=` a body needs at=x,y,z to say where on it — or name a point to stand on");
            return;
        }
        match self.surface_point_with_keys(self.default_record(), args) {
            Ok(record) => {
                self.make_surface_point(record, None, Stand::Plane);
            }
            Err(why) => self.warn(why),
        }
    }

    /// A record with a line's keys read onto it: what making one and `set` share.
    ///
    /// `at=x,y,z` is a spot, typed in world: the surface there, facing the way it faces, or the way
    /// `facing=` says. `near=` is a seed, and `u=`, `v=`, `dir=`, `flip` and `back` a ray.
    fn surface_point_with_keys(&mut self, mut record: SurfacePoint, args: &Args) -> Result<SurfacePoint, String> {
        use console_keys::{flag, operand_named, param_one, Slot};
        let verb = "point";
        let lookup = |t: &str| self.lookup(t);
        if let Some(body) = operand_named(verb, "on", args, &lookup)?.and_then(|o| o.source()) {
            record.body = body;
        }
        if !self.part_graph().node(record.body).is_some_and(|n| !matches!(n, Node::Point { .. } | Node::Plane { .. })) {
            return Err("`point on=` names the body it stands on, like on=Cube_0".to_string());
        }
        if let Some(v) = args.get("from") {
            record.frame = if v.text().eq_ignore_ascii_case("body") {
                GuideFrame::Body
            } else {
                let node = operand_named(verb, "from", args, &lookup)?
                    .and_then(|o| o.source())
                    .ok_or("`point from=` names a plane to measure from")?;
                if !matches!(self.part_graph().node(node), Some(Node::Plane { .. })) {
                    return Err("`point from=` names a plane to measure from".to_string());
                }
                GuideFrame::Plane { node }
            };
        }
        let numbers = |key: &str, example: &str| -> Result<Option<[f64; 3]>, String> {
            let Some(v) = args.get(key) else { return Ok(None) };
            let t = v.text();
            let parsed: Vec<f64> = t.split(',').filter_map(|n| n.trim().parse().ok()).collect();
            let [x, y, z] = parsed[..] else {
                return Err(format!("`point {key}=` wants three numbers, like {key}={example}"));
            };
            Ok(Some([x, y, z]))
        };
        if let Some([x, y, z]) = numbers("near", "10,0,5")? {
            let frame = guide_frame(&self.doc, self.part_graph(), &record)?;
            record.guide = seed_at(&frame, Point3::new(x, y, z));
            return Ok(record);
        }
        if let Some([x, y, z]) = numbers("at", "10,0,5")? {
            let graph = self.part_graph().clone();
            let asked = ask(&self.doc, &graph, &record, &mut self.leaves)?;
            let typed = Point3::new(x, y, z);
            let facing = match numbers("facing", "0,0,1")? {
                Some(f) => Vector3::from(f)
                    .try_normalize(1e-9)
                    .ok_or_else(|| "`point facing=` wants a direction, not 0,0,0".to_string())?,
                // The facing kept, when a spot is moved and the line says no other; the surface's
                // own nearest the spot otherwise.
                None => match record.guide {
                    Guide::Spot { facing, .. } => asked.frame.isometry.rotation * Vector3::from(facing),
                    _ => asked
                        .nearest(typed)
                        .ok_or_else(|| "no surface settles near that spot — type one nearer the body".to_string())?
                        .1,
                },
            };
            record.guide = spot_at(&asked.frame, typed, facing);
            return Ok(record);
        }
        let reads_a_ray = ["u", "v", "dir", "flip", "back"].iter().any(|k| args.get(k).is_some());
        if reads_a_ray && !matches!(record.guide, Guide::Ray { .. }) {
            record.guide = self.default_record().guide;
        }
        if let Guide::Ray { along, flipped, u, v, facing } = &mut record.guide {
            if let Some(value) = args.get("u") {
                *u = param_one(verb, "u", value, Slot::Number, &lookup)?;
            }
            if let Some(value) = args.get("v") {
                *v = param_one(verb, "v", value, Slot::Number, &lookup)?;
            }
            if let Some(value) = args.get("dir") {
                *along = match value.text().trim().to_ascii_lowercase().as_str() {
                    "x" => fieldform_scene::Axis::X,
                    "y" => fieldform_scene::Axis::Y,
                    "z" => fieldform_scene::Axis::Z,
                    other => return Err(format!("`point dir=` is x, y or z, not {other:?}")),
                };
            }
            if args.get("flip").is_some() {
                *flipped = flag(args, "flip");
            }
            if args.get("back").is_some() {
                *facing = if flag(args, "back") { Facing::Back } else { Facing::Front };
            }
        }
        if args.get("fixed").is_some() {
            record.live = !flag(args, "fixed");
        }
        Ok(record)
    }

    /// `set` on a point that stands on a surface. While it is Live a typed `at` moves the question
    /// — the spot, or the seed — and the surface answers; a ray has no spot to move, so its U and V
    /// are what to type.
    pub(crate) fn set_surface_point_keys(&mut self, id: NodeId, record: SurfacePoint, args: &Args) -> Result<(), String> {
        if record.live && args.get("at").is_some() && matches!(record.guide, Guide::Ray { .. }) {
            return Err(format!(
                "{} stands on a surface by its U and V — set those, or `fix` it to type a position",
                self.body_label(id)
            ));
        }
        let edited = self.surface_point_with_keys(record, args)?;
        if edited.live
            && !record.live
            && let Some(why) = self.surface_point_live_refusal(id)
        {
            return Err(why);
        }
        self.commit_surface_point(id, record, edited);
        Ok(())
    }

    /// One record edit as one undo step, solved at once.
    fn commit_surface_point(&mut self, id: NodeId, before: SurfacePoint, after: SurfacePoint) {
        if before == after {
            return;
        }
        let command = Box::new(fieldform_ops::SetSurfacePoint::new(id, Some(before), Some(after)));
        if let Err(e) = self.stack.execute(&mut self.doc, command) {
            self.warn(e.to_string());
        }
        self.mark_changed();
        self.rebuild_render_resources();
    }

    /// `fix` and `live`, on the block named or the selected one.
    pub(crate) fn surface_point_live_command(&mut self, args: &Args, live: bool) {
        let lookup = |t: &str| self.lookup(t);
        let named = match console_keys::operand_named("point", "target", args, &lookup) {
            Ok(named) => named.and_then(|o| o.source()),
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let target = named.or(self.selected_body).filter(|id| self.doc.surface_points.contains_key(id));
        let Some(id) = target else {
            self.warn("That is for a point on a surface — select one, or name it with `target=`");
            return;
        };
        let Some(record) = self.doc.surface_points.get(&id).copied() else { return };
        if live && !record.live
            && let Some(why) = self.surface_point_live_refusal(id)
        {
            self.warn(why);
            return;
        }
        self.commit_surface_point(id, record, SurfacePoint { live, ..record });
        let word = if live { "live" } else { "fixed" };
        self.say(format!("{} is {word}", self.body_label(id)));
    }

    // ------------------------------------------------------------------------------- sliding

    /// The live point on a surface whose dot is under `pos`, if one is within reach.
    pub(crate) fn surface_point_at(&self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) -> Option<NodeId> {
        let mut best: Option<(NodeId, f32)> = None;
        for (id, record) in &self.doc.surface_points {
            if !record.live || !self.is_visible(*id) {
                continue;
            }
            let Some(Node::Point { at }) = self.part_graph().node(*id) else { continue };
            let at = Point3::new(self.resolve(at[0]), self.resolve(at[1]), self.resolve(at[2]));
            let Some(on_screen) = self.camera.project_to_screen(aspect, rect, at.cast::<f32>()) else { continue };
            let distance = on_screen.distance(pos);
            if distance <= GRAB_PX && best.is_none_or(|(_, b)| distance < b) {
                best = Some((*id, distance));
            }
        }
        best.map(|(id, _)| id)
    }

    /// The body of `id` made ready to be asked, kept for the length of a gesture. `None` when `id`
    /// is not a live point on a surface, or its body cannot be asked.
    fn slide_for(&mut self, id: NodeId, grabbed: Option<Graph>) -> Option<&mut Slide> {
        if self.slide.as_ref().is_none_or(|s| s.point != id) {
            let record = self.doc.surface_points.get(&id).copied().filter(|r| r.live)?;
            let graph = self.part_graph().clone();
            let asked = ask(&self.doc, &graph, &record, &mut self.leaves).ok()?;
            self.slide = Some(Slide { point: id, before: record, asked, grabbed, moved: false });
        }
        self.slide.as_mut()
    }

    /// Forgets the body a gesture was asking. Called when the gesture ends, however it ends.
    pub(crate) fn end_slide(&mut self) {
        self.slide = None;
    }

    /// Forgets the body a gesture on `id` was asking, and puts back the record as the gesture
    /// found it: the answer the instants wrote goes back with the graph, or a plane on the point
    /// would stay turned the way the abandoned drag left it until the next solve.
    pub(crate) fn cancel_slide(&mut self, id: NodeId) {
        if let Some(slide) = self.slide.take_if(|s| s.point == id)
            && let Some(record) = self.doc.surface_points.get_mut(&id)
        {
            *record = slide.before;
        }
    }

    /// Where a live point on a surface goes when a drag asks it to stand at `target`: the surface
    /// nearest, in world, and the way it faces there. `None` leaves the point where it is.
    ///
    /// **Projected every instant, not once at the end.** The point is seen sliding over the surface,
    /// round a corner and onto the next face, and what is built on it turns with it; the first build
    /// let it fly free and snapped it back on release, which reads as a point that ignores the
    /// surface until it is let go.
    pub(crate) fn slide_surface_point(&mut self, id: NodeId, target: Point3<f64>) -> Option<Point3<f64>> {
        let slide = self.slide_for(id, None)?;
        let (at, normal) = slide.asked.nearest(target)?;
        self.show_slid(id, at, normal);
        Some(at)
    }

    /// Shows a point slid to `at`, facing `normal`, before anything is committed: its record's answer
    /// is written so a plane on it turns with the surface this instant.
    fn show_slid(&mut self, id: NodeId, at: Point3<f64>, normal: Vector3<f64>) {
        let Some(slide) = self.slide.as_mut().filter(|s| s.point == id) else { return };
        slide.moved = true;
        let first_across = match slide.before.guide {
            Guide::Ray { along, .. } => across(along),
            _ => (0, 1),
        };
        let x = roll_about(normal, &frame_axes(&slide.asked.frame), first_across);
        let solved = Solved { normal: [normal.x, normal.y, normal.z], x: [x.x, x.y, x.z] };
        if let Some(record) = self.doc.surface_points.get_mut(&id) {
            record.solved = Some(solved);
        }
        let _ = at;
    }

    /// A press on a point's dot: the start of its own drag, which slides it under the cursor.
    pub(crate) fn grab_surface_point(&mut self, id: NodeId) {
        let before_graph = self.part_graph().clone();
        self.select_body(Some(id));
        self.slide_for(id, Some(before_graph));
    }

    /// The dot's drag, continued: the surface of its own body under the cursor, when there is one.
    /// Over another body, or over nothing, the point waits where it is.
    pub(crate) fn continue_surface_point_grab(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let Some(id) = self.slide.as_ref().filter(|s| s.grabbed.is_some()).map(|s| s.point) else { return };
        let Some(hit) = self.normal_axis_landing(aspect, rect, pos) else { return };
        let on_body = self.slide.as_ref().is_some_and(|s| s.asked.holds(hit));
        if !on_body {
            return;
        }
        let Some((at, normal)) = self.slide.as_ref().and_then(|s| s.asked.nearest(hit)) else { return };
        self.show_slid(id, at, normal);
        let mut graph = self.part_graph().clone();
        if let Some(Node::Point { at: slot }) = graph.node_mut(id) {
            *slot = [at.x, at.y, at.z].map(ParamRef::Literal);
        }
        // As the manipulator's own instants are: folded into one gesture, drawn at draft, settled
        // when the hand comes off.
        self.set_part_graph(graph);
        self.refresh_scene_bounds();
        self.traced.invalidate(traced::Quality::Draft);
    }

    /// The dot's drag, ended: where it was left becomes the question, as one undo entry. A press
    /// that never moved it commits nothing.
    pub(crate) fn end_surface_point_grab(&mut self) {
        let Some(slide) = self.slide.as_ref().filter(|s| s.grabbed.is_some()) else { return };
        let (id, moved) = (slide.point, slide.moved);
        let Some(before_graph) = slide.grabbed.clone() else { return };
        if !moved {
            self.end_slide();
            return;
        }
        let at = match self.part_graph().node(id) {
            Some(Node::Point { at }) => Point3::new(self.resolve(at[0]), self.resolve(at[1]), self.resolve(at[2])),
            _ => {
                self.end_slide();
                return;
            }
        };
        self.surface_point_dragged(id, at, before_graph);
    }

    /// The dot's drag, abandoned: everything back where the press found it.
    pub(crate) fn cancel_surface_point_grab(&mut self) {
        let Some(slide) = self.slide.take().filter(|s| s.grabbed.is_some()) else { return };
        if let Some(record) = self.doc.surface_points.get_mut(&slide.point) {
            *record = slide.before;
        }
        if let Some(graph) = slide.grabbed {
            self.put_back_pressed_graph(graph);
        }
        self.rebuild_render_resources();
    }

    /// A drag of a Live point has ended at `moved`, a point on the surface: where it went becomes
    /// the question, and the surface answers it. Answers whether this was such a point.
    ///
    /// **The numbers change, not the position**, or the next solve would put it straight back. A
    /// spot keeps its kind, and so do a ray and a seed made from the console: the ray's U and V are
    /// read off where it landed, which keeps a point placed on a grid on that grid.
    pub(crate) fn surface_point_dragged(&mut self, id: NodeId, moved: Point3<f64>, before_graph: Graph) -> bool {
        let Some(now) = self.doc.surface_points.get(&id).copied().filter(|r| r.live) else { return false };
        // The record as the gesture found it, since the instants in between wrote its answer.
        let record = match self.slide.as_ref().filter(|s| s.point == id) {
            Some(slide) => slide.before,
            None => now,
        };
        let facing = self
            .slide_for(id, None)
            .and_then(|s| s.asked.nearest(moved))
            .map(|(_, normal)| normal);
        self.end_slide();
        self.set_part_graph(before_graph);
        let edited = match guide_frame(&self.doc, self.part_graph(), &record) {
            Ok(frame) => match record.guide {
                Guide::Ray { along, flipped, facing, .. } => {
                    let index = (0..3).find(|i| {
                        let (a, b) = across(along);
                        *i != a && *i != b
                    });
                    let guide = ray_through(&frame, index.unwrap_or(2), flipped, moved);
                    let Guide::Ray { u, v, .. } = guide else { unreachable!("made a ray") };
                    SurfacePoint { guide: Guide::Ray { along, flipped, u, v, facing }, ..record }
                }
                Guide::Nearest { .. } => SurfacePoint { guide: seed_at(&frame, moved), ..record },
                Guide::Spot { facing: was, .. } => {
                    let facing = facing.unwrap_or_else(|| frame.isometry.rotation * Vector3::from(was));
                    SurfacePoint { guide: spot_at(&frame, moved, facing), ..record }
                }
            },
            Err(why) => {
                self.warn(why);
                record
            }
        };
        // The answer the instants wrote is put back before the edit, so the entry reverses to the
        // record the gesture found and not to one the gesture half-wrote.
        if let Some(slot) = self.doc.surface_points.get_mut(&id) {
            *slot = record;
        }
        self.commit_surface_point(id, record, edited);
        self.rebuild_render_resources();
        true
    }

    // --------------------------------------------------------------------------------- panel

    /// The block of a point on a surface: where it stands, what it stands on, and what the last
    /// solve said.
    ///
    /// **As little as the question needs.** A spot shows where it is; a ray or a seed typed from the
    /// console shows its numbers, since those are what was typed. Which wall is meant is asked only
    /// where there is more than one on the line, and how the solve looks is not asked at all.
    pub(crate) fn surface_point_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        at: [ParamRef; 3],
        record: SurfacePoint,
    ) {
        ui.label(
            "A place on a body's surface. Drag it over the surface; it stays where it was put when \
             the body changes, and whatever stands on it follows.",
        );
        ui.separator();
        let mut edited = record;
        let mut touched = false;
        let mut stepped = false;

        let graph = self.part_graph().clone();
        let bodies: Vec<(NodeId, String)> = self
            .doc
            .listed_bodies(&graph)
            .into_iter()
            .filter(|b| !matches!(graph.node(*b), Some(Node::Plane { .. } | Node::Point { .. })))
            .map(|b| (b, self.body_label(b)))
            .collect();
        ui.horizontal(|ui| {
            row_label(ui, "On:");
            egui::ComboBox::from_id_salt(("surface-point-body", id))
                .selected_text(self.body_label(record.body))
                .show_ui(ui, |ui| {
                    for (body, name) in &bodies {
                        if ui.selectable_value(&mut edited.body, *body, name).clicked() {
                            stepped = true;
                        }
                    }
                });
        });
        if let GuideFrame::Plane { node } = record.frame {
            ui.label(egui::RichText::new(format!("measured from {}", self.body_label(node))).weak());
        }

        let mut number = |this: &mut Self, ui: &mut egui::Ui, label: &str, value: &mut ParamRef, hover: &str| {
            let (now, changed) = this.number_field(
                ui,
                label,
                *value,
                0.5,
                -1.0e6..=1.0e6,
                hover,
                " mm",
                Naming::Offered,
                Labelled::Row,
                Grain::Continuous,
            );
            if changed {
                *value = now;
                touched = true;
            }
        };
        match &mut edited.guide {
            // Where it stands, in world. Typed, the spot moves there and the surface answers.
            Guide::Spot { facing, .. } if record.live => {
                let typed = self.centre_fields(ui, "At:", at);
                if typed != at
                    && let Ok(frame) = guide_frame(&self.doc, &graph, &record)
                {
                    let world = Point3::new(self.resolve(typed[0]), self.resolve(typed[1]), self.resolve(typed[2]));
                    let way = frame.isometry.rotation * Vector3::from(*facing);
                    edited.guide = spot_at(&frame, world, way);
                    touched = true;
                }
            }
            Guide::Spot { .. } => {}
            Guide::Ray { along, flipped, u, v, facing } => {
                number(self, ui, "U:", u, "Millimetres across the line, in the body's own frame");
                number(self, ui, "V:", v, "Millimetres across the line, in the body's own frame");
                let side = if *flipped { "minus" } else { "plus" };
                let face = if *facing == Facing::Back { ", meeting the far side" } else { "" };
                ui.label(
                    egui::RichText::new(format!("along {along:?} from the {side} side{face} — typed from the console"))
                        .weak(),
                );
            }
            Guide::Nearest { at } => {
                let hover = "A seed in the body's own frame; the point stands at the surface nearest it";
                for (i, name) in ["Near X:", "Near Y:", "Near Z:"].into_iter().enumerate() {
                    number(self, ui, name, &mut at[i], hover);
                }
            }
        }

        let state = self.surface_point_states.get(&id).cloned().unwrap_or_default();
        let mut step_to: Option<isize> = None;
        if record.live
            && let Some((which, of)) = state.crossing
            && of > 1
        {
            ui.horizontal(|ui| {
                row_label(ui, "Wall:");
                if ui.add_enabled(which > 1, egui::Button::new("\u{2039}")).on_hover_text("The one before").clicked() {
                    step_to = Some(-1);
                }
                ui.label(format!("{which} of {of}"))
                    .on_hover_text("Several walls lie behind one another here; this is the one it stands on");
                if ui.add_enabled(which < of, egui::Button::new("\u{203a}")).on_hover_text("The one after").clicked() {
                    step_to = Some(1);
                }
            });
        }

        ui.horizontal(|ui| {
            row_label(ui, "Follows:");
            if ui
                .add(mode_button(edited.live, "Live"))
                .on_hover_text("Stays on the surface whenever the part changes")
                .clicked()
                && !edited.live
            {
                match self.surface_point_live_refusal(id) {
                    Some(why) => self.warn(why),
                    None => {
                        edited.live = true;
                        stepped = true;
                    }
                }
            }
            if ui
                .add(mode_button(!edited.live, "Fixed"))
                .on_hover_text("Left where it is, whatever the body does")
                .clicked()
                && edited.live
            {
                edited.live = false;
                stepped = true;
            }
        });
        if let Some(why) = &state.error {
            ui.colored_label(ui.visuals().error_fg_color, why);
        } else if let Some(why) = &state.warning {
            ui.colored_label(ui.visuals().warn_fg_color, why);
        }
        if !record.live {
            let new_at = self.centre_fields(ui, "At:", at);
            if new_at != at {
                self.replace_node(id, Node::Point { at: new_at });
            }
        }

        if let (Some(way), Some((which, _))) = (step_to, state.crossing) {
            let mut leaves = std::mem::take(&mut self.leaves);
            let list = surface_point_crossings(&self.doc, &graph, &record, &mut leaves);
            self.leaves = leaves;
            match list {
                Ok(list) => {
                    if let Some(to) = list.get((which as isize - 1 + way) as usize) {
                        // Where it stands is what chooses among walls, so stepping to another is
                        // standing there.
                        self.replace_node(id, Node::Point { at: [to.x, to.y, to.z].map(ParamRef::Literal) });
                        self.rebuild_render_resources();
                    }
                }
                Err(why) => self.warn(why),
            }
        }

        // **One history entry per gesture**, as an axis's numbers are: the record before the first
        // change is kept, and the entry is written when a frame passes with no change and no button
        // held.
        if touched && self.surface_point_edit_before.is_none() {
            self.surface_point_edit_before = Some((id, record));
        }
        let settled = !touched && !ui.input(|i| i.pointer.any_down());
        if edited != record {
            self.doc.surface_points.insert(id, edited);
            self.mark_changed();
            self.rebuild_render_resources();
        }
        let commit = if stepped && edited != record {
            Some(record)
        } else if settled {
            self.surface_point_edit_before.take().filter(|(p, _)| *p == id).map(|(_, before)| before)
        } else {
            None
        };
        if let Some(before) = commit {
            let now = self.doc.surface_points.get(&id).copied().unwrap_or(edited);
            self.commit_surface_point(id, before, now);
        }
    }

    /// The dot of every point on a surface, in the colour its last solve left it, and a short tick
    /// along the selected one's normal: a point draws nothing of its own, one that is red has to be
    /// findable, and which way it faces is what a plane on it will take.
    pub(crate) fn draw_surface_points(&self, painter: &egui::Painter, aspect: f32, rect: egui::Rect) {
        for (id, record) in &self.doc.surface_points {
            let Some(Node::Point { at }) = self.part_graph().node(*id) else { continue };
            let at = Point3::new(self.resolve(at[0]), self.resolve(at[1]), self.resolve(at[2]));
            let Some(pos) = self.camera.project_to_screen(aspect, rect, at.cast::<f32>()) else { continue };
            if !self.is_visible(*id) {
                continue;
            }
            let state = self.surface_point_states.get(id);
            let visuals = painter.ctx().global_style().visuals.clone();
            let selected = self.selected_body == Some(*id);
            let colour = match state {
                Some(s) if s.error.is_some() => visuals.error_fg_color,
                Some(s) if s.warning.is_some() => visuals.warn_fg_color,
                _ if selected => visuals.selection.bg_fill,
                _ => visuals.strong_text_color(),
            };
            if selected && let Some(solved) = record.solved {
                let reach = f64::from(self.camera.view_half_height(at.cast::<f32>())) * 0.08;
                let tip = at + Vector3::from(solved.normal) * reach;
                if let Some(end) = self.camera.project_to_screen(aspect, rect, tip.cast::<f32>()) {
                    painter.line_segment([pos, end], egui::Stroke::new(1.5_f32, colour));
                }
            }
            painter.circle_filled(pos, 3.5, colour);
            painter.circle_stroke(pos, 3.5, egui::Stroke::new(1.0_f32, visuals.extreme_bg_color));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_scene::{Axis, BooleanOp, Place};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn cuboid(graph: &mut Graph, centre: [f64; 3], size: [f64; 3]) -> NodeId {
        graph.add(Node::Cuboid {
            edge: Default::default(),
            centre: Place::At(centre.map(lit)),
            size: size.map(lit),
            turn: fieldform_scene::no_turn(),
        })
    }

    fn cube_union(graph: &mut Graph, boss_at: f64) -> (NodeId, NodeId) {
        let main = cuboid(graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
        let boss = cuboid(graph, [boss_at, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let body = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![main.into(), boss.into()] });
        (body, boss)
    }

    fn ray(body: NodeId, u: f64, v: f64) -> SurfacePoint {
        SurfacePoint {
            body,
            frame: GuideFrame::Body,
            guide: Guide::Ray { along: Axis::Z, flipped: false, u: lit(u), v: lit(v), facing: Facing::Front },
            live: true,
            solved: None,
        }
    }

    fn spot(body: NodeId, at: [f64; 3], facing: [f64; 3]) -> SurfacePoint {
        SurfacePoint { guide: Guide::Spot { at: at.map(lit), facing }, ..ray(body, 0.0, 0.0) }
    }

    fn solve(graph: &Graph, record: &SurfacePoint, stored: Option<Point3<f64>>) -> Result<Solution, String> {
        solve_surface_point(&Document::new(), graph, record, stored, &mut LeafCache::default())
    }

    /// **§12.1. Growing the body elsewhere leaves the point where it is**, where the same place held
    /// as a fraction of the box slides: the control, which is what proves the case can fail.
    #[test]
    fn growing_the_body_elsewhere_leaves_the_point_where_it_is() {
        let mut graph = Graph::empty();
        let (body, boss) = cube_union(&mut graph, 30.0);
        let record = ray(body, 5.0, 5.0);
        let found = solve(&graph, &record, None).expect("solves");
        let before = found.at;
        assert!((before - Point3::new(5.0, 5.0, 5.0)).norm() < 1e-3, "{before}");

        let axis = fieldform_scene::SurfaceAxis {
            body,
            direction: Axis::Z,
            flipped: false,
            u: lit(0.4),
            v: lit(0.75),
            length: lit(10.0),
            seed: None,
        };
        let doc = Document::new();
        let held = crate::split_axis::solve_surface_axis(&doc, &graph, &axis, &mut LeafCache::default()).unwrap()[0];

        if let Some(Node::Cuboid { centre, .. }) = graph.node_mut(boss) {
            *centre = Place::At([60.0, 0.0, 0.0].map(lit));
        }
        // The box grew, so the search's tolerance did, and what is compared is what gets written.
        let after = solve(&graph, &record, Some(before)).expect("solves");
        assert!(!worth_writing(Some((before, found.solved)), &after), "the point stayed: {before} to {}", after.at);

        let slid = crate::split_axis::solve_surface_axis(&doc, &graph, &axis, &mut LeafCache::default()).unwrap()[0];
        assert!((slid - held).norm() > 1.0, "the control moves, or this proves nothing: {held} to {slid}");
    }

    /// **A spot stays on the face it was put on when that face moves away along its own normal**,
    /// where the nearest surface to the spot is then the face round the corner: the control, which
    /// is what a seed would have done.
    #[test]
    fn a_spot_near_an_edge_stays_on_its_face_when_the_face_moves_away() {
        let mut graph = Graph::empty();
        let block = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
        // A millimetre in from the top face's long edge.
        let record = spot(block, [5.0, 9.0, 5.0], [0.0, 0.0, 1.0]);
        let first = solve(&graph, &record, None).expect("solves");
        assert!((first.at - Point3::new(5.0, 9.0, 5.0)).norm() < 1e-3, "{}", first.at);

        if let Some(Node::Cuboid { size, .. }) = graph.node_mut(block) {
            *size = [40.0, 20.0, 20.0].map(lit);
        }
        let after = solve(&graph, &record, Some(first.at)).expect("solves");
        assert!((after.at - Point3::new(5.0, 9.0, 10.0)).norm() < 1e-3, "straight up to the new top: {}", after.at);

        let field = GraphField::compile_graph(&Document::new(), &graph, 1e-3).expect("compiles");
        let nearest = walk_to_surface(&field, Point3::new(5.0, 9.0, 5.0), 1e-3, 1e-4).expect("settles");
        assert!((nearest.y - 10.0).abs() < 1e-2, "the nearest surface is the side wall, so a seed would have hopped: {nearest}");
    }

    /// **§12.2. Turning the body carries the point round with it.**
    #[test]
    fn turning_the_body_carries_the_point_with_it() {
        let turned = |degrees: f64| {
            let mut graph = Graph::empty();
            let block = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
            let body = graph.add(Node::Transform {
                input: block.into(),
                translation: [lit(0.0); 3],
                rotation: [lit(0.0), lit(0.0), lit(degrees)],
                scale: lit(1.0),
                about: Place::origin(),
            });
            (graph, body)
        };
        let (graph, body) = turned(30.0);
        let record = spot(body, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0]);
        let first = solve(&graph, &record, None).expect("solves").at;
        let (graph, body) = turned(60.0);
        let second = solve(&graph, &spot(body, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0]), Some(first)).expect("solves").at;
        let r = nalgebra::Rotation3::from_axis_angle(&Vector3::z_axis(), 60f64.to_radians());
        let expected = r * Point3::new(5.0, 5.0, 5.0);
        assert!((second - expected).norm() < 1e-3, "{second} against {expected}");
    }

    fn cup(graph: &mut Graph, outer: f64, inner: f64) -> NodeId {
        let cylinder = |g: &mut Graph, r: f64, z0: f64| {
            g.add(Node::Cylinder {
                edge: Default::default(),
                ends: fieldform_scene::Ends::between(Place::at(0.0, 0.0, z0), Place::at(0.0, 0.0, 40.0)),
                radius: lit(r),
            })
        };
        let a = cylinder(graph, outer, 0.0);
        let b = cylinder(graph, inner, 2.0);
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![a.into(), b.into()] })
    }

    /// **§12.3. A thin wall that moves keeps the point on its outer face**, though the inner face
    /// ends up nearer where the point stood. The control is the same solve with the facing rule
    /// taken away: both faces counted, nearest wins, and it lands on the wrong wall.
    #[test]
    fn a_thin_wall_that_moves_keeps_the_point_on_its_own_face() {
        let mut graph = Graph::empty();
        let body = cup(&mut graph, 20.0, 19.0);
        let record = |body| spot(body, [20.0, 0.0, 20.0], [1.0, 0.0, 0.0]);
        let first = solve(&graph, &record(body), None).expect("solves").at;
        assert!((first.x - 20.0).abs() < 1e-3, "the outer face first: {first}");

        let mut moved = Graph::empty();
        let body = cup(&mut moved, 23.0, 22.0);
        let after = solve(&moved, &record(body), Some(first)).expect("solves").at;
        assert!((after.x - 23.0).abs() < 1e-3, "still the outer face: {after}");

        // The control: nearest of every crossing, whichever way it faces.
        let field = GraphField::compile_graph(&Document::new(), &moved, 1e-3).expect("compiles");
        let all = crossings(
            &field,
            Point3::new(60.0, 0.0, 20.0),
            -Vector3::x(),
            (Point3::new(-23.0, -23.0, 0.0), Point3::new(23.0, 23.0, 40.0)),
            1e-3,
        )
        .expect("lists");
        let nearest = all.iter().min_by(|a, b| (a.point - first).norm().total_cmp(&(b.point - first).norm())).unwrap();
        assert!((nearest.point.x - 22.0).abs() < 1e-3, "without the facing it would take the inner face");
    }

    /// **§12.4. A new wall in front does not capture the point**, and the readout says there are
    /// two to choose from. The control is what the first hit would have taken.
    #[test]
    fn a_new_wall_in_front_does_not_capture_the_point() {
        let mut graph = Graph::empty();
        let lower = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 40.0, 10.0]);
        let body = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![lower.into(), lower.into()] });
        let record = ray(body, 3.0, 4.0);
        let first = solve(&graph, &record, None).expect("solves");
        assert!((first.at.z - 5.0).abs() < 1e-3);

        let upper = cuboid(&mut graph, [0.0, 0.0, 30.0], [40.0, 40.0, 10.0]);
        if let Some(Node::Boolean { inputs, .. }) = graph.node_mut(body) {
            inputs[1] = upper.into();
        }
        let second = solve(&graph, &record, Some(first.at)).expect("solves");
        assert!((second.at.z - 5.0).abs() < 1e-3, "still the lower slab: {}", second.at);
        assert_eq!(second.crossing, Some((2, 2)), "the second of two along the ray");
        assert!((solve(&graph, &record, None).unwrap().at.z - 35.0).abs() < 1e-3, "the first hit would have jumped");
    }

    /// **§12.5. A seed on a wall's mid-plane stays on the face the point was made on**, and says so.
    #[test]
    fn a_seed_midway_between_two_walls_stays_on_its_wall_and_warns() {
        let mut graph = Graph::empty();
        let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [100.0, 100.0, 10.0]);
        let record = SurfacePoint {
            guide: Guide::Nearest { at: [lit(0.0), lit(0.0), lit(0.0)] },
            ..ray(body, 0.0, 0.0)
        };
        let mut warned = 0;
        for top in [5.0, -5.0] {
            let clicked = Point3::new(0.0, 0.0, top);
            let found = solve(&graph, &record, Some(clicked)).expect("solves");
            assert!((found.at.z - top).abs() < 1e-3, "{top}: {}", found.at);
            warned += usize::from(found.warning.is_some());
            let again = solve(&graph, &record, Some(found.at)).expect("solves");
            assert!((again.at - found.at).norm() < 1e-3, "and solving again stays");
        }
        // The walk from the seed lands on one wall, so the point on the other is the one to say so.
        assert_eq!(warned, 1, "it says so where the seed and the point disagree");
    }

    /// **§12.6. A ray off the body is an error that names the numbers to change**, and a spot whose
    /// face has gone is one that says what to do.
    #[test]
    fn a_line_that_misses_is_an_error_that_says_what_to_change() {
        let mut graph = Graph::empty();
        let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
        let why = solve(&graph, &ray(body, 500.0, 0.0), None).unwrap_err();
        assert!(why.contains("U 500") && why.contains("move U or V"), "{why}");
        let why = solve(&graph, &spot(body, [500.0, 0.0, 5.0], [0.0, 0.0, 1.0]), None).unwrap_err();
        assert!(why.contains("nothing faces that way") && why.contains("drag it"), "{why}");
    }

    /// **A spot on an edge stands at the spot and says so**: its line runs along a face and crosses
    /// nothing, which is not the same as the face having gone. Typed to the millimetre from the
    /// console, this is the corner of a block; and the same spot off the body is still the error.
    #[test]
    fn a_spot_on_an_edge_stands_at_the_spot_and_says_so() {
        let mut graph = Graph::empty();
        let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
        // Either face's normal, or the edge's own direction: each line lies in a face.
        for facing in [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            let found = solve(&graph, &spot(body, [-20.0, 0.0, 5.0], facing), None).expect("solves");
            assert!((found.at - Point3::new(-20.0, 0.0, 5.0)).norm() < 1e-3, "{facing:?}: {}", found.at);
            assert!(found.warning.as_deref().is_some_and(|w| w.contains("edge")), "{:?}", found.warning);
        }
        // The same spot once the block has shrunk away from it is not on the surface, and says so.
        if let Some(Node::Cuboid { size, .. }) = graph.node_mut(body) {
            *size = [20.0, 20.0, 10.0].map(lit);
        }
        let why = solve(&graph, &spot(body, [-20.0, 0.0, 5.0], [0.0, 1.0, 0.0]), None).unwrap_err();
        assert!(why.contains("nothing faces that way"), "{why}");
    }

    /// **§12.7. A grazing line warns, and a square one does not.**
    #[test]
    fn a_grazing_ray_warns_and_a_square_one_does_not() {
        let mut graph = Graph::empty();
        let body = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let warning = |u: f64| solve(&graph, &ray(body, u, 0.0), None).expect("solves").warning;
        assert!(warning(9.99).is_some_and(|w| w.contains("small edit")), "{:?}", warning(9.99));
        assert_eq!(warning(5.0), None);
    }

    /// **§12.8. Solving from its own answer gives that answer**, to the bit, and writes nothing.
    #[test]
    fn solving_from_its_own_answer_gives_that_answer_and_writes_nothing() {
        let mut graph = Graph::empty();
        let body = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        for record in [ray(body, 3.0, 4.0), spot(body, [3.0, 4.0, 8.66], [0.3, 0.4, 0.866])] {
            let first = solve(&graph, &record, None).expect("solves");
            let second = solve(&graph, &record, Some(first.at)).expect("solves");
            assert_eq!(first.at, second.at);
            assert!(!worth_writing(Some((first.at, first.solved)), &second));
            assert!(worth_writing(None, &second), "a point never solved is always written");
        }
    }

    /// **§12.11. Measured from a turned plane, a point lands on the analytic ray-sphere answer.**
    #[test]
    fn measured_from_a_plane_it_lands_on_the_analytic_answer() {
        let turn = 45f64.to_radians();
        let rotation = nalgebra::Rotation3::from_axis_angle(&Vector3::x_axis(), turn);
        let mut graph = Graph::empty();
        let plane = graph.add(Node::Plane {
            origin: Place::At([0.0, 0.0, 30.0].map(lit)),
            flipped: false,
            turn: fieldform_scene::Turn::By([lit(45.0), lit(0.0), lit(0.0)]),
        });
        let normal = rotation * Vector3::z();
        let origin = Point3::new(0.0, 0.0, 30.0);
        let centre = origin - normal * 40.0;
        let sphere = graph.add(Node::Sphere { centre: Place::At([centre.x, centre.y, centre.z].map(lit)), radius: lit(20.0) });
        let record = SurfacePoint {
            frame: GuideFrame::Plane { node: plane },
            ..ray(sphere, 3.0, 4.0)
        };
        let found = solve(&graph, &record, None).expect("solves").at;
        // Down the plane's normal from (3, 4) on it: a sphere 40 below, so 40 − √(20² − 5²) along.
        let foot = origin + rotation * Vector3::new(3.0, 4.0, 0.0);
        let expected = foot - normal * (40.0 - (400.0f64 - 25.0).sqrt());
        assert!((found - expected).norm() < 1e-3, "{found} against {expected}");
    }

    /// **§12.12. The roll turns smoothly as a point crosses the top of a sphere.**
    #[test]
    fn the_roll_turns_smoothly_across_a_sphere() {
        let mut graph = Graph::empty();
        let body = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(30.0) });
        let mut last: Option<Vector3<f64>> = None;
        for i in -40..=40 {
            let found = solve(&graph, &ray(body, f64::from(i) * 0.1, 2.0), None).expect("solves");
            let x = Vector3::from(found.solved.x);
            let n = Vector3::from(found.solved.normal);
            assert!(x.dot(&n).abs() < 1e-9, "the roll lies in the tangent plane");
            if let Some(last) = last {
                assert!(x.dot(&last).clamp(-1.0, 1.0).acos().to_degrees() < 1.0, "at {i}");
            }
            last = Some(x);
        }
    }

    /// **A body under a placement is asked in its own space and answered in world.**
    #[test]
    fn a_body_inside_a_placement_is_answered_in_world() {
        let mut graph = Graph::empty();
        let block = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
        graph.add(Node::Transform {
            input: block.into(),
            translation: [lit(100.0), lit(0.0), lit(0.0)],
            rotation: [lit(0.0); 3],
            scale: lit(1.0),
            about: Place::origin(),
        });
        let found = solve(&graph, &ray(block, 5.0, 5.0), None).expect("solves").at;
        assert!((found - Point3::new(105.0, 5.0, 5.0)).norm() < 1e-3, "{found}");
    }

    /// A body that stands on its own point reads itself; one that does not, does not.
    #[test]
    fn a_body_placed_on_its_own_point_is_a_loop() {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let point = graph.add(Node::Point { at: [lit(0.0); 3] });
        let sphere = graph.add(Node::Sphere { centre: Place::From { node: point }, radius: lit(10.0) });
        let other = graph.add(Node::Sphere { centre: Place::At([lit(50.0); 3]), radius: lit(10.0) });
        doc.surface_points.insert(point, ray(sphere, 0.0, 0.0));
        assert!(reads_itself(&doc, &graph, point));
        doc.surface_points.insert(point, ray(other, 0.0, 0.0));
        assert!(!reads_itself(&doc, &graph, point));
    }

    /// A point on a body built on another surface point is solved after that point.
    #[test]
    fn a_point_on_a_body_on_a_point_is_solved_second() {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let first = graph.add(Node::Point { at: [lit(0.0); 3] });
        let sphere = graph.add(Node::Sphere { centre: Place::From { node: first }, radius: lit(10.0) });
        let second = graph.add(Node::Point { at: [lit(0.0); 3] });
        let block = cuboid(&mut graph, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let part = doc.add_part(fieldform_scene::Part::from_cage("part", fieldform_scene::CageId::new()));
        doc.parts.get_mut(&part).unwrap().graph = graph.clone();
        doc.surface_points.insert(first, ray(block, 0.0, 0.0));
        doc.surface_points.insert(second, ray(sphere, 0.0, 0.0));
        let order = solve_order(&doc);
        let at = |id| order.iter().position(|n| *n == id).unwrap();
        assert!(at(first) < at(second), "{order:?}");
    }

    /// **A touch leaves a spot where the surface is, facing the way it faces**, on a side wall as
    /// on the top, and the solve lands back on it.
    #[test]
    fn a_touch_leaves_a_spot_facing_the_way_the_surface_faces() {
        let mut graph = Graph::empty();
        let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
        let doc = Document::new();
        let at = Point3::new(-20.0, 3.0, 2.0);
        let record = record_for_touch(&doc, &graph, body, at, &mut LeafCache::default()).expect("a record");
        let Guide::Spot { facing, .. } = record.guide else { panic!("a spot") };
        assert!((Vector3::from(facing) + Vector3::x()).norm() < 1e-6, "the wall faces minus X: {facing:?}");
        let found = solve(&graph, &record, Some(at)).expect("solves");
        assert!((found.at - at).norm() < 1e-3, "{}", found.at);
    }

    /// **A plane aimed at a point on a surface lies on the surface**: `Frame::From` the point reads
    /// the surface's normal and roll, and a point not on one still answers up.
    #[test]
    fn a_frame_from_a_surface_point_is_the_surfaces_own() {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let body = cuboid(&mut graph, [0.0, 0.0, 0.0], [40.0, 20.0, 10.0]);
        let point = graph.add(Node::Point { at: [lit(-20.0), lit(3.0), lit(2.0)] });
        let numbers = |r: &ParamRef| doc.resolve(r);
        let (_, up) = resolve_frame(&doc, &graph, &Frame::From { node: point }, &numbers).expect("a frame");
        assert!((up * Vector3::z() - Vector3::z()).norm() < 1e-9, "no record, so up");

        let mut record = spot(body, [-20.0, 3.0, 2.0], [-1.0, 0.0, 0.0]);
        record.solved = Some(solve(&graph, &record, None).expect("solves").solved);
        doc.surface_points.insert(point, record);
        let numbers = |r: &ParamRef| doc.resolve(r);
        let (origin, facing) = resolve_frame(&doc, &graph, &Frame::From { node: point }, &numbers).expect("a frame");
        assert!((origin - Vector3::new(-20.0, 3.0, 2.0)).norm() < 1e-9);
        assert!((facing * Vector3::z() + Vector3::x()).norm() < 1e-6, "Z is the wall's normal: {}", facing * Vector3::z());
        let x = facing * Vector3::x();
        assert!(x.x.abs() < 1e-6 && (facing.matrix().determinant() - 1.0).abs() < 1e-9, "X lies in the wall, right-handed: {x}");
    }

    fn position(app: &App, id: NodeId) -> Point3<f64> {
        match app.part_graph().node(id) {
            Some(Node::Point { at }) => Point3::new(app.resolve(at[0]), app.resolve(at[1]), app.resolve(at[2])),
            other => panic!("a point, not {other:?}"),
        }
    }

    /// An app with a 40 × 20 × 10 block and a point on its top face at (5, -3), typed as a ray.
    fn app_with_a_point() -> Option<(App, NodeId, NodeId)> {
        let mut app = crate::app_tests::headless_app()?;
        crate::app_tests::frame(&mut app, Vec::new());
        app.run_command("cube size=40,20,10 'block'");
        app.run_command("point on=block u=5 v=-3 'spot'");
        crate::app_tests::frame(&mut app, Vec::new());
        let block = app.node_named("block").expect("made");
        let spot = app.node_named("spot").expect("made");
        Some((app, block, spot))
    }

    /// **`point on=` makes a point on the surface, and one undo takes the block and the record.**
    #[test]
    fn a_point_is_made_on_the_surface_from_the_console_and_undone_in_one_step() {
        let Some((mut app, _, spot)) = app_with_a_point() else { return };
        assert!((position(&app, spot) - Point3::new(5.0, -3.0, 5.0)).norm() < 1e-3, "{}", position(&app, spot));
        let record = app.doc.surface_points.get(&spot).copied().expect("a record");
        assert!(record.live && record.solved.is_some());
        let normal = Vector3::from(record.solved.unwrap().normal);
        assert!((normal - Vector3::z()).norm() < 1e-6, "{normal}");
        // Unnamed, because a trailing name is an edit of its own after the making.
        app.run_command("point on=block u=1 v=1");
        crate::app_tests::frame(&mut app, Vec::new());
        let made: Vec<NodeId> = app.doc.surface_points.keys().copied().filter(|p| *p != spot).collect();
        assert_eq!(made.len(), 1);
        app.run_command("undo");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!(app.part_graph().node(made[0]).is_none(), "the block went");
        assert!(!app.doc.surface_points.contains_key(&made[0]), "and the record with it");
        assert!(app.doc.surface_points.contains_key(&spot), "and nothing else");
    }

    /// **`point on=<body> at=x,y,z` is a spot**: the surface nearest the numbers, facing its way, and
    /// a typed `at` afterwards moves the spot rather than being refused.
    #[test]
    fn a_point_typed_at_a_place_is_a_spot_and_a_typed_place_moves_it() {
        let Some((mut app, _, _)) = app_with_a_point() else { return };
        app.run_command("point on=block at=-20,3,2 'side'");
        crate::app_tests::frame(&mut app, Vec::new());
        let side = app.node_named("side").expect("made");
        assert!((position(&app, side) - Point3::new(-20.0, 3.0, 2.0)).norm() < 1e-3, "{}", position(&app, side));
        let record = app.doc.surface_points[&side];
        let Guide::Spot { facing, .. } = record.guide else { panic!("a spot, not {:?}", record.guide) };
        assert!((Vector3::from(facing) + Vector3::x()).norm() < 1e-6, "{facing:?}");

        // Typed well off the wall along its normal: the spot moves, and the surface answers.
        app.run_command("set side.at=-50,-4,1");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, side) - Point3::new(-20.0, -4.0, 1.0)).norm() < 1e-3, "{}", position(&app, side));
        assert!(matches!(app.doc.surface_points[&side].guide, Guide::Spot { .. }), "and it is still a spot");
        app.run_command("undo");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, side) - Point3::new(-20.0, 3.0, 2.0)).norm() < 1e-3, "one step back");
    }

    /// **What stands on a point moves with it, in the same rebuild.** A ball centred on the point
    /// and cut by the block is half a ball only while the point is on the block's flat face, so the
    /// volume is a closed form that fails if the point leaves the surface or the ball is left behind.
    #[test]
    fn a_ball_on_a_point_stays_half_in_the_block_as_the_body_changes() {
        let Some((mut app, block, spot)) = app_with_a_point() else { return };
        let half_ball = 2.0 / 3.0 * std::f64::consts::PI * 8.0;
        let mut graph = app.part_graph().clone();
        let ball = graph.add(Node::Sphere { centre: Place::From { node: spot }, radius: lit(2.0) });
        let cut = graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![block.into(), ball.into()] });
        app.set_part_graph(graph);
        crate::app_tests::frame(&mut app, Vec::new());
        let volume = |app: &App| {
            let graph = app.part_graph().subtree(cut).expect("the cut");
            let field = GraphField::compile_graph(&app.doc, &graph, 1e-3).expect("compiles");
            let (mut inside, h) = (0usize, 0.125);
            let centre = position(app, spot);
            for i in -28..=28 {
                for j in -28..=28 {
                    for k in -28..=28 {
                        // Half a cell off, so no layer of samples sits on the face being measured.
                        let p = centre + Vector3::new(f64::from(i) + 0.5, f64::from(j) + 0.5, f64::from(k) + 0.5) * h;
                        inside += usize::from(field.distance(p) < 0.0);
                    }
                }
            }
            inside as f64 * h * h * h
        };
        assert!((volume(&app) - half_ball).abs() < 0.2, "{} against {half_ball}", volume(&app));
        app.run_command("set block.centre=30,0,0");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - Point3::new(35.0, -3.0, 5.0)).norm() < 1e-3, "{}", position(&app, spot));
        assert!((volume(&app) - half_ball).abs() < 0.2, "{} against {half_ball}", volume(&app));
    }

    /// **A body that moves carries its point**, a number moves it, a typed position on a ray is
    /// refused while it is Live, and Fixed gives the position back.
    #[test]
    fn a_point_follows_its_body_and_its_numbers_and_a_ray_refuses_a_typed_position_until_fixed() {
        let Some((mut app, _, spot)) = app_with_a_point() else { return };
        app.run_command("set block.centre=100,0,0");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - Point3::new(105.0, -3.0, 5.0)).norm() < 1e-3, "{}", position(&app, spot));

        app.run_command("set spot.u=-8");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - Point3::new(92.0, -3.0, 5.0)).norm() < 1e-3, "{}", position(&app, spot));

        let before = position(&app, spot);
        app.run_command("set spot.at=1,2,3");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - before).norm() < 1e-9, "refused, so it stays");

        app.run_command("fix target=spot");
        app.run_command("set block.centre=0,0,0");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - before).norm() < 1e-9, "fixed, so the body leaves it behind");
        app.run_command("set spot.at=1,2,3");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-9, "its own again");

        app.run_command("live target=spot");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - Point3::new(-8.0, -3.0, 5.0)).norm() < 1e-3, "{}", position(&app, spot));
    }

    /// **§12.6. An error never moves a point**: off the body it stays and is red, and moving back
    /// clears it. And §12.14: a variable moves it, and deleting the variable is an error too.
    #[test]
    fn an_error_leaves_the_point_where_it_stood_and_a_variable_drives_it() {
        let Some((mut app, _, spot)) = app_with_a_point() else { return };
        let before = position(&app, spot);
        app.run_command("set spot.u=500");
        crate::app_tests::frame(&mut app, Vec::new());
        assert_eq!(position(&app, spot), before, "an error never moves a point");
        let why = app.surface_point_states.get(&spot).and_then(|s| s.error.clone()).expect("it says why");
        assert!(why.contains("U 500"), "{why}");

        app.run_command("param name=across value=7");
        app.run_command("set spot.u=across");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot).x - 7.0).abs() < 1e-3, "{}", position(&app, spot));
        assert!(app.surface_point_states.get(&spot).is_some_and(|s| s.error.is_none()), "the error clears");
        app.run_command("param name=across value=11");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot).x - 11.0).abs() < 1e-3, "{}", position(&app, spot));

        let id = app.doc.parameter_named("across").expect("the parameter");
        assert!(app.doc.parameter_is_read(id, None), "a record that names it reads it");
    }

    /// **§12.9. A body placed on its own point makes the point Fixed**, nothing moves, it says why,
    /// and Live is refused while the loop stands.
    #[test]
    fn a_body_placed_on_its_own_point_fixes_it_and_live_is_refused() {
        let Some((mut app, _, spot)) = app_with_a_point() else { return };
        let mut graph = app.part_graph().clone();
        graph.add(Node::Sphere { centre: Place::From { node: spot }, radius: lit(4.0) });
        app.set_part_graph(graph);
        crate::app_tests::frame(&mut app, Vec::new());
        app.run_command("sphere radius=4 'other'");
        let ball = app
            .part_graph()
            .nodes()
            .find(|(_, n)| matches!(n, Node::Sphere { centre: Place::From { .. }, .. }))
            .map(|(id, _)| id)
            .expect("the ball on the point");
        let record = app.doc.surface_points.get(&spot).copied().expect("a record");
        app.commit_surface_point(spot, record, SurfacePoint { body: ball, ..record });
        crate::app_tests::frame(&mut app, Vec::new());
        let record = app.doc.surface_points.get(&spot).copied().expect("a record");
        assert!(!record.live, "fixed where it stands");
        assert!(app.console_log.iter().any(|l| l.text.contains("is now fixed")), "and it says why");
        let before = position(&app, spot);
        app.run_command("live target=spot");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!(!app.doc.surface_points[&spot].live, "refused while the loop stands");
        assert_eq!(position(&app, spot), before);
    }

    /// **A drag slides a point over its surface and round the corner**: every instant lands on the
    /// body, and what is written at the end is where it was left, as one undo step. A ray keeps its
    /// kind and reads its U and V off where it landed.
    #[test]
    fn a_dragged_point_slides_over_the_surface_and_lands_where_it_was_left() {
        let Some((mut app, _, spot)) = app_with_a_point() else { return };
        // Asked for a place above the top face: it is put on the face, not left in the air.
        let slid = app.slide_surface_point(spot, Point3::new(8.0, 2.0, 99.0)).expect("slides");
        assert!((slid - Point3::new(8.0, 2.0, 5.0)).norm() < 1e-3, "{slid}");
        // Asked for a place past the edge: it goes round onto the side wall, facing that way.
        let round = app.slide_surface_point(spot, Point3::new(30.0, 2.0, 4.0)).expect("slides");
        assert!((round - Point3::new(20.0, 2.0, 4.0)).norm() < 1e-3, "{round}");
        let normal = Vector3::from(app.doc.surface_points[&spot].solved.unwrap().normal);
        assert!((normal - Vector3::x()).norm() < 1e-6, "the instant's answer faces the wall: {normal}");

        let graph = app.part_graph().clone();
        assert!(app.surface_point_dragged(spot, Point3::new(8.0, 2.0, 5.0), graph));
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - Point3::new(8.0, 2.0, 5.0)).norm() < 1e-3, "{}", position(&app, spot));
        assert!(matches!(app.doc.surface_points[&spot].guide, Guide::Ray { .. }), "a ray stays a ray");
        assert!(app.slide.is_none(), "the gesture's body is let go");
        app.run_command("undo");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, spot) - Point3::new(5.0, -3.0, 5.0)).norm() < 1e-3, "{}", position(&app, spot));
    }

    /// **The dot's own drag is one undo entry, and a press that never moved it is none.**
    #[test]
    fn the_dots_own_drag_commits_once_and_a_click_commits_nothing() {
        let Some((mut app, _, _)) = app_with_a_point() else { return };
        app.run_command("point on=block at=0,0,5 'dot'");
        crate::app_tests::frame(&mut app, Vec::new());
        let dot = app.node_named("dot").expect("made");
        let entries = app.stack.history().0.len();
        app.grab_surface_point(dot);
        app.end_surface_point_grab();
        assert_eq!(app.stack.history().0.len(), entries, "a click is not an edit");
        assert!(app.slide.is_none());

        app.grab_surface_point(dot);
        for x in [2.0, 4.0, 6.0] {
            let Some(slide) = app.slide.as_ref() else { panic!("a gesture") };
            let (at, normal) = slide.asked.nearest(Point3::new(x, 1.0, 9.0)).expect("on the body");
            app.show_slid(dot, at, normal);
            let mut graph = app.part_graph().clone();
            if let Some(Node::Point { at: slot }) = graph.node_mut(dot) {
                *slot = [at.x, at.y, at.z].map(ParamRef::Literal);
            }
            app.set_part_graph(graph);
        }
        app.end_surface_point_grab();
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, dot) - Point3::new(6.0, 1.0, 5.0)).norm() < 1e-3, "{}", position(&app, dot));
        let Guide::Spot { at, .. } = app.doc.surface_points[&dot].guide else { panic!("a spot") };
        assert!((app.resolve(at[0]) - 6.0).abs() < 1e-3, "the question moved with it");
        app.run_command("undo");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, dot) - Point3::new(0.0, 0.0, 5.0)).norm() < 1e-3, "one step back: {}", position(&app, dot));
    }

    /// **The echo says the question, not the answer**, and running it makes the same point — a ray
    /// and a spot alike.
    #[test]
    fn the_echo_of_a_point_on_a_surface_makes_the_same_point() {
        let Some((mut app, _, spot)) = app_with_a_point() else { return };
        app.run_command("set spot.dir=x spot.u=2 spot.v=1");
        app.run_command("point on=block at=-20,3,2 'side'");
        crate::app_tests::frame(&mut app, Vec::new());
        let side = app.node_named("side").expect("made");
        for (id, word, key) in [(spot, "spot", "dir=x"), (side, "side", "facing=")] {
            let lines = crate::echo::create_lines(&app.doc, app.part_graph(), id).expect("echoes");
            assert!(lines[0].contains("on=block") && lines[0].contains(key), "{lines:?}");
            let copy = format!("{word}_copy");
            let line = lines[0].replace(&format!("'{word}'"), &format!("'{copy}'"));
            app.run_command(&line);
            crate::app_tests::frame(&mut app, Vec::new());
            let made = app.node_named(&copy).expect("made");
            assert!((position(&app, made) - position(&app, id)).norm() < 1e-6, "{word}");
        }
    }

    /// **A plane on a surface stands square to it and follows the body**, by the console and by
    /// the click's own path, as one undo step with its point.
    #[test]
    fn a_plane_on_a_surface_stands_square_to_it_and_follows_the_body() {
        let Some((mut app, _, _)) = app_with_a_point() else { return };
        app.run_command("plane on=block at=-20,3,2 'face'");
        crate::app_tests::frame(&mut app, Vec::new());
        let plane = app.node_named("face").expect("made");
        let normal_of = |app: &App| app.placement_above(plane).isometry.rotation * Vector3::z();
        assert!((normal_of(&app) + Vector3::x()).norm() < 1e-6, "square to the side wall: {}", normal_of(&app));
        let Some(Node::Plane { origin: Place::From { node: point }, turn: Turn::From { node: same }, .. }) =
            app.part_graph().node(plane).cloned()
        else {
            panic!("a plane standing on a point")
        };
        assert_eq!(point, same);
        assert!(app.doc.surface_points.contains_key(&point));

        // The body moves and turns: the plane goes with the wall.
        app.run_command("set block.centre=10,0,0");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!((position(&app, point) - Point3::new(-10.0, 3.0, 2.0)).norm() < 1e-3, "{}", position(&app, point));
        assert!((normal_of(&app) + Vector3::x()).norm() < 1e-6);

        // And a plane on an existing point, from the console.
        app.run_command("point on=block at=0,0,5 'top'");
        app.run_command("plane on=top 'lid'");
        crate::app_tests::frame(&mut app, Vec::new());
        let lid = app.node_named("lid").expect("made");
        let up = app.placement_above(lid).isometry.rotation * Vector3::z();
        assert!((up - Vector3::z()).norm() < 1e-6, "{up}");

        // Unnamed, because a trailing name is an edit of its own: one undo takes the plane and
        // the point it stood on together.
        let before: Vec<NodeId> = app.part_graph().nodes().map(|(id, _)| id).collect();
        app.run_command("plane on=block at=0,0,-5");
        crate::app_tests::frame(&mut app, Vec::new());
        let made: Vec<NodeId> = app.part_graph().nodes().map(|(id, _)| id).filter(|id| !before.contains(id)).collect();
        assert_eq!(made.len(), 2, "a point and a plane");
        app.run_command("undo");
        crate::app_tests::frame(&mut app, Vec::new());
        assert!(made.iter().all(|id| app.part_graph().node(*id).is_none()), "both gone in one step");
        assert!(made.iter().all(|id| !app.doc.surface_points.contains_key(id)), "and the record with them");
    }
}
