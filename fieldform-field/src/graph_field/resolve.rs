// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a block stands and which way it faces: frames, places, pivots, turns and ends, resolved.

use super::{CompileError, operand_extent, place_at};
use fieldform_scene::{Document, Graph, NodeId};
use nalgebra::{Point3, Rotation3, Vector3};

/// Resolves a primitive's centre, refusing rather than defaulting an unresolvable one.
///
/// A position that quietly became the origin would put the shape somewhere plausible and
/// wrong, which is the failure mode this codebase keeps refusing.
pub(crate) fn resolve_point(
    centre: &[fieldform_scene::ParamRef; 3],
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<Point3<f64>, CompileError> {
    let mut out = [0.0; 3];
    for (slot, r) in out.iter_mut().zip(centre) {
        *slot = resolve(r).ok_or(CompileError::UnresolvedCentre)?;
    }
    Ok(Point3::new(out[0], out[1], out[2]))
}

/// Which way a [`Sweep`] runs, in world coordinates.
///
/// A curve gives its direction end to end, which is what makes "extrude along that line" mean
/// the line's direction rather than its position -- the line can be anywhere, and usually is.
///
/// [`Sweep`]: fieldform_scene::Sweep
pub fn sweep_direction(
    document: &fieldform_scene::Document,
    sweep: &fieldform_scene::Sweep,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Option<Vector3<f64>> {
    match sweep {
        // Handled by the caller, which knows the profile's plane; this only answers for the
        // directions that are stated rather than derived.
        fieldform_scene::Sweep::Normal => None,
        fieldform_scene::Sweep::Along { curve } => {
            let c = document.curves.get(curve)?;
            let (from, to) = c.ends();
            (to - from).try_normalize(1e-12)
        }
        fieldform_scene::Sweep::Toward { xyz } => {
            let mut out = Vector3::zeros();
            for (slot, r) in out.iter_mut().zip(xyz) {
                *slot = resolve(r)?;
            }
            out.try_normalize(1e-12)
        }
    }
}

/// The two points an [`Ends`] names, for the bounds walk.
///
/// The same question [`resolve_ends`] answers, asked where nothing may fail: a bound is computed
/// per frame and for every node, so a curve that has gone missing yields a degenerate span here
/// and is reported once, properly, by the compile.
pub fn ends_at(
    ends: &fieldform_scene::Ends,
    document: &fieldform_scene::Document,
    graph: &Graph,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> f64,
) -> (Point3<f64>, Point3<f64>) {
    match ends {
        fieldform_scene::Ends::At { from, to } => {
            (place_at(document, graph, from, resolve), place_at(document, graph, to, resolve))
        }
        fieldform_scene::Ends::From { curve } => document
            .curves
            .get(curve)
            .map(|c| c.ends())
            .unwrap_or((Point3::origin(), Point3::origin())),
        // Standing on a block that has a facing: from its origin, along its normal. The same
        // reading `resolve_ends` does, and it cannot fail here for the same reason nothing else
        // in this walk can -- a bound is computed every frame for every node.
        fieldform_scene::Ends::On { node, length } => {
            let numbers = |r: &fieldform_scene::ParamRef| Some(resolve(r));
            match resolve_frame(
                document,
                graph,
                &fieldform_scene::Frame::From { node: *node },
                &numbers,
            ) {
                Ok((origin, rotation)) => {
                    let from = Point3::from(origin);
                    (from, from + rotation * Vector3::z() * resolve(length))
                }
                Err(_) => (Point3::origin(), Point3::origin()),
            }
        }
    }
}

/// The two points an [`Ends`] names.
///
/// **A curve's ends are evaluated, not read off its control points** — see [`Curve::ends`],
/// which is shared with the bounds walk so the two cannot disagree about where a cylinder is.
///
/// [`Curve::ends`]: fieldform_scene::Curve::ends
///
/// The shape between the ends is ignored, deliberately: a cylinder aimed at an arc is a straight
/// cylinder between that arc's endpoints. Sweeping *along* a curve is `Node::Pipe`.
pub(crate) fn resolve_ends(
    ends: &fieldform_scene::Ends,
    document: &fieldform_scene::Document,
    graph: &Graph,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<(Point3<f64>, Point3<f64>), CompileError> {
    match ends {
        // Either end is a place, so either can sit on a point -- two bars ending at one point
        // stay ending at one point when it moves.
        fieldform_scene::Ends::At { from, to } => Ok((
            resolve_place(document, graph, from, resolve)?,
            resolve_place(document, graph, to, resolve)?,
        )),
        fieldform_scene::Ends::From { curve } => {
            let c = document
                .curves
                .get(curve)
                .ok_or(CompileError::MissingCurve(*curve))?;
            Ok(c.ends())
        }
        // **Standing on a block that has a facing**: the ends run from its origin along its
        // normal, this long. Read through `resolve_frame`, which is the one reading of "where
        // and which way" in this codebase -- so a cylinder standing on a plane, a mirror in that
        // plane and a cuboid aimed at it all agree about which way it faces, and all three move
        // when it does.
        fieldform_scene::Ends::On { node, length } => {
            let (origin, rotation) = resolve_frame(
                document,
                graph,
                &fieldform_scene::Frame::From { node: *node },
                resolve,
            )?;
            let length = resolve(length).ok_or(CompileError::UnresolvedCentre)?;
            let from = Point3::from(origin);
            Ok((from, from + rotation * Vector3::z() * length))
        }
    }
}

/// Resolves a transform's translation or rotation, refusing an unresolvable one for the same
/// reason [`resolve_point`] does: a placement that quietly became the identity would put the
/// body somewhere plausible and wrong.
pub(crate) fn resolve_placement(
    triple: &[fieldform_scene::ParamRef; 3],
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<Vector3<f64>, CompileError> {
    let mut out = [0.0; 3];
    for (slot, r) in out.iter_mut().zip(triple) {
        *slot = resolve(r).ok_or(CompileError::UnresolvedPlacement)?;
    }
    Ok(Vector3::new(out[0], out[1], out[2]))
}

/// Where a place is, in millimetres.
///
/// **The counterpart of [`resolve_frame`], and deliberately the only one.** A place is asked for
/// by a primitive's centre and a plane's origin, and the field and the shader both come here
/// rather than each working it out — the same arrangement that keeps a frame meaning one thing
/// in both. A second reading is a second answer, and the two agree until they do not.
///
/// A place taken from a block asks that block where it is. Two can say: a point, which is a
/// position and nothing else, and a plane, whose own origin is a place in turn — so a plane can
/// sit on a point and a body can sit on the plane. That recursion is bounded rather than trusted:
/// the picker will not offer anything that consumes the asker, but a hand-edited file is not the
/// picker, and a loop here would be a stack overflow rather than an error.
pub fn resolve_place(
    document: &Document,
    graph: &Graph,
    place: &fieldform_scene::Place,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<Point3<f64>, CompileError> {
    resolve_place_at(document, graph, place, resolve, 0)
}

/// Everything a `Transform` above `target` does to its coordinates.
///
/// A node's own numbers are read in its own frame; a placement wrapping it turns that frame
/// into the world. Needed by [`resolve_place_for`], which has to speak both.
///
/// **The one walk.** The application had a second copy that read the unexpanded graph and
/// composed each block's own turn as it went -- two implementations of one idea, which is what
/// D9 exists to stop, and which cost three rounds of the same bug being fixed in one of them and
/// not the other. The application passes its graph through [`with_turns_flattened`] and calls
/// this instead: a turned block *is* that block inside a placement, so once the turns are
/// placements there is nothing left for a second walk to know.
pub fn placement_above(
    document: &Document,
    graph: &Graph,
    target: NodeId,
    without: Option<NodeId>,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> nalgebra::Similarity3<f64> {
    placement_above_at(document, graph, target, without, resolve, 0)
}

/// Where a placement walk starts: every block that is not an *ingredient* of another.
///
/// **Not [`Graph::roots`].** Roots are the bodies, and a plane a cuboid is aimed at is not one --
/// something consumes it. But the walk descends through operands only, so starting at the bodies
/// never reaches it, and a block the walk never reaches is a block with no placement: the square
/// drawn for it snapped back to the origin, unturned, the moment anything was aimed at it.
/// Reported twice in one message, of a torus and of a gyroid.
///
/// So the seeds are the blocks nothing is *built from*, which is the bodies plus the ones that
/// are only pointed at. Each of those stands where it stands, which is exactly what a walk down
/// from it says.
pub(crate) fn placement_seeds(graph: &Graph) -> Vec<NodeId> {
    let consumed: std::collections::HashSet<NodeId> =
        graph.nodes().flat_map(|(_, n)| n.operands()).collect();
    let mut out: Vec<NodeId> =
        graph.nodes().map(|(id, _)| id).filter(|id| !consumed.contains(id)).collect();
    // Sorted, because this decides the order two paths to one node are tried in, and an order
    // that varies between runs is an answer that varies between runs.
    out.sort_by_key(|id| id.raw());
    out
}

fn placement_above_at(
    document: &Document,
    graph: &Graph,
    target: NodeId,
    without: Option<NodeId>,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    depth: usize,
) -> nalgebra::Similarity3<f64> {
    let identity = nalgebra::Similarity3::identity();
    if depth > REFERENCE_DEPTH {
        return identity;
    }
    // Down from each body, composing on the right as it descends: the outermost placement is
    // met first and applies last, which is the order the field itself evaluates them in.
    for seed in placement_seeds(graph) {
        let mut stack = vec![(seed, identity)];
        let mut depth = 0usize;
        while let Some((id, above)) = stack.pop() {
            depth += 1;
            if depth > 4096 {
                break;
            }
            // **Not through the block that is standing on it.** A place edge makes the source
            // an input of its consumer, which would otherwise put the source inside the
            // consumer's placement -- so asking where a point is would answer "wherever the
            // thing sitting on it is", and binding a plane to a point would move the point.
            // Reported as a plane's origin landing off in space; it was both of them landing
            // off in space together.
            if Some(id) == without {
                continue;
            }
            let Some(node) = graph.node(id) else { continue };
            let here = match node {
                fieldform_scene::Node::Transform { .. } => {
                    above * placement_similarity_at(document, graph, id, node, resolve, depth + 1)
                }
                _ => above,
            };
            if id == target {
                return here;
            }
            stack.extend(node.operands().into_iter().map(|i| (i, here)));
        }
    }
    identity
}

/// The frame a placement's own numbers are written in, which is the one **outside** it.
///
/// [`placement_above`] answers with the frame a block sits *in*, which for a placement includes
/// what that placement itself does. Its own move, turn and pivot are not written in that frame —
/// they are written in the frame its parent establishes and which it then changes. So a pivot
/// resolved through `placement_above` comes back with this placement already applied to it,
/// which is the placement it is meant to be defining.
fn placement_outside_at(
    document: &Document,
    graph: &Graph,
    target: NodeId,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    depth: usize,
) -> nalgebra::Similarity3<f64> {
    let identity = nalgebra::Similarity3::identity();
    if depth > REFERENCE_DEPTH {
        return identity;
    }
    for seed in placement_seeds(graph) {
        let mut stack = vec![(seed, identity)];
        let mut depth = 0usize;
        while let Some((id, above)) = stack.pop() {
            depth += 1;
            if depth > 4096 {
                break;
            }
            // Answered before this node's own placement is composed, which is the whole
            // difference from `placement_above`.
            if id == target {
                return above;
            }
            let Some(node) = graph.node(id) else { continue };
            let here = match node {
                fieldform_scene::Node::Transform { .. } => {
                    above * placement_similarity_at(document, graph, id, node, resolve, depth + 1)
                }
                _ => above,
            };
            stack.extend(node.operands().into_iter().map(|i| (i, here)));
        }
    }
    identity
}

/// What a placement does, as a similarity, pivot included.
///
/// **One answer, because four things ask.** The compile builds an `Op::Transform` from it, the
/// extent carries a box through it, and both placement walks compose it — and a pivot that four
/// sites each worked out for themselves would be four chances to get the cancelling translation
/// wrong in a way only one of them shows.
///
/// The pivot is a fixed point: `x` goes to `c + s·R·(x − c) + t`. That is the ordinary placement
/// with its translation shifted by `c − s·R·c`, so nothing downstream has to know a pivot
/// exists — which is why `Op::Transform` still takes three numbers and the shader still takes
/// one matrix.
pub(crate) fn placement_similarity(
    document: &Document,
    graph: &Graph,
    id: NodeId,
    node: &fieldform_scene::Node,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> nalgebra::Similarity3<f64> {
    placement_similarity_at(document, graph, id, node, resolve, 0)
}

fn placement_similarity_at(
    document: &Document,
    graph: &Graph,
    id: NodeId,
    node: &fieldform_scene::Node,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    depth: usize,
) -> nalgebra::Similarity3<f64> {
    let fieldform_scene::Node::Transform { translation, rotation, scale, about, .. } = node else {
        return nalgebra::Similarity3::identity();
    };
    if depth > REFERENCE_DEPTH {
        return nalgebra::Similarity3::identity();
    }
    let number = |r: &fieldform_scene::ParamRef| resolve(r).unwrap_or(0.0);
    let t = Vector3::new(
        number(&translation[0]),
        number(&translation[1]),
        number(&translation[2]),
    );
    let r = Vector3::new(
        number(&rotation[0]),
        number(&rotation[1]),
        number(&rotation[2]),
    );
    let s = resolve(scale).unwrap_or(1.0).max(1e-9);
    let turn = nalgebra::UnitQuaternion::from_euler_angles(
        r.x.to_radians(),
        r.y.to_radians(),
        r.z.to_radians(),
    );
    let c = resolve_pivot_at(document, graph, id, about, resolve, depth).unwrap_or_else(|_| Point3::origin());
    nalgebra::Similarity3::from_parts(
        nalgebra::Translation3::from(t + c.coords - turn * (c.coords * s)),
        turn,
        s,
    )
}

/// Where a placement's pivot is, in the frame the placement's own numbers are written in.
///
/// Typed numbers are already there. A pivot taken from a point has to come out of that point's
/// own placements into the world and back in through the placements *outside* this one — not
/// through this one, which is the thing being defined.
pub fn resolve_pivot(
    document: &Document,
    graph: &Graph,
    placement: NodeId,
    about: &fieldform_scene::Place,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<Point3<f64>, CompileError> {
    resolve_pivot_at(document, graph, placement, about, resolve, 0)
}

fn resolve_pivot_at(
    document: &Document,
    graph: &Graph,
    placement: NodeId,
    about: &fieldform_scene::Place,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    depth: usize,
) -> Result<Point3<f64>, CompileError> {
    let fieldform_scene::Place::From { node } = about else {
        return resolve_place_at(document, graph, about, resolve, depth + 1);
    };
    if depth > REFERENCE_DEPTH {
        return resolve_place_at(document, graph, about, resolve, depth + 1);
    }
    let local = resolve_place_at(document, graph, about, resolve, depth + 1)?;
    let world = placement_above_at(document, graph, *node, Some(placement), resolve, depth + 1) * local;
    Ok(placement_outside_at(document, graph, placement, resolve, depth + 1).inverse() * world)
}

/// Where a place is, **in the frame of the block that holds it**.
///
/// **Because "sit on that point" means sit on it, not near it.** A block's own coordinates are
/// read inside whatever placements wrap it, and a point's are read inside whatever wrap *it* —
/// so handing one block's numbers to another mixes two frames. Measured before it was fixed: a
/// plane sitting on a point at `z = 7`, with a placement of `+20` above the plane, put its
/// surface at `z = 27`. Every plane made while drawing on a plane carries such a placement, so
/// this was most of them.
///
/// The source's position goes out to the world through its own placements and comes back into
/// the asker's frame through the inverse of theirs. With neither wrapped in anything, which is
/// the common case, both are the identity and this is exactly [`resolve_place`].
pub fn resolve_place_for(
    document: &Document,
    graph: &Graph,
    consumer: NodeId,
    place: &fieldform_scene::Place,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<Point3<f64>, CompileError> {
    let fieldform_scene::Place::From { node } = place else {
        // Typed numbers are already in this block's own frame, which is where they were typed.
        return resolve_place_at(document, graph, place, resolve, 0);
    };
    let local = resolve_place_at(document, graph, place, resolve, 0)?;
    // The source's own placement, ignoring the path through whatever is standing on it -- so a
    // point reached only that way is where it is in the world, which is what a point means.
    let world = placement_above(document, graph, *node, Some(consumer), resolve) * local;
    Ok(placement_above(document, graph, consumer, None, resolve).inverse() * world)
}

fn resolve_place_at(
    document: &Document,
    graph: &Graph,
    place: &fieldform_scene::Place,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    depth: usize,
) -> Result<Point3<f64>, CompileError> {
    // The same guard the other graph walks use. A cyclic graph is refused at compile time, but
    // this runs as part of that compile and must not recurse forever on the way to saying so.
    if depth > REFERENCE_DEPTH {
        return Err(CompileError::Cyclic);
    }
    match place {
        fieldform_scene::Place::At(xyz) => resolve_point(xyz, resolve),
        fieldform_scene::Place::From { node } => match graph.node(*node) {
            // A point is a position and nothing else, which is exactly a place.
            Some(fieldform_scene::Node::Point { at }) => resolve_point(at, resolve),
            // A plane has one too, and its own may be taken from something further down.
            Some(fieldform_scene::Node::Plane { origin, .. }) => {
                resolve_place_at(document, graph, origin, resolve, depth + 1)
            }
            // **Through a placement to whatever it holds.** A block that carries a turn *is*
            // that block inside a `Node::Transform` -- `Graph::with_turns_expanded` says so, and
            // the compiler reads the expanded graph -- so by the time this runs, a plane that
            // has been turned at all is a placement. Refusing it meant a block could sit on a
            // plane only while that plane was left facing up.
            Some(placed @ fieldform_scene::Node::Transform { input, .. }) => {
                // **A placement with nothing in it holds no point and no plane**, so it supplies
                // no place -- which is the answer this arm's `_` already gives for a block that
                // never could, said through the same refusal rather than a second one. A block
                // *aimed* at a hole is downstream of it and is skipped whole before the compile
                // reads a place at all; this is for the readers that ask outside that loop.
                let held = input.source().ok_or(CompileError::NotAPlaceSource(*node))?;
                let inner = resolve_place_at(
                    document,
                    graph,
                    &fieldform_scene::Place::From { node: held },
                    resolve,
                    depth + 1,
                )?;
                Ok(placement_similarity_at(document, graph, *node, placed, resolve, depth + 1) * inner)
            }
            _ => Err(CompileError::NotAPlaceSource(*node)),
        },
        // **Where a curve begins.** The one reading, so a plane standing at the start of a pipe
        // and the pipe itself agree about where that is -- and follow together when the curve is
        // redrawn, which is what a position naming a curve is for.
        fieldform_scene::Place::StartOf { curve } => document
            .curves
            .get(curve)
            .map(|c| c.ends().0)
            .ok_or(CompileError::MissingCurve(*curve)),
    }
}

/// How far any of these walks may follow one reference into another before giving up.
///
/// **Because a reference can lead to a reference.** A plane's facing can be taken from another
/// plane, a placement's pivot from a point that another placement stands on, and each of those
/// walks calls the next: `resolve_frame` reads a turn, a turn reads a frame; `placement_above`
/// composes a placement, a placement resolves a pivot, a pivot walks the placements above its
/// source. Every one of those pairs is mutual recursion, and until this existed none of them had
/// a bound — the per-walk iteration counters count *steps within one walk*, which is a different
/// quantity and does not stop the stack growing.
///
/// A graph that closes a loop is refused at compile time, so this is not how a cycle is caught.
/// What it is for is the state a document passes through while it is being edited, where the
/// panel asks these questions every frame on whatever the graph currently says. Running out of
/// stack there kills the window without a panic and so without a crash log, which is what
/// happened: the window vanished and the log was empty.
///
/// Sixty-four is far past any real chain — the deepest thing a person builds is a handful — and
/// far short of the stack.
const REFERENCE_DEPTH: usize = 64;

/// What a block's own turn does to whichever way it was facing.
///
/// **The one reading of a turn**, so a block aimed at a plane points the same way in the field,
/// in the shader, and in the square the panel draws for it. Typed angles are intrinsic XYZ in
/// degrees, which is what a `Node::Transform`'s rotation is and what the panel shows; a turn
/// taken from a plane or a line is the rotation that carries `+Z` onto that direction, which is
/// exactly what [`resolve_frame`] already answers for every aimed block.
///
/// A source that cannot be read leaves the block unturned rather than failing. A turn is read
/// while a document is being edited — mid-drag, with a curve half-drawn — and an error there
/// would take the viewport with it; pointing the way it pointed before is the answer that lets
/// the next edit fix it.
pub fn turn_rotation(
    document: &fieldform_scene::Document,
    graph: &Graph,
    turn: &fieldform_scene::Turn,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Rotation3<f64> {
    turn_rotation_at(document, graph, turn, resolve, 0)
}

fn turn_rotation_at(
    document: &fieldform_scene::Document,
    graph: &Graph,
    turn: &fieldform_scene::Turn,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    depth: usize,
) -> Rotation3<f64> {
    let aim = |frame: fieldform_scene::Frame| {
        resolve_frame_at(document, graph, &frame, resolve, depth + 1)
            .map(|(_, rotation)| rotation)
            .unwrap_or_else(|_| Rotation3::identity())
    };
    match turn {
        fieldform_scene::Turn::By(angles) => {
            let number = |r: &fieldform_scene::ParamRef| resolve(r).unwrap_or(0.0);
            Rotation3::from_euler_angles(
                number(&angles[0]).to_radians(),
                number(&angles[1]).to_radians(),
                number(&angles[2]).to_radians(),
            )
        }
        fieldform_scene::Turn::From { node } => aim(fieldform_scene::Frame::From { node: *node }),
        fieldform_scene::Turn::Along { curve } => {
            aim(fieldform_scene::Frame::Along { curve: *curve })
        }
        // Read through the frame's own arm, so two points mean one thing whether a block is
        // aimed by a `Frame` or carries its facing as a `Turn`.
        fieldform_scene::Turn::Between { from, to } => {
            aim(fieldform_scene::Frame::Between { from: *from, to: *to })
        }
        fieldform_scene::Turn::Tangent { curve } => {
            aim(fieldform_scene::Frame::Tangent { curve: *curve })
        }
        // Nothing: the rotation this block was turned by is in the placement above it, and the
        // references are all that is left. Reading it again is the double turn this variant
        // exists to stop. See [`fieldform_scene::Turn::Absorbed`].
        fieldform_scene::Turn::Absorbed { .. } => Rotation3::identity(),
    }
}

/// A graph with every turn rewritten as the placement it means.
///
/// **What both compilers call**, in place of `Graph::with_turns_expanded` — that one cannot read
/// an aimed turn, because working out where a plane or a line points needs the document and a
/// resolver for named numbers, and neither belongs in the scene crate. This is the one function
/// that has all three, so the field and the shader cannot come to different answers about which
/// way a block is turned.
pub fn with_turns_flattened(
    document: &fieldform_scene::Document,
    graph: &Graph,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Graph {
    graph.with_turns_expanded_using(&|_, turn| {
        let (x, y, z) = turn_rotation(document, graph, turn, resolve).euler_angles();
        Some([x.to_degrees(), y.to_degrees(), z.to_degrees()])
    })
}

/// Where a frame sits and which way it points, resolved to numbers.
///
/// **The one reader of the shared type (D9).** Every block that takes a [`Frame`] resolves it
/// here, so "what can supply a frame" is answered once and they all agree. A block that
/// resolved its own would be free to accept something the others do not, which is the state
/// this type exists to end.
///
/// Returns the origin and the rotation carrying `+Z` onto the frame's axis — the pair a
/// transform is built from, since aiming anything at a frame is putting its points into that
/// frame first.
pub fn resolve_frame(
    document: &fieldform_scene::Document,
    graph: &Graph,
    frame: &fieldform_scene::Frame,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<(Vector3<f64>, Rotation3<f64>), CompileError> {
    resolve_frame_at(document, graph, frame, resolve, 0)
}

fn resolve_frame_at(
    document: &fieldform_scene::Document,
    graph: &Graph,
    frame: &fieldform_scene::Frame,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    depth: usize,
) -> Result<(Vector3<f64>, Rotation3<f64>), CompileError> {
    if depth > REFERENCE_DEPTH {
        return Err(CompileError::Cyclic);
    }
    match frame {
        fieldform_scene::Frame::At { origin, axis } => Ok((
            resolve_placement(origin, resolve)?,
            rotation_from_z(resolve_placement(axis, resolve)?),
        )),
        // Taken from another block. Three can answer, and anything else is refused by name
        // rather than guessed at.
        fieldform_scene::Frame::From { node } => match graph.node(*node) {
            // A plane has a point and a facing, which is exactly a frame.
            //
            // **Its own turn is part of that facing.** A plane carries one now, and this read
            // ±Z regardless — so a mirror or a twist aimed at a plane that had been turned was
            // aimed somewhere the plane was not, which is half of how "I can't specify a plane"
            // was reported. The other half is the placement arm below.
            Some(fieldform_scene::Node::Plane { origin, flipped, turn }) => {
                let up = if *flipped { -Vector3::z() } else { Vector3::z() };
                Ok((
                    // **`_for`, with the plane as its own consumer — the same call the plane's
                    // leaf makes** (see the `Node::Plane` arm of the compile above). A place
                    // that names another block is read in that block's world position and then
                    // brought back into the frame this one is written in, and `resolve_place_at`
                    // does only the first half. So a plane standing on a point, with a placement
                    // anywhere above either of them, came out here at the point's world position
                    // — which the placement then moved *again*. The plane was drawn in one place
                    // and mirrored about another, by exactly the placement: measured at 10 mm
                    // each way by `plane_origin_probe`, in opposite directions depending on
                    // which of the two carried the move.
                    resolve_place_for(document, graph, *node, origin, resolve)?.coords,
                    turn_rotation_at(document, graph, turn, resolve, depth + 1)
                        * rotation_from_z(up),
                ))
            }
            // **Through a placement to whatever it holds**, which is the same fact
            // `resolve_place_at` reads: a turned block is that block inside a
            // `Node::Transform`, and the compiler sees the expanded graph. Aiming at one was
            // `NotAFrameSource`, so a plane could be aimed at only while it was left facing up.
            Some(placed @ fieldform_scene::Node::Transform { input, .. }) => {
                // Empty, and so no frame either, for the reason `resolve_place_at` gives: there
                // is no block under the placement to take an origin or a facing from.
                let held = input.source().ok_or(CompileError::NotAFrameSource(*node))?;
                let (origin, rotation) = resolve_frame_at(
                    document,
                    graph,
                    &fieldform_scene::Frame::From { node: held },
                    resolve,
                    depth + 1,
                )?;
                let by = placement_similarity_at(document, graph, *node, placed, resolve, depth + 1);
                Ok((
                    (by * Point3::from(origin)).coords,
                    by.isometry.rotation.to_rotation_matrix() * rotation,
                ))
            }
            // **A point answers the place and leaves the facing at `+Z`.** Asked for directly:
            // draw a point and put another block's origin on it, so moving the point moves
            // everything aimed at it. A point genuinely has no direction, and `+Z` is the same
            // answer a zero axis already gets here rather than a new convention.
            //
            // Which means a point is the right source for a block that wants a *place* -- a
            // lattice's centre, a pattern's hub -- and only half an answer for one that wants a
            // direction too. A revolve aimed at a point spins about the world's up through it,
            // which is a real frame and may not be the one that was meant; the block's own axis
            // controls are still there to say otherwise.
            Some(fieldform_scene::Node::Point { at }) => Ok((
                resolve_point(at, resolve)?.coords,
                rotation_from_z(Vector3::z()),
            )),
            _ => Err(CompileError::NotAFrameSource(*node)),
        },
        // Taken from a curve: it begins where the curve begins and points the way the curve
        // runs. This is the second answer the comment here promised when frames were built —
        // "a curve or a pair of points could answer too, and the place to add them is here,
        // once" — and adding it here is why every aimed block gained it at the same moment.
        //
        // End minus start rather than the tangent at the start: a line drawn to mean an axis
        // means the direction it runs overall, and for a line the two agree anyway.
        // `rotation_from_z` takes a zero vector to `+Z`, so a closed curve is a frame pointing
        // up rather than an error thrown mid-drag.
        fieldform_scene::Frame::Along { curve } => {
            let c = document
                .curves
                .get(curve)
                .ok_or(CompileError::MissingCurve(*curve))?;
            let (from, to) = c.ends();
            Ok((from.coords, rotation_from_z(to - from)))
        }
        // **Two blocks: it sits at the first and points at the second.** The same reading as a
        // curve, with the two ends named separately -- which is what somebody says when there is
        // no line to draw. Backlog items 18 and 25 both asked for it and both parked on settling
        // it once; every block aimed by a frame gains it here together, and a `Turn` reads it
        // through this same arm.
        //
        // Each end is asked for a *place* and nothing more, so a point answers, and so does
        // anything else that has a position. Ends in the same spot leave no direction between
        // them and fall back to `+Z`, which `rotation_from_z` already does for a zero vector --
        // a state you pass through while dragging one of them, not an error to raise mid-frame.
        // **Where the curve starts, facing the way it sets off.** The tangent, not the chord --
        // the two agree for a straight line and part company for anything else, and a plane meant
        // to stand square across a pipe has to be square across it *where it stands*. Read as a
        // one-sided difference at the start of the domain, which is what a tangent is and is all
        // `Curve` needs to be asked for.
        fieldform_scene::Frame::Tangent { curve } => {
            let c = document
                .curves
                .get(curve)
                .ok_or(CompileError::MissingCurve(*curve))?;
            let (lo, hi) = c.domain();
            let at = c.evaluate(lo);
            // A step small against the curve and large against the arithmetic. The span is the
            // scale to measure it in, since a knot vector is not in millimetres.
            let step = (hi - lo) * 1e-4;
            let ahead = c.evaluate((lo + step).min(hi));
            Ok((at.coords, rotation_from_z(ahead - at)))
        }
        fieldform_scene::Frame::Between { from, to } => {
            let at = |node: &NodeId| {
                resolve_place_at(
                    document,
                    graph,
                    &fieldform_scene::Place::From { node: *node },
                    resolve,
                    depth + 1,
                )
            };
            let (start, end) = (at(from)?, at(to)?);
            Ok((start.coords, rotation_from_z(end - start)))
        }
    }
}

/// How far a box reaches from an axis, which is what a twist stretches material by.
///
/// **The true distance from the line, over the eight corners.** A twist turns a point by
/// `rate . d` along the axis, so two points at the same height and radius `r` are carried apart
/// by `r . rate` — and `r` is the distance from the *axis line*, not from a plane or an origin.
///
/// This used to be `hypot` of the box's reach along the two axes the chosen one was not, which
/// is only the same answer while the axis is one of X, Y or Z. A frame points anywhere, so the
/// projection has to be done properly: subtract the component along the axis and measure what
/// is left.
///
/// Both compilers call this. They divide by what it returns, so a difference between them puts
/// the surface in one place and scales the field around it differently — the kind of drift only
/// a residual against the other would catch.
pub fn reach_across_axis(
    (lo, hi): (Point3<f64>, Point3<f64>),
    origin: Vector3<f64>,
    rotation: Rotation3<f64>,
) -> f64 {
    let aim = rotation * Vector3::z();
    let mut worst: f64 = 0.0;
    for c in 0..8 {
        let corner = Point3::new(
            if c & 1 == 0 { lo.x } else { hi.x },
            if c & 2 == 0 { lo.y } else { hi.y },
            if c & 4 == 0 { lo.z } else { hi.z },
        );
        let from = corner.coords - origin;
        worst = worst.max((from - aim * from.dot(&aim)).norm());
    }
    worst
}

/// Below what length an axis is treated as no axis at all, in millimetres.
///
/// **Named because two sides read it.** `rotation_from_z` uses it here; `aimed` in the shader's
/// prelude has to make the same decision about the same number, and a shader that kept a
/// direction the field had already given up on would light a surface the field does not have.
/// `the_shader_gives_up_on_an_axis_at_the_same_length_the_field_does` holds the two spellings
/// together.
///
/// The comparison is `<=`, which is nalgebra's: `try_normalize` refuses at `n <= min_norm`.
pub const AXIS_FLOOR: f64 = 1e-9;

/// Which side of its own plane a joined mirror's body lies on, when it lies wholly on one.
///
/// `+1` for the plane's positive side, `-1` for the negative, `None` when the body straddles the
/// plane or cannot be bounded.
///
/// **Both compilers ask this, so both write the same composition.** A joined mirror means
/// `min(f(p), f(mirror(p)))`, which is the body written out twice — two spine walks instead of
/// one for a swept body, and the driver's compile time is worse than cubic in how many it has to
/// unroll. When the body never crosses the plane, one reading does: reflect the query onto the
/// body's side and ask there.
///
/// It is not an approximation. For any body point `b` and any query `p` on the body's side,
/// `|p − mirror(b)|² − |p − b|² = 4 (p·n − c)(b·n − c) ≥ 0`, because both factors carry the same
/// sign — so the query's own half is always the nearer one and the `min` never had a choice.
/// That argument is about true distances, and a field here may understate one; measured over a
/// grid on five bodies, two of them understating, the two forms agree to the bit, and the fold
/// never once claimed more clear room than the pair. `mirror_fold_probe` prints the table.
///
/// Decided from the extent, which is a bound on the body rather than the body: a body the extent
/// says straddles the plane may not really, and then this only misses the saving. The error that
/// would matter — saying "one side" of a body that crosses — cannot come from a bound that
/// encloses it.
pub fn mirror_one_side(document: &Document, graph: &Graph, id: NodeId) -> Option<f64> {
    // The same switch `FIELDFORM_NO_SPINE_INDEX` is, and for the same reason: the two
    // compositions are supposed to be the same number, so the way to show it is to build both
    // and compare, and the way to price the saving is to build both and time them.
    if std::env::var_os("FIELDFORM_NO_MIRROR_FOLD").is_some() {
        return None;
    }
    let (least, most) = mirror_bound_across_plane(document, graph, id)?;
    // Touching the plane is still one side: the reflection touches it in the same place.
    if least >= 0.0 {
        Some(1.0)
    } else if most <= 0.0 {
        Some(-1.0)
    } else {
        None
    }
}

/// Whether a joined mirror's body reaches its own plane, so the union keeps a surface buried in
/// the solid.
///
/// Backlog item 113's option 1, and the fact the normalization decision needs. A joined mirror is
/// `min(f(p), f(mirror(p)))` — a union — and a union of operands that touch keeps their shared
/// face inside it: the sign is right everywhere, which is why the mirror alone renders clean, but
/// the magnitude reads nought where the solid is deep. A blend reads magnitudes, so it takes that
/// buried face for a surface and grooves along it. The field is still a valid lower bound, so
/// nothing is illegal; what is wrong is *claiming* an exact distance, which is what
/// [`crate::carries_true_distance`] and `normalization_of` used to do here, because a mirror
/// passed its body's answer straight through as a rigid motion. The reflection is one; the join
/// is not.
///
/// **Touching counts, and a proved gap is the only thing that does not.** `true` when the body's
/// bound meets or crosses the plane, and `true` for a body that cannot be bounded or a plane that
/// cannot be read, since neither proves a gap. Only `least > 0` or `most < 0` — the whole bound
/// strictly to one side — answers `false`.
///
/// **Decided from the extent, so it errs the safe way.** The extent encloses the body, so a body
/// the bound says reaches the plane may not really, and then this costs an `≈ approximate` mark
/// the fillet did not need. The error that would matter is the other one — calling a mirror exact
/// whose body does bury a face — and a bound that encloses the body cannot produce it. That is the
/// opposite direction from [`mirror_one_side`], which wants to prove a body is *not* near the
/// plane and so gives up the saving on the same doubt.
///
/// **`FIELDFORM_NO_MIRROR_FOLD` is deliberately not read here.** That switch prices the fold
/// against writing the body out twice, and both compositions are the same number; this is about
/// what the field claims of that number, which no measuring switch may turn off.
///
/// `false` for a mirror that does not join: there is no union, so there is nothing to bury.
pub fn mirror_reaches_its_plane(document: &Document, graph: &Graph, id: NodeId) -> bool {
    if !matches!(graph.node(id), Some(fieldform_scene::Node::Mirror { join: true, .. })) {
        return false;
    }
    match mirror_bound_across_plane(document, graph, id) {
        Some((least, most)) => !(least > 0.0 || most < 0.0),
        // No bound, or no readable plane: a gap is what has to be proved, and nothing here
        // proves one.
        None => true,
    }
}

/// How far a joined mirror's body bound reaches either side of its plane, as signed distances
/// along the normal: the least and the most over the bound's eight corners.
///
/// One reading for [`mirror_one_side`] and [`mirror_reaches_its_plane`], which ask opposite
/// questions of the same two numbers and must not answer from two sweeps that could drift.
/// `None` for a mirror that does not join, a body with no box — `Empty` for a mirror with nothing
/// in it, which gives the fold up by the route an unbounded body already takes — or a plane that
/// does not resolve.
fn mirror_bound_across_plane(
    document: &Document,
    graph: &Graph,
    id: NodeId,
) -> Option<(f64, f64)> {
    let Some(fieldform_scene::Node::Mirror { input, frame, join: true }) = graph.node(id) else {
        return None;
    };
    let (lo, hi) = operand_extent(document, graph, input).as_box()?;
    let numbers = |r: &fieldform_scene::ParamRef| document.resolve(r);
    let (origin, rotation) = resolve_frame(document, graph, frame, &numbers).ok()?;
    let normal = rotation * Vector3::z();
    let at = Point3::from(origin);
    let mut least = f64::INFINITY;
    let mut most = f64::NEG_INFINITY;
    for i in 0..8 {
        let corner = Point3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let s = (corner - at).dot(&normal);
        least = least.min(s);
        most = most.max(s);
    }
    Some((least, most))
}

/// The rotation that carries `+Z` onto `axis`, by the shortest turn.
///
/// **One rotation of many, chosen because it is the one nobody has to think about.** An axis
/// pins down two of a rotation's three degrees of freedom; the roll about that axis is free, and
/// the shortest turn is the choice that introduces none. For a cylindrical or spherical lattice
/// the roll is invisible anyway — the pattern is symmetric about its own axis — so this is not a
/// compromise there, only a decision that has to be made somewhere.
///
/// A degenerate axis means the default: a zero vector cannot point anywhere, and an axis exactly
/// opposite `+Z` has a whole circle of shortest turns, so one is picked rather than left to a
/// cross product that vanishes.
pub fn rotation_from_z(axis: Vector3<f64>) -> Rotation3<f64> {
    let Some(a) = axis.try_normalize(AXIS_FLOOR) else {
        return Rotation3::identity();
    };
    let z = Vector3::z();
    let dot = a.dot(&z);
    if dot > 1.0 - 1e-12 {
        return Rotation3::identity();
    }
    if dot < -1.0 + 1e-12 {
        // Half a turn about any perpendicular. X is as good as another and is the one that can
        // be written down.
        return Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI);
    }
    Rotation3::rotation_between(&z, &a).unwrap_or_else(Rotation3::identity)
}

/// The node's Euler angles, in degrees, as a rotation.
///
/// Intrinsic X-then-Y-then-Z, i.e. `Rz * Ry * Rx` applied to a column vector — nalgebra's own
/// roll/pitch/yaw convention, taken rather than invented so the matrix here and any rotation
/// built elsewhere in the workspace agree.
pub(crate) fn euler_degrees_to_rotation(degrees: Vector3<f64>) -> Rotation3<f64> {
    Rotation3::from_euler_angles(
        degrees.x.to_radians(),
        degrees.y.to_radians(),
        degrees.z.to_radians(),
    )
}

/// Maps a world point into the space the transformed field is defined in.
///
/// The inverse of "rotate, scale, then translate", applied in reverse: undo the translation,
/// undo the rotation, undo the scale.
pub(crate) fn to_local(
    p: Point3<f64>,
    rotation: &Rotation3<f64>,
    translation: &Vector3<f64>,
    scale: f64,
) -> Point3<f64> {
    Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)) / scale)
}
