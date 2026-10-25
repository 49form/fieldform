// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Making, joining, deleting and moving curves and their points, as the commands that do it.

use super::*;

/// A single-span Bézier of the given degree, spanning `span` along X.
///
/// One span is what makes it a Bézier rather than a general B-spline: every knot sits at one
/// end or the other, so the curve is a single polynomial piece and its control points are the
/// whole of what shapes it. A degree-`n` one has exactly `n + 1` of them, and the knot vector
/// is `n + 1` zeros followed by `n + 1` ones — which is the `control_points + degree + 1`
/// length [`fieldform_scene::Curve::new`] requires, arrived at rather than padded to.
///
/// The starting shape is a half-sine: flat at both ends, highest in the middle. A curve that
/// looks like its degree from the moment it exists, where evenly spaced collinear points would
/// draw a straight line and leave degree 3 and degree 9 indistinguishable until something had
/// been dragged. It degenerates correctly too — at degree 1 the sine is zero at both of its two
/// points, so a degree-1 curve and a line are the same thing rather than quietly different.
///
/// **The rise is in Y, and it used to be in Z.** A curve is built in the drawing plane's own
/// coordinates and lifted into the world, so anything with a local Z is a curve arching *out of*
/// the plane it was drawn on — which is precisely what a plane is for preventing. Flat in local
/// XY is the only shape that survives the lift, and with no plane set it means a new curve lies
/// on the ground rather than standing up, which is also the more useful answer: it is a profile
/// until somebody makes it something else.
pub(crate) fn single_span_bezier(
    id: CurveId,
    degree: usize,
    span: f64,
) -> Result<fieldform_scene::Curve, fieldform_scene::CurveError> {
    let rise = span * 0.4;
    let points: Vec<Point3<f64>> = (0..=degree)
        .map(|i| {
            let t = i as f64 / degree as f64;
            Point3::new(span * t, rise * (std::f64::consts::PI * t).sin(), 0.0)
        })
        .collect();
    single_span(id, degree, points)
}

/// A single-span Bézier of `degree` from `a` to `b`, straight to begin with.
///
/// What two clicks make. The `degree − 1` interior points sit evenly along the chord, so the
/// curve starts as a straight run of the right degree and is dragged into shape — deliberately
/// not the place-every-control-point gesture of other tools, since a degree-9 curve is ten precise clicks
/// and the shape from ten hurried clicks is worse than the shape from two plus dragging
/// (`sketcher-spec.md` §7). At degree 1 the two points are the whole curve, and it is a line.
pub(crate) fn bezier_between(
    id: CurveId,
    degree: usize,
    a: Point3<f64>,
    b: Point3<f64>,
) -> Result<fieldform_scene::Curve, fieldform_scene::CurveError> {
    let points: Vec<Point3<f64>> = (0..=degree)
        .map(|i| a + (b - a) * (i as f64 / degree as f64))
        .collect();
    single_span(id, degree, points)
}

/// The one place a single-span curve is put together, whichever shape its points were given.
///
/// The knot vector is `degree + 1` zeros followed by `degree + 1` ones — the
/// `control_points + degree + 1` length [`fieldform_scene::Curve::new`] requires, arrived at
/// rather than padded to — and the name is decided here, so a placed line and a built one are
/// called the same thing.
pub(crate) fn single_span(
    id: CurveId,
    degree: usize,
    points: Vec<Point3<f64>>,
) -> Result<fieldform_scene::Curve, fieldform_scene::CurveError> {
    let knots: Vec<f64> = std::iter::repeat_n(0.0, degree + 1)
        .chain(std::iter::repeat_n(1.0, degree + 1))
        .collect();
    let weights = vec![1.0; points.len()];
    fieldform_scene::Curve::new(
        id,
        // A degree-1 curve is a line, so it is called one. Naming it "Curve 1" would be
        // accurate and would leave every line in the document called something nobody says.
        if degree == 1 { "Line".to_string() } else { format!("Curve {degree}") },
        degree,
        points,
        weights,
        knots,
        false,
    )
}

/// Where each point block of `part` is, resolved the way the field resolves it.
///
/// A point whose coordinate fails to evaluate — an expression that divides by nought, a
/// reference to a parameter that is gone — is left out rather than read as the origin, so the
/// ends sitting on it stay where they are, which is the safer of the two wrong answers.
fn point_positions(doc: &Document, part: PartId) -> std::collections::HashMap<NodeId, Point3<f64>> {
    let mut at = std::collections::HashMap::new();
    let Some(graph) = doc.parts.get(&part).map(|p| &p.graph) else { return at };
    let numbers = |r: &ParamRef| r.resolve(&|id| doc.parameter_value(id));
    for (id, node) in graph.nodes() {
        if let Node::Point { at: position } = node
            && let Ok(p) = fieldform_field::resolve_place(
                doc,
                graph,
                &fieldform_scene::Place::At(*position),
                &numbers,
            )
        {
            at.insert(id, p);
        }
    }
    at
}

/// Writes the position of every point block into the curve ends sitting on it.
///
/// **The single derivation**, as a function of the document rather than of the application,
/// so the same rule runs in a test and in a drag. Returns whether anything moved.
pub(crate) fn settle_curves_on_points(doc: &mut Document, part: PartId) -> bool {
    let at = point_positions(doc, part);
    let lookup = |id: NodeId| at.get(&id).copied();
    let mut moved = false;
    for curve in doc.curves.values_mut() {
        moved |= curve.apply_places(&lookup);
    }
    moved
}

/// The rows of `these` that are not rows of `those`, by membership rather than order.
pub(crate) fn rows_not_in(these: &[Vec<CurveId>], those: &[Vec<CurveId>]) -> Vec<usize> {
    let sorted = |row: &[CurveId]| {
        let mut r = row.to_vec();
        r.sort();
        r
    };
    let those: Vec<Vec<CurveId>> = those.iter().map(|r| sorted(r)).collect();
    these
        .iter()
        .enumerate()
        .filter(|(_, row)| !those.contains(&sorted(row)))
        .map(|(i, _)| i)
        .collect()
}

/// The loops an edit just closed: rows of `current` that were not rows before.
///
/// **No previous partition means a document just opened**, whose loops closed before it did.
/// Those are found and not announced, and not joined either (`profile-group-spec.md` §4.2,
/// "not on open"): mutating a document because it was opened is a change nobody asked for.
pub(crate) fn newly_closed_rows(previous: Option<&[Vec<CurveId>]>, current: &[Vec<CurveId>]) -> Vec<usize> {
    previous.map(|p| rows_not_in(current, p)).unwrap_or_default()
}

/// A point block's coordinates, when every one of them is a typed number.
pub(crate) fn point_literals(graph: &Graph, node: NodeId) -> Option<Point3<f64>> {
    let Some(Node::Point { at }) = graph.node(node) else { return None };
    let mut out = [0.0; 3];
    for (slot, r) in out.iter_mut().zip(at) {
        let ParamRef::Literal(v) = r else { return None };
        *slot = *v;
    }
    Some(Point3::new(out[0], out[1], out[2]))
}

/// Moves a point block by writing its literals.
///
/// `false`, with nothing written, when a coordinate is driven by a parameter: a drag must not
/// overwrite a number somebody typed an expression for, and the variable is where that move
/// belongs.
pub(crate) fn move_point_node(graph: &mut Graph, node: NodeId, to: Point3<f64>) -> bool {
    if point_literals(graph, node).is_none() {
        return false;
    }
    let Some(Node::Point { at }) = graph.node_mut(node) else { return false };
    *at = [ParamRef::Literal(to.x), ParamRef::Literal(to.y), ParamRef::Literal(to.z)];
    true
}

/// Which of a curve's control points sit on a point block, and which are free.
///
/// A drag routes the first kind to the block and the second to the curve
/// (`profile-group-spec.md` §4.3): the point is where the end is, so moving either moves both,
/// and moving the control point alone would only be snapped back by
/// [`settle_curves_on_points`] the moment anything rebuilt.
pub(crate) fn drag_targets(curve: &fieldform_scene::Curve, indices: &[usize]) -> (Vec<(usize, NodeId)>, Vec<usize>) {
    let mut bound = Vec::new();
    let mut free = Vec::new();
    for &i in indices {
        match curve.placed_on(i) {
            Some(node) => bound.push((i, node)),
            None => free.push(i),
        }
    }
    (bound, free)
}

/// The commands that hold a loop together: a point block at every meeting of two ends that
/// does not have one, with both ends sitting on it. One entry, however many joints.
///
/// An end already on a point keeps it, and the other end joins that one rather than a new
/// point being made. Two ends already on different points are left as they are: merging
/// points is not this command's to decide. The count is how many meetings gained a joint, so
/// a loop already held costs nothing and says so.
///
/// The points are made unnamed. A name is what makes a block an operand offered to every
/// other block (`unique_variable_name`), and thirty joints of a thirty-curve loop would be
/// thirty entries in every list; `profile-group-spec.md` §6 says promotion is the person's
/// gesture, and the row is where the points are found until then.
pub(crate) fn join_loop_commands(
    doc: &Document,
    part: PartId,
    members: &[CurveId],
) -> (Vec<Box<dyn fieldform_ops::Command>>, usize, Vec<String>) {
    let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
    let Some(before) = doc.parts.get(&part).map(|p| p.graph.clone()) else {
        return (commands, 0, Vec::new());
    };
    let mut graph = before.clone();
    let mut curves: std::collections::HashMap<CurveId, fieldform_scene::Curve> = members
        .iter()
        .filter_map(|id| doc.curve(*id).map(|c| (*id, c.clone())))
        .collect();
    let mut changed: Vec<CurveId> = Vec::new();
    let mut added = false;
    let mut joints = 0;
    // The relations that kept a corner loose, so the caller can say which rather than leaving a loop
    // quietly open.
    let mut loose: Vec<String> = Vec::new();
    for meeting in fieldform_field::loop_meetings(doc, members) {
        let [a, b] = meeting.ends;
        // **A point a relation derives is that relation's to move, not a block's (§6), and the join
        // was the one path not asking.** `owners_of` is already what the `set` word and the curve
        // panel's bind consult before putting a point on a block; this function consulted neither, so
        // an ordinary loop close would bind an end a mirror or a concentric claims. Nothing
        // oscillated — `settle_curves_on_points` settles then re-derives, so the relation is the last
        // write every pass — but `Curve::on_point` became a lie: `placed_on` said the end was the
        // block's while `control_points` held the relation's answer, and the *other* end on that same
        // joint did follow the block. So the corner tore open, as wide as the drag, and the row
        // dissolved on the next refresh saying only "opened".
        //
        // **The whole meeting is skipped, not the claimed end.** Binding only the free end would make
        // a half-joint — a point block holding one curve — which is worse than a corner left loose:
        // it looks joined and behaves as though it is not.
        let claimed = [a, b].into_iter().find_map(|(curve, index)| {
            fieldform_scene::relation::owners_of(doc, curve, index, None)
                .first()
                .and_then(|owner| doc.relations.get(owner))
                .map(|record| record.relation.describe(doc))
        });
        if let Some(what) = claimed {
            loose.push(what);
            continue;
        }
        let on_a = curves.get(&a.0).and_then(|c| c.placed_on(a.1));
        let on_b = curves.get(&b.0).and_then(|c| c.placed_on(b.1));
        let node = match (on_a, on_b) {
            (Some(_), Some(_)) => continue,
            (Some(p), None) | (None, Some(p)) => p,
            (None, None) => {
                added = true;
                graph.add(Node::Point {
                    at: [
                        ParamRef::Literal(meeting.at.x),
                        ParamRef::Literal(meeting.at.y),
                        ParamRef::Literal(meeting.at.z),
                    ],
                })
            }
        };
        for (end, on) in [(a, on_a), (b, on_b)] {
            if on.is_some() {
                continue;
            }
            if let Some(c) = curves.get_mut(&end.0) {
                c.place_on(end.1, Some(node));
                if !changed.contains(&end.0) {
                    changed.push(end.0);
                }
            }
        }
        joints += 1;
    }
    if added {
        commands.push(Box::new(SetPartGraph::new(part, before, graph)));
    }
    for id in changed {
        if let (Some(was), Some(now)) = (doc.curve(id).cloned(), curves.remove(&id)) {
            commands.push(Box::new(SetCurve::new(id, Some(was), Some(now))));
        }
    }
    loose.sort();
    loose.dedup();
    (commands, joints, loose)
}

/// Whether moving this curve changes a field — **following relations, which is the part that was
/// missing**.
///
/// This gate decides whether a curve gesture retraces the view, during the drag and on release. It
/// asked two questions: does a block read this curve, and does it sit on a point a block reads. Both
/// are about the curve in the hand, and a relation's *slave* is neither — so dragging a loose master
/// whose slave is an extrude's profile left the solid frozen while both overlay lines moved, and it
/// stayed frozen. Orbiting, zooming, selecting and waiting all redraw from the same stale shader and
/// the same stale `pick_field`, so **clicking the solid picked the old shape too**. It cleared only on
/// the next unrelated edit, an undo, or reopening the file. Nothing was lost — the document holds the
/// re-derived points throughout — but a picture that lies for an unbounded time is worse than one that
/// lags.
///
/// **Forward through the relations, with a visited set, because relations chain.** A master's move
/// re-derives its slave, whose own move re-derives *its* slave; any of them feeding a block is enough.
/// `Relation::masters` is what answers "is this curve a master here", and it already counts a mirror's
/// axis, which is a master in the sense that matters: moving it moves the reflection.
///
/// A broken relation is skipped, because `relation::rederive` skips it too — the gate has to agree
/// with what will actually move, not with what was once declared.
///
/// Free of the app so the rule can be run on a document without a window or a GPU device, which is
/// the arrangement `operand_choices_in` and `blocks_reading` already use.
pub(crate) fn curve_shapes_a_body_in(doc: &Document, graph: &Graph, curve: CurveId) -> bool {
    let mut walk = vec![curve];
    let mut seen: Vec<CurveId> = Vec::new();
    while let Some(id) = walk.pop() {
        if seen.contains(&id) {
            continue;
        }
        seen.push(id);
        if graph.nodes().any(|(_, n)| n.curves().contains(&id)) {
            return true;
        }
        let on_a_read_point =
            doc.curve(id).is_some_and(|c| c.sits_on().any(|p| graph.is_consumed(p)));
        if on_a_read_point {
            return true;
        }
        for record in doc.relations.values() {
            if record.broken.is_none() && record.relation.masters().contains(&id) {
                walk.push(record.relation.slave());
            }
        }
    }
    false
}

/// Deleting a whole row: its relations, its curves, **and the joint points holding them**.
///
/// **The cleanup was missing, not misplaced.** `delete_profile` went straight to
/// `delete_curves_commands`, which takes the member curves and their relations and knows nothing
/// about points — so deleting a four-line square left four rows reading "Point" and four dots at the
/// corners of a square that was gone, saved and reopened with them, offered them in every blend's
/// operand list, and made each one its own delete and its own undo entry. Written down as known at
/// the profile-groups merge and never run, which is how it survived.
///
/// **Not `delete_curves_commands` plus `ungroup_commands`, which was the obvious shape and is wrong.**
/// `ungroup` keeps its curves, so it emits a `SetCurve` per member to place that member's ends off the
/// point — and `ungroup` reads the document as it was *before* the deletes, so those commands run
/// after the curves are gone and **put them back**. A test asserting the curves were deleted is what
/// caught it. What is shared with `ungroup` is the safety test, and it is restated here with the one
/// difference that matters: a curve being deleted does not count as sitting on a point.
///
/// **The graph edit goes last, so undo puts it back first.** A `Batch` undoes backwards, and the
/// points have to exist again before the curves whose ends name them are restored.
pub(crate) fn delete_row_commands(
    doc: &Document,
    part: PartId,
    members: &[CurveId],
) -> Vec<Box<dyn fieldform_ops::Command>> {
    let mut commands = delete_curves_commands(doc, members);
    let Some(before) = doc.parts.get(&part).map(|p| p.graph.clone()) else {
        return commands;
    };
    let mut graph = before.clone();
    let mut removed = false;
    // Every point the members were sitting on, each considered once.
    let mut held: Vec<NodeId> = Vec::new();
    for id in members {
        if let Some(c) = doc.curve(*id) {
            for node in c.sits_on() {
                if !held.contains(&node) {
                    held.push(node);
                }
            }
        }
    }
    for node in held {
        // The same two-part test `ungroup_commands` applies, with one difference that is the whole
        // reason this is not that function: a curve being deleted does not count as sitting on the
        // point. `ungroup` keeps its curves and so must ask about all of them.
        let read = graph.is_consumed(node);
        let sat_on = doc
            .curves
            .iter()
            .any(|(id, c)| !members.contains(id) && c.sits_on().any(|n| n == node));
        if !read && !sat_on && graph.remove(node) {
            removed = true;
        }
    }
    if removed {
        commands.push(Box::new(SetPartGraph::new(part, before, graph)));
    }
    commands
}

/// The curves a block's sections are walked through, found the way the compile finds them: from
/// the curve each section names, leaving out the one the block is aimed by
/// ([`Node::aimed_by`]).
///
/// **Why a delete asks this as well as `Node::curves`.** An extrude, revolve or sweep names one
/// curve of its section and is built on the loop walked from it, so taking out any other curve of
/// that loop changes the block as surely as taking out the one it names. A wall is walked by
/// `outline_chain` — the loop if one closes, the open run if not — so a wall on an open chain
/// reads the whole chain, which asking over `loop_from` could not see. A loft holds a section per
/// station and each is walked the same way; matched by name because `Node::outline` answers for
/// the three blocks holding one section, and a loft holds a list. A copy carries them for the same
/// reason ([`copied_body`]).
fn section_curves(doc: &Document, node: &Node) -> Vec<CurveId> {
    use fieldform_scene::Outline;
    let ignore = node.aimed_by();
    let sections: Vec<&Outline> = match node {
        Node::Loft { sections, .. } => sections.iter().collect(),
        other => other.outline().into_iter().collect(),
    };
    sections
        .into_iter()
        .flat_map(|section| match section {
            Outline::Closed { profile } => fieldform_field::loop_from(doc, *profile, ignore)
                .unwrap_or_else(|_| vec![*profile]),
            Outline::Wall { curve, .. } => fieldform_field::outline_chain(doc, *curve, ignore)
                .map(|chain| chain.curves)
                .unwrap_or_else(|_| vec![*curve]),
        })
        .collect()
}

/// Every block outside `part` that reads one of `curves`, with the part it is in and whether it
/// reads them only through its section — once each, however many of them it names.
///
/// **Every part, because a curve belongs to the document.** A part's graph names a curve by id and
/// the curves live beside the parts, so one curve can be what an extrude in one part and a pipe in
/// another are both built on. A reader is a node naming one in `Node::curves`, which is how
/// `block-ownership-spec.md` I2 counts a curve's readers, **or one whose section is walked through
/// one** ([`section_curves`]): I2 decides where a curve is listed, and a delete has to ask who it
/// would break as well. Only those two, and that is why the walk is each block's own rather than
/// one over the deleted curve's loop: a pipe along a neighbouring side, or a frame aimed by one,
/// reads that curve and nothing else, and a refusal asked over the loop would name it as a reader
/// of a curve it never reads.
fn readers_outside(
    doc: &Document,
    part: PartId,
    curves: &[CurveId],
) -> Vec<(PartId, NodeId, bool)> {
    let mut readers = Vec::new();
    for (p, other) in doc.parts.iter().filter(|(p, _)| **p != part) {
        for (node, n) in other.graph.nodes() {
            if n.curves().iter().any(|c| curves.contains(c)) {
                readers.push((*p, node, false));
            } else if section_curves(doc, n).iter().any(|c| curves.contains(c)) {
                readers.push((*p, node, true));
            }
        }
    }
    readers
}

/// The blocks in other parts that read a curve being deleted, for the line that refuses it.
pub(crate) struct ReadElsewhere {
    /// Each as "<block> in <part>", sorted and once each.
    pub(crate) readers: Vec<String>,
    /// Whether every one of them reads it only as one of the curves its section is walked
    /// through, naming another. Said differently, because "used by" is baffling when the block
    /// names some other curve.
    pub(crate) through_sections: bool,
    /// How many parts they are in.
    pub(crate) parts: usize,
}

impl ReadElsewhere {
    pub(crate) fn is_empty(&self) -> bool {
        self.readers.is_empty()
    }

    /// The end of the refusal, counted, so that readers in two parts are not "another part".
    pub(crate) fn built_on(&self) -> &'static str {
        if self.parts > 1 {
            "other parts are built on it, so it stays"
        } else {
            "another part is built on it, so it stays"
        }
    }
}

/// The blocks outside `part` that read one of `curves` ([`readers_outside`] says which those are),
/// named as "<block> in <part>": what a delete from `part` names when it refuses. `label` is handed
/// the reader's own part's graph, since the part on screen does not hold it.
///
/// **Refused, never followed, and that is the difference from a reader in `part`.** A delete can
/// take a block in the part on screen along with it, or open its loop into a wall, because that
/// block is in front of the person deleting. One in another part is in front of nobody: following
/// it would be an edit to a part nobody is looking at, and leaving it reading a curve that has gone
/// breaks that part out of sight, with nothing said where it broke. The deletes asked the part on
/// screen alone until 2026-10-11 (refactor plan §3.3, root cause 10, and §7, 2.6).
pub(crate) fn readers_in_other_parts(
    doc: &Document,
    part: PartId,
    curves: &[CurveId],
    label: impl Fn(&Graph, NodeId) -> String,
) -> ReadElsewhere {
    let found = readers_outside(doc, part, curves);
    let through_sections = found.iter().all(|(_, _, through)| *through);
    let mut parts: Vec<PartId> = found.iter().map(|(p, _, _)| *p).collect();
    parts.sort();
    parts.dedup();
    let mut readers: Vec<String> = found
        .into_iter()
        .filter_map(|(p, node, _)| {
            let other = doc.parts.get(&p)?;
            Some(format!("{} in {}", label(&other.graph, node), other.name))
        })
        .collect();
    readers.sort();
    readers.dedup();
    ReadElsewhere { readers, through_sections, parts: parts.len() }
}

/// What copying `root` out of `graph` puts on the clipboard: the block, everything beneath it, the
/// cages those blocks name and every curve they read.
///
/// The whole subtree for a body, because a boolean without the two solids it joins is not a copy
/// of anything — and the cages and curves those blocks name, so what comes back on paste stands on
/// its own.
///
/// **Every curve a block reads, not only the one under a pipe.** This took `Node::Pipe`'s curve and
/// no other, so a pasted extrude, revolve, sweep or loft named the original's curves: the copy
/// shared its profile, and a curve two blocks read has no owner, so the profile left the original's
/// row and stood on its own (refactor plan §3.3, root cause 5, and §7, 2.7). So it asks
/// `Node::curves`, which knows every way a block names a curve (D9), and the curves each section is
/// walked through ([`section_curves`]), since an extrude names one curve of its loop and is built on
/// all of them. Once each, however many blocks read it: two pipes along one line paste as two pipes
/// along one copy of it, as the originals share one line.
pub(crate) fn copied_body(doc: &Document, graph: &Graph, root: NodeId) -> Clipboard {
    // Depth first, pushing each node after everything it consumes, so the paste can insert them in
    // order and never name an input that is not there yet.
    let mut nodes: Vec<(NodeId, Node)> = Vec::new();
    let mut stack = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = graph.node(id) else { continue };
        stack.extend(node.inputs());
        nodes.push((id, node.clone()));
    }
    nodes.reverse();
    // The body as it is, red or not: a copy of an open cage is an open cage.
    let cages: Vec<(CageId, fieldform_cage::CageBody)> = nodes
        .iter()
        .filter_map(|(_, n)| match n {
            Node::Cage(c) => Some(*c),
            _ => None,
        })
        .filter_map(|c| doc.cage_body(c).map(|cage| (c, cage.clone())))
        .collect();
    let mut read: Vec<CurveId> = Vec::new();
    for (_, node) in &nodes {
        for id in node.curves().into_iter().chain(section_curves(doc, node)) {
            if !read.contains(&id) {
                read.push(id);
            }
        }
    }
    let curves: Vec<(CurveId, fieldform_scene::Curve)> = read
        .into_iter()
        .filter_map(|c| doc.curve(c).map(|curve| (c, curve.clone())))
        .collect();
    Clipboard::Body { nodes, root, cages, curves }
}

/// Pastes a body [`copied_body`] put aside into `part`, as **one** entry on `stack`, and answers the
/// copy of its root. Fresh ids for every node, cage and curve, so the copy shares nothing with what
/// it came from.
///
/// **One batch: a `SetCurve` for each copied curve, then the graph edit.** A paste was three
/// entries — the graph set to the pasted one, set back and set again — and the curves it copied
/// were written into the document beside all three, so undoing it took the pasted blocks out and
/// left their curves behind, read by nothing (refactor plan §3.3, root cause 5). In one batch they
/// go and come back together, and the graph edit runs last so that undo takes the blocks out
/// before the curves they name.
///
/// **A curve that will not copy refuses the paste**, where it used to be skipped. Skipped, the
/// pasted block went on naming the original's curve, which is the sharing this exists to end.
///
/// **Every copy lies exactly on the curve it was copied from**, since a paste lands in place, and
/// an extrude's loop is walked over every open curve in the document. The walk takes, of curves
/// lying on each other, the one of its seed's own generation (`passed_over` in
/// `fieldform-field`'s `profile.rs`), so the original stays on its own curves and the paste is
/// built on its copies. A copy carries the curve and nothing keyed by its id: an axis stood on a
/// surface pastes as a plain line where the axis was, and a loop whose row was dismissed pastes
/// as one whose row is not, as a curve copied on its own always has.
///
/// **A cage is still written straight into the document**, as adding a body writes one: there is
/// no command that adds a cage, so an undo leaves the copy there with no block naming it. That gap
/// is every cage-making path's, not paste's, and it is the state deleting the pasted body leaves
/// too, since nothing takes a cage out of the document once it is in. Written after the copies
/// that can refuse, so a refused paste leaves no cage behind.
///
/// Free of the `App` so a test can hold the rule, as `record_graph_edit` is.
pub(crate) fn paste_body(
    stack: &mut CommandStack,
    doc: &mut Document,
    part: PartId,
    nodes: &[(NodeId, Node)],
    root: NodeId,
    cages: &[(CageId, fieldform_cage::CageBody)],
    curves: &[(CurveId, fieldform_scene::Curve)],
) -> Result<NodeId, String> {
    let before = doc.parts.get(&part).ok_or("the part to paste into is gone")?.graph.clone();
    let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
    let mut curve_map = std::collections::HashMap::new();
    for (old, curve) in curves {
        let id = CurveId::new();
        let fresh = curve
            .copy_as(id, format!("{} copy", curve.name))
            .map_err(|e| format!("Could not copy {}, so nothing was pasted: {e}", curve.name))?;
        commands.push(Box::new(SetCurve::insert(fresh)));
        curve_map.insert(*old, id);
    }
    let mut cage_map = std::collections::HashMap::new();
    for (old, cage) in cages {
        cage_map.insert(*old, doc.add_cage_body(cage.clone()));
    }
    let mut graph = before.clone();
    let mut node_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    for (old, node) in nodes {
        let id = graph.add(node.remapped(&node_map, &cage_map, &curve_map));
        node_map.insert(*old, id);
    }
    let pasted = *node_map.get(&root).ok_or("the copy has lost the block it was taken from")?;
    commands.push(Box::new(SetPartGraph::new(part, before, graph)));
    stack
        .execute(doc, Box::new(fieldform_ops::Batch::new("edit.paste", commands)))
        .map_err(|e| e.to_string())?;
    Ok(pasted)
}

/// What deleting one curve out of a loop a block is built on did, for the line that says so.
pub(crate) struct LoopMemberDeleted {
    /// The blocks that read the loop and now read the open chain, as walls.
    pub(crate) walled: Vec<NodeId>,
    /// The joints at the deleted curve's two ends that went with it.
    pub(crate) joints: usize,
}

/// Deleting one curve out of a loop that a block is built on: the curve and its relations go,
/// the joints at its two ends go with it, and every block that read the loop reads the open
/// chain that is left — as a wall.
///
/// **Asked for**: "when I extrude a profile and then change my mind and want to delete a curve
/// out of it and it becomes open, I should be able to do that — right now it won't let me; also
/// the points that are joints should auto delete." It was refused because a block built on a
/// loop cannot be built on an open chain. It can now, as a wall, so the block follows the
/// curves rather than holding them hostage: a closed profile becomes a wall of the last
/// thickness, and a wall stays one. Whichever block named the deleted curve is pointed at the
/// next curve round the loop, so its walk still starts somewhere.
///
/// **The two joints, and only those.** A joint holds two ends together; the ones at the deleted
/// curve's ends now hold one, which is a point with a curve sitting on it and nothing joined. The
/// neighbour's end is set loose where it is and the point goes, unless something else reads it.
/// The chain's other joints still join two curves and stay.
///
/// Free of the app, as `delete_row_commands` is, so the rule can be run on a document alone. The
/// graph edit goes last, so undo puts the points back before the curves whose ends name them.
/// `None` when a block in another part reads the curve — names it, or is built on the loop
/// through it — which is refused rather than followed ([`readers_in_other_parts`] says why).
pub(crate) fn delete_loop_member_commands(
    doc: &Document,
    part: PartId,
    id: CurveId,
    mates: &[CurveId],
    wall_thickness: f64,
) -> Option<(Vec<Box<dyn fieldform_ops::Command>>, Graph, LoopMemberDeleted)> {
    use fieldform_scene::{Outline, WallEnds, WallSide};
    // **Not while a block in another part reads this curve.** The walls below are made in `part`
    // alone, so one built on this loop elsewhere would be left reading an open chain as a closed
    // profile and stop evaluating out of sight. The app refuses first and names it
    // (`readers_in_other_parts`); asked here as well, so that nothing building these commands can
    // get round it. Asked of this curve rather than of every mate: a pipe elsewhere along another
    // side reads that side alone, and keeps it.
    if !readers_outside(doc, part, &[id]).is_empty() {
        return None;
    }
    let before = doc.parts.get(&part)?.graph.clone();
    let mut graph = before.clone();
    let gone = doc.curve(id)?.clone();
    let mut commands = delete_curves_commands(doc, &[id]);

    // The joints at its ends: free every other curve's end sitting on each, then remove it.
    let mut joints = 0;
    for point in gone.sits_on() {
        for (other_id, other) in &doc.curves {
            if *other_id == id {
                continue;
            }
            let n = other.control_points().len();
            let ends: Vec<usize> = [0, n.saturating_sub(1)]
                .into_iter()
                .filter(|&i| n >= 2 && other.placed_on(i) == Some(point))
                .collect();
            if ends.is_empty() {
                continue;
            }
            let mut loosened = other.clone();
            for i in ends {
                loosened.place_on(i, None);
            }
            commands.push(Box::new(SetCurve::new(*other_id, Some(other.clone()), Some(loosened))));
        }
        let read = graph.is_consumed(point);
        if !read && graph.remove(point) {
            joints += 1;
        }
    }

    // The blocks that read the loop: pointed past the deleted curve, and open now, so walls.
    let at = mates.iter().position(|m| *m == id);
    let next = at.map(|i| mates[(i + 1) % mates.len()]).filter(|m| *m != id);
    let mut walled = Vec::new();
    let readers: Vec<NodeId> = graph
        .nodes()
        .filter(|(_, n)| n.outline().is_some_and(|o| mates.contains(&o.curve())))
        .map(|(node, _)| node)
        .collect();
    for node in readers {
        let Some(outline) = graph.node_mut(node).and_then(Node::outline_mut) else { continue };
        if outline.curve() == id {
            *outline.curve_mut() = next?;
        }
        if let Outline::Closed { profile } = *outline {
            *outline = Outline::Wall {
                curve: profile,
                thickness: ParamRef::Literal(wall_thickness),
                side: WallSide::Centre,
                ends: WallEnds::Square,
            };
            walled.push(node);
        }
    }
    commands.push(Box::new(SetPartGraph::new(part, before, graph.clone())));
    Some((commands, graph, LoopMemberDeleted { walled, joints }))
}

/// The commands that take a loop apart: every member's ends off their points, and the points
/// nothing else needs removed. One entry, and nothing moves — an end taken off its point stays
/// where it is, which is what `place_on(None)` already means.
///
/// A point stays when a block still reads it or a curve outside the loop still sits on it: the
/// loop is letting go of it, not deleting somebody else's. The count is the points removed.
pub(crate) fn ungroup_commands(
    doc: &Document,
    part: PartId,
    members: &[CurveId],
) -> (Vec<Box<dyn fieldform_ops::Command>>, usize) {
    let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
    let Some(before) = doc.parts.get(&part).map(|p| p.graph.clone()) else {
        return (commands, 0);
    };
    let mut graph = before.clone();
    let mut curves: std::collections::HashMap<CurveId, fieldform_scene::Curve> = members
        .iter()
        .filter_map(|id| doc.curve(*id).map(|c| (*id, c.clone())))
        .collect();
    let mut changed: Vec<CurveId> = Vec::new();
    let mut freed: Vec<NodeId> = Vec::new();
    for id in members {
        let Some(c) = curves.get_mut(id) else { continue };
        let n = c.control_points().len();
        if n < 2 {
            continue;
        }
        for i in [0, n - 1] {
            if let Some(node) = c.placed_on(i) {
                c.place_on(i, None);
                if !freed.contains(&node) {
                    freed.push(node);
                }
                if !changed.contains(id) {
                    changed.push(*id);
                }
            }
        }
    }
    let mut removed = 0;
    for node in freed {
        let read = graph.is_consumed(node);
        let sat_on = doc.curves.iter().any(|(id, c)| {
            curves.get(id).unwrap_or(c).sits_on().any(|n| n == node)
        });
        if !read && !sat_on && graph.remove(node) {
            removed += 1;
        }
    }
    for id in changed {
        if let (Some(was), Some(now)) = (doc.curve(id).cloned(), curves.remove(&id)) {
            commands.push(Box::new(SetCurve::new(id, Some(was), Some(now))));
        }
    }
    if removed > 0 {
        commands.push(Box::new(SetPartGraph::new(part, before, graph)));
    }
    (commands, removed)
}

/// The relations naming any of `curves`, as records, so a delete can take them along.
fn relations_naming(
    doc: &Document,
    curves: &[CurveId],
) -> Vec<(fieldform_scene::RelationId, fieldform_scene::RelationRecord)> {
    doc.relations
        .iter()
        .filter(|(_, r)| curves.iter().any(|c| r.relation.names(*c)))
        .map(|(id, r)| (*id, r.clone()))
        .collect()
}

/// The commands that delete `curves` and every relation naming one of them, for one entry.
///
/// **Deleting a curve deletes every relation naming it, inside the same command**
/// (`profile-group-spec.md` §5.4). The relations go first, so that undo -- which runs the
/// batch backwards -- puts the curves back before the records that name them.
pub(crate) fn delete_curves_commands(doc: &Document, curves: &[CurveId]) -> Vec<Box<dyn fieldform_ops::Command>> {
    let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
    for (id, record) in relations_naming(doc, curves) {
        commands.push(Box::new(SetRelation::remove(id, record)));
    }
    for id in curves {
        if let Some(curve) = doc.curve(*id).cloned() {
            commands.push(Box::new(SetCurve::remove(curve)));
        }
    }
    commands
}

/// Adds `x` to `v`, or takes it out if it is there: what Shift does to a selection.
pub(crate) fn toggle_in<T: PartialEq>(v: &mut Vec<T>, x: T) {
    match v.iter().position(|y| *y == x) {
        Some(at) => {
            v.remove(at);
        }
        None => v.push(x),
    }
}

/// The handles inside a box on screen, in the order given.
pub(crate) fn points_in_box(
    handles: &[(CurveId, usize, egui::Pos2)],
    bx: egui::Rect,
) -> Vec<(CurveId, usize)> {
    handles.iter().filter(|(_, _, at)| bx.contains(*at)).map(|(c, i, _)| (*c, *i)).collect()
}

/// A point selection the way the app holds one: the curve that holds the handles and its
/// points, then everything picked on other curves. `prefer` keeps that place when any point is
/// on it; otherwise the first curve picked takes it. `None` when nothing is picked.
pub(crate) fn split_selection(
    picked: &[(CurveId, usize)],
    prefer: Option<CurveId>,
) -> Option<(CurveId, Vec<usize>, Vec<(CurveId, usize)>)> {
    let primary = prefer
        .filter(|p| picked.iter().any(|(c, _)| c == p))
        .or_else(|| picked.first().map(|(c, _)| *c))?;
    let (mut indices, mut others) = (Vec::new(), Vec::new());
    for &(c, i) in picked {
        if c == primary {
            if !indices.contains(&i) {
                indices.push(i);
            }
        } else if !others.contains(&(c, i)) {
            others.push((c, i));
        }
    }
    Some((primary, indices, others))
}

/// Picked points grouped by curve, in the order the curves first appear.
pub(crate) fn group_by_curve(picked: &[(CurveId, usize)]) -> Vec<(CurveId, Vec<usize>)> {
    let mut groups: Vec<(CurveId, Vec<usize>)> = Vec::new();
    for &(c, i) in picked {
        match groups.iter_mut().find(|(g, _)| *g == c) {
            Some((_, v)) => v.push(i),
            None => groups.push((c, vec![i])),
        }
    }
    groups
}

/// `x,y,z` as a point, for `set`.
fn parse_xyz(text: &str) -> Option<Point3<f64>> {
    let parts: Vec<f64> =
        text.split(',').map(|s| s.trim().parse::<f64>()).collect::<Result<_, _>>().ok()?;
    let &[x, y, z] = parts.as_slice() else { return None };
    Some(Point3::new(x, y, z))
}

/// What `set` was asked to move: each `<curve>.p<i>` or `<curve>.start` / `<curve>.end` key,
/// with the `x,y,z` it goes to. `p0` is the start, counting the way the curve runs. Refused,
/// with the reason, when a curve is not there, a point is past the curve's last, or a value is
/// not three numbers; keys without a `.` are not points and are left alone.
pub(crate) fn set_targets(
    doc: &Document,
    pairs: &[(String, String)],
    named: impl Fn(&str) -> Option<CurveId>,
) -> Result<Vec<(CurveId, usize, Point3<f64>)>, String> {
    let mut out = Vec::new();
    for (key, value) in pairs {
        let Some((name, which)) = key.rsplit_once('.') else { continue };
        let curve = named(name).ok_or_else(|| format!("no curve is called `{name}`"))?;
        let count = doc.curve(curve).map_or(0, |c| c.control_points().len());
        let last = count.saturating_sub(1);
        let index = match which {
            "start" => 0,
            "end" => last,
            p => p
                .strip_prefix('p')
                .and_then(|i| i.parse::<usize>().ok())
                .filter(|i| *i < count)
                .ok_or_else(|| {
                    format!("`{which}` is not a point of {name} — say p0 to p{last}, start or end")
                })?,
        };
        let at = parse_xyz(value)
            .ok_or_else(|| format!("`{key}={value}` — a position is three numbers, x,y,z"))?;
        out.push((curve, index, at));
    }
    Ok(out)
}

/// What re-aligning relations found broken on open comes to: the commands, what moved, and a line
/// saying what each relation was.
///
/// **A function of the document, so the clearing can be run without a window.** `realign` was the one
/// part of the relations work with no test of its own — the tests covered the check that marks a
/// relation broken and the re-derivation it runs, and not the step between: clear the mark, then
/// re-derive from the master's *current* state, as one batch whose undo puts the mark and the points
/// back. `None` when there is nothing to mend.
pub(crate) fn realign_commands(
    doc: &Document,
    broken: &[fieldform_scene::RelationId],
) -> Option<(Vec<Box<dyn fieldform_ops::Command>>, Vec<(CurveId, usize, Point3<f64>)>, Vec<String>)> {
    let mut after = doc.clone();
    let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
    let mut said = Vec::new();
    for id in broken {
        if let Some(record) = doc.relations.get(id).cloned() {
            said.push(record.relation.describe(doc));
            let mut mended = record.clone();
            mended.broken = None;
            after.relations.insert(*id, mended.clone());
            commands.push(Box::new(SetRelation::new(*id, Some(record), Some(mended))));
        }
    }
    if commands.is_empty() {
        return None;
    }
    let moved = fieldform_scene::relation::rederive(&mut after);
    commands.extend(rederived_commands(doc, &moved));
    Some((commands, moved, said))
}

/// The `SetControlPoints` that record what a re-derivation moved, one per curve, so the move
/// lands on the history beside the edit that caused it.
pub(crate) fn rederived_commands(
    before: &Document,
    moved: &[(CurveId, usize, Point3<f64>)],
) -> Vec<Box<dyn fieldform_ops::Command>> {
    let mut by_curve: std::collections::BTreeMap<CurveId, (Vec<usize>, Vec<Point3<f64>>, Vec<Point3<f64>>)> =
        std::collections::BTreeMap::new();
    for (curve, index, to) in moved {
        let Some(was) = before.curve(*curve).and_then(|c| c.control_points().get(*index).copied())
        else {
            continue;
        };
        let entry = by_curve.entry(*curve).or_default();
        entry.0.push(*index);
        entry.1.push(was);
        entry.2.push(*to);
    }
    by_curve
        .into_iter()
        .map(|(curve, (indices, was, now))| {
            Box::new(SetControlPoints::new(curve, indices, was, now)) as Box<dyn fieldform_ops::Command>
        })
        .collect()
}

/// The same curve moved rigidly, so its start lands at `at` and its shape is kept.
///
/// A copy, through [`fieldform_scene::Curve::copy_as`], so it sits on nothing: a pasted curve
/// is loose wherever it lands (`profile-group-spec.md` §4.4).
pub(crate) fn moved_to_start_at(
    curve: &fieldform_scene::Curve,
    id: CurveId,
    name: impl Into<String>,
    at: Point3<f64>,
) -> Result<fieldform_scene::Curve, fieldform_scene::CurveError> {
    let mut moved = curve.copy_as(id, name)?;
    let Some(start) = curve.control_points().first().copied() else { return Ok(moved) };
    let shift = at - start;
    for (i, p) in curve.control_points().iter().enumerate() {
        moved.set_control_point(i, p + shift);
    }
    Ok(moved)
}

/// The degrees a new curve can be made at.
///
/// The odd ones plus 2, which is the set a surfacing tool actually uses: 1 is a line, 2 holds
/// an exact conic, 3 is the working default nearly everything is built from, and 5 and 7 are
/// the higher-continuity curves used where a highlight has to run clean across a join. 4, 6 and
/// 8 are absent because even degrees are rarely reached for once 3 exists — nothing stops one
/// being added, but a picker offering ten choices where six are used is a worse picker.
///
/// **Degree 9 left on 2026-09-24** (`sketcher-spec.md` §15.4, `profile-group-spec.md` §7):
/// degree 7 already carries G3 at both ends, so 9 bought nothing a join could use. Only the
/// tools stopped offering it. A degree-9 curve in a saved document still opens and still
/// edits, because [`fieldform_scene::Curve::new`] accepts any degree and refusing saved work
/// would destroy it — `a_degree_nine_curve_saved_earlier_still_opens_and_edits` holds that.
pub(crate) const CURVE_DEGREES: [usize; 5] = [1, 2, 3, 5, 7];
