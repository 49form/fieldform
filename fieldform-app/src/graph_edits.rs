// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Recording an edit to a part's graph as one undo step, with everything that must travel with it.

use crate::*;

/// Whether two graphs differ only in their numbers.
///
/// The test that separates a drag from an act. Dragging a radius rewrites a parameter and
/// leaves every node and every edge exactly where it was; adding a body, deleting one, or
/// wrapping one in a placement does not. So this asks the structural question and ignores the
/// values, which is precisely the line between "still the same gesture" and "a new one".
///
/// **The curves a node reads are left out on purpose**, though they are edges to who owns what
/// (`block-ownership-spec.md` I2). A console line builds its block on defaults and then gives it
/// the keys it was told, and it is one undo step because those keys continue the line's gesture:
/// `revolve profile= axis=` points the frame at a curve after the revolve is made. Comparing curves
/// would split such a line in two, and its first entry would settle the default for good. A curve
/// re-pointed inside a gesture is owned as one edit from the gesture's start owns it, because
/// every later instant settles from there (`SetPartGraph`).
pub(crate) fn same_structure(a: &Graph, b: &Graph) -> bool {
    if a.nodes().count() != b.nodes().count() {
        return false;
    }
    // Marking a surface critical is a deliberate act, never an instant of a drag, so it takes
    // its own undo entry however soon after another edit it comes.
    if a.critical() != b.critical() {
        return false;
    }
    a.nodes().all(|(id, node)| match b.node(id) {
        // Edges, and the kind. A node whose *kind* changed is a different act even at the same
        // id — switching a boolean from subtract to union is not a drag.
        Some(other) => {
            node.inputs() == other.inputs()
                && std::mem::discriminant(node) == std::mem::discriminant(other)
        }
        None => false,
    })
}

/// Puts a graph edit on the history: a new entry, or the next instant of the gesture in progress.
///
/// The history half of `App::set_part_graph`, which says what a continuation is and why. Free of
/// the `App` so a test can hold the rule, since an `App` needs a window and a device.
pub(crate) fn record_graph_edit(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    part: PartId,
    graph: Graph,
) -> Result<(), fieldform_ops::OpError> {
    let current = doc
        .parts
        .get(&part)
        .expect("active_part must name a part in the document")
        .graph
        .clone();
    let continues = gesture.as_ref().is_some_and(|g| {
        g.part == part && g.at.elapsed() < GESTURE_WINDOW && same_structure(&current, &graph)
    });
    // A continuation reaches back to where the gesture began, for who owned what as for the
    // graph: add a blend and drag its radius, and the drag's entry undoes to a graph with no
    // blend, so the ownership it puts back must be from before the blend adopted its operands.
    // And it settles from there, so the entry it leaves is the one an edit straight from the start
    // makes, while saying only what moved since the instant before, which the last frame said
    // (`SetPartGraph`, refactor plan 2.22). Shared rather than copied, so building each instant's
    // entry copies nothing.
    let (before, ownership) = match (gesture.as_ref(), continues) {
        (Some(g), true) => (g.before.clone(), g.ownership.clone()),
        _ => (current, OwnershipSnapshot::of(doc)),
    };
    *gesture = Some(Gesture {
        part,
        before: before.clone(),
        ownership: ownership.clone(),
        at: std::time::Instant::now(),
    });
    let cmd = Box::new(SetPartGraph::with_ownership(part, before, graph, Some(ownership)));
    if continues {
        stack.replace_top(doc, cmd)
    } else {
        stack.execute(doc, cmd)
    }
}

/// Undoes one entry of the history (`back`) or redoes one, and ends the graph gesture in
/// progress.
///
/// **A gesture may only replace an entry of its own, and undo and redo both move the top.** A
/// gesture folds each new instant into the entry on top of the history, so after an undo it would
/// fold into the entry *below* the one taken back: a same-structure edit inside `GESTURE_WINDOW`
/// replaced an edit nobody had undone, and nothing could bring it back. Ending it here is the
/// whole fix, because a new gesture can only start with a new entry. Free of the `App` for the
/// reason `record_graph_edit` is.
pub(crate) fn step_history(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    back: bool,
) -> Option<String> {
    *gesture = None;
    if back { stack.undo(doc) } else { stack.redo(doc) }
}

/// What a person is told of the moves settling made (`block-ownership-spec.md` §6.3): one line,
/// for a block nothing reads any more.
///
/// **It used to say two things, and item 112 took one away.** The other line was "is now a
/// variable", for the promotion a second reader caused; settling never declares now
/// (`notebook-spec.md` §12, R5), so a second reader makes a shared instance and there is nothing
/// to announce. Adoption is still not said, for the reason it never was: adopting is what every
/// consume does, so it would be a line under every boolean. What is left is the one move nobody
/// asked for in so many words — the block that held one deleted.
///
/// **Once, and only while it is still true.** `changes` is everything settled since the frame
/// last asked, so it can hold a block twice — let go of, then adopted again by the next edit —
/// and can hold a move an undo has taken back, since an undo restores who owned what without
/// settling. So each block is judged by its last change, and only if the document bears it out.
///
/// Readers are the ones reading the block as the line is said, named in the order they were made
/// rather than by id, which is random, so the same edits always say the same line. Free of the
/// `App`, which supplies what to call a block, so a test can hold the rule.
pub(crate) fn ownership_news(
    doc: &Document,
    changes: &[OwnershipChange],
    label: impl Fn(BlockKey) -> String,
) -> Vec<String> {
    let block_of = |change: &OwnershipChange| change.block();
    let still_here = |block| match block {
        BlockKey::Node(id) => doc.parts.values().any(|p| p.graph.node(id).is_some()),
        BlockKey::Curve(id) => doc.curves.contains_key(&id),
    };
    let mut heard = std::collections::HashSet::new();
    let mut last: Vec<&OwnershipChange> =
        changes.iter().rev().filter(|c| heard.insert(block_of(c))).collect();
    last.reverse();
    last.into_iter()
        .filter(|c| still_here(block_of(c)))
        .filter_map(|change| match change {
            // **The promotion is news again, 2026-10-24** (R1 inverted): a second reader
            // declares the block, which takes it out of the slot it was drawn in and gives it a
            // row of its own. Nobody asked for that in so many words -- they wired a second
            // block to it -- so it is said, with who uses it, and the name invented for it comes
            // from the same drain it did before 112.
            //
            // Adoption stays silent: a block moving into the one thing that reads it is what the
            // person just asked for by wiring it.
            OwnershipChange::BecameVariable { block, .. }
                if doc.ownership(*block) == Ownership::Variable =>
            {
                // Who uses it as the line is said, not who did when the settle ran: a reader
                // deleted since is no block at all, and the block stays a variable either way.
                let mut readers = doc.block_readers(*block);
                readers.sort_by_key(|id| (doc.made_at(*id), id.raw()));
                let names: Vec<String> =
                    readers.iter().map(|id| label(BlockKey::Node(*id))).collect();
                let using = match names.as_slice() {
                    [] => "nothing uses it".to_string(),
                    [only] => format!("{only} uses it"),
                    [first, second] => format!("{first} and {second} both use it"),
                    [rest @ .., last] => format!("{} and {last} all use it", rest.join(", ")),
                };
                Some(format!("{} is now a variable \u{2014} {using}", label(*block)))
            }
            OwnershipChange::Released { block } if doc.ownership(*block) == Ownership::Free => {
                Some(format!("{} is on its own again", label(*block)))
            }
            _ => None,
        })
        .collect()
}

/// The blocks a drain names, in the order it names them (`block-ownership-spec.md` §4.2, §9.2).
///
/// **One rule for both now, 2026-10-24** (R1 inverted): the unnamed variables two or more blocks
/// read. In a frame they are the blocks the settle just declared, and a variable a *person* made
/// is named by the command that made it, so it is never among them. After a load they are what
/// §9.2 has always had this name: the rule a file from before 3.15 is placed by makes such a block
/// a variable directly, which no settle reports.
///
/// Not every unnamed variable: one with fewer readers is a cleared name, saved, and naming it on
/// opening lifted it back out of the reader it had been put back into, so its row moved between a
/// save and the next open.
///
/// Free of the `App`, which supplies the names, so a test can hold which blocks get one.
pub(crate) fn variables_to_name(doc: &Document, _after_load: bool) -> Vec<NodeId> {
    let mut unnamed = doc.unnamed_variables_among(doc.ownership_nodes.keys().copied());
    // Counted only for the few with no name, since each count walks the document.
    unnamed.retain(|id| doc.block_readers(BlockKey::Node(*id)).len() >= 2);
    unnamed
}

/// Parameter edits a panel made this frame, held for the graph edit they go with.
///
/// **Naming a value, and the × that puts it back to a number, are each one act with two halves:**
/// the parameter, and the graph edit pointing a field at it or away from it. The parameter half
/// used to be written straight into the document while the graph half went on the history, so
/// Ctrl+Z took back one half: undoing a × restored a `ParamRef::Named` to a parameter that was
/// gone, and undoing a naming left the parameter behind with nothing reading it (refactor plan
/// §3.3, root cause 7). Held here, the two go on the history as one entry
/// ([`record_graph_edit_then`]); what no graph edit takes is settled by
/// [`record_parameter_edits`].
/// The rows one block holds, for the fields inside it to draw (see [`App::hosted`]).
pub(crate) struct Hosted {
    /// The block whose controls are being drawn: the holder a hosted row is nested in, which is
    /// what its grip needs to take it out of the slot it is in (§7.2).
    pub(crate) holder: NodeId,
    /// What it holds and no field has claimed yet.
    pub(crate) rows: Vec<BlockRow>,
}

#[derive(Default)]
pub(crate) struct ParameterEdits {
    /// Parameters made this frame. Already in the document, so the rest of the frame reads the
    /// value the field now names; putting them on the history writes them again, unchanged.
    pub(crate) made: Vec<SetParameter>,
    /// Parameters a field stopped reading. Whether one goes is decided against the graph the edit
    /// writes, since until then the field's own block is still a reader.
    pub(crate) released: Vec<ParamId>,
}

/// Makes the parameter a value is named as: in the document now, and on the history with the
/// graph edit that points the field at it, if one does.
pub(crate) fn name_value(doc: &mut Document, edits: &mut ParameterEdits, name: String, value: f64) -> ParamId {
    let id = ParamId::new();
    let parameter = fieldform_scene::Parameter {
        id,
        name,
        value: fieldform_scene::ParamValue::Literal(value),
    };
    let mut made = SetParameter::insert(parameter);
    // An insert writes a map entry and cannot fail.
    let _ = fieldform_ops::Command::execute(&mut made, doc);
    edits.made.push(made);
    id
}

/// [`record_graph_edit`], with the parameter edits that came with it, the `shown` flags it
/// leaves stale ([`stale_shown`]), and `then`, in the same entry.
///
/// The parameters made go in front of the graph edit and the ones let go of behind it, so undo
/// puts a released parameter back before the graph reads it again, and takes a made one out only
/// once the graph has stopped reading it.
///
/// **`then` is the rest of an act the graph edit is only half of**, as the eye Pattern and Mirror
/// turn on for the body they consume is (`block-ownership-spec.md` §7.3). It goes after the graph
/// edit, which settles who owns what inside its own execute, so `then` acts on the settled state, a
/// redo replays the two in that order, and an undo takes both back. As two entries, taking a
/// pattern back took two presses of Ctrl+Z, and the one Ctrl+Y after them brought the pattern back
/// with its source no longer drawn: the eye was the entry nothing had redone. After the stale flags
/// too, so what the caller asks for is the last word on a flag.
///
/// **Never the next instant of a gesture, and the end of one.** A gesture folds each instant into
/// the entry on top by replacing it with a bare graph edit, so continuing one here would drop the
/// parameter half, or `then`, from the history while the document kept it, and a drag straight
/// after would do the same to this entry. A flag is let go of only by an edit that changes what
/// reads a block, which a gesture's next instant never does. With nothing held, nothing let go of
/// and nothing to follow, this is `record_graph_edit` exactly.
pub(crate) fn record_graph_edit_then(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    edits: &mut ParameterEdits,
    part: PartId,
    graph: Graph,
    then: Vec<Box<dyn fieldform_ops::Command>>,
) -> Result<(), fieldform_ops::OpError> {
    let mut released = std::mem::take(&mut edits.released);
    released.sort();
    released.dedup();
    let removed: Vec<SetParameter> = released
        .into_iter()
        .filter(|&id| !doc.parameter_is_read(id, Some((part, &graph))))
        .filter_map(|id| doc.parameters.get(&id).cloned().map(SetParameter::remove))
        .collect();
    let made = std::mem::take(&mut edits.made);
    let stale = stale_shown(
        doc,
        &doc.parts.get(&part).expect("active_part must name a part in the document").graph,
        &graph,
    );
    if made.is_empty() && removed.is_empty() && stale.is_empty() && then.is_empty() {
        return record_graph_edit(stack, doc, gesture, part, graph);
    }
    *gesture = None;
    let before = doc
        .parts
        .get(&part)
        .expect("active_part must name a part in the document")
        .graph
        .clone();
    // A delete that lets go of a block reads "Undo: part.set_graph" as any other graph edit does;
    // the flag it clears is not an act of its own, and nor is the eye a pattern turns on.
    let name = match (made.is_empty(), removed.is_empty()) {
        (false, _) => "parameter.name_value",
        (true, false) => "parameter.unname_value",
        (true, true) => "part.set_graph",
    };
    let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
    for edit in made {
        commands.push(Box::new(edit));
    }
    commands.push(Box::new(SetPartGraph::new(part, before, graph)));
    for flag in stale {
        commands.push(Box::new(flag));
    }
    commands.extend(then);
    for edit in removed {
        commands.push(Box::new(edit));
    }
    stack.execute(doc, Box::new(fieldform_ops::Batch::new(name, commands)))
}

/// [`record_graph_edit_then`] with nothing to follow the graph edit: what `App::set_part_graph`
/// runs, for the tests that hold it.
#[cfg(test)]
pub(crate) fn record_graph_edit_with(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    edits: &mut ParameterEdits,
    part: PartId,
    graph: Graph,
) -> Result<(), fieldform_ops::OpError> {
    record_graph_edit_then(stack, doc, gesture, edits, part, graph, Vec::new())
}

/// The `shown` flags a graph edit from `before` to `after` leaves stale, each as the command that
/// clears it: one on every block `before` consumed that nothing in `after` reads.
///
/// **Let go of, it is a body again and draws as one,** so the flag that drew it as an ingredient
/// has outlived its reason. Left standing, it drew the block beside whatever consumed it next as
/// though that had been asked for: pattern a sphere, delete the pattern, union the sphere, and the
/// sphere stood beside the union (refactor plan §3.3, root cause 9; cleared, as decided in §7).
/// Clearing it changes nothing on screen at the time: a body draws unless hidden, whatever this
/// flag says.
///
/// Only `shown`: `hidden` is what a body's eye means, and every consume clears it on purpose so a
/// block hidden and then consumed comes back when let go of (`block-ownership-spec.md` §7.2).
///
/// The flag is asked first, and few blocks hold it, so a drag's every frame walks the graph only
/// for those.
pub(crate) fn stale_shown(doc: &Document, before: &Graph, after: &Graph) -> Vec<SetShown> {
    let mut ids: Vec<NodeId> = doc
        .node_meta
        .iter()
        .filter(|(id, meta)| {
            meta.shown
                && after.node(**id).is_some()
                && before.is_consumed(**id)
                && !after.is_consumed(**id)
        })
        .map(|(id, _)| *id)
        .collect();
    // In a fixed order, so the history holds the same entry however the map iterates.
    ids.sort();
    ids.into_iter().map(|id| SetShown::new(id, false)).collect()
}

/// What is left of a frame's parameter edits once no graph edit has taken them. A parameter let go
/// of that nothing reads goes on the history as an entry of its own; whether anything did.
///
/// **A parameter made for a reference no graph edit wrote stays off the history.** The draft
/// analysis's pull offers "Name this value" on X, Y and Z, and the pull is the panel's own state,
/// not the document's. On the history alone, the parameter was the next Ctrl+Z's to take, and the
/// pull went on reading it: the field said `missing` and the pull resolved to nought. A parameter
/// goes on the history with the reference to it or not at all, so this one stays in the document as
/// every named value did before naming could be undone. Removal is the other way round, because
/// undoing it puts back a parameter that at worst nothing reads.
pub(crate) fn record_parameter_edits(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    edits: &mut ParameterEdits,
) -> Result<bool, fieldform_ops::OpError> {
    edits.made.clear();
    let mut recorded = false;
    let mut released = std::mem::take(&mut edits.released);
    released.sort();
    released.dedup();
    for id in released {
        recorded |= remove_unread_parameter(stack, doc, gesture, id)?;
    }
    Ok(recorded)
}

/// Removes parameter `id` on the history, if nothing reads it; whether it did.
///
/// Ends the graph's gesture, since the next instant of a drag would otherwise replace this entry
/// and the removal would be off the history again.
pub(crate) fn remove_unread_parameter(
    stack: &mut CommandStack,
    doc: &mut Document,
    gesture: &mut Option<Gesture>,
    id: ParamId,
) -> Result<bool, fieldform_ops::OpError> {
    if doc.parameter_is_read(id, None) {
        return Ok(false);
    }
    let Some(parameter) = doc.parameters.get(&id).cloned() else {
        return Ok(false);
    };
    *gesture = None;
    stack.execute(doc, Box::new(SetParameter::remove(parameter)))?;
    Ok(true)
}

/// The numbers a typed direction's X, Y and Z read now, held across a step through the history for
/// [`keep_typed_direction`]; `None` for a direction with nothing typed.
pub(crate) fn typed_direction(doc: &Document, sweep: &fieldform_scene::Sweep) -> Option<[f64; 3]> {
    match sweep {
        fieldform_scene::Sweep::Toward { xyz } => Some(xyz.map(|r| doc.resolve(&r).unwrap_or(0.0))),
        _ => None,
    }
}

/// Puts each of a typed direction's X, Y and Z that names a parameter the document no longer holds
/// back to the number it read before the step, `was`.
///
/// **For the draft pull, the one reader of a parameter the history cannot see.** The pull is the
/// analysis panel's own state, so it is not on the history, and its "or use" can point it at a
/// parameter a naming made. Undoing that naming takes the parameter out once nothing in the
/// document reads it, and the pull read `missing` and resolved to nought. It goes back to the
/// number instead, which is what the × beside a named value does.
pub(crate) fn keep_typed_direction(doc: &Document, sweep: &mut fieldform_scene::Sweep, was: Option<[f64; 3]>) {
    let (fieldform_scene::Sweep::Toward { xyz }, Some(was)) = (sweep, was) else {
        return;
    };
    for (r, number) in xyz.iter_mut().zip(was) {
        if let ParamRef::Named(id) = *r
            && !doc.parameters.contains_key(&id)
        {
            *r = ParamRef::Literal(number);
        }
    }
}
