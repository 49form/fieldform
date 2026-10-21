// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which blocks draw: the rule a block's eye reads, and the graph it leaves the viewport.
//!
//! **Functions of a document and a graph, and so here rather than in the application**, where
//! they were written. Everything a frame derives from a part reads this rule -- where its points
//! and planes are drawn, the box picking is sized to, the graph the tracer is handed -- and
//! building those without a window is what lets them be measured
//! (`fieldform-io/examples/overlay_cost.rs`).

use crate::{Document, EyeFlag, Graph, Node, NodeId};

/// The graph the viewport draws: the part's, with the blocks asked for added and the hidden
/// bodies taken out.
///
/// **A function of a graph rather than a method on the application**, so the order below can be
/// asked about in a test. It is the order that carries the meaning, and it was wrong: a block
/// someone asked to see was copied in *after* the hidden bodies had gone, and a hidden body
/// takes everything only it used away with it -- so the block being asked for had already left
/// with its consumer, and the copy was made out of a node that was no longer there. Reported as
/// "create a sphere, pattern it, include it, hide the pattern, show the sphere -- nothing".
///
/// Asked for first, the copy is another reader of that subtree, and
/// [`Graph::remove_body_and_its_own`]'s existing rule -- take away only what nothing else reads --
/// keeps it. Nothing about that is particular to a pattern, and nothing here asks what kind of
/// block consumed anything: it holds for a mirror, a boolean, an offset, a transform, or
/// whatever is added next.
pub fn drawn_graph(
    part: &Graph,
    shown: &[NodeId],
    hidden: &dyn Fn(NodeId) -> bool,
) -> (Graph, std::collections::HashMap<NodeId, NodeId>) {
    let mut graph = part.clone();
    // Which added root stands in for which named block, so the drawn order can be read back in
    // the caller's own terms.
    let mut stand_ins: std::collections::HashMap<NodeId, NodeId> =
        std::collections::HashMap::new();

    // **The variables asked to be drawn, before anything is taken away.** A *copy* of the
    // block, because a body is a root and this block is not one: something consumes it, which
    // is the whole reason it is not on screen already. Copying it leaves the part exactly as it
    // was -- its bodies are still the roots of its graph, derived and never stored -- and adds
    // a second, unconsumed reference to the same shape. The copy shares its operands, so it is
    // a second root over one subtree rather than a second subtree.
    for id in shown {
        if let Some(copy) = graph.add_stand_in(*id) {
            stand_ins.insert(copy, *id);
        }
    }

    for id in graph.roots() {
        if stand_ins.contains_key(&id) {
            continue;
        }
        // A plane or a lattice on its own is reference geometry, like a curve. Tracing either
        // fills the viewport with a solid that hides everything including itself, which is true
        // to the geometry and useless to look at.
        //
        // Only as a *root*. Consumed by a boolean each is doing its job, and the result is
        // bounded by whatever it cut — the whole reason an infinite operand is worth having at
        // all.
        if hidden(id) {
            // With everything only this body used, not just the node itself. Removing one node
            // leaves its operands unconsumed, which makes them bodies — so hiding a shelled
            // cube drew the solid cube it was made from, and hiding a boolean drew both the
            // shapes it had joined.
            //
            // Deleting a block deliberately does leave its operands: you are undoing an
            // operation and the shapes survive it. Hiding is not that. It says "do not draw
            // this", and drawing what it was made of instead is the one answer that is
            // certainly wrong.
            graph.remove_body_and_its_own(id);
        }
    }
    (graph, stand_ins)
}

/// The blocks something in `graph` consumes that are drawn as well, oldest first: the list
/// `fieldform_field::display_graph_with_stand_ins` hands [`drawn_graph`].
///
/// A function of the document and a graph for `drawn_graph`'s reason, so a test can ask what a
/// sequence of edits leaves drawn without a window.
pub fn shown_ingredients_in(doc: &Document, graph: &Graph) -> Vec<NodeId> {
    // Once for the walk, for the eye's reading below as well as for "not a root": a body is a
    // root exactly when nothing reads it, so the set answers both.
    let consumed = graph.consumed();
    let mut ids: Vec<NodeId> = graph
        .nodes()
        .map(|(id, _)| id)
        .filter(|id| {
            consumed.contains(id)
                && graph.node(*id).is_some_and(|n| n.is_body())
                // **Never a plane or a point.** A plane copied out as a root is a half-space
                // traced across the whole viewport, which is what "if I declare that plane a
                // variable and show it then I see a solid field body" was: the copy is added
                // after the loop that strips reference bodies out, so it was never stripped.
                // Both of these are on screen anyway, drawn by the overlay, and their eye
                // means `hidden` for that reason.
                && !draws_as_overlay_in(graph, *id)
                // What is left is consumed and drawn by no overlay, so its eye means `shown`.
                && is_visible_among(doc, graph, &consumed, *id)
        })
        .collect();
    ids.sort_by_key(|id| doc.created.get(id).copied().unwrap_or(0));
    ids
}

/// Whether block `id` of `graph` is a plane or a point, which the overlay draws:
/// `App::draws_as_overlay`, which says why.
pub fn draws_as_overlay_in(graph: &Graph, id: NodeId) -> bool {
    matches!(graph.node(id), Some(Node::Plane { .. }) | Some(Node::Point { .. }))
}

/// Which flag the eye of block `id` of `graph` means: `App::eye_flag`, which says why. For one
/// block, such as the one a click is on; a walk over many reads [`is_visible_among`].
pub fn eye_flag_in(graph: &Graph, id: NodeId) -> EyeFlag {
    eye_flag_by(graph, id, || graph.is_consumed(id))
}

/// The eye's rule, written once, with whether anything reads `id` left to the caller to answer.
///
/// "A body" is [`Graph::roots`]' rule asked of the one block. `consumed` is asked last and only
/// of a block whose answer turns on it, because asking the graph walks every node: a plane or a
/// point means `hidden` without it, which is what keeps the overlay's own per-frame reads free.
pub fn eye_flag_by(graph: &Graph, id: NodeId, consumed: impl FnOnce() -> bool) -> EyeFlag {
    if draws_as_overlay_in(graph, id)
        || (graph.node(id).is_some_and(Node::is_body) && !consumed())
    {
        EyeFlag::Hidden
    } else {
        EyeFlag::Shown
    }
}

/// Whether block `id` of `graph` draws, read through the flag its eye means: `App::is_visible`.
pub fn is_visible_in(doc: &Document, graph: &Graph, id: NodeId) -> bool {
    doc.visible_by(id, eye_flag_in(graph, id))
}

/// [`is_visible_in`] for a walk over many blocks, given `consumed`, the graph's
/// [`Graph::consumed`] built once for the whole walk.
///
/// **What a walk reads, every frame.** Asked block by block, each body's answer walks the whole
/// graph, so the drawn list -- read every frame through `selected_body_index` -- grew as bodies
/// times blocks where it had been one flag read a body, and an import of a few hundred solids
/// paid for it in milliseconds a frame. The rule is the same function either way, and
/// `a_walk_over_every_block_reads_each_eye_as_asking_that_block_alone_does` holds the two
/// readings together.
pub fn is_visible_among(
    doc: &Document,
    graph: &Graph,
    consumed: &std::collections::HashSet<NodeId>,
    id: NodeId,
) -> bool {
    doc.visible_by(id, eye_flag_by(graph, id, || consumed.contains(&id)))
}
