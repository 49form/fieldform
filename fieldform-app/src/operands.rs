// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Which blocks can stand as another block's operands, and which pair a new one starts from.

use super::*;

/// Whether a twist could take `candidate`, asked before any twist exists.
///
/// **The same rule the panel applies, asked by the tool that creates the block.** A tool that
/// hands itself an operand its own panel would grey is a tool disagreeing with itself, and the
/// way to stop that is for both to ask one function. `operand_refusal` needs a consumer node to
/// read, and at creation there is not one yet -- so this builds the question it would ask: is this
/// bounded, which is what a twist needs and what `CompileError::TwistNeedsBounds` refuses without.
pub(crate) fn twist_would_take(
    document: &fieldform_scene::Document,
    graph: &Graph,
    candidate: NodeId,
) -> bool {
    !matches!(graph.node(candidate), Some(Node::Point { .. }))
        && !fieldform_field::graph_extent(document, graph, candidate).is_unbounded()
}

/// Why `candidate` cannot stand in as an operand of `consumer`, or `None` if it can.
///
/// **A field offers only what compiles.** The list used to be one list for every block — anything
/// that is a field, minus what would loop — so a twist offered a plane and a boolean offered a
/// point, and the way you found out was a refused compile or a body that quietly did nothing.
/// `CompileError::TwistNeedsBounds` is exactly "you twisted something unbounded"; the picker was
/// the place to say so, and it was the place saying nothing.
///
/// **The rule is what the block does to the field**, not a list of bad pairs:
///
/// - **Needs an interior**, so it refuses a point — which has none, its value never being
///   negative. A boolean folds by `min`/`max` and reads only the sign, so a point adds nothing to
///   one; a shell has no wall to build; a morph has nothing to mix toward; an attractor has
///   nothing to warp.
/// - **Needs a bound**, so it refuses anything unbounded. Asked by *extent* rather than by kind,
///   which is the question `is_reference_body` already asks and the reason a trimmed lattice is
///   not caught by it: a twist divides by a stretch taken from the body's own extent, and an
///   attractor and a pattern both reach over one.
/// - **Rigid or additive**, so it carries anything. A mirror and a transform are motions; an
///   offset of a point is a sphere, exactly defined, and is a legitimate way to make one — which
///   is why "a point is never valid" would be the wrong rule.
/// - **A blend reaches to a point on purpose** (`fieldform-blend-to-a-point`), so it is the one
///   combiner that takes one.
///
/// A pattern of a point is refused on the owner's ruling and on a fact about the application
/// rather than the mathematics: the copies would be drawn by nothing and named by nothing. Every
/// consumer of a point — the viewport overlay, a control point's target, a blend's reach — matches
/// `Node::Point` itself, and a pattern wrapping one is not that, so the copies exist only as a
/// field that is never negative. "If you can pattern a point and make copies it may be useful,
/// otherwise no."
pub(crate) fn operand_refusal(
    document: &fieldform_scene::Document,
    graph: &Graph,
    consumer: NodeId,
    candidate: NodeId,
) -> Option<&'static str> {
    let taker = graph.node(consumer)?;
    let given = graph.node(candidate)?;
    let is_point = matches!(given, Node::Point { .. });
    // By extent, not by kind: a lattice trimmed to a shell is bounded and a bare one is not, and
    // the compile asks the same question of a twist.
    let unbounded = fieldform_field::graph_extent(document, graph, candidate).is_unbounded();

    if is_point {
        return match taker {
            Node::Boolean { .. } => {
                Some("a point encloses no volume, so a boolean has nothing to combine with it")
            }
            Node::Shell { .. } => Some("a point has no interior, so there is no wall to build"),
            Node::Morph { .. } => Some("a morph mixes two solids, and a point has no volume"),
            Node::Twist { .. } => Some("there is nothing in a point to wind"),
            Node::Attract { .. } => Some("a point has no material to draw toward anything"),
            Node::Pattern { .. } => {
                Some("copies of a point are drawn by nothing and named by nothing")
            }
            // A blend reaches to one on purpose; an offset of one is a sphere; a mirror and a
            // transform are motions and carry anything.
            _ => None,
        };
    }
    if unbounded {
        return match taker {
            Node::Twist { .. } => {
                Some("a twist divides by a stretch taken from the body's own extent, and this one \
                      has none")
            }
            Node::Attract { .. } => Some("an attractor reaches over a body, and this one has no end"),
            Node::Pattern { .. } => Some("copies of something unbounded all fill the same space"),
            _ => None,
        };
    }
    None
}

/// The nodes that may stand in as an operand of `id`: every block but a ramp, whose value is a
/// number rather than a field, except `id` itself and anything that already reads it, which
/// would make a loop. The rule and the name gate it replaced are told beside the code, and
/// `every_block_is_offered_to_every_other_and_only_a_loop_is_refused` holds it.
///
/// A second consumer is ordinary, and once it was not: choosing a node something else already
/// consumed gave it two, and if the other was a placement the shader could not be generated at
/// all, so the viewport went blank with only a line of text to say why. Point contexts
/// (backlog 21) made a point a *path* rather than an index, so a node reached through two
/// parents is evaluated at two contexts; `one_body_can_be_an_operand_of_two_others` builds
/// exactly that and checks both the CPU and the lowering.
///
/// A body stays a body: the roots are still the nodes nothing consumes, so a node picked up by a
/// second operation simply stops being one, and the part's field does not gain a duplicate.
pub(crate) fn operand_choices_in(
    graph: &Graph,
    id: NodeId,
    made_at: impl Fn(NodeId) -> u64,
) -> Vec<NodeId> {
    // **Nothing that already consumes this block, however far up.** Choosing one would make a
    // loop -- this block feeding the thing that feeds it -- and a cyclic graph does not compile,
    // so the answer would be a part that has stopped drawing and an error naming a node rather
    // than the click that caused it. It is also saveable, which makes it a file that opens to
    // the same error.
    //
    // Cheap to exclude and impossible to reach, which beats detecting it afterwards. This
    // mattered little while only a boolean offered a choice; every block offers one now.
    let loops_back = graph.downstream_of(id);
    // **And every point**, which is not a body (`Node::is_body`) and so is no longer offered
    // as one, but is still something a block takes: a blend reaching to a point above all,
    // asked for as "when u pick a sdf and a point it auto changes to g1 and u can blend it".
    // A boolean declines a point on its own account, in its panel and in its starting pair.
    let is_point = |n: NodeId| matches!(graph.node(n), Some(Node::Point { .. }));
    // **Everything, named or not.** A name used to be the gate: a block something else consumed
    // was offered only once it had one, on the reading that a name is what declares a block
    // reusable. Asked for directly -- "both un-named and named objects should be available for
    // boolean" -- and the reading was serving the wrong question. A name says how to *refer* to
    // a block among several that would otherwise read alike; whether a block can be taken is a
    // question about the graph, and the graph's answer is yes for anything that is a field and
    // does not already read this block.
    //
    // What is still refused, and why each: this block itself, and anything that already reads it
    // however far up (`loops_back`) -- those make a cycle, which does not compile and is
    // saveable, so the answer would be a part that has stopped drawing and an error naming a
    // node rather than the click that caused it. And a `Ramp`, which `Node::is_body` excludes:
    // a ramp is a number, not a field, so there is nothing for a boolean to do with one.
    let takeable = |n: NodeId| graph.node(n).is_some_and(|node| node.is_body()) || is_point(n);
    let mut ids: Vec<NodeId> = graph
        .nodes()
        .map(|(n, _)| n)
        .filter(|n| *n != id && !loops_back.contains(n) && takeable(*n))
        .collect();
    // By when they were made, like every other list of blocks: a dropdown ordered by random id
    // is a dropdown that reshuffles when something unrelated is created.
    ids.sort_by_key(|n| (made_at(*n), n.raw()));
    ids
}

/// Every block a pair-taking tool may start from: the bodies first, then everything else that is
/// a field.
///
/// **A name used to be the gate on the tail, and is not any more.** It was, on the reading that
/// naming declares a block reusable — which held while a name was also what let a consumed block
/// into an operand picker at all. `operand_choices_in` stopped asking (a96b7c5, "both un-named and
/// named objects should be available for boolean"), and this is the other half of the same
/// opening: whether a block can be taken is a question about the graph, and the graph's answer is
/// yes for anything that is a field. A name says how to *refer* to one.
///
/// What it buys beyond consistency: `Pattern` and `Mirror` used to name the body they consumed on
/// the user's behalf, for no reason except to keep it in this list — so the app was writing names
/// nobody asked for, and "named" stopped meaning "the person named it". That mattered once naming
/// became the thing that lifts a block out of the block it is drawn inside
/// (`Document::nesting_in`), which is why the gate went here rather than being worked around
/// again.
///
/// Reported before that, and still the case it has to answer: a sphere and a cube both fed into a
/// boolean, with the morph tool refusing because the part had one root left. The two operands were
/// right there.
///
/// Bodies first and in their own order, then the points, then the rest — every group by when it
/// was made, so the pair a tool starts from is the same pair every time it is asked and a tool's
/// *starting* pair is still a body wherever there are two. The bodies arrived in `roots()` order
/// once, which is a random id, so the pair moved whenever an unrelated block was created.
/// `made_at` arrives as a closure because the rule is worth asserting on its own and an `App`
/// needs a window and a device before it will exist.
pub(crate) fn pair_candidates_in(graph: &Graph, made_at: &impl Fn(NodeId) -> u64) -> Vec<NodeId> {
    let mut out = graph.roots();
    out.sort_by_key(|id| (made_at(*id), id.raw()));
    // Then the points, which are not bodies but which a blend reaches to; a boolean filters
    // them back out through `starting_pair_among`.
    let mut points: Vec<NodeId> = graph
        .nodes()
        .filter(|(id, n)| matches!(n, Node::Point { .. }) && !out.contains(id))
        .map(|(id, _)| id)
        .collect();
    points.sort_by_key(|id| (made_at(*id), id.raw()));
    out.extend(points);
    // Everything else a block could be given, which is the same test `operand_choices_in`
    // applies: a field, whether or not anybody has named it. A `Ramp` is the one thing left
    // out, by `is_body`, because its value is a number rather than a distance.
    let mut also: Vec<NodeId> = graph
        .nodes()
        .filter(|(id, n)| !out.contains(id) && n.is_body())
        .map(|(id, _)| id)
        .collect();
    also.sort_by_key(|id| (made_at(*id), id.raw()));
    out.extend(also);
    out
}

/// The outermost block `id` is drawn inside, or `id` itself when it has a row of its own.
///
/// **What a selection means once blocks nest.** A row inside a pattern is the block the pattern is
/// *made of*; selecting it and asking for a boolean used to reach past the pattern and take that
/// block, which is not what "union these two" means. Worse, it was self-fulfilling: a block nests
/// while exactly one thing consumes it, so the boolean taking the pipe as well as its pattern gave
/// the pipe two readers and evicted it to a row of its own. Reported as "it kicks the pipe out of
/// the pattern and it tries to union the pipe with the pattern instead of the pattern with the
/// pattern".
///
/// **Outermost rather than one step up**, because nesting is a chain: a pipe inside a pattern
/// inside a boolean is reached from the boolean's row, and stopping at the pattern would answer a
/// row nobody clicked. Bounded, because this walks a relation built from the graph and a graph
/// that cycles is refused elsewhere rather than here.
///
/// **Every remembered operand is lifted, and the first version of this lifted only the
/// selection.** The distinction that matters is not `selected_body` against `boolean_a`: it is a
/// tool's *memory* against a block's *own operand fields*. `boolean_a` and `boolean_b` are only
/// ever set as a seed — "point the boolean tool at what was just made", "the result becomes the
/// next pair's first operand" — and never by the panel, which edits the `Node::Boolean`'s inputs
/// directly. A seed is a guess about what somebody will reach for next, so it has to follow what
/// they did to it; an operand field is a statement, and nothing here touches one.
///
/// What leaving the seeds unlifted cost, reported with two screenshots: a pipe, then a pattern of
/// it, then `boolean -u`. The pipe tool had pointed `boolean_b` at the raw `Pipe` node, the
/// pattern was built over it afterwards, and the union took the pipe the seed still named — so
/// the pipe gained a second reader, left the pattern it was drawn inside, and the list came apart
/// into a pipe, an empty pattern, a union and the sweep. Nothing had gone wrong with the nesting;
/// a stale seed had named something the person had since put inside something else.
pub(crate) fn outermost_in(nesting: &fieldform_scene::Nesting, id: NodeId) -> NodeId {
    // **The lowest id where a block has more than one owner, since 112.** A shared instance is
    // drawn inside every block that reads it (§7.3), so "the block this is inside" can be two
    // blocks and the walk has to pick one. The lowest id is deterministic, which is what this
    // needs — the same edits must lift a seed to the same block every time — and where there is
    // one owner, which is every case this was written for, it is that owner.
    let mut at = id;
    for _ in 0..256 {
        match nesting.owners_of(at).first() {
            Some(owner) => at = *owner,
            None => break,
        }
    }
    at
}

/// The pairing rule with every candidate admitted. Kept for the tests that pin the rule; the
/// application asks [`starting_pair_among`], which is where a boolean declines a point.
#[cfg(test)]
pub(crate) fn starting_pair_in(
    graph: &Graph,
    made_at: impl Fn(NodeId) -> u64,
    chosen_a: Option<NodeId>,
    chosen_b: Option<NodeId>,
) -> Option<(NodeId, NodeId)> {
    starting_pair_among(graph, made_at, chosen_a, chosen_b, |_| true)
}

/// The pair a block that takes two operands starts from, given what is chosen and what exists,
/// among the candidates `allowed` admits. A boolean admits no point, since a point encloses no
/// volume to combine; a blend can reach to one, so the rule is the caller's.
///
/// Free of the app for the same reason `operand_choices_in` is: the rule is worth asserting, and
/// an `App` needs a window and a device before it will exist.
pub(crate) fn starting_pair_among(
    graph: &Graph,
    made_at: impl Fn(NodeId) -> u64,
    chosen_a: Option<NodeId>,
    chosen_b: Option<NodeId>,
    allowed: impl Fn(NodeId) -> bool,
) -> Option<(NodeId, NodeId)> {
    let usable: Vec<NodeId> = pair_candidates_in(graph, &made_at)
        .into_iter()
        .filter(|id| allowed(*id))
        .collect();
    let pick = |chosen: Option<NodeId>, avoid: Option<NodeId>| {
        chosen
            .filter(|id| usable.contains(id) && Some(*id) != avoid)
            .or_else(|| usable.iter().copied().find(|id| Some(*id) != avoid))
    };
    let a = pick(chosen_a, None)?;
    Some((a, pick(chosen_b, Some(a))?))
}
