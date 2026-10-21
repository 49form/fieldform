// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A part's composition graph: the parametric layer, above the direct/destructive cage layer.
//!
//! Nodes produce and combine **fields**. A cage produces one via its limit surface; later, a
//! closed curve will produce one by extrusion and a free curve by distance-to-curve. Once
//! everything is a field, the graph composes them identically and does not care where any of
//! them came from — that is what keeps the two layers one system rather than two.
//!
//! # The boundary rule
//!
//! Nodes reference **whole cages and whole parts**, never individual faces or edges. That is
//! not an accident of the current node set; it is what keeps topological naming from ever
//! arising. "Blend part A with part B" stays valid however A is edited. "Fillet face 7 of
//! part A" would not, because the node re-evaluates against a cage that may since have
//! changed, and face 7 may no longer mean the same thing.
//!
//! In v1 a graph is usually a single [`Node::Cage`]. The structure is present anyway, because
//! that is the difference between the parametric layer arriving as pure addition and arriving
//! as a rewrite.

use crate::id::{CageId, NodeId};
use ::serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

mod lattice;
mod node;
mod operand;
mod params;
mod pattern;
mod serde;

pub use lattice::{Banding, CellMap, LatticeCell, LatticePhase};
pub use node::Node;
pub use operand::Operand;
pub use params::{
    no_turn, Attractor, BooleanOp, Continuity, Fillet, ParamRef, RampShape, RimEnds, Rims,
};
pub use pattern::{
    pattern_count, Axis, CountError, CountFit, PatternKind, PatternOptions, PatternSecond, Spacing,
    FULL_TURN_DEGREES, PATTERN_COUNT_MAX,
};
pub use self::serde::{UnknownNode, Unrecognised};
pub(crate) use params::turn_is_none;

/// Three Euler angles in degrees, applied intrinsically about X then Y then Z.
///
/// **A block that has a facing says so itself.** A box, a torus and a plane are each built about
/// an axis, so "which way is it turned" is part of what they are — and it used to live only in a
/// [`Node::Transform`] wrapped around them, which meant a freshly made box had no orientation to
/// edit at all and gained one only by being dragged. Two blocks holding one fact, with the block
/// that owns the shape not holding it.
///
/// **Not every shape gets one.** A sphere has no facing to turn. A cylinder, a cone and a capsule
/// are drawn *between two points*, so their direction already falls out of where those points
/// are — a separate angle beside them would be a second description of the same thing and the two
/// could disagree, which is the reason [`crate::Ends`] exists.
///
/// Three angles rather than a [`crate::Frame`] when it is typed, because a frame names an axis
/// and an axis fixes only two of the three degrees of freedom: the roll about it has to come from
/// somewhere, and a box cares about its roll. Those are the same triple [`Node::Transform`]
/// stores, read the same way, so a turn typed on a block and a turn typed on a placement mean the
/// same thing (D9).
///
/// It can also be taken from a plane or a line, which is how a cuboid or a torus gets aimed the
/// way a cylinder already could. See [`crate::turn::Turn`].
pub use crate::turn::Turn;

/// A part's graph, holding one or more **bodies**.
///
/// A body is a [root](Graph::roots): a node nothing else consumes. A part's field is the union
/// of its bodies, so two solids that have not been joined sit side by side in the same part and
/// both appear, which is what lets a pipe be built before deciding what to do with it.
///
/// Roots are derived rather than stored. A stored output is a second description of the same
/// structure, and the two can disagree — a node listed as the output but consumed by another,
/// or a body left out of the list and silently invisible. Deriving them means every node is
/// either consumed by something or is a body, with no third state to get wrong.
///
/// Equal when the nodes under every id and the critical marks are: what a shader generated from
/// one is kept against (the traced view's `program_for`), so a field added here joins the
/// comparison without anyone remembering to add it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(into = "GraphData", try_from = "GraphData")]
pub struct Graph {
    nodes: HashMap<NodeId, Node>,
    /// The surfaces somebody has said must not move. See [`Critical`] for why this is a list
    /// on the graph and not a kind of node. Kept sorted by node, so two saves agree.
    critical: Vec<Critical>,
}

/// A surface that must end up exactly where the hard boolean would put it — a bore, a seating
/// face, a mounting plane (`blend-integrity-spec.md` §4).
///
/// **A mark, not a node.** It is not a body and not an operation, and as a node kind it would
/// have made [`Graph::roots`] and every walk learn to skip it. It names a node and says what is
/// asked of it; what is *measured* about it is a report (`fieldform_field::verify_critical`),
/// which is derived and never stored here.
///
/// The mark covers the **critical patch**, not the node's whole surface — a bore's cylinder
/// runs out past the part. The patch is where the *hard part*, the same graph with every fillet
/// radius at nought, has its surface on this node. That isolates exactly what blends do:
/// offsets, twists and warps apply to both and are not blamed for moves they were asked to make.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Critical {
    pub node: NodeId,
    /// Largest movement reported as "held", in millimetres.
    pub tolerance: f64,
    /// Width of a band either side of the surface kept exact (spec §6). Zero by default, and
    /// read by nothing until protection is built; stored now so the file does not change shape
    /// when it is.
    pub clearance: ParamRef,
}

impl Critical {
    /// What `critical node=…` means with nothing else said: a hundredth of a millimetre.
    pub(crate) const DEFAULT_TOLERANCE_MM: f64 = 0.01;

    pub fn new(node: NodeId) -> Self {
        Self {
            node,
            tolerance: Self::DEFAULT_TOLERANCE_MM,
            clearance: ParamRef::Literal(0.0),
        }
    }
}

/// How a graph is written to a file.
///
/// A [`BTreeMap`] rather than the in-memory [`HashMap`], so two saves of the same graph
/// produce the same bytes. Hash order is not stable between runs, and a format whose diffs
/// are noise cannot support the semantic versioning a team needs from it — "blend radius
/// 3mm → 5mm" is only readable if nothing else moved.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphData {
    pub nodes: BTreeMap<NodeId, Node>,
    /// Format 1's single output, read so that files written by it still open. Never written.
    ///
    /// A 1.x graph meant "this node is the shape, and anything not feeding it is inert". Under
    /// bodies an unconsumed node *is* a body, so those inert nodes become visible on load.
    ///
    /// The alternative was to prune to what the output reached, and it is wrong twice over.
    /// It drops nodes, which is the one thing this format promises not to do — and it cannot
    /// be done safely anyway: a node this build could not read has no readable inputs, so
    /// pruning around one would take the nodes it was using with it. What actually appears is
    /// nothing to fear: an unreadable node refuses to evaluate either way, and the only inert
    /// node the app ever wrote was the cage that "Pipe only" left behind — which showing again
    /// is the correction, not the regression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<NodeId>,
    /// Format 3.1. **Left out when empty**, so a part nobody has marked is the same bytes a 3.0
    /// build wrote and opens in one. A part *with* marks is refused by a 3.0 reader — this
    /// struct denies unknown fields — and that is the right half of "additive": once a mark
    /// tapers a fillet (spec §6) it changes the shape, and an older build opening the file
    /// without it would be opening it wrongly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub critical: Vec<Critical>,
}

impl From<Graph> for GraphData {
    fn from(g: Graph) -> Self {
        Self { nodes: g.nodes.into_iter().collect(), output: None, critical: g.critical }
    }
}

impl TryFrom<GraphData> for Graph {
    type Error = GraphError;

    fn try_from(d: GraphData) -> Result<Self, GraphError> {
        // Dangling operands are refused here rather than at evaluation, where the only
        // available reading of a missing node is "cycle". An unknown node is exempt: what it
        // consumes is exactly the part this build cannot read, and a graph holding one
        // refuses to compile anyway.
        for (id, node) in &d.nodes {
            for input in node.inputs() {
                if !d.nodes.contains_key(&input) {
                    return Err(GraphError::DanglingInput { node: *id, input });
                }
            }
        }

        // A 1.x output that names a node the file does not contain is a damaged file, and
        // saying so beats loading a graph whose own description of itself did not hold.
        if let Some(output) = d.output {
            if !d.nodes.contains_key(&output) {
                return Err(GraphError::MissingOutput(output));
            }
        }
        // A mark on a node that is not there protects nothing while reading as though it did,
        // so it is refused by name rather than dropped (D3). In memory this cannot arise:
        // [`Graph::remove`] takes a node's mark with it.
        for mark in &d.critical {
            if !d.nodes.contains_key(&mark.node) {
                return Err(GraphError::MissingCritical(mark.node));
            }
        }
        let mut graph = Self { nodes: d.nodes.into_iter().collect(), critical: Vec::new() };
        for mark in d.critical {
            graph.mark_critical(mark);
        }
        Ok(graph)
    }
}

/// Refuses a graph that cannot be read as one, rather than loading a broken one.
#[derive(thiserror::Error, Debug, PartialEq)]
pub(crate) enum GraphError {
    #[error("the graph's output is node {0}, which the graph does not contain")]
    MissingOutput(NodeId),
    #[error("node {node} consumes node {input}, which the graph does not contain")]
    DanglingInput { node: NodeId, input: NodeId },
    #[error("node {0} is marked critical, and the graph does not contain it")]
    MissingCritical(NodeId),
}

impl Graph {
    /// The trivial graph: one cage, one body.
    pub fn from_cage(cage: CageId) -> Self {
        let mut graph = Self::empty();
        graph.add(Node::Cage(cage));
        graph
    }

    /// An empty graph, holding no bodies at all.
    pub fn empty() -> Self {
        Self { nodes: HashMap::new(), critical: Vec::new() }
    }

    /// The surfaces marked as ones that must not move, in a stable order.
    pub fn critical(&self) -> &[Critical] {
        &self.critical
    }

    /// Marks a node critical, replacing any mark it already carries — a node is critical once,
    /// and saying it again is how its tolerance is changed. `false` when the node is not here.
    pub fn mark_critical(&mut self, mark: Critical) -> bool {
        if !self.nodes.contains_key(&mark.node) {
            return false;
        }
        self.critical.retain(|m| m.node != mark.node);
        self.critical.push(mark);
        self.critical.sort_by_key(|m| m.node.raw());
        true
    }

    /// This graph with every fillet made hard, except `keep`'s if one is named.
    ///
    /// **The hard part**: what the same blocks make with nothing rounded. A critical surface is
    /// measured against it, because it isolates exactly what blends do — every offset, twist
    /// and warp is in both. With `keep`, it is the part as *one* blend alone would leave it,
    /// which is how a movement is laid at a particular blend's door.
    ///
    /// The continuity is left alone and only the radius is zeroed, so the node is the same
    /// kind with the same edges and everything keyed by node id still finds it.
    pub fn hardened(&self, keep: Option<NodeId>) -> Graph {
        let mut out = self.clone();
        for (id, node) in out.nodes.iter_mut() {
            if Some(*id) == keep {
                continue;
            }
            if let Some(fillet) = node.fillet_mut() {
                fillet.radius = ParamRef::Literal(0.0);
            }
        }
        out
    }

    /// `root` and everything it is built from, as a graph of its own — so `root` is its only
    /// body. `None` when the node is not here. Marks are not carried: this is a way of asking
    /// about one node's field, not a part.
    pub fn subtree(&self, root: NodeId) -> Option<Graph> {
        self.nodes.get(&root)?;
        let nodes = self
            .subtree_of(root)
            .into_iter()
            .filter_map(|id| self.nodes.get(&id).map(|node| (id, node.clone())))
            .collect();
        Some(Graph { nodes, critical: Vec::new() })
    }

    /// `root` and everything it is built from, however far down: [`Self::subtree`] as ids.
    ///
    /// Through [`Node::inputs`], so what a block stands on, is aimed at or is driven by is part of
    /// it as much as its operands are. The reverse of [`Self::downstream_of`], with one difference
    /// that is deliberate: **`root` is always in the answer**, where `downstream_of` has its start
    /// only through a cycle. Every caller asks about a block *together with* what it is made of --
    /// what it weighs, which rows travel with it, what has to stay when it is measured alone.
    ///
    /// An edge to a block that is not here is in the answer too, for the reason
    /// [`Self::consumed`] keeps one: the set is of what is named, not of what is here. A caller
    /// that wants the nodes asks [`Self::node`] of each, as `subtree` does. Each block is walked
    /// once, so a block read by two others costs one visit and a cycle ends the walk rather than
    /// running it forever.
    pub fn subtree_of(&self, root: NodeId) -> std::collections::HashSet<NodeId> {
        let mut seen = std::collections::HashSet::new();
        let mut walk = vec![root];
        while let Some(id) = walk.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Some(node) = self.nodes.get(&id) {
                walk.extend(node.inputs());
            }
        }
        seen
    }

    /// The blocks that read `id` directly: one step up, where [`Self::downstream_of`] goes all
    /// the way.
    ///
    /// Through [`Node::inputs`], so a block standing on `id`, aimed at it or driven by it reads it
    /// as much as one built from it does. **Each reader once**, however many times it names `id`:
    /// this walks the nodes rather than the edges, so a boolean handed the same body for both
    /// operands is one block reading it. In the graph's own order, which is no order at all; a
    /// caller that needs a chosen one sorts.
    ///
    /// **`id` is its own reader when it names itself.** That is a cycle, which nothing compiles
    /// and the editor never builds, but a file or a script can hold one. A caller asking who
    /// *else* reads it leaves `id` out itself.
    pub fn consumers_of(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes().filter(move |(_, node)| node.reads(id)).map(|(consumer, _)| consumer)
    }

    /// Whether anything reads `id`: [`Self::consumers_of`], asked only whether it is empty.
    ///
    /// Not the same question as whether it is a body. A [`Node::Ramp`] or a [`Node::Point`]
    /// nothing reads is unconsumed and still not one, and [`Self::roots`] says why.
    pub fn is_consumed(&self, id: NodeId) -> bool {
        self.consumers_of(id).next().is_some()
    }

    /// Everything downstream of `id`: what consumes it, and what consumes that, however far up.
    ///
    /// What a block may *not* be given, since a block reading something built on itself is a
    /// cycle, and a cyclic graph does not compile. Walked through [`Node::inputs`], which is
    /// where every graph edge is reported — a place, a frame and an operand alike — so a block
    /// merely standing on `id` or aimed at it is downstream too.
    ///
    /// `id` itself is in the answer only when a cycle already leads back to it. The pickers that
    /// ask this refuse `id` beside it rather than through it, so that is left to them.
    pub fn downstream_of(&self, id: NodeId) -> std::collections::HashSet<NodeId> {
        let mut found: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let mut walk = vec![id];
        while let Some(here) = walk.pop() {
            for (other, node) in self.nodes() {
                if node.inputs().contains(&here) && found.insert(other) {
                    walk.push(other);
                }
            }
        }
        found
    }

    /// The first block with an empty slot, and which slot, in id order — for a refusal to name.
    ///
    /// Ordered so that the same graph always refuses with the same block: a `HashMap`'s order is
    /// not one, and a message that names a different block each time it is asked is a message
    /// nobody can act on.
    pub fn first_empty_operand(&self) -> Option<(NodeId, usize)> {
        let mut ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        ids.sort();
        for id in ids {
            let node = &self.nodes[&id];
            let mut at = 0;
            let mut found = None;
            node.for_each_operand_slot_public(&mut |slot| {
                if slot.is_empty() && found.is_none() {
                    found = Some(at);
                }
                at += 1;
            });
            if let Some(slot) = found {
                return Some((id, slot));
            }
        }
        None
    }

    /// Every block that cannot be evaluated because a slot below it is empty: the holed blocks,
    /// and everything that reads them, however far up.
    ///
    /// **Derived from the edges, the way [`Self::roots`] is, and never stored.** A stored list
    /// would be a second description of the same structure and the two can disagree — the rule
    /// `Graph`'s own output field was removed for.
    ///
    /// **Closed downstream, which is the load-bearing part.** A block with a hole has no value, so
    /// neither has anything reading it, and a walk that skipped only the holed block would hand
    /// its consumer a missing operand — which is exactly the panic
    /// `a_body_with_an_empty_slot_is_refused_by_name_rather_than_panicking` records. With
    /// the closure skipped as one, every node a walk does reach has every operand present, so the
    /// walks need no guard of their own.
    pub fn incomplete(&self) -> std::collections::HashSet<NodeId> {
        let mut found: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        for (id, node) in self.nodes() {
            let mut holed = false;
            node.for_each_operand_slot_public(&mut |slot| holed |= slot.is_empty());
            if holed {
                found.insert(id);
            }
        }
        for id in found.clone() {
            found.extend(self.downstream_of(id));
        }
        found
    }

    /// This graph with no surface marked critical: the part as it would be unprotected.
    ///
    /// A mark tapers every fillet near it, so it changes the shape. This is how to ask what the
    /// shape was going to be — which is what says how much a mark is doing.
    pub fn without_critical(&self) -> Graph {
        Graph { nodes: self.nodes.clone(), critical: Vec::new() }
    }

    /// Takes a node's mark off. `false` when it carried none.
    pub fn unmark_critical(&mut self, node: NodeId) -> bool {
        let before = self.critical.len();
        self.critical.retain(|m| m.node != node);
        self.critical.len() != before
    }

    /// Rewrites every block that carries its own turn into that block plus a placement.
    ///
    /// **One description of what a turn means, read by both paths.** A rotated box is exactly a
    /// box wrapped in a [`Node::Transform`] turning about its own centre — that is what the turn
    /// *is*, and saying so here rather than in each evaluator means the field and the shader
    /// cannot come to different answers about it. Everything downstream then needs no new arm:
    /// intervals, normals, extents and Lipschitz bounds already know what a placement does.
    ///
    /// **The turned block keeps its id and becomes the placement**, with a fresh id for the
    /// shape inside. So whatever consumed it still consumes it, the roots are the same roots,
    /// and anything keyed by node — a body's index, a morph's divisor — keeps pointing at the
    /// thing it named.
    ///
    /// The pivot is the block's own place, so a turn spins it where it stands rather than
    /// swinging it about the part's origin. A block whose place is taken from another block
    /// turns about that block, which is the same answer and needs no special case.
    ///
    /// Both compilers reach the rewrite through [`Self::with_turns_expanded_using`], as the
    /// note inside says, so only the tests call this plain form and only the tests compile it.
    /// The description above is of the rewrite, which is the same in either form, and is the
    /// one the comments elsewhere that name this function point at.
    #[cfg(test)]
    fn with_turns_expanded(&self) -> Graph {
        // **A turn aimed at a plane or a line stays where it is.** Working out what it points
        // along needs the document, the curves and a resolver for named numbers — none of which
        // belongs in the scene crate. `fieldform_field::with_turns_flattened` passes those in
        // through [`Self::with_turns_expanded_using`], and is what both compilers call; this
        // plain form leaves an aimed block unturned, which is the state it was in before
        // anybody aimed it rather than a wrong answer about where it points.
        self.with_turns_expanded_using(&|_, _| None)
    }

    /// The same, with somebody who can say which way an aimed turn actually points.
    ///
    /// `angles_of` is asked only about the turns that are references, and answers in degrees —
    /// intrinsic XYZ, like every other rotation here. `None` leaves that block unturned.
    pub fn with_turns_expanded_using(
        &self,
        angles_of: &dyn Fn(NodeId, &Turn) -> Option<[f64; 3]>,
    ) -> Graph {
        let mut out = self.clone();
        let turned: Vec<(NodeId, Node)> = self
            .nodes
            .iter()
            // An absorbed turn is one this has already been through: its rotation is in the
            // placement and only the references are left, so there is nothing to move.
            .filter(|(_, n)| {
                n.turn().is_some_and(|t| !turn_is_none(t) && !t.is_absorbed())
            })
            .map(|(id, n)| (*id, n.clone()))
            .collect();
        for (id, node) in turned {
            let Some(carried) = node.turn().cloned() else { continue };
            let turn: [ParamRef; 3] = match carried.angles() {
                // Kept as references rather than resolved: an angle can be a named parameter,
                // and turning it into a number here would be the one place a document's own
                // name stopped driving the shape.
                Some(angles) => *angles,
                None => match angles_of(id, &carried) {
                    Some(degrees) => degrees.map(ParamRef::Literal),
                    None => continue,
                },
            };
            let about = node.places().first().map(|p| (*p).clone()).unwrap_or_else(crate::Place::origin);
            let mut flat = node;
            // **The references stay; the rotation does not.** Its numbers have moved to the
            // placement, and a block that still carries them is a rotation written down twice.
            //
            // The whole turn used to be left here when it was an aimed one, because a turn is
            // also a graph *edge* — the plane a cuboid is aimed at is consumed by it — and
            // dropping it makes that plane a body again, which is a half-space drawn across the
            // whole viewport. The comment then claimed "Nothing downstream reads a turn: the
            // placement is what everything sees", and that was the false half: `resolve_frame`'s
            // `Node::Plane` arm reads one. So a plane aimed at another was turned once for the
            // picture and *twice* for anything aimed at it — 40 degrees drawn against 80 degrees
            // mirrored, measured by `aimed_turn_probe`.
            //
            // Keeping the edge and keeping the rotation are two requirements, and only one of
            // them was ever about the turn. [`Turn::Absorbed`] is the first without the second.
            if let Some(slot) = flat.turn_mut() {
                *slot = slot.absorbed();
            }
            let inner = out.add(flat);
            out.nodes.insert(
                id,
                Node::Transform {
                    input: inner.into(),
                    translation: [ParamRef::Literal(0.0); 3],
                    rotation: turn,
                    scale: ParamRef::Literal(1.0),
                    about,
                },
            );
        }
        out
    }

    /// Adds a node. If it consumes others, those stop being bodies in the same step.
    pub fn add(&mut self, node: Node) -> NodeId {
        let id = NodeId::new();
        self.nodes.insert(id, node);
        id
    }

    /// A second, unconsumed reference to a block already in this graph — so it is a root, and
    /// so the viewport draws it beside the bodies.
    ///
    /// **Under a derived id, not a fresh one.** See [`NodeId::stand_in`]: roots are sorted by
    /// id, so a minted id moves the copy through the drawn list from one call to the next, and
    /// everything keyed on its position moves with it. `None` when there is no such block, or
    /// when the stand-in is already there — asking twice is not two copies.
    ///
    /// The copy shares its operands rather than duplicating them, so this is a second root
    /// over one subtree and not a second subtree.
    pub fn add_stand_in(&mut self, source: NodeId) -> Option<NodeId> {
        let id = source.stand_in();
        if self.nodes.contains_key(&id) {
            return None;
        }
        let node = self.nodes.get(&source)?.clone();
        self.nodes.insert(id, node);
        Some(id)
    }

    /// Copies a node and everything it is built from, with fresh ids, and returns the copy.
    ///
    /// Cages, curves and meshes are **referenced, not duplicated**. A copy of a body that
    /// mirrors a cage should mirror *that* cage — copying it would make two cages that drift
    /// apart the first time either is edited, which is the shape of bug this codebase keeps
    /// meeting.
    ///
    /// Returns `None` if the node is not in this graph.
    ///
    /// The application's Duplicate does not come here: it is copy and paste with the clipboard
    /// put back, and that pair gives the copy cages and curves of its own, which is the other
    /// answer to the question above. So only the tests call this, and only the tests compile it.
    #[cfg(test)]
    fn duplicate_subtree(&mut self, root: NodeId) -> Option<NodeId> {
        // Operands before the nodes that consume them, so every input already has its copy by
        // the time the node referring to it is rebuilt.
        let mut order: Vec<NodeId> = Vec::new();
        let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let node = self.node(id)?;
            stack.extend(node.inputs());
            order.push(id);
        }
        order.reverse();

        let mut fresh: std::collections::HashMap<NodeId, NodeId> =
            std::collections::HashMap::new();
        for id in order {
            let mut node = self.node(id)?.clone();
            node.remap_inputs(&fresh);
            let new_id = self.add(node);
            fresh.insert(id, new_id);
        }
        fresh.get(&root).copied()
    }

    /// Redirects everything that consumed `node` to consume `replacement` instead.
    ///
    /// This is what "wrap a node" has to mean in a graph where a node may already be an
    /// operand. Adding a wrapper and stopping there leaves the original with *two* consumers —
    /// the wrapper and whatever consumed it before — which is wrong in three separate ways at
    /// once: the wrapper has no consumer so it becomes a second body drawing a duplicate, the
    /// original is still read unwrapped by the old consumer, and the shader generator refuses
    /// the whole graph because one node would have to be sampled at two different points.
    ///
    /// `replacement` is skipped, since it is the one node that is *supposed* to consume
    /// `node`.
    pub fn rewire_consumers(&mut self, node: NodeId, replacement: NodeId) {
        // Through `remap_inputs` rather than through a second walk of the arms. The two used
        // to be written out separately and had to be kept in step by hand; a node kind added
        // to one and forgotten in the other rewires everywhere except the place that matters.
        let map = std::collections::HashMap::from([(node, replacement)]);
        for (id, n) in self.nodes.iter_mut() {
            if *id == replacement {
                continue;
            }
            n.remap_inputs(&map);
        }
    }

    /// The placement directly consuming `id`, if one does.
    ///
    /// What lets a caller tell an *already placed* body from an unplaced one. A gesture that
    /// moves a body works by wrapping it in a placement, and without this it wraps it again
    /// every time: the block list can select the node inside a placement as readily as the
    /// placement itself, so the second drag sees a bare node and gives it a second placement,
    /// nested, each holding a fraction of where the thing was moved to.
    ///
    /// **Returns the first it finds, and that used to be the only one there could be.** A node
    /// consumed twice would have to be sampled at two points, which shader generation refused
    /// outright — so this could not be ambiguous. Point contexts removed that refusal (backlog
    /// 21) and a node may now have any number of consumers, which
    /// `one_body_can_be_an_operand_of_two_others` builds and evaluates on both paths. So a body
    /// placed twice would give this a choice it makes arbitrarily. Nothing offers that today:
    /// a placement is created by a drag, and a drag rewires every consumer rather than adding a
    /// second placement. It is written down because the guarantee this comment used to claim is
    /// gone, and a caller relying on it would be relying on nothing.
    pub fn enclosing_transform(&self, id: NodeId) -> Option<NodeId> {
        self.nodes()
            .find(|(_, n)| matches!(n, Node::Transform { input, .. } if input.source() == Some(id)))
            .map(|(at, _)| at)
    }

    /// Removes one body, leaving whatever it consumed behind as bodies of their own.
    ///
    /// Deleting a boolean leaves the two solids it joined, which is what a user deleting one
    /// means — they are undoing the operation, not throwing away the shapes it was applied to.
    /// This file previously argued the opposite, on the grounds that a blend's operands exist
    /// to feed it and revealing them is a surprise; using it settled the question the other
    /// way. Deleting a leaf still removes exactly that leaf, since it consumes nothing.
    ///
    /// Only a body can be removed. Removing an interior node would leave whatever consumes it
    /// pointing at nothing, and there is no honest answer to which operand should replace it.
    pub fn remove_body(&mut self, id: NodeId) -> bool {
        if !self.roots().contains(&id) {
            return false;
        }
        self.remove(id)
    }

    /// Removes any node, body or not, without checking what still points at it.
    ///
    /// The unguarded one, for callers that already know the answer to the question
    /// [`Self::remove_body`] asks. A [`Node::Ramp`] or a [`Node::Point`] needs it, because
    /// neither is ever a body and the guard would refuse both: a ramp is routinely built and
    /// discarded behind the scenes, and taking a loop apart removes the joint points it no
    /// longer needs.
    pub fn remove(&mut self, id: NodeId) -> bool {
        // Its mark goes with it: a mark naming nothing is the state a file is refused for.
        self.critical.retain(|m| m.node != id);
        self.nodes.remove(&id).is_some()
    }

    /// Removes a block, keeping the blocks that used it wherever they can stand without it.
    ///
    /// **Deleting a sphere should not delete the boolean it was in.** It used to: a consumer was
    /// rewired to the surviving operand and then deleted itself, and the recursion carried on
    /// upward, so removing one leaf took the whole chain above it with it. Reported as exactly
    /// that — "when i delete a component thats being used in a Boolean block ... we should not also
    /// delete the Boolean or respective blocks thats very destructive especially if u are trying to
    /// repurpose the file/definition".
    ///
    /// So the rule is now about what a block still has to work with. A boolean, a blend or anything
    /// else made of a second operand keeps it, keeps its operation, its fillet radius and its name,
    /// and simply has the missing edge moved onto what remains — ready for another operand to be
    /// picked. A block left with no operand cannot stand, whatever else it reads: an offset of
    /// nothing is nothing, so it goes, and the same question is asked of *its* consumers.
    ///
    /// A boolean whose two operands become the same block is a real state and is left alone rather
    /// than guessed at; the boolean's own panel says what it means, since a difference of a body
    /// with itself is empty and that is worth knowing before wondering where the body went.
    ///
    /// On the graph rather than in the application, so the rule can be asserted without one.
    pub fn remove_cascading(&mut self, id: NodeId) {
        let consumers: Vec<NodeId> = self.consumers_of(id).collect();

        for consumer in consumers {
            let Some(node) = self.node(consumer).cloned() else { continue };
            // A boolean of three or more loses this operand and keeps the rest: "these five,
            // joined" less one is "these four, joined", not nothing.
            if let Node::Boolean { inputs, .. } = &node
                && inputs.len() > 2
                && let Some(Node::Boolean { inputs: live, .. }) = self.node_mut(consumer)
            {
                // **The slot goes rather than emptying**, which is what "these four, joined"
                // means: a boolean above two operands keeps its arity by losing the whole slot,
                // where emptying it would leave a hole nobody asked for in a block that is still
                // complete. §7.2's drag-out is the gesture that empties one, and it is not this.
                live.retain(|input| input.source() != Some(id));
                continue;
            }
            // Otherwise, whatever else it is made of. A block with another operand has something
            // left to act on, so it stays and the edge moves; one with nothing left has no meaning
            // without its input and goes, which cascades again from there.
            //
            // **An operand, never a block it is only aimed at or driven by.** The stand-in used
            // to be the first other *input*, and a frame's block, a pivot's point and a ramp are
            // all inputs -- so deleting the body of a mirror aimed at a plane left the mirror
            // reflecting the plane, and a driven offset growing its own ramp (refactor plan
            // §3.3, root cause 1). Those told the block where, or by how much; none of them is
            // something to act on.
            //
            // An attract's target is the one such block `operands` lists, because the point is
            // sampled inside whatever placement wraps the attract and the placement walk has to
            // reach it there. It is still where the body is pulled, not a body to pull.
            let target = match &node {
                Node::Attract { toward, .. } => toward.node(),
                _ => None,
            };
            match node.operands().into_iter().find(|o| *o != id && Some(*o) != target) {
                Some(survivor) => {
                    let map = std::collections::HashMap::from([(id, survivor)]);
                    if let Some(node) = self.node_mut(consumer) {
                        node.remap_inputs(&map);
                    }
                }
                None => self.remove_cascading(consumer),
            }
        }
        // **`remove_body` asks whether this is a root, and a ramp can never answer yes.** By this
        // line every consumer has already been rewired or removed above, which *is* the question
        // the guard exists to ask — so asking it again of a node kind that is never a body would
        // make a ramp the one block in the list that cannot be deleted. `ramp-decision.md` §7.2
        // puts a ramp in the list; being in the list and undeletable is worse than not being
        // there.
        if !self.remove_body(id) {
            self.remove(id);
        }
    }

    /// Removes a body together with everything only it was using.
    ///
    /// Downward, where [`Self::remove_cascading`] goes upward. An operand kept alive by something
    /// else stays, which is what makes this safe on a graph where one body feeds two.
    pub fn remove_body_and_its_own(&mut self, id: NodeId) {
        let held = self.node(id).map(|n| n.inputs()).unwrap_or_default();
        if !self.remove_body(id) {
            return;
        }
        for input in held {
            // Only now can this be a root, and only if nothing else was reading it.
            if !self.is_consumed(input) {
                self.remove_body_and_its_own(input);
            }
        }
    }

    /// Turns every parameter `id` was driving into the number it currently reads.
    ///
    /// On the graph so the property can be asserted without the application, and because it is
    /// the whole of what deleting a ramp has to do before the cascade runs.
    /// [`Self::remove_cascading`] moves every edge naming the deleted block, a driver's among
    /// them, onto another of the consumer's operands, and would hand a driven socket that
    /// operand, which is a body's field driving an amount: refused at compile as
    /// `DrivenParameterNeedsARamp`, so the part goes dark on a delete.
    ///
    /// Every socket, not the ones a panel happens to know about. A driver is an edge like any other
    /// ([`Node::inputs`] gathers it from `param_refs`), so the sweep is over `param_refs` too.
    pub fn release_drivers(&mut self, id: NodeId, settle_to: f64) {
        let mut released: Vec<(NodeId, usize)> = Vec::new();
        for (holder, node) in self.nodes() {
            for (i, reference) in node.param_refs().iter().enumerate() {
                if reference.driver() == Some(id) {
                    released.push((holder, i));
                }
            }
        }
        for (holder, i) in released {
            if let Some(node) = self.node_mut(holder)
                && let Some(slot) = node.param_refs_mut().get_mut(i)
            {
                **slot = ParamRef::Literal(settle_to);
            }
        }
    }

    /// The bodies: nodes nothing else consumes. **Which ones, not in what order.**
    ///
    /// Sorted by id rather than left in hash order, so the answer cannot vary between runs. That
    /// is the whole of the promise: an id is drawn at random (`id.rs` says why, and why that is
    /// right), so ranking bodies by one ranks them by nothing, and a new body took a rank drawn
    /// uniformly among these — renaming the generated function of every body it landed in front
    /// of, and recompiling a part that had not changed.
    ///
    /// So **a reader that needs a chosen order asks the document.**
    /// [`crate::Document::folded_bodies`] ranks the same nodes by when they were made, and
    /// [`crate::Document::listed_bodies`] gives the order the block list shows. They are two
    /// orders because a drag reorders the list without touching the fold (`block-order-spec.md`
    /// §8.5, which is also why the fold's function stopped being called `bodies_of`).
    /// Deterministic is not the same as decided, and this only ever offered the first.
    /// `body-order-decision.md` carries the reasoning and
    /// `fieldform-gpu/examples/body_order_probe.rs` the measurements.
    ///
    /// A [`Node::Ramp`] or a [`Node::Point`] is never a body, even when nothing reads it, and
    /// [`Node::is_body`] is where both reasons live. A ramp's value is a thickness or a radius
    /// rather than a distance, so its zero set is not a surface. A point's field is never
    /// negative, so it adds nothing to the union (decided 2026-09-26). A reader that wants
    /// points as well -- a blend's operand list, the view's framing -- adds them itself.
    pub fn roots(&self) -> Vec<NodeId> {
        let consumed = self.consumed();
        let mut roots: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(id, node)| !consumed.contains(id) && node.is_body())
            .map(|(id, _)| *id)
            .collect();
        roots.sort_by_key(|id| id.raw());
        roots
    }

    /// Every node something reads, so a node not in it is a head: where a walk of the whole
    /// graph has to start.
    ///
    /// In one pass, where asking [`Self::is_consumed`] of every node walks the graph once per
    /// node. [`Self::roots`] are the heads that are geometry, and the evaluation order is seeded
    /// from all of them; both read this. An edge to a node already removed counts too, because
    /// the set is of what is read, not of what is here.
    pub fn consumed(&self) -> std::collections::HashSet<NodeId> {
        let mut consumed = std::collections::HashSet::new();
        for node in self.nodes.values() {
            node.for_each_input(|id| {
                consumed.insert(id);
            });
        }
        consumed
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// A node, to change what it holds — a pipe's radius, a fillet's size.
    ///
    /// Editing a node's *inputs* through this would leave the bodies stale for as long as the
    /// borrow lasts, but they are derived on demand rather than cached, so there is nothing to
    /// invalidate: the next call to `roots` sees whatever the graph now says.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Every node in the graph, including any the output does not reach.
    ///
    /// Order is unspecified — it is a `HashMap` — so callers that need a stable one sort.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(id, node)| (*id, node))
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every cage any node reads.
    pub fn referenced_cages(&self) -> Vec<CageId> {
        let mut out: Vec<CageId> = self
            .nodes
            .values()
            .filter_map(|n| match n {
                Node::Cage(c) => Some(*c),
                _ => None,
            })
            .collect();
        out.sort_by_key(|c| c.raw());
        out.dedup();
        out
    }

    /// Nodes in dependency order, inputs before the nodes that consume them, bodies last.
    ///
    /// Every node is covered, not only what one output reaches: each body is its own branch and
    /// all of them are drawn.
    ///
    /// Returns `None` if the graph contains a cycle. A cycle should be impossible to build
    /// through the editing UI, but the graph is also reachable from scripts and from files,
    /// so evaluation must refuse rather than recurse forever. A cycle also has no root, so it
    /// would otherwise be silently invisible rather than refused.
    pub fn evaluation_order(&self) -> Option<Vec<NodeId>> {
        self.evaluation_order_from(&|_| 0)
    }

    /// The same, with the seeds ranked by something the graph cannot know.
    ///
    /// **Because where the walk starts decides every name in the generated source.** The seeds
    /// are the unconsumed nodes, and they were sorted by raw id — which is drawn at random, so a
    /// newly created node seeded the walk at a random point and renumbered the `n{k}` slots and
    /// `params[..]` indices of everything visited after it. Fixing the *fold* order alone left
    /// this: the body functions kept their names and their contents were rewritten, so the driver
    /// still compiled the whole part. Caught by
    /// `codegen::tests::a_body_already_in_the_part_keeps_its_function_when_another_is_added`,
    /// which compares the generated text rather than the order — an order-level test passes with
    /// this bug in place.
    ///
    /// Ranked by *all* the unconsumed nodes rather than only the bodies, because the seeds are all
    /// of them: a [`Node::Ramp`] nothing reads is unconsumed and not a body, and leaving it out of
    /// the seeding would make it unreachable and read as a cycle — see the note below.
    ///
    /// A caller holding a document passes `Document::made_at`, and
    /// [`crate::Document::evaluation_order_of`] is that call. A caller with no document passes
    /// nothing and gets the raw-id order, which is arbitrary and at least fixed.
    pub(crate) fn evaluation_order_from(
        &self,
        rank: &impl Fn(NodeId) -> u64,
    ) -> Option<Vec<NodeId>> {
        let mut order = Vec::new();
        let mut state: HashMap<NodeId, u8> = HashMap::new(); // 0 unvisited, 1 in progress, 2 done
        // Seeded from everything nothing consumes, which is *not* the same as the bodies:
        // `roots` additionally leaves out nodes that are not geometry, and one of those with
        // no consumer would then be unreachable here and read as a cycle by the count below.
        // That is exactly what happened when ramps stopped being bodies — a part holding an
        // unread ramp refused to evaluate at all, and the block list was empty because a graph
        // reporting a cycle has no roots to show.
        let consumed = self.consumed();
        let mut unconsumed: Vec<NodeId> =
            self.nodes.keys().copied().filter(|id| !consumed.contains(id)).collect();
        // Sorted for the same reason `roots` is: this decides evaluation and generation order,
        // and neither may vary between runs. By the caller's rank first, so a creation seeds the
        // walk *after* everything already there instead of in the middle of it.
        unconsumed.sort_by_key(|id| (rank(*id), id.raw()));
        let mut stack: Vec<(NodeId, bool)> =
            unconsumed.into_iter().rev().map(|id| (id, false)).collect();

        while let Some((id, expanded)) = stack.pop() {
            if expanded {
                state.insert(id, 2);
                order.push(id);
                continue;
            }
            match state.get(&id) {
                Some(2) => continue,
                Some(1) => return None, // back edge: cycle
                _ => {}
            }
            state.insert(id, 1);
            stack.push((id, true));
            let node = self.nodes.get(&id)?;
            node.for_each_input(|input| stack.push((input, false)));
        }
        // A node reached from nothing unconsumed is one a cycle is holding: anything else is
        // either consumed — and so reachable from whatever consumes it — or was a seed above.
        if order.len() != self.nodes.len() {
            return None;
        }
        Some(order)
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod boolean_list_tests;
