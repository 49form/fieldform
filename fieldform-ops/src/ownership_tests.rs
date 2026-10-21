// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Spec tests 11 and 12 of `block-ownership-spec.md`: every edit made through a real history
//! leaves the ownership invariants holding, and an undo puts ownership back exactly.
//!
//! Here rather than in `fieldform-scene`, where settling is, because what they hold is a property
//! of the history: settling runs inside the commands and not in the stack (spec §4.1), and only
//! this crate can drive a real [`CommandStack`] without the application.

use crate::{
    Batch, Command, CommandStack, Document, OwnershipSnapshot, SetCurve, SetOwnership,
    SetPartGraph, SetShown,
};
use fieldform_scene::{
    BlockKey, BooleanOp, Curve, CurveId, Fillet, Frame, Graph, Node, NodeId, Outline, Ownership,
    OwnershipChange, ParamRef, Part, PartId, PatternKind, Place, Sides, Sweep,
};
use nalgebra::Point3;
use std::collections::{BTreeMap, HashMap};

/// Numbers that are the same every run, so a seed that fails fails again — given lists to pick
/// from that are the same every run too, which is what [`Born`] keeps.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn percent(&mut self, chance: u64) -> bool {
        self.next() % 100 < chance
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> Option<T> {
        (!from.is_empty()).then(|| from[self.below(from.len())])
    }
}

/// The order every block was made in, which is the order the cases pick from.
///
/// Not the order of their ids: an id is drawn at random on every run, so a list sorted by id is a
/// different list each time, the same seed makes a different sequence of edits, and a failure's
/// seed and step would not replay it. A block is noted where it is made, so one an undo or a redo
/// brings back keeps the place it had.
#[derive(Default)]
struct Born(HashMap<BlockKey, usize>);

impl Born {
    fn node(&mut self, id: NodeId) -> NodeId {
        self.note(BlockKey::Node(id));
        id
    }

    fn curve(&mut self, id: CurveId) -> CurveId {
        self.note(BlockKey::Curve(id));
        id
    }

    fn note(&mut self, block: BlockKey) {
        let next = self.0.len();
        self.0.entry(block).or_insert(next);
    }

    /// `items` in the order their blocks were made. One made without being noted has no place in
    /// that order, and stops the test rather than being put anywhere.
    fn in_order<T: Copy>(
        &self,
        items: impl IntoIterator<Item = T>,
        block: impl Fn(T) -> BlockKey,
    ) -> Vec<T> {
        let mut out: Vec<T> = items.into_iter().collect();
        out.sort_by_key(|item| {
            let key = block(*item);
            *self.0.get(&key).unwrap_or_else(|| panic!("{key:?} was made without being noted"))
        });
        out
    }
}

fn lit(value: f64) -> ParamRef {
    ParamRef::Literal(value)
}

fn ball(graph: &mut Graph) -> NodeId {
    graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(5.0) })
}

fn line(name: &str) -> Curve {
    Curve::polyline(
        CurveId::new(),
        name,
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
        false,
    )
    .expect("a two-point line")
}

/// A document of `parts` empty parts and `curves` lines nothing reads yet, settled, and the order
/// its lines were made in.
fn document(parts: usize, curves: usize) -> (Document, Vec<PartId>, Born) {
    let mut doc = Document::new();
    let mut born = Born::default();
    let parts = (0..parts)
        .map(|i| {
            doc.add_part(Part {
                id: PartId::new(),
                name: format!("Part {i}"),
                material: None,
                graph: Graph::empty(),
                inputs: Vec::new(),
            })
        })
        .collect();
    for i in 0..curves {
        let curve = line(&format!("Line {i}"));
        doc.curves.insert(born.curve(curve.id), curve);
    }
    doc.settle_ownership();
    doc.pending_ownership_changes.clear();
    (doc, parts, born)
}

fn graph(doc: &Document, part: PartId) -> Graph {
    doc.parts[&part].graph.clone()
}

/// A graph edit as the application makes one that starts no gesture: the part's graph as it is,
/// and the one it becomes.
fn edit(doc: &mut Document, stack: &mut CommandStack, part: PartId, after: Graph) {
    let before = graph(doc, part);
    stack.execute(doc, Box::new(SetPartGraph::new(part, before, after))).expect("edits the graph");
}

/// Who reads a block, counted the way spec I2 says and written out again here rather than taken
/// from the code under test: a node's distinct readers in its own part, a curve's in every part.
fn readers_in(doc: &Document, block: BlockKey) -> Vec<NodeId> {
    let mut out: Vec<NodeId> = Vec::new();
    for part in doc.parts.values() {
        for (id, node) in part.graph.nodes() {
            let reads = match block {
                BlockKey::Node(child) => {
                    part.graph.node(child).is_some() && node.inputs().contains(&child)
                }
                BlockKey::Curve(curve) => node.curves().contains(&curve),
            };
            if reads {
                out.push(id);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Every block there is to own: each node of each part and each curve of the document.
fn every_block(doc: &Document) -> Vec<BlockKey> {
    let mut out: Vec<BlockKey> = doc
        .parts
        .values()
        .flat_map(|p| p.graph.nodes().map(|(id, _)| BlockKey::Node(id)))
        .chain(doc.curves.keys().map(|id| BlockKey::Curve(*id)))
        .collect();
    out.sort();
    out
}

fn ownership_of(doc: &Document, block: BlockKey) -> Option<Ownership> {
    match block {
        BlockKey::Node(id) => doc.ownership_nodes.get(&id).copied(),
        BlockKey::Curve(id) => doc.ownership_curves.get(&id).copied(),
    }
}

fn name_of(doc: &Document, block: BlockKey) -> Option<String> {
    match block {
        BlockKey::Node(id) => doc.node_meta.get(&id).and_then(|m| m.name.clone()),
        BlockKey::Curve(id) => doc.curves.get(&id).map(|c| c.name.clone()),
    }
}

/// The application's half of a settle (spec §4.2), as far as the invariants need it: it takes the
/// changes and leaves the document as it found it.
///
/// **It used to name here, and since 112 there is nothing to name.** A second reader promoted a
/// block to a variable, which arrived nameless, so the application invented a name off the history
/// and an undo left it standing — an allowance every state comparison below had to carry. Settling
/// never declares now (`notebook-spec.md` §12, R5): a `Variable` is written by the person asking
/// and by nothing else, with the name they typed in the same entry, so every name is on the
/// history and an undo has to put every one of them back exactly.
fn drain(doc: &mut Document) -> Vec<OwnershipChange> {
    std::mem::take(&mut doc.pending_ownership_changes)
}

/// I2 and I4 over every block, and nothing left for a settle to move.
///
/// An id with no entry reads as free (spec §3). A `Nested` block has at least one reader, which is
/// what I2's "the set of readers is exactly `{p}`" became when the parent stopped being stored
/// (`notebook-spec.md` §12), readers counted as I2 counts them — in its own part for a node. The
/// last check is the one that says the edit settled at all: a settle run now must find nothing to
/// adopt and nothing to release.
///
/// **I1 and I3 went with the stored parent.** I1 asked that the block a nested one named exist and
/// I3 that no chain of those names close on itself; `Nested` names nothing, so neither can be
/// written down any more. The ring a malformed file can still hold is a ring of *readers*, and what
/// has to survive it is the walk that draws the rows — `Document::nesting_in` and its
/// `without_reader_rings` repair, asserted where that walk lives.
fn assert_invariants(doc: &Document, context: &str) {
    for block in every_block(doc) {
        match ownership_of(doc, block).unwrap_or(Ownership::Free) {
            Ownership::Nested => assert!(
                !readers_in(doc, block).is_empty(),
                "{context}: {block:?} is nested with nothing reading it (I2)",
            ),
            Ownership::Variable => assert!(
                name_of(doc, block).is_some(),
                "{context}: {block:?} is a variable with no name (I4)",
            ),
            Ownership::Free => {}
        }
    }
    let mut again = doc.clone();
    again.pending_ownership_changes.clear();
    again.settle_ownership();
    assert_eq!(
        again.pending_ownership_changes,
        Vec::new(),
        "{context}: the edit left moves unsettled",
    );
}

/// What undo and redo are held to: both ownership maps, and every block's name. The maps are read
/// off the document here rather than through [`OwnershipSnapshot`], so the test does not lean on
/// the type it is testing to say what was there.
struct State {
    ownership: BTreeMap<BlockKey, Ownership>,
    names: BTreeMap<BlockKey, Option<String>>,
}

impl State {
    fn of(doc: &Document) -> Self {
        let nodes = doc.ownership_nodes.iter().map(|(id, o)| (BlockKey::Node(*id), *o));
        let curves = doc.ownership_curves.iter().map(|(id, o)| (BlockKey::Curve(*id), *o));
        Self {
            ownership: nodes.chain(curves).collect(),
            names: every_block(doc).into_iter().map(|b| (b, name_of(doc, b))).collect(),
        }
    }

    /// The document is back at this state: the same blocks, the same ownership maps entry for
    /// entry, and every name exactly as it was.
    ///
    /// **Every name, with no allowance, since 112.** Spec §4.3's last paragraph exempted the name
    /// the application gave a nameless new variable, which no undo took back; settling never
    /// declares now, so no name is written off the history and an undo that left one standing is a
    /// failure rather than the one permitted drift.
    fn assert_at(&self, doc: &Document, context: &str) {
        let now = State::of(doc);
        assert_eq!(
            now.names.keys().collect::<Vec<_>>(),
            self.names.keys().collect::<Vec<_>>(),
            "{context}: not the same blocks",
        );
        assert_eq!(now.ownership, self.ownership, "{context}: ownership not put back exactly");
        assert_eq!(now.names, self.names, "{context}: a name was not put back");
    }
}

/// What the application's gesture is to carry (spec §4.3, refactor plan step 2.14): the graph and
/// both ownership maps from where it started, handed to every entry it builds.
struct Gesture {
    part: PartId,
    before: Graph,
    ownership: OwnershipSnapshot,
    /// The whole document where it started, which the application does not keep: the model holds
    /// each later instant to what one edit from here leaves (refactor plan 2.22).
    start: Document,
}

impl Gesture {
    fn start(doc: &Document, part: PartId) -> Self {
        Self {
            part,
            before: graph(doc, part),
            ownership: OwnershipSnapshot::of(doc),
            start: doc.clone(),
        }
    }
}

/// A graph edit of one of the kinds the application makes, or `None` when the graph has nothing it
/// applies to. Every kind of edge is made: operands, a reference (a mirror aimed at a block), a
/// curve read, a block slid in between (a placement's transform), and a delete.
fn structural_edit(
    rng: &mut Lcg,
    born: &mut Born,
    graph: &mut Graph,
    curves: &[CurveId],
) -> Option<&'static str> {
    let nodes: Vec<NodeId> = born.in_order(graph.nodes().map(|(id, _)| id), BlockKey::Node);
    match rng.below(8) {
        0 => {
            born.node(ball(graph));
            Some("a body added")
        }
        1 => {
            // Repeats allowed: a block named twice by one reader is one reader.
            let inputs: Vec<NodeId> =
                (0..1 + rng.below(3)).filter_map(|_| rng.pick(&nodes)).collect();
            if inputs.is_empty() {
                return None;
            }
            born.node(graph.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: inputs.into_iter().map(Into::into).collect(),
            }));
            Some("a union added")
        }
        2 => {
            let (a, b) = (rng.pick(&nodes)?, rng.pick(&nodes)?);
            let fillet = Fillet { radius: lit(1.0), ..Fillet::hard() };
            born.node(graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet }));
            Some("a blend added")
        }
        3 => {
            let input = rng.pick(&nodes)?;
            born.node(graph.add(Node::Shell { input: input.into(), thickness: lit(1.0) }));
            Some("a shell added")
        }
        4 => {
            let curve = rng.pick(curves)?;
            born.node(graph.add(Node::Pipe { curve, radius: lit(1.0) }));
            Some("a pipe added")
        }
        5 => {
            let (input, at) = (rng.pick(&nodes)?, rng.pick(&nodes)?);
            let frame = Frame::From { node: at };
            born.node(graph.add(Node::Mirror { input: input.into(), frame, join: true }));
            Some("a mirror aimed at a block")
        }
        6 => {
            // How a placement wraps a block: the transform takes its consumers' edges.
            let input = rng.pick(&nodes)?;
            let transform = born.node(graph.add(Node::Transform {
                input: input.into(),
                translation: [lit(1.0), lit(0.0), lit(0.0)],
                rotation: [lit(0.0); 3],
                scale: lit(1.0),
                about: Place::At([lit(0.0); 3]),
            }));
            graph.rewire_consumers(input, transform);
            Some("a transform slid in")
        }
        _ => {
            graph.remove_cascading(rng.pick(&nodes)?);
            Some("a block deleted")
        }
    }
}

/// A number changed and nothing else — the instant of a drag, which continues a gesture.
fn number_edit(rng: &mut Lcg, born: &Born, graph: &mut Graph) -> bool {
    let nodes: Vec<NodeId> = born.in_order(graph.nodes().map(|(id, _)| id), BlockKey::Node);
    let Some(id) = rng.pick(&nodes) else { return false };
    let value = rng.below(100) as f64 / 10.0;
    let node = graph.node_mut(id).expect("picked from the graph");
    match node.param_refs_mut().into_iter().find(|p| matches!(p, ParamRef::Literal(_))) {
        Some(slot) => {
            *slot = lit(value);
            true
        }
        None => false,
    }
}

/// A pipe pointed at another curve and nothing else changed. A curve is not an input, so this is an
/// instant of a drag as much as a number is, and unlike a number it moves who owns what: the
/// console's `sweep profile= spine=` was such an instant (refactor plan 2.22).
fn repoint_edit(rng: &mut Lcg, born: &Born, graph: &mut Graph, curves: &[CurveId]) -> bool {
    let nodes: Vec<NodeId> = born.in_order(graph.nodes().map(|(id, _)| id), BlockKey::Node);
    let pipes: Vec<NodeId> =
        nodes.into_iter().filter(|id| matches!(graph.node(*id), Some(Node::Pipe { .. }))).collect();
    let (Some(id), Some(to)) = (rng.pick(&pipes), rng.pick(curves)) else { return false };
    match graph.node_mut(id) {
        Some(Node::Pipe { curve, .. }) if *curve != to => {
            *curve = to;
            true
        }
        _ => false,
    }
}

/// **Every edit leaves the invariants holding** (spec §12 test 11).
///
/// Random sequences through a real [`CommandStack`]: graph edits of every kind of edge in two
/// parts over curves both read, drags that replace their own entry as the application's gestures
/// do, a paste-like batch, the three ownership acts and a write the readers do not bear out, undo
/// and redo. After each, as the application will after each frame, the changes are drained; then I2
/// and I4 hold and nothing is left for a settle to move. Each undo is held to the state its entry
/// found, and each redo to the one it left — the maps entry for entry — which is the exactness of
/// spec §4.3 over every sequence rather than over three.
///
/// **A variable is asked for here, never wired into being** (`notebook-spec.md` §12, R5). The
/// generator used to make one by giving a block a second reader, since settling promoted it; a
/// second reader makes a shared instance now, so the only thing that writes a `Variable` is the act
/// that asks for one — and the shared instance the wiring makes instead is a case of its own below.
///
/// A drag's next instant is held to two more things, since the application drains between them as
/// this does (refactor plan 2.22): it leaves the maps one edit from where the gesture began leaves,
/// though a curve it re-points moves who owns what; and it reports only what moved since the instant
/// before, so a move made earlier in the gesture is not heard again (spec §6.3).
///
/// The cases each have to come up, so a generator that stopped making one fails here rather than
/// passing by never asking.
#[test]
fn every_edit_leaves_the_invariants_holding() {
    let mut seen: HashMap<&'static str, usize> = HashMap::new();
    for seed in 0..150u64 {
        let mut rng = Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x2545_f491_4f6c_dd1d);
        let (mut doc, parts, mut born) = document(2, 3);
        let mut stack = CommandStack::new(10_000);
        let mut gesture: Option<Gesture> = None;
        // What each entry on the history found and what it left.
        let mut done: Vec<(State, State)> = Vec::new();
        let mut undone: Vec<(State, State)> = Vec::new();

        for step in 0..60 {
            let context = format!("seed {seed}, step {step}");
            let found = State::of(&doc);
            let part = rng.pick(&parts).expect("two parts");
            // The curves there are now: an undone paste takes its curve with it.
            let curves: Vec<CurveId> = born.in_order(doc.curves.keys().copied(), BlockKey::Curve);
            // What one act puts on the history, or `None` for an undo, a redo or nothing done.
            let mut entry: Option<(bool, Box<dyn Command>)> = None;
            let mut asked: Option<(BlockKey, Ownership)> = None;
            // For a drag's next instant: the maps one edit from the gesture's start leaves.
            let mut as_one: Option<BTreeMap<BlockKey, Ownership>> = None;
            match rng.below(20) {
                0..=6 => {
                    let mut after = graph(&doc, part);
                    if let Some(case) = structural_edit(&mut rng, &mut born, &mut after, &curves) {
                        *seen.entry(case).or_default() += 1;
                        let started = Gesture::start(&doc, part);
                        let cmd = SetPartGraph::with_ownership(
                            part,
                            started.before.clone(),
                            after,
                            Some(started.ownership.clone()),
                        );
                        entry = Some((false, Box::new(cmd)));
                        gesture = Some(started);
                    }
                }
                7..=9 => {
                    let mut after = graph(&doc, part);
                    let repointed =
                        rng.percent(30) && repoint_edit(&mut rng, &born, &mut after, &curves);
                    if repointed || number_edit(&mut rng, &born, &mut after) {
                        // A drag's next instant replaces the entry it started, with that entry's
                        // before-graph and maps; anything else starts a gesture of its own.
                        match gesture.as_ref().filter(|g| g.part == part && rng.percent(70)) {
                            Some(g) => {
                                *seen.entry("a drag replaced its own entry").or_default() += 1;
                                if repointed {
                                    *seen.entry("a drag re-pointed a curve").or_default() += 1;
                                }
                                let mut one = g.start.clone();
                                SetPartGraph::new(part, g.before.clone(), after.clone())
                                    .execute(&mut one)
                                    .unwrap_or_else(|e| panic!("{context}: {e}"));
                                as_one = Some(State::of(&one).ownership);
                                let cmd = SetPartGraph::with_ownership(
                                    part,
                                    g.before.clone(),
                                    after,
                                    Some(g.ownership.clone()),
                                );
                                entry = Some((true, Box::new(cmd)));
                            }
                            None => {
                                let started = Gesture::start(&doc, part);
                                let cmd = SetPartGraph::with_ownership(
                                    part,
                                    started.before.clone(),
                                    after,
                                    Some(started.ownership.clone()),
                                );
                                entry = Some((false, Box::new(cmd)));
                                gesture = Some(started);
                            }
                        }
                    }
                }
                10..=13 => {
                    gesture = None;
                    let blocks = born.in_order(every_block(&doc), |b| b);
                    // Half the time the act is aimed at a block two or more blocks read, where the
                    // rules 112 reversed are: making one a variable takes it out of every reader at
                    // once, and putting it back is the case that used to be refused (§6.6). Left to
                    // an even pick those blocks are a minority of a growing part, and the cases
                    // below would be reached by luck.
                    let shared: Vec<BlockKey> =
                        blocks.iter().copied().filter(|b| readers_in(&doc, *b).len() > 1).collect();
                    let picked = match rng.percent(50) && !shared.is_empty() {
                        true => rng.pick(&shared),
                        false => rng.pick(&blocks),
                    };
                    if let Some(block) = picked {
                        let was = ownership_of(&doc, block).unwrap_or(Ownership::Free);
                        let name = name_of(&doc, block);
                        let readers = readers_in(&doc, block);
                        match rng.below(4) {
                            // What the command's own settle is for: a write its readers do not bear
                            // out — a stale put-back, say — goes where they put it instead.
                            //
                            // **"Told to sit inside a block that does not read it" was how this
                            // case used to be written**, and a nested block names no block at all
                            // now, so the write the readers contradict is instead nested with
                            // nothing reading it, or free with something reading it.
                            3 => {
                                *seen.entry("told where its readers do not put it").or_default() +=
                                    1;
                                let (told, settled) = match readers.is_empty() {
                                    true => (Ownership::Nested, Ownership::Free),
                                    false => (Ownership::Free, Ownership::Nested),
                                };
                                asked = Some((block, settled));
                                let cmd = SetOwnership::new(block, told, name.as_deref());
                                entry = Some((false, Box::new(cmd)));
                            }
                            0 if was != Ownership::Variable => {
                                *seen.entry("made a variable").or_default() += 1;
                                let name = name.unwrap_or_else(|| format!("Var {step}"));
                                let variable = Ownership::Variable;
                                asked = Some((block, variable));
                                let cmd = SetOwnership::new(block, variable, Some(&name));
                                entry = Some((false, Box::new(cmd)));
                            }
                            1 if was == Ownership::Variable => {
                                // **Never refused for reader count** (`notebook-spec.md` §6.6):
                                // two readers used to refuse, there being no one block to be
                                // inside, and the block is simply drawn in both of them now — a
                                // shared instance (§7.3). With no readers it is a top-level row.
                                let target = doc.put_back_target(block);
                                *seen.entry("put back").or_default() += 1;
                                if readers.len() > 1 {
                                    *seen.entry("put back into every reader").or_default() += 1;
                                }
                                asked = Some((block, target));
                                let cmd = SetOwnership::new(block, target, name.as_deref());
                                entry = Some((false, Box::new(cmd)));
                            }
                            _ => {
                                // A variable's name is never cleared: the application refuses that
                                // where it is typed (spec §5.3).
                                let clear = was != Ownership::Variable && rng.percent(40);
                                let renamed = (!clear).then(|| format!("Name {step}"));
                                *seen.entry("renamed").or_default() += 1;
                                asked = Some((block, was));
                                let cmd = SetOwnership::new(block, was, renamed.as_deref());
                                entry = Some((false, Box::new(cmd)));
                            }
                        }
                    }
                }
                14..=15 => {
                    // A paste: its curve first, then the block reading it, and sometimes the curve
                    // made a variable in the same act — a member acting on what the graph edit
                    // before it settled.
                    gesture = None;
                    *seen.entry("a batch").or_default() += 1;
                    let curve = line(&format!("Pasted {step}"));
                    born.curve(curve.id);
                    let mut after = graph(&doc, part);
                    born.node(after.add(Node::Pipe { curve: curve.id, radius: lit(2.0) }));
                    let mut members: Vec<Box<dyn Command>> = vec![
                        Box::new(SetCurve::insert(curve.clone())),
                        Box::new(SetPartGraph::new(part, graph(&doc, part), after)),
                    ];
                    if rng.percent(50) {
                        asked = Some((BlockKey::Curve(curve.id), Ownership::Variable));
                        members.push(Box::new(SetOwnership::new(
                            BlockKey::Curve(curve.id),
                            Ownership::Variable,
                            Some("Path"),
                        )));
                    }
                    entry = Some((false, Box::new(Batch::new("edit.paste", members))));
                }
                16..=18 => {
                    gesture = None;
                    match stack.undo(&mut doc) {
                        Some(_) => {
                            let (was, left) = done.pop().expect("the model has the entry");
                            drain(&mut doc);
                            was.assert_at(&doc, &format!("{context}, after undo"));
                            *seen.entry("undo").or_default() += 1;
                            // The case a settle on undo could not get right: the entry made a
                            // variable of a block that was nested before it, and a settle leaves a
                            // variable where it is, so only the snapshot can nest it again.
                            let declared_undone = left.ownership.iter().any(|(block, now)| {
                                let before = was.ownership.get(block);
                                *now == Ownership::Variable
                                    && matches!(before, Some(Ownership::Nested))
                            });
                            if declared_undone {
                                *seen.entry("a variable put back by undo").or_default() += 1;
                            }
                            undone.push((was, left));
                        }
                        None => assert!(done.is_empty(), "{context}: there was an entry to undo"),
                    }
                }
                _ => {
                    gesture = None;
                    match stack.redo(&mut doc) {
                        Some(_) => {
                            let (was, left) = undone.pop().expect("the model has the entry");
                            drain(&mut doc);
                            left.assert_at(&doc, &format!("{context}, after redo"));
                            *seen.entry("redo").or_default() += 1;
                            done.push((was, left));
                        }
                        None => assert!(undone.is_empty(), "{context}: there was an entry to redo"),
                    }
                }
            }

            if let Some((replaces, cmd)) = entry {
                let result = match replaces {
                    true => stack.replace_top(&mut doc, cmd),
                    false => stack.execute(&mut doc, cmd),
                };
                result.unwrap_or_else(|e| panic!("{context}: {e}"));
                let changes = drain(&mut doc);
                if let Some(wanted) = as_one {
                    assert_eq!(
                        State::of(&doc).ownership,
                        wanted,
                        "{context}: a drag's next instant left maps one edit from its start would not",
                    );
                    // `found` is what the last drain heard: the application's previous frame.
                    for change in &changes {
                        let was = found.ownership.get(&change.block()).copied();
                        assert_ne!(
                            ownership_of(&doc, change.block()).unwrap_or(Ownership::Free),
                            was.unwrap_or(Ownership::Free),
                            "{context}: a drag's next instant reports {change:?}, which it did not move",
                        );
                    }
                }
                // **Two of the four kinds of change went with the stored parent.** `Rehomed`
                // reported a nested block's one parent being replaced, which nothing can be now;
                // `BecameVariable` reported the promotion a second reader used to cause, and
                // settling declares nothing (`notebook-spec.md` §12, R5). What the wiring those
                // two reported makes instead is counted below, off the document rather than off a
                // change, because it is a state and no longer news.
                for change in changes {
                    let case = match change {
                        OwnershipChange::Adopted { .. } => "adopted",
                        OwnershipChange::Released { .. } => "released",
                    };
                    *seen.entry(case).or_default() += 1;
                }
                if let Some((block, wanted)) = asked {
                    assert_eq!(
                        ownership_of(&doc, block),
                        Some(wanted),
                        "{context}: {block:?} is not where it was asked to go",
                    );
                }
                undone.clear();
                // A replaced entry keeps what the gesture found when it started.
                let found = match replaces {
                    true => done.pop().expect("an entry to replace").0,
                    false => found,
                };
                done.push((found, State::of(&doc)));
            }
            assert_invariants(&doc, &context);

            // The two states the reversed rule leaves, which the invariants above hold to the new
            // rule wherever they are reached: a block two blocks read is nested, in both of them
            // (§7.3), and a curve two *parts* read is nested too, drawn inside the reader in each
            // of their lists. Both were variables under the old table, named and announced.
            if every_block(&doc).into_iter().any(|b| {
                ownership_of(&doc, b) == Some(Ownership::Nested) && readers_in(&doc, b).len() > 1
            }) {
                *seen.entry("a shared instance, nested in every reader").or_default() += 1;
            }
            if doc.curves.keys().any(|id| {
                ownership_of(&doc, BlockKey::Curve(*id)) == Some(Ownership::Nested)
                    && doc
                        .parts
                        .values()
                        .filter(|p| p.graph.nodes().any(|(_, n)| n.curves().contains(id)))
                        .count()
                        > 1
            }) {
                *seen.entry("a curve two parts read, nested in each").or_default() += 1;
            }
        }
    }

    for case in [
        "a body added",
        "a union added",
        "a blend added",
        "a shell added",
        "a pipe added",
        "a mirror aimed at a block",
        "a transform slid in",
        "a block deleted",
        "a drag replaced its own entry",
        "a drag re-pointed a curve",
        "made a variable",
        "put back",
        "put back into every reader",
        "renamed",
        "told where its readers do not put it",
        "a batch",
        "undo",
        "redo",
        "adopted",
        "released",
        "a shared instance, nested in every reader",
        "a curve two parts read, nested in each",
        "a variable put back by undo",
    ] {
        let count = seen.get(case).copied().unwrap_or(0);
        assert!(count >= 10, "only {count} of \"{case}\" in every seed: {seen:?}");
    }
}

/// **Undo puts ownership back exactly** (spec §12 test 12), in the three cases of spec §4.3, each
/// of which a design that settled again on undo would get wrong or right only by accident.
#[test]
fn undo_restores_ownership_exactly() {
    // A nested block the person makes a variable, and undo puts it back inside — which no settle
    // could do, since a settle leaves a variable where it is.
    //
    // **A second reader is what used to take it out of the block it was in**, by promoting it; it
    // makes a shared instance now (`notebook-spec.md` §12, §7.3), so it is asserted here as the
    // thing that moves nothing, and only the person's act takes the block out of its readers.
    let (mut doc, parts, _) = document(1, 0);
    let part = parts[0];
    let mut stack = CommandStack::new(64);
    let mut g = graph(&doc, part);
    let nested = ball(&mut g);
    let shell = g.add(Node::Shell { input: nested.into(), thickness: lit(1.0) });
    edit(&mut doc, &mut stack, part, g);
    assert_eq!(doc.ownership_nodes.get(&nested), Some(&Ownership::Nested));
    let alone = OwnershipSnapshot::of(&doc);

    let mut g = graph(&doc, part);
    let union = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![nested.into()] });
    edit(&mut doc, &mut stack, part, g);
    assert_eq!(
        doc.ownership_nodes.get(&nested),
        Some(&Ownership::Nested),
        "a second reader makes a shared instance, not a variable",
    );
    let mut readers = vec![shell, union];
    readers.sort();
    assert_eq!(doc.block_readers(BlockKey::Node(nested)), readers, "drawn inside both of them");
    let was = OwnershipSnapshot::of(&doc);

    let out = SetOwnership::new(BlockKey::Node(nested), Ownership::Variable, Some("Ball"));
    stack.execute(&mut doc, Box::new(out)).expect("makes it a variable");
    assert_eq!(doc.ownership_nodes.get(&nested), Some(&Ownership::Variable), "the person asked");
    stack.undo(&mut doc).expect("undoes the variable");
    assert_eq!(doc.ownership_nodes.get(&nested), Some(&Ownership::Nested));
    assert_eq!(OwnershipSnapshot::of(&doc), was, "both maps, entry for entry");
    stack.redo(&mut doc).expect("redoes it");
    assert_eq!(doc.ownership_nodes.get(&nested), Some(&Ownership::Variable), "a variable again");
    stack.undo(&mut doc).expect("undoes the variable again");
    stack.undo(&mut doc).expect("undoes the second reader");
    assert_eq!(OwnershipSnapshot::of(&doc), alone, "and the second reader leaves no entry behind");

    // A block deleted and undeleted comes back with the ownership it had, not one worked out again:
    // a variable the person made of a block only one other reads, which that reader would adopt.
    // And deleting the block holding things releases them, and undo puts them back inside it.
    let (mut doc, parts, _) = document(1, 0);
    let part = parts[0];
    let mut stack = CommandStack::new(64);
    let mut g = graph(&doc, part);
    let (inner, hub) = (ball(&mut g), ball(&mut g));
    let union = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![inner.into(), hub.into()] });
    edit(&mut doc, &mut stack, part, g);
    stack
        .execute(
            &mut doc,
            Box::new(SetOwnership::new(BlockKey::Node(hub), Ownership::Variable, Some("Hub"))),
        )
        .expect("makes the hub a variable");
    let was = OwnershipSnapshot::of(&doc);
    assert_eq!(was, {
        let mut expected = Document::new();
        expected.ownership_nodes = HashMap::from([
            (inner, Ownership::Nested),
            (hub, Ownership::Variable),
            (union, Ownership::Free),
        ]);
        OwnershipSnapshot::of(&expected)
    });

    let mut g = graph(&doc, part);
    g.remove_cascading(union);
    edit(&mut doc, &mut stack, part, g);
    assert_eq!(
        doc.ownership_nodes.get(&inner),
        Some(&Ownership::Free),
        "released with the block that read it",
    );
    assert_eq!(doc.ownership_nodes.get(&hub), Some(&Ownership::Variable), "a variable stays one");
    stack.undo(&mut doc).expect("undoes the delete");
    assert_eq!(OwnershipSnapshot::of(&doc), was, "back inside the union it was in");

    let mut g = graph(&doc, part);
    g.remove_cascading(hub);
    edit(&mut doc, &mut stack, part, g);
    stack.undo(&mut doc).expect("undoes the delete");
    assert_eq!(doc.ownership_nodes.get(&hub), Some(&Ownership::Variable));
    assert_eq!(
        doc.put_back_target(BlockKey::Node(hub)),
        Ownership::Nested,
        "worked out again it would be nested in the block that reads it, so this was remembered",
    );
    assert_eq!(OwnershipSnapshot::of(&doc), was);

    // Delete a pattern and undo: its source is back inside it, and drawn. The delete is the
    // application's — the graph edit, and in the same entry the clear of the `shown` flag the
    // pattern had set, now that nothing reads the source (refactor plan §3.3, cause 9). Settling
    // writes no visibility, so the flag comes back by that clear's own undo and nothing else.
    let (mut doc, parts, _) = document(1, 0);
    let part = parts[0];
    let mut stack = CommandStack::new(64);
    let mut g = graph(&doc, part);
    let source = ball(&mut g);
    let pattern = g.add(Node::Pattern {
        options: Default::default(),
        input: source.into(),
        count: lit(4.0),
        kind: PatternKind::Linear { step: [lit(30.0), lit(0.0), lit(0.0)] },
        join: false,
        fillet: Fillet::hard(),
    });
    edit(&mut doc, &mut stack, part, g);
    // As Pattern sets it, off the history.
    doc.node_meta.entry(source).or_default().shown = true;
    let was = OwnershipSnapshot::of(&doc);
    assert_eq!(doc.ownership_nodes.get(&source), Some(&Ownership::Nested));

    let mut g = graph(&doc, part);
    g.remove_cascading(pattern);
    let delete: Vec<Box<dyn Command>> = vec![
        Box::new(SetPartGraph::new(part, graph(&doc, part), g)),
        Box::new(SetShown::new(source, false)),
    ];
    stack.execute(&mut doc, Box::new(Batch::new("part.set_graph", delete))).expect("deletes");
    assert_eq!(doc.ownership_nodes.get(&source), Some(&Ownership::Free), "a body again");
    stack.undo(&mut doc).expect("undoes the delete");
    assert_eq!(doc.ownership_nodes.get(&source), Some(&Ownership::Nested));
    assert_eq!(OwnershipSnapshot::of(&doc), was);
    assert!(
        doc.node_meta.get(&source).is_some_and(|m| m.shown)
            && doc.parts[&part].graph.node(pattern).is_some_and(|n| n.inputs() == vec![source]),
        "consumed by the pattern again and shown, so drawn beside it as before",
    );
}

/// **Undoing a gesture that followed a new block leaves no orphaned ownership** (spec §4.3).
///
/// Add a blend, then drag its radius: the drag replaces the blend's entry with one whose
/// before-graph has no blend. Maps taken at that replacement would hold the blend's operands
/// nested, and the undo would put back a graph without the blend beside maps that have two blocks
/// nested with nothing reading them — I2 broken, and two rows drawn inside nothing. Maps taken
/// where the gesture started are the ones that go with its before-graph.
///
/// The second arm is the control: the same gesture with each replacement taking its own maps, as
/// a `SetPartGraph` given none does. It must orphan the operands, or this test is not reaching the
/// case.
#[test]
fn undoing_a_gesture_that_followed_a_new_block_leaves_no_orphaned_ownership() {
    for carried in [true, false] {
        let (mut doc, parts, _) = document(1, 0);
        let part = parts[0];
        let mut stack = CommandStack::new(64);
        let mut g = graph(&doc, part);
        let (a, b) = (ball(&mut g), ball(&mut g));
        edit(&mut doc, &mut stack, part, g);
        let started = graph(&doc, part);
        let ownership = OwnershipSnapshot::of(&doc);
        let given = || carried.then(|| ownership.clone());

        let mut g = started.clone();
        let fillet = Fillet { radius: lit(0.5), ..Fillet::hard() };
        let blend = g.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet });
        let first = SetPartGraph::with_ownership(part, started.clone(), g.clone(), given());
        stack.execute(&mut doc, Box::new(first)).expect("adds the blend");
        assert_eq!(doc.ownership_nodes.get(&a), Some(&Ownership::Nested), "adopted");
        for radius in [1.0, 2.0, 3.0] {
            if let Some(Node::Blend { fillet, .. }) = g.node_mut(blend) {
                fillet.radius = lit(radius);
            }
            let instant = SetPartGraph::with_ownership(part, started.clone(), g.clone(), given());
            stack.replace_top(&mut doc, Box::new(instant)).expect("drags the radius");
        }

        stack.undo(&mut doc).expect("undoes the gesture");
        assert!(doc.parts[&part].graph.node(blend).is_none(), "the blend is gone");
        // **Orphaned is nested with nothing reading it, since no block names its parent.** The
        // stale entry is the same one it always was — the operands left inside a blend the undo
        // took away — and what says so now is the graph the maps came back beside.
        let graph = &doc.parts[&part].graph;
        let orphaned: Vec<NodeId> = doc
            .ownership_nodes
            .iter()
            .filter(|(id, o)| {
                graph.node(**id).is_some()
                    && **o == Ownership::Nested
                    && readers_in(&doc, BlockKey::Node(**id)).is_empty()
            })
            .map(|(id, _)| *id)
            .collect();
        if carried {
            assert_eq!(orphaned, Vec::new(), "no block is nested with nothing reading it");
            assert_eq!(OwnershipSnapshot::of(&doc), ownership, "the maps the gesture started from");
            assert_invariants(&doc, "after undoing the gesture");
        } else {
            let mut both = vec![a, b];
            both.sort();
            let mut orphaned = orphaned;
            orphaned.sort();
            assert_eq!(orphaned, both, "maps taken at each replacement orphan the operands");
        }
    }
}

/// **The curve an opened loop's extrude names comes out by its V and goes back** — backlog item
/// 73's reported case, through the commands the V makes (spec §10, refactor plan 2.19).
///
/// An extrude names one curve of its loop and walks the rest, so settling adopts that curve and no
/// other: loop members are not adopted (§10). While the loop closes its profile row hides that;
/// break one joint and the loop opens, the three curves the extrude does not name stand on their
/// own, and the one it names stays inside it — "That one is now trapped inside extrude. And I
/// cannot release it." Make variable takes it out though the extrude still reads it, and put back
/// inside returns it there: each one undo, the curve's own name kept throughout. Asked of
/// `nesting_in`, which is what the list is drawn from.
#[test]
fn the_curve_an_opened_loop_s_extrude_names_comes_out_by_its_v_and_goes_back() {
    let (mut doc, parts, _) = document(1, 0);
    let part = parts[0];
    let mut stack = CommandStack::new(64);
    let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
    let corners = [at(0.0, 0.0), at(10.0, 0.0), at(10.0, 10.0), at(0.0, 10.0)];
    let square: Vec<CurveId> = (0..4)
        .map(|i| {
            let ends = vec![corners[i], corners[(i + 1) % 4]];
            let side = Curve::polyline(CurveId::new(), format!("Line {}", i + 1), ends, false)
                .expect("a side");
            let id = side.id;
            stack.execute(&mut doc, Box::new(SetCurve::insert(side))).expect("adds a side");
            id
        })
        .collect();
    let mut g = graph(&doc, part);
    let extrude = g.add(Node::Extrude { rims: Default::default(),
        outline: Outline::closed(square[0]),
        sweep: Sweep::Normal,
        height: lit(10.0),
        start: lit(0.0),
        sides: Sides::One,
        draft: lit(0.0),
        corners: Default::default(),
    });
    edit(&mut doc, &mut stack, part, g);
    // The joint the second and third sides meet at, broken: the second's end pulled off it.
    let side = doc.curves[&square[1]].clone();
    let mut pulled = side.clone();
    pulled.set_control_point(1, at(10.0, 14.0));
    stack
        .execute(&mut doc, Box::new(SetCurve::new(square[1], Some(side), Some(pulled))))
        .expect("breaks the joint");

    // Every block each row is drawn inside, since a nested row is drawn in every block that reads
    // it (`notebook-spec.md` R1) and `curve_owners_of` is where the single `owner_of_curve` went.
    // One reader here either way, so each list holds one id or none.
    let rows = |doc: &Document| {
        let nesting = doc.nesting_in(&doc.parts[&part].graph);
        square.iter().map(|c| nesting.curve_owners_of(*c).to_vec()).collect::<Vec<_>>()
    };
    let trapped = vec![vec![extrude], Vec::new(), Vec::new(), Vec::new()];
    let on_their_own = vec![Vec::<NodeId>::new(); 4];
    assert_eq!(rows(&doc), trapped, "the named one inside the extrude, the rest on their own");

    let key = BlockKey::Curve(square[0]);
    let name = doc.curves[&square[0]].name.clone();
    let out = SetOwnership::new(key, Ownership::Variable, Some(&name));
    stack.execute(&mut doc, Box::new(out)).expect("makes it a variable");
    assert_eq!(rows(&doc), on_their_own, "out of the extrude, a row of its own");
    assert_eq!(
        doc.parts[&part].graph.node(extrude).map(Node::curves),
        Some(vec![square[0]]),
        "and the extrude still reads it: where a row is drawn is not an edge",
    );
    stack.undo(&mut doc).expect("one undo");
    assert_eq!(rows(&doc), trapped, "undone, inside again");
    stack.redo(&mut doc).expect("one redo");
    assert_eq!(rows(&doc), on_their_own, "redone, out again");

    // Never refused for reader count (`notebook-spec.md` §6.6); the extrude is the one block
    // reading it, so back inside is back inside the extrude.
    let back = doc.put_back_target(key);
    assert_eq!(back, Ownership::Nested);
    assert_eq!(doc.block_readers(key), vec![extrude], "and the extrude is who reads it");
    let back = SetOwnership::new(key, back, Some(&name));
    stack.execute(&mut doc, Box::new(back)).expect("puts it back");
    assert_eq!(rows(&doc), trapped, "back inside the extrude");
    assert_eq!(doc.curves[&square[0]].name, name, "with its own name");
    stack.undo(&mut doc).expect("one undo");
    assert_eq!(rows(&doc), on_their_own, "undone, a variable again");
}

/// **The entries a gesture builds share its ownership maps rather than copying them** (spec §4.3).
///
/// Every entry a drag builds is handed the snapshot the gesture took at its start, and each entry
/// has to own one, since `replace_top` drops the entry before it and its undo goes with it. So
/// every frame clones the snapshot, and the clone has to share the maps rather than copy them, or
/// building the entry copies them as well as the entry putting them in the document before its
/// settle (`OwnershipSnapshot::exchange`). Taking a snapshot of the document is the arm that must
/// copy, so sharing is not being reported of everything.
#[test]
fn the_entries_a_gesture_builds_share_its_ownership_maps_rather_than_copy_them() {
    let (mut doc, parts, _) = document(1, 2);
    let part = parts[0];
    let mut stack = CommandStack::new(64);
    let mut g = graph(&doc, part);
    let input = ball(&mut g);
    g.add(Node::Shell { input: input.into(), thickness: lit(1.0) });
    edit(&mut doc, &mut stack, part, g);

    let started = OwnershipSnapshot::of(&doc);
    for _ in 0..3 {
        let frame = started.clone();
        assert_eq!(frame, started);
        assert!(frame.shares_maps_with(&started), "a frame holds the gesture's maps, not a copy");
    }
    assert!(!OwnershipSnapshot::of(&doc).shares_maps_with(&started), "taking one copies the maps");
}
