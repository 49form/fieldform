// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a block sits in the list, stored rather than read off its name and its readers —
//! `block-ownership-spec.md`, backlog item 82.
//!
//! The list used to decide every row again on every frame from facts the document holds for other
//! reasons (`Document::nesting_in`): whether a block has a name, and how many blocks read it. A
//! rule recomputed from facts other features also write moves rows whenever any of them is written,
//! for whatever reason, and every block-list defect recorded since nesting was built is an instance
//! of that (spec §1). So the answer is stored, one per block, and only two things change it: the
//! person asking, and an edit to who reads the block, by the fixed rules of
//! [`Document::settle_ownership`].
//!
//! **The list reads it** (`nesting_in`), now that every path that writes a graph settles: read
//! before that, the first edit that skipped a settle would have flattened every row — the order
//! `refactor-plan.md` §3.5 lays down. Nodes first, then curves and profile rows once their own V
//! wrote the curve ownership (refactor plan 2.19).

use crate::{CurveId, Document, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Where one block is drawn in the list, and so whether it belongs to another (spec §2.1).
///
/// Read as slots, every input of every block is a value held in the node, a block drawn in the
/// slot — `Nested` — or a reference to anything else, in practice a `Variable` shown as a chip.
/// This is an annotation on the child: nothing about `Graph`, `Node` or `inputs()` changes with
/// it, which is the per-node-annotation-not-a-stored-list rule `body-order-decision.md` §3 settled.
///
/// **No parent is stored, since 112.** `Inside(p)` named the one block a nested block belonged to;
/// a nested block is now drawn inside *every* block that reads it (`notebook-spec.md` §12, R1), so
/// there is no one parent to name and the readers are the graph's to answer. What stays stored is
/// the one bit no reader implies: nested, top-level, or declared.
///
/// Written `"free"`, `"variable"` and `"nested"`; `{"inside": "<id>"}` is still read, which is
/// format 3.15's spelling (spec §9.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ownership {
    /// Drawn inside every block that reads it, and the same block in each of them — a shared
    /// instance (spec §7.3). A `Nested` block has at least one reader, which is the invariant
    /// `Inside(p)`'s "the readers are exactly `{p}`" became (spec §12).
    Nested,
    /// A row of its own that nothing has claimed — a body, or a block about to be consumed. The
    /// first block to read it adopts it.
    Free,
    /// A row of its own, declared reusable. The blocks reading it reference it and none of them
    /// owns it, so it is never adopted: the whole difference from [`Ownership::Free`].
    Variable,
}

/// Read by hand, because a derive cannot keep the promise the doc above makes.
///
/// `Serialize` is still derived and writes `"nested"`, `"free"`, `"variable"`. Reading has to
/// accept format 3.15's `{"inside": "<id>"}` as well, and a derive over three unit variants
/// answers `unknown variant \`inside\`` to it — which is not a block placed oddly but a **file
/// that will not open**: the entry is read inside a `deny_unknown_fields` struct with `?`, so one
/// such block refuses the whole document, and every 3.15 file a shipped build wrote holds one for
/// every block it has.
///
/// The id is read and thrown away. That is the whole of the migration: a nested block names no
/// parent since 112, and the readers say which blocks draw it.
impl<'de> Deserialize<'de> for Ownership {
    fn deserialize<D: serde::Deserializer<'de>>(reader: D) -> Result<Self, D::Error> {
        const WORDS: &[&str] = &["nested", "free", "variable"];

        struct Spelling;

        impl<'de> serde::de::Visitor<'de> for Spelling {
            type Value = Ownership;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("\"nested\", \"free\", \"variable\", or 3.15's {\"inside\": \"<id>\"}")
            }

            fn visit_str<E: serde::de::Error>(self, word: &str) -> Result<Ownership, E> {
                match word {
                    "nested" => Ok(Ownership::Nested),
                    "free" => Ok(Ownership::Free),
                    "variable" => Ok(Ownership::Variable),
                    other => Err(E::unknown_variant(other, WORDS)),
                }
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut entry: M,
            ) -> Result<Ownership, M::Error> {
                let Some(key) = entry.next_key::<String>()? else {
                    return Err(serde::de::Error::custom("an ownership entry with nothing in it"));
                };
                if key != "inside" {
                    return Err(serde::de::Error::unknown_variant(&key, WORDS));
                }
                // Checked for shape and discarded: reading it as `IgnoredAny` would accept a
                // nested object where 3.15 wrote an id, and a file that is wrong about the
                // parent is a file worth refusing while anyone still has one.
                entry.next_value::<String>()?;
                Ok(Ownership::Nested)
            }
        }

        reader.deserialize_any(Spelling)
    }
}

/// A block that has an ownership: a node of some part's graph, or a curve of the document.
///
/// Two kinds because the two live in two places and are read from two places (spec I2): a node is
/// read by blocks in its own part, a curve by blocks in every part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlockKey {
    Node(NodeId),
    Curve(CurveId),
}

/// One block [`Document::settle_ownership`] moved, for the application to act on (spec §4.2).
///
/// Settling writes no name and no visibility, so what follows from a move — a line in the status
/// bar — is the application's, and this is how it finds out. Each says the block and who reads it
/// now, which is what the line says: "Sphere is now inside Boolean and Shell".
///
/// **`Rehomed` went with 112 and stays gone.** It reported a nested block's one parent being
/// replaced by another; no parent is stored now, so sliding a transform under a block is `Nested`
/// to `Nested` and there is nothing to report.
///
/// **`BecameVariable` is back, 2026-10-24, with the rule it reports** (R1 inverted by the owner —
/// see [`settle_one`]). A block a second block starts reading is declared, so the promotion is
/// news again, and so is the name the application invents for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OwnershipChange {
    /// A top-level block is read now, so it is drawn inside what reads it. Every reader,
    /// ascending — which for an adoption is exactly one, two being a variable now.
    Adopted { block: BlockKey, readers: Vec<NodeId> },
    /// Nothing reads it any more, so it is a row of its own again.
    Released { block: BlockKey },
    /// Two or more blocks read it, so it is declared and referenced by name. Every reader,
    /// ascending, for the line that says who uses it.
    BecameVariable { block: BlockKey, readers: Vec<NodeId> },
}

impl OwnershipChange {
    /// The block that moved.
    pub fn block(&self) -> BlockKey {
        match self {
            OwnershipChange::Adopted { block, .. }
            | OwnershipChange::Released { block, .. }
            | OwnershipChange::BecameVariable { block, .. } => *block,
        }
    }
}

/// Every block's distinct readers, each list ascending, counted the way spec I2 counts them.
#[derive(Default)]
struct Readers {
    nodes: HashMap<NodeId, Vec<NodeId>>,
    curves: HashMap<CurveId, Vec<NodeId>>,
}

impl Readers {
    fn of(&self, block: BlockKey) -> &[NodeId] {
        let list = match block {
            BlockKey::Node(id) => self.nodes.get(&id),
            BlockKey::Curve(id) => self.curves.get(&id),
        };
        list.map(Vec::as_slice).unwrap_or(&[])
    }
}

/// Spec §4's table: what a block stored as `stored` becomes when `readers` read it.
///
/// No entry reads as free, which is what a block made by a path that has not settled yet is (spec
/// §3); from there the free rows apply.
///
/// **A second reader declares the block, ruled by the owner 2026-10-24**, which inverts R1 and
/// §7.3 and restores what this did before 112. The reasoning is the containment the notebook is
/// being built toward: a slot *holds* the block it reads, drawn inside the field, so a block can
/// be in one slot and no more. Two readers cannot each contain it, and the honest way to be read
/// twice is the one the reference model uses — declared once, referenced by name in each reader.
/// R1's shared instance only worked while a nested block was a row in a flat list that could be
/// cloned per owner; put the block in the field and the same expandable block is drawn twice,
/// which is the redundancy the owner objected to.
///
/// So the stored entry is a record of a *decision* where a person made it, and of a *wiring* where
/// two readers force it. 112's argument for the other way — that `Variable` should be written by
/// nothing but the person asking — is the thing being given up, deliberately: the promotion is
/// announced and the block is named, so nothing happens silently.
///
/// What has not changed: a stored `Variable` stays one however many read it, and a block nothing
/// reads is free.
fn settle_one(stored: Option<Ownership>, readers: &[NodeId]) -> Ownership {
    match (stored.unwrap_or(Ownership::Free), readers) {
        (Ownership::Variable, _) => Ownership::Variable,
        (_, []) => Ownership::Free,
        (_, [_]) => Ownership::Nested,
        // Two or more: declared, because one slot cannot hold it twice.
        (_, _) => Ownership::Variable,
    }
}

/// The move from `was` to `now` as the application hears of it, or nothing when there is nothing to
/// say: the first entry a block is given, free, is a placement and not a move.
fn change(
    block: BlockKey,
    was: Option<Ownership>,
    now: Ownership,
    readers: &[NodeId],
) -> Option<OwnershipChange> {
    match (was.unwrap_or(Ownership::Free), now) {
        (was, now) if was == now => None,
        (Ownership::Nested, Ownership::Free) => Some(OwnershipChange::Released { block }),
        (_, Ownership::Nested) => {
            Some(OwnershipChange::Adopted { block, readers: readers.to_vec() })
        }
        // Settling writes a variable again (R1 inverted, 2026-10-24), and that is the one move
        // worth announcing: the block leaves the slot it was drawn in for a row of its own, which
        // nobody asked for in so many words.
        (_, Ownership::Variable) => {
            Some(OwnershipChange::BecameVariable { block, readers: readers.to_vec() })
        }
        // Free to free was the first arm; what is left is a release of something that was not
        // nested, which is not a move anybody sees.
        (_, Ownership::Free) => None,
    }
}

impl Document {
    /// Where every block sat under format 3.14 and earlier (spec §9.2) — nodes, then curves.
    ///
    /// The rule `nesting_in` laid a list out by, stated as an ownership:
    ///
    /// - a node is a variable if it is named, **or** if it is unnamed and two or more blocks read
    ///   it; inside its reader if unnamed with exactly one; free otherwise;
    /// - a curve is a variable if it has a profile name or was declared (`declared_curves`), or
    ///   if two or more blocks read it; inside its reader with exactly one; free otherwise.
    ///
    /// **The two-reader clauses answer variable, not free, and 112 makes that the only answer
    /// that keeps the promise.** It used to be the answer because free would have let the next
    /// settle promote every one of them and announce it, opening an old file as a burst of "is now
    /// a variable" about blocks that had not moved. Settling never promotes now, so that reason is
    /// gone — and the row it was protecting is the thing at stake instead: the old list drew a
    /// block two blocks read as **a row of its own**, and `notebook-spec.md` §15 promises an old
    /// file opens drawing the same things. Nested would redraw it inside both readers, so variable
    /// is what preserves the row. A new document makes a shared instance there, and *put back
    /// inside* is how one of these becomes one.
    ///
    /// **A curve's readers are counted in every part**, where the old rule counted the part being
    /// listed. That is the one row the migration moves, on purpose: a curve read once in each of
    /// two parts was drawn inside a block in each, as though each owned it, and is a variable now
    /// (ruled §14 Q5).
    ///
    /// Every block in one walk, rather than a block at a time, because a file from before this is
    /// placed whole on opening and a walk per block would be one per block of the document.
    pub fn legacy_ownership(&self) -> (HashMap<NodeId, Ownership>, HashMap<CurveId, Ownership>) {
        let readers = self.readers();
        let legacy = |declared: bool, block: BlockKey| match (declared, readers.of(block)) {
            (true, _) => Ownership::Variable,
            (false, []) => Ownership::Free,
            (false, [_only]) => Ownership::Nested,
            (false, _) => Ownership::Variable,
        };
        let nodes = self
            .node_ids()
            .into_iter()
            .map(|id| {
                let named = self.node_meta.get(&id).is_some_and(|m| m.name.is_some());
                (id, legacy(named, BlockKey::Node(id)))
            })
            .collect();
        let curves = self
            .curves
            .keys()
            .map(|id| {
                let declared =
                    self.profile_names.contains_key(id) || self.declared_curves.contains(id);
                (*id, legacy(declared, BlockKey::Curve(*id)))
            })
            .collect();
        (nodes, curves)
    }

    /// The curves `history/declared-curves.json` lists for an older build (spec §9.4), ascending:
    /// the variables its rule would not find to be one without the entry.
    ///
    /// That rule ([`Self::legacy_ownership`]) makes a curve a variable when a profile row's name is
    /// on it or two blocks read it, so the entry is the rest of the variables — the ones the person
    /// made one while a single block, or none, reads it. **Worked out on the way out, because the V
    /// writes the ownership and nothing else**: a curve put back inside that the opened file had
    /// declared is not declared again, and one made a variable since is. An older build lays its
    /// list out from this by its own rule, which differs at most in a row — a profile row named
    /// while inside its extrude is lifted out there — and never in geometry.
    pub fn curves_declared_for_older_builds(&self) -> Vec<CurveId> {
        let readers = self.readers();
        let mut out: Vec<CurveId> = self
            .curves
            .keys()
            .filter(|id| self.ownership(BlockKey::Curve(**id)) == Ownership::Variable)
            .filter(|id| !self.profile_names.contains_key(id))
            .filter(|id| readers.of(BlockKey::Curve(**id)).len() < 2)
            .copied()
            .collect();
        out.sort();
        out
    }

    /// Brings every block's ownership into line with who reads it (spec §3, §4), and pushes what
    /// moved onto [`Self::pending_ownership_changes`].
    ///
    /// **One pass, in ascending id order, with nothing cascading.** A block's ownership never
    /// changes another block's readers, so nothing settled early in the pass can be undone by
    /// something settled later, and the order only decides the order the changes are reported in.
    /// Idempotent for the same reason: a second pass finds every block already where its readers
    /// put it.
    ///
    /// **Writes no name and no visibility, and prunes nothing.** The name a new variable should
    /// carry ("Cube", not "Cuboid") resolves parameters, which is the application's business; what
    /// draws stays the stored flags' business (spec §7); and an entry for a block that has gone
    /// stays until a save, because an undo can bring the block back and its entry has to be here
    /// when it does (I5).
    ///
    /// Run by the edits that change who reads a block, not per frame — the list reads what this
    /// keeps, and keeping it is the whole point of storing it.
    pub fn settle_ownership(&mut self) {
        let readers = self.readers();
        let nodes: Vec<(NodeId, Ownership)> = self
            .node_ids()
            .into_iter()
            .map(|id| {
                let stored = self.ownership_nodes.get(&id).copied();
                (id, settle_one(stored, readers.of(BlockKey::Node(id))))
            })
            .collect();
        let mut curves: Vec<CurveId> = self.curves.keys().copied().collect();
        curves.sort();

        for (id, now) in nodes {
            let block = BlockKey::Node(id);
            let was = self.ownership_nodes.insert(id, now);
            self.pending_ownership_changes.extend(change(block, was, now, readers.of(block)));
        }
        for id in curves {
            let block = BlockKey::Curve(id);
            let now = settle_one(self.ownership_curves.get(&id).copied(), readers.of(block));
            let was = self.ownership_curves.insert(id, now);
            self.pending_ownership_changes.extend(change(block, was, now, readers.of(block)));
        }
    }

    /// Where `block` is stored as sitting, with a block no settle has placed yet read as free
    /// (spec §3) — the one rule for an absent entry, so the list and the commands that ask
    /// whether a block is a variable cannot read one two ways.
    pub fn ownership(&self, block: BlockKey) -> Ownership {
        let stored = match block {
            BlockKey::Node(id) => self.ownership_nodes.get(&id),
            BlockKey::Curve(id) => self.ownership_curves.get(&id),
        };
        stored.copied().unwrap_or(Ownership::Free)
    }

    /// Where *put back inside* (`notebook-spec.md` §6.6) takes a block, or the readers that
    /// refuse it.
    ///
    /// Nested when one block reads it, free when nothing does. Writes nothing — the write is a
    /// command's, so that it can be undone. The other refusal is a value variable whose contents
    /// are an expression, which is the parameter's matter and stays where the name is typed.
    /// **Refused again for two or more readers, 2026-10-24** (R1 inverted, [`settle_one`]).
    /// §6.6's "never refused for reader count" rested on the shared instance: with two readers the
    /// block could be nested in both. It cannot now — a settle would declare it straight back —
    /// so putting it back would be a gesture that undid itself silently. The answer is who the
    /// readers are, ascending, for the refusal to name them ("used by Boolean and Shell").
    pub fn put_back_target(&self, block: BlockKey) -> Result<Ownership, Vec<NodeId>> {
        match self.readers().of(block) {
            [] => Ok(Ownership::Free),
            [_] => Ok(Ownership::Nested),
            more => Err(more.to_vec()),
        }
    }

    /// Who reads `block` now, ascending, counted as I2 counts them — a curve's in every part — so
    /// who a settle run now would see.
    ///
    /// For the application, which says who uses a block a settle made a variable, and names the
    /// unnamed variables two blocks read when a file is opened (spec §6.3, §9.2). What a settle
    /// reported is who read the block when it ran, and an edit since can have changed that.
    pub fn block_readers(&self, block: BlockKey) -> Vec<NodeId> {
        self.readers().of(block).to_vec()
    }

    /// The blocks among `blocks` that are variables with no name, once each, in the order they
    /// were made: the blocks the application names after a settle, in the order it names them
    /// (spec §4.2, I4).
    ///
    /// **Made, not id, order**, since ids are random and the invented names suffix: of two spheres
    /// the one made first should be `Sphere` and the other `Sphere_1`, whichever id came out
    /// lower. A block nothing stamped sorts first, as in every list kept in the order of making
    /// ([`Self::made_at`]), and ids break ties.
    ///
    /// Only what is true now. What a settle reported is news from when it ran, and an undo or a
    /// later edit before the application heard of it can have put the block back inside, named
    /// it, or taken it away.
    pub fn unnamed_variables_among(&self, blocks: impl IntoIterator<Item = NodeId>) -> Vec<NodeId> {
        let mut found: Vec<NodeId> = blocks
            .into_iter()
            .filter(|id| self.ownership_nodes.get(id) == Some(&Ownership::Variable))
            .filter(|id| self.node_meta.get(id).is_none_or(|m| m.name.is_none()))
            .filter(|id| self.parts.values().any(|p| p.graph.node(*id).is_some()))
            .collect();
        found.sort_by_key(|id| (self.made_at(*id), id.raw()));
        found.dedup();
        found
    }

    /// Every node of every part, ascending.
    fn node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> =
            self.parts.values().flat_map(|p| p.graph.nodes().map(|(id, _)| id)).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Who reads each block, in one walk over every part (spec I2).
    ///
    /// A node is read by the blocks of its own part that name it among their inputs — every edge
    /// [`crate::Node::inputs`] returns, so a frame's plane and a driving ramp are read as much as
    /// an operand is. A curve belongs to the document, so it is read by any block of any part whose
    /// [`crate::Node::curves`] names it. Distinct readers: a boolean given one body twice is one
    /// reader of it, not two owners.
    fn readers(&self) -> Readers {
        let mut readers = Readers::default();
        for part in self.parts.values() {
            for (id, node) in part.graph.nodes() {
                node.for_each_input(|input| {
                    // An edge to a block not in this part is a dangling one, and reads nothing.
                    if part.graph.node(input).is_some() {
                        readers.nodes.entry(input).or_default().push(id);
                    }
                });
                for curve in node.curves() {
                    readers.curves.entry(curve).or_default().push(id);
                }
            }
        }
        for list in readers.nodes.values_mut().chain(readers.curves.values_mut()) {
            list.sort();
            list.dedup();
        }
        readers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BooleanOp, Curve, Frame, Graph, Node, Operand, Outline, ParamRef, Part, PartId, Place,
        RampShape,
    };
    use nalgebra::Point3;

    /// What `Document::nesting_in` returned before ownership was stored: the oracle's own copy of
    /// the shape, so nothing done to [`crate::Nesting`] later can change what the oracle answers.
    #[derive(Clone, Debug, Default)]
    struct Nesting {
        nodes: HashMap<NodeId, NodeId>,
        curves: HashMap<CurveId, NodeId>,
        // Built by the copied body and read by nothing here: where each row sits is the property,
        // and these are the counts a row's V reads, which move nothing.
        #[allow(dead_code)]
        readers: HashMap<NodeId, usize>,
        #[allow(dead_code)]
        curve_readers: HashMap<CurveId, usize>,
    }

    impl Nesting {
        fn owner_of(&self, id: NodeId) -> Option<NodeId> {
            self.nodes.get(&id).copied()
        }

        fn owner_of_curve(&self, id: CurveId) -> Option<NodeId> {
            self.curves.get(&id).copied()
        }
    }

    impl Document {
        /// **The oracle, and it must not be improved.** `nesting_in` as it stood on the day
        /// ownership began to be stored, copied word for word: the rule a document saved before
        /// then was laid out by, which `legacy_ownership` then `settle_ownership` has to reproduce
        /// row for row (spec §9.2, §12 test 1). Correcting anything here would move the target the
        /// migration is measured against, and the test would pass against a list nobody ever saw.
        fn old_nesting_in(&self, graph: &Graph) -> Nesting {
            // **Counted once, not asked per row.** The list is redrawn every frame, and "who
            // consumes this" asked per block walks every node once per block — which is the shape
            // that turns a hundred-block part into ten thousand `inputs()` allocations a frame.
            let mut consumers: HashMap<NodeId, (NodeId, usize)> = HashMap::new();
            let mut readers: HashMap<CurveId, (NodeId, usize)> = HashMap::new();
            // **Distinct consumers, so one block counts once however many times it names another.**
            // A node can hold the same edge twice -- a boolean given the same body for both operands,
            // a block whose driver is also its input -- and counting it twice would say "two owners"
            // about one.
            let mut seen: Vec<NodeId> = Vec::new();
            let mut named: Vec<CurveId> = Vec::new();
            for (id, node) in graph.nodes() {
                seen.clear();
                named.clear();
                for input in node.inputs() {
                    if seen.contains(&input) {
                        continue;
                    }
                    seen.push(input);
                    consumers.entry(input).and_modify(|(_, n)| *n += 1).or_insert((id, 1));
                }
                // **A curve is a dependency too, and `inputs` does not say so.** A profile lives in
                // the document rather than the graph, so `Node::operands` is empty for an extrude, a
                // sweep, a revolve and a pipe; the edge is in [`crate::Node::curves`] and nothing
                // else consults it. Without this the commonest case of all — a sketch and the
                // extrude made from it — stays flat.
                for curve in node.curves() {
                    if named.contains(&curve) {
                        continue;
                    }
                    named.push(curve);
                    readers.entry(curve).and_modify(|(_, n)| *n += 1).or_insert((id, 1));
                }
            }
            // **A name, and only a name.** The eye used to count too, on the reasoning that a block on
            // screen needs a row to be found by. Reversed by the owner on 2026-09-30, testing a sphere
            // inside a boolean: "when I clicked on the eyeball ... it automatically turned that into a
            // variable, which it shouldn't. It should only be when you click on the V box."
            //
            // The two controls answer different questions and the eye's own doc already said so —
            // naming says a block is *reusable*, showing says it can be *looked at*. Letting the eye
            // declare meant the answer to "can I see this" silently also answered "is this a
            // variable", and the row moved out from under the pointer that asked. A shown block that
            // nobody has named now draws in the viewport and keeps its row inside its consumer, where
            // the eye that turned it on is still sitting.
            let declared = |id: &NodeId| {
                self.node_meta.get(id).is_some_and(|m| m.name.is_some())
            };
            Nesting {
                nodes: consumers
                    .iter()
                    .filter(|(id, (_, count))| *count == 1 && !declared(id))
                    .map(|(id, (owner, _))| (*id, *owner))
                    .collect(),
                // **A curve's own name cannot be the declaration, because a curve is born with
                // one.** `unique_curve_name` gives every curve "Line", then "Line 2", so a name
                // says nothing about whether anybody has claimed it — where a node has no name at
                // all until somebody types one. So the consumer count is the whole test for a
                // curve, and a curve rises when a second block reads it or when its reader goes.
                //
                // A *profile* row is the exception and is handled here rather than by the caller: it
                // is a row with an optional name (`profile_names`, `profile-group-spec.md` §3.3),
                // so renaming one pulls the loop out exactly the way renaming a block does. Keyed
                // by the loop's first curve, which is the key the row is already drawn under.
                curves: readers
                    .iter()
                    .filter(|(id, (_, count))| {
                        *count == 1
                            && !self.profile_names.contains_key(id)
                            && !self.declared_curves.contains(id)
                    })
                    .map(|(id, (owner, _))| (*id, *owner))
                    .collect(),
                // Every block's count, not only the ones that nest: the panel asks it of named blocks
                // too, which are exactly the ones left out of `nodes` above.
                readers: consumers.iter().map(|(id, (_, count))| (*id, *count)).collect(),
                // The same for curves, which a profile row needs for the same reason a block row
                // does: whether turning its name off has anywhere to put it back.
                curve_readers: readers.iter().map(|(id, (_, count))| (*id, *count)).collect(),
            }
        }
    }

    /// A row of the list as the application lays it out before nesting: a node, a curve, or a
    /// profile row under its loop's first curve. The rest of `Block`'s rows never nest.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Row {
        Node(NodeId),
        Curve(CurveId),
        Profile(CurveId),
    }

    /// The rows in the order `App::ordered_blocks` gives them: the nodes as
    /// [`Document::listed_blocks`] lists them, merged with the curves by when each was made (a tie
    /// to the curve), then each loop folded into one row at its first curve, a member `own_row`
    /// says has a row of its own coming back out (`folded_curve_row_in`) — one an older build
    /// declared, as the old list read it, and a variable nothing reads, as the list reads it now.
    ///
    /// The order matters to the oracle and not only to the eye: `nest_rows` puts a row inside a
    /// block only when that block's row comes later, and a curve stamped after the block reading it
    /// lands above it.
    fn rows_of(
        doc: &Document,
        graph: &Graph,
        loops: &[Vec<CurveId>],
        own_row: &dyn Fn(CurveId) -> bool,
    ) -> Vec<Row> {
        let mut curves: Vec<CurveId> = doc.curves.keys().copied().collect();
        curves.sort_by_key(|id| (doc.curves_created.get(id).copied().unwrap_or(0), id.raw()));
        let made = |id: &NodeId| doc.created.get(id).copied().unwrap_or(0);
        let mut c = curves.into_iter().peekable();
        let mut n = doc.listed_blocks(graph).into_iter().peekable();
        let mut merged = Vec::new();
        loop {
            match (c.peek().copied(), n.peek().copied()) {
                (None, None) => break,
                (Some(curve), Some(node))
                    if doc.curves_created.get(&curve).copied().unwrap_or(0) > made(&node) =>
                {
                    n.next();
                    merged.push(Row::Node(node));
                }
                (Some(curve), _) => {
                    c.next();
                    merged.push(Row::Curve(curve));
                }
                (None, Some(node)) => {
                    n.next();
                    merged.push(Row::Node(node));
                }
            }
        }
        let key_of = |id: CurveId| loops.iter().find(|l| l.contains(&id)).map(|l| l[0]);
        merged
            .into_iter()
            .filter_map(|row| match row {
                Row::Curve(id) => match key_of(id) {
                    None => Some(Row::Curve(id)),
                    Some(key) if key == id => Some(Row::Profile(key)),
                    Some(_) if own_row(id) => Some(Row::Curve(id)),
                    Some(_) => None,
                },
                other => Some(other),
            })
            .collect()
    }

    /// Which row each row is drawn inside, by `nest_rows`' two rules over an owner lookup: a
    /// profile row goes inside the first member of its loop, in walk order, that has an owner;
    /// and any row goes inside its owner only when the owner's row comes after it.
    ///
    /// Both sides of the migration test go through this, because both will be drawn by
    /// `nest_rows`: the old list by the old nesting, the new one by the stored ownership.
    fn drawn_inside(
        rows: &[Row],
        loops: &[Vec<CurveId>],
        owner_of: &dyn Fn(NodeId) -> Option<NodeId>,
        owner_of_curve: &dyn Fn(CurveId) -> Option<NodeId>,
    ) -> Vec<(Row, Option<NodeId>)> {
        let at: HashMap<NodeId, usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, row)| match row {
                Row::Node(id) => Some((*id, i)),
                _ => None,
            })
            .collect();
        rows.iter()
            .enumerate()
            .map(|(i, row)| {
                let owner = match row {
                    Row::Node(id) => owner_of(*id),
                    Row::Curve(id) => owner_of_curve(*id),
                    Row::Profile(key) => loops
                        .iter()
                        .find(|l| l[0] == *key)
                        .and_then(|l| l.iter().find_map(|member| owner_of_curve(*member))),
                };
                (*row, owner.filter(|owner| at.get(owner).is_some_and(|owner_at| *owner_at > i)))
            })
            .collect()
    }

    /// The block an ownership puts a row inside, which is what the list will read from it.
    ///
    /// **Read off the readers, since 112.** `Nested` names no parent (`notebook-spec.md` §12), so
    /// the block a row is drawn in is whatever reads it, as [`crate::Nesting::owners_of`] answers
    /// it. The migration leaves every `Nested` block with exactly one reader — two readers migrate
    /// to `Variable`, which is what keeps the row the old list drew (§15) — so there is a single
    /// owner here to hold against the oracle's, and a `Nested` block with any other number of
    /// readers is the migration having moved a row rather than a comparison to make.
    fn inside(doc: &Document, block: BlockKey) -> Option<NodeId> {
        match ownership_of(doc, block) {
            Some(Ownership::Nested) => {
                let readers = readers_in(doc, block);
                assert_eq!(readers.len(), 1, "{block:?} is nested and read by {readers:?}");
                readers.first().copied()
            }
            _ => None,
        }
    }

    /// Numbers that are the same every run, so a seed that fails fails again.
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
    }

    /// A document the migration has to reproduce, and the profile loops the application would
    /// gather its curves into, each in walk order with its key first.
    struct Sample {
        doc: Document,
        part: PartId,
        loops: Vec<Vec<CurveId>>,
    }

    /// A random one-part document: a DAG of blocks of every edge kind — operands, a driver, a
    /// frame and a place pointing at another block, curves read once, twice or by two blocks —
    /// names on a random subset of blocks, profile names and declarations on a random subset of
    /// curves, the eye on some, and curves stamped before, between and after the blocks reading
    /// them.
    fn sample(seed: u64) -> Sample {
        let mut rng = Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x2545_f491_4f6c_dd1d);
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let curves: Vec<CurveId> = (0..rng.below(7))
            .map(|i| {
                let curve = Curve::polyline(
                    CurveId::new(),
                    format!("Line {i}"),
                    vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, i as f64, 0.0)],
                    false,
                )
                .expect("a line");
                let id = curve.id;
                doc.curves.insert(id, curve);
                id
            })
            .collect();
        let mut graph = Graph::empty();
        let mut nodes: Vec<NodeId> = Vec::new();
        let count = 2 + rng.below(13);
        for _ in 0..count {
            let any = |rng: &mut Lcg| nodes[rng.below(nodes.len())];
            let curve = |rng: &mut Lcg| curves[rng.below(curves.len())];
            let needs_block = nodes.is_empty();
            let needs_curve = curves.is_empty();
            let node = match rng.below(12) {
                1 => Node::Point { at: [lit(0.0); 3] },
                2 if !needs_block => Node::Boolean {
                    op: BooleanOp::Union,
                    // Repeats allowed: a block named twice by one reader is one reader.
                    inputs: (0..1 + rng.below(3)).map(|_| any(&mut rng).into()).collect(),
                },
                3 if !needs_block => Node::Shell { input: any(&mut rng).into(), thickness: lit(1.0) },
                4 if !needs_block => {
                    Node::Offset { input: any(&mut rng).into(), distance: ParamRef::Field(any(&mut rng)) }
                }
                5 if !needs_block => Node::Mirror {
                    input: any(&mut rng).into(),
                    frame: Frame::From { node: any(&mut rng) },
                    join: true,
                },
                6 => Node::Ramp {
                    input: match !needs_block && rng.percent(70) {
                        true => any(&mut rng).into(),
                        false => Operand::empty(),
                    },
                    in_min: lit(0.0),
                    in_max: lit(1.0),
                    out_min: lit(0.0),
                    out_max: lit(1.0),
                    shape: RampShape::Linear,
                },
                7 if !needs_curve => Node::Extrude {
                    outline: Outline::closed(curve(&mut rng)),
                    sweep: crate::Sweep::Normal,
                    height: lit(10.0),
                    start: lit(0.0),
                    sides: crate::Sides::One,
                    draft: lit(0.0),
                    corners: Default::default(),
                    rims: Default::default(),
                },
                8 if !needs_curve => Node::Pipe { curve: curve(&mut rng), radius: lit(1.0) },
                9 if !needs_curve => Node::Sweep {
                    outline: Outline::closed(curve(&mut rng)),
                    spine: curve(&mut rng),
                    section: crate::Section::Square,
                    law: Default::default(),
                    rims: Default::default(),
                },
                10 if !needs_block => Node::Transform {
                    input: any(&mut rng).into(),
                    translation: [lit(0.0); 3],
                    rotation: [lit(0.0); 3],
                    scale: lit(1.0),
                    about: Place::At([lit(0.0); 3]),
                },
                11 if !needs_block => {
                    Node::Sphere { centre: Place::From { node: any(&mut rng) }, radius: lit(5.0) }
                }
                _ => Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(5.0) },
            };
            nodes.push(graph.add(node));
        }

        const MOMENT: u64 = 1_700_000_000_000_000;
        for (i, id) in nodes.iter().enumerate() {
            doc.created.insert(*id, MOMENT + 10 * (i as u64 + 1));
            if rng.percent(30) {
                doc.node_meta.entry(*id).or_default().name = Some(format!("Block {i}"));
            }
            if rng.percent(20) {
                doc.node_meta.entry(*id).or_default().shown = true;
            }
            if rng.percent(10) {
                doc.node_meta.entry(*id).or_default().hidden = true;
            }
        }
        // On a node's moment as often as between two, so the merge's tie goes somewhere; and
        // sometimes never stamped, as a file from before curves were.
        for id in &curves {
            if !rng.percent(15) {
                let slot = rng.below(count + 2) as u64;
                doc.curves_created.insert(*id, MOMENT + 10 * slot + 5 * rng.below(2) as u64);
            }
            if rng.percent(20) {
                doc.declared_curves.insert(*id);
            }
        }
        let mut loops: Vec<Vec<CurveId>> = Vec::new();
        let mut rest = curves.clone();
        while rest.len() >= 2 {
            let take = (2 + rng.below(2)).min(rest.len());
            let chunk: Vec<CurveId> = rest.drain(..take).collect();
            if rng.percent(50) {
                if rng.percent(40) {
                    doc.profile_names.insert(chunk[0], "Profile".into());
                }
                loops.push(chunk);
            }
        }
        // A profile name on a curve no loop has: the old rule read it there too.
        if let Some(stray) = rest.first().filter(|_| rng.percent(30)) {
            doc.profile_names.insert(*stray, "Stray".into());
        }

        let part = doc.add_part(Part {
            id: PartId::new(),
            name: "Part".into(),
            material: None,
            graph,
            inputs: Vec::new(),
        });
        Sample { doc, part, loops }
    }

    /// Who reads a block, counted the way spec I2 says and written out again here rather than
    /// taken from the code under test: a node's distinct readers in its own part, a curve's in
    /// every part.
    fn readers_in(doc: &Document, block: BlockKey) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = Vec::new();
        for part in doc.parts.values() {
            let holds = |id: NodeId| match block {
                BlockKey::Node(node) => {
                    part.graph.node(node).is_some()
                        && part.graph.node(id).is_some_and(|n| n.inputs().contains(&node))
                }
                BlockKey::Curve(curve) => {
                    part.graph.node(id).is_some_and(|n| n.curves().contains(&curve))
                }
            };
            out.extend(part.graph.nodes().map(|(id, _)| id).filter(|id| holds(*id)));
        }
        out.sort();
        out.dedup();
        out
    }

    /// Every block there is to own, as the spec's I5 means it: each node of each part and each
    /// curve of the document.
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

    /// Over every block: each has an ownership after a settle, and a `Nested` one has at least one
    /// reader — I2 as 112 restates it (spec §12).
    ///
    /// **I1 and I3 went with the stored parent.** I1 asked that an `Inside(p)`'s `p` exist and I3
    /// that no chain of `Inside` close on itself; `Nested` names no parent, so neither can be
    /// written down any more. The ring a malformed file can still hold is a ring of *readers*, and
    /// what has to survive it is the walk that draws the rows —
    /// [`Document::nesting_in`], asserted by
    /// `a_reader_ring_a_file_holds_still_leaves_every_row_reachable`.
    fn assert_invariants(doc: &Document, context: &str) {
        for block in every_block(doc) {
            let Some(ownership) = ownership_of(doc, block) else {
                panic!("{context}: {block:?} has no ownership after a settle");
            };
            if ownership == Ownership::Nested {
                assert!(
                    !readers_in(doc, block).is_empty(),
                    "{context}: {block:?} is nested with nothing reading it",
                );
            }
        }
    }

    /// A one-part document holding `graph`, settled once so every block has its ownership, with
    /// what that announced already heard.
    fn settled(graph: Graph) -> (Document, PartId) {
        let mut doc = Document::new();
        let part = doc.add_part(Part {
            id: PartId::new(),
            name: "Part".into(),
            material: None,
            graph,
            inputs: Vec::new(),
        });
        doc.settle_ownership();
        doc.pending_ownership_changes.clear();
        (doc, part)
    }

    fn graph_of(doc: &mut Document, part: PartId) -> &mut Graph {
        &mut doc.parts.get_mut(&part).expect("the part").graph
    }

    fn ball(graph: &mut Graph) -> NodeId {
        let lit = ParamRef::Literal;
        graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(5.0) })
    }

    fn union(graph: &mut Graph, inputs: Vec<NodeId>) -> NodeId {
        let inputs = inputs.into_iter().map(Operand::filled).collect();
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs })
    }

    fn shell(graph: &mut Graph, input: NodeId) -> NodeId {
        graph.add(Node::Shell { input: input.into(), thickness: ParamRef::Literal(1.0) })
    }

    /// Three blocks, one of each ownership: a sphere inside the union that is its only reader, a
    /// body nothing reads, and a named variable the union also reads.
    struct Three {
        doc: Document,
        nested: NodeId,
        free: NodeId,
        variable: NodeId,
    }

    fn three() -> Three {
        let mut graph = Graph::empty();
        let nested = ball(&mut graph);
        let free = ball(&mut graph);
        let variable = ball(&mut graph);
        union(&mut graph, vec![nested, variable]);
        let (mut doc, _) = settled(graph);
        doc.node_meta.entry(variable).or_default().name = Some("Hub".into());
        doc.ownership_nodes.insert(variable, Ownership::Variable);
        Three { doc, nested, free, variable }
    }

    /// **A file from before ownership was stored opens with every row where it was.**
    ///
    /// The migration's whole promise (spec §9.2), held against the rule those files were laid
    /// out by: `old_nesting_in`, today's `nesting_in` word for word, drawn through `nest_rows`'
    /// own two rules — a profile row goes inside the first member of its loop with an owner, and
    /// nothing goes inside a row above it. The other side is `legacy_ownership` then
    /// `settle_ownership`, drawn through the same two rules, since `nest_rows` is what will draw
    /// it. And opening one announces nothing, because the migration already writes what a settle
    /// would: every block it hands over is where its readers put it, so the settle that follows
    /// finds nothing to move.
    ///
    /// **One row goes, and it is the spec's own conflict rather than a choice: a loop member an
    /// older build declared while a block read it.** It is drawn in its loop's row, where the old
    /// list gave it a row of its own; the profile row stays where it was. The declaration is a
    /// variable now (§10), and so is a member two blocks read (§9.2), and the old list drew the
    /// first outside its loop's row and the second inside — so with the declaration no longer kept
    /// (refactor plan 2.19) nothing can tell the two apart. The list keeps a member a block reads
    /// in its loop's row, where the second always was, because lifting it out instead split the
    /// loop by which of its curves a block was made from. Each row that goes is required to be
    /// exactly that one, and every other row to be where it was, in its order. A loop member two
    /// blocks read, which the old list folded in, is counted, and required to have come up.
    ///
    /// Many seeds, and the cases the old rule distinguished are each counted and required to
    /// have come up, so a generator that stopped producing one would fail here rather than pass
    /// by never asking.
    #[test]
    fn a_document_from_before_ownership_lists_every_row_where_it_was() {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for seed in 0..600 {
            let Sample { doc, part, loops } = sample(seed);
            let graph = &doc.parts[&part].graph;
            let rows = rows_of(&doc, graph, &loops, &|id| doc.declared_curves.contains(&id));
            let old = doc.old_nesting_in(graph);
            let was = drawn_inside(&rows, &loops, &|id| old.owner_of(id), &|id| {
                old.owner_of_curve(id)
            });

            let mut opened = doc.clone();
            let (nodes, curves) = opened.legacy_ownership();
            opened.ownership_nodes = nodes;
            opened.ownership_curves = curves;
            opened.settle_ownership();
            let lifted = |id| {
                opened.ownership_curves.get(&id) == Some(&Ownership::Variable)
                    && readers_in(&opened, BlockKey::Curve(id)).is_empty()
            };
            let opened_rows = rows_of(&opened, graph, &loops, &lifted);
            let is = drawn_inside(
                &opened_rows,
                &loops,
                &|id| inside(&opened, BlockKey::Node(id)),
                &|id| inside(&opened, BlockKey::Curve(id)),
            );
            let mut kept = Vec::new();
            for (row, owner) in &was {
                if opened_rows.contains(row) {
                    kept.push((*row, *owner));
                    continue;
                }
                let Row::Curve(id) = *row else {
                    panic!("seed {seed}: {row:?} is gone from the list");
                };
                assert!(
                    doc.declared_curves.contains(&id)
                        && !readers_in(&doc, BlockKey::Curve(id)).is_empty()
                        && loops.iter().any(|l| l[1..].contains(&id)),
                    "seed {seed}: a curve row went that is not a loop member declared while read",
                );
                *seen.entry("a member declared while read is drawn in its loop's row").or_default() +=
                    1;
            }
            assert_eq!(
                is.iter().map(|(row, _)| *row).collect::<Vec<_>>(),
                kept.iter().map(|(row, _)| *row).collect::<Vec<_>>(),
                "seed {seed}: no row the old list lacked, and the rest in its order",
            );
            for (before, after) in kept.iter().zip(&is) {
                assert_eq!(after, before, "seed {seed}: the row moved");
            }
            for member in loops.iter().flat_map(|l| &l[1..]) {
                let read = readers_in(&doc, BlockKey::Curve(*member)).len();
                if read >= 2 && !doc.declared_curves.contains(member) {
                    *seen.entry("a member two blocks read stays in its loop's row").or_default() +=
                        1;
                }
            }
            assert_eq!(
                opened.pending_ownership_changes,
                Vec::new(),
                "seed {seed}: opening the file would announce moves",
            );
            assert_invariants(&opened, &format!("seed {seed}"));

            for (row, owner) in &was {
                let key = match (row, owner) {
                    (Row::Node(_), Some(_)) => "a block drawn inside another",
                    (Row::Curve(_), Some(_)) => "a curve drawn inside a block",
                    (Row::Profile(_), Some(_)) => "a profile row drawn inside a block",
                    (Row::Profile(_), None) => "a profile row of its own",
                    _ => "a row of its own",
                };
                *seen.entry(key).or_default() += 1;
            }
            for block in every_block(&doc) {
                let readers = readers_in(&doc, block).len();
                let named = match block {
                    BlockKey::Node(id) => doc.node_meta.get(&id).is_some_and(|m| m.name.is_some()),
                    BlockKey::Curve(id) => {
                        doc.profile_names.contains_key(&id) || doc.declared_curves.contains(&id)
                    }
                };
                let key = match (named, readers) {
                    (true, 1) => "a named block one block reads",
                    (false, 2..) => "an unnamed block two blocks read",
                    (true, 2..) => "a named block two blocks read",
                    _ => continue,
                };
                *seen.entry(key).or_default() += 1;
            }
            let held_back = was.iter().zip(&rows).any(|((_, owner), row)| {
                owner.is_none()
                    && match row {
                        Row::Curve(id) => old.owner_of_curve(*id).is_some(),
                        Row::Node(id) => old.owner_of(*id).is_some(),
                        Row::Profile(_) => false,
                    }
            });
            if held_back {
                *seen.entry("an owner listed above its row").or_default() += 1;
            }
        }
        for case in [
            "a block drawn inside another",
            "a curve drawn inside a block",
            "a profile row drawn inside a block",
            "a profile row of its own",
            "a named block one block reads",
            "an unnamed block two blocks read",
            "a named block two blocks read",
            "an owner listed above its row",
            "a member two blocks read stays in its loop's row",
            "a member declared while read is drawn in its loop's row",
        ] {
            let times = seen.get(case).copied().unwrap_or(0);
            assert!(times >= 10, "{case} came up {times} times in 600 documents: {seen:?}");
        }
    }

    /// **A curve two parts read is a variable once its file is opened** — the one row the
    /// migration moves, on purpose (spec §9.2, ruled §14 Q5).
    ///
    /// The old rule counted a curve's readers in the part being listed, so a pipe in each of two
    /// parts drew the curve inside both pipes, as though each owned it. It is one object of the
    /// document read by two blocks, and two owners is not ownership.
    ///
    /// **A variable still, and ruling 1 is why it has to be.** A second reader makes a shared
    /// instance in a new document now (`notebook-spec.md` §7.3), so one might expect this curve to
    /// migrate to `Nested` and be drawn in both pipes. The migration is deliberately unchanged
    /// (§15): the old list gave a curve two blocks read **a row of its own**, and an old file has
    /// to open drawing the same things. `Nested` would redraw it inside each reader, which is a
    /// row that moved; `Variable` is the one answer that keeps it. *Put back inside* is how one of
    /// these becomes a shared instance, by the person asking.
    #[test]
    fn a_curve_two_parts_read_becomes_a_variable_on_migration() {
        let mut doc = Document::new();
        let spine = Curve::polyline(
            CurveId::new(),
            "Line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            false,
        )
        .expect("a line");
        let spine_id = spine.id;
        doc.curves.insert(spine_id, spine);
        let mut pipes = Vec::new();
        for name in ["Left", "Right"] {
            let mut graph = Graph::empty();
            pipes.push(graph.add(Node::Pipe { curve: spine_id, radius: ParamRef::Literal(2.0) }));
            doc.add_part(Part {
                id: PartId::new(),
                name: name.into(),
                material: None,
                graph,
                inputs: Vec::new(),
            });
        }
        for (part, pipe) in doc.parts_ordered().iter().zip(&pipes) {
            assert_eq!(
                doc.old_nesting_in(&part.graph).owner_of_curve(spine_id),
                Some(*pipe),
                "the old rule drew it inside the block reading it, in each part",
            );
        }

        let (nodes, curves) = doc.legacy_ownership();
        assert_eq!(curves.get(&spine_id), Some(&Ownership::Variable), "read by two: a variable");
        doc.ownership_nodes = nodes;
        doc.ownership_curves = curves;
        doc.settle_ownership();
        assert_eq!(doc.ownership_curves.get(&spine_id), Some(&Ownership::Variable));
        assert_eq!(doc.pending_ownership_changes, Vec::new(), "and opening it says nothing");
    }

    /// **A rename never moves a block, nor does clearing the name** — inside, free or a variable.
    ///
    /// The defect this whole model exists to end: a name was the declaration, so writing one for
    /// any reason lifted the row out, and Pattern and Mirror naming what they consumed did exactly
    /// that (spec §1). A name is a label now, and nothing that settles reads it.
    #[test]
    fn renaming_a_block_never_moves_it() {
        let Three { mut doc, nested, free, variable, .. } = three();
        let before = doc.ownership_nodes.clone();
        assert_eq!(before.get(&nested), Some(&Ownership::Nested));
        assert_eq!(before.get(&free), Some(&Ownership::Free));
        for id in [nested, free, variable] {
            doc.node_meta.entry(id).or_default().name = Some("Renamed".into());
            doc.settle_ownership();
            assert_eq!(doc.ownership_nodes, before, "naming {id:?} moved a block");
            doc.node_meta.entry(id).or_default().name = None;
            doc.settle_ownership();
            assert_eq!(doc.ownership_nodes, before, "clearing {id:?}'s name moved a block");
        }
        assert_eq!(doc.pending_ownership_changes, Vec::new(), "and nothing was announced");
    }

    /// **The eye never moves a block**, whichever of the two flags it writes.
    ///
    /// Reported 2026-09-30 when the eye still counted: "when I clicked on the eyeball ... it
    /// automatically turned that into a variable, which it shouldn't."
    #[test]
    fn the_eye_never_moves_a_block() {
        let Three { mut doc, nested, free, variable, .. } = three();
        let before = doc.ownership_nodes.clone();
        assert_eq!(before.get(&nested), Some(&Ownership::Nested));
        for id in [nested, free, variable] {
            for (shown, hidden) in [(true, false), (false, true), (true, true), (false, false)] {
                let meta = doc.node_meta.entry(id).or_default();
                meta.shown = shown;
                meta.hidden = hidden;
                doc.settle_ownership();
                assert_eq!(doc.ownership_nodes, before, "{id:?} shown {shown}, hidden {hidden}");
            }
        }
        assert_eq!(doc.pending_ownership_changes, Vec::new());
    }

    /// **A variable is never adopted** — not by a first reader, a second, or after losing them all.
    ///
    /// The whole difference between a variable and a free body (spec §4): making something a
    /// variable is a decision the person makes, and settling silently undoing it would change what
    /// the next edit to a shared block does.
    #[test]
    fn a_variable_is_never_adopted() {
        let mut graph = Graph::empty();
        let hub = ball(&mut graph);
        let (mut doc, part) = settled(graph);
        assert_eq!(doc.ownership_nodes.get(&hub), Some(&Ownership::Free));
        doc.ownership_nodes.insert(hub, Ownership::Variable);

        let first = shell(graph_of(&mut doc, part), hub);
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes.get(&hub), Some(&Ownership::Variable), "one reader");
        assert_eq!(doc.ownership_nodes.get(&first), Some(&Ownership::Free), "the reader is a body");

        let second = union(graph_of(&mut doc, part), vec![hub]);
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes.get(&hub), Some(&Ownership::Variable), "two readers");

        let graph = graph_of(&mut doc, part);
        graph.remove(first);
        graph.remove(second);
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes.get(&hub), Some(&Ownership::Variable), "no readers");
        assert_eq!(doc.pending_ownership_changes, Vec::new(), "and none of it was a move");
    }

    /// **A free body goes inside the first block to read it**, and through the drawn rule that
    /// already exists it stops being drawn: it is no longer a body.
    ///
    /// Today's behaviour, kept on purpose: "shell this cube" adopts the cube. A block made by a
    /// path that has not settled yet has no entry at all, which reads as free, and is adopted the
    /// same way.
    #[test]
    fn a_free_body_is_adopted_by_its_first_consumer() {
        let mut graph = Graph::empty();
        let body = ball(&mut graph);
        let (mut doc, part) = settled(graph);
        assert_eq!(doc.ownership_nodes.get(&body), Some(&Ownership::Free));
        let unplaced = ball(graph_of(&mut doc, part));
        assert_eq!(doc.ownership_nodes.get(&unplaced), None, "made without a settle");

        let both = union(graph_of(&mut doc, part), vec![body, unplaced]);
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes.get(&body), Some(&Ownership::Nested));
        assert_eq!(doc.ownership_nodes.get(&unplaced), Some(&Ownership::Nested));
        assert_eq!(doc.ownership_nodes.get(&both), Some(&Ownership::Free), "a body of its own");
        let mut heard = doc.pending_ownership_changes.clone();
        heard.sort_by_key(|c| format!("{c:?}"));
        let mut expected = vec![
            OwnershipChange::Adopted { block: BlockKey::Node(body), readers: vec![both] },
            OwnershipChange::Adopted { block: BlockKey::Node(unplaced), readers: vec![both] },
        ];
        expected.sort_by_key(|c| format!("{c:?}"));
        assert_eq!(heard, expected);

        let graph = &doc.parts[&part].graph;
        assert!(!graph.roots().contains(&body), "consumed, so not a body");
        assert!(!doc.node_meta.get(&body).is_some_and(|m| m.shown), "and nothing shows it");
    }

    /// **A nested block nothing reads any more is released**, and it is a body that draws again.
    ///
    /// Settling only ever *places* a block where its readers put it, so this is the whole of what
    /// it does when the block reading something goes: nothing reads it, so it is `Free` and the
    /// move is announced once.
    ///
    /// **And this is the whole of it, since the owner declined `notebook-spec.md` §7.4**, which
    /// would have had a delete *take* what is nested in it. Delete leaves what a block was made
    /// from — undoing the operation rather than discarding the shapes — so the orphan becoming a
    /// top-level row is not a gap waiting for a command, it is the behaviour. The graph is emptied
    /// by hand (`remove_cascading`) to leave settle the orphans it would see.
    #[test]
    fn a_nested_block_nothing_reads_any_more_becomes_free() {
        let mut graph = Graph::empty();
        let sphere = ball(&mut graph);
        let cube = ball(&mut graph);
        let both = union(&mut graph, vec![sphere, cube]);
        let (mut doc, part) = settled(graph);
        assert_eq!(doc.ownership_nodes.get(&sphere), Some(&Ownership::Nested));

        graph_of(&mut doc, part).remove_cascading(both);
        doc.settle_ownership();
        for id in [sphere, cube] {
            assert_eq!(doc.ownership_nodes.get(&id), Some(&Ownership::Free));
            assert!(
                doc.pending_ownership_changes
                    .contains(&OwnershipChange::Released { block: BlockKey::Node(id) }),
                "released, and with no parent stored there is no block to name it as leaving",
            );
            assert!(doc.parts[&part].graph.roots().contains(&id), "a body again, so it draws");
        }
        assert_eq!(doc.pending_ownership_changes.len(), 2);
        assert_eq!(
            doc.ownership_nodes.get(&both),
            Some(&Ownership::Free),
            "the deleted block's entry stays until a save prunes it, for an undo to find",
        );
    }

    /// **A second reader makes a variable**: the block leaves the slot it was drawn in for a row
    /// of its own, and the move is announced (`notebook-spec.md` §7.3 and §17, ruled 2026-10-24).
    ///
    /// **Inverted back from the shared instance (§17), which 112 had inverted from this.** The
    /// reason is containment: a slot holds the block it reads, so a block is in one slot or none,
    /// and two readers cannot each hold it. Recorded in both directions here because the rule has
    /// now been written twice and the owner has seen both working.
    ///
    /// Two starting states, because they differ in what is said. The sphere is **nested in one
    /// block already** and a second starts reading it: it is declared, and `BecameVariable` names
    /// both readers. The loose block is **free** and two readers pick it up in the same settle: it
    /// goes straight to a variable as well, in one change rather than an adoption followed by a
    /// promotion — the settle sees the end state, not the path to it.
    ///
    /// Neither is named here: naming a promoted block is the application's (`variables_to_name`),
    /// which is why the settle leaves `node_meta` alone.
    #[test]
    fn a_second_reader_makes_a_variable() {
        let mut graph = Graph::empty();
        let sphere = ball(&mut graph);
        let first = union(&mut graph, vec![sphere]);
        let loose = ball(&mut graph);
        let (mut doc, part) = settled(graph);
        assert_eq!(doc.ownership_nodes.get(&sphere), Some(&Ownership::Nested));
        assert_eq!(doc.ownership_nodes.get(&loose), Some(&Ownership::Free));

        let graph = graph_of(&mut doc, part);
        let second = shell(graph, sphere);
        let a = shell(graph, loose);
        let b = shell(graph, loose);
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes.get(&sphere), Some(&Ownership::Variable), "declared");
        assert_eq!(doc.ownership_nodes.get(&loose), Some(&Ownership::Variable), "and so is this");
        let sorted = |mut v: Vec<NodeId>| {
            v.sort();
            v
        };
        // A variable's row is its own: the list draws it at the top level, and each reader
        // references it rather than containing it.
        let nesting = doc.nesting_in(&doc.parts[&part].graph);
        assert_eq!(nesting.owners_of(sphere).to_vec(), Vec::new());
        assert_eq!(nesting.owners_of(loose).to_vec(), Vec::new());
        let mut said = doc.pending_ownership_changes.clone();
        said.sort_by_key(|c| match c.block() {
            BlockKey::Node(id) => id.raw(),
            BlockKey::Curve(id) => id.raw(),
        });
        let mut wanted = vec![
            OwnershipChange::BecameVariable {
                block: BlockKey::Node(sphere),
                readers: sorted(vec![first, second]),
            },
            OwnershipChange::BecameVariable {
                block: BlockKey::Node(loose),
                readers: sorted(vec![a, b]),
            },
        ];
        wanted.sort_by_key(|c| match c.block() {
            BlockKey::Node(id) => id.raw(),
            BlockKey::Curve(id) => id.raw(),
        });
        assert_eq!(said, wanted, "both promotions are announced, with who reads each");
        assert_eq!(
            doc.node_meta.get(&sphere).and_then(|m| m.name.clone()),
            None,
            "and neither is named here: the application names what a settle declared",
        );
    }

    /// **The variables a settle leaves unnamed are named in the order they were made, not by id**
    /// (spec §4.2) — so of two spheres, the one made first is `Sphere` and the other `Sphere_1`.
    ///
    /// Ids are random, so the two orders disagree about half the time; here they are made to
    /// disagree every time, by stamping the higher id first. Only blocks that are variables now,
    /// have no name and are still here are listed: a variable already named, a block nobody
    /// declared, and one deleted since are each among those asked about and left out.
    ///
    /// **The variables are written, not wired, since 112.** This used to make one by giving a
    /// block two readers, which settling turned into a variable; a second reader makes a shared
    /// instance now (§7.3), so the only thing that writes `Variable` is the command the person
    /// asks for — and writing the entry is what that command does.
    #[test]
    fn unnamed_variables_are_named_in_the_order_they_were_made() {
        let mut graph = Graph::empty();
        let (one, two) = (ball(&mut graph), ball(&mut graph));
        let (early, late) = if one > two { (one, two) } else { (two, one) };
        let named = ball(&mut graph);
        let gone = ball(&mut graph);
        let single = ball(&mut graph);
        union(&mut graph, vec![single]);
        let (mut doc, part) = settled(graph);
        for (at, block) in [early, late, named, gone, single].into_iter().enumerate() {
            doc.created.insert(block, 100 + at as u64);
        }
        for block in [early, late, named, gone] {
            doc.ownership_nodes.insert(block, Ownership::Variable);
        }
        doc.node_meta.entry(named).or_default().name = Some("Hub".into());
        assert_eq!(
            doc.ownership_nodes.get(&single),
            Some(&Ownership::Nested),
            "the union reads it, so nobody made it a variable",
        );
        graph_of(&mut doc, part).remove(gone);
        assert_eq!(doc.ownership_nodes.get(&gone), Some(&Ownership::Variable), "the entry stays");

        let asked = [late, single, gone, named, early];
        assert_eq!(doc.unnamed_variables_among(asked), vec![early, late], "made first, named first");
        assert_eq!(doc.unnamed_variables_among([late, late, early]), vec![early, late], "once each");
    }

    /// **A block slid under a transform stays nested and nothing about it is announced**, least of
    /// all a declaration.
    ///
    /// Placing a body wraps it: the transform reads it and whatever read it reads the transform.
    /// **The block itself has nothing to report, since 112.** It was nested and it is nested — the
    /// block it is drawn inside changed, and that was `Rehomed`'s news while a parent was stored;
    /// no parent is stored now, so the stored entry is the same entry it was and the list reads the
    /// new owner off the readers by itself. What *is* announced is the transform's own adoption: it
    /// is a new block, free until the union read it. The thing this test has always guarded is the
    /// other half — a slide must not declare, which is what made a row leap out of its consumer.
    #[test]
    fn a_block_slid_under_a_transform_stays_nested_and_reports_nothing() {
        let mut graph = Graph::empty();
        let sphere = ball(&mut graph);
        let cube = ball(&mut graph);
        let both = union(&mut graph, vec![sphere, cube]);
        let (mut doc, part) = settled(graph);

        let graph = graph_of(&mut doc, part);
        let lit = ParamRef::Literal;
        let placed = graph.add(Node::Transform {
            input: sphere.into(),
            translation: [lit(5.0), lit(0.0), lit(0.0)],
            rotation: [lit(0.0); 3],
            scale: lit(1.0),
            about: Place::At([lit(0.0); 3]),
        });
        graph.rewire_consumers(sphere, placed);
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes.get(&sphere), Some(&Ownership::Nested), "not declared");
        assert_eq!(doc.ownership_nodes.get(&placed), Some(&Ownership::Nested));
        assert_eq!(doc.ownership_nodes.get(&cube), Some(&Ownership::Nested), "untouched");
        assert_eq!(
            doc.pending_ownership_changes,
            vec![OwnershipChange::Adopted { block: BlockKey::Node(placed), readers: vec![both] }],
            "the transform's adoption, and nothing said about the block it now holds",
        );
        // And the list draws it in the transform, read off the readers rather than the entry.
        let nesting = doc.nesting_in(&doc.parts[&part].graph);
        assert_eq!(nesting.owners_of(sphere).to_vec(), vec![placed]);
        assert_eq!(nesting.owners_of(placed).to_vec(), vec![both]);
        assert_eq!(nesting.owners_of(cube).to_vec(), vec![both]);
    }

    /// **Putting a variable back inside nests it in its one reader, frees it when nothing reads
    /// it, and is refused by two or more** — readers counted as I2 counts them, so a curve's in
    /// every part (`notebook-spec.md` §6.6).
    ///
    /// **Refused for reader count again, 2026-10-24** (R1 inverted, §17). 112 had two readers
    /// answer `Nested`, the block being drawn in both as a shared instance; a second reader
    /// declares the block now, so a block put back inside one of two readers would be promoted
    /// straight back by the settle that follows — the same frame, and in silence. The refusal
    /// carries the readers so the gesture can name them. The other refusal is a value variable
    /// whose contents are an expression, which is the parameter's matter and lives where the
    /// expression is typed, not here.
    #[test]
    fn putting_back_nests_one_reader_frees_none_and_refuses_two() {
        let mut graph = Graph::empty();
        let shared = ball(&mut graph);
        let kept = ball(&mut graph);
        let alone = ball(&mut graph);
        let both = union(&mut graph, vec![shared, kept]);
        let hollow = shell(&mut graph, shared);
        let (mut doc, part) = settled(graph);
        for id in [shared, kept, alone] {
            doc.node_meta.entry(id).or_default().name = Some(format!("{id}"));
            doc.ownership_nodes.insert(id, Ownership::Variable);
        }

        let mut readers = vec![both, hollow];
        readers.sort();
        assert_eq!(doc.block_readers(BlockKey::Node(shared)), readers, "read by two");
        assert_eq!(
            doc.put_back_target(BlockKey::Node(shared)),
            Err(readers.clone()),
            "so it has to stay a variable, and the refusal says who reads it",
        );
        assert_eq!(doc.put_back_target(BlockKey::Node(kept)), Ok(Ownership::Nested), "by one");
        assert_eq!(doc.put_back_target(BlockKey::Node(alone)), Ok(Ownership::Free), "by none");

        // What it answers is already settled: written, it stays, and the name stays with it. The
        // refused one is left as it is, which is the state the gesture leaves too.
        for id in [kept, alone] {
            let to = doc.put_back_target(BlockKey::Node(id)).expect("one reader or none");
            doc.ownership_nodes.insert(id, to);
        }
        let before = doc.ownership_nodes.clone();
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes, before);
        assert_eq!(doc.pending_ownership_changes, Vec::new());
        assert_eq!(doc.node_meta.get(&kept).and_then(|m| m.name.clone()), Some(format!("{kept}")));

        // A curve read once in each of two parts has two readers, though each part sees one.
        let line = Curve::polyline(
            CurveId::new(),
            "Line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            false,
        )
        .expect("a line");
        let line_id = line.id;
        doc.curves.insert(line_id, line);
        let here = graph_of(&mut doc, part)
            .add(Node::Pipe { curve: line_id, radius: ParamRef::Literal(1.0) });
        let mut there = Graph::empty();
        let elsewhere = there.add(Node::Pipe { curve: line_id, radius: ParamRef::Literal(1.0) });
        doc.add_part(Part {
            id: PartId::new(),
            name: "Other".into(),
            material: None,
            graph: there,
            inputs: Vec::new(),
        });
        let mut readers = vec![here, elsewhere];
        readers.sort();
        assert_eq!(doc.block_readers(BlockKey::Curve(line_id)), readers, "a reader in each part");
        assert_eq!(
            doc.put_back_target(BlockKey::Curve(line_id)),
            Err(readers),
            "so it has to stay a variable, counted across the parts as I2 counts it",
        );
    }

    /// **Settling a second time changes nothing**, from any starting state, over the migration
    /// test's documents.
    ///
    /// Each block starts with no entry, or free, a variable, or nested — every row of the table,
    /// and the states a file or an undo can leave. The first settle has to reach a state the
    /// invariants hold in, and a second must find nothing to move and nothing to say.
    ///
    /// **`Nested` is the one entry the readers can contradict, and the generator seeds it
    /// everywhere so that they do.** The three wrong states this used to seed were `Inside` a real
    /// reader, `Inside` a block that does not read it, and `Inside` one that is gone — all three
    /// distinctions of a stored parent, and all three now spelt `Nested`. What is left to get
    /// wrong is a block stored nested with nothing reading it, which the seeding produces wherever
    /// it lands on a block no block reads, and which the settle has to free.
    #[test]
    fn settling_twice_changes_nothing() {
        let mut repaired = 0usize;
        for seed in 0..400 {
            let Sample { mut doc, .. } = sample(seed);
            let mut rng = Lcg(seed ^ 0x5eed);
            for block in every_block(&doc) {
                let stored = match rng.below(4) {
                    0 => None,
                    1 => Some(Ownership::Free),
                    2 => Some(Ownership::Variable),
                    _ => Some(Ownership::Nested),
                };
                if let Some(stored) = stored {
                    match block {
                        BlockKey::Node(id) => doc.ownership_nodes.insert(id, stored),
                        BlockKey::Curve(id) => doc.ownership_curves.insert(id, stored),
                    };
                }
            }
            let variables: Vec<BlockKey> = every_block(&doc)
                .into_iter()
                .filter(|b| ownership_of(&doc, *b) == Some(Ownership::Variable))
                .collect();
            // The seeded state a settle has to repair: nested with nothing reading it.
            let orphaned: Vec<BlockKey> = every_block(&doc)
                .into_iter()
                .filter(|b| ownership_of(&doc, *b) == Some(Ownership::Nested))
                .filter(|b| readers_in(&doc, *b).is_empty())
                .collect();
            repaired += orphaned.len();

            doc.settle_ownership();
            assert_invariants(&doc, &format!("seed {seed}"));
            for block in &variables {
                assert_eq!(ownership_of(&doc, *block), Some(Ownership::Variable), "seed {seed}");
            }
            for block in &orphaned {
                assert_eq!(
                    ownership_of(&doc, *block),
                    Some(Ownership::Free),
                    "seed {seed}: {block:?} was nested with no reader, so it is a row of its own",
                );
            }
            let (nodes, curves) = (doc.ownership_nodes.clone(), doc.ownership_curves.clone());
            doc.pending_ownership_changes.clear();
            doc.settle_ownership();
            assert_eq!(doc.ownership_nodes, nodes, "seed {seed}: the second settle moved a block");
            assert_eq!(doc.ownership_curves, curves, "seed {seed}: the second moved a curve");
            assert_eq!(doc.pending_ownership_changes, Vec::new(), "seed {seed}");
        }
        // Required to have come up, so a generator that stopped seeding the wrong state would
        // fail here rather than pass by never asking it.
        assert!(repaired >= 10, "nested with no reader came up {repaired} times in 400 documents");
    }

    /// **A reader ring a file holds still leaves every row reachable, and the walk that draws them
    /// ends.**
    ///
    /// **The ring moved from the entry to the reading, and this test moved with it.** It used to
    /// be a ring of `Inside`, repaired in the stored entry by `without_rings`; a stored entry names
    /// no parent since 112, so there is nothing there to close on itself — but the block a row is
    /// drawn in is derived from the readers again, and a ring of *readers*, which nothing in the
    /// application builds and a malformed file can still say, closes in
    /// [`Document::nesting_in`]'s walk instead. Left alone that is not a mark on the list but a
    /// **hang**: no block in the ring reaches a top-level row, and climbing its owners never ends.
    ///
    /// So the repair lives there now (`without_reader_rings`): the lowest block that cannot reach a
    /// top-level row is given a row of its own, which grounds the rest. It is derived, so the
    /// stored entry stays `Nested` — every block in the ring really is read — and a settle after a
    /// reading leaves the entries where they are, which is what keeps the list the same from frame
    /// to frame.
    #[test]
    fn a_reader_ring_a_file_holds_still_leaves_every_row_reachable() {
        let mut graph = Graph::empty();
        let seed = ball(&mut graph);
        let a = shell(&mut graph, seed);
        let b = shell(&mut graph, a);
        let c = shell(&mut graph, b);
        if let Some(Node::Shell { input, .. }) = graph.node_mut(a) {
            *input = c.into();
        }
        let (mut doc, part) = settled(graph);
        assert_invariants(&doc, "a ring of three");
        for id in [a, b, c] {
            assert_eq!(
                doc.ownership_nodes.get(&id),
                Some(&Ownership::Nested),
                "stored nested, since each of the three really is read",
            );
        }
        assert_eq!(doc.ownership_nodes.get(&seed), Some(&Ownership::Free), "nothing reads it now");

        let nesting = doc.nesting_in(&doc.parts[&part].graph);
        let lowest = [a, b, c].into_iter().min().expect("three");
        assert!(
            !nesting.is_nested(lowest),
            "the lowest block that cannot reach a top-level row is given one",
        );
        // Every row reaches a row of its own, in a bounded climb: this is the walk that draws the
        // list, and the bound is what stands in for "it does not hang".
        for start in [seed, a, b, c] {
            let mut front = vec![start];
            for step in 0..=4 {
                front = front.iter().flat_map(|id| nesting.owners_of(*id).to_vec()).collect();
                if front.is_empty() {
                    break;
                }
                assert!(step < 4, "climbing from {start:?} never reaches a row of its own");
            }
        }

        let before = doc.ownership_nodes.clone();
        doc.settle_ownership();
        assert_eq!(doc.ownership_nodes, before, "and a settle after a reading moves nothing");
        assert_eq!(doc.pending_ownership_changes, Vec::new());
    }

    /// **An ownership is written the way the file's entry spells it** (spec §9.1), `Nested` as
    /// `"nested"` since 112.
    ///
    /// **And `{"inside": "<id>"}`, which is format 3.15's spelling, is not read**, though this
    /// module's own doc comment says it is. The derive above knows three unit variants and nothing
    /// maps the old entry onto one of them, so a 3.15 file's ownership fails to deserialize rather
    /// than opening with its rows nested. Asserted as it behaves, not as it should: the back-compat
    /// read is a custom `Deserialize` (or a `#[serde(alias)]`-style shim) that nobody has written,
    /// and when somebody does, this is the line that turns red and becomes the round-trip.
    #[test]
    fn an_ownership_is_written_as_the_spec_spells_it() {
        for (ownership, text) in [
            (Ownership::Free, "\"free\""),
            (Ownership::Variable, "\"variable\""),
            (Ownership::Nested, "\"nested\""),
        ] {
            assert_eq!(serde_json::to_string(&ownership).expect("written"), text);
            assert_eq!(serde_json::from_str::<Ownership>(text).expect("read"), ownership);
        }
        // **3.15's spelling still reads, and the id is thrown away.** Every block of every file a
        // 3.15 build wrote carries `{"inside": "<id>"}`, and the entry is read inside a
        // `deny_unknown_fields` struct with `?` — so refusing the word would not place one block
        // oddly, it would refuse the whole document.
        let parent = NodeId::from_raw(0x00c0_ffee_0000_0001);
        let old = format!("{{\"inside\":\"{parent}\"}}");
        assert_eq!(
            serde_json::from_str::<Ownership>(&old).expect("3.15's spelling reads"),
            Ownership::Nested,
            "a block 3.15 put inside one parent is nested, and its readers say where it is drawn",
        );
        // A word nobody wrote is still refused, so a typo in a hand-edited file is not silently
        // read as a placement.
        assert!(serde_json::from_str::<Ownership>("\"inside\"").is_err());
        assert!(serde_json::from_str::<Ownership>("{\"beside\":\"x\"}").is_err());
    }
}
