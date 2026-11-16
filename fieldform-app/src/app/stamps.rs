// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Creation stamps, which are what order the things in a document.

use crate::*;

impl App {
    /// The clock the stamps are taken from, in microseconds since the epoch.
    ///
    /// Microseconds because a stamp two nodes share orders them by nothing — see
    /// `Document::created`, which carries the whole argument and the format consequence.
    pub(crate) fn stamp_clock() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    /// Records when anything new was made, so the block list can be in the order you made it.
    ///
    /// Done here rather than at each of the half-dozen places that add a node, because this is
    /// the one gate every graph edit already passes through — and a stamp that has to be
    /// remembered at each call site is a stamp that will be forgotten at the next one.
    ///
    /// Existing entries are never overwritten, so a node keeps the moment it was made however
    /// often the graph around it is rewritten. That is also what lets a path with an order of its
    /// own declare it first, through [`Self::stamp_in_order`], and leave this to fill the rest.
    ///
    /// **A distinct stamp per node, not one reading shared by the batch.** This used to write one
    /// clock reading to every fresh node, and bodies that tie fall back to their ids — which is
    /// the random order the whole arrangement exists to replace, measured at the same rate by
    /// `fieldform-gpu/examples/body_order_probe.rs`. A write that adds one node is the common
    /// case and is unaffected; a write that adds several used to be silently ordered by nothing.
    ///
    /// The batch is sorted by id before the offsets are handed out, which is arbitrary and is
    /// *not* nothing: the alternative is `HashMap` order, which is randomly seeded per process,
    /// so the same edit made twice would stamp two different orders and neither would be
    /// reproducible in a test. Arbitrary-but-fixed is what this can promise on its own; a path
    /// that knows better says so through [`Self::stamp_in_order`].
    pub(crate) fn stamp_new_nodes(&mut self) {
        let now = Self::stamp_clock();
        let mut fresh: Vec<NodeId> = self
            .part_graph()
            .nodes()
            .map(|(id, _)| id)
            .filter(|id| !self.doc.created.contains_key(id))
            .collect();
        fresh.sort_by_key(|id| id.raw());
        let stamps = stamps_from(now, fresh.len());
        for (id, at) in fresh.into_iter().zip(stamps) {
            self.doc.created.insert(id, at);
        }
        // Curves as well, and by the same clock. A curve is a block in the same list as the
        // bodies, so an import that arrives without a time sorts against them by its random id
        // — which is how an imported curve landed at the top of a tree it had just been added
        // to the bottom of.
        let mut fresh: Vec<CurveId> = self
            .doc
            .curves
            .keys()
            .copied()
            .filter(|id| !self.doc.curves_created.contains_key(id))
            .collect();
        fresh.sort_by_key(|id| id.raw());
        let stamps = stamps_from(now, fresh.len());
        for (id, at) in fresh.into_iter().zip(stamps) {
            self.doc.curves_created.insert(id, at);
        }
    }

    /// Stamps nodes in an order the caller knows and the graph does not.
    ///
    /// **For the one kind of path that has an order to state.** An import builds an ordered list
    /// and numbers it — `file (1)`, `file (2)` — and a re-import pairs the solids up *by that
    /// position* (`cad-import-block-spec.md` §4). The gate above cannot recover that: it reads
    /// the batch out of a `HashMap`, so it can promise each node a distinct stamp and nothing
    /// about their order. Called before the edit settles, this writes the order the file had, and
    /// the gate then fills in anything not named here.
    ///
    /// Not a duty spread across every call site, which is the thing `stamp_new_nodes`'s own
    /// comment rightly warns against: a path that forgets to call this still gets stamped, just
    /// arbitrarily, so forgetting costs an order rather than a stamp. The half-dozen paths with
    /// no order to state should keep saying nothing.
    ///
    /// Never overwrites, for the same reason the gate does not: a node keeps the moment it was
    /// made however often the graph around it is rewritten.
    pub(crate) fn stamp_in_order(&mut self, ids: &[NodeId]) {
        let now = Self::stamp_clock();
        for (id, at) in ids.iter().zip(stamps_from(now, ids.len())) {
            self.doc.created.entry(*id).or_insert(at);
        }
    }

    /// The same, for curves — an IGES import brings them in an order too.
    pub(crate) fn stamp_curves_in_order(&mut self, ids: &[CurveId]) {
        let now = Self::stamp_clock();
        for (id, at) in ids.iter().zip(stamps_from(now, ids.len())) {
            self.doc.curves_created.entry(*id).or_insert(at);
        }
    }
}
