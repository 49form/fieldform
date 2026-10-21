// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The echo: every GUI action written into the console as the line that reproduces it
//! (`interaction-spec.md` §6).

use crate::*;

/// What the echo last wrote up to: the document as it stood, and the count of moves it stood at.
pub(crate) struct EchoBase {
    pub(crate) snapshot: echo::Snapshot,
    pub(crate) part: PartId,
    pub(crate) moves: u64,
}

impl App {
    /// The document as it stands, as the point the next echo is written from.
    pub(crate) fn echo_snapshot(&self) -> EchoBase {
        let mut snapshot = echo::Snapshot::of(&self.doc, self.part_graph());
        // What a line calls each unnamed row by, so a rename can say which row it was (§6).
        snapshot.labels = self
            .part_graph()
            .nodes()
            .filter(|(id, _)| !snapshot.names.contains_key(id))
            .map(|(id, _)| (id, self.derived_label(id)))
            .collect();
        snapshot.profile_labels = self.profiles.loops.iter().map(|l| (l[0], self.profile_label(l[0]))).collect();
        EchoBase { snapshot, part: self.active_part, moves: self.document_moves }
    }

    /// Takes the document as it stands as said already: after a typed line, which is its own
    /// record, and after a document is opened or begun.
    pub(crate) fn echo_rebase(&mut self) {
        // Never past a pending session's block: its making is written when it is kept.
        if self.session.is_some() {
            return;
        }
        self.echo_base = self.echo_snapshot();
    }

    /// One echoed line, into the console as if typed, the history and the record a test reads.
    pub(crate) fn echo_line(&mut self, line: String) {
        self.say(format!("> {line}"));
        self.console_history.push(line.clone());
        self.echo_log.push(line);
    }

    /// Writes what changed since the last echo, if anything did: the GUI's edits as the lines
    /// that make them (§6.1, §6.2).
    ///
    /// **Blocks made since are named here, at the moment their line is written** (§15.2): a line
    /// must name what it makes so a later line can say it, and naming at the echo is what lets a
    /// pending session's block go unnamed until it is kept (§3.5). Shallowest first and never by
    /// id ([`echo::new_blocks`]), so the same edits name the same blocks in every run.
    pub(crate) fn echo_settle(&mut self) {
        if self.document_moves == self.echo_base.moves || self.session.is_some() {
            return;
        }
        if self.echo_base.part != self.active_part {
            // Another part is on screen: what changed belongs to a different graph. Said from here.
            self.echo_rebase();
            return;
        }
        let before = self.echo_base.snapshot.graph.clone();
        // Named one at a time, shallowest first: a block's place in the order reads its inputs'
        // names, and those are always shallower, so each is named before it is needed.
        let order = loop {
            let order = echo::new_blocks(&before, &self.doc, self.part_graph());
            let Some(id) = order.iter().copied().find(|id| self.body_name(*id).is_none()) else { break order };
            let word = self.part_graph().node(id).map(names::operation_word).unwrap_or("Block");
            let name = names::instance_name(&self.doc, word);
            self.doc.node_meta.entry(id).or_default().name = Some(name);
        };
        // Renames first: every line after them calls a row by its new name.
        let mut lines = echo::rename_lines(&self.echo_base.snapshot, &self.echo_snapshot().snapshot);
        lines.extend(echo::lines(&self.echo_base.snapshot, &self.doc, self.part_graph(), &order));
        for line in lines {
            self.echo_line(line);
        }
        self.echo_rebase();
    }

    /// Writes the echo once nothing is half-done: no button held, nothing being typed into, no
    /// drag or placement mid-way. A drag is one edit and is written once, when the hand comes off.
    pub(crate) fn echo_when_quiet(&mut self, ctx: &egui::Context) {
        // A pending session is written when it is kept, with the values it was kept with.
        if self.document_moves == self.echo_base.moves || self.script_running || self.session.is_some() {
            return;
        }
        let busy = ctx.input(|i| i.pointer.any_down())
            || ctx.egui_wants_keyboard_input()
            || self.move_state.is_some()
            || self.gizmo_drag.is_some()
            || self.point_drag.is_some()
            || self.marquee.is_some()
            || self.curve_edit_before.is_some()
            || self.weight_edit_before.is_some();
        if !busy {
            self.echo_settle();
        }
    }

    /// `set Sphere_0.radius=12 Sphere_0.centre=0,0,10`: a block's own keys, by the words its
    /// create verb reads (`block_edit`), as one edit. What the echo writes for an edit, and what a
    /// person types to make one.
    pub(crate) fn set_block_keys(&mut self, id: NodeId, args: &Args) -> Result<(), String> {
        self.refresh_profiles();
        let node = self.part_graph().node(id).cloned().ok_or("that block is gone")?;
        let edited = {
            let lookup = |t: &str| self.lookup(t);
            let curve = |t: &str| self.curve_named(t);
            let profile = |t: &str| {
                self.curve_named(t).or_else(|| {
                    self.profiles.loops.iter().map(|l| l[0]).find(|key| self.profile_label(*key).eq_ignore_ascii_case(t.trim()))
                })
            };
            let inside = |n: &Node, c: CurveId| self.inside_of(n, c);
            let offered = |c: &LatticeCell| fieldform_field::CellKind::from(*c).supported_bandings();
            let seed = |b: Banding| seeded_phase(b, DEFAULT_BODY_MM * 0.02);
            let cx = block_edit::Context {
                lookup: &lookup,
                curve: &curve,
                profile: &profile,
                inside: &inside,
                offered: &offered,
                seed: &seed,
            };
            block_edit::node_with_keys(node, args, &cx)?
        };
        self.replace_node(id, edited);
        Ok(())
    }
}
