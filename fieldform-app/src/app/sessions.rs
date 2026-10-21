// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A tool session: the block a command makes, pending until it is kept or taken back
//! (`interaction-spec.md` §3).
//!
//! **No Apply, and nothing to be stuck in.** A session command makes its block at once, with the
//! values it was last kept with or its defaults, and the block is live in the viewport and the
//! notebook. Edits to it fold into one undo entry with its making. Enter, a click on empty space,
//! selecting something else or running any other command keeps it; Esc takes it back as though it
//! had never been made -- the document, its names and the history exactly as they were (T3).

use crate::*;

/// The keys a command's last kept values are *not* remembered for: what it is made from. Those
/// come from the selection each time, or a session would build on the last session's operand.
use super::tool_panel::MADE_FROM;

/// What a pending session needs to be kept or taken back.
pub(crate) struct Session {
    pub(crate) command: &'static commands::Command,
    /// The block it made, which the panel shows and edits.
    pub(crate) block: NodeId,
    /// The document as the session found it: a cancel puts this back whole, which is exact where
    /// undoing each entry in turn could only be as exact as each entry's undo.
    doc_before: Document,
    mark: fieldform_ops::StackMark,
    selected_before: (Option<NodeId>, Option<CurveId>, Option<outputs::BlockRef>),
    view_mode_before: ViewMode,
    /// Its panel's own state: the row in focus, what is being typed, a scrub (§4).
    pub(crate) panel: super::tool_panel::Panel,
    /// The block's rows in the list are to open with the keyboard in their first number, which
    /// is what a session shows when the floating panel is off. Cleared once done; Tab with
    /// nothing focused sets it again.
    pub(crate) awaiting: bool,
}

impl App {
    /// Runs a session command from the interface: makes its block pending, or -- where it makes
    /// none, as the curve tool begins clicking or a refused command does nothing -- behaves as an
    /// immediate command would.
    pub(crate) fn begin_session(&mut self, command: &'static commands::Command) {
        let before: std::collections::HashSet<NodeId> = self.part_graph().nodes().map(|(id, _)| id).collect();
        let doc_before = self.doc.clone();
        let mark = self.stack.mark();
        let selected_before = (self.selected_body, self.selected_curve, self.selected_block);
        let view_mode_before = self.view_mode;
        let moves = self.document_moves;
        // The values it was last kept with, this session of the application (§3.2), over its own
        // fixed arguments; its defaults otherwise.
        let mut args = command.fixed_args();
        if let Some(last) = self.last_values.get(command.id) {
            for (k, v) in last.iter() {
                if args.get(k).is_none() {
                    args = args.with(k, v.clone());
                }
            }
        }
        // Its own entry, never the next instant of a gesture before it.
        self.gesture = None;
        self.dispatch(command.op, &args);
        let made: Vec<NodeId> = self
            .part_graph()
            .nodes()
            .map(|(id, _)| id)
            .filter(|id| !before.contains(id) && !echo::is_joint(&self.doc, self.part_graph(), *id))
            .collect();
        let block = self.selected_body.filter(|id| made.contains(id)).or_else(|| made.last().copied());
        let Some(block) = block else {
            // Nothing pending. The redo list the mark set aside goes back if nothing was recorded.
            // A tool that began picking -- the curve tool -- has changed nothing yet, so there is
            // nothing to write: each curve it places is written when it lands. One that changed the
            // document without making a block is written as an immediate command is.
            if self.document_moves == moves {
                self.stack.cancel_to(mark);
            } else {
                drop(mark);
                // What no line spells -- a Tessellate block is an output, not a block of the graph
                // -- is written as the command that made it, as `invoke_command` does.
                let written = self.echo_log.len();
                self.echo_settle();
                if self.echo_log.len() == written {
                    self.echo_line(command.console_line());
                    self.echo_rebase();
                }
            }
            return;
        };
        // The pending block is the selection, so its panel is the one showing and a click on
        // anything else is what keeps it.
        if self.selected_body != Some(block) {
            self.select_body(Some(block));
        }
        self.session = Some(Session {
            command,
            block,
            doc_before,
            mark,
            selected_before,
            view_mode_before,
            panel: Default::default(),
            awaiting: true,
        });
    }

    /// Keeps the pending session, if there is one: one undo entry for its making and every edit
    /// since, its values remembered for next time, and its line written (§3.4).
    pub(crate) fn commit_session(&mut self) {
        let Some(session) = self.session.take() else { return };
        self.stack.keep_since(session.mark, session.command.display_name.trim_end_matches("..."));
        // Kept, so it is what Shift+R runs next (§8).
        self.last_command = Some(session.command);
        // The next edit is an entry of its own, not a continuation of the session's last.
        self.gesture = None;
        if let Some(node) = self.part_graph().node(session.block).cloned()
            && let Ok(written) = echo::keys_of(&node, &echo::Names { doc: &self.doc })
        {
            let remembered = written
                .keys
                .iter()
                .filter(|(k, _)| !MADE_FROM.contains(k) && *k != "wrap")
                .fold(Args::new(), |a, (k, v)| a.with(k, console::parse_value(v)));
            self.last_values.insert(session.command.id, remembered);
        }
        self.echo_settle();
    }

    /// Takes the pending session back, if there is one: the document, its names and the history as
    /// they were before it began, and the selection with them (§3.5).
    pub(crate) fn cancel_session(&mut self) {
        let Some(session) = self.session.take() else { return };
        self.doc = session.doc_before;
        self.stack.cancel_to(session.mark);
        self.gesture = None;
        self.material_gesture = None;
        let (body, curve, block) = session.selected_before;
        self.selected_body = body;
        self.selected_curve = curve;
        self.selected_block = block;
        self.view_mode = session.view_mode_before;
        self.after_document_change();
        // Nothing happened, so the echo has nothing to say about it.
        self.echo_base.moves = self.document_moves;
        self.say(format!("{} taken back", session.command.display_name.trim_end_matches("...")));
    }

    /// Whether `id` is the block a session is waiting on.
    pub(crate) fn is_pending(&self, id: NodeId) -> bool {
        self.session.as_ref().is_some_and(|s| s.block == id)
    }

    /// Keeps the pending session once its block is no longer what is selected: a click on empty
    /// space, or on something else (§3.4). Asked at the end of a frame, when nothing is mid-drag.
    pub(crate) fn keep_when_selection_moves(&mut self, ctx: &egui::Context) {
        let Some(session) = &self.session else { return };
        let moved = self.selected_body != Some(session.block);
        let busy = ctx.input(|i| i.pointer.any_down()) || self.gizmo_drag.is_some() || self.move_state.is_some();
        if moved && !busy {
            self.commit_session();
        }
    }
}
