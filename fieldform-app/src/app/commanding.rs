// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The one door every surface runs a command through, and what it asks of the moment first.

use crate::*;

impl App {
    /// What the registry's questions are answered from, now (`commands::Ctx`).
    pub(crate) fn command_context(&self) -> commands::Ctx {
        commands::Ctx {
            undo: self.stack.peek_undo().map(str::to_string),
            redo: self.stack.peek_redo().map(str::to_string),
            selected: self.selected_body.is_some()
                || self.selected_curve.is_some()
                || self.selected_block.is_some(),
            clipboard: self.clipboard.is_some(),
            clipboard_curve: matches!(self.clipboard, Some(Clipboard::Curve(_))),
            bodies: self.part_graph().roots().len(),
            pair_candidates: self.pair_candidates().len(),
            field_drawn: self.field_is_drawn(),
            cage_control: self.control == ControlMesh::Cage,
            section_on: self.section.on,
            occlusion_possible: self.display_quality.occlusion(),
            control_points: self.show_control_points,
            view_cube: self.show_view_cube,
            floating_panel: self.floating_panel,
            idle_occlusion: self.idle_occlusion,
            draft_on: self.draft.enabled,
            grid_on: self.show_grid,
            view_mode: match self.view_mode {
                ViewMode::Polygon => "polygon",
                ViewMode::Field => "field",
                ViewMode::Overlay => "overlay",
            },
            quality: self.display_quality.label().to_lowercase(),
        }
    }

    /// The command Shift+R would run (§8).
    pub(crate) fn last_repeatable(&self) -> Option<&'static commands::Command> {
        self.last_command
    }

    /// Runs `command` again: through the one door, so it keeps what is pending, starts a session
    /// from the values it was last kept with, and works on the selection as it is now (§8).
    pub(crate) fn repeat_last(&mut self, command: &'static commands::Command) {
        self.invoke_command(command);
    }

    /// What a key runs: its registry command through the one door, so a key keeps a pending
    /// session and is echoed exactly as the menu row it shares an operation with; an operation the
    /// registry does not list -- a binding in someone's file to an operation without an entry --
    /// is dispatched as it is.
    pub(crate) fn run_binding(&mut self, op: &str, args: &Args) {
        match commands::for_binding(op, args) {
            Some(command) => self.invoke_command(command),
            None => self.dispatch(op, args),
        }
    }

    /// Runs a registry command from the interface: a menu row, the palette, the marking menu, the
    /// shelf or a key.
    ///
    /// **One door, so every surface behaves alike** (`interaction-spec.md` §2.2). What a command
    /// does is `dispatch`'s; what the interface owes around it — that it was on offer, and later
    /// the pending session and the echo — is said once, here, rather than by each surface.
    pub(crate) fn invoke_command(&mut self, command: &'static commands::Command) {
        if let Err(why) = (command.context)(&self.command_context()) {
            self.warn(format!("{} is not available — {why}", command.display_name));
            return;
        }
        // Repeat runs another command through this same door, which keeps, echoes and notes it;
        // its own word would replay as "the last line again", which is not what was done.
        if command.op == "edit.repeat" {
            self.dispatch(command.op, &command.fixed_args());
            return;
        }
        // **Running anything keeps the session pending first** (§3.4, T5), so at most one is ever
        // pending; then what the GUI changed before this command is written, in the order it
        // happened.
        if command.keeps_session() {
            self.commit_session();
        }
        self.note_recent(command.id);
        self.echo_settle();
        if command.kind == commands::Kind::Session {
            self.begin_session(command);
            return;
        }
        let moves = self.document_moves;
        let warned = self.console_warnings;
        self.dispatch(command.op, &command.fixed_args());
        // A command that was refused did nothing, and a line that does nothing is not a record of
        // anything: it would be refused again on replay, which T1 found.
        if self.document_moves == moves && self.console_warnings > warned {
            return;
        }
        // Rename from a menu or a key only opens the row's name for typing: the name typed is the
        // edit, and is written as a `rename` line when it lands.
        if command.op == "block.rename" {
            return;
        }
        // **What it changed in the document is written as the lines that make the change**, not
        // as the command's word: replaying `undo` would take back one typed line where the GUI
        // took back one gesture, and a block made with defaults is written with every number it
        // ended up with. A command that changes how the document is *looked at* has nothing in the
        // document to write, so its own line is the record; and a file opened or begun replaces
        // the document, so its line is the record of that too.
        // What Shift+R runs next (§8): the last command that did
        // something, other than opening a window, the palette or the console's own words.
        if command.keeps_session() || command.op.starts_with("view.") {
            if !command.op.starts_with("help.") && !command.op.starts_with("console.") && !command.op.starts_with("palette.") {
                self.last_command = Some(command);
            }
        }
        let replaces = matches!(command.op, "file.new" | "file.open" | "file.import" | "file.import_obj" | "file.import_iges" | "file.import_step");
        if replaces || self.document_moves == moves {
            self.echo_line(command.console_line());
            self.echo_rebase();
        } else {
            // A change the lines cannot spell -- an eye turned over -- is written as the command
            // that made it, so the record is never silent about an action (§6.1).
            let written = self.echo_log.len();
            self.echo_settle();
            if self.echo_log.len() == written && self.session.is_none() {
                self.echo_line(command.console_line());
            }
        }
    }
}
