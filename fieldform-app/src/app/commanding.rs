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
                || self.selected_block.is_some()
                || self.section_selected.is_some(),
            clipboard: self.clipboard.is_some(),
            clipboard_curve: matches!(self.clipboard, Some(Clipboard::Curve(_))),
            bodies: self.part_graph().roots().len(),
            field_drawn: self.field_is_drawn(),
            cage_control: self.control == ControlMesh::Cage,
            section_on: self.sections().iter().any(|s| s.on),
            occlusion_possible: self.view.quality.occlusion(),
            reflections_possible: self.view.quality.reflections(),
            control_points: self.curve_ui.show_control_points,
            view_cube: self.view.show_view_cube,
            allow_agent: self.agent_link.allowed,
            agent_launched: self
                .agent_link.host
                .as_ref()
                .is_some_and(|h| h.mode == crate::agent::server::HostMode::Launch),
            floating_panel: self.tools.floating_panel,
            idle_occlusion: self.view.idle_occlusion,
            idle_reflections: self.view.idle_reflections,
            draft_on: self.view.draft.enabled,
            grid_on: self.view.show_grid,
            graph_shown: self.doc.graph_layout.shown,
            plane_selected: self.selected_plane().is_some(),
            curve_selected: self.selected_curve.is_some(),
            sketching: self.sketching(),
            view_mode: match self.view.mode {
                ViewMode::Polygon => "polygon",
                ViewMode::Field => "field",
                ViewMode::Overlay => "overlay",
            },
            quality: self.view.quality.label().to_lowercase(),
            cage_detail: self.mesh_resolution,
        }
    }

    /// The command Shift+R would run (§8).
    pub(crate) fn last_repeatable(&self) -> Option<&'static commands::Command> {
        self.tools.last_command
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
        // A join's preview is put back before anything runs, so a key pressed with the list open
        // -- undo, above all -- acts on the document and not on a preview of it.
        self.end_join_preview();
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
            // A pending extrude or inset is kept the same way -- which is how E again chains:
            // the cap is kept, and the next lift starts from it.
            self.keep_cage_session();
        }
        // **And puts an armed cage tool down** (R7), unless it only changes how the document is
        // looked at; a tool arming itself again is the tool, not another command.
        if command.kind != commands::Kind::Tool && command.keeps_session() {
            self.disarm_cage_tool();
        }
        self.note_recent(command.id);
        self.echo_settle();
        if command.kind == commands::Kind::Session {
            self.begin_session(command);
            return;
        }
        // A cage tool arms and waits for clicks, each its own line (`app/cage_tools.rs`).
        if command.kind == commands::Kind::Tool {
            self.arm_cage_tool(command);
            return;
        }
        let moves = self.document_moves;
        let warned = self.console.warnings;
        self.console.cage_line = None;
        self.dispatch(command.op, &command.fixed_args());
        // A command that was refused did nothing, and a line that does nothing is not a record of
        // anything: it would be refused again on replay, which T1 found.
        if self.document_moves == moves && self.console.warnings > warned {
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
                self.tools.last_command = Some(command);
            }
        }
        let replaces = matches!(command.op, "file.new" | "file.open" | "file.import" | "file.import_obj" | "file.import_iges" | "file.import_step");
        if replaces || self.document_moves == moves {
            self.echo_line(command.console_line());
            self.echo_rebase();
        } else {
            // A change the lines cannot spell -- an eye turned over -- is written as the command
            // that made it, so the record is never silent about an action (§6.1).
            let written = self.console.echo_log.len();
            self.echo_settle();
            if self.console.echo_log.len() == written && self.tools.session.is_none() {
                // A cage verb is written with the elements it read from the selection, which its
                // bare word would not say.
                let line = self.console.cage_line.take().unwrap_or_else(|| command.console_line());
                self.echo_line(line);
            }
        }
    }
}
