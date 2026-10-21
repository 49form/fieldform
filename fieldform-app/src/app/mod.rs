// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! App's methods, by what they serve: each module adds an `impl App` of its own to `main.rs`'s.

mod document_io;
mod import;
mod console;
mod menu;
// What every surface that runs a command shares: the context it is offered in, and the one door.
mod commanding;
// The GUI's edits written as the console lines that make them.
pub(crate) mod echoing;
// A command's block, pending until it is kept or taken back.
pub(crate) mod sessions;
// Its numbers, typed or scrubbed in place.
pub(crate) mod tool_panel;
// Space, a few letters, Enter.
pub(crate) mod palette_ui;
// Every command's keys, rebound by pressing them.
mod keys_panel;
// Right-click: eight directions, fixed per kind of thing selected.
pub(crate) mod marking_ui;
// A person's own strip of commands, and the hint an empty document shows.
pub(crate) mod shelf;
mod block_list;
mod block_controls;
mod frame_controls;
mod operations;
mod curves;
mod manipulator;
mod measure;
// The one rule the three statements about the part ask before making one.
pub(crate) use measure::no_statement_while_a_slot_is_empty;
pub(crate) mod inspector;
