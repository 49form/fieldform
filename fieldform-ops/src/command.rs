// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Reversible operations and the undo/redo stack.

use crate::Document;

#[derive(thiserror::Error, Debug)]
pub enum OpError {
    #[error("unknown operation \"{0}\"")]
    UnknownOp(String),
    #[error("operation \"{op}\" is missing required argument \"{arg}\"")]
    MissingArg { op: String, arg: String },
    #[error("operation \"{op}\": argument \"{arg}\" has the wrong type ({detail})")]
    BadArg {
        op: String,
        arg: String,
        detail: String,
    },
    #[error("operation \"{op}\" failed: {detail}")]
    Failed { op: String, detail: String },
}

/// A single reversible edit.
///
/// `undo` must restore exactly the state `execute` found, and `execute` must be safe to run
/// again after `undo` (redo replays it). Commands therefore store whatever "before" state
/// they need at construction or execution time, rather than recomputing it — recomputing
/// would read an already-modified document and restore the wrong thing.
pub trait Command: Send {
    /// Operation name, e.g. `"cage.translate_vertices"`. Shown in undo history and used by
    /// the registry and keymap.
    fn name(&self) -> &str;

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError>;
    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError>;

    /// What this entry holds, in bytes, for the stack's budget. Nought for the ordinary command,
    /// whose before-state is a few numbers; a cage edit carries two copies of the mesh and says
    /// so, since a thousand of those is what the budget is for.
    fn byte_size(&self) -> usize {
        0
    }
}

/// Undo/redo history.
///
/// **Two limits, and the oldest entry goes when either is hit.** The count kept a drag from
/// filling the history with its frames; it says nothing about size, and a cage edit holds a
/// snapshot of the mesh before and after -- about a megabyte each at the largest cage measured --
/// so 256 of those would be half a gigabyte with nothing to say so. The byte budget is what says
/// so.
pub struct CommandStack {
    done: Vec<Box<dyn Command>>,
    undone: Vec<Box<dyn Command>>,
    limit: usize,
    byte_limit: usize,
}

/// The default byte budget for the history.
pub const DEFAULT_BYTE_LIMIT: usize = 512 << 20;

impl Default for CommandStack {
    fn default() -> Self {
        Self::new(256)
    }
}

impl CommandStack {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            done: Vec::new(),
            undone: Vec::new(),
            limit,
            byte_limit: DEFAULT_BYTE_LIMIT,
        }
    }

    /// The same history under another byte budget -- for a test that wants to see an entry go.
    pub fn with_byte_limit(mut self, bytes: usize) -> Self {
        self.byte_limit = bytes;
        self
    }

    /// What the entries that can be undone hold between them.
    pub fn bytes_held(&self) -> usize {
        self.done.iter().map(|c| c.byte_size()).sum()
    }

    /// Runs a command and records it for undo.
    ///
    /// A newly executed command invalidates the redo history — the classic linear-undo
    /// model: once you do something new, the branch you'd undone is gone.
    pub fn execute(&mut self, doc: &mut Document, mut cmd: Box<dyn Command>) -> Result<(), OpError> {
        cmd.execute(doc)?;
        self.undone.clear();
        self.done.push(cmd);
        if self.done.len() > self.limit {
            self.done.remove(0);
        }
        // The newest entry always stays, however large: a budget that dropped the edit just made
        // would be an undo that does nothing, which is worse than a budget exceeded once.
        while self.done.len() > 1 && self.bytes_held() > self.byte_limit {
            self.done.remove(0);
        }
        Ok(())
    }

    /// Runs a command *in place of* the most recent one, for a gesture still in progress.
    ///
    /// A drag is one act and belongs on the stack once, not once per frame. The caller keeps the
    /// state the gesture started from and hands over a command describing the whole of it so
    /// far; this drops the entry that described the previous instant of the same gesture and
    /// records the new one.
    ///
    /// The dropped command is **not undone** — the document already holds its effect, and the
    /// replacement's own "before" reaches back past it to where the gesture started. Undoing it
    /// first would rewind the document to that point and then re-apply, which is visible as a
    /// flicker and pointless besides.
    ///
    /// So the replacement must overwrite everything the dropped one did. A graph edit's settle is
    /// the case that needed saying, since a settle never changes a variable: a replacement given
    /// the ownership its gesture started from settles from that, not from what it finds
    /// ([`crate::SetPartGraph`]).
    ///
    /// Falls back to a plain execute when there is nothing to replace, so a caller need not
    /// check whether its gesture is the first thing in the history.
    pub fn replace_top(
        &mut self,
        doc: &mut Document,
        cmd: Box<dyn Command>,
    ) -> Result<(), OpError> {
        self.done.pop();
        self.execute(doc, cmd)
    }

    /// Undoes the most recent command. Returns its name, or `None` if there was nothing to
    /// undo.
    pub fn undo(&mut self, doc: &mut Document) -> Option<String> {
        let mut cmd = self.done.pop()?;
        if cmd.undo(doc).is_err() {
            // A command that can't undo cleanly would leave the document in a state its
            // neighbours on the stack no longer describe, so drop the whole history rather
            // than let later undos corrupt things further.
            self.done.clear();
            self.undone.clear();
            return None;
        }
        let name = cmd.name().to_string();
        self.undone.push(cmd);
        Some(name)
    }

    /// Redoes the most recently undone command.
    pub fn redo(&mut self, doc: &mut Document) -> Option<String> {
        let mut cmd = self.undone.pop()?;
        if cmd.execute(doc).is_err() {
            self.undone.clear();
            return None;
        }
        let name = cmd.name().to_string();
        self.done.push(cmd);
        Some(name)
    }

    pub fn can_undo(&self) -> bool {
        !self.done.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.undone.is_empty()
    }

    /// Name of the command the next undo would reverse, for UI labels.
    pub fn peek_undo(&self) -> Option<&str> {
        self.done.last().map(|c| c.name())
    }

    pub fn peek_redo(&self) -> Option<&str> {
        self.undone.last().map(|c| c.name())
    }

    /// Names of the commands that can be undone, most recent first.
    ///
    /// Written for a browsable history, which the application's Edit menu listed once and then
    /// dropped on purpose: the menu grew a row per edit and ran off the screen, and "undo to
    /// here" is Ctrl+Z held down. The note where that menu is built records the ruling. So only
    /// the tests read this now, and only the tests compile it. `pub(crate)` rather than private
    /// because the test that reads it lives in `ops`, not here.
    #[cfg(test)]
    pub(crate) fn undo_names(&self) -> impl DoubleEndedIterator<Item = &str> {
        self.done.iter().rev().map(|c| c.name())
    }

    /// Names of the commands that can be redone, next-to-redo first. Test-only for the same
    /// reason as `undo_names`.
    #[cfg(test)]
    pub(crate) fn redo_names(&self) -> impl DoubleEndedIterator<Item = &str> {
        self.undone.iter().rev().map(|c| c.name())
    }

    /// Discards all history, e.g. after loading a new document.
    pub fn clear(&mut self) {
        self.done.clear();
        self.undone.clear();
    }

    /// Where the history stands, for a tool session that will later be kept as one entry or taken
    /// back without trace (`interaction-spec.md` §3).
    ///
    /// **The redo list is set aside, not cleared.** The session's first edit would clear it, as
    /// any new edit does; a cancel has to put it back, since a cancel is defined as leaving the
    /// history exactly as it was found (T3). It is handed back by [`Self::cancel_to`] and dropped
    /// by [`Self::keep_since`].
    pub fn mark(&mut self) -> StackMark {
        StackMark { done: self.done.len(), undone: std::mem::take(&mut self.undone) }
    }

    /// Every entry since `mark`, folded into one named entry: the session kept (§3.4). Nothing is
    /// executed again -- the document already holds each entry's effect -- and an undo of the one
    /// entry takes back all of them, newest first. `false` when nothing was recorded since.
    pub fn keep_since(&mut self, mark: StackMark, name: impl Into<String>) -> bool {
        let from = mark.done.min(self.done.len());
        let commands: Vec<Box<dyn Command>> = self.done.drain(from..).collect();
        if commands.is_empty() {
            return false;
        }
        self.done.push(Box::new(crate::Batch::new(name, commands)));
        true
    }

    /// Drops every entry since `mark` without undoing it, and puts back the redo list the mark
    /// set aside: the session taken back (§3.5). The caller restores the document itself, from
    /// the copy it took when the session began, which is exact where undoing each entry in turn
    /// could only be as exact as each entry's own undo.
    pub fn cancel_to(&mut self, mark: StackMark) {
        self.done.truncate(mark.done.min(self.done.len()));
        self.undone = mark.undone;
    }

    /// The names of the entries that can be undone, oldest first, and of those that can be
    /// redone, next first -- what a test compares to say the history is as it was.
    pub fn history(&self) -> (Vec<String>, Vec<String>) {
        (
            self.done.iter().map(|c| c.name().to_string()).collect(),
            self.undone.iter().rev().map(|c| c.name().to_string()).collect(),
        )
    }
}

/// The history as a tool session found it: how many entries it held, and its redo list, set
/// aside until the session is kept or taken back.
pub struct StackMark {
    done: usize,
    undone: Vec<Box<dyn Command>>,
}

impl StackMark {
    /// How many entries the history held when the mark was taken.
    pub fn depth(&self) -> usize {
        self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Arc;

    /// Adds `by` to a shared number, so a test can read what the history has done.
    struct Add {
        name: &'static str,
        by: i64,
        to: Arc<AtomicI64>,
    }

    impl Command for Add {
        fn name(&self) -> &str {
            self.name
        }
        fn execute(&mut self, _doc: &mut Document) -> Result<(), OpError> {
            self.to.fetch_add(self.by, Ordering::SeqCst);
            Ok(())
        }
        fn undo(&mut self, _doc: &mut Document) -> Result<(), OpError> {
            self.to.fetch_sub(self.by, Ordering::SeqCst);
            Ok(())
        }
    }

    fn add(name: &'static str, by: i64, to: &Arc<AtomicI64>) -> Box<dyn Command> {
        Box::new(Add { name, by, to: Arc::clone(to) })
    }

    /// **A kept session is one entry, whatever it recorded** (T4): undo takes all of it back,
    /// newest first, and redo puts all of it back.
    #[test]
    fn a_kept_session_is_one_entry_that_undoes_and_redoes_whole() {
        let n = Arc::new(AtomicI64::new(0));
        let mut doc = Document::new();
        let mut stack = CommandStack::default();
        stack.execute(&mut doc, add("before", 1, &n)).unwrap();
        let mark = stack.mark();
        for by in [10, 100, 1000] {
            stack.execute(&mut doc, add("edit", by, &n)).unwrap();
        }
        assert!(stack.keep_since(mark, "create.sphere"));
        assert_eq!(stack.history().0, ["before", "create.sphere"]);
        assert_eq!(n.load(Ordering::SeqCst), 1111, "kept, not run again");
        assert_eq!(stack.undo(&mut doc).as_deref(), Some("create.sphere"));
        assert_eq!(n.load(Ordering::SeqCst), 1);
        assert_eq!(stack.redo(&mut doc).as_deref(), Some("create.sphere"));
        assert_eq!(n.load(Ordering::SeqCst), 1111);
    }

    /// **A cancelled session leaves the history as it found it** (T3): its entries are gone and
    /// the redo list its first edit would have cleared is back.
    #[test]
    fn a_cancelled_session_leaves_the_history_as_it_was_found() {
        let n = Arc::new(AtomicI64::new(0));
        let mut doc = Document::new();
        let mut stack = CommandStack::default();
        stack.execute(&mut doc, add("a", 1, &n)).unwrap();
        stack.execute(&mut doc, add("b", 2, &n)).unwrap();
        stack.undo(&mut doc);
        let before = stack.history();
        assert_eq!(before, (vec!["a".to_string()], vec!["b".to_string()]));
        let mark = stack.mark();
        stack.execute(&mut doc, add("pending", 50, &n)).unwrap();
        assert!(!stack.can_redo(), "a new edit clears redo, as always");
        stack.cancel_to(mark);
        assert_eq!(stack.history(), before, "the same entries, and redo back");
        assert_eq!(stack.redo(&mut doc).as_deref(), Some("b"));
    }

    /// Nothing recorded is nothing kept: no empty entry appears.
    #[test]
    fn a_session_that_recorded_nothing_keeps_nothing() {
        let mut stack = CommandStack::default();
        let mark = stack.mark();
        assert!(!stack.keep_since(mark, "create.sphere"));
        assert!(!stack.can_undo());
    }
}
