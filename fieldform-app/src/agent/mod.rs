// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The agent host: what an MCP client's tool calls do to the application
//! (`mcp-server-spec.md` §3–§5). This is the protocol-free core — each method takes plain data
//! and returns plain data — so it is held by tests on a headless `App` before any transport
//! carries a request to it.
//!
//! **Every edit is a console line** run through `run_line` (P1). A call is one undo entry (P2)
//! and, by default, all or nothing (P3): the document is restored from the copy taken at the
//! mark, exactly as a cancelled tool session is, and the history is left as found.

use super::*;
use crate::outcome::{LineError, LineOutcome};

pub(crate) mod paths;
pub(crate) mod read;
pub(crate) mod resources;
pub(crate) mod snapshot;
pub(crate) mod script;
pub(crate) mod server;
pub(crate) mod tools;

/// Most lines one call may carry.
pub(crate) const MAX_LINES: usize = 500;

/// The state an agent call needs between calls.
#[derive(Default)]
pub(crate) struct AgentState {
    /// Set while a call runs: every file a line names goes through [`paths::resolve`].
    pub(crate) guard: Option<Guard>,
    /// The history depth right after the last entry an agent call kept. `undo` takes back only
    /// an entry still at this depth unless forced (T10).
    pub(crate) depth: Option<usize>,
    /// Where files may be read and written. The first is what a relative path means.
    pub(crate) roots: Vec<std::path::PathBuf>,
    /// Set by a client's cancellation (or Escape); checked between lines.
    pub(crate) cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The `snapshot` call waiting for its picture, if one is.
    pub(crate) snapshot: Option<snapshot::SnapshotWait>,
    /// Numbers the picture files, so two never share a name.
    pub(crate) snapshots: u32,
    /// How long one `run` call may take before it is cancelled and taken back; `--agent-timeout`.
    /// `None` is the default of [`DEFAULT_RUN_SECS`].
    pub(crate) run_timeout: Option<std::time::Duration>,
    /// Calls that have arrived and not yet run, with when they arrived; an attached agent's wait
    /// for the person to finish what they are doing.
    pub(crate) held: std::collections::VecDeque<(server::Job, std::time::Instant)>,
    /// A test's stand-in for a gesture of the person's, which a headless app cannot start.
    #[cfg(test)]
    pub(crate) pretend_busy: bool,
}

/// A `run` call's time allowed when `--agent-timeout` does not say.
pub(crate) const DEFAULT_RUN_SECS: u64 = 120;

pub(crate) struct Guard {
    pub(crate) allow_overwrite: bool,
    /// What the call's first `new` or `open` replaced, for a refusal to put back.
    pub(crate) replaced: Option<Replaced>,
}

/// What goes when a document is replaced, besides the document. A call that is taken back has the
/// document from its own copy, but the rest points into it: left behind, the active part names
/// nothing in the document put back, and the next read of the graph panics.
pub(crate) struct Replaced {
    file: Option<(std::path::PathBuf, fieldform_io::Archive)>,
    active_part: PartId,
    active_cage: CageId,
    selected_curve: Option<CurveId>,
    saved_revision: u64,
    history: CommandStack,
}

pub(crate) struct RunRequest {
    pub(crate) lines: String,
    pub(crate) atomic: bool,
    pub(crate) expected_revision: Option<u64>,
    pub(crate) allow_overwrite: bool,
    pub(crate) label: Option<String>,
    pub(crate) timeout: Option<std::time::Duration>,
}

impl Default for RunRequest {
    fn default() -> Self {
        Self {
            lines: String::new(),
            atomic: true,
            expected_revision: None,
            allow_overwrite: false,
            label: None,
            timeout: None,
        }
    }
}

pub(crate) struct LineResult {
    pub(crate) n: usize,
    pub(crate) line: String,
    pub(crate) outcome: LineOutcome,
}

pub(crate) struct RunResponse {
    pub(crate) ok: bool,
    pub(crate) kept: bool,
    pub(crate) revision_before: u64,
    pub(crate) revision_after: u64,
    pub(crate) lines: Vec<LineResult>,
    /// A failure of the call itself rather than of a line.
    pub(crate) error: Option<LineError>,
}

/// Verbs an agent may not run (`mcp-server-spec.md` §9.3): they end the process or open a
/// window of the application's own.
pub(crate) const REFUSED: &[(&str, &str)] = &[
    ("file.exit", "it would end the application the person has open"),
    ("help.about", "it opens a window"),
];

fn tool_error(code: &'static str, message: impl Into<String>) -> LineError {
    LineError { code, message: message.into(), verb: None, key: None, accepted: Vec::new(), did_you_mean: None }
}

fn refused_outcome(error: LineError) -> LineOutcome {
    LineOutcome {
        ok: false,
        said: None,
        warnings: vec![error.message.clone()],
        error: Some(error),
        blocks: Default::default(),
    }
}

impl App {
    /// Runs console lines as one transaction.
    pub(crate) fn agent_run(&mut self, req: &RunRequest) -> RunResponse {
        let revision_before = self.revision;
        let fail = |app: &App, error: LineError| RunResponse {
            ok: false,
            kept: false,
            revision_before,
            revision_after: app.revision,
            lines: Vec::new(),
            error: Some(error),
        };
        if let Some(wanted) = req.expected_revision
            && wanted != self.revision
        {
            let mut e = tool_error(
                "stale_revision",
                format!("the document is at revision {}, not {wanted}", self.revision),
            );
            e.key = Some(self.revision.to_string());
            return fail(self, e);
        }
        let lines: Vec<(usize, String)> = req
            .lines
            .lines()
            .enumerate()
            .filter_map(|(i, l)| {
                let l = l.trim();
                (!l.is_empty() && !l.starts_with('#')).then(|| (i + 1, l.to_string()))
            })
            .collect();
        if lines.len() > MAX_LINES {
            return fail(
                self,
                tool_error("bad_value", format!("{} lines is more than the {MAX_LINES} one call may carry", lines.len())),
            );
        }

        // A pending tool session is kept first, as a typed line keeps it.
        self.commit_session();
        let doc_before = self.doc.clone();
        let mark = self.stack.mark();
        let started = std::time::Instant::now();
        self.agent.cancel.store(false, std::sync::atomic::Ordering::SeqCst);
        self.agent.guard = Some(Guard { allow_overwrite: req.allow_overwrite, replaced: None });
        // **Script mode (P6):** a command that would ask a question is refused by name, and a
        // choice made here is not saved as the person's.
        let was_running = std::mem::replace(&mut self.script_running, true);

        let mut results: Vec<LineResult> = Vec::new();
        let mut call_error: Option<LineError> = None;
        let mut first_verb: Option<String> = None;
        for (n, line) in lines {
            if self.agent.cancel.load(std::sync::atomic::Ordering::SeqCst) {
                call_error = Some(tool_error("cancelled", format!("cancelled before line {n}")));
                break;
            }
            if req.timeout.is_some_and(|t| started.elapsed() > t) {
                call_error = Some(tool_error("timeout", format!("ran out of time before line {n}")));
                break;
            }
            first_verb.get_or_insert_with(|| line.split_whitespace().next().unwrap_or("").to_lowercase());
            self.gesture = None;
            let outcome = match self.refused_verb(&line) {
                Some(error) => refused_outcome(error),
                None => {
                    let mut outcome = self.run_line(&line);
                    self.drain_ownership_changes(false);
                    // A snapshot waits for frames this loop does not draw (`mcp-server-spec.md`
                    // §5.11 is a tool of its own); better refused than left pending.
                    if self.pending_snap.take().is_some() {
                        outcome = refused_outcome(tool_error("not_available", "`snap` needs frames; use the snapshot tool"));
                    }
                    outcome
                }
            };
            let ok = outcome.ok;
            results.push(LineResult { n, line, outcome });
            if !ok && req.atomic {
                break;
            }
        }
        self.script_running = was_running;
        let replaced = self.agent.guard.take().and_then(|guard| guard.replaced);

        let failed = results.iter().any(|r| !r.outcome.ok) || call_error.is_some();
        if failed && req.atomic {
            // Exact where undoing each entry in turn could only be as exact as each entry's undo.
            self.doc = doc_before;
            if let Some(replaced) = replaced {
                self.file = replaced.file;
                self.active_part = replaced.active_part;
                self.active_cage = replaced.active_cage;
                self.selected_curve = replaced.selected_curve;
                self.saved_revision = replaced.saved_revision;
                self.stack = replaced.history;
            }
            self.stack.cancel_to(mark);
            self.gesture = None;
            self.after_document_change();
            // The document is what it was, so its revision is too: a taken-back call leaves
            // nothing for a save to be dirty about.
            self.revision = revision_before;
            return RunResponse {
                ok: false,
                kept: false,
                revision_before,
                revision_after: self.revision,
                lines: results,
                error: call_error,
            };
        }
        let name = match (&req.label, &first_verb) {
            (Some(label), _) => format!("agent: {label}"),
            (None, Some(verb)) => format!("agent: {verb}\u{2026}"),
            (None, None) => "agent".to_string(),
        };
        // The history was cleared under the mark, so all of what is in it now is this call's.
        let since = if replaced.is_some() { fieldform_ops::StackMark::default() } else { mark };
        let stepped = self.stack.keep_since(since, name);
        self.gesture = None;
        self.agent.depth = stepped.then(|| self.stack.history().0.len());
        // A `new` or an `open` is kept with no step to show for it, since it clears the history.
        let kept = stepped || self.revision != revision_before;
        RunResponse { ok: !failed, kept, revision_before, revision_after: self.revision, lines: results, error: call_error }
    }

    /// Called by `new` and `open` as they replace the document. Sets aside, the first time in a
    /// call, what a refusal later in the call would have to put back. Taken rather than copied:
    /// the history cannot be cloned, and it was about to be cleared.
    pub(crate) fn note_document_replaced(&mut self) {
        let Some(guard) = &mut self.agent.guard else { return };
        if guard.replaced.is_some() {
            return;
        }
        guard.replaced = Some(Replaced {
            file: self.file.take(),
            active_part: self.active_part,
            active_cage: self.active_cage,
            selected_curve: self.selected_curve,
            saved_revision: self.saved_revision,
            history: std::mem::take(&mut self.stack),
        });
    }

    /// The refusal for a verb an agent may not run, if `line` begins with one.
    fn refused_verb(&self, line: &str) -> Option<LineError> {
        let word = line.split_whitespace().next()?.to_lowercase();
        let op = console_commands().into_iter().find(|(w, _)| *w == word).map(|(_, op)| op)?;
        let (_, why) = REFUSED.iter().find(|(o, _)| *o == op)?;
        let mut e = tool_error("agent_refused", format!("`{word}` is not available to an agent: {why}"));
        e.verb = Some(word);
        Some(e)
    }

    /// While an agent is driving, every file key goes through the guard (`paths.rs`). Returns the
    /// arguments with each path made absolute, or the refusal.
    pub(crate) fn agent_guard_paths(&self, op: &str, mut args: Args) -> Result<Args, LineError> {
        let Some(guard) = &self.agent.guard else { return Ok(args) };
        let writes = matches!(op, "file.save_as" | "view.snap");
        for key in paths::FILE_KEYS {
            let Some(given) = args.get(key).and_then(|v| v.as_str()).map(str::to_string) else { continue };
            match paths::resolve(&given, &self.agent.roots, writes, guard.allow_overwrite) {
                Ok(real) => args = args.with(key, Value::Str(real.to_string_lossy().into_owned())),
                Err(paths::PathRefusal::Outside) => {
                    let roots: Vec<String> = self.agent.roots.iter().map(|r| r.display().to_string()).collect();
                    let mut e = tool_error(
                        "path_outside_root",
                        format!("`{key}={given}` is outside the folders an agent may use: {}", roots.join(", ")),
                    );
                    e.key = Some((*key).to_string());
                    e.accepted = roots;
                    return Err(e);
                }
                Err(paths::PathRefusal::WouldOverwrite(path)) => {
                    let mut e = tool_error(
                        "would_overwrite",
                        format!("{} exists; pass allow_overwrite to replace it", path.display()),
                    );
                    e.key = Some((*key).to_string());
                    return Err(e);
                }
            }
        }
        Ok(args)
    }

    /// Steps the shared history, as Ctrl+Z / Ctrl+Y do. Takes back a person's edit only when
    /// `force` (T10). Returns the names stepped over and the new revision, or the refusal.
    pub(crate) fn agent_undo(&mut self, steps: usize, redo: bool, force: bool) -> Result<(Vec<String>, u64), LineError> {
        if !redo && !force {
            let (done, _) = self.stack.history();
            if self.agent.depth != Some(done.len()) {
                let top = done.last().cloned().unwrap_or_else(|| "(nothing)".into());
                let mut e = tool_error(
                    "not_agents_step",
                    format!("the newest step, `{top}`, was not made by an agent call; pass force to take it back"),
                );
                e.key = Some(top);
                return Err(e);
            }
        }
        let mut names = Vec::new();
        for _ in 0..steps.max(1) {
            self.material_gesture = None;
            self.outputs.gesture = None;
            let Some(name) = step_history(&mut self.stack, &mut self.doc, &mut self.gesture, !redo) else { break };
            self.after_document_change();
            names.push(name);
        }
        // Whatever the agent last kept is no longer on top.
        self.agent.depth = None;
        Ok((names, self.revision))
    }

    /// What this host is, for `session_info`.
    pub(crate) fn agent_session_info(&self) -> serde_json::Value {
        serde_json::json!({
            "host": self.agent_host.as_ref().map_or("launch", |h| h.mode.name()),
            "app_version": env!("CARGO_PKG_VERSION"),
            "document": {
                "dirty": self.is_dirty(),
                "revision": self.revision,
                "display_unit": "mm",
            },
            "roots": self.agent.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
            "capabilities": {
                // A person's viewport is not the agent's to aim (§5.11).
                "snapshot": self.agent_host.as_ref().is_none_or(|h| h.mode == server::HostMode::Launch),
                "import_step": true,
                "measure": true,
            },
            "busy": self.person_is_busy(),
            // The viewport, not the document: `sample` and `measure` already read the edit, and a
            // `snapshot` waits for this to clear. Never the same thing as `busy`, which is a person.
            "compiling": self.traced.is_compiling().is_some(),
        })
    }

    /// The document as data: parts, their blocks, parameters and curves.
    pub(crate) fn agent_document(&self) -> serde_json::Value {
        let drawn = self.drawn_bodies();
        let parts: Vec<serde_json::Value> = self
            .doc
            .parts
            .values()
            .map(|part| {
                let roots = part.graph.roots();
                let blocks: Vec<serde_json::Value> = part
                    .graph
                    .nodes()
                    .map(|(id, node)| {
                        let kind = serde_json::to_value(node)
                            .ok()
                            .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(str::to_string))
                            .unwrap_or_else(|| "unknown".into());
                        serde_json::json!({
                            "id": format!("n:{:016x}", id.raw()),
                            "name": self.doc.node_meta.get(&id).and_then(|m| m.name.clone()),
                            "kind": kind,
                            "body": roots.contains(&id),
                            "drawn": part.id == self.active_part && drawn.contains(&id),
                        })
                    })
                    .collect();
                serde_json::json!({
                    "id": part.id.to_string(),
                    "name": part.name,
                    "active": part.id == self.active_part,
                    "blocks": blocks,
                })
            })
            .collect();
        let parameters: Vec<serde_json::Value> = self
            .doc
            .parameters
            .values()
            .map(|p| {
                serde_json::json!({
                    "id": format!("p:{}", p.id),
                    "name": p.name,
                    "value": self.doc.parameter_value(p.id),
                    "source": format!("{:?}", p.value),
                })
            })
            .collect();
        let curves: Vec<serde_json::Value> =
            self.doc.curves.values().map(|c| serde_json::json!({ "id": format!("c:{}", c.id), "name": c.name })).collect();
        serde_json::json!({ "revision": self.revision, "parts": parts, "parameters": parameters, "curves": curves })
    }
}

#[cfg(test)]
mod tests;
