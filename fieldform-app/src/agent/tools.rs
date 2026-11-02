// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The tools, as JSON in and JSON out (`mcp-server-spec.md` §5), with no protocol in sight.
//!
//! **Kept free of rmcp on purpose.** A tool is a function of the `App` and some arguments; the
//! transport only carries them. That makes every tool testable on a headless `App`, and means a
//! change of SDK touches `server.rs` and nothing here.

use super::*;
use crate::catalogue::{self, CommandInfo};
use crate::outcome::{BlockChanges, BlockRef};
use serde_json::{Map, Value as J, json};

/// What a tool call answers: the structured content, and whether it is a tool-level failure
/// (P4: a failure is a normal result the model can read, never a protocol error).
pub(crate) struct Reply {
    pub(crate) json: J,
    pub(crate) is_error: bool,
}

impl Reply {
    fn ok(json: J) -> Self {
        Self { json, is_error: false }
    }
    pub(crate) fn err(error: &LineError) -> Self {
        Self { json: json!({ "ok": false, "error": error_json(error) }), is_error: true }
    }
}

pub(crate) fn error_json(e: &LineError) -> J {
    let mut o = Map::new();
    o.insert("code".into(), json!(e.code));
    o.insert("message".into(), json!(e.message));
    if let Some(v) = &e.verb {
        o.insert("verb".into(), json!(v));
    }
    if let Some(k) = &e.key {
        o.insert("key".into(), json!(k));
    }
    if !e.accepted.is_empty() {
        o.insert("accepted".into(), json!(e.accepted));
    }
    if let Some(d) = &e.did_you_mean {
        o.insert("did_you_mean".into(), json!(d));
    }
    J::Object(o)
}

fn block_json(b: &BlockRef) -> J {
    json!({ "id": b.id, "name": b.name, "kind": b.kind, "body": b.body })
}

fn changes_json(c: &BlockChanges) -> (J, J, J) {
    (
        c.created.iter().map(block_json).collect(),
        c.changed.iter().map(block_json).collect(),
        c.removed.iter().map(block_json).collect(),
    )
}

fn line_json(r: &LineResult) -> J {
    let (created, changed, removed) = changes_json(&r.outcome.blocks);
    let mut o = json!({
        "n": r.n,
        "line": r.line,
        "ok": r.outcome.ok,
        "said": r.outcome.said,
        "created": created,
        "changed": changed,
        "removed": removed,
        "warnings": r.outcome.warnings,
    });
    if let Some(e) = &r.outcome.error {
        o["error"] = error_json(e);
    }
    o
}

pub(crate) fn info_json(i: &CommandInfo) -> J {
    json!({
        "verb": i.verb,
        "also": i.also,
        "category": i.category,
        "summary": i.summary,
        "declared": i.declared,
        "example": i.example,
        "keys": i.keys.iter().map(|k| json!({
            "key": k.key,
            "fields": k.fields,
            "unit": k.unit,
            "type": k.kind,
            "default": k.default.and_then(|d| serde_json::from_str::<J>(d).ok()),
        })).collect::<Vec<_>>(),
        "not_set_here": i.not_read.iter().map(|(f, why)| json!({ "field": f, "why": why })).collect::<Vec<_>>(),
    })
}

fn string_arg(args: &Map<String, J>, key: &str) -> Option<String> {
    args.get(key).and_then(J::as_str).map(str::to_string)
}

fn missing(key: &str) -> LineError {
    let mut e = tool_error("missing_key", format!("`{key}` is required"));
    e.key = Some(key.to_string());
    e
}

impl App {
    /// Runs one tool by name. Everything an MCP client can ask passes through here.
    pub(crate) fn agent_handle(&mut self, name: &str, args: &Map<String, J>) -> Reply {
        match name {
            "session_info" => Reply::ok(self.agent_session_info()),
            "get_document" => match args.get("format").and_then(J::as_str).unwrap_or("json") {
                "json" => Reply::ok(self.agent_document()),
                "text" => {
                    let script = self.agent_script();
                    Reply::ok(json!({ "revision": self.revision, "text": script.text, "omitted": script.omitted }))
                }
                other => {
                    let mut e = tool_error("bad_value", format!("`{other}` is not a format"));
                    e.key = Some("format".into());
                    e.accepted = vec!["json".into(), "text".into()];
                    Reply::err(&e)
                }
            },
            "list_commands" => {
                let wanted = string_arg(args, "category").map(|c| c.to_lowercase());
                let commands: Vec<J> = catalogue::catalogue()
                    .iter()
                    .filter(|i| wanted.as_ref().is_none_or(|w| i.category.to_lowercase().contains(w)))
                    .map(|i| {
                        let refused = REFUSED.iter().find(|(op, _)| *op == i.op);
                        let mut o = json!({ "verb": i.verb, "category": i.category, "summary": i.summary });
                        if let Some((_, why)) = refused {
                            o["agent"] = json!("refused");
                            o["reason"] = json!(why);
                        }
                        o
                    })
                    .collect();
                Reply::ok(json!({ "commands": commands }))
            }
            "describe_command" => {
                let Some(verb) = string_arg(args, "verb") else { return Reply::err(&missing("verb")) };
                match catalogue::describe(&verb) {
                    Some(info) => Reply::ok(info_json(&info)),
                    None => {
                        let words: Vec<String> = crate::commands::console_words().into_iter().map(|(w, _)| w).collect();
                        let mut e = tool_error("unknown_verb", format!("`{verb}` is not a command"));
                        e.verb = Some(verb.clone());
                        e.did_you_mean = crate::outcome::nearest(&verb.to_lowercase(), words.iter().map(String::as_str));
                        Reply::err(&e)
                    }
                }
            }
            "run" => {
                let Some(lines) = string_arg(args, "lines") else { return Reply::err(&missing("lines")) };
                let req = RunRequest {
                    lines,
                    atomic: args.get("mode").and_then(J::as_str) != Some("keep_going"),
                    expected_revision: args.get("expected_revision").and_then(J::as_u64),
                    allow_overwrite: args.get("allow_overwrite").and_then(J::as_bool).unwrap_or(false),
                    label: string_arg(args, "label"),
                    timeout: Some(self.agent.run_timeout.unwrap_or(std::time::Duration::from_secs(super::DEFAULT_RUN_SECS))),
                };
                let out = self.agent_run(&req);
                let mut o = json!({
                    "ok": out.ok,
                    "kept": out.kept,
                    "revision_before": out.revision_before,
                    "revision_after": out.revision_after,
                    "lines": out.lines.iter().map(line_json).collect::<Vec<_>>(),
                });
                if let Some(e) = &out.error {
                    o["error"] = error_json(e);
                }
                Reply { json: o, is_error: !out.ok }
            }
            "inspect" => self.agent_inspect(args),
            "measure" => self.agent_measure(args),
            "check" => self.agent_check(),
            "sample" => self.agent_sample(args),
            "undo" => {
                let steps = args.get("steps").and_then(J::as_u64).unwrap_or(1) as usize;
                let redo = args.get("redo").and_then(J::as_bool).unwrap_or(false);
                let force = args.get("force").and_then(J::as_bool).unwrap_or(false);
                match self.agent_undo(steps, redo, force) {
                    Ok((names, revision)) => Reply::ok(json!({ "ok": true, "stepped": names, "revision": revision })),
                    Err(e) => Reply::err(&e),
                }
            }
            other => {
                let mut e = tool_error("not_available", format!("`{other}` is not a tool of this host"));
                e.key = Some(other.to_string());
                Reply::err(&e)
            }
        }
    }
}

/// The tools this host offers, as `(name, description, schema, read_only)`.
pub(crate) fn definitions() -> Vec<(&'static str, &'static str, J, bool)> {
    let none = json!({ "type": "object", "properties": {}, "additionalProperties": false });
    vec![
        (
            "session_info",
            "What host this is, the document's revision and path state, and the folders files may be read and written in.",
            none.clone(),
            true,
        ),
        (
            "list_commands",
            "Every console verb, one line each. Call before guessing a verb.",
            json!({ "type": "object", "properties": { "category": { "type": "string", "description": "Only verbs whose menu path contains this text." } } }),
            true,
        ),
        (
            "describe_command",
            "One verb's keys with type, unit and default, and a worked example. Call before using a verb you have not used.",
            json!({ "type": "object", "properties": { "verb": { "type": "string" } }, "required": ["verb"] }),
            true,
        ),
        (
            "run",
            "Run console lines as one undo step. All lines succeed or the whole call is taken back (mode=atomic, the default). Lengths are millimetres. Returns, per line, what it said and the blocks it created, changed or removed, by id and name.",
            json!({
                "type": "object",
                "properties": {
                    "lines": { "type": "string", "description": "Console lines, newline-separated. # comments and blank lines are skipped." },
                    "mode": { "type": "string", "enum": ["atomic", "keep_going"] },
                    "expected_revision": { "type": "integer", "description": "Refuse if the document has changed since you last read it." },
                    "allow_overwrite": { "type": "boolean" },
                    "label": { "type": "string", "description": "Names the undo step." }
                },
                "required": ["lines"]
            }),
            false,
        ),
        (
            "undo",
            "Undo or redo whole steps. Takes back only steps an agent call made unless force is true.",
            json!({
                "type": "object",
                "properties": {
                    "steps": { "type": "integer", "minimum": 1 },
                    "redo": { "type": "boolean" },
                    "force": { "type": "boolean" }
                }
            }),
            false,
        ),
        (
            "inspect",
            "One block in full: every key with its value, the parameters it reads, what it reads and what reads it, whether it is a body, and how exact its field is.",
            json!({ "type": "object", "properties": { "block": { "type": "string", "description": "A name or an id (n:…)." } }, "required": ["block"] }),
            true,
        ),
        (
            "measure",
            "Volume, surface area and centroid of a body, or of the whole part, by meshing the field. States its resolution and cell size. resolution=fine doubles the cells and is slow.",
            json!({
                "type": "object",
                "properties": {
                    "target": { "type": "string", "description": "A body's name or id; the whole part if omitted." },
                    "quantities": { "type": "array", "items": { "type": "string", "enum": ["volume", "area", "centroid", "bbox"] } },
                    "resolution": { "type": "string", "enum": ["standard", "fine"] }
                }
            }),
            true,
        ),
        (
            "check",
            "Problems with the document: empty slots, parameters that do not resolve, approximate blends, hidden bodies nothing draws, a part that does not compile. An empty list means nothing known is wrong. Call before measure and before saving.",
            none.clone(),
            true,
        ),
        (
            "sample",
            "The field's value at points (mm, up to 10000): negative inside. Also says whether the magnitude is a distance or only a lower bound. Answers is-this-point-inside and how-thick-here.",
            json!({
                "type": "object",
                "properties": {
                    "points": { "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 } },
                    "target": { "type": "string" },
                    "gradient": { "type": "boolean" }
                },
                "required": ["points"]
            }),
            true,
        ),
        (
            "snapshot",
            "A picture of the viewport as a PNG, from a named view (iso, front, back, left, right, top, bottom) or the current one. The picture is the viewport's own size, and taking it moves this host's camera.",
            json!({ "type": "object", "properties": { "view": { "type": "string", "enum": ["iso", "front", "back", "left", "right", "top", "bottom", "current"] } } }),
            true,
        ),
        (
            "get_document",
            "The document: parts, their blocks (id, name, kind, whether a body), parameters with their values, and curves. With format=text, console lines that rebuild the active part when run on an empty document; blocks it cannot write are listed in `omitted`.",
            json!({ "type": "object", "properties": { "format": { "type": "string", "enum": ["json", "text"] } } }),
            true,
        ),
    ]
}

/// What the server tells a client when it connects (`mcp-server-spec.md` §8).
pub(crate) const INSTRUCTIONS: &str = "FieldForm is a field-based CAD kernel. All lengths are millimetres. \
A part's shape is the union of its bodies; a body is a block nothing else consumes; a boolean takes a list \
and consumes what it is given. Names are labels and ids are stable: refer to blocks by name in lines, by id \
when reading results. Call list_commands and describe_command before guessing keys. Prefer one run call per \
meaningful step, so each is one undo. Pass expected_revision when the document may have changed under you.";
