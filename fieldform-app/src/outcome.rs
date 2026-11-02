// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What one console line did, as data (`mcp-server-spec.md` §6.1).
//!
//! **Built from the words the console reported, not beside them.** `App::report` hands each
//! line it prints to a capture while a line runs, so `said` is the console's own sentence and an
//! agent cannot be told something the console did not say. A refusal that has a structure
//! (an unknown key, an unknown verb) is recorded where it is refused; any other warning is a
//! `refused` with its text, which is honest about not knowing more.

use crate::console_keys::{self, Keys};
use fieldform_ops::Args;

/// A refusal, in the vocabulary of `mcp-server-spec.md` §7.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineError {
    pub code: &'static str,
    pub message: String,
    pub verb: Option<String>,
    pub key: Option<String>,
    /// What the verb does read, when the refusal is about a key.
    pub accepted: Vec<String>,
    pub did_you_mean: Option<String>,
}

/// A block, named the way a result names one: by id, which is stable, and by name, which is a
/// label and can change under the reader (`mcp-server-spec.md` P5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRef {
    pub id: String,
    pub name: Option<String>,
    pub kind: String,
    /// Whether nothing consumes it, so it is a root of its part. Always false for a removed block.
    pub body: bool,
}

/// The blocks of a document as a line found them, to be set against what it left.
///
/// **By diff, not by parsing the console's words.** "Added a sphere" is prose and changes; the
/// graph before and after is the fact. A block is changed when its serialized form or its name
/// differs, which is every edit the file would record.
pub(crate) struct BlockSnapshot {
    /// Keyed `n:<id>`, `p:<id>` (a parameter) or `c:<id>` (a curve), so the three kinds share one diff.
    blocks: std::collections::BTreeMap<String, (serde_json::Value, Option<String>)>,
}

/// What a line made, altered and took away.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockChanges {
    pub created: Vec<BlockRef>,
    pub changed: Vec<BlockRef>,
    pub removed: Vec<BlockRef>,
}

impl BlockSnapshot {
    pub(crate) fn of(doc: &fieldform_ops::Document) -> Self {
        let mut blocks = std::collections::BTreeMap::new();
        for part in doc.parts.values() {
            for (id, node) in part.graph.nodes() {
                let name = doc.node_meta.get(&id).and_then(|m| m.name.clone());
                let value = serde_json::to_value(node).unwrap_or(serde_json::Value::Null);
                blocks.insert(format!("n:{:016x}", id.raw()), (value, name));
            }
        }
        for (id, p) in &doc.parameters {
            let value = serde_json::Value::String(format!("{:?}", p.value));
            blocks.insert(format!("p:{id}"), (value, Some(p.name.clone())));
        }
        for (id, c) in &doc.curves {
            let value = serde_json::to_value(c).unwrap_or(serde_json::Value::Null);
            blocks.insert(format!("c:{id}"), (value, Some(c.name.clone())));
        }
        Self { blocks }
    }

    pub(crate) fn changes_to(&self, after: &Self, doc: &fieldform_ops::Document) -> BlockChanges {
        let bodies: std::collections::BTreeSet<String> = doc
            .parts
            .values()
            .flat_map(|part| {
                part.graph.nodes().filter(|(_, n)| n.is_body()).map(|(id, _)| format!("n:{:016x}", id.raw()))
            })
            .collect();
        let describe = |key: &str, held: &(serde_json::Value, Option<String>), body: bool| BlockRef {
            id: key.to_string(),
            name: held.1.clone(),
            kind: match key.as_bytes()[0] {
                b'p' => "parameter".to_string(),
                b'c' => "curve".to_string(),
                _ => held.0.get("kind").and_then(|k| k.as_str()).unwrap_or("unknown").to_string(),
            },
            body,
        };
        let mut out = BlockChanges::default();
        for (raw, now) in &after.blocks {
            let body = bodies.contains(raw);
            match self.blocks.get(raw) {
                None => out.created.push(describe(raw, now, body)),
                Some(was) if was != now => out.changed.push(describe(raw, now, body)),
                Some(_) => {}
            }
        }
        for (raw, was) in &self.blocks {
            if !after.blocks.contains_key(raw) {
                out.removed.push(describe(raw, was, false));
            }
        }
        out
    }
}

/// What a line did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineOutcome {
    pub ok: bool,
    /// The success lines, as `say` printed them, joined.
    pub said: Option<String>,
    /// What `warn` marked `!!`. A warning does not by itself make the line fail to run, but the
    /// console treats any as a refusal, so `ok` is false whenever there is one.
    pub warnings: Vec<String>,
    pub error: Option<LineError>,
    pub blocks: BlockChanges,
}

impl LineOutcome {
    /// `caught` is everything reported since the line began, starting with the line's own echo
    /// (`> line`); what came before the echo belongs to the GUI, not to this line.
    pub(crate) fn from_reports(line: &str, caught: Vec<(bool, String)>, error: Option<LineError>) -> Self {
        let echo = format!("> {}", line.trim());
        let after = caught.iter().position(|(_, t)| *t == echo).map_or(0, |i| i + 1);
        let mut said = Vec::new();
        let mut warnings = Vec::new();
        for (ok, text) in caught.into_iter().skip(after) {
            if ok {
                said.push(text);
            } else {
                warnings.push(text);
            }
        }
        let error = error.or_else(|| {
            warnings.first().map(|message| LineError {
                code: "refused",
                message: message.clone(),
                verb: None,
                key: None,
                accepted: Vec::new(),
                did_you_mean: None,
            })
        });
        Self {
            ok: warnings.is_empty(),
            said: (!said.is_empty()).then(|| said.join(" ")),
            warnings,
            error,
            blocks: BlockChanges::default(),
        }
    }
}

/// The closest of `candidates` to `word`, only when it is within two edits: a nearer guess than
/// that is a different word, and offering it would teach the wrong key.
pub fn nearest<'a>(word: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<String> {
    candidates
        .into_iter()
        .map(|c| (edit_distance(word, c), c))
        .filter(|(d, _)| *d <= 2)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c.to_string())
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (row[j + 1] + 1).min(row[j] + 1).min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

/// A key the verb does not read, with what it does.
pub(crate) fn unknown_key_error(word: &str, op: &str, args: &Args, message: &str) -> LineError {
    let accepted: Vec<String> = match console_keys::verb(op).map(|v| &v.keys) {
        Some(Keys::Declared { reads, .. }) => reads.iter().map(|k| k.word.to_string()).collect(),
        _ => Vec::new(),
    };
    let mut unknown: Vec<&str> = args
        .iter()
        .map(|(k, _)| k)
        .filter(|k| !accepted.iter().any(|a| a == k))
        .collect();
    unknown.sort_unstable();
    let key = unknown.first().map(|k| k.to_string());
    let did_you_mean = key
        .as_deref()
        .and_then(|k| nearest(k, accepted.iter().map(String::as_str)));
    LineError {
        code: "unknown_key",
        message: message.to_string(),
        verb: Some(word.to_string()),
        key,
        accepted,
        did_you_mean,
    }
}

/// The registry's refusal, in the spec's vocabulary (§7): `UnknownOp` is `unknown_verb`,
/// `MissingArg` is `missing_key`, `BadArg` is `bad_value`, `Failed` is `refused`.
pub(crate) fn from_op_error(e: &fieldform_ops::OpError) -> LineError {
    use fieldform_ops::OpError::*;
    let (code, verb, key) = match e {
        UnknownOp(op) => ("unknown_verb", Some(op.clone()), None),
        MissingArg { op, arg } => ("missing_key", Some(op.clone()), Some(arg.clone())),
        BadArg { op, arg, .. } => ("bad_value", Some(op.clone()), Some(arg.clone())),
        Failed { op, .. } => ("refused", Some(op.clone()), None),
    };
    let did_you_mean = match (code, &verb) {
        ("unknown_verb", Some(v)) => nearest(v, crate::commands::console_words().iter().map(|(w, _)| w.as_str())),
        _ => None,
    };
    LineError { code, message: e.to_string(), verb, key, accepted: Vec::new(), did_you_mean }
}

/// A word that is no verb.
pub(crate) fn unknown_verb_error(word: &str) -> LineError {
    let words = crate::commands::console_words();
    LineError {
        code: "unknown_verb",
        message: format!("{word:?} is not a command — type `help` for the list"),
        verb: Some(word.to_string()),
        key: None,
        accepted: Vec::new(),
        did_you_mean: nearest(word, words.iter().map(|(w, _)| w.as_str())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_near_miss_is_offered_and_a_different_word_is_not() {
        assert_eq!(nearest("sise", ["size", "centre"]), Some("size".into()));
        assert_eq!(nearest("side", ["size", "centre"]), Some("size".into()));
        assert_eq!(nearest("radius", ["size", "centre"]), None);
    }

    #[test]
    fn what_came_before_the_echo_is_not_this_lines() {
        let caught = vec![
            (true, "GUI said this".to_string()),
            (true, "> sphere".to_string()),
            (true, "Added a sphere".to_string()),
        ];
        let out = LineOutcome::from_reports("sphere", caught, None);
        assert!(out.ok);
        assert_eq!(out.said.as_deref(), Some("Added a sphere"));
    }
}
