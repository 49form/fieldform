// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Every console verb as data: what it is, where it lives, what it reads (`mcp-server-spec.md`
//! §6.3).
//!
//! **A reading of two tables, never a third.** The registry (`commands.rs`) says what a verb is
//! and where the menu puts it; `console_keys::VERBS` says what it reads. This joins them, so
//! `list_commands`, `describe_command` and a future `reference.json` answer from the same place
//! `help` does. Types, defaults, ranges and examples are not in either table yet, so they are
//! absent here rather than invented (spec P7); a key's unit is given only where a session
//! command's panel already names one.

use crate::commands::{self, Command, Unit};
use crate::console_keys::{self, Keys};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyInfo {
    pub key: &'static str,
    /// The block's serialized fields this key sets.
    pub fields: Vec<&'static str>,
    /// `"mm"`, `"deg"`, … where the registry knows it.
    pub unit: Option<&'static str>,
    /// `number`, `flag`, `word`, `vector3`, … where one field stands behind the key
    /// (`key_types.rs`); `None` is "not known", never a default.
    pub kind: Option<&'static str>,
    /// What the field holds on the block the verb makes bare, as JSON (`key_defaults.rs`);
    /// `None` where the verb cannot be run without an input.
    pub default: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandInfo {
    pub verb: String,
    pub also: Vec<String>,
    pub op: &'static str,
    pub category: String,
    pub summary: String,
    /// Whether the verb has declared its keys; one that has not passes unknown keys unchecked.
    pub declared: bool,
    /// A line that works, from `examples.rs`.
    pub example: Option<&'static str>,
    pub keys: Vec<KeyInfo>,
    /// `(field, why)` the console does not set.
    pub not_read: Vec<(&'static str, &'static str)>,
}

fn unit_word(unit: Unit) -> Option<&'static str> {
    match unit {
        Unit::Mm => Some("mm"),
        Unit::Degrees => Some("deg"),
        Unit::Count | Unit::Plain | Unit::Word => None,
    }
}

fn info(c: &Command) -> CommandInfo {
    let (declared, keys, not_read) = match console_keys::verb(c.op).map(|v| &v.keys) {
        Some(Keys::Declared { reads, not_read }) => (
            true,
            reads
                .iter()
                .map(|k| KeyInfo {
                    key: k.word,
                    fields: k.covers.to_vec(),
                    kind: crate::key_types::type_of(c.op, k.word),
                    default: crate::key_defaults::default_of(c.op, k.word),
                    unit: c.params.iter().find(|p| p.key == k.word).and_then(|p| unit_word(p.unit)),
                })
                .collect(),
            not_read.to_vec(),
        ),
        _ => (false, Vec::new(), Vec::new()),
    };
    let summary = console_keys::about(c.op)
        .map(str::to_string)
        .or_else(|| (!c.hint.is_empty()).then(|| c.hint.to_string()))
        .unwrap_or_else(|| c.display_name.to_string());
    CommandInfo {
        verb: c.console_name.to_string(),
        also: c.console_also.iter().map(|w| w.to_string()).collect(),
        op: c.op,
        category: c.category.join(" \u{203a} "),
        summary,
        declared,
        example: crate::examples::example_of(c.console_name),
        keys,
        not_read,
    }
}

/// Every verb this build has, once each, in registry order. Entries that share a verb (a menu
/// item that runs it with fixed arguments) are one verb.
pub fn catalogue() -> Vec<CommandInfo> {
    let mut out: Vec<CommandInfo> = Vec::new();
    for c in commands::COMMANDS.iter().filter(|c| c.built()) {
        if !out.iter().any(|i| i.verb == c.console_name) {
            out.push(info(c));
        }
    }
    out
}

/// One verb, by its word or any other spelling the console accepts.
pub fn describe(word: &str) -> Option<CommandInfo> {
    let word = word.to_lowercase();
    catalogue().into_iter().find(|i| i.verb == word || i.also.iter().any(|a| *a == word))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T4: nothing the console accepts is missing from the catalogue, and nothing the key table
    /// says a verb reads is missing from its entry.
    #[test]
    fn every_console_verb_is_listed_and_described() {
        let listed = catalogue();
        for (word, op) in commands::console_words() {
            let info = describe(&word).unwrap_or_else(|| panic!("`{word}` is typed but not listed"));
            assert!(!info.summary.is_empty(), "`{word}` has no summary");
            if let Some(Keys::Declared { reads, .. }) = console_keys::verb(op).map(|v| &v.keys) {
                // Several verbs may share an op with different fixed arguments; the first
                // entry for the verb carries the keys.
                let _ = &listed;
                for k in *reads {
                    assert!(
                        info.keys.iter().any(|i| i.key == k.word),
                        "`{word}` reads `{}` but the catalogue does not say so",
                        k.word
                    );
                }
            }
        }
    }

    #[test]
    fn a_sphere_radius_is_a_number_and_a_cube_size_a_vector() {
        let key = |verb: &str, key: &str| {
            describe(verb).unwrap().keys.into_iter().find(|k| k.key == key).unwrap().kind
        };
        assert_eq!(key("sphere", "radius"), Some("number"));
        assert_eq!(key("cube", "size"), Some("vector3"));
    }

    #[test]
    fn a_default_is_reported_where_the_bare_verb_makes_one() {
        let sphere = describe("sphere").unwrap();
        let radius = sphere.keys.iter().find(|k| k.key == "radius").unwrap();
        assert_eq!(radius.default, Some("50.0"));
    }

    #[test]
    fn a_verb_is_found_by_any_spelling() {
        for info in catalogue() {
            for also in &info.also {
                assert_eq!(describe(also).map(|i| i.verb), Some(info.verb.clone()));
            }
        }
    }
}
