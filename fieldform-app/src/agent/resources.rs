// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The `fieldform://` resources (`mcp-server-spec.md` §5.12): read-only text a model can pull in.
//!
//! **Three are fixed at build time and one is the document.** The guide and the examples are files
//! in the source tree, so what a model reads is what the repository carries; the reference is the
//! command catalogue serialised the way `describe_command` answers it, so it cannot drift from the
//! verbs. Only `document` needs the UI thread, and the server fetches it as a `get_document` job.

use super::tools::info_json;
use rmcp::model::{Resource, ResourceTemplate};
use serde_json::json;

pub(crate) const GUIDE: &str = include_str!("guide.md");

/// The smoke scripts, as worked examples. Their `expect` lines say what each result measures.
pub(crate) const EXAMPLES: [(&str, &str, &str); 6] = [
    (
        "phase-a",
        "Every create verb reading its numbers, refusals by name, fillets, placements and a parameter-driven radius",
        include_str!("../../../benchmarks/smoke/phase-a.ffc"),
    ),
    (
        "curves",
        "Curves and what they measure",
        include_str!("../../../benchmarks/smoke/curves.ffc"),
    ),
    (
        "ownership",
        "Bodies, consumption and what a boolean takes",
        include_str!("../../../benchmarks/smoke/ownership.ffc"),
    ),
    (
        "gesture-undo",
        "One undo step per line, and what undo restores",
        include_str!("../../../benchmarks/smoke/gesture-undo.ffc"),
    ),
    (
        "helix-hole",
        "A thread and a hole, checked against closed forms",
        include_str!("../../../benchmarks/smoke/helix-hole.ffc"),
    ),
    (
        "owner-checks",
        "A run of checks written as expect lines",
        include_str!("../../../benchmarks/smoke/owner-checks.ffc"),
    ),
];

pub(crate) fn list() -> Vec<Resource> {
    let doc = |uri: &str, name: &str, description: &str, mime: &str| {
        Resource::new(uri, name).with_description(description).with_mime_type(mime)
    };
    let mut out = vec![
        doc(
            "fieldform://guide",
            "guide",
            "How to author in FieldForm: bodies, names, parameters, blends, what refuses and why",
            "text/markdown",
        ),
        doc(
            "fieldform://reference",
            "reference",
            "Every command with its keys, units, types and defaults, as JSON",
            "application/json",
        ),
        doc(
            "fieldform://document",
            "document",
            "The active part as console lines that rebuild it, with what could not be written",
            "text/plain",
        ),
    ];
    for (name, about, _) in EXAMPLES {
        out.push(doc(&format!("fieldform://examples/{name}"), name, about, "text/plain"));
    }
    out
}

pub(crate) fn templates() -> Vec<ResourceTemplate> {
    vec![
        ResourceTemplate::new("fieldform://examples/{name}", "example")
            .with_description("A worked console script; its expect lines say what the result measures")
            .with_mime_type("text/plain"),
    ]
}

/// The reference as JSON text: the whole catalogue, each verb as `describe_command` gives it.
pub(crate) fn reference() -> String {
    let commands: Vec<_> = crate::catalogue::catalogue().iter().map(info_json).collect();
    serde_json::to_string_pretty(&json!({ "commands": commands })).unwrap_or_else(|_| "{}".into())
}

/// What a fixed resource holds, or `None` for a URI that is not one (or is `document`).
pub(crate) fn fixed(uri: &str) -> Option<(&'static str, String)> {
    let rest = uri.strip_prefix("fieldform://")?;
    match rest {
        "guide" => Some(("text/markdown", GUIDE.to_string())),
        "reference" => Some(("application/json", reference())),
        _ => {
            let name = rest.strip_prefix("examples/")?;
            EXAMPLES.iter().find(|(n, _, _)| *n == name).map(|(_, _, text)| ("text/plain", text.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_resource_but_the_document_can_be_read_without_the_app() {
        for r in list() {
            if r.uri == "fieldform://document" {
                continue;
            }
            let (_, text) = fixed(&r.uri).unwrap_or_else(|| panic!("{} is listed and cannot be read", r.uri));
            assert!(!text.trim().is_empty(), "{} is empty", r.uri);
        }
    }

    #[test]
    fn the_reference_names_every_command_the_catalogue_does() {
        let reference: serde_json::Value = serde_json::from_str(&reference()).unwrap();
        let verbs: Vec<&str> =
            reference["commands"].as_array().unwrap().iter().filter_map(|c| c["verb"].as_str()).collect();
        for info in crate::catalogue::catalogue() {
            assert!(verbs.contains(&info.verb.as_str()), "{} is missing from the reference", info.verb);
        }
    }

    #[test]
    fn an_unknown_example_is_not_a_resource() {
        assert!(fixed("fieldform://examples/nonesuch").is_none());
        assert!(fixed("file:///etc/passwd").is_none());
    }
}
