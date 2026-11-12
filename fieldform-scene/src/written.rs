// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A field written as arithmetic: the Expression block's text and what it claims -- backlog
//! item 40, `expression-field-spec.md`.
//!
//! **The text is structure.** Its tree is what both compilers lower, so editing it recompiles
//! the shader; the parameters it names are slots, so dragging one does not. The tree holds
//! parameter *ids*, as a parameter's own expression does, and the text is re-rendered when one
//! is renamed (`Expression`'s module notes say why names would not do). A file writes the text
//! alone, and a reader parses it back under the document's names once those are known -- which
//! is after the parts are read, so a node read from a file carries its text unparsed until the
//! document resolves it ([`Written::resolve`]). A compile refuses an unresolved one by name
//! rather than drawing nothing in its place.

use crate::expression::{Expression, ParseError};
use crate::id::ParamId;
use serde::{Deserialize, Serialize};

/// What a written field says about its value (`expression-field-spec.md` §3.1).
///
/// `LowerBound` is the safe default: booleans are exact under it, tracing and pruning are safe
/// under it, and a blend that uses it announces itself. `Distance` is the opt-in that buys an
/// exact fillet radius, and the block's verify action is how it is checked.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WrittenClaim {
    Distance,
    LowerBound,
}

impl WrittenClaim {
    pub fn word(self) -> &'static str {
        match self {
            WrittenClaim::Distance => "distance",
            WrittenClaim::LowerBound => "bound",
        }
    }

    pub fn from_word(word: &str) -> Option<Self> {
        match word.trim().to_ascii_lowercase().as_str() {
            "distance" | "exact" => Some(WrittenClaim::Distance),
            "bound" | "lower_bound" | "lowerbound" | "lower" => Some(WrittenClaim::LowerBound),
            _ => None,
        }
    }
}

/// The text of a field expression and, once the document's names have been looked up, its tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Written {
    source: String,
    /// `None` for a text read from a file and not yet resolved against the document's names,
    /// and for one that did not parse the last time it was.
    parsed: Option<Expression>,
}

impl Written {
    /// A text as read from a file, not yet parsed: [`Self::resolve`] is what parses it.
    pub fn unparsed(text: impl Into<String>) -> Self {
        Self { source: text.into(), parsed: None }
    }

    /// Parses `text` in the field scope -- `x`, `y`, `z` the coordinates, every other name a
    /// parameter through `names` -- and keeps both.
    pub fn parse(text: &str, names: &dyn Fn(&str) -> Option<ParamId>) -> Result<Self, ParseError> {
        let parsed = Expression::parse_field(text, names)?;
        Ok(Self { source: parsed.source().to_string(), parsed: Some(parsed) })
    }

    /// Parses the text in place under `names`. On failure the tree is cleared and the error
    /// says where, so a text that stopped parsing -- a parameter it read was deleted -- is a
    /// block that says so rather than one drawing its last good shape.
    pub fn resolve(&mut self, names: &dyn Fn(&str) -> Option<ParamId>) -> Result<(), ParseError> {
        match Expression::parse_field(&self.source, names) {
            Ok(parsed) => {
                self.parsed = Some(parsed);
                Ok(())
            }
            Err(e) => {
                self.parsed = None;
                Err(e)
            }
        }
    }

    /// The text, as typed or as last re-rendered.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The tree, once resolved.
    pub fn expression(&self) -> Option<&Expression> {
        self.parsed.as_ref()
    }

    /// Every parameter the text reads, once resolved; none before.
    pub fn references(&self) -> Vec<ParamId> {
        self.parsed.as_ref().map(Expression::references).unwrap_or_default()
    }

    /// Re-renders the text from the tree with the names `names` gives -- what a rename does to
    /// every expression that read the renamed parameter. An unresolved text is left as it is.
    pub fn rerender(&mut self, names: &dyn Fn(ParamId) -> String) {
        if let Some(parsed) = &mut self.parsed {
            parsed.rerender(names);
            self.source = parsed.source().to_string();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_text_resolves_under_the_documents_names_and_says_where_it_cannot() {
        let radius = ParamId::new();
        let names = |name: &str| (name == "radius").then_some(radius);
        let mut w = Written::unparsed("sqrt(x*x + y*y + z*z) - radius");
        assert!(w.expression().is_none(), "a text read from a file is not yet a tree");
        w.resolve(&names).expect("resolves");
        assert_eq!(w.references(), vec![radius]);
        assert!(w.expression().unwrap().is_spatial());
        let mut gone = Written::unparsed("x - wall");
        let err = gone.resolve(&|_| None).expect_err("wall is nobody");
        assert_eq!(err.offset, 4);
        assert!(gone.expression().is_none(), "a text that stopped parsing holds no tree");
    }

    #[test]
    fn a_rename_re_renders_the_text_and_a_claim_reads_its_words() {
        let r = ParamId::new();
        let mut w = Written::parse("x - r", &|n| (n == "r").then_some(r)).unwrap();
        w.rerender(&|_| "radius".to_string());
        assert_eq!(w.source(), "x - radius");
        assert_eq!(WrittenClaim::from_word("exact"), Some(WrittenClaim::Distance));
        assert_eq!(WrittenClaim::from_word("bound"), Some(WrittenClaim::LowerBound));
        assert_eq!(WrittenClaim::from_word("maybe"), None);
    }
}
