// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The command palette's matching and ranking (`interaction-spec.md` §5).
//!
//! **Every command in a few keystrokes.** The palette searches the registry -- display name,
//! console verb and the aliases a command answers to -- and orders what it finds by how surely it
//! is what was meant: a console verb the query begins, then a name or alias it begins, then what
//! was run lately, then how well the letters fall in order. A command not on offer now is still
//! listed, after every one that is, dimmed and saying why, so the palette answers "where is it"
//! even when the answer is "not here yet".
//!
//! **And the console is one keystroke away from it.** A query that reads as a whole console line
//! with arguments runs as typed (§5.5), so the palette and the console are one input path.
//!
//! Free of the `App`, like the registry it reads.

use crate::commands::{Command, Ctx, COMMANDS};

/// How well `query` falls in order through `text`, ignoring case, or `None` where its letters do
/// not all appear in order. Consecutive letters and letters starting a word score higher, and a
/// match that starts early beats one that starts late.
pub fn fuzzy(query: &str, text: &str) -> Option<i32> {
    let query: Vec<char> = query.to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    if query.is_empty() {
        return Some(0);
    }
    let text: Vec<char> = text.to_lowercase().chars().collect();
    let mut score = 0;
    let mut at = 0;
    let mut previous: Option<usize> = None;
    for q in query {
        let found = (at..text.len()).find(|&i| text[i] == q)?;
        score += 1;
        if previous.is_some_and(|p| p + 1 == found) {
            score += 5;
        }
        if found == 0 || !text[found - 1].is_alphanumeric() {
            score += 3;
        }
        if previous.is_none() {
            score -= found.min(10) as i32;
        }
        previous = Some(found);
        at = found + 1;
    }
    Some(score)
}

/// One row of the palette: a command, and why it is not on offer if it is not.
pub struct Hit {
    pub command: &'static Command,
    pub available: Result<(), &'static str>,
}

/// The palette's list for `query`, best first (§5.2).
///
/// `recent` is the commands run lately, most recent first; the palette keeps the last twenty.
pub fn ranked(query: &str, recent: &[&str], ctx: &Ctx) -> Vec<Hit> {
    let q = query.trim().to_lowercase();
    let mut scored: Vec<((bool, bool, bool, usize, i32, usize), Hit)> = COMMANDS
        .iter()
        .enumerate()
        .filter(|(_, c)| c.built())
        .filter_map(|(index, c)| {
            let names = std::iter::once(c.display_name).chain(std::iter::once(c.console_name)).chain(c.aliases.iter().copied());
            let fit = names.filter_map(|n| fuzzy(&q, n)).max()?;
            let starts = |text: &str| !q.is_empty() && text.to_lowercase().starts_with(&q);
            let console_prefix = starts(c.console_name) || c.console_also.iter().any(|w| starts(w));
            let other_prefix = starts(c.display_name) || c.aliases.iter().any(|a| starts(a));
            let recency = recent.iter().position(|id| *id == c.id).unwrap_or(usize::MAX);
            let available = (c.context)(ctx);
            let key = (available.is_err(), !console_prefix, !other_prefix, recency, -fit, index);
            Some((key, Hit { command: c, available }))
        })
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0));
    scored.into_iter().map(|(_, hit)| hit).collect()
}

/// Whether `query` is a whole console line with arguments, which Enter runs as typed (§5.5): a
/// verb the console knows, something after it, and an argument grammar the console reads.
pub fn is_console_line(query: &str) -> bool {
    let words = crate::console::console_words(query.trim());
    let Some(((verb, _), rest)) = words.split_first() else { return false };
    if rest.is_empty() {
        return false;
    }
    let verb = crate::console::split_degree_suffix(&verb.to_lowercase()).0;
    crate::console_commands().iter().any(|(w, _)| *w == verb) && crate::console::console_arguments(rest.to_vec()).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(query: &str, recent: &[&str], ctx: &Ctx) -> &'static str {
        ranked(query, recent, ctx).first().map(|h| h.command.id).unwrap_or("nothing")
    }

    /// **A console verb the query begins wins, then a name or an alias, then what was run lately,
    /// then the letters in order** (§5.2).
    #[test]
    fn the_palette_ranks_by_verb_then_name_then_recency_then_fit() {
        let ctx = Ctx { bodies: 2, pair_candidates: 2, selected: true, ..Default::default() };
        assert_eq!(first("ext", &[], &ctx), "create.extrude", "a verb it begins");
        assert_eq!(first("pull", &[], &ctx), "create.extrude", "an alias it begins");
        assert_eq!(first("sph", &[], &ctx), "create.sphere");
        assert_eq!(first("box", &[], &ctx), "create.cube", "an alias");
        // Many verbs begin with c; the one run lately comes first among them.
        assert_eq!(first("c", &["create.cylinder"], &ctx), "create.cylinder");
        assert_eq!(first("c", &["create.cone", "create.cylinder"], &ctx), "create.cone", "most recent first");
        // Letters in order, when nothing begins with them.
        assert_eq!(first("frsel", &[], &ctx), "view.frame_selection");
        assert!(ranked("zzzq", &[], &ctx).is_empty(), "nothing matches nonsense");
    }

    /// **A command not on offer is still found, after every one that is, saying why** (§5.2).
    #[test]
    fn a_command_not_on_offer_is_listed_after_and_says_why() {
        let none = Ctx::default();
        let hits = ranked("o", &[], &none);
        let offset = hits.iter().position(|h| h.command.id == "modify.offset").expect("offset is listed");
        assert_eq!(hits[offset].available, Err("Needs a body"));
        assert!(hits[..offset].iter().all(|h| h.available.is_ok() || h.command.id == "modify.offset" || h.available.is_err()));
        let last_available = hits.iter().rposition(|h| h.available.is_ok()).expect("something is on offer");
        let first_unavailable = hits.iter().position(|h| h.available.is_err()).expect("something is not");
        assert!(last_available < first_unavailable, "every available command comes first");
    }

    /// **A whole console line runs as typed; a bare verb or a search does not** (§5.5).
    #[test]
    fn a_whole_console_line_is_told_from_a_search() {
        assert!(is_console_line("sphere radius=10 'ball'"));
        assert!(is_console_line("boolean -u in=a,b"));
        assert!(is_console_line("curve3 p=0,0,0;1,0,0;2,1,0;3,0,0"));
        assert!(!is_console_line("sphere"), "a verb alone is a search, and runs the highlighted command");
        assert!(!is_console_line("frame sel"), "a stray word is not an argument");
        assert!(!is_console_line("nonsense radius=1"), "not a verb");
    }

    #[test]
    fn letters_in_order_score_better_together_and_at_word_starts() {
        assert!(fuzzy("fs", "Frame selection") > fuzzy("fs", "offset"));
        assert!(fuzzy("ext", "extrude") > fuzzy("ext", "the extra"));
        assert_eq!(fuzzy("xyz", "extrude"), None);
    }
}
