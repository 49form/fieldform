// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Key bindings: the registry's defaults, a person's own file over them, and what collides
//! (`interaction-spec.md` §9).
//!
//! **A plain-text file a person can read, edit and keep in version control** --
//! `%APPDATA%\fieldform\bindings.txt` -- one binding a line, in the notation the menus print:
//!
//! ```text
//! # chord = command
//! Ctrl+D = edit.duplicate
//! Alt+P = palette.open
//! Slash = none          # takes a default off
//! ```
//!
//! A line binds a chord to a registry id, or to `none` to take a default off; the file's lines win
//! over the registry's defaults. **Two commands on one chord is a conflict, and is listed rather
//! than settled quietly**: in the file, the later line wins and the panel says so; from the panel,
//! a chord already in use is refused by name. And some chords belong to something else first: the
//! navigation owns its own (§9.2), and while a tool is pending its panel owns Enter, Esc, Tab and
//! what starts typing (§4.3), so a binding on one of those is listed as one that cannot always fire.
//!
//! Free of the `App`: the file is text in and text out.

use crate::commands::{by_id, Chord, Command, COMMANDS};

/// One line of a bindings file: a chord, and the command it runs or `None` for "no command".
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub chord: Chord,
    pub command: Option<&'static str>,
}

/// Reads a bindings file. Each line it cannot read is named with its number and why, and the rest
/// are read anyway: one mistyped line is no reason to lose every other binding.
pub fn parse(text: &str) -> (Vec<Line>, Vec<String>) {
    let mut lines = Vec::new();
    let mut errors = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((chord, command)) = line.split_once('=') else {
            errors.push(format!("line {}: {line:?} is not `chord = command`", n + 1));
            continue;
        };
        let chord = match Chord::parse(chord.trim()) {
            Ok(chord) => chord,
            Err(why) => {
                errors.push(format!("line {}: {why}", n + 1));
                continue;
            }
        };
        let command = command.trim();
        if command.eq_ignore_ascii_case("none") {
            lines.push(Line { chord, command: None });
            continue;
        }
        match by_id(command) {
            Some(c) => lines.push(Line { chord, command: Some(c.id) }),
            None => errors.push(format!("line {}: no command is called {command:?}", n + 1)),
        }
    }
    (lines, errors)
}

/// A bindings file holding `lines`, with the comment that says how to write one.
pub fn write(lines: &[Line]) -> String {
    let mut out = String::from(
        "# fieldform key bindings: one `chord = command` a line, over the defaults.\n\
         # `chord = none` takes a default off. Commands are the ids `help` and the palette show.\n",
    );
    for line in lines {
        out.push_str(&format!("{} = {}\n", line.chord.label(), line.command.unwrap_or("none")));
    }
    out
}

/// Why a chord cannot always fire, or fires two things.
#[derive(Clone, Debug, PartialEq)]
pub struct Conflict {
    pub chord: Chord,
    /// The commands bound to it.
    pub commands: Vec<&'static str>,
    pub why: String,
}

/// The chords the panel of a pending tool reads first (§4.3): Enter keeps, Esc takes back, Tab
/// moves between rows, and a digit, `-`, `.` or `(` starts typing.
pub fn reserved_while_pending(chord: &Chord) -> bool {
    let plain = !chord.ctrl && !chord.alt;
    plain
        && (matches!(chord.key.as_str(), "Enter" | "Escape" | "Tab" | "Minus" | "Period" | "OpenBracket")
            || (chord.key.len() == 1 && chord.key.chars().all(|c| c.is_ascii_digit()))
            || chord.key.starts_with("Num") && chord.key[3..].chars().all(|c| c.is_ascii_digit()))
}

/// What the navigation preset owns, which no binding may take (§9.2). The current preset drives
/// the camera with the mouse alone -- left, middle and right drags, the wheel, and Alt+Shift with
/// any of them -- so it owns no key; it is a function so a preset that does has one place to say so.
pub fn reserved_by_navigation(_chord: &Chord) -> bool {
    false
}

/// Every chord in use and the command it runs, defaults overridden by `user`, with every conflict.
pub fn effective(user: &[Line]) -> (Vec<(Chord, &'static Command)>, Vec<Conflict>) {
    let mut bound: Vec<(Chord, &'static Command, bool)> = Vec::new();
    let mut conflicts = Vec::new();
    for c in COMMANDS.iter().filter(|c| c.built()) {
        for chord in c.chords() {
            bound.push((chord, c, false));
        }
    }
    for line in user {
        let held = bound.iter().position(|(ch, _, _)| *ch == line.chord);
        let Some(id) = line.command else {
            // `none`: the chord is taken off whatever held it, which is what was asked.
            if let Some(at) = held {
                bound.remove(at);
            }
            continue;
        };
        let Some(c) = by_id(id) else { continue };
        if let Some(at) = held {
            let (_, earlier, from_file) = bound.remove(at);
            if earlier.id != c.id {
                let name = |c: &Command| c.display_name.trim_end_matches("...").to_string();
                conflicts.push(Conflict {
                    chord: line.chord.clone(),
                    commands: vec![earlier.id, c.id],
                    why: if from_file {
                        format!("{} is bound twice in the file; the later line, {}, wins", line.chord.label(), name(c))
                    } else {
                        format!("{} was {}'s; the file gives it to {}", line.chord.label(), name(earlier), name(c))
                    },
                });
            }
        }
        bound.push((line.chord.clone(), c, true));
    }
    let bound: Vec<(Chord, &'static Command)> = bound.into_iter().map(|(ch, c, _)| (ch, c)).collect();
    for (chord, c) in &bound {
        if reserved_by_navigation(chord) {
            conflicts.push(Conflict {
                chord: chord.clone(),
                commands: vec![c.id],
                why: format!("{} belongs to the navigation, which wins", chord.label()),
            });
        } else if reserved_while_pending(chord) {
            conflicts.push(Conflict {
                chord: chord.clone(),
                commands: vec![c.id],
                why: format!(
                    "{} is the tool panel's while a tool is pending, so {} does not run then",
                    chord.label(),
                    c.display_name
                ),
            });
        }
    }
    (bound, conflicts)
}

/// Why `chord` may not be given to `id` from the panel, if it may not: another command holds it
/// (refused, never shared -- §9.2), or the navigation owns it.
pub fn refusal(user: &[Line], id: &str, chord: &Chord) -> Option<String> {
    if reserved_by_navigation(chord) {
        return Some(format!("{} belongs to the navigation", chord.label()));
    }
    let (bound, _) = effective(user);
    bound
        .iter()
        .find(|(ch, c)| ch == chord && c.id != id)
        .map(|(_, c)| format!("{} is {}'s — take it off there first", chord.label(), c.display_name.trim_end_matches("...")))
}

/// `user` with `id` bound to `chord`, replacing what it had: what a rebind from the panel writes.
pub fn rebound(user: &[Line], id: &'static str, chord: Chord) -> Vec<Line> {
    let mut out: Vec<Line> = user.iter().filter(|l| l.chord != chord && l.command != Some(id)).cloned().collect();
    // Take the command's defaults off, so it answers to the chord given and nothing else.
    if let Some(c) = by_id(id) {
        for default in c.chords() {
            if default != chord && !out.iter().any(|l| l.chord == default) {
                out.push(Line { chord: default, command: None });
            }
        }
    }
    out.push(Line { chord, command: Some(id) });
    out
}

/// `user` with every binding of `id` taken off, defaults included.
pub fn cleared(user: &[Line], id: &'static str) -> Vec<Line> {
    let mut out: Vec<Line> = user.iter().filter(|l| l.command != Some(id)).cloned().collect();
    if let Some(c) = by_id(id) {
        for default in c.chords() {
            if !out.iter().any(|l| l.chord == default) {
                out.push(Line { chord: default, command: None });
            }
        }
    }
    out
}

/// `user` with nothing of its own about `id`: the command answers to its defaults again.
pub fn reset(user: &[Line], id: &'static str) -> Vec<Line> {
    let defaults: Vec<Chord> = by_id(id).map(|c| c.chords()).unwrap_or_default();
    user.iter().filter(|l| l.command != Some(id) && !(l.command.is_none() && defaults.contains(&l.chord))).cloned().collect()
}

/// Where a person's bindings file is: beside the settings file.
pub fn path() -> Option<std::path::PathBuf> {
    crate::display_quality::settings_path().map(|p| p.with_file_name("bindings.txt"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chord(text: &str) -> Chord {
        Chord::parse(text).unwrap()
    }

    /// The file reads what it can and names what it cannot, by line.
    #[test]
    fn a_bindings_file_reads_every_line_it_can_and_names_the_rest() {
        let (lines, errors) = parse("# a comment\nCtrl+D = edit.duplicate\n\nAlt+P = palette.open  # trailing\nSlash = none\nCtrl+Q = no.such.thing\nnonsense\nHyper+Z = edit.undo\n");
        assert_eq!(lines, vec![
            Line { chord: chord("Ctrl+D"), command: Some("edit.duplicate") },
            Line { chord: chord("Alt+P"), command: Some("palette.open") },
            Line { chord: chord("Slash"), command: None },
        ]);
        assert_eq!(errors.len(), 3, "{errors:?}");
        assert!(errors[0].starts_with("line 6"), "{errors:?}");
        let (again, none) = parse(&write(&lines));
        assert_eq!(again, lines, "what it writes it reads back");
        assert!(none.is_empty());
    }

    /// **T10: two commands on one chord is a conflict, listed; a rebind to a chord in use is
    /// refused; a chord the pending tool's panel reads first is listed** (`interaction-spec.md`
    /// §9.2, §11).
    #[test]
    fn t10_conflicts_are_detected_refused_and_listed() {
        // The file binds Ctrl+Z, which is Undo's, to Redo: the later wins, and it is listed.
        let (lines, _) = parse("Ctrl+Z = edit.redo\n");
        let (bound, conflicts) = effective(&lines);
        assert_eq!(bound.iter().find(|(c, _)| *c == chord("Ctrl+Z")).map(|(_, c)| c.id), Some("edit.redo"));
        assert!(conflicts.iter().any(|c| c.chord == chord("Ctrl+Z") && c.commands == ["edit.undo", "edit.redo"]), "{conflicts:?}");
        // From the panel, a chord in use is refused by name, and a free one is not.
        assert_eq!(
            refusal(&[], "edit.duplicate", &chord("Ctrl+S")).as_deref(),
            Some("Ctrl+S is Save's — take it off there first")
        );
        assert_eq!(refusal(&[], "edit.duplicate", &chord("Ctrl+Alt+D")), None);
        assert_eq!(refusal(&[], "edit.undo", &chord("Ctrl+Z")), None, "a command may keep its own");
        // A binding on what the pending tool's panel reads first cannot always fire, and says so.
        let (_, conflicts) = effective(&parse("5 = view.frame_selection\nEnter = file.save\n").0);
        assert!(conflicts.iter().any(|c| c.chord == chord("Enter") && c.why.contains("tool panel")), "{conflicts:?}");
        assert!(conflicts.iter().any(|c| c.chord == chord("5")), "{conflicts:?}");
        // The defaults collide only where the spec means them to: Tab moves between a pending
        // tool's rows rather than cycling the display mode (§4.2), and that is listed.
        let defaults = effective(&[]).1;
        assert!(defaults.iter().all(|c| c.chord == chord("Tab")), "{defaults:?}");
    }

    /// Rebinding moves a command to the chord given and off its defaults; clearing takes it off
    /// everything; resetting gives it its defaults back.
    #[test]
    fn rebind_clear_and_reset_move_one_command_and_nothing_else() {
        let user = rebound(&[], "view.frame_selection", chord("Ctrl+F"));
        let (bound, _) = effective(&user);
        let frame: Vec<&Chord> = bound.iter().filter(|(_, c)| c.id == "view.frame_selection").map(|(ch, _)| ch).collect();
        assert_eq!(frame, vec![&chord("Ctrl+F")], "on the chord given, off its default");
        let user = cleared(&user, "view.frame_selection");
        assert!(effective(&user).0.iter().all(|(_, c)| c.id != "view.frame_selection"));
        let user = reset(&user, "view.frame_selection");
        assert!(user.is_empty(), "nothing of its own left: {user:?}");
        assert_eq!(effective(&user).0.len(), effective(&[]).0.len());
    }
}
