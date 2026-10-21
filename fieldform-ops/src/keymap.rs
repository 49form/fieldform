// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Key bindings, as data rather than code.
//!
//! Deliberately independent of any UI toolkit: keys are named by string, so a keymap
//! round-trips through a user preferences file without egui (or any other windowing layer)
//! leaking into the format. The app translates its toolkit's key events into these names.

use crate::Args;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Modifiers {
    #[serde(default)]
    pub ctrl: bool,
    #[serde(default)]
    pub shift: bool,
    #[serde(default)]
    pub alt: bool,
}

impl Modifiers {
    pub(crate) const NONE: Self = Self {
        ctrl: false,
        shift: false,
        alt: false,
    };

    /// Whether any modifier is held.
    ///
    /// Used to tell a shortcut from a keystroke that is also text: an unmodified key belongs to
    /// whatever has the keyboard, while Ctrl+S means the same wherever the caret is.
    pub fn any(self) -> bool {
        self.ctrl || self.shift || self.alt
    }

    pub fn ctrl() -> Self {
        Self {
            ctrl: true,
            ..Self::NONE
        }
    }

    pub fn shift() -> Self {
        Self {
            shift: true,
            ..Self::NONE
        }
    }

    pub fn ctrl_shift() -> Self {
        Self {
            ctrl: true,
            shift: true,
            alt: false,
        }
    }
}

/// A key plus its modifiers. `key` is a canonical name like `"Z"`, `"Tab"`, or `"Slash"`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shortcut {
    pub key: String,
    #[serde(default)]
    pub modifiers: Modifiers,
}

impl Shortcut {
    pub fn new(key: &str, modifiers: Modifiers) -> Self {
        Self {
            key: key.to_string(),
            modifiers,
        }
    }

    /// A shortcut with no modifiers held.
    pub fn plain(key: &str) -> Self {
        Self::new(key, Modifiers::NONE)
    }
}

/// One binding: a shortcut, the operation it invokes, and the arguments to invoke it with.
///
/// Carrying `args` is what lets one operation serve several bindings — the classic example
/// being an operation with a direction or amount parameter bound to different keys.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Binding {
    pub shortcut: Shortcut,
    pub op: String,
    #[serde(default)]
    pub args: Args,
}

/// The set of active bindings.
///
/// Later bindings win over earlier ones for the same shortcut, so a user preferences file
/// loaded after the defaults can override any of them without having to remove the default
/// first.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Keymap {
    bindings: Vec<Binding>,
}

impl Keymap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind(&mut self, shortcut: Shortcut, op: &str, args: Args) {
        self.bindings.push(Binding {
            shortcut,
            op: op.to_string(),
            args,
        });
    }

    /// Finds the binding for a shortcut, if any. Searches in reverse so later (user)
    /// bindings shadow earlier (default) ones.
    pub fn lookup(&self, shortcut: &Shortcut) -> Option<&Binding> {
        self.bindings.iter().rev().find(|b| &b.shortcut == shortcut)
    }

    /// Every binding, in the order they shadow: the keybinding panel lists them, and a conflict
    /// is two of them on one shortcut.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// Merges another keymap on top of this one, with the other's bindings taking priority.
    ///
    /// The merge a person's bindings file goes through: their keys over the registry's defaults.
    pub fn extend(&mut self, other: Keymap) {
        self.bindings.extend(other.bindings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A few bindings of the shape the application's defaults have. The defaults themselves are
    /// the application's command registry's now (`fieldform-app/src/commands.rs`), so this crate
    /// holds the mechanism and tests it on a keymap of its own.
    fn sample() -> Keymap {
        let mut km = Keymap::new();
        km.bind(Shortcut::new("Z", Modifiers::ctrl()), "edit.undo", Args::new());
        km.bind(Shortcut::new("Z", Modifiers::ctrl_shift()), "edit.redo", Args::new());
        km.bind(Shortcut::new("X", Modifiers::shift()), "view.section_flip", Args::new());
        km.bind(Shortcut::plain("Tab"), "view.cycle_mode", Args::new());
        km
    }

    #[test]
    fn lookup_matches_modifiers_exactly() {
        let km = sample();

        // Ctrl+Z and Ctrl+Shift+Z are different bindings, not the same key.
        let undo = km.lookup(&Shortcut::new("Z", Modifiers::ctrl())).unwrap();
        assert_eq!(undo.op, "edit.undo");
        let redo = km
            .lookup(&Shortcut::new("Z", Modifiers::ctrl_shift()))
            .unwrap();
        assert_eq!(redo.op, "edit.redo");

        // Bare Z isn't bound, and must not fall through to the Ctrl+Z binding.
        assert!(km.lookup(&Shortcut::plain("Z")).is_none());
    }

    #[test]
    fn later_bindings_shadow_earlier_ones() {
        // The mechanism that lets user preferences override defaults.
        let mut km = sample();
        assert_eq!(
            km.lookup(&Shortcut::plain("Tab")).unwrap().op,
            "view.cycle_mode"
        );

        let mut user = Keymap::new();
        user.bind(Shortcut::plain("Tab"), "something.else", Args::new());
        km.extend(user);

        assert_eq!(
            km.lookup(&Shortcut::plain("Tab")).unwrap().op,
            "something.else"
        );
    }

    #[test]
    fn a_modifier_is_what_separates_a_shortcut_from_a_keystroke() {
        // The app suppresses unmodified shortcuts while a text field has the keyboard — Tab
        // belongs to the field it is moving between, not to the viewport behind it — and this
        // is the test that decides which is which.
        assert!(!Modifiers::default().any(), "a bare key is text");
        assert!(Modifiers::ctrl().any());
        assert!(Modifiers::ctrl_shift().any());
        assert!(Modifiers { shift: true, ..Modifiers::default() }.any());
        assert!(Modifiers { alt: true, ..Modifiers::default() }.any());
    }

    #[test]
    fn keymap_round_trips_through_json() {
        // Keymaps have to survive a user preferences file.
        let km = sample();
        let json = serde_json::to_string(&km).unwrap();
        let back: Keymap = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.lookup(&Shortcut::new("Z", Modifiers::ctrl())).unwrap().op,
            "edit.undo"
        );
        assert_eq!(back.bindings().len(), km.bindings().len());
    }
}
