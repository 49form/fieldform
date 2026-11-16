// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Keyboard shortcuts.

use crate::*;

impl App {
    /// Translates this frame's key presses through the keymap and dispatches whatever they
    /// are bound to. Nothing here knows what Ctrl+Z *means* — that lives in the keymap, which
    /// is why it can be remapped from user preferences later without touching this code.
    pub(crate) fn handle_keys(&mut self, ui: &egui::Ui) {
        // **Keys being captured for a binding are the panel's, and nothing else's** (§9.2): the
        // next chord pressed is the one given, whatever it is bound to now.
        if self.keys.capturing.is_some() {
            let pressed = ui.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Key { key, pressed: true, modifiers, .. } => Some((*key, *modifiers)),
                    _ => None,
                })
            });
            if let Some((key, modifiers)) = pressed {
                ui.ctx().input_mut(|i| i.consume_key(modifiers, key));
                self.captured(commands::Chord {
                    key: key.name().to_string(),
                    ctrl: modifiers.ctrl || modifiers.command,
                    shift: modifiers.shift,
                    alt: modifiers.alt,
                });
            }
            return;
        }
        // A field with the keyboard owns the plain keys, because there they are text: Tab
        // moves between entries, and cycling the viewport behind them as well is what typing a
        // coordinate and tabbing to the next one used to do.
        //
        // Only the *unmodified* ones. Ctrl+S while a number is selected is still a save — no
        // text entry means anything by it, and a shortcut that stops working depending on
        // where the caret happens to be is its own kind of confusing.
        let typing = ui.ctx().egui_wants_keyboard_input();

        // Key events, kept alongside the egui key they came from so a binding can *consume*
        // its key. Consuming is the point: Tab is also egui's own focus traversal, so a Tab
        // this app acts on has to be taken off the queue, or one press both cycles the mode
        // and walks the focus into the menu bar — which is exactly what it did.
        let pressed: Vec<(egui::Key, egui::Modifiers, Shortcut)> = ui.input(|i| {
            let m = Modifiers {
                ctrl: i.modifiers.ctrl || i.modifiers.command,
                shift: i.modifiers.shift,
                alt: i.modifiers.alt,
            };
            i.events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Key { key, pressed: true, modifiers, .. } => {
                        Some((*key, *modifiers, Shortcut::new(key.name(), m)))
                    }
                    _ => None,
                })
                .collect()
        });

        // **Typing into a pending session's panel needs no click** (`interaction-spec.md` §4.3):
        // a digit or the first letter of a parameter's name starts it, and the plain keys this
        // frame are then text, not bindings -- `h` starting `height` must not also hide.
        let typed_into_panel = self.panel_typed(ui.ctx()) || self.cage_panel_typed(ui.ctx()) || self.cage_tool_panel_typed(ui.ctx());

        // The Tab that `raw_input_hook` took off the queue, dispatched here so it still goes
        // through the keymap and can still be rebound — the interception is about *when* egui
        // sees the key, not about who decides what it means. While a session is pending it moves
        // between the panel's rows instead (§4.2).
        let knife_armed = self.cage_tool.as_ref().is_some_and(|t| t.command.op == "cage.cut");
        if let Some(modifiers) = self.keys.stolen_tab.take() {
            if self.cage_session.is_some() && !modifiers.ctrl && !modifiers.alt {
                // A pending extrude or inset: Tab walks its panel's rows.
                self.cage_panel_tab(modifiers.shift);
            } else if self.cage_tool_has_panel() && !modifiers.ctrl && !modifiers.alt {
                // An armed click tool's settings window, the same way.
                self.cage_tool_panel_tab(modifiers.shift);
            } else if self.tools.session.is_some() && !modifiers.ctrl && !modifiers.alt {
                // With no panel, Tab with nothing focused goes back to the pending block's first
                // number, so the keyboard can always find its way to the block being made.
                if self.tools.floating_panel {
                    self.panel_tab(modifiers.shift);
                } else if let Some(session) = self.tools.session.as_mut() {
                    session.awaiting = true;
                }
            } else if let Some(binding) = self.keys.keymap.lookup(&Shortcut::new("Tab", modifiers)) {
                let (op, args) = (binding.op.clone(), binding.args.clone());
                self.run_binding(&op, &args);
            }
        }

        for (key, egui_modifiers, shortcut) in pressed {
            // **Space with an extrude or inset pending keeps it and lifts the next** (the owner,
            // 2026-11-12), ahead of the palette's binding and of the panel's own field, whose
            // number it takes first. Everywhere else Space is still the palette.
            if key == egui::Key::Space && !shortcut.modifiers.any() && self.space_chains_cage_session(ui.ctx()) {
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                self.cage_session_again(ui.ctx());
                continue;
            }
            if (typing || typed_into_panel) && !shortcut.modifiers.any() {
                continue;
            }
            // **Enter keeps a pending session and Esc takes it back** (`interaction-spec.md` §3.4,
            // §3.5), ahead of any binding: they are what a session is left by, and nothing else
            // is blocked while one is pending.
            if self.tools.session.is_some() && !shortcut.modifiers.any() && matches!(key, egui::Key::Enter | egui::Key::Escape) {
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                if key == egui::Key::Enter {
                    self.commit_session();
                } else {
                    self.cancel_session();
                }
                continue;
            }
            // **Enter keeps a pending extrude or inset and Esc takes it back**, as a block's
            // session is kept or taken back (R7).
            if self.cage_session.is_some() && !shortcut.modifiers.any() && matches!(key, egui::Key::Enter | egui::Key::Escape) {
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                if key == egui::Key::Enter {
                    self.keep_cage_session();
                } else {
                    self.cancel_cage_session();
                }
                continue;
            }
            // **The knife's path**: Enter cuts it, Backspace takes a point back, and Esc
            // discards it before a second Esc puts the knife down.
            if knife_armed && !shortcut.modifiers.any() && matches!(key, egui::Key::Enter | egui::Key::Backspace) {
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                if key == egui::Key::Enter {
                    self.knife_commit();
                } else {
                    self.knife_back();
                }
                continue;
            }
            // **Esc puts an armed cage tool down** (R7), and nothing else: a tool has nothing
            // pending between clicks, so there is nothing to take back -- but the knife's
            // path, which the first Esc discards.
            if self.cage_tool.is_some() && !shortcut.modifiers.any() && key == egui::Key::Escape {
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                if self.knife_discard() {
                    continue;
                }
                if let Some(tool) = self.cage_tool.take() {
                    self.say(format!("{} put down", tool.command.display_name));
                }
                continue;
            }
            // **Esc stops a sketch**, when nothing begun since takes the key first: a curve being
            // placed, a drag, a tool. Each of those is the thing most recently begun, and Esc
            // means the most recent thing.
            if !shortcut.modifiers.any() && key == egui::Key::Escape && self.escape_stops_sketching() {
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                continue;
            }
            if let Some(binding) = self.keys.keymap.lookup(&shortcut) {
                let (op, args) = (binding.op.clone(), binding.args.clone());
                ui.ctx().input_mut(|i| i.consume_key(egui_modifiers, key));
                let op = self.shortcut_op(op);
                self.run_binding(&op, &args);
            }
        }

        // **Ctrl+C, Ctrl+X and Ctrl+V never arrive as keys.** egui-winit turns them into
        // `Copy`, `Cut` and `Paste` events and drops the key event on the floor
        // (`on_keyboard_input` returns before pushing it), so a keymap bound to the key alone
        // was unreachable from the keyboard: Paste worked from the Edit menu and from nowhere
        // else, and so did Copy and Cut. Read back as the shortcuts they are, so a rebinding
        // in the keymap still decides what they do. A field with the keyboard keeps them,
        // since there they are text.
        if !typing {
            let clipboard: Vec<Shortcut> = ui.input(|i| {
                i.events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::Copy => Some(Shortcut::new("C", Modifiers::ctrl())),
                        egui::Event::Cut => Some(Shortcut::new("X", Modifiers::ctrl())),
                        egui::Event::Paste(_) => Some(Shortcut::new("V", Modifiers::ctrl())),
                        _ => None,
                    })
                    .collect()
            });
            for shortcut in clipboard {
                if let Some(binding) = self.keys.keymap.lookup(&shortcut) {
                    let (op, args) = (binding.op.clone(), binding.args.clone());
                    let op = self.shortcut_op(op);
                    self.run_binding(&op, &args);
                }
            }
        }
    }

    /// What a shortcut means where the cursor is: Ctrl+V lands the copied curve under the
    /// cursor when the cursor is over the viewport (`profile-group-spec.md` §4.4). The same
    /// word from the Edit menu, a row's menu or the console pastes in place, because none of
    /// those has a cursor over the viewport when it fires.
    pub(crate) fn shortcut_op(&self, op: String) -> String {
        if op == "edit.paste" && self.viewport_hover.is_some() {
            return "edit.paste_here".to_string();
        }
        // **Keys by selection** (R9): with cage vertices, edges or faces picked, the letter that
        // extrudes a profile extrudes the cage, the one that deletes a body deletes faces, and so
        // on -- the same letters, meaning the cage while the cage is what is selected.
        // K is the exception: the knife needs no selection to start, so on the cage it is the
        // knife always, and Split is what K means only off the cage (the owner, 2026-10-28).
        if self.control == ControlMesh::Cage && op == "modify.split" {
            return "cage.cut".to_string();
        }
        if self.control == ControlMesh::Cage && !self.selection.is_empty() {
            let faces = self.selection.parts().iter().any(|s| matches!(s, Selection::Face(_)));
            if let Some(cage_op) = commands::cage_element_op(&op, faces) {
                return cage_op.to_string();
            }
        }
        op
    }
}
