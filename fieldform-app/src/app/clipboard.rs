// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Copy, cut, duplicate and paste.

use crate::*;

impl App {
    /// Puts the selection on the clipboard: a curve, or a body with everything it is made of
    /// ([`copied_body`] says what that takes).
    pub(crate) fn copy_selection(&mut self, announce: bool) {
        if let Some(id) = self.selected_curve
            && let Some(curve) = self.doc.curve(id).cloned()
        {
            self.clipboard = Some(Clipboard::Curve(curve));
            if announce {
                self.say("Copied the curve");
            }
            return;
        }
        let Some(root) = self.selected_body else {
            self.warn("Nothing selected");
            return;
        };
        let label = self.body_label(root);
        self.clipboard = Some(copied_body(&self.doc, self.part_graph(), root));
        if announce {
            self.say(format!("Copied {label}"));
        }
    }

    /// Copies, then deletes.
    pub(crate) fn cut_selection(&mut self) {
        self.copy_selection(true);
        if self.clipboard.is_some() {
            self.delete_selection();
        }
    }

    /// A second, independent copy of what is selected, beside it.
    ///
    /// **Copy and paste with the clipboard put back**, rather than a third traversal of the
    /// graph. The pair already does everything a duplicate has to: fresh ids for every node,
    /// cage and curve underneath, so the copy shares nothing with what it came from, and the
    /// whole of it in one undo step. Writing it again would be a second answer to the question
    /// "what does copying a body mean", and the two would drift.
    ///
    /// What it must not do is spend the clipboard. That belongs to whoever put something in
    /// it, and losing it to a menu item used for something else is the kind of small theft
    /// nobody reports and everybody notices.
    pub(crate) fn duplicate_selection(&mut self) {
        let held = self.clipboard.take();
        self.copy_selection(false);
        if self.clipboard.is_none() {
            // `copy_selection` has already said why there was nothing to take.
            self.clipboard = held;
            return;
        }
        self.paste(false);
        self.clipboard = held;
        // Said once, here, rather than twice by the two halves this is made of.
        match (self.selected_curve, self.selected_body) {
            (Some(_), _) => self.say("Duplicated the curve"),
            (_, Some(id)) => {
                let label = self.body_label(id);
                self.say(format!("Duplicated as {label} — drag it off the original"));
            }
            _ => {}
        }
    }

    /// Pastes whatever was copied, as something independent of it.
    ///
    /// In place rather than offset by a guess. The copy is selected on arrival and the
    /// manipulator is already on it, so moving it is the next gesture rather than a number this
    /// has to invent.
    pub(crate) fn paste(&mut self, announce: bool) {
        let Some(clipboard) = self.clipboard.clone() else {
            self.warn("Nothing copied");
            return;
        };
        match clipboard {
            Clipboard::Curve(curve) => {
                let id = CurveId::new();
                let Ok(fresh) = curve.copy_as(id, format!("{} copy", curve.name)) else {
                    self.warn("Could not copy that curve");
                    return;
                };
                self.insert_pasted_curve(fresh, announce.then(|| "Pasted the curve".to_string()));
            }
            Clipboard::Body { nodes, root, cages, curves } => {
                // What `set_part_graph` does around a graph edit, which this is as well, though it
                // goes on the history as a batch with the curves ([`paste_body`] says why). The
                // gestures end because the entry on top is no longer theirs to fold into.
                self.splash_shown = false;
                self.material_gesture = None;
                self.gesture = None;
                let part = self.active_part;
                let pasted =
                    paste_body(&mut self.stack, &mut self.doc, part, &nodes, root, &cages, &curves);
                let new_root = match pasted {
                    Ok(id) => id,
                    Err(why) => {
                        self.warn(why);
                        return;
                    }
                };
                self.stamp_new_nodes();
                self.select_body(Some(new_root));
                self.mark_changed();
                self.rebuild_curve_display();
                self.rebuild_render_resources();
                // **Loose, as a pasted curve is** (`insert_pasted_curve`). The copies carry no
                // places, and a loop they close is found here without being held, so no join
                // lands on the history after the paste as an entry of its own.
                self.refresh_profiles_with(false);
                if announce {
                    self.say("Pasted — drag it to move it off the original");
                }
            }
        }
    }

    /// Puts a pasted curve into the document and says where it landed.
    ///
    /// **Loose, whatever it touches.** A copy carries no places, so its ends sit on nothing: a
    /// cut curve pasted straight back touches its loop again and the row re-forms, but nothing
    /// rejoins behind your back (`profile-group-spec.md` §4.4). The row marks it "not held",
    /// and `join` holds it. The loop it may have re-formed is reported here, before the block
    /// list would find it, so that the finding is the one that does not join.
    pub(crate) fn insert_pasted_curve(&mut self, fresh: fieldform_scene::Curve, said: Option<String>) {
        let id = fresh.id;
        // Not a graph edit, so it ends any gesture a value drag had open, as a placed curve does.
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(SetCurve::insert(fresh))) {
            self.warn(e.to_string());
            return;
        }
        self.select_curve(id);
        self.mark_changed();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        self.refresh_profiles_with(false);
        if let Some(mut line) = said {
            if let Some(key) = self.curve_ui.profiles.key_of(id) {
                line.push_str(&format!(
                    " — it touches {} but is not held; join holds it",
                    self.profile_label(key)
                ));
            }
            self.say(line);
        }
    }

    /// Pastes the copied curve with its start under the cursor, moved as one piece.
    ///
    /// The viewport's **Paste here**, and Ctrl+V while the cursor is over the viewport
    /// (`profile-group-spec.md` §4.4). A click is a place the person chose, which is what
    /// [`Self::paste`] rightly refuses to guess at; where it lands is decided by
    /// `landing_point`, exactly as a placed curve's clicks are.
    pub(crate) fn paste_here(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let Some(Clipboard::Curve(curve)) = self.clipboard.clone() else {
            self.warn("Nothing copied that can be put down at a point — copy a curve first");
            return;
        };
        let at = match self.landing_under(aspect, rect, pos) {
            Ok(at) => at,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let id = CurveId::new();
        let Ok(fresh) = moved_to_start_at(&curve, id, format!("{} copy", curve.name), at) else {
            self.warn("Could not copy that curve");
            return;
        };
        self.insert_pasted_curve(
            fresh,
            Some(format!(
                "Pasted the curve with its start at {:.1}, {:.1}, {:.1}",
                at.x, at.y, at.z
            )),
        );
    }

    /// Ctrl+V: a curve lands under the cursor when the cursor is over the viewport; anything
    /// else, or anywhere else, pastes in place as it always did.
    pub(crate) fn paste_at_hover(&mut self) {
        match (self.viewport_hover, &self.clipboard) {
            (Some((aspect, rect, pos)), Some(Clipboard::Curve(_))) => {
                self.paste_here(aspect, rect, pos)
            }
            _ => self.paste(true),
        }
    }
}
