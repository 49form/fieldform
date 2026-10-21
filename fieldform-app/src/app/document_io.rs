// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! New, open and save, and the autosave: the ways a document reaches and leaves a file.

use crate::*;

impl App {
    /// Starts an empty document, after asking about anything unsaved.
    ///
    /// The same state the application opens in — one cage in the document, no body drawing it,
    /// so Create > Cage puts *that* cage back rather than making a second. Built by the same
    /// two lines `new` uses rather than by a fresh path, because "what a new document is" must
    /// have exactly one answer; two of them drift the first time either is changed.
    pub(crate) fn new_document(&mut self) {
        if !self.may_discard("starting a new document") {
            return;
        }
        let cage = Cage::cube_of(DEFAULT_BODY_MM);
        let mut doc = Document::with_cage(cage);
        let active_part = *doc.parts.keys().next().expect("with_cage inserts one");
        doc.parts
            .get_mut(&active_part)
            .expect("with_cage inserts one")
            .graph = Graph::empty();
        // A graph written past the history settles like any other (`block-ownership-spec.md`
        // §4.1). An empty one has nothing to place, so this moves nothing and leaves nothing to
        // say, and it keeps the rule without an exception to remember.
        doc.settle_ownership();

        self.active_cage = *doc.cages.keys().next().expect("with_cage inserts one");
        self.active_part = active_part;
        self.doc = doc;
        // No path, so the next Save asks for one. Carrying the old document's path forward
        // would make Save overwrite a file this document was never opened from.
        self.file = None;
        self.selection.clear();
        self.hover = Selection::None;
        self.move_state = None;
        self.pending_curve = None;
        // Undo describes edits to the document that was just replaced; reversing one against
        // this one would apply it to geometry that is not there. Same reason `open_document`
        // clears it.
        self.stack.clear();
        self.selected_curve = None;
        // Nothing is drawn, so nothing may still claim to be. These outlive the graph that
        // set them otherwise, which is the shape of bug this codebase keeps meeting.
        // Nothing carried over: a new document's bodies are its own, and the slot that
        // remembers what is hidden travels with the document rather than with the session.
        self.selected_body = None;
        self.boolean_a = None;
        self.boolean_b = None;
        self.profiles = ProfileRows::default();
        self.rebuild_curve_display();
        self.rebuild_render_resources();
        // Clean rather than dirty: an empty document has nothing worth saving, and marking it
        // dirty would prompt on the way out of a document nobody edited.
        self.mark_saved();
        self.camera = OrbitCamera::default();
        self.say("New document");
    }

    /// Opens a native document, replacing everything currently being edited.
    ///
    /// Opens `asked`, or asks for a path when nothing names one — `open file="C:/p.fform"`
    /// opens it without the dialog, as `saveas file=` writes without one.
    pub(crate) fn open_document(&mut self, asked: Option<std::path::PathBuf>) {
        if !self.may_discard("opening another document") {
            return;
        }
        let path = match asked {
            Some(path) => path,
            None => {
                if !self.may_open_a_dialog("open", "open file=<path>") {
                    return;
                }
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("fieldform document", DOCUMENT_EXTENSIONS)
                    .pick_file()
                else {
                    return; // user cancelled
                };
                path
            }
        };

        match fieldform_io::open(&path) {
            Ok((doc, archive)) => {
                let Some(part) = doc.parts.keys().next().copied() else {
                    self.warn("That document has no parts");
                    return;
                };
                let Some(cage) = doc
                    .parts
                    .get(&part)
                    .and_then(|p| p.graph.referenced_cages().first().copied())
                    .or_else(|| doc.cages.keys().next().copied())
                else {
                    self.warn("That document has no cage to edit");
                    return;
                };

                self.doc = doc;
                self.active_part = part;
                self.active_cage = cage;
                // Quietly: whatever opening the file settled is where its rows already were,
                // and a file is not changed by being read (`block-ownership-spec.md` §9.2).
                // After the active part is this document's, since naming a block can fall back
                // to describing it by the active part's graph.
                self.drain_ownership_changes(true);
                self.file = Some((path.clone(), archive));
                self.selection.clear();
                self.hover = Selection::None;
                self.move_state = None;
                self.pending_curve = None;
                // The undo stack describes edits to the document that was just replaced, and
                // reversing one against this one would apply it to unrelated geometry.
                self.stack.clear();
                self.selected_curve = self.doc.curves.keys().next().copied();
                // The loops this file holds were closed before it was opened, so the first
                // partition is computed quietly rather than reported as edits.
                self.profiles = ProfileRows::default();
                // **Relations are checked, not enforced** (`profile-group-spec.md` §5.4): one
                // that does not hold -- the file was edited by a build that could not keep it
                // true -- is kept, marked with why, and not applied. A file that opened with a
                // slightly different shape than it saved with would be wrong. `realign`
                // re-derives it as an ordinary undoable edit.
                let broken = fieldform_scene::relation::check(&self.doc);
                for (id, why) in &broken {
                    if let Some(record) = self.doc.relations.get_mut(id) {
                        record.broken = Some(why.clone());
                    }
                }
                for (_, why) in &broken {
                    self.warn(format!("relation broken: {why} — `realign` re-derives it"));
                }
                // Opening a document shows it, which means opening into a mode that can draw
                // it. The mode used to be left at whatever it already was -- and at launch
                // that is Polygon, which draws the control cage and nothing else. A part built
                // from primitives has no cage node at all, so a perfectly good file opened to
                // an empty viewport, ignored the camera, and came back only after some
                // unrelated edit happened to switch modes. Every Create command already sets
                // this; Open was the one that did not.
                self.view_mode = if self.part_graph().roots().is_empty()
                    && self.visible_cage_node().is_some()
                {
                    // Nothing to trace but a cage to edit: the one case Polygon is right for.
                    ViewMode::Polygon
                } else {
                    ViewMode::Field
                };
                // Driven coordinates catch up before anything reads a curve. A file stores
                // both the binding and the value it last resolved to, and the binding is the
                // truth -- so a document opened after its variable was edited elsewhere agrees
                // with the variable rather than with the number frozen beside it.
                self.settle_curves_on_points();
                self.rebuild_curve_display();
                self.rebuild_render_resources();
                // What is in memory is exactly what was read, so nothing is pending.
                self.mark_saved();
                if let Some((center, radius)) = self.scene_bounds() {
                    self.camera.frame(center, radius);
                }
                // A graph this build cannot evaluate still opens — that is the whole point of
                // the unknown-node rule — but it draws nothing, and a viewport showing nothing
                // has to say why rather than look broken.
                //
                // **Only that case warns.** A clean open is said, not warned: the console marks
                // a warning `!!` and a script run stops at the first one unless told to keep
                // going, so announcing a success through that channel made `open file=` read as
                // a refusal and would have ended a run that had done nothing wrong.
                match self.compile_error() {
                    Some(why) => self.warn(format!("Opened {} — {why}", file_name(&path))),
                    None => self.say(format!("Opened {}", file_name(&path))),
                }
            }
            Err(e) => self.warn(e.to_string()),
        }
    }

    /// Writes the document beside the logs when it has changed, at most every
    /// [`AUTOSAVE_EVERY`].
    ///
    /// **Item 35 step 2: whatever else happens, do not lose the document.** Losing the Vulkan
    /// device is indistinguishable from a crash — no panic, no dialog, and an exit code nobody
    /// sees — and step 1 made it say so in the crash log. This is what makes it survivable.
    ///
    /// **Why a timer on the frame path, and not only a write when the device is lost.** That
    /// callback is `'static` and cannot reach the document; and by the time it runs every GPU
    /// call already fails, so the application aborts on the next one and there may be no frame
    /// left to save in. The only save that can be relied on is one that already happened. Which
    /// is also why this is worth more than the callback it serves: it covers a TDR, a panic, a
    /// power cut and a killed window just as well, none of which get a chance to write anything.
    /// [`App::keep_the_document_if_the_gpu_was_lost`] covers the minute since, when the loss
    /// leaves a frame to do it in.
    ///
    /// **Never the document's own file.** An autosave that overwrote it would make a crash
    /// destroy the last state the person deliberately saved, which is the opposite of the point.
    /// A clean exit removes it ([`App::forget_autosave`]), so finding one at launch means the
    /// last session did not end cleanly rather than merely that there was one.
    pub(crate) fn autosave(&mut self) {
        // A test's `App` leaves nothing in the temp folder for a later session to offer.
        if cfg!(test) {
            return;
        }
        if !autosave_due(
            self.revision,
            self.autosaved_revision,
            self.is_dirty(),
            self.autosaved_at.elapsed(),
        ) {
            return;
        }
        // The clock restarts either way, so a failing disk is retried on the same cadence
        // rather than on every frame.
        self.autosaved_at = std::time::Instant::now();
        // Before any save, this one included (`block-ownership-spec.md` §4.2): a variable a settle
        // made since the frame began is named before the file is written, as `save_document`
        // does, so not even an autosave holds one that cannot be typed (I4).
        self.drain_ownership_changes(false);
        let path = autosave_path();
        let started = std::time::Instant::now();
        match fieldform_io::to_archive(&self.doc).and_then(|archive| archive.write(&path)) {
            // **Logged with what it cost, every time.** The interval is chosen against this
            // number, and it is the only way to notice the day a document gets heavy enough
            // that a minute's insurance costs a visible stutter. In the log, not the console:
            // it is a fact about the machinery, not about the model.
            Ok(()) => {
                self.autosaved_revision = self.revision;
                log::info!(
                    "autosaved to {} in {:.0} ms",
                    path.display(),
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
            // The log, not the console: a person who has not lost anything does not need to be
            // told about the arrangements, and one who has will be reading the log.
            Err(e) => log::warn!("autosave to {} failed: {e}", path.display()),
        }
    }

    /// Keeps the document in [`recovered_path`] once the GPU has been lost, and says where in the
    /// console, the log and the crash note (loop spec S3; item 35 step 2).
    ///
    /// **Here, on the interface thread, because the callback cannot do it.** wgpu runs the
    /// device-lost callback on whichever thread made the call that found the device gone, and the
    /// document lives in the application, behind no lock: the loop spec would take it behind a
    /// mutex kept for the console, and there is none. So the callback raises [`App::gpu_lost`]
    /// and asks for a frame, and this, called first and last in every frame, answers it. Until a
    /// loss, an atomic load is all it costs.
    ///
    /// **Everything since the last save, not since the last autosave.** The autosave is at most
    /// a minute old and is still the save that can be relied on, since it already happened when a
    /// loss leaves no frame at all; this is what keeps the minute since.
    pub(crate) fn keep_the_document_if_the_gpu_was_lost(&mut self) {
        if self.kept_after_loss || !self.gpu_lost.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        self.kept_after_loss = true;
        let said = if !self.is_dirty() {
            // Nothing unsaved, so the document's own file is already all of it, and a copy
            // beside it would only be a second file to wonder about.
            match &self.file {
                Some((path, _)) => format!(
                    "The GPU was lost with nothing unsaved: {} is the whole document.",
                    path.display()
                ),
                None => "The GPU was lost with nothing in the document to keep.".to_string(),
            }
        } else {
            // A variable a settle made since the frame began is named first, as every save does,
            // so the kept file holds none that cannot be typed (`block-ownership-spec.md` I4).
            self.drain_ownership_changes(false);
            match write_recovered(&self.doc, self.file.as_ref()) {
                // Forward slashes, so the line can be typed back as it reads: see
                // `offer_the_autosave`.
                Ok(kept) => format!(
                    "The GPU was lost. The document as it stood is kept at {} — \
                     `open file=\"{}\"` reads it.",
                    kept.display(),
                    kept.to_string_lossy().replace('\\', "/")
                ),
                Err(e) => format!(
                    "The GPU was lost, and keeping the document failed: {e}. The autosave at {} \
                     holds it as of the last minute.",
                    autosave_path().display()
                ),
            }
        };
        log::error!("{said}");
        note_in_crash_log("fieldform document kept", &said);
        self.warn(said);
    }

    /// Says that an autosave is waiting and **types the line that reads it into the console**,
    /// so recovering the work is one keystroke.
    ///
    /// **Said in the console, not only in the log.** A clean exit removes the autosave, so one
    /// being here means the last session died — which is the single most useful thing the
    /// application can say at that moment, and the log is not where a person looks.
    ///
    /// **Offered rather than opened.** Reading a file back over what someone asked for is a
    /// surprise, and the one case where it would be wrong — they meant to start fresh — is
    /// exactly the case where not pressing Enter is easy.
    ///
    /// The path is written with forward slashes. `Path::display` gives backslashes on Windows,
    /// and a line typed for someone to run has to be a line the parser reads back the same way;
    /// every `file=` example in this application is spelt with forward slashes for that reason.
    ///
    /// **Another session's file, and only offered** (see [`autosave_path`]): it may belong to an
    /// instance that is still running, so this neither removes nor moves it, and its own session
    /// is the only one that ever does.
    pub(crate) fn offer_the_autosave(&mut self) {
        let dir = std::env::temp_dir();
        let own = autosave_path()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let listing: Vec<(String, std::time::SystemTime)> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let at = entry.metadata().ok()?.modified().ok()?;
                Some((entry.file_name().to_string_lossy().into_owned(), at))
            })
            .collect();
        let others = other_autosaves(&listing, &own);
        let Some((newest, at)) = others.first() else { return };
        let path = dir.join(newest);
        let when = at
            .elapsed()
            .map_or_else(|_| String::new(), |d| format!(" from {}", how_long_ago(d)));
        let more = match others.len() {
            1 => String::new(),
            n => format!(" ({} older ones are beside it in {})", n - 1, dir.display()),
        };
        self.warn(format!(
            "An autosave{when} from another session is waiting — if that session crashed or was \
             closed, press Enter to open it, or clear the line to start fresh{more}."
        ));
        self.console_input = format!("open file=\"{}\"", path.to_string_lossy().replace('\\', "/"));
    }

    /// Removes **this session's** autosave, which a clean exit does so that its presence means
    /// something — and only this session's, so no exit can take another session's work with it.
    pub(crate) fn forget_autosave(&mut self) {
        let path = autosave_path();
        if path.exists() {
            if let Err(e) = std::fs::remove_file(&path) {
                log::warn!("the autosave {} could not be removed: {e}", path.display());
            }
        }
    }

    /// Saves the document, asking for a path when there is not one to reuse.
    ///
    /// Returns whether the document reached disk, which the discard prompts need: "save, then
    /// close" must not close when the save was cancelled at the file dialog or failed.
    pub(crate) fn save_document(&mut self, mode: SaveAs) -> bool {
        let existing = match (mode, &self.file) {
            (SaveAs::SamePath, Some((path, _))) => Some(path.clone()),
            (SaveAs::To(path), _) => Some(if path.extension().is_none() {
                path.with_extension(DOCUMENT_EXTENSION)
            } else {
                path
            }),
            _ => None,
        };
        let path = match existing {
            Some(path) => path,
            None => {
                if !self.may_open_a_dialog("saveas", "saveas file=<path>") {
                    return false;
                }
                let Some(path) = rfd::FileDialog::new()
                    // Saving offers the current extension only. Opening accepts both; writing
                    // is where a format stops accumulating names.
                    .add_filter("fieldform document", &[DOCUMENT_EXTENSION])
                    .set_file_name(format!("untitled.{DOCUMENT_EXTENSION}"))
                    .save_file()
                else {
                    return false; // user cancelled
                };
                // A name typed without one gets the extension; a name typed *with* one keeps
                // it, so "save over this .fieldform" saves over that file rather than beside it.
                if path.extension().is_none() {
                    path.with_extension(DOCUMENT_EXTENSION)
                } else {
                    path
                }
            }
        };

        // Saving over a document that was opened goes through *its* archive, so the entries
        // this build does not interpret — a plugin's data, the file an import came from —
        // are written back rather than dropped.
        //
        // Any other path gets a fresh archive, including one that already holds a document.
        // Those entries belong to *that* document's content, so carrying them across a Save As
        // would attach a plugin's data to geometry it was never about.
        //
        // A variable a settle made since the frame began is named first, so the file never holds
        // one that cannot be typed (`block-ownership-spec.md` I4).
        self.drain_ownership_changes(false);
        self.prune_profile_names();
        let result = match &mut self.file {
            Some((previous, archive)) if *previous == path => {
                fieldform_io::update_archive(archive, &self.doc)
                    .and_then(|()| archive.write(&path))
            }
            _ => fieldform_io::to_archive(&self.doc).and_then(|archive| {
                let written = archive.write(&path);
                if written.is_ok() {
                    self.file = Some((path.clone(), archive));
                }
                written
            }),
        };

        match result {
            Ok(()) => {
                self.mark_saved();
                self.say(format!("Saved {}", file_name(&path)));
                true
            }
            Err(e) => {
                self.warn(e.to_string());
                false
            }
        }
    }

    /// Whether a modal dialog may be opened, and the refusal when it may not.
    ///
    /// **A script never meets a dialog.** `--run` exists so the application can be driven with
    /// no hand on the window, and a dialog waiting to be clicked is not a slow line but a
    /// stopped run — `open file=` ignored its key, so a run sat on the file dialog for five and
    /// a half minutes having printed nothing, and the owner met it as the application being
    /// stuck. Which is the worst shape a fault can take here: the one thing the scripting path
    /// promises is that a run either says what happened or ends.
    ///
    /// So a command that would ask is refused by name instead, with the key that replaces it.
    /// The refusal is what a `--keep-going` run carries on past and what `--exit` counts.
    pub(crate) fn may_open_a_dialog(&mut self, verb: &str, instead: &str) -> bool {
        if self.script.is_none() {
            return true;
        }
        self.warn(format!(
            "`{verb}` would open a dialog, which a script cannot answer — say `{instead}`"
        ));
        false
    }

    /// Asks what to do about unsaved changes before something discards them, and returns
    /// whether to go ahead.
    ///
    /// Every path that replaces the document goes through here — starting a new one, opening
    /// another, closing the window. Saving exists now, so losing an hour of cage work to a
    /// misclicked menu item is a thing that can happen, and only a prompt stops it.
    pub(crate) fn may_discard(&mut self, before: &str) -> bool {
        if !self.is_dirty() {
            return true;
        }
        // **A script's own document is never worth keeping**, which is the rule a folder run
        // already follows between its benchmarks (`mark_saved` before `new`). There is nobody
        // to answer a prompt during a run, so the alternative to discarding is a stopped run.
        // Said in the log, so the record shows what was dropped rather than only what was done.
        if self.script.is_some() {
            self.say(format!("Discarding unsaved changes before {before}: a script said so."));
            return true;
        }
        let named = match &self.file {
            Some((path, _)) => file_name(path),
            None => "This document".to_string(),
        };
        // Yes/No/Cancel rather than Save/Discard/Cancel: rfd only relabels buttons on Windows
        // with its `common-controls-v6` feature, and a dialog that might not appear is worse
        // than one whose buttons are named by the question. So the question is phrased to make
        // yes and no unambiguous.
        let choice = rfd::MessageDialog::new()
            .set_title("Unsaved changes")
            .set_description(format!(
                "{named} has changes that are not saved.\n\nSave them before {before}?"
            ))
            .set_buttons(rfd::MessageButtons::YesNoCancel)
            .set_level(rfd::MessageLevel::Warning)
            .show();
        match choice {
            // Yes: save first, and only continue if it actually got written.
            rfd::MessageDialogResult::Yes => self.save_document(SaveAs::SamePath),
            // No: go ahead and lose them, which is what was asked for.
            rfd::MessageDialogResult::No => true,
            _ => false,
        }
    }
}
