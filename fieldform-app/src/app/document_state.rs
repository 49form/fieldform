// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Whether the document has changed, what follows when it does, and what the window says about it.

use crate::*;

impl App {
    /// Records that the document now differs from the file, and returns nothing else — the
    /// rebuild paths call this rather than touching `revision`, so a new kind of edit cannot
    /// be added without passing through here.
    pub(crate) fn mark_changed(&mut self) {
        self.saving.revision = self.saving.revision.wrapping_add(1);
        self.document_moved();
        // Not everything that adds something goes through `set_part_graph` — an import adds
        // curves and no nodes at all — but everything that changes the document says so here.
        self.stamp_new_nodes();
    }

    /// Says the document may have moved, so the overlay reads it again
    /// ([`Self::document_moves`]).
    ///
    /// **Called from the few places every change already passes through, not from each change:**
    /// marking an edit, writing the part's graph, handing out the graph or a control mesh to be
    /// written, redrawing the curves and rebuilding what is drawn. Each is there for a reason of
    /// its own that a missed call already shows — a file that does not ask to be saved, an edit
    /// undo cannot reach, a line left where it was, a picture left as it was — so a change that
    /// reaches none of them is wrong on screen before the readings are. The two accessors are the
    /// drags' way in, since a drag is recorded only when the hand comes off.
    pub(crate) fn document_moved(&mut self) {
        self.document_moves = self.document_moves.wrapping_add(1);
    }

    /// Whether there is anything in memory the file on disk does not have.
    pub(crate) fn is_dirty(&self) -> bool {
        self.saving.revision != self.saving.saved_revision
    }

    /// Called after the document has been written or read, so the two agree again.
    pub(crate) fn mark_saved(&mut self) {
        self.saving.saved_revision = self.saving.revision;
    }

    /// Called when another document has taken this one's place. Nothing is pending, so it is
    /// clean, but the revision still moves: it also says *which* document this is, and whoever
    /// holds the old number has read something that is gone.
    pub(crate) fn mark_replaced(&mut self) {
        self.saving.revision = self.saving.revision.wrapping_add(1);
        self.document_moved();
        self.mark_saved();
    }

    /// Anything that changes the document invalidates the derived meshes, and may have
    /// removed the geometry the current selection referred to.
    pub(crate) fn after_document_change(&mut self) {
        self.mark_changed();
        // The cages' faults and orientation re-read, and the check queued (`check_model.rs`).
        self.cages_changed();
        // **The curves are drawn from a copy, and this is where it is refreshed.** Undoing a
        // curve move reverted the document and left the drawn line exactly where it was, which
        // is indistinguishable from undo not working — and it was not only the move: every
        // command that touches a curve went the same way. One line, and it is here rather than
        // at each call site because "the document changed" is precisely when the drawing is
        // stale.
        self.rebuild_curve_display();
        let vertex_count = self.control().vertex_count();
        let face_count = self.control().face_count();
        self.selection.keep_valid_for(vertex_count, face_count);
        self.hover = Selection::None;
        self.rebuild_render_resources();
        // A cage block deleted, added or undone back can leave the active cage pointing at
        // nothing in the part.
        self.resync_active_cage();
    }

    /// Draws the splash centred in the viewport, scaled to fit without distorting it.
    pub(crate) fn draw_splash(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let texture = match &self.splash {
            Some(texture) => texture,
            None => {
                // Embedded rather than read from disk: the binary is the thing that gets run,
                // often from somewhere other than the source tree, and a splash that depends
                // on the working directory is a splash that is usually missing.
                const BYTES: &[u8] = include_bytes!("../../assets/splash-screen.png");
                let Ok(decoded) = image::load_from_memory(BYTES) else {
                    // A splash that will not decode is not worth a word to the user, but it
                    // should not be retried every frame either.
                    self.splash_shown = false;
                    return;
                };
                let rgba = decoded.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                self.splash.insert(ui.ctx().load_texture(
                    "splash",
                    image,
                    egui::TextureOptions::LINEAR,
                ))
            }
        };

        let native = texture.size_vec2();
        // Fits inside the viewport with a margin, and never enlarged past its own size — an
        // upscaled splash on a large monitor looks like a mistake.
        let scale = (rect.width() * 0.8 / native.x)
            .min(rect.height() * 0.8 / native.y)
            .min(1.0);
        let size = native * scale;
        let at = egui::Rect::from_center_size(rect.center(), size);
        egui::Image::new((texture.id(), size)).paint_at(ui, at);
    }

    /// Keeps the window title showing which document this is and whether it is saved.
    ///
    /// Sent only when the string changes: a viewport command every frame is a message to the
    /// windowing system sixty times a second to say nothing.
    pub(crate) fn sync_window_title(&mut self, ctx: &egui::Context) {
        let name = match &self.saving.file {
            Some((path, _)) => file_name(path),
            None => "untitled".to_string(),
        };
        let marker = if self.is_dirty() { "*" } else { "" };
        let title = format!("{marker}{name} — fieldform");
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    /// Intercepts the window's close button while there are unsaved changes.
    ///
    /// The close is cancelled first and re-issued after the prompt, because the prompt is
    /// modal and the answer arrives too late to stop a close already in flight.
    ///
    /// The answer has to be *remembered*, which is the whole reason for the flag. Choosing to
    /// discard leaves the document dirty — that is what discarding means — so the close this
    /// method then issues arrives back here one frame later, meets the same unsaved changes,
    /// and is cancelled and asked about again. The window flickered and would not close, and
    /// only saving escaped it, by making the second pass see a clean document.
    pub(crate) fn guard_close(&mut self, ctx: &egui::Context) {
        if !ctx.input(|i| i.viewport().close_requested()) {
            return;
        }
        if self.saving.closing || !self.is_dirty() {
            // Nothing to ask about, so this is the exit itself: the autosave goes with it.
            self.forget_autosave();
            return;
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        if self.may_discard("closing", app::document_io::AfterDiscard::Close) {
            self.saving.closing = true;
            self.forget_autosave();
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
