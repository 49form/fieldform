// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Whether this system can show a file picker, asked before one is opened.
//!
//! On Linux the picker is not part of the application or of the toolkit: it is asked of the
//! desktop's file-chooser portal, and failing that of an external dialog program. When neither is
//! there the dialog library logs the failure and answers exactly as a cancel does, so the click
//! does nothing and says nothing. That is the one outcome a menu item may not have, so the
//! question is asked first and a missing picker is reported by name, with the console line that
//! does the same thing without one.
//!
//! Windows and macOS always have a picker, so there the answer is always yes.

/// Why no file picker can be shown, or `None` when one can.
pub(crate) fn unavailable() -> Option<&'static str> {
    imp::unavailable()
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub(super) fn unavailable() -> Option<&'static str> {
        None
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Remembered once true and never once false: a picker does not go away during a session,
    /// but one can be installed during it, and the person who just did so should not have to
    /// restart to be believed.
    static FOUND: AtomicBool = AtomicBool::new(false);

    pub(super) fn unavailable() -> Option<&'static str> {
        if FOUND.load(Ordering::Relaxed) {
            return None;
        }
        if portal_has_file_chooser() || on_path("zenity") {
            FOUND.store(true, Ordering::Relaxed);
            return None;
        }
        Some(
            "this desktop has no file chooser (install xdg-desktop-portal-gtk or zenity)",
        )
    }

    /// Whether the session bus has a portal that answers for the file chooser. Reading the
    /// interface's `version` property is the cheapest question that only a real implementation
    /// answers: the portal front end is often running with no back end behind it.
    fn portal_has_file_chooser() -> bool {
        let Ok(bus) = zbus::blocking::Connection::session() else {
            return false;
        };
        bus.call_method(
            Some("org.freedesktop.portal.Desktop"),
            "/org/freedesktop/portal/desktop",
            Some("org.freedesktop.DBus.Properties"),
            "Get",
            &("org.freedesktop.portal.FileChooser", "version"),
        )
        .is_ok()
    }

    /// The fallback the dialog library runs when the portal fails.
    fn on_path(program: &str) -> bool {
        std::env::var_os("PATH").is_some_and(|path| {
            std::env::split_paths(&path).any(|dir| dir.join(program).is_file())
        })
    }
}

/// Which question a picker asks.
#[derive(Clone, Copy)]
pub(crate) enum Ask {
    /// An existing file to read.
    Open,
    /// A file to write, which may not exist yet.
    Save,
    /// A folder.
    Folder,
}

/// What runs with the path a picker answered. A cancel runs nothing, as it always did.
type Then = Box<dyn FnOnce(&mut crate::App, std::path::PathBuf)>;

/// A picker that is open, and what is waiting on its answer.
pub(crate) struct Waiting {
    answer: std::sync::mpsc::Receiver<Option<std::path::PathBuf>>,
    then: Then,
}

impl crate::App {
    /// Opens a file picker and runs `then` with its answer on a later frame.
    ///
    /// **A picker never blocks the frame loop.** It used to be called where the answer was
    /// wanted, which holds the thread that owns the window until the picker closes. Where the
    /// picker belongs to the application's own process that is harmless, because the system
    /// keeps the window's messages moving underneath it. On a Wayland desktop the picker is
    /// another process, the window stops answering the compositor for as long as it is open,
    /// and the compositor reports the application as not responding: a Save As that took forty
    /// seconds to answer was met as a hang. So the picker runs on a thread of its own and the
    /// window keeps drawing, with [`Self::show_file_picker_wait`] standing in front of the
    /// document until it answers.
    ///
    /// One at a time: a second picker asked for while one is open is refused, since two
    /// answers arriving in either order is nothing any caller means.
    pub(crate) fn ask_for_path(
        &mut self,
        dialog: rfd::FileDialog,
        ask: Ask,
        then: impl FnOnce(&mut crate::App, std::path::PathBuf) + 'static,
    ) {
        if self.saving.file_picker.is_some() {
            self.warn("A file dialog is already open \u{2014} answer or close that one first");
            return;
        }
        let (tx, answer) = std::sync::mpsc::channel();
        let ctx = self.egui_ctx.clone();
        let spawned = std::thread::Builder::new().name("file picker".to_string()).spawn(move || {
            let path = match ask {
                Ask::Open => dialog.pick_file(),
                Ask::Save => dialog.save_file(),
                Ask::Folder => dialog.pick_folder(),
            };
            let _ = tx.send(path);
            // The window may be idle and drawing nothing; wake it to collect the answer.
            ctx.request_repaint();
        });
        match spawned {
            Ok(_) => self.saving.file_picker = Some(Waiting { answer, then: Box::new(then) }),
            Err(e) => self.warn(format!("The file dialog could not be opened: {e}")),
        }
    }

    /// Collects a picker's answer once it has one, and until then says what is being waited for.
    ///
    /// The wait is modal, as the picker itself was: the document cannot be edited underneath a
    /// question about where to write it. *Stop waiting* is the way out when the picker cannot be
    /// found, which a desktop that opens it behind the window makes possible; the picker is
    /// another process and cannot be closed from here, so its answer is then ignored.
    pub(crate) fn show_file_picker_wait(&mut self, ctx: &eframe::egui::Context) {
        use eframe::egui;
        let Some(waiting) = &self.saving.file_picker else {
            return;
        };
        match waiting.answer.try_recv() {
            Ok(path) => {
                let waiting = self.saving.file_picker.take().expect("checked above");
                if let Some(path) = path {
                    (waiting.then)(self, path);
                }
                return;
            }
            // The picker's thread ended without answering, which only a panic inside the
            // dialog library does. Treated as a cancel rather than waited on for ever.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.saving.file_picker = None;
                self.warn("The file dialog closed without an answer");
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
        let mut stop = false;
        egui::Modal::new(egui::Id::new("file-picker-wait")).show(ctx, |ui| {
            ui.set_width(340.0);
            ui.heading("Waiting for the file dialog");
            ui.add_space(6.0);
            ui.label("Choose a file in the dialog that opened. It may be behind this window.");
            ui.add_space(12.0);
            stop = ui.button("Stop waiting").clicked();
        });
        // Keeps drawing while the picker is open. An idle window draws nothing, and a discrete
        // card that has gone unused for a few seconds powers down; the first frames after the
        // answer then wait a second each for a display that is still waking.
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
        if stop {
            self.saving.file_picker = None;
            self.say("Stopped waiting for the file dialog; its answer will be ignored.");
        }
    }
}
