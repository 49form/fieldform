// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Autosave, the files a crash leaves to be recovered, and the crash log.

use crate::*;

/// How often the document is written to [`autosave_path`] while it has unsaved changes.
///
/// A minute is the most work a sudden death may cost. **Measured before it was chosen:** the
/// whole document of the heaviest scene to hand — a radial pattern of 48 gyroid-filled shells —
/// writes through the archive in **11 ms**, which is why this runs on the interface thread and
/// not a worker. A document is a graph, so its size follows the number of blocks rather than
/// how heavy they are to draw; the autosave logs its own cost every time precisely so the day
/// that stops being true is visible.
pub(crate) const AUTOSAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(60);

/// Whether the autosave is owed, given the document's revision, the one last written, whether
/// there is anything unsaved, and how long since the last write.
///
/// All four conditions, and each is there for its own reason: nothing unsaved means the
/// document's own file is already the record; an unchanged revision means the autosave already
/// *is* this document, which matters because `is_dirty` stays true after an autosave (an
/// autosave is not a save); and the interval is what keeps a heavy document's write off every
/// frame.
pub(crate) fn autosave_due(revision: u64, autosaved: u64, dirty: bool, since: std::time::Duration) -> bool {
    dirty && revision != autosaved && since >= AUTOSAVE_EVERY
}

/// How long ago, in the roundest words that are still true: a recovery offer wants "about an
/// hour ago", not a duration to parse.
pub(crate) fn how_long_ago(d: std::time::Duration) -> String {
    let minutes = d.as_secs() / 60;
    match minutes {
        0 => "less than a minute ago".to_string(),
        1 => "a minute ago".to_string(),
        2..=59 => format!("{minutes} minutes ago"),
        60..=119 => "about an hour ago".to_string(),
        _ => {
            let hours = minutes / 60;
            if hours < 24 {
                format!("about {hours} hours ago")
            } else {
                let days = hours / 24;
                format!("{days} day{} ago", if days == 1 { "" } else { "s" })
            }
        }
    }
}

/// The folder work is kept in for a later session to find: autosaves, a document kept when the
/// GPU was lost, an agent host's unsaved work.
///
/// **Somewhere a restart does not empty.** These files exist for the session that ended badly,
/// and a power cut is one of the ways. Where the temp directory is a disk, it serves. On Linux it
/// is commonly held in memory and gone at the next boot, taking with it exactly the file the
/// restart was going to offer, so there the folder is the account's own state directory, which
/// also keeps one account's unsaved documents out of every other's reach. The logs stay in the
/// temp directory: they describe a session and are not needed after the machine restarts.
pub(crate) fn recovery_dir() -> std::path::PathBuf {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::DirBuilderExt;
        let state = std::env::var_os("XDG_STATE_HOME")
            .map(std::path::PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/state")));
        if let Some(state) = state {
            let dir = state.join("fieldform");
            // Private to the account, since what goes here is somebody's unsaved work.
            if dir.is_dir() || std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir).is_ok() {
                return dir;
            }
        }
    }
    std::env::temp_dir()
}

/// Where **this process's** autosave is written: beside the console and crash logs, not beside
/// the document, and named by the process so it is never another session's file.
///
/// **One name for every session was how a crashed session's work was deleted, 2026-10-15.** The
/// owner's test instance was killed, leaving its autosave; the next instance — a scripted run —
/// exited cleanly and removed "the" autosave, which was not its own. One shared name also let a
/// second live instance overwrite the first one's. So each process writes and removes only the file
/// named by its own id, and [`other_autosaves`] is how the others are found: offered, never removed.
pub(crate) fn autosave_path() -> std::path::PathBuf {
    recovery_dir().join(format!(
        "fieldform-autosave-{}.{DOCUMENT_EXTENSION}",
        std::process::id()
    ))
}

/// Which files in the temp directory are autosaves of sessions other than `own`, newest first.
///
/// Pure over a listing so the rule can be held to a property. Matches this build's per-process
/// names and the single shared name earlier builds wrote, so the crash of an older build is still
/// offered. **Never a reason to delete anything:** a file listed here may belong to a session that is
/// still running, and only that session knows when it is finished with it.
pub(crate) fn other_autosaves(
    listing: &[(String, std::time::SystemTime)],
    own: &str,
) -> Vec<(String, std::time::SystemTime)> {
    let ext = format!(".{DOCUMENT_EXTENSION}");
    let mut out: Vec<(String, std::time::SystemTime)> = listing
        .iter()
        .filter(|(name, _)| {
            name != own
                && name.starts_with("fieldform-autosave")
                && name.ends_with(&ext)
                && {
                    // Exactly the shared name, or the shared name with a process id: nothing a
                    // person happened to call "fieldform-autosave backup" or similar.
                    let middle = &name["fieldform-autosave".len()..name.len() - ext.len()];
                    middle.is_empty()
                        || (middle.len() > 1
                            && middle.starts_with('-')
                            && middle[1..].chars().all(|c| c.is_ascii_digit()))
                }
        })
        .cloned()
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

/// Where the document is kept when the GPU is lost: `<name>.recovered.fform` beside the file it
/// was opened from or last saved to, or beside the logs for one never saved (loop spec S3).
///
/// **Beside the document, unlike the autosave.** The autosave is one file for whatever was being
/// edited, overwritten every minute; this is the newest state of one particular document, and the
/// place to look for it is next to that document. **Never the document's own file**, for the
/// autosave's reason: a dying GPU must not overwrite the last state somebody deliberately saved.
/// In the current extension whatever the file was opened as, since that is what this build writes.
pub(crate) fn recovered_path(file: Option<&std::path::Path>) -> std::path::PathBuf {
    match file {
        Some(path) => {
            let name = path
                .file_stem()
                .map_or_else(|| "untitled".into(), |stem| stem.to_string_lossy());
            path.with_file_name(format!("{name}.recovered.{DOCUMENT_EXTENSION}"))
        }
        // Prefixed like the logs and the autosave, so the temp directory lists them together.
        None => recovery_dir()
            .join(format!("fieldform-untitled.recovered.{DOCUMENT_EXTENSION}")),
    }
}

/// Writes `doc` to [`recovered_path`] and answers where it went.
///
/// **Through the archive the document was opened from**, for `save_document`'s reason: a fresh
/// archive drops the `ext/` and `sources/` entries the file carried, and a kept document somebody
/// later saves over their own would lose them. A clone of it, so the application's copy still
/// matches the file on disk.
///
/// **The temp directory, under the same name, when the document's folder refuses.** A share that
/// dropped or a drive pulled out is no reason to lose the work, and the logs' folder is the one
/// place this application always writes.
pub(crate) fn write_recovered(
    doc: &Document,
    file: Option<&(std::path::PathBuf, fieldform_io::Archive)>,
) -> Result<std::path::PathBuf, fieldform_io::FormatError> {
    let archive = match file {
        Some((_, opened)) => {
            let mut archive = opened.clone();
            fieldform_io::update_archive(&mut archive, doc)?;
            archive
        }
        None => fieldform_io::to_archive(doc)?,
    };
    let beside = recovered_path(file.map(|(path, _)| path.as_path()));
    match archive.write(&beside) {
        Ok(()) => Ok(beside),
        Err(refused) => {
            let instead = recovery_dir().join(beside.file_name().unwrap_or_default());
            if instead == beside {
                return Err(refused);
            }
            archive.write(&instead).map(|()| instead)
        }
    }
}

/// Appends a headed note to `fieldform-crash.log`, the file a person is sent to when the window
/// has gone. Best effort: a note that cannot be written must not become a second failure.
pub(crate) fn note_in_crash_log(heading: &str, said: &str) {
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::temp_dir().join("fieldform-crash.log"))
    {
        let _ = writeln!(file, "\n=== {heading} ===\n{said}");
    }
}

/// Writes any panic, with a backtrace, to `fieldform-crash.log` in the temp directory.
///
/// A windowed application's stderr goes nowhere the user can see — launched from Explorer
/// there is no console at all — so without this a crash is just the window vanishing, and the
/// one piece of information needed to fix it is the piece that gets lost. The default hook
/// still runs afterwards, so a terminal launch keeps printing as before.
pub(crate) fn install_crash_log() -> std::path::PathBuf {
    // Whatever the stack pointer is here, which is near enough the top of it: this runs from
    // `main` before anything else.
    let here = &path_base() as *const _ as usize;
    STACK_BASE.store(here, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join("fieldform-crash.log");
    let target = path.clone();
    let previous = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target)
        {
            let _ = writeln!(file, "\n=== fieldform panic ===");
            let _ = writeln!(file, "{info}");
            // force_capture rather than capture: RUST_BACKTRACE is almost never set for a
            // double-clicked application, and a backtrace-less crash report is close to
            // useless.
            let _ = writeln!(file, "{}", std::backtrace::Backtrace::force_capture());
        }
        previous(info);
        // Then die, rather than unwind. A panic raised inside a Windows message callback -- which
        // is where every frame of this application runs -- does not unwind back out of it: the
        // window stays on screen, stops answering, and holds the GPU while it does. The log
        // above is what a crash is for; a hung window on top of it is not. Seen on 2026-09-06,
        // when an egui allocation failed and the machine had to be restarted to be rid of it.
        std::process::abort();
    }));

    path
}
