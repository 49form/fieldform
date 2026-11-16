// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The console's lines, and the log file that keeps what each one said.

/// One line of the console: what was said, whether it went well, and how many times running.
pub(crate) struct ConsoleLine {
    pub(crate) ok: bool,
    pub(crate) text: String,
    /// How many times in a row this exact line was reported. One, unless something repeated.
    pub(crate) said: u32,
}

/// How many lines the console keeps. Old ones fall off the top.
pub(crate) const CONSOLE_LINES: usize = 4000;

/// How large the console's file may get before it is rewritten from what is still in memory.
pub(crate) const CONSOLE_FILE_BYTES: u64 = 1 << 20;

/// Starts this session's console log, in the temp directory beside the crash log.
///
/// Truncated rather than appended, so the file is one session and reading it cannot turn up a
/// message from yesterday that looks like it happened just now. `None` if it cannot be opened,
/// which is not worth complaining about — the console on screen is unaffected.
pub(crate) fn open_console_log() -> Option<std::path::PathBuf> {
    let path = std::env::temp_dir().join("fieldform-console.log");
    std::fs::write(&path, "").ok()?;
    Some(path)
}

/// A local to take the address of, which is all `install_crash_log` wants from it.
pub(crate) fn path_base() -> usize {
    0
}

/// Flattens anything that would break the one-entry-per-line shape of the log file.
///
/// Console text is written by this application rather than by a document, so this is tidiness
/// rather than a defence — but a file name comes from the filesystem, and an error message can
/// carry whatever a library put in it, so neither is guaranteed to be one line of printable text.
pub(crate) fn one_line(text: &str) -> String {
    text.chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}
