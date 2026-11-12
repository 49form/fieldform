// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! An agent host keeps its unsaved document where the next host can hand it back.
//!
//! **Why the application's own autosave is not enough here.** It runs on a minute's timer and
//! lives in the temp folder, which the path guard keeps every agent out of; and the host an agent
//! drives can be ended under it -- the relay relaunches one, a person clicks the window's agent
//! button -- and comes back as a new process with an empty document and a new folder. An agent
//! that had built a part in forty calls was left with nothing it could reach.
//!
//! So after every call it keeps, the host writes the document to `unsaved-<pid>.fform` in its own
//! first root, and a host starting up copies the newest such file another host left into *its*
//! first root as `recovered-<pid>.fform`, reports it in `session_info`, and says so. Copied, never
//! moved or opened: the other host may still be running, and reading a file back over what was
//! asked for is a surprise, exactly as the application's autosave offer says.

use super::*;

/// The file name a host with process id `pid` keeps its unsaved document under.
fn checkpoint_name(pid: u32) -> String {
    format!("unsaved-{pid}.{DOCUMENT_EXTENSION}")
}

/// The process id in a checkpoint's file name, if `name` is one.
pub(crate) fn checkpoint_pid(name: &str) -> Option<u32> {
    let middle = name.strip_prefix("unsaved-")?.strip_suffix(&format!(".{DOCUMENT_EXTENSION}"))?;
    (!middle.is_empty() && middle.chars().all(|c| c.is_ascii_digit())).then(|| middle.parse().ok()).flatten()
}

/// `path` without the `\\?\` a canonical Windows path carries, which is what makes it a line a
/// person or an agent can type: `//?/C:/…` with its slashes turned is not a path `open` reads.
pub(crate) fn plain(path: &std::path::Path) -> std::path::PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => std::path::PathBuf::from(rest),
        _ => path.to_path_buf(),
    }
}

/// What a starting host found another host had left, for `session_info`.
pub(crate) struct Recovered {
    /// The copy, in this host's first root: what `open file=` reads.
    pub(crate) file: std::path::PathBuf,
    /// Where the other host had kept it.
    pub(crate) from: std::path::PathBuf,
    /// How long before this host started it was written.
    pub(crate) age: std::time::Duration,
}

/// The newest checkpoint among `listing` that is not `own`'s, as `(path, modified)`.
///
/// Pure over a listing, so the rule is held by a test without a temp folder.
pub(crate) fn newest_left_behind(
    listing: &[(std::path::PathBuf, std::time::SystemTime)],
    own: u32,
) -> Option<(std::path::PathBuf, std::time::SystemTime)> {
    listing
        .iter()
        .filter(|(path, _)| {
            path.file_name()
                .and_then(|n| n.to_str())
                .and_then(checkpoint_pid)
                .is_some_and(|pid| pid != own)
        })
        .max_by_key(|(_, at)| *at)
        .cloned()
}

impl App {
    /// Where this host keeps its unsaved document, or `None` with no root to keep it in.
    fn checkpoint_path(&self) -> Option<std::path::PathBuf> {
        Some(self.agent.roots.first()?.join(checkpoint_name(std::process::id())))
    }

    /// Writes the document to this host's checkpoint after a call that kept something, or removes
    /// the checkpoint once nothing is unsaved. Only while an agent host is running.
    pub(crate) fn keep_agent_checkpoint(&mut self) {
        if self.agent_host.is_none() || cfg!(test) {
            return;
        }
        let Some(path) = self.checkpoint_path() else { return };
        if !self.is_dirty() {
            let _ = std::fs::remove_file(&path);
            self.agent.checkpoint_revision = None;
            return;
        }
        if self.agent.checkpoint_revision == Some(self.revision) {
            return;
        }
        // A variable a settle made is named before the file is written, as every save does.
        self.drain_ownership_changes(false);
        let started = std::time::Instant::now();
        match fieldform_io::to_archive(&self.doc).and_then(|archive| archive.write(&path)) {
            Ok(()) => {
                self.agent.checkpoint_revision = Some(self.revision);
                log::info!("agent checkpoint {} in {:.0} ms", path.display(), started.elapsed().as_secs_f64() * 1000.0);
            }
            Err(e) => log::warn!("agent checkpoint to {} failed: {e}", path.display()),
        }
    }

    /// Removes this host's checkpoint, as a clean exit removes the autosave.
    pub(crate) fn forget_agent_checkpoint(&mut self) {
        if let Some(path) = self.checkpoint_path() {
            let _ = std::fs::remove_file(path);
        }
    }

    /// At start: copies the newest document another agent host left unsaved into this host's
    /// first root, and says where. Looks in every root and in the temp folder's agent folders,
    /// since a host started without `--root` had a folder of its own named by its process.
    pub(crate) fn recover_agent_checkpoint(&mut self) {
        let Some(root) = self.agent.roots.first().cloned() else { return };
        let mut folders: Vec<std::path::PathBuf> = self.agent.roots.clone();
        if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("fieldform-agent-") && entry.path().is_dir() {
                    folders.push(entry.path());
                }
            }
        }
        let mut listing: Vec<(std::path::PathBuf, std::time::SystemTime)> = Vec::new();
        for folder in &folders {
            let Ok(entries) = std::fs::read_dir(folder) else { continue };
            for entry in entries.flatten() {
                if let Ok(at) = entry.metadata().and_then(|m| m.modified()) {
                    listing.push((entry.path(), at));
                }
            }
        }
        let Some((from, at)) = newest_left_behind(&listing, std::process::id()) else { return };
        let pid = from.file_name().and_then(|n| n.to_str()).and_then(checkpoint_pid).unwrap_or(0);
        let file = root.join(format!("recovered-{pid}.{DOCUMENT_EXTENSION}"));
        if std::fs::copy(&from, &file).is_err() {
            log::warn!("could not copy {} to {}", from.display(), file.display());
            return;
        }
        let age = at.elapsed().unwrap_or_default();
        let file = plain(&file);
        self.say(format!(
            "Another agent session left unsaved work {}; it is copied to {} — `open file=\"{}\"` reads it",
            how_long_ago(age),
            file.display(),
            file.to_string_lossy().replace('\\', "/"),
        ));
        self.agent.recovered = Some(Recovered { file, from, age });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn a_recovered_path_is_one_a_line_can_name() {
        assert_eq!(plain(std::path::Path::new(r"\\?\C:\Temp\x.fform")), std::path::PathBuf::from(r"C:\Temp\x.fform"));
        assert_eq!(plain(std::path::Path::new(r"C:\Temp\x.fform")), std::path::PathBuf::from(r"C:\Temp\x.fform"));
        // A UNC share keeps its prefix: stripped, it would name a local path that is not it.
        assert_eq!(plain(std::path::Path::new(r"\\?\UNC\host\s\x")), std::path::PathBuf::from(r"\\?\UNC\host\s\x"));
    }

    #[test]
    fn only_a_checkpoint_named_by_a_process_is_one() {
        assert_eq!(checkpoint_pid("unsaved-1234.fform"), Some(1234));
        for not in ["unsaved-.fform", "unsaved-12a.fform", "unsaved-12.fieldform", "recovered-12.fform", "unsaved-12.fform.bak"] {
            assert_eq!(checkpoint_pid(not), None, "{not}");
        }
    }

    #[test]
    fn the_newest_checkpoint_another_host_left_is_the_one_handed_back() {
        let t = |s: u64| SystemTime::UNIX_EPOCH + Duration::from_secs(s);
        let p = |s: &str| std::path::PathBuf::from(s);
        let listing = vec![
            (p("a/unsaved-10.fform"), t(100)),
            (p("b/unsaved-11.fform"), t(300)),
            // Its own, newest of all, is never handed back to itself.
            (p("c/unsaved-12.fform"), t(900)),
            (p("c/snapshot-12-1.png"), t(950)),
            (p("d/recovered-9.fform"), t(990)),
        ];
        assert_eq!(newest_left_behind(&listing, 12).map(|(path, _)| path), Some(p("b/unsaved-11.fform")));
        assert_eq!(newest_left_behind(&listing[2..], 12), None);
    }
}
