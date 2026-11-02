// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The one guard every file a console line names goes through while an agent is driving
//! (`mcp-server-spec.md` §9.2).
//!
//! **One function, asked of the arguments, not one check per verb.** It is called where a line's
//! arguments are read, so a verb added tomorrow that takes `file=` is covered without anyone
//! remembering. A path is resolved against the first root when relative, normalised without
//! touching the disk, then canonicalised through its nearest existing ancestor so a link that
//! points out of a root is followed before it is compared. Drive-relative and UNC forms are
//! refused outright: they name places no root can vouch for.

use std::path::{Component, Path, PathBuf};

/// Why a path was refused, with the spec's code.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PathRefusal {
    Outside,
    WouldOverwrite(PathBuf),
}

/// The keys that name a file. Asked by key name so no verb has to opt in.
pub(crate) const FILE_KEYS: &[&str] = &["file", "path", "out"];

/// Resolves `given` to an absolute path inside one of `roots`, or says why not.
pub(crate) fn resolve(given: &str, roots: &[PathBuf], writes: bool, allow_overwrite: bool) -> Result<PathBuf, PathRefusal> {
    if roots.is_empty() || given.trim().is_empty() {
        return Err(PathRefusal::Outside);
    }
    let mut text = given.trim();
    // The verbatim form of a drive path (a `?` after two backslashes) is what canonicalising a root gives on
    // Windows, so a client handed a root back is allowed to say it that way.
    if let Some(rest) = text.strip_prefix("\\\\?\\")
        && rest.as_bytes().get(1) == Some(&b':')
    {
        text = rest;
    }
    // UNC (host-and-share forms with doubled slashes or backslashes, and verbatim prefixes) and drive-relative (`C:foo`) forms.
    if text.starts_with("\\\\") || text.starts_with("//") {
        return Err(PathRefusal::Outside);
    }
    let bytes = text.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && !matches!(bytes.get(2), Some(b'/') | Some(b'\\')) {
        return Err(PathRefusal::Outside);
    }
    let path = Path::new(text);
    let joined = if path.is_absolute() { path.to_path_buf() } else { roots[0].join(path) };
    let mut normal = PathBuf::new();
    for part in joined.components() {
        match part {
            Component::ParentDir => {
                if !normal.pop() {
                    return Err(PathRefusal::Outside);
                }
            }
            Component::CurDir => {}
            other => normal.push(other.as_os_str()),
        }
    }
    let real = canonical_through_ancestor(&normal).ok_or(PathRefusal::Outside)?;
    let inside = roots
        .iter()
        .filter_map(|r| r.canonicalize().ok())
        .any(|root| real.starts_with(&root));
    if !inside {
        return Err(PathRefusal::Outside);
    }
    if writes && !allow_overwrite && real.exists() {
        return Err(PathRefusal::WouldOverwrite(real));
    }
    Ok(real)
}

/// The path with its nearest existing ancestor canonicalised (links followed) and the rest
/// appended, so a file that does not exist yet can still be judged.
fn canonical_through_ancestor(path: &Path) -> Option<PathBuf> {
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut at = path.to_path_buf();
    loop {
        if let Ok(real) = at.canonicalize() {
            let mut out = real;
            out.extend(tail.iter().rev());
            return Some(out);
        }
        tail.push(at.file_name()?.to_os_string());
        at = at.parent()?.to_path_buf();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> (tempdir::Dir, PathBuf) {
        let dir = tempdir::Dir::new();
        let root = dir.path().join("work");
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        (dir, root)
    }

    /// A scratch folder removed on drop, without a dependency for it.
    mod tempdir {
        use std::path::{Path, PathBuf};
        pub struct Dir(PathBuf);
        impl Dir {
            pub fn new() -> Self {
                let p = std::env::temp_dir().join(format!("fieldform-paths-{}-{}", std::process::id(), super::unique()));
                std::fs::create_dir_all(&p).unwrap();
                Dir(p)
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    fn unique() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn no_path_escapes_a_root() {
        let (dir, root) = root();
        let roots = vec![root.clone()];
        let outside = dir.path().join("secret.txt");
        std::fs::write(&outside, "x").unwrap();
        for bad in [
            "../secret.txt",
            "a/../../secret.txt",
            outside.to_str().unwrap(),
            "C:secret.txt",
            "\\\\host\\share\\x",
            "//host/share/x",
            "",
        ] {
            assert_eq!(resolve(bad, &roots, false, true), Err(PathRefusal::Outside), "{bad:?}");
        }
        // Inside, relative or absolute, existing or not.
        assert!(resolve("part.fform", &roots, true, false).unwrap().starts_with(&root));
        assert!(resolve("sub/dir/part.fform", &roots, true, false).unwrap().starts_with(&root));
        assert!(resolve(root.join("x.fform").to_str().unwrap(), &roots, true, false).is_ok());
        assert!(resolve("a/../x.fform", &roots, true, false).is_ok());
    }

    #[test]
    fn a_link_out_of_a_root_is_followed_and_refused() {
        let (dir, root) = root();
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        let link = root.join("door");
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&elsewhere, &link).is_ok();
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_dir(&elsewhere, &link).is_ok();
        if !made {
            // Creating links needs a privilege some machines withhold; the test cannot say anything then.
            return;
        }
        assert_eq!(resolve("door/out.fform", &[root], true, true), Err(PathRefusal::Outside));
    }

    #[test]
    fn an_existing_file_is_not_overwritten_unasked() {
        let (_dir, root) = root();
        let roots = vec![root.clone()];
        std::fs::write(root.join("part.fform"), "x").unwrap();
        assert!(matches!(resolve("part.fform", &roots, true, false), Err(PathRefusal::WouldOverwrite(_))));
        assert!(resolve("part.fform", &roots, true, true).is_ok());
        assert!(resolve("part.fform", &roots, false, false).is_ok(), "reading is not overwriting");
    }
}
