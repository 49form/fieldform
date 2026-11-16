// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The document's file extensions, and the names shown for a path.

/// The extension a native document is written with.
pub(crate) const DOCUMENT_EXTENSION: &str = "fform";

/// Every extension a native document is *opened* from, newest first.
///
/// **Renamed from `.fieldform` on 2026-09-13, and the old name still opens.** Nothing inside the
/// container changed and nothing reads the extension to decide how to parse — `fieldform_io::open`
/// reads the archive itself — so an older document is not an older *format*, only an older name.
/// Dropping the old one from this list would make files people already have invisible in the
/// dialog while being perfectly readable, which is the worst of both.
pub(crate) const DOCUMENT_EXTENSIONS: &[&str] = &["fform", "fieldform"];

/// What to call a benchmark in the console and in the report: its folder's name where the file is
/// the suite's own `scene.ffc`, and the file's name otherwise.
///
/// A folder of benchmarks is `*/scene.ffc`, so every one of them is called `scene.ffc` and a
/// report naming them that identifies none. The folder is what the suite calls each one.
pub(crate) fn benchmark_name(path: &std::path::Path) -> String {
    if path.file_name().is_some_and(|n| n == "scene.ffc") {
        if let Some(folder) = path.parent().and_then(|p| p.file_name()) {
            return folder.to_string_lossy().into_owned();
        }
    }
    file_name(path)
}

/// A path's file name, for status lines. The full path is too long for the bar and the name
/// is what the user recognises.
pub(crate) fn file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
