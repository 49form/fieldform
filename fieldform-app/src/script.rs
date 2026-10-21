// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A file of console lines run one after another, what the run counts, and what it exits with.

use super::*;

/// Whether a save reuses the path the document came from or asks for a new one.
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum SaveAs {
    SamePath,
    AskForPath,
    /// `saveas file="…"`: a path given, so no dialog -- which is what lets a script save.
    To(std::path::PathBuf),
}

/// A file of console lines, run through `run_command` one after another
/// (`benchmark-suite-instructions.md` §2 A7, and A8's `--run`).
///
/// **No second path.** A script line is a typed line: it goes through the same `run_command`,
/// is refused by the same key table, and reports into the same console, so the two cannot come
/// to mean different things. What the script adds is an order and a stop: `#` starts a comment,
/// blank lines are skipped, every line is announced with its number, and **the run stops at the
/// first refusal**, because a script that carries on past one is building on a block that is
/// not there. `keepgoing` runs to the end anyway, for triage. Each line is its own undo step --
/// the gesture that folds a quick second edit into the first is ended between lines, since two
/// lines written 400 ms apart by a file are not one gesture.
///
/// From the command line: `fieldform.exe --run <file> [--keep-going] [--exit]`. The lines run
/// after the first frame has drawn, so the GPU and the viewport exist for every one of them;
/// `--exit` closes the window afterwards without the unsaved-changes question, since a run never
/// writes over a document anyone opened.
pub(crate) struct Script {
    pub(crate) path: std::path::PathBuf,
    /// Line number and text of every line that is not blank or a comment.
    pub(crate) lines: Vec<(usize, String)>,
    pub(crate) next: usize,
    pub(crate) keep_going: bool,
    pub(crate) exit_after: bool,
    /// `--report <file>`: where the run's JSON goes.
    pub(crate) report: Option<std::path::PathBuf>,
    /// Frames drawn since it was armed. The first frame is for the GPU to come up.
    pub(crate) frames: u32,
    /// The scripts still to run after this one, in the order a folder gave them. Each starts
    /// from a new document, so one benchmark cannot leave a body standing in the next.
    pub(crate) then: Vec<std::path::PathBuf>,
    /// What every script in this run has reported, so a folder writes one report covering all
    /// of them rather than one that remembers only the last.
    pub(crate) done: Vec<serde_json::Value>,
}

/// A snapshot asked for and not yet written.
///
/// **It waits, because a still is not finished when the line is typed.** The viewport averages
/// eight jittered frames into the picture a person sees, one a frame, so a file written the
/// moment `snap` runs would carry the noisiest of them. The verb asks the view for a final
/// frame, this sits here, and the frame loop writes the file once `TracedView::is_settled`
/// says the picture is the finished one. A script **stops** while one is pending, so two
/// snapshots in a row cannot write each other's image.
pub(crate) struct PendingSnap {
    pub(crate) path: std::path::PathBuf,
    /// When the wait started, so a view that never settles says so instead of hanging a run.
    pub(crate) since: std::time::Instant,
    /// The longest the driver has been seen compiling a shader during this wait, which does not
    /// count against the limit — see [`SNAP_WAIT_MAX`].
    pub(crate) compiling: std::time::Duration,
}

/// What a run has counted so far: which line it stopped at, every `expect` it has reached, and
/// when it started. Kept on the `App` because a run spans frames whenever it takes a snapshot,
/// and a tally that started again each frame would report only the last stretch.
#[derive(Default)]
pub(crate) struct ScriptTally {
    pub(crate) stopped: Option<usize>,
    /// Line number and index into `expect_results`, for the report.
    pub(crate) expects: Vec<(usize, usize)>,
    pub(crate) started: Option<std::time::Instant>,
}

/// How long a snapshot waits for the still before giving up and saying so, not counting time the
/// driver spends compiling a shader.
///
/// **Counted in time rather than frames, and the compile does not count.** It was 240 frames,
/// which the smoke test's own seven-body scene exhausted: the driver took several seconds over
/// that shader, the frames went by while nothing could be drawn, and a picture that would have
/// arrived was reported as a view that never settled. What the limit is for is a run ending with
/// a message rather than a window nobody closes, so it measures the thing that would hang — a
/// view that is neither converging nor compiling.
pub(crate) const SNAP_WAIT_MAX: std::time::Duration = std::time::Duration::from_secs(20);

/// One `expect`'s result, as the console said it and as the report writes it.
pub(crate) struct ExpectRecord {
    pub(crate) what: String,
    pub(crate) wanted: String,
    pub(crate) got: String,
    pub(crate) tol: String,
    pub(crate) verdict: expect::Verdict,
    pub(crate) said: String,
}

/// What a `--run … --exit` asks the process to exit with: the number of failed benchmarks
/// (`benchmark-suite-instructions.md` A8), and a script is one benchmark -- so 1 if any expect
/// failed or was unresolved or the run stopped at a refusal, else 0. Negative until a run has
/// finished, and read by `main` once the window has closed.
pub(crate) static SCRIPT_EXIT_CODE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

impl Script {
    pub(crate) fn read(path: &std::path::Path, keep_going: bool, exit_after: bool) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read the script {}: {e}", path.display()))?;
        let lines = text
            .lines()
            .enumerate()
            .filter_map(|(i, line)| {
                let line = line.trim();
                (!line.is_empty() && !line.starts_with('#')).then(|| (i + 1, line.to_string()))
            })
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            lines,
            next: 0,
            keep_going,
            exit_after,
            report: None,
            frames: 0,
            then: Vec::new(),
            done: Vec::new(),
        })
    }

    /// Every `*/scene.ffc` under `folder`, in sorted order — the suite's own arrangement
    /// (`benchmark-suite-instructions.md` A8). A folder naming none is an error rather than an
    /// empty run, since a mistyped path is the likelier reason for finding nothing.
    fn scripts_in(folder: &std::path::Path) -> Result<Vec<std::path::PathBuf>, String> {
        let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(folder)
            .map_err(|e| format!("cannot read the folder {}: {e}", folder.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .map(|p| p.join("scene.ffc"))
            .filter(|p| p.is_file())
            .collect();
        found.sort();
        if found.is_empty() {
            return Err(format!(
                "no benchmark under {} — a folder run looks for `*/scene.ffc`",
                folder.display()
            ));
        }
        Ok(found)
    }

    /// One script, or a folder of them: `--run` and `run file=` both take either.
    pub(crate) fn read_one_or_folder(
        path: &std::path::Path,
        keep_going: bool,
        exit_after: bool,
    ) -> Result<Self, String> {
        if !path.is_dir() {
            return Self::read(path, keep_going, exit_after);
        }
        let mut scripts = Self::scripts_in(path)?;
        let first = scripts.remove(0);
        let mut script = Self::read(&first, keep_going, exit_after)?;
        script.then = scripts;
        Ok(script)
    }

    /// `--run <file> [--keep-going] [--report <file>] [--exit]` from the process's own
    /// arguments, or nothing.
    pub(crate) fn from_command_line() -> Option<Self> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let at = args.iter().position(|a| a == "--run")?;
        let Some(path) = args.get(at + 1) else {
            log::error!("--run wants the script's path after it");
            return None;
        };
        let keep_going = args.iter().any(|a| a == "--keep-going");
        let exit_after = args.iter().any(|a| a == "--exit");
        let report = args
            .iter()
            .position(|a| a == "--report")
            .and_then(|i| args.get(i + 1))
            .map(std::path::PathBuf::from);
        match Self::read_one_or_folder(std::path::Path::new(path), keep_going, exit_after) {
            Ok(script) => Some(Self { report, ..script }),
            Err(why) => {
                log::error!("{why}");
                // A run that cannot start is a failed run, so `--exit` still says so.
                if exit_after {
                    SCRIPT_EXIT_CODE.store(1, std::sync::atomic::Ordering::SeqCst);
                }
                None
            }
        }
    }
}
