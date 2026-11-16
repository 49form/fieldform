// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A guard against a walk that recurses without end, which would otherwise end the process without a word.

/// Roughly where the main thread's stack starts, recorded at startup.
///
/// Zero until `install_crash_log` sets it, which is the first thing `main` does.
pub(crate) static STACK_BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How far the stack may grow before a walk is treated as a runaway rather than as deep work.
///
/// Half a megabyte. Nothing here recurses honestly beyond a few hundred frames, and the main
/// thread has megabytes — so this trips long before the guard page does, which is the whole
/// point: past the guard page there is no stack left to write a report with.
pub(crate) const STACK_RUNAWAY_BYTES: usize = 512 * 1024;

/// Aborts with a backtrace when the stack has run away, rather than letting it overflow.
///
/// **Because a stack overflow leaves nothing behind.** It aborts without unwinding, so the panic
/// hook never runs and `fieldform-crash.log` stays empty — which is exactly what two crash
/// reports looked like: the window vanished, the log was empty, and the only evidence was the
/// one line Windows prints to a stderr nobody sees. This turns the next one into a named
/// backtrace in the file the hook already writes.
///
/// Called from the cheap functions nearly every walk goes through, so whichever recursion is
/// running away has this on its stack and names itself.
pub(crate) fn guard_stack(what: &str) {
    let here = &what as *const _ as usize;
    let base = STACK_BASE.load(std::sync::atomic::Ordering::Relaxed);
    // Stacks grow down on every platform this runs on, and a base of zero means startup has not
    // recorded one yet -- in both cases there is nothing to compare against.
    if base == 0 || here >= base || base - here < STACK_RUNAWAY_BYTES {
        return;
    }
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::temp_dir().join("fieldform-crash.log"))
    {
        let _ = writeln!(file, "\n=== fieldform runaway walk ===");
        let _ = writeln!(file, "in {what}, {} bytes of stack deep", base - here);
        let _ = writeln!(file, "{}", std::backtrace::Backtrace::force_capture());
        let _ = file.flush();
    }
    std::process::abort();
}
