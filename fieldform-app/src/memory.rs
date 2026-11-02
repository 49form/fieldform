// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! How much memory this machine can give a tessellation, asked of the operating system.
//!
//! **Asked directly rather than through a crate**, because the question is one call on each
//! platform and the answer is two numbers. The platform bindings already in the lock file are
//! there for other crates' sake, at versions those crates choose; depending on one directly
//! would tie this file to keeping in step with them, for twenty lines of code.

/// Physical memory free now and installed, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Physical {
    pub(crate) available: u64,
    pub(crate) total: u64,
}

/// What this machine says it has free, or `None` where it does not say.
///
/// **Free physical memory, not the commit limit.** A tessellation that spills into the page file
/// still finishes, but a machine paging a mesh of several gigabytes in and out stops answering
/// the mouse for the length of the run, which reads as the application having hung. What is free
/// in memory is the honest ceiling on what a run can use without that.
pub(crate) fn physical() -> Option<Physical> {
    imp::physical()
}

#[cfg(windows)]
mod imp {
    use super::Physical;

    /// The layout the system call fills, field for field.
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub(super) fn physical() -> Option<Physical> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: the buffer is a live, correctly sized structure whose length field is set, which
        // is all the call asks; it writes only within it.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(Physical { available: status.avail_phys, total: status.total_phys })
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::Physical;

    /// `MemAvailable`, which is the kernel's own estimate of what can be allocated without
    /// swapping — free memory plus the caches it would give back.
    pub(super) fn physical() -> Option<Physical> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let read = |key: &str| -> Option<u64> {
            let line = text.lines().find(|l| l.starts_with(key))?;
            let kb: u64 = line[key.len()..].trim().trim_end_matches("kB").trim().parse().ok()?;
            Some(kb * 1024)
        };
        Some(Physical { available: read("MemAvailable:")?, total: read("MemTotal:")? })
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod imp {
    pub(super) fn physical() -> Option<super::Physical> {
        None
    }
}

/// What a run may hold out of `available`, leaving room for everything else on the machine.
///
/// **A tenth of the machine or a gigabyte, whichever is more**, held back: the application's own
/// viewport, the other programs open beside it, and the estimate's own error all live in that
/// room, and a run that took the last byte free would leave none of them any.
pub(crate) fn usable(physical: Physical) -> u64 {
    let margin = (physical.total / 10).max(1 << 30);
    physical.available.saturating_sub(margin)
}

/// A number of bytes as a person reads it: "820 MB", "1.2 GB", "7.6 TB".
pub(crate) fn describe(bytes: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;
    let b = bytes as f64;
    if b >= TB {
        format!("{:.1} TB", b / TB)
    } else if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= 10.0 * MB {
        format!("{:.0} MB", b / MB)
    } else {
        format!("{:.1} MB", b / MB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The machine answers, and the answer is a machine: something free, less than is installed.
    #[test]
    fn the_machine_says_how_much_memory_it_has_free() {
        if !cfg!(any(windows, target_os = "linux")) {
            return;
        }
        let p = physical().expect("this platform says");
        assert!(p.total > 0 && p.available > 0, "{p:?}");
        assert!(p.available <= p.total, "{p:?}");
    }

    #[test]
    fn a_run_is_never_given_the_last_of_the_machine() {
        let gb = 1u64 << 30;
        assert_eq!(usable(Physical { available: 13 * gb, total: 32 * gb }), 13 * gb - 32 * gb / 10);
        assert_eq!(usable(Physical { available: 3 * gb, total: 4 * gb }), 2 * gb);
        assert_eq!(usable(Physical { available: gb / 2, total: 4 * gb }), 0);
    }

    #[test]
    fn a_size_reads_in_the_unit_a_person_would_say() {
        assert_eq!(describe(1_288_490_189), "1.2 GB");
        assert_eq!(describe(820 * 1024 * 1024), "820 MB");
        assert_eq!(describe(3 * 1024 * 1024 / 2), "1.5 MB");
        assert_eq!(describe(7770 * 1024 * 1024 * 1024), "7.6 TB");
    }
}
