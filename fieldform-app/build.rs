// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Gives `fieldform.exe` its icon in Explorer, on the taskbar's pinned shortcut and in Alt-Tab.
//!
//! The *window's* icon is a different thing, set at run time from `assets/window-icon.png`
//! (`window_icon` in `main.rs`) and visible only while the program is open. A file's icon has
//! to be a resource inside the executable, because Explorer reads it without running anything.
//! Both come from the one emblem: `assets/window-icon.svg` is the source, and `fieldform.ico`
//! holds it at 16, 32, 48 and 256 pixels, which are the sizes Explorer asks for.
//!
//! Windows only, and only the application -- `fieldform-import` is a worker process that
//! nobody opens, so it does not carry one.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/fieldform.ico");

    // The *target*, not the machine the build runs on: a build script is compiled for the host,
    // so `cfg!(windows)` here would answer for the wrong one when cross-compiling.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let mut resource = winresource::WindowsResource::new();
    resource
        .set_icon("assets/fieldform.ico")
        .set("ProductName", "fieldform")
        .set("FileDescription", "fieldform");
    // A missing resource compiler leaves a working program with the default icon, which is
    // worse than the icon and better than a build that does not finish: say so, and carry on.
    if let Err(e) = resource.compile() {
        println!("cargo:warning=fieldform.exe will have no icon of its own: {e}");
    }
}
