// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The lighting pass the application runs survives every path the viewport can take it down, and
//! reads back what it should.
//!
//! **Why this is a test and not only an example** (loop spec S5). The 9-against-12 crash — the
//! shaded pass's `fields` bind group holding nine entries against a layout of twelve — passed
//! every example, because the examples march plainly and only the application shades;
//! `overlay_smoke` ran the real pass, but nothing ran it before that build shipped. So the same
//! checks now run under `cargo test`. They are the example's own code, included rather than
//! copied, so the gate and the table the example prints are one thing:
//!
//! - every environment and material through the drag frame, the selection outline, the draft
//!   read both ways and an eight-frame still, with no non-finite value in the HDR buffer;
//! - an outline that changes the picture, so a run cannot pass with no outline drawn;
//! - a still that differs from its draft, so the still's pipeline is the one with occlusion;
//! - a still traced in bands that is the still traced whole;
//! - a still at one pixel in four (loop spec P5: built, and off as the viewport draws) that marches
//!   about a quarter of its pixels and finishes as the still marched at every pixel does, on large
//!   faces and on struts a pixel or two across alike, and in bands as whole.
//!
//! The pipeline is built through the same `ShadingResources::new` the application calls.
//!
//! **No adapter is a skip, said out loud, and nothing else is.** CI has no GPU, and a gate that
//! is red on every machine without one is a gate nobody can keep. But only
//! `RenderError::NoAdapter` skips: a device the adapter refuses, or a pipeline the driver
//! refuses, is exactly the class of fault this exists to catch.
//!
//! **It shares the card with whatever else is running**, as the vendored `sphere_wgpu` does, and
//! that one already flakes under GPU load. Under `--no-fail-fast` such a failure is one red line
//! rather than a truncated run, and the captured output opens with the adapter it ran on — so
//! run it again on a quiet card before reading it as a fault in the pass.

use std::io::Write;

use fieldform_gpu::RenderError;

// The example's `main` is the example's, and nothing here calls it. The example's own build
// still lints the file for anything else left unused, so this hides nothing that one would show.
#[allow(dead_code)]
#[path = "../examples/overlay_smoke.rs"]
mod overlay_smoke;

#[test]
fn the_shaded_pass_survives_every_lighting_path() {
    match overlay_smoke::run() {
        Ok(frames) => {
            println!("{frames} frames presented and read back; the device is still answering.")
        }
        Err(RenderError::NoAdapter) => {
            // Straight to the stream rather than through `eprintln!`, which the harness captures
            // and shows only for a failure: a skip that passes silently reads as a pass.
            let _ = writeln!(
                std::io::stderr(),
                "SKIPPED the_shaded_pass_survives_every_lighting_path: no GPU adapter on this \
                 machine, so the application's lighting pass was not exercised"
            );
        }
        Err(e) => panic!("the renderer would not start on an adapter that exists: {e}"),
    }
}
