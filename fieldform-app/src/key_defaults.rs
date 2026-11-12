// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What each key's field holds on the block its verb makes when no keys are given
//! (`mcp-server-spec.md` §5.3).
//!
//! **Read from the application, then held to it.** `app_tests::derived_key_defaults` runs every
//! create verb bare on a scratch document and reads the field each key covers; [`DEFAULTS`] is that
//! written down, as compact JSON, so the catalogue can answer without an app. The test fails when a
//! default changes, and prints the table to paste:
//! `FIELDFORM_PRINT_KEY_DEFAULTS=1 cargo test -p fieldform-app key_defaults -- --nocapture`.
//! A verb that is refused bare (it wants a profile or a target) has no row: unknown, not guessed.

/// `(operation, key, default as JSON)`.
pub const DEFAULTS: &[(&str, &str, &str)] = &[
    ("create.capsule", "end_radius", "25.0"),
    ("create.capsule", "start_radius", "25.0"),
    ("create.coil", "diameter", "80.0"),
    ("create.coil", "hand", "\"right\""),
    ("create.coil", "section", "\"circle\""),
    ("create.coil", "wire", "8.0"),
    ("create.cone", "base_radius", "50.0"),
    ("create.cone", "top_radius", "0.0"),
    ("create.cube", "centre", "[0.0,0.0,0.0]"),
    ("create.cube", "size", "[100.0,100.0,100.0]"),
    ("create.cylinder", "radius", "50.0"),
    ("create.pipe", "radius", "4.3"),
    ("create.plane", "flipped", "false"),
    ("create.plane", "origin", "[0.0,0.0,0.0]"),
    ("create.point", "at", "[0.0,0.0,0.0]"),
    ("create.ramp", "shape", "\"linear\""),
    ("create.skewbox", "centre", "[0.0,0.0,0.0]"),
    ("create.skewbox", "edges", "[[100.0,0.0,0.0],[30.0,85.0,0.0],[15.0,10.0,85.0]]"),
    ("create.expression", "claim", "\"lower_bound\""),
    ("create.expression", "divisor", "1.0"),
    ("create.expression", "k", "1.0"),
    ("create.expression", "size", "[120.0,120.0,120.0]"),
    ("create.expression", "text", "\"sqrt(x*x + y*y + z*z) - 50\""),
    ("create.sphere", "centre", "[0.0,0.0,0.0]"),
    ("create.sphere", "radius", "50.0"),
    ("create.thread", "clearance", "0.0"),
    ("create.thread", "hand", "\"right\""),
    ("create.thread", "length", "100.0"),
    ("create.thread", "major", "40.0"),
    ("create.thread", "pitch", "6.0"),
    ("create.torus", "centre", "[0.0,0.0,0.0]"),
    ("create.torus", "major_radius", "35.0"),
    ("create.torus", "minor_radius", "15.0"),
    ("create.tpms", "cell", "\"gyroid\""),
    ("create.tpms", "cell_size", "20.0"),
    ("create.tpms", "map", "\"rectangular\""),
    ("create.tpms", "thickness", "0.2"),
];

pub fn default_of(op: &str, key: &str) -> Option<&'static str> {
    DEFAULTS.iter().find(|(o, k, _)| *o == op && *k == key).map(|(_, _, v)| *v)
}
