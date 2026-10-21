// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Getting fieldform's geometry onto a GPU.
//!
//! The viewport meshes the composed field and uploads triangles, which caps what the tool can
//! show: a 1 m cube of 50 mm gyroid cells is 6.25e9 triangles and ~225 GB meshed, and traces
//! in a fraction of a second with nothing stored. This crate is the path off that.
//!
//! The split it exists to serve: **procedural geometry stays procedural, and only irreducible
//! leaves get baked.** A lattice is a few operations and evaluates directly. A cage's limit
//! surface has no closed form, so it is sampled once into a grid the GPU reads — see
//! [`BakedField`].
//!
//! Kept separate from `fieldform-field` so that graphics dependencies stay out of the geometry
//! layer, which has none and should keep none.

mod bake;
pub mod codegen;
/// The march on the CPU, for holding the GPU's to it. An instrument, not an API.
#[doc(hidden)]
pub mod cpu_march;
mod environment;
mod flake;
mod render;
mod shading;
#[cfg(feature = "march-stats")]
pub mod stats;

pub use bake::{BakedField, BakedFieldSource};
pub use codegen::{
    ShaderSize,
    generate, generate_for_measurement, generate_with, generate_with_masks, generate_with_options,
    generate_with_run_floor,
    BakedLeaf,
    CodegenError, CurveUse,
    ShaderProgram, PRELUDE,
};
// `Environment`, `CubeLevel`, `Panel`, `SkipGridReading` and `Light` are named by nothing
// outside, but another crate is handed each of them — `EnvironmentMaps::bake` takes an
// environment and the maps it returns hold cube levels, `Studio::panels` holds panels,
// `read_skip_grid` returns a reading, `Look::lights` holds lights — and a type another crate
// receives is one it should be able to name, which only the root can promise once these modules
// are private.
pub use environment::{
    BakeSettings, CubeLevel, Environment, EnvironmentChoice, EnvironmentMaps, Panel, Studio,
};
pub use render::{
    BuildCost,
    bake_resolution_limit, required_limits, timing_features, Camera, Hit, Occlusion,
    RenderError, Renderer, SkipGridReading, TileMaskReading,
};
pub use shading::{
    halton_jitter, DisplayImage, Draft, Light, Look, ShadingResources, SubmissionCost,
    SubmissionTicket,
};
