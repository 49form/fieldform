// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The operation layer: every edit to a document goes through here.
//!
//! This exists because three separate requirements turn out to be the same requirement:
//!
//! - **Undo** needs each edit to be a discrete, reversible thing → [`Command`].
//! - **Rebindable hotkeys** need keys to name an operation rather than call code inline →
//!   [`Keymap`] maps a [`Shortcut`] to an operation name plus arguments.
//! - **A Python API** needs operations callable by name with dynamic arguments →
//!   [`OpRegistry`] dispatches `("cage.translate_vertices", args)` to a [`Command`].
//!
//! All three are then the same pipeline: *name + args → Command → execute on the Document,
//! recorded on the undo stack*. A key press and a Python call are the same thing arriving
//! through different front doors, so they cannot drift apart in behaviour.
//!
//! ```text
//!   key press ──┐
//!   Python call ├──► OpRegistry::build(name, args) ──► Box<dyn Command>
//!   menu item ──┘                                            │
//!                                        CommandStack::execute(doc, cmd)
//!                                              │                    │
//!                                        doc mutated          pushed for undo
//! ```

mod command;
mod keymap;
mod ops;
#[cfg(test)]
mod ownership_tests;
mod registry;

pub use command::{Command, CommandStack, OpError, StackMark};
// The document model lives in `fieldform-scene`; ops operate on it rather than defining it.
pub use fieldform_scene::Document;
pub use keymap::{Keymap, Modifiers, Shortcut};
pub use ops::{
    Batch, Named, OwnershipSnapshot, SetControlPoints, SetControlWeights, SetCurve, SetName,
    SetOutputs, SetOwnership, SetParameter, SetPartGraph, SetPartMaterial, SetRelation, SetShown,
    SetSource, SetSurface, SetSurfaceAxis, SetSurfacePositions, SetVertexPositions, SetVisible,
};
pub use registry::{Args, OpRegistry, Value};
