// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A section cut: a plane the viewport cuts the part away at, kept in the document as a block.
//!
//! **A view, kept as a block.** It changes what is on screen and nothing about the part: the
//! graph, a mesh export and the field are untouched. It was application state at first, on the
//! reading that two windows on one document could cut it differently. The owner then asked for it
//! to be a block -- switched on and off, opened, and placed with the same rows a plane block is
//! placed with -- and for it to be saved and undone like one (2026-11-07). So it is a record of the
//! document beside the outputs, as a Save Image block is, and not a node: a node is a field, an
//! operand and a body, and a section is none of those.
//!
//! **Placed the way a plane is placed** (D9): an origin that is a [`Place`] and a facing that is a
//! [`Turn`], so the panel draws the plane's own rows for it, and a section can sit on a point or
//! face the way another plane faces. The half past the plane, along its normal, is what is cut
//! away; `flipped` keeps the other half, as a plane's does.

use serde::{Deserialize, Serialize};

use crate::id::{PartId, SectionCutId};
use crate::{Place, Turn};

/// One section cut.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SectionCut {
    pub id: SectionCutId,
    /// Its number, in the order made: "Section 3" stays Section 3 when Section 1 is removed.
    pub number: u32,
    /// The part whose view it cuts. A section on another part waits for that part to be shown.
    pub part: PartId,
    /// Whether it cuts. Off, it stays in the list, where it is, ready to be switched back on.
    pub on: bool,
    /// Where the plane passes through.
    pub origin: Place,
    /// Which way it faces: its normal is the turn's `+Z`.
    #[serde(default)]
    pub turn: Turn,
    /// Keeps the other half: the same plane facing the other way, which is not a rotation.
    #[serde(default)]
    pub flipped: bool,
}

impl SectionCut {
    /// What a section is called until it is renamed: by its number.
    pub fn title(&self) -> String {
        format!("Section {}", self.number)
    }
}
