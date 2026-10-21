// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What leaves the program: a tessellation of a body, and a file written from one.
//!
//! **Two blocks, because they are two acts** — backlog item 53, ruled by the owner against
//! how other CAD software actually behaves. Making a mesh is the slow, interesting part: it has a body, a detail
//! and a result worth looking at. Writing a file is where that result goes. Keeping them apart is
//! what lets one mesh feed several files, and several tessellations of one part sit side by side
//! at different details, without meshing anything twice.
//!
//! Both are **document** things, saved with it, so a workflow's exports can be run again. Neither
//! is a node of a part's graph: they are not geometry and nothing composes them. A tessellation
//! names the part and body it meshes the way a conformal lattice names its surface, and an export
//! names the tessellation it writes.
//!
//! The meshes themselves are not here and are never saved. A mesh is derived from the geometry,
//! and a stored copy would be a second description of it that could disagree with the first.

use serde::{Deserialize, Serialize};

use crate::id::{ExportId, NodeId, PartId, TessellationId};

/// How closely a mesh must follow the field.
///
/// **Tolerances, not a resolution.** A cell count is a statement about the sampling grid; these
/// are statements about the *answer*, which is what a person choosing a setting actually has an
/// opinion about. The mesher spends whatever cells each region needs to meet them.
///
/// Lives here rather than beside the mesher so that a block saved in a document can hold one:
/// `fieldform-field` re-exports it, and it is the one type for "how finely to mesh" (D9).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshTolerance {
    /// How far the mesh may sit from the true surface, in millimetres.
    ///
    /// The *sag* between a chord and the arc it cuts. This is the one to reach for when the
    /// question is "how accurate does this part need to be".
    pub chord_mm: f64,
    /// How much the surface may turn within one cell, in degrees.
    ///
    /// Separate from `chord_mm` because it catches what that cannot: a crease has **no** sag —
    /// two flat faces meeting at an edge sit exactly on their own planes — but it turns sharply.
    /// This is the control that refines hard edges, and lowering it makes them crisp.
    pub angle_deg: f64,
    /// The longest a cell may be however flat it is, in millimetres. Zero means no cap.
    ///
    /// A large flat face satisfies both tests above at any size, so without this it meshes as a
    /// handful of enormous triangles. Long thin triangles integrate badly in analysis and
    /// round-trip poorly, so this caps them — but for a file going only to a slicer, no cap is
    /// the right answer and the saving is the whole point.
    pub max_edge_mm: f64,
    /// The smallest a cell may become, in millimetres.
    ///
    /// The floor on the whole process, and what makes it terminate: a true crease never
    /// satisfies the angular test at any size, so without a floor it would subdivide forever.
    pub min_cell_mm: f64,
}

impl Default for MeshTolerance {
    fn default() -> Self {
        Self {
            chord_mm: 0.02,
            angle_deg: 15.0,
            max_edge_mm: 0.0,
            min_cell_mm: 0.02,
        }
    }
}

/// How finely to tessellate: a preset, or tolerances set by hand.
///
/// **A preset is stored as a preset, not as the numbers it wrote.** Each one is relative to the
/// size of what is being meshed, so "Normal" means the same thing on a 10 mm part and a 1000 mm
/// one; storing the millimetres it produced once would freeze them, and a body that grew would
/// be meshed at a detail that no longer means Normal. Only `Custom` carries numbers, because
/// only there are the numbers what the person chose.
///
/// Measured on a 185 mm sphere-gyroid-and-cuboid document:
///
/// | | triangles | STL | time |
/// |---|---|---|---|
/// | Coarse | 128 466 | 6.1 MB | 1.1 s |
/// | Normal | 459 260 | 21.9 MB | 4.4 s |
/// | Full | 2 303 328 | 109.8 MB | 29.3 s |
///
/// Roughly six times the triangles per step, so the ends of the range are two orders of
/// magnitude apart and there is somewhere useful to go in both directions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshDetail {
    Coarse,
    Normal,
    /// Renamed from "Fine" at the owner's request, to match the Display quality ladder — Fast,
    /// Balanced, Full, Enchilada. A document saved before the rename still reads.
    #[serde(alias = "fine")]
    Full,
    /// The top of the range, and named as the top of the Display quality ladder is — the two
    /// are one vocabulary for "as much as it will do", so a person who has learned one has
    /// learned the other.
    Enchilada,
    /// The tolerances were set by hand, so no preset describes them.
    Custom(MeshTolerance),
}

impl MeshDetail {
    /// The presets, in the order they are offered.
    pub const PRESETS: [MeshDetail; 4] = [
        MeshDetail::Coarse,
        MeshDetail::Normal,
        MeshDetail::Full,
        MeshDetail::Enchilada,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MeshDetail::Coarse => "Coarse",
            MeshDetail::Normal => "Normal",
            MeshDetail::Full => "Full",
            MeshDetail::Enchilada => "Enchilada",
            MeshDetail::Custom(_) => "Custom",
        }
    }

    /// The tolerances this detail means for something `longest_mm` across.
    pub fn tolerance(self, longest_mm: f64) -> MeshTolerance {
        // (floor, chord, angle). The floor is what bounds the cost, the chord is how closely a
        // curve is followed, and the angle is what sharpens a crease -- tightened together,
        // since a person asking for more detail wants all three.
        let (floor, chord, angle_deg) = match self {
            MeshDetail::Coarse => (200.0, 1000.0, 30.0),
            MeshDetail::Normal => (460.0, 3700.0, 20.0),
            MeshDetail::Full => (2000.0, 12000.0, 12.0),
            MeshDetail::Enchilada => (5000.0, 30000.0, 8.0),
            MeshDetail::Custom(tolerance) => return tolerance,
        };
        let l = longest_mm.max(1e-6);
        MeshTolerance {
            chord_mm: l / chord,
            angle_deg,
            // Never capped by a preset. A flat face meshing as a few large triangles is the
            // whole saving, and a cap is a thing to reach for deliberately.
            max_edge_mm: 0.0,
            min_cell_mm: l / floor,
        }
    }
}

/// A body turned into triangles — the first of the two export blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tessellation {
    pub id: TessellationId,
    /// A name given to it, or `None` for [`Tessellation::DEFAULT_NAME`].
    ///
    /// **The block's own name is permanent and a given one is not.** Clearing a rename puts the
    /// default back, the way a body's row falls back to what the body is; so the default is not
    /// stored as though somebody had typed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The part whose body this meshes.
    pub part: PartId,
    /// One body of that part, or every body in it when absent.
    ///
    /// Named by its node, so *parts, never faces* holds. A body is a root of its graph and a
    /// boolean can consume it; when that happens the block says it has nothing to mesh rather
    /// than quietly meshing something else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<NodeId>,
    pub detail: MeshDetail,
    /// Slide vertices along the surface until the triangles are better shaped. Same file, same
    /// worst deviation, roughly three times as long — on by default, since it takes nothing.
    pub improve: bool,
    /// Collapse edges the surface does not need. A third fewer triangles for most of an order of
    /// magnitude in time: a real trade, so off by default.
    pub reduce: bool,
    /// Whether it re-runs by itself when what it meshes changes.
    pub active: bool,
}

/// A file written from a tessellation — the second of the two export blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Export {
    pub id: ExportId,
    /// A name given to it, or `None` for [`Export::DEFAULT_NAME`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The mesh it writes. Several exports may name one tessellation; nothing meshes twice.
    pub source: TessellationId,
    /// Where it writes, with the format taken from the extension.
    pub path: String,
    /// Whether it writes again by itself each time its tessellation finishes.
    pub active: bool,
}

impl Tessellation {
    /// What the block is called until somebody names it.
    pub const DEFAULT_NAME: &'static str = "Tessellate";

    pub fn title(&self) -> &str {
        self.name.as_deref().unwrap_or(Self::DEFAULT_NAME)
    }
}

impl Export {
    /// What the block is called until somebody names it — another CAD package's name for the
    /// same block, since
    /// that is the vocabulary the owner brought.
    pub const DEFAULT_NAME: &'static str = "Export Mesh";

    pub fn title(&self) -> &str {
        self.name.as_deref().unwrap_or(Self::DEFAULT_NAME)
    }
}
