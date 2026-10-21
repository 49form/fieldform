// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! An imported triangle mesh, as document data.
//!
//! Finished geometry rather than scaffolding, which is the whole reason it is not a
//! [`fieldform_cage::Cage`]: a cage is a handful of control points to push around and must be
//! closed, manifold and all-quad to mean anything, while this is a solid someone else authored
//! and this document only references, cuts against or blends with.
//!
//! Data and no behaviour. Turning one into a field is `fieldform-field`'s job, and reading one
//! out of a `.stp` is `fieldform-io`'s; keeping all three apart is what stops the document
//! layer growing an opinion about either.

use serde::{Deserialize, Serialize};

/// A closed triangle mesh, and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshData {
    /// What to call it in the block list. The source file's name, usually.
    pub name: String,
    /// Vertex positions in canonical millimetres, like all geometry (D2).
    pub positions: Vec<[f64; 3]>,
    /// Triangles as indices into `positions`, wound so the normal points *out* of the solid.
    ///
    /// Which way round is not a convention to be assumed: exporters disagree, and getting it
    /// wrong swaps inside for outside while leaving a mesh that passes every structural check.
    /// The importer measures the signed volume and reverses when it comes out negative.
    pub triangles: Vec<[u32; 3]>,
    /// The tolerance the source was tessellated at, in millimetres, when it came from a
    /// curved-surface format.
    ///
    /// Recorded because without it a document cannot tell whether its mesh is a faithful
    /// sample of the source or a coarse one, and a later exact re-import would have nothing to
    /// compare against. `None` for a format that was already triangles, where there is no
    /// tolerance to record and inventing one would be a lie.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance_mm: Option<f64>,
    /// Edges used by other than exactly two triangles: holes, and seams where sheets meet.
    ///
    /// Zero means the mesh is a closed surface and can become a signed distance field. Anything
    /// else means it cannot — there is no inside for a sign to refer to — and the mesh is
    /// *still kept*, because seeing what arrived is worth more than refusing it. A real CAD
    /// export whose tessellator dropped a handful of hard faces is the ordinary case, not a
    /// corrupt file.
    ///
    /// Counted at import rather than on demand: it decides how the body behaves everywhere,
    /// and recomputing it per frame over a million triangles would be absurd.
    #[serde(default)]
    pub open_edges: usize,
    /// The file this came from, when it came from one.
    ///
    /// What makes several solids from one assembly a single block: they share this, and the
    /// block list groups by it. `None` for a mesh that arrived some other way.
    ///
    /// The *group* is derived from this rather than stored anywhere, which is the same rule
    /// bodies follow — a stored list of members is a second description of the same structure
    /// and the two can disagree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<crate::id::SourceId>,
    /// The B-rep edges of the solid this was tessellated from, as polylines.
    ///
    /// **What makes an import look like CAD rather than like a mesh.** A cylinder tessellated
    /// to triangles and a cylinder with its silhouette and seam curves drawn read as different
    /// kinds of object at the same tessellation, and the difference is these.
    ///
    /// They are *decoration on the same tessellation*, not a second opinion about where the
    /// surface is: they come from the same OCCT mesh pass the triangles do. That distinction is
    /// the one `import.md` warns about — two descriptions of one solid that disagree is the
    /// failure a second display path would be.
    ///
    /// Empty for a format that has no edges to give, which is every one except STEP.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<Vec<[f64; 3]>>,
}

impl MeshData {
    /// Whether this mesh encloses a volume, and so whether it can become a field.
    pub fn is_closed(&self) -> bool {
        self.open_edges == 0
    }

    /// Counts edges not shared by exactly two triangles.
    pub fn count_open_edges(triangles: &[[u32; 3]]) -> usize {
        let mut uses: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        for t in triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        uses.values().filter(|c| **c != 2).count()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// The axis-aligned box the mesh occupies, or `None` when it has no vertices.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let first = *self.positions.first()?;
        let mut lo = first;
        let mut hi = first;
        for p in &self.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Some((lo, hi))
    }
}
