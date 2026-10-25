// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What the direct-layer tools have picked on the control mesh, and which mesh that is.

use super::*;

/// Which quad mesh the direct-layer tools are editing.
///
/// **The cage, or a surface a conformal lattice follows** — the owner asked for the second to be
/// edited "the same way we can edit the sub-d control cage", and it is the same kind of thing: a
/// quad mesh whose current state is the truth, moved point, edge and face at a time, each gesture
/// one undo step. So it takes the cage's place rather than growing tools of its own: in Polygon
/// and Overlay it is what is drawn as the control mesh, what a click picks on, what a drag moves
/// and what the manipulator sits on. One at a time, because the selection names vertices by index
/// and an index means nothing without saying whose.
///
/// Chosen by clicking a surface's row, and given back to the cage by picking the cage's body.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum ControlMesh {
    #[default]
    Cage,
    Surface(fieldform_scene::QuadMeshId),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum Selection {
    #[default]
    None,
    Vertex(VertexId),
    Edge(VertexId, VertexId),
    Face(FaceId),
}

/// Everything picked on the control mesh, in the order it was picked.
///
/// **A set rather than one element**, asked for by the owner: *"would like to add more things to
/// the active selection — for example, maybe I want to move 4 faces at once"*. Shift adds to it
/// and takes away from it; a plain click replaces it. A move, the manipulator and the highlight
/// all read the union of its parts, so four faces drag as one piece and the one undo entry a
/// gesture makes covers all of them.
///
/// Kinds mix freely — a face and the two vertices beside it is a legitimate thing to want to move
/// — because everything downstream works in vertices, and the union is what it asks for.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct Picked(Vec<Selection>);

impl Picked {
    pub(crate) fn one(part: Selection) -> Self {
        Self(if part.is_some() { vec![part] } else { Vec::new() })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn parts(&self) -> &[Selection] {
        &self.0
    }

    pub(crate) fn contains(&self, part: Selection) -> bool {
        part.is_some() && self.0.contains(&part)
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    /// Adds `part`, or takes it away if it is already there — what Shift-clicking means
    /// everywhere else, including this app's own control points.
    pub(crate) fn toggle(&mut self, part: Selection) {
        if !part.is_some() {
            return;
        }
        match self.0.iter().position(|p| *p == part) {
            Some(at) => {
                self.0.remove(at);
            }
            None => self.0.push(part),
        }
    }

    /// The vertices of every part, each one once: two faces sharing an edge drag as one piece
    /// rather than moving their shared corners twice as far.
    pub(crate) fn vertices(&self, mesh: &QuadMesh) -> Vec<VertexId> {
        let mut out: Vec<VertexId> = Vec::new();
        for part in &self.0 {
            for v in part.vertices(mesh) {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
        }
        out
    }

    /// Where the manipulator sits and what a drag measures from: the middle of the points that
    /// will move, not the middle of the parts, so a face and a vertex pull their own weight.
    pub(crate) fn centroid(&self, mesh: &QuadMesh, placement: &Similarity3<f64>) -> Option<Point3<f32>> {
        let verts = self.vertices(mesh);
        if verts.is_empty() {
            return None;
        }
        let sum = verts
            .iter()
            .fold(Vector3::zeros(), |acc, v| acc + vertex_pos(mesh, placement, *v).coords);
        Some(Point3::from(sum / verts.len() as f32))
    }

    /// The face normal when exactly one face is picked. Several faces have several normals and
    /// averaging them would turn the handles to a direction none of them points.
    pub(crate) fn normal(&self, mesh: &QuadMesh, placement: &Similarity3<f64>) -> Option<Vector3<f32>> {
        match self.0.as_slice() {
            [only] => only.normal(mesh, placement),
            // Several faces -- an extruded cap -- turn the gizmo to their average, so the arrow
            // that leaves the surface is the one an extrude drags along.
            many if !many.is_empty() && many.iter().all(|p| matches!(p, Selection::Face(_))) => {
                let sum: Vector3<f32> = many.iter().filter_map(|p| p.normal(mesh, placement)).sum();
                sum.try_normalize(1e-9)
            }
            _ => None,
        }
    }

    /// Only one point picked, which is the one case with no extent to turn or scale about.
    pub(crate) fn is_single_vertex(&self) -> bool {
        matches!(self.0.as_slice(), [Selection::Vertex(_)])
    }

    /// Drops any part naming geometry that is no longer there, rather than emptying the whole
    /// selection: an undo that shrinks the mesh should cost the parts it removed and no more.
    pub(crate) fn keep_valid_for(&mut self, vertex_count: usize, face_count: usize) {
        self.0.retain(|part| part.is_valid_for(vertex_count, face_count));
    }
}

impl Selection {
    /// The vertices that make up this selection, i.e. the ones a move drags together.
    fn vertices(&self, cage: &QuadMesh) -> Vec<VertexId> {
        match *self {
            Selection::None => vec![],
            Selection::Vertex(v) => vec![v],
            Selection::Edge(a, b) => vec![a, b],
            Selection::Face(f) => cage.face_vertices(f).to_vec(),
        }
    }

    /// Whether this selection still refers to geometry that exists. Undo (and later,
    /// topology-changing operations) can shrink the cage out from under a stale selection.
    fn is_valid_for(&self, vertex_count: usize, face_count: usize) -> bool {
        match *self {
            Selection::None => true,
            Selection::Vertex(v) => v.index() < vertex_count,
            Selection::Edge(a, b) => a.index() < vertex_count && b.index() < vertex_count,
            Selection::Face(f) => f.index() < face_count,
        }
    }

    /// Whether this is something rather than nothing — a picked element.
    fn is_some(&self) -> bool {
        !matches!(self, Selection::None)
    }

    /// Which way a face selection points, or `None` for anything without a well-defined
    /// normal.
    ///
    /// Newell's method rather than a cross product of two edges: a cage face is a quad and
    /// need not be planar, and two edges chosen arbitrarily out of a warped quad give a normal
    /// that swings as the face flexes. Newell's is the area-weighted average of all four, so
    /// it degrades smoothly instead.
    ///
    /// A vertex or an edge is deliberately excluded. A vertex's normal would have to be
    /// averaged from the faces around it, and an edge's from the two it divides — both are
    /// defensible and neither is what "move along this" means, so the manipulator falls back
    /// to world axes there rather than picking one.
    fn normal(&self, cage: &QuadMesh, placement: &Similarity3<f64>) -> Option<Vector3<f32>> {
        let Selection::Face(f) = *self else { return None };
        let verts = cage.face_vertices(f);
        let mut n = Vector3::zeros();
        for i in 0..verts.len() {
            let a = vertex_pos(cage, placement, verts[i]);
            let b = vertex_pos(cage, placement, verts[(i + 1) % verts.len()]);
            n += Vector3::new(
                (a.y - b.y) * (a.z + b.z),
                (a.z - b.z) * (a.x + b.x),
                (a.x - b.x) * (a.y + b.y),
            );
        }
        n.try_normalize(1e-9)
    }
}
