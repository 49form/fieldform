// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Triangulates a [`fieldform_cage::Cage`] into a GPU-uploadable vertex/index buffer, purely
//! for drawing the cage in the viewport. This is display-only — the cage's own data stays a
//! quad half-edge mesh; nothing here feeds back into editing or the field representation.
//!
//! This is also the **narrowing boundary**: the document is `f64` (see D2), GPU buffers are
//! `f32`. Everything crossing into this module is cast once, here, rather than the geometry
//! code carrying two precisions around. `f32` is right for display and for picking — screen
//! precision is a few thousand pixels, nowhere near where `f32` struggles — but it is not
//! right for the model, which is why the cast only ever goes this direction.

use fieldform_cage::{FaceId, QuadMesh};
use nalgebra::{Point3, Vector3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Default)]
pub struct CageMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// The source cage face each triangle came from (one entry per triangle, i.e.
    /// `indices.len() / 3` entries) — lets a raycast hit be traced back to a pickable face.
    face_of_triangle: Vec<FaceId>,
    /// Drawn from both sides. A cage is closed, so a face turned away is the far side of the
    /// solid and culling it is what draws the near side correctly without a depth buffer; a
    /// surface a lattice follows is an open sheet whose underside is as much the surface as its
    /// top, and culled it vanishes when turned over.
    pub two_sided: bool,
}

impl CageMesh {
    /// Builds a flat-shaded mesh from arbitrary triangles — an import, rather than a cage.
    ///
    /// Flat-shaded like the cage's own, and for a plainer reason than the cage has: an
    /// imported mesh may be open, so a smooth normal averaged around a vertex would be
    /// averaging over a hole. A facet tells the truth about what arrived.
    ///
    /// No `face_of_triangle`: there are no cage faces to pick here. An import is referenced,
    /// not edited, so nothing needs to trace a hit back to something draggable.
    pub fn from_triangles(positions: &[Point3<f64>], triangles: &[[u32; 3]]) -> Self {
        let mut vertices = Vec::with_capacity(triangles.len() * 3);
        let mut indices = Vec::with_capacity(triangles.len() * 3);
        for t in triangles {
            let Some(p) = (0..3)
                .map(|k| positions.get(t[k] as usize).map(|p| p.cast::<f32>()))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            // A degenerate triangle has no normal to take; it is dropped rather than drawn
            // with a NaN one, which would take the whole draw call with it.
            let Some(normal) = (p[1] - p[0]).cross(&(p[2] - p[0])).try_normalize(1e-12) else {
                continue;
            };
            let base = vertices.len() as u32;
            for corner in &p {
                vertices.push(Vertex {
                    position: [corner.x, corner.y, corner.z],
                    normal: [normal.x, normal.y, normal.z],
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
        Self { vertices, indices, face_of_triangle: Vec::new(), two_sided: false }
    }

    /// Builds a flat-shaded triangle mesh for rendering: each face gets its own vertices
    /// (not shared with neighbors) so every triangle has a consistent per-face normal
    /// without needing flat-interpolation shader qualifiers.
    pub fn from_cage(cage: &QuadMesh) -> Self {
        let mut vertices = Vec::with_capacity(cage.face_count() * 4);
        let mut indices = Vec::with_capacity(cage.face_count() * 6);
        let mut face_of_triangle = Vec::with_capacity(cage.face_count() * 2);

        for face in cage.face_ids() {
            let corners = cage.face_vertices(face);
            let p: [Point3<f32>; 4] =
                corners.map(|v| cage.vertices()[v.index()].position.cast::<f32>());

            let normal = (p[1] - p[0]).cross(&(p[2] - p[0])).normalize();
            let base = vertices.len() as u32;
            for corner in p {
                vertices.push(Vertex {
                    position: [corner.x, corner.y, corner.z],
                    normal: [normal.x, normal.y, normal.z],
                });
            }
            // Two triangles for the quad, matching the CCW winding already validated by
            // Cage's construction-time invariants.
            indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base,
                base + 2,
                base + 3,
            ]);
            face_of_triangle.push(face);
            face_of_triangle.push(face);
        }

        Self {
            vertices,
            indices,
            face_of_triangle,
            two_sided: cage.is_open(),
        }
    }

    /// Recomputes the vertices of specific cage faces in place.
    ///
    /// [`Self::from_cage`] gives each face its own four vertices at `face.index() * 4`, so a
    /// face's data is contiguous and independent — moving a vertex only invalidates the faces
    /// that contain it, and those can be rewritten without touching the rest. Flat shading
    /// means the face normal changes too, which is why this recomputes rather than just
    /// translating the positions.
    pub fn update_cage_faces(&mut self, cage: &QuadMesh, faces: &[FaceId]) {
        for &face in faces {
            let corners = cage.face_vertices(face);
            let p: [Point3<f32>; 4] =
                corners.map(|v| cage.vertex_position(v).cast::<f32>());
            let normal = (p[1] - p[0])
                .cross(&(p[2] - p[0]))
                .try_normalize(1e-12)
                .unwrap_or_else(Vector3::z);

            let base = face.index() * 4;
            for (k, corner) in p.iter().enumerate() {
                self.vertices[base + k] = Vertex {
                    position: [corner.x, corner.y, corner.z],
                    normal: [normal.x, normal.y, normal.z],
                };
            }
        }
    }

    /// The vertex range a cage face occupies in a [`Self::from_cage`] mesh.
    pub fn cage_face_vertex_range(face: FaceId) -> std::ops::Range<usize> {
        let base = face.index() * 4;
        base..base + 4
    }

    /// Read-only access to the vertex array, for uploading sub-ranges to the GPU.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    // The limit-surface half of this file lived here: `from_limit_surface` built an n x n
    // patch grid per cage face, `retessellate_faces` refreshed a few faces of it during a
    // drag, and `face_vertex_range` described the per-face layout both depended on. Nothing
    // drew that mesh once the viewport became field-traced, so all three went with it. The
    // limit surface itself is not gone — it is what the traced view bakes and what export
    // will mesh, through `fieldform_field::CageField` rather than through here.

    /// Casts a ray against the mesh (brute force over all triangles — the cage is small
    /// enough that this doesn't need an acceleration structure) and returns the closest hit
    /// point and the face it belongs to.
    ///
    /// Used both for point-of-interest picking on tumble-drag start (the usual
    /// raycast-to-set-POI step of a point-of-interest orbit camera)
    /// and for face selection.
    pub fn raycast(&self, origin: Point3<f32>, dir: Vector3<f32>) -> Option<(Point3<f32>, FaceId)> {
        let dir = dir.normalize();
        let mut closest: Option<(f32, usize)> = None;

        for (tri_index, tri) in self.indices.chunks_exact(3).enumerate() {
            let a = Point3::from(self.vertices[tri[0] as usize].position);
            let b = Point3::from(self.vertices[tri[1] as usize].position);
            let c = Point3::from(self.vertices[tri[2] as usize].position);

            if let Some(t) = moller_trumbore(origin, dir, a, b, c) {
                if closest.is_none_or(|(best, _)| t < best) {
                    closest = Some((t, tri_index));
                }
            }
        }

        closest.map(|(t, tri_index)| (origin + dir * t, self.face_of_triangle[tri_index]))
    }
}

/// Standard Möller–Trumbore ray-triangle intersection; returns the ray parameter `t` of
/// the closest forward intersection, if any.
fn moller_trumbore(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,
) -> Option<f32> {
    const EPSILON: f32 = 1e-6;
    let edge1 = b - a;
    let edge2 = c - a;
    let h = dir.cross(&edge2);
    let det = edge1.dot(&h);
    if det.abs() < EPSILON {
        return None; // ray parallel to the triangle
    }
    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = inv_det * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(&edge1);
    let v = inv_det * dir.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = inv_det * edge2.dot(&q);
    (t > EPSILON).then_some(t)
}
