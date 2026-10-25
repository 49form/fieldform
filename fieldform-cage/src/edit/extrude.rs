// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Extrude and inset: one topology, two placements (the cage-editing notes, item 77).
//!
//! **The lift is built once and tested for its winding before either placement is written.** A
//! region of faces is lifted by duplicating its *border* vertices only -- its interior vertices
//! belong to the cap alone and move with it, keeping their indices -- and joining each border edge
//! to its duplicate with a wall quad. The cap keeps the region's face slots, so a selection
//! survives the lift and the next extrude chains from it. Extrude then moves the cap off the
//! surface; inset moves the cap's border inward within it.

use super::{rebuild, EditError, EditResult, Highlight};
use crate::validate::face_normal;
use crate::{CageError, FaceId, QuadMesh, VertexId};
use nalgebra::Vector3;
use std::collections::{HashMap, HashSet};

/// Whether connected faces lift as one, or each face on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Region,
    Individual,
}

/// Which way an extrude moves its cap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    /// Along the region's area-weighted normal, the whole cap together.
    Normal,
    /// Each cap vertex along its own normal, scaled so a flat-sided region keeps its wall at the
    /// distance asked.
    VertexNormals,
    /// Along a fixed axis.
    Axis(Vector3<f64>),
}

/// One region of a lift: its faces, its border in region winding, and where each border vertex's
/// duplicate went.
#[derive(Clone, Debug)]
pub struct Region {
    pub faces: Vec<FaceId>,
    /// Border edges as the region's faces run them, `a → b` with the outside on the other side.
    pub border: Vec<(VertexId, VertexId)>,
    /// Border vertex → its duplicate on the cap.
    pub duplicate: HashMap<VertexId, VertexId>,
    /// The region's interior vertices, which the cap keeps.
    pub interior: Vec<VertexId>,
}

impl Region {
    /// Every vertex of the cap: the duplicates and the interior.
    pub fn cap_vertices(&self) -> Vec<VertexId> {
        let mut out: Vec<VertexId> = self.duplicate.values().copied().collect();
        out.extend(self.interior.iter().copied());
        out.sort();
        out
    }
}

/// A lifted mesh with its regions, before any placement.
pub struct Lift {
    pub result: EditResult,
    pub regions: Vec<Region>,
}

/// The edge-connected regions of `faces`, each as its face list.
fn regions_of(mesh: &QuadMesh, faces: &[FaceId], mode: Mode) -> Vec<Vec<FaceId>> {
    let chosen: HashSet<FaceId> = faces.iter().copied().collect();
    if mode == Mode::Individual {
        return faces.iter().map(|&f| vec![f]).collect();
    }
    let mut seen: HashSet<FaceId> = HashSet::new();
    let mut out = Vec::new();
    for &start in faces {
        if !seen.insert(start) {
            continue;
        }
        let mut region = vec![start];
        let mut stack = vec![start];
        while let Some(f) = stack.pop() {
            for he in mesh.face_half_edges(f) {
                let twin = mesh.half_edge(he).twin;
                if twin == he {
                    continue;
                }
                let g = mesh.half_edge(twin).face;
                if chosen.contains(&g) && seen.insert(g) {
                    region.push(g);
                    stack.push(g);
                }
            }
        }
        out.push(region);
    }
    out
}

/// Lifts the regions of `faces`: the topology both placements share (notes §1).
pub fn lift(mesh: &QuadMesh, faces: &[FaceId], mode: Mode) -> Result<Lift, EditError> {
    if faces.is_empty() {
        return Err(EditError::refuse("select the faces to lift first", Highlight::default()));
    }
    for &f in faces {
        if f.index() >= mesh.face_count() {
            return Err(EditError::refuse(format!("face {} does not exist", f.index()), Highlight::default()));
        }
    }
    let mut positions = mesh.position_list();
    let mut quads = mesh.quad_lists();
    let original_vertices = positions.len();
    let original_faces = quads.len();
    let mut regions = Vec::new();

    for region_faces in regions_of(mesh, faces, mode) {
        let in_region: HashSet<FaceId> = region_faces.iter().copied().collect();
        // The border, as the region runs it.
        let mut border: Vec<(VertexId, VertexId)> = Vec::new();
        for &f in &region_faces {
            for he in mesh.face_half_edges(f) {
                let twin = mesh.half_edge(he).twin;
                let outside = twin == he || !in_region.contains(&mesh.half_edge(twin).face);
                if outside {
                    border.push((mesh.half_edge(he).origin, mesh.destination_of(he)));
                }
            }
        }
        let border_vertices: HashSet<VertexId> = border.iter().flat_map(|&(a, b)| [a, b]).collect();
        let mut used: Vec<VertexId> = region_faces.iter().flat_map(|&f| mesh.face_vertices(f)).collect();
        used.sort();
        used.dedup();
        let interior: Vec<VertexId> = used.iter().copied().filter(|v| !border_vertices.contains(v)).collect();

        // Duplicate the border vertices, appended; the cap re-points at them.
        let mut duplicate: HashMap<VertexId, VertexId> = HashMap::new();
        let mut border_sorted: Vec<VertexId> = border_vertices.iter().copied().collect();
        border_sorted.sort();
        for v in border_sorted {
            positions.push(positions[v.index()]);
            duplicate.insert(v, VertexId::from_index(positions.len() - 1));
        }
        for &f in &region_faces {
            for corner in quads[f.index()].iter_mut() {
                if let Some(d) = duplicate.get(&VertexId::from_index(*corner as usize)) {
                    *corner = d.index() as u32;
                }
            }
        }
        // The wall: `(a, b, b', a')`, the one winding the neighbour outside and the cap allow.
        for &(a, b) in &border {
            let (a2, b2) = (duplicate[&a], duplicate[&b]);
            quads.push(vec![a.index() as u32, b.index() as u32, b2.index() as u32, a2.index() as u32]);
        }
        regions.push(Region { faces: region_faces, border, duplicate, interior });
    }

    let new_vertex_count = positions.len();
    let new_face_count = quads.len();
    let built = QuadMesh::from_faces(positions, &quads).map_err(|e| match e {
        // Two regions sharing a vertex and no edge each duplicated it; the duplicates meet at the
        // original as two fans. Never resolved by guessing which region should own it.
        CageError::NonManifoldVertex { vertex } => EditError::refuse(
            "these regions touch at a corner — extrude them separately or in individual mode",
            Highlight { vertices: vec![vertex], ..Highlight::default() },
        ),
        e => EditError::bug("lift", e),
    })?;
    Ok(Lift {
        result: EditResult {
            mesh: built,
            new_vertices: (original_vertices..new_vertex_count).map(VertexId::from_index).collect(),
            new_faces: (original_faces..new_face_count).map(FaceId::from_index).collect(),
            replaced: HashMap::new(),
            vertex_map: None,
            face_map: None,
        },
        regions,
    })
}

/// A region's area-weighted normal: the sum of its faces' diagonal cross products.
fn region_normal(mesh: &QuadMesh, faces: &[FaceId]) -> Vector3<f64> {
    faces.iter().map(|&f| face_normal(mesh, f)).sum::<Vector3<f64>>().try_normalize(1e-18).unwrap_or_else(Vector3::z)
}

/// Where each cap vertex goes for an extrude of `distance` in `direction` (notes §2), on the
/// lifted mesh, as positions to set.
pub fn extrude_offsets(lift: &Lift, direction: Direction, distance: f64) -> Vec<(VertexId, Vector3<f64>)> {
    let mesh = &lift.result.mesh;
    let mut out = Vec::new();
    for region in &lift.regions {
        let normal = region_normal(mesh, &region.faces);
        let axis = match direction {
            Direction::Axis(axis) => axis.try_normalize(1e-18).unwrap_or(normal),
            _ => normal,
        };
        for v in region.cap_vertices() {
            let offset = match direction {
                Direction::VertexNormals => {
                    // Averaged over the region faces using the vertex, scaled against the worst
                    // of those faces so a flat-sided region keeps its wall at `distance`.
                    let faces: Vec<FaceId> = mesh.adjacent_faces(v).into_iter().filter(|f| region.faces.contains(f)).collect();
                    let n = faces.iter().map(|&f| face_normal(mesh, f).try_normalize(1e-18).unwrap_or(normal)).sum::<Vector3<f64>>();
                    let n = n.try_normalize(1e-18).unwrap_or(normal);
                    let cos = faces
                        .iter()
                        .map(|&f| n.dot(&face_normal(mesh, f).try_normalize(1e-18).unwrap_or(normal)))
                        .fold(1.0f64, f64::min);
                    n * (distance / cos.max(0.25))
                }
                _ => axis * distance,
            };
            out.push((v, offset));
        }
    }
    out
}

/// Lifts and moves: the extrude as one call, for the console verb.
pub fn extrude(
    mesh: &QuadMesh,
    faces: &[FaceId],
    mode: Mode,
    direction: Direction,
    distance: f64,
) -> Result<EditResult, EditError> {
    let lift = lift(mesh, faces, mode)?;
    let offsets = extrude_offsets(&lift, direction, distance);
    let mut result = lift.result;
    for (v, offset) in offsets {
        let p = result.mesh.vertex_position(v);
        result.mesh.set_vertex_position(v, p + offset);
    }
    Ok(result)
}

/// Grows an open mesh at selected rim edges: one quad per edge, chains sharing their new vertices
/// (notes §3). An interior edge is refused: extruding it would join three faces at one edge.
pub fn extrude_boundary_edges(
    mesh: &QuadMesh,
    edges: &[(VertexId, VertexId)],
    distance: f64,
    axis: Option<Vector3<f64>>,
) -> Result<EditResult, EditError> {
    if edges.is_empty() {
        return Err(EditError::refuse("select the open edges to extrude first", Highlight::default()));
    }
    // Each edge as the half-edge its one face runs it by.
    let mut rim: Vec<(VertexId, VertexId, FaceId)> = Vec::new();
    for &(a, b) in edges {
        let he = [mesh.directed(a, b), mesh.directed(b, a)].into_iter().flatten().find(|&h| mesh.is_boundary(h));
        let Some(he) = he else {
            return Err(EditError::refuse(
                "only an open edge can be extruded — extruding this one would join three faces at one edge",
                Highlight { edges: vec![(a, b)], ..Highlight::default() },
            ));
        };
        rim.push((mesh.half_edge(he).origin, mesh.destination_of(he), mesh.half_edge(he).face));
    }

    let mut positions = mesh.position_list();
    let mut quads = mesh.quad_lists();
    let (original_vertices, original_faces) = (positions.len(), quads.len());
    // The outward direction of each edge, in its face's plane: the face lies to the left of
    // `a → b`, so `(b − a) × n` points away from it.
    let mut outward: HashMap<VertexId, Vector3<f64>> = HashMap::new();
    for &(a, b, f) in &rim {
        let n = face_normal(mesh, f).try_normalize(1e-18).unwrap_or_else(Vector3::z);
        let dir = (mesh.vertex_position(b) - mesh.vertex_position(a)).cross(&n).try_normalize(1e-18).unwrap_or(n);
        for v in [a, b] {
            *outward.entry(v).or_insert_with(Vector3::zeros) += dir;
        }
    }
    let mut duplicate: HashMap<VertexId, VertexId> = HashMap::new();
    let mut ends: Vec<VertexId> = outward.keys().copied().collect();
    ends.sort();
    for v in ends {
        let dir = match axis {
            Some(axis) => axis.try_normalize(1e-18).unwrap_or_else(Vector3::z),
            None => outward[&v].try_normalize(1e-18).unwrap_or_else(Vector3::z),
        };
        positions.push(mesh.vertex_position(v) + dir * distance);
        duplicate.insert(v, VertexId::from_index(positions.len() - 1));
    }
    // `(b, a, a', b')`: the face runs `a → b`, so the new quad runs `b → a`.
    for &(a, b, _) in &rim {
        quads.push(vec![b.index() as u32, a.index() as u32, duplicate[&a].index() as u32, duplicate[&b].index() as u32]);
    }
    let (nv, nf) = (positions.len(), quads.len());
    Ok(EditResult {
        mesh: rebuild("extrude_boundary_edges", positions, &quads)?,
        new_vertices: (original_vertices..nv).map(VertexId::from_index).collect(),
        new_faces: (original_faces..nf).map(FaceId::from_index).collect(),
        replaced: HashMap::new(),
        vertex_map: None,
        face_map: None,
    })
}

/// Where each cap border vertex goes for an inset of `thickness` (notes §4): inward along the
/// bisector of its two border edges' inward directions, mitred so the wall strip keeps an even
/// width along a straight run. Interior cap vertices stay.
fn inset_offsets(lift: &Lift, thickness: f64) -> Vec<(VertexId, Vector3<f64>)> {
    let mesh = &lift.result.mesh;
    let mut out = Vec::new();
    for region in &lift.regions {
        // The inward direction of each border edge, in the plane of the region face on it.
        let mut inward: HashMap<VertexId, Vec<Vector3<f64>>> = HashMap::new();
        for &(a, b) in &region.border {
            let face = region
                .faces
                .iter()
                .copied()
                .find(|&f| {
                    let c = mesh.face_vertices(f);
                    let (a2, b2) = (region.duplicate[&a], region.duplicate[&b]);
                    (0..4).any(|i| c[i] == a2 && c[(i + 1) % 4] == b2)
                })
                .unwrap_or(region.faces[0]);
            let n = face_normal(mesh, face).try_normalize(1e-18).unwrap_or_else(Vector3::z);
            let dir = n.cross(&(mesh.vertex_position(b) - mesh.vertex_position(a))).try_normalize(1e-18).unwrap_or(n);
            inward.entry(a).or_default().push(dir);
            inward.entry(b).or_default().push(dir);
        }
        for (v, dirs) in inward {
            let sum: Vector3<f64> = dirs.iter().sum();
            // Two unit inward normals meeting at the interior angle α sum to 2·sin(α/2) long.
            let half_sine = (sum.norm() / 2.0).max(0.25);
            let bisector = sum.try_normalize(1e-18).unwrap_or(dirs[0]);
            out.push((region.duplicate[&v], bisector * (thickness / half_sine)));
        }
    }
    out
}

/// Whether every cap face and wall quad of the lift still turns the way it did at thickness
/// nought, with the cap's border at `offsets`: the fold test the clamp bisects against.
fn keeps_turning(lift: &Lift, offsets: &[(VertexId, Vector3<f64>)], reference: &HashMap<FaceId, Vector3<f64>>) -> bool {
    let mut mesh = lift.result.mesh.clone();
    for &(v, o) in offsets {
        let p = mesh.vertex_position(v);
        mesh.set_vertex_position(v, p + o);
    }
    for (&f, &n) in reference {
        let c = mesh.face_vertices(f).map(|v| mesh.vertex_position(v));
        for i in 0..4 {
            let (prev, cur, next) = (c[(i + 3) % 4], c[i], c[(i + 1) % 4]);
            if (next - cur).cross(&(prev - cur)).dot(&n) <= 0.0 {
                return false;
            }
        }
    }
    true
}

/// Lifts and insets: the cap's border moved inward by `thickness` -- clamped where a wall or a
/// cap corner would fold, by bisection -- then the whole cap by `depth` along the region normal.
/// Returns the thickness actually used beside the result, so the tool can say "clamped at".
pub fn inset(mesh: &QuadMesh, faces: &[FaceId], mode: Mode, thickness: f64, depth: f64) -> Result<(EditResult, f64), EditError> {
    let lift = lift(mesh, faces, mode)?;
    for region in &lift.regions {
        if region.border.iter().any(|&(a, b)| mesh.edge_faces(a, b).is_none()) {
            return Err(EditError::refuse(
                "inset can't run along an open edge yet",
                Highlight { edges: region.border.clone(), ..Highlight::default() },
            ));
        }
    }
    // Which way each cap face and wall turns before anything moves: the winding reference. A
    // wall has no area yet, so it takes the normal of the cap face it hangs from -- the inset
    // keeps it in that plane -- and a folded wall is then a quad whose corners turn both ways.
    // (The cap itself cannot be caught by its own turning: inset past the centre reflects the
    // square through a point, which keeps its winding. The walls are where the fold shows.)
    let lifted = &lift.result.mesh;
    let mut reference: HashMap<FaceId, Vector3<f64>> = HashMap::new();
    for region in &lift.regions {
        for &f in &region.faces {
            if let Some(n) = face_normal(lifted, f).try_normalize(1e-18) {
                reference.insert(f, n);
            }
        }
    }
    for &w in &lift.result.new_faces {
        let corners = lifted.face_vertices(w);
        // `(a, b, b', a')`: the cap face holding both duplicates is the one this wall hangs from.
        let (a2, b2) = (corners[2], corners[3]);
        let holder = lift.regions.iter().flat_map(|r| r.faces.iter().copied()).find(|&f| {
            let c = lifted.face_vertices(f);
            c.contains(&a2) && c.contains(&b2)
        });
        if let Some(n) = holder.and_then(|f| reference.get(&f).copied()) {
            reference.insert(w, n);
        }
    }
    let mut used = thickness.max(0.0);
    let mut offsets = inset_offsets(&lift, used);
    if !keeps_turning(&lift, &offsets, &reference) {
        let (mut lo, mut hi) = (0.0, used);
        for _ in 0..20 {
            let mid = (lo + hi) * 0.5;
            if keeps_turning(&lift, &inset_offsets(&lift, mid), &reference) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        used = lo;
        offsets = inset_offsets(&lift, used);
    }
    let mut result = lift.result.clone();
    for (v, o) in offsets {
        let p = result.mesh.vertex_position(v);
        result.mesh.set_vertex_position(v, p + o);
    }
    if depth != 0.0 {
        let pushed = Lift { result: result.clone(), regions: lift.regions.clone() };
        for (v, o) in extrude_offsets(&pushed, Direction::Normal, depth) {
            let p = result.mesh.vertex_position(v);
            result.mesh.set_vertex_position(v, p + o);
        }
    }
    Ok((result, used))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cage;
    use nalgebra::Point3;

    fn f(i: usize) -> FaceId {
        FaceId::from_index(i)
    }

    fn cube() -> QuadMesh {
        Cage::cube_of(1.0).mesh().clone()
    }

    fn unit_cube_top() -> FaceId {
        // +Z is face 1 of `Cage::cube`.
        f(1)
    }

    /// Notes §7.1: one face of a cube, region, Normal 10 mm → 10 faces, 12 vertices, closed,
    /// χ = 2, the cap exactly 10 above, in its old slot.
    #[test]
    fn extruding_one_face_of_a_cube_lifts_a_box_on_it() {
        let m = cube();
        let r = extrude(&m, &[unit_cube_top()], Mode::Region, Direction::Normal, 10.0).unwrap();
        assert_eq!(r.mesh.face_count(), 10);
        assert_eq!(r.mesh.vertex_count(), 12);
        assert_eq!(r.mesh.euler_characteristic(), 2);
        assert!(Cage::try_from_mesh(r.mesh.clone()).is_ok(), "closed");
        for v in r.mesh.face_vertices(unit_cube_top()) {
            assert!((r.mesh.vertex_position(v).z - 10.5).abs() < 1e-12, "the cap at +10 from z = 0.5");
        }
        assert_eq!(r.new_vertices.len(), 4);
        assert_eq!(r.new_faces.len(), 4);
    }

    /// Notes §7.2: two adjacent faces in region mode share no wall; individually they get two.
    #[test]
    fn a_region_of_two_faces_has_no_wall_between_them_and_individual_faces_do() {
        let m = cube();
        let region = extrude(&m, &[f(1), f(5)], Mode::Region, Direction::Normal, 1.0).unwrap();
        assert_eq!(region.mesh.face_count(), 6 + 6, "a border of six edges");
        let each = extrude(&m, &[f(1), f(5)], Mode::Individual, Direction::Normal, 1.0).unwrap();
        assert_eq!(each.mesh.face_count(), 6 + 8, "four walls each");
    }

    /// Notes §7.3 (revised): a region with an interior vertex keeps it on the cap, undoubled.
    #[test]
    fn a_regions_interior_vertex_moves_with_the_cap_and_is_not_duplicated() {
        let m = crate::edit::tests::split(&cube());
        let top: Vec<FaceId> = m.face_ids().filter(|&f| m.face_vertices(f).iter().all(|&v| m.vertex_position(v).z > 0.49)).collect();
        assert_eq!(top.len(), 4, "the top face split in four");
        let centre = m.vertex_ids().find(|&v| (m.vertex_position(v) - Point3::new(0.0, 0.0, 0.5)).norm() < 1e-9).unwrap();
        let r = extrude(&m, &top, Mode::Region, Direction::Normal, 2.0).unwrap();
        assert_eq!(r.new_vertices.len(), 8, "the border only");
        assert!((r.mesh.vertex_position(centre).z - 2.5).abs() < 1e-12, "the centre moved with the cap, at its index");
        assert!(Cage::try_from_mesh(r.mesh.clone()).is_ok());
        assert_eq!(r.mesh.euler_characteristic(), 2);
    }

    /// Notes §7.4 said two faces touching at one corner are refused as a bowtie. **Measured
    /// otherwise, 2026-10-24:** each region duplicates the shared vertex for its own cap, and the
    /// original keeps the two faces outside the regions and the four walls, which is one fan --
    /// two boxes standing on a face, touching at a point, and a valid closed cage. So it is built,
    /// and the refusal stays in the code only for a lift that does make a bowtie.
    #[test]
    fn two_faces_touching_at_a_corner_lift_as_two_boxes() {
        let m = crate::edit::tests::split(&cube());
        let top: Vec<FaceId> = m.face_ids().filter(|&f| m.face_vertices(f).iter().all(|&v| m.vertex_position(v).z > 0.49)).collect();
        let centre = m.vertex_ids().find(|&v| (m.vertex_position(v) - Point3::new(0.0, 0.0, 0.5)).norm() < 1e-9).unwrap();
        let a = top[0];
        let b = top.iter().copied().find(|&g| {
            g != a && {
                let ca = m.face_vertices(a);
                let cb = m.face_vertices(g);
                cb.iter().filter(|v| ca.contains(v)).count() == 1
            }
        }).unwrap();
        let r = extrude(&m, &[a, b], Mode::Region, Direction::Normal, 1.0).unwrap();
        assert!(Cage::try_from_mesh(r.mesh.clone()).is_ok(), "closed and manifold");
        assert_eq!(r.new_vertices.len(), 8, "each region's four border vertices, the centre twice");
        assert_eq!(r.mesh.valence(centre), 6, "the original centre keeps the two other top quads and four walls");
        assert_eq!(r.mesh.euler_characteristic(), 2);
    }

    /// Notes §7.5: along vertex normals a flat region keeps its wall exactly `d`.
    #[test]
    fn along_vertex_normals_a_flat_region_keeps_its_wall_thickness() {
        let m = crate::edit::tests::split(&cube());
        let top: Vec<FaceId> = m.face_ids().filter(|&f| m.face_vertices(f).iter().all(|&v| m.vertex_position(v).z > 0.49)).collect();
        let r = extrude(&m, &top, Mode::Region, Direction::VertexNormals, 3.0).unwrap();
        for &v in &r.new_vertices {
            assert!((r.mesh.vertex_position(v).z - 3.5).abs() < 1e-9);
        }
    }

    /// Notes §7.6: a boundary-edge extrude grows an open strip and keeps it open; an interior edge
    /// is refused.
    #[test]
    fn extruding_a_rim_edge_grows_the_strip_and_an_interior_edge_is_refused() {
        let strip = crate::edit::tests::grid(2, 1, &[]);
        let rim: Vec<(VertexId, VertexId)> = strip.edges().filter(|&(a, b)| strip.edge_faces(a, b).is_none() && strip.vertex_position(a).y > 0.5 && strip.vertex_position(b).y > 0.5).collect();
        assert_eq!(rim.len(), 2, "the top side of a 2×1 strip");
        let r = extrude_boundary_edges(&strip, &rim, 1.0, None).unwrap();
        assert_eq!(r.mesh.face_count(), 4);
        assert_eq!(r.new_vertices.len(), 3, "a chain of two edges shares its middle");
        assert!(r.mesh.is_open());
        for &v in &r.new_vertices {
            assert!((r.mesh.vertex_position(v).y - 2.0).abs() < 1e-9, "outward is +y in the strip's plane");
        }
        let interior = strip.edges().find(|&(a, b)| strip.edge_faces(a, b).is_some()).unwrap();
        assert!(extrude_boundary_edges(&strip, &[interior], 1.0, None).unwrap_err().reason.contains("open edge"));
    }

    /// Notes §7.7: inset 0.2 on a unit cube face → a 0.6 inner face, four coplanar walls; 0.6 →
    /// clamped just under 0.5 with nothing folded.
    #[test]
    fn inset_on_a_unit_cube_face_makes_the_inner_square_and_clamps_before_the_fold() {
        let m = cube();
        let (r, used) = inset(&m, &[unit_cube_top()], Mode::Region, 0.2, 0.0).unwrap();
        assert!((used - 0.2).abs() < 1e-12);
        for v in r.mesh.face_vertices(unit_cube_top()) {
            let p = r.mesh.vertex_position(v);
            assert!((p.x.abs() - 0.3).abs() < 1e-9 && (p.y.abs() - 0.3).abs() < 1e-9 && (p.z - 0.5).abs() < 1e-12, "{p:?}");
        }
        for &w in &r.new_faces {
            assert!(r.mesh.face_vertices(w).iter().all(|&v| (r.mesh.vertex_position(v).z - 0.5).abs() < 1e-12), "a wall lies in the face");
        }
        let (clamped, used) = inset(&m, &[unit_cube_top()], Mode::Region, 0.6, 0.0).unwrap();
        assert!(used < 0.5 && used > 0.49, "clamped at the fold: {used}");
        assert!(clamped.mesh.face_count() == 10);
    }

    /// Notes §7.8: inset with depth equals inset then extrude, vertex for vertex.
    #[test]
    fn inset_with_depth_is_inset_then_extrude() {
        let m = cube();
        let (both, _) = inset(&m, &[unit_cube_top()], Mode::Region, 0.2, 0.7).unwrap();
        let (flat, _) = inset(&m, &[unit_cube_top()], Mode::Region, 0.2, 0.0).unwrap();
        for v in both.mesh.vertex_ids() {
            let expect = flat.mesh.vertex_position(v) + if both.mesh.face_vertices(unit_cube_top()).contains(&v) { Vector3::new(0.0, 0.0, 0.7) } else { Vector3::zeros() };
            assert!((both.mesh.vertex_position(v) - expect).norm() < 1e-12);
        }
    }

    #[test]
    fn inset_along_an_open_edge_is_refused_for_now() {
        let strip = crate::edit::tests::grid(2, 1, &[]);
        let err = inset(&strip, &[f(0)], Mode::Region, 0.1, 0.0).unwrap_err();
        assert!(err.reason.contains("open edge"), "{}", err.reason);
    }
}
