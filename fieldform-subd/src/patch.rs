// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A small **open** quad mesh: the neighbourhood around a face, extracted so the limit-surface
//! evaluator can subdivide *it* instead of the whole model.
//!
//! # Why this type exists
//!
//! [`Cage`] is closed, 2-manifold and all-quad by construction, and that is deliberate — it is
//! the editing representation, and those invariants are what keep the rest of the system
//! simple. But a neighbourhood carved out of a cage necessarily has a boundary, so it cannot
//! be a `Cage`. Hence a second, deliberately minimal mesh type that tolerates boundaries and
//! is never exposed outside this crate.
//!
//! The cost of that is a second implementation of the Catmull-Clark rules, which could drift
//! from [`crate::subdivide`]. That risk is met head-on by
//! `patch_subdivision_matches_cage_subdivision`, which subdivides the same mesh both ways and
//! requires the positions to agree. The two accumulate the same terms in different orders, so
//! they agree to floating-point round-off (1e-12 on unit-scale geometry) rather than
//! bit-for-bit — the rules are identical, the summation order is not.
//!
//! # How boundaries are handled
//!
//! Catmull-Clark is local but not pointwise: a new vertex position depends on its whole
//! 1-ring. Near the patch boundary that ring is missing, so those positions are *unknowable*
//! from the patch alone — not merely approximate. Rather than invent them, subdivision marks
//! them invalid and drops every face that touches one.
//!
//! So a patch **shrinks by one ring per subdivision**, and the geometry that survives is what
//! whole-cage subdivision would have produced, to that same round-off. That is the property the
//! evaluator relies on, and it is pinned by `local_subdivision_matches_global_subdivision`.
//! Extract with enough rings and the face of interest is still there at the depth you need.

use crate::evaluate::QuadTopology;
use fieldform_cage::{FaceId, HalfEdgeId, QuadMesh, VertexId};
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

/// Marks a sub-face subdivision could not produce, because one of its corners sat on the
/// patch boundary where the 1-ring needed to place it was missing.
pub(crate) const DROPPED: u32 = u32::MAX;

pub(crate) struct Patch {
    positions: Vec<Point3<f64>>,
    faces: Vec<[VertexId; 4]>,
    /// One entry per half-edge, indexed `face * 4 + corner`. `None` on the boundary.
    twin: Vec<Option<HalfEdgeId>>,
    /// `None` where the vertex's ring is incomplete, so its valence cannot be known.
    valence: Vec<Option<usize>>,
    vertex_faces: Vec<Vec<FaceId>>,
    /// One flag per half-edge, as `twin`: whether its edge is creased. Copied from the cage on
    /// extraction and handed down by subdivision -- both halves of a creased edge stay creased.
    sharp: Vec<bool>,
    /// Whether any flag is set, so an uncreased patch pays nothing for the rules.
    any_sharp: bool,
    /// Per vertex, the far ends of its creased edges, as far as the patch can see them.
    crease_ends: Vec<Vec<VertexId>>,
    /// Per vertex, whether two creased edges run straight through it (see `QuadTopology`).
    straight: Vec<bool>,
}

/// Half-edge `k` of face `f` runs from corner `k` to corner `k + 1`.
fn half_edge_of(face: usize, corner: usize) -> HalfEdgeId {
    HalfEdgeId::from_index(face * 4 + corner)
}

fn undirected(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

impl Patch {
    fn new(positions: Vec<Point3<f64>>, faces: Vec<[VertexId; 4]>, sharp: Vec<bool>) -> Self {
        debug_assert_eq!(sharp.len(), faces.len() * 4, "one crease flag per half-edge");
        let vertex_count = positions.len();

        // Directed (origin, destination) -> half-edge, so a twin is found by looking up the
        // reversed pair. A patch interior behaves exactly like a cage here; only the boundary
        // half-edges come up missing.
        let mut directed: HashMap<(usize, usize), HalfEdgeId> = HashMap::new();
        for (f, corners) in faces.iter().enumerate() {
            for k in 0..4 {
                let o = corners[k].index();
                let d = corners[(k + 1) % 4].index();
                directed.insert((o, d), half_edge_of(f, k));
            }
        }

        let mut twin = vec![None; faces.len() * 4];
        let mut vertex_faces: Vec<Vec<FaceId>> = vec![Vec::new(); vertex_count];
        for (f, corners) in faces.iter().enumerate() {
            for k in 0..4 {
                let o = corners[k].index();
                let d = corners[(k + 1) % 4].index();
                twin[f * 4 + k] = directed.get(&(d, o)).copied();
            }
            for &c in corners.iter() {
                vertex_faces[c.index()].push(FaceId::from_index(f));
            }
        }

        // A vertex's ring is complete when every half-edge touching it — leaving *and*
        // arriving — has a twin. That is what lets its valence, and so its subdivided
        // position, be computed from the patch alone.
        let mut valence: Vec<Option<usize>> = vec![None; vertex_count];
        for v in 0..vertex_count {
            if vertex_faces[v].is_empty() {
                continue;
            }
            let complete = vertex_faces[v].iter().all(|&f| {
                let corners = &faces[f.index()];
                let k = corners
                    .iter()
                    .position(|&c| c.index() == v)
                    .expect("vertex_faces only lists faces containing this vertex");
                twin[f.index() * 4 + k].is_some() && twin[f.index() * 4 + (k + 3) % 4].is_some()
            });
            if complete {
                // Around an interior vertex of a quad mesh the incident faces, the incident
                // edges and the valence all count the same.
                valence[v] = Some(vertex_faces[v].len());
            }
        }

        let any_sharp = sharp.iter().any(|&s| s);
        let mut crease_ends: Vec<Vec<VertexId>> = vec![Vec::new(); vertex_count];
        let mut straight = vec![false; vertex_count];
        if any_sharp {
            for (f, corners) in faces.iter().enumerate() {
                for k in 0..4 {
                    if sharp[f * 4 + k] {
                        let (a, b) = (corners[k], corners[(k + 1) % 4]);
                        if !crease_ends[a.index()].contains(&b) {
                            crease_ends[a.index()].push(b);
                        }
                        if !crease_ends[b.index()].contains(&a) {
                            crease_ends[b.index()].push(a);
                        }
                    }
                }
            }
            for v in 0..vertex_count {
                let ends = &crease_ends[v];
                // Opposite at a vertex of four faces: no face round it holds both far ends.
                straight[v] = ends.len() == 2
                    && valence[v] == Some(4)
                    && !vertex_faces[v].iter().any(|f| {
                        let c = &faces[f.index()];
                        c.contains(&ends[0]) && c.contains(&ends[1])
                    });
            }
        }

        Self {
            positions,
            faces,
            twin,
            valence,
            vertex_faces,
            sharp,
            any_sharp,
            crease_ends,
            straight,
        }
    }

    /// Extracts the faces within `rings` steps of `seeds`, stepping through shared vertices —
    /// so a "ring" is the full 1-ring of faces, diagonal neighbours included, which is what
    /// the control grid actually reaches for.
    ///
    /// Returns the patch and, for each seed, where it landed in it.
    pub(crate) fn extract(
        cage: &QuadMesh,
        seeds: &[FaceId],
        rings: usize,
    ) -> (Self, HashMap<FaceId, FaceId>) {
        let mut included: HashSet<usize> = seeds.iter().map(|f| f.index()).collect();
        let mut frontier: Vec<usize> = included.iter().copied().collect();

        for _ in 0..rings {
            let mut next = Vec::new();
            for &f in &frontier {
                for v in cage.face_vertices(FaceId::from_index(f)) {
                    for g in cage.adjacent_faces(v) {
                        if included.insert(g.index()) {
                            next.push(g.index());
                        }
                    }
                }
            }
            if next.is_empty() {
                break; // the patch already covers the whole cage
            }
            frontier = next;
        }

        // Deterministic face order: this feeds a performance gate, and a HashSet's iteration
        // order would make face indices — and so cache behaviour and timings — vary run to
        // run for no reason.
        let mut face_list: Vec<usize> = included.into_iter().collect();
        face_list.sort_unstable();

        let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
        let mut positions = Vec::new();
        let mut faces = Vec::with_capacity(face_list.len());
        let mut sharp = Vec::with_capacity(face_list.len() * 4);
        for &f in &face_list {
            // `face_half_edges[k]` leaves corner `k`, as the patch's own half-edge `k` does.
            for he in cage.face_half_edges(FaceId::from_index(f)) {
                sharp.push(cage.is_sharp(he));
            }
            let corners = cage.face_vertices(FaceId::from_index(f));
            let mut local = [VertexId::from_index(0); 4];
            for (k, &c) in corners.iter().enumerate() {
                let id = *vertex_map.entry(c.index()).or_insert_with(|| {
                    positions.push(cage.vertex_position(c));
                    VertexId::from_index(positions.len() - 1)
                });
                local[k] = id;
            }
            faces.push(local);
        }

        let face_map = seeds
            .iter()
            .filter_map(|&s| {
                face_list
                    .binary_search(&s.index())
                    .ok()
                    .map(|i| (s, FaceId::from_index(i)))
            })
            .collect();

        (Self::new(positions, faces, sharp), face_map)
    }

    /// One Catmull-Clark step, dropping whatever the patch is too small to determine.
    ///
    /// Also returns, per parent face, its four sub-face indices in [`crate::evaluate`]'s
    /// descent order (corner `k` first), or [`DROPPED`].
    pub(crate) fn subdivide(&self) -> (Self, Vec<[u32; 4]>) {
        let mut positions = Vec::new();

        // --- updated original vertices, where the ring allows it ---
        let mut vertex_point: Vec<Option<u32>> = vec![None; self.positions.len()];
        for v in 0..self.positions.len() {
            let Some(n) = self.valence[v] else { continue };
            let vid = VertexId::from_index(v);
            let p = self.positions[v];

            let mut face_avg = Vector3::zeros();
            for &f in &self.vertex_faces[v] {
                face_avg += self.face_centroid(f).coords;
            }
            face_avg /= n as f64;

            let neighbours = self.ring_vertices(vid);
            let mut edge_mid_avg = Vector3::zeros();
            for a in &neighbours {
                edge_mid_avg += (p.coords + self.positions[a.index()].coords) / 2.0;
            }
            edge_mid_avg /= neighbours.len() as f64;

            let n_f = n as f64;
            let smooth = (face_avg + edge_mid_avg * 2.0 + p.coords * (n_f - 3.0)) / n_f;
            // The sharp rules: on a crease the vertex follows the crease curve, `(1, 6, 1)/8` of
            // its two crease neighbours; where three or more meet it stays put. A dart -- one
            // creased edge -- is smooth.
            let updated = match self.crease_ends[v].as_slice() {
                [a, b] => (self.positions[a.index()].coords + p.coords * 6.0 + self.positions[b.index()].coords) / 8.0,
                ends if ends.len() >= 3 => p.coords,
                _ => smooth,
            };
            vertex_point[v] = Some(positions.len() as u32);
            positions.push(Point3::from(updated));
        }

        // --- face points: always determinable, since the face is wholly present ---
        let mut face_point = Vec::with_capacity(self.faces.len());
        for f in 0..self.faces.len() {
            face_point.push(positions.len() as u32);
            positions.push(self.face_centroid(FaceId::from_index(f)));
        }

        // --- edge points, only where both adjacent faces are in the patch ---
        let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut creased_edge: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for (f, corners) in self.faces.iter().enumerate() {
            for k in 0..4 {
                let a = corners[k].index();
                let b = corners[(k + 1) % 4].index();
                edge_faces.entry(undirected(a, b)).or_default().push(f);
                if self.sharp[f * 4 + k] {
                    creased_edge.insert(undirected(a, b));
                }
            }
        }
        let mut edge_point: HashMap<(usize, usize), u32> = HashMap::new();
        for (&key, fs) in &edge_faces {
            if fs.len() != 2 {
                continue; // a boundary edge: its point depends on a face we don't have
            }
            let (a, b) = key;
            // A creased edge's point is its midpoint, which reads neither face: that is what
            // makes each side of a crease independent of the other.
            let point = if creased_edge.contains(&key) {
                (self.positions[a].coords + self.positions[b].coords) / 2.0
            } else {
                (self.positions[a].coords
                    + self.positions[b].coords
                    + self.face_centroid(FaceId::from_index(fs[0])).coords
                    + self.face_centroid(FaceId::from_index(fs[1])).coords)
                    / 4.0
            };
            edge_point.insert(key, positions.len() as u32);
            positions.push(Point3::from(point));
        }

        // --- new faces, in the same corner order `crate::subdivide` emits ---
        let mut faces: Vec<[VertexId; 4]> = Vec::new();
        let mut sharp: Vec<bool> = Vec::new();
        let mut children = vec![[DROPPED; 4]; self.faces.len()];
        for (f, corners) in self.faces.iter().enumerate() {
            for k in 0..4 {
                let prev = corners[(k + 3) % 4].index();
                let cur = corners[k].index();
                let next = corners[(k + 1) % 4].index();

                let (Some(vp), Some(&e_cn), Some(&e_pc)) = (
                    vertex_point[cur],
                    edge_point.get(&undirected(cur, next)),
                    edge_point.get(&undirected(prev, cur)),
                ) else {
                    continue;
                };

                children[f][k] = faces.len() as u32;
                faces.push([
                    VertexId::from_index(vp as usize),
                    VertexId::from_index(e_cn as usize),
                    VertexId::from_index(face_point[f] as usize),
                    VertexId::from_index(e_pc as usize),
                ]);
                // The child's first and last edges are halves of the parent's edges at this
                // corner and keep their creases; the two inner edges are new and smooth.
                sharp.extend([self.sharp[f * 4 + k], false, false, self.sharp[f * 4 + (k + 3) % 4]]);
            }
        }

        // Positions no surviving face references are dead weight, and this runs once per level
        // per patch, so compact rather than carry them down.
        let mut used: Vec<Option<VertexId>> = vec![None; positions.len()];
        let mut compact = Vec::new();
        for corners in faces.iter_mut() {
            for c in corners.iter_mut() {
                let slot = &mut used[c.index()];
                let id = *slot.get_or_insert_with(|| {
                    compact.push(positions[c.index()]);
                    VertexId::from_index(compact.len() - 1)
                });
                *c = id;
            }
        }

        (Self::new(compact, faces, sharp), children)
    }

    /// The vertices sharing an edge with `v`, each once.
    fn ring_vertices(&self, v: VertexId) -> Vec<VertexId> {
        let mut out = Vec::new();
        for &f in &self.vertex_faces[v.index()] {
            let corners = self.faces[f.index()];
            let k = corners
                .iter()
                .position(|&c| c == v)
                .expect("vertex_faces only lists faces containing this vertex");
            for n in [corners[(k + 1) % 4], corners[(k + 3) % 4]] {
                if !out.contains(&n) {
                    out.push(n);
                }
            }
        }
        out
    }

    fn face_centroid(&self, f: FaceId) -> Point3<f64> {
        let corners = self.faces[f.index()];
        let sum = corners.iter().fold(Vector3::zeros(), |acc, &c| {
            acc + self.positions[c.index()].coords
        });
        Point3::from(sum / 4.0)
    }

    // Read only by the tests and by `LimitSurface::cached_faces`, which is itself test-only;
    // left in a normal build it would be dead code.
    #[cfg(test)]
    pub(crate) fn face_count(&self) -> usize {
        self.faces.len()
    }
}

impl QuadTopology for Patch {
    fn face_vertices(&self, face: FaceId) -> [VertexId; 4] {
        self.faces[face.index()]
    }
    fn face_half_edges(&self, face: FaceId) -> [HalfEdgeId; 4] {
        [
            half_edge_of(face.index(), 0),
            half_edge_of(face.index(), 1),
            half_edge_of(face.index(), 2),
            half_edge_of(face.index(), 3),
        ]
    }
    fn origin(&self, he: HalfEdgeId) -> VertexId {
        self.faces[he.index() / 4][he.index() % 4]
    }
    fn next(&self, he: HalfEdgeId) -> HalfEdgeId {
        half_edge_of(he.index() / 4, (he.index() % 4 + 1) % 4)
    }
    fn twin(&self, he: HalfEdgeId) -> Option<HalfEdgeId> {
        self.twin[he.index()]
    }
    fn valence(&self, v: VertexId) -> Option<usize> {
        self.valence[v.index()]
    }
    fn vertex_position(&self, v: VertexId) -> Point3<f64> {
        self.positions[v.index()]
    }
    fn adjacent_faces(&self, v: VertexId) -> Vec<FaceId> {
        self.vertex_faces[v.index()].clone()
    }
    fn adjacent_vertices(&self, v: VertexId) -> Vec<VertexId> {
        self.ring_vertices(v)
    }
    fn face_of(&self, he: HalfEdgeId) -> FaceId {
        FaceId::from_index(he.index() / 4)
    }
    fn has_sharp(&self) -> bool {
        self.any_sharp
    }
    fn is_sharp(&self, he: HalfEdgeId) -> bool {
        self.sharp[he.index()]
    }
    fn crease_class(&self, v: VertexId) -> (usize, bool) {
        (self.crease_ends[v.index()].len(), self.straight[v.index()])
    }
    fn crease_ends(&self, v: VertexId) -> Vec<VertexId> {
        self.crease_ends[v.index()].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate::regular_grid;
    use fieldform_cage::Cage;
    use crate::{subdivide, subdivide_n};

    /// Tight enough to catch a wrong rule, loose enough to allow reassociated summation:
    /// the two implementations accumulate the same terms in different orders, so they agree
    /// to round-off rather than bit-for-bit.
    const EPS: f64 = 1e-12;

    fn assert_close(a: Point3<f64>, b: Point3<f64>, what: &str) {
        assert!(
            (a - b).norm() < EPS,
            "{what}: {a:?} vs {b:?} (delta {})",
            (a - b).norm()
        );
    }

    fn all_faces(cage: &Cage) -> Vec<FaceId> {
        cage.face_ids().collect()
    }

    /// The duplicated-math guard. Given the *whole* closed cage as a patch — so nothing is
    /// dropped for want of a ring — the patch rules must reproduce `crate::subdivide`
    /// exactly, sub-face for sub-face and corner for corner.
    #[test]
    fn patch_subdivision_matches_cage_subdivision() {
        let cage = subdivide_n(&Cage::cube(), 2).unwrap();
        let (patch, _) = Patch::extract(&cage, &all_faces(&cage), 0);
        assert_eq!(patch.face_count(), cage.face_count(), "patch lost faces");

        let global = subdivide(&cage).unwrap();
        let (local, children) = patch.subdivide();
        assert_eq!(local.face_count(), global.face_count());

        for f in 0..cage.face_count() {
            for k in 0..4 {
                let child = children[f][k];
                assert_ne!(child, DROPPED, "closed patch should drop nothing");

                // `subdivide` emits face f's sub-faces at 4f..4f+4, corner k first.
                let g = FaceId::from_index(f * 4 + k);
                let expected = global.face_vertices(g);
                let actual = local.faces[child as usize];
                for c in 0..4 {
                    assert_close(
                        local.positions[actual[c].index()],
                        global.vertex_position(expected[c]),
                        &format!("face {f} corner {k} vertex {c}"),
                    );
                }
            }
        }
    }

    /// **The same guard with creases on**: darts, a straight run, turns, a pole and a junction.
    /// The patch's sharp rules must reproduce `crate::subdivide`'s, position for position and
    /// crease for crease, or the descent and the refine would disagree about a creased surface.
    #[test]
    fn a_creased_patch_subdivides_as_the_creased_cage_does() {
        let cage = crate::crease_tests::every_kind_of_crease();
        let (patch, _) = Patch::extract(&cage, &all_faces(&cage), 0);
        let global = subdivide(&cage).unwrap();
        let (local, children) = patch.subdivide();
        let mut creased_children = 0;
        for (f, kids) in children.iter().enumerate().take(cage.face_count()) {
            for (k, &child) in kids.iter().enumerate() {
                let child = child as usize;
                let g = FaceId::from_index(f * 4 + k);
                let expected = global.face_vertices(g);
                let actual = local.faces[child];
                for c in 0..4 {
                    assert_close(
                        local.positions[actual[c].index()],
                        global.vertex_position(expected[c]),
                        &format!("face {f} corner {k} vertex {c}"),
                    );
                    let sharp = local.sharp[child * 4 + c];
                    assert_eq!(sharp, global.is_creased(expected[c], expected[(c + 1) % 4]), "face {f} corner {k} edge {c}");
                    creased_children += sharp as usize;
                }
            }
        }
        assert!(creased_children > 0, "no crease reached the children");
    }

    /// The property the whole optimisation rests on: subdividing a *neighbourhood* gives the
    /// same geometry as subdividing the whole model, for every face that survives. If this
    /// holds, local descent is not an approximation of global descent — it is the same
    /// answer, computed over less mesh.
    #[test]
    fn local_subdivision_matches_global_subdivision() {
        let cage = subdivide_n(&Cage::cube(), 2).unwrap();
        let depth = 3;

        // An irregular face, which is the only kind that ever needs descending.
        let seed = cage
            .face_ids()
            .find(|&f| {
                cage.mesh()
                    .face_vertices(f)
                    .iter()
                    .any(|&v| cage.mesh().valence(v) != 4)
            })
            .expect("a subdivided cube keeps its valence-3 corners");

        let (patch, map) = Patch::extract(&cage, &[seed], depth + 2);
        assert!(
            patch.face_count() < cage.face_count(),
            "the neighbourhood should be smaller than the cage, or this proves nothing \
             (patch {} vs cage {})",
            patch.face_count(),
            cage.face_count()
        );

        let mut levels = vec![patch];
        let mut child_maps = Vec::new();
        for _ in 0..depth {
            let (next, children) = levels.last().unwrap().subdivide();
            levels.push(next);
            child_maps.push(children);
        }

        let mut global = cage.clone();
        for _ in 0..depth {
            global = subdivide(&global).unwrap();
        }

        // Walk every descent path from the seed and check the ones the patch still has.
        let mut checked = 0;
        let mut paths = vec![(map[&seed].index(), seed.index())];
        for level in 0..depth {
            let mut next = Vec::new();
            for &(p, g) in &paths {
                for k in 0..4 {
                    let child = child_maps[level][p][k];
                    if child != DROPPED {
                        next.push((child as usize, g * 4 + k));
                    }
                }
            }
            paths = next;
        }

        for (p, g) in paths {
            let local_face = levels[depth].faces[p];
            let global_face = global.face_vertices(FaceId::from_index(g));
            for c in 0..4 {
                assert_close(
                    levels[depth].positions[local_face[c].index()],
                    global.vertex_position(global_face[c]),
                    &format!("descended face {g} corner {c}"),
                );
            }
            checked += 1;
        }

        assert_eq!(
            checked,
            4usize.pow(depth as u32),
            "padding of {} rings should have kept every descendant of the seed alive",
            depth + 2
        );
    }

    /// The control-grid walk is orientation sensitive and shared between `Cage` and `Patch`
    /// through `QuadMesh`. If the patch's half-edge wiring were subtly wrong — twins crossed,
    /// winding reversed — the grid would still be built, just from the wrong points. So
    /// compare it against the cage's own, point for point.
    #[test]
    fn patch_control_grid_matches_the_cages() {
        let cage = subdivide_n(&Cage::cube(), 3).unwrap();
        let (patch, map) = Patch::extract(&cage, &all_faces(&cage), 0);

        let mut compared = 0;
        for f in cage.face_ids() {
            let Some(expected) = regular_grid(&cage, f) else {
                continue; // irregular in the cage: nothing to compare
            };
            let actual = regular_grid(&patch, map[&f]).expect("a closed patch matches the cage");
            for i in 0..4 {
                for j in 0..4 {
                    assert_close(actual[i][j], expected[i][j], &format!("face {f:?} g[{i}][{j}]"));
                }
            }
            compared += 1;
        }
        assert!(compared > 0, "no regular faces were compared");
    }

    /// A patch must shrink by exactly one ring per subdivision — that is the accounting the
    /// evaluator's padding depends on. Too little shrinkage would mean invented geometry;
    /// too much would mean the seed vanishes before the descent finishes.
    #[test]
    fn a_patch_shrinks_by_one_ring_per_subdivision() {
        let cage = subdivide_n(&Cage::cube(), 3).unwrap();
        let seed = FaceId::from_index(0);

        for rings in 1..=4 {
            let (patch, map) = Patch::extract(&cage, &[seed], rings);
            let mut current = patch;
            let mut face = map[&seed].index();

            // The seed's descendants must survive exactly `rings - 1` further levels: one
            // ring of padding is consumed per subdivision, and the last ring is the one the
            // control grid itself needs.
            for level in 0..rings {
                let (next, children) = current.subdivide();
                let child = children[face][0];
                assert_ne!(
                    child, DROPPED,
                    "seed's sub-face vanished at level {level} with {rings} rings of padding"
                );
                face = child as usize;
                current = next;
            }
        }
    }
}
