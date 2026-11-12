// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Creased edges: where the surface folds sharply instead of rounding over.
//!
//! **A crease is a property of an edge, stored on the mesh, and nothing else.** The subdivision
//! rules that make it sharp live in the subdivision crate; what is here is which edges are creased
//! and the questions those rules ask -- is this half-edge sharp, how many creased edges meet at
//! this vertex, do two of them run straight through it -- answered from an index rebuilt whenever
//! the set changes, because the evaluator asks them once per surface sample.
//!
//! The set is keyed by vertex pair rather than by half-edge, so it survives everything that keeps
//! vertex numbers -- moving vertices, flipping a shell, an edit that appends -- and an edit that
//! renumbers says how through [`QuadMesh::carry_creases`].

use crate::{HalfEdgeId, QuadMesh, VertexId};

/// The derived half of the crease data: what the evaluator reads per sample.
#[derive(Debug, Clone, Default)]
pub(crate) struct CreaseIndex {
    /// One flag per half-edge; both halves of a creased edge are set. Empty while nothing is
    /// creased, which is the test [`QuadMesh::has_creases`] makes cheap.
    sharp: Vec<bool>,
    /// Per vertex, how many creased edges meet there.
    count: Vec<u8>,
    /// Per vertex, whether exactly two creased edges run straight through it: an interior vertex
    /// of four faces whose two creased edges are opposite, not side by side. The one creased
    /// vertex a bicubic patch can still describe.
    straight: Vec<bool>,
}

/// Creases asked of edges the mesh does not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreaseError {
    pub missing: Vec<(VertexId, VertexId)>,
}

impl std::fmt::Display for CreaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = if self.missing.len() == 1 { "an edge" } else { "edges" };
        let list: Vec<String> = self.missing.iter().map(|(a, b)| format!("{}-{}", a.index(), b.index())).collect();
        write!(f, "a crease names {what} the mesh does not have: {}", list.join(", "))
    }
}

impl std::error::Error for CreaseError {}

fn ordered(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a <= b { (a, b) } else { (b, a) }
}

impl QuadMesh {
    /// Whether any edge is creased.
    pub fn has_creases(&self) -> bool {
        !self.creases.is_empty()
    }

    /// How many edges are creased.
    pub fn crease_count(&self) -> usize {
        self.creases.len()
    }

    /// Every creased edge once, as (lower, higher) vertex, in a fixed order.
    pub fn creases(&self) -> impl Iterator<Item = (VertexId, VertexId)> + '_ {
        self.creases.iter().copied()
    }

    /// Whether the edge between `a` and `b` is creased, either way round.
    pub fn is_creased(&self, a: VertexId, b: VertexId) -> bool {
        self.creases.contains(&ordered(a, b))
    }

    /// Whether this half-edge's edge is creased -- the per-sample question, answered by index.
    pub fn is_sharp(&self, he: HalfEdgeId) -> bool {
        self.crease_index.sharp.get(he.index()).copied().unwrap_or(false)
    }

    /// How many creased edges meet at `v`.
    pub fn creases_at(&self, v: VertexId) -> usize {
        self.crease_index.count.get(v.index()).copied().unwrap_or(0) as usize
    }

    /// Whether exactly two creased edges run straight through `v` -- opposite each other at an
    /// interior vertex of four faces.
    pub fn crease_runs_straight(&self, v: VertexId) -> bool {
        self.crease_index.straight.get(v.index()).copied().unwrap_or(false)
    }

    /// The far ends of the creased edges at `v`.
    pub fn creased_neighbours(&self, v: VertexId) -> Vec<VertexId> {
        if self.creases_at(v) == 0 {
            return Vec::new();
        }
        self.adjacent_vertices(v).into_iter().filter(|&n| self.is_creased(v, n)).collect()
    }

    /// Creases or uncreases the edge between `a` and `b`. `None` when there is no such edge;
    /// otherwise whether anything changed.
    pub fn set_creased(&mut self, a: VertexId, b: VertexId, creased: bool) -> Option<bool> {
        if a == b || a.index() >= self.vertex_count() || b.index() >= self.vertex_count() {
            return None;
        }
        if !self.adjacent_vertices(a).contains(&b) {
            return None;
        }
        let key = ordered(a, b);
        let changed = if creased { self.creases.insert(key) } else { self.creases.remove(&key) };
        if changed {
            self.rebuild_crease_index();
        }
        Some(changed)
    }

    /// Creases every listed edge, refusing -- with every offender named -- if any is not an edge
    /// of this mesh. A file is read through this, so a crease on an edge that does not exist
    /// refuses the document rather than being dropped.
    pub fn apply_creases(&mut self, edges: &[(VertexId, VertexId)]) -> Result<(), CreaseError> {
        let missing: Vec<(VertexId, VertexId)> = edges
            .iter()
            .copied()
            .filter(|&(a, b)| {
                a == b
                    || a.index() >= self.vertex_count()
                    || b.index() >= self.vertex_count()
                    || !self.adjacent_vertices(a).contains(&b)
            })
            .collect();
        if !missing.is_empty() {
            return Err(CreaseError { missing });
        }
        for &(a, b) in edges {
            self.creases.insert(ordered(a, b));
        }
        self.rebuild_crease_index();
        Ok(())
    }

    /// Takes every crease off.
    pub fn clear_creases(&mut self) {
        self.creases.clear();
        self.rebuild_crease_index();
    }

    fn rebuild_crease_index(&mut self) {
        if self.creases.is_empty() {
            self.crease_index = CreaseIndex::default();
            return;
        }
        let mut sharp = vec![false; self.half_edges.len()];
        for (i, flag) in sharp.iter_mut().enumerate() {
            let he = &self.half_edges[i];
            let dest = self.half_edges[he.next.index()].origin;
            *flag = self.creases.contains(&ordered(he.origin, dest));
        }
        let mut count = vec![0u8; self.vertices.len()];
        for &(a, b) in &self.creases {
            count[a.index()] = count[a.index()].saturating_add(1);
            count[b.index()] = count[b.index()].saturating_add(1);
        }
        let mut straight = vec![false; self.vertices.len()];
        for (v, flag) in straight.iter_mut().enumerate() {
            if count[v] != 2 {
                continue;
            }
            let v = VertexId::new(v);
            if self.valence(v) != 4 || self.vertex_on_boundary(v) {
                continue;
            }
            let ends: Vec<VertexId> =
                self.adjacent_vertices(v).into_iter().filter(|&n| self.creases.contains(&ordered(v, n))).collect();
            // Side by side, the two edges are two sides of one face at `v`; opposite, no face at
            // `v` holds both far ends.
            *flag = ends.len() == 2
                && !self.adjacent_faces(v).into_iter().any(|f| {
                    let c = self.face_vertices(f);
                    c.contains(&ends[0]) && c.contains(&ends[1])
                });
        }
        self.crease_index = CreaseIndex { sharp, count, straight };
    }

    /// Lays `old`'s creases onto this mesh, which an edit built from it.
    ///
    /// **Every edit rebuilds its mesh from a face list, which knows nothing of creases**, so each
    /// one calls this with what happened to the vertices: `image` gives, for an old vertex, the
    /// vertices of this mesh it may now be, most likely first. A creased edge lands on the first
    /// pair that is an edge here. Where none is -- the edge was split -- and `split` is set, it
    /// lands on the run of new vertices (those numbered past `old`'s last) lying along the old
    /// edge between its ends, so both halves of a split crease are creased. An edge whose ends
    /// went, or met, has no crease left to keep.
    pub(crate) fn carry_creases(&mut self, old: &QuadMesh, image: impl Fn(VertexId) -> Vec<VertexId>, split: bool) {
        if !old.has_creases() {
            return;
        }
        let mut kept: Vec<(VertexId, VertexId)> = Vec::new();
        for (a, b) in old.creases() {
            let (ia, ib) = (image(a), image(b));
            let direct = ia.iter().flat_map(|&x| ib.iter().map(move |&y| (x, y))).find(|&(x, y)| {
                x != y && x.index() < self.vertex_count() && y.index() < self.vertex_count() && self.adjacent_vertices(x).contains(&y)
            });
            if let Some(edge) = direct {
                kept.push(edge);
                continue;
            }
            if !split {
                continue;
            }
            if let (Some(&x), Some(&y)) = (ia.first(), ib.first())
                && let Some(run) = self.run_along(x, y, old.vertex_count())
            {
                kept.extend(run);
            }
        }
        for (a, b) in kept {
            self.creases.insert(ordered(a, b));
        }
        self.rebuild_crease_index();
    }

    /// The edges from `from` to `to` through vertices numbered `first_new` or later that lie on
    /// the straight segment between them, in order -- an edge as a split left it.
    fn run_along(&self, from: VertexId, to: VertexId, first_new: usize) -> Option<Vec<(VertexId, VertexId)>> {
        let (p, q) = (self.vertex_position(from), self.vertex_position(to));
        let d = q - p;
        let length_sq = d.norm_squared();
        if length_sq <= 0.0 {
            return None;
        }
        let tolerance = 1e-9 * length_sq.sqrt().max(1.0);
        let along = |v: VertexId| -> Option<f64> {
            let r = self.vertex_position(v) - p;
            let t = r.dot(&d) / length_sq;
            ((r - d * t).norm() <= tolerance).then_some(t)
        };
        let mut run = Vec::new();
        let (mut at, mut t_at) = (from, 0.0);
        for _ in 0..=self.vertex_count() {
            let neighbours = self.adjacent_vertices(at);
            if neighbours.contains(&to) && !run.is_empty() {
                run.push((at, to));
                return Some(run);
            }
            let next = neighbours
                .into_iter()
                .filter(|&n| n.index() >= first_new)
                .filter_map(|n| along(n).map(|t| (n, t)))
                .filter(|&(_, t)| t > t_at && t < 1.0)
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))?;
            run.push((at, next.0));
            at = next.0;
            t_at = next.1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cage, MeshSnapshot, QuadMesh, VertexId};

    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }

    #[test]
    fn a_crease_is_an_edge_either_way_round_and_only_an_edge() {
        let mut cube = Cage::cube();
        assert_eq!(cube.set_creased(v(1), v(0), true), Some(true));
        assert!(cube.is_creased(v(0), v(1)) && cube.is_creased(v(1), v(0)));
        assert_eq!(cube.set_creased(v(0), v(1), true), Some(false), "already creased");
        assert_eq!(cube.set_creased(v(0), v(6), true), None, "a diagonal is not an edge");
        assert_eq!(cube.crease_count(), 1);
        assert_eq!(cube.creases_at(v(0)), 1);
        assert_eq!(cube.set_creased(v(0), v(1), false), Some(true));
        assert!(!cube.has_creases());
    }

    #[test]
    fn both_halves_of_a_creased_edge_are_sharp_and_no_other_half_edge_is() {
        let mut cube = Cage::cube();
        cube.set_creased(v(0), v(1), true);
        let sharp: Vec<(VertexId, VertexId)> = (0..cube.half_edge_count())
            .map(crate::HalfEdgeId::from_index)
            .filter(|&he| cube.is_sharp(he))
            .map(|he| (cube.half_edge(he).origin, cube.destination_of(he)))
            .collect();
        assert_eq!(sharp.len(), 2);
        assert!(sharp.contains(&(v(0), v(1))) && sharp.contains(&(v(1), v(0))));
    }

    #[test]
    fn two_creased_edges_run_straight_only_when_they_are_opposite() {
        // A 2x2 grid of quads: its centre is an interior vertex of four faces, which is all the
        // question needs.
        let positions: Vec<nalgebra::Point3<f64>> =
            (0..9).map(|i| nalgebra::Point3::new((i % 3) as f64, (i / 3) as f64, 0.0)).collect();
        let quads = vec![vec![0, 1, 4, 3], vec![1, 2, 5, 4], vec![3, 4, 7, 6], vec![4, 5, 8, 7]];
        let mut grid = QuadMesh::from_faces(positions, &quads).unwrap();
        assert!(!grid.vertex_on_boundary(v(4)));
        grid.set_creased(v(3), v(4), true);
        grid.set_creased(v(4), v(5), true);
        assert!(grid.crease_runs_straight(v(4)), "3-4-5 is a straight line through the centre");
        grid.set_creased(v(4), v(5), false);
        grid.set_creased(v(4), v(7), true);
        assert!(!grid.crease_runs_straight(v(4)), "3-4-7 turns a corner at the centre");
    }

    #[test]
    fn a_snapshot_keeps_its_creases_and_an_uncreased_one_writes_nothing_new() {
        let plain = serde_json::to_value(MeshSnapshot::of(&Cage::cube())).unwrap();
        assert!(plain.get("creases").is_none(), "an uncreased cage's file is what it always was");

        let mut cube = Cage::cube();
        cube.set_creased(v(0), v(1), true);
        cube.set_creased(v(5), v(6), true);
        let snapshot = MeshSnapshot::of(&cube);
        let json = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(json["creases"], serde_json::json!([{ "edge": [0, 1] }, { "edge": [5, 6] }]));
        let back = snapshot.restore();
        assert_eq!(back.creases().collect::<Vec<_>>(), vec![(v(0), v(1)), (v(5), v(6))]);

        let read: Cage = serde_json::from_value(json).unwrap();
        assert!(read.is_creased(v(6), v(5)));
    }

    #[test]
    fn a_file_creasing_an_edge_the_mesh_lacks_is_refused_by_name() {
        let mut json = serde_json::to_value(MeshSnapshot::of(&Cage::cube())).unwrap();
        json["creases"] = serde_json::json!([{ "edge": [0, 6] }, { "edge": [0, 1] }]);
        let err = serde_json::from_value::<Cage>(json).unwrap_err().to_string();
        assert!(err.contains("0-6"), "{err}");
    }

    #[test]
    fn a_field_this_build_does_not_know_on_a_crease_is_refused_rather_than_dropped() {
        let mut json = serde_json::to_value(MeshSnapshot::of(&Cage::cube())).unwrap();
        json["creases"] = serde_json::json!([{ "edge": [0, 1], "radius": 2.0 }]);
        assert!(serde_json::from_value::<Cage>(json).is_err());
    }

    #[test]
    fn creasing_changes_the_fingerprint_and_an_uncreased_mesh_keeps_its_own() {
        let cube = Cage::cube();
        let before = cube.geometry_fingerprint();
        let mut creased = cube.clone();
        creased.set_creased(v(0), v(1), true);
        assert_ne!(creased.geometry_fingerprint(), before);
        creased.set_creased(v(0), v(1), false);
        assert_eq!(creased.geometry_fingerprint(), before);
    }
}
