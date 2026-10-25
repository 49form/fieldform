// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Topology edits on a [`QuadMesh`]: the primitives the cage tools are built from
//! (the cage-editing notes §3).
//!
//! **Every primitive rewrites the face list and rebuilds.** It reads the positions and quads,
//! computes new ones, and hands them to [`QuadMesh::from_faces`] — never surgery on the half-edges
//! in place, which is where twin pointers and boundary fan starts go silently wrong. That makes the
//! constructor the one place the invariants are checked, so a primitive cannot leave a mesh the
//! constructor would refuse: if the rebuild fails, the primitive has a bug, and it says so rather
//! than handing back a half-edited mesh.
//!
//! Each works on any `QuadMesh`, open or closed, so the same code edits a cage and a surface to
//! follow (R15). What a cage additionally needs — closure — is asked afterwards, by
//! [`crate::Cage::try_from_mesh`].
//!
//! The knife's per-face patterns, extrude and inset are items 76 and 77 and live beside these when
//! they are built; insert ring, delete, fill hole, bridge and weld are item 74's.

use crate::{CageError, FaceId, HalfEdgeId, QuadMesh, VertexId};
use nalgebra::Point3;
use std::collections::HashMap;

/// What a primitive made, so a tool can select it or chain from it (spec §3.2).
///
/// **Existing vertex indices never change** except in a primitive that removes vertices, which
/// returns `vertex_map`. New vertices are appended. A face that is split keeps its slot for the
/// first of its replacements and the rest are appended, so an untouched face keeps its index unless
/// the primitive removes faces, which returns `face_map`.
#[derive(Debug, Clone)]
pub struct EditResult {
    pub mesh: QuadMesh,
    pub new_vertices: Vec<VertexId>,
    pub new_faces: Vec<FaceId>,
    /// The faces of the result that replaced each input face that was split.
    pub replaced: HashMap<FaceId, Vec<FaceId>>,
    /// Old index → new, for a primitive that removes vertices (weld, delete). A welded vertex maps
    /// to the vertex it became, a deleted one to `None`.
    pub vertex_map: Option<Vec<Option<VertexId>>>,
    /// Old index → new, for a primitive that removes faces (delete). Not in the spec's sketch of
    /// this type, which had no primitive that removes faces in mind; delete is one, and the
    /// selection after it needs to know where the surviving faces went.
    pub face_map: Option<Vec<Option<FaceId>>>,
}

/// The elements a refusal points at, in the indices of the mesh it was asked of — the one on screen.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Highlight {
    pub vertices: Vec<VertexId>,
    pub edges: Vec<(VertexId, VertexId)>,
    pub faces: Vec<FaceId>,
}

/// Why a primitive did nothing, said to the user, with what to show red.
///
/// `cause` is set only when a rebuild failed, which is a bug in the primitive rather than a
/// refusal; the ops layer reports that as `OpError::Failed`.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
#[error("{reason}")]
pub struct EditError {
    pub reason: String,
    pub highlight: Highlight,
    pub cause: Option<CageError>,
}

impl EditError {
    fn refuse(reason: impl Into<String>, highlight: Highlight) -> Self {
        Self { reason: reason.into(), highlight, cause: None }
    }

    fn bug(what: &str, cause: CageError) -> Self {
        Self {
            reason: format!("{what} produced a mesh that is not valid ({cause}) — this is a bug"),
            highlight: Highlight::default(),
            cause: Some(cause),
        }
    }
}

impl QuadMesh {
    fn next(&self, he: HalfEdgeId) -> HalfEdgeId {
        self.half_edges[he.index()].next
    }

    fn twin(&self, he: HalfEdgeId) -> HalfEdgeId {
        self.half_edges[he.index()].twin
    }

    fn origin(&self, he: HalfEdgeId) -> VertexId {
        self.half_edges[he.index()].origin
    }

    fn destination(&self, he: HalfEdgeId) -> VertexId {
        self.origin(self.next(he))
    }

    /// The half-edge running `a → b`, if a face has one.
    fn directed(&self, a: VertexId, b: VertexId) -> Option<HalfEdgeId> {
        self.half_edges_around_vertex(a)
            .into_iter()
            .find(|&he| self.destination(he) == b)
    }

    /// Every boundary half-edge, keyed by the vertex it leaves. One per boundary vertex, because a
    /// mesh `from_faces` accepted has no bowtie.
    fn boundary_by_origin(&self) -> HashMap<VertexId, HalfEdgeId> {
        (0..self.half_edges.len())
            .map(HalfEdgeId::new)
            .filter(|&he| self.is_boundary(he))
            .map(|he| (self.origin(he), he))
            .collect()
    }

    fn quad_lists(&self) -> Vec<Vec<u32>> {
        self.face_ids()
            .map(|f| self.face_vertices(f).iter().map(|v| v.index() as u32).collect())
            .collect()
    }

    fn position_list(&self) -> Vec<Point3<f64>> {
        self.vertices.iter().map(|v| v.position).collect()
    }
}

fn lerp(a: Point3<f64>, b: Point3<f64>, t: f64) -> Point3<f64> {
    a + (b - a) * t
}

fn rebuild(what: &str, positions: Vec<Point3<f64>>, faces: &[Vec<u32>]) -> Result<QuadMesh, EditError> {
    QuadMesh::from_faces(positions, faces).map_err(|e| EditError::bug(what, e))
}

fn appended_vertices(from: usize, to: usize) -> Vec<VertexId> {
    (from..to).map(VertexId::new).collect()
}

fn appended_faces(from: usize, to: usize) -> Vec<FaceId> {
    (from..to).map(FaceId::new).collect()
}

/// Splits every face of the edge ring through `edge` into two quads, with the new loop crossing
/// each ring edge at `t` (spec §3.3).
///
/// `t` is measured from `edge.0` on the edge named, and carried round the ring so the loop stays
/// straight: in a face entered through `u → v` at `s` from `u`, the loop leaves through the
/// opposite edge at `s` from the corner next to `u`, which is where the next face's entry edge
/// starts. So the reading is the same left-to-right one at every edge of the ring, twisted strip or
/// not.
pub fn insert_ring(mesh: &QuadMesh, edge: (VertexId, VertexId), t: f64) -> Result<EditResult, EditError> {
    let (a, b) = edge;
    let edge_highlight = Highlight { edges: vec![edge], ..Highlight::default() };
    if !(t > 0.0 && t < 1.0) {
        return Err(EditError::refuse(
            "a loop has to cross the edge between its ends, not at one",
            edge_highlight,
        ));
    }
    let (start, s) = match (mesh.directed(a, b), mesh.directed(b, a)) {
        (Some(he), _) => (he, t),
        (None, Some(he)) => (he, 1.0 - t),
        (None, None) => {
            return Err(EditError::refuse("no edge joins those two vertices", edge_highlight));
        }
    };

    // The faces of the ring, each with the half-edge the loop enters it through and how far along
    // that half-edge from its origin. Forward first; then, unless the ring closed, backward from
    // the other side of the starting edge.
    let limit = mesh.faces.len() * 2 + 2;
    let mut ring: Vec<(HalfEdgeId, f64)> = vec![(start, s)];
    let mut closed = false;
    let mut entry = start;
    loop {
        let opposite = mesh.next(mesh.next(entry));
        if mesh.is_boundary(opposite) {
            break;
        }
        entry = mesh.twin(opposite);
        if entry == start {
            closed = true;
            break;
        }
        ring.push((entry, s));
        if ring.len() > limit {
            return Err(EditError::bug("insert_ring", CageError::Empty));
        }
    }
    if !closed && !mesh.is_boundary(start) {
        let mut entry = mesh.twin(start);
        loop {
            ring.push((entry, 1.0 - s));
            let opposite = mesh.next(mesh.next(entry));
            if mesh.is_boundary(opposite) {
                break;
            }
            entry = mesh.twin(opposite);
            if ring.len() > limit {
                return Err(EditError::bug("insert_ring", CageError::Empty));
            }
        }
    }

    // A ring that comes back through a face it already split — through a pole, folding on itself —
    // would need that face split twice in two directions. Refused, naming the face.
    let mut seen = std::collections::HashSet::new();
    for &(he, _) in &ring {
        let face = mesh.half_edges[he.index()].face;
        if !seen.insert(face) {
            return Err(EditError::refuse(
                "this loop comes back through a face it has already crossed — insert it on another edge",
                Highlight { faces: vec![face], ..Highlight::default() },
            ));
        }
    }

    let mut positions = mesh.position_list();
    let mut faces = mesh.quad_lists();
    let original_vertices = positions.len();
    let original_faces = faces.len();
    let mut on_edge: HashMap<(u32, u32), u32> = HashMap::new();
    let mut point = |from: VertexId, to: VertexId, s: f64, positions: &mut Vec<Point3<f64>>| -> u32 {
        let (i, j) = (from.index() as u32, to.index() as u32);
        *on_edge.entry((i.min(j), i.max(j))).or_insert_with(|| {
            positions.push(lerp(positions[from.index()], positions[to.index()], s));
            (positions.len() - 1) as u32
        })
    };

    let mut replaced = HashMap::new();
    for &(entry, s) in &ring {
        // Face (u, v, w, x), entered through u → v; it leaves through w → x, at s from x.
        let (u, v) = (mesh.origin(entry), mesh.destination(entry));
        let w = mesh.destination(mesh.next(entry));
        let x = mesh.destination(mesh.next(mesh.next(entry)));
        let p = point(u, v, s, &mut positions);
        let q = point(x, w, s, &mut positions);
        let (u, v, w, x) = (u.index() as u32, v.index() as u32, w.index() as u32, x.index() as u32);
        let face = mesh.half_edges[entry.index()].face;
        faces[face.index()] = vec![u, p, q, x];
        faces.push(vec![p, v, w, q]);
        replaced.insert(face, vec![face, FaceId::new(faces.len() - 1)]);
    }

    let new_vertex_count = positions.len();
    let new_face_count = faces.len();
    Ok(EditResult {
        mesh: rebuild("insert_ring", positions, &faces)?,
        new_vertices: appended_vertices(original_vertices, new_vertex_count),
        new_faces: appended_faces(original_faces, new_face_count),
        replaced,
        vertex_map: None,
        face_map: None,
    })
}

/// Removes faces. On a cage the result is open, which is a red state, not a refusal (R2).
///
/// A vertex left with no face is removed with them rather than refused — it would be an unused
/// vertex, which the constructor refuses, and it means nothing once its faces are gone. The one
/// refusal is a bowtie: two fans of what remains meeting at a corner.
pub fn delete_faces(mesh: &QuadMesh, delete: &[FaceId]) -> Result<EditResult, EditError> {
    let mut gone = vec![false; mesh.face_count()];
    for &f in delete {
        match gone.get_mut(f.index()) {
            Some(g) => *g = true,
            None => {
                return Err(EditError::refuse(
                    format!("face {} does not exist", f.index()),
                    Highlight::default(),
                ));
            }
        }
    }
    if gone.iter().all(|&g| g) {
        return Err(EditError::refuse(
            "deleting every face would leave nothing to edit",
            Highlight { faces: delete.to_vec(), ..Highlight::default() },
        ));
    }

    let quads = mesh.quad_lists();
    let mut face_map = vec![None; quads.len()];
    let mut kept_faces = Vec::new();
    for (f, quad) in quads.into_iter().enumerate() {
        if !gone[f] {
            face_map[f] = Some(FaceId::new(kept_faces.len()));
            kept_faces.push(quad);
        }
    }
    let (positions, faces, vertex_map) = without_unused(mesh.position_list(), kept_faces);

    let new = QuadMesh::from_faces(positions, &faces).map_err(|e| match e {
        CageError::NonManifoldVertex { vertex } => {
            let old = vertex_map
                .iter()
                .position(|&n| n == Some(vertex))
                .map(VertexId::new)
                .unwrap_or(vertex);
            EditError::refuse(
                "deleting these faces leaves two fans of faces meeting at one corner",
                Highlight { vertices: vec![old], ..Highlight::default() },
            )
        }
        e => EditError::bug("delete_faces", e),
    })?;
    Ok(EditResult {
        mesh: new,
        new_vertices: Vec::new(),
        new_faces: Vec::new(),
        replaced: HashMap::new(),
        vertex_map: Some(vertex_map),
        face_map: Some(face_map),
    })
}

/// Positions and faces with the unused vertices gone, and where each old vertex went.
type Compacted = (Vec<Point3<f64>>, Vec<Vec<u32>>, Vec<Option<VertexId>>);

/// Takes edges out of the mesh, merging the faces either side of each: the inverse of
/// [`insert_ring`], which is what Delete on a selected edge loop means (the owner, 2026-10-24).
///
/// **Quads stay quads by the vertices going with the edges.** Two quads merged across an edge
/// are a hexagon; what makes a dissolved ring come out as the quads it split is that each loop
/// vertex -- valence four, two of its edges gone -- is left with two faces and two edges in a
/// straight line, and goes too. So after the merges, every vertex an edge was taken from that has
/// only two faces left is removed from both. A result with a face that is not a quad is refused
/// with that face named: a lone interior edge, say, whose hexagon has no vertex to lose.
pub fn dissolve_edges(mesh: &QuadMesh, edges: &[(VertexId, VertexId)]) -> Result<EditResult, EditError> {
    let all = Highlight { edges: edges.to_vec(), ..Highlight::default() };
    if edges.is_empty() {
        return Err(EditError::refuse("no edge to dissolve", all));
    }
    for &(a, b) in edges {
        if mesh.edge_faces(a, b).is_none() {
            return Err(EditError::refuse(
                "an open edge has one face and nothing to merge it with; delete the face instead",
                Highlight { edges: vec![(a, b)], ..Highlight::default() },
            ));
        }
    }

    // Polygons, merged one edge at a time. A face merged earlier is looked up by what it is now.
    let mut polygons: Vec<Option<Vec<u32>>> = mesh.quad_lists().into_iter().map(Some).collect();
    let mut where_is: Vec<usize> = (0..polygons.len()).collect();
    let find = |where_is: &[usize], f: usize| -> usize {
        let mut f = f;
        while where_is[f] != f {
            f = where_is[f];
        }
        f
    };
    for &(a, b) in edges {
        let (f, g) = mesh.edge_faces(a, b).expect("checked above");
        let (f, g) = (find(&where_is, f.index()), find(&where_is, g.index()));
        if f == g {
            // Both sides are already one polygon: the edge is a chord of it, and taking it out
            // would open the face into a ring.
            return Err(EditError::refuse(
                "dissolving these edges together would open a face into a ring",
                Highlight { edges: vec![(a, b)], ..Highlight::default() },
            ));
        }
        let (pf, pg) = (polygons[f].take().expect("live"), polygons[g].take().expect("live"));
        let merged = merge_across(&pf, &pg, a.index() as u32, b.index() as u32).ok_or_else(|| EditError::bug("dissolve_edges", CageError::Empty))?;
        polygons[f] = Some(merged);
        where_is[g] = f;
    }

    // The vertices the edges were taken from, left with two faces each, go with them.
    let mut ends: Vec<VertexId> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
    ends.sort();
    ends.dedup();
    for v in ends {
        let i = v.index() as u32;
        let holders: Vec<usize> = polygons.iter().enumerate().filter(|(_, p)| p.as_ref().is_some_and(|p| p.contains(&i))).map(|(k, _)| k).collect();
        if holders.len() == 2 && !mesh.vertex_on_boundary(v) {
            for k in holders {
                if let Some(p) = polygons[k].as_mut() {
                    p.retain(|&x| x != i);
                }
            }
        }
    }

    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut face_map = vec![None; polygons.len()];
    for (k, p) in polygons.into_iter().enumerate() {
        let Some(p) = p else { continue };
        if p.len() != 4 {
            return Err(EditError::refuse(
                "dissolving these edges would leave a face that is not a quad; take out a whole loop, or delete faces",
                Highlight { faces: vec![FaceId::from_index(k)], ..Highlight::default() },
            ));
        }
        face_map[k] = Some(FaceId::from_index(faces.len()));
        faces.push(p);
    }
    let (positions, faces, vertex_map) = without_unused(mesh.position_list(), faces);
    Ok(EditResult {
        mesh: rebuild("dissolve_edges", positions, &faces)?,
        new_vertices: Vec::new(),
        new_faces: Vec::new(),
        replaced: HashMap::new(),
        vertex_map: Some(vertex_map),
        face_map: Some(face_map),
    })
}

/// Two polygons joined across the edge `a–b`, which `f` runs as `a → b` and `g` as `b → a`:
/// `f` from `b` round to `a`, then `g` from `a` round to `b`, each end excluded once.
fn merge_across(f: &[u32], g: &[u32], a: u32, b: u32) -> Option<Vec<u32>> {
    let run = |p: &[u32], from: u32, to: u32| -> Option<Vec<u32>> {
        let n = p.len();
        let i = p.iter().position(|&x| x == from)?;
        let mut out = Vec::new();
        let mut k = (i + 1) % n;
        while p[k] != to {
            out.push(p[k]);
            k = (k + 1) % n;
            if out.len() > n {
                return None;
            }
        }
        Some(out)
    };
    // Whichever way round each polygon holds the edge.
    let (f_from, f_to, g_from, g_to) = {
        let fi = f.iter().position(|&x| x == a)?;
        if f[(fi + 1) % f.len()] == b { (b, a, a, b) } else { (a, b, b, a) }
    };
    let mut out = vec![f_to];
    out.extend(run(f, f_from, f_to)?);
    out.push(f_from);
    out.extend(run(g, g_from, g_to)?);
    // `out` now reads: a, (f's interior), b, (g's interior) -- or the mirror, depending on which
    // way `f` held the edge; in both the two ends appear once each.
    Some(out)
}

/// Drops the vertices no face uses and renumbers the faces, returning old → new.
fn without_unused(positions: Vec<Point3<f64>>, faces: Vec<Vec<u32>>) -> Compacted {
    let mut used = vec![false; positions.len()];
    for &i in faces.iter().flatten() {
        used[i as usize] = true;
    }
    let mut map = vec![None; positions.len()];
    let mut kept = Vec::with_capacity(positions.len());
    for (i, p) in positions.into_iter().enumerate() {
        if used[i] {
            map[i] = Some(VertexId::new(kept.len()));
            kept.push(p);
        }
    }
    let faces = faces
        .into_iter()
        .map(|f| f.into_iter().map(|i| map[i as usize].unwrap().index() as u32).collect())
        .collect();
    (kept, faces, map)
}

/// The boundary loop through `edge`, as vertices in the mesh's winding — the order its boundary
/// half-edges run.
fn boundary_loop(mesh: &QuadMesh, edge: (VertexId, VertexId)) -> Result<Vec<VertexId>, EditError> {
    let (a, b) = edge;
    let start = [mesh.directed(a, b), mesh.directed(b, a)]
        .into_iter()
        .flatten()
        .find(|&he| mesh.is_boundary(he))
        .ok_or_else(|| {
            EditError::refuse(
                "that edge is not on the rim of a hole",
                Highlight { edges: vec![edge], ..Highlight::default() },
            )
        })?;
    let by_origin = mesh.boundary_by_origin();
    let mut loop_vertices = vec![mesh.origin(start)];
    let mut he = start;
    loop {
        let next_vertex = mesh.destination(he);
        if next_vertex == mesh.origin(start) {
            break;
        }
        loop_vertices.push(next_vertex);
        he = by_origin[&next_vertex];
        if loop_vertices.len() > mesh.vertex_count() {
            return Err(EditError::bug("boundary_loop", CageError::Empty));
        }
    }
    Ok(loop_vertices)
}

fn loop_edges(vertices: &[VertexId]) -> Vec<(VertexId, VertexId)> {
    (0..vertices.len())
        .map(|i| (vertices[i], vertices[(i + 1) % vertices.len()]))
        .collect()
}

/// Fills the hole whose rim `edge` is on (spec §3.4): one quad for four edges; for an even count
/// of six or more, recursive chord splitting, with a ring inset first from eight up. An odd count
/// cannot be filled with quads and is refused.
pub fn fill_hole(mesh: &QuadMesh, edge: (VertexId, VertexId)) -> Result<EditResult, EditError> {
    let rim = boundary_loop(mesh, edge)?;
    let n = rim.len();
    if n % 2 == 1 {
        return Err(EditError::refuse(
            "a hole with an odd number of edges cannot be filled with quads; cut one edge of the rim first",
            Highlight { edges: loop_edges(&rim), ..Highlight::default() },
        ));
    }

    let mut positions = mesh.position_list();
    let mut faces = mesh.quad_lists();
    let (original_vertices, original_faces) = (positions.len(), faces.len());

    // The new faces run the rim the other way from the faces already on it.
    let polygon: Vec<u32> = rim.iter().rev().map(|v| v.index() as u32).collect();
    if n >= 8 {
        // Inset ring first: the rim keeps a clean loop of quads, and the long chords stay off it.
        let centroid = Point3::from(
            polygon.iter().map(|&i| positions[i as usize].coords).sum::<nalgebra::Vector3<f64>>()
                / n as f64,
        );
        let inner: Vec<u32> = polygon
            .iter()
            .map(|&i| {
                positions.push(lerp(positions[i as usize], centroid, 0.3));
                (positions.len() - 1) as u32
            })
            .collect();
        for k in 0..n {
            let k1 = (k + 1) % n;
            faces.push(vec![polygon[k], polygon[k1], inner[k1], inner[k]]);
        }
        faces.extend(chord_fill(&inner, &positions));
    } else {
        faces.extend(chord_fill(&polygon, &positions));
    }

    let (new_vertex_count, new_face_count) = (positions.len(), faces.len());
    Ok(EditResult {
        mesh: rebuild("fill_hole", positions, &faces)?,
        new_vertices: appended_vertices(original_vertices, new_vertex_count),
        new_faces: appended_faces(original_faces, new_face_count),
        replaced: HashMap::new(),
        vertex_map: None,
        face_map: None,
    })
}

/// Splits an even polygon into quads by chords between its own vertices (spec §3.4).
///
/// For `2k` vertices the chord runs `k` along when `k` is odd and `k − 1` when it is even, which
/// leaves both halves even; each half recurses until it is a quad. The chord taken is the shortest
/// of its kind round the polygon, which keeps a non-convex hole from being cut outside itself.
fn chord_fill(polygon: &[u32], positions: &[Point3<f64>]) -> Vec<Vec<u32>> {
    let n = polygon.len();
    if n == 4 {
        return vec![polygon.to_vec()];
    }
    let k = n / 2;
    let along = if k % 2 == 1 { k } else { k - 1 };
    let length = |s: usize| {
        (positions[polygon[(s + along) % n] as usize] - positions[polygon[s] as usize]).norm()
    };
    let s = (0..n)
        .min_by(|&x, &y| length(x).partial_cmp(&length(y)).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    let first: Vec<u32> = (0..=along).map(|i| polygon[(s + i) % n]).collect();
    let second: Vec<u32> = (along..=n).map(|i| polygon[(s + i) % n]).collect();
    let mut out = chord_fill(&first, positions);
    out.extend(chord_fill(&second, positions));
    out
}

/// One side of a bridge: a run of boundary vertices in the mesh's winding, and whether it closes.
struct Rim {
    vertices: Vec<VertexId>,
    closed: bool,
}

impl Rim {
    fn edge_count(&self) -> usize {
        if self.closed { self.vertices.len() } else { self.vertices.len() - 1 }
    }
}

/// Orders a set of boundary edges into one chain or one loop.
fn rim_of(mesh: &QuadMesh, edges: &[(VertexId, VertexId)]) -> Result<Rim, EditError> {
    let all = Highlight { edges: edges.to_vec(), ..Highlight::default() };
    if edges.is_empty() {
        return Err(EditError::refuse("each side of a bridge needs at least one edge", all));
    }
    let mut by_origin: HashMap<VertexId, HalfEdgeId> = HashMap::new();
    for &(a, b) in edges {
        let he = [mesh.directed(a, b), mesh.directed(b, a)]
            .into_iter()
            .flatten()
            .find(|&he| mesh.is_boundary(he))
            .ok_or_else(|| {
                EditError::refuse(
                    "only open edges can be bridged",
                    Highlight { edges: vec![(a, b)], ..Highlight::default() },
                )
            })?;
        by_origin.insert(mesh.origin(he), he);
    }
    let destinations: std::collections::HashSet<VertexId> =
        by_origin.values().map(|&he| mesh.destination(he)).collect();
    let start = by_origin.keys().copied().find(|v| !destinations.contains(v));
    let closed = start.is_none();
    let first = start.unwrap_or_else(|| *by_origin.keys().min().unwrap());

    let mut vertices = vec![first];
    let mut at = first;
    while let Some(&he) = by_origin.get(&at) {
        at = mesh.destination(he);
        if at == first {
            break;
        }
        vertices.push(at);
        if vertices.len() > edges.len() + 1 {
            break;
        }
    }
    let rim = Rim { vertices, closed };
    if rim.edge_count() != edges.len() {
        return Err(EditError::refuse(
            "the edges on one side of a bridge must form one chain or one loop",
            all,
        ));
    }
    Ok(rim)
}

/// Joins two rims — two boundary loops or two boundary chains — with quads, `divisions` deep
/// (spec §3.3).
///
/// **The pairing runs the second rim backwards, always.** A bridge quad has to use each rim's edge
/// the other way from the face already on it, and that forces the vertices to pair in opposite
/// order along the two rims. The spec's "flip the second loop automatically" is this. A rim's
/// direction is its own, so for two disjoint rims this pairing is always consistent: rims on faces
/// that point opposite ways give a *twisted* bridge, which the validator shows amber, not a
/// refusal here. The rebuild can still refuse — where a side edge of the bridge already exists —
/// and that refusal is mapped to a reason below.
pub fn bridge(
    mesh: &QuadMesh,
    side_a: &[(VertexId, VertexId)],
    side_b: &[(VertexId, VertexId)],
    divisions: usize,
) -> Result<EditResult, EditError> {
    let both = Highlight {
        edges: side_a.iter().chain(side_b).copied().collect(),
        ..Highlight::default()
    };
    if divisions == 0 {
        return Err(EditError::refuse("a bridge needs at least one division", both));
    }
    let a = rim_of(mesh, side_a)?;
    let b = rim_of(mesh, side_b)?;
    if a.closed != b.closed {
        return Err(EditError::refuse("bridge a loop to a loop, or a chain to a chain", both));
    }
    if a.edge_count() != b.edge_count() {
        return Err(EditError::refuse(
            format!(
                "the two sides have {} and {} edges; a bridge needs the same number on each",
                a.edge_count(),
                b.edge_count()
            ),
            both,
        ));
    }
    if a.vertices.iter().any(|v| b.vertices.contains(v)) {
        return Err(EditError::refuse(
            "the two sides share a vertex; weld joins a seam, a bridge spans a gap",
            both,
        ));
    }

    let mut positions = mesh.position_list();
    let mut faces = mesh.quad_lists();
    let (original_vertices, original_faces) = (positions.len(), faces.len());
    let n = a.vertices.len();
    let m = b.vertices.len();

    // Which vertex of the second rim each vertex of the first pairs with, running backwards.
    let partner: Vec<VertexId> = if a.closed {
        let cost = |c: usize| -> f64 {
            (0..n)
                .map(|i| {
                    (positions[a.vertices[i].index()] - positions[b.vertices[(c + m - i) % m].index()])
                        .norm()
                })
                .sum()
        };
        let c = (0..m)
            .min_by(|&x, &y| cost(x).partial_cmp(&cost(y)).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0);
        (0..n).map(|i| b.vertices[(c + m - i) % m]).collect()
    } else {
        (0..n).map(|i| b.vertices[n - 1 - i]).collect()
    };

    // Rings of vertices from the first rim (0) to the second (divisions); the ones between are new.
    let mut rings: Vec<Vec<u32>> = vec![a.vertices.iter().map(|v| v.index() as u32).collect()];
    for j in 1..divisions {
        let t = j as f64 / divisions as f64;
        rings.push(
            (0..n)
                .map(|i| {
                    positions.push(lerp(positions[a.vertices[i].index()], positions[partner[i].index()], t));
                    (positions.len() - 1) as u32
                })
                .collect(),
        );
    }
    rings.push(partner.iter().map(|v| v.index() as u32).collect());

    let edges = a.edge_count();
    for j in 0..divisions {
        for i in 0..edges {
            let i1 = (i + 1) % n;
            faces.push(vec![rings[j][i1], rings[j][i], rings[j + 1][i], rings[j + 1][i1]]);
        }
    }

    let (new_vertex_count, new_face_count) = (positions.len(), faces.len());
    let new = QuadMesh::from_faces(positions, &faces).map_err(|e| match e {
        CageError::InconsistentWinding { .. } | CageError::NonManifoldEdge { .. } => {
            EditError::refuse(
                "these two rims face the same way; a bridge between them would turn one side inside out",
                both.clone(),
            )
        }
        e => EditError::bug("bridge", e),
    })?;
    Ok(EditResult {
        mesh: new,
        new_vertices: appended_vertices(original_vertices, new_vertex_count),
        new_faces: appended_faces(original_faces, new_face_count),
        replaced: HashMap::new(),
        vertex_map: None,
        face_map: None,
    })
}

/// Takes a region of faces off each side and bridges the rims they leave: what "bridge between
/// these faces" means (the owner, 2026-10-24), and the one bridge that starts from faces rather
/// than open edges. Delete faces, then [`bridge`], as one step.
///
/// Each side is one edge-connected region whose border is a single loop; the two borders must
/// not share a vertex, and neither may run along the mesh's own rim, since that rim edge would
/// have no face left to bridge from. The counts match as a bridge's must.
pub fn bridge_faces(
    mesh: &QuadMesh,
    side_a: &[FaceId],
    side_b: &[FaceId],
    divisions: usize,
) -> Result<EditResult, EditError> {
    let both = Highlight { faces: side_a.iter().chain(side_b).copied().collect(), ..Highlight::default() };
    if side_a.is_empty() || side_b.is_empty() {
        return Err(EditError::refuse("bridge faces needs faces on each side", both));
    }
    if let Some(&f) = side_a.iter().find(|f| side_b.contains(f)) {
        return Err(EditError::refuse(
            format!("face {} is on both sides", f.index()),
            Highlight { faces: vec![f], ..Highlight::default() },
        ));
    }
    for &f in side_a.iter().chain(side_b) {
        if f.index() >= mesh.face_count() {
            return Err(EditError::refuse(format!("face {} does not exist", f.index()), Highlight::default()));
        }
    }
    // A region's border: its faces' edges whose other side is outside the region.
    let border = |faces: &[FaceId]| -> Result<Vec<(VertexId, VertexId)>, EditError> {
        let mut out = Vec::new();
        for &f in faces {
            let c = mesh.face_vertices(f);
            for i in 0..4 {
                let (a, b) = (c[i], c[(i + 1) % 4]);
                match mesh.edge_faces(a, b) {
                    Some((f0, f1)) => {
                        let other = if f0 == f { f1 } else { f0 };
                        if !faces.contains(&other) {
                            out.push((a, b));
                        }
                    }
                    None => {
                        return Err(EditError::refuse(
                            "a side on the cage's open rim can't be bridged \u{2014} fill or bridge the rim first",
                            Highlight { faces: vec![f], edges: vec![(a, b)], ..Highlight::default() },
                        ));
                    }
                }
            }
        }
        Ok(out)
    };
    let rim_a = border(side_a)?;
    let rim_b = border(side_b)?;
    let touching = rim_a.iter().flat_map(|&(a, b)| [a, b]).find(|v| rim_b.iter().any(|&(c, d)| c == *v || d == *v));
    if let Some(v) = touching {
        return Err(EditError::refuse(
            "the two sides touch; a bridge spans a gap \u{2014} pick faces apart",
            Highlight { vertices: vec![v], ..both },
        ));
    }
    let all: Vec<FaceId> = side_a.iter().chain(side_b).copied().collect();
    let deleted = delete_faces(mesh, &all)?;
    let map = deleted.vertex_map.clone().expect("delete maps its vertices");
    let carry = |rim: &[(VertexId, VertexId)]| -> Result<Vec<(VertexId, VertexId)>, EditError> {
        rim.iter()
            .map(|&(a, b)| match (map[a.index()], map[b.index()]) {
                (Some(a), Some(b)) => Ok((a, b)),
                _ => Err(EditError::bug("bridge_faces", CageError::Empty)),
            })
            .collect()
    };
    let bridged = bridge(&deleted.mesh, &carry(&rim_a)?, &carry(&rim_b)?, divisions)?;
    Ok(EditResult {
        mesh: bridged.mesh,
        new_vertices: bridged.new_vertices,
        new_faces: bridged.new_faces,
        replaced: HashMap::new(),
        vertex_map: deleted.vertex_map,
        face_map: deleted.face_map,
    })
}

/// Merges each pair's first vertex into its second, which keeps its position (spec §3.3). The tool
/// moves the second first when the weld is to the midpoint.
///
/// Refused, with the would-be problem named, when the result would collapse a face, join three
/// faces at an edge, flip a face against its neighbour, or make a bowtie.
pub fn weld(mesh: &QuadMesh, pairs: &[(VertexId, VertexId)]) -> Result<EditResult, EditError> {
    let n = mesh.vertex_count();
    let mut into: Vec<usize> = (0..n).collect();
    fn root(into: &[usize], mut v: usize) -> usize {
        while into[v] != v {
            v = into[v];
        }
        v
    }
    for &(from, to) in pairs {
        if from.index() >= n || to.index() >= n {
            return Err(EditError::refuse("a welded vertex does not exist", Highlight::default()));
        }
        let (f, t) = (root(&into, from.index()), root(&into, to.index()));
        if f != t {
            into[f] = t;
        }
    }

    let quads = mesh.quad_lists();
    let welded: Vec<Vec<u32>> = quads
        .iter()
        .map(|q| q.iter().map(|&i| root(&into, i as usize) as u32).collect())
        .collect();
    for (f, q) in welded.iter().enumerate() {
        if (0..4).any(|x| (x + 1..4).any(|y| q[x] == q[y])) {
            return Err(EditError::refuse(
                "this weld collapses a face — its corners would meet",
                Highlight { faces: vec![FaceId::new(f)], ..Highlight::default() },
            ));
        }
    }

    // Renumber: every surviving vertex in order; a merged one maps to what it became.
    let positions = mesh.position_list();
    let mut kept_index = vec![None; n];
    let mut kept = Vec::with_capacity(n);
    for v in 0..n {
        if root(&into, v) == v {
            kept_index[v] = Some(kept.len());
            kept.push(positions[v]);
        }
    }
    let vertex_map: Vec<Option<VertexId>> = (0..n)
        .map(|v| kept_index[root(&into, v)].map(VertexId::new))
        .collect();
    let faces: Vec<Vec<u32>> = welded
        .iter()
        .map(|q| q.iter().map(|&i| kept_index[i as usize].unwrap() as u32).collect())
        .collect();

    let back = |v: VertexId| -> VertexId {
        VertexId::new(kept_index.iter().position(|&k| k == Some(v.index())).unwrap_or(v.index()))
    };
    let new = QuadMesh::from_faces(kept, &faces).map_err(|e| match e {
        CageError::NonManifoldEdge { from, to } => EditError::refuse(
            "this weld would join three faces at one edge",
            Highlight { edges: vec![(back(from), back(to))], ..Highlight::default() },
        ),
        CageError::InconsistentWinding { from, to } => EditError::refuse(
            "this weld joins faces that face opposite ways",
            Highlight { edges: vec![(back(from), back(to))], ..Highlight::default() },
        ),
        CageError::NonManifoldVertex { vertex } => EditError::refuse(
            "this weld makes two fans of faces meet at one corner",
            Highlight { vertices: vec![back(vertex)], ..Highlight::default() },
        ),
        e => EditError::bug("weld", e),
    })?;
    Ok(EditResult {
        mesh: new,
        new_vertices: Vec::new(),
        new_faces: Vec::new(),
        replaced: HashMap::new(),
        vertex_map: Some(vertex_map),
        face_map: None,
    })
}

pub mod extrude;
pub mod knife;

#[cfg(test)]
pub(crate) mod tests;
