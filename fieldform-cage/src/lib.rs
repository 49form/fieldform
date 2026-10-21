// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The control cage: a closed, 2-manifold, all-quad mesh that the designer edits directly.
//!
//! The cage is never the model — see the project plan. This crate owns the mesh data
//! structure (invariant-enforcing: closed, manifold, all-quad) and, eventually, the polygon
//! editing operations (extrude, bevel, insert loop, ...) that act on it. It has no dependency
//! on Fidget; `fieldform-subd` and `fieldform-field` consume its output to build the field
//! representation.
//!
//! The mesh is stored as a half-edge (doubly-connected edge list) structure. Every half-edge
//! has an explicit twin found during construction; a half-edge without a twin means the input
//! wasn't closed, and construction fails rather than producing a mesh with a hole.

use nalgebra::Point3;
use std::collections::HashMap;

macro_rules! index_type {
    ($name:ident) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u32);

        impl $name {
            fn new(i: usize) -> Self {
                Self(i as u32)
            }

            /// Builds an id from a raw index.
            ///
            /// Callers are responsible for the index being in range — the ops layer
            /// validates against the live cage before constructing one, since ids arriving
            /// from a script or config file are untrusted.
            pub fn from_index(i: usize) -> Self {
                Self::new(i)
            }

            /// Returns the index into the corresponding arena
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

index_type!(VertexId);
index_type!(HalfEdgeId);
index_type!(FaceId);

/// A vertex in the cage
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Point3<f64>,
    /// One half-edge originating at this vertex
    half_edge: HalfEdgeId,
}

/// A half-edge: one directed side of an edge shared by exactly two faces
#[derive(Copy, Clone, Debug)]
pub struct HalfEdge {
    /// Vertex this half-edge starts at
    pub origin: VertexId,
    /// The other half-edge along the same undirected edge, going the opposite direction
    pub twin: HalfEdgeId,
    /// Next half-edge around this half-edge's face, in the same winding direction
    pub next: HalfEdgeId,
    pub face: FaceId,
}

/// A quad face
#[derive(Copy, Clone, Debug)]
pub struct Face {
    /// One half-edge on this face's boundary loop
    half_edge: HalfEdgeId,
}

/// Errors that reject an input from becoming a [`Cage`], keeping its invariants
/// (closed, 2-manifold, all-quad) true by construction rather than by later validation.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum CageError {
    #[error("face {face} has {len} vertices; cages are quad-only")]
    NonQuadFace { face: usize, len: usize },
    #[error("face {face} references vertex index {index}, out of range")]
    VertexIndexOutOfRange { face: usize, index: u32 },
    #[error(
        "edge ({from:?} -> {to:?}) is shared by more than two faces (non-manifold)"
    )]
    NonManifoldEdge { from: VertexId, to: VertexId },
    #[error(
        "edge ({from:?} -> {to:?}) has no matching reverse edge in another face \
         (mesh is not closed / has a boundary)"
    )]
    UnmatchedEdge { from: VertexId, to: VertexId },
    #[error("cage has no faces")]
    Empty,
}

/// A 2-manifold, all-quad mesh, which may have a boundary.
///
/// **The open case is the point.** A closed one is a [`Cage`], which is this checked closed --
/// and that check is what a signed distance field needs, since an open surface has no inside for
/// a sign to refer to. A cell map needs no such thing: it needs a surface to follow, and a panel
/// or a shell is exactly the open surface somebody wants reinforced.
#[derive(Debug, Clone)]
pub struct QuadMesh {
    vertices: Vec<Vertex>,
    half_edges: Vec<HalfEdge>,
    faces: Vec<Face>,
}

impl QuadMesh {
    /// Builds a cage from a vertex list and a list of faces (each face is a list of
    /// indices into `positions`, wound consistently — e.g. all counter-clockwise as seen
    /// from outside the surface). Faces are runtime-checked (not statically typed as
    /// 4-element arrays) because the main caller of this constructor is OBJ import, where
    /// an arbitrary file might contain triangles or n-gons that must be rejected with a
    /// clear error rather than assumed away.
    ///
    /// Fails if any face isn't a quad, references an out-of-range vertex, or any edge is
    /// shared by more than two faces. A boundary is **allowed** — see [`Cage::from_faces`],
    /// which is this plus the one check that makes a mesh closed.
    pub fn from_faces(
        positions: Vec<Point3<f64>>,
        input_faces: &[Vec<u32>],
    ) -> Result<Self, CageError> {
        if input_faces.is_empty() {
            return Err(CageError::Empty);
        }

        let mut vertices: Vec<Vertex> = positions
            .into_iter()
            .map(|position| Vertex {
                position,
                // placeholder, filled in once half-edges exist
                half_edge: HalfEdgeId::new(0),
            })
            .collect();

        let mut half_edges = Vec::with_capacity(input_faces.len() * 4);
        let mut faces = Vec::with_capacity(input_faces.len());
        // Maps a directed edge (origin -> dest) to its half-edge index, so twins can be
        // resolved once every face has been walked.
        let mut edge_lookup: HashMap<(VertexId, VertexId), HalfEdgeId> = HashMap::new();

        for (face_i, quad) in input_faces.iter().enumerate() {
            if quad.len() != 4 {
                return Err(CageError::NonQuadFace {
                    face: face_i,
                    len: quad.len(),
                });
            }
            let face_id = FaceId::new(face_i);
            let first_he = HalfEdgeId::new(half_edges.len());
            faces.push(Face {
                half_edge: first_he,
            });

            for corner in 0..4 {
                let from_idx = quad[corner];
                let to_idx = quad[(corner + 1) % 4];
                if from_idx as usize >= vertices.len() {
                    return Err(CageError::VertexIndexOutOfRange {
                        face: face_i,
                        index: from_idx,
                    });
                }
                let origin = VertexId::new(from_idx as usize);
                let dest = VertexId::new({
                    if to_idx as usize >= vertices.len() {
                        return Err(CageError::VertexIndexOutOfRange {
                            face: face_i,
                            index: to_idx,
                        });
                    }
                    to_idx as usize
                });

                let he_id = HalfEdgeId::new(half_edges.len());
                let next = HalfEdgeId::new(
                    half_edges.len() - corner + (corner + 1) % 4,
                );
                half_edges.push(HalfEdge {
                    origin,
                    // resolved in the pass below
                    twin: he_id,
                    next,
                    face: face_id,
                });
                vertices[origin.index()].half_edge = he_id;

                if edge_lookup.insert((origin, dest), he_id).is_some() {
                    return Err(CageError::NonManifoldEdge {
                        from: origin,
                        to: dest,
                    });
                }
            }
        }

        // Resolve twins: half-edge (a -> b)'s twin is (b -> a). One with no reverse edge is on
        // the boundary and keeps the self-twin it was constructed with — see
        // [`Self::is_boundary`], where that convention is written down.
        for i in 0..half_edges.len() {
            let origin = half_edges[i].origin;
            let dest = half_edges[half_edges[i].next.index()].origin;
            if let Some(&twin) = edge_lookup.get(&(dest, origin)) {
                half_edges[i].twin = twin;
            }
        }

        // **A boundary vertex has to start its fan on the boundary.** The loop above left each
        // vertex pointing at whichever of its half-edges was built last, and the walk around a
        // vertex is an arc rather than a cycle when it ends on the boundary — so a start in the
        // middle of the arc would walk off one end and never see the other. Pointing every
        // boundary vertex at its own boundary half-edge makes the stored start the arc's start,
        // which is what `half_edges_around_vertex` then relies on and what closes the case for
        // `edge_faces`, `valence` and the rest at once.
        for i in 0..half_edges.len() {
            let he = HalfEdgeId::new(i);
            if half_edges[i].twin == he {
                vertices[half_edges[i].origin.index()].half_edge = he;
            }
        }

        Ok(Self {
            vertices,
            half_edges,
            faces,
        })
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of undirected edges (half the half-edge count, since every edge has
    /// exactly two half-edges once the cage is constructed)
    /// The number of undirected edges.
    ///
    /// **Not `half_edges / 2`, which is what this was.** An interior edge has two half-edges
    /// and a boundary edge has one, so halving under-reports an open mesh by half its boundary
    /// length — and *rounds* rather than erroring, so nothing says so. Counted by the same
    /// predicate [`Self::edges`] yields each undirected edge by, for the same reason.
    pub fn edge_count(&self) -> usize {
        self.edges().count()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn faces(&self) -> &[Face] {
        &self.faces
    }

    /// Iterates the ids of every face in the cage, e.g. for triangulating it for display
    pub fn face_ids(&self) -> impl Iterator<Item = FaceId> {
        (0..self.faces.len()).map(FaceId::new)
    }

    /// Iterates the ids of every vertex in the cage
    pub fn vertex_ids(&self) -> impl Iterator<Item = VertexId> {
        (0..self.vertices.len()).map(VertexId::new)
    }

    /// Iterates every undirected edge exactly once, as its two endpoint vertex ids
    pub fn edges(&self) -> impl Iterator<Item = (VertexId, VertexId)> + '_ {
        self.half_edges.iter().enumerate().filter_map(move |(i, he)| {
            // Emit only from the lower-indexed side of each half-edge/twin pair, so each
            // undirected edge is yielded once rather than twice — and from a boundary
            // half-edge, which is the only one its edge has. `i < twin` excludes a self-twin,
            // so the boundary arm is a separate one rather than a widened comparison.
            let boundary = he.twin.index() == i;
            (boundary || i < he.twin.index()).then(|| {
                let destination = self.half_edges[he.next.index()].origin;
                (he.origin, destination)
            })
        })
    }

    /// Whether this half-edge is on the mesh's boundary — nothing on the other side of it.
    ///
    /// **A self-twin is the representation, and it costs nothing.** `HalfEdge::twin` is not an
    /// `Option`: the constructor initialises it to the half-edge's own id, and the resolution
    /// pass overwrites it only where a reverse edge exists. So `twin == self` was already the
    /// in-flight sentinel for "unresolved", and it is exactly the right permanent meaning for
    /// "there is nothing on the other side" — no `Option`, no memory, no type change. Said here
    /// rather than left to be inferred from the order two passes happen to run in.
    pub fn is_boundary(&self, he: HalfEdgeId) -> bool {
        self.half_edges[he.index()].twin == he
    }

    /// Whether this vertex sits on the mesh's boundary.
    ///
    /// Asked of the fan rather than of the stored start, so it does not depend on the
    /// constructor's arrangement of pointing a boundary vertex at its own boundary half-edge.
    /// That arrangement is what makes the walk single-pass; this is what the walk means.
    pub fn vertex_on_boundary(&self, vertex: VertexId) -> bool {
        self.half_edges_around_vertex(vertex)
            .into_iter()
            .any(|he| {
                let arriving = self.half_edges
                    [self.half_edges[self.half_edges[he.index()].next.index()].next.index()]
                .next;
                self.is_boundary(he) || self.is_boundary(arriving)
            })
    }

    /// Whether any of this mesh's edges is on a boundary — whether it is open.
    pub fn is_open(&self) -> bool {
        (0..self.half_edges.len()).any(|i| self.is_boundary(HalfEdgeId::new(i)))
    }

    /// Current position of a vertex
    pub fn vertex_position(&self, id: VertexId) -> Point3<f64> {
        self.vertices[id.index()].position
    }

    /// Walks the half-edges originating at `vertex`, one per incident face, in order around
    /// the vertex.
    ///
    /// **Around an interior vertex this is a cycle; around a boundary vertex it is an arc**,
    /// and that difference is the whole of why this is written in two passes. The single loop
    /// it used to be walked until it came back to where it started, and said the assumption
    /// out loud in its own doc comment: *on a closed manifold this always returns to the
    /// start*. On a boundary vertex it does not — the walk reaches a boundary half-edge,
    /// follows its self-twin back to itself, and spins there.
    ///
    /// Every neighbourhood query goes through this one function — [`Self::valence`],
    /// [`Self::adjacent_vertices`], [`Self::adjacent_faces`], [`Self::edge_faces`] — so it
    /// alone decides whether the crate survives an open mesh.
    fn half_edges_around_vertex(&self, vertex: VertexId) -> Vec<HalfEdgeId> {
        let start = self.vertices[vertex.index()].half_edge;
        let mut out = vec![start];
        let mut he = start;
        loop {
            // On a quad, `next` three times from a half-edge leaving the vertex is the one
            // arriving at it on the same face; its twin then leaves the vertex again on the
            // face round from here. A boundary there is the end of the arc — and the start is
            // already the other end, because the constructor points a boundary vertex at its
            // own boundary half-edge.
            let arriving = self.half_edges
                [self.half_edges[self.half_edges[he.index()].next.index()].next.index()]
            .next;
            if self.is_boundary(arriving) {
                break;
            }
            he = self.half_edges[arriving.index()].twin;
            if he == start {
                break;
            }
            out.push(he);
        }
        out
    }

    /// Number of edges meeting at this vertex. Note that a valid cage can still have
    /// "extraordinary" vertices (valence != 4) — a cube's corners are all valence 3.
    pub fn valence(&self, vertex: VertexId) -> usize {
        self.half_edges_around_vertex(vertex).len()
    }

    /// The vertices directly connected to `vertex` by an edge, in order around it.
    ///
    /// **`valence` of them around an interior vertex and `valence + 1` around a boundary one.**
    /// Each half-edge of the fan contributes the vertex it points at, which is every neighbour
    /// on a cycle -- the fan comes back round and the last one's other end is the first one's
    /// start. An arc does not come back round, so the vertex at its far end is reached by an
    /// edge and by no half-edge *leaving* this vertex, and is missed unless it is added.
    ///
    /// The asymmetry is correct rather than a wart: the two ends of an arc are each reached by
    /// one edge instead of being shared between two faces. It is also the kind of off-by-one
    /// that yields a limit surface which is subtly wrong rather than obviously broken, which is
    /// why `a_boundary_vertexs_fan_is_an_arc_and_an_interior_vertexs_is_a_cycle` checks it on a
    /// single quad, where the answer is two and anybody can see it.
    pub fn adjacent_vertices(&self, vertex: VertexId) -> Vec<VertexId> {
        let fan = self.half_edges_around_vertex(vertex);
        let mut out: Vec<VertexId> = fan
            .iter()
            .map(|he| self.half_edges[self.half_edges[he.index()].next.index()].origin)
            .collect();
        if let Some(&last) = fan.last() {
            // The half-edge arriving at this vertex on the fan's last face. Where that is a
            // boundary the fan is an arc, and this is the neighbour at its far end.
            let arriving = self.half_edges
                [self.half_edges[self.half_edges[last.index()].next.index()].next.index()]
            .next;
            if self.is_boundary(arriving) {
                out.push(self.half_edges[arriving.index()].origin);
            }
        }
        out
    }

    /// The faces incident to `vertex`, in order around it.
    pub fn adjacent_faces(&self, vertex: VertexId) -> Vec<FaceId> {
        self.half_edges_around_vertex(vertex)
            .into_iter()
            .map(|he| self.half_edges[he.index()].face)
            .collect()
    }

    /// The two faces sharing the edge between `a` and `b`, if such an edge exists and has two.
    ///
    /// **`None` on a boundary edge, which has one.** A self-twin points back at its own face,
    /// so without this the answer would be `Some((f, f))` -- the same face twice, from a
    /// function whose whole promise is that it names two different ones. Nothing would say so.
    pub fn edge_faces(&self, a: VertexId, b: VertexId) -> Option<(FaceId, FaceId)> {
        let he = self.half_edges_around_vertex(a).into_iter().find(|he| {
            self.half_edges[self.half_edges[he.index()].next.index()].origin == b
        })?;
        if self.is_boundary(he) {
            return None;
        }
        let twin = self.half_edges[he.index()].twin;
        Some((
            self.half_edges[he.index()].face,
            self.half_edges[twin.index()].face,
        ))
    }

    /// Moves a vertex to a new position. Purely geometric — doesn't touch topology, so it
    /// can't violate the closed/manifold/all-quad invariants `from_faces` enforces.
    pub fn set_vertex_position(&mut self, id: VertexId, position: Point3<f64>) {
        self.vertices[id.index()].position = position;
    }

    /// A number that differs whenever the mesh does — for a cache that has to know whether what
    /// it built from this mesh is still this mesh.
    ///
    /// **An id does not say that.** A cage and a surface to follow are both edited in place, under
    /// the ids they already had, so anything keyed by id alone goes on answering for the shape
    /// before the edit. Positions are hashed by their bits, so any move at all is a different
    /// mesh; the faces' corners too, since a mesh replaced under the same id could keep its
    /// positions and change how they are joined. One pass over the vertices and faces, which is
    /// nothing beside the subdivision or tessellation a cache hit saves.
    pub fn geometry_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.vertices.len().hash(&mut h);
        for v in &self.vertices {
            v.position.x.to_bits().hash(&mut h);
            v.position.y.to_bits().hash(&mut h);
            v.position.z.to_bits().hash(&mut h);
        }
        self.faces.len().hash(&mut h);
        for face in self.face_ids() {
            for v in self.face_vertices(face) {
                v.index().hash(&mut h);
            }
        }
        h.finish()
    }

    /// The four half-edges bounding a face, in winding order. `result[i]` originates at
    /// `face_vertices(face)[i]`.
    ///
    /// Exposed for algorithms that need to walk the 1-ring around a face — exact
    /// limit-surface evaluation needs the 4x4 control grid, which spans the neighbouring
    /// faces and can only be gathered in a consistent order through the half-edge links.
    pub fn face_half_edges(&self, face: FaceId) -> [HalfEdgeId; 4] {
        let mut out = [HalfEdgeId::new(0); 4];
        let mut he = self.faces[face.index()].half_edge;
        for slot in out.iter_mut() {
            *slot = he;
            he = self.half_edges[he.index()].next;
        }
        out
    }

    /// Looks up a half-edge by id.
    pub fn half_edge(&self, id: HalfEdgeId) -> &HalfEdge {
        &self.half_edges[id.index()]
    }

    /// Returns the four vertex ids of a quad face, in winding order
    pub fn face_vertices(&self, face: FaceId) -> [VertexId; 4] {
        let mut out = [VertexId::new(0); 4];
        let mut he = self.faces[face.index()].half_edge;
        for slot in out.iter_mut() {
            *slot = self.half_edges[he.index()].origin;
            he = self.half_edges[he.index()].next;
        }
        out
    }

    /// Euler characteristic, `V − E + F`.
    ///
    /// **`2 − 2g − b`**, for genus `g` and `b` boundary components — so a closed genus-0 cage
    /// like the default cube gives 2, a torus 0, and a single open quad 1. The arithmetic never
    /// changed; what a boundary changes is what the number *means*, and this is read to decide
    /// how many extraordinary vertices a mesh is obliged to have, so the `b` term is not a
    /// footnote.
    pub fn euler_characteristic(&self) -> i64 {
        self.vertex_count() as i64 - self.edge_count() as i64 + self.face_count() as i64
    }
}

/// A quad mesh that has been checked closed.
///
/// **A newtype, not a flag.** An `is_open: bool` on one type would put a check in every
/// downstream consumer that none of them can enforce, and [`CageField`] in particular would gain
/// a runtime failure mode where it has a compile-time guarantee today. Closed is a property of
/// the mesh, so it belongs in the type of the mesh: a `Cage` is closed because there is no way
/// to make one that is not.
///
/// Everything a [`QuadMesh`] can answer, a cage answers the same way — it derefs — so this adds
/// the guarantee without adding a second vocabulary for the queries.
///
/// [`CageField`]: https://example.invalid
#[derive(Clone, Debug)]
pub struct Cage(QuadMesh);

impl std::ops::Deref for Cage {
    type Target = QuadMesh;

    fn deref(&self) -> &QuadMesh {
        &self.0
    }
}

impl std::ops::DerefMut for Cage {
    /// Only the geometry is reachable this way in practice: [`QuadMesh`] exposes no method that
    /// changes topology, so a cage cannot be opened through its own deref.
    fn deref_mut(&mut self) -> &mut QuadMesh {
        &mut self.0
    }
}

impl Cage {
    /// The same constructor [`QuadMesh::from_faces`] is, plus the check that makes it a cage.
    ///
    /// Fails for everything that one fails for, and additionally when any edge has no matching
    /// reverse edge — that is, when the mesh has a boundary and is not closed.
    pub fn from_faces(
        positions: Vec<Point3<f64>>,
        input_faces: &[Vec<u32>],
    ) -> Result<Self, CageError> {
        let mesh = QuadMesh::from_faces(positions, input_faces)?;
        for i in 0..mesh.half_edges.len() {
            let he = HalfEdgeId::new(i);
            if mesh.is_boundary(he) {
                let origin = mesh.half_edges[i].origin;
                let destination = mesh.half_edges[mesh.half_edges[i].next.index()].origin;
                return Err(CageError::UnmatchedEdge {
                    from: origin,
                    to: destination,
                });
            }
        }
        Ok(Self(mesh))
    }

    /// The mesh underneath, for a consumer that does not need the closure — the cell map is the
    /// one that does not.
    pub fn mesh(&self) -> &QuadMesh {
        &self.0
    }

    /// Builds a unit cube cage, centered on the origin, with all quad faces wound
    /// counter-clockwise as seen from outside.
    /// A cube cage `size` millimetres on a side, centred on the origin.
    ///
    /// Sizes are millimetres like everything else, so a caller asking for a metre asks for
    /// 1000 — there is no separate "model scale" to keep in step.
    pub fn cube_of(size: f64) -> Self {
        let cube = Self::cube();
        let positions: Vec<Point3<f64>> = cube
            .vertex_ids()
            .map(|v| Point3::from(cube.vertex_position(v).coords * size))
            .collect();
        let faces: Vec<Vec<u32>> = cube
            .face_ids()
            .map(|f| {
                cube.face_vertices(f)
                    .iter()
                    .map(|v| v.index() as u32)
                    .collect()
            })
            .collect();
        Self::from_faces(positions, &faces).expect("scaling preserves the cage invariants")
    }

    /// The unit cube: one millimetre on a side, centred on the origin.
    pub fn cube() -> Self {
        let p = [
            Point3::new(-0.5, -0.5, -0.5), // 0
            Point3::new(0.5, -0.5, -0.5),  // 1
            Point3::new(0.5, 0.5, -0.5),   // 2
            Point3::new(-0.5, 0.5, -0.5),  // 3
            Point3::new(-0.5, -0.5, 0.5),  // 4
            Point3::new(0.5, -0.5, 0.5),   // 5
            Point3::new(0.5, 0.5, 0.5),    // 6
            Point3::new(-0.5, 0.5, 0.5),   // 7
        ];
        let quads: [Vec<u32>; 6] = [
            vec![0, 3, 2, 1], // -Z
            vec![4, 5, 6, 7], // +Z
            vec![0, 1, 5, 4], // -Y
            vec![3, 7, 6, 2], // +Y
            vec![0, 4, 7, 3], // -X
            vec![1, 2, 6, 5], // +X
        ];
        Self::from_faces(p.to_vec(), &quads)
            .expect("hardcoded cube topology is always a valid cage")
    }
}

/// How a cage is written to a file: the same positions and quads
/// [`Cage::from_faces`](Cage::from_faces) takes, and nothing else.
///
/// The half-edge structure is *derived*, so writing it would put a second, redundant copy of
/// the topology in the file — one that a hand-edited or third-party-written document could
/// contradict. Reconstructing through the constructor means a cage read from a file has been
/// through exactly the checks a cage built in the app has, and the closed/manifold/all-quad
/// invariant holds by the same mechanism rather than by trusting the writer.
///
/// Positions are plain `[x, y, z]` arrays rather than nalgebra's own serialized shape. The
/// on-disk format should not change if a dependency changes how it serializes a point, and
/// `{"coords":{...}}` per vertex is not what an inspectable format looks like.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CageData {
    positions: Vec<[f64; 3]>,
    /// Each face's four vertex indices, wound as they were on the way in.
    faces: Vec<[u32; 4]>,
}

/// An open mesh serializes **identically to a cage** — the same positions and quads — and its
/// openness stays *derived*, exactly as its half-edge topology does. A file describing a closed
/// mesh read back as a `QuadMesh` is closed; the same file read as a `Cage` is a cage. Nothing in
/// the format says which, because nothing in the geometry does.
impl serde::Serialize for QuadMesh {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let data = CageData {
            positions: self
                .vertices
                .iter()
                .map(|v| [v.position.x, v.position.y, v.position.z])
                .collect(),
            faces: self
                .face_ids()
                .map(|f| self.face_vertices(f).map(|v| v.index() as u32))
                .collect(),
        };
        data.serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for QuadMesh {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let data = CageData::deserialize(d)?;
        let positions = data
            .positions
            .into_iter()
            .map(|[x, y, z]| Point3::new(x, y, z))
            .collect();
        let faces: Vec<Vec<u32>> = data.faces.into_iter().map(|f| f.to_vec()).collect();
        QuadMesh::from_faces(positions, &faces).map_err(serde::de::Error::custom)
    }
}

impl serde::Serialize for Cage {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let data = CageData {
            positions: self
                .vertices
                .iter()
                .map(|v| [v.position.x, v.position.y, v.position.z])
                .collect(),
            faces: self
                .face_ids()
                .map(|f| self.face_vertices(f).map(|v| v.index() as u32))
                .collect(),
        };
        data.serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for Cage {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let data = CageData::deserialize(d)?;
        let positions = data
            .positions
            .into_iter()
            .map(|[x, y, z]| Point3::new(x, y, z))
            .collect();
        let faces: Vec<Vec<u32>> = data.faces.into_iter().map(|f| f.to_vec()).collect();
        Cage::from_faces(positions, &faces).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single quad: four vertices, four edges, one face, and a boundary all the way round.
    /// Every answer below is checkable by hand, which is why this is the shape to ask.
    fn one_quad() -> QuadMesh {
        QuadMesh::from_faces(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3]],
        )
        .expect("a single quad is a quad mesh, even though it is not a cage")
    }

    /// **Two boundary components: an annulus.** `2 - 2g - b` gives 0 here, and `b = 2` is
    /// exactly where a hardcoded `2 - b` assumption would show. Named in the spec's edge cases
    /// for that reason.
    #[test]
    fn a_mesh_with_two_boundaries_has_euler_characteristic_nought() {
        // A 3x3 grid of quads with the middle one left out: an outer boundary and an inner one.
        let mut positions = Vec::new();
        for j in 0..4 {
            for i in 0..4 {
                positions.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * 4 + i) as u32;
        let mut faces = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                if (i, j) == (1, 1) {
                    continue;
                }
                faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        let mesh = QuadMesh::from_faces(positions, &faces).expect("an annulus of quads");

        assert_eq!(mesh.face_count(), 8, "nine less the hole");
        assert_eq!(mesh.vertex_count(), 16);
        assert_eq!(mesh.edge_count(), 24, "counted, not halved");
        assert_eq!(
            mesh.euler_characteristic(),
            0,
            "16 - 24 + 8, which is 2 - 0 - 2 for two boundary components"
        );
        assert!(mesh.is_open());

        // The four vertices round the hole are on a boundary and have three faces each.
        for (i, j) in [(1, 1), (2, 1), (2, 2), (1, 2)] {
            let v = VertexId::new(j * 4 + i);
            assert!(mesh.vertex_on_boundary(v), "({i}, {j}) rings the hole");
            assert_eq!(mesh.valence(v), 3, "three of the four quads survive at ({i}, {j})");
            assert_eq!(
                mesh.adjacent_vertices(v).len(),
                4,
                "an arc of three faces has four neighbours"
            );
        }
    }

    /// **The same mesh is a quad mesh and not a cage**, which is the whole of the split.
    #[test]
    fn a_single_quad_is_a_mesh_but_not_a_cage() {
        let mesh = one_quad();
        assert!(mesh.is_open(), "a lone quad is open");
        assert!(
            matches!(
                Cage::from_faces(
                    vec![
                        Point3::new(0.0, 0.0, 0.0),
                        Point3::new(1.0, 0.0, 0.0),
                        Point3::new(1.0, 1.0, 0.0),
                        Point3::new(0.0, 1.0, 0.0),
                    ],
                    &[vec![0, 1, 2, 3]],
                ),
                Err(CageError::UnmatchedEdge { .. })
            ),
            "the closure check is what makes a cage, and it still refuses this"
        );
        assert!(!Cage::cube().is_open(), "a cube is closed");
    }

    /// **A boundary vertex's fan is an arc, so it has one more neighbour than it has faces.**
    ///
    /// The off-by-one that produces a limit surface which is subtly wrong rather than obviously
    /// broken: `adjacent_vertices` returns `valence + 1` entries on a boundary vertex and
    /// `valence` on an interior one, because the two ends of an arc are each reached by one
    /// edge rather than shared between two faces. Asked on a single quad, where a corner has
    /// one face and two neighbours and anybody can check it by eye.
    #[test]
    fn a_boundary_vertexs_fan_is_an_arc_and_an_interior_vertexs_is_a_cycle() {
        let mesh = one_quad();
        for v in mesh.vertex_ids() {
            assert_eq!(mesh.valence(v), 1, "one quad, so one face at every corner");
            assert_eq!(
                mesh.adjacent_vertices(v).len(),
                2,
                "a corner of a lone quad has two edge-neighbours, not one"
            );
            assert_eq!(mesh.adjacent_faces(v), vec![FaceId::new(0)]);
        }

        // And the closed case is unchanged: a cube's corner has three faces and three
        // neighbours, the cycle it always had.
        let cube = Cage::cube();
        for v in cube.vertex_ids() {
            assert_eq!(cube.valence(v), 3);
            assert_eq!(cube.adjacent_vertices(v).len(), 3);
        }
    }

    /// **Four edges, not two.** `half_edges / 2` under-reports an open mesh by half its
    /// boundary length and rounds rather than erroring, which is why this is counted rather
    /// than divided.
    #[test]
    fn an_open_meshs_edges_are_counted_rather_than_halved() {
        let mesh = one_quad();
        assert_eq!(mesh.edge_count(), 4, "a quad has four edges");
        assert_eq!(mesh.edges().count(), 4, "and yields each of them once");

        let listed: Vec<(usize, usize)> = mesh
            .edges()
            .map(|(a, b)| {
                let (a, b) = (a.index(), b.index());
                if a < b { (a, b) } else { (b, a) }
            })
            .collect();
        for pair in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            assert!(listed.contains(&pair), "{pair:?} missing from {listed:?}");
        }

        // The closed count is what it always was, by the same function.
        assert_eq!(Cage::cube().edge_count(), 12);
    }

    /// **`2 − 2g − b`**: a closed genus-0 cage gives 2 and a disc gives 1. The arithmetic did
    /// not change; what a boundary changes is what the number means.
    #[test]
    fn the_euler_characteristic_counts_boundary_components_too() {
        assert_eq!(Cage::cube().euler_characteristic(), 2, "closed, genus 0");
        assert_eq!(
            one_quad().euler_characteristic(),
            1,
            "a disc: 4 - 4 + 1, which is 2 - 0 - 1"
        );
    }

    /// A strip of two quads: the shared edge is interior and the other six are boundary, so
    /// the two middle vertices have two faces and three neighbours each.
    #[test]
    fn an_interior_edge_of_an_open_mesh_still_finds_both_its_faces() {
        let mesh = QuadMesh::from_faces(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(2.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 4, 5], vec![1, 2, 3, 4]],
        )
        .expect("two quads sharing an edge");

        assert_eq!(mesh.edge_count(), 7, "six around the outside and one shared");
        assert_eq!(mesh.euler_characteristic(), 1, "still a disc");

        let (middle_a, middle_b) = (VertexId::new(1), VertexId::new(4));
        for v in [middle_a, middle_b] {
            assert_eq!(mesh.valence(v), 2, "two faces meet here");
            assert_eq!(
                mesh.adjacent_vertices(v).len(),
                3,
                "an arc of two faces has three neighbours"
            );
        }
        assert!(
            mesh.edge_faces(middle_a, middle_b).is_some(),
            "the shared edge has two faces and should say so"
        );
        assert_eq!(
            mesh.edge_faces(VertexId::new(0), VertexId::new(1)),
            None,
            "a boundary edge has one face, and `edge_faces` promises two"
        );
    }

    #[test]
    fn a_cage_round_trips_through_json_with_its_topology_intact() {
        let cage = Cage::cube_of(30.0);
        let json = serde_json::to_string(&cage).unwrap();
        let back: Cage = serde_json::from_str(&json).unwrap();

        assert_eq!(back.vertex_count(), cage.vertex_count());
        assert_eq!(back.edge_count(), cage.edge_count());
        assert_eq!(back.face_count(), cage.face_count());
        for v in cage.vertex_ids() {
            assert_eq!(back.vertex_position(v), cage.vertex_position(v));
        }
        // Indices, not just counts: everything above the cage addresses vertices and faces
        // by index, so a reordering on load would silently rewire every graph that referred
        // to this cage.
        for f in cage.face_ids() {
            assert_eq!(back.face_vertices(f), cage.face_vertices(f));
        }
    }

    #[test]
    fn an_edited_cage_round_trips() {
        // The file holds current positions, not the shape as constructed — the direct layer
        // is destructive, so the edited state is the only truth there is to write.
        let mut cage = Cage::cube_of(10.0);
        cage.set_vertex_position(VertexId::new(0), Point3::new(-9.0, 2.5, 3.0));
        let back: Cage = serde_json::from_str(&serde_json::to_string(&cage).unwrap()).unwrap();
        assert_eq!(back.vertex_position(VertexId::new(0)), Point3::new(-9.0, 2.5, 3.0));
    }

    #[test]
    fn serializing_writes_positions_and_quads_and_no_derived_topology() {
        // The file's shape is part of the format's promise to be inspectable, so it is
        // asserted rather than left to whatever the struct happens to look like.
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&Cage::cube()).unwrap()).unwrap();
        let object = value.as_object().unwrap();
        assert_eq!(object.len(), 2, "positions and faces, nothing else");
        assert_eq!(object["positions"].as_array().unwrap().len(), 8);
        assert_eq!(object["faces"].as_array().unwrap().len(), 6);
        assert_eq!(object["positions"][0].as_array().unwrap().len(), 3, "flat [x, y, z]");
    }

    #[test]
    fn a_file_describing_an_impossible_cage_is_refused_on_load() {
        // Reconstructing through `from_faces` is what makes this true: a document that has
        // been hand-edited, truncated, or written by something that got the winding wrong
        // fails here rather than becoming a cage whose invariants do not hold.
        let open = r#"{"positions":[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],"faces":[[0,1,2,3]]}"#;
        assert!(serde_json::from_str::<Cage>(open).is_err(), "one quad is not closed");

        let triangle = r#"{"positions":[[0,0,0],[1,0,0],[1,1,0]],"faces":[[0,1,2]]}"#;
        assert!(serde_json::from_str::<Cage>(triangle).is_err(), "cages are quad-only");

        let dangling = r#"{"positions":[[0,0,0]],"faces":[[0,1,2,3]]}"#;
        assert!(serde_json::from_str::<Cage>(dangling).is_err(), "index out of range");
    }

    #[test]
    fn cube_topology() {
        let cube = Cage::cube();
        assert_eq!(cube.vertex_count(), 8);
        assert_eq!(cube.edge_count(), 12);
        assert_eq!(cube.face_count(), 6);
        assert_eq!(cube.euler_characteristic(), 2);
    }

    #[test]
    fn cube_faces_are_quads_with_valid_loops() {
        let cube = Cage::cube();
        for i in 0..cube.face_count() {
            let verts = cube.face_vertices(FaceId::new(i));
            // All four corners distinct
            for a in 0..4 {
                for b in (a + 1)..4 {
                    assert_ne!(verts[a], verts[b]);
                }
            }
        }
    }

    #[test]
    fn edges_yields_each_undirected_edge_exactly_once() {
        let cube = Cage::cube();
        let edges: Vec<_> = cube.edges().collect();
        assert_eq!(edges.len(), cube.edge_count());
        // No duplicate undirected pairs.
        let mut seen = std::collections::HashSet::new();
        for (a, b) in edges {
            let key = (a.min(b), a.max(b));
            assert!(seen.insert(key), "edge {key:?} yielded more than once");
        }
    }

    #[test]
    fn cube_vertex_neighborhoods() {
        let cube = Cage::cube();
        for v in cube.vertex_ids() {
            // Every corner of a cube has three edges and three faces meeting at it.
            assert_eq!(cube.valence(v), 3, "valence of {v:?}");
            let adj = cube.adjacent_vertices(v);
            assert_eq!(adj.len(), 3);
            let unique: std::collections::HashSet<_> = adj.iter().collect();
            assert_eq!(unique.len(), 3, "adjacent vertices must be distinct");
            assert!(!adj.contains(&v), "vertex must not be its own neighbor");

            let faces = cube.adjacent_faces(v);
            assert_eq!(faces.len(), 3);
            for f in faces {
                assert!(cube.face_vertices(f).contains(&v));
            }
        }
    }

    #[test]
    fn edge_faces_returns_the_two_faces_sharing_an_edge() {
        let cube = Cage::cube();
        for (a, b) in cube.edges() {
            let (f0, f1) = cube.edge_faces(a, b).expect("edge should exist");
            assert_ne!(f0, f1);
            for f in [f0, f1] {
                let verts = cube.face_vertices(f);
                assert!(verts.contains(&a) && verts.contains(&b));
            }
        }
    }

    #[test]
    fn set_vertex_position_moves_only_that_vertex() {
        let mut cube = Cage::cube();
        let v0 = cube.vertex_ids().next().unwrap();
        let original = cube.vertex_position(v0);
        let moved = Point3::new(9.0, 9.0, 9.0);
        cube.set_vertex_position(v0, moved);
        assert_eq!(cube.vertex_position(v0), moved);
        assert_ne!(cube.vertex_position(v0), original);
        // Topology (face/edge/vertex counts) is untouched by a purely geometric move.
        assert_eq!(cube.vertex_count(), 8);
        assert_eq!(cube.edge_count(), 12);
        assert_eq!(cube.face_count(), 6);
    }

    #[test]
    fn triangle_face_is_rejected() {
        // A lone triangle: not a quad, so it must be rejected before any manifold/closure
        // checks even run.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![vec![0, 1, 2]];
        let result = Cage::from_faces(positions, &faces);
        assert_eq!(
            result.unwrap_err(),
            CageError::NonQuadFace { face: 0, len: 3 }
        );
    }

    #[test]
    fn an_open_mesh_is_refused_as_a_cage() {
        // A single quad has boundary edges (each edge belongs to only one face), so it
        // must be rejected as not closed.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![vec![0, 1, 2, 3]];
        let result = Cage::from_faces(positions, &faces);
        assert_eq!(
            result.unwrap_err(),
            CageError::UnmatchedEdge {
                from: VertexId::new(0),
                to: VertexId::new(1),
            }
        );
    }

    #[test]
    fn non_manifold_edge_is_rejected() {
        // Three quads all sharing the same directed edge (0 -> 1) is non-manifold.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(0.0, 1.0, -1.0),
        ];
        let faces = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 4, 5],
            vec![0, 1, 6, 7],
        ];
        let result = Cage::from_faces(positions, &faces);
        assert_eq!(
            result.unwrap_err(),
            CageError::NonManifoldEdge {
                from: VertexId::new(0),
                to: VertexId::new(1),
            }
        );
    }

    #[test]
    fn empty_cage_is_rejected() {
        assert_eq!(
            Cage::from_faces(vec![], &[]).unwrap_err(),
            CageError::Empty
        );
    }
}
