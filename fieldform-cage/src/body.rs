// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A cage as the document holds it: the edited mesh, which may be open, and the [`Cage`] it is
//! when it is closed (the cage-editing notes §2.1–2.2), with what the checks found wrong with it.
//!
//! **The slot holds the mesh, not the guarantee.** `Document::cages` used to hold a `Cage`, so a
//! cage could never be open — and a tool that deletes a face has to leave one open (R2). Holding
//! a `QuadMesh` and asking it to promote on every read would clone it for the question, so the
//! answer is kept instead: the body is either a `Cage` or a mesh that is not one, decided when the
//! mesh is set and never in between, because a position edit cannot change whether a mesh is
//! closed. Nothing is cached and nothing can go stale.
//!
//! **Red is more than open.** A closed mesh with a zero-length edge, a collapsed face or two
//! vertices in one place is not a solid either, and neither is one whose limit surface passes
//! through itself. So the body also holds the geometry faults, found whenever the mesh is set or
//! refreshed, and the self-intersection verdict, which is expensive and is handed in by whoever
//! ran it -- tagged with the fingerprint it was computed for, so a stale one is refused. The
//! promoted cage is given out only while none of those say red.

use crate::validate::{self, Check, Fault, Thresholds};
use crate::{Cage, CageError, FaceId, HalfEdgeId, QuadMesh, VertexId};
use nalgebra::{Point3, Vector3};

/// The document's cage: a closed mesh that is a [`Cage`], or an open one that is not yet.
#[derive(Clone, Debug)]
pub struct CageBody {
    state: State,
    /// The red geometry faults of the mesh as it stands: zero-length edges, collapsed faces,
    /// coincident vertices. Refreshed with the mesh.
    red: Vec<Fault>,
    /// The self-intersection verdict, for the fingerprint it was computed at.
    surface: Option<(u64, Vec<FaceId>)>,
    /// How many shells were turned right side out when the mesh was set, said once in the info
    /// line.
    flipped: usize,
    /// A tool's lift is pending on this mesh, so its geometry is not judged yet: an extrude shows
    /// its cap before it has been pulled, and at that moment every wall is a quad of no height,
    /// which the red checks would call collapsed and refuse to draw. The field is finite there
    /// (measured in the field crate's tests), so the draft is drawn, and the faults are read when
    /// the tool is kept or the mesh replaced.
    draft: bool,
}

#[derive(Clone, Debug)]
enum State {
    Closed(Cage),
    Open(QuadMesh),
}

impl CageBody {
    /// Holds `mesh`, every shell turned right side out, promoted to a cage if it is closed, and
    /// its geometry checked.
    pub fn new(mesh: QuadMesh) -> Self {
        let (mesh, flipped) = oriented(mesh);
        let red = validate::red_faults(&mesh);
        let state = match Cage::try_from_mesh(mesh) {
            Ok(cage) => State::Closed(cage),
            Err((mesh, _)) => State::Open(mesh),
        };
        Self { state, red, surface: None, flipped, draft: false }
    }

    /// Stops judging the geometry until [`Self::end_draft`] or [`Self::set_mesh`]: the red and
    /// amber geometry faults are withheld, so a lift at nought is drawn rather than refused. The
    /// structural faults -- open, non-manifold, flipped -- are not geometry and stand.
    pub fn begin_draft(&mut self) {
        self.draft = true;
        self.red.clear();
    }

    /// Reads the geometry again, as a kept tool asks.
    pub fn end_draft(&mut self) {
        self.draft = false;
        self.refresh();
    }

    pub fn is_draft(&self) -> bool {
        self.draft
    }

    /// The edited mesh, whatever its state.
    pub fn mesh(&self) -> &QuadMesh {
        match &self.state {
            State::Closed(cage) => cage.mesh(),
            State::Open(mesh) => mesh,
        }
    }

    /// The mesh, for geometry edits. `QuadMesh` exposes no topology mutator, so nothing reached
    /// this way can open a closed cage or close an open one; a topology edit goes through
    /// [`Self::set_mesh`]. A move can make a geometry fault or mend one, and can turn a shell
    /// inside out, so [`Self::refresh`] is asked once the gesture is over.
    pub fn mesh_mut(&mut self) -> &mut QuadMesh {
        match &mut self.state {
            State::Closed(cage) => cage,
            State::Open(mesh) => mesh,
        }
    }

    /// The promoted cage — `None` while the mesh is open, or while a red fault stands against
    /// it. **A self-intersection not yet checked does not withhold it**: the surface follows a
    /// drag live and the check answers after release (R14).
    pub fn cage(&self) -> Option<&Cage> {
        match &self.state {
            State::Closed(cage) if self.red.is_empty() && !self.surface_crosses() => Some(cage),
            _ => None,
        }
    }

    /// Whether the mesh is closed, red or not: what Refine and the self-intersection check ask,
    /// since both need a closed mesh and neither cares about the verdict.
    pub fn closed(&self) -> Option<&Cage> {
        match &self.state {
            State::Closed(cage) => Some(cage),
            State::Open(_) => None,
        }
    }

    /// Replaces the mesh after a topology edit, promoting the new one if it is closed. The
    /// surface verdict goes with the old mesh.
    pub fn set_mesh(&mut self, mesh: QuadMesh) {
        *self = Self::new(mesh);
    }

    /// Re-reads the geometry faults and the shells' orientation after a position edit, which
    /// cannot change whether the mesh is closed but can change everything else. The surface
    /// verdict is kept where the mesh has not moved, and dropped where it has.
    pub fn refresh(&mut self) {
        let surface = self.surface.take();
        let draft = self.draft;
        let mesh = std::mem::replace(self, Self::new(QuadMesh::placeholder())).into_mesh();
        *self = Self::new(mesh);
        if let Some((fp, faces)) = surface {
            self.set_surface_check(fp, faces);
        }
        // A position edit under a pending tool is the tool's drag: still the draft.
        if draft {
            self.begin_draft();
        }
    }

    pub fn into_mesh(self) -> QuadMesh {
        match self.state {
            State::Closed(cage) => cage.into_mesh(),
            State::Open(mesh) => mesh,
        }
    }

    /// Hands in the self-intersection check's answer for the mesh at `fingerprint`. Ignored,
    /// and `false`, for any other fingerprint: a result for a mesh that has moved on says nothing
    /// about the one here.
    pub fn set_surface_check(&mut self, fingerprint: u64, crossing: Vec<FaceId>) -> bool {
        if self.mesh().geometry_fingerprint() != fingerprint {
            return false;
        }
        self.surface = Some((fingerprint, crossing));
        true
    }

    /// Whether the self-intersection check has yet to answer for the mesh as it stands -- the
    /// "checking…" state. Only a closed mesh is ever checked; an open one is red already.
    pub fn surface_unchecked(&self) -> bool {
        matches!(self.state, State::Closed(_))
            && !matches!(&self.surface, Some((fp, _)) if *fp == self.mesh().geometry_fingerprint())
    }

    fn surface_crosses(&self) -> bool {
        matches!(&self.surface, Some((fp, faces)) if *fp == self.mesh().geometry_fingerprint() && !faces.is_empty())
    }

    /// How many shells were turned right side out when the mesh was last set.
    pub fn shells_flipped(&self) -> usize {
        self.flipped
    }

    /// Why this body is not a cage: the first unmatched edge, as `Cage::from_faces` reports it.
    pub fn first_error(&self) -> Option<CageError> {
        let mesh = self.mesh();
        (0..mesh.half_edge_count())
            .map(HalfEdgeId::from_index)
            .find(|&he| mesh.is_boundary(he))
            .map(|he| CageError::UnmatchedEdge { from: mesh.half_edge(he).origin, to: mesh.destination_of(he) })
    }

    /// The boundary loops of an open body: how many holes it has, and how many edges each rim
    /// has. Empty for a closed mesh. Holes are counted as loops rather than edges because "1 hole,
    /// 8 edges" is what a person is told.
    pub fn holes(&self) -> Vec<usize> {
        self.rims().into_iter().map(|rim| rim.len()).collect()
    }

    /// Each hole's rim as its edges, in the boundary's order.
    pub fn rims(&self) -> Vec<Vec<(VertexId, VertexId)>> {
        let mesh = self.mesh();
        let boundary: Vec<HalfEdgeId> =
            (0..mesh.half_edge_count()).map(HalfEdgeId::from_index).filter(|&he| mesh.is_boundary(he)).collect();
        let by_origin: std::collections::HashMap<VertexId, HalfEdgeId> =
            boundary.iter().map(|&he| (mesh.half_edge(he).origin, he)).collect();
        let mut seen = std::collections::HashSet::new();
        let mut loops = Vec::new();
        for &start in &boundary {
            if !seen.insert(start) {
                continue;
            }
            let mut rim = vec![(mesh.half_edge(start).origin, mesh.destination_of(start))];
            let mut he = start;
            while let Some(&next) = by_origin.get(&mesh.destination_of(he)) {
                if next == start || !seen.insert(next) {
                    break;
                }
                rim.push((mesh.half_edge(next).origin, mesh.destination_of(next)));
                he = next;
            }
            loops.push(rim);
        }
        loops
    }

    /// Every red fault, the open rim first: what the checker lists and the overlay paints.
    pub fn red_faults(&self) -> Vec<Fault> {
        let mut out = Vec::new();
        let rims = self.rims();
        if !rims.is_empty() {
            out.push(Fault {
                check: Check::Open,
                vertices: Vec::new(),
                edges: rims.into_iter().flatten().collect(),
                faces: Vec::new(),
                measure: Some(self.holes().len() as f64),
            });
        }
        out.extend(self.red.iter().cloned());
        if let Some((fp, faces)) = &self.surface {
            if *fp == self.mesh().geometry_fingerprint() && !faces.is_empty() {
                out.push(Fault {
                    check: Check::SelfIntersection,
                    vertices: Vec::new(),
                    edges: Vec::new(),
                    faces: faces.clone(),
                    measure: Some(faces.len() as f64),
                });
            }
        }
        out
    }

    /// The amber faults under `thresholds`.
    pub fn amber_faults(&self, thresholds: &Thresholds) -> Vec<Fault> {
        if self.draft {
            return Vec::new();
        }
        validate::amber_faults(self.mesh(), thresholds)
    }

    /// The error as a person reads it: "open: 1 hole, 8 edges", or the first other red fault.
    pub fn describe_error(&self) -> Option<String> {
        let holes = self.holes();
        if !holes.is_empty() {
            let edges: usize = holes.iter().sum();
            let s = |n: usize, word: &str| if n == 1 { format!("1 {word}") } else { format!("{n} {word}s") };
            return Some(format!("open: {}, {}", s(holes.len(), "hole"), s(edges, "edge")));
        }
        if let Some(fault) = self.red.first() {
            let n = fault.edges.len().max(fault.faces.len()).max(fault.vertices.len() / 2);
            return Some(if n > 1 { format!("{} ×{n}", fault.check.label()) } else { fault.check.label().to_string() });
        }
        if self.surface_crosses() {
            let n = self.surface.as_ref().map_or(0, |(_, f)| f.len());
            return Some(format!("self-intersecting surface on {n} face{}", if n == 1 { "" } else { "s" }));
        }
        None
    }
}

impl QuadMesh {
    /// The smallest mesh there is, for a slot that is about to be refilled.
    fn placeholder() -> QuadMesh {
        QuadMesh::from_faces(
            vec![Point3::origin(), Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            &[vec![0, 1, 2, 3]],
        )
        .expect("one quad")
    }
}

/// The mesh with every shell turned to face outward -- or inward, for a shell inside another,
/// which makes a hollow -- and how many were turned.
///
/// Consistent winding is structural *within* a shell, but a whole shell can be wound the wrong
/// way, and the field would then read negative outside it. The expected sign of a shell's volume
/// is positive at even nesting depth and negative at odd: an inner shell faces into the hollow.
/// Unambiguous, so fixed rather than shown red. Nesting is read off the cage itself by ray parity
/// against the other shells' quads, which is enough for a question about containment. An open
/// shell has no volume to speak of and is left as it came.
fn oriented(mesh: QuadMesh) -> (QuadMesh, usize) {
    let shells = shells_of(&mesh);
    let mut flip = vec![false; shells.len()];
    let mut flipped = 0;
    for (s, faces) in shells.iter().enumerate() {
        if faces.iter().any(|&f| mesh.face_half_edges(f).iter().any(|&he| mesh.is_boundary(he))) {
            continue;
        }
        let volume = signed_volume(&mesh, faces);
        let sample = mesh.vertex_position(mesh.face_vertices(faces[0])[0]);
        let depth = shells.iter().enumerate().filter(|&(o, other)| o != s && contains(&mesh, other, sample)).count();
        let wants_positive = depth % 2 == 0;
        if volume != 0.0 && (volume > 0.0) != wants_positive {
            flip[s] = true;
            flipped += 1;
        }
    }
    if flipped == 0 {
        return (mesh, 0);
    }
    let positions: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| mesh.vertex_position(v)).collect();
    let mut faces: Vec<Vec<u32>> =
        mesh.face_ids().map(|f| mesh.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
    for (s, shell) in shells.iter().enumerate() {
        if flip[s] {
            for &f in shell {
                faces[f.index()].reverse();
            }
        }
    }
    (QuadMesh::from_faces(positions, &faces).expect("reversing whole shells keeps a mesh valid"), flipped)
}

/// The faces of each connected shell.
pub(crate) fn shells_of(mesh: &QuadMesh) -> Vec<Vec<FaceId>> {
    let mut shell_of = vec![usize::MAX; mesh.face_count()];
    let mut shells: Vec<Vec<FaceId>> = Vec::new();
    for start in mesh.face_ids() {
        if shell_of[start.index()] != usize::MAX {
            continue;
        }
        let s = shells.len();
        let mut faces = Vec::new();
        let mut stack = vec![start];
        shell_of[start.index()] = s;
        while let Some(f) = stack.pop() {
            faces.push(f);
            for he in mesh.face_half_edges(f) {
                let twin = mesh.half_edge(he).twin;
                if twin == he {
                    continue;
                }
                let g = mesh.half_edge(twin).face;
                if shell_of[g.index()] == usize::MAX {
                    shell_of[g.index()] = s;
                    stack.push(g);
                }
            }
        }
        shells.push(faces);
    }
    shells
}

/// Six times the signed volume enclosed by `faces`, by the divergence theorem over each quad split
/// along its shorter diagonal. Only the sign is read.
pub(crate) fn signed_volume(mesh: &QuadMesh, faces: &[FaceId]) -> f64 {
    let mut six = 0.0;
    for &f in faces {
        let [a, b, c, d] = mesh.face_vertices(f).map(|v| mesh.vertex_position(v));
        let tris =
            if (c - a).norm_squared() <= (d - b).norm_squared() { [[a, b, c], [a, c, d]] } else { [[a, b, d], [b, c, d]] };
        for [p, q, r] in tris {
            six += p.coords.dot(&q.coords.cross(&r.coords));
        }
    }
    six
}

/// Whether `point` is inside the shell `faces`, by the parity of a ray's crossings of its quads.
/// The ray leaves along a direction no edge is likely to lie along exactly.
fn contains(mesh: &QuadMesh, faces: &[FaceId], point: Point3<f64>) -> bool {
    let dir = Vector3::new(0.577_215_664_9, 0.301_029_995_7, 0.760_832_8).normalize();
    let mut crossings = 0;
    for &f in faces {
        let [a, b, c, d] = mesh.face_vertices(f).map(|v| mesh.vertex_position(v));
        for [p, q, r] in [[a, b, c], [a, c, d]] {
            if ray_hits_triangle(point, dir, p, q, r) {
                crossings += 1;
            }
        }
    }
    crossings % 2 == 1
}

fn ray_hits_triangle(o: Point3<f64>, dir: Vector3<f64>, a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> bool {
    let (e1, e2) = (b - a, c - a);
    let h = dir.cross(&e2);
    let det = e1.dot(&h);
    if det.abs() < 1e-14 {
        return false;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = inv * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(&e1);
    let v = inv * dir.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    inv * e2.dot(&q) > 1e-12
}

impl From<Cage> for CageBody {
    fn from(cage: Cage) -> Self {
        Self::new(cage.into_mesh())
    }
}

impl std::ops::Deref for CageBody {
    type Target = QuadMesh;
    fn deref(&self) -> &QuadMesh {
        self.mesh()
    }
}

/// Written as the mesh is — the same positions and quads — and read back through
/// [`CageBody::new`], so an open cage in a file loads as a red one rather than refusing the
/// document, while a file violating the structural invariants is still refused.
impl serde::Serialize for CageBody {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.mesh().serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for CageBody {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        QuadMesh::deserialize(d).map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_parts() -> (Vec<Point3<f64>>, Vec<Vec<u32>>) {
        let cube = Cage::cube_of(10.0);
        let positions = cube.vertex_ids().map(|v| cube.vertex_position(v)).collect();
        let faces =
            cube.face_ids().map(|f| cube.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
        (positions, faces)
    }

    fn open_cube() -> QuadMesh {
        let (positions, faces) = cube_parts();
        QuadMesh::from_faces(positions, &faces[1..]).unwrap()
    }

    /// **Promotion is decided when the mesh is set, and a position edit cannot change it.**
    #[test]
    fn a_body_is_a_cage_exactly_when_its_mesh_is_closed() {
        let mut body = CageBody::new(Cage::cube().mesh().clone());
        assert!(body.cage().is_some());
        body.mesh_mut().set_vertex_position(VertexId::from_index(0), Point3::new(9.0, 9.0, 9.0));
        assert!(body.cage().is_some(), "moving a point does not open a cage");
        assert_eq!(body.cage().unwrap().vertex_position(VertexId::from_index(0)).x, 9.0);

        body.set_mesh(open_cube());
        assert!(body.cage().is_none(), "a face gone is a red state");
        assert_eq!(body.holes(), vec![4]);
        assert_eq!(body.describe_error().as_deref(), Some("open: 1 hole, 4 edges"));
        assert!(matches!(body.first_error(), Some(CageError::UnmatchedEdge { .. })));
        assert_eq!(body.red_faults()[0].check, Check::Open);
        assert_eq!(body.red_faults()[0].edges.len(), 4);
    }

    /// An open cage in a file loads red instead of refusing the document (R16).
    #[test]
    fn an_open_cage_round_trips_through_json_as_a_red_body() {
        let body = CageBody::new(open_cube());
        let json = serde_json::to_string(&body).unwrap();
        let back: CageBody = serde_json::from_str(&json).unwrap();
        assert!(back.cage().is_none());
        assert_eq!(back.face_count(), 5);
        assert_eq!(back.geometry_fingerprint(), body.geometry_fingerprint());

        let closed: CageBody = serde_json::from_str(&serde_json::to_string(&Cage::cube()).unwrap()).unwrap();
        assert!(closed.cage().is_some(), "a closed file is a cage, as before");
        assert!(serde_json::from_str::<CageBody>(r#"{"positions":[[0,0,0],[1,0,0],[1,1,0]],"faces":[[0,1,2]]}"#).is_err());
    }

    /// Two holes are two loops, whatever their edge counts.
    #[test]
    fn holes_are_counted_as_loops_not_edges() {
        let (positions, faces) = cube_parts();
        let tube: Vec<Vec<u32>> = faces[2..].to_vec();
        let body = CageBody::new(QuadMesh::from_faces(positions, &tube).unwrap());
        assert_eq!(body.holes(), vec![4, 4], "-Z and +Z gone: an open tube");
        assert_eq!(body.describe_error().as_deref(), Some("open: 2 holes, 8 edges"));
    }

    /// **A closed mesh with a geometry fault is red too**, and mending the fault by a move brings
    /// the cage back once the body is refreshed.
    #[test]
    fn a_zero_length_edge_makes_a_closed_mesh_red_until_it_is_mended() {
        let (mut positions, faces) = cube_parts();
        positions[1] = positions[0];
        let mut body = CageBody::new(QuadMesh::from_faces(positions, &faces).unwrap());
        assert!(body.closed().is_some(), "closed");
        assert!(body.cage().is_none(), "and not a solid");
        assert_eq!(body.describe_error().as_deref(), Some("zero-length edge"));

        body.mesh_mut().set_vertex_position(VertexId::from_index(1), Point3::new(5.0, -5.0, -5.0));
        assert!(body.cage().is_none(), "a move is read on refresh, not on the keystroke");
        body.refresh();
        assert!(body.cage().is_some());
        assert_eq!(body.describe_error(), None);
    }

    /// **A lift at nought is red until it is a draft, and a draft until the tool is kept.** The
    /// extrude session's first state is four walls of no height; the body withholds the geometry
    /// faults while the session is pending, through the drag's own refreshes, and reads them
    /// again when it ends -- so a cap pulled to a height keeps, and one left at nought is red.
    #[test]
    fn a_pending_lift_is_drawn_as_a_draft_and_judged_when_kept() {
        use crate::edit::extrude::{extrude, Direction, Mode};
        let cube = Cage::cube_of(10.0);
        let top = cube.face_ids().find(|&f| cube.face_vertices(f).iter().all(|&v| cube.vertex_position(v).z > 0.0)).unwrap();
        let lifted = extrude(cube.mesh(), &[top], Mode::Region, Direction::Normal, 0.0).unwrap();
        let cap: Vec<VertexId> = lifted.new_vertices.clone();
        let mut body = CageBody::new(lifted.mesh);
        assert!(body.cage().is_none(), "walls of no height are collapsed faces");
        assert_eq!(body.describe_error().as_deref(), Some("zero-length edge ×4"));

        body.begin_draft();
        assert!(body.cage().is_some(), "a draft is drawn");
        assert!(body.red_faults().is_empty() && body.amber_faults(&Thresholds::default()).is_empty());
        body.refresh();
        assert!(body.is_draft() && body.cage().is_some(), "a drag's refresh keeps the draft");

        body.end_draft();
        assert!(body.cage().is_none(), "kept at nought, it is judged: red");

        body.begin_draft();
        for &v in &cap {
            let p = body.mesh().vertex_position(v);
            body.mesh_mut().set_vertex_position(v, p + Vector3::new(0.0, 0.0, 3.0));
        }
        body.end_draft();
        assert!(body.cage().is_some(), "pulled to a height, it is a solid");
        assert!(!body.is_draft());
        body.set_mesh(cube.mesh().clone());
        assert!(!body.is_draft(), "a new mesh is never a draft");
    }

    /// **An inside-out cube promotes, flipped, and says so once** (the notes' test 5): the field
    /// reads negative at the centre whatever winding the file had.
    #[test]
    fn an_inside_out_cube_is_turned_right_side_out_on_promotion() {
        let (positions, mut faces) = cube_parts();
        for f in &mut faces {
            f.reverse();
        }
        let body = CageBody::new(QuadMesh::from_faces(positions, &faces).unwrap());
        assert_eq!(body.shells_flipped(), 1);
        let cage = body.cage().expect("a solid");
        assert!(signed_volume(cage, &cage.face_ids().collect::<Vec<_>>()) > 0.0);
        assert_eq!(CageBody::new(Cage::cube().mesh().clone()).shells_flipped(), 0, "a right-way cube is left alone");
    }

    /// **A cube inside a larger cube is a hollow, not a union, whatever the file's winding** (the
    /// notes' hollow case): the inner shell ends up facing into the hollow.
    #[test]
    fn a_shell_inside_another_faces_inward_whatever_winding_it_arrived_with() {
        let outer = Cage::cube_of(20.0);
        let inner = Cage::cube_of(10.0);
        for inner_reversed in [false, true] {
            let mut positions: Vec<Point3<f64>> = outer.vertex_ids().map(|v| outer.vertex_position(v)).collect();
            let mut faces: Vec<Vec<u32>> =
                outer.face_ids().map(|f| outer.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
            positions.extend(inner.vertex_ids().map(|v| inner.vertex_position(v)));
            for f in inner.face_ids() {
                let mut q: Vec<u32> = inner.face_vertices(f).iter().map(|v| v.index() as u32 + 8).collect();
                if inner_reversed {
                    q.reverse();
                }
                faces.push(q);
            }
            let body = CageBody::new(QuadMesh::from_faces(positions, &faces).unwrap());
            let mesh = body.mesh();
            let shells = shells_of(mesh);
            assert_eq!(shells.len(), 2);
            let (outer_faces, inner_faces) =
                if shells[0].contains(&FaceId::from_index(0)) { (&shells[0], &shells[1]) } else { (&shells[1], &shells[0]) };
            assert!(signed_volume(mesh, outer_faces) > 0.0, "the outer shell faces out");
            assert!(
                signed_volume(mesh, inner_faces) < 0.0,
                "the inner shell faces into the hollow (reversed in the file: {inner_reversed})"
            );
            assert_eq!(body.shells_flipped(), if inner_reversed { 0 } else { 1 });
        }
    }

    /// A verdict is kept only for the mesh it was computed on.
    #[test]
    fn a_surface_verdict_is_refused_for_a_mesh_that_has_moved_on() {
        let mut body = CageBody::new(Cage::cube().mesh().clone());
        assert!(body.surface_unchecked());
        let stale = body.geometry_fingerprint();
        body.mesh_mut().set_vertex_position(VertexId::from_index(0), Point3::new(2.0, 2.0, 2.0));
        assert!(!body.set_surface_check(stale, vec![FaceId::from_index(0)]), "stale, discarded");
        assert!(body.surface_unchecked() && body.cage().is_some());
        let now = body.geometry_fingerprint();
        assert!(body.set_surface_check(now, vec![FaceId::from_index(0)]));
        assert!(body.cage().is_none(), "red while the surface crosses");
        assert!(body.describe_error().unwrap().starts_with("self-intersecting"));
        assert!(body.set_surface_check(now, Vec::new()));
        assert!(body.cage().is_some());
    }
}
