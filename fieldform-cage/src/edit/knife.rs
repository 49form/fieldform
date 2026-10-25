// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The knife: a drawn path becomes new edges, and every face stays a quad (the cage-editing
//! notes, item 76).
//!
//! A cut point on an edge gives each face either side of it an odd number of boundary points,
//! and no quad tiling of an odd-sided patch exists -- so where the path does not carry a point
//! onward, **a front runs on from it** along the edge ring through face after face, as Insert
//! Edge does, until it reaches a boundary, closes on itself, or meets another front of the same
//! cut, where the two join (a straight edge if they meet head-on, a corner if side-on). Fronts
//! advance together, one face a step, so where they meet does not depend on which was seeded
//! first. Nothing is applied until every touched face has been matched to a pattern; a face no
//! pattern fits refuses the whole cut, with the face named.
//!
//! A cut from a corner to anything but a neighbouring point along its own edge is refused: the
//! vertex it would make is shared by two faces, and the surface folds about five degrees there
//! (`valence_two` in the subdivision crate measured it), which is a crease where the user drew a
//! straight line.

use super::{lerp, rebuild, EditError, EditResult, Highlight};
use crate::validate::epsilon_length;
use crate::{CageError, FaceId, HalfEdgeId, QuadMesh, VertexId};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// A point of a cut path: on a vertex, or on an edge at `t` from `edge.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CutPoint {
    Vertex(VertexId),
    Edge { edge: (VertexId, VertexId), t: f64 },
}

/// A cut worked out but not applied: what the preview draws.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// The new edges on the drawn path, as positions.
    pub path_edges: Vec<(Point3<f64>, Point3<f64>)>,
    /// The edges the fronts added, as positions.
    pub front_edges: Vec<(Point3<f64>, Point3<f64>)>,
    /// Every point the cut places, as positions: the drawn ones and the fronts'.
    pub points: Vec<Point3<f64>>,
    /// The faces whose pattern is built, with the quads each becomes, in the mesh's own indices
    /// plus new vertices numbered from `positions.len()` of the mesh in `new_positions` order.
    faces: Vec<(FaceId, Vec<Vec<u32>>)>,
    new_positions: Vec<Point3<f64>>,
    /// New edges on the drawn path, in the indices above.
    cut_edges: Vec<(u32, u32)>,
    /// The pattern each touched face resolved to, in face order.
    pub patterns: Vec<(FaceId, Pattern)>,
}

/// How a face is split, from the notes' table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    /// A: one edge across, between opposite edges -- two quads.
    Straight,
    /// B: a turn between adjacent edges, through a new interior vertex -- three quads.
    Corner,
    /// X: two edges across, crossing at a new interior vertex -- four quads.
    Crossing,
    /// S: two edges across the same pair of edges, side by side -- three quads.
    Strip,
}

/// An applied cut.
#[derive(Debug, Clone)]
pub struct Cut {
    pub result: EditResult,
    /// The new edges that lie on the drawn path -- not the fronts' -- so the tool can select them.
    pub cut_edges: Vec<(VertexId, VertexId)>,
}

/// Works the cut out without applying it.
pub fn plan(mesh: &QuadMesh, path: &[CutPoint]) -> Result<Plan, EditError> {
    plan_ordered(mesh, path, false)
}

/// Applies the cut.
pub fn cut(mesh: &QuadMesh, path: &[CutPoint]) -> Result<Cut, EditError> {
    let plan = plan(mesh, path)?;
    apply(mesh, plan)
}

/// `plan`, with the fronts advanced in the reverse of their fixed order within a step. The result
/// must be the same mesh; a test holds it so.
pub(crate) fn plan_ordered(mesh: &QuadMesh, path: &[CutPoint], reverse_fronts: bool) -> Result<Plan, EditError> {
    let eps = epsilon_length(mesh);
    let points = normalise(mesh, path, eps)?;
    let mut work = Work::new(mesh, eps);
    work.draw(&points)?;
    work.propagate(reverse_fronts)?;
    work.resolve()
}

fn apply(mesh: &QuadMesh, plan: Plan) -> Result<Cut, EditError> {
    let mut positions = mesh.position_list();
    let original_vertices = positions.len();
    positions.extend(plan.new_positions.iter().copied());
    let mut faces = mesh.quad_lists();
    let original_faces = faces.len();
    let mut replaced = HashMap::new();
    for (face, quads) in &plan.faces {
        let mut slots = Vec::with_capacity(quads.len());
        for (i, quad) in quads.iter().enumerate() {
            if i == 0 {
                faces[face.index()] = quad.clone();
                slots.push(*face);
            } else {
                faces.push(quad.clone());
                slots.push(FaceId::new(faces.len() - 1));
            }
        }
        replaced.insert(*face, slots);
    }
    let new_vertex_count = positions.len();
    let new_face_count = faces.len();
    let result = EditResult {
        mesh: rebuild("cut", positions, &faces)?,
        new_vertices: (original_vertices..new_vertex_count).map(VertexId::new).collect(),
        new_faces: (original_faces..new_face_count).map(FaceId::new).collect(),
        replaced,
        vertex_map: None,
        face_map: None,
    };
    let cut_edges = plan.cut_edges.iter().map(|&(a, b)| (VertexId::new(a as usize), VertexId::new(b as usize))).collect();
    Ok(Cut { result, cut_edges })
}

// ---------------------------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------------------------

/// Where a point sits, in the mesh's own terms: a vertex, or an edge `(a, b)` with `a < b` and `s`
/// from `a`. The canonical spelling, so a point two faces share is one point.
#[derive(Debug, Clone, Copy, PartialEq)]
enum On {
    Vertex(VertexId),
    Edge { a: VertexId, b: VertexId, s: f64 },
}

/// A point within `ε` of a vertex becomes that vertex, and no piece of an edge shorter than `ε` is
/// made (the notes §1.1).
fn normalise(mesh: &QuadMesh, path: &[CutPoint], eps: f64) -> Result<Vec<On>, EditError> {
    let mut out = Vec::with_capacity(path.len());
    for (i, p) in path.iter().enumerate() {
        let on = match *p {
            CutPoint::Vertex(v) => {
                if v.index() >= mesh.vertex_count() {
                    return Err(EditError::refuse(format!("point {} names vertex {}, which the mesh does not have", i + 1, v.index()), Highlight::default()));
                }
                On::Vertex(v)
            }
            CutPoint::Edge { edge: (a, b), t } => {
                if mesh.directed(a, b).is_none() && mesh.directed(b, a).is_none() {
                    return Err(EditError::refuse(
                        format!("point {}: no edge joins vertices {} and {}", i + 1, a.index(), b.index()),
                        Highlight::default(),
                    ));
                }
                let len = (mesh.vertex_position(a) - mesh.vertex_position(b)).norm();
                if !t.is_finite() || t * len < eps {
                    On::Vertex(a)
                } else if (1.0 - t) * len < eps {
                    On::Vertex(b)
                } else {
                    canonical(a, b, t)
                }
            }
        };
        // The same point twice in a row is one point.
        if out.last().is_some_and(|last| same(last, &on)) {
            continue;
        }
        out.push(on);
    }
    Ok(out)
}

fn canonical(a: VertexId, b: VertexId, s_from_a: f64) -> On {
    if a.index() < b.index() {
        On::Edge { a, b, s: s_from_a }
    } else {
        On::Edge { a: b, b: a, s: 1.0 - s_from_a }
    }
}

fn same(p: &On, q: &On) -> bool {
    match (p, q) {
        (On::Vertex(v), On::Vertex(w)) => v == w,
        (On::Edge { a, b, s }, On::Edge { a: c, b: d, s: r }) => a == c && b == d && (s - r).abs() < 1e-9,
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------
// The work: points, the segments per face, the fronts
// ---------------------------------------------------------------------------------------------

/// A point's place on one face's boundary: corner `i`, or on edge `i` (`ci → ci+1`) at `s` from
/// `ci`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Loc {
    Corner(usize),
    Edge(usize, f64),
}

/// One new edge inside a face, between two of its boundary points.
#[derive(Debug, Clone, Copy)]
struct Segment {
    a: usize,
    b: usize,
    /// Drawn by the user, or laid by a front.
    drawn: bool,
}

/// A front's passage through one face: in through `entry` at point `entry_pt`, and out through
/// the opposite edge at `exit_pt` -- or, where it joined another front, no exit.
#[derive(Debug, Clone, Copy)]
struct Visit {
    face: FaceId,
    entry: HalfEdgeId,
    entry_pt: usize,
    exit_pt: Option<usize>,
}

#[derive(Debug, Clone)]
struct Front {
    /// The half-edge the front next enters through, and its `s` from that half-edge's origin.
    entry: HalfEdgeId,
    s: f64,
    /// The point it is standing on: the entry point of its next visit.
    point: usize,
    start: usize,
    visits: Vec<Visit>,
    alive: bool,
}

struct Work<'m> {
    mesh: &'m QuadMesh,
    eps: f64,
    points: Vec<On>,
    /// The faces the drawn path crosses, with the segments drawn in each.
    drawn: HashMap<FaceId, Vec<Segment>>,
    fronts: Vec<Front>,
}

impl<'m> Work<'m> {
    fn new(mesh: &'m QuadMesh, eps: f64) -> Self {
        Self { mesh, eps, points: Vec::new(), drawn: HashMap::new(), fronts: Vec::new() }
    }

    /// The index of `on` among the points, added if new, and whether it was there already.
    fn point(&mut self, on: On) -> (usize, bool) {
        if let Some(i) = self.points.iter().position(|p| same(p, &on)) {
            return (i, true);
        }
        self.points.push(on);
        (self.points.len() - 1, false)
    }

    fn faces_of(&self, on: On) -> Vec<FaceId> {
        match on {
            On::Vertex(v) => self.mesh.adjacent_faces(v),
            On::Edge { a, b, .. } => [self.mesh.directed(a, b), self.mesh.directed(b, a)]
                .into_iter()
                .flatten()
                .map(|he| self.mesh.half_edges[he.index()].face)
                .collect(),
        }
    }

    /// Where `on` sits on `face`'s boundary, if it does.
    fn locate(&self, face: FaceId, on: On) -> Option<Loc> {
        let c = self.mesh.face_vertices(face);
        match on {
            On::Vertex(v) => c.iter().position(|&x| x == v).map(Loc::Corner),
            On::Edge { a, b, s } => (0..4).find_map(|i| {
                let (p, q) = (c[i], c[(i + 1) % 4]);
                if p == a && q == b {
                    Some(Loc::Edge(i, s))
                } else if p == b && q == a {
                    Some(Loc::Edge(i, 1.0 - s))
                } else {
                    None
                }
            }),
        }
    }

    /// Reads the drawn path into segments, one per pair of consecutive points in the face they
    /// share, and seeds a front from each end that lies on an edge.
    fn draw(&mut self, path: &[On]) -> Result<(), EditError> {
        if path.len() < 2 {
            return Err(EditError::refuse("a cut needs two points", Highlight::default()));
        }
        let ids: Vec<usize> = path.iter().map(|&on| self.point(on).0).collect();
        let mut drawn_any = false;
        // The face each end's segment lies in: a front leaves each end on its other side.
        let mut end_faces: [Option<(usize, FaceId)>; 2] = [None, None];
        for w in ids.windows(2) {
            let (p, q) = (w[0], w[1]);
            let (on_p, on_q) = (self.points[p], self.points[q]);
            // The one face both points lie on. Two edge points on one edge share two faces and
            // lie on an existing edge: dropped below, as is a corner and a point on its edge.
            let shared: Vec<FaceId> = self.faces_of(on_p).into_iter().filter(|f| self.faces_of(on_q).contains(f)).collect();
            let Some(&face) = shared.first() else {
                return Err(EditError::refuse(
                    format!("points {} and {} of the cut don't share a face", p + 1, q + 1),
                    highlight_of(&[on_p, on_q]),
                ));
            };
            let (lp, lq) = (self.locate(face, on_p).expect("shared"), self.locate(face, on_q).expect("shared"));
            match (lp, lq) {
                // Along an existing edge: not a cut.
                (Loc::Corner(i), Loc::Corner(j)) if (i + 1) % 4 == j || (j + 1) % 4 == i => continue,
                (Loc::Corner(i), Loc::Edge(j, _)) | (Loc::Edge(j, _), Loc::Corner(i)) if j == i || (j + 1) % 4 == i => continue,
                (Loc::Edge(i, _), Loc::Edge(j, _)) if i == j => continue,
                (Loc::Corner(_), _) | (_, Loc::Corner(_)) => {
                    return Err(EditError::refuse(
                        "a diagonal cut makes a point the surface can't pass smoothly \u{2014} cut to an edge instead",
                        Highlight { faces: vec![face], ..highlight_of(&[on_p, on_q]) },
                    ));
                }
                _ => {}
            }
            self.drawn.entry(face).or_default().push(Segment { a: p, b: q, drawn: true });
            if end_faces[0].is_none() {
                end_faces[0] = Some((p, face));
            }
            end_faces[1] = Some((q, face));
            drawn_any = true;
        }
        if !drawn_any {
            return Err(EditError::refuse("that cut lies along existing edges", highlight_of(path)));
        }
        // A front from each end on an edge, out through the side its own segment is not on --
        // whether or not the path crosses that side too: a face the path crosses that holds an
        // end's point with no segment at it would be odd, and the front passing through is what
        // makes it even (a crossing or a strip there).
        for (end, side) in end_faces.into_iter().flatten() {
            if let On::Edge { a, b, s } = self.points[end] {
                for he in [self.mesh.directed(a, b), self.mesh.directed(b, a)].into_iter().flatten() {
                    let face = self.mesh.half_edges[he.index()].face;
                    if face == side {
                        continue;
                    }
                    let s_he = if self.mesh.origin(he) == a { s } else { 1.0 - s };
                    self.fronts.push(Front { entry: he, s: s_he, point: end, start: end, visits: Vec::new(), alive: true });
                }
            }
        }
        // A closed drawn loop's two ends are one point with a segment either side: no fronts.
        if ids.len() > 2 && ids[0] == ids[ids.len() - 1] {
            self.fronts.clear();
        }
        Ok(())
    }

    /// Advances every front a face at a time until each has stopped (the notes §2.3).
    fn propagate(&mut self, reverse: bool) -> Result<(), EditError> {
        let limit = self.mesh.face_count() * 2 + 2;
        for _step in 0..limit {
            if !self.fronts.iter().any(|f| f.alive) {
                return Ok(());
            }
            let order: Vec<usize> = if reverse { (0..self.fronts.len()).rev().collect() } else { (0..self.fronts.len()).collect() };
            for i in order {
                if !self.fronts[i].alive {
                    continue;
                }
                self.advance(i)?;
            }
        }
        Err(EditError::bug("cut", CageError::Empty))
    }

    fn advance(&mut self, i: usize) -> Result<(), EditError> {
        let mesh = self.mesh;
        let entry = self.fronts[i].entry;
        let s = self.fronts[i].s;
        let face = mesh.half_edges[entry.index()].face;
        let entry_pt = self.fronts[i].point;
        let opposite = mesh.next(mesh.next(entry));

        // Another front already in this face: they join here, and whatever it did beyond this
        // face is discarded. A face the path crosses is passed through instead (X or S), and so
        // is one this front crossed itself.
        if !self.drawn.contains_key(&face) {
            let other = self
                .fronts
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .find_map(|(j, f)| f.visits.iter().position(|v| v.face == face).map(|k| (j, k)));
            if let Some((j, k)) = other {
                let theirs = self.fronts[j].visits[k].entry;
                let adjacent = mesh.next(entry) == theirs || mesh.next(opposite) == theirs;
                if theirs != opposite && !adjacent {
                    return Err(EditError::refuse(
                        "this cut's loops can't close \u{2014} end it on an edge, or on another cut",
                        Highlight { faces: vec![face], ..Highlight::default() },
                    ));
                }
                self.fronts[j].visits.truncate(k + 1);
                self.fronts[j].visits[k].exit_pt = None;
                self.fronts[j].alive = false;
                self.fronts[i].visits.push(Visit { face, entry, entry_pt, exit_pt: None });
                self.fronts[i].alive = false;
                return Ok(());
            }
        }

        // Straight through: out of the opposite edge at the same `s`, measured as Insert Edge
        // measures it, so the new edge is the straight one a ring makes.
        let (w, x) = (mesh.origin(opposite), mesh.destination(opposite));
        let exit_on = canonical(x, w, s);
        let (exit_pt, existed) = self.point(exit_on);
        self.fronts[i].visits.push(Visit { face, entry, entry_pt, exit_pt: Some(exit_pt) });
        if mesh.is_boundary(opposite) {
            self.fronts[i].alive = false;
            return Ok(());
        }
        if exit_pt == self.fronts[i].start {
            // Closed on itself without meeting its partner: on a closed mesh that is the pair of
            // belts that cross only once, which no pattern resolves.
            self.fronts[i].alive = false;
            return Err(EditError::refuse(
                "this cut's loops can't close \u{2014} end it on an edge, or on another cut",
                Highlight { faces: vec![face], ..Highlight::default() },
            ));
        }
        // Landed on a point the cut already has -- a drawn one, or another front's -- it has
        // joined the cut there: the point carries a segment on each side already.
        if existed {
            self.fronts[i].alive = false;
            return Ok(());
        }
        let next = mesh.twin(opposite);
        let twice = self.fronts[i].visits.iter().filter(|v| v.face == mesh.half_edges[next.index()].face).count();
        if twice >= 2 {
            return Err(EditError::bug("cut", CageError::Empty));
        }
        self.fronts[i].entry = next;
        self.fronts[i].point = exit_pt;
        Ok(())
    }

    /// Matches every touched face to its pattern and builds the plan.
    fn resolve(self) -> Result<Plan, EditError> {
        let mesh = self.mesh;
        // Every face's segments: the drawn ones, then the fronts'.
        let mut per_face: HashMap<FaceId, Vec<Segment>> = self.drawn.clone();
        for front in &self.fronts {
            for v in &front.visits {
                let Some(exit) = v.exit_pt else { continue };
                per_face.entry(v.face).or_default().push(Segment { a: v.entry_pt, b: exit, drawn: false });
            }
            // A join: the last visit of each of the two fronts has no exit, and the pair is one
            // segment between their entry points, laid once.
        }
        for (i, front) in self.fronts.iter().enumerate() {
            let Some(last) = front.visits.last() else { continue };
            if last.exit_pt.is_some() {
                continue;
            }
            // Its partner's last visit is in the same face.
            let partner = self.fronts.iter().enumerate().find(|&(j, f)| {
                j != i && f.visits.last().is_some_and(|v| v.face == last.face && v.exit_pt.is_none())
            });
            if let Some((j, p)) = partner
                && i < j
            {
                let theirs = p.visits.last().expect("checked");
                per_face.entry(last.face).or_default().push(Segment { a: last.entry_pt, b: theirs.entry_pt, drawn: false });
            }
        }

        let position = |on: On| -> Point3<f64> {
            match on {
                On::Vertex(v) => mesh.vertex_position(v),
                On::Edge { a, b, s } => lerp(mesh.vertex_position(a), mesh.vertex_position(b), s),
            }
        };
        // New vertex ids: one per edge point a segment uses, in point order, then the interior
        // ones as made. A point a front placed and then gave up when it joined -- its visits
        // beyond the join were discarded -- is in the pool but on no segment, and gets no vertex.
        let base = mesh.vertex_count() as u32;
        let referenced: std::collections::HashSet<usize> =
            per_face.values().flatten().flat_map(|s| [s.a, s.b]).collect();
        // **Parity, checked rather than trusted.** Every edge point lies on two faces, and each
        // must have a segment ending at it, or that face has an odd number of boundary points
        // and no quad tiling. The fronts are what keep this true; this is what says so if they
        // did not.
        let mut odd: Vec<FaceId> = Vec::new();
        for &i in &referenced {
            if let On::Edge { .. } = self.points[i] {
                for face in self.faces_of(self.points[i]) {
                    let has = per_face.get(&face).is_some_and(|segs| segs.iter().any(|s| s.a == i || s.b == i));
                    if !has && !odd.contains(&face) {
                        odd.push(face);
                    }
                }
            }
        }
        if !odd.is_empty() {
            odd.sort_by_key(|f| f.index());
            return Err(EditError::refuse(
                "this cut needs a non-quad face here \u{2014} move the cut, or insert a loop first",
                Highlight { faces: odd, ..Highlight::default() },
            ));
        }
        // Numbered by where they sit, not by which front placed them first, so the mesh is the
        // same whatever order the fronts advanced in.
        let mut ordered: Vec<usize> = (0..self.points.len())
            .filter(|i| referenced.contains(i) && matches!(self.points[*i], On::Edge { .. }))
            .collect();
        ordered.sort_by(|&i, &j| match (self.points[i], self.points[j]) {
            (On::Edge { a, b, s }, On::Edge { a: c, b: d, s: r }) => {
                (a.index(), b.index()).cmp(&(c.index(), d.index())).then(s.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal))
            }
            _ => std::cmp::Ordering::Equal,
        });
        let mut new_positions: Vec<Point3<f64>> = Vec::new();
        let mut id_of: HashMap<usize, u32> = HashMap::new();
        for i in ordered {
            id_of.insert(i, base + new_positions.len() as u32);
            new_positions.push(position(self.points[i]));
        }
        let vertex_id = |i: usize| -> u32 {
            match self.points[i] {
                On::Vertex(v) => v.index() as u32,
                On::Edge { .. } => id_of[&i],
            }
        };

        let mut plan = Plan::default();
        let mut refused: Vec<FaceId> = Vec::new();
        let mut faces: Vec<(FaceId, Vec<Segment>)> = per_face.into_iter().collect();
        faces.sort_by_key(|(f, _)| f.index());
        for (face, segments) in faces {
            let corners = mesh.face_vertices(face);
            let c = |i: usize| corners[i % 4].index() as u32;
            let cp = |i: usize| mesh.vertex_position(corners[i % 4]);
            let locs: Vec<(Loc, Loc)> = segments
                .iter()
                .map(|s| (self.locate(face, self.points[s.a]).expect("on face"), self.locate(face, self.points[s.b]).expect("on face")))
                .collect();
            let at = |i: usize| position(self.points[i]);
            let mut edge_of = |seg: &Segment, a: u32, b: u32, pa: Point3<f64>, pb: Point3<f64>| {
                if seg.drawn {
                    plan.path_edges.push((pa, pb));
                    plan.cut_edges.push((a, b));
                } else {
                    plan.front_edges.push((pa, pb));
                }
            };
            let quads: Option<(Pattern, Vec<Vec<u32>>)> = match (segments.as_slice(), locs.as_slice()) {
                ([seg], [(Loc::Edge(i, _), Loc::Edge(j, _))]) if (i + 2) % 4 == *j => {
                    // A: straight across.
                    let (p, q) = (vertex_id(seg.a), vertex_id(seg.b));
                    edge_of(seg, p, q, at(seg.a), at(seg.b));
                    let i = *i;
                    Some((Pattern::Straight, vec![vec![c(i), p, q, c(i + 3)], vec![p, c(i + 1), c(i + 2), q]]))
                }
                ([seg], [(Loc::Edge(i, _), Loc::Edge(j, _))]) if (i + 1) % 4 == *j || (j + 1) % 4 == *i => {
                    // B: a corner. `p` on edge i, `q` on edge i+1, sharing corner i+1.
                    let (i, pa, qa) = if (i + 1) % 4 == *j { (*i, seg.a, seg.b) } else { (*j, seg.b, seg.a) };
                    let (p, q) = (vertex_id(pa), vertex_id(qa));
                    let (pp, pq) = (at(pa), at(qa));
                    let m = lerp(pp, pq, 0.5);
                    let far = cp(i + 3);
                    let interior = m + (far - m) / 4.0;
                    let centre = base + new_positions.len() as u32;
                    new_positions.push(interior);
                    plan.points.push(interior);
                    edge_of(seg, p, centre, pp, interior);
                    edge_of(seg, centre, q, interior, pq);
                    Some((
                        Pattern::Corner,
                        vec![
                            vec![p, c(i + 1), q, centre],
                            vec![c(i), p, centre, c(i + 3)],
                            vec![centre, q, c(i + 2), c(i + 3)],
                        ],
                    ))
                }
                // Two segments sharing a point -- a front that landed on a drawn point in the
                // face the point's own segment is in -- leave three points on the boundary,
                // which is odd: no quad tiling has it.
                ([s1, s2], _) if [s1.a, s1.b].contains(&s2.a) || [s1.a, s1.b].contains(&s2.b) => {
                    refused.push(face);
                    None
                }
                ([s1, s2], [(l1a, l1b), (l2a, l2b)]) => {
                    // Two straight segments: crossing (X) or side by side (S).
                    let straight = |a: &Loc, b: &Loc| match (a, b) {
                        (Loc::Edge(i, _), Loc::Edge(j, _)) if (i + 2) % 4 == *j => Some(*i),
                        _ => None,
                    };
                    match (straight(l1a, l1b), straight(l2a, l2b)) {
                        (Some(i1), Some(i2)) if i1 % 2 != i2 % 2 => {
                            // X: `p–q` on edges i, i+2; `r–u` on edges i+1, i+3.
                            let (i, pq, ru) = if (i2 + 4 - i1) % 4 == 1 { (i1, s1, s2) } else { (i2, s2, s1) };
                            let s_of = |pt: usize, edge: usize| match self.locate(face, self.points[pt]).expect("on face") {
                                Loc::Edge(e, s) if e == edge => s,
                                _ => unreachable!("located on the matched edge"),
                            };
                            let (p, q) = (vertex_id(pq.a), vertex_id(pq.b));
                            let (r, u) = (vertex_id(ru.a), vertex_id(ru.b));
                            let (sp, sq) = (s_of(pq.a, i), s_of(pq.b, (i + 2) % 4));
                            let (sr, su) = (s_of(ru.a, (i + 1) % 4), s_of(ru.b, (i + 3) % 4));
                            // In the face's bilinear parameters, `p–q` runs (sp, 0) → (1 − sq, 1)
                            // and `r–u` runs (1, sr) → (0, 1 − su).
                            let (a0, b0, a1, b1) = (sp, 0.0, 1.0 - sq, 1.0);
                            let (a2, b2, a3, b3) = (1.0, sr, 0.0, 1.0 - su);
                            let den = (a1 - a0) * (b3 - b2) - (b1 - b0) * (a3 - a2);
                            if den.abs() < 1e-12 {
                                refused.push(face);
                                None
                            } else {
                                let t = ((a2 - a0) * (b3 - b2) - (b2 - b0) * (a3 - a2)) / den;
                                let (a, b) = (a0 + t * (a1 - a0), b0 + t * (b1 - b0));
                                let bottom = lerp(cp(i), cp(i + 1), a);
                                let top = lerp(cp(i + 3), cp(i + 2), a);
                                let crossing = lerp(bottom, top, b);
                                let centre = base + new_positions.len() as u32;
                                new_positions.push(crossing);
                                plan.points.push(crossing);
                                edge_of(pq, p, centre, at(pq.a), crossing);
                                edge_of(pq, centre, q, crossing, at(pq.b));
                                edge_of(ru, r, centre, at(ru.a), crossing);
                                edge_of(ru, centre, u, crossing, at(ru.b));
                                Some((
                                    Pattern::Crossing,
                                    vec![
                                        vec![c(i), p, centre, u],
                                        vec![p, c(i + 1), r, centre],
                                        vec![centre, r, c(i + 2), q],
                                        vec![u, centre, q, c(i + 3)],
                                    ],
                                ))
                            }
                        }
                        (Some(i1), Some(i2)) if i1 == i2 || (i1 + 2) % 4 == i2 => {
                            // S: both across the same pair of edges, nearer corner i first.
                            let i = i1.min(i2) % 2;
                            let (i, j) = (i, (i + 2) % 4);
                            let on = |pt: usize| match self.locate(face, self.points[pt]).expect("on face") {
                                Loc::Edge(e, s) if e == i => (true, s),
                                Loc::Edge(e, s) if e == j => (false, s),
                                _ => unreachable!("located on the matched edges"),
                            };
                            let ends = |seg: &Segment| {
                                let (pa, pb) = (on(seg.a), on(seg.b));
                                if pa.0 { ((seg.a, pa.1), (seg.b, pb.1)) } else { ((seg.b, pb.1), (seg.a, pa.1)) }
                            };
                            let (mut first, mut second) = (ends(s1), ends(s2));
                            let (mut f_seg, mut s_seg) = (s1, s2);
                            if first.0 .1 > second.0 .1 {
                                std::mem::swap(&mut first, &mut second);
                                std::mem::swap(&mut f_seg, &mut s_seg);
                            }
                            // Edge j runs back toward corner i+3, so the nearer segment's far end
                            // sits further along it; crossing segments are not a strip.
                            if first.1 .1 <= second.1 .1 {
                                refused.push(face);
                                None
                            } else {
                                let (p1, q1) = (vertex_id(first.0 .0), vertex_id(first.1 .0));
                                let (p2, q2) = (vertex_id(second.0 .0), vertex_id(second.1 .0));
                                edge_of(f_seg, p1, q1, at(first.0 .0), at(first.1 .0));
                                edge_of(s_seg, p2, q2, at(second.0 .0), at(second.1 .0));
                                Some((
                                    Pattern::Strip,
                                    vec![vec![c(i), p1, q1, c(i + 3)], vec![p1, p2, q2, q1], vec![p2, c(i + 1), c(i + 2), q2]],
                                ))
                            }
                        }
                        _ => {
                            refused.push(face);
                            None
                        }
                    }
                }
                _ => {
                    refused.push(face);
                    None
                }
            };
            if let Some((pattern, quads)) = quads {
                plan.faces.push((face, quads));
                plan.patterns.push((face, pattern));
            }
        }
        if !refused.is_empty() {
            return Err(EditError::refuse(
                "this cut needs a non-quad face here \u{2014} move the cut, or insert a loop first",
                Highlight { faces: refused, ..Highlight::default() },
            ));
        }
        for (i, &on) in self.points.iter().enumerate() {
            if let On::Edge { .. } = on
                && referenced.contains(&i)
            {
                plan.points.push(position(on));
            }
        }
        plan.new_positions = new_positions;
        let _ = self.eps;
        Ok(plan)
    }
}

fn highlight_of(points: &[On]) -> Highlight {
    let mut h = Highlight::default();
    for &p in points {
        match p {
            On::Vertex(v) => h.vertices.push(v),
            On::Edge { a, b, .. } => h.edges.push((a, b)),
        }
    }
    h
}

#[allow(dead_code)]
fn unused(_: Vector3<f64>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::tests::{cube, grid, split, torus_grid};
    use crate::Cage;

    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }

    /// A face's edge `i` as a cut point at `t` from its corner `i`.
    fn on_edge(mesh: &QuadMesh, face: usize, i: usize, t: f64) -> CutPoint {
        let c = mesh.face_vertices(FaceId::from_index(face));
        CutPoint::Edge { edge: (c[i % 4], c[(i + 1) % 4]), t }
    }

    fn closed(mesh: &QuadMesh) -> Cage {
        Cage::try_from_mesh(mesh.clone()).map_err(|(_, e)| e).expect("closed")
    }

    fn pattern_sizes(cut: &Cut) -> Vec<usize> {
        let mut sizes: Vec<usize> = cut.result.replaced.values().map(|v| v.len()).collect();
        sizes.sort();
        sizes
    }

    /// **A straight cut across one face sends a front out of each end, and the two meet head-on
    /// on the far side** (the notes' test 1): one closed loop, the belt's every face split in two.
    #[test]
    fn a_straight_cut_across_one_face_closes_as_one_loop_round_the_cage() {
        let m = split(&cube());
        let cut = cut(&m, &[on_edge(&m, 5, 0, 0.3), on_edge(&m, 5, 2, 0.6)]).expect("a cut");
        closed(&cut.result.mesh);
        assert_eq!(cut.result.mesh.euler_characteristic(), 2);
        // A ring round a split cube crosses eight faces, each becoming two.
        assert_eq!(pattern_sizes(&cut), vec![2; 8]);
        assert_eq!(cut.result.new_vertices.len(), 8);
        assert_eq!(cut.cut_edges.len(), 1, "one drawn edge; the rest are the fronts'");
        let eps = epsilon_length(&m);
        for (a, b) in cut.result.mesh.edges() {
            assert!((cut.result.mesh.vertex_position(a) - cut.result.mesh.vertex_position(b)).norm() > eps);
        }
    }

    /// **A full loop drawn through opposite edges all the way round seeds no front**: the faces
    /// added are exactly the faces crossed (test 2).
    #[test]
    fn a_closed_drawn_loop_propagates_nothing() {
        let m = split(&cube());
        // Follow the ring from face 5's edge 0 round the cube, as a user would draw it.
        let ring = crate::edit::insert_ring(&m, (m.face_vertices(FaceId::from_index(5))[0], m.face_vertices(FaceId::from_index(5))[1]), 0.5).unwrap();
        let faces: Vec<FaceId> = ring.replaced.keys().copied().collect();
        assert_eq!(faces.len(), 8);
        // The points: the ring's own new vertices tell which edges it crossed; draw the path
        // through those edges, in ring order, closing on the first.
        let mut path = Vec::new();
        let mut at = (m.face_vertices(FaceId::from_index(5))[0], m.face_vertices(FaceId::from_index(5))[1]);
        let mut face = FaceId::from_index(5);
        for _ in 0..8 {
            path.push(CutPoint::Edge { edge: at, t: 0.5 });
            let c = m.face_vertices(face);
            let i = (0..4).find(|&i| (c[i] == at.0 && c[(i + 1) % 4] == at.1) || (c[i] == at.1 && c[(i + 1) % 4] == at.0)).unwrap();
            at = (c[(i + 2) % 4], c[(i + 3) % 4]);
            let (f0, f1) = m.edge_faces(at.0, at.1).unwrap();
            face = if f0 == face { f1 } else { f0 };
        }
        path.push(path[0]);
        let cut = cut(&m, &path).expect("a loop");
        closed(&cut.result.mesh);
        assert_eq!(pattern_sizes(&cut), vec![2; 8]);
        assert_eq!(cut.result.new_vertices.len(), 8);
        assert_eq!(cut.cut_edges.len(), 8, "every edge is drawn");
    }

    /// **A corner turn's two fronts meet side-on and turn a second corner there** (test 3, R5):
    /// two corner patterns, no crossing, one closed loop.
    #[test]
    fn a_corner_turn_closes_with_a_second_corner_where_its_fronts_meet() {
        let m = split(&cube());
        let cut = cut(&m, &[on_edge(&m, 5, 0, 0.5), on_edge(&m, 5, 1, 0.5)]).expect("a corner");
        closed(&cut.result.mesh);
        assert_eq!(cut.result.mesh.euler_characteristic(), 2);
        let sizes = pattern_sizes(&cut);
        assert_eq!(sizes.iter().filter(|&&n| n == 3).count(), 2, "two corners: {sizes:?}");
        assert_eq!(sizes.iter().filter(|&&n| n == 4).count(), 0, "no crossing: {sizes:?}");
        // The drawn corner's interior vertex sits a quarter of the way from the chord's midpoint
        // to the far corner.
        let c = m.face_vertices(FaceId::from_index(5));
        let p = lerp(m.vertex_position(c[0]), m.vertex_position(c[1]), 0.5);
        let q = lerp(m.vertex_position(c[1]), m.vertex_position(c[2]), 0.5);
        let mid = lerp(p, q, 0.5);
        let expected = mid + (m.vertex_position(c[3]) - mid) / 4.0;
        let found = cut.result.new_vertices.iter().any(|&v| (cut.result.mesh.vertex_position(v) - expected).norm() < 1e-9);
        assert!(found, "the corner vertex is where the notes put it");
        assert_eq!(cut.cut_edges.len(), 2);
    }

    /// **A diagonal is refused, with the measured reason** (R6): corner to corner, and corner to a
    /// point on a non-adjacent edge.
    #[test]
    fn a_cut_from_a_corner_to_anything_but_its_own_edges_is_refused() {
        let m = split(&cube());
        let c = m.face_vertices(FaceId::from_index(5));
        let diagonal = cut(&m, &[CutPoint::Vertex(c[0]), CutPoint::Vertex(c[2])]).unwrap_err();
        assert!(diagonal.reason.contains("diagonal"), "{}", diagonal.reason);
        let to_edge = cut(&m, &[CutPoint::Vertex(c[0]), on_edge(&m, 5, 1, 0.5)]).unwrap_err();
        assert!(to_edge.reason.contains("diagonal"), "{}", to_edge.reason);
        assert_eq!(to_edge.highlight.faces, vec![FaceId::from_index(5)]);
        // Along an edge is no cut at all.
        let along = cut(&m, &[CutPoint::Vertex(c[0]), CutPoint::Vertex(c[1])]).unwrap_err();
        assert!(along.reason.contains("existing edges"), "{}", along.reason);
        let to_own_edge = cut(&m, &[CutPoint::Vertex(c[0]), on_edge(&m, 5, 0, 0.5)]).unwrap_err();
        assert!(to_own_edge.reason.contains("existing edges"), "{}", to_own_edge.reason);
    }

    /// **A point within ε of a vertex is that vertex** (test 11): the refusal is the diagonal's.
    #[test]
    fn a_point_within_epsilon_of_a_vertex_becomes_it() {
        let m = split(&cube());
        let c = m.face_vertices(FaceId::from_index(5));
        let snapped = cut(&m, &[CutPoint::Edge { edge: (c[0], c[1]), t: 1e-12 }, on_edge(&m, 5, 1, 0.5)]).unwrap_err();
        let as_vertex = cut(&m, &[CutPoint::Vertex(c[0]), on_edge(&m, 5, 1, 0.5)]).unwrap_err();
        assert_eq!(snapped, as_vertex);
    }

    /// **Two corner patterns in one face are refused, naming it** (test 7): on a 3×3 sheet the path
    /// goes round the centre vertex and back through its first face.
    #[test]
    fn a_path_that_turns_a_corner_twice_in_one_face_is_refused() {
        let m = grid(3, 3, &[]);
        // Faces round the centre vertex, found by sharing it.
        let centre = (0..m.vertex_count()).map(v).find(|&x| m.valence(x) == 4).expect("a 3×3 grid has one interior vertex");
        let around = m.adjacent_faces(centre);
        assert_eq!(around.len(), 4);
        let f0 = around[0];
        // The path: in through one of f0's outer edges, then through each face round the centre
        // in turn, back into f0 and out its other outer edge.
        let c0 = m.face_vertices(f0);
        let k = c0.iter().position(|&x| x == centre).unwrap();
        // Outer edges of f0: the two not touching the centre, k+1 and k+2 (edges k-1 and k touch it).
        let start = CutPoint::Edge { edge: (c0[(k + 1) % 4], c0[(k + 2) % 4]), t: 0.5 };
        let end = CutPoint::Edge { edge: (c0[(k + 2) % 4], c0[(k + 3) % 4]), t: 0.5 };
        // Then the edges from the centre outward, in order round it: each is shared by two
        // consecutive faces round the centre.
        let spokes: Vec<(VertexId, VertexId)> = m.adjacent_vertices(centre).into_iter().map(|a| (centre, a)).collect();
        let mut path = vec![start];
        // Walk the spokes starting from the one f0 shares with the next face, round and back.
        let f0_spokes: Vec<usize> = spokes.iter().enumerate().filter(|(_, e)| c0.contains(&e.1)).map(|(i, _)| i).collect();
        assert_eq!(f0_spokes.len(), 2);
        let mut order = Vec::new();
        let mut i = f0_spokes[0];
        for _ in 0..4 {
            order.push(i);
            // Next spoke round the centre: the other spoke of the face across spoke i.
            let (a, b) = spokes[i];
            let (fa, fb) = m.edge_faces(a, b).unwrap();
            let face = if order.len() == 1 { if c0 == m.face_vertices(fa) { fb } else { fa } } else {
                // The face that is not the one we came from.
                let prev = spokes[order[order.len() - 2]];
                let came = m.edge_faces(prev.0, prev.1).unwrap();
                if fa == came.0 || fa == came.1 { fb } else { fa }
            };
            let cf = m.face_vertices(face);
            i = spokes.iter().position(|e| e.1 != b && cf.contains(&e.1)).unwrap();
        }
        for &s in &order {
            path.push(CutPoint::Edge { edge: spokes[s], t: 0.5 });
        }
        path.push(end);
        let err = cut(&m, &path).unwrap_err();
        assert!(err.reason.contains("non-quad"), "{}", err.reason);
        assert_eq!(err.highlight.faces, vec![f0]);
    }

    /// **On an open mesh a front stops at the boundary** (test 8): a cut across the middle of a
    /// strip splits every face and makes no new hole.
    #[test]
    fn a_front_stops_at_an_open_boundary() {
        let m = grid(3, 1, &[]);
        let c = m.face_vertices(FaceId::from_index(1));
        // The middle face's edges shared with its neighbours: the ones with a face either side.
        let shared: Vec<usize> = (0..4).filter(|&i| m.edge_faces(c[i], c[(i + 1) % 4]).is_some()).collect();
        assert_eq!(shared.len(), 2);
        let cut = cut(&m, &[on_edge(&m, 1, shared[0], 0.5), on_edge(&m, 1, shared[1], 0.5)]).expect("a cut");
        assert_eq!(cut.result.mesh.face_count(), 6);
        assert!(Cage::try_from_mesh(cut.result.mesh.clone()).is_err(), "still open");
        let boundary = (0..cut.result.mesh.half_edge_count()).filter(|&h| cut.result.mesh.is_boundary(HalfEdgeId::new(h))).count();
        assert_eq!(boundary, 10, "the rim gained two vertices and no hole");
    }

    /// **The order fronts advance in within a step does not change the mesh** (test 13).
    #[test]
    fn the_fronts_order_within_a_step_does_not_change_the_result() {
        let m = split(&cube());
        for path in [
            vec![on_edge(&m, 5, 0, 0.3), on_edge(&m, 5, 2, 0.6)],
            vec![on_edge(&m, 5, 0, 0.5), on_edge(&m, 5, 1, 0.5)],
            vec![on_edge(&m, 5, 0, 0.25), on_edge(&m, 5, 1, 0.75)],
        ] {
            let a = apply(&m, plan_ordered(&m, &path, false).unwrap()).unwrap();
            let b = apply(&m, plan_ordered(&m, &path, true).unwrap()).unwrap();
            assert_eq!(a.result.mesh.geometry_fingerprint(), b.result.mesh.geometry_fingerprint());
        }
    }

    /// **Every single-face straight cut on the split cube closes** (tests 1 and 12 over the
    /// corpus): whichever edge it starts on, the fronts meet and the visit bound never fires.
    #[test]
    fn every_straight_cut_on_the_split_cube_closes() {
        let m = split(&cube());
        for f in 0..m.face_count() {
            for i in 0..2 {
                let cut = cut(&m, &[on_edge(&m, f, i, 0.4), on_edge(&m, f, i + 2, 0.4)]).unwrap_or_else(|e| panic!("face {f} edge {i}: {}", e.reason));
                closed(&cut.result.mesh);
                assert_eq!(cut.result.mesh.euler_characteristic(), 2);
            }
        }
    }

    /// **Every corner turn on the split cube closes with a second corner** (test 3 over the corpus).
    #[test]
    fn every_corner_turn_on_the_split_cube_closes() {
        let m = split(&cube());
        for f in 0..m.face_count() {
            for i in 0..4 {
                let cut = cut(&m, &[on_edge(&m, f, i, 0.5), on_edge(&m, f, i + 1, 0.5)]).unwrap_or_else(|e| panic!("face {f} edge {i}: {}", e.reason));
                closed(&cut.result.mesh);
                assert_eq!(cut.result.mesh.euler_characteristic(), 2);
                let sizes = pattern_sizes(&cut);
                assert_eq!(sizes.iter().filter(|&&n| n == 3).count(), 2, "face {f} edge {i}: {sizes:?}");
            }
        }
    }

    /// **Fronts on two belts that cross only once are refused** (test 14): a corner turn on a
    /// torus, one front round the tube and one along it.
    #[test]
    fn a_corner_turn_on_a_torus_whose_belts_cross_once_is_refused() {
        let m = torus_grid(4, 8, true);
        closed(&m);
        assert_eq!(m.euler_characteristic(), 0);
        let err = cut(&m, &[on_edge(&m, 0, 0, 0.5), on_edge(&m, 0, 1, 0.5)]).unwrap_err();
        assert!(err.reason.contains("can't close"), "{}", err.reason);
        // And every straight cut across one face of it still closes round the tube or along it.
        for i in 0..2 {
            let cut = cut(&m, &[on_edge(&m, 0, i, 0.5), on_edge(&m, 0, i + 2, 0.5)]).expect("a loop");
            closed(&cut.result.mesh);
            assert_eq!(cut.result.mesh.euler_characteristic(), 0);
        }
    }

    /// **A plan draws what a cut makes**: the preview's edges are the applied mesh's new edges.
    #[test]
    fn the_plan_shows_the_edges_the_cut_adds() {
        let m = split(&cube());
        let path = [on_edge(&m, 5, 0, 0.5), on_edge(&m, 5, 1, 0.5)];
        let plan = plan(&m, &path).unwrap();
        let cut = cut(&m, &path).unwrap();
        let before: std::collections::HashSet<(usize, usize)> = m.edges().map(|(a, b)| (a.index().min(b.index()), a.index().max(b.index()))).collect();
        let added = cut.result.mesh.edges().filter(|(a, b)| !before.contains(&(a.index().min(b.index()), a.index().max(b.index())))).count();
        // Every new edge is either drawn or a front's, bar the split halves of the edges the
        // points sit on.
        assert_eq!(plan.path_edges.len(), 2);
        assert!(plan.front_edges.len() + plan.path_edges.len() <= added);
        assert!(!plan.points.is_empty());
    }

    /// A path drawn by walking: from `face`, in through edge `entry`, out through the edge
    /// `moves[k]` places round from it (2 straight, 1 or 3 a turn), each point at 0.4 from the
    /// exit edge's first corner in that face, alternately 0.3 and 0.6 so a face passed twice the same way holds two points.
    fn walk(mesh: &QuadMesh, face: FaceId, entry: usize, moves: &[usize]) -> Option<Vec<CutPoint>> {
        let mut path = vec![on_edge(mesh, face.index(), entry, 0.3)];
        let (mut face, mut entry) = (face, entry);
        for &m in moves {
            let exit = (entry + m) % 4;
            path.push(on_edge(mesh, face.index(), exit, if path.len() % 2 == 0 { 0.3 } else { 0.6 }));
            let c = mesh.face_vertices(face);
            let (a, b) = (c[exit], c[(exit + 1) % 4]);
            let (fa, fb) = mesh.edge_faces(a, b)?;
            let next = if fa == face { fb } else { fa };
            let cn = mesh.face_vertices(next);
            entry = (0..4).find(|&k| cn[k] == b && cn[(k + 1) % 4] == a)?;
            face = next;
        }
        Some(path)
    }

    /// **A drawn path that crosses its own earlier segment resolves that face as a crossing, and
    /// one that runs beside it as a strip** (test 6), found by walking every short path on the
    /// split cube; and every cut the walk makes that is not refused is closed and all-quad.
    #[test]
    fn a_path_crossing_or_running_beside_itself_resolves_as_a_crossing_or_a_strip() {
        let m = split(&cube());
        let (mut crossings, mut cuts) = (0, 0);
        let mut moves = vec![2usize; 6];
        // Every sequence of six moves over {1, 2, 3}: 729 paths from each start.
        for code in 0..729 {
            let mut n = code;
            for slot in moves.iter_mut() {
                *slot = 1 + n % 3;
                n /= 3;
            }
            for f in [0usize, 5, 13] {
                let Some(path) = walk(&m, FaceId::from_index(f), 0, &moves) else { continue };
                let Ok(plan) = plan(&m, &path) else { continue };
                let cut = apply(&m, plan.clone()).unwrap_or_else(|e| panic!("{moves:?} from face {f}: {}", e.reason));
                closed(&cut.result.mesh);
                assert_eq!(cut.result.mesh.euler_characteristic(), 2, "{moves:?} from face {f}");
                cuts += 1;
                crossings += plan.patterns.iter().filter(|(_, p)| *p == Pattern::Crossing).count();
            }
        }
        assert!(cuts > 100, "{cuts} paths cut");
        assert!(crossings > 0, "no path on the split cube made a crossing");
    }

    /// **Two cuts side by side across one face are a strip** (test 6's S), drawn on a single
    /// quad: across, back along the far edge (which is no cut), and across again.
    #[test]
    fn two_cuts_across_the_same_pair_of_edges_make_a_strip() {
        let m = grid(1, 1, &[]);
        let path = [on_edge(&m, 0, 0, 0.3), on_edge(&m, 0, 2, 0.7), on_edge(&m, 0, 2, 0.4), on_edge(&m, 0, 0, 0.6)];
        let plan = plan(&m, &path).unwrap();
        assert_eq!(plan.patterns, vec![(FaceId::from_index(0), Pattern::Strip)]);
        let cut = apply(&m, plan).unwrap();
        assert_eq!(cut.result.mesh.face_count(), 3);
        assert_eq!(cut.cut_edges.len(), 2);
        // Crossing each other instead, they are no strip.
        let crossed = [on_edge(&m, 0, 0, 0.3), on_edge(&m, 0, 2, 0.4), on_edge(&m, 0, 2, 0.7), on_edge(&m, 0, 0, 0.6)];
        assert!(super::plan(&m, &crossed).unwrap_err().reason.contains("non-quad"));
    }
}
