// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The knife as a tool: points clicked on the cage, the segment between two of them traced
//! across the faces it crosses on screen, the cut previewed as it would land -- the drawn edges
//! green, the loops that run on from its ends dashed, a face no pattern fits red -- and Enter
//! committing the whole path as one `cut` line (the cage-editing notes, item 76).
//!
//! **Tracing is geometric, not picked.** The screen segment between two points and the eye span
//! a plane; walking from the first point's face, each face is left through the edge that plane
//! cuts ahead of the pointer, into the face beyond it, until the face holding the second point
//! is reached. That gives every edge crossing exactly where the drawn line crosses it, with no
//! picking along the way, and it follows the surface through faces the line passes over.

use super::check_model::{ERROR_COLOR, ERROR_FILL};
use super::tool_panel::Row;
use crate::overlay_draw::{PREVIEW_COLOR, PROPAGATED_COLOR};
use crate::*;
use fieldform_cage::edit::knife::{self, CutPoint, Plan};
use fieldform_cage::{FaceId, QuadMesh, VertexId};

/// What a click may resolve to, the panel's Pick type (`cage-tools-instructions.md` §1). `Ctrl`
/// always forces the nearest vertex regardless, the meaning it already has for a curve's
/// endpoints while placing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PickType {
    Vertices,
    Edges,
    Both,
}

impl Default for PickType {
    fn default() -> Self {
        PickType::Both
    }
}

impl PickType {
    fn word(self) -> &'static str {
        match self {
            PickType::Vertices => "vertices",
            PickType::Edges => "edges",
            PickType::Both => "both",
        }
    }
    fn from_word(w: &str) -> Option<Self> {
        match w {
            "vertices" => Some(PickType::Vertices),
            "edges" => Some(PickType::Edges),
            "both" => Some(PickType::Both),
            _ => None,
        }
    }
}

/// How a typed or dragged offset is read (`cage-knife-spec.md` §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OffsetMode {
    Mm,
    Fraction,
}

impl Default for OffsetMode {
    fn default() -> Self {
        OffsetMode::Mm
    }
}

impl OffsetMode {
    pub(crate) fn word(self) -> &'static str {
        match self {
            OffsetMode::Mm => "mm",
            OffsetMode::Fraction => "fraction",
        }
    }
    pub(crate) fn from_word(w: &str) -> Option<Self> {
        match w {
            "mm" => Some(OffsetMode::Mm),
            "fraction" => Some(OffsetMode::Fraction),
            _ => None,
        }
    }
}

/// Which faces Cut Through considers (`cage-knife-spec.md` §1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ThroughFaces {
    All,
    FrontFacing,
    Selected,
}

impl Default for ThroughFaces {
    fn default() -> Self {
        ThroughFaces::All
    }
}

impl ThroughFaces {
    fn word(self) -> &'static str {
        match self {
            ThroughFaces::All => "all",
            ThroughFaces::FrontFacing => "front-facing",
            ThroughFaces::Selected => "selected",
        }
    }
    fn from_word(w: &str) -> Option<Self> {
        match w {
            "all" => Some(ThroughFaces::All),
            "front-facing" | "frontfacing" => Some(ThroughFaces::FrontFacing),
            "selected" => Some(ThroughFaces::Selected),
            _ => None,
        }
    }
}

/// The knife's state while armed.
pub(crate) struct Knife {
    /// The points placed so far, every edge crossing between clicks included.
    pub(crate) path: Vec<CutPoint>,
    /// The crossings from the last placed point to the hovered one, and the hovered point.
    pub(crate) ahead: Vec<CutPoint>,
    /// The cut as it would land with the hovered point added, or why it would not.
    pub(crate) plan: Option<Plan>,
    pub(crate) refused: Option<(String, Vec<FaceId>)>,
    /// Cut Through: the screen line's first point, and its second once placed. The line stays
    /// where it was drawn on screen while the view turns, and is read against the view each frame.
    pub(crate) through: Option<(egui::Pos2, Option<egui::Pos2>)>,
    /// What the placed line cuts in this frame's view: each path, with its plan or its refusal.
    pub(crate) through_paths: Vec<(Vec<CutPoint>, Result<Plan, (String, Vec<FaceId>)>)>,
    /// Where the pointer was this frame, for the rubber line.
    pub(crate) cursor: Option<egui::Pos2>,
    /// The panel's settings (`cage-tools-instructions.md` §1).
    pub(crate) pick_type: PickType,
    pub(crate) offset_mode: OffsetMode,
    pub(crate) angle_increment: f64,
    pub(crate) through_faces: ThroughFaces,
    /// The last placed point, while it is an edge point: the end its offset reads and writes
    /// from, fixed when it was placed -- nearer the point before it, or the cursor for the
    /// path's first point (`cage-knife-spec.md` §3).
    pub(crate) active_ref_end: Option<VertexId>,
}

impl Default for Knife {
    fn default() -> Self {
        Knife {
            path: Vec::new(),
            ahead: Vec::new(),
            plan: None,
            refused: None,
            through: None,
            through_paths: Vec::new(),
            cursor: None,
            pick_type: PickType::default(),
            offset_mode: OffsetMode::default(),
            angle_increment: 45.0,
            through_faces: ThroughFaces::default(),
            active_ref_end: None,
        }
    }
}

/// The faces an edge `(a, b)` lies on: two, or one at a rim.
fn faces_of_edge(mesh: &QuadMesh, a: VertexId, b: VertexId) -> Vec<FaceId> {
    mesh.adjacent_faces(a)
        .into_iter()
        .filter(|&f| {
            let c = mesh.face_vertices(f);
            (0..4).any(|i| (c[i] == a && c[(i + 1) % 4] == b) || (c[i] == b && c[(i + 1) % 4] == a))
        })
        .collect()
}

fn faces_of(mesh: &QuadMesh, p: CutPoint) -> Vec<FaceId> {
    match p {
        CutPoint::Vertex(v) => mesh.adjacent_faces(v),
        CutPoint::Edge { edge: (a, b), .. } => faces_of_edge(mesh, a, b),
    }
}

/// A point's position in the cage's own space.
fn position(mesh: &QuadMesh, p: CutPoint) -> Point3<f64> {
    match p {
        CutPoint::Vertex(v) => mesh.vertex_position(v),
        CutPoint::Edge { edge: (a, b), t } => {
            let (pa, pb) = (mesh.vertex_position(a), mesh.vertex_position(b));
            pa + (pb - pa) * t
        }
    }
}

/// The console spelling of a point: a vertex index, or `a-b@t`.
fn spell(p: CutPoint) -> String {
    match p {
        CutPoint::Vertex(v) => v.index().to_string(),
        CutPoint::Edge { edge: (a, b), t } => format!("{}-{}@{}", a.index(), b.index(), echo::number(t)),
    }
}


/// `t` from `edge.0` for a point at `value` -- millimetres or a fraction, per `mode` -- from the
/// end `ref_end`, clamped to the edge. Offset is length along the **cage** edge, not along the
/// limit surface: that is the edge the person sees (`cage-knife-spec.md` §3).
pub(crate) fn t_from_offset(mesh: &QuadMesh, edge: (VertexId, VertexId), ref_end: VertexId, mode: OffsetMode, value: f64) -> f64 {
    let (a, b) = edge;
    let len = (mesh.vertex_position(a) - mesh.vertex_position(b)).norm().max(1e-12);
    let t_from_ref = match mode {
        OffsetMode::Mm => (value / len).clamp(0.0, 1.0),
        OffsetMode::Fraction => value.clamp(0.0, 1.0),
    };
    if ref_end == a { t_from_ref } else { 1.0 - t_from_ref }
}
impl App {
    fn knife(&self) -> Option<&Knife> {
        self.cage_tool.as_ref().filter(|t| t.command.op == "cage.cut").map(|t| &t.knife)
    }

    fn knife_mut(&mut self) -> Option<&mut Knife> {
        self.cage_tool.as_mut().filter(|t| t.command.op == "cage.cut").map(|t| &mut t.knife)
    }

    /// The point `hover` names: the vertex, or the hovered edge at the tool's fraction along it
    /// (Shift: its middle) -- gated by the panel's Pick type, which `ctrl` overrides to force the
    /// nearest vertex regardless. `hover` is `self.hover` for the pointer where it actually is, or
    /// a re-picked selection at an angle-snapped position (`knife_hover`).
    fn knife_hovered_point(&self, aspect: f32, rect: egui::Rect, pointer: Option<egui::Pos2>, shift: bool, ctrl: bool, hover: Selection) -> Option<CutPoint> {
        /// How near a vertex the pointer snaps to it, in points: wider than the pick's own
        /// threshold, because a cut aimed near a corner is meant for the corner.
        const SNAP: f32 = 14.0;
        let mesh = self.cage();
        let placement = self.control_placement();
        let pick_type = self.knife().map(|k| k.pick_type).unwrap_or_default();
        let screen = |v: VertexId| self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, v));
        // The nearest point on an edge to the pointer, as the fraction along it and how far off.
        let on_edge = |a: VertexId, b: VertexId, pos: egui::Pos2| -> Option<(f64, f32)> {
            let (pa, pb) = (screen(a)?, screen(b)?);
            let d = pb - pa;
            if d.length_sq() < 1.0 {
                return None;
            }
            let t = ((pos - pa).dot(d) / d.length_sq()).clamp(0.02, 0.98);
            let at = pa + d * t;
            Some((t as f64, (pos - at).length()))
        };
        let snapped = |candidates: &[VertexId], pos: egui::Pos2| -> Option<VertexId> {
            candidates
                .iter()
                .filter_map(|&v| screen(v).map(|s| (v, (s - pos).length())))
                .filter(|(_, d)| *d <= SNAP)
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .map(|(v, _)| v)
        };
        let nearest = |candidates: &[VertexId], pos: egui::Pos2| -> Option<VertexId> {
            candidates.iter().filter_map(|&v| screen(v).map(|s| (v, (s - pos).length()))).min_by(|x, y| x.1.total_cmp(&y.1)).map(|(v, _)| v)
        };
        let allow_vertex = ctrl || pick_type != PickType::Edges;
        match (hover, pointer) {
            (Selection::Vertex(v), _) if allow_vertex => Some(CutPoint::Vertex(v)),
            // Edges only, aimed at a vertex: the nearest point on one of its own edges, not the
            // vertex itself -- there is nowhere else for the point to be.
            (Selection::Vertex(v), Some(pos)) => mesh
                .adjacent_faces(v)
                .into_iter()
                .flat_map(|f| {
                    let c = mesh.face_vertices(f);
                    (0..4).map(move |i| (c[i], c[(i + 1) % 4]))
                })
                .filter(|&(a, b)| a == v || b == v)
                .filter_map(|(a, b)| on_edge(a, b, pos).map(|(t, off)| ((a, b), t, off)))
                .min_by(|x, y| x.2.total_cmp(&y.2))
                .map(|((a, b), t, _)| CutPoint::Edge { edge: (a, b), t: if shift { 0.5 } else { t } }),
            // Near either end of the hovered edge: the vertex, not a sliver beside it.
            (Selection::Edge(a, b), Some(pos)) => {
                if allow_vertex && let Some(v) = snapped(&[a, b], pos) {
                    return Some(CutPoint::Vertex(v));
                }
                if pick_type == PickType::Vertices {
                    return nearest(&[a, b], pos).map(CutPoint::Vertex);
                }
                let t = if shift { 0.5 } else { on_edge(a, b, pos).map_or(0.5, |(t, _)| t) };
                Some(CutPoint::Edge { edge: (a, b), t })
            }
            (Selection::Edge(a, b), None) => Some(CutPoint::Edge { edge: (a, b), t: 0.5 }),
            // Over a face: the nearest point on its rim, so the knife follows the pointer across
            // the face instead of waiting for an edge under it (the owner, 2026-10-24).
            (Selection::Face(f), Some(pos)) => {
                let c = mesh.face_vertices(f);
                if allow_vertex && let Some(v) = snapped(&c, pos) {
                    return Some(CutPoint::Vertex(v));
                }
                if pick_type == PickType::Vertices {
                    return nearest(&c, pos).map(CutPoint::Vertex);
                }
                (0..4)
                    .filter_map(|i| {
                        let (a, b) = (c[i], c[(i + 1) % 4]);
                        on_edge(a, b, pos).map(|(t, off)| ((a, b), t, off))
                    })
                    .min_by(|x, y| x.2.total_cmp(&y.2))
                    .map(|((a, b), t, _)| CutPoint::Edge { edge: (a, b), t: if shift { 0.5 } else { t } })
            }
            _ => None,
        }
    }

    /// The edge crossings between `from` and `to` on screen, then `to`; `None` where the line
    /// cannot be followed over the cage (off its rim, or a face edge-on to the view).
    fn knife_trace(&self, aspect: f32, rect: egui::Rect, from: CutPoint, to: CutPoint) -> Option<Vec<CutPoint>> {
        let mesh = self.cage();
        let placement = self.control_placement();
        let world = |p: Point3<f64>| placement.transform_point(&p);
        let to_screen = |p: Point3<f64>| {
            let w = world(p);
            self.camera.project_to_screen(aspect, rect, Point3::new(w.x as f32, w.y as f32, w.z as f32))
        };
        let a_screen = to_screen(position(mesh, from))?;
        let b_screen = to_screen(position(mesh, to))?;
        let (oa, da) = self.camera.pick_ray(aspect, a_screen, rect);
        let (ob, db) = self.camera.pick_ray(aspect, b_screen, rect);
        let (oa, da, ob, db) = (oa.cast::<f64>(), da.cast::<f64>(), ob.cast::<f64>(), db.cast::<f64>());
        // The plane through both rays: through the eye in perspective, through the two origins
        // when the view is orthographic and the rays are parallel.
        let mut n = da.cross(&db);
        if n.norm() < 1e-9 {
            n = (ob - oa).cross(&da);
        }
        let n = nalgebra::Unit::try_new(n, 1e-12)?;
        let dir = b_screen - a_screen;
        if dir.length_sq() < 1.0 {
            return Some(vec![to]);
        }
        // How far along the drawn line a world point projects, 0 at `from` and 1 at `to`.
        let along = |p: Point3<f64>| to_screen(p).map(|s| ((s - a_screen).dot(dir) / dir.length_sq()) as f64);

        let targets = faces_of(mesh, to);
        let mut out = Vec::new();
        // The exit of `f` ahead of the pointer: the edge the plane cuts, other than the one the
        // line came in through, at the point furthest along the line but not past `to`.
        let exit_of = |f: FaceId, came: Option<(VertexId, VertexId)>, start: CutPoint| -> Option<((VertexId, VertexId), f64, f64)> {
            let c = mesh.face_vertices(f);
            let mut best: Option<((VertexId, VertexId), f64, f64)> = None;
            for i in 0..4 {
                let (p, q) = (c[i], c[(i + 1) % 4]);
                let same_edge = |e: Option<(VertexId, VertexId)>| e.is_some_and(|(a, b)| (a == p && b == q) || (a == q && b == p));
                if same_edge(came) {
                    continue;
                }
                if let CutPoint::Edge { edge, .. } = start
                    && same_edge(Some(edge))
                {
                    continue;
                }
                if let CutPoint::Vertex(v) = start
                    && (v == p || v == q)
                {
                    continue;
                }
                let (wp, wq) = (world(mesh.vertex_position(p)), world(mesh.vertex_position(q)));
                let d = (wq - wp).dot(&n);
                if d.abs() < 1e-12 {
                    continue;
                }
                let t = (oa - wp).dot(&n) / d;
                if !(t > 0.0 && t < 1.0) {
                    continue;
                }
                let x = mesh.vertex_position(p) + (mesh.vertex_position(q) - mesh.vertex_position(p)) * t;
                let Some(u) = along(x) else { continue };
                if u <= 1e-6 || u >= 1.0 - 1e-6 {
                    continue;
                }
                if best.is_none_or(|(_, _, bu)| u < bu) {
                    best = Some(((p, q), t, u));
                }
            }
            best
        };

        // The first face: one of the start's that the line goes into.
        let mut start_faces = faces_of(mesh, from);
        if let Some(f) = start_faces.iter().find(|f| targets.contains(f)) {
            let _ = f;
            return Some(vec![to]);
        }
        start_faces.sort_by_key(|f| f.index());
        let mut first: Option<(FaceId, ((VertexId, VertexId), f64, f64))> = None;
        for f in start_faces {
            if let Some(exit) = exit_of(f, None, from)
                && first.is_none_or(|(_, (_, _, u))| exit.2 < u)
            {
                first = Some((f, exit));
            }
        }
        let (f, exit) = first?;
        let mut face = Some(f);
        let mut exit = Some(exit);
        for _ in 0..mesh.face_count() * 2 {
            let ((p, q), t, _) = exit?;
            out.push(CutPoint::Edge { edge: (p, q), t });
            let here = face?;
            let next = faces_of_edge(mesh, p, q).into_iter().find(|&g| g != here)?;
            if targets.contains(&next) {
                out.push(to);
                return Some(out);
            }
            face = Some(next);
            exit = exit_of(next, Some((p, q)), CutPoint::Edge { edge: (p, q), t });
        }
        None
    }

    /// Whether the pointer is back on the path's first point: the same vertex, or the same edge
    /// within the snap distance of where the path began on it.
    fn knife_closes(&self, aspect: f32, rect: egui::Rect, pos: egui::Pos2, first: CutPoint, hovered: CutPoint) -> bool {
        const SNAP: f32 = 14.0;
        let same_place = match (first, hovered) {
            (CutPoint::Vertex(a), CutPoint::Vertex(b)) => a == b,
            (CutPoint::Edge { edge: (a, b), .. }, CutPoint::Edge { edge: (c, d), .. }) => (a == c && b == d) || (a == d && b == c),
            _ => false,
        };
        if !same_place {
            return false;
        }
        let mesh = self.cage();
        let placement = self.control_placement();
        let p = placement.transform_point(&position(mesh, first));
        self.camera
            .project_to_screen(aspect, rect, Point3::new(p.x as f32, p.y as f32, p.z as f32))
            .is_some_and(|s| (s - pos).length() <= SNAP)
    }

    /// Once a frame while the knife is armed: the segment to the hovered point traced and the
    /// whole cut planned, for the preview.
    pub(crate) fn knife_hover(&mut self, aspect: f32, rect: egui::Rect, pointer: Option<egui::Pos2>, shift: bool, ctrl: bool, alt: bool) {
        if let Some(k) = self.knife_mut() {
            k.cursor = pointer;
        }
        // A placed Cut Through line is read against this frame's view.
        if let Some((a, Some(b))) = self.knife().and_then(|k| k.through) {
            let paths = self.through_paths(aspect, rect, a, b);
            let resolved: Vec<(Vec<CutPoint>, Result<Plan, (String, Vec<FaceId>)>)> = match paths {
                Ok(paths) => paths
                    .into_iter()
                    .map(|path| {
                        let plan = knife::plan(self.cage(), &path).map_err(|e| (e.reason, e.highlight.faces));
                        (path, plan)
                    })
                    .collect(),
                Err(why) => vec![(Vec::new(), Err((why, Vec::new())))],
            };
            if let Some(k) = self.knife_mut() {
                k.through_paths = resolved;
                k.ahead.clear();
                k.plan = None;
                k.refused = None;
            }
            return;
        }
        let path = self.knife().map(|k| k.path.clone()).unwrap_or_default();
        // Alt snaps the direction from the path's last point to the cursor, in screen space, to
        // a multiple of the panel's angle increment (`cage-knife-spec.md` §4): the snapped ray is
        // then read like any other pointer position, re-picked against the cage.
        let angle_increment = self.knife().map(|k| k.angle_increment).unwrap_or(45.0).max(1.0);
        let (hover_sel, pointer) = match (alt, path.last(), pointer) {
            (true, Some(&last), Some(pos)) => {
                let mesh = self.cage();
                let placement = self.control_placement();
                let w = placement.transform_point(&position(mesh, last));
                let origin = self.camera.project_to_screen(aspect, rect, Point3::new(w.x as f32, w.y as f32, w.z as f32));
                match origin {
                    Some(origin) if (pos - origin).length() > 1.0 => {
                        let d = pos - origin;
                        let angle = f64::from(d.y.atan2(d.x).to_degrees());
                        let snapped_angle = (angle / angle_increment).round() * angle_increment;
                        let rad = (snapped_angle as f32).to_radians();
                        let snapped = origin + egui::vec2(rad.cos(), rad.sin()) * d.length();
                        let sel = pick(self.control(), &self.control_placement(), &self.cage_mesh, &self.camera, aspect, rect, snapped);
                        (sel, Some(snapped))
                    }
                    _ => (self.hover, pointer),
                }
            }
            _ => (self.hover, pointer),
        };
        let Some(hovered) = self.knife_hovered_point(aspect, rect, pointer, shift, ctrl, hover_sel) else {
            if let Some(k) = self.knife_mut() {
                k.ahead.clear();
                k.plan = None;
                k.refused = None;
            }
            return;
        };
        // Back near where the path started, the hovered point *is* the first point: the loop
        // closes exactly, and the click that lands there commits it (the owner, 2026-10-24).
        let hovered = match (path.first(), pointer) {
            (Some(&first), Some(pos)) if path.len() >= 2 && self.knife_closes(aspect, rect, pos, first, hovered) => first,
            _ => hovered,
        };
        let ahead = match path.last() {
            None => vec![hovered],
            Some(&last) => match self.knife_trace(aspect, rect, last, hovered) {
                Some(traced) => traced,
                None => {
                    if let Some(k) = self.knife_mut() {
                        k.ahead.clear();
                        k.plan = None;
                        k.refused = Some(("draw the cut over the cage \u{2014} this line leaves it".to_string(), Vec::new()));
                    }
                    return;
                }
            },
        };
        let (plan, refused) = if path.is_empty() {
            (None, None)
        } else {
            let mut whole = path.clone();
            whole.extend(ahead.iter().copied());
            match knife::plan(self.cage(), &whole) {
                Ok(plan) => (Some(plan), None),
                Err(e) => (None, Some((e.reason, e.highlight.faces))),
            }
        };
        if let Some(k) = self.knife_mut() {
            k.ahead = ahead;
            k.plan = plan;
            k.refused = refused;
        }
    }

    /// The paths a Cut Through line from `a` to `b` on screen resolves to: the plane through the
    /// eye and the line, crossed with every edge of the faces the panel's Cut Through faces
    /// setting considers (All, Front-facing or Selected), the crossings within the line's span
    /// chained face to face. A face the plane crosses more than twice -- a saddle -- is refused
    /// by name.
    pub(crate) fn through_paths(&self, aspect: f32, rect: egui::Rect, a: egui::Pos2, b: egui::Pos2) -> Result<Vec<Vec<CutPoint>>, String> {
        use std::collections::{HashMap, HashSet};
        let mesh = self.cage();
        let placement = self.control_placement();
        let dir = b - a;
        if dir.length_sq() < 4.0 {
            return Err("draw a longer line".to_string());
        }
        let (oa, da) = self.camera.pick_ray(aspect, a, rect);
        let (ob, db) = self.camera.pick_ray(aspect, b, rect);
        let (oa, da, ob, db) = (oa.cast::<f64>(), da.cast::<f64>(), ob.cast::<f64>(), db.cast::<f64>());
        let mut n = da.cross(&db);
        if n.norm() < 1e-9 {
            n = (ob - oa).cross(&da);
        }
        let n = nalgebra::Unit::try_new(n, 1e-12).ok_or_else(|| "the line is edge-on to the view".to_string())?;
        let world = |p: Point3<f64>| placement.transform_point(&p);
        let along = |p: Point3<f64>| {
            let w = world(p);
            self.camera.project_to_screen(aspect, rect, Point3::new(w.x as f32, w.y as f32, w.z as f32)).map(|s| (s - a).dot(dir) / dir.length_sq())
        };
        let through_faces = self.knife().map(|k| k.through_faces).unwrap_or_default();
        let faces: Vec<FaceId> = match through_faces {
            ThroughFaces::Selected => self.selection.parts().iter().filter_map(|s| if let Selection::Face(f) = s { Some(*f) } else { None }).collect(),
            ThroughFaces::FrontFacing => mesh.face_ids().filter(|&f| self.knife_face_front_facing(f)).collect(),
            ThroughFaces::All => mesh.face_ids().collect(),
        };
        let considered: HashSet<FaceId> = faces.iter().copied().collect();

        // One crossing per edge, keyed by the edge; each face's crossings by the edges' keys.
        let mut crossing: HashMap<(usize, usize), CutPoint> = HashMap::new();
        let mut per_face: HashMap<FaceId, Vec<(usize, usize)>> = HashMap::new();
        for &f in &faces {
            let c = mesh.face_vertices(f);
            for i in 0..4 {
                let (p, q) = (c[i], c[(i + 1) % 4]);
                let key = (p.index().min(q.index()), p.index().max(q.index()));
                if let Some(_) = crossing.get(&key) {
                    per_face.entry(f).or_default().push(key);
                    continue;
                }
                let (wp, wq) = (world(mesh.vertex_position(p)), world(mesh.vertex_position(q)));
                let d = (wq - wp).dot(&n);
                if d.abs() < 1e-12 {
                    continue;
                }
                let t = (oa - wp).dot(&n) / d;
                if !(t > 0.0 && t < 1.0) {
                    continue;
                }
                let x = mesh.vertex_position(p) + (mesh.vertex_position(q) - mesh.vertex_position(p)) * t;
                let Some(u) = along(x) else { continue };
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                // Spelt from the lower index, so the two faces of the edge name one point.
                let (from, to, t) = if p.index() < q.index() { (p, q, t) } else { (q, p, 1.0 - t) };
                crossing.insert(key, CutPoint::Edge { edge: (from, to), t });
                per_face.entry(f).or_default().push(key);
            }
        }
        // A face the edge list reached only once can still be on both sides of a crossing that
        // its neighbour found first: gather again so every considered face lists every crossing
        // on its rim.
        per_face.clear();
        for &f in &faces {
            let c = mesh.face_vertices(f);
            for i in 0..4 {
                let (p, q) = (c[i], c[(i + 1) % 4]);
                let key = (p.index().min(q.index()), p.index().max(q.index()));
                if crossing.contains_key(&key) {
                    per_face.entry(f).or_default().push(key);
                }
            }
        }
        if let Some((f, _)) = per_face.iter().find(|(_, keys)| keys.len() > 2) {
            return Err(format!("the line crosses face {} more than twice \u{2014} it is folded; cut it by hand", f.index()));
        }
        if crossing.is_empty() {
            return Err("the line crosses no edge of the cage".to_string());
        }

        // The faces either side of a crossing's edge, among those considered.
        let faces_of_key = |key: (usize, usize)| -> Vec<FaceId> {
            faces_of_edge(mesh, VertexId::from_index(key.0), VertexId::from_index(key.1)).into_iter().filter(|f| considered.contains(f)).collect()
        };
        let other_crossing = |f: FaceId, key: (usize, usize)| -> Option<(usize, usize)> {
            per_face.get(&f).and_then(|keys| keys.iter().copied().find(|k| *k != key))
        };
        let mut keys: Vec<(usize, usize)> = crossing.keys().copied().collect();
        keys.sort();
        let is_end = |key: (usize, usize)| faces_of_key(key).iter().any(|f| per_face.get(f).is_none_or(|k| k.len() < 2)) || faces_of_key(key).len() < 2;
        let mut used: HashSet<(usize, usize)> = HashSet::new();
        let mut paths: Vec<Vec<CutPoint>> = Vec::new();
        let walk = |start: (usize, usize), used: &mut HashSet<(usize, usize)>| -> Vec<CutPoint> {
            let mut path = vec![crossing[&start]];
            let mut visited: HashSet<FaceId> = HashSet::new();
            used.insert(start);
            let mut at = start;
            loop {
                let next = faces_of_key(at)
                    .into_iter()
                    .filter(|f| !visited.contains(f))
                    .find_map(|f| other_crossing(f, at).map(|k| (f, k)));
                let Some((f, k)) = next else { break };
                visited.insert(f);
                if k == start {
                    path.push(crossing[&start]);
                    break;
                }
                if used.contains(&k) {
                    break;
                }
                path.push(crossing[&k]);
                used.insert(k);
                at = k;
            }
            path
        };
        // Open paths first, from their ends, then whatever is left closes on itself.
        for &key in keys.iter().filter(|&&k| is_end(k)) {
            if !used.contains(&key) {
                paths.push(walk(key, &mut used));
            }
        }
        for &key in &keys {
            if !used.contains(&key) {
                paths.push(walk(key, &mut used));
            }
        }
        paths.retain(|p| p.len() >= 2);
        Ok(paths)
    }

    /// A click: the traced segment joins the path. `true` when the click was the knife's. A click
    /// on empty canvas with no path begun draws a Cut Through line: its first point, then its
    /// second, which is read against the view until Enter cuts it.
    pub(crate) fn knife_click(&mut self, _shift: bool, pos: Option<egui::Pos2>, aspect: f32, rect: egui::Rect) -> bool {
        let Some(k) = self.knife() else { return false };
        if k.ahead.is_empty() {
            if !k.path.is_empty() {
                return false;
            }
            let Some(pos) = pos else { return false };
            match k.through {
                None | Some((_, Some(_))) => {
                    if let Some(k) = self.knife_mut() {
                        k.through = Some((pos, None));
                        k.through_paths.clear();
                    }
                    self.say("Cut through: click where the line ends; it cuts every face it spans. Esc cancels");
                }
                Some((a, None)) => {
                    if let Some(k) = self.knife_mut() {
                        k.through = Some((a, Some(pos)));
                    }
                    self.say("Cut through: turn the view to line it up; Enter cuts, Esc cancels, a click draws another line");
                }
            }
            return true;
        }
        if let Some((why, _)) = &k.refused {
            let why = why.clone();
            self.warn(why);
            return true;
        }
        let ahead = k.ahead.clone();
        let n = ahead.len();
        let closes = k.path.len() >= 2 && k.path.first() == ahead.last();
        let prior_last = k.path.last().copied();
        if let Some(k) = self.knife_mut() {
            k.path.extend(ahead.iter().copied());
            k.ahead.clear();
        }
        // The newly placed point becomes active: its offset is read and written from the end
        // nearer the point before it, or the cursor for the path's first point
        // (`cage-knife-spec.md` §3).
        self.knife_set_active_ref_end(&ahead, prior_last, pos, aspect, rect);
        // A closed loop is a finished cut: committed here, and the next click starts a new path
        // rather than growing one that could only cross itself.
        if closes {
            self.knife_commit();
            return true;
        }
        let count = self.knife().map_or(0, |k| k.path.len());
        self.say(format!(
            "Knife: {count} point{} on the path{}; click the next, Enter to cut, Backspace to take one back",
            if count == 1 { "" } else { "s" },
            if n > 1 { format!(" ({} edge crossings traced)", n - 1) } else { String::new() }
        ));
        true
    }

    /// Enter: the path committed as one `cut` line, its new edges left selected -- or a placed Cut
    /// Through line's paths, one line each, in the points they resolved to.
    pub(crate) fn knife_commit(&mut self) {
        let Some(k) = self.knife() else { return };
        if let Some((_, Some(_))) = k.through {
            let paths: Vec<Vec<CutPoint>> = k.through_paths.iter().filter(|(_, plan)| plan.is_ok()).map(|(p, _)| p.clone()).collect();
            let refused = k.through_paths.iter().filter(|(_, plan)| plan.is_err()).count();
            if paths.is_empty() {
                let why = k.through_paths.iter().find_map(|(_, p)| p.as_ref().err().map(|(w, _)| w.clone())).unwrap_or_else(|| "nothing to cut".to_string());
                self.warn(format!("Cut through: {why}"));
                return;
            }
            if let Some(k) = self.knife_mut() {
                k.through = None;
                k.through_paths.clear();
            }
            let mut made = 0;
            for path in &paths {
                let line = format!("knife points={}", path.iter().map(|&p| spell(p)).collect::<Vec<_>>().join(","));
                let moves = self.document_moves;
                self.run_command(&line);
                if self.document_moves != moves {
                    made += 1;
                }
            }
            self.say(format!(
                "Cut through: {made} cut{} made{}",
                if made == 1 { "" } else { "s" },
                if refused > 0 { format!(", {refused} refused") } else { String::new() }
            ));
            return;
        }
        if k.path.len() < 2 {
            self.warn("Knife: a cut needs two points");
            return;
        }
        let path = k.path.clone();
        let plan = knife::plan(self.cage(), &path);
        let line = format!("knife points={}", path.iter().map(|&p| spell(p)).collect::<Vec<_>>().join(","));
        let moves = self.document_moves;
        self.run_command(&line);
        if let Some(k) = self.knife_mut() {
            k.path.clear();
            k.ahead.clear();
            k.plan = None;
            k.refused = None;
        }
        if self.document_moves != moves
            && let Ok(plan) = plan
        {
            // The edges on the drawn path, by the numbering the cut gave them.
            let mesh = self.cage();
            let edges: Vec<(VertexId, VertexId)> = mesh.edges().filter(|&(a, b)| plan.path_edges.iter().any(|(p, q)| {
                let (pa, pb) = (mesh.vertex_position(a), mesh.vertex_position(b));
                ((pa - p).norm() < 1e-9 && (pb - q).norm() < 1e-9) || ((pa - q).norm() < 1e-9 && (pb - p).norm() < 1e-9)
            })).collect();
            self.selection.clear();
            for (a, b) in edges {
                self.selection.toggle(Selection::Edge(a, b));
            }
        }
    }

    /// Backspace: the last placed point taken back, with the crossings traced to it.
    pub(crate) fn knife_back(&mut self) {
        let Some(k) = self.knife_mut() else { return };
        // Back to the previous *clicked* point: crossings were traced, not clicked, but the path
        // does not record which was which, so one point goes.
        k.path.pop();
        k.ahead.clear();
        k.plan = None;
        k.active_ref_end = None;
        let left = k.path.len();
        self.say(format!("Knife: {left} point{} left on the path", if left == 1 { "" } else { "s" }));
    }

    /// Esc: the path discarded. `true` when there was one; a second Esc puts the tool down.
    pub(crate) fn knife_discard(&mut self) -> bool {
        let Some(k) = self.knife_mut() else { return false };
        if k.through.is_some() {
            k.through = None;
            k.through_paths.clear();
            self.say("Cut through: line discarded; Esc again puts the knife down");
            return true;
        }
        if k.path.is_empty() {
            return false;
        }
        k.path.clear();
        k.ahead.clear();
        k.plan = None;
        k.refused = None;
        k.active_ref_end = None;
        self.say("Knife: path discarded; Esc again puts the knife down");
        true
    }

    /// Sets the newly extended path's active point -- the one a typed or dragged offset reads
    /// and writes -- and its reference end: nearer the point before it (`ahead`'s own
    /// second-to-last, or the path's previous last point when `ahead` has one element), or nearer
    /// the cursor for the path's very first point, where there is no previous point at all.
    fn knife_set_active_ref_end(&mut self, ahead: &[CutPoint], prior_last: Option<CutPoint>, pos: Option<egui::Pos2>, aspect: f32, rect: egui::Rect) {
        let Some(&last) = ahead.last() else { return };
        let CutPoint::Edge { edge: (a, b), .. } = last else {
            if let Some(k) = self.knife_mut() {
                k.active_ref_end = None;
            }
            return;
        };
        let prev = if ahead.len() >= 2 { Some(ahead[ahead.len() - 2]) } else { prior_last };
        let mesh = self.cage();
        let placement = self.control_placement();
        let ref_end = match prev {
            Some(p) => {
                let target = placement.transform_point(&position(mesh, p));
                let da = (placement.transform_point(&mesh.vertex_position(a)) - target).norm();
                let db = (placement.transform_point(&mesh.vertex_position(b)) - target).norm();
                if da <= db { a } else { b }
            }
            None => match pos {
                Some(pos) => {
                    let sa = self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, a));
                    let sb = self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, b));
                    match (sa, sb) {
                        (Some(sa), Some(sb)) if (sb - pos).length() < (sa - pos).length() => b,
                        _ => a,
                    }
                }
                None => a,
            },
        };
        if let Some(k) = self.knife_mut() {
            k.active_ref_end = Some(ref_end);
        }
    }

    /// The active point's offset from its reference end, in the panel's current unit, or `None`
    /// where there is no active edge point to read it from.
    fn knife_active_offset(&self) -> Option<f64> {
        let k = self.knife()?;
        let ref_end = k.active_ref_end?;
        let CutPoint::Edge { edge: (a, b), t } = *k.path.last()? else { return None };
        let mesh = self.cage();
        let len = (mesh.vertex_position(a) - mesh.vertex_position(b)).norm();
        let t_from_ref = if ref_end == a { t } else { 1.0 - t };
        Some(match k.offset_mode {
            OffsetMode::Mm => t_from_ref * len,
            OffsetMode::Fraction => t_from_ref,
        })
    }

    /// `t` from `edge.0` for a point at `value` (mm or a fraction, per `mode`) from `ref_end`,
    /// clamped to the edge -- offset is length along the **cage** edge (`cage-knife-spec.md` §3).
    fn knife_t_from_offset(&self, edge: (VertexId, VertexId), ref_end: VertexId, mode: OffsetMode, value: f64) -> f64 {
        t_from_offset(self.cage(), edge, ref_end, mode, value)
    }

    /// A typed offset (§4.3, no click needed since Offset is the focused row): moves the active
    /// point along its own edge to exactly `value`, read from its reference end.
    fn knife_set_offset(&mut self, value: f64) -> Result<(), String> {
        let Some(k) = self.knife() else { return Ok(()) };
        let Some(ref_end) = k.active_ref_end else {
            return Err("place a point on an edge first".to_string());
        };
        let Some(&CutPoint::Edge { edge, .. }) = k.path.last() else {
            return Err("the active point is a vertex, which has no offset".to_string());
        };
        let mode = k.offset_mode;
        let t = self.knife_t_from_offset(edge, ref_end, mode, value);
        if let Some(k) = self.knife_mut()
            && let Some(last) = k.path.last_mut()
        {
            *last = CutPoint::Edge { edge, t };
        }
        Ok(())
    }

    /// Whether `pos` is near the active point on screen, within the handle's grab distance --
    /// the gate that keeps a left-drag starting there from also tumbling the camera.
    pub(crate) fn knife_active_point_near(&self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) -> bool {
        const GRAB: f32 = 10.0;
        let Some(k) = self.knife() else { return false };
        let Some(last) = k.path.last().copied() else { return false };
        if !matches!(last, CutPoint::Edge { .. }) {
            return false;
        }
        let mesh = self.cage();
        let placement = self.control_placement();
        let w = placement.transform_point(&position(mesh, last));
        self.camera
            .project_to_screen(aspect, rect, Point3::new(w.x as f32, w.y as f32, w.z as f32))
            .is_some_and(|s| (s - pos).length() <= GRAB)
    }

    /// Left-drag after placing an edge point slides it along its edge; the offset readout updates
    /// live (`cage-knife-spec.md` §3). The edge itself does not change, only `t` on it.
    pub(crate) fn knife_drag_active(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let Some(k) = self.knife() else { return };
        let Some(CutPoint::Edge { edge: (a, b), .. }) = k.path.last().copied() else { return };
        let mesh = self.cage();
        let placement = self.control_placement();
        let (sa, sb) = (
            self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, a)),
            self.camera.project_to_screen(aspect, rect, vertex_pos(mesh, &placement, b)),
        );
        let Some((sa, sb)) = sa.zip(sb) else { return };
        let d = sb - sa;
        if d.length_sq() < 1.0 {
            return;
        }
        let t = (((pos - sa).dot(d)) / d.length_sq()).clamp(0.0, 1.0) as f64;
        if let Some(k) = self.knife_mut()
            && let Some(last) = k.path.last_mut()
        {
            *last = CutPoint::Edge { edge: (a, b), t };
        }
    }

    /// Whether face `f` faces the camera, for Cut Through's Front-facing setting.
    fn knife_face_front_facing(&self, f: FaceId) -> bool {
        let mesh = self.cage();
        let placement = self.control_placement();
        let n = fieldform_cage::validate::face_normal(mesh, f);
        let world_n = placement.transform_vector(&n);
        world_n.dot(&(-self.camera.forward().cast::<f64>())) > 0.0
    }

    pub(crate) fn draw_knife(&self, painter: &egui::Painter, aspect: f32, rect: egui::Rect) {
        let Some(k) = self.knife() else { return };
        let mesh = self.cage();
        let placement = self.control_placement();
        let to_world = |p: &Point3<f64>| {
            let w = placement.transform_point(p);
            Point3::new(w.x as f32, w.y as f32, w.z as f32)
        };
        let project = |p: &Point3<f64>| self.camera.project_to_screen(aspect, rect, to_world(p));
        let path_stroke = egui::Stroke::new(2.5_f32, PREVIEW_COLOR);
        let front_stroke = egui::Stroke::new(2.0_f32, PROPAGATED_COLOR);
        let draw_plan = |plan: &Plan| {
            for (p, q) in &plan.path_edges {
                if let (Some(a), Some(b)) = (project(p), project(q)) {
                    painter.line_segment([a, b], path_stroke);
                }
            }
            for (p, q) in &plan.front_edges {
                draw_dashed_polyline(painter, &self.camera, aspect, rect, &[to_world(p), to_world(q)], front_stroke);
            }
            for p in &plan.points {
                if let Some(s) = project(p) {
                    painter.circle_filled(s, 3.0, PREVIEW_COLOR);
                }
            }
        };
        // A Cut Through line: the line itself, fixed on screen, and what it cuts in this view.
        if let Some((a, b)) = k.through {
            let end = b.or(k.cursor);
            if let Some(end) = end {
                painter.line_segment([a, end], egui::Stroke::new(1.5_f32, PREVIEW_COLOR));
            }
            painter.circle_filled(a, 4.0, PREVIEW_COLOR);
            if let Some(b) = b {
                painter.circle_filled(b, 4.0, PREVIEW_COLOR);
            }
            for (_, plan) in &k.through_paths {
                match plan {
                    Ok(plan) => draw_plan(plan),
                    Err((why, faces)) => {
                        for &f in faces {
                            draw_highlight(painter, mesh, &placement, &self.camera, aspect, rect, Selection::Face(f), ERROR_COLOR, ERROR_FILL, 3.0);
                        }
                        painter.text(a + egui::vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, why, egui::FontId::proportional(12.0), ERROR_COLOR);
                    }
                }
            }
            return;
        }
        if let Some(plan) = &k.plan {
            draw_plan(plan);
        } else {
            // No plan yet: the placed points and the traced segment, as a polyline.
            let mut line: Vec<Point3<f32>> = k.path.iter().map(|&p| to_world(&position(mesh, p))).collect();
            line.extend(k.ahead.iter().map(|&p| to_world(&position(mesh, p))));
            for w in line.windows(2) {
                if let (Some(a), Some(b)) = (
                    self.camera.project_to_screen(aspect, rect, w[0]),
                    self.camera.project_to_screen(aspect, rect, w[1]),
                ) {
                    painter.line_segment([a, b], path_stroke);
                }
            }
        }
        for &p in &k.path {
            if let Some(s) = project(&position(mesh, p)) {
                painter.circle_filled(s, 4.0, PREVIEW_COLOR);
            }
        }
        // The active point's reference end: a short tick across it, so a typed or dragged
        // offset is read against the end it is measured from (`cage-knife-spec.md` §3).
        if let (Some(&last), Some(end)) = (k.path.last(), k.active_ref_end)
            && let Some(p) = project(&position(mesh, last))
            && let Some(s) = project(&mesh.vertex_position(end))
        {
            let dir = (s - p).normalized();
            let perp = egui::vec2(-dir.y, dir.x) * 6.0;
            painter.line_segment([s - perp, s + perp], egui::Stroke::new(2.0_f32, PREVIEW_COLOR));
        }
        // The point under the pointer, before and between clicks: a ring where the next point
        // would land, so the knife reads as ready wherever it hovers (the owner, 2026-10-24).
        if let Some(&next) = k.ahead.last()
            && let Some(s) = project(&position(mesh, next))
        {
            painter.circle_filled(s, 3.0, PREVIEW_COLOR);
            painter.circle_stroke(s, 7.0, egui::Stroke::new(1.5_f32, PREVIEW_COLOR));
        }
        if let Some((why, faces)) = &k.refused {
            for &f in faces {
                draw_highlight(painter, mesh, &placement, &self.camera, aspect, rect, Selection::Face(f), ERROR_COLOR, ERROR_FILL, 3.0);
            }
            let at = k
                .path
                .last()
                .and_then(|&p| project(&position(mesh, p)))
                .unwrap_or(rect.left_top() + egui::vec2(12.0, 24.0));
            painter.text(at + egui::vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, why, egui::FontId::proportional(12.0), ERROR_COLOR);
        }
    }

    /// The knife's panel rows, in the order `cage-tools-instructions.md` §1 lists them: Pick
    /// type, Offset mode, Offset, Angle increment, Cut Through faces.
    pub(crate) fn knife_panel_rows(&self) -> Vec<Row> {
        let Some(k) = self.knife() else { return Vec::new() };
        let offset_text = self.knife_active_offset().map(echo::number).unwrap_or_else(|| "\u{2014}".to_string());
        vec![
            Row { key: "pick_type", text: k.pick_type.word().to_string(), unit: commands::Unit::Word },
            Row { key: "offset_mode", text: k.offset_mode.word().to_string(), unit: commands::Unit::Word },
            Row {
                key: "offset",
                text: offset_text,
                unit: if k.offset_mode == OffsetMode::Mm { commands::Unit::Mm } else { commands::Unit::Plain },
            },
            Row { key: "angle_increment", text: echo::number(k.angle_increment), unit: commands::Unit::Degrees },
            Row { key: "cut_through_faces", text: k.through_faces.word().to_string(), unit: commands::Unit::Word },
        ]
    }

    /// Confirms a row: a word checked against its choices, a number applied to the setting or,
    /// for Offset, to the active point.
    pub(crate) fn knife_panel_set(&mut self, key: &'static str, text: &str) -> Result<(), String> {
        if self.knife().is_none() {
            return Ok(());
        }
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        match key {
            "pick_type" => {
                let Some(v) = PickType::from_word(&text.to_ascii_lowercase()) else {
                    return Err("pick type is one of vertices, edges, both".to_string());
                };
                if let Some(k) = self.knife_mut() {
                    k.pick_type = v;
                }
                Ok(())
            }
            "offset_mode" => {
                let Some(v) = OffsetMode::from_word(&text.to_ascii_lowercase()) else {
                    return Err("offset mode is one of mm, fraction".to_string());
                };
                if let Some(k) = self.knife_mut() {
                    k.offset_mode = v;
                }
                Ok(())
            }
            "cut_through_faces" => {
                let Some(v) = ThroughFaces::from_word(&text.to_ascii_lowercase()) else {
                    return Err("cut through faces is one of all, front-facing, selected".to_string());
                };
                if let Some(k) = self.knife_mut() {
                    k.through_faces = v;
                }
                Ok(())
            }
            "offset" => {
                let value: f64 = text.parse().map_err(|_| format!("{text:?} is not a number"))?;
                if !value.is_finite() {
                    return Err("that is not a finite number".to_string());
                }
                self.knife_set_offset(value)
            }
            "angle_increment" => {
                let value: f64 = text.parse().map_err(|_| format!("{text:?} is not a number"))?;
                if !(value.is_finite() && value > 0.0) {
                    return Err("an angle increment is more than nought".to_string());
                }
                if let Some(k) = self.knife_mut() {
                    k.angle_increment = value;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
