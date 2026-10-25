// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The cage's topology edits as operations: one command shape for all of them, and the console
//! verbs that build it.
//!
//! **Every edit is a snapshot pair, and undo and redo restore snapshots** -- they never run a
//! primitive again. The primitive runs once, when the verb builds the command, on the mesh as it
//! stands; what goes on the stack is the mesh before and the mesh after, each as its positions
//! and quads. That is the direct layer's rule (commands reverse against the exact state they were
//! applied to) with nothing left to recompute, and it is what lets a refusal happen before anything
//! is on the stack: a primitive that refuses builds no command.

use crate::{Args, Command, Document, OpError, OpRegistry, Value};
use fieldform_cage::{edit, CageBody, EditError, FaceId, MeshSnapshot, QuadMesh, VertexId};
use fieldform_scene::CageId;
use nalgebra::Point3;

/// A cage's mesh replaced outright: the one command every topology edit is.
pub struct SetCageMesh {
    cage: CageId,
    name: &'static str,
    before: MeshSnapshot,
    after: MeshSnapshot,
}

impl SetCageMesh {
    pub fn new(cage: CageId, name: &'static str, before: MeshSnapshot, after: MeshSnapshot) -> Self {
        Self { cage, name, before, after }
    }

    fn put(&self, doc: &mut Document, snapshot: &MeshSnapshot) -> Result<(), OpError> {
        let body = doc.cage_body_mut(self.cage).ok_or_else(|| OpError::Failed {
            op: self.name.to_string(),
            detail: format!("cage {} is no longer in the document", self.cage),
        })?;
        body.set_mesh(snapshot.restore());
        Ok(())
    }
}

impl Command for SetCageMesh {
    fn name(&self) -> &str {
        self.name
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.put(doc, &self.after.clone())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.put(doc, &self.before.clone())
    }

    fn byte_size(&self) -> usize {
        self.before.byte_size() + self.after.byte_size()
    }
}

/// The cage a verb acts on, by `cage=<hex>`, or the only one.
fn body<'d>(op: &str, args: &Args, doc: &'d Document) -> Result<(CageId, &'d CageBody), OpError> {
    let id = crate::ops::resolve_cage(op, args, doc)?;
    let body = doc.cage_body(id).ok_or_else(|| crate::ops::missing_cage(op, id))?;
    Ok((id, body))
}

fn bad(op: &str, arg: &str, detail: impl Into<String>) -> OpError {
    OpError::BadArg { op: op.to_string(), arg: arg.to_string(), detail: detail.into() }
}

/// A refusal, said as the primitive said it. The elements it would highlight are the tool's
/// to draw; a typed line gets the reason.
fn refused(op: &str, e: EditError) -> OpError {
    OpError::Failed { op: op.to_string(), detail: e.reason }
}

/// `vertices=3,7,12`: indices, each checked against the mesh before anything is built, exactly as
/// `cage.translate_vertices` checks its own.
fn vertices(op: &str, args: &Args, key: &str, mesh: &QuadMesh) -> Result<Vec<VertexId>, OpError> {
    let raw = args.require(op, key, Value::as_int_list, "a list of vertex indices")?;
    raw.into_iter()
        .map(|i| {
            if i < 0 || i as usize >= mesh.vertex_count() {
                return Err(bad(op, key, format!("index {i} is out of range (cage has {})", mesh.vertex_count())));
            }
            Ok(VertexId::from_index(i as usize))
        })
        .collect()
}

fn faces(op: &str, args: &Args, key: &str, mesh: &QuadMesh) -> Result<Vec<FaceId>, OpError> {
    let raw = args.require(op, key, Value::as_int_list, "a list of face indices")?;
    raw.into_iter()
        .map(|i| {
            if i < 0 || i as usize >= mesh.face_count() {
                return Err(bad(op, key, format!("index {i} is out of range (cage has {} faces)", mesh.face_count())));
            }
            Ok(FaceId::from_index(i as usize))
        })
        .collect()
}

/// `a-b`: two vertex indices joined by a dash, the spelling an edge has on the console.
fn edge(op: &str, key: &str, text: &str, mesh: &QuadMesh) -> Result<(VertexId, VertexId), OpError> {
    let (a, b) = text
        .split_once('-')
        .ok_or_else(|| bad(op, key, format!("expected an edge as two vertex indices joined by a dash, got {text:?}")))?;
    let parse = |s: &str| -> Result<VertexId, OpError> {
        let i: usize = s.trim().parse().map_err(|_| bad(op, key, format!("{s:?} is not a vertex index")))?;
        if i >= mesh.vertex_count() {
            return Err(bad(op, key, format!("index {i} is out of range (cage has {})", mesh.vertex_count())));
        }
        Ok(VertexId::from_index(i))
    };
    Ok((parse(a)?, parse(b)?))
}

/// `edges=1-2,2-3`: a comma list of edges.
fn edges(op: &str, args: &Args, key: &str, mesh: &QuadMesh) -> Result<Vec<(VertexId, VertexId)>, OpError> {
    let text = args.require(op, key, |v| Some(v.text()), "a list of edges, each two vertex indices joined by a dash")?;
    text.split(',').map(|e| edge(op, key, e, mesh)).collect()
}

/// `to=1,2,3;4,5,6`: one point per vertex, semicolon between points.
fn points(op: &str, args: &Args, key: &str, count: usize) -> Result<Vec<Point3<f64>>, OpError> {
    let text = args.require(op, key, |v| Some(v.text()), "points as x,y,z joined by semicolons")?;
    let parsed: Vec<Point3<f64>> = text
        .split(';')
        .map(|p| crate::registry::parse_triple(p).map(Point3::from))
        .collect::<Option<_>>()
        .ok_or_else(|| bad(op, key, format!("expected points as x,y,z joined by semicolons, got {text:?}")))?;
    if parsed.len() != count {
        return Err(bad(op, key, format!("{} points for {count} vertices", parsed.len())));
    }
    Ok(parsed)
}

/// `mode=region` (the default) or `mode=individual`.
fn mode_of(op: &str, args: &Args) -> Result<edit::extrude::Mode, OpError> {
    match args.get("mode").map(|v| v.text()).as_deref() {
        None | Some("region") => Ok(edit::extrude::Mode::Region),
        Some("individual") => Ok(edit::extrude::Mode::Individual),
        Some(other) => Err(bad(op, "mode", format!("expected region or individual, got {other:?}"))),
    }
}

fn number(op: &str, args: &Args, key: &str, default: f64) -> Result<f64, OpError> {
    match args.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .or_else(|| v.text().trim().parse().ok())
            .ok_or_else(|| bad(op, key, "expected a number")),
    }
}

/// A finished edit as its command: the mesh before, the primitive's result after.
fn command(cage: CageId, name: &'static str, before: &QuadMesh, after: QuadMesh) -> Box<dyn Command> {
    Box::new(SetCageMesh::new(cage, name, MeshSnapshot::of(before), MeshSnapshot::of(&after)))
}

/// The verbs. Each reads its keys, checks every index against the live mesh, runs the primitive
/// once, and hands back the snapshot pair -- or the primitive's refusal, with nothing built.
pub(crate) fn register(registry: &mut OpRegistry) {
    registry.register("cage.insert_ring", |args, doc| {
        const OP: &str = "cage.insert_ring";
        let (id, body) = body(OP, args, doc)?;
        let text = args.require(OP, "edge", |v| Some(v.text()), "an edge as two vertex indices joined by a dash")?;
        let e = edge(OP, "edge", &text, body.mesh())?;
        let t = number(OP, args, "t", 0.5)?;
        let count = number(OP, args, "count", 1.0)?;
        if count < 1.0 || count.fract() != 0.0 || count > 64.0 {
            return Err(bad(OP, "count", "expected a whole number of rings from 1 to 64"));
        }
        // `count` rings evenly spaced along the edge, at k/(n+1) from its first end, as one
        // step. Each is inserted on the mesh the one before it made, through the piece of the
        // named edge that is left -- from the last ring's vertex on it to the far end -- so the
        // k-th ring sits at 1/(n+2-k) of that piece. One ring takes `t` as typed.
        let mut mesh = body.mesh().clone();
        let n = count as usize;
        let mut from = e;
        for k in 1..=n {
            let s = if n == 1 { t } else { 1.0 / (n as f64 + 2.0 - k as f64) };
            let (pa, pb) = (mesh.vertex_position(from.0), mesh.vertex_position(from.1));
            let result = edit::insert_ring(&mesh, from, s).map_err(|er| refused(OP, er))?;
            mesh = result.mesh;
            // The ring's vertex on the named edge, found where the primitive put it rather than
            // by how it numbered what it added.
            let at = pa + (pb - pa) * s;
            let on_edge = result
                .new_vertices
                .iter()
                .copied()
                .find(|&v| (mesh.vertex_position(v) - at).norm() < 1e-9)
                .ok_or_else(|| OpError::Failed {
                    op: OP.to_string(),
                    detail: "the ring left no vertex on the edge it was asked through".to_string(),
                })?;
            from = (on_edge, from.1);
        }
        Ok(command(id, "cage.insert_ring", body.mesh(), mesh))
    });

    registry.register("cage.delete_faces", |args, doc| {
        const OP: &str = "cage.delete_faces";
        let (id, body) = body(OP, args, doc)?;
        let gone = faces(OP, args, "faces", body.mesh())?;
        let result = edit::delete_faces(body.mesh(), &gone).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.delete_faces", body.mesh(), result.mesh))
    });

    registry.register("cage.fill_hole", |args, doc| {
        const OP: &str = "cage.fill_hole";
        let (id, body) = body(OP, args, doc)?;
        let text = args.require(OP, "edge", |v| Some(v.text()), "an edge of the hole's rim, two vertex indices joined by a dash")?;
        let e = edge(OP, "edge", &text, body.mesh())?;
        let result = edit::fill_hole(body.mesh(), e).map_err(|er| refused(OP, er))?;
        Ok(command(id, "cage.fill_hole", body.mesh(), result.mesh))
    });

    registry.register("cage.bridge_faces", |args, doc| {
        const OP: &str = "cage.bridge_faces";
        let (id, body) = body(OP, args, doc)?;
        let a = faces(OP, args, "a", body.mesh())?;
        let b = faces(OP, args, "b", body.mesh())?;
        let divisions = number(OP, args, "divisions", 1.0)?;
        if divisions < 1.0 || divisions.fract() != 0.0 || divisions > 256.0 {
            return Err(bad(OP, "divisions", "expected a whole number from 1 to 256"));
        }
        let result = edit::bridge_faces(body.mesh(), &a, &b, divisions as usize).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.bridge_faces", body.mesh(), result.mesh))
    });

    registry.register("cage.bridge", |args, doc| {
        const OP: &str = "cage.bridge";
        let (id, body) = body(OP, args, doc)?;
        let a = edges(OP, args, "a", body.mesh())?;
        let b = edges(OP, args, "b", body.mesh())?;
        let divisions = number(OP, args, "divisions", 1.0)?;
        if divisions < 1.0 || divisions.fract() != 0.0 || divisions > 256.0 {
            return Err(bad(OP, "divisions", "expected a whole number from 1 to 256"));
        }
        let result = edit::bridge(body.mesh(), &a, &b, divisions as usize).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.bridge", body.mesh(), result.mesh))
    });

    registry.register("cage.cut", |args, doc| {
        const OP: &str = "cage.cut";
        let (id, body) = body(OP, args, doc)?;
        // `points=12,3-4@0.25,7-8@0.5`: a vertex by its index, an edge point as `a-b@t` with `t`
        // from `a`. One path per line.
        let text = args.require(OP, "points", |v| Some(v.text()), "the cut's points: a vertex index, or an edge point as a-b@t")?;
        let mesh = body.mesh();
        let mut path = Vec::new();
        for item in text.split(',') {
            let item = item.trim();
            let point = match item.split_once('@') {
                Some((e, t)) => {
                    let edge = edge(OP, "points", e, mesh)?;
                    let t: f64 = t.trim().parse().map_err(|_| bad(OP, "points", format!("{t:?} is not a fraction along the edge")))?;
                    if !(t > 0.0 && t < 1.0) {
                        return Err(bad(OP, "points", format!("{item}: the fraction must be between 0 and 1")));
                    }
                    edit::knife::CutPoint::Edge { edge, t }
                }
                None => {
                    let i: usize = item.parse().map_err(|_| bad(OP, "points", format!("{item:?} is not a vertex index or an edge point")))?;
                    if i >= mesh.vertex_count() {
                        return Err(bad(OP, "points", format!("index {i} is out of range (cage has {})", mesh.vertex_count())));
                    }
                    edit::knife::CutPoint::Vertex(VertexId::from_index(i))
                }
            };
            path.push(point);
        }
        let cut = edit::knife::cut(mesh, &path).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.cut", mesh, cut.result.mesh))
    });

    registry.register("cage.weld", |args, doc| {
        const OP: &str = "cage.weld";
        let (id, body) = body(OP, args, doc)?;
        // `pairs=3-7,4-8`: each pair is the vertex that moves, then the one it moves onto.
        let pairs = edges(OP, args, "pairs", body.mesh())?;
        // `at=target` (the default) leaves the second where it is; `at=midpoint` meets halfway.
        let at = args.get("at").map(|v| v.text()).unwrap_or_else(|| "target".to_string());
        let mesh = match at.as_str() {
            "target" => body.mesh().clone(),
            "midpoint" => {
                let mut m = body.mesh().clone();
                for &(from, to) in &pairs {
                    let mid = nalgebra::center(&m.vertex_position(from), &m.vertex_position(to));
                    m.set_vertex_position(to, mid);
                }
                m
            }
            other => return Err(bad(OP, "at", format!("expected target or midpoint, got {other:?}"))),
        };
        let result = edit::weld(&mesh, &pairs).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.weld", body.mesh(), result.mesh))
    });

    registry.register("cage.extrude", |args, doc| {
        const OP: &str = "cage.extrude";
        let (id, body) = body(OP, args, doc)?;
        let lifted = faces(OP, args, "faces", body.mesh())?;
        let mode = mode_of(OP, args)?;
        let direction = match args.get("direction").map(|v| v.text()).as_deref() {
            None | Some("normal") => edit::extrude::Direction::Normal,
            Some("vertex_normals") => edit::extrude::Direction::VertexNormals,
            Some("x") => edit::extrude::Direction::Axis(nalgebra::Vector3::x()),
            Some("y") => edit::extrude::Direction::Axis(nalgebra::Vector3::y()),
            Some("z") => edit::extrude::Direction::Axis(nalgebra::Vector3::z()),
            Some(other) => return Err(bad(OP, "direction", format!("expected normal, vertex_normals, x, y or z, got {other:?}"))),
        };
        let distance = number(OP, args, "distance", 0.0)?;
        let result = edit::extrude::extrude(body.mesh(), &lifted, mode, direction, distance).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.extrude", body.mesh(), result.mesh))
    });

    registry.register("cage.extrude_edges", |args, doc| {
        const OP: &str = "cage.extrude_edges";
        let (id, body) = body(OP, args, doc)?;
        let rim = edges(OP, args, "edges", body.mesh())?;
        let distance = number(OP, args, "distance", 0.0)?;
        let axis = match args.get("direction").map(|v| v.text()).as_deref() {
            None | Some("normal") => None,
            Some("x") => Some(nalgebra::Vector3::x()),
            Some("y") => Some(nalgebra::Vector3::y()),
            Some("z") => Some(nalgebra::Vector3::z()),
            Some(other) => return Err(bad(OP, "direction", format!("expected normal, x, y or z, got {other:?}"))),
        };
        let result = edit::extrude::extrude_boundary_edges(body.mesh(), &rim, distance, axis).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.extrude_edges", body.mesh(), result.mesh))
    });

    registry.register("cage.inset", |args, doc| {
        const OP: &str = "cage.inset";
        let (id, body) = body(OP, args, doc)?;
        let lifted = faces(OP, args, "faces", body.mesh())?;
        let mode = mode_of(OP, args)?;
        let thickness = number(OP, args, "thickness", 0.0)?;
        let depth = number(OP, args, "depth", 0.0)?;
        let (result, _used) = edit::extrude::inset(body.mesh(), &lifted, mode, thickness, depth).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.inset", body.mesh(), result.mesh))
    });

    registry.register("cage.dissolve_edges", |args, doc| {
        const OP: &str = "cage.dissolve_edges";
        let (id, body) = body(OP, args, doc)?;
        let gone = edges(OP, args, "edges", body.mesh())?;
        let result = edit::dissolve_edges(body.mesh(), &gone).map_err(|e| refused(OP, e))?;
        Ok(command(id, "cage.dissolve_edges", body.mesh(), result.mesh))
    });

    registry.register("cage.set_positions", |args, doc| {
        const OP: &str = "cage.set_positions";
        let (id, body) = body(OP, args, doc)?;
        let moved = vertices(OP, args, "vertices", body.mesh())?;
        let to = points(OP, args, "to", moved.len())?;
        let before: Vec<Point3<f64>> = moved.iter().map(|&v| body.mesh().vertex_position(v)).collect();
        Ok(Box::new(crate::ops::SetVertexPositions::new(id, moved, before, to)))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommandStack;
    use fieldform_cage::Cage;

    fn cube_document() -> (Document, CageId) {
        let doc = Document::with_cage(Cage::cube_of(10.0));
        let id = *doc.cages.keys().next().unwrap();
        (doc, id)
    }

    fn registry() -> OpRegistry {
        OpRegistry::default()
    }

    fn args(pairs: &[(&str, &str)]) -> Args {
        let mut a = Args::new();
        for (k, v) in pairs {
            a = a.with(k, Value::Str(v.to_string()));
        }
        a
    }

    /// **Undo and redo restore snapshots, bit for bit, and run nothing again** (spec §4.4).
    #[test]
    fn a_topology_edit_undoes_and_redoes_to_the_exact_mesh() {
        let (mut doc, id) = cube_document();
        let before = doc.cage_body(id).unwrap().geometry_fingerprint();
        let cmd = registry().build("cage.delete_faces", &args(&[("faces", "1")]), &doc).unwrap();
        let mut stack = CommandStack::default();
        stack.execute(&mut doc, cmd).unwrap();
        assert!(doc.cage_is_red(id), "a face gone is a red cage");
        let after = doc.cage_body(id).unwrap().geometry_fingerprint();
        assert!(stack.bytes_held() > 0, "a mesh edit says what it holds");

        stack.undo(&mut doc);
        assert_eq!(doc.cage_body(id).unwrap().geometry_fingerprint(), before);
        assert!(!doc.cage_is_red(id));
        stack.redo(&mut doc);
        assert_eq!(doc.cage_body(id).unwrap().geometry_fingerprint(), after);
    }

    /// Delete a face, fill the hole from the console: the cage is a solid again and `χ` is back.
    #[test]
    fn delete_and_fill_from_the_console_close_the_cage_again() {
        let (mut doc, id) = cube_document();
        let mut stack = CommandStack::default();
        let reg = registry();
        let delete = reg.build("cage.delete_faces", &args(&[("faces", "1")]), &doc).unwrap();
        stack.execute(&mut doc, delete).unwrap();
        let rim = {
            let m = doc.cage_body(id).unwrap().mesh();
            m.edges().find(|&(a, b)| m.edge_faces(a, b).is_none()).unwrap()
        };
        let edge_text = format!("{}-{}", rim.0.index(), rim.1.index());
        let fill = reg.build("cage.fill_hole", &args(&[("edge", &edge_text)]), &doc).unwrap();
        stack.execute(&mut doc, fill).unwrap();
        assert!(!doc.cage_is_red(id));
        assert_eq!(doc.cage_body(id).unwrap().euler_characteristic(), 2);
    }

    /// A refusal builds nothing and leaves the document as it was.
    #[test]
    fn a_refused_edit_puts_nothing_on_the_stack() {
        let (doc, id) = cube_document();
        let before = doc.cage_body(id).unwrap().geometry_fingerprint();
        let err = registry().build("cage.fill_hole", &args(&[("edge", "0-1")]), &doc).err().expect("refused");
        assert!(err.to_string().contains("not on the rim"), "{err}");
        assert_eq!(doc.cage_body(id).unwrap().geometry_fingerprint(), before);
        let err = registry().build("cage.delete_faces", &args(&[("faces", "99")]), &doc).err().expect("refused");
        assert!(err.to_string().contains("out of range"), "{err}");
    }

    /// `count=3` is three evenly spaced rings in one undo step.
    #[test]
    fn insert_ring_with_a_count_makes_evenly_spaced_rings_in_one_step() {
        let (mut doc, id) = cube_document();
        let cmd = registry()
            .build("cage.insert_ring", &args(&[("edge", "0-1"), ("count", "3")]), &doc)
            .unwrap();
        let mut stack = CommandStack::default();
        stack.execute(&mut doc, cmd).unwrap();
        let mesh = doc.cage_body(id).unwrap();
        assert_eq!(mesh.face_count(), 6 + 4 * 3, "three rings, each splitting four faces");
        let mut xs: Vec<f64> = (8..mesh.vertex_count()).map(|i| mesh.vertex_position(VertexId::from_index(i)).x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert_eq!(xs.len(), 3, "three distinct planes: {xs:?}");
        for (x, want) in xs.iter().zip([-2.5, 0.0, 2.5]) {
            assert!((x - want).abs() < 1e-9, "ring at x = {x}, wanted {want}");
        }
        stack.undo(&mut doc);
        assert_eq!(doc.cage_body(id).unwrap().face_count(), 6, "one step back is the cube");
    }

    /// A list typed at the console arrives as text and is read as the list it is.
    #[test]
    fn set_positions_reads_a_typed_list_and_typed_points() {
        let (mut doc, id) = cube_document();
        let cmd = registry()
            .build("cage.set_positions", &args(&[("vertices", "0,1"), ("to", "1,2,3;4,5,6")]), &doc)
            .unwrap();
        CommandStack::default().execute(&mut doc, cmd).unwrap();
        let m = doc.cage_body(id).unwrap();
        assert_eq!(m.vertex_position(VertexId::from_index(0)), Point3::new(1.0, 2.0, 3.0));
        assert_eq!(m.vertex_position(VertexId::from_index(1)), Point3::new(4.0, 5.0, 6.0));
        assert!(registry().build("cage.set_positions", &args(&[("vertices", "0,1"), ("to", "1,2,3")]), &doc).is_err());
    }

    /// The byte budget drops the oldest entry, never the newest.
    #[test]
    fn the_byte_budget_drops_the_oldest_entry_and_keeps_the_newest() {
        let (mut doc, _) = cube_document();
        let one = {
            let cmd = registry().build("cage.delete_faces", &args(&[("faces", "1")]), &doc).unwrap();
            cmd.byte_size()
        };
        let mut stack = CommandStack::default().with_byte_limit(one + one / 2);
        let reg = registry();
        let first = reg.build("cage.insert_ring", &args(&[("edge", "0-1")]), &doc).unwrap();
        stack.execute(&mut doc, first).unwrap();
        let second = reg.build("cage.insert_ring", &args(&[("edge", "0-3")]), &doc).unwrap();
        stack.execute(&mut doc, second).unwrap();
        assert_eq!(stack.history().0.len(), 1, "the first ring's entry went to make room");
        assert!(stack.bytes_held() > one, "and the one left is the newest, kept whatever its size");
        assert!(stack.can_undo());
    }
}
