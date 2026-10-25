// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Built-in operations.

use crate::{Args, Command, Document, OpError, OpRegistry, Value};
use fieldform_cage::VertexId;
use fieldform_scene::{
    BlockKey, CageId, Curve, CurveId, EyeFlag, Graph, Material, NodeId, Ownership, ParamId,
    Parameter, PartId,
};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;
use std::sync::Arc;

/// Resolves which cage an operation acts on.
///
/// A document can hold several cages, so operations name one. But naming it is only
/// *necessary* when there is a choice: with exactly one cage the argument may be omitted, and
/// with several it becomes required. That keeps scripts and keybindings short in the common
/// case without ever guessing when guessing would be wrong.
///
/// Ids arrive as hex strings — the form `CageId`'s `Display` produces — because random 64-bit
/// values exceed `i64` and would come through JSON as negative numbers.
pub(crate) fn resolve_cage(op: &str, args: &Args, doc: &Document) -> Result<CageId, OpError> {
    match args.get("cage") {
        Some(Value::Str(s)) => CageId::from_hex(s)
            .filter(|id| doc.cages.contains_key(id))
            .ok_or_else(|| OpError::BadArg {
                op: op.to_string(),
                arg: "cage".to_string(),
                detail: format!("no cage with id {s}"),
            }),
        Some(other) => Err(OpError::BadArg {
            op: op.to_string(),
            arg: "cage".to_string(),
            detail: format!("expected a cage id as a hex string, got {other:?}"),
        }),
        None => {
            let mut ids = doc.cages.keys();
            match (ids.next(), ids.next()) {
                (Some(&only), None) => Ok(only),
                (None, _) => Err(OpError::Failed {
                    op: op.to_string(),
                    detail: "the document has no cages".to_string(),
                }),
                _ => Err(OpError::MissingArg {
                    op: op.to_string(),
                    arg: "cage".to_string(),
                }),
            }
        }
    }
}

pub(crate) fn register_builtins(registry: &mut OpRegistry) {
    crate::cage_edits::register(registry);
    registry.register("cage.translate_vertices", |args, doc| {
        const OP: &str = "cage.translate_vertices";
        let cage_id = resolve_cage(OP, args, doc)?;
        let ids = args.require(OP, "vertices", Value::as_int_list, "a list of vertex indices")?;
        let delta = args.require(OP, "delta", Value::as_vec3, "a [x, y, z] vector")?;

        let cage = doc.cage(cage_id).ok_or_else(|| OpError::Failed {
            op: OP.to_string(),
            detail: format!("cage {cage_id} vanished"),
        })?;
        let vertex_count = cage.vertex_count();

        let mut vertices = Vec::with_capacity(ids.len());
        for id in ids {
            if id < 0 || id as usize >= vertex_count {
                return Err(OpError::BadArg {
                    op: OP.to_string(),
                    arg: "vertices".to_string(),
                    detail: format!("index {id} is out of range (cage has {vertex_count})"),
                });
            }
            vertices.push(VertexId::from_index(id as usize));
        }

        Ok(Box::new(TranslateVertices::new(
            cage_id,
            vertices,
            Vector3::new(delta[0], delta[1], delta[2]),
        )))
    });

    registry.register("cage.refine", |args, doc| {
        const OP: &str = "cage.refine";
        let cage_id = resolve_cage(OP, args, doc)?;
        let cage = doc.cage(cage_id).ok_or_else(|| missing_cage(OP, cage_id))?;
        // Refused here as well as in `execute`, so a console line says so before anything is
        // put on the undo stack, and `execute` says so again if the cage grew in between.
        check_refinable(OP, cage)?;
        Ok(Box::new(RefineCage::new(cage_id)))
    });
}

/// Moves a set of cage vertices by a fixed offset.
///
/// Records each vertex's position at execute time rather than deriving the inverse from
/// `delta`. Subtracting `delta` to undo would look equivalent but accumulates floating-point
/// error over repeated undo/redo cycles, drifting the geometry; storing the exact prior
/// positions restores them bit-for-bit.
pub(crate) struct TranslateVertices {
    cage: CageId,
    vertices: Vec<VertexId>,
    delta: Vector3<f64>,
    previous: Vec<Point3<f64>>,
}

impl TranslateVertices {
    pub fn new(cage: CageId, vertices: Vec<VertexId>, delta: Vector3<f64>) -> Self {
        Self {
            cage,
            vertices,
            delta,
            previous: Vec::new(),
        }
    }
}

/// Shared failure for an operation whose cage is gone — possible if a script holds an id
/// across a document change, so it is an error rather than a panic.
pub(crate) fn missing_cage(op: &str, cage: CageId) -> OpError {
    OpError::Failed {
        op: op.to_string(),
        detail: format!("cage {cage} is no longer in the document"),
    }
}

impl Command for TranslateVertices {
    fn name(&self) -> &str {
        "cage.translate_vertices"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        let delta = self.delta;
        let cage = doc
            .cage_mesh_mut(self.cage)
            .ok_or_else(|| missing_cage("cage.translate_vertices", self.cage))?;

        self.previous = self
            .vertices
            .iter()
            .map(|&v| cage.vertex_position(v))
            .collect();
        for &v in &self.vertices {
            let p = cage.vertex_position(v);
            cage.set_vertex_position(v, p + delta);
        }
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        let cage = doc
            .cage_mesh_mut(self.cage)
            .ok_or_else(|| missing_cage("cage.translate_vertices", self.cage))?;
        for (&v, &p) in self.vertices.iter().zip(&self.previous) {
            cage.set_vertex_position(v, p);
        }
        Ok(())
    }
}

/// Replaces a set of vertex positions outright.
///
/// This is what an interactive drag produces: the app mutates the cage live for feedback,
/// then on release records one command describing the whole gesture. Storing absolute
/// before/after positions (rather than a delta) keeps that exact, however the drag was
/// applied frame to frame.
pub struct SetVertexPositions {
    cage: CageId,
    vertices: Vec<VertexId>,
    before: Vec<Point3<f64>>,
    after: Vec<Point3<f64>>,
}

impl SetVertexPositions {
    pub fn new(
        cage: CageId,
        vertices: Vec<VertexId>,
        before: Vec<Point3<f64>>,
        after: Vec<Point3<f64>>,
    ) -> Self {
        Self {
            cage,
            vertices,
            before,
            after,
        }
    }

    fn apply(&self, doc: &mut Document, positions: &[Point3<f64>]) -> Result<(), OpError> {
        let cage = doc
            .cage_mesh_mut(self.cage)
            .ok_or_else(|| missing_cage("cage.set_vertex_positions", self.cage))?;
        for (&v, &p) in self.vertices.iter().zip(positions) {
            cage.set_vertex_position(v, p);
        }
        Ok(())
    }
}

impl Command for SetVertexPositions {
    fn name(&self) -> &str {
        "cage.set_vertex_positions"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone())
    }
}

/// Moves a surface's points to given absolute positions — [`SetVertexPositions`] for a surface
/// a conformal lattice follows.
///
/// **Its own command rather than a cage command given a second kind of id.** A surface is edited
/// exactly as a cage is — points, edges and faces dragged, the gesture committed once — and it is
/// the same kind of edit: below the line, reversing against the exact prior positions. But it is
/// found in a different table of the document, and a command that could name either would be one
/// more place deciding which, where two small ones each have nothing to decide.
pub struct SetSurfacePositions {
    surface: fieldform_scene::QuadMeshId,
    vertices: Vec<VertexId>,
    before: Vec<Point3<f64>>,
    after: Vec<Point3<f64>>,
}

impl SetSurfacePositions {
    pub fn new(
        surface: fieldform_scene::QuadMeshId,
        vertices: Vec<VertexId>,
        before: Vec<Point3<f64>>,
        after: Vec<Point3<f64>>,
    ) -> Self {
        Self { surface, vertices, before, after }
    }

    fn apply(&self, doc: &mut Document, positions: &[Point3<f64>]) -> Result<(), OpError> {
        let mesh = doc.quad_meshes.get_mut(&self.surface).ok_or_else(|| OpError::Failed {
            op: "surface.set_vertex_positions".to_string(),
            detail: format!("surface {} is no longer in the document", self.surface),
        })?;
        for (&v, &p) in self.vertices.iter().zip(positions) {
            mesh.set_vertex_position(v, p);
        }
        Ok(())
    }
}

impl Command for SetSurfacePositions {
    fn name(&self) -> &str {
        "surface.set_vertex_positions"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone())
    }
}

/// Moves a curve's control points to given absolute positions.
///
/// The curve twin of [`SetVertexPositions`], and deliberately the same shape. A curve is a
/// parametric-layer object — it is serialized, and `Node::Pipe` reads it — but its control
/// points are internal state the way a cage's vertex array is, so moving one is a
/// direct-layer edit that reverses against the exact prior positions. The alternative would
/// be a named `Parameter` per control point, which is per-element identity by another name:
/// the thing D1 declined to take on, reintroduced one level up.
pub struct SetControlPoints {
    curve: CurveId,
    points: Vec<usize>,
    before: Vec<Point3<f64>>,
    after: Vec<Point3<f64>>,
}

impl SetControlPoints {
    pub fn new(
        curve: CurveId,
        points: Vec<usize>,
        before: Vec<Point3<f64>>,
        after: Vec<Point3<f64>>,
    ) -> Self {
        Self {
            curve,
            points,
            before,
            after,
        }
    }

    fn apply(&self, doc: &mut Document, positions: &[Point3<f64>]) -> Result<(), OpError> {
        let curve = doc
            .curve_mut(self.curve)
            .ok_or_else(|| OpError::Failed {
                op: "curve.set_control_points".to_string(),
                detail: format!("the document has no curve {:?}", self.curve),
            })?;
        for (&i, &p) in self.points.iter().zip(positions) {
            curve.set_control_point(i, p);
        }
        Ok(())
    }
}

impl Command for SetControlPoints {
    fn name(&self) -> &str {
        "curve.set_control_points"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone())
    }
}

/// Which name a [`SetName`] sets. Three stores, because a name lives wherever the thing it names
/// does: a block's beside its other metadata, a curve's on the curve, a profile row's in a map
/// keyed by its first curve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Named {
    Block(fieldform_scene::NodeId),
    Curve(CurveId),
    /// A profile row, by the curve it is keyed on.
    Profile(CurveId),
}

/// Names a block, a curve or a profile row — **and puts it on the undo stack**, which renaming
/// did not used to be.
///
/// A name is an address: a reference field resolves typed text to one block, a blend's operand
/// list shows it, and an expression refers to a parameter by it. Something that can be referred
/// to and cannot be undone is the one kind of edit where Ctrl+Z silently does the wrong thing —
/// it takes back the edit *before* the rename and leaves the new name standing on it.
///
/// One command rather than three, because the three are one act with three stores. An empty name
/// clears, which is how a name typed and then deleted leaves the thing described by what it is;
/// a curve falls back to "Curve", since an unnamed curve still has to be readable in a list.
///
/// The clash check stays with the caller: whether a name may be taken is a question about the
/// document *and* about what the person meant, and the answer has to reach them on the keystroke
/// rather than as a failed command.
pub struct SetName {
    what: Named,
    after: Option<String>,
    before: Before,
}

/// What a [`SetName`] found before it wrote, which is what its undo writes back.
#[derive(Clone)]
enum Before {
    /// A block's whole metadata entry, absence included — not only its name.
    ///
    /// The name shares the entry with the eye's flags, and writing a remembered name back into a
    /// fresh entry is how undoing the bright V lost `shown`: the clear had dropped the entry, and
    /// the undo rebuilt it with the name alone. Holding the entry makes undo exact by
    /// construction rather than by `NodeMeta::is_empty` being right about every field. The eye
    /// writes either flag on the stack ([`SetVisible`]), and so does the clear a graph edit makes
    /// of a stale `shown` ([`SetShown`]); a linear stack undoes those before this, so they never
    /// cross. What is still written off it is the clear of `hidden` an operation makes on the block
    /// it consumes, and that goes back with the entry if it lands between a rename and its undo.
    Entry(fieldform_scene::NodeId, Option<fieldform_scene::NodeMeta>),
    /// A curve's or a profile row's name, which is all that store holds for it.
    Name(Option<String>),
}

impl SetName {
    pub fn new(what: Named, after: &str) -> Self {
        let after = after.trim();
        Self {
            what,
            after: (!after.is_empty()).then(|| after.to_string()),
            before: Before::Name(None),
        }
    }

    /// What the document holds now, so undo can put it back.
    fn current(&self, doc: &Document) -> Before {
        match &self.what {
            Named::Block(id) => Before::Entry(*id, doc.node_meta.get(id).cloned()),
            Named::Curve(id) => Before::Name(doc.curves.get(id).map(|c| c.name.clone())),
            Named::Profile(key) => Before::Name(doc.profile_names.get(key).cloned()),
        }
    }

    fn write(&self, doc: &mut Document, name: Option<String>) -> Result<(), OpError> {
        write_name(doc, &self.what, name, "edit.set_name")
    }
}

/// Writes one name into whichever store holds it — [`SetName`]'s writer, which [`SetOwnership`]
/// writes its name through too, so a name means the same thing whichever command set it. `op` is
/// the command a refusal is reported as.
fn write_name(
    doc: &mut Document,
    what: &Named,
    name: Option<String>,
    op: &str,
) -> Result<(), OpError> {
    match what {
        Named::Block(id) => {
            let slot = doc.node_meta.entry(*id).or_default();
            slot.name = name;
            // An entry holding nothing is no entry: it would otherwise match every unnamed
            // block in a search for a name. One that still holds a flag is kept — dropping
            // it is what made the bright V stop drawing a shown variable it put back.
            if slot.is_empty() {
                doc.node_meta.remove(id);
            }
        }
        Named::Curve(id) => {
            let Some(curve) = doc.curves.get_mut(id) else {
                return Err(OpError::Failed { op: op.into(), detail: "the curve is gone".into() });
            };
            // A curve with no name at all leaves its row blank and the Ends lists
            // unreadable, so an empty one falls back rather than being stored.
            curve.name = name.unwrap_or_else(|| "Curve".to_string());
        }
        Named::Profile(key) => match name {
            Some(name) => {
                doc.profile_names.insert(*key, name);
            }
            None => {
                doc.profile_names.remove(key);
            }
        },
    }
    Ok(())
}

impl Command for SetName {
    fn name(&self) -> &str {
        "edit.set_name"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.before = self.current(doc);
        self.write(doc, self.after.clone())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        match self.before.clone() {
            Before::Entry(id, Some(entry)) => {
                doc.node_meta.insert(id, entry);
            }
            Before::Entry(id, None) => {
                doc.node_meta.remove(&id);
            }
            Before::Name(name) => return self.write(doc, name),
        }
        Ok(())
    }
}

/// Sets whether a block something consumes is drawn as well (`NodeMeta::shown`) -- **that flag
/// and nothing else in the entry**.
///
/// On the history so a graph edit can take the flag with it in one entry. A block let go of by
/// the edit is a body again and draws as one, so a flag left standing outlives the reason it was
/// set: the next block to consume it draws it beside itself as though asked to (refactor plan
/// §3.3, root cause 9). Clearing it in the edit's own entry means undoing the edit asks for it
/// again.
///
/// The flag alone rather than the whole entry, unlike [`SetName`]'s undo: `hidden` shares the
/// entry and is written off the history by the operations that consume a block, so restoring a
/// copy of the entry would take back edits this command never made. `shown` is written by the eye
/// as well, through [`SetVisible`], and goes back to what this found either way. The name would
/// come to no harm either way: only `SetName` and `SetOwnership` write it, both on the history,
/// and a rename made after this is undone before it.
pub struct SetShown {
    id: fieldform_scene::NodeId,
    after: bool,
    before: bool,
}

impl SetShown {
    pub fn new(id: fieldform_scene::NodeId, shown: bool) -> Self {
        Self { id, after: shown, before: shown }
    }

    fn write(&self, doc: &mut Document, shown: bool) {
        let slot = doc.node_meta.entry(self.id).or_default();
        slot.shown = shown;
        // An entry holding nothing is no entry, for the reason `SetName` drops one.
        if slot.is_empty() {
            doc.node_meta.remove(&self.id);
        }
    }
}

impl Command for SetShown {
    fn name(&self) -> &str {
        "block.set_shown"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.before = doc.node_meta.get(&self.id).is_some_and(|m| m.shown);
        self.write(doc, self.after);
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.write(doc, self.before);
        Ok(())
    }
}

/// Turns a block's eye on or off — **the one write every eye makes**, and an undo step, which the
/// eye was not (`block-ownership-spec.md` §7.3, §8).
///
/// The flag is handed in rather than worked out here: which of the two the eye means depends on
/// where the block sits in its part's graph (`EyeFlag` says how), and that is asked by the app,
/// once, at the moment of the click. `before` and `after` are whether the block draws as that flag
/// holds it. Only that flag is written, for [`SetShown`]'s reason: the other is written off the
/// history by the operations that consume a block, so restoring a copy of the entry would take
/// back edits this command never made. Nothing here moves a block, and nothing settles.
pub struct SetVisible {
    node: fieldform_scene::NodeId,
    flag: EyeFlag,
    before: bool,
    after: bool,
}

impl SetVisible {
    pub fn new(node: fieldform_scene::NodeId, flag: EyeFlag, visible: bool) -> Self {
        Self { node, flag, before: visible, after: visible }
    }

    fn write(&self, doc: &mut Document, visible: bool) {
        let slot = doc.node_meta.entry(self.node).or_default();
        slot.set_visible_by(self.flag, visible);
        // An entry holding nothing is no entry, for the reason `SetName` drops one: undoing an eye
        // on a block that had none leaves none.
        if slot.is_empty() {
            doc.node_meta.remove(&self.node);
        }
    }
}

impl Command for SetVisible {
    fn name(&self) -> &str {
        "block.set_visible"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.before = doc.visible_by(self.node, self.flag);
        self.write(doc, self.after);
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.write(doc, self.before);
        Ok(())
    }
}

/// Puts a curve into the document, takes one out, or swaps one for another.
///
/// All three, because they are one edit with the ends left open: `None` before and `Some` after
/// is an insert, the reverse is a delete, and two `Some`s is a replacement. Undo is then just
/// applying the other end, and none of the three needs to know which of the others it is.
///
/// The replacement case is the one the finer commands cannot express. [`SetControlPoints`] and
/// [`SetControlWeights`] both assume the points they name still exist afterwards, which raising
/// a curve's degree makes untrue — it changes how many there are. Holding the whole curve at
/// both ends is what lets undo land on the exact one that was there rather than on a curve of
/// the right degree and an approximated shape.
pub struct SetCurve {
    curve: CurveId,
    before: Option<Curve>,
    after: Option<Curve>,
}

impl SetCurve {
    pub fn new(curve: CurveId, before: Option<Curve>, after: Option<Curve>) -> Self {
        Self { curve, before, after }
    }

    /// Puts a curve in that was not there.
    pub fn insert(curve: Curve) -> Self {
        Self::new(curve.id, None, Some(curve))
    }

    /// Takes one out.
    pub fn remove(curve: Curve) -> Self {
        Self::new(curve.id, Some(curve), None)
    }

    fn apply(&self, doc: &mut Document, state: &Option<Curve>) {
        match state {
            Some(curve) => {
                doc.curves.insert(self.curve, curve.clone());
            }
            None => {
                doc.curves.remove(&self.curve);
            }
        }
    }
}

impl Command for SetCurve {
    fn name(&self) -> &str {
        "curve.set"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone());
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone());
        Ok(())
    }
}

/// Puts a parameter into the document, takes one out, or swaps one for another — [`SetCurve`]'s
/// three cases in one command, for its reason: undo is applying the other end.
///
/// **A parameter is what a `ParamRef::Named` points at**, so making and removing one belong on the
/// history the graph edits pointing at it are on. Written straight into the document they were
/// not, and undo took back one half of an act: undoing the × that put a named radius back to a
/// number restored the reference and not the parameter, so the block read a value that was no
/// longer there (refactor plan §3.3, root cause 7). The whole parameter at both ends, its name and
/// any expression included, so what undo puts back is what was taken rather than a number under
/// the same id.
///
/// **Each end is the parameter as the document last held it there, and the document as it is
/// decides the rest.** A parameter's name, number and expression are edited off the history, so
/// between this entry being made and an undo or a redo reaching it the parameter can have been
/// renamed, given another number, or come to be read by an expression typed since. Replaying the
/// copies the entry was built with undid those edits as a side effect, and taking out a parameter
/// something had come to read left that reader dangling -- the failure this command is here to
/// end. So:
///
/// - leaving an end that holds a parameter refreshes it from the document, so what comes back is
///   what went;
/// - a parameter is taken out only while nothing reads it (`Document::parameter_is_read`). One
///   that is read stays, so the worst an undo leaves is a parameter nothing reads, and the same
///   entry takes it on a later pass once nothing does;
/// - putting one in where the document already holds it keeps the one held, which is newer than
///   anything here;
/// - one put in goes through `Document::put_parameter`, which gives it a free name if its own was
///   taken meanwhile and re-renders any expression a rename left saying an old name.
///
/// That departs from "`undo` restores exactly the state `execute` found" (`Command`), on purpose:
/// exactness is owed for the history's own edits, and these are edits it never saw.
pub struct SetParameter {
    id: ParamId,
    before: Option<Parameter>,
    after: Option<Parameter>,
}

impl SetParameter {
    pub fn new(id: ParamId, before: Option<Parameter>, after: Option<Parameter>) -> Self {
        Self { id, before, after }
    }

    /// Puts a parameter in that was not there.
    pub fn insert(parameter: Parameter) -> Self {
        Self::new(parameter.id, None, Some(parameter))
    }

    /// Takes one out.
    pub fn remove(parameter: Parameter) -> Self {
        Self::new(parameter.id, Some(parameter), None)
    }

    /// Moves parameter `id`'s slot from end `from` to end `to`, refreshing `from` with what the
    /// slot holds as it leaves.
    fn step(id: ParamId, doc: &mut Document, from: &mut Option<Parameter>, to: &Option<Parameter>) {
        let now = doc.parameters.get(&id).cloned();
        if let (Some(held), Some(now)) = (from.as_mut(), &now) {
            *held = now.clone();
        }
        match to {
            None => {
                if now.is_some() && !doc.parameter_is_read(id, None) {
                    doc.parameters.remove(&id);
                }
            }
            // Put in where it was absent, and already there: a removal left it for a reader, or
            // the field that made it wrote it before this entry did. Either way it is newer.
            Some(_) if from.is_none() && now.is_some() => {}
            Some(parameter) => doc.put_parameter(parameter.clone()),
        }
    }
}

impl Command for SetParameter {
    fn name(&self) -> &str {
        "parameter.set"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        Self::step(self.id, doc, &mut self.before, &self.after);
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        Self::step(self.id, doc, &mut self.after, &self.before);
        Ok(())
    }
}

/// A relation between curves made, broken, marked or re-aligned (`profile-group-spec.md`
/// §5). The record alone: the points a relation drives are ordinary control points, moved by
/// their own commands or by the re-derivation that follows any edit, so breaking one is a
/// pure metadata edit and moves nothing.
pub struct SetRelation {
    id: fieldform_scene::id::RelationId,
    before: Option<fieldform_scene::RelationRecord>,
    after: Option<fieldform_scene::RelationRecord>,
}

impl SetRelation {
    pub fn new(
        id: fieldform_scene::id::RelationId,
        before: Option<fieldform_scene::RelationRecord>,
        after: Option<fieldform_scene::RelationRecord>,
    ) -> Self {
        Self { id, before, after }
    }

    pub fn insert(id: fieldform_scene::id::RelationId, record: fieldform_scene::RelationRecord) -> Self {
        Self::new(id, None, Some(record))
    }

    pub fn remove(id: fieldform_scene::id::RelationId, record: fieldform_scene::RelationRecord) -> Self {
        Self::new(id, Some(record), None)
    }

    fn apply(&self, doc: &mut Document, state: &Option<fieldform_scene::RelationRecord>) {
        match state {
            Some(record) => {
                doc.relations.insert(self.id, record.clone());
            }
            None => {
                doc.relations.remove(&self.id);
            }
        }
    }
}

impl Command for SetRelation {
    fn name(&self) -> &str {
        "relation.set"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone());
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone());
        Ok(())
    }
}

/// An axis stood on a body's surface made, edited or removed (`Document::surface_axes`).
///
/// The record alone, as [`SetRelation`] is: the two points it derives are ordinary control points,
/// set by their own commands or by the re-derivation that follows any change, so reversing an edit of
/// the U, V or length is a pure metadata edit and the points catch up on the next derivation — which
/// is the same path a moved body takes, so there is one way an axis comes to be where it is.
pub struct SetSurfaceAxis {
    curve: CurveId,
    before: Option<fieldform_scene::SurfaceAxis>,
    after: Option<fieldform_scene::SurfaceAxis>,
}

impl SetSurfaceAxis {
    pub fn new(
        curve: CurveId,
        before: Option<fieldform_scene::SurfaceAxis>,
        after: Option<fieldform_scene::SurfaceAxis>,
    ) -> Self {
        Self { curve, before, after }
    }

    pub fn insert(curve: CurveId, axis: fieldform_scene::SurfaceAxis) -> Self {
        Self::new(curve, None, Some(axis))
    }

    fn apply(&self, doc: &mut Document, state: &Option<fieldform_scene::SurfaceAxis>) {
        match state {
            Some(axis) => {
                doc.surface_axes.insert(self.curve, *axis);
            }
            None => {
                doc.surface_axes.remove(&self.curve);
            }
        }
    }
}

impl Command for SetSurfaceAxis {
    fn name(&self) -> &str {
        "surface_axis.set"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after);
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before);
        Ok(())
    }
}

/// Sets the weights of a curve's control points.
///
/// The twin of [`SetControlPoints`], and the same shape for the same reason: a weight is the
/// other half of what a control point is, so moving one and pulling one harder should reverse
/// the same way.
pub struct SetControlWeights {
    curve: CurveId,
    points: Vec<usize>,
    before: Vec<f64>,
    after: Vec<f64>,
}

impl SetControlWeights {
    pub fn new(curve: CurveId, points: Vec<usize>, before: Vec<f64>, after: Vec<f64>) -> Self {
        Self { curve, points, before, after }
    }

    fn apply(&self, doc: &mut Document, weights: &[f64]) -> Result<(), OpError> {
        let curve = doc.curve_mut(self.curve).ok_or_else(|| OpError::Failed {
            op: "curve.set_weights".to_string(),
            detail: format!("the document has no curve {:?}", self.curve),
        })?;
        for (&i, &w) in self.points.iter().zip(weights) {
            curve.set_weight(i, w);
        }
        Ok(())
    }
}

impl Command for SetControlWeights {
    fn name(&self) -> &str {
        "curve.set_weights"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone())
    }
}

/// Replaces a part's whole composition graph.
///
/// Coarser than the cage and curve commands above, and deliberately so. A gizmo drag on a body
/// does not always edit one node: placing a body that has never been placed *adds* a node and
/// reparents the root, where placing one already placed only changes three numbers. A
/// before/after pair of whole graphs covers both without the command having to know which
/// happened, and it cannot describe a half-applied edit the way a node-level command could if
/// the two cases were ever confused.
///
/// The cost is holding two copies of the graph per undo entry. A graph is nodes and parameter
/// references — no geometry, which lives in the cages and curves the nodes name — so it is
/// small next to anything else on the stack.
///
/// **It settles who owns what, and its undo puts that back rather than settling again**
/// (`block-ownership-spec.md` §4.1, §4.3). Every graph edit is one of these, so a settle here is
/// a settle after every edit to who reads a block, and inside a [`Batch`] it happens at this
/// point *in* the batch, so a later member acts on the settled state — which is why it is not in
/// [`crate::CommandStack`]. Undo writes back the graph and both ownership maps: the maps a gesture
/// handed it, or else those that stood before its first execute. Settling on undo instead would
/// get a block a second reader made a variable wrong, since a settle never changes a variable; a
/// redo settles again, from the state an undo returns to, and reaches the answer it reached the
/// first time.
///
/// **It settles from those same maps, and reports only what moved from the ones it found**
/// (spec §4.3, §6.3). The two differ only for a gesture's later instants, which `replace_top`
/// executes on a document still holding the first instant's settle. Settled from that, a block the
/// first instant gave a second reader stayed a variable after the next instant took the reader
/// away: a curve is not an input, so the console's `sweep profile= spine=` re-pointed its sweep
/// inside the gesture that built it, and a curve a pipe held was a variable for good (refactor
/// plan 2.22). Settled from the gesture's maps instead, every later instant makes again each move
/// the first one made, and the application, which hears of moves at the top of every frame, said
/// each of them again on every frame of the drag. So a move the document already held is not
/// reported again.
pub struct SetPartGraph {
    part: PartId,
    before: Graph,
    after: Graph,
    /// Both ownership maps as they were with `before`: handed over by a gesture, or taken on the
    /// first execute. `None` only until then.
    ownership: Option<OwnershipSnapshot>,
}

impl SetPartGraph {
    pub fn new(part: PartId, before: Graph, after: Graph) -> Self {
        Self::with_ownership(part, before, after, None)
    }

    /// [`Self::new`], with the ownership maps an undo puts back given rather than taken.
    ///
    /// **For a gesture** (spec §4.3). A drag replaces its own entry every frame with one whose
    /// `before` is the graph the gesture started from, and that can differ in structure from the
    /// graph the replacement executes on: add a blend, then drag its radius. Maps taken at the
    /// replacement would hold the blend's operands inside a blend the restored graph does not
    /// contain. So the gesture takes the maps where it takes `before`, and hands a clone of them to
    /// every entry it builds — which shares them, so no entry copies them. Each replacement settles
    /// from them, not from the maps it finds. `None` is [`Self::new`]: taken on the first execute.
    pub fn with_ownership(
        part: PartId,
        before: Graph,
        after: Graph,
        ownership: Option<OwnershipSnapshot>,
    ) -> Self {
        Self { part, before, after, ownership }
    }

    fn apply(&self, doc: &mut Document, graph: &Graph) -> Result<(), OpError> {
        let part = doc.parts.get_mut(&self.part).ok_or_else(|| OpError::Failed {
            op: "part.set_graph".to_string(),
            detail: format!("the document has no part {:?}", self.part),
        })?;
        part.graph = graph.clone();
        Ok(())
    }
}

impl Command for SetPartGraph {
    fn name(&self) -> &str {
        "part.set_graph"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone())?;
        // Taken only the first time: a redo finds the document as this found it, and a gesture's
        // maps are the ones that go with its `before`, which a later execute could not take. Taken
        // here, they are the document's own, and there is nothing to put in their place.
        let found = match &self.ownership {
            None => {
                self.ownership = Some(OwnershipSnapshot::of(doc));
                None
            }
            Some(given) => Some(given.exchange(doc)),
        };
        let heard = doc.pending_ownership_changes.len();
        doc.settle_ownership();
        if let Some(found) = found {
            // Kept or dropped by where the block stood when this began, which is what the
            // application last heard. Only a later instant of a gesture drops anything: anywhere
            // else these are the maps the settle started from.
            let settled = doc.pending_ownership_changes.split_off(heard);
            let moved: Vec<_> = settled
                .into_iter()
                .filter(|change| found.ownership(change.block()) != doc.ownership(change.block()))
                .collect();
            doc.pending_ownership_changes.extend(moved);
        }
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone())?;
        if let Some(ownership) = &self.ownership {
            ownership.restore(doc);
        }
        Ok(())
    }
}

/// A document's two ownership maps at one moment — what an undo writes back.
///
/// **Both maps whole, not the blocks an edit moved**, because a settle moves whatever its readers
/// say, and only the maps from before can say where each block was: including that a block had no
/// entry yet, which no [`Ownership`] can say. They are a few bytes a block, copied once an entry.
///
/// Its own type so a gesture can carry one from its first instant to its last without naming what
/// is in it (spec §4.3). **A clone shares the maps rather than copying them**, because a gesture
/// clones its snapshot into every entry it builds: each entry must own one for its undo, and
/// `replace_top` drops the entry before, so nothing can be moved out of it. Shared, building an
/// entry copies nothing; the one copy a drag makes each frame is [`Self::exchange`]'s, into the
/// document, before the settle it is for.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnershipSnapshot {
    nodes: Arc<HashMap<NodeId, Ownership>>,
    curves: Arc<HashMap<CurveId, Ownership>>,
}

impl OwnershipSnapshot {
    /// Both maps as `doc` holds them now.
    pub fn of(doc: &Document) -> Self {
        Self {
            nodes: Arc::new(doc.ownership_nodes.clone()),
            curves: Arc::new(doc.ownership_curves.clone()),
        }
    }

    /// Puts both maps back, entry for entry. Copied rather than moved, since an entry can be undone
    /// again after a redo.
    fn restore(&self, doc: &mut Document) {
        doc.ownership_nodes = HashMap::clone(&self.nodes);
        doc.ownership_curves = HashMap::clone(&self.curves);
    }

    /// Puts both maps in the document's place, as [`Self::restore`] does, and hands back the ones
    /// it held: moved out rather than copied, since nothing else will keep them.
    fn exchange(&self, doc: &mut Document) -> Self {
        let nodes = std::mem::replace(&mut doc.ownership_nodes, HashMap::clone(&self.nodes));
        let curves = std::mem::replace(&mut doc.ownership_curves, HashMap::clone(&self.curves));
        Self { nodes: Arc::new(nodes), curves: Arc::new(curves) }
    }

    /// Where `block` sat in these maps, one with no entry read as free, as
    /// [`Document::ownership`] reads it.
    fn ownership(&self, block: BlockKey) -> Ownership {
        let stored = match block {
            BlockKey::Node(id) => self.nodes.get(&id),
            BlockKey::Curve(id) => self.curves.get(&id),
        };
        stored.copied().unwrap_or(Ownership::Free)
    }

    /// Whether `other` holds these very maps rather than a copy of them.
    #[cfg(test)]
    pub(crate) fn shares_maps_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.nodes, &other.nodes) && Arc::ptr_eq(&self.curves, &other.curves)
    }
}

/// Where one block sits in the list, and its name: make variable, put back inside, and rename
/// (`block-ownership-spec.md` §5, §8).
///
/// **The name goes with the ownership** because the acts need both at once: a block made a
/// variable needs a name to be typed by (I4), one put back keeps its own, and a rename leaves it
/// where it is. One entry, so one Ctrl+Z takes back both.
///
/// **It settles after writing**, as [`SetPartGraph`] does and for the same reason: told to sit
/// inside a block that does not read it, a block goes where its readers put it rather than staying
/// where it was told. And its undo puts back both maps whole rather than this block's entry, so
/// whatever that settle moved goes back with it. It does not settle on undo.
///
/// The spec's `before` is the block's ownership and name. This keeps the name and both maps, which
/// hold the block's ownership too, absence included, and takes them on every execute as [`SetName`]
/// takes its own. The name alone goes back, not the whole metadata entry `SetName` restores: the
/// eye's flags share that entry and are written off the history, and an ownership change that left
/// the name as it was would otherwise take back an eye toggled since.
///
/// Refusals stay with the caller, as `SetName`'s clash check does: whether a block can be put back
/// is [`Document::put_back_target`]'s answer, and clearing a variable's name is refused where the
/// name is typed (§5.3).
pub struct SetOwnership {
    block: BlockKey,
    after: (Ownership, Option<String>),
    before: Option<(Option<String>, OwnershipSnapshot)>,
}

impl SetOwnership {
    /// Puts `block` at `ownership` and names it `name`; a blank name clears it, as [`SetName`]'s
    /// does, and a curve's falls back to "Curve".
    pub fn new(block: BlockKey, ownership: Ownership, name: Option<&str>) -> Self {
        let name = name.map(str::trim).filter(|n| !n.is_empty()).map(str::to_string);
        Self { block, after: (ownership, name), before: None }
    }

    /// Where the block's name lives: [`SetName`]'s stores, a block's and a curve's.
    fn named(&self) -> Named {
        match self.block {
            BlockKey::Node(id) => Named::Block(id),
            BlockKey::Curve(id) => Named::Curve(id),
        }
    }
}

impl Command for SetOwnership {
    fn name(&self) -> &str {
        "block.set_ownership"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        let name = match self.block {
            BlockKey::Node(id) if doc.parts.values().any(|p| p.graph.node(id).is_some()) => {
                doc.node_meta.get(&id).and_then(|m| m.name.clone())
            }
            BlockKey::Curve(id) if doc.curves.contains_key(&id) => {
                Some(doc.curves[&id].name.clone())
            }
            // An entry for a block that is gone would only wait for the save to prune it.
            _ => {
                return Err(OpError::Failed {
                    op: "block.set_ownership".into(),
                    detail: "the block is gone".into(),
                });
            }
        };
        self.before = Some((name, OwnershipSnapshot::of(doc)));
        write_name(doc, &self.named(), self.after.1.clone(), "block.set_ownership")?;
        match self.block {
            BlockKey::Node(id) => doc.ownership_nodes.insert(id, self.after.0),
            BlockKey::Curve(id) => doc.ownership_curves.insert(id, self.after.0),
        };
        doc.settle_ownership();
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        let (name, ownership) = self.before.clone().ok_or_else(|| OpError::Failed {
            op: "block.set_ownership".into(),
            detail: "there is no earlier ownership to put back".into(),
        })?;
        write_name(doc, &self.named(), name, "block.set_ownership")?;
        ownership.restore(doc);
        Ok(())
    }
}

/// Sets a part's material, or clears it back to the default with `None`.
///
/// Whole-value before and after, the same shape as [`SetPartGraph`] and for the same reason:
/// picking a preset changes every channel at once and dragging a slider changes one, and a
/// command that recorded a channel at a time would have to know which happened. A material is
/// eight numbers, so two copies cost nothing.
pub struct SetPartMaterial {
    part: PartId,
    before: Option<Material>,
    after: Option<Material>,
}

impl SetPartMaterial {
    pub fn new(part: PartId, before: Option<Material>, after: Option<Material>) -> Self {
        Self { part, before, after }
    }

    fn apply(&self, doc: &mut Document, material: &Option<Material>) -> Result<(), OpError> {
        let part = doc.parts.get_mut(&self.part).ok_or_else(|| OpError::Failed {
            op: "part.set_material".to_string(),
            detail: format!("the document has no part {:?}", self.part),
        })?;
        part.material = material.clone();
        Ok(())
    }
}

impl Command for SetPartMaterial {
    fn name(&self) -> &str {
        "part.set_material"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone())
    }
}

/// Sets every Tessellate and Export block at once — backlog item 53.
///
/// **Both lists whole, before and after**, for the reason [`SetPartGraph`] holds whole graphs: a
/// create, a delete, a rename and an edit to a detail are all "the blocks were this and are now
/// that", and one command that cannot tell them apart cannot get any of them half right. The
/// lists are a few dozen small records; holding two copies per undo entry costs nothing next to
/// a graph.
pub struct SetOutputs {
    before: (Vec<fieldform_scene::Tessellation>, Vec<fieldform_scene::Export>),
    after: (Vec<fieldform_scene::Tessellation>, Vec<fieldform_scene::Export>),
}

impl SetOutputs {
    pub fn new(
        before: (Vec<fieldform_scene::Tessellation>, Vec<fieldform_scene::Export>),
        after: (Vec<fieldform_scene::Tessellation>, Vec<fieldform_scene::Export>),
    ) -> Self {
        Self { before, after }
    }
}

impl Command for SetOutputs {
    fn name(&self) -> &str {
        "outputs.set"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        doc.tessellations = self.after.0.clone();
        doc.exports = self.after.1.clone();
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        doc.tessellations = self.before.0.clone();
        doc.exports = self.before.1.clone();
        Ok(())
    }
}

/// Sets or removes an import's record.
pub struct SetSource {
    id: fieldform_scene::id::SourceId,
    before: Option<fieldform_scene::ImportSource>,
    after: Option<fieldform_scene::ImportSource>,
}

impl SetSource {
    pub fn new(
        id: fieldform_scene::id::SourceId,
        before: Option<fieldform_scene::ImportSource>,
        after: Option<fieldform_scene::ImportSource>,
    ) -> Self {
        Self { id, before, after }
    }

    fn apply(&self, doc: &mut Document, state: &Option<fieldform_scene::ImportSource>) {
        match state {
            Some(record) => {
                doc.sources.insert(self.id, record.clone());
            }
            None => {
                doc.sources.remove(&self.id);
            }
        }
    }
}

impl Command for SetSource {
    fn name(&self) -> &str {
        "import.set"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone());
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone());
        Ok(())
    }
}

/// Sets or removes a surface to follow, with its name.
pub struct SetSurface {
    id: fieldform_scene::QuadMeshId,
    before: Option<(fieldform_cage::QuadMesh, Option<String>)>,
    after: Option<(fieldform_cage::QuadMesh, Option<String>)>,
}

impl SetSurface {
    pub fn new(
        id: fieldform_scene::QuadMeshId,
        before: Option<(fieldform_cage::QuadMesh, Option<String>)>,
        after: Option<(fieldform_cage::QuadMesh, Option<String>)>,
    ) -> Self {
        Self { id, before, after }
    }

    fn apply(&self, doc: &mut Document, state: &Option<(fieldform_cage::QuadMesh, Option<String>)>) {
        match state {
            Some((mesh, name)) => {
                doc.quad_meshes.insert(self.id, mesh.clone());
                match name {
                    Some(name) => {
                        doc.surface_names.insert(self.id, name.clone());
                    }
                    None => {
                        doc.surface_names.remove(&self.id);
                    }
                }
            }
            None => {
                doc.quad_meshes.remove(&self.id);
                doc.surface_names.remove(&self.id);
            }
        }
    }
}

impl Command for SetSurface {
    fn name(&self) -> &str {
        "surface.set"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.after.clone());
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        self.apply(doc, &self.before.clone());
        Ok(())
    }
}

/// Several commands as one entry in the history.
///
/// **One act, one undo.** Deleting an import takes its bodies out of the graph, its surfaces and
/// curves out of the document and its record out of the list — four kinds of change, and a
/// person who pressed Delete once expects Ctrl+Z once to put all of it back. Run in order and
/// undone in reverse, so each part undoes against the state it was applied to.
pub struct Batch {
    name: String,
    commands: Vec<Box<dyn Command>>,
}

impl Batch {
    pub fn new(name: impl Into<String>, commands: Vec<Box<dyn Command>>) -> Self {
        Self { name: name.into(), commands }
    }
}

impl Command for Batch {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        for i in 0..self.commands.len() {
            if let Err(e) = self.commands[i].execute(doc) {
                // Put back what already ran, so a failure leaves the document as it was found
                // rather than half deleted.
                for done in self.commands[..i].iter_mut().rev() {
                    let _ = done.undo(doc);
                }
                return Err(e);
            }
        }
        Ok(())
    }

    fn byte_size(&self) -> usize {
        self.commands.iter().map(|c| c.byte_size()).sum()
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        for cmd in self.commands.iter_mut().rev() {
            cmd.undo(doc)?;
        }
        Ok(())
    }
}

/// The most faces a refine may leave a cage with.
///
/// **A guard, not a tuned value.** A cube goes 6, 24, 96, 384, 1 536, 6 144, 24 576, and both
/// the cage field's evaluation and tessellation grow with the face count, so this is here so
/// that one reflexive extra click cannot wedge the application. Raise it on evidence rather
/// than on feel (`cage-refine-spec.md` §6) — a cube takes five refines and is stopped at the
/// sixth.
pub(crate) const REFINE_MAX_FACES: usize = 20_000;

/// `12345` as `12,345`, so the two numbers in a refusal can be told apart at a glance.
fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Refuses a refine that would leave more than [`REFINE_MAX_FACES`], with both numbers in the
/// message — the one that would result and the one that is allowed.
fn check_refinable(op: &str, cage: &fieldform_cage::Cage) -> Result<(), OpError> {
    // Every quad becomes four, exactly: the count is known before any of the work is done.
    let would_have = cage.face_count() * 4;
    if would_have > REFINE_MAX_FACES {
        return Err(OpError::Failed {
            op: op.to_string(),
            detail: format!(
                "refining this cage would give {} faces; the limit is {}",
                grouped(would_have),
                grouped(REFINE_MAX_FACES)
            ),
        });
    }
    Ok(())
}

/// One Catmull-Clark step applied to a whole cage: four times the faces, the same surface.
///
/// **Refinement leaves the limit surface where it was.** That is what it is, not a side
/// effect — `LimitSurface` already evaluates that surface exactly, so the model was as smooth
/// at six faces as it will ever be and this buys one thing only: more control points to move.
/// It is therefore called *refine*, and there is deliberately no linear variant and no flag
/// for one: the topology change is identical and the positions are not, and two operations
/// that look the same and produce different shapes is the silence `cage-refine-spec.md` §3
/// exists to prevent.
///
/// The cage is replaced under the **same id**, so every `Node::Cage` that names it still
/// resolves. Vertex and face ids do *not* survive — the original vertices keep their indices
/// but have moved under the vertex-point rule, and every face is new — which is why a caller
/// holding a selection has to drop it (§7). Undo puts the whole previous cage back rather than
/// deriving an inverse, as `TranslateVertices` does for positions: a cage is positions and
/// quads, about 100 KB at the guard's ceiling, and a restore that is bit-for-bit is worth far
/// more than the memory.
///
/// **Redo restores a snapshot too.** It used to run `subdivide` again, which was the one cage
/// command re-running a primitive; a redo that recomputes is a redo that can differ from what
/// was undone, and nothing would say so. The first execute subdivides once and keeps both
/// meshes; every execute after that is the restore.
pub(crate) struct RefineCage {
    cage: CageId,
    snapshots: Option<(fieldform_cage::MeshSnapshot, fieldform_cage::MeshSnapshot)>,
}

impl RefineCage {
    pub fn new(cage: CageId) -> Self {
        Self { cage, snapshots: None }
    }

    fn put(&self, doc: &mut Document, snapshot: &fieldform_cage::MeshSnapshot) -> Result<(), OpError> {
        let body = doc.cage_body_mut(self.cage).ok_or_else(|| missing_cage("cage.refine", self.cage))?;
        body.set_mesh(snapshot.restore());
        Ok(())
    }
}

impl Command for RefineCage {
    fn name(&self) -> &str {
        "cage.refine"
    }

    fn execute(&mut self, doc: &mut Document) -> Result<(), OpError> {
        if let Some((_, after)) = &self.snapshots {
            return self.put(doc, &after.clone());
        }
        let current = doc
            .cage(self.cage)
            .ok_or_else(|| missing_cage("cage.refine", self.cage))?;
        check_refinable("cage.refine", current)?;
        // A `CageError` here would mean a fault in the rules, which are tested in
        // `fieldform-subd` — so it is reported, not unwrapped, and nothing has changed.
        let refined = fieldform_subd::subdivide(current).map_err(|e| OpError::Failed {
            op: "cage.refine".to_string(),
            detail: e.to_string(),
        })?;
        let before = fieldform_cage::MeshSnapshot::of(current);
        let after = fieldform_cage::MeshSnapshot::of(&refined);
        doc.cages.insert(self.cage, refined.into());
        self.snapshots = Some((before, after));
        Ok(())
    }

    fn undo(&mut self, doc: &mut Document) -> Result<(), OpError> {
        let (before, _) = self.snapshots.as_ref().ok_or_else(|| OpError::Failed {
            op: "cage.refine".to_string(),
            detail: "there is no earlier cage to put back".to_string(),
        })?;
        self.put(doc, &before.clone())
    }

    fn byte_size(&self) -> usize {
        self.snapshots.as_ref().map_or(0, |(b, a)| b.byte_size() + a.byte_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommandStack;
    use fieldform_cage::Cage;

    /// A rename is an undo step, in all three stores a name lives in, and undoing one puts back
    /// exactly what was there -- including nothing, for a name that had none.
    #[test]
    fn a_rename_can_be_undone_wherever_the_name_lives() {
        let mut doc = Document::with_cage(Cage::cube());
        let part = *doc.parts.keys().next().expect("a part");
        let block = doc
            .parts
            .get_mut(&part)
            .expect("the part")
            .graph
            .add(fieldform_scene::Node::Sphere {
                centre: fieldform_scene::Place::origin(),
                radius: fieldform_scene::ParamRef::Literal(10.0),
            });
        let curve = fieldform_scene::Curve::polyline(
            CurveId::new(),
            "Line",
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(10.0, 0.0, 0.0),
            ],
            false,
        )
        .expect("a two-point line");
        let curve_id = curve.id;
        doc.curves.insert(curve_id, curve);

        let mut stack = CommandStack::new(64);
        // A block with no name at all, so undo has to restore an absence.
        stack
            .execute(&mut doc, Box::new(SetName::new(Named::Block(block), "hub")))
            .expect("names the block");
        assert_eq!(doc.node_meta.get(&block).and_then(|m| m.name.as_deref()), Some("hub"));
        stack.undo(&mut doc).expect("undoes");
        assert!(doc.node_meta.get(&block).and_then(|m| m.name.as_deref()).is_none(), "and leaves no entry");
        stack.redo(&mut doc).expect("redoes");
        assert_eq!(doc.node_meta.get(&block).and_then(|m| m.name.as_deref()), Some("hub"));

        // A curve always has one, so undo restores the old name rather than clearing it.
        stack
            .execute(&mut doc, Box::new(SetName::new(Named::Curve(curve_id), "spine")))
            .expect("names the curve");
        assert_eq!(doc.curves[&curve_id].name, "spine");
        stack.undo(&mut doc).expect("undoes");
        assert_eq!(doc.curves[&curve_id].name, "Line");
        // Cleared, it falls back rather than going blank.
        stack
            .execute(&mut doc, Box::new(SetName::new(Named::Curve(curve_id), "  ")))
            .expect("clears the curve's name");
        assert_eq!(doc.curves[&curve_id].name, "Curve");

        // A profile row's name is keyed by its first curve and is absent until given.
        stack
            .execute(&mut doc, Box::new(SetName::new(Named::Profile(curve_id), "Outline")))
            .expect("names the row");
        assert_eq!(doc.profile_names.get(&curve_id).map(String::as_str), Some("Outline"));
        stack.undo(&mut doc).expect("undoes");
        assert!(!doc.profile_names.contains_key(&curve_id));
    }

    /// **Clearing a shown block's name leaves it shown, and undo puts the whole entry back.**
    ///
    /// The bright V puts a variable back inside by clearing its name (`nest_back` →
    /// `set_body_name(id, "")`), and that is this command. The entry used to be dropped as soon as
    /// the name went, because `is_empty` did not count `shown` — so a variable on screen stopped
    /// drawing the moment it was put back, before any save. Undo then wrote the name into a fresh
    /// entry and the flag stayed lost (block-ownership-spec §1, §12 test 14).
    #[test]
    fn clearing_a_shown_blocks_name_keeps_it_shown_and_undo_puts_the_whole_entry_back() {
        let mut doc = Document::with_cage(Cage::cube());
        let part = *doc.parts.keys().next().expect("a part");
        let graph = &mut doc.parts.get_mut(&part).expect("the part").graph;
        let ball = graph.add(fieldform_scene::Node::Sphere {
            centre: fieldform_scene::Place::origin(),
            radius: fieldform_scene::ParamRef::Literal(10.0),
        });
        // Consumed, so it is a variable rather than a body, and draws only because it is shown.
        graph.add(fieldform_scene::Node::Offset {
            input: ball.into(),
            distance: fieldform_scene::ParamRef::Literal(1.0),
        });
        let variable = fieldform_scene::NodeMeta {
            name: Some("ball".to_string()),
            hidden: false,
            shown: true,
        };
        doc.node_meta.insert(ball, variable.clone());

        let mut stack = CommandStack::new(64);
        stack
            .execute(&mut doc, Box::new(SetName::new(Named::Block(ball), "")))
            .expect("clears the name");
        let put_back = fieldform_scene::NodeMeta { name: None, ..variable.clone() };
        assert_eq!(doc.node_meta.get(&ball), Some(&put_back), "the name goes and the block still draws");

        stack.undo(&mut doc).expect("undoes");
        assert_eq!(doc.node_meta.get(&ball), Some(&variable), "the entry as it was, flag and all");

        stack.redo(&mut doc).expect("redoes");
        assert_eq!(doc.node_meta.get(&ball), Some(&put_back));
    }

    /// **The eye is an undo step, and it writes the one flag it is handed and nothing else.**
    ///
    /// Which flag is the app's to decide — a body's eye means `hidden`, a consumed block's means
    /// `shown` (`block-ownership-spec.md` §7.2). This holds the command to that flag alone, so the
    /// other one, the name and where every block sits come through an eye and its undo as they
    /// were; and an entry that was absent is absent again, since a block with nothing to remember
    /// has none.
    #[test]
    fn the_eye_writes_only_its_own_flag_and_undo_puts_back_exactly_what_was_there() {
        use fieldform_scene::{EyeFlag, Node, NodeMeta, ParamRef, Place};

        let mut doc = Document::with_cage(Cage::cube());
        let part = *doc.parts.keys().next().expect("a part");
        let graph = &mut doc.parts.get_mut(&part).expect("the part").graph;
        let ball =
            graph.add(Node::Sphere { centre: Place::origin(), radius: ParamRef::Literal(10.0) });
        let grown = graph.add(Node::Offset { input: ball.into(), distance: ParamRef::Literal(1.0) });
        doc.settle_ownership();
        let placed = (doc.ownership_nodes.clone(), doc.ownership_curves.clone());
        let mut stack = CommandStack::new(64);

        // A body's eye turned off: `hidden`, in an entry that was not there.
        assert!(!doc.node_meta.contains_key(&grown));
        stack
            .execute(&mut doc, Box::new(SetVisible::new(grown, EyeFlag::Hidden, false)))
            .expect("hides the body");
        let hidden = NodeMeta { hidden: true, ..NodeMeta::default() };
        assert_eq!(doc.node_meta.get(&grown), Some(&hidden));
        stack.undo(&mut doc).expect("undoes");
        assert!(!doc.node_meta.contains_key(&grown), "and leaves no entry");
        stack.redo(&mut doc).expect("redoes");
        assert_eq!(doc.node_meta.get(&grown), Some(&hidden));

        // A consumed block's eye turned on: `shown`, beside a name and a `hidden` it leaves alone.
        let named = NodeMeta { name: Some("ball".to_string()), hidden: true, shown: false };
        doc.node_meta.insert(ball, named.clone());
        stack
            .execute(&mut doc, Box::new(SetVisible::new(ball, EyeFlag::Shown, true)))
            .expect("draws the ball as well");
        assert_eq!(doc.node_meta.get(&ball), Some(&NodeMeta { shown: true, ..named.clone() }));
        stack.undo(&mut doc).expect("undoes");
        assert_eq!(doc.node_meta.get(&ball), Some(&named), "the entry as it was");
        stack.redo(&mut doc).expect("redoes");
        stack.undo(&mut doc).expect("undoes again");
        assert_eq!(doc.node_meta.get(&ball), Some(&named), "and as it was however often");

        assert_eq!(
            (doc.ownership_nodes.clone(), doc.ownership_curves.clone()),
            placed,
            "and no block moved",
        );
    }

    fn tessellation(name: &str) -> fieldform_scene::Tessellation {
        fieldform_scene::Tessellation {
            id: fieldform_scene::TessellationId::new(),
            name: Some(name.into()),
            part: fieldform_scene::PartId::new(),
            body: None,
            detail: fieldform_scene::MeshDetail::Normal,
            improve: true,
            reduce: false,
            active: true,
        }
    }

    /// **A surface's points move and come back exactly**, the way a cage's do: the positions
    /// before the gesture are what an undo writes, not the move subtracted again.
    #[test]
    fn undoing_a_surface_move_puts_its_points_back_exactly() {
        let (mut doc, _) = doc();
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let panel = fieldform_cage::QuadMesh::from_faces(positions.clone(), &[vec![0, 1, 2, 3]])
            .expect("one quad");
        let id = doc.add_quad_mesh(panel, "panel");
        let corner = VertexId::from_index(2);
        let moved = Point3::new(1.1, 1.3, 0.7);
        let mut stack = CommandStack::default();
        stack
            .execute(
                &mut doc,
                Box::new(SetSurfacePositions::new(id, vec![corner], vec![positions[2]], vec![moved])),
            )
            .unwrap();
        assert_eq!(doc.quad_meshes[&id].vertex_position(corner), moved);
        stack.undo(&mut doc);
        assert_eq!(doc.quad_meshes[&id].vertex_position(corner), positions[2]);
        stack.redo(&mut doc);
        assert_eq!(doc.quad_meshes[&id].vertex_position(corner), moved);
    }

    /// **Undoing a deleted block puts it back where it was, not at the end.** The list is the
    /// order the rows are shown in, so a block that came back last would look like a different
    /// block. Holding both whole lists is what makes that automatic.
    #[test]
    fn undoing_an_output_delete_restores_it_in_place() {
        let (mut doc, _) = doc();
        let (a, b, c) = (tessellation("a"), tessellation("b"), tessellation("c"));
        doc.tessellations = vec![a.clone(), b.clone(), c.clone()];
        let mut stack = CommandStack::default();
        let before = (doc.tessellations.clone(), Vec::new());
        stack
            .execute(
                &mut doc,
                Box::new(SetOutputs::new(before, (vec![a.clone(), c.clone()], Vec::new()))),
            )
            .unwrap();
        assert_eq!(doc.tessellations, vec![a.clone(), c.clone()]);
        stack.undo(&mut doc);
        assert_eq!(doc.tessellations, vec![a, b, c]);
    }

    /// **One Delete, one Ctrl+Z.** A batch undoes every part, in reverse, so each part undoes
    /// against the state it was applied to — here a curve removed and a block removed together
    /// come back together.
    #[test]
    fn a_batch_undoes_every_part_of_itself_at_once() {
        let (mut doc, _) = doc();
        let curve = Curve::polyline(
            CurveId::new(),
            "edge",
            vec![Point3::origin(), Point3::new(1.0, 0.0, 0.0)],
            false,
        )
        .unwrap();
        doc.curves.insert(curve.id, curve.clone());
        let block = tessellation("mesh");
        doc.tessellations = vec![block.clone()];

        let mut stack = CommandStack::default();
        let batch = Batch::new(
            "import.delete",
            vec![
                Box::new(SetCurve::remove(curve.clone())),
                Box::new(SetOutputs::new(
                    (vec![block.clone()], Vec::new()),
                    (Vec::new(), Vec::new()),
                )),
            ],
        );
        stack.execute(&mut doc, Box::new(batch)).unwrap();
        assert!(doc.curves.is_empty() && doc.tessellations.is_empty());
        assert_eq!(stack.undo(&mut doc).as_deref(), Some("import.delete"));
        assert!(doc.curves.contains_key(&curve.id));
        assert_eq!(doc.tessellations, vec![block]);
    }

    /// **A parameter put in, taken out or swapped comes back exactly as it was.** Its name and its
    /// expression with it, not only a number under the same id, since an expression read back as
    /// its value would turn a relationship into a constant.
    #[test]
    fn a_parameter_put_in_taken_out_or_swapped_is_put_back_exactly_by_undo() {
        let (mut doc, _) = doc();
        let radius = doc.add_parameter("radius", 4.0);
        let boss = doc.add_parameter("boss_r", 0.0);
        doc.set_parameter_expression(boss, "radius * 5 - 2").unwrap();
        let held = |doc: &Document, id| doc.parameters.get(&id).cloned();
        let (radius_was, boss_was) = (held(&doc, radius), held(&doc, boss));
        let mut stack = CommandStack::default();

        let wall = Parameter {
            id: ParamId::new(),
            name: "wall".into(),
            value: fieldform_scene::ParamValue::Literal(2.4),
        };
        stack.execute(&mut doc, Box::new(SetParameter::insert(wall.clone()))).unwrap();
        assert_eq!(held(&doc, wall.id).as_ref(), Some(&wall));
        stack.execute(&mut doc, Box::new(SetParameter::remove(boss_was.clone().unwrap()))).unwrap();
        assert_eq!(held(&doc, boss), None);
        let mut renamed = radius_was.clone().unwrap();
        renamed.name = "r".into();
        stack
            .execute(
                &mut doc,
                Box::new(SetParameter::new(radius, radius_was.clone(), Some(renamed.clone()))),
            )
            .unwrap();
        assert_eq!(held(&doc, radius), Some(renamed));

        for _ in 0..3 {
            assert_eq!(stack.undo(&mut doc).as_deref(), Some("parameter.set"));
        }
        assert_eq!(held(&doc, radius), radius_was);
        assert_eq!(held(&doc, boss), boss_was, "the expression came back, not its value");
        assert_eq!(held(&doc, wall.id), None);
        assert_eq!(doc.evaluate_parameter(boss).ok(), Some(18.0));
    }

    /// A literal parameter that is not in any document yet.
    fn loose(name: &str, value: f64) -> Parameter {
        Parameter {
            id: ParamId::new(),
            name: name.into(),
            value: fieldform_scene::ParamValue::Literal(value),
        }
    }

    /// **Undo and redo never take out a parameter something has come to read since.**
    ///
    /// Expressions are typed off the history, so a reader can arrive after the entry that put a
    /// parameter in: `boss_r = radius * 5 - 2` typed once `radius` was named. Taking `radius` out
    /// then left `boss_r` computing nothing, and the saved file refused to open on it -- the
    /// dangling reference the command is there to prevent. Left in place, the worst an undo leaves
    /// is a parameter nothing reads, and once nothing does the same entry takes it.
    #[test]
    fn undo_and_redo_never_take_out_a_parameter_something_has_come_to_read() {
        let (mut doc, _) = doc();
        let boss = doc.add_parameter("boss_r", 48.0);
        let radius = loose("radius", 10.0);
        let mut stack = CommandStack::default();
        stack.execute(&mut doc, Box::new(SetParameter::insert(radius.clone()))).unwrap();
        doc.set_parameter_expression(boss, "radius * 5 - 2").unwrap();

        assert_eq!(stack.undo(&mut doc).as_deref(), Some("parameter.set"));
        assert!(doc.parameters.contains_key(&radius.id), "an expression reads it");
        assert_eq!(doc.parameter_value(boss), Some(48.0));

        assert_eq!(stack.redo(&mut doc).as_deref(), Some("parameter.set"));
        doc.set_parameter_literal(boss, 48.0);
        assert_eq!(stack.undo(&mut doc).as_deref(), Some("parameter.set"));
        assert!(!doc.parameters.contains_key(&radius.id), "nothing reads it now");

        // A removal redone is held to the same rule.
        let wall = doc.add_parameter("wall", 2.0);
        let wall_was = doc.parameters[&wall].clone();
        stack.execute(&mut doc, Box::new(SetParameter::remove(wall_was))).unwrap();
        assert_eq!(stack.undo(&mut doc).as_deref(), Some("parameter.set"));
        doc.set_parameter_expression(boss, "wall * 24").unwrap();
        assert_eq!(stack.redo(&mut doc).as_deref(), Some("parameter.set"));
        assert!(doc.parameters.contains_key(&wall), "an expression reads it");
        assert_eq!(doc.parameter_value(boss), Some(48.0));
    }

    /// **A parameter comes back as it was when it left, not as it was when the entry was made.**
    ///
    /// Its name, number and expression are edited off the history, so between an entry being made
    /// and an undo or redo reaching it the parameter may have been renamed or given another number.
    /// Putting back the copy the entry was built with undid those edits as a side effect: naming a
    /// value, typing over its placeholder name, then undo and redo gave the placeholder back, and
    /// every block reading it moved with the number.
    #[test]
    fn a_parameter_comes_back_as_it_last_left_not_as_it_was_made() {
        let (mut doc, _) = doc();
        let made = loose("radius", 10.0);
        let id = made.id;
        let mut stack = CommandStack::default();
        stack.execute(&mut doc, Box::new(SetParameter::insert(made))).unwrap();
        doc.rename_parameter(id, "wall").unwrap();
        doc.set_parameter_literal(id, 20.0);
        stack.undo(&mut doc).unwrap();
        assert!(!doc.parameters.contains_key(&id));
        stack.redo(&mut doc).unwrap();
        let back = &doc.parameters[&id];
        assert_eq!((back.name.as_str(), back.value.as_literal()), ("wall", Some(20.0)));

        // A removal the same way round: put back, renamed, taken out again by redo, put back.
        let gone = doc.parameters[&id].clone();
        stack.execute(&mut doc, Box::new(SetParameter::remove(gone))).unwrap();
        stack.undo(&mut doc).unwrap();
        doc.rename_parameter(id, "rim").unwrap();
        stack.redo(&mut doc).unwrap();
        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.parameters[&id].name, "rim");
    }

    /// **A parameter put back takes a free name if another has taken its own meanwhile.**
    ///
    /// Parameters are made and renamed off the history, so while one is out another can take its
    /// name. Putting it back beside that one under the same name made a document
    /// `rename_parameter` exists to refuse, and a file refused on load once any expression is in
    /// it. It comes back as `radius_2`, which is what naming a second value `radius` gives.
    #[test]
    fn a_parameter_put_back_takes_a_free_name_if_another_has_taken_its_own() {
        let (mut doc, _) = doc();
        let radius = doc.add_parameter("radius", 10.0);
        let radius_was = doc.parameters[&radius].clone();
        let mut stack = CommandStack::default();
        stack.execute(&mut doc, Box::new(SetParameter::remove(radius_was))).unwrap();
        let other = doc.add_parameter("radius", 3.0);
        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.parameters[&other].name, "radius");
        assert_eq!(doc.parameters[&radius].name, "radius_2");
        assert_eq!(doc.parameter_value(radius), Some(10.0));
    }

    /// **Every expression a put-back or a swap touches says the names its parameters have now.**
    ///
    /// An expression reads by id and keeps its text for display and for the file, which parses it
    /// again by name. A rename re-renders the expressions in the document at the time, so one taken
    /// out while a parameter it reads was renamed came back saying the old name, and the saved file
    /// refused to open on it. A swap that renames is the same from the other side: what reads the
    /// parameter has to follow it, as it follows `rename_parameter`.
    #[test]
    fn every_expression_a_put_back_or_a_swap_touches_says_the_names_its_parameters_have_now() {
        let (mut doc, _) = doc();
        let radius = doc.add_parameter("radius", 4.0);
        let boss = doc.add_parameter("boss_r", 0.0);
        doc.set_parameter_expression(boss, "radius * 5 - 2").unwrap();
        let said = |doc: &Document| {
            doc.parameters[&boss].value.expression().map(|e| e.source().to_string())
        };
        let mut stack = CommandStack::default();

        let boss_was = doc.parameters[&boss].clone();
        stack.execute(&mut doc, Box::new(SetParameter::remove(boss_was))).unwrap();
        doc.rename_parameter(radius, "wall").unwrap();
        stack.undo(&mut doc).unwrap();
        assert_eq!(said(&doc).as_deref(), Some("wall * 5 - 2"));

        let radius_was = doc.parameters[&radius].clone();
        let mut renamed = radius_was.clone();
        renamed.name = "rim".into();
        let swap = SetParameter::new(radius, Some(radius_was), Some(renamed));
        stack.execute(&mut doc, Box::new(swap)).unwrap();
        assert_eq!(said(&doc).as_deref(), Some("rim * 5 - 2"));
        stack.undo(&mut doc).unwrap();
        assert_eq!(said(&doc).as_deref(), Some("wall * 5 - 2"));
        assert_eq!(doc.evaluate_parameter(boss).ok(), Some(18.0));
    }

    fn doc() -> (Document, CageId) {
        let d = Document::with_cage(Cage::cube());
        let id = *d.cages.keys().next().unwrap();
        (d, id)
    }

    fn position(d: &Document, cage: CageId, v: VertexId) -> Point3<f64> {
        d.cage(cage).unwrap().vertex_position(v)
    }

    /// A cube with one corner pulled out and another dragged in, so no symmetry can let a
    /// wrongly transcribed rule agree with the right one by accident.
    fn lopsided_doc() -> (Document, CageId) {
        let (mut d, id) = doc();
        let cage = d.cage_mesh_mut(id).unwrap();
        let (a, b) = (VertexId::from_index(0), VertexId::from_index(5));
        let (pa, pb) = (cage.vertex_position(a), cage.vertex_position(b));
        cage.set_vertex_position(a, pa + Vector3::new(-0.31, -0.17, 0.22));
        cage.set_vertex_position(b, pb + Vector3::new(-0.12, 0.08, -0.27));
        (d, id)
    }

    fn refine(d: &mut Document, stack: &mut CommandStack, cage: CageId) -> Result<(), OpError> {
        stack.execute(d, Box::new(RefineCage::new(cage)))
    }

    #[test]
    fn refining_a_cube_gives_the_counts_catmull_clark_promises() {
        let (mut d, id) = doc();
        let mut stack = CommandStack::default();
        refine(&mut d, &mut stack, id).unwrap();

        let cage = d.cage(id).unwrap();
        assert_eq!(
            (cage.vertex_count(), cage.edge_count(), cage.face_count()),
            (26, 48, 24),
            "a cube's 8 / 12 / 6 becomes 26 / 48 / 24"
        );
        assert_eq!(cage.euler_characteristic(), 2, "still a sphere");
        assert!(!cage.is_open(), "and still closed");
    }

    /// **The load-bearing one: the limit surface does not move.** This is what separates a
    /// refine from a linear subdivide, which has the same topology and visibly inflates the
    /// model, and it is what catches a rule transcribed wrong. Every original vertex keeps its
    /// index and its limit position; and the *new* vertices — the face points — land on the
    /// original surface at the middle of the face they were made from.
    #[test]
    fn refining_leaves_the_limit_surface_where_it_was() {
        let (mut d, id) = lopsided_doc();
        let before = d.cage(id).unwrap().clone();
        let mut stack = CommandStack::default();
        refine(&mut d, &mut stack, id).unwrap();
        let after = d.cage(id).unwrap();

        let mut worst = 0.0f64;
        for v in before.vertex_ids() {
            let was = fieldform_subd::limit_position(&before, v);
            let is = fieldform_subd::limit_position(after, v);
            worst = worst.max((was - is).norm());
        }
        assert!(worst < 1e-9, "an original vertex's limit position moved by {worst:e}");

        // Face points are laid out after the originals, one per original face, in face order.
        let mut surface = fieldform_subd::LimitSurface::new(&before);
        let n = before.vertex_count();
        let mut worst_mid = 0.0f64;
        for (i, face) in before.face_ids().enumerate() {
            let on_original = surface.evaluate(face, 0.5, 0.5).unwrap();
            let face_point = fieldform_subd::limit_position(after, VertexId::from_index(n + i));
            worst_mid = worst_mid.max((on_original - face_point).norm());
        }
        eprintln!("limit surface after a refine: vertices {worst:e}, face midpoints {worst_mid:e}");
        assert!(
            worst_mid < 1e-9,
            "a face point is {worst_mid:e} off the surface it was made from"
        );
    }

    /// Linear subdivision: the same topology change as a refine and none of the smoothing — the
    /// operation `cage-refine-spec.md` §3 says must never be a mode of this one.
    fn linear_subdivide(cage: &Cage) -> Cage {
        use std::collections::HashMap;
        let mut positions: Vec<Point3<f64>> =
            cage.vertex_ids().map(|v| cage.vertex_position(v)).collect();
        let mut face_point = Vec::new();
        for face in cage.face_ids() {
            let sum = cage
                .face_vertices(face)
                .iter()
                .fold(Vector3::zeros(), |s, v| s + cage.vertex_position(*v).coords);
            face_point.push(positions.len() as u32);
            positions.push(Point3::from(sum / 4.0));
        }
        let mut edge_point: HashMap<(u32, u32), u32> = HashMap::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();
        for (i, face) in cage.face_ids().enumerate() {
            let c = cage.face_vertices(face).map(|v| v.index() as u32);
            let mut mid = [0u32; 4];
            for k in 0..4 {
                let (a, b) = (c[k], c[(k + 1) % 4]);
                mid[k] = *edge_point.entry((a.min(b), a.max(b))).or_insert_with(|| {
                    let at =
                        (positions[a as usize].coords + positions[b as usize].coords) / 2.0;
                    positions.push(Point3::from(at));
                    positions.len() as u32 - 1
                });
            }
            let f = face_point[i];
            faces.push(vec![c[0], mid[0], f, mid[3]]);
            faces.push(vec![c[1], mid[1], f, mid[0]]);
            faces.push(vec![c[2], mid[2], f, mid[1]]);
            faces.push(vec![c[3], mid[3], f, mid[2]]);
        }
        Cage::from_faces(positions, &faces).expect("a linear subdivision of a cage is a cage")
    }

    /// **The control for the test above**, because a check that cannot fail proves nothing: the
    /// same comparison, on the one operation it exists to tell apart from a refine. Linear
    /// subdivision has the same counts and passes every structural assertion — it is only this
    /// one that sees it: 0.278 on the lopsided cube where a refine reads 1.1e-16.
    #[test]
    fn the_limit_surface_check_can_tell_a_linear_subdivision_from_a_refine() {
        let (d, id) = lopsided_doc();
        let before = d.cage(id).unwrap();
        let linear = linear_subdivide(before);
        assert_eq!(
            (linear.vertex_count(), linear.edge_count(), linear.face_count()),
            (26, 48, 24),
            "the same counts as a refine, so a count check cannot see the difference"
        );
        let worst = before
            .vertex_ids()
            .map(|v| {
                (fieldform_subd::limit_position(before, v)
                    - fieldform_subd::limit_position(&linear, v))
                .norm()
            })
            .fold(0.0, f64::max);
        eprintln!("a linear subdivision moves an original vertex's limit position by {worst:e}");
        assert!(worst > 1e-2, "the check read {worst:e} for a subdivision that inflates the model");
    }

    #[test]
    fn a_refined_cage_keeps_its_id_and_changes_what_a_cache_keyed_on_it_would_see() {
        let (mut d, id) = lopsided_doc();
        let before = d.cage(id).unwrap().geometry_fingerprint();
        let mut stack = CommandStack::default();
        refine(&mut d, &mut stack, id).unwrap();
        assert!(d.cages.contains_key(&id), "every Node::Cage naming it must keep resolving");
        assert_eq!(d.cages.len(), 1, "replaced, not added beside");
        assert_ne!(
            d.cage(id).unwrap().geometry_fingerprint(),
            before,
            "the leaf cache, the traced bake and the tessellate block are keyed on this"
        );
    }

    /// Refine twice, undo twice: every position bit-for-bit and every count restored, against
    /// a copy taken before anything happened.
    #[test]
    fn undoing_a_refine_puts_the_cage_back_exactly() {
        let (mut d, id) = lopsided_doc();
        let original = d.cage(id).unwrap().clone();
        let mut stack = CommandStack::default();
        refine(&mut d, &mut stack, id).unwrap();
        let once = d.cage(id).unwrap().clone();
        refine(&mut d, &mut stack, id).unwrap();
        assert_eq!(d.cage(id).unwrap().face_count(), 96);

        assert_eq!(stack.undo(&mut d).as_deref(), Some("cage.refine"));
        assert_eq!(d.cage(id).unwrap().geometry_fingerprint(), once.geometry_fingerprint());
        assert_eq!(stack.undo(&mut d).as_deref(), Some("cage.refine"));
        let back = d.cage(id).unwrap();
        assert_eq!(back.geometry_fingerprint(), original.geometry_fingerprint());
        assert_eq!(
            (back.vertex_count(), back.edge_count(), back.face_count()),
            (original.vertex_count(), original.edge_count(), original.face_count())
        );
        for v in original.vertex_ids() {
            assert_eq!(back.vertex_position(v), original.vertex_position(v));
        }

        // And redo is the same refine again: subdividing the same cage with the same code is
        // bit-identical, which is why nothing caches the refined cage.
        assert!(stack.redo(&mut d).is_some());
        assert_eq!(d.cage(id).unwrap().geometry_fingerprint(), once.geometry_fingerprint());
    }

    #[test]
    fn a_refine_past_the_face_limit_is_refused_with_both_numbers() {
        let (mut d, id) = doc();
        let mut stack = CommandStack::default();
        for _ in 0..5 {
            refine(&mut d, &mut stack, id).unwrap();
        }
        assert_eq!(d.cage(id).unwrap().face_count(), 6_144, "five refines of a cube are allowed");

        let before = d.cage(id).unwrap().geometry_fingerprint();
        let refused = refine(&mut d, &mut stack, id).unwrap_err().to_string();
        assert!(refused.contains("24,576"), "what it would have given: {refused}");
        assert!(refused.contains("20,000"), "what is allowed: {refused}");
        assert_eq!(
            d.cage(id).unwrap().geometry_fingerprint(),
            before,
            "a refusal changes nothing"
        );
        assert!(
            stack.undo(&mut d).is_some() && d.cage(id).unwrap().face_count() == 1_536,
            "and puts nothing on the undo stack: the next undo is the fifth refine"
        );
    }

    #[test]
    fn the_registry_builds_a_refine_and_refuses_one_over_the_limit_before_it_is_run() {
        let (mut d, id) = doc();
        let registry = OpRegistry::default();
        assert!(registry.names().contains(&"cage.refine"));
        let mut stack = CommandStack::default();
        let cmd = registry.build("cage.refine", &Args::new(), &d).unwrap();
        stack.execute(&mut d, cmd).unwrap();
        assert_eq!(d.cage(id).unwrap().face_count(), 24);

        // Four more takes it to 6 144 faces, and the next would give 24 576: the registry says
        // no when asked to *build* the command, before anything is on the undo stack.
        for _ in 0..4 {
            refine(&mut d, &mut stack, id).unwrap();
        }
        assert!(registry.build("cage.refine", &Args::new(), &d).is_err());
    }

    /// A surface axis is made, edited and removed through the stack, each reversing exactly: the
    /// record before is what an undo writes, and an edit's undo does not leave the edit's numbers.
    #[test]
    fn a_surface_axis_is_made_edited_and_undone_through_the_stack() {
        use fieldform_scene::{Axis, NodeId, SurfaceAxis};
        let (mut d, _) = doc();
        let curve = CurveId::new();
        let made = SurfaceAxis {
            body: NodeId::from_raw(3),
            direction: Axis::Z,
            flipped: false,
            u: 0.5,
            v: 0.5,
            length: 50.0,
            seed: None,
        };
        let edited = SurfaceAxis { u: 0.2, length: 12.0, ..made };
        let mut stack = CommandStack::default();

        stack.execute(&mut d, Box::new(SetSurfaceAxis::insert(curve, made))).unwrap();
        assert_eq!(d.surface_axes.get(&curve), Some(&made));

        stack
            .execute(&mut d, Box::new(SetSurfaceAxis::new(curve, Some(made), Some(edited))))
            .unwrap();
        assert_eq!(d.surface_axes.get(&curve), Some(&edited));

        assert_eq!(stack.undo(&mut d).as_deref(), Some("surface_axis.set"));
        assert_eq!(d.surface_axes.get(&curve), Some(&made), "the edit is gone, the axis is not");
        assert_eq!(stack.undo(&mut d).as_deref(), Some("surface_axis.set"));
        assert!(d.surface_axes.is_empty(), "undoing the making takes the record out");
        assert!(stack.redo(&mut d).is_some());
        assert_eq!(d.surface_axes.get(&curve), Some(&made));
    }

    #[test]
    fn grouped_numbers_read_with_commas() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(20_000), "20,000");
        assert_eq!(grouped(24_576), "24,576");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn setting_a_material_then_undoing_puts_back_exactly_what_the_part_had() {
        let (mut d, _) = doc();
        let part = *d.parts.keys().next().unwrap();
        let mut stack = CommandStack::default();

        let chrome = Material::preset("Chrome");
        let cmd = Box::new(SetPartMaterial::new(part, None, chrome.clone()));
        stack.execute(&mut d, cmd).unwrap();
        assert_eq!(d.parts[&part].material, chrome);

        // The second edit remembers the first as its "before", not the default: undoing one
        // step goes back one step.
        let paint = Material::preset("Car paint");
        let cmd = Box::new(SetPartMaterial::new(part, chrome.clone(), paint.clone()));
        stack.execute(&mut d, cmd).unwrap();
        assert_eq!(d.parts[&part].material, paint);

        assert_eq!(stack.undo(&mut d), Some("part.set_material".to_string()));
        assert_eq!(d.parts[&part].material, chrome);
        assert_eq!(stack.undo(&mut d), Some("part.set_material".to_string()));
        assert_eq!(d.parts[&part].material, None);
        assert!(stack.redo(&mut d).is_some());
        assert_eq!(d.parts[&part].material, chrome);
    }

    #[test]
    fn translate_then_undo_restores_exact_positions() {
        let (mut d, cage) = doc();
        let mut stack = CommandStack::default();
        let v = VertexId::from_index(0);
        let before = position(&d, cage, v);

        let cmd = Box::new(TranslateVertices::new(
            cage,
            vec![v],
            Vector3::new(1.0, 2.0, 3.0),
        ));
        stack.execute(&mut d, cmd).unwrap();
        assert_eq!(
            position(&d, cage, v),
            before + Vector3::new(1.0, 2.0, 3.0)
        );

        assert!(stack.undo(&mut d).is_some());
        assert_eq!(position(&d, cage, v), before);
    }

    /// The curve half of the same guarantee the cage already has: one gesture is one entry,
    /// and reversing it puts back the exact positions rather than a recomputed approximation.
    #[test]
    fn moving_control_points_then_undoing_restores_exact_positions() {
        use fieldform_scene::{Curve, CurveId};

        let mut doc = Document::new();
        let id = CurveId::new();
        let start = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(20.0, 0.0, 0.0),
        ];
        doc.curves
            .insert(id, Curve::polyline(id, "spine", start.clone(), false).unwrap());

        let moved = vec![Point3::new(3.0, 4.0, 5.0), Point3::new(23.5, -1.0, 0.25)];
        let mut stack = CommandStack::default();
        stack
            .execute(
                &mut doc,
                Box::new(SetControlPoints::new(
                    id,
                    vec![0, 2],
                    vec![start[0], start[2]],
                    moved.clone(),
                )),
            )
            .unwrap();

        let after = doc.curve(id).unwrap().control_points().to_vec();
        assert_eq!(after[0], moved[0]);
        assert_eq!(after[2], moved[1]);
        assert_eq!(after[1], start[1], "an unselected point must not move");

        assert_eq!(stack.undo(&mut doc), Some("curve.set_control_points".to_string()));
        assert_eq!(doc.curve(id).unwrap().control_points(), start.as_slice());
    }

    /// A weight is the other half of what a control point is, so it reverses the same way — and
    /// it has to actually bend the curve, or the field would be editing a number nothing reads.
    #[test]
    fn changing_a_weight_bends_the_curve_and_undoing_puts_it_back() {
        use fieldform_scene::{Curve, CurveId};

        let mut doc = Document::new();
        let id = CurveId::new();
        let start = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(20.0, 0.0, 0.0),
        ];
        doc.curves.insert(id, Curve::polyline(id, "arc", start, false).unwrap());
        let midpoint = |d: &Document| d.curve(id).unwrap().evaluate(0.5);
        let before_shape = midpoint(&doc);

        let mut stack = CommandStack::default();
        stack
            .execute(
                &mut doc,
                Box::new(SetControlWeights::new(id, vec![1], vec![1.0], vec![8.0])),
            )
            .unwrap();

        let pulled = midpoint(&doc);
        assert!(
            (pulled - before_shape).norm() > 1e-6,
            "a heavier middle point should pull the curve toward it",
        );
        assert!(doc.curve(id).unwrap().is_rational());

        assert_eq!(stack.undo(&mut doc), Some("curve.set_weights".to_string()));
        assert!((midpoint(&doc) - before_shape).norm() < 1e-12);
    }

    /// A curve deleted out from under a stale command is an error the stack can report, not a
    /// panic — the same treatment a missing cage already gets.
    #[test]
    fn moving_points_on_a_curve_that_is_gone_is_an_error_not_a_panic() {
        use fieldform_scene::CurveId;

        let mut doc = Document::new();
        let mut cmd = SetControlPoints::new(
            CurveId::new(),
            vec![0],
            vec![Point3::origin()],
            vec![Point3::new(1.0, 0.0, 0.0)],
        );
        assert!(cmd.execute(&mut doc).is_err());
    }

    #[test]
    fn the_history_reads_back_most_recent_first() {
        // What a browsable history would show, as the Edit menu's once did (see `undo_names`).
        // Order is the property: a list that reads oldest first would undo the wrong entry when
        // clicked, and both ends look plausible.
        let (mut d, cage) = doc();
        let mut stack = CommandStack::default();
        for i in 1..=3 {
            let cmd = Box::new(TranslateVertices::new(
                cage,
                vec![VertexId::from_index(i)],
                Vector3::new(1.0, 0.0, 0.0),
            ));
            stack.execute(&mut d, cmd).unwrap();
        }
        assert_eq!(stack.undo_names().count(), 3);
        assert_eq!(stack.undo_names().next(), stack.peek_undo(), "the head is the next undo");
        assert_eq!(stack.redo_names().count(), 0);

        stack.undo(&mut d);
        assert_eq!(stack.undo_names().count(), 2);
        assert_eq!(stack.redo_names().next(), stack.peek_redo(), "and so is the next redo");
    }

    #[test]
    fn redo_reapplies_and_new_edit_clears_the_redo_branch() {
        let (mut d, cage) = doc();
        let mut stack = CommandStack::default();
        let v = VertexId::from_index(0);
        let before = position(&d, cage, v);
        let delta = Vector3::new(0.5, 0.0, 0.0);

        stack
            .execute(
                &mut d,
                Box::new(TranslateVertices::new(cage, vec![v], delta)),
            )
            .unwrap();
        stack.undo(&mut d);
        assert_eq!(position(&d, cage, v), before);

        assert!(stack.can_redo());
        stack.redo(&mut d);
        assert_eq!(position(&d, cage, v), before + delta);

        stack.undo(&mut d);
        assert!(stack.can_redo());
        stack
            .execute(
                &mut d,
                Box::new(TranslateVertices::new(
                    cage,
                    vec![v],
                    Vector3::new(0.0, 1.0, 0.0),
                )),
            )
            .unwrap();
        assert!(!stack.can_redo(), "a new edit should clear the redo history");
    }

    #[test]
    fn repeated_undo_redo_does_not_drift() {
        // The reason commands store absolute prior positions rather than inverting a delta.
        let (mut d, cage) = doc();
        let mut stack = CommandStack::default();
        let v = VertexId::from_index(0);
        let before = position(&d, cage, v);

        stack
            .execute(
                &mut d,
                Box::new(TranslateVertices::new(
                    cage,
                    vec![v],
                    Vector3::new(0.1, 0.1, 0.1),
                )),
            )
            .unwrap();
        for _ in 0..50 {
            stack.undo(&mut d);
            stack.redo(&mut d);
        }
        stack.undo(&mut d);
        assert_eq!(
            position(&d, cage, v),
            before,
            "position must return exactly, with no accumulated error"
        );
    }

    #[test]
    fn undo_on_empty_stack_is_a_no_op() {
        let (mut d, _) = doc();
        let mut stack = CommandStack::default();
        assert!(stack.undo(&mut d).is_none());
        assert!(!stack.can_undo());
    }

    #[test]
    fn registry_dispatches_by_name_with_dynamic_args() {
        // The path a Python call or a keymap binding will take.
        let (mut d, cage) = doc();
        let registry = OpRegistry::default();
        let mut stack = CommandStack::default();
        let v = VertexId::from_index(2);
        let before = position(&d, cage, v);

        let args = Args::new()
            .with("vertices", Value::IntList(vec![2]))
            .with("delta", Value::Vec3([0.0, 0.0, 1.5]));
        let cmd = registry
            .build("cage.translate_vertices", &args, &d)
            .expect("registered op should build");
        stack.execute(&mut d, cmd).unwrap();

        assert_eq!(
            position(&d, cage, v),
            before + Vector3::new(0.0, 0.0, 1.5)
        );
        stack.undo(&mut d);
        assert_eq!(position(&d, cage, v), before);
    }

    #[test]
    fn the_cage_argument_is_optional_only_while_it_is_unambiguous() {
        // One cage: omitting it is fine, because there is nothing to guess.
        let (mut d, cage) = doc();
        let registry = OpRegistry::default();
        let args = Args::new()
            .with("vertices", Value::IntList(vec![0]))
            .with("delta", Value::Vec3([1.0, 0.0, 0.0]));
        assert!(registry.build("cage.translate_vertices", &args, &d).is_ok());

        // Two cages: it becomes required rather than silently picking one.
        let second = d.add_cage(Cage::cube());
        let err = registry
            .build("cage.translate_vertices", &args, &d)
            .err()
            .expect("ambiguous cage must be refused");
        assert!(matches!(err, OpError::MissingArg { .. }), "got {err:?}");

        // Naming one resolves it again — using the hex form ids print as.
        let named = args.clone().with("cage", Value::Str(second.to_string()));
        assert!(registry
            .build("cage.translate_vertices", &named, &d)
            .is_ok());
        let _ = cage;
    }

    #[test]
    fn an_unknown_cage_id_is_an_error_not_a_panic() {
        let (d, _) = doc();
        let registry = OpRegistry::default();
        let args = Args::new()
            .with("vertices", Value::IntList(vec![0]))
            .with("delta", Value::Vec3([0.0, 0.0, 0.0]))
            .with("cage", Value::Str("deadbeefdeadbeef".into()));
        let err = registry
            .build("cage.translate_vertices", &args, &d)
            .err()
            .expect("unknown id must be refused");
        assert!(matches!(err, OpError::BadArg { .. }), "got {err:?}");
    }

    #[test]
    fn registry_reports_useful_errors() {
        let (d, _) = doc();
        let registry = OpRegistry::default();

        let err = registry
            .build("cage.nonexistent", &Args::new(), &d)
            .err()
            .expect("expected this call to fail");
        assert!(matches!(err, OpError::UnknownOp(_)), "got {err:?}");

        let err = registry
            .build("cage.translate_vertices", &Args::new(), &d)
            .err()
            .expect("expected this call to fail");
        assert!(matches!(err, OpError::MissingArg { .. }), "got {err:?}");

        let args = Args::new()
            .with("vertices", Value::Str("nope".into()))
            .with("delta", Value::Vec3([0.0, 0.0, 0.0]));
        let err = registry
            .build("cage.translate_vertices", &args, &d)
            .err()
            .expect("expected this call to fail");
        assert!(matches!(err, OpError::BadArg { .. }), "got {err:?}");

        let args = Args::new()
            .with("vertices", Value::IntList(vec![9999]))
            .with("delta", Value::Vec3([0.0, 0.0, 0.0]));
        let err = registry
            .build("cage.translate_vertices", &args, &d)
            .err()
            .expect("expected this call to fail");
        assert!(matches!(err, OpError::BadArg { .. }), "got {err:?}");
    }
}
