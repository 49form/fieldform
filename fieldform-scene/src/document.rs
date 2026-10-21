// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The document model: what gets saved, scripted, and exchanged.
//!
//! # Shape
//!
//! ```text
//! Document ── unit (presentation only; geometry is always mm)
//!          ├─ cages       ── the direct/destructive editing layer
//!          ├─ curves      ── parametric leaves (Phase B)
//!          ├─ parameters  ── named values the graph can reference
//!          └─ parts       ── each owns a graph that evaluates to a field
//! ```
//!
//! # Why parts share one document
//!
//! Several parts live in one document and can reference the same cages, curves and
//! parameters. That is deliberate, and it is what makes cross-references between separately
//! owned parts unnecessary: co-dependent geometry is modelled together in one context rather
//! than one part reaching into another. Some other CAD systems work this way, and the
//! fragility of the alternative — in-context/external references in older CAD systems —
//! is well documented.
//!
//! # Scope, and what v1 deliberately leaves out
//!
//! Assemblies are out of scope for v1. What is *not* left out is the structure that lets them
//! arrive later as pure addition rather than a rewrite:
//!
//! - the document is the single root **scope**, and objects belong to a scope;
//! - a part's graph declares named **inputs**, even though nothing binds them yet.
//!
//! When assemblies land, an assembly is simply a scope that binds its children's inputs, and
//! a v1 file is a one-node scope tree that still loads. The rule that keeps this sound is:
//! **a scope may reference its own objects and any ancestor's — never a sibling's, never a
//! descendant's.** That is acyclic by construction.

use crate::{
    curve::Curve,
    expression::{EvalError, Expression, ParseError},
    graph::{Graph, ParamRef},
    id::{CageId, CurveId, MeshId, NodeId, ParamId, PartId, QuadMeshId, SourceId},
    import::StepOptions,
    material::Material,
    mesh::MeshData,
    ownership::{BlockKey, Ownership, OwnershipChange},
    unit::Unit,
};
use fieldform_cage::Cage;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A named value the graph can reference — a number typed, or one computed from others.
///
/// **Document-scoped.** `wall = 2.4` set once is the same 2.4 in the housing, the lid and the
/// bracket, and changing it moves all three. The cost is one flat namespace for the whole
/// file, so names must be unique file-wide; part-local scope can be added later, and the
/// ordering matters — starting document-scoped and adding local scope is straightforward,
/// starting local and retrofitting sharing is not.
///
/// Not serialized here: the file writes an expression as its text, and reading that back
/// needs the whole parameter table to turn names into ids, which is `fieldform-io`'s job.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub id: ParamId,
    pub name: String,
    pub value: ParamValue,
}

/// What a body is called and whether it is drawn.
///
/// Absent for most bodies, which is the point: a body with nothing to remember costs nothing,
/// and the map holds only the ones that have actually been named, hidden or shown.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeMeta {
    /// What to call it, when it has been given a name.
    ///
    /// `None` means the list works one out from what the body *is* — "Cage", "Pipe",
    /// "Cube (shelled)" — which is right until two of them are the same thing and a person
    /// needs to tell them apart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Whether the viewport leaves it out.
    ///
    /// About a **body** — something that would otherwise draw. A block consumed by another one
    /// does not draw in the first place, so this says nothing about it; see [`Self::shown`].
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
    /// Whether a **block that something else consumes** is drawn as well.
    ///
    /// The other half of [`Self::hidden`], and deliberately a second flag rather than a second
    /// meaning for the first. A body draws unless hidden; an ingredient does not draw unless
    /// shown. One field answering both would have to mean opposite things depending on where the
    /// block sat in the graph, and would flip meaning under the user the moment a block was
    /// consumed — which is exactly when this matters.
    ///
    /// Named or not. It was once read only beside a name; since 2026-09-30 the eye shows an
    /// unnamed operand, and Pattern and Mirror leave the body they consume shown without naming
    /// it — so a slot holding this and nothing else is a real slot (see [`Self::is_empty`]).
    /// Nothing about the *part* changes: its bodies are still the roots of its graph, derived and
    /// never stored. This adds to what is drawn, which is a question about a viewport.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub shown: bool,
}

impl NodeMeta {
    /// Whether there is nothing here worth writing down: no name, and neither flag set.
    ///
    /// A slot that has been emptied — the name cleared, a hidden body shown again, a consumed
    /// block no longer drawn — is removed rather than written as an empty object, so a document
    /// that has been put back the way it was saves the same bytes it did before.
    ///
    /// **Every field, and `shown` is the one that was missed.** It used to read only the name and
    /// `hidden`, from when `shown` mattered only beside a name. Once it did not, an unnamed block
    /// drawn because a pattern or the eye asked for it held a slot this called empty — so the save
    /// dropped it, and clearing a shown variable's name dropped it at once
    /// (`block-ownership-spec.md` §1). A field added here belongs in this test too.
    pub fn is_empty(&self) -> bool {
        self.name.is_none() && !self.hidden && !self.shown
    }

    /// Whether the block draws, as `flag` holds it.
    pub fn visible_by(&self, flag: EyeFlag) -> bool {
        match flag {
            EyeFlag::Hidden => !self.hidden,
            EyeFlag::Shown => self.shown,
        }
    }

    /// Writes whether the block draws into `flag`, and leaves the other flag as it is.
    pub fn set_visible_by(&mut self, flag: EyeFlag, visible: bool) {
        match flag {
            EyeFlag::Hidden => self.hidden = !visible,
            EyeFlag::Shown => self.shown = visible,
        }
    }
}

/// Which of a block's two stored flags its eye means: [`NodeMeta::hidden`] or
/// [`NodeMeta::shown`].
///
/// **Two flags and one eye** (`block-ownership-spec.md` §7). The flags stay two because one could
/// not say what draws today without something rewriting it every time a block was consumed or let
/// go of; what was wrong was that the eye spoke to them from several places that each chose a flag
/// for themselves. Which one it is depends on where the block sits in its graph, so the choice is
/// named here and made once, by [`crate::eye_flag_by`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeFlag {
    /// A body draws unless hidden — and so does a plane or a point whatever consumes it, since
    /// the overlay draws those either way and reads this flag.
    Hidden,
    /// A block something else consumes draws only when shown.
    Shown,
}

/// A file an import came from, and how it was read.
///
/// Kept so an import can stay a *live* part of the document rather than a one-way door: the
/// block shows which file it is, the settings it was read with are editable there, and pointing
/// it at a new revision re-reads it and re-evaluates everything downstream.
///
/// The path is where it was last read from and is advisory — a document moved between machines
/// keeps the name and loses the path, which is honest. The bytes themselves live in the
/// archive's `sources/`, so a re-import needs the file only when the file is what changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportSource {
    pub id: SourceId,
    /// The file's own name, which is what the block is called.
    pub name: String,
    /// Where it was last read from, for the folder button to start in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// What the reader was told, so a re-import repeats it unless the block changes it.
    #[serde(default)]
    pub settings: ImportSettings,
    /// Cages this import produced, when a file was read as a subdivision cage.
    ///
    /// **Listed here because a cage cannot say where it came from.** An imported mesh carries its
    /// source on itself; a `Cage` is `fieldform-cage`'s type and this layer's facts do not go in
    /// it. So the import names what it made, and the block list asks the import. The same for
    /// the two below: a surface to follow and an IGES file's curves are not bodies at all, and
    /// the import is the only thing that knows they belong together.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cages: Vec<CageId>,
    /// Surfaces this import produced, when a file was read as a surface to follow.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub surfaces: Vec<QuadMeshId>,
    /// Curves this import produced — what an IGES file brings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub curves: Vec<CurveId>,
}

/// What a reader was told, in the reader's own terms — one variant per format.
///
/// An enum rather than a bag of optional fields because the formats genuinely disagree about
/// what there is to say: STEP tessellates curved surfaces and has five parameters about how,
/// an OBJ is already triangles and the only open question is what one unit meant, and IGES
/// carries its own unit and is converted rather than tessellated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportSettings {
    Step(StepOptions),
    /// The scale one unit of the file meant, in millimetres. An OBJ declares no unit.
    Obj { unit_mm: f64 },
    /// IGES declares its own unit, and the importer reports what it used.
    Iges,
    /// A format this build has no settings for, or a mesh that came from none.
    None,
}

impl Default for ImportSettings {
    fn default() -> Self {
        ImportSettings::None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    /// A number, in canonical millimetres where it denotes a length.
    Literal(f64),
    /// A number computed from other parameters — see [`Expression`].
    Expression(Expression),
}

impl ParamValue {
    /// The number typed, or `None` if it is computed. For the *value* of a parameter, computed
    /// or not, ask the document: [`Document::parameter_value`].
    pub fn as_literal(&self) -> Option<f64> {
        match self {
            ParamValue::Literal(v) => Some(*v),
            ParamValue::Expression(_) => None,
        }
    }

    pub fn expression(&self) -> Option<&Expression> {
        match self {
            ParamValue::Literal(_) => None,
            ParamValue::Expression(e) => Some(e),
        }
    }
}

/// Why an expression could not be stored on a parameter.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ExpressionError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    /// The names that would read each other, first and last the same: `a → b → a`.
    #[error("would loop: {}", .0.join(" \u{2192} "))]
    Cycle(Vec<String>),
    #[error("no parameter has id {0}")]
    NoSuchParameter(ParamId),
}

/// Why a parameter could not be given a name.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum RenameError {
    #[error("`{0}` cannot be a name \u{2014} a name has something in it and no `[` or `]`")]
    NotAName(String),
    #[error("`{0}` is already a parameter")]
    Taken(String),
    #[error("no parameter has id {0}")]
    NoSuchParameter(ParamId),
}

/// Why a parameter has no value, said the way its block shows it.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ParameterError {
    /// A step of the expression — `5 / 0`, `sqrt(-4)` — did not produce a finite number.
    #[error("{0} is not a number")]
    NotFinite(String),
    /// A parameter it reads has no value; look at that one's block for the reason.
    #[error("`{0}` has no value")]
    Unresolved(String),
    /// A reference to a parameter that has left the document. Removal is guarded, so this
    /// takes a document built by hand.
    #[error("reads a parameter that is no longer in the document")]
    Missing,
    /// The chain of parameters read through one another is longer than any real document's.
    /// Cycles are refused when an expression is stored, so this is the guard for a file that
    /// was edited by hand — see [`MAX_PARAMETER_DEPTH`].
    #[error("reads through more than {0} parameters")]
    TooDeep(usize),
}

/// How many parameters one may read through before evaluation gives up.
///
/// A cycle cannot be stored through [`Document::set_parameter_expression`] and is refused on
/// load, so this never fires on a document the application built. It exists because the
/// document is also reachable from files and scripts, and a walk that follows one reference
/// into another must be bounded whatever produced the references.
pub(crate) const MAX_PARAMETER_DEPTH: usize = 64;

/// A named input a part's graph expects from its containing scope.
///
/// Unused in v1 — there is one scope, so nothing binds these. It exists because a part is
/// already a graph, and a graph with named inputs is the natural form for an assembly to wire
/// into later. Each carries a default so the part still evaluates standalone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputPort {
    pub name: String,
    pub kind: InputKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    Curve,
    Parameter,
}

/// A manufacturable unit: the thing a material is assigned to, a PLM system references, and
/// manufacturing validation runs against.
///
/// Its shape is whatever its graph evaluates to. In v1 that graph is typically a single node
/// naming one cage — trivial, but structurally the same object the parametric layer will
/// grow into.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    pub id: PartId,
    pub name: String,
    /// Materials belong to parts, as they do in manufacturing — a machined bracket is
    /// aluminium, the whole part is. Never per face.
    ///
    /// `None` means the default appearance, [`Material::default`], and is left out of the file:
    /// a part nobody has dressed carries no material, rather than a copy of the default that
    /// would have to be recognised as one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<Material>,
    pub graph: Graph,
    /// Empty in v1 — nothing binds these yet, so an empty list is the norm and is left out
    /// of the file rather than written as `[]` on every part.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<InputPort>,
}

impl Part {
    /// A part whose shape is exactly one cage's limit surface.
    pub fn from_cage(name: impl Into<String>, cage: CageId) -> Self {
        Self {
            id: PartId::new(),
            name: name.into(),
            material: None,
            graph: Graph::from_cage(cage),
            inputs: Vec::new(),
        }
    }
}

/// Everything that gets saved.
#[derive(Clone, Debug)]
pub struct Document {
    /// Authoring/display unit. Geometry is always millimetres regardless — see [`Unit`].
    pub unit: Unit,
    pub cages: HashMap<CageId, Cage>,
    /// Imported triangle meshes. Beside the cages rather than among them: both are geometry
    /// the graph references by id, and neither is the other — a cage is scaffolding that gets
    /// edited, a mesh is finished geometry that gets referenced.
    pub meshes: HashMap<MeshId, MeshData>,
    /// Open quad meshes: surfaces to follow rather than solids to evaluate.
    ///
    /// **A third collection rather than a relaxed `Cage`.** A cage is closed by construction and
    /// `CageField` depends on that — an open surface has no inside for a sign to refer to — so
    /// widening what a cage may be would put a runtime failure where a compile-time guarantee
    /// is. These are the other thing: a panel or a shell somebody wants reinforced, which is
    /// never going to be a solid and is referenced by a conformal lattice rather than evaluated
    /// as a field. Beside the cages for the same reason the imported meshes are beside them.
    pub quad_meshes: HashMap<QuadMeshId, fieldform_cage::QuadMesh>,
    /// What each surface is called — the name its file gave it.
    ///
    /// **Beside the mesh rather than inside it**, because a `QuadMesh` is `fieldform-cage`'s
    /// geometry and a name is this layer's. The same arrangement `curves_created` has, and for
    /// the same reason: the collection holds what the thing *is*, and the document holds what
    /// this document calls it.
    pub surface_names: HashMap<QuadMeshId, String>,
    pub curves: HashMap<CurveId, Curve>,
    pub parameters: HashMap<ParamId, Parameter>,
    pub parts: HashMap<PartId, Part>,
    /// When each node was made, in **microseconds** since the Unix epoch.
    ///
    /// The block list reads it and so do [`Self::folded_bodies`] and [`Self::listed_blocks`], which
    /// between them are every reader that needs
    /// a chosen order — to answer "what order did I make these in?", which nothing else in the
    /// document records. Ids are random by design (see `id.rs`: a
    /// counter collides the moment two people branch), and the graph is a set of nodes with
    /// no sequence in it, so the order a person remembers is genuinely absent unless it is
    /// written down.
    ///
    /// **Microseconds rather than milliseconds, and every value distinct.** A stamp two bodies
    /// share orders them by nothing, exactly as their ids did, so whatever writes one owes each
    /// node its own value — which needs room below the clock's own resolution to put it in. The
    /// widening costs no format change, because the stamp is a bare `u64`: every millisecond
    /// value an older build wrote is three orders of magnitude smaller than every microsecond
    /// one, so those nodes keep sorting first, which is what they are — older.
    ///
    /// A timestamp rather than a counter, for exactly the reason ids are random rather than
    /// sequential. Two people adding a body on separate branches both get counter 7 and a
    /// merge has to invent an order between them; two timestamps already interleave.
    ///
    /// Deliberately *not* part of `document.json`. It rides in its own archive entry, so a
    /// build that has never heard of it carries the bytes through untouched instead of
    /// refusing the file — which is what the format's minor-version promise requires and what
    /// `deny_unknown_fields` on the document JSON would have broken. Missing entries are
    /// ordinary: an older build that adds a node leaves it unstamped, and the list falls back
    /// to dependency order for those.
    pub created: HashMap<NodeId, u64>,
    /// Per-body notes the graph has no room for: what it is called, and whether it is shown.
    ///
    /// **One slot rather than two fields**, and outside the node rather than on it. A new field
    /// on a node kind is preserved-and-refused by an older reader (D3), so adding `name` to
    /// `Node` would make every node in the file stop evaluating in a build that predates it —
    /// for a label. This rides in its own archive entry beside `created`, which a build that
    /// has never heard of it carries through untouched.
    ///
    /// One slot because the next thing a body wants remembered — a colour, a note, a material
    /// — costs a field here instead of a second map and a second decision.
    pub node_meta: HashMap<NodeId, NodeMeta>,
    /// The files imports came from, by id. See [`ImportSource`].
    pub sources: HashMap<SourceId, ImportSource>,
    /// Where a person has dragged each block to in the list.
    ///
    /// Absent for a block nobody has moved, and **absent entirely for a document nobody has
    /// rearranged** — which is most of them, and why this is a map rather than a field on every
    /// node. An entry means "put it here"; no entry means "wherever it was made", which is what
    /// [`Self::created`] already answers. So clearing this map is "put the list back the way I
    /// built it", and that is the reason the two are separate rather than a drag overwriting the
    /// stamp: when a block was made is a *fact*, and where a person wants it is an *intention*.
    ///
    /// **A map from node to rank, never a list of rows**, for the reason `body-order-decision.md`
    /// §3 gives: a list can name a block that no longer exists or omit one that does, and a map
    /// can do neither.
    ///
    /// Deliberately **not** a field on [`NodeMeta`], which is where a per-body annotation otherwise
    /// belongs and whose own comment invites one. `NodeMeta` is `deny_unknown_fields` and
    /// `fieldform-io` loads it with `get_json(..)?`, so a field added here would not be a dropped
    /// name in an older build — it would be a **failed document load**, which is the opposite of
    /// the preserved-and-refused rule the format follows (D3). Its own archive entry is carried
    /// through untouched instead, which `unrecognised_entries_anywhere_are_carried_untouched`
    /// proves and which is the arrangement `created` and `node_meta` chose for the same reason.
    ///
    /// Nothing about the *part* changes: the bodies are still the roots of its graph, derived and
    /// never stored, and the order the compilers fold them in ignores this entirely — see
    /// [`Self::folded_bodies`].
    pub arrangement: HashMap<NodeId, u32>,
    /// [`Self::created`], for curves.
    ///
    /// A second map rather than one keyed by something both ids fit into, because the first is
    /// already written to a file and changing its shape would make the documents saved before
    /// this unreadable. Two entries beside each other cost nothing; a migration costs a format
    /// decision.
    pub curves_created: HashMap<CurveId, u64>,
    /// A name for a profile row, keyed by the row's first curve (`profile-group-spec.md`
    /// §3.3). The only stored piece of a row, and optional: which curves make the row is
    /// derived by walking, every time it is asked. When the key curve leaves its loop the
    /// name no longer names a row, and the application drops it on the next save rather than
    /// carrying it forever.
    ///
    /// **A label, and nothing reads it to decide a row** (`block-ownership-spec.md` §10). It was
    /// the profile row's declaration until the ownership was stored, so naming a row lifted the
    /// loop out of the block made from it; the row's V writes [`Self::ownership_curves`] now. The
    /// rule a file from before that is placed by still reads a name as a declaration
    /// ([`Self::legacy_ownership`]), because that is how its list was laid out.
    pub profile_names: HashMap<CurveId, String>,
    /// The curves an older build was asked to keep on a row of their own, rather than drawn inside
    /// the block that reads them — `history/declared-curves.json` as the file said it.
    ///
    /// **Read once, when a file is opened, and by nothing after.** The curve row's V was this flag
    /// from 2026-10-01 (backlog item 73: an extrude names one curve of its loop, so when the loop
    /// opened that curve stayed inside the extrude with no way out) until the ownership was stored
    /// (`block-ownership-spec.md` §10). Now the V writes [`Self::ownership_curves`], and this is
    /// what [`Self::legacy_ownership`] reads to place a curve a file does not place — every curve,
    /// in a file from before 3.15. Nothing writes it while a document is open: a save writes the
    /// entry from the ownership, for an older build to read
    /// ([`Self::curves_declared_for_older_builds`]).
    ///
    /// A flag rather than a name because a curve is born named — `unique_curve_name` gives it
    /// "Line", then "Line 2" — so a name never said whether anybody had claimed it.
    pub declared_curves: std::collections::BTreeSet<CurveId>,
    /// The curves that decline to be gathered into a profile row.
    ///
    /// **Because a profile row is a reading of the geometry and cannot be deleted.** It is derived by
    /// `fieldform_field::profile_partition`, which keeps the chains that close *by proximity* — so
    /// "delete the row and keep the curves" undoes itself on the next refresh, and Ungroup, which
    /// takes the joint points out, leaves the row standing because the loop still closes without
    /// them. Reported: "you can't delete a profile — IMO you should be able to and the components
    /// need to stay/separate on profile deletion ... we are stuck!"
    ///
    /// This is the stored half that makes the dismissal stick. A curve named here is skipped when the
    /// *rows* are gathered, so its loop is listed as separate curves.
    ///
    /// **It changes no geometry, and that is the line it must not cross.** `profile_loop` — what an
    /// extrude, a revolve and a sweep walk to find their boundary — does not consult this, so a
    /// dismissed loop is still extrudable and a solid built on one goes on being built. The set is
    /// about how the list reads, exactly as [`Self::ownership_curves`] is.
    pub ungrouped_curves: std::collections::BTreeSet<CurveId>,
    /// Axes stood on a body's surface: the curve each one draws, and how its two points are
    /// *derived* from the body (`SurfaceAxis`).
    ///
    /// **Beside the curves, for the reason the relations are.** The curve is an ordinary two-point
    /// line, so everything that takes a line takes it and an older build draws the stored points
    /// identically; this is only the rule that says where those points come from, and that is a
    /// property of how the axis is *edited*, not of its geometry. The application re-derives them
    /// whenever the part changes, so a body that moves takes its axes with it.
    pub surface_axes: std::collections::BTreeMap<CurveId, crate::SurfaceAxis>,
    /// Relations between curves (`profile-group-spec.md` §5): which curve follows which, how,
    /// and whether it was found not to hold on open. Beside the curves rather than on them,
    /// since a curve referencing another curve is the dependency direction this document
    /// refuses; a `BTreeMap` so a file lists them in one order.
    pub relations: std::collections::BTreeMap<crate::id::RelationId, crate::relation::RelationRecord>,
    /// Bodies turned into triangles, in the order they were made — backlog item 53.
    ///
    /// **A list, not a map**, because the block list shows them in that order and a map would
    /// have to be sorted into one on every frame. Each names its part, so deleting a part leaves
    /// its tessellations saying so rather than meshing something else.
    pub tessellations: Vec<crate::Tessellation>,
    /// Files written from tessellations, in the order they were made.
    pub exports: Vec<crate::Export>,
    /// Where each node is drawn in the list: inside the one block that owns it, free, or a
    /// variable — see [`Ownership`] and `block-ownership-spec.md`.
    ///
    /// **Stored, where the list used to read the name and the reader count again every frame**,
    /// so that a rename, an eye or an edit made for another reason can no longer move a row. The
    /// list is drawn from it ([`Self::nesting_in`]), and only [`Self::settle_ownership`] and the
    /// person change it. A map of its own rather than a field on [`NodeMeta`], for the reason
    /// [`Self::arrangement`] gives: `NodeMeta` is `deny_unknown_fields`, so a field there would
    /// fail the load in an older build.
    ///
    /// An id with no entry reads as [`Ownership::Free`] until the next settle places it (spec §3).
    /// Nothing is pruned from it but at save, because an undo can bring an id back and its entry
    /// has to be here when it does.
    pub ownership_nodes: HashMap<NodeId, Ownership>,
    /// [`Self::ownership_nodes`], for curves — whose readers are counted in every part, since a
    /// curve belongs to the document rather than to one part. What curve rows and profile rows
    /// are drawn from, and what their V writes — a profile row's on its loop's first curve, the
    /// key the row is drawn under (`block-ownership-spec.md` §10).
    pub ownership_curves: HashMap<CurveId, Ownership>,
    /// What settling has moved since the application last looked, for it to name and announce
    /// (spec §4.2). Never saved: it is news about the document, not a fact in it.
    pub pending_ownership_changes: Vec<OwnershipChange>,
}

/// Frees the lowest block that cannot reach a top-level row, until every chain of owners ends.
///
/// **The repair `without_rings` used to make, moved to where the ring can now be.** Ownership
/// follows readers, so a ring needs a ring in the graph: nothing in the application builds one,
/// and a file can still say one. It used to be repaired in the *stored* entry, where a chain of
/// `Inside` could close on itself. A stored entry names no parent since 112, so there is nothing
/// left to repair there — but the parent is derived from the readers again, which means a reader
/// ring in a malformed file closes in the walk that *draws* the rows instead. Left alone that is
/// not a mark on the list but a hang, so the guard has to live here.
///
/// Freeing one block breaks the ring it sits on or above and gives it a row of its own, which is
/// the one repair the invariants allow; the next pass finds the next ring, and each pass removes
/// an entry, so this ends. A settle after it adopts the block back and the next reading frees it
/// again, so the list stays the same from frame to frame.
fn without_reader_rings(mut owners: HashMap<NodeId, Vec<NodeId>>) -> HashMap<NodeId, Vec<NodeId>> {
    loop {
        // Grounded: every chain of owners from here is known to end at a top-level row.
        let mut grounded: HashSet<NodeId> = HashSet::new();
        let mut grew = true;
        while grew {
            grew = false;
            for (id, ups) in &owners {
                if grounded.contains(id) {
                    continue;
                }
                if ups.iter().all(|up| !owners.contains_key(up) || grounded.contains(up)) {
                    grounded.insert(*id);
                    grew = true;
                }
            }
        }
        match owners.keys().copied().filter(|id| !grounded.contains(id)).min() {
            Some(lowest) => owners.remove(&lowest),
            None => return owners,
        };
    }
}

/// Which block each block is drawn inside in the list. [`Document::nesting_in`] is the rule.
///
/// **A map from block to owner, never a stored tree.** The same reason `arrangement` is a map:
/// a tree can name a block that no longer exists or leave one out, and a total map from what
/// the graph holds can do neither. The list reads it afresh whenever the document has moved
/// ([`ListReadings`]), from the stored ownership of each node and curve: what is saved is the
/// ownership, and this is one part's reading of it, so there is nothing here to save and nothing
/// to migrate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Nesting {
    nodes: HashMap<NodeId, Vec<NodeId>>,
    curves: HashMap<CurveId, Vec<NodeId>>,
    readers: HashMap<NodeId, usize>,
    curve_readers: HashMap<CurveId, usize>,
    roots: HashSet<NodeId>,
    /// For each block, the inputs it reads that live outside it as variables, in the order it
    /// names them (backlog item 108). See [`Nesting::references_of`].
    references: HashMap<NodeId, Vec<BlockKey>>,
}

impl Nesting {
    /// The inputs `id` reads that are variables — rows of their own elsewhere in the list, which
    /// this block refers to rather than holds.
    ///
    /// **Backlog item 108: a block must account for every input it reads.** The list draws an
    /// input that is `Inside` a block nested in that block's row, and one that is a `Variable` as
    /// a row of its own at the top — and then drew nothing at all where the block reads it. A
    /// sweep with one of its profiles made a variable showed one profile fewer than it takes, and
    /// nothing in the block said which input had gone or where. `block-ownership-spec.md` §2.1
    /// already reads every input as a value, an owned block or a *reference*; this is the third
    /// kind, which was never given a way to be drawn.
    ///
    /// Collected in the walk that decides the nesting, for the reason `readers` is: asked per row
    /// instead, it is a pass over every node per row per frame, and kept here it cannot disagree
    /// with where the rows are drawn. Each input once, however many times the block names it.
    pub fn references_of(&self, id: NodeId) -> &[BlockKey] {
        self.references.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Every block in this graph that reads `key` as a variable, ascending — what a reference
    /// says on hover: *which* blocks share it, the durable form of the sentence settle composes
    /// once in the status line ("Sphere is now a variable — Boolean and Shell both use it").
    /// A scan, because it is asked on hover rather than on every row.
    pub fn readers_of_variable(&self, key: BlockKey) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = self
            .references
            .iter()
            .filter(|(_, keys)| keys.contains(&key))
            .map(|(reader, _)| *reader)
            .collect();
        out.sort();
        out
    }

    /// Every block this one is drawn inside, ascending, or empty for a row of its own.
    ///
    /// **A list, not one owner, since 112.** A nested block is drawn inside *every* block that
    /// reads it and is the same block in each (`notebook-spec.md` R1, §7.3), so a row is no longer
    /// identified by its [`NodeId`] alone — the same block is a row in as many places as read it.
    pub fn owners_of(&self, id: NodeId) -> &[NodeId] {
        self.nodes.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Whether this block is drawn inside anything, which is the question `owner_of` answered.
    pub fn is_nested(&self, id: NodeId) -> bool {
        !self.owners_of(id).is_empty()
    }

    /// Every block a profile row is drawn inside, ascending, or empty for a row of its own.
    pub fn curve_owners_of(&self, id: CurveId) -> &[NodeId] {
        self.curves.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Whether this block is a body: one of [`Graph::roots`], and so always a row of its own.
    ///
    /// **Looked up, where the list asked [`Graph::roots`] of every row.** Each ask walks every
    /// node, so an ask a row made the list quadratic in the part: at 500 bodies, 500 walks of
    /// 0.14 ms each, most of a 90 ms frame (`fieldform-io/examples/list_cost.rs`, dev build). The
    /// set comes out of the count of readers this already makes, since a root is a body kind that
    /// nothing reads.
    pub fn is_root(&self, id: NodeId) -> bool {
        self.roots.contains(&id)
    }

    /// How many blocks read this one, each counted once however many times it names it.
    ///
    /// **Carried here because it is already counted here.** [`Document::nesting_in`] builds this
    /// count to decide what nests, and its own note says why asking per block instead is the wrong
    /// shape: "the list is redrawn every frame, and *who consumes this* asked per block walks every
    /// node once per block — which is the shape that turns a hundred-block part into ten thousand
    /// `inputs()` allocations a frame." The panel needs the same number to say in how many places
    /// a block is drawn — a shared instance is one block in every reader (`notebook-spec.md`
    /// §7.3) — and taking it from the walk that already ran is the difference between one pass and
    /// one pass per row.
    pub fn readers_of(&self, id: NodeId) -> usize {
        self.readers.get(&id).copied().unwrap_or(0)
    }

    /// How many blocks read this curve, each counted once however many times it names it, and
    /// counted in every part, since a curve is the document's (spec I2).
    ///
    /// Asked by a curve's row and a profile's, which a second reader draws in both places like any
    /// other shared instance: the count is how many places that is. It is not a guard — putting a
    /// curve back inside is never refused for the number of readers (`notebook-spec.md` §6.6).
    pub fn readers_of_curve(&self, id: CurveId) -> usize {
        self.curve_readers.get(&id).copied().unwrap_or(0)
    }

    /// Whether a profile row is drawn inside anything — [`Self::curve_owners_of`] asked as the
    /// question the single-owner `owner_of_curve` answered before 112.
    pub fn curve_is_nested(&self, id: CurveId) -> bool {
        !self.curve_owners_of(id).is_empty()
    }
}

/// What the block list reads off one part: where each block is drawn
/// ([`Document::nesting_in`]) and the order the rows come in ([`Document::listed_blocks`]).
///
/// **Read once each time the document moves, not every frame.** The list is drawn every frame,
/// and each of these walks the whole part; the list's order also sorts its branches by a walk of
/// each one. Neither reads anything but the document and the part, so the application keeps one
/// of these against the count of the document's moves and the part on screen, as it keeps the
/// overlay's readings (refactor plan step 3.4).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ListReadings {
    pub nesting: Nesting,
    pub listed: Vec<NodeId>,
}

impl ListReadings {
    /// Both readings of `graph`, a part of `doc`.
    pub fn build(doc: &Document, graph: &Graph) -> Self {
        Self { nesting: doc.nesting_in(graph), listed: doc.listed_blocks(graph) }
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl Document {
    /// An empty document in millimetres.
    pub fn new() -> Self {
        Self {
            unit: Unit::Millimetre,
            cages: HashMap::new(),
            meshes: HashMap::new(),
            quad_meshes: HashMap::new(),
            surface_names: HashMap::new(),
            curves: HashMap::new(),
            parameters: HashMap::new(),
            parts: HashMap::new(),
            node_meta: HashMap::new(),
            sources: HashMap::new(),
            created: HashMap::new(),
            arrangement: HashMap::new(),
            curves_created: HashMap::new(),
            profile_names: HashMap::new(),
            declared_curves: std::collections::BTreeSet::new(),
            ungrouped_curves: std::collections::BTreeSet::new(),
            surface_axes: std::collections::BTreeMap::new(),
            relations: std::collections::BTreeMap::new(),
            tessellations: Vec::new(),
            exports: Vec::new(),
            ownership_nodes: HashMap::new(),
            ownership_curves: HashMap::new(),
            pending_ownership_changes: Vec::new(),
        }
    }

    /// A document holding one part driven by one cage — the state the app opens in.
    pub fn with_cage(cage: Cage) -> Self {
        let mut doc = Self::new();
        let cage_id = doc.add_cage(cage);
        doc.add_part(Part::from_cage("Part", cage_id));
        doc
    }

    /// Whether block `id` draws, as `flag` holds it. A block with no entry has both flags at
    /// their defaults: a body draws and an ingredient does not.
    pub fn visible_by(&self, id: NodeId, flag: EyeFlag) -> bool {
        match self.node_meta.get(&id) {
            Some(meta) => meta.visible_by(flag),
            None => NodeMeta::default().visible_by(flag),
        }
    }

    /// Puts an imported mesh in the document and hands back its id.
    pub fn add_mesh(&mut self, mesh: MeshData) -> MeshId {
        let id = MeshId::new();
        self.meshes.insert(id, mesh);
        id
    }

    pub fn mesh(&self, id: MeshId) -> Option<&MeshData> {
        self.meshes.get(&id)
    }

    /// Adds a surface to follow, under the name its file gave it.
    ///
    /// Not a body — nothing composes it and it has no inside — but a **block**: it appears in
    /// the list and is drawn as reference geometry, exactly as a curve is. A thing in the
    /// document that nothing shows is a thing that arrived invisibly, which is what the first
    /// cut of the import did.
    pub fn add_quad_mesh(
        &mut self,
        mesh: fieldform_cage::QuadMesh,
        name: impl Into<String>,
    ) -> QuadMeshId {
        let id = QuadMeshId::new();
        self.quad_meshes.insert(id, mesh);
        self.surface_names.insert(id, name.into());
        id
    }

    /// What to call a surface. Falls back rather than answering `None`: every list that shows
    /// one needs a word, and "Surface" is a truer one than a hex id.
    pub fn surface_name(&self, id: QuadMeshId) -> &str {
        self.surface_names
            .get(&id)
            .map(String::as_str)
            .unwrap_or("Surface")
    }

    /// Every surface in the document, in a settled order.
    ///
    /// **By id, because a `HashMap` has no order and a list of choices must not reshuffle.** The
    /// ids are drawn at random and never change, so this is arbitrary but fixed -- which is the
    /// property a picker needs and the one D7 asks for.
    pub fn quad_meshes_ordered(&self) -> Vec<QuadMeshId> {
        let mut ids: Vec<QuadMeshId> = self.quad_meshes.keys().copied().collect();
        ids.sort_by_key(|id| id.to_string());
        ids
    }

    /// A graph's bodies in the order both compilers **fold** them, which is the order they were
    /// made.
    ///
    /// [`Graph::roots`] answers *which* nodes are bodies, and that is all it can answer: a graph
    /// is the structure, and when a node was made is not in it. So the graph sorts by raw id
    /// merely to be deterministic within a run, and a reader that needs a chosen order asks the
    /// document.
    ///
    /// # Which of the two orders this is
    ///
    /// **This one is for the compilers, and nothing a person arranges reaches it.** Read by the
    /// value fold and the normal pass in `fieldform-gpu`, by `GraphField`'s own fold, and by the
    /// drawn list and the index the shader highlights — which is the one number that crosses
    /// between the CPU and the shader, and both sides of it read *this* function.
    ///
    /// What a person sees and what their tools act on is [`Self::listed_bodies`], which respects
    /// the order they have arranged the list into. The two came apart because their requirements
    /// are different and because nothing compares a list position with a fold position: a fold
    /// needs to **not move** when an unrelated body is created, and a list needs to be what was
    /// asked for. Keeping them one thing would mean a row dragged two places rebuilt the whole
    /// shader — 1.1 s to 3.8 s cold — for a gesture that changes nothing about the model, which is
    /// the cost `body-order-decision.md` removed. `block-order-spec.md` §4 is that argument.
    ///
    /// **Why this function exists at all.** Ids are random by design (`id.rs`), so ranking bodies
    /// by id ranks them by nothing. A new body then took a rank drawn uniformly among the roots —
    /// last only 1/(n+1) of the time — and `split_into_bodies` names each body function by its
    /// position, so the other 67% to 83% of creations renamed a body that had not changed and the
    /// driver compiled the whole part again, seconds at a time.
    /// `fieldform-gpu/examples/body_order_probe.rs` measures all of it and
    /// `body-order-decision.md` carries the ruling.
    ///
    /// Three things worth knowing about the key:
    ///
    /// - **A stamp is a per-node annotation, not a stored list of bodies.** That is what keeps
    ///   "roots are derived, never stored" satisfied: the failures that rule guards against — a
    ///   node named as the output but consumed by something, a body left out and silently
    ///   invisible — are both failures a *list* can commit and a total map cannot.
    /// - **A shared stamp is worth exactly as much as a random id**, so whatever writes one owes
    ///   a distinct value per node. Measured: bodies sharing a moment reshuffle at the same rate
    ///   as bodies ordered by id.
    /// - **A stand-in sorts where its source sorts.** `NodeId::stand_in` derives a second id for
    ///   the same block rather than minting one, so a stand-in is in no `created` map; it is an
    ///   involution, so asking for the source's stamp is one more lookup. Getting this wrong puts
    ///   every shown ingredient at the top of the drawn list — stable and wrong, which is the
    ///   harder failure to notice.
    ///
    /// An unstamped node reads as zero and sorts first, which is what an older file's nodes are:
    /// older than anything this build has stamped. Ties fall back to the id, so the answer is
    /// always total even when the stamps say nothing.
    ///
    /// Ordered by creation time alone, with no dependency constraint — which it needs none of,
    /// since two roots never constrain each other: a root is by definition unconsumed, so no root
    /// can read another. A body function is emitted per root whatever order they come in.
    pub fn folded_bodies(&self, graph: &Graph) -> Vec<NodeId> {
        let mut bodies = graph.roots();
        bodies.sort_by_key(|id| (self.made_at(*id), id.raw()));
        bodies
    }

    /// Every block in the order the list shows it: **branch by branch, branches in the order the
    /// person arranged them.**
    ///
    /// A post-order walk from the unconsumed nodes rather than a sort, because two things have to
    /// hold at once and neither is a key:
    ///
    /// - **Nothing appears above what it is made from.** `interface-modality.md` fixes this: a
    ///   block above its own operand reads as though it came first. A post-order walk emits a
    ///   node's inputs before the node, so it holds by construction.
    /// - **A body and the blocks only it reads sit together.** They did not, and that was measured
    ///   (`a_bodys_own_blocks_are_not_contiguous_in_the_list`): the walk this replaces took the
    ///   earliest-stamped of everything whose operands were placed, so a cube, an unrelated sphere
    ///   and a shell of the cube listed as `cube, sphere, shell` — an unrelated body inside the
    ///   shell's own branch. Nothing contiguous to drag, which is what `block-order-spec.md` §2.1
    ///   found and what the owner ruled to fix.
    ///
    /// **A block two branches share belongs to the first of them.** It has to stay above both, so
    /// the earliest branch that reads it is where it can go and the later one names it as an input,
    /// which the list already does for every operand. This is the one place a branch is not purely
    /// its own, and it is unavoidable rather than a compromise: "declare it a variable" exists
    /// precisely so a block can be read twice.
    ///
    /// The heads are every unconsumed node, not only the bodies, so a [`Node::Ramp`] nothing reads
    /// still appears — `evaluation_order` says why leaving one out makes it unreachable and reads
    /// as a cycle.
    ///
    /// Returns nothing for a cyclic graph, like the order it is built on: a cycle has no head to
    /// start from, and the honest answer is an empty list rather than a partial one.
    pub fn listed_blocks(&self, graph: &Graph) -> Vec<NodeId> {
        // A cycle is refused here for the reason it is refused there, and asking also gives the
        // per-node rank the tie-breaks below use.
        let Some(topological) = self.evaluation_order_of(graph) else {
            return Vec::new();
        };
        let settled: HashMap<NodeId, usize> =
            topological.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        // Where a person put it, then when it was made, then a position that is already
        // deterministic — so two blocks of one moment cannot swap places between frames.
        //
        // `u32::MAX` for a block nobody has moved, **not `Option`**, which sorts `None` first: a
        // block that has not been arranged goes *after* every block that has, because that is
        // where a newly created one belongs once the list has been rearranged.
        let key = |id: &NodeId| {
            (self.arranged_at(*id).unwrap_or(u32::MAX), self.made_at(*id), settled[id])
        };

        let consumed = graph.consumed();
        let mut heads: Vec<NodeId> =
            graph.nodes().map(|(id, _)| id).filter(|id| !consumed.contains(id)).collect();
        // **A branch goes where it was *started*, not where it was finished.** Keyed on the head's
        // own moment, shelling the first body you made would move that whole branch below
        // everything else -- the shell is the newest block in the document, so its branch would
        // sort last. A person who made a cube, shelled it, then made a sphere thinks of the cube as
        // the first thing they made, and it still is. So a branch sorts by the earliest moment
        // anything in it was made, and the head's own rank only breaks a tie.
        //
        // Where a person has dragged the row still wins outright: that is them saying so about this
        // branch, which is the whole point of the rank.
        let started = |head: &NodeId| {
            graph
                .subtree_of(*head)
                .into_iter()
                .fold(self.made_at(*head), |oldest, id| oldest.min(self.made_at(id)))
        };
        heads.sort_by_key(|id| {
            (self.arranged_at(*id).unwrap_or(u32::MAX), started(id), settled[id])
        });

        let mut out = Vec::with_capacity(topological.len());
        let mut placed: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        for head in heads {
            // Iterative rather than recursive: a graph is reachable from files and scripts, and a
            // deep chain must not take the stack with it. `(id, expanded)` is the same shape
            // `evaluation_order` uses.
            let mut stack: Vec<(NodeId, bool)> = vec![(head, false)];
            while let Some((id, expanded)) = stack.pop() {
                if expanded {
                    // Placed only now, after everything it reads.
                    if placed.insert(id) {
                        out.push(id);
                    }
                    continue;
                }
                if placed.contains(&id) {
                    continue;
                }
                stack.push((id, true));
                let Some(mut inputs) = graph.node(id).map(|n| n.inputs()) else { continue };
                // Operands in the same order the heads are in, so within a branch the list still
                // reads in the order it was made. Reversed because the stack pops backwards.
                inputs.sort_by_key(key);
                inputs.dedup();
                for input in inputs.into_iter().rev() {
                    stack.push((input, false));
                }
            }
        }
        out
    }

    /// The bodies in the order the list shows them — what a person sees, and what "the first body"
    /// means to every tool that acts on one.
    ///
    /// [`Self::listed_blocks`] restricted to the roots, so a body's place among the bodies is its
    /// place among the rows and the two cannot disagree. **Not** [`Self::folded_bodies`], which is
    /// the compilers' order and deliberately ignores how the list has been arranged; that function
    /// says why.
    pub fn listed_bodies(&self, graph: &Graph) -> Vec<NodeId> {
        let roots = graph.roots();
        self.listed_blocks(graph).into_iter().filter(|id| roots.contains(id)).collect()
    }

    /// Which block each block of `graph` is drawn *inside* in the list, and how many blocks read
    /// each — the whole nesting, in one pass.
    ///
    /// **A block is drawn where it is stored** ([`Self::ownership_nodes`],
    /// [`Self::ownership_curves`]): inside every block that reads it when it is `Nested`, and on a
    /// row of its own when it is free, a variable, or not placed by any settle yet (spec §3). A
    /// union of a sphere and a cube holds both of them because the union adopted them when it
    /// first read them — they are what that union *is* — which is the nesting the owner asked
    /// for, "dependent blocks should be nested... the notebook and codeblock layout"
    /// (`block-order-spec.md` §9). An extrude holds the curve it names the same way.
    ///
    /// **Read, not decided here.** This used to decide every row again on every frame, from
    /// whether a block had a name and how many blocks read it — and a rule recomputed from facts
    /// other features also write moved rows whenever any of them was written: a name typed for
    /// another reason lifted a block out, a second reader took it out and its going put it back
    /// (`block-ownership-spec.md` §1). Now only the person (make variable, put back inside, a
    /// drop on the list) and an edit to who reads a block, by [`Self::settle_ownership`]'s fixed
    /// rules, move a row. A name is a label — a profile row's too — and the eye says only whether
    /// a block draws.
    ///
    /// Nothing here reaches [`Self::folded_bodies`], the generated shader or a compile: it decides
    /// which row a row is drawn in, and that is the whole of it. A root has no reader, so a body
    /// is always a top-level row. Ordering is not this function's business —
    /// [`Self::listed_blocks`] already gathers each branch contiguously, so the rows arrive in
    /// nesting order and this only says how deep each one is.
    pub fn nesting_in(&self, graph: &Graph) -> Nesting {
        // **Counted once, not asked per row.** The list is redrawn every frame, and "who
        // consumes this" asked per block walks every node once per block — which is the shape
        // that turns a hundred-block part into ten thousand `inputs()` allocations a frame. The
        // counts are what a row reads to say in how many places the block is drawn.
        let mut consumers: HashMap<NodeId, usize> = HashMap::new();
        // **Who reads each block, not only how many**, since 112: a nested block is drawn inside
        // every one of them (R1), where `Inside(p)` named the single parent this used to count
        // towards. Sorted below, since `graph.nodes()` is a `HashMap` and says its order is
        // unspecified.
        let mut node_readers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        // **Distinct consumers, so one block counts once however many times it names another.**
        // A node can hold the same edge twice -- a boolean given the same body for both operands,
        // a block whose driver is also its input -- and counting it twice would say "two owners"
        // about one.
        let mut seen: Vec<NodeId> = Vec::new();
        for (reader, node) in graph.nodes() {
            seen.clear();
            for input in node.inputs() {
                if seen.contains(&input) {
                    continue;
                }
                seen.push(input);
                *consumers.entry(input).or_default() += 1;
                node_readers.entry(input).or_default().push(reader);
            }
        }
        // **Ascending, which is a promise `owners_of` makes and the walk above cannot keep.** The
        // readers come out in `HashMap` order, so a shared instance would be drawn in one order
        // this frame and another the next — and everything keyed on a row's position would move
        // with it. The curve lists below sort for the same reason.
        for list in node_readers.values_mut() {
            list.sort();
        }
        // **A curve is a dependency too, and `inputs` does not say so.** A profile lives in the
        // document rather than the graph, so `Node::operands` is empty for an extrude, a sweep, a
        // revolve and a pipe; the edge is in [`crate::Node::curves`] and nothing else consults it.
        //
        // **And its readers are counted in every part** (spec I2), as `put_back_target` counts
        // them, because a curve belongs to the document: counted in the part on screen alone, a
        // curve read by a block in another part would read here as read by nothing, when it is
        // drawn inside that block in that part's list. Every reader by its id, so a `graph` that
        // is a copy of a part's counts each once, and the part that `graph` is, when it is one,
        // is walked only once.
        let mut readers: HashMap<CurveId, Vec<NodeId>> = HashMap::new();
        let others = self.parts.values().map(|p| &p.graph).filter(|g| !std::ptr::eq(*g, graph));
        for (id, node) in std::iter::once(graph).chain(others).flat_map(|g| g.nodes()) {
            for curve in node.curves() {
                let list = readers.entry(curve).or_default();
                if !list.contains(&id) {
                    list.push(id);
                }
            }
        }
        // `Graph::roots` from the count above rather than another pass of its own: a block
        // nothing reads has no count, and the read set is the same edges either way.
        let roots = graph
            .nodes()
            .filter(|(id, node)| !consumers.contains_key(id) && node.is_body())
            .map(|(id, _)| id)
            .collect();
        Nesting {
            // A `Nested` block that no block of *this* graph reads is not nested in this list, so
            // it keeps a row of its own here rather than vanishing into nothing; settling never
            // leaves one (I1), and a file can still say one.
            nodes: without_reader_rings(
                graph
                    .nodes()
                    .filter_map(|(id, _)| match self.ownership(BlockKey::Node(id)) {
                        Ownership::Nested => {
                            let owners = node_readers.get(&id).cloned().unwrap_or_default();
                            (!owners.is_empty()).then_some((id, owners))
                        }
                        _ => None,
                    })
                    .collect(),
            ),
            // **Where it is stored, as a node is**, and for the reason a curve needed a handle at
            // all: an extrude names one curve of its loop, so when the loop opened the one it
            // named stayed inside it with no way out (backlog item 73). The curve row's V makes it
            // a variable and puts it back like a block's; a flag of its own (`declared_curves`)
            // and a profile row's name did that until the ownership was stored, and the name moved
            // the row whenever it was typed for any other reason. A profile row is drawn inside
            // whatever holds a curve of its loop, which `nest_rows` asks of this.
            curves: self
                .curves
                .keys()
                .filter_map(|id| match self.ownership(BlockKey::Curve(*id)) {
                    Ownership::Nested => {
                        // Only the readers in *this* part draw a row here, though the count below
                        // is every part's (I2): a curve two parts read is drawn inside the reader
                        // in each of their lists, which is what a shared instance across parts is.
                        let owners: Vec<NodeId> = readers
                            .get(id)
                            .map(|list| {
                                let mut here: Vec<NodeId> = list
                                    .iter()
                                    .copied()
                                    .filter(|r| graph.node(*r).is_some())
                                    .collect();
                                here.sort();
                                here
                            })
                            .unwrap_or_default();
                        (!owners.is_empty()).then_some((*id, owners))
                    }
                    _ => None,
                })
                .collect(),
            // Every block's count, not only the ones that nest: the panel asks it of variables
            // too, which are exactly the ones left out of `nodes` above.
            readers: consumers,
            // The same for curves, which a profile row needs for the same reason a block row
            // does: whether putting it back has one block to go into.
            curve_readers: readers.iter().map(|(id, list)| (*id, list.len())).collect(),
            roots,
            // The third kind of input, beside a value and an owned block (item 108): what each
            // block of this graph reads that lives outside it as a variable — graph inputs first,
            // then the curves a profile, a spine or a section is read from, each once.
            references: graph
                .nodes()
                .filter_map(|(id, node)| {
                    let mut keys: Vec<BlockKey> = Vec::new();
                    let inputs = node.inputs().into_iter().map(BlockKey::Node);
                    let curves = node.curves().into_iter().map(BlockKey::Curve);
                    for key in inputs.chain(curves) {
                        if !keys.contains(&key) && self.ownership(key) == Ownership::Variable {
                            keys.push(key);
                        }
                    }
                    (!keys.is_empty()).then_some((id, keys))
                })
                .collect(),
        }
    }

    /// Where a person has put a block in the list, if they have moved it.
    ///
    /// `None` — read here as "after everything they have arranged" — means they have not, so it
    /// falls back to when it was made. A body created after a reorder therefore goes to the
    /// bottom, which is where a new body goes.
    pub fn arranged_at(&self, id: NodeId) -> Option<u32> {
        self.arrangement.get(&id).copied()
    }

    /// [`Graph::evaluation_order`], seeded in the order the nodes were made.
    ///
    /// **What keeps a generated program stable, together with [`Self::folded_bodies`].** The fold order
    /// decides the body functions' *names*; this decides their *contents* — the `n{k}` slots and
    /// the `params[..]` indices, which are handed out as the walk visits nodes. Both have to be
    /// settled by the same thing or a creation still rewrites bodies it did not touch, which is
    /// the failure `a_body_already_in_the_part_keeps_its_function_when_another_is_added` catches.
    ///
    /// Every compile path has a document, so every compile path should call this rather than the
    /// graph's own bare version.
    pub fn evaluation_order_of(&self, graph: &Graph) -> Option<Vec<NodeId>> {
        graph.evaluation_order_from(&|id| self.made_at(id))
    }

    /// When a node was made, following a stand-in back to the block it stands for.
    ///
    /// Zero for a node nothing stamped, which sorts it first — see [`Self::folded_bodies`].
    pub fn made_at(&self, id: NodeId) -> u64 {
        self.created
            .get(&id)
            .or_else(|| self.created.get(&id.stand_in()))
            .copied()
            .unwrap_or(0)
    }

    pub fn add_cage(&mut self, cage: Cage) -> CageId {
        let id = CageId::new();
        self.cages.insert(id, cage);
        id
    }

    pub fn add_part(&mut self, part: Part) -> PartId {
        let id = part.id;
        self.parts.insert(id, part);
        id
    }

    pub fn add_parameter(&mut self, name: impl Into<String>, value: f64) -> ParamId {
        let id = ParamId::new();
        self.parameters.insert(
            id,
            Parameter {
                id,
                name: name.into(),
                value: ParamValue::Literal(value),
            },
        );
        id
    }

    /// The parameter called `name`, if there is one.
    ///
    /// Names are unique through [`Self::rename_parameter`], but a file can say otherwise; two
    /// with one name resolve to the lower id, so the answer is at least the same every time.
    pub fn parameter_named(&self, name: &str) -> Option<ParamId> {
        self.parameters.values().filter(|p| p.name == name).map(|p| p.id).min()
    }

    /// Whether `name` can be a parameter's name: something in it, no space at either end, and
    /// no square bracket, since brackets are how an expression says a name that is not an
    /// identifier (`[cap-1] * 2`, see `expression::name_in_expression`). A dash, a space or a
    /// leading digit make a name one that is said in brackets, not one that is refused: it is
    /// a label, not arithmetic. Asked for as "cap-1" by the owner, who wanted to number the
    /// values relating a pattern's points.
    pub fn is_parameter_name(name: &str) -> bool {
        let trimmed = name.trim();
        !trimmed.is_empty() && trimmed == name && !name.contains(['[', ']'])
    }

    /// Parses `text` against this document's parameter names, storing nothing.
    pub(crate) fn parse_expression(&self, text: &str) -> Result<Expression, ParseError> {
        Expression::parse(text, &|name| self.parameter_named(name))
    }

    /// Makes parameter `id` compute `text`.
    ///
    /// **A cycle is refused here, on the edit, not found at evaluation.** `a = b + 1` with
    /// `b = a + 1` is rejected when the second is typed, naming the loop, rather than stored
    /// and discovered when the document is next opened — a stored cycle is a document that
    /// cannot be evaluated, which is worse than a rejected keystroke. The parameters form
    /// their own dependency graph, separate from the part graphs, so `Graph::evaluation_order`
    /// never sees this and could not catch it.
    ///
    /// Nothing changes on an error.
    pub fn set_parameter_expression(
        &mut self,
        id: ParamId,
        text: &str,
    ) -> Result<(), ExpressionError> {
        if !self.parameters.contains_key(&id) {
            return Err(ExpressionError::NoSuchParameter(id));
        }
        let expression = self.parse_expression(text)?;
        if let Some(cycle) = self.loop_closed_by(id, &expression.references()) {
            return Err(ExpressionError::Cycle(
                cycle.into_iter().map(|p| self.parameter_name(p)).collect(),
            ));
        }
        self.parameters
            .get_mut(&id)
            .expect("checked above")
            .value = ParamValue::Expression(expression);
        Ok(())
    }

    /// Makes parameter `id` a plain number again. `false` if there is no such parameter.
    pub fn set_parameter_literal(&mut self, id: ParamId, value: f64) -> bool {
        match self.parameters.get_mut(&id) {
            Some(p) => {
                p.value = ParamValue::Literal(value);
                true
            }
            None => false,
        }
    }

    /// The loop that giving `id` an expression reading `reads` would close, if any, as the
    /// ids along it with `id` at both ends.
    fn loop_closed_by(&self, id: ParamId, reads: &[ParamId]) -> Option<Vec<ParamId>> {
        for &first in reads {
            let mut path = vec![id, first];
            let mut visited = HashSet::new();
            if self.reaches(first, id, &mut path, &mut visited) {
                return Some(path);
            }
        }
        None
    }

    /// Whether `target` is reachable from `from` through what the stored expressions read,
    /// extending `path` with the route when it is. Each parameter is entered once, which is
    /// what bounds the walk whatever the file says.
    fn reaches(
        &self,
        from: ParamId,
        target: ParamId,
        path: &mut Vec<ParamId>,
        visited: &mut HashSet<ParamId>,
    ) -> bool {
        if from == target {
            return true;
        }
        if !visited.insert(from) {
            return false;
        }
        for next in self.parameter_reads(from) {
            path.push(next);
            if self.reaches(next, target, path, visited) {
                return true;
            }
            path.pop();
        }
        false
    }

    /// The parameters the expression of `id` reads; none for a literal.
    fn parameter_reads(&self, id: ParamId) -> Vec<ParamId> {
        self.parameters
            .get(&id)
            .and_then(|p| p.value.expression())
            .map(|e| e.references())
            .unwrap_or_default()
    }

    /// Every parameter whose expression reads `id`, in a stable order.
    ///
    /// The other half of what a node's `param_refs` answers: removing a parameter has to be
    /// refused while anything reads it, and an expression is a reader as much as a radius is.
    pub fn parameters_reading(&self, id: ParamId) -> Vec<ParamId> {
        let mut out: Vec<ParamId> = self
            .parameters
            .values()
            .filter(|p| self.parameter_reads(p.id).contains(&id))
            .map(|p| p.id)
            .collect();
        out.sort();
        out
    }

    /// Whether any node in any part, or any other parameter's expression, reads parameter `id`
    /// -- with one part read as the graph an edit is about to give it, when `pending` names one.
    ///
    /// Every part, not only the active one: a parameter is a document-wide name, so a second
    /// part driving its pipe off it is exactly the case that must stop it being removed. An
    /// expression is a reader the same way: `boss_r = radius * 5 - 2` keeps `radius`. Here
    /// rather than in the application so that the × asking whether a parameter may go and an
    /// undo taking one out ask the same question.
    pub fn parameter_is_read(&self, id: ParamId, pending: Option<(PartId, &Graph)>) -> bool {
        let reads = |graph: &Graph| {
            graph
                .nodes()
                .any(|(_, node)| node.param_refs().iter().any(|r| **r == ParamRef::Named(id)))
        };
        !self.parameters_reading(id).is_empty()
            || self.parts.iter().any(|(&p, part)| match pending {
                Some((edited, graph)) if edited == p => reads(graph),
                _ => reads(&part.graph),
            })
    }

    /// Writes `parameter` into its slot the way an undo or a redo puts one back: under a name
    /// no other parameter has, and with every expression it touches -- its own, and those
    /// reading it -- saying the names the parameters have now.
    ///
    /// **What a parameter left under is not always what it can come back as.** Names are edited
    /// off the history, so while it was out another parameter may have taken its name, and two
    /// with one name is a document [`Self::rename_parameter`] refuses to make and a file that
    /// will not open. A taken name becomes the first free `_2`, `_3` after it, which is what
    /// naming a second value the same gives. And a rename re-renders only the expressions in
    /// the document at the time, so one that was out while a parameter it reads was renamed
    /// comes back saying the old name: it still reads correctly, by id, until the file parses
    /// the text again by name. An expression whose text no longer says what it reads is
    /// rendered again; the rest keep the text they were typed as.
    pub fn put_parameter(&mut self, mut parameter: Parameter) {
        let id = parameter.id;
        let taken = |name: &str| self.parameters.values().any(|p| p.id != id && p.name == name);
        if taken(&parameter.name) {
            let base = parameter.name.clone();
            parameter.name = (2..)
                .map(|n| format!("{base}_{n}"))
                .find(|name| !taken(name))
                .expect("always terminates");
        }
        self.parameters.insert(id, parameter);
        let mut touched = self.parameters_reading(id);
        touched.push(id);
        let stale: Vec<ParamId> = touched
            .into_iter()
            .filter(|p| {
                self.parameters[p].value.expression().is_some_and(|e| {
                    !matches!(self.parse_expression(e.source()), Ok(said) if said == *e)
                })
            })
            .collect();
        let names: HashMap<ParamId, String> =
            self.parameters.values().map(|p| (p.id, p.name.clone())).collect();
        for p in stale {
            if let Some(ParamValue::Expression(e)) =
                self.parameters.get_mut(&p).map(|p| &mut p.value)
            {
                e.rerender(&|read| names.get(&read).cloned().unwrap_or_else(|| read.to_string()));
            }
        }
    }

    /// The parameters in an order that puts what is read before what reads it, or the loop
    /// that makes no such order possible, with the same parameter at both ends.
    ///
    /// The application never stores a cycle, so `Err` means a file said so; the reader
    /// refuses the file with the loop named rather than opening a document that cannot
    /// evaluate.
    pub fn parameter_order(&self) -> Result<Vec<ParamId>, Vec<ParamId>> {
        let mut ids: Vec<ParamId> = self.parameters.keys().copied().collect();
        ids.sort();
        let mut state: HashMap<ParamId, u8> = HashMap::new(); // 1 in progress, 2 done
        let mut order = Vec::with_capacity(ids.len());
        let mut stack = Vec::new();
        for id in ids {
            self.visit_in_order(id, &mut state, &mut order, &mut stack)?;
        }
        Ok(order)
    }

    fn visit_in_order(
        &self,
        id: ParamId,
        state: &mut HashMap<ParamId, u8>,
        order: &mut Vec<ParamId>,
        stack: &mut Vec<ParamId>,
    ) -> Result<(), Vec<ParamId>> {
        match state.get(&id) {
            Some(2) => return Ok(()),
            Some(1) => {
                // Back on a parameter still being entered: the loop is the stack from its
                // first appearance, closed by it.
                let from = stack.iter().position(|p| *p == id).unwrap_or(0);
                let mut cycle = stack[from..].to_vec();
                cycle.push(id);
                return Err(cycle);
            }
            _ => {}
        }
        state.insert(id, 1);
        stack.push(id);
        for next in self.parameter_reads(id) {
            if self.parameters.contains_key(&next) {
                self.visit_in_order(next, state, order, stack)?;
            }
        }
        stack.pop();
        state.insert(id, 2);
        order.push(id);
        Ok(())
    }

    /// What `id` is worth, or why it is worth nothing.
    ///
    /// A literal is its number. An expression is evaluated through the parameters it reads,
    /// each of which is evaluated the same way — a walk rather than a table, because a
    /// document has tens of parameters and a lookup that recomputes a chain of three costs
    /// nothing anyone can measure. The depth guard is what stops a hand-built cycle from
    /// recursing forever; see [`MAX_PARAMETER_DEPTH`].
    pub fn evaluate_parameter(&self, id: ParamId) -> Result<f64, ParameterError> {
        self.evaluate_parameter_at(id, 0)
    }

    fn evaluate_parameter_at(&self, id: ParamId, depth: usize) -> Result<f64, ParameterError> {
        if depth > MAX_PARAMETER_DEPTH {
            return Err(ParameterError::TooDeep(MAX_PARAMETER_DEPTH));
        }
        let Some(parameter) = self.parameters.get(&id) else {
            return Err(ParameterError::Missing);
        };
        match &parameter.value {
            ParamValue::Literal(v) => Ok(*v),
            ParamValue::Expression(expression) => expression
                .evaluate(&|read| self.evaluate_parameter_at(read, depth + 1).ok())
                .map_err(|err| match err {
                    EvalError::NotFinite(step) => ParameterError::NotFinite(step),
                    EvalError::Unresolved(read) => match self.parameters.get(&read) {
                        Some(p) => ParameterError::Unresolved(p.name.clone()),
                        None => ParameterError::Missing,
                    },
                }),
        }
    }

    /// The number `id` is worth, or `None` if it is worth nothing — a failed expression, a
    /// missing parameter. **This is what every `ParamRef::resolve` lookup reads.** `None` is
    /// what an unresolvable name already gave, and what every caller already refuses, which is
    /// why a failed expression goes the same way rather than reaching a radius as NaN.
    pub fn parameter_value(&self, id: ParamId) -> Option<f64> {
        self.evaluate_parameter(id).ok()
    }

    /// Every parameter's value, evaluated now, by [`Self::parameter_value`]: the table a reader
    /// carries where it cannot carry the document, as the viewport's lookup does once a frame.
    ///
    /// A failed expression is simply absent, which is how a lookup reports it, and the same as an
    /// unknown name.
    pub fn parameter_values(&self) -> HashMap<ParamId, f64> {
        self.parameters
            .keys()
            .filter_map(|id| self.parameter_value(*id).map(|v| (*id, v)))
            .collect()
    }

    /// What `r` is worth in this document: a literal is its number, a name is asked of
    /// [`Document::parameter_value`], and a reference driven by a field has no single value.
    ///
    /// **One spelling for the lookup the field and the shader generator wrote out at every
    /// site** as `r.resolve(&|id| document.parameter_value(id))`. It is exactly that and no
    /// more: what a `None` becomes — a refusal, a zero, the weaker answer — stays each caller's
    /// decision, because the callers do not agree and are not meant to.
    pub fn resolve(&self, r: &ParamRef) -> Option<f64> {
        r.resolve(&|id| self.parameter_value(id))
    }

    /// Renames a parameter, and re-renders every expression that read it by the old name.
    ///
    /// Refuses a name an expression could not say, and one another parameter already has:
    /// two parameters with one name is a document in which `wall * 2` means nothing definite.
    pub fn rename_parameter(&mut self, id: ParamId, name: &str) -> Result<(), RenameError> {
        if !self.parameters.contains_key(&id) {
            return Err(RenameError::NoSuchParameter(id));
        }
        if !Self::is_parameter_name(name) {
            return Err(RenameError::NotAName(name.to_string()));
        }
        if self.parameters.values().any(|p| p.id != id && p.name == name) {
            return Err(RenameError::Taken(name.to_string()));
        }
        self.parameters.get_mut(&id).expect("checked above").name = name.to_string();
        // Only the expressions that read this one: the rest keep the text they were typed as.
        let names: HashMap<ParamId, String> =
            self.parameters.values().map(|p| (p.id, p.name.clone())).collect();
        for reader in self.parameters_reading(id) {
            if let Some(ParamValue::Expression(e)) =
                self.parameters.get_mut(&reader).map(|p| &mut p.value)
            {
                e.rerender(&|read| names.get(&read).cloned().unwrap_or_else(|| read.to_string()));
            }
        }
        Ok(())
    }

    /// A parameter's name, or its id written out when there is no such parameter — for a
    /// message, never for resolution.
    fn parameter_name(&self, id: ParamId) -> String {
        self.parameters.get(&id).map(|p| p.name.clone()).unwrap_or_else(|| id.to_string())
    }

    pub fn cage(&self, id: CageId) -> Option<&Cage> {
        self.cages.get(&id)
    }

    pub fn cage_mut(&mut self, id: CageId) -> Option<&mut Cage> {
        self.cages.get_mut(&id)
    }

    pub fn curve(&self, id: CurveId) -> Option<&Curve> {
        self.curves.get(&id)
    }

    pub fn curve_mut(&mut self, id: CurveId) -> Option<&mut Curve> {
        self.curves.get_mut(&id)
    }

    /// Every cage a part's graph reads.
    ///
    /// Walks the graph rather than assuming one cage per part, so it keeps working once the
    /// graph has more than one leaf.
    ///
    /// Written for the app to know which cages to draw and which to re-tessellate, but the app
    /// asks the graph itself, through `App::part_graph` and [`Graph::referenced_cages`]: the
    /// document is the composition, and the app reads it rather than a lookup beside it. So
    /// only the tests ask by part, and only the tests compile this.
    #[cfg(test)]
    fn cages_used_by(&self, part: PartId) -> Vec<CageId> {
        self.parts
            .get(&part)
            .map(|p| p.graph.referenced_cages())
            .unwrap_or_default()
    }

    /// A stable, deterministic ordering of parts.
    ///
    /// Random ids give no natural order, and `HashMap` iteration order varies run to run —
    /// which would make the UI list, and anything derived from it, jitter. Sorting by name
    /// then id gives a fixed order without reintroducing sequential ids.
    pub fn parts_ordered(&self) -> Vec<&Part> {
        let mut parts: Vec<&Part> = self.parts.values().collect();
        parts.sort_by(|a, b| a.name.cmp(&b.name).then(a.id.raw().cmp(&b.id.raw())));
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A body and the blocks only it reads are next to each other in the list.**
    ///
    /// The property the branch walk exists for, and the one the walk it replaced did not have. That
    /// one took the earliest-stamped of everything whose operands were placed, so an unrelated body
    /// made between a block and the block built from it sat between them: a cube, a sphere, and a
    /// shell of the cube listed as `cube, sphere, shell`, with nothing contiguous to drag.
    /// Measured before it was changed, which is what `block-order-spec.md` §2.1 rests on.
    ///
    /// Three blocks: a cube, then a sphere, then a shell of the cube. The shell depends on the
    /// cube and the sphere depends on nothing, so dependency permits either arrangement — which
    /// is exactly why this needs asserting rather than following from the constraint.
    #[test]
    fn a_bodys_own_blocks_are_contiguous_in_the_list() {
        use crate::{Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: crate::no_turn(),
        });
        let sphere = graph.add(Node::Sphere {
            centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
            radius: lit(9.0),
        });
        let shell = graph.add(Node::Shell { input: cube.into(), thickness: lit(2.0) });
        for (at, id) in [(1u64, cube), (2, sphere), (3, shell)] {
            doc.created.insert(id, 1_700_000_000_000_000 + at);
        }

        // The bodies are the shell and the sphere; the cube is the shell's own ingredient.
        assert_eq!(doc.folded_bodies(&graph), vec![sphere, shell], "two bodies, oldest first");
        // The shell's branch comes first because the block it starts from was made first, and the
        // sphere is not inside it.
        assert_eq!(
            doc.listed_blocks(&graph),
            vec![cube, shell, sphere],
            "the shell sits directly under the cube it is made from",
        );
        let list = doc.listed_blocks(&graph);
        let at = |id| list.iter().position(|x| *x == id).expect("in the list");
        assert_eq!(at(shell), at(cube) + 1, "nothing comes between a body and its own block");
        assert!(at(cube) < at(shell), "and nothing appears above what it is made from");
        // And the bodies' own order, which is what a tool acting on "the first body" reads.
        assert_eq!(doc.listed_bodies(&graph), vec![shell, sphere]);
    }

    /// **A block nobody named, that one thing consumes, is drawn inside that thing.**
    ///
    /// The nesting the owner asked for, in its own example: a union of a sphere and a cube that
    /// have not been declared variables holds both of them, since the union adopted both when it
    /// first read them. `block-order-spec.md` §9, `block-ownership-spec.md` §4.
    #[test]
    fn an_unnamed_block_with_one_consumer_nests_inside_it() {
        use crate::{BooleanOp, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: crate::no_turn(),
        });
        let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        for (at, id) in [(1u64, sphere), (2, cube), (3, union)] {
            doc.created.insert(id, 1_700_000_000_000_000 + at);
        }
        let graph = settled_in(&mut doc, graph);

        let nesting = doc.nesting_in(&graph);
        assert_eq!(
            nesting.owners_of(sphere).to_vec(),
            vec![union],
            "the sphere is part of the union",
        );
        assert_eq!(nesting.owners_of(cube).to_vec(), vec![union]);
        assert!(!nesting.is_nested(union), "and the union is a body, so it is a row");
        // The rows already arrive in nesting order, which is what makes the depth the only
        // thing this adds: both ingredients come before the block that holds them.
        assert_eq!(doc.listed_blocks(&graph), vec![sphere, cube, union]);
    }

    /// `graph` as the one part of `doc`, settled as every edit leaves a document, so each block
    /// sits where its readers put it; and the graph back, for `nesting_in` to be asked of.
    fn settled_in(doc: &mut Document, graph: Graph) -> Graph {
        doc.add_part(Part {
            id: PartId::new(),
            name: "Part".into(),
            material: None,
            graph: graph.clone(),
            inputs: Vec::new(),
        });
        doc.settle_ownership();
        doc.pending_ownership_changes.clear();
        graph
    }

    /// **The list is drawn from where each block is stored, and a name moves no row.**
    ///
    /// The rows used to be read off names and reader counts every frame, so a name typed for any
    /// reason lifted a block out of the block it belonged to — the mechanism behind every row
    /// that moved on its own (`block-ownership-spec.md` §1). Read from the stored ownership
    /// instead: a block named inside its union stays inside it, a variable has a row of its own
    /// whether or not it has a name, and a block no settle has placed yet reads as free (§3) —
    /// a row of its own until the next settle puts it somewhere. The graph is never rewired by
    /// any of it, and the counts the V reads are the graph's, not the ownership's.
    #[test]
    fn the_list_is_drawn_from_the_stored_ownership_and_a_name_moves_no_row() {
        use crate::{BooleanOp, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: crate::no_turn(),
        });
        let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        let graph = settled_in(&mut doc, graph);
        assert_eq!(
            doc.nesting_in(&graph).owners_of(sphere).to_vec(),
            vec![union],
            "adopted by its reader",
        );

        doc.node_meta.entry(sphere).or_default().name = Some("hub".into());
        assert_eq!(
            doc.nesting_in(&graph).owners_of(sphere).to_vec(),
            vec![union],
            "named, and still the union's: a name is a label",
        );

        doc.ownership_nodes.insert(cube, Ownership::Variable);
        let nesting = doc.nesting_in(&graph);
        assert!(!nesting.is_nested(cube), "a variable has a row of its own, named or not");
        assert_eq!(nesting.readers_of(cube), 1, "and the union still reads it");
        assert_eq!(
            nesting.owners_of(sphere).to_vec(),
            vec![union],
            "the one beside it has not moved",
        );

        doc.ownership_nodes.remove(&sphere);
        assert!(
            !doc.nesting_in(&graph).is_nested(sphere),
            "a block no settle has placed reads as free",
        );
        // The graph is exactly what it was: where a row is drawn is not an edge.
        assert_eq!(graph.node(union).expect("the union").inputs(), vec![sphere, cube]);
        assert_eq!(graph.roots(), vec![union], "the sphere is still nobody's body");
    }

    /// **A block two others read is drawn inside both of them — one block, two places.**
    ///
    /// A shared instance (`notebook-spec.md` R1, §7.3): nesting no longer says ownership, so two
    /// readers are not a contradiction to be resolved but two places the same block is drawn in,
    /// and an edit in either is an edit everywhere. This asserted the opposite until 112, on the
    /// reasoning that two owners is not ownership and the block had to stay above both consumers —
    /// which was `interface-modality.md`'s objection to indenting a boolean's operands, and held
    /// only while a nested block named one parent. None is stored now (spec §12), so there is no
    /// hierarchy to invent: who reads the block is the graph's to answer, and it answers both.
    /// A second reader declares nothing either (R5) — the settle that promoted such a block to a
    /// variable is what 112 reversed.
    #[test]
    fn a_block_two_others_read_is_drawn_inside_both_of_them() {
        use crate::{BooleanOp, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let shared =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: crate::no_turn(),
        });
        let cut =
            graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), shared.into()] });
        let second = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![cut.into(), shared.into()],
        });
        let graph = settled_in(&mut doc, graph);

        let nesting = doc.nesting_in(&graph);
        // Ids are random, so the ascending pair is the one to compare against rather than the
        // order the two were made in.
        let mut both_cuts = vec![cut, second];
        both_cuts.sort();
        assert_eq!(
            nesting.owners_of(shared).to_vec(),
            both_cuts,
            "drawn inside each reader, ascending, and the same block in both",
        );
        assert_eq!(nesting.readers_of(shared), 2, "two readers, each counted once");
        assert_eq!(
            doc.ownership(BlockKey::Node(shared)),
            Ownership::Nested,
            "and nested is all that is stored: no parent, so nothing says which of the two",
        );
        assert_eq!(
            nesting.owners_of(block).to_vec(),
            vec![cut],
            "and the one-consumer block still nests",
        );
        assert_eq!(nesting.owners_of(cut).to_vec(), vec![second]);
    }

    /// **The eye does not declare a variable; only making it one does.**
    ///
    /// This asserted the opposite until 2026-09-30, when the owner hit it: clicking the eye on a
    /// sphere inside a boolean lifted the sphere out, which reads as the eye having made it a
    /// variable. Two controls, two questions — reusable, and visible — and only the first moves a
    /// row. That first control was a name until the list was drawn from the stored ownership; it
    /// is the V now, which writes [`Ownership::Variable`].
    ///
    /// **What the old rule was guarding, and what answers it now.** `Pattern` and `Mirror` leave
    /// the body they consumed shown, because it was on screen a moment before; under the old rule
    /// that body kept a row, so the eye that stops it drawing was always reachable. Now it nests,
    /// and inside a *collapsed* block that eye needs the block opened first. Accepted deliberately:
    /// a row that moves when you ask to see something is the worse surprise, the control is still
    /// on the row rather than gone, and the same expansion already reaches every other thing a
    /// block is made of.
    #[test]
    fn the_eye_does_not_lift_a_block_out_of_its_consumer() {
        use crate::{Fillet, Node, ParamRef, PatternKind, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let pattern = graph.add(Node::Pattern {
            options: Default::default(),
            input: sphere.into(),
            count: lit(4.0),
            kind: PatternKind::Linear { step: [lit(30.0), lit(0.0), lit(0.0)] },
            join: false,
            fillet: Fillet::hard(),
        });
        let graph = settled_in(&mut doc, graph);

        assert_eq!(doc.nesting_in(&graph).owners_of(sphere).to_vec(), vec![pattern]);
        doc.node_meta.entry(sphere).or_default().shown = true;
        assert_eq!(
            doc.nesting_in(&graph).owners_of(sphere).to_vec(),
            vec![pattern],
            "the eye says it can be looked at, not that it is a variable",
        );
        // And the control that *does* lift it out still does.
        doc.ownership_nodes.insert(sphere, Ownership::Variable);
        assert!(!doc.nesting_in(&graph).is_nested(sphere), "a variable has a row of its own");
    }

    /// **A block names the variables it reads, so its row can show them** (backlog item 108).
    ///
    /// The owner's report: make one of a sweep's profiles a variable and the sweep no longer shows
    /// it. This is that case in its smallest form — a union of a sphere it owns and a cube that is
    /// a variable — plus the curve half, since a sweep reads its profile as a curve, not a node.
    #[test]
    fn a_block_names_the_variables_it_reads() {
        use crate::ownership::BlockKey;
        use crate::{BooleanOp, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let profile = a_line(&mut doc, "Line");
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: crate::no_turn(),
        });
        let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        let extrude = graph.add(extrude_of(profile));
        let graph = settled_in(&mut doc, graph);
        assert!(
            doc.nesting_in(&graph).references_of(union).is_empty(),
            "both inputs are inside the union, so it refers to nothing outside itself",
        );

        doc.ownership_nodes.insert(cube, Ownership::Variable);
        doc.ownership_curves.insert(profile, Ownership::Variable);
        let nesting = doc.nesting_in(&graph);
        assert_eq!(
            nesting.references_of(union),
            &[BlockKey::Node(cube)],
            "the cube went to a row of its own; the union still reads it and now says so",
        );
        assert_eq!(nesting.owners_of(sphere), &[union], "the sphere it owns is not a reference");
        assert_eq!(nesting.references_of(extrude), &[BlockKey::Curve(profile)]);
        assert_eq!(nesting.references_of(sphere), &[] as &[BlockKey], "reads nothing at all");

        // Who shares it, for the hover: here one reader each.
        assert_eq!(nesting.readers_of_variable(BlockKey::Node(cube)), vec![union]);
        assert_eq!(nesting.readers_of_variable(BlockKey::Curve(profile)), vec![extrude]);

        // Put back inside, it is owned again and not a reference.
        doc.ownership_nodes.insert(cube, Ownership::Nested);
        assert!(doc.nesting_in(&graph).references_of(union).is_empty());
    }

    /// A two-point line, put in `doc`: a curve for a block to read.
    fn a_line(doc: &mut Document, name: &str) -> CurveId {
        let line = Curve::polyline(
            CurveId::new(),
            name,
            vec![nalgebra::Point3::origin(), nalgebra::Point3::new(10.0, 0.0, 0.0)],
            false,
        )
        .expect("a two-point line");
        let id = line.id;
        doc.curves.insert(id, line);
        id
    }

    /// An extrude of the loop `profile` closes.
    fn extrude_of(profile: CurveId) -> crate::Node {
        let lit = ParamRef::Literal;
        crate::Node::Extrude {
            outline: crate::Outline::closed(profile),
            sweep: crate::Sweep::Normal,
            height: lit(10.0),
            start: lit(0.0),
            sides: crate::Sides::One,
            draft: lit(0.0),
            corners: Default::default(),
            rims: Default::default(),
        }
    }

    /// `graph` put in place of the one part's, and settled, as an edit through the history leaves
    /// the document.
    fn edited(doc: &mut Document, graph: &Graph) {
        doc.parts.values_mut().next().expect("one part").graph = graph.clone();
        doc.settle_ownership();
        doc.pending_ownership_changes.clear();
    }

    /// **An extrude holds the profile it was drawn from, and a profile's name is a label.**
    ///
    /// The commonest nesting of all, and the one the graph does not carry as an edge: a profile
    /// is a curve, which lives in the document, so `Node::operands` is empty for an extrude and
    /// the dependency is in `Node::curves`. A nesting built from `inputs` alone leaves every
    /// sketch flat. The extrude adopted the curve when it first read it, as a union adopts a
    /// sphere, and the list reads where the curve is stored.
    ///
    /// A profile row's name used to be its declaration, so naming the row pulled the loop out of
    /// the extrude made from it, which is the defect a block's name had (`block-ownership-spec.md`
    /// §1, §10). It is a label now, keyed as before by the loop's first curve.
    #[test]
    fn an_extrude_holds_the_profile_it_was_drawn_from_and_a_profile_s_name_moves_no_row() {
        let mut doc = Document::new();
        let profile = a_line(&mut doc, "Line");
        let mut graph = Graph::empty();
        let extrude = graph.add(extrude_of(profile));
        let mut graph = settled_in(&mut doc, graph);
        assert_eq!(doc.nesting_in(&graph).curve_owners_of(profile).to_vec(), vec![extrude]);
        assert_eq!(
            doc.nesting_in(&graph).readers_of_curve(profile),
            1,
            "one reader, so the row is drawn in the one place",
        );

        doc.profile_names.insert(profile, "Flange".into());
        assert_eq!(
            doc.nesting_in(&graph).curve_owners_of(profile).to_vec(),
            vec![extrude],
            "named, and still the extrude's",
        );

        // A second reader draws the same loop in both places and declares nothing (R5, §7.3):
        // one curve, a row inside the extrude and a row inside the revolve, with the count
        // saying how many places that is. Promoting it to a variable here is what 112 reversed.
        let revolve = graph.add(crate::Node::Revolve { rims: Default::default(),
            outline: crate::Outline::closed(profile),
            frame: crate::Frame::at_along(crate::Axis::Z, [ParamRef::Literal(0.0); 3]),
            angle: ParamRef::Literal(360.0),
            sides: crate::AngleSides::One,
        });
        edited(&mut doc, &graph);
        let mut both_readers = vec![extrude, revolve];
        both_readers.sort();
        assert_eq!(doc.nesting_in(&graph).curve_owners_of(profile).to_vec(), both_readers);
        assert_eq!(doc.nesting_in(&graph).readers_of_curve(profile), 2);
        assert_eq!(
            doc.nesting_in(&graph).readers_of_curve(CurveId::new()),
            0,
            "and a curve nothing reads reads as nought rather than panicking",
        );
    }

    /// **A curve's readers are counted in every part**, as `block-ownership-spec.md` I2 counts
    /// them, because a curve belongs to the document and not to one part.
    ///
    /// So a pipe in a part not on screen reads it as much as one in the part that is, and the count
    /// a row shows is how many places the curve is drawn in altogether. Where it is drawn is the
    /// other half of the answer and is read per part: only the readers in *this* list draw a row
    /// here, so each part's list nests the curve inside its own pipe — a shared instance across
    /// parts, where two readers used to make it a variable in both. Asked of a copy of a part's
    /// graph — what a test or a preview holds — each reader still counts once.
    #[test]
    fn a_curve_s_readers_are_counted_in_every_part() {
        let mut doc = Document::new();
        let spine = a_line(&mut doc, "Spine");
        let pipe = |graph: &mut Graph| {
            graph.add(crate::Node::Pipe { curve: spine, radius: ParamRef::Literal(1.0) })
        };
        let mut here = Graph::empty();
        let pipe_here = pipe(&mut here);
        let here = settled_in(&mut doc, here);
        let mut there = Graph::empty();
        pipe(&mut there);
        settled_in(&mut doc, there);

        let on_screen = &doc.parts_ordered()[0].graph;
        assert_eq!(doc.nesting_in(on_screen).readers_of_curve(spine), 2, "one in each part");
        assert_eq!(doc.nesting_in(&here).readers_of_curve(spine), 2, "a copy counts each once");
        assert_eq!(
            doc.nesting_in(&here).curve_owners_of(spine).to_vec(),
            vec![pipe_here],
            "and two readers nest it in each, so this list draws it inside its own pipe",
        );
    }

    /// **A dismissed loop still closes, because dismissing it is about the list and not the shape.**
    ///
    /// The line `ungrouped_curves` must not cross. A profile row is `profile_partition`'s reading of
    /// which chains close, so dismissing one has to stop *that* and nothing else — an extrude built on
    /// the loop walks `profile_loop`, which does not consult the set, so the solid goes on being built.
    /// Asserted here because the alternative is a gesture that silently unmakes geometry.
    #[test]
    fn dismissing_a_grouping_leaves_the_loop_and_its_nesting_alone() {
        let mut doc = Document::new();
        let profile = a_line(&mut doc, "Line");
        let mut graph = Graph::empty();
        let extrude = graph.add(extrude_of(profile));
        let graph = settled_in(&mut doc, graph);

        // The dismissal is read where rows are gathered, which is not here: the graph edge and the
        // nesting it drives are untouched by it.
        doc.ungrouped_curves.insert(profile);
        assert_eq!(
            doc.nesting_in(&graph).curve_owners_of(profile).to_vec(),
            vec![extrude],
            "dismissing a grouping says nothing about who reads the curve",
        );
        assert!(
            graph.node(extrude).is_some_and(|n| n.curves().contains(&profile)),
            "and the extrude still names it, so the solid is still built",
        );
    }

    /// **A curve made a variable has a row of its own, and put back it is inside its reader
    /// again** — where the list reads it (`block-ownership-spec.md` §10, refactor plan 2.19).
    ///
    /// The case a curve needed a handle for, reported 2026-10-01: an extrude stores a single
    /// `profile: CurveId` and derives the rest of its loop by walking meetings, so the graph edge
    /// names one curve of four. While the loop is whole the profile row groups them and nothing
    /// shows. When it opens, the three it does not name have no reader and stand on their own,
    /// while the named one keeps its single reader and stays inside the extrude — "that one is now
    /// trapped inside extrude. And I cannot release it."
    ///
    /// A curve is born named (`unique_curve_name`), so a name could never be its handle, and the
    /// handle was a flag of its own, `declared_curves`. It is the stored ownership now, the one
    /// handle every row has, and the flag moves nothing: what an older build declared is read once,
    /// by the rule its list was laid out by, when the file is opened.
    #[test]
    fn a_curve_made_a_variable_has_a_row_of_its_own_and_goes_back() {
        let mut doc = Document::new();
        let profile = a_line(&mut doc, "Line");
        let mut graph = Graph::empty();
        let extrude = graph.add(extrude_of(profile));
        let graph = settled_in(&mut doc, graph);
        assert_eq!(
            doc.nesting_in(&graph).curve_owners_of(profile).to_vec(),
            vec![extrude],
            "adopted by its one reader, so it is drawn inside the extrude",
        );

        doc.ownership_curves.insert(profile, Ownership::Variable);
        let nesting = doc.nesting_in(&graph);
        assert!(
            !nesting.curve_is_nested(profile),
            "a variable, so it has a row of its own even though the extrude still reads it",
        );
        assert_eq!(nesting.readers_of_curve(profile), 1, "read by the one block it came out of");

        // And it goes back, which is the half that makes it a toggle rather than a trapdoor.
        // `Nested` names nobody: where it lands is wherever reads it, which is the extrude.
        doc.ownership_curves.insert(profile, Ownership::Nested);
        assert_eq!(doc.nesting_in(&graph).curve_owners_of(profile).to_vec(), vec![extrude]);

        doc.declared_curves.insert(profile);
        assert_eq!(
            doc.nesting_in(&graph).curve_owners_of(profile).to_vec(),
            vec![extrude],
            "an older build's declaration is read when a file opens, and moves nothing after",
        );
    }

    /// **A block one thing names twice is drawn inside it once, not twice.**
    ///
    /// A node can hold the same edge twice — a boolean handed the same body for both operands is
    /// the reachable case — and counting the edges rather than the consumers would read that as two
    /// readers: the same block drawn twice inside the one block, reported as a shared instance
    /// (§7.3) where there is nothing shared.
    #[test]
    fn a_block_named_twice_by_one_consumer_still_nests_inside_it() {
        use crate::{BooleanOp, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let both =
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), sphere.into()] });
        let graph = settled_in(&mut doc, graph);
        assert_eq!(doc.nesting_in(&graph).owners_of(sphere).to_vec(), vec![both]);
        assert_eq!(
            doc.nesting_in(&graph).readers_of(sphere),
            1,
            "and one reader, counted once however many times it is named",
        );
    }

    /// **A reader ring still leaves every block a row the drawing walk reaches, and the walk
    /// ends.**
    ///
    /// The one thing a *derived* owner can do that a stored one could not, and the reason the
    /// repair `without_rings` made on the stored entry moved here. Ownership follows readers, so a
    /// ring needs a ring in the graph: nothing in the application builds one and a file can still
    /// say one. Where a stored chain of `Inside` closing on itself was a wrong row, a reader ring
    /// closes in the walk that *draws* the rows — so left alone it is a hang, not a mark, which is
    /// what makes this a guard rather than tidiness.
    ///
    /// The rule is that the **lowest** block that cannot reach a top-level row is freed, which
    /// breaks the ring it sits on or above and gives that block a row of its own; what hung off
    /// the ring keeps its owners and is reached through that new row. Asserted on the map rather
    /// than through a graph because the graph cannot be made to hold a ring — every edge names a
    /// node that already exists — and on `from_raw` ids because the rule names the lowest, which a
    /// minted id would decide at random.
    #[test]
    fn a_reader_ring_leaves_every_block_a_row_the_drawing_walk_reaches() {
        // `low` and `high` read each other. `hanging` is drawn inside `low`, so it reaches a row
        // only through the ring; `ordinary` is drawn inside `top`, a top-level row, which is what
        // every well-formed nesting looks like and must come out of the repair untouched.
        let [low, high, hanging, ordinary, top] = [1, 2, 3, 4, 9].map(NodeId::from_raw);
        let owners: HashMap<NodeId, Vec<NodeId>> =
            [(low, vec![high]), (high, vec![low]), (hanging, vec![low]), (ordinary, vec![top])]
                .into_iter()
                .collect();

        let repaired = without_reader_rings(owners);
        assert!(!repaired.contains_key(&low), "the lowest of the ring is freed, which breaks it");
        assert_eq!(
            repaired.get(&high).map(Vec::as_slice),
            Some([low].as_slice()),
            "and the other side of the ring is drawn inside the freed row, not gone",
        );
        assert_eq!(
            repaired.get(&hanging).map(Vec::as_slice),
            Some([low].as_slice()),
            "as is what hung off it",
        );
        assert_eq!(
            repaired.get(&ordinary).map(Vec::as_slice),
            Some([top].as_slice()),
            "and an ordinary nesting is left exactly as it was",
        );

        // Every chain of owners now ends, which is the property the drawing walk needs. Walked
        // with a budget, so a ring the repair left behind fails here instead of hanging the suite.
        let nesting = Nesting { nodes: repaired, ..Default::default() };
        for id in [low, high, hanging, ordinary] {
            let mut frontier = vec![id];
            let mut steps = 0;
            while let Some(next) = frontier.pop() {
                steps += 1;
                assert!(steps <= 8, "the walk up from {id:?} does not end");
                frontier.extend(nesting.owners_of(next).iter().copied());
            }
        }
    }

    /// **Nesting reaches neither of the two orders.**
    ///
    /// It is a reading of the rows, so it must leave the compilers' fold alone — a changed fold
    /// renames every body function and rebuilds the whole shader — and it must leave the list's
    /// own order alone as well, since it says only how deep each row is. Names are written as well
    /// as ownership, since a name was what moved a row until the ownership was stored.
    /// `an_ownership_change_does_not_change_the_generated_source` is the same property asserted on
    /// the text the driver is handed.
    #[test]
    fn nesting_reaches_neither_the_fold_nor_the_list_order() {
        use crate::{BooleanOp, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: crate::no_turn(),
        });
        let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        let loose = graph.add(Node::Sphere {
            centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
            radius: lit(9.0),
        });
        for (at, id) in [(1u64, sphere), (2, cube), (3, union), (4, loose)] {
            doc.created.insert(id, 1_700_000_000_000_000 + at);
        }
        let folded = doc.folded_bodies(&graph);
        let listed = doc.listed_blocks(&graph);

        for id in [sphere, cube, union, loose] {
            doc.node_meta.entry(id).or_default().name = Some(format!("{}", id.raw()));
            doc.ownership_nodes.insert(id, Ownership::Variable);
        }
        assert_eq!(doc.folded_bodies(&graph), folded, "the fold is keyed on when, not on names");
        assert_eq!(doc.listed_blocks(&graph), listed, "and the rows arrive in the same order");
    }

    /// **Arranging the list moves the rows and leaves the compilers' order exactly as it was.**
    ///
    /// The property the two orders came apart for. A drag is presentation — it changes nothing about
    /// the model — so it must not reach the generated program, where a changed body order renames
    /// the body functions and rebuilds the whole shader. `block-order-spec.md` §4 is the argument
    /// and `arranging_the_list_does_not_change_the_generated_source` is the same property asserted
    /// on the text the driver is actually handed.
    #[test]
    fn arranging_the_list_moves_the_rows_and_leaves_the_fold_alone() {
        use crate::{Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let bodies: Vec<NodeId> = (0..4)
            .map(|i| {
                let id = graph.add(Node::Sphere {
                    centre: Place::At([lit(i as f64 * 30.0), lit(0.0), lit(0.0)]),
                    radius: lit(9.0),
                });
                doc.created.insert(id, 1_700_000_000_000_000 + i as u64 + 1);
                id
            })
            .collect();
        let folded = doc.folded_bodies(&graph);
        assert_eq!(folded, bodies, "made in order, so folded in order");
        assert_eq!(doc.listed_bodies(&graph), bodies, "and listed in it too");

        // Drag the last to the front, as `App::arrange_block` writes it.
        let arranged = vec![bodies[3], bodies[0], bodies[1], bodies[2]];
        for (i, id) in arranged.iter().enumerate() {
            doc.arrangement.insert(*id, i as u32);
        }

        assert_eq!(doc.listed_bodies(&graph), arranged, "the list is what was asked for");
        assert_eq!(
            doc.folded_bodies(&graph),
            folded,
            "and the compilers' order has not moved, so nothing is recompiled",
        );
    }

    /// A body made after a reorder goes to the bottom, where a new body belongs.
    ///
    /// Which is what an absent rank has to mean. `Option` sorts `None` first, so reading the rank
    /// as one would have put every newly created body at the *top* of a list somebody had just
    /// arranged — the one place they had said they did not want things.
    #[test]
    fn a_body_made_after_a_reorder_goes_to_the_bottom() {
        use crate::{Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let mut at = 0u64;
        let mut ball = |graph: &mut Graph, doc: &mut Document| {
            at += 1;
            let id = graph.add(Node::Sphere {
                centre: Place::At([lit(at as f64 * 30.0), lit(0.0), lit(0.0)]),
                radius: lit(9.0),
            });
            doc.created.insert(id, 1_700_000_000_000_000 + at);
            id
        };
        let a = ball(&mut graph, &mut doc);
        let b = ball(&mut graph, &mut doc);
        // Arranged: b above a.
        doc.arrangement.insert(b, 0);
        doc.arrangement.insert(a, 1);
        assert_eq!(doc.listed_bodies(&graph), vec![b, a]);

        let fresh = ball(&mut graph, &mut doc);
        assert_eq!(
            doc.listed_bodies(&graph),
            vec![b, a, fresh],
            "the new body goes last, not above the rows that were arranged",
        );
    }

    /// **Building on the oldest body does not move it to the bottom of the list.**
    ///
    /// The reason a branch is keyed on when it was *started* rather than on its head's own moment,
    /// and it was nearly the other way. A shell is the newest block in the document, so keyed on
    /// the head the whole branch sorted last — shelling the first body you made would drop it below
    /// everything, which is the list rearranging itself in response to an edit that added to one
    /// row.
    #[test]
    fn building_on_the_oldest_body_leaves_its_branch_where_it_was() {
        use crate::{Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: crate::no_turn(),
        });
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
            radius: lit(9.0),
        });
        let rod = graph.add(Node::Sphere {
            centre: Place::At([lit(120.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        let mut at = 0u64;
        let mut stamp = |doc: &mut Document, id| {
            at += 1;
            doc.created.insert(id, 1_700_000_000_000_000 + at);
        };
        for id in [cube, ball, rod] {
            stamp(&mut doc, id);
        }
        assert_eq!(doc.listed_blocks(&graph), vec![cube, ball, rod], "in the order they were made");

        // Now shell the oldest one, which makes the newest block in the document part of the
        // oldest branch.
        let shell = graph.add(Node::Shell { input: cube.into(), thickness: lit(2.0) });
        stamp(&mut doc, shell);
        assert_eq!(
            doc.listed_blocks(&graph),
            vec![cube, shell, ball, rod],
            "the cube's branch stays first and the shell joins it",
        );
        assert_eq!(
            doc.listed_bodies(&graph),
            vec![shell, ball, rod],
            "and the shelled cube is still the first body, as the cube was",
        );
    }

    /// **A body already in the part keeps its place when another body is added.**
    ///
    /// The property the whole of `folded_bodies` exists for. Ordered by `NodeId`, a new body took a
    /// rank drawn uniformly among the existing ones — last only 1/(n+1) of the time, measured at
    /// that rate by `fieldform-gpu/examples/body_order_probe.rs` — and every body it landed in
    /// front of was renamed in the generated shader and compiled again.
    ///
    /// Twenty rounds, because the failure is a random id falling on one side of another rather
    /// than the other: one round passes by luck a quarter of the time.
    #[test]
    fn a_body_keeps_its_place_when_another_body_is_made() {
        use crate::{Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        for round in 0..20 {
            let mut doc = Document::new();
            let mut graph = Graph::empty();
            let mut made = 0u64;
            let ball = |graph: &mut Graph, doc: &mut Document, made: &mut u64| {
                let id = graph.add(Node::Sphere {
                    centre: Place::At([lit(*made as f64 * 30.0), lit(0.0), lit(0.0)]),
                    radius: lit(9.0),
                });
                *made += 1;
                doc.created.insert(id, 1_700_000_000_000_000 + *made);
                id
            };
            let existing: Vec<NodeId> =
                (0..4).map(|_| ball(&mut graph, &mut doc, &mut made)).collect();
            assert_eq!(doc.folded_bodies(&graph), existing, "round {round}: in the order made");

            let fresh = ball(&mut graph, &mut doc, &mut made);
            let mut expected = existing.clone();
            expected.push(fresh);
            assert_eq!(
                doc.folded_bodies(&graph),
                expected,
                "round {round}: the new body goes last and disturbs nothing",
            );
        }
    }

    /// **A stand-in sorts where the block it stands for sorts.**
    ///
    /// The viewport draws a consumed block by adding a second, unconsumed reference to it, under
    /// an id *derived* from the source's rather than minted (`NodeId::stand_in`). So a stand-in is
    /// in no `created` map, and reading its stamp naively gives zero — which sorts every shown
    /// ingredient to the top of the drawn list. Stable and wrong, which is the harder failure to
    /// notice: nothing flickers, the wrong body is simply highlighted.
    #[test]
    fn a_stand_in_sorts_where_the_block_it_stands_for_sorts() {
        use crate::{BooleanOp, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        // An older body, then a block consumed by a boolean, then a newer body — so the block
        // belongs in the middle and cannot get there by accident.
        let old = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(9.0) });
        let inner = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(30.0), lit(0.0), lit(0.0)]),
            size: [lit(8.0); 3],
            turn: crate::no_turn(),
        });
        let joined = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![inner.into()] });
        let new = graph.add(Node::Sphere { centre: Place::At([lit(60.0); 3]), radius: lit(9.0) });
        for (at, id) in [(1u64, old), (2, inner), (3, joined), (4, new)] {
            doc.created.insert(id, 1_700_000_000_000_000 + at);
        }

        let mut drawn = graph.clone();
        let copy = drawn.add_stand_in(inner).expect("the block is in the graph");
        assert!(doc.created.get(&copy).is_none(), "a stand-in is never stamped");
        assert_eq!(
            doc.made_at(copy),
            doc.made_at(inner),
            "so its moment is the moment of what it stands for",
        );
        assert_eq!(
            doc.folded_bodies(&drawn),
            vec![old, copy, joined, new],
            "and it sits where that block sits, not at the top",
        );
    }

    /// Bodies nothing has stamped still come back in *an* order, rather than in hash order.
    ///
    /// What an older file's nodes are, and what the starting document is. The order is the ids',
    /// which is arbitrary — the point is only that it is the same twice, since a list that
    /// reshuffles between frames is worse than one whose order is merely a convention.
    #[test]
    fn unstamped_bodies_still_come_back_in_one_order() {
        use crate::{Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        for i in 0..6 {
            graph.add(Node::Sphere {
                centre: Place::At([lit(i as f64 * 30.0), lit(0.0), lit(0.0)]),
                radius: lit(9.0),
            });
        }
        let doc = Document::new();
        let first = doc.folded_bodies(&graph);
        assert_eq!(first.len(), 6);
        for _ in 0..20 {
            assert_eq!(doc.folded_bodies(&graph), first, "the answer must not move between calls");
        }
        let mut by_id = first.clone();
        by_id.sort_by_key(|id| id.raw());
        assert_eq!(first, by_id, "with every stamp tied, the id is the tiebreak");
    }

    #[test]
    fn with_cage_builds_a_part_that_references_it() {
        let doc = Document::with_cage(Cage::cube());
        assert_eq!(doc.cages.len(), 1);
        assert_eq!(doc.parts.len(), 1);

        let part = doc.parts_ordered()[0];
        let used = doc.cages_used_by(part.id);
        assert_eq!(used.len(), 1);
        assert!(doc.cage(used[0]).is_some(), "graph must name a real cage");
    }

    #[test]
    fn several_parts_share_one_document_scope() {
        // The Part Studio property: parts coexist and may reference the same objects, so
        // cross-references between separately owned parts are never needed.
        let mut doc = Document::new();
        let shared = doc.add_cage(Cage::cube());
        let a = doc.add_part(Part::from_cage("A", shared));
        let b = doc.add_part(Part::from_cage("B", shared));

        assert_ne!(a, b);
        assert_eq!(doc.cages_used_by(a), doc.cages_used_by(b));
        assert_eq!(doc.cages.len(), 1, "one cage, referenced twice");
    }

    #[test]
    fn part_ordering_is_stable_despite_random_ids() {
        // Random ids mean HashMap order varies between runs; the UI must not.
        let mut doc = Document::new();
        let cage = doc.add_cage(Cage::cube());
        for name in ["zeta", "alpha", "mu"] {
            doc.add_part(Part::from_cage(name, cage));
        }
        let names: Vec<&str> = doc.parts_ordered().iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["alpha", "mu", "zeta"]);
    }

    #[test]
    fn geometry_is_millimetres_regardless_of_display_unit() {
        // Changing the display unit must not touch geometry — the property that makes unit
        // switching lossless and instant.
        let mut doc = Document::with_cage(Cage::cube());
        let cage_id = *doc.cages.keys().next().unwrap();
        let before: Vec<_> = doc
            .cage(cage_id)
            .unwrap()
            .vertex_ids()
            .map(|v| doc.cage(cage_id).unwrap().vertex_position(v))
            .collect();

        doc.unit = Unit::Inch;

        let after: Vec<_> = doc
            .cage(cage_id)
            .unwrap()
            .vertex_ids()
            .map(|v| doc.cage(cage_id).unwrap().vertex_position(v))
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn a_literal_parameter_is_its_number() {
        let mut doc = Document::new();
        let id = doc.add_parameter("wall_thickness", 2.0);
        assert_eq!(doc.parameters[&id].value.as_literal(), Some(2.0));
        assert_eq!(doc.parameter_value(id), Some(2.0));
    }

    /// The worked case from the spec: `radius = 5`, `boss_r = radius * 5 - 2` reads 23, and
    /// 28 once `radius` is 6 with nothing else touched.
    #[test]
    fn an_expression_reads_other_parameters_and_follows_them() {
        let mut doc = Document::new();
        let radius = doc.add_parameter("radius", 5.0);
        let boss_r = doc.add_parameter("boss_r", 0.0);
        doc.set_parameter_expression(boss_r, "radius * 5 - 2").unwrap();

        assert_eq!(doc.parameter_value(boss_r), Some(23.0));
        assert_eq!(doc.parameters[&boss_r].value.as_literal(), None, "computed, not typed");
        assert_eq!(
            doc.parameters[&boss_r].value.expression().unwrap().source(),
            "radius * 5 - 2"
        );

        doc.set_parameter_literal(radius, 6.0);
        assert_eq!(doc.parameter_value(boss_r), Some(28.0));
        assert_eq!(doc.parameters_reading(radius), vec![boss_r]);
        assert_eq!(doc.parameter_order(), Ok(vec![radius, boss_r]));
    }

    #[test]
    fn a_direct_cycle_is_refused_on_the_edit() {
        let mut doc = Document::new();
        let a = doc.add_parameter("a", 1.0);
        assert_eq!(
            doc.set_parameter_expression(a, "a + 1"),
            Err(ExpressionError::Cycle(vec!["a".into(), "a".into()]))
        );
        assert_eq!(doc.parameters[&a].value, ParamValue::Literal(1.0), "nothing was stored");
    }

    #[test]
    fn an_indirect_cycle_is_refused_and_the_loop_is_named() {
        let mut doc = Document::new();
        let a = doc.add_parameter("a", 1.0);
        let b = doc.add_parameter("b", 1.0);
        let c = doc.add_parameter("c", 1.0);
        doc.set_parameter_expression(a, "b + 1").unwrap();
        doc.set_parameter_expression(b, "c + 1").unwrap();
        let err = doc.set_parameter_expression(c, "a + 1").unwrap_err();
        assert_eq!(
            err,
            ExpressionError::Cycle(vec!["c".into(), "a".into(), "b".into(), "c".into()])
        );
        assert_eq!(err.to_string(), "would loop: c \u{2192} a \u{2192} b \u{2192} c");
        assert_eq!(doc.parameter_value(c), Some(1.0), "c is still the literal it was");
        assert_eq!(doc.parameter_value(a), Some(3.0));
    }

    /// The reason references are ids: the new name contains the old one, and then the old
    /// one contains the new, which is exactly where a text rewrite goes wrong.
    #[test]
    fn renaming_does_not_break_a_reference_even_to_a_substring_of_the_old_name() {
        let mut doc = Document::new();
        let wall = doc.add_parameter("wall", 2.0);
        let inner = doc.add_parameter("inner", 10.0);
        let outer = doc.add_parameter("outer", 0.0);
        doc.set_parameter_expression(outer, "inner+2*wall").unwrap();
        assert_eq!(doc.parameter_value(outer), Some(14.0));

        doc.rename_parameter(wall, "wall_thickness").unwrap();
        assert_eq!(doc.parameter_value(outer), Some(14.0));
        assert_eq!(
            doc.parameters[&outer].value.expression().unwrap().source(),
            "inner + 2 * wall_thickness",
            "re-rendered from the tree"
        );
        assert_eq!(doc.parameter_named("wall_thickness"), Some(wall));
        assert_eq!(doc.parameter_named("wall"), None);
        assert_eq!(doc.parameters[&inner].name, "inner");

        doc.rename_parameter(wall, "wall").unwrap();
        assert_eq!(
            doc.parameters[&outer].value.expression().unwrap().source(),
            "inner + 2 * wall"
        );
        assert_eq!(
            doc.parse_expression("wall_thickness").unwrap_err().message,
            "`wall_thickness` is not a parameter"
        );
    }

    #[test]
    fn a_name_is_anything_with_something_in_it_and_no_bracket_that_nobody_else_has() {
        let mut doc = Document::new();
        let a = doc.add_parameter("a", 1.0);
        let _b = doc.add_parameter("b", 1.0);
        assert_eq!(doc.rename_parameter(a, ""), Err(RenameError::NotAName(String::new())));
        assert_eq!(doc.rename_parameter(a, "a[1]"), Err(RenameError::NotAName("a[1]".into())));
        assert_eq!(doc.rename_parameter(a, " a"), Err(RenameError::NotAName(" a".into())));
        assert_eq!(doc.rename_parameter(a, "b"), Err(RenameError::Taken("b".into())));
        assert_eq!(doc.rename_parameter(a, "a"), Ok(()), "keeping a name is not taking it");
        assert_eq!(doc.rename_parameter(a, "_a2"), Ok(()));
        // A dash, a space or a leading digit are said in brackets by an expression, not
        // refused: "cap-1" is a label, not arithmetic.
        for name in ["cap-1", "boss r", "2x", "cap0"] {
            assert_eq!(doc.rename_parameter(a, name), Ok(()), "{name}");
        }
        assert!(Document::is_parameter_name("boss_r") && Document::is_parameter_name("boss-r"));
    }

    /// Renaming to a name beyond an identifier rewrites its readers with brackets, and the
    /// rewritten text parses back to the same reading.
    #[test]
    fn a_reader_of_a_bracketed_name_is_rewritten_with_the_brackets_and_reparses() {
        let mut doc = Document::new();
        let wall = doc.add_parameter("wall", 2.0);
        let outer = doc.add_parameter("outer", 0.0);
        doc.set_parameter_expression(outer, "wall * 2").unwrap();
        doc.rename_parameter(wall, "wall-1").unwrap();
        let source = doc.parameters[&outer].value.expression().unwrap().source().to_string();
        assert_eq!(source, "[wall-1] * 2");
        assert_eq!(doc.parse_expression(&source).unwrap().references(), vec![wall]);
        assert_eq!(doc.parameter_value(outer), Some(4.0));
    }

    /// The test that keeps NaN out of the field: a failed expression is worth nothing, and so
    /// is anything reading it, and both say why.
    #[test]
    fn a_failed_expression_has_no_value_and_neither_does_what_reads_it() {
        let mut doc = Document::new();
        let d = doc.add_parameter("d", 0.0);
        let e = doc.add_parameter("e", 0.0);
        doc.set_parameter_expression(d, "1 / 0").unwrap();
        doc.set_parameter_expression(e, "d + 1").unwrap();
        assert_eq!(doc.evaluate_parameter(d), Err(ParameterError::NotFinite("1 / 0".into())));
        assert_eq!(doc.parameter_value(d), None);
        assert_eq!(doc.evaluate_parameter(e), Err(ParameterError::Unresolved("d".into())));
        assert_eq!(doc.parameter_value(e), None);
        assert_eq!(doc.evaluate_parameter(ParamId::new()), Err(ParameterError::Missing));
    }

    /// The lookup every compiler and walk wrote out by hand, now asked of the document: each
    /// kind of reference answers what the hand-written closure answered, and a failure is
    /// `None` rather than a number.
    #[test]
    fn a_reference_resolves_through_the_document_exactly_as_the_closure_did() {
        let mut doc = Document::new();
        let radius = doc.add_parameter("radius", 5.0);
        let boss_r = doc.add_parameter("boss_r", 0.0);
        let broken = doc.add_parameter("broken", 0.0);
        doc.set_parameter_expression(boss_r, "radius * 5 - 2").unwrap();
        doc.set_parameter_expression(broken, "1 / 0").unwrap();

        let cases = [
            (ParamRef::Literal(2.5), Some(2.5)),
            (ParamRef::Named(radius), Some(5.0)),
            (ParamRef::Named(boss_r), Some(23.0)),
            (ParamRef::Named(broken), None),
            (ParamRef::Named(ParamId::new()), None),
            (ParamRef::Field(NodeId::new()), None),
        ];
        for (r, worth) in cases {
            assert_eq!(doc.resolve(&r), worth, "{r:?}");
            assert_eq!(doc.resolve(&r), r.resolve(&|id| doc.parameter_value(id)), "{r:?}");
        }
    }

    /// A file can say what the editor refuses. The order reports the loop, and evaluation
    /// stops at the depth guard rather than recursing forever.
    #[test]
    fn a_cycle_built_by_hand_is_reported_by_the_order_and_bounded_in_evaluation() {
        let mut doc = Document::new();
        let a = doc.add_parameter("a", 1.0);
        let b = doc.add_parameter("b", 1.0);
        doc.set_parameter_expression(a, "b + 1").unwrap();
        // Around the guard, as a hand-edited file would be.
        let back = doc.parse_expression("a + 1").unwrap();
        doc.parameters.get_mut(&b).unwrap().value = ParamValue::Expression(back);

        let cycle = doc.parameter_order().unwrap_err();
        assert_eq!(cycle.len(), 3);
        assert_eq!(cycle[0], cycle[2]);
        assert!(cycle.contains(&a) && cycle.contains(&b));
        assert_eq!(doc.parameter_value(a), None);
        // The innermost failure is the depth guard; from the outside it reads as the other
        // parameter having no value, which is true.
        assert_eq!(doc.evaluate_parameter(a), Err(ParameterError::Unresolved("b".into())));
    }
}
