// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Reading and writing a [`Document`] as a `.fform` archive.
//!
//! [`Archive`] is the container and knows nothing about what it holds; this is the layer that
//! decides what goes in which entry, and it is deliberately the only place that decision is
//! made.
//!
//! # Why cages are separate entries
//!
//! ```text
//! bracket.fform
//! ├── manifest.json     format version, producer, authoring unit
//! ├── document.json     curves, parameters, parts (and each part's graph)
//! ├── cages/<id>.json   one entry per cage
//! └── surfaces/<id>.json   one entry per open quad mesh
//! ```
//!
//! The split is along the two-layer line, and it is what makes the format's promises
//! deliverable rather than merely stated. `document.json` is the **serialized program**: small,
//! human-scale, and reviewable as a diff — "blend radius 3mm → 5mm" is a legible change. A
//! cage is a **state snapshot** of a hundred thousand coordinates, and inlining one would bury
//! the program in it and make every diff of the document a diff of the mesh.
//!
//! # The unit lives in the manifest
//!
//! Once, not twice. It is the one field a reader wants before deciding whether to open the
//! rest, and two copies of a fact are two chances for them to disagree.
//!
//! # Forward compatibility, precisely
//!
//! The graph is the part that versions gracefully: a node kind this build has never heard of
//! is preserved and refused, never dropped (see [`fieldform_scene::Node::Unknown`]), so
//! **adding a node kind is a minor bump**.
//!
//! Everything else here is the opposite, on purpose. A curve, a part, a parameter or a cage
//! carrying a field this build cannot account for is *refused at open* rather than loaded
//! without it — serde's default of quietly ignoring what it does not recognise would mean
//! reading a document, dropping the part of it we did not understand, and writing that loss
//! back over the user's file. So **adding a field to any of those is a major bump**, and an
//! older build says so plainly instead of silently discarding it.

use crate::format::{Archive, FormatError, Manifest};
use fieldform_cage::Cage;
use fieldform_scene::{
    BlockKey, CageId, Curve, CurveId, Document, Expression, NodeId, Ownership, ParamId,
    ParamValue, Parameter, Part, PartId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Path of the serialized program within the archive.
pub(crate) const DOCUMENT_PATH: &str = "document.json";

/// Where node creation times live.
///
/// Its own entry rather than a field in `document.json`, because the document JSON refuses
/// fields it does not know — which is the right rule for the model itself and the wrong one
/// for a note beside it. Anything the core does not recognise is carried through untouched,
/// so an older build opening this file keeps the times it cannot read and hands them back on
/// save, exactly as it does for a plugin's bytes.
pub(crate) const CREATED_PATH: &str = "history/created.json";

/// What each body is called and whether it is shown.
///
/// Its own entry for the same reason `created` has one: a build that has never heard of it
/// carries the bytes through instead of refusing the document, which putting the same
/// information on the node itself would not allow.
pub(crate) const NODE_META_PATH: &str = "history/node-meta.json";

/// Where a person has dragged each block to in the list.
///
/// **Its own entry, and that is a format decision rather than tidiness.** The obvious home is a
/// field on [`fieldform_scene::NodeMeta`], whose own comment invites the next per-body annotation
/// — but `NodeMeta` is `deny_unknown_fields` and it is loaded below with `get_json(..)?`, so a
/// field added to it is not a dropped name in an older build, it is a **failed document load**.
/// That is the opposite of the preserved-and-refused rule the format exists to follow (D3). An
/// entry of its own is carried through untouched instead, which
/// `unrecognised_entries_anywhere_are_carried_untouched` proves.
pub(crate) const ARRANGEMENT_PATH: &str = "history/arrangement.json";

/// The same as [`CREATED_PATH`], for curves.
pub(crate) const CURVES_CREATED_PATH: &str = "history/curves-created.json";
/// Names given to profile rows, by the row's first curve (`profile-group-spec.md` §3.3). A
/// side entry like the creation times, for the same reason: a name changes no shape, so an
/// older build that never reads it draws exactly the right geometry and carries the entry
/// through byte-intact. Left out when there are none.
pub(crate) const PROFILE_NAMES_PATH: &str = "history/profile-names.json";
/// The curves asked to keep a row of their own, as a build from before 3.15 reads them
/// (`Document::declared_curves`). Written from the ownership since refactor plan 2.19
/// (`Document::curves_declared_for_older_builds`).
///
/// An entry of its own rather than a field on anything existing, for the reason every side entry
/// since 3.1 has been: an older build opens the file, draws it identically — a declaration moves a
/// row and no geometry — and carries this through byte-intact, where an unknown field in a
/// `deny_unknown_fields` table would be a failed load instead.
pub(crate) const DECLARED_CURVES_PATH: &str = "history/declared-curves.json";
/// Where every block sits in the list: nested, free, or a variable (`block-ownership-spec.md`
/// §9). A nested block names no parent since 112 — it is drawn inside every block that reads it,
/// and the readers are the graph's to answer (`notebook-spec.md` §12).
///
/// An entry of its own for the reason [`ARRANGEMENT_PATH`] is: `NodeMeta` denies a field it does
/// not know, so a field there would be a failed load in an older build, where an entry of its own
/// is carried through untouched. **Written in full** — every node and every curve, free ones
/// included — against the left-out-when-empty habit of every side entry before it, because an id
/// it leaves out has to mean "the writer did not know this block", which is what a block an older
/// build added is. That holds only if a writer that knows a block always says where it is.
pub(crate) const OWNERSHIP_PATH: &str = "history/ownership.json";
/// The curves that decline to be gathered into a profile row (`Document::ungrouped_curves`).
///
/// Its own entry for the reason `DECLARED_CURVES_PATH` is: an older build opens the file, draws it
/// identically -- a dismissal changes which rows the list shows and no geometry -- and carries the
/// entry through byte-intact.
pub(crate) const UNGROUPED_CURVES_PATH: &str = "history/ungrouped-curves.json";
/// Axes stood on a body's surface (`Document::surface_axes`), by the curve each one draws.
///
/// An entry of its own for the reason every side entry since 3.1 has been: the curve is an ordinary
/// two-point line with its points stored, so an older build opens the file, draws the axis
/// **identically** — only the rule for *re-deriving* those points is missing — and carries the entry
/// through byte-intact. Left out when there are none.
pub(crate) const SURFACE_AXES_PATH: &str = "history/surface-axes.json";
/// Relations between curves (`profile-group-spec.md` §5.5), a side entry and a minor bump.
/// A relation does not change the shape: the points it drives are already in the curves, so
/// an older build that never reads this draws exactly the right geometry and carries the
/// entry through byte-intact. What it loses is the future behaviour -- editing a master there
/// will not drag its slave -- and the check on open in a newer build reports what moved.
pub(crate) const RELATIONS_PATH: &str = "history/relations.json";
/// Relations of the kinds added in 3.3 -- concentric and mirror. Kept out of
/// [`RELATIONS_PATH`] because a 3.2 build reads that strictly, and one kind it did not know
/// there would stop it opening the file; in an entry of their own it opens the file, draws it
/// exactly, and carries them through.
pub(crate) const RELATIONS_ADDED_PATH: &str = "history/relations-added.json";

/// Whether a 3.2 build knows this kind of relation, and so where it is written.
fn read_by_3_2(relation: &fieldform_scene::Relation) -> bool {
    use fieldform_scene::Relation;
    matches!(
        relation,
        Relation::Joint { .. }
            | Relation::Parallel { .. }
            | Relation::Perpendicular { .. }
            | Relation::EqualLength { .. }
    )
}
/// What each surface is called. Beside the meshes rather than inside them, so the geometry
/// files stay exactly the geometry.
pub(crate) const SURFACE_NAMES_PATH: &str = "surfaces/names.json";
/// The tessellation blocks, in the order they were made — backlog item 53. Their meshes are not
/// saved: a mesh is derived, and an active block rebuilds its own on open.
pub(crate) const TESSELLATIONS_PATH: &str = "outputs/tessellations.json";
/// The export blocks, in the order they were made.
pub(crate) const EXPORTS_PATH: &str = "outputs/exports.json";

/// Directory prefix holding one entry per cage.
pub(crate) const CAGES_PREFIX: &str = "cages/";

/// Directory prefix holding one entry per imported mesh.
///
/// Its own entry per mesh for exactly the reason a cage gets one: `document.json` is the
/// serialized *program* and should stay readable, while a mesh is a state snapshot of possibly
/// millions of coordinates. Inlining one would bury the program in it.
pub(crate) const MESHES_PREFIX: &str = "meshes/";

/// The entry path an imported mesh is stored at.
pub(crate) fn mesh_path(id: fieldform_scene::MeshId) -> String {
    format!("{MESHES_PREFIX}{id}.json")
}

/// The entry path a cage is stored at.
pub(crate) fn cage_path(id: CageId) -> String {
    format!("{CAGES_PREFIX}{id}.json")
}

/// Directory prefix holding one entry per open quad mesh.
///
/// **`surfaces/` rather than `quad-meshes/`**, because that is what one is for: a panel or a
/// shell to follow, not a solid to evaluate. The distinction from `cages/` is not the shape of
/// the data — an open mesh serializes identically to a cage, positions and quads and nothing
/// else — it is what the document may do with it, and the prefix is where a reader learns that
/// without parsing anything.
pub(crate) const SURFACES_PREFIX: &str = "surfaces/";

/// The entry path an open quad mesh is stored at.
pub(crate) fn surface_path(id: fieldform_scene::QuadMeshId) -> String {
    format!("{SURFACES_PREFIX}{id}.json")
}

/// `document.json`: everything above the direct layer.
///
/// [`BTreeMap`] rather than the document's `HashMap`, so the same document saves to the same
/// bytes every time. Hash order varies between runs, and a format whose diffs are noise cannot
/// support reviewing a change to a part.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentJson {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    curves: BTreeMap<CurveId, Curve>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    parameters: BTreeMap<ParamId, ParameterJson>,
    /// The files imports came from. `default` rather than required, so a document written
    /// before imports were blocks still opens.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    sources: BTreeMap<fieldform_scene::id::SourceId, fieldform_scene::ImportSource>,
    parts: BTreeMap<PartId, Part>,
}

/// [`OWNERSHIP_PATH`]: every node and every curve, and where each sits.
///
/// [`BTreeMap`]s for `document.json`'s reason, so the same document saves to the same bytes. Both
/// maps are always written and both are required, and a key this build does not know is refused
/// rather than dropped (D3): something later builds want to say about where blocks sit goes in an
/// entry of its own, as 3.3's relations did, and this one stays what a 3.15 build can read whole.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnershipJson {
    nodes: BTreeMap<NodeId, Ownership>,
    curves: BTreeMap<CurveId, Ownership>,
}

/// A parameter as the file holds it: a literal as its number, an expression as its text.
///
/// The text rather than the tree, because `document.json` is meant to be read. It has to
/// round-trip through name resolution on load, so a file naming a parameter that no longer
/// exists is refused with the name in the message rather than opened with a zero in its place
/// (D3) — which is [`parameters_from_json`].
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterJson {
    id: ParamId,
    name: String,
    value: ParamValueJson,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ParamValueJson {
    Literal(f64),
    Expression(String),
}

impl ParameterJson {
    fn of(parameter: &Parameter) -> Self {
        Self {
            id: parameter.id,
            name: parameter.name.clone(),
            value: match &parameter.value {
                ParamValue::Literal(v) => ParamValueJson::Literal(*v),
                ParamValue::Expression(e) => ParamValueJson::Expression(e.source().to_string()),
            },
        }
    }
}

/// The file's parameters as the document's, with every expression's names resolved.
///
/// Refuses, naming the parameter and the reason, when a name resolves to nothing or to two
/// things. The cycle check comes after, on the document, because it needs every expression
/// in place to walk.
fn parameters_from_json(
    json: BTreeMap<ParamId, ParameterJson>,
) -> Result<std::collections::HashMap<ParamId, Parameter>, FormatError> {
    let mut by_name: std::collections::HashMap<String, Vec<ParamId>> =
        std::collections::HashMap::new();
    for (id, p) in &json {
        by_name.entry(p.name.clone()).or_default().push(*id);
    }
    // Two parameters with one name make every expression saying that name ambiguous. Refused
    // only when there is an expression to be ambiguous about, so a file of plain numbers that
    // happens to repeat a name still opens.
    if json.values().any(|p| matches!(p.value, ParamValueJson::Expression(_))) {
        if let Some(name) = by_name.iter().find(|(_, ids)| ids.len() > 1).map(|(n, _)| n) {
            return Err(FormatError::Parameter {
                name: name.clone(),
                reason: "two parameters have this name, so an expression cannot say which it \
                         means"
                    .into(),
            });
        }
    }
    let lookup = |name: &str| by_name.get(name).and_then(|ids| (ids.len() == 1).then(|| ids[0]));
    let mut out = std::collections::HashMap::with_capacity(json.len());
    for (id, p) in json {
        let value = match p.value {
            ParamValueJson::Literal(v) => ParamValue::Literal(v),
            ParamValueJson::Expression(text) => ParamValue::Expression(
                Expression::parse(&text, &lookup).map_err(|e| FormatError::Parameter {
                    name: p.name.clone(),
                    reason: e.to_string(),
                })?,
            ),
        };
        out.insert(id, Parameter { id, name: p.name, value });
    }
    Ok(out)
}

/// Builds a fresh archive holding `document`, for a file that does not exist yet.
///
/// To save over a file that *was* opened, use [`update_archive`] on the archive [`open`]
/// returned — a new archive has none of the third-party `ext/` data or `sources/` the original
/// carried, and writing it would destroy exactly what the container exists to preserve.
pub fn to_archive(document: &Document) -> Result<Archive, FormatError> {
    let mut archive = Archive::new(&Manifest { unit: document.unit, ..Manifest::default() });
    update_archive(&mut archive, document)?;
    Ok(archive)
}

/// Writes `document` into an existing archive, leaving every entry it does not own alone.
///
/// Entries under `ext/` and `sources/` are untouched, which is the guarantee a marketplace
/// rests on: a user without a plugin opens the file, edits the geometry, saves, and the
/// plugin's bytes come back unchanged.
pub fn update_archive(archive: &mut Archive, document: &Document) -> Result<(), FormatError> {
    let manifest = Manifest { unit: document.unit, ..Manifest::default() };
    archive.put_json(crate::format::MANIFEST_PATH, &manifest)?;

    let json = DocumentJson {
        curves: document.curves.iter().map(|(id, c)| (*id, c.clone())).collect(),
        parameters: document
            .parameters
            .iter()
            .map(|(id, p)| (*id, ParameterJson::of(p)))
            .collect(),
        sources: document.sources.iter().map(|(id, s)| (*id, s.clone())).collect(),
        parts: document.parts.iter().map(|(id, p)| (*id, p.clone())).collect(),
    };
    archive.put_json(DOCUMENT_PATH, &json)?;

    // Written as a sorted map for the same reason the graph is: two saves of a document that
    // has not changed have to produce the same bytes, or every diff is noise.
    if document.created.is_empty() {
        archive.remove(CREATED_PATH);
    } else {
        let created: std::collections::BTreeMap<fieldform_scene::NodeId, u64> =
            document.created.iter().map(|(id, at)| (*id, *at)).collect();
        archive.put_json(CREATED_PATH, &created)?;
    }
    // Absent entirely for a document nobody has rearranged, which is most of them.
    if document.arrangement.is_empty() {
        archive.remove(ARRANGEMENT_PATH);
    } else {
        let arrangement: std::collections::BTreeMap<fieldform_scene::NodeId, u32> =
            document.arrangement.iter().map(|(id, at)| (*id, *at)).collect();
        archive.put_json(ARRANGEMENT_PATH, &arrangement)?;
    }
    // Only the bodies that have something to remember, so a document nobody has renamed, hidden
    // or shown anything in writes no entry at all.
    let meta: std::collections::BTreeMap<fieldform_scene::NodeId, fieldform_scene::NodeMeta> =
        document
            .node_meta
            .iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|(id, m)| (*id, m.clone()))
            .collect();
    if meta.is_empty() {
        archive.remove(NODE_META_PATH);
    } else {
        archive.put_json(NODE_META_PATH, &meta)?;
    }
    // **Every node and every curve, and nothing that has gone** (`block-ownership-spec.md` I5). The
    // document keeps the entry of a block an edit removed, because an undo can bring the block back
    // and its entry has to be there when it does; a saved file has no undo to serve, so only the
    // blocks that are here are written. One no settle has reached yet has no entry, and is written
    // as `Document::ownership` reads it — free, until one does — so the file and the list cannot
    // disagree about it.
    let ownership = OwnershipJson {
        nodes: document
            .parts
            .values()
            .flat_map(|part| part.graph.nodes().map(|(id, _)| id))
            .map(|id| (id, document.ownership(BlockKey::Node(id))))
            .collect(),
        curves: document
            .curves
            .keys()
            .map(|id| (*id, document.ownership(BlockKey::Curve(*id))))
            .collect(),
    };
    archive.put_json(OWNERSHIP_PATH, &ownership)?;
    // Left out when empty, so a document with no export blocks writes exactly what it wrote
    // before they existed. A file that has them is refused by a build that predates them, by
    // name, because `Tessellation` and `Export` deny fields they do not know (D3).
    if document.tessellations.is_empty() {
        archive.remove(TESSELLATIONS_PATH);
    } else {
        archive.put_json(TESSELLATIONS_PATH, &document.tessellations)?;
    }
    if document.exports.is_empty() {
        archive.remove(EXPORTS_PATH);
    } else {
        archive.put_json(EXPORTS_PATH, &document.exports)?;
    }
    if document.curves_created.is_empty() {
        archive.remove(CURVES_CREATED_PATH);
    } else {
        let created: std::collections::BTreeMap<CurveId, u64> =
            document.curves_created.iter().map(|(id, at)| (*id, *at)).collect();
        archive.put_json(CURVES_CREATED_PATH, &created)?;
    }
    // What 3.2 reads where 3.2 reads it, and the kinds added since beside it.
    let (known, added): (Vec<_>, Vec<_>) =
        document.relations.iter().partition(|(_, r)| read_by_3_2(&r.relation));
    for (path, records) in [(RELATIONS_PATH, known), (RELATIONS_ADDED_PATH, added)] {
        if records.is_empty() {
            archive.remove(path);
        } else {
            let records: std::collections::BTreeMap<_, _> =
                records.into_iter().map(|(id, r)| (*id, r.clone())).collect();
            archive.put_json(path, &records)?;
        }
    }
    if document.ungrouped_curves.is_empty() {
        archive.remove(UNGROUPED_CURVES_PATH);
    } else {
        let ungrouped: Vec<CurveId> = document.ungrouped_curves.iter().copied().collect();
        archive.put_json(UNGROUPED_CURVES_PATH, &ungrouped)?;
    }
    // **Only the axes whose curve is still here.** Deleting an axis's curve leaves its record in the
    // document, deliberately: undoing the delete brings the curve back and the record must still be
    // there to meet it. A *saved* file has no undo to serve, so a record with no curve is dropped on
    // the way out rather than carried for ever.
    let surface_axes: std::collections::BTreeMap<_, _> = document
        .surface_axes
        .iter()
        .filter(|(curve, _)| document.curves.contains_key(*curve))
        .map(|(curve, axis)| (*curve, *axis))
        .collect();
    if surface_axes.is_empty() {
        archive.remove(SURFACE_AXES_PATH);
    } else {
        archive.put_json(SURFACE_AXES_PATH, &surface_axes)?;
    }
    // **A reading of the ownership above, for an older build** (spec §9.4): the curves that are
    // variables and that build's rule would not make one without this entry. This build reads the
    // ownership and reads this only for a curve the ownership does not place, and the curve row's
    // V writes the ownership alone, so the set the file was opened with says nothing about now:
    // written from it, a curve put back inside since would be lifted out again in the older build,
    // and one made a variable since would stay inside there.
    let declared = document.curves_declared_for_older_builds();
    if declared.is_empty() {
        archive.remove(DECLARED_CURVES_PATH);
    } else {
        archive.put_json(DECLARED_CURVES_PATH, &declared)?;
    }
    if document.profile_names.is_empty() {
        archive.remove(PROFILE_NAMES_PATH);
    } else {
        let names: std::collections::BTreeMap<CurveId, String> = document
            .profile_names
            .iter()
            .map(|(id, name)| (*id, name.clone()))
            .collect();
        archive.put_json(PROFILE_NAMES_PATH, &names)?;
    }

    // A cage deleted in the app has to leave the file too, or reopening resurrects it. Safe to
    // do by subtraction because `read_document` refuses any cage entry it could not read, so
    // an archive that has been opened contains no cage this build is unaware of.
    let stale: Vec<String> = archive
        .paths_under(CAGES_PREFIX)
        .filter(|path| !document.cages.keys().any(|id| cage_path(*id) == *path))
        .map(str::to_string)
        .collect();
    for path in stale {
        archive.remove(&path);
    }

    for (id, cage) in &document.cages {
        archive.put_json(cage_path(*id), cage)?;
    }

    // The same subtraction, for the same reason: a mesh deleted in the app has to leave the
    // file or reopening resurrects it.
    let stale: Vec<String> = archive
        .paths_under(MESHES_PREFIX)
        .filter(|path| !document.meshes.keys().any(|id| mesh_path(*id) == *path))
        .map(str::to_string)
        .collect();
    for path in stale {
        archive.remove(&path);
    }
    for (id, mesh) in &document.meshes {
        archive.put_json(mesh_path(*id), mesh)?;
    }

    // And once more for the surfaces. Three collections, one rule: what the document no longer
    // holds has to leave the file, or reopening resurrects it.
    let stale: Vec<String> = archive
        .paths_under(SURFACES_PREFIX)
        .filter(|path| {
            !document
                .quad_meshes
                .keys()
                .any(|id| surface_path(*id) == *path)
        })
        .map(str::to_string)
        .collect();
    for path in stale {
        archive.remove(&path);
    }
    for (id, mesh) in &document.quad_meshes {
        archive.put_json(surface_path(*id), mesh)?;
    }
    if document.surface_names.is_empty() {
        archive.remove(SURFACE_NAMES_PATH);
    } else {
        let names: std::collections::BTreeMap<_, _> = document
            .surface_names
            .iter()
            .map(|(id, name)| (*id, name.clone()))
            .collect();
        archive.put_json(SURFACE_NAMES_PATH, &names)?;
    }
    Ok(())
}

/// Reads a document out of an archive.
pub fn from_archive(archive: &Archive) -> Result<Document, FormatError> {
    let manifest = archive.manifest()?;
    let json: DocumentJson = archive
        .get_json(DOCUMENT_PATH)?
        .ok_or(FormatError::MissingDocument)?;

    let mut cages = std::collections::HashMap::new();
    for path in archive.paths_under(CAGES_PREFIX).map(str::to_string).collect::<Vec<_>>() {
        // A file under `cages/` that is not `<id>.json` is something a newer build put there.
        // Refused rather than skipped: skipping would mean saving a document whose cages this
        // build cannot see, and the geometry it names would come back empty.
        let id = path
            .strip_prefix(CAGES_PREFIX)
            .and_then(|name| name.strip_suffix(".json"))
            .and_then(CageId::from_hex)
            .ok_or_else(|| FormatError::UnexpectedEntry(path.clone()))?;
        let cage: Cage = archive.get_json(&path)?.expect("path came from the archive");
        cages.insert(id, cage);
    }

    let mut meshes = std::collections::HashMap::new();
    for path in archive.paths_under(MESHES_PREFIX).map(str::to_string).collect::<Vec<_>>() {
        // Refused rather than skipped, exactly as a cage entry is: skipping would save a
        // document whose meshes this build cannot see, and the geometry the graph names would
        // come back empty rather than loudly missing.
        let id = path
            .strip_prefix(MESHES_PREFIX)
            .and_then(|name| name.strip_suffix(".json"))
            .and_then(fieldform_scene::MeshId::from_hex)
            .ok_or_else(|| FormatError::UnexpectedEntry(path.clone()))?;
        let mesh: fieldform_scene::MeshData =
            archive.get_json(&path)?.expect("path came from the archive");
        meshes.insert(id, mesh);
    }

    let mut quad_meshes = std::collections::HashMap::new();
    for path in archive.paths_under(SURFACES_PREFIX).map(str::to_string).collect::<Vec<_>>() {
        // The names live under the same prefix and are not a surface. Skipped by name rather
        // than by shape, so a file that somehow holds neither still refuses below.
        if path == SURFACE_NAMES_PATH {
            continue;
        }
        // Refused rather than skipped, for the third time and the same reason. A surface a
        // conformal lattice names has to be there or the lattice has nothing to follow, and
        // silently dropping it would put an empty body on screen instead of an error.
        let id = path
            .strip_prefix(SURFACES_PREFIX)
            .and_then(|name| name.strip_suffix(".json"))
            .and_then(fieldform_scene::QuadMeshId::from_hex)
            .ok_or_else(|| FormatError::UnexpectedEntry(path.clone()))?;
        let mesh: fieldform_cage::QuadMesh =
            archive.get_json(&path)?.expect("path came from the archive");
        quad_meshes.insert(id, mesh);
    }

    // Absent in anything written before this existed, and absent again if an older build
    // saved over it — neither is an error, they just mean the order has to be worked out from
    // the graph instead.
    let created = archive
        .get_json::<std::collections::BTreeMap<fieldform_scene::NodeId, u64>>(CREATED_PATH)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    let arrangement = archive
        .get_json::<std::collections::BTreeMap<fieldform_scene::NodeId, u32>>(ARRANGEMENT_PATH)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    let node_meta = archive
        .get_json::<std::collections::BTreeMap<fieldform_scene::NodeId, fieldform_scene::NodeMeta>>(
            NODE_META_PATH,
        )?
        .unwrap_or_default()
        .into_iter()
        .collect();

    let tessellations = archive
        .get_json::<Vec<fieldform_scene::Tessellation>>(TESSELLATIONS_PATH)?
        .unwrap_or_default();
    let exports = archive
        .get_json::<Vec<fieldform_scene::Export>>(EXPORTS_PATH)?
        .unwrap_or_default();

    let curves_created = archive
        .get_json::<std::collections::BTreeMap<CurveId, u64>>(CURVES_CREATED_PATH)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    let profile_names = archive
        .get_json::<std::collections::BTreeMap<CurveId, String>>(PROFILE_NAMES_PATH)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    // Absent in every document written before 3.5, which reads as nothing declared — the state those
    // files were saved in.
    let declared_curves: std::collections::BTreeSet<CurveId> = archive
        .get_json::<Vec<CurveId>>(DECLARED_CURVES_PATH)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    // Absent in every document written before 3.9, which reads as no axis on a surface.
    let surface_axes: std::collections::BTreeMap<CurveId, fieldform_scene::SurfaceAxis> = archive
        .get_json::<std::collections::BTreeMap<CurveId, fieldform_scene::SurfaceAxis>>(
            SURFACE_AXES_PATH,
        )?
        .unwrap_or_default();
    let ungrouped_curves: std::collections::BTreeSet<CurveId> = archive
        .get_json::<Vec<CurveId>>(UNGROUPED_CURVES_PATH)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    let mut relations = std::collections::BTreeMap::new();
    for path in [RELATIONS_PATH, RELATIONS_ADDED_PATH] {
        relations.extend(
            archive
                .get_json::<std::collections::BTreeMap<
                    fieldform_scene::id::RelationId,
                    fieldform_scene::RelationRecord,
                >>(path)?
                .unwrap_or_default(),
        );
    }

    let surface_names = archive
        .get_json::<std::collections::BTreeMap<fieldform_scene::QuadMeshId, String>>(
            SURFACE_NAMES_PATH,
        )?
        .unwrap_or_default()
        .into_iter()
        .collect();
    // Absent in every document written before 3.15, and every block in it is placed by the rule
    // its list was laid out by: once the document is whole, since that rule reads it
    // (`place_ownership`).
    let ownership = archive.get_json::<OwnershipJson>(OWNERSHIP_PATH)?.unwrap_or_default();

    let mut document = Document {
        unit: manifest.unit,
        created,
        arrangement,
        node_meta,
        curves_created,
        profile_names,
        declared_curves,
        ungrouped_curves,
        surface_axes,
        relations,
        cages,
        quad_meshes,
        surface_names,
        tessellations,
        exports,
        meshes,
        curves: json.curves.into_iter().collect(),
        parameters: parameters_from_json(json.parameters)?,
        sources: json.sources.into_iter().collect(),
        parts: json.parts.into_iter().collect(),
        // Filled once everything else is read, by `place_ownership`.
        ownership_nodes: std::collections::HashMap::new(),
        ownership_curves: std::collections::HashMap::new(),
        pending_ownership_changes: Vec::new(),
    };
    // The editor refuses a cycle on the keystroke that would close it; a file can say one
    // anyway, and a document holding one cannot evaluate. Refused with the loop named, as the
    // editor would have named it.
    if let Err(cycle) = document.parameter_order() {
        let names: Vec<String> = cycle
            .iter()
            .map(|id| {
                document.parameters.get(id).map(|p| p.name.clone()).unwrap_or_else(|| id.to_string())
            })
            .collect();
        return Err(FormatError::Parameter {
            name: names[0].clone(),
            reason: format!("would loop: {}", names.join(" \u{2192} ")),
        });
    }
    place_ownership(&mut document, ownership);
    Ok(document)
}

/// Puts every block where the file says, places each block it leaves out by the rule the list was
/// laid out by before 3.15, then settles (`block-ownership-spec.md` §9.2, §9.3).
///
/// **A block the file leaves out is placed by the old rule, never as free**, though free is what
/// one reads as at run time. Here it means the writer did not know the block: the whole document,
/// if it was written before 3.15, or the blocks an older build added to a 3.15 file whose entry it
/// carried through. Either way the old rule is where that build showed them. Read as free, a named
/// block one other block reads would be adopted by the settle below, and the row would move on
/// opening.
///
/// The settle repairs what the file's word no longer bears out — an older build that deleted the
/// only block reading one leaves it nested in nothing — and what it moved is left on
/// `pending_ownership_changes` for the application, whose drain after a load says nothing and marks
/// nothing changed (§4.2). An entry for a block that is not here is kept: nothing is pruned but at
/// save.
fn place_ownership(document: &mut Document, file: OwnershipJson) {
    let (nodes, curves) = document.legacy_ownership();
    document.ownership_nodes = file.nodes.into_iter().collect();
    for (id, legacy) in nodes {
        document.ownership_nodes.entry(id).or_insert(legacy);
    }
    document.ownership_curves = file.curves.into_iter().collect();
    for (id, legacy) in curves {
        document.ownership_curves.entry(id).or_insert(legacy);
    }
    document.settle_ownership();
}

/// Opens a native document file, returning the document and the archive it came from.
///
/// **The extension plays no part.** The container says what it is; this reads the
/// archive. `.fform` is what the application writes and `.fieldform` is what it wrote
/// before 2026-09-13, and both arrive here as the same bytes.
///
/// The archive comes back with it because saving needs it: it carries the `ext/` and
/// `sources/` entries this build does not interpret, and hand it to [`update_archive`] to save
/// without losing them.
pub fn open(path: &Path) -> Result<(Document, Archive), FormatError> {
    let archive = Archive::read(path)?;
    let document = from_archive(&archive)?;
    Ok((document, archive))
}

/// Writes a document to a path that has no existing document to preserve.
pub fn save_new(document: &Document, path: &Path) -> Result<(), FormatError> {
    to_archive(document)?.write(path)
}

#[cfg(test)]
mod tests {
    use fieldform_scene::Fillet;
    use super::*;
    use fieldform_scene::{BooleanOp, Continuity, CurveId, Node, ParamRef, Unit};
    use nalgebra::Point3;

    /// **Export blocks survive a round trip, and a preset stays a preset.**
    ///
    /// The second half is the point. A preset is relative to what it meshes, so storing the
    /// millimetres it produced would freeze them and a body that grew would come back meshed at
    /// a detail that no longer meant Normal. `Custom` is the only detail that carries numbers,
    /// and it has to carry all four of them back.
    #[test]
    fn export_blocks_round_trip_and_a_preset_stays_a_preset() {
        let mut doc = Document::new();
        let part = fieldform_scene::PartId::new();
        let body = fieldform_scene::NodeId::new();
        let preset = fieldform_scene::Tessellation {
            id: fieldform_scene::TessellationId::new(),
            name: Some("Fine one".into()),
            part,
            body: Some(body),
            detail: fieldform_scene::MeshDetail::Full,
            improve: true,
            reduce: false,
            active: true,
        };
        let by_hand = fieldform_scene::Tessellation {
            id: fieldform_scene::TessellationId::new(),
            name: None,
            part,
            body: None,
            detail: fieldform_scene::MeshDetail::Custom(fieldform_scene::MeshTolerance {
                chord_mm: 0.07,
                angle_deg: 11.0,
                max_edge_mm: 4.0,
                min_cell_mm: 0.03,
            }),
            improve: false,
            reduce: true,
            active: false,
        };
        let export = fieldform_scene::Export {
            id: fieldform_scene::ExportId::new(),
            name: Some("to the printer".into()),
            source: preset.id,
            path: "C:/out/part.stl".into(),
            active: true,
        };
        doc.tessellations = vec![preset.clone(), by_hand.clone()];
        doc.exports = vec![export.clone()];

        let back = from_archive(&to_archive(&doc).expect("writes")).expect("reads");
        assert_eq!(back.tessellations, vec![preset, by_hand], "order and contents both");
        assert_eq!(back.exports, vec![export]);

        // And none at all writes no entry, so a document without them is the file it was.
        let archive = to_archive(&Document::new()).expect("writes");
        assert!(archive.paths_under("outputs/").next().is_none());
    }

    /// **An open surface survives a round trip, and comes back open.**
    ///
    /// The document's third geometry collection, added for item 46: a panel or a shell a
    /// conformal lattice follows. It serializes identically to a cage — positions and quads —
    /// and its openness stays *derived*, exactly as its half-edge topology does. So the test is
    /// not that the bytes match but that the thing reconstructed from them is still open, and
    /// still refuses to be a cage.
    #[test]
    fn an_open_surface_round_trips_and_is_still_open() {
        let mut doc = Document::new();
        let mut positions = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                positions.push(nalgebra::Point3::new(i as f64 * 10.0, j as f64 * 10.0, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * 3 + i) as u32;
        let faces: Vec<Vec<u32>> = (0..2)
            .flat_map(|j| {
                (0..2).map(move |i| vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
            })
            .collect();
        let panel = fieldform_cage::QuadMesh::from_faces(positions.clone(), &faces)
            .expect("a two-by-two panel");
        assert!(panel.is_open(), "the fixture has to be open or this tests nothing");

        let id = fieldform_scene::QuadMeshId::new();
        doc.quad_meshes.insert(id, panel);

        let archive = to_archive(&doc).expect("writes");
        assert!(
            archive.paths_under(SURFACES_PREFIX).count() == 1,
            "the surface should get its own entry, like a cage"
        );

        let back = from_archive(&archive).expect("reads");
        let returned = back.quad_meshes.get(&id).expect("the surface came back");
        assert!(returned.is_open(), "it came back closed, which it never was");
        assert_eq!(returned.face_count(), 4);
        assert_eq!(returned.edge_count(), 12, "counted, not halved");
        assert_eq!(returned.euler_characteristic(), 1, "still a disc");

        // And the same geometry is still refused as a cage, which is the rule the third
        // collection exists to avoid relaxing.
        assert!(
            fieldform_cage::Cage::from_faces(positions, &faces).is_err(),
            "an open panel must still not be a cage"
        );
    }

    /// A surface deleted in the app leaves the file, or reopening resurrects it — the same
    /// subtraction cages and meshes get.
    #[test]
    fn a_deleted_surface_leaves_the_archive() {
        let mut doc = Document::new();
        let panel = fieldform_cage::QuadMesh::from_faces(
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(1.0, 0.0, 0.0),
                nalgebra::Point3::new(1.0, 1.0, 0.0),
                nalgebra::Point3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3]],
        )
        .expect("one quad");
        let id = fieldform_scene::QuadMeshId::new();
        doc.quad_meshes.insert(id, panel);

        let mut archive = to_archive(&doc).expect("writes");
        assert_eq!(archive.paths_under(SURFACES_PREFIX).count(), 1);

        doc.quad_meshes.remove(&id);
        update_archive(&mut archive, &doc).expect("writes again");
        assert_eq!(
            archive.paths_under(SURFACES_PREFIX).count(),
            0,
            "the entry outlived the surface"
        );
    }

    /// A document with something of every kind in it: two cages, a curve, a named parameter
    /// driving a blend, and a graph that is not the trivial one-node case.
    fn populated() -> Document {
        let mut doc = Document::with_cage(Cage::cube_of(40.0));
        let second = doc.add_cage(Cage::cube_of(10.0));
        let radius = doc.add_parameter("fillet", 3.0);
        let curve = Curve::polyline(
            CurveId::new(),
            "spine",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(50.0, 0.0, 20.0)],
            false,
        )
        .unwrap();
        let curve_id = curve.id;
        doc.curves.insert(curve_id, curve);
        doc.unit = Unit::Inch;

        let part = doc.parts.values_mut().next().expect("with_cage made one");
        let cage_node = part.graph.roots()[0];
        let other = part.graph.add(Node::Cage(second));
        let pipe = part.graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(2.0),
        });
        let cut = part.graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![cage_node.into(), pipe.into()],
        });
        part.graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: cut.into(),
            b: other.into(),
            fillet: Fillet { radius: ParamRef::Named(radius), continuity: Continuity::G1 },
        });
        // A surface that must not move, with a tolerance that is not the default so that a
        // loader which rebuilt the mark rather than reading it would be caught.
        part.graph.mark_critical(fieldform_scene::Critical {
            tolerance: 0.025,
            ..fieldform_scene::Critical::new(pipe)
        });
        part.material = fieldform_scene::Material::preset("Brushed aluminium");
        // Settled, as every graph edit made through a command leaves a document. Built by hand it
        // would not be, and the first open would place its blocks — a document that reads back
        // differently from how it was written, which is not the document the tests here mean.
        doc.settle_ownership();
        doc
    }

    fn round_trip(document: &Document) -> Document {
        from_archive(&to_archive(document).unwrap()).unwrap()
    }

    /// A degree-9 curve saved earlier still opens.
    ///
    /// Degree 9 left the curve tools on 2026-09-24 (`profile-group-spec.md` §7), and a file
    /// written before that day may hold one. The tools stopped *offering* the degree; nothing
    /// stopped *reading* it, because a saved curve refused on open is work destroyed. This is
    /// the reading half; the editing half is
    /// `a_degree_nine_curve_saved_earlier_still_opens_and_edits` in the application.
    #[test]
    fn a_degree_nine_curve_saved_earlier_still_opens() {
        let mut doc = populated();
        let id = CurveId::new();
        let points: Vec<Point3<f64>> =
            (0..10).map(|i| Point3::new(i as f64 * 5.0, (i as f64).sin() * 3.0, 0.0)).collect();
        let knots: Vec<f64> = std::iter::repeat_n(0.0, 10).chain(std::iter::repeat_n(1.0, 10)).collect();
        let nine = Curve::new(id, "Curve 9", 9, points.clone(), vec![1.0; 10], knots, false)
            .expect("a degree-9 Bézier is still a well-formed curve");
        doc.curves.insert(id, nine);

        let back = round_trip(&doc);
        let read = back.curves.get(&id).expect("the curve is still in the document");
        assert_eq!(read.degree(), 9);
        assert_eq!(read.control_points(), points.as_slice());
        assert_eq!(read.name, "Curve 9");
    }

    #[test]
    fn a_dismissed_grouping_survives_a_save_and_none_writes_no_entry() {
        let mut doc = populated();
        let archive = to_archive(&doc).unwrap();
        assert!(
            archive.get_json::<serde_json::Value>(UNGROUPED_CURVES_PATH).unwrap().is_none(),
            "nothing dismissed, so nothing written — which is what keeps 3.6 additive",
        );

        let id = *doc.curves.keys().next().expect("the fixture holds a curve");
        doc.ungrouped_curves.insert(id);
        let back = round_trip(&doc);
        assert!(
            back.ungrouped_curves.contains(&id),
            "the dismissal came back, which is the whole point of storing it",
        );

        let mut cleared = back;
        cleared.ungrouped_curves.clear();
        let archive = to_archive(&cleared).unwrap();
        assert!(
            archive.get_json::<serde_json::Value>(UNGROUPED_CURVES_PATH).unwrap().is_none(),
            "grouped again, so the entry is gone rather than left empty",
        );
    }

    /// **A surface axis survives a save, and a document with none writes no entry** — the same
    /// bargain as a declared curve, which is what makes 3.9 additive: a document nobody has stood an
    /// axis on a surface in is byte for byte what 3.7 wrote.
    #[test]
    fn a_surface_axis_survives_a_save_and_none_writes_no_entry() {
        use fieldform_scene::{Axis, NodeId, SurfaceAxis};
        let mut doc = populated();
        let archive = to_archive(&doc).unwrap();
        assert!(
            archive.get_json::<serde_json::Value>(SURFACE_AXES_PATH).unwrap().is_none(),
            "no axis on a surface, so nothing written",
        );

        let id = *doc.curves.keys().next().expect("the fixture holds a curve");
        let axis = SurfaceAxis {
            body: NodeId::from_raw(7),
            direction: Axis::Y,
            flipped: true,
            u: 0.25,
            v: 0.75,
            length: 12.5,
            // Seeded, so the round trip covers the field 3.10 added rather than only the ones it
            // already had. An axis without one is the other half and is what every 3.9 file is.
            seed: Some([0.1, 0.2, 0.3]),
        };
        doc.surface_axes.insert(id, axis);
        let back = round_trip(&doc);
        assert_eq!(back.surface_axes.get(&id), Some(&axis), "the axis came back whole");

        let mut cleared = back;
        cleared.surface_axes.clear();
        let archive = to_archive(&cleared).unwrap();
        assert!(
            archive.get_json::<serde_json::Value>(SURFACE_AXES_PATH).unwrap().is_none(),
            "cleared, so the entry is gone again",
        );

        // And a record whose curve has been deleted is not carried into the file.
        let mut orphaned = round_trip(&doc);
        orphaned.curves.remove(&id);
        assert!(!orphaned.surface_axes.is_empty(), "the record outlives the curve in the document");
        let archive = to_archive(&orphaned).unwrap();
        assert!(
            archive.get_json::<serde_json::Value>(SURFACE_AXES_PATH).unwrap().is_none(),
            "but a saved file does not keep it",
        );
    }

    /// **A curve made a variable is declared for an older build, and a document with none writes
    /// no entry.**
    ///
    /// The second half is what makes 3.5 additive rather than a break: a document nobody has declared
    /// a curve in is byte for byte what 3.4 wrote, so a 3.4 build opens it unchanged. A document that
    /// *does* declare one is still drawn identically by an older build — a declaration moves a row and
    /// no geometry — and the entry is carried through byte-intact because it is an entry of its own
    /// rather than a field in a table that denies what it does not know.
    ///
    /// The declaration is the ownership's now (refactor plan 2.19): the curve row's V makes a
    /// variable, and the entry says so to a build that reads no ownership. So put back inside, the
    /// curve is not declared again, though the file it was opened from declared it.
    #[test]
    fn a_curve_made_a_variable_is_declared_for_an_older_build_and_none_writes_no_entry() {
        let mut doc = populated();
        doc.settle_ownership();
        let archive = to_archive(&doc).unwrap();
        assert!(
            archive.get_json::<serde_json::Value>(DECLARED_CURVES_PATH).unwrap().is_none(),
            "nothing declared, so nothing written",
        );

        let id = *doc.curves.keys().next().expect("the fixture holds a curve");
        let reader = doc.block_readers(BlockKey::Curve(id));
        assert_eq!(reader.len(), 1, "the fixture's pipe reads it, so it is inside the pipe");
        doc.ownership_curves.insert(id, Ownership::Variable);
        let back = round_trip(&doc);
        assert!(back.declared_curves.contains(&id), "the declaration came back");
        assert_eq!(back.ownership_curves.get(&id), Some(&Ownership::Variable), "and the variable");

        // And putting it back takes the entry back out, rather than leaving an empty one behind.
        let mut put_back = back;
        put_back.ownership_curves.insert(id, Ownership::Nested);
        let archive = to_archive(&put_back).unwrap();
        assert!(
            archive.get_json::<serde_json::Value>(DECLARED_CURVES_PATH).unwrap().is_none(),
            "put back, so the entry is gone again",
        );
    }

    /// A profile's name survives a save, and a document with none writes no entry.
    ///
    /// The name is the one stored piece of a profile row (`profile-group-spec.md` §3.3); the
    /// row itself is derived. A document that names nothing must write nothing, so its file is
    /// byte for byte what it was before names existed.
    #[test]
    fn a_profile_name_survives_a_save_and_none_writes_no_entry() {
        let mut doc = populated();
        let archive = to_archive(&doc).unwrap();
        assert!(archive.get_json::<serde_json::Value>(PROFILE_NAMES_PATH).unwrap().is_none());

        let key = *doc.curves.keys().next().expect("the fixture holds a curve");
        doc.profile_names.insert(key, "Wing section".to_string());
        let back = round_trip(&doc);
        assert_eq!(back.profile_names.get(&key).map(String::as_str), Some("Wing section"));
    }

    /// A document with no relations writes no relations entry.
    #[test]
    fn a_document_with_no_relations_writes_no_relations_entry() {
        let doc = populated();
        let archive = to_archive(&doc).unwrap();
        assert!(archive.get(RELATIONS_PATH).is_none());
    }

    /// Relations of a kind added after 3.2 ride in their own entry, so a 3.2 reader -- which
    /// reads `relations.json` strictly -- still opens the file and carries them through.
    #[test]
    fn a_relation_of_a_newer_kind_rides_in_its_own_entry() {
        use fieldform_scene::{Relation, RelationRecord};
        let mut doc = populated();
        let (a, b, axis) = (CurveId::new(), CurveId::new(), CurveId::new());
        let record = |relation| RelationRecord { relation, broken: None };
        doc.relations.insert(
            fieldform_scene::id::RelationId::new(),
            record(Relation::Parallel { master: a, slave: b }),
        );
        doc.relations.insert(
            fieldform_scene::id::RelationId::new(),
            record(Relation::Mirror { master: a, slave: b, axis }),
        );
        assert_eq!(round_trip(&doc).relations, doc.relations);
        let archive = to_archive(&doc).unwrap();
        let text = |path| String::from_utf8(archive.get(path).expect("written").to_vec()).unwrap();
        let older = text(RELATIONS_PATH);
        assert!(older.contains("parallel") && !older.contains("mirror"), "only what 3.2 knows");
        assert!(text(RELATIONS_ADDED_PATH).contains("mirror"));
    }

    /// Relations round-trip, and ride through an older reader untouched.
    ///
    /// The second half is the `format.rs` promise applied to this entry: a build that does not
    /// read `history/relations.json` carries its bytes through a save, so a document edited
    /// there comes back to a newer build with its relations intact -- to be checked, not
    /// enforced. Here the same bytes go through `update_archive` unchanged, which is the path
    /// such a save takes.
    #[test]
    fn relations_round_trip_through_an_older_reader_untouched() {
        use fieldform_scene::{CurveEnd, End, JointContinuity, Relation, RelationRecord};
        let mut doc = populated();
        let (master, slave) = (CurveId::new(), CurveId::new());
        let id = fieldform_scene::id::RelationId::new();
        doc.relations.insert(
            id,
            RelationRecord {
                relation: Relation::Joint {
                    master: CurveEnd { curve: master, end: End::End },
                    slave: CurveEnd { curve: slave, end: End::Start },
                    continuity: JointContinuity::G2,
                },
                broken: Some("point 1 of Slave is 0.420 mm from where it should be".into()),
            },
        );
        let back = round_trip(&doc);
        assert_eq!(back.relations, doc.relations);

        let mut archive = to_archive(&doc).unwrap();
        let bytes = archive.get(RELATIONS_PATH).expect("written").to_vec();
        update_archive(&mut archive, &doc).unwrap();
        assert_eq!(archive.get(RELATIONS_PATH).expect("kept"), bytes.as_slice());
    }

    /// Creation times are the whole reason they are written down: the order a person made
    /// things in is nowhere else in the document, so if it does not survive a save it may as
    /// well not exist. Ids are random, and the graph is a set — reopening a file would put the
    /// blocks back in an order nobody chose.
    #[test]
    fn creation_times_survive_a_save() {
        let mut doc = populated();
        let ids: Vec<fieldform_scene::NodeId> = doc
            .parts
            .values()
            .flat_map(|p| p.graph.nodes().map(|(id, _)| id))
            .collect();
        assert!(!ids.is_empty(), "the fixture needs a node to stamp");
        for (n, id) in ids.iter().enumerate() {
            doc.created.insert(*id, 1_700_000_000_000 + n as u64);
        }

        let back = round_trip(&doc);
        assert_eq!(back.created, doc.created);
    }

    /// An arrangement survives a save for the same reason a creation time does: where a person
    /// dragged a block to is nowhere else in the document, so a list that comes back in a
    /// different order than it was left in is a list that was not saved.
    #[test]
    fn an_arrangement_survives_a_save() {
        let mut doc = populated();
        let ids: Vec<fieldform_scene::NodeId> = doc
            .parts
            .values()
            .flat_map(|p| p.graph.nodes().map(|(id, _)| id))
            .collect();
        assert!(!ids.is_empty(), "the fixture needs a node to arrange");
        for (n, id) in ids.iter().enumerate() {
            doc.arrangement.insert(*id, n as u32);
        }

        let back = round_trip(&doc);
        assert_eq!(back.arrangement, doc.arrangement);
    }

    /// **And it rides beside `document.json` rather than inside `NodeMeta`, which would have been
    /// the natural home and is the one that breaks older builds.**
    ///
    /// `NodeMeta` is `deny_unknown_fields` and is loaded with `get_json(..)?`, so a field added to
    /// it is a failed document load in a build that predates it — not a note quietly ignored. This
    /// asserts the arrangement is nowhere near either JSON that refuses what it does not know.
    #[test]
    fn an_arrangement_rides_in_its_own_entry_so_an_older_build_can_ignore_it() {
        let mut doc = populated();
        let id = doc
            .parts
            .values()
            .flat_map(|p| p.graph.nodes().map(|(n, _)| n))
            .next()
            .expect("the fixture has a node");
        doc.arrangement.insert(id, 3);

        let archive = to_archive(&doc).unwrap();
        assert!(archive.get(ARRANGEMENT_PATH).is_some(), "the entry should be written");

        let document_json = archive.get(DOCUMENT_PATH).expect("a document is written");
        let text = std::str::from_utf8(document_json).unwrap();
        assert!(
            !text.contains("arrangement"),
            "an arrangement must not be inside document.json, which denies unknown fields",
        );
        let meta = archive.get(NODE_META_PATH);
        assert!(
            meta.is_none_or(|bytes| !std::str::from_utf8(bytes).unwrap().contains("arrangement")),
            "nor inside node-meta.json, which also denies them",
        );
    }

    /// A document nobody has rearranged writes no entry at all, like one nobody has stamped.
    #[test]
    fn a_document_nobody_has_rearranged_writes_no_entry() {
        let doc = populated();
        assert!(doc.arrangement.is_empty(), "the fixture arranges nothing");
        let archive = to_archive(&doc).unwrap();
        assert!(archive.get(ARRANGEMENT_PATH).is_none(), "no entry for nothing to say");
        assert!(round_trip(&doc).arrangement.is_empty());
    }

    /// The entry is not part of `document.json`, and that is deliberate: the document JSON
    /// refuses fields it does not know, so a build without this would have refused the whole
    /// file rather than ignoring a note beside it. Carried as its own entry, it falls under the
    /// rule that anything unrecognised is preserved untouched.
    #[test]
    fn creation_times_ride_in_their_own_entry_so_an_older_build_can_ignore_them() {
        let mut doc = populated();
        let id = doc
            .parts
            .values()
            .flat_map(|p| p.graph.nodes().map(|(n, _)| n))
            .next()
            .expect("the fixture has a node");
        doc.created.insert(id, 1_700_000_000_000);

        let archive = to_archive(&doc).unwrap();
        assert!(archive.get(CREATED_PATH).is_some(), "the entry should be written");

        let document_json = archive.get(DOCUMENT_PATH).expect("a document is written");
        let text = std::str::from_utf8(document_json).unwrap();
        assert!(
            !text.contains("created"),
            "creation times must not be inside document.json, which denies unknown fields",
        );
    }

    /// A document nobody has stamped writes no entry at all, rather than an empty one — an
    /// entry that appears in every file to say nothing is noise in every diff.
    #[test]
    fn a_document_with_no_creation_times_writes_no_entry() {
        let doc = populated();
        assert!(doc.created.is_empty());
        let archive = to_archive(&doc).unwrap();
        assert!(archive.get(CREATED_PATH).is_none());
        // And reading one back is ordinary, not an error.
        assert!(round_trip(&doc).created.is_empty());
    }

    /// A body's name and whether it is hidden survive a save, and an empty slot writes nothing.
    ///
    /// The second half is what keeps the file honest. A body nobody has named or hidden must
    /// leave no trace, and one that is named and then un-named must leave no trace either —
    /// otherwise a document put back the way it was saves different bytes from the one it was,
    /// and every diff carries noise nobody made.
    #[test]
    fn a_bodys_name_and_visibility_survive_a_save_and_an_empty_one_writes_nothing() {
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube());
        let part = *doc.parts.keys().next().expect("with_cage inserts one");
        let body = *doc.parts[&part].graph.roots().first().expect("the cage is a body");

        doc.node_meta.insert(
            body,
            fieldform_scene::NodeMeta {
                name: Some("left bracket".into()),
                hidden: true,
                shown: false,
            },
        );
        let back = round_trip(&doc);
        let meta = back.node_meta.get(&body).expect("the slot survived");
        assert_eq!(meta.name.as_deref(), Some("left bracket"));
        assert!(meta.hidden);

        // Emptied, and the entry goes with it.
        doc.node_meta.insert(body, fieldform_scene::NodeMeta::default());
        let archive = crate::to_archive(&doc).expect("writes");
        assert!(
            archive.get(NODE_META_PATH).is_none(),
            "an empty slot should leave no entry behind",
        );
        assert!(round_trip(&doc).node_meta.is_empty());
    }

    /// A sphere in a part, consumed by `consumer`, and shown the way the app shows one: a slot
    /// holding `shown` and nothing else, no name.
    fn shown_and_consumed(
        consumer: impl FnOnce(fieldform_scene::NodeId) -> Node,
    ) -> (Document, fieldform_scene::NodeId) {
        let mut doc = Document::with_cage(Cage::cube_of(40.0));
        let graph = &mut doc.parts.values_mut().next().expect("with_cage made one").graph;
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::origin(),
            radius: ParamRef::Literal(10.0),
        });
        graph.add(consumer(sphere));
        assert!(!graph.roots().contains(&sphere), "the fixture's sphere has to be consumed");
        let shown = fieldform_scene::NodeMeta { shown: true, ..Default::default() };
        doc.node_meta.insert(sphere, shown);
        (doc, sphere)
    }

    /// **A block nobody named, drawn because a pattern consumed it, still draws after a reopen.**
    ///
    /// Pattern and Mirror leave the body they consume shown and unnamed: it was on screen a moment
    /// before, and a name is the owner's to give, not the operation's. The save writes only the
    /// slots `is_empty` calls non-empty, and `is_empty` used to read the name and `hidden` but
    /// never `shown` — so this slot was dropped, and the original no longer drew once the file was
    /// opened again (block-ownership-spec §1, §12 test 14).
    #[test]
    fn a_shown_block_nobody_named_survives_a_save() {
        let (doc, sphere) = shown_and_consumed(|input| Node::Pattern {
            options: Default::default(),
            input: input.into(),
            count: ParamRef::Literal(4.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [ParamRef::Literal(30.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
            },
            join: false,
            fillet: Fillet::hard(),
        });
        let back = round_trip(&doc);
        assert_eq!(
            back.node_meta.get(&sphere),
            doc.node_meta.get(&sphere),
            "the pattern's source stopped drawing",
        );
    }

    /// **The eye on a block inside another one survives a save too.** The same slot as a pattern's
    /// source, written by a person rather than by an operation: the eye on an unnamed operand sets
    /// `shown` and names nothing, because seeing a block is not declaring it a variable.
    #[test]
    fn the_eye_on_an_unnamed_consumed_block_survives_a_save() {
        let (doc, sphere) = shown_and_consumed(|input| Node::Offset {
            input: input.into(),
            distance: ParamRef::Literal(2.0),
        });
        let back = round_trip(&doc);
        assert_eq!(
            back.node_meta.get(&sphere),
            doc.node_meta.get(&sphere),
            "the operand stopped drawing",
        );
    }

    /// Everything in a document survives being written and read back.
    ///
    /// **Was not running.** It lost its `#[test]` at some point and became an ordinary private
    /// function nothing called, which the compiler reports as dead code and nothing else does
    /// -- a test that has stopped running looks exactly like a test that passes.
    #[test]
    fn a_document_round_trips_through_an_archive() {
        let doc = populated();
        let back = round_trip(&doc);

        assert_eq!(back.unit, doc.unit);
        assert_eq!(back.cages.len(), doc.cages.len());
        for (id, cage) in &doc.cages {
            let loaded = back.cages.get(id).expect("cage kept its id");
            assert_eq!(loaded.face_count(), cage.face_count());
            for v in cage.vertex_ids() {
                assert_eq!(loaded.vertex_position(v), cage.vertex_position(v));
            }
        }
        assert_eq!(back.curves, doc.curves);
        assert_eq!(back.parameters, doc.parameters);

        let (id, part) = doc.parts.iter().next().unwrap();
        let loaded = back.parts.get(id).expect("part kept its id");
        assert_eq!(loaded.name, part.name);
        assert_eq!(loaded.material, part.material);
        assert_eq!(loaded.graph.roots()[0], part.graph.roots()[0]);
        for (node_id, node) in part.graph.nodes() {
            assert_eq!(loaded.graph.node(node_id), Some(node));
        }
        assert_eq!(loaded.graph.critical().len(), 1);
        assert_eq!(loaded.graph.critical(), part.graph.critical());
    }

    #[test]
    fn the_same_document_writes_the_same_bytes_every_time() {
        // What makes a `.fform` diffable, and what makes "did this change?" answerable
        // without parsing: nothing in the file may depend on hash iteration order.
        let doc = populated();
        let mut buffer = Vec::new();
        to_archive(&doc)
            .unwrap()
            .write_to(std::io::Cursor::new(&mut buffer))
            .unwrap();
        for _ in 0..8 {
            let mut again = Vec::new();
            to_archive(&round_trip(&doc))
                .unwrap()
                .write_to(std::io::Cursor::new(&mut again))
                .unwrap();
            assert_eq!(again, buffer, "a re-saved document differs byte for byte");
        }
    }

    #[test]
    fn saving_preserves_third_party_data_this_build_cannot_read() {
        // The property the whole container exists for, at document level: a user without the
        // plugin opens the file, edits the geometry, saves, and the plugin's bytes survive.
        let doc = populated();
        let mut archive = to_archive(&doc).unwrap();
        let plugin = br#"{"acme":"lattice parameters this build never parses"}"#.to_vec();
        let plugin_path = format!("{}acme.lattice.1/data.json", crate::format::EXTENSIONS_PREFIX);
        archive.put(plugin_path.as_str(), plugin.clone());
        archive.put_source("bracket.obj", b"v 0 0 0\n".to_vec());

        let (mut reopened, mut archive) = {
            let mut bytes = Vec::new();
            archive.write_to(std::io::Cursor::new(&mut bytes)).unwrap();
            let archive = Archive::read_from(std::io::Cursor::new(&bytes)).unwrap();
            let document = from_archive(&archive).unwrap();
            (document, archive)
        };

        // Edit the geometry the way the app would, then save through the same archive.
        let cage_id = *reopened.cages.keys().next().unwrap();
        reopened
            .cages
            .get_mut(&cage_id)
            .unwrap()
            .set_vertex_position(fieldform_cage::VertexId::from_index(0), Point3::new(1.0, 2.0, 3.0));
        update_archive(&mut archive, &reopened).unwrap();

        assert_eq!(archive.get(&plugin_path), Some(plugin.as_slice()));
        assert_eq!(archive.get("sources/bracket.obj"), Some(b"v 0 0 0\n".as_slice()));
        let back = from_archive(&archive).unwrap();
        assert_eq!(
            back.cages[&cage_id].vertex_position(fieldform_cage::VertexId::from_index(0)),
            Point3::new(1.0, 2.0, 3.0),
            "the edit was saved too"
        );
    }

    #[test]
    fn a_deleted_cage_does_not_come_back_on_reopen() {
        let mut doc = populated();
        let mut archive = to_archive(&doc).unwrap();
        let doomed = *doc.cages.keys().next().unwrap();
        doc.cages.remove(&doomed);
        update_archive(&mut archive, &doc).unwrap();

        assert!(archive.get(&cage_path(doomed)).is_none(), "its entry is gone");
        assert!(!from_archive(&archive).unwrap().cages.contains_key(&doomed));
    }

    #[test]
    fn a_document_written_by_a_newer_build_is_refused_rather_than_loaded_without_the_part_it_added()
    {
        // Everything outside the graph is refused-not-guessed. Loading a part while dropping
        // a field this build does not know would write that loss back over the user's file on
        // the next save, which is the one outcome the format rules out.
        let doc = populated();
        let mut archive = to_archive(&doc).unwrap();
        let mut json: serde_json::Value =
            serde_json::from_slice(archive.get(DOCUMENT_PATH).unwrap()).unwrap();
        json["parts"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("finish".to_string(), serde_json::json!("anodised"));
        archive.put_json(DOCUMENT_PATH, &json).unwrap();

        let error = from_archive(&archive).unwrap_err().to_string();
        assert!(error.contains("finish"), "the message names what it could not read: {error}");
    }

    #[test]
    fn an_unrecognised_node_survives_a_save_by_a_build_that_cannot_evaluate_it() {
        // The marketplace case end to end, through the file rather than through serde.
        let doc = populated();
        let mut archive = to_archive(&doc).unwrap();
        let mut json: serde_json::Value =
            serde_json::from_slice(archive.get(DOCUMENT_PATH).unwrap()).unwrap();
        let part = json["parts"].as_object_mut().unwrap().values_mut().next().unwrap();
        let node_id = fieldform_scene::NodeId::new().to_string();
        part["graph"]["nodes"].as_object_mut().unwrap().insert(
            node_id.clone(),
            serde_json::json!({"kind": "acme.lattice", "cell": 5.0, "wall": 1.0}),
        );
        archive.put_json(DOCUMENT_PATH, &json).unwrap();

        let reopened = from_archive(&archive).unwrap();
        let mut saved = archive.clone();
        update_archive(&mut saved, &reopened).unwrap();

        let written: serde_json::Value =
            serde_json::from_slice(saved.get(DOCUMENT_PATH).unwrap()).unwrap();
        let part = written["parts"].as_object().unwrap().values().next().unwrap();
        assert_eq!(
            part["graph"]["nodes"][&node_id],
            serde_json::json!({"kind": "acme.lattice", "cell": 5.0, "wall": 1.0}),
            "a node this build cannot read came back exactly as it went in"
        );
    }

    #[test]
    fn a_file_with_no_document_entry_is_not_a_document() {
        let mut archive = Archive::new(&Manifest::default());
        archive.put("cages/deadbeefdeadbeef.json", b"{}".to_vec());
        assert!(matches!(
            from_archive(&archive),
            Err(FormatError::MissingDocument)
        ));
    }

    #[test]
    fn an_unexpected_entry_under_cages_is_refused() {
        let doc = populated();
        let mut archive = to_archive(&doc).unwrap();
        archive.put("cages/index.json", b"{}".to_vec());
        assert!(matches!(from_archive(&archive), Err(FormatError::UnexpectedEntry(_))));
    }

    #[test]
    fn a_failed_save_leaves_the_previous_document_intact() {
        // The reason writing goes via a temporary: a save that dies halfway must not take the
        // last good version of the user's work with it.
        let dir = std::env::temp_dir().join(format!("fieldform-atomic-{}", PartId::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bracket.fform");

        let doc = populated();
        save_new(&doc, &path).unwrap();
        let good = std::fs::read(&path).unwrap();

        // A directory where the temporary file wants to be: `File::create` fails, and the
        // question is what happened to `path`.
        let blocked = std::path::PathBuf::from({
            let mut name = path.as_os_str().to_os_string();
            name.push(".saving");
            name
        });
        std::fs::create_dir(&blocked).unwrap();
        assert!(save_new(&doc, &path).is_err(), "the save could not have succeeded");
        assert_eq!(std::fs::read(&path).unwrap(), good, "the old document is untouched");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_document_survives_a_trip_through_an_actual_file() {
        let dir = std::env::temp_dir().join(format!("fieldform-doc-{}", PartId::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bracket.fform");

        let doc = populated();
        save_new(&doc, &path).unwrap();
        let (back, _archive) = open(&path).unwrap();
        assert_eq!(back.unit, doc.unit);
        assert_eq!(back.parts.len(), doc.parts.len());
        assert_eq!(back.cages.len(), doc.cages.len());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The file holds the text, and it comes back as the same tree worth the same number.
    #[test]
    fn an_expression_parameter_round_trips_as_its_text_and_evaluates_the_same() {
        let mut doc = populated();
        let radius = doc.add_parameter("radius", 5.0);
        let boss_r = doc.add_parameter("boss_r", 0.0);
        doc.set_parameter_expression(boss_r, "radius * 5 - 2").unwrap();

        let archive = to_archive(&doc).unwrap();
        let text = std::str::from_utf8(archive.get(DOCUMENT_PATH).unwrap()).unwrap();
        assert!(
            text.contains("\"expression\"") && text.contains("radius * 5 - 2"),
            "the file holds the expression as text: {text}"
        );

        let back = from_archive(&archive).unwrap();
        assert_eq!(back.parameters, doc.parameters);
        assert_eq!(back.parameter_value(boss_r), Some(23.0));
        assert_eq!(back.parameters_reading(radius), vec![boss_r]);
    }

    /// D3: a reference to nothing is refused and reported, never read as zero.
    #[test]
    fn an_expression_naming_a_parameter_the_file_does_not_have_refuses_to_load() {
        let mut doc = populated();
        let _radius = doc.add_parameter("radius", 5.0);
        let boss_r = doc.add_parameter("boss_r", 0.0);
        doc.set_parameter_expression(boss_r, "radius * 5 - 2").unwrap();

        let mut archive = to_archive(&doc).unwrap();
        let mut json: DocumentJson = archive.get_json(DOCUMENT_PATH).unwrap().unwrap();
        json.parameters.get_mut(&boss_r).unwrap().value =
            ParamValueJson::Expression("ghost * 5 - 2".into());
        archive.put_json(DOCUMENT_PATH, &json).unwrap();

        let err = from_archive(&archive).unwrap_err();
        assert!(
            matches!(&err, FormatError::Parameter { name, reason }
                if name == "boss_r" && reason.contains("`ghost` is not a parameter")),
            "{err}"
        );
    }

    #[test]
    fn a_file_whose_parameters_loop_refuses_to_load() {
        let mut doc = populated();
        let a = doc.add_parameter("a", 1.0);
        let b = doc.add_parameter("b", 1.0);
        doc.set_parameter_expression(a, "b + 1").unwrap();

        let mut archive = to_archive(&doc).unwrap();
        // The editor refuses `b = a + 1`; a file can say it anyway.
        let mut json: DocumentJson = archive.get_json(DOCUMENT_PATH).unwrap().unwrap();
        json.parameters.get_mut(&b).unwrap().value = ParamValueJson::Expression("a + 1".into());
        archive.put_json(DOCUMENT_PATH, &json).unwrap();

        let err = from_archive(&archive).unwrap_err();
        assert!(
            matches!(&err, FormatError::Parameter { reason, .. } if reason.contains("would loop")),
            "{err}"
        );
        assert!(err.to_string().contains("a \u{2192} b \u{2192} a") || err.to_string().contains("b \u{2192} a \u{2192} b"), "{err}");
    }

    #[test]
    fn two_parameters_with_one_name_are_refused_once_an_expression_could_mean_either() {
        let mut doc = populated();
        let a = doc.add_parameter("twin", 1.0);
        let _b = doc.add_parameter("twin", 2.0);
        assert!(from_archive(&to_archive(&doc).unwrap()).is_ok(), "plain numbers still open");

        let c = doc.add_parameter("c", 0.0);
        doc.set_parameter_expression(c, "1 + 1").unwrap();
        doc.parameters.get_mut(&a).unwrap().name = "twin".into();
        let err = from_archive(&to_archive(&doc).unwrap()).unwrap_err();
        assert!(
            matches!(&err, FormatError::Parameter { name, .. } if name == "twin"),
            "{err}"
        );
    }

    // Where each block sits in the list, format 3.15 (`block-ownership-spec.md` §9, §12).

    fn ball(graph: &mut fieldform_scene::Graph) -> NodeId {
        graph.add(Node::Sphere {
            centre: fieldform_scene::Place::origin(),
            radius: ParamRef::Literal(5.0),
        })
    }

    fn union(graph: &mut fieldform_scene::Graph, inputs: Vec<NodeId>) -> NodeId {
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        })
    }

    fn offset(graph: &mut fieldform_scene::Graph, input: NodeId) -> NodeId {
        graph.add(Node::Offset { input: input.into(), distance: ParamRef::Literal(2.0) })
    }

    fn pipe(graph: &mut fieldform_scene::Graph, curve: CurveId) -> NodeId {
        graph.add(Node::Pipe { curve, radius: ParamRef::Literal(1.0) })
    }

    /// A two-point line, put in `doc`.
    fn line(doc: &mut Document, name: &str) -> CurveId {
        let curve = Curve::polyline(
            CurveId::new(),
            name,
            vec![Point3::origin(), Point3::new(10.0, 0.0, 0.0)],
            false,
        )
        .expect("a two-point line");
        let id = curve.id;
        doc.curves.insert(id, curve);
        id
    }

    /// A document of one part with nothing in it yet, and that part.
    fn one_part() -> (Document, PartId) {
        let mut doc = Document::new();
        let part = doc.add_part(Part {
            id: PartId::new(),
            name: "Part".into(),
            material: None,
            graph: fieldform_scene::Graph::empty(),
            inputs: Vec::new(),
        });
        (doc, part)
    }

    fn graph_of(doc: &mut Document, part: PartId) -> &mut fieldform_scene::Graph {
        &mut doc.parts.get_mut(&part).expect("the part is there").graph
    }

    fn name(doc: &mut Document, id: NodeId, to: &str) {
        doc.node_meta.entry(id).or_default().name = Some(to.to_string());
    }

    /// The manifest a build from before 3.15 writes: the same, but for its own minor.
    fn manifest_before_315() -> Manifest {
        Manifest { format_minor: 14, ..Manifest::default() }
    }

    /// What a build from before 3.15 would have saved `doc` as: every entry this build writes but
    /// the one that build had never heard of, which `a_file_a_314_build_reads_draws_what_315_drew`
    /// holds this build to; and the curves it declared, which it kept in a flag of their own
    /// (`Document::declared_curves`) rather than reading them off an ownership it did not have.
    fn as_saved_before_315(doc: &Document) -> Archive {
        let mut archive = to_archive(doc).unwrap();
        archive.remove(OWNERSHIP_PATH);
        written_as_an_older_build_declares(&mut archive, doc);
        archive.put_json(crate::format::MANIFEST_PATH, &manifest_before_315()).unwrap();
        archive
    }

    /// `history/declared-curves.json` as a build from before 3.15 writes it: its own flag, and no
    /// entry when that is empty.
    fn written_as_an_older_build_declares(archive: &mut Archive, doc: &Document) {
        if doc.declared_curves.is_empty() {
            archive.remove(DECLARED_CURVES_PATH);
        } else {
            let declared: Vec<CurveId> = doc.declared_curves.iter().copied().collect();
            archive.put_json(DECLARED_CURVES_PATH, &declared).unwrap();
        }
    }

    /// The blocks that draw, by the two flags the application's rule reads (spec §7.2): a body
    /// unless it is hidden, and a block something consumes only when it is shown.
    fn what_draws(doc: &Document) -> std::collections::BTreeSet<NodeId> {
        let mut drawn = std::collections::BTreeSet::new();
        for part in doc.parts.values() {
            let roots = part.graph.roots();
            for (id, _) in part.graph.nodes() {
                let meta = doc.node_meta.get(&id).cloned().unwrap_or_default();
                let draws = if roots.contains(&id) { !meta.hidden } else { meta.shown };
                if draws {
                    drawn.insert(id);
                }
            }
        }
        drawn
    }

    /// **Where every block sits survives a save, and the file names every block there is and none
    /// that has gone** (spec §9.1, invariant I5).
    ///
    /// The cases that tell a reader of the file from a reader of the old rule are in it: a block
    /// named while inside another, which a name used to lift out, and a curve made a variable while
    /// one block reads it, which the old rule would put inside that block. The file's word has to
    /// win over the rule for both, or a 3.15 file opens with the rows of a 3.14 one.
    ///
    /// And the two the writer has to settle for itself. A block no settle has reached yet has no
    /// entry, which reads as free at run time, so it is written free. An entry for a block an edit
    /// removed is kept by the document, because an undo can bring the block back, but a saved file
    /// has no undo to serve, so it is left out.
    #[test]
    fn ownership_survives_a_save_and_the_file_names_every_block_and_none_that_is_gone() {
        let (mut doc, part) = one_part();
        let g = graph_of(&mut doc, part);
        let (a, b) = (ball(g), ball(g));
        let joined = union(g, vec![a, b]);
        let lone = ball(g);
        let spine = line(&mut doc, "Spine");
        let tube = pipe(graph_of(&mut doc, part), spine);
        doc.settle_ownership();
        name(&mut doc, b, "Bee");
        doc.ownership_curves.insert(spine, Ownership::Variable);
        let fresh = ball(graph_of(&mut doc, part));
        let (gone, gone_curve) = (NodeId::new(), CurveId::new());
        doc.ownership_nodes.insert(gone, Ownership::Free);
        doc.ownership_curves.insert(gone_curve, Ownership::Variable);

        let archive = to_archive(&doc).unwrap();
        let file: OwnershipJson =
            archive.get_json(OWNERSHIP_PATH).unwrap().expect("the entry is always written");
        let every_node: Vec<NodeId> = {
            let mut ids: Vec<NodeId> = doc.parts[&part].graph.nodes().map(|(id, _)| id).collect();
            ids.sort();
            ids
        };
        assert_eq!(
            file.nodes.keys().copied().collect::<Vec<_>>(),
            every_node,
            "every node there is, and none that has gone",
        );
        assert_eq!(file.curves.keys().copied().collect::<Vec<_>>(), vec![spine]);
        assert_eq!(file.nodes.get(&fresh), Some(&Ownership::Free), "unsettled, so written free");

        let back = from_archive(&archive).unwrap();
        let expected: std::collections::HashMap<NodeId, Ownership> = [
            (a, Ownership::Nested),
            (b, Ownership::Nested),
            (joined, Ownership::Free),
            (lone, Ownership::Free),
            (tube, Ownership::Free),
            (fresh, Ownership::Free),
        ]
        .into_iter()
        .collect();
        assert_eq!(back.ownership_nodes, expected, "named, and still nested");
        assert_eq!(
            back.ownership_curves,
            [(spine, Ownership::Variable)].into_iter().collect(),
            "a variable one block reads, and still a variable",
        );
        assert_eq!(back.pending_ownership_changes, Vec::new(), "nothing moved on the way in");
    }

    /// **A file a shipped 3.15 build wrote opens with its nested blocks nested** (spec §9.1,
    /// `notebook-spec.md` §12).
    ///
    /// 3.15 spelt a placement `{"inside": "<id>"}`, naming the one block a nested block belonged
    /// to. No parent is stored since 112, so the id is read and thrown away — and *read* is the
    /// word: the entry is a `deny_unknown_fields` struct deserialized with `?`, so a spelling this
    /// build refused would not place one block oddly, it would refuse the whole document, and
    /// every block of every file a 3.15 build wrote carries one.
    #[test]
    fn a_315_file_spelling_a_placement_inside_a_parent_still_opens() {
        let (mut doc, part) = one_part();
        let g = graph_of(&mut doc, part);
        let kept = ball(g);
        let shell = offset(g, kept);
        doc.settle_ownership();
        let mut archive = to_archive(&doc).unwrap();
        // Written by hand, because this build writes `"nested"` and never the old spelling, so
        // there is no round trip to produce it from. An id is the hex string `Display` writes.
        let as_315_wrote_it = format!(
            "{{\"nodes\":{{\"{kept}\":{{\"inside\":\"{shell}\"}},\"{shell}\":\"free\"}},\
             \"curves\":{{}}}}"
        );
        archive.put(OWNERSHIP_PATH, as_315_wrote_it.into_bytes());

        let back = from_archive(&archive).unwrap();
        assert_eq!(
            back.ownership_nodes.get(&kept),
            Some(&Ownership::Nested),
            "the block 3.15 put inside its shell is nested, and its reader is the shell",
        );
        assert_eq!(back.block_readers(fieldform_scene::BlockKey::Node(kept)), vec![shell]);
        assert_eq!(back.ownership_nodes.get(&shell), Some(&Ownership::Free));
        assert_eq!(back.pending_ownership_changes, Vec::new(), "and nothing moved on the way in");
    }

    /// **A 3.14 build opens a 3.15 file and draws what the 3.15 build drew** (spec §9.4, §12 test
    /// 15).
    ///
    /// It decides that from two entries, and both are written in the shape they always had:
    /// `node-meta.json`, whose `hidden` and `shown` are what draws, and `declared-curves.json`,
    /// which lifts a curve out of the block reading it. Where a block sits rides in an entry of its
    /// own that such a build carries through untouched, so nothing it reads strictly —
    /// `document.json` and `NodeMeta` both deny a field they do not know — has learnt a word it
    /// would refuse. The declarations are that ownership as such a build reads it: the curve
    /// variables its rule would not find without them (refactor plan 2.19), so reading the file
    /// it finds every curve variable this build had — and, by its own two-reader rule, one more
    /// that this build draws nested in both its readers.
    ///
    /// What it does not get is the rows. It lays them out by its own rule, so a block named while
    /// nested is lifted out there, and a block two blocks read is a row of its own where this
    /// build nests it in both: row differences, and no geometry difference.
    #[test]
    fn a_file_a_314_build_reads_draws_what_315_drew() {
        use fieldform_scene::NodeMeta;
        let (mut doc, part) = one_part();
        let g = graph_of(&mut doc, part);
        let source = ball(g);
        g.add(Node::Pattern {
            options: Default::default(),
            input: source.into(),
            count: ParamRef::Literal(3.0),
            kind: fieldform_scene::PatternKind::Linear {
                step: [ParamRef::Literal(30.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
            },
            join: false,
            fillet: Fillet::hard(),
        });
        let (left, right) = (ball(g), ball(g));
        let joined = union(g, vec![left, right]);
        let hidden = ball(g);
        let (rail, section) = (line(&mut doc, "Rail"), line(&mut doc, "Section"));
        let shared = line(&mut doc, "Shared");
        let g = graph_of(&mut doc, part);
        pipe(g, rail);
        pipe(g, section);
        pipe(g, shared);
        pipe(g, shared);
        doc.settle_ownership();
        // What a pattern leaves on the block it consumed, and a body put out of sight.
        doc.node_meta.insert(source, NodeMeta { shown: true, ..NodeMeta::default() });
        doc.node_meta.insert(hidden, NodeMeta { hidden: true, ..NodeMeta::default() });
        name(&mut doc, left, "Left");
        // Two curve variables — one the V made while one pipe reads it, one a profile name is on
        // — and a third curve two pipes read, which is a shared instance drawn in both of them
        // since 112 rather than the variable a second reader used to make (spec §12, R5).
        doc.ownership_curves.insert(rail, Ownership::Variable);
        doc.ownership_curves.insert(section, Ownership::Variable);
        doc.profile_names.insert(section, "Section".into());
        assert_eq!(doc.ownership_curves.get(&shared), Some(&Ownership::Nested));

        let archive = to_archive(&doc).unwrap();
        assert!(archive.get(OWNERSHIP_PATH).is_some(), "a 3.15 file says where every block sits");
        assert_eq!(
            archive.manifest().unwrap().format_major,
            3,
            "and a 3.14 build opens any minor of its own major",
        );

        // Nothing a 3.14 build reads depends on where a block sits, but for the declarations, which
        // are where its curves sit as it reads that: written with the ownership forgotten, every
        // entry but those and the one it carries through comes out byte for byte the same, and
        // with nothing a variable nothing is declared.
        let mut forgotten = doc.clone();
        forgotten.ownership_nodes.clear();
        forgotten.ownership_curves.clear();
        let unplaced = to_archive(&forgotten).unwrap();
        let read_from_ownership = [OWNERSHIP_PATH, DECLARED_CURVES_PATH];
        for path in archive.paths().filter(|p| !read_from_ownership.contains(p)) {
            assert_eq!(archive.get(path), unplaced.get(path), "{path} depends on ownership");
        }
        assert_eq!(unplaced.get(DECLARED_CURVES_PATH), None, "no variable, so nothing declared");

        // The two entries it decides by, read the way it reads them, say what they always said.
        let meta: BTreeMap<NodeId, NodeMeta> =
            archive.get_json(NODE_META_PATH).unwrap().expect("something to remember");
        let remembered: BTreeMap<NodeId, NodeMeta> = doc
            .node_meta
            .iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|(id, m)| (*id, m.clone()))
            .collect();
        assert_eq!(meta, remembered, "the flags that decide what draws");
        let declared: Vec<CurveId> =
            archive.get_json(DECLARED_CURVES_PATH).unwrap().expect("a curve is declared");
        assert_eq!(declared, vec![rail], "the variable its rule would not find by itself");

        // So it draws what this build drew, and finds every curve this build made a variable.
        let mut older = archive.clone();
        older.remove(OWNERSHIP_PATH);
        let older = from_archive(&older).unwrap();
        assert_eq!(what_draws(&older), what_draws(&doc));
        for variable in [rail, section] {
            assert_eq!(
                older.ownership_curves.get(&variable),
                doc.ownership_curves.get(&variable),
                "the declarations name the variables its own rule would miss",
            );
        }
        assert_eq!(
            older.ownership_curves.get(&shared),
            Some(&Ownership::Variable),
            "and it draws the curve two pipes read as a row of its own, where this build nests it \
             in both — a row difference, and no geometry difference (spec §15)",
        );
        let mut drawn: std::collections::BTreeSet<NodeId> =
            doc.parts[&part].graph.roots().into_iter().filter(|id| *id != hidden).collect();
        drawn.insert(source);
        assert_eq!(
            what_draws(&doc),
            drawn,
            "the fixture draws a block something consumes and leaves a body out, or it tests \
             neither flag",
        );

        // And the other row difference: named while nested in its union, `left` is nested still
        // — a name never moves a row (R5) — while the old rule lifts it out as a variable, which
        // `legacy_ownership` keeps on purpose so an old file draws what it drew.
        assert_eq!(doc.ownership_nodes.get(&left), Some(&Ownership::Nested));
        assert_eq!(
            doc.ownership_nodes.get(&joined),
            Some(&Ownership::Free),
            "and the union nothing reads is the row it is drawn in",
        );
        assert_eq!(older.legacy_ownership().0.get(&left), Some(&Ownership::Variable));
    }

    /// **A block an older build added to a 3.15 file is placed where that build showed it** (spec
    /// §9.3, §12 test 16).
    ///
    /// A 3.14 build opens a 3.15 file, adds blocks and saves. It carries `ownership.json` through
    /// untouched, so the blocks it made are not in it. Read as free, which a block with no entry
    /// is at run time, the settle on opening would adopt a named block into the one block reading
    /// it, take a declared curve into its pipe, and nest a block two others read inside both of
    /// them as a shared instance: three rows moved in a file nobody has touched since it was
    /// saved. So on opening, an id the file leaves out is placed by the rule that build laid its
    /// list out by.
    ///
    /// What the file does place stays where it says, and the settle repairs only what an edit made
    /// since has taken away, leaving the move for the application to hear of.
    #[test]
    fn blocks_an_older_build_added_are_placed_by_the_old_rule() {
        let (mut doc, part) = one_part();
        let g = graph_of(&mut doc, part);
        let kept = ball(g);
        let shell = offset(g, kept);
        let orphaned = ball(g);
        let holder = offset(g, orphaned);
        doc.settle_ownership();
        doc.pending_ownership_changes.clear();
        let saved = to_archive(&doc).unwrap();
        let carried = saved.get(OWNERSHIP_PATH).expect("a 3.15 file").to_vec();

        // What the older build does with it: adds blocks, deletes one, and saves.
        let mut older = from_archive(&saved).unwrap();
        let g = graph_of(&mut older, part);
        let named_once = ball(g);
        let reader = offset(g, named_once);
        let shared = ball(g);
        let (first, second) = (offset(g, shared), offset(g, shared));
        assert!(g.remove(holder), "the older build's delete");
        let rail = line(&mut older, "Rail");
        let tube = pipe(graph_of(&mut older, part), rail);
        name(&mut older, named_once, "Boss");
        older.declared_curves.insert(rail);
        let mut archive = saved.clone();
        update_archive(&mut archive, &older).unwrap();
        // The three marks of an older build's save: its own minor, the entry it had never heard
        // of carried back exactly as it found it, and its declarations from its own flag.
        written_as_an_older_build_declares(&mut archive, &older);
        archive.put(OWNERSHIP_PATH, carried);
        archive.put_json(crate::format::MANIFEST_PATH, &manifest_before_315()).unwrap();

        let back = from_archive(&archive).unwrap();
        let at = |id: NodeId| back.ownership_nodes.get(&id).copied();
        assert_eq!(
            at(named_once),
            Some(Ownership::Variable),
            "named, so a variable to the old rule, and not adopted by the one block reading it",
        );
        assert_eq!(
            at(shared),
            Some(Ownership::Variable),
            "read by two blocks, which the old rule drew as a row of its own, and a settle no \
             longer declares anything to disagree with it",
        );
        assert_eq!(
            back.ownership_curves.get(&rail),
            Some(&Ownership::Variable),
            "declared, so a variable, and not taken into its pipe",
        );
        for free in [reader, first, second, tube] {
            assert_eq!(at(free), Some(Ownership::Free), "nothing reads it");
        }
        assert_eq!(at(kept), Some(Ownership::Nested), "where the file put it");
        assert_eq!(
            back.block_readers(fieldform_scene::BlockKey::Node(kept)),
            vec![shell],
            "and the readers say which block draws it, now that the entry names no parent",
        );
        assert_eq!(at(orphaned), Some(Ownership::Free), "what read it has gone");
        assert_eq!(
            back.pending_ownership_changes,
            vec![fieldform_scene::OwnershipChange::Released {
                block: fieldform_scene::BlockKey::Node(orphaned),
            }],
            "the release is left for the application, and no block the older build added moved",
        );
    }

    /// **Opening a file from before 3.15 moves nothing and leaves nothing to announce** — the half
    /// of spec §12 test 17 this crate can hold.
    ///
    /// Such a file has no ownership entry, so every block is placed by the rule its list was laid
    /// out by and then settled. The settle has to find nothing to do: a block it moved would be a
    /// row moved by opening the file, and a change it pushed would be news the application
    /// announces. The other half — that the drain after a load is silent, and the document opens
    /// unmodified — is the application's (refactor plan step 2.14).
    #[test]
    fn opening_an_old_file_does_not_mark_it_changed() {
        let (mut doc, part) = one_part();
        let g = graph_of(&mut doc, part);
        let (a, b) = (ball(g), ball(g));
        let joined = union(g, vec![a, b]);
        let boss = ball(g);
        let shell = offset(g, boss);
        let shared = ball(g);
        let (first, second) = (offset(g, shared), offset(g, shared));
        let lone = ball(g);
        let rail = line(&mut doc, "Rail");
        let section = line(&mut doc, "Section");
        let loose = line(&mut doc, "Loose");
        let spare = line(&mut doc, "Spare");
        let g = graph_of(&mut doc, part);
        let (on_rail, on_section, on_loose) = (pipe(g, rail), pipe(g, section), pipe(g, loose));
        name(&mut doc, boss, "Boss");
        doc.declared_curves.insert(rail);
        doc.profile_names.insert(section, "Section".into());

        let back = from_archive(&as_saved_before_315(&doc)).unwrap();
        let nodes: std::collections::HashMap<NodeId, Ownership> = [
            (a, Ownership::Nested),
            (b, Ownership::Nested),
            (joined, Ownership::Free),
            (boss, Ownership::Variable),
            (shell, Ownership::Free),
            (shared, Ownership::Variable),
            (first, Ownership::Free),
            (second, Ownership::Free),
            (lone, Ownership::Free),
            (on_rail, Ownership::Free),
            (on_section, Ownership::Free),
            (on_loose, Ownership::Free),
        ]
        .into_iter()
        .collect();
        let curves: std::collections::HashMap<CurveId, Ownership> = [
            (rail, Ownership::Variable),
            (section, Ownership::Variable),
            (loose, Ownership::Nested),
            (spare, Ownership::Free),
        ]
        .into_iter()
        .collect();
        assert_eq!(back.ownership_nodes, nodes, "every block where the old list had it");
        assert_eq!(back.ownership_curves, curves);
        assert_eq!(back.pending_ownership_changes, Vec::new(), "and nothing to announce");

        // Saved as it opened, it reads back the same from its own entry, and still says nothing.
        let again = round_trip(&back);
        assert_eq!(again.ownership_nodes, nodes);
        assert_eq!(again.ownership_curves, curves);
        assert_eq!(again.pending_ownership_changes, Vec::new());
    }
}
