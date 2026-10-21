// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The `.fform` container.
//!
//! A zip holding an inspectable tree of parts, in the tradition of 3MF and ODF,
//! and other zip-based formats. Any of them can be opened with the unzip tool already on the machine, which is
//! what an open-format claim has to survive; an opaque blob cannot be checked by a third party
//! no matter what the specification says about it.
//!
//! ```text
//! bracket.fform
//! ├── manifest.json              format version, producer, authoring unit
//! ├── document.json              parts, graph, curves, parameters
//! ├── cages/<id>.json            cage snapshots
//! ├── sources/<id>.obj           imported files, exactly as they arrived
//! └── ext/<vendor>.<id>.<ver>/   third-party data, never interpreted here
//! ```
//!
//! # The property this exists for
//!
//! A user without plugin X opens a document containing X's data, saves, and **X's bytes come
//! back unchanged**. Without that, one person on a team lacking one paid module silently
//! destroys other people's work by opening a file, and a marketplace is not viable in any team
//! setting.
//!
//! [`Archive`] gets this by not being clever: it is a map from path to bytes, and reading then
//! writing is the identity on every entry it was not asked to change. Nothing is interpreted
//! on load beyond the manifest, so nothing can be lost by failing to understand it.
//!
//! To be precise about "byte-intact": it is the *entry contents* that are preserved, not the
//! containing zip's own bytes. Rewriting recompresses, so the file on disk differs while every
//! entry inside it is identical. That is the guarantee that matters — the plugin reads its
//! entry, not the archive framing around it.
//!
//! # Versioning policy
//!
//! Two numbers, with different promises:
//!
//! - **Major** changes when a reader that does not know it would misread the file — or, as at
//!   2.0, could not read it at all. A *newer* major is refused
//!   ([`FormatError::UnsupportedVersion`]) rather than guessed at; an *older* one is read,
//!   because what it meant is known and refusing would make each format change destroy the
//!   documents written before it.
//! - **Minor** changes when something is added that an older reader can safely ignore. A
//!   higher minor loads, and whatever the reader does not recognise is preserved by the rule
//!   above.
//!
//! So a newer file opens in an older build whenever it honestly can, and refuses when it
//! cannot — instead of the third option, which is opening it wrongly.
//!
//! # Imported files, and what keeping them makes possible
//!
//! `sources/` holds what was imported, byte for byte, because import is lossy: an OBJ carries
//! names, groups, materials, normals and texture coordinates that a
//! [`Cage`](fieldform_cage::Cage) does not keep.
//!
//! That buys more than provenance. **While the cage's topology still matches its source, the
//! two are index-compatible**, so a changed cage can be written back into the original file by
//! replacing its `v` lines and leaving every other line alone — materials, groups and UVs
//! intact. Import an OBJ, adjust the form, hand it back to the pipeline it came from. The
//! discarded data was never lost, only unused.
//!
//! That correspondence holds exactly as long as the topology does. Moving vertices preserves
//! it; extrude, bevel or a loop insert does not, and once it breaks, anything the source
//! indexes by vertex or face — `vt`/`vn` references, per-face `usemtl` — no longer lines up.
//! `SourceProvenance` records enough at import to tell the two cases apart rather than
//! guessing, and the projection is refused when the fingerprint no longer matches.
//!
//! None of this makes import *replayable*. The projection runs forward, from current positions
//! onto a preserved template; nothing below the line is re-executed, so the direct layer stays
//! destructive and the two-layer rule holds.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};
use std::path::Path;

/// Path of the manifest within the archive. The only entry whose absence is an error.
pub(crate) const MANIFEST_PATH: &str = "manifest.json";

/// Directory prefix for imported files kept verbatim.
///
/// Like `EXTENSIONS_PREFIX`, reading needs no list of places: entries under it are carried
/// because every entry is. Only storing a source and finding its record need the prefix, and
/// only the tests do either until the application keeps what it imports, so only the tests
/// compile it.
#[cfg(test)]
const SOURCES_PREFIX: &str = "sources/";

/// Directory prefix for third-party data this crate never interprets.
///
/// Nothing here needs to know where that data sits, since every entry the core does not read
/// is carried wherever it is. Only the tests that put foreign data in a file read this, and only
/// the tests compile it.
#[cfg(test)]
pub(crate) const EXTENSIONS_PREFIX: &str = "ext/";

/// The container revision this build writes.
///
/// Bump the major only when an older reader would *misread* the result; bump the minor when it
/// would merely miss something.
/// 2.0 made a part's shape the union of its graph's **bodies** — the nodes nothing else
/// consumes — and stopped writing the single `output` a 1.x graph named. A 1.x reader meets a
/// graph with no `output` at all, which its parse requires, so it cannot open the file rather
/// than misreading it. That is what makes this a major bump.
///
/// Reading *older* majors is a different question from reading newer ones, and this build does
/// read 1.x: see [`Archive::read_from`].
///
/// 3.0 made a boolean a **list**: `a` and `b` became `inputs`. A 2.x reader meets `inputs` in
/// the field list it holds for `"boolean"`, does not recognise it, and refuses the document
/// rather than reading a boolean with no operands. Going the other way, this build does **not**
/// read `a`/`b` into a two-element list — deliberately. That would be a migration, and the
/// standing rule while nobody depends on saved files is not to write them: a 2.x document with a
/// boolean in it is refused at open with the field named. Revisit at the first release.
pub(crate) const FORMAT_MAJOR: u32 = 3;
/// 2.1 added the `sphere` and `cube` node kinds. Adding a kind is the additive path: a 2.0
/// reader meets one it does not know, preserves it, and refuses to evaluate that node rather
/// than misreading the document. (1.1 had added `op` and `continuity` to `blend`, both still
/// read and written.)
///
/// 2.2 replaced an extrude's `frame` with a `sweep`. Additive in the same sense and for the same
/// reason: a 2.1 reader meets `sweep` in the field list it holds for `"extrude"`, does not
/// recognise it, and preserves the node unevaluated rather than reading an extrude with no aim
/// as one aimed along Z. Going the other way, this build reads a `frame` where it wants a
/// `sweep` and migrates it — `read_sweep` says why straight-out is the faithful answer — so
/// documents written either side of the change open on both.
///
/// 3.1 added a graph's `critical` list (`blend-integrity-spec.md` §4). Left out of the file when
/// empty, so a part with no marks is byte for byte what 3.0 wrote and a 3.0 build opens it. A
/// part *with* marks is refused by a 3.0 build — `GraphData` denies fields it does not know —
/// rather than opened with the marks dropped. That is deliberate and is why this is not the
/// "ignores it and round-trips" the spec first described: once a mark tapers a fillet (§6) it
/// changes the shape, and dropping it silently would be opening the file wrongly.
///
/// 3.2 added `history/relations.json` (`profile-group-spec.md` §5.5): which curve follows
/// which, and how. The other kind of side entry from 3.1's: a relation changes no shape, since
/// the points it drives are already in the curves, so a 3.1 build opens the file, draws it
/// exactly, and carries the entry through byte-intact. What it cannot do is keep the relation
/// true while editing, and a 3.2 build reports that on the next open rather than snapping
/// anything.
///
/// 3.3 added two kinds of relation, concentric and mirror (`profile-group-spec.md` §5.1). They
/// are written to `history/relations-added.json` rather than beside the others, because a 3.2
/// build reads `relations.json` strictly and one kind it did not know there would stop it
/// opening the file. In an entry of their own a 3.2 build opens the file, draws it exactly --
/// a relation changes no shape -- and carries the entry through byte-intact.
///
/// Four changes to what a file holds came with no minor of their own, so the number does
/// not tell a reader about them.
///
/// `history/profile-names.json` (`profile-group-spec.md` §3.3) went in within 3.1, as
/// `history/curves-created.json` had within 2.1. A name changes no shape, so an older build
/// draws the file exactly and carries the entry through byte-intact.
///
/// An extrude's `symmetric` (2026-09-07) is written only when true, and the fields the reader
/// admits for `"extrude"` left it out until 2026-09-26. A build from that window keeps a
/// symmetric extrude as an unknown node, preserved but never evaluated, even one it wrote
/// itself. A current build reads it, and `every_node_this_build_writes_is_a_node_it_can_read_back`
/// round-trips the fields that are written only when they are not the default.
///
/// Since 2026-09-26 an expression says a parameter name that is not an identifier in square
/// brackets: `[cap-1] * 2` (`expression-parameters-spec.md` §3). An older build cannot parse
/// the bracket, and its `parameters_from_json` turns that into [`FormatError::Parameter`], so
/// it refuses the whole file at open, naming the parameter, rather than misreading it. This
/// arrived during 3.2, so a 3.2 file written from then on may not open in an earlier 3.2 build.
///
/// `history/arrangement.json` (`block-order-spec.md`, 2026-09-27) holds where a person has dragged
/// each block in the list. It went in within 3.3, and as an entry of its own rather than a field
/// on `NodeMeta`, which is `deny_unknown_fields` and read with `get_json(..)?` — so a field there
/// would be a failed load in an older build rather than a dropped name. An older build carries the
/// entry through untouched, and where a row sits changes no shape.
///
/// 3.4 let a pattern's copy count be a variable (`Node::Pattern::count`, 2026-09-30). Additive in
/// the only sense that matters here: a count that is just a number is still written as the bare
/// `"count": 6` it always was, so a document nobody has parameterized is byte for byte what 3.3
/// wrote and opens in a 3.3 build. Only a count that genuinely names a variable is written in the
/// `ParamRef` spelling, and a 3.3 build refuses that file at open with the field named rather than
/// reading a count it cannot evaluate — which is the right half of additive, and the same bargain
/// 3.1 struck over `critical`. Always writing the richer spelling would have made every patterned
/// document ever saved unopenable in every shipped build, for nothing; `count_json` is where that
/// is decided. The reader goes the other way and takes both, because `6` has exactly one reading
/// and refusing it would destroy work — `read_count` says so at more length.
///
/// 3.5 added `history/declared-curves.json` (`Document::declared_curves`, 2026-10-01): which curves
/// have been asked to keep a row of their own rather than be drawn inside the block that reads them.
/// The same bargain every side entry since 3.1 has struck — an older build opens the file, draws it
/// **identically**, because a declaration moves a row and no geometry, and carries the entry through
/// byte-intact. Left out of the file when empty, so a document nobody has declared a curve in is byte
/// for byte what 3.4 wrote.
///
/// 3.6 added `history/ungrouped-curves.json` (`Document::ungrouped_curves`, 2026-10-01): the curves
/// that decline to be gathered into a profile row. The same bargain as 3.5 and every side entry since
/// 3.1 -- an older build opens the file and draws it **identically**, because a dismissal changes which
/// rows the list shows and no geometry, and carries the entry through byte-intact. Left out when empty.
///
/// 3.7 let an extrude be built on a *wall* -- a curve given a thickness -- where it had only
/// been built on a closed profile (`outline-wall-spec.md` §4). A wall is written in the
/// profile's place: `wall` and `thickness`, then `side` and `ends` where they are not the
/// defaults. Additive in 2.2's sense and for 2.2's reason: a 3.6 reader meets `wall` in the
/// field list it holds for `"extrude"`, does not recognise it, and preserves the node
/// unevaluated rather than reading an extrude with no profile. An extrude of a closed profile
/// writes what it always wrote, so a document with no wall in it differs from a 3.6 one only in
/// this number.
///
/// 3.8 gave an extrude a `start` -- where along the sweep its material begins -- and a second
/// side, `height_2` and `draft_2`, where it had one side or the `symmetric` flag
/// (`extrude-revolve-options-spec.md` §2.1). Each is written only when it is not what an
/// older document means, so an extrude using none of them writes 3.7's bytes. Additive for
/// 2.2's reason: a 3.7 reader meets `height_2` or `start` in the field list it holds for
/// `"extrude"` and preserves the node unevaluated rather than building half of a solid.
///
/// 3.9 added `history/surface-axes.json` (`Document::surface_axes`, 2026-10-02): which curves are
/// axes stood on a body's surface, and the body, direction, U, V and length each is derived from. The
/// same bargain as 3.5 and every side entry since 3.1 -- the curve is an ordinary two-point line with
/// its points stored, so an older build opens the file and draws the axis **identically**, and carries
/// the entry through byte-intact; what it lacks is the rule that re-derives the points when the body
/// moves. Left out when empty.
///
/// 3.10 gave a surface axis an optional `seed` -- a point 0 to 1 across the body's box -- and
/// placed the axis at the surface point *nearest* it instead of where a ray meets the surface
/// (`surface_axis.rs`, 2026-10-03). Additive in the strictest sense: the key is written only when
/// there is a seed, so an axis placed by a ray writes 3.9's bytes exactly, and an entry without
/// the key means the ray, which is what every 3.9 file means. The stored two points are unchanged
/// either way, so the 3.1 bargain holds and an older build draws any of them identically.
///
/// 3.11 gave a pattern options (`PatternOptions`, 2026-10-03): `symmetric` to centre the copies on
/// the body, `skip` to leave copies out, `spacing: "extent"` to read a row's step as its whole run,
/// `step2` with `count2` to repeat a row along a second step into a grid (the two are written
/// together or not at all, and a file with one is kept and refused), and `angle` to stop a ring
/// short of a turn. Each is written only when it is not what an older
/// document meant, so a pattern using none of them writes 3.10's bytes exactly. Additive for 2.2's
/// reason: a 3.10 reader meets `symmetric` or `angle` in the field list it holds for `"pattern"`,
/// does not recognise it, and preserves the node unevaluated rather than drawing a full ring where
/// the file describes an arc. A new node kind, `skewed_box`, went in the same day and needs no
/// number of its own — a kind a reader does not know is already kept and refused (2.1).
///
/// 3.12 gave a sweep a **section law**: how its section tapers (`taper_scale` or
/// `taper_draft`, never both), how far it turns (`twist`), how the two are spread along the
/// path (`law`, one of the three a ramp has), and the stretch of path it runs over (`extent`,
/// two percentages) -- `sweep-v2-spec.md` §6. Each is written only when it is not the default,
/// so a sweep that has not asked writes the bytes it wrote before, and its shader is the call
/// it always was. Additive for 2.2's reason: an older reader meets `twist` in the field list
/// it holds for `"sweep"`, does not recognise it, and preserves the node unevaluated rather
/// than drawing a prism where a taper was meant.
///
/// It was written as 3.11 and renumbered here: the pattern options of 3.11 above landed on
/// master the same day, and two changes cannot share a minor.
///
/// 3.13 let a draft say how it cuts the section's **corners**: `corners` on an extrude and
/// on a sweep's law, `"sharp"` where the two walls are to stay planes and meet at an edge
/// rather than rounding by the offset's own amount (`fieldform_scene::Corners`, the owner's
/// ruling of 2026-10-03). Written only when sharp, so every drafted solid saved before the
/// choice existed writes the bytes it wrote. Additive for 2.2's reason: an older reader meets
/// `corners` in the field list it holds for `"extrude"`, does not recognise it, and preserves
/// the node rather than drawing rounded corners where mitred ones were meant -- which is the
/// one failure here that would be invisible, both being perfectly ordinary shapes.
///
/// It was written as 3.12 and renumbered here, for the reason the entry above gives.
///
/// 3.14 added a node kind, `"loft"`: closed sections interpolated along a centerline
/// (`loft-spec.md` §7). Its `sections` are a list of outline objects spelled exactly as an
/// extrude's profile is; `spine` is written only when a centerline is named, absent meaning one
/// derived from the sections; `smooth` only when it is not G0. A new kind rather than new fields
/// on an old one, so an older reader meets `"kind": "loft"`, does not know it, and preserves the
/// node byte for byte and refuses to draw it — D3's arrangement for every kind added since 2.0.
///
/// It was written as 3.13 and renumbered here, for the reason the entries above give.
///
/// 3.15 added `history/ownership.json` (`Document::ownership_nodes` and `ownership_curves`,
/// 2026-10-11, `block-ownership-spec.md` §9): where every block sits in the list — inside the one
/// block that owns it, free, or a variable — stored where it had been read off each block's name
/// and how many blocks read it. The same bargain as 3.5 and every side entry since 3.1: an older
/// build opens the file and draws it **identically**, because where a row sits changes no geometry,
/// and carries the entry through byte-intact. `node-meta.json` and `declared-curves.json` are
/// written as they were, so that build also lays the list out as it always did, by its own rule —
/// a block named while inside another is lifted out there, a row difference and no geometry
/// difference. Since the curve row's V writes the ownership (refactor plan 2.19),
/// `declared-curves.json` is worked out from it on the way out: the curve variables that build's
/// rule would not find without the entry (`Document::curves_declared_for_older_builds`).
///
/// Two things set it apart from the side entries before it. It is written **in full**, every node
/// and every curve with free ones included, where they are left out when empty: an id it does not
/// name has to mean "the writer did not know this block", which is what a block an older build
/// added to a 3.15 file is. And the reader acts on that, placing such a block, and every block of a
/// file from before 3.15, by the rule that build laid its list out by
/// (`Document::legacy_ownership`) rather than as free — which is what keeps a file from before the
/// change opening with every row where it was.
///
/// It was specified as 3.7 and renumbered here: walls had taken 3.7, and 3.14 was the latest when
/// it was built.
///
/// 3.16 respells what that entry *holds* (`notebook-spec.md` §12, backlog item 112, 2026-10-16):
/// a nested block is written `"nested"` where 3.15 wrote `{"inside": "<id>"}`, because a block
/// nested in the notebook's model is drawn inside **every** block that reads it and so names no
/// one parent.
///
/// **This is the first entry whose bargain is not kept in both directions, and the asymmetry is
/// the whole of what to know about it.** Every side entry since 3.1 could be met by an older
/// build and carried through byte-intact, because an entry it does not understand is one it
/// ignores. Here the *values inside* a known entry changed, not a field beside them: a 3.15 build
/// reads `history/ownership.json` into a `deny_unknown_fields` struct with `?`, meets `"nested"`,
/// and refuses **the whole document** rather than one block. So a 3.16 file does not open in a
/// 3.15 build at all. Forwards is fine and is what matters for anyone's existing work: 3.16 reads
/// `{"inside": "<id>"}`, taking the id for its shape and discarding it, so every file a shipped
/// 3.15 build wrote opens with its rows where they were.
///
/// `declared-curves.json` keeps its job unchanged. A curve two blocks read is a shared instance
/// now and stays `Nested`, but the migration an older build applies
/// (`Document::legacy_ownership`) still answers `Variable` for one — §15 promises an old file
/// draws what it drew, and the old list gave such a curve a row of its own — so that entry's
/// `len() < 2` filter still names the same curves.
///
/// 3.17 adds a value, not a field: a fillet's `"continuity"` may be `"chamfer"` (backlog item 66,
/// `round-chamfer-spec.md` §2), the same form at exponent one. A blend or a pattern carrying it is
/// kept byte for byte and refused by a build that has not heard of it, the vocabulary's own
/// arrangement, so the bargain holds in both directions again: a 3.17 file opens in a 3.16 build
/// with only its chamfered joins unread, and every 3.16 file is a 3.17 file.
///
/// And it adds two fields to the profile solids, `rims` and `rims_continuity` on an extrude, a
/// sweep and a loft (`round-chamfer-spec.md` §4): how far back the rims are cut where the side
/// meets each end, and at which shape, written flat as a blend's fillet is and only when the rims
/// are not hard. A 3.16 build meets an unknown field and keeps the node rather than drawing it
/// with a hard rim it does not have, so the bargain is the same one in the same direction.
///
/// 3.18 adds `rims_at` beside them: which ends the rims round, `"start"` or `"end"`, written only
/// when not both -- which is what every rimmed solid of 3.17 means, so every 3.17 file is a 3.18
/// file. A 3.17 build meets the unknown field and keeps the node rather than rounding an end it
/// was told to leave sharp.
///
/// 3.19 adds `edge` and `edge_continuity` to a cuboid and a cylinder (`round-chamfer-spec.md`
/// §3): every edge rounded or chamfered alike, written flat and only when not hard. A 3.18 build
/// keeps such a node rather than drawing it with the sharp edges it does not have.
///
/// 3.20 adds the same two fields to a cone (`round-chamfer-spec.md` §3.2): its circular edges,
/// and its apex where a radius is nought, cut back alike. The same bargain with a 3.19 build.
///
/// 3.21 adds `rims`, `rims_continuity` and `rims_at` to a revolve (`round-chamfer-spec.md`
/// §4.3): a partial revolve's two flat ends rounded or chamfered where the turned side meets
/// them, written as an extrude's are and only when not hard. A 3.20 build keeps such a node.
pub(crate) const FORMAT_MINOR: u32 = 21;

#[derive(thiserror::Error, Debug)]
pub enum FormatError {
    #[error("failed to read or write the archive: {0}")]
    Io(#[from] std::io::Error),
    #[error("the file is not a valid zip container: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("the archive has no {MANIFEST_PATH}, so it is not a fieldform document")]
    MissingManifest,
    #[error(
        "the archive has no {}, so it is a container but not a document",
        crate::document::DOCUMENT_PATH
    )]
    MissingDocument,
    #[error(
        "entry {0:?} is not something this build knows how to read, and it sits where this \
         build owns the layout — the file was written by a newer fieldform"
    )]
    UnexpectedEntry(String),
    #[error("{path} is not valid JSON: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "the file is format {found_major}.{found_minor}, and this build reads up to \
         {FORMAT_MAJOR} — it was written by a newer fieldform and cannot be read safely"
    )]
    UnsupportedVersion { found_major: u32, found_minor: u32 },
    #[error("entry path {0:?} is not permitted: it escapes the archive root")]
    UnsafePath(String),
    /// A parameter the file describes cannot be what it says: an expression naming a
    /// parameter the file does not have, two parameters sharing the name one of them uses, or
    /// a loop of parameters reading one another. Refused and reported (D3), never opened with
    /// a number substituted.
    #[error("parameter {name:?}: {reason}")]
    Parameter { name: String, reason: String },
}

/// What the container is, independent of what it holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Manifest {
    /// Incompatible revision. A reader that does not know this number refuses the file.
    pub format_major: u32,
    /// Additive revision. A reader may load a higher minor and preserve what it cannot read.
    pub format_minor: u32,
    /// What wrote the file, for diagnosing a bad file in the wild. Advisory only — nothing
    /// branches on it, because behaviour keyed to a producer string is how formats rot.
    pub producer: String,
    /// The unit the document was authored in. Presentation only: geometry is always stored in
    /// millimetres (see [`fieldform_scene::Unit`]).
    pub unit: fieldform_scene::Unit,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            producer: format!("fieldform {}", env!("CARGO_PKG_VERSION")),
            unit: fieldform_scene::Unit::Millimetre,
        }
    }
}

/// What a cage was imported from, and whether it can still be written back.
///
/// Stored beside the source bytes as `sources/<name>.provenance.json`. The fingerprint is the
/// point: it answers "is this cage still index-compatible with the file it came from?" without
/// re-reading or re-parsing that file, so a write-back can be offered when it is honest and
/// refused when it is not.
///
/// Nothing writes `sources/` outside the tests until the application keeps what it imports
/// (see `Archive::put_source`), so nothing reads a record back either. This, `ImportedAs` and
/// `TopologyFingerprint` are the record's layout, held by the tests that write and read one,
/// and only the tests compile them (decision 1.4). They become public again with their first
/// caller.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SourceProvenance {
    /// The `sources/` entry this describes.
    pub source_path: String,
    /// What the file was called when it was imported. Advisory, for showing the user.
    pub original_name: String,
    /// The unit the file's coordinates were read as. An OBJ declares none; an IGES declares
    /// its own, and this records what was actually used.
    pub unit: fieldform_scene::Unit,
    /// What the import produced, and therefore what the source is still good for.
    pub imported_as: ImportedAs,
}

/// What an import turned a source file into.
///
/// The variants differ in what keeping the source buys, which is why this is an enum rather
/// than a bag of optional fields — the two cases have genuinely different payoffs and neither
/// should be described in the other's terms.
///
/// Test-only for the same reason as `SourceProvenance`.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ImportedAs {
    /// A cage, below the line and edited destructively.
    ///
    /// Worth keeping the source because vertex *i* means the same thing in both while the
    /// topology holds, so edits can be written back into the original file.
    Cage {
        /// Connectivity at import: vertex count, then each face's vertex indices.
        ///
        /// Positions are deliberately excluded — moving a vertex is exactly the edit that must
        /// *not* invalidate the correspondence.
        topology: TopologyFingerprint,
    },
    /// Curves, above the line and parametric. IGES arrives this way.
    ///
    /// There is no index correspondence to write back through: an IGES entity becomes a
    /// [`Curve`](fieldform_scene::Curve) by conversion, not by renaming. The payoff is the
    /// other one, and it is larger.
    ///
    /// IGES import is **partial by design** — surfaces, solids and annotations are not
    /// converted, and `import_iges` reports what it skipped precisely so a run that dropped
    /// the interesting half of a file cannot look like a clean one. Those entities are not
    /// recoverable from the curves that did come through. They are recoverable from the source,
    /// so a later build that understands more of the format can re-import and pick them up.
    ///
    /// That is the sharpest argument for `sources/` in the whole design: **the document gets
    /// better when the software does**, instead of being permanently limited to what the
    /// importer understood on the day it was first opened.
    Curves {
        count: usize,
        /// Entity kinds present in the source that this build could not convert, by name and
        /// count — what a future re-import would stand to recover.
        unconverted: BTreeMap<String, usize>,
    },
    /// Kept, and not interpreted by this build at all.
    ///
    /// A file imported by a plugin the core knows nothing about lands here, and rides along
    /// under the same rule as everything else it does not understand.
    Opaque,
}

/// A cage's connectivity, independent of where its vertices are.
///
/// Test-only for the same reason as `SourceProvenance`.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TopologyFingerprint {
    pub vertex_count: usize,
    /// Face-vertex indices, in file order. Kept in full rather than hashed: it is small, it
    /// makes a mismatch inspectable in the file rather than merely reported, and an open
    /// format should not require the reader to know which hash was used.
    pub faces: Vec<Vec<u32>>,
}

#[cfg(test)]
impl TopologyFingerprint {
    /// Takes a cage's connectivity, ignoring where its vertices are.
    fn of(cage: &fieldform_cage::Cage) -> Self {
        Self {
            vertex_count: cage.vertex_ids().count(),
            faces: cage
                .face_ids()
                .map(|f| cage.face_vertices(f).iter().map(|v| v.index() as u32).collect())
                .collect(),
        }
    }

    /// Whether a cage with this connectivity can still be projected onto that one.
    fn matches(&self, other: &TopologyFingerprint) -> bool {
        self == other
    }
}

/// A `.fform` file, as a set of named byte entries.
///
/// Deliberately untyped. Higher layers ask for the entries they understand and write the ones
/// they own; everything else is carried along untouched. That is the whole preservation
/// mechanism, and keeping it at this level means no future part of the system can accidentally
/// opt out of it.
///
/// Entries are held in a [`BTreeMap`], so a given set of entries always writes in the same
/// order. Reproducible output is a requirement for certified releases (D7) and costs nothing
/// to get here rather than retrofit.
#[derive(Debug, Clone, Default)]
pub struct Archive {
    entries: BTreeMap<String, Vec<u8>>,
}

impl Archive {
    /// A new archive containing just a manifest.
    pub(crate) fn new(manifest: &Manifest) -> Self {
        let mut archive = Self::default();
        archive.put_json(MANIFEST_PATH, manifest).expect("a Manifest always serializes");
        archive
    }

    /// Reads an archive, interpreting nothing but the manifest.
    ///
    /// The version check happens here and only here: a file this build cannot read correctly
    /// is refused up front, rather than being partially understood and then written back with
    /// the parts it misread replaced by its own guesses.
    pub(crate) fn read(path: &Path) -> Result<Self, FormatError> {
        let file = std::fs::File::open(path)?;
        Self::read_from(std::io::BufReader::new(file))
    }

    /// Reads from any seekable stream, for callers holding a document in memory.
    pub fn read_from<R: Read + Seek>(reader: R) -> Result<Self, FormatError> {
        let mut zip = zip::ZipArchive::new(reader)?;
        let mut entries = BTreeMap::new();

        for i in 0..zip.len() {
            let mut entry = zip.by_index(i)?;
            if entry.is_dir() {
                continue;
            }
            // `enclosed_name` rejects absolute paths and `..` traversal. A malicious archive
            // must not be able to name an entry that escapes the container, because this map
            // is what a later write walks.
            let name = match entry.enclosed_name() {
                Some(_) => entry.name().to_string(),
                None => return Err(FormatError::UnsafePath(entry.name().to_string())),
            };
            let mut bytes = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut bytes)?;
            entries.insert(name, bytes);
        }

        let archive = Self { entries };
        let manifest = archive.manifest()?;
        // Newer is refused; older is read. The two are not symmetric: a file from the future
        // may mean things this build would misread, while a file from the past means only what
        // it meant then, and this build knows what that was. Refusing those would make every
        // format change destroy the documents that came before it, which is a worse promise
        // than any version policy is worth.
        if manifest.format_major > FORMAT_MAJOR {
            return Err(FormatError::UnsupportedVersion {
                found_major: manifest.format_major,
                found_minor: manifest.format_minor,
            });
        }
        Ok(archive)
    }

    /// Writes the archive, entries and all.
    ///
    /// Every entry present is written, whether or not this build knows what it means. There is
    /// no filtering step, by construction rather than by discipline.
    ///
    /// Written beside the target and renamed over it, never into it. A save that fails
    /// halfway — disk full, process killed, a network share that dropped — would otherwise
    /// leave a truncated zip where the user's work used to be, and the previous version is
    /// already gone by then. The rename is atomic on every platform this runs on, so the file
    /// at `path` is either the old document or the new one.
    pub fn write(&self, path: &Path) -> Result<(), FormatError> {
        let mut temporary = path.as_os_str().to_os_string();
        temporary.push(".saving");
        let temporary = std::path::PathBuf::from(temporary);

        let result = (|| {
            let file = std::fs::File::create(&temporary)?;
            self.write_to(std::io::BufWriter::new(file))?;
            std::fs::rename(&temporary, path).map_err(FormatError::from)
        })();
        if result.is_err() {
            // Nothing useful to do if this fails too, and the original is still intact —
            // which is the outcome that matters.
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }

    /// Writes to any seekable sink.
    pub fn write_to<W: Write + Seek>(&self, writer: W) -> Result<(), FormatError> {
        let mut zip = zip::ZipWriter::new(writer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, bytes) in &self.entries {
            zip.start_file(name, options)?;
            zip.write_all(bytes)?;
        }
        zip.finish()?;
        Ok(())
    }

    /// The manifest, which every valid archive has.
    pub(crate) fn manifest(&self) -> Result<Manifest, FormatError> {
        self.get_json(MANIFEST_PATH)?.ok_or(FormatError::MissingManifest)
    }

    /// The raw bytes of an entry.
    pub(crate) fn get(&self, path: &str) -> Option<&[u8]> {
        self.entries.get(path).map(|v| v.as_slice())
    }

    /// Replaces or adds an entry.
    pub(crate) fn put(&mut self, path: impl Into<String>, bytes: Vec<u8>) {
        self.entries.insert(path.into(), bytes);
    }

    /// Removes an entry, returning its bytes.
    pub(crate) fn remove(&mut self, path: &str) -> Option<Vec<u8>> {
        self.entries.remove(path)
    }

    /// Deserializes an entry as JSON, or `None` if it is absent.
    pub(crate) fn get_json<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
    ) -> Result<Option<T>, FormatError> {
        let Some(bytes) = self.get(path) else {
            return Ok(None);
        };
        serde_json::from_slice(bytes)
            .map(Some)
            .map_err(|source| FormatError::Json { path: path.to_string(), source })
    }

    /// Serializes a value into an entry as pretty-printed JSON.
    ///
    /// Pretty rather than compact on purpose. These files are meant to be opened and read, and
    /// a graph diff is only reviewable — the argument for versioning a document semantically —
    /// if the serialized form breaks across lines.
    pub(crate) fn put_json<T: Serialize>(
        &mut self,
        path: impl Into<String>,
        value: &T,
    ) -> Result<(), FormatError> {
        let path = path.into();
        let bytes = serde_json::to_vec_pretty(value)
            .map_err(|source| FormatError::Json { path: path.clone(), source })?;
        self.put(path, bytes);
        Ok(())
    }

    /// Every entry path, in sorted order.
    pub(crate) fn paths(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    /// Paths under a prefix — `sources/`, `ext/`, `cages/`.
    pub(crate) fn paths_under<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a str> {
        self.paths().filter(move |p| p.starts_with(prefix))
    }

    /// Stores an imported file exactly as it arrived, under `sources/`.
    ///
    /// Only the tests store a source until the application writes `sources/` on import, which
    /// `cad-import-block-spec.md` §6 still lists as open, so only the tests compile it.
    /// `pub(crate)` because the document's tests store one too, and they are not a child of
    /// this module.
    #[cfg(test)]
    pub(crate) fn put_source(&mut self, name: &str, bytes: Vec<u8>) {
        self.put(format!("{SOURCES_PREFIX}{name}"), bytes);
    }

    /// Where a source's provenance sidecar lives, given the source's name.
    ///
    /// Test-only with `put_import`, the one writer that names it.
    #[cfg(test)]
    fn provenance_path(name: &str) -> String {
        format!("{SOURCES_PREFIX}{name}.provenance.json")
    }

    /// Stores an import: the file exactly as it arrived, and the record of what was made of it.
    ///
    /// One call rather than two, because the source and its provenance are only useful
    /// together — bytes with no record of how they were read cannot be re-imported
    /// consistently, and a record pointing at bytes that were not kept is worse than neither.
    /// Making the correct thing the single obvious call is what stops them drifting apart.
    ///
    /// Test-only for the same reason as `put_source`. When the application does keep what it
    /// imports, this is the call it should make, and it becomes public again then.
    #[cfg(test)]
    fn put_import(
        &mut self,
        name: &str,
        bytes: Vec<u8>,
        imported_as: ImportedAs,
        unit: fieldform_scene::Unit,
    ) -> Result<(), FormatError> {
        let source_path = format!("{SOURCES_PREFIX}{name}");
        let provenance = SourceProvenance {
            source_path: source_path.clone(),
            original_name: name.to_string(),
            unit,
            imported_as,
        };
        self.put(source_path, bytes);
        self.put_json(Self::provenance_path(name), &provenance)
    }

    /// Every import in the archive, paired with its bytes.
    ///
    /// Skips sidecars whose source is missing and sources with no sidecar: both are archives
    /// written by something else, and neither is this build's to repair. They stay in the file
    /// regardless — being unreadable here is not grounds for dropping anything.
    ///
    /// Test-only for the same reason as `put_source`: only the tests store an import, so only
    /// they read one back.
    #[cfg(test)]
    fn imports(&self) -> impl Iterator<Item = (SourceProvenance, &[u8])> {
        self.paths_under(SOURCES_PREFIX)
            .filter(|p| p.ends_with(".provenance.json"))
            .filter_map(move |p| {
                let provenance: SourceProvenance = self.get_json(p).ok().flatten()?;
                let bytes = self.get(&provenance.source_path)?;
                Some((provenance, bytes))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn round_trip(archive: &Archive) -> Archive {
        let mut buf = Vec::new();
        archive.write_to(Cursor::new(&mut buf)).expect("writes");
        Archive::read_from(Cursor::new(buf)).expect("reads back")
    }

    /// The property the marketplace rests on, stated as directly as it can be.
    ///
    /// A build that has never heard of `acme.lattice` opens the file, changes something it
    /// does own, saves, and Acme's bytes are the bytes Acme wrote. If this ever fails, one
    /// person on a team without one plugin destroys everyone else's work by opening a
    /// document.
    #[test]
    fn data_from_an_unknown_extension_survives_a_round_trip_byte_for_byte() {
        // Deliberately not valid JSON, and deliberately containing a NUL: the core must not be
        // parsing, re-encoding or even assuming text.
        let plugin_bytes: Vec<u8> = vec![0x00, 0xFF, 0x10, b'{', b'n', b'o', 0x80, 0x7F];

        let path = format!("{EXTENSIONS_PREFIX}acme.lattice.3/state.bin");
        let mut archive = Archive::new(&Manifest::default());
        archive.put(path.as_str(), plugin_bytes.clone());

        let mut reopened = round_trip(&archive);
        // The user edits something this build *does* understand, then saves.
        reopened.put_json("document.json", &serde_json::json!({ "parts": [] })).unwrap();
        let saved = round_trip(&reopened);

        assert_eq!(
            saved.get(&path),
            Some(plugin_bytes.as_slice()),
            "an unknown extension's bytes must come back exactly"
        );
    }

    /// Preservation must not be a property of the `ext/` prefix specifically. Anything the
    /// core does not recognise is carried, wherever it sits — the rule is "do not interpret",
    /// not "interpret a known list of places".
    #[test]
    fn unrecognised_entries_anywhere_are_carried_untouched() {
        let mut archive = Archive::new(&Manifest::default());
        archive.put("who/knows/what.dat", vec![1, 2, 3]);
        archive.put("cages/from-a-newer-build.json", b"{\"shape\":\"unknown\"}".to_vec());

        let saved = round_trip(&round_trip(&archive));
        assert_eq!(saved.get("who/knows/what.dat"), Some([1u8, 2, 3].as_slice()));
        assert_eq!(
            saved.get("cages/from-a-newer-build.json"),
            Some(b"{\"shape\":\"unknown\"}".as_slice())
        );
    }

    /// An imported file is kept as it arrived, including the parts import throws away.
    #[test]
    fn an_imported_obj_is_kept_exactly_as_it_arrived() {
        // Comments, groups and a material reference — none of which survive into a Cage.
        let obj = b"# exported by something else\nmtllib bracket.mtl\ng shell\nv 0 0 0\n";

        let mut archive = Archive::new(&Manifest::default());
        archive.put_source("bracket.obj", obj.to_vec());

        let saved = round_trip(&archive);
        assert_eq!(saved.get("sources/bracket.obj"), Some(obj.as_slice()));
        assert_eq!(saved.paths_under(SOURCES_PREFIX).count(), 1);
    }

    /// The distinction the whole write-back idea rests on: moving vertices must not break the
    /// correspondence with the source file, and changing connectivity must.
    ///
    /// If a position edit invalidated the fingerprint, write-back would be refused after the
    /// single most common edit there is, and preserving the source would buy nothing beyond
    /// provenance.
    #[test]
    fn moving_a_vertex_keeps_the_source_correspondence_and_retopology_breaks_it() {
        use fieldform_cage::Cage;
        use nalgebra::Vector3;

        let mut cage = Cage::cube();
        let at_import = TopologyFingerprint::of(&cage);

        for v in cage.vertex_ids().collect::<Vec<_>>() {
            let p = cage.vertex_position(v);
            cage.set_vertex_position(v, p + Vector3::new(0.3, -1.7, 12.0));
        }
        assert!(
            at_import.matches(&TopologyFingerprint::of(&cage)),
            "moving every vertex must leave the cage index-compatible with its source"
        );

        // Connectivity that is not the one imported: vertex i no longer means what the file
        // says it means, so anything the file indexes by vertex or face no longer lines up.
        let mut retopologised = at_import.clone();
        retopologised.faces[0].swap(0, 2);
        assert!(!at_import.matches(&retopologised));

        let mut extra_vertex = at_import.clone();
        extra_vertex.vertex_count += 1;
        assert!(!at_import.matches(&extra_vertex));
    }

    /// Provenance rides in the archive like everything else, so it survives a build that does
    /// not know what to do with it.
    #[test]
    fn an_import_round_trips_with_the_record_of_how_it_was_read() {
        let mut archive = Archive::new(&Manifest::default());
        archive
            .put_import(
                "bracket.obj",
                b"v 0 0 0\n".to_vec(),
                ImportedAs::Cage { topology: TopologyFingerprint::of(&fieldform_cage::Cage::cube()) },
                fieldform_scene::Unit::Millimetre,
            )
            .unwrap();

        let saved = round_trip(&archive);
        let (provenance, bytes) = saved.imports().next().expect("one import");
        assert_eq!(provenance.original_name, "bracket.obj");
        assert_eq!(bytes, b"v 0 0 0\n");
    }

    /// An IGES keeps its own bytes, and the record says what a re-import would be worth.
    ///
    /// IGES import is partial by design: curves convert, surfaces and solids do not. Those
    /// entities cannot be recovered from the curves that came through — only from the source.
    /// So the source is what lets a later build that understands more of the format pick up
    /// what this one skipped, and `unconverted` is the record of how much is waiting.
    #[test]
    fn an_iges_keeps_the_entities_this_build_could_not_convert() {
        // Fixed-column IGES cards, including a trailing section terminator — the kind of exact
        // formatting a re-encode would disturb and a byte-preserving store will not.
        let igs = b"                                                                        S      1\n\
                    1H,,1H;,4Hpart,                                                         G      1\n\
                    S      1G      1D      2P      1                                        T      1\n";

        let unconverted = BTreeMap::from([
            ("Rational B-Spline Surface".to_string(), 12usize),
            ("Trimmed Surface".to_string(), 4),
        ]);

        let mut archive = Archive::new(&Manifest::default());
        archive
            .put_import(
                "housing.igs",
                igs.to_vec(),
                ImportedAs::Curves { count: 37, unconverted: unconverted.clone() },
                fieldform_scene::Unit::Millimetre,
            )
            .unwrap();

        let saved = round_trip(&round_trip(&archive));
        let (provenance, bytes) = saved.imports().next().expect("the IGES import");

        assert_eq!(bytes, igs.as_slice(), "the IGES must survive column-exact");
        match provenance.imported_as {
            ImportedAs::Curves { count, unconverted: skipped } => {
                assert_eq!(count, 37);
                assert_eq!(skipped, unconverted, "what a re-import would recover");
            }
            other => panic!("expected Curves, got {other:?}"),
        }
    }

    /// A file this build cannot interpret at all is still kept, and still described. An
    /// importer that arrives as a plugin later is the case this covers.
    #[test]
    fn an_import_the_core_cannot_interpret_is_still_kept() {
        let mut archive = Archive::new(&Manifest::default());
        archive
            .put_import(
                "scan.e57",
                vec![0x00, 0xE5, 0x07, 0xFF],
                ImportedAs::Opaque,
                fieldform_scene::Unit::Metre,
            )
            .unwrap();

        let saved = round_trip(&archive);
        let (provenance, bytes) = saved.imports().next().expect("the opaque import");
        assert_eq!(bytes, [0x00, 0xE5, 0x07, 0xFF]);
        assert_eq!(provenance.imported_as, ImportedAs::Opaque);
        assert_eq!(provenance.unit, fieldform_scene::Unit::Metre);
    }

    /// Several imports of different kinds coexist without interfering.
    #[test]
    fn imports_of_different_kinds_coexist() {
        let mut archive = Archive::new(&Manifest::default());
        archive
            .put_import(
                "a.obj",
                b"v 1 2 3\n".to_vec(),
                ImportedAs::Cage { topology: TopologyFingerprint::of(&fieldform_cage::Cage::cube()) },
                fieldform_scene::Unit::Millimetre,
            )
            .unwrap();
        archive
            .put_import(
                "b.igs",
                b"S      1\n".to_vec(),
                ImportedAs::Curves { count: 2, unconverted: BTreeMap::new() },
                fieldform_scene::Unit::Inch,
            )
            .unwrap();

        let saved = round_trip(&archive);
        let names: Vec<String> =
            saved.imports().map(|(p, _)| p.original_name.clone()).collect();
        assert_eq!(names, vec!["a.obj".to_string(), "b.igs".to_string()]);
    }

    /// A file from a future incompatible revision is refused rather than half-understood.
    /// Reading it wrongly and then writing it back is the outcome worth preventing.
    #[test]
    fn a_newer_major_version_is_refused_rather_than_guessed_at() {
        let manifest = Manifest { format_major: FORMAT_MAJOR + 1, ..Manifest::default() };
        let mut archive = Archive::default();
        archive.put_json(MANIFEST_PATH, &manifest).unwrap();

        let mut buf = Vec::new();
        archive.write_to(Cursor::new(&mut buf)).unwrap();
        let err = Archive::read_from(Cursor::new(buf)).unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedVersion { .. }), "got {err:?}");
    }

    /// An older major is a document this build knows how to read, and refusing it would mean
    /// every format change destroys what came before it.
    #[test]
    fn an_older_major_version_still_opens() {
        let manifest = Manifest { format_major: 1, format_minor: 1, ..Manifest::default() };
        let mut archive = Archive::default();
        archive.put_json(MANIFEST_PATH, &manifest).unwrap();

        let mut buf = Vec::new();
        archive.write_to(Cursor::new(&mut buf)).unwrap();
        let reopened = Archive::read_from(Cursor::new(buf)).expect("format 1 still opens");
        assert_eq!(reopened.manifest().unwrap().format_major, 1);
    }

    /// A higher minor is additive by definition, so it must load — and the thing the older
    /// reader does not understand must still be there afterwards.
    #[test]
    fn a_newer_minor_version_loads_and_keeps_what_it_does_not_understand() {
        let manifest = Manifest { format_minor: FORMAT_MINOR + 7, ..Manifest::default() };
        let mut archive = Archive::default();
        archive.put_json(MANIFEST_PATH, &manifest).unwrap();
        archive.put("something-added-later.json", b"{}".to_vec());

        let saved = round_trip(&round_trip(&archive));
        assert_eq!(saved.get("something-added-later.json"), Some(b"{}".as_slice()));
        assert_eq!(saved.manifest().unwrap().format_minor, FORMAT_MINOR + 7);
    }

    /// Not a fieldform document, and said so plainly rather than failing later on a missing key.
    #[test]
    fn an_archive_without_a_manifest_is_not_a_document() {
        let mut archive = Archive::default();
        archive.put("random.txt", b"hello".to_vec());
        let mut buf = Vec::new();
        archive.write_to(Cursor::new(&mut buf)).unwrap();

        assert!(matches!(
            Archive::read_from(Cursor::new(buf)).unwrap_err(),
            FormatError::MissingManifest
        ));
    }

    /// Certified releases need the same input to produce the same output (D7). Entry order is
    /// the part of that this layer controls, so it is fixed here rather than left to
    /// insertion order.
    #[test]
    fn entry_order_does_not_depend_on_insertion_order() {
        let mut a = Archive::new(&Manifest::default());
        a.put("z.json", b"1".to_vec());
        a.put("a.json", b"2".to_vec());

        let mut b = Archive::new(&Manifest::default());
        b.put("a.json", b"2".to_vec());
        b.put("z.json", b"1".to_vec());

        let paths_a: Vec<_> = a.paths().collect();
        let paths_b: Vec<_> = b.paths().collect();
        assert_eq!(paths_a, paths_b);

        let mut buf_a = Vec::new();
        let mut buf_b = Vec::new();
        a.write_to(Cursor::new(&mut buf_a)).unwrap();
        b.write_to(Cursor::new(&mut buf_b)).unwrap();
        assert_eq!(buf_a, buf_b, "the same entries should produce the same file");
    }

    /// An entry naming its way out of the archive is rejected on read, since the entry map is
    /// what a later write walks.
    #[test]
    fn an_entry_escaping_the_archive_root_is_rejected() {
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(Cursor::new(&mut buf));
            let options = zip::write::SimpleFileOptions::default();
            zip.start_file("../../etc/passwd", options).unwrap();
            zip.write_all(b"nope").unwrap();
            zip.finish().unwrap();
        }
        assert!(matches!(
            Archive::read_from(Cursor::new(buf)).unwrap_err(),
            FormatError::UnsafePath(_)
        ));
    }
}
