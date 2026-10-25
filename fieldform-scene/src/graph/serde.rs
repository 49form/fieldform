// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How a node is written to a file and read back, and what is kept of one this build cannot read.

use super::{
    no_turn, turn_is_none, Attractor, Axis, CellMap, Continuity, Fillet, RimEnds, Rims, LatticeCell, LatticePhase,
    Node, ParamRef, PatternKind, PatternOptions, PatternSecond, Spacing, Turn, FULL_TURN_DEGREES,
    PATTERN_COUNT_MAX,
};
use crate::graph::Operand;
use crate::Frame;
use crate::{
    AngleSides, Corners, LoftSmooth, Outline, Sides, Sweep, SweepLaw, Taper, WallEnds, WallSide,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Why a node could not be read, which is what an error message has to say to be useful.
#[derive(Clone, Debug, PartialEq)]
pub enum Unrecognised {
    /// The `kind` itself is one this build has never heard of.
    Kind,
    /// A field this build knows carries a value from a vocabulary that has grown since.
    ///
    /// `continuity: "g2"` on a blend, written by a build that has curvature-continuous ones.
    /// The node is readable in every other respect and is kept intact; what cannot be done is
    /// evaluate it, because the value that was not understood is the one that decides the
    /// shape.
    Value { field: String, value: String },
    /// The kind is known, but the node carries fields this build cannot account for.
    ///
    /// Preserved *and* refused, deliberately. An unrecognised field on a blend may well be
    /// decorative — or it may be `falloff`, in which case evaluating without it gives a shape
    /// that is wrong in a way nothing downstream can notice. There is no way to tell those
    /// two apart from here, and guessing is the one option the format rules out.
    Fields(Vec<String>),
}

/// A node kept verbatim because this build could not fully read it.
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownNode {
    raw: Value,
    pub(super) reason: Unrecognised,
}

impl UnknownNode {
    /// Wraps raw JSON whose `kind` this build does not know.
    pub(crate) fn of_unknown_kind(raw: Value) -> Self {
        Self { raw, reason: Unrecognised::Kind }
    }

    /// Wraps raw JSON of a known kind that carries fields this build cannot account for.
    pub(crate) fn with_unrecognised_fields(raw: Value, fields: Vec<String>) -> Self {
        Self { raw, reason: Unrecognised::Fields(fields) }
    }

    /// Wraps raw JSON whose value for a known field is one this build does not know.
    pub(crate) fn with_unrecognised_value(raw: Value, field: String, value: String) -> Self {
        Self { raw, reason: Unrecognised::Value { field, value } }
    }

    /// The node's `kind` as the file spells it, or `None` if it does not have one.
    pub(crate) fn kind(&self) -> Option<&str> {
        self.raw.get(KIND).and_then(Value::as_str)
    }

    /// Why this build did not recognise the node. The message a user sees comes from `Display`,
    /// so only the tests ask for the reason itself, and only the tests compile this (decision
    /// 1.4). `pub(crate)` because those tests are `graph`'s, not this module's.
    #[cfg(test)]
    pub(crate) fn reason(&self) -> &Unrecognised {
        &self.reason
    }
}

impl std::fmt::Display for UnknownNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = self.kind().unwrap_or("(a node with no kind)");
        match &self.reason {
            Unrecognised::Kind => write!(
                f,
                "node kind {kind:?}, which this build does not know — the file was written \
                 by a newer fieldform, or by a plugin this build does not have"
            ),
            Unrecognised::Fields(fields) => {
                write!(f, "{kind} node carrying unrecognised field(s): {}", fields.join(", "))
            }
            Unrecognised::Value { field, value } => write!(
                f,
                "{kind} node whose {field} is {value:?}, which this build does not know — the \
                 file was written by a newer fieldform"
            ),
        }
    }
}

/// A pattern's count, read from either spelling.
///
/// **Both, because the old one is unambiguous and refusing it would destroy work.** Every document
/// saved before the count became a [`ParamRef`] carries a bare `"count": 6`, and `read` turns a type
/// mismatch into a hard serde error — not the [`Node::Unknown`] degradation that normally protects a
/// file this build does not fully understand. So without this, a document holding any pattern at all
/// would fail to open as a whole, with the error naming `document.json` rather than the field. The
/// governing line is the degree-nine curve's: a saved document refused on open is work destroyed.
///
/// `6` has exactly one reading, `ParamRef::Literal(6.0)`, which is what makes this a migration
/// rather than a guess — the distinction `format.rs` draws when it declines to migrate a boolean's
/// `a`/`b` pair, whose old shape has no single meaning.
fn read_count<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<ParamRef, E> {
    let value = object.get("count").ok_or_else(|| E::missing_field("count"))?;
    read_count_of(kind, "count", value)
}

/// A count read from a value by the name it is written under, so a second count is read by the
/// rule the first one is.
fn read_count_of<E: serde::de::Error>(
    kind: &str,
    field: &str,
    value: &Value,
) -> Result<ParamRef, E> {
    // A bare number is the old shape. Tried first and by value, so a `ParamRef` spelling that
    // happens to be a JSON number cannot be misread — there is no such spelling.
    if let Some(n) = value.as_f64() {
        return Ok(ParamRef::Literal(n));
    }
    read(kind, field, value)
}

/// What to write for a count, so a document nobody has parameterized is byte-identical to what the
/// previous format wrote.
///
/// **The bare number is kept for a plain literal**, which is what makes this a *minor* bump rather
/// than a major one: a pattern whose count is just a number still opens in an older build, and only
/// a count that genuinely uses a variable is refused there — with the field named. Always writing
/// the `ParamRef` spelling would make every existing patterned document unopenable in every shipped
/// build, for no gain.
fn count_json<E: serde::ser::Error>(count: &ParamRef) -> Result<Value, E> {
    match count {
        // Integral and in range is the shape an older build can read back and mean the same thing
        // by. A fractional literal is refused at evaluation anyway, but it is still written in the
        // richer spelling rather than rounded on the way out: the file must say what was typed.
        ParamRef::Literal(v)
            if v.fract() == 0.0 && *v >= 1.0 && *v <= f64::from(PATTERN_COUNT_MAX) =>
        {
            json(&(*v as u32))
        }
        other => json(other),
    }
}

/// An extrude's sweep, migrating the `Frame` it used to carry.
///
/// **The old frame becomes `Normal`, and that is faithful rather than a guess.** It said two
/// things at once: the plane the profile was read in, and the direction it was thickened along.
/// The plane is derived from the profile now, so only the direction is left to carry over — and
/// the frame's direction *was* its own normal, which was meant to be the profile's plane. Where
/// it actually was, the block did not compile, so there is no working document whose meaning
/// this changes; there are broken ones it repairs.
fn read_sweep<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<Sweep, E> {
    match object.get("sweep") {
        Some(v) => read(kind, "sweep", v),
        None => Ok(Sweep::Normal),
    }
}

/// A block's outline, as the fields it is written in.
///
/// **Flat on the node, and a closed profile writes the one field it always wrote.** The type is
/// a choice between two cases and the file is not a nested object saying which: a profile is
/// still `"profile": <curve>`, so every document saved before there was a choice reads back
/// unchanged and writes back byte for byte. A wall writes `"wall"` in its place, with its
/// thickness, and its side and its ends only where they are not the default — the treatment
/// `symmetric` got, for the same reason.
fn outline_fields<E: serde::ser::Error>(
    outline: &Outline,
) -> Result<Vec<(&'static str, Value)>, E> {
    Ok(match outline {
        Outline::Closed { profile } => vec![("profile", json(profile)?)],
        Outline::Wall { curve, thickness, side, ends } => {
            let mut fields = vec![("wall", json(curve)?), ("thickness", json(thickness)?)];
            if *side != WallSide::default() {
                fields.push(("side", json(side)?));
            }
            if *ends != WallEnds::default() {
                fields.push(("ends", json(ends)?));
            }
            fields
        }
    })
}

/// How a swept section changes along its path, written flat and only where it is not the
/// default -- `taper_scale` or `taper_draft`, `twist`, `law`, `extent` -- so a sweep that has
/// not asked writes exactly the bytes it did. `sweep-v2-spec.md` §6.
fn law_fields<E: serde::ser::Error>(law: &SweepLaw) -> Result<Vec<(&'static str, Value)>, E> {
    let plain = SweepLaw::default();
    let mut fields = Vec::new();
    match &law.taper {
        Taper::None => {}
        Taper::Scale(v) => fields.push(("taper_scale", json(v)?)),
        Taper::Draft(v) => fields.push(("taper_draft", json(v)?)),
    }
    if law.twist != plain.twist {
        fields.push(("twist", json(&law.twist)?));
    }
    if law.shape != plain.shape {
        fields.push(("law", json(&law.shape)?));
    }
    if law.extent != plain.extent {
        fields.push(("extent", json(&law.extent)?));
    }
    if !law.corners.is_default() {
        fields.push(("corners", json(&law.corners)?));
    }
    Ok(fields)
}

/// The same, read back. Absent means the default, which is what leaving a field out on write
/// promises; both tapers at once is two answers to one question and is refused.
fn read_law<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<SweepLaw, E> {
    let plain = SweepLaw::default();
    let taper = match (object.get("taper_scale"), object.get("taper_draft")) {
        (Some(scale), Some(draft)) => {
            return Err(E::custom(format!(
                "{kind} node says both \"taper_scale\" and \"taper_draft\", which is two \
                 answers to how its section tapers: {scale} and {draft}"
            )));
        }
        (Some(v), None) => Taper::Scale(read(kind, "taper_scale", v)?),
        (None, Some(v)) => Taper::Draft(read(kind, "taper_draft", v)?),
        (None, None) => Taper::None,
    };
    Ok(SweepLaw {
        taper,
        twist: match object.get("twist") {
            Some(v) => read(kind, "twist", v)?,
            None => plain.twist,
        },
        shape: match object.get("law") {
            Some(v) => read(kind, "law", v)?,
            None => plain.shape,
        },
        extent: match object.get("extent") {
            Some(v) => read(kind, "extent", v)?,
            None => plain.extent,
        },
        corners: read_corners(kind, object)?,
    })
}

/// A profile solid's rims, written flat as a blend's fillet is -- `rims` the reach and
/// `rims_continuity` the shape -- and only where there is one: a hard rim writes nothing, so
/// every extrude, sweep and loft written before 3.17 is written as it was. The shape is left out
/// at G1, the one every fillet means unless it says otherwise. Flat rather than nested so the
/// vocabulary table can refuse a shape this build has not heard of, which it reads from the
/// node's own fields.
fn rim_fields<E: serde::ser::Error>(rims: &Rims) -> Result<Vec<(&'static str, Value)>, E> {
    if rims.is_plainly_hard() {
        return Ok(Vec::new());
    }
    let mut fields = vec![("rims", json(&rims.radius)?)];
    if rims.continuity != Continuity::G1 {
        fields.push(("rims_continuity", json(&rims.continuity)?));
    }
    // Which ends, only when not both -- what every rimmed solid written before it meant.
    if rims.at != RimEnds::Both {
        fields.push(("rims_at", json(&rims.at)?));
    }
    // A chamfer's second leg, 3.22's, only where there is one.
    if let Some(second) = &rims.second {
        fields.push(("rims_second", json(second)?));
    }
    Ok(fields)
}

/// The same, read back. Absent means hard; a shape without a reach is a file that says how to
/// round a rim it does not round, and is refused rather than guessed at.
fn read_rims<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<Rims, E> {
    match (object.get("rims"), object.get("rims_continuity")) {
        (None, None) => Ok(Rims::hard()),
        (None, Some(_)) => Err(E::custom(format!(
            "{kind} node says how its rims are shaped (\"rims_continuity\") but not how far \
             (\"rims\")"
        ))),
        (Some(radius), shape) => Ok(Rims {
            radius: read(kind, "rims", radius)?,
            continuity: match shape {
                Some(v) => read(kind, "rims_continuity", v)?,
                None => Continuity::G1,
            },
            at: match object.get("rims_at") {
                Some(v) => read(kind, "rims_at", v)?,
                None => RimEnds::Both,
            },
            second: match object.get("rims_second") {
                Some(v) => Some(read(kind, "rims_second", v)?),
                None => None,
            },
        }),
    }
}

/// A box's or a cylinder's edge, written flat as a profile solid's rims are -- `edge` the reach
/// and `edge_continuity` the shape -- and only when it is not hard.
fn edge_fields<E: serde::ser::Error>(edge: &Fillet) -> Result<Vec<(&'static str, Value)>, E> {
    if edge.is_plainly_hard() {
        return Ok(Vec::new());
    }
    let mut fields = vec![("edge", json(&edge.radius)?)];
    if edge.continuity != Continuity::G1 {
        fields.push(("edge_continuity", json(&edge.continuity)?));
    }
    Ok(fields)
}

/// The same, read back. Absent is hard; a shape without a reach is refused as damaged, as a
/// rim's is.
fn read_edge<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<Fillet, E> {
    match (object.get("edge"), object.get("edge_continuity")) {
        (None, None) => Ok(Fillet::hard()),
        (None, Some(_)) => Err(E::custom(format!(
            "{kind} node says how its edges are shaped (\"edge_continuity\") but not how far (\"edge\")"
        ))),
        (Some(radius), shape) => Ok(Fillet {
            radius: read(kind, "edge", radius)?,
            continuity: match shape {
                Some(v) => read(kind, "edge_continuity", v)?,
                None => Continuity::G1,
            },
            second: None,
        }),
    }
}

/// How a draft cuts the section's corners, from the one spelling the writer uses. Absent means
/// round, which is what every drafted solid written before the choice existed means.
fn read_corners<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<Corners, E> {
    match object.get("corners") {
        Some(v) => read(kind, "corners", v),
        None => Ok(Corners::default()),
    }
}

/// [`outline_fields`], read back. `"wall"` is what says which case it is.
fn read_outline<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<Outline, E> {
    let missing =
        |name: &str| E::custom(format!("{kind} node has no {name:?} field"));
    let Some(curve) = object.get("wall") else {
        let profile = object.get("profile").ok_or_else(|| missing("profile"))?;
        return Ok(Outline::Closed { profile: read(kind, "profile", profile)? });
    };
    Ok(Outline::Wall {
        curve: read(kind, "wall", curve)?,
        // A wall with no thickness is damaged rather than newer, like any other missing field.
        thickness: read(
            kind,
            "thickness",
            object.get("thickness").ok_or_else(|| missing("thickness"))?,
        )?,
        // Absent means the default, which is what leaving them out on write promises.
        side: match object.get("side") {
            Some(v) => read(kind, "side", v)?,
            None => WallSide::default(),
        },
        ends: match object.get("ends") {
            Some(v) => read(kind, "ends", v)?,
            None => WallEnds::default(),
        },
    })
}

/// Which way an extrude's depth lies about its plane, read back from the three spellings the
/// writer uses: nothing, `"symmetric": true`, or `height_2` with `draft_2`.
///
/// **A node saying two of them is kept and refused** (D3), as an extrude naming both a profile
/// and a wall is: `symmetric` beside `height_2` is a file saying two things about the same
/// sides, and guessing which was meant would build a solid nobody drew. `symmetric: false`
/// written out is the one side it always was, and says nothing.
fn read_sides<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<Sides, E> {
    let symmetric = match object.get("symmetric") {
        Some(v) => read::<bool, E>(kind, "symmetric", v)?,
        None => false,
    };
    match object.get("height_2") {
        Some(height) if symmetric => Err(E::custom(format!(
            "{kind} node says both \"symmetric\" and \"height_2\", which is two answers to \
             which way its depth lies: {height}"
        ))),
        Some(height) => Ok(Sides::Two {
            height: read(kind, "height_2", height)?,
            // Absent means upright, as side one's draft does.
            draft: match object.get("draft_2") {
                Some(v) => read(kind, "draft_2", v)?,
                None => ParamRef::Literal(0.0),
            },
        }),
        None if symmetric => Ok(Sides::Symmetric),
        None => Ok(Sides::One),
    }
}

/// Which way a revolve's angle lies about its profile's half-plane, from the three spellings the
/// writer uses: nothing, `"symmetric": true`, or `angle_2`. Both at once is refused, as an
/// extrude's `symmetric` beside `height_2` is.
fn read_angle_sides<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<AngleSides, E> {
    let symmetric = match object.get("symmetric") {
        Some(v) => read::<bool, E>(kind, "symmetric", v)?,
        None => false,
    };
    match object.get("angle_2") {
        Some(angle) if symmetric => Err(E::custom(format!(
            "{kind} node says both \"symmetric\" and \"angle_2\", which is two answers to \
             which way its angle lies: {angle}"
        ))),
        Some(angle) => Ok(AngleSides::Two { angle: read(kind, "angle_2", angle)? }),
        None if symmetric => Ok(AngleSides::Symmetric),
        None => Ok(AngleSides::One),
    }
}

/// Reads a node's frame, migrating a document that predates the type.
///
/// **The migration is the point.** Before D9 there were four spellings of "a place and a
/// direction", and every one of them is in somebody's saved file. A reader that only understood
/// the new field would leave those documents opening to a twist about Z and a mirror in the
/// wrong place — quietly, since both are valid frames. So the old spellings are read where they
/// are found and converted here, once, rather than in each node's arm.
///
/// The `"frame"` field wins when present. Failing that: `"plane": "node"` with a `"plane_node"`
/// is a frame taken from that node, and an `"axis"` string with an `"origin"` is a frame at that
/// point along that axis. A node with none of them means the world frame, which is what a twist
/// or a pattern written without one always meant.
fn read_frame<E: serde::de::Error>(
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<Frame, E> {
    if let Some(v) = object.get("frame") {
        return read(kind, "frame", v);
    }
    // A plane fed in, which a mirror could say and nothing else could.
    if object.get("plane").and_then(Value::as_str) == Some("node")
        && let Some(node) = object.get("plane_node")
    {
        return Ok(Frame::From { node: read(kind, "plane_node", node)? });
    }
    // An axis name and a point, which is what a twist, a radial pattern and a mirror all wrote.
    let named: Axis = match object.get("axis") {
        Some(v) => read(kind, "axis", v)?,
        None => Axis::Z,
    };
    let origin: [ParamRef; 3] = match object.get("origin") {
        Some(v) => read(kind, "origin", v)?,
        None => [ParamRef::Literal(0.0); 3],
    };
    let Frame::At { axis, .. } = Frame::along(named) else {
        unreachable!("`along` builds an explicit frame")
    };
    Ok(Frame::At { origin, axis })
}

/// The tag every serialized node carries.
const KIND: &str = "kind";

impl Serialize for Node {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Built as a `Value` rather than streamed, so the unknown case can hand back exactly
        // what it was given and the two paths cannot drift apart.
        let value = match self {
            Node::Cage(cage) => json_object([(KIND, json(&"cage")?), ("cage", json(cage)?)]),
            Node::Mesh(mesh) => json_object([(KIND, json(&"mesh")?), ("mesh", json(mesh)?)]),
            Node::Sphere { centre, radius } => json_object([
                (KIND, json(&"sphere")?),
                ("centre", json(centre)?),
                ("radius", json(radius)?),
            ]),
            // The turn is written only when there is one, so a file gains nothing for a block
            // that was never turned and an older reader is not handed a field it cannot place.
            // And the edge only when it is not hard, for the same reason (3.19).
            Node::Cuboid { centre, size, turn, edge } => json_fields(
                [(KIND, json(&"cuboid")?), ("centre", json(centre)?), ("size", json(size)?)]
                    .into_iter()
                    .chain((!turn_is_none(turn)).then(|| json(turn).map(|t| ("turn", t))).transpose()?)
                    .chain(edge_fields(edge)?),
            ),
            Node::SkewedBox { centre, edges } => json_object([
                (KIND, json(&"skewed_box")?),
                ("centre", json(centre)?),
                ("edges", json(edges)?),
            ]),
            Node::Cylinder { ends, radius, edge } => json_fields(
                [(KIND, json(&"cylinder")?), ("ends", json(ends)?), ("radius", json(radius)?)]
                    .into_iter()
                    .chain(edge_fields(edge)?),
            ),
            Node::Cone { ends, base_radius, top_radius, edge } => json_fields(
                [
                    (KIND, json(&"cone")?),
                    ("ends", json(ends)?),
                    ("base_radius", json(base_radius)?),
                    ("top_radius", json(top_radius)?),
                ]
                .into_iter()
                .chain(edge_fields(edge)?),
            ),
            Node::Torus { centre, major_radius, minor_radius, turn } if turn_is_none(turn) => {
                json_object([
                    (KIND, json(&"torus")?),
                    ("centre", json(centre)?),
                    ("major_radius", json(major_radius)?),
                    ("minor_radius", json(minor_radius)?),
                ])
            }
            Node::Torus { centre, major_radius, minor_radius, turn } => json_object([
                (KIND, json(&"torus")?),
                ("centre", json(centre)?),
                ("major_radius", json(major_radius)?),
                ("minor_radius", json(minor_radius)?),
                ("turn", json(turn)?),
            ]),
            // Flat fields beside a string tag. See [`CellMap`] for why, and note that the
            // rectangular arm writes exactly the object it wrote before cylindrical existed.
            Node::Lattice { cell, map, phase, frame } => {
                // A sheet writes exactly the field it wrote before the phase existed and writes
                // no tag at all, so a document nobody has switched is byte-identical and its
                // shader is too. Solid is the new thing, so solid is what says so — the same
                // rule the frame below follows, and for the same reason.
                let mut sized: Vec<(&str, Value)> = match phase {
                    LatticePhase::Sheet { thickness, offset } => {
                        let mut sized = vec![("thickness", json(thickness)?)];
                        // Written only when it says something, so a sheet nobody has offset is
                        // the object it always was.
                        if *offset != ParamRef::Literal(0.0) {
                            sized.push(("offset", json(offset)?));
                        }
                        sized
                    }
                    LatticePhase::Solid { density } => {
                        vec![("phase", json(&"solid")?), ("density", json(density)?)]
                    }
                    LatticePhase::Strut { radius } => {
                        vec![("phase", json(&"strut")?), ("radius", json(radius)?)]
                    }
                };
                // A stochastic cell's own two numbers, flat beside the phase's. Every other cell
                // writes neither, so every lattice saved before these existed is unchanged.
                if let (Some(jitter), Some(seed)) = (cell.jitter(), cell.seed()) {
                    sized.push(("jitter", json(jitter)?));
                    sized.push(("seed", json(&seed)?));
                }
                // Written only when it says something. A lattice nobody has aimed carries the
                // world frame, and a document from before frames existed means the same thing,
                // so leaving both out keeps this arm writing exactly the object it wrote
                // before -- and keeps an older build from meeting a field it has to refuse.
                let aimed: Vec<(&str, Value)> = if frame.is_default() {
                    Vec::new()
                } else {
                    // One nested field, because a `Frame` is one thing and several node kinds
                    // will carry it. A *new* field name rather than a changed one, so an older
                    // build meets something it has never heard of and refuses the node with an
                    // explanation — which is what D3 promises — instead of meeting a field
                    // whose JSON type has changed under it, which is the case a reader cannot
                    // refuse gracefully.
                    vec![("frame", json(frame)?)]
                };
                match map {
                CellMap::Rectangular { cell: size } => json_fields(
                    [
                        (KIND, json(&"lattice")?),
                        ("cell", json(&cell.name())?),
                        ("map", json(&"rectangular")?),
                        ("cell_size", json(size)?),
                    ]
                    .into_iter()
                    .chain(sized.clone())
                    .chain(aimed.clone()),
                ),
                CellMap::Spherical { cell_radius, azimuth_count, polar_count } => json_fields(
                    [
                        (KIND, json(&"lattice")?),
                        ("cell", json(&cell.name())?),
                        ("map", json(&"spherical")?),
                        ("cell_radius", json(cell_radius)?),
                        ("azimuth_count", json(azimuth_count)?),
                        ("polar_count", json(polar_count)?),
                    ]
                    .into_iter()
                    .chain(sized.clone())
                    .chain(aimed.clone()),
                ),
                CellMap::Cylindrical { cell_radius, cell_height, arc_count } => json_fields(
                    [
                        (KIND, json(&"lattice")?),
                        ("cell", json(&cell.name())?),
                        ("map", json(&"cylindrical")?),
                        ("cell_radius", json(cell_radius)?),
                        ("cell_height", json(cell_height)?),
                        ("arc_count", json(arc_count)?),
                    ]
                    .into_iter()
                    .chain(sized.clone())
                    .chain(aimed.clone()),
                ),
                CellMap::Conformal { surface, cells_per_face, depth, thickness } => {
                    // Written only when there is one, so a lattice that tiles forever writes
                    // exactly the object it wrote before shells existed.
                    let shell: Vec<(&str, Value)> = match thickness {
                        Some(t) => vec![("shell", json(t)?)],
                        None => Vec::new(),
                    };
                    json_fields(
                        [
                            (KIND, json(&"lattice")?),
                            ("cell", json(&cell.name())?),
                            ("map", json(&"conformal")?),
                            ("surface", json(surface)?),
                            ("cells_per_face", json(cells_per_face)?),
                            ("cell_depth", json(depth)?),
                        ]
                        .into_iter()
                        .chain(shell)
                        .chain(sized)
                        .chain(aimed),
                    )
                }
                }
            }
            Node::Extrude { outline, sweep, start, height, sides, draft, corners, rims } => {
                // Each left out when it is what an older document means, so an extrude that
                // uses none of them writes exactly what it wrote before the fields existed and
                // a diff of two saves stays readable: `symmetric` only when true, as it always
                // was; `start` only when not nought; side two's pair only when there is one.
                let both: Vec<(&str, Value)> = match sides {
                    Sides::One => vec![],
                    Sides::Symmetric => vec![("symmetric", json(&true)?)],
                    Sides::Two { height, draft } => {
                        vec![("height_2", json(height)?), ("draft_2", json(draft)?)]
                    }
                };
                let begun = match start {
                    ParamRef::Literal(at) if *at == 0.0 => vec![],
                    _ => vec![("start", json(start)?)],
                };
                // And how the draft cuts the corners, only when it is not the rounding every
                // drafted solid did before there was a choice.
                let cut: Vec<(&str, Value)> = match corners.is_default() {
                    true => vec![],
                    false => vec![("corners", json(corners)?)],
                };
                json_fields(
                    [(KIND, json(&"extrude")?)]
                        .into_iter()
                        .chain(outline_fields(outline)?)
                        .chain([
                            ("sweep", json(sweep)?),
                            ("height", json(height)?),
                            ("draft", json(draft)?),
                        ])
                        .chain(begun)
                        .chain(both)
                        .chain(cut)
                        .chain(rim_fields(rims)?),
                )
            }
            Node::Revolve { outline, frame, angle, sides, rims } => {
                // Left out when they are what every revolve was, a full turn one way, so a
                // revolve that uses neither writes exactly what it wrote before they existed.
                let turned = match angle {
                    ParamRef::Literal(degrees) if *degrees == 360.0 => vec![],
                    _ => vec![("angle", json(angle)?)],
                };
                let both: Vec<(&str, Value)> = match sides {
                    AngleSides::One => vec![],
                    AngleSides::Symmetric => vec![("symmetric", json(&true)?)],
                    AngleSides::Two { angle } => vec![("angle_2", json(angle)?)],
                };
                json_fields(
                    [(KIND, json(&"revolve")?)]
                        .into_iter()
                        .chain(outline_fields(outline)?)
                        .chain([("frame", json(frame)?)])
                        .chain(turned)
                        .chain(both)
                        .chain(rim_fields(rims)?),
                )
            }
            Node::Plane { origin, flipped, turn } if turn_is_none(turn) => json_object([
                (KIND, json(&"plane")?),
                ("origin", json(origin)?),
                ("flipped", json(flipped)?),
            ]),
            Node::Plane { origin, flipped, turn } => json_object([
                (KIND, json(&"plane")?),
                ("origin", json(origin)?),
                ("flipped", json(flipped)?),
                ("turn", json(turn)?),
            ]),
            Node::Point { at } => json_object([(KIND, json(&"point")?), ("at", json(at)?)]),
            Node::Capsule { ends, start_radius, end_radius } => json_object([
                (KIND, json(&"capsule")?),
                ("ends", json(ends)?),
                ("start_radius", json(start_radius)?),
                ("end_radius", json(end_radius)?),
            ]),
            Node::Sweep { outline, spine, section, law, rims } => json_fields(
                [(KIND, json(&"sweep")?)]
                    .into_iter()
                    .chain(outline_fields(outline)?)
                    .chain([("spine", json(spine)?), ("section", json(section)?)])
                    .chain(law_fields(law)?)
                    .chain(rim_fields(rims)?),
            ),
            Node::Loft { sections, spine, smooth, rims } => {
                let mut listed: Vec<Value> = Vec::with_capacity(sections.len());
                for section in sections {
                    listed.push(json_fields(outline_fields(section)?.into_iter()));
                }
                let mut fields: Vec<(&str, Value)> =
                    vec![(KIND, json(&"loft")?), ("sections", Value::Array(listed))];
                // Left out when derived, which is what a loft with no centerline means: there
                // is no curve to name, and naming nothing would be a field an older reader has
                // to be taught to read as absent.
                if let Some(spine) = spine {
                    fields.push(("spine", json(spine)?));
                }
                if !smooth.is_default() {
                    fields.push(("smooth", json(smooth)?));
                }
                fields.extend(rim_fields(rims)?);
                json_fields(fields.into_iter())
            }
            Node::Pipe { curve, radius } => json_object([
                (KIND, json(&"pipe")?),
                ("curve", json(curve)?),
                ("radius", json(radius)?),
            ]),
            // **The fillet writes flat, as `radius` and `continuity`.** It is a type in the
            // code and two fields on disk, which is deliberate: making it a nested object would
            // move every blend ever saved for no gain, and the format's job is to be stable
            // rather than to mirror the types.
            Node::Blend { op, a, b, fillet } => json_fields(
                [
                    (KIND, json(&"blend")?),
                    ("op", json(op)?),
                    ("radius", json(&fillet.radius)?),
                    ("continuity", json(&fillet.continuity)?),
                ]
                .into_iter()
                // A chamfer's second leg, 3.22's, written only where there is one: every blend
                // written before it is written as it was.
                .chain(fillet.second.as_ref().map(|s| json(s).map(|v| ("second", v))).transpose()?)
                .chain(slot_field("a", a)?)
                .chain(slot_field("b", b)?),
            ),
            Node::Morph { a, b, t } => json_fields(
                [(KIND, json(&"morph")?), ("t", json(t)?)]
                    .into_iter()
                    .chain(slot_field("a", a)?)
                    .chain(slot_field("b", b)?),
            ),
            Node::Boolean { op, inputs } => json_object([
                (KIND, json(&"boolean")?),
                ("op", json(op)?),
                ("inputs", slots_array(inputs)?),
            ]),
            // **`arrangement`, not `kind`, and that was a real bug.** [`KIND`] *is* the string
            // `"kind"`, so writing the pattern's own kind under the same name overwrote the
            // node's: a saved pattern came back tagged `"radial"`, which is not a node kind this
            // build knows, so it was preserved and refused and stopped being a pattern.
            //
            // Nothing caught it, and the reason is worth keeping. A round trip through this
            // module still *passed*: an unrecognised node is preserved byte for byte on purpose
            // (D3), so the JSON went out and came back identical while the thing it described
            // quietly stopped existing. Comparing the text is not the same question as comparing
            // the node, which is what `a_pattern_comes_back_as_a_pattern` now asks.
            Node::Pattern { input, count, kind, join, fillet, options } => {
                let mut fields = match kind {
                    PatternKind::Linear { step } => vec![
                        (KIND, json(&"pattern")?),
                        ("count", count_json(count)?),
                        ("arrangement", json(&"linear")?),
                        ("step", json(step)?),
                        ("join", json(join)?),
                        ("radius", json(&fillet.radius)?),
                        ("continuity", json(&fillet.continuity)?),
                    ],
                    PatternKind::Radial { frame } => vec![
                        (KIND, json(&"pattern")?),
                        ("count", count_json(count)?),
                        ("arrangement", json(&"radial")?),
                        ("frame", json(frame)?),
                        ("join", json(join)?),
                        ("radius", json(&fillet.radius)?),
                        ("continuity", json(&fillet.continuity)?),
                    ],
                };
                // Written through the same helper as every other slot, so an emptied pattern omits
                // the key rather than naming a block that is no longer in it.
                fields.extend(slot_field("input", input)?);
                // **Each option only where it is not what an older document meant**, so a pattern
                // that uses none of them writes the bytes it always wrote. And only the ones that
                // mean something to this kind: a ring's file never carries a row's spacing.
                if options.symmetric {
                    fields.push(("symmetric", json(&true)?));
                }
                if !options.skip.is_empty() {
                    fields.push(("skip", json(&options.skip)?));
                }
                match kind {
                    PatternKind::Linear { .. } => {
                        if options.spacing != Spacing::Step {
                            fields.push(("spacing", json(&options.spacing)?));
                        }
                        if let Some(second) = &options.second {
                            fields.push(("step2", json(&second.step)?));
                            fields.push(("count2", count_json(&second.count)?));
                        }
                    }
                    PatternKind::Radial { .. } => {
                        if options.angle != ParamRef::Literal(FULL_TURN_DEGREES) {
                            fields.push(("angle", json(&options.angle)?));
                        }
                    }
                }
                json_fields(fields)
            }
            Node::Shell { input, thickness } => json_fields(
                [
                    (KIND, json(&"shell")?),
                    ("thickness", json(thickness)?),
                ]
                .into_iter()
                .chain(slot_field("input", input)?),
            ),
            // Two arms until 112, one for a wired ramp and one for an unwired one. Every slot
            // can be empty now and `slot_field` omits the key for all of them, so the special
            // case is the general rule and the duplicate arm is gone.
            Node::Ramp { input, in_min, in_max, out_min, out_max, shape } => json_fields(
                [
                    (KIND, json(&"ramp")?),
                    ("in_min", json(in_min)?),
                    ("in_max", json(in_max)?),
                    ("out_min", json(out_min)?),
                    ("out_max", json(out_max)?),
                    ("shape", json(shape)?),
                ]
                .into_iter()
                .chain(slot_field("input", input)?),
            ),
            Node::Twist { input, degrees_per_mm, frame } => json_fields(
                [
                    (KIND, json(&"twist")?),
                    ("degrees_per_mm", json(degrees_per_mm)?),
                    ("frame", json(frame)?),
                ]
                .into_iter()
                .chain(slot_field("input", input)?),
            ),
            // Written as the field it is, so a target that is a point is not mistaken for a
            // missing curve. A file from before points could be attracted to carries `curve`
            // and still reads, because that is exactly what `Attractor::Curve` means.
            Node::Attract { input, toward: Attractor::Curve(curve), strength, falloff } => {
                json_fields(
                    [
                        (KIND, json(&"attract")?),
                        ("curve", json(curve)?),
                        ("strength", json(strength)?),
                        ("falloff", json(falloff)?),
                    ]
                    .into_iter()
                    .chain(slot_field("input", input)?),
                )
            }
            Node::Attract { input, toward: Attractor::Point(node), strength, falloff } => {
                json_fields(
                    [
                        (KIND, json(&"attract")?),
                        ("point", json(node)?),
                        ("strength", json(strength)?),
                        ("falloff", json(falloff)?),
                    ]
                    .into_iter()
                    .chain(slot_field("input", input)?),
                )
            }
            Node::Mirror { input, frame, join } => json_fields(
                [
                    (KIND, json(&"mirror")?),
                    ("frame", json(frame)?),
                    ("join", json(join)?),
                ]
                .into_iter()
                .chain(slot_field("input", input)?),
            ),
            Node::Offset { input, distance } => json_fields(
                [
                    (KIND, json(&"offset")?),
                    ("distance", json(distance)?),
                ]
                .into_iter()
                .chain(slot_field("input", input)?),
            ),
            Node::Transform { input, translation, rotation, scale, about } => {
                // Left out at the origin, which is what every placement written before this
                // existed meant, so a save of one of those is byte for byte what it was.
                let pivot = match about == &crate::Place::origin() {
                    true => vec![],
                    false => vec![("about", json(about)?)],
                };
                json_fields(
                    [
                        (KIND, json(&"transform")?),
                        ("translation", json(translation)?),
                        ("rotation", json(rotation)?),
                        ("scale", json(scale)?),
                    ]
                    .into_iter()
                    .chain(pivot)
                    .chain(slot_field("input", input)?),
                )
            }
            Node::Unknown(unknown) => unknown.raw.clone(),
        };
        value.serialize(s)
    }
}

fn json<T: Serialize, E: serde::ser::Error>(value: &T) -> Result<Value, E> {
    serde_json::to_value(value).map_err(E::custom)
}

/// A slot's field, or nothing at all when the slot is empty.
///
/// **Absence, never a null**, which is the arrangement [`Node::Loft`]'s spine already has and for
/// the reason written there: naming nothing would be a field an older reader has to be taught to
/// read as absent. A `null` would be worse than untidy here — `read` would meet it where a
/// [`NodeId`] belongs and answer a hard error, so writing one would turn an empty slot into a file
/// that does not open.
fn slot_field<'k, E: serde::ser::Error>(
    key: &'k str,
    slot: &Operand,
) -> Result<Option<(&'k str, Value)>, E> {
    match slot.source() {
        Some(id) => Ok(Some((key, json(&id)?))),
        None => Ok(None),
    }
}

/// A *list* of slots, where an empty one is written `null`.
///
/// **The one place a null is right, because here position is the information.** A boolean's
/// operands are ordered and `BooleanOp::Difference` folds `kept.max(-d)` from the first of them,
/// so an empty slot has to keep its index: dropping it from the array would silently promote
/// operand 1 to "the thing being kept from" and no error anywhere would say so. A scalar slot can
/// be absent because nothing else depends on where it sat; an element of a list cannot.
fn slots_array<E: serde::ser::Error>(slots: &[Operand]) -> Result<Value, E> {
    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        out.push(match slot.source() {
            Some(id) => json(&id)?,
            None => Value::Null,
        });
    }
    Ok(Value::Array(out))
}

fn json_object<const N: usize>(fields: [(&str, Value); N]) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// The same, over an iterator, for an arm whose field list is not fixed at compile time.
///
/// One arm needs it: a lattice writes its frame only when it has been aimed, so the object it
/// produces is two fields longer in that case and the array's length is not a constant.
fn json_fields<'a>(fields: impl IntoIterator<Item = (&'a str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let raw = Value::deserialize(d)?;
        let Some(object) = raw.as_object() else {
            // Not even an object: nothing here to interpret, but still not ours to discard.
            return Ok(Node::Unknown(UnknownNode::of_unknown_kind(raw)));
        };
        let Some(kind) = object.get(KIND).and_then(Value::as_str) else {
            return Ok(Node::Unknown(UnknownNode::of_unknown_kind(raw.clone())));
        };

        // What this build knows each kind to consist of. A field outside its kind's list is
        // the signal that something newer wrote this node.
        let known: &[&str] = match kind {
            "cage" => &[KIND, "cage"],
            "mesh" => &[KIND, "mesh"],
            "sphere" => &[KIND, "centre", "radius"],
            // Both, because a `"cube"` written by an older build still has to open. Only
            // `"cuboid"` is ever written.
            "cube" => &[KIND, "centre", "size", "turn"],
            // `edge` and `edge_continuity` are 3.19's.
            "cuboid" => &[KIND, "centre", "size", "turn", "edge", "edge_continuity"],
            "skewed_box" => &[KIND, "centre", "edges"],
            "cylinder" => &[KIND, "ends", "radius", "edge", "edge_continuity"],
            // `edge` and `edge_continuity` are 3.20's on a cone.
            "cone" => &[KIND, "ends", "base_radius", "top_radius", "edge", "edge_continuity"],
            "torus" => &[KIND, "centre", "major_radius", "minor_radius", "turn"],
            // Every field any map or phase may write. A node carrying the wrong set for its
            // tags is caught when it is read, not here — this list is only about what this
            // build has heard of at all.
            "lattice" => &[
                KIND,
                "cell",
                "map",
                "cell_size",
                "cell_radius",
                "cell_height",
                "arc_count",
                "azimuth_count",
                "polar_count",
                "surface",
                "cells_per_face",
                "cell_depth",
                "shell",
                "phase",
                "jitter",
                "seed",
                "thickness",
                "offset",
                "density",
                "radius",
                "frame",
            ],
            "plane" => &[KIND, "origin", "flipped", "turn"],
            "point" => &[KIND, "at"],
            "capsule" => &[KIND, "ends", "start_radius", "end_radius"],
            "pipe" => &[KIND, "curve", "radius"],
            // A wall is a different set of fields and not more of them, here as on an extrude
            // and for the reason given there.
            // And the law's four, written flat and only when set (`sweep-v2-spec.md` §6).
            // A loft's sections are objects inside its own list, so their fields are not the
            // node's to declare.
            // `rims` and `rims_continuity` are 3.17's, on all three profile solids
            // (`round-chamfer-spec.md` §4); a build before them meets an unknown field and keeps
            // the node rather than drawing it with a hard rim it was not.
            "loft" => &[KIND, "sections", "spine", "smooth", "rims", "rims_continuity", "rims_at", "rims_second"],
            "sweep" if object.contains_key("wall") => &[
                KIND, "wall", "thickness", "side", "ends", "spine", "section", "taper_scale",
                "taper_draft", "twist", "law", "extent", "corners", "rims", "rims_continuity", "rims_at", "rims_second",
            ],
            "sweep" => &[
                KIND, "profile", "spine", "section", "taper_scale", "taper_draft", "twist", "law",
                "extent", "corners", "rims", "rims_continuity", "rims_at", "rims_second",
            ],
            // "frame" is listed beside "sweep" for the same reason a mirror's old spellings
            // are: an extrude used to carry one, documents written that way exist, and they
            // are migrated on read rather than met as something unknown.
            // "symmetric" is written only when true and read either way -- and was missing
            // here, so every document holding a symmetric extrude was refused on open with
            // the field named as unknown, and the node could not evaluate. Found by the owner
            // on `plane-test.fform`, which opened to a viewport that never settled.
            // **A wall is a different set of fields, not more of them.** An extrude is built
            // from a profile or from a wall and never both, so a node that names a wall may
            // not also name a profile — nor carry the frame only a profile ever came with —
            // and one that names a profile may not carry a thickness. Either is a file saying
            // something this build cannot mean, and is kept and refused like any other.
            // `start`, `height_2` and `draft_2` are format 3.8's: where the material begins
            // along the sweep, and a second side with a depth and a draft of its own
            // (`extrude-revolve-options-spec.md` §2.1). A node saying both `symmetric` and
            // `height_2` says two things about its sides, and is refused below.
            "extrude" if object.contains_key("wall") => &[
                KIND, "wall", "thickness", "side", "ends", "sweep", "start", "height", "draft",
                "symmetric", "height_2", "draft_2", "corners", "rims", "rims_continuity", "rims_at", "rims_second",
            ],
            "extrude" => &[
                KIND, "profile", "sweep", "start", "height", "draft", "symmetric", "height_2",
                "draft_2", "corners", "frame", "rims", "rims_continuity", "rims_at", "rims_second",
            ],
            // `angle`, `symmetric` and `angle_2` are format 3.8's: a partial revolve, and which
            // way about the profile's half-plane it turns (`extrude-revolve-options-spec.md`
            // §3.1). As on an extrude, `symmetric` beside `angle_2` is refused.
            "revolve" if object.contains_key("wall") => {
                &[KIND, "wall", "thickness", "side", "ends", "frame", "angle", "symmetric", "angle_2"]
            }
            // `rims`, `rims_continuity` and `rims_at` are 3.21's on a revolve.
            "revolve" => &[KIND, "profile", "frame", "angle", "symmetric", "angle_2", "rims", "rims_continuity", "rims_at", "rims_second"],
            // `second` is 3.22's: a chamfer's second leg.
            "blend" => &[KIND, "op", "a", "b", "radius", "continuity", "second"],
            "boolean" => &[KIND, "op", "inputs"],
            "morph" => &[KIND, "a", "b", "t"],
            "transform" => &[KIND, "input", "translation", "rotation", "scale", "about"],
            "offset" => &[KIND, "input", "distance"],
            // The old spellings stay listed beside "frame": documents carrying them are still
            // out there and are migrated on read, so meeting one is not meeting something
            // unknown. See `read_frame`.
            "mirror" => &[
                KIND, "input", "frame", "plane", "axis", "origin", "plane_node", "join",
            ],
            "twist" => &[KIND, "input", "degrees_per_mm", "frame", "axis", "origin"],
            "attract" => &[KIND, "input", "curve", "point", "strength", "falloff"],
            "ramp" => &[KIND, "input", "in_min", "in_max", "out_min", "out_max", "shape"],
            "shell" => &[KIND, "input", "thickness"],
            "pattern" => &[
                KIND,
                "input",
                "count",
                "arrangement",
                "step",
                "frame",
                "axis",
                "origin",
                "join",
                "radius",
                "continuity",
                "symmetric",
                "skip",
                "spacing",
                "step2",
                "count2",
                "angle",
            ],
            _ => return Ok(Node::Unknown(UnknownNode::of_unknown_kind(raw.clone()))),
        };
        let unrecognised: Vec<String> = object
            .keys()
            .filter(|k| !known.contains(&k.as_str()))
            .cloned()
            .collect();
        if !unrecognised.is_empty() {
            return Ok(Node::Unknown(UnknownNode::with_unrecognised_fields(
                raw.clone(),
                unrecognised,
            )));
        }

        // A field whose *vocabulary* has grown reads like a broken file to serde — an unknown
        // enum variant is a hard error — and failing the whole document over one is the
        // opposite of what the format promises. `continuity: "g2"` is the case this exists
        // for: written by a build that has curvature-continuous blends, opened by one that
        // does not. Preserved and refused, exactly like an unknown node kind.
        //
        // Only these two fields are checked, because only these two are vocabularies rather
        // than data. A malformed *value* — a radius that is not a number — stays an error,
        // since no future build makes that meaningful.
        let vocabularies: &[(&str, &[&str])] = match kind {
            "boolean" => &[("op", &["union", "difference", "intersection"])],
            // All three parts of a lattice are expected to grow — more cells, more
            // tilings, and a phase whose vocabulary opens the moment a cell is not a minimal
            // surface — so all three are declared here rather than only the ones with variants
            // today.
            "lattice" => &[
                (
                    "cell",
                    &[
                        "gyroid",
                        "schwarz_p",
                        "schwarz_d",
                        "schoen_iwp",
                        "neovius",
                        "fischer_koch_s",
                        "cubic",
                        "bcc",
                        "fcc",
                        "octet",
                        "diamond",
                        "honeycomb",
                        "voronoi_foam",
                        "voronoi_beam",
                    ],
                ),
                // **Every map the reader below can build, and `conformal` was not among them.**
                // The arm that reads one has been there since the map was, and so has the
                // writer; this list was never told, so a conformal lattice worked for as long
                // as the session that made it and came back from its own save as "Unreadable",
                // leaving the part with no bodies. Nothing reported it, because refusing a
                // value this list has not heard of is the format doing what it promises.
                // `every_map_this_build_writes_is_one_it_reads` holds the two together now.
                ("map", &["rectangular", "cylindrical", "spherical", "conformal"]),
                ("phase", &["sheet", "solid", "strut"]),
            ],
            "mirror" => &[("axis", &["x", "y", "z"]), ("plane", &["axis", "node"])],
            "cuboid" | "cylinder" | "cone" => &[("edge_continuity", &["g1", "g2", "chamfer"])],
            "twist" => &[("axis", &["x", "y", "z"])],
            // How the section is held, which is expected to grow: a sweep that follows a second
            // rail is the obvious next one, and a build that has not heard of it must refuse the
            // node rather than quietly sweeping it square.
            //
            // And where a wall's material lies and how it stops, on each of the three blocks
            // that take one. Both are expected to grow — a mitred end, a side given as an offset
            // — and a build that has not heard of one must not draw the wall somewhere else
            // instead. Said three times rather than once because this table is by kind; the
            // words are the same three times because the type is the same (D9).
            "sweep" => &[
                ("section", &["square", "skewed", "parallel"]),
                ("side", &["centre", "left", "right"]),
                ("ends", &["square", "round"]),
                ("rims_continuity", &["g1", "g2", "chamfer"]), ("rims_at", &["both", "start", "end"]),
            ],
            "extrude" | "revolve" => &[
                ("side", &["centre", "left", "right"]),
                ("ends", &["square", "round"]),
                ("rims_continuity", &["g1", "g2", "chamfer"]), ("rims_at", &["both", "start", "end"]),
            ],
            "loft" => &[("smooth", &["g0", "g1"]), ("rims_continuity", &["g1", "g2", "chamfer"]), ("rims_at", &["both", "start", "end"])],
            // A shape this build has not heard of decides how fast the ramp moves, and
            // therefore the bound everything above it divides by. Refused, not guessed at.
            "ramp" => &[("shape", &["linear", "smooth", "smoother"])],
            // A pattern's fold carries the same fillet a blend does, so the same vocabulary — and
            // it was missing here, which made a pattern written by a build with a newer
            // continuity a hard parse error rather than a node kept and refused.
            "pattern" => &[
                ("arrangement", &["linear", "radial"]),
                ("axis", &["x", "y", "z"]),
                ("continuity", &["g1", "g2", "chamfer"]),
                ("spacing", &["step", "extent"]),
            ],
            "blend" => &[
                ("op", &["union", "difference", "intersection"]),
                ("continuity", &["g1", "g2", "chamfer"]),
            ],
            _ => &[],
        };
        for (name, known_values) in vocabularies {
            if let Some(text) = object.get(*name).and_then(Value::as_str)
                && !known_values.contains(&text)
            {
                return Ok(Node::Unknown(UnknownNode::with_unrecognised_value(
                    raw.clone(),
                    (*name).to_string(),
                    text.to_string(),
                )));
            }
        }

        // A *missing* field is a different thing from an unrecognised one: it means the file
        // is damaged rather than newer, and accepting it would preserve nothing.
        let field = |name: &str| -> Result<&Value, D::Error> {
            object
                .get(name)
                .ok_or_else(|| D::Error::custom(format!("{kind} node has no {name:?} field")))
        };
        // **A slot: filled when the field is there, empty when it is absent.** Beside `field`
        // rather than through it, because the two answer opposite questions about a missing key —
        // `field` says the file is damaged, and an absent slot says the block is waiting to be
        // told what to read, which is a state somebody saved on purpose (`notebook-spec.md` §7.2).
        // The ramp's own arm read its input this way before 112; every slot can be empty now, so
        // the special case became the rule.
        let slot = |name: &str| -> Result<Operand, D::Error> {
            match object.get(name) {
                Some(value) => Ok(Operand::filled(read(kind, name, value)?)),
                None => Ok(Operand::empty()),
            }
        };
        // The boolean's list, where an empty slot is `null` so that it keeps its index — see
        // `slots_array` for why position is the information there.
        let slot_list = |name: &str| -> Result<Vec<Operand>, D::Error> {
            let mut out = Vec::new();
            for value in read::<Vec<Value>, D::Error>(kind, name, field(name)?)? {
                out.push(match value {
                    Value::Null => Operand::empty(),
                    named => Operand::filled(read(kind, name, &named)?),
                });
            }
            Ok(out)
        };
        Ok(match kind {
            "cage" => Node::Cage(read(kind, "cage", field("cage")?)?),
            "mesh" => Node::Mesh(read(kind, "mesh", field("mesh")?)?),
            "sphere" => Node::Sphere {
                centre: read(kind, "centre", field("centre")?)?,
                radius: read(kind, "radius", field("radius")?)?,
            },
            // One number in all three slots, which is what a cube is. Migrated on read rather
            // than kept as a second kind: two ways to say "box" in the format is two things a
            // reader has to agree with, and they can disagree.
            "cube" => {
                let size: ParamRef = read(kind, "size", field("size")?)?;
                Node::Cuboid {
                    centre: read(kind, "centre", field("centre")?)?,
                    size: [size; 3],
                    turn: read_turn(kind, object)?,
                    edge: Fillet::hard(),
                }
            }
            "cuboid" => Node::Cuboid {
                centre: read(kind, "centre", field("centre")?)?,
                size: read(kind, "size", field("size")?)?,
                turn: read_turn(kind, object)?,
                edge: read_edge(kind, object)?,
            },
            "skewed_box" => Node::SkewedBox {
                centre: read(kind, "centre", field("centre")?)?,
                edges: read(kind, "edges", field("edges")?)?,
            },
            "cylinder" => Node::Cylinder {
                ends: read(kind, "ends", field("ends")?)?,
                radius: read(kind, "radius", field("radius")?)?,
                edge: read_edge(kind, object)?,
            },
            "cone" => Node::Cone {
                ends: read(kind, "ends", field("ends")?)?,
                base_radius: read(kind, "base_radius", field("base_radius")?)?,
                top_radius: read(kind, "top_radius", field("top_radius")?)?,
                edge: read_edge(kind, object)?,
            },
            "torus" => Node::Torus {
                centre: read(kind, "centre", field("centre")?)?,
                major_radius: read(kind, "major_radius", field("major_radius")?)?,
                minor_radius: read(kind, "minor_radius", field("minor_radius")?)?,
                turn: read_turn(kind, object)?,
            },
            "lattice" => {
                // The tag decides which fields are required, so a document that names a map
                // and then omits its parameters is refused by name rather than defaulted into
                // some other lattice. The tag itself was already checked against the known
                // vocabulary above, so anything reaching here is one this build can build.
                let tag: String = read(kind, "map", field("map")?)?;
                let map = match tag.as_str() {
                    "spherical" => CellMap::Spherical {
                        cell_radius: read(kind, "cell_radius", field("cell_radius")?)?,
                        azimuth_count: read(kind, "azimuth_count", field("azimuth_count")?)?,
                        polar_count: read(kind, "polar_count", field("polar_count")?)?,
                    },
                    "cylindrical" => CellMap::Cylindrical {
                        cell_radius: read(kind, "cell_radius", field("cell_radius")?)?,
                        cell_height: read(kind, "cell_height", field("cell_height")?)?,
                        arc_count: read(kind, "arc_count", field("arc_count")?)?,
                    },
                    "conformal" => CellMap::Conformal {
                        surface: read(kind, "surface", field("surface")?)?,
                        cells_per_face: read(
                            kind,
                            "cells_per_face",
                            field("cells_per_face")?,
                        )?,
                        depth: read(kind, "cell_depth", field("cell_depth")?)?,
                        // Absent means it tiles forever, which is what every conformal lattice
                        // written before shells existed does.
                        thickness: match object.get("shell") {
                            Some(v) => Some(read(kind, "shell", v)?),
                            None => None,
                        },
                    },
                    _ => CellMap::Rectangular {
                        cell: read(kind, "cell_size", field("cell_size")?)?,
                    },
                };
                // Absent means the world frame, which is what every document written before
                // frames existed means and what a lattice nobody has aimed still means. Read
                // as a pair rather than one at a time: half a frame is not a thing to have,
                // and defaulting the missing half would aim the lattice somewhere nobody
                // asked for.
                // Absent means the world frame, which is what every document written before
                // frames existed means and what a lattice nobody has aimed still means.
                let frame = match object.get("frame") {
                    Some(v) => read(kind, "frame", v)?,
                    None => Frame::default(),
                };
                // Absent means sheet, which is what every lattice written before the phase
                // existed is and what a lattice nobody has switched still is. The tag decides
                // which number is required, the same way the map's tag does above: a document
                // that says solid and then carries a thickness is refused by name rather than
                // read as a density that happens to be 2.
                let phase = match object.get("phase").and_then(Value::as_str) {
                    Some("solid") => LatticePhase::Solid {
                        density: read(kind, "density", field("density")?)?,
                    },
                    Some("strut") => LatticePhase::Strut {
                        radius: read(kind, "radius", field("radius")?)?,
                    },
                    _ => LatticePhase::Sheet {
                        thickness: read(kind, "thickness", field("thickness")?)?,
                        // Absent means centred, which is what every sheet written before an
                        // offset existed is.
                        offset: match object.get("offset") {
                            Some(v) => read(kind, "offset", v)?,
                            None => ParamRef::Literal(0.0),
                        },
                    },
                };
                Node::Lattice {
                    cell: {
                        let name: String = read(kind, "cell", field("cell")?)?;
                        // Absent means the default, which only a hand-written document would
                        // leave out; a cell with neither number ignores both.
                        let jitter = match object.get("jitter") {
                            Some(v) => read(kind, "jitter", v)?,
                            None => ParamRef::Literal(0.75),
                        };
                        let seed = match object.get("seed") {
                            Some(v) => read(kind, "seed", v)?,
                            None => 0u64,
                        };
                        // The vocabulary check above has already refused a name this build
                        // does not know, so this only fails if that table and `from_name` part.
                        LatticeCell::from_name(&name, jitter, seed)
                            .ok_or_else(|| serde::de::Error::custom(format!("unknown cell {name}")))?
                    },
                    map,
                    phase,
                    frame,
                }
            }
            "plane" => Node::Plane {
                origin: read(kind, "origin", field("origin")?)?,
                flipped: read(kind, "flipped", field("flipped")?)?,
                turn: read_turn(kind, object)?,
            },
            "point" => Node::Point { at: read(kind, "at", field("at")?)? },
            "capsule" => Node::Capsule {
                ends: read(kind, "ends", field("ends")?)?,
                start_radius: read(kind, "start_radius", field("start_radius")?)?,
                end_radius: read(kind, "end_radius", field("end_radius")?)?,
            },
            "extrude" => Node::Extrude {
                outline: read_outline(kind, object)?,
                sweep: read_sweep(kind, object)?,
                height: read(kind, "height", field("height")?)?,
                // Absent means upright, which is what every extrude written before there was a
                // draft was — so an older document keeps meaning what it meant. A *missing*
                // field is the one case where defaulting is right rather than guessing.
                draft: match object.get("draft") {
                    Some(v) => read(kind, "draft", v)?,
                    None => ParamRef::Literal(0.0),
                },
                // Absent means at the plane, which is what every extrude written before this
                // existed was — the same reasoning the draft above is read by.
                start: match object.get("start") {
                    Some(v) => read(kind, "start", v)?,
                    None => ParamRef::Literal(0.0),
                },
                sides: read_sides(kind, object)?,
                corners: read_corners(kind, object)?,
                rims: read_rims(kind, object)?,
            },
            "revolve" => Node::Revolve {
                outline: read_outline(kind, object)?,
                frame: read_frame(kind, object)?,
                // Absent means all the way round, one way, which is what every revolve written
                // before this existed was.
                angle: match object.get("angle") {
                    Some(v) => read(kind, "angle", v)?,
                    None => ParamRef::Literal(360.0),
                },
                sides: read_angle_sides(kind, object)?,
                rims: read_rims(kind, object)?,
            },
            "pipe" => Node::Pipe {
                curve: read(kind, "curve", field("curve")?)?,
                radius: read(kind, "radius", field("radius")?)?,
            },
            "sweep" => Node::Sweep {
                outline: read_outline(kind, object)?,
                spine: read(kind, "spine", field("spine")?)?,
                // Absent in every document written before the choice existed, and square is
                // what those were.
                section: match object.get("section") {
                    Some(v) => read(kind, "section", v)?,
                    None => crate::Section::default(),
                },
                law: read_law(kind, object)?,
                rims: read_rims(kind, object)?,
            },
            // **Each section is an outline object of its own**, read by the reader every other
            // outline is read by, so a section is spelled exactly as an extrude's profile is.
            "loft" => {
                let listed = field("sections")?.as_array().ok_or_else(|| {
                    D::Error::custom(format!("{kind} node's \"sections\" is not a list"))
                })?;
                let mut sections = Vec::with_capacity(listed.len());
                for section in listed {
                    let object = section.as_object().ok_or_else(|| {
                        D::Error::custom(format!("a {kind} section is not an object"))
                    })?;
                    sections.push(read_outline(kind, object)?);
                }
                Node::Loft {
                    sections,
                    // Absent means derived, which is what writing nothing promises.
                    spine: match object.get("spine") {
                        Some(v) => Some(read(kind, "spine", v)?),
                        None => None,
                    },
                    smooth: match object.get("smooth") {
                        Some(v) => read(kind, "smooth", v)?,
                        None => LoftSmooth::default(),
                    },
                    rims: read_rims(kind, object)?,
                }
            }
            "blend" => Node::Blend {
                op: read(kind, "op", field("op")?)?,
                a: slot("a")?,
                b: slot("b")?,
                fillet: Fillet {
                    radius: read(kind, "radius", field("radius")?)?,
                    continuity: read(kind, "continuity", field("continuity")?)?,
                    second: match object.get("second") {
                        Some(v) => Some(read(kind, "second", v)?),
                        None => None,
                    },
                },
            },
            // The axis and its origin default when absent, because a twist written before
            // they existed wound about Z through the world origin — so an older document keeps
            // meaning what it meant. A *missing* field is the one case where defaulting is
            // right rather than guessing: the file predates the question, so there is a true
            // answer to it.
            "pattern" => {
                let tag: String = read(kind, "arrangement", field("arrangement")?)?;
                let repeat = match tag.as_str() {
                    "radial" => PatternKind::Radial {
                        frame: read_frame(kind, object)?,
                    },
                    _ => PatternKind::Linear {
                        step: read(kind, "step", field("step")?)?,
                    },
                };
                Node::Pattern {
                    input: slot("input")?,
                    count: read_count(kind, object)?,
                    kind: repeat,
                    // Absent means the block holds instance zero, which is what every pattern
                    // written before the field existed meant -- so an older document keeps
                    // meaning what it meant. The same reasoning an extrude's `draft` is read by.
                    join: match object.get("join") {
                        Some(v) => read(kind, "join", v)?,
                        None => true,
                    },
                    // Absent means hard, which is the only thing a pattern could do before the
                    // fold could round. The same reasoning `join` above is read by.
                    fillet: Fillet {
                        radius: match object.get("radius") {
                            Some(v) => read(kind, "radius", v)?,
                            None => ParamRef::Literal(0.0),
                        },
                        continuity: match object.get("continuity") {
                            Some(v) => read(kind, "continuity", v)?,
                            None => Continuity::G1,
                        },
                        second: None,
                    },
                    // Each absent option is what the pattern did before it had any, so an older
                    // document keeps meaning what it meant — the reasoning `join` is read by.
                    options: PatternOptions {
                        spacing: match object.get("spacing") {
                            Some(v) => read(kind, "spacing", v)?,
                            None => Spacing::Step,
                        },
                        symmetric: match object.get("symmetric") {
                            Some(v) => read(kind, "symmetric", v)?,
                            None => false,
                        },
                        angle: match object.get("angle") {
                            Some(v) => read(kind, "angle", v)?,
                            None => ParamRef::Literal(FULL_TURN_DEGREES),
                        },
                        second: match (object.get("step2"), object.get("count2")) {
                            (Some(step), Some(count)) => Some(PatternSecond {
                                step: read(kind, "step2", step)?,
                                count: read_count_of(kind, "count2", count)?,
                            }),
                            // One without the other is not a second direction, and guessing the
                            // missing half would invent geometry. Refused with the node kept.
                            (None, None) => None,
                            _ => {
                                return Ok(Node::Unknown(UnknownNode::of_unknown_kind(
                                    raw.clone(),
                                )));
                            }
                        },
                        skip: match object.get("skip") {
                            Some(v) => read(kind, "skip", v)?,
                            None => Vec::new(),
                        },
                    },
                }
            }
            "twist" => Node::Twist {
                input: slot("input")?,
                degrees_per_mm: read(kind, "degrees_per_mm", field("degrees_per_mm")?)?,
                frame: read_frame(kind, object)?,
            },
            // Exactly one of `curve` and `point`, because an attractor names one thing. Both
            // is a file describing two pulls with no rule for combining them; neither is a file
            // describing no pull at all. Refused either way rather than picking one -- D3's
            // rule, and the same reason a node carrying fields this build cannot account for is
            // kept and refused rather than read half-way.
            "attract" => Node::Attract {
                input: slot("input")?,
                toward: match (object.get("curve"), object.get("point")) {
                    (Some(curve), None) => Attractor::Curve(read(kind, "curve", curve)?),
                    (None, Some(node)) => Attractor::Point(read(kind, "point", node)?),
                    _ => {
                        return Ok(Node::Unknown(UnknownNode::of_unknown_kind(raw.clone())));
                    }
                },
                strength: read(kind, "strength", field("strength")?)?,
                falloff: read(kind, "falloff", field("falloff")?)?,
            },
            "mirror" => Node::Mirror {
                input: slot("input")?,
                frame: read_frame(kind, object)?,
                join: read(kind, "join", field("join")?)?,
            },
            "offset" => Node::Offset {
                input: slot("input")?,
                distance: read(kind, "distance", field("distance")?)?,
            },
            "shell" => Node::Shell {
                input: slot("input")?,
                thickness: read(kind, "thickness", field("thickness")?)?,
            },
            // Until 112 the one node whose `input` could be absent, read by hand here. Every slot
            // may be empty now and `slot` reads absence as empty for all of them, so the bespoke
            // match is gone and a ramp is read like anything else.
            "ramp" => Node::Ramp {
                input: slot("input")?,
                in_min: read(kind, "in_min", field("in_min")?)?,
                in_max: read(kind, "in_max", field("in_max")?)?,
                out_min: read(kind, "out_min", field("out_min")?)?,
                out_max: read(kind, "out_max", field("out_max")?)?,
                shape: read(kind, "shape", field("shape")?)?,
            },
            "transform" => Node::Transform {
                input: slot("input")?,
                translation: read(kind, "translation", field("translation")?)?,
                rotation: read(kind, "rotation", field("rotation")?)?,
                scale: read(kind, "scale", field("scale")?)?,
                // Absent means the origin, which is what every placement written before there
                // was a pivot turned about — the same reasoning an extrude reads its draft by.
                about: match object.get("about") {
                    Some(v) => read(kind, "about", v)?,
                    None => crate::Place::origin(),
                },
            },
            "morph" => Node::Morph {
                a: slot("a")?,
                b: slot("b")?,
                t: read(kind, "t", field("t")?)?,
            },
            "boolean" => Node::Boolean {
                op: read(kind, "op", field("op")?)?,
                inputs: slot_list("inputs")?,
            },
            _ => unreachable!("every kind reaching here matched the list above"),
        })
    }
}

/// A block's own turn, or none when the file does not carry one.
///
/// Absent rather than missing: a file written before blocks had a turn describes a block that was
/// not turned, which is exactly what none means. So this is not a D3 case — nothing about the
/// shape is being guessed.
fn read_turn<E>(kind: &str, object: &serde_json::Map<String, Value>) -> Result<Turn, E>
where
    E: serde::de::Error,
{
    match object.get("turn") {
        Some(value) => read(kind, "turn", value),
        None => Ok(no_turn()),
    }
}

/// Reads one field of a known node kind, naming both in the error rather than leaving serde's
/// own message to describe a document the reader cannot see.
fn read<T, E>(kind: &str, name: &str, value: &Value) -> Result<T, E>
where
    T: serde::de::DeserializeOwned,
    E: serde::de::Error,
{
    serde_json::from_value(value.clone())
        .map_err(|e| E::custom(format!("{kind} node's {name:?}: {e}")))
}
