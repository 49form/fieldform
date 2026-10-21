// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The document model — what gets saved, scripted, and exchanged.
//!
//! Two structural decisions shape everything here, both settled in the architecture notes
//! (the commercial-readiness design note):
//!
//! **The two-layer model.** Cage editing is direct and destructive — the cage's current state
//! is the truth, nothing replays. Curves, parameters and the composition graph are parametric:
//! stored, and re-evaluated when their inputs change. Fields are the currency between them.
//!
//! **Lexical scope.** An object has exactly one owning scope, and a scope may reference its
//! own objects and any ancestor's — never a sibling's, never a descendant's. Acyclic by
//! construction. v1 has a single scope (the document); assemblies add levels later without a
//! format break, because [`Part`]s already declare named inputs for a parent to bind.
//!
//! Together those are why element identity never became necessary: nothing replays polygon
//! operations, so no node ever needs to re-find "face 7" after an edit. Vertices and faces
//! stay plain indices; only [`id`]'s few durable things get stable identities.

mod import;
mod place;
mod turn;
mod curve;
mod document;
mod ends;
mod expression;
mod eye;
mod frame;
mod sweep;
mod outline;
mod corners;
mod law;
mod loft;
mod sides;
mod graph;
pub mod id;
mod material;
mod mesh;
mod ownership;
pub mod relation;
mod surface_axis;
mod output;
mod unit;

pub use curve::{Curve, CurveData, CurveError, JOINED_MM};
pub use relation::{CurveEnd, End, JointContinuity, Relation, RelationError, RelationRecord};
pub use surface_axis::SurfaceAxis;
pub use import::StepOptions;
pub use document::{
    Document, ExpressionError, EyeFlag, ImportSettings, ImportSource, InputKind, InputPort,
    ListReadings, Nesting, NodeMeta, ParamValue, Parameter, ParameterError, Part, RenameError,
};
pub use expression::{Expression, ParseError};
pub use eye::{
    drawn_graph, draws_as_overlay_in, eye_flag_by, eye_flag_in, is_visible_among, is_visible_in,
    shown_ingredients_in,
};
pub use graph::{
    no_turn, Attractor, Turn,
    Axis, Banding, BooleanOp, CellMap, Continuity, Critical, Graph,
    LatticeCell,
    Fillet, LatticePhase, Node, Operand, ParamRef, PatternKind, RampShape, RimEnds, Rims,
    UnknownNode,
    Unrecognised,
    // The count resolver and its refusals. One way a `Node::Pattern`'s count becomes an integer,
    // reachable by every crate that reads one — see `pattern_count`.
    pattern_count, CountError, CountFit, PATTERN_COUNT_MAX,
    // What a pattern can do beyond repeating a body evenly — see `PatternOptions`.
    PatternOptions, PatternSecond, Spacing, FULL_TURN_DEGREES,
};
pub use ends::Ends;
pub use frame::Frame;
pub use sweep::{Section, Sweep};
pub use outline::{Outline, WallEnds, WallSide};
pub use corners::{Corners, MITRE_LIMIT};
pub use law::{SweepLaw, Taper};
pub use loft::LoftSmooth;
pub use sides::{AngleSides, Sides};
pub use id::{
    CageId, CurveId, ExportId, MeshId, NodeId, ParamId, PartId, QuadMeshId, RelationId,
    TessellationId,
};
pub use output::{Export, MeshDetail, MeshTolerance, Tessellation};
// `MaterialValues` is named by nothing outside, but the application is handed one in every
// `Preset::material` it turns into a `Material`, and a type another crate receives is one it
// should be able to name, which only the root can promise now that `material` is private.
pub use material::{Channel, Kind, Material, MaterialValues, Preset};

pub use place::Place;
pub use mesh::MeshData;
pub use ownership::{BlockKey, Ownership, OwnershipChange};
pub use unit::Unit;
