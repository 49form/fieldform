// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a block does in the graph, which is the one thing the block list colours a row by.
//!
//! **By graph role, not by subject and not by output.** A block's family says whether it joins
//! bodies, makes one from nothing, changes one, is a curve, a cage, a file, or a reference other
//! blocks are placed or driven by. It is a reading of the block's kind, derived on every draw and
//! stored nowhere: nothing here touches the graph, the file or a result.

use super::Node;

/// The seven families (the block-family notes, section 2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Reads two or more bodies and makes one.
    Combine,
    /// Makes a body from no body.
    Create,
    /// Reads one body and makes one.
    Modify,
    /// A curve, or a loop of them.
    Curve,
    /// A subdivision cage this document owns and edits.
    Cage,
    /// Geometry that came from a file, or a file written from geometry.
    File,
    /// Not a shape: something other blocks are placed by or driven by.
    Reference,
}

impl Family {
    /// Every family, in the order the notes list them.
    pub const ALL: [Family; 7] = [
        Family::Combine,
        Family::Create,
        Family::Modify,
        Family::Curve,
        Family::Cage,
        Family::File,
        Family::Reference,
    ];

    /// What the family is called where a person reads it.
    pub fn label(self) -> &'static str {
        match self {
            Family::Combine => "Combine",
            Family::Create => "Create",
            Family::Modify => "Modify",
            Family::Curve => "Curve",
            Family::Cage => "Cage",
            Family::File => "File",
            Family::Reference => "Reference",
        }
    }

    /// The rule that puts a block in this family, as one clause: the hover line on a row's bar
    /// reads `"{label} — {rule}"`, which is the whole legend there is.
    ///
    /// **Create says "or a formula"**, which the notes' table does not: the Expression block, a
    /// field written as arithmetic, arrived the day the table was written and makes a body from
    /// no body as a sphere does.
    pub fn rule(self) -> &'static str {
        match self {
            Family::Combine => "reads two or more bodies and makes one",
            Family::Create => "makes a body from no body: numbers, curves, a cell or a formula",
            Family::Modify => "reads one body and makes one",
            Family::Curve => "a curve, or a loop of them",
            Family::Cage => "a subdivision cage this document owns and edits",
            Family::File => "geometry that came from a file, or a file written from geometry",
            Family::Reference => "not a shape: something other blocks are placed by or driven by",
        }
    }

    /// The line a row's bar says on hover.
    pub fn hover_line(self) -> String {
        format!("{} \u{2014} {}", self.label(), self.rule())
    }
}

impl Node {
    /// What this block does in the graph.
    ///
    /// **Exhaustive on purpose, with no wildcard arm**: a new kind of node does not compile until
    /// it says which family it is, which is the only check that cannot be forgotten. A wildcard
    /// would file it under whatever the wildcard said and never complain.
    ///
    /// `None` for [`Node::Unknown`] alone — a kind this build cannot read cannot honestly be
    /// filed, so its row draws the error stroke and no bar.
    ///
    /// Three placements that could be argued the other way: a lattice is Create, being a leaf
    /// that reads no body; an infill is Modify, one body in and the same body filled out; a
    /// hole, a thread and a coil are Create, each a cutter a boolean subtracts. A plane is
    /// Reference although it can trim a part as a body, because it is almost always placed on,
    /// sketched on or mirrored about.
    pub fn family(&self) -> Option<Family> {
        Some(match self {
            Node::Boolean { .. } | Node::Blend { .. } | Node::Morph { .. } => Family::Combine,
            Node::Sphere { .. }
            | Node::Cuboid { .. }
            | Node::SkewedBox { .. }
            | Node::Cylinder { .. }
            | Node::Cone { .. }
            | Node::Torus { .. }
            | Node::Capsule { .. }
            | Node::Coil { .. }
            | Node::Thread { .. }
            | Node::Hole { .. }
            | Node::Extrude { .. }
            | Node::Revolve { .. }
            | Node::Sweep { .. }
            | Node::Loft { .. }
            | Node::Pipe { .. }
            | Node::Lattice { .. }
            | Node::Expression { .. } => Family::Create,
            Node::Offset { .. }
            | Node::Shell { .. }
            | Node::Twist { .. }
            | Node::Attract { .. }
            | Node::Transform { .. }
            | Node::Mirror { .. }
            | Node::Scale { .. }
            | Node::Pattern { .. }
            | Node::Infill { .. } => Family::Modify,
            Node::Cage(_) => Family::Cage,
            Node::Mesh(_) => Family::File,
            Node::Plane { .. } | Node::Point { .. } | Node::Ramp { .. } => Family::Reference,
            Node::Unknown(_) => return None,
        })
    }
}
