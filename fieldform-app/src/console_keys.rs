// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What each console verb reads, and what a value may be.
//!
//! **Every block is reachable from the console with every number it takes, and the keys are
//! the block's own field names.** The file, the panel and the console then use one word for one
//! thing (D9), and a script can say anything a click can. Before this module a create verb read
//! nothing -- `sphere r=10 at=40,0,0` made a default sphere and said "Added a sphere" -- and an
//! unknown key was accepted in silence, which is the same failure as a value that falls back to
//! a default without a word. Both are refused here, by name, with what the verb does read.
//!
//! **One table, read three ways.** [`VERBS`] says which keys each operation reads and which
//! serialized field each key covers. `run_command` asks it whether a typed key is unknown;
//! `help <verb>` prints it; and `every_field_a_block_has_is_a_key_its_verb_reads` holds it
//! against the blocks themselves, by serializing a sample of every kind with every field set
//! and checking each field is covered or recorded as not read, with the reason. [`reach_of`] is
//! the other half of that test: an exhaustive match over `Node`, so a kind added without a
//! console verb does not compile until someone says which verb, or why none.
//!
//! **A value is a number, a parameter's name, or a block's name** (`benchmark-suite-instructions.md`
//! §2 A9). Inline expressions are not accepted -- `param name=h expr="s/2"` then `h` -- so there
//! is one expression mechanism and one cycle check. A block's name is accepted only where the
//! kernel accepts a driver (`driven_refusal`): an offset's distance, a morph's amount, a pipe's
//! radius, a blend's radius and a lattice phase's own number. Everywhere else a block's name is refused by name
//! rather than compiled into a graph the kernel then rejects a long way from the keystroke.
//!
//! **And `none` wherever a value names an operand**, which is the one word that empties a slot:
//! `target=none`, `in=none`, `input=none`, read by [`operand_named`] and [`operands_named`]. An
//! empty slot is a legal, savable state (`notebook-spec.md` §7.2), so a line can make a block
//! before it has anything to put in it and a script can build a part top-down -- which was the
//! one thing a click could do and a line could not. The word wins over a block called *none*: the
//! way to name that one is to rename it, as a word that is both a parameter and a block is
//! settled by a rename (A9).
//!
//! **A place, a turn, a pair of ends or a frame is typed from the console, never taken.**
//! `centre=`, `turn=`, `from=`/`to=` and `origin=`/`axis=` make `Place::At`, `Turn::By`,
//! `Ends::At` and `Frame::At`. The reference variants -- `Place::From`/`StartOf`,
//! `Turn::From`/`Along`/`Between`/`Tangent`, `Ends::From`/`On`, `Frame::From`/`Along`/`Between`/
//! `Tangent` -- have no console spelling anywhere, so A1 leaves them out, and this paragraph is
//! the recorded gap. `not_read` cannot hold it: a taken place serializes under the same `centre`
//! key as a typed one, so the drift test counts it covered. A spelling lands with item 81's A6,
//! where curves get names a console can say.
//!
//! Free of the `App` so every rule here is held by a test: an `App` needs a window and a device.

use fieldform_ops::{Args, Value};
use fieldform_scene::{
    Attractor, Axis, BooleanOp, CellMap, Continuity, Ends, Fillet, Frame, LatticeCell,
    LatticePhase, Node, NodeId, Operand, ParamId, ParamRef, PatternKind, Place, RampShape, Turn,
};

/// What a bare word names in the document, asked of the app.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    Nothing,
    Parameter(ParamId),
    Block(NodeId),
    /// A parameter and a block share the word. Refused rather than guessed (A9).
    Both,
}

/// What a slot will take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// A number or a parameter: every slot the kernel will not let a field drive.
    Number,
    /// A number, a parameter, or a block whose field drives it.
    NumberOrField,
}

/// A key a verb reads.
pub struct Key {
    /// The word typed after the verb.
    pub word: &'static str,
    /// The block's serialized fields this key sets, so the drift test can tell a field that is
    /// reachable from one that is not. A flag that only chooses between keys covers nothing.
    pub covers: &'static [&'static str],
}

/// What a verb reads.
pub enum Keys {
    /// Every key the verb reads, and every field of its block it deliberately does not.
    Declared {
        reads: &'static [Key],
        /// `(field, why)`: fields the console does not set, each with its reason. A field here
        /// is a recorded gap, not an oversight; the drift test fails on an unrecorded one.
        not_read: &'static [(&'static str, &'static str)],
    },
    /// The verb reads its own keys in its own code and has not declared them yet. Unknown keys
    /// pass it unchecked. The reason says which item closes the gap. No verb is here since A6
    /// declared the curve verbs; kept for a verb added before its keys are written down, which
    /// is better said in the table than left as a hole in it.
    #[allow(dead_code)]
    Undeclared(&'static str),
}

pub struct Verb {
    /// The operation `dispatch` knows it by.
    pub op: &'static str,
    pub keys: Keys,
}

const fn key(word: &'static str, covers: &'static [&'static str]) -> Key {
    Key { word, covers }
}

/// A key that only picks between others, or names what the command makes.
const fn switch(word: &'static str) -> Key {
    Key { word, covers: &[] }
}

const TARGET: Key = key("target", &["input"]);

/// The table. Keys are the block's serialized field names wherever one exists; where the
/// instructions chose a shorter word (`rate`, `r`, `arc`) both are read and both cover the field.
pub const VERBS: &[Verb] = &[
    Verb {
        op: "create.point",
        keys: Keys::Declared { reads: &[key("at", &["at"])], not_read: &[] },
    },
    Verb {
        op: "create.plane",
        keys: Keys::Declared {
            reads: &[key("origin", &["origin"]), key("turn", &["turn"]), key("flipped", &["flipped"])],
            not_read: &[],
        },
    },
    Verb {
        op: "create.sphere",
        keys: Keys::Declared {
            reads: &[key("radius", &["radius"]), key("centre", &["centre"])],
            not_read: &[],
        },
    },
    Verb {
        op: "create.cube",
        keys: Keys::Declared {
            reads: &[
                key("size", &["size"]),
                key("centre", &["centre"]),
                key("turn", &["turn"]),
                key("edge", &["edge"]),
                key("edge_continuity", &["edge_continuity"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.skewbox",
        keys: Keys::Declared {
            reads: &[key("centre", &["centre"]), key("edges", &["edges"])],
            not_read: &[],
        },
    },
    Verb {
        op: "create.cylinder",
        keys: Keys::Declared {
            reads: &[
                key("radius", &["radius"]),
                key("from", &["from"]),
                key("to", &["to"]),
                key("edge", &["edge"]),
                key("edge_continuity", &["edge_continuity"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.cone",
        keys: Keys::Declared {
            reads: &[
                key("base_radius", &["base_radius"]),
                key("top_radius", &["top_radius"]),
                key("from", &["from"]),
                key("to", &["to"]),
                key("edge", &["edge"]),
                key("edge_continuity", &["edge_continuity"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.torus",
        keys: Keys::Declared {
            reads: &[
                key("major_radius", &["major_radius"]),
                key("minor_radius", &["minor_radius"]),
                key("centre", &["centre"]),
                key("turn", &["turn"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.capsule",
        keys: Keys::Declared {
            reads: &[
                key("start_radius", &["start_radius"]),
                key("end_radius", &["end_radius"]),
                key("from", &["from"]),
                key("to", &["to"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.tpms",
        keys: Keys::Declared {
            reads: &[
                key("cell", &["cell"]),
                key("map", &["map"]),
                key("cell_size", &["cell_size"]),
                key("cell_radius", &["cell_radius"]),
                key("cell_height", &["cell_height"]),
                key("arc", &["arc_count"]),
                key("arc_count", &["arc_count"]),
                key("azimuth", &["azimuth_count"]),
                key("polar", &["polar_count"]),
                key("phase", &["phase"]),
                key("thickness", &["thickness"]),
                key("offset", &["offset"]),
                key("density", &["density"]),
                key("radius", &["radius"]),
                key("origin", &["origin"]),
                key("axis", &["axis"]),
                key("jitter", &["jitter"]),
                key("seed", &["seed"]),
            ],
            not_read: &[
                ("surface", "a conformal map follows a quad mesh, which is picked in the panel"),
                ("cells_per_face", "a conformal map's cells are set in the panel with its mesh"),
                ("cell_depth", "a conformal map's depth is set in the panel with its mesh"),
                ("shell", "a conformal map's shell is set in the panel with its mesh"),
            ],
        },
    },
    Verb {
        op: "create.ramp",
        keys: Keys::Declared {
            reads: &[
                key("in", &["in_min", "in_max"]),
                key("out", &["out_min", "out_max"]),
                key("shape", &["shape"]),
                key("input", &["input"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.boolean",
        keys: Keys::Declared {
            reads: &[
                key("in", &["inputs", "a", "b"]),
                key("op", &["op"]),
                switch("u"),
                switch("union"),
                switch("s"),
                switch("subtract"),
                switch("i"),
                switch("intersect"),
                key("r", &["radius"]),
                key("radius", &["radius"]),
                key("g", &["continuity"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "modify.morph",
        keys: Keys::Declared {
            reads: &[key("a", &["a"]), key("b", &["b"]), key("t", &["t"])],
            not_read: &[],
        },
    },
    Verb {
        op: "modify.offset",
        keys: Keys::Declared { reads: &[TARGET, key("distance", &["distance"])], not_read: &[] },
    },
    Verb {
        op: "modify.shell",
        keys: Keys::Declared { reads: &[TARGET, key("thickness", &["thickness"])], not_read: &[] },
    },
    Verb {
        op: "modify.twist",
        keys: Keys::Declared {
            reads: &[
                TARGET,
                key("rate", &["degrees_per_mm"]),
                key("degrees_per_mm", &["degrees_per_mm"]),
                key("axis", &["axis"]),
                key("origin", &["origin"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "modify.mirror",
        keys: Keys::Declared {
            reads: &[TARGET, key("axis", &["axis"]), key("origin", &["origin"]), key("join", &["join"])],
            not_read: &[],
        },
    },
    Verb {
        op: "modify.attract",
        keys: Keys::Declared {
            reads: &[
                TARGET,
                key("toward", &["curve", "point"]),
                key("strength", &["strength"]),
                key("falloff", &["falloff"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "modify.pattern",
        keys: Keys::Declared {
            reads: &[
                TARGET,
                key("count", &["count"]),
                key("kind", &["arrangement"]),
                key("step", &["step"]),
                key("axis", &["axis"]),
                key("origin", &["origin"]),
                key("r", &["radius"]),
                key("radius", &["radius"]),
                key("g", &["continuity"]),
                key("join", &["join"]),
                key("symmetric", &["symmetric"]),
                key("extent", &["spacing"]),
                key("spacing", &["spacing"]),
                key("angle", &["angle"]),
                key("skip", &["skip"]),
                key("rows", &["count2"]),
                key("count2", &["count2"]),
                key("step2", &["step2"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "modify.move",
        keys: Keys::Declared { reads: &[TARGET, key("by", &["translation"]), switch("wrap")], not_read: &[] },
    },
    Verb {
        op: "modify.turn",
        keys: Keys::Declared {
            reads: &[TARGET, key("by", &["rotation"]), key("about", &["about"]), switch("wrap")],
            not_read: &[],
        },
    },
    Verb {
        op: "modify.scale",
        keys: Keys::Declared {
            reads: &[TARGET, key("by", &["scale"]), key("about", &["about"]), switch("wrap")],
            not_read: &[],
        },
    },
    Verb {
        op: "create.cage",
        keys: Keys::Declared {
            reads: &[],
            not_read: &[("cage", "the cage is the block: its points are edited by hand, not typed")],
        },
    },
    // `delete target=Sphere_0`: what goes, by name -- the echo's spelling of a delete, which must not
    // lean on whatever the replay has selected. Alone, it deletes the selection, as the key does.
    Verb { op: "edit.delete_body", keys: Keys::Declared { reads: &[switch("target")], not_read: &[] } },
    // `split body=Sphere by=Plane`: the body cut and what cuts it, both by name. Its halves are
    // blocks of kinds other verbs make, so it covers no field of its own; declared so a stray key
    // is refused like any other verb's, which until the registry's test found it it was not.
    Verb { op: "modify.split", keys: Keys::Declared { reads: &[switch("body"), switch("by")], not_read: &[] } },
    // Verbs that make no block, declared so their keys are refused like any other's.
    Verb { op: "script.run", keys: Keys::Declared { reads: &[switch("file"), switch("keepgoing")], not_read: &[] } },
    Verb { op: "file.save_as", keys: Keys::Declared { reads: &[switch("file")], not_read: &[] } },
    // **The verbs whose `name` is their own argument**, not a name for a block they made.
    // Declaring it here is what tells `run_command` apart from the rest: anywhere else `name=`
    // is another way of writing `sphere 'ball'`, and here it is the thing being set. The grammar
    // used to decide, and swallowed `name=` for every verb — which is why
    // `environment name=zebra` listed the environments instead of choosing one.
    Verb { op: "view.environment", keys: Keys::Declared { reads: &[switch("name")], not_read: &[] } },
    // The V typed (`block-ownership-spec.md` §5.1, §5.2, §11): `variable 'Sphere'` and
    // `inline 'Sphere'`. The name is the row acted on, so it is read as the verb's own, which is
    // also what lets the quoted word be it. Nothing else: where a row goes is the rule's to say,
    // not the line's, exactly as the V has no second question to ask.
    Verb { op: "block.make_variable", keys: Keys::Declared { reads: &[switch("name")], not_read: &[] } },
    Verb { op: "block.put_back_inside", keys: Keys::Declared { reads: &[switch("name")], not_read: &[] } },
    // `rename 'Sphere_0' to=ball`: the row by what it is called, as `variable` takes it, and what
    // it is to be called. The echo's spelling of a name typed into a row (`interaction-spec.md` §6).
    Verb { op: "block.rename", keys: Keys::Declared { reads: &[switch("name"), switch("to")], not_read: &[] } },
    Verb {
        op: "view.snap",
        keys: Keys::Declared { reads: &[switch("file"), switch("view"), switch("zoom"), switch("size")], not_read: &[] },
    },
    Verb {
        op: "param.set",
        keys: Keys::Declared { reads: &[switch("name"), switch("value"), switch("expr")], not_read: &[] },
    },
    Verb {
        op: "create.number",
        keys: Keys::Declared { reads: &[switch("name"), switch("expr")], not_read: &[] },
    },
    Verb {
        op: "check.expect",
        keys: Keys::Declared {
            reads: &[
                switch("target"),
                switch("diff"),
                switch("ratio"),
                switch("volume"),
                switch("area"),
                switch("bbox"),
                switch("tol"),
                switch("approx"),
                switch("thickness_min"),
                switch("equal"),
            ],
            not_read: &[],
        },
    },
    // Read-only: a lattice's estimate, for the selected body or `node=`, and with `stiffness=` a
    // percentage, what each source would need (item 94). Makes no block, so it covers no field.
    Verb { op: "view.estimate", keys: Keys::Declared { reads: &[switch("node"), switch("stiffness")], not_read: &[] } },
    // The blocks built on curves (A6). A curve is named, not typed, so `profile=` and `curve=`
    // name one; the keys an arm read before this table existed (`depth`, `two`, `draft2`, `wall`
    // as a thickness) are kept as aliases of the field names beside them.
    Verb {
        op: "create.extrude",
        keys: Keys::Declared {
            reads: &[
                key("profile", &["profile"]),
                key("curve", &["wall"]),
                key("wall", &["thickness"]),
                key("thickness", &["thickness"]),
                key("side", &["side"]),
                key("ends", &["ends"]),
                key("height", &["height"]),
                key("depth", &["height"]),
                key("start", &["start"]),
                key("draft", &["draft"]),
                key("symmetric", &["symmetric"]),
                key("height_2", &["height_2"]),
                key("two", &["height_2"]),
                key("draft_2", &["draft_2"]),
                key("draft2", &["draft_2"]),
                key("corners", &["corners"]),
                key("rims", &["rims"]),
                key("rims_continuity", &["rims_continuity"]),
                key("rims_at", &["rims_at"]),
            ],
            not_read: &[(
                "sweep",
                "which way an extrude travels -- the profile's normal, along a line, toward a point -- is chosen in the panel",
            )],
        },
    },
    Verb {
        op: "create.revolve",
        keys: Keys::Declared {
            reads: &[
                key("profile", &["profile"]),
                key("curve", &["wall"]),
                key("wall", &["thickness"]),
                key("thickness", &["thickness"]),
                key("side", &["side"]),
                key("ends", &["ends"]),
                key("axis", &["axis", "curve"]),
                key("origin", &["origin"]),
                key("angle", &["angle"]),
                key("symmetric", &["symmetric"]),
                key("angle_2", &["angle_2"]),
                key("two", &["angle_2"]),
                key("rims", &["rims"]),
                key("rims_continuity", &["rims_continuity"]),
                key("rims_at", &["rims_at"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.sweep",
        keys: Keys::Declared {
            reads: &[
                key("profile", &["profile"]),
                key("curve", &["wall"]),
                key("wall", &["thickness"]),
                key("thickness", &["thickness"]),
                key("side", &["side"]),
                key("ends", &["ends"]),
                key("spine", &["spine"]),
                key("section", &["section"]),
                // The section law (`sweep-v2-spec.md` §6), by its own field names.
                key("taper_scale", &["taper_scale"]),
                key("taper-scale", &["taper_scale"]),
                key("taper_draft", &["taper_draft"]),
                key("twist", &["twist"]),
                key("law", &["law"]),
                key("extent", &["extent"]),
                key("corners", &["corners"]),
                key("rims", &["rims"]),
                key("rims_continuity", &["rims_continuity"]),
                key("rims_at", &["rims_at"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.loft",
        keys: Keys::Declared {
            reads: &[
                key("sections", &["sections"]),
                key("spine", &["spine"]),
                key("smooth", &["smooth"]),
                key("g1", &["smooth"]),
                key("rims", &["rims"]),
                key("rims_continuity", &["rims_continuity"]),
                key("rims_at", &["rims_at"]),
            ],
            not_read: &[],
        },
    },
    Verb {
        op: "create.pipe",
        keys: Keys::Declared { reads: &[key("curve", &["curve"]), key("radius", &["radius"])], not_read: &[] },
    },
    // Curves are not blocks, so their keys cover no block field: `line p0= p1=`, `curve p=...;...
    // w=...` (and `curve3`, whose degree is in the word).
    Verb {
        op: "create.line",
        keys: Keys::Declared { reads: &[switch("p0"), switch("p1"), switch("p")], not_read: &[] },
    },
    Verb {
        op: "create.curve",
        keys: Keys::Declared { reads: &[switch("degree"), switch("p"), switch("w")], not_read: &[] },
    },
];

/// How a block kind is reached from the console.
///
/// Read only by `every_field_a_block_has_is_a_key_its_verb_reads`; its work is done by
/// compiling, since [`reach_of`] is exhaustive.
#[cfg_attr(not(test), allow(dead_code))]
pub enum Reach {
    /// The operations that make or edit it. More than one where the block is several verbs'
    /// business, as a placement is `move`, `turn` and `scale`.
    Ops(&'static [&'static str]),
    /// Not made from the console, and why. The reason is for whoever reads the match; nothing
    /// prints it.
    NotFromTheConsole(#[allow(dead_code)] &'static str),
}

/// Which verbs reach a block of this kind.
///
/// **Exhaustive on purpose.** A kind added to `Node` fails to compile here until its author says
/// which verb makes it, or why none does -- which is the rule "every block is reachable from
/// the console" written where the compiler reads it.
#[cfg_attr(not(test), allow(dead_code))]
pub fn reach_of(node: &Node) -> Reach {
    match node {
        Node::Cage(_) => Reach::Ops(&["create.cage"]),
        Node::Mesh(_) => Reach::NotFromTheConsole("an imported mesh arrives through `import`, and carries no numbers"),
        Node::Sphere { .. } => Reach::Ops(&["create.sphere"]),
        Node::Cuboid { .. } => Reach::Ops(&["create.cube"]),
        Node::SkewedBox { .. } => Reach::Ops(&["create.skewbox"]),
        Node::Cylinder { .. } => Reach::Ops(&["create.cylinder"]),
        Node::Cone { .. } => Reach::Ops(&["create.cone"]),
        Node::Lattice { .. } => Reach::Ops(&["create.tpms"]),
        Node::Extrude { .. } => Reach::Ops(&["create.extrude"]),
        Node::Revolve { .. } => Reach::Ops(&["create.revolve"]),
        Node::Plane { .. } => Reach::Ops(&["create.plane"]),
        Node::Point { .. } => Reach::Ops(&["create.point"]),
        Node::Torus { .. } => Reach::Ops(&["create.torus"]),
        Node::Capsule { .. } => Reach::Ops(&["create.capsule"]),
        Node::Pipe { .. } => Reach::Ops(&["create.pipe"]),
        Node::Sweep { .. } => Reach::Ops(&["create.sweep"]),
        Node::Loft { .. } => Reach::Ops(&["create.loft"]),
        Node::Blend { .. } => Reach::Ops(&["create.boolean"]),
        Node::Boolean { .. } => Reach::Ops(&["create.boolean"]),
        Node::Morph { .. } => Reach::Ops(&["modify.morph"]),
        Node::Twist { .. } => Reach::Ops(&["modify.twist"]),
        Node::Attract { .. } => Reach::Ops(&["modify.attract"]),
        Node::Pattern { .. } => Reach::Ops(&["modify.pattern"]),
        Node::Mirror { .. } => Reach::Ops(&["modify.mirror"]),
        Node::Shell { .. } => Reach::Ops(&["modify.shell"]),
        Node::Ramp { .. } => Reach::Ops(&["create.ramp"]),
        Node::Offset { .. } => Reach::Ops(&["modify.offset"]),
        Node::Transform { .. } => Reach::Ops(&["modify.move", "modify.turn", "modify.scale"]),
        Node::Unknown(_) => Reach::NotFromTheConsole("a block this build cannot read is kept as it was written, never made"),
    }
}

fn verb(op: &str) -> Option<&'static Verb> {
    VERBS.iter().find(|v| v.op == op)
}

/// What a verb is for, where its word does not say: `help <verb>` leads with it.
///
/// A table rather than a field on [`Verb`], because a create verb's word is the block it makes
/// and needs no sentence; one that reads and reports is the kind that does.
const ABOUT: &[(&str, &str)] = &[
    ("view.estimate", "a lattice's stiffness and strength, estimated from published fits, and never an edit"),
    (
        "block.make_variable",
        "the dim V: what is called that takes a row of its own, and a name if it has none, so any \
         block can refer to it",
    ),
    (
        "block.put_back_inside",
        "the bright V: the variable called that goes back inside the one block that reads it, keeping \
         its name, or a parameter's number back into the one field holding it",
    ),
];

/// Whether `op` reads a `name` key of its own, so `name=` is its argument rather than a name for
/// whatever it made. Five verbs do; see their rows.
pub fn reads_its_own_name(op: &str) -> bool {
    match verb(op).map(|v| &v.keys) {
        Some(Keys::Declared { reads, .. }) => reads.iter().any(|k| k.word == "name"),
        _ => false,
    }
}

/// What `help <verb>` says: the keys `word` reads, each with the block's field where the two
/// differ, and the fields it deliberately leaves to the panel. `None` for a verb the table does
/// not know at all.
pub fn help_for(word: &str, op: &str) -> Option<String> {
    let v = verb(op)?;
    let about = ABOUT.iter().find(|(o, _)| *o == op).map(|(_, about)| *about);
    Some(match (&v.keys, about) {
        (Keys::Declared { reads, .. }, Some(about)) => {
            let keys: Vec<&str> = reads.iter().map(|k| k.word).collect();
            format!("`{word}` — {about}; it reads {}", keys.join(", "))
        }
        (Keys::Declared { reads, not_read }, None) => {
            let keys: Vec<String> = reads
                .iter()
                .map(|k| match k.covers {
                    [field] if *field != k.word => format!("{} ({field})", k.word),
                    _ => k.word.to_string(),
                })
                .collect();
            let mut line = if keys.is_empty() {
                format!("`{word}` takes no keys")
            } else {
                format!("`{word}` reads {}", keys.join(", "))
            };
            if !not_read.is_empty() {
                line.push_str(&format!(
                    " — set in the panel, not here: {}",
                    not_read.iter().map(|(f, _)| *f).collect::<Vec<_>>().join(", ")
                ));
            }
            line
        }
        (Keys::Undeclared(why), _) => {
            format!("`{word}` reads its own keys and has not declared them yet ({why})")
        }
    })
}

/// Why `args` may not go to `op`, if they may not: a key the verb does not read.
///
/// `word` is what was typed, for the message. Only a verb that has declared its keys refuses;
/// a menu passes no keys and is never refused.
pub fn refusal(word: &str, op: &str, args: &Args) -> Option<String> {
    let Some(Verb { keys: Keys::Declared { reads, .. }, .. }) = verb(op) else { return None };
    // Exact, because every reader is exact: `console_arguments` lowers every key as it is
    // parsed, so a key that reaches here in another case was not parsed there and is not read.
    let mut unknown: Vec<&str> = args
        .iter()
        .map(|(k, _)| k)
        .filter(|k| !reads.iter().any(|r| r.word == *k))
        .collect();
    if unknown.is_empty() {
        return None;
    }
    unknown.sort_unstable();
    let known: Vec<&str> = reads.iter().map(|k| k.word).collect();
    Some(if known.is_empty() {
        format!("`{word}` doesn't read `{}` — it takes no keys", unknown.join("`, `"))
    } else {
        format!("`{word}` doesn't read `{}` — it reads {}", unknown.join("`, `"), known.join(", "))
    })
}

// ------------------------------------------------------------------------------------ values

/// One value as a `ParamRef`: a number, a parameter's name, or -- where the slot takes one -- a
/// block's name (A9).
pub fn param_ref(
    verb: &str,
    word: &str,
    text: &str,
    slot: Slot,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<ParamRef, String> {
    let text = text.trim();
    if let Ok(n) = text.parse::<f64>() {
        if !n.is_finite() {
            return Err(format!("`{verb} {word}=` wants a finite number, not {text:?}"));
        }
        return Ok(ParamRef::Literal(n));
    }
    if text.is_empty() {
        return Err(format!("`{verb} {word}=` is missing its value"));
    }
    match lookup(text) {
        Lookup::Parameter(id) => Ok(ParamRef::Named(id)),
        Lookup::Block(id) => match slot {
            Slot::NumberOrField => Ok(ParamRef::Field(id)),
            Slot::Number => Err(format!(
                "`{verb} {word}=` takes a number or a parameter, and {text:?} is a block — only an \
                 offset's distance, a morph's amount, a pipe's radius, a blend's radius and a \
                 lattice phase's own number can be driven by a field"
            )),
        },
        Lookup::Both => Err(format!(
            "{text:?} is both a parameter and a block — rename one, so `{verb} {word}=` knows which"
        )),
        Lookup::Nothing if text.contains(['+', '-', '*', '/', '(', ')']) => Err(format!(
            "`{verb} {word}=` takes no expression — `param name=h expr=\"{text}\"`, then `{word}=h`"
        )),
        Lookup::Nothing => Err(format!(
            "nothing is called {text:?} — `{verb} {word}=` takes a number or a parameter's name"
        )),
    }
}

/// Three values separated by commas, each as [`param_ref`] reads one.
pub fn param_triple(
    verb: &str,
    word: &str,
    value: &Value,
    slot: Slot,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<[ParamRef; 3], String> {
    let text = value.text();
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("`{verb} {word}=` wants three numbers, x,y,z — not {text:?}"));
    }
    Ok([
        param_ref(verb, word, parts[0], slot, lookup)?,
        param_ref(verb, word, parts[1], slot, lookup)?,
        param_ref(verb, word, parts[2], slot, lookup)?,
    ])
}

/// One value, as [`param_ref`] reads it.
pub fn param_one(
    verb: &str,
    word: &str,
    value: &Value,
    slot: Slot,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<ParamRef, String> {
    param_ref(verb, word, &value.text(), slot, lookup)
}

/// A direction: one of the world's axes by letter, or three numbers.
pub fn axis_triple(
    verb: &str,
    word: &str,
    value: &Value,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<[ParamRef; 3], String> {
    let text = value.text();
    if let Some(axis) = Axis::ALL.into_iter().find(|a| a.label().eq_ignore_ascii_case(text.trim())) {
        let Frame::At { axis, .. } = Frame::along(axis) else { unreachable!("`along` builds an explicit frame") };
        return Ok(axis);
    }
    param_triple(verb, word, value, Slot::Number, lookup)
        .map_err(|_| format!("`{verb} {word}=` wants x, y, z or three numbers — not {text:?}"))
}

/// A flag: given at all, and not `=false`.
pub fn flag(args: &Args, word: &str) -> bool {
    args.get(word).is_some_and(|v| !matches!(v, Value::Bool(false)))
}

/// A whole number, refused rather than rounded.
pub fn whole(verb: &str, word: &str, value: &Value) -> Result<u32, String> {
    match value {
        Value::Int(n) if *n >= 0 => u32::try_from(*n).map_err(|_| format!("`{verb} {word}=` is too large: {n}")),
        Value::Float(f) if f.fract() == 0.0 && *f >= 0.0 && *f < 4.0e9 => Ok(*f as u32),
        other => Err(format!("`{verb} {word}=` wants a whole number, not {}", other.text())),
    }
}

/// A profile solid's rims from `rims=`, `rims_continuity=` and `rims_at=`, the field names: the
/// reach, nought for hard; the shape by the words a blend's `g=` takes; and which ends, `both`,
/// `start` or `end`. `None` when none is said, so the block keeps the rims it has; a shape or an
/// end with no reach and nothing to give it one is refused, since a hard rim has neither to set.
pub(crate) fn rims_asked(
    verb: &str,
    args: &Args,
    current: fieldform_scene::Rims,
) -> Result<Option<fieldform_scene::Rims>, String> {
    let (reach, shape, at) = (args.get("rims"), args.get("rims_continuity"), args.get("rims_at"));
    if reach.is_none() && shape.is_none() && at.is_none() {
        return Ok(None);
    }
    let mut rims = current;
    if let Some(v) = reach {
        let mm = v.as_f64().filter(|mm| *mm >= 0.0).ok_or_else(|| {
            format!("`{verb} rims=` is how far back the rims are cut, in millimetres, not {}", v.text())
        })?;
        rims.radius = ParamRef::Literal(mm);
    }
    if let Some(v) = shape {
        rims.continuity = continuity(&format!("{verb} rims_continuity"), v)?;
        if rims.is_plainly_hard() {
            return Err(format!(
                "`{verb} rims_continuity=` shapes a rim, and these rims are hard -- say how far too, rims=2"
            ));
        }
    }
    if let Some(v) = at {
        rims.at = match v.text().trim().to_ascii_lowercase().as_str() {
            "both" => fieldform_scene::RimEnds::Both,
            "start" => fieldform_scene::RimEnds::Start,
            "end" => fieldform_scene::RimEnds::End,
            other => {
                return Err(format!("`{verb} rims_at=` is both, start or end, not {other:?}"));
            }
        };
        if rims.is_plainly_hard() {
            return Err(format!(
                "`{verb} rims_at=` says which ends a rim rounds, and these rims are hard -- say how far too, rims=2"
            ));
        }
    }
    Ok(Some(rims))
}

/// A box's or a cylinder's edge from `edge=` and `edge_continuity=`, the field names: the reach
/// -- a number or a parameter's name, read by the caller as every number here is -- and the shape
/// by the words a blend's `g=` takes. A shape with no reach is refused, since a sharp edge has no
/// shape to set.
fn edge_with_keys(
    word: &str,
    args: &Args,
    edge: &mut fieldform_scene::Fillet,
    reach: Option<ParamRef>,
) -> Result<(), String> {
    if let Some(r) = reach {
        edge.radius = r;
    }
    if let Some(v) = args.get("edge_continuity") {
        edge.continuity = continuity(&format!("{word} edge_continuity"), v)?;
        if edge.is_plainly_hard() {
            return Err(format!(
                "`{word} edge_continuity=` shapes an edge, and these edges are sharp -- say how far too, edge=2"
            ));
        }
    }
    Ok(())
}

/// A fillet's shape from `g=1`, `g=2` or `g=chamfer`.
///
/// The key stays `g` for the chamfer although a chamfer is no continuity: it is the field the
/// value is written to (`continuity`, the wart recorded in `Continuity::Chamfer`), and a second
/// key for one field is the alias rule's to refuse.
fn continuity(verb: &str, value: &Value) -> Result<Continuity, String> {
    match value.text().trim().to_ascii_lowercase().as_str() {
        "1" | "g1" => Ok(Continuity::G1),
        "2" | "g2" => Ok(Continuity::G2),
        "c" | "chamfer" => Ok(Continuity::Chamfer),
        other => Err(format!("`{verb} g=` is 1, 2 or chamfer, not {other:?}")),
    }
}

// --------------------------------------------------------------------------------- primitives

/// A primitive as `kind.node` made it, with the keys typed after the verb applied.
///
/// Omitted keys keep the defaults. `size=40` is `size=40,40,40`, and `from`/`to` move the ends
/// whole so a cylinder typed as two points is the cylinder between them.
pub fn primitive_with_keys(
    mut node: Node,
    word: &str,
    args: &Args,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<Node, String> {
    let one = |k: &str| -> Result<Option<ParamRef>, String> {
        args.get(k).map(|v| param_one(word, k, v, Slot::Number, lookup)).transpose()
    };
    let three = |k: &str| -> Result<Option<[ParamRef; 3]>, String> {
        args.get(k).map(|v| param_triple(word, k, v, Slot::Number, lookup)).transpose()
    };
    let place = |k: &str| -> Result<Option<Place>, String> { Ok(three(k)?.map(Place::At)) };
    let turn = |k: &str| -> Result<Option<Turn>, String> { Ok(three(k)?.map(Turn::By)) };
    let ends = |ends: &mut Ends| -> Result<(), String> {
        let (from, to) = (place("from")?, place("to")?);
        if from.is_none() && to.is_none() {
            return Ok(());
        }
        let (was_from, was_to) = match ends {
            Ends::At { from, to } => (from.clone(), to.clone()),
            _ => (Place::origin(), Place::at(0.0, 0.0, 1.0)),
        };
        *ends = Ends::At { from: from.unwrap_or(was_from), to: to.unwrap_or(was_to) };
        Ok(())
    };
    match &mut node {
        Node::Sphere { centre, radius } => {
            if let Some(c) = place("centre")? {
                *centre = c;
            }
            if let Some(r) = one("radius")? {
                *radius = r;
            }
        }
        Node::Cuboid { centre, size, turn: t, edge } => {
            edge_with_keys(word, args, edge, one("edge")?)?;
            if let Some(c) = place("centre")? {
                *centre = c;
            }
            if let Some(v) = args.get("size") {
                *size = match v.text().contains(',') {
                    true => param_triple(word, "size", v, Slot::Number, lookup)?,
                    false => {
                        let s = param_one(word, "size", v, Slot::Number, lookup)?;
                        [s, s, s]
                    }
                };
            }
            if let Some(by) = turn("turn")? {
                *t = by;
            }
        }
        Node::SkewedBox { centre, edges } => {
            if let Some(c) = place("centre")? {
                *centre = c;
            }
            if let Some(v) = args.get("edges") {
                let text = v.text();
                let rows: Vec<&str> = text.split(';').collect();
                if rows.len() != 3 {
                    return Err(format!(
                        "`{word} edges=` wants three edges of three numbers, x,y,z;x,y,z;x,y,z — not {text:?}"
                    ));
                }
                for (edge, row) in edges.iter_mut().zip(rows) {
                    *edge = param_triple(word, "edges", &Value::Str(row.to_string()), Slot::Number, lookup)?;
                }
            }
        }
        Node::Cylinder { ends: e, radius, edge } => {
            edge_with_keys(word, args, edge, one("edge")?)?;
            ends(e)?;
            if let Some(r) = one("radius")? {
                *radius = r;
            }
        }
        Node::Cone { ends: e, base_radius, top_radius, edge } => {
            edge_with_keys(word, args, edge, one("edge")?)?;
            ends(e)?;
            if let Some(r) = one("base_radius")? {
                *base_radius = r;
            }
            if let Some(r) = one("top_radius")? {
                *top_radius = r;
            }
        }
        Node::Torus { centre, major_radius, minor_radius, turn: t } => {
            if let Some(c) = place("centre")? {
                *centre = c;
            }
            if let Some(r) = one("major_radius")? {
                *major_radius = r;
            }
            if let Some(r) = one("minor_radius")? {
                *minor_radius = r;
            }
            if let Some(by) = turn("turn")? {
                *t = by;
            }
        }
        Node::Capsule { ends: e, start_radius, end_radius } => {
            ends(e)?;
            if let Some(r) = one("start_radius")? {
                *start_radius = r;
            }
            if let Some(r) = one("end_radius")? {
                *end_radius = r;
            }
        }
        Node::Plane { origin, flipped, turn: t } => {
            if let Some(o) = place("origin")? {
                *origin = o;
            }
            if let Some(by) = turn("turn")? {
                *t = by;
            }
            if args.get("flipped").is_some() {
                *flipped = flag(args, "flipped");
            }
        }
        Node::Point { at } => {
            if let Some(p) = three("at")? {
                *at = p;
            }
        }
        other => {
            return Err(format!("`{word}` does not make a {}", kind_word(other)));
        }
    }
    Ok(node)
}

fn kind_word(node: &Node) -> &'static str {
    match node {
        Node::Lattice { .. } => "lattice",
        Node::Ramp { .. } => "ramp",
        Node::Boolean { .. } | Node::Blend { .. } => "boolean",
        _ => "block of that kind",
    }
}

// ------------------------------------------------------------------------------------ lattice

/// A lattice as `add_lattice_with` seeds it, with the keys typed after `tpms` applied.
///
/// `phase=sheet|solid|strut` picks the phase, and the phase's own number is read by the
/// phase's own name -- `thickness` (and `offset`) for a sheet, `density` for a solid, `radius`
/// for a strut. A phase the cell does not offer is refused by name. The phase's number is the
/// one slot here a field may drive (`driven_refusal`).
pub fn lattice_with_keys(
    mut node: Node,
    args: &Args,
    offered: &dyn Fn(&LatticeCell) -> &'static [fieldform_scene::Banding],
    seed_for: &dyn Fn(fieldform_scene::Banding) -> LatticePhase,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<Node, String> {
    use fieldform_scene::Banding;
    let word = "tpms";
    let Node::Lattice { cell, frame, map, phase } = &mut node else {
        return Err("`tpms` makes a lattice".to_string());
    };
    let positive = |k: &str| -> Result<Option<ParamRef>, String> {
        args.get(k)
            .map(|v| {
                let r = param_one(word, k, v, Slot::Number, lookup)?;
                if let ParamRef::Literal(n) = r
                    && n <= 0.0
                {
                    return Err(format!("`{word} {k}=` must be above nought, not {n}"));
                }
                Ok(r)
            })
            .transpose()
    };
    // The cell, by the name the file writes (`gyroid`, `schwarz_p`) or the label with its spaces
    // closed up (`schwarzp`, `fischer-kochs`), which is what the console has always taken.
    if let Some(v) = args.get("cell") {
        let name = v.text().trim().to_ascii_lowercase();
        let jitter = args.get("jitter").map(|v| param_one(word, "jitter", v, Slot::Number, lookup)).transpose()?;
        let seed = args.get("seed").map(|v| whole(word, "seed", v)).transpose()?;
        let found = LatticeCell::from_name(
            &name,
            jitter.clone().unwrap_or(ParamRef::Literal(0.75)),
            u64::from(seed.unwrap_or(0)),
        )
        .or_else(|| {
            LatticeCell::from_name(
                &name.replace(['-', ' '], ""),
                jitter.clone().unwrap_or(ParamRef::Literal(0.75)),
                u64::from(seed.unwrap_or(0)),
            )
        })
        .or_else(|| {
            // The labels with their spaces and hyphens closed, which is how `tpms cell=` spelt
            // them before the file's names were accepted.
            let squeezed: Vec<(String, LatticeCell)> = [
                "gyroid", "schwarz_p", "schwarz_d", "schoen_iwp", "neovius", "fischer_koch_s", "cubic", "bcc",
                "fcc", "octet", "diamond", "honeycomb", "voronoi_foam", "voronoi_beam",
            ]
            .iter()
            .filter_map(|n| {
                LatticeCell::from_name(n, jitter.clone().unwrap_or(ParamRef::Literal(0.75)), u64::from(seed.unwrap_or(0)))
                    .map(|c| (n.replace('_', ""), c))
            })
            .collect();
            squeezed.into_iter().find(|(n, _)| *n == name.replace(['-', ' ', '_'], "")).map(|(_, c)| c)
        });
        match found {
            Some(c) if (jitter.is_some() || seed.is_some()) && !c.is_stochastic() => {
                return Err(format!(
                    "`tpms jitter=` and `seed=` belong to a Voronoi cell, and {} has no seed grid — say `cell=voronoi_foam` or `cell=voronoi_beam`",
                    c.name()
                ))
            }
            Some(c) => *cell = c,
            None => {
                return Err(format!(
                    "no cell is called {name:?} — the cells are gyroid, schwarz_p, schwarz_d, schoen_iwp, neovius, \
                     fischer_koch_s, cubic, bcc, fcc, octet, diamond, honeycomb, voronoi_foam, voronoi_beam"
                ))
            }
        }
    } else if args.get("jitter").is_some() || args.get("seed").is_some() {
        return Err("`tpms jitter=` and `seed=` belong to a Voronoi cell — say `cell=voronoi_foam` or `cell=voronoi_beam`".to_string());
    }
    // The map, and its own numbers by their own names.
    let map_word = args.get("map").map(|v| v.text().trim().to_ascii_lowercase());
    let default_size = match map {
        CellMap::Rectangular { cell } => cell.clone(),
        _ => ParamRef::Literal(20.0),
    };
    match map_word.as_deref() {
        None | Some("rectangular") => {
            if map_word.is_some() || args.get("cell_size").is_some() {
                *map = CellMap::Rectangular { cell: positive("cell_size")?.unwrap_or(default_size) };
            }
            for k in ["cell_radius", "cell_height", "arc", "arc_count", "azimuth", "polar"] {
                if args.get(k).is_some() {
                    return Err(format!("`tpms {k}=` belongs to a cylindrical or spherical map — say `map=cylindrical` or `map=spherical`"));
                }
            }
        }
        Some("cylindrical") => {
            for k in ["cell_size", "azimuth", "polar"] {
                if args.get(k).is_some() {
                    let owner = if k == "cell_size" { "rectangular" } else { "spherical" };
                    return Err(format!("`tpms {k}=` belongs to a {owner} map, not a cylindrical one — say `map={owner}`"));
                }
            }
            let cell_radius = positive("cell_radius")?.unwrap_or(default_size.clone());
            let cell_height = positive("cell_height")?.unwrap_or(default_size.clone());
            let arc = args.get("arc").or(args.get("arc_count"));
            let arc_count = match arc {
                Some(v) => {
                    let n = whole(word, "arc", v)?;
                    if n == 0 {
                        return Err("`tpms arc=` must be one or more cells round".to_string());
                    }
                    n
                }
                None => match cell_radius {
                    ParamRef::Literal(r) => CellMap::arc_count_for(r, 50.0),
                    _ => 24,
                },
            };
            *map = CellMap::Cylindrical { cell_radius, cell_height, arc_count };
        }
        Some("spherical") => {
            for k in ["cell_size", "cell_height", "arc", "arc_count"] {
                if args.get(k).is_some() {
                    let owner = if k == "cell_size" { "rectangular" } else { "cylindrical" };
                    return Err(format!("`tpms {k}=` belongs to a {owner} map, not a spherical one — say `map={owner}`"));
                }
            }
            let cell_radius = positive("cell_radius")?.unwrap_or(default_size.clone());
            let count = |k: &str, fallback: u32| -> Result<u32, String> {
                match args.get(k) {
                    Some(v) => {
                        let n = whole(word, k, v)?;
                        if n == 0 {
                            return Err(format!("`tpms {k}=` must be one or more cells"));
                        }
                        Ok(n)
                    }
                    None => Ok(fallback),
                }
            };
            *map = CellMap::Spherical {
                cell_radius,
                azimuth_count: count("azimuth", 24)?,
                polar_count: count("polar", 12)?,
            };
        }
        Some(other) => {
            return Err(format!(
                "no map is called {other:?} — rectangular, cylindrical and spherical are typed; a conformal map is picked in the panel"
            ))
        }
    }
    // What this cell offers; a cell chosen above that does not offer the seeded phase is
    // reseeded with what it does, as the panel would on its next frame.
    let offered = offered(cell);
    let banding_of = |phase: &LatticePhase| match phase {
        LatticePhase::Sheet { .. } => Banding::Sheet,
        LatticePhase::Solid { .. } => Banding::Solid,
        LatticePhase::Strut { .. } => Banding::Strut,
    };
    if !offered.contains(&banding_of(phase))
        && let Some(first) = offered.first()
    {
        *phase = seed_for(*first);
    }
    // The phase: named, or implied by the number given.
    let banding_named = match args.get("phase").map(|v| v.text().trim().to_ascii_lowercase()).as_deref() {
        None => None,
        Some("sheet") => Some(Banding::Sheet),
        Some("solid") => Some(Banding::Solid),
        Some("strut") => Some(Banding::Strut),
        Some(other) => return Err(format!("`tpms phase=` is sheet, solid or strut, not {other:?}")),
    };
    // Each phase's own numbers, and a line that gives two phases' numbers is refused rather than
    // read for one and not the other.
    let given: Vec<Banding> = [
        (Banding::Solid, args.get("density").is_some()),
        (Banding::Sheet, args.get("thickness").is_some() || args.get("offset").is_some()),
        (Banding::Strut, args.get("radius").is_some()),
    ]
    .into_iter()
    .filter(|(_, yes)| *yes)
    .map(|(b, _)| b)
    .collect();
    if given.len() > 1 {
        return Err(format!(
            "`tpms` reads one phase's numbers at a time — a sheet has a thickness and an offset, a solid a density, a strut a radius; this line gives a {} and a {}",
            banding_word(given[0]),
            banding_word(given[1])
        ));
    }
    let banding_implied = given.first().copied();
    if let (Some(named), Some(implied)) = (banding_named, banding_implied)
        && named != implied
    {
        return Err(format!(
            "`tpms phase={}` does not take `{}` — a sheet has a thickness and an offset, a solid a density, a strut a radius",
            banding_word(named),
            banding_word(implied)
        ));
    }
    if let Some(wanted) = banding_named.or(banding_implied) {
        if !offered.contains(&wanted) {
            return Err(format!(
                "a {} cell has no {} phase — it offers {}",
                cell.name(),
                banding_word(wanted),
                offered.iter().map(|b| banding_word(*b)).collect::<Vec<_>>().join(" and ")
            ));
        }
        if banding_of(phase) != wanted {
            *phase = seed_for(wanted);
        }
    }
    let driven = |k: &str| -> Result<Option<ParamRef>, String> {
        args.get(k).map(|v| param_one(word, k, v, Slot::NumberOrField, lookup)).transpose()
    };
    match phase {
        LatticePhase::Sheet { thickness, offset } => {
            if let Some(t) = driven("thickness")? {
                *thickness = t;
            }
            if let Some(o) = args.get("offset").map(|v| param_one(word, "offset", v, Slot::Number, lookup)).transpose()? {
                *offset = o;
            }
        }
        LatticePhase::Solid { density } => {
            if let Some(d) = driven("density")? {
                *density = d;
            }
        }
        LatticePhase::Strut { radius } => {
            if let Some(r) = driven("radius")? {
                *radius = r;
            }
        }
    }
    // Where it sits and which way it runs: the lattice's frame.
    if args.get("origin").is_some() || args.get("axis").is_some() {
        let (was_origin, was_axis) = match frame {
            Frame::At { origin, axis } => (origin.clone(), axis.clone()),
            _ => ([ParamRef::Literal(0.0); 3], [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(1.0)]),
        };
        let origin = args
            .get("origin")
            .map(|v| param_triple(word, "origin", v, Slot::Number, lookup))
            .transpose()?
            .unwrap_or(was_origin);
        let axis = args.get("axis").map(|v| axis_triple(word, "axis", v, lookup)).transpose()?.unwrap_or(was_axis);
        *frame = Frame::At { origin, axis };
    }
    Ok(node)
}

fn banding_word(b: fieldform_scene::Banding) -> &'static str {
    match b {
        fieldform_scene::Banding::Sheet => "sheet",
        fieldform_scene::Banding::Solid => "solid",
        fieldform_scene::Banding::Strut => "strut",
    }
}

// --------------------------------------------------------------------------------------- ramp

/// A ramp as `add_ramp_with` seeds it, with `in=lo,hi`, `out=lo,hi`, `shape=` and `input=`
/// applied.
///
/// A range is a pair because it is one idea; a single number for one is refused rather than
/// half-read. The shape is taken by its label (`G0`, `G1`, `G2`) or by the file's word
/// (`linear`, `smooth`, `smoother`).
pub fn ramp_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let word = "ramp";
    let Node::Ramp { input, in_min, in_max, out_min, out_max, shape } = &mut node else {
        return Err("`ramp` makes a ramp".to_string());
    };
    let pair = |k: &str| -> Result<Option<(ParamRef, ParamRef)>, String> {
        let Some(v) = args.get(k) else { return Ok(None) };
        let text = v.text();
        let Some((lo, hi)) = text.split_once(',') else {
            return Err(format!("`{word} {k}=` is a range, lo,hi — not {text:?}"));
        };
        if hi.contains(',') {
            return Err(format!("`{word} {k}=` is a range of two numbers, lo,hi — not {text:?}"));
        }
        Ok(Some((param_ref(word, k, lo, Slot::Number, lookup)?, param_ref(word, k, hi, Slot::Number, lookup)?)))
    };
    if let Some((lo, hi)) = pair("in")? {
        *in_min = lo;
        *in_max = hi;
    }
    if let Some((lo, hi)) = pair("out")? {
        *out_min = lo;
        *out_max = hi;
    }
    if let Some(v) = args.get("shape") {
        let name = v.text();
        let found = RampShape::ALL.into_iter().find(|s| {
            s.label().eq_ignore_ascii_case(name.trim())
                || match s {
                    RampShape::Linear => name.trim().eq_ignore_ascii_case("linear"),
                    RampShape::Smooth => name.trim().eq_ignore_ascii_case("smooth"),
                    RampShape::Smoother => name.trim().eq_ignore_ascii_case("smoother"),
                }
        });
        match found {
            Some(s) => *shape = s,
            None => return Err(format!("`{word} shape=` is G0, G1 or G2 — not {name:?}")),
        }
    }
    if let Some(asked) = operand_named(word, "input", args, lookup)? {
        *input = asked;
    }
    Ok(node)
}

// ------------------------------------------------------------------------------------ operands

/// The blocks named in a comma list, each resolved as a block.
pub fn blocks_named(verb: &str, word: &str, value: &Value, lookup: &dyn Fn(&str) -> Lookup) -> Result<Vec<NodeId>, String> {
    let text = value.text();
    let mut out = Vec::new();
    for name in text.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        out.push(block_called(verb, word, name, lookup)?);
    }
    if out.is_empty() {
        return Err(format!("`{verb} {word}=` names at least one block"));
    }
    Ok(out)
}

/// The block `name` names, or why it names none. A word that is both a parameter and a block is
/// refused here as everywhere (A9), even though only a block would do: the user typed a word
/// that means two things, and the fix is a rename, not a guess.
pub fn block_called(verb: &str, word: &str, name: &str, lookup: &dyn Fn(&str) -> Lookup) -> Result<NodeId, String> {
    match lookup(name) {
        Lookup::Block(id) => Ok(id),
        Lookup::Both => Err(format!("{name:?} is both a parameter and a block — rename one, so `{verb} {word}=` knows which")),
        Lookup::Parameter(_) => Err(format!("`{verb} {word}=` names a block, and {name:?} is a parameter")),
        Lookup::Nothing => Err(format!("no block is called {name:?}")),
    }
}

/// One block named by `word`, or `None` where the key was not given.
pub fn block_named(verb: &str, word: &str, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Option<NodeId>, String> {
    let Some(v) = args.get(word) else { return Ok(None) };
    let found = blocks_named(verb, word, v, lookup)?;
    if found.len() != 1 {
        return Err(format!("`{verb} {word}=` names one block"));
    }
    Ok(Some(found[0]))
}

/// The word that empties a slot: `target=none`, `in=none`, `input=none`.
///
/// **One word wherever a line names an operand**, which is the rule `Operand` itself is a type
/// for (D9): a slot is a slot, so "nothing in it" cannot be `none` on one verb and `empty` on
/// the next. It wins over a block that happens to be called *none* -- the way to name that one
/// is to rename it, as a word that is both a parameter and a block is settled by a rename (A9).
pub const NOTHING: &str = "none";

/// One operand a line names: a block, or the empty slot `none` asks for.
///
/// `Ok(None)` where the key was not given, so the caller falls back to the block it would have
/// guessed; `Ok(Some(Operand::empty()))` where `none` was typed, which is the line saying *do
/// not* guess. An empty slot is a legal, savable state (`notebook-spec.md` §7.2), so a script
/// can make the block before it has anything to put in it and build a part top-down -- which is
/// the one thing a click could do and a line could not.
pub fn operand_named(verb: &str, word: &str, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Option<Operand>, String> {
    let Some(v) = args.get(word) else { return Ok(None) };
    let found = operands_named(verb, word, v, lookup)?;
    if found.len() != 1 {
        return Err(format!("`{verb} {word}=` names one block, or `none` for an empty slot"));
    }
    Ok(Some(found[0]))
}

/// The operands a comma list names, each a block or `none`: [`operand_named`]'s list form, for
/// the boolean's `in=`.
///
/// `in=box,none` is a join whose second slot is waiting, which is how a line says the shape of a
/// part before it says every block in it.
pub fn operands_named(verb: &str, word: &str, value: &Value, lookup: &dyn Fn(&str) -> Lookup) -> Result<Vec<Operand>, String> {
    let text = value.text();
    let mut out = Vec::new();
    for name in text.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        out.push(operand_called(verb, word, name, lookup)?);
    }
    if out.is_empty() {
        return Err(format!("`{verb} {word}=` names at least one block, or `none` for an empty slot"));
    }
    Ok(out)
}

/// One item of an operand list: the empty slot where it is [`NOTHING`], and otherwise whatever
/// [`block_called`] makes of it -- so a misspelt block is still refused by name rather than read
/// as a hole.
fn operand_called(verb: &str, word: &str, name: &str, lookup: &dyn Fn(&str) -> Lookup) -> Result<Operand, String> {
    if name.eq_ignore_ascii_case(NOTHING) {
        return Ok(Operand::empty());
    }
    block_called(verb, word, name, lookup).map(Operand::filled)
}

// ------------------------------------------------------------------------------------- the V

/// A row the V is on, as `variable` and `inline` find it by what it is called: a block, a curve,
/// a profile by its loop's first curve, or a parameter, which is a variable from birth and whose
/// way back is `inline`'s (`block-ownership-spec.md` §11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowCalled {
    Block(NodeId),
    Curve(fieldform_scene::CurveId),
    Profile(fieldform_scene::CurveId),
    Parameter(ParamId),
}

/// The one row `name` names, or why it names none or several.
///
/// `named` is every row carrying `name` as its own name, and `labelled` every row with no name
/// that shows it -- an unnamed sphere's "Sphere", a profile's "Profile 2". **The name is asked
/// first and a label only where nothing carries the name**, because a name is what a row is typed
/// by (`block-ownership-spec.md` §6.1) and a label is only what an unnamed row happens to show.
///
/// **Two of either are refused, never guessed between.** The V this stands for acts on one row,
/// and a line that could mean either has not said which; the refusal says what shares the word, as
/// a value that is both a parameter and a block is refused (A9). `word` is the verb, for the
/// message.
pub fn row_called(word: &str, name: &str, named: &[RowCalled], labelled: &[RowCalled]) -> Result<RowCalled, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(format!("`{word}` wants the name of a row in the list — `{word} 'Sphere'`, or `{word} name=Sphere`"));
    }
    match (named, labelled) {
        ([one], _) | ([], [one]) => Ok(*one),
        ([], []) => Err(format!(
            "nothing is called {name:?} — `{word}` takes what a row in the list is called: a block's, a curve's, \
             a profile's or a parameter's name, or the label an unnamed block or profile shows"
        )),
        ([], several) => Err(format!(
            "{name:?} is what {} show, and none of them is named it — name the one you mean, so `{word}` knows which",
            kinds_of(several)
        )),
        (several, _) => Err(format!("{name:?} names {} — rename one, so `{word}` knows which", kinds_of(several))),
    }
}

/// What a list of rows are, counted by kind: "a block and a curve", "2 blocks".
fn kinds_of(rows: &[RowCalled]) -> String {
    let counts = [
        ("a block", "blocks", rows.iter().filter(|r| matches!(r, RowCalled::Block(_))).count()),
        ("a curve", "curves", rows.iter().filter(|r| matches!(r, RowCalled::Curve(_))).count()),
        ("a profile", "profiles", rows.iter().filter(|r| matches!(r, RowCalled::Profile(_))).count()),
        ("a parameter", "parameters", rows.iter().filter(|r| matches!(r, RowCalled::Parameter(_))).count()),
    ];
    let said: Vec<String> = counts
        .into_iter()
        .filter(|(_, _, n)| *n > 0)
        .map(|(one, many, n)| if n == 1 { one.to_string() } else { format!("{n} {many}") })
        .collect();
    match said.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => String::new(),
    }
}

/// The line a reply gives for someone to type, naming `row` to `word`: `inline 'r'`.
///
/// **Quoted, always**, because the console has no bare word: `inline r` is refused by the grammar
/// (`console_arguments`) before `inline` sees it, which is what the answer to `variable 'r'` once
/// told people to type. And in whichever mark the name does not hold, since a name is free text
/// and `Bob's` closes a single quote early; a name holding both marks cannot be typed by any line.
pub fn line_naming(word: &str, row: &str) -> String {
    let mark = if row.contains('\'') { '"' } else { '\'' };
    format!("{word} {mark}{row}{mark}")
}

/// What `boolean` was asked for: which operation, which operands, and whether a fillet.
#[derive(Debug, Clone, PartialEq)]
pub struct BooleanAsked {
    pub op: Option<BooleanOp>,
    /// The slots `in=` named, each a block or an empty one -- `None` where it was not given.
    pub inputs: Option<Vec<Operand>>,
    pub fillet: Option<Fillet>,
}

/// `boolean -u in=a,b r=2 g=2` read. A fillet joins exactly two bodies (A3), so `r=` with three
/// names in `in=` is refused rather than chained.
pub fn boolean_asked(args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<BooleanAsked, String> {
    let verb = "boolean";
    let op = [
        (BooleanOp::Union, ["u", "union"]),
        (BooleanOp::Difference, ["s", "subtract"]),
        (BooleanOp::Intersection, ["i", "intersect"]),
    ]
    .into_iter()
    .find(|(_, names)| {
        names.iter().any(|n| flag(args, n))
            || args.get("op").is_some_and(|v| names.iter().any(|n| v.text().eq_ignore_ascii_case(n)))
    })
    .map(|(op, _)| op);
    if let Some(v) = args.get("op")
        && op.is_none()
    {
        return Err(format!("`boolean op=` is union, subtract or intersect — not {:?}", v.text()));
    }
    // **`in=none` is the pair a boolean is born with, both slots waiting**, and not one slot: a
    // boolean of arity one is an arity nobody asked for, where two empty slots are the block a
    // script makes before it has the bodies to join (`notebook-spec.md` §7.2). A longer list
    // takes `none` in any position, so `in=box,none` is half a join.
    let inputs = match args.get("in") {
        None => None,
        Some(v) if v.text().trim().eq_ignore_ascii_case(NOTHING) => {
            Some(vec![Operand::empty(), Operand::empty()])
        }
        Some(v) => Some(operands_named(verb, "in", v, lookup)?),
    };
    if let Some(list) = &inputs
        && list.len() < 2
    {
        return Err("`boolean in=` names two or more blocks, like in=box,ball".to_string());
    }
    // A block's name is taken here: a blend's radius may follow a ramp (`driven_refusal`,
    // `round-chamfer-spec.md` §9.1), and the kernel refuses anything but a ramp by name.
    let radius = args
        .get("r")
        .or(args.get("radius"))
        .map(|v| param_one(verb, "r", v, Slot::NumberOrField, lookup))
        .transpose()?;
    let fillet = match radius {
        Some(ParamRef::Literal(r)) if r <= 0.0 => None,
        Some(radius) => {
            let continuity = args.get("g").map(|v| continuity(verb, v)).transpose()?.unwrap_or(Continuity::G1);
            Some(Fillet { radius, continuity })
        }
        None => {
            if args.get("g").is_some() {
                return Err("`boolean g=` goes with a radius, `r=`".to_string());
            }
            None
        }
    };
    if fillet.is_some()
        && let Some(list) = &inputs
        && list.len() > 2
    {
        return Err("a fillet joins two bodies; union the rest first".to_string());
    }
    Ok(BooleanAsked { op, inputs, fillet })
}

// ----------------------------------------------------------------------------------- modifiers

/// A twist as `apply_twist` made it, with `rate=`/`degrees_per_mm=`, `axis=` and `origin=` applied.
pub fn twist_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let word = "twist";
    let Node::Twist { degrees_per_mm, frame, .. } = &mut node else { return Err("`twist` makes a twist".to_string()) };
    if let Some(v) = args.get("rate").or(args.get("degrees_per_mm")) {
        *degrees_per_mm = param_one(word, "rate", v, Slot::Number, lookup)?;
    }
    frame_with_keys(word, frame, args, lookup)?;
    Ok(node)
}

/// A mirror as `apply_mirror` made it, with `axis=`, `origin=` and `-join` applied.
pub fn mirror_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let word = "mirror";
    let Node::Mirror { frame, join, .. } = &mut node else { return Err("`mirror` makes a mirror".to_string()) };
    frame_with_keys(word, frame, args, lookup)?;
    if args.get("join").is_some() {
        *join = flag(args, "join");
    }
    Ok(node)
}

/// `axis=` and `origin=` onto a frame, keeping whichever was not given.
fn frame_with_keys(word: &str, frame: &mut Frame, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<(), String> {
    if args.get("origin").is_none() && args.get("axis").is_none() {
        return Ok(());
    }
    let (was_origin, was_axis) = match frame {
        Frame::At { origin, axis } => (origin.clone(), axis.clone()),
        _ => ([ParamRef::Literal(0.0); 3], [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(1.0)]),
    };
    let origin = args
        .get("origin")
        .map(|v| param_triple(word, "origin", v, Slot::Number, lookup))
        .transpose()?
        .unwrap_or(was_origin);
    let axis = args.get("axis").map(|v| axis_triple(word, "axis", v, lookup)).transpose()?.unwrap_or(was_axis);
    *frame = Frame::At { origin, axis };
    Ok(())
}

/// A shell with `thickness=` applied. A wall is a thickness, so a negative one is taken as
/// written and made positive rather than refused: the intent is unambiguous.
pub fn shell_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let Node::Shell { thickness, .. } = &mut node else { return Err("`shell` makes a shell".to_string()) };
    if let Some(v) = args.get("thickness") {
        *thickness = match param_one("shell", "thickness", v, Slot::Number, lookup)? {
            ParamRef::Literal(t) => ParamRef::Literal(t.abs()),
            other => other,
        };
    }
    Ok(node)
}

/// An offset with `distance=` applied -- the one modifier whose number a field may drive.
pub fn offset_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let Node::Offset { distance, .. } = &mut node else { return Err("`offset` makes an offset".to_string()) };
    if let Some(v) = args.get("distance") {
        *distance = param_one("offset", "distance", v, Slot::NumberOrField, lookup)?;
    }
    Ok(node)
}

/// A morph with `t=` applied; its operands are read by `a=` and `b=` before it is built.
pub fn morph_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let Node::Morph { t, .. } = &mut node else { return Err("`morph` makes a morph".to_string()) };
    if let Some(v) = args.get("t") {
        *t = param_one("morph", "t", v, Slot::NumberOrField, lookup)?;
    }
    Ok(node)
}

/// An attractor with `strength=` and `falloff=` applied. `toward=` is resolved by the app,
/// since a curve is not a node and only the app holds both lists.
pub fn attract_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let Node::Attract { strength, falloff, .. } = &mut node else { return Err("`attract` makes an attractor".to_string()) };
    if let Some(v) = args.get("strength") {
        *strength = param_one("attract", "strength", v, Slot::Number, lookup)?;
    }
    if let Some(v) = args.get("falloff") {
        *falloff = param_one("attract", "falloff", v, Slot::Number, lookup)?;
    }
    Ok(node)
}

/// A pattern as `apply_pattern` made it, with its keys applied: `count`, `kind`, `step` (one
/// number along the first axis, or three), `axis`/`origin` for a ring, `r=`/`g=` for a filleted
/// join, `join`, and the options `pattern_options_from` already read.
pub fn pattern_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let word = "pattern";
    let Node::Pattern { count, kind, join, fillet, options, .. } = &mut node else {
        return Err("`pattern` makes a pattern".to_string())
    };
    if let Some(v) = args.get("count") {
        *count = param_one(word, "count", v, Slot::Number, lookup)?;
    }
    if let Some(v) = args.get("kind") {
        match v.text().trim().to_ascii_lowercase().as_str() {
            "linear" => {
                if !matches!(kind, PatternKind::Linear { .. }) {
                    *kind = PatternKind::Linear { step: [ParamRef::Literal(50.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)] };
                }
            }
            "radial" => {
                if !matches!(kind, PatternKind::Radial { .. }) {
                    *kind = PatternKind::Radial { frame: Frame::default() };
                }
            }
            other => return Err(format!("`pattern kind=` is linear or radial, not {other:?}")),
        }
    }
    match kind {
        PatternKind::Linear { step } => {
            if let Some(v) = args.get("step") {
                *step = if v.text().contains(',') {
                    param_triple(word, "step", v, Slot::Number, lookup)?
                } else {
                    [param_one(word, "step", v, Slot::Number, lookup)?, ParamRef::Literal(0.0), ParamRef::Literal(0.0)]
                };
            }
            if args.get("axis").is_some() || args.get("origin").is_some() {
                return Err("`pattern axis=` and `origin=` belong to a ring — say `kind=radial`".to_string());
            }
            if args.get("angle").is_some() {
                return Err("`pattern angle=` belongs to a ring — say `kind=radial`".to_string());
            }
        }
        PatternKind::Radial { frame } => {
            frame_with_keys(word, frame, args, lookup)?;
            if args.get("step").is_some() {
                return Err("`pattern step=` belongs to a row — say `kind=linear`".to_string());
            }
            if args.get("rows").is_some() || args.get("count2").is_some() || args.get("step2").is_some() {
                return Err("`pattern rows=` and `step2=` make a grid from a row — say `kind=linear`".to_string());
            }
            if args.get("extent").is_some() || args.get("spacing").is_some() {
                return Err("`pattern extent` and `spacing=` read a row's step as its whole run — say `kind=linear`".to_string());
            }
        }
    }
    if let Some(v) = args.get("r").or(args.get("radius")) {
        let radius = param_one(word, "r", v, Slot::Number, lookup)?;
        let continuity = args.get("g").map(|v| continuity(word, v)).transpose()?.unwrap_or(Continuity::G1);
        *fillet = Fillet { radius, continuity };
        *join = true;
    } else if args.get("g").is_some() {
        return Err("`pattern g=` goes with a radius, `r=`".to_string());
    }
    if args.get("join").is_some() {
        *join = flag(args, "join");
    }
    // A second direction typed as three numbers, which the row's own `step2=` single number
    // along Y cannot say.
    if let Some(v) = args.get("count2").or(args.get("rows")) {
        let count2 = param_one(word, "count2", v, Slot::Number, lookup)?;
        let step2 = match args.get("step2") {
            Some(s) if s.text().contains(',') => param_triple(word, "step2", s, Slot::Number, lookup)?,
            Some(s) => [ParamRef::Literal(0.0), param_one(word, "step2", s, Slot::Number, lookup)?, ParamRef::Literal(0.0)],
            None => options
                .second
                .as_ref()
                .map(|s| s.step.clone())
                .unwrap_or([ParamRef::Literal(0.0), ParamRef::Literal(50.0), ParamRef::Literal(0.0)]),
        };
        options.second = Some(fieldform_scene::PatternSecond { step: step2, count: count2 });
    } else if args.get("step2").is_some() {
        return Err("`pattern step2=` goes with `rows=` (or `count2=`)".to_string());
    }
    Ok(node)
}

// ------------------------------------------------------------------------------------ curves

/// Points typed as `x,y,z;x,y,z;...`. Numbers only: a curve's control points are numbers in the
/// document, not parameters, so a name here is refused rather than resolved to today's value.
pub fn points(verb: &str, word: &str, value: &Value) -> Result<Vec<nalgebra::Point3<f64>>, String> {
    let text = value.text();
    text.split(';')
        .map(|one| {
            let parts: Option<Vec<f64>> =
                one.split(',').map(|p| p.trim().parse::<f64>().ok().filter(|v| v.is_finite())).collect();
            match parts {
                Some(p) if p.len() == 3 => Ok(nalgebra::Point3::new(p[0], p[1], p[2])),
                _ => Err(format!(
                    "`{verb} {word}=` is points of three numbers separated by `;`, like 0,0,0;10,0,0 \u{2014} not {one:?} (a curve's points are numbers, not parameters)"
                )),
            }
        })
        .collect()
}

/// What `line` or `curve` asked to place: a degree, its points and their weights -- or nothing,
/// so the command starts click placement as it always did.
///
/// `line p0=... p1=...` or `line p=a;b`; `curve p=...` with the degree from `curve3`, `degree=`,
/// or the count of points; `w=` one weight a point, for an exact arc (`curve2 p=...;...;...
/// w=1,0.7071,1`). A single span, so a degree-n curve is exactly n + 1 points and anything else
/// is refused by count rather than resampled.
pub fn curve_asked(
    verb: &str,
    args: &Args,
) -> Result<Option<(usize, Vec<nalgebra::Point3<f64>>, Vec<f64>)>, String> {
    let line = verb == "line";
    let pts = match (args.get("p"), args.get("p0"), args.get("p1")) {
        (None, None, None) => {
            if args.get("w").is_some() {
                return Err(format!("`{verb} w=` weighs the points given by `p=`"));
            }
            return Ok(None);
        }
        (Some(p), None, None) => points(verb, "p", p)?,
        (None, Some(a), Some(b)) if line => {
            let mut both = points(verb, "p0", a)?;
            both.extend(points(verb, "p1", b)?);
            if both.len() != 2 {
                return Err("`line p0=` and `p1=` are one point each".to_string());
            }
            both
        }
        (None, Some(_), None) | (None, None, Some(_)) if line => {
            return Err("`line` wants both ends \u{2014} p0=x,y,z p1=x,y,z".to_string());
        }
        _ => return Err(format!("`{verb}` takes its points one way \u{2014} p=a;b;... or (a line) p0= and p1=")),
    };
    let degree = match args.get("degree") {
        _ if line => 1,
        Some(v) => match whole(verb, "degree", v)? {
            0 => return Err(format!("`{verb} degree=` is one or more")),
            d => d as usize,
        },
        None => pts.len().saturating_sub(1).max(1),
    };
    if pts.len() != degree + 1 {
        return Err(format!(
            "a degree {degree} curve is {} points, one span \u{2014} this gives {}",
            degree + 1,
            pts.len()
        ));
    }
    if pts.windows(2).all(|w| (w[1] - w[0]).norm() < 1e-9) {
        return Err(format!("the points of `{verb}` are all in one place \u{2014} a curve needs length"));
    }
    let weights = match args.get("w") {
        None => vec![1.0; pts.len()],
        Some(v) => {
            let text = v.text();
            let w: Option<Vec<f64>> =
                text.split(',').map(|t| t.trim().parse::<f64>().ok().filter(|w| *w > 0.0 && w.is_finite())).collect();
            match w {
                Some(w) if w.len() == pts.len() => w,
                _ => {
                    return Err(format!(
                        "`{verb} w=` is one weight above nought for each of the {} points \u{2014} not {text:?}",
                        pts.len()
                    ))
                }
            }
        }
    };
    Ok(Some((degree, pts, weights)))
}

/// `args` with each alias renamed to the field name it stands for, so one reader serves both
/// spellings. Both given at once is two answers to one question, and refused.
pub fn aliased(verb: &str, args: &Args, pairs: &[(&str, &str)]) -> Result<Args, String> {
    let mut out = Args::new();
    for (k, v) in args.iter() {
        let to = pairs.iter().find(|(alias, _)| *alias == k).map(|(_, canonical)| *canonical);
        match to {
            Some(canonical) if args.get(canonical).is_some() => {
                return Err(format!("`{verb} {k}=` and `{canonical}=` are one thing said twice \u{2014} say it once"));
            }
            Some(canonical) => out = out.with(canonical, v.clone()),
            None => out = out.with(k, v.clone()),
        }
    }
    Ok(out)
}

/// Whether a sweep line names its curves, which is what routes it to the arm that reads them.
pub fn names_a_curve(args: &Args) -> bool {
    ["profile", "curve", "spine", "section"].iter().any(|k| args.get(k).is_some())
}

/// A loft's smoothness by its word: g0 creased at every section, g1 through them without one.
pub fn loft_smooth_named(value: &Value) -> Result<fieldform_scene::LoftSmooth, String> {
    use fieldform_scene::LoftSmooth;
    match value {
        // `-smooth` alone asks for the smooth one, as `--g1` did.
        Value::Bool(true) => return Ok(LoftSmooth::G1),
        Value::Bool(false) => return Ok(LoftSmooth::G0),
        _ => {}
    }
    match value.text().trim().to_ascii_lowercase().as_str() {
        "g0" | "creased" => Ok(LoftSmooth::G0),
        "g1" | "smooth" => Ok(LoftSmooth::G1),
        other => Err(format!("`loft smooth=` is g0 or g1 \u{2014} not {other:?}")),
    }
}

/// How a drafted corner is cut: rounded by the offset's own arc, or mitred to an edge.
pub fn corners_named(value: &Value) -> Result<fieldform_scene::Corners, String> {
    use fieldform_scene::Corners;
    match value {
        // `-corners` alone, and `corners=false`, read as the flag this was before it had a word.
        Value::Bool(true) => return Ok(Corners::Sharp),
        Value::Bool(false) => return Ok(Corners::Round),
        _ => {}
    }
    match value.text().trim().to_ascii_lowercase().as_str() {
        "round" | "rounded" => Ok(Corners::Round),
        "sharp" | "mitre" | "miter" => Ok(Corners::Sharp),
        other => Err(format!("`corners=` is round or sharp \u{2014} not {other:?}")),
    }
}

/// Two percentages, `from,to`, each 0 to 100 and in order: a stretch of a path.
pub fn percent_pair(verb: &str, word: &str, value: &Value) -> Result<[f64; 2], String> {
    let text = value.text();
    let parts: Option<Vec<f64>> =
        text.split(',').map(|p| p.trim().parse::<f64>().ok().filter(|v| v.is_finite())).collect();
    let [from, to] = match parts.as_deref() {
        Some([a, b]) => [*a, *b],
        _ => {
            return Err(format!(
                "`{verb} {word}=` is two percentages of the path, from,to \u{2014} not {text:?}"
            ))
        }
    };
    if !(0.0..=100.0).contains(&from) || !(0.0..=100.0).contains(&to) {
        return Err(format!("`{verb} {word}=` is measured in percent, so each number is 0 to 100"));
    }
    if from >= to {
        return Err(format!("`{verb} {word}=` runs from the smaller number to the larger"));
    }
    Ok([from, to])
}

/// A sweep's section by its word: square, skewed or parallel.
pub fn section_named(value: &Value) -> Result<fieldform_scene::Section, String> {
    use fieldform_scene::Section;
    match value.text().trim().to_ascii_lowercase().as_str() {
        "square" => Ok(Section::Square),
        "skewed" | "flush" => Ok(Section::Skewed),
        "parallel" => Ok(Section::Parallel),
        other => Err(format!("`sweep section=` is square, skewed or parallel \u{2014} not {other:?}")),
    }
}

// ---------------------------------------------------------------------------------- placement

/// Which of the three placement verbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placing {
    Move,
    Turn,
    Scale,
}

impl Placing {
    pub fn word(self) -> &'static str {
        match self {
            Placing::Move => "move",
            Placing::Turn => "turn",
            Placing::Scale => "scale",
        }
    }
}

/// A placement block for `target`, or the one it already is edited in place.
///
/// **Ten `move`s leave one block** (A4) -- but only where adding to the block's numbers *is*
/// the motion the typed sequence means. The kernel applies a placement as one map, scale about
/// the pivot, then the turn about the pivot, then the move (`graph_field.rs`), whatever order
/// the verbs were typed; so folding a second edit into the same block is right only when the two
/// commute. A move always does: it lands in the translation, which sits outside the rest. A turn
/// folds in only onto a block that holds nothing but a turn about the same axis from the same
/// pivot, and a scale only onto a block that holds nothing but a scale about the same pivot.
/// Anything else wraps the block in a fresh placement, which composes exactly. The first version
/// added components regardless, and `turn by=0,0,90` then `turn by=90,0,0` came out with X
/// applied first, which is not what was typed.
///
/// **A parameter cannot be added to a number without an expression**, which `by=` refuses (A9),
/// so an amount that is a parameter folds in only where the slot is still the identity; a
/// number already there is wrapped rather than overwritten. `scale by=1,2,1` is refused until a
/// scale can be three numbers (item 69).
pub fn placed(
    placing: Placing,
    target: NodeId,
    existing: Option<(NodeId, &Node)>,
    args: &Args,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<(Node, Option<NodeId>), String> {
    let word = placing.word();
    let Some(by) = args.get("by") else {
        return Err(format!(
            "`{word}` wants `by=` — {}",
            match placing {
                Placing::Move => "`move target=box by=10,0,0`",
                Placing::Turn => "`turn target=box by=0,0,45`",
                Placing::Scale => "`scale target=box by=2`",
            }
        ));
    };
    let about = args.get("about").map(|v| param_triple(word, "about", v, Slot::Number, lookup).map(Place::At)).transpose()?;
    if placing == Placing::Move && about.is_some() {
        return Err("`move about=` means nothing — a move is the same from anywhere".to_string());
    }
    // The amount first, so the choice between folding in and wrapping can see it.
    let amounts: Vec<ParamRef> = match placing {
        Placing::Move | Placing::Turn => param_triple(word, "by", by, Slot::Number, lookup)?.to_vec(),
        Placing::Scale => {
            if by.text().contains(',') {
                return Err("`scale by=` is one number for now — a scale of three waits for item 69".to_string());
            }
            let factor = param_one(word, "by", by, Slot::Number, lookup)?;
            if let ParamRef::Literal(b) = factor
                && b <= 0.0
            {
                return Err(format!("`scale by=` must be above nought, not {b}"));
            }
            vec![factor]
        }
    };

    let is = |r: &ParamRef, v: f64| matches!(r, ParamRef::Literal(x) if *x == v);
    let identity = |slot: &ParamRef| is(slot, if placing == Placing::Scale { 1.0 } else { 0.0 });
    // Whether `amount` can be folded onto `slot`: two numbers add (or multiply); a parameter
    // lands only where there is no number to lose; and an amount that changes nothing (nought,
    // or a scale of one) leaves any slot as it is.
    let folds = |slot: &ParamRef, amount: &ParamRef| {
        matches!((slot, amount), (ParamRef::Literal(_), ParamRef::Literal(_))) || identity(slot) || identity(amount)
    };
    let fold = |slot: &ParamRef, amount: &ParamRef| match (slot, amount) {
        (ParamRef::Literal(a), ParamRef::Literal(b)) => {
            ParamRef::Literal(if placing == Placing::Scale { a * b } else { a + b })
        }
        _ if identity(amount) => slot.clone(),
        _ => amount.clone(),
    };

    // `-wrap` asks for a placement of its own whatever the target already is: the echo's way of
    // saying a placement that wraps another (`interaction-spec.md` §6.3), which folding would
    // otherwise merge into the one it wraps.
    let in_place = !flag(args, "wrap") && match existing.map(|(_, node)| node) {
        Some(Node::Transform { translation, rotation, scale, about: was_about, .. }) => {
            let no_turn = rotation.iter().all(|r| is(r, 0.0));
            let no_scale = is(scale, 1.0);
            let no_move = translation.iter().all(|t| is(t, 0.0));
            // The pivot asked for is the one the block has, or the block has not used its pivot
            // yet, so it may still be chosen.
            let pivot_agrees = match &about {
                None => true,
                Some(wanted) => wanted == was_about || (no_turn && no_scale),
            };
            let commutes = match placing {
                Placing::Move => translation.iter().zip(&amounts).all(|(s, a)| folds(s, a)),
                Placing::Turn => {
                    // One axis, the same axis: between what has turned and what is asked for,
                    // at most one component is anything but nought. Turns about one axis add;
                    // turns about two do not commute.
                    let axes = (0..3).filter(|&k| !is(&rotation[k], 0.0) || !is(&amounts[k], 0.0)).count();
                    no_move && no_scale && axes <= 1 && rotation.iter().zip(&amounts).all(|(s, a)| folds(s, a))
                }
                Placing::Scale => no_move && no_turn && folds(scale, &amounts[0]),
            };
            commutes && pivot_agrees
        }
        _ => false,
    };

    // Folded into the outermost placement over the target, or wrapped around it -- never around
    // the body itself while a placement already holds it, which would place the body twice.
    let mut node = if in_place {
        existing.map(|(_, node)| node.clone()).expect("an existing placement")
    } else {
        Node::Transform {
            input: Operand::filled(existing.map(|(id, _)| id).unwrap_or(target)),
            translation: [ParamRef::Literal(0.0); 3],
            rotation: [ParamRef::Literal(0.0); 3],
            scale: ParamRef::Literal(1.0),
            about: Place::origin(),
        }
    };
    let Node::Transform { translation, rotation, scale, about: at, .. } = &mut node else { unreachable!("built a placement") };
    match placing {
        Placing::Move => {
            for (slot, amount) in translation.iter_mut().zip(&amounts) {
                *slot = fold(slot, amount);
            }
        }
        Placing::Turn => {
            for (slot, amount) in rotation.iter_mut().zip(&amounts) {
                *slot = fold(slot, amount);
            }
        }
        Placing::Scale => *scale = fold(scale, &amounts[0]),
    }
    if let Some(about) = about {
        *at = about;
    }
    Ok((node, in_place.then(|| existing.map(|(id, _)| id)).flatten()))
}

/// An attractor's target by name: a curve first, then a point, as the app resolves both.
pub fn toward_named(args: &Args, curve: &dyn Fn(&str) -> Option<fieldform_scene::CurveId>, lookup: &dyn Fn(&str) -> Lookup) -> Result<Option<Attractor>, String> {
    let Some(v) = args.get("toward") else { return Ok(None) };
    let name = v.text();
    if let Some(c) = curve(name.trim()) {
        return Ok(Some(Attractor::Curve(c)));
    }
    match lookup(name.trim()) {
        Lookup::Block(id) => Ok(Some(Attractor::Point(id))),
        Lookup::Both => Err(format!("{name:?} is both a parameter and a block — rename one, so `attract toward=` knows which")),
        Lookup::Parameter(_) => Err(format!("`attract toward=` wants a curve or a point, and {name:?} is a parameter")),
        Lookup::Nothing => Err(format!("no curve or block is called {name:?} for `attract toward=`")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use fieldform_scene::{
        AngleSides, CurveId, Outline, PatternOptions, PatternSecond, QuadMeshId, Sides, Spacing, WallEnds, WallSide,
    };
    use std::collections::BTreeSet;

    fn args(pairs: &[(&str, Value)]) -> Args {
        pairs.iter().fold(Args::new(), |a, (k, v)| a.with(k, v.clone()))
    }

    fn s(text: &str) -> Value {
        Value::Str(text.to_string())
    }

    fn nobody(_: &str) -> Lookup {
        Lookup::Nothing
    }

    // ------------------------------------------------------------------- curves (A6)

    #[test]
    // `0.70710678` below is the weight as typed, and the test holds the curve to exactly that: a
    // curve keeps the number its line gave. 1/√2 to full precision is another number, so the
    // constant this lint asks for would fail the assertion rather than tidy it.
    #[allow(clippy::approx_constant)]
    fn a_curve_is_placed_from_its_points_and_a_wrong_count_is_refused() {
        let line = curve_asked("line", &args(&[("p0", s("0,0,0")), ("p1", s("40,0,0"))])).unwrap().unwrap();
        assert_eq!(line.0, 1);
        assert_eq!(line.1[1], nalgebra::Point3::new(40.0, 0.0, 0.0));
        assert!(curve_asked("line", &Args::new()).unwrap().is_none(), "no points: click placement");
        assert!(curve_asked("line", &args(&[("p0", s("0,0,0"))])).is_err(), "half a line");
        // The degree from the count, or from `curve3`, and they must agree.
        let cubic = curve_asked("curve", &args(&[("p", s("0,0,0;10,10,0;20,-10,0;30,0,0"))])).unwrap().unwrap();
        assert_eq!(cubic.0, 3);
        let refused = curve_asked("curve", &args(&[("degree", Value::Int(3)), ("p", s("0,0,0;10,0,0;20,0,0"))])).unwrap_err();
        assert!(refused.contains("is 4 points"), "{refused}");
        // An exact quarter arc: weights one a point.
        let arc = curve_asked("curve", &args(&[("degree", Value::Int(2)), ("p", s("10,0,0;10,10,0;0,10,0")), ("w", s("1,0.70710678,1"))]))
            .unwrap()
            .unwrap();
        assert_eq!(arc.2[1], 0.70710678);
        assert!(curve_asked("curve", &args(&[("p", s("10,0,0;10,10,0;0,10,0")), ("w", s("1,1"))])).is_err(), "a weight short");
        assert!(curve_asked("curve", &args(&[("p", s("0,0,0;0,0,0"))])).is_err(), "no length");
        assert!(curve_asked("curve", &args(&[("p", s("0,0,0;h,0,0"))])).is_err(), "points are numbers");
    }

    #[test]
    fn an_alias_reads_as_its_field_and_both_at_once_is_refused() {
        let pairs = &[("height", "depth"), ("height_2", "two")];
        let read = aliased("extrude", &args(&[("height", Value::Int(30)), ("draft", Value::Int(2))]), pairs).unwrap();
        assert!(read.get("depth").is_some() && read.get("height").is_none() && read.get("draft").is_some());
        assert!(aliased("extrude", &args(&[("height", Value::Int(30)), ("depth", Value::Int(20))]), pairs).is_err());
    }

    // ------------------------------------------------------------------- the drift test

    /// One of every kind a declared verb reaches, with every field set to something other than
    /// its default so the serializer writes it. A kind with several shapes (a lattice's maps and
    /// phases, a pattern's row and ring, an attractor's curve and point) is here once per shape.
    pub(crate) fn samples() -> Vec<Node> {
        let lit = ParamRef::Literal;
        let p = |x: f64, y: f64, z: f64| [lit(x), lit(y), lit(z)];
        let id = NodeId::new();
        let other = NodeId::new();
        let curve = CurveId::new();
        let frame = Frame::At { origin: p(1.0, 2.0, 3.0), axis: p(0.0, 1.0, 0.0) };
        let lattice = |cell: LatticeCell, map: CellMap, phase: LatticePhase| Node::Lattice { cell, frame: frame.clone(), map, phase };
        vec![
            Node::Point { at: p(1.0, 2.0, 3.0) },
            Node::Plane { origin: Place::At(p(1.0, 2.0, 3.0)), flipped: true, turn: Turn::By(p(10.0, 20.0, 30.0)) },
            Node::Sphere { centre: Place::At(p(1.0, 2.0, 3.0)), radius: lit(5.0) },
            Node::Cuboid { edge: fieldform_scene::Fillet { radius: lit(1.0), continuity: fieldform_scene::Continuity::Chamfer }, centre: Place::At(p(1.0, 2.0, 3.0)), size: p(4.0, 5.0, 6.0), turn: Turn::By(p(10.0, 20.0, 30.0)) },
            Node::SkewedBox { centre: Place::At(p(1.0, 2.0, 3.0)), edges: [p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0), p(0.0, 0.0, 1.0)] },
            Node::Cylinder { edge: fieldform_scene::Fillet { radius: lit(0.5), continuity: fieldform_scene::Continuity::G2 }, ends: Ends::between(Place::At(p(0.0, 0.0, 0.0)), Place::At(p(0.0, 0.0, 9.0))), radius: lit(3.0) },
            Node::Cone { edge: fieldform_scene::Fillet { radius: lit(0.5), continuity: fieldform_scene::Continuity::G2 }, ends: Ends::between(Place::At(p(0.0, 0.0, 0.0)), Place::At(p(0.0, 0.0, 9.0))), base_radius: lit(3.0), top_radius: lit(1.0) },
            Node::Torus { centre: Place::At(p(1.0, 2.0, 3.0)), major_radius: lit(5.0), minor_radius: lit(1.0), turn: Turn::By(p(10.0, 20.0, 30.0)) },
            Node::Capsule { ends: Ends::between(Place::At(p(0.0, 0.0, 0.0)), Place::At(p(0.0, 0.0, 9.0))), start_radius: lit(3.0), end_radius: lit(2.0) },
            lattice(LatticeCell::Gyroid, CellMap::Rectangular { cell: lit(8.0) }, LatticePhase::Sheet { thickness: lit(1.0), offset: lit(0.3) }),
            lattice(LatticeCell::Gyroid, CellMap::Cylindrical { cell_radius: lit(5.0), cell_height: lit(6.0), arc_count: 12 }, LatticePhase::Solid { density: lit(0.4) }),
            lattice(LatticeCell::Bcc, CellMap::Spherical { cell_radius: lit(5.0), azimuth_count: 12, polar_count: 6 }, LatticePhase::Strut { radius: lit(0.5) }),
            lattice(
                LatticeCell::voronoi_foam(7),
                CellMap::Conformal { surface: QuadMeshId::new(), cells_per_face: 2, depth: lit(4.0), thickness: Some(lit(1.0)) },
                LatticePhase::Sheet { thickness: lit(1.0), offset: lit(0.0) },
            ),
            Node::Ramp { input: Operand::filled(id), in_min: lit(0.0), in_max: lit(1.0), out_min: lit(2.0), out_max: lit(3.0), shape: RampShape::Smooth },
            Node::Boolean { op: BooleanOp::Union, inputs: vec![Operand::filled(id), Operand::filled(other)] },
            Node::Blend { op: BooleanOp::Union, a: Operand::filled(id), b: Operand::filled(other), fillet: Fillet { radius: lit(2.0), continuity: Continuity::G2 } },
            Node::Morph { a: Operand::filled(id), b: Operand::filled(other), t: lit(0.5) },
            Node::Offset { input: Operand::filled(id), distance: lit(2.0) },
            Node::Shell { input: Operand::filled(id), thickness: lit(1.0) },
            Node::Twist { input: Operand::filled(id), degrees_per_mm: lit(1.0), frame: frame.clone() },
            Node::Mirror { input: Operand::filled(id), frame: frame.clone(), join: true },
            Node::Attract { input: Operand::filled(id), toward: Attractor::Curve(curve), strength: lit(0.5), falloff: lit(10.0) },
            Node::Attract { input: Operand::filled(id), toward: Attractor::Point(other), strength: lit(0.5), falloff: lit(10.0) },
            Node::Pattern {
                input: Operand::filled(id),
                count: lit(4.0),
                kind: PatternKind::Linear { step: p(10.0, 0.0, 0.0) },
                join: true,
                fillet: Fillet { radius: lit(1.0), continuity: Continuity::G1 },
                options: PatternOptions {
                    spacing: Spacing::Extent,
                    symmetric: true,
                    angle: lit(360.0),
                    second: Some(PatternSecond { step: p(0.0, 10.0, 0.0), count: lit(3.0) }),
                    skip: vec![2],
                },
            },
            Node::Pattern {
                input: Operand::filled(id),
                count: lit(6.0),
                kind: PatternKind::Radial { frame: frame.clone() },
                join: false,
                fillet: Fillet::hard(),
                options: PatternOptions { angle: lit(180.0), ..PatternOptions::default() },
            },
            Node::Transform { input: Operand::filled(id), translation: p(1.0, 2.0, 3.0), rotation: p(10.0, 20.0, 30.0), scale: lit(2.0), about: Place::At(p(1.0, 1.0, 1.0)) },
            Node::Cage(fieldform_scene::CageId::new()),
            // The blocks built on curves, once with a profile and once as a wall, with every
            // number off its default so the serializer writes it.
            Node::Extrude { rims: fieldform_scene::Rims { radius: lit(1.5), continuity: fieldform_scene::Continuity::G2, at: fieldform_scene::RimEnds::End },
                outline: Outline::closed(curve),
                sweep: fieldform_scene::Sweep::Normal,
                start: lit(1.0),
                height: lit(30.0),
                sides: Sides::Two { height: lit(5.0), draft: lit(2.0) },
                draft: lit(3.0),
                corners: fieldform_scene::Corners::Sharp,
            },
            Node::Extrude { rims: Default::default(),
                outline: Outline::Wall { curve, thickness: lit(2.0), side: WallSide::Left, ends: WallEnds::Round },
                sweep: fieldform_scene::Sweep::Normal,
                start: lit(0.0),
                height: lit(30.0),
                sides: Sides::Symmetric,
                draft: lit(0.0),
                corners: fieldform_scene::Corners::Round,
            },
            Node::Revolve {
                outline: Outline::closed(curve),
                frame: frame.clone(),
                angle: lit(90.0),
                sides: AngleSides::Two { angle: lit(30.0) },
                rims: fieldform_scene::Rims { radius: lit(1.5), continuity: fieldform_scene::Continuity::G2, at: fieldform_scene::RimEnds::End },
            },
            Node::Revolve {
                outline: Outline::Wall { curve, thickness: lit(2.0), side: WallSide::Right, ends: WallEnds::Round },
                frame: Frame::Along { curve },
                angle: lit(360.0),
                sides: AngleSides::Symmetric,
                rims: Default::default(),
            },
            Node::Sweep { rims: fieldform_scene::Rims { radius: lit(1.5), continuity: fieldform_scene::Continuity::G2, at: fieldform_scene::RimEnds::End },
                outline: Outline::closed(curve),
                spine: curve,
                section: fieldform_scene::Section::Skewed,
                // Every number of the law, off its default so the serializer writes it.
                law: fieldform_scene::SweepLaw {
                    taper: fieldform_scene::Taper::Scale(lit(0.5)),
                    corners: fieldform_scene::Corners::Sharp,
                    twist: lit(90.0),
                    shape: RampShape::Smooth,
                    extent: [lit(10.0), lit(90.0)],
                },
            },
            Node::Sweep { rims: Default::default(),
                outline: Outline::closed(curve),
                spine: curve,
                section: fieldform_scene::Section::Square,
                // The other taper, which writes `taper_draft` where the one above writes
                // `taper_scale`: one field, two spellings on disk.
                law: fieldform_scene::SweepLaw {
                    taper: fieldform_scene::Taper::Draft(lit(5.0)),
                    ..Default::default()
                },
            },
            Node::Sweep { rims: Default::default(),
                outline: Outline::Wall { curve, thickness: lit(1.0), side: WallSide::Left, ends: WallEnds::Round },
                spine: curve,
                section: fieldform_scene::Section::Parallel,
                law: fieldform_scene::SweepLaw::default(),
            },
            Node::Pipe { curve, radius: lit(3.0) },
            Node::Loft { rims: fieldform_scene::Rims { radius: lit(1.5), continuity: fieldform_scene::Continuity::G2, at: fieldform_scene::RimEnds::End },
                sections: vec![Outline::closed(curve), Outline::closed(curve)],
                spine: Some(curve),
                smooth: fieldform_scene::LoftSmooth::G1,
            },
        ]
    }

    /// The serialized keys of a node, with a nested frame or pair of ends opened one level, so
    /// the test speaks the console's words (`from`, `to`, `origin`, `axis`).
    fn fields_of(node: &Node) -> BTreeSet<String> {
        let value = serde_json::to_value(node).expect("a sample serializes");
        let object = value.as_object().expect("a node is an object");
        let mut out = BTreeSet::new();
        for (k, v) in object {
            if k == "kind" {
                continue;
            }
            let inner = v.as_object().filter(|o| o.len() == 1).and_then(|o| o.values().next()).and_then(|v| v.as_object());
            match inner {
                Some(inner) if matches!(k.as_str(), "ends" | "frame") => out.extend(inner.keys().cloned()),
                _ => {
                    out.insert(k.clone());
                }
            }
        }
        out
    }

    /// Every field every block carries is a key its console verb reads, or is recorded as not
    /// read with the reason -- so a number added to a block without a console word fails here,
    /// and so does a key in the table that no block has any more.
    #[test]
    fn every_field_a_block_has_is_a_key_its_verb_reads() {
        let mut seen_ops: BTreeSet<&str> = BTreeSet::new();
        let mut fields_by_op: std::collections::BTreeMap<&str, BTreeSet<String>> = Default::default();
        for node in samples() {
            let Reach::Ops(ops) = reach_of(&node) else { panic!("a sample is a kind the console reaches") };
            let fields = fields_of(&node);
            let covered: BTreeSet<&str> = ops
                .iter()
                .flat_map(|op| match &verb(op).unwrap_or_else(|| panic!("{op} is in VERBS")).keys {
                    Keys::Declared { reads, not_read } => reads
                        .iter()
                        .flat_map(|k| k.covers.iter().copied())
                        .chain(not_read.iter().map(|(f, _)| *f))
                        .collect::<Vec<_>>(),
                    Keys::Undeclared(why) => panic!("{op} has not declared its keys ({why}) yet a sample reaches it"),
                })
                .collect();
            for field in &fields {
                assert!(
                    covered.contains(field.as_str()),
                    "{ops:?}: the block's field {field:?} is reached by no console key and not recorded as unread — fields {fields:?}"
                );
            }
            for op in ops.iter() {
                seen_ops.insert(op);
                fields_by_op.entry(op).or_default().extend(fields.iter().cloned());
            }
        }
        // The other way: a key or an exception in the table names a field some block has.
        for verb in VERBS {
            let Keys::Declared { reads, not_read } = &verb.keys else { continue };
            let Some(fields) = fields_by_op.get(verb.op) else {
                // A verb that makes no block (`run`, `saveas`) covers no field and needs no
                // sample; a verb that covers fields must have a block to hold them to.
                assert!(
                    reads.iter().all(|k| k.covers.is_empty()) && not_read.is_empty(),
                    "{} declares keys but no sample block reaches it — add one to `samples`",
                    verb.op
                );
                continue;
            };
            for named in reads.iter().flat_map(|k| k.covers.iter()).chain(not_read.iter().map(|(f, _)| f)) {
                assert!(fields.contains(*named), "{}: the table names a field {named:?} that its block does not write", verb.op);
            }
        }
        // And every declared verb is a word the console knows.
        let commands = crate::console_commands();
        for verb in VERBS {
            assert!(commands.iter().any(|(_, op)| *op == verb.op), "{} is in the key table but not in the console's commands", verb.op);
        }
    }

    #[test]
    fn an_unknown_key_is_refused_by_name_with_what_the_verb_reads() {
        let refused = refusal("cube", "create.cube", &args(&[("radius", Value::Int(5))])).expect("refused");
        assert_eq!(refused, "`cube` doesn't read `radius` — it reads size, centre, turn, edge, edge_continuity");
        assert!(refusal("cube", "create.cube", &args(&[("size", s("1,2,3"))])).is_none());
        // Case does not matter for a key as typed, because the grammar lowers it before anything
        // reads it; a key that arrives in another case was not typed through the grammar and is
        // refused, since no reader would see it.
        let (typed, _) =
            crate::console_arguments(crate::console_words("cube Size=1,2,3 -Flipped").into_iter().skip(1)).unwrap();
        assert!(typed.get("size").is_some() && typed.get("flipped").is_some(), "lowered as parsed");
        assert!(refusal("cube", "create.cube", &args(&[("Size", s("1,2,3"))])).is_some());
        // The curve verbs declare theirs too, since A6.
        assert!(refusal("extrude", "create.extrude", &args(&[("anything", Value::Int(1))])).is_some());
        assert!(refusal("extrude", "create.extrude", &args(&[("height", Value::Int(30))])).is_none());
        // Nor does a menu, which passes no keys.
        assert!(refusal("sphere", "create.sphere", &Args::new()).is_none());
    }

    /// `estimate` reads a start and a target and nothing else, and `help estimate` says what it
    /// is for before what it reads, from the same table that refuses the rest.
    #[test]
    fn estimate_reads_only_the_keys_it_declares() {
        let refused = refusal("estimate", "view.estimate", &args(&[("radius", Value::Int(1))])).expect("refused");
        assert_eq!(refused, "`estimate` doesn't read `radius` — it reads node, stiffness");
        assert!(refusal("estimate", "view.estimate", &args(&[("node", s("infill")), ("stiffness", Value::Int(5))])).is_none());
        assert!(refusal("estimate", "view.estimate", &Args::new()).is_none());
        let help = help_for("estimate", "view.estimate").expect("in the table");
        assert!(help.starts_with("`estimate` — a lattice's stiffness and strength"), "{help}");
        assert!(help.ends_with("it reads node, stiffness"), "{help}");
        // A verb with no sentence of its own reads as it always did.
        assert_eq!(help_for("cube", "create.cube").as_deref(), Some("`cube` reads size, centre, turn, edge, edge_continuity"));
    }

    /// **`variable` and `inline` are the V typed, and read the row's name and nothing else**
    /// (`block-ownership-spec.md` §5.1, §5.2, §11). The name is their own argument, as a
    /// parameter's is `param`'s, so a quoted word is it rather than a name for something made;
    /// any other key is refused by name; and `help` says what each is for before what it reads.
    #[test]
    fn variable_and_inline_read_a_row_s_name_and_refuse_every_other_key() {
        let commands = crate::console_commands();
        for (word, op) in [("variable", "block.make_variable"), ("inline", "block.put_back_inside")] {
            assert!(commands.iter().any(|(w, o)| w == word && *o == op), "`{word}` is the console's word for {op}");
            assert!(reads_its_own_name(op), "`{word} 'Sphere'` names the row it acts on");
            assert!(refusal(word, op, &args(&[("name", s("Sphere"))])).is_none());
            assert_eq!(
                refusal(word, op, &args(&[("radius", Value::Int(5))])).as_deref(),
                Some(format!("`{word}` doesn't read `radius` — it reads name").as_str()),
            );
            let help = help_for(word, op).expect("in the table");
            assert!(help.starts_with(&format!("`{word}` — ")), "{help}");
            assert!(help.ends_with("it reads name"), "{help}");
        }
    }

    /// **A row is found by its name before any row's label, and two are never guessed between.**
    ///
    /// A name is what a row is typed by (`block-ownership-spec.md` §6.1), so a block called
    /// `Sphere` answers `variable 'Sphere'` although an unnamed sphere's row shows the same word;
    /// a label is looked at only where nothing carries the name. Two of either is the question the
    /// line does not answer, and the V it stands for acts on one row, so it is refused with what
    /// shares the word -- as a value naming both a parameter and a block is (A9).
    #[test]
    fn a_row_is_found_by_its_name_before_any_label_and_two_are_never_guessed_between() {
        let (block, other) = (RowCalled::Block(NodeId::new()), RowCalled::Block(NodeId::new()));
        let curve = RowCalled::Curve(fieldform_scene::CurveId::new());
        let parameter = RowCalled::Parameter(ParamId::new());
        assert_eq!(row_called("variable", "ball", &[block], &[other]), Ok(block), "the name wins");
        assert_eq!(row_called("variable", "Sphere", &[], &[other]), Ok(other), "else the label");
        assert_eq!(row_called("inline", "r", &[parameter], &[]), Ok(parameter));

        let two = row_called("variable", "l1", &[block, curve], &[]).expect_err("a block and a curve");
        assert!(two.contains("a block and a curve") && two.contains("rename one"), "{two}");
        let shown = row_called("variable", "Sphere", &[], &[block, other]).expect_err("two spheres");
        assert!(shown.contains("2 blocks") && shown.contains("name the one you mean"), "{shown}");
        let none = row_called("inline", "nothing", &[], &[]).expect_err("nothing is called that");
        assert!(none.starts_with("nothing is called \"nothing\""), "{none}");
        let blank = row_called("variable", "  ", &[block], &[]).expect_err("no name");
        assert!(blank.contains("`variable 'Sphere'`"), "{blank}");
    }

    /// **A line a reply gives to type is read back as the row it names.** The answer to
    /// `variable 'r'` said to type `inline r`, which the grammar refuses before `inline` sees it,
    /// since the console has no bare word. Held against the console's own reading of a line, for
    /// the names a row can carry: a parameter's may hold a dash or a space, and any an apostrophe.
    #[test]
    fn a_line_a_reply_gives_to_type_is_read_back_as_the_row_it_names() {
        for row in ["r", "cap-1", "left bearing", "Profile 2", "Bob's", "the \"top\""] {
            let line = line_naming("inline", row);
            let mut words = crate::console_words(&line).into_iter();
            assert_eq!(words.next().map(|(w, _)| w).as_deref(), Some("inline"), "{line}");
            let (args, naming) =
                crate::console_arguments(words).unwrap_or_else(|why| panic!("{line}: {why}"));
            assert!(args.iter().next().is_none(), "{line}: no key=value");
            assert_eq!(naming.as_deref(), Some(row), "{line}");
        }
    }

    // ------------------------------------------------------------------------ values (A9)

    #[test]
    fn a_value_is_a_number_a_parameter_or_where_the_slot_allows_a_block() {
        let param = ParamId::new();
        let block = NodeId::new();
        let lookup = |name: &str| match name {
            "s" => Lookup::Parameter(param),
            "ball" => Lookup::Block(block),
            "both" => Lookup::Both,
            _ => Lookup::Nothing,
        };
        assert_eq!(param_ref("sphere", "radius", "40", Slot::Number, &lookup), Ok(ParamRef::Literal(40.0)));
        assert_eq!(param_ref("sphere", "radius", "s", Slot::Number, &lookup), Ok(ParamRef::Named(param)));
        assert_eq!(param_ref("offset", "distance", "ball", Slot::NumberOrField, &lookup), Ok(ParamRef::Field(block)));
        let refused = param_ref("sphere", "radius", "ball", Slot::Number, &lookup).unwrap_err();
        assert!(refused.contains("is a block"), "{refused}");
        let ambiguous = param_ref("sphere", "radius", "both", Slot::Number, &lookup).unwrap_err();
        assert!(ambiguous.contains("both a parameter and a block"), "{ambiguous}");
        let nothing = param_ref("sphere", "radius", "nope", Slot::Number, &lookup).unwrap_err();
        assert!(nothing.starts_with("nothing is called \"nope\""), "{nothing}");
        // An expression is not a value: one expression mechanism, in `param`.
        assert!(param_ref("sphere", "radius", "s/2", Slot::Number, &lookup).is_err());
        // A triple is three such values.
        let centre = param_triple("sphere", "centre", &s("s,40,0"), Slot::Number, &lookup).unwrap();
        assert_eq!(centre, [ParamRef::Named(param), ParamRef::Literal(40.0), ParamRef::Literal(0.0)]);
        assert!(param_triple("sphere", "centre", &s("1,2"), Slot::Number, &lookup).is_err());
        // A single number arrives typed by `parse_value`, and reads the same.
        assert_eq!(param_one("sphere", "radius", &Value::Int(7), Slot::Number, &lookup), Ok(ParamRef::Literal(7.0)));
        assert_eq!(param_one("sphere", "radius", &Value::Float(7.5), Slot::Number, &lookup), Ok(ParamRef::Literal(7.5)));
    }

    #[test]
    fn an_axis_is_a_letter_or_three_numbers() {
        let z = axis_triple("twist", "axis", &s("z"), &nobody).unwrap();
        assert_eq!(z, [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(1.0)]);
        let any = axis_triple("twist", "axis", &s("1,1,0"), &nobody).unwrap();
        assert_eq!(any, [ParamRef::Literal(1.0), ParamRef::Literal(1.0), ParamRef::Literal(0.0)]);
        assert!(axis_triple("twist", "axis", &s("w"), &nobody).is_err());
    }

    // ----------------------------------------------------------------------- primitives (A1)

    fn default_sphere() -> Node {
        Node::Sphere { centre: Place::At([ParamRef::Literal(0.0); 3]), radius: ParamRef::Literal(50.0) }
    }

    #[test]
    fn a_sphere_reads_its_radius_and_centre_and_keeps_what_was_not_said() {
        let node = primitive_with_keys(default_sphere(), "sphere", &args(&[("radius", Value::Int(10)), ("centre", s("40,0,0"))]), &nobody).unwrap();
        let Node::Sphere { centre: Place::At(c), radius } = node else { panic!("still a sphere") };
        assert_eq!(radius, ParamRef::Literal(10.0));
        assert_eq!(c, [ParamRef::Literal(40.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]);
        let node = primitive_with_keys(default_sphere(), "sphere", &args(&[("radius", Value::Int(10))]), &nobody).unwrap();
        let Node::Sphere { centre: Place::At(c), .. } = node else { panic!("still a sphere") };
        assert_eq!(c, [ParamRef::Literal(0.0); 3], "an omitted key keeps the default");
    }

    #[test]
    fn a_cube_s_size_is_three_numbers_or_one_for_all_three() {
        let cube = Node::Cuboid { edge: Default::default(), centre: Place::At([ParamRef::Literal(0.0); 3]), size: [ParamRef::Literal(100.0); 3], turn: fieldform_scene::no_turn() };
        let node = primitive_with_keys(cube.clone(), "cube", &args(&[("size", s("10,20,30")), ("turn", s("0,0,30"))]), &nobody).unwrap();
        let Node::Cuboid { size, turn: Turn::By(t), .. } = node else { panic!("still a cuboid") };
        assert_eq!(size, [ParamRef::Literal(10.0), ParamRef::Literal(20.0), ParamRef::Literal(30.0)]);
        assert_eq!(t, [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(30.0)]);
        let node = primitive_with_keys(cube, "cube", &args(&[("size", Value::Int(40))]), &nobody).unwrap();
        let Node::Cuboid { size, .. } = node else { panic!("still a cuboid") };
        assert_eq!(size, [ParamRef::Literal(40.0); 3]);
    }

    /// **A box's and a cylinder's edges are `edge=` and `edge_continuity=`**, the field names, by
    /// the words a blend's `g=` takes; a shape on a sharp edge is refused, as a rim's is.
    #[test]
    fn a_cube_s_cylinder_s_or_cone_s_edge_is_its_reach_and_its_shape() {
        let cube = Node::Cuboid { edge: Default::default(), centre: Place::At([ParamRef::Literal(0.0); 3]), size: [ParamRef::Literal(100.0); 3], turn: fieldform_scene::no_turn() };
        let node = primitive_with_keys(cube.clone(), "cube", &args(&[("edge", Value::Int(4)), ("edge_continuity", s("chamfer"))]), &nobody).unwrap();
        let Node::Cuboid { edge, .. } = node else { panic!("still a cuboid") };
        assert_eq!((edge.radius, edge.continuity), (ParamRef::Literal(4.0), Continuity::Chamfer));
        let cyl = Node::Cylinder { edge: Default::default(), ends: Ends::between(Place::at(0.0, 0.0, -50.0), Place::at(0.0, 0.0, 50.0)), radius: ParamRef::Literal(50.0) };
        let node = primitive_with_keys(cyl.clone(), "cylinder", &args(&[("edge", Value::Int(3))]), &nobody).unwrap();
        let Node::Cylinder { edge, .. } = node else { panic!("still a cylinder") };
        assert_eq!((edge.radius, edge.continuity), (ParamRef::Literal(3.0), Continuity::G1), "G1 unless told");
        let cone = Node::Cone { edge: Default::default(), ends: Ends::between(Place::at(0.0, 0.0, 0.0), Place::at(0.0, 0.0, 20.0)), base_radius: ParamRef::Literal(8.0), top_radius: ParamRef::Literal(0.0) };
        let node = primitive_with_keys(cone, "cone", &args(&[("edge", Value::Int(2)), ("edge_continuity", s("g2"))]), &nobody).unwrap();
        let Node::Cone { edge, .. } = node else { panic!("still a cone") };
        assert_eq!((edge.radius, edge.continuity), (ParamRef::Literal(2.0), Continuity::G2));
        assert!(primitive_with_keys(cube, "cube", &args(&[("edge_continuity", s("g2"))]), &nobody).is_err(), "a sharp edge has no shape");
        assert!(primitive_with_keys(cyl, "cylinder", &args(&[("edge", Value::Int(3)), ("edge_continuity", s("g7"))]), &nobody).is_err());
    }

    #[test]
    fn a_cylinder_s_ends_are_from_and_to_and_either_alone_moves_one_end() {
        let cyl = Node::Cylinder { edge: Default::default(), ends: Ends::between(Place::at(0.0, 0.0, -50.0), Place::at(0.0, 0.0, 50.0)), radius: ParamRef::Literal(50.0) };
        let node = primitive_with_keys(cyl.clone(), "cylinder", &args(&[("radius", Value::Int(5)), ("from", s("0,0,0")), ("to", s("0,0,10"))]), &nobody).unwrap();
        let Node::Cylinder { edge: _, ends: Ends::At { from: Place::At(f), to: Place::At(t) }, radius } = node else { panic!("explicit ends") };
        assert_eq!(radius, ParamRef::Literal(5.0));
        assert_eq!(f, [ParamRef::Literal(0.0); 3]);
        assert_eq!(t, [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(10.0)]);
        let node = primitive_with_keys(cyl, "cylinder", &args(&[("to", s("0,0,10"))]), &nobody).unwrap();
        let Node::Cylinder { ends: Ends::At { from: Place::At(f), .. }, .. } = node else { panic!("explicit ends") };
        assert_eq!(f, [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(-50.0)], "the end not named stays");
    }

    #[test]
    fn a_skewed_box_s_edges_are_three_triples_and_a_plane_can_be_flipped() {
        let skew = Node::SkewedBox { centre: Place::origin(), edges: [[ParamRef::Literal(1.0); 3]; 3] };
        let node = primitive_with_keys(skew.clone(), "skewedbox", &args(&[("edges", s("10,0,0;0,20,0;0,0,30"))]), &nobody).unwrap();
        let Node::SkewedBox { edges, .. } = node else { panic!("still skewed") };
        assert_eq!(edges[1], [ParamRef::Literal(0.0), ParamRef::Literal(20.0), ParamRef::Literal(0.0)]);
        assert!(primitive_with_keys(skew, "skewedbox", &args(&[("edges", s("10,0,0;0,20,0"))]), &nobody).is_err());
        let plane = Node::Plane { origin: Place::origin(), flipped: false, turn: fieldform_scene::no_turn() };
        let node = primitive_with_keys(plane, "plane", &args(&[("flipped", Value::Bool(true)), ("origin", s("0,0,-30"))]), &nobody).unwrap();
        let Node::Plane { flipped, origin: Place::At(o), .. } = node else { panic!("still a plane") };
        assert!(flipped);
        assert_eq!(o[2], ParamRef::Literal(-30.0));
    }

    // --------------------------------------------------------------------- the lattice

    fn seed(b: fieldform_scene::Banding) -> LatticePhase {
        use fieldform_scene::Banding;
        match b {
            Banding::Sheet => LatticePhase::sheet(ParamRef::Literal(2.0)),
            Banding::Solid => LatticePhase::Solid { density: ParamRef::Literal(0.4) },
            Banding::Strut => LatticePhase::Strut { radius: ParamRef::Literal(1.2) },
        }
    }

    fn default_lattice() -> Node {
        Node::Lattice {
            cell: LatticeCell::Gyroid,
            frame: Frame::default(),
            map: CellMap::Rectangular { cell: ParamRef::Literal(20.0) },
            phase: LatticePhase::sheet(ParamRef::Literal(2.0)),
        }
    }

    #[test]
    fn a_lattice_reads_its_cell_map_phase_and_frame_by_their_own_names() {
        use fieldform_scene::Banding;
        let offered = |_: &LatticeCell| -> &'static [Banding] { &[Banding::Sheet, Banding::Solid] };
        let node = lattice_with_keys(
            default_lattice(),
            &args(&[("cell", s("schwarz_p")), ("cell_size", Value::Int(10)), ("phase", s("solid")), ("density", Value::Float(0.3)), ("axis", s("x")), ("origin", s("1,2,3"))]),
            &offered,
            &seed,
            &nobody,
        )
        .unwrap();
        let Node::Lattice { cell, map: CellMap::Rectangular { cell: size }, phase: LatticePhase::Solid { density }, frame: Frame::At { axis, origin } } = node else {
            panic!("a solid schwarz P on a rectangular map, aimed")
        };
        assert_eq!(cell.name(), "schwarz_p");
        assert_eq!(size, ParamRef::Literal(10.0));
        assert_eq!(density, ParamRef::Literal(0.3));
        assert_eq!(axis, [ParamRef::Literal(1.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]);
        assert_eq!(origin[2], ParamRef::Literal(3.0));
        // The old spelling of a cell still lands.
        let node = lattice_with_keys(default_lattice(), &args(&[("cell", s("schwarzp"))]), &offered, &seed, &nobody).unwrap();
        let Node::Lattice { cell, .. } = node else { panic!("a lattice") };
        assert_eq!(cell.name(), "schwarz_p");
        // A number implies its phase, and a phase the cell does not offer is refused by name.
        let node = lattice_with_keys(default_lattice(), &args(&[("density", Value::Float(0.5))]), &offered, &seed, &nobody).unwrap();
        assert!(matches!(node, Node::Lattice { phase: LatticePhase::Solid { density: ParamRef::Literal(d) }, .. } if d == 0.5));
        let refused = lattice_with_keys(default_lattice(), &args(&[("phase", s("strut"))]), &offered, &seed, &nobody).unwrap_err();
        assert!(refused.contains("has no strut phase"), "{refused}");
        let refused = lattice_with_keys(default_lattice(), &args(&[("phase", s("solid")), ("thickness", Value::Int(1))]), &offered, &seed, &nobody).unwrap_err();
        assert!(refused.contains("does not take"), "{refused}");
        let refused = lattice_with_keys(default_lattice(), &args(&[("cell", s("gyroids"))]), &offered, &seed, &nobody).unwrap_err();
        assert!(refused.starts_with("no cell is called"), "{refused}");
        // The map's own numbers go with their map.
        let node = lattice_with_keys(default_lattice(), &args(&[("map", s("cylindrical")), ("cell_radius", Value::Int(5)), ("cell_height", Value::Int(6)), ("arc", Value::Int(12))]), &offered, &seed, &nobody).unwrap();
        assert!(matches!(node, Node::Lattice { map: CellMap::Cylindrical { arc_count: 12, .. }, .. }));
        assert!(lattice_with_keys(default_lattice(), &args(&[("arc", Value::Int(12))]), &offered, &seed, &nobody).is_err());
        assert!(lattice_with_keys(default_lattice(), &args(&[("map", s("cylindrical")), ("arc", Value::Float(2.5))]), &offered, &seed, &nobody).is_err());
        // A known key the chosen map, cell or phase does not read is refused, never dropped.
        let refused = lattice_with_keys(default_lattice(), &args(&[("map", s("cylindrical")), ("cell_size", Value::Int(5))]), &offered, &seed, &nobody).unwrap_err();
        assert!(refused.contains("belongs to a rectangular map"), "{refused}");
        let refused = lattice_with_keys(default_lattice(), &args(&[("map", s("spherical")), ("arc", Value::Int(5))]), &offered, &seed, &nobody).unwrap_err();
        assert!(refused.contains("belongs to a cylindrical map"), "{refused}");
        let refused = lattice_with_keys(default_lattice(), &args(&[("cell", s("gyroid")), ("seed", Value::Int(3))]), &offered, &seed, &nobody).unwrap_err();
        assert!(refused.contains("has no seed grid"), "{refused}");
        let refused = lattice_with_keys(default_lattice(), &args(&[("thickness", Value::Int(1)), ("radius", Value::Float(0.5))]), &offered, &seed, &nobody).unwrap_err();
        assert!(refused.contains("one phase's numbers at a time"), "{refused}");
        let node = lattice_with_keys(default_lattice(), &args(&[("cell", s("voronoi_foam")), ("seed", Value::Int(3)), ("jitter", Value::Float(0.5))]), &offered, &seed, &nobody).unwrap();
        assert!(matches!(node, Node::Lattice { cell: LatticeCell::VoronoiFoam { seed: 3, jitter: ParamRef::Literal(j) }, .. } if j == 0.5));
    }

    #[test]
    fn a_lattice_phase_s_own_number_may_be_driven_and_nothing_else_may() {
        let block = NodeId::new();
        let lookup = |name: &str| if name == "r" { Lookup::Block(block) } else { Lookup::Nothing };
        let offered = |_: &LatticeCell| -> &'static [fieldform_scene::Banding] { &[fieldform_scene::Banding::Sheet] };
        let node = lattice_with_keys(default_lattice(), &args(&[("thickness", s("r"))]), &offered, &seed, &lookup).unwrap();
        assert!(matches!(node, Node::Lattice { phase: LatticePhase::Sheet { thickness: ParamRef::Field(f), .. }, .. } if f == block));
        assert!(lattice_with_keys(default_lattice(), &args(&[("cell_size", s("r"))]), &offered, &seed, &lookup).is_err());
        assert!(lattice_with_keys(default_lattice(), &args(&[("offset", s("r"))]), &offered, &seed, &lookup).is_err());
    }

    // ----------------------------------------------------------------------------- ramp

    #[test]
    fn a_ramp_s_ranges_are_pairs_and_a_bad_one_is_refused_not_dropped() {
        let ramp = Node::Ramp { input: Operand::empty(), in_min: ParamRef::Literal(0.0), in_max: ParamRef::Literal(1.0), out_min: ParamRef::Literal(0.0), out_max: ParamRef::Literal(1.0), shape: RampShape::Linear };
        let node = ramp_with_keys(ramp.clone(), &args(&[("in", s("0,60")), ("out", s("2,5")), ("shape", s("G2"))]), &nobody).unwrap();
        let Node::Ramp { in_max, out_min, shape, .. } = node else { panic!("a ramp") };
        assert_eq!(in_max, ParamRef::Literal(60.0));
        assert_eq!(out_min, ParamRef::Literal(2.0));
        assert_eq!(shape, RampShape::Smoother);
        assert!(ramp_with_keys(ramp.clone(), &args(&[("in", Value::Int(60))]), &nobody).is_err(), "half a range");
        assert!(ramp_with_keys(ramp.clone(), &args(&[("shape", s("G5"))]), &nobody).is_err());
        assert!(ramp_with_keys(ramp.clone(), &args(&[("input", s("nothing"))]), &nobody).is_err());
        // `input=none` is the empty slot said out loud, which is what the ramp is born with: the
        // one word every operand key takes, so a line can leave a slot waiting on purpose.
        let read = NodeId::new();
        let names = |name: &str| if name == "plane" { Lookup::Block(read) } else { Lookup::Nothing };
        let filled = ramp_with_keys(ramp, &args(&[("input", s("plane"))]), &names).unwrap();
        assert!(matches!(&filled, Node::Ramp { input, .. } if input.source() == Some(read)));
        let emptied = ramp_with_keys(filled, &args(&[("input", s("none"))]), &names).unwrap();
        assert!(matches!(&emptied, Node::Ramp { input, .. } if input.is_empty()));
    }

    // ------------------------------------------------------------- operands and fillets

    #[test]
    fn a_boolean_names_its_operands_and_a_fillet_joins_exactly_two() {
        let (a, b, c) = (NodeId::new(), NodeId::new(), NodeId::new());
        let lookup = |name: &str| match name {
            "box" => Lookup::Block(a),
            "ball" => Lookup::Block(b),
            "rod" => Lookup::Block(c),
            _ => Lookup::Nothing,
        };
        let asked = boolean_asked(&args(&[("u", Value::Bool(true)), ("in", s("box,ball")), ("r", Value::Int(2))]), &lookup).unwrap();
        assert_eq!(asked.op, Some(BooleanOp::Union));
        assert_eq!(asked.inputs, Some(vec![Operand::filled(a), Operand::filled(b)]));
        assert_eq!(asked.fillet, Some(Fillet { radius: ParamRef::Literal(2.0), continuity: Continuity::G1 }));
        let asked = boolean_asked(&args(&[("s", Value::Bool(true)), ("in", s("box,ball,rod"))]), &lookup).unwrap();
        assert_eq!(asked.inputs.map(|i| i.len()), Some(3));
        assert_eq!(asked.fillet, None);
        let refused = boolean_asked(&args(&[("in", s("box,ball,rod")), ("r", Value::Int(2))]), &lookup).unwrap_err();
        assert_eq!(refused, "a fillet joins two bodies; union the rest first");
        assert!(boolean_asked(&args(&[("in", s("box,nothing"))]), &lookup).unwrap_err().starts_with("no block is called"));
        assert!(boolean_asked(&args(&[("in", s("box"))]), &lookup).is_err(), "one operand is not a boolean");
        // `in=none` is both slots waiting -- the block a script makes before it has the bodies.
        assert_eq!(
            boolean_asked(&args(&[("in", s("none"))]), &lookup).unwrap().inputs,
            Some(vec![Operand::empty(), Operand::empty()]),
            "a boolean is a pair, so the word empties the pair rather than making an arity of one"
        );
        assert_eq!(
            boolean_asked(&args(&[("in", s("box,none"))]), &lookup).unwrap().inputs,
            Some(vec![Operand::filled(a), Operand::empty()]),
            "and half a join is half a join"
        );
        // A profile solid's rims, by their field names: nothing said keeps them, a reach alone
        // rounds at the shape they had, a shape alone reshapes a rim that has a reach, and a
        // shape on a hard rim is refused rather than set where it would do nothing.
        use fieldform_scene::Fillet;
        let none = rims_asked("extrude", &args(&[("depth", Value::Int(3))]), fieldform_scene::Rims::hard()).unwrap();
        assert_eq!(none, None);
        let round = rims_asked("extrude", &args(&[("rims", Value::Int(2))]), fieldform_scene::Rims::hard()).unwrap().unwrap();
        assert_eq!((round.radius, round.continuity), (ParamRef::Literal(2.0), Continuity::G1));
        let bevel = rims_asked("extrude", &args(&[("rims_continuity", s("chamfer"))]), round).unwrap().unwrap();
        assert_eq!(bevel.continuity, Continuity::Chamfer);
        assert!(rims_asked("extrude", &args(&[("rims_continuity", s("g2"))]), fieldform_scene::Rims::hard()).is_err());
        assert!(rims_asked("extrude", &args(&[("rims", Value::Int(-1))]), fieldform_scene::Rims::hard()).is_err());
        // Which ends, by the words the panel uses.
        let end = rims_asked("extrude", &args(&[("rims_at", s("end"))]), round).unwrap().unwrap();
        assert_eq!((end.at, end.radius), (fieldform_scene::RimEnds::End, round.radius));
        let start = rims_asked("sweep", &args(&[("rims", Value::Int(2)), ("rims_at", s("Start"))]), fieldform_scene::Rims::hard());
        assert_eq!(start.unwrap().unwrap().at, fieldform_scene::RimEnds::Start);
        assert!(rims_asked("extrude", &args(&[("rims_at", s("end"))]), fieldform_scene::Rims::hard()).is_err(), "hard has no ends to choose");
        assert!(rims_asked("loft", &args(&[("rims_at", s("middle"))]), round).is_err(), "a word it does not know");
        let g2 = boolean_asked(&args(&[("in", s("box,ball")), ("r", Value::Int(2)), ("g", Value::Int(2))]), &lookup).unwrap();
        assert_eq!(g2.fillet.map(|f| f.continuity), Some(Continuity::G2));
        // `g=chamfer` and `g=c` are the bevel, under the same key the field is written to.
        for word in ["chamfer", "c", "Chamfer"] {
            let bevel = boolean_asked(&args(&[("in", s("box,ball")), ("r", Value::Int(2)), ("g", s(word))]), &lookup).unwrap();
            assert_eq!(bevel.fillet.map(|f| f.continuity), Some(Continuity::Chamfer), "g={word}");
        }
        assert!(boolean_asked(&args(&[("in", s("box,ball")), ("r", Value::Int(2)), ("g", s("bevel"))]), &lookup)
            .unwrap_err()
            .contains("1, 2 or chamfer"));
        assert!(boolean_asked(&args(&[("op", s("xor"))]), &lookup).is_err());
        // A block's name for the radius is a ramp driving it (`round-chamfer-spec.md` §9.1).
        let ramp = NodeId::new();
        let graded = |name: &str| if name == "grade" { Lookup::Block(ramp) } else { lookup(name) };
        let driven = boolean_asked(&args(&[("in", s("box,ball")), ("r", s("grade"))]), &graded).unwrap();
        assert!(
            matches!(driven.fillet, Some(Fillet { radius: ParamRef::Field(f), .. }) if f == ramp),
            "a blend's radius may be driven"
        );
        // `r=0` is a plain boolean, as the instructions say.
        assert_eq!(boolean_asked(&args(&[("r", Value::Int(0))]), &lookup).unwrap().fillet, None);
    }

    // --------------------------------------------------------------------- modifiers (A5)

    #[test]
    fn modifiers_read_their_numbers_by_their_own_names() {
        let id = NodeId::new();
        let twist = Node::Twist { input: Operand::filled(id), degrees_per_mm: ParamRef::Literal(1.0), frame: Frame::default() };
        let node = twist_with_keys(twist.clone(), &args(&[("rate", Value::Float(2.5)), ("axis", s("y")), ("origin", s("0,0,5"))]), &nobody).unwrap();
        let Node::Twist { degrees_per_mm, frame: Frame::At { axis, origin }, .. } = node else { panic!("a twist") };
        assert_eq!(degrees_per_mm, ParamRef::Literal(2.5));
        assert_eq!(axis, [ParamRef::Literal(0.0), ParamRef::Literal(1.0), ParamRef::Literal(0.0)]);
        assert_eq!(origin[2], ParamRef::Literal(5.0));
        let node = twist_with_keys(twist, &args(&[("degrees_per_mm", Value::Int(3))]), &nobody).unwrap();
        assert!(matches!(node, Node::Twist { degrees_per_mm: ParamRef::Literal(r), .. } if r == 3.0), "the field's own name works too");

        let mirror = Node::Mirror { input: Operand::filled(id), frame: Frame::default(), join: false };
        let node = mirror_with_keys(mirror, &args(&[("axis", s("x")), ("join", Value::Bool(true))]), &nobody).unwrap();
        assert!(matches!(node, Node::Mirror { join: true, frame: Frame::At { .. }, .. }));

        let shell = Node::Shell { input: Operand::filled(id), thickness: ParamRef::Literal(1.0) };
        let node = shell_with_keys(shell, &args(&[("thickness", Value::Int(-2))]), &nobody).unwrap();
        assert!(matches!(node, Node::Shell { thickness: ParamRef::Literal(t), .. } if t == 2.0));

        let offset = Node::Offset { input: Operand::filled(id), distance: ParamRef::Literal(1.0) };
        let driver = NodeId::new();
        let lookup = |name: &str| if name == "r" { Lookup::Block(driver) } else { Lookup::Nothing };
        let node = offset_with_keys(offset, &args(&[("distance", s("r"))]), &lookup).unwrap();
        assert!(matches!(node, Node::Offset { distance: ParamRef::Field(f), .. } if f == driver), "an offset may be driven");

        let attract = Node::Attract { input: Operand::filled(id), toward: Attractor::Point(id), strength: ParamRef::Literal(0.5), falloff: ParamRef::Literal(1.0) };
        let node = attract_with_keys(attract, &args(&[("strength", Value::Float(0.8)), ("falloff", Value::Int(12))]), &nobody).unwrap();
        assert!(matches!(node, Node::Attract { falloff: ParamRef::Literal(f), .. } if f == 12.0));
    }

    #[test]
    fn a_pattern_reads_a_row_or_a_ring_and_refuses_the_other_s_keys() {
        let id = NodeId::new();
        let row = Node::Pattern {
            input: Operand::filled(id),
            count: ParamRef::Literal(3.0),
            kind: PatternKind::Linear { step: [ParamRef::Literal(50.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)] },
            join: false,
            fillet: Fillet::hard(),
            options: PatternOptions::default(),
        };
        let node = pattern_with_keys(row.clone(), &args(&[("count", Value::Int(5)), ("step", s("0,20,0")), ("r", Value::Int(1))]), &nobody).unwrap();
        let Node::Pattern { count, kind: PatternKind::Linear { step }, join, fillet, .. } = node else { panic!("a row") };
        assert_eq!(count, ParamRef::Literal(5.0));
        assert_eq!(step[1], ParamRef::Literal(20.0));
        assert!(join, "a radius joins the copies");
        assert_eq!(fillet.radius, ParamRef::Literal(1.0));
        let node = pattern_with_keys(row.clone(), &args(&[("step", Value::Int(30))]), &nobody).unwrap();
        assert!(matches!(node, Node::Pattern { kind: PatternKind::Linear { step: [ParamRef::Literal(x), _, _] }, .. } if x == 30.0), "one number steps along the first axis");
        let node = pattern_with_keys(row.clone(), &args(&[("kind", s("radial")), ("axis", s("z")), ("count", Value::Int(8))]), &nobody).unwrap();
        assert!(matches!(node, Node::Pattern { kind: PatternKind::Radial { .. }, .. }));
        assert!(pattern_with_keys(row.clone(), &args(&[("axis", s("z"))]), &nobody).is_err(), "a row has no axis");
        assert!(pattern_with_keys(row.clone(), &args(&[("kind", s("radial")), ("extent", Value::Bool(true))]), &nobody).is_err(), "a ring has no extent");
        let node = pattern_with_keys(row, &args(&[("rows", Value::Int(3)), ("step2", s("0,0,10"))]), &nobody).unwrap();
        let Node::Pattern { options, .. } = node else { panic!("a grid") };
        assert_eq!(options.second.map(|s| s.step[2].clone()), Some(ParamRef::Literal(10.0)));
    }

    // ---------------------------------------------------------------------- placing (A4)

    /// `placed` over `existing`, as `apply_placing` hands it the outermost placement: the node
    /// and whether it edited that placement (`Some(id)`) or made a new one around it (`None`).
    fn place(
        placing: Placing,
        body: NodeId,
        existing: Option<(NodeId, &Node)>,
        pairs: &[(&str, Value)],
        lookup: &dyn Fn(&str) -> Lookup,
    ) -> (Node, Option<NodeId>) {
        placed(placing, body, existing, &args(pairs), lookup).unwrap()
    }

    /// Which block a placement reads, as its slot answers -- `None` for one left waiting, which
    /// `placed` never makes and a test should say rather than unwrap.
    fn input_of(node: &Node) -> Option<NodeId> {
        let Node::Transform { input, .. } = node else { panic!("a placement") };
        input.source()
    }

    #[test]
    fn a_move_folds_into_a_placement_and_an_edit_that_does_not_commute_wraps_it() {
        let body = NodeId::new();
        let placement = NodeId::new();
        // The first move wraps the body.
        let (first, edited) = place(Placing::Move, body, None, &[("by", s("10,0,0"))], &nobody);
        assert_eq!(edited, None);
        assert_eq!(input_of(&first), Some(body));
        assert!(matches!(&first, Node::Transform { translation, .. } if translation[0] == ParamRef::Literal(10.0)));
        // A second move adds to that placement: ten moves, one block.
        let (again, edited) = place(Placing::Move, body, Some((placement, &first)), &[("by", s("5,1,0"))], &nobody);
        assert_eq!(edited, Some(placement), "a move always folds in");
        assert_eq!(input_of(&again), Some(body));
        assert!(matches!(&again, Node::Transform { translation, .. } if translation[0] == ParamRef::Literal(15.0) && translation[1] == ParamRef::Literal(1.0)));
        // A turn after a move does not commute with it -- the kernel turns first -- so it wraps
        // the placement, never the body again.
        let (turned, edited) = place(Placing::Turn, body, Some((placement, &again)), &[("by", s("0,0,45"))], &nobody);
        assert_eq!(edited, None, "turn after move wraps");
        assert_eq!(input_of(&turned), Some(placement), "around the placement, not the body");
        assert!(matches!(&turned, Node::Transform { rotation, .. } if rotation[2] == ParamRef::Literal(45.0)));
        // A move after that turn folds into the turn's block: a translation sits outside.
        let outer = NodeId::new();
        let (moved_after, edited) = place(Placing::Move, body, Some((outer, &turned)), &[("by", s("0,0,20"))], &nobody);
        assert_eq!(edited, Some(outer));
        assert!(matches!(&moved_after, Node::Transform { translation, rotation, .. } if translation[2] == ParamRef::Literal(20.0) && rotation[2] == ParamRef::Literal(45.0)));
        // Two turns about one axis from one pivot add; a turn about another axis wraps.
        let (z, _) = place(Placing::Turn, body, None, &[("by", s("0,0,90"))], &nobody);
        let (z2, edited) = place(Placing::Turn, body, Some((placement, &z)), &[("by", s("0,0,45"))], &nobody);
        assert_eq!(edited, Some(placement));
        assert!(matches!(&z2, Node::Transform { rotation, .. } if rotation[2] == ParamRef::Literal(135.0)));
        let (_, edited) = place(Placing::Turn, body, Some((placement, &z)), &[("by", s("90,0,0"))], &nobody);
        assert_eq!(edited, None, "X after Z is not X then Z");
        // A pivot asked for on a block that has turned about another one wraps; on a block that
        // has not used its pivot yet, it is taken.
        let (_, edited) = place(Placing::Turn, body, Some((placement, &z)), &[("by", s("0,0,90")), ("about", s("100,0,0"))], &nobody);
        assert_eq!(edited, None, "a second pivot wraps");
        let (_, edited) = place(Placing::Turn, body, Some((placement, &again)), &[("by", s("0,0,45")), ("about", s("1,1,1"))], &nobody);
        assert_eq!(edited, None, "and so does a turn onto a move");
        let (fresh_pivot, _) = place(Placing::Scale, body, None, &[("by", Value::Int(2)), ("about", s("1,1,1"))], &nobody);
        assert!(matches!(&fresh_pivot, Node::Transform { about: Place::At(a), scale: ParamRef::Literal(s), .. } if a[0] == ParamRef::Literal(1.0) && *s == 2.0));
        let (scaled_again, edited) = place(Placing::Scale, body, Some((placement, &fresh_pivot)), &[("by", Value::Int(3)), ("about", s("1,1,1"))], &nobody);
        assert_eq!(edited, Some(placement), "the same pivot folds");
        assert!(matches!(&scaled_again, Node::Transform { scale: ParamRef::Literal(s), .. } if *s == 6.0));
        // A scale after a move wraps: scaling the moved body about the origin moves it further.
        let (_, edited) = place(Placing::Scale, body, Some((placement, &again)), &[("by", Value::Int(2))], &nobody);
        assert_eq!(edited, None);
        assert!(placed(Placing::Scale, body, None, &args(&[("by", s("1,2,1"))]), &nobody).is_err(), "three numbers wait for item 69");
        assert!(placed(Placing::Move, body, None, &Args::new(), &nobody).is_err(), "a move says by how much");
        assert!(placed(Placing::Move, body, None, &args(&[("about", s("0,0,0")), ("by", s("1,0,0"))]), &nobody).is_err());
    }

    #[test]
    fn a_parameter_amount_never_overwrites_a_number_a_placement_holds() {
        let body = NodeId::new();
        let placement = NodeId::new();
        let param = ParamId::new();
        let lookup = |name: &str| if name == "h" { Lookup::Parameter(param) } else { Lookup::Nothing };
        // Onto the identity it lands; onto a number it wraps.
        let (first, _) = place(Placing::Move, body, None, &[("by", s("h,0,0"))], &lookup);
        assert!(matches!(&first, Node::Transform { translation, .. } if translation[0] == ParamRef::Named(param)));
        let (moved, _) = place(Placing::Move, body, None, &[("by", s("10,0,0"))], &nobody);
        let (over, edited) = place(Placing::Move, body, Some((placement, &moved)), &[("by", s("h,0,0"))], &lookup);
        assert_eq!(edited, None, "10 + h needs an expression, so the placement is wrapped");
        assert_eq!(input_of(&over), Some(placement));
        assert!(matches!(&over, Node::Transform { translation, .. } if translation[0] == ParamRef::Named(param)));
        // And a placement holding a parameter is not rewritten by a number: a new one wraps it.
        let driven = Node::Transform { input: Operand::filled(body), translation: [ParamRef::Named(param), ParamRef::Literal(0.0), ParamRef::Literal(0.0)], rotation: [ParamRef::Literal(0.0); 3], scale: ParamRef::Literal(1.0), about: Place::origin() };
        let (wrapped, edited) = place(Placing::Move, body, Some((placement, &driven)), &[("by", s("1,0,0"))], &nobody);
        assert_eq!(edited, None);
        assert_eq!(input_of(&wrapped), Some(placement));
        // The other two components of a move onto a parameter slot still add where they can.
        let (mixed, edited) = place(Placing::Move, body, Some((placement, &driven)), &[("by", s("0,5,0"))], &nobody);
        assert_eq!(edited, Some(placement), "nought onto a parameter changes nothing, five onto nought adds");
        assert!(matches!(&mixed, Node::Transform { translation, .. } if translation[0] == ParamRef::Named(param) && translation[1] == ParamRef::Literal(5.0)));
    }

    #[test]
    fn every_sample_is_a_node_the_scene_still_reads() {
        // Every sample round-trips through the file's own reader, so the drift test's field
        // names are the reader's field names and not an artefact of serialization alone. Node
        // by node, since the samples name operands that are in no graph.
        for node in samples() {
            let value = serde_json::to_value(&node).expect("serializes");
            let back: Node = serde_json::from_value(value.clone()).expect("reads back");
            assert!(!matches!(back, Node::Unknown(_)), "the reader does not know {value}");
            assert_eq!(serde_json::to_value(&back).expect("serializes again"), value, "and writes it back the same");
        }
    }
}
