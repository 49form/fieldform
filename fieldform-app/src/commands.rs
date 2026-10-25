// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The command registry: every user-visible action, once (`interaction-spec.md` §2).
//!
//! **One table, read six ways.** The menu bar, the console's words, the default keymap, the help
//! window's list of keys, the palette and the marking menu are all generated from [`COMMANDS`].
//! Before this there were four hand-kept lists of one set -- the menu, `OTHER_CONSOLE_COMMANDS`,
//! the keymap's defaults and the shortcut window -- and the plane and the point once landed in
//! the menu alone, reachable by mouse and not by keyboard. A second list is how that happens, so
//! there is no second list.
//!
//! **The id is the operation name `dispatch` already knows** (`create.extrude`), because the
//! console, scripts and the keymap have used those names since before there were menus. An
//! entry that runs an operation with fixed arguments -- Display quality › Fast is `display
//! quality=fast` -- has an id of its own and the operation it runs beside it, so a menu item and
//! a typed line still cannot mean two things.
//!
//! **What a command reads is not repeated here.** The keys each verb reads live in
//! `console_keys::VERBS`, which is held against the blocks themselves; a session command's
//! [`Param`]s name keys from that table and the tests refuse one that is not in it.
//!
//! Free of the `App` so the rules are held by tests: what the app knows about the moment is
//! handed in as a [`Ctx`].

use eframe::egui;
use fieldform_ops::{Args, Keymap, Modifiers, Shortcut};
use std::sync::LazyLock;

/// Whether a command finishes when it runs or leaves something pending (§2.1, §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Runs and is done: delete, frame, undo.
    Immediate,
    /// Makes a block that stays pending until it is kept or taken back (§3).
    Session,
    /// A cage tool: armed until Esc or another command, each click its own operation and undo
    /// step (R7, `app/cage_tools.rs`). Typed with its arguments it runs at once instead.
    Tool,
}

/// What a parameter's number is measured in, for the tool panel's unit column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// A length, in the document's canonical millimetres.
    Mm,
    Degrees,
    /// A whole number of things.
    Count,
    /// A number with no unit: a strength, a fraction, a seed.
    Plain,
    /// A word or a name: a cell, a side, a block.
    Word,
}

// Read by the tool panel and the palette, which later steps of §12 build; the tests read it now.
#[cfg_attr(not(test), allow(dead_code))]
impl Unit {
    /// What the panel prints after the value.
    pub fn suffix(self) -> &'static str {
        match self {
            Unit::Mm => "mm",
            Unit::Degrees => "\u{b0}",
            Unit::Count | Unit::Plain | Unit::Word => "",
        }
    }
}

/// One number a session command shows in its tool panel, in the order shown.
///
/// `key` is a word the command's verb reads in `console_keys::VERBS`, so what the panel edits,
/// what the console types and what the echo writes are the same word for the same field (D9).
#[derive(Clone, Copy, Debug)]
// Read by the tool panel and the palette, which later steps of §12 build; the tests read it now.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Param {
    pub key: &'static str,
    pub unit: Unit,
    /// Whether the viewport offers a handle that drags this number (§4.5).
    pub handle: bool,
}

const fn p(key: &'static str, unit: Unit) -> Param {
    Param { key, unit, handle: false }
}

/// What a command's input picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accepts {
    Body,
    Curve,
    /// A closed loop of curves, or one curve, as an extrude takes.
    Profile,
}

/// A selection the command consumes, or picks when the selection does not hold one (§3.2).
#[derive(Clone, Copy, Debug)]
// Read by the tool panel and the palette, which later steps of §12 build; the tests read it now.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Input {
    /// The key the console names it by.
    pub key: &'static str,
    pub accepts: Accepts,
    /// How many it needs; more are taken where the key reads a list.
    pub min: u8,
}

const fn input(key: &'static str, accepts: Accepts, min: u8) -> Input {
    Input { key, accepts, min }
}

/// A key and its modifiers, as the keymap names them.
///
/// **Written as text in the table: `Ctrl+Shift+Z`.** That is how a menu prints it and how a
/// person writes one in the bindings file (§9.2), so the table and the file are one notation and
/// [`Chord::parse`] reads both.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Chord {
    /// egui's name for the key: `Z`, `Tab`, `Slash`, `Delete`, `Space`.
    pub key: String,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl Chord {
    /// `Ctrl+Shift+Z`, `Alt+H`, `Space`, `/`. Modifiers in any order and any case; the key last.
    pub fn parse(text: &str) -> Result<Chord, String> {
        let mut chord = Chord { key: String::new(), ctrl: false, shift: false, alt: false };
        let parts: Vec<&str> = text.split('+').map(str::trim).collect();
        let Some((key, mods)) = parts.split_last() else {
            return Err(format!("{text:?} names no key"));
        };
        for m in mods {
            match m.to_ascii_lowercase().as_str() {
                "ctrl" | "control" | "cmd" => chord.ctrl = true,
                "shift" => chord.shift = true,
                "alt" | "option" => chord.alt = true,
                other => return Err(format!("{other:?} in {text:?} is not Ctrl, Shift or Alt")),
            }
        }
        let written = match *key {
            "" => return Err(format!("{text:?} names no key")),
            "Del" | "del" => "Delete".to_string(),
            k if k.len() == 1 => k.to_ascii_uppercase(),
            k => {
                let mut c = k.chars();
                c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
            }
        };
        // egui's own name for the key, which is what a pressed key is looked up by: `/` is
        // `Slash`, `5` is `5`.
        let Some(found) = egui::Key::from_name(&written) else {
            return Err(format!("{key:?} in {text:?} is not a key"));
        };
        chord.key = found.name().to_string();
        Ok(chord)
    }

    pub fn shortcut(&self) -> Shortcut {
        Shortcut::new(&self.key, Modifiers { ctrl: self.ctrl, shift: self.shift, alt: self.alt })
    }

    /// How a menu, the palette and the help window print it: `Ctrl+Shift+Z`.
    pub fn label(&self) -> String {
        let key = match self.key.as_str() {
            "Slash" => "/",
            "Delete" => "Del",
            other => other,
        };
        let mut out = String::new();
        if self.ctrl {
            out.push_str("Ctrl+");
        }
        if self.alt {
            out.push_str("Alt+");
        }
        if self.shift {
            out.push_str("Shift+");
        }
        out.push_str(key);
        out
    }
}

/// What the app knows about the moment, for the questions a command asks before it is offered.
///
/// Filled once a frame by `App::command_context`. Plain data, so a test can build any state.
#[derive(Clone, Debug, Default)]
pub struct Ctx {
    /// The names on top of the history, for "Undo Sphere" and for whether there is any.
    pub undo: Option<String>,
    pub redo: Option<String>,
    /// A body, a curve or a block is selected.
    pub selected: bool,
    pub clipboard: bool,
    pub clipboard_curve: bool,
    /// The part's bodies, and what a two-body tool can take.
    pub bodies: usize,
    pub pair_candidates: usize,
    /// The viewport draws the field, so a surface can be clicked.
    pub field_drawn: bool,
    /// The cage tools are editing the cage rather than a surface.
    pub cage_control: bool,
    pub section_on: bool,
    /// Display quality allows occlusion at all.
    pub occlusion_possible: bool,
    pub control_points: bool,
    pub view_cube: bool,
    /// A pending block shows the floating tool panel, rather than only its rows in the list.
    pub floating_panel: bool,
    pub idle_occlusion: bool,
    pub draft_on: bool,
    pub grid_on: bool,
    /// `polygon`, `field` or `overlay`.
    pub view_mode: &'static str,
    /// The display quality's label, lowercased.
    pub quality: String,
}

/// Whether a command is on offer now, and if not, why not.
pub type Available = fn(&Ctx) -> Result<(), &'static str>;

/// One user-visible action.
pub struct Command {
    /// Stable, unique: the operation name, or the operation and its fixed arguments.
    pub id: &'static str,
    /// What `dispatch` runs.
    pub op: &'static str,
    /// Arguments the entry always runs with, as typed: `quality=fast`.
    pub args: &'static [(&'static str, &'static str)],
    /// The canonical console verb. Several entries may share one when they share an operation.
    pub console_name: &'static str,
    /// Other spellings the console accepts for the same verb (`tess`, `tesselate`).
    pub console_also: &'static [&'static str],
    pub display_name: &'static str,
    /// Extra words the palette matches (§5.2).
    pub aliases: &'static [&'static str],
    /// Where it sits in the menu bar: `["Create", "Sweep"]`. Empty for a command reached only
    /// by typing, the palette or a key.
    pub category: &'static [&'static str],
    /// Separator groups inside its menu: a change of group draws a line.
    pub group: u8,
    pub hint: &'static str,
    pub params: &'static [Param],
    pub inputs: &'static [Input],
    pub context: Available,
    /// For a setting: whether it is on, which the menu shows as a tick.
    pub checked: Option<fn(&Ctx) -> bool>,
    /// Its keys out of the box, as text (`Ctrl+Shift+Z`); a person's bindings file overrides them.
    pub default_bindings: &'static [&'static str],
    pub kind: Kind,
    /// Only in a build with this cargo feature on.
    pub feature: Option<&'static str>,
}

fn always(_: &Ctx) -> Result<(), &'static str> {
    Ok(())
}
fn has_undo(c: &Ctx) -> Result<(), &'static str> {
    c.undo.as_ref().map(|_| ()).ok_or("Nothing to undo")
}
fn has_redo(c: &Ctx) -> Result<(), &'static str> {
    c.redo.as_ref().map(|_| ()).ok_or("Nothing to redo")
}
fn something_selected(c: &Ctx) -> Result<(), &'static str> {
    if c.selected { Ok(()) } else { Err("Select something first") }
}
fn has_clipboard(c: &Ctx) -> Result<(), &'static str> {
    if c.clipboard { Ok(()) } else { Err("Copy something first") }
}
fn has_clipboard_curve(c: &Ctx) -> Result<(), &'static str> {
    if c.clipboard_curve { Ok(()) } else { Err("Copy a curve first") }
}
fn a_body(c: &Ctx) -> Result<(), &'static str> {
    if c.bodies > 0 { Ok(()) } else { Err("Needs a body") }
}
fn two_bodies(c: &Ctx) -> Result<(), &'static str> {
    if c.pair_candidates >= 2 { Ok(()) } else { Err("Needs two bodies") }
}
fn field_drawn(c: &Ctx) -> Result<(), &'static str> {
    if c.field_drawn {
        Ok(())
    } else {
        Err("Polygon view draws the cage, not the surface — Tab for Field")
    }
}
fn cage_control(c: &Ctx) -> Result<(), &'static str> {
    if c.cage_control { Ok(()) } else { Err("Works on the cage; a surface is being edited") }
}
fn section_on(c: &Ctx) -> Result<(), &'static str> {
    if c.section_on { Ok(()) } else { Err("Turn Section on first") }
}
fn occlusion_possible(c: &Ctx) -> Result<(), &'static str> {
    if c.occlusion_possible {
        Ok(())
    } else {
        Err("Off while Display quality is Fast, which never traces occlusion or shadows. Choose \
             Balanced or Full to have this back.")
    }
}

const fn cmd(id: &'static str, op: &'static str, word: &'static str, name: &'static str) -> Command {
    Command {
        id,
        op,
        args: &[],
        console_name: word,
        console_also: &[],
        display_name: name,
        aliases: &[],
        category: &[],
        group: 0,
        hint: "",
        params: &[],
        inputs: &[],
        context: always,
        checked: None,
        default_bindings: &[],
        kind: Kind::Immediate,
        feature: None,
    }
}

/// An entry whose id is its operation, which is all of them but the fixed-argument variants.
const fn op(op: &'static str, word: &'static str, name: &'static str) -> Command {
    cmd(op, op, word, name)
}

impl Command {
    const fn menu(mut self, category: &'static [&'static str], group: u8) -> Self {
        self.category = category;
        self.group = group;
        self
    }
    const fn hint(mut self, hint: &'static str) -> Self {
        self.hint = hint;
        self
    }
    const fn also(mut self, words: &'static [&'static str]) -> Self {
        self.console_also = words;
        self
    }
    const fn aliases(mut self, words: &'static [&'static str]) -> Self {
        self.aliases = words;
        self
    }
    const fn args(mut self, args: &'static [(&'static str, &'static str)]) -> Self {
        self.args = args;
        self
    }
    const fn tool(mut self) -> Self {
        self.kind = Kind::Tool;
        self
    }

    const fn session(mut self, params: &'static [Param]) -> Self {
        self.kind = Kind::Session;
        self.params = params;
        self
    }
    const fn inputs(mut self, inputs: &'static [Input]) -> Self {
        self.inputs = inputs;
        self
    }
    const fn when(mut self, context: Available) -> Self {
        self.context = context;
        self
    }
    const fn checked(mut self, checked: fn(&Ctx) -> bool) -> Self {
        self.checked = Some(checked);
        self
    }
    const fn keys(mut self, chords: &'static [&'static str]) -> Self {
        self.default_bindings = chords;
        self
    }
    const fn feature(mut self, feature: &'static str) -> Self {
        self.feature = Some(feature);
        self
    }

    /// Whether this build has it.
    pub fn built(&self) -> bool {
        match self.feature {
            None => true,
            Some("march-stats") => cfg!(feature = "march-stats"),
            Some(_) => false,
        }
    }

    /// The fixed arguments, read by the console's own value grammar so a menu item's arguments
    /// are exactly what typing its console line would give.
    pub fn fixed_args(&self) -> Args {
        self.args
            .iter()
            .fold(Args::new(), |args, (k, v)| args.with(k, crate::console::parse_value(v)))
    }

    /// The line that runs it from the console: the verb and any fixed arguments.
    pub fn console_line(&self) -> String {
        let mut line = self.console_name.to_string();
        for (k, v) in self.args {
            line.push(' ');
            line.push_str(k);
            line.push('=');
            line.push_str(v);
        }
        line
    }

    /// Its default bindings, read. The table's own text is held to parse by a test, so a
    /// chord that does not is left out here rather than refusing to start.
    pub fn chords(&self) -> Vec<Chord> {
        self.default_bindings.iter().filter_map(|c| Chord::parse(c).ok()).collect()
    }

    /// Whether running this keeps a pending session first (§3.4). A command that only changes
    /// how the document is looked at, or opens a window or the palette, does not: framing the view
    /// or searching for the next command is not a reason to end the tool in hand.
    pub fn keeps_session(&self) -> bool {
        !(self.op.starts_with("view.") || self.op.starts_with("help.") || self.op.starts_with("palette.") || self.op.starts_with("console."))
    }

    /// The first default binding, printed, for the menu and the palette.
    pub fn binding_label(&self) -> Option<String> {
        self.chords().first().map(Chord::label)
    }
}

// ------------------------------------------------------------------------------- categories

const FILE: &[&str] = &["File"];
const EDIT: &[&str] = &["Edit"];
const VIEW: &[&str] = &["View"];
const VIEW_MODE: &[&str] = &["View", "Display mode"];
const VIEW_QUALITY: &[&str] = &["View", "Display quality"];
const CREATE: &[&str] = &["Create"];
const PRIMITIVES: &[&str] = &["Create", "Primitives"];
const SWEEP: &[&str] = &["Create", "Sweep"];
const MODIFY: &[&str] = &["Modify"];
/// The cage tools, in a menu of their own (the owner, 2026-10-24): what edits the cage's
/// topology, what grows a selection along it, and what checks it.
const SUBD: &[&str] = &["Sub-D"];
const HELP: &[&str] = &["Help"];

/// The top-level menus, left to right.
pub const MENUS: &[&str] = &["File", "Edit", "View", "Create", "Modify", "Sub-D", "Help"];

/// Which separator group each submenu sits in, inside its parent: the submenu's own entries
/// carry their groups *within* it.
const SUBMENUS: &[(&[&str], u8)] = &[
    (VIEW_MODE, 4),
    (VIEW_QUALITY, 5),
    (PRIMITIVES, 0),
    (SWEEP, 0),
];

// ---------------------------------------------------------------------------- the params

const POINT_P: &[Param] = &[p("at", Unit::Mm)];
const PLANE_P: &[Param] = &[p("origin", Unit::Mm), p("turn", Unit::Degrees), p("flipped", Unit::Word)];
const SPHERE_P: &[Param] = &[p("radius", Unit::Mm), p("centre", Unit::Mm)];
const CUBE_P: &[Param] = &[p("size", Unit::Mm), p("centre", Unit::Mm), p("turn", Unit::Degrees)];
const SKEWBOX_P: &[Param] = &[p("centre", Unit::Mm), p("edges", Unit::Mm)];
const CYLINDER_P: &[Param] = &[p("radius", Unit::Mm), p("from", Unit::Mm), p("to", Unit::Mm)];
const CONE_P: &[Param] = &[
    p("base_radius", Unit::Mm),
    p("top_radius", Unit::Mm),
    p("from", Unit::Mm),
    p("to", Unit::Mm),
];
const TORUS_P: &[Param] = &[
    p("major_radius", Unit::Mm),
    p("minor_radius", Unit::Mm),
    p("centre", Unit::Mm),
    p("turn", Unit::Degrees),
];
const CAPSULE_P: &[Param] = &[
    p("start_radius", Unit::Mm),
    p("end_radius", Unit::Mm),
    p("from", Unit::Mm),
    p("to", Unit::Mm),
];
const TPMS_P: &[Param] = &[
    p("cell", Unit::Word),
    p("map", Unit::Word),
    p("cell_size", Unit::Mm),
    p("phase", Unit::Word),
    p("thickness", Unit::Mm),
    p("density", Unit::Plain),
    p("origin", Unit::Mm),
];
const RAMP_P: &[Param] =
    &[p("input", Unit::Word), p("in", Unit::Mm), p("out", Unit::Plain), p("shape", Unit::Word)];
const BOOLEAN_P: &[Param] =
    &[p("in", Unit::Word), p("op", Unit::Word), p("radius", Unit::Mm), p("g", Unit::Count)];
const MORPH_P: &[Param] = &[p("a", Unit::Word), p("b", Unit::Word), p("t", Unit::Plain)];
const OFFSET_P: &[Param] = &[p("target", Unit::Word), p("distance", Unit::Mm)];
const SHELL_P: &[Param] = &[p("target", Unit::Word), p("thickness", Unit::Mm)];
const TWIST_P: &[Param] = &[
    p("target", Unit::Word),
    p("degrees_per_mm", Unit::Plain),
    p("axis", Unit::Word),
    p("origin", Unit::Mm),
];
const MIRROR_P: &[Param] =
    &[p("target", Unit::Word), p("axis", Unit::Word), p("origin", Unit::Mm), p("join", Unit::Word)];
const ATTRACT_P: &[Param] = &[
    p("target", Unit::Word),
    p("toward", Unit::Word),
    p("strength", Unit::Plain),
    p("falloff", Unit::Mm),
];
const PATTERN_P: &[Param] = &[
    p("target", Unit::Word),
    p("kind", Unit::Word),
    p("count", Unit::Count),
    p("step", Unit::Mm),
    p("axis", Unit::Word),
    p("origin", Unit::Mm),
    p("join", Unit::Word),
    p("radius", Unit::Mm),
];
const MOVE_P: &[Param] = &[p("target", Unit::Word), p("by", Unit::Mm)];
const TURN_P: &[Param] = &[p("target", Unit::Word), p("by", Unit::Degrees), p("about", Unit::Mm)];
const SCALE_P: &[Param] = &[p("target", Unit::Word), p("by", Unit::Plain), p("about", Unit::Mm)];
const EXTRUDE_P: &[Param] = &[
    p("profile", Unit::Word),
    Param { key: "height", unit: Unit::Mm, handle: true },
    p("start", Unit::Mm),
    p("draft", Unit::Degrees),
    p("symmetric", Unit::Word),
    p("height_2", Unit::Mm),
    p("draft_2", Unit::Degrees),
];
const REVOLVE_P: &[Param] = &[
    p("profile", Unit::Word),
    p("axis", Unit::Word),
    p("origin", Unit::Mm),
    p("angle", Unit::Degrees),
    p("symmetric", Unit::Word),
    p("angle_2", Unit::Degrees),
];
const SWEEP_P: &[Param] = &[
    p("profile", Unit::Word),
    p("spine", Unit::Word),
    p("section", Unit::Word),
    p("twist", Unit::Degrees),
    p("taper_scale", Unit::Plain),
];
const LOFT_P: &[Param] = &[p("sections", Unit::Word), p("spine", Unit::Word), p("smooth", Unit::Word)];
const PIPE_P: &[Param] = &[p("curve", Unit::Word), p("radius", Unit::Mm)];

const BODY: &[Input] = &[input("target", Accepts::Body, 1)];
const TWO_BODIES: &[Input] = &[input("in", Accepts::Body, 2)];
const MORPH_IN: &[Input] = &[input("a", Accepts::Body, 1), input("b", Accepts::Body, 1)];
const PROFILE: &[Input] = &[input("profile", Accepts::Profile, 1)];
const SWEEP_IN: &[Input] = &[input("profile", Accepts::Profile, 1), input("spine", Accepts::Curve, 1)];
const LOFT_IN: &[Input] = &[input("sections", Accepts::Profile, 2)];
const PIPE_IN: &[Input] = &[input("curve", Accepts::Curve, 1)];
const SPLIT_IN: &[Input] = &[input("body", Accepts::Body, 1)];

// ----------------------------------------------------------------------------- the table

/// Every command, in menu order where it has a menu.
pub static COMMANDS: LazyLock<Vec<Command>> = LazyLock::new(build);

#[rustfmt::skip]
fn build() -> Vec<Command> {
    use crate::display_quality::DisplayQuality as Q;
    vec![
        // ---------------------------------------------------------------------------- File
        op("file.new", "new", "New").menu(FILE, 0)
            .hint("An empty document. Asks first if this one has changes that are not saved."),
        op("file.open", "open", "Open...").menu(FILE, 0).keys(&["Ctrl+O"])
            .hint("Opens a document. `open file=\"C:/in/part.fform\"` opens it without the dialog."),
        op("file.save", "save", "Save").menu(FILE, 0).keys(&["Ctrl+S"])
            .hint("Writes the document over its file, or asks where if it has none."),
        op("file.save_as", "saveas", "Save As...").menu(FILE, 0).keys(&["Ctrl+Shift+S"])
            .hint("Writes the document to a new file. `saveas file=…` without the dialog."),
        op("file.export_stl", "export", "Export STL...").menu(FILE, 1)
            .hint("Meshes the part and writes it as STL. `export file=… detail=fine` without the dialog."),
        op("file.import", "import", "Import...").menu(FILE, 1)
            .hint("STEP, OBJ or IGES — the file says which, and its options are asked after."),
        op("file.exit", "exit", "Exit").menu(FILE, 2).also(&["quit"])
            .hint("Closes fieldform, asking first if there are unsaved changes."),

        // ---------------------------------------------------------------------------- Edit
        op("edit.undo", "undo", "Undo").menu(EDIT, 0).when(has_undo)
            .keys(&["Ctrl+Z"])
            .hint("Takes back the last change."),
        op("edit.redo", "redo", "Redo").menu(EDIT, 0).when(has_redo)
            .keys(&["Ctrl+Shift+Z", "Ctrl+Y"])
            .hint("Puts back what Undo took."),
        // §8: the last command kept or run, again, on the current selection and with the values
        // it was last kept with. Not the marking menu's centre, which the owner had emptied.
        op("edit.repeat", "repeat", "Repeat last").menu(EDIT, 0).keys(&["Shift+R"])
            .aliases(&["again"])
            .hint("Runs the last command again, on what is selected now, starting from the values \
                   it was last kept with."),
        // **A selected block counts as something.** `select_block` clears the body and the
        // curve selection, so Cut, Copy and Delete once went grey the moment a profile row was
        // picked -- while the handler would have taken it.
        op("edit.cut", "cut", "Cut").menu(EDIT, 1).when(something_selected)
            .keys(&["Ctrl+X"])
            .hint("Copies the selection and deletes it."),
        op("edit.copy", "copy", "Copy").menu(EDIT, 1).when(something_selected)
            .keys(&["Ctrl+C"])
            .hint("Copies the selection, to paste in place or under the cursor."),
        op("edit.paste", "paste", "Paste").menu(EDIT, 1).when(has_clipboard)
            .keys(&["Ctrl+V"])
            .hint("Puts down what was copied. Ctrl+V over the viewport lands a curve under the cursor."),
        op("edit.duplicate", "duplicate", "Duplicate").menu(EDIT, 1).when(something_selected)
            .keys(&["Ctrl+D"])
            .hint("A second copy beside this one, sharing nothing with it. Leaves the clipboard alone."),
        op("edit.delete_body", "delete", "Delete").menu(EDIT, 1).when(something_selected)
            .keys(&["Delete"])
            .hint("Removes the selected block and nothing else: what it was made from stays."),
        op("block.rename", "rename", "Rename...").menu(EDIT, 1).when(something_selected)
            .keys(&["F2"])
            .hint("A new name for the selected row, typed into the row. From the console any row by \
                   name: `rename 'Sphere_0' to=ball`."),
        op("edit.paste_here", "pastehere", "Paste here").when(has_clipboard_curve)
            .hint("Puts the copied curve down with its start under the cursor, moved as one piece \
                   so its shape is kept."),

        // ---------------------------------------------------------------------------- View
        // A checkbox reading the real state rather than a button that toggles it blind: the
        // menu has to say whether the cut is on, since a viewport with nothing in it cannot.
        op("view.section", "section", "Section").menu(VIEW, 0).checked(|c| c.section_on)
            .keys(&["X"])
            .hint("Cuts away everything past a plane so you can see inside. Drag the handles to \
                   move it. Changes the view only — the part is not modified. The first makes a \
                   section; after that it switches the current one on or off."),
        op("view.section_new", "addsection", "New section").menu(VIEW, 0)
            .hint("Another cut, made where the selection is and switched on beside the others: \
                   what is past any of them is cut away. Each section has a row in the block list."),
        op("view.measure", "measure", "Measure").menu(VIEW, 0)
            .hint("Volume, surface area and balance point of the selected body — or of the whole \
                   part when nothing is selected. Reported in the console."),
        // Beside Measure because it is the same kind of question asked of the same thing.
        op("view.thickness", "thickness", "Thickness").menu(VIEW, 0)
            .hint("How thin the part gets, where, and how much of its surface is below that — \
                   read from the field rather than the mesh. Reported in the console. \
                   `thickness min=0.8` names the threshold."),
        op("view.section_flip", "flipsection", "Flip section").menu(VIEW, 1).when(section_on)
            .aliases(&["flip"]).keys(&["Shift+X"])
            .hint("Keeps the other half"),
        op("view.control_points", "points", "Show control points").menu(VIEW, 2)
            .checked(|c| c.control_points)
            .hint("Every curve's control points, not only the selected curve's. Click one to pick \
                   it, Shift+click to add or take away across curves, Ctrl+drag a box to pick \
                   every point inside it, and drag any picked point to move them all."),
        op("view.view_cube", "viewcube", "Show ViewCube").menu(VIEW, 2).checked(|c| c.view_cube)
            .hint("The orientation cube in the top-right of the viewport"),
        op("view.tool_panel", "toolpanel", "Floating tool panel").menu(VIEW, 2).checked(|c| c.floating_panel)
            .hint("A small window of the block's numbers beside the pointer while it is being made. \
                   Off, the block opens in the list with its first number ready to type."),
        // Greyed out while Display quality is Fast, which holds both off whatever this says.
        // The tick is kept underneath so the preference comes back with Balanced or Full.
        op("view.idle_occlusion", "occlusion", "Shadows and occlusion when idle").menu(VIEW, 2)
            .when(occlusion_possible).checked(|c| c.idle_occlusion)
            .hint("Once the view has been still for a moment the frame is traced again with \
                   ambient occlusion and a soft shadow from the key light, then refined until it \
                   stops changing. Dragging never pays for either. Turn it off if a settled frame \
                   takes too long on this machine."),
        op("view.draft", "draft", "Draft analysis").menu(VIEW, 2).checked(|c| c.draft_on)
            .hint("Colour each face by whether it has enough draft to leave a mould. The pull \
                   direction, angle and tolerance are under Display in the part panel, or \
                   `draft angle=3 tolerance=1 vector=<curve>`."),
        op("view.grid", "grid", "Grid on the drawing plane").menu(VIEW, 2).checked(|c| c.grid_on)
            .hint("A grid on the plane you are drawing on, shown in an orthographic view, as fine \
                   as the zoom allows in steps of 1, 2 and 5 mm. Hold Shift while dragging a \
                   control point or placing a line to snap to it; Ctrl snaps to a curve's end \
                   instead. Off, there is nothing to snap to. Also under the ViewCube."),
        // Clickable modes, not only the Tab cycle: one setting with three values, so grouped.
        cmd("view.mode.polygon", "view.display", "display", "Cage").args(&[("mode", "polygon")])
            .menu(VIEW_MODE, 0).checked(|c| c.view_mode == "polygon")
            .hint("The control cage alone, flat-shaded so its facets read as facets."),
        cmd("view.mode.field", "view.display", "display", "Field").args(&[("mode", "field")])
            .menu(VIEW_MODE, 0).checked(|c| c.view_mode == "field")
            .hint("The surface itself, traced from the field."),
        cmd("view.mode.overlay", "view.display", "display", "Field + Cage").args(&[("mode", "overlay")])
            .menu(VIEW_MODE, 0).checked(|c| c.view_mode == "overlay")
            .hint("The surface with the cage drawn over it, both pickable."),
        op("view.march_stats", "marchstats", "Capture march stats").menu(VIEW, 5)
            .feature("march-stats")
            .hint("Steps per ray for this view at 240 and 1080 px, written beside the perf log. \
                   A measuring build only."),
        cmd("view.quality.fast", "view.display", "display", "Fast").args(&[("quality", "fast")])
            .menu(VIEW_QUALITY, 0).checked(|c| c.quality == "fast").hint(Q::Fast.hint()),
        cmd("view.quality.balanced", "view.display", "display", "Balanced").args(&[("quality", "balanced")])
            .menu(VIEW_QUALITY, 0).checked(|c| c.quality == "balanced").hint(Q::Balanced.hint()),
        cmd("view.quality.full", "view.display", "display", "Full").args(&[("quality", "full")])
            .menu(VIEW_QUALITY, 0).checked(|c| c.quality == "full").hint(Q::Full.hint()),
        cmd("view.quality.enchilada", "view.display", "display", "Enchilada").args(&[("quality", "enchilada")])
            .menu(VIEW_QUALITY, 0).checked(|c| c.quality == "enchilada").hint(Q::Enchilada.hint()),
        // Last, and on its own, because it is the only thing in this menu that does something
        // rather than setting how the part is looked at.
        // F, as §9.1 fixes it; it was `/` before the registry.
        op("view.frame_selection", "frame", "Frame selection").menu(VIEW, 6)
            .keys(&["F"])
            .hint("Fits the view to the selection, or to everything when nothing is selected."),

        // -------------------------------------------------------------------------- Create
        // **In the order a part gets built**: reference geometry, then a curve, then solids;
        // then the ways a curve becomes a solid; then the entries that are each one thing.
        op("create.point", "point", "Point").menu(PRIMITIVES, 0).session(POINT_P)
            .hint("A position to measure from; encloses nothing and draws nothing"),
        op("create.plane", "plane", "Plane").menu(PRIMITIVES, 0).session(PLANE_P)
            .hint("A half-space, for cutting with — infinite until you do"),
        // **The modelling keys are a first assignment, asked for and approved by the owner 2026-10-17** in place
        // of §9.1's "the team assigns them": the word's first letter where it is free (E, R, S, L,
        // C, B, P, M, O, T), Shift with it for the second command of a letter (Shift+S shell,
        // Shift+P pipe), and K -- the knife -- for split, whose S is taken. Thirteen, under the
        // spec's fifteen; `the_modelling_defaults_stay_within_the_spec_s_fifteen` holds the cap.
        // **One entry, and Line is not a second one.** A degree-1 curve *is* a line, and the
        // degree is changeable in the block afterwards. `line` still works in the console.
        op("create.curve", "curve", "Curve").menu(PRIMITIVES, 0).keys(&["C"]).session(&[])
            .hint("Click the start, then the end. Points land on the plane you are drawing on, or \
                   on the model. Straight to begin with; change its degree in the block. \
                   `curve3` is `curve degree=3`."),
        // **Only the derived axis is offered**: a plain axis is a two-point line, and the menu
        // was offering one thing under two names. `axis` in the console still draws one.
        cmd("construct.axis.surface", "construct.axis", "axis", "Axis on a surface")
            .args(&[("normal", "true")]).menu(PRIMITIVES, 0).when(field_drawn).session(&[])
            .hint("Click a surface: the axis stands on it, leaving along the field's direction, \
                   50 mm long. It is derived from the body, so it follows it: its block has a U, \
                   a V and a length rather than points."),
        op("create.sphere", "sphere", "Sphere").menu(PRIMITIVES, 1).session(SPHERE_P)
            .hint("A ball, by centre and radius"),
        op("create.cube", "cube", "Cube").menu(PRIMITIVES, 1).session(CUBE_P).aliases(&["box"])
            .hint("A box with every side the same"),
        op("create.skewbox", "skewedbox", "Skewed box").menu(PRIMITIVES, 1).session(SKEWBOX_P)
            .hint("A box whose faces lean — three edge vectors rather than three sides"),
        op("create.cylinder", "cylinder", "Cylinder").menu(PRIMITIVES, 1).session(CYLINDER_P)
            .aliases(&["rod", "bar"])
            .hint("A round bar between two points"),
        op("create.cone", "cone", "Cone").menu(PRIMITIVES, 1).session(CONE_P)
            .hint("Between two points, a radius at each — zero at one end for a point"),
        op("create.torus", "torus", "Torus").menu(PRIMITIVES, 1).session(TORUS_P).aliases(&["ring"])
            .hint("A ring in the XY plane"),
        op("create.capsule", "capsule", "Capsule").menu(PRIMITIVES, 1).session(CAPSULE_P)
            .hint("Between two points, a radius at each, with rounded ends"),
        // **Named for what it does, not for what it takes**: the ways a curve becomes a solid.
        // A closed loop is filled and an open curve is given a thickness, so one entry each.
        op("create.extrude", "extrude", "Extrude").menu(SWEEP, 0).keys(&["E"]).session(EXTRUDE_P).inputs(PROFILE)
            .aliases(&["pull", "push"])
            .hint("Gives a closed curve thickness along its own normal, out of one side of the \
                   plane it was drawn on. An open curve is given a thickness instead and extruded \
                   as a wall."),
        op("create.revolve", "revolve", "Revolve").menu(SWEEP, 0).keys(&["R"]).session(REVOLVE_P).inputs(PROFILE)
            .aliases(&["lathe", "spin"])
            .hint("Sweeps a closed curve about an axis. The profile has to stay clear of the axis, \
                   or the solid would pass through itself. An open curve is given a thickness \
                   instead and revolved as a wall."),
        op("create.sweep", "sweep", "Sweep").menu(SWEEP, 0).keys(&["S"]).session(SWEEP_P).inputs(SWEEP_IN)
            .hint("Carries a closed curve along another, held square to it the whole way. A \
                   straight path is an extrude and reads exactly; a bent one always reads a \
                   little short, and a fillet against it says so. With no closed curve to carry, \
                   the selected one is given a thickness and carried as a wall."),
        op("create.loft", "loft", "Loft").menu(SWEEP, 0).keys(&["L"]).session(LOFT_P).inputs(LOFT_IN)
            .hint("Passes a solid through two or more closed curves in turn, along a centerline — \
                   a curve you name, or one built from the first and last sections. Takes every \
                   closed curve in the document to start with; the block is where sections are \
                   added, removed and put in order."),
        op("create.pipe", "pipe", "Pipe").menu(SWEEP, 0).keys(&["Shift+P"]).session(PIPE_P).inputs(PIPE_IN)
            .aliases(&["tube"])
            .hint("Sweeps a curve into a tube. Makes a straight one to sweep if the document has \
                   none."),
        // Below the submenus, each being one thing rather than a family.
        op("create.cage", "cage", "Cage").menu(CREATE, 1)
            .aliases(&["subd", "control mesh"])
            .hint("A control cage whose limit surface is the body. Puts the document's cage back \
                   if its body was deleted."),
        op("create.tpms", "tpms", "TPMS").menu(CREATE, 1).keys(&["T"]).session(TPMS_P)
            .aliases(&["lattice", "gyroid", "infill"])
            .hint("A triply periodic minimal surface. Gyroid to begin with; the cell is changeable \
                   in the block. Intersect it with a body to infill."),
        // Beside the cage rather than in Primitives, because it is a number, not a shape.
        op("create.ramp", "ramp", "Ramp").menu(CREATE, 1).session(RAMP_P)
            .aliases(&["gradient", "grade"])
            .hint("A number that varies through space: it reads one field and maps a range of it \
                   onto a range of values. Draws nothing itself — pick it under Amount on an \
                   offset or a morph. One ramp can drive several."),
        op("create.number", "number", "Number").menu(CREATE, 1).session(&[])
            .aliases(&["expression", "variable"])
            .hint("A number computed from other parameters: radius + 10, min(a, b) * 2. Reads the \
                   variables you have named, changes none of them, and its result is offered \
                   under V on any field. Draws nothing itself."),
        op("create.tessellate", "tessellate", "Tessellate").menu(CREATE, 2).also(&["tess", "tesselate"])
            .session(&[]).aliases(&["mesh"])
            .hint("Turns the selected body — or the whole part — into triangles, and shows them in \
                   the viewport. Export Mesh writes one to a file; File → Export makes both."),

        // -------------------------------------------------------------------------- Modify
        // **Counted over what a tool can actually take**, the bodies *and* anything named.
        op("create.boolean", "boolean", "Boolean").menu(MODIFY, 0).keys(&["B"]).when(two_bodies)
            .session(BOOLEAN_P).inputs(TWO_BODIES).aliases(&["union", "subtract", "intersect", "fillet"])
            .hint("Combines two bodies; which two is editable in its block. `boolean -u`, `-s` or \
                   `-i` says which operation."),
        op("modify.pattern", "pattern", "Pattern").menu(MODIFY, 1).keys(&["P"]).when(a_body)
            .session(PATTERN_P).inputs(BODY).aliases(&["array", "repeat"])
            .hint("Repeats the selected body and unions the copies. How many, and in a row or \
                   around an axis, are editable in its block."),
        op("modify.mirror", "mirror", "Mirror").menu(MODIFY, 1).keys(&["M"]).when(a_body)
            .session(MIRROR_P).inputs(BODY).aliases(&["reflect"])
            .hint("Reflects the selected body about the origin and unions the two. The plane and \
                   the axis are editable in its block."),
        op("modify.twist", "twist", "Twist").menu(MODIFY, 1).when(a_body)
            .session(TWIST_P).inputs(BODY)
            .hint("Winds the selected body about Z. The rate is editable in its block."),
        op("modify.attract", "attract", "Attract").menu(MODIFY, 1).when(a_body)
            .session(ATTRACT_P).inputs(BODY)
            .hint("Draws the selected body toward a curve. Which curve, how hard and how far it \
                   reaches are editable in its block."),
        op("modify.offset", "offset", "Offset").menu(MODIFY, 1).keys(&["O"]).when(a_body)
            .session(OFFSET_P).inputs(BODY).aliases(&["grow", "thicken"])
            .hint("Moves the selected body's surface outward, or inward for a negative distance. \
                   The amount is editable in its block."),
        op("modify.morph", "morph", "Morph").menu(MODIFY, 1).when(two_bodies)
            .session(MORPH_P).inputs(MORPH_IN).aliases(&["blend between", "interpolate"])
            .hint("Averages two bodies into one, so that one becomes the other as the amount is \
                   dragged; which two is editable in its block. Works when they are in the same \
                   place — two bodies apart from each other vanish in between rather than \
                   travelling."),
        // "shell", which is what every CAD tool calls it.
        op("modify.shell", "shell", "Shell").menu(MODIFY, 1).keys(&["Shift+S"]).when(a_body)
            .session(SHELL_P).inputs(BODY).aliases(&["hollow"])
            .hint("Hollows the selected body out to a wall thickness, measured inward so the \
                   outside dimensions do not change."),
        op("modify.split", "split", "Split").menu(MODIFY, 1).keys(&["K"]).when(a_body)
            .session(&[]).inputs(SPLIT_IN).aliases(&["cut", "divide"])
            .hint("Cuts the selected body in two along a plane — one is made through its middle if \
                   there is none, and moving it moves the cut — or along another body \
                   (`split body=… by=…`). Both halves are named, A on the plane's solid side and \
                   B on the rest; the original and the cutter stay."),
        // **The Sub-D menu: the cage's own tools** (the owner, 2026-10-24), between Modify and Help.
        // Group 0 edits the cage's topology, group 1 grows a selection along it, group 2 checks it.
        // Each is also a console verb (R11): one line, one undo step, a refusal by name. A tool
        // whose elements a menu click cannot name reads them from the selection, and Insert Edge
        // arms and waits for a click on an edge.
        op("cage.refine", "refine", "Refine cage").menu(SUBD, 0).when(cage_control)
            .aliases(&["subdivide"])
            .hint("One Catmull-Clark step: four times the faces, so more control points to move. \
                   The surface does not change and does not get smoother. Undo puts the cage back \
                   exactly."),
        op("cage.extrude", "cageextrude", "Extrude faces").menu(SUBD, 0).when(cage_control).tool()
            .hint("Lifts the selected faces and lets you drag the cap out; Enter keeps, Esc puts \
                   it back, E again extrudes from the new cap. Or `cageextrude faces=1 distance=10 \
                   direction=normal|vertex_normals|x|y|z mode=region|individual`."),
        op("cage.extrude_edges", "extrudeedges", "Extrude open edges").when(cage_control)
            .hint("Grows an open mesh at its rim: `extrudeedges edges=0-1,1-2 distance=5`."),
        op("cage.inset", "cageinset", "Inset faces").menu(SUBD, 0).when(cage_control).tool().keys(&["I"])
            .hint("Insets the selected faces within their surface and lets you push the cap in or \
                   out; Enter keeps, Esc puts it back. Or `cageinset faces=1 thickness=2 depth=0`."),
        op("cage.insert_ring", "insertring", "Insert edge ring").menu(SUBD, 0).when(cage_control).tool().keys(&["Ctrl+R"])
            .hint("A loop of new edges through the one named, splitting every face it crosses: \
                   `insertring edge=0-1 t=0.5`, or `count=3` for three evenly spaced. From the menu \
                   or the key, hover an edge and click; Shift for the midpoint, Esc to stop."),
        op("cage.delete_faces", "deletefaces", "Delete faces").menu(SUBD, 0).when(cage_control)
            .hint("Takes the selected faces off the cage: `deletefaces faces=1,2`. The cage is open \
                   afterwards and shows red until the hole is filled."),
        op("cage.fill_hole", "fillhole", "Fill hole").menu(SUBD, 0).when(cage_control)
            .hint("Fills the hole whose rim the selected edge is on with quads: `fillhole edge=0-1`. \
                   A rim with an odd number of edges cannot be filled; cut one first."),
        op("cage.dissolve_edges", "dissolveedges", "Dissolve edges").menu(SUBD, 0).when(cage_control)
            .hint("Takes the selected edges out, merging the faces either side -- the inverse of \
                   Insert edge ring, and what Delete does with edges selected: \
                   `dissolveedges edges=0-1,1-2`."),
        op("cage.bridge_faces", "bridgefaces", "Bridge faces").menu(SUBD, 0).when(cage_control)
            .hint("Takes a region of faces off each side and bridges the rims they leave: select \
                   the two regions and run it, or `bridgefaces a=1,2 b=5,6 divisions=1`. B with \
                   faces selected."),
        op("cage.bridge", "bridge", "Bridge").menu(SUBD, 0).when(cage_control).tool()
            .hint("Quads between two open edges, or two runs of them: click an edge, then the \
                   edge to bridge it to; Shift-click adds edges to the side being gathered, and \
                   the bridge runs once both sides have as many. Or \
                   `bridge a=0-1,1-2 b=4-5,5-6 divisions=1`."),
        op("cage.cut", "knife", "Knife").menu(SUBD, 0).when(cage_control).tool()
            .hint("Click points on the cage's edges to draw a cut; every face stays a quad, so a \
                   loop runs on from each end until it closes. Enter commits, Backspace takes the \
                   last point back, Esc discards the path. Or `knife points=3-4@0.25,7-8@0.5`."),
        op("cage.weld", "weld", "Weld").menu(SUBD, 0).when(cage_control).tool()
            .hint("Click the vertex to move, then the vertex it moves onto; Shift on the second \
                   meets halfway. Or `weld pairs=3-7,4-8 at=midpoint`."),
        // Growing a selection along the cage's structure: a double-click on an edge is the loop,
        // Alt and a double-click the ring.
        op("select.loop", "selectloop", "Select edge loop").menu(SUBD, 1).when(cage_control)
            .hint("Extends a selected edge along its loop: straight on through every four-way \
                   vertex, stopping at a pole or a rim. Double-click an edge."),
        op("select.ring", "selectring", "Select edge ring").menu(SUBD, 1).when(cage_control)
            .hint("Extends a selected edge across its ring: the edge opposite it in each face. \
                   Alt and double-click an edge."),
        op("select.grow", "selectgrow", "Grow selection").menu(SUBD, 1).when(cage_control)
            .hint("Everything one step out from what is picked."),
        op("select.shrink", "selectshrink", "Shrink selection").menu(SUBD, 1).when(cage_control)
            .hint("Takes the border off what is picked."),
        op("cage.check", "check", "Check model").menu(SUBD, 2).when(cage_control)
            .hint("What is wrong with the cage: the red faults that stop it being a solid and the \
                   amber ones worth a look, listed with their counts; a click selects and frames \
                   them. From a script, the list is printed."),
        op("cage.set_positions", "setpositions", "Set cage positions").when(cage_control)
            .hint("Puts vertices where it says, as a drag does: `setpositions vertices=0,1 \
                   to=1,2,3;4,5,6`. What the echo writes for a cage vertex moved by hand."),

        // ---------------------------------------------------------------------------- Help
        // **The palette is in a menu as well as on Space**, because the menu is where a person
        // looks for what exists, and the palette is how they will reach everything after (P5).
        op("palette.open", "palette", "Command palette").menu(HELP, 0).keys(&["Space"])
            .aliases(&["search", "find a command", "commands"])
            .hint("Any command in a few keystrokes: type part of its name, its console word or a \
                   word it answers to, then Enter. A whole console line runs as typed."),
        op("help.shortcuts", "keys", "Keys...").menu(HELP, 0).also(&["shortcuts"])
            .aliases(&["keyboard shortcuts", "bindings", "rebind"])
            .hint("Every key and mouse gesture, and every command's keys: rebind by pressing them. \
                   Kept in bindings.txt beside the settings."),
        op("help.about", "about", "About fieldform...").menu(HELP, 0)
            .hint("Version, build and the third-party notice."),

        // ------------------------------------------------------ reached by typing or a key
        // `move target=box by=10,0,0`, `turn … by=0,0,45 about=0,0,0`, `scale … by=2`: each
        // wraps its target in one placement block, or adds to the placement it already is.
        op("modify.move", "move", "Move").session(MOVE_P).inputs(BODY).aliases(&["translate"])
            .hint("Moves a block by a vector, folding into the placement it already has."),
        op("modify.turn", "turn", "Turn").session(TURN_P).inputs(BODY).aliases(&["rotate"])
            .hint("Turns a block by three angles about a point."),
        op("modify.scale", "scale", "Scale").session(SCALE_P).inputs(BODY).aliases(&["resize"])
            .hint("Scales a block about a point."),
        op("create.line", "line", "Line").session(&[])
            .hint("A straight curve: `line p0=0,0,0 p1=40,0,0` places one now; `line` alone starts \
                   clicking."),
        op("construct.axis", "axis", "Axis").session(&[])
            .hint("A straight line to use as an axis: `axis from=0,0,0 to=0,0,50`, \
                   `axis through=Cylinder`, `axis normal=true` on a surface."),
        op("draw.plane", "draw", "Drawing plane").aliases(&["sketch plane", "workplane"])
            .hint("The plane clicks land on: `draw on=xy`."),
        op("view.cycle_mode", "mode", "Cycle display mode").keys(&["Tab"])
            .hint("Cage, field, field + cage, in turn."),
        op("view.toggle_visibility", "hide", "Hide / show").menu(VIEW, 3).keys(&["H"])
            .when(something_selected)
            .hint("Turns the selected block's eye over."),
        op("view.show_all", "showall", "Show all").menu(VIEW, 3).keys(&["Alt+H"])
            .aliases(&["unhide"])
            .hint("Every hidden body drawn again."),
        op("view.display", "display", "Display")
            .hint("`display quality=fast|balanced|full|enchilada`, `display mode=polygon|field|overlay`; \
                   alone, says which."),
        op("view.environment", "environment", "Environment").aliases(&["lighting", "studio"])
            .hint("The lighting the part is shown in: `environment name=zebra`."),
        op("view.perf", "perf", "Performance log")
            .hint("A line per traced frame into a file beside the console's, saying what it cost."),
        op("view.estimate", "estimate", "Estimate lattice")
            .hint("A lattice's stiffness and strength, estimated from published fits."),
        op("view.snap", "snap", "Snapshot").aliases(&["screenshot"])
            .hint("The finished still as a PNG: `snap file=\"ref.png\" view=iso`."),
        op("param.set", "param", "Parameter").aliases(&["variable", "dimension"])
            .hint("`param` lists them; `param name=wall value=2.4` makes or sets one; \
                   `param name=r2 expr=\"radius * 2\"` computes one."),
        op("block.make_variable", "variable", "Make variable")
            .hint("The dim V: what is called that takes a row of its own, and a name if it has none."),
        op("block.put_back_inside", "inline", "Put back inside")
            .hint("The bright V: a variable back inside the one block that reads it."),
        op("profile.join", "join", "Join loop")
            .hint("Holds the selected loop together with a point at every corner."),
        op("profile.ungroup", "ungroup", "Ungroup loop")
            .hint("Takes a loop's points out again, moving nothing."),
        op("profile.align", "align", "Align ends")
            .hint("A G1 or G2 joint: `align master=Line end=end slave=Curve at=start g=1`."),
        op("profile.relate", "relate", "Relate curves")
            .hint("`relate -parallel master=A slave=B`, or -perpendicular, -equal, -concentric, -mirror."),
        op("profile.break", "break", "Break relation")
            .hint("Deletes what drives a curve, moving nothing: `break slave=B`."),
        op("profile.realign", "realign", "Realign")
            .hint("Re-derives a relation found broken on open: `realign slave=B`."),
        op("curve.set_points", "set", "Set")
            .hint("A control point by name: `set Line.p1=10,20,0`."),
        op("part.set_material", "material", "Material")
            .hint("What the part is made of: `material preset=chrome`."),
        op("part.critical", "critical", "Critical surface")
            .hint("Marks a surface that must not move: `critical node=bore tol=0.01`."),
        op("part.verify", "verify", "Verify")
            .hint("Measures every critical surface against its mark."),
        op("script.run", "run", "Run script")
            .hint("The file's lines through this console: `run file=\"C:/tests/scene.ffc\"`."),
        op("check.expect", "expect", "Expect")
            .hint("A measurement asserted: `expect target=box volume=64000 tol=0.1%`."),
        op("console.help", "help", "Help")
            .hint("Every command, or `help <verb>` for what one reads."),
        op("console.clear", "clear", "Clear console")
            .hint("Empties the console's log."),
    ]
}

// ---------------------------------------------------------------------------- reading it

/// The entry a key binding runs: the one whose operation and fixed arguments it names.
pub fn for_binding(op: &str, args: &Args) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.built() && c.op == op && c.fixed_args() == *args)
}

/// **The cage's meaning of a key** (R9): with cage vertices, edges or faces selected, the letters
/// that act on bodies and profiles act on the cage instead -- the same letter, aimed at what is
/// selected. Only the operations built so far are here; extrude, inset and the knife join it with
/// their own items.
pub fn cage_element_op(op: &str, faces_selected: bool) -> Option<&'static str> {
    match op {
        // Delete takes faces off, or dissolves edges: a selected loop is taken out the way
        // Insert Edge put it in (the owner, 2026-10-24).
        "edit.delete_body" if faces_selected => Some("cage.delete_faces"),
        "edit.delete_body" => Some("cage.dissolve_edges"),
        // B bridges the selected faces' regions, or open edges.
        "create.boolean" if faces_selected => Some("cage.bridge_faces"),
        "create.boolean" => Some("cage.bridge"),
        // K is the knife on the cage, as it is Split on a body.
        "modify.split" => Some("cage.cut"),
        // E extrudes the cage's faces, or its open edges, as it extrudes a profile.
        "create.extrude" if faces_selected => Some("cage.extrude"),
        "create.extrude" => Some("cage.extrude_edges"),
        _ => None,
    }
}

/// The entry with this id, if there is one in this build.
// Read by the tool panel and the palette, which later steps of §12 build; the tests read it now.
#[cfg_attr(not(test), allow(dead_code))]
pub fn by_id(id: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.id == id && c.built())
}

/// Every word the console knows and the operation it runs, each word once.
///
/// The canonical verb and every other spelling of it. Entries that share an operation share a
/// word, and it is listed once.
pub fn console_words() -> Vec<(String, &'static str)> {
    let mut out: Vec<(String, &'static str)> = Vec::new();
    for c in COMMANDS.iter().filter(|c| c.built()) {
        for word in std::iter::once(&c.console_name).chain(c.console_also) {
            if !out.iter().any(|(w, _)| w == word) {
                out.push((word.to_string(), c.op));
            }
        }
    }
    out
}

/// One thing in a menu.
pub enum MenuItem {
    Command(&'static Command),
    Submenu(&'static str, Vec<MenuItem>),
    Separator,
}

/// The menu bar, generated (§10.1): each top-level menu with its items in table order, a line
/// wherever the group changes, and a submenu where its first entry falls.
pub fn menu_bar() -> Vec<(&'static str, Vec<MenuItem>)> {
    MENUS.iter().map(|top| (*top, menu_under(&[top]))).collect()
}

fn menu_under(path: &[&str]) -> Vec<MenuItem> {
    let mut items: Vec<MenuItem> = Vec::new();
    let mut last_group: Option<u8> = None;
    let mut submenus_placed: Vec<&'static str> = Vec::new();
    let mut place = |items: &mut Vec<MenuItem>, group: u8, item: MenuItem| {
        if last_group.is_some_and(|g| g != group) {
            items.push(MenuItem::Separator);
        }
        last_group = Some(group);
        items.push(item);
    };
    for c in COMMANDS.iter().filter(|c| c.built()) {
        if c.category.len() == path.len() && c.category.iter().zip(path).all(|(a, b)| a == b) {
            place(&mut items, c.group, MenuItem::Command(c));
        } else if c.category.len() == path.len() + 1
            && c.category.iter().zip(path).all(|(a, b)| a == b)
        {
            let name = c.category[path.len()];
            if submenus_placed.contains(&name) {
                continue;
            }
            submenus_placed.push(name);
            let group = SUBMENUS
                .iter()
                .find(|(p, _)| *p == c.category)
                .map_or(0, |(_, g)| *g);
            place(&mut items, group, MenuItem::Submenu(name, menu_under(c.category)));
        }
    }
    items
}

/// Every command a menu offers, depth first, for the test that each resolves.
#[cfg(test)]
pub fn menu_commands(items: &[MenuItem]) -> Vec<&'static Command> {
    let mut out = Vec::new();
    for item in items {
        match item {
            MenuItem::Command(c) => out.push(*c),
            MenuItem::Submenu(_, inner) => out.extend(menu_commands(inner)),
            MenuItem::Separator => {}
        }
    }
    out
}

/// The default keymap: every default binding in the table, in table order.
pub fn default_keymap() -> Keymap {
    let mut keymap = Keymap::default();
    for c in COMMANDS.iter().filter(|c| c.built()) {
        for chord in c.chords() {
            keymap.bind(chord.shortcut(), c.op, c.fixed_args());
        }
    }
    keymap
}

/// What the menu calls an entry now: Undo and Redo name what they would take back.
pub fn label(c: &Command, ctx: &Ctx) -> String {
    match (c.id, &ctx.undo, &ctx.redo) {
        ("edit.undo", Some(name), _) => format!("Undo {name}"),
        ("edit.redo", _, Some(name)) => format!("Redo {name}"),
        _ => c.display_name.to_string(),
    }
}

/// The dim text at the right of a menu row (§10.1, P5): the key if it has one, then the
/// console line, so the slow path teaches both fast ones.
pub fn menu_shortcut_text(c: &Command) -> String {
    match c.binding_label() {
        Some(key) => format!("{key}   {}", c.console_line()),
        None => c.console_line(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console_keys::{Keys, VERBS};

    fn built() -> impl Iterator<Item = &'static Command> {
        COMMANDS.iter().filter(|c| c.built())
    }

    /// **T2, first half: every entry has a console form, and typing it runs that entry**
    /// (`interaction-spec.md` §2.2, §11). The line is read by the console's own grammar, its verb
    /// resolves to the entry's operation, its arguments are the entry's fixed ones, and the key
    /// table refuses none of them -- so a menu row and its typed line cannot mean two things.
    #[test]
    fn every_command_s_console_line_parses_back_to_that_command() {
        let words = console_words();
        for c in built() {
            let line = c.console_line();
            let mut parsed = crate::console::console_words(&line).into_iter();
            let (verb, _) = parsed.next().unwrap_or_else(|| panic!("{}: an empty console line", c.id));
            let op = words
                .iter()
                .find(|(w, _)| *w == verb)
                .map(|(_, op)| *op)
                .unwrap_or_else(|| panic!("{}: `{verb}` is not a console word", c.id));
            assert_eq!(op, c.op, "{}: `{line}` runs {op}, not {}", c.id, c.op);
            let (args, naming) = crate::console::console_arguments(parsed)
                .unwrap_or_else(|why| panic!("{}: `{line}` does not parse: {why}", c.id));
            assert!(naming.is_none(), "{}: `{line}` names something", c.id);
            assert_eq!(args, c.fixed_args(), "{}: `{line}` reads other arguments", c.id);
            assert_eq!(crate::console_keys::refusal(&verb, op, &args), None, "{}: `{line}` is refused", c.id);
        }
    }

    /// **T2, second half: every menu item resolves to a registry id**, and each id is one entry.
    #[test]
    fn every_menu_item_resolves_to_one_registry_entry() {
        let mut ids = std::collections::HashSet::new();
        for c in COMMANDS.iter() {
            assert!(ids.insert(c.id), "{} is two entries", c.id);
        }
        let mut in_menus = 0;
        for (title, items) in &menu_bar() {
            assert!(!items.is_empty(), "the {title} menu is empty");
            for c in menu_commands(items) {
                let found = by_id(c.id).unwrap_or_else(|| panic!("{} is in a menu and not the registry", c.id));
                assert!(std::ptr::eq(found, c), "{} resolves to another entry", c.id);
                in_menus += 1;
            }
        }
        // And nothing with a menu place is left out of the bar: a category the bar does not draw
        // would be an entry that thinks it is in a menu and is not.
        let placed = built().filter(|c| !c.category.is_empty()).count();
        assert_eq!(in_menus, placed, "every entry with a category is drawn exactly once");
        for c in built().filter(|c| !c.category.is_empty()) {
            assert!(MENUS.contains(&c.category[0]), "{}: no menu called {}", c.id, c.category[0]);
            assert!(c.category.len() <= 2, "{}: menus nest one deep", c.id);
            if c.category.len() == 2 {
                assert!(SUBMENUS.iter().any(|(p, _)| *p == c.category), "{}: {:?} has no place", c.id, c.category);
            }
        }
    }

    /// No menu starts or ends on a line, and no two lines meet: a group with nothing in it.
    #[test]
    fn a_menu_never_draws_a_line_with_nothing_on_one_side_of_it() {
        fn check(path: &str, items: &[MenuItem]) {
            assert!(!matches!(items.first(), Some(MenuItem::Separator)), "{path} starts on a line");
            assert!(!matches!(items.last(), Some(MenuItem::Separator)), "{path} ends on a line");
            for pair in items.windows(2) {
                assert!(
                    !matches!(pair, [MenuItem::Separator, MenuItem::Separator]),
                    "{path} has two lines together"
                );
            }
            for item in items {
                if let MenuItem::Submenu(name, inner) = item {
                    check(&format!("{path} › {name}"), inner);
                }
            }
        }
        for (title, items) in menu_bar() {
            check(title, &items);
        }
    }

    /// What a session command's panel shows and what its picks name are keys its verb reads, so
    /// the panel, the console and the echo say one word for one field (D9).
    #[test]
    fn every_param_and_input_is_a_key_its_verb_reads() {
        for c in built() {
            let Some(verb) = VERBS.iter().find(|v| v.op == c.op) else {
                assert!(
                    c.params.is_empty() && c.inputs.is_empty(),
                    "{} names keys and its verb declares none",
                    c.id
                );
                continue;
            };
            let Keys::Declared { reads, .. } = &verb.keys else { continue };
            for key in c.params.iter().map(|p| p.key).chain(c.inputs.iter().map(|i| i.key)) {
                assert!(
                    reads.iter().any(|k| k.word == key),
                    "{}: `{key}` is not a key `{}` reads",
                    c.id,
                    c.console_name
                );
            }
            // A handle drags a length: the extrude's depth arrows are the one there is (§4.5).
            for param in c.params.iter().filter(|p| p.handle) {
                assert_eq!(param.unit.suffix(), "mm", "{}: `{}` has a handle", c.id, param.key);
            }
            for input in c.inputs {
                assert!(input.min >= 1, "{}: {:?} as `{}` needs at least one", c.id, input.accepts, input.key);
            }
        }
    }

    /// Every default binding is a chord a person could write in the bindings file, and no
    /// shortcut is bound twice out of the box.
    #[test]
    fn every_default_binding_parses_and_none_is_bound_twice() {
        let mut seen: std::collections::HashMap<Shortcut, &str> = Default::default();
        for c in built() {
            for text in c.default_bindings {
                let chord = Chord::parse(text).unwrap_or_else(|why| panic!("{}: {why}", c.id));
                assert_eq!(Chord::parse(&chord.label()).as_ref(), Ok(&chord), "{text} prints as it reads");
                if let Some(other) = seen.insert(chord.shortcut(), c.id) {
                    panic!("{text} is bound to both {other} and {}", c.id);
                }
            }
        }
    }

    /// The keymap the application starts with is the one it had before the registry, binding for
    /// binding: moving the table must not move a key.
    #[test]
    fn the_registry_s_keymap_is_the_keymap_the_application_had() {
        let km = default_keymap();
        let m = |ctrl, shift| Modifiers { ctrl, shift, alt: false };
        for (key, mods, op) in [
            ("Z", m(true, false), "edit.undo"),
            ("Z", m(true, true), "edit.redo"),
            ("Y", m(true, false), "edit.redo"),
            ("O", m(true, false), "file.open"),
            ("S", m(true, false), "file.save"),
            ("S", m(true, true), "file.save_as"),
            ("C", m(true, false), "edit.copy"),
            ("X", m(true, false), "edit.cut"),
            ("V", m(true, false), "edit.paste"),
            ("Delete", m(false, false), "edit.delete_body"),
            ("H", m(false, false), "view.toggle_visibility"),
            ("Tab", m(false, false), "view.cycle_mode"),
            ("F", m(false, false), "view.frame_selection"),
            ("D", m(true, false), "edit.duplicate"),
            ("H", Modifiers { ctrl: false, shift: false, alt: true }, "view.show_all"),
            ("X", m(false, false), "view.section"),
            ("X", m(false, true), "view.section_flip"),
            ("Space", m(false, false), "palette.open"),
            ("R", m(false, true), "edit.repeat"),
            ("F2", m(false, false), "block.rename"),
            ("C", m(false, false), "create.curve"),
            ("E", m(false, false), "create.extrude"),
            ("R", m(false, false), "create.revolve"),
            ("S", m(false, false), "create.sweep"),
            ("L", m(false, false), "create.loft"),
            ("P", m(false, true), "create.pipe"),
            ("T", m(false, false), "create.tpms"),
            ("B", m(false, false), "create.boolean"),
            ("P", m(false, false), "modify.pattern"),
            ("M", m(false, false), "modify.mirror"),
            ("O", m(false, false), "modify.offset"),
            ("S", m(false, true), "modify.shell"),
            ("K", m(false, false), "modify.split"),
            ("R", m(true, false), "cage.insert_ring"),
            ("I", m(false, false), "cage.inset"),
        ] {
            let bound = km.lookup(&Shortcut::new(key, mods)).map(|b| b.op.as_str());
            assert_eq!(bound, Some(op), "{key} {mods:?}");
        }
        // The fifteen it had with F for `/`, and the palette's Space, Duplicate's Ctrl+D and Show
        // all's Alt+H from §9.1's table; Shift+R for repeat, F2 for rename, the owner's thirteen
        // modelling keys, and Ctrl+R and I for the cage's Insert Edge and Inset (R9) -- which
        // brings the modelling keys to the fifteen the cap allows.
        assert_eq!(km.bindings().len(), 35, "no binding added or lost beyond those");
        assert!(km.lookup(&Shortcut::new("Slash", m(false, false))).is_none(), "F replaced /");
    }

    /// **Fifteen modelling bindings at most** (`interaction-spec.md` §9.1): a key on every tool is a
    /// keyboard nobody can learn, and the palette is what reaches the rest. Counted over what makes
    /// or changes geometry, since the owner ruled the cap is about those (§15.3).
    #[test]
    fn the_modelling_defaults_stay_within_the_spec_s_fifteen() {
        let modelling = ["create.", "modify.", "cage.", "construct.", "draw."];
        let bound: Vec<&str> = COMMANDS
            .iter()
            .filter(|c| modelling.iter().any(|p| c.op.starts_with(p)))
            .flat_map(|c| c.default_bindings.iter().copied())
            .collect();
        assert!(bound.len() <= 15, "{} modelling bindings: {bound:?}", bound.len());
    }

    /// Every word the console knew before the registry still reaches the same operation.
    #[test]
    fn every_console_word_the_application_had_still_runs_the_same_operation() {
        let words = console_words();
        for (word, op) in [
            ("cage", "create.cage"), ("move", "modify.move"), ("turn", "modify.turn"),
            ("scale", "modify.scale"), ("run", "script.run"), ("expect", "check.expect"),
            ("snap", "view.snap"), ("number", "create.number"), ("param", "param.set"),
            ("ramp", "create.ramp"), ("line", "create.line"), ("extrude", "create.extrude"),
            ("revolve", "create.revolve"), ("sweep", "create.sweep"), ("loft", "create.loft"),
            ("curve", "create.curve"), ("pipe", "create.pipe"), ("boolean", "create.boolean"),
            ("offset", "modify.offset"), ("mirror", "modify.mirror"), ("twist", "modify.twist"),
            ("attract", "modify.attract"), ("pattern", "modify.pattern"), ("shell", "modify.shell"),
            ("refine", "cage.refine"), ("split", "modify.split"), ("axis", "construct.axis"),
            ("morph", "modify.morph"), ("new", "file.new"), ("draw", "draw.plane"),
            ("section", "view.section"), ("open", "file.open"), ("save", "file.save"),
            ("saveas", "file.save_as"), ("import", "file.import"), ("export", "file.export_stl"),
            ("tess", "create.tessellate"), ("tessellate", "create.tessellate"),
            ("tesselate", "create.tessellate"), ("undo", "edit.undo"), ("redo", "edit.redo"),
            ("delete", "edit.delete_body"), ("variable", "block.make_variable"),
            ("inline", "block.put_back_inside"), ("copy", "edit.copy"), ("cut", "edit.cut"),
            ("duplicate", "edit.duplicate"), ("paste", "edit.paste"), ("join", "profile.join"),
            ("ungroup", "profile.ungroup"), ("align", "profile.align"), ("relate", "profile.relate"),
            ("break", "profile.break"), ("realign", "profile.realign"), ("set", "curve.set_points"),
            ("measure", "view.measure"), ("estimate", "view.estimate"), ("thickness", "view.thickness"),
            ("hide", "view.toggle_visibility"), ("mode", "view.cycle_mode"),
            ("material", "part.set_material"), ("critical", "part.critical"), ("verify", "part.verify"),
            ("environment", "view.environment"), ("draft", "view.draft"), ("grid", "view.grid"),
            ("perf", "view.perf"), ("display", "view.display"), ("tpms", "create.tpms"),
            ("point", "create.point"), ("plane", "create.plane"), ("sphere", "create.sphere"),
            ("cube", "create.cube"), ("skewedbox", "create.skewbox"), ("cylinder", "create.cylinder"),
            ("cone", "create.cone"), ("torus", "create.torus"), ("capsule", "create.capsule"),
            ("help", "console.help"), ("clear", "console.clear"),
        ] {
            let found = words.iter().find(|(w, _)| w == word).map(|(_, o)| *o);
            assert_eq!(found, Some(op), "`{word}`");
        }
    }

    /// A display quality or mode added without an entry would be one the menu cannot choose.
    #[test]
    fn every_display_quality_and_mode_has_an_entry_that_chooses_it() {
        for q in crate::display_quality::DisplayQuality::ALL {
            let value = q.label().to_lowercase();
            assert!(
                built().any(|c| c.op == "view.display" && c.args == [("quality", value.as_str())]),
                "no entry chooses {}",
                q.label()
            );
        }
        for mode in ["polygon", "field", "overlay"] {
            assert!(built().any(|c| c.op == "view.display" && c.args == [("mode", mode)]), "{mode}");
        }
    }

    /// The menu's "Undo" names what it would take back, as it did.
    #[test]
    fn undo_and_redo_say_what_they_would_take_back() {
        let undo = by_id("edit.undo").unwrap();
        let ctx = Ctx { undo: Some("graph edit".to_string()), ..Default::default() };
        assert_eq!(label(undo, &ctx), "Undo graph edit");
        assert_eq!(label(undo, &Ctx::default()), "Undo");
        assert_eq!((undo.context)(&Ctx::default()), Err("Nothing to undo"));
    }

    #[test]
    fn a_chord_reads_the_way_a_person_writes_one() {
        let c = Chord::parse("ctrl+shift+z").unwrap();
        assert_eq!((c.key.as_str(), c.ctrl, c.shift, c.alt), ("Z", true, true, false));
        assert_eq!(c.label(), "Ctrl+Shift+Z");
        assert_eq!(Chord::parse("/").unwrap().key, "Slash");
        assert_eq!(Chord::parse("Alt+H").unwrap().label(), "Alt+H");
        assert_eq!(Chord::parse("space").unwrap().key, "Space");
        assert!(Chord::parse("Hyper+Z").is_err());
        assert!(Chord::parse("Ctrl+").is_err());
        assert!(Chord::parse("Ctrl+Nonsense").is_err());
    }
}
