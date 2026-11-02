// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The text form of a document: console lines that rebuild it (`mcp-server-spec.md` §5.6).
//!
//! **A writer for the kinds it can say, and a comment for the rest.** Nothing here guesses: a
//! block whose numbers the console cannot type (a taken place, a rounded edge, a kind with no
//! writer yet) is left out and *named* in a `# omitted:` line, so the script never claims to be
//! the document when it is not. The test that holds it is a round trip -- run the script on an
//! empty document and compare what comes out.
//!
//! A block that something consumes needs a name to be referred to by, so an unnamed one is given
//! `b1`, `b2`, … in the script. That is the one thing the rebuilt document has that the original
//! did not, and it is said in the header.

use super::*;
use fieldform_scene::{BooleanOp, Ends, Fillet, Node, NodeId, ParamRef, Place, Turn};
use std::collections::{BTreeMap, HashMap, HashSet};

/// The script, and the blocks and parameters it could not say.
pub(crate) struct Script {
    pub(crate) text: String,
    pub(crate) omitted: Vec<String>,
}

struct Writer<'a> {
    app: &'a App,
    names: HashMap<NodeId, String>,
    lines: Vec<String>,
    omitted: Vec<String>,
    done: HashSet<NodeId>,
    ok: HashSet<NodeId>,
    spent: usize,
}

fn bare(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl Writer<'_> {
    fn number(&self, p: &ParamRef) -> Option<String> {
        match p {
            ParamRef::Literal(v) => Some(format!("{v}")),
            ParamRef::Named(id) => self.app.doc.parameters.get(id).map(|p| p.name.clone()).filter(|n| bare(n)),
            _ => None,
        }
    }

    fn triple(&self, t: &[ParamRef; 3]) -> Option<String> {
        Some(format!("{},{},{}", self.number(&t[0])?, self.number(&t[1])?, self.number(&t[2])?))
    }

    fn place(&self, p: &Place) -> Option<String> {
        match p {
            Place::At(t) => self.triple(t),
            _ => None,
        }
    }

    fn is_zero(t: &[ParamRef; 3]) -> bool {
        t.iter().all(|p| matches!(p, ParamRef::Literal(v) if *v == 0.0))
    }

    /// What this block is called in the script: its own name if it can be typed, else a made-up
    /// one, so a consumer can say it.
    fn name_of(&mut self, id: NodeId) -> Option<String> {
        if let Some(n) = self.names.get(&id) {
            return Some(n.clone());
        }
        let own = self.app.doc.node_meta.get(&id).and_then(|m| m.name.clone());
        let name = match own {
            Some(n) if bare(&n) => n,
            Some(_) => return None,
            None => {
                self.spent += 1;
                format!("b{}", self.spent)
            }
        };
        self.names.insert(id, name.clone());
        Some(name)
    }

    /// The label to put after a line: the block's own name, or the made-up one if something
    /// consumes it.
    fn label(&mut self, id: NodeId, consumed: bool) -> String {
        let own = self.app.doc.node_meta.get(&id).and_then(|m| m.name.clone());
        match own {
            Some(n) if bare(&n) => format!(" '{n}'"),
            Some(n) if !n.contains('\'') && !consumed => format!(" '{n}'"),
            _ if consumed => self.name_of(id).map(|n| format!(" '{n}'")).unwrap_or_default(),
            _ => String::new(),
        }
    }

    fn omit(&mut self, id: NodeId, node: &Node, why: &str) {
        let kind = serde_json::to_value(node)
            .ok()
            .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(str::to_string))
            .unwrap_or_else(|| "block".into());
        let name = self.app.doc.node_meta.get(&id).and_then(|m| m.name.clone()).unwrap_or_default();
        self.omitted.push(format!("{kind} {name:?}: {why}"));
    }

    /// Writes the block after everything it reads. A block that cannot be said is omitted, and
    /// so is anything that reads it.
    fn visit(&mut self, id: NodeId, consumed: HashSet<NodeId>) -> bool {
        if self.done.contains(&id) {
            return self.ok.contains(&id);
        }
        self.done.insert(id);
        let Some(node) = self.app.part_graph().node(id).cloned() else { return false };
        let wanted = consumed.contains(&id);
        for input in node.inputs() {
            if !self.visit(input, consumed.clone()) {
                self.omit(id, &node, "reads a block that could not be written");
                return false;
            }
        }
        let said = self.say(id, &node, wanted);
        match said {
            Ok(()) => {
                self.ok.insert(id);
                true
            }
            Err(why) => {
                self.omit(id, &node, why);
                false
            }
        }
    }

    fn say(&mut self, id: NodeId, node: &Node, consumed: bool) -> Result<(), &'static str> {
        const TYPED: &str = "a number the console cannot type";
        match node {
            Node::Sphere { centre, radius } => {
                let (c, r) = (self.place(centre).ok_or(TYPED)?, self.number(radius).ok_or(TYPED)?);
                let name = self.label(id, consumed);
                self.lines.push(format!("sphere radius={r} centre={c}{name}"));
                self.mark(id);
            }
            Node::Cuboid { centre, size, turn, edge } => {
                if *edge != Fillet::default() {
                    return Err("a rounded edge");
                }
                let (c, s) = (self.place(centre).ok_or(TYPED)?, self.triple(size).ok_or(TYPED)?);
                let turn = match turn {
                    Turn::By(t) if Self::is_zero(t) => String::new(),
                    Turn::By(t) => format!(" turn={}", self.triple(t).ok_or(TYPED)?),
                    _ => return Err("a facing taken from another block"),
                };
                let name = self.label(id, consumed);
                self.lines.push(format!("cube size={s} centre={c}{turn}{name}"));
                self.mark(id);
            }
            Node::Cylinder { ends, radius, edge } => {
                if *edge != Fillet::default() {
                    return Err("a rounded edge");
                }
                let Ends::At { from, to } = ends else { return Err("ends taken from another block") };
                let (f, t, r) =
                    (self.place(from).ok_or(TYPED)?, self.place(to).ok_or(TYPED)?, self.number(radius).ok_or(TYPED)?);
                let name = self.label(id, consumed);
                self.lines.push(format!("cylinder radius={r} from={f} to={t}{name}"));
                self.mark(id);
            }
            Node::Boolean { op, inputs } => {
                let mut members = Vec::new();
                for operand in inputs {
                    let source = operand.source().ok_or("an empty slot")?;
                    members.push(self.names.get(&source).cloned().ok_or("an operand with no name to refer to")?);
                }
                let flag = match op {
                    BooleanOp::Union => "-u",
                    BooleanOp::Difference => "-s",
                    BooleanOp::Intersection => "-i",
                };
                let name = self.label(id, consumed);
                self.lines.push(format!("boolean {flag} in={}{name}", members.join(",")));
                self.mark(id);
            }
            Node::Transform { input, translation, rotation, scale, about } => {
                if !matches!(about, Place::At(t) if Self::is_zero(t)) {
                    return Err("a pivot that is not the origin");
                }
                let source = input.source().ok_or("an empty slot")?;
                let target = self.names.get(&source).cloned().ok_or("an operand with no name to refer to")?;
                if !Self::is_zero(translation) {
                    let by = self.triple(translation).ok_or(TYPED)?;
                    self.lines.push(format!("move target={target} by={by}"));
                }
                if !Self::is_zero(rotation) {
                    let by = self.triple(rotation).ok_or(TYPED)?;
                    self.lines.push(format!("turn target={target} by={by}"));
                }
                if !matches!(scale, ParamRef::Literal(v) if *v == 1.0) {
                    let by = self.number(scale).ok_or(TYPED)?;
                    self.lines.push(format!("scale target={target} by={by}"));
                }
                // A placement is not a block of its own in the script: a later line reaches the
                // body through the name it already has.
                self.names.insert(id, target);
                self.done.insert(id);
            }
            _ => return Err("no writer for this kind yet"),
        }
        Ok(())
    }

    /// Records the name a written block can be referred to by.
    fn mark(&mut self, id: NodeId) {
        let name = self.app.doc.node_meta.get(&id).and_then(|m| m.name.clone()).filter(|n| bare(n));
        let name = name.or_else(|| self.names.get(&id).cloned());
        if let Some(n) = name {
            self.names.insert(id, n);
        }
    }
}

impl App {
    /// The active part, and the parameters, as console lines.
    pub(crate) fn agent_script(&self) -> Script {
        let mut w = Writer {
            app: self,
            names: HashMap::new(),
            lines: Vec::new(),
            omitted: Vec::new(),
            done: HashSet::new(),
            ok: HashSet::new(),
            spent: 0,
        };
        let mut params: BTreeMap<String, &fieldform_scene::Parameter> = BTreeMap::new();
        for p in self.doc.parameters.values() {
            params.insert(p.name.clone(), p);
        }
        let mut head = Vec::new();
        for (name, p) in &params {
            match self.doc.parameter_value(p.id) {
                Some(v) if matches!(p.value, fieldform_scene::ParamValue::Literal(_)) && bare(name) => {
                    head.push(format!("param name={name} value={v}"));
                }
                _ => w.omitted.push(format!("parameter {name:?}: an expression is not written yet")),
            }
        }
        // Everything something else consumes needs a name to be said by.
        let graph = self.part_graph();
        let mut consumed: HashSet<NodeId> = HashSet::new();
        for (_, node) in graph.nodes() {
            consumed.extend(node.inputs());
        }
        let mut ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
        ids.sort_by_key(|id| id.raw());
        for id in ids {
            w.visit(id, consumed.clone());
        }
        let mut text = String::from(
            "# Rebuilds the active part on an empty document. A block nothing names is called b1, b2, ...\n",
        );
        for l in head.iter().chain(w.lines.iter()) {
            text.push_str(l);
            text.push('\n');
        }
        for o in &w.omitted {
            text.push_str(&format!("# omitted: {o}\n"));
        }
        Script { text, omitted: w.omitted }
    }
}
