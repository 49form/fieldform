// SPDX-License-Identifier: GPL-3.0-only
//! What the graph view draws, derived from the notebook's own reading of the part.
//!
//! The view holds no copy of the document: this is rebuilt from the same `Nesting` the block list
//! is drawn from, so a relationship cannot appear in one view and not the other. It is a free
//! function over a plain [`Source`] rather than over `App`, for the reason `nest_rows` is — an
//! `App` needs a window and a device before it will exist, and the property worth testing is that
//! the two views agree.
//!
//! **Every block is a node, and every input of a block is a row.** A block nested in another is
//! drawn as a node of its own with a wire to the row it fills, exactly as a variable is; there is
//! no card drawn inside a node (the owner's ruling, 2026-10-30: nesting inside a node was not
//! intuitive). The notebook's containment is still visible, as the "owned" bracket on a nested
//! block's node.

use std::collections::{HashMap, HashSet};

use fieldform_scene::{CurveId, NodeId};

/// What a node stands for. Curves, profiles and surfaces are not graph nodes of the part but the
/// notebook lists them as blocks, so the graph draws them as nodes too and the click that selects
/// one has to know which kind of thing it landed on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Block,
    Number,
    Curve(CurveId),
    /// A closed loop of curves, folded into one row as the notebook folds it; keyed by its first.
    Profile(CurveId),
    Surface,
}

/// One block as the notebook reads it. The adapter in `mod.rs` fills these from the document.
#[derive(Clone, Debug)]
pub(crate) struct Source {
    pub(crate) id: NodeId,
    pub(crate) label: String,
    /// `Ownership::Variable`: a node of its own, marked with a V.
    pub(crate) variable: bool,
    /// Every block this one is drawn inside (`Nesting::owners_of`); empty for a row of its own.
    pub(crate) owners: Vec<NodeId>,
    /// The blocks this one reads that are nested in it, in the order it names them.
    pub(crate) children: Vec<NodeId>,
    /// The variables this one reads (`Nesting::references_of`): each is a chip in the notebook.
    pub(crate) references: Vec<NodeId>,
    /// Whether the block is drawn in the viewport, for the eye; `None` for a block with no eye
    /// (a ramp is a number, not a shape).
    pub(crate) eye: Option<bool>,
    /// A named number rather than a block: it has no inputs and its node holds a value field.
    pub(crate) number: Option<Number>,
    pub(crate) status: Status,
    pub(crate) kind: Kind,
}

/// A named number (`reach = 11`), drawn as a variable node with its value in a field.
///
/// It stands among the blocks under an id made from the parameter's own (`mod.rs`), which is why a
/// click on one must not be taken for a click on a block: the canvas reports it as a number.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Number {
    pub(crate) value: f64,
    /// Typed rather than computed, so the field may change it; an expression is shown, not edited.
    pub(crate) editable: bool,
}

/// What the notebook already says is wrong with a block, drawn as its node's border.
///
/// Only states the document can answer are here. The spec's "unbuilt / built" has no counterpart:
/// a block is evaluated whenever it is drawn, so there is nothing to be unbuilt, and inventing the
/// distinction for the graph would be a status the notebook never shows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum Status {
    #[default]
    Ok,
    /// Evaluates, but not as typed: an approximate fillet. The reason is for the hover.
    Warning(String),
    /// Cannot be evaluated: an empty input here or below, or a number whose expression failed.
    Error(String),
}

/// What a wire stands for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WireKind {
    /// A chip: a variable read by this slot.
    Chip,
    /// A nested block, still owned by the block it feeds.
    Owned,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Wire {
    pub(crate) from: NodeId,
    pub(crate) to: NodeId,
    pub(crate) kind: WireKind,
}

/// One input of a node: the row its wire ends on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) from: NodeId,
    pub(crate) label: String,
    pub(crate) kind: WireKind,
}

#[derive(Clone, Debug)]
pub(crate) struct GNode {
    pub(crate) id: NodeId,
    pub(crate) label: String,
    pub(crate) variable: bool,
    /// Nested in another block: drawn with the owned bracket, never as a variable.
    pub(crate) owned: bool,
    /// Its inputs, nested blocks first and then variables, in the order the block names them.
    pub(crate) rows: Vec<Row>,
    pub(crate) eye: Option<bool>,
    pub(crate) number: Option<Number>,
    pub(crate) status: Status,
    pub(crate) kind: Kind,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GraphModel {
    /// In the notebook's row order, which is what makes the layout deterministic.
    pub(crate) nodes: Vec<GNode>,
    pub(crate) wires: Vec<Wire>,
}

impl GraphModel {
    /// The nodes `from` is fed by (`upstream`) or feeds, itself included: what *Show what feeds
    /// this* and *Show what this feeds* keep lit (§6). Read-only, nothing stored.
    pub(crate) fn reach(&self, from: NodeId, upstream: bool) -> HashSet<NodeId> {
        let mut seen = HashSet::from([from]);
        let mut frontier = vec![from];
        while let Some(at) = frontier.pop() {
            for w in &self.wires {
                let next = match (upstream, w.to == at, w.from == at) {
                    (true, true, _) => w.from,
                    (false, _, true) => w.to,
                    _ => continue,
                };
                if seen.insert(next) {
                    frontier.push(next);
                }
            }
        }
        seen
    }

    /// The (reader, source) pairs the graph shows: what the notebook must show too.
    #[cfg(test)]
    pub(crate) fn edges(&self) -> std::collections::BTreeSet<(NodeId, NodeId)> {
        self.wires.iter().map(|w| (w.to, w.from)).collect()
    }
}

/// Draws `sources` (the notebook's rows, in row order) as nodes and wires.
pub(crate) fn build(sources: &[Source]) -> GraphModel {
    let by_id: HashMap<NodeId, &Source> = sources.iter().map(|s| (s.id, s)).collect();
    let mut model = GraphModel::default();
    let mut seen: HashSet<(NodeId, NodeId, bool)> = HashSet::new();
    for s in sources {
        let mut rows: Vec<Row> = Vec::new();
        for c in &s.children {
            if let Some(src) = by_id.get(c)
                && seen.insert((*c, s.id, true))
            {
                rows.push(Row { from: *c, label: src.label.clone(), kind: WireKind::Owned });
                model.wires.push(Wire { from: *c, to: s.id, kind: WireKind::Owned });
            }
        }
        for v in &s.references {
            if let Some(src) = by_id.get(v)
                && seen.insert((*v, s.id, false))
            {
                rows.push(Row { from: *v, label: src.label.clone(), kind: WireKind::Chip });
                model.wires.push(Wire { from: *v, to: s.id, kind: WireKind::Chip });
            }
        }
        model.nodes.push(GNode {
            id: s.id,
            label: s.label.clone(),
            variable: s.variable,
            owned: !s.variable && !s.owners.is_empty(),
            rows,
            eye: s.eye,
            number: s.number,
            status: s.status.clone(),
            kind: s.kind,
        });
    }
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> NodeId {
        NodeId::from_hex(&format!("{n:016x}")).unwrap()
    }

    fn src(n: u64, label: &str) -> Source {
        Source {
            id: id(n),
            label: label.into(),
            variable: false,
            owners: vec![],
            children: vec![],
            references: vec![],
            eye: None,
            number: None,
            status: Status::Ok,
            kind: Kind::Block,
        }
    }

    /// A boolean holding a sphere, and a cuboid that reads the variable `r` the sphere also reads.
    fn sample() -> Vec<Source> {
        let mut sphere = src(1, "Sphere");
        sphere.owners = vec![id(3)];
        sphere.references = vec![id(2)];
        let mut var = src(2, "r");
        var.variable = true;
        let mut boolean = src(3, "Boolean");
        boolean.children = vec![id(1)];
        let mut cuboid = src(4, "Cuboid");
        cuboid.references = vec![id(2)];
        vec![var, sphere, boolean, cuboid]
    }

    #[test]
    fn a_named_number_is_a_variable_node_wired_to_the_block_that_reads_it() {
        let mut sources = sample();
        let mut reach = src(9, "reach");
        reach.variable = true;
        reach.number = Some(Number { value: 11.0, editable: true });
        sources[3].references.push(id(9));
        sources.insert(0, reach);
        let model = build(&sources);
        let node = model.nodes.iter().find(|n| n.id == id(9)).unwrap();
        assert!(node.variable && node.number.is_some() && node.rows.is_empty());
        let cuboid = model.nodes.iter().find(|n| n.id == id(4)).unwrap();
        assert!(cuboid.rows.iter().any(|r| r.from == id(9) && r.kind == WireKind::Chip));
    }

    #[test]
    fn a_nodes_status_is_carried_from_its_source() {
        let mut sources = sample();
        sources[1].status = Status::Warning("approximate".into());
        let model = build(&sources);
        let node = model.nodes.iter().find(|n| n.id == id(1)).unwrap();
        assert_eq!(node.status, Status::Warning("approximate".into()));
        assert!(model.nodes.iter().filter(|n| n.id != id(1)).all(|n| n.status == Status::Ok));
    }

    #[test]
    fn a_curve_node_is_wired_to_the_block_that_reads_it_like_any_other_input() {
        let mut sources = sample();
        let curve = CurveId::new();
        let mut c = src(8, "outline");
        c.kind = Kind::Curve(curve);
        c.owners = vec![id(4)];
        sources[3].children.push(id(8));
        sources.insert(0, c);
        let model = build(&sources);
        let cuboid = model.nodes.iter().find(|n| n.id == id(4)).unwrap();
        assert!(cuboid.rows.iter().any(|r| r.from == id(8) && r.kind == WireKind::Owned));
        assert_eq!(model.nodes.iter().find(|n| n.id == id(8)).unwrap().kind, Kind::Curve(curve));
    }

    #[test]
    fn every_block_is_a_node_of_its_own() {
        let model = build(&sample());
        let ids: Vec<_> = model.nodes.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![id(2), id(1), id(3), id(4)]);
    }

    #[test]
    fn a_nested_block_is_wired_to_the_row_it_fills() {
        let model = build(&sample());
        let boolean = model.nodes.iter().find(|n| n.id == id(3)).unwrap();
        assert_eq!(boolean.rows.iter().map(|r| (r.from, r.kind)).collect::<Vec<_>>(), vec![(id(1), WireKind::Owned)]);
        assert!(model.wires.iter().any(|w| w.from == id(1) && w.to == id(3) && w.kind == WireKind::Owned));
        let sphere = model.nodes.iter().find(|n| n.id == id(1)).unwrap();
        assert!(sphere.owned && !sphere.variable, "nested is not a variable");
    }

    #[test]
    fn a_variable_read_by_two_blocks_has_a_row_and_a_wire_in_each() {
        let model = build(&sample());
        let mut to: Vec<_> = model.wires.iter().filter(|w| w.from == id(2)).map(|w| w.to).collect();
        to.sort();
        assert_eq!(to, vec![id(1), id(4)]);
        for reader in [id(1), id(4)] {
            let node = model.nodes.iter().find(|n| n.id == reader).unwrap();
            assert!(node.rows.iter().any(|r| r.from == id(2) && r.kind == WireKind::Chip));
        }
        assert!(model.nodes.iter().find(|n| n.id == id(2)).unwrap().variable);
    }

    #[test]
    fn every_wire_ends_on_a_row_and_every_row_has_a_wire() {
        let model = build(&sample());
        for w in &model.wires {
            let to = model.nodes.iter().find(|n| n.id == w.to).unwrap();
            assert!(to.rows.iter().any(|r| r.from == w.from && r.kind == w.kind));
        }
        let rows: usize = model.nodes.iter().map(|n| n.rows.len()).sum();
        assert_eq!(rows, model.wires.len());
    }

    #[test]
    fn the_graph_shows_exactly_the_relationships_the_notebook_has() {
        let sources = sample();
        let want: std::collections::BTreeSet<_> = sources
            .iter()
            .flat_map(|s| s.children.iter().chain(&s.references).map(move |c| (s.id, *c)))
            .collect();
        assert_eq!(build(&sources).edges(), want);
    }

    #[test]
    fn isolation_follows_wires_one_way_and_leaves_the_rest_dim() {
        let model = build(&sample());
        assert_eq!(model.reach(id(2), false), HashSet::from([id(2), id(1), id(3), id(4)]));
        assert_eq!(model.reach(id(4), true), HashSet::from([id(4), id(2)]));
        assert_eq!(model.reach(id(3), false), HashSet::from([id(3)]));
    }

    #[test]
    fn the_same_sources_build_the_same_model_twice() {
        let (a, b) = (build(&sample()), build(&sample()));
        assert_eq!(a.wires, b.wires);
        assert_eq!(a.nodes.iter().map(|n| n.id).collect::<Vec<_>>(), b.nodes.iter().map(|n| n.id).collect::<Vec<_>>());
    }
}
