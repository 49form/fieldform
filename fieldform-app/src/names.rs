// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Which block names are in use.
//!
//! **A name outlives its block on purpose, so the name table is not the answer.** `node_meta` is
//! keyed by block id and is never pruned when a block is deleted, because undo restores the
//! graph through `SetPartGraph` and the block has to come back with its name. The cost is that
//! the table holds names of blocks that no longer exist, and the first clash check read the table
//! as if every entry were alive: delete a sphere called `ball`, type `sphere … 'ball'` again, and
//! the console said the name was taken by a sphere nobody could see. Reported by the owner the
//! first day the console could name things. So a name is *in use* only where some part's graph
//! still holds the block, which is what this answers.
//!
//! Free of the `App` so the rule has a test: an `App` needs a window and a device.

use fieldform_scene::{Document, Node, NodeId};

/// Every named block that still exists in some part, with its name.
pub fn in_use(doc: &Document) -> Vec<(NodeId, String)> {
    doc.node_meta
        .iter()
        .filter_map(|(id, meta)| {
            let name = meta.name.as_deref()?;
            let alive = doc.parts.values().any(|part| part.graph.node(*id).is_some());
            alive.then(|| (*id, name.to_string()))
        })
        .collect()
}

/// The word a block of this kind is named after when the interface names it: `Sphere_0`
/// (`notebook-spec.md` §5, `interaction-spec.md` §6.1).
///
/// **One word, no spaces**, so the name is one token wherever a line names it -- `target=Sphere_0`
/// needs no quotes -- and an identifier wherever an expression could.
pub fn operation_word(node: &Node) -> &'static str {
    match node {
        Node::Cage(_) => "Cage",
        Node::Mesh(_) => "Mesh",
        Node::Sphere { .. } => "Sphere",
        Node::Cuboid { .. } => "Cube",
        Node::SkewedBox { .. } => "SkewedBox",
        Node::Cylinder { .. } => "Cylinder",
        Node::Cone { .. } => "Cone",
        Node::Lattice { .. } => "Lattice",
        Node::Extrude { .. } => "Extrude",
        Node::Revolve { .. } => "Revolve",
        Node::Plane { .. } => "Plane",
        Node::Point { .. } => "Point",
        Node::Torus { .. } => "Torus",
        Node::Capsule { .. } => "Capsule",
        Node::Pipe { .. } => "Pipe",
        Node::Sweep { .. } => "Sweep",
        Node::Loft { .. } => "Loft",
        Node::Blend { .. } | Node::Boolean { .. } => "Boolean",
        Node::Morph { .. } => "Morph",
        Node::Twist { .. } => "Twist",
        Node::Attract { .. } => "Attract",
        Node::Pattern { .. } => "Pattern",
        Node::Mirror { .. } => "Mirror",
        Node::Shell { .. } => "Shell",
        Node::Ramp { .. } => "Ramp",
        Node::Offset { .. } => "Offset",
        Node::Transform { .. } => "Placement",
        Node::Unknown(_) => "Block",
    }
}

/// The lowest `<word>_<n>` that no living block, curve or parameter answers to.
///
/// **Against all three, ignoring case**, because a console word is looked up as a parameter, a
/// block and a curve alike, and a name that two of them share is refused at every reader (A9):
/// a name the interface invents must never be one a line cannot type.
pub fn instance_name(doc: &Document, word: &str) -> String {
    let taken: Vec<String> = in_use(doc)
        .into_iter()
        .map(|(_, n)| n)
        .chain(doc.parameters.values().map(|p| p.name.clone()))
        .chain(doc.curves.values().map(|c| c.name.clone()))
        .map(|n| n.to_lowercase())
        .collect();
    (0..=taken.len())
        .map(|n| format!("{word}_{n}"))
        .find(|candidate| !taken.contains(&candidate.to_lowercase()))
        .expect("one of len + 1 candidates is free")
}

/// A name for a new variable, after `base`: `base` itself if nothing answers to it, else the
/// lowest `base_<n>` that nothing does -- `Height`, then `Height_1` (`notebook-spec.md` §6.2).
/// Checked against blocks, curves and parameters in any case, for the reason [`instance_name`]
/// gives.
pub fn variable_name(doc: &Document, base: &str) -> String {
    let taken: Vec<String> = in_use(doc)
        .into_iter()
        .map(|(_, n)| n)
        .chain(doc.parameters.values().map(|p| p.name.clone()))
        .chain(doc.curves.values().map(|c| c.name.clone()))
        .map(|n| n.to_lowercase())
        .collect();
    if !taken.contains(&base.to_lowercase()) {
        return base.to_string();
    }
    (1..=taken.len() + 1)
        .map(|n| format!("{base}_{n}"))
        .find(|candidate| !taken.contains(&candidate.to_lowercase()))
        .expect("one of len + 1 candidates is free")
}

/// The block, other than `except`, that holds `name` -- or none, so the name is free.
pub fn holder_of(doc: &Document, name: &str, except: NodeId) -> Option<NodeId> {
    in_use(doc).into_iter().find(|(id, n)| *id != except && n == name).map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_cage::Cage;
    use fieldform_scene::{Node, ParamRef, Place};

    /// **An invented name is the lowest one free, and never one a line could confuse**: not a
    /// living block's, a parameter's or a curve's, in any case, and a deleted block's is free.
    #[test]
    fn an_instance_name_is_the_lowest_free_and_clashes_with_nothing() {
        let mut doc = Document::with_cage(Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("a part");
        assert_eq!(instance_name(&doc, "Sphere"), "Sphere_0");
        let add = |doc: &mut Document, name: &str| {
            let id = doc.parts.get_mut(&part).expect("the part").graph.add(Node::Sphere {
                centre: Place::origin(),
                radius: ParamRef::Literal(1.0),
            });
            doc.node_meta.entry(id).or_default().name = Some(name.to_string());
            id
        };
        let first = add(&mut doc, "Sphere_0");
        add(&mut doc, "sphere_1");
        assert_eq!(instance_name(&doc, "Sphere"), "Sphere_2", "taken in any case");
        assert!(doc.parts.get_mut(&part).expect("the part").graph.remove(first));
        assert_eq!(instance_name(&doc, "Sphere"), "Sphere_0", "a deleted block's name is free");
        let id = fieldform_scene::ParamId::new();
        doc.parameters.insert(id, fieldform_scene::Parameter {
            id,
            name: "Sphere_0".to_string(),
            value: fieldform_scene::ParamValue::Literal(1.0),
        });
        assert_eq!(instance_name(&doc, "Sphere"), "Sphere_2", "nor a parameter's");
        assert!(!operation_word(&Node::Cuboid { edge: Default::default(),
            centre: Place::origin(),
            size: [ParamRef::Literal(1.0); 3],
            turn: fieldform_scene::no_turn(),
        })
        .contains(' '));
    }

    #[test]
    fn a_deleted_block_s_name_is_free_again_and_undo_still_finds_it() {
        let mut doc = Document::with_cage(Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("a part");
        let ball = doc.parts.get_mut(&part).expect("the part").graph.add(Node::Sphere {
            centre: Place::origin(),
            radius: ParamRef::Literal(10.0),
        });
        doc.node_meta.entry(ball).or_default().name = Some("ball".to_string());
        let other = NodeId::new();
        assert_eq!(holder_of(&doc, "ball", other), Some(ball), "a living block holds its name");
        assert_eq!(holder_of(&doc, "ball", ball), None, "and may keep it");

        // Deleted: the graph no longer holds the block, the table still holds its name.
        assert!(doc.parts.get_mut(&part).expect("the part").graph.remove(ball));
        assert!(doc.node_meta.get(&ball).is_some_and(|m| m.name.as_deref() == Some("ball")), "kept for undo");
        assert_eq!(holder_of(&doc, "ball", other), None, "but the name is free");
        assert!(in_use(&doc).is_empty());
    }
}
