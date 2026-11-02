// SPDX-License-Identifier: GPL-3.0-only
//! The automatic layout: layered, left to right, deterministic.
//!
//! Deterministic is a requirement and not a nicety (`notebook-graph-view-spec.md` §5): tests and
//! screenshots depend on the same document laying out the same way on every machine. So nothing
//! here iterates a hash map, every tie is broken by row order, and the arithmetic is a fixed
//! sequence of `f32` operations with no dependence on thread count.

use std::collections::HashMap;

use fieldform_scene::NodeId;

use super::model::GraphModel;

pub(crate) const NODE_WIDTH: f32 = 190.0;
pub(crate) const HEADER: f32 = 24.0;
pub(crate) const ROW: f32 = 18.0;
pub(crate) const COLUMN_GAP: f32 = 70.0;
pub(crate) const ROW_GAP: f32 = 22.0;

/// A node's height: its header, plus a line for each input.
pub(crate) fn node_height(rows: usize) -> f32 {
    HEADER + ROW * rows as f32 + 6.0
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Layout {
    /// Top-left corner of each node, in graph space, in the model's node order.
    pub(crate) at: Vec<[f32; 2]>,
    pub(crate) size: Vec<[f32; 2]>,
}

impl Layout {
    pub(crate) fn index_of(&self, model: &GraphModel) -> HashMap<NodeId, usize> {
        model.nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect()
    }
}

pub(crate) fn layout(model: &GraphModel) -> Layout {
    let n = model.nodes.len();
    let index: HashMap<NodeId, usize> = model.nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

    // Rank by dependency depth: a source sits left of everything it feeds. The longest path, found
    // by repeated relaxation capped at `n` rounds, so a cycle passing through mid-edit terminates.
    let mut rank = vec![0usize; n];
    for _ in 0..n {
        let mut moved = false;
        for w in &model.wires {
            let (Some(&f), Some(&t)) = (index.get(&w.from), index.get(&w.to)) else { continue };
            if f != t && rank[t] < rank[f] + 1 && rank[f] + 1 <= n {
                rank[t] = rank[f] + 1;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    let columns = rank.iter().copied().max().map_or(0, |m| m + 1);
    let mut order: Vec<Vec<usize>> = vec![Vec::new(); columns];
    for (i, r) in rank.iter().enumerate() {
        order[*r].push(i);
    }

    // Crossing reduction: a few barycentre sweeps, each column sorted by the mean position of its
    // neighbours in the column before, with the row index as the tie-break.
    let neighbours = |i: usize, towards_left: bool| -> Vec<usize> {
        model
            .wires
            .iter()
            .filter_map(|w| {
                let (Some(&f), Some(&t)) = (index.get(&w.from), index.get(&w.to)) else { return None };
                if towards_left && t == i && f != i {
                    Some(f)
                } else if !towards_left && f == i && t != i {
                    Some(t)
                } else {
                    None
                }
            })
            .collect()
    };
    for _ in 0..4 {
        for forward in [true, false] {
            let cols: Vec<usize> = if forward { (1..columns).collect() } else { (0..columns.saturating_sub(1)).rev().collect() };
            for c in cols {
                let mut position = vec![0usize; n];
                for col in &order {
                    for (p, i) in col.iter().enumerate() {
                        position[*i] = p;
                    }
                }
                let mut keyed: Vec<(f32, usize)> = order[c]
                    .iter()
                    .map(|&i| {
                        let ns = neighbours(i, forward);
                        let key = if ns.is_empty() {
                            position[i] as f32
                        } else {
                            ns.iter().map(|&j| position[j] as f32).sum::<f32>() / ns.len() as f32
                        };
                        (key, i)
                    })
                    .collect();
                keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
                order[c] = keyed.into_iter().map(|(_, i)| i).collect();
            }
        }
    }

    let mut at = vec![[0.0f32; 2]; n];
    let mut size = vec![[NODE_WIDTH, 0.0f32]; n];
    for (c, col) in order.iter().enumerate() {
        let mut y = 0.0f32;
        for &i in col {
            let h = node_height(model.nodes[i].rows.len() + model.nodes[i].number.is_some() as usize);
            at[i] = [c as f32 * (NODE_WIDTH + COLUMN_GAP), y];
            size[i] = [NODE_WIDTH, h];
            y += h + ROW_GAP;
        }
    }
    Layout { at, size }
}

#[cfg(test)]
mod tests {
    use super::super::model::{build, Source};
    use super::*;

    fn id(n: u64) -> NodeId {
        NodeId::from_hex(&format!("{n:016x}")).unwrap()
    }

    fn chain() -> Vec<Source> {
        let mk = |n: u64, v: bool, refs: Vec<u64>| Source {
            id: id(n),
            label: format!("b{n}"),
            variable: v,
            owners: vec![],
            children: vec![],
            references: refs.into_iter().map(id).collect(),
            eye: None,
            number: None,
            status: Default::default(),
            kind: crate::graph_view::model::Kind::Block,
        };
        vec![mk(1, true, vec![]), mk(2, false, vec![1]), mk(3, false, vec![1]), mk(4, true, vec![]), mk(5, false, vec![4, 1])]
    }

    #[test]
    fn a_variable_sits_left_of_every_block_that_reads_it() {
        let model = build(&chain());
        let l = layout(&model);
        let ix = l.index_of(&model);
        for w in &model.wires {
            assert!(l.at[ix[&w.from]][0] < l.at[ix[&w.to]][0]);
        }
    }

    #[test]
    fn no_two_nodes_overlap() {
        let model = build(&chain());
        let l = layout(&model);
        for a in 0..l.at.len() {
            for b in a + 1..l.at.len() {
                let apart_x = l.at[a][0] + l.size[a][0] <= l.at[b][0] || l.at[b][0] + l.size[b][0] <= l.at[a][0];
                let apart_y = l.at[a][1] + l.size[a][1] <= l.at[b][1] || l.at[b][1] + l.size[b][1] <= l.at[a][1];
                assert!(apart_x || apart_y, "{a} overlaps {b}");
            }
        }
    }

    #[test]
    fn layout_is_identical_across_runs() {
        let model = build(&chain());
        assert_eq!(layout(&model), layout(&model));
    }

    #[test]
    fn a_cycle_mid_edit_terminates() {
        let mut s = chain();
        s[1].references.push(id(3));
        s[2].references.push(id(2));
        let model = build(&s);
        let _ = layout(&model);
    }
}
