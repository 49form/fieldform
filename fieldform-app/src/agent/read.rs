// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The read tools: `inspect`, `measure`, `check` and `sample` (`mcp-server-spec.md` §5.7–5.10).
//!
//! **They read and never edit**, so none of them marks an undo step or moves the revision, and
//! each asks the document the questions the application already asks: the measure the console's
//! `measure` runs, the compile the viewport runs, `approximate_blends` the panel marks a fillet
//! with. A second opinion about the document that disagreed with the first would be the failure.

use super::tools::Reply;
use super::*;
use fieldform_field::{GraphField, MassProperties, Normalization, ScalarField};
use fieldform_scene::{Node, NodeId, ParamRef};
use nalgebra::Point3;
use serde_json::{Map, Value as J, json};

/// Cells along the longest axis: the console's own, and twice it (`measure_accuracy.rs`).
const STANDARD_CELLS: usize = crate::MEASURE_CELLS;
const MAX_SAMPLES: usize = 10_000;

fn failure(code: &'static str, message: String, extra: Vec<(&str, J)>) -> Reply {
    let mut error = json!({ "code": code, "message": message });
    for (k, v) in extra {
        error[k] = v;
    }
    Reply { json: json!({ "ok": false, "error": error }), is_error: true }
}

fn id_text(id: NodeId) -> String {
    format!("n:{:016x}", id.raw())
}

impl App {
    fn block_summary(&self, id: NodeId) -> J {
        let kind = self.part_graph().node(id).map(node_kind).unwrap_or_default();
        json!({ "id": id_text(id), "name": self.body_name(id), "kind": kind })
    }

    /// A block by id (`n:…`) or by name, exact and then case-insensitive. A name two blocks share
    /// is refused with every candidate rather than resolved to the first.
    fn resolve_block(&self, given: &str) -> Result<NodeId, Reply> {
        let graph = self.part_graph();
        let mut ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
        ids.sort_by_key(|id| id.raw());
        if let Some(hex) = given.strip_prefix("n:") {
            return ids
                .iter()
                .copied()
                .find(|id| format!("{:016x}", id.raw()) == hex)
                .ok_or_else(|| failure("not_found", format!("no block has the id `{given}`"), vec![("name", json!(given))]));
        }
        let by = |same: &dyn Fn(&str, &str) -> bool| -> Vec<NodeId> {
            ids.iter()
                .copied()
                .filter(|id| self.body_name(*id).is_some_and(|n| same(n, given)))
                .collect()
        };
        let mut hits = by(&|a, b| a == b);
        if hits.is_empty() {
            hits = by(&|a, b| a.eq_ignore_ascii_case(b));
        }
        match hits.len() {
            1 => Ok(hits[0]),
            0 => {
                let names: Vec<&str> = ids.iter().filter_map(|id| self.body_name(*id)).collect();
                let near = crate::outcome::nearest(&given.to_lowercase(), names.iter().copied());
                Err(failure(
                    "not_found",
                    format!("no block is called `{given}`"),
                    vec![("name", json!(given)), ("did_you_mean", json!(near))],
                ))
            }
            _ => Err(failure(
                "ambiguous_name",
                format!("{} blocks are called `{given}`; use an id", hits.len()),
                vec![("candidates", J::Array(hits.iter().map(|id| self.block_summary(*id)).collect()))],
            )),
        }
    }

    /// The part's drawn graph, or only `target`'s body of it, compiled to be measured: the graph
    /// the viewport draws, at the tolerance `measure` and `expect` read, so no section answers
    /// from its display grid.
    fn compiled(&self, target: Option<NodeId>) -> Result<(GraphField, fieldform_scene::Graph), Reply> {
        let mut graph = self.display_graph();
        if let Some(id) = target {
            for other in graph.roots() {
                if other != id {
                    graph.remove_body(other);
                }
            }
        }
        match GraphField::compile_graph(&self.doc, &graph, crate::MEASURE_TOLERANCE_MM) {
            Ok(field) => Ok((field, graph)),
            Err(e) => Err(failure("refused", format!("the part does not compile: {e}"), vec![])),
        }
    }

    pub(crate) fn agent_measure(&self, args: &Map<String, J>) -> Reply {
        let target = match args.get("target").and_then(J::as_str) {
            Some(t) => match self.resolve_block(t) {
                Ok(id) => Some(id),
                Err(r) => return r,
            },
            None => None,
        };
        if let Some(why) = crate::app::no_statement_while_a_slot_is_empty(
            self.part_graph(),
            |id| self.body_label(id),
            "measure",
        ) {
            return failure("refused", why, vec![]);
        }
        let fine = args.get("resolution").and_then(J::as_str) == Some("fine");
        let cells = if fine { STANDARD_CELLS * 2 } else { STANDARD_CELLS };
        let (field, _) = match self.compiled(target) {
            Ok(c) => c,
            Err(r) => return r,
        };
        let Some((lo, hi)) = self.measure_bounds(target) else {
            return failure("refused", "the body has no bounds to measure within (it may be unbounded)".into(), vec![]);
        };
        let Some(m) = MassProperties::of(&field, (lo, hi), cells) else {
            return failure("refused", "nothing was found to measure inside the body's bounds".into(), vec![]);
        };
        let longest = (hi - lo).amax();
        let wanted: Vec<&str> = match args.get("quantities").and_then(J::as_array) {
            Some(q) => q.iter().filter_map(J::as_str).collect(),
            None => vec!["volume", "area", "centroid", "bbox"],
        };
        let mut o = json!({
            "ok": true,
            "target": target.map(|id| self.block_summary(id)),
            "resolution": if fine { "fine" } else { "standard" },
            "cells_along_longest_axis": cells,
            "cell_size_mm": longest / cells as f64,
        });
        if wanted.contains(&"volume") {
            o["volume_mm3"] = json!(m.volume_mm3);
        }
        if wanted.contains(&"area") {
            o["surface_area_mm2"] = json!(m.surface_area_mm2);
        }
        if wanted.contains(&"centroid") {
            let c = m.centre_of_mass;
            o["centroid_mm"] = json!([c.x, c.y, c.z]);
        }
        if wanted.contains(&"bbox") {
            // A bound the field proves, not a tight box: it can be larger than the body.
            o["bounds_mm"] = json!({ "lo": [lo.x, lo.y, lo.z], "hi": [hi.x, hi.y, hi.z], "note": "the measuring box (a bound with margin), not a tight bounding box" });
        }
        Reply { json: o, is_error: false }
    }

    pub(crate) fn agent_sample(&self, args: &Map<String, J>) -> Reply {
        let Some(points) = args.get("points").and_then(J::as_array) else {
            return failure("missing_key", "`points` is required".into(), vec![("key", json!("points"))]);
        };
        if points.len() > MAX_SAMPLES {
            return failure("bad_value", format!("{} points is more than the {MAX_SAMPLES} one call may carry", points.len()), vec![("key", json!("points"))]);
        }
        let mut parsed: Vec<Point3<f64>> = Vec::with_capacity(points.len());
        for (i, p) in points.iter().enumerate() {
            let c: Option<Vec<f64>> = p.as_array().map(|a| a.iter().filter_map(J::as_f64).collect());
            match c {
                Some(c) if c.len() == 3 && p.as_array().is_some_and(|a| a.len() == 3) && c.iter().all(|v| v.is_finite()) => {
                    parsed.push(Point3::new(c[0], c[1], c[2]));
                }
                _ => return failure("bad_value", format!("point {i} is not three finite numbers"), vec![("key", json!("points"))]),
            }
        }
        let target = match args.get("target").and_then(J::as_str) {
            Some(t) => match self.resolve_block(t) {
                Ok(id) => Some(id),
                Err(r) => return r,
            },
            None => None,
        };
        let (field, _) = match self.compiled(target) {
            Ok(c) => c,
            Err(r) => return r,
        };
        let gradient = args.get("gradient").and_then(J::as_bool).unwrap_or(false);
        let values: Vec<J> = parsed
            .iter()
            .map(|p| {
                let mut o = json!({ "p": [p.x, p.y, p.z], "value": field.distance(*p) });
                if gradient {
                    let g = field.normal(*p).unwrap_or_else(|| {
                        let h = 1e-3;
                        let d = |dx: f64, dy: f64, dz: f64| field.distance(Point3::new(p.x + dx, p.y + dy, p.z + dz));
                        nalgebra::Vector3::new(
                            d(h, 0.0, 0.0) - d(-h, 0.0, 0.0),
                            d(0.0, h, 0.0) - d(0.0, -h, 0.0),
                            d(0.0, 0.0, h) - d(0.0, 0.0, -h),
                        ) / (2.0 * h)
                    });
                    o["gradient"] = json!([g.x, g.y, g.z]);
                }
                o
            })
            .collect();
        let normalization = match field.normalization() {
            Normalization::Distance => "Distance",
            _ => "LowerBound",
        };
        Reply {
            json: json!({
                "ok": true,
                "target": target.map(|id| self.block_summary(id)),
                "normalization": normalization,
                "meaning": if normalization == "Distance" {
                    "negative inside, positive outside; the magnitude is a distance in mm (never an overestimate)"
                } else {
                    "negative inside, positive outside; the magnitude is only a lower bound on the distance"
                },
                "samples": values,
            }),
            is_error: false,
        }
    }

    pub(crate) fn agent_check(&self) -> Reply {
        let graph = self.part_graph();
        let mut problems: Vec<J> = Vec::new();
        let mut add = |code: &str, id: Option<NodeId>, this: &App, message: String| {
            let mut p = json!({ "code": code, "message": message });
            if let Some(id) = id {
                p["block"] = this.block_summary(id);
            }
            problems.push(p);
        };
        if let Some((id, slot)) = graph.first_empty_operand() {
            add("empty_slot", Some(id), self, format!("slot {slot} is empty, so the part is read with a body missing"));
        }
        let mut ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
        ids.sort_by_key(|id| id.raw());
        for id in &ids {
            let Some(node) = graph.node(*id) else { continue };
            if matches!(node, Node::Unknown(_)) {
                add("unknown_block", Some(*id), self, "a block this build cannot read; it is kept as written".into());
            }
            for r in node.param_refs() {
                if let ParamRef::Named(p) = r
                    && self.doc.parameter_value(*p).is_none()
                {
                    let name = self.doc.parameters.get(p).map(|p| p.name.clone()).unwrap_or_else(|| "?".into());
                    add("unresolved_parameter", Some(*id), self, format!("parameter `{name}` has no value (a missing parameter, a failed expression or a division by zero)"));
                }
            }
        }
        let consumed = graph.consumed();
        for id in &ids {
            if self.doc.node_meta.get(id).is_some_and(|m| m.hidden)
                && !consumed.contains(id)
                && graph.node(*id).is_some_and(Node::is_body)
            {
                add("hidden_body", Some(*id), self, "hidden and consumed by nothing, so nothing draws it and it is not measured".into());
            }
        }
        // Not a fault, but the one thing here that changes what `measure` and `sample` answer
        // without changing any body: the block is inside another and counted beside it as well.
        for id in self.shown_ingredients() {
            add("drawn_ingredient", Some(id), self, "inside another block and drawn beside it as well, so the part's measure and sample count it too".into());
        }
        for id in fieldform_field::approximate_blends(&self.doc, graph) {
            add("approximate_blend", Some(id), self, "a blend against a field that is only a lower bound on distance: the radius is not the radius asked for (the fillet is up to about 1.7x larger)".into());
        }
        // An empty part has nothing to evaluate, which is a blank page and not a fault.
        if graph.nodes().next().is_some()
            && let Some(why) = self.compile_error()
        {
            add("does_not_compile", None, self, why);
        }
        Reply { json: json!({ "ok": problems.is_empty(), "problems": problems }), is_error: false }
    }

    pub(crate) fn agent_inspect(&self, args: &Map<String, J>) -> Reply {
        let Some(given) = args.get("block").and_then(J::as_str) else {
            return failure("missing_key", "`block` is required".into(), vec![("key", json!("block"))]);
        };
        let id = match self.resolve_block(given) {
            Ok(id) => id,
            Err(r) => return r,
        };
        let graph = self.part_graph();
        let Some(node) = graph.node(id) else {
            return failure("not_found", format!("no block has the id `{given}`"), vec![]);
        };
        let readers: Vec<J> = graph
            .nodes()
            .filter(|(_, n)| n.inputs().contains(&id))
            .map(|(r, _)| self.block_summary(r))
            .collect();
        let reads: Vec<J> = node.inputs().into_iter().map(|i| self.block_summary(i)).collect();
        let keys = serde_json::to_value(node).unwrap_or(J::Null);
        // A reference by name or id is shown as the parameter's name beside its value.
        let parameters: Vec<J> = node
            .param_refs()
            .into_iter()
            .filter_map(|r| match r {
                ParamRef::Named(p) => Some(json!({
                    "name": self.doc.parameters.get(p).map(|p| p.name.clone()),
                    "value": self.doc.parameter_value(*p),
                })),
                _ => None,
            })
            .collect();
        let exact = fieldform_field::carries_true_distance(&self.doc, graph, id);
        let approximate = fieldform_field::approximate_blends(&self.doc, graph).contains(&id);
        let mut o = json!({
            "ok": true,
            "block": self.block_summary(id),
            "body": graph.roots().contains(&id),
            // Not the same question: a pattern's source is no body, and is drawn.
            "drawn": self.drawn_bodies().contains(&id),
            "hidden": self.doc.node_meta.get(&id).is_some_and(|m| m.hidden),
            "keys": keys,
            "reads": reads,
            "read_by": readers,
            "parameters_read": parameters,
            "exactness": if exact { "Distance" } else { "LowerBound or Scalar" },
            "approximate_blend": approximate,
        });
        if let Some((_, slot)) = graph.first_empty_operand().filter(|(n, _)| *n == id) {
            o["empty_slot"] = json!(slot);
        }
        Reply { json: o, is_error: false }
    }
}

fn node_kind(node: &Node) -> String {
    serde_json::to_value(node)
        .ok()
        .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(str::to_string))
        .unwrap_or_else(|| "unknown".into())
}
