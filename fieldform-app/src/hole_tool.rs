// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Create › Hole and the `hole` verb: a drilled hole, made already subtracted from a body.
//!
//! **Placed by a click, remembered as two points.** The click finds the surface and the field's
//! gradient there, turned round and normalised, is the way in; both are stored as literal ends at
//! that moment, so the hole remembers no face and stays where it was put when the body changes
//! (the boundary rule). Typed, `at=` and `dir=` say the same two things.
//!
//! **Made subtracted.** A body that is already a difference takes the hole as one more cutter --
//! a second hole in the same block is one more input of the same difference, not a difference of
//! a difference, and a rounded one is the same list -- and any other body is wrapped in a new
//! one. Either way the hole stays its own block, in the list and editable.

use super::*;
use fieldform_scene::{BooleanOp, HoleDepth, HoleKind, HoleTip, Operand};

impl App {
    /// `hole`: made from the line's keys where they say where, else waits for the click.
    pub(crate) fn hole_command(&mut self, args: &Args) {
        let placed = ["at", "from"].iter().any(|k| args.get(k).is_some());
        if placed {
            let target = match self.hole_target(args, None) {
                Ok(t) => t,
                Err(why) => {
                    self.warn(why);
                    return;
                }
            };
            self.make_hole(target, args);
            return;
        }
        // The click finishes it, with these keys.
        self.pending_hole = Some(args.clone());
        self.begin_axis_placement_for(AxisKind::Hole, None);
    }

    /// The one click of a hole: the surface under the cursor, the way into it, and the body it is.
    pub(crate) fn place_hole_click(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let Some(at) = self.normal_axis_landing(aspect, rect, pos) else {
            self.warn("No surface under the cursor — click on the body itself");
            return;
        };
        let Some(into) = self.pick_field.as_ref().and_then(|f| fieldform_field::into_material(f, at)) else {
            self.warn("The surface there has no direction to drill along — click somewhere else on it");
            return;
        };
        let args = self.pending_hole.take().unwrap_or_default();
        let graph = self.part_graph().clone();
        let under = crate::split_axis::body_under(&self.doc, &graph, at, &mut self.leaves);
        let target = match self.hole_target(&args, under) {
            Ok(t) => t,
            Err(why) => {
                self.pending_hole = Some(args);
                self.warn(why);
                return;
            }
        };
        let triple = |v: Vector3<f64>| Value::Str(format!("{},{},{}", v.x, v.y, v.z));
        let args = args.with("at", triple(at.coords)).with("dir", triple(into));
        self.pending_curve = None;
        self.make_hole(target, &args);
    }

    /// Which body a hole is cut from: the one the line named, else the one clicked, else the
    /// selection. A hole with nothing to cut is refused: it is a cutter, and a cutter on its own is
    /// a solid nobody asked for.
    fn hole_target(&mut self, args: &Args, clicked: Option<NodeId>) -> Result<NodeId, String> {
        let named = console_keys::operand_named("hole", "target", args, &|t| self.lookup(t))?;
        let bodies = self.doc.listed_bodies(self.part_graph());
        named
            .and_then(|o| o.source())
            .or(clicked)
            .or(self.selected_body.filter(|id| bodies.contains(id)))
            .ok_or_else(|| "A hole is cut from a body — name one with `target=`, or click on it".to_string())
    }

    /// Makes the hole from `args` and subtracts it from `target`.
    fn make_hole(&mut self, target: NodeId, args: &Args) {
        let seed = Node::Hole {
            ends: fieldform_scene::Ends::default(),
            kind: HoleKind::default(),
            depth: HoleDepth::default(),
            tip: HoleTip::Drill(ParamRef::Literal(118.0)),
        };
        let node = match console_keys::hole_with_keys(seed, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let mut graph = self.part_graph().clone();
        let hole = graph.add(node);
        let body = wire_hole(&mut graph, target, hole);
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.clear_hidden(target);
        self.select_body(Some(body));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(format!("Cut a hole in {}", self.body_label(body)));
    }
}

/// Subtracts `hole` from `target` and answers the body that is the result: `target` itself when
/// it is already a difference, hard or rounded, which takes the hole as one more cutter, and a
/// new difference of the two otherwise.
///
/// **Rounded too**, because a subtract given a radius after its first hole was wrapped by the
/// second in a subtract of its own -- the nest the owner reported (2026-10-28), when a rounded
/// join took exactly two operands. It takes a list now, and the hole's rim is rounded like the
/// first's.
pub(crate) fn wire_hole(graph: &mut Graph, target: NodeId, hole: NodeId) -> NodeId {
    if let Some(
        Node::Boolean { op: BooleanOp::Difference, inputs }
        | Node::Blend { op: BooleanOp::Difference, inputs, .. },
    ) = graph.node_mut(target)
    {
        inputs.push(Operand::filled(hole));
        return target;
    }
    graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![target.into(), hole.into()] })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A second hole on the same body appends to the same difference**, and the first wraps
    /// the body in one.
    #[test]
    fn a_second_hole_joins_the_difference_the_first_made() {
        let mut graph = Graph::empty();
        let block = graph.add(Node::Cuboid {
            centre: fieldform_scene::Place::origin(),
            size: [ParamRef::Literal(40.0); 3],
            turn: fieldform_scene::no_turn(),
            edge: Default::default(),
        });
        let hole = |g: &mut Graph| {
            g.add(Node::Hole {
                ends: fieldform_scene::Ends::default(),
                kind: HoleKind::default(),
                depth: HoleDepth::Through,
                tip: HoleTip::Flat,
            })
        };
        let first = hole(&mut graph);
        let body = wire_hole(&mut graph, block, first);
        assert_ne!(body, block);
        let second = hole(&mut graph);
        assert_eq!(wire_hole(&mut graph, body, second), body, "the same difference");
        let Some(Node::Boolean { op: BooleanOp::Difference, inputs }) = graph.node(body) else { panic!("a difference") };
        let ids: Vec<_> = inputs.iter().filter_map(|o| o.source()).collect();
        assert_eq!(ids, vec![block, first, second]);
        assert_eq!(graph.roots(), vec![body], "one body, the holes inside it");

        // **Rounded, the same list** -- the case the owner met: a radius on the subtract, and the
        // next hole joined the rounded one rather than wrapping it in a subtract of its own.
        let Some(Node::Boolean { op, inputs }) = graph.node(body).cloned() else { panic!("a difference") };
        let fillet = fieldform_scene::Fillet {
            radius: ParamRef::Literal(1.0),
            continuity: fieldform_scene::Continuity::G1,
            second: None,
        };
        *graph.node_mut(body).expect("there") = Node::Blend { op, inputs, fillet };
        let third = hole(&mut graph);
        assert_eq!(wire_hole(&mut graph, body, third), body, "the same rounded difference");
        let Some(Node::Blend { inputs, .. }) = graph.node(body) else { panic!("still rounded") };
        let ids: Vec<_> = inputs.iter().filter_map(|o| o.source()).collect();
        assert_eq!(ids, vec![block, first, second, third]);
        assert_eq!(graph.roots(), vec![body]);
    }
}
