// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Split and the Axis command — `cad-parity-v1-instructions.md` §1 and §3.
//!
//! Neither adds a kind of block. A split is two booleans over one body and one tool, and an axis
//! is a straight line, which is what every block that takes a direction already takes. Both live
//! here rather than among `App`'s methods in `main.rs` and `app/`, so that what they add to those
//! is the menu entries and the console words and nothing else — no logic for a session editing the
//! rest of them to merge around.

use super::*;
use fieldform_field::LeafCache;
use fieldform_ops::{Batch, SetSurfaceAxis};
use fieldform_scene::{Operand, Place, SurfaceAxis};

/// How long an axis drawn normal to a surface is, when nothing says. Long enough to be seen on a
/// default-sized body and short enough not to leave it; the line's end is a control point, so
/// dragging it is the way to a different length and `length=` is the way to type one.
const AXIS_NORMAL_LENGTH_MM: f64 = 50.0;

/// What an axis being placed in the viewport is waiting for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum AxisKind {
    /// A start and an end, the way any line is placed.
    TwoPoints,
    /// One click on a surface: the axis leaves it along the field's normal there.
    Normal,
}

/// What a split added to the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Split {
    /// `body ∩ tool`: the part on the tool's solid side.
    pub inside: NodeId,
    /// `body − tool`: the rest.
    pub outside: NodeId,
    /// What cut it. Newly made when none was given.
    pub tool: NodeId,
    pub made_tool: bool,
}

/// Adds a split of `body` by `tool` to `graph`.
///
/// **Both halves read the same two nodes, never copies of them**, so editing the body or moving
/// the tool updates both halves at once — the point of building it from one pair rather than two.
/// The body and the tool each end up with two consumers, which is also what keeps them in the
/// block list as rows of their own (`Document::nesting_in`: a block two others read stays above
/// both), so "the original and the tool are hidden, not deleted" needs nothing stored: they are
/// consumed, so they are not bodies, and they are still there to be edited.
///
/// The tool is a plane through `through` when none is given. A plane is unbounded, which is fine:
/// the intersection kills the unboundedness and the difference is bounded by the body.
pub(crate) fn split_in(
    graph: &mut Graph,
    body: NodeId,
    tool: Option<NodeId>,
    through: Point3<f64>,
) -> Split {
    let made_tool = tool.is_none();
    let tool = tool.unwrap_or_else(|| {
        graph.add(Node::Plane {
            origin: Place::At([
                ParamRef::Literal(through.x),
                ParamRef::Literal(through.y),
                ParamRef::Literal(through.z),
            ]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        })
    });
    // Both slots filled at birth: a split is made of two blocks that exist, so neither
    // boolean is ever born waiting to be told what to read.
    let slots = || vec![Operand::filled(body), Operand::filled(tool)];
    let inside = graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: slots() });
    let outside = graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: slots() });
    Split { inside, outside, tool, made_tool }
}

/// Names for the two halves: `<label> A` and `<label> B`, or `<label>_1 A` and so on when either
/// is taken.
///
/// **Invented, so suffixed rather than refused** — the same rule `unique_variable_name` follows
/// for a name nobody chose. The pair moves together: two halves called `bracket A` and `bracket_1
/// B` would read as unrelated.
fn half_names(taken: &std::collections::HashSet<&str>, label: &str) -> (String, String) {
    (0..)
        .map(|n| if n == 0 { label.to_string() } else { format!("{label}_{n}") })
        .map(|base| (format!("{base} A"), format!("{base} B")))
        .find(|(a, b)| !taken.contains(a.as_str()) && !taken.contains(b.as_str()))
        .expect("an unbounded range has a free pair")
}

/// Three numbers from a console value: `[x, y, z]`, or the text `x,y,z`.
fn triple(value: &Value) -> Option<[f64; 3]> {
    match value {
        Value::Vec3(t) => Some(*t),
        Value::Str(text) => {
            let parts: Vec<f64> =
                text.split(',').map(|part| part.trim().parse().ok()).collect::<Option<_>>()?;
            <[f64; 3]>::try_from(parts).ok().filter(|t| t.iter().all(|v| v.is_finite()))
        }
        _ => None,
    }
}

/// Which way the field increases at `p`, as a unit vector.
///
/// **Normalised explicitly.** A gradient is unit length only for a `Distance` field, and most of
/// the interesting targets — a lattice, a lean extrude — are `LowerBound`. The block's own
/// analytic normal is taken when it has one; otherwise a central difference of its distance, at
/// a step the caller sizes to the part.
fn field_direction(field: &dyn ScalarField, p: Point3<f64>, h: f64) -> Option<Vector3<f64>> {
    if let Some(n) = field.normal(p).and_then(|n| n.try_normalize(1e-9)) {
        return Some(n);
    }
    let at = |dx: f64, dy: f64, dz: f64| field.distance(p + Vector3::new(dx, dy, dz));
    let gradient = Vector3::new(
        at(h, 0.0, 0.0) - at(-h, 0.0, 0.0),
        at(0.0, h, 0.0) - at(0.0, -h, 0.0),
        at(0.0, 0.0, h) - at(0.0, 0.0, -h),
    );
    gradient.try_normalize(1e-9)
}

/// The surface point nearest `from`, found by walking the field's own gradient.
///
/// **What an SDF is for.** The value at a point *is* the distance to the surface, so stepping
/// `-f(p)` along the unit gradient lands on it in one move for a true distance field, and creeps in
/// for one that understates (a twist, a lattice, a blend) — which is why this iterates rather than
/// stepping once. Converged when the reading is inside `tolerance`, which is the same tolerance the
/// field was compiled at.
///
/// It works where a projection cannot: every face is reachable because there is no ray to be
/// parallel to one, and a torus or a C-shaped shell is no harder than a ball, because the question
/// asked is about the surface rather than about a box around it. The one place it is awkward is the
/// medial axis, where two walls are equally near and a small move of the seed swaps which one wins
/// — local, and no worse than the ray's own discontinuity at a silhouette.
///
/// `None` when the walk does not settle, which a caller reports rather than papering over: a seed
/// exactly on the medial axis of a symmetric body has no nearest point to give.
fn nearest_surface_point(
    field: &dyn ScalarField,
    from: Point3<f64>,
    tolerance: f64,
    step: f64,
) -> Option<Point3<f64>> {
    let mut at = from;
    let mut best: Option<(f64, Point3<f64>)> = None;
    for _ in 0..64 {
        let reading = field.distance(at);
        if reading.abs() <= tolerance {
            return Some(at);
        }
        // Kept, so a walk that wanders is still answered by the closest it ever came rather than
        // by nothing. A field that understates can step short for ever and never cross.
        if best.is_none_or(|(d, _)| reading.abs() < d) {
            best = Some((reading.abs(), at));
        }
        let down = field_direction(field, at, step)?;
        at -= down * reading;
    }
    // Within a few tolerances is a surface point for this purpose; further and the walk has not
    // found one and the caller should say so.
    best.filter(|(d, _)| *d <= tolerance * 8.0).map(|(_, p)| p)
}

/// Stands `axis` on its body: the two points of the line it draws, or why it cannot be.
///
/// **A function of the document, not of the application**, for the reason `settle_curves_on_points`
/// is: the rule is worth asserting without a window, and the panel, the creation and the
/// re-derivation after every change all call this one.
///
/// The body's field is compiled on its own — `Graph::subtree`, so it is the only body — and a ray is
/// run along the axis's direction through the place U and V name across its box, from outside, to the
/// first surface it meets. The line leaves that point along the field's own direction there,
/// normalised (a gradient is unit only for a `Distance` field), `length` long.
pub(crate) fn solve_surface_axis(
    doc: &Document,
    graph: &Graph,
    axis: &SurfaceAxis,
    leaves: &mut LeafCache,
) -> Result<[Point3<f64>; 2], String> {
    let body = graph
        .subtree(axis.body)
        .ok_or_else(|| "the body it stands on is gone".to_string())?;
    let (lo, hi) = fieldform_field::graph_extent(doc, &body, axis.body)
        .as_box()
        .ok_or_else(|| {
            "the body it stands on has no limit to stand against — it is empty, or it reaches \
             forever"
                .to_string()
        })?;
    let diagonal = (hi - lo).norm();
    if diagonal < 1e-9 {
        return Err("the body it stands on has no size".to_string());
    }
    let field = GraphField::compile_cached(doc, &body, (diagonal * 1e-3).max(1e-4), leaves)
        .map_err(|e| format!("the body it stands on cannot be evaluated: {e}"))?;
    let tolerance = (diagonal * 1e-3).max(1e-4);
    let step = (diagonal * 1e-4).max(1e-6);
    // **A seed is the default and the ray is the fallback**, which is also what an older file
    // means: it has no seed, so it takes the second arm and reads exactly as it always did.
    let hit = match axis.seed_point([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]) {
        Some(seed) => nearest_surface_point(
            &field,
            Point3::new(seed[0], seed[1], seed[2]),
            tolerance,
            step,
        )
        .ok_or_else(|| {
            "no nearest surface point settles from that seed — it may be equally far from two \
             walls; move it toward the one you mean"
                .to_string()
        })?,
        None => {
            let margin = diagonal * 0.1 + 1.0;
            let (origin, direction) = axis.ray([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z], margin);
            trace_field(
                &field,
                Point3::new(origin[0], origin[1], origin[2]),
                Vector3::new(direction[0], direction[1], direction[2]),
                diagonal + 2.0 * margin,
                diagonal * 1e-4,
                512,
            )
            .ok_or_else(|| {
                "the ray at that U and V misses the body — move U or V, or look along another axis"
                    .to_string()
            })?
        }
    };
    let along = field_direction(&field, hit, (diagonal * 1e-4).max(1e-6))
        .ok_or_else(|| "the field has no direction there — it is flat, or on a crease".to_string())?;
    Ok([hit, hit + along * axis.length.max(0.0)])
}

/// The body whose surface `point` is on, among the roots of `graph`: the one whose own field reads
/// nearest nought there. A click lands on the part's field, which is every body's union, so which
/// body it belongs to has to be asked.
pub(crate) fn body_under(
    doc: &Document,
    graph: &Graph,
    point: Point3<f64>,
    leaves: &mut LeafCache,
) -> Option<NodeId> {
    graph
        .roots()
        .into_iter()
        .filter(|id| {
            graph.node(*id).is_some_and(|n| !matches!(n, Node::Plane { .. } | Node::Point { .. }))
        })
        .filter_map(|id| {
            let body = graph.subtree(id)?;
            let (lo, hi) = fieldform_field::graph_extent(doc, &body, id).as_box()?;
            let tolerance = ((hi - lo).norm() * 1e-3).max(1e-4);
            let field = GraphField::compile_cached(doc, &body, tolerance, leaves).ok()?;
            Some((field.distance(point).abs(), id))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, id)| id)
}

/// The axis that stands on `body` where `point` is: which way to look, and the U and V that put the
/// ray through it.
///
/// **The direction is the axis the surface faces most**, so the ray meets it as squarely as a ray
/// along a coordinate axis can, from the side the surface faces. The candidates are tried in that
/// order and the first whose ray *comes back to the point* — within a thousandth of the body — wins:
/// on a concave body the squarest ray may meet another part of the surface first. If none does, the
/// nearest the rays get is used, and the caller says where it landed.
pub(crate) fn surface_axis_through(
    doc: &Document,
    graph: &Graph,
    body: NodeId,
    point: Point3<f64>,
    length: f64,
    leaves: &mut LeafCache,
) -> Result<SurfaceAxis, String> {
    let sub = graph.subtree(body).ok_or_else(|| "the body is gone".to_string())?;
    let (lo, hi) = fieldform_field::graph_extent(doc, &sub, body)
        .as_box()
        .ok_or_else(|| "that body has no limit to stand an axis against".to_string())?;
    let diagonal = (hi - lo).norm();
    let field = GraphField::compile_cached(doc, &sub, (diagonal * 1e-3).max(1e-4), leaves)
        .map_err(|e| format!("that body cannot be evaluated: {e}"))?;
    let facing = field_direction(&field, point, (diagonal * 1e-4).max(1e-6))
        .ok_or_else(|| "the field has no direction there — it is flat, or on a crease".to_string())?;

    // **Seeded from the clicked point, which is already on the surface.** The seed is what the
    // axis is placed by from now on, so this is the whole of it: no ray has to reach the point, and
    // nothing about the body's shape can make the click unplaceable. `u`, `v`, `direction` and
    // `flipped` are filled in beside it all the same, so an older build opens the file and still
    // has a ray to fall back on, and so switching the panel to the ray has sensible numbers to
    // switch *to* rather than nought.
    let seed = SurfaceAxis::seed_of(
        [point.x, point.y, point.z],
        [lo.x, lo.y, lo.z],
        [hi.x, hi.y, hi.z],
    );

    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| facing[*b].abs().total_cmp(&facing[*a].abs()));
    let mut nearest: Option<(f64, SurfaceAxis)> = None;
    for i in order {
        let direction = Axis::ALL[i];
        let (u, v) = SurfaceAxis::place_of(
            direction,
            [point.x, point.y, point.z],
            [lo.x, lo.y, lo.z],
            [hi.x, hi.y, hi.z],
        );
        let candidate = SurfaceAxis {
            body,
            direction,
            flipped: facing[i] < 0.0,
            u: u.clamp(0.0, 1.0),
            v: v.clamp(0.0, 1.0),
            length,
            seed: None,
        };
        // Asked of the *ray*, since what is being chosen here is which ray to record beside the
        // seed. The seed itself needs no search.
        if let Ok([hit, _]) = solve_surface_axis(doc, graph, &candidate, leaves) {
            let off = (hit - point).norm();
            if off <= diagonal * 1e-3 {
                return Ok(SurfaceAxis { seed: Some(seed), ..candidate });
            }
            if nearest.is_none_or(|(best, _)| off < best) {
                nearest = Some((off, candidate));
            }
        }
    }
    // **No ray reaching it is no longer a refusal.** It is exactly the case the seed was added for
    // — a face parallel to every ray, the inside of a shell, a torus's throat — so the axis stands
    // on the seed and the ray beside it is whichever came closest, or the squarest if none ran.
    let fallback = nearest.map(|(_, axis)| axis).unwrap_or(SurfaceAxis {
        body,
        direction: Axis::ALL[order[0]],
        flipped: facing[order[0]] < 0.0,
        u: 0.5,
        v: 0.5,
        length,
        seed: None,
    });
    Ok(SurfaceAxis { seed: Some(seed), ..fallback })
}

impl App {
    /// The graph that holds `body`, whichever part it is in. A curve belongs to the document and an
    /// axis on a surface names a node, so the part has to be found from the node.
    pub(crate) fn graph_holding(&self, body: NodeId) -> Option<&Graph> {
        self.doc.parts.values().map(|p| &p.graph).find(|g| g.node(body).is_some())
    }

    /// Re-derives every axis on a surface from its body, writing the two points of its line.
    /// Answers whether any line moved.
    ///
    /// **Run when the part is rebuilt**, which is every change, undo, redo and open — the same
    /// "one derivation, whichever path moved the master" rule the relations follow — so a body
    /// that moves takes its axes with it, and an undo of an edit to U, V or length catches the line
    /// up with no code of its own. Not run mid-drag: a drag skips the rebuild by design, so an axis
    /// follows a dragged body when the hand comes off. An axis whose ray no longer meets its body
    /// keeps its last good place and says why in its panel (`surface_axis_notes`).
    pub(crate) fn refresh_surface_axes(&mut self) -> bool {
        if self.doc.surface_axes.is_empty() {
            self.surface_axis_notes.clear();
            return false;
        }
        let axes: Vec<(CurveId, SurfaceAxis)> =
            self.doc.surface_axes.iter().map(|(id, a)| (*id, *a)).collect();
        let mut moved = false;
        for (id, axis) in axes {
            if self.doc.curve(id).is_none() {
                continue;
            }
            moved |= self.refresh_surface_axis(id, axis);
        }
        moved
    }

    /// One axis, derived and written. See [`Self::refresh_surface_axes`].
    fn refresh_surface_axis(&mut self, id: CurveId, axis: SurfaceAxis) -> bool {
        let Some(graph) = self.graph_holding(axis.body).cloned() else {
            self.surface_axis_notes
                .insert(id, "the body it stands on is gone — the line keeps its last place".to_string());
            return false;
        };
        match solve_surface_axis(&self.doc, &graph, &axis, &mut self.leaves) {
            Ok(points) => {
                self.surface_axis_notes.remove(&id);
                let Some(curve) = self.doc.curve_mut(id) else { return false };
                if curve.control_points().len() != 2 {
                    return false;
                }
                let differs = points
                    .iter()
                    .zip(curve.control_points())
                    .any(|(to, from)| (to - from).norm() > 1e-9);
                if differs {
                    curve.set_control_point(0, points[0]);
                    curve.set_control_point(1, points[1]);
                }
                differs
            }
            Err(why) => {
                self.surface_axis_notes.insert(id, format!("{why} — the line keeps its last place"));
                false
            }
        }
    }

    /// Why a curve's points may not be moved by hand, if it is an axis on a surface.
    ///
    /// Its start is where the body is and its end is a length from it, so a hand on either would be
    /// overruled by the next derivation — and a refusal that says so is better than a point that
    /// snaps back. One question for every path that moves a point (`begin_curve_gesture` is the
    /// single gate: drag, manipulator and the console's `set`).
    pub(crate) fn surface_axis_refusal(&self, curve: CurveId) -> Option<String> {
        self.doc.surface_axes.contains_key(&curve).then(|| {
            let name = self.doc.curve(curve).map_or_else(|| "That axis".to_string(), |c| c.name.clone());
            format!(
                "\u{201c}{name}\u{201d} stands on a surface — its place is a U, a V and a length \
                 in its block, not points to drag"
            )
        })
    }

    /// Makes an axis on a surface: the line and its record, as one undo step.
    fn make_surface_axis(&mut self, axis: SurfaceAxis) -> Option<CurveId> {
        let Some(graph) = self.graph_holding(axis.body).cloned() else {
            self.warn("That body is not in a part");
            return None;
        };
        let points = match solve_surface_axis(&self.doc, &graph, &axis, &mut self.leaves) {
            Ok(points) => points,
            Err(why) => {
                self.warn(why);
                return None;
            }
        };
        let id = CurveId::new();
        let mut curve = match bezier_between(id, 1, points[0], points[1]) {
            Ok(curve) => curve,
            Err(e) => {
                self.warn(format!("could not make an axis: {e}"));
                return None;
            }
        };
        curve.name = self.next_axis_name();
        let name = curve.name.clone();
        let commands: Vec<Box<dyn fieldform_ops::Command>> =
            vec![Box::new(SetCurve::insert(curve)), Box::new(SetSurfaceAxis::insert(id, axis))];
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(Batch::new("axis.surface", commands))) {
            self.warn(e.to_string());
            return None;
        }
        self.select_curve(id);
        self.rebuild_curve_display();
        self.mark_changed();
        self.say(format!(
            "Added {name} on {} — U {:.3}, V {:.3}, {:.1} mm. Its place is U, V and length in its \
             block",
            self.body_label(axis.body),
            axis.u,
            axis.v,
            axis.length
        ));
        Some(id)
    }

    /// An axis stood on the surface at `at`, reading which body and which ray from the point.
    fn stand_axis_on_surface_for(
        &mut self,
        at: Point3<f64>,
        length: f64,
        restanding: Option<CurveId>,
    ) -> bool {
        let graph = self.part_graph().clone();
        let Some(body) = body_under(&self.doc, &graph, at, &mut self.leaves) else {
            self.warn("No body to stand an axis on there");
            return false;
        };
        match surface_axis_through(&self.doc, &graph, body, at, length, &mut self.leaves) {
            Ok(axis) => match restanding {
                Some(id) => self.restand_surface_axis(id, axis),
                None => self.make_surface_axis(axis).is_some(),
            },
            Err(why) => {
                self.warn(why);
                false
            }
        }
    }

    /// Stands an axis that already exists somewhere new: its record and its two points, as one
    /// undo step, keeping its name and its length.
    ///
    /// **Why this exists at all.** An axis stood on a body whose body is then deleted keeps its
    /// stored points and goes on drawing, but has nothing left to re-derive from — and until this
    /// there was no way back: the block could not be pointed at another body, so the only route
    /// was to delete it and make another. Reported as exactly that: "once you delete the sphere
    /// it's then a dumb block ... you have to delete it and recreate it, I think that's kind of a
    /// waste." A reference that can be re-pointed is the ordinary arrangement everywhere else in
    /// this panel; this is the one that could not be.
    fn restand_surface_axis(&mut self, id: CurveId, axis: SurfaceAxis) -> bool {
        let Some(graph) = self.graph_holding(axis.body).cloned() else {
            self.warn("That body is not in a part");
            return false;
        };
        let points = match solve_surface_axis(&self.doc, &graph, &axis, &mut self.leaves) {
            Ok(points) => points,
            Err(why) => {
                self.warn(why);
                return false;
            }
        };
        let Some(before) = self.doc.curve(id).cloned() else {
            self.warn("That axis is gone");
            return false;
        };
        let mut moved = match bezier_between(id, 1, points[0], points[1]) {
            Ok(curve) => curve,
            Err(e) => {
                self.warn(format!("could not stand it there: {e}"));
                return false;
            }
        };
        // **Its own name, not a fresh one.** This is the same block standing somewhere else, and
        // anything that refers to it by name goes on referring to it.
        moved.name = before.name.clone();
        let name = moved.name.clone();
        let was = self.doc.surface_axes.get(&id).copied();
        let commands: Vec<Box<dyn fieldform_ops::Command>> = vec![
            Box::new(SetCurve::new(id, Some(before), Some(moved))),
            Box::new(SetSurfaceAxis::new(id, was, Some(axis))),
        ];
        if let Err(e) =
            self.stack.execute(&mut self.doc, Box::new(Batch::new("axis.restand", commands)))
        {
            self.warn(e.to_string());
            return false;
        }
        self.select_curve(id);
        self.rebuild_curve_display();
        self.mark_changed();
        self.say(format!("{name} now stands on {}", self.body_label(axis.body)));
        true
    }

    /// Where the selected axis's seed sits, and where that axis stands — the two ends of the
    /// leader the viewport draws between them.
    ///
    /// `None` unless a seeded surface axis is selected: an axis placed by a ray has no seed to
    /// show, and an unselected one is not the number anybody is dragging.
    ///
    /// **Read off the curve rather than solved**, because this runs every frame. The axis's two
    /// stored points are already derived and already current — `refresh_surface_axes` writes them
    /// whenever anything moves — so the first of them is where it stands, and solving again here
    /// would compile a field sixty times a second to learn what the document already says.
    pub(crate) fn selected_axis_seed(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        let id = self.selected_curve?;
        let axis = self.doc.surface_axes.get(&id)?;
        // Only a seeded axis has anything to draw; a ray's place is its U and V.
        axis.seed?;
        let graph = self.graph_holding(axis.body)?;
        let sub = graph.subtree(axis.body)?;
        let (lo, hi) = fieldform_field::graph_extent(&self.doc, &sub, axis.body).as_box()?;
        let at = axis.seed_point([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z])?;
        let stands = *self.doc.curve(id)?.control_points().first()?;
        Some((Point3::new(at[0], at[1], at[2]), stands))
    }

    /// The controls of an axis on a surface, in place of the X, Y and Z of a plain line.
    ///
    /// **No coordinates, because the body decides them.** What is editable is what the axis owns:
    /// which way it looks, U and V across the body's box, and its length. A drag changes the record
    /// and the line live and puts *one* entry on the history when the hand comes off, the rule every
    /// drag here follows; the direction buttons are single steps. Written without a range on the
    /// widget — a range rewrites the value every frame and reports it as an edit (the DragValue
    /// clamp trap) — the values are kept in range where they are made.
    pub(crate) fn surface_axis_controls(&mut self, ui: &mut egui::Ui, id: CurveId, axis: SurfaceAxis) {
        let mut edited = axis;
        let (mut began, mut touched, mut settled, mut stepped) = (false, false, false, false);
        let (u_axis, v_axis) = {
            let (a, b) = axis.across();
            (Axis::ALL[a].label(), Axis::ALL[b].label())
        };
        let seeded = edited.seed.is_some();
        ui.label(
            egui::RichText::new(if seeded {
                "Stands on a surface, at the point nearest its seed. The seed is held across the \
                 body's box — 0 to 1 spans it, outside is allowed — so it moves with the body."
            } else {
                "Stands on a surface, where a ray along one axis first meets it. Its place is a U \
                 and a V."
            })
            .weak(),
        );
        // **The body may be gone**, and an axis whose body is gone still draws from its stored
        // points but has nothing to re-derive from. Said on the row, with the way out beside it.
        let body_is_there = self.graph_holding(axis.body).is_some();
        ui.horizontal(|ui| {
            row_label(ui, "On:");
            if body_is_there {
                ui.label(egui::RichText::new(self.body_label(axis.body)).italics());
            } else {
                ui.label(
                    egui::RichText::new("the body it stood on is gone")
                        .italics()
                        .color(ui.visuals().warn_fg_color),
                );
            }
        });
        ui.horizontal(|ui| {
            row_label(ui, "");
            if ui
                .button(if body_is_there { "Stand it elsewhere…" } else { "Stand it on a surface…" })
                .on_hover_text(
                    "Click a surface: this axis stands there instead, keeping its name and its \
                     length. A body may be deleted and the axis pointed at another one — it does \
                     not have to be remade.",
                )
                .clicked()
            {
                self.begin_axis_placement_for(AxisKind::Normal, Some(id));
            }
        });
        // **Two ways to say where, and the choice is on the row.** The seed reaches every face and
        // every shape; the ray gives a predictable grid on a flat one and cannot reach a face
        // parallel to itself. Switching keeps the other's numbers, so it is a view of the same
        // axis rather than a new one -- a seed turned off falls back to the U and V recorded
        // beside it, and turned on again seeds from where the axis currently stands.
        ui.horizontal(|ui| {
            row_label(ui, "Placed by:");
            if ui
                .add(mode_button(seeded, "Nearest point"))
                .on_hover_text(
                    "A seed, and the surface point nearest it. Reaches every face of any body \
                     — a torus, the inside of a shell — and has no poles. Near the middle of a \
                     thin wall two sides are equally near, and nudging the seed swaps them.",
                )
                .clicked()
                && !seeded
            {
                // **Seeded where the ray came from**, so switching lands the axis where it
                // already stands rather than somewhere else. `u` and `v` run across two axes the
                // *direction* chooses, not across X and Y, so they are placed by index; the third
                // goes just outside the box on the side the ray starts from, and the nearest
                // surface point to that is the one the ray would have met.
                let (a, b) = edited.across();
                let d = edited.direction_index();
                let mut from_ray = [0.5; 3];
                from_ray[a] = edited.u;
                from_ray[b] = edited.v;
                from_ray[d] = if edited.flipped { -0.1 } else { 1.1 };
                edited.seed = Some(from_ray);
                stepped = true;
            }
            if ui
                .add(mode_button(!seeded, "Ray (U, V)"))
                .on_hover_text(
                    "A ray along one axis through a U and a V of the body's box. Even across a \
                     flat face, and cannot reach a face parallel to the ray — on a cube with the \
                     ray along Z, never a side.",
                )
                .clicked()
                && seeded
            {
                edited.seed = None;
                stepped = true;
            }
        });
        // **Asked again, because the row above may just have changed it.** `seeded` was read
        // before the two buttons were drawn, and the Ray button clears the seed on the very frame
        // it is clicked -- so the rows below were still taking the seeded branch, reading
        // `edited.seed.unwrap_or([0.5; 3])` and *writing that back*. Switching to the ray
        // therefore planted a seed at the middle of the body and switched straight back, which on
        // a sphere is the one place with no nearest surface point at all: every direction is
        // equally near, and the panel said so in red. Found by driving the panel rather than by
        // reading it.
        let seeded = edited.seed.is_some();

        // **Only where it means something.** A direction and a side are the *ray's* — a seeded
        // axis is placed by the nearest surface point and nothing here moves it, so showing the
        // row said these numbers mattered when they did not. The values are kept, so switching
        // back to the ray finds the direction it had.
        if !seeded {
            ui.horizontal(|ui| {
                row_label(ui, "Looking along:");
                for direction in Axis::ALL {
                    if ui
                        .add(mode_button(edited.direction == direction, direction.label()))
                        .on_hover_text(
                            "The axis the ray runs along; U and V run across the other two",
                        )
                        .clicked()
                    {
                        edited.direction = direction;
                        stepped = true;
                    }
                }
                if ui
                    .add(mode_button(edited.flipped, "from minus"))
                    .on_hover_text("Come at the body from its minus side instead of its plus side")
                    .clicked()
                {
                    edited.flipped = !edited.flipped;
                    stepped = true;
                }
            });
        }
        let mut number = |ui: &mut egui::Ui, label: &str, value: &mut f64, speed: f64, suffix: &str, hover: &str| {
            ui.horizontal(|ui| {
                row_label(ui, label);
                let response = ui
                    .add(
                        egui::DragValue::new(value)
                            .speed(speed)
                            .max_decimals(4)
                            .suffix(suffix)
                            .clamp_existing_to_range(false),
                    )
                    .on_hover_text(hover);
                began |= response.drag_started() || response.gained_focus();
                touched |= response.changed();
                settled |= response.drag_stopped() || response.lost_focus();
            });
        };
        if seeded {
            // The seed's own three, which are what places it. Across the box, which is why they
            // are fractions rather than millimetres and why they survive the body moving.
            let mut seed = edited.seed.unwrap_or([0.5; 3]);
            for (i, name) in ["Seed X:", "Seed Y:", "Seed Z:"].into_iter().enumerate() {
                number(
                    ui,
                    name,
                    &mut seed[i],
                    0.005,
                    "",
                    "0 to 1 across the body's box. The axis stands at the surface point nearest \
                     this, so it may sit outside the body or inside it",
                );
            }
            edited.seed = Some(seed);
            // **Said when the seed has left the box**, which is allowed and is not obvious: the
            // rows read "0 to 1" and a value of 1.95 looks like a number that has gone wrong. It
            // has not — a seed outside the body still has a nearest surface point, and that is
            // where the axis stands — but nothing said so, and a drag passes 1 without a nudge.
            if seed.iter().any(|s| !(0.0..=1.0).contains(s)) {
                ui.label(
                    egui::RichText::new(
                        "the seed is outside the body's box — still fine: the axis stands at the \
                         nearest point of the surface to it",
                    )
                    .weak(),
                );
            }
        } else {
            number(
                ui,
                "U:",
                &mut edited.u,
                0.005,
                "",
                &format!("0 to 1 across the body's box, along {u_axis}. A projection, not the surface's own parameter"),
            );
            number(
                ui,
                "V:",
                &mut edited.v,
                0.005,
                "",
                &format!("0 to 1 across the body's box, along {v_axis}"),
            );
        }
        number(ui, "Length:", &mut edited.length, 0.5, " mm", "How far the line reaches from the surface, along the field's direction");
        if let Some(note) = self.surface_axis_notes.get(&id) {
            ui.colored_label(ui.visuals().warn_fg_color, note.clone());
        }

        // Kept in range where it is made rather than by the widget (see above), and never below a
        // length a line can have.
        edited.u = edited.u.clamp(0.0, 1.0);
        edited.v = edited.v.clamp(0.0, 1.0);
        edited.length = edited.length.max(0.001);

        if began && self.surface_axis_edit_before.is_none() {
            self.surface_axis_edit_before = Some((id, axis));
        }
        if edited != axis {
            // Live: the record and the line follow the hand.
            self.doc.surface_axes.insert(id, edited);
            self.refresh_surface_axis(id, edited);
            self.rebuild_curve_display();
            self.mark_changed();
        }
        // A step is one entry at once; a drag is one entry when it ends.
        let commit = if stepped && edited != axis {
            Some(axis)
        } else if settled {
            self.surface_axis_edit_before.take().filter(|(c, _)| *c == id).map(|(_, before)| before)
        } else {
            None
        };
        if let Some(before) = commit {
            let now = self.doc.surface_axes.get(&id).copied().unwrap_or(edited);
            if before != now {
                let command = Box::new(SetSurfaceAxis::new(id, Some(before), Some(now)));
                if let Err(e) = self.stack.execute(&mut self.doc, command) {
                    self.warn(e.to_string());
                }
                self.mark_changed();
                self.rebuild_render_resources();
            }
        }
    }

    /// A block a split can cut: a field with volume. A point has none, and a plane is the usual
    /// cutter rather than the thing cut.
    fn can_be_split(&self, id: NodeId) -> bool {
        self.part_graph()
            .node(id)
            .is_some_and(|n| n.is_body() && !matches!(n, Node::Plane { .. } | Node::Point { .. }))
    }

    /// A console argument that names a block, if it was given.
    fn argument_block(&self, args: &Args, key: &str) -> Result<Option<NodeId>, String> {
        match args.get(key) {
            None => Ok(None),
            Some(Value::Str(name)) => self
                .node_named(name)
                .map(Some)
                .ok_or_else(|| format!("no block is called {name:?}")),
            Some(other) => Err(format!("{key}= wants a block's name, not {other:?}")),
        }
    }

    /// `split`: cuts a body in two with a plane or another body, as one step.
    ///
    /// `Modify > Split` and `split body=<name> by=<name>` — both optional, the body falling
    /// back to the selection and the tool to the newest plane the part holds. **With no tool at
    /// all it makes one**, a plane through the middle of the body, for the reason Attract draws a
    /// line when the part has none: needing something is not a reason to refuse to make the block,
    /// and it is editable in its own block straight away.
    ///
    /// The halves are named `<body> A` (the tool's solid side) and `<body> B`, because two
    /// booleans reading "Intersect" and "Subtract" do not say which half is which. That is the one
    /// place the application names blocks on the user's behalf, and it is asked for by the
    /// spec. Naming goes through `set_body_name` like every other name, and so — as for a point —
    /// onto the undo stack, as two steps of its own after the split's: Ctrl+Z takes back the names
    /// first and then the halves, and leaves no name behind.
    pub(crate) fn split_body(&mut self, args: &Args) {
        let (asked_body, asked_tool) =
            match (self.argument_block(args, "body"), self.argument_block(args, "by")) {
                (Ok(body), Ok(tool)) => (body, tool),
                (Err(why), _) | (_, Err(why)) => {
                    self.warn(why);
                    return;
                }
            };
        let body = asked_body
            .or_else(|| self.selected_body.filter(|id| self.can_be_split(*id)))
            .or_else(|| {
                self.doc
                    .listed_bodies(self.part_graph())
                    .into_iter()
                    .find(|id| self.can_be_split(*id))
            });
        let Some(body) = body else {
            self.warn("Needs a body to split");
            return;
        };
        if !self.can_be_split(body) {
            self.warn(format!(
                "{} is not something a split can cut — it needs a body with volume",
                self.body_label(body)
            ));
            return;
        }

        if let Some(tool) = asked_tool {
            if tool == body {
                self.warn("A body cannot split itself — name a plane or another body");
                return;
            }
            match self.part_graph().node(tool) {
                Some(Node::Point { .. }) => {
                    self.warn("A point encloses no volume to split by — use a plane or a body");
                    return;
                }
                Some(n) if n.is_body() => {}
                _ => {
                    self.warn(format!("{} is not a field a split can use", self.body_label(tool)));
                    return;
                }
            }
        }
        // The newest plane the part holds and nothing consumes, which is the one just made for
        // the purpose.
        let tool = asked_tool.or_else(|| {
            self.part_graph()
                .roots()
                .into_iter()
                .filter(|id| {
                    *id != body && matches!(self.part_graph().node(*id), Some(Node::Plane { .. }))
                })
                .max_by_key(|id| (self.doc.made_at(*id), id.raw()))
        });

        let through = match self.body_extent(body).as_box() {
            Some((lo, hi)) => Point3::from((lo.coords + hi.coords) * 0.5),
            None => Point3::origin(),
        };
        let mut graph = self.part_graph().clone();
        let split = split_in(&mut graph, body, tool, through);
        // Checked before it is committed, as a boolean's is: a refusal leaves the part exactly as
        // it was, with no plane and no half-built pair.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }

        let label = self.body_label(body);
        let tool_label = self.body_label(split.tool);
        let (a, b) = {
            let taken: std::collections::HashSet<&str> =
                self.doc.node_meta.values().filter_map(|m| m.name.as_deref()).collect();
            half_names(&taken, &label)
        };
        // One graph edit: the plane (if there is one) and both halves arrive together, so one
        // Ctrl+Z takes all three away.
        self.set_part_graph(graph);
        self.set_body_name(split.inside, &a);
        self.set_body_name(split.outside, &b);
        // Both consumed now, so neither is a body any more and a hidden flag on either would be a
        // switch on an interior node that nothing drawn can turn back on (`apply_boolean`).
        self.clear_hidden(body);
        self.clear_hidden(split.tool);
        self.select_body(Some(split.inside));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(if split.made_tool {
            format!(
                "Split {label} into “{a}” and “{b}” along a new plane through its middle — \
                 move the plane in its block to move the cut"
            )
        } else {
            format!("Split {label} into “{a}” and “{b}” along {tool_label}")
        });
    }

    /// The next free `Axis n`.
    ///
    /// Numbered from one rather than left bare the first time, so the first axis is not the
    /// odd one out in a list of three; the rest of the curves' naming (`unique_curve_name`)
    /// leaves a bare first name, which is right for "Line" and wrong for a word that is nearly
    /// always used more than once.
    fn next_axis_name(&self) -> String {
        (1..)
            .map(|n| format!("Axis {n}"))
            .find(|name| !self.doc.curves.values().any(|c| &c.name == name))
            .expect("an unbounded range has a free name")
    }

    /// Makes the axis: a straight line, one undo step, named `Axis n`.
    ///
    /// **No new type** — a line already is an axis, and everything that accepts one (`Ends`, a
    /// revolve's or a twist's frame taken from a line, a pattern's axis) accepts this unchanged.
    fn add_axis(&mut self, from: Point3<f64>, to: Point3<f64>) -> Option<CurveId> {
        if (to - from).norm() < 1e-6 {
            self.warn("An axis needs two different points — those are the same place");
            return None;
        }
        let id = CurveId::new();
        let mut curve = match bezier_between(id, 1, from, to) {
            Ok(curve) => curve,
            Err(e) => {
                self.warn(format!("could not make an axis: {e}"));
                return None;
            }
        };
        curve.name = self.next_axis_name();
        let name = curve.name.clone();
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(SetCurve::insert(curve))) {
            self.warn(e.to_string());
            return None;
        }
        self.select_curve(id);
        self.rebuild_curve_display();
        self.mark_changed();
        self.say(format!("Added {name}, {:.1} mm long", (to - from).norm()));
        Some(id)
    }

    /// An axis along a cylinder's, cone's or capsule's own axis.
    ///
    /// A copy of the two points rather than a reference to them: `Ends::From` is how a block
    /// *follows* a line, and this is the other direction — a line is being made from a block —
    /// so it is a plain line the block's later edits do not move. The ends are read the way the
    /// bounds walk reads them (`ends_at`), so a block whose ends stand on a plane or follow a
    /// curve gives the same two points the field is built from.
    fn axis_through(&mut self, id: NodeId) {
        let Some(node) = self.part_graph().node(id) else {
            self.warn("That block is not in the part");
            return;
        };
        let Some(ends) = node.ends().first().copied() else {
            self.warn(format!(
                "{} has no axis of its own — a cylinder, a cone and a capsule do",
                self.body_label(id)
            ));
            return;
        };
        let numbers = |r: &ParamRef| self.resolve(*r);
        let (from, to) = fieldform_field::ends_at(ends, &self.doc, self.part_graph(), &numbers);
        self.add_axis(from, to);
    }

    /// An axis stood on the surface at `at`, leaving along the field's direction there — a *derived*
    /// axis, with a U, a V and a length rather than points of its own.
    fn axis_normal_at(&mut self, at: Point3<f64>, length: f64) -> bool {
        self.stand_axis_on_surface_for(at, length, None)
    }

    /// Where the field's surface is under the cursor, or `None`.
    ///
    /// The field itself, whatever the view draws: the normal is the field's, and a hit on the cage
    /// would be a point the field's gradient says nothing true about — the reason picking never
    /// falls back from one to the other (`pick_pivot`). The same trace, with the same budget.
    fn field_hit_under(
        &self,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Option<Point3<f64>> {
        let field = self.pick_field.as_ref()?;
        let (origin, dir) = self.camera.pick_ray(aspect, pos, rect);
        let bounds = self.pick_bounds();
        let radius = ((bounds.1 - bounds.0).norm() * 0.5).max(1e-6);
        let eye = Point3::new(origin.x as f64, origin.y as f64, origin.z as f64);
        let d = Vector3::new(dir.x as f64, dir.y as f64, dir.z as f64).try_normalize(1e-12)?;
        trace_field(field, eye, d, self.trace_reach(eye, bounds), radius * 1e-4, 512)
    }

    /// Whether the view draws the field, so that clicking a surface means the field's surface.
    pub(crate) fn field_is_drawn(&self) -> bool {
        self.view_mode != ViewMode::Polygon
    }

    /// Begins an axis the viewport will finish: two clicks, or one on a surface.
    fn begin_axis_placement(&mut self, kind: AxisKind) {
        self.begin_axis_placement_for(kind, None);
    }

    /// The same, re-pointing `restanding` instead of making a new axis when one is given.
    fn begin_axis_placement_for(&mut self, kind: AxisKind, restanding: Option<CurveId>) {
        if kind == AxisKind::Normal && !self.field_is_drawn() {
            self.warn(
                "A normal is the field's, and Polygon view draws the cage instead — press Tab \
                 for Field or Overlay, then click the surface",
            );
            return;
        }
        self.pending_curve = Some(PendingCurve {
            placed: 0,
            degree: 1,
            first: None,
            preview: None,
            axis: Some(kind),
            restanding,
        });
        self.say(match (kind, restanding.is_some()) {
            (AxisKind::TwoPoints, _) => {
                "Axis — click the start point, then the end point. Esc cancels"
            }
            (AxisKind::Normal, false) => "Axis — click the surface it should stand on. Esc cancels",
            (AxisKind::Normal, true) => {
                "Click the surface this axis should stand on — it keeps its name and its length. \
                 Esc cancels"
            }
        });
    }

    /// The one click of a normal-to-surface axis. A click that finds no surface keeps waiting.
    pub(crate) fn place_normal_axis(&mut self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) {
        let Some(at) = self.field_hit_under(aspect, rect, pos) else {
            self.warn("No surface under the cursor — click on the body itself");
            return;
        };
        // The length is the one the axis already has when it is being re-pointed: a restand moves
        // where it stands, and nothing else about it.
        let restanding = self.pending_curve.as_ref().and_then(|p| p.restanding);
        let length = restanding
            .and_then(|id| self.doc.surface_axes.get(&id).map(|a| a.length))
            .unwrap_or(AXIS_NORMAL_LENGTH_MM);
        if self.stand_axis_on_surface_for(at, length, restanding) {
            self.pending_curve = None;
        }
    }

    /// The point a normal-axis preview marker should sit at: the surface, not whatever a curve
    /// point would land on.
    pub(crate) fn normal_axis_landing(
        &self,
        aspect: f32,
        rect: egui::Rect,
        cursor: egui::Pos2,
    ) -> Option<Point3<f64>> {
        self.field_hit_under(aspect, rect, cursor)
    }

    /// `axis`: a straight line to use as an axis, made one of four ways.
    ///
    /// | | |
    /// |---|---|
    /// | `axis` | click two points |
    /// | `axis from=0,0,0 to=0,0,50` | two typed points |
    /// | `axis through=Cylinder` or `-through` | copy a cylinder's, cone's or capsule's own axis (by name, or the selected one) |
    /// | `axis normal=10,0,5` or `-normal` | stand on the surface at a typed point, or one clicked, leaving along the field's direction there; `length=` says how far, 50 by default. **Derived**: it has a U, a V and a length, not points |
    /// | `axis on=Sphere u=0.25 v=0.75 dir=z length=30` | the same, by its numbers: a body, a place across its box, the axis the ray runs along (`-flip` from the minus side) |
    /// | `axis on=Sphere seed=0.5,0.5,1.2` | stood at the surface point *nearest* a seed, which is where a clicked one stands. Three numbers across the body's box; outside 0 to 1 is allowed |
    ///
    /// `key=value` and `-flag`, which is the console's grammar — the spec's `axis --through cyl2`
    /// and bare coordinates are not words this console parses, and one grammar is worth more than
    /// matching the sketch.
    pub(crate) fn axis_command(&mut self, args: &Args) {
        let point = |t: [f64; 3]| Point3::new(t[0], t[1], t[2]);

        match (args.get("from"), args.get("to")) {
            (Some(a), Some(b)) => {
                match (triple(a), triple(b)) {
                    (Some(a), Some(b)) => {
                        self.add_axis(point(a), point(b));
                    }
                    _ => self.warn(
                        "from= and to= each want three numbers, like `axis from=0,0,0 to=0,0,50`",
                    ),
                }
                return;
            }
            (None, None) => {}
            _ => {
                self.warn("An axis between two points needs both: `axis from=0,0,0 to=0,0,50`");
                return;
            }
        }

        match args.get("through") {
            Some(Value::Str(name)) => {
                match self.node_named(name) {
                    Some(id) => self.axis_through(id),
                    None => self.warn(format!("no block is called {name:?}")),
                }
                return;
            }
            Some(Value::Bool(true)) => {
                match self.selected_body {
                    Some(id) => self.axis_through(id),
                    None => self.warn("Select a cylinder, cone or capsule first, or name one: `axis through=Cylinder`"),
                }
                return;
            }
            Some(other) => {
                self.warn(format!("through= wants a block's name, not {other:?}"));
                return;
            }
            None => {}
        }

        // `axis on=Sphere u=0.25 v=0.75 dir=z length=30`: a body, a place on it, a length — what
        // the panel edits, and so what a script can set. `-flip` comes from the minus side.
        if let Some(on) = args.get("on") {
            let Value::Str(name) = on else {
                self.warn("on= wants a block's name, like `axis on=Sphere`");
                return;
            };
            let Some(body) = self.node_named(name) else {
                self.warn(format!("no block is called {name:?}"));
                return;
            };
            let number = |key: &str, default: f64| {
                args.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
            };
            let direction = match args.get("dir").and_then(|v| v.as_str()) {
                None => Axis::Z,
                Some(text) => match Axis::ALL.into_iter().find(|a| a.label().eq_ignore_ascii_case(text)) {
                    Some(axis) => axis,
                    None => {
                        self.warn("dir= is the axis the ray runs along: x, y or z");
                        return;
                    }
                },
            };
            let length = number("length", SurfaceAxis::DEFAULT_LENGTH_MM);
            if length <= 0.0 {
                self.warn("length= must be more than nought");
                return;
            }
            let flipped = matches!(args.get("flip"), Some(Value::Bool(true)));
            // **`seed=` is the other placement, said in numbers.** Until this it could only be
            // reached by clicking a surface, so the one the application *defaults* to had no way
            // in from a script and no way to be set exactly -- which is backlog item 81's
            // complaint about the create verbs in its smallest form, and it bit while testing:
            // the nearest-point placement could not be exercised without a hand on the mouse.
            //
            // Three numbers across the body's box, the same the panel's rows hold. Outside 0 to 1
            // is allowed and not clamped, for the reason the rows are not: a seed outside the body
            // still has a nearest surface point, and that is where the axis stands.
            let seed = match args.get("seed") {
                None => None,
                Some(value) => match triple(value) {
                    Some(at) => Some(at),
                    None => {
                        self.warn(
                            "seed= wants three numbers across the body's box, like \
                             `axis on=Sphere seed=0.5,0.5,1.2`",
                        );
                        return;
                    }
                },
            };
            let axis = SurfaceAxis {
                body,
                direction,
                flipped,
                u: number("u", 0.5).clamp(0.0, 1.0),
                v: number("v", 0.5).clamp(0.0, 1.0),
                length,
                // **Typed U and V mean the ray**, which is the whole reason the ray is kept: it is
                // the one that gives a predictable grid on a flat face, and somebody writing
                // `u=0.25 v=0.5` is asking for exactly that. A `seed=` beside them says the other
                // thing, and wins -- the U and V are still recorded, so switching the panel to the
                // ray finds them.
                seed,
            };
            self.make_surface_axis(axis);
            return;
        }

        match args.get("normal") {
            Some(Value::Bool(true)) => {
                self.begin_axis_placement(AxisKind::Normal);
                return;
            }
            Some(value) => {
                let Some(at) = triple(value) else {
                    self.warn("normal= wants a point, like `axis normal=10,0,5`, or use -normal and click");
                    return;
                };
                let length = args
                    .get("length")
                    .and_then(|v| v.as_f64())
                    .filter(|l| *l > 0.0)
                    .unwrap_or(AXIS_NORMAL_LENGTH_MM);
                self.axis_normal_at(point(at), length);
                return;
            }
            None => {}
        }

        self.begin_axis_placement(AxisKind::TwoPoints);
    }

    /// Whether the selection is something an axis can be drawn through.
    ///
    /// **Unused while "through the selected solid" is hidden from the menu** (owner, 2026-10-02:
    /// "for now please hide it, and make a note that it's hidden"). Kept, with its entry's wiring
    /// one line away, because it works: `axis -through` and `axis through=<name>` still do.
    /// Backlog item 70 carries the note.
    #[allow(dead_code)]
    pub(crate) fn selected_has_an_axis(&self) -> bool {
        self.selected_body
            .and_then(|id| self.part_graph().node(id))
            .is_some_and(|n| !n.ends().is_empty())
    }

    /// What a Ctrl-click while placing a curve lands on: the nearest curve end or point block in
    /// reach, by screen distance, and a phrase for the log.
    ///
    /// **A point block as well as a curve end**, asked for on 2026-10-02 ("select existing points in
    /// the viewport using Ctrl to snap, like line"). Line only ever snapped to curve ends, so an
    /// axis — and a line — could not be started from a point somebody had placed. The points are
    /// `point_positions`, the same list the dots are drawn from, so a snap goes where a dot is.
    pub(crate) fn placement_snap(
        &self,
        aspect: f32,
        rect: egui::Rect,
        cursor: egui::Pos2,
    ) -> Option<(Point3<f64>, String)> {
        let on_screen = |at: Point3<f64>| {
            self.camera
                .project_to_screen(aspect, rect, at.cast::<f32>())
                .map(|screen| screen.distance(cursor))
        };
        let curve_end = self.endpoint_near(aspect, rect, cursor, None).map(|(id, at)| {
            let name = self.doc.curve(id).map_or_else(|| "a curve".to_string(), |c| c.name.clone());
            (on_screen(at).unwrap_or(0.0), at, format!("the end of {name}"))
        });
        let point = self
            .point_positions()
            .into_iter()
            .filter_map(|(id, at)| {
                let d = on_screen(at)?;
                (d <= SNAP_PIXELS).then(|| (d, at, self.body_label(id)))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        [curve_end, point]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, at, what)| (at, what))
    }

    /// The next axis's name, for the placement code that makes the curve itself.
    pub(crate) fn axis_name(&self) -> String {
        self.next_axis_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_scene::Document;

    fn sphere_graph() -> (Graph, NodeId) {
        let mut graph = Graph::empty();
        let id = graph.add(PrimitiveKind::Sphere.node([ParamRef::Literal(0.0); 3], 40.0));
        (graph, id)
    }

    fn compile(graph: &Graph) -> GraphField {
        GraphField::compile_graph(&Document::new(), graph, 1e-3).expect("the graph compiles")
    }

    /// A sphere of radius 20 at the origin, as a body, in a graph of its own.
    fn sphere_at(centre: [f64; 3]) -> (Graph, NodeId) {
        let mut graph = Graph::empty();
        let id = graph.add(PrimitiveKind::Sphere.node(centre.map(ParamRef::Literal), 40.0));
        (graph, id)
    }

    fn stand(body: NodeId, direction: Axis, flipped: bool, u: f64, v: f64) -> SurfaceAxis {
        SurfaceAxis { body, direction, flipped, u, v, length: 30.0, seed: None }
    }

    fn solve(graph: &Graph, axis: &SurfaceAxis) -> Result<[Point3<f64>; 2], String> {
        solve_surface_axis(&Document::new(), graph, axis, &mut LeafCache::default())
    }

    /// A click on a sphere makes a seed inside the box, and the axis stands where it was clicked.
    ///
    /// **Asked because the panel showed a seed of 1.610, 1.950, 0.9907** on a fresh axis, which is
    /// outside the 0-to-1 the row says it spans. Either creation produces one out of range, which
    /// would be a fault here, or the numbers had been dragged -- and only a case can say which.
    #[test]
    fn a_click_on_a_sphere_seeds_inside_the_box_and_stands_where_it_was_clicked() {
        let (graph, body) = sphere_at([0.0; 3]);
        let doc = Document::new();
        let mut leaves = LeafCache::default();
        // A point on the sphere: radius 20 at the origin, so the +X pole.
        let clicked = Point3::new(20.0, 0.0, 0.0);
        let axis = surface_axis_through(&doc, &graph, body, clicked, 30.0, &mut leaves)
            .expect("a click on the surface places an axis");

        let seed = axis.seed.expect("creation seeds");
        for (i, s) in seed.iter().enumerate() {
            assert!(
                (-0.01..=1.01).contains(s),
                "seed {i} is {s}, which is outside the box the row says it spans: {seed:?}"
            );
        }

        let [from, _] = solve(&graph, &axis).expect("it stands");
        assert!(
            (from - clicked).norm() < 0.2,
            "it should stand where it was clicked, and stands at {from:?}"
        );
    }

    /// An axis whose body is gone says so, and the same record pointed at another body stands.
    ///
    /// **The state the restand button exists for.** Deleting a body leaves the axis's record in the
    /// document and its two stored points on the curve, so it goes on drawing -- and has nothing
    /// left to re-derive from. Until the button there was no way back, because the record names a
    /// body and nothing could change which: "once you delete the sphere it's then a dumb block ...
    /// you have to delete it and recreate it, I think that's kind of a waste."
    ///
    /// Both halves are asserted, because only the pair says the block is worth keeping: the first
    /// that a missing body is a clean refusal rather than a wrong answer, the second that the very
    /// same record solves again as soon as it names a body that is there.
    #[test]
    fn an_axis_whose_body_is_gone_stands_again_as_soon_as_it_names_another() {
        let (graph, body) = sphere_at([0.0; 3]);
        let stood = SurfaceAxis { seed: Some([0.5, 0.5, 1.2]), ..stand(body, Axis::Z, false, 0.5, 0.5) };
        let [from, _] = solve(&graph, &stood).expect("it stands while its body is there");
        assert!((from.z - 20.0).abs() < 0.2, "on the sphere's pole: {from:?}");

        // The body deleted: the record is untouched and now names a node the graph does not hold.
        let empty = Graph::empty();
        let why = solve(&empty, &stood).expect_err("a missing body cannot be stood on");
        assert!(
            why.contains("gone"),
            "the refusal should say the body is gone, and says {why:?}"
        );

        // Another body, in another place: the same record, re-pointed, stands on it. This is what
        // the button does, and the axis keeps everything else it had.
        let (elsewhere, other) = sphere_at([100.0, 0.0, 0.0]);
        let restood = SurfaceAxis { body: other, ..stood };
        let [moved, tip] = solve(&elsewhere, &restood).expect("it stands on the new body");
        assert!(
            (moved.x - 100.0).abs() < 0.3 && (moved.z - 20.0).abs() < 0.3,
            "on the new sphere's pole: {moved:?}"
        );
        assert!(
            ((tip - moved).norm() - stood.length).abs() < 1e-6,
            "and keeps the length it had"
        );
    }

    /// A seed given in numbers places the axis exactly where a clicked one would.
    ///
    /// **The placement the application defaults to had no way in except a mouse.** `axis -normal`
    /// seeds, and needs a click; `axis on=... u= v=` takes the ray. So the nearest-point form
    /// could not be scripted, could not be set exactly, and could not be tested without a hand on
    /// the viewport -- which is backlog item 81's complaint in its smallest form, and it is what
    /// stopped this being exercised when the panel was driven.
    ///
    /// Asserted against the click, not against itself: a point on the surface, seeded two ways,
    /// has to stand in the same place. A case that only checked `seed=` parsed would pass over a
    /// verb that placed the axis somewhere else entirely.
    #[test]
    fn a_seed_given_in_numbers_stands_where_the_same_seed_clicked_would() {
        let (graph, body) = sphere_at([0.0; 3]);
        let doc = Document::new();
        let mut leaves = LeafCache::default();

        // What a click on the +X pole makes.
        let clicked = Point3::new(20.0, 0.0, 0.0);
        let from_click = surface_axis_through(&doc, &graph, body, clicked, 30.0, &mut leaves)
            .expect("a click places an axis");
        let seed = from_click.seed.expect("a click seeds");

        // The same seed, as the console hands it over: three numbers across the box, straight into
        // the record, with no click anywhere.
        let from_numbers = SurfaceAxis { seed: Some(seed), ..stand(body, Axis::Z, false, 0.5, 0.5) };

        let [clicked_at, _] = solve(&graph, &from_click).expect("stands");
        let [typed_at, _] = solve(&graph, &from_numbers).expect("stands");
        assert!(
            (clicked_at - typed_at).norm() < 1e-6,
            "the two placements should be the same point: {clicked_at:?} against {typed_at:?}"
        );
        assert!((clicked_at - clicked).norm() < 0.2, "and both on the pole that was clicked");

        // **Outside the box is allowed and is not clamped**, which is the half a parser is most
        // likely to get wrong: a seed beyond 1 still has a nearest surface point, and refusing it
        // would make the verb narrower than the rows it mirrors.
        let far = SurfaceAxis { seed: Some([0.5, 0.5, 4.0]), ..from_numbers };
        let [from_far, _] = solve(&graph, &far).expect("a far seed still has a nearest point");
        assert!((from_far.z - 20.0).abs() < 0.2, "on the +Z pole: {from_far:?}");
    }

    /// A seed stands the axis at the nearest surface point, on a face no ray along Z can reach.
    ///
    /// **The case the seed exists for.** A ray along Z meets the top and the bottom of a cube and
    /// never a side, at any `u` and `v` — the side faces are parallel to it. A seed just outside a
    /// side face has that face as its nearest surface, so the axis stands there and leaves along
    /// the face's own normal.
    ///
    /// The control is the same seed with the ray asked instead: it lands on the top, which is what
    /// "cannot reach a face parallel to the ray" means and what makes this case worth having.
    #[test]
    fn a_seed_reaches_a_face_no_ray_along_that_axis_can() {
        let mut graph = Graph::empty();
        let lit = ParamRef::Literal;
        let body = graph.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        // The box is -20..20 in each axis, so 0..1 across it puts the +X face at 1.0 and the
        // middle at 0.5. A seed just outside the +X face, level with the centre.
        let seeded = SurfaceAxis {
            seed: Some([1.1, 0.5, 0.5]),
            ..stand(body, Axis::Z, false, 0.5, 0.5)
        };
        let [from, to] = solve(&graph, &seeded).expect("a seed always has a nearest point");
        assert!(
            (from.x - 20.0).abs() < 0.2 && from.y.abs() < 0.2 && from.z.abs() < 0.2,
            "it should stand on the +X face, and stands at {from:?}"
        );
        assert!((to.x - from.x) > 25.0, "and leave along that face's normal: {to:?}");

        // The control: the very same record with the seed taken away is the ray, and the ray
        // along Z cannot see a side face at all -- it lands on the top.
        let [ray_from, _] = solve(&graph, &stand(body, Axis::Z, false, 0.5, 0.5))
            .expect("the ray still finds the top");
        assert!(
            (ray_from.z - 20.0).abs() < 0.2,
            "a ray along Z meets the top, never a side: {ray_from:?}"
        );
    }

    /// A seed moves with the body, which is the whole reason it is held across the box.
    #[test]
    fn a_seeded_axis_follows_the_body_it_stands_on() {
        let lit = ParamRef::Literal;
        let stood = |x: f64| {
            let mut graph = Graph::empty();
            let body = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(x), lit(0.0), lit(0.0)]),
                size: [lit(40.0); 3],
                turn: fieldform_scene::no_turn(),
            });
            let axis = SurfaceAxis {
                seed: Some([0.5, 0.5, 1.1]),
                ..stand(body, Axis::Z, false, 0.5, 0.5)
            };
            solve(&graph, &axis).expect("stands")
        };
        let [here, _] = stood(0.0);
        let [moved, _] = stood(100.0);
        assert!((here.z - 20.0).abs() < 0.2, "on the top face: {here:?}");
        assert!(
            (moved.x - (here.x + 100.0)).abs() < 0.2,
            "the body moved a hundred and the axis went with it: {here:?} then {moved:?}"
        );
    }

    /// The pole of a sphere, from above and from below: the start is on the surface and the line
    /// The pole of a sphere, from above and from below: the start is on the surface and the line
    /// leaves along the radius for exactly its length.
    #[test]
    fn a_surface_axis_stands_on_a_spheres_pole_and_leaves_along_the_radius() {
        let (graph, body) = sphere_at([0.0; 3]);
        let [from, to] = solve(&graph, &stand(body, Axis::Z, false, 0.5, 0.5)).unwrap();
        assert!((from - Point3::new(0.0, 0.0, 20.0)).norm() < 0.05, "{from:?}");
        assert!((to - Point3::new(0.0, 0.0, 50.0)).norm() < 0.05, "{to:?}");

        let [from, to] = solve(&graph, &stand(body, Axis::Z, true, 0.5, 0.5)).unwrap();
        assert!((from - Point3::new(0.0, 0.0, -20.0)).norm() < 0.05, "{from:?}");
        assert!((to - Point3::new(0.0, 0.0, -50.0)).norm() < 0.05, "{to:?}");
        assert!(((to - from).norm() - 30.0).abs() < 1e-6, "the length is the length");
    }

    /// Off-centre, along each axis: the start is on the sphere, the ray's own coordinates are the
    /// ones U and V name, and the line leaves along the radius through that point.
    #[test]
    fn u_and_v_name_the_place_across_the_box_and_the_line_leaves_along_the_radius_there() {
        let (graph, body) = sphere_at([0.0; 3]);
        for (direction, u, v) in [
            (Axis::Z, 0.75, 0.5),
            (Axis::X, 0.25, 0.6),
            (Axis::Y, 0.4, 0.8),
        ] {
            let axis = stand(body, direction, false, u, v);
            let [from, to] = solve(&graph, &axis).unwrap();
            assert!((from.coords.norm() - 20.0).abs() < 0.05, "{direction:?}: on the sphere, {from:?}");
            // The across-coordinates are U and V of the box [-20, 20].
            let (a, b) = axis.across();
            assert!((from[a] - (-20.0 + u * 40.0)).abs() < 1e-6, "{direction:?} u");
            assert!((from[b] - (-20.0 + v * 40.0)).abs() < 1e-6, "{direction:?} v");
            // Along the radius: the line is the start pushed out by its length.
            let out = from.coords.normalize() * 30.0;
            assert!((to - from - out).norm() < 0.05, "{direction:?}: leaves along the radius");
        }
    }

    /// The corner of the box is clear of a sphere, so the ray misses — and says what to do about it
    /// rather than standing the axis somewhere else.
    #[test]
    fn a_ray_that_misses_the_body_says_so() {
        let (graph, body) = sphere_at([0.0; 3]);
        let why = solve(&graph, &stand(body, Axis::Z, false, 0.02, 0.02)).unwrap_err();
        assert!(why.contains("misses"), "{why}");
    }

    #[test]
    fn a_body_that_is_gone_or_has_no_limit_cannot_be_stood_on() {
        let (graph, _) = sphere_at([0.0; 3]);
        let gone = stand(NodeId::from_raw(0xdead), Axis::Z, false, 0.5, 0.5);
        assert!(solve(&graph, &gone).unwrap_err().contains("gone"));

        let mut plane_graph = Graph::empty();
        let plane = plane_graph.add(PrimitiveKind::Plane.node([ParamRef::Literal(0.0); 3], 100.0));
        let why = solve(&plane_graph, &stand(plane, Axis::Z, false, 0.5, 0.5)).unwrap_err();
        assert!(why.contains("no limit"), "a plane reaches forever: {why}");
    }

    /// **It follows its body**, which is the whole point of deriving it: the same numbers on the same
    /// body moved 10 mm along X stand 10 mm along X.
    #[test]
    fn a_surface_axis_follows_its_body() {
        let (home, body) = sphere_at([0.0; 3]);
        let axis = stand(body, Axis::Z, false, 0.5, 0.5);
        let [from, _] = solve(&home, &axis).unwrap();

        // The same node, moved: rebuilt in a graph where it sits elsewhere.
        let mut moved = Graph::empty();
        let id = moved.add(PrimitiveKind::Sphere.node([10.0, 0.0, 0.0].map(ParamRef::Literal), 40.0));
        let [from_moved, _] = solve(&moved, &SurfaceAxis { body: id, ..axis }).unwrap();
        assert!(
            (from_moved - from - Vector3::new(10.0, 0.0, 0.0)).norm() < 0.05,
            "{from:?} -> {from_moved:?}"
        );
    }

    /// A clicked point reads back as the axis that puts the line there: the direction the surface
    /// faces most, from the side it faces, and a U and V that reproduce the click.
    #[test]
    fn a_clicked_point_reads_as_the_ray_that_returns_to_it() {
        let (graph, body) = sphere_at([0.0; 3]);
        let doc = Document::new();
        let mut leaves = LeafCache::default();
        for (point, direction, flipped) in [
            (Point3::new(0.0, 0.0, 20.0), Axis::Z, false),
            (Point3::new(0.0, 0.0, -20.0), Axis::Z, true),
            (Point3::new(20.0, 0.0, 0.0), Axis::X, false),
            (Point3::new(0.0, -20.0, 0.0), Axis::Y, true),
        ] {
            let axis = surface_axis_through(&doc, &graph, body, point, 30.0, &mut leaves).unwrap();
            assert_eq!((axis.direction, axis.flipped), (direction, flipped), "{point:?}");
            let [from, _] = solve_surface_axis(&doc, &graph, &axis, &mut leaves).unwrap();
            assert!((from - point).norm() < 0.05, "{point:?} -> {from:?}");
        }
        // A slanted point still comes back to itself, along whichever axis it faces most.
        let slanted = Point3::from(Vector3::new(1.0, 2.0, -2.0).normalize() * 20.0);
        let axis = surface_axis_through(&doc, &graph, body, slanted, 30.0, &mut leaves).unwrap();
        let [from, _] = solve_surface_axis(&doc, &graph, &axis, &mut leaves).unwrap();
        assert!((from - slanted).norm() < 0.05, "{axis:?}: {from:?}");
    }

    /// Of two bodies, a click belongs to the one whose surface it is on.
    #[test]
    fn a_click_belongs_to_the_body_whose_surface_it_is_on() {
        let mut graph = Graph::empty();
        let left = graph.add(PrimitiveKind::Sphere.node([-60.0, 0.0, 0.0].map(ParamRef::Literal), 40.0));
        let right = graph.add(PrimitiveKind::Sphere.node([60.0, 0.0, 0.0].map(ParamRef::Literal), 40.0));
        let doc = Document::new();
        let mut leaves = LeafCache::default();
        assert_eq!(body_under(&doc, &graph, Point3::new(-60.0, 0.0, 20.0), &mut leaves), Some(left));
        assert_eq!(body_under(&doc, &graph, Point3::new(80.0, 0.0, 0.0), &mut leaves), Some(right));
    }

    #[test]
    fn a_split_is_two_booleans_that_read_one_body_and_one_tool() {
        let (mut graph, body) = sphere_graph();
        let split = split_in(&mut graph, body, None, Point3::origin());

        assert!(split.made_tool, "no tool was given, so one was made");
        assert!(matches!(graph.node(split.tool), Some(Node::Plane { .. })));
        match graph.node(split.inside) {
            Some(Node::Boolean { op: BooleanOp::Intersection, inputs }) => {
                assert_eq!(inputs, &vec![Operand::filled(body), Operand::filled(split.tool)])
            }
            other => panic!("the inside is an intersection of body and tool, got {other:?}"),
        }
        match graph.node(split.outside) {
            Some(Node::Boolean { op: BooleanOp::Difference, inputs }) => {
                assert_eq!(
                    inputs,
                    &vec![Operand::filled(body), Operand::filled(split.tool)],
                    "the same two nodes, not copies — so editing the body edits both halves"
                )
            }
            other => panic!("the outside is a difference of body and tool, got {other:?}"),
        }
    }

    #[test]
    fn after_a_split_the_halves_are_the_only_bodies() {
        let (mut graph, body) = sphere_graph();
        assert_eq!(graph.roots(), vec![body]);
        let split = split_in(&mut graph, body, None, Point3::origin());

        let mut roots = graph.roots();
        roots.sort_by_key(|id| id.raw());
        let mut halves = vec![split.inside, split.outside];
        halves.sort_by_key(|id| id.raw());
        assert_eq!(roots, halves, "the original and the tool are consumed, not deleted");
        assert!(graph.node(body).is_some() && graph.node(split.tool).is_some());
    }

    #[test]
    fn a_given_tool_is_used_and_no_plane_is_made() {
        let (mut graph, body) = sphere_graph();
        let cutter = graph.add(PrimitiveKind::Cube.node([ParamRef::Literal(30.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)], 40.0));
        let before = graph.nodes().count();
        let split = split_in(&mut graph, body, Some(cutter), Point3::origin());
        assert!(!split.made_tool);
        assert_eq!(split.tool, cutter);
        assert_eq!(graph.nodes().count(), before + 2, "two booleans, nothing else");
    }

    /// The halves together are the body: a union of "inside the tool" and "not inside it" is
    /// everything the body was, except on the cut itself, where both halves are open.
    #[test]
    fn a_split_part_still_fills_the_body_it_cut() {
        let (plain, _) = sphere_graph();
        let whole = compile(&plain);

        let (mut graph, body) = sphere_graph();
        // Off-centre, so neither half is the half a symmetric cut would hand back by accident.
        split_in(&mut graph, body, None, Point3::new(0.0, 0.0, 3.7));
        let halves = compile(&graph);
        // The application refuses a graph the shader cannot be generated for (`refuse_reason`),
        // so a split that compiled on the CPU and not for the GPU would be a button that always
        // answers "cannot be drawn" — the plane is the unbounded leaf, so it is the one to ask.
        fieldform_gpu::generate(&Document::new(), &graph)
            .expect("a split, plane and all, can be drawn");

        let mut checked = 0;
        for i in -8..=8 {
            for j in -8..=8 {
                for k in -8..=8 {
                    let p = Point3::new(i as f64 * 4.3, j as f64 * 4.3, k as f64 * 4.3 + 0.01);
                    let (w, h) = (whole.distance(p), halves.distance(p));
                    if (p.z - 3.7).abs() < 1e-3 || w.abs() < 1e-6 {
                        continue;
                    }
                    assert_eq!(w < 0.0, h < 0.0, "{p:?}: the body reads {w}, the two halves {h}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 3_000, "the grid reached {checked} points");
    }

    #[test]
    fn half_names_say_which_is_which_and_move_together_when_taken() {
        let none = std::collections::HashSet::new();
        assert_eq!(half_names(&none, "bracket"), ("bracket A".into(), "bracket B".into()));

        let taken: std::collections::HashSet<&str> = ["bracket A", "bracket B"].into();
        assert_eq!(half_names(&taken, "bracket"), ("bracket_1 A".into(), "bracket_1 B".into()));

        // One of the pair taken is enough to move both: A and B must read as a pair.
        let one: std::collections::HashSet<&str> = ["bracket B"].into();
        assert_eq!(half_names(&one, "bracket"), ("bracket_1 A".into(), "bracket_1 B".into()));
    }

    #[test]
    fn a_console_triple_reads_commas_and_refuses_what_is_not_three_numbers() {
        assert_eq!(triple(&Value::Str("1,2,3".into())), Some([1.0, 2.0, 3.0]));
        assert_eq!(triple(&Value::Str(" 0 , -2.5 ,1e2 ".into())), Some([0.0, -2.5, 100.0]));
        assert_eq!(triple(&Value::Vec3([4.0, 5.0, 6.0])), Some([4.0, 5.0, 6.0]));
        for bad in ["1,2", "1,2,3,4", "a,b,c", "", "1,2,nan", "1,2,inf"] {
            assert_eq!(triple(&Value::Str(bad.into())), None, "{bad:?}");
        }
        assert_eq!(triple(&Value::Int(3)), None);
    }

    /// A sphere's own normal, taken through the block's analytic answer.
    #[test]
    fn the_direction_on_a_spheres_surface_points_out_along_the_radius() {
        let (graph, _) = sphere_graph();
        let field = compile(&graph);
        let radius = 20.0;
        for dir in [Vector3::x(), Vector3::y(), Vector3::z(), Vector3::new(1.0, 2.0, -2.0).normalize()] {
            let on_surface = Point3::from(dir * radius);
            let n = field_direction(&field, on_surface, 1e-3).expect("a sphere has a normal");
            assert!((n - dir).norm() < 1e-3, "{dir:?}: got {n:?}");
            assert!((n.norm() - 1.0).abs() < 1e-9, "unit length");
        }
    }

    /// A field that offers no analytic normal falls back to differencing its distance — and the
    /// fallback is normalised, since a distance that is only a bound does not have unit gradient.
    #[test]
    fn a_field_with_no_normal_of_its_own_is_differenced_and_normalised() {
        struct SteepFloor;
        impl ScalarField for SteepFloor {
            fn distance(&self, p: Point3<f64>) -> f64 {
                // Rises five times faster than a distance would: a LowerBound-style reading.
                5.0 * p.z
            }
        }
        let n = field_direction(&SteepFloor, Point3::new(3.0, -2.0, 0.0), 1e-3).unwrap();
        assert!((n - Vector3::z()).norm() < 1e-9, "{n:?}");
        assert!(
            field_direction(&SteepFloor, Point3::origin(), 1e-3).unwrap().z > 0.0,
            "and it points the way the field increases"
        );

        struct Flat;
        impl ScalarField for Flat {
            fn distance(&self, _: Point3<f64>) -> f64 {
                1.0
            }
        }
        assert!(field_direction(&Flat, Point3::origin(), 1e-3).is_none(), "no direction, no axis");
    }
}
