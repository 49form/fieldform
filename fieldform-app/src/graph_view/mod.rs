// SPDX-License-Identifier: GPL-3.0-only
//! The notebook drawn as a graph (`notebook-graph-view-spec.md`, V1: look, don't touch).
//!
//! **One document, two views.** Nothing here owns part of the model. [`sources`] reads the same
//! `Nesting` the block list is drawn from, [`model::build`] turns it into nodes and wires, and
//! [`GraphView`] keeps only what is presentation (G1): the camera and the positions somebody
//! dragged a node to. None of it reaches the document, the
//! generated source or the modified flag.
//!
//! A custom canvas rather than a graph library: both candidates need a newer egui than the
//! application's, and what the view needs is small: pan, zoom, cards, bezier wires and hit testing.

pub(crate) mod layout;
pub(crate) mod model;

use std::collections::{HashMap, HashSet};

use fieldform_scene::{
    BlockKey, CellMap, CurveId, Document, Graph, Node, Nesting, NodeId, Ownership, ParamId, ParamRef, ParamValue,
    QuadMeshId,
};

use eframe::egui;
use fieldform_ops::{Args, Value};
use layout::{Layout, HEADER, ROW};
use model::{GraphModel, Kind, Number, Source, Status, WireKind};

/// Below this zoom, nodes draw as header-only blocks and literals are hidden (§6).
const LOD_ZOOM: f32 = 0.55;

/// The id a named number stands under among the blocks: its own, written as a block's.
///
/// Both are random 64-bit ids, so they cannot meet; the canvas reports a click on one as a number
/// (`GNode::number`), never as a block, so the id is never handed to anything that selects.
pub(crate) fn number_id(p: ParamId) -> NodeId {
    NodeId::from_hex(&format!("{:016x}", p.raw())).expect("sixteen hex digits")
}

/// The id a curve stands under among the blocks, made the way [`number_id`] is and for the same
/// reason: it is never handed to anything that selects a block.
pub(crate) fn curve_id(c: CurveId) -> NodeId {
    NodeId::from_hex(&format!("{:016x}", c.raw())).expect("sixteen hex digits")
}

fn surface_id(s: QuadMeshId) -> NodeId {
    NodeId::from_hex(&format!("{:016x}", s.raw())).expect("sixteen hex digits")
}

/// What the notebook does with a curve's row once loops are folded (`folded_curve_row`): the graph
/// asks the same question so a loop of thirty curves is one node here as it is one row there.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CurveRow {
    /// A row of its own.
    Own,
    /// The row that heads a closed loop of this many curves.
    Profile(usize),
    /// Drawn inside the row of the loop headed by this curve, so a block reading it reads that node.
    Inside(CurveId),
}

/// The notebook's rows as the graph reads them: one [`Source`] per node of the part, in row order.
pub(crate) fn sources(
    doc: &Document,
    graph: &Graph,
    nesting: &Nesting,
    listed: &[NodeId],
    label: &dyn Fn(NodeId) -> String,
    eye: &dyn Fn(NodeId) -> Option<bool>,
    curve_row: &dyn Fn(CurveId) -> CurveRow,
) -> Vec<Source> {
    // What the notebook already marks, asked once for the part: the same two questions its rows
    // ask (`incomplete_in` for an empty slot, `approximate_blends` for the `≈` beside a radius).
    let holed = doc.incomplete_in(graph);
    let approximate: HashSet<NodeId> = fieldform_field::approximate_blends(doc, graph).into_iter().collect();
    // The named numbers first, by name, as the notebook lists them: inputs, above what reads them.
    let mut numbers: Vec<_> = doc.parameters.values().collect();
    numbers.sort_by(|a, b| a.name.cmp(&b.name));
    let mut out: Vec<Source> = numbers
        .into_iter()
        .map(|p| Source {
            id: number_id(p.id),
            label: p.name.clone(),
            variable: true,
            owners: Vec::new(),
            children: Vec::new(),
            references: Vec::new(),
            eye: None,
            kind: Kind::Number,
            number: Some(Number {
                value: doc.parameter_value(p.id).unwrap_or(f64::NAN),
                editable: matches!(p.value, ParamValue::Literal(_)),
            }),
            // The same `None` every `ParamRef::resolve` caller refuses: a division by zero, or a
            // reference to another failure.
            status: if doc.parameter_value(p.id).is_none() {
                Status::Error("Its expression does not compute".into())
            } else {
                Status::Ok
            },
        })
        .collect();
    // Curves, then surfaces: inputs the notebook lists above what reads them, in the notebook's own
    // order (creation time, falling back to the id so an unstamped file is still stable).
    let mut curves: Vec<CurveId> = doc.curves.keys().copied().collect();
    curves.sort_by_key(|id| (doc.curves_created.get(id).copied().unwrap_or(0), id.raw()));
    for c in curves {
        let (label, kind) = match curve_row(c) {
            CurveRow::Own => (doc.curve(c).map(|k| k.name.clone()).unwrap_or_default(), Kind::Curve(c)),
            CurveRow::Profile(n) => (
                format!("{} \u{b7} {n} curves", doc.curve(c).map(|k| k.name.as_str()).unwrap_or("Profile")),
                Kind::Profile(c),
            ),
            CurveRow::Inside(_) => continue,
        };
        out.push(Source {
            id: curve_id(c),
            label,
            variable: doc.ownership(BlockKey::Curve(c)) == Ownership::Variable,
            owners: nesting.curve_owners_of(c).to_vec(),
            children: Vec::new(),
            references: Vec::new(),
            eye: None,
            number: None,
            status: Status::Ok,
            kind,
        });
    }
    for s in doc.quad_meshes_ordered() {
        out.push(Source {
            id: surface_id(s),
            label: doc.surface_name(s).to_string(),
            variable: true,
            owners: Vec::new(),
            children: Vec::new(),
            references: Vec::new(),
            eye: None,
            number: None,
            status: Status::Ok,
            kind: Kind::Surface,
        });
    }
    out.extend(listed
        .iter()
        .filter_map(|&id| {
            let node = graph.node(id)?;
            let mut children: Vec<NodeId> = Vec::new();
            for input in node.inputs() {
                if nesting.owners_of(input).contains(&id) && !children.contains(&input) {
                    children.push(input);
                }
            }
            let mut references: Vec<NodeId> = nesting
                .references_of(id)
                .iter()
                .filter_map(|k| match k {
                    BlockKey::Node(v) => Some(*v),
                    BlockKey::Curve(_) => None,
                })
                .collect();
            // The curves it reads: nested in it when it is the only reader, a wire from the curve's
            // own node when the curve is a variable, exactly as the block's other inputs are. A
            // member of a folded loop is read through the node that heads the loop.
            for c in node.curves() {
                let at = match curve_row(c) {
                    CurveRow::Inside(key) => key,
                    _ => c,
                };
                let target = curve_id(at);
                if nesting.curve_owners_of(c).contains(&id) {
                    if !children.contains(&target) {
                        children.push(target);
                    }
                } else if !references.contains(&target) {
                    references.push(target);
                }
            }
            // The surface a conformal lattice follows is a document resource, wired like a variable.
            if let Node::Lattice { map: CellMap::Conformal { surface, .. }, .. } = node
                && !references.contains(&surface_id(*surface))
            {
                references.push(surface_id(*surface));
            }
            // The named numbers this block's own fields read.
            for r in node.param_refs() {
                if let ParamRef::Named(p) = r
                    && doc.parameters.contains_key(p)
                    && !references.contains(&number_id(*p))
                {
                    references.push(number_id(*p));
                }
            }
            Some(Source {
                id,
                label: label(id),
                variable: doc.ownership(BlockKey::Node(id)) == Ownership::Variable,
                owners: nesting.owners_of(id).to_vec(),
                children,
                references,
                eye: eye(id),
                number: None,
                kind: Kind::Block,
                status: if holed.contains(&id) {
                    Status::Error("An input is empty here or below, so this cannot be evaluated".into())
                } else if approximate.contains(&id) {
                    Status::Warning("Approximate: the radius will not be the radius typed".into())
                } else {
                    Status::Ok
                },
            })
        }));
    out
}

/// What a frame of the canvas asks the application to do.
#[derive(Default)]
pub(crate) struct Outcome {
    /// A node was clicked: select it, as a click on its row would.
    pub(crate) clicked: Option<NodeId>,
    /// A curve's node (or a loop's) was clicked: select it, as its row does.
    pub(crate) clicked_curve: Option<CurveId>,
    /// A node was double-clicked: V1's "show in notebook".
    pub(crate) open_in_notebook: Option<NodeId>,
    /// An eye was clicked: the application turns that block's visibility over, as its row's eye does.
    pub(crate) eye: Option<NodeId>,
    /// Registry commands the canvas asks for: the application dispatches them.
    pub(crate) commands: Vec<(&'static str, Args)>,
}

pub(crate) struct GraphView {
    pan: egui::Vec2,
    zoom: f32,
    /// Positions somebody dragged a node to, in graph space: the document's own
    /// (`Document::graph_layout`), copied in at the start of a frame and written back on a drag.
    pins: HashMap<NodeId, [f32; 2]>,
    /// A frame asked for by the registry's commands: `Some(None)` is everything.
    frame_request: Option<Option<NodeId>>,
    /// A node to bring into view on the next frame drawn, at the zoom the view already has.
    bring_request: Option<NodeId>,
    /// *Show what feeds this* (`true`) or *what this feeds* (`false`): dims the rest, stores nothing.
    isolate: Option<(NodeId, bool)>,
    minimap: bool,
    /// What the find field holds: nodes whose name does not contain it are faded, nothing is stored.
    find: String,
    /// Ctrl+F asked for the field: it takes the keyboard on the next frame drawn.
    focus_find: bool,
    /// Frame the matches rather than what `frame_request` names, on the next frame drawn.
    frame_matches: bool,
    cached: Option<(Vec<(NodeId, usize)>, Vec<model::Wire>, Layout)>,
}

/// The nodes whose name contains `text`, ignoring case: what *Find* lights. Empty text matches none,
/// so a cleared field leaves the graph as it was.
pub(crate) fn matching(model: &GraphModel, text: &str) -> HashSet<NodeId> {
    let text = text.trim().to_lowercase();
    if text.is_empty() {
        return HashSet::new();
    }
    model.nodes.iter().filter(|n| n.label.to_lowercase().contains(&text)).map(|n| n.id).collect()
}

impl Default for GraphView {
    fn default() -> Self {
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            pins: HashMap::new(),
            frame_request: Some(None),
            bring_request: None,
            isolate: None,
            minimap: true,
            find: String::new(),
            focus_find: false,
            frame_matches: false,
            cached: None,
        }
    }
}

impl GraphView {
    pub(crate) fn set_minimap(&mut self, on: bool) {
        self.minimap = on;
    }

    /// `find 'text'`: lights the matches and brings them into view; empty text clears.
    pub(crate) fn set_find(&mut self, text: &str) {
        self.find = text.to_string();
        self.frame_matches = !self.find.trim().is_empty();
        self.frame_request = Some(None);
    }

    /// Ctrl+F: puts the keyboard in the find field.
    pub(crate) fn focus_find(&mut self) {
        self.focus_find = true;
    }

    /// Frames `only` (F), or everything (A, Home) on the next frame drawn.
    pub(crate) fn request_frame(&mut self, only: Option<NodeId>) {
        self.frame_request = Some(only);
    }

    /// Brings a node into view on the next frame drawn. See [`Self::pan_to_show`].
    pub(crate) fn request_bring(&mut self, id: NodeId) {
        self.bring_request = Some(id);
    }

    /// Centres `node` if any of it is outside `view`, and otherwise leaves the camera alone.
    ///
    /// **Pans and never zooms**, which is the difference from F: this runs because something was
    /// chosen somewhere else, not because anybody asked to look at this node, so the view keeps
    /// the scale it was left at. And it does nothing to a node already in view, so a click on a
    /// node does not slide the sheet out from under the pointer.
    fn pan_to_show(&mut self, node: egui::Rect, view: egui::Rect) {
        if !view.contains_rect(node) {
            self.pan += view.center() - node.center();
        }
    }

    fn laid_out(&mut self, model: &GraphModel) -> &Layout {
        let shape: Vec<(NodeId, usize)> = model.nodes.iter().map(|n| (n.id, n.rows.len())).collect();
        let stale = match &self.cached {
            Some((s, w, _)) => *s != shape || *w != model.wires,
            None => true,
        };
        if stale {
            self.cached = Some((shape, model.wires.clone(), layout::layout(model)));
        }
        &self.cached.as_ref().expect("just set").2
    }

    fn rect_of(&self, layout: &Layout, i: usize, id: NodeId, origin: egui::Pos2) -> egui::Rect {
        let [x, y] = self.pins.get(&id).copied().unwrap_or(layout.at[i]);
        let [w, h] = layout.size[i];
        let min = origin + (egui::vec2(x, y) * self.zoom) + self.pan;
        egui::Rect::from_min_size(min, egui::vec2(w, h) * self.zoom)
    }

    /// Frames every node (A / Home) or just `only` (F).
    fn frame(&mut self, layout: &Layout, model: &GraphModel, view: egui::Rect, only: Option<&HashSet<NodeId>>) {
        let mut bounds: Option<egui::Rect> = None;
        for (i, n) in model.nodes.iter().enumerate() {
            if only.is_some_and(|o| !o.contains(&n.id)) {
                continue;
            }
            let [x, y] = self.pins.get(&n.id).copied().unwrap_or(layout.at[i]);
            let r = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(layout.size[i][0], layout.size[i][1]));
            bounds = Some(bounds.map_or(r, |b| b.union(r)));
        }
        let Some(b) = bounds else { return };
        let pad = 24.0;
        let zoom = ((view.width() - 2.0 * pad) / b.width().max(1.0))
            .min((view.height() - 2.0 * pad) / b.height().max(1.0))
            .clamp(0.1, 1.5);
        self.zoom = zoom;
        // `rect_of` places a node at origin + at * zoom + pan, with origin = view.min.
        self.pan = (view.size() - b.size() * zoom) * 0.5 - b.min.to_vec2() * zoom;
    }

    /// Scales the view by `factor`, keeping the point under `about` where it is.
    fn zoom_about(&mut self, about: egui::Pos2, origin: egui::Pos2, factor: f32) {
        let old = self.zoom;
        self.zoom = (self.zoom * factor).clamp(0.1, 3.0);
        let from_origin = about - origin - self.pan;
        self.pan -= from_origin * (self.zoom / old - 1.0);
    }

    /// Graph paper behind the nodes: the lines the orthographic grid draws, without its scale.
    ///
    /// It moves and scales with the camera, so panning and zooming read as moving over a sheet.
    /// Every fifth line is a little stronger. Lines closer than a few pixels are left out, so a
    /// zoomed-out view is a quiet texture rather than a wash.
    fn paint_grid(&self, painter: &egui::Painter, rect: egui::Rect, visuals: &egui::Visuals) {
        const CELL: f32 = 24.0;
        let step = CELL * self.zoom;
        if step < 4.0 {
            return;
        }
        let ink = visuals.widgets.noninteractive.bg_stroke.color;
        let minor = egui::Stroke::new(1.0_f32, ink.gamma_multiply(0.35));
        let major = egui::Stroke::new(1.0_f32, ink.gamma_multiply(0.75));
        let origin = rect.min + self.pan;
        let first = |from: f32, at: f32| ((from - at) / step).floor() as i64;
        let (i0, i1) = (first(rect.left(), origin.x), first(rect.right(), origin.x) + 1);
        for i in i0..=i1 {
            let x = origin.x + i as f32 * step;
            let stroke = if i.rem_euclid(5) == 0 { major } else { minor };
            if step >= 8.0 || i.rem_euclid(5) == 0 {
                painter.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], stroke);
            }
        }
        let (j0, j1) = (first(rect.top(), origin.y), first(rect.bottom(), origin.y) + 1);
        for j in j0..=j1 {
            let y = origin.y + j as f32 * step;
            let stroke = if j.rem_euclid(5) == 0 { major } else { minor };
            if step >= 8.0 || j.rem_euclid(5) == 0 {
                painter.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)], stroke);
            }
        }
    }

    /// Draws the canvas. `selected` is the application's selected body, so the highlight is the
    /// notebook's, not a second selection.
    pub(crate) fn show(&mut self, ui: &mut egui::Ui, model: &GraphModel, selected: Option<NodeId>, height: f32, state: &mut fieldform_scene::GraphLayout) -> Outcome {
        let mut out = Outcome::default();
        // The document's pins are the truth; a pin for a node that is gone is dropped here.
        let present: HashSet<NodeId> = model.nodes.iter().map(|n| n.id).collect();
        state.pins.retain(|id, _| present.contains(id));
        self.pins = state.pins.iter().map(|(k, v)| (*k, *v)).collect();
        let size = egui::vec2(ui.available_width().max(120.0), height.max(120.0));
        let (rect, canvas) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        // The toolbar's room is taken now, below the canvas, and drawn into at the end: anything
        // placed on the canvas by hand moves the cursor, and the toolbar followed it.
        let (toolbar, _) = ui.allocate_exact_size(egui::vec2(size.x, 26.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        // The panel's own colour, so the graph reads as part of the window; the grid is what says
        // it is a graph. Drawn once the camera for this frame is settled, below.
        painter.rect_filled(rect, 0.0, ui.visuals().panel_fill);

        let layout = self.laid_out(model).clone();
        if let Some(only) = self.frame_request.take() {
            let found = std::mem::take(&mut self.frame_matches).then(|| matching(model, &self.find));
            let only = found.or_else(|| only.map(|o| HashSet::from([o])));
            self.frame(&layout, model, rect.translate(-rect.min.to_vec2()), only.as_ref());
        }
        // After the frame, so a selection made as the graph opens lands in the framing it opened on.
        if let Some(id) = self.bring_request.take()
            && let Some(i) = model.nodes.iter().position(|n| n.id == id)
        {
            self.pan_to_show(self.rect_of(&layout, i, id, rect.min), rect);
        }

        // The viewport's conventions, as far as a flat canvas has them: the wheel zooms about the
        // cursor, the middle button pans, and the right button dollies (drag up to come closer,
        // as the viewport's does). The left button turns the camera there and pans here, since
        // a sheet has nothing to turn; a node under the pointer takes the left button first.
        if canvas.dragged_by(egui::PointerButton::Middle) || canvas.dragged_by(egui::PointerButton::Primary) {
            self.pan += canvas.drag_delta();
        }
        if canvas.dragged_by(egui::PointerButton::Secondary) {
            let about = canvas.interact_pointer_pos().unwrap_or(rect.center());
            self.zoom_about(about, rect.min, 1.0 - canvas.drag_delta().y * 0.005);
        }
        if canvas.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(p) = ui.input(|i| i.pointer.hover_pos()) {
                    self.zoom_about(p, rect.min, 1.0 + scroll * 0.001);
                }
                // Taken, so the list's own scrolling does not move the panel under the canvas.
                ui.input_mut(|i| i.smooth_scroll_delta = egui::Vec2::ZERO);
            }
        }

        self.paint_grid(&painter, rect, ui.visuals());

        let index = layout.index_of(model);
        let rects: Vec<egui::Rect> = model
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| self.rect_of(&layout, i, n.id, rect.min))
            .collect();

        let visuals = ui.visuals().clone();
        let chip = visuals.selection.bg_fill;
        let owned = visuals.widgets.noninteractive.fg_stroke.color;
        let selecting = selected;

        let lit: Option<HashSet<NodeId>> = self.isolate.map(|(id, up)| model.reach(id, up));
        // Wires first, so nodes sit on top. Dimmed unless they touch the selection (§6).
        for w in &model.wires {
            let (Some(&f), Some(&t)) = (index.get(&w.from), index.get(&w.to)) else { continue };
            let a = egui::pos2(rects[f].right(), rects[f].top() + HEADER * 0.5 * self.zoom);
            // A chip lands on its own row of the reader, so each input of a block has its wire;
            // anything else lands on the header.
            let row = model.nodes[t].rows.iter().position(|c| c.from == w.from && c.kind == w.kind);
            let b = match row {
                Some(k) => egui::pos2(rects[t].left(), rects[t].top() + (HEADER + ROW * k as f32 + ROW * 0.5 + 3.0) * self.zoom),
                None => egui::pos2(rects[t].left(), rects[t].top() + HEADER * 0.5 * self.zoom),
            };
            let touches = selecting.is_some_and(|s| s == w.from || s == w.to);
            let dim = (selecting.is_some() && !touches)
                || lit.as_ref().is_some_and(|l| !(l.contains(&w.from) && l.contains(&w.to)));
            let base = if w.kind == WireKind::Chip { chip } else { owned };
            let colour = if dim { base.gamma_multiply(0.25) } else { base };
            let dx = ((b.x - a.x).abs() * 0.5).max(30.0 * self.zoom);
            painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                [a, a + egui::vec2(dx, 0.0), b - egui::vec2(dx, 0.0), b],
                false,
                egui::Color32::TRANSPARENT,
                egui::Stroke::new(if touches { 2.2 } else { 1.4 } * self.zoom.max(0.6), colour),
            ));
        }

        let lod = self.zoom < LOD_ZOOM;
        for (i, n) in model.nodes.iter().enumerate() {
            let r = rects[i];
            if !r.intersects(rect) {
                continue;
            }
            let response = ui.interact(r, ui.id().with(("graph_node", n.id)), egui::Sense::click_and_drag());
            if response.dragged_by(egui::PointerButton::Primary) {
                let at = self.pins.get(&n.id).copied().unwrap_or(layout.at[i]);
                let d = response.drag_delta() / self.zoom;
                self.pins.insert(n.id, [at[0] + d.x, at[1] + d.y]);
                state.pins.insert(n.id, [at[0] + d.x, at[1] + d.y]);
            }
            // A named number is not a block: a click on one selects nothing.
            if response.clicked() {
                match n.kind {
                    Kind::Block => out.clicked = Some(n.id),
                    Kind::Curve(c) | Kind::Profile(c) => out.clicked_curve = Some(c),
                    Kind::Number | Kind::Surface => {}
                }
            }
            if response.double_clicked() && n.kind == Kind::Block {
                out.open_in_notebook = Some(n.id);
            }

            let picked = selected == Some(n.id);
            let (header_fill, body_fill) = if n.variable {
                (visuals.selection.bg_fill.gamma_multiply(0.8), visuals.faint_bg_color)
            } else {
                (visuals.widgets.inactive.bg_fill, visuals.window_fill)
            };
            let flag = match &n.status {
                Status::Ok => None,
                Status::Warning(_) => Some(visuals.warn_fg_color),
                Status::Error(_) => Some(visuals.error_fg_color),
            };
            let stroke = match (picked, flag) {
                (_, Some(colour)) => egui::Stroke::new(2.0_f32, colour),
                (true, None) => visuals.selection.stroke,
                (false, None) => egui::Stroke::new(1.0_f32, visuals.widgets.noninteractive.bg_stroke.color),
            };
            let radius = 4.0 * self.zoom;
            painter.rect(r, radius, body_fill, stroke, egui::StrokeKind::Inside);
            // A flagged node that is also selected keeps both: the selection ring sits just inside.
            if picked && flag.is_some() {
                painter.rect_stroke(r.shrink(2.5), radius, visuals.selection.stroke, egui::StrokeKind::Inside);
            }
            if let Status::Warning(why) | Status::Error(why) = &n.status {
                response.clone().on_hover_text(why);
            }
            let header = egui::Rect::from_min_size(r.min, egui::vec2(r.width(), HEADER * self.zoom));
            painter.rect_filled(header, radius, header_fill);
            // The owned bracket: a block pulled out of its parent keeps a mark saying it is
            // nested, not a variable.
            if n.owned {
                painter.line_segment(
                    [r.left_top() + egui::vec2(0.0, 2.0), r.left_bottom() - egui::vec2(0.0, 2.0)],
                    egui::Stroke::new(3.0 * self.zoom.max(0.6), visuals.widgets.noninteractive.fg_stroke.color),
                );
            }
            let font = egui::FontId::proportional((12.5 * self.zoom).clamp(7.0, 22.0));
            painter.text(
                header.left_center() + egui::vec2(6.0 * self.zoom, 0.0),
                egui::Align2::LEFT_CENTER,
                &n.label,
                font.clone(),
                visuals.strong_text_color(),
            );
            // From the right edge: the V of a variable, then the eye.
            let mut slot = 11.0_f32;
            if n.variable {
                let v = egui::Rect::from_center_size(
                    header.right_center() - egui::vec2(slot * self.zoom, 0.0),
                    egui::vec2(16.0, 16.0) * self.zoom.max(0.6),
                );
                painter.text(v.center(), egui::Align2::CENTER_CENTER, "V", font.clone(), visuals.selection.stroke.color);
                ui.interact(v, ui.id().with(("graph_v", n.id)), egui::Sense::hover())
                    .on_hover_text("A variable: read by name wherever it is used");
                slot += 19.0;
            }
            if let Some(on) = n.eye {
                let from_right = slot;
                let eye = egui::Rect::from_center_size(
                    header.right_center() - egui::vec2(from_right * self.zoom, 0.0),
                    egui::vec2(16.0, 16.0) * self.zoom.max(0.6),
                );
                painter.text(
                    eye.center(),
                    egui::Align2::CENTER_CENTER,
                    "\u{1f441}",
                    font.clone(),
                    if on { visuals.text_color() } else { visuals.weak_text_color().gamma_multiply(0.6) },
                );
                let e = ui.interact(eye, ui.id().with(("graph_eye", n.id)), egui::Sense::click());
                if e.clicked() {
                    out.eye = Some(n.id);
                }
                e.on_hover_text(if on { "Hide" } else { "Show" });
            }
            response.context_menu(|ui| {
                if ui.button("Show what feeds this").clicked() {
                    self.isolate = Some((n.id, true));
                    ui.close();
                }
                if ui.button("Show what this feeds").clicked() {
                    self.isolate = Some((n.id, false));
                    ui.close();
                }
                if self.isolate.is_some() && ui.button("Show everything").clicked() {
                    self.isolate = None;
                    ui.close();
                }
            });
            if let (Some(num), false) = (n.number, lod) {
                // The value, in a field: the notebook's own edit, said as `param name=.. value=..`.
                let field = egui::Rect::from_min_size(
                    egui::pos2(r.left() + 6.0 * self.zoom, r.top() + (HEADER + 3.0) * self.zoom),
                    egui::vec2(r.width() - 12.0 * self.zoom, (ROW - 2.0) * self.zoom),
                );
                // Held while it is being dragged or typed, and applied once when it is let go of, so
                // a drag is one command and one undo step rather than one for every pixel.
                let key = ui.id().with(("graph_number", n.id));
                let mut v = ui.data(|d| d.get_temp::<f64>(key)).unwrap_or(num.value);
                // A child of its own, so the field does not move the panel's layout cursor.
                let mut cell = ui.new_child(egui::UiBuilder::new().max_rect(field));
                let reply = cell
                    .add_enabled(num.editable, egui::DragValue::new(&mut v).speed(0.1).max_decimals(4))
                    .on_disabled_hover_text("Computed from other numbers: change it where it is defined");
                if reply.dragged() || reply.has_focus() {
                    ui.data_mut(|d| d.insert_temp(key, v));
                } else {
                    ui.data_mut(|d| d.remove_temp::<f64>(key));
                }
                if (reply.drag_stopped() || reply.lost_focus()) && v != num.value {
                    out.commands.push((
                        "param.set",
                        Args::new().with("name", Value::Str(n.label.clone())).with("value", Value::Float(v)),
                    ));
                }
            }
            if !lod {
                // Its inputs, as rows the wires end on.
                for (k, chip) in n.rows.iter().enumerate() {
                    let y = r.top() + (HEADER + ROW * k as f32 + 3.0) * self.zoom;
                    let row = egui::Rect::from_min_size(
                        egui::pos2(r.left() + 6.0 * self.zoom, y),
                        egui::vec2(r.width() - 12.0 * self.zoom, (ROW - 2.0) * self.zoom),
                    );
                    painter.rect(
                        row,
                        3.0 * self.zoom,
                        if chip.kind == WireKind::Chip { visuals.selection.bg_fill.gamma_multiply(0.35) } else { visuals.faint_bg_color },
                        egui::Stroke::new(1.0_f32, if chip.kind == WireKind::Chip { visuals.selection.bg_fill } else { visuals.widgets.noninteractive.bg_stroke.color }),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        row.left_center() + egui::vec2(4.0 * self.zoom, 0.0),
                        egui::Align2::LEFT_CENTER,
                        &chip.label,
                        egui::FontId::proportional((11.0 * self.zoom).clamp(7.0, 20.0)),
                        visuals.text_color(),
                    );
                }
            }
        }
        // What isolation leaves out is faded over, so it reads as there but not the subject.
        if let Some(lit) = &lit {
            for (i, n) in model.nodes.iter().enumerate() {
                if !lit.contains(&n.id) && rects[i].intersects(rect) {
                    painter.rect_filled(rects[i], 4.0 * self.zoom, visuals.panel_fill.gamma_multiply(0.7));
                }
            }
        }

        // What a search leaves out is faded the same way, and what it finds is ringed.
        let found = matching(model, &self.find);
        if !found.is_empty() {
            for (i, n) in model.nodes.iter().enumerate() {
                if !rects[i].intersects(rect) {
                    continue;
                }
                if found.contains(&n.id) {
                    painter.rect_stroke(rects[i].expand(3.0), 6.0 * self.zoom, visuals.selection.stroke, egui::StrokeKind::Outside);
                } else {
                    painter.rect_filled(rects[i], 4.0 * self.zoom, visuals.panel_fill.gamma_multiply(0.7));
                }
            }
        }

        // The minimap: every node, and the part of the sheet in view. Click or drag to go there.
        if self.minimap && !model.nodes.is_empty() {
            let map = egui::Rect::from_min_size(rect.right_bottom() - egui::vec2(146.0, 106.0), egui::vec2(136.0, 96.0));
            painter.rect(
                map,
                3.0,
                visuals.panel_fill.gamma_multiply(0.92),
                egui::Stroke::new(1.0_f32, visuals.widgets.noninteractive.bg_stroke.color),
                egui::StrokeKind::Inside,
            );
            let mut bounds: Option<egui::Rect> = None;
            let at = |i: usize, n: &model::GNode| self.pins.get(&n.id).copied().unwrap_or(layout.at[i]);
            for (i, n) in model.nodes.iter().enumerate() {
                let [x, y] = at(i, n);
                let r = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(layout.size[i][0], layout.size[i][1]));
                bounds = Some(bounds.map_or(r, |b| b.union(r)));
            }
            if let Some(b) = bounds {
                let inner = map.shrink(6.0);
                let scale = (inner.width() / b.width().max(1.0)).min(inner.height() / b.height().max(1.0));
                let to_map = |p: egui::Pos2| inner.min + (p - b.min) * scale;
                for (i, n) in model.nodes.iter().enumerate() {
                    let [x, y] = at(i, n);
                    let r = egui::Rect::from_min_max(
                        to_map(egui::pos2(x, y)),
                        to_map(egui::pos2(x + layout.size[i][0], y + layout.size[i][1])),
                    );
                    let fill = if n.variable { visuals.selection.bg_fill } else { visuals.widgets.inactive.bg_fill };
                    painter.rect_filled(r, 1.0, fill);
                }
                // The part of the sheet in view, in the map's own space.
                let seen = egui::Rect::from_min_max(
                    to_map(egui::pos2(-self.pan.x / self.zoom, -self.pan.y / self.zoom)),
                    to_map(egui::pos2((rect.width() - self.pan.x) / self.zoom, (rect.height() - self.pan.y) / self.zoom)),
                )
                .intersect(map);
                painter.rect_stroke(seen, 1.0, visuals.selection.stroke, egui::StrokeKind::Inside);
                let click = ui.interact(map, ui.id().with("graph_minimap"), egui::Sense::click_and_drag());
                if (click.clicked() || click.dragged())
                    && let Some(p) = click.interact_pointer_pos()
                {
                    let g = b.min + (p - inner.min) / scale;
                    self.pan = rect.size() * 0.5 - g.to_vec2() * self.zoom;
                }
            }
        }

        // The toolbar issues the registry's commands rather than changing state itself, so a
        // button, a key and a typed line are one path (G2) and each echoes the same line.
        let mut bar = ui.new_child(
            egui::UiBuilder::new().max_rect(toolbar).layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        {
            let ui = &mut bar;
            if ui.small_button("Frame all").clicked() {
                out.commands.push(("view.frame_all", Args::new()));
            }
            let map = if self.minimap { "off" } else { "on" };
            if ui.selectable_label(self.minimap, "Map").on_hover_text("The minimap").clicked() {
                out.commands.push(("view.panel", Args::new().with("map", Value::Str(map.into()))));
            }
            if ui.small_button("Re-layout").on_hover_text("Forget dragged positions").clicked() {
                out.commands.push(("view.graph_layout", Args::new()));
            }
            // Typing lights the matches as it goes (the field is the view's own state); Enter is
            // the command, which also brings them into view and says how many there were.
            let field = ui.add(
                egui::TextEdit::singleline(&mut self.find)
                    .hint_text("Find")
                    .desired_width(110.0),
            );
            if std::mem::take(&mut self.focus_find) {
                field.request_focus();
            }
            if field.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.find.clear();
                field.surrender_focus();
            }
            if field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                out.commands.push(("view.find", Args::new().with("name", Value::Str(self.find.clone()))));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(400.0, 300.0))
    }

    /// **A node off the sheet is centred, at the zoom the view had.** Three blocks deep or three
    /// hundred millimetres of graph away, the point is that it can be seen, not that it fills the
    /// panel, so the scale somebody left it at stays.
    #[test]
    fn a_node_out_of_view_is_centred_without_changing_the_zoom() {
        let mut graph = GraphView { zoom: 0.4, ..Default::default() };
        let node = egui::Rect::from_min_size(egui::pos2(900.0, 700.0), egui::vec2(80.0, 40.0));
        graph.pan_to_show(node, view());
        assert_eq!(graph.zoom, 0.4);
        assert_eq!(node.translate(graph.pan).center(), view().center());
    }

    /// **The same for one only partly in view**, which is the case a test of its centre alone
    /// would call visible while half its rows were cut off by the panel's edge.
    #[test]
    fn a_node_cut_by_the_edge_of_the_view_counts_as_out_of_it() {
        let mut graph = GraphView::default();
        let node = egui::Rect::from_min_size(egui::pos2(470.0, 100.0), egui::vec2(80.0, 40.0));
        graph.pan_to_show(node, view());
        assert_eq!(node.translate(graph.pan).center(), view().center());
    }

    /// **A node already in view stays exactly where it is.** The control for the two above, and the
    /// reason a click on a node does not move the sheet under the pointer.
    #[test]
    fn a_node_in_view_does_not_move_the_camera() {
        let mut graph = GraphView { pan: egui::vec2(12.0, -7.0), ..Default::default() };
        let node = egui::Rect::from_min_size(egui::pos2(200.0, 100.0), egui::vec2(80.0, 40.0));
        graph.pan_to_show(node, view());
        assert_eq!(graph.pan, egui::vec2(12.0, -7.0));
    }
}
