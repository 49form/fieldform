// SPDX-License-Identifier: GPL-3.0-only
//! The graph view's seat in the application: reads the notebook's own nesting, draws the canvas,
//! and turns a click into the selection a click on the row would make.

use eframe::egui;
use fieldform_scene::{CurveId, NodeId};

use crate::blocks::list::Block;
use crate::graph_view::CurveRow;

use crate::App;

impl App {
    /// The notebook's blocks as the graph draws them, and the node a curve selection lights.
    ///
    /// One place, so the canvas and `find` read the same names.
    pub(crate) fn graph_model(&mut self) -> (crate::graph_view::model::GraphModel, Option<NodeId>) {
        self.refresh_profiles();
        let list = self.list_readings();
        let label = |id: NodeId| -> String {
            self.body_name(id).map(str::to_owned).unwrap_or_else(|| self.derived_label(id))
        };
        let nesting_ref = &list.nesting;
        let curve_row = |c: CurveId| match self.folded_curve_row(c, nesting_ref) {
            Some(Block::Profile(key)) => CurveRow::Profile(self.profiles.members(key).map_or(0, <[_]>::len)),
            None => self.profiles.key_of(c).map_or(CurveRow::Own, CurveRow::Inside),
            Some(_) => CurveRow::Own,
        };
        let sources = crate::graph_view::sources(
            &self.doc,
            self.part_graph(),
            &list.nesting,
            &list.listed,
            &label,
            &|id| {
                let has_eye = self.part_graph().node(id).is_some_and(|n| n.is_body()) || self.draws_as_overlay(id);
                has_eye.then(|| self.is_visible(id))
            },
            &curve_row,
        );
        let model = crate::graph_view::model::build(&sources);
        // A selected curve lights the node that draws it: its own, or its loop's.
        let selected = self.selected_body.or_else(|| {
            self.selected_curve.map(|c| match curve_row(c) {
                CurveRow::Inside(key) => crate::graph_view::curve_id(key),
                _ => crate::graph_view::curve_id(c),
            })
        });
        (model, selected)
    }

    pub(crate) fn graph_panel(&mut self, ui: &mut egui::Ui) {
        let (model, selected) = self.graph_model();
        // **Followed here as the notebook follows it, and remembered apart from it**, so that going
        // to the other view finds the selection there too. Every block is a node of its own, so
        // there is nothing to open: what it can lack is being on the sheet.
        if selected != self.graph_followed {
            self.graph_followed = selected;
            if let Some(id) = selected {
                self.graph_view.request_bring(id);
            }
        }
        let height = (ui.clip_rect().height() - 36.0).max(160.0);
        let outcome = self.graph_view.show(ui, &model, selected, height, &mut self.doc.graph_layout);
        for (op, args) in outcome.commands {
            self.dispatch(op, &args);
        }
        if let Some(id) = outcome.eye {
            self.toggle_visibility(id);
        }
        if let Some(c) = outcome.clicked_curve {
            self.select_curve(c);
        }
        if let Some(id) = outcome.clicked {
            self.select_body(Some(id));
        }
        // V1's double-click is "show in notebook": the user is never stuck in a view-only graph.
        if let Some(id) = outcome.open_in_notebook {
            self.select_body(Some(id));
            self.doc.graph_layout.shown = false;
        }
    }
}

impl App {
    /// `find 'ball'` (Ctrl+F with no name): lights the nodes whose name contains the text and brings
    /// them into view. Without text it only puts the keyboard in the field. Presentation only (G3).
    pub(crate) fn find_from_console(&mut self, args: &fieldform_ops::Args) {
        if !self.doc.graph_layout.shown {
            self.warn("Find searches the graph — `view show=graph` first");
            return;
        }
        let Some(text) = args.get("name").and_then(|v| v.as_str().map(str::to_owned)) else {
            self.graph_view.focus_find();
            return;
        };
        self.graph_view.set_find(&text);
        if text.trim().is_empty() {
            self.say("Find cleared".to_string());
            return;
        }
        let (model, _) = self.graph_model();
        let n = crate::graph_view::matching(&model, &text).len();
        if n == 0 {
            self.warn(format!("Nothing in the graph is called {text}"));
        } else {
            self.say(format!("{n} found for {text}"));
        }
    }

    /// `view show=graph|notebook map=on|off`: the blocks panel's two views and its minimap. Presentation only, so it is not an undo step (G3).
    pub(crate) fn panel_from_console(&mut self, args: &fieldform_ops::Args) {
        if let Some(show) = args.get("show").and_then(|v| v.as_str().map(str::to_ascii_lowercase)) {
            match show.as_str() {
                "graph" => {
                    // Framed on arrival: the panel's size is only known once it is drawn.
                    if !self.doc.graph_layout.shown {
                        self.graph_view.request_frame(None);
                    }
                    self.doc.graph_layout.shown = true;
                }
                "notebook" | "list" => self.doc.graph_layout.shown = false,
                other => {
                    self.warn(format!("No view called {other} — graph or notebook"));
                    return;
                }
            }
        }
        if let Some(map) = args.get("map").and_then(|v| v.as_str().map(str::to_ascii_lowercase)) {
            self.graph_view.set_minimap(!matches!(map.as_str(), "off" | "no" | "false" | "0"));
        }
        self.say(if self.doc.graph_layout.shown {
            "Blocks shown as a graph".to_string()
        } else {
            "Blocks shown as a list".to_string()
        });
    }
}
