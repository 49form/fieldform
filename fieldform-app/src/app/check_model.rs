// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Check Model: what is red and what is amber on the cage, listed, painted, and typed.
//!
//! Three things live here. The **self-intersection check**, which is the one expensive check and
//! runs on another thread after a gesture ends, tagged with the fingerprint it was computed for
//! so a stale answer is thrown away (R14). The **markers**: red faults painted whenever there are
//! any, amber ones only while the panel is open or for a moment after an edit, since amber marks
//! that never go away are wallpaper. And the **panel**, with each check's count, its thresholds,
//! and a click that selects and frames the offenders -- the same list `cage.check` prints, so a
//! script reads what a person sees.

use crate::*;
use fieldform_cage::validate::{Check, Fault};
use fieldform_cage::FaceId;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// A self-intersection check in flight, for one cage at one fingerprint.
pub(crate) struct SurfaceCheck {
    cage: CageId,
    fingerprint: u64,
    rx: mpsc::Receiver<Vec<FaceId>>,
}

/// How long after an edit the amber marks show without the panel open.
const AMBER_AFTER_EDIT: Duration = Duration::from_secs(3);
/// How long after the last document move a check waits, so a run of quick edits is checked once.
const CHECK_DEBOUNCE: Duration = Duration::from_millis(150);

/// The colours of the marks, beside the selection's in `overlay_draw`.
pub(crate) const ERROR_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 70, 60);
pub(crate) const ERROR_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(120, 30, 25, 70);
pub(crate) const WARNING_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 180, 40);
pub(crate) const WARNING_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(110, 80, 15, 60);

impl App {
    /// Every red fault of the active cage, then every amber one under the document's thresholds.
    fn cage_faults(&self) -> (Vec<Fault>, Vec<Fault>) {
        let Some(body) = self.doc.cage_body(self.active_cage) else { return (Vec::new(), Vec::new()) };
        (body.red_faults(), body.amber_faults(&self.doc.validation))
    }

    /// Runs after a document change: the geometry faults and shell orientation re-read, and the
    /// amber marks shown for a moment.
    pub(crate) fn cages_changed(&mut self) {
        self.doc.refresh_cages();
        // A pending extrude or inset is a draft: its lift at nought has walls the geometry
        // checks would refuse, and the drag's own refreshes must not start judging it. Read
        // here, before the compile below, so the lift is drawn rather than warned about.
        if let Some(cage) = self.cage_session.as_ref().map(|s| s.cage)
            && let Some(body) = self.doc.cage_body_mut(cage)
        {
            body.begin_draft();
        }
        self.amber_until = Some(Instant::now() + AMBER_AFTER_EDIT);
        self.check_due = Instant::now() + CHECK_DEBOUNCE;
    }

    /// Takes in a finished self-intersection check and starts the next one due. Once a frame.
    pub(crate) fn poll_surface_checks(&mut self) {
        if let Some(check) = &self.surface_check {
            match check.rx.try_recv() {
                Ok(crossing) => {
                    let (cage, fingerprint) = (check.cage, check.fingerprint);
                    self.surface_check = None;
                    let was_red = self.doc.cage_is_red(cage);
                    let taken = self
                        .doc
                        .cage_body_mut(cage)
                        .is_some_and(|body| body.set_surface_check(fingerprint, crossing));
                    if taken && self.doc.cage_is_red(cage) != was_red {
                        // The verdict changed what is drawn: the cage joins or leaves the part.
                        self.rebuild_render_resources();
                    }
                }
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => self.surface_check = None,
            }
        }
        // Not during a gesture, and not until the edits have settled.
        if self.gizmo_drag.is_some() || self.move_state.is_some() || Instant::now() < self.check_due {
            return;
        }
        let due = self.doc.cages.iter().find(|(_, body)| body.surface_unchecked());
        let Some((&cage, body)) = due else { return };
        let Some(closed) = body.closed() else { return };
        let fingerprint = body.mesh().geometry_fingerprint();
        // The tessellation the field is built from, at the depth the viewport uses for this cage.
        let (lo, hi) = cage_box(body.mesh());
        let tolerance = (((hi - lo).norm() / 64.0) * 0.5).max(1e-4);
        let levels = fieldform_field::levels_for_tolerance((hi - lo).norm(), tolerance);
        let epsilon = fieldform_cage::validate::epsilon_length(body.mesh());
        let cage_copy = closed.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("cage self-intersection check".into())
            .spawn(move || {
                let found = fieldform_field::self_intersection::find(&cage_copy, levels, epsilon);
                let _ = tx.send(found.faces);
            })
            .expect("a thread for the check");
        self.surface_check = Some(SurfaceCheck { cage, fingerprint, rx });
    }

    /// Whether the active cage's surface is still being checked -- the "checking…" state.
    pub(crate) fn cage_checking(&self, cage: CageId) -> bool {
        self.doc.cage_body(cage).is_some_and(|b| b.surface_unchecked())
    }

    /// `cage.check`: the list, printed. Under a script it is the whole answer; otherwise the panel
    /// opens too.
    pub(crate) fn check_from_console(&mut self) {
        let (red, amber) = self.cage_faults();
        if red.is_empty() && amber.is_empty() {
            self.say(if self.cage_checking(self.active_cage) {
                "Check Model: nothing found so far; the surface is still being checked"
            } else {
                "Check Model: nothing found"
            });
        }
        for fault in red.iter().chain(&amber) {
            self.say(format!("{}: {}", if fault.check.is_red() { "error" } else { "warning" }, fault_line(fault)));
        }
        if !self.script_running {
            self.show_check_model = true;
        }
    }

    /// The Check Model window.
    pub(crate) fn check_model_panel(&mut self, ctx: &egui::Context) {
        if !self.show_check_model {
            return;
        }
        let mut open = true;
        let (red, amber) = self.cage_faults();
        let checking = self.cage_checking(self.active_cage);
        let mut pick: Option<Fault> = None;
        let mut thresholds = self.doc.validation;
        let mut only_errors = self.check_only_errors;
        egui::Window::new("Check Model").open(&mut open).default_width(380.0).show(ctx, |ui| {
            if checking {
                ui.label(egui::RichText::new("checking the surface…").weak());
            }
            ui.checkbox(&mut only_errors, "only errors");
            let mut listed = 0;
            for (title, faults, colour) in [
                ("Errors", &red, ui.visuals().error_fg_color),
                ("Warnings", &amber, ui.visuals().warn_fg_color),
            ] {
                if only_errors && title == "Warnings" {
                    break;
                }
                ui.separator();
                ui.label(egui::RichText::new(title).strong());
                if faults.is_empty() {
                    ui.label(egui::RichText::new("none").weak());
                }
                for fault in faults {
                    listed += 1;
                    if ui.add(egui::Button::new(egui::RichText::new(fault_line(fault)).color(colour)).frame(false))
                        .on_hover_text("Select and frame these")
                        .clicked()
                    {
                        pick = Some(fault.clone());
                    }
                }
            }
            let _ = listed;
            ui.separator();
            ui.label(egui::RichText::new("Warning thresholds, relative to the median edge").strong());
            egui::Grid::new("check-thresholds").num_columns(2).show(ui, |ui| {
                ui.label("Star point from valence");
                ui.add(egui::DragValue::new(&mut thresholds.star_valence).range(3..=64));
                ui.end_row();
                ui.label("Show every extraordinary vertex");
                ui.checkbox(&mut thresholds.show_all_extraordinary, "");
                ui.end_row();
                ui.label("Short edge below");
                ui.add(egui::DragValue::new(&mut thresholds.short_edge).speed(0.01).range(0.0..=1.0).suffix(" m"));
                ui.end_row();
                ui.label("Near-coincident below");
                ui.add(egui::DragValue::new(&mut thresholds.near_coincident).speed(0.01).range(0.0..=1.0).suffix(" m"));
                ui.end_row();
                ui.label("Folded face past");
                ui.add(egui::DragValue::new(&mut thresholds.fold_degrees).speed(1.0).range(90.0..=180.0).suffix("°"));
                ui.end_row();
                ui.label("Twisted face past");
                ui.add(egui::DragValue::new(&mut thresholds.twist_degrees).speed(1.0).range(10.0..=180.0).suffix("°"));
                ui.end_row();
            });
        });
        self.show_check_model = open;
        self.check_only_errors = only_errors;
        if thresholds != self.doc.validation {
            // Stored per document: they are about this model's scale and intent.
            self.doc.validation = thresholds;
            self.mark_changed();
        }
        if let Some(fault) = pick {
            self.select_fault(&fault);
        }
    }

    /// Selects a fault's elements and frames them.
    fn select_fault(&mut self, fault: &Fault) {
        self.selection.clear();
        for &v in &fault.vertices {
            self.selection.toggle(Selection::Vertex(v));
        }
        for &(a, b) in &fault.edges {
            self.selection.toggle(Selection::Edge(a, b));
        }
        for &f in &fault.faces {
            self.selection.toggle(Selection::Face(f));
        }
        let mesh = self.cage();
        let placement = self.control_placement();
        let mut points: Vec<Point3<f32>> = Vec::new();
        for part in self.selection.parts() {
            match *part {
                Selection::Vertex(v) => points.push(vertex_pos(mesh, &placement, v)),
                Selection::Edge(a, b) => points.extend([vertex_pos(mesh, &placement, a), vertex_pos(mesh, &placement, b)]),
                Selection::Face(f) => points.extend(mesh.face_vertices(f).map(|v| vertex_pos(mesh, &placement, v))),
                Selection::None => {}
            }
        }
        if points.is_empty() {
            return;
        }
        let centre = Point3::from(points.iter().map(|p| p.coords).sum::<nalgebra::Vector3<f32>>() / points.len() as f32);
        let radius = points.iter().map(|p| (p - centre).norm()).fold(0.0f32, f32::max).max(1.0);
        self.camera.frame(centre, radius);
    }

    /// The marks on the cage: red always, amber while the panel is open or just after an edit.
    /// Drawn as overlays, so none of it is picked -- in Field mode too, where the cage itself is
    /// not drawn. (A dashed ghost of the last valid cage used to draw here while the cage was red;
    /// the owner dropped it on 2026-10-24, since beside the cage itself it read as a stale copy.)
    pub(crate) fn draw_cage_faults(&self, painter: &egui::Painter, aspect: f32, rect: egui::Rect) {
        let Some(body) = self.doc.cage_body(self.active_cage) else { return };
        let placement = self.control_placement();
        let mesh = body.mesh();
        let (red, amber) = self.cage_faults();
        let show_amber = self.show_check_model || self.amber_until.is_some_and(|t| Instant::now() < t);
        for (faults, colour, fill, weight) in [
            (&amber, WARNING_COLOR, WARNING_FILL, 2.0f32),
            (&red, ERROR_COLOR, ERROR_FILL, 4.0f32),
        ] {
            if faults.is_empty() || (colour == WARNING_COLOR && !show_amber) {
                continue;
            }
            for fault in faults.iter() {
                for &v in &fault.vertices {
                    draw_highlight(painter, mesh, &placement, &self.camera, aspect, rect, Selection::Vertex(v), colour, fill, weight);
                }
                for &(a, b) in &fault.edges {
                    draw_highlight(painter, mesh, &placement, &self.camera, aspect, rect, Selection::Edge(a, b), colour, fill, weight);
                }
                for &f in &fault.faces {
                    draw_highlight(painter, mesh, &placement, &self.camera, aspect, rect, Selection::Face(f), colour, fill, weight * 0.5);
                }
            }
        }
    }
}

/// One fault as a line: the check, the count, and what it measured.
pub(crate) fn fault_line(fault: &Fault) -> String {
    let n = fault.faces.len().max(fault.edges.len()).max(fault.vertices.len());
    let what = match fault.check {
        Check::Open => format!("{} hole{}, {} edge{}", fault.measure.unwrap_or(0.0) as usize, plural(fault.measure.unwrap_or(0.0) as usize), fault.edges.len(), plural(fault.edges.len())),
        Check::StarPoint => format!("vertex {} of valence {}", fault.vertices[0].index(), fault.measure.unwrap_or(0.0) as usize),
        Check::FoldedFace => format!("{} face{}, up to {:.0}°", n, plural(n), fault.measure.unwrap_or(0.0)),
        Check::SelfIntersection => format!("{} face{}", n, plural(n)),
        _ => {
            let kind = if !fault.faces.is_empty() { "face" } else if !fault.edges.is_empty() { "edge" } else { "vertex" };
            if fault.check == Check::CoincidentVertices || fault.check == Check::NearCoincidentVertices {
                format!("{} pair{}", n / 2, plural(n / 2))
            } else {
                format!("{n} {kind}{}", plural(n))
            }
        }
    };
    format!("{}: {what}", fault.check.label())
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}
