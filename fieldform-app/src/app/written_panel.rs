// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Expression block in the application: its verb, its panel and its verify action
//! (backlog item 40, `expression-field-spec.md` §3.2, §5.1, §9, §10).

use crate::*;
use fieldform_scene::{Frame, Function, Node, Written, WrittenClaim};

/// What verify measured for a block, and when.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WrittenVerdict {
    /// How far the field claimed more room than there was, in millimetres: nought is the
    /// contract kept, anything above it is holes in the traced picture.
    pub(crate) overstatement: f64,
    /// How far the gradient's length strayed from one, where it was measured.
    pub(crate) deviation: f64,
    /// Points of the region at which the text had no value.
    pub(crate) missing: usize,
    pub(crate) probes: usize,
    /// The divisor that would have kept the contract, where one would.
    pub(crate) suggested: Option<f64>,
    pub(crate) when: String,
}

/// The sentence a person needs before they paste (`expression-field-spec.md` §5.1).
pub(crate) const WRITTEN_HELP: &str = "Expressions here are v1. Angles are in degrees, not radians. There is no vector type \
and no p — write coordinates as x, y, z. There are no conditionals; use min, max, clamp and \
sign. SDF code copied from published sources will not work as written — and the parts that do \
parse may draw the wrong shape rather than failing. Rewrite it rather than pasting it.";

/// The text an Expression block starts with: a sphere, written in the distance idiom.
pub(crate) fn default_text(radius: f64) -> String {
    format!("sqrt(x*x + y*y + z*z) - {radius}")
}

/// Whether a pasted text carries a float within a hair of π, 2π or π/2 -- radian code the
/// `pi` rule cannot catch, since nobody typed the word.
pub(crate) fn looks_like_radians(text: &str) -> bool {
    let mut number = String::new();
    let mut found = false;
    for c in text.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() || c == '.' {
            number.push(c);
            continue;
        }
        if let Ok(v) = number.parse::<f64>() {
            for known in [std::f64::consts::PI, std::f64::consts::TAU, std::f64::consts::FRAC_PI_2] {
                if (v - known).abs() < 0.002 {
                    found = true;
                }
            }
        }
        number.clear();
    }
    found
}

impl App {
    /// `expression [text="…"] [size=] [divisor=] [claim=distance|bound] [origin=] [axis=] ['name']`:
    /// an Expression block, selected. With no keys it is a written sphere.
    pub(crate) fn expression_command(&mut self, args: &Args) {
        let radius = DEFAULT_BODY_MM * 0.5;
        let names = |name: &str| self.doc.parameter_named(name);
        let text = match Written::parse(&default_text(radius), &names) {
            Ok(text) => text,
            Err(e) => {
                self.warn(format!("the default expression did not parse: {e}"));
                return;
            }
        };
        let side = ParamRef::Literal(DEFAULT_BODY_MM * 1.2);
        let node = Node::Expression {
            text,
            frame: Frame::default(),
            size: [side, side, side],
            divisor: ParamRef::Literal(1.0),
            claim: WrittenClaim::LowerBound,
        };
        let node = match console_keys::expression_with_keys(node, "expression", args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let mut graph = self.part_graph().clone();
        let id = graph.add(node);
        self.set_part_graph(graph);
        self.select_body(Some(id));
        self.boolean_b = Some(id);
        if self.boolean_a.is_none() || self.boolean_a == Some(id) {
            self.boolean_a = self.doc.listed_bodies(self.part_graph()).into_iter().find(|b| *b != id);
        }
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say("Added a written field: edit its text in the panel, and Verify measures what it claims");
    }

    /// The Expression block's panel: the text, the help a person needs before they paste, the
    /// region and its frame, the divisor and the claim, and verify's readout.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expression_controls(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        text: Written,
        frame: Frame,
        size: [ParamRef; 3],
        divisor: ParamRef,
        claim: WrittenClaim,
    ) {
        ui.label("A field written as arithmetic over x, y, z in its frame, cut to its region.");
        ui.collapsing("Before you paste", |ui| {
            ui.label(egui::RichText::new(WRITTEN_HELP).small());
            ui.separator();
            ui.label(egui::RichText::new("Functions, angles in degrees:").small().strong());
            for f in Function::ALL {
                ui.label(egui::RichText::new(format!("{} — {}", f.name(), f.describes())).small().weak());
            }
        });
        ui.separator();

        // The text: typed into a draft, committed on Enter, the document unchanged until the
        // text is accepted (§9). The draft is the application's; the node holds the last
        // accepted text.
        let draft = self.written_drafts.entry(id).or_insert_with(|| text.source().to_string());
        let mut commit = false;
        ui.horizontal(|ui| {
            row_label(ui, "Text:");
            let response = ui.add(egui::TextEdit::singleline(draft).desired_width(f32::INFINITY).hint_text("sqrt(x*x + y*y + z*z) - radius"));
            if response.has_focus()
                && ui.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Paste(t) if looks_like_radians(t))))
            {
                self.written_notices.insert(
                    id,
                    "That paste carries a number near π: angles here are in degrees, so a radian formula draws the wrong shape.".to_string(),
                );
            }
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                commit = true;
            }
        });
        if let Some(notice) = self.written_notices.get(&id).cloned() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(notice).small().color(ui.visuals().warn_fg_color));
                if ui.small_button("Dismiss").clicked() {
                    self.written_notices.remove(&id);
                }
            });
        }
        if let Some(error) = self.written_errors.get(&id).cloned() {
            // The caret where the parser stopped, under the text as the Number block shows it.
            let caret = format!("{}^", " ".repeat(error.0));
            ui.label(egui::RichText::new(caret).monospace().small().color(ui.visuals().error_fg_color));
            ui.label(egui::RichText::new(&error.1).small().color(ui.visuals().error_fg_color));
        }
        let draft_now = self.written_drafts.get(&id).cloned().unwrap_or_default();
        let mut new_text = text.clone();
        if commit && draft_now != text.source() {
            match Written::parse(&draft_now, &|name| self.doc.parameter_named(name)) {
                Ok(parsed) => {
                    self.written_errors.remove(&id);
                    self.written_verdicts.remove(&id);
                    new_text = parsed;
                }
                Err(e) => {
                    self.written_errors.insert(id, (e.offset, e.message.clone()));
                }
            }
        }

        ui.separator();
        let new_frame = self.frame_controls(ui, id, "Axis:", frame.clone(), None);
        ui.separator();
        let mut new_size = size.clone();
        for (axis, slot) in ["X:", "Y:", "Z:"].iter().zip(new_size.iter_mut()) {
            let (v, _) = self.parametric_field(ui, axis, slot.clone(), 0.5, POSITIVE_MM, "The region's full extent along this axis of the frame.");
            *slot = v;
        }
        ui.separator();
        let (new_divisor, _) = self.parametric_field_in(
            ui,
            "Divisor:",
            divisor.clone(),
            0.1,
            0.001..=1.0e6,
            "k: how fast the text's value can change per millimetre, at most. The value is divided by it so it never claims more room than there is. One for a field written as a distance.",
            "",
        );
        let mut new_claim = claim;
        ui.horizontal(|ui| {
            row_label(ui, "Claim:");
            egui::ComboBox::from_id_salt(("written claim", id.raw()))
                .selected_text(match new_claim { WrittenClaim::Distance => "Distance", WrittenClaim::LowerBound => "Lower bound" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut new_claim, WrittenClaim::LowerBound, "Lower bound").on_hover_text("Safe: booleans are exact, tracing is safe, and a blend against it says it is approximate.");
                    ui.selectable_value(&mut new_claim, WrittenClaim::Distance, "Distance").on_hover_text("The value is the distance: an exact fillet radius. Verify checks it.");
                });
        });

        if new_text != text || new_frame != frame || new_size != size || new_divisor != divisor || new_claim != claim {
            self.replace_node(
                id,
                Node::Expression { text: new_text, frame: new_frame, size: new_size, divisor: new_divisor, claim: new_claim },
            );
            if new_size != size || new_divisor != divisor || new_claim != claim {
                self.written_verdicts.remove(&id);
            }
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Verify").on_hover_text("Samples the region and measures what the field claims: whether it ever overstates the distance, how far its gradient is from unit, and where the text has no value.").clicked() {
                self.verify_written(id);
            }
            match self.written_verdicts.get(&id) {
                None => {
                    ui.label(egui::RichText::new("Unverified").small().weak());
                }
                Some(v) => {
                    let over = if v.overstatement > 1e-9 {
                        egui::RichText::new(format!("overstates by {:.3} mm", v.overstatement)).small().color(ui.visuals().error_fg_color)
                    } else {
                        egui::RichText::new("never overstates").small()
                    };
                    ui.label(over);
                    ui.label(egui::RichText::new(format!("gradient off unit by {:.3}", v.deviation)).small());
                    if v.missing > 0 {
                        ui.label(egui::RichText::new(format!("{} of {} points have no value", v.missing, v.probes)).small().color(ui.visuals().warn_fg_color));
                    }
                    ui.label(egui::RichText::new(&v.when).small().weak());
                }
            }
        });
        if let Some(v) = self.written_verdicts.get(&id)
            && let Some(k) = v.suggested
        {
            let now = self.resolve(divisor.clone());
            ui.label(
                egui::RichText::new(format!("A divisor of {k:.3} would keep the contract (now {now:.3}).")).small().color(ui.visuals().warn_fg_color),
            );
        }
    }

    /// Verify (§3.2): the block compiled alone, its region sampled, and the three measurements
    /// the design names taken with the checks that already exist.
    pub(crate) fn verify_written(&mut self, id: NodeId) {
        let graph = self.part_graph().clone();
        let Some(Node::Expression { divisor, .. }) = graph.node(id).cloned() else { return };
        let alone = expect::graph_of(&graph, id);
        let Some((lo, hi)) = fieldform_field::part_extent(&self.doc, &alone).as_box() else {
            self.warn("the written field has no region to sample");
            return;
        };
        let field = match fieldform_field::GraphField::compile_graph(&self.doc, &alone, MEASURE_TOLERANCE_MM) {
            Ok(f) => f,
            Err(e) => {
                self.warn(format!("the written field cannot be measured: {e}"));
                return;
            }
        };
        // Scattered through the region, off every lattice.
        let span = hi - lo;
        let probes: Vec<Point3<f64>> = (0..4000)
            .map(|i| {
                let t = i as f64 + 0.37;
                let u = |a: f64| 0.5 + 0.5 * (t * a).sin();
                Point3::new(lo.x + span.x * u(0.618), lo.y + span.y * u(0.271), lo.z + span.z * u(0.173))
            })
            .collect();
        use fieldform_field::ScalarField as _;
        let readings: Vec<f64> = probes.iter().map(|p| field.distance(*p)).collect();
        let missing = readings.iter().filter(|d| **d > 1e100).count();
        let outside: Vec<Point3<f64>> = probes.iter().zip(&readings).filter(|(_, d)| **d > 0.0 && **d < 1e100).map(|(p, _)| *p).collect();
        let step = (span.x.max(span.y).max(span.z) * 0.002).max(1e-4);
        let overstatement = fieldform_field::overstatement(&field, &outside, &fieldform_field::probe_directions(), step, 1.2);
        // Away from the surface and the region's own faces, where a gradient is defined.
        let inner: Vec<Point3<f64>> = probes
            .iter()
            .zip(&readings)
            .filter(|(p, d)| d.abs() > step * 4.0 && **d < 1e100 && (lo.x + step * 4.0..hi.x - step * 4.0).contains(&p.x) && (lo.y + step * 4.0..hi.y - step * 4.0).contains(&p.y) && (lo.z + step * 4.0..hi.z - step * 4.0).contains(&p.z))
            .map(|(p, _)| *p)
            .collect();
        let deviation = fieldform_field::gradient_deviation(&field, &inner, step * 0.5);
        let k_now = self.resolve(divisor);
        // A gradient over one is the rate the divisor has to cover: the value is already over
        // `k`, so the rate it reaches times `k` is what `k` should have been.
        let suggested = (overstatement > 1e-9 || deviation > 0.02).then(|| k_now * (1.0 + deviation));
        let when = image_export::now_tokens();
        self.written_verdicts.insert(
            id,
            WrittenVerdict { overstatement, deviation, missing, probes: probes.len(), suggested, when: format!("{} {}", when.0, when.1) },
        );
        let name = self.derived_label(id);
        if overstatement > 1e-9 {
            self.warn(format!("{name} overstates the distance by {overstatement:.3} mm: a ray following it walks through the surface. A divisor of {:.3} would keep the contract.", k_now * (1.0 + deviation)));
        } else {
            self.say(format!("{name}: never overstates; gradient off unit by {deviation:.3}; {missing} of {} points have no value", probes.len()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_paste_near_pi_is_noticed_and_degrees_are_not() {
        assert!(looks_like_radians("sin(x * 6.28318 / cell)"));
        assert!(looks_like_radians("3.14159 * r"));
        assert!(looks_like_radians("atan(1) * 1.5708"));
        assert!(!looks_like_radians("sin(x * 360 / cell)"));
        assert!(!looks_like_radians("3.0 + 1.4"));
    }
}
