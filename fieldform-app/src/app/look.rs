// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The material, the environment and the draft overlay, from the panel and from the console.

use crate::*;

impl App {
    /// `material preset=<name>` from the console, or `material` alone to hear the choices.
    pub(crate) fn set_material_from_console(&mut self, args: &Args) {
        let names: Vec<&str> = Material::presets().iter().map(|p| p.name).collect();
        let Some(wanted) = args.get("preset").and_then(|v| v.as_str().map(str::to_string)) else {
            self.say(format!(
                "Presets: {}. Say `material preset=<name>`, or `preset=default` to clear.",
                names.join(", ")
            ));
            return;
        };
        if wanted.eq_ignore_ascii_case("default") || wanted.eq_ignore_ascii_case("none") {
            self.set_part_material(None);
            self.say("Material: default");
            return;
        }
        match Material::preset(&wanted) {
            Some(m) => {
                let name = m.preset_name().unwrap_or("custom");
                self.set_part_material(if m == Material::default() { None } else { Some(m) });
                self.say(format!("Material: {name}"));
            }
            None => self.warn(format!(
                "No material called {wanted}. Presets: {}",
                names.join(", ")
            )),
        }
    }

    /// Lights the part by another room, and says what the room is for.
    pub(crate) fn set_environment(&mut self, choice: fieldform_gpu::EnvironmentChoice) {
        if self.view.environment == choice {
            return;
        }
        self.view.environment = choice;
        // Said when it is chosen: the zebra in particular does nothing visible on clay, and
        // the one sentence that explains why is only interesting once it has been picked.
        self.say(format!("Environment: {} — {}", choice.label(), choice.hint()));
    }

    /// `environment name=<studio|showroom|zebra>` from the console, or `environment` alone to
    /// hear the choices.
    pub(crate) fn set_environment_from_console(&mut self, args: &Args) {
        let names: Vec<&str> = fieldform_gpu::EnvironmentChoice::ALL
            .iter()
            .map(|c| c.label())
            .collect();
        let Some(wanted) = args.get("name").and_then(|v| v.as_str().map(str::to_string)) else {
            self.say(format!(
                "Environments: {}. Say `environment name=<which>`.",
                names.join(", ")
            ));
            return;
        };
        match fieldform_gpu::EnvironmentChoice::from_name(&wanted) {
            Some(choice) => self.set_environment(choice),
            None => self.warn(format!(
                "No environment called {wanted}. Environments: {}",
                names.join(", ")
            )),
        }
    }

    /// The draft overlay as the lighting pass wants it, or `None` when it is off or its pull
    /// cannot be read — a line that has been deleted, a typed direction of zero length.
    pub(crate) fn draft_overlay(&self) -> Option<fieldform_gpu::Draft> {
        if !self.view.draft.enabled {
            return None;
        }
        // The same function an extrude's sweep is read by, so "along that line" means the same
        // direction here that it means there.
        let mut pull = fieldform_field::sweep_direction(&self.doc, &self.view.draft.pull, &|r| {
            Some(self.resolve(*r))
        })?;
        if self.view.draft.flipped {
            pull = -pull;
        }
        Some(fieldform_gpu::Draft::new(
            pull,
            self.view.draft.angle_deg as f32,
            self.view.draft.tolerance_deg as f32,
        ))
    }

    /// `draft` toggles the overlay; `draft angle=3 tolerance=1 vector=<curve>` sets it up and
    /// turns it on; `draft off` turns it off. `line=` is accepted for `vector=`.
    pub(crate) fn draft_from_console(&mut self, args: &Args) {
        let mut set_up = false;
        if let Some(angle) = args.get("angle").and_then(Value::as_f64) {
            self.view.draft.angle_deg = angle.clamp(0.0, 45.0);
            set_up = true;
        }
        if let Some(tolerance) = args.get("tolerance").and_then(Value::as_f64) {
            self.view.draft.tolerance_deg = tolerance.clamp(0.01, 10.0);
            set_up = true;
        }
        let named = args
            .get("vector")
            .or_else(|| args.get("line"))
            .and_then(|v| v.as_str().map(str::to_string));
        if let Some(name) = named {
            match self.doc.curves.values().find(|c| c.name.eq_ignore_ascii_case(&name)) {
                Some(c) => {
                    self.view.draft.pull = fieldform_scene::Sweep::Along { curve: c.id };
                    set_up = true;
                }
                None => {
                    self.warn(format!("No curve called {name} to pull along"));
                    return;
                }
            }
        }
        if args.get("off").is_some() {
            self.view.draft.enabled = false;
        } else {
            self.view.draft.enabled = set_up || !self.view.draft.enabled;
        }
        if self.view.draft.enabled {
            self.say(format!(
                "Draft analysis on: blue has at least {:.1}\u{b0} of draft along the pull, red \
                 has not, blurred {:.1}\u{b0} either side. Flip the pull for the other half.",
                self.view.draft.angle_deg, self.view.draft.tolerance_deg
            ));
        } else {
            self.say("Draft analysis off");
        }
    }

    /// The draft-angle overlay's controls: on or off, which way the mould pulls, the angle a
    /// face needs and how softly the line is drawn.
    pub(crate) fn draft_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.view.draft.enabled, "Draft analysis").on_hover_text(
                "Colour each face by whether it has enough draft to leave a mould pulled in \
                 the chosen direction: soft blue where it does, soft red where it does not \
                 (vertical, or undercut). The line between blurs over the tolerance, so a \
                 curved face shows a band rather than a contour.",
            );
            if self.view.draft.enabled
                && ui
                    .small_button(if self.view.draft.flipped { "Flipped" } else { "Flip" })
                    .on_hover_text("Pull the other way: the read for the other half of the mould.")
                    .clicked()
            {
                self.view.draft.flipped = !self.view.draft.flipped;
            }
        });
        if !self.view.draft.enabled {
            return;
        }
        let pull = self.view.draft.pull;
        self.view.draft.pull = self.direction_controls(
            ui,
            "draft_pull",
            "Pull:",
            pull,
            None,
            DRAFT_WORDS,
            &[],
            None,
        );
        ui.horizontal(|ui| {
            ui.label("Angle:");
            ui.add(
                egui::DragValue::new(&mut self.view.draft.angle_deg)
                    .speed(0.1).max_decimals(4)
                    .range(0.0..=45.0)
                    .suffix("\u{b0}"),
            )
            .on_hover_text("The draft a face needs, in degrees from vertical along the pull.");
            ui.label("Tolerance:");
            ui.add(
                egui::DragValue::new(&mut self.view.draft.tolerance_deg)
                    .speed(0.05).max_decimals(4)
                    .range(0.01..=10.0)
                    .suffix("\u{b0}"),
            )
            .on_hover_text(
                "How far either side of the angle the boundary blurs. Wider is softer on a \
                 curved face.",
            );
        });
    }
}
