// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Drawing the traced viewport, and saying why when nothing is traced.

use crate::*;

impl App {
    /// Draws the composed field by evaluating it per pixel on the GPU.
    ///
    /// The expensive half — compiling the shader and baking the cage — happens only when
    /// something has actually changed, which is why staleness is checked before a `Scene` is
    /// gathered rather than after.
    pub(crate) fn draw_traced(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        // **First, before anything is asked of the device: whether it is the device everything
        // was made on** (loop spec S4). A lost device fails every call rather than waiting on
        // one, and everything the view holds — the march's pipelines and buffers, the lighting
        // pass, the textures egui samples — belongs to it. The callback that found the loss may
        // have run on any thread, so this is the frame answering it for the picture, as
        // `keep_the_document_if_the_gpu_was_lost` answers it for the document — that one runs
        // first and last in every frame, so the work is kept before this says where it is.
        if !self.traced.ready_to_draw_on(
            &self.wgpu_render_state,
            self.gpu_lost.load(std::sync::atomic::Ordering::Acquire),
        ) {
            // Nothing is traced, nothing is submitted and nothing of the old picture is left on
            // screen; what is said is the view's own reason for an empty viewport.
            self.say_why_nothing_is_traced();
            return;
        }
        // Told every frame rather than on change, because it is the one place that knows both
        // the setting and the view, and setting it to what it already is does nothing. The
        // rebuild below is handed its walks from this same reading, so what the view compares
        // is what it was built from.
        // The rung's cells, held down for a big cage: what the settled bake costs grows with the
        // cage's faces, and its accuracy does not (`display_quality::cell_cap`).
        let biggest_cage = self
            .part_graph()
            .nodes()
            .filter_map(|(_, node)| if let Node::Cage(id) = node { self.doc.cage_body(*id) } else { None })
            .map(|body| body.mesh().face_count())
            .max()
            .unwrap_or(0);
        let fineness = self.view.quality.fineness_for(biggest_cage);
        self.traced.set_fineness(fineness);
        // A settled band that has finished being computed elsewhere comes in here, and asks for
        // the rebuild that draws with it. While one is still being computed the viewport keeps
        // asking for frames — nothing else would, since the news arrives from another thread.
        if self.traced.collect_bakes() || self.traced.is_baking() {
            ui.ctx().request_repaint();
        }
        // The same for a shader the driver is still chewing on. Until it lands the viewport
        // keeps drawing the shape it had, which is why the wait is said out loud: what is on
        // screen is honest about the *old* document, and nothing else would explain that.
        if self.traced.collect_compile() || self.traced.is_compiling().is_some() {
            ui.ctx().request_repaint();
        }
        if let Some(waited) = self.traced.compile_worth_saying() {
            self.say(format!(
                "The driver is still compiling this shape's shader ({:.0} s so far) — \
                 what is drawn is the shape before the change until it lands.",
                waited.as_secs_f32()
            ));
        }
        let tracing = std::time::Instant::now();
        if self.traced.is_stale() {
            // With no field generator applied there is still a cage to trace, and drawing
            // nothing was indistinguishable from the renderer being broken. `pick_bounds`
            // already answers for both cases.
            // What the kernel refuses, the viewport does not draw.
            //
            // These used to disagree: `fieldform_gpu::generate` checks for unknown nodes and
            // cycles but says nothing about normalization, so a blend against a lattice --
            // refused by D8 in `GraphField::compile_graph` -- was lowered to WGSL and traced
            // anyway. The result was a document you could see and could not export, measure or
            // click on, and the shape on screen was the one D8 exists to stop you trusting: a
            // blend mixes by magnitude, so an operand that understates its distance pulls the
            // seam toward itself and the radius is not the radius asked for.
            //
            // The CPU compile is the authority rather than a second opinion, because it is the
            // one that owns the rule. Drawing nothing and explaining why in the panel is worse
            // to look at than a wrong shape and better than one.
            let graph = if self.compile_error.is_some() {
                Graph::empty()
            } else {
                self.display_graph()
            };
            let bounds = self.pick_bounds();
            let extent = bounds.1 - bounds.0;
            let cell = extent.x.max(extent.y).max(extent.z) / self.mesh_resolution as f64;
            let tolerance = (cell * 0.5).max(1e-4);

            // An angle, not a tolerance. The tube's normal turns a corner at every joint, and
            // shading *is* the normal, so what shows is the size of that corner — where a
            // tolerance only says how far the spine strays from the curve. A thousandth of the
            // extent sounds tight and measured 6.35° per joint on a 310 mm cubic, which is
            // bands across the tube and, through a blend, across whatever it was cut into.
            // One degree is the threshold where they stop being visible; see
            // `Curve::tessellate_to_angle` for the measurements.
            //
            // The shader walks these points per sample, so the cost is linear in their number
            // and a fine spine is cheap next to being visibly wrong.
            let spine_degrees = fineness.spine_degrees;
            let spine_floor = (extent.norm() * 1e-5).max(1e-4);
            // Tessellated by id and handed over unordered: the shader binds curves in the
            // order its own compilation interned them, and `traced` is where that order is
            // known. Producing the list in what this side guessed that order to be worked
            // only while there was one curve.
            // **`Node::curves` rather than a match on `Node::Pipe`.** A hand-written match
            // is a list of node kinds that name a curve, kept in one more place than the one
            // that knows -- and the failure when it falls behind is silent in the worst way:
            // the curve never reaches the shader, the span arrives with a count of zero, and
            // the body renders as though nothing had been asked of it. An attractor drawn
            // toward nothing looks exactly like an attractor set to zero. The same omission
            // in `gpu_matches_cpu` is what it cost to find this one.
            //
            // Extra entries are free -- a profile's curve is keyed by node rather than by id
            // on the other side and simply goes unread -- so over-collecting is the safe
            // direction and under-collecting is not.
            //
            // Through the view, which keeps each spine against the curve, angle and floor it was
            // walked from: a rebuild that moved no curve walks none. The angle here is the
            // rung's, which a settled rebuild walks at; a draft walks at two degrees, which is
            // all a drag frame can show, and the view is what knows which rebuild comes next.
            let read: std::collections::BTreeSet<CurveId> =
                graph.nodes().flat_map(|(_, node)| node.curves()).collect();
            let curves = self.traced.spines(&self.doc.curves, read, spine_degrees, spine_floor);
            // Cloned because `Scene` borrows it while `self.traced` is borrowed mutably. A
            // control cage is scaffolding and small by design, and this happens on a rebuild
            // rather than per frame.
            let cage = self.cage().clone();
            let device = self.wgpu_render_state.device.clone();
            let queue = self.wgpu_render_state.queue.clone();

            self.traced.rebuild(
                &device,
                &queue,
                &traced::Scene {
                    document: &self.doc,
                    meshes: self.doc.meshes.iter().map(|(id, m)| (*id, m)).collect(),
                    graph: &graph,
                    // From the field compiled for picking, over this same graph, so the shader
                    // divides by exactly what the CPU divided by.
                    morph_divisors: self
                        .pick_field
                        .as_ref()
                        .map(|f| f.morph_divisors())
                        .unwrap_or(&[]),
                    tolerance_mm: tolerance,
                    // The occlusion's radius for the scene the view will be drawn with -- the
                    // same box, so the same radius `refresh_scene_bounds` hands the camera.
                    reads_mm: fieldform_gpu::Occlusion::reach(extent.norm() * 0.5),
                    profile_fineness: self.view.quality.profile_bake_fineness(),
                    profile_degrees: fineness.profile_degrees,
                    curves,
                    cage_bounds: cage_box(&cage),
                },
            );
        }

        // Every parameter's value, as the overlay's readings keep the table: the closure outlives
        // this borrow of the document, so it carries the table rather than the document. A failed
        // expression is simply absent, which the compile reports the way it reports an unknown
        // name.
        let readings = self.overlay_readings();
        let lookup = move |id: fieldform_scene::ParamId| readings.values.get(&id).copied();

        let sections = self.section_half_spaces();
        let selected = self.selected_body_index();
        // The part's own material, or the default when it has none. Read every frame rather
        // than kept, so an undo that puts a material back is drawn without anyone remembering
        // to say so.
        let material = self
            .doc
            .parts
            .get(&self.active_part)
            .and_then(|p| p.material.clone())
            .unwrap_or_default();
        let view = traced::View {
            selected,
            sections,
            eye: self.camera.eye.cast::<f64>(),
            target: self.camera.target.cast::<f64>(),
            up: self.camera.up.cast::<f64>(),
            scene: (
                Point3::new(
                    self.camera.scene_centre.x as f64,
                    self.camera.scene_centre.y as f64,
                    self.camera.scene_centre.z as f64,
                ),
                self.camera.scene_radius as f64,
            ),
            fov_deg: 45.0,
            orthographic: self.camera.projection == Projection::Orthographic,
            lookup: &lookup,
            material,
            occlusion: self.view.idle_occlusion && self.view.quality.occlusion(),
            reflections: self.view.idle_reflections && self.view.quality.reflections(),
            exact_finish: self.view.quality.exact_finish(),
            environment: self.view.environment,
            draft: self.draft_overlay(),
            max_pixels: self.view.quality.still_pixels(),
            interactive_pixels: self.view.quality.drag_pixels(),
            still_frames: self.view.quality.still_frames(),
        };
        self.traced.draw(ui, rect, &view, &self.wgpu_render_state);

        self.note_stage("rebuilding the traced view", tracing);
        // Its three parts, so a slow one says which: generating the shader, preparing what the
        // card is handed (bands, spines, surfaces), or the driver's own pipeline and uploads.
        if let Some((quality, cost)) = self.traced.take_rebuild_parts() {
            let tier = match quality {
                traced::Quality::Draft => "draft",
                traced::Quality::Final => "settled",
            };
            let mut parts = vec![
                (format!("{tier}: generating the shader"), cost.generate_ms),
                (format!("{tier}: preparing the bands"), cost.prepare_ms),
                (format!("{tier}: the driver's pipeline and uploads"), cost.pipeline_ms),
                (format!("{tier}: letting the old renderer go"), cost.release_ms),
            ];
            // The renderer's own four, so "the driver's pipeline and uploads" says which.
            if let Some(build) = self.traced.take_build_parts() {
                for (what, ms) in build.parts() {
                    parts.push((format!("{tier}: {what}"), ms));
                }
            }
            for (leaf, field_ms, bake_ms) in self.traced.take_bake_parts() {
                parts.push((format!("building {leaf}'s field"), field_ms));
                parts.push((format!("baking {leaf}"), bake_ms));
            }
            for (what, ms) in parts {
                if ms >= 1.0 {
                    self.perf.stages.push((Box::leak(what.into_boxed_str()), ms));
                }
            }
        }
        self.say_why_nothing_is_traced();
    }

    /// Says the traced view's error in the console, once, when it changes.
    ///
    /// A viewport that silently shows nothing is indistinguishable from a model that is not
    /// there, so a shader that failed to compile, a device that refused a buffer or a device
    /// that has gone has to be said rather than stopping at the field that recorded it.
    ///
    /// **Only when it changes.** This is called every frame and the error it reads is a stored
    /// field, not an event — so a part with no bodies is un-compilable for as long as it stays
    /// that way, and saying so each time filled the console with one sentence at the frame
    /// rate. The condition is continuous; the report is not.
    pub(crate) fn say_why_nothing_is_traced(&mut self) {
        let error = self.traced.error().map(|e| e.to_string());
        if error != self.console.traced_error_said {
            if let Some(why) = &error {
                self.warn(why.clone());
            }
            self.console.traced_error_said = error;
        }
    }

    // `build_composed_mesh` lived here. It was the viewport's only use of `mesh_field`, and
    // its output could not be made to look right: the grid is uniform over a box stretched by
    // the swept curve, so the tube was 3.1 cells across at the default and 9.3 at triple the
    // resolution, with silhouette faceting that stopped improving after about 124 cells.
    // Raising the setting added triangles to the sphere and left the tube a polygon. The
    // traced view draws the same field per pixel, with no grid, no resolution setting and no
    // such floor. Meshing stays for export, where the cost is paid once and offline.
}
