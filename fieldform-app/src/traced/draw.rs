// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A frame: the march sized to its tier, sent in bands, and a still averaged until it converges.

use super::*;

impl TracedView {
    /// Draws the field into `rect` using whatever was last built.
    ///
    /// Does nothing if [`is_stale`](Self::is_stale) still holds, so a caller that skipped the
    /// rebuild draws nothing rather than something out of date.
    pub fn draw(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        view: &View<'_>,
        render_state: &egui_wgpu::RenderState,
    ) {
        self.frame += 1;
        let (Some(renderer), Some(program)) = (&self.renderer, &self.program) else {
            return;
        };
        // Without blocking, so that what the GPU finished during the last frame reports now
        // rather than at the end of this one. wgpu fires its callbacks only when the device is
        // polled, and eframe polls at its own submit — which comes *after* this draw, so a cost
        // that arrived in time would otherwise be read a frame late, and a finished draft would
        // read as still in flight.
        let _ = render_state.device.poll(wgpu::PollType::Poll);

        // Values every frame. This is the cheap path the whole GPU route exists to provide:
        // no cage is retessellated, no field recompiled, no shader rebuilt.
        let values = resolve_values(&mut self.spare_values, &program.params, view.lookup);
        renderer.set_params(&values);
        #[cfg(feature = "march-stats")]
        {
            self.last_values = values.clone();
        }

        // **The skip grid follows the field, not the camera.** It is a fact about where the
        // surface is not, so an orbit never touches it and anything that moves the surface
        // refills it: a new renderer, a parameter's value, the cube it covers. Filled here,
        // straight after the values it is a function of, and ahead of any march this frame
        // submits. Only for a bounded part, since the cube is the scene's sphere's; and only
        // where the draft is dear enough to gain by it -- on a bare sphere three lookups a
        // step cost more than the readings they save (`SKIP_WORTH_MS`).
        let cube = [
            view.scene.0.x.to_bits(),
            view.scene.0.y.to_bits(),
            view.scene.0.z.to_bits(),
            view.scene.1.to_bits(),
        ];
        let draft_ms = self.draft_ms_per_pixel.map(|ms| {
            let cap = view.interactive_pixels.clamp(1, INTERACTIVE_PIXELS) as f32;
            let scale = ui.ctx().pixels_per_point();
            let (wide, high) = (rect.width() * scale, rect.height() * scale);
            let shrink = (cap / wide.max(high).max(1.0)).min(1.0);
            ms * (wide * shrink) * (high * shrink)
        });
        // Decided whenever a draft has been measured and kept between times, so that a drag
        // which rebuilds every step -- and so never has a measurement in hand -- keeps the
        // grid it had. Once on, it takes a draft half as dear to turn it off: the draft it
        // measures with the grid is the cheaper one, and judging that by the same line would
        // switch it off and on for ever.
        if let Some(ms) = draft_ms {
            let line = if self.skip_wanted { SKIP_WORTH_MS * 0.5 } else { SKIP_WORTH_MS };
            self.skip_wanted = ms >= line;
        }
        let wanted = self.scene_bounded && view.scene.1 > 0.0 && self.skip_wanted;
        if wanted {
            // Coarse while anything is changing, fine once it has stopped. The fine grid is
            // dearer to fill and a good deal better to march (`SKIP_CELLS_SETTLED`), and it is
            // kept for as long as the field stands -- an orbit afterwards marches it, since the
            // camera is not something a grid depends on.
            let rested = self.view_changed_at.map_or(true, |t| t.elapsed() >= SETTLE_AFTER);
            let cells = if rested { SKIP_CELLS_SETTLED } else { SKIP_CELLS };
            let current = self.skip_key.as_ref().is_some_and(|(id, seen, at, had)| {
                *id == renderer.identity() && *seen == values && *at == cube && *had >= cells
            });
            if !current {
                let reach = nalgebra::Vector3::repeat(view.scene.1);
                renderer.build_skip_grid(view.scene.0 - reach, view.scene.1 * 2.0, cells, true);
                self.skip_key = Some((renderer.identity(), values.clone(), cube, cells));
            }
        } else if self.skip_key.take().is_some() {
            renderer.clear_skip_grid();
        }

        let scale = ui.ctx().pixels_per_point();
        let wide = (rect.width() * scale).max(1.0);
        let high = (rect.height() * scale).max(1.0);

        // Anything different from last frame — a moved camera, a dragged slider, a picked
        // material — starts the still over from the interactive tier.
        let key = ViewKey::of(view, values, wide as u32, high as u32);
        if note_view(&mut self.last_view, &mut self.spare_values, key) {
            self.view_changed_at = Some(Instant::now());
            self.accumulated = 0;
            // Cancellation is free: whatever bands of a still were in hand are simply not
            // continued, and the still's texture is not shown until a whole frame of the new
            // view is in it.
            self.still_cursor = 0;
            self.still_band = 0;
            self.still_shown = false;
            // **The staleness guard for the growth cap**, and it is a clearing rather than a
            // test on arrival: a cost cannot say which view it measured, so the cap's input is
            // not taken from one. The next band may only grow on a band of this view.
            self.last_band_rows = None;
            // The real thing is about to run, and warms the card better than a stand-in.
            self.warm_until = None;
        }
        let since_change = self.view_changed_at.map_or(SETTLE_AFTER, |t| t.elapsed());
        let still = since_change >= SETTLE_AFTER;
        // Never more than the constants, whatever the caller asks: they are the sizes the
        // G-buffer and the jitter sequence were measured at.
        let still_frames = view.still_frames.clamp(1, STILL_FRAMES);
        let interactive_pixels = view.interactive_pixels.clamp(1, INTERACTIVE_PIXELS);

        if still && self.accumulated >= still_frames {
            // Converged. Show what is there and trace nothing: this is the branch that makes
            // an idle viewport cost nothing.
            let shown = self.shading.as_ref().map_or(0, |s| s.generation());
            if let Some((_, id)) = self.display.iter().find(|(g, _)| *g == shown) {
                paint(ui, rect, *id);
            }

            // ...except for the moment before it is about to cost a great deal. The pointer
            // moving over a viewport whose card has sat idle is the best warning there is that
            // a drag is coming, and the first draft of that drag is the slow one: the card has
            // clocked down. So the clocks are brought up while the hand is still travelling.
            // See `CLOCKS_HOLD` for what was measured and what this does not do.
            let now = Instant::now();
            let cold = self.last_gpu_work.map_or(true, |t| now.duration_since(t) >= CLOCKS_HOLD);
            let stirring =
                ui.rect_contains_pointer(rect) && ui.ctx().input(|i| i.pointer.is_moving());
            // What a draft of this view is expected to cost with the clocks up. A view whose
            // draft is a few milliseconds is a view whose slow first draft nobody sees, and it
            // is not worth a watt.
            let draft_cap = interactive_pixels.min(DRAFT_LADDER[self.draft_rung]) as f32;
            let shrink = (draft_cap / wide.max(high)).min(1.0);
            let draft_pixels = (wide * shrink) * (high * shrink);
            let dear = self
                .draft_ms_per_pixel
                .is_some_and(|ms| ms * draft_pixels >= DRAFT_BUDGET_MS / WORTH_WARMING);
            if self.warm_until.is_none() && cold && stirring && dear {
                self.warm_until = Some(now + WARM_FOR);
            }
            if let Some(until) = self.warm_until {
                if now >= until {
                    self.warm_until = None;
                } else {
                    if let (Some(shading), Some((camera, _, _, budget))) =
                        (&mut self.shading, self.last_frame)
                    {
                        // One band in flight, as everywhere else: a press that lands during
                        // the warm-up waits for one band of it and no more.
                        if !shading.busy() {
                            let mut camera = camera;
                            camera.jitter = [0.0, 0.0];
                            camera.occlusion = None;
                            let pixels = (draft_pixels / WARM_SHARE).ceil() as u32;
                            if let Some(ticket) =
                                renderer.warm_rows(shading, &camera, budget, pixels)
                            {
                                self.last_gpu_work = Some(now);
                                self.submitted.push(Submitted {
                                    ticket,
                                    frame: self.frame,
                                    still: false,
                                    width: pixels,
                                    height: 1,
                                    rows: 1,
                                    band: None,
                                    still_frame: None,
                                    skipped: 0,
                                    warm: true,
                                });
                            }
                        }
                    }
                    ui.ctx().request_repaint();
                }
            }
            return;
        }

        // A still draws at the size the setting asks for, capped at what a G-buffer may be
        // asked to hold; a drag keeps to its own cap regardless, since sharpness is what a
        // gesture trades away. Coarse asks for 720, which makes the two tiers one size.
        let still_cap = if view.max_pixels == 0 {
            STILL_PIXELS
        } else {
            view.max_pixels.min(STILL_PIXELS)
        };
        let cap = if still {
            still_cap
        } else {
            interactive_pixels.min(still_cap).min(DRAFT_LADDER[self.draft_rung])
        };
        let shrink = (cap as f32 / wide.max(high)).min(1.0);
        let width = (wide * shrink).round().max(1.0) as u32;
        let height = (high * shrink).round().max(1.0) as u32;

        let mut camera = Camera::looking_at(
            view.eye,
            view.target,
            view.up,
            view.scene,
            view.fov_deg,
            width as f64 / height as f64,
        );
        camera.sections = fieldform_gpu::Sections::of(&view.sections);
        // The app's sphere is the union of every bounded body's extent, so it encloses all of
        // them exactly when there is no unbounded one -- which the rebuild checked.
        camera.start_at_scene = self.scene_bounded;
        // **No per-tile cull while a shader is in flight.** The renderer on screen is the
        // previous program, and its cull extents are literals fixed when *it* was generated --
        // so they describe the document as it was before whatever edit started this compile. The
        // numbers a frame draws with are current, the extents are not, and the gap is a body
        // culled out of tiles it has since grown into. See `Camera::cull`.
        //
        // The same hole the box guards have always had, and this closes it for both: a guard is
        // skipped where the body's own extent slots are, and both read the same stale literals.
        camera.cull = self.compiling.is_none();
        // The same number the step budget is bought with, for the same reason: a divided field
        // is honest and is not a distance estimate. The budget is what stops it losing the
        // silhouette; this is what stops it shading black. See `Camera::value_scale`.
        camera.value_scale = self.value_scale.unwrap_or(1.0);
        if view.orthographic {
            // The same half-height the app's own projection matrix uses: the perspective one
            // taken out to the distance it is being viewed from. Sharing the derivation keeps
            // the traced surface the same size as the cage drawn over it in overlay mode — two
            // framings of one scene would show as the wireframe sitting a little inside or
            // outside the surface it describes.
            let distance = (view.eye - view.target).norm();
            camera.orthographic = true;
            camera.half_height = (distance * (view.fov_deg.to_radians() * 0.5).tan()).max(1e-9);
            // Parallel rays start on a plane through the eye rather than at a point, so a ray
            // at the edge of the frame begins further out than the centre one does. The budget
            // has to cover that, or the model clips along its own silhouette.
            camera.far += camera.half_height * 2.0;
        }

        // The studio rig follows the camera, as the viewport's one light always has, so the
        // model is lit from wherever it is being looked at. The environment does not: its
        // reflections turn as you orbit, which is what makes a polished surface read as one.
        let mut look = Look::for_environment(
            view.environment,
            view.material.clone(),
            camera.forward,
            camera.right,
            camera.up,
        );
        look.selected = view.selected;
        look.draft = view.draft;
        // Two screen pixels wide whatever the frame's scale: a 720-pixel drag frame shown
        // three times enlarged wants one texel, a full-size still wants two.
        look.outline_radius = (2.0 * shrink).round().clamp(1.0, 3.0) as u32;
        let accumulated = if still { self.accumulated } else { 0 };
        if still {
            camera.jitter = halton_jitter(accumulated);
            // The frames somebody looks at finish on the outline (`Camera::exact_finish`), on
            // the rung that asks (`DisplayQuality::exact_finish`, every rung since the read is
            // local), and by the env var for an example or a rung pulled back.
            // `FIELDFORM_BAKE_FINISH=1` holds every still to its bake, the control arm for a
            // picture that may or may not be the finish's doing.
            camera.exact_finish = (view.exact_finish
                || std::env::var_os("FIELDFORM_EXACT_FINISH").is_some())
                && std::env::var_os("FIELDFORM_BAKE_FINISH").is_none();
            if view.occlusion && view.environment.wants_occlusion() {
                camera.occlusion =
                    Some(Occlusion::for_scene(look.lights[0].direction, view.scene.1));
            }
        }

        let device = &render_state.device;
        let queue = &render_state.queue;
        // Baked on first use and kept: tens of milliseconds on the development machine —
        // `material_sweep` prints the number — and independent of everything in the document.
        let maps = self
            .environments
            .entry(view.environment)
            .or_insert_with(|| view.environment.bake());
        let shading = match &mut self.shading {
            Some(s) => s,
            None => match ShadingResources::new(device, queue, maps) {
                Ok(s) => {
                    self.environment = Some(view.environment);
                    // The lighting pass is of this device as much as the march's pipeline is,
                    // which is what `Renderer::device` alone did not answer. See
                    // [`TracedView::built_on`].
                    self.built_on = Some(device.clone());
                    self.shading.insert(s)
                }
                Err(e) => {
                    self.error = Some(format!("could not build the lighting pass: {e}"));
                    return;
                }
            },
        };
        if self.environment != Some(view.environment) {
            shading.set_environment(device, queue, maps);
            self.environment = Some(view.environment);
        }

        // A still goes up in bands, each its own submission, so that no one packet can run
        // the driver into its timeout on a slow card (loop spec S2) and so that input between
        // bands cancels the rest. The band is sized to the budget from the dearest row cost
        // this view has shown, floored by what the draft cost per pixel — a still can only
        // cost more — and it may grow only so much on the band before it, since the cost of
        // a row changes smoothly down the frame and a jump from the background into the model
        // is where an unbounded estimate would put a whole frame in one packet. The first band
        // of a view is the smallest band that fills the card, which is the cheapest honest
        // measurement there is. A drag is one submission: its size is what shrinks when it
        // misses its budget, since a draft that needs bands is a draft that is too big.
        let rows = if still {
            let from_draft =
                self.draft_ms_per_pixel.map(|ms| ms * width as f32 * DRAFT_TO_STILL);
            let from_still = self
                .still_ms_per_row
                .filter(|(w, h, _)| *w == width && *h == height)
                .map(|(_, _, ms)| ms);
            let per_row = match (from_still, from_draft) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            // Master extracted this budget into band_rows; the branch wrapped the result
            // in tile_aligned_rows. Both, because band_rows does not align and dropping the
            // alignment would silently change how a band is sized.
            tile_aligned_rows(
                band_rows(BAND_BUDGET_MS, per_row, self.last_band_rows, width, height),
                height,
            )
        } else if draft_bands(self.draft_rung, self.draft_ms_per_pixel, width, height) {
            // **A draft bands only once it is known to be too dear as a whole frame, and the
            // ladder has not got it under the budget.** Everything the ladder can shrink, it
            // still shrinks; this is for the documents that miss their budget at the ladder's
            // floor, where there is no smaller picture left to draw. On the owner's heavy case
            // a 240 px draft measured 55 ms there, three and a half interface frames, and
            // every menu click and keystroke waited behind one.
            //
            // The rung is passed because this comment and the condition disagreed once, and the
            // comment was the right one of the two (item 104).
            let per_row = self
                .draft_ms_per_row
                .filter(|(w, h, _)| *w == width && *h == height)
                .map(|(_, _, ms)| ms);
            tile_aligned_rows(
                band_rows(
                    INTERFACE_BUDGET_MS,
                    per_row,
                    self.last_draft_band_rows,
                    width,
                    height,
                ),
                height,
            )
        } else {
            height
        };
        let first = if still {
            self.still_cursor.min(height)
        } else {
            self.draft_cursor.min(height)
        };
        let last = (first + rows).min(height);
        let bands = height.div_ceil(rows);
        let band = (still || bands > 1).then(|| {
            (if still { self.still_band } else { self.draft_band } + 1, bands)
        });
        // **One march in flight, never a queue of them.** The march is submitted from inside
        // the interface's own frame, so every draft queued behind a running one is a frame the
        // picture lags the hand by. If the last is still running, this frame shows it again
        // and asks for another frame rather than adding to the queue — which bounds the depth
        // to one and is most of what the loop spec's P10 wants, at the cost of ten lines.
        //
        // Stills too, for a second reason: a band is sized from what the last one cost, and
        // the interface can run two or three frames ahead of the GPU, so without this the
        // next band was sized before the last had reported — from nothing, which is the whole
        // frame. On a card where a band is a few milliseconds this costs nothing, since the
        // cost has arrived by the next frame anyway.
        //
        // And no still band at all while there is input. The march goes into the one queue the
        // interface presents through, so a band in flight is a band every menu click waits
        // behind — 100 to 190 ms each on a patterned pipe, which the owner met as "the UI is
        // slow". A band is the unit that can be cancelled, and input is the reason to; holding
        // the still while the pointer moves or a key is down is P10's cheap half for the
        // still, as one march in flight was for the draft. The still resumes the frame the
        // input stops.
        //
        // **And from here, not on a cadence of its own**, which was the rest of P10's design and
        // was measured and declined (2026-10-13, `interface_latency`'s second table — corrected
        // the same day, since its first run waited for a band inside a blocking `poll`, which
        // holds the device's fence against this thread's `submit`, and read a CPU lock as the
        // queue). Corrected, a thread submitting the next band as the last ends keeps sixty
        // frames a second on every banded sweep and about doubles the picture's rate; on a whole
        // draft over two vertical blanks — this path at every rung but the last, since the ladder
        // shrinks before it bands — it costs frames, 44 → 37 at 517 px and 27 → 20 at 720 on 48
        // gyroid shells, with a vertical blank more of median. What it buys is the picture's
        // rate, not the interface's frames or its latency, which was the test. Submitted while
        // the application runs, a band also goes up before the paint waits for its swapchain
        // image, which measured the better of the two places in the frame.
        let interacting = ui.ctx().input(|i| {
            i.pointer.is_moving() || i.pointer.any_down() || !i.events.is_empty()
        });
        let skip = shading.busy() || (still && interacting);
        self.last_frame = Some((camera, width, height, self.step_budget.unwrap_or(STEP_BUDGET)));
        if !skip {
            self.last_gpu_work = Some(Instant::now());
        }
        if skip {
            if !still {
                self.draft_skipped += 1;
            }
        } else {
            match renderer.present_rows(
                shading,
                &camera,
                width,
                height,
                self.step_budget.unwrap_or(STEP_BUDGET),
                &look,
                accumulated,
                first..last,
            ) {
                Ok(ticket) => self.submitted.push(Submitted {
                    ticket,
                    frame: self.frame,
                    still,
                    width,
                    height,
                    rows: last - first,
                    band,
                    still_frame: still.then_some((accumulated + 1, still_frames)),
                    skipped: if still { 0 } else { std::mem::take(&mut self.draft_skipped) },
                    warm: false,
                }),
                Err(e) => {
                    self.error = Some(format!("field render failed: {e}"));
                    return;
                }
            }
        }
        if skip {
            ui.ctx().request_repaint();
        } else if still {
            self.still_band += 1;
            self.still_cursor = last;
            self.last_band_rows = Some(last - first);
            if last >= height {
                // A whole frame of this view is in the still's texture: one more sample
                // averaged, and it is fit to show.
                self.accumulated = accumulated + 1;
                self.still_cursor = 0;
                self.still_band = 0;
                self.still_shown = true;
            }
        } else {
            self.accumulated = 0;
            self.draft_generation = Some(shading.generation());
            self.draft_band += 1;
            self.draft_cursor = last;
            self.last_draft_band_rows = Some(last - first);
            if last >= height {
                self.draft_cursor = 0;
                self.draft_band = 0;
            }
            if bands > 1 {
                // The rest of the sweep is work this frame did not do, so the next frame is
                // asked for rather than waited for.
                ui.ctx().request_repaint();
            }
        }

        // egui draws the texture the lighting pass wrote — no readback and no upload. One
        // handle per set of targets: the drag frame's and the still's are two textures that
        // persist, so switching tiers is a change of handle and not of texture. A set that was
        // replaced — the window was resized — takes its handle with it.
        //
        // Until a still's first frame is whole, the draft's texture stays up: the still's
        // holds the previous view in every row its bands have not reached.
        let live: Vec<u64> = shading.live_generations().collect();
        let generation = match (still && !self.still_shown, self.draft_generation) {
            (true, Some(draft)) if live.contains(&draft) => draft,
            _ => shading.generation(),
        };
        let Some(texture) = shading.display_view() else { return };
        let id = {
            let mut egui_renderer = render_state.renderer.write();
            self.display.retain(|(g, id)| {
                let keep = live.contains(g);
                if !keep {
                    egui_renderer.free_texture(id);
                }
                keep
            });
            match self.display.iter().find(|(g, _)| *g == generation) {
                Some((_, id)) => *id,
                None => {
                    // Only the current targets can be registered from here, so a wanted draft
                    // texture that has no handle falls back to what the pass just wrote.
                    let current = shading.generation();
                    match self.display.iter().find(|(g, _)| *g == current) {
                        Some((_, id)) => *id,
                        None => {
                            let id = egui_renderer.register_native_texture(
                                device,
                                texture,
                                wgpu::FilterMode::Linear,
                            );
                            self.display.push((current, id));
                            id
                        }
                    }
                }
            }
        };
        paint(ui, rect, id);

        // Neither tier's next frame has an input to bring it about, so ask. Interactive: the
        // frame that switches to the still, when the input has been quiet long enough. Still:
        // the next sample, straight away, until there are enough of them.
        if !still {
            ui.ctx().request_repaint_after(SETTLE_AFTER.saturating_sub(since_change));
        } else if self.accumulated < still_frames {
            ui.ctx().request_repaint();
        }
    }
}

/// Draws the presented frame over the viewport.
fn paint(ui: &egui::Ui, rect: egui::Rect, id: egui::TextureId) {
    ui.painter().image(
        id,
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}
