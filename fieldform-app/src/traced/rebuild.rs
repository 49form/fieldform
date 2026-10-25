// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A rebuild: the graph compiled to a shader, its leaves baked or reused, its curves laid out.

use super::*;

/// A cage's own axis-aligned box, with the margin a bake wants around it.
///
/// The same measurement the app makes for the cage it is editing — shared rather than repeated,
/// because a bake box that disagreed with the one the camera is framed on would clip the body
/// at whichever edge the two differ by.
fn cage_box(cage: &fieldform_cage::QuadMesh) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    if !lo.x.is_finite() {
        return (Point3::origin(), Point3::origin());
    }
    // The limit surface lies inside the control hull, so the hull with a little room is a box
    // the surface certainly fits in.
    let margin = (hi - lo).norm() * 0.05;
    (
        Point3::new(lo.x - margin, lo.y - margin, lo.z - margin),
        Point3::new(hi.x + margin, hi.y + margin, hi.z + margin),
    )
}

/// How far a cage is subdivided before its field is sampled for a bake.
///
/// Its own subdivision depth, from its own size: a 10 mm cage and a metre one need different
/// numbers of levels to land within the same tolerance, and using one cage's depth for another
/// is how a small body comes out faceted beside a large one that looks fine.
pub(super) fn cage_levels(cage: &fieldform_cage::QuadMesh, tolerance_mm: f64) -> usize {
    let (lo, hi) = cage_box(cage);
    fieldform_field::levels_for_tolerance((hi - lo).norm(), tolerance_mm)
}

/// What one leaf's band is filed under: everything the band depends on, so a band is reused
/// exactly when none of it has changed.
pub(super) fn bake_key(
    document: &fieldform_scene::Document,
    leaf: BakedLeaf,
    tolerance_mm: f64,
    resolution: usize,
    bounds: (Point3<f64>, Point3<f64>),
    band: f64,
) -> BakeKey {
    let cage = match leaf {
        BakedLeaf::Cage(id) => document.cages.get(&id),
        BakedLeaf::Mesh(_) => None,
    };
    BakeKey {
        leaf,
        resolution,
        box_bits: [
            bounds.0.x.to_bits(),
            bounds.0.y.to_bits(),
            bounds.0.z.to_bits(),
            bounds.1.x.to_bits(),
            bounds.1.y.to_bits(),
            bounds.1.z.to_bits(),
        ],
        band_bits: band.to_bits(),
        geometry: cage.map_or(0, |cage| cage.geometry_fingerprint()),
        levels: cage.map_or(0, |cage| cage_levels(cage, tolerance_mm)),
    }
}

/// One narrow band, ready to share.
fn bake_of(
    field: &(dyn fieldform_field::ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    resolution: usize,
    band: f64,
) -> std::sync::Arc<fieldform_gpu::BakedField> {
    std::sync::Arc::new(fieldform_gpu::BakedField::new(field, bounds, resolution, band))
}

impl TracedView {
    /// Compiles the graph to a shader and bakes its cages. Called by the app when
    /// [`is_stale`](Self::is_stale) says so.
    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene<'_>) {
        // **Against the rewrite the shader is generated from** (backlog item 113). A joined
        // mirror of a section touching its plane is one extrude in both compilers, and its
        // section is a curve only the rewritten document holds — so the sections this rebuild
        // bakes must be read from the same pair, or the program would name a profile the
        // viewport cannot find. Once per rebuild, never per frame, and only when a mirror merges.
        if let Some((document, graph)) =
            fieldform_field::with_shared_faces_merged(scene.document, scene.graph)
        {
            let merged = Scene {
                document: &document,
                graph: &graph,
                curves: scene.curves.clone(),
                meshes: scene.meshes.clone(),
                ..*scene
            };
            return self.rebuild_scene(device, queue, &merged);
        }
        self.rebuild_scene(device, queue, scene);
    }

    fn rebuild_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene<'_>) {
        let began = Instant::now();
        let quality = self.pending();
        let resolution = match quality {
            Quality::Draft => DRAFT_BAKE_RESOLUTION,
            // Never zero, so a caller that never set one still gets a usable view rather than
            // an empty texture.
            Quality::Final => {
                if self.fineness.resolution == 0 {
                    BAKE_RESOLUTION
                } else {
                    self.fineness.resolution
                }
            }
        };
        self.stale = None;
        self.built = Some(quality);
        self.error = None;
        self.asked = self.asked.wrapping_add(1);
        let serial = self.asked;
        // Kept in hand rather than dropped: if the source comes out the same, its pipeline is
        // the one the new renderer wants (plan §5 V2, `Renderer::rebuilt_from`).
        let previous = self.renderer.take();
        // What the ladder learned, of the shader about to be replaced — handed back by
        // `hand_back_learned_costs` if the rebuild produces the same one (item 105). Taken before
        // the reset below, which still runs: a different shader must start over.
        let learned = self.program.as_ref().map(|p| LearnedCosts {
            source: source_identity(&p.source),
            draft_rung: self.draft_rung,
            draft_fast_streak: self.draft_fast_streak,
            draft_ms_per_pixel: self.draft_ms_per_pixel,
            draft_ms_per_row: self.draft_ms_per_row,
            last_draft_band_rows: self.last_draft_band_rows,
            still_ms_per_row: self.still_ms_per_row,
        });
        // A failed rebuild leaves the stash for the next one, which is still a comparison
        // against the shader the costs were measured on.
        if learned.is_some() {
            self.learned = learned;
        }
        // A new renderer draws a new image: whatever was averaged so far was of the old one,
        // and what it cost says nothing about what this one will — unless it is the same one.
        self.last_view = None;
        self.accumulated = 0;
        self.still_cursor = 0;
        self.still_band = 0;
        self.still_shown = false;
        self.still_ms_per_row = None;
        self.draft_ms_per_pixel = None;
        self.draft_cursor = 0;
        self.draft_band = 0;
        self.draft_ms_per_row = None;
        self.last_draft_band_rows = None;
        self.draft_sweep = None;
        // Back to the top rung: the new graph may be cheaper, and the ladder only climbs
        // slowly.
        self.draft_rung = 0;
        self.draft_fast_streak = 0;
        // Decided here, over the graph being compiled, because the sphere the camera is handed
        // each frame was fitted to the bounded bodies only (`pick_bounds`) -- a plane widens it
        // to nothing rather than to everything, so with one in the scene the sphere does not
        // enclose what is drawn and the ray must start at the eye.
        // **Worked out here and applied with the renderer, not before it.** While a shader is
        // being compiled elsewhere the viewport goes on drawing the shape it has, and these
        // three describe the *new* one: a march budget, a value scale and whether the scene is
        // bounded. Applied early, they would be the new graph's numbers marching the old
        // graph's picture — a wrong image for as long as the driver takes.
        let scene_bounded =
            !fieldform_field::part_extent(scene.document, scene.graph).is_unbounded();

        // Read here rather than per frame: it depends only on the graph, and the graph is what
        // a rebuild is for.
        let multiplier = fieldform_field::step_multiplier_with(
            scene.document,
            scene.graph,
            scene.morph_divisors,
        );
        let step_budget = Some(
            ((STEP_BUDGET as f64 * multiplier * STEP_HEADROOM).ceil() as u32)
                .clamp(STEP_BUDGET, MAX_STEP_BUDGET),
        );
        // No headroom here: the budget wants to be generous because spare steps are free, while
        // this is an estimate of the value's own shortfall and multiplying it by two would make
        // an undivided body read *more* open than it is. Its own walk rather than the budget's
        // number, because a sweep read in its tube falls short only inside its caps, which the
        // budget pays for and the shading never samples.
        let value_scale = Some(fieldform_field::value_scale_with(
            scene.document,
            scene.graph,
            scene.morph_divisors,
        ));
        // What the picture is marched with, kept together so it can be applied together.
        let marching = Marching { scene_bounded, step_budget, value_scale };

        // Generated, or the one generated last while nothing has invalidated the view since.
        let program = match self.program_for(scene) {
            Ok((p, _)) => p,
            Err(e) => {
                self.error = Some(e.to_string());
                self.nothing_to_draw();
                return;
            }
        };

        let generated = began.elapsed();

        // Everything the shader wants baked, in its binding order — reusing a band whenever
        // nothing it depends on has changed. A show/hide changes which *bodies* are in the
        // graph and nothing about the geometry of the ones that stay, so on an assembly this is
        // where the wait went: 890 ms of re-baking on thirty solids, for one click.
        //
        // The leaf's field is built only on a miss, which matters as much as the bake: a
        // `MeshField`'s BVH and crease normals were also being rebuilt per rebuild, separately
        // from the CPU side's own cache.
        let leaf_count = program.baked.len().max(1);
        let resolution = fieldform_gpu::bake_resolution_limit(device, leaf_count, resolution);

        let mut bakes: Vec<std::sync::Arc<fieldform_gpu::BakedField>> =
            Vec::with_capacity(program.baked.len());
        // Which leaves this rebuild dealt with, and exactly which bands it is standing on: a
        // band of a leaf asked for under a different key is an earlier shape of it, while a leaf
        // not asked for at all is a hidden body and keeps what it has.
        self.last_bakes.clear();
        let mut asked: std::collections::HashSet<BakedLeaf> = std::collections::HashSet::new();
        let mut keep: std::collections::HashSet<BakeKey> = std::collections::HashSet::new();
        // Every leaf this rebuild asked for, as its key with the resolution taken out — what a
        // band in flight is judged against. Two bakes of one shape differing only in how finely
        // they sample it describe the same thing, so a draft rebuild during a drag can still
        // say whether the settled band being made is of the shape now on screen.
        let mut identities: std::collections::HashSet<BakeKey> = std::collections::HashSet::new();
        // Settled bands to compute elsewhere: the key they will be filed under, the field to
        // sample, and where and how finely.
        type Queued = (
            BakeKey,
            std::sync::Arc<dyn fieldform_field::ScalarField + Send + Sync>,
            (Point3<f64>, Point3<f64>),
            usize,
            f64,
        );
        let mut later: Vec<Queued> = Vec::new();
        for leaf in &program.baked {
            let bounds = match leaf {
                // Its own box, resolved against the document by id. This used to be *the*
                // cage's box for every cage leaf, on the reasoning that a document held one —
                // which is what stopped a second one being importable at all. A cage arriving
                // from an OBJ is a body like any other, and baking it over another cage's box
                // would spend its grid on empty space and clip it where the boxes differ.
                BakedLeaf::Cage(id) => match scene.document.cages.get(id) {
                    Some(cage) => cage_box(cage),
                    None => scene.cage_bounds,
                },
                BakedLeaf::Mesh(id) => {
                    let Some((_, data)) = scene.meshes.iter().find(|(seen, _)| seen == id) else {
                        self.error = Some(format!(
                            "the graph references mesh {id}, which is not in the document"
                        ));
                        self.nothing_to_draw();
                        return;
                    };
                    // Its own extent with a margin, not the scene's: baking a mesh over the
                    // composed bounds would spend the grid on whatever else is in the document.
                    match data.bounds() {
                        Some((lo, hi)) => {
                            let lo = Point3::new(lo[0], lo[1], lo[2]);
                            let hi = Point3::new(hi[0], hi[1], hi[2]);
                            let margin = (hi - lo).norm() * 0.05;
                            (
                                Point3::new(lo.x - margin, lo.y - margin, lo.z - margin),
                                Point3::new(hi.x + margin, hi.y + margin, hi.z + margin),
                            )
                        }
                        None => scene.cage_bounds,
                    }
                }
            };
            // **Each leaf's own band, from its own box.** It used to be a fraction of the
            // *active cage's* box, which put a number that moves with one body into every other
            // body's key: dragging a cage face grew that box a little every frame, so every
            // other body in an assembly missed its bake and was rebuilt and re-baked per frame
            // of the drag. A band is how far from its own surface a body's grid is worth
            // keeping, which is a fact about that body.
            let band = (bounds.1 - bounds.0).norm() * 0.15;
            let key =
                bake_key(scene.document, *leaf, scene.tolerance_mm, resolution, bounds, band);
            asked.insert(*leaf);
            identities.insert(shape_of(&key));
            // **Already being made.** Stand in with the draft band and leave the worker alone:
            // rebuilding this leaf's field to queue a bake that is already running is the work
            // that made a settle of several bodies quadratic.
            let in_flight = self
                .baking
                .as_ref()
                .is_some_and(|baking| baking.pending.contains(&key));
            if in_flight {
                let coarse = BakeKey { resolution: DRAFT_BAKE_RESOLUTION, ..key };
                if let Some(standing_in) = self.bakes.get(&coarse) {
                    keep.insert(coarse);
                    keep.insert(key);
                    bakes.push(std::sync::Arc::clone(standing_in));
                    continue;
                }
            }
            if let Some(existing) = self.bakes.get(&key) {
                keep.insert(key);
                bakes.push(std::sync::Arc::clone(existing));
                continue;
            }

            // A miss, so the field itself has to be built before it can be sampled. Held in an
            // `Arc` because a settled band is sampled on another thread.
            let making = Instant::now();
            let mut detail = String::new();
            let field: std::sync::Arc<dyn fieldform_field::ScalarField + Send + Sync> = match leaf {
                BakedLeaf::Cage(id) => {
                    let Some(cage) = scene.document.cages.get(id) else {
                        self.error = Some(format!(
                            "the graph references cage {id}, which is not in the document"
                        ));
                        self.nothing_to_draw();
                        return;
                    };
                    // The level the band is filed under, so what is made is what the key says.
                    let levels = key.levels;
                    let extent = {
                        let (lo, hi) = cage_box(cage);
                        (hi - lo).norm()
                    };
                    detail = format!(
                        "cage {levels} levels from {:.3} mm over {:.0} mm, box {:.0}x{:.0}x{:.0}, band {band:.1}",
                        scene.tolerance_mm,
                        extent,
                        bounds.1.x - bounds.0.x,
                        bounds.1.y - bounds.0.y,
                        bounds.1.z - bounds.0.z,
                    );
                    // A red cage never gets here: codegen leaves its block out of the program,
                    // so no `BakedLeaf` names it. If one did, that is a bug and this says so.
                    let Some(cage) = cage.cage() else {
                        self.error = Some(format!("cage {id} is not a solid and cannot be baked"));
                        self.nothing_to_draw();
                        return;
                    };
                    match CageField::from_cage(cage, levels) {
                        Ok(f) => std::sync::Arc::new(f),
                        Err(e) => {
                            self.error = Some(e.to_string());
                            self.nothing_to_draw();
                            return;
                        }
                    }
                }
                BakedLeaf::Mesh(id) => {
                    let Some((_, data)) = scene.meshes.iter().find(|(seen, _)| seen == id) else {
                        unreachable!("found above");
                    };
                    let positions =
                        data.positions.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect();
                    match MeshField::from_mesh(positions, data.triangles.clone()) {
                        Ok(f) => std::sync::Arc::new(f),
                        Err(e) => {
                            self.error = Some(format!("mesh {}: {e}", data.name));
                            self.nothing_to_draw();
                            return;
                        }
                    }
                }
            };

            // **A settled band is not baked here.** A draft one is a few milliseconds and a
            // settled one is seconds, so the draft is made now — or reused, if a drag just made
            // it — and the settled one is queued for the worker below.
            let field_ms = making.elapsed().as_secs_f32() * 1e3;
            let baking_started = Instant::now();
            if quality == Quality::Final && resolution > DRAFT_BAKE_RESOLUTION {
                let coarse = BakeKey { resolution: DRAFT_BAKE_RESOLUTION, ..key };
                let standing_in = match self.bakes.get(&coarse) {
                    Some(band) => std::sync::Arc::clone(band),
                    None => {
                        let band = bake_of(&*field, bounds, DRAFT_BAKE_RESOLUTION, band);
                        self.bakes.insert(coarse, std::sync::Arc::clone(&band));
                        band
                    }
                };
                keep.insert(coarse);
                keep.insert(key);
                later.push((key, field, bounds, resolution, band));
                bakes.push(standing_in);
                self.last_bakes.push((
                    format!("at {DRAFT_BAKE_RESOLUTION} while {resolution} is made [{detail}]"),
                    field_ms,
                    baking_started.elapsed().as_secs_f32() * 1e3,
                ));
                continue;
            }

            let built = bake_of(&*field, bounds, resolution, band);
            self.bakes.insert(key, std::sync::Arc::clone(&built));
            keep.insert(key);
            bakes.push(built);
            self.last_bakes.push((
                format!("at {resolution} [{detail}]"),
                field_ms,
                baking_started.elapsed().as_secs_f32() * 1e3,
            ));
        }

        // **A band in flight is stopped only when nothing wants it any more**, which is what
        // this rebuild has just worked out: a pending key whose shape is not among the ones asked
        // for is a band of geometry that has since moved. Everything else — a section dragged, a
        // body placed, a parameter turned, a band that simply has not arrived — leaves the worker
        // alone, because stopping it means starting its seconds of work again from nothing.
        if let Some(baking) = &self.baking
            && bake_is_obsolete(&baking.pending, &identities)
        {
            self.stop_baking();
        }

        // Off to a worker, one band at a time so a scene of several bodies sharpens as each
        // arrives.
        if !later.is_empty() {
            let queued: Vec<BakeKey> = later.iter().map(|(key, ..)| *key).collect();
            let outstanding = queued.len();
            let (tx, arrivals) = std::sync::mpsc::channel();
            // One worker at a time: whatever is still in flight is for a shape this rebuild no
            // longer asked for, or it would have been stood in for above.
            self.stop_baking();
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let worker_stop = std::sync::Arc::clone(&stop);
            // **In a pool of its own, and a smaller one.** The bake fans out across rayon's
            // threads, so a settled bake on the shared pool takes every core and the interface's
            // own small bakes wait behind whole slabs of it. Two cores are kept back: one for the
            // frame, one for everything else the application does while this runs. Built once and
            // kept, because a slider drag asks for a settled rebuild per frame.
            let pool = match &self.pool {
                Some(pool) => Some(std::sync::Arc::clone(pool)),
                None => {
                    let threads = std::thread::available_parallelism()
                        .map(|n| n.get().saturating_sub(2).max(1))
                        .unwrap_or(1);
                    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                        Ok(pool) => {
                            let pool = std::sync::Arc::new(pool);
                            self.pool = Some(std::sync::Arc::clone(&pool));
                            Some(pool)
                        }
                        // No pool means baking on the worker thread alone: slow, and correct.
                        Err(_) => None,
                    }
                }
            };
            let spawned = std::thread::Builder::new()
                .name("settle bake".to_string())
                .spawn(move || {
                    let run = move || {
                        for (key, field, bounds, resolution, band) in later {
                            let stopped =
                                || worker_stop.load(std::sync::atomic::Ordering::Relaxed);
                            let Some(baked) = fieldform_gpu::BakedField::new_cancellable(
                                &*field, bounds, resolution, band, &stopped,
                            ) else {
                                return;
                            };
                            if tx.send((key, std::sync::Arc::new(baked))).is_err() {
                                return;
                            }
                        }
                    };
                    match pool {
                        Some(pool) => pool.install(run),
                        None => run(),
                    }
                });
            match spawned {
                Ok(_) => {
                    self.baking = Some(Baking {
                        stop,
                        pending: queued,
                        arrivals,
                        outstanding,
                    })
                }
                // A machine that cannot spawn a thread still gets a picture, coarse and honest,
                // rather than a viewport that never sharpens and never says why.
                Err(e) => self.error = Some(format!("could not start the settled bake: {e}")),
            }
        }

        // Bands for geometry no longer referenced, or baked at a resolution nothing asks for
        // any more, are dropped — otherwise every resolution ever tried stays resident. And a
        // band of a leaf this rebuild asked for under a *different* key is an earlier shape of
        // it: a drag rebakes every frame, and keeping each would hold one bake per frame of the
        // gesture. A leaf not asked for at all is a hidden body, and keeps its band.
        self.bakes.retain(|k, _| {
            // What this rebuild is drawing with, and what it is waiting for.
            keep.contains(k)
                // A leaf nobody asked for is hidden, and keeps the band it had at this
                // resolution rather than paying to bake it again when it comes back. Its band
                // width is its own, so there is nothing here to compare it against.
                || (!asked.contains(&k.leaf) && k.resolution == resolution)
        });

        // In the shader's binding order. A spine goes up as it is; a profile is flattened into
        // its own block's frame first — see `CurveUse` for why that happens here rather than in
        // the shader, and `Profile` for what the two flattenings mean.
        //
        // A curve the graph names but the document does not hold is a broken document rather
        // than an empty pipe, so it fails loudly either way.
        let mut spines = Vec::with_capacity(program.curves.len());
        // A wall is asked for three times running, once for each polyline it is read from, and
        // finding its bare edges grows as the square of its run count -- 3.9 ms at 256 runs
        // and 30 at a thousand (`wall_cost`), on this thread. So it is built once and kept
        // until the next node's.
        let mut wall: Option<(fieldform_scene::NodeId, Option<fieldform_field::Wall>)> = None;
        // A loft is asked for two bindings and one more per section, and building one samples
        // the difference of every neighbouring pair of sections on a grid -- 17 ms for four
        // (`loft_bound_cost`). So it too is built once and kept until the next node's.
        let mut loft: Option<(fieldform_scene::NodeId, Option<std::sync::Arc<fieldform_field::Lofted>>)> = None;
        for use_of in &program.curves {
            let points = match use_of {
                // A spine is walked, never baked: it is a path rather than a boundary, so
                // there is no inside for a grid to hold.
                fieldform_gpu::CurveUse::Spine(id) => scene
                    .curves
                    .iter()
                    .find(|(curve, _)| curve == id)
                    .map(|(_, spine)| (spine.clone(), None)),
                // A point an attractor is drawn toward, as the one-vertex spine it is. The
                // nearest-place search answers a span of one with that vertex, so nothing in
                // the shader has to know this is not a curve — see `CurveUse::Point`.
                fieldform_gpu::CurveUse::Point(node) => {
                    match scene.graph.node(*node) {
                        Some(fieldform_scene::Node::Point { at }) => {
                            let n = |r: &fieldform_scene::ParamRef| {
                                r.resolve(&|pid| scene.document.parameter_value(pid))
                                .unwrap_or(0.0)
                            };
                            Some((
                                vec![nalgebra::Point3::new(n(&at[0]), n(&at[1]), n(&at[2]))],
                                None,
                            ))
                        }
                        // Refused at compile as `AttractorIsNotAPoint`, so this is the belt to
                        // those braces rather than a case that draws anything.
                        _ => Some((Vec::new(), None)),
                    }
                }
                // A sweep's frame: the rotation-minimising normal at each sample of the spine
                // it walks, computed from the same polyline this loop uploads for that spine,
                // so the two bindings index the same samples -- see `CurveUse::Frame`.
                fieldform_gpu::CurveUse::Frame(node) => match scene.graph.node(*node) {
                    Some(fieldform_scene::Node::Sweep { outline, spine, .. }) => scene
                        .curves
                        .iter()
                        .find(|(curve, _)| curve == spine)
                        .and_then(|(_, points)| {
                            fieldform_field::sweep_frame(scene.document, outline, *spine, points)
                        })
                        .map(|normals| (normals, None)),
                    // A loft's frame rides the centerline its own build flattened, below, and
                    // not a curve of the document's -- a derived centerline is none.
                    Some(fieldform_scene::Node::Loft { .. }) => {
                        loft_for(&mut loft, scene.document, scene.graph, *node).map(|built| {
                            (built.frame_normals().iter().map(|n| Point3::from(*n)).collect(), None)
                        })
                    }
                    _ => None,
                },
                // A loft's centerline and its sections, all from the one build the field
                // compiled from, so the stations codegen wrote into `params` are arcs along the
                // very polyline uploaded here -- see `CurveUse::LoftSpine`.
                fieldform_gpu::CurveUse::LoftSpine(node) => {
                    loft_for(&mut loft, scene.document, scene.graph, *node)
                        .map(|built| (built.spine.points().to_vec(), None))
                }
                fieldform_gpu::CurveUse::LoftSection(node, k) => {
                    loft_for(&mut loft, scene.document, scene.graph, *node).map(|built| {
                        let points = built
                            .section_points(*k)
                            .iter()
                            .map(|q| Point3::new(q.x, q.y, 0.0))
                            .collect();
                        (points, None)
                    })
                }
                fieldform_gpu::CurveUse::LoftSlope(node, k) => {
                    loft_for(&mut loft, scene.document, scene.graph, *node).map(|built| {
                        let slopes = built
                            .section_slopes(*k)
                            .iter()
                            .map(|m| Point3::new(m.x, m.y, 0.0))
                            .collect();
                        (slopes, None)
                    })
                }
                // A wall, as the three polylines the shader reads it from, for whichever of
                // the three blocks it is on: an extrude's or a sweep's in its own plane, a
                // revolve's in the half-plane about its axis. Built
                // from the wall the field builds, at the fineness the viewport draws curves at,
                // and baked by the rule a profile is -- coarser while a gesture is in flight,
                // for the reason given at the profile's arm below. The grid rides on the chain,
                // which is the span the shader asks; see `CurveUse::Wall`.
                fieldform_gpu::CurveUse::Wall(node, part) => {
                    if wall.as_ref().map(|(built_for, _)| built_for) != Some(node) {
                        wall = Some((
                            *node,
                            fieldform_field::outline_wall_baked(
                                scene.document,
                                scene.graph,
                                *node,
                                scene.profile_degrees,
                                match quality {
                                    Quality::Draft => scene.tolerance_mm * 4.0,
                                    Quality::Final => scene.tolerance_mm / scene.profile_fineness,
                                },
                                scene.reads_mm,
                            ),
                        ));
                    }
                    wall.as_ref().and_then(|(_, built)| built.as_ref()).map(|built| {
                        let grid = match part {
                            fieldform_field::WallPart::Chain => built.bake().cloned(),
                            _ => None,
                        };
                        (built.packed(*part), grid)
                    })
                }
                // Where each of the section's corners goes per millimetre of draft, for a
                // block whose draft mitres them. From the kernel's own reading, so the shader
                // displaces the boundary the field displaces.
                fieldform_gpu::CurveUse::Miters(node) => fieldform_field::outline_miters(
                    scene.document,
                    scene.graph,
                    *node,
                    scene.profile_degrees,
                )
                .map(|points| (points, None)),
                fieldform_gpu::CurveUse::Profile(node) => profile_points(
                    scene.document,
                    scene.graph,
                    *node,
                    // **Coarser while a gesture is in flight**, which is the same trade the
                    // cage bake makes two lines above and for exactly the same reason. Baking a
                    // profile means searching its boundary once per grid point -- about a
                    // hundred milliseconds at viewport resolution -- and a rebuild happens on
                    // every change, so a drag would pay that per frame and be slower than the
                    // walk it replaced. Four times the tolerance is a sixteenth of the samples.
                    match quality {
                        Quality::Draft => scene.tolerance_mm * 4.0,
                        // And finer than that on Enchilada (`profile_fineness`): the margin a
                        // baked outline is drawn fat by halves, and a knife edge's chamfer with it.
                        Quality::Final => scene.tolerance_mm / scene.profile_fineness,
                    },
                    scene.profile_degrees,
                    scene.reads_mm,
                )
                .map(|p| {
                    let points =
                        p.points().iter().map(|q| Point3::new(q.x, q.y, 0.0)).collect();
                    (points, p.bake().cloned())
                }),
            };
            match points {
                Some(points) => spines.push(points),
                None => {
                    self.error = Some(format!(
                        "the graph references {use_of:?}, which the document cannot supply"
                    ));
                    self.program = None;
                    return;
                }
            }
        }

        // **Tessellated to the rule, not to a number chosen here.** `packed_surfaces` asks
        // `surface_levels` exactly as the compile arm does, so the cells the shader lays out sit
        // on the same limit surface the field's do. Built without the application's leaf cache
        // -- the shader is rebuilt on a structural change rather than per frame, and reaching
        // the cache from here would mean threading it through the scene struct for a cost that
        // has not shown up.
        let surfaces = match fieldform_field::packed_surfaces(
            scene.document,
            &program.surfaces,
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.error = Some(format!("a conformal lattice's surface is unusable: {e}"));
                self.nothing_to_draw();
                return;
            }
        };
        let prepared = began.elapsed();
        // Only a renderer of this same device: after a device is lost and made again, the old
        // one's pipeline belongs to a device that is gone.
        // **A shader the driver has already compiled is kept; a new one goes to a worker.**
        // The first is every drag of a number and costs nothing; the second is what froze the
        // application for half a minute.
        let previous = previous.filter(|p| p.device() == device);
        let same_shader = previous.as_ref().is_some_and(|p| p.source() == program.source);
        if !same_shader {
            // See `compile_already_wanted`: adopt the worker rather than start a second one
            // for the same source. `Compiling::source` was put there for this and never read.
            if let Some(waiting) = self
                .compiling
                .as_mut()
                .filter(|c| compile_already_wanted(&c.source, &program.source))
            {
                waiting.serial = serial;
                waiting.marching = marching;
                // And this rebuild's program and uploads, which the worker does not have: its
                // slots hold this rebuild's numbers and its curves this rebuild's polylines.
                waiting.program = Some(program);
                waiting.uploads = Some((bakes, spines, surfaces));
                // Keep drawing the old shape, exactly as the spawn below would.
                self.renderer = previous;
                return;
            }
            let (tx, done) = std::sync::mpsc::channel();
            // Cloned for the worker, so the originals are still here for the fallback below:
            // a `Vec<Arc<_>>` and a handful of handles, none of it the geometry itself.
            let (worker_device, worker_queue) = (device.clone(), queue.clone());
            let (worker_bakes, worker_spines, worker_surfaces) =
                (bakes.clone(), spines.clone(), surfaces.clone());
            let source = program.source.clone();
            let for_worker = program.clone();
            let spawned = std::thread::Builder::new()
                .name("shader compile".to_string())
                .spawn(move || {
                    let built = Renderer::with_device(
                        worker_device,
                        worker_queue,
                        &for_worker,
                        &worker_bakes,
                        &worker_spines,
                        &worker_surfaces,
                    );
                    let _ = tx.send(built);
                });
            match spawned {
                Ok(_) => {
                    // The old renderer keeps drawing: put it back, and with it the program it
                    // belongs to, which `rebuild` has not replaced yet.
                    self.renderer = previous;
                    self.compiling = Some(Compiling {
                        serial,
                        marching,
                        done,
                        program: Some(program),
                        source,
                        started: Instant::now(),
                        announced: false,
                        uploads: None,
                    });
                    return;
                }
                // A machine that cannot spawn a thread compiles here, as it always did.
                Err(_) => {
                    let built = Renderer::with_device(
                        device.clone(),
                        queue.clone(),
                        &program,
                        &bakes,
                        &spines,
                        &surfaces,
                    );
                    self.install(built, program, marching, began.elapsed());
                    return;
                }
            }
        }
        // **Let the old one go on a clock of its own.** It used to be dropped here by falling
        // out of the arm, which put the driver's free inside `pipeline_ms` where nothing named
        // it. Same order, same moment — only now it is timed.
        let (built, released) = match previous {
            Some(previous) => {
                let built = Renderer::rebuilt_from(&previous, &program, &bakes, &spines, &surfaces);
                let releasing = Instant::now();
                drop(previous);
                (built, releasing.elapsed())
            }
            None => (
                Renderer::with_device(
                    device.clone(),
                    queue.clone(),
                    &program,
                    &bakes,
                    &spines,
                    &surfaces,
                ),
                Duration::ZERO,
            ),
        };
        match built {
            Ok(r) => {
                self.scene_bounded = marching.scene_bounded;
                self.step_budget = marching.step_budget;
                self.value_scale = marching.value_scale;
                // The device this rebuild built on, which is the device every resource in hand
                // is now of. See [`TracedView::built_on`].
                self.built_on = Some(device.clone());
                self.renderer = Some(r);
                let size = program.size;
                self.program = Some(program);
                self.hand_back_learned_costs();
                let ms = |d: Duration| d.as_secs_f32() * 1e3;
                let cost = RebuildCost {
                    generate_ms: ms(generated),
                    prepare_ms: ms(prepared.saturating_sub(generated)),
                    pipeline_ms: ms(began
                        .elapsed()
                        .saturating_sub(prepared)
                        .saturating_sub(released)),
                    release_ms: ms(released),
                    size,
                };
                self.last_rebuild = Some((quality, cost));
                self.last_build = self.renderer.as_ref().map(|r| r.build_cost());
                self.rebuilds.push(cost);
                // Nobody is reading them unless the perf log is on.
                if self.rebuilds.len() > 64 {
                    self.rebuilds.remove(0);
                }
            }
            Err(e) => {
                self.error = Some(format!("could not build the field renderer: {e}"));
                self.program = None;
            }
        }
    }
}

/// The loft `node` is, built once and kept in `cache` until another node's is asked for -- the
/// arrangement the wall cache in `rebuild` has, for the same reason: one block, several
/// bindings, one build.
fn loft_for<'a>(
    cache: &'a mut Option<(fieldform_scene::NodeId, Option<std::sync::Arc<fieldform_field::Lofted>>)>,
    document: &fieldform_scene::Document,
    graph: &fieldform_scene::Graph,
    node: fieldform_scene::NodeId,
) -> Option<&'a fieldform_field::Lofted> {
    if cache.as_ref().map(|(built_for, _)| *built_for) != Some(node) {
        let built = match graph.node(node) {
            Some(fieldform_scene::Node::Loft { sections, spine, smooth, rims: _ }) => {
                fieldform_field::loft_built(document, sections, *spine, *smooth)
            }
            _ => None,
        };
        *cache = Some((node, built));
    }
    cache.as_ref().and_then(|(_, built)| built.as_deref())
}

/// A profile flattened into its own block's frame, as the shader wants it.
///
/// **The same two readings the kernel makes**, and they have to stay the same: an extrude takes
/// `x` and `y` across the plane, a revolve takes `(r, z)` about the axis, and a difference
/// between the two sides would draw one shape and export another. Built from
/// `fieldform_field::Profile`, which is where both readings live, rather than repeating the
/// arithmetic here.
///
/// `z` is left at zero. The shader reads only `.xy`, and a third component that meant nothing
/// would be a thing to wonder about later.
///
/// `None` when the block is not one of these, the curve is missing, or the profile is not one —
/// all of which the kernel refuses with a message, so the caller only has to say that nothing
/// could be supplied.
fn profile_points(
    document: &fieldform_scene::Document,
    graph: &fieldform_scene::Graph,
    node: fieldform_scene::NodeId,
    tolerance_mm: f64,
    turn_degrees: f64,
    reads_mm: f64,
) -> Option<fieldform_field::Profile> {
    let numbers = |r: &fieldform_scene::ParamRef| {
        r.resolve(&|id| document.parameter_value(id))
    };
    // An extrude has no frame any more: its plane is its profile's own, so the polyline is
    // read in the plane fitted to it. A revolve still has one, because its frame is an axis
    // rather than a plane and cannot be deduced from the profile.
    let (curve, ignore, revolved) = match graph.node(node)? {
        // A profile's curve. A block built on a wall is not asked for one of these: its three
        // polylines come from the wall itself, through `CurveUse::Wall`.
        fieldform_scene::Node::Extrude { outline, sweep, .. } => {
            (outline.curve(), sweep.source(), false)
        }
        // Read in its own fitted plane exactly as an extrude's is, with the spine left out of
        // the loop walk -- the same reading the compile makes.
        fieldform_scene::Node::Sweep { outline, spine, .. } => {
            (outline.curve(), Some(*spine), false)
        }
        fieldform_scene::Node::Revolve { outline, frame, .. } => {
            (outline.curve(), frame.curve_source(), true)
        }
        _ => return None,
    };
    // The same walk the compile makes, through the same function, so the shader and the field
    // cannot disagree about which curves the profile is made of.
    let world = fieldform_field::profile_loop_turning(document, curve, turn_degrees, ignore)
        .ok()?
        .points;
    let (translation, rotation) = match graph.node(node)? {
        fieldform_scene::Node::Revolve { frame, .. } => {
            fieldform_field::resolve_frame(document, graph, frame, &numbers).ok()?
        }
        // About the point the path carries the section by, which is the spine's start dropped
        // onto the plane and not the loop's centroid -- the compile's own reading, since a
        // profile flattened about one point and ridden by another is the solid in two places.
        fieldform_scene::Node::Sweep { outline, spine, .. } => {
            let laid =
                fieldform_field::sweep_outline_section(document, outline, *spine, turn_degrees)
                    .ok()?;
            (laid.origin.coords, fieldform_field::rotation_from_z(laid.normal))
        }
        _ => {
            let (origin, normal) = fieldform_field::plane_of(&world)?;
            (origin.coords, fieldform_field::rotation_from_z(normal))
        }
    };
    let into = move |p: Point3<f64>| {
        Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
    };
    let profile = if revolved {
        fieldform_field::Profile::about_axis(&world, into).ok()?
    } else {
        fieldform_field::Profile::in_plane(&world, into, tolerance_mm.max(1e-4)).ok()?
    };
    // Baked by the same rule the field compiles with, so the two sides describe one surface.
    // `gpu_matches_cpu` is what says they do; baking on one side only would move it by the
    // margin, and the number would be reporting a difference in method rather than a mistake.
    let (lo, hi) = profile.bounds();
    let span = (hi - lo).max();
    // How far off the boundary this node reads, by the same rule the field uses -- a drafted
    // wall is a contour some way out from the profile, and a band that does not reach it is a
    // cliff in the field rather than a slightly worse number -- and then as far again as the
    // display reads off every surface, which the field's rule has no way to know.
    let reads = fieldform_field::reads_widened(
        fieldform_field::profile_reads_out_to(document, graph, node),
        reads_mm,
    );
    // A tolerance too fine to bake falls back to the rule's own before it falls back to the walk,
    // so a finer rung never costs a profile its bake: on Enchilada the ask is twice as fine, and a
    // profile that spans the scene with a wide band can run past the cells a bake may hold.
    let tolerance_mm = tolerance_mm.max(1e-4);
    let cells = fieldform_field::bake_cells(span, tolerance_mm, reads)
        .or_else(|| fieldform_field::bake_cells(span, tolerance_mm * 2.0, reads));
    let profile = match cells {
        Some(cells) => profile.baked(cells, reads),
        None => profile,
    };
    Some(profile)
}
