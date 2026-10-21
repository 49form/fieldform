// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Evaluates a part's composition graph as a single field.
//!
//! This is where the two layers meet. Cages are edited destructively and their limit surfaces
//! become fields; curves are parametric and become pipe fields; the graph says how to combine
//! them. Nothing here knows what produced a field — that is the point of [`ScalarField`], and
//! it is why adding the pipe generator needed no change to the boolean code.
//!
//! # Booleans on distance fields, honestly
//!
//! `min` and `max` compose distance fields, but the result is **not** a true distance field
//! everywhere:
//!
//! - `min(a, b)` (union) is exact outside the union. Inside, near a concave crease where the
//!   two surfaces meet, it *underestimates* — the true distance to the boundary can be larger
//!   than either operand suggests.
//! - `max(a, b)` (intersection) and `max(a, -b)` (difference) are exact inside, and
//!   underestimate outside near a convex corner.
//!
//! Underestimating is the safe direction: it is what sphere tracing and interval pruning need
//! in order to stay conservative, so nothing marches through a surface. But it means the value
//! is a *lower bound* on distance rather than the distance, in those regions. That is inherent
//! to CSG on fields, not a shortcut taken here, and it is why the surface itself — the zero
//! level set, which is what gets meshed and displayed — is exact regardless.

use crate::ScalarField;
use fieldform_scene::{Document, Graph, NodeId};
use nalgebra::Point3;

mod blend;
pub use blend::Rim;
pub(crate) use blend::{blend_bound, blend_radius_rate};
mod compile;
mod extent;
mod lipschitz;
mod loft_sweep;
mod pattern;
mod resolve;
mod sections;

pub use blend::taper_ratio;
pub use compile::{CompileError, GraphField, Region};
pub use extent::{graph_extent, operand_extent, part_extent};
pub use lipschitz::{
    approximate_blends, carries_true_distance, driven_blend_divisor, driven_lattice_divisor,
    driven_morph_divisor, driven_offset_divisor, driven_pipe_divisor, fillet_reach,
    graph_lipschitz, may_divide,
    operand_carries_true_distance, operand_lipschitz, step_multiplier, step_multiplier_with,
    value_scale_with,
};
pub use loft_sweep::{
    LoftParts, loft_built, loft_parts, loft_stretch, sweep_axes, sweep_frame, sweep_law,
    sweep_lean, sweep_offset, sweep_reads_true, sweep_stretch, sweep_tube, wall_bakes_under,
};
pub use pattern::{
    pattern_grid_anchor_in, pattern_origin_in, pattern_reach, pattern_reach_axes_in,
    pattern_reach_in,
};
pub use resolve::{
    AXIS_FLOOR, ends_at, mirror_one_side, mirror_reaches_its_plane, placement_above,
    reach_across_axis, resolve_frame,
    resolve_pivot, resolve_place, resolve_place_for, rotation_from_z, sweep_direction,
    turn_rotation, with_turns_flattened,
};
pub use sections::{
    ExtrudeSection, block_wall, extrude_section, outline_miters, outline_wall, outline_wall_baked,
    revolve_wedge, sweep_lean_degrees, sweep_outline_section,
};
pub(crate) use resolve::resolve_point;
use blend::{blend, blend_divisor, pattern_join_divisor, smooth_min_interval, tapered};
use compile::{Amount, Op};
use extent::place_at;
use lipschitz::{
    driven_refusal, lipschitz_of, normalization_of, own_divisor, ramp_lipschitz, twist_stretch,
};
use loft_sweep::{build_loft, sweep_section};
use pattern::pattern_fillet;
use resolve::{
    euler_degrees_to_rotation, placement_similarity, resolve_ends, resolve_placement, to_local,
};
use sections::{spine_start, wall_about_axis, wall_in_plane};

/// Bounds on how far a cage is subdivided before its limit surface is sampled for distance
/// queries. See [`levels_for_tolerance`].
///
/// The floor is set by **shading**, never by position accuracy: `levels_for_tolerance` derives
/// the level a tolerance needs, so all a floor decides is what a cage costs, and how it shades,
/// when nobody asked for accuracy. It was chosen while shading came from the field's gradient,
/// analytic normals then looked as though they had taken its reason away, and measured beyond
/// the cube they had not. Both halves of that are worth keeping.
///
/// A composed surface used to be shaded from the field's gradient. When that gradient came from
/// finite-differencing a distance-to-triangles field it was piecewise constant across each
/// triangle, so the visible defect was angular error that position tolerance does not
/// describe — and the floor had to be high enough to shrink the triangles until it stopped
/// showing. `ScalarField::normal` removed the cause: the normal is now interpolated from the
/// limit surface's own analytic normals and never derived by probing the field.
///
/// Measured on a 1000 mm cube. Position from `cage_field_accuracy`; the two normal columns
/// from `normal_accuracy` (the old finite-difference path) and `shading_normal_accuracy`
/// (the analytic one), so the change is visible rather than asserted:
///
/// | level | triangles | position | normal, differenced | normal, analytic | mesh @64 |
/// |---|---|---|---|---|---|
/// | 2 | 192 | 14.40 mm | 8.99 deg | 1.27 deg | 20 ms |
/// | 3 | 768 | 3.48 mm | 4.90 deg | 0.25 deg | 40 ms |
/// | 4 | 3072 | 0.92 mm | 1.48 deg | 0.08 deg | 88 ms |
/// | 5 | 12288 | 0.22 mm | 0.18 deg | 0.01 deg | 194 ms |
/// | 6 | 49152 | 0.06 mm | ~0.02 deg | — | 487 ms |
///
/// On the cube, level 2 now shades better than level 4 used to, and the floor looked free to
/// fall. **It is not, because the cube is the one shape that flatters it.** Its faces turn
/// through the gentlest angle a six-face cage can; stretch it and the turn tightens over the
/// same number of triangles. The worst analytic normal against the exact limit surface, at
/// forty-one samples to a face edge (the table's eleven under-read the cube at level 3):
///
/// | box, mm | level 3 | level 4 |
/// |---|---|---|
/// | 1000 cube | 0.41 deg | 0.12 deg |
/// | 1000 x 500 x 500 | 1.21 deg | 0.29 deg |
/// | 1000 x 1000 x 333 | 2.58 deg | 0.67 deg |
/// | 1000 x 1000 x 200 | 8.08 deg | 1.87 deg |
/// | 1000 x 100 x 100 | 27.7 deg | 9.04 deg |
///
/// One degree is where a shaded error stops being visible, so **4 is the lowest level that keeps
/// a box of ordinary proportions, up to three to one, inside it**; 3 does so only to about four
/// to three. And the viewport's default tolerance asks for no more than 3 on every box above,
/// so the floor is what decides how each of them shades. Lowering it to 3 was built and
/// measured on 2026-10-12 (loop spec P6): it halved the viewport's cage bakes — on the starting
/// cube `bake_cost` read 12.1 → 6.4 ms for the draft at 32 cells and 453 → 245 ms for
/// Balanced's settle at 128 — and paid for it with shading about four times worse on any cage
/// that is not a cube. Declined.
///
/// Past three to one no constant floor holds a degree: a slab a fifth as high as it is wide
/// wants 5, a bar a tenth as thick as it is long 6, and a plate a twentieth as thick as it is
/// wide is 30 deg out at level 4 and still 2 deg at the ceiling. A floor that answered to
/// shading on every shape would be read off the cage's own proportions rather than fixed here,
/// which is a design question the loop spec's P6 records, not a constant to tune.
/// `the_cage_floor_is_the_lowest_level_that_shades_a_slab_a_third_as_high_as_it_is_wide_within_a_degree`
/// holds this constant to the reason it has.
pub(crate) const MIN_CAGE_LEVELS: usize = 4;
pub(crate) const MAX_CAGE_LEVELS: usize = 6;

/// How much of a cage's size the level-0 tessellation error is, before subdivision.
///
/// Catmull-Clark error falls by roughly 4x per level, so error(level) ~= K * size * 4^-level.
/// K is measured, not assumed: on a 1000 mm cube a level-2 field sits 14.4 mm from the true
/// limit surface at worst, which puts K at about 0.23. `cage_field_accuracy` is the example
/// that produced that, and re-running it is how to check this still holds.
const CAGE_ERROR_COEFFICIENT: f64 = 0.23;

/// Subdivision level whose tessellation error is under `tolerance_mm` for a cage of
/// `size_mm`.
///
/// Fixing this at a constant was wrong in both directions: level 2 left a 1000 mm cube 14 mm
/// from its true surface, which reads as visible faceting — a cube that plainly is not the
/// rounded form its limit surface should be — while a higher constant would make every small
/// model pay for accuracy no display could show. Deriving it from a tolerance means the
/// caller's sampling resolution decides, and raising that resolution actually improves the
/// surface instead of hitting a fixed floor.
pub fn levels_for_tolerance(size_mm: f64, tolerance_mm: f64) -> usize {
    if !(size_mm > 0.0) || !(tolerance_mm > 0.0) {
        return MIN_CAGE_LEVELS;
    }
    let ratio = CAGE_ERROR_COEFFICIENT * size_mm / tolerance_mm;
    if ratio <= 1.0 {
        return MIN_CAGE_LEVELS;
    }
    let level = ratio.log(4.0).ceil() as usize;
    level.clamp(MIN_CAGE_LEVELS, MAX_CAGE_LEVELS)
}

/// How finely to tessellate a surface a conformal lattice follows.
///
/// **Not `levels_for_tolerance`, and asking that was a real mistake with a real symptom.** That
/// function answers how closely a tessellation has to track a limit surface for its *distance
/// field* to be accurate to a modelling tolerance — the right question for a cage, where the
/// triangles are what the distance is measured against. A conformal map measures nothing against
/// them: it wants `(s, t)`, and what it needs is for that pair to vary smoothly across a cell.
///
/// The difference is not academic. Every sample of a conformal lattice walks a hierarchy over
/// every triangle, so this number multiplies the cost of the whole field, and a 6x6 panel at
/// modelling tolerance came out at 73 728 triangles and wedged the GPU driver — twice, needing
/// a reboot each time. Measured by `fieldform-gpu/examples/conformal_cost.rs`:
///
/// ```text
/// levels   triangles (6x6 panel)
///      2       1 152
///      3       4 608
///      4      18 432
///      5      73 728
///      6     294 912
/// ```
///
/// So the depth follows the **cells**, which is what the user asked for and the only thing the
/// tessellation has to resolve: roughly four grid steps across each cell, which is where
/// interpolating `(s, t)` over a triangle stops being visible. A finer surface buys a smoother
/// closest point and nothing else, and pays for it on every sample of every ray.
///
/// Its own floor and ceiling rather than the cage's: `MIN_CAGE_LEVELS` is 4, which is 18 432
/// triangles before anything has been asked for at all.
pub fn surface_levels(cells_per_face: u32) -> usize {
    const MIN_SURFACE_LEVELS: usize = 3;
    /// Five is 73 728 triangles on a modest panel. Past that the cost is not worth any
    /// smoothness it buys, and a lattice that fine wants a finer *mesh*, not a finer grid over
    /// a coarse one.
    const MAX_SURFACE_LEVELS: usize = 5;
    // Four grid steps per cell: `n >= 4N`, so `levels >= log2(N) + 2`.
    let steps = (cells_per_face.max(1) as f64 * 4.0).log2().ceil() as usize;
    steps.clamp(MIN_SURFACE_LEVELS, MAX_SURFACE_LEVELS)
}

/// The surfaces a shader needs, packed, in the order its program asks for them.
///
/// **The same tessellation the field compiled, because the depth comes from one rule.** A
/// conformal lattice's cells are laid out on the tessellated limit surface, so a GPU that
/// tessellated to a different depth would put its walls in slightly different places than the
/// field does -- and the difference would appear as a shading error, a long way from a
/// subdivision count. `surface_levels` is asked here exactly as the compile arm asks it; a
/// window computed twice is two chances to compute it differently, which is the bug this crate
/// has already paid for once in `pattern_reach`. That is also why the count travels with the id:
/// the depth depends on it, so a caller holding only ids could not arrive at the same answer.
///
/// Pass the `cache` a compile used and the tessellation is shared rather than repeated.
pub fn packed_surfaces(
    document: &Document,
    ids: &[(fieldform_scene::QuadMeshId, u32)],
    mut cache: Option<&mut LeafCache>,
) -> Result<Vec<crate::PackedSurface>, CompileError> {
    let mut packed = Vec::with_capacity(ids.len());
    for (id, cells_per_face) in ids {
        let mesh = document
            .quad_meshes
            .get(id)
            .ok_or(CompileError::MissingQuadMesh(*id))?;
        let levels = surface_levels(*cells_per_face);
        let key = (*id, levels, mesh.geometry_fingerprint());
        let existing = cache
            .as_ref()
            .and_then(|c| c.surfaces.get(&key))
            .map(std::sync::Arc::clone);
        let map = match existing {
            Some(map) => map,
            None => {
                let map = std::sync::Arc::new(
                    crate::SurfaceMap::new(mesh, levels)
                        .map_err(|e| CompileError::BadQuadMesh(*id, e))?,
                );
                if let Some(c) = cache.as_mut() {
                    c.keep_surface(key, std::sync::Arc::clone(&map));
                }
                map
            }
        };
        packed.push(map.packed());
    }
    Ok(packed)
}

/// The bounding-box diagonal of a quad mesh, in millimetres.
///
/// Takes the mesh rather than the cage, since a cage *is* one and a surface to follow is one
/// that is deliberately not closed. Same question, same answer.
fn mesh_size(cage: &fieldform_cage::QuadMesh) -> f64 {
    let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    if lo.x.is_finite() {
        (hi - lo).norm()
    } else {
        0.0
    }
}

/// Compiled leaves kept between compiles.
///
/// A cage's subdivision and a mesh's BVH are pure functions of geometry that does not change:
/// hiding a body, adding a boolean or dragging a radius alters the *graph*, not the mesh a
/// `Node::Mesh` names. Rebuilding them anyway is what made every show/hide and every menu
/// action pause — measured at **90 ms for 103 680 triangles**, and the interface routes both
/// through the same rebuild.
///
/// Keyed by what the leaf is built from, so a stale entry is impossible rather than merely
/// unlikely: a mesh id names one immutable mesh, and a cage or surface entry carries the depth it
/// was built at and its geometry's fingerprint. **The fingerprint was missing until 2026-09-19**:
/// a cage is edited in place under its own id, and its depth only moves with its size, so a
/// corner pulled *inwards* left every face of the box where it was and the cache went on
/// answering for the shape before the edit — `a_cage_moved_inside_its_own_box_is_not_answered_from_the_cache`.
#[derive(Default)]
pub struct LeafCache {
    meshes: std::collections::HashMap<
        fieldform_scene::MeshId,
        std::sync::Arc<dyn ScalarField + Send + Sync>,
    >,
    cages: std::collections::HashMap<
        (fieldform_scene::CageId, usize, u64),
        std::sync::Arc<dyn ScalarField + Send + Sync>,
    >,
    /// A conformal lattice's surface, which is a tessellation and an index over it — the same
    /// cost a cage's subdivision is, and rebuilt by the same things that used to rebuild that.
    surfaces: std::collections::HashMap<
        (fieldform_scene::QuadMeshId, usize, u64),
        std::sync::Arc<crate::SurfaceMap>,
    >,
    /// A profile's baked grid, by everything the grid is a function of — and a wall's, which
    /// is baked by the same rule and kept on the same shelf under [`crate::Wall::fingerprint`].
    /// That hash begins with a word of its own, so a wall and the profile it runs along are
    /// never one key: over the same points their grids hold different fields.
    ///
    /// **Because the bake is the compile.** Measured on a swept aerofoil by
    /// `examples/sweep_compile_cost.rs`: 6.193 ms of a 6.296 ms `compile_graph`, against 0.006 ms
    /// for flattening the spine and 0.100 for walking the loop. And it is the half that does not
    /// change while a sweep is being edited — dragging the *path* does not touch the *profile* —
    /// so a drag rebuilt a grid it could have kept, every frame.
    ///
    /// Keyed by `(cells, band, fingerprint)` and by nothing else, which is the whole of what the
    /// bytes depend on:
    ///
    /// - the **points**, through [`crate::Profile::fingerprint`] — of the profile handed to the
    ///   bake, not of the loop behind it, for the reason that method gives;
    /// - the **cell count**, not the tolerance. Tolerance reaches the bytes only through
    ///   `bake_cells`, so two tolerances landing on one count bake identically — and the shader's
    ///   parallel bake passes `tolerance_mm.max(1e-4)` where this passes it raw, which would have
    ///   been a collision on a fine tolerance had the raw number been the key;
    /// - the **band**, through `band_range`, not the raw `reads`. `band_range` rounds outward in
    ///   sixteenths of the span precisely so that a drag does not rebuild a grid per frame, and
    ///   keying on what it was handed instead of on what it answers would put that churn straight
    ///   back at the cache: a fillet radius being dragged would miss every frame while the bytes
    ///   were byte-identical.
    profiles: std::collections::HashMap<(usize, u64, u64, u64), crate::BakedProfile>,
    /// Which of those the last compile actually wanted, so the shelf stays the size of a part.
    ///
    /// A profile's key has no id in it — it cannot, since the bake is a function of the points —
    /// so `retain_live` has nothing to match against a document. What it keeps instead is what
    /// was used, which drops a grid one compile after the last thing that wanted it stopped.
    profiles_used: std::collections::HashSet<(usize, u64, u64, u64)>,
}

impl LeafCache {
    /// Drops entries for geometry the document no longer holds.
    ///
    /// Called on the way in to a compile rather than on delete, because a delete is undoable
    /// and the cache would then have to be repopulated by the undo — where this simply keeps
    /// what is still referenced and costs a pass over a handful of keys.
    fn retain_live(&mut self, document: &Document) {
        self.meshes.retain(|id, _| document.meshes.contains_key(id));
        self.cages
            .retain(|(id, _, _), _| document.cages.contains_key(id));
        self.surfaces
            .retain(|(id, _, _), _| document.quad_meshes.contains_key(id));
        // A profile is kept while something is still asking for it. Taken rather than borrowed
        // so the retain can read it.
        let used = std::mem::take(&mut self.profiles_used);
        self.profiles.retain(|key, _| used.contains(key));
    }

    /// Stores a cage's leaf as the only one for that cage.
    ///
    /// **One per cage, because an edit is a new key.** Every drag ends in a compile, and keeping
    /// the leaves of every shape the cage has passed through would hold a subdivision per
    /// gesture for as long as the document is open. What an undo brings back is rebuilt, once.
    fn keep_cage(
        &mut self,
        key: (fieldform_scene::CageId, usize, u64),
        leaf: std::sync::Arc<dyn ScalarField + Send + Sync>,
    ) {
        self.cages.retain(|(id, _, _), _| *id != key.0);
        self.cages.insert(key, leaf);
    }

    /// The same for a surface to follow.
    fn keep_surface(
        &mut self,
        key: (fieldform_scene::QuadMeshId, usize, u64),
        map: std::sync::Arc<crate::SurfaceMap>,
    ) {
        self.surfaces.retain(|(id, _, _), _| *id != key.0);
        self.surfaces.insert(key, map);
    }
}

/// A profile answering from a grid, when the tolerance is loose enough to allow one — or a
/// wall, by the same rule.
///
/// One place rather than three, so an extrude, a revolve and a leaning sweep cannot end up baked
/// to different rules — and so the shader has one rule to match rather than three. **And one
/// rather than two for what is baked**: a wall is sized, keyed and kept exactly as a profile is,
/// which is what `Bakes` is for.
fn bake_to_tolerance<S: crate::baked_profile::Bakes>(
    profile: S,
    tolerance_mm: f64,
    reads: (f64, f64),
    cache: Option<&mut LeafCache>,
) -> S {
    let (lo, hi) = profile.bounds();
    let span = (hi - lo).max();
    // A profile too small or too coarse to be worth a grid keeps its polyline, and is not
    // stored: there is nothing to store and a key over no bake would answer for one.
    let Some(cells) = crate::bake_cells(span, tolerance_mm, reads) else {
        return profile;
    };
    let Some(cache) = cache else {
        return profile.baked(cells, reads);
    };
    // **This is the one place all three arms meet.** A sweep, an extrude and a revolve reach the
    // bake by different routes -- two through `Profile::in_plane` and one through
    // `Profile::about_axis` -- so a key taken at any of the call sites would be three rules that
    // could drift. Taken here it is one, and it is downstream of the fitted plane, the resolved
    // axis frame and the subdivision alike. See `LeafCache::profiles`.
    let band = crate::band_range(span.max(1e-6), reads);
    let key = (cells, band.0.to_bits(), band.1.to_bits(), profile.fingerprint());
    cache.profiles_used.insert(key);
    if let Some(found) = cache.profiles.get(&key) {
        return profile.with_bake(found.clone());
    }
    let baked = profile.baked(cells, reads);
    if let Some(grid) = baked.bake() {
        cache.profiles.insert(key, grid.clone());
    }
    baked
}

/// The range of contours, either side of a profile's own boundary, that will be read off it.
///
/// **Public because the bake is built in three places and all three have to agree**: the field
/// compiles one, the viewport builds the shader's from the same rule, and `gpu_matches_cpu`
/// builds a third to compare them. A band wide on one side and not the other is a surface in two
/// places, which is exactly what the shader-versus-field measurement exists to catch -- and did,
/// at 1.32 mm, once a draft was there to move the reading off the zero set.
///
/// Two things move a reading off the boundary, and both had to be found the same way -- by a
/// user seeing the band's edge in a picture.
///
/// **The extrude's own draft.** At height `z` a drafted wall is the `d = z·tan(draft)` contour,
/// so the furthest read is at the far end of the sweep. Signed by the draft's angle, not the
/// depth's: flipping a depth flips the direction of travel and the draft with it, leaving the
/// same contours.
///
/// **And whatever consumes it.** A fillet of radius `k` evaluates both its operands wherever
/// either is within `k` of its own surface -- that is what a fillet *is* -- so it reads this
/// profile out to `k` on both sides. An offset and a shell do the same by their own amounts, and
/// they compose: a 10 mm fillet that is then offset 5 mm reads 15 mm. This was found when a
/// 10 mm fillet between an extrude and a sphere came out in terraces, against a band four
/// tolerances wide. `examples/blend_over_a_bake.rs` measures it: while the radius fits in the
/// band the baked and exact fields agree to 0.07 mm, and past it they part by 0.87 mm with the
/// difference stepping at 24 mm per mm -- which a blend turns into a visible crease.
pub fn profile_reads_out_to(document: &Document, graph: &Graph, node: NodeId) -> (f64, f64) {
    let resolve = |r: &fieldform_scene::ParamRef| document.resolve(r);
    // Each side's far wall, since with two sides each has a draft of its own and they may
    // open opposite ways: the band has to reach the furthest contour either side reads.
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    if let Some(fieldform_scene::Node::Extrude { height, draft, sides, .. }) = graph.node(node) {
        let mut reaches = |height: f64, draft: f64| {
            let own = draft_offset(height, draft);
            lo = lo.min(own);
            hi = hi.max(own);
        };
        reaches(resolve(height).unwrap_or(0.0), resolve(draft).unwrap_or(0.0));
        if let fieldform_scene::Sides::Two { height, draft } = sides {
            reaches(resolve(height).unwrap_or(0.0), resolve(draft).unwrap_or(0.0));
        }
    }
    let asked = demanded_of(document, graph, node, 0);
    (lo - asked, hi + asked)
}

/// `reads` widened by `by` on both sides: what a bake built *for the display* asks for, over
/// what the field itself asks for.
///
/// **The third thing that reads off the boundary, and the one the field cannot know about.**
/// The viewport's occlusion samples the composed field out to a radius of its own along every
/// surface's normal -- `fieldform_gpu::Occlusion::reach`, six percent of the scene -- and the
/// shadow starts its march inside that. The band as `profile_reads_out_to` sizes it reaches a
/// block's diagonal past the boundary, a third of the way, so the outer samples read the coarse
/// level at some heights and the fine one at others, by where the surface happens to sit in the
/// block grid; each one that falls outside reads short by seven fine margins, and the surface
/// comes out striped at the block's period. `examples/bake_normal_bands.rs` measures it on a
/// vase that is open everywhere, asking each field from its own surface as the shader does:
/// 0.75 to 1.00 down the surface in three flat levels -- 1.00, 0.90, 0.75 by how many of the
/// five samples fall outside -- stepping about twice a block, where the exact field reads 1.00
/// throughout; widened by the radius it reads 1.00 throughout too, for a bake of 15 ms against
/// 5 and three times the blocks. Found the same way as the other two, by the owner seeing it
/// in a picture, 2026-10-02.
///
/// Both sides, because a cut reads a section's inside as the slot's walls. Only the viewport's
/// bakes take this; the field's own, and the harness that compares the two, read what the
/// field reads.
pub fn reads_widened(reads: (f64, f64), by: f64) -> (f64, f64) {
    let by = if by.is_finite() { by.max(0.0) } else { 0.0 };
    (reads.0 - by, reads.1 + by)
}

/// The most a corner of the section can move when the walls move `grown`.
///
/// **A wall moves by the offset and a mitred corner moves further** -- `1 / sin(half the
/// included angle)` times as far, held to [`fieldform_scene::MITRE_LIMIT`]. A box padded by the
/// wall's own movement would clip the corners off a sharply drafted solid, which is the one
/// failure an extent must not have, so the padding takes the limit rather than the section's
/// own worst corner: loose by up to the limit over the true factor (1.41 at a right angle),
/// and the alternative is flattening the section here to find out, at every call.
///
/// Round corners never outrun their wall, so they are the offset itself.
fn corner_reach(grown: f64, corners: fieldform_scene::Corners) -> f64 {
    match corners {
        fieldform_scene::Corners::Round => grown,
        fieldform_scene::Corners::Sharp => grown * fieldform_scene::MITRE_LIMIT,
    }
}

/// How far a draft moves an extrude's walls at `height` from its plane: `|height| · tan(draft)`,
/// negative where the draft narrows, and nought for a draft at or past vertical or not a number.
///
/// **One spelling for two readers**, which is why it is a function. The bake reads out this far
/// (`profile_reads_out_to`), and the extent has to reach this far (`extent_of`'s extrude arm) —
/// and the extent used to stop at the drawn profile, so a frustum flaring twenty millimetres had a
/// box of its base alone while the bake beside it knew better.
fn draft_offset(height: f64, draft_degrees: f64) -> f64 {
    if height.is_finite() && draft_degrees.is_finite() && draft_degrees.abs() < 90.0 {
        height.abs() * draft_degrees.to_radians().tan()
    } else {
        0.0
    }
}

/// How far off its own surface anything downstream of `node` reads it, in millimetres.
///
/// Taken as the largest demand over every consumer, because one bake serves them all, and
/// accumulated down a chain, because an offset of a fillet reads through both.
fn demanded_of(document: &Document, graph: &Graph, node: NodeId, depth: usize) -> f64 {
    // The same guard the other graph walks use: a cyclic graph is refused at compile time, but
    // this runs before that and must not recurse forever.
    if depth > 256 {
        return 0.0;
    }
    let resolve = |r: &fieldform_scene::ParamRef| document.resolve(r).unwrap_or(0.0);
    let mut most: f64 = 0.0;
    for (id, consumer) in graph.nodes() {
        if !consumer.inputs().contains(&node) {
            continue;
        }
        let above = demanded_of(document, graph, id, depth + 1);
        let here = match consumer {
            // A fillet reads both operands out to its radius; that is what makes it a fillet.
            // To the largest radius a ramp can give it, where one drives it.
            fieldform_scene::Node::Blend { fillet, .. } => {
                above + lipschitz::fillet_reach(document, graph, fillet)
            }
            // An offset marches to the `d = r` contour, so it reads its input that far out.
            fieldform_scene::Node::Offset { distance, .. } => above + resolve(distance).abs(),
            // A shell is two offsets, and the further of them sets the reach.
            fieldform_scene::Node::Shell { thickness, .. } => above + resolve(thickness).abs(),
            // A similarity rescales distances, so a demand above it is a smaller demand here.
            fieldform_scene::Node::Transform { scale, .. } => {
                let s = resolve(scale);
                if s.is_finite() && s.abs() > 1e-9 { above / s.abs() } else { above }
            }
            // Everything else passes a distance through unchanged -- booleans depend only on
            // sign, and rigid motions and patterns move distances without stretching them.
            _ => above,
        };
        most = most.max(here);
    }
    most
}

#[cfg(test)]
mod shell_extent_tests;

#[cfg(test)]
mod tests;


#[cfg(test)]
mod boolean_list_tests;

#[cfg(test)]
mod point_as_a_frame;

/// `blend-integrity-spec.md` §2 and §8: what a blend does to a face its operands share, and what
/// it may claim about its own magnitude there.
///
/// Every older blend test builds perpendicular walls, where the two gradients are orthogonal, the
/// fillet is an exact arc and the blended value's slope is 1. A *shared* face is the other
/// extreme — the gradients are parallel — and it is the case that was never built.
#[cfg(test)]
mod blend_integrity_tests;

#[cfg(test)]
mod leaf_cache_tests;

/// A thin extrude through the whole graph: the chain walked, its plane chosen, the wall built
/// on it and the solid placed. `wall.rs` holds the wall to its own definition; these hold the
/// block to what `outline-wall-spec.md` §7 says it should be, in closed form.
#[cfg(test)]
mod thin_extrude_tests;

/// A section answered from a grid, and what is owed of it when it is *subtracted*.
///
/// A baked section's reading is the true one less a margin. Outside the section that is a
/// lower bound, which is the contract. Cut the section out of something and the slot is
/// marched from the section's **inside**, by the depth — and a depth with a margin taken off
/// a negative number has a margin *added* to it. `gpu_matches_cpu` found it while walls were
/// being given a bake of their own: a slot cut by a baked wall had its sides drawn two
/// millimetres deep in the block. A profile had read the same way since profiles were first
/// baked, and no scene cut one out at a tolerance coarse enough to show it.
#[cfg(test)]
mod baked_cut_tests;

/// A thin extrude at a tolerance loose enough to bake at, and the shelf its grid is kept on.
#[cfg(test)]
mod baked_wall_tests;

/// A curve given a thickness, turned about an axis and carried along a path: the thin revolve
/// and the open-section sweep. Each is held to a solid whose distance is known in closed form,
/// which is what says the wall was laid where the block says it is and to the side it says.
#[cfg(test)]
mod thin_revolve_and_sweep_tests;

/// An extrude's start and its sides, through the whole graph: the solids
/// `extrude-revolve-options-spec.md` section 2.3 says they are, in closed form or against the
/// solid they must equal.
#[cfg(test)]
mod extrude_sides_tests;

/// A profile solid's rims, rounded or chamfered, through the whole graph: the shapes
/// `round-chamfer-spec.md` section 4 says they are, and that none overstates.
#[cfg(test)]
mod rims_tests;

/// A revolve through part of a turn, through the whole graph: the solids
/// `extrude-revolve-options-spec.md` section 3.3 says it makes, against references written
/// without the leaf's own arithmetic.
#[cfg(test)]
mod revolve_wedge_tests;
