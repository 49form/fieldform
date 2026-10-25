// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Turns a cage's limit surface into a signed distance field, and composes fields.
//!
//! **Nothing here depends on Fidget any more.** The M0 spike proved a cage could be
//! embedded in a Fidget expression as a per-point `Var::V`, and `bridge.rs` did that for a
//! year without a single caller outside its own tests: the viewport went to generated WGSL
//! (`fieldform-gpu`), meshing went to our own dual contouring (`mesh.rs`, which says why),
//! and the graph composes `ScalarField`s directly (`graph_field.rs`). Keeping a bridge
//! nothing crossed kept an MPL-2.0 dependency in the product for nothing, so it is gone.
//! Three examples still measure against Fidget's own tape, which is what a dev-dependency
//! is for.
//!
//! # What "signed distance" means here
//!
//! The distance is measured to a dense triangulation of the limit surface, whose vertices
//! lie *exactly* on that surface (see `fieldform_subd::limit_cage`). So the field is exact
//! at those vertices and interpolates linearly between them; refining the subdivision level
//! tightens the approximation. Truly continuous evaluation needs the eigenbasis evaluator
//! and Newton iteration for closest-point-on-patch — still ahead.
//!
//! The **sign** is not approximate: it uses the angle-weighted pseudonormal test
//! (Bærentzen & Aanæs), which is provably correct for closed manifold meshes — including
//! when the closest point falls exactly on an edge or vertex, where a naive
//! "compare against the closest triangle's normal" test gives wrong answers in concave
//! regions.

mod adaptive;
mod attract;
mod bvh;
mod curve_field;
mod extent;
mod graph_field;
mod lattice;
mod measure;
mod mesh;
mod overlay;
pub use overlay::{
    display_graph_with_stand_ins, is_reference_body, pick_bounds, plane_frames, point_positions,
    KeptReadings, OverlayInputs, OverlayReadings,
};
mod primitive;
mod baked_profile;
mod lofted;
mod profile;
mod sdf;
pub mod self_intersection;
mod shared_faces;
pub use shared_faces::{merge_loop_across_plane, merged_section_curve, with_shared_faces_merged};
mod surface_map;
pub use surface_map::{
    mesh_fold_mm, query_packed, PackedHit, PackedSurface, SurfaceLayout, SurfaceMap, SurfacePoint,
};
mod simplify;
mod tessellate;
mod spine;
mod swept;
mod verify;
mod wall;
mod pattern_layout;
pub use pattern_layout::{
    grid_dual, pattern_body_copy, pattern_body_copy_grid, PatternLayout, PatternRefusal, PatternShape,
};

pub use attract::{
    FOLD_PUSH_STRENGTH, FOLD_STRENGTH, PEAK_PULL_AT, folds,
    max_pull, stretch,
    warp_point,
};
pub use curve_field::{CurveField, DEFAULT_TOLERANCE_MM};
pub use extent::Extent;
pub use graph_field::{
    AXIS_FLOOR, CompileError, GraphField, LeafCache, approximate_blends,
    carries_true_distance, driven_blend_divisor, driven_lattice_divisor, driven_morph_divisor,
    driven_offset_divisor, driven_pipe_divisor, fillet_reach, graph_extent, graph_lipschitz,
    levels_for_tolerance, loft_built, loft_parts, loft_stretch, LoftParts, mirror_one_side,
    mirror_reaches_its_plane,
    operand_carries_true_distance, operand_extent, operand_lipschitz,
    outline_miters, packed_surfaces, part_extent, pattern_grid_anchor_in,
    pattern_origin_in, pattern_reach, pattern_reach_axes_in, pattern_reach_in,
    profile_reads_out_to, reach_across_axis, reads_widened, red_cage_behind, surface_levels, sweep_axes,
    sweep_frame, sweep_law, sweep_lean, sweep_offset, sweep_reads_true, sweep_stretch, sweep_tube,
    taper_ratio, wall_bakes_under,
    resolve_frame, resolve_pivot, resolve_place, resolve_place_for, rotation_from_z,
    ends_at, placement_above, turn_rotation, with_turns_flattened,
    may_divide, step_multiplier, step_multiplier_with, sweep_direction, sweep_lean_degrees,
    value_scale_with,
    revolve_wedge,
    ExtrudeSection, block_wall, extrude_section, outline_wall, outline_wall_baked,
    sweep_outline_section,
};
// `Region` and `ShaderBake` are named by nothing outside, but another crate is handed each of
// them — `GraphField::for_region` returns a region to `brick_timing`, `BakedProfile::for_shader`
// a layout to the renderer — and a type another crate receives is one it should be able to name,
// which only the root can promise while the modules are private.
pub use graph_field::Region;
pub use baked_profile::ShaderBake;
pub use fieldform_scene::Banding;
pub use lattice::{
    feature_size_mm, voronoi_hash3, voronoi_measure, voronoi_seed, BeamKind, CellKind, Lattice, LatticeMap, Phase,
    PrismKind, StochasticKind, TpmsKind,
    phase_of, r3_point, uniform_density, RelativeDensity, Unresolved,
};
/// Item 94: published stiffness and strength fits, against the density [`uniform_density`]
/// reads off the field. A module as well as a re-export, because its names -- `Fit`, `Source`,
/// `Reading` -- read best qualified.
pub mod property_table;
pub use property_table::{
    default_row, excluded_for, other_axis_row, ranked_rows, rows_for, DataKind, DensityBasis,
    DensityRange, ExcludedRow, Exponent, Fit, LoadDirection, Lower, Property, PropertyClass,
    PropertyRow, Provenance, Reading, Refusal, SolidBasis, Source, Standing, StrengthKind,
    EXCLUDED, ROWS,
};
pub use measure::{
    Anchored, MassProperties, Measured, Thinness, thinness,
};
pub use verify::{
    find_overstep, verify_critical, verify_unprotected, BlendMovement, CriticalReport, Overstep,
    Verdict,
};
pub use adaptive::{MeshTolerance, mesh_field_adaptive};
pub use mesh::{FieldMesh, mesh_field};
pub use baked_profile::{BakedProfile, bake_cells, band_range, LIST_REACH};
pub use profile::fillet_under_lean;
pub use profile::{
    LoopError, Profile, ProfileError, ProfileLoop, Revolution,
    Meeting, axis_beside, lean_shortfall, loop_from, loop_is_flat, loop_meetings, plane_of,
    profile_loop, profile_loop_turning, profile_partition, profile_seed,
    Chain, PROFILE_TURN_DEGREES, outline_chain,
};
pub(crate) use profile::{
    Depth, Extrusion, RevolvedWedge, SkewExtrusion, chain_plane, outline_chain_turning,
};
pub use primitive::{Cuboid, Edge, EdgedCuboid, SkewedBox, Sphere};
pub(crate) use primitive::{EdgedCone, EdgedCylinder};
pub(crate) use primitive::{Capsule, Cone, Cylinder, Plane, Point, Torus};
pub use simplify::{Simplified, improve_shape, relax, simplify, weld};
pub use tessellate::{Cancel, Progress, Stage, Tessellated, tessellate};
pub use sdf::{CageField, ClosestPoint, MeshError, MeshField};
pub use spine::{Nearest, Spine};
pub use lofted::{derived_centerline_cubic, LoftError, Lofted};
pub(crate) use lofted::{derived_centerline, Smooth};
pub use swept::{Held, SectionLaw, SpineFrame, SweepError, Swept, Taper};
pub(crate) use swept::max_curvature;
pub use wall::{Footprint, Wall, WallError, WallPart};

use nalgebra::{Point3, Vector3};

/// What a field's *magnitude* means, as opposed to its sign.
///
/// Every [`ScalarField`] gets the sign right and never overestimates. This says whether the
/// number itself is a distance, and it exists because two things downstream want different
/// guarantees from the same value.
///
/// A **boolean**'s surface is decided by sign alone. `min(a, b)` is zero exactly where `a` or
/// `b` is zero, whatever their magnitudes, so a lower-bound operand moves nothing. Sphere
/// tracing and interval pruning are the same: both need a value that never overestimates, and
/// neither cares that it understates.
///
/// A **blend**'s surface is decided by the magnitudes. `smooth_min(a, b, k)` mixes the two by
/// how far apart they are, so a field reading 60% of the true distance pulls the seam toward
/// itself and the blend that comes out is asymmetric with a radius that is not the number the
/// user typed. That is the failure D8 in the architecture notes describes, and it is silent —
/// the geometry looks plausible and is wrong.
///
/// So a blend requires [`Normalization::Distance`] operands, and a boolean accepts either.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// The magnitude is the Euclidean distance: unit gradient, and a radius means what it says.
    ///
    /// CSG creases are the long-standing exception — `min` underestimates inside a concave
    /// join and `max` outside a convex one, by a bounded amount, local to the crease. That is
    /// inherent to composing distance fields and predates this enum; it is not what
    /// [`Normalization::LowerBound`] is for.
    Distance,
    /// The magnitude only bounds the distance from below, and its scale is not a distance.
    ///
    /// A lattice is the motivating case. `sin(X)cos(Y) + sin(Y)cos(Z) + sin(Z)cos(X)` has the
    /// right zero set and the right sign, but dividing it by a bound on its gradient gives a
    /// value that runs well under the true distance — measurably 58–71% of it for a gyroid
    /// divided by the provable `sqrt(6) k`. Safe to trace, safe to prune, wrong to blend.
    LowerBound,
}

/// The worst deviation from unit gradient over `points`, measured by central differences.
///
/// This is the conformance check for [`Normalization::Distance`]. D8 in the architecture notes
/// asks for it to be applied to every field source as a matter of course rather than as a
/// one-off, because a source that quietly fails it makes every blend in the document subtly
/// wrong, and the symptom appears nowhere near the cause.
///
/// Sample away from surfaces and creases: the gradient is genuinely undefined on a crease and
/// at the centre of a sphere, so a point there fails this for reasons that are not a bug.
pub fn gradient_deviation(field: &dyn ScalarField, points: &[Point3<f64>], h: f64) -> f64 {
    let mut worst: f64 = 0.0;
    for &p in points {
        let d = |o: Vector3<f64>| field.distance(p + o);
        let grad = Vector3::new(
            d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
            d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
            d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
        ) / (2.0 * h);
        worst = worst.max((grad.norm() - 1.0).abs());
    }
    worst
}

/// The worst amount by which a field claims more room than there is, over `points`.
///
/// This is the conformance check for [`Normalization::LowerBound`], and the other half of what
/// D8 asks for: [`gradient_deviation`] answers "is this a distance?", which a lower bound is
/// deliberately not, so a lattice or a warp needs a different question asked of it. That
/// question is the *contract* itself — [`ScalarField::distance`] must never overestimate — and
/// it is the one property every source has to hold whether it is exact or not.
///
/// **Marched past the claim, because a march that stops at the claim can only confirm it.**
/// From each point, walk each direction in steps until the field goes negative; the first
/// crossing must be no nearer than the value claimed there. Returning zero means every claim
/// held. Anything above zero is how far *inside* the surface a ray following this field would
/// have travelled believing it was still outside — which is a hole in the rendered geometry,
/// appearing and disappearing with the viewing angle.
///
/// `step` sets both the resolution and the cost: the answer is only accurate to one step, so a
/// step much smaller than the feature being tested is what makes a pass meaningful. `reach` is
/// how far past the claim to keep looking, and 1.2 is enough to catch a bound that is wrong by
/// a fifth while staying cheap.
///
/// Points inside the surface are skipped rather than failed: the contract is about the room
/// *before* a surface, and a negative value has already passed one.
pub fn overstatement(
    field: &dyn ScalarField,
    points: &[Point3<f64>],
    directions: &[Vector3<f64>],
    step: f64,
    reach: f64,
) -> f64 {
    let mut worst: f64 = 0.0;
    for &p in points {
        let claimed = field.distance(p);
        if claimed <= 0.0 {
            continue;
        }
        for d in directions {
            let Some(d) = d.try_normalize(1e-12) else {
                continue;
            };
            let mut t = 0.0;
            while t < claimed * reach {
                if field.distance(p + d * t) <= 0.0 {
                    worst = worst.max(claimed - t);
                    break;
                }
                t += step;
            }
        }
    }
    worst.max(0.0)
}

/// Six axis directions and four diagonals, which is the spread a surface is usually missed by.
///
/// Offered rather than left to each caller because the directions decide what the check can
/// catch: axes alone miss a bound that is wrong only across a corner, and a caller writing its
/// own list tends to write the ones its own geometry is aligned to.
pub fn probe_directions() -> Vec<Vector3<f64>> {
    vec![
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.577, 0.577, 0.577),
        Vector3::new(-0.577, 0.577, -0.577),
        Vector3::new(0.577, -0.577, 0.577),
        Vector3::new(-0.577, -0.577, 0.577),
    ]
}

/// Anything that answers "how far is this point from my surface, and which side is it on?".
///
/// This is the seam the architecture calls *fields as the universal currency*. A cage limit
/// surface and a pipe swept along a curve have nothing in common structurally, but both
/// answer this one question — so the graph composes them without knowing which is which,
/// and a boolean between them needs no knowledge of either.
///
/// # The contract
///
/// [`distance`] must be negative inside, positive outside, and must **never overestimate** the
/// Euclidean distance to the surface. Returning the exact distance is ideal; returning a lower
/// bound is permitted. Overestimating is not, because sphere tracing and interval pruning both
/// treat the value as a safe travel distance, and a field that overestimates makes them step
/// through surfaces.
///
/// Exactness is not required, but *boundedness* is: [`lipschitz`] states how fast the returned
/// value can change per unit of distance travelled, and it must be an upper bound. A true
/// Euclidean distance has unit gradient, so the default of `1.0` is correct for one and this
/// whole paragraph costs existing implementations nothing.
///
/// The relaxation is deliberate. A gyroid — `sin(X)cos(Y) + sin(Y)cos(Z) + sin(Z)cos(X)` — has
/// the right zero set and the right sign, but its magnitude is not a distance and its gradient
/// is nowhere near unit length. Under a contract demanding true distance it could not enter a
/// composed graph at all without silently breaking the guarantees the contract exists to
/// protect. Under this one it declares its constant and everything downstream stays correct.
/// See `examples/gyroid_spike.rs`, which measures what happens when a field of that shape is
/// used as though it were a true distance.
///
/// [`distance`]: ScalarField::distance
/// [`lipschitz`]: ScalarField::lipschitz
pub trait ScalarField: Send + Sync {
    fn distance(&self, p: Point3<f64>) -> f64;

    /// The field's exact analytic normal near `p`, if it has one.
    ///
    /// `None` (the default) tells the caller to fall back to finite-differencing [`distance`],
    /// which is correct for every field but costs accuracy: a field built from a
    /// triangulation, like [`CageField`], has a gradient that is piecewise constant across
    /// each triangle, so finite-differencing it reproduces flat facets no matter how small
    /// the step is. [`CageField`] overrides this with a real answer; nothing else needs to.
    ///
    /// [`distance`]: ScalarField::distance
    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        let _ = p;
        None
    }

    /// Whether this field's magnitude is a distance or merely a bound on one.
    ///
    /// The default claims [`Normalization::Distance`], matching the default of `1.0` from
    /// [`lipschitz`] — together they say "this is an ordinary true distance field", which is
    /// what every source in this crate is. **A source whose magnitude is not a distance must
    /// override this**, or blends against it will be silently wrong rather than refused.
    /// [`crate::gradient_deviation`] is the check that catches a wrong claim.
    ///
    /// [`lipschitz`]: ScalarField::lipschitz
    fn normalization(&self) -> Normalization {
        Normalization::Distance
    }

    /// A copy of this field shortened to what can matter inside a box, when there is one.
    ///
    /// **Backlog item 17's mechanism, offered where any field can be asked for it.** A composed
    /// field is mostly branches that cannot reach a given region — a pattern's other instances, a
    /// part's other bodies — and evaluating them anyway is the whole cost of meshing and baking.
    /// [`crate::GraphField::for_region`] is what answers; everything else keeps this default and
    /// is sampled whole, which is the right answer for a leaf with no branches to drop.
    ///
    /// `None` means "sample me as I am", so a caller never has to know which kind of field it
    /// holds. That is what lets `mesh_field` and the bake gain this without either of them
    /// naming `GraphField`.
    ///
    /// The box is the caller's to choose and the promise is only about points inside it: the
    /// shortened field is not a field anywhere else and must not be kept past the region it was
    /// asked for.
    fn pruned_to(
        &self,
        _lo: Point3<f64>,
        _hi: Point3<f64>,
    ) -> Option<std::sync::Arc<dyn ScalarField + Send + Sync>> {
        None
    }

    /// An upper bound on how fast [`distance`] changes per unit of distance travelled.
    ///
    /// `1.0` — the default, and correct for any true Euclidean distance field — says the value
    /// can change by at most a millimetre per millimetre moved. A field whose magnitude is a
    /// pseudo-distance rather than a distance reports the constant that bounds *its* rate of
    /// change instead.
    ///
    /// This must be an upper bound, not a typical value. It divides the safe step in a sphere
    /// tracer and widens the slack in [`interval`], so understating it produces surfaces that
    /// are stepped through rather than merely rendered slowly.
    ///
    /// [`distance`]: ScalarField::distance
    /// [`interval`]: ScalarField::interval
    fn lipschitz(&self) -> f64 {
        1.0
    }

    /// The thinnest thing this field can hold, in millimetres — a lattice's wall, a strut's
    /// diameter — or `None` when it declares none.
    ///
    /// **For a mesher that reads the field only at corners.** A wall thinner than a cell can pass
    /// between two corners with both of them outside it, and a sheet lattice is nothing *but*
    /// such walls: on 21 mm cells a 2 mm Schwarz P band was found or not according to where the
    /// grid happened to fall, and in one box it was not found at all — 0 triangles for a part
    /// that meshes to 130 000 a few millimetres away (`examples/surface_edit_tessellation.rs`).
    /// A mesher told this keeps dividing wherever the surface might be until its cells are no
    /// larger than it, and then no wall can fit between two corners of an edge.
    fn feature_size(&self) -> Option<f64> {
        None
    }

    /// Conservative bounds on this field's value everywhere inside an axis-aligned box.
    ///
    /// This is the method that makes pruning possible. A caller holding a region of space can
    /// ask what the field could possibly do there, and act on the answer without sampling: a
    /// box whose lower bound is positive contains no surface and can be skipped whole, and a
    /// subexpression whose interval cannot affect a composed result can be dropped from the
    /// evaluation entirely. That is what the per-body box guard and the skip grid do here
    /// (`fieldform-gpu`), and a field that can only answer at points cannot take part in it —
    /// which is why a cage reports its box and is read at every point inside it.
    ///
    /// The returned pair must *contain* every value [`distance`] takes inside the box. Wider
    /// than necessary is safe and merely slow; too narrow is a correctness bug that surfaces
    /// as geometry vanishing.
    ///
    /// The default derives the bound from [`lipschitz`] and one evaluation at the box's
    /// centre: nothing in the box is further from the centre than the half-diagonal, so no
    /// value inside can differ from the centre's by more than that times the constant. Exact
    /// for a plane and loose for a box straddling a tight concavity, but always valid.
    /// Implementations with cheap access to their own geometry — a mesh field with a bounding
    /// volume hierarchy, say — can do considerably better by overriding this.
    ///
    /// [`distance`]: ScalarField::distance
    /// [`lipschitz`]: ScalarField::lipschitz
    fn interval(&self, lo: Point3<f64>, hi: Point3<f64>) -> (f64, f64) {
        let centre = Point3::from((lo.coords + hi.coords) * 0.5);
        let slack = self.lipschitz() * (hi - lo).norm() * 0.5;
        let d = self.distance(centre);
        (d - slack, d + slack)
    }
}

impl ScalarField for CageField {
    fn distance(&self, p: Point3<f64>) -> f64 {
        CageField::distance(self, p)
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        Some(self.closest(p).normal)
    }

    // `interval` is deliberately left as the trait's default. A tighter bound is available
    // here — the bounding volume hierarchy could be walked against the query box to get the
    // true minimum and maximum distance to the triangulation, rather than inferring both from
    // one centre sample — and that is worth doing when pruning becomes the bottleneck. It is
    // not worth doing before there is a measurement saying the default's slack costs anything.
}

#[cfg(test)]
mod tests {

    /// The overstatement check catches a field that overstates, which is the only evidence
    /// that a zero from it means anything.
    ///
    /// A property test that cannot fail is worse than no test: it reports success for a
    /// mechanism that was never exercised. So the instrument is pointed at a field built to
    /// break the contract — a sphere whose distance is doubled, which is exactly the failure
    /// the contract forbids — and at the honest one beside it.
    #[test]
    fn the_overstatement_check_catches_a_field_that_claims_too_much() {
        struct Doubled(crate::Sphere);
        impl ScalarField for Doubled {
            fn distance(&self, p: Point3<f64>) -> f64 {
                self.0.distance(p) * 2.0
            }
        }

        let honest = crate::Sphere {
            centre: Point3::origin(),
            radius: 10.0,
        };
        let liar = Doubled(crate::Sphere {
            centre: Point3::origin(),
            radius: 10.0,
        });
        let probes: Vec<Point3<f64>> = (0..12)
            .map(|i| {
                let t = i as f64;
                Point3::new(14.0 + t * 1.3, t * 0.7, -3.0 + t * 0.5)
            })
            .collect();
        let directions = crate::probe_directions();

        assert!(
            crate::overstatement(&honest, &probes, &directions, 0.01, 1.2) < 1e-9,
            "an exact sphere overstates nothing",
        );
        let caught = crate::overstatement(&liar, &probes, &directions, 0.01, 1.2);
        assert!(
            caught > 1.0,
            "a field claiming twice the room should be caught, and was off by {caught}",
        );
    }
    use super::*;
    use fieldform_cage::Cage;

    /// Samples a box densely and returns the extreme values the field actually takes in it.
    fn observed_range(field: &dyn ScalarField, lo: Point3<f64>, hi: Point3<f64>) -> (f64, f64) {
        let n = 12;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    let t = |a: f64, b: f64, s: usize| a + (b - a) * s as f64 / n as f64;
                    let p = Point3::new(t(lo.x, hi.x, i), t(lo.y, hi.y, j), t(lo.z, hi.z, k));
                    let d = field.distance(p);
                    min = min.min(d);
                    max = max.max(d);
                }
            }
        }
        (min, max)
    }

    /// A field whose value moves three times as fast as position — the shape of any lattice or
    /// other pseudo-distance, reduced to something with an answer that can be checked by hand.
    struct Steep;

    impl ScalarField for Steep {
        fn distance(&self, p: Point3<f64>) -> f64 {
            // A plane through the origin, but scaled so the gradient has magnitude 3.
            p.x * 3.0
        }
        fn lipschitz(&self) -> f64 {
            3.0
        }
        fn normalization(&self) -> Normalization {
            Normalization::LowerBound
        }
    }

    /// The D8 conformance check, as a roll-call over every field source that claims a true
    /// distance unconditionally.
    ///
    /// **The list is the point.** Seven of these shapes have a thorough gradient test of their
    /// own beside the implementation, with the crease filter each shape needs — the eight named
    /// `..._has_a_unit_gradient_...` in `primitive.rs`, and one in `sdf.rs`. Three do not:
    /// `Plane`, `Point` and `CurveField` are covered here and nowhere else, `CurveField`
    /// additionally by `a_curved_pipe_has_a_unit_gradient_away_from_the_inside_of_its_bend`,
    /// which is the hard case for it.
    ///
    /// What no per-source test can be is *complete*. D8 asks that every new field source get
    /// the unit-gradient test, and before this list existed a new source could take the trait's
    /// default [`Normalization::Distance`], be wrong about it, and fail nothing at all — the
    /// symptom appearing in some unrelated blend, a long way from the cause. So the probes here
    /// are a handful per source, each sitting squarely inside one smooth feature; depth belongs
    /// to the per-source tests, and coverage belongs here.
    ///
    /// **Adding a field source without adding a line here is the omission D8 names.**
    ///
    /// Three kinds of `ScalarField` are deliberately absent, and saying which is what keeps the
    /// name of this test honest:
    ///
    /// - **Those declaring [`Normalization::LowerBound`]** — `Swept` and `SkewExtrusion`. Unit
    ///   gradient is the wrong question to ask them: they are honest by declaration, so a blend
    ///   against one already reports itself approximate.
    /// - **Those whose claim depends on their own configuration** — `Lattice`, `Extrusion` and
    ///   `Revolution`. "Does this source carry a distance?" has no answer without a
    ///   configuration, which is why a beam lattice on an even rectangular map is exact and the
    ///   same cell on any other map is not. The check belongs where the configuration is chosen:
    ///   `a_rectangular_beam_lattice_carries_a_true_distance` and
    ///   `an_extrusion_and_a_revolution_both_move_at_unit_rate`.
    /// - **`GraphField`**, which composes rather than sources. Its normalization is whatever its
    ///   leaves and operations make it, and
    ///   `the_graph_walk_agrees_with_the_compiler_about_true_distance` is what holds it.
    ///
    /// Test fixtures are not sources and are not listed: several modules define a small
    /// `ScalarField` inside `#[cfg(test)]` to stand in for a shape with a known answer. They are
    /// instruments, not geometry the application can build.
    ///
    /// `CageField` measures to a triangulation rather than to a smooth surface, which is why it
    /// alone carries the looser tolerance this check has always given it.
    #[test]
    fn every_field_source_claiming_distance_has_unit_gradient() {
        use fieldform_scene::{Curve, CurveId};

        // A straight spine is all the roll-call needs; the curved one, where the `min` over
        // many segments could break the claim, has its own test beside the field.
        let line = Curve::polyline(
            CurveId::new(),
            "line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(100.0, 0.0, 0.0)],
            false,
        )
        .expect("two points are a polyline");

        // Name, source, probes, and the tolerance those probes are held to. Every probe sits
        // inside a single smooth feature — one face, one cap, one round side — and never on a
        // crease, an axis, a spine or a centre, where the gradient genuinely has no value and a
        // failure would say nothing about the field.
        let sources: Vec<(&str, Box<dyn ScalarField>, Vec<Point3<f64>>, f64)> = vec![
            (
                "CageField",
                Box::new(CageField::from_cage(&Cage::cube_of(100.0), 4).expect("a cube bakes")),
                vec![
                    Point3::new(120.0, 30.0, 20.0),
                    Point3::new(-90.0, 15.0, -40.0),
                    Point3::new(70.0, -110.0, 35.0),
                    Point3::new(200.0, 200.0, 200.0),
                ],
                1e-3,
            ),
            (
                "CurveField",
                Box::new(CurveField::with_radius(&line, 5.0)),
                vec![
                    // Off the side, off each end's cap, and one inside the wall — all well
                    // clear of the spine itself, where "away from the axis" has no direction.
                    Point3::new(50.0, 12.0, 0.0),
                    Point3::new(30.0, 0.0, 9.0),
                    Point3::new(120.0, 0.0, 0.0),
                    Point3::new(-14.0, 3.0, 4.0),
                    Point3::new(50.0, 2.0, 0.0),
                ],
                1e-6,
            ),
            (
                "Sphere",
                Box::new(Sphere { centre: Point3::new(3.0, -2.0, 1.0), radius: 7.0 }),
                vec![
                    Point3::new(20.0, -2.0, 1.0),
                    Point3::new(3.0, 12.0, 1.0),
                    Point3::new(3.0, -2.0, -14.0),
                    Point3::new(-9.0, 8.0, 6.0),
                    // Inside the wall, 2.4 mm off the centre, which is a sphere's one
                    // singularity.
                    Point3::new(5.0, -1.0, 2.0),
                ],
                1e-6,
            ),
            (
                "Cuboid",
                Box::new(Cuboid {
                    centre: Point3::new(1.0, -2.0, 0.5),
                    size: Vector3::new(40.0, 10.0, 3.0),
                }),
                vec![
                    // Beyond one face only, beyond a corner, beyond the thin face, and inside
                    // where the nearest face is unambiguously the one on the thinnest axis.
                    Point3::new(60.0, -2.0, 0.5),
                    Point3::new(41.0, 13.0, 10.5),
                    Point3::new(1.0, -2.0, 9.5),
                    Point3::new(-40.0, -2.0, 0.5),
                    Point3::new(2.0, -1.0, 0.6),
                ],
                1e-6,
            ),
            (
                "Torus",
                Box::new(Torus {
                    centre: Point3::new(2.0, 1.0, -1.0),
                    major_radius: 6.0,
                    minor_radius: 2.0,
                }),
                vec![
                    // Clear of the ring circle and clear of the axis — the two places the fold
                    // onto the ring has no direction.
                    Point3::new(12.0, 1.0, -1.0),
                    Point3::new(2.0, 4.0, 1.0),
                    Point3::new(-2.0, -3.0, -4.0),
                    Point3::new(8.0, 1.0, 4.0),
                    Point3::new(2.0, -9.0, -1.0),
                ],
                1e-6,
            ),
            (
                "Cylinder",
                Box::new(Cylinder {
                    base: Point3::new(2.0, 3.0, -5.0),
                    top: Point3::new(2.0, 3.0, 5.0),
                    radius: 4.0,
                }),
                vec![
                    // Off the side wall, off a cap, diagonally off the rim, and inside at the
                    // half-height, where the side wall is nearer than either cap by 4 mm.
                    Point3::new(12.0, 3.0, 0.0),
                    Point3::new(6.0, 3.0, 12.0),
                    Point3::new(11.0, 3.0, 9.0),
                    Point3::new(2.0, -9.0, 0.0),
                    Point3::new(5.0, 3.0, 0.0),
                ],
                1e-6,
            ),
            (
                "Cone",
                Box::new(Cone {
                    base: Point3::new(0.0, 0.0, -4.5),
                    top: Point3::new(0.0, 0.0, 4.5),
                    base_radius: 5.0,
                    top_radius: 2.0,
                }),
                vec![
                    Point3::new(14.0, 0.0, 0.0),
                    Point3::new(0.0, 9.0, -2.0),
                    Point3::new(3.0, 0.0, 12.0),
                    Point3::new(-11.0, 0.0, -3.0),
                    // Inside, off the axis, with the slant clearly nearer than either cap.
                    Point3::new(1.5, 0.0, 0.0),
                ],
                1e-6,
            ),
            (
                "Capsule",
                Box::new(Capsule {
                    start: Point3::new(0.0, 0.0, -4.0),
                    end: Point3::new(0.0, 0.0, 4.0),
                    start_radius: 3.0,
                    end_radius: 3.0,
                }),
                vec![
                    Point3::new(9.0, 0.0, 0.0),
                    Point3::new(0.0, 7.0, 2.0),
                    Point3::new(2.0, 0.0, 10.0),
                    Point3::new(-5.0, -5.0, 3.0),
                    // Inside the wall, off the spine, which is a capsule's one singularity.
                    Point3::new(1.5, 0.0, -1.0),
                ],
                1e-6,
            ),
            (
                "Plane",
                Box::new(Plane { origin: Point3::new(0.0, 0.0, 4.0), flipped: false }),
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(10.0, -3.0, 9.0),
                    Point3::new(-7.0, 2.0, 4.5),
                ],
                1e-6,
            ),
            (
                "Point",
                Box::new(crate::Point { at: Point3::new(1.0, 2.0, 3.0) }),
                vec![
                    Point3::new(11.0, 2.0, 3.0),
                    Point3::new(1.0, -8.0, 3.0),
                    Point3::new(4.0, 6.0, -2.0),
                ],
                1e-6,
            ),
        ];

        for (name, field, probes, tolerance) in &sources {
            assert_eq!(
                field.normalization(),
                Normalization::Distance,
                "{name} is on this list because it claims a distance",
            );
            let worst = gradient_deviation(&**field, probes, 1e-4);
            assert!(worst < *tolerance, "{name}: gradient deviates by {worst}");
        }

        // The claim has to be falsifiable, or none of the above proves anything: a field whose
        // magnitude is not a distance must fail it.
        let worst = gradient_deviation(&Steep, &sources[2].2, 1e-4);
        assert!(
            (worst - 2.0).abs() < 1e-6,
            "a field with gradient 3 should deviate from unit by 2, got {worst}"
        );
    }

    /// The one property the whole mechanism rests on. A bound that is too wide costs speed; a
    /// bound that is too narrow tells a caller a box is empty when it is not, and the geometry
    /// inside it silently disappears.
    #[test]
    fn an_interval_contains_every_value_the_field_takes_in_the_box() {
        let field = CageField::from_cage(&Cage::cube_of(100.0), 3).unwrap();

        for (lo, hi) in [
            // Straddling the surface, where the bound matters most.
            (
                Point3::new(40.0, -10.0, -10.0),
                Point3::new(60.0, 10.0, 10.0),
            ),
            // Wholly inside.
            (
                Point3::new(-10.0, -10.0, -10.0),
                Point3::new(10.0, 10.0, 10.0),
            ),
            // Wholly outside, and far.
            (
                Point3::new(200.0, 200.0, 200.0),
                Point3::new(260.0, 260.0, 260.0),
            ),
            // Long and thin, so the half-diagonal is much larger than two of the extents.
            (Point3::new(-80.0, -1.0, -1.0), Point3::new(80.0, 1.0, 1.0)),
        ] {
            let (bound_lo, bound_hi) = field.interval(lo, hi);
            let (seen_lo, seen_hi) = observed_range(&field, lo, hi);
            assert!(
                bound_lo <= seen_lo && bound_hi >= seen_hi,
                "interval ({bound_lo:.3}, {bound_hi:.3}) fails to contain observed \
                 ({seen_lo:.3}, {seen_hi:.3}) for box {lo:?}..{hi:?}",
            );
        }
    }

    /// The bound has to be useful as well as valid: an interval that always straddles zero
    /// would contain every value and prune nothing.
    #[test]
    fn a_box_far_from_the_surface_is_proved_empty_without_sampling_it() {
        let field = CageField::from_cage(&Cage::cube_of(100.0), 3).unwrap();

        let (lo, hi) = (
            Point3::new(200.0, 200.0, 200.0),
            Point3::new(210.0, 210.0, 210.0),
        );
        let (bound_lo, _) = field.interval(lo, hi);
        assert!(
            bound_lo > 0.0,
            "a box 200 mm clear of a 100 mm cube should be provably outside, got {bound_lo}"
        );

        let (lo, hi) = (Point3::new(-5.0, -5.0, -5.0), Point3::new(5.0, 5.0, 5.0));
        let (_, bound_hi) = field.interval(lo, hi);
        assert!(
            bound_hi < 0.0,
            "a box at the centre of a 100 mm cube should be provably inside, got {bound_hi}"
        );
    }

    /// **An extrude lifted at nought evaluates.** The extrude session shows its cap before it
    /// has been pulled, and at that moment the four walls are quads of no height: every corner
    /// of each sits on its twin. The validator calls that collapsed and would refuse it; the
    /// session tolerates it as a draft. That is only honest if the field is finite there --
    /// the limit normal at a collapsed wall is a cross product of nothing -- so this measures
    /// it: the lifted cube's field is finite on a grid through the cap, and agrees with the
    /// plain cube's away from the cap's edge.
    #[test]
    fn a_cage_whose_walls_have_no_height_yet_still_gives_a_finite_field() {
        use fieldform_cage::edit::extrude::{extrude, Direction, Mode};
        use fieldform_cage::FaceId;
        let cube = Cage::cube_of(100.0);
        let top = cube
            .face_ids()
            .find(|&f| cube.face_vertices(f).iter().all(|&v| cube.vertex_position(v).z > 0.0))
            .expect("a cube has a top");
        let lifted =
            extrude(cube.mesh(), &[top], Mode::Region, Direction::Normal, 0.0).expect("the lift at nought");
        let lifted = Cage::try_from_mesh(lifted.mesh).map_err(|(_, e)| e).expect("a closed lift");
        let field = CageField::from_cage(&lifted, 3).expect("the lift bakes");
        let plain = CageField::from_cage(&cube, 3).expect("the cube bakes");

        let mut worst_gap: f64 = 0.0;
        let mut worst_at = (Point3::origin(), 0.0, 0.0);
        for i in -8i32..=8 {
            for j in -8i32..=8 {
                for k in -8i32..=8 {
                    let p = Point3::new(i as f64 * 10.0, j as f64 * 10.0, k as f64 * 10.0);
                    let d = field.distance(p);
                    assert!(d.is_finite(), "the lifted field is {d} at {p:?}");
                    // The twinned rim is a crease the plain cube does not have, and it pulls
                    // the top of the limit surface out toward the cap: measured 22.7 mm at
                    // (-60, -40, 80), which is the lift being a lift and not a fault. Only the
                    // far half is held to the cube's own value.
                    if p.z < -20.0 {
                        let gap = (d - plain.distance(p)).abs();
                        if gap > worst_gap {
                            worst_gap = gap;
                            worst_at = (p, d, plain.distance(p));
                        }
                    }
                }
            }
        }
        assert!(worst_gap < 5.0, "the lifted cube reads {worst_gap} mm from the plain one at {worst_at:?}");
    }

    /// A field that moves faster than its position must widen its own bound to match, or the
    /// bound is a lie that reads as correct right up until something steps through a surface.
    #[test]
    fn a_steep_field_widens_its_own_interval_by_its_constant() {
        let lo = Point3::new(-1.0, -1.0, -1.0);
        let hi = Point3::new(1.0, 1.0, 1.0);

        let (bound_lo, bound_hi) = Steep.interval(lo, hi);
        let (seen_lo, seen_hi) = observed_range(&Steep, lo, hi);
        assert!(
            bound_lo <= seen_lo && bound_hi >= seen_hi,
            "({bound_lo}, {bound_hi}) must contain ({seen_lo}, {seen_hi})"
        );

        // The slack is the half-diagonal times the constant: sqrt(3) * 3.
        let slack = 3.0 * (hi - lo).norm() * 0.5;
        assert!(
            (bound_hi - slack).abs() < 1e-9,
            "centre is 0, so hi is pure slack"
        );

        // Had it inherited the default of 1.0, the bound would have been too narrow to hold
        // the values the field actually takes — which is the failure this method prevents.
        struct Understated;
        impl ScalarField for Understated {
            fn distance(&self, p: Point3<f64>) -> f64 {
                p.x * 3.0
            }
        }
        let (naive_lo, naive_hi) = Understated.interval(lo, hi);
        assert!(
            naive_lo > seen_lo || naive_hi < seen_hi,
            "understating the constant should produce a bound that fails to contain the field"
        );
    }
}
