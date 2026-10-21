// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Triply periodic minimal surfaces, as a field source.
//!
//! A lattice is the reason the viewport is field-traced rather than meshed. the project notes does
//! the arithmetic: a 1 m cube of 50 mm gyroid cells with 1 mm walls is 6.25e9 triangles and
//! ~225 GB meshed, and sphere-traces in 0.65 s at 1920² with nothing stored. So this is a
//! field and stays one — `gyroid_spike.rs` measured the meshing side and is why.
//!
//! # This field is a lower bound, and that is load-bearing
//!
//! A TPMS formula has the right zero set and the wrong magnitude: its gradient is nowhere near
//! unit length, so `|F|` is not a distance. Dividing by a provable bound on `|∇F|` makes the
//! value 1-Lipschitz and therefore *safe* — it never overestimates, so a sphere tracer cannot
//! step through a wall and interval pruning stays correct. It is still not the distance:
//! `gyroid_spike.rs` measured 58–71% of the true value.
//!
//! So [`ScalarField::normalization`] reports [`Normalization::LowerBound`]. A lattice composes
//! into a part most cleanly by **boolean intersection**, which depends only on sign; blending one
//! into a shell mixes by magnitude, so an operand reading short puts the seam somewhat toward the
//! lattice. **That was a refusal at graph-compile time until D8 was reversed on 2026-09-02, and
//! is now a report**: the blend happens, `approximate_blends` names it, and the panel marks it
//! beside the radius. Wanting the blend was legitimate, which is why it won; the local-gradient
//! normalization D8 names as future work is what would make it exact rather than merely honest.
//!
//! # Two constants per kind, and where they come from
//!
//! Each surface needs a bound on `|∇F|` (for safety) and a *typical* `|∇F|` at the surface (for
//! turning a wall thickness in millimetres into a threshold on `|F|`). The first is proved
//! on [`TpmsKind::gradient_bound`] and checked numerically; the second is measured, because it
//! is an average over a surface and there is nothing to prove. Both are reported by
//! `fieldform-field/examples/tpms_profile.rs`.
//!
//! That example also measures what the lower bound costs, which had been asserted and never
//! quantified: against the distance actually available, the value comes out at **0.65 or less
//! in the worst case and around 0.2 on average**. Never above 1, which is the contract; but a
//! tracer takes roughly three to five times more steps than a true distance would need, and
//! that factor is what local-gradient normalization would buy back.

use crate::{Normalization, ScalarField};
// One `Banding`, not one per crate. The scene owns the vocabulary because that is where
// it is written to the file and chosen in the panel; this crate reads the same type
// rather than a matching one of its own, which is the rule D9 states and the thing two
// enums with the same two variants quietly break.
use fieldform_scene::Banding;
use nalgebra::{Point3, Vector3};

mod beam;
mod map;
mod prism;
mod tpms;
mod voronoi;

pub use beam::BeamKind;
pub use map::LatticeMap;
pub use prism::PrismKind;
pub use tpms::TpmsKind;
pub use voronoi::{voronoi_hash3, voronoi_measure, voronoi_seed, StochasticKind};

/// The smallest material feature a phase implies, in millimetres.
///
/// Free-standing rather than only a method, because the panel has to answer the same question
/// while the user is still typing — it shows how much of a cylindrical lattice will come out
/// solid — and it has no `Lattice` in hand at that point. A second copy of this rule in the
/// panel would disagree with this one eventually, and the symptom would be a readout that is
/// merely wrong rather than anything that fails.
///
/// See [`Lattice::feature_size_mm`] for what the solid arm is estimating and what each
/// direction of its error costs.
pub fn feature_size_mm(banding: Banding, sizing: f64, smallest_cell_extent: f64) -> f64 {
    match banding {
        Banding::Sheet => sizing,
        Banding::Solid => {
            let d = sizing.clamp(0.0, 1.0);
            d.min(1.0 - d) * smallest_cell_extent
        }
        // **The one arm that is exact**, and worth saying so beside the arm that is not: a
        // strut's smallest feature is its diameter, measured rather than estimated. The
        // estimation above is a property of the solid phase, not of the question.
        Banding::Strut => 2.0 * sizing,
    }
}

/// The `n`th point of Roberts' R3 sequence in the unit cube: `frac(½ + n·(1/φ₃, 1/φ₃², 1/φ₃³))`,
/// `φ₃` the positive root of `x⁴ = x + 1`. (The design-time scripts named it the plastic number,
/// which is the root of `x³ = x + 1`, the two-dimensional sequence's constant, and used a value
/// off in the tenth decimal; nothing here depends on theirs.)
///
/// **The one sequence every fill table here was sampled with**, public so that a test sampling
/// the field and an example measuring a table draw the *same* points: the first 2²⁰ a test
/// samples are the first 2²⁰ of the 2²⁴ a table was built from, and the example prints what those
/// read, so a test's tolerance is a measured residual and a failure is the field and the table
/// disagreeing rather than two samplings. Low-discrepancy rather than random, because a volume
/// fraction from it converges near `1/N` rather than `1/√N`: `beam_profile.rs`'s control, the
/// cubic strut against its closed form, is within 0.05% at every row on 2²⁴ points.
pub fn r3_point(n: u64) -> Point3<f64> {
    const PHI3: f64 = 1.220_744_084_605_759_5;
    let a = [1.0 / PHI3, 1.0 / (PHI3 * PHI3), 1.0 / (PHI3 * PHI3 * PHI3)];
    let at = |k: usize| (0.5 + n as f64 * a[k]).fract();
    Point3::new(at(0), at(1), at(2))
}

/// What a lattice's cell is: a minimal surface, or a wireframe of beams.
///
/// **Two families, and they are not variants of one enum by accident.** A TPMS is an implicit
/// surface — a trig sum banded by an isovalue, whose value has the right zero set and the wrong
/// magnitude, so it is divided by a bound on its gradient and comes out a *lower bound* on the
/// distance. A beam cell is a union of capsules, which is an exact distance function. They differ
/// in what their parameter means, in what their gradient bound is, in which bandings they can
/// give, and in whether a blend against them is exact. Sharing an enum would mean five `match`
/// arms answering "which sort is this" in five places.
///
/// **Not `Eq`**, since the fourth family carries a jitter. Nothing compared two of these for
/// equality; `PartialEq` is kept because the tests do.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CellKind {
    Tpms(TpmsKind),
    Beam(BeamKind),
    Prism(PrismKind),
    /// A cell built from a grid of *jittered seeds* rather than from a formula: the first one
    /// whose shape depends on numbers of its own. `voronoi-lattice-spec.md`, Track A.
    Stochastic {
        kind: StochasticKind,
        /// Nought to one. At nought the seeds sit at cell centres; at one a seed may sit
        /// anywhere inside its own cell and **never leaves it**, which is the property every
        /// bound in `voronoi.rs` rests on. Clamped where it is installed rather than trusted.
        jitter: f64,
        /// The arrangement. Folded from the node's stored `u64` -- see
        /// [`Lattice::with_scatter`], which is what installs both.
        seed: u32,
    },
}

impl CellKind {
    pub fn label(self) -> &'static str {
        match self {
            CellKind::Tpms(kind) => kind.label(),
            CellKind::Beam(kind) => kind.label(),
            CellKind::Prism(kind) => kind.label(),
            CellKind::Stochastic { kind, .. } => kind.label(),
        }
    }

    /// The jitter a stochastic cell was built with, after clamping -- what a test asks to check
    /// that [`Lattice::with_scatter`] did clamp it. Every other reader takes the jitter by
    /// matching the cell, so only the tests compile this.
    #[cfg(test)]
    fn stochastic_jitter(self) -> Option<f64> {
        match self {
            CellKind::Stochastic { jitter, .. } => Some(jitter),
            _ => None,
        }
    }

    /// The bound on the cell field's rate of change, per unit of cell coordinate.
    ///
    /// **A beam's is 1, and that is the whole of why it traces faster.** A distance field has
    /// unit gradient by definition, so there is nothing to divide away; a TPMS carries between
    /// 1.13 and 5.05 times more bound than its surface actually has, and every step a ray takes
    /// is divided by that.
    pub fn gradient_bound(self) -> f64 {
        match self {
            CellKind::Tpms(kind) => kind.gradient_bound(),
            // A distance either way, so there is nothing to divide away. A Voronoi measure is
            // not a distance, but it *is* 1-Lipschitz in cell units by the argument in
            // `voronoi_measure`, which is what this number is a bound on.
            CellKind::Beam(_) | CellKind::Prism(_) | CellKind::Stochastic { .. } => 1.0,
        }
    }

    pub fn supported_bandings(self) -> &'static [Banding] {
        match self {
            CellKind::Tpms(kind) => kind.supported_bandings(),
            // One side, so one banding. This is the case `supported_bandings` was made a set
            // for: a sheet of a beam is not a thing to ask for, and the panel never offers it.
            CellKind::Beam(_) => &[Banding::Strut],
            // A wall, so a sheet -- and an exact one, which no surface cell's is.
            CellKind::Prism(_) => &[Banding::Sheet],
            CellKind::Stochastic { kind, .. } => kind.supported_bandings(),
        }
    }
}

/// **Identity, and nothing else.** A stochastic cell's jitter and seed are document values, and
/// the project notes' rule is that a parameter's value is asked of the document rather than read off
/// the parameter -- so they arrive as nought here and are installed by [`Lattice::with_scatter`]
/// where the document is in hand. Nought is a meaning rather than a placeholder: it is a
/// regular grid of seeds, so a forgotten install shows as an unmistakably square foam rather
/// than as a plausible one of the wrong shape.
///
/// This conversion exists for the questions that are about the *kind* -- which bandings it
/// supports, what its gradient bound is -- which the panel asks without a document.
impl From<fieldform_scene::LatticeCell> for CellKind {
    fn from(cell: fieldform_scene::LatticeCell) -> Self {
        match cell {
            fieldform_scene::LatticeCell::Cubic => CellKind::Beam(BeamKind::Cubic),
            fieldform_scene::LatticeCell::Bcc => CellKind::Beam(BeamKind::Bcc),
            fieldform_scene::LatticeCell::Fcc => CellKind::Beam(BeamKind::Fcc),
            fieldform_scene::LatticeCell::Octet => CellKind::Beam(BeamKind::Octet),
            fieldform_scene::LatticeCell::Diamond => CellKind::Beam(BeamKind::Diamond),
            fieldform_scene::LatticeCell::Honeycomb => CellKind::Prism(PrismKind::Honeycomb),
            fieldform_scene::LatticeCell::VoronoiFoam { .. } => CellKind::Stochastic {
                kind: StochasticKind::VoronoiFoam,
                jitter: 0.0,
                seed: 0,
            },
            fieldform_scene::LatticeCell::VoronoiBeam { .. } => CellKind::Stochastic {
                kind: StochasticKind::VoronoiBeam,
                jitter: 0.0,
                seed: 0,
            },
            other => CellKind::Tpms(other.into()),
        }
    }
}

/// A banding together with the number that sizes it.
///
/// **The two parameters are different quantities, and this enum is what stops them being
/// confused.** A sheet has a wall thickness in millimetres, which becomes an isovalue through
/// [`TpmsKind::surface_gradient`] and therefore depends on the frequency the map is driving the
/// cell at where the sample is. A labyrinth has no wall: it has a *relative density*, a
/// dimensionless fraction of a cell's volume, and putting that through the thickness conversion
/// produces a number with no meaning. One live field beside one dormant field is the
/// arrangement where the dormant one eventually gets read.
///
/// One consequence that looks like an omission until it is said: **a solid's isovalue does not
/// vary with position.** A density is a fraction of a cell rather than a length, so it survives
/// a map that squeezes the cells without being rescaled — which is why [`Lattice`]'s
/// isovalue takes a point that its solid arm ignores.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Phase {
    Sheet {
        /// Nominal wall thickness in millimetres. See [`TpmsKind::surface_gradient`] for why
        /// nominal.
        thickness_mm: f64,
        /// How far the wall's middle sits off the cell's own surface, in millimetres, toward
        /// the side where `F` is positive. Zero is a plain sheet. See `LatticePhase::Sheet`.
        offset_mm: f64,
    },
    Solid {
        /// Fraction of a cell's volume that is material, in `0..=1`, read through
        /// [`TpmsKind::isovalue_for_density`].
        density: f64,
    },
    Strut {
        /// Radius of the beams, in millimetres, subtracted from a distance — so unlike the other
        /// two this number is exact, with nothing averaged or tabulated between what is asked
        /// for and what is built.
        radius_mm: f64,
    },
}

impl Phase {
    /// A plain sheet: a wall of this thickness centred on the cell's own surface.
    pub fn sheet(thickness_mm: f64) -> Self {
        Phase::Sheet { thickness_mm, offset_mm: 0.0 }
    }

    /// Which side is material, without the number that sizes it.
    pub(crate) fn banding(self) -> Banding {
        match self {
            Phase::Sheet { .. } => Banding::Sheet,
            Phase::Solid { .. } => Banding::Solid,
            Phase::Strut { .. } => Banding::Strut,
        }
    }
}

/// Which number of a block's phase did not resolve, named the way the panel names it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Unresolved {
    /// `"wall"`, `"offset"`, `"density"` or `"radius"`.
    pub which: &'static str,
}

/// A block's phase as the kernel builds it: **the one conversion**, used by the compile and by
/// the property readout alike, so the two cannot disagree about what a block is -- the argument
/// [`feature_size_mm`] already makes for the smallest feature.
///
/// `own` reads the phase's sizing number. The compile passes the typed number, or a driving
/// ramp's ends as it sizes the block's own lattice; the readout passes one Out end at a time.
/// `resolve` reads everything else -- a sheet's offset. Neither is ever read as nought when it
/// fails: a parameter that does not resolve is `Unresolved`, by name.
///
/// **`LatticePhase::Strut { radius }` is matched with no `..`, on purpose.** Item 92 adds a node
/// radius to that variant, and this is where its number must land for the compile and the
/// readout both, so the change must not compile until it is written here.
pub fn phase_of(
    phase: &fieldform_scene::LatticePhase,
    own: impl Fn(&fieldform_scene::ParamRef) -> Option<f64>,
    resolve: impl Fn(&fieldform_scene::ParamRef) -> Option<f64>,
) -> Result<Phase, Unresolved> {
    use fieldform_scene::LatticePhase;
    match phase {
        LatticePhase::Sheet { thickness, offset } => {
            let offset_mm = resolve(offset).ok_or(Unresolved { which: "offset" })?;
            Ok(Phase::Sheet {
                offset_mm,
                thickness_mm: own(thickness).ok_or(Unresolved { which: "wall" })?,
            })
        }
        LatticePhase::Solid { density } => Ok(Phase::Solid {
            density: own(density).ok_or(Unresolved { which: "density" })?,
        }),
        LatticePhase::Strut { radius } => Ok(Phase::Strut {
            radius_mm: own(radius).ok_or(Unresolved { which: "radius" })?,
        }),
    }
}

/// How a lattice's relative density was come by, so whatever quotes it can say so.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RelativeDensity {
    /// Through a table measured on this field: a sheet's wall, a beam's radius, or a solid's
    /// typed number read as the density it actually builds.
    Tabled(f64),
    /// From a closed form exact for this field: cubic struts, honeycomb walls.
    Exact(f64),
    /// A TPMS solid at or below its saddle: separate pieces, not one lattice. `built` is what
    /// the typed number fills; `joins_above` is the fill at the saddle, where it becomes one.
    Disconnected { built: f64, joins_above: f64 },
    /// None can be quoted, and the reason is said where the number would be.
    Unavailable(&'static str),
}

impl RelativeDensity {
    /// The fraction, where one is quoted for a lattice that is one piece.
    pub fn value(self) -> Option<f64> {
        match self {
            RelativeDensity::Tabled(d) | RelativeDensity::Exact(d) => Some(d),
            RelativeDensity::Disconnected { .. } | RelativeDensity::Unavailable(_) => None,
        }
    }
}

/// The relative density of a lattice of cubic cells `cell_mm` on a side -- for a prism, the
/// flat-to-flat width -- as `Lattice::distance` builds it. The caller decides the cells are
/// cubes: a map that stretches them leaves a density alone but not the cell every source
/// measured, and that is the readout's question, not this one's.
///
/// No sampling at the call: every arm is a closed form or a table measured once and held to the
/// field by a test, the way [`TpmsKind::density_isovalues`] is.
///
/// **The strut arms are written `Phase::Strut { radius_mm }`, with no `..`, on purpose.** Item
/// 92 adds a node radius to that variant; a filleted node adds material no strut table here
/// holds, so the change must not compile until someone decides it here. The decision made in
/// advance: a node radius of nought reads as before, and one above nought is `Unavailable`
/// ("node fillets add material the strut table does not hold") until item 88's sampler or a
/// table of its own can say.
pub fn uniform_density(cell: CellKind, phase: Phase, cell_mm: f64) -> RelativeDensity {
    use RelativeDensity::{Disconnected, Exact, Tabled, Unavailable};
    const NOT_POSITIVE: &str = "the number does not resolve to a positive value";
    const MISPAIRED: &str = "this cell does not take this phase";
    let positive = |x: f64| x.is_finite() && x > 0.0;
    if !positive(cell_mm) {
        return Unavailable(NOT_POSITIVE);
    }
    match cell {
        CellKind::Tpms(kind) => match phase {
            Phase::Solid { density } => {
                if !positive(density) {
                    return Unavailable(NOT_POSITIVE);
                }
                let built = kind.solid_density_built(density);
                // Compared on isovalues, so the answer is exact whatever the fill table says.
                let (floor, joins_above) = kind.solid_floor();
                if kind.isovalue_for_density(density) <= floor {
                    Disconnected { built, joins_above }
                } else {
                    Tabled(built)
                }
            }
            Phase::Sheet { thickness_mm, offset_mm } => {
                if !positive(thickness_mm) || !offset_mm.is_finite() {
                    return Unavailable(NOT_POSITIVE);
                }
                if offset_mm != 0.0 {
                    return Unavailable(concat!(
                        "an offset wall divides the cell unequally; ",
                        "the sources measured centred sheets"
                    ));
                }
                match kind.sheet_density(thickness_mm / cell_mm) {
                    Some(d) => Tabled(d),
                    None => Unavailable("past this cell's wall limit the walls have met"),
                }
            }
            Phase::Strut { radius_mm: _ } => Unavailable(MISPAIRED),
        },
        CellKind::Beam(beam) => match phase {
            Phase::Strut { radius_mm } => {
                if !positive(radius_mm) {
                    return Unavailable(NOT_POSITIVE);
                }
                // One sentence per cell, since the merge radius is the cell's and a reason is a
                // `&'static str`.
                let merged = match beam {
                    BeamKind::Cubic => "past a radius of half the cell the struts have met",
                    BeamKind::Bcc => "past a radius of 0.354 of the cell the struts have met",
                    BeamKind::Fcc => "past a radius of 0.289 of the cell the struts have met",
                    BeamKind::Octet => "past a radius of a quarter cell the struts have met",
                    BeamKind::Diamond => "past a radius of 0.217 of the cell the struts have met",
                };
                match (beam.strut_density(radius_mm / cell_mm), beam) {
                    (Some(d), BeamKind::Cubic) => Exact(d),
                    (Some(d), _) => Tabled(d),
                    (None, _) => Unavailable(merged),
                }
            }
            Phase::Sheet { .. } | Phase::Solid { .. } => Unavailable(MISPAIRED),
        },
        CellKind::Prism(prism) => match phase {
            // The field reads a prism's wall and nothing else, so an offset changes nothing
            // built and nothing here.
            Phase::Sheet { thickness_mm, .. } => {
                if !positive(thickness_mm) {
                    return Unavailable(NOT_POSITIVE);
                }
                match prism.wall_density(thickness_mm / cell_mm) {
                    Some(d) => Exact(d),
                    None => Unavailable("the walls fill the cell"),
                }
            }
            Phase::Solid { .. } | Phase::Strut { radius_mm: _ } => Unavailable(MISPAIRED),
        },
        CellKind::Stochastic { .. } => {
            Unavailable("a Voronoi cell's fill depends on its jitter and seed and is not tabled")
        }
    }
}

/// A TPMS wall, tiled through all of space.
///
/// Infinite by construction, which is exactly what [`crate::Extent`] exists to say: a lattice
/// on its own bounds nothing, and becomes a part by being intersected with one. `brick_pruning`
/// measured why that matters — a bare gyroid straddles zero in every region, and the same
/// gyroid confined to a part rejects 82% of them.
///
/// No frame of its own. A lattice is positioned by a `Node::Transform` wrapped around it, the
/// same as everything else, so the manipulator repositions one for free and there is no second
/// place for its orientation to be written down.
///
/// **Clone rather than Copy**, since a conformal map carries a shared surface. Cloning one is
/// an `Arc` bump and three scalars; it was a memcpy before and the difference has not shown up
/// anywhere that measures.
#[derive(Clone, Debug)]
pub struct Lattice {
    pub cell: CellKind,
    /// How the cells tile space. Cell *size* lives here rather than on the lattice, because it
    /// describes how the pattern tiles and not what the pattern is — a gyroid is a gyroid
    /// whatever size you make the cells. Drawing that line is what lets a cylindrical map carry
    /// different parameters from a rectangular one instead of being forced into a shared shape.
    pub map: LatticeMap,
    /// Which side of the surface is material, and the number that sizes it. See [`Phase`] for
    /// why the number lives in there rather than beside this field.
    pub phase: Phase,
}

impl Lattice {
    /// The same lattice with its phase's number replaced -- what a driven phase evaluates, once
    /// per sample, with the ramp's value there.
    ///
    /// **A copy rather than a second evaluation path**, so a graded lattice at a point is, by
    /// construction, the fixed lattice of that size at that point: the same isovalue, the same
    /// degenerate radius, the same bound. A second `distance` taking the number as an argument
    /// would be two transcriptions of the arithmetic below; the copy is three scalars and, for a
    /// conformal map, an `Arc` bump.
    ///
    /// Clamped to what the phase can mean: a wall or a radius is not negative, and a density is
    /// a fraction. A ramp is already clamped to its own output range; this is for the range
    /// somebody typed into it.
    pub(crate) fn sized(&self, sizing: f64) -> Lattice {
        let phase = match self.phase {
            Phase::Sheet { offset_mm, .. } => {
                Phase::Sheet { offset_mm, thickness_mm: sizing.max(0.0) }
            }
            Phase::Solid { .. } => Phase::Solid { density: sizing.clamp(0.0, 1.0) },
            Phase::Strut { .. } => Phase::Strut { radius_mm: sizing.max(0.0) },
        };
        Lattice { cell: self.cell, map: self.map.clone(), phase }
    }

    /// The same lattice with its cell's jitter and seed installed, for the two cells that have
    /// them and unchanged for the twelve that do not.
    ///
    /// The counterpart of [`Self::sized`] for the numbers that belong to the *cell* rather than
    /// to the phase, and it exists for the same reason: the compile has the document, so it
    /// resolves the jitter through `Document::parameter_value` and hands the number down.
    /// Clamped to nought and one here rather than trusted, because a seed leaving its own cell
    /// breaks every bound `voronoi_measure` rests on.
    pub(crate) fn with_scatter(&self, jitter: f64, seed: u32) -> Lattice {
        let cell = match self.cell {
            CellKind::Stochastic { kind, .. } => CellKind::Stochastic {
                kind,
                jitter: jitter.clamp(0.0, 1.0),
                seed,
            },
            other => other,
        };
        Lattice { cell, map: self.map.clone(), phase: self.phase }
    }

    /// Millimetres into units of `F`, at this point: the local frequencies against the cell's
    /// per-argument gradient, in quadrature. What a wall's thickness and its mid-surface offset
    /// are both converted through, so the two are nominal in exactly the same way.
    fn sheet_scale(&self, p: Point3<f64>) -> f64 {
        let w = self.map.local_frequency(p);
        let g = match self.cell {
            CellKind::Tpms(kind) => kind.argument_gradient(),
            // None of these reaches here: the per-argument split exists only to turn a wall
            // thickness into an isovalue, and a strut radius, a prism's wall and a Voronoi
            // wall are all lengths taken off something already measured in millimetres.
            CellKind::Beam(_) | CellKind::Prism(_) | CellKind::Stochastic { .. } => {
                Vector3::new(1.0, 1.0, 1.0)
            }
        };
        ((w.x * g.x).powi(2) + (w.y * g.y).powi(2) + (w.z * g.z).powi(2)).sqrt()
    }

    /// The value of `F` the wall is banded about: nought for a plain sheet, and the offset
    /// converted the way the thickness is for a walled one (backlog item 48). An offset past
    /// the cell's own extremes leaves nothing for the band to sit on and the lattice is empty,
    /// which is a statement about the numbers typed rather than an error.
    fn midsurface_at(&self, p: Point3<f64>) -> f64 {
        match self.phase {
            Phase::Sheet { offset_mm, .. } if offset_mm != 0.0 => offset_mm * self.sheet_scale(p),
            _ => 0.0,
        }
    }

    /// The threshold on `|F|` that produces the requested wall thickness **at this point**.
    ///
    /// Position-dependent, and it has to be evaluated per sample rather than hoisted. Under a
    /// map whose Jacobian is not a uniform scale, the frequency the cell is being driven at
    /// varies from place to place: a cylindrical map's circumferential term is `N/r`, twice as
    /// high at half the radius. A single hoisted isovalue would be right at one radius and
    /// wrong everywhere else, and wrong *more* toward the axis — geometry that looks entirely
    /// plausible and is the wrong shape, which is the failure D8 exists to name.
    ///
    /// The Jacobian is diagonal in the orthonormal local basis, so the frequencies combine in
    /// quadrature against the cell's per-argument gradient.
    fn isovalue_at(&self, p: Point3<f64>) -> f64 {
        match self.phase {
            Phase::Sheet { thickness_mm, .. } => thickness_mm * 0.5 * self.sheet_scale(p),
            // Constant, and the position is ignored on purpose: a density is a fraction of a
            // cell rather than a length, so squeezing the cells leaves it alone. Everything
            // above about the Jacobian is a statement about converting millimetres, and there
            // are none to convert here.
            Phase::Solid { density } => match self.cell {
                CellKind::Tpms(kind) => kind.isovalue_for_density(density),
                // Refused when the document is compiled, so this cannot arrive from the app.
                CellKind::Beam(_) | CellKind::Prism(_) | CellKind::Stochastic { .. } => 0.0,
            },
            // A beam has no isovalue at all: its radius is subtracted from a distance, in
            // millimetres, rather than converted into a threshold on some other quantity.
            Phase::Strut { .. } => 0.0,
        }
    }

    /// The smallest material feature this lattice has, in millimetres.
    ///
    /// Asked by [`LatticeMap::degenerate_radius`] and [`Self::bound_near`], both of which want to
    /// know at what scale the tiling stops resolving. For a sheet it is the wall thickness and
    /// there is nothing to derive.
    ///
    /// **For a solid it is an estimate, and the spec's open question.** The material is a fixed
    /// *fraction* of each cell, so unlike a wall it never reaches a fixed width — it shrinks
    /// with the cells. `min(d, 1−d)` of the smallest fixed cell extent is what is used: the
    /// `min` is because the thin feature is the strut at low density and the *void* at high
    /// density, and both matter for the same reason.
    ///
    /// Both directions of error are bounded and neither is a correctness problem, which is why
    /// an estimate is allowed to stand here:
    ///
    /// - **Too small** and the degenerate radius sits nearer the axis than it should, so the
    ///   real formula is evaluated in cells too fine to resolve. That costs a tracer steps; it
    ///   cannot make it step through anything, because the bound it divides by only goes up.
    /// - **Too large** and a slightly wider hub reads as solid. That is a shape claim rather
    ///   than a safety one, and it is sound because [`ScalarField::distance`] returns solid
    ///   inside the same radius — the two agree by construction.
    ///
    /// Against a beam-like `d ∝ (w/L)²`, this reads low at small densities and high near a half.
    /// Measuring the real neck width per cell is what would replace it.
    pub(crate) fn feature_size_mm(&self) -> f64 {
        let (banding, sizing) = match self.phase {
            Phase::Sheet { thickness_mm, .. } => (Banding::Sheet, thickness_mm),
            Phase::Solid { density } => (Banding::Solid, density),
            Phase::Strut { radius_mm } => (Banding::Strut, radius_mm),
        };
        feature_size_mm(banding, sizing, self.map.smallest_cell_extent())
    }

    /// The lattice cut down to the shell it is confined to, if it is confined to one.
    ///
    /// **An intersection, written where the map is rather than as a boolean above it.** A
    /// conformal lattice repeats away from its surface forever — the third argument is
    /// `2π·d/depth` and the cell function is `2π`-periodic — so without this the block is a
    /// tiling and the part is whatever it is intersected with. A box gives a box, which is
    /// exactly what the first user test produced.
    ///
    /// `max` of a lower bound and an exact distance is still a lower bound, so the contract
    /// holds: the shell's own term is the true distance to the band's face.
    fn within_shell(&self, p: Point3<f64>, value: f64) -> f64 {
        let LatticeMap::Conformal { surface, reach: Some(reach), .. } = &self.map else {
            return value;
        };
        value.max(surface.at(p).offset.abs() - reach)
    }

    /// The bound on `|∇F|` per millimetre travelled, which is what the raw value is divided by.
    ///
    /// **Local, and made safe by asking twice.** A single constant has to be taken where the
    /// cells collapse, since that is where the frequency is highest — and out at the rim of a
    /// real part that understated the available room by up to 332x, which costs a ray that
    /// many more steps than it needs and shows as speckle wherever one runs out.
    ///
    /// The two-step construction is what makes a local bound sound. A first estimate `d0` uses
    /// the bound where the point actually is. A ray may travel `d0` inward, so the second pass
    /// takes the bound at `r - d0` — where the frequency is *higher*, so the bound is larger
    /// and the answer smaller. Every point within the returned distance therefore has a
    /// frequency no greater than the one divided by, which is exactly the property a lower
    /// bound needs.
    ///
    /// Never nearer than the degenerate radius, since inside that the field returns solid and
    /// there is nothing to step through.
    fn bound_near(&self, p: Point3<f64>, numerator: f64) -> f64 {
        let g = self.cell.gradient_bound();
        let r = self.map.radius_of(p);
        if !r.is_finite() {
            // A rectangular map has no radius and no variation; one evaluation is the answer.
            return self.map.frequency_at_radius(r) * g;
        }
        let floor = self
            .map
            .degenerate_radius(self.feature_size_mm())
            .unwrap_or(0.0);
        let first = self.map.frequency_at_radius(r) * g;
        let reach = (numerator / first).abs();
        self.map
            .frequency_at_radius((r - reach).max(floor).max(1e-9))
            * g
    }
}

impl ScalarField for Lattice {
    /// Negative inside the wall, positive outside, and never further from zero than the true
    /// distance — the contract a lower-bound field owes.
    fn distance(&self, p: Point3<f64>) -> f64 {
        // Inside the radius where the tiling has collapsed, the walls have merged and the
        // answer is solid — which is what the arithmetic is converging to anyway, but it
        // converges through arbitrarily large isovalues, so it is stated rather than computed.
        if let Some(r_min) = self.map.degenerate_radius(self.feature_size_mm()) {
            // Distance from the axis for a cylindrical map, from the centre for a spherical
            // one: each measures where *its own* cells have closed up.
            let r = match self.map {
                LatticeMap::Spherical { .. } => p.coords.norm(),
                _ => (p.x * p.x + p.y * p.y).sqrt(),
            };
            if r < r_min {
                // As deep as the region is wide, so the value still grows with depth and a
                // tracer approaching from outside is never told to step further than it may.
                return -(r_min - r).min(self.feature_size_mm() * 0.5);
            }
        }
        if let CellKind::Prism(prism) = self.cell {
            // **In millimetres from the start, with no cell coordinates at all.** The walls are
            // a planar set carried through the third axis, so the distance to them is a 2D
            // distance and is exact; half the wall comes off it. Only a rectangular map is a
            // uniform scale, which is what that needs, and the compile refuses the rest by
            // name -- so anything else arriving here is empty rather than quietly stretched.
            let LatticeMap::Rectangular { cell } = &self.map else {
                return f64::INFINITY;
            };
            let thickness_mm = match self.phase {
                Phase::Sheet { thickness_mm, .. } => thickness_mm,
                _ => 0.0,
            };
            // The flat-to-flat width is the period along the second axis -- see
            // `PrismKind::natural_aspect`.
            return prism.wall_distance(p.x, p.y, cell.y) - thickness_mm * 0.5;
        }
        let args = self.map.to_cell_coords(p);
        if let CellKind::Stochastic { kind, jitter, seed } = self.cell {
            // A difference of order statistics over the seed grid, in cell units, converted the
            // way a beam's distance is: multiplied back into argument units and divided by the
            // largest local frequency, in the two steps `bound_near` takes so that a curved map
            // stays sound. What is subtracted is a length in millimetres either way -- half a
            // wall for the foam, a whole radius for the beam -- so there is no isovalue here.
            let (measure, _) =
                voronoi_measure(kind, args / std::f64::consts::TAU, jitter, seed, self.map.cell_period());
            let size = match self.phase {
                // Half each side, and the number typed rather than a calibrated one: the measure
                // understates, so the wall comes out at least as thick as asked and sometimes
                // thicker, which is the safe direction for a minimum wall.
                // `StochasticKind::wall_spread` carries the measurement and says why no
                // constant is divided out.
                Phase::Sheet { thickness_mm, .. } => thickness_mm * 0.5,
                Phase::Strut { radius_mm } => radius_mm,
                // Refused when the document is compiled: a density needs a table this family
                // does not have. `voronoi-lattice-spec.md` §4.5.
                Phase::Solid { .. } => 0.0,
            };
            let raw = measure * std::f64::consts::TAU;
            return self.within_shell(p, raw / self.bound_near(p, raw) - size);
        }
        if let CellKind::Beam(beam) = self.cell {
            // A different arithmetic, not a different isovalue. The struts are a set of axes in
            // cell space and the distance to them is already a distance — so it is converted
            // into millimetres and the radius is taken off it, with nothing to band and nothing
            // to average through.
            //
            // Dividing by the *largest* local frequency is what makes the conversion safe under
            // a map that scales the axes differently: a cell-space step of `Δ` is at least
            // `Δ / max(ω)` millimetres, with equality when the three agree. `bound_near` does
            // that division, and does it in two steps so a curved map stays sound; the cell's
            // own gradient bound is 1, so what is left is exactly the frequency.
            let radius_mm = match self.phase {
                Phase::Strut { radius_mm } => radius_mm,
                // Refused when the document is compiled. Zero here rather than a guess, so a
                // lattice that somehow arrives mispaired is visibly empty rather than quietly
                // the wrong size.
                _ => 0.0,
            };
            let to_axis = beam.axis_distance(args / std::f64::consts::TAU)
                * std::f64::consts::TAU;
            return self
                .within_shell(p, to_axis / self.bound_near(p, to_axis) - radius_mm);
        }
        let CellKind::Tpms(kind) = self.cell else {
            unreachable!("the beam arm returned above");
        };
        // The one character the phase is about. A sheet is the band around the surface, so it
        // is `|F|` that has to be small; a solid is one of the two labyrinths, so it is `F`
        // itself, and the other labyrinth is what the sign then excludes.
        let value = match self.phase.banding() {
            // About the wall's own middle, which is the cell's surface unless it was offset.
            Banding::Sheet => (kind.f(args) - self.midsurface_at(p)).abs(),
            Banding::Solid | Banding::Strut => kind.f(args),
        };
        let raw = value - self.isovalue_at(p);
        self.within_shell(p, raw / self.bound_near(p, raw))
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        if let CellKind::Prism(prism) = self.cell {
            // Away from the nearest wall, in the plane: a prism has no gradient along its axis.
            let LatticeMap::Rectangular { cell } = &self.map else {
                return None;
            };
            let h = 1e-5;
            let d = |dx: f64, dy: f64| prism.wall_distance(p.x + dx, p.y + dy, cell.y);
            return Vector3::new(d(h, 0.0) - d(-h, 0.0), d(0.0, h) - d(0.0, -h), 0.0)
                .try_normalize(1e-12);
        }
        let args = self.map.to_cell_coords(p);
        // The wall is the region where |F| is small, so its outward direction is the gradient
        // of |F| — the gradient of F, turned around wherever F is negative. Taken in *argument*
        // space and then carried back through the map's own scaling, which is what makes the
        // direction right under a map that stretches one axis differently from another.
        //
        // **A solid does not get the flip**, and this is the one place the phase can be wrong
        // without looking wrong. A wall has two sides, which is what the sign is tracking; a
        // labyrinth has one, and its outward direction is ∇F unturned. Flipping it anyway
        // shades half the surface inside-out, which reads as a lighting bug a long way from
        // here.
        let g = match self.cell {
            // The gradient of a distance to a strut set, taken by central difference in cell
            // space. Analytic would mean differentiating a `min` over segments, which is the
            // same expression with the branch written out and one more place for a sign to be
            // wrong — and this is not on the marching path, which reads the G-buffer.
            CellKind::Beam(beam) => {
                let h = 1e-5;
                let d = |o: Vector3<f64>| beam.axis_distance(args / std::f64::consts::TAU + o);
                Vector3::new(
                    d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
                    d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
                    d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
                ) / (2.0 * h)
            }
            CellKind::Tpms(kind) => match self.phase.banding() {
                Banding::Sheet => {
                    kind.gradient(args) * (kind.f(args) - self.midsurface_at(p)).signum()
                }
                Banding::Solid | Banding::Strut => kind.gradient(args),
            },
            // Analytic rather than a central difference, and that is a cost decision as much
            // as an accuracy one: a difference would be six more ring passes, and the pass
            // already returns both seeds the gradient is made of. In cell units, like the beam
            // arm above -- the factor of `2*pi` between that and argument space is the same on
            // all three axes, so it drops out of the direction.
            CellKind::Stochastic { kind, jitter, seed } => {
                voronoi_measure(
                    kind,
                    args / std::f64::consts::TAU,
                    jitter,
                    seed,
                    self.map.cell_period(),
                )
                .1
            }
            CellKind::Prism(_) => unreachable!("a prism answered above"),
        };
        // **The shell's own face, where that is what the surface is.** `within_shell` takes
        // the larger of the two, so wherever the band is the larger the outward direction is
        // the surface's normal and not the cell's — and shading the cell's there would light a
        // flat face as though it were a gyroid wall.
        if let LatticeMap::Conformal { surface, reach: Some(reach), .. } = &self.map {
            let hit = surface.at(p);
            let band = hit.offset.abs() - reach;
            let cell = {
                let raw = match self.cell {
                    CellKind::Tpms(kind) => match self.phase.banding() {
                        Banding::Sheet => (kind.f(args) - self.midsurface_at(p)).abs(),
                        Banding::Solid | Banding::Strut => kind.f(args),
                    },
                    CellKind::Beam(beam) => {
                        beam.axis_distance(args / std::f64::consts::TAU) * std::f64::consts::TAU
                    }
                    // Refused when the document is compiled -- a conformal map gives each
                    // face its own coordinates and the neighbour search would stop at every
                    // face edge -- so this is only reachable from a hand-built lattice. The
                    // honest value rather than a panic, since the arithmetic is to hand.
                    CellKind::Stochastic { kind, jitter, seed } => {
                        voronoi_measure(
                            kind,
                            args / std::f64::consts::TAU,
                            jitter,
                            seed,
                            self.map.cell_period(),
                        )
                        .0 * std::f64::consts::TAU
                    }
                    CellKind::Prism(_) => unreachable!("a prism answered above"),
                } - self.isovalue_at(p);
                raw / self.bound_near(p, raw)
            };
            if band > cell {
                // **Away from the closest point, which is the band's own gradient.** Over the
                // surface that is the normal, turned to the point's side. Around the rim of an
                // open surface the band is a rounded cap about the edge and its outward
                // direction is radial -- the surface's normal there is up to a right angle off,
                // which `gpu_matches_cpu` measured at 85 degrees for one hit in a hundred.
                return Some(
                    (p - hit.point)
                        .try_normalize(1e-9)
                        .unwrap_or(hit.normal * hit.offset.signum()),
                );
            }
        }
        let w = self.map.local_frequency(p);
        match &self.map {
            // **Exact, and the chain rule is all of it.** `grad F = sum_k (dF/darg_k) grad arg_k`,
            // and a surface map knows all three of its argument gradients: `2*pi*N` times the
            // parametric ones, and `2*pi/depth` along the normal. No basis to rotate out of,
            // because the surface hands over world-space vectors already.
            LatticeMap::Conformal {
                surface,
                cells_per_face,
                depth,
                ..
            } => {
                let hit = surface.at(p);
                let n = std::f64::consts::TAU * *cells_per_face as f64;
                (hit.ds * (g.x * n) + hit.dt * (g.y * n)
                    + hit.normal * (g.z * std::f64::consts::TAU / depth.max(1e-9)))
                .try_normalize(1e-12)
            }
            LatticeMap::Rectangular { .. } => {
                Vector3::new(g.x * w.x, g.y * w.y, g.z * w.z).try_normalize(1e-12)
            }
            LatticeMap::Spherical { .. } => {
                // Out of (radial, azimuthal, polar) and back into the frame's basis. All three
                // directions depend on where the point is, which is the same reason the
                // cylindrical arm exists and one more axis of it.
                let r = p.coords.norm();
                let rho = (p.x * p.x + p.y * p.y).sqrt();
                if r < 1e-9 || rho < 1e-9 {
                    return None;
                }
                let (c, s_) = (p.x / rho, p.y / rho);
                let (sin_polar, cos_polar) = (rho / r, p.z / r);
                let radial = Vector3::new(c * sin_polar, s_ * sin_polar, cos_polar) * (g.x * w.x);
                let azimuthal = Vector3::new(-s_, c, 0.0) * (g.y * w.y);
                let polar = Vector3::new(c * cos_polar, s_ * cos_polar, -sin_polar) * (g.z * w.z);
                (radial + azimuthal + polar).try_normalize(1e-12)
            }
            LatticeMap::Cylindrical { .. } => {
                // The gradient arrives in the cylindrical basis and has to be turned back into
                // the frame's. Radial and circumferential directions depend on where the point
                // is, which is the whole difference from the rectangular case.
                let rho = (p.x * p.x + p.y * p.y).sqrt();
                if rho < 1e-9 {
                    return None;
                }
                let (c, s) = (p.x / rho, p.y / rho);
                let radial = Vector3::new(c, s, 0.0) * (g.x * w.x);
                let circumferential = Vector3::new(-s, c, 0.0) * (g.y * w.y);
                let axial = Vector3::new(0.0, 0.0, g.z * w.z);
                (radial + circumferential + axial).try_normalize(1e-12)
            }
        }
    }

    /// The whole reason this type is careful. See the module docs.
    ///
    /// **A beam lattice on an isotropic rectangular map is the exception, and it is a real one.**
    /// It is a union of capsules, and a capsule is an exact signed distance function; the minimum
    /// of exact distances is exact, with the same crease caveat every CSG union here already
    /// carries. So it may be blended exactly, and `approximate_blends` will not name it — which
    /// is the first time that has been true of any lattice.
    ///
    /// The two conditions are both load-bearing. **Curved maps** scale the axes differently, so
    /// the conversion out of cell space divides by the largest frequency and understates; that is
    /// a bound, not a distance. **Unequal cell extents** do the same thing on a rectangular map,
    /// which the scene cannot currently express — it writes one cell size into all three — but
    /// this type can, so the check is on the extents rather than on the map's name.
    fn normalization(&self) -> Normalization {
        match (self.cell, &self.map) {
            (CellKind::Beam(_), LatticeMap::Rectangular { cell })
                if (cell.x - cell.y).abs() < 1e-12 && (cell.y - cell.z).abs() < 1e-12 =>
            {
                Normalization::Distance
            }
            // A prism measures in millimetres and never passes through cell space, so its
            // extents need not be equal -- they are not, by `natural_aspect` -- only uniform.
            (CellKind::Prism(_), LatticeMap::Rectangular { .. }) => Normalization::Distance,
            // **A Voronoi measure understates under every map**, which is the one place this
            // family is simpler than a beam rather than harder: there is no configuration that
            // makes it exact, so there is no condition to check. `voronoi-lattice-spec.md` §4.4.
            (CellKind::Stochastic { .. }, _) => Normalization::LowerBound,
            _ => Normalization::LowerBound,
        }
    }

    /// `1.0`: the value was already divided by a bound on the gradient, so it moves at most one
    /// unit per unit travelled. Understating this would let a tracer step through a wall.
    fn lipschitz(&self) -> f64 {
        1.0
    }

    /// The wall, strut or neck — see [`Lattice::feature_size_mm`], which the panel's readout
    /// reads too, so what the mesher resolves and what the panel reports are one number.
    fn feature_size(&self) -> Option<f64> {
        Some(self.feature_size_mm()).filter(|f| f.is_finite() && *f > 0.0)
    }
}

#[cfg(test)]
mod tests;
