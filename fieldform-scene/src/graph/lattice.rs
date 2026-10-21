// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a lattice is made of: its cell, which side of it is material, and how cells tile space.

use super::ParamRef;
use serde::{Deserialize, Serialize};
#[cfg(doc)] use super::Node;

/// Which repeating cell a lattice is built from.
///
/// Half of the pair that makes a lattice: *what* the cell is, kept separate from *how* cells
/// tile space ([`CellMap`]). Another CAD package's third-generation latticing settled on that
/// split and it is
/// the reason a gyroid can become cylindrical or conformal without a new node kind each time —
/// swapping either input is a swap of one field, not a different tool. Adopted here before the
/// first lattice shipped, because it costs a field now and a format migration later.
///
/// Today one family, six surfaces. Beam and Voronoi cells are variants of this enum when they
/// come, which is what D3 anticipates by preserving and refusing unknown enum values.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticeCell {
    Gyroid,
    SchwarzP,
    SchwarzD,
    /// Schoen I-WP. Thicker nodes than the gyroid, and the one people reach for when a lattice
    /// has to take load in every direction rather than look even.
    SchoenIwp,
    /// High surface area for its volume, which is why it turns up in heat exchangers — and the
    /// first cell here whose two labyrinths are *not* the same size, which is what the solid
    /// phase's per-cell density table exists for.
    Neovius,
    /// Cyclically symmetric rather than fully, like the gyroid.
    FischerKochS,
    /// Beams along the three axes through each lattice point — the simplest wireframe there is,
    /// and stiff along its own axes and floppy across them.
    Cubic,
    /// Body-centred cubic: beams from each corner of the cell to its centre. Near-isotropic,
    /// which is what makes it the usual first choice.
    Bcc,
    /// Face-centred cubic: the cube's edges plus both diagonals on every face.
    Fcc,
    /// The octet truss — an octahedron on the six face centres, each of its faces capped by a
    /// tetrahedron reaching a cube corner. The stiffest of these per unit mass, and the reason
    /// the family is in aerospace.
    Octet,
    /// Diamond cubic: every site bonded tetrahedrally to four others. The only cell here with no
    /// mirror symmetry about the cell's own mid-planes, which costs it four times the shader
    /// work of the others — see `beam-lattice-spec.md` §4.
    Diamond,
    /// A hexagonal honeycomb: a planar tiling carried straight through the map's third axis, for
    /// coring and stiffening a panel rather than filling a volume. Backlog item 49.
    ///
    /// **A prism, which is the first cell here that is constant along an axis**, and the first
    /// whose own shape is not square: a row of regular hexagons repeats every flat-to-flat
    /// width one way and every `sqrt(3)` of them the other. The cell says so (its natural
    /// aspect) and the map's extents follow, which is the contract change item 49 named. The
    /// cell size typed is the hexagon's flat-to-flat width, and its wall is a distance, so the
    /// thickness typed is the thickness built -- exact, like a beam's radius.
    Honeycomb,
    /// A foam: the walls between the Voronoi cells of a jittered grid of seeds.
    /// `voronoi-lattice-spec.md`, Track A.
    ///
    /// **The first cells that carry numbers of their own**, which is why they are the first
    /// variants with fields. `jitter` runs nought to one: at nought the seeds sit at cell
    /// centres and the foam is a grid of boxes, at one a seed may sit anywhere in its own cell
    /// and never leaves it, which is the property the whole evaluation rests on. `seed` is
    /// stored rather than a parameter, because re-rolling until it looks right is the workflow
    /// and an arrangement has to be a function of what is saved; ramping it would mean nothing.
    ///
    /// Procedural, so there is no point list and no topology: infinite, memoryless and cheap
    /// for the shader, and a lower bound rather than a distance, so its walls come out
    /// somewhat thicker than typed and a fillet against it is marked approximate. Another CAD
    /// package's own Voronoi is the explicit kind, which is Track B.
    VoronoiFoam { jitter: ParamRef, seed: u64 },
    /// Beams along the edges where three of those same cells meet.
    VoronoiBeam { jitter: ParamRef, seed: u64 },
}

impl LatticeCell {
    /// Whether this cell is beams rather than a triply periodic minimal surface.
    ///
    /// The two families answer the same questions and answer them differently — a beam has a
    /// strut radius where a surface has a wall thickness, and is an exact distance where a
    /// surface is a bound. Asked here rather than by listing the beam cells at each call site,
    /// which is three places that would drift apart.
    pub fn is_beam(self) -> bool {
        match self {
            LatticeCell::Cubic
            | LatticeCell::Bcc
            | LatticeCell::Fcc
            | LatticeCell::Octet
            | LatticeCell::Diamond => true,
            LatticeCell::Gyroid
            | LatticeCell::SchwarzP
            | LatticeCell::SchwarzD
            | LatticeCell::SchoenIwp
            | LatticeCell::Neovius
            | LatticeCell::FischerKochS
            | LatticeCell::Honeycomb
            | LatticeCell::VoronoiFoam { .. }
            | LatticeCell::VoronoiBeam { .. } => false,
        }
    }

    /// A foam of this jitter and seed. Three quarters jittered by default: irregular enough to
    /// read as organic, short of where two seeds start landing on top of each other.
    pub fn voronoi_foam(seed: u64) -> Self {
        LatticeCell::VoronoiFoam { jitter: ParamRef::Literal(0.75), seed }
    }

    pub fn voronoi_beam(seed: u64) -> Self {
        LatticeCell::VoronoiBeam { jitter: ParamRef::Literal(0.75), seed }
    }

    /// Whether this cell is built from a jittered seed grid, and so has a jitter and a seed.
    pub fn is_stochastic(self) -> bool {
        matches!(self, LatticeCell::VoronoiFoam { .. } | LatticeCell::VoronoiBeam { .. })
    }

    /// The jitter, for the two cells that have one. An `Option` so `Node::param_refs` can
    /// extend with it and every other cell contributes nothing -- the arrangement a sheet's
    /// offset already has.
    pub fn jitter(&self) -> Option<&ParamRef> {
        match self {
            LatticeCell::VoronoiFoam { jitter, .. } | LatticeCell::VoronoiBeam { jitter, .. } => {
                Some(jitter)
            }
            _ => None,
        }
    }

    pub fn jitter_mut(&mut self) -> Option<&mut ParamRef> {
        match self {
            LatticeCell::VoronoiFoam { jitter, .. } | LatticeCell::VoronoiBeam { jitter, .. } => {
                Some(jitter)
            }
            _ => None,
        }
    }

    pub fn seed(&self) -> Option<u64> {
        match self {
            LatticeCell::VoronoiFoam { seed, .. } | LatticeCell::VoronoiBeam { seed, .. } => {
                Some(*seed)
            }
            _ => None,
        }
    }

    pub fn seed_mut(&mut self) -> Option<&mut u64> {
        match self {
            LatticeCell::VoronoiFoam { seed, .. } | LatticeCell::VoronoiBeam { seed, .. } => {
                Some(seed)
            }
            _ => None,
        }
    }

    /// The name this cell is written under. One string beside flat fields, as the map and the
    /// phase are written, rather than serde's nested object for a variant with fields.
    pub fn name(&self) -> &'static str {
        match self {
            LatticeCell::Gyroid => "gyroid",
            LatticeCell::SchwarzP => "schwarz_p",
            LatticeCell::SchwarzD => "schwarz_d",
            LatticeCell::SchoenIwp => "schoen_iwp",
            LatticeCell::Neovius => "neovius",
            LatticeCell::FischerKochS => "fischer_koch_s",
            LatticeCell::Cubic => "cubic",
            LatticeCell::Bcc => "bcc",
            LatticeCell::Fcc => "fcc",
            LatticeCell::Octet => "octet",
            LatticeCell::Diamond => "diamond",
            LatticeCell::Honeycomb => "honeycomb",
            LatticeCell::VoronoiFoam { .. } => "voronoi_foam",
            LatticeCell::VoronoiBeam { .. } => "voronoi_beam",
        }
    }

    /// The cell a name means, given the jitter and seed read beside it -- ignored by every cell
    /// that has neither.
    pub fn from_name(name: &str, jitter: ParamRef, seed: u64) -> Option<Self> {
        Some(match name {
            "gyroid" => LatticeCell::Gyroid,
            "schwarz_p" => LatticeCell::SchwarzP,
            "schwarz_d" => LatticeCell::SchwarzD,
            "schoen_iwp" => LatticeCell::SchoenIwp,
            "neovius" => LatticeCell::Neovius,
            "fischer_koch_s" => LatticeCell::FischerKochS,
            "cubic" => LatticeCell::Cubic,
            "bcc" => LatticeCell::Bcc,
            "fcc" => LatticeCell::Fcc,
            "octet" => LatticeCell::Octet,
            "diamond" => LatticeCell::Diamond,
            "honeycomb" => LatticeCell::Honeycomb,
            "voronoi_foam" => LatticeCell::VoronoiFoam { jitter, seed },
            "voronoi_beam" => LatticeCell::VoronoiBeam { jitter, seed },
            _ => return None,
        })
    }

    /// Whether this cell is a planar tiling carried through the third axis. Like a beam it is an
    /// exact distance; unlike one it has a wall rather than a strut.
    pub fn is_prism(self) -> bool {
        matches!(self, LatticeCell::Honeycomb)
    }
}

/// Which side of a cell's surface is material.
///
/// A minimal surface divides space into two interpenetrating labyrinths with the surface between
/// them. A **sheet** is that surface given thickness — the wall. A **solid** is one of the two
/// labyrinths, filled. Same cell, same map, same frame; one character of arithmetic apart, and
/// the stiffer of the two in compression at equal mass is the solid.
///
/// Which ones a given cell can give is the cell's answer, not this enum's — `TpmsKind::
/// supported_bandings` in `fieldform-field`. Every surface here gives both; a beam cell, when
/// one arrives, is struts and gives only [`Banding::Solid`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Banding {
    Sheet,
    Solid,
    /// Material within a radius of a wireframe — a beam cell, which has no two sides to wall
    /// and no labyrinth to fill, so neither of the others describes it.
    Strut,
}

/// A banding together with the parameter that sizes it.
///
/// **The two parameters are different quantities, which is why this is an enum and not a flag
/// beside a thickness.** A sheet has a wall thickness, a length in millimetres. A solid has a
/// *relative density*, a dimensionless fraction of each cell's volume. Neither number means
/// anything under the other banding, and holding one live field beside one dormant field is the
/// arrangement where the dormant one eventually gets read.
///
/// Switching banding therefore does not carry the number across, and cannot: there is no
/// conversion between 2 mm and 40%.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LatticePhase {
    Sheet {
        /// Nominal wall thickness. Nominal because the conversion from millimetres to an
        /// isovalue goes through an average gradient — see `TpmsKind::surface_gradient`.
        thickness: ParamRef,
        /// How far the wall's middle sits off the cell's own surface, in millimetres, signed
        /// toward the side where the cell's function is positive. Backlog item 48.
        ///
        /// **What makes the two passages differ.** A sheet is `|F| <= iso`, a band centred on
        /// the surface, so the two labyrinths it separates are always the same size. Banding
        /// about `F = c` instead lets one be larger than the other -- two fluids in a heat
        /// exchanger, which is what other CAD software calls a walled TPMS. Nominal in the same way
        /// and for
        /// the same reason the thickness is: it converts through the same average gradient.
        ///
        /// Zero is a plain sheet and is what every sheet written before this existed is, so it
        /// is not written to a file at all; [`LatticePhase::sheet`] builds one.
        offset: ParamRef,
    },
    Solid {
        /// Fraction of each cell that is material, `0..=1`. Approximate: it is read through a
        /// measured per-cell table, and `TpmsKind::isovalue_for_density` carries what that
        /// costs — under one percentage point, in absolute terms, from 0.1 to 0.9, and far more
        /// relative to a number typed below 0.1 (a typed 0.05 builds 0.026 to 0.043).
        /// `TpmsKind::solid_density_built` is what each typed number actually fills.
        density: ParamRef,
    },
    Strut {
        /// Radius of the beams, in millimetres — **not** a diameter, which is the other
        /// defensible reading and a factor of two away.
        ///
        /// The only lattice number in this file that is exact. A wall thickness is converted
        /// through an average gradient and a density through a measured table; a strut radius is
        /// subtracted from a distance, so what is asked for is what is built.
        radius: ParamRef,
    },
}

impl LatticePhase {
    /// A plain sheet: a wall of this thickness, centred on the cell's own surface.
    pub fn sheet(thickness: ParamRef) -> Self {
        LatticePhase::Sheet { thickness, offset: ParamRef::Literal(0.0) }
    }

    /// A sheet's mid-surface offset, which no other phase has.
    ///
    /// An `Option` rather than a second `param`, so `Node::param_refs` can extend with it and a
    /// solid or a strut simply contributes nothing.
    pub fn offset(&self) -> Option<&ParamRef> {
        match self {
            LatticePhase::Sheet { offset, .. } => Some(offset),
            _ => None,
        }
    }

    /// Which side is material, without the number that sizes it.
    pub fn banding(&self) -> Banding {
        match self {
            LatticePhase::Sheet { .. } => Banding::Sheet,
            LatticePhase::Solid { .. } => Banding::Solid,
            LatticePhase::Strut { .. } => Banding::Strut,
        }
    }

    /// The one number, whichever it is.
    ///
    /// Here rather than at each call site for the reason D9 gives: a node that carries a phase
    /// gets its parameter wired into `param_refs` without anyone remembering to, which is the
    /// arrangement `Frame` already has.
    pub fn param(&self) -> &ParamRef {
        match self {
            LatticePhase::Sheet { thickness, .. } => thickness,
            LatticePhase::Solid { density } => density,
            LatticePhase::Strut { radius } => radius,
        }
    }

    pub fn param_mut(&mut self) -> &mut ParamRef {
        match self {
            LatticePhase::Sheet { thickness, .. } => thickness,
            LatticePhase::Solid { density } => density,
            LatticePhase::Strut { radius } => radius,
        }
    }
}

/// How a lattice's cells tile space.
///
/// The other half of the pair, and it carries the **cell size**. That is the load-bearing part
/// of the split and it is right on principle: a cell size describes how the pattern tiles, not
/// what the pattern is — a gyroid is a gyroid whatever size the cells are. Once that line is
/// drawn, a cylindrical map naturally carries different parameters from a rectangular one
/// (two lengths and a count, against one length) instead of being forced into a shared shape.
///
/// **Serialized as a string tag beside flat fields**, not as a nested object, because this
/// format's forward-compatibility machinery is built on exactly that: an unknown *value* of a
/// tagged field is preserved and refused, and so is an unknown *field*. A build that predates
/// cylindrical maps meets `cell_radius`, does not recognise it, and refuses the node with an
/// explanation — which is the behaviour D3 promises. Nesting the parameters inside `map` would
/// change that field's JSON *type*, and a type change is the one thing a reader cannot refuse
/// gracefully.
///
/// A rectangular lattice therefore serializes byte-identically to what it did before this
/// variant existed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CellMap {
    /// Cartesian tiling, on the lattice's own frame.
    Rectangular {
        /// One full period of the cell, in millimetres.
        cell: ParamRef,
    },
    /// Tiling in cylindrical coordinates about the lattice frame's own +Z axis, so the pattern
    /// wraps an axis rather than filling a box.
    Cylindrical {
        /// Cell length in the radial direction, in millimetres.
        cell_radius: ParamRef,
        /// Cell length along the axis, in millimetres.
        cell_height: ParamRef,
        /// How many cells go around the circumference.
        ///
        /// **A count, and a whole number, not a length.** Going once around takes the angle
        /// through `2π`, so the cell coordinate advances by `2πN` — and the cell formula is
        /// `2π`-periodic, so the two sides of the seam at `θ = ±π` agree *only if `N` is an
        /// integer*. A fractional count puts a discontinuity down the full height of the
        /// cylinder: walls not meeting walls, which reads as a modelling bug and is really a
        /// parameterization error.
        ///
        /// Typed as an integer rather than rounded inside the evaluator, so the number the user
        /// typed and the geometry they get cannot disagree without saying so.
        arc_count: u32,
    },
    /// Tiling that **follows a surface**: cells laid out in an open quad mesh's own
    /// parametrization rather than in a box drawn around it.
    ///
    /// The one map whose cell size is not typed in at all. `s` and `t` run nought to one across
    /// every face however large the face is, so **the mesh is the grading** — a finer quad gets
    /// finer cells, and a graded lattice falls out of the input instead of needing a field to
    /// drive it. Only the third argument is a length, because only it leaves the surface.
    ///
    /// Names a [`QuadMeshId`] and not a node, for the same reason [`Node::Cage`] and
    /// [`Node::Mesh`] name their geometry: a surface to follow is a document resource, not an
    /// input to compose with. It is deliberately not a `Cage` — a cage is closed by
    /// construction, and the panel or shell somebody wants reinforced is exactly the open
    /// surface a cage cannot be.
    Conformal {
        /// The surface to follow.
        surface: crate::QuadMeshId,
        /// How many cells across each face, in both parametric directions.
        ///
        /// **One count for the whole mesh, and whole for the reason `arc_count` is.** Two faces
        /// sharing an edge have to divide it the same number of times or their cells do not
        /// meet along it, which one global count makes automatic and per-face counts would not.
        cells_per_face: u32,
        /// One full period of the cell through the surface, in millimetres.
        depth: ParamRef,
        /// How thick a shell of the surface to keep, in millimetres — or `None` to tile
        /// outward forever.
        ///
        /// **Without this the block is a tiling and not a part.** The third cell argument is
        /// `2π·d/depth` and the cell function is `2π`-periodic, so the pattern at `d` and at
        /// `d + depth` is the same pattern: a conformal lattice repeats away from its surface
        /// exactly as a rectangular one repeats through space. Intersect that with a box and
        /// you get a box — which is what the first user test produced, and read as the surface
        /// not being followed at all.
        ///
        /// `None` is still worth having and is not the same as a very large number: it is what
        /// you want when the lattice is infilling a part that already has a shape and the
        /// surface is only there to orient the cells.
        ///
        /// A shell also keeps the lattice inside the region where its map is single-valued.
        /// Past the surface's own curvature radius a closest point is genuinely ambiguous —
        /// `fieldform-field/examples/conformal_agreement.rs` measures `s` and `t` a full 1.0
        /// apart there — so a thickness comfortably inside that radius is not a restriction,
        /// it is the domain.
        thickness: Option<ParamRef>,
    },
    /// Tiling in spherical coordinates about the lattice frame's origin, so the pattern wraps a
    /// centre rather than an axis.
    Spherical {
        /// Cell length outward from the centre, in millimetres.
        cell_radius: ParamRef,
        /// Cells around the equator. Whole, for the reason `arc_count` is.
        azimuth_count: u32,
        /// Cells from pole to pole. Whole for the same reason, in the other direction.
        polar_count: u32,
    },
}

impl CellMap {
    /// The tag this map writes, which is also the vocabulary a reader gates on.
    ///
    /// The reader's gate spells the vocabulary out in `serde`'s table of known values, so only
    /// the tests use this, to name a map in a failure, and only the tests compile it (decision
    /// 1.4). `pub(crate)` because those tests are `graph`'s, not this module's.
    #[cfg(test)]
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            CellMap::Rectangular { .. } => "rectangular",
            CellMap::Cylindrical { .. } => "cylindrical",
            CellMap::Spherical { .. } => "spherical",
            CellMap::Conformal { .. } => "conformal",
        }
    }

    /// The map's own numbers — its cell sizes, and a conformal map's depth and shell.
    ///
    /// **So a block that tiles carries its tiling's numbers in its own parameter list** (D9's
    /// "the shared type carries its own wiring"). `Node::Lattice` listed its phase, its jitter and
    /// its frame and bound the map with `..`, so a variable used only as a cell size was invisible
    /// to everything that walks a node's parameters: `parameter_is_used` called it unused, and the
    /// cleanup that drops unused variables could delete one a lattice still read, leaving the map
    /// naming a parameter that was gone. The counts stay `u32` and are not parameters, which is
    /// `arc_count`'s own ruling.
    ///
    /// Walked rather than listed, since `Node::for_each_input` asks it of every node on every edit.
    pub fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let (first, second) = match self {
            CellMap::Rectangular { cell } => (cell, None),
            CellMap::Cylindrical { cell_radius, cell_height, .. } => {
                (cell_radius, Some(cell_height))
            }
            CellMap::Spherical { cell_radius, .. } => (cell_radius, None),
            CellMap::Conformal { depth, thickness, .. } => (depth, thickness.as_ref()),
        };
        std::iter::once(first).chain(second)
    }

    /// The same, to be edited — in the same order, which [`Node::param_refs_mut`] relies on.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            CellMap::Rectangular { cell } => vec![cell],
            CellMap::Cylindrical { cell_radius, cell_height, .. } => vec![cell_radius, cell_height],
            CellMap::Spherical { cell_radius, .. } => vec![cell_radius],
            CellMap::Conformal { depth, thickness, .. } => {
                std::iter::once(depth).chain(thickness.iter_mut()).collect()
            }
        }
    }

    /// How many cells fit around a circle of radius `r_ref` at a desired cell size.
    ///
    /// The helper that goes the other way from [`CellMap::Cylindrical::arc_count`]: users think
    /// in millimetres and the domain is closed, so the count is quantized and the achieved size
    /// has to be reported back rather than assumed. One relationship, drivable from either end.
    pub fn arc_count_for(cell_mm: f64, r_ref: f64) -> u32 {
        let n = (std::f64::consts::TAU * r_ref / cell_mm.max(1e-9)).round();
        n.clamp(1.0, u32::MAX as f64) as u32
    }

    /// The circumferential cell size this map actually achieves at radius `r`.
    pub fn arc_cell_mm(&self, r: f64) -> Option<f64> {
        match self {
            CellMap::Rectangular { .. } => None,
            // Nothing closes up on a surface: `s` and `t` run nought to one across a face
            // whatever size it is, where an angular argument sweeps a fixed count through a
            // shrinking circle. There is no circumference here to report a cell size around.
            CellMap::Conformal { .. } => None,
            CellMap::Cylindrical { arc_count, .. } => {
                Some(std::f64::consts::TAU * r / (*arc_count).max(1) as f64)
            }
            // At the equator, which is where a spherical map's cells are largest.
            CellMap::Spherical { azimuth_count, .. } => {
                Some(std::f64::consts::TAU * r / (*azimuth_count).max(1) as f64)
            }
        }
    }
}
