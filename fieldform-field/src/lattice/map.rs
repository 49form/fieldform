// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How a lattice's cells tile space: each map's coordinates, and the frequencies it drives them at.

use nalgebra::{Point3, Vector3};
#[cfg(doc)] use super::Lattice;

/// How a lattice's cells tile space: the arithmetic behind [`fieldform_scene::CellMap`].
///
/// A cell map answers one question — *given a point, where is it inside the tiling pattern?*
/// Everything else about a lattice is the cell's business, which is what lets any cell drop
/// into any map without either knowing about the other.
///
/// **Clone rather than Copy**, because one of these carries a whole surface. The others are a
/// handful of scalars and always were; a conformal map holds an `Arc` to a tessellated mesh with
/// a spatial index over it, which is shared rather than duplicated and is why the payload's
/// shape is the thing `conformal-cell-map-spec.md` §10.6 asked to confirm.
#[derive(Clone, Debug)]
pub enum LatticeMap {
    /// Cartesian tiling, one period per cell along each axis.
    Rectangular { cell: Vector3<f64> },
    /// Tiling in cylindrical coordinates about the frame's own +Z axis, so the pattern wraps
    /// an axis rather than filling a box.
    ///
    /// `arc_count` is a count and not a length, and it has to be a whole number. The points at
    /// `θ = π⁻` and `θ = −π⁺` are the same place; their cell coordinates differ by `2πN`, and
    /// the cell formula is `2π`-periodic, so the two agree **only if `N` is an integer**. A
    /// fractional count puts a discontinuity down the full height of the cylinder — walls not
    /// meeting walls, which reads as a modelling bug and is a parameterization error.
    Cylindrical {
        cell_radius: f64,
        cell_height: f64,
        arc_count: u32,
    },
    /// Tiling in spherical coordinates about the frame's origin, so the pattern wraps a centre
    /// rather than an axis.
    ///
    /// Two counts, and both for the same reason `arc_count` is one: azimuth closes on itself
    /// after `2π`, and *polar angle* closes after `2π` as well once the pattern is continued
    /// through the pole. A fraction of either tears a seam — one running pole to pole, the
    /// other around the equator.
    Spherical {
        cell_radius: f64,
        /// Cells around the equator.
        azimuth_count: u32,
        /// Cells from pole to pole.
        polar_count: u32,
    },
    /// Tiling that **follows a surface**: cells laid out in the mesh's own parametrization
    /// rather than in a box drawn around it.
    ///
    /// The motivating case is reinforcement — stiffening a panel or a shell with a structure
    /// that follows its curvature, which is what a rectangular map cannot do at all and what
    /// another CAD package's surface-lattice blocks take an open quad mesh for.
    ///
    /// Three arguments from a world point, via the closest point on the limit surface:
    ///
    /// ```text
    /// u = 2π · N · s     within the face
    /// v = 2π · N · t
    /// w = 2π · d / depth   d signed by which side of the surface the point is on
    /// ```
    ///
    /// **Cells are per face, and that is not a limitation being apologised for.** There is no
    /// global `(u, v)` grid over a quad mesh — that is a seamless parametrization, which is a
    /// different and much larger feature (spec §6.5). Each face carries its own unit square and
    /// continuity across an edge comes from the meshing. One consequence is worth knowing: two
    /// faces sharing an edge must subdivide it the same number of times or the cells mismatch
    /// along it, which one global `N` makes automatic and per-face counts would not.
    ///
    /// **Cell size in millimetres therefore follows quad size**, so the mesh *is* the grading.
    /// Graded conformal latticing falls out of the input rather than needing a field to drive
    /// it.
    Conformal {
        /// The surface to follow, shared rather than copied — it carries a tessellation and a
        /// spatial index over it.
        surface: std::sync::Arc<crate::SurfaceMap>,
        /// How many cells across each face, in both parametric directions.
        ///
        /// **One count for the whole mesh**, which is what makes neighbouring faces agree at
        /// every shared edge without a global integer solve. Per-face counts are the deferred
        /// half of this (spec §6.2).
        cells_per_face: u32,
        /// One full period of the cell through the surface, in millimetres — the third
        /// argument's scale, and the only one measured in length rather than counted.
        depth: f64,
        /// Half the shell's thickness, if it is confined to one.
        ///
        /// **Half rather than whole, because that is what the arithmetic wants and the whole
        /// is what the user types.** The band is `|offset| <= reach`, so storing the half
        /// removes a division from the innermost loop of the field and from the shader; the
        /// panel halves it once, where there is a person to read the label.
        reach: Option<f64>,
    },
}

/// Compared by what it is, and a surface by *identity* rather than by its contents.
///
/// Two maps over the same tessellated mesh are the same map; comparing a hundred thousand
/// triangles to discover that would cost more than every other comparison in the crate put
/// together, and the only caller is a cache key.
impl PartialEq for LatticeMap {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LatticeMap::Rectangular { cell: a }, LatticeMap::Rectangular { cell: b }) => a == b,
            (
                LatticeMap::Cylindrical {
                    cell_radius: ar,
                    cell_height: ah,
                    arc_count: ac,
                },
                LatticeMap::Cylindrical {
                    cell_radius: br,
                    cell_height: bh,
                    arc_count: bc,
                },
            ) => (ar, ah, ac) == (br, bh, bc),
            (
                LatticeMap::Spherical {
                    cell_radius: ar,
                    azimuth_count: aa,
                    polar_count: ap,
                },
                LatticeMap::Spherical {
                    cell_radius: br,
                    azimuth_count: ba,
                    polar_count: bp,
                },
            ) => (ar, aa, ap) == (br, ba, bp),
            (
                LatticeMap::Conformal {
                    surface: a,
                    cells_per_face: an,
                    depth: ad,
                    reach: ah,
                },
                LatticeMap::Conformal {
                    surface: b,
                    cells_per_face: bn,
                    depth: bd,
                    reach: bh,
                },
            ) => std::sync::Arc::ptr_eq(a, b) && (an, ad, ah) == (bn, bd, bh),
            _ => false,
        }
    }
}

impl LatticeMap {
    /// Frame-local point to unit-cell arguments, each `2π`-periodic.
    ///
    /// The unit cell's formula is untouched by which map is in front of it: this is a pure
    /// change of coordinates, which is the whole reason the split works.
    pub(crate) fn to_cell_coords(&self, p: Point3<f64>) -> Point3<f64> {
        match self {
            LatticeMap::Rectangular { cell } => Point3::new(
                std::f64::consts::TAU * p.x / cell.x.max(1e-9),
                std::f64::consts::TAU * p.y / cell.y.max(1e-9),
                std::f64::consts::TAU * p.z / cell.z.max(1e-9),
            ),
            LatticeMap::Cylindrical {
                cell_radius,
                cell_height,
                arc_count,
            } => {
                let r = (p.x * p.x + p.y * p.y).sqrt();
                // `atan2(0, 0)` is 0 rather than NaN in Rust, so the axis itself is safe here.
                // The division that is *not* safe is in `local_frequency`, which guards its own.
                let theta = p.y.atan2(p.x);
                Point3::new(
                    std::f64::consts::TAU * r / cell_radius.max(1e-9),
                    *arc_count as f64 * theta,
                    std::f64::consts::TAU * p.z / cell_height.max(1e-9),
                )
            }
            LatticeMap::Spherical {
                cell_radius,
                azimuth_count,
                polar_count,
            } => {
                let r = p.coords.norm();
                let azimuth = p.y.atan2(p.x);
                // Polar angle from +Z, in [0, π]. `acos` of a clamped ratio, so a point at the
                // centre reads as a pole rather than as NaN.
                let polar = if r > 1e-9 {
                    (p.z / r).clamp(-1.0, 1.0).acos()
                } else {
                    0.0
                };
                Point3::new(
                    std::f64::consts::TAU * r / cell_radius.max(1e-9),
                    *azimuth_count as f64 * azimuth,
                    *polar_count as f64 * polar,
                )
            }
            LatticeMap::Conformal {
                surface,
                cells_per_face,
                depth,
                ..
            } => {
                let hit = surface.at(p);
                let n = *cells_per_face as f64;
                Point3::new(
                    std::f64::consts::TAU * n * hit.st[0],
                    std::f64::consts::TAU * n * hit.st[1],
                    // Along the normal rather than the signed distance, which is the same
                    // number over the surface and the continuous one past its rim.
                    std::f64::consts::TAU * hit.along / depth.max(1e-9),
                )
            }
        }
    }

    /// **The gradient magnitude of each cell argument with respect to world position**,
    /// `‖∇_world arg_k‖`.
    ///
    /// Constant for a rectangular map. For a cylindrical one the circumferential term is `N/r`,
    /// which varies with radius — near the axis the pattern is compressed, far from it
    /// stretched — and that is the whole reason wall thickness cannot be a constant.
    ///
    /// # Why it is defined this way and not as a rate along a direction
    ///
    /// It used to be described as the derivative of each argument with respect to arc length
    /// along that argument's own direction, which only means anything for an *orthogonal* map.
    /// The two definitions agree for every map that has one — rectangular and cylindrical both
    /// — so nothing here changes value. What changes is what the abstraction covers.
    ///
    /// `cell-map-family.md` §1 derives it. With `J[k][i] = ∂arg_k/∂x_i` and the cell's
    /// per-argument gradient components taken as uncorrelated with RMS magnitudes `g_k`,
    ///
    /// ```text
    /// E[|∇_world F|²] = Σ_k g_k² · ‖row k of J‖²
    /// ```
    ///
    /// so **only the row norms survive**. A map with shear — a conformal one over a quad mesh
    /// is the case in hand — has off-diagonal Jacobian entries, and they contribute *into the
    /// row norm* rather than needing to be exposed separately. That is why this can stay a
    /// `Vector3` instead of becoming a 3×3, and why the thickness correction needs no second
    /// code path for a map that is not orthogonal.
    ///
    /// Written down before a third map exists, because it is a doc comment today and an
    /// awkward conversation afterwards.
    pub(crate) fn local_frequency(&self, p: Point3<f64>) -> Vector3<f64> {
        match self {
            LatticeMap::Rectangular { cell } => Vector3::new(
                std::f64::consts::TAU / cell.x.max(1e-9),
                std::f64::consts::TAU / cell.y.max(1e-9),
                std::f64::consts::TAU / cell.z.max(1e-9),
            ),
            LatticeMap::Cylindrical {
                cell_radius,
                cell_height,
                arc_count,
            } => {
                // Guarded independently of any inner cutoff: a query landing exactly on the
                // axis is rare and not impossible, and `N/0` poisons every boolean downstream
                // with NaN — which surfaces as missing geometry rather than as an error.
                let r = (p.x * p.x + p.y * p.y).sqrt().max(1e-9);
                Vector3::new(
                    std::f64::consts::TAU / cell_radius.max(1e-9),
                    *arc_count as f64 / r,
                    std::f64::consts::TAU / cell_height.max(1e-9),
                )
            }
            LatticeMap::Spherical {
                cell_radius,
                azimuth_count,
                polar_count,
            } => {
                let r = p.coords.norm().max(1e-9);
                // Azimuth is measured on a circle of radius `r·sin(polar)`, not `r`: near a
                // pole that circle shrinks to nothing, so the azimuthal frequency rises there
                // the same way the cylindrical one rises at the axis. Missing this is how a
                // spherical lattice gets walls that thin toward its poles.
                let rho = (p.x * p.x + p.y * p.y).sqrt().max(1e-9);
                Vector3::new(
                    std::f64::consts::TAU / cell_radius.max(1e-9),
                    *azimuth_count as f64 / rho,
                    *polar_count as f64 / r,
                )
            }
            // **The case the redefinition above was for.** A quad mesh's parametrization has
            // shear, so `grad s` and `grad t` are neither orthogonal to each other nor aligned
            // with anything convenient — and it does not matter, because only their norms reach
            // the thickness formula. `cell-map-family.md` §1 is the derivation.
            //
            // `grad s` comes from the surface itself and is exact for the tessellation: the
            // parametric coordinates are linear across a triangle, so their gradients are
            // constant on it. That is also what makes cell size follow quad size — a smaller
            // face has a larger `grad s` and its cells come out smaller, with nothing asked of
            // the user.
            LatticeMap::Conformal {
                surface,
                cells_per_face,
                depth,
                ..
            } => {
                let hit = surface.at(p);
                let n = std::f64::consts::TAU * *cells_per_face as f64;
                Vector3::new(
                    n * hit.ds.norm(),
                    n * hit.dt.norm(),
                    std::f64::consts::TAU / depth.max(1e-9),
                )
            }
        }
    }

    /// How many whole cells each axis closes on, where it closes on itself at all.
    ///
    /// Only the cells that hash an *absolute* cell index care -- every formula cell is periodic
    /// in its argument and wraps for free. A Voronoi cell does not: the cell one step below zero
    /// and the cell one step below the count are the same place on a cylinder and must draw the
    /// same seed, or the pattern tears along the seam. See `nearest_seeds`.
    ///
    /// Polar angle is deliberately absent even though it closes through the pole. It runs
    /// `0..=pi`, so its argument covers half a period and never wraps within the domain; the
    /// continuation through the pole would have to reflect the azimuth as well, which is a
    /// different fix from a modulo and is not one this needs.
    pub(crate) fn cell_period(&self) -> [Option<u32>; 3] {
        // **Never nought.** A count of nought is not a document the panel can make -- it clamps
        // at one -- but it is a document that can be written, and every other reader of these
        // counts already takes `.max(1)` for the same reason. Here it is the difference between
        // a degenerate lattice and a panic: `rem_euclid(0)` is a division by zero, which on this
        // path is a crash of the application rather than a strange picture.
        match self {
            LatticeMap::Rectangular { .. } | LatticeMap::Conformal { .. } => [None, None, None],
            LatticeMap::Cylindrical { arc_count, .. } => [None, Some((*arc_count).max(1)), None],
            LatticeMap::Spherical { azimuth_count, .. } => {
                [None, Some((*azimuth_count).max(1)), None]
            }
        }
    }

    /// The radius inside which the tiling has degenerated, if it ever does.
    ///
    /// A cylindrical lattice stops being a lattice where one cell's circumferential arc length
    /// has shrunk to the size of the material feature in it, because neighbouring material
    /// touches there: `2π·r/N = feature`. Inside it the geometry is converging on solid, so
    /// solid is what it should be — a solid hub is often exactly what a part wants. It is worth
    /// *saying*, though, because a large arc count pushes this radius out fast and can turn a
    /// surprising fraction of a part dense.
    ///
    /// **Takes a feature size rather than a wall thickness**, because a solid-phase lattice has
    /// no wall and still has to answer this. [`Lattice::feature_size_mm`] is what supplies
    /// it for both phases.
    pub(crate) fn degenerate_radius(&self, feature_mm: f64) -> Option<f64> {
        match self {
            // **Nothing collapses on a surface.** A cylindrical map's cells close up at the
            // axis and a spherical one's at the centre, because an angular argument sweeps the
            // same cell count through a shrinking circle. `s` and `t` do no such thing — they
            // run nought to one across each face however big the face is — so there is no
            // radius inside which the pattern is degenerate.
            LatticeMap::Conformal { .. } => None,
            LatticeMap::Rectangular { .. } => None,
            LatticeMap::Cylindrical { arc_count, .. } => {
                Some(*arc_count as f64 * feature_mm / std::f64::consts::TAU)
            }
            // Whichever of the two closings collapses first. The azimuthal cells shrink toward
            // the poles as well as toward the centre, so the radius that matters is the larger
            // of the two counts.
            LatticeMap::Spherical {
                azimuth_count,
                polar_count,
                ..
            } => Some(
                (*azimuth_count).max(*polar_count) as f64 * feature_mm / std::f64::consts::TAU,
            ),
        }
    }

    /// The largest frequency this map presents at a given distance from its own centre or
    /// axis.
    ///
    /// The building block of a bound that is *local* rather than global. A cylindrical map's
    /// frequency rises without limit toward the axis, so one constant valid everywhere has to
    /// be taken where the cells collapse — and that made the field understate the room by up to
    /// 332x out at the rim, which costs a ray that many more steps than it needs.
    pub(crate) fn frequency_at_radius(&self, r: f64) -> f64 {
        let r = r.max(1e-9);
        match self {
            // **The worst anywhere, because a surface's pattern does not vary with any radius.**
            // It varies with face size, and a ray does not know which face it is about to meet.
            // Asking at the point the ray is standing on would let it step into a finer region
            // and through a wall, which is the failure this whole function exists to avoid.
            LatticeMap::Conformal {
                surface,
                cells_per_face,
                depth,
                ..
            } => {
                let angular =
                    std::f64::consts::TAU * *cells_per_face as f64
                        * surface.worst_parametric_gradient();
                angular.max(std::f64::consts::TAU / depth.max(1e-9))
            }
            LatticeMap::Rectangular { cell } => {
                let f = Vector3::new(
                    std::f64::consts::TAU / cell.x.max(1e-9),
                    std::f64::consts::TAU / cell.y.max(1e-9),
                    std::f64::consts::TAU / cell.z.max(1e-9),
                );
                f.x.max(f.y).max(f.z)
            }
            LatticeMap::Cylindrical {
                cell_radius,
                cell_height,
                arc_count,
            } => (std::f64::consts::TAU / cell_radius.max(1e-9))
                .max(*arc_count as f64 / r)
                .max(std::f64::consts::TAU / cell_height.max(1e-9)),
            LatticeMap::Spherical {
                cell_radius,
                azimuth_count,
                polar_count,
            } => (std::f64::consts::TAU / cell_radius.max(1e-9))
                .max(*azimuth_count as f64 / r)
                .max(*polar_count as f64 / r),
        }
    }

    /// How far from the axis or centre a point is, in whichever sense this map's cells close up.
    pub(crate) fn radius_of(&self, p: Point3<f64>) -> f64 {
        match self {
            // Neither of these has a sense in which its cells close up, so neither has a
            // radius to measure it by.
            LatticeMap::Rectangular { .. } | LatticeMap::Conformal { .. } => f64::INFINITY,
            LatticeMap::Cylindrical { .. } => (p.x * p.x + p.y * p.y).sqrt(),
            LatticeMap::Spherical { .. } => p.coords.norm(),
        }
    }

    /// The largest local frequency this map can present, over the radii where it is still a
    /// lattice.
    ///
    /// What the Lipschitz bound was once built from. A cylindrical map's frequency rises
    /// without limit toward the axis, so a bound taken at any representative radius overstates
    /// how far a ray may step nearer in — and a tracer that steps too far goes through a wall,
    /// which shows as dropouts. Taken at the degenerate radius, since inside that the field is
    /// solid and there is nothing to step through.
    ///
    /// The field no longer divides by one constant: [`Lattice::bound_near`] takes the frequency
    /// where the point is, asked twice, since a bound built from this figure understated the
    /// room out at the rim by up to 332x. So only the tests ask for the worst, and only the
    /// tests compile it.
    #[cfg(test)]
    pub(crate) fn worst_frequency(&self, feature_mm: f64) -> f64 {
        match self {
            // No degenerate radius to take it at, so the worst over the whole surface is the
            // answer -- which `frequency_at_radius` already computes, at any radius.
            LatticeMap::Conformal { .. } => self.frequency_at_radius(1.0),
            LatticeMap::Rectangular { cell } => {
                let f = self.local_frequency(Point3::origin());
                let _ = cell;
                f.x.max(f.y).max(f.z)
            }
            LatticeMap::Cylindrical {
                cell_radius,
                cell_height,
                arc_count,
            } => {
                let r_min = (*arc_count as f64 * feature_mm / std::f64::consts::TAU).max(1e-9);
                let circumferential = *arc_count as f64 / r_min;
                (std::f64::consts::TAU / cell_radius.max(1e-9))
                    .max(circumferential)
                    .max(std::f64::consts::TAU / cell_height.max(1e-9))
            }
            LatticeMap::Spherical {
                cell_radius,
                azimuth_count,
                polar_count,
            } => {
                let r_min = self.degenerate_radius(feature_mm).unwrap_or(1e-9).max(1e-9);
                (std::f64::consts::TAU / cell_radius.max(1e-9))
                    .max(*azimuth_count as f64 / r_min)
                    .max(*polar_count as f64 / r_min)
            }
        }
    }

    /// The smallest cell dimension that is a fixed length rather than a function of radius.
    ///
    /// A curved map has one cell extent that shrinks toward its axis — the circumferential one,
    /// and the whole reason [`Self::degenerate_radius`] exists — and one or two that do not.
    /// This is the smallest of the ones that do not, which is the only cell size a solid-phase
    /// lattice can measure a strut against without asking where it is.
    pub(crate) fn smallest_cell_extent(&self) -> f64 {
        match self {
            LatticeMap::Rectangular { cell } => cell.x.min(cell.y).min(cell.z),
            LatticeMap::Cylindrical {
                cell_radius,
                cell_height,
                ..
            } => cell_radius.min(*cell_height),
            LatticeMap::Spherical { cell_radius, .. } => *cell_radius,
            // One cell spans `1 / (N · ‖∇s‖)` millimetres, so the *smallest* cell is over the
            // largest gradient -- the finest face of the mesh. The depth is the third extent
            // and is a length already.
            LatticeMap::Conformal {
                surface,
                cells_per_face,
                depth,
                ..
            } => {
                let across = surface.worst_parametric_gradient()
                    * (*cells_per_face as f64).max(1.0);
                if across > 1e-12 {
                    (1.0 / across).min(*depth)
                } else {
                    *depth
                }
            }
        }
    }
}
