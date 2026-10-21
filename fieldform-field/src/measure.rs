// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Volume, surface area and centre of mass, measured from a field.
//!
//! # Why these come from a mesh rather than from the field directly
//!
//! Counting how many grid samples read negative is the obvious way, and it converges badly: the
//! error is a whole cell deep along the entire boundary, so it falls off as the *first* power of
//! the cell size while the thing being measured grows as the third. A sphere measured that way
//! at 64 cells is out by a few percent, and the only way to improve it is a finer grid.
//!
//! Meshing first and integrating the mesh converges as the square instead, because marching
//! cubes places each vertex by interpolating the field along an edge rather than rounding it to
//! the nearest corner. That sub-cell placement is the whole difference, and it is already
//! written — so this module is arithmetic over a mesh and owns no sampling of its own.
//!
//! # What "from a field" costs, and why it is stated rather than hidden
//!
//! Every number here is an approximation of a surface this kernel never stores. The measurements
//! below are what that costs at each resolution, and [`MassProperties::of`] takes a resolution
//! rather than choosing one, because whether 0.3% is fine or unacceptable is the caller's
//! question and not this module's.

use nalgebra::{Point3, Vector3};

use rayon::prelude::*;

use crate::{FieldMesh, ScalarField, mesh_field};

/// How much solid lies behind a surface point, up to `limit`.
///
/// **The one measurement of local thickness**, used by the inspection command and by the
/// mesher's own subdivision rule. Walking inward from a surface point along its inward normal,
/// the distance grows to the middle of the wall and shrinks again toward the far side, so the
/// deepest value reached is the half-thickness. A mesh cannot answer this — its vertices are on
/// the surface and nothing in it relates one side of a wall to the other — which is why it is
/// asked of the field.
///
/// Bounded by `limit`, and the bound is the point: a caller asking "is this thinner than X"
/// gains nothing from the exact thickness of a wall that is thicker, and pays for it in samples.
/// A result at `limit` means "at least this", not "exactly this".
///
/// `steps` trades accuracy for cost. The inspection command wants a number to report and takes
/// sixteen; the mesher only wants a yes or no and takes fewer, at every cell it considers.
pub(crate) fn half_thickness(
    field: &dyn ScalarField,
    p: Point3<f64>,
    outward: Vector3<f64>,
    limit: f64,
    steps: usize,
) -> f64 {
    let mut deepest = 0.0f64;
    for step in 1..=steps.max(1) {
        let t = limit * step as f64 / steps.max(1) as f64;
        // Depth, so positive inside. It rises to the middle of the wall and falls after it, and
        // the turn is what says the far surface has been passed.
        let depth = -field.distance(p - outward * t);
        if depth < deepest {
            break;
        }
        deepest = depth;
    }
    deepest
}

/// Where a solid is thinner than it should be, and how thin it gets.
///
/// **The measurement a knife edge needs, rather than a tidier triangulation of one.** Where a
/// lattice wall is cut by the body it infills, the solid tapers to zero thickness — and no mesh
/// of that is printable, however clean, because a wall below the printer's line width is not a
/// wall. So the useful answer is not a better fringe of triangles but a number: how thin does
/// this part get, and where.
///
/// It is also what would make refining for thin features measurable, which is why it comes
/// first: a subdivision rule aimed at thin walls needs something to say whether it worked.
#[derive(Debug, Clone, PartialEq)]
pub struct Thinness {
    /// The thinnest the solid gets anywhere sampled, in millimetres.
    pub thinnest_mm: f64,
    /// Where that was, so it can be looked at rather than only reported.
    pub thinnest_at: Point3<f64>,
    /// How many sampled points sat below the threshold, out of how many were looked at.
    pub below: usize,
    pub sampled: usize,
}

impl Thinness {
    /// The fraction of the surface that is too thin, between 0 and 1.
    pub fn fraction_below(&self) -> f64 {
        if self.sampled == 0 {
            0.0
        } else {
            self.below as f64 / self.sampled as f64
        }
    }
}

/// Measures the local thickness of `field` at every vertex of `mesh`.
///
/// **Read from the field, not from the mesh.** A mesh cannot say how thick a solid is — its
/// vertices are on the surface and nothing in it relates one side of a wall to the other. The
/// field can: walking inward from a surface point along its own inward normal, the distance
/// grows to the middle of the wall and then shrinks again toward the far side, so the deepest
/// value reached is the half-thickness. The mesh is only being used for *where* to ask.
///
/// Bounded by `threshold_mm`, deliberately. The walk stops once the solid is comfortably thicker
/// than the threshold, because the question is "is anything too thin" and the exact thickness of
/// a part that is thick enough costs samples to establish and answers nothing. Anything reported
/// at or above `2 * threshold_mm` means "at least this", not "exactly this".
pub fn thinness(
    field: &(dyn ScalarField + Sync),
    mesh: &FieldMesh,
    threshold_mm: f64,
) -> Option<Thinness> {
    if mesh.positions.is_empty() {
        return None;
    }
    let limit = threshold_mm.max(1e-6);
    // Sixteen steps out to one threshold of depth: fine enough that a wall right at the
    // threshold is not misreported by the sampling, cheap enough to run over a million
    // vertices.
    const STEPS: usize = 16;

    let measured: Vec<(f64, usize)> = mesh
        .positions
        .par_iter()
        .zip(mesh.normals.par_iter())
        .enumerate()
        .map(|(i, (p, n))| {
            (half_thickness(field, *p, *n, limit, STEPS) * 2.0, i)
        })
        .collect();

    let (thinnest_mm, at) = measured
        .iter()
        .copied()
        .fold((f64::INFINITY, 0usize), |best, x| if x.0 < best.0 { x } else { best });
    let below = measured.iter().filter(|(t, _)| *t < threshold_mm).count();

    Some(Thinness {
        thinnest_mm,
        thinnest_at: mesh.positions[at],
        below,
        sampled: measured.len(),
    })
}

/// The most grid corners an anchored measurement samples: the mesher holds a value for every
/// one, so this is 320 MB of `f64`. A 40 mm body measures at h = 1/8 mm inside it; finer than
/// that is [`Anchored::TooFine`] until a narrow-band sampler exists.
pub(crate) const ANCHORED_CORNERS_MAX: usize = 40_000_000;

/// The mesh of `field` on the grid anchored at the world origin, every cell `h` on a side, or
/// how many corners a grid that fine would have needed.
///
/// One function because two measurements read the same lattice: a body's mass properties and its
/// local thickness. Two copies of this arithmetic would be two grids, and the whole point of the
/// anchoring (`benchmark-suite-spec.md` §3.2) is that there is one.
fn anchored_mesh(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    h: f64,
) -> Result<crate::FieldMesh, usize> {
    let (lo, hi) = bounds;
    let snap_down = |v: f64| (v / h).floor() * h - 2.0 * h;
    let snap_up = |v: f64| (v / h).ceil() * h + 2.0 * h;
    let lo = Point3::new(snap_down(lo.x), snap_down(lo.y), snap_down(lo.z));
    let hi = Point3::new(snap_up(hi.x), snap_up(hi.y), snap_up(hi.z));
    let cells = (hi - lo).map(|e| (e / h).round() as usize);
    let corners = (cells.x + 1) * (cells.y + 1) * (cells.z + 1);
    if corners > ANCHORED_CORNERS_MAX {
        return Err(corners);
    }
    // The longest axis in cells is the mesher's resolution, and the other two axes then come
    // out at exactly `h` too, because every extent is a whole number of cells.
    let resolution = cells.x.max(cells.y).max(cells.z);
    Ok(mesh_field(field, (lo, hi), resolution))
}

/// One anchored measurement: the mass properties and the bounds of the surface measured.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Measured {
    pub props: MassProperties,
    /// The axis-aligned bounds of the measured mesh (`benchmark-suite-spec.md` §3.6).
    pub lo: Point3<f64>,
    pub hi: Point3<f64>,
    /// The grid spacing it was measured at.
    pub h: f64,
    /// The local thickness of the thinnest place sampled, or infinity where it was not asked for
    /// (§3.7). Carried here so one reading of one grid answers a mass property or a thickness,
    /// and the two refine together.
    pub thinnest_mm: f64,
}

/// What [`MassProperties::anchored`] found.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Anchored {
    Measured(Measured),
    /// No surface inside the box.
    Empty,
    /// The grid would need more corners than [`ANCHORED_CORNERS_MAX`].
    TooFine { corners: usize },
}

/// What a body weighs and where it balances, in millimetres and millimetres cubed.
///
/// Mass is deliberately absent. It is density times volume and this kernel does not know a
/// document's material, so reporting it would mean inventing one — the volume is the part this
/// can answer, and the multiplication belongs where the material lives.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MassProperties {
    pub volume_mm3: f64,
    pub surface_area_mm2: f64,
    /// The centroid of the enclosed volume, not of the surface. The two differ for anything
    /// that is not symmetric, and it is the volume's that balances.
    pub centre_of_mass: Point3<f64>,
}

impl MassProperties {
    /// Measures a field over a box, meshing it at `resolution` cells along its longest axis.
    ///
    /// `bounds` must contain the whole body. Anything crossing the box is measured as though
    /// the box were a wall, since the mesher closes the surface there — so a bound that clips
    /// the model reports a smaller, differently balanced one rather than failing.
    ///
    /// Returns `None` when the field has no surface inside the box at all, which is a different
    /// answer from a volume of zero: nothing was found to measure, rather than something was
    /// measured and came to nothing.
    pub fn of(
        field: &(dyn ScalarField + Sync),
        bounds: (Point3<f64>, Point3<f64>),
        resolution: usize,
    ) -> Option<Self> {
        Self::of_mesh(&mesh_field(field, bounds, resolution))
    }

    /// Measures a field on a grid **anchored at the world origin**, every cell `h` on a side.
    ///
    /// **Why anchored, and why `h` is the caller's.** `of` lays its grid from the corner of the
    /// box it is given, so two bodies that differ only by a translation are sampled at different
    /// points relative to their own geometry, and each carries its own discretisation error --
    /// about 0.02 % of a 93 000 mm³ body, which swamps a 80 mm³ fillet measured as the difference
    /// of two such bodies. On a grid fixed to the origin, two bodies a whole number of cells
    /// apart are sampled identically and the error **cancels exactly** in a difference or a
    /// ratio (`benchmark-suite-spec.md` §3.2). And `h` is chosen by the caller because a single
    /// measurement says nothing about its own error: `expect` measures at `h` and `h/2` and reads
    /// the difference as the error estimate (§3.4), which only works when both grids are the
    /// same grid at two spacings.
    ///
    /// `bounds` is grown by two cells and snapped outward to multiples of `h`, so the box never
    /// clips the body and its corners lie on the anchored lattice; `mesh_field` then lays cells
    /// of exactly `h` from that corner. `h` should be a power of two, which keeps every multiple
    /// of it exact in `f64`.
    ///
    /// The mesher holds every corner of the box in memory, so a resolution past
    /// [`ANCHORED_CORNERS_MAX`] is refused as [`Anchored::TooFine`] rather than attempted --
    /// which is "measurement is the limit" said before the machine runs out, not after.
    pub fn anchored(
        field: &(dyn ScalarField + Sync),
        bounds: (Point3<f64>, Point3<f64>),
        h: f64,
    ) -> Anchored {
        let mesh = match anchored_mesh(field, bounds, h) {
            Ok(mesh) => mesh,
            Err(corners) => return Anchored::TooFine { corners },
        };
        let Some(props) = Self::of_mesh(&mesh) else { return Anchored::Empty };
        let mut low = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut high = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &mesh.positions {
            low = low.inf(p);
            high = high.sup(p);
        }
        Anchored::Measured(Measured { props, lo: low, hi: high, h, thinnest_mm: f64::INFINITY })
    }

    /// The same reading, with the thinnest place measured too.
    ///
    /// `threshold_mm` bounds the walk as [`thinness`] does: a benchmark asks whether anything is
    /// thinner than a number, and establishing the exact thickness of a part that is thick
    /// enough costs samples and answers nothing.
    pub fn anchored_with_thickness(
        field: &(dyn ScalarField + Sync),
        bounds: (Point3<f64>, Point3<f64>),
        h: f64,
        threshold_mm: f64,
    ) -> Anchored {
        let mesh = match anchored_mesh(field, bounds, h) {
            Ok(mesh) => mesh,
            Err(corners) => return Anchored::TooFine { corners },
        };
        let Some(props) = Self::of_mesh(&mesh) else { return Anchored::Empty };
        let mut low = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut high = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &mesh.positions {
            low = low.inf(p);
            high = high.sup(p);
        }
        let thinnest_mm =
            thinness(field, &mesh, threshold_mm).map_or(f64::INFINITY, |t| t.thinnest_mm);
        Anchored::Measured(Measured { props, lo: low, hi: high, h, thinnest_mm })
    }

    fn of_mesh(mesh: &crate::FieldMesh) -> Option<Self> {
        if mesh.triangles.is_empty() {
            return None;
        }

        // The divergence theorem, once per triangle. The volume integral over a closed surface
        // reduces to a sum of signed tetrahedron volumes on the origin, and the first moment
        // reduces similarly — so both come out of one pass with no interior sampling at all.
        //
        // Signed, so a triangle facing away subtracts. That is what makes a hollow body come out
        // as its wall rather than as its envelope, and it is why the winding has to be right.
        let mut volume = 0.0;
        let mut area = 0.0;
        let mut moment = Vector3::zeros();
        for t in &mesh.triangles {
            let (a, b, c) = (
                mesh.positions[t[0] as usize].coords,
                mesh.positions[t[1] as usize].coords,
                mesh.positions[t[2] as usize].coords,
            );
            let cross = (b - a).cross(&(c - a));
            area += cross.norm() * 0.5;
            let signed_six = a.dot(&cross);
            volume += signed_six;
            // The tetrahedron's own centroid is the mean of its four corners, one of which is
            // the origin — so a quarter of the sum of the other three, weighted by its volume.
            moment += (a + b + c) * signed_six;
        }
        volume /= 6.0;
        // The 6 from the tetrahedra and the 4 from their centroids, folded together.
        moment /= 24.0;

        let centre_of_mass = if volume.abs() > 1e-12 {
            Point3::from(moment / volume)
        } else {
            // No enclosed volume to balance — a sheet, or a body that cancelled itself. The
            // surface's own mean is the honest answer, and it is at least somewhere on the body.
            let sum = mesh
                .positions
                .iter()
                .fold(Vector3::zeros(), |acc, p| acc + p.coords);
            Point3::from(sum / mesh.positions.len().max(1) as f64)
        };

        Some(Self {
            // Absolute, so a field whose surface came out wound inward still reports a positive
            // size. Which way a surface faces is the mesher's business and not the user's.
            volume_mm3: volume.abs(),
            surface_area_mm2: area,
            centre_of_mass,
        })
    }
}

#[cfg(test)]
mod tests {

    /// A slab of known thickness measures as that thickness.
    ///
    /// The simplest shape whose answer is not in doubt: two parallel planes a fixed distance
    /// apart, so every point on either face is exactly that far from the other.
    #[test]
    fn a_slab_measures_as_thick_as_it_is() {
        struct Slab {
            half: f64,
        }
        impl ScalarField for Slab {
            fn distance(&self, p: Point3<f64>) -> f64 {
                p.z.abs() - self.half
            }
            fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
                Some(Vector3::z() * p.z.signum())
            }
        }

        let thickness = 2.0;
        let field = Slab { half: thickness * 0.5 };
        let mesh = mesh_field(
            &field,
            (Point3::new(-6.0, -6.0, -6.0), Point3::new(6.0, 6.0, 6.0)),
            32,
        );
        // Asked with a threshold well above the answer, so the walk reaches the far side rather
        // than stopping at its own bound.
        let report = thinness(&field, &mesh, thickness * 2.0).expect("a slab meshes");
        assert!(
            (report.thinnest_mm - thickness).abs() < 0.2,
            "a {thickness} mm slab measured {:.3} mm",
            report.thinnest_mm
        );
    }

    /// A wedge is thin at its edge and thick at its back, and the report finds the edge.
    ///
    /// The case the whole measurement exists for: a solid that tapers to nothing. The answer is
    /// not a single number for the part but a *worst* one, and where it is.
    #[test]
    fn a_tapering_wedge_reports_its_thin_edge_and_where_it_is() {
        // Thickness falls linearly with x: zero at x = 0 and 2 mm at x = 10.
        struct Wedge;
        impl ScalarField for Wedge {
            fn distance(&self, p: Point3<f64>) -> f64 {
                let half = (p.x * 0.1).max(0.0);
                let inside_slab = p.z.abs() - half;
                inside_slab.max(-p.x).max(p.x - 10.0)
            }
            fn normalization(&self) -> crate::Normalization {
                crate::Normalization::LowerBound
            }
        }

        let mesh = mesh_field(
            &Wedge,
            (Point3::new(-2.0, -2.0, -3.0), Point3::new(12.0, 2.0, 3.0)),
            64,
        );
        let report = thinness(&Wedge, &mesh, 1.0).expect("the wedge meshes");

        assert!(
            report.thinnest_mm < 0.5,
            "the thin edge should measure well under half a millimetre, got {:.3}",
            report.thinnest_mm
        );
        // And it is at the edge rather than somewhere in the body of the wedge.
        assert!(
            report.thinnest_at.x < 3.0,
            "the thinnest place should be near x = 0, found at x = {:.2}",
            report.thinnest_at.x
        );
        assert!(report.below > 0, "some of the surface is below a 1 mm threshold");
        assert!(
            report.fraction_below() < 1.0,
            "and not all of it: the thick end is 2 mm"
        );
    }
    use super::*;
    use crate::{Cuboid, Sphere};

    fn around(centre: Point3<f64>, half: f64) -> (Point3<f64>, Point3<f64>) {
        (
            centre - Vector3::repeat(half),
            centre + Vector3::repeat(half),
        )
    }

    /// A sphere's volume and area against the closed forms, at the resolution the app settles
    /// at.
    ///
    /// Both are checked, because they fail differently: a volume that is systematically small
    /// says the surface is being placed inside where it belongs, and an area that is
    /// systematically *large* says the mesh is faceted — a polyhedron inscribed in a sphere has
    /// less volume and, once it has enough faces, very nearly the right area.
    #[test]
    fn a_spheres_volume_and_area_match_the_closed_forms() {
        let r = 20.0;
        let sphere = Sphere {
            centre: Point3::new(3.0, -4.0, 5.0),
            radius: r,
        };
        let m = MassProperties::of(&sphere, around(sphere.centre, r * 1.5), 128)
            .expect("a sphere has a surface");

        let expected_volume = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let expected_area = 4.0 * std::f64::consts::PI * r * r;
        let volume_error = (m.volume_mm3 - expected_volume).abs() / expected_volume;
        let area_error = (m.surface_area_mm2 - expected_area).abs() / expected_area;
        assert!(
            volume_error < 0.005,
            "volume off by {:.3}%",
            volume_error * 100.0
        );
        assert!(area_error < 0.01, "area off by {:.3}%", area_error * 100.0);

        // And it balances where it was put, to well inside a cell.
        let drift = (m.centre_of_mass - sphere.centre).norm();
        assert!(drift < 0.05, "centre of mass drifted {drift} mm");
    }

    /// A box, where every number is exact rather than transcendental — so an error here is
    /// arithmetic and not convergence.
    #[test]
    fn a_boxs_volume_and_area_match_its_own_dimensions() {
        let size = Vector3::new(40.0, 10.0, 24.0);
        let centre = Point3::new(-2.0, 6.0, 1.0);
        let boxed = Cuboid { centre, size };
        let m = MassProperties::of(&boxed, around(centre, 30.0), 128).expect("a box has a surface");

        let expected_volume = size.x * size.y * size.z;
        let expected_area = 2.0 * (size.x * size.y + size.y * size.z + size.z * size.x);
        assert!(
            (m.volume_mm3 - expected_volume).abs() / expected_volume < 0.01,
            "volume {} against {expected_volume}",
            m.volume_mm3,
        );
        assert!(
            (m.surface_area_mm2 - expected_area).abs() / expected_area < 0.02,
            "area {} against {expected_area}",
            m.surface_area_mm2,
        );
        assert!((m.centre_of_mass - centre).norm() < 0.2);
    }

    /// The centre of mass is the volume's, not the surface's.
    ///
    /// Two spheres of different sizes: the surface's mean sits nearer the midpoint between them
    /// than the volume's does, because area grows as the square and volume as the cube. A
    /// measurement that averaged vertices would land in the wrong place, and on a symmetric
    /// body — every other test here — the two agree and nothing would notice.
    #[test]
    fn the_centre_of_mass_is_weighted_by_volume_and_not_by_surface() {
        struct TwoSpheres {
            big: Sphere,
            small: Sphere,
        }
        impl ScalarField for TwoSpheres {
            fn distance(&self, p: Point3<f64>) -> f64 {
                self.big.distance(p).min(self.small.distance(p))
            }
        }
        let field = TwoSpheres {
            big: Sphere {
                centre: Point3::new(-20.0, 0.0, 0.0),
                radius: 15.0,
            },
            small: Sphere {
                centre: Point3::new(30.0, 0.0, 0.0),
                radius: 5.0,
            },
        };
        let m = MassProperties::of(
            &field,
            (
                Point3::new(-50.0, -20.0, -20.0),
                Point3::new(50.0, 20.0, 20.0),
            ),
            160,
        )
        .expect("two spheres have a surface");

        // Volumes are 4/3 pi r^3, so the big one outweighs the small by 27 to 1: the balance
        // point sits close to it.
        let big_v = 15f64.powi(3);
        let small_v = 5f64.powi(3);
        let expected_x = (-20.0 * big_v + 30.0 * small_v) / (big_v + small_v);
        assert!(
            (m.centre_of_mass.x - expected_x).abs() < 0.5,
            "balanced at {} against {expected_x}",
            m.centre_of_mass.x,
        );
        // The surfaces' own mean would sit further right, since area falls off more slowly.
        let by_area = (-20.0 * 15f64.powi(2) + 30.0 * 5f64.powi(2)) / (225.0 + 25.0);
        assert!(
            (m.centre_of_mass.x - by_area).abs() > 1.0,
            "volume and area weighting should disagree here, or the test proves nothing",
        );
    }

    /// A field with nothing in the box says so, rather than reporting a body of no size.
    #[test]
    fn a_box_with_no_surface_in_it_measures_nothing_rather_than_zero() {
        let sphere = Sphere {
            centre: Point3::new(500.0, 0.0, 0.0),
            radius: 5.0,
        };
        let m = MassProperties::of(
            &sphere,
            (
                Point3::new(-10.0, -10.0, -10.0),
                Point3::new(10.0, 10.0, 10.0),
            ),
            32,
        );
        assert!(m.is_none());
    }

    /// Finer meshing converges, which is the claim the module docs make for meshing at all.
    ///
    /// Asserted as a trend rather than at one resolution: a single tolerance passes just as well
    /// for a measurement that is accidentally right, and the thing worth knowing is that asking
    /// for more accuracy actually buys it.
    #[test]
    fn a_finer_mesh_measures_a_sphere_more_closely() {
        let r = 20.0;
        let sphere = Sphere {
            centre: Point3::origin(),
            radius: r,
        };
        let exact = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let error_at = |n: usize| {
            let m = MassProperties::of(&sphere, around(Point3::origin(), 30.0), n).unwrap();
            (m.volume_mm3 - exact).abs() / exact
        };
        let coarse = error_at(24);
        let fine = error_at(96);
        assert!(fine < coarse, "{fine} should beat {coarse}");
        assert!(fine < 0.01, "96 cells should be within 1%, got {fine}");
    }

    /// A box whose faces lie on the anchored grid is measured exactly, at any spacing that
    /// divides its faces' coordinates: every face is on a grid plane, so marching cubes rebuilds
    /// it with no cut corner (`benchmark-suite-spec.md` §3.5). This is the case `expect`'s
    /// convention probe t0-01 leans on.
    #[test]
    fn a_box_on_the_anchored_grid_measures_exactly() {
        let block = Cuboid { centre: Point3::origin(), size: Vector3::new(10.0, 20.0, 30.0) };
        let bounds = (Point3::new(-5.0, -10.0, -15.0), Point3::new(5.0, 10.0, 15.0));
        for h in [1.0, 0.5, 0.25] {
            let Anchored::Measured(m) = MassProperties::anchored(&block, bounds, h) else {
                panic!("a box measures at h = {h}")
            };
            assert!((m.props.volume_mm3 - 6000.0).abs() < 1e-6, "volume at {h}: {}", m.props.volume_mm3);
            assert!((m.props.surface_area_mm2 - 2200.0).abs() < 1e-6, "area at {h}: {}", m.props.surface_area_mm2);
            assert!((m.lo - bounds.0).norm() < 1e-9 && (m.hi - bounds.1).norm() < 1e-9, "bounds at {h}");
        }
    }

    /// Two copies of a body a whole number of cells apart measure **identically**, which is what
    /// makes a fillet measurable as the difference of two large volumes. On a grid laid from each
    /// body's own corner the two would carry independent errors; on the anchored grid they are
    /// sampled at the same points relative to their geometry.
    #[test]
    fn a_body_moved_by_whole_cells_measures_the_same_to_the_last_digit() {
        let h = 0.5;
        let at = |x: f64| Sphere { centre: Point3::new(x, 0.25, -0.75), radius: 7.3 };
        let measure = |s: &Sphere| match MassProperties::anchored(s, around(s.centre, 7.3), h) {
            Anchored::Measured(m) => m.props,
            other => panic!("a sphere measures: {other:?}"),
        };
        let here = measure(&at(0.0));
        let there = measure(&at(13.0));
        assert!((here.volume_mm3 - there.volume_mm3).abs() < 1e-9, "{} vs {}", here.volume_mm3, there.volume_mm3);
        assert!((here.surface_area_mm2 - there.surface_area_mm2).abs() < 1e-9);
        // And off the grid by a fraction of a cell, it is not: the test can tell the difference.
        let off = measure(&at(13.2));
        assert!((here.volume_mm3 - off.volume_mm3).abs() > 1e-6, "a fractional move is sampled differently");
    }

    #[test]
    fn a_grid_too_fine_to_hold_is_refused_before_it_is_built() {
        let ball = Sphere { centre: Point3::origin(), radius: 50.0 };
        assert!(matches!(
            MassProperties::anchored(&ball, around(ball.centre, 50.0), 1.0 / 64.0),
            Anchored::TooFine { .. }
        ));
    }
}
