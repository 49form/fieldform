// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The lattice's tests: every cell's bounds and fill tables held to its field, under every map.

    /// A flat panel of `n` by `n` quads in the XY plane, open all the way round.
    fn test_panel(n: usize, size: f64) -> fieldform_cage::QuadMesh {
        let step = size / n as f64;
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                positions.push(Point3::new(i as f64 * step, j as f64 * step, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let mut faces = Vec::new();
        for j in 0..n {
            for i in 0..n {
                faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a flat panel")
    }

    /// **A conformal map puts one whole period of cell in each parametric step.**
    ///
    /// The property that makes it conformal rather than merely positioned: the cell arguments
    /// advance by `2π` across `1/N` of a face, wherever on the mesh that face is and however
    /// big it is. On a flat panel `s` and `t` are the panel's own axes, so the check is
    /// arithmetic anybody can follow.
    #[test]
    fn a_conformal_map_advances_one_period_per_cell_across_a_face() {
        let map = LatticeMap::Conformal {
            surface: std::sync::Arc::new(
                crate::SurfaceMap::new(&test_panel(4, 4.0), 4).expect("tessellates"),
            ),
            cells_per_face: 2,
            reach: None,
            depth: 3.0,
        };

        // Face (0, 0) spans x and y from 0 to 1. With two cells across it, the first argument
        // advances 2π every half millimetre.
        let a = map.to_cell_coords(Point3::new(0.0, 0.5, 0.0));
        let b = map.to_cell_coords(Point3::new(0.5, 0.5, 0.0));
        assert!(
            (b.x - a.x - std::f64::consts::TAU).abs() < 0.05,
            "half a face should be one cell: {} to {}",
            a.x,
            b.x
        );

        // And off the surface, the third argument is the depth.
        let above = map.to_cell_coords(Point3::new(0.5, 0.5, 3.0));
        assert!(
            (above.z - std::f64::consts::TAU).abs() < 1e-6,
            "three millimetres off a depth of three is one period, got {}",
            above.z
        );
        assert!(
            (map.to_cell_coords(Point3::new(0.5, 0.5, 0.0)).z).abs() < 1e-9,
            "on the surface the third argument is nought"
        );
    }

    /// **Cell size follows quad size — the mesh is the grading.**
    ///
    /// Two panels of the same width, one meshed twice as finely. The finer one puts twice as
    /// many cells across the same millimetre, and nobody had to say so: `‖∇s‖` is larger on a
    /// smaller face and that is the whole mechanism. This is what makes graded conformal
    /// latticing fall out of the input rather than needing a field to drive it.
    #[test]
    fn a_finer_mesh_gives_finer_cells_without_being_told() {
        let frequency = |n: usize| {
            let map = LatticeMap::Conformal {
                surface: std::sync::Arc::new(
                    crate::SurfaceMap::new(&test_panel(n, 8.0), 4).expect("tessellates"),
                ),
                cells_per_face: 1,
                reach: None,
            depth: 2.0,
            };
            map.local_frequency(Point3::new(4.0, 4.0, 0.5)).x
        };

        let coarse = frequency(2);
        let fine = frequency(4);
        assert!(
            (fine / coarse - 2.0).abs() < 1e-6,
            "twice the faces should be twice the frequency: {coarse} then {fine}"
        );
    }

    /// Nothing on a surface collapses, so nothing about it is degenerate — unlike the two
    /// curved maps, whose cells close up at an axis and at a centre.
    #[test]
    fn a_conformal_map_has_no_degenerate_radius() {
        let map = LatticeMap::Conformal {
            surface: std::sync::Arc::new(
                crate::SurfaceMap::new(&test_panel(2, 2.0), 3).expect("tessellates"),
            ),
            cells_per_face: 3,
            reach: None,
            depth: 1.0,
        };
        assert_eq!(map.degenerate_radius(0.5), None);
        assert_eq!(map.radius_of(Point3::new(9.0, 9.0, 9.0)), f64::INFINITY);
        assert!(map.worst_frequency(0.5).is_finite());
    }
    use super::*;
    use super::beam::point_to_segment;
    use super::voronoi::{nearest_seeds, Seed};

    /// A cylindrical lattice at a stated cell size and count, for the tests below.
    fn cylindrical(kind: TpmsKind, arc_count: u32, thickness: f64) -> Lattice {
        Lattice {
            cell: CellKind::Tpms(kind),
            map: LatticeMap::Cylindrical {
                cell_radius: 5.0,
                cell_height: 5.0,
                arc_count,
            },
            phase: Phase::sheet(thickness),
        }
    }

    /// The seam at `theta = ±π` is not a seam.
    ///
    /// The two sides are the same physical place, and their cell coordinates differ by `2πN`.
    /// The cell formula is `2π`-periodic, so they agree only when `N` is a whole number — which
    /// is why the count is typed as one. This samples across the join and asks for continuity
    /// rather than trusting the argument.
    ///
    /// All three kinds, with no per-kind path in the map: if any of them needed special-casing,
    /// the cell/map contract would have been violated somewhere.
    #[test]
    fn every_kind_is_continuous_across_the_seam_at_pi() {
        for kind in TpmsKind::ALL {
            let lattice = cylindrical(kind, 38, 1.0);
            for i in 0..40 {
                let r = 20.0 + i as f64 * 0.5;
                let z = -10.0 + i as f64 * 0.5;
                // A hair either side of the join, which is where a fractional count would show.
                let eps = 1e-7;
                let a = Point3::new(
                    r * (std::f64::consts::PI - eps).cos(),
                    r * (std::f64::consts::PI - eps).sin(),
                    z,
                );
                let b = Point3::new(
                    r * (-std::f64::consts::PI + eps).cos(),
                    r * (-std::f64::consts::PI + eps).sin(),
                    z,
                );
                let (da, db) = (lattice.distance(a), lattice.distance(b));
                // Bounded by a multiple of the probe offset, not by a fixed number. The two
                // samples sit `2·eps` apart in space, so a *continuous* field differs across
                // them by its own slope times that separation — which is O(eps) and shrinks
                // with it. A tear does not: it stays the same size however close the probes
                // get. A fixed tolerance cannot tell those apart, and this one duly failed
                // when the field's scaling got steeper while remaining perfectly continuous.
                assert!(
                    (da - db).abs() < eps * 100.0,
                    "{kind:?} jumps by {} across the seam at r = {r}, which is {} times the \
                     probe offset — a slope scales with it, a tear does not",
                    (da - db).abs(),
                    (da - db).abs() / eps,
                );
            }
        }
    }

    /// The spherical map's azimuthal seam joins up, for all three surfaces and with no
    /// map-specific code anywhere in the cell.
    ///
    /// The same test as the cylindrical one, on a different map, which is the whole claim the
    /// cell/map split makes: any cell drops into any map. If a kind or a map needed special
    /// casing to pass this, the contract would have been broken somewhere.
    #[test]
    fn every_kind_is_continuous_across_the_spherical_seam() {
        for kind in TpmsKind::ALL {
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Spherical {
                    cell_radius: 5.0,
                    azimuth_count: 24,
                    polar_count: 12,
                },
                phase: Phase::sheet(1.0),
            };
            let eps = 1e-7;
            for i in 0..30 {
                let r = 25.0 + i as f64 * 0.7;
                // Away from the poles, where the azimuth is genuinely undefined rather than
                // discontinuous.
                let polar = 0.4 + i as f64 * 0.07;
                let at = |azimuth: f64| {
                    Point3::new(
                        r * polar.sin() * azimuth.cos(),
                        r * polar.sin() * azimuth.sin(),
                        r * polar.cos(),
                    )
                };
                let da = lattice.distance(at(std::f64::consts::PI - eps));
                let db = lattice.distance(at(-std::f64::consts::PI + eps));
                // A multiple of the probe offset, for the reason the cylindrical seam test
                // gives: a continuous field differs across two nearby points by its slope, and
                // that shrinks with the separation where a tear would not.
                assert!(
                    (da - db).abs() < eps * 100.0,
                    "{kind:?} jumps by {} across the azimuthal seam at r = {r}, which is {} \
                     times the probe offset",
                    (da - db).abs(),
                    (da - db).abs() / eps,
                );
            }
        }
    }

    /// A spherical lattice never overstates the room before its wall either.
    ///
    /// The lower-bound contract, on the map whose frequencies vary in *two* directions rather
    /// than one — so a bound that only accounted for the radial closing would fail here and
    /// not on the cylinder.
    #[test]
    fn a_spherical_lattice_never_overstates_the_room_before_its_wall() {
        let lattice = Lattice {
            cell: CellKind::Tpms(TpmsKind::Gyroid),
            map: LatticeMap::Spherical {
                cell_radius: 6.0,
                azimuth_count: 20,
                polar_count: 10,
            },
            phase: Phase::sheet(1.0),
        };
        let r_min = lattice.map.degenerate_radius(1.0).unwrap();
        for i in 0..20 {
            for j in 0..8 {
                let r = r_min + 2.0 + i as f64 * 2.3;
                let azimuth = j as f64 * 0.77;
                let polar = 0.5 + (i % 5) as f64 * 0.4;
                let p = Point3::new(
                    r * polar.sin() * azimuth.cos(),
                    r * polar.sin() * azimuth.sin(),
                    r * polar.cos(),
                );
                let claimed = lattice.distance(p);
                if claimed <= 0.0 {
                    continue;
                }
                for d in [
                    Vector3::new(1.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                    Vector3::new(0.0, 0.0, 1.0),
                    Vector3::new(0.577, 0.577, 0.577),
                ] {
                    let mut t = 0.0;
                    while t < claimed * 1.5 {
                        if lattice.distance(p + d * t) <= 0.0 {
                            assert!(
                                t >= claimed - 1e-6,
                                "claimed {claimed} but a wall is {t} away",
                            );
                            break;
                        }
                        t += 0.002;
                    }
                }
            }
        }
    }

    /// Toward a pole the azimuthal cells close up, so the isovalue has to rise there — the same
    /// correction the cylindrical map needs toward its axis, in a second direction.
    ///
    /// Missing this is how a spherical lattice gets walls that thin toward its poles: the
    /// azimuth is measured on a circle of radius `r·sin(polar)`, not `r`, and that circle
    /// vanishes at the pole while `r` does not.
    #[test]
    fn the_spherical_isovalue_rises_toward_a_pole() {
        let lattice = Lattice {
            cell: CellKind::Tpms(TpmsKind::Gyroid),
            map: LatticeMap::Spherical {
                cell_radius: 5.0,
                azimuth_count: 24,
                polar_count: 12,
            },
            phase: Phase::sheet(1.0),
        };
        let r = 40.0;
        let at_polar =
            |polar: f64| lattice.isovalue_at(Point3::new(r * polar.sin(), 0.0, r * polar.cos()));
        let equator = at_polar(std::f64::consts::FRAC_PI_2);
        let near_pole = at_polar(0.15);
        assert!(
            near_pole > equator,
            "the isovalue should rise toward a pole: {near_pole} against {equator}",
        );
    }

    /// The cylindrical lattice `every_kind_is_continuous_across_the_seam_at_pi` samples would
    /// tear with a count that is *not* a whole number.
    ///
    /// The type system prevents that, so this asserts the reason rather than the symptom: a
    /// half-integer count evaluated by hand does jump, which is what makes `u32` load-bearing
    /// rather than tidy.
    #[test]
    fn a_fractional_arc_count_would_tear_the_seam_which_is_why_it_cannot_be_one() {
        let kind = TpmsKind::Gyroid;
        let eps = 1e-7;
        let r = 30.0;
        let at = |n: f64, theta: f64| {
            let args = Point3::new(std::f64::consts::TAU * r / 5.0, n * theta, 0.0);
            kind.f(args)
        };
        let whole =
            (at(38.0, std::f64::consts::PI - eps) - at(38.0, -std::f64::consts::PI + eps)).abs();
        let half =
            (at(38.5, std::f64::consts::PI - eps) - at(38.5, -std::f64::consts::PI + eps)).abs();
        // Looser than the tear by four orders of magnitude, and looser than the probe offset
        // itself: stepping 1e-7 either side of the join moves the argument by `N · 2e-7`, so a
        // value difference of that order is the offset and not a discontinuity.
        assert!(whole < 1e-4, "a whole count joins up, got {whole}");
        assert!(half > 0.1, "a fractional count should tear, got {half}");
    }

    /// Far from the axis over a small angular span, a cylinder is flat — so a cylindrical map
    /// should agree with the rectangular one that matches it there.
    ///
    /// The cheapest check that the coordinate arithmetic is right, and it fails loudly for the
    /// mistakes that are easy to make: a factor of `2π` in the wrong place, or radius and
    /// height swapped.
    #[test]
    fn at_a_large_radius_a_cylindrical_map_agrees_with_the_flat_one() {
        let r0 = 4000.0;
        // Whatever count puts cells nearest 5 mm at this radius.
        let arc_count = (std::f64::consts::TAU * r0 / 5.0).round() as u32;
        let cyl = Lattice {
            cell: CellKind::Tpms(TpmsKind::Gyroid),
            map: LatticeMap::Cylindrical {
                cell_radius: 5.0,
                cell_height: 5.0,
                arc_count,
            },
            phase: Phase::sheet(1.0),
        };
        // The matching flat lattice, in the local frame where x is radial and y circumferential.
        let flat = Lattice {
            cell: CellKind::Tpms(TpmsKind::Gyroid),
            map: LatticeMap::Rectangular {
                cell: Vector3::new(5.0, std::f64::consts::TAU * r0 / arc_count as f64, 5.0),
            },
            phase: Phase::sheet(1.0),
        };
        // **Signs, not values.** The two fields describe the same surface but do not carry the
        // same scale: a cylindrical map divides by the worst frequency it can present, which is
        // the one at the degenerate radius, so far from the axis its values are uniformly
        // smaller than the flat map's. That is deliberate — the bound has to hold everywhere a
        // ray might go — and it means agreement is about *where the material is*, which is what
        // the sign says.
        let mut checked = 0;
        for i in 0..60 {
            let dr = i as f64 * 0.1 - 3.0;
            let arc = i as f64 * 0.07 - 2.0;
            let theta = arc / r0;
            let p = Point3::new((r0 + dr) * theta.cos(), (r0 + dr) * theta.sin(), 1.7);
            let q = Point3::new(r0 + dr, arc, 1.7);
            let (a, b) = (cyl.distance(p), flat.distance(q));
            // Skip points sitting essentially on a wall, where either field's sign is a
            // coin toss and disagreement means nothing.
            if a.abs() < 1e-3 || b.abs() < 1e-3 {
                continue;
            }
            assert_eq!(
                a < 0.0,
                b < 0.0,
                "at dr = {dr}, arc = {arc}: cylindrical says {a}, flat says {b}",
            );
            checked += 1;
        }
        assert!(
            checked > 30,
            "only {checked} points were far enough from a wall to compare"
        );
    }

    /// Inside the radius where cells have collapsed, the lattice is solid rather than NaN,
    /// infinite, or full of holes.
    #[test]
    fn the_axis_goes_solid_rather_than_degenerating() {
        let lattice = cylindrical(TpmsKind::Gyroid, 38, 1.0);
        let r_min = lattice
            .map
            .degenerate_radius(1.0)
            .expect("cylindrical maps have one");
        assert!(
            (r_min - 38.0 / std::f64::consts::TAU).abs() < 1e-9,
            "r_min is N*t/2pi"
        );

        for i in 0..20 {
            let r = r_min * i as f64 / 20.0;
            let p = Point3::new(r, 0.0, 3.0);
            let d = lattice.distance(p);
            assert!(d.is_finite(), "NaN or infinity at r = {r}");
            assert!(d <= 0.0, "inside r_min should be solid, got {d} at r = {r}");
        }
        // And exactly on the axis, which is the point most likely to divide by zero.
        assert!(lattice.distance(Point3::new(0.0, 0.0, 0.0)).is_finite());
    }

    /// Wall thickness has to be computed per point, not hoisted.
    ///
    /// Under a cylindrical map the circumferential frequency is `N/r`, so the isovalue that
    /// produces a given wall differs by radius — at half the radius it is twice as high. This
    /// asserts the isovalue actually moves, which is what a hoisted constant would break, and
    /// the direction it moves in.
    #[test]
    fn the_isovalue_rises_toward_the_axis_because_the_frequency_does() {
        let lattice = cylindrical(TpmsKind::Gyroid, 38, 1.0);
        let near = lattice.isovalue_at(Point3::new(20.0, 0.0, 0.0));
        let far = lattice.isovalue_at(Point3::new(40.0, 0.0, 0.0));
        assert!(
            near > far,
            "the isovalue should rise toward the axis: {near} vs {far}"
        );

        // A rectangular map has no such variation, which is the case the generalisation must
        // not have disturbed.
        let flat = Lattice {
            cell: CellKind::Tpms(TpmsKind::Gyroid),
            map: LatticeMap::Rectangular {
                cell: Vector3::new(5.0, 5.0, 5.0),
            },
            phase: Phase::sheet(1.0),
        };
        let a = flat.isovalue_at(Point3::new(20.0, 0.0, 0.0));
        let b = flat.isovalue_at(Point3::new(40.0, 3.0, -7.0));
        assert!(
            (a - b).abs() < 1e-12,
            "a flat map's isovalue is the same everywhere"
        );
    }

    /// The generalised thickness formula reduces to the one it replaced.
    ///
    /// With all three frequencies equal and the cell's gradient split evenly between its
    /// arguments, the quadrature collapses to `(thickness/2) · omega · g` — the rectangular
    /// expression. This is the check that the `1/sqrt(3)` convention in
    /// [`TpmsKind::argument_gradient`] is the right one, and getting it wrong would scale every
    /// wall in every model by that factor.
    #[test]
    fn the_thickness_formula_reduces_to_the_rectangular_one() {
        for kind in TpmsKind::ALL {
            let cell_mm = 7.0;
            let thickness = 1.3;
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Rectangular {
                    cell: Vector3::new(cell_mm, cell_mm, cell_mm),
                },
                phase: Phase::sheet(thickness),
            };
            let omega = std::f64::consts::TAU / cell_mm;
            let expected = thickness * 0.5 * omega * kind.surface_gradient();
            let got = lattice.isovalue_at(Point3::new(1.0, 2.0, 3.0));
            assert!(
                (got - expected).abs() < 1e-12,
                "{kind:?}: {got} against the old formula's {expected}",
            );
        }
    }

    /// The value never overstates the room before a wall, under a cylindrical map too.
    ///
    /// The lower-bound contract, which is the whole reason this field declares itself one. A
    /// cylindrical map makes the understatement worse rather than introducing a new kind of
    /// problem — but "worse" must still mean *under*, or a tracer steps through a wall.
    #[test]
    fn a_cylindrical_lattice_never_overstates_the_room_before_its_wall() {
        let lattice = cylindrical(TpmsKind::Gyroid, 38, 1.0);
        let r_min = lattice.map.degenerate_radius(1.0).unwrap();
        for i in 0..24 {
            for j in 0..12 {
                let r = r_min + 1.0 + i as f64 * 1.7;
                let theta = j as f64 * 0.51;
                let p = Point3::new(r * theta.cos(), r * theta.sin(), i as f64 * 0.3 - 3.0);
                let claimed = lattice.distance(p);
                if claimed <= 0.0 {
                    continue;
                }
                // March along a few directions for the true nearest surface, going *past* what
                // the field claims: a march that stops at the claim can only ever confirm it.
                for d in [
                    Vector3::new(1.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                    Vector3::new(0.0, 0.0, 1.0),
                    Vector3::new(0.577, 0.577, 0.577),
                ] {
                    let mut t = 0.0;
                    let step = 0.002;
                    while t < claimed * 1.5 {
                        if lattice.distance(p + d * t) <= 0.0 {
                            assert!(
                                t >= claimed - 1e-6,
                                "claimed {claimed} but a wall is {t} away at {p:?}",
                            );
                            break;
                        }
                        t += step;
                    }
                }
            }
        }
    }

    /// Points spread over one cell, avoiding nothing: the bound has to hold everywhere a
    /// tracer might evaluate, not only near the surface.
    fn cell_probes(n: usize) -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        let step = std::f64::consts::TAU / n as f64;
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    out.push(Point3::new(
                        i as f64 * step,
                        j as f64 * step,
                        l as f64 * step,
                    ));
                }
            }
        }
        out
    }

    /// The bound must actually bound. If a sample exceeds it, the field overestimates distance
    /// somewhere, and every traced image and every pruned region built on it is untrustworthy.
    #[test]
    fn every_kinds_gradient_bound_is_an_upper_bound() {
        for kind in TpmsKind::ALL {
            let mut worst = 0.0f64;
            for p in cell_probes(40) {
                worst = worst.max(kind.gradient(p).norm());
            }
            assert!(
                worst <= kind.gradient_bound() + 1e-9,
                "{}: sampled |grad| {worst} exceeds the bound {}",
                kind.label(),
                kind.gradient_bound(),
            );
        }
    }

    /// The analytic gradient has to be the gradient of the analytic value, or the normals point
    /// somewhere the surface is not. Checked against a central difference.
    #[test]
    fn every_kinds_gradient_matches_a_difference_of_its_value() {
        let h = 1e-6;
        for kind in TpmsKind::ALL {
            let mut worst = 0.0f64;
            for p in cell_probes(12) {
                let at = |o: Vector3<f64>| kind.f(p + o);
                let numeric = Vector3::new(
                    (at(Vector3::new(h, 0.0, 0.0)) - at(Vector3::new(-h, 0.0, 0.0))) / (2.0 * h),
                    (at(Vector3::new(0.0, h, 0.0)) - at(Vector3::new(0.0, -h, 0.0))) / (2.0 * h),
                    (at(Vector3::new(0.0, 0.0, h)) - at(Vector3::new(0.0, 0.0, -h))) / (2.0 * h),
                );
                worst = worst.max((kind.gradient(p) - numeric).norm());
            }
            assert!(
                worst < 1e-6,
                "{}: gradient disagrees by {worst}",
                kind.label()
            );
        }
    }

    /// The field must never overestimate the distance to the wall, which is what makes it safe
    /// to sphere-trace. Checked the honest way: march from a point until the sign flips, and
    /// confirm the value never claimed more room than there turned out to be.
    #[test]
    fn the_value_never_overstates_the_room_before_the_wall() {
        let lattice = Lattice {
            cell: CellKind::Tpms(TpmsKind::Gyroid),
            map: LatticeMap::Rectangular {
                cell: Vector3::new(50.0, 50.0, 50.0),
            },
            phase: Phase::sheet(5.0),
        };
        // Both phases, because the bound is what makes either safe to trace and the solid arm
        // reaches it by a different route — a constant isovalue rather than one that varies with
        // the local frequency.
        for phase in [
            Phase::sheet(5.0),
            Phase::Solid { density: 0.4 },
        ] {
        for kind in TpmsKind::ALL {
            let lattice = Lattice { cell: CellKind::Tpms(kind), phase, ..lattice.clone() };
            for p in cell_probes(9) {
                // Out of cell coordinates and into millimetres.
                let from = Point3::new(p.x * 8.0, p.y * 8.0, p.z * 8.0);
                let claimed = lattice.distance(from);
                if claimed <= 0.0 {
                    continue;
                }
                // Step along +X by what the field said was safe; the sign must not have
                // changed on the way, or the value promised room that was not there.
                let steps = 32;
                for s in 1..=steps {
                    let t = claimed * s as f64 / steps as f64;
                    let at = Point3::new(from.x + t, from.y, from.z);
                    assert!(
                        lattice.distance(at) > -1e-9,
                        "{} {phase:?}: crossed the wall {t} into a claimed {claimed}",
                        kind.label(),
                    );
                }
            }
        }
        }
    }

    /// A wall is where the field is negative, and there has to be one: a lattice whose value is
    /// positive everywhere is an empty part, and one negative everywhere is a solid block.
    #[test]
    fn a_lattice_has_both_wall_and_void() {
        for (kind, phase) in TpmsKind::ALL.into_iter().flat_map(|k| {
            [
                (k, Phase::sheet(2.0)),
                // The solid's version of the same property: `F <= iso` has to be neither
                // everything nor nothing. An isovalue past either extreme of the cell gives one
                // or the other, which is what a mis-transcribed table would produce.
                (k, Phase::Solid { density: 0.4 }),
            ]
        }) {
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Rectangular {
                    cell: Vector3::new(20.0, 20.0, 20.0),
                },
                phase,
            };
            let mut inside = 0;
            let mut outside = 0;
            for p in cell_probes(12) {
                // One cell, in millimetres. `cell_probes` spans [0, TAU) in each axis, so the
                // scale is the cell over a full period — the same on all three, which x did not
                // used to be: it carried a stray factor that swept it over 6.28 cells while y
                // and z stayed within one. Harmless to what this asserts, since a TPMS repeats,
                // but it made the comment above untrue.
                let mm_per_radian = 20.0 / std::f64::consts::TAU;
                let at = Point3::new(
                    p.x * mm_per_radian,
                    p.y * mm_per_radian,
                    p.z * mm_per_radian,
                );
                if lattice.distance(at) < 0.0 {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
            assert!(inside > 0, "{} {phase:?}: no material anywhere", kind.label());
            assert!(outside > 0, "{} {phase:?}: no void anywhere", kind.label());
        }
    }

    fn corners() -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    out.push(Point3::new(x, y, z));
                }
            }
        }
        out
    }

    fn face_centres() -> Vec<Point3<f64>> {
        vec![
            Point3::new(0.5, 0.5, 0.0),
            Point3::new(0.5, 0.5, 1.0),
            Point3::new(0.5, 0.0, 0.5),
            Point3::new(0.5, 1.0, 0.5),
            Point3::new(0.0, 0.5, 0.5),
            Point3::new(1.0, 0.5, 0.5),
        ]
    }

    /// Every strut of a beam cell, unfolded, as segments in the unit cell.
    ///
    /// The honest description, with no symmetry taken advantage of — which is the point: it is
    /// what the folded formula is checked against.
    fn struts(beam: BeamKind) -> Vec<(Point3<f64>, Point3<f64>)> {
        let at = |x: f64, y: f64, z: f64| Point3::new(x, y, z);
        match beam {
            // One segment along each axis. Their images tile into the full lines.
            BeamKind::Cubic => vec![
                (at(0.0, 0.0, 0.0), at(1.0, 0.0, 0.0)),
                (at(0.0, 0.0, 0.0), at(0.0, 1.0, 0.0)),
                (at(0.0, 0.0, 0.0), at(0.0, 0.0, 1.0)),
            ],
            // Cube edges plus both diagonals on every face.
            BeamKind::Fcc => {
                let mut out = struts(BeamKind::Cubic);
                for axis in 0..3 {
                    for level in [0.0, 1.0] {
                        let face: Vec<Point3<f64>> = corners()
                            .into_iter()
                            .filter(|c| c[axis] == level)
                            .collect();
                        for i in 0..face.len() {
                            for j in i + 1..face.len() {
                                if ((face[i] - face[j]).norm() - 2f64.sqrt()).abs() < 1e-9 {
                                    out.push((face[i], face[j]));
                                }
                            }
                        }
                    }
                }
                out
            }
            // The octahedron on the face centres, plus every corner to the face centres it
            // touches.
            BeamKind::Octet => {
                let edge = 0.5f64.sqrt();
                let mut out = Vec::new();
                let faces = face_centres();
                for i in 0..faces.len() {
                    for j in i + 1..faces.len() {
                        if ((faces[i] - faces[j]).norm() - edge).abs() < 1e-9 {
                            out.push((faces[i], faces[j]));
                        }
                    }
                }
                for k in corners() {
                    for f in face_centres() {
                        if ((k - f).norm() - edge).abs() < 1e-9 {
                            out.push((k, f));
                        }
                    }
                }
                out
            }
            // Two interpenetrating FCC sublattices, every A site bonded to the four B sites at
            // a quarter of the body diagonal.
            BeamKind::Diamond => {
                let a = [
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.5, 0.5, 0.0),
                    Point3::new(0.5, 0.0, 0.5),
                    Point3::new(0.0, 0.5, 0.5),
                ];
                let bond = 3f64.sqrt() / 4.0;
                let mut out = Vec::new();
                for pa in a {
                    for pb in a {
                        let pb = pb + Vector3::new(0.25, 0.25, 0.25);
                        for i in -1..=1 {
                            for j in -1..=1 {
                                for k in -1..=1 {
                                    let q = pb + Vector3::new(i as f64, j as f64, k as f64);
                                    if ((pa - q).norm() - bond).abs() < 1e-9 {
                                        out.push((pa, q));
                                    }
                                }
                            }
                        }
                    }
                }
                out
            }
            // All eight corners to the centre.
            BeamKind::Bcc => {
                let centre = at(0.5, 0.5, 0.5);
                let mut out = Vec::new();
                for x in [0.0, 1.0] {
                    for y in [0.0, 1.0] {
                        for z in [0.0, 1.0] {
                            out.push((at(x, y, z), centre));
                        }
                    }
                }
                out
            }
        }
    }

    /// The two smallest strut distances, minimised over the 27 neighbouring images.
    fn nearest_two(beam: BeamKind, u: Point3<f64>) -> (f64, f64) {
        // Wrapped into the base cell first, by *periodicity* alone — which is not what is in
        // question here. The reflection fold is what the test is checking, and this does not use
        // it. Without the wrap, a query a cell or two out has its nearest strut outside the 27
        // images below, and the brute force reports a larger distance than the truth: the first
        // run of this test failed that way, and the folded formula was the one that was right.
        let wrap = |t: f64| t - t.floor();
        let u = Point3::new(wrap(u.x), wrap(u.y), wrap(u.z));
        let mut all = Vec::new();
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    let o = Vector3::new(i as f64, j as f64, k as f64);
                    for (a, b) in struts(beam) {
                        all.push(point_to_segment(u, a + o, b + o));
                    }
                }
            }
        }
        all.sort_by(f64::total_cmp);
        (all[0], all[1])
    }

    /// The fold is exact, not an approximation — checked against the minimum it stands in for.
    ///
    /// **The one test that can catch a fold which is right for most of a cell and wrong in a
    /// corner**, which is the failure that still looks like a lattice from every angle. The
    /// folded formula evaluates one or three segments; this evaluates every strut in 27 images
    /// and they must agree exactly, because the fold is a claim about symmetry rather than a
    /// tolerance.
    #[test]
    fn the_folded_strut_distance_is_the_honest_minimum_over_every_image() {
        for beam in BeamKind::ALL {
            let n = 23;
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        // Deliberately spanning more than one cell, so the periodicity is
                        // exercised rather than assumed.
                        let u = Point3::new(
                            -1.0 + 3.0 * i as f64 / n as f64,
                            -1.0 + 3.0 * j as f64 / n as f64,
                            -1.0 + 3.0 * k as f64 / n as f64,
                        );
                        let folded = beam.axis_distance(u);
                        let (brute, _) = nearest_two(beam, u);
                        assert!(
                            (folded - brute).abs() < 1e-12,
                            "{}: folded {folded} against {brute} at {u:?}",
                            beam.label(),
                        );
                    }
                }
            }
        }
    }

    /// Points well away from any crease, where a distance field has a gradient at all.
    ///
    /// A crease is where the nearest strut changes hands, and the field is genuinely
    /// non-differentiable there — the same medial-axis ridge every CSG union in this crate has.
    /// Identified from the brute force rather than from the geometry, so it needs no separate
    /// argument per cell, and conservatively: a tie between two collinear segments is not a
    /// crease and is rejected anyway, which only shrinks the sample.
    fn away_from_creases(beam: BeamKind, cell_mm: f64) -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        let n = 17;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let u = Point3::new(
                        (i as f64 + 0.5) / n as f64,
                        (j as f64 + 0.5) / n as f64,
                        (k as f64 + 0.5) / n as f64,
                    );
                    let (first, second) = nearest_two(beam, u);
                    // Far enough from the axis to be outside the strut — the radius below is
                    // 0.1 of a cell — and far enough from the runner-up that the nearest strut
                    // is not about to change hands. The margin is 0.03 of a cell against a
                    // difference step of 5e-6, so there is no chance of straddling a crease.
                    //
                    // These were 0.15 and 0.05, which suited the two sparse cells and rejected
                    // *every* point of the two dense ones — FCC's whole octant is within 0.49 of
                    // a strut where cubic's reaches 0.69. The count assertion below is what
                    // caught it rather than a silently empty sample passing.
                    if first > 0.12 && second - first > 0.03 {
                        out.push(u * cell_mm);
                    }
                }
            }
        }
        out
    }

    /// A beam lattice on an even rectangular map is a *true distance*, not a lower bound.
    ///
    /// **The claim the whole beam family rests on, and the project notes' rule made concrete.** Every
    /// lattice before this reported `LowerBound`: a TPMS formula has the right zero set and the
    /// wrong magnitude, so its value is divided by a gradient bound and lands at a fraction of
    /// the real distance. A beam cell is a union of capsules, and a capsule is an exact signed
    /// distance — so this one can be blended exactly, and traces without the 1.13–5.05 penalty
    /// the surfaces carry.
    ///
    /// If this ever fails, the honest repair is to report `LowerBound` and lose the exact blend.
    /// Loosening the tolerance would leave `approximate_blends` staying quiet about a blend that
    /// is not, in fact, the radius the user typed — which is the failure D8 exists to name.
    #[test]
    fn a_rectangular_beam_lattice_carries_a_true_distance() {
        for beam in BeamKind::ALL {
            let cell = 20.0;
            let lattice = Lattice {
                cell: CellKind::Beam(beam),
                map: LatticeMap::Rectangular {
                    cell: Vector3::new(cell, cell, cell),
                },
                phase: Phase::Strut { radius_mm: 2.0 },
            };
            assert_eq!(
                lattice.normalization(),
                Normalization::Distance,
                "{}",
                beam.label()
            );
            let points = away_from_creases(beam, cell);
            assert!(points.len() > 40, "{}: too few probes", beam.label());
            let worst = crate::gradient_deviation(&lattice, &points, 1e-4);
            assert!(
                worst < 1e-3,
                "{}: unit gradient off by {worst} over {} points",
                beam.label(),
                points.len(),
            );
        }
    }

    /// A curved map makes the same cell a lower bound, and it has to say so.
    ///
    /// The conversion out of cell space divides by the largest local frequency, which is equality
    /// only when the three agree. Under a cylindrical map they do not, so the value understates —
    /// safe to trace, and no longer a radius a blend can be trusted with.
    #[test]
    fn a_curved_beam_lattice_is_only_a_bound() {
        for beam in BeamKind::ALL {
            let lattice = Lattice {
                cell: CellKind::Beam(beam),
                map: LatticeMap::Cylindrical {
                    cell_radius: 10.0,
                    cell_height: 10.0,
                    arc_count: 16,
                },
                phase: Phase::Strut { radius_mm: 1.0 },
            };
            assert_eq!(
                lattice.normalization(),
                Normalization::LowerBound,
                "{}",
                beam.label()
            );
        }
    }

    /// A beam lattice is neither empty nor solid, under every map.
    #[test]
    fn a_beam_lattice_has_both_strut_and_void() {
        for beam in BeamKind::ALL {
            for map in [
                LatticeMap::Rectangular {
                    cell: Vector3::new(20.0, 20.0, 20.0),
                },
                LatticeMap::Cylindrical {
                    cell_radius: 10.0,
                    cell_height: 10.0,
                    arc_count: 16,
                },
            ] {
                let lattice = Lattice {
                    cell: CellKind::Beam(beam),
                    map,
                    phase: Phase::Strut { radius_mm: 1.5 },
                };
                let (mut material, mut void) = (0, 0);
                for p in cell_probes(12) {
                    let at = Point3::new(20.0 + p.x * 3.0, 20.0 + p.y * 3.0, p.z * 3.0);
                    if lattice.distance(at) < 0.0 {
                        material += 1;
                    } else {
                        void += 1;
                    }
                }
                assert!(material > 0, "{}: no material anywhere", beam.label());
                assert!(void > 0, "{}: no void anywhere", beam.label());
            }
        }
    }

    /// The value never overstates the room before a strut, under every map.
    ///
    /// The property that makes any of this safe to trace, asked of the family whose arithmetic
    /// reaches it by a different route: a frequency division rather than a gradient bound.
    #[test]
    fn a_beam_lattice_never_overstates_the_room_before_a_strut() {
        for beam in BeamKind::ALL {
            for map in [
                LatticeMap::Rectangular {
                    cell: Vector3::new(20.0, 20.0, 20.0),
                },
                LatticeMap::Cylindrical {
                    cell_radius: 10.0,
                    cell_height: 10.0,
                    arc_count: 16,
                },
                LatticeMap::Spherical {
                    cell_radius: 10.0,
                    azimuth_count: 16,
                    polar_count: 8,
                },
            ] {
                let lattice = Lattice {
                    cell: CellKind::Beam(beam),
                    map,
                    phase: Phase::Strut { radius_mm: 1.5 },
                };
                for p in cell_probes(9) {
                    let from = Point3::new(18.0 + p.x * 4.0, 13.0 + p.y * 4.0, p.z * 4.0);
                    let claimed = lattice.distance(from);
                    if claimed <= 0.0 {
                        continue;
                    }
                    let steps = 32;
                    for step in 1..=steps {
                        let t = claimed * step as f64 / steps as f64;
                        let at = Point3::new(from.x + t, from.y, from.z);
                        assert!(
                            lattice.distance(at) > -1e-9,
                            "{}: crossed a strut {t} into a claimed {claimed}",
                            beam.label(),
                        );
                    }
                }
            }
        }
    }

    /// Every cell's three arguments are equally steep, which is what the `/√3` split assumes.
    ///
    /// [`TpmsKind::argument_gradient`] splits the whole gradient evenly between the three
    /// arguments. That is exact only for a cell whose partials have the same distribution, and
    /// every cell here has one — the fully symmetric ones by any permutation, the cyclic ones
    /// (gyroid, Fischer-Koch S) by `x → y → z → x`, which is enough.
    ///
    /// **Asserted rather than assumed, because this is what a beam cell will break.** Struts
    /// differ per axis, so its three would not agree, and the shortcut would mis-size its walls
    /// under a curved map — quietly, since the pattern still looks like a lattice. This failing
    /// is the signal to make `argument_gradient` measure per axis rather than divide, which is
    /// why it already returns a vector.
    #[test]
    fn every_kinds_arguments_are_equally_steep() {
        for kind in TpmsKind::ALL {
            let mut sums = [0.0f64; 3];
            let mut count = 0usize;
            for p in cell_probes(40) {
                // At the surface, which is where the split is used.
                if kind.f(p).abs() >= 0.05 {
                    continue;
                }
                let g = kind.gradient(p);
                sums[0] += g.x * g.x;
                sums[1] += g.y * g.y;
                sums[2] += g.z * g.z;
                count += 1;
            }
            assert!(count > 0, "{}: found no surface to sample", kind.label());
            let rms = sums.map(|sum| (sum / count as f64).sqrt());
            let hi = rms[0].max(rms[1]).max(rms[2]);
            let lo = rms[0].min(rms[1]).min(rms[2]);
            assert!(
                (hi - lo) / hi < 0.05,
                "{}: the three arguments differ — RMS {rms:?}, spread {:.1}%",
                kind.label(),
                100.0 * (hi - lo) / hi,
            );
        }
    }

    /// A 2 mm wall comes out between the two numbers its doc comment promises.
    ///
    /// **The acceptance test `tpms-infill-spec.md` §14 asked for and never got.** A thickness is
    /// converted to an isovalue through an *average* gradient, so the wall is nominal: thinner
    /// where the surface is steeper, thicker where it is flatter. `tpms_profile.rs` measures how
    /// much, and the table on [`TpmsKind::surface_gradient`] records it. Until now nothing
    /// *asserted* it — the example printed a number and a person read it.
    ///
    /// Measured the honest way, not off the field's value: the value is a lower bound and says
    /// nothing about thickness. From each point inside the wall, march along the gradient — the
    /// direction across the slab — until the field turns positive, bisect the crossing, and do
    /// the same the other way. The chord is the wall's normal thickness there, wherever in the
    /// wall the point sat.
    ///
    /// The band is restated here from the doc table on purpose. Deriving it from the same
    /// constants would only prove the arithmetic agrees with itself; this proves the *table* is
    /// still true of the *code*, so a retuned `surface_gradient` or a changed formula fails here
    /// as "the doc is stale" rather than drifting unread. Every cell is measured before any is
    /// judged, so a stale table names all of its stale rows at once.
    #[test]
    fn a_wall_comes_out_within_the_band_its_doc_comment_promises() {
        let asked = 2.0;
        let cell = 20.0;
        let tau = std::f64::consts::TAU;
        let mut rows = Vec::new();
        for kind in [
            TpmsKind::Gyroid,
            TpmsKind::SchwarzP,
            TpmsKind::SchwarzD,
            TpmsKind::SchoenIwp,
            TpmsKind::Neovius,
            TpmsKind::FischerKochS,
        ] {
            // The band the panel states, scaled to the wall asked for here: one set of numbers,
            // said on the hover text and held to the code by this test.
            let (low, high) = {
                let (low, high) = kind.wall_band();
                (low * asked, high * asked)
            };
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Rectangular {
                    cell: Vector3::new(cell, cell, cell),
                },
                phase: Phase::sheet(asked),
            };
            let mut measured = Vec::new();
            for p in cell_probes(28) {
                let at = Point3::from(p.coords * cell / tau);
                // Well inside the wall, so both marches start from material. Half the isovalue
                // rather than a fixed number, because Neovius's F runs to ±13 where the gyroid's
                // runs to ±1.5, and one threshold would keep thousands of one and dozens of the
                // other.
                if kind.f(p).abs() > 0.5 * lattice.isovalue_at(at) {
                    continue;
                }
                // Across the slab. The map is uniform, so a direction in cell coordinates is the
                // same direction in millimetres.
                let Some(dir) = kind.gradient(p).try_normalize(1e-9) else {
                    continue;
                };
                let crossing = |sign: f64| -> Option<f64> {
                    let step = 0.05;
                    let mut t = 0.0;
                    while lattice.distance(at + dir * (sign * t)) < 0.0 {
                        t += step;
                        if t > cell {
                            return None;
                        }
                    }
                    let (mut inside, mut outside) = (t - step, t);
                    for _ in 0..30 {
                        let mid = 0.5 * (inside + outside);
                        if lattice.distance(at + dir * (sign * mid)) < 0.0 {
                            inside = mid;
                        } else {
                            outside = mid;
                        }
                    }
                    Some(0.5 * (inside + outside))
                };
                if let (Some(a), Some(b)) = (crossing(1.0), crossing(-1.0)) {
                    measured.push(a + b);
                }
            }
            measured.sort_by(f64::total_cmp);
            let n = measured.len();
            assert!(n > 100, "{}: only {n} points inside the wall", kind.label());
            let (p10, median, p90) = (measured[n / 10], measured[n / 2], measured[n * 9 / 10]);
            println!(
                "{:>15}  asked {asked:.1}  p10 {p10:.3}  median {median:.3}  p90 {p90:.3}  \
                 doc {low:.2}-{high:.2}  ({n} points)",
                kind.label(),
            );
            rows.push((kind, p10, median, p90, low, high));
        }

        let mut stale = Vec::new();
        for (kind, p10, median, p90, low, high) in rows {
            // The probe grid is fixed, so these numbers reproduce exactly; the margin only has to
            // absorb a harmless change to the march step or the bisection depth.
            if !(p10 >= low * 0.97 && p90 <= high * 1.03) {
                stale.push(format!(
                    "{}: a {asked} mm wall came out {p10:.3}-{p90:.3} at the 10th-90th, and the \
                     doc table promises {low:.2}-{high:.2}",
                    kind.label()
                ));
            }
            // The median is printed and recorded in the doc table but not asserted: every cell's
            // sits above the number asked for, by a per-cell amount that is a property of its
            // gradient's skew, and a threshold here would only restate the table.
            let _ = median;
        }
        assert!(stale.is_empty(), "the doc table is stale:\n{}", stale.join("\n"));
    }

    /// **What a walled sheet is for** (backlog item 48): a plain sheet divides a cell into two
    /// passages of the same size, and moving the wall's middle off the cell's surface makes one
    /// larger than the other without changing how much wall there is. Counted rather than
    /// derived: the fractions of a cell that are wall, passage above it and passage below it.
    #[test]
    fn an_offset_wall_makes_its_two_passages_differ_and_keeps_its_thickness() {
        let cell = 20.0;
        let tau = std::f64::consts::TAU;
        let kind = TpmsKind::Gyroid;
        let walled = |offset_mm: f64| Lattice {
            cell: CellKind::Tpms(kind),
            map: LatticeMap::Rectangular { cell: Vector3::new(cell, cell, cell) },
            phase: Phase::Sheet { thickness_mm: 2.0, offset_mm },
        };
        let census = |lattice: &Lattice| {
            let n = 36;
            let (mut wall, mut above, mut below) = (0usize, 0usize, 0usize);
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let at = Point3::new(
                            (i as f64 + 0.5) * cell / n as f64,
                            (j as f64 + 0.5) * cell / n as f64,
                            (k as f64 + 0.5) * cell / n as f64,
                        );
                        let f = kind.f(Point3::from(at.coords * tau / cell));
                        if lattice.distance(at) < 0.0 {
                            wall += 1;
                        } else if f > lattice.midsurface_at(at) {
                            above += 1;
                        } else {
                            below += 1;
                        }
                    }
                }
            }
            let total = (n * n * n) as f64;
            (wall as f64 / total, above as f64 / total, below as f64 / total)
        };

        let (wall0, above0, below0) = census(&walled(0.0));
        let (wall1, above1, below1) = census(&walled(1.5));
        println!("plain   wall {wall0:.3}  above {above0:.3}  below {below0:.3}");
        println!("+1.5 mm wall {wall1:.3}  above {above1:.3}  below {below1:.3}");
        assert!((above0 - below0).abs() < 0.02, "a gyroid's own passages are a pair");
        assert!(
            below1 - above1 > 0.10,
            "the wall moved toward the positive side, so that passage shrank: {above1} vs {below1}"
        );
        assert!(
            (wall1 - wall0).abs() < 0.15 * wall0,
            "about as much wall as before: {wall0} then {wall1}"
        );
    }

    /// The contract, with the wall off its surface: the gradient of `|F - c|` is the gradient
    /// of `F`, so the bound that held for a plain sheet holds for a walled one.
    #[test]
    fn an_offset_wall_never_overstates_the_room_before_its_material() {
        let cell = 20.0;
        for kind in [TpmsKind::Gyroid, TpmsKind::Neovius] {
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Rectangular { cell: Vector3::new(cell, cell, cell) },
                phase: Phase::Sheet { thickness_mm: 1.5, offset_mm: 1.2 },
            };
            let n = 12;
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let from = Point3::new(
                            cell * i as f64 / n as f64,
                            cell * j as f64 / n as f64,
                            cell * k as f64 / n as f64,
                        );
                        let claimed = lattice.distance(from);
                        if claimed <= 0.0 {
                            continue;
                        }
                        for dir in [Vector3::x(), Vector3::new(0.5, -0.7, 0.5).normalize()] {
                            for step in 1..=20 {
                                let t = claimed * step as f64 / 20.0;
                                assert!(
                                    lattice.distance(from + dir * t) > -1e-9,
                                    "{}: crossed the wall {t} into a claimed {claimed} at {from:?}",
                                    kind.label()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The chords across a sheet's wall, from every probe well inside it: march along the
    /// cell's gradient both ways to the wall's two faces. Sorted. A probe whose march finds no
    /// far side within a cell is counted rather than dropped, because that is what a wall that
    /// has met its neighbour looks like.
    fn typed_wall_chords(kind: TpmsKind, cell: f64, asked: f64) -> (Vec<f64>, usize) {
        let tau = std::f64::consts::TAU;
        let lattice = Lattice {
            cell: CellKind::Tpms(kind),
            map: LatticeMap::Rectangular { cell: Vector3::new(cell, cell, cell) },
            phase: Phase::sheet(asked),
        };
        let (mut chords, mut lost) = (Vec::new(), 0usize);
        for p in cell_probes(20) {
            let at = Point3::from(p.coords * cell / tau);
            if kind.f(p).abs() > 0.5 * lattice.isovalue_at(at) {
                continue;
            }
            let Some(dir) = kind.gradient(p).try_normalize(1e-9) else { continue };
            let crossing = |sign: f64| -> Option<f64> {
                let step = 0.05;
                let mut t = 0.0;
                while lattice.distance(at + dir * (sign * t)) < 0.0 {
                    t += step;
                    if t > cell {
                        return None;
                    }
                }
                Some(t)
            };
            match (crossing(1.0), crossing(-1.0)) {
                (Some(a), Some(b)) => chords.push(a + b),
                _ => lost += 1,
            }
        }
        chords.sort_by(f64::total_cmp);
        (chords, lost)
    }

    /// **`wall_limit` is where the walls meet, and a quarter past it they have.** Every surface
    /// is measured before any is judged, so a stale table names all of its stale rows at once.
    #[test]
    fn a_wall_past_its_cells_limit_has_met_its_neighbours() {
        let cell = 20.0;
        let merged = |kind: TpmsKind, ratio: f64| {
            let asked = cell * ratio;
            let (chords, lost) = typed_wall_chords(kind, cell, asked);
            let n = (chords.len() + lost).max(1);
            let fat = chords.iter().filter(|c| **c > 2.0 * asked).count() + lost;
            fat as f64 / n as f64
        };
        let mut stale = Vec::new();
        for kind in TpmsKind::ALL {
            let limit = kind.wall_limit();
            let (at, past) = (merged(kind, limit), merged(kind, limit * 1.25));
            println!(
                "{:>15}  limit {:>4.0}%  merged there {:>5.1}%  a quarter past it {:>5.1}%",
                kind.label(),
                limit * 100.0,
                at * 100.0,
                past * 100.0
            );
            if at > 0.10 || past <= 0.10 {
                stale.push(format!(
                    "{}: {:.1}% merged at its limit of {:.0}%, {:.1}% a quarter past it",
                    kind.label(),
                    at * 100.0,
                    limit * 100.0,
                    past * 100.0
                ));
            }
        }
        assert!(stale.is_empty(), "`wall_limit` is stale:\n{}", stale.join("\n"));
    }

    fn honeycomb(across: f64, wall: f64) -> Lattice {
        Lattice {
            cell: CellKind::Prism(PrismKind::Honeycomb),
            // As the compile arm builds it: the period follows the cell's own shape.
            map: LatticeMap::Rectangular {
                cell: Vector3::new(across, across, across)
                    .component_mul(&PrismKind::Honeycomb.natural_aspect()),
            },
            phase: Phase::sheet(wall),
        }
    }

    /// **The wall is as thick as it was asked to be, in every direction a wall faces.** From a
    /// hexagon's centre, along each of the three wall normals: clear until half a cell less
    /// half a wall, material for exactly one wall, clear again. No surface cell can say this;
    /// theirs is nominal.
    #[test]
    fn a_honeycomb_wall_is_exactly_as_thick_as_it_was_asked_to_be() {
        let (across, wall) = (12.0, 1.5);
        let lattice = honeycomb(across, wall);
        let third = std::f64::consts::FRAC_PI_3;
        // Both sublattices of centres, and a centre a few periods out.
        for centre in [
            Point3::new(0.0, 0.0, 3.0),
            Point3::new(3f64.sqrt() * across * 0.5, across * 0.5, -7.0),
            Point3::new(3f64.sqrt() * across * 4.0, across * 3.0, 0.0),
        ] {
            assert!(
                (lattice.distance(centre) - (across - wall) * 0.5).abs() < 1e-9,
                "a hexagon's centre is half a cell less half a wall from material"
            );
            for k in 0..6 {
                // Flats face up and down, so the wall normals sit at 90 degrees and every 60.
                let angle = std::f64::consts::FRAC_PI_2 + third * k as f64;
                let dir = Vector3::new(angle.cos(), angle.sin(), 0.0);
                let at = |t: f64| lattice.distance(centre + dir * t);
                let (near, far) = ((across - wall) * 0.5, (across + wall) * 0.5);
                assert!(at(near - 1e-6) > 0.0 && at(near + 1e-6) < 0.0, "in at {near}");
                assert!(at(far - 1e-6) < 0.0 && at(far + 1e-6) > 0.0, "out at {far}");
                assert!((at(across * 0.5) + wall * 0.5).abs() < 1e-9, "deepest mid-wall");
            }
        }
    }

    /// The hard rule for a new source, and what lets a fillet against a honeycomb be exact: the
    /// gradient is one. Probed away from the creases, which for a honeycomb are the lines from
    /// each centre to its corners and the corners themselves.
    #[test]
    fn a_rectangular_honeycomb_carries_a_true_distance() {
        let (across, wall) = (12.0, 1.5);
        let lattice = honeycomb(across, wall);
        assert_eq!(lattice.normalization(), Normalization::Distance);
        assert_eq!(CellKind::Prism(PrismKind::Honeycomb).supported_bandings(), &[Banding::Sheet]);

        let mut points = Vec::new();
        let n = 23;
        for i in 0..n {
            for j in 0..n {
                let p = Point3::new(
                    3f64.sqrt() * across * (i as f64 + 0.37) / n as f64,
                    across * (j as f64 + 0.41) / n as f64,
                    1.7 * i as f64,
                );
                // A crease is where two walls are equally near: moving a little in any
                // in-plane direction changes which wins, and the distance with it by less
                // than the move. Keep the points where one wall clearly wins.
                let h = 0.05;
                let here = PrismKind::Honeycomb.wall_distance(p.x, p.y, across);
                let smooth = [(h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)].iter().all(|(dx, dy)| {
                    let there = PrismKind::Honeycomb.wall_distance(p.x + dx, p.y + dy, across);
                    let second = PrismKind::Honeycomb.wall_distance(p.x - dx, p.y - dy, across);
                    ((there - here) + (second - here)).abs() < 1e-9
                });
                if smooth && here > 0.2 {
                    points.push(p);
                }
            }
        }
        assert!(points.len() > 100, "too few probes: {}", points.len());
        let worst = crate::gradient_deviation(&lattice, &points, 1e-4);
        assert!(worst < 1e-3, "unit gradient off by {worst} over {} points", points.len());
    }

    /// A prism: nothing changes along its axis, and the pattern repeats across it with the
    /// period its own shape gives -- the flat-to-flat width one way, root three of them the other.
    #[test]
    fn a_honeycomb_is_constant_along_its_axis_and_repeats_with_its_own_period() {
        let (across, wall) = (12.0, 1.5);
        let lattice = honeycomb(across, wall);
        let n = 9;
        for i in 0..n {
            for j in 0..n {
                let p = Point3::new(2.3 * i as f64 - 7.0, 1.9 * j as f64 - 5.0, 0.0);
                let here = lattice.distance(p);
                for z in [-40.0, 3.3, 125.0] {
                    assert_eq!(lattice.distance(Point3::new(p.x, p.y, z)), here);
                }
                let along = Vector3::new(3f64.sqrt() * across, 0.0, 0.0);
                let up = Vector3::new(0.0, across, 0.0);
                assert!((lattice.distance(p + along) - here).abs() < 1e-9);
                assert!((lattice.distance(p + up) - here).abs() < 1e-9);
                assert!((lattice.distance(p + (along + up) * 0.5) - here).abs() < 1e-9);
            }
        }
    }

    /// A density asked for is the density that comes out, within what the table costs.
    ///
    /// **The property the whole solid phase rests on**, and the one a stale or mis-transcribed
    /// table breaks silently: the geometry still looks like a lattice, it is merely not the
    /// amount of material anybody asked for. Measured the way the table was built — count the
    /// samples that are material — so it fails if the table and the formula stop agreeing, which
    /// is the case with no other symptom.
    ///
    /// The tolerance is the interpolation error `tpms_profile.rs` reports -- under one point *in
    /// absolute terms* across this band, which is up to 2.1% relative (Neovius typed 0.35 builds
    /// 0.343; see [`TpmsKind::solid_density_built`] for what each typed number builds) -- plus the
    /// sampling error of a grid this size.
    #[test]
    fn a_solid_fills_the_density_it_was_asked_for() {
        for kind in TpmsKind::ALL {
            for asked in [0.2, 0.35, 0.5, 0.65, 0.8] {
                let lattice = Lattice {
                    cell: CellKind::Tpms(kind),
                    map: LatticeMap::Rectangular {
                        cell: Vector3::new(20.0, 20.0, 20.0),
                    },
                    phase: Phase::Solid { density: asked },
                };
                // One whole cell in millimetres. A rectangular map scales linearly, so an even
                // grid in millimetres is an even grid in cell coordinates and the count is a
                // volume fraction with no weighting needed.
                let n = 40;
                let mut material = 0;
                for i in 0..n {
                    for j in 0..n {
                        for l in 0..n {
                            let at = Point3::new(
                                (i as f64 + 0.5) * 20.0 / n as f64,
                                (j as f64 + 0.5) * 20.0 / n as f64,
                                (l as f64 + 0.5) * 20.0 / n as f64,
                            );
                            if lattice.distance(at) < 0.0 {
                                material += 1;
                            }
                        }
                    }
                }
                let achieved = material as f64 / (n * n * n) as f64;
                assert!(
                    (achieved - asked).abs() < 0.03,
                    "{}: asked {asked}, got {achieved}",
                    kind.label(),
                );
            }
        }
    }

    /// A solid's outward direction is the gradient unturned.
    ///
    /// **The one thing the phase can get wrong without looking wrong.** A sheet's normal is the
    /// gradient of `|F|` and carries a sign flip; a labyrinth has one side and must not. With
    /// the flip left in, half the surface shades inside-out — which reads as a lighting bug a
    /// long way from the cause, so it is checked against the field's own slope rather than left
    /// to a picture.
    #[test]
    fn a_solids_normal_is_not_turned_around() {
        for kind in TpmsKind::ALL {
            let lattice = Lattice {
                cell: CellKind::Tpms(kind),
                map: LatticeMap::Rectangular {
                    cell: Vector3::new(20.0, 20.0, 20.0),
                },
                phase: Phase::Solid { density: 0.5 },
            };
            let h = 1e-4;
            let mut checked = 0;
            for p in cell_probes(7) {
                let at = Point3::new(p.x * 3.18, p.y * 3.18, p.z * 3.18);
                let Some(n) = lattice.normal(at) else {
                    continue;
                };
                // The field rises outward, so stepping along the reported normal must increase
                // it. A flipped normal makes this negative over the half of the cell where F is
                // negative, which is precisely the half a sheet's flip exists for.
                let slope = lattice.distance(at + n * h) - lattice.distance(at - n * h);
                assert!(
                    slope > 0.0,
                    "{}: the normal points into the material at {at:?}",
                    kind.label(),
                );
                checked += 1;
            }
            assert!(checked > 0, "{}: no normals to check", kind.label());
        }
    }

    /// The two phases select different material, in the direction claimed.
    ///
    /// Deep inside a labyrinth — far from the surface in `F` — a solid is material and a sheet
    /// is empty. If the two agreed everywhere, the `.abs()` would not be doing anything and this
    /// whole addition would be a no-op that still passed every other test here.
    #[test]
    fn a_sheet_is_empty_where_a_solid_is_full() {
        for kind in TpmsKind::ALL {
            let map = LatticeMap::Rectangular {
                cell: Vector3::new(20.0, 20.0, 20.0),
            };
            let sheet = Lattice {
                cell: CellKind::Tpms(kind),
                map: map.clone(),
                phase: Phase::sheet(1.0),
            };
            let solid = Lattice {
                cell: CellKind::Tpms(kind),
                map,
                phase: Phase::Solid { density: 0.5 },
            };
            let mut deep = 0;
            for p in cell_probes(10) {
                let at = Point3::new(p.x * 3.18, p.y * 3.18, p.z * 3.18);
                if solid.distance(at) < 0.0 && sheet.distance(at) > 0.0 {
                    deep += 1;
                }
            }
            assert!(
                deep > 0,
                "{}: nowhere is inside the labyrinth and outside the wall",
                kind.label(),
            );
        }
    }

    // ------------------------------------------------------------------ Voronoi, Track A

    /// A rectangular Voronoi lattice of the named kind.
    fn voronoi(kind: StochasticKind, cell: f64, jitter: f64, size: f64) -> Lattice {
        Lattice {
            cell: CellKind::Stochastic { kind, jitter, seed: 0x5EED_1234 },
            map: LatticeMap::Rectangular { cell: Vector3::new(cell, cell, cell) },
            phase: match kind {
                StochasticKind::VoronoiFoam => Phase::sheet(size),
                StochasticKind::VoronoiBeam => Phase::Strut { radius_mm: size },
            },
        }
    }

    /// The three smallest distances found over a ring wide enough that nothing outside it can
    /// matter, with no clamp -- the truth the cheap path is checked against.
    fn reference_seeds(u: Point3<f64>, jitter: f64, seed: u32) -> [Seed; 3] {
        nearest_seeds(u, jitter, seed, [None; 3], 4)
    }

    /// Points spread over a few cells, off every lattice symmetry.
    fn voronoi_probes(n: usize) -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    out.push(Point3::new(
                        -1.3 + 3.7 * (i as f64 + 0.317) / n as f64,
                        0.41 + 4.1 * (j as f64 + 0.233) / n as f64,
                        -2.7 + 3.3 * (k as f64 + 0.451) / n as f64,
                    ));
                }
            }
        }
        out
    }

    /// The hash is the thing two devices have to agree on exactly, and a table is the only way
    /// to say so that a shader can be held to. `voronoi-lattice-spec.md` section 4.1 disqualifies
    /// the float-sine idiom on determinism grounds; this is PCG3D instead, and these values were
    /// produced by an independent implementation of the same paper rather than by running the
    /// code below -- so the table is a check on the transcription and not a photograph of it.
    ///
    /// `prelude.wgsl`'s `voronoi_hash3` is the same arithmetic in `u32`, which wraps in WGSL as
    /// it does here, and `fieldform-gpu/examples/voronoi_probe.rs` is what holds the two
    /// together: it runs the shader's own hash on the device against this one.
    #[test]
    fn a_hash_is_the_same_number_wherever_it_is_computed() {
        let table: [([i32; 3], u32, [u32; 3]); 9] = [
            ([0, 0, 0], 0, [2611992518, 2833812075, 1058359340]),
            ([1, 0, 0], 0, [2717828630, 1424287207, 2065551680]),
            ([0, 1, 0], 0, [450040476, 1322785807, 1953739034]),
            ([0, 0, 1], 0, [2136848594, 1759552940, 990954100]),
            ([-1, -1, -1], 0, [2784268096, 2756918915, 1364953698]),
            ([0, 0, 0], 1, [2736486018, 2700751892, 2793548944]),
            ([0, 0, 0], 2654435769, [2514007664, 2563295925, 803329826]),
            ([7, -3, 11], 12345, [3869433177, 2001005170, 3385663864]),
            ([2147483647, -2147483648, 0], 4294967295, [3429542796, 992995500, 2892729091]),
        ];
        for (c, seed, want) in table {
            assert_eq!(voronoi_hash3(c, seed), want, "hash3({c:?}, {seed})");
        }
        // Changing one lane of the index has to change all three outputs, or a re-roll would
        // shift the pattern along one axis instead of rearranging it.
        let a = voronoi_hash3([4, 9, 2], 77);
        let b = voronoi_hash3([5, 9, 2], 77);
        let c = voronoi_hash3([4, 9, 2], 78);
        for k in 0..3 {
            assert_ne!(a[k], b[k], "lane {k} ignored a neighbouring cell");
            assert_ne!(a[k], c[k], "lane {k} ignored the seed");
        }
    }

    /// **The property every bound below rests on.** A seed displaced by more than half a cell
    /// would land in a neighbour's, and then a ring search that skipped that neighbour could
    /// miss the true nearest seed -- which is the failure section 4.3 is about.
    #[test]
    fn a_seed_never_leaves_the_cell_it_belongs_to() {
        // Out of range as well as in it: `with_scatter` clamps, and `prelude.wgsl`'s
        // `voronoi_measure` clamps the same way, because above one a seed leaves its own cell
        // and the ring search can miss the true nearest.
        for jitter in [-0.4, 0.0, 0.25, 0.5, 0.75, 1.0, 1.7] {
            let jitter = Lattice {
                cell: CellKind::Stochastic {
                    kind: StochasticKind::VoronoiFoam,
                    jitter: 0.0,
                    seed: 0,
                },
                map: LatticeMap::Rectangular { cell: Vector3::new(1.0, 1.0, 1.0) },
                phase: Phase::sheet(0.1),
            }
            .with_scatter(jitter, 0)
            .cell
            .stochastic_jitter()
            .expect("a stochastic cell");
            for i in -3..4 {
                for j in -3..4 {
                    for k in -3..4 {
                        let u = Point3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5);
                        let own = nearest_seeds(u, jitter, 0xABCD, [None; 3], 0)[0];
                        for axis in 0..3 {
                            let off = (own.at[axis] - u[axis]).abs();
                            assert!(
                                off <= jitter * 0.5 + 1e-12,
                                "jitter {jitter}: axis {axis} moved {off} of a cell",
                            );
                        }
                    }
                }
            }
        }
    }

    /// **The ring radius is a correctness parameter.** One ring does not always hold the true
    /// nearest seed, and if it does not, the difference of order statistics can move either way
    /// -- a wall locally too thin, and a tracer through it. The clamp is what makes the cheap
    /// path sound, and this is the brute-force check `beam-lattice-spec.md` section 4 asks for
    /// by the same name: the cheap answer against an unfolded reference, at every jitter.
    #[test]
    fn the_clamped_ring_never_overstates_a_search_wide_enough_to_be_the_truth() {
        for kind in StochasticKind::ALL {
            for jitter in [0.0, 0.3, 0.464, 0.5, 0.8, 1.0] {
                let mut worst: f64 = 0.0;
                for u in voronoi_probes(11) {
                    let (cheap, _) = voronoi_measure(kind, u, jitter, 0x5EED_1234, [None; 3]);
                    let truth = reference_seeds(u, jitter, 0x5EED_1234);
                    let exact = (truth[kind.order()].distance - truth[0].distance) * 0.5;
                    assert!(
                        cheap <= exact + 1e-12,
                        "{}: jitter {jitter} at {u:?} claimed {cheap} of a true {exact}",
                        kind.label(),
                    );
                    worst = worst.max(exact - cheap);
                }
                // Below the threshold section 4.3 derives, one ring provably holds the true
                // nearest seed, so the clamp cannot be biting and the two must agree exactly.
                if jitter <= (3.0 - 3f64.sqrt()) / (1.0 + 3f64.sqrt()) {
                    assert!(
                        worst < 1e-12,
                        "{}: jitter {jitter} is below the threshold and still lost {worst}",
                        kind.label(),
                    );
                }
            }
        }
    }

    /// Points spread finely enough to land in the 0.08% of volume where one ring cannot answer.
    fn dense_probes(n: usize) -> Vec<Point3<f64>> {
        let mut out = Vec::with_capacity(n * n * n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    out.push(Point3::new(
                        -2.7 + 5.3 * (i as f64 + 0.317) / n as f64,
                        1.3 + 4.9 * (j as f64 + 0.233) / n as f64,
                        -4.1 + 5.7 * (k as f64 + 0.451) / n as f64,
                    ));
                }
            }
        }
        out
    }

    /// **The surface where one ring stops answering, and the step that is not there any more.**
    ///
    /// §4.3 prescribed widening to `R = 2` and repeating where `d_1` exceeds the ceiling. The
    /// ceiling is `R + (1 - jitter)/2`, so widening raises it by a whole cell and the value
    /// steps *upward* across `{d_1 = l}` -- nought on one side and the unclamped truth on the
    /// other. At jitter 1 a 10 mm foam with a 1 mm wall read -0.49 mm one side of that surface
    /// and +0.52 mm an epsilon away: half a millimetre of room reported beside material, which
    /// is a ray straight through it. The ceiling is fixed at `l_1` now and the value is
    /// continuous.
    ///
    /// **The surface has to be hunted for rather than stumbled upon**, and that is why every
    /// other test here missed it: it is 0.08% of the volume at jitter 1 and does not exist at
    /// all below about 0.8. So this finds a point the ring cannot answer, walks from it straight
    /// at its own nearest seed -- which lowers `d_1` at exactly unit rate -- bisects onto the
    /// crossing, and compares the two sides.
    #[test]
    fn a_measure_does_not_step_across_the_surface_where_one_ring_stops_answering() {
        let seed = 0x5EED_1234;
        let jitter = 1.0;
        let ceiling = 1.0 + (1.0 - jitter) * 0.5;
        let d1 = |u: Point3<f64>| nearest_seeds(u, jitter, seed, [None; 3], 1)[0];
        for kind in StochasticKind::ALL {
            let (mut crossings, mut worst) = (0usize, 0.0f64);
            for u in dense_probes(40) {
                let start = d1(u);
                if start.distance <= ceiling {
                    continue;
                }
                // Straight at the nearest seed: `d_1` falls one for one, so the crossing is
                // inside this segment and a bisection finds it.
                let toward = (start.at - u) / start.distance;
                let (mut lo, mut hi) = (0.0f64, start.distance);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if d1(u + toward * mid).distance > ceiling {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let step = hi - lo;
                if step > 1e-9 {
                    continue;
                }
                let (a, _) = voronoi_measure(kind, u + toward * lo, jitter, seed, [None; 3]);
                let (b, _) = voronoi_measure(kind, u + toward * hi, jitter, seed, [None; 3]);
                crossings += 1;
                worst = worst.max((a - b).abs());
            }
            assert!(
                crossings > 20,
                "{}: only {crossings} crossings found, so this tested nothing",
                kind.label(),
            );
            // Either side of a crossing narrower than a nanometre, so anything a tracer would
            // notice is orders of magnitude above this.
            assert!(
                worst < 1e-6,
                "{}: the value stepped {worst} across the ceiling, over {crossings} crossings",
                kind.label(),
            );
        }
    }

    /// The measure advances at most a cell unit per cell unit travelled, which is the whole of
    /// what `gradient_bound` of one claims -- and what lets a tracer step by it. Checked as a
    /// finite difference rather than argued, and the analytic gradient checked against the same
    /// difference, since `normal` returns the analytic one.
    #[test]
    fn the_measure_never_moves_faster_than_the_point_does() {
        let h = 1e-6;
        for kind in StochasticKind::ALL {
            for jitter in [0.0, 0.5, 1.0] {
                for u in voronoi_probes(7) {
                    let (here, g) = voronoi_measure(kind, u, jitter, 0x5EED_1234, [None; 3]);
                    assert!(g.norm() <= 1.0 + 1e-9, "gradient {} at {u:?}", g.norm());
                    for axis in 0..3 {
                        let mut step = Vector3::zeros();
                        step[axis] = h;
                        let (ahead, _) = voronoi_measure(kind, u + step, jitter, 0x5EED_1234, [None; 3]);
                        let (behind, _) = voronoi_measure(kind, u - step, jitter, 0x5EED_1234, [None; 3]);
                        assert!(
                            (ahead - here).abs() <= h + 1e-12 && (behind - here).abs() <= h + 1e-12,
                            "{}: moved faster than the point at {u:?}",
                            kind.label(),
                        );
                        // Away from the wall complex, where the order statistics do not swap,
                        // the analytic gradient is the difference quotient.
                        let numeric = (ahead - behind) / (2.0 * h);
                        if (ahead - here).abs() > 1e-9 && (behind - here).abs() > 1e-9 {
                            assert!(
                                (numeric - g[axis]).abs() < 1e-4,
                                "{}: analytic {} against numeric {numeric} at {u:?}",
                                kind.label(),
                                g[axis],
                            );
                        }
                    }
                }
            }
        }
    }

    /// **The thickness is at least the number typed, and that is the whole reason there is no
    /// slack constant.** `voronoi-lattice-spec.md` section 4.5 asked for a per-kind constant
    /// calibrated the way `tpms-infill-spec.md` calibrated `g_kind`, so the wall measured comes
    /// out the width asked for on average. Measurement said not to: the measure understates, so
    /// the wall is thicker in places and never thinner, and dividing out the *median* would put
    /// the thinnest places at half the number typed -- which for a minimum wall is the one
    /// direction that is not allowed to be wrong.
    ///
    /// `voronoi_profile` is where the whole table lives; this is the column it turns on.
    #[test]
    fn a_voronoi_wall_is_never_thinner_than_the_number_typed() {
        let (cell, typed) = (10.0, 1.0);
        let step = 0.01;
        for jitter in [0.0, 0.5, 1.0] {
            let lattice = voronoi(StochasticKind::VoronoiFoam, cell, jitter, typed);
            let mut thinnest = f64::INFINITY;
            let mut crossings = 0usize;
            for i in 0..6 {
                for j in 0..6 {
                    let from = Point3::new(-7.0 + 3.1 * i as f64, -5.0 + 2.9 * j as f64, -23.37);
                    let mut entered: Option<f64> = None;
                    // Read at the start rather than assumed: a ray that begins inside material
                    // would otherwise record its first exit as a whole crossing, and that
                    // truncated chord is shorter than any wall. It read 0.15 mm of a 1 mm wall.
                    let mut was = lattice.distance(from) < 0.0;
                    let mut t = 0.0;
                    while t < 47.0 {
                        t += step;
                        let inside = lattice.distance(Point3::new(from.x, from.y, from.z + t)) < 0.0;
                        if inside && !was {
                            entered = Some(t);
                        } else if !inside && was && let Some(a) = entered.take() {
                            thinnest = thinnest.min(t - a);
                            crossings += 1;
                        }
                        was = inside;
                    }
                }
            }
            assert!(crossings > 100, "jitter {jitter}: only {crossings} crossings to measure");
            assert!(
                thinnest > typed - 2.0 * step,
                "jitter {jitter}: a wall came out {thinnest} mm where {typed} was typed",
            );
        }
    }

    /// **The normal points the way the value falls, under every map** -- which is a different
    /// claim from `every_kinds_gradient_matches_a_difference_of_its_value` one level down, and
    /// the one that was not being made.
    ///
    /// `normal` takes its gradient in the map's own basis and has to turn it back into the
    /// frame's, and for a curved map that rotation depends on where the point is. Every cell
    /// goes through the same `match self.map` at the bottom of the function to get it. A cell
    /// that answers early and does the rectangular arm's arithmetic itself is correct on a
    /// rectangular map and silently wrong on the other two -- which is what the Voronoi arm did
    /// when it was first written, and nothing here noticed: the viewport shades from the G-buffer
    /// and `gpu_matches_cpu` holds the shader's normal against the field's *numeric* gradient
    /// rather than against this.
    ///
    /// Creases are skipped rather than tolerated: a difference of order statistics is genuinely
    /// undefined on the wall complex, and a beam's distance is on its own medial axis.
    #[test]
    fn a_normal_points_the_way_the_value_falls_under_every_map() {
        let cell = 10.0;
        let rectangular = LatticeMap::Rectangular { cell: Vector3::new(cell, cell, cell) };
        let cylindrical = LatticeMap::Cylindrical {
            cell_radius: cell,
            cell_height: cell,
            arc_count: 16,
        };
        let spherical = LatticeMap::Spherical {
            cell_radius: cell,
            azimuth_count: 16,
            polar_count: 8,
        };
        let cells: [(CellKind, Phase); 4] = [
            (CellKind::Tpms(TpmsKind::Gyroid), Phase::sheet(1.0)),
            (CellKind::Beam(BeamKind::Bcc), Phase::Strut { radius_mm: 0.8 }),
            (
                CellKind::Stochastic {
                    kind: StochasticKind::VoronoiFoam,
                    jitter: 0.8,
                    seed: 0x5EED_1234,
                },
                Phase::sheet(1.0),
            ),
            (
                CellKind::Stochastic {
                    kind: StochasticKind::VoronoiBeam,
                    jitter: 0.8,
                    seed: 0x5EED_1234,
                },
                Phase::Strut { radius_mm: 0.8 },
            ),
        ];
        for (kind, phase) in cells {
            for map in [rectangular.clone(), cylindrical.clone(), spherical.clone()] {
                let lattice = Lattice { cell: kind, map: map.clone(), phase };
                let h = 1e-4;
                let (mut worst, mut probes) = (0.0f64, 0usize);
                for i in 0..9 {
                    for j in 0..9 {
                        for k in 0..5 {
                            let p = Point3::new(
                                31.0 + 2.17 * i as f64,
                                -13.0 + 1.93 * j as f64,
                                4.0 + 2.71 * k as f64,
                            );
                            let at = |o: Vector3<f64>| lattice.distance(p + o);
                            let axis = |o: Vector3<f64>| (at(o) - at(-o)) / (2.0 * h);
                            let numeric = Vector3::new(
                                axis(Vector3::new(h, 0.0, 0.0)),
                                axis(Vector3::new(0.0, h, 0.0)),
                                axis(Vector3::new(0.0, 0.0, h)),
                            );
                            // A crease: the one-sided differences disagree, so there is no
                            // gradient to compare against and the assertion would be about the
                            // probe rather than about the code.
                            let here = lattice.distance(p);
                            let smooth = [
                                Vector3::new(h, 0.0, 0.0),
                                Vector3::new(0.0, h, 0.0),
                                Vector3::new(0.0, 0.0, h),
                            ]
                            .iter()
                            .all(|o| ((at(*o) - here) + (at(-*o) - here)).abs() < 1e-7);
                            if !smooth || numeric.norm() < 0.5 {
                                continue;
                            }
                            let Some(normal) = lattice.normal(p) else { continue };
                            probes += 1;
                            worst = worst.max(
                                normal.angle(&numeric.normalize()).to_degrees(),
                            );
                        }
                    }
                }
                assert!(probes > 40, "{}: only {probes} usable probes", kind.label());
                assert!(
                    worst < 2.0,
                    "{} on {}: the normal is {worst:.1} degrees off the way the value falls",
                    kind.label(),
                    match map {
                        LatticeMap::Rectangular { .. } => "a rectangular map",
                        LatticeMap::Cylindrical { .. } => "a cylindrical map",
                        _ => "a spherical map",
                    },
                );
            }
        }
    }

    /// A Voronoi cell is a lower bound under every map, with no configuration that makes it
    /// exact -- section 4.4. So a blend against one is approximate wherever it appears, and the
    /// panel says so; the arm reporting honestly is what makes that happen.
    #[test]
    fn a_voronoi_cell_is_a_bound_under_every_map() {
        for kind in StochasticKind::ALL {
            let even = voronoi(kind, 10.0, 0.7, 1.0);
            assert_eq!(even.normalization(), Normalization::LowerBound);
            let curved = Lattice {
                map: LatticeMap::Cylindrical {
                    cell_radius: 10.0,
                    cell_height: 10.0,
                    arc_count: 12,
                },
                ..even.clone()
            };
            assert_eq!(curved.normalization(), Normalization::LowerBound);
            assert_eq!(CellKind::Stochastic { kind, jitter: 0.7, seed: 1 }.gradient_bound(), 1.0);
        }
    }

    /// **The seam a wrapped map would otherwise tear.** A cylindrical map's angular argument
    /// closes after `arc_count` cells, and a Voronoi cell hashes an absolute index -- so without
    /// the wrap the cell just below nought and the cell just below the count draw different
    /// seeds, the walls do not meet, and the field jumps. A jump is a ray stepping through a
    /// wall, not merely a mark on the surface.
    #[test]
    fn a_wrapped_map_closes_on_itself_rather_than_tearing() {
        let lattice = Lattice {
            cell: CellKind::Stochastic {
                kind: StochasticKind::VoronoiFoam,
                jitter: 1.0,
                seed: 0x5EED_1234,
            },
            map: LatticeMap::Cylindrical {
                cell_radius: 9.0,
                cell_height: 9.0,
                arc_count: 16,
            },
            phase: Phase::sheet(1.2),
        };
        assert_eq!(lattice.map.cell_period(), [None, Some(16), None]);
        // Either side of the seam at theta = pi, and either side of an ordinary cell edge for
        // contrast: the step across the seam must be the step across any other place.
        let step = 1e-4;
        for radius in [14.0, 31.0, 57.0] {
            for z in [-8.0, 0.35, 12.7] {
                let at = |theta: f64| {
                    Point3::new(radius * f64::cos(theta), radius * f64::sin(theta), z)
                };
                let seam = (lattice.distance(at(std::f64::consts::PI - step))
                    - lattice.distance(at(-std::f64::consts::PI + step)))
                .abs();
                assert!(
                    seam < 2.0 * step * radius + 1e-6,
                    "the seam at r = {radius}, z = {z} jumped by {seam}",
                );
            }
        }
    }

    // ---- Item 94: what a lattice's relative density is, held to the field it describes ----

    use std::f64::consts::{FRAC_PI_2, PI};

    /// The fraction of one period that `Lattice::distance` calls material, at the first `n`
    /// points of the sequence the fill tables were built from.
    ///
    /// **The same points, not merely as many**: `tpms_profile.rs` and `beam_profile.rs` print what
    /// the first 2²⁰ of their own 2²⁴ read against each table row these tests check, so a
    /// tolerance here is twice a measured residual rather than a guess, and a table that drifts
    /// from the field fails by more than the sampling ever could.
    fn field_fill(lattice: &Lattice, period: Vector3<f64>, n: u64) -> f64 {
        use rayon::prelude::*;
        let inside: u64 = (0..n)
            .into_par_iter()
            .map(|i| {
                let u = r3_point(i);
                let p = Point3::new(u.x * period.x, u.y * period.y, u.z * period.z);
                u64::from(lattice.distance(p) <= 0.0)
            })
            .sum();
        inside as f64 / n as f64
    }

    const FILL_POINTS: u64 = 1 << 20;
    /// The thinnest rows, where the fewest points land, are checked on four times as many.
    const THIN_POINTS: u64 = 1 << 22;

    fn cubic_period(cell_mm: f64) -> Vector3<f64> {
        Vector3::new(cell_mm, cell_mm, cell_mm)
    }

    /// `value` at the first `n` points of the tables' own sequence, sorted: what holds **every**
    /// pasted row to the measurement, cheaply, where `field_fill` holds three rows a table to the
    /// field itself. Between them a mis-pasted row and a conversion that drifted from the field
    /// both fail.
    fn sorted_sample(n: u64, value: impl Fn(Point3<f64>) -> f64 + Sync) -> Vec<f64> {
        use rayon::prelude::*;
        let mut out: Vec<f64> = (0..n).into_par_iter().map(|i| value(r3_point(i))).collect();
        out.par_sort_unstable_by(f64::total_cmp);
        out
    }

    fn fraction_at_or_below(sorted: &[f64], at: f64) -> f64 {
        sorted.partition_point(|v| *v <= at) as f64 / sorted.len() as f64
    }

    fn rectangular(cell: CellKind, phase: Phase, cell_mm: f64) -> Lattice {
        Lattice { cell, map: LatticeMap::Rectangular { cell: cubic_period(cell_mm) }, phase }
    }

    /// **The density a solid's typed number actually builds**, read off the field rather than
    /// asserted. Replaces a design that read a solid at the number typed: below a tenth the two
    /// differ by up to two thirds, so that test would have asserted an identity the geometry does
    /// not have.
    ///
    /// Two holds, each at twice what `tpms_profile.rs` measured. Through `Lattice::distance` at
    /// typed 0.05, 0.50 and 0.95: at most 0.065% off on the first 2²⁰ points. Every one of the
    /// 101 rows against the formula on the same points: at most 0.145% where the fill is at least
    /// 0.02, and 5.7e-5 absolute below.
    #[test]
    fn the_solid_density_table_is_the_fraction_the_field_fills() {
        for kind in TpmsKind::ALL {
            for typed in [0.05, 0.50, 0.95] {
                let lattice =
                    rectangular(CellKind::Tpms(kind), Phase::Solid { density: typed }, 10.0);
                let measured = field_fill(&lattice, cubic_period(10.0), FILL_POINTS);
                let table = kind.solid_density_built(typed);
                assert!(
                    (measured - table).abs() / table < 0.0015,
                    "{} typed {typed}: the field fills {measured}, the table says {table}",
                    kind.label(),
                );
            }
            let sample =
                sorted_sample(FILL_POINTS, |u| kind.f(Point3::from(u.coords * std::f64::consts::TAU)));
            let close = |read: f64, table: f64| {
                let off = (read - table).abs();
                if table >= 0.02 { off / table < 0.003 } else { off < 1.2e-4 }
            };
            for k in 0..=100 {
                let typed = k as f64 / 100.0;
                let table = kind.solid_density_built(typed);
                let read = fraction_at_or_below(&sample, kind.isovalue_for_density(typed));
                assert!(
                    close(read, table),
                    "{} row {k}: the formula reads {read}, the table says {table}",
                    kind.label(),
                );
            }
            // The floor's fill is said to a person ("it joins up above about 0.21 built") and
            // bounds the inverse, so it is held to the same points as every row.
            let (saddle, fill) = kind.solid_floor();
            let read = fraction_at_or_below(&sample, saddle);
            assert!(
                close(read, fill),
                "{}: the field fills {read} at its saddle {saddle}, the floor says {fill}",
                kind.label(),
            );
        }
    }

    /// Whether some component of `{F ≤ t}` wraps one period, on an `n`-cubed grid.
    ///
    /// **Lifted labels**: each voxel reached carries the whole-period offset it was reached at,
    /// and a component wraps when it reaches one voxel at two different offsets. A plain periodic
    /// flood cannot tell wrapping from touching the boundary.
    fn spans(kind: TpmsKind, t: f64, n: usize) -> bool {
        let h = std::f64::consts::TAU / n as f64;
        let at = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
        let mut solid = vec![false; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let p = Point3::new(
                        (i as f64 + 0.5) * h,
                        (j as f64 + 0.5) * h,
                        (k as f64 + 0.5) * h,
                    );
                    solid[at(i, j, k)] = kind.f(p) <= t;
                }
            }
        }
        let mut offset: Vec<Option<[i32; 3]>> = vec![None; n * n * n];
        let mut queue = std::collections::VecDeque::new();
        for start in 0..n * n * n {
            if !solid[start] || offset[start].is_some() {
                continue;
            }
            offset[start] = Some([0, 0, 0]);
            queue.push_back(start);
            while let Some(c) = queue.pop_front() {
                let here = [(c / (n * n)) as i32, ((c / n) % n) as i32, (c % n) as i32];
                let lift = offset[c].expect("queued voxels are labelled");
                for (axis, step) in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)] {
                    let (mut there, mut lifted) = (here, lift);
                    there[axis] += step;
                    if there[axis] < 0 {
                        there[axis] += n as i32;
                        lifted[axis] -= 1;
                    } else if there[axis] >= n as i32 {
                        there[axis] -= n as i32;
                        lifted[axis] += 1;
                    }
                    let next = at(there[0] as usize, there[1] as usize, there[2] as usize);
                    if !solid[next] {
                        continue;
                    }
                    match offset[next] {
                        Some(seen) if seen != lifted => return true,
                        Some(_) => {}
                        None => {
                            offset[next] = Some(lifted);
                            queue.push_back(next);
                        }
                    }
                }
            }
        }
        false
    }

    /// **Below its saddle a solid is blobs**, and the floor is pinned either side of each kind's
    /// own. A 40³ grid finds spanning a little above a true saddle, since a neck must be a voxel
    /// wide -- 0.027 at most, Schoen IWP's, measured by `check94.py` and again by
    /// `tpms_profile.rs` at 64³ -- so 0.05 either side clears it with room.
    #[test]
    fn a_solid_below_its_saddle_is_separate_pieces() {
        for kind in TpmsKind::ALL {
            let (floor, fill) = kind.solid_floor();
            assert!(
                !spans(kind, floor - 0.05, 40),
                "{}: already one piece below its floor {floor}",
                kind.label()
            );
            assert!(
                spans(kind, floor + 0.05, 40),
                "{}: still separate pieces above its floor {floor}",
                kind.label()
            );
            // And the saddle is a critical point of the cell, at the value the floor claims --
            // checked where the closed forms put one.
            let saddle_at = match kind {
                TpmsKind::SchwarzP => Some(Point3::new(PI, PI, 0.0)),
                TpmsKind::SchoenIwp => Some(Point3::new(PI, 0.0, FRAC_PI_2)),
                TpmsKind::Neovius => Some(Point3::new(PI, (0.75f64).acos(), (0.75f64).acos())),
                _ => None,
            };
            if let Some(p) = saddle_at {
                assert!((kind.f(p) - floor).abs() < 1e-12, "{}: F there {}", kind.label(), kind.f(p));
                assert!(kind.gradient(p).norm() < 1e-12, "{}: not a critical point", kind.label());
            }
            // The fill there is a fill.
            assert!(fill > 0.0 && fill < 0.5, "{}: {fill}", kind.label());
        }
    }

    /// The sheet table, held two ways at twice what `tpms_profile.rs` measured. Through
    /// `Lattice::distance` at the first, middle and last rows: the first -- a 1% wall, where the
    /// fewest points land -- on the first 2²² points, at most 0.096% off; the others on 2²⁰, at
    /// most 0.027%. Every row against the formula on the first 2²⁰ points: at most 0.34% at the
    /// first row (Schwarz P) and 0.090% at any other.
    #[test]
    fn the_sheet_density_table_is_the_fraction_the_field_fills() {
        for kind in TpmsKind::ALL {
            let rows = (kind.wall_limit() * 100.0).round() as usize;
            let middle = (rows / 2 + 1) as f64 / 100.0;
            for (w, points, tolerance) in [
                (0.01, THIN_POINTS, 0.002),
                (middle, FILL_POINTS, 0.0006),
                (kind.wall_limit(), FILL_POINTS, 0.0006),
            ] {
                let lattice = rectangular(CellKind::Tpms(kind), Phase::sheet(w * 10.0), 10.0);
                let measured = field_fill(&lattice, cubic_period(10.0), points);
                let table = kind.sheet_density(w).expect("inside the wall limit");
                assert!(
                    (measured - table).abs() / table < tolerance,
                    "{} at w {w}: the field fills {measured}, the table says {table}",
                    kind.label(),
                );
            }
            let g = kind.surface_gradient();
            let sample = sorted_sample(FILL_POINTS, |u| {
                kind.f(Point3::from(u.coords * std::f64::consts::TAU)).abs()
            });
            for i in 1..=rows {
                let w = i as f64 / 100.0;
                let table = kind.sheet_density(w).expect("inside the wall limit");
                let read = fraction_at_or_below(&sample, PI * g * w);
                let tolerance = if i == 1 { 0.007 } else { 0.002 };
                assert!(
                    (read - table).abs() / table < tolerance,
                    "{} row {i}: the formula reads {read}, the table says {table}",
                    kind.label(),
                );
            }
        }
    }

    /// The strut tables, held two ways at twice what `beam_profile.rs` measured. Through
    /// `Lattice::distance` at the first, middle and last rows -- the last at the merge radius
    /// itself: the first, r/L 0.01, on the first 2²² points, at most 0.31% off (diamond); the
    /// others on 2²⁰, at most 0.009%. Every row against the axis distance on the first 2²⁰
    /// points: at most 0.78% at r/L 0.01 (diamond) and 0.146% from 0.02.
    #[test]
    fn the_strut_density_table_is_the_fraction_the_field_fills() {
        for (beam, middle) in [
            (BeamKind::Bcc, 0.18),
            (BeamKind::Fcc, 0.15),
            (BeamKind::Octet, 0.13),
            (BeamKind::Diamond, 0.11),
        ] {
            for (r, points, tolerance) in [
                (0.01, THIN_POINTS, 0.0065),
                (middle, FILL_POINTS, 0.0002),
                (beam.merge_radius(), FILL_POINTS, 0.0002),
            ] {
                let lattice =
                    rectangular(CellKind::Beam(beam), Phase::Strut { radius_mm: r * 10.0 }, 10.0);
                let measured = field_fill(&lattice, cubic_period(10.0), points);
                let table = beam.strut_density(r).expect("inside the merge radius");
                assert!(
                    (measured - table).abs() / table < tolerance,
                    "{} at r/L {r}: the field fills {measured}, the table says {table}",
                    beam.label(),
                );
            }
            let sample = sorted_sample(FILL_POINTS, |u| beam.axis_distance(u));
            let hundredths = (beam.merge_radius() * 100.0 + 1e-9).floor() as usize;
            let radii = (1..=hundredths)
                .map(|k| k as f64 / 100.0)
                .chain(std::iter::once(beam.merge_radius()));
            for (k, r) in radii.enumerate() {
                let table = beam.strut_density(r).expect("inside the merge radius");
                let read = fraction_at_or_below(&sample, r);
                let tolerance = if k == 0 { 0.016 } else { 0.003 };
                assert!(
                    (read - table).abs() / table < tolerance,
                    "{} at r/L {r}: the axis distance reads {read}, the table says {table}",
                    beam.label(),
                );
            }
        }
    }

    /// **The control arm**: the one beam cell whose fill is known in closed form, sampled the way
    /// every table was. It should show no difference, and `beam_profile.rs` measured at most
    /// 0.070% at these four radii on the first 2²⁰ points.
    #[test]
    fn a_cubic_strut_fills_its_closed_form() {
        for r in [0.02, 0.10, 0.30, 0.50] {
            let lattice = rectangular(
                CellKind::Beam(BeamKind::Cubic),
                Phase::Strut { radius_mm: r * 10.0 },
                10.0,
            );
            let measured = field_fill(&lattice, cubic_period(10.0), FILL_POINTS);
            let exact = 3.0 * PI * r * r - 8.0 * 2f64.sqrt() * r * r * r;
            assert_eq!(BeamKind::Cubic.strut_density(r), Some(exact));
            assert!(
                (measured - exact).abs() / exact < 0.0015,
                "r/L {r}: the field fills {measured}, the closed form says {exact}",
            );
        }
    }

    /// The closed form peaks at `π/(4√2) ≈ 0.555` and falls to 0.74 at 0.7, where the cell is
    /// full: past half a cell it is not a fill at all, so it is not quoted.
    #[test]
    fn a_cubic_strut_past_half_a_cell_is_not_quoted() {
        let cubic = CellKind::Beam(BeamKind::Cubic);
        let at_half = uniform_density(cubic, Phase::Strut { radius_mm: 5.0 }, 10.0);
        let exact = 3.0 * PI * 0.25 - 8.0 * 2f64.sqrt() * 0.125;
        assert_eq!(at_half, RelativeDensity::Exact(exact));
        assert!((exact - 0.942).abs() < 1e-3, "{exact}");
        assert!(matches!(
            uniform_density(cubic, Phase::Strut { radius_mm: 5.1 }, 10.0),
            RelativeDensity::Unavailable(_)
        ));
        assert_eq!(BeamKind::Cubic.strut_density(0.51), None);
    }

    /// Closest distance between two segments, in closed form (Ericson, *Real-Time Collision
    /// Detection*, 5.1.9) -- independent of `point_to_segment`, which the folded distance uses.
    fn segment_distance(p1: Point3<f64>, q1: Point3<f64>, p2: Point3<f64>, q2: Point3<f64>) -> f64 {
        let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
        let (a, e, f) = (d1.dot(&d1), d2.dot(&d2), d2.dot(&r));
        let (c, b) = (d1.dot(&r), d1.dot(&d2));
        let denominator = a * e - b * b;
        let mut s = if denominator > 1e-14 {
            ((b * f - c * e) / denominator).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut t = (b * s + f) / e;
        if t < 0.0 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else if t > 1.0 {
            t = 1.0;
            s = ((b - c) / a).clamp(0.0, 1.0);
        }
        ((p1 + d1 * s) - (p2 + d2 * t)).norm()
    }

    /// **Exact, from the honest strut list**: every strut of the base cell against every strut in
    /// the 3×3×3 block round it, skipping pairs that share a node and pairs that cross (FCC's two
    /// diagonals on one face, which meet at its centre and overlap from the start, as struts at a
    /// node do). Half the smallest gap left is where struts with nothing in common first touch.
    #[test]
    fn the_merge_radius_is_half_the_nearest_gap_between_struts_that_do_not_meet() {
        for beam in BeamKind::ALL {
            let base = struts(beam);
            let mut block = Vec::new();
            for i in -1..=1 {
                for j in -1..=1 {
                    for k in -1..=1 {
                        let o = Vector3::new(i as f64, j as f64, k as f64);
                        block.extend(base.iter().map(|(a, b)| (a + o, b + o)));
                    }
                }
            }
            let same = |p: Point3<f64>, q: Point3<f64>| (p - q).norm() < 1e-9;
            let mut nearest = f64::INFINITY;
            for (a, b) in &base {
                for (c, d) in &block {
                    if same(*a, *c) || same(*a, *d) || same(*b, *c) || same(*b, *d) {
                        continue;
                    }
                    let gap = segment_distance(*a, *b, *c, *d);
                    if gap > 1e-9 {
                        nearest = nearest.min(gap);
                    }
                }
            }
            assert!(
                (nearest * 0.5 - beam.merge_radius()).abs() < 1e-12,
                "{}: half the nearest gap is {}, merge_radius says {}",
                beam.label(),
                nearest * 0.5,
                beam.merge_radius(),
            );
        }
    }

    /// The honeycomb's fill is exact, and this is it read off the field over exactly one period:
    /// root three flat-to-flat widths by one. `beam_profile.rs` measured the first 2²⁰ points at
    /// most 0.010% off at these walls.
    #[test]
    fn a_honeycomb_fills_one_minus_its_open_hexagon() {
        let across = 10.0;
        let period = Vector3::new(3f64.sqrt() * across, across, across);
        for w in [0.05, 0.15, 0.30] {
            let lattice = honeycomb(across, w * across);
            let measured = field_fill(&lattice, period, FILL_POINTS);
            let exact = 1.0 - (1.0 - w) * (1.0 - w);
            assert_eq!(
                uniform_density(lattice.cell, lattice.phase, across),
                RelativeDensity::Exact(exact)
            );
            assert!(
                (measured - exact).abs() / exact < 0.0002,
                "t/across {w}: the field fills {measured}, exact {exact}",
            );
        }
    }

    /// A thin strut is a cylinder: its fill is `π r²` times the strut length per cell, and the
    /// first two rows of each table, carried linearly to nought, land on that. `beam_profile.rs`
    /// measured within 0.12%; the tolerance is 1%.
    #[test]
    fn a_thin_strut_fills_pi_r_squared_per_unit_of_strut_length() {
        for beam in BeamKind::ALL {
            let coefficient = |r: f64| beam.strut_density(r).expect("a thin strut") / (r * r);
            let extrapolated = 2.0 * coefficient(0.01) - coefficient(0.02);
            let exact = PI * beam.strut_length_per_cell();
            assert!(
                (extrapolated - exact).abs() / exact < 0.01,
                "{}: {extrapolated} against pi l = {exact}",
                beam.label(),
            );
        }
    }

    /// **What makes the inverse sound**: a target density has one wall, one radius, one typed
    /// number, only if every table rises.
    #[test]
    fn the_fill_rises_with_the_wall_with_the_radius_and_with_the_density() {
        let rises = |name: &str, f: &dyn Fn(f64) -> Option<f64>, end: f64| {
            let mut last = f64::NEG_INFINITY;
            for i in 1..=1000 {
                let x = end * i as f64 / 1000.0;
                let v = f(x).unwrap_or_else(|| panic!("{name}: nothing at {x}"));
                assert!(v > last, "{name}: {v} at {x} after {last}");
                last = v;
            }
        };
        for kind in TpmsKind::ALL {
            rises(kind.label(), &|t| Some(kind.solid_density_built(t)), 1.0);
            rises(kind.label(), &|w| kind.sheet_density(w), kind.wall_limit());
        }
        for beam in BeamKind::ALL {
            rises(beam.label(), &|r| beam.strut_density(r), beam.merge_radius());
        }
        rises("honeycomb", &|w| PrismKind::Honeycomb.wall_density(w), 0.999);
    }

    /// Where no table reaches, the reason is said in the number's place -- and never a number.
    #[test]
    fn a_density_is_not_quoted_where_the_table_does_not_reach() {
        let gyroid = CellKind::Tpms(TpmsKind::Gyroid);
        let unavailable = |d: RelativeDensity| match d {
            RelativeDensity::Unavailable(reason) => !reason.trim().is_empty(),
            _ => false,
        };
        // An offset sheet divides the cell unequally; the sources measured centred ones.
        assert!(unavailable(uniform_density(
            gyroid,
            Phase::Sheet { thickness_mm: 1.0, offset_mm: 0.3 },
            10.0
        )));
        // Past the wall limit the walls have met.
        assert!(unavailable(uniform_density(gyroid, Phase::sheet(2.6), 10.0)));
        assert!(matches!(uniform_density(gyroid, Phase::sheet(2.5), 10.0), RelativeDensity::Tabled(_)));
        // Past the merge radius the struts have met their neighbours.
        for beam in BeamKind::ALL {
            let past = (beam.merge_radius() + 0.005) * 10.0;
            assert!(unavailable(uniform_density(CellKind::Beam(beam), Phase::Strut { radius_mm: past }, 10.0)));
        }
        // A honeycomb whose walls fill the cell.
        assert!(unavailable(uniform_density(
            CellKind::Prism(PrismKind::Honeycomb),
            Phase::sheet(10.0),
            10.0
        )));
        // A Voronoi cell at any jitter.
        for kind in StochasticKind::ALL {
            let cell = CellKind::Stochastic { kind, jitter: 0.4, seed: 3 };
            assert!(unavailable(uniform_density(cell, Phase::sheet(1.0), 10.0)));
            assert!(unavailable(uniform_density(cell, Phase::Strut { radius_mm: 0.5 }, 10.0)));
        }
        // A cell that does not take the phase -- the compile refuses these, and so does this.
        assert!(unavailable(uniform_density(gyroid, Phase::Strut { radius_mm: 0.5 }, 10.0)));
        assert!(unavailable(uniform_density(
            CellKind::Beam(BeamKind::Octet),
            Phase::Solid { density: 0.3 },
            10.0
        )));
        // And a solid below its saddle is not a density of a lattice at all.
        let neovius = uniform_density(
            CellKind::Tpms(TpmsKind::Neovius),
            Phase::Solid { density: 0.30 },
            10.0,
        );
        let RelativeDensity::Disconnected { built, joins_above } = neovius else {
            panic!("a Neovius solid at 0.30 is separate pieces, not {neovius:?}");
        };
        assert!(built < joins_above && (joins_above - 0.335).abs() < 0.001, "{built} {joins_above}");
        assert_eq!(neovius.value(), None);
        assert!(matches!(
            uniform_density(CellKind::Tpms(TpmsKind::Neovius), Phase::Solid { density: 0.40 }, 10.0),
            RelativeDensity::Tabled(_)
        ));
    }

    /// A NaN that reaches a table is read as row nought by a saturating cast, which would quote a
    /// density nobody typed; every non-finite or non-positive number is refused first.
    #[test]
    fn a_density_is_never_quoted_from_a_number_that_is_not_finite() {
        let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0];
        let refused = |d: RelativeDensity| matches!(d, RelativeDensity::Unavailable(_));
        let gyroid = CellKind::Tpms(TpmsKind::Gyroid);
        for x in bad {
            assert!(refused(uniform_density(gyroid, Phase::Solid { density: x }, 10.0)), "{x}");
            assert!(refused(uniform_density(gyroid, Phase::sheet(x), 10.0)), "{x}");
            // An offset of nought is a plain sheet; any other, finite or not, is refused.
            if x != 0.0 {
                assert!(
                    refused(uniform_density(
                        gyroid,
                        Phase::Sheet { thickness_mm: 1.0, offset_mm: x },
                        10.0
                    )),
                    "offset {x}"
                );
            }
            for beam in BeamKind::ALL {
                assert!(
                    refused(uniform_density(CellKind::Beam(beam), Phase::Strut { radius_mm: x }, 10.0)),
                    "{x}"
                );
            }
            assert!(refused(uniform_density(
                CellKind::Prism(PrismKind::Honeycomb),
                Phase::sheet(x),
                10.0
            )));
            assert!(refused(uniform_density(gyroid, Phase::sheet(1.0), x)), "cell {x}");
            assert!(refused(uniform_density(
                CellKind::Beam(BeamKind::Cubic),
                Phase::Strut { radius_mm: 0.5 },
                x
            )));
        }
        for kind in TpmsKind::ALL {
            assert!(kind.solid_density_built(f64::NAN).is_nan());
            assert_eq!(kind.sheet_density(f64::NAN), None);
        }
        for beam in BeamKind::ALL {
            assert_eq!(beam.strut_density(f64::NAN), None);
            assert_eq!(beam.strut_density(-0.01), None);
        }
    }

    /// **The compile builds the phase `phase_of` returns**: the compiled field and a lattice
    /// built from `phase_of`'s answer agree at every point, for every phase, typed or named. The
    /// readout's half of the chain -- that what it quotes is `uniform_density` of `phase_of`'s
    /// answer -- is the app's `the_density_quoted_is_the_density_of_the_phase_the_compile_builds`.
    #[test]
    fn the_compile_builds_the_phase_phase_of_returns() {
        use fieldform_scene::{CellMap, Document, Frame, Graph, LatticeCell, LatticePhase, Node, ParamRef};
        let mut document = Document::new();
        let named = ParamRef::Named(document.add_parameter("w", 0.8));
        let lit = ParamRef::Literal;
        let cases = [
            (LatticeCell::Gyroid, LatticePhase::Sheet { thickness: lit(1.2), offset: lit(0.3) }),
            (LatticeCell::Gyroid, LatticePhase::sheet(named)),
            (LatticeCell::SchwarzD, LatticePhase::Solid { density: lit(0.3) }),
            (LatticeCell::Octet, LatticePhase::Strut { radius: lit(0.6) }),
            (LatticeCell::Diamond, LatticePhase::Strut { radius: named }),
            (LatticeCell::Honeycomb, LatticePhase::sheet(lit(1.0))),
        ];
        let resolve = |r: &ParamRef| r.resolve(&|id| document.parameter_value(id));
        for (cell, phase) in cases {
            let mut graph = Graph::empty();
            graph.add(Node::Lattice {
                cell,
                map: CellMap::Rectangular { cell: lit(10.0) },
                phase,
                frame: Frame::default(),
            });
            let compiled = crate::GraphField::compile_graph(&document, &graph, 1e-3)
                .unwrap_or_else(|e| panic!("{cell:?} {phase:?}: {e:?}"));
            let read = phase_of(&phase, resolve, resolve).expect("every number resolves");
            let kind: CellKind = cell.into();
            let period = match kind {
                CellKind::Prism(prism) => cubic_period(10.0).component_mul(&prism.natural_aspect()),
                _ => cubic_period(10.0),
            };
            let lattice = Lattice { cell: kind, map: LatticeMap::Rectangular { cell: period }, phase: read };
            for i in 0..200 {
                let p = Point3::from(r3_point(i).coords * 23.0);
                let (a, b) = (compiled.distance(p), lattice.distance(p));
                assert!((a - b).abs() < 1e-12, "{cell:?} {phase:?} at {p:?}: compiled {a}, read {b}");
            }
        }
        // A number that does not resolve is said by name, never read as nought.
        let missing = ParamRef::Named(fieldform_scene::ParamId::new());
        assert_eq!(
            phase_of(&LatticePhase::sheet(missing), resolve, resolve),
            Err(Unresolved { which: "wall" })
        );
        assert_eq!(
            phase_of(&LatticePhase::Sheet { thickness: lit(1.0), offset: missing }, resolve, resolve),
            Err(Unresolved { which: "offset" })
        );
        assert_eq!(
            phase_of(&LatticePhase::Solid { density: missing }, resolve, resolve),
            Err(Unresolved { which: "density" })
        );
        assert_eq!(
            phase_of(&LatticePhase::Strut { radius: missing }, resolve, resolve),
            Err(Unresolved { which: "radius" })
        );
    }
