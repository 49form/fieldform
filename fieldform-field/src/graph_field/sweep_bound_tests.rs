// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A sweep's reading held to the contract where it has been caught claiming too much: a held
//! section's end caps, the tube's cap on held sections and moving laws, and a closed polyline
//! where its frame closes.
    use super::*;
    use fieldform_scene::{Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Section, SweepLaw};
    use nalgebra::{Point3, Rotation3, Unit, Vector3};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// A degree-one curve through `points`, put in the document.
    fn drawn(doc: &mut Document, points: Vec<Point3<f64>>, closed: bool) -> CurveId {
        let id = CurveId::new();
        doc.curves.insert(id, Curve::polyline(id, "line", points, closed).expect("a polyline"));
        id
    }

    /// The awkward placement every case here is built in: tipped about an axis on no plane of
    /// the world, and set well off the origin, so nothing measured is helped by an alignment.
    struct Placed {
        turn: Rotation3<f64>,
        origin: Point3<f64>,
    }

    impl Placed {
        fn awkward() -> Self {
            Self {
                turn: Rotation3::from_axis_angle(
                    &Unit::new_normalize(Vector3::new(0.3, -0.5, 0.8)),
                    0.9,
                ),
                origin: Point3::new(37.0, -21.0, 14.0),
            }
        }

        fn at(&self, local: Point3<f64>) -> Point3<f64> {
            self.origin + self.turn * local.coords
        }

        fn along(&self, local: Vector3<f64>) -> Vector3<f64> {
            self.turn * local
        }
    }

    /// A lens in the local `z = 0` plane: a chord of 40 along `x`, 8 thick at its middle,
    /// convex, the shape the owner's wing holds.
    fn lens(doc: &mut Document, placed: &Placed) -> CurveId {
        let (a, h) = (20.0, 4.0);
        let n = 24;
        let mut points = Vec::new();
        for i in 0..n {
            let x = -a + 2.0 * a * i as f64 / n as f64;
            points.push(placed.at(Point3::new(x, -h * (1.0 - (x / a).powi(2)), 0.0)));
        }
        for i in 0..n {
            let x = a - 2.0 * a * i as f64 / n as f64;
            points.push(placed.at(Point3::new(x, h * (1.0 - (x / a).powi(2)), 0.0)));
        }
        drawn(doc, points, true)
    }

    fn swept(doc: &Document, profile: CurveId, spine: CurveId, section: Section, law: SweepLaw) -> GraphField {
        let mut graph = Graph::empty();
        graph.add(Node::Sweep {
            rims: Default::default(),
            outline: Outline::closed(profile),
            spine,
            section,
            law,
        });
        GraphField::compile_graph(doc, &graph, 1e-3).expect("the sweep compiles")
    }

    /// The probe directions, with the placement's own axes added: a cap is missed most along
    /// its own normal, and that is not one of the world's.
    fn directions(placed: &Placed) -> Vec<Vector3<f64>> {
        let mut out = crate::probe_directions();
        for local in [Vector3::x(), Vector3::y(), Vector3::z()] {
            out.push(placed.along(local));
            out.push(-placed.along(local));
        }
        out
    }

    /// **A held section's end caps never claim more room than there is.** A lens held flush at
    /// sixty degrees to a straight spine, which is the owner's wing: the cap is the section's
    /// own plane, and a point a quarter of a millimetre past it is a quarter of a millimetre
    /// from the solid. The slab was read in millimetres of *arc*, which is the distance along
    /// the path, and along a path leaning sixty degrees off the cap's normal that is twice the
    /// distance across the cap: 0.25 read 0.5. Held parallel along the same path the cap is the
    /// same plane and the fault was the same.
    #[test]
    fn a_held_sections_end_caps_never_claim_more_room_than_there_is() {
        let placed = Placed::awkward();
        let lean = 60f64.to_radians();
        let t = Vector3::new(lean.sin(), 0.0, lean.cos());
        let length = 100.0;
        for section in [Section::Skewed, Section::Parallel] {
            let mut doc = Document::new();
            let profile = lens(&mut doc, &placed);
            let spine = drawn(
                &mut doc,
                vec![placed.at(Point3::origin()), placed.at(Point3::from(t * length))],
                false,
            );
            let field = swept(&doc, profile, spine, section, SweepLaw::default());
            // Square to the end cap, a quarter of a millimetre out from the middle of it.
            let end = Point3::from(t * length);
            let past = placed.at(end + Vector3::new(0.0, 0.0, 0.25));
            let read = field.distance(past);
            assert!(
                read <= 0.25 + 1e-9 && read > 0.2,
                "{section:?}: 0.25 mm past the end cap reads {read}"
            );
            let start = placed.at(Point3::new(3.0, 1.0, -0.25));
            let read = field.distance(start);
            assert!(read <= 0.25 + 1e-9, "{section:?}: 0.25 mm before the start cap reads {read}");

            let mut probes = Vec::new();
            for &x in &[-17.0, -9.0, 0.0, 6.0, 14.0, 23.0] {
                for &y in &[-5.0, -2.0, 0.0, 1.5, 4.5] {
                    for &s in &[0.25, 1.0, 3.0, 7.0] {
                        probes.push(placed.at(end + Vector3::new(x, y, s)));
                        probes.push(placed.at(Point3::new(x, y, -s)));
                    }
                }
            }
            let over = crate::overstatement(&field, &probes, &directions(&placed), 0.01, 1.3);
            assert!(over < 0.011, "{section:?}: the caps claimed {over} mm more room than there was");
        }
    }


    /// A spine leaning sixty degrees off the lens's plane and bending gently as it goes: an arc
    /// of radius 300 turning thirty degrees in the plane of the lean, so the lean eases from
    /// sixty degrees toward thirty along the way.
    fn leaning_bend(placed: &Placed) -> Vec<Point3<f64>> {
        let (radius, turn, start) = (300.0f64, 30f64.to_radians(), 60f64.to_radians());
        (0..=48)
            .map(|i| {
                let psi = turn * i as f64 / 48.0;
                placed.at(Point3::new(
                    radius * ((start - psi).cos() - start.cos()),
                    0.0,
                    radius * (start.sin() - (start - psi).sin()),
                ))
            })
            .collect()
    }

    /// Probes in a box about each end of `spine`, and about its middle, `reach` out.
    fn probes_about(spine: &[Point3<f64>], reach: f64) -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        let places = [spine[0], spine[spine.len() / 2], spine[spine.len() - 1]];
        for c in places {
            for i in 0..5 {
                for j in 0..5 {
                    for k in 0..5 {
                        let f = |n: i32| (n as f64 / 4.0 * 2.0 - 1.0) * reach;
                        // Off the lattice a little, so no probe sits on a plane of symmetry.
                        out.push(c + Vector3::new(f(i) + 0.37, f(j) - 0.21, f(k) + 0.13));
                    }
                }
            }
        }
        out
    }

    /// **And round a bend and under a law that moves the section, still not.** The same lens,
    /// held flush and parallel along a spine whose lean eases from sixty degrees to thirty, and
    /// held flush and square under a scale, a twist and a draft: probes about each cap and the
    /// middle, marched past every claim.
    #[test]
    fn a_held_or_moving_sections_caps_never_claim_more_room_round_a_bend() {
        let placed = Placed::awkward();
        let laws: Vec<(&str, SweepLaw)> = vec![
            ("the plain law", SweepLaw::default()),
            ("a scale to 0.6", SweepLaw {
                taper: fieldform_scene::Taper::Scale(lit(0.6)),
                ..SweepLaw::default()
            }),
            ("a twist of 40 degrees", SweepLaw { twist: lit(40.0), ..SweepLaw::default() }),
            ("a draft of 4 degrees", SweepLaw {
                taper: fieldform_scene::Taper::Draft(lit(4.0)),
                ..SweepLaw::default()
            }),
            ("a window of 10 to 85", SweepLaw {
                extent: [lit(10.0), lit(85.0)],
                ..SweepLaw::default()
            }),
        ];
        for section in [Section::Square, Section::Skewed, Section::Parallel] {
            for (what, law) in &laws {
                if section == Section::Parallel && what.starts_with("a twist") {
                    continue;
                }
                let mut doc = Document::new();
                let profile = lens(&mut doc, &placed);
                let points = leaning_bend(&placed);
                let spine = drawn(&mut doc, points.clone(), false);
                let field = swept(&doc, profile, spine, section.clone(), law.clone());
                let over = crate::overstatement(
                    &field,
                    &probes_about(&points, 30.0),
                    &directions(&placed),
                    0.05,
                    1.3,
                );
                assert!(over < 0.051, "{section:?} under {what} claimed {over} mm more room than there was");
            }
        }
    }

    /// **The shader learns how fast a mitred corner moves from the mitres it is handed**: one
    /// entry after them, the longest, which the sharp extrude multiplies its draft's slope by as
    /// the leaf does. And the step budget buys the same divisor. A 20 mm square's corners are right
    /// angles, so the entry is √2, and a 25° sharp draft divides by `√(1 + 2·tan² 25°)`.
    #[test]
    fn a_sharp_drafts_mitre_rate_reaches_the_shader_and_the_step_budget() {
        let mut doc = Document::new();
        let square = drawn(
            &mut doc,
            vec![
                Point3::new(-10.0, -10.0, 0.0),
                Point3::new(10.0, -10.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(-10.0, 10.0, 0.0),
            ],
            true,
        );
        let mut graph = Graph::empty();
        let id = graph.add(Node::Extrude {
            rims: Default::default(),
            outline: Outline::closed(square),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(100.0),
            draft: lit(25.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: fieldform_scene::Corners::Sharp,
        });
        let miters = crate::outline_miters(&doc, &graph, id, crate::PROFILE_TURN_DEGREES)
            .expect("a square has mitres");
        assert_eq!(miters.len(), 5, "four mitres and the longest after them");
        assert!((miters[4].x - std::f64::consts::SQRT_2).abs() < 1e-9, "{:?}", miters[4]);
        let tan = 25f64.to_radians().tan();
        let sharp = super::lipschitz::extrude_stretch(&doc, &graph, id);
        assert!((sharp - (1.0 + 2.0 * tan * tan).sqrt()).abs() < 1e-9, "sharp divides by {sharp}");
        if let Some(Node::Extrude { corners, .. }) = graph.node_mut(id) {
            *corners = fieldform_scene::Corners::Round;
        }
        let round = super::lipschitz::extrude_stretch(&doc, &graph, id);
        assert!((round - (1.0 + tan * tan).sqrt()).abs() < 1e-9, "round divides by {round}");
    }
