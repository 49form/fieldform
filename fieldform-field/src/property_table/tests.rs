// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The property table's tests: every row held to its source, its range and the ledger.

    use super::*;
    use super::rows::{
        BCC, CUBIC, DIAMOND, E, FCC, FISCHER_KOCH_S, GYROID, HONEYCOMB, NEOVIUS, OCTET, S,
        SCHOEN_IWP, SCHWARZ_D, SCHWARZ_P, VORONOI_BEAM,
    };

    fn row(id: &str) -> &'static PropertyRow {
        ROWS.iter().find(|r| r.id == id).unwrap_or_else(|| panic!("no row {id}"))
    }

    /// Every density a test sweeps: a thousand steps across `(0, 1]`.
    fn sweep() -> impl Iterator<Item = f64> {
        (1..=1000).map(|i| i as f64 / 1000.0)
    }

    /// A row the panel quotes must say where it came from, what kind of data it is, over what
    /// densities, and which way it was loaded -- and say what was corrected in it.
    #[test]
    fn every_shipped_row_names_its_source_its_data_its_range_and_its_direction() {
        assert_eq!(ROWS.len(), 48, "the shipped rows are the verified 53 less five");
        for r in ROWS {
            let s = r.source;
            for (what, text) in [("short", s.short), ("citation", s.citation), ("locator", s.locator)] {
                assert!(!text.trim().is_empty(), "{}: empty {what}", r.id);
            }
            assert!(
                s.link.starts_with("doi:") || s.link.starts_with("http"),
                "{}: link {}",
                r.id,
                s.link
            );
            let year = s.citation.as_bytes().windows(6).any(|w| {
                w[0] == b'('
                    && w[5] == b')'
                    && std::str::from_utf8(&w[1..5])
                        .ok()
                        .and_then(|y| y.parse::<u32>().ok())
                        .is_some_and(|y| (1900..2100).contains(&y))
            });
            assert!(year, "{}: no (yyyy) in the citation", r.id);
            match r.range {
                DensityRange::Span { lo: Lower::At(lo), hi } => {
                    assert!(0.0 <= lo && lo < hi && hi <= 1.0, "{}: span {lo}–{hi}", r.id)
                }
                DensityRange::Span { lo: Lower::NotStated { surely_from }, hi } => {
                    assert!(0.0 < surely_from && surely_from < hi && hi <= 1.0, "{}", r.id)
                }
                DensityRange::Single(at) => assert!(0.0 < at && at < 1.0, "{}: one point {at}", r.id),
                DensityRange::NotStated => {}
            }
            // Every row was corrected or qualified by its verification, and says how.
            assert!(!r.notes.trim().is_empty(), "{}: no notes", r.id);
            // A divisor that is said to be stated or implied has a number.
            if matches!(r.solid_basis, SolidBasis::Stated | SolidBasis::Implied) {
                assert!(r.solid_value_mpa.is_some(), "{}: a {:?} basis with no value", r.id, r.solid_basis);
            }
            assert!(!r.direction.phrase().is_empty() && !r.data.phrase().is_empty());
            assert!(!r.solid_basis.phrase().is_empty() && !r.material.is_empty());
        }
        let mut ids: Vec<&str> = ROWS.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ROWS.len(), "ids are unique");
    }

    /// A cube axis is a claim about how the cell sat in the specimen, which a source stating no
    /// direction -- or only its printer's build direction -- does not make.
    #[test]
    fn no_row_whose_source_does_not_state_a_direction_reads_as_a_cube_axis() {
        for direction in [LoadDirection::NotStated, LoadDirection::BuildDirection] {
            let phrase = direction.phrase();
            for forbidden in ["[100]", "[111]", "cube axis", "diagonal"] {
                assert!(!phrase.contains(forbidden), "{direction:?} says {forbidden}");
            }
        }
        for r in ROWS {
            if let Some(partner) = other_axis_row(r) {
                assert!(
                    matches!(partner.direction, LoadDirection::Axis100 | LoadDirection::Diagonal111),
                    "{} is paired with {}, which states no cube axis",
                    r.id,
                    partner.id
                );
            }
            if matches!(r.direction, LoadDirection::NotStated | LoadDirection::BuildDirection) {
                assert!(other_axis_row(r).is_none(), "{} has a partner axis", r.id);
            }
        }
    }

    /// We read these sources ourselves, so we know the densities they measured.
    #[test]
    fn a_row_read_at_its_source_states_its_density_range() {
        for r in ROWS.iter().filter(|r| r.provenance == Provenance::Primary) {
            assert_ne!(r.range, DensityRange::NotStated, "{} is primary with no range", r.id);
        }
    }

    /// A review's metadata is the review's claim, and the line says whose numbers it is quoting.
    #[test]
    fn a_quoted_row_says_via() {
        let quoted: Vec<_> =
            ROWS.iter().filter(|r| matches!(r.provenance, Provenance::Review { .. })).collect();
        assert_eq!(quoted.len(), 26, "22, 23 and 25–48 of the design's list");
        for r in quoted {
            assert!(r.source.short.contains(" via "), "{}: {}", r.id, r.source.short);
        }
    }

    /// **What makes the inverse sound**: bisection finds the one density only if the fit rises.
    #[test]
    fn every_fit_rises_with_density() {
        for r in ROWS {
            let mut last = f64::NEG_INFINITY;
            for rho in sweep() {
                let v = r.fit.at(rho);
                assert!(v > last, "{} falls at {rho}: {v} after {last}", r.id);
                last = v;
            }
        }
    }

    /// Every reading that is given inverts to the density it was read at. The trips are counted,
    /// so a change that refused nearly everything could not pass by having nothing to check:
    /// 48 rows at 91 densities is 4368, of which 245 are refused (the ledger's §4.3).
    #[test]
    fn the_inverse_returns_the_density_it_was_given() {
        let mut worst = 0.0f64;
        let mut trips = 0;
        for r in ROWS {
            for i in 5..=95 {
                let rho = i as f64 / 100.0;
                let Some(value) = r.read(rho).value else { continue };
                let back = r.density_for(value).unwrap_or_else(|| panic!("{} at {rho}", r.id));
                worst = worst.max((back - rho).abs());
                trips += 1;
            }
        }
        assert!(worst < 1e-9, "worst round trip {worst}");
        assert_eq!(trips, 4368 - 245, "round trips made");
    }

    /// `density_for` answers `None` on each of its documented branches: a target the fit never
    /// reaches, a density the fit reaches only where its reading would be refused, and a target
    /// that is no target.
    #[test]
    fn density_for_answers_none_where_the_fit_does_not_reach() {
        // Lu's Fischer-Koch S never falls below its offset, 0.0242, so a 2% target bisects down
        // to nothing, where the fit is above the solid's share and refused.
        let fks = row("lu20-fks-solid-e");
        assert_eq!(fks.density_for(0.02), None);
        assert_eq!(fks.read(1e-20).standing, Standing::Refused(Refusal::AboveSolidShare));
        // Lu's gyroid reaches 0.0001 only within its near-zero span, where it is refused.
        assert_eq!(row("lu20-g-solid-e").density_for(0.0001), None);
        for r in ROWS {
            let top = r.fit.at(1.0);
            assert_eq!(r.density_for(top * 1.01 + 1e-6), None, "{} above its top {top}", r.id);
            for target in [0.0, -0.05, f64::NAN, f64::INFINITY] {
                assert_eq!(r.density_for(target), None, "{} at {target}", r.id);
            }
        }
        // The control: a target inside what the fit gives is answered.
        assert!(fks.density_for(0.08).is_some_and(|rho| (fks.fit.at(rho) - 0.08).abs() < 1e-12));
    }

    /// The Voigt bound: no lattice is stiffer than its solid share. A fit that breaks it inside
    /// its own range was a reason to exclude a row (Libonati 2021), so none that ships may.
    #[test]
    fn no_modulus_row_claims_more_than_the_solid_s_share_inside_its_own_range() {
        for r in ROWS.iter().filter(|r| r.class() == PropertyClass::Modulus) {
            let (lo, hi) = match r.range {
                DensityRange::Span { lo: Lower::At(lo), hi } => (lo, hi),
                DensityRange::Span { lo: Lower::NotStated { surely_from }, hi } => (surely_from, hi),
                DensityRange::Single(_) | DensityRange::NotStated => (0.0, 1.0),
            };
            for rho in sweep().filter(|rho| (lo..=hi).contains(rho)) {
                assert!(r.fit.at(rho) <= rho, "{} at {rho}: {}", r.id, r.fit.at(rho));
            }
        }
    }

    /// Each refusal beside its control: the same row just on the other side of its crossing
    /// gives a value, so the refusal is where the crossing is and not everywhere.
    #[test]
    fn a_reading_beyond_the_solid_s_share_is_refused_not_shown() {
        for (id, refused_at, read_at) in [
            ("lu20-fks-solid-e", 0.01, 0.03),
            ("lu20-d-solid-e", 0.001, 0.003),
            ("wang22-cubic-strut-strength", 0.80, 0.70),
            ("sun22-g-sheet-strength", 0.60, 0.45),
            ("alabort19-neovius-solid-strength", 0.90, 0.80),
        ] {
            let reading = row(id).read(refused_at);
            assert_eq!(reading.standing, Standing::Refused(Refusal::AboveSolidShare), "{id} at {refused_at}");
            assert_eq!(reading.value, None, "{id}: a refused value is not given");
            let control = row(id).read(read_at);
            assert!(
                control.value.is_some_and(|v| v <= read_at)
                    && !matches!(control.standing, Standing::Refused(_)),
                "{id} at {read_at}, inside its crossing: {control:?}"
            );
        }
    }

    #[test]
    fn a_fit_that_falls_to_nought_is_refused_not_shown() {
        for (id, rho) in [("lu20-p-solid-e", 0.15), ("lu20-g-solid-e", 0.03)] {
            let reading = row(id).read(rho);
            assert_eq!(reading.standing, Standing::Refused(Refusal::NotPositive), "{id} at {rho}");
            assert_eq!(reading.value, None);
        }
        for rho in [0.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert_eq!(row("lu20-p-solid-e").read(rho).standing, Standing::Refused(Refusal::NoDensity));
        }
    }

    /// **Just above an offset fit's zero the value is noise**, not a reading: Lu's gyroid at a
    /// built 0.043 -- about what a typed 0.05 builds -- would read 0.0025% with a slope of 95, so a
    /// 1% density error moves it by 95%. Refused there, and read once the slope is under ten.
    #[test]
    fn a_fit_near_its_zero_is_refused_not_shown() {
        for (id, near, clear) in [("lu20-g-solid-e", 0.043, 0.05), ("lu20-p-solid-e", 0.19, 0.21)] {
            let r = row(id);
            assert!(r.fit.at(near) > 0.0, "{id} is above nought at {near}");
            assert!(r.fit.log_slope(near) > MOST_SENSITIVE, "{id}: {}", r.fit.log_slope(near));
            assert_eq!(r.read(near), Reading { value: None, standing: Standing::Refused(Refusal::NearZero) });
            assert!(r.fit.log_slope(clear) < MOST_SENSITIVE);
            assert!(r.read(clear).value.is_some(), "{id} at {clear}");
        }
        // The slope is the exponent for every power law, so none of those is ever refused for it.
        for r in ROWS {
            if let Fit::Power { n, .. } = r.fit {
                assert!(n < MOST_SENSITIVE && r.fit.log_slope(0.01) == n, "{}", r.id);
            }
        }
        // And it agrees with a numeric derivative, where the formula is not just the exponent.
        let fit = row("lu20-g-solid-e").fit;
        for rho in [0.05, 0.1, 0.3] {
            let h = 1e-6;
            let numeric = (fit.at(rho * (1.0 + h)).ln() - fit.at(rho * (1.0 - h)).ln())
                / ((1.0 + h).ln() - (1.0 - h).ln());
            assert!((numeric - fit.log_slope(rho)).abs() < 1e-5, "{rho}: {numeric} against {}", fit.log_slope(rho));
        }
    }

    /// Lu 2020's Schwarz P is 0.336 at 0.30 read without its offset and 0.117 with it, and the
    /// paper's own Fig 5a agrees with the second.
    #[test]
    fn an_offset_fit_is_never_read_without_its_offset() {
        let value = row("lu20-p-solid-e").read(0.30).value.expect("read at 0.30");
        assert!((value - 0.117).abs() < 0.001, "{value}");
    }

    #[test]
    fn the_two_branch_fit_is_continuous_at_its_knee_and_reaches_one_at_full_density() {
        for id in ["nx23-p-sheet-e", "nx23-g-sheet-e", "nx23-iwp-sheet-e"] {
            let Fit::TwoBranch { c1, n1, knee, c2, n2, c3 } = row(id).fit else {
                panic!("{id} is a two-branch fit");
            };
            let below = c1 * knee.powf(n1);
            let above = c2 * knee.powf(n2) + c3;
            assert!((below - above).abs() < 1e-12, "{id}: {below} against {above} at the knee");
            assert!((row(id).fit.at(1.0) - 1.0).abs() < 1e-12, "{id} at full density");
            assert_eq!(row(id).read(1.0).value, Some(row(id).fit.at(1.0)), "{id}: not refused at 1");
        }
    }

    /// A lattice at full density is the solid, so no modulus fit may read more than the solid
    /// there; the two-branch fits are built to reach exactly 1.
    #[test]
    fn no_modulus_row_reads_more_than_the_solid_at_full_density() {
        for r in ROWS.iter().filter(|r| r.class() == PropertyClass::Modulus) {
            assert!(r.fit.at(1.0) <= 1.0 + 1e-9, "{}: {}", r.id, r.fit.at(1.0));
        }
    }

    /// The guard that catches a coefficient in absolute units, which no shipped row may be: the
    /// excluded ones run 4.5 to 1363 MPa, and Libonati's 1.5–2.3 at n ≈ 1. **Not** a bound at
    /// full density -- eight strength fits pass the solid there, and are refused there instead
    /// (`a_strength_fit_above_the_solid_at_full_density_is_refused_there`).
    #[test]
    fn no_coefficient_is_in_absolute_units() {
        for r in ROWS {
            let c = match r.fit {
                Fit::Power { c, .. } | Fit::PowerOffset { c, .. } => c,
                Fit::TwoBranch { c2, c3, .. } => c2 + c3,
            };
            assert!(c <= 3.0, "{}: C {c}", r.id);
        }
    }

    /// Eight strength fits read more than the solid itself at full density -- Sun 2022's gyroid
    /// sheet 2.07 times it -- and the readout never shows one there: each is refused, as above the
    /// solid's share, at ρ = 1.
    #[test]
    fn a_strength_fit_above_the_solid_at_full_density_is_refused_there() {
        let mut above: Vec<&str> = ROWS.iter().filter(|r| r.fit.at(1.0) > 1.0).map(|r| r.id).collect();
        above.sort_unstable();
        assert_eq!(
            above,
            [
                "alabort19-d-solid-strength",
                "alabort19-neovius-solid-strength",
                "sun22-d-sheet-strength",
                "sun22-g-sheet-strength",
                "wang22-cubic-strut-strength",
                "wang22-octet-strut-strength",
                "yan15-d-solid-strength",
                "yan15-g-solid-strength",
            ]
        );
        for id in above {
            assert_eq!(row(id).class(), PropertyClass::Strength, "{id}");
            assert_eq!(row(id).read(1.0).standing, Standing::Refused(Refusal::AboveSolidShare), "{id}");
        }
    }

    /// **A cube axis is said only where the source's own model fixes it**: Cellucci 2016's beam
    /// cell and Lu 2020's homogenisation, each loading a strain along one cell axis. Every other
    /// row states no direction, a build direction, the honeycomb's plane, or -- Deshpande's
    /// octet -- the body diagonal. A new row that claims [100] fails here until it is read.
    #[test]
    fn the_rows_that_state_a_cube_axis_are_those_whose_model_fixes_it() {
        let mut axis: Vec<&str> =
            ROWS.iter().filter(|r| r.direction == LoadDirection::Axis100).map(|r| r.id).collect();
        axis.sort_unstable();
        assert_eq!(
            axis,
            [
                "cellucci16-octet-strut-e100",
                "lu20-cubic-strut-e",
                "lu20-d-solid-e",
                "lu20-fks-solid-e",
                "lu20-g-solid-e",
                "lu20-p-solid-e",
            ]
        );
        for id in axis {
            let r = row(id);
            assert!(matches!(r.data, DataKind::Fe) && r.solid_basis == SolidBasis::ModelInput, "{id}");
        }
    }

    /// Every research row is shipped or excluded, once, with a reason. (The research ledger it
    /// was once checked against is not published, so only the code's side of the count remains.)
    #[test]
    fn every_ledger_row_is_shipped_or_excluded_with_a_reason() {
        let mut numbers: Vec<u16> = ROWS
            .iter()
            .map(|r| r.ledger_row)
            .chain(EXCLUDED.iter().map(|r| r.ledger_row))
            .collect();
        numbers.sort_unstable();
        assert_eq!(numbers, (0..=74).collect::<Vec<u16>>(), "0 to 74, each once");
        assert_eq!(EXCLUDED.len(), 27);
        for e in EXCLUDED {
            assert!(!e.reason.trim().is_empty() && !e.source.trim().is_empty(), "row {}", e.ledger_row);
        }
    }

    /// The defaults of design §3.3, by the stated rank.
    #[test]
    fn exactly_one_default_per_key_and_it_is_the_best_ranked() {
        use Banding::{Sheet, Solid, Strut};
        let golden: [(CellKind, Banding, PropertyClass, &str); 23] = [
            (SCHWARZ_P, Sheet, E, "nx23-p-sheet-e"),
            (GYROID, Sheet, E, "nx23-g-sheet-e"),
            (SCHOEN_IWP, Sheet, E, "nx23-iwp-sheet-e"),
            (GYROID, Sheet, S, "naghavi22-g-sheet-yield"),
            (SCHWARZ_D, Sheet, E, "naghavi22-d-sheet-e"),
            (SCHWARZ_D, Sheet, S, "naghavi22-d-sheet-yield"),
            (GYROID, Solid, E, "lu20-g-solid-e"),
            (GYROID, Solid, S, "li19-g-solid-plateau"),
            (SCHWARZ_D, Solid, E, "lu20-d-solid-e"),
            (SCHWARZ_P, Solid, E, "lu20-p-solid-e"),
            (FISCHER_KOCH_S, Solid, E, "lu20-fks-solid-e"),
            (SCHWARZ_D, Solid, S, "alabort19-d-solid-strength"),
            (NEOVIUS, Solid, E, "alabort19-neovius-solid-e"),
            (NEOVIUS, Solid, S, "alabort19-neovius-solid-strength"),
            (CUBIC, Strut, E, "lu20-cubic-strut-e"),
            (CUBIC, Strut, S, "wang22-cubic-strut-strength"),
            (BCC, Strut, E, "maskery17-bcc-strut-e"),
            (BCC, Strut, S, "maskery17-bcc-strut-plateau"),
            (OCTET, Strut, E, "cellucci16-octet-strut-e100"),
            (OCTET, Strut, S, "bhate19-octet-strut-strength"),
            (DIAMOND, Strut, E, "ahmadi15-diamond-strut-e"),
            (HONEYCOMB, Sheet, E, "beigrezaee26-honeycomb-sheet-e"),
            (SCHWARZ_P, Sheet, S, "sun22-p-sheet-strength"),
        ];
        // Every key a row has is in the golden list, so a new row's key cannot slip in without
        // its default being decided here.
        for r in ROWS {
            assert!(
                golden.iter().any(|(c, p, k, _)| same_cell(*c, r.cell) && *p == r.phase && *k == r.class()),
                "{}'s key has no golden default",
                r.id
            );
        }
        for (cell, phase, class, id) in golden {
            let chosen = default_row(cell, phase, class).expect("a key with rows has a default");
            assert_eq!(chosen.id, id, "{cell:?} {phase:?} {class:?}");
            for other in rows_for(cell, phase, class) {
                assert!(chosen.rank() <= other.rank(), "{} outranks the default {id}", other.id);
            }
        }
    }

    /// The order the readout's hover lists and falls through in is the rank's, starting at the
    /// default, and leaves no row of the key out.
    #[test]
    fn the_ranked_rows_begin_with_the_default_and_hold_every_row_of_the_key() {
        use Banding::{Sheet, Solid, Strut};
        let mut keys_seen = 0;
        for cell in [GYROID, SCHWARZ_P, SCHWARZ_D, SCHOEN_IWP, NEOVIUS, FISCHER_KOCH_S, CUBIC, BCC, FCC, OCTET, DIAMOND, HONEYCOMB, VORONOI_BEAM] {
            for phase in [Sheet, Solid, Strut] {
                for class in [E, S] {
                    let ranked = ranked_rows(cell, phase, class);
                    assert_eq!(ranked.len(), rows_for(cell, phase, class).count(), "{cell:?} {phase:?} {class:?}");
                    if ranked.is_empty() {
                        assert!(default_row(cell, phase, class).is_none());
                        continue;
                    }
                    keys_seen += 1;
                    assert_eq!(ranked[0].id, default_row(cell, phase, class).unwrap().id);
                    for pair in ranked.windows(2) {
                        assert!(pair[0].rank() <= pair[1].rank(), "{} before {}", pair[0].id, pair[1].id);
                    }
                }
            }
        }
        assert_eq!(keys_seen, 23, "the golden list has 23 keys");
    }

    /// **Never a neighbour's row.** Where nothing was found, nothing is said but that.
    #[test]
    fn a_cell_with_no_row_answers_none_rather_than_a_neighbour_s_row() {
        let none = [
            (NEOVIUS, Banding::Sheet, E),
            (FISCHER_KOCH_S, Banding::Sheet, E),
            (SCHOEN_IWP, Banding::Solid, E),
            (FCC, Banding::Strut, E),
            (FCC, Banding::Strut, S),
            (SCHWARZ_P, Banding::Solid, S),
            (DIAMOND, Banding::Strut, S),
            (HONEYCOMB, Banding::Sheet, S),
        ];
        for (cell, phase, class) in none {
            assert!(default_row(cell, phase, class).is_none(), "{cell:?} {phase:?} {class:?}");
            assert_eq!(rows_for(cell, phase, class).count(), 0);
        }
        for kind in StochasticKind::ALL {
            for (jitter, seed) in [(0.0, 0), (0.5, 7), (1.0, u32::MAX)] {
                let cell = CellKind::Stochastic { kind, jitter, seed };
                for phase in [Banding::Sheet, Banding::Solid, Banding::Strut] {
                    for class in [E, S] {
                        assert!(default_row(cell, phase, class).is_none(), "{cell:?} {phase:?}");
                    }
                }
            }
        }
        // And the one found and not used is named, at any jitter.
        let voronoi = CellKind::Stochastic { kind: StochasticKind::VoronoiBeam, jitter: 0.3, seed: 9 };
        assert_eq!(excluded_for(voronoi, Banding::Strut, E).map(|e| e.ledger_row).collect::<Vec<_>>(), [46]);
        assert_eq!(excluded_for(FCC, Banding::Strut, E).map(|e| e.ledger_row).collect::<Vec<_>>(), [39]);
        // Sokollu's Schwarz P has no known phase, so it is named for both.
        assert!(excluded_for(SCHWARZ_P, Banding::Solid, E).any(|e| e.ledger_row == 57));
        assert!(excluded_for(SCHWARZ_P, Banding::Sheet, E).any(|e| e.ledger_row == 57));
    }

    #[test]
    fn a_one_density_row_says_so_away_from_its_density() {
        let r = row("li19-g-sheet-e");
        assert_eq!(r.read(0.30).standing, Standing::Inside);
        assert_eq!(r.read(0.304).standing, Standing::Inside);
        assert_eq!(r.read(0.40).standing, Standing::OnePoint { at: 0.30 });
    }

    #[test]
    fn a_lowest_density_not_stated_is_said_below_where_the_data_surely_reach() {
        let r = row("nx23-g-sheet-e");
        assert_eq!(r.read(0.10).standing, Standing::LowestNotStated { surely_from: 0.15 });
        assert_eq!(r.read(0.30).standing, Standing::Inside);
        assert_eq!(
            r.read(0.90).standing,
            Standing::Outside { lo: Lower::NotStated { surely_from: 0.15 }, hi: 0.80 }
        );
        assert_eq!(
            row("naghavi22-g-sheet-e").read(0.20).standing,
            Standing::Outside { lo: Lower::At(0.28), hi: 0.46 }
        );
        assert_eq!(row("wang22-octet-strut-e").read(0.20).standing, Standing::RangeNotStated);
    }

    #[test]
    fn the_octet_reads_along_both_cubic_axes() {
        let default = default_row(OCTET, Banding::Strut, E).expect("the octet has a modulus");
        assert_eq!(default.direction, LoadDirection::Axis100);
        assert_eq!(default.fit, Fit::Power { c: 1.0 / 9.0, n: 1.0 });
        let other = other_axis_row(default).expect("and a [111] partner");
        assert_eq!(other.direction, LoadDirection::Diagonal111);
        assert_eq!(other.fit, Fit::Power { c: 0.2, n: 1.0 });
        assert_eq!(other_axis_row(other).map(|r| r.id), Some(default.id));
        // No other cell has a second axis.
        for r in ROWS.iter().filter(|r| !same_cell(r.cell, OCTET)) {
            assert!(other_axis_row(r).is_none(), "{}", r.id);
        }
    }

    /// **The two octet rows are one stiffness tensor**, which is what makes quoting them side by
    /// side honest. From a pin-jointed octet -- bars along the six <110> directions, an equal
    /// share of the material each -- the stiffness is `C_ijkl = ρE_s/6 · Σ n_i n_j n_k n_l`.
    /// Inverted and read along [100] and [111], it gives exactly the two rows' coefficients.
    #[test]
    fn the_octet_s_two_axes_are_one_stiffness_tensor() {
        let h = 0.5f64.sqrt();
        let bars = [[h, h, 0.0], [h, -h, 0.0], [h, 0.0, h], [h, 0.0, -h], [0.0, h, h], [0.0, h, -h]];
        // Voigt pairs, with engineering shear: (11, 22, 33, 23, 13, 12).
        let pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
        let mut c = nalgebra::Matrix6::<f64>::zeros();
        for (a, (i, j)) in pairs.iter().enumerate() {
            for (b, (k, l)) in pairs.iter().enumerate() {
                c[(a, b)] = bars.iter().map(|n| n[*i] * n[*j] * n[*k] * n[*l]).sum::<f64>() / 6.0;
            }
        }
        assert!((c[(0, 0)] - 1.0 / 6.0).abs() < 1e-15 && (c[(0, 1)] - 1.0 / 12.0).abs() < 1e-15);
        assert!((c[(3, 3)] - 1.0 / 12.0).abs() < 1e-15, "C44 = C12 for the bar model");
        let s = c.try_inverse().expect("a pin-jointed octet is rigid");
        let modulus_along = |d: [f64; 3]| {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let d = [d[0] / n, d[1] / n, d[2] / n];
            let sigma = nalgebra::Vector6::new(
                d[0] * d[0],
                d[1] * d[1],
                d[2] * d[2],
                d[1] * d[2],
                d[0] * d[2],
                d[0] * d[1],
            );
            1.0 / (sigma.transpose() * s * sigma)[(0, 0)]
        };
        let along_100 = modulus_along([1.0, 0.0, 0.0]);
        let along_111 = modulus_along([1.0, 1.0, 1.0]);
        assert!((1.0 / along_100 - 9.0).abs() < 1e-9, "1/E[100] = {}", 1.0 / along_100);
        assert!((1.0 / along_111 - 5.0).abs() < 1e-9, "1/E[111] = {}", 1.0 / along_111);
        assert!((row("cellucci16-octet-strut-e100").fit.at(1.0) - along_100).abs() < 1e-12);
        assert!((row("deshpande01-octet-strut-e111").fit.at(1.0) - along_111).abs() < 1e-12);
    }
