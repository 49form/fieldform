// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The readout's tests: what each lattice reads as, what it refuses, and what a target asks for.

    use super::*;
    use fieldform_field::ROWS;
    use fieldform_scene::{Attractor, BooleanOp, Frame, NodeMeta, ParamId, Place, RampShape};

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn lattice(cell: LatticeCell, map: CellMap, phase: LatticePhase) -> Node {
        Node::Lattice { frame: Frame::default(), cell, map, phase }
    }

    fn cubes(cell_mm: f64) -> CellMap {
        CellMap::Rectangular { cell: lit(cell_mm) }
    }

    fn sheet(t: f64) -> LatticePhase {
        LatticePhase::sheet(lit(t))
    }

    fn solid(d: f64) -> LatticePhase {
        LatticePhase::Solid { density: lit(d) }
    }

    fn strut(r: f64) -> LatticePhase {
        LatticePhase::Strut { radius: lit(r) }
    }

    /// A document and a graph holding `node`, and its id.
    fn one(node: Node) -> (Document, Graph, NodeId) {
        let mut graph = Graph::empty();
        let id = graph.add(node);
        (Document::new(), graph, id)
    }

    fn readout(node: Node) -> Vec<ReadoutLine> {
        let (doc, graph, id) = one(node);
        lattice_readout(&doc, &graph, id, None)
    }

    fn labelled<'a>(lines: &'a [ReadoutLine], label: &str) -> Option<&'a ReadoutLine> {
        lines.iter().find(|l| l.label == label)
    }

    /// Everything a line prints, which is what a person reads off it.
    fn printed(line: &ReadoutLine) -> String {
        console_lines(std::slice::from_ref(line)).join("\n")
    }

    fn row(id: &str) -> &'static PropertyRow {
        ROWS.iter().find(|r| r.id == id).unwrap_or_else(|| panic!("no row {id}"))
    }

    /// Every cell the panel offers that has a density to read, in each phase it takes, at a
    /// number every one of them builds on a 10 mm cell: a sheet under the thinnest wall limit
    /// (Neovius, 6%), a solid above the highest floor (Neovius, 0.344 typed), a strut under the
    /// smallest merge radius (diamond, 0.217).
    fn every_cell() -> Vec<(LatticeCell, Node)> {
        let cells = [
            LatticeCell::Gyroid,
            LatticeCell::SchwarzP,
            LatticeCell::SchwarzD,
            LatticeCell::SchoenIwp,
            LatticeCell::Neovius,
            LatticeCell::FischerKochS,
            LatticeCell::Cubic,
            LatticeCell::Bcc,
            LatticeCell::Fcc,
            LatticeCell::Octet,
            LatticeCell::Diamond,
            LatticeCell::Honeycomb,
        ];
        let mut out = Vec::new();
        for cell in cells {
            for banding in CellKind::from(cell).supported_bandings() {
                let phase = match banding {
                    Banding::Sheet => sheet(0.5),
                    Banding::Solid => solid(0.45),
                    Banding::Strut => strut(0.5),
                };
                out.push((cell, lattice(cell, cubes(10.0), phase)));
            }
        }
        out
    }

    /// **Ruling 7, said where it is read.** A line says a cube axis or a body diagonal only where
    /// a row of its own cell, phase and class states that direction -- the octet's two, and Lu
    /// 2020's homogenised [100] -- and never for a row that states none.
    #[test]
    fn the_readout_never_calls_an_unstated_direction_a_cube_axis() {
        let mut octet_said_both = false;
        let mut lu_said_its_axis = false;
        for (cell, node) in every_cell() {
            let kind = CellKind::from(cell);
            let Node::Lattice { phase, .. } = &node else { unreachable!() };
            for line in readout(node.clone()) {
                let said = printed(&line);
                let class = match line.label {
                    "Stiffness:" => Some(PropertyClass::Modulus),
                    "Strength:" => Some(PropertyClass::Strength),
                    _ => None,
                };
                let states = |direction: LoadDirection| {
                    class.is_some_and(|class| {
                        ranked_rows(kind, phase.banding(), class).iter().any(|r| r.direction == direction)
                    })
                };
                for (claims, direction) in [
                    (["[100]", "cube axis"], LoadDirection::Axis100),
                    (["[111]", "body diagonal"], LoadDirection::Diagonal111),
                ] {
                    for claim in claims {
                        assert!(
                            !said.contains(claim) || states(direction),
                            "{cell:?} says {claim:?} with no row stating it: {said}"
                        );
                    }
                }
                if cell == LatticeCell::Octet {
                    octet_said_both |= said.contains("[100]") && said.contains("[111]");
                }
                if cell == LatticeCell::Gyroid && line.label == "Stiffness:" && said.contains("Lu 2020") {
                    lu_said_its_axis |= line.text.contains("along a cube axis [100] · Lu 2020");
                }
            }
        }
        assert!(octet_said_both, "the octet's two axes are both on its line");
        assert!(lu_said_its_axis, "Lu 2020's homogenised axis is on the gyroid solid's line");
    }

    /// A number never stands without its source, and every reading says it is an estimate: "≈"
    /// on the line, "estimate" closing the detail, and the caveat closing the hover. The
    /// honeycomb's one row is in-plane, which is said on the main line, not left to the detail
    /// (ruling 7): out of plane it is about sixteen times stiffer.
    #[test]
    fn every_readout_value_names_its_source_and_says_estimate() {
        let shorts: Vec<&str> = ROWS.iter().map(|r| r.source.short).collect();
        let mut read = 0;
        for (cell, node) in every_cell() {
            for line in readout(node) {
                if !matches!(line.label, "Stiffness:" | "Strength:") || !line.text.contains('%') {
                    continue;
                }
                read += 1;
                assert!(line.text.contains('≈'), "{cell:?}: a value without ≈ on {:?}", line.text);
                if cell == LatticeCell::Honeycomb && line.label == "Stiffness:" {
                    assert!(line.text.contains("% of the solid's modulus, in-plane · "), "{}", line.text);
                }
                assert!(
                    shorts.iter().any(|s| line.text.contains(s)),
                    "{cell:?}: no source named on {:?}",
                    line.text
                );
                let detail = line.detail.as_deref().unwrap_or_else(|| panic!("{cell:?}: no detail"));
                assert!(detail.ends_with("estimate"), "{cell:?}: {detail}");
                assert!(line.hover.ends_with(CAVEAT), "{cell:?}: the hover does not close with the caveat");
            }
        }
        assert!(read >= 20, "only {read} readings were made — the sweep is not reaching the table");
    }

    #[test]
    fn a_solid_lattice_reads_and_nothing_reads_on_a_curved_map() {
        let rectangular = readout(lattice(LatticeCell::Gyroid, cubes(10.0), solid(0.45)));
        assert!(
            labelled(&rectangular, "Stiffness:").is_some_and(|l| l.text.contains('%')),
            "{rectangular:#?}"
        );
        let curved_maps = [
            ("cylindrical", CellMap::Cylindrical { cell_radius: lit(10.0), cell_height: lit(10.0), arc_count: 24 }),
            ("spherical", CellMap::Spherical { cell_radius: lit(10.0), azimuth_count: 24, polar_count: 12 }),
            (
                "conformal",
                CellMap::Conformal {
                    surface: fieldform_scene::QuadMeshId::new(),
                    cells_per_face: 2,
                    depth: lit(10.0),
                    thickness: None,
                },
            ),
        ];
        for (name, map) in curved_maps {
            for (cell, phase) in [
                (LatticeCell::Gyroid, sheet(1.0)),
                (LatticeCell::Gyroid, solid(0.45)),
                (LatticeCell::Octet, strut(0.5)),
                (LatticeCell::Neovius, solid(0.30)),
            ] {
                let lines = readout(lattice(cell, map, phase));
                assert_eq!(lines.len(), 1, "{name} {cell:?}: {lines:#?}");
                let only = &lines[0];
                assert_eq!(only.label, "Properties:");
                assert!(only.text.starts_with("not estimated") && only.text.contains(name), "{}", only.text);
                assert!(!only.text.contains('%'));
                // The floor is the density's, so it is said on a curved map too (ruling 11).
                let separate = only.warn.iter().any(|w| w.contains("separate pieces"));
                assert_eq!(separate, cell == LatticeCell::Neovius, "{name} {cell:?}: {:?}", only.warn);
            }
        }
    }

    #[test]
    fn a_solid_that_is_separate_pieces_is_not_estimated() {
        for (cell, typed) in [(LatticeCell::Neovius, 0.30), (LatticeCell::SchwarzP, 0.05)] {
            let lines = readout(lattice(cell, cubes(10.0), solid(typed)));
            let properties = labelled(&lines, "Properties:").unwrap_or_else(|| panic!("{cell:?}: {lines:#?}"));
            assert!(properties.text.contains("separate pieces"), "{}", properties.text);
            assert!(labelled(&lines, "Stiffness:").is_none() && labelled(&lines, "Strength:").is_none(), "{lines:#?}");
            assert!(lines.iter().all(|l| !l.text.contains('%')), "{lines:#?}");
        }
    }

    /// Ruling 10: a typed 0.05 builds 0.031 in a Schwarz P solid, and that is what is said and
    /// what a fit is read at.
    #[test]
    fn a_solid_reads_the_density_it_builds() {
        let lines = readout(lattice(LatticeCell::SchwarzP, cubes(10.0), solid(0.05)));
        let built = labelled(&lines, "Density built:").expect("a gap over 1% is said");
        assert!(built.text.starts_with("0.031") && built.tone == Tone::Warn, "{}", built.text);
        assert!(labelled(&lines, "Properties:").unwrap().text.contains("at 0.031 a Schwarz P solid"));

        // A gyroid holds together at 0.05, so it is read -- at what it builds, 0.043.
        let lines = readout(lattice(LatticeCell::Gyroid, cubes(10.0), solid(0.05)));
        assert!(labelled(&lines, "Density built:").unwrap().text.starts_with("0.043"));
        let at_built = row("li19-g-solid-e").read(TpmsKind::Gyroid.solid_density_built(0.05)).value.unwrap();
        let at_typed = row("li19-g-solid-e").read(0.05).value.unwrap();
        // Lu 2020's offset fit is refused that near its zero, so Li 2019 is shown.
        let hover = &labelled(&lines, "Stiffness:").unwrap().hover;
        assert!(hover.contains(&format!("• Li 2019 (shown): ≈ {}%", pct(at_built))), "{hover}");
        assert_ne!(pct(at_built), pct(at_typed), "the test must tell the two apart");
        // Where the two agree, nothing is said.
        let lines = readout(lattice(LatticeCell::Gyroid, cubes(10.0), solid(0.40)));
        assert!(labelled(&lines, "Density built:").is_none());
    }

    #[test]
    fn a_reading_outside_the_source_s_range_says_so() {
        // Lu 2020 fitted the gyroid solid over 0.30-0.50.
        let lines = readout(lattice(LatticeCell::Gyroid, cubes(10.0), solid(0.60)));
        let stiffness = labelled(&lines, "Stiffness:").unwrap();
        assert!(stiffness.text.contains("Lu 2020"), "{}", stiffness.text);
        assert!(
            stiffness.warn.iter().any(|w| w == "outside the source's density range, 0.30–0.50"),
            "{:?}",
            stiffness.warn
        );
        // Inside Lu's range nothing is said of Lu's -- what is said is of the others, by name.
        let inside = readout(lattice(LatticeCell::Gyroid, cubes(10.0), solid(0.40)));
        let warn = &labelled(&inside, "Stiffness:").unwrap().warn;
        assert!(warn.iter().all(|w| w.starts_with("of the 3 other sources, ")), "{warn:?}");
    }

    /// **Every source a figure on the line rests on is held to ruling 8**, not only the one
    /// shown. A gyroid sheet with a 0.3 mm wall in 10 mm cells is ρ 0.093: the range it leads
    /// with once had Naghavi's fit, read at a third of its range, as its top, and said nothing.
    #[test]
    fn every_source_a_range_rests_on_says_where_it_stands() {
        let lines = readout(lattice(LatticeCell::Gyroid, cubes(10.0), sheet(0.3)));
        let rho = TpmsKind::Gyroid.sheet_density(0.03).unwrap();
        assert!((rho - 0.093).abs() < 0.001, "{rho}");
        let stiffness = labelled(&lines, "Stiffness:").unwrap();
        assert!(stiffness.text.contains("across 5 sources"), "{}", stiffness.text);
        let named = stiffness
            .warn
            .iter()
            .find(|w| w.starts_with("of the 5 sources in the range, "))
            .unwrap_or_else(|| panic!("no caution names the range's sources: {:?}", stiffness.warn));
        for said in [
            "read outside its own range: Naghavi 2022 (0.28–0.46)",
            "measured at one density: Li 2019 (0.30)",
            "no range stated: Polley 2022 via Distefano 2023, Sun 2022 via Distefano 2023",
        ] {
            assert!(named.contains(said), "{said:?} is not in {named:?}");
        }
        // The shown row's own standing is said once, on its own, not again in the list.
        assert!(!named.contains("Nguyen-Xuan"), "{named}");
        assert!(stiffness.warn.iter().any(|w| w.contains("below 0.15 this may be an extrapolation")));

        // The others line is held to it too.
        let strength = labelled(&lines, "Strength:").unwrap();
        assert!(strength.others[0].starts_with("3 other sources, strength defined differently"));
        let named = strength.warn.iter().find(|w| w.starts_with("of the 3 other sources, ")).unwrap();
        assert!(
            named.contains("Li 2019 (0.30)") && named.contains("no range stated: Polley 2022 via Distefano 2023, Sun 2022"),
            "{named}"
        );
        // And the console prints every one of them: nothing ruling 8 asks for is hover-only.
        let printed = console_lines(&lines).join("\n");
        assert!(printed.contains("caution: of the 5 sources in the range, read outside its own range: Naghavi 2022"));
    }

    #[test]
    fn a_range_not_stated_is_a_warning() {
        // A cubic strut's only strength row is quoted from a review that gives no range.
        let lines = readout(lattice(LatticeCell::Cubic, cubes(10.0), strut(1.0)));
        let strength = labelled(&lines, "Strength:").unwrap();
        assert!(strength.text.contains("Wang 2022"), "{}", strength.text);
        assert!(strength.warn.iter().any(|w| w.contains("range is not stated")), "{:?}", strength.warn);
    }

    /// Ruling 5: a strength of another definition is counted and ranged, not dropped.
    #[test]
    fn rows_defined_differently_are_counted_not_dropped() {
        let lines = readout(lattice(LatticeCell::Gyroid, cubes(10.0), solid(0.30)));
        let strength = labelled(&lines, "Strength:").unwrap();
        assert!(strength.text.contains("plateau stress") && strength.text.contains("Li 2019"), "{}", strength.text);
        let others = strength.others.first().expect("Yan and Ataee are counted");
        assert!(others.starts_with("2 other sources, strength defined differently: "), "{others}");
        let yan = row("yan15-g-solid-strength").read(0.300417).value.unwrap();
        let ataee = row("ataee18-g-solid-strength").read(0.300417).value.unwrap();
        assert!(others.contains(&format!("{}–{}%", pct(ataee), pct(yan))), "{others}");
    }

    #[test]
    fn a_cell_with_no_row_says_none_was_found() {
        let lines = readout(lattice(LatticeCell::Neovius, cubes(10.0), sheet(0.5)));
        let properties = labelled(&lines, "Properties:").expect("one line for both");
        assert_eq!(properties.text, "no published fit for a Neovius sheet");
        assert!(!properties.text.chars().any(|c| c.is_ascii_digit()));
        assert!(properties.hover.contains("research ledger"));
        assert!(labelled(&lines, "Stiffness:").is_none() && labelled(&lines, "Strength:").is_none());
        // One class without a row says so on its own line, and the other still reads.
        let lines = readout(lattice(LatticeCell::SchoenIwp, cubes(10.0), sheet(1.0)));
        assert!(labelled(&lines, "Stiffness:").unwrap().text.contains('%'));
        assert_eq!(labelled(&lines, "Strength:").unwrap().text, "no published fit for a Schoen IWP sheet");
        // A Voronoi cell has no density to read, so it never reaches the line above; the one row
        // found for it and not used is named all the same (ledger row 46, §6.3).
        let voronoi = LatticeCell::voronoi_beam(9);
        let lines = readout(lattice(voronoi, cubes(10.0), strut(0.5)));
        let properties = labelled(&lines, "Properties:").expect("one line");
        assert!(properties.text.starts_with("not estimated — a Voronoi cell's fill"), "{}", properties.text);
        assert!(properties.hover.contains("research ledger"), "{}", properties.hover);
        assert!(
            properties.hover.contains("One was found and not used: Roberts & Garboczi 2002"),
            "{}",
            properties.hover
        );
    }

    #[test]
    fn a_number_that_does_not_resolve_is_not_read_as_nought() {
        let missing = ParamRef::Named(ParamId::new());
        let lines = readout(lattice(LatticeCell::Gyroid, cubes(10.0), LatticePhase::sheet(missing)));
        assert_eq!(lines.len(), 1, "{lines:#?}");
        assert_eq!(lines[0].text, "not estimated — the wall does not resolve");
        let lines = readout(lattice(LatticeCell::Gyroid, CellMap::Rectangular { cell: missing }, sheet(1.0)));
        assert_eq!(lines[0].text, "not estimated — the cell size does not resolve");
        // An expression that fails is the same `None` (the project notes: a parameter's value is asked
        // of the document).
        let (mut doc, mut graph, _) = one(lattice(LatticeCell::Gyroid, cubes(10.0), sheet(1.0)));
        let zero = doc.add_parameter("zero", 0.0);
        let broken = doc.add_parameter("broken", 1.0);
        doc.set_parameter_expression(broken, "1 / zero").expect("the expression parses");
        let id = graph.add(lattice(LatticeCell::Gyroid, cubes(10.0), LatticePhase::sheet(ParamRef::Named(broken))));
        let _ = zero;
        let lines = lattice_readout(&doc, &graph, id, None);
        assert_eq!(lines[0].text, "not estimated — the wall does not resolve", "{lines:#?}");
    }

    /// **The readout's half of "what is estimated is what is built."** The kernel's
    /// `the_compile_builds_the_phase_phase_of_returns` holds the compile to `phase_of`; this
    /// holds the density the readout quotes to `uniform_density` of `phase_of`'s answer, for
    /// typed and named numbers alike -- so the number on the line is the one the field has.
    #[test]
    fn the_density_quoted_is_the_density_of_the_phase_the_compile_builds() {
        let mut doc = Document::new();
        let w = ParamRef::Named(doc.add_parameter("w", 0.8));
        let cases = [
            (LatticeCell::Gyroid, LatticePhase::sheet(w)),
            (LatticeCell::SchwarzD, LatticePhase::sheet(lit(1.3))),
            (LatticeCell::Octet, LatticePhase::Strut { radius: lit(0.6) }),
            (LatticeCell::Diamond, LatticePhase::Strut { radius: w }),
            (LatticeCell::Cubic, LatticePhase::Strut { radius: lit(1.5) }),
            (LatticeCell::Honeycomb, LatticePhase::sheet(lit(1.0))),
            (LatticeCell::Gyroid, LatticePhase::Solid { density: lit(0.05) }),
            (LatticeCell::Neovius, LatticePhase::Solid { density: lit(0.36) }),
        ];
        let resolve = |r: &ParamRef| r.resolve(&|id| doc.parameter_value(id));
        for (cell, phase) in cases {
            let mut graph = Graph::empty();
            let id = graph.add(lattice(cell, cubes(10.0), phase));
            let lines = lattice_readout(&doc, &graph, id, None);
            let built = phase_of(&phase, resolve, resolve).expect("every number resolves");
            let rho = uniform_density(CellKind::from(cell), built, 10.0).value().expect("it builds");
            let quoted = match phase {
                LatticePhase::Solid { .. } => {
                    let line = labelled(&lines, "Density built:").expect("both solids differ from typed");
                    (line.text.clone(), format!("{rho:.3}"))
                }
                _ => (labelled(&lines, "Density:").expect("a density line").text.clone(), fmt_rho(rho)),
            };
            assert!(quoted.0.starts_with(&format!("{} —", quoted.1)), "{cell:?}: {} against {}", quoted.0, quoted.1);
        }
    }

    /// A lattice whose phase number is driven by a ramp from `out_min` to `out_max`.
    fn driven(cell: LatticeCell, phase: fn(ParamRef) -> LatticePhase, out_min: f64, out_max: f64) -> (Graph, NodeId) {
        let mut graph = Graph::empty();
        let ramp = graph.add(Node::Ramp {
            input: fieldform_scene::Operand::empty(),
            in_min: lit(0.0),
            in_max: lit(50.0),
            out_min: lit(out_min),
            out_max: lit(out_max),
            shape: RampShape::Linear,
        });
        let id = graph.add(lattice(cell, cubes(10.0), phase(ParamRef::Field(ramp))));
        (graph, id)
    }

    /// A ramp's ends are read by name, in the order the user chose -- each end's own value
    /// under its own name -- and the middle of the ramp is read rather than averaged (§6.5).
    #[test]
    fn a_graded_lattice_reads_each_end_by_name_and_the_middle_is_not_the_mean() {
        // Thick to thin: Out-min is the 1.6 mm wall, Out-max the 0.4 mm one.
        let (graph, id) = driven(LatticeCell::Gyroid, LatticePhase::sheet, 1.6, 0.4);
        let lines = lattice_readout(&Document::new(), &graph, id, None);
        let stiffness = labelled(&lines, "Stiffness:").unwrap();
        let (at_min, at_max) = stiffness
            .text
            .split_once("; at the Out-max end, ")
            .unwrap_or_else(|| panic!("the Out-max end follows the Out-min one: {}", stiffness.text));
        assert!(at_min.starts_with("at the Out-min end, "), "{at_min}");
        let nx = row("nx23-g-sheet-e");
        let fill = |wall_mm: f64| TpmsKind::Gyroid.sheet_density(wall_mm / 10.0).unwrap();
        let reads = |wall_mm: f64| nx.read(fill(wall_mm)).value.unwrap();
        assert_ne!(pct(reads(1.6)), pct(reads(0.4)), "the test must tell the ends apart");
        assert!(at_min.contains(&format!("Nguyen-Xuan 2023 ≈ {}%", pct(reads(1.6)))), "{at_min}");
        assert!(at_max.contains(&format!("Nguyen-Xuan 2023 ≈ {}%", pct(reads(0.4)))), "{at_max}");
        // The middle of the ramp is a 1.0 mm wall, and it is read there, not averaged.
        let (middle, mean) = (reads(1.0), 0.5 * (reads(1.6) + reads(0.4)));
        assert_ne!(pct(middle), pct(mean), "the test must tell the middle from the mean");
        let note = stiffness.note.as_deref().expect("the middle is said");
        assert!(note.contains(&format!("≈ {}% by Nguyen-Xuan 2023", pct(middle))), "{note}");
        assert!(note.contains(&format!("not the {}% midway", pct(mean))), "{note}");
        let density = &labelled(&lines, "Density:").unwrap().text;
        assert!(
            density.starts_with(&format!("{} at the Out-min end, {} at the Out-max end", fmt_rho(fill(1.6)), fmt_rho(fill(0.4)))),
            "{density}"
        );
    }

    /// **A driven end is read exactly as a typed number is** (ruling 5 is unconditional): the
    /// range across sources, the octet's [111] partner, the fall-through past a refused default,
    /// the others counted, every caution -- each under its end's name. Held by reading a typed
    /// block at each end's number and finding its line inside the driven one.
    #[test]
    fn a_driven_end_reads_as_a_typed_number_would() {
        let cases: [(LatticeCell, fn(ParamRef) -> LatticePhase, f64, f64); 4] = [
            // Out-min builds 0.043, where Lu's fit is refused and the line falls through to Li.
            (LatticeCell::Gyroid, |p| LatticePhase::Solid { density: p }, 0.05, 0.40),
            (LatticeCell::Octet, |p| LatticePhase::Strut { radius: p }, 0.3, 0.8),
            (LatticeCell::Gyroid, LatticePhase::sheet, 1.6, 0.4),
            (LatticeCell::Honeycomb, LatticePhase::sheet, 0.5, 2.0),
        ];
        for (cell, phase, out_min, out_max) in cases {
            let (graph, id) = driven(cell, phase, out_min, out_max);
            let lines = lattice_readout(&Document::new(), &graph, id, None);
            for label in ["Stiffness:", "Strength:"] {
                let line = labelled(&lines, label).unwrap();
                if line.text.starts_with("no published fit") {
                    continue;
                }
                let typed_at = |number: f64| {
                    let lines = readout(lattice(cell, cubes(10.0), phase(lit(number))));
                    labelled(&lines, label).unwrap().clone()
                };
                let (min, max) = (typed_at(out_min), typed_at(out_max));
                assert_eq!(
                    line.text,
                    format!("at the Out-min end, {}; at the Out-max end, {}", min.text, max.text),
                    "{cell:?} {label}"
                );
                let expected_warn: Vec<String> = min
                    .warn
                    .iter()
                    .map(|w| format!("at the Out-min end: {w}"))
                    .chain(max.warn.iter().map(|w| format!("at the Out-max end: {w}")))
                    .collect();
                assert_eq!(line.warn, expected_warn, "{cell:?} {label}");
                let expected_others: Vec<String> = min
                    .others
                    .iter()
                    .map(|o| format!("at the Out-min end: {o}"))
                    .chain(max.others.iter().map(|o| format!("at the Out-max end: {o}")))
                    .collect();
                assert_eq!(line.others, expected_others, "{cell:?} {label}");
            }
        }
        // What the finding named, said outright: the octet keeps both axes at both ends, and a
        // driven gyroid solid falls through where its default's fit gives nothing.
        let (graph, id) = driven(LatticeCell::Octet, |p| LatticePhase::Strut { radius: p }, 0.3, 0.8);
        let octet = lattice_readout(&Document::new(), &graph, id, None);
        let text = &labelled(&octet, "Stiffness:").unwrap().text;
        for end in text.split("; at the Out-max end, ") {
            assert!(end.contains("[100]") && end.contains("[111] (Deshpande 2001 via Cellucci 2016"), "{end}");
        }
        let (graph, id) = driven(LatticeCell::Gyroid, |p| LatticePhase::Solid { density: p }, 0.05, 0.40);
        let solid = lattice_readout(&Document::new(), &graph, id, None);
        let stiffness = labelled(&solid, "Stiffness:").unwrap();
        assert!(stiffness.text.contains("Li 2019 ≈ ") && stiffness.text.contains("(Lu 2020's fit, the default here"));
        assert!(stiffness.others.iter().any(|o| o.starts_with("at the Out-max end: 3 other sources, direction not stated")));
    }

    /// The honeycomb's hover says what its one in-plane row leaves out at every end it is read
    /// at, not only at a typed number.
    #[test]
    fn a_driven_honeycomb_says_out_of_plane_at_each_end() {
        let (graph, id) = driven(LatticeCell::Honeycomb, LatticePhase::sheet, 0.5, 2.0);
        let lines = lattice_readout(&Document::new(), &graph, id, None);
        let hover = &labelled(&lines, "Stiffness:").unwrap().hover;
        for end in ["At the Out-min end, out of plane", "At the Out-max end, out of plane"] {
            assert!(hover.contains(end), "{end:?} not in {hover}");
        }
        assert!(hover.contains("At the Out-max end, read at the thin-wall density"), "{hover}");
        let typed = readout(lattice(LatticeCell::Honeycomb, cubes(10.0), sheet(1.0)));
        assert!(labelled(&typed, "Stiffness:").unwrap().hover.contains("\nOut of plane a honeycomb"));
    }

    /// The modulus the default source reads for a lattice as it stands: what the readout reads.
    fn default_modulus(doc: &Document, graph: &Graph, id: NodeId) -> f64 {
        let Some(Node::Lattice { cell, map: CellMap::Rectangular { cell: size }, phase, .. }) = graph.node(id) else {
            panic!("a rectangular lattice")
        };
        let kind = CellKind::from(*cell);
        let numbers = numbers_at_ends(doc, graph, phase).unwrap();
        let states = states_at(doc, phase, kind, resolve(doc, size).unwrap(), &numbers).unwrap();
        let rho = states[0].density.value().expect("the target builds");
        ranked_rows(kind, phase.banding(), PropertyClass::Modulus)[0].read(rho).value.unwrap()
    }

    /// What Set does with what it is given, as the panel does it: a phase into the lattice, a
    /// ramp in place of the ramp.
    fn apply(graph: &mut Graph, lattice_id: NodeId, write: SetWrite) {
        match write {
            SetWrite::Phase(next) => match graph.node_mut(lattice_id) {
                Some(Node::Lattice { phase, .. }) => *phase = next,
                _ => panic!("a lattice"),
            },
            SetWrite::Ramp(ramp, node) => *graph.node_mut(ramp).unwrap() = node,
        }
    }

    /// A target lands on itself, and **every number offered says where its source stands**:
    /// a caution exactly where the density it needs is past or without the source's range, on
    /// the console's line and the panel's hover alike -- the console never writes, so a caution
    /// that waited for the readout after Set would never reach it.
    #[test]
    fn a_target_lands_on_itself_through_the_density_table() {
        let cases = [
            lattice(LatticeCell::Gyroid, cubes(10.0), sheet(1.0)),
            lattice(LatticeCell::Octet, cubes(10.0), strut(0.5)),
            lattice(LatticeCell::Honeycomb, cubes(10.0), sheet(1.0)),
            lattice(LatticeCell::Gyroid, cubes(10.0), solid(0.4)),
        ];
        let mut cautioned = 0;
        for node in cases {
            for target in [0.02, 0.05, 0.10, 0.20] {
                let (doc, mut graph, id) = one(node.clone());
                let answer = number_for_target(&doc, &graph, id, None, End::Typed, target).unwrap();
                let Ok(write) = answer.write.clone() else { continue };
                assert!(matches!(write, SetWrite::Phase(_)), "a typed number writes the phase");
                apply(&mut graph, id, write);
                let read = default_modulus(&doc, &graph, id);
                assert!(
                    ((read - target) / target).abs() < 0.005,
                    "{node:?} at {target}: wrote {:?}, which reads {read}",
                    answer.default.number
                );
                for offer in std::iter::once(&answer.default).chain(&answer.others) {
                    let Ok(_) = offer.number else { continue };
                    let rho = offer.row.density_for(target).unwrap();
                    let past = !matches!(offer.row.read(rho).standing, Standing::Inside);
                    assert_eq!(offer.caution.is_some(), past, "{} at {target}: {:?}", offer.row.id, offer.caution);
                    if let Some(caution) = &offer.caution {
                        assert!(caution.starts_with(&format!("needs a density of {}, ", fmt_rho(rho))), "{caution}");
                        assert!(offers_hover(&answer).contains(caution.as_str()));
                        assert!(target_lines(&answer, target).join("\n").contains(caution.as_str()));
                    }
                }
                cautioned += usize::from(answer.default.caution.is_some());
            }
        }
        assert!(cautioned >= 4, "only {cautioned} default numbers were cautioned — the sweep misses the case");
        // The reviewer's octet, named: 5% by Cellucci needs a density of 0.45 against its 0.001–0.10.
        let (doc, graph, id) = one(lattice(LatticeCell::Octet, cubes(10.0), strut(0.5)));
        let answer = number_for_target(&doc, &graph, id, None, End::Typed, 0.05).unwrap();
        assert_eq!(answer.default.caution.as_deref(), Some("needs a density of 0.45, outside its range, 0.001–0.10"));
        let printed = target_lines(&answer, 0.05);
        assert_eq!(printed[1], "      caution: Cellucci 2016 needs a density of 0.45, outside its range, 0.001–0.10");
    }

    #[test]
    fn a_target_never_offers_a_number_the_geometry_cannot_build() {
        // Sun 2022 needs a gyroid sheet of density 0.85 for 10%, and its walls meet at 0.80.
        let (doc, graph, id) = one(lattice(LatticeCell::Gyroid, cubes(10.0), sheet(1.0)));
        let answer = number_for_target(&doc, &graph, id, None, End::Typed, 0.10).unwrap();
        let sun = answer.others.iter().find(|o| o.row.id == "sun22-g-sheet-e").expect("Sun is offered");
        let why = sun.number.as_ref().expect_err("not buildable");
        assert!(why.contains("0.85") && why.contains("walls meet at 0.80"), "{why}");
        assert!(offers_hover(&answer).contains("walls meet"), "the hover says it too");
        // A 1% Neovius solid is lighter than one that holds together.
        let (doc, graph, id) = one(lattice(LatticeCell::Neovius, cubes(10.0), solid(0.4)));
        let answer = number_for_target(&doc, &graph, id, None, End::Typed, 0.01).unwrap();
        let why = answer.default.number.as_ref().expect_err("below the floor");
        assert!(why.contains("separate pieces"), "{why}");
        assert!(answer.write.is_err(), "nothing to Set");
        // And every number that is offered builds, across targets and cells.
        for node in [
            lattice(LatticeCell::Gyroid, cubes(10.0), sheet(1.0)),
            lattice(LatticeCell::Neovius, cubes(10.0), solid(0.4)),
            lattice(LatticeCell::SchwarzP, cubes(10.0), solid(0.4)),
            lattice(LatticeCell::Octet, cubes(10.0), strut(0.5)),
            lattice(LatticeCell::Cubic, cubes(10.0), strut(0.5)),
            lattice(LatticeCell::Honeycomb, cubes(10.0), sheet(1.0)),
        ] {
            let Node::Lattice { cell, phase, .. } = &node else { unreachable!() };
            let kind = CellKind::from(*cell);
            let (doc, graph, id) = one(node.clone());
            for percent in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0] {
                let answer = number_for_target(&doc, &graph, id, None, End::Typed, percent / 100.0).unwrap();
                for offer in std::iter::once(&answer.default).chain(&answer.others) {
                    let Ok(number) = offer.number else { continue };
                    let at = phase_of(phase, |_| Some(number), |r| resolve(&doc, r)).unwrap();
                    assert!(
                        uniform_density(kind, at, 10.0).value().is_some(),
                        "{} offers {number} for {percent}% of {cell:?}, which does not build",
                        offer.row.id
                    );
                }
            }
        }
    }

    #[test]
    fn setting_a_number_from_a_target_never_rewrites_a_named_parameter() {
        let mut doc = Document::new();
        let w = doc.add_parameter("w", 1.0);
        let mut graph = Graph::empty();
        let id = graph.add(lattice(LatticeCell::Gyroid, cubes(10.0), LatticePhase::sheet(ParamRef::Named(w))));
        let answer = number_for_target(&doc, &graph, id, None, End::Typed, 0.05).unwrap();
        assert!(answer.default.number.is_ok(), "the number is still offered");
        let refused = answer.write.expect_err("but not written");
        assert!(refused.contains("the parameter 'w'") && refused.contains("never rewrites"), "{refused}");
    }

    fn graded(graph: &mut Graph, out_min: f64, out_max: f64) -> (NodeId, NodeId) {
        let ramp = graph.add(Node::Ramp {
            input: fieldform_scene::Operand::empty(),
            in_min: lit(0.0),
            in_max: lit(50.0),
            out_min: lit(out_min),
            out_max: lit(out_max),
            shape: RampShape::Linear,
        });
        let id = graph.add(lattice(LatticeCell::Gyroid, cubes(10.0), LatticePhase::sheet(ParamRef::Field(ramp))));
        (ramp, id)
    }

    /// B3: a grade that runs thick to thin stays that way when one end is set.
    #[test]
    fn a_target_keeps_the_direction_of_the_grade() {
        let mut graph = Graph::empty();
        let (ramp, id) = graded(&mut graph, 2.0, 0.5);
        let doc = Document::new();
        assert!(number_for_target(&doc, &graph, id, None, End::Typed, 0.05).is_err(), "a ramp has no typed end");
        let answer = number_for_target(&doc, &graph, id, None, End::OutMax, 0.05).unwrap();
        let write = answer.write.expect("a ramp driving only this lattice is written");
        let SetWrite::Ramp(block, written) = write else { panic!("the ramp, not the lattice") };
        assert_eq!(block, ramp, "the ramp, not the lattice");
        let Node::Ramp { out_min, out_max, .. } = written else { panic!("a ramp") };
        assert_eq!(out_min, lit(2.0), "the other end is untouched");
        let ParamRef::Literal(high) = out_max else { panic!("a literal") };
        assert!(high < 2.0 && (high - answer.default.number.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn a_shared_ramp_s_ends_are_not_rewritten_by_a_target() {
        let mut graph = Graph::empty();
        let (ramp, id) = graded(&mut graph, 0.5, 2.0);
        graph.add(lattice(LatticeCell::SchwarzD, cubes(10.0), LatticePhase::sheet(ParamRef::Field(ramp))));
        let answer = number_for_target(&Document::new(), &graph, id, None, End::OutMin, 0.05).unwrap();
        let refused = answer.write.expect_err("the ramp is shared");
        assert!(refused.contains("drives something else"), "{refused}");
    }

    #[test]
    fn a_lattice_reshaped_downstream_says_its_readout_is_for_its_own_cells() {
        let changed = |lines: &[ReadoutLine]| lines.iter().find(|l| l.text.contains("downstream changes them")).cloned();
        let base = || lattice(LatticeCell::Gyroid, cubes(10.0), sheet(1.0));

        let (mut doc, mut graph, id) = one(base());
        let offset = graph.add(Node::Offset { input: id.into(), distance: lit(0.2) });
        doc.node_meta.insert(offset, NodeMeta { name: Some("grown".to_string()), ..NodeMeta::default() });
        let line = changed(&lattice_readout(&doc, &graph, id, None)).expect("an offset reshapes the cells");
        assert!(line.text.contains("'grown'"), "{}", line.text);

        let (doc, mut graph, id) = one(base());
        graph.add(Node::Transform {
            input: id.into(),
            translation: [lit(5.0), lit(0.0), lit(0.0)],
            rotation: [lit(0.0); 3],
            scale: lit(2.0),
            about: Place::origin(),
        });
        assert!(changed(&lattice_readout(&doc, &graph, id, None)).is_none(), "a placement keeps every ratio");

        let (doc, mut graph, id) = one(base());
        let ball = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(20.0) });
        graph.add(Node::Attract { input: ball.into(), toward: Attractor::Point(id), strength: lit(0.5), falloff: lit(10.0) });
        assert!(changed(&lattice_readout(&doc, &graph, id, None)).is_none(), "being aimed at changes nothing");

        let (doc, mut graph, id) = one(base());
        let ball = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(20.0) });
        graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![ball.into(), id.into()] });
        let lines = lattice_readout(&doc, &graph, id, None);
        assert!(changed(&lines).is_none(), "a boolean clips, it does not reshape");
        assert!(labelled(&lines, "Stiffness:").unwrap().hover.contains("joins"), "and the hover says joins are not described");
    }

    #[test]
    fn the_estimate_verb_finds_the_lattices_inside_the_selected_body() {
        let mut graph = Graph::empty();
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::origin(),
            size: [lit(40.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let infill = graph.add(lattice(LatticeCell::Gyroid, cubes(10.0), sheet(1.0)));
        let part = graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![block.into(), infill.into()] });
        assert_eq!(lattices_within(&graph, part), vec![infill]);
        let label = |id: NodeId| if id == block { "Box".to_string() } else { "Lattice".to_string() };
        let doc = Document::new();
        let report = estimate_report(&doc, &graph, part, &label, Some(0.05)).unwrap();
        assert_eq!(report[0], "Lattice — a gyroid sheet:");
        assert!(report.iter().any(|l| l.starts_with("  Stiffness: ")), "{report:#?}");
        assert!(report.iter().any(|l| l.starts_with("  Target 5% of the solid's modulus → wall ")), "{report:#?}");
        assert_eq!(estimate_report(&doc, &graph, block, &label, None), Err("'Box' contains no lattice".to_string()));
    }

    /// The panel's floor note and the kernel's separate-pieces reading are one decision, at
    /// every typed hundredth of every surface.
    #[test]
    fn the_floor_note_agrees_with_the_density_the_readout_refuses() {
        for kind in TpmsKind::ALL {
            for i in 1..=99 {
                let typed = i as f64 / 100.0;
                let refused = matches!(
                    uniform_density(CellKind::Tpms(kind), Phase::Solid { density: typed }, 10.0),
                    RelativeDensity::Disconnected { .. }
                );
                assert_eq!(solid_floor_note(kind, typed).is_some(), refused, "{kind:?} at {typed}");
            }
        }
        let note = solid_floor_note(TpmsKind::Neovius, 0.30).unwrap();
        assert_eq!(note, "below about 0.34 built (0.344 typed) a Neovius solid is separate pieces, not one lattice");
    }

    #[test]
    fn a_percentage_is_two_significant_figures() {
        assert_eq!(pct(0.07), "7.0");
        assert_eq!(pct(0.1449), "14");
        assert_eq!(pct(0.0036), "0.36");
        assert_eq!(pct(1.33), "130");
        assert_eq!(pct(0.0996), "10");
        assert_eq!(fmt_rho(0.001), "0.001");
        assert_eq!(fmt_rho(0.0569), "0.057");
        assert_eq!(fmt_rho(0.3), "0.30");
    }
