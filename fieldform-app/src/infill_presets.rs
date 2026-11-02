// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The lattices Modify > Infill offers, and the numbers a new one starts from.
//!
//! **UI defaulting, not geometry**, which is why this sits beside the command and not in the
//! scene. A preset is a cell and a proportion. Picking one writes values into the lattice block
//! and the file holds that block and nothing else, so retuning a number here changes no saved
//! part. The numbers are starting points rather than claims: the readout says what each one fills.

use fieldform_scene::{LatticeCell, LatticePhase, ParamRef};

/// A preset's starting number, as a proportion so that it looks right at any scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Start {
    /// A wall, as a fraction of the cell.
    Sheet(f64),
    /// A density, which is already a fraction.
    Solid(f64),
    /// A strut's radius, as a fraction of the cell.
    Strut(f64),
}

/// One row of the picker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    pub group: &'static str,
    pub label: &'static str,
    /// The cell's name in the file, which is what the console reads.
    pub name: &'static str,
    pub start: Start,
}

const fn row(group: &'static str, label: &'static str, name: &'static str, start: Start) -> Preset {
    Preset { group, label, name, start }
}

/// Every built-in cell, once. `every_cell_the_lattice_panel_offers_has_a_preset` holds it to that.
pub const PRESETS: &[Preset] = &[
    row("Surfaces", "Gyroid", "gyroid", Start::Sheet(0.10)),
    row("Surfaces", "Schwarz P", "schwarz_p", Start::Sheet(0.10)),
    row("Surfaces", "Schwarz D", "schwarz_d", Start::Sheet(0.08)),
    row("Surfaces", "Schoen I-WP", "schoen_iwp", Start::Solid(0.30)),
    row("Surfaces", "Neovius", "neovius", Start::Sheet(0.08)),
    row("Surfaces", "Fischer-Koch S", "fischer_koch_s", Start::Sheet(0.08)),
    row("Beams", "Cubic", "cubic", Start::Strut(0.08)),
    row("Beams", "BCC", "bcc", Start::Strut(0.07)),
    row("Beams", "FCC", "fcc", Start::Strut(0.06)),
    row("Beams", "Octet", "octet", Start::Strut(0.05)),
    row("Beams", "Diamond", "diamond", Start::Strut(0.07)),
    row("Prism", "Honeycomb", "honeycomb", Start::Sheet(0.08)),
    row("Foam", "Voronoi foam", "voronoi_foam", Start::Sheet(0.08)),
    row("Foam", "Voronoi beam", "voronoi_beam", Start::Strut(0.06)),
];

impl Preset {
    /// The row for a cell, whichever numbers a stochastic one carries.
    pub fn of(cell: LatticeCell) -> Option<&'static Preset> {
        PRESETS.iter().find(|p| p.name == cell.name())
    }

    /// The cell, with the jitter a new Voronoi lattice starts from.
    pub fn cell(&self) -> LatticeCell {
        LatticeCell::from_name(self.name, ParamRef::Literal(0.75), 0)
            .expect("every preset names a cell")
    }

    /// The phase at a cell of `cell_mm`, as values. Never stored as a preset's name.
    pub fn phase(&self, cell_mm: f64) -> LatticePhase {
        match self.start {
            Start::Sheet(share) => {
                LatticePhase::sheet(ParamRef::Literal((share * cell_mm).max(0.001)))
            }
            Start::Solid(density) => LatticePhase::Solid { density: ParamRef::Literal(density) },
            Start::Strut(share) => {
                LatticePhase::Strut { radius: ParamRef::Literal((share * cell_mm).max(0.001)) }
            }
        }
    }
}

/// `v` snapped to the 1-2-5 series, by ratio rather than by difference, since that is what the
/// series is spaced by.
pub fn nice(v: f64) -> f64 {
    if !(v.is_finite() && v > 0.0) {
        return 1.0;
    }
    let decade = 10f64.powf(v.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|step| step * decade)
        .min_by(|a, b| (a / v).ln().abs().total_cmp(&(b / v).ln().abs()))
        .unwrap_or(v)
}

/// What a click with nothing typed starts from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FirstClick {
    pub cell_mm: f64,
    pub wall_mm: f64,
}

/// A cell and a wall for a body whose smallest side is `side`.
///
/// **The cell comes first, read off the whole side, and the wall from the cell.** Each was
/// specified from the other, which has no starting point. With a wall typed, that wall is taken
/// off both sides first, since what is being divided up is the inside. A wall that fills the body
/// falls back to the whole side: the block still gets made at the numbers asked for, and says so.
pub fn first_click(side: f64, typed_wall: Option<f64>) -> FirstClick {
    let cell_of = |across: f64| nice(across / 5.0);
    match typed_wall {
        None => {
            let cell_mm = cell_of(side);
            FirstClick { cell_mm, wall_mm: nice(cell_mm / 5.0).max(0.5) }
        }
        Some(wall_mm) => {
            let inner = side - 2.0 * wall_mm;
            FirstClick { cell_mm: cell_of(if inner > 0.0 { inner } else { side }), wall_mm }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_field::{GraphField, ScalarField};
    use fieldform_scene::{CellMap, Document, Frame, Graph, Node};
    use nalgebra::Point3;

    #[test]
    fn a_sixty_millimetre_cube_starts_where_the_benchmark_does() {
        assert_eq!(first_click(60.0, None), FirstClick { cell_mm: 10.0, wall_mm: 2.0 });
        assert_eq!(first_click(60.0, Some(2.0)), FirstClick { cell_mm: 10.0, wall_mm: 2.0 });
    }

    #[test]
    fn nice_lands_on_the_series_at_every_scale() {
        for (v, snapped) in [(11.2, 10.0), (1.2, 1.0), (3.0, 2.0), (4.0, 5.0), (0.3, 0.2), (70.0, 50.0), (80.0, 100.0)] {
            assert!((nice(v) - snapped).abs() < 1e-9, "nice({v}) was {}", nice(v));
        }
    }

    #[test]
    fn a_small_body_keeps_a_wall_a_printer_can_make() {
        assert_eq!(first_click(6.0, None).wall_mm, 0.5);
    }

    #[test]
    fn a_wall_that_fills_the_body_still_leaves_a_cell_size() {
        let click = first_click(10.0, Some(8.0));
        assert_eq!(click.wall_mm, 8.0, "the number asked for");
        assert!(click.cell_mm > 0.0);
    }

    #[test]
    fn every_cell_the_lattice_panel_offers_has_a_preset() {
        for cell in crate::LATTICE_CELLS {
            assert!(Preset::of(cell).is_some(), "{} has no preset", cell.name());
        }
        assert_eq!(PRESETS.len(), crate::LATTICE_CELLS.len(), "and one each");
    }

    #[test]
    fn every_preset_offers_a_phase_its_cell_has() {
        for preset in PRESETS {
            let offered = fieldform_field::CellKind::from(preset.cell()).supported_bandings();
            assert!(offered.contains(&preset.phase(10.0).banding()), "{}", preset.label);
        }
    }

    /// **T6.** A proportion is what lets one number hold at every size, so that is what is walked:
    /// each preset compiles, and fills neither nothing nor everything, from 2 to 100 mm.
    #[test]
    fn every_preset_is_a_lattice_at_every_cell_size() {
        for preset in PRESETS {
            for cell_mm in [2.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
                let mut graph = Graph::empty();
                graph.add(Node::Lattice {
                    frame: Frame::default(),
                    cell: preset.cell(),
                    map: CellMap::Rectangular { cell: ParamRef::Literal(cell_mm) },
                    phase: preset.phase(cell_mm),
                });
                let doc = Document::new();
                let field = GraphField::compile_graph(&doc, &graph, cell_mm / 200.0)
                    .unwrap_or_else(|e| panic!("{} at {cell_mm}: {e}", preset.label));
                // Twenty-seven cells on a grid no cell's period divides evenly.
                let (mut solid, mut all) = (0_u32, 0_u32);
                let across = 40;
                for i in 0..across {
                    for j in 0..across {
                        for k in 0..across {
                            let at = |n: i32| (f64::from(n) + 0.37) / f64::from(across) * 3.0 * cell_mm;
                            all += 1;
                            solid += u32::from(field.distance(Point3::new(at(i), at(j), at(k))) < 0.0);
                        }
                    }
                }
                let filled = f64::from(solid) / f64::from(all);
                assert!(
                    (0.01..0.8).contains(&filled),
                    "{} at {cell_mm} mm fills {filled:.3}",
                    preset.label,
                );
            }
        }
    }
}
