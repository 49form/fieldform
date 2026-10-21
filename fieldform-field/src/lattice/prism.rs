// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The prism cells: a planar tiling of walls, carried straight through the map's third axis.

use nalgebra::Vector3;

/// A planar tiling carried straight through the map's third axis. Backlog item 49.
///
/// **A third sort of cell, for the same reason a beam was a second.** A surface cell is a
/// function with a magnitude that has to be bounded; a beam is a distance to a set of struts; a
/// prism is a distance to a set of *walls*, in two dimensions, and nothing at all in the third.
/// It shares the beam's exactness and the sheet's parameter -- a wall thickness -- and neither
/// enum describes it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrismKind {
    Honeycomb,
}

impl PrismKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            PrismKind::Honeycomb => "Honeycomb",
        }
    }

    /// The shape of one period, per unit of cell size: what a map multiplies its extents by.
    ///
    /// **The contract change item 49 named.** Every other cell tiles a cube, so a map could own
    /// cell size entirely and hand over one number. A row of regular hexagons repeats every
    /// flat-to-flat width one way and every `sqrt(3)` of them the other, so a hexagon drawn into
    /// a square period comes out stretched. The cell declares its shape and the map respects it;
    /// the third axis is whatever the map makes it, since the cell is constant along it.
    pub(crate) fn natural_aspect(self) -> Vector3<f64> {
        match self {
            PrismKind::Honeycomb => Vector3::new(3f64.sqrt(), 1.0, 1.0),
        }
    }

    /// The fraction of a cell a wall fills, against the wall over the flat-to-flat width.
    ///
    /// **Exact**: material is everything within half a wall of a hexagon's boundary, so what is
    /// left open is a hexagon `across − t` wide, and areas of similar hexagons go as the square
    /// of their width -- `1 − (1 − t/across)²`, which is Gibson and Ashby's corrected
    /// `2t/(√3 l) − t²/(3l²)` with `l = across/√3`, the side. `None` for a wall that fills the
    /// cell, or a number that is not a fraction of one.
    pub fn wall_density(self, wall_over_across: f64) -> Option<f64> {
        match self {
            PrismKind::Honeycomb => {
                let w = wall_over_across;
                (w.is_finite() && (0.0..1.0).contains(&w)).then_some(1.0 - (1.0 - w) * (1.0 - w))
            }
        }
    }

    /// Distance from `(x, y)` to the nearest wall's mid-plane, in the units of `across`, which
    /// is the hexagon's flat-to-flat width.
    ///
    /// Exact. Regular hexagons are the Voronoi cells of their own centres, so the nearest centre
    /// names the hexagon the point is in, and inside a convex polygon the distance to its
    /// boundary is the distance to its nearest edge. Two interleaved rectangular lattices of
    /// centres, `sqrt(3) * across` by `across`, the second offset by half a period both ways;
    /// flats face up and down.
    pub(crate) fn wall_distance(self, x: f64, y: f64, across: f64) -> f64 {
        match self {
            PrismKind::Honeycomb => {
                let (px, py) = (3f64.sqrt() * across, across);
                let near = |cx: f64, cy: f64| (x - cx, y - cy);
                let a = near((x / px).round() * px, (y / py).round() * py);
                let b = near(
                    ((x / px - 0.5).round() + 0.5) * px,
                    ((y / py - 0.5).round() + 0.5) * py,
                );
                let (qx, qy) = if a.0 * a.0 + a.1 * a.1 <= b.0 * b.0 + b.1 * b.1 { a } else { b };
                // Inside a flat-topped hexagon of apothem `r`: fold into one sextant, and the
                // edge is the horizontal one at `y = r`.
                let r = across * 0.5;
                let (kx, ky) = (-0.866_025_403_784_438_6_f64, 0.5_f64);
                let (mut qx, mut qy) = (qx.abs(), qy.abs());
                let fold = 2.0 * (kx * qx + ky * qy).min(0.0);
                qx -= fold * kx;
                qy -= fold * ky;
                let half_edge = r / 3f64.sqrt();
                let dx = qx - qx.clamp(-half_edge, half_edge);
                let dy = qy - r;
                // Negative inside, so the distance to the wall is its size.
                (dx * dx + dy * dy).sqrt() * if dy < 0.0 { 1.0 } else { -1.0 }
            }
        }
    }
}
