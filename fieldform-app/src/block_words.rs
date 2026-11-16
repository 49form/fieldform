// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The words the panel uses for a fillet, a boolean and a lattice cell.

use crate::*;

/// How a boolean meets: a hard edge, or a fillet of some continuity.
///
/// `None` is G0 — position continuity and nothing more, which is a plain boolean rather than a
/// blend of radius zero. That is the default because it is what a boolean *is*; a fillet is
/// something you ask for.
pub(crate) type BlendMode = Option<Continuity>;

/// The continuity choices the tool offers, in the order they are shown.
pub(crate) const BLEND_MODES: [(BlendMode, &str); 4] = [
    (None, "G0"),
    (Some(Continuity::G1), "G1"),
    (Some(Continuity::G2), "G2"),
    (Some(Continuity::Chamfer), "Chamfer"),
];

/// What the row holding a fillet's number is called.
///
/// **It is a radius only at G1, and only at a right angle** — the zero set of the form leaves each
/// face where the *other* is `k` away, whatever the exponent and the angle, which is a reach. G1
/// keeps the word people expect and is a circle of that radius between square faces; G2 says what
/// the number is, because its tightest bend is not it (0.56 × at a right angle). Ruled 2026-10-02,
/// on a measurement (`Continuity`'s doc): the number stays the same one when the shape changes, so
/// switching never moves where the rounding starts.
pub(crate) fn fillet_label(continuity: Continuity) -> &'static str {
    match continuity {
        Continuity::G1 => "Fillet:",
        Continuity::G2 => "Reach:",
        // The leg, which is the reach under its own name: how far along each face the cut
        // runs. A chamfer's size is said that way on every drawing.
        Continuity::Chamfer => "Chamfer:",
    }
}

/// The read-only line under a fillet's number saying what it does to the bend, or nothing at G1,
/// where the tightest bend *is* the number between square faces.
///
/// Derived and never stored — the number is the one thing the node holds — so the line cannot
/// disagree with it. "At a right angle" is in the text because it is a property of the corner as
/// well as of the shape: at 60° the same G2 reach bends at 0.26 ×, at 120° at 1.37 ×.
pub(crate) fn fillet_bend_note(continuity: Continuity, reach_mm: f64) -> Option<String> {
    if continuity == Continuity::Chamfer {
        return Some(format!(
            "A flat bevel, cut back {reach_mm:.2} mm along each face from the other; at a right \
             angle a 45\u{b0} chamfer of that leg."
        ));
    }
    (continuity != Continuity::G1).then(|| {
        let ratio = continuity.apex_radius_ratio();
        format!(
            "Leaves each face {reach_mm:.2} mm from the other. Tightest bend \u{2248} {:.2} mm \
             at a right angle ({ratio:.2} \u{d7} the reach).",
            ratio * reach_mm
        )
    })
}

/// What the G1 and G2 buttons say when hovered, on a blend and on a pattern's fold alike.
pub(crate) fn fillet_shape_hover(continuity: Continuity) -> &'static str {
    match continuity {
        Continuity::G1 => {
            "A circular fillet: tangent to both faces, and between square faces a circle of the \
             number below."
        }
        Continuity::G2 => {
            "A curvature-continuous fillet: it leaves each face the same reach from the other \
             as G1 does, but eases in with curvature rising from nought rather than stepping to \
             1/k, so the bend in the middle is tighter."
        }
        Continuity::Chamfer => {
            "A flat bevel: each face cut back the number below from the other, with a plane \
             between the two cuts -- the same footprint as G1 and G2, so switching moves nothing \
             but the shape between them. Its value reads short across the bevel, so the march \
             takes a few more steps there; the surface is exact."
        }
    }
}

/// A fresh phase of the given banding, sized for a cell of `cell_mm`.
///
/// One function because the panel seeds a phase from two places — switching banding, and
/// switching to a cell that cannot take the banding it had — and two copies of "what a new solid
/// starts at" would drift. Nothing carries across from the old phase, and nothing can: there is
/// no conversion between a wall of 2 mm, a density of 40% and a beam radius.
pub(crate) fn seeded_phase(banding: Banding, cell_mm: f64) -> LatticePhase {
    match banding {
        // A tenth of the cell, which is a lattice rather than a block of cheese.
        Banding::Sheet => {
            LatticePhase::sheet(ParamRef::Literal((cell_mm * 0.1).max(0.001)))
        }
        // What a part is usually printed at.
        Banding::Solid => LatticePhase::Solid { density: ParamRef::Literal(0.4) },
        // Thinner than a wall for the same cell, because a strut is material in every direction
        // where a wall is thin in one.
        Banding::Strut => {
            LatticePhase::Strut { radius: ParamRef::Literal((cell_mm * 0.06).max(0.001)) }
        }
    }
}

/// The range a fresh grade runs over: half to half again of the number the lattice had, so the
/// switch reads as a grade of the lattice on screen rather than as a resize -- and, for a
/// density, kept inside what a fraction can mean.
pub(crate) fn grade_range(banding: Banding, was: f64) -> (f64, f64) {
    match banding {
        Banding::Solid => {
            let was = if was > 1e-9 && was <= 1.0 { was } else { 0.4 };
            ((was * 0.5).clamp(0.05, 0.9), (was * 1.5).clamp(0.1, 0.95))
        }
        Banding::Sheet | Banding::Strut => {
            let was = if was > 1e-9 { was } else { DEFAULT_BODY_MM * 0.01 };
            (was * 0.5, was * 1.5)
        }
    }
}

/// A lattice cell's name, in one place so the block, the menu and the console agree.
pub(crate) fn lattice_cell_label(cell: LatticeCell) -> &'static str {
    match cell {
        LatticeCell::Gyroid => "Gyroid",
        LatticeCell::SchwarzP => "Schwarz P",
        LatticeCell::SchwarzD => "Schwarz D",
        LatticeCell::SchoenIwp => "Schoen IWP",
        LatticeCell::Neovius => "Neovius",
        LatticeCell::FischerKochS => "Fischer-Koch S",
        LatticeCell::Cubic => "Cubic",
        LatticeCell::Bcc => "BCC",
        LatticeCell::Fcc => "FCC",
        LatticeCell::Octet => "Octet",
        LatticeCell::Diamond => "Diamond",
        LatticeCell::Honeycomb => "Honeycomb",
        LatticeCell::VoronoiFoam { .. } => "Voronoi foam",
        LatticeCell::VoronoiBeam { .. } => "Voronoi beam",
    }
}

/// Every lattice cell, in the order Create lists them.
pub(crate) const LATTICE_CELLS: [LatticeCell; 14] = [
    LatticeCell::Gyroid,
    LatticeCell::SchwarzP,
    LatticeCell::SchwarzD,
    LatticeCell::SchoenIwp,
    LatticeCell::Neovius,
    LatticeCell::FischerKochS,
    // The beams last, after the six surfaces: two families in one list, in the order they were
    // built rather than interleaved, so the list reads as what it is.
    LatticeCell::Cubic,
    LatticeCell::Bcc,
    LatticeCell::Fcc,
    LatticeCell::Octet,
    LatticeCell::Diamond,
    // A prism: hexagons in the plane, straight through the third axis. Its cell size is the
    // hexagon's flat-to-flat width and its wall is exact.
    LatticeCell::Honeycomb,
    // The two built from a jittered seed grid, and the first cells carrying numbers of their
    // own. Their defaults are what `LatticeCell::voronoi_*` says, so a document that leaves
    // either out reads back as the cell this list offers.
    LatticeCell::VoronoiFoam { jitter: ParamRef::Literal(0.75), seed: 0 },
    LatticeCell::VoronoiBeam { jitter: ParamRef::Literal(0.75), seed: 0 },
];

/// What to call a boolean operation in the interface.
///
/// "Subtract" rather than "Difference" because the tool is a verb: you subtract one body from
/// another. The graph's own name for it stays `BooleanOp::Difference`.
pub(crate) fn boolean_label(op: BooleanOp) -> &'static str {
    match op {
        BooleanOp::Union => "Union",
        BooleanOp::Difference => "Subtract",
        BooleanOp::Intersection => "Intersect",
    }
}
