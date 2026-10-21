// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How well a packed surface query agrees with the surface itself, as the surface curves.
//!
//! **Written to separate two stories about one measurement.** `gpu_matches_cpu`'s conformal scene
//! agrees to 0.0131 mm over a flat panel — as well as any scene in that suite — and drifts to
//! 1.30 mm as the same panel is domed. Two things could do that:
//!
//! - the shader's `f32` arithmetic, which the field does in `f64`; or
//! - the closest-point map itself being genuinely multi-valued away from the surface, so that
//!   two implementations pick different branches of a real ambiguity rather than one of them
//!   being wrong.
//!
//! This asks the second question with the first one removed: `query_packed` reads exactly the
//! `f32` arrays a shader reads but does its arithmetic in `f64`, so a disagreement here is the
//! map's own ambiguity and a disagreement only on the GPU is precision.
//!
//! ```text
//! cargo run --release -p fieldform-field --example conformal_agreement
//! ```

use fieldform_cage::QuadMesh;
use fieldform_field::{query_packed, SurfaceMap};
use nalgebra::Point3;

/// A square panel bent into a dome, `rise` millimetres at the middle.
fn dome(n: usize, span: f64, rise: f64) -> QuadMesh {
    let step = span / n as f64;
    let mut positions = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (i as f64 * step - span * 0.5, j as f64 * step - span * 0.5);
            let (u, v) = (x / (span * 0.5), y / (span * 0.5));
            positions.push(Point3::new(x, y, rise * (1.0 - u * u) * (1.0 - v * v)));
        }
    }
    let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
    let mut faces = Vec::new();
    for j in 0..n {
        for i in 0..n {
            faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    QuadMesh::from_faces(positions, &faces).expect("a domed panel")
}

fn main() {
    // The scene's own proportions: a 4x4 panel over 0.8 of the part, sampled through the slab
    // that confines the lattice.
    let part = 1000.0;
    let span = part * 0.8;
    println!(
        "{:>8}  {:>6}  {:>9}  {:>9}  {:>9}  {:>8}",
        "rise/span", "levels", "worst s,t", "worst mm", "worst deg", "split"
    );
    for (fraction, levels) in [
        (0.0, 3),
        (0.0625, 3),
        (0.125, 3),
        (0.25, 3),
        (0.3125, 3),
        // The same curvatures again, resolved more finely. **How much of the ambiguity is the
        // tessellation and how much is the map** — a coarser surface has larger facets, and a
        // facet's edge and corner Voronoi regions are where two answers are equidistant. If
        // these rows are much better than those, the depth `surface_levels` picks is buying
        // agreement as well as costing time.
        (0.0625, 5),
        (0.125, 5),
        (0.25, 5),
        (0.3125, 5),
    ] {
        let rise = span * fraction;
        let map = SurfaceMap::new(&dome(4, span, rise), levels).expect("tessellates");
        let packed = map.packed();

        let mut worst_st = 0.0f64;
        let mut worst_offset = 0.0f64;
        let mut worst_normal = 0.0f64;
        let mut split = 0usize;
        let mut checked = 0usize;
        // A lattice fills the slab it is confined to, so that is the volume worth agreeing over:
        // the same box `gpu_matches_cpu` intersects with, on a grid rather than a ray.
        let steps = 40;
        for i in 0..steps {
            for j in 0..steps {
                for k in 0..steps {
                    let f = |n: usize| n as f64 / (steps - 1) as f64;
                    let p = Point3::new(
                        // **Past the rim, not only over the panel.** A shelled lattice has no
                        // confining body, so it reaches beyond the boundary where the nearest
                        // point is an edge or a corner rather than a face -- which is where
                        // `gpu_matches_cpu`'s shelled scene puts its worst disagreement, and so
                        // is the region this has to cover to say whether that is ambiguity.
                        (f(i) - 0.5) * span * 1.2,
                        (f(j) - 0.5) * span * 1.2,
                        rise * 0.5 + (f(k) - 0.5) * (rise * 2.0).max(part * 0.1),
                    );
                    let hit = map.at(p);
                    let got = query_packed(&packed, p);
                    worst_st = worst_st
                        .max((got.st[0] - hit.st[0]).abs())
                        .max((got.st[1] - hit.st[1]).abs());
                    worst_offset = worst_offset.max((got.offset - hit.offset).abs());
                    worst_normal =
                        worst_normal.max(got.normal.angle(&hit.normal).to_degrees());
                    if got.triangle != hit.triangle {
                        split += 1;
                    }
                    checked += 1;
                }
            }
        }
        println!(
            "{fraction:>9.4}  {levels:>6}  {worst_st:>9.2e}  {worst_offset:>9.2e}  \
             {worst_normal:>9.2e}  {:>7.3}%",
            100.0 * split as f64 / checked as f64,
        );
    }
    println!(
        "\n`query_packed` reads the shader's own f32 arrays and does the arithmetic in f64.\n\
         A row that stays small says the map is single-valued over this volume and the\n\
         disagreement on the GPU is precision; a row that grows says the map is ambiguous\n\
         there and no two implementations can be made to agree."
    );
}
