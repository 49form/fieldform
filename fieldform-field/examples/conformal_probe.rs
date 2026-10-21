// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! One point, both walks, every quantity the lattice reads from them.
//!
//! **A point, not a sweep, because the sweep said the wrong thing.** `gpu_matches_cpu`'s shelled
//! conformal scene disagrees by 8.352 mm and does so at *every* step budget and every hit
//! threshold — so it is neither a ray running out of steps nor one failing to converge, and a
//! grid of samples over the same volume reported the map agreeing to 1.5e-3. Something at one
//! place is different, and this asks that place directly.
//!
//! `query_packed` is the shader's own walk over the shader's own arrays, in `f64`. If it and
//! `SurfaceMap::at` answer differently here, the ambiguity is real and the grid simply missed
//! it. If they agree, the fault is in the shader.
//!
//! ```text
//! cargo run --release -p fieldform-field --example conformal_probe
//! ```

use fieldform_cage::QuadMesh;
use fieldform_field::{
    query_packed, surface_levels, CellKind, Lattice, LatticeMap, Phase, ScalarField, SurfaceMap,
    TpmsKind,
};
use nalgebra::Point3;

const CAGE_MM: f64 = 1000.0;

/// The scene's panel, exactly.
fn panel() -> QuadMesh {
    let n = 4usize;
    let span = CAGE_MM * 0.8;
    let rise = CAGE_MM * 0.05;
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
    let map = SurfaceMap::new(&panel(), surface_levels(2)).expect("tessellates");
    let packed = map.packed();

    // Where the suite reported its worst, and a short walk around it — a single point could be
    // a coincidence, its neighbourhood cannot.
    let worst = Point3::new(414.61, -370.93, -1.76);
    println!(
        "{:>26}  {:>10} {:>10} {:>10} {:>9} {:>9}",
        "point", "d(s)", "d(t)", "d(offset)", "d(nrm) deg", "same tri"
    );
    let mut shown = 0;
    for i in -2..=2i32 {
        for j in -2..=2i32 {
            for k in -2..=2i32 {
                let p = worst
                    + nalgebra::Vector3::new(f64::from(i), f64::from(j), f64::from(k)) * 4.0;
                let hit = map.at(p);
                let got = query_packed(&packed, p);
                let ds = (got.st[0] - hit.st[0]).abs();
                let dt = (got.st[1] - hit.st[1]).abs();
                let doff = (got.offset - hit.offset).abs();
                let dn = got.normal.angle(&hit.normal).to_degrees();
                // Only what differs, so the interesting rows are not buried in 125 identical
                // ones.
                if ds.max(dt) > 1e-6 || doff > 1e-6 || dn > 1e-3 || got.triangle != hit.triangle {
                    println!(
                        "({:>7.1},{:>7.1},{:>6.1})  {ds:>10.3e} {dt:>10.3e} {doff:>10.3e} \
                         {dn:>9.3e} {:>9}",
                        p.x,
                        p.y,
                        p.z,
                        got.triangle == hit.triangle,
                    );
                    shown += 1;
                }
            }
        }
    }
    if shown == 0 {
        println!("  (nothing differs anywhere in this neighbourhood)");
    }

    // And what the surface actually says there, since the numbers above are differences and a
    // difference of nothing can still sit on a quantity that is surprising.
    let hit = map.at(worst);
    println!(
        "\n  at the worst point: face {:?}, s={:.6} t={:.6}, offset={:.4} mm\n  \
         normal ({:.4}, {:.4}, {:.4}), |grad s| = {:.6}/mm, |grad t| = {:.6}/mm",
        hit.face,
        hit.st[0],
        hit.st[1],
        hit.offset,
        hit.normal.x,
        hit.normal.y,
        hit.normal.z,
        hit.ds.norm(),
        hit.dt.norm(),
    );
    println!(
        "  worst parametric gradient over the whole surface: {:.6}/mm",
        map.worst_parametric_gradient()
    );

    // --- the lattice itself, both ways -------------------------------------------------
    //
    // The surface agrees, so whatever differs is above it. This builds the scene's lattice and
    // evaluates it, beside a transcription of what `conformal_lattice_distance` computes from
    // the same hit — every intermediate printed, since the answer is a chain of five and any
    // one of them could be the one that differs.
    let depth = CAGE_MM * 0.12;
    let wall = CAGE_MM * 0.02;
    let reach = CAGE_MM * 0.05 * 0.5;
    let cells = 2u32;
    let lattice = Lattice {
        cell: CellKind::Tpms(TpmsKind::Gyroid),
        map: LatticeMap::Conformal {
            surface: std::sync::Arc::new(SurfaceMap::new(&panel(), surface_levels(cells))
                .expect("tessellates")),
            cells_per_face: cells,
            depth,
            reach: Some(reach),
        },
        phase: Phase::sheet(wall),
    };

    let tau = std::f64::consts::TAU;
    let n = tau * f64::from(cells);
    let args = nalgebra::Vector3::new(
        n * hit.st[0],
        n * hit.st[1],
        tau * hit.offset / depth,
    );
    let w = nalgebra::Vector3::new(
        n * hit.ds.norm(),
        n * hit.dt.norm(),
        tau / depth,
    );
    let kind = TpmsKind::Gyroid;
    let g = kind.argument_gradient();
    let iso = wall * 0.5
        * ((w.x * g.x).powi(2) + (w.y * g.y).powi(2) + (w.z * g.z).powi(2)).sqrt();
    let f = kind.f(nalgebra::Point3::from(args));
    let raw = f.abs() - iso;
    let frequency = (n * map.worst_parametric_gradient()).max(tau / depth);
    let bound = frequency * CellKind::Tpms(kind).gradient_bound();
    let band = hit.offset.abs() - reach;
    let by_hand = (raw / bound).max(band);

    println!(
        "
  args ({:.4}, {:.4}, {:.4})
  w ({:.6}, {:.6}, {:.6})
           F={f:.6} iso={iso:.6} raw={raw:.6}
  frequency={frequency:.6} bound={bound:.6}          band={band:.4}",
        args.x, args.y, args.z, w.x, w.y, w.z,
    );
    println!(
        "
  the shader's formula, by hand: {by_hand:.6} mm
           the field's own `distance`:     {:.6} mm",
        lattice.distance(worst),
    );
}
