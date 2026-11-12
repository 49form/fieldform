// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether creasing a cage changes its field anywhere but near the crease.
//!
//! A table leg's cage with its end cap's four edges creased drew surface fragments in empty space.
//! This builds the field of that cage with and without the creases and compares their signs on a
//! grid: a crease moves the surface only near the creased edges, so a sign that differs far from
//! both surfaces is a fault in the field, not a shape. It also counts the samples that share a
//! point -- the block the descent snaps to an irregular vertex's limit -- and the tessellation's
//! open and overused edges, since the sign test needs a closed mesh.
//!
//! Measured 2026-11-12, grid 48^3 over the cage's box padded by a tenth of its 169 m diagonal:
//!
//! | level | coincident samples | sign flips | of them farther than 2% of the diagonal, before / after the weld |
//! |---|---|---|---|
//! | 4 | 112 | 136 | 147 / **0** |
//! | 5 | 672 | 136 | 20 / **0** |
//! | 6 | 3136 | 136 | 138 / **0** |
//!
//! Before the weld the worst point read 32.8 m inside while 36.5 m outside; its nearest surface
//! was a rim edge whose triangle on the side face had collapsed to no area. The 136 flips left
//! are the grid's first layer inside the cap, 0.7 m in, where the creased cap is flat and the
//! smooth one has pulled inward: a shape the crease made, not a fault.
//!
//! Run with: `cargo run --release -p fieldform-field --example crease_cap_probe`

use fieldform_cage::{Cage, VertexId};
use fieldform_field::CageField;
use fieldform_subd::LimitSurface;
use nalgebra::Point3;
use std::collections::{HashMap, HashSet};

/// The owner's table-leg cage, in millimetres, as saved after the crease.
fn table_leg() -> (Vec<Point3<f64>>, Vec<Vec<u32>>, Vec<(u32, u32)>) {
    let p = [
        [54000.0, 0.0, 7000.0], [18350.412368774418, 8947.632789611816, -6523.301124572754],
        [16629.104614257812, 8729.155540466309, -8779.63924407959], [54000.0, 0.0, -13000.0],
        [16629.104614257812, 11270.844459533691, -8779.63924407959], [54000.0, 20000.0, -13000.0],
        [18350.412368774418, 11052.367210388184, -6523.301124572754], [54000.0, 20000.0, 7000.0],
        [-74591.23992919922, 8682.605743408203, -16185.945510864258],
        [-73779.57153320312, 8677.207946777344, -22533.397674560547],
        [-73779.57153320312, 11322.792053222656, -22533.397674560547],
        [-74591.23992919922, 11317.394256591797, -16185.945510864258],
        [-89691.64276123047, 4225.790977478027, -6887.783050537109],
        [-85457.6416015625, 8463.43994140625, -21038.28811645508],
        [-85457.6416015625, 11536.56005859375, -21038.28811645508],
        [-89691.64276123047, 15774.209022521973, -6887.783050537109],
        [-92280.60150146484, 6844.611167907715, -42753.787994384766],
        [-92280.60150146484, 13155.389785766602, -42753.787994384766],
        [-91224.38049316406, 11000.0, -36988.948822021484], [-91224.38049316406, 9000.0, -36988.948822021484],
        [-100000.0, 0.0, 0.0], [-100000.0, 0.0, -20000.0], [-100000.0, 20000.0, -20000.0],
        [-100000.0, 20000.0, 0.0], [-100000.0, 0.0, -40000.0], [-100000.0, 0.0, -60000.0],
        [-100000.0, 20000.0, -60000.0], [-100000.0, 20000.0, -40000.0],
    ];
    let f: [[u32; 4]; 26] = [
        [0, 1, 2, 3], [3, 2, 4, 5], [5, 4, 6, 7], [7, 6, 1, 0], [1, 8, 9, 2], [2, 9, 10, 4],
        [4, 10, 11, 6], [6, 11, 8, 1], [8, 12, 13, 9], [10, 14, 15, 11], [11, 15, 12, 8],
        [9, 16, 17, 10], [10, 17, 18, 14], [14, 18, 19, 13], [13, 19, 16, 9], [12, 20, 21, 13],
        [13, 21, 22, 14], [14, 22, 23, 15], [15, 23, 20, 12], [19, 24, 25, 16], [16, 25, 26, 17],
        [17, 26, 27, 18], [18, 27, 24, 19], [7, 0, 3, 5], [20, 23, 22, 21], [24, 27, 26, 25],
    ];
    let creases = vec![(20, 21), (20, 23), (21, 22), (22, 23)];
    (p.iter().map(|q| Point3::new(q[0], q[1], q[2])).collect(), f.iter().map(|q| q.to_vec()).collect(), creases)
}

fn edge_uses(triangles: &[[u32; 3]]) -> (usize, usize) {
    let mut uses: HashMap<(u32, u32), usize> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    (uses.values().filter(|&&c| c == 1).count(), uses.values().filter(|&&c| c > 2).count())
}

fn main() {
    let (positions, faces, creases) = table_leg();
    let smooth = Cage::from_faces(positions.clone(), &faces).expect("the table leg is a cage");
    let mut mesh = smooth.mesh().clone();
    let pairs: Vec<(VertexId, VertexId)> =
        creases.iter().map(|&(a, b)| (VertexId::from_index(a as usize), VertexId::from_index(b as usize))).collect();
    mesh.apply_creases(&pairs).expect("every crease is an edge");
    let creased = Cage::try_from_mesh(mesh).map_err(|(_, e)| e).expect("still a cage");

    let (lo, hi) = positions.iter().fold(
        (Point3::from([f64::INFINITY; 3]), Point3::from([f64::NEG_INFINITY; 3])),
        |(lo, hi), p| (lo.inf(p), hi.sup(p)),
    );
    let size = (hi - lo).norm();
    println!("cage: {} vertices, {} faces, diagonal {:.0} mm", positions.len(), faces.len(), size);

    for levels in [4usize, 5, 6] {
        let t = LimitSurface::new(&creased).tessellate(levels);
        let (open, over) = edge_uses(&t.triangles);
        let distinct: HashSet<[u64; 3]> = t.positions.iter().map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect();
        println!(
            "
level {levels}: {} triangles, {open} open and {over} overused edges, {} samples on a point another holds",
            t.triangles.len(),
            t.positions.len() - distinct.len()
        );

        let fs = CageField::from_cage(&smooth, levels).expect("smooth field");
        let fc = CageField::from_cage(&creased, levels).expect("creased field");
        let pad = 0.1 * size;
        let n = 48;
        let (mut flips, mut far, mut worst) = (0usize, 0usize, (0.0f64, Point3::origin(), 0.0, 0.0));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let s = |a: f64, b: f64, t: usize| a - pad + (b - a + 2.0 * pad) * (t as f64 + 0.5) / n as f64;
                    let p = Point3::new(s(lo.x, hi.x, i), s(lo.y, hi.y, j), s(lo.z, hi.z, k));
                    let (ds, dc) = (fs.distance(p), fc.distance(p));
                    if ds.signum() != dc.signum() {
                        flips += 1;
                        let off = ds.abs().min(dc.abs());
                        if off > 0.02 * size {
                            far += 1;
                        }
                        if off > worst.0 {
                            worst = (off, p, ds, dc);
                        }
                    }
                }
            }
        }
        println!(
            "  grid {n}^3: {flips} sign flips, {far} farther than 2% of the diagonal from either surface; worst {:.0} mm from both at ({:.0}, {:.0}, {:.0}): smooth {:.0}, creased {:.0}",
            worst.0, worst.1.x, worst.1.y, worst.1.z, worst.2, worst.3
        );
    }
}
