// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What settling a document's cages costs, cage by cage, and where in the bake it goes.
//!
//! `bake_cost` times the starting cube. A cube's field is cheap to ask wherever you stand, so it
//! says nothing about a cage of thin branching members, where a sample far from the surface has
//! many members about equally near and the hierarchy cannot rule most of them out. This reads a
//! saved document and, for each cage a part draws, lays out the box, the band and the subdivision
//! depth as the viewport does, then times:
//!
//! - building the cage's field;
//! - the draft bake, the half-way one and the settled one, with how many samples were evaluated
//!   rather than skipped as provably past the band;
//! - the settled bake with the distance and the normal asked for apart, which is two searches a
//!   sample and what a bake did before `ScalarField::distance_and_normal`;
//! - the settled bake with a band a few cells wide, which is the part of the cost that is the
//!   band's width and not the surface;
//! - one distance query, by how far from the surface it was asked.
//!
//! Run with: `cargo run -p fieldform-gpu --example cage_bake_cost -- chair.fform [cells]`
//! in the profile the application is run in, since the dev profile is what a session pays.
//!
//! **What it found** is in the table `main` prints; the numbers this was written from are in
//! the comment on `main`.

use fieldform_field::{levels_for_tolerance, CageField, ScalarField};
use fieldform_gpu::BakedField;
use fieldform_scene::Node;
use nalgebra::{Point3, Vector3};
use std::time::Instant;

/// The viewport's box for a cage: its hull, and a twentieth of its diagonal all round.
fn cage_box(cage: &fieldform_cage::QuadMesh) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    let margin = (hi - lo).norm() * 0.05;
    let pad = Vector3::new(margin, margin, margin);
    (lo - pad, hi + pad)
}

/// How many of a bake's samples hold a value inside the band, which is to say were evaluated.
fn evaluated(baked: &BakedField) -> usize {
    let edge = (baked.band * 0.999) as f32;
    baked.data.iter().filter(|v| v.abs() < edge).count()
}

/// A cage's field that answers its distance and its normal only apart: the control arm, the
/// same field and the same answers at the cost of a search for each.
struct Apart<'a>(&'a CageField);

impl ScalarField for Apart<'_> {
    fn distance(&self, p: Point3<f64>) -> f64 {
        self.0.distance(p)
    }

    fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
        ScalarField::normal(self.0, p)
    }
}

/// Measured on a chair of five cages -- a seat and a back 8 mm thick, three frames of bones 12
/// to 28 mm across -- in the dev profile on 16 threads, 2026-11-15, settled at the 192 cells of
/// the top two display rungs. Seconds:
///
/// | cage | faces | field | settled, asked apart | settled | evaluated | half-way | band of 8 cells |
/// |---|---|---|---|---|---|---|---|
/// | seat | 96 | 0.08 | 1.7 | 1.0 | 97% of 2.0 M | 0.14 | 0.4 |
/// | back | 82 | 0.05 | 3.1 | 1.8 | 83% of 2.9 M | 0.28 | 0.5 |
/// | rear frame, low | 342 | 0.39 | 7.3 | 4.0 | 84% of 4.7 M | 0.54 | 1.3 |
/// | front frame | 426 | 0.39 | 9.9 | 5.8 | 80% of 6.0 M | 0.75 | 1.4 |
/// | rear frame, high | 510 | 0.58 | 6.9 | 3.6 | 78% of 3.7 M | 0.59 | 1.1 |
/// | all five | | 1.5 | 28.8 | 16.2 | | 2.3 | 4.7 |
///
/// The band is 15% of the box's diagonal, 94 to 121 mm here, which is wider than the gaps
/// between the bones: nothing is past it, so the "narrow band" is nearly the whole box and
/// three quarters of a bake is exact distances far from any surface. A query costs 2 to 7 us
/// beside the surface and 9 to 25 us past 80 mm, since far from a frame of thin members many
/// are about equally near.
///
/// Asking for the distance and the normal together is 1.8 times quicker and changes nothing in
/// the band. The band's width is left alone: an offset, a blend and a ramp all read a cage's
/// field far from its surface, and what a narrow band costs the march in empty space is not
/// measured. The half-way column is what the stand-in at half the cells costs: a seventh of
/// the settled bake, and it is what the scene is drawn from while that is made. The same
/// machine read every row half as fast an hour earlier with two other sessions open, so the
/// columns are to be compared across, not down the afternoon.
fn main() {
    let path = std::env::args().nth(1).expect("a .fform path");
    let cells: usize = std::env::args().nth(2).and_then(|n| n.parse().ok()).unwrap_or(192);
    let (doc, _archive) = fieldform_io::open(std::path::Path::new(&path)).expect("the document opens");
    println!("{} threads; settled at {cells} cells", rayon::current_num_threads());
    let drawn: Vec<_> = doc
        .parts
        .values()
        .flat_map(|part| part.graph.nodes())
        .filter_map(|(_, node)| match node {
            Node::Cage(id) => Some(*id),
            _ => None,
        })
        .collect();
    let (mut apart_s, mut settled_s, mut half_s, mut narrow_s, mut field_s) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for id in drawn {
        let Some(body) = doc.cages.get(&id) else { continue };
        let Some(cage) = body.cage() else {
            println!("cage {id}: not a solid, skipped");
            continue;
        };
        let (lo, hi) = cage_box(body.mesh());
        let extent = hi - lo;
        let longest = extent.x.max(extent.y).max(extent.z);
        let band = extent.norm() * 0.15;
        let cell = longest / cells as f64;
        let levels = levels_for_tolerance(extent.norm(), (cell * 0.5).max(1e-4));
        println!(
            "\ncage {id}: {} faces, box {:.0} x {:.0} x {:.0} mm, cell {cell:.2} mm, band {band:.0} mm, {levels} levels",
            body.mesh().face_count(),
            extent.x,
            extent.y,
            extent.z,
        );

        let start = Instant::now();
        let field = CageField::from_cage(cage, levels).expect("the cage subdivides");
        let field_ms = start.elapsed().as_secs_f64() * 1e3;
        field_s += field_ms / 1e3;
        println!("  field built in {field_ms:.0} ms");

        let apart = Apart(&field);
        let arms: [(&str, &(dyn ScalarField + Sync), usize, f64); 6] = [
            ("draft", &field, 32, band),
            ("half-way", &field, cells / 2, band),
            ("settled, asked apart", &apart, cells, band),
            ("settled", &field, cells, band),
            ("settled, band of 8 cells", &field, cells, 8.0 * cell),
            ("settled, band of 4 cells", &field, cells, 4.0 * cell),
        ];
        for (label, asked, resolution, width) in arms {
            let start = Instant::now();
            let baked = BakedField::new(asked, (lo, hi), resolution, width);
            let ms = start.elapsed().as_secs_f64() * 1e3;
            let total = baked.data.len();
            let done = evaluated(&baked);
            println!(
                "  {label:<26} {ms:>8.0} ms  {} x {} x {} = {:.2} M samples, {:.0}% evaluated, {:.1} us each on a thread",
                baked.dims[0],
                baked.dims[1],
                baked.dims[2],
                total as f64 / 1e6,
                100.0 * done as f64 / total as f64,
                ms * 1e3 * rayon::current_num_threads() as f64 / done.max(1) as f64,
            );
            match label {
                "half-way" => half_s += ms / 1e3,
                "settled, asked apart" => apart_s += ms / 1e3,
                "settled" => settled_s += ms / 1e3,
                "settled, band of 8 cells" => narrow_s += ms / 1e3,
                _ => {}
            }
        }

        // One query, by how far from the surface it was asked: a lattice of points through the
        // box, each timed alone on this thread.
        let mut buckets = [(0usize, 0.0f64); 5];
        let edges = [5.0, 15.0, 40.0, 80.0, f64::MAX];
        let n = 22;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let p = lo + Vector3::new(
                        extent.x * (i as f64 + 0.5) / n as f64,
                        extent.y * (j as f64 + 0.5) / n as f64,
                        extent.z * (k as f64 + 0.5) / n as f64,
                    );
                    let start = Instant::now();
                    let d = field.distance(p).abs();
                    let us = start.elapsed().as_secs_f64() * 1e6;
                    let b = edges.iter().position(|e| d < *e).unwrap_or(4);
                    buckets[b].0 += 1;
                    buckets[b].1 += us;
                }
            }
        }
        print!("  one query, by distance from the surface:");
        let names = ["under 5 mm", "5-15", "15-40", "40-80", "past 80"];
        for (name, (count, us)) in names.iter().zip(buckets) {
            if count > 0 {
                print!("  {name}: {:.0} us ({count})", us / count as f64);
            }
        }
        println!();
    }
    println!(
        "\nall cages, one after another: fields {field_s:.1} s; settled, asked apart {apart_s:.1} s; \
         settled {settled_s:.1} s; half-way {half_s:.1} s; settled with a band of 8 cells {narrow_s:.1} s"
    );
}
