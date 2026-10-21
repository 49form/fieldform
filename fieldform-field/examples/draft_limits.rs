// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a draft asks of a baked profile, and what it costs to answer.
//!
//! Reported from the application: past some combination of depth and draft angle the solid
//! pinches and speckles. Speckle is a ray march stepping *through* a surface, which happens when
//! a field overstates — and the one place a baked profile does is the edge of its band, where the
//! reading jumps from the fine level's margin to the coarse level's, eight times wider.
//!
//! It never mattered before because an undrafted extrude's surface sits on the profile's own zero
//! set, which is what the band is built around. **A draft moves the surface off it**: at height
//! `z` the wall is the `d = z·tan(draft)` contour, so a 12 mm extrude at 15 degrees reads out to
//! 3.2 mm while the band is four tolerances — 0.2 mm at the finest setting — wide.
//!
//! Run with: `cargo run --release -p fieldform-field --example draft_limits`

use fieldform_field::Profile;
use nalgebra::Point2;

/// How far a 12 mm extrude at 15 degrees of draft reads off its profile, in millimetres.
const READS: f64 = 3.2154;

fn main() {
    println!("How far a draft reads off the zero set, against how far the band used to go:\n");
    println!("  tolerance   band was   12 mm at 15°   40 mm at 45°");
    for tol in [0.05f64, 0.1, 0.25, 0.5, 2.0] {
        println!(
            "  {tol:>9}   {:>8.2}   {:>12.2}   {:>12.2}",
            // `reach = BLOCK * cell * sqrt(2)` with `cell = tolerance * 0.25 * sqrt(2)`, so the
            // band was four tolerances however fine the grid — it scales with the cell, and a
            // draft does not.
            4.0 * tol,
            12.0 * 15f64.to_radians().tan(),
            40.0 * 45f64.to_radians().tan()
        );
    }

    println!("\nWhat that does to the field, and what covering it costs:\n");
    println!(
        "  {:>9} {:>9} {:>11} {:>10} {:>8} {:>10} {:>11}",
        "tolerance", "reads to", "steepest", "short by", "blocks", "bake", "to build"
    );
    // Both a four-segment square and a five-hundred-segment loop, because the bake's cost is a
    // boundary walk per sample and the walk is linear in the segment count. A square measures
    // the band; a tessellated curve measures what the application actually pays.
    for segments in [4usize, 500] {
        println!("  -- a profile of {segments} segments --");
        for tol in [0.5f64, 0.1, 0.05] {
            for reads_out_to in [0.0f64, READS] {
                measure(tol, reads_out_to, segments);
            }
        }
    }
    println!(
        "\nA distance may rise at most one millimetre per millimetre. A band that stops short of\n\
         what is read leaves a cliff at its edge — the ray is told it has a hundred times the\n\
         room it has, and goes through the surface."
    );
}

/// The field's steepest rise over the contours a draft reads, and what covering them costs.
///
/// **The window is the same whether or not the band was widened**, because the question is what
/// the field does where the drafted wall actually is. Scanning each bake out to its own band
/// edge would compare two different regions and report that both were fine, which is how this
/// measurement went wrong once before.
fn measure(tol: f64, reads_out_to: f64, segments: usize) {
    let half = 10.0;
    let points: Vec<Point2<f64>> = if segments == 4 {
        vec![
            Point2::new(-half, -half),
            Point2::new(half, -half),
            Point2::new(half, half),
            Point2::new(-half, half),
        ]
    } else {
        // A rounded square, so the walk has as many segments as a tessellated curve does and
        // the boundary is still a known distance from the scan line.
        (0..segments)
            .map(|i| {
                let a = i as f64 / segments as f64 * std::f64::consts::TAU;
                Point2::new(half * a.cos(), half * a.sin())
            })
            .collect()
    };
    let profile = Profile::new(points).expect("closed");
    let (lo, hi) = profile.bounds();
    let cells =
        fieldform_field::bake_cells((hi - lo).max(), tol, (0.0, reads_out_to)).expect("bakes");
    let started = std::time::Instant::now();
    let baked = profile.clone().baked(cells, (0.0, reads_out_to));
    let built_ms = started.elapsed().as_secs_f64() * 1e3;

    // Stepped far finer than a cell, so the step from the last fine block to the first coarse
    // one is landed on rather than jumped over. A central difference at 1e-4 mm misses it: the
    // cliff is a hair wide, and a probe narrower than the sample spacing walks straight past.
    let step = tol * 0.01;
    let mut steepest: f64 = 0.0;
    let mut short: f64 = 0.0;
    // From just inside the boundary out to the furthest contour a 12 mm wall at 15 degrees
    // reaches -- the case that was reported.
    let near = half - tol;
    let far = half + READS + tol;
    let mut x = near;
    let mut previous = baked.distance(Point2::new(x, 0.0));
    while x < far {
        x += step;
        let here = baked.distance(Point2::new(x, 0.0));
        // The magnitude, not the rise: a ray travels *inward*, so the drop this scan sees
        // going out is the jump it sees coming in. Either sign over one is a field changing
        // faster than a distance can.
        steepest = steepest.max(((here - previous) / step).abs());
        short = short.max(profile.walk(Point2::new(x, 0.0)) - here);
        previous = here;
    }

    let bake = baked.bake().expect("baked");
    let (stored, all) = bake.stored_blocks();
    println!(
        "  {tol:>9} {reads_out_to:>9.2} {steepest:>11.1} {short:>10.4} {:>8} {:>7.0} KB {built_ms:>8.0} ms",
        format!("{stored}/{all}"),
        bake.samples() as f64 * 4.0 / 1024.0
    );
}
