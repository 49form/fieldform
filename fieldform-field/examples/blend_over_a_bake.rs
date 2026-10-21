// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a fillet reads off a baked profile, and whether the band goes that far.
//!
//! Reported from the application: a 10 mm fillet between an extrude and a sphere came out
//! faceted, in terraces, and the owner guessed the bake. This checks that.
//!
//! The mechanism would be the one a draft already hit. `Node::Blend` evaluates both operands
//! wherever either is within `k` of its own surface — that is what a fillet *is* — so a 10 mm
//! fillet reads the extrude's field out to 10 mm, and the extrude reads its profile out to 10 mm
//! from the profile's boundary. A baked profile answers finely only inside a band about four
//! tolerances wide. Past it the coarse level answers: bilinear over blocks eight cells across,
//! which is piecewise-flat rather than smooth, and a blend is exactly the operation that turns a
//! kink in an operand into a visible crease.
//!
//! Run with: `cargo run --release -p fieldform-field --example blend_over_a_bake`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Fillet, BooleanOp, Continuity, Curve, CurveId, Document, Graph, Node, ParamRef};
use nalgebra::Point3;

const HALF: f64 = 20.0;

fn main() {
    println!(
        "A {} mm square extruded, filleted to a sphere beside it. The fillet reads the\n\
         extrude's field out to its own radius, and the extrude reads the profile that far\n\
         from the boundary.\n",
        HALF * 2.0
    );
    println!(
        "  {:>9}  {:>7}  {:>10}  {:>15}  {:>13}  {:>8}  {:>10}",
        "tolerance", "fillet", "band goes", "worst difference", "its steepest", "blocks", "to build"
    );
    for tolerance in [0.5f64, 0.1] {
        for radius in [1.0f64, 5.0, 10.0] {
            let (worst, roughest, band, blocks, built) = probe(tolerance, radius);
            println!(
                "  {tolerance:>9}  {radius:>7}  {band:>10.2}  {worst:>15.4}  {roughest:>13.1}                   {blocks:>8}  {built:>7.0} ms"
            );
        }
    }
    println!(
        "\n`band goes` is the band the bake actually laid down, which now follows the fillet\n\
         rather than the tolerance. It was `4 x tolerance`, so turning quality *up* narrowed\n\
         it and put more of the fillet on the coarse level -- which is why the picture used\n\
         to get better while being dragged, a draft frame being coarser and so wider-banded.\n\n\
         The two middle columns are the baked scene against the exact one over exactly the\n\
         region a fillet of that radius reads. Before the band followed the fillet they read\n\
         0.89 mm and 24.7 per mm: a step, which a blend turns into a terrace. A distance may\n\
         change by at most one per millimetre, so the right-hand figure staying under one is\n\
         the whole point.\n\n\
         The last row shows the budget at work -- past 32768 blocks the grid coarsens rather\n\
         than the band stopping short, because a smooth approximate answer is what a blend\n\
         needs and a sharp stepped one is not."
    );
}

/// How far the baked scene's field departs from the exact one over the region a fillet uses.
///
/// **Compared as fields over the whole crease, not along a chosen line.** Two earlier versions
/// of this hunted for the surface along `+x` and found the sphere's far side instead — the
/// fillet lives in the crease between the two bodies, which a straight scan out of the middle
/// never crosses. Differencing the fields over the region needs no surface at all, and it
/// answers what was asked: whether baking the profile changes what the blend is built from.
///
/// Reports the worst difference in millimetres, and the worst rate of change of that
/// difference — a bake that is merely *offset* moves a surface, while one that is *stepped*
/// creases it, and a fillet turns a crease in an operand into a visible terrace.
fn probe(tolerance: f64, radius: f64) -> (f64, f64, f64, usize, f64) {
    let started = std::time::Instant::now();
    let baked = build(tolerance, radius);
    let built = started.elapsed().as_secs_f64() * 1e3;
    // The same scene compiled fine enough that the profile keeps its exact boundary. The only
    // difference between the two is the grid.
    let exact = build(1e-5, radius);

    let mut worst: f64 = 0.0;
    let mut roughest: f64 = 0.0;
    let step = (tolerance * 0.05).max(1e-3);
    // The crease: from well inside the extrude's wall out past where a fillet of this radius
    // can reach, over the height the sphere meets it.
    let mut z = 0.5;
    while z < 15.5 {
        // **Exactly what a fillet of this radius reads**, which is where its operand is within
        // `k` of its own surface -- no wider, or the window includes a cliff the fillet never
        // touches and every radius looks equally bad.
        let mut x = HALF - radius;
        let mut previous = f64::NAN;
        while x < HALF + radius {
            let at = Point3::new(x, 0.0, z);
            let gap = baked.distance(at) - exact.distance(at);
            worst = worst.max(gap.abs());
            if previous.is_finite() {
                roughest = roughest.max(((gap - previous) / step).abs());
            }
            previous = gap;
            x += step;
        }
        z += 0.25;
    }
    // The band the bake actually laid down, reported rather than restated from a rule -- the
    // rule was `4 * tolerance` and is not any more, and a column that repeats a stale rule is
    // worse than no column.
    let (band, blocks) = describe(tolerance, radius);
    (worst, roughest, band, blocks, built)
}

/// How far the fine band reaches and how many blocks it took, from the bake itself.
fn describe(tolerance: f64, radius: f64) -> (f64, usize) {
    let profile = fieldform_field::Profile::new(vec![
        nalgebra::Point2::new(-HALF, -HALF),
        nalgebra::Point2::new(HALF, -HALF),
        nalgebra::Point2::new(HALF, HALF),
        nalgebra::Point2::new(-HALF, HALF),
    ])
    .expect("closed");
    let (lo, hi) = profile.bounds();
    let span = (hi - lo).max();
    let reads = (-radius, radius);
    let Some(cells) = fieldform_field::bake_cells(span, tolerance, reads) else {
        return (0.0, 0);
    };
    let baked = profile.baked(cells, reads);
    let band = fieldform_field::band_range(span, reads).1;
    let stored = baked.bake().map_or(0, |b| b.stored_blocks().0);
    (band, stored)
}

/// A square extruded, filleted to a sphere set beside its corner.
fn build(tolerance: f64, radius: f64) -> GraphField {
    let mut doc = Document::new();
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            "Square".to_string(),
            vec![
                Point3::new(-HALF, -HALF, 0.0),
                Point3::new(HALF, -HALF, 0.0),
                Point3::new(HALF, HALF, 0.0),
                Point3::new(-HALF, HALF, 0.0),
            ],
            true,
        )
        .expect("closed"),
    );
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let solid = graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(id),
        sweep: fieldform_scene::Sweep::Normal,
        height: lit(16.0),
        draft: lit(0.0),
        start: lit(0.0),
        sides: fieldform_scene::Sides::One,
        corners: Default::default(),
    });
    let ball = graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([lit(HALF + 6.0), lit(0.0), lit(8.0)]),
        radius: lit(12.0),
    });
    graph.add(Node::Blend {
        op: BooleanOp::Union,
        a: solid.into(),
        b: ball.into(),
        fillet: Fillet { radius: lit(radius), continuity: Continuity::G1 },
    });
    GraphField::compile_graph(&doc, &graph, tolerance).expect("compiles")
}
