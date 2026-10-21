// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Why a thin drafted extrude comes out stepped along its edge.
//!
//! Reported from a model on screen: a 173 mm extrude at -0.8 degrees off a profile made of a
//! 66 mm degree-two curve and a 64 mm line, showing a staircase down one side. Is it because the
//! part is small, or does scale not matter?
//!
//! Three suspects, and the point of this is to tell them apart rather than pick one:
//!
//! - the **bake**, since a profile answers from a grid when the tolerance allows one,
//! - the **tessellation**, since a curve reaches the field as a polyline and the shader walks it,
//! - the **draft**, since it moves the wall off the profile and so asks about a different contour.
//!
//! Run with: `cargo run --release -p fieldform-field --example knife_edge`

use fieldform_field::{bake_cells, Profile};
use nalgebra::Point2;

/// The model on screen, near enough.
const SPAN_MM: f64 = 66.0;
const DEPTH_MM: f64 = 173.0;
const DRAFT_DEG: f64 = 0.8;
/// What the viewport asks for at a normal camera distance.
const TOLERANCE_MM: f64 = 0.25;

fn main() {
    let reads = DEPTH_MM * DRAFT_DEG.to_radians().tan();
    println!(
        "A {SPAN_MM} mm profile extruded {DEPTH_MM} mm at {DRAFT_DEG} degrees. The far end's wall\n\
         is the {reads:.3} mm contour of the profile, so the field is asked about the profile\n\
         that far off its own boundary.\n"
    );

    // ---------------------------------------------------------------- the bake, ruled out
    println!("First suspect, the bake. How fine a grid the profile is answered from:\n");
    println!("  {:>10}  {:>9}  {:>9}  {:>9}", "span mm", "reads to", "cells", "cell mm");
    for scale in [0.1f64, 1.0, 4.0] {
        let span = SPAN_MM * scale;
        let out = DEPTH_MM * scale * DRAFT_DEG.to_radians().tan();
        match bake_cells(span, TOLERANCE_MM, (0.0, out)) {
            Some(cells) => println!(
                "  {span:>10.1}  {out:>9.3}  {cells:>9}  {:>9.4}",
                span / cells as f64
            ),
            None => println!("  {span:>10.1}  {out:>9.3}  {:>9}  {:>9}", "walked", "-"),
        }
    }
    println!(
        "\nThe cell tracks the span, so the same shape at any size bakes to the same picture --\n\
         **scale does not matter**, which answers half the question. And a cell of about a\n\
         twentieth of a millimetre on a 66 mm part is far below a pixel at any sane camera\n\
         distance, so the bake is not what is being seen."
    );

    // ---------------------------------------------------------------- the tessellation
    println!(
        "\nSecond suspect, the tessellation. A curve reaches the field as a polyline, refined\n\
         until no joint turns more than the display setting allows:\n"
    );
    println!(
        "  {:>12}  {:>8}  {:>10}  {:>12}  {:>14}",
        "max turn", "points", "facet mm", "sag mm", "sag at the wall"
    );
    // The arc the profile's long side is: 66 mm of a shallow degree-two curve. Its total turn is
    // what decides the point count, so it is stated rather than assumed.
    let total_turn_deg = 42.0;
    for turn in [4.0f64, 2.0, 1.0, 0.25] {
        let points = (total_turn_deg / turn).ceil().max(1.0);
        let facet = SPAN_MM / points;
        // Sagitta of a chord subtending `turn` on a circle of the same radius the arc has.
        let radius = SPAN_MM / total_turn_deg.to_radians();
        let sag = radius * (1.0 - (turn.to_radians() * 0.5).cos());
        // The same facet on the inward contour the drafted wall reads, whose radius is smaller
        // by exactly how far in it sits -- so the same angular step covers less arc and the
        // corner between two facets is sharper.
        let sag_wall = (radius - reads) * (1.0 - (turn.to_radians() * 0.5).cos());
        println!(
            "  {turn:>11.2}\u{b0}  {points:>8.0}  {facet:>10.2}  {sag:>12.5}  {sag_wall:>14.5}"
        );
    }
    println!(
        "\n`facet mm` is how wide one flat of the extruded solid is. At the 2 degrees a Medium\n\
         display uses, a 42-degree curve becomes 21 flats of about 3 mm each -- and a flat is\n\
         flat, so on a silhouette seen nearly edge-on each one ends where the next begins. That\n\
         is a step, and there are as many of them as there are flats."
    );

    // ---------------------------------------------------------------- the draft, ruled out too
    //
    // It was the obvious suspect and it is not the cause, which is worth printing rather than
    // quietly dropping: an inward contour has a smaller radius, so the same angular joint covers
    // less arc, and the guess was that the flats would bite harder out there. The numbers say
    // they barely change.
    println!(
        "\nThird suspect, the draft. A negative one closes the walls in, so the far end reads the\n\
         contour {reads:.3} mm *inside* the profile, where every arc is tighter. If that\n\
         sharpened the flats it would be the reason a deep shallow-drafted extrude shows them:\n"
    );
    let radius = SPAN_MM / 42f64.to_radians();
    let sag = |r: f64| r * (1.0 - (1f64.to_radians()).cos());
    println!(
        "  by arithmetic the flats sag {:.5} mm on the boundary and {:.5} mm on the wall's\n\
         \x20 contour \u{2014} {:.1}% less, which is nothing. The draft moves *which* contour is\n\
         \x20 read; it does not make the reading any coarser.",
        sag(radius),
        sag(radius - reads),
        (1.0 - sag(radius - reads) / sag(radius)) * 100.0
    );

    // The same number from the field rather than from trigonometry, because the arithmetic above
    // assumes the facets are chords of a circle and the thing on screen is whatever the
    // tessellation actually produced. Sampling the faceted profile *along the true arc*: where
    // the two agree the reading is nought, and the largest reading is the sag.
    let faceted = Profile::new(arc_and_line(SPAN_MM * 0.5, 42.0, 2.0)).expect("closes");
    let mut measured = 0.0f64;
    for step in 0..2001 {
        let a = (-21.0 + 42.0 * step as f64 / 2000.0).to_radians();
        let on_arc = Point2::new(
            radius * a.sin(),
            radius * a.cos() - radius + SPAN_MM * 0.5,
        );
        measured = measured.max(faceted.distance(on_arc).abs());
    }
    println!(
        "  measured off the field, the faceted boundary is {measured:.5} mm from the true arc at\n\
         \x20 its worst, against {:.5} predicted \u{2014} so the flats are what the picture is\n\
         \x20 made of, not an artefact of the way this was reasoned about.",
        sag(radius)
    );

    println!(
        "\nSo: not the size, not the bake, not the draft. It is the tessellation, and \u{201c}does\n\
         scale matter\u{201d} has a clean answer \u{2014} no. The bake's cell tracks the span, and\n\
         the tessellation is an *angle*, which has no length in it at all. The same shape at ten\n\
         times the size shows the same steps in the same places.\n\n\
         What decides it is how far the profile turns and how coarsely that turn is cut up. A\n\
         knife edge is the worst case because the silhouette runs *along* a flat rather than\n\
         across it, so one flat holds one normal for the whole length of the edge and the next\n\
         one starts somewhere else.\n\n\
         \x20 \u{2022} Display resolution to High takes the joint from 2 degrees to 1 and doubles\n\
         \x20   the flats. It costs frame time linearly \u{2014} 433 points against 217 on a cubic\n\
         \x20   ring, about 75 ms against 37 for one revolve at 720 \u{2014} and a pattern\n\
         \x20   multiplies that by its count.\n\
         \x20 \u{2022} The fix is that the shader walks every point of the profile at every sample,\n\
         \x20   which is what makes the trade linear and forces the setting to exist at all.\n\
         \x20   That is backlog 33; until then this is a dial rather than a defect."
    );
}

/// A shallow arc closed by a straight line — the shape of the reported profile.
fn arc_and_line(half: f64, turn_deg: f64, joint_deg: f64) -> Vec<Point2<f64>> {
    let radius = (half * 2.0) / turn_deg.to_radians();
    let steps = (turn_deg / joint_deg).ceil() as usize;
    let mut out = Vec::with_capacity(steps + 2);
    for i in 0..=steps {
        let a = (-turn_deg * 0.5 + turn_deg * i as f64 / steps as f64).to_radians();
        // Centred so the arc's middle sits at the top and its ends come back down.
        out.push(Point2::new(radius * a.sin(), radius * a.cos() - radius + half));
    }
    out
}
