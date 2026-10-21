// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Does a sweep's box still hold the sweep?
//!
//! **Written to tighten that box, and kept because it showed the tightening was wrong.** The
//! bound is a ball of the profile's reach at every spine sample (`Node::Sweep`'s arm of
//! `graph_extent`). It is valid — every point of the body is within the reach of some point of
//! the spine — and it pads the box by that reach in every direction, including back along the
//! path where the body stops at its end cap. That padding is why a mirrored wing cannot fold:
//! `mirror_one_side` asks whether the body crosses the mirror plane, the body does not, and the
//! box does, by 0.0006 mm.
//!
//! A section is a disc in its own plane, not a ball, and a disc of radius `r` with unit normal
//! `n` has an axis-aligned half-extent of `r sqrt(1 - n_k^2)` in axis `k` — nothing at all along
//! its own normal. Building the box from the discs instead gave, on the cases below, between a
//! quarter and a third off the volume:
//!
//! | sweep | ball mm^3 | from discs | escaped |
//! |---|---|---|---|
//! | square, a gentle bend | 2.673e6 | 2.063e6 | 0 |
//! | square, a tight bend | 2.979e6 | 1.960e6 | **33** |
//! | square, a straight line | 1.022e6 | 7.517e5 | 0 |
//! | parallel, a tight bend | 2.979e6 | 1.734e6 | **3** |
//!
//! `escaped` counts points the field says are solid and the box says are not there, and one is a
//! bug: the box culls rays, sizes bakes and decides whether a mirror can fold. So the tightening
//! was backed out, and the reason it escaped is worth writing down because it is not the geometry.
//!
//! **The two sides tessellate the spine differently.** The field builds its `Spine::of_curve` at
//! the *compile* tolerance, whatever the caller asked for; `graph_extent` builds its own at a
//! fixed 0.1 mm. So the body is swept along one polyline and bounded along another, and the two
//! differ by up to the sum of their tolerances — 0.6 mm for a compile at 0.5, against the 0.77 mm
//! by which the body escaped. The ball survived that because 32 mm of padding hides it.
//!
//! Which makes the fix a signature rather than a formula: `graph_extent` has to be told the
//! tolerance the field will be built at, or the two have to share one polyline. Until then the
//! honest bound is the loose one, and this example is here to fail the next attempt that is not.
//!
//! **And the loose one was then attacked on purpose, to see whether it is safe by design or only
//! by luck.** The padding is the profile's reach, so a thin section has almost none of it; the
//! mismatch grows with how coarsely the field reads the spine, and tessellation refines to the
//! tolerance above a floor of six seeds a span. So: thin sections, coarse tolerances, a two-metre
//! spar with a real bend in it, a hairpin, and every held section. Thirteen cases, none escaped.
//!
//! That is a stronger result than it looks, and it is also why the disc bound failed. A coarser
//! polyline's chords lie *inside* the curve, so a body swept along them stays within the reach of
//! a finely-read curve — the padding is isotropic and covers the deviation in whichever direction
//! it happens to act. Take the padding away along the section's own normal, which is what a disc
//! does, and there is nothing left covering it there.
//!
//! **Sampled along the body, not over its box.** A grid over the bounding box cannot resolve a
//! thin sweep: a 4 mm wire in a 2.4 m box needs six hundred cells an axis before it finds the
//! body at all, and a grid that finds nothing reports that nothing escaped. Two earlier readings
//! here said exactly that, with 199 points inside, before the sampling was made to follow the
//! spine.
//!
//! Run with: `cargo run --release -p fieldform-field --example sweep_extent_probe`

use fieldform_field::ScalarField;
use fieldform_scene::{Curve, CurveId, Document, Graph, Node, ParamRef, Section};
use nalgebra::{Point3, Vector3};

fn main() {
    println!(
        "{:<40} {:>5} {:>9} {:>11} {:>10} {:>9}",
        "sweep", "spine", "inside", "escaped", "ball mm^3", "now mm^3"
    );
    let mut escapes = 0usize;
    for (label, doc, graph, tolerance) in cases() {
        let field = match fieldform_field::GraphField::compile_graph(&doc, &graph, tolerance) {
            Ok(f) => f,
            // A sweep the kernel refuses has no box to check; say so rather than stopping.
            Err(e) => {
                println!("{label:<46} refused: {e}");
                continue;
            }
        };
        let (lo, hi) = fieldform_field::part_extent(&doc, &graph)
            .as_box()
            .expect("a bounded sweep");

        // The box the old rule would have given, which is also the region to search: it is the
        // looser of the two, so anything the new box misses is inside it.
        let (blo, bhi) = ball_box(&doc, &graph);
        // Sized to the *body*, not to the search box. A fixed grid over a box 150 mm across
        // puts 2.5 mm between samples, which steps clean over a wire 3 mm thick and reports no
        // escapes because it found almost nothing at all. The cell has to be a fraction of the
        // profile's reach for "nothing escaped" to mean anything.
        let mut inside = 0usize;
        let mut escaped = 0usize;
        let _ = (blo, bhi);
        // **Sampled along the body, not over the box.** A grid over the bounding box cannot
        // resolve a thin sweep: a 4 mm wire in a 2.4 m box needs a 600-cell grid per axis before
        // it hits the body at all, and a coarse grid that finds nothing reports "nothing
        // escaped". So the sampling follows the spine and looks around it, where the body is.
        let reach = section_reach(&doc, &graph);
        let (spine_points, offset) = spine_of(&doc, &graph);
        let cell = reach * 0.25;
        let half = 8i32; // reaches 2x the section, so end caps and outside are both covered
        for c in &spine_points {
            let c = c + offset;
            for i in -half..=half {
                for j in -half..=half {
                    for k in -half..=half {
                        let p = c + Vector3::new(i as f64, j as f64, k as f64) * cell;
                        if field.distance(p) > 0.0 {
                            continue;
                        }
                        inside += 1;
                        let out = (0..3).any(|a| p[a] < lo[a] - 1e-6 || p[a] > hi[a] + 1e-6);
                        if out {
                            escaped += 1;
                            if escaped == 1 {
                                println!(
                                    "    ESCAPED at {p:?} reading {:.4}, box is {lo:?} .. {hi:?}",
                                    field.distance(p)
                                );
                            }
                        }
                    }
                }
            }
        }
        escapes += escaped;
        let vol = |a: Point3<f64>, b: Point3<f64>| {
            let d = b - a;
            d.x * d.y * d.z
        };
        println!(
            "{label:<40} {:>5} {inside:>9} {escaped:>11} {:>10.3e} {:>9.3e}",
            spine_points.len(),
            vol(blo, bhi),
            vol(lo, hi)
        );
    }
    println!(
        "\n`escaped` is the only column that can make this wrong: it counts points the field\n\
         says are solid and the box says are not there."
    );
    if escapes > 0 {
        std::process::exit(1);
    }
}

/// The box the old rule gave: a ball of the profile's reach at every spine sample.
fn ball_box(doc: &Document, graph: &Graph) -> (Point3<f64>, Point3<f64>) {
    let (id, node) = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Sweep { .. }))
        .expect("a sweep");
    let _ = id;
    let Node::Sweep { outline, spine, .. } = node else { unreachable!() };
    let curve = &doc.curves[spine];
    let offset =
        fieldform_field::sweep_offset(doc, outline, *spine).unwrap_or_else(Vector3::zeros);
    let world = fieldform_field::profile_loop_turning(doc, outline.curve(), 1.0, Some(*spine))
        .expect("a closed loop")
        .points;
    let (origin, _) = fieldform_field::plane_of(&world).expect("planar");
    let reach = world.iter().map(|p| (p - origin).norm()).fold(0.0f64, f64::max);
    let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in curve.tessellate(0.1) {
        let q = p + offset;
        for k in 0..3 {
            lo[k] = lo[k].min(q[k] - reach);
            hi[k] = hi[k].max(q[k] + reach);
        }
    }
    (lo, hi)
}

/// The spine as the extent reads it, finely, and where the sweep carries it to.
fn spine_of(doc: &Document, graph: &Graph) -> (Vec<Point3<f64>>, Vector3<f64>) {
    let (_, node) = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Sweep { .. }))
        .expect("a sweep");
    let Node::Sweep { outline, spine, .. } = node else { unreachable!() };
    let offset =
        fieldform_field::sweep_offset(doc, outline, *spine).unwrap_or_else(Vector3::zeros);
    (doc.curves[spine].tessellate(0.02), offset)
}

/// The profile's reach, which is the padding the ball bound has to hide the mismatch behind.
fn section_reach(doc: &Document, graph: &Graph) -> f64 {
    let (_, node) = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Sweep { .. }))
        .expect("a sweep");
    let Node::Sweep { outline, spine, .. } = node else { unreachable!() };
    let world = fieldform_field::profile_loop_turning(doc, outline.curve(), 1.0, Some(*spine))
        .expect("a closed loop")
        .points;
    let (origin, _) = fieldform_field::plane_of(&world).expect("planar");
    world.iter().map(|p| (p - origin).norm()).fold(0.0f64, f64::max).max(1e-6)
}

fn cases() -> Vec<(&'static str, Document, Graph, f64)> {
    vec![
        swept("square, a gentle bend", Section::Square, Bend::Gentle, 30.0, 0.5),
        swept("square, a tight bend", Section::Square, Bend::Tight, 30.0, 0.5),
        swept("square, a hairpin", Section::Square, Bend::Hairpin, 30.0, 0.5),
        swept("square, a straight line", Section::Square, Bend::Straight, 30.0, 0.5),
        swept("parallel, a tight bend", Section::Parallel, Bend::Tight, 30.0, 0.5),
        swept("flush start, a tight bend", Section::Skewed, Bend::Tight, 30.0, 0.5),
        // The padding is the profile's reach, so a *thin* sweep has almost none of it -- and the
        // two polylines still differ by the tolerances. This is where the ball stops hiding the
        // mismatch, if it ever does.
        swept("a thin wire, gentle bend", Section::Square, Bend::Gentle, 1.5, 0.5),
        swept("a thin wire, tight bend", Section::Square, Bend::Tight, 1.5, 0.5),
        swept("a thin wire, coarse tolerance", Section::Square, Bend::Tight, 1.5, 2.0),
        swept("a thin wire, very coarse", Section::Square, Bend::Tight, 1.5, 5.0),
        swept("an ordinary section, very coarse", Section::Square, Bend::Tight, 30.0, 5.0),
        // **The case the padding cannot cover.** Tessellation bottoms out at six seeds a span
        // whatever the tolerance, but above that floor it refines to the tolerance -- so a long,
        // genuinely curved spine read at 5 mm is 5 mm away from the same spine read at 0.1, and
        // a profile reaching 2 mm has only 2 mm of padding to hide that in.
        swept("a long curve, thin section, coarse", Section::Square, Bend::LongSweep, 2.0, 5.0),
        swept("the same, read finely", Section::Square, Bend::LongSweep, 2.0, 0.1),
        swept("the same, ordinary section", Section::Square, Bend::LongSweep, 30.0, 5.0),
    ]
}

#[derive(Clone, Copy)]
enum Bend {
    Straight,
    Gentle,
    Tight,
    Hairpin,
    /// Two metres with a real bend in it, which is a wing spar rather than a test shape.
    LongSweep,
}

fn swept(
    label: &'static str,
    section: Section,
    bend: Bend,
    half_width: f64,
    tolerance: f64,
) -> (&'static str, Document, Graph, f64) {
    let lit = ParamRef::Literal;
    let _ = lit;
    let mut doc = Document::new();
    let profile = CurveId::new();
    doc.curves.insert(
        profile,
        Curve::polyline(
            profile,
            "section",
            vec![
                Point3::new(-half_width, -half_width * 0.4, 0.0),
                Point3::new(half_width, -half_width * 0.4, 0.0),
                Point3::new(half_width, half_width * 0.4, 0.0),
                Point3::new(-half_width, half_width * 0.4, 0.0),
            ],
            true,
        )
        .expect("a rectangle"),
    );
    let spine = CurveId::new();
    let points = match bend {
        Bend::Straight => vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 60.0),
            Point3::new(0.0, 0.0, 120.0),
            Point3::new(0.0, 0.0, 180.0),
        ],
        Bend::Gentle => vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 60.0),
            Point3::new(30.0, 0.0, 120.0),
            Point3::new(70.0, 20.0, 170.0),
        ],
        Bend::Tight => vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 50.0),
            Point3::new(90.0, 0.0, 70.0),
            Point3::new(90.0, 60.0, 90.0),
        ],
        Bend::LongSweep => vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 700.0, 0.0),
            Point3::new(500.0, 1500.0, 200.0),
            Point3::new(1400.0, 2000.0, 600.0),
        ],
        // Back on itself, which is the case the "opposite signs" arm of the bound is for.
        Bend::Hairpin => vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 130.0),
            Point3::new(60.0, 0.0, 130.0),
            Point3::new(60.0, 0.0, -20.0),
        ],
    };
    doc.curves.insert(
        spine,
        Curve::new(
            spine,
            "spine",
            3,
            points,
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a cubic"),
    );
    let mut graph = Graph::empty();
    graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section, law: Default::default() });
    (label, doc, graph, tolerance)
}
