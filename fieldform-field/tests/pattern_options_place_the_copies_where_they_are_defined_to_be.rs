// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A pattern's options are checked against a field the test builds by hand.
//!
//! **Spacing is where a fence-post error lives, and it is silent**: a row of five copies over a
//! run of 100 mm puts the last one 100 mm away only if the gap is `run/(n − 1)`, and a gap of
//! `run/n` looks like a row too. So each case here lists where the copies are *from the
//! definition* — written out in the test, not read back from `PatternLayout` — and asks the
//! compiled field, at a few thousand points, whether it equals the least of the distances to those
//! balls. A ball is an exact distance, so the least of them is the answer exactly, and any copy in
//! the wrong place, left in that should be out, or missing from the fold's window moves the field
//! at once.
//!
//! The counts are large and the balls small, so the window the fold walks is a few cells wide and
//! not the whole pattern: a case that fell back to walking every copy would pass without ever
//! having asked the fold which copy a point belongs to, which is the part an arc can get wrong.

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    Document, Fillet, Frame, Graph, Node, ParamRef, PatternKind, PatternOptions, Place, Spacing,
};
use nalgebra::{Point3, Vector3};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

const BALL: f64 = 2.0;

/// One ball of radius [`BALL`] at the origin, patterned, and the compiled field.
fn field_of(kind: PatternKind, count: u32, join: bool, options: PatternOptions) -> GraphField {
    let mut graph = Graph::empty();
    let body = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(BALL) });
    graph.add(Node::Pattern {
        input: body.into(),
        count: lit(f64::from(count)),
        kind,
        join,
        fillet: Fillet::hard(),
        options,
    });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("the case compiles")
}

fn row(step: [f64; 3]) -> PatternKind {
    PatternKind::Linear { step: step.map(lit) }
}

/// A ring about +Z through the origin, with the ball moved out to radius 100 so that it has
/// something to turn.
fn ring_field(count: u32, join: bool, options: PatternOptions) -> GraphField {
    let mut graph = Graph::empty();
    let ball = graph.add(Node::Sphere {
        centre: Place::At([lit(100.0), lit(0.0), lit(0.0)]),
        radius: lit(BALL),
    });
    graph.add(Node::Pattern {
        input: ball.into(),
        count: lit(f64::from(count)),
        kind: PatternKind::Radial { frame: Frame::default() },
        join,
        fillet: Fillet::hard(),
        options,
    });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("the case compiles")
}

/// Where the ball of copy `i` of a ring stands, `degrees` round from where the body is.
fn on_ring(degrees: f64) -> Point3<f64> {
    let a = degrees.to_radians();
    Point3::new(100.0 * a.cos(), 100.0 * a.sin(), 0.0)
}

/// The most the field and the least of the distances to `centres` differ, over a scatter.
///
/// A deterministic spread through the box that holds the copies and some way beyond it, dense
/// enough that every copy has points close to its surface and every gap between neighbours has
/// some in it.
fn disagreement(field: &GraphField, centres: &[Point3<f64>], lo: Vector3<f64>, hi: Vector3<f64>) -> f64 {
    let n = 40usize;
    let mut worst: f64 = 0.0;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n * n * n / 8 {
        let p = Point3::new(
            lo.x + (hi.x - lo.x) * next(),
            lo.y + (hi.y - lo.y) * next(),
            lo.z + (hi.z - lo.z) * next(),
        );
        let truth = centres.iter().map(|c| (p - c).norm() - BALL).fold(f64::INFINITY, f64::min);
        worst = worst.max((field.distance(p) - truth).abs());
    }
    // And on every ball's own surface, which is where a missing or misplaced copy shows first.
    // Compared with the least distance rather than with nought: a few cases pitch their copies
    // closer than a diameter, so a point on one ball's surface can be inside its neighbour.
    for c in centres {
        for k in 0..16 {
            let a = f64::from(k) * 0.39;
            let p = Point3::new(c.x + BALL * a.cos(), c.y + BALL * a.sin(), c.z);
            let truth = centres.iter().map(|c| (p - c).norm() - BALL).fold(f64::INFINITY, f64::min);
            worst = worst.max((field.distance(p) - truth).abs());
        }
    }
    worst
}

/// The `n` copies of a row from the definition: copy `i` is `(i + bias)` gaps from the body.
fn row_centres(count: u32, gap: [f64; 3], bias: f64, left_out: &[u32]) -> Vec<Point3<f64>> {
    (0..count)
        .filter(|i| !left_out.contains(i))
        .map(|i| {
            let k = f64::from(i) + bias;
            Point3::new(gap[0] * k, gap[1] * k, gap[2] * k)
        })
        .collect()
}

/// A row whose step is its whole run: 40 copies over 390 mm are 10 mm apart and the last is 390 mm
/// out — not 9.75 mm apart, which is what dividing by the count would give.
#[test]
fn a_row_spaced_by_its_whole_run_ends_one_run_away() {
    let options = PatternOptions { spacing: Spacing::Extent, ..PatternOptions::default() };
    let field = field_of(row([390.0, 0.0, 0.0]), 40, true, options);
    let centres = row_centres(40, [10.0, 0.0, 0.0], 0.0, &[]);
    let err = disagreement(&field, &centres, Vector3::new(-30.0, -15.0, -15.0), Vector3::new(420.0, 15.0, 15.0));
    assert!(err < 1e-9, "the field differs from forty balls ten apart by {err} mm");
    assert!(field.distance(Point3::new(390.0, 0.0, 0.0)) < -BALL + 1e-9, "the last copy is a whole run away");
}

/// The same row spaced by a gap puts the last copy `n − 1` gaps away.
#[test]
fn a_row_spaced_by_a_gap_ends_n_minus_one_gaps_away() {
    let field = field_of(row([10.0, 0.0, 0.0]), 40, true, PatternOptions::default());
    let centres = row_centres(40, [10.0, 0.0, 0.0], 0.0, &[]);
    let err = disagreement(&field, &centres, Vector3::new(-30.0, -15.0, -15.0), Vector3::new(420.0, 15.0, 15.0));
    assert!(err < 1e-9, "{err}");
    assert!(field.distance(Point3::new(390.0, 0.0, 0.0)) < -BALL + 1e-9);
    assert!(field.distance(Point3::new(400.0, 0.0, 0.0)) > 0.0, "and nothing stands past it");
}

/// A centred row of 41 has the body in the middle and 20 copies either side; one of 40 has 19
/// below it and 20 above, because the count is the number of bodies and a whole step is the only
/// bias that leaves one of them on the body.
#[test]
fn a_centred_row_runs_evenly_either_side_of_the_body() {
    let options = PatternOptions { symmetric: true, ..PatternOptions::default() };
    for count in [41u32, 40] {
        let field = field_of(row([10.0, 0.0, 0.0]), count, true, options.clone());
        let bias = -f64::from((count - 1) / 2);
        let centres = row_centres(count, [10.0, 0.0, 0.0], bias, &[]);
        let err = disagreement(&field, &centres, Vector3::new(-230.0, -15.0, -15.0), Vector3::new(230.0, 15.0, 15.0));
        assert!(err < 1e-9, "{count} copies centred: {err}");
    }
}

/// An oblique row, so the step is not along an axis and the dot product that finds a point's copy
/// is a real one.
#[test]
fn a_centred_oblique_row_is_still_where_it_is_defined_to_be() {
    let options = PatternOptions { symmetric: true, spacing: Spacing::Extent, ..PatternOptions::default() };
    let field = field_of(row([240.0, 160.0, 80.0]), 33, true, options);
    let gap = [240.0 / 32.0, 160.0 / 32.0, 80.0 / 32.0];
    let centres = row_centres(33, gap, -16.0, &[]);
    let err = disagreement(&field, &centres, Vector3::new(-300.0, -200.0, -120.0), Vector3::new(300.0, 200.0, 120.0));
    assert!(err < 1e-9, "{err}");
}

/// Skip leaves out exactly the copies named, and the others stay where they were.
#[test]
fn skip_removes_exactly_the_copies_named() {
    let options = PatternOptions { skip: vec![3, 4, 17, 99], ..PatternOptions::default() };
    let field = field_of(row([8.0, 0.0, 0.0]), 30, true, options);
    let centres = row_centres(30, [8.0, 0.0, 0.0], 0.0, &[3, 4, 17]);
    let err = disagreement(&field, &centres, Vector3::new(-20.0, -12.0, -12.0), Vector3::new(260.0, 12.0, 12.0));
    assert!(err < 1e-9, "{err}");
    assert!(field.distance(Point3::new(24.0, 0.0, 0.0)) > 0.0, "copy 3 is not there");
    assert!(field.distance(Point3::new(16.0, 0.0, 0.0)) < 0.0, "copy 2 still is");
}

/// A block that does not hold the body leaves out the copy standing on it: the first for a row that
/// starts at it, and `⌊(n − 1)/2⌋` for a centred one, odd or even.
#[test]
fn a_block_not_holding_the_body_leaves_out_the_copy_on_it() {
    let at = |x: f64| Point3::new(x, 0.0, 0.0);
    // Starts at the body: copy 0 is out.
    let field = field_of(row([10.0, 0.0, 0.0]), 30, false, PatternOptions::default());
    assert!(field.distance(at(0.0)) > 0.0, "the body's own copy is not the block's");
    assert!(field.distance(at(10.0)) < 0.0);
    // Centred and odd: the middle copy is the one on the body.
    let centred = PatternOptions { symmetric: true, ..PatternOptions::default() };
    let field = field_of(row([10.0, 0.0, 0.0]), 31, false, centred.clone());
    assert!(field.distance(at(0.0)) > 0.0, "the middle copy stands on the body and is left out");
    assert!(field.distance(at(10.0)) < 0.0 && field.distance(at(-10.0)) < 0.0);
    // Centred and even: copy 14 stands on the body and is left out, and the two beside it are
    // a whole step away rather than half a step.
    let field = field_of(row([10.0, 0.0, 0.0]), 30, false, centred);
    assert!(field.distance(at(0.0)) > 0.0, "the copy on the body is left out at an even count too");
    assert!(field.distance(at(10.0)) < 0.0 && field.distance(at(-10.0)) < 0.0);
    assert!(field.distance(at(5.0)) > 0.0, "and nothing stands half a step off");
}

/// An arc: 40 copies over 270 degrees are `270/39` apart, the last at the far end of the angle.
/// The points behind the arc — the gap — are where the fold has to pick the nearer end, so the
/// scatter is the whole disc and not only the arc.
#[test]
fn an_arc_ends_at_the_far_end_of_its_angle_and_the_gap_belongs_to_the_nearer_end() {
    let options = PatternOptions { angle: lit(270.0), ..PatternOptions::default() };
    let field = ring_field(40, true, options);
    let centres: Vec<_> = (0..40).map(|i| on_ring(f64::from(i) * 270.0 / 39.0)).collect();
    let err = disagreement(&field, &centres, Vector3::new(-130.0, -130.0, -8.0), Vector3::new(130.0, 130.0, 8.0));
    assert!(err < 1e-9, "the arc's field differs from its balls by {err} mm");
    assert!(field.distance(on_ring(270.0)) < -BALL + 1e-9, "a copy stands at the far end");
}

/// A centred arc runs either side of the body, biased by a whole number of its own steps: 40 over
/// 270 degrees step by 270/39, and the bias is 19 of them, so one copy sits on the body.
#[test]
fn a_centred_arc_is_spread_about_the_body() {
    let options =
        PatternOptions { angle: lit(270.0), symmetric: true, ..PatternOptions::default() };
    let field = ring_field(40, true, options);
    let step = 270.0 / 39.0;
    let centres: Vec<_> =
        (0..40).map(|i| on_ring((f64::from(i) - 19.0) * step)).collect();
    let err = disagreement(&field, &centres, Vector3::new(-130.0, -130.0, -8.0), Vector3::new(130.0, 130.0, 8.0));
    assert!(err < 1e-9, "{err}");
}

/// A full turn with an angle of 360, or more, is the ring it always was.
#[test]
fn a_full_turn_is_the_ring_it_always_was() {
    for angle in [360.0, 720.0] {
        let options = PatternOptions { angle: lit(angle), ..PatternOptions::default() };
        let field = ring_field(30, true, options);
        let centres: Vec<_> = (0..30).map(|i| on_ring(f64::from(i) * 12.0)).collect();
        let err = disagreement(&field, &centres, Vector3::new(-130.0, -130.0, -8.0), Vector3::new(130.0, 130.0, 8.0));
        assert!(err < 1e-9, "angle {angle}: {err}");
    }
}

/// A centred full turn is the ring it always was, at every count: a whole-step bias about a closed
/// turn only renumbers the copies, and the ring is the same ring. It used to be half a step off at
/// an even count, which drew a ring nobody asked for from ticking "centred" on a full turn.
#[test]
fn a_centred_full_turn_is_the_ring_it_always_was() {
    let options = PatternOptions { symmetric: true, ..PatternOptions::default() };
    let field = ring_field(30, true, options);
    let centres: Vec<_> = (0..30).map(|i| on_ring(f64::from(i) * 12.0)).collect();
    let err = disagreement(&field, &centres, Vector3::new(-130.0, -130.0, -8.0), Vector3::new(130.0, 130.0, 8.0));
    assert!(err < 1e-9, "{err}");
}

/// Skip on a ring, with a partial angle, together: the options compose.
#[test]
fn skip_and_an_arc_compose() {
    let options =
        PatternOptions { angle: lit(180.0), skip: vec![5, 6, 20], ..PatternOptions::default() };
    let field = ring_field(30, true, options);
    let centres: Vec<_> = (0..30)
        .filter(|i| ![5, 6, 20].contains(i))
        .map(|i| on_ring(f64::from(i) * 180.0 / 29.0))
        .collect();
    let err = disagreement(&field, &centres, Vector3::new(-130.0, -130.0, -8.0), Vector3::new(130.0, 130.0, 8.0));
    assert!(err < 1e-9, "{err}");
}

/// A grid of balls: `columns` along `step`, `rows` along `step2`, from the definition.
///
/// The steps are oblique to each other and to the axes, and the second leaves the first's plane, so
/// the two coordinates a point's copy is read from are not the plain components of anything — a fold
/// that read them off the wrong vectors would be right only for a square grid.
fn grid_field(columns: u32, rows: u32, options: PatternOptions, step: [f64; 3]) -> GraphField {
    let mut graph = Graph::empty();
    let body = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(BALL) });
    graph.add(Node::Pattern {
        input: body.into(),
        count: lit(f64::from(columns)),
        kind: row(step),
        join: true,
        fillet: Fillet::hard(),
        options: PatternOptions {
            second: Some(fieldform_scene::PatternSecond {
                step: [lit(3.0), lit(9.0), lit(1.5)],
                count: lit(f64::from(rows)),
            }),
            ..options
        },
    });
    GraphField::compile_graph(&Document::new(), &graph, 1e-4).expect("the grid compiles")
}

/// Column `i mod columns` of row `i div columns`, each `(index + bias)` steps from the body.
fn grid_centres(
    columns: u32,
    rows: u32,
    steps: ([f64; 3], [f64; 3]),
    biases: (f64, f64),
    left_out: &[u32],
) -> Vec<Point3<f64>> {
    let (e1, e2) = (Vector3::from(steps.0), Vector3::from(steps.1));
    (0..columns * rows)
        .filter(|i| !left_out.contains(i))
        .map(|i| {
            let (a, b) = (f64::from(i % columns) + biases.0, f64::from(i / columns) + biases.1);
            Point3::from(e1 * a + e2 * b)
        })
        .collect()
}

const E2: [f64; 3] = [3.0, 9.0, 1.5];

/// A 14 × 11 grid of small balls: the window is a few cells along each step, not the whole grid, so
/// the fold has to say which column and which row a point belongs to — at two oblique steps.
#[test]
fn a_grid_is_where_it_is_defined_to_be() {
    let e1 = [10.0, 2.0, 0.0];
    let field = grid_field(14, 11, PatternOptions::default(), e1);
    let centres = grid_centres(14, 11, (e1, E2), (0.0, 0.0), &[]);
    let err = disagreement(&field, &centres, Vector3::new(-30.0, -30.0, -20.0), Vector3::new(180.0, 140.0, 30.0));
    assert!(err < 1e-9, "the grid's field differs from its 154 balls by {err} mm");
    assert_eq!(centres.len(), 154);
}

/// Centred in both directions and spaced by whole runs: the run of each is the vector over one less
/// than its count.
#[test]
fn a_centred_grid_spaced_by_whole_runs_is_where_it_is_defined_to_be() {
    let (columns, rows) = (13u32, 10u32);
    let options =
        PatternOptions { symmetric: true, spacing: Spacing::Extent, ..PatternOptions::default() };
    let run = [120.0, 24.0, 0.0];
    let field = grid_field(columns, rows, options, run);
    let gap1 = [run[0] / 12.0, run[1] / 12.0, run[2] / 12.0];
    let gap2 = [E2[0] / 9.0, E2[1] / 9.0, E2[2] / 9.0];
    // Each direction biased by a whole number of its own gaps: 6 of 13 columns, 4 of 10 rows.
    let centres = grid_centres(columns, rows, (gap1, gap2), (-6.0, -4.0), &[]);
    let err = disagreement(&field, &centres, Vector3::new(-90.0, -60.0, -20.0), Vector3::new(90.0, 60.0, 20.0));
    assert!(err < 1e-9, "{err}");
}

/// One short direction and one long: the short way is walked whole and the long way by window, each
/// by its own reach, rather than the whole grid being walked because one axis is short.
#[test]
fn a_long_thin_grid_windows_one_direction_and_walks_the_other_whole() {
    let e1 = [10.0, 2.0, 0.0];
    let field = grid_field(3, 40, PatternOptions::default(), e1);
    let centres = grid_centres(3, 40, (e1, E2), (0.0, 0.0), &[]);
    let err = disagreement(&field, &centres, Vector3::new(-30.0, -30.0, -20.0), Vector3::new(60.0, 400.0, 80.0));
    assert!(err < 1e-9, "{err}");
}

/// Copies of a grid left out: numbered along the first direction, row after row.
#[test]
fn a_grid_leaves_out_exactly_the_copies_named() {
    let e1 = [10.0, 2.0, 0.0];
    let skip = vec![5, 14, 15, 100, 153];
    let options = PatternOptions { skip: skip.clone(), ..PatternOptions::default() };
    let field = grid_field(14, 11, options, e1);
    let centres = grid_centres(14, 11, (e1, E2), (0.0, 0.0), &skip);
    let err = disagreement(&field, &centres, Vector3::new(-30.0, -30.0, -20.0), Vector3::new(180.0, 140.0, 30.0));
    assert!(err < 1e-9, "{err}");
    assert_eq!(centres.len(), 154 - 5);
}

/// The options a pattern cannot honour are refused at compile, by name, and never quietly dropped.
#[test]
fn what_cannot_be_honoured_is_refused_by_name() {
    let mut graph = Graph::empty();
    let body = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(2.0) });
    let add = |graph: &mut Graph, count: u32, kind: PatternKind, options: PatternOptions| {
        graph.add(Node::Pattern {
            input: body.into(),
            count: lit(f64::from(count)),
            kind,
            join: true,
            fillet: Fillet::hard(),
            options,
        });
    };
    let compile = |graph: &Graph| GraphField::compile_graph(&Document::new(), graph, 1e-4).err();

    let mut g = graph.clone();
    add(&mut g, 1, row([10.0, 0.0, 0.0]), PatternOptions { spacing: Spacing::Extent, ..Default::default() });
    let e = compile(&g).expect("a whole run of one copy is refused").to_string();
    assert!(e.contains("at least two copies"), "{e}");

    let mut g = graph.clone();
    add(&mut g, 1, PatternKind::Radial { frame: Frame::default() }, PatternOptions { angle: lit(90.0), ..Default::default() });
    let e = compile(&g).expect("an arc of one copy is refused").to_string();
    assert!(e.contains("at least two copies"), "{e}");

    let mut g = graph.clone();
    add(&mut g, 4, row([10.0, 0.0, 0.0]), PatternOptions { skip: vec![0], ..Default::default() });
    let e = compile(&g).expect("skipping the body is refused").to_string();
    assert!(e.contains("cannot be skipped"), "{e}");

    let mut g = graph.clone();
    add(&mut g, 3, PatternKind::Radial { frame: Frame::default() }, PatternOptions { angle: lit(-10.0), ..Default::default() });
    let e = compile(&g).expect("a negative angle is refused").to_string();
    assert!(e.contains("positive number of degrees"), "{e}");
}
