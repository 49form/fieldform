// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Two ways to give a curve a thickness, measured against the one contract both owe.
//!
//! `outline-wall-spec.md` §3 describes a wall in **tube coordinates**, the sweep's construction
//! one dimension down: find the nearest point of the curve, read a lateral coordinate and an
//! arc length there, treat the wall as a rectangle in those two, divide by the worst stretch
//! `1 / (1 − κ·r)` and refuse past `κ·r = 1`. `fieldform_field::Wall` is built from **fibres**
//! instead: every segment is asked whether the foot of the query's perpendicular is on it and
//! which side the query is on, and the nearest foot on each side is kept apart.
//!
//! **A table is only as good as the cases it crosses, and this one's first printing was not.**
//! Every corner in it was 90° or less and every bend wider than its wall, and on those rows the
//! fibres read nought too much — while being wrong, by up to a whole thickness, past a right
//! angle and wherever a run's end was left bare. The last three chains are those cases. What
//! holds the construction now is not this table but
//! `a_wall_is_the_union_of_its_fibres_whatever_the_chain`, which checks it against the wall
//! written out piece by piece on chains that wander as they like.
//!
//! That is a departure from the spec as first written, and the owner ruled for it on 2026-10-01
//! from this table rather than from the argument for it. Four questions, asked of both over the
//! same chains:
//!
//! - **Does it ever claim more room than there is?** `overstatement`, which marches past each
//!   claim. Anything above nought is a ray stepping through a wall.
//! - **Does it move at unit rate?** The share of probes outside the material where it does
//!   not, which is whether it may call itself a distance. The tube is the arm that shows the
//!   check can fail: wherever its divisor is above one, every probe should be off.
//! - **How short does it read?** The smallest ratio of the tube's reading to the fibres', where
//!   both are outside. Less than one is steps a tracer takes for nothing, and a fillet that
//!   comes out larger than the radius typed.
//! - **Do they describe the same solid?** How many probes the two put on opposite sides of
//!   the surface, and how deep in the fibres' wall the worst of them is.
//!
//! The tube here is the spec's, given the benefit of each doubt. The corner beyond an end is a
//! hypotenuse, the way `Swept` takes it. The curvature is read only on the side the wall lies.
//! And at a joint the lateral coordinate is the distance from the joint, signed by side —
//! rather than a projection onto the mean of the two normals, which is what a sweep's frame
//! does and what leaves the seam `sweep-spec.md` §9 records. Pass `seam` to see that reading.
//!
//! Run with:
//! `cargo run -p fieldform-field --example wall_construction`

use fieldform_field::{gradient_deviation, overstatement, ScalarField, Wall};
use fieldform_scene::{WallEnds, WallSide};
use nalgebra::{Point2, Point3, Vector2, Vector3};

/// The spec's wall: nearest point, lateral and arc coordinates, one divisor for the whole field.
struct Tube {
    points: Vec<Point2<f64>>,
    along: Vec<Vector2<f64>>,
    length: Vec<f64>,
    /// Arc length to each point.
    arc: Vec<f64>,
    half: f64,
    /// The middle of the band, as a lateral offset.
    middle: f64,
    stretch: f64,
    /// Whether a joint projects onto its mean normal, as a sweep's frame does.
    seam: bool,
}

/// The tube, or the product `κ·r` it was refused for.
fn tube(points: &[Point2<f64>], thickness: f64, side: WallSide, seam: bool) -> Result<Tube, f64> {
    let along: Vec<Vector2<f64>> =
        points.windows(2).map(|w| (w[1] - w[0]).normalize()).collect();
    let length: Vec<f64> = points.windows(2).map(|w| (w[1] - w[0]).norm()).collect();
    let mut arc = vec![0.0];
    for l in &length {
        arc.push(arc[arc.len() - 1] + l);
    }
    // The tightest bend on each side: a joint's turn over the mean of the two lengths meeting
    // there, which is how `max_curvature` reads a polyline. A left turn bends toward the left.
    let (mut left, mut right) = (0.0f64, 0.0f64);
    for i in 1..along.len() {
        let (a, b) = (along[i - 1], along[i]);
        let turn = (a.x * b.y - a.y * b.x).atan2(a.dot(&b));
        let curvature = turn.abs() / (0.5 * (length[i - 1] + length[i]));
        if turn > 0.0 {
            left = left.max(curvature);
        } else {
            right = right.max(curvature);
        }
    }
    let half = thickness * 0.5;
    let (middle, product) = match side {
        WallSide::Centre => (0.0, left.max(right) * half),
        WallSide::Left => (half, left * thickness),
        WallSide::Right => (-half, right * thickness),
    };
    if product >= 1.0 {
        return Err(product);
    }
    Ok(Tube {
        points: points.to_vec(),
        along,
        length,
        arc,
        half,
        middle,
        stretch: 1.0 / (1.0 - product),
        seam,
    })
}

impl Tube {
    fn distance(&self, q: Point2<f64>) -> f64 {
        // The nearest point, the earliest segment winning a tie as `Spine::nearest` has it.
        let (mut best, mut at, mut fraction, mut near) = (f64::INFINITY, 0usize, 0.0, q);
        for i in 0..self.along.len() {
            let f = ((q - self.points[i]).dot(&self.along[i]) / self.length[i]).clamp(0.0, 1.0);
            let c = self.points[i] + self.along[i] * (f * self.length[i]);
            let squared = (q - c).norm_squared();
            if squared < best {
                (best, at, fraction, near) = (squared, i, f, c);
            }
        }
        let last = self.along.len() - 1;
        let total = self.arc[self.arc.len() - 1];
        let rel = q - near;
        let side_of = |tangent: Vector2<f64>| tangent.x * rel.y - tangent.y * rel.x;
        // The tangent where it was met: the segment's own along its length, the mean of the two
        // at a joint, and the end segment's at an end — where both coordinates carry on
        // straight, which is what "extended linearly along the end tangent" asks for.
        let (lateral, arc) = if fraction <= 0.0 && at == 0 {
            (side_of(self.along[0]), rel.dot(&self.along[0]))
        } else if fraction >= 1.0 && at == last {
            (side_of(self.along[last]), total + rel.dot(&self.along[last]))
        } else if fraction <= 0.0 || fraction >= 1.0 {
            let joint = if fraction <= 0.0 { at } else { at + 1 };
            let projected = side_of((self.along[joint - 1] + self.along[joint]).normalize());
            let lateral = if self.seam { projected } else { projected.signum() * rel.norm() };
            (lateral, self.arc[joint])
        } else {
            (side_of(self.along[at]), self.arc[at] + fraction * self.length[at])
        };
        let across = (lateral - self.middle).abs() - self.half;
        let slab = (arc - total * 0.5).abs() - total * 0.5;
        let combined =
            if across > 0.0 && slab > 0.0 { across.hypot(slab) } else { across.max(slab) };
        combined / self.stretch
    }
}

/// A section read as a solid of endless height, so the kernel's own checks can ask it.
struct Upright<F: Fn(Point2<f64>) -> f64 + Send + Sync>(F);

impl<F: Fn(Point2<f64>) -> f64 + Send + Sync> ScalarField for Upright<F> {
    fn distance(&self, p: Point3<f64>) -> f64 {
        (self.0)(Point2::new(p.x, p.y))
    }
}

fn arc_about(centre: (f64, f64), radius: f64, from: f64, to: f64) -> Vec<Point2<f64>> {
    let steps = (to - from).abs().ceil() as usize;
    (0..=steps)
        .map(|i| {
            let a = (from + (to - from) * i as f64 / steps as f64).to_radians();
            Point2::new(centre.0 + radius * a.cos(), centre.1 + radius * a.sin())
        })
        .collect()
}

fn chains() -> Vec<(&'static str, Vec<Point2<f64>>)> {
    // Two arcs of 40 mm radius, each with a joint every degree, meeting at (40, 0) where the
    // second sets off 30 degrees to the left of where the first arrived.
    let mut cornered = arc_about((0.0, 0.0), 40.0, -60.0, 0.0);
    let second = (40.0 - 40.0 * (30.0f64).to_radians().cos(), -20.0);
    cornered.extend(arc_about(second, 40.0, 30.0, 90.0).into_iter().skip(1));
    let spiral = (0..=300)
        .map(|i| {
            let a = (i as f64 * 3.0).to_radians();
            let r = 20.0 + 12.0 * a / std::f64::consts::TAU;
            Point2::new(r * a.cos(), r * a.sin())
        })
        .collect();
    vec![
        ("a straight run, 100 mm", vec![Point2::new(0.0, 0.0), Point2::new(100.0, 0.0)]),
        ("an arc, R 40 over 120 deg", arc_about((0.0, 0.0), 40.0, 0.0, 120.0)),
        (
            "an L of two 50 mm runs",
            vec![Point2::new(0.0, 0.0), Point2::new(50.0, 0.0), Point2::new(50.0, 50.0)],
        ),
        ("two arcs at a 30 deg corner", cornered),
        ("a spiral, 12 mm a turn", spiral),
        // The three below are where the fibres' own first build was wrong, and no row above
        // crosses any of them: a corner past a right angle, a run too short to cover its
        // neighbour's end, and a bend tighter than the wall is thick.
        (
            "a V of two 50 mm runs, 135",
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(50.0, 0.0),
                Point2::new(50.0 - 50.0 * (0.5f64).sqrt(), 50.0 * (0.5f64).sqrt()),
            ],
        ),
        (
            "an L with a 3 mm return",
            vec![Point2::new(0.0, 0.0), Point2::new(50.0, 0.0), Point2::new(50.0, 3.0)],
        ),
        ("an arc, R 5 over 120 deg", arc_about((0.0, 0.0), 5.0, 0.0, 120.0)),
    ]
}

const ACROSS: usize = 34;

fn probes(points: &[Point2<f64>], margin: f64) -> Vec<Point2<f64>> {
    let (mut lo, mut hi) = (points[0], points[0]);
    for p in points {
        lo = Point2::new(lo.x.min(p.x), lo.y.min(p.y));
        hi = Point2::new(hi.x.max(p.x), hi.y.max(p.y));
    }
    let mut out = Vec::new();
    for j in 0..ACROSS {
        for i in 0..ACROSS {
            let (u, v) = ((i as f64 + 0.37) / ACROSS as f64, (j as f64 + 0.61) / ACROSS as f64);
            out.push(Point2::new(
                lo.x - margin + u * (hi.x - lo.x + 2.0 * margin),
                lo.y - margin + v * (hi.y - lo.y + 2.0 * margin),
            ));
        }
    }
    out
}

/// The share of probes outside a field's material where it is not moving at unit rate.
///
/// A share rather than a worst case, because the worst probe is always one that happens to sit
/// where two faces are equally near, and there no distance has a gradient at all.
fn off_unit_rate(field: &dyn ScalarField, probes: &[Point3<f64>]) -> f64 {
    let outside: Vec<Point3<f64>> =
        probes.iter().copied().filter(|p| field.distance(*p) > 0.5).collect();
    let off = outside.iter().filter(|p| gradient_deviation(field, &[**p], 1e-5) > 1e-4).count();
    100.0 * off as f64 / outside.len().max(1) as f64
}

fn main() {
    let seam = std::env::args().any(|a| a == "seam");
    let compass: Vec<Vector3<f64>> = (0..8)
        .map(|i| {
            let a = (i as f64 * 45.0 + 11.0).to_radians();
            Vector3::new(a.cos(), a.sin(), 0.0)
        })
        .collect();
    let (step, reach) = (0.05, 1.2);

    println!("Every wall has square ends. `over` is the worst claim of room that is not there, in mm.");
    println!("`off` is the share of outside probes not moving at unit rate. `reads` is the tube's");
    println!("shortest reading as a share of the fibres'. `differ` is how many of {} probes the two", ACROSS * ACROSS);
    println!("put on opposite sides of the surface, and `deepest` how far into the fibres' wall.");
    println!("A joint in the tube is read {}.\n", if seam { "by projection, as a sweep's frame reads it" } else { "by its distance, signed by side" });
    println!(
        "{:<28} {:<7} {:>2} | {:>7} {:>6} {:>6} {:>5} | {:>6} {:>6} | {:>6} {:>7}",
        "chain", "side", "t", "stretch", "over", "off", "reads", "over", "off", "differ", "deepest"
    );
    println!("{:<28} {:<7} {:>2} | {:^27} | {:^13} |", "", "", "", "tube (the spec)", "fibres");
    for (name, chain) in chains() {
        for thickness in [4.0, 8.0] {
            for side in [WallSide::Centre, WallSide::Left, WallSide::Right] {
                let fibres = Wall::open(chain.clone(), thickness, side, WallEnds::Square)
                    .expect("a chain with length and a thickness above nought");
                let grid = probes(&chain, 3.0 * thickness + 6.0);
                let lifted: Vec<Point3<f64>> =
                    grid.iter().map(|q| Point3::new(q.x, q.y, 0.0)).collect();

                let exact = Upright(|q| fibres.distance(q));
                let fibres_over = overstatement(&exact, &lifted, &compass, step, reach);
                let fibres_off = off_unit_rate(&exact, &lifted);

                let label = format!("{side:?}").to_lowercase();
                match tube(&chain, thickness, side, seam) {
                    Err(product) => println!(
                        "{name:<28} {label:<7} {thickness:>2.0} | {:>27} | {fibres_over:>6.3} {fibres_off:>5.1}% |",
                        format!("refused, kr = {product:.2}"),
                    ),
                    Ok(t) => {
                        let spec = Upright(|q| t.distance(q));
                        let tube_over = overstatement(&spec, &lifted, &compass, step, reach);
                        let tube_off = off_unit_rate(&spec, &lifted);
                        let (mut reads, mut differ, mut deepest) = (1.0f64, 0usize, 0.0f64);
                        for q in &grid {
                            let (a, b) = (t.distance(*q), fibres.distance(*q));
                            if a > 0.0 && b > 1.0 {
                                reads = reads.min(a / b);
                            }
                            if (a < 0.0) != (b < 0.0) && b.abs() > 0.05 {
                                differ += 1;
                                deepest = deepest.max(-b);
                            }
                        }
                        println!(
                            "{name:<28} {label:<7} {thickness:>2.0} | {:>7.3} {tube_over:>6.3} {tube_off:>5.1}% {reads:>5.2} | {fibres_over:>6.3} {fibres_off:>5.1}% | {differ:>6} {deepest:>7.2}",
                            t.stretch,
                        );
                    }
                }
            }
        }
    }
}
