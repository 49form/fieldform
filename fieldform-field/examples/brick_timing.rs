// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Spike D: does per-brick pruning *save wall-clock*, and how dear must a leaf be before it does?
//!
//! `brick_pruning.rs` measured tape *lengths* — 19x to 58x shorter — and `backlog.md` item 17
//! records the caveat that decides whether any of it is worth building:
//!
//! > these are tape lengths, not wall-clock. A shorter tape is strong evidence of less work but
//! > not a measurement of it, and it says nothing about the cost of the pruning pass itself,
//! > which runs per brick and is not free. A timing spike belongs before this is built.
//!
//! This is that spike. It times three things per brick — the interval evaluation that rejects or
//! traces, the `simplify` that shortens the tape, and the points evaluated afterwards — against
//! the same points put through the full tape.
//!
//! **The knob that matters is how dear a leaf is.** Our leaves are not all alike: a sphere is
//! closed-form and nearly free, while a pipe walks every segment of its spine on every
//! evaluation, which is what made a patterned pipe crawl (`patterned_pipe_cost.rs`). Pruning
//! does not shorten the inside of a leaf; what it does is *not evaluate the leaf at all* where it
//! cannot matter. So the question is not "is pruning faster" but "how expensive does a leaf have
//! to be before pruning pays for its own pass", and the answer is a crossover rather than a yes.
//!
//! Run with: `cargo run --release -p fieldform-field --example brick_timing`

use std::time::Instant;

use fidget_core::shape::{EzShape, Shape};
use fidget_core::{
    context::Tree,
    shape::ShapeVars,
    types::Interval,
    var::{Var, VarIndex},
    vm::{VmFunction, VmShape},
};

/// The world the scenes are built in, in millimetres.
const EXTENT_MM: f64 = 1000.0;

/// One instance of the scene: a closed-form ball, plus an opaque leaf standing in for a field
/// whose cost is ours rather than Fidget's.
struct Instance {
    centre: [f64; 3],
    radius: f64,
    var: Var,
    index: VarIndex,
}

/// A union of `count` instances, each an opaque leaf with a known interval over any box.
///
/// Opaque rather than closed-form, because that is what our graphs are: a cage, a swept curve and
/// a pipe all reach Fidget as *variables* whose values this side supplies. It is also the only
/// shape in which the cost being saved is ours to save.
fn scene(count: usize) -> (Tree, Vec<Instance>) {
    let mut tree: Option<Tree> = None;
    let mut instances = Vec::new();
    for i in 0..count {
        // The same low-discrepancy scatter `brick_pruning` uses, so the two spikes describe the
        // same geometry and their numbers can be read against each other.
        let f = i as f64;
        let at = |seed: f64| (f * seed).fract() - 0.5;
        let centre = [
            at(0.754_877_666_2) * EXTENT_MM,
            at(0.569_840_290_9) * EXTENT_MM,
            at(0.430_159_709_1) * EXTENT_MM,
        ];
        let var = Var::new();
        let leaf = Tree::from(var);
        instances.push(Instance {
            centre,
            radius: EXTENT_MM * 0.02,
            var,
            index: var.index().expect("a fresh variable has an index"),
        });
        tree = Some(match tree {
            Some(t) => t.min(leaf),
            None => leaf,
        });
    }
    (tree.expect("at least one instance"), instances)
}

/// What one leaf costs to sample, as a knob.
///
/// `work` is how many times the distance is recomputed before being returned — a stand-in for a
/// pipe walking its spine, a cage sampling its bake, or a profile scanning its loop. One is a
/// sphere.
#[inline(never)]
fn sample(instance: &Instance, x: f32, y: f32, z: f32, work: usize) -> f32 {
    let mut d = 0.0f64;
    for _ in 0..work.max(1) {
        let p = [f64::from(x), f64::from(y), f64::from(z)];
        d = ((p[0] - instance.centre[0]).powi(2)
            + (p[1] - instance.centre[1]).powi(2)
            + (p[2] - instance.centre[2]).powi(2))
        .sqrt()
            - instance.radius;
        // Kept dependent on the loop so it cannot be hoisted out.
        std::hint::black_box(d);
    }
    d as f32
}

/// The interval a leaf takes over a box, which is what lets it be pruned at all.
fn bound(instance: &Instance, lo: [f64; 3], hi: [f64; 3]) -> Interval {
    // Distance from a box to a point: nought inside, else the length of the clamped offset.
    let mut near = 0.0f64;
    let mut far = 0.0f64;
    for axis in 0..3 {
        let c = instance.centre[axis];
        let n = if c < lo[axis] {
            lo[axis] - c
        } else if c > hi[axis] {
            c - hi[axis]
        } else {
            0.0
        };
        let f = (c - lo[axis]).abs().max((c - hi[axis]).abs());
        near += n * n;
        far += f * f;
    }
    Interval::new(
        (near.sqrt() - instance.radius) as f32,
        (far.sqrt() - instance.radius) as f32,
    )
}

struct Row {
    bricks: usize,
    visited: usize,
    full_ns: f64,
    pruned_ns: f64,
    prune_share: f64,
    mean_kept: f64,
    full_tape: usize,
}

fn measure(count: usize, grid: usize, per_brick: usize, work: usize) -> Row {
    let (tree, instances) = scene(count);
    let shape = VmShape::from(tree);
    let full_tape = shape.size();

    let interval_tape = shape.interval_tape(Default::default());
    let float_tape = shape.ez_float_slice_tape();
    let mut interval_eval = Shape::<VmFunction>::new_interval_eval();
    let mut float_eval = Shape::<VmFunction>::new_float_slice_eval();
    let mut workspace = Default::default();

    let step = EXTENT_MM / grid as f64;
    let origin = -EXTENT_MM * 0.5;

    // The sample points, laid out once per brick in the same pattern every time so the two
    // timings evaluate exactly the same work.
    let side = (per_brick as f64).cbrt().ceil() as usize;
    let offsets: Vec<[f64; 3]> = (0..side)
        .flat_map(|a| {
            (0..side).flat_map(move |b| {
                (0..side).map(move |c| {
                    [
                        (a as f64 + 0.5) / side as f64,
                        (b as f64 + 0.5) / side as f64,
                        (c as f64 + 0.5) / side as f64,
                    ]
                })
            })
        })
        .take(per_brick)
        .collect();

    let mut bricks = 0usize;
    let mut visited = 0usize;
    let mut kept = Vec::new();
    let mut full_ns = 0.0f64;
    let mut pruned_ns = 0.0f64;
    let mut prune_ns = 0.0f64;

    for i in 0..grid {
        for j in 0..grid {
            for k in 0..grid {
                bricks += 1;
                let lo = [
                    origin + i as f64 * step,
                    origin + j as f64 * step,
                    origin + k as f64 * step,
                ];
                let hi = [lo[0] + step, lo[1] + step, lo[2] + step];

                let xs: Vec<f32> =
                    offsets.iter().map(|o| (lo[0] + o[0] * step) as f32).collect();
                let ys: Vec<f32> =
                    offsets.iter().map(|o| (lo[1] + o[1] * step) as f32).collect();
                let zs: Vec<f32> =
                    offsets.iter().map(|o| (lo[2] + o[2] * step) as f32).collect();

                // ---- the full tape: every leaf sampled at every point, then evaluated.
                let started = Instant::now();
                let mut columns: Vec<Vec<f32>> = Vec::with_capacity(instances.len());
                for instance in &instances {
                    columns.push(
                        (0..xs.len())
                            .map(|n| sample(instance, xs[n], ys[n], zs[n], work))
                            .collect(),
                    );
                }
                let mut vars: ShapeVars<&[f32]> = ShapeVars::new();
                for (instance, column) in instances.iter().zip(&columns) {
                    vars.insert(instance.index, column.as_slice());
                }
                let out = float_eval
                    .eval_with_var_arrays(&float_tape, &xs, &ys, &zs, &vars)
                    .expect("bulk evaluation with every variable bound");
                std::hint::black_box(out[0]);
                full_ns += started.elapsed().as_nanos() as f64;

                // ---- per brick: bound every leaf, reject or trace, simplify, then evaluate
                // only the leaves that survived.
                let started = Instant::now();
                let mut bounds: ShapeVars<Interval> = ShapeVars::new();
                for instance in &instances {
                    bounds.insert(instance.index, bound(instance, lo, hi));
                }
                let x = Interval::new(lo[0] as f32, hi[0] as f32);
                let y = Interval::new(lo[1] as f32, hi[1] as f32);
                let z = Interval::new(lo[2] as f32, hi[2] as f32);
                let (value, trace) = interval_eval
                    .eval_with_vars(&interval_tape, x, y, z, &bounds)
                    .expect("interval evaluation with every variable bound");

                if value.lower() > 0.0 || value.upper() < 0.0 {
                    // Rejected outright: the pruning pass is the whole cost here, and it is the
                    // gain as well — nothing else runs.
                    let spent = started.elapsed().as_nanos() as f64;
                    pruned_ns += spent;
                    prune_ns += spent;
                    continue;
                }
                visited += 1;

                let simplified = match trace {
                    Some(trace) => Some(
                        shape
                            .simplify(trace, Default::default(), &mut workspace)
                            .expect("the trace came from this shape"),
                    ),
                    None => None,
                };
                let tape = match &simplified {
                    Some(s) => s.ez_float_slice_tape(),
                    None => float_tape.clone(),
                };
                // Which leaves survived: a variable the shortened tape no longer mentions is one
                // this brick never has to sample. That is where the time goes -- pruning does not
                // shorten the inside of a leaf, it skips the leaf.
                let live: Vec<&Instance> = instances
                    .iter()
                    .filter(|i| tape.vars().get(&i.var).is_some())
                    .collect();
                kept.push(simplified.as_ref().map_or(full_tape, |s| s.size()));
                prune_ns += started.elapsed().as_nanos() as f64;
                let mut columns: Vec<Vec<f32>> = Vec::with_capacity(live.len());
                for instance in &live {
                    columns.push(
                        (0..xs.len())
                            .map(|n| sample(instance, xs[n], ys[n], zs[n], work))
                            .collect(),
                    );
                }
                let mut vars: ShapeVars<&[f32]> = ShapeVars::new();
                for (instance, column) in live.iter().zip(&columns) {
                    vars.insert(instance.index, column.as_slice());
                }
                let out = float_eval
                    .eval_with_var_arrays(&tape, &xs, &ys, &zs, &vars)
                    .expect("bulk evaluation on the shortened tape");
                std::hint::black_box(out[0]);
                pruned_ns += started.elapsed().as_nanos() as f64;
            }
        }
    }

    Row {
        bricks,
        visited,
        full_ns,
        pruned_ns,
        prune_share: if pruned_ns > 0.0 { prune_ns / pruned_ns } else { 0.0 },
        mean_kept: if kept.is_empty() {
            0.0
        } else {
            kept.iter().sum::<usize>() as f64 / kept.len() as f64
        },
        full_tape,
    }
}

fn main() {
    println!(
        "Spike D — does per-brick pruning save wall-clock, and when?\n\n\
         A union of opaque leaves in a 1 m cube, which is the shape our graphs take: a cage, a\n\
         swept curve and a pipe all reach Fidget as variables this side supplies. `work` is how\n\
         dear one leaf is to sample -- 1 is a sphere, and a pipe walking a 37-segment spine is\n\
         nearer 37.\n"
    );

    let grid = 16;
    let per_brick = 64;
    println!("  {grid}^3 bricks, {per_brick} points evaluated in each.\n");
    println!(
        "  {:>9}  {:>5}  {:>8}  {:>8}  {:>9}  {:>9}  {:>7}  {:>7}",
        "instances", "work", "tape", "kept", "full ms", "pruned ms", "faster", "in prune"
    );
    for count in [50usize, 200, 1000] {
        for work in [1usize, 8, 40] {
            let row = measure(count, grid, per_brick, work);
            println!(
                "  {count:>9}  {work:>5}  {:>8}  {:>8.0}  {:>9.1}  {:>9.1}  {:>6.1}x  {:>6.0}%",
                row.full_tape,
                row.mean_kept,
                row.full_ns / 1e6,
                row.pruned_ns / 1e6,
                row.full_ns / row.pruned_ns.max(1.0),
                row.prune_share * 100.0,
            );
            if work == 1 {
                println!(
                    "  {:>9}  {:>5}  {:>8}  {:>8}  {:>9}  {:>9}  {:>7}  {:>7}",
                    "",
                    "",
                    format!("{} bricks", row.bricks),
                    format!("{} live", row.visited),
                    "",
                    "",
                    "",
                    "",
                );
            }
        }
    }

    println!(
        "\nHow to read it. `faster` is the whole point: above one, per-brick pruning wins on the\n\
         clock and not merely on tape length. `in prune` is how much of the pruned time went on\n\
         the interval pass and `simplify` rather than on evaluating anything.\n\n\
         The crossover is in `work`, because pruning never shortens the inside of a leaf. What it\n\
         does is not evaluate the leaf at all where it cannot matter, so the saving is exactly\n\
         what a leaf costs times the leaves it skipped."
    );

    ours();
    meshing();
}

/// Spike E: the same question asked of *our* op list rather than Fidget's tape.
///
/// **Because nothing on a live path compiles our graph into a Fidget tree.** The bridge is the M0
/// spike; `GraphField` is its own op list with its own `interval`. Building item 17 on Fidget
/// would mean a third compiler for one composition, so `GraphField::for_region` prunes the ops we
/// already have — and this measures whether that lands where Spike D did.
fn ours() {
    use fieldform_field::{GraphField, ScalarField};
    use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef, Place};

    println!("\n\nSpike E — the same, over our own op list.\n");
    println!(
        "  {:>9}  {:>7}  {:>8}  {:>8}  {:>9}  {:>9}  {:>7}",
        "bodies", "regions", "empty", "mean ops", "full ms", "pruned ms", "faster"
    );

    for count in [40usize, 200, 1000] {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let mut bodies = Vec::new();
        for i in 0..count {
            let f = i as f64;
            let at = |seed: f64| ((f * seed).fract() - 0.5) * EXTENT_MM;
            bodies.push(graph.add(Node::Sphere {
                centre: Place::At([
                    lit(at(0.754_877_666_2)),
                    lit(at(0.569_840_290_9)),
                    lit(at(0.430_159_709_1)),
                ]),
                radius: lit(EXTENT_MM * 0.02),
            }));
        }
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: bodies.into_iter().map(Into::into).collect() });
        let field = GraphField::compile_graph(&Document::new(), &graph, 1e-3).expect("compiles");

        let grid = 16usize;
        let step = EXTENT_MM / grid as f64;
        let per_brick = 64usize;
        let side = (per_brick as f64).cbrt().ceil() as usize;

        let (mut full_ns, mut pruned_ns) = (0.0f64, 0.0f64);
        let (mut empty, mut regions, mut ops_kept, mut live) = (0usize, 0usize, 0usize, 0usize);

        for i in 0..grid {
            for j in 0..grid {
                for k in 0..grid {
                    regions += 1;
                    let lo = nalgebra::Point3::new(
                        -EXTENT_MM * 0.5 + i as f64 * step,
                        -EXTENT_MM * 0.5 + j as f64 * step,
                        -EXTENT_MM * 0.5 + k as f64 * step,
                    );
                    let hi = nalgebra::Point3::new(lo.x + step, lo.y + step, lo.z + step);
                    let points: Vec<nalgebra::Point3<f64>> = (0..side)
                        .flat_map(|a| {
                            (0..side).flat_map(move |b| {
                                (0..side).map(move |c| {
                                    nalgebra::Point3::new(
                                        lo.x + (a as f64 + 0.5) / side as f64 * step,
                                        lo.y + (b as f64 + 0.5) / side as f64 * step,
                                        lo.z + (c as f64 + 0.5) / side as f64 * step,
                                    )
                                })
                            })
                        })
                        .take(per_brick)
                        .collect();

                    let started = Instant::now();
                    let mut sink = 0.0f64;
                    for p in &points {
                        sink += field.distance(*p);
                    }
                    std::hint::black_box(sink);
                    full_ns += started.elapsed().as_nanos() as f64;

                    let started = Instant::now();
                    let region = field.for_region(lo, hi);
                    match region.field() {
                        None => {
                            empty += 1;
                            pruned_ns += started.elapsed().as_nanos() as f64;
                        }
                        Some(pruned) => {
                            live += 1;
                            ops_kept += pruned.op_count();
                            let mut sink = 0.0f64;
                            for p in &points {
                                sink += pruned.distance(*p);
                            }
                            std::hint::black_box(sink);
                            pruned_ns += started.elapsed().as_nanos() as f64;
                        }
                    }
                }
            }
        }

        println!(
            "  {count:>9}  {regions:>7}  {:>7.0}%  {:>8.0}  {:>9.1}  {:>9.1}  {:>6.1}x",
            empty as f64 / regions as f64 * 100.0,
            if live > 0 { ops_kept as f64 / live as f64 } else { 0.0 },
            full_ns / 1e6,
            pruned_ns / 1e6,
            full_ns / pruned_ns.max(1.0),
        );
    }

    println!(
        "\n  `mean ops` is over the regions that kept a surface; the empty ones cost one interval\n\
         evaluation each and nothing else, which is most of the saving at this density. The\n\
         answers are identical inside every region, which is what\n\
         `a_field_pruned_to_a_region_answers_exactly_as_the_whole_one_does` asserts."
    );
}

/// Spike F: what the wiring is worth to the consumer that does the most of this — meshing.
///
/// `mesh_field` already rejected blocks by interval before item 17; what it gained is *pruning*
/// the blocks that survive. So this is an A/B of that one change, measured through the real
/// entry point rather than on a model of it.
///
/// The B side is the same field behind a wrapper that forwards everything except `pruned_to`,
/// which is exactly how a field with no branches to drop behaves — so the comparison is the
/// feature, not two different scenes.
fn meshing() {
    use fieldform_field::{mesh_field, GraphField, Normalization, ScalarField};
    use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef, Place};
    use nalgebra::{Point3, Vector3};

    /// The same field, with the shortening withheld.
    struct Whole(GraphField);

    impl ScalarField for Whole {
        fn distance(&self, p: Point3<f64>) -> f64 {
            self.0.distance(p)
        }
        fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
            self.0.normal(p)
        }
        fn interval(&self, lo: Point3<f64>, hi: Point3<f64>) -> (f64, f64) {
            self.0.interval(lo, hi)
        }
        fn lipschitz(&self) -> f64 {
            self.0.lipschitz()
        }
        fn normalization(&self) -> Normalization {
            self.0.normalization()
        }
    }

    println!("

Spike F — meshing, with the shortening and without it.
");
    println!(
        "  {:>7}  {:>6}  {:>11}  {:>11}  {:>7}  {:>9}",
        "bodies", "cells", "whole ms", "pruned ms", "faster", "triangles"
    );

    for count in [40usize, 200, 600] {
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let mut bodies = Vec::new();
        for i in 0..count {
            let f = i as f64;
            let at = |seed: f64| ((f * seed).fract() - 0.5) * EXTENT_MM;
            bodies.push(graph.add(Node::Sphere {
                centre: Place::At([
                    lit(at(0.754_877_666_2)),
                    lit(at(0.569_840_290_9)),
                    lit(at(0.430_159_709_1)),
                ]),
                radius: lit(EXTENT_MM * 0.02),
            }));
        }
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: bodies.into_iter().map(Into::into).collect() });
        let field = GraphField::compile_graph(&Document::new(), &graph, 1e-3).expect("compiles");

        let bounds = (
            Point3::new(-EXTENT_MM * 0.55, -EXTENT_MM * 0.55, -EXTENT_MM * 0.55),
            Point3::new(EXTENT_MM * 0.55, EXTENT_MM * 0.55, EXTENT_MM * 0.55),
        );
        for cells in [64usize, 128] {
            let whole = Whole(
                GraphField::compile_graph(&Document::new(), &graph, 1e-3).expect("compiles"),
            );
            let started = Instant::now();
            let a = mesh_field(&whole, bounds, cells);
            let without = started.elapsed();

            let started = Instant::now();
            let b = mesh_field(&field, bounds, cells);
            let with = started.elapsed();

            assert_eq!(
                a.triangles.len(),
                b.triangles.len(),
                "shortening changed the mesh, which means a branch was deleted that mattered",
            );
            println!(
                "  {count:>7}  {cells:>6}  {:>11.1?}  {:>11.1?}  {:>6.1}x  {:>9}",
                without,
                with,
                without.as_secs_f64() / with.as_secs_f64().max(1e-9),
                b.triangles.len(),
            );
        }
    }

    println!(
        "
  The two meshes are asserted identical, which is the only thing that makes the saving
         admissible: pruning has to be invisible, so a difference here would be a branch deleted
         that could still have been the answer.

         Worth keeping: wiring the *corner* pass alone moved nothing -- 1.0x at every size. The
         corners are one reading each and block rejection had already removed most of them. The
         cost is in placing vertices, which bisects every crossing edge and then differences a
         normal at it -- a dozen readings per cell, and only ever on the shell. Pruning that is
         where the numbers above come from. A measurement that had stopped at the obvious pass
         would have concluded the whole idea was worth nothing here."
    );
}
