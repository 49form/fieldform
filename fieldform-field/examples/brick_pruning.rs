// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Spike A and B from the brick-evaluation-note design note: does per-brick interval
//! pruning actually shorten the expression here, and is space actually sparse?
//!
//! The note argues that a brick evaluator's speed comes from three separate gains and that the
//! first two are the transferable ones. Both are claims about *this* codebase's geometry, not
//! about brick evaluators in general, so neither is worth believing without numbers.
//!
//! - **Gain one (§3)** — over a small box, most of a wide `min` tree cannot be the minimum
//!   anywhere inside it, so those branches can be deleted from the compiled tape rather than
//!   skipped at runtime. Measured here as tape length before and after
//!   `Shape::simplify`.
//! - **Gain two (§4)** — most bricks contain no surface at all, and one interval evaluation
//!   rejects each. Measured here as the fraction of bricks whose interval straddles zero. The
//!   note flags this as *genuinely unclear* for lattices, which have surface almost
//!   everywhere, so a gyroid is measured beside the scattered-primitive case rather than
//!   assumed to behave like it.
//!
//! This is also the measurement `surface-array-spec.md` §10 asks for before building that
//! node, and `backlog.md` item 10 lists under "Do first" — a wide `min` tree is exactly what
//! scattering N instances produces. One spike, two consumers.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example brick_pruning`

use std::f64::consts::TAU;

use fidget_core::shape::Shape;
use fidget_core::{
    context::Tree,
    shape::ShapeVars,
    types::Interval,
    var::{Var, VarIndex},
    vm::{VmFunction, VmShape},
};

/// The world the scenes are built in, in millimetres.
const EXTENT_MM: f64 = 1000.0;

fn main() {
    println!("Spike A — does per-brick pruning shorten the tape?\n");
    println!(
        "{:>10}  {:>8}  {:>9}  {:>9}  {:>9}  {:>7}",
        "instances", "bricks", "full tape", "mean kept", "median", "factor"
    );
    for count in [10usize, 50, 200, 1000] {
        let scene = scattered_spheres(count);
        let row = measure(&scene, 16);
        println!(
            "{count:>10}  {:>8}  {:>9}  {:>9.1}  {:>9}  {:>6.1}x",
            row.bricks, row.full_tape, row.mean_kept, row.median_kept, row.factor,
        );
    }

    println!("\nSpike A — the same 200-instance scene at three brick resolutions\n");
    println!(
        "{:>8}  {:>10}  {:>8}  {:>9}  {:>9}  {:>7}",
        "grid", "brick (mm)", "bricks", "full tape", "mean kept", "factor"
    );
    let scene = scattered_spheres(200);
    for grid in [4usize, 8, 16, 32] {
        let row = measure(&scene, grid);
        println!(
            "{:>8}  {:>10.0}  {:>8}  {:>9}  {:>9.1}  {:>6.1}x",
            format!("{grid}^3"),
            EXTENT_MM / grid as f64,
            row.bricks,
            row.full_tape,
            row.mean_kept,
            row.factor,
        );
    }

    println!("\nSpike B — how much of space actually holds surface?\n");
    println!(
        "{:>28}  {:>8}  {:>10}  {:>9}  {:>8}",
        "scene", "grid", "brick (mm)", "straddling", "share"
    );
    for grid in [8usize, 16, 32] {
        let scene = scattered_spheres(200);
        let row = measure(&scene, grid);
        println!(
            "{:>28}  {:>8}  {:>10.0}  {:>9}  {:>7.1}%",
            "200 scattered spheres",
            format!("{grid}^3"),
            EXTENT_MM / grid as f64,
            row.straddling,
            100.0 * row.straddling as f64 / row.bricks as f64,
        );
    }
    // The case the note says is unclear, and the one the whole line of work is for.
    for cell_mm in [200.0f64, 50.0] {
        for grid in [8usize, 16, 32] {
            let scene = gyroid_block(cell_mm, 1.0);
            let row = measure(&scene, grid);
            println!(
                "{:>28}  {:>8}  {:>10.0}  {:>9}  {:>7.1}%",
                format!("gyroid, {cell_mm:.0} mm cells"),
                format!("{grid}^3"),
                EXTENT_MM / grid as f64,
                row.straddling,
                100.0 * row.straddling as f64 / row.bricks as f64,
            );
        }
    }

    // The crux. Gains one and two were measured on different scenes above, which leaves the
    // question the note actually cares about unanswered: when sparsity is gone, does pruning
    // still pay? A lattice is the case where gain two is known to evaporate.
    println!("\nBoth gains on one scene — including the case where sparsity is gone\n");
    println!(
        "{:>34}  {:>9}  {:>9}  {:>7}  {:>8}",
        "scene (16^3 bricks)", "full tape", "mean kept", "factor", "straddle"
    );
    let scenes: [(&str, Scene); 5] = [
        ("200 scattered spheres", scattered_spheres(200)),
        ("gyroid alone, 50 mm cells", gyroid_block(50.0, 1.0)),
        ("gyroid + 200 spheres", gyroid_with_spheres(50.0, 1.0, 200)),
        (
            "gyroid + 1000 spheres",
            gyroid_with_spheres(50.0, 1.0, 1000),
        ),
        (
            "1000 spheres, gyroid-infilled",
            gyroid_infill(50.0, 1.0, 1000),
        ),
    ];
    for (label, scene) in &scenes {
        let row = measure(scene, 16);
        println!(
            "{label:>34}  {:>9}  {:>9.1}  {:>6.1}x  {:>7.1}%",
            row.full_tape,
            row.mean_kept,
            row.factor,
            100.0 * row.straddling as f64 / row.bricks as f64,
        );
    }

    println!(
        "\nTape length is `Shape::size`. \"kept\" counts only bricks that straddle zero: a\n\
         rejected brick keeps nothing, and averaging those in would flatter the result."
    );
}

/// What one scene measured over one brick grid comes to.
struct Row {
    bricks: usize,
    straddling: usize,
    full_tape: usize,
    mean_kept: f64,
    median_kept: usize,
    factor: f64,
}

/// A scene, as a Fidget tree plus the opaque field variables it references.
struct Scene {
    tree: Tree,
    /// Each opaque leaf's variable, and a closure giving its interval over a box. Empty for
    /// the closed-form scenes.
    opaque: Vec<(VarIndex, Box<dyn Fn([f64; 3], [f64; 3]) -> (f64, f64)>)>,
}

/// Subdivides the world into `grid^3` bricks, prunes the tape against each, and totals up.
fn measure(scene: &Scene, grid: usize) -> Row {
    let shape = VmShape::from(scene.tree.clone());
    let full_tape = shape.size();
    let tape = shape.interval_tape(Default::default());
    let mut eval = Shape::<VmFunction>::new_interval_eval();
    let mut workspace = Default::default();

    let step = EXTENT_MM / grid as f64;
    let origin = -EXTENT_MM * 0.5;
    let mut kept = Vec::new();
    let mut bricks = 0;

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

                // Every opaque leaf needs a range or it is unbounded, and one unbounded
                // variable prunes nothing anywhere it appears — the reason
                // `FieldTerm::interval` exists at all.
                let mut vars: ShapeVars<Interval> = ShapeVars::new();
                for (index, bound) in &scene.opaque {
                    let (a, b) = bound(lo, hi);
                    vars.insert(*index, Interval::new(a as f32, b as f32));
                }

                let x = Interval::new(lo[0] as f32, hi[0] as f32);
                let y = Interval::new(lo[1] as f32, hi[1] as f32);
                let z = Interval::new(lo[2] as f32, hi[2] as f32);
                let (value, trace) = eval
                    .eval_with_vars(&tape, x, y, z, &vars)
                    .expect("interval evaluation with every variable bound");

                // Entirely one sign means no surface in this brick: nothing to draw, and
                // nothing to evaluate a single point of.
                if value.lower() > 0.0 || value.upper() < 0.0 {
                    continue;
                }
                let size = match trace {
                    Some(trace) => shape
                        .simplify(trace, Default::default(), &mut workspace)
                        .expect("the trace came from this shape")
                        .size(),
                    // Nothing to specialise on: the whole tape is live here.
                    None => full_tape,
                };
                kept.push(size);
            }
        }
    }

    kept.sort_unstable();
    let straddling = kept.len();
    let mean_kept = if kept.is_empty() {
        0.0
    } else {
        kept.iter().sum::<usize>() as f64 / kept.len() as f64
    };
    Row {
        bricks,
        straddling,
        full_tape,
        mean_kept,
        median_kept: kept.get(kept.len() / 2).copied().unwrap_or(0),
        factor: if mean_kept > 0.0 {
            full_tape as f64 / mean_kept
        } else {
            0.0
        },
    }
}

/// `count` spheres on a deterministic scatter, unioned — the shape a surface array makes, and
/// the wide `min` tree §3's argument is about.
///
/// Closed-form rather than opaque variables on purpose: a scene of `FieldTerm`s is one tape
/// slot per leaf, so pruning it would measure how many *leaves* were dropped rather than how
/// much *expression*. Both matter, and the mixed case is measured separately below.
fn scattered_spheres(count: usize) -> Scene {
    let mut tree: Option<Tree> = None;
    for i in 0..count {
        // A low-discrepancy scatter, so the spheres spread rather than clumping on a lattice
        // whose period could line up with the brick grid and flatter the result.
        let f = i as f64;
        let at = |seed: f64| (f * seed).fract() - 0.5;
        let centre = [
            at(0.754_877_666_2) * EXTENT_MM,
            at(0.569_840_290_9) * EXTENT_MM,
            at(0.430_159_709_1) * EXTENT_MM,
        ];
        let radius = EXTENT_MM * 0.02;
        let sphere = ((Tree::x() - centre[0]).square()
            + (Tree::y() - centre[1]).square()
            + (Tree::z() - centre[2]).square())
        .sqrt()
            - radius;
        tree = Some(match tree {
            Some(t) => t.min(sphere),
            None => sphere,
        });
    }
    Scene {
        tree: tree.expect("at least one instance"),
        opaque: Vec::new(),
    }
}

/// A gyroid wall, written closed-form so its interval comes from Fidget rather than from the
/// `ScalarField::interval` default.
///
/// That distinction is the point. The default bound is a centre sample widened by the
/// Lipschitz constant times the box's half-diagonal, and a gyroid's constant is `sqrt(6) k` —
/// enormous for a small cell — so over a brick of any size that bound covers everything and
/// rejects nothing. Written as an expression, Fidget bounds the trigonometry directly and the
/// answer is about the geometry rather than about the wrapper.
fn gyroid_block(cell_mm: f64, wall_mm: f64) -> Scene {
    let k = TAU / cell_mm;
    let lipschitz = 6f64.sqrt() * k;
    let (x, y, z) = (Tree::x() * k, Tree::y() * k, Tree::z() * k);
    let f = x.clone().sin() * y.clone().cos()
        + y.clone().sin() * z.clone().cos()
        + z.clone().sin() * x.clone().cos();
    // The same thresholded-and-normalised form `gyroid_spike.rs` measures, so the two agree
    // about what surface is being counted.
    let c = 0.5 * lipschitz * wall_mm * 0.5;
    let wall = (f.abs() - c) / lipschitz;
    // Trimmed to the block, which is what makes it a part rather than an infinite field.
    let half = EXTENT_MM * 0.5 - 1.0;
    let block = Tree::x().abs().max(Tree::y().abs()).max(Tree::z().abs()) - half;
    Scene {
        tree: wall.max(block),
        opaque: Vec::new(),
    }
}

/// A gyroid-filled block with scattered spheres unioned on — a part with an infill, which is
/// closer to what this kernel actually builds than either half alone.
///
/// The point of measuring this and not just the two separately: the gyroid makes every brick
/// straddle zero, so gain two is gone, and what is left is whether gain one still pays on the
/// sphere half of the tree while the lattice half survives everywhere.
fn gyroid_with_spheres(cell_mm: f64, wall_mm: f64, count: usize) -> Scene {
    let gyroid = gyroid_block(cell_mm, wall_mm);
    let spheres = scattered_spheres(count);
    Scene {
        tree: gyroid.tree.min(spheres.tree),
        opaque: Vec::new(),
    }
}

/// A part hollowed out to a gyroid infill: the union of the spheres *intersected* with the
/// lattice wall.
///
/// The topology that matters, and it is not the union above. An infill is `max`, and a `max`
/// prunes on a different condition from a `min` — a branch survives unless it cannot be the
/// *largest*. Measuring only the union would have credited pruning with a result that came
/// from the gyroid being a small value everywhere and so dominating a minimum.
fn gyroid_infill(cell_mm: f64, wall_mm: f64, count: usize) -> Scene {
    let gyroid = gyroid_block(cell_mm, wall_mm);
    let spheres = scattered_spheres(count);
    Scene {
        tree: spheres.tree.max(gyroid.tree),
        opaque: Vec::new(),
    }
}

/// Silences the unused-variant warning while no scene here uses an opaque leaf.
///
/// Kept rather than deleted because the mixed case — a cage or a pipe beside closed-form
/// shapes — is the one fieldform actually builds, and the plumbing for it is the interesting
/// part: an unbound variable is unbounded, and one of those turns pruning off wholesale.
#[allow(dead_code)]
fn opaque_leaf(
    bound: impl Fn([f64; 3], [f64; 3]) -> (f64, f64) + 'static,
) -> (
    Tree,
    (VarIndex, Box<dyn Fn([f64; 3], [f64; 3]) -> (f64, f64)>),
) {
    let var = Var::new();
    let index = var.index().expect("Var::new is always indexed");
    (Tree::from(var), (index, Box::new(bound)))
}
