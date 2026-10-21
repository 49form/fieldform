// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a Voronoi cell actually costs and actually builds.
//!
//! `voronoi-lattice-spec.md` §4.2 asks for three numbers to be *measured* rather than assumed,
//! and §4.5 asks for a fourth:
//!
//! 1. **How far the measure understates the true distance.** The measure is exact on the axis
//!    joining two seeds and degrades off it, without limit as the lateral offset grows. In a
//!    lattice the relevant seeds are the near ones, so the degradation should be bounded — but
//!    "should be" is not a number. The truth here is the exact one §4.4 names: for a point whose
//!    nearest seed is `a`, the distance to the boundary of `a`'s Voronoi cell is
//!    `min over b of dot(p − (a+b)/2, n̂_ab)`, because the cell is an intersection of half-spaces
//!    and a convex polyhedron's nearest boundary point is the foot of the perpendicular to its
//!    nearest supporting plane. That is unavailable to the *field* — it needs the complete
//!    neighbour set, and a truncated one overstates — but it is perfectly available here, where a
//!    wide ring can be paid for.
//! 2. **What that costs a tracer**, which is the consequence the ratio has.
//! 3. **How often one ring cannot answer at all**, which is what the fixed ceiling costs. §4.3
//!    prescribed widening to `R = 2` there, and that is unsound: the ceiling is
//!    `R + (1 - jitter)/2`, so widening raises it by a whole cell and steps the value upward
//!    across the surface where `d_1` reaches it. `voronoi_measure` carries the argument.
//! 4. **How much thicker the wall comes out than the number typed.** §4.5 asked for a per-kind
//!    slack constant to divide that out; the *thinnest* column is what argued against one, and
//!    `StochasticKind::wall_spread` carries the conclusion.
//!
//! The seeds are rebuilt here from the public `voronoi_hash3` rather than read out of the
//! lattice's own pass, so a measurement is a second opinion rather than a photograph.
//!
//! ```bash
//! cargo run --release -p fieldform-field --example voronoi_profile
//! ```

use fieldform_field::{
    voronoi_hash3, CellKind, Lattice, LatticeMap, Phase, ScalarField, StochasticKind,
};
use nalgebra::{Point3, Vector3};

const SEED: u32 = 0x5EED_1234;
const CELL: f64 = 10.0;

fn seed_at(c: [i32; 3], jitter: f64) -> Point3<f64> {
    let h = voronoi_hash3(c, SEED);
    let unit = |w: u32| w as f64 * (1.0 / 4_294_967_296.0);
    Point3::new(
        c[0] as f64 + 0.5 + jitter * (unit(h[0]) - 0.5),
        c[1] as f64 + 0.5 + jitter * (unit(h[1]) - 0.5),
        c[2] as f64 + 0.5 + jitter * (unit(h[2]) - 0.5),
    )
}

/// Every seed within `radius` cells of `u`, nearest first.
fn ring(u: Point3<f64>, jitter: f64, radius: i32) -> Vec<(f64, Point3<f64>)> {
    let base = [u.x.floor() as i32, u.y.floor() as i32, u.z.floor() as i32];
    let mut out = Vec::new();
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let s = seed_at([base[0] + dx, base[1] + dy, base[2] + dz], jitter);
                out.push(((u - s).norm(), s));
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// The exact distance from `u` to the wall complex, in cell units — §4.4's half-space form over
/// a neighbourhood wide enough that no omitted seed can contribute a nearer plane.
fn exact_wall_distance(u: Point3<f64>, jitter: f64) -> f64 {
    let seeds = ring(u, jitter, 3);
    let a = seeds[0].1;
    let mut best = f64::INFINITY;
    for (_, b) in seeds.iter().skip(1) {
        let axis = b - a;
        let len = axis.norm();
        if len < 1e-12 {
            continue;
        }
        // Toward `b` from the query, so a point sitting on `a` reads half the gap rather than
        // minus half of it. The first spelling of this had the sign the other way, and the
        // symptom was every probe reading nearer the wall than it could possibly be.
        best = best.min(((a + axis * 0.5) - u).dot(&(axis / len)));
    }
    best
}

/// Points spread over a few cells, off every symmetry.
fn probes(n: usize) -> Vec<Point3<f64>> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                out.push(Point3::new(
                    -1.3 + 5.7 * (i as f64 + 0.317) / n as f64,
                    0.41 + 6.1 * (j as f64 + 0.233) / n as f64,
                    -2.7 + 5.3 * (k as f64 + 0.451) / n as f64,
                ));
            }
        }
    }
    out
}

fn lattice(kind: StochasticKind, jitter: f64, size: f64) -> Lattice {
    Lattice {
        cell: CellKind::Stochastic { kind, jitter, seed: SEED },
        map: LatticeMap::Rectangular { cell: Vector3::new(CELL, CELL, CELL) },
        phase: match kind {
            StochasticKind::VoronoiFoam => Phase::sheet(size),
            StochasticKind::VoronoiBeam => Phase::Strut { radius_mm: size },
        },
    }
}

/// The lengths of the material crossings a line makes, in millimetres.
fn chords(field: &dyn ScalarField, from: Point3<f64>, along: Vector3<f64>, far: f64) -> Vec<f64> {
    let step = 0.002;
    let dir = along.normalize();
    let mut out = Vec::new();
    let mut entered: Option<f64> = None;
    let mut t = 0.0;
    let mut was = field.distance(from) < 0.0;
    while t < far {
        t += step;
        let inside = field.distance(from + dir * t) < 0.0;
        if inside && !was {
            entered = Some(t);
        } else if !inside && was {
            if let Some(a) = entered.take() {
                out.push(t - a);
            }
        }
        was = inside;
    }
    out
}

/// Smallest, one-in-a-hundred, median and mean of a sample.
fn summary(mut v: Vec<f64>) -> (f64, f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (v[0], v[v.len() / 100], v[v.len() / 2], mean)
}

fn main() {
    println!("Voronoi, Track A — cell {CELL} mm, seed {SEED:#x}\n");

    // ---------------------------------------------------------------- 1. the bound's slack
    println!("How much the measure understates the true distance to the wall complex");
    println!("(§4.4's exact half-space form over a three-cell ring, away from the walls)\n");
    println!("  jitter    samples   mean ratio   worst ratio");
    for jitter in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let mut ratios = Vec::new();
        for u in probes(15) {
            let truth = exact_wall_distance(u, jitter);
            // Away from the wall, where a ratio means something: right on it both are nought.
            if truth < 0.05 {
                continue;
            }
            let (measure, _) = measure_of(u, jitter, StochasticKind::VoronoiFoam);
            ratios.push(measure / truth);
        }
        let (worst, _, _, mean) = summary(ratios.clone());
        println!(
            "  {jitter:>5.2}    {:>7}   {mean:>10.4}   {worst:>11.4}",
            ratios.len()
        );
    }

    // ---------------------------------------------------------------- 2. what it costs a ray
    println!("\nSphere-trace steps to reach a surface. A gyroid sheet of the same cell and the");
    println!("same wall is the peer -- both are closed surfaces a ray meets at once, where a beam");
    println!("lattice is mostly empty and flatters itself on the rays that pass clean through.");
    println!("Counted over the rays that hit.\n");
    println!("  field                          hit    mean steps   worst");
    let gyroid = Lattice {
        cell: CellKind::Tpms(fieldform_field::TpmsKind::Gyroid),
        map: LatticeMap::Rectangular { cell: Vector3::new(CELL, CELL, CELL) },
        phase: Phase::sheet(1.0),
    };
    let beam = Lattice {
        cell: CellKind::Beam(fieldform_field::BeamKind::Bcc),
        map: LatticeMap::Rectangular { cell: Vector3::new(CELL, CELL, CELL) },
        phase: Phase::Strut { radius_mm: 0.5 },
    };
    for (name, field) in [
        ("Gyroid sheet, wall 1.0", &gyroid),
        ("BCC beam, r 0.5", &beam),
        ("Voronoi foam, wall 1.0 j0.0", &lattice(StochasticKind::VoronoiFoam, 0.0, 1.0)),
        ("Voronoi foam, wall 1.0 j0.5", &lattice(StochasticKind::VoronoiFoam, 0.5, 1.0)),
        ("Voronoi foam, wall 1.0 j1.0", &lattice(StochasticKind::VoronoiFoam, 1.0, 1.0)),
        ("Voronoi beam, r 0.5 j1.0", &lattice(StochasticKind::VoronoiBeam, 1.0, 0.5)),
    ] {
        let (hit, mean, worst) = trace_cost(field);
        println!("  {name:<29}  {:>4.0}%   {mean:>10.1}   {worst:>5}", hit * 100.0);
    }

    // ------------------------------------------- 3. how often one ring cannot answer at all
    println!("\nHow often one ring cannot answer — where `d₁` reaches the ceiling nothing may be");
    println!("concluded and the measure reads nought, which is material. §4.3 prescribed widening");
    println!("to R = 2 there; that moves the ceiling by a whole cell and steps the value, so the");
    println!("ceiling is fixed and this is what the fixed one costs instead.\n");
    println!("  jitter    ℓ₁      silenced");
    for jitter in [0.0, 0.25, 0.464, 0.5, 0.75, 0.9, 1.0] {
        let limit = 1.0 + (1.0 - jitter) * 0.5;
        let all = probes(21);
        let fired = all
            .iter()
            .filter(|u| ring(**u, jitter, 1)[0].0 > limit)
            .count();
        println!(
            "  {jitter:>5.3}   {limit:>5.3}   {:>8.2}%",
            100.0 * fired as f64 / all.len() as f64
        );
    }

    // ----------------------------------------------- 4. the wall that comes out of a number
    println!("\nWhat a typed thickness actually builds — never thinner, often thicker");
    println!("(chords across the material along 900 lines, 2 µm steps)\n");
    println!("  kind            jitter   typed   thinnest      1%    median     mean   median/typed");
    for kind in StochasticKind::ALL {
        for jitter in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let typed = 1.0;
            let field = lattice(kind, jitter, typed);
            let mut all = Vec::new();
            for i in 0..30 {
                for j in 0..30 {
                    let from = Point3::new(
                        -17.0 + 0.9137 * i as f64,
                        -13.0 + 0.8713 * j as f64,
                        -40.37,
                    );
                    all.extend(chords(&field, from, Vector3::new(0.0, 0.0, 1.0), 80.0));
                }
            }
            // The wall is what it is; a chord taken at a glancing angle to one is longer, and a
            // chord through a junction longer still. The median is the honest middle of that.
            let built = if kind == StochasticKind::VoronoiFoam { typed } else { typed * 2.0 };
            let (thinnest, low, median, mean) = summary(all.clone());
            println!(
                "  {:<14}  {jitter:>5.2}   {built:>5.2}   {thinnest:>8.4} {low:>7.4}  {median:>7.4} {mean:>8.4}   {:>8.3}",
                kind.label(),
                median / built,
            );
        }
    }
}

/// The lattice's own measure, in cell units, read back out of a field built with no thickness —
/// so its value is the measure and nothing else.
fn measure_of(u: Point3<f64>, jitter: f64, kind: StochasticKind) -> (f64, ()) {
    let field = lattice(kind, jitter, 0.0);
    let p = Point3::new(u.x * CELL, u.y * CELL, u.z * CELL);
    (field.distance(p) / CELL, ())
}

fn trace_cost(field: &Lattice) -> (f64, f64, usize) {
    let mut steps = Vec::new();
    let mut worst = 0usize;
    let mut cast = 0usize;
    let n = 64;
    for i in 0..n {
        for j in 0..n {
            // Off the cell grid on every axis. Starting at a round multiple of the cell put
            // every ray on a wall plane at nought jitter, and the whole column read one step.
            let from = Point3::new(
                -22.0 + 0.6871 * i as f64,
                -19.0 + 0.6133 * j as f64,
                -63.7,
            );
            let dir = Vector3::new(0.0, 0.0, 1.0);
            let mut t = 0.0;
            let mut count = 0usize;
            let mut hit = false;
            cast += 1;
            while t < 120.0 && count < 512 {
                let d = field.distance(from + dir * t);
                count += 1;
                if d < 1e-3 {
                    hit = true;
                    break;
                }
                t += d;
            }
            if hit {
                steps.push(count as f64);
                worst = worst.max(count);
            }
        }
    }
    (
        steps.len() as f64 / cast as f64,
        steps.iter().sum::<f64>() / steps.len() as f64,
        worst,
    )
}
