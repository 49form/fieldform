// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! One ray of `tile_cull_sweep`'s failing frame, marched on the CPU step for step as
//! `trace_primary` marches it, in two arms: the fold over every body and the fold over the bodies
//! the tile kept. The question is *which step* loses the hit, and by what mechanism -- because the
//! vouch (plan §5 V8) is a sound argument that a relaxed step never crosses a surface of the field
//! it reads, and the kept fold is an exact distance to the kept bodies, so the masked march ought
//! to find exactly what the unmasked one finds. It measurably does not, so either the march
//! deviates from the argument or something beside the step does.
//!
//! Every step records whether its segment crossed the material of a body the tile kept, which is
//! the thing no vouched step may do. A crossing that was never withdrawn is the mechanism.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example cull_ray_trace -- <document.fform> [--pixel 230,196]
//!  [--turn 47] [--lift 0.0] [--away 3.6] [--persp] [--relax 1.4] [--no-grid] [--no-confirm]
//!  [--read-before-hop] [--vouch-by-grid] [--gpu]`

use fieldform_field::{graph_extent, part_extent, Extent, GraphField, ScalarField};
use fieldform_gpu::{generate_for_measurement, Camera, Renderer};
use fieldform_scene::{Graph, NodeId, ParamRef};
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

const WIDTH: u32 = 720;
const HEIGHT: u32 = 405;
const STEP_SAFETY: f32 = 0.9;
const RELAX_REST: u32 = 3;
const CONFIRM: f32 = 4.0;
const CONFIRM_REACH: f32 = 256.0;
const SKIP_HOPS: u32 = 96;
const CULL_PAD_FRACTION: f64 = 1e-3;
const CULL_PAD_FLOOR_MM: f64 = 0.5;
/// The step budget the sweep's reference arm marches with. The sweep's own 512 is what the
/// viewport uses, and a plain march can exhaust it on a ray that grazes a sphere for a long way
/// -- `--turn 32 --away 3.6 --persp --pixel 471,187` does, at 512 steps and relaxation one -- so
/// a reference at that budget reports the relaxed arm as *inventing* the body behind. The
/// reference marches with sixteen times that, so a difference against it is the arm's.
const REFERENCE_BUDGET: u32 = 8192;

static VERBOSE_HOPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// The graph cut down to what one root reads, so it compiles to that body's field alone.
fn subgraph_of(graph: &Graph, root: NodeId) -> Graph {
    let mut keep = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !keep.insert(id) {
            continue;
        }
        if let Some(node) = graph.node(id) {
            stack.extend(node.inputs());
        }
    }
    let mut out = graph.clone();
    let drop: Vec<NodeId> = out.nodes().map(|(id, _)| id).filter(|id| !keep.contains(id)).collect();
    for id in drop {
        out.remove(id);
    }
    out
}

/// `tile_cull`'s five planes on the CPU, copied from `tile_cull_sweep.rs`.
fn ruled_out_on_the_cpu(
    camera: &Camera,
    tile: (u32, u32),
    lo: Point3<f64>,
    hi: Point3<f64>,
) -> (bool, [f64; 5]) {
    const PAD: f64 = 1.0;
    let t = 8.0;
    let u0 = ((tile.0 as f64) * t - PAD) / WIDTH as f64 * 2.0 - 1.0;
    let u1 = (((tile.0 + 1) as f64) * t + PAD) / WIDTH as f64 * 2.0 - 1.0;
    let v0 = 1.0 - ((tile.1 as f64) * t - PAD) / HEIGHT as f64 * 2.0;
    let v1 = 1.0 - (((tile.1 + 1) as f64) * t + PAD) / HEIGHT as f64 * 2.0;
    let ray = |u: f64, v: f64| {
        let across = camera.right * (u * camera.half_height * camera.aspect)
            + camera.up * (v * camera.half_height);
        if camera.orthographic {
            (camera.eye + across, camera.forward.normalize())
        } else {
            (camera.eye, (camera.forward + across).normalize())
        }
    };
    let corner = [ray(u0, v0), ray(u1, v0), ray(u1, v1), ray(u0, v1)];
    let middle = ray((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    let mut planes: Vec<(Vector3<f64>, Point3<f64>)> = Vec::new();
    for k in 0..4 {
        let a = corner[k];
        let b = corner[(k + 1) % 4];
        let mut n = a.1.cross(&((b.0 + b.1) - a.0));
        let away = corner[(k + 2) % 4];
        if n.dot(&((away.0 + away.1) - a.0)) < 0.0 {
            n = -n;
        }
        planes.push((n, a.0));
    }
    planes.push((middle.1, middle.0));
    let mut gaps = [0.0_f64; 5];
    let mut out = false;
    for (k, (n, through)) in planes.iter().enumerate() {
        let far = Point3::new(
            if n.x > 0.0 { hi.x } else { lo.x },
            if n.y > 0.0 { hi.y } else { lo.y },
            if n.z > 0.0 { hi.z } else { lo.z },
        );
        gaps[k] = n.dot(&(far - through)) / n.norm().max(1e-30);
        if gaps[k] < 0.0 {
            out = true;
        }
    }
    (out, gaps)
}

/// The bodies' fields, folded as the shader folds them: a running `min` over a chosen subset.
struct Fold<'a> {
    fields: &'a [GraphField],
    kept: Vec<bool>,
}

impl Fold<'_> {
    /// The value in `f32`, as the shader's `field` returns it, with the body that gave it.
    fn value(&self, p: Point3<f32>) -> (f32, usize) {
        let q = Point3::new(p.x as f64, p.y as f64, p.z as f64);
        let mut best = 3.4e38_f32;
        let mut owner = 0;
        for (i, f) in self.fields.iter().enumerate() {
            if !self.kept[i] {
                continue;
            }
            let v = f.distance(q) as f32;
            if v < best {
                best = v;
                owner = i;
            }
        }
        (best, owner)
    }
}

/// The skip grid as `skip_build` fills it, from the unmasked field, a cell at a time on demand.
struct SkipGrid<'a> {
    lo: Vector3<f32>,
    n: u32,
    cell: f32,
    half: f32,
    all: Fold<'a>,
    cells: std::cell::RefCell<HashMap<u32, f32>>,
}

impl SkipGrid<'_> {
    fn clearance(&self, at: Vector3<u32>) -> f32 {
        let key = (at.z * self.n + at.y) * self.n + at.x;
        if let Some(c) = self.cells.borrow().get(&key) {
            return *c;
        }
        let centre = self.lo
            + Vector3::new(at.x as f32 + 0.5, at.y as f32 + 0.5, at.z as f32 + 0.5) * self.cell;
        let (value, _) = self.all.value(Point3::from(centre));
        let c = value - self.half;
        self.cells.borrow_mut().insert(key, c);
        c
    }

    /// `skip_distance` in march.wgsl.
    fn distance(&self, p: Point3<f32>, dir: Vector3<f32>, epsilon: f32) -> f32 {
        let n = self.n as f32;
        let g = (p.coords - self.lo) / self.cell;
        if g.x < 0.0 || g.y < 0.0 || g.z < 0.0 || g.x >= n || g.y >= n || g.z >= n {
            return 0.0;
        }
        let at = Vector3::new(g.x as u32, g.y as u32, g.z as u32);
        let clear = self.clearance(at);
        if clear <= epsilon {
            return 0.0;
        }
        let inside = g - g.map(f32::floor);
        let mut wall = f32::MAX;
        for k in 0..3 {
            let ahead = if dir[k] > 0.0 { 1.0 - inside[k] } else { inside[k] };
            let per = if dir[k].abs() > 1e-12 { ahead / dir[k].abs() } else { 3.4e38 };
            wall = wall.min(per);
        }
        let wall = wall * self.cell;
        clear.max(wall + (self.cell * 1e-3).min(clear * 0.5))
    }

    /// `skip_ahead` in march.wgsl.
    fn ahead(&self, origin: Point3<f32>, dir: Vector3<f32>, start: f32, far: f32, epsilon: f32) -> f32 {
        let mut t = start;
        let verbose = VERBOSE_HOPS.load(std::sync::atomic::Ordering::Relaxed);
        for _ in 0..3 * SKIP_HOPS {
            let p = origin + dir * t;
            let clear = self.distance(p, dir, epsilon);
            if verbose {
                let g = (p.coords - self.lo) / self.cell;
                let cell_clear = if g.x >= 0.0 && g.y >= 0.0 && g.z >= 0.0 && g.x < self.n as f32 && g.y < self.n as f32 && g.z < self.n as f32 {
                    self.clearance(Vector3::new(g.x as u32, g.y as u32, g.z as u32))
                } else {
                    f32::NAN
                };
                println!(
                    "        hop? at t {t:.3} cell ({},{},{}) clearance {cell_clear:.3} true field {:.3} -> hop {clear:.3}",
                    g.x.floor(), g.y.floor(), g.z.floor(), self.all.value(p).0
                );
            }
            if clear <= 0.0 {
                break;
            }
            t += clear;
            if t > far {
                break;
            }
        }
        t
    }
}

fn confirm_distance(d: f32, came_from: f32, came_by: f32, epsilon: f32) -> f32 {
    let least = CONFIRM * epsilon;
    if came_by <= 0.0 {
        return least;
    }
    let rate = (came_from - d) / came_by;
    if rate <= 0.0 {
        return least;
    }
    ((d + (CONFIRM - 1.0) * epsilon) / rate).clamp(least, CONFIRM_REACH * epsilon)
}

#[derive(Clone)]
struct Step {
    /// Where the ray stood when the step began, and where the reading was taken.
    from: f32,
    hopped_to: f32,
    d: f32,
    owner: usize,
    reach: f32,
    before: f32,
    event: String,
    /// Where the step left the ray.
    to: f32,
    /// Whether any segment travelled during this step -- hop or step -- passed through the
    /// material of a kept body, and whether the step was withdrawn.
    crossed_kept: bool,
    withdrawn: bool,
}

struct Options {
    relax: f32,
    grid: bool,
    confirm: bool,
    /// The candidate repair, as `march.wgsl` would carry it: a relaxed step is vouched for
    /// before the grid may hop from where it landed -- by the landing cell's clearance where
    /// that is enough, since it is a lower bound on the reading the vouch would take, and by a
    /// reading where it is not. After a reading vouches, the hop it held back is taken.
    fixed: bool,
    budget: u32,
}

struct Outcome {
    hit: Option<(f32, usize)>,
    steps: Vec<Step>,
    readings: usize,
}

/// Whether the kept fold is negative anywhere on `[a, b]` along the ray, sampled finely.
fn crosses(fold: &Fold, origin: Point3<f32>, dir: Vector3<f32>, a: f32, b: f32) -> bool {
    if b <= a {
        return false;
    }
    let samples = 400;
    (0..=samples).any(|i| {
        let t = a + (b - a) * i as f32 / samples as f32;
        fold.value(origin + dir * t).0 < 0.0
    })
}

/// `trace_primary` in march.wgsl, from the point where the scene sphere was entered.
#[allow(clippy::too_many_arguments)]
fn march(
    fold: &Fold,
    grid: Option<&SkipGrid>,
    origin: Point3<f32>,
    dir: Vector3<f32>,
    start: f32,
    far: f32,
    epsilon: f32,
    o: &Options,
) -> Outcome {
    let mut steps = Vec::new();
    let mut readings = 0usize;
    let mut t = start;
    let mut hit = None;
    let relax = o.relax;
    let mut reach = 0.0_f32;
    let mut before = 0.0_f32;
    let mut rest = 0u32;
    let mut came_from = 0.0_f32;
    let mut came_by = 0.0_f32;
    let mut refused_at = -1.0_f32;
    for _ in 0..o.budget {
        if t > far {
            break;
        }
        let from = t;
        let mut record = Step {
            from,
            hopped_to: t,
            d: f32::NAN,
            owner: usize::MAX,
            reach,
            before,
            event: String::new(),
            to: t,
            crossed_kept: false,
            withdrawn: false,
        };
        let mut may_hop = o.grid && grid.is_some();
        if o.fixed && reach > 0.0 {
            let clear = match (o.grid, grid) {
                (true, Some(g)) => {
                    let gg = ((origin + dir * t).coords - g.lo) / g.cell;
                    let n = g.n as f32;
                    let inside_grid = gg.x >= 0.0
                        && gg.y >= 0.0
                        && gg.z >= 0.0
                        && gg.x < n
                        && gg.y < n
                        && gg.z < n;
                    if inside_grid {
                        g.clearance(Vector3::new(gg.x as u32, gg.y as u32, gg.z as u32)).max(0.0)
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
            if (clear + before) * STEP_SAFETY >= reach {
                record.event.push_str(&format!("vouched by grid (clear {clear:.3}) "));
                reach = 0.0;
            } else {
                may_hop = false;
            }
        }
        if may_hop {
            let hopped = grid.unwrap().ahead(origin, dir, t, far, epsilon);
            if hopped > t {
                reach = 0.0;
                came_by = 0.0;
                record.event.push_str(&format!("hop +{:.3} ", hopped - t));
            }
            t = hopped;
            record.hopped_to = t;
        }
        if t > far {
            record.event.push_str("past far");
            record.to = t;
            steps.push(record);
            break;
        }
        let (d, owner) = fold.value(origin + dir * t);
        readings += 1;
        record.d = d;
        record.owner = owner;
        if reach > 0.0 && (d < epsilon || (d.abs() + before) * STEP_SAFETY < reach) {
            let plain = (before * STEP_SAFETY).max(epsilon);
            t = t - reach + plain;
            record.event.push_str(&format!(
                "WITHDRAWN ((|d|+before)*0.9 = {:.3} < reach {:.3}) -> t {:.3}",
                (d.abs() + before) * STEP_SAFETY,
                reach,
                t
            ));
            record.withdrawn = true;
            reach = 0.0;
            rest = RELAX_REST;
            came_from = before;
            came_by = plain;
            record.to = t;
            record.crossed_kept = crosses(fold, origin, dir, from, record.hopped_to);
            steps.push(record);
            continue;
        }
        if reach > 0.0 {
            record.event.push_str("vouched ");
            if o.fixed {
                reach = 0.0;
                if o.grid {
                    if let Some(g) = grid {
                        let hopped = g.ahead(origin, dir, t, far, epsilon);
                        if hopped > t {
                            record.event.push_str(&format!(
                                "by reading; then hop +{:.3} (reading spent)",
                                hopped - t
                            ));
                            t = hopped;
                            came_by = 0.0;
                            record.to = t;
                            steps.push(record);
                            continue;
                        }
                    }
                }
            }
        }
        if d < 0.0 && refused_at >= 0.0 {
            t = refused_at;
            hit = Some((t, owner));
            record.event.push_str(&format!("RECLAIMED refused candidate at {t:.3}: HIT"));
            record.to = t;
            steps.push(record);
            break;
        }
        if d >= epsilon {
            refused_at = -1.0;
        }
        if d < epsilon {
            if !o.confirm {
                hit = Some((t, owner));
                record.event.push_str("candidate, HIT (no confirm)");
                record.to = t;
                steps.push(record);
                break;
            }
            let look = confirm_distance(d, came_from, came_by, epsilon);
            let (beyond, _) = fold.value(origin + dir * (t + look));
            readings += 1;
            if beyond < 0.0 || beyond < d {
                hit = Some((t, owner));
                record.event.push_str(&format!(
                    "candidate, look {look:.4} reads {beyond:.5}: HIT"
                ));
                record.to = t;
                steps.push(record);
                break;
            }
            refused_at = t;
            came_from = d;
            came_by = CONFIRM * epsilon;
            t += came_by;
            reach = 0.0;
            record.event.push_str(&format!(
                "candidate REFUSED (look {look:.4} reads {beyond:.5}), walk 4eps"
            ));
            record.to = t;
            record.crossed_kept = crosses(fold, origin, dir, from, t);
            steps.push(record);
            continue;
        }
        came_from = d;
        if relax > 1.0 && rest == 0 {
            reach = (d * relax).max(epsilon);
            before = d;
            came_by = reach;
            t += reach;
            record.event.push_str(&format!("relaxed step {reach:.3}"));
        } else {
            rest = if rest > 0 { rest - 1 } else { 0 };
            reach = 0.0;
            came_by = (d * STEP_SAFETY).max(epsilon);
            t += came_by;
            record.event.push_str(&format!("plain step {came_by:.3} (rest {rest})"));
        }
        record.to = t;
        record.crossed_kept = crosses(fold, origin, dir, from, t);
        steps.push(record);
    }
    Outcome { hit, steps, readings }
}

/// A step that crossed kept material is withdrawn, if at all, by the reading the *next* step
/// takes -- so the verdict on step `i` is read off step `i + 1`. A crossing whose next step hopped
/// instead of reading was never examined at all.
fn unwithdrawn_crossings(out: &Outcome) -> Vec<usize> {
    out.steps
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            s.crossed_kept && !s.withdrawn && !out.steps.get(i + 1).is_some_and(|n| n.withdrawn)
        })
        .map(|(i, _)| i)
        .collect()
}

fn print_steps(label: &str, out: &Outcome) {
    println!("--- {label}: {} steps, {} readings, {}", out.steps.len(), out.readings, match out.hit {
        Some((t, b)) => format!("HIT body {b} at t = {t:.4}"),
        None => "NO HIT".to_string(),
    });
    println!("{:>4} {:>10} {:>10} {:>11} {:>3} {:>9} {:>9}  event", "step", "from", "read at", "d", "own", "reach", "before");
    let bad = unwithdrawn_crossings(out);
    for (i, s) in out.steps.iter().enumerate() {
        let verdict = if bad.contains(&i) {
            if out.steps.get(i + 1).is_some_and(|n| n.event.starts_with("hop")) {
                "   <== CROSSED KEPT MATERIAL; the next step HOPPED, so it was never vouched"
            } else {
                "   <== CROSSED KEPT MATERIAL, never withdrawn"
            }
        } else if s.crossed_kept {
            "   (crossed; withdrawn by the next reading)"
        } else {
            ""
        };
        println!(
            "{i:>4} {:>10.3} {:>10.3} {:>11.4} {:>3} {:>9.3} {:>9.3}  {}{}",
            s.from,
            s.hopped_to,
            s.d,
            if s.owner == usize::MAX { "-".to_string() } else { s.owner.to_string() },
            s.reach,
            s.before,
            s.event,
            verdict,
        );
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .filter(|a| !a.starts_with("--"))
        .expect("usage: cull_ray_trace <document.fform>");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let part = doc.parts_ordered().first().copied().expect("a part");
    let graph = part.graph.clone();
    let (lo, hi) = match part_extent(&doc, &graph) {
        Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("unbounded"),
    };
    println!("{path}");
    println!("part extent lo {:?} hi {:?}", lo, hi);

    let pixel: (u32, u32) = arg("--pixel")
        .map(|s| {
            let (a, b) = s.split_once(',').expect("x,y");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .unwrap_or((230, 196));
    let turn: u32 = arg("--turn").map(|s| s.parse().unwrap()).unwrap_or(47);
    let lift: f64 = arg("--lift").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let away: f64 = arg("--away").map(|s| s.parse().unwrap()).unwrap_or(3.6);
    let ortho = !flag("--persp");
    let relax: f32 = arg("--relax").map(|s| s.parse().unwrap()).unwrap_or(1.4);

    // The bodies in the shader's order, each as its own CPU field.
    let bodies = doc.folded_bodies(&graph);
    println!("{} bodies:", bodies.len());
    let mut fields = Vec::new();
    let mut extents = Vec::new();
    for (i, id) in bodies.iter().enumerate() {
        let sub = subgraph_of(&graph, *id);
        let kinds: Vec<String> = doc
            .evaluation_order_of(&sub)
            .unwrap()
            .iter()
            .map(|n| {
                let node = sub.node(*n).unwrap();
                let name = format!("{node:?}");
                name.split(|c: char| c == ' ' || c == '{' || c == '(').next().unwrap_or("?").to_string()
            })
            .collect();
        let ext = match graph_extent(&doc, &graph, *id) {
            Extent::Bounded { lo, hi } => {
                let pad = ((hi - lo).norm() * CULL_PAD_FRACTION).max(CULL_PAD_FLOOR_MM);
                (
                    Point3::new(lo.x - pad, lo.y - pad, lo.z - pad),
                    Point3::new(hi.x + pad, hi.y + pad, hi.z + pad),
                )
            }
            other => panic!("body {i} extent {other:?}"),
        };
        println!(
            "  body {i}: {} ; cull extent [{:.2} {:.2} {:.2}] .. [{:.2} {:.2} {:.2}]",
            kinds.join(" <- "),
            ext.0.x, ext.0.y, ext.0.z, ext.1.x, ext.1.y, ext.1.z
        );
        let f = GraphField::compile_graph(&doc, &sub, 0.25).expect("compiles");
        fields.push(f);
        extents.push(ext);
    }

    // The program, for what it says about masks and for the cull slots it would upload.
    let program = generate_for_measurement(&doc, &graph, true, true).expect("generates");
    println!(
        "program: {} params, cull_base {:?}, masked patterns: {} ({:?}), baked: {}",
        program.params.len(),
        program.cull_base,
        program.masked.len(),
        program.masked,
        program.baked.len()
    );
    if let Some(base) = program.cull_base {
        for (i, ext) in extents.iter().enumerate() {
            let slot = |k: usize| match &program.params[base + i * 6 + k] {
                ParamRef::Literal(v) => *v,
                other => unreachable!("{other:?}"),
            };
            let same = (0..3).all(|k| (slot(k) - ext.0[k]).abs() < 1e-9)
                && (0..3).all(|k| (slot(3 + k) - ext.1[k]).abs() < 1e-9);
            if !same {
                println!("  body {i}: program's cull slots differ from this file's extent!");
            }
        }
    }

    // The camera, as the sweep builds it.
    let centre = Point3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, (lo.z + hi.z) * 0.5);
    let scene_radius = (hi - lo).norm() * 0.5;
    let a = turn as f64 * std::f64::consts::TAU / 48.0;
    let radius = 64.0 * away;
    let eye = Point3::new(
        centre.x + radius * a.cos() * lift.cos(),
        centre.y + radius * a.sin() * lift.cos(),
        centre.z + radius * lift.sin(),
    );
    let mut camera = Camera::looking_at(
        eye,
        centre,
        Vector3::z(),
        (centre, scene_radius),
        35.0,
        WIDTH as f64 / HEIGHT as f64,
    );
    camera.start_at_scene = true;
    camera.orthographic = ortho;
    if ortho {
        camera.half_height = scene_radius;
    }
    println!(
        "camera: turn {turn} lift {lift} away {away} {}; eye {:?}; epsilon {:.6}; far {:.3}; scene r {:.3}",
        if ortho { "ortho" } else { "persp" },
        eye,
        camera.epsilon,
        camera.far,
        scene_radius
    );

    // The ray through the pixel, in f32 as `primary_ray` computes it.
    let f = |v: Vector3<f64>| Vector3::new(v.x as f32, v.y as f32, v.z as f32);
    let eye32 = Point3::new(eye.x as f32, eye.y as f32, eye.z as f32);
    let half_h = camera.half_height as f32;
    let aspect = camera.aspect as f32;
    let u = (pixel.0 as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0;
    let v = 1.0 - (pixel.1 as f32 + 0.5) / HEIGHT as f32 * 2.0;
    let across = f(camera.right) * (u * half_h * aspect) + f(camera.up) * (v * half_h);
    let (origin, dir) = if ortho {
        (eye32 + across, f(camera.forward).normalize())
    } else {
        (eye32, (f(camera.forward) + across).normalize())
    };
    let epsilon = camera.epsilon as f32;
    let far = camera.far as f32;
    let budget = 512;

    // The scene sphere entry.
    let sc = Point3::new(centre.x as f32, centre.y as f32, centre.z as f32);
    let sr = scene_radius as f32;
    let oc = origin - sc;
    let b = oc.dot(&dir);
    let c = oc.dot(&oc) - sr * sr;
    let disc = b * b - c;
    assert!(disc >= 0.0, "the ray misses the scene sphere");
    let start = (-b - disc.sqrt()).max(0.0);
    println!(
        "pixel {:?} tile {:?}: origin {:?} dir {:?} start {start:.4}",
        pixel,
        (pixel.0 / 8, pixel.1 / 8),
        origin,
        dir
    );

    // The kept set for this tile.
    let tile = (pixel.0 / 8, pixel.1 / 8);
    let mut kept = vec![true; bodies.len()];
    for (i, ext) in extents.iter().enumerate() {
        let (out, gaps) = ruled_out_on_the_cpu(&camera, tile, ext.0, ext.1);
        kept[i] = !out;
        println!(
            "  body {i}: {} -- plane gaps {:.1} {:.1} {:.1} {:.1} {:.1}",
            if out { "RULED OUT" } else { "kept" },
            gaps[0], gaps[1], gaps[2], gaps[3], gaps[4]
        );
    }

    // Which bodies the ray actually passes through, by a fine sample along it: the ground truth
    // the march is measured against.
    println!("ground truth along the ray (first negative reading per body, fine sample):");
    for (i, fld) in fields.iter().enumerate() {
        let mut first = None;
        let samples = 200_000;
        for k in 0..=samples {
            let t = start + (far - start) * k as f32 / samples as f32;
            let p = origin + dir * t;
            if fld.distance(Point3::new(p.x as f64, p.y as f64, p.z as f64)) < 0.0 {
                first = Some(t);
                break;
            }
        }
        match first {
            Some(t) => println!("  body {i}: material first at t = {t:.3}"),
            None => println!("  body {i}: never"),
        }
    }

    let all = Fold { fields: &fields, kept: vec![true; bodies.len()] };
    let masked = Fold { fields: &fields, kept: kept.clone() };
    let grid_lo = centre - Vector3::repeat(scene_radius);
    let edge = scene_radius * 2.0;
    let cell = (edge / 64.0) as f32;
    let grid = SkipGrid {
        lo: Vector3::new(grid_lo.x as f32, grid_lo.y as f32, grid_lo.z as f32),
        n: 64,
        cell,
        half: cell * 0.8660254 * 1.001 + 1e-5,
        all: Fold { fields: &fields, kept: vec![true; bodies.len()] },
        cells: Default::default(),
    };
    println!("skip grid: lo {:?} cell {cell:.4} half {:.4}", grid.lo, grid.half);
    VERBOSE_HOPS.store(flag("--hops"), std::sync::atomic::Ordering::Relaxed);

    // The unmasked fold along a stretch of the ray, for seeing where material really is.
    if let Some(span) = arg("--probe") {
        let (a, b) = span.split_once(',').expect("a,b");
        let (a, b): (f32, f32) = (a.parse().unwrap(), b.parse().unwrap());
        println!("unmasked fold along the ray from {a} to {b}, every 0.5 mm, then each body's own value:");
        let mut t = a;
        while t <= b {
            let p = origin + dir * t;
            let (d, owner) = all.value(p);
            let each: Vec<String> = fields
                .iter()
                .map(|f| format!("{:>9.3}", f.distance(Point3::new(p.x as f64, p.y as f64, p.z as f64))))
                .collect();
            println!("    t {t:>9.3}  d {d:>10.4}  body {owner}  | {}  at ({:.2}, {:.2}, {:.2})", each.join(" "), p.x, p.y, p.z);
            t += 0.5;
        }
    }
    if flag("--nodes") {
        for (i, id) in bodies.iter().enumerate() {
            println!("body {i} nodes:");
            let sub = subgraph_of(&graph, *id);
            for n in doc.evaluation_order_of(&sub).unwrap() {
                println!("    {:?}", sub.node(n).unwrap());
            }
        }
    }

    let options = Options {
        relax,
        grid: !flag("--no-grid"),
        confirm: !flag("--no-confirm"),
        fixed: flag("--fixed"),
        budget,
    };
    println!(
        "\noptions: relax {} grid {} confirm {} fixed {}",
        options.relax, options.grid, options.confirm, options.fixed
    );

    let unmasked = march(&all, Some(&grid), origin, dir, start, far, epsilon, &options);
    let culled = march(&masked, Some(&grid), origin, dir, start, far, epsilon, &options);
    println!();
    print_steps("UNMASKED (all bodies)", &unmasked);
    println!();
    print_steps("MASKED (kept bodies only)", &culled);

    // The first step where the two arms stand somewhere different.
    let first = unmasked
        .steps
        .iter()
        .zip(culled.steps.iter())
        .position(|(a, b)| (a.hopped_to - b.hopped_to).abs() > 1e-4 || (a.d - b.d).abs() > 1e-4);
    println!();
    match first {
        Some(i) => println!(
            "first divergence at step {i}: unmasked read {:.4} at {:.3}, masked read {:.4} at {:.3}",
            unmasked.steps[i].d, unmasked.steps[i].hopped_to, culled.steps[i].d, culled.steps[i].hopped_to
        ),
        None => println!("the arms never diverge"),
    }
    for (label, out) in [("unmasked", &unmasked), ("masked", &culled)] {
        println!(
            "{label}: steps that crossed kept material and were never withdrawn: {:?}",
            unwithdrawn_crossings(out)
        );
    }

    if flag("--gpu") {
        gpu(&doc, &graph, &program, &camera, pixel, relax);
    }
    if flag("--sweep") {
        sweep(&doc, &program, (lo, hi), relax);
    }
}

/// Hit-or-miss, owner and depth within 2 epsilon, as `masked_matches_unmasked` compares them.
fn differs(a: &fieldform_gpu::Hit, b: &fieldform_gpu::Hit, epsilon: f64) -> bool {
    (a.depth >= 0.0) != (b.depth >= 0.0)
        || a.body != b.body
        || (a.depth >= 0.0 && (a.depth - b.depth).abs() > 2.0 * epsilon)
}

/// How square-on a hit has to be before a change to it is read as material rather than as the
/// known silhouette scatter: a ray within about eleven degrees of a surface's plane can land a
/// hair either side of it from one step sequence and not another, and that is the residual the
/// gate already accepts. A face hit at any steeper angle that one arm finds and the other does
/// not is a surface gone or a surface invented.
const GRAZING: f64 = 0.2;

/// Lost hits, gained hits, other changes, and the longest run of differing pixels down a column
/// or along a row, with where it starts. `faces` is the lost and gained hits that are not grazing.
struct Comparison {
    lost: usize,
    gained: usize,
    moved: usize,
    longest: usize,
    at: (u32, u32),
    faces_lost: usize,
    faces_gained: usize,
    /// The first few face hits lost, as pixel, depth and |normal . ray|.
    sample_lost: Vec<(u32, u32, f64, f64)>,
}

fn compare(
    reference: &[fieldform_gpu::Hit],
    arm: &[fieldform_gpu::Hit],
    epsilon: f64,
    ray_dir: impl Fn(u32, u32) -> Vector3<f64>,
) -> Comparison {
    let (mut lost, mut gained, mut moved) = (0, 0, 0);
    let (mut faces_lost, mut faces_gained) = (0, 0);
    let mut sample_lost = Vec::new();
    for i in 0..(WIDTH * HEIGHT) as usize {
        let (a, b) = (&reference[i], &arm[i]);
        let (x, y) = (i as u32 % WIDTH, i as u32 / WIDTH);
        if a.depth >= 0.0 && b.depth < 0.0 {
            lost += 1;
            let facing = a.normal.dot(&ray_dir(x, y)).abs();
            if facing > GRAZING {
                faces_lost += 1;
                if sample_lost.len() < 3 {
                    sample_lost.push((x, y, a.depth, facing));
                }
            }
        } else if a.depth < 0.0 && b.depth >= 0.0 {
            gained += 1;
            if b.normal.dot(&ray_dir(x, y)).abs() > GRAZING {
                faces_gained += 1;
            }
        } else if differs(a, b, epsilon) {
            moved += 1;
        }
    }
    let mut longest = 0usize;
    let mut at = (0u32, 0u32);
    for col in 0..WIDTH {
        let mut run = 0usize;
        for row in 0..HEIGHT {
            let i = (row * WIDTH + col) as usize;
            if differs(&reference[i], &arm[i], epsilon) {
                run += 1;
                if run > longest {
                    longest = run;
                    at = (col, row + 1 - run as u32);
                }
            } else {
                run = 0;
            }
        }
    }
    for row in 0..HEIGHT {
        let mut run = 0usize;
        for col in 0..WIDTH {
            let i = (row * WIDTH + col) as usize;
            if differs(&reference[i], &arm[i], epsilon) {
                run += 1;
                if run > longest {
                    longest = run;
                    at = (col + 1 - run as u32, row);
                }
            } else {
                run = 0;
            }
        }
    }
    Comparison { lost, gained, moved, longest, at, faces_lost, faces_gained, sample_lost }
}

/// The ray through a pixel, in `f64`, for reading a hit's facing.
fn pixel_ray(camera: &Camera, x: u32, y: u32) -> Vector3<f64> {
    let u = (x as f64 + 0.5) / WIDTH as f64 * 2.0 - 1.0;
    let v = 1.0 - (y as f64 + 0.5) / HEIGHT as f64 * 2.0;
    let across = camera.right * (u * camera.half_height * camera.aspect)
        + camera.up * (v * camera.half_height);
    if camera.orthographic {
        camera.forward.normalize()
    } else {
        (camera.forward + across).normalize()
    }
}

/// `tile_cull_sweep`'s 768 cameras, with the **plain unmasked march as the reference** for all
/// three other arms. The sweep itself holds the two folds against each other at one relaxation,
/// which cannot see a hole the relaxed march has on its own.
fn sweep(
    doc: &fieldform_scene::Document,
    program: &fieldform_gpu::ShaderProgram,
    bounds: (Point3<f64>, Point3<f64>),
    relax: f32,
) {
    let renderer = match pollster::block_on(Renderer::new(program, &[], &[], &[], 32, 1.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no GPU ({e})");
            return;
        }
    };
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    let (lo, hi) = bounds;
    let centre = Point3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, (lo.z + hi.z) * 0.5);
    let scene_radius = (hi - lo).norm() * 0.5;
    renderer.build_skip_grid(centre - Vector3::repeat(scene_radius), scene_radius * 2.0, 64, true);
    renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("fills");

    let arms = [(relax, false), (relax, true), (1.0, true)];
    struct Tally {
        frames_differing: usize,
        lost: usize,
        gained: usize,
        moved: usize,
        faces_lost: usize,
        faces_gained: usize,
        frames_losing_faces: usize,
        runs_of_4: usize,
        longest: usize,
        worst: String,
        samples: Vec<String>,
    }
    let mut tallies: Vec<Tally> = arms
        .iter()
        .map(|_| Tally {
            frames_differing: 0,
            lost: 0,
            gained: 0,
            moved: 0,
            faces_lost: 0,
            faces_gained: 0,
            frames_losing_faces: 0,
            runs_of_4: 0,
            longest: 0,
            worst: String::new(),
            samples: Vec::new(),
        })
        .collect();
    let mut frames = 0;
    for turn in 0..48 {
        for &lift in &[-0.35_f64, 0.0, 0.3, 0.75] {
            for &away in &[0.95_f64, 1.25, 2.2, 3.6] {
                let a = turn as f64 * std::f64::consts::TAU / 48.0;
                let radius = 64.0 * away;
                let eye = Point3::new(
                    centre.x + radius * a.cos() * lift.cos(),
                    centre.y + radius * a.sin() * lift.cos(),
                    centre.z + radius * lift.sin(),
                );
                let mut camera = Camera::looking_at(
                    eye,
                    centre,
                    Vector3::z(),
                    (centre, scene_radius),
                    35.0,
                    WIDTH as f64 / HEIGHT as f64,
                );
                camera.start_at_scene = true;
                camera.orthographic = turn % 2 == 1;
                if camera.orthographic {
                    camera.half_height = scene_radius;
                }
                renderer.set_relaxation(1.0);
                camera.cull = false;
                let reference =
                    renderer.render(&camera, WIDTH, HEIGHT, REFERENCE_BUDGET).expect("renders");
                frames += 1;
                for (k, (rx, cull)) in arms.iter().enumerate() {
                    renderer.set_relaxation(*rx);
                    camera.cull = *cull;
                    let arm = renderer.render(&camera, WIDTH, HEIGHT, 512).expect("renders");
                    let c = compare(&reference, &arm, camera.epsilon, |x, y| pixel_ray(&camera, x, y));
                    let tally = &mut tallies[k];
                    if c.lost + c.gained + c.moved > 0 {
                        tally.frames_differing += 1;
                    }
                    tally.lost += c.lost;
                    tally.gained += c.gained;
                    tally.moved += c.moved;
                    tally.faces_lost += c.faces_lost;
                    tally.faces_gained += c.faces_gained;
                    if c.faces_lost > 0 {
                        tally.frames_losing_faces += 1;
                    }
                    if c.longest >= 4 {
                        tally.runs_of_4 += 1;
                    }
                    if c.longest > tally.longest {
                        tally.longest = c.longest;
                        tally.worst = format!(
                            "turn {turn} lift {lift:+.2} away {away:.1} {} at {:?}: {} lost ({} faces), {} gained ({} faces), {} moved",
                            if camera.orthographic { "ortho" } else { "persp" },
                            c.at, c.lost, c.faces_lost, c.gained, c.faces_gained, c.moved
                        );
                    }
                    if tally.samples.len() < 6 {
                        for (x, y, depth, facing) in &c.sample_lost {
                            if tally.samples.len() < 6 {
                                tally.samples.push(format!(
                                    "--turn {turn} --lift {lift} --away {away} {} --pixel {x},{y} (depth {depth:.2}, |n.ray| {facing:.2})",
                                    if camera.orthographic { "" } else { "--persp" },
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("\nsweep of {frames} cameras; reference is relax 1.0, cull off (the plain unmasked march) at {REFERENCE_BUDGET} steps; arms at 512:");
    println!("  a 'face' is a hit whose normal is more than {GRAZING} off perpendicular to the ray; the rest is silhouette scatter");
    for (k, (rx, cull)) in arms.iter().enumerate() {
        let t = &tallies[k];
        println!(
            "  relax {rx} cull {cull:5}: {} frames differ; {} hits lost of which {} faces (in {} frames), {} gained of which {} faces, {} moved; {} frames with a run of 4+; longest run {} ({})",
            t.frames_differing, t.lost, t.faces_lost, t.frames_losing_faces, t.gained, t.faces_gained, t.moved, t.runs_of_4, t.longest, t.worst
        );
        for s in &t.samples {
            println!("      face lost: {s}");
        }
    }
}

/// The same frame on the card, four ways, and the pixel's answer in each -- plus, since the
/// readback is in hand, how the *unmasked* relaxed march compares with the unmasked plain one
/// over the whole frame. The sweep never took that comparison: both of its arms share a relaxation.
fn gpu(
    doc: &fieldform_scene::Document,
    graph: &Graph,
    program: &fieldform_gpu::ShaderProgram,
    camera: &Camera,
    pixel: (u32, u32),
    relax: f32,
) {
    let _ = graph;
    let renderer = match pollster::block_on(Renderer::new(program, &[], &[], &[], 32, 1.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no GPU ({e})");
            return;
        }
    };
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    let centre = camera.scene_centre;
    let r = camera.scene_radius;
    let index = (pixel.1 * WIDTH + pixel.0) as usize;
    let mut frames: Vec<(String, Vec<fieldform_gpu::Hit>)> = Vec::new();
    for rx in [relax, 1.0] {
        renderer.set_relaxation(rx);
        renderer.build_skip_grid(centre - Vector3::repeat(r), r * 2.0, 64, true);
        renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("fills");
        for cull in [false, true] {
            let mut cam = camera.clone();
            cam.cull = cull;
            let hits = renderer.render(&cam, WIDTH, HEIGHT, 512).expect("renders");
            let h = hits[index];
            println!(
                "GPU relax {rx} cull {cull}: pixel {:?} -> {}",
                pixel,
                if h.depth >= 0.0 { format!("HIT body {} at {:.4}", h.body, h.depth) } else { "MISS".to_string() }
            );
            frames.push((format!("relax {rx} cull {cull}"), hits));
        }
    }
    let compare = |x: usize, y: usize| {
        let (lx, fx) = &frames[x];
        let (ly, fy) = &frames[y];
        let c = compare(fx, fy, camera.epsilon, |px, py| pixel_ray(camera, px, py));
        println!(
            "  [{lx}] vs [{ly}]: {} hits lost ({} faces), {} gained ({} faces), {} moved/changed owner; longest run {} at {:?}",
            c.lost, c.faces_lost, c.gained, c.faces_gained, c.moved, c.longest, c.at
        );
    };
    println!("whole-frame comparisons (first named arm is the reference):");
    compare(2, 0); // plain unmasked vs relaxed unmasked
    compare(2, 3); // plain unmasked vs plain masked
    compare(2, 1); // plain unmasked vs relaxed masked
    compare(0, 1); // relaxed unmasked vs relaxed masked (the sweep's own comparison)
}
