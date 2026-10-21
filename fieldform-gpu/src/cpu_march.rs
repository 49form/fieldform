// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! `trace_primary`, re-run on the CPU: the one copy of the march loop that the GPU can be held to.
//!
//! **Why a second copy of the march exists at all.** `gpu_matches_cpu` compares every GPU hit
//! against the CPU *field* — is this point on a surface? — and a ray that tunnels through one wall
//! and stops on the next passes that test, because it is on a surface. That is how the hop that
//! skipped V8's vouch stayed invisible from the day the grid and the relaxation were combined
//! until a 2 mm shell showed it (`tile-mask-design-01941c5.md` §1.7c). What catches a march that
//! stops in the wrong place is a march that stops in the right one: the same loop, the same
//! numbers, the same grid, in `f32` on the CPU, compared hit for hit.
//!
//! **The hazard is the obvious one.** Two descriptions of one loop can drift, and a replica that
//! has drifted is an oracle that lies in whichever direction it drifted. Three things hold it:
//! `the_replica_runs_the_shader_s_numbers` pins every constant and the vouch line to the WGSL
//! text, so a change to one without the other fails a test; the gate compares the two on real
//! documents, so a drift that changes a hit shows up as a disagreement rather than a quiet
//! agreement; and this module is **not** used by anything that draws — it is an instrument, read
//! by `cpu_march_gate.rs` and `tile_cull_cpu_trace.rs`, and never by the renderer.
//!
//! Hidden from the documentation for the same reason: it is nothing a user of the crate should
//! build on.

use crate::Camera;
use nalgebra::{Point3, Vector3};

pub type V3 = Vector3<f32>;

/// A tile's side in pixels, `TILE` in `march.wgsl`.
pub(crate) const TILE: u32 = 8;
pub(crate) const STEP_SAFETY: f32 = 0.9;
pub(crate) const RELAX_REST: u32 = 3;
pub(crate) const CONFIRM: f32 = 4.0;
pub(crate) const CONFIRM_REACH: f32 = 256.0;
pub(crate) const SKIP_HOPS: u32 = 96;

/// The skip grid, filled as `skip_build` fills it.
pub struct Grid {
    pub lo: V3,
    pub n: u32,
    pub cell: f32,
    pub clear: Vec<f32>,
}

impl Grid {
    /// `skip_build`, in the march's own `f32`: the unmasked field at every cell's centre, less a
    /// hair over half the diagonal. `field` must be the whole scene's field, never a masked one --
    /// the real `skip_build` is its own entry point and never loads a mask.
    pub fn build(lo: Point3<f64>, edge: f64, n: u32, field: &dyn Fn(V3) -> f32) -> Self {
        let cell = (edge / f64::from(n)) as f32;
        let half = cell * 0.866_025_4 * 1.001 + 1e-5;
        let lo = V3::new(lo.x as f32, lo.y as f32, lo.z as f32);
        let mut clear = vec![0.0; (n * n * n) as usize];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let centre = lo + (V3::new(x as f32, y as f32, z as f32) + V3::repeat(0.5)) * cell;
                    clear[((z * n + y) * n + x) as usize] = field(centre) - half;
                }
            }
        }
        Self { lo, n, cell, clear }
    }

    /// The grid as the GPU filled it, from `Renderer::read_skip_grid` over the same cube.
    ///
    /// **This is the one the gate marches with.** A grid this module builds is the CPU field's
    /// grid, and a march that uses it is held to a grid the shader never read; a march that uses
    /// the shader's own fill is held to the shader. `build` stays for the comparison between the
    /// two, which is what sees a `skip_build` that over-reads.
    pub fn from_reading(lo: Point3<f64>, edge: f64, n: u32, clearance: Vec<f32>) -> Self {
        assert_eq!(clearance.len(), (n * n * n) as usize, "a reading of {n} cells a side");
        let cell = (edge / f64::from(n)) as f32;
        let lo = V3::new(lo.x as f32, lo.y as f32, lo.z as f32);
        Self { lo, n, cell, clear: clearance }
    }

    /// The raw clearance of the cell holding `p`, or nothing where `p` is off the grid --
    /// `cell_clearance` before its clamp to nought.
    pub(crate) fn clearance(&self, p: V3) -> Option<f32> {
        let g = (p - self.lo) / self.cell;
        let n = self.n as f32;
        if g.x < 0.0 || g.y < 0.0 || g.z < 0.0 || g.x >= n || g.y >= n || g.z >= n {
            return None;
        }
        let at = (g.x as u32, g.y as u32, g.z as u32);
        Some(self.clear[((at.2 * self.n + at.1) * self.n + at.0) as usize])
    }

    pub(crate) fn skip_distance(&self, p: V3, dir: V3, epsilon: f32) -> f32 {
        let Some(clear) = self.clearance(p) else { return 0.0 };
        if clear <= epsilon {
            return 0.0;
        }
        let g = (p - self.lo) / self.cell;
        let inside = V3::new(g.x - g.x.floor(), g.y - g.y.floor(), g.z - g.z.floor());
        let per = |i: usize| {
            let a = if dir[i] > 0.0 { 1.0 - inside[i] } else { inside[i] };
            if dir[i].abs() > 1e-12 { a / dir[i].abs() } else { 3.4e38 }
        };
        let wall = per(0).min(per(1)).min(per(2)) * self.cell;
        clear.max(wall + (self.cell * 1e-3).min(clear * 0.5))
    }

    pub(crate) fn skip_ahead(&self, origin: V3, dir: V3, start: f32, far: f32, epsilon: f32) -> f32 {
        let mut t = start;
        for _ in 0..3 * SKIP_HOPS {
            let clear = self.skip_distance(origin + dir * t, dir, epsilon);
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

pub(crate) fn confirm_distance(d: f32, came_from: f32, came_by: f32, epsilon: f32) -> f32 {
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

pub(crate) const SETTLE_REACH: f32 = 4.0;
pub(crate) const SETTLE_FREE: f32 = 2.0;
pub(crate) const SETTLE_STEPS: u32 = 4;

/// `Settle`: a confirmed candidate being settled onto its surface by the loop's own readings.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Settle {
    lo: f32,
    lo_read: f32,
    hi: f32,
    hi_read: f32,
    rate: f32,
    bracketed: bool,
    stepped_in: bool,
    left: u32,
}

/// `settle_begin`.
pub(crate) fn settle_begin(d: f32, beyond: f32, came_from: f32, came_by: f32, look: f32, epsilon: f32) -> Settle {
    let mut s = Settle {
        lo: 0.0,
        lo_read: d,
        hi: look,
        hi_read: 0.0,
        rate: 1.0,
        bracketed: beyond <= 0.0,
        stepped_in: false,
        left: 0,
    };
    if d <= 0.0 || came_by <= 0.0 {
        return s;
    }
    let rate = (came_from - d) / came_by;
    if rate <= 0.0 {
        return s;
    }
    s.rate = rate;
    let first = d / rate;
    if !s.bracketed {
        s.hi = SETTLE_REACH * look;
        if first > look {
            let fall = (d - beyond) / look;
            if fall > 0.0 {
                s.rate = fall;
            }
            s.lo = look;
            s.lo_read = beyond;
        }
    }
    if s.lo == 0.0 && first <= SETTLE_FREE * epsilon {
        return s;
    }
    s.left = SETTLE_STEPS;
    s
}

/// `settle_next_at`.
pub(crate) fn settle_next_at(s: &Settle, epsilon: f32) -> f32 {
    if s.left == 0 {
        return -1.0;
    }
    let at = if s.stepped_in {
        s.lo + (s.hi - s.lo) * s.lo_read / (s.lo_read - s.hi_read)
    } else {
        let step = s.lo_read / s.rate;
        if step <= epsilon {
            return -1.0;
        }
        let at = s.lo + step;
        if at >= s.hi {
            if s.bracketed { 0.5 * (s.lo + s.hi) } else { s.hi }
        } else {
            at
        }
    };
    if at <= s.lo || (s.bracketed && at >= s.hi) {
        return -1.0;
    }
    at
}

/// `settle_take`.
pub(crate) fn settle_take(mut s: Settle, at: f32, v: f32) -> Settle {
    s.left -= 1;
    if v <= 0.0 {
        s.hi = at;
        s.hi_read = v;
        s.bracketed = true;
        s.stepped_in = true;
    } else if v >= s.lo_read {
        s.left = 0;
    } else {
        let fall = (s.lo_read - v) / (at - s.lo);
        if fall > 0.0 {
            s.rate = fall;
        }
        s.lo = at;
        s.lo_read = v;
    }
    s
}

/// `settle_depth`.
pub(crate) fn settle_depth(s: &Settle, epsilon: f32) -> f32 {
    if s.stepped_in && -s.hi_read < epsilon {
        return s.lo + (s.hi - s.lo) * s.lo_read / (s.lo_read - s.hi_read);
    }
    s.lo
}

/// What the grid may do with a relaxed step's landing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hop {
    /// The loop as shipped since `dfdf029`: the landing cell's clearance vouches the long step, or
    /// the ray reads there instead of hopping past the question.
    Vouched,
    /// The loop as it was before: a hop from a clear landing dropped the vouch. Kept so the
    /// replica can still show the fault it was built to find.
    Unvouched,
}

pub struct Arm<'a> {
    pub field: &'a dyn Fn(V3) -> f32,
    pub grid: Option<&'a Grid>,
    pub relax: f32,
    pub hop: Hop,
    /// `SETTLES`: whether a confirmed candidate is settled onto its surface, which the generator
    /// writes from the graph's value scale (`ShaderProgram::settles`).
    pub settles: bool,
    /// The field read from the outlines rather than their bakes, when the frame finishes a ray on
    /// them (`Camera::exact_finish`, `exact_reads` in the prelude): the march reads `field` until
    /// its first confirmed candidate and this from then on. `None` is a frame that does not ask.
    pub exact: Option<&'a dyn Fn(V3) -> f32>,
}

pub struct Outcome {
    pub hit: Option<f32>,
    pub readings: u32,
    /// Hops that discarded a relaxed step's vouch, as (t before, reach, before, hopped to).
    /// Only ever non-empty under [`Hop::Unvouched`].
    pub unvouched: Vec<(f32, f32, f32, f32)>,
}

/// `trace_primary`'s loop, line for line, in `f32`. `log` prints every step.
pub fn march(arm: &Arm, origin: V3, dir: V3, start: f32, far: f32, epsilon: f32, budget: u32, log: bool) -> Outcome {
    let mut t = start;
    let mut hit = false;
    let mut reach = 0.0f32;
    let mut before = 0.0f32;
    let mut rest = 0u32;
    let mut came_from = 0.0f32;
    let mut came_by = 0.0f32;
    let mut refused_at = -1.0f32;
    let mut settling: Option<(f32, Settle)> = None;
    let mut readings = 0u32;
    let mut unvouched = Vec::new();
    // `exact_reads`: flipped at the first confirmed candidate when the arm has an exact field.
    let mut exact_reads = false;
    let read = |p: V3, readings: &mut u32, exact: bool| {
        *readings += 1;
        match (exact, arm.exact) {
            (true, Some(f)) => f(p),
            _ => (arm.field)(p),
        }
    };
    for step in 0..budget {
        if t > far && settling.is_none() {
            break;
        }
        if let (Some(grid), None) = (arm.grid, settling) {
            let mut may_hop = true;
            if reach > 0.0 && arm.hop == Hop::Vouched {
                // `cell_clearance` clamps to nought; the WGSL then tests the overlap alone.
                // `skip_ahead` refuses to hop at a clearance under `epsilon` either way, so the
                // reading is taken there in both -- the two agree, and the drift review holds it.
                let known = grid.clearance(origin + dir * t).unwrap_or(0.0).max(0.0);
                may_hop = (known + before) * STEP_SAFETY >= reach;
            }
            if may_hop {
                let hopped = grid.skip_ahead(origin, dir, t, far, epsilon);
                if hopped > t {
                    if reach > 0.0 && arm.hop == Hop::Unvouched {
                        unvouched.push((t - reach, reach, before, hopped));
                        if log {
                            println!(
                                "    step {step:3}: HOP from t={t:.3} to {hopped:.3} with a relaxed step of {reach:.3} \
                                 (from a reading of {before:.3}) still unvouched -- reach dropped"
                            );
                        }
                    } else if log {
                        println!("    step {step:3}: hop from t={t:.3} to {hopped:.3}");
                    }
                    reach = 0.0;
                    came_by = 0.0;
                }
                t = hopped;
                if t > far {
                    break;
                }
            }
        }
        let d = read(origin + dir * t, &mut readings, exact_reads);
        if let Some((base, s)) = settling {
            let s = settle_take(s, t - base, d);
            if log {
                println!("    step {step:3}: settling read {d:.6} at t={t:.4}: {s:?}");
            }
            let next = settle_next_at(&s, epsilon);
            if next < 0.0 {
                t = base + settle_depth(&s, epsilon);
                hit = true;
                settling = None;
                break;
            }
            t = base + next;
            settling = Some((base, s));
            continue;
        }
        if reach > 0.0 && (d < epsilon || (d.abs() + before) * STEP_SAFETY < reach) {
            let plain = (before * STEP_SAFETY).max(epsilon);
            if log {
                println!(
                    "    step {step:3}: t={t:.3} d={d:.3} WITHDRAWN (reach {reach:.3}, before {before:.3}); back to t={:.3}",
                    t - reach + plain
                );
            }
            t = t - reach + plain;
            reach = 0.0;
            rest = RELAX_REST;
            came_from = before;
            came_by = plain;
            continue;
        }
        if d < 0.0 && refused_at >= 0.0 {
            if arm.exact.is_some() && !exact_reads {
                if log {
                    println!("    step {step:3}: t={t:.3} d={d:.3} negative past a refusal on the bake: reading the outline from {refused_at:.3}");
                }
                exact_reads = true;
                t = refused_at;
                refused_at = -1.0;
                reach = 0.0;
                rest = RELAX_REST;
                came_by = 0.0;
                continue;
            }
            if log {
                println!("    step {step:3}: t={t:.3} d={d:.3} negative past a refusal: hit at {refused_at:.3}");
            }
            t = refused_at;
            hit = true;
            break;
        }
        if d >= epsilon {
            refused_at = -1.0;
        }
        if d < epsilon {
            let look = confirm_distance(d, came_from, came_by, epsilon);
            let beyond = read(origin + dir * (t + look), &mut readings, exact_reads);
            if beyond < 0.0 || beyond < d {
                if arm.exact.is_some() && !exact_reads {
                    if log {
                        println!("    step {step:3}: t={t:.3} d={d:.6} confirmed on the bake: reading the outline from here");
                    }
                    exact_reads = true;
                    refused_at = -1.0;
                    reach = 0.0;
                    rest = RELAX_REST;
                    came_from = d;
                    came_by = 0.0;
                    continue;
                }
                if arm.settles {
                    let s = settle_begin(d, beyond, came_from, came_by, look, epsilon);
                    let next = settle_next_at(&s, epsilon);
                    if next >= 0.0 {
                        if log {
                            println!("    step {step:3}: t={t:.3} d={d:.6} confirmed (beyond {beyond:.6} at +{look:.4}): settling from {s:?}");
                        }
                        settling = Some((t, s));
                        reach = 0.0;
                        t += next;
                        continue;
                    }
                    t += settle_depth(&s, epsilon);
                }
                if log {
                    println!("    step {step:3}: t={t:.3} d={d:.6} confirmed (beyond {beyond:.6} at +{look:.4}): HIT");
                }
                hit = true;
                break;
            }
            if log {
                println!("    step {step:3}: t={t:.3} d={d:.6} refused (beyond {beyond:.6} at +{look:.4})");
            }
            refused_at = t;
            came_from = d;
            came_by = CONFIRM * epsilon;
            t += came_by;
            reach = 0.0;
            continue;
        }
        came_from = d;
        if arm.relax > 1.0 && rest == 0 {
            reach = (d * arm.relax).max(epsilon);
            before = d;
            came_by = reach;
            if log {
                println!("    step {step:3}: t={t:.3} d={d:.3} relaxed step {reach:.3} to t={:.3}", t + reach);
            }
            t += reach;
        } else {
            rest = if rest > 0 { rest - 1 } else { 0 };
            reach = 0.0;
            came_by = (d * STEP_SAFETY).max(epsilon);
            if log {
                println!("    step {step:3}: t={t:.3} d={d:.3} plain step {came_by:.3} to t={:.3}", t + came_by);
            }
            t += came_by;
        }
    }
    // A budget spent while settling is a hit at what is known, as `trace_primary` takes it.
    if let Some((base, s)) = settling {
        t = base + settle_depth(&s, epsilon);
        hit = true;
    }
    Outcome { hit: hit.then_some(t), readings, unvouched }
}

/// `tile_cull`'s five planes, for one tile of a `frame` (width, height) and one body's extent:
/// whether the body is ruled out, and the clearance to each plane in millimetres.
pub fn ruled_out_on_the_cpu(
    camera: &Camera,
    frame: (u32, u32),
    tile: (u32, u32),
    lo: Point3<f64>,
    hi: Point3<f64>,
) -> (bool, [f64; 5]) {
    // `CULL_PIXEL_PAD` in `march.wgsl`.
    const PAD: f64 = 1.0;
    let t = f64::from(TILE);
    let (width, height) = (f64::from(frame.0), f64::from(frame.1));
    let u0 = ((tile.0 as f64) * t - PAD) / width * 2.0 - 1.0;
    let u1 = (((tile.0 + 1) as f64) * t + PAD) / width * 2.0 - 1.0;
    let v0 = 1.0 - ((tile.1 as f64) * t - PAD) / height * 2.0;
    let v1 = 1.0 - (((tile.1 + 1) as f64) * t + PAD) / height * 2.0;
    let ray = |u: f64, v: f64| {
        let across = camera.right * (u * camera.half_height * camera.aspect) + camera.up * (v * camera.half_height);
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
        // Origins first, as the shader has it: in `f64` the order is harmless, and it is kept the
        // same so that `cull_plane_precision.rs` can run this very expression in `f32` and get the
        // shader's number rather than a better one.
        let mut n = a.1.cross(&((b.0 - a.0) + b.1));
        let away = corner[(k + 2) % 4];
        if n.dot(&((away.0 - a.0) + away.1)) < 0.0 {
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

/// `primary_ray` in `f32`, jitter nought, for pixel `(x, y)` of a `frame`.
pub fn pixel_ray(camera: &Camera, frame: (u32, u32), x: u32, y: u32) -> (V3, V3) {
    let u = (x as f32 + 0.5) / frame.0 as f32 * 2.0 - 1.0;
    let v = 1.0 - (y as f32 + 0.5) / frame.1 as f32 * 2.0;
    let f = |v: Vector3<f64>| V3::new(v.x as f32, v.y as f32, v.z as f32);
    let half_h = camera.half_height as f32;
    let aspect = camera.aspect as f32;
    let mut origin = V3::new(camera.eye.x as f32, camera.eye.y as f32, camera.eye.z as f32);
    let mut dir = f(camera.forward);
    let across = f(camera.right) * (u * half_h * aspect) + f(camera.up) * (v * half_h);
    if camera.orthographic {
        origin += across;
    } else {
        dir += across;
    }
    (origin, dir.normalize())
}

/// Where the ray enters the scene's sphere, or `None` where it misses it. Only meaningful for a
/// camera with `start_at_scene`; a march from the eye starts at nought.
pub fn scene_entry(camera: &Camera, origin: V3, dir: V3) -> Option<f32> {
    let c = V3::new(camera.scene_centre.x as f32, camera.scene_centre.y as f32, camera.scene_centre.z as f32);
    let r = camera.scene_radius as f32;
    let oc = origin - c;
    let b = oc.dot(&dir);
    let cc = oc.dot(&oc) - r * r;
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    Some((-b - disc.sqrt()).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::MARCH;

    /// Every number this replica marches with is the number the shader marches with.
    ///
    /// **Two descriptions of one loop, held together by the only thing that can hold them.** A
    /// constant changed in `march.wgsl` and not here would leave a gate that compares the GPU to a
    /// march it no longer runs, and it would pass or fail on the difference rather than on the
    /// renderer. So each one is read back out of the shader's text, and the vouch line -- the one
    /// the whole arrangement turned on -- is pinned as text, since it is the line most likely to be
    /// touched next.
    #[test]
    fn the_replica_runs_the_shader_s_numbers() {
        let pinned = |name: &str, ty: &str| -> f64 {
            let needle = format!("const {name}: {ty} = ");
            let at = MARCH.find(&needle).unwrap_or_else(|| panic!("{name} is declared in march.wgsl"));
            let rest = &MARCH[at + needle.len()..];
            let end = rest.find(';').expect("a declaration ends");
            rest[..end].trim_end_matches('u').parse::<f64>().unwrap_or_else(|_| panic!("{name} parses"))
        };
        // Compared in the shader's own width: `0.9` widened from `f32` is not the `f64` 0.9.
        assert_eq!(pinned("STEP_SAFETY", "f32") as f32, STEP_SAFETY);
        assert_eq!(pinned("RELAX_REST", "u32") as u32, RELAX_REST);
        assert_eq!(pinned("CONFIRM", "f32") as f32, CONFIRM);
        assert_eq!(pinned("CONFIRM_REACH", "f32") as f32, CONFIRM_REACH);
        assert_eq!(pinned("SKIP_HOPS", "u32") as u32, SKIP_HOPS);
        assert_eq!(pinned("TILE", "u32") as u32, TILE);
        assert_eq!(pinned("CULL_PIXEL_PAD", "f32") as f32, 1.0, "the tile's pad, as ruled_out_on_the_cpu has it");
        // The mask's place in the side-table buffer, which the renderer's readback and the
        // shader's writes have to agree on and which until this line agreed by comment alone.
        assert_eq!(pinned("TILE_MASK_BASE", "u32") as u64, crate::render::TILE_MASK_BASE, "where the tile mask starts");
        assert_eq!(pinned("MASK_WORDS_MAX", "u32") as u32, crate::render::TILE_MASK_WORDS, "words a tile, at most");
        assert_eq!(pinned("CULL_TILES", "u32") as u32, crate::render::TILE_MASK_TILES, "tiles the mask can hold");

        // The half-diagonal and the nudge, as text: they are expressions, not constants.
        assert!(MARCH.contains("cell * 0.8660254 * 1.001 + 1e-5"), "skip_build's half-diagonal");
        assert!(MARCH.contains("wall + min(cell * 1e-3, clear * 0.5)"), "skip_distance's nudge");
        // The cull's plane, with the origins subtracted before the direction is added: the order
        // `cull_plane_precision.rs` measured, and the one that would quietly come back the other
        // way round in a tidy-up.
        assert!(MARCH.contains("cross(a.dir, (b.origin - a.origin) + b.dir)"), "the plane's second vector, origins first");
        assert!(MARCH.contains("dot(n, (away.origin - a.origin) + away.dir)"), "the orientation test, origins first");
        // The vouch, which is the line this replica exists to check.
        assert!(MARCH.contains("may_hop = (known + before) * STEP_SAFETY >= reach;"), "the hop vouch");
        assert!(
            MARCH.contains("if reach > 0.0 && (d < epsilon || (abs(d) + before) * STEP_SAFETY < reach)"),
            "the withdrawal"
        );
        assert!(MARCH.contains("if beyond < 0.0 || beyond < d {"), "the confirmation");
        // And the move onto the surface after it, with its reach and its check.
        assert_eq!(pinned("SETTLE_REACH", "f32") as f32, SETTLE_REACH);
        assert_eq!(pinned("SETTLE_FREE", "f32") as f32, SETTLE_FREE);
        assert_eq!(pinned("SETTLE_STEPS", "u32") as u32, SETTLE_STEPS);
        // The settling, as states of the loop and read at its one reading, which is the point.
        for (line, what) in [
            ("settle = settle_begin(d, beyond, came_from, came_by, look, epsilon);", "the confirmation starts it"),
            ("if HAS_OUTLINES && camera.jitter.z > 0.5 && !exact_reads {", "the first confirmation flips to the outline"),
            ("settle = settle_take(settle, t - base, d);", "it takes the loop's own reading"),
            ("if SETTLES && settling {", "only where the generator says so"),
            ("var may_hop = !(SETTLES && settling);", "no hop while settling"),
            ("if t > far && !(SETTLES && settling) {", "nor a stop at the far plane"),
            ("var normal = field_normal(origin + dir * t_normal);", "the normal read outside"),
            ("if s.lo == 0.0 && first <= SETTLE_FREE * epsilon {", "an exact body's candidate stands"),
            ("var s = Settle(0.0, d, look, 0.0, 1.0, beyond <= 0.0, false, 0u);", "where it starts"),
            ("let fall = (d - beyond) / look;", "the slope from the look"),
            ("let fall = (s.lo_read - v) / (at - s.lo);", "the slope between outside readings"),
            ("} else if v >= s.lo_read {", "stopping on a rise"),
            ("if v <= 0.0 {", "nought is inside"),
            ("at = select(s.hi, 0.5 * (s.lo + s.hi), s.bracketed);", "the halving before a step in"),
            ("at = s.lo + (s.hi - s.lo) * s.lo_read / (s.lo_read - s.hi_read);", "the line after one"),
            ("if s.stepped_in && -s.hi_read < epsilon {", "when the line is the answer"),
            ("if at <= s.lo || (s.bracketed && at >= s.hi) {", "when the steps stop"),
        ] {
            assert!(MARCH.contains(line), "{what}: {line}");
        }
        // Both ways a candidate is taken on the bake flip to the outline: the look-ahead's
        // confirmation and a negative reading past a refusal, which drew a dotted line of hits one
        // margin proud along a knife (the owner's specks, `divided_surface_error`'s strays).
        assert_eq!(
            MARCH.matches("if HAS_OUTLINES && camera.jitter.z > 0.5 && !exact_reads {").count(),
            2,
            "a hit past a refusal flips to the outline as a confirmed one does"
        );
        assert!(
            crate::codegen::PRELUDE.contains("let local = outline_distance_local(span, q);"),
            "the profile read honours the flip"
        );
        assert!(
            crate::codegen::PRELUDE.contains(&format!("const LIST_REACH: f32 = {:.1};", fieldform_field::LIST_REACH)),
            "the shader lists segments as far as the bake does"
        );
    }
}
