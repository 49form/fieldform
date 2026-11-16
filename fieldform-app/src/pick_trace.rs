// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Sphere-tracing one ray on the CPU, for picking.

use crate::*;

/// How far past a candidate hit `trace_field` looks, in hit thresholds. `CONFIRM` in
/// `march.wgsl`, and the two are the same number for the same reason.
pub(crate) const TRACE_CONFIRM: f64 = 4.0;

/// Sphere-traces a field, for picking.
///
/// Separate from the GPU marcher because this answers one ray on the CPU rather than a
/// million on a GPU, and because a click can afford to be careful where a frame cannot.
///
/// The step takes the full reported distance rather than a fraction of it. Every field in
/// fieldform undertakes never to overestimate, so a full step cannot pass through a surface,
/// and shortening it only costs iterations. What *did* cause misses was giving up too early:
/// a ray that runs out of budget looks exactly like a ray that hit nothing, and the caller
/// then silently reused the previous pivot.
///
/// A candidate hit is confirmed the same way `march.wgsl` confirms one, and for the same
/// reason: a small reading says the bound is small, not that there is a surface, and where two
/// solids share a face `min` reads nought through solid material. Picking has to agree with
/// what is drawn — the rule this file already holds to for cage against field — so if the
/// viewport goes through that sheet, a click must go through it too, or the pivot lands on a
/// surface nobody can see. `fieldform-field/examples/opposed_extrudes.rs` measures it.
pub(crate) fn trace_field(
    field: &GraphField,
    origin: Point3<f64>,
    dir: Vector3<f64>,
    far: f64,
    epsilon: f64,
    budget: u32,
) -> Option<Point3<f64>> {
    let mut t = 0.0;
    for _ in 0..budget {
        if t > far {
            return None;
        }
        let p = origin + dir * t;
        let d = field.distance(p);
        if d < epsilon {
            let beyond = field.distance(origin + dir * (t + TRACE_CONFIRM * epsilon));
            // Negative past it is a boundary; still falling is a loose bound closing on one.
            // Only a value that dips and comes back up is refused, and no surface does that.
            if beyond < 0.0 || beyond < d {
                return Some(p);
            }
            t += TRACE_CONFIRM * epsilon;
            continue;
        }
        t += d.max(epsilon);
    }
    None
}
