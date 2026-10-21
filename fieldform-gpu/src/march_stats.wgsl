// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
// Steps per ray, counted. Compiled in only under the `march-stats` cargo feature, appended
// after `march.wgsl` so it shares that file's camera, its `scene` and its two constants.
//
// **The primary march, duplicated rather than parameterised.** The question this answers is
// whether the march is many steps or dear steps, and a counter threaded through the loop every
// frame runs would be a cost on the thing being measured; so the loop is copied here, under
// the feature, and the frame's own `march` is untouched. It has to stay a copy: the same start
// at the scene's sphere, the same confirmation of a candidate hit, the same step. What it
// leaves out is everything after the hit -- the normal, the occlusion samples and the shadow
// march -- because those are not steps of the primary ray.
//
// One count is one pass of the loop, which is one reading of `scene`. A candidate hit that is
// confirmed or refused reads the field once more to look ahead; that reading is not counted,
// and the report says so.

@group(2) @binding(0) var<storage, read_write> stat_steps: array<u32>;

// Why the ray stopped:
//   0  missed the scene's bounding sphere, and never entered the loop
//   1  converged on a surface
//   2  passed the far plane
//   3  ran out of steps
@group(2) @binding(1) var<storage, read_write> stat_exits: array<u32>;

// How many times a pattern's sub-field was called along the ray, which over the steps is how
// wide one reading of the field is. `stat_sub_calls` is a private counter the generator
// declares in a measuring build and bumps beside every call of a `pattern_sub`; it is zeroed
// here before the march and put back around the look-ahead that confirms a hit, so what is
// counted is exactly the readings that `stat_steps` counts.
@group(2) @binding(2) var<storage, read_write> stat_calls: array<u32>;

// The field on a lattice of points, for holding one generated program to another bit for bit
// (`Renderer::field_on_grid`). `grid_spec` is the box's low corner and the samples a side, then
// its high corner.
@group(2) @binding(3) var<storage, read_write> grid_values: array<f32>;
@group(2) @binding(4) var<storage, read> grid_spec: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn grid_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(grid_spec[3]);
    if gid.x >= n || gid.y >= n || gid.z >= n {
        return;
    }
    let lo = vec3<f32>(grid_spec[0], grid_spec[1], grid_spec[2]);
    let hi = vec3<f32>(grid_spec[4], grid_spec[5], grid_spec[6]);
    let f = vec3<f32>(gid) / f32(max(n, 2u) - 1u);
    grid_values[(gid.z * n + gid.y) * n + gid.x] = field(lo + (hi - lo) * f);
}

@compute @workgroup_size(8, 8, 1)
fn march_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = camera.frame.x;
    let height = camera.frame.y;
    let row = gid.y + camera.band.x;
    if gid.x >= width || gid.y >= camera.band.y || row >= height {
        return;
    }
    let pixel = vec2<u32>(gid.x, row);
    let index = row * width + gid.x;

    let ray = primary_ray(camera, pixel);
    let origin = ray.origin;
    let dir = ray.dir;

    let far = camera.lens.z;
    let epsilon = camera.lens.w;
    let budget = camera.frame.z;

    var t = 0.0;
    if camera.scene.w > 0.0 {
        let oc = origin - camera.scene.xyz;
        let b = dot(oc, dir);
        let c = dot(oc, oc) - camera.scene.w * camera.scene.w;
        let disc = b * b - c;
        if disc < 0.0 {
            stat_steps[index] = 0u;
            stat_exits[index] = 0u;
            stat_calls[index] = 0u;
            return;
        }
        t = max(-b - sqrt(disc), 0.0);
    }

    var steps = 0u;
    var code = 3u;
    stat_sub_calls = 0u;
    // The frame's march, step for step, including the long steps and their withdrawal: a
    // withdrawn step is a reading that was taken, so it is counted.
    let relax = skip_f(7u);
    var reach = 0.0;
    var before = 0.0;
    var rest = 0u;
    var came_from = 0.0;
    var came_by = 0.0;
    var refused = false;
    for (var step = 0u; step < budget; step = step + 1u) {
        if t > far {
            code = 2u;
            break;
        }
        // As the frame's march does, and not counted: a step is a reading of the field, and
        // these are the hops that took none.
        let hopped = skip_ahead(origin, dir, t, far, epsilon);
        if hopped > t {
            reach = 0.0;
            came_by = 0.0;
        }
        t = hopped;
        if t > far {
            code = 2u;
            break;
        }
        steps = steps + 1u;
        let d = scene(origin + dir * t);
        if reach > 0.0 && (d < epsilon || (abs(d) + before) * STEP_SAFETY < reach) {
            let plain = max(before * STEP_SAFETY, epsilon);
            t = t - reach + plain;
            reach = 0.0;
            rest = RELAX_REST;
            came_from = before;
            came_by = plain;
            continue;
        }
        if d < 0.0 && refused {
            // Through what was refused: the frame's march goes back and takes it.
            code = 1u;
            break;
        }
        if d >= epsilon {
            refused = false;
        }
        if d < epsilon {
            let ahead = origin + dir * (t + confirm_distance(d, came_from, came_by, epsilon));
            let counted = stat_sub_calls;
            let beyond = max(field_here(ahead), section_distance(ahead));
            stat_sub_calls = counted;
            if beyond < 0.0 || beyond < d {
                code = 1u;
                break;
            }
            refused = true;
            came_from = d;
            came_by = CONFIRM * epsilon;
            t = t + came_by;
            reach = 0.0;
            continue;
        }
        came_from = d;
        if relax > 1.0 && rest == 0u {
            reach = max(d * relax, epsilon);
            before = d;
            came_by = reach;
            t = t + reach;
        } else {
            rest = select(0u, rest - 1u, rest > 0u);
            reach = 0.0;
            came_by = max(d * STEP_SAFETY, epsilon);
            t = t + came_by;
        }
    }
    stat_steps[index] = steps;
    stat_exits[index] = code;
    stat_calls[index] = stat_sub_calls;
}
