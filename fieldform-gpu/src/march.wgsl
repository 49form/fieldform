// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
// Sphere tracing, appended after the generated `field()` so it can call it.
//
// WGSL requires a function to be declared before it is used, which is why this is a separate
// piece concatenated last rather than part of the prelude. The `Camera` struct and the ray
// through a pixel come from `camera.wgsl`, which the lighting pass shares.

@group(0) @binding(4) var<uniform> camera: Camera;

// One entry per pixel: xyz is the surface normal, w is the distance along the ray, negative
// where the ray hit nothing. Normal and depth rather than a colour, so shading, silhouette
// detection and picking can all be done afterwards from the same render.
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

// Which body the surface at each pixel belongs to, or 0xffffffff where the ray hit nothing.
//
// Written from `owner`, which the union fold in `field` sets as a by-product of finding the
// nearest body. A separate buffer rather than a fifth channel on `output`: the normal is three
// floats and the depth is one, and packing an index into either would be a second meaning for
// bits that already have one.
@group(0) @binding(6) var<storage, read_write> owners: array<u32>;

// How open the surface is and how much of the key light reaches it, per pixel, as two half
// floats: x is the ambient occlusion (1 = nothing nearby), y the shadow (1 = fully lit). Both
// are 1 where the frame did not ask for them.
//
// Measured here rather than in the lighting pass because both are marches through the field,
// and the field exists in this shader alone: it is generated from the graph, and the lighting
// pass is a fixed program that reads what the march wrote.
@group(0) @binding(8) var<storage, read_write> extras: array<u32>;

// **The skip grid** (performance-plan-of-action.md §5 V5). A cube of `n³` cells about the scene,
// each holding a distance no point of that cell is nearer the surface than -- or nothing, where
// the cell might hold surface. The march asks the grid before it asks the field, and where the
// cell is known to be empty it crosses it without a reading.
//
// Why it is sound, since a wrong skip is a hole in the model. The field never overestimates the
// distance to the surface, which is the contract every source here holds to; and distance
// itself changes by at most a millimetre a millimetre. So with `c` the cell's centre and `h`
// its half diagonal, every point `p` of the cell has
//
//     distance(p) >= distance(c) - |p - c| >= field(c) - h.
//
// Nothing is assumed about the field's own rate of change, only that it reads short. A cell
// whose `field(c) - h` is positive holds no surface, so a ray may leave it by its far wall; and
// from anywhere inside it a step of that length is safe as well, so the march takes whichever
// is longer. The section plane only ever removes material from what the field describes, so a
// region empty of the field's surface is empty of the scene's.
//
// Layout, in words: `[0..3)` the cube's low corner, `[3]` cells a side, `[4]` a cell's edge,
// `[5]` whether the grid describes the field as it stands -- nought while it is stale or was
// never built -- `[6]` whether the masks are to be used; then `n³` cells of clearance from
// `[8]`, after them `n³` cells of copy mask and after those `n³` of operand mask (V6:
// `COPY_MASK` and `OperandMasks` in codegen.rs). Words rather
// than floats because the masks are bits; every float in here is read through `skip_f`. Built
// by `skip_build` below, from this same `field`.
@group(1) @binding(1) var<storage, read> skip_grid: array<u32>;

// The same buffer, as the pass that fills it sees it. Two names because one module cannot
// declare one binding with two accesses; the march's pipeline never binds this one.
@group(1) @binding(2) var<storage, read_write> skip_out: array<u32>;

fn skip_f(i: u32) -> f32 {
    return bitcast<f32>(skip_grid[i]);
}


// **The per-tile body cull** (`tile-mask-design-01941c5.md` §1.5, step 11). One bit per body per
// tile, set where that body's padded extent lies wholly outside the tile's frustum -- so the fold
// in `field` can skip it for every pixel and every step of that tile at once, where the box guard
// can only skip it at a point.
//
// **A set bit rules a body out, and nothing else does.** Every way this can go wrong -- a tile
// out of range, a mask from a different frame, a program that does not cull -- leaves the bit
// clear and the body evaluated, which is the slow answer and never the wrong one.
//
// `CULL_BODIES`, `CULL_BASE` and `MASK_WORDS` are generated in `codegen.rs`, which is how one
// fixed copy of this serves every program. `CULL_BODIES` is nought where a program does not cull,
// and every branch below is behind a test on it -- a constant, so the driver removes the whole of
// this from such a program rather than testing it per pixel.
//
// **The mask rides in the skip grid's buffer rather than in one of its own, and that is a device
// limit talking, not tidiness.** `max_storage_buffers_per_shader_stage` on this laptop's card is
// twelve and the march bound exactly twelve; a thirteenth refused the pipeline outright, with the
// whole viewport gone rather than one feature missing. Asking for thirteen was the other way, and
// it moves the device the application needs -- on a card that offers only twelve, nothing would
// draw at all. So the mask is a fourth region of a buffer that already holds a header and three
// planes of cells, read through `skip_grid` and written through `skip_out`, which is the pair of
// names that already existed for precisely this.
//
// Based at the *maximum* grid's end, not the current grid's: the offset must not move when the
// resolution does, or a mask written at one resolution would be read at another's addresses.
const TILE_MASK_BASE: u32 = 786440u;

// A tile is a workgroup of the primary march, which is 8 by 8 pixels. Not a number to tune
// independently: the mask is loaded once per invocation and the saving is shared across the
// threads that load the same word, which is exactly a workgroup.
const TILE: u32 = 8u;

// `CULL_CAP / 32` in codegen.rs -- the stride the buffer is sized for, where `MASK_WORDS` is the
// stride this program uses. The buffer is allocated for the cap so that a program with more
// bodies than the last one needs no reallocation; a program with fewer simply leaves the tail of
// each tile's run unread.
const MASK_WORDS_MAX: u32 = 8u;

// How many tiles the mask holds, which is `TILE_MASK_TILES` in render.rs -- 512 by 288 tiles of
// eight pixels, so a frame up to 4096 by 2304. A frame with more tiles than this does not cull:
// `load_tile_mask` leaves the mask dead rather than reading past the end, because **an
// out-of-range read is not zero** -- wgpu clamps the index to the last element on Metal and on
// Vulkan without robust-buffer-access 2, so a read past the end returns *some other tile's* bits,
// which would rule out a body that is on screen.
const CULL_TILES: u32 = 147456u;

fn tiles_across() -> u32 {
    return (camera.frame.x + TILE - 1u) / TILE;
}

fn tiles_down() -> u32 {
    return (camera.frame.y + TILE - 1u) / TILE;
}

// Loads this pixel's tile mask, or leaves the cull switched off for this invocation.
//
// **Called by the entry point, not by `field_here`** (review point 10): the hit's owner and its
// normal are read by `field` and `field_normal` directly inside `trace_primary`, not through
// `field_here`, so a flag set there would have left those two reading a different set of bodies
// from the march that found the hit.
fn load_tile_mask(pixel: vec2<u32>) {
    tile_live = false;
    if CULL_BODIES == 0u {
        return;
    }
    // **"Is this mask for this frame" is a uniform word, not an accident of the buffer**
    // (review point 4). The buffer outlives frames, so a submission that did not record the cull
    // -- a warm-up, a readback, a frame drawn while a new shader compiles -- would otherwise read
    // whatever the last one left, for a different camera and possibly a different width.
    if camera.band.z == 0u {
        return;
    }
    let tiles_x = tiles_across();
    let tx = pixel.x / TILE;
    let ty = pixel.y / TILE;
    if tx >= tiles_x || ty >= tiles_down() {
        return;
    }
    let tile = ty * tiles_x + tx;
    if tile >= CULL_TILES {
        return;
    }
    // **An empty mask is no mask**, and saying so is worth the branch. A live mask costs this ray
    // its over-relaxation -- the fold answers the distance to the *kept* bodies, and only a plain
    // step of that is guaranteed not to cross one -- so a tile that rules nothing out would give
    // that up for no saving at all.
    //
    // **A threshold on how much it rules out was tried and does not discriminate**: at 20 percent
    // and at 40 the table is the same to a tenth of a millisecond, because a tile either rules out
    // a great many bodies or none. What separates the scene the cull wins on from the one it loses
    // on is not the fraction culled but how far the rays travel -- fifty spread shells gain 2.2x,
    // fifty touching bodies lose 15 percent, and both cull most of their bodies in most tiles. A
    // rule that told those apart would be fitted to two scenes, so there is none.
    var ruled = 0u;
    for (var w = 0u; w < MASK_WORDS; w = w + 1u) {
        let word = skip_grid[TILE_MASK_BASE + tile * MASK_WORDS + w];
        tile_mask[w] = word;
        ruled = ruled | word;
    }
    tile_live = ruled != 0u;
}

// How far a tile is grown, in pixels, before its frustum is built.
//
// **For the jitter, and for nothing else.** A still frame offsets every ray by up to a pixel
// (`Camera.jitter`), so a ray belonging to a tile can leave the tile's own square. The gap
// between a body's box and the surface the march actually finds is covered in world space by
// `CULL_PAD_FRACTION` in codegen.rs, which is a different quantity and not interchangeable with
// this one -- a screen pad shrinks with distance and a world pad does not.
const CULL_PIXEL_PAD: f32 = 1.0;

// One invocation per tile of this band's tile rows: builds the tile's frustum and writes a bit
// per body that lies wholly outside it.
//
// **The band's tile rows, not the band's rows.** Each band is its own submission and records its
// own cull, so the work is proportional to the band; and a band starts on a tile row
// (`every_still_band_after_the_first_starts_on_a_tile_row`), so a tile is never half in one band
// and half in another with only one of them having written it. A tile straddling a band's *end*
// is written whole by that band and written again, identically, by the next -- the frustum of a
// tile does not depend on which band asked.
@compute @workgroup_size(64)
fn tile_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    if CULL_BODIES == 0u {
        return;
    }
    let tiles_x = tiles_across();
    let first_row = camera.band.x / TILE;
    let last_row = min((camera.band.x + camera.band.y + TILE - 1u) / TILE, tiles_down());
    if first_row >= last_row {
        return;
    }
    if gid.x >= (last_row - first_row) * tiles_x {
        return;
    }
    let tile = (first_row + gid.x / tiles_x) * tiles_x + gid.x % tiles_x;
    // Past capacity the cull returns rather than wrapping, and `load_tile_mask` refuses the same
    // tiles, so the frame draws unculled.
    if tile >= CULL_TILES {
        return;
    }

    // The tile's square in the image's [-1, 1] frame, a pixel wider on every side. `v` runs
    // opposite to the row, so `v0` is the *top* edge and is the larger of the two.
    let width = f32(camera.frame.x);
    let height = f32(camera.frame.y);
    let tx = f32(tile % tiles_x);
    let ty = f32(tile / tiles_x);
    let t = f32(TILE);
    let u0 = (tx * t - CULL_PIXEL_PAD) / width * 2.0 - 1.0;
    let u1 = ((tx + 1.0) * t + CULL_PIXEL_PAD) / width * 2.0 - 1.0;
    let v0 = 1.0 - (ty * t - CULL_PIXEL_PAD) / height * 2.0;
    let v1 = 1.0 - ((ty + 1.0) * t + CULL_PIXEL_PAD) / height * 2.0;

    var corner: array<Ray, 4>;
    corner[0] = ray_at(camera, u0, v0);
    corner[1] = ray_at(camera, u1, v0);
    corner[2] = ray_at(camera, u1, v1);
    corner[3] = ray_at(camera, u0, v1);
    let middle = ray_at(camera, (u0 + u1) * 0.5, (v0 + v1) * 0.5);

    // Five inward planes as (normal, point): the four sides, and one at the eye facing along the
    // tile's own axis. **The near plane is what stops a box behind the camera being accepted**;
    // there is deliberately no far plane, since a box beyond `camera.lens.z` is not drawn but is
    // also not wrong to keep, and testing it would be one more plane for nothing.
    var normal: array<vec3<f32>, 5>;
    var through: array<vec3<f32>, 5>;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let a = corner[k];
        let b = corner[(k + 1u) % 4u];
        // The plane containing both rays. Under perspective they share their origin and the two
        // directions span it; under orthographic they share their direction and the offset
        // between their origins is the second vector. One expression for both, which is also why
        // `ray_at` is shared with the march rather than copied.
        //
        // **The origins are subtracted first, and the order is the whole point.** Written as
        // `b.origin + b.dir - a.origin`, perspective adds a unit vector to the eye's coordinates
        // -- hundreds of millimetres, with an `f32` ulp of 3e-5 -- and subtracts them again, so
        // the direction comes back rounded at the eye's scale rather than its own. Two corner
        // rays of one tile are a hundredth of a radian apart, and a cross product of
        // near-parallel vectors divides that rounding by the angle: `cpu_march_gate` found the
        // planes 0.15 mm off at an 800 mm scene, which `cull_plane_precision.rs` puts at the pad
        // ten times further out. Subtracting the origins first gives exactly nought under
        // perspective and the small offset between two parallel rays under orthographic, and
        // either adds to the direction without loss.
        var n = cross(a.dir, (b.origin - a.origin) + b.dir);
        // **Oriented by the *opposite* corner, never by the tile's middle.** The first version
        // used the middle and lost a body in the middle of a tile: a tile is eight pixels, so its
        // centre ray lies very nearly *in* each of its own side planes, and `dot(n, middle)` came
        // out at the noise floor -- the sign was arithmetic dust, so some planes pointed outward
        // and culled what they should have kept. The opposite corner is a proper triple product,
        // as far off each plane as the tile is wide.
        let away = corner[(k + 2u) % 4u];
        if dot(n, (away.origin - a.origin) + away.dir) < 0.0 {
            n = -n;
        }
        normal[k] = n;
        through[k] = a.origin;
    }
    normal[4] = middle.dir;
    through[4] = middle.origin;

    var words: array<u32, MASK_WORDS_MAX>;
    for (var w = 0u; w < MASK_WORDS_MAX; w = w + 1u) {
        words[w] = 0u;
    }
    for (var b = 0u; b < CULL_BODIES; b = b + 1u) {
        let s = CULL_BASE + b * 6u;
        let lo = vec3<f32>(params[s], params[s + 1u], params[s + 2u]);
        let hi = vec3<f32>(params[s + 3u], params[s + 4u], params[s + 5u]);
        // **An empty body is an inverted box, and it is rejected before any plane test**
        // (review point 7). Four side planes and no far plane accept a far box straight down the
        // view, so an empty body written as a box a long way off would be kept by the tiles
        // looking at it.
        var out = lo.x > hi.x || lo.y > hi.y || lo.z > hi.z;
        for (var k = 0u; k < 5u && !out; k = k + 1u) {
            // The box's own corner furthest along the plane's normal. If even that one is on the
            // outside, every point of the box is, and the body cannot be in this tile. The
            // converse does not hold -- a box can fail no single plane and still miss the
            // frustum -- which is why this rules bodies out and never in.
            let far = select(lo, hi, normal[k] > vec3<f32>(0.0));
            out = dot(normal[k], far - through[k]) < 0.0;
        }
        if out {
            words[b / 32u] = words[b / 32u] | (1u << (b % 32u));
        }
    }
    for (var w = 0u; w < MASK_WORDS; w = w + 1u) {
        skip_out[TILE_MASK_BASE + tile * MASK_WORDS + w] = words[w];
    }
}

const SKIP_HEADER: u32 = 8u;
// The most cells a side the grid may have; `SKIP_GRID_MAX` in render.rs.
const SKIP_HOPS: u32 = 96u;

@compute @workgroup_size(4, 4, 4)
fn skip_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(bitcast<f32>(skip_out[3]));
    if gid.x >= n || gid.y >= n || gid.z >= n {
        return;
    }
    let cell = bitcast<f32>(skip_out[4]);
    let lo = vec3<f32>(
        bitcast<f32>(skip_out[0]),
        bitcast<f32>(skip_out[1]),
        bitcast<f32>(skip_out[2]),
    );
    let centre = lo + (vec3<f32>(gid) + vec3<f32>(0.5)) * cell;
    // Half the cell's diagonal, and a little more for the arithmetic: a bound that is a hair
    // short costs nothing, and one a hair long is the failure this must not have.
    let half = cell * 0.8660254 * 1.001 + 1e-5;
    // One reading answers both questions: how clear the cell is, and -- from the pattern, which
    // is listening for `mask_mode` -- which of its copies the cell keeps.
    mask_mode = 1u;
    mask_slack = 2.0 * half;
    mask_kept = 0u;
    mask_seen = false;
    ops_kept = 0u;
    ops_seen = 0u;
    let value = field(centre);
    mask_mode = 0u;
    let at = (gid.z * n + gid.y) * n + gid.x;
    skip_out[SKIP_HEADER + at] = bitcast<u32>(value - half);
    // The copies to pass over: those it did not keep, if it was asked at all.
    skip_out[SKIP_HEADER + n * n * n + at] = select(0u, ~mask_kept, mask_seen);
    // And the operands, each on its own account: one that was never reached -- it sits under a
    // body the box guard passed over at the centre -- has ruled nothing out.
    skip_out[SKIP_HEADER + 2u * n * n * n + at] = ~ops_kept & ops_seen;
}

// `field`, told which copies of a pattern the cell `p` is in has ruled out. Every reading the
// march takes goes through here, and none of them leaves `mask_off` set behind it: the mask
// belongs to one point, and the next caller of `field` may be asking about another.
fn field_here(p: vec3<f32>) -> f32 {
    mask_off = 0u;
    ops_off = 0u;
    if skip_grid[5] != 0u && skip_grid[6] != 0u {
        let n = skip_f(3u);
        let g = (p - vec3<f32>(skip_f(0u), skip_f(1u), skip_f(2u))) / skip_f(4u);
        if all(g >= vec3<f32>(0.0)) && all(g < vec3<f32>(n)) {
            let at = vec3<u32>(g);
            let side = u32(n);
            let cell = (at.z * side + at.y) * side + at.x;
            mask_off = skip_grid[SKIP_HEADER + side * side * side + cell];
            ops_off = skip_grid[SKIP_HEADER + 2u * side * side * side + cell];
        }
    }
    let d = field(p);
    mask_off = 0u;
    ops_off = 0u;
    return d;
}

// How far the ray at `p` may go without a reading, by the grid: nought where it cannot say.
fn skip_distance(p: vec3<f32>, dir: vec3<f32>, epsilon: f32) -> f32 {
    if skip_grid[5] == 0u {
        return 0.0;
    }
    let n = skip_f(3u);
    let cell = skip_f(4u);
    let g = (p - vec3<f32>(skip_f(0u), skip_f(1u), skip_f(2u))) / cell;
    if any(g < vec3<f32>(0.0)) || any(g >= vec3<f32>(n)) {
        return 0.0;
    }
    let at = vec3<u32>(g);
    let side = u32(n);
    let clear = skip_f(SKIP_HEADER + (at.z * side + at.y) * side + at.x);
    if clear <= epsilon {
        return 0.0;
    }
    // To the cell's far wall along the ray. A component the ray does not move in never reaches
    // its wall, which the large number says.
    let inside = g - floor(g);
    let ahead = select(inside, vec3<f32>(1.0) - inside, dir > vec3<f32>(0.0));
    let per = select(vec3<f32>(3.4e38), ahead / abs(dir), abs(dir) > vec3<f32>(1e-12));
    let wall = min(per.x, min(per.y, per.z)) * cell;
    // A nudge past the wall, so the next lookup is of the next cell; never more than half of
    // what is known to be clear, so the nudge itself cannot cross anything.
    return max(clear, wall + min(cell * 1e-3, clear * 0.5));
}

// The grid's clearance at `p`: a radius about `p` known to hold no surface, or nought where the
// grid cannot say. The same lookup as `skip_distance`, without the far wall: the wall is how far
// the ray may *advance*, which is a direction, where this is a ball, which is what a vouch needs.
fn cell_clearance(p: vec3<f32>, epsilon: f32) -> f32 {
    if skip_grid[5] == 0u {
        return 0.0;
    }
    let n = skip_f(3u);
    let cell = skip_f(4u);
    let g = (p - vec3<f32>(skip_f(0u), skip_f(1u), skip_f(2u))) / cell;
    if any(g < vec3<f32>(0.0)) || any(g >= vec3<f32>(n)) {
        return 0.0;
    }
    let at = vec3<u32>(g);
    let side = u32(n);
    let clear = skip_f(SKIP_HEADER + (at.z * side + at.y) * side + at.x);
    return max(clear, 0.0);
}

// The ray at `t`, carried across every cell the grid knows to be empty, up to the far plane.
//
// **All the hops of one step together, and then the step's one reading.** Written the obvious
// way -- a hop *or* a reading each time round the march's loop -- the grid took 62 % of the
// readings off the patterned pipe and 14 % off its frame. The card runs sixty-four rays in
// lockstep, and a pass of the loop costs what its dearest ray does: with hops and readings as
// separate passes, neighbouring rays fell out of step with each other and nearly every pass had
// somebody reading. Here every pass is some hops, which cost almost nothing, and then exactly
// one reading for every ray still marching -- so a group makes as many dear passes as its
// longest ray has readings, and no more.
fn skip_ahead(origin: vec3<f32>, dir: vec3<f32>, start: f32, far: f32, epsilon: f32) -> f32 {
    var t = start;
    // Bounded, as every loop here is: three walls a cell at the very most, and a ray that is
    // still hopping after this many takes a reading and carries on.
    for (var hop = 0u; hop < 3u * SKIP_HOPS; hop = hop + 1u) {
        let clear = skip_distance(origin + dir * t, dir, epsilon);
        if clear <= 0.0 {
            break;
        }
        t = t + clear;
        if t > far {
            break;
        }
    }
    return t;
}

// **Stepping further than the reading allows, and being made to prove it** (plan §5 V8).
//
// A reading `d` says the ball of radius `d` about the point is empty, and the plain march steps
// to just inside its edge. Most of the time it could have gone further: the next ball is about
// as big, and the two overlap by nearly a whole radius of wasted ground. So the march steps
// `relax * d` instead, and the reading it takes there has to vouch for the step. If the new
// ball and the old one still touch -- `r_new + r_old >= step` -- then every point of the step
// lies in one empty ball or the other, and nothing was stepped over. If they do not, or if
// the step landed somewhere that reads as a surface, the step is withdrawn: the ray goes back
// to where the plain march would have stood, and walks plainly for a few readings.
//
// This is over-relaxed sphere tracing (Keinert et al., 2014). What makes it the right form
// of "tighten the bounds" for this kernel is what it does *not* need: no bound on any field's
// gradient, no per-source proof. It needs the reading to be no more than the distance, which
// is the one contract every source here already keeps, and it checks every long step against
// that contract as it goes. A hit is only ever accepted from a plain step, so the surface is
// found exactly where it was found before.
//
// `relax` is `[7]` of the skip grid's header: nought, or anything not above one, is the plain
// march. Both balls are shrunk by `STEP_SAFETY` in the test, as the plain step is, for the
// sources whose reading is a hair long -- a baked cage between its samples.
//
// How many plain readings follow a withdrawn step. A ray that has just been refused is a ray
// running close to something, and asking again at once is a reading wasted most of the time.
const RELAX_REST: u32 = 3u;

// How much of the reported distance to actually travel.
//
// The field never overestimates — that is the contract every source in fieldform holds to, and
// what booleans and baked cages preserve — so a full step is safe in principle. The margin is
// for the places where the value is a *lower bound* that happens to be tight: near a concave
// crease, where a boolean's `min` is exact from one side, floating point can otherwise leave a
// ray sitting exactly on the surface without ever registering a hit.
const STEP_SAFETY: f32 = 0.9;

// How far past a candidate hit to look before believing it, in hit thresholds.
//
// **Because a small reading is not the same fact as a surface.** The march stops where the
// value drops below the threshold, and that conflates "there is a boundary here" with "the
// bound is loose here" -- which are different things, and the second happens in a place users
// build on purpose. Union two solids that share a face and `min` reads nought all along the
// join: both surfaces genuinely are right there, and it has no way to answer that neither
// bounds the union. Inside the union nothing comes of it, since a ray stops at the outer wall
// first. Subtract that union from something and the sheet lands in the open, where every ray
// crossing the cut stops on it -- read as a sliver of material left in the cut.
//
// Looking ahead separates them, because a boundary is a sign change and a loose bound is not.
// Four thresholds is enough to be past a surface the march has converged onto and not so far
// as to skip a thin one: the step before a candidate is at most one threshold, so a real
// surface is within one of where the ray stands.
//
// `fieldform-field/examples/opposed_extrudes.rs` measures the whole thing. A 40 mm square
// extruded 15 mm each way, unioned, cut out of a block: the union reads 0.000000 at the join
// where the honest answer is -15, the plain march stops there after 6 steps, and the confirmed
// one goes through and finds the floor of the cut at 27.000 mm after 23.
//
// **How far ahead, which is not always four.** Four thresholds down the ray only gets behind a
// sheet when the ray meets it within about sixty degrees of square-on: the candidate stands
// just under one threshold off, four thresholds of ray close `4 cos(a)` of them, and the far
// side only reads higher than the near once `4 cos(a) - 1 > 1`. More oblique than that and the
// value is "still falling", which is the other thing the test accepts -- so a cutter flush with
// a face left that face standing from most views and gone from a few, which the owner reported
// as something like z-fighting. `coincident_faces` counts it: a third of every frame's hits,
// at every angle from sixty degrees to one.
//
// The march already knows the angle. The value fell from `came_from` to `d` over `came_by` of
// ray, and that rate *is* the cosine of the angle to the surface's normal -- over whatever the
// field is divided by, which is wanted too, since a divided field's threshold is that many
// times further from its surface. So it looks ahead by `(d + 3 thresholds) / rate`: exactly as
// far as it takes to stand three thresholds behind whatever is there. Behind a surface that
// reads negative; behind a sheet it reads three thresholds *positive*; and square-on it comes
// to the four it always was. One reading, as before, and nothing computed that was not in hand.
//
// Capped, because along a ray that is nearly in the surface's own plane "behind it" is a long
// way off, and a look-ahead that long starts to reach past thin material. Sixty-four
// thresholds is a ray within about three and a half degrees of the plane; inside that a sheet
// can still show, as a line rather than a face.
//
// One extra sample on the pixels that hit anything, and `traced_cost` at 720 says what that is
// worth: 5.6 to 5.7 ms for an extrude, 5.5 to 5.6 for a revolve, 127.2 to 130.2 for a hundred
// patterned copies, and 6.4 to 6.8 for a gyroid -- two to six percent, dearest on the lattice,
// which is where the value is a lower bound and a candidate most often takes the second branch
// below rather than the first.
const CONFIRM: f32 = 4.0;

// The furthest the look-ahead goes, in hit thresholds. See above.
const CONFIRM_REACH: f32 = 256.0;

// How far down the ray to look from a candidate reading `d`, given the reading before it and
// how much ray lay between them -- nought when that is not known, after a hop or a skip.
fn confirm_distance(d: f32, came_from: f32, came_by: f32, epsilon: f32) -> f32 {
    let least = CONFIRM * epsilon;
    if came_by <= 0.0 {
        return least;
    }
    let rate = (came_from - d) / came_by;
    if rate <= 0.0 {
        return least;
    }
    return clamp((d + (CONFIRM - 1.0) * epsilon) / rate, least, CONFIRM_REACH * epsilon);
}

// How much further down the ray the surface is from a confirmed candidate.
//
// **A candidate is where the reading fell below a threshold, and for a divided body that is not
// where the surface is.** A body divided by S to stay a lower bound reads its distance over S, so
// it drops below the threshold S thresholds out, and the ray stopped there: the body was drawn
// that much proud of itself. Alone that is a body a fraction of a millimetre fat. Beside a body
// divided less, face to face, it is a step at the seam -- reported by the owner on two sweeps
// leaving one profile in opposite directions, one narrowing to a hundredth and divided by 82.9, the
// other by 1.18, on a shape that is level across the seam to four places
// (`divided_surface_error` measures it in millimetres, which `gpu_matches_cpu`, scoring a hit by
// the divided reading, cannot).
//
// The march already knows how far. The reading fell from `came_from` to `d` over `came_by` of
// ray, so it falls at `rate` per millimetre and reaches nought `d / rate` further on: S thresholds
// for the divided body, under one for an exact one. Every slope used is read between two points
// *outside*, because the inside of a divided body need not be divided alike: the owner's sweep
// read 25 times steeper inside than out, and a line drawn through readings either side of the
// surface was dragged back toward the candidate by it.
//
// **Newton's step, kept inside a bracket.** From the last point read outside, step its reading
// over the slope; a landing outside becomes the new outside point and gives a fresh slope between
// two outside readings; a landing inside (or on: nought is taken as inside) closes the bracket, and
// from then on the next point is the line across it. A step lands a hair one side of the surface,
// and halving instead threw away a landing 0.0003 mm in and spent every reading left climbing back.
//
// **The answer is two points.** The depth is the line across the bracket once a step has landed
// inside and read within a threshold of nought; the normal is read at the last point read outside,
// never on the surface itself, where a primitive's normal -- the direction to its nearest point --
// has none on an edge (a mirrored cone's rim, measured: NaN, 88 times in a sweep of views). Outside
// within a threshold is where every hit's normal was always read. Without a step inside, both are
// the last point outside.
//
// **Its readings are the march's own**, as states of the march loop rather than a loop of their
// own: every place the field is read is a copy of the whole composition in the compiled shader,
// and a loop of four readings beside the march cost a hundred patterned revolves 9 percent of their
// frame (108 to 118 ms) without settling one of their pixels. And **only where something is
// divided** (`SETTLES`, which the generator writes from the graph's value scale): a scene with
// nothing divided compiles without any of this and draws what it always drew.
//
// **When the look went through**, a surface is known to lie inside it, and that is the bracket.
// **When the reading was still falling at the look**, the look is capped at `CONFIRM_REACH`
// thresholds, and a divided body seen obliquely puts its surface past the cap -- on the owner's
// sweeps the reading a whole look on was 0.00002 against a threshold of 0.0014. The look point is
// then outside and nearer, so the steps start there, out to `SETTLE_REACH` looks.
//
// **Two ways it stops short.** A landing outside that reads *more* than the last ends it: the ray
// is passing the surface by -- across a gap between two bodies, or past an edge it grazed -- and
// going on would stop in mid-air (0.32 mm off, measured). So every outside point kept reads no more
// than the candidate did, which is the march's own standard for a surface. And a candidate within
// `SETTLE_FREE` thresholds of its surface by the fall stands where it is, which is every exact
// body, divided scene or not.
//
// What is not covered: a wall of some other body thinner than a step, standing wholly inside it,
// is stepped past when the landing is outside again. The fall is read toward the nearest surface,
// so it has to be another body's, and thinner than the distance the divided one is short by.
struct Settle {
    // The last point read outside, from the candidate, and what it read.
    lo: f32,
    lo_read: f32,
    // The nearest point read inside, and what it read; until one is, the furthest a step may go.
    hi: f32,
    hi_read: f32,
    rate: f32,
    bracketed: bool,
    stepped_in: bool,
    left: u32,
}

// A confirmed candidate reading `d`, whose look `look` on read `beyond`.
fn settle_begin(d: f32, beyond: f32, came_from: f32, came_by: f32, look: f32, epsilon: f32) -> Settle {
    var s = Settle(0.0, d, look, 0.0, 1.0, beyond <= 0.0, false, 0u);
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
    return s;
}

// Where to read next, from the candidate; negative when there is nothing more to read.
fn settle_next_at(s: Settle, epsilon: f32) -> f32 {
    if s.left == 0u {
        return -1.0;
    }
    var at = 0.0;
    if s.stepped_in {
        at = s.lo + (s.hi - s.lo) * s.lo_read / (s.lo_read - s.hi_read);
    } else {
        let step = s.lo_read / s.rate;
        if step <= epsilon {
            return -1.0;
        }
        at = s.lo + step;
        if at >= s.hi {
            at = select(s.hi, 0.5 * (s.lo + s.hi), s.bracketed);
        }
    }
    if at <= s.lo || (s.bracketed && at >= s.hi) {
        return -1.0;
    }
    return at;
}

// What the field read at `at`, taken in.
fn settle_take(s_in: Settle, at: f32, v: f32) -> Settle {
    var s = s_in;
    s.left = s.left - 1u;
    if v <= 0.0 {
        s.hi = at;
        s.hi_read = v;
        s.bracketed = true;
        s.stepped_in = true;
    } else if v >= s.lo_read {
        s.left = 0u;
    } else {
        let fall = (s.lo_read - v) / (at - s.lo);
        if fall > 0.0 {
            s.rate = fall;
        }
        s.lo = at;
        s.lo_read = v;
    }
    return s;
}

// The hit's depth, from the candidate.
fn settle_depth(s: Settle, epsilon: f32) -> f32 {
    if s.stepped_in && -s.hi_read < epsilon {
        return s.lo + (s.hi - s.lo) * s.lo_read / (s.lo_read - s.hi_read);
    }
    return s.lo;
}

// How far `settle` may go when the look did not reach a surface, in looks.
const SETTLE_REACH: f32 = 4.0;

// How far short of its surface, by the fall, a candidate may be and stand: an exact field's.
const SETTLE_FREE: f32 = 2.0;

// How many readings `settle` takes at most, which only a divided body's pixels pay.
const SETTLE_STEPS: u32 = 4u;

/// Signed distance to the section's half-space, or "infinitely far inside" when there is none.
///
/// Returned as a distance rather than applied as a discard so the cut face comes out **solid**:
/// intersecting the field with a half-space is `max`, and the result is a capped cross-section
/// showing material, which is what a section is for. Discarding instead would show the inside
/// of a hollow shell and read as a modelling error rather than as a cut.
fn section_distance(p: vec3<f32>) -> f32 {
    if dot(camera.section.xyz, camera.section.xyz) < 0.5 {
        return -3.4e38;
    }
    return dot(p, camera.section.xyz) - camera.section.w;
}

// The part as it is being looked at: the field intersected with the section's half-space.
//
// The same `max` a boolean difference performs, done here rather than in the graph because a
// section is a way of *looking* at the part and must not change what the part is. Everything
// that marches — the primary ray, the occlusion samples, the shadow — asks this, so the cut
// face occludes and shadows like the solid face it stands in for.
fn scene(p: vec3<f32>) -> f32 {
    return max(field_here(p), section_distance(p));
}

// How open the surface is at `p`: five samples stepping out along the normal, each asking how
// much nearer the field is than the step itself. On an unobstructed face the field reads
// exactly the step and nothing accumulates; in a crease it reads less, by more the closer the
// other wall is.
//
// The classic estimate, and an estimate is all it is — but a cheap one, five field
// evaluations against the hundred or so the primary march spent, and the thing that makes a
// join read as a join rather than two surfaces that happen to meet.
fn occlusion(p: vec3<f32>, n: vec3<f32>, radius: f32) -> f32 {
    // A divided field reads short everywhere, and this function reads short as "material is
    // close" -- so without putting the divisor back a driven morph occludes itself to black.
    // See `Camera::value_scale`.
    let scale = max(camera.occlusion.w, 1.0);
    var occluded = 0.0;
    var falloff = 1.0;
    for (var i = 1u; i <= 5u; i = i + 1u) {
        let h = radius * f32(i) / 5.0;
        let d = scene(p + n * h) * scale;
        occluded = occluded + (h - d) * falloff;
        falloff = falloff * 0.7;
    }
    // **Deliberately saturating, and kept that way by the owner on 2026-09-14.** The falloff
    // weights sum to 1.29 radii rather than one, so with a gain of 1.5 this reaches nought at
    // 52% of full occlusion and stays there -- a crevice past that contour is absolutely black
    // with a hard edge along it, which is what a ring of overlapping spheres shows.
    //
    // Dividing by what full occlusion would actually read removes the hard edge and cannot
    // saturate, and was tried: `rounded_fold_look` renders both. It reads far too weak
    // everywhere else, which is the half of the trade that matters more -- the strength this
    // has comes from exactly the part of the curve that gets clipped. Weigh the two before
    // changing it again; the saturation is the price of the contrast, not an oversight.
    return clamp(1.0 - 1.5 * occluded / radius, 0.0, 1.0);
}

// How much of the key light reaches `p`: a march toward the light keeping the smallest ratio
// of clearance to distance travelled, which is the width of the penumbra a cone of that
// angle would see. Its correctness rests on the same never-overestimate contract the
// primary march does — a field that read long here would let light through solid material.
//
// Starts a little off the surface along the normal. On the surface the field reads zero and
// every ratio would be zero, shadowing everything by itself.
fn shadow(p: vec3<f32>, n: vec3<f32>, light: vec3<f32>, reach: f32, softness: f32) -> f32 {
    let scale = max(camera.occlusion.w, 1.0);
    let start = p + n * (reach * 0.004);
    var lit = 1.0;
    var t = reach * 0.004;
    for (var i = 0u; i < 48u; i = i + 1u) {
        let d = scene(start + light * t);
        // **The ratio is scaled and the step is not.** How lit a point is depends on how far
        // away the material is, which is what the scale restores; how far it is safe to travel
        // depends on the value being a lower bound, which the scale would break -- a step taken
        // on a scaled reading walks through thin material and lights what should be shadowed.
        lit = min(lit, softness * d * scale / t);
        if lit < 0.005 || t > reach {
            break;
        }
        t = t + clamp(d, reach * 0.002, reach * 0.08);
    }
    return clamp(lit, 0.0, 1.0);
}

// What the primary ray found, for whichever entry point asked.
//
// **Two entry points share this because the draft tier must not compile what it never runs.**
// `march` reads the field at five places and `field_normal` once, so every operand of the
// composition is written into one entry point six times over, and the driver's compile is a power
// of about 2 to 2.5 in the result (`union-compile-8fa06b7.md`). Two of those six are the
// occlusion's five samples and the shadow's walk, which the draft tier never takes — it was a
// uniform branch, so the draft waited for the driver to compile both anyway. Measured at about
// **half the compile** on anything that walks: `draft_tier_cost.rs`.
//
// One function and two thin entry points rather than two copies of the march. A second copy of
// this body is the thing that has been broken twice by being "simplified" into one path, and it
// would be broken again by being duplicated into two.
//
// **Since loop spec P5 there is one entry point here, and the still's extras are `occlude`'s.**
// They went to a dispatch of their own so they could be marched for one pixel in four, which a
// lane of this march cannot do for itself without the other three lanes waiting on it; one in four
// is off, and the split pays at every pixel too, so the still traces with `march`. This stays a
// function so that whatever reads the primary ray reads the one copy of it.
struct Primary {
    // False for a thread with no pixel: past the band, or past the image edge. Such a thread has
    // written nothing and must do nothing.
    live: bool,
    // Whether it reached a surface. A miss has already written the buffers that say so.
    hit: bool,
    index: u32,
    p: vec3<f32>,
    normal: vec3<f32>,
}

/// The pixel this thread owns, marched. Writes `output` and `owners`; leaves `extras` to the
/// caller.
fn trace_primary(gid: vec3<u32>) -> Primary {
    let nothing = Primary(false, false, 0u, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0));
    let width = camera.frame.x;
    let height = camera.frame.y;
    // The dispatch covers this band's rows only; `gid.y` counts from the band's first row.
    let row = gid.y + camera.band.x;
    if gid.x >= width || gid.y >= camera.band.y || row >= height {
        return nothing;
    }
    let pixel = vec2<u32>(gid.x, row);
    let index = row * width + gid.x;

    // **The cull is a statement about the primary ray and nothing else.** It is loaded here and
    // cleared before every return below, and the occlusion and the shadow -- marched by `occlude`,
    // a dispatch of its own that never loads it -- run with the whole scene. They have to: a shadow
    // ray goes toward the light, which can be across the view, and a body outside this tile's
    // frustum can perfectly well stand between this surface and the sun. One rule, rather than an
    // exception per secondary ray.
    load_tile_mask(pixel);

    let ray = primary_ray(camera, pixel);
    let origin = ray.origin;
    let dir = ray.dir;

    let far = camera.lens.z;
    let epsilon = camera.lens.w;
    let budget = camera.frame.z;

    // Start where the ray enters the scene's sphere, when there is one -- see `Camera.scene`.
    // Every step across the empty space in front of a model was a step the field was asked
    // for nothing, and a divided field pays for those at its stretch factor; this removes
    // them exactly. `max(.., 0)` keeps an eye inside the sphere where it is.
    var t = 0.0;
    var hit = false;
    if camera.scene.w > 0.0 {
        let oc = origin - camera.scene.xyz;
        let b = dot(oc, dir);
        let c = dot(oc, oc) - camera.scene.w * camera.scene.w;
        let disc = b * b - c;
        if disc < 0.0 {
            output[index] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
            owners[index] = 0xffffffffu;
            tile_live = false;
            return Primary(true, false, index, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0));
        }
        t = max(-b - sqrt(disc), 0.0);
    }
    // See `RELAX_REST`. `reach` is the step that brought the ray here when it was a long one,
    // and nought when it was plain; `before` is the reading it was taken from.
    //
    // **A masked ray over-relaxes like any other, and the hop below is what makes that sound.**
    // The fold with a tile mask answers the distance to the *kept* bodies -- an exact distance to
    // everything this ray can hit, which is all Keinert's argument needs: a long step is vouched
    // by the next ball meeting the last, and if they meet, no kept surface lies between. For a
    // day that argument was thought to fail here, and masked rays marched plainly at a cost of
    // 14 percent on dense scenes; what actually failed was a hop that skipped the vouch, which
    // `tile-mask-design-01941c5.md` §1.7c traces to the pixel.
    let relax = skip_f(7u);
    var reach = 0.0;
    var before = 0.0;
    var rest = 0u;
    // The reading before this one and the ray between them, for `confirm_distance`.
    var came_from = 0.0;
    var came_by = 0.0;
    // Where the last candidate was refused; negative while there is none.
    var refused_at = -1.0;
    // A confirmed candidate being settled onto its surface, by readings this loop takes: see
    // `Settle`. `base` is the candidate, and `t_normal` where the hit's normal is read.
    var settling = false;
    var settle = Settle(0.0, 0.0, 0.0, 0.0, 1.0, false, false, 0u);
    var base = 0.0;
    var t_normal = -1.0;
    for (var step = 0u; step < budget; step = step + 1u) {
        if t > far && !(SETTLES && settling) {
            break;
        }
        // **A long step is vouched before anything hops from where it landed.** The hop below
        // used to run first and, finding the landing cell clear, carry the ray on with `reach`
        // zeroed -- so the step's own tail, the 0.4 d past where a plain step would have stood,
        // was never checked by anything. The grid vouches the hop; it does not vouch the step. On
        // an unmasked fold that was nearly harmless, because a reading is only ever taken in a
        // cell the grid could not clear, so `d` is at most about two half-diagonals and the tail
        // under one. A masked fold has no such bound: a cell is uncleared because of a body the
        // tile ruled out, the kept reading is the distance to something far away, and the tail
        // swallowed a whole shell -- 256 mm read where the true distance was 12, a step of 359,
        // the landing 98 mm past the wall in a clear hollow, a hop, and a miss (§1.7c, with the
        // replica that found it).
        //
        // So: the landing cell's clearance is a ball about this point known to be empty, and if
        // that ball meets the one the step left from, the step is vouched by the grid's own bound
        // standing in for a reading -- Keinert's test exactly, with a lower bound where the
        // reading would be. Then the hop may go. If the balls do not meet, nothing hops: the
        // reading is taken here and the withdrawal below decides, as it always did when the cell
        // was not clear. One grid read, taken only after a long step.
        var may_hop = !(SETTLES && settling);
        if reach > 0.0 {
            let known = cell_clearance(origin + dir * t, epsilon);
            may_hop = (known + before) * STEP_SAFETY >= reach;
        }
        if may_hop {
            // The grid first: across the cells it knows to be empty there is nothing to read.
            let hopped = skip_ahead(origin, dir, t, far, epsilon);
            if hopped > t {
                // A hop is vouched for by the grid, and the ball left behind says nothing here.
                reach = 0.0;
                came_by = 0.0;
            }
            t = hopped;
            if t > far {
                break;
            }
        }
        let d = scene(origin + dir * t);
        if SETTLES && settling {
            settle = settle_take(settle, t - base, d);
            let next = settle_next_at(settle, epsilon);
            if next < 0.0 {
                t_normal = base + settle.lo;
                t = base + settle_depth(settle, epsilon);
                hit = true;
                break;
            }
            t = base + next;
            continue;
        }
        if reach > 0.0 && (d < epsilon || (abs(d) + before) * STEP_SAFETY < reach) {
            // The long step is not vouched for. Back to where the plain one would have put
            // the ray, which is inside the old ball and so is known to be clear.
            let plain = max(before * STEP_SAFETY, epsilon);
            t = t - reach + plain;
            reach = 0.0;
            rest = RELAX_REST;
            came_from = before;
            came_by = plain;
            continue;
        }
        if d < 0.0 && refused_at >= 0.0 {
            // **The ray has gone through what it refused.** A candidate is refused when the
            // value comes back up behind it, and the ray steps past; a negative reading
            // straight after says there was material after all -- a corner clipped inside the
            // look-ahead's length. The candidate was within a threshold of that surface and
            // this point is not, so the hit is taken there.
            //
            // **Unless the still finishes on the outline and this ray has not yet** (`exact_reads`,
            // as at the confirmation below). A knife's bake is fat enough that a ray grazing its
            // tip reads small, comes back up, and reads negative a few thresholds on: a hit on the
            // bake, one margin proud, in a dotted line along an edge every neighbour finishes
            // exactly -- the owner's specks outside the silhouette. So the ray goes back to where
            // it refused and marches on from there reading the outline, which finds the knife or
            // finds nothing.
            if HAS_OUTLINES && camera.jitter.z > 0.5 && !exact_reads {
                exact_reads = true;
                t = refused_at;
                refused_at = -1.0;
                reach = 0.0;
                rest = RELAX_REST;
                came_by = 0.0;
                continue;
            }
            t = refused_at;
            hit = true;
            break;
        }
        if d >= epsilon {
            // Clear of whatever it was: the refusal stood.
            refused_at = -1.0;
        }
        if d < epsilon {
            // Confirmed rather than trusted. Negative ahead means the ray went through a
            // boundary, which is what a hit is. Still falling means the field is a loose bound
            // closing on a surface the ray has not reached -- accepted too, because that is
            // what this stopped at before confirmation existed, and refusing it would turn a
            // baked or lattice surface into a hole. Only a value that dips and comes back up
            // is rejected, and nothing that is actually a surface does that.
            let look = confirm_distance(d, came_from, came_by, epsilon);
            let ahead = origin + dir * (t + look);
            let beyond = max(field_here(ahead), section_distance(ahead));
            if beyond < 0.0 || beyond < d {
                // **Onto the outline, not its bake** (`exact_reads`): the first confirmed
                // candidate of a still's ray is where the bake has done its work, and the march
                // goes on from there reading the outline itself -- a few plain steps across the
                // margin, confirmed and settled again, exactly. A ray that was only ever near a
                // knife's fat bake finds nothing there and marches on, which is the right answer.
                // Compiled only into a program with an outline to walk (`HAS_OUTLINES`): a gyroid
                // paid 2 ms a frame for a re-confirmation it had nothing to gain from.
                if HAS_OUTLINES && camera.jitter.z > 0.5 && !exact_reads {
                    exact_reads = true;
                    refused_at = -1.0;
                    reach = 0.0;
                    rest = RELAX_REST;
                    came_from = d;
                    came_by = 0.0;
                    continue;
                }
                // Onto the surface, not where the reading first got small. See `Settle`.
                if SETTLES {
                    settle = settle_begin(d, beyond, came_from, came_by, look, epsilon);
                    let next = settle_next_at(settle, epsilon);
                    if next >= 0.0 {
                        base = t;
                        settling = true;
                        reach = 0.0;
                        t = base + next;
                        continue;
                    }
                    t_normal = t + settle.lo;
                    t = t + settle_depth(settle, epsilon);
                }
                hit = true;
                break;
            }
            // Past the whole window, or the next reading is the same small number and the
            // march stands still. Four thresholds and not the look-ahead's length: what was
            // looked across has been seen at one point only, and is walked, not jumped.
            refused_at = t;
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

    if SETTLES && settling && !hit {
        t_normal = base + settle.lo;
        t = base + settle_depth(settle, epsilon);
        hit = true;
    }
    if t_normal < 0.0 {
        t_normal = t;
    }

    if !hit {
        output[index] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        owners[index] = 0xffffffffu;
        tile_live = false;
        return Primary(true, false, index, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0));
    }

    // The ray's own origin, not the eye. They are the same point only under perspective; under
    // orthographic each ray starts somewhere different, and taking the eye here sampled the
    // normal up to half a frame away from the surface that was actually hit — the trace was
    // right and the shading was read off somewhere else.
    let p = origin + dir * t;
    // Whichever operand defined the value here owns the normal, exactly as a boolean's does.
    // On the cut face that is the plane, and taking the body's normal there would shade the
    // cap as though it were curved — the giveaway that a section was drawn rather than traced.
    let cut = section_distance(p);
    let here = field(p);
    // Read straight after the call that set it, and before anything calls `field` again and
    // overwrites it. A section's cut face still belongs to the body it cut, which is what makes
    // a highlighted body stay highlighted through a section.
    owners[index] = owner;
    // Read where the march last stood outside, which for every hit `settle` did not move is the
    // hit itself; see `Settle` for why not on the surface.
    var normal = field_normal(origin + dir * t_normal);
    if cut >= here {
        normal = camera.section.xyz;
    }
    output[index] = vec4<f32>(normal, t);
    tile_live = false;
    return Primary(true, true, index, p, normal);
}

// **The primary march, for both tiers.** No occlusion and no shadow, by the display rules — so this
// entry point does not mention them and the driver does not compile them into it. That is the whole
// of the saving, and a still keeps it: its two extra marches are `occlude`'s, dispatched after this.
//
// `extras` still written, rather than left alone: it is the previous frame's until something
// writes it, and a stale ambient term reads as dirt on the model. On a still this is the `1, 1`
// that `occlude` overwrites at every pixel that hit -- with a reading, or with `NO_READING` where
// it leaves the pixel to the lighting pass -- and what is left if the driver refused `occlude`.
@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let found = trace_primary(gid);
    if !found.live {
        return;
    }
    extras[found.index] = pack2x16float(vec2<f32>(1.0, 1.0));
}

// **The still tier's occlusion and shadow, at the pixels `occlusion_pixels` names** — every one as
// the viewport draws, or one in four with the lighting pass filling in the rest where it can (loop
// spec P5, built and off until the owner rules on it: see `Occlusion::half_resolution`).
//
// An entry point of its own rather than a branch in `march`, because the saving is not there to
// be had inside one dispatch: the lanes of a wave run in step, and three idle while the fourth
// marched its shadow would wait the whole march for it. Dispatched over the measured pixels alone,
// every lane that runs has work. It also leaves the still's primary march to `march`, a pipeline
// already compiled, and what is compiled here is the composition twice -- the occlusion's sample
// and the shadow's step -- where the still's own entry point used to carry the primary march's six
// copies besides. That half holds at every pixel too: the still's pipeline compiles in a fraction
// of the draft's, and a lattice's still, whose primary rays diverge, no longer holds the occlusion
// waiting on them.
//
// After `march`, in the same pass, so it reads the depth and the normal this band has just written.
// The point is found from them by the arithmetic `trace_primary` found it by: the same ray, the
// same distance along it. The `key_light.w` test is kept rather than assumed, as it was when this
// was the still's march: a still drawn with the light off must read as the draft does.
//
// **And at one in four, a second dispatch decides the other three pixels of each block**: filled
// in by the lighting pass, or marched here as the measured one was. The fill is an interpolation,
// so it is used only where there is something to interpolate between and it agrees: two measured
// pixels or more about the pixel, on its own surface (`one_surface`), whose readings differ by no
// more than `FILL_SPREAD`. Anything else is marched. Without the first condition a pixel with
// none of its own surface beside it -- the last pixel of a silhouette, a strut a pixel or two
// across -- was given a reading of the background or of the body behind: a strut lit as though
// open to the sky, or wearing a shadow cast on something else. Without the second, a strut that
// another strut's shadow crosses a pixel wide had the shadow smeared across its neighbours. A
// lattice of struts 0.8 mm thick at 0.45 mm a pixel finished 8.4 % of its still more than four
// levels from the still measured at every pixel with the fill unconditional, 1.4 % with the first
// condition alone, and 0.03 % with both (`overlay_smoke`). A pixel left to the fill is marked
// `NO_READING`.
//
// **What that costs is why it is off.** A wave pays for the longest march any of its lanes takes,
// and the pixels this marches are scattered along every silhouette, crease and shadow's edge, so
// on the owner's documents nearly every wave that touches the model marches one: a still frame
// came out from 10 % cheaper than every pixel's, where large smooth faces fill the picture, to 8 %
// dearer (`still_cost`).
//
// The second dispatch is this entry point again, three deep, one slice for each of a block's
// unmeasured pixels: left of the measured one, above it, and above and left. `num_workgroups.z`
// is what tells the two apart, so that the composition is compiled into one pipeline rather than
// two, and both run in the one pass the band's timestamps bracket. Left and above, because the
// test reads the measured rows either side of a pixel and the row below an unmeasured one is the
// measured row this thread's block stands on -- the last one this band's march wrote. The band's
// own last row, when it is unmeasured, waits for the next band, whose first measured row lies
// beneath it; that is the row the lighting pass already holds back (`present_rows`). An image's
// last row and column, when they are unmeasured, have nothing measured beyond them, so the second
// dispatch runs one block past the image's edge to give them a thread, which answers for nothing
// else.
@compute @workgroup_size(8, 8, 1)
fn occlude(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    if camera.key_light.w < 0.5 {
        return;
    }
    let measured = occlusion_pixels(camera);
    let stride = measured.stride;
    let width = camera.frame.x;
    let height = camera.frame.y;
    let first = camera.band.x;
    let end = min(first + camera.band.y, height);
    // The band's first row that is measured. A band need not start on one: it starts on a tile
    // row in the viewport, but anywhere for a caller that asks.
    let lead = (measured.offset.y + stride - first % stride) % stride;
    let x = gid.x * stride + measured.offset.x;
    let row = first + lead + gid.y * stride;
    let measuring = groups.z == 1u;
    var at = vec2<u32>(x, row);
    if measuring {
        if x >= width || row >= end {
            return;
        }
    } else {
        // Past the image's edge by more than the one block its last row and column need, or on
        // the next band's first row, whose block is the next band's.
        if x > width || row > end || (row == end && end < height) {
            return;
        }
        let left = (gid.z + 1u) & 1u;
        let up = (gid.z + 1u) >> 1u;
        if left > x || up > row {
            return;
        }
        at = vec2<u32>(x - left, row - up);
        if at.x >= width || at.y >= end {
            return;
        }
    }
    let index = at.y * width + at.x;
    let found = output[index];
    if found.w < 0.0 {
        return;
    }
    if !measuring && fillable(at, found, measured) {
        extras[index] = pack2x16float(vec2<f32>(NO_READING, NO_READING));
        return;
    }
    let ray = primary_ray(camera, at);
    let p = ray.origin + ray.dir * found.w;
    let open = occlusion(p, found.xyz, camera.occlusion.x);
    let lit = shadow(p, found.xyz, camera.key_light.xyz, camera.occlusion.y, camera.occlusion.z);
    extras[index] = pack2x16float(vec2<f32>(open, lit));
}

// How far apart, in either half, the measured readings about a pixel may be for the lighting pass
// to fill it with their mean rather than have it marched. A fill between two readings that differ
// by this is off by about half of it at most where the light varies smoothly between them, and a
// still is filled in at three frames of four.
const FILL_SPREAD: f32 = 1.0 / 32.0;

// Whether the lighting pass may fill in the unmeasured pixel `q`, whose G-buffer word is `found`:
// whether two or more of the measured pixels about it lie on its own surface, and their readings
// agree. The surface is judged by the test the fill makes, from the same numbers -- see
// `one_surface` -- so that what is left to the fill is what the fill will use.
fn fillable(q: vec2<u32>, found: vec4<f32>, measured: OcclusionPixels) -> bool {
    let width = camera.frame.x;
    let ray = primary_ray(camera, q);
    let p = ray.origin + ray.dir * found.w;
    let n = normalize(found.xyz);
    let reach = pixel_footprint(camera, found.w);
    let across = bracket(q.x, measured.offset.x, width);
    let down = bracket(q.y, measured.offset.y, camera.frame.y);
    var low = vec2<f32>(1.0);
    var high = vec2<f32>(0.0);
    var count = 0u;
    for (var j = 0u; j < down.z; j = j + 1u) {
        let y = select(down.x, down.y, j == 1u);
        for (var i = 0u; i < across.z; i = i + 1u) {
            let x = select(across.x, across.y, i == 1u);
            let at = y * width + x;
            let near = output[at];
            if one_surface(plane_gap(camera, p, n, vec2<u32>(x, y), near), reach, n, near) {
                let value = unpack2x16float(extras[at]);
                low = min(low, value);
                high = max(high, value);
                count = count + 1u;
            }
        }
    }
    return count >= 2u && all(high - low <= vec2<f32>(FILL_SPREAD));
}

// Shading comes from `field_normal`, generated alongside `field`, which carries each leaf's
// own normal through the composition rather than recovering one by differencing.
//
// Differencing was what made a traced surface look tessellated. It recovers whatever the
// sampling underneath made piecewise-constant -- the bake's grid cells, and beneath them the
// triangulation a cage field is built on -- and measured 4.56 degrees off the true normal at
// this viewport's settings, against 1.48 degrees for faceting already judged visible. The
// carried normal measures 0.03. See `bake_normal_accuracy`.
