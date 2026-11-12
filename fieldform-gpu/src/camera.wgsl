// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
// The camera every pass shares, and the ray it casts through a pixel.
//
// One definition, prepended to both the march and the lighting pass, because the lighting pass
// has to rebuild exactly the ray the march traced in order to turn a depth back into a world
// position. A second copy of this arithmetic that drifted by half a pixel would shade every
// surface point slightly beside where it was found — and the drift would be invisible on a
// flat face and a seam on every curved one.

struct Camera {
    eye: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // x: half-height  y: aspect  z: far distance  w: hit threshold
    //
    // The half-height is tan(fov/2) under perspective, where it scales with distance along the
    // ray, and a world length under orthographic, where it does not. One field for both,
    // because the mode already says which is meant.
    lens: vec4<f32>,
    // x: width  y: height  z: step budget  w: 1 if orthographic
    frame: vec4<u32>,
    // The section planes, packed from the front: xyz is a unit normal, w its offset along that
    // normal, so the kept half of each is where `dot(p, xyz) - w <= 0`, and the kept region is
    // where that holds for every one — the intersection, so two planes make a corner cutaway.
    //
    // A zero normal means no section, and the first one ends the list. That is not a sentinel
    // standing in for a count — a plane with no direction genuinely describes no half-space, so
    // the disabled state and the degenerate state are the same state and there is nothing for
    // the two to disagree about. Eight is `MAX_SECTIONS` in render.rs.
    sections: array<vec4<f32>, 8>,
    // xyz: unit direction *toward* the key light, the one that casts the shadow.
    // w: 1 when this frame wants occlusion and shadows, 0 when it does not. A drag does not:
    // both cost field evaluations per pixel, and the interactive tier spends its budget on
    // keeping up (shading-spec.md §9).
    key_light: vec4<f32>,
    // x: how far out along the normal ambient occlusion looks, in world units.
    // y: how far toward the light the shadow march goes.
    // z: the shadow's softness — larger is sharper.
    // All three scale with the model, because a 10 mm part and a metre one occlude alike.
    // w: what to multiply a field reading by before treating it as a distance. One for a field
    // that reports true distance; larger where a driven parameter divided it. See
    // `Camera::value_scale` in render.rs — this is the only place the two shading marches may
    // use it, and the primary march must not, because there the raw value is what keeps a step
    // inside the surface.
    occlusion: vec4<f32>,
    // xy: sub-pixel offset of every ray this frame, in pixels. Zero while interacting; a
    // low-discrepancy sequence while a still image accumulates, which is what anti-aliases the
    // silhouettes of a traced surface — there are no triangle edges for multisampling to find.
    // xy: the frame's sub-pixel offset. z: 1 when a ray finishes on a profile's outline rather
    // than its bake once it has a candidate (`exact_reads` in the prelude), which the still tier
    // asks and the draft does not. w: unused.
    jitter: vec4<f32>,
    // x: the first row this dispatch covers  y: how many rows it covers.
    //
    // A frame is submitted as bands of rows rather than as one packet, because a card that
    // cannot preempt a compute dispatch is reset by Windows when one runs past about two
    // seconds — and a heavy still on a slow card does. Each band is its own submission, sized
    // by what the last one cost (`traced.rs`), so no single packet approaches the limit and
    // input between bands can cancel the rest. A buffer value rather than shader text, so the
    // band size costs no recompile.
    //
    // w: which pixels a still marches its occlusion and shadow for -- see `occlusion_pixels`.
    band: vec4<u32>,
    // xyz: the centre of a sphere every surface lies inside  w: its radius, or 0 for none.
    //
    // Where the primary ray starts: at its entry into this sphere rather than at the eye, since
    // nothing can be hit before it. Exact when the sphere encloses everything drawn, which is
    // what the caller promises by giving a radius; a scene with an unbounded body in it -- a
    // plane -- gives none, and the march starts at the eye as it always did. A ray that misses
    // the sphere altogether misses everything.
    scene: vec4<f32>,
    // The image this frame is one tile of: xy the tile's top-left pixel in it, zw its size in
    // pixels. zw nought means the frame *is* the picture, which is every frame but a saved
    // image too large for one set of targets. Every ray goes through the one projection of the
    // whole image, offset to where the tile sits -- a sub-frustum of the same camera -- so the
    // seams between tiles are not there to be seen. `image_uv` is the one place that reads it.
    tile: vec4<u32>,
    // The still's reflected march, when this frame makes one (backlog item 54): x the steps a
    // reflected ray may take, y how much of what it finds replaces the prefiltered environment
    // sample, z the traced lobe's roughness as the panel shows it, w which lobe -- 1 the base,
    // 2 the clearcoat, 0 for no reflected march this frame, which is every draft and every still
    // of a matte part. `Reflections` in render.rs is where the numbers are decided; none of them
    // is shader text, so a roughness slider never costs a recompile.
    reflect: vec4<f32>,
}

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

// The ray through a point of the image, in a [-1, 1] frame with y up.
//
// Split out of `primary_ray` so the per-tile cull can cast the rays through a tile's *corners*
// without a second copy of this arithmetic -- which is the one thing the header above forbids,
// since a copy that drifted by half a pixel would build a frustum that does not contain the rays
// the march traces, and the cull would rule out a body that is on screen.
fn ray_at(cam: Camera, u: f32, v: f32) -> Ray {
    let half_h = cam.lens.x;
    let aspect = cam.lens.y;

    // Where the two projections differ, and the only place they do: under perspective every
    // ray leaves the same point and fans out, under orthographic every ray leaves a different
    // point and they stay parallel. The march neither knows nor cares which it was handed.
    var origin = cam.eye.xyz;
    var dir = cam.forward.xyz;
    let across = cam.right.xyz * (u * half_h * aspect) + cam.up.xyz * (v * half_h);
    if cam.frame.w == 1u {
        origin = origin + across;
    } else {
        dir = dir + across;
    }
    return Ray(origin, normalize(dir));
}

// The size in pixels of the picture this frame's projection is of: the frame itself, or the
// image it is a tile of.
fn image_size(cam: Camera) -> vec2<f32> {
    if cam.tile.z > 0u {
        return vec2<f32>(f32(cam.tile.z), f32(cam.tile.w));
    }
    return vec2<f32>(f32(cam.frame.x), f32(cam.frame.y));
}

// A point of this frame, given in its own pixels, as a position in the [-1, 1] frame of the
// picture with y up -- the picture being the image this frame is a tile of, when it is one.
//
// **The one place a pixel becomes a position**, for `primary_ray` and the cull's tile corners
// alike, so a tile offset applied in one and not the other cannot happen: the header above says
// why a second copy of this arithmetic is the thing this file forbids.
fn image_uv(cam: Camera, px: f32, py: f32) -> vec2<f32> {
    let size = image_size(cam);
    var at = vec2<f32>(px, py);
    if cam.tile.z > 0u {
        at = at + vec2<f32>(f32(cam.tile.x), f32(cam.tile.y));
    }
    return vec2<f32>(at.x / size.x * 2.0 - 1.0, 1.0 - at.y / size.y * 2.0);
}

// The ray through a pixel's centre, offset by the frame's jitter.
fn primary_ray(cam: Camera, pixel: vec2<u32>) -> Ray {
    // Pixel centre, in a [-1, 1] frame with y up.
    let uv = image_uv(cam, f32(pixel.x) + 0.5 + cam.jitter.x, f32(pixel.y) + 0.5 + cam.jitter.y);
    return ray_at(cam, uv.x, uv.y);
}

// Which pixels a still frame marches its occlusion and its shadow for (loop spec P5): every one,
// or one in each two-by-two block, at a corner of the block that moves from frame to frame.
//
// Here, beside the ray, because both passes have to agree on it as exactly as they agree on the
// ray: the march measures at these pixels and the lighting pass fills in the rest from them -- all
// but those it could not fill faithfully, which the march measures as well (`fillable` in
// march.wgsl) -- and a fill that read a pixel the march had skipped would light the surface by a
// reading of nothing.
// Packed into `band.w` by `occlusion_pixels` in render.rs: bit 2 set for one in four, and bits
// 0 and 1 the corner's column and row.
struct OcclusionPixels {
    // One where every pixel is measured, two where one in each block is.
    stride: u32,
    // Which pixel of each block that is, this frame.
    offset: vec2<u32>,
}

fn occlusion_pixels(cam: Camera) -> OcclusionPixels {
    let word = cam.band.w;
    if (word & 4u) == 0u {
        return OcclusionPixels(1u, vec2<u32>(0u, 0u));
    }
    return OcclusionPixels(2u, vec2<u32>(word & 1u, (word >> 1u) & 1u));
}

// The world-space width of one pixel at distance `t` along a ray: what a detail has to be
// larger than to be seen, and so what anything sized "per pixel" is measured against.
fn pixel_footprint(cam: Camera, t: f32) -> f32 {
    // Of the whole picture's height where this frame is a tile of one: a tile's pixels are the
    // image's pixels, not a coarser frame's.
    let extent = 2.0 * cam.lens.x / image_size(cam).y;
    if cam.frame.w == 1u {
        return extent;
    }
    return extent * t;
}

// **What decides, at one pixel in four, whether a pixel is filled in or marched** (loop spec P5).
// The lighting pass fills a pixel from the measured pixels about it that lie on its own surface,
// and `occlude` marches every pixel with fewer than two of those, or with two that disagree -- the
// last pixel of a silhouette, a strut a pixel or two across, a face seen past the edge of another,
// the edge of a shadow. Here, beside the ray, because the two passes must agree on what "its own
// surface" is: a copy in each that drifted would leave a pixel the march thought could be filled
// and the fill thought could not, and it would be given a reading of something else, which is the
// fault this arrangement exists to remove.

// What `occlude` writes in place of a reading at a pixel it leaves to the lighting pass's fill.
// Negative, which neither reading ever is, so the lighting pass can tell the two apart.
const NO_READING: f32 = -1.0;

// How much of a normal's cosine two neighbours may differ by and still be one surface: about
// twenty-five degrees. A crease turns further than that, and either side of one a face can be lit
// where the other is shadowed.
const SAME_SURFACE: f32 = 0.9;

// The measured pixels either side of `i` along one axis of the image: `i` itself where it is
// measured, otherwise its neighbours inside the frame -- which are measured, because every
// other pixel is. As the first, the second and how many of the two there are.
fn bracket(i: u32, offset: u32, size: u32) -> vec3<u32> {
    if ((i + offset) & 1u) == 0u {
        return vec3<u32>(i, i, 1u);
    }
    if i == 0u {
        return vec3<u32>(1u, 1u, select(0u, 1u, size > 1u));
    }
    if i + 1u >= size {
        return vec3<u32>(i - 1u, i - 1u, 1u);
    }
    return vec3<u32>(i - 1u, i + 1u, 2u);
}

// How far the point the march found at pixel `at`, whose G-buffer word is `found`, lies from the
// plane through `p` square to `n`.
//
// From the plane rather than along the ray, because a face seen edge-on steps a long way in depth
// from one pixel to the next and is still one face, where a silhouette over the background or over
// another body is far from the plane at any angle.
fn plane_gap(cam: Camera, p: vec3<f32>, n: vec3<f32>, at: vec2<u32>, found: vec4<f32>) -> f32 {
    let ray = primary_ray(cam, at);
    return abs(dot(ray.origin + ray.dir * found.w - p, n));
}

// Whether a neighbour `gap` from a pixel's tangent plane, whose G-buffer word is `found`, is the
// same surface as that pixel, whose normal is `n` and whose width is `reach`: no further from the
// plane than a pixel's width, and turned from the normal by no more than `SAME_SURFACE`.
fn one_surface(gap: f32, reach: f32, n: vec3<f32>, found: vec4<f32>) -> bool {
    return found.w >= 0.0 && gap <= reach && dot(normalize(found.xyz), n) >= SAME_SURFACE;
}

// **The reflected ray, defined once for the march that traces it and the lighting pass that
// shades what it found** (backlog item 54; the shading notes, section 15). The march writes the
// hit's normal and distance and nothing else, so the lighting pass rebuilds the ray from the
// same inputs -- the G-buffer's normal, the primary ray, this camera -- through these functions
// and no others. Two copies that drifted by a rounding would shade the hit a little beside
// where it was found, which is the fault this file's header exists to forbid; and a glossy
// sample is drawn from a hash of the pixel and the still's frame index, both of which travel
// in the camera, so the two passes cannot be seeded differently.

// How far off the surface a reflected ray starts, in hit thresholds: outside, so that its first
// readings are positive. The loop reads a negative reading after a refusal as a hit at the
// refused point, and from an origin inside the surface that point is the origin -- a reflective
// part reflecting itself everywhere, which reads as a material bug. About four, and not grown:
// a larger offset detaches reflections from the points where two surfaces touch.
const REFLECT_OFFSET: f32 = 4.0;
// Under this the traced lobe is a mirror: one unjittered ray along the reflected direction.
// `Reflections::MIRROR_BELOW` in render.rs is the same number, and the gate there is what
// decides whether a glossy frame is traced at all.
const REFLECT_MIRROR_BELOW: f32 = 0.05;

fn reflect_origin(cam: Camera, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return p + n * (REFLECT_OFFSET * cam.lens.w);
}

// An integer hash of three words and a salt to a unit float: a multiply per word, then two
// rounds of shift-xor-multiply. Integer arithmetic throughout, so the two pipelines that call
// it agree to the bit.
fn hash_unit(x: u32, y: u32, z: u32, salt: u32) -> f32 {
    var h = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (z * 0xcb1ab31fu) ^ (salt * 0x9e3779b9u);
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h >> 8u) / 16777216.0;
}

// The direction the traced lobe reflects along at a pixel whose surface normal is `n`, seen
// from `v`: the mirror direction for a mirror -- `reflect(-v, n)`, the expression
// `shade_surface` samples the environment along, so the two agree exactly -- and for a glossy
// lobe one sample of its GGX distribution of visible normals (Heitz, "Sampling the GGX
// Distribution of Visible Normals", JCGT 2018), seeded by the pixel and the still's frame index
// so that the still's own accumulation integrates the lobe. A sample that would go into the
// surface is taken as the mirror direction rather than dropped: the lobe is narrow at these
// roughnesses, so it is rare, and a dropped sample would leave the pixel a frame short.
fn reflected_direction(cam: Camera, pixel: vec2<u32>, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let mirror = reflect(-v, n);
    let roughness = cam.reflect.z;
    if roughness < REFLECT_MIRROR_BELOW {
        return mirror;
    }
    let alpha = roughness * roughness;
    let frame = u32(max(cam.jitter.w, 0.0));
    let u1 = hash_unit(pixel.x, pixel.y, frame, 1u);
    let u2 = hash_unit(pixel.x, pixel.y, frame, 2u);
    // A tangent frame about the normal.
    var reference = vec3<f32>(0.0, 0.0, 1.0);
    if abs(n.z) > 0.9 {
        reference = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(reference, n));
    let b = cross(n, t);
    let vl = vec3<f32>(dot(v, t), dot(v, b), dot(v, n));
    // Heitz's sampler: the view stretched by the roughness, a disc sample projected onto the
    // hemisphere about it, and the result unstretched back into a normal.
    let vh = normalize(vec3<f32>(alpha * vl.x, alpha * vl.y, vl.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if lensq > 0.0 {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq);
    }
    let t2 = cross(vh, t1);
    let r = sqrt(u1);
    let phi = 2.0 * 3.14159265359 * u2;
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;
    let nh = p1 * t1 + p2 * t2 + sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * vh;
    let ne = normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(nh.z, 0.0)));
    let h = t * ne.x + b * ne.y + n * ne.z;
    let l = reflect(-v, h);
    if dot(l, n) <= 0.0 {
        return mirror;
    }
    return l;
}
